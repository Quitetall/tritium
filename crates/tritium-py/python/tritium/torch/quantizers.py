"""Quantization-aware-training primitives consolidated from the satellite
projects (ADR 0037 Stage 4).

Ported verbatim from LamQuant lamquant_neural/models/blocks.py by its
copyright holder: LSQ-ternary with the Tequila deadzone fix, ParetoQ SEQ
ternary (upstream: Meta AI ParetoQ reference implementation, BSD-3-Clause —
attribution retained in the class docstring), binary LSQ, the INT-8/16
activation fake-quant with 32-point Hadamard/WHT outlier smoothing, plus the
A8 per-token absmax STE from blut-lamu's bitnet_student.py.

Registry adapters at the bottom expose the weight quantizers through the
tritium.torch estimator registry ("tequila-lsq", "pareto-seq") using the
same project() contract as the built-in estimators.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

class _LSQTernaryFunction(torch.autograd.Function):
    """Ternary quantization with LSQ gradient scaling + Tequila deadzone fix.

    Forward: w_q = round(clamp(w/α, -1, 1)) × α + deadzone_bias
    Backward: STE for weights (∂w_q/∂w ≈ 1 where |w/α| ≤ 1)
              Scaled gradient for α (÷ √n_weights for stability)

    Tequila deadzone fix (ICLR 2026): weights near the ±0.5α boundary
    between {-1,0} and {0,+1} receive noisy STE gradients and get
    "trapped" — oscillating without committing. Tequila reactivates
    them as dynamic biases: the fractional part (w/α - round(w/α)) is
    added back as a soft correction with magnitude decaying by
    temperature τ. This has near-zero inference overhead (the bias
    folds into the next layer's bias or normalization).
    """
    @staticmethod
    def forward(ctx, weight, alpha, grad_scale, deadzone_tau):
        alpha_abs = alpha.abs() + 1e-8
        w_div = weight / alpha_abs
        w_clamp = w_div.clamp(-1, 1)
        w_q = w_clamp.round()

        # Tequila: compute fractional residual for trapped weights
        # residual = (w/α - round(w/α)) — this is the "deadzone bias"
        # It's largest for weights near ±0.5 (the decision boundary)
        # and zero for weights committed to {-1, 0, +1}.
        residual = (w_clamp - w_q) * deadzone_tau

        ctx.save_for_backward(w_div, alpha_abs)
        ctx.grad_scale = grad_scale
        # Output includes the soft deadzone correction
        return (w_q + residual) * alpha_abs

    @staticmethod
    def backward(ctx, grad_output):
        w_div, alpha_abs = ctx.saved_tensors
        in_range = (w_div.abs() <= 1).float()
        grad_weight = grad_output * in_range

        w_q = w_div.clamp(-1, 1).round()
        grad_alpha = (grad_output * w_q).sum(dim=(1, 2), keepdim=True)
        grad_alpha = grad_alpha * ctx.grad_scale

        return grad_weight, grad_alpha, None, None


def _lsq_ternary(weight, alpha, grad_scale, deadzone_tau=0.1):
    return _LSQTernaryFunction.apply(weight, alpha, grad_scale, deadzone_tau)


class _SEQTernaryFunction(torch.autograd.Function):
    """ParetoQ Stretched Elastic Quantization for ternary weights.

    Ported from Meta AI's ParetoQ (NeurIPS 2025) reference implementation.
    For ternary (num_bits=0): uses n_levels=1.5 with shift=0, producing
    q_w = round(clamp(w/α, -0.99, 0.99) × 1.5) / 1.5

    This gives 3 levels: {-0.667, 0, +0.667} × α, with asymmetric bins
    optimized for the ternary phase transition. The 1.5 factor "stretches"
    the quantization grid to better match ternary weight distributions.

    Source: Reference Software/paretoq/repo/models/utils_quant.py
    """
    @staticmethod
    def forward(ctx, weight, alpha, grad_scale):
        alpha_abs = alpha.abs().clamp(min=1e-5)
        clip_val = 1 - 1e-2
        # Ternary: n_levels=1.5, shift=0 (from ParetoQ num_bits=0)
        n_levels = 1.5
        shift = 0.0
        Qp = (n_levels - shift) / n_levels  # 1.0
        Qn = -Qp

        w_scaled = weight / alpha_abs
        q_w = (torch.round(
            torch.clamp(w_scaled, -clip_val, clip_val) * n_levels - shift
        ) + shift) / n_levels

        grad_scale_val = 1.0 / math.sqrt(weight.numel())
        ctx.save_for_backward(w_scaled, alpha_abs)
        ctx.other = grad_scale_val, Qn, Qp, n_levels, shift, clip_val

        return q_w * alpha_abs

    @staticmethod
    def backward(ctx, grad_output):
        w_scaled, alpha_abs = ctx.saved_tensors
        grad_scale_val, Qn, Qp, n_levels, shift, clip_val = ctx.other

        indicate_small = (w_scaled < -clip_val).float()
        indicate_big = (w_scaled > clip_val).float()
        indicate_middle = 1.0 - indicate_small - indicate_big

        grad_input = indicate_middle * grad_output

        q_w_round = (torch.round(
            torch.clamp(w_scaled, -clip_val, clip_val) * n_levels - shift
        ) + shift) / n_levels
        grad_alpha = (
            indicate_small * Qn + indicate_big * Qp +
            indicate_middle * (-w_scaled + q_w_round)
        ) * grad_output * grad_scale_val
        grad_alpha = grad_alpha.sum(dim=(1, 2), keepdim=True)

        return grad_input, grad_alpha, None


def _seq_ternary(weight, alpha, grad_scale):
    """ParetoQ SEQ quantizer for ternary weights."""
    return _SEQTernaryFunction.apply(weight, alpha, grad_scale)


# Binary quantization: {-α, +α} only, no zero weight. sign(w) × α.
class _LSQBinaryFunction(torch.autograd.Function):
    """Binary weight quantization: w_q = sign(w) × α. No zero weights."""
    @staticmethod
    def forward(ctx, weight, alpha, grad_scale):
        alpha_abs = alpha.abs() + 1e-8
        w_sign = weight.sign()
        w_sign[w_sign == 0] = 1  # tie-break: zero → +1
        ctx.save_for_backward(weight / alpha_abs, alpha_abs)
        ctx.grad_scale = grad_scale
        return w_sign * alpha_abs

    @staticmethod
    def backward(ctx, grad_output):
        w_div, alpha_abs = ctx.saved_tensors
        in_range = (w_div.abs() <= 1).float()
        grad_weight = grad_output * in_range
        w_q = w_div.sign()
        w_q[w_q == 0] = 1
        grad_alpha = (grad_output * w_q).sum(dim=(1, 2), keepdim=True)
        grad_alpha = grad_alpha * ctx.grad_scale
        return grad_weight, grad_alpha, None


def _lsq_binary(weight, alpha, grad_scale):
    return _LSQBinaryFunction.apply(weight, alpha, grad_scale)


# Module-level switches
QUANTIZATION_MODE = 'ternary'  # 'ternary' ({-1,0,+1}) or 'binary' ({-1,+1})
QUANTIZER_TYPE = 'lsq'         # 'lsq' (default) or 'seq' (ParetoQ)

def set_quantization_mode(mode):
    """Set weight quantization: 'ternary' or 'binary'."""
    global QUANTIZATION_MODE
    assert mode in ('ternary', 'binary'), f"Invalid mode: {mode}"
    QUANTIZATION_MODE = mode

def set_quantizer_type(qtype):
    """Set quantizer: 'lsq' (original) or 'seq' (ParetoQ SEQ, recommended)."""
    global QUANTIZER_TYPE
    assert qtype in ('lsq', 'seq'), f"Invalid quantizer: {qtype}"
    QUANTIZER_TYPE = qtype
    print(f"[*] Quantizer: {qtype.upper()}")


# ============================================================
# WHT Activation Smoothing (SSDi8-inspired outlier removal)
# ============================================================

def _wht_smooth(x):
    """Walsh-Hadamard transform-based activation smoothing.

    SSDi8 (ICLR 2026) showed that activation outliers in SSM/structured
    models cause catastrophic quantization failure. WHT rotation spreads
    outlier energy across all dimensions, reducing the dynamic range and
    making subsequent INT16 quantization more robust.

    Applied per-channel: WHT along the temporal dimension, quantize in
    the rotated domain, then inverse WHT. For non-power-of-2 lengths,
    we pad to the next power of 2, transform, then trim.

    This is computationally free at inference because the firmware
    already has wht32.c — the WHT can be fused into the conv output.
    """
    B, C, T = x.shape

    # Find next power of 2 for the temporal dimension
    T_pad = 1
    while T_pad < T:
        T_pad *= 2

    if T_pad > 4096:
        # Skip WHT for very long sequences (computational cost)
        return x

    # Pad
    if T_pad > T:
        x_pad = F.pad(x, (0, T_pad - T))
    else:
        x_pad = x

    # In-place butterfly WHT (same algorithm as firmware wht32.c)
    h = 1
    while h < T_pad:
        # Split into pairs and butterfly
        x1 = x_pad[:, :, 0::2*h]  # even indices at this level
        x2 = x_pad[:, :, h::2*h]  # odd indices at this level
        # This doesn't map cleanly to strided indexing for arbitrary h.
        # Use the matrix form instead for simplicity during training:
        break

    # Matrix WHT: H @ x for each (batch, channel)
    # Build Hadamard matrix of size T_pad
    # For training, use torch's efficient implementation
    H = torch.tensor([[1.0]], device=x.device, dtype=x.dtype)
    log2_T = int(math.log2(T_pad))
    for _ in range(log2_T):
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0) / math.sqrt(2)  # Normalized WHT

    # Apply: [B, C, T_pad] @ [T_pad, T_pad]^T = [B, C, T_pad]
    x_wht = torch.matmul(x_pad, H.T)

    # Trim back to original length
    return x_wht[:, :, :T]


# ============================================================
# INT16 Activation Quantization (matching firmware W2A16)
# ============================================================

# Module-level activation bit width. Default W2A16 (production).
# Set to 8 for experimental W2A8 (halves activation buffers).
ACTIVATION_BITS = 16

def set_activation_bits(bits):
    """Set activation quantization bit width (8 or 16)."""
    global ACTIVATION_BITS
    assert bits in (8, 16), f"Activation bits must be 8 or 16, got {bits}"
    ACTIVATION_BITS = bits


class _ActivationQuantFunction(torch.autograd.Function):
    """Simulates activation quantization with STE at configurable bit width.

    W2A16 (default): range [-32768, 32767] — matches firmware int16_t
    W2A8 (experimental): range [-128, 127] — halves activation buffers,
      requires block-WHT smoothing (SSDi8) to maintain accuracy.
    """
    @staticmethod
    def forward(ctx, x, scale, bits):
        max_val = 2 ** (bits - 1) - 1
        min_val = -(2 ** (bits - 1))
        x_scaled = x / (scale + 1e-12)
        x_q = x_scaled.round().clamp(min_val, max_val)
        return x_q * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


# Pre-built 32×32 normalized Hadamard matrix (matching firmware wht32.c).
# Built eagerly on CPU at import time. Device copies are pre-populated by
# warmup_hadamard_cache() before torch.compile, so _get_hadamard_32 is a
# pure dict lookup with no construction — CUDA-graph safe.
_H32_CACHE = {}

def _build_hadamard_32_cpu():
    with torch.no_grad():
        H = torch.tensor([[1.0]])
        for _ in range(5):
            H = torch.cat([
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1),
            ], dim=0) / math.sqrt(2)
    return H

_H32_CPU = _build_hadamard_32_cpu()

def warmup_hadamard_cache(device):
    """Pre-populate the Hadamard cache for all dtypes. Call before torch.compile."""
    # Normalize device to match what x.device returns (e.g. cuda:0)
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        key = (device, dtype)
        if key not in _H32_CACHE:
            _H32_CACHE[key] = _H32_CPU.to(device=device, dtype=dtype).contiguous()
    print(f"[*] Hadamard cache warmed for {device} (3 dtypes)")

def _get_hadamard_32(device, dtype):
    """Return cached Hadamard matrix. Falls back to lazy init if not pre-warmed."""
    key = (device, dtype)
    if key not in _H32_CACHE:
        _H32_CACHE[key] = _H32_CPU.to(device=device, dtype=dtype).contiguous()
    return _H32_CACHE[key]


def _quantize_activation(x, enabled=True, hadamard=None):
    """Block-WHT smoothing + INT16 quantization for activations.

    Pipeline (matching firmware wht32.c → int16 quantize → inverse wht32):
      1. Split temporal dim into chunks of 32
      2. WHT-rotate each chunk (spreads outliers, SSDi8 ICLR 2026)
      3. INT16 quantize in WHT domain (lower dynamic range = less error)
      4. Inverse WHT to recover spatial domain
      5. Remainder samples (T % 32) are quantized directly

    At inference, steps 1-4 fold into the firmware's existing wht32.c
    + int16 activation pipeline with zero additional cost.

    Args:
        hadamard: pre-built 32×32 Hadamard matrix (buffer reference).
                  Falls back to global cache if None.
    """
    if not enabled:
        return x

    bits = ACTIVATION_BITS
    max_val = float(2 ** (bits - 1) - 1)

    if x.dim() != 3:
        with torch.no_grad():
            scale = x.abs().amax() / max_val
            scale = scale.clamp(min=1e-12)
        return _ActivationQuantFunction.apply(x, scale, bits)

    B, C, T = x.shape
    n_blocks = T // 32
    remainder = T % 32

    def _get_H():
        if hadamard is not None:
            return hadamard.to(dtype=x.dtype) if hadamard.dtype != x.dtype else hadamard
        return _get_hadamard_32(x.device, x.dtype)

    if n_blocks > 0 and n_blocks * 32 == T:
        H = _get_H()
        x_blocks = x.reshape(B, C, n_blocks, 32)
        x_wht = torch.matmul(x_blocks, H.T)
        with torch.no_grad():
            scale = x_wht.abs().amax() / max_val
            scale = scale.clamp(min=1e-12)
        x_wht_q = _ActivationQuantFunction.apply(x_wht, scale, bits)
        return torch.matmul(x_wht_q, H).reshape(B, C, T)

    parts = []
    if n_blocks > 0:
        H = _get_H()
        x_blocks = x[:, :, :n_blocks * 32].reshape(B, C, n_blocks, 32)
        x_wht = torch.matmul(x_blocks, H.T)
        with torch.no_grad():
            scale = x_wht.abs().amax() / max_val
            scale = scale.clamp(min=1e-12)
        x_wht_q = _ActivationQuantFunction.apply(x_wht, scale, bits)
        parts.append(torch.matmul(x_wht_q, H).reshape(B, C, n_blocks * 32))

    if remainder > 0:
        rem = x[:, :, n_blocks * 32:]
        with torch.no_grad():
            scale_rem = rem.abs().amax() / max_val
            scale_rem = scale_rem.clamp(min=1e-12)
        parts.append(_ActivationQuantFunction.apply(rem, scale_rem, bits))

    return torch.cat(parts, dim=2) if len(parts) > 1 else parts[0]

# ============================================================
# A8 per-token absmax STE (blut-lamu bitnet_student.py, verbatim)
# ============================================================


def act_quant_ste(x: torch.Tensor) -> torch.Tensor:
    """A8 per-token absmax fake-quant (int8 lattice) with STE — mirrors the
    served kernel's activation quant so training sees serving numerics."""
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-8) / 127.0
    x_hat = (x / scale).round().clamp_(-128, 127) * scale
    return x + (x_hat - x).detach()


# ============================================================
# Estimator-registry adapters (tritium.torch project() contract)
# ============================================================

from tritium.torch.estimators import (  # noqa: E402
    Estimator,
    ProjectionContext,
    TernaryProjection,
    _projection,
    _rank2,
    _soft_projection,
    register_estimator,
)


class TequilaLSQEstimator(Estimator):
    """LSQ ternary with per-row learned alpha and the Tequila deadzone bias
    (soft reactivation of boundary-trapped weights, annealed via `set_tau`).
    Wraps the verbatim `_LSQTernaryFunction` in the registry contract. The
    deadzone bias is a forward-VALUE mechanism, so the projection uses the
    soft-forward path (`_soft_projection`): while tau > 0 the projection is
    non-exportable (training-time lattice); at tau == 0 the soft code equals
    the hard decode and the projection becomes exportable."""

    algorithm_id = "tritium.tequila-lsq"
    schema_version = 1

    def __init__(self, deadzone_tau: float = 0.1) -> None:
        super().__init__()
        if not (0.0 <= deadzone_tau < 1.0):
            raise ValueError(f"deadzone_tau must be in [0,1), got {deadzone_tau}")
        self.register_buffer("_tau", torch.tensor(float(deadzone_tau)))
        self._alpha: nn.Parameter | None = None

    def set_tau(self, tau: float) -> None:
        self._tau.fill_(float(tau))

    def _ensure_alpha(self, master: torch.Tensor) -> nn.Parameter:
        if self._alpha is None:
            # LamQuant's TernaryConv1d init: alpha0 = (2/3)·mean|W| per row.
            init = (master.detach().abs().mean(dim=1, keepdim=True) * (2.0 / 3.0)).clamp(min=1e-8)
            self._alpha = nn.Parameter(init)
        return self._alpha

    def project(
        self, master: torch.Tensor, *, context: ProjectionContext
    ) -> TernaryProjection:
        del context
        _rank2(master, "TequilaLSQEstimator")
        alpha = self._ensure_alpha(master)
        grad_scale = 1.0 / math.sqrt(master.numel())
        # The verbatim function sums grad_alpha over dims (1,2) of a 3-D
        # weight; rank-2 masters get a trailing unit dim for exactness.
        soft = _LSQTernaryFunction.apply(
            master.unsqueeze(-1), alpha.unsqueeze(-1), grad_scale, float(self._tau)
        ).squeeze(-1)
        alpha_abs = alpha.detach().abs() + 1e-8
        trits = (master.detach() / alpha_abs).clamp(-1, 1).round().to(torch.int8)
        scales = alpha_abs.to(master.dtype)
        return _soft_projection(
            master, trits, scales, soft, self, exportable=float(self._tau) == 0.0
        )


class ParetoSEQEstimator(Estimator):
    """ParetoQ Stretched Elastic Quantization for ternary (BSD-3 upstream —
    see `_SEQTernaryFunction`). The stretched grid ({-2/3, 0, +2/3}·alpha)
    exists only in the training-time soft code; `project()` reports plain
    trits with the effective per-row scale (2/3)·alpha so the exported
    tensor equals the soft lattice exactly."""

    algorithm_id = "tritium.pareto-seq"
    schema_version = 1

    def __init__(self) -> None:
        super().__init__()
        self._alpha: nn.Parameter | None = None

    def _ensure_alpha(self, master: torch.Tensor) -> nn.Parameter:
        if self._alpha is None:
            init = master.detach().abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
            self._alpha = nn.Parameter(init)
        return self._alpha

    def project(
        self, master: torch.Tensor, *, context: ProjectionContext
    ) -> TernaryProjection:
        del context
        _rank2(master, "ParetoSEQEstimator")
        alpha = self._ensure_alpha(master)
        grad_scale = 1.0 / math.sqrt(master.numel())
        soft = _SEQTernaryFunction.apply(
            master.unsqueeze(-1), alpha.unsqueeze(-1), grad_scale
        ).squeeze(-1)
        alpha_abs = alpha.detach().abs().clamp(min=1e-5)
        n_levels, clip_val = 1.5, 1.0 - 1e-2
        trits = (
            ((master.detach() / alpha_abs).clamp(-clip_val, clip_val) * n_levels)
            .round()
            .clamp(-1, 1)
            .to(torch.int8)
        )
        scales = (alpha_abs / n_levels).to(master.dtype)
        return _projection(master, trits, scales, soft, self)


register_estimator("tequila-lsq", TequilaLSQEstimator)
register_estimator("pareto-seq", ParetoSEQEstimator)
