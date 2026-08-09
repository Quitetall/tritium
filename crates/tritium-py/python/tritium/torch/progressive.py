# Ported verbatim from LamQuant lamquant/student/progressive_quant.py by its
# copyright holder (ADR 0037 Stage 4).
"""ai_models/student/progressive_quant.py — Progressive quantization schedule.

Trains the encoder through decreasing bit widths:
  INT8 (256 levels) → INT4 (16 levels) → INT2 (4 levels) → Ternary (3 levels)

Each transition is a small quantization shock that the optimizer can handle.
Features learned at N bits are inherently compatible with N-1 bits because
they already live in discrete space. This avoids the catastrophic FP32→ternary
jump that destroys pretrained representations.

The key insight: discrete→discrete transitions preserve structure much better
than continuous→discrete because the network never learns features that
require continuous precision.

Usage:
    schedule = ProgressiveQuantSchedule(total_epochs=400)
    for epoch in range(400):
        bits = schedule.get_bits(epoch)
        # bits follows: 8 → 4 → 2 → ternary over the training run
        for batch in dataloader:
            w_q = progressive_quantize(weight, bits=bits)
            ...

Reference: Inspired by progressive quantization (Zhuang et al. 2018),
but applied from-scratch rather than post-training.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProgressiveQuantSchedule:
    """Maps training epoch to bit width.

    Default schedule for 400 epochs:
      Epochs   0- 40 (10%): INT8 (256 levels) — learn discrete features
      Epochs  40-120 (20%): INT4 (16 levels)  — compress to coarse discrete
      Epochs 120-200 (20%): INT2 (4 levels)   — near-ternary
      Epochs 200-400 (50%): Ternary (3 levels) — final quantization

    The fractions are tunable. More time at higher bits = better feature
    quality but slower convergence to ternary. More time at ternary =
    more optimization at the deployment precision.
    """

    def __init__(self, total_epochs: int,
                 schedule: list = None):
        """
        schedule: list of (fraction, bits) tuples, must sum to 1.0.
            Default: [(0.10, 8), (0.20, 4), (0.20, 2), (0.50, 'ternary')]
        """
        self.total_epochs = total_epochs
        if schedule is None:
            schedule = [
                (0.15, 8),          # 15%: INT8 — learn discrete features
                (0.25, 4),          # 25%: INT4 — compress to coarse discrete
                (0.60, 'ternary'),  # 60%: ternary — optimize at deployment precision
            ]
        self.schedule = schedule

        # Compute epoch boundaries
        self.boundaries = []
        cumulative = 0
        for frac, bits in schedule:
            start = cumulative
            cumulative += int(total_epochs * frac)
            self.boundaries.append((start, cumulative, bits))
        # Extend last phase to cover any rounding
        if self.boundaries:
            self.boundaries[-1] = (
                self.boundaries[-1][0], total_epochs + 1,
                self.boundaries[-1][2])

    def get_bits(self, epoch: int):
        """Return bit width for the given epoch.

        Returns int (8, 4, 2) or string 'ternary'.
        """
        for start, end, bits in self.boundaries:
            if start <= epoch < end:
                return bits
        return self.boundaries[-1][2] if self.boundaries else 'ternary'

    def __repr__(self):
        parts = []
        for start, end, bits in self.boundaries:
            b = f'{bits}b' if isinstance(bits, int) else bits
            parts.append(f'ep{start}-{end}: {b}')
        return f'ProgressiveQuantSchedule({", ".join(parts)})'


class _ProgressiveQuantFunction(torch.autograd.Function):
    """Symmetric uniform quantization at variable bit width with STE.

    Forward:
      scale = mean(|W|) * 2 / (2^bits - 1)   [per-tensor]
      w_q = round(clamp(W / scale, -max_val, max_val)) * scale
      where max_val = 2^(bits-1) - 1

    For 'ternary' mode:
      scale = mean(|W|)
      w_q = round(clamp(W / scale, -1, 1)) * scale

    Backward: STE (gradient passes through within clamp range)
    """
    @staticmethod
    def forward(ctx, weight, bits):
        if bits == 'ternary' or bits <= 1:
            # Ternary: {-1, 0, +1} × scale
            scale = weight.abs().mean().clamp(min=1e-5)
            w_scaled = weight / scale
            w_q = w_scaled.round().clamp(-1, 1)
            ctx.save_for_backward(w_scaled)
            ctx.clip_val = 1.0
            return w_q * scale
        else:
            # Symmetric INT-N
            max_val = 2 ** (bits - 1) - 1
            scale = weight.abs().mean().clamp(min=1e-5) * 2.0 / max_val
            w_scaled = weight / scale
            w_q = w_scaled.round().clamp(-max_val, max_val)
            ctx.save_for_backward(w_scaled)
            ctx.clip_val = float(max_val)
            return w_q * scale

    @staticmethod
    def backward(ctx, grad_output):
        w_scaled, = ctx.saved_tensors
        clip_val = ctx.clip_val
        mask = (w_scaled.abs() <= clip_val).float()
        return grad_output * mask, None


def progressive_quantize(weight, bits):
    """Quantize weight tensor at the given bit width."""
    return _ProgressiveQuantFunction.apply(weight, bits)


class ProgressiveConv1d(nn.Module):
    """Conv1d with progressive quantization schedule.

    The bit width is controlled externally via set_bits(). During training,
    the training loop calls set_bits() each epoch based on the schedule.
    At deployment, bits is set to 'ternary' permanently.

    Drop-in replacement for TernaryConv1d / BitLinearConv1d.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 stride: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        padding = kernel_size // 2
        self.weight = nn.Parameter(
            torch.randn(out_ch, in_ch // groups, kernel_size) *
            (2.0 / (in_ch * kernel_size)) ** 0.5
        )
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = 1
        self.groups = groups
        self.out_ch = out_ch
        self._bits = 8  # start at INT8
        # RMSNorm for stability (from BitNet)
        self.norm = nn.GroupNorm(1, out_ch)  # equivalent to LayerNorm

    def set_bits(self, bits):
        """Set the current quantization bit width."""
        self._bits = bits

    def forward(self, x: torch.Tensor, quantize: bool = True) -> torch.Tensor:
        if not quantize:
            return F.conv1d(x, self.weight, self.bias,
                           self.stride, self.padding, self.dilation, self.groups)

        w_q = progressive_quantize(self.weight, self._bits)

        # Activation quantization: INT8 absmax with STE
        with torch.no_grad():
            scale = x.abs().max().clamp(min=1e-5) / 127.0
        x_q = x + ((x / scale).round().clamp(-128, 127) * scale - x).detach()

        out = F.conv1d(x_q, w_q, self.bias,
                       self.stride, self.padding, self.dilation, self.groups)
        out = self.norm(out)
        return out

    @torch.no_grad()
    def get_ternary_weights(self) -> torch.Tensor:
        """Export as ternary for firmware."""
        scale = self.weight.abs().mean().clamp(min=1e-5)
        return (self.weight / scale).round().clamp(-1, 1)


class ProgressiveConvTranspose1d(nn.Module):
    """Transposed Conv1d with progressive quantization."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 stride: int = 2, groups: int = 1, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(in_ch, out_ch // groups, kernel_size) *
            (2.0 / (in_ch * kernel_size)) ** 0.5
        )
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self.stride = stride
        self.padding = kernel_size // 2
        self.output_padding = stride - 1
        self.dilation = 1
        self.groups = groups
        self._bits = 8
        self.norm = nn.GroupNorm(1, out_ch)

    def set_bits(self, bits):
        self._bits = bits

    def forward(self, x: torch.Tensor, quantize: bool = True) -> torch.Tensor:
        if not quantize:
            return F.conv_transpose1d(x, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.output_padding, self.groups,
                                       self.dilation)
        w_q = progressive_quantize(self.weight, self._bits)
        with torch.no_grad():
            scale = x.abs().max().clamp(min=1e-5) / 127.0
        x_q = x + ((x / scale).round().clamp(-128, 127) * scale - x).detach()
        out = F.conv_transpose1d(x_q, w_q, self.bias,
                                  self.stride, self.padding,
                                  self.output_padding, self.groups,
                                  self.dilation)
        out = self.norm(out)
        return out

    @torch.no_grad()
    def get_ternary_weights(self):
        scale = self.weight.abs().mean().clamp(min=1e-5)
        return (self.weight / scale).round().clamp(-1, 1)


class ProgressiveINT8Conv1d(nn.Module):
    """INT8 Conv1d with progressive quantization for the projection layer.

    At bits=8, this is identical to deployment. At higher bits during
    early training, the projection has more precision for learning
    the information bottleneck.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 stride: int = 1, groups: int = 1, bias: bool = True):
        super().__init__()
        padding = kernel_size // 2
        self.weight = nn.Parameter(
            torch.randn(out_ch, in_ch // groups, kernel_size) *
            (2.0 / (in_ch * kernel_size)) ** 0.5
        )
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = 1
        self.groups = groups
        self._bits = 8  # projection stays INT8 always
        self.norm = nn.GroupNorm(1, out_ch)

    def set_bits(self, bits):
        # Projection layer stays at max(bits, 8) — never goes below INT8
        self._bits = max(bits if isinstance(bits, int) else 8, 8)

    def forward(self, x: torch.Tensor, quantize: bool = True) -> torch.Tensor:
        if not quantize:
            return F.conv1d(x, self.weight, self.bias,
                           self.stride, self.padding, self.dilation, self.groups)
        # INT8 quantization (fixed at 8 bits for projection)
        scale = self.weight.abs().max().clamp(min=1e-5) / 127.0
        w_q = ((self.weight / scale).round().clamp(-127, 127) * scale
               - self.weight).detach() + self.weight
        with torch.no_grad():
            scale_x = x.abs().max().clamp(min=1e-5) / 127.0
        x_q = x + ((x / scale_x).round().clamp(-128, 127) * scale_x - x).detach()
        out = F.conv1d(x_q, w_q, self.bias,
                       self.stride, self.padding, self.dilation, self.groups)
        out = self.norm(out)
        return out


def set_model_bits(model, bits):
    """Set quantization bits on all ProgressiveConv modules in a model."""
    for module in model.modules():
        if hasattr(module, 'set_bits'):
            module.set_bits(bits)


__all__ = [
    'ProgressiveQuantSchedule', 'progressive_quantize',
    'ProgressiveConv1d', 'ProgressiveConvTranspose1d',
    'ProgressiveINT8Conv1d', 'set_model_bits',
]
