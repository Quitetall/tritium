"""PyTorch autograd wrappers for Tritium's ternary Conv1d + FSQ ops (ADR 0030).

.. warning::
   **This module is a correctness reference, not a performance path. Do not benchmark it.**
   Tensors cross to Rust as flat ``f32`` **Python lists**, so every call pays list
   materialisation in both directions — the cost is dominated by marshalling, not arithmetic,
   and a measurement here says nothing about Tritium's kernels.

   For anything performance-related use :mod:`tritium.torch` instead, which dispatches to the
   native backends. The two module names are easy to confuse; this note exists because they
   have been.

Each Rust op's ``forward``/``vjp`` is wrapped in a :class:`torch.autograd.Function`, and
:class:`TernaryConv1d` / :class:`FSQ` are drop-in :class:`torch.nn.Module` s. The ternary path
matches the transformer linears: latent fp32 weights, per-output-channel AbsMean scale, STE round in
the forward and a straight-through backward — so the gradients are exactly the ones the Rust
``gradcheck`` suite validates. Tensors cross to Rust as flat ``f32`` lists (a numpy/dlpack zero-copy
bridge is a performance follow-on); results are byte-exact with the Rust ops.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from ._tritium import (
    conv1d_forward,
    conv1d_vjp,
    fsq_forward,
    fsq_vjp,
    lsq_forward,
    lsq_vjp,
    ste_absmean_scale,
    ste_quantize_forward,
    ste_quantize_vjp,
)


def _flat(t: torch.Tensor) -> list:
    return t.detach().to(torch.float32).contiguous().view(-1).tolist()


def _to(flat: list, shape, ref: torch.Tensor) -> torch.Tensor:
    return torch.tensor(flat, dtype=torch.float32, device=ref.device).view(shape)


def _l_out(cfg) -> int:
    _b, _cin, _cout, l_in, k, stride, dilation, pad_left, pad_right, _g = cfg
    eff = dilation * (k - 1) + 1
    return (l_in + pad_left + pad_right - eff) // stride + 1


class _Conv1dFn(torch.autograd.Function):
    """``conv1d_forward``/``conv1d_vjp`` as an autograd Function.

    ``x``: ``[B, C_in, L]``; ``w``: ``[C_out, (C_in/groups)*K]``; ``scale``: ``[C_out]``.
    ``cfg`` is the 10-tuple ``(B, C_in, C_out, L_in, K, stride, dilation, pad_left, pad_right, groups)``.
    """

    @staticmethod
    def forward(ctx, x, w, scale, cfg):
        ctx.cfg = cfg
        ctx.save_for_backward(x, w, scale)
        y = conv1d_forward(_flat(x), _flat(w), _flat(scale), *cfg)
        return _to(y, (cfg[0], cfg[2], _l_out(cfg)), x)

    @staticmethod
    def backward(ctx, grad_out):
        x, w, scale = ctx.saved_tensors
        gx, gw, gs = conv1d_vjp(_flat(x), _flat(w), _flat(scale), _flat(grad_out), *ctx.cfg)
        return _to(gx, x.shape, x), _to(gw, w.shape, w), _to(gs, scale.shape, scale), None


class _TernarizeSTE(torch.autograd.Function):
    """Latent weight ``wf`` → ternary ``{-1,0,1}`` (round in the forward), masked straight-through
    backward (``gWf = grad/s_q · 1[|wf/s_q|<1]``). ``s_q`` is a detached, stop-gradient constant."""

    @staticmethod
    def forward(ctx, wf, s_q, rows, cols):
        ctx.save_for_backward(wf, s_q)
        ctx.shape = (rows, cols)
        trits = ste_quantize_forward(_flat(wf), _flat(s_q), rows, cols)
        return _to(trits, wf.shape, wf)

    @staticmethod
    def backward(ctx, grad_out):
        wf, s_q = ctx.saved_tensors
        rows, cols = ctx.shape
        gwf = ste_quantize_vjp(_flat(wf), _flat(s_q), rows, cols, _flat(grad_out))
        return _to(gwf, wf.shape, wf), None, None, None


class _LSQTernarizeFn(torch.autograd.Function):
    """Latent weight ``wf`` → ternary reconstruction ``round(clamp(wf/α))·α`` with a **learned** per-row
    step size ``alpha``. Backward routes the STE weight grad to ``wf`` and the LSQ step-size gradient to
    ``alpha`` (both train)."""

    @staticmethod
    def forward(ctx, wf, alpha, rows, cols):
        ctx.save_for_backward(wf, alpha)
        ctx.shape = (rows, cols)
        q = lsq_forward(_flat(wf), _flat(alpha), rows, cols)
        return _to(q, wf.shape, wf)

    @staticmethod
    def backward(ctx, grad_out):
        wf, alpha = ctx.saved_tensors
        rows, cols = ctx.shape
        gwf, ga = lsq_vjp(_flat(wf), _flat(alpha), rows, cols, _flat(grad_out))
        return _to(gwf, wf.shape, wf), _to(ga, alpha.shape, alpha), None, None


class _FSQFn(torch.autograd.Function):
    """``fsq_forward``/``fsq_vjp`` as an autograd Function over a ``[channels, len]`` activation."""

    @staticmethod
    def forward(ctx, x, channels, length, levels, bound, ste, alpha, seed):
        ctx.save_for_backward(x)
        ctx.args = (channels, length, levels, bound, ste, alpha)
        q = fsq_forward(_flat(x), channels, length, levels, bound, ste, float(alpha), int(seed))
        return _to(q, x.shape, x)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        channels, length, levels, bound, ste, alpha = ctx.args
        gx = fsq_vjp(_flat(x), _flat(grad_out), channels, length, levels, bound, ste, float(alpha))
        return _to(gx, x.shape, x), None, None, None, None, None, None, None


def ternary_conv1d(x: torch.Tensor, weight: torch.Tensor, cfg) -> torch.Tensor:
    """Functional ternary conv: quantize ``weight`` (per-output-channel AbsMean STE) then convolve."""
    c_out = cfg[2]
    k_g = (cfg[1] // cfg[9]) * cfg[4]
    wf = weight.reshape(c_out, k_g)
    s_q = torch.tensor(
        ste_absmean_scale(_flat(wf), c_out, k_g), dtype=torch.float32, device=x.device
    )
    trits = _TernarizeSTE.apply(wf, s_q, c_out, k_g)  # [C_out, K_g] in {-1,0,1}
    return _Conv1dFn.apply(x, trits, s_q, cfg)  # conv folds s_q → ternary conv


class TernaryConv1d(nn.Module):
    """Drop-in ternary 1-D convolution: latent fp32 weights, ternarized (STE) at forward time.

    Args mirror :class:`torch.nn.Conv1d` (``padding`` may be an ``int`` for symmetric, or a
    ``(left, right)`` tuple). The convolution runs on the Tritium CPU op; gradients are the ternary
    STE gradients validated by the Rust ``gradcheck`` suite.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding=0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        if in_channels % groups or out_channels % groups:
            raise ValueError("in_channels and out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        if isinstance(padding, (tuple, list)):
            self.pad_left, self.pad_right = int(padding[0]), int(padding[1])
        else:
            self.pad_left = self.pad_right = int(padding)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, c_in, l_in = x.shape
        cfg = (
            batch,
            self.in_channels,
            self.out_channels,
            l_in,
            self.kernel_size,
            self.stride,
            self.dilation,
            self.pad_left,
            self.pad_right,
            self.groups,
        )
        y = ternary_conv1d(x, self.weight, cfg)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1)
        return y


class LearnedTernaryConv1d(nn.Module):
    """Ternary Conv1d with an **LSQ learned** per-output-channel step size ``alpha`` instead of the fixed
    AbsMean scale (ADR 0030 Tier 1). ``alpha`` is a trained parameter, initialized to the AbsMean of the
    initial weight; both the latent weight and ``alpha`` receive gradients. Same constructor as
    :class:`TernaryConv1d`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding=0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        if in_channels % groups or out_channels % groups:
            raise ValueError("in_channels and out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        if isinstance(padding, (tuple, list)):
            self.pad_left, self.pad_right = int(padding[0]), int(padding[1])
        else:
            self.pad_left = self.pad_right = int(padding)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        k_g = (in_channels // groups) * kernel_size
        with torch.no_grad():
            wf = self.weight.reshape(out_channels, k_g)
            init_alpha = ste_absmean_scale(_flat(wf), out_channels, k_g)
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, c_in, l_in = x.shape
        c_out = self.out_channels
        k_g = (self.in_channels // self.groups) * self.kernel_size
        cfg = (
            batch,
            self.in_channels,
            c_out,
            l_in,
            self.kernel_size,
            self.stride,
            self.dilation,
            self.pad_left,
            self.pad_right,
            self.groups,
        )
        wf = self.weight.reshape(c_out, k_g)
        w_ternary = _LSQTernarizeFn.apply(wf, self.alpha, c_out, k_g)  # round(clamp(wf/α))·α
        ones = torch.ones(c_out, dtype=torch.float32, device=x.device)
        y = _Conv1dFn.apply(x, w_ternary, ones, cfg)  # α already folded into the weight ⇒ scale = 1
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1)
        return y


class FSQ(nn.Module):
    """Finite scalar quantization of a ``[channels, len]`` latent — the codec rate knob.

    ``levels`` is one ``L`` (>= 2) per channel; ``bound`` is ``"tanh"`` or ``"clamp"`` (use ``"clamp"``
    for a byte-exact deploy grid); ``ste`` is ``"hard"``, ``"soft"`` (with ``alpha``), or
    ``"stochastic"`` (with ``seed``).
    """

    def __init__(
        self,
        levels: Sequence[int],
        bound: str = "tanh",
        ste: str = "hard",
        alpha: float = 1.0,
        seed: int = 0,
    ):
        super().__init__()
        self.levels = [int(x) for x in levels]
        self.bound = bound
        self.ste = ste
        self.alpha = float(alpha)
        self.seed = int(seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FSQ is elementwise with a per-channel level, so any leading batch dims are folded into the
        # per-channel length: [..., C, L] → channel-major [C, prod(...)·L] → quantize → restore. The
        # codec latent is 2-D [C, L]; the decoder feeds 3-D [B, C, L].
        channels = x.shape[-2]
        if len(self.levels) != channels:
            raise ValueError(f"levels has {len(self.levels)} entries but x has {channels} channels")
        if x.dim() == 2:
            return _FSQFn.apply(
                x, channels, x.shape[-1], self.levels, self.bound, self.ste, self.alpha, self.seed
            )
        ndim = x.dim()
        lead = x.shape[:-2]
        length = x.shape[-1]
        to_front = (ndim - 2, *range(ndim - 2), ndim - 1)  # [C, lead..., L]
        xp = x.permute(to_front).reshape(channels, -1)
        q = _FSQFn.apply(
            xp, channels, xp.shape[1], self.levels, self.bound, self.ste, self.alpha, self.seed
        )
        q = q.reshape(channels, *lead, length)
        back = (*range(1, ndim - 1), 0, ndim - 1)  # [lead..., C, L]
        return q.permute(back).contiguous()
