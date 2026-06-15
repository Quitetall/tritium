#!/usr/bin/env python3
# ruff: noqa: E501
r"""Generate per-op torch reference goldens for the Tritium v0.20 inference spine.

These JSON files are committed under ``crates/tritium-nn/tests/goldens/`` and
replayed by the Rust per-op tests (WF-2) to pin each op to a torch reference
within the ADR-0004 tolerances (non-ternary fp32 ≤ 2e-3). All math is done in
**float32** on CPU for reorder-free, deterministic vectors.

The reference is HuggingFace ``transformers`` 5.5.3
``transformers.models.bitnet.modeling_bitnet`` (BitNet b1.58 2B4T). Where it
matters we call the *exact* library functions (``rotate_half``,
``apply_rotary_pos_emb``, ``eager_attention_forward``) rather than a paraphrase,
so the goldens cannot drift from the oracle.

Output JSON schema (mirrors the tritium-testkit ``ConformanceVector`` JSON style:
flat row-major ``Vec<f32>`` payloads, a stable string ``id`` per case, params as a
flat object). Each file is::

    {
      "op": "<op name>",
      "convention": "<short note, e.g. the RoPE pairing>",
      "reference": "transformers 5.5.3 modeling_bitnet (float32, CPU)",
      "cases": [
        { "id": "...", "params": { ... }, "inputs": { ... }, "expected_output": { ... } },
        ...
      ]
    }

All tensors in ``inputs``/``expected_output`` are flat row-major lists of f32 (or
ints for positions), laid out to match the Rust op signatures exactly so a test
can ``serde`` them straight into the op's argument slices.

============================================================================
OP FORMULAS (as pinned to modeling_bitnet.py)
============================================================================

rmsnorm  (BitNetRMSNorm.forward, lines 53-58)
    Computed in f32. For a length-``H`` row ``x`` with weight ``w`` and ``eps``:
        var      = mean(x_i^2)                       # mean over the H elements
        x_hat_i  = x_i * rsqrt(var + eps)
        out_i    = w_i * x_hat_i
    Matches the Rust signature rmsnorm(x, w, eps, out) on a flat [H] buffer.
    (The Rust op multiplies by w *after* normalize; ordering is associative in
    f32 to within tolerance — we emit w == ones and w == random to cover both.)

rope  (rotate_half + apply_rotary_pos_emb + BitNetRotaryEmbedding, lines 81-111, 264-326)
    *** CONFIRMED CONVENTION: NeoX-style "half-rotated" (NOT GPT-J interleaved). ***
    Evidence in modeling_bitnet.py:
      - rotate_half(x) = cat((-x2, x1)) where x1 = x[..., :d/2], x2 = x[..., d/2:]
        (lines 81-85). Verified: rotate_half([1,2,3,4]) == [-3,-4,1,2].
      - inv_freq = 1 / base ** (arange(0, d, 2)/d)  -> d/2 frequencies (lines 308-310).
      - emb = cat((freqs, freqs), dim=-1); cos = emb.cos(); sin = emb.sin()  (lines 321-323)
        so cos/sin have length d and element j shares its angle with j + d/2.
      - q_embed = q*cos + rotate_half(q)*sin  (lines 109-110).
    => Lane j in [0, d/2) is paired with lane j + d/2 (NOT j*2 / j*2+1).
       theta_j = pos * base ** (-2j/d), and for the pair (a, b) = (x[j], x[j+d/2]):
         out[j]      = a*cos(theta_j) - b*sin(theta_j)
         out[j+d/2]  = b*cos(theta_j) + a*sin(theta_j)
    base (rope_theta) = 500000.0, head_dim = 128 for BitNet 2B4T.
    The Rust signature rope_apply(x, positions, n_head, head_dim, theta) takes a
    flat [n_token, n_head, head_dim] buffer; goldens are laid out identically.

gqa_attention  (eager_attention_forward + repeat_kv, lines 114-148)
    Grouped-query attention, causal, naive, f32. For BitNet: n_head=20 Q heads,
    n_head_kv=5 KV heads (group size n_rep = n_head / n_head_kv = 4); query head h
    reads KV head h // n_rep (repeat_kv expands each KV head to n_rep consecutive
    Q heads — lines 114-123). scale = head_dim ** -0.5.
      scores[h, i, j] = scale * <q[i, h, :], k[j, kv(h), :]>     for j visible to i
      visible(i, j)   = j <= causal_offset + i                   (causal mask)
      masked j        => score = -inf  (added as a large negative bias)
      a[h, i, :]      = softmax_j(scores[h, i, :])               (over visible j)
      out[i, h, :]    = sum_j a[h, i, j] * v[j, kv(h), :]
    A fully-masked row (no visible keys) yields softmax over all -inf; the
    convention (uniform vs zero) is fixed by the softmax golden below and the WF-2
    op — here we emit a case whose mask leaves a row with a single visible key and
    a separate softmax golden for the all-(-inf) degenerate row.
    Rust signature: gqa_attention(q, k, v, seq, ctx, n_head, n_head_kv, head_dim,
    scale, causal_offset, out); q/out are [seq, n_head, head_dim], k/v are
    [ctx, n_head_kv, head_dim], all flat row-major.

softmax  (nn.functional.softmax over the last dim, used at line 143; f32)
    For a length-``L`` row ``x``:
        m       = max_i x_i
        e_i     = exp(x_i - m)
        out_i   = e_i / sum_i e_i
    Numerically-stable max-subtraction. Includes a large-magnitude row (to prove
    the max-subtraction) and a fully-masked (all -inf) row. For the all -inf row
    torch's softmax yields NaN (0/0); the committed golden records NaN so the Rust
    op's chosen convention is graded explicitly against torch.

============================================================================
Usage:  python3 tools/gen_op_goldens.py [OUTDIR]
        (default OUTDIR = crates/tritium-nn/tests/goldens)
Deps:   torch (>=2.0), transformers (with models.bitnet), numpy.
"""

from __future__ import annotations

import json
import math
import os
import sys

import torch

# Pin the exact library ops so the goldens track the oracle, not a paraphrase.
from transformers.models.bitnet.modeling_bitnet import (
    apply_rotary_pos_emb,
    eager_attention_forward,
    rotate_half,
)

torch.manual_seed(0)
DT = torch.float32

# BitNet b1.58 2B4T config constants (config.json on microsoft/bitnet-b1.58-2B-4T).
HEAD_DIM = 128
ROPE_THETA = 500000.0
N_HEAD = 20
N_HEAD_KV = 5
RMS_EPS = 1e-5
MAX_POS = 4096  # context_length


# Significant figures kept in the committed goldens. 7 sig-figs is ~1e-6
# relative, far inside the ADR-0004 non-ternary tolerance (2e-3), so rounding the
# stored vectors never changes the grading; it just keeps the JSON compact.
SIG_FIGS = 7


def _round_sig(x: float) -> float:
    """Round a finite float to SIG_FIGS significant figures; pass NaN/inf through."""
    if x == 0.0 or not math.isfinite(x):
        return x
    digits = SIG_FIGS - 1 - math.floor(math.log10(abs(x)))
    return round(x, digits)


def flat(t: torch.Tensor) -> list[float]:
    """Row-major flat f32 list (what the Rust op slices expect), rounded to
    SIG_FIGS significant figures to keep the committed goldens small."""
    return [_round_sig(v) for v in t.detach().to(torch.float32).reshape(-1).tolist()]


def rand(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, dtype=DT)


# --------------------------------------------------------------------------- #
# rmsnorm
# --------------------------------------------------------------------------- #
def rmsnorm_ref(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """BitNetRMSNorm.forward in f32 (lines 53-58)."""
    x = x.to(torch.float32)
    var = x.pow(2).mean(-1, keepdim=True)
    x_hat = x * torch.rsqrt(var + eps)
    return w * x_hat


def gen_rmsnorm() -> dict:
    cases = []

    def case(cid: str, x: torch.Tensor, w: torch.Tensor, eps: float):
        out = rmsnorm_ref(x, w, eps)
        cases.append(
            {
                "id": cid,
                "params": {"hidden": x.numel(), "eps": eps},
                "inputs": {"x": flat(x), "w": flat(w)},
                "expected_output": {"out": flat(out)},
            }
        )

    # Unit weights, small hand row (matches the Rust unit test x=[3,4]).
    case("unit-3-4", torch.tensor([3.0, 4.0]), torch.ones(2), 0.0)
    # Realistic hidden width with eps and random weights.
    case("h2560-eps1e-5", rand(2560), rand(2560), RMS_EPS)
    # Random weights, mid width.
    case("h128-randw", rand(128), rand(128), RMS_EPS)
    # Large-magnitude row (exercises the f32 variance accumulation).
    case("h64-largemag", rand(64) * 1.0e3, torch.ones(64), RMS_EPS)
    # All-equal row: var == value^2, out == w * sign(value)/sqrt(1+eps/value^2).
    case("h32-constant", torch.full((32,), 2.5), rand(32), RMS_EPS)

    return {
        "op": "rmsnorm",
        "convention": "out = w * x * rsqrt(mean(x^2) + eps); f32; w applied per-element",
        "reference": "transformers 5.5.3 modeling_bitnet BitNetRMSNorm (float32, CPU)",
        "cases": cases,
    }


# --------------------------------------------------------------------------- #
# rope  (NeoX half-rotated, via the exact transformers functions)
# --------------------------------------------------------------------------- #
def rope_cos_sin(positions: list[int], head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce BitNetRotaryEmbedding.forward cos/sin (lines 308-326) in f32.

    inv_freq[i] = 1 / theta ** (2i/head_dim), i in [0, head_dim/2).
    freqs       = outer(positions, inv_freq)                 -> [T, head_dim/2]
    emb         = cat((freqs, freqs), -1)                    -> [T, head_dim]
    cos, sin    = emb.cos(), emb.sin()
    """
    pos = torch.tensor(positions, dtype=DT)  # [T]
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(DT) / head_dim)
    )  # [head_dim/2]
    freqs = torch.outer(pos, inv_freq)  # [T, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [T, head_dim]
    return emb.cos(), emb.sin()


def rope_ref(x: torch.Tensor, positions: list[int], n_head: int, head_dim: int, theta: float) -> torch.Tensor:
    """Apply RoPE to a [n_token, n_head, head_dim] tensor using transformers'
    apply_rotary_pos_emb (NeoX half-rotated). Returns the rotated tensor in the
    same [n_token, n_head, head_dim] layout the Rust op writes back.
    """
    n_token = len(positions)
    cos, sin = rope_cos_sin(positions, head_dim, theta)  # [T, head_dim]
    # apply_rotary_pos_emb expects q,k as [batch, heads, seq, head_dim] and
    # cos/sin as [batch, seq, head_dim]; it unsqueezes cos/sin at dim 1.
    q = x.reshape(n_token, n_head, head_dim).transpose(0, 1).unsqueeze(0)  # [1, n_head, T, hd]
    cos_b = cos.unsqueeze(0)  # [1, T, hd]
    sin_b = sin.unsqueeze(0)  # [1, T, hd]
    q_rot, _ = apply_rotary_pos_emb(q, q, cos_b, sin_b)  # rotate q (k arg unused here)
    # back to [n_token, n_head, head_dim]
    out = q_rot.squeeze(0).transpose(0, 1).contiguous()
    return out


def gen_rope() -> dict:
    # Sanity-check the convention against a closed-form NeoX rotation so the
    # generator self-verifies (no silent drift if transformers changes).
    _assert_neox_closed_form()

    cases = []
    # Positions: 0, 1, a mid value, and near max context. head_dim 128 per BitNet.
    position_sets = {
        "pos-0": [0],
        "pos-1": [1],
        "pos-mid": [2048],
        "pos-nearmax": [MAX_POS - 1],  # 4095
        # A multi-token decode/prefill block covering all four regimes at once.
        "pos-block": [0, 1, 2048, MAX_POS - 1],
    }
    n_head = 2  # small; convention is per-head-dim, head count is irrelevant
    for cid, positions in position_sets.items():
        x = rand(len(positions), n_head, HEAD_DIM)
        out = rope_ref(x, positions, n_head, HEAD_DIM, ROPE_THETA)
        cases.append(
            {
                "id": cid,
                "params": {
                    "n_token": len(positions),
                    "n_head": n_head,
                    "head_dim": HEAD_DIM,
                    "theta": ROPE_THETA,
                    "positions": positions,
                },
                "inputs": {"x": flat(x)},
                "expected_output": {"out": flat(out)},
            }
        )

    return {
        "op": "rope",
        "convention": (
            "NeoX half-rotated: lane j in [0,d/2) paired with j+d/2; "
            "theta_j = pos * base**(-2j/d); rotate_half(x)=cat(-x[d/2:], x[:d/2]). "
            "Confirmed against transformers modeling_bitnet rotate_half/apply_rotary_pos_emb."
        ),
        "reference": "transformers 5.5.3 modeling_bitnet apply_rotary_pos_emb (float32, CPU)",
        "cases": cases,
    }


def _assert_neox_closed_form() -> None:
    """Independently rotate one head with the closed-form NeoX pairing and check
    it equals the transformers path — guards against a convention regression."""
    pos = [1, 7]
    hd = HEAD_DIM
    x = rand(len(pos), 1, hd)
    got = rope_ref(x, pos, 1, hd, ROPE_THETA).reshape(len(pos), hd)
    # closed form
    want = torch.empty_like(got)
    half = hd // 2
    for ti, p in enumerate(pos):
        for j in range(half):
            theta_j = p * (ROPE_THETA ** (-2.0 * j / hd))
            c, s = math.cos(theta_j), math.sin(theta_j)
            a = x[ti, 0, j].item()
            b = x[ti, 0, j + half].item()
            want[ti, j] = a * c - b * s
            want[ti, j + half] = b * c + a * s
    err = (got - want).abs().max().item()
    assert err < 1e-4, f"RoPE convention mismatch vs NeoX closed form: max err {err}"


# --------------------------------------------------------------------------- #
# softmax
# --------------------------------------------------------------------------- #
def gen_softmax() -> dict:
    cases = []

    def case(cid: str, rows: list[list[float]]):
        x = torch.tensor(rows, dtype=DT)
        out = torch.softmax(x, dim=-1)  # torch's stable softmax; matches line 143
        cases.append(
            {
                "id": cid,
                "params": {"rows": x.shape[0], "row_len": x.shape[1]},
                "inputs": {"x": flat(x)},
                "expected_output": {"out": flat(out)},
            }
        )

    NEG = float("-inf")
    case("uniform", [[0.0, 0.0, 0.0, 0.0]])
    case("simple", [[1.0, 2.0, 3.0, 4.0]])
    # Large-magnitude row: without max-subtraction exp overflows to inf.
    case("large-magnitude", [[1000.0, 1001.0, 999.0, 1000.5]])
    # Partially masked row (one key masked) — the common attention case.
    case("partial-mask", [[2.0, NEG, 0.5, -1.0]])
    # Fully-masked row: all -inf. torch yields NaN (0/0); golden records it so the
    # Rust op's convention is graded explicitly against torch.
    case("fully-masked", [[NEG, NEG, NEG, NEG]])
    # Multi-row batch (exercises row striding).
    case(
        "multi-row",
        [[0.1, 0.2, 0.3], [5.0, -5.0, 0.0], [1.0, 1.0, 1.0]],
    )

    return {
        "op": "softmax",
        "convention": (
            "row-wise, max-subtraction stable: out_i = exp(x_i - max)/sum exp(x - max); "
            "fully-masked (all -inf) row -> NaN per torch (0/0)."
        ),
        "reference": "torch.softmax dim=-1 (float32, CPU); matches modeling_bitnet line 143",
        "cases": cases,
    }


# --------------------------------------------------------------------------- #
# gqa_attention  (via transformers' eager_attention_forward)
# --------------------------------------------------------------------------- #
class _AttnMod:
    """Minimal stand-in for the bits eager_attention_forward reads off `module`."""

    def __init__(self, n_head: int, n_head_kv: int):
        self.num_key_value_groups = n_head // n_head_kv
        self.training = False


def gqa_ref(
    q: torch.Tensor,  # [seq, n_head, head_dim]
    k: torch.Tensor,  # [ctx, n_head_kv, head_dim]
    v: torch.Tensor,  # [ctx, n_head_kv, head_dim]
    n_head: int,
    n_head_kv: int,
    head_dim: int,
    scale: float,
    causal_offset: int,
) -> torch.Tensor:
    """Naive causal GQA via transformers' eager_attention_forward (repeat_kv +
    softmax + matmul). Returns [seq, n_head, head_dim] to match the Rust `out`."""
    seq = q.shape[0]
    ctx = k.shape[0]
    # to [batch, heads, seq, head_dim]
    qb = q.transpose(0, 1).unsqueeze(0)  # [1, n_head, seq, hd]
    kb = k.transpose(0, 1).unsqueeze(0)  # [1, n_head_kv, ctx, hd]
    vb = v.transpose(0, 1).unsqueeze(0)
    # additive causal mask [1, 1, seq, ctx]: query i (abs pos causal_offset+i)
    # sees key j iff j <= causal_offset + i.
    mask = torch.zeros(1, 1, seq, ctx, dtype=DT)
    for i in range(seq):
        limit = causal_offset + i
        for j in range(ctx):
            if j > limit:
                mask[0, 0, i, j] = float("-inf")
    mod = _AttnMod(n_head, n_head_kv)
    attn_out, _ = eager_attention_forward(mod, qb, kb, vb, mask, scaling=scale, dropout=0.0)
    # eager_attention_forward returns [batch, seq, n_head, head_dim] (it does the
    # transpose(1,2) internally). -> [seq, n_head, head_dim]
    return attn_out.squeeze(0).contiguous()


def gen_gqa() -> dict:
    cases = []

    def case(cid: str, seq: int, ctx: int, causal_offset: int, n_head: int, n_head_kv: int):
        head_dim = HEAD_DIM
        scale = head_dim**-0.5
        q = rand(seq, n_head, head_dim)
        k = rand(ctx, n_head_kv, head_dim)
        v = rand(ctx, n_head_kv, head_dim)
        out = gqa_ref(q, k, v, n_head, n_head_kv, head_dim, scale, causal_offset)
        cases.append(
            {
                "id": cid,
                "params": {
                    "seq": seq,
                    "ctx": ctx,
                    "n_head": n_head,
                    "n_head_kv": n_head_kv,
                    "head_dim": head_dim,
                    "scale": scale,
                    "causal_offset": causal_offset,
                },
                "inputs": {"q": flat(q), "k": flat(k), "v": flat(v)},
                "expected_output": {"out": flat(out)},
            }
        )

    # Decode step: 1 new query over a full 8-token context (offset 7 -> sees all).
    case("decode-seq1", seq=1, ctx=8, causal_offset=7, n_head=N_HEAD, n_head_kv=N_HEAD_KV)
    # Prefill: 4 query tokens, ctx==seq, offset 0 -> triangular causal mask.
    case("prefill-4", seq=4, ctx=4, causal_offset=0, n_head=N_HEAD, n_head_kv=N_HEAD_KV)
    # Decode with partial context: 1 query at abs pos 2, ctx 5 -> sees keys 0..=2,
    # keys 3,4 masked (a row that is *not* fully visible).
    case("decode-partial", seq=1, ctx=5, causal_offset=2, n_head=N_HEAD, n_head_kv=N_HEAD_KV)
    # Multi-token prefill, larger context than seq (cached prefix): seq 3 over ctx
    # 6 with offset 3 -> query i sees keys 0..=3+i.
    case("prefill-cached", seq=3, ctx=6, causal_offset=3, n_head=N_HEAD, n_head_kv=N_HEAD_KV)

    # Fully-masked row: a 2-query block where the FIRST query (abs pos = offset)
    # is given an offset of -1 so it can see *no* keys (j <= -1 is never true).
    # This stresses the degenerate softmax path inside attention. torch produces
    # NaN for that row; the golden records it so the Rust op's all-masked
    # convention is graded against torch explicitly.
    head_dim = HEAD_DIM
    scale = head_dim**-0.5
    seq, ctx = 2, 3
    q = rand(seq, N_HEAD, head_dim)
    k = rand(ctx, N_HEAD_KV, head_dim)
    v = rand(ctx, N_HEAD_KV, head_dim)
    out = gqa_ref(q, k, v, N_HEAD, N_HEAD_KV, head_dim, scale, causal_offset=-1)
    cases.append(
        {
            "id": "fully-masked-row",
            "params": {
                "seq": seq,
                "ctx": ctx,
                "n_head": N_HEAD,
                "n_head_kv": N_HEAD_KV,
                "head_dim": head_dim,
                "scale": scale,
                "causal_offset": -1,
            },
            "inputs": {"q": flat(q), "k": flat(k), "v": flat(v)},
            "expected_output": {"out": flat(out)},
            "note": "row 0 (abs pos -1) sees no keys -> torch softmax NaN; tests the all-masked convention",
        }
    )

    return {
        "op": "gqa_attention",
        "convention": (
            "causal GQA, n_head=20 Q / n_head_kv=5 KV (group 4), scale=1/sqrt(head_dim); "
            "query head h reads KV head h//(n_head/n_head_kv); query i sees key j iff "
            "j <= causal_offset + i. Fully-masked row -> torch NaN."
        ),
        "reference": "transformers 5.5.3 modeling_bitnet eager_attention_forward + repeat_kv (float32, CPU)",
        "cases": cases,
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    outdir = sys.argv[1] if len(sys.argv) > 1 else "crates/tritium-nn/tests/goldens"
    os.makedirs(outdir, exist_ok=True)

    generators = {
        "rmsnorm.json": gen_rmsnorm,
        "rope.json": gen_rope,
        "softmax.json": gen_softmax,
        "gqa_attention.json": gen_gqa,
    }

    for fname, gen in generators.items():
        payload = gen()
        path = os.path.join(outdir, fname)
        with open(path, "w") as fh:
            # Compact separators (no per-element whitespace) keep the large
            # attention vectors small; NaN/inf preserved via allow_nan.
            json.dump(payload, fh, separators=(",", ":"), allow_nan=True)
            fh.write("\n")
        n = len(payload["cases"])
        print(f"wrote {path}  ({n} cases)")

    print("RoPE convention: NeoX half-rotated (confirmed vs modeling_bitnet rotate_half/apply_rotary_pos_emb)")


if __name__ == "__main__":
    main()
