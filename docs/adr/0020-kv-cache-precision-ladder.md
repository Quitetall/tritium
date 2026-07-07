# ADR 0020 — KV-cache precision ladder (f16 → i8 → ternary "KVTQ")

Status: **RUNG 1 (f16) ACCEPTED** (2026-07-07) — implemented + measured:
long-context decode (ctx≈4K) **68–72 → 95.6 tok/s (+33–40%)**, short-context
unchanged (latency-bound, as predicted), perplexity rel err 2.659e-4 →
1.582e-3 (~0.16%, inside the quality bar), KV memory halved. Opt-in via
`TRITIUM_KV_F16=1`; the f32 default stays bit-exact and gate-covered.
i8-grouped and ternary rungs remain PROPOSED.

## Context

The KV cache is f32: `[max_ctx, kv_width]` per layer per direction —
~630 MB at 4K context for BitNet 2B4T, and the bytes attention must re-read
every token. Two distinct motivations pull the same lever:

1. **Long-context speed.** At short context, decode attention is
   latency-bound (ncu: ~2% of every throughput ceiling) and KV compression
   buys *nothing*. Past ~1–2K context the score/reduce phases become
   KV-bandwidth-bound; at 4K, KV traffic (~630 MB/token) rivals the entire
   weight stream. Halving or 16×-ing those bytes is then a direct speedup.
2. **Memory.** Longer contexts, more concurrent sequences (M=N decode),
   or smaller devices.

The user's question — "KVTQ: compress the KV cache to ternary digits with
fine-grained dynamic quantization" — is the aggressive end of a spectrum,
and the honest engineering answer is to build the *ladder*, measure each
rung against the perplexity gate, and keep what survives.

## The ladder

| rung | bytes @ 4K ctx | expected quality | notes |
|---|---|---|---|
| f32 (today) | 630 MB | reference | bit-parity baseline |
| **f16** | 315 MB | ~lossless (one rounding per written K/V) | pure dtype swap; industry default |
| **i8 + per-group scales** (g=64) | ~165 MB | near-lossless (KIVI/KVQuant-class) | K wants per-channel grouping (RoPE outliers); V per-token |
| **ternary + per-group scales** ("KVTQ", g=32–64) | ~55 MB | open research — the experiment | integer attention: i8 Q × ternary K via the dp4a/A8 exactness argument, per-group dequant after integer accumulation |

Ternary-specific notes:
- KV values are **activations, not weights** — nothing makes them ternary by
  nature. Fine-grained *dynamic* scales (per 32–64-element group, per head,
  per token, computed at append time) are what could make 1.58-bit viable;
  published 2-bit KV work (KIVI) already needs per-channel K grouping and
  benefits from residual/sink handling.
- The prize beyond memory: Q quantized per-token to i8 against ternary K
  makes the score phase *integer* — the same multiplication-free story as
  the weight GEMMs, with exact int accumulation inside each group.
- K and V will likely land on different rungs (K is harder; V quantizes
  gently). The ladder is per-tensor-role, not global.

## Contract

Every rung below f32 **breaks bit-parity with the f32 reference** (each
written K/V rounds once). Following the ADR 0013 precedent (f16 LM head in
the graph path): non-f32 KV is **opt-in per model instance**
(`kv_precision` in the decode-model config; default f32), and each rung
ships behind the perplexity gate (e2e rel-err bound) plus a long-context
quality probe. CPU backend follows the same ladder when a rung is promoted
to default, keeping CPU↔CUDA lockstep meaningful per rung.

## Implementation shape

One abstraction, not N forks: the KV arena gets an element codec
(store/load pair) that every producer/consumer goes through —
~15 kernel sites: `rope_kv_fused_g`, `kv_append_{batch,tree,mdecode}`,
attention `{warp, scores/reduce, tree pair, ctrl pair, batch, mdecode
split}` — each becomes codec-templated (CUDA: `__half2float`/f32 identity
first; group-scale variants add a scales side-arena sized
`ctx × kv_width/G`).

Sequencing: f16 first (plumbing pass, expected keep), i8-grouped second
(validates the scales side-arena machinery), ternary last (the research
rung, group-size and K/V-asymmetry sweeps on the perplexity harness).

## Ops notes

- `TRITIUM_KV_F16` is read by the **CUDA backend only**; ROCm/Metal/CPU
  ignore it (their rungs land when promoted). Values other than `1`/`0` are
  rejected loudly at model build.

## Consequences

- Long-context decode stops being KV-bound at whatever rung survives the
  gate; short-context decode is untouched (and unhelped).
- The M=N batch path's per-sequence KV shrinks identically — more
  concurrent sequences per GB.
- Verify-tree KV (provisional rows) inherits the codec transparently since
  appends and attention share the arena code paths.
