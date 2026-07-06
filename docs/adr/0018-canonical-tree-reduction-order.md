# ADR 0018 — Canonical tree-reduction order for cross-backend f32 sums

Status: **PROPOSED** (measured evidence below; decision pending)

## Context

The bit-match discipline (ADR 0004/0013) requires every f32 reduction on every
backend to fold in the host's sequential order. That made cross-backend parity
trivial to reason about, but it puts a hard latency floor under decode: a
sequential fadd chain of length `n` costs ~4 cycles per element no matter how
many idle SMs surround it.

After the v1.x contract-free pass (dp4a i8 GEMMs at 70–85% of weight-bandwidth
SOL, split ctx-parallel attention, i8 activation pipeline), profiling shows the
remaining decode GPU time on the 4090 is dominated by exactly these sequential
folds:

| component | share of decode GPU time | why it can't go faster |
|---|---|---|
| `rmsnorm_quant_i8` (4/layer) | ~42% | sequential sum-of-squares, 2560–6912-long fadd chain |
| attention softmax sum | (inside 15%) | sequential j-order fold |
| `rmsnorm_shared_f32` (final norm) | ~1% | same |

## Probe (2026-07-06, measured on the real model)

`rmsnorm_quant_i8`'s sum was temporarily switched to a **deterministic tree
order** (256 strided per-thread partials folded per-thread in index order, then
a power-of-two shared-memory tree) — the same shape a SIMD CPU implementation
would use. Results on BitNet 2B4T, RTX 4090:

- **decode: 187 → 327 tok/s (+75%)**, 38.6% of the 848 tok/s roofline —
  from this ONE kernel's sum.
- **perplexity IMPROVED**: rel err vs the fp reference went 2.957e-3 → 2.659e-4
  (the tree sum rounds less; it is *more* accurate than the sequential fold).
- `cuda_greedy_matches_transformers`: **PASSES** (the actual quality gate).
- `cuda_batch_decode_graph_argmax_matches_greedy`: passes.
- `cpu_cuda_parity`: **FAILS** — CUDA no longer bit-matches the CPU host,
  because only one side changed. This is the gate the decision is about.

## Decision (proposed)

Adopt a **documented canonical reduction order** for the f32 sums that sit on
the decode critical path — first `rmsnorm` sum-of-squares, then the attention
softmax sum — implemented identically on the host (`tritium-nn/src/ops`) and
every backend:

```
partials[t] = fold_{i ≡ t (mod 256), ascending i} x[i]²   (t = 0..255)
tree: for off in {128, 64, …, 1}: partials[t] += partials[t + off]
```

The order is fixed and platform-independent, so cross-backend bit-parity is
preserved *by construction* — it is a different canonical order, not an
abandonment of determinism. CPU-side it is SIMD-friendly (unblocks the AVX2
reduction items the sequential fold currently forbids).

## Consequences

- `cpu_cuda_parity` and greedy lockstep gates hold again once host + backends
  move together (one atomic change).
- Golden files / conformance vectors touching rmsnorm regenerate once.
- `*_matches_transformers` gates re-validate empirically (measured passing on
  CUDA; the tree sum is strictly more accurate, so risk is low).
- Metal/ROCm/wgpu backends must adopt the same order in the same release.
- Projected decode after rmsnorm + softmax-sum conversion plus the remaining
  contract-free work: ~500–650 tok/s (60–77% of roofline; the f16 LM head
  read, already at SOL, becomes the dominant term).

## Alternatives considered

- **Keep the sequential contract**: decode stays capped ≈ 200–280 tok/s on the
  4090 regardless of kernel quality; CPU SIMD reductions stay blocked.
- **GPU-graph-path-only divergence**: smaller blast radius but permanently
  forfeits cross-backend bit-parity (the repo's core testing lever) — worse
  than either consistent choice.
