# ADR 0019 — Persistent/megakernel decode

Status: **DEFERRED — premise measured and found weak** (2026-07-06)

## Context

After the v1.x decode passes (dp4a i8 GEMMs, split attention, ADR 0018
canonical reductions, rope+kv fusion, shuffle reduction tails), decode sits at
~345 tok/s = 40.6% of the 848 tok/s weight-bandwidth roofline, executed as a
CUDA graph of ~370 kernel nodes per token. The hypothesis: replace the graph
with one persistent cooperative kernel (stages separated by `grid.sync()`),
eliminating per-node dispatch and inter-kernel drain — the classic megakernel
argument for many-small-kernel decoders.

## Premise experiment (before any design work)

`scratchpad kbench_mega.cu`, RTX 4090, 370 stage boundaries, 10-block stages
(decode-normlike), 200 reps:

| boundary | cost per boundary |
|---|---|
| CUDA-graph kernel node | 0.77–1.22 µs |
| cooperative `grid.sync()` | 0.64–0.98 µs |
| eager stream launch | 1.3–1.4 µs |

Two further facts from the decode profile (`decode_r7`):

1. The sum of per-kernel execution medians (~3.0 ms) ≈ the measured wall time
   per token (~2.9 ms) — graph node boundaries are already close to free.
2. The dominant kernels are at internal floors a megakernel cannot move: the
   f16 LM head reads 656 MB/token at bandwidth speed-of-light, the dp4a GEMMs
   are DRAM-bound, rmsnorm/attention are latency-chain-bound *inside* their
   stages.

Projected megakernel upside: ~0.2 µs × 370 boundaries ≈ 74 µs/token (~3%),
against an engineering cost of merging every decode kernel into one
cooperative launch (fixed co-resident grid, per-stage block remapping, the
whole ADR 0018/bit-match discipline re-verified inside one translation unit),
plus new constraints (cooperative-launch occupancy caps, no per-stage grid
sizing, harder debugging/profiling).

## Decision

**Defer.** A ~3% ceiling does not justify the largest structural rewrite in
the codebase. Revisit only if (a) a future workload has far more, far smaller
stages, or (b) profiling with ncu (perf-counter access pending on the build
box) shows inter-node bubbles materially larger than this experiment's
estimate.

The higher-leverage successors, in order: BASTION drafter integration (spec
decode multiplies tokens per forward — ADR 0014's verifier is live), ncu-guided
work on the in-kernel floors, IMMA cp.async/ldmatrix for compute-bound prefill.

## Notes

Cooperative launch IS supported on the box (sm_89, checked at runtime), and
`grid.sync()` at 10-block grids costs ~0.6–1.0 µs — useful data for any future
fusion of 2–3 adjacent stages (e.g. a per-layer fused norm+quant+GEMM-prologue
experiment), which is a cheaper way to buy the same boundary savings
incrementally and does not require the full rewrite.
