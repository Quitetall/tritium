# ADR 0023 — Opt-in relaxed-reduction tier (`TRITIUM_KERNEL_TIER=fast`)

Status: **REJECTED BY MEASUREMENT** (2026-07-11). Step 1 (rmsnorm_fast) was
implemented in full, gated, benchmarked, and DELETED per this ADR's own
decision rule — see the verdict section at the end. The attention fast pair
was never authorized (its gate was rmsnorm reaching ≥3%).

## Context

The bit-exact sum-order contract (ADR 0018) is Tritium's product identity: the
f32 single-sequence path reproduces the host oracle `to_bits()`-equal, greedy
decode is token-exact vs `transformers`, and every kernel keeps one canonical
reduction order. That contract is also now the **dominant decode cost**:

- Recorded profiles (OPTIMIZATION-LOG): the flat GPU-time distribution at
  line 339 reads **attn 33% / rmsnorm 32% / lm_head 12% / GEMM 20%**; round-8
  per-layer counters read rmsnorm_quant 5.82µs + attn scores/reduce
  6.7+3.8µs vs GEMMs ~30µs (a GEMM-heavier per-layer mix — the two profiles
  bracket the truth). Both agree the reduction kernels are **latency-bound**
  (~1–2% DRAM) and sit at 30–65% of decode time depending on the cut.
- IMPORTANT scope correction (review of this RFC's draft): the rmsnorm
  reduction is ALREADY the canonical tree implemented with warp shuffles
  (ADR 0018, decode.cu:60-211) — "switch to a shuffle tree" shipped. The
  remaining rmsnorm levers that genuinely need a relaxed tier are
  **`--fmad=true` compilation** (fused multiply-add changes roundings) and
  **restructuring the per-thread strided fold** (changes accumulation
  order); attention's scores/reduce accumulation order is the other real
  target. The bit-exact tier's floor stands at ~45.9% of roofline (round 8).

Meanwhile the repo already ships and documents TWO non-bit-exact numerics
domains (book, "Numerics domains"): the M=N batch path (token-parity) and the
KV rungs (perplexity-gated). The precedent is established: a relaxed domain is
acceptable when it is **opt-in, gated, and honestly documented**.

## Proposal

A third numerics domain: `TRITIUM_KERNEL_TIER=fast` (default `exact`; any
other value rejected loudly, mirroring `TRITIUM_KV`).

- **Scope (initial)**: `rmsnorm_quant_*` with `--fmad=true` + a reordered
  per-thread fold, and the decode attention `scores/reduce` pair (reordered
  f32 accumulation, fmad on). GEMMs are OUT of scope (integer accumulation
  is already order-free; their limits are warp occupancy, not sum order).
- **Selection**: the same host-side single-dispatch-point pattern as
  `KvDtype::pick` (ADR 0022 guardrail 3) — a `KernelTier` enum in the CUDA
  backend; twin `_fast` kernels are new members of the existing families and
  the ADR 0022 twin-sync rule + drift test extend to them.
- **Contract**: the fast tier is **token-parity + perplexity gated**, exactly
  the batch domain's bar: greedy tokens equal the exact tier over the
  acceptance horizon; e2e perplexity rel-err within the KV-rung bound
  (≤ ~2e-3). Logit bits are explicitly NOT comparable. The default `exact`
  tier remains the conformance identity and CI default — no existing gate
  changes.
- **Composition rule** (added to the book's numerics-domains table): tiers
  compose with KV rungs and batching; each combination inherits the WEAKEST
  member's contract. `exact` + f32 KV + single-sequence remains the only
  bit-exact configuration.

## Expected win (to be measured, not promised)

rmsnorm + attention are 30–65% of decode GPU time depending on the profile
cut (see Context). fmad alone typically buys 10–30% on FMA-dense reduction
kernels; fold/order restructuring is workload-dependent. The honest
projection band is **+5–15% e2e decode** at M=1, additive with Track A's
memory work (different kernels). The measurement plan holds the tier to
that: any kernel whose fast variant wins <3% e2e is dropped (complexity not
carried for noise).

## Measurement & acceptance plan

1. `rmsnorm_fast` first (smallest surface): fmad-on + reordered fold,
   gate (token-parity over the 256-token acceptance horizon + ppl bound),
   bench. Decision point: proceed to attention only if ≥3% e2e.
2. `attn_scores/reduce_fast`: same procedure.
3. Each lands as its own reviewed commit with before/after numbers in
   OPTIMIZATION-LOG; "no win" results recorded and the variant deleted.
4. serve exposes the active tier in `/healthz` (operators must be able to see
   which numerics domain answered).

## Risks

- **Domain sprawl**: three domains × KV rungs × batch is a matrix. Mitigated
  by the composition rule + the one-line healthz disclosure + keeping `fast`
  out of every default path.
- **Token drift on long horizons**: near-tie argmaxes can flip late (known
  from the batch domain). The gate pins the acceptance horizon; the book
  documents that `fast` is not for reproducibility-critical use.
- **Twin growth**: +2 families × 1 tier. Inside ADR 0022's consolidation
  math; the codec-template consolidation (Track B) should treat tier as a
  template axis when it lands.

## Decision

PENDING review. Accepting this ADR authorizes step 1 (rmsnorm_fast) only;
each further kernel needs its measured decision point.

## Verdict (2026-07-11, measured — the tier is rejected)

Step 1 was executed end-to-end: `rmsnorm_quant_i8_fast` (the M=1 decode
graph's hottest reduction — 4 launches/layer) with the fold FUSED into the
staging pass (one barrier fewer), four independent `__fmaf_rn` accumulators
(4-way ILP, one rounding per element), all-thread folding and a
blockDim-generic combine; `TRITIUM_KERNEL_TIER=exact|fast` resolved once per
model build (graphs bake the picked symbol), loud-reject selector, healthz
disclosure. Gates all green under `fast`: 256-token greedy EXACTLY equal to
the transformers reference; perplexity rel 2.93e-3 vs the fp32 oracle
(exact tier: 2.66e-4 — the tier's own drift ≈ 3.2e-3, slightly above this
RFC's optimistic ~2e-3 band, inside the 1% acceptance bar).

**Bench (4090, quiet box, 512 decode steps, order-alternated ABBA ×2):
exact median 274.0 tok/s (266.8–274.9), fast median 278.5 (278.3–279.3) —
+1.6–1.9% e2e, pairwise median +1.75%. Below the ≥3% bar. Deleted.**

What the measurement settled: the two conflicting profiles (flat: rmsnorm
~32% of GPU time; per-layer counters: ~5%) are resolved in favor of the
counters. The rmsnorm cost at M=1 is launch overhead + barriers + the
elementwise passes — structural, not sum-order — so relaxing the reduction
order buys only the fold's latency chain, worth ~2% e2e. The kernel and
plumbing live in git history (this commit's parent) if a future architecture
changes the calculus; the honest projection for any revival is bounded by
this measurement, not by the flat profile.
