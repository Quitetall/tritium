# ADR 0023 — Opt-in relaxed-reduction tier (`TRITIUM_KERNEL_TIER=fast`)

Status: **PROPOSED** (2026-07-10). RFC — implementation gated on this ADR's
acceptance; measurement plan below decides kernel-by-kernel.

## Context

The bit-exact sum-order contract (ADR 0018) is Tritium's product identity: the
f32 single-sequence path reproduces the host oracle `to_bits()`-equal, greedy
decode is token-exact vs `transformers`, and every kernel keeps one canonical
reduction order. That contract is also now the **dominant decode cost**:

- Round-8+ profiling: attention (scores+reduce ~21%) and rmsnorm (~29%) are
  ~50% of decode GPU time and **latency-bound** (~1–2% DRAM), pinned to
  sync-heavy canonical-order reductions. The GEMMs, by contrast, are 20–25%.
- Every reduction speedup tried under the contract was rejected for reordering
  sums (recorded in kernel comments and the OPTIMIZATION-LOG); the bit-exact
  tier has hit its structural floor at ~45% of roofline.

Meanwhile the repo already ships and documents TWO non-bit-exact numerics
domains (book, "Numerics domains"): the M=N batch path (token-parity) and the
KV rungs (perplexity-gated). The precedent is established: a relaxed domain is
acceptable when it is **opt-in, gated, and honestly documented**.

## Proposal

A third numerics domain: `TRITIUM_KERNEL_TIER=fast` (default `exact`; any
other value rejected loudly, mirroring `TRITIUM_KV`).

- **Scope (initial)**: `rmsnorm_f32`/`rmsnorm_quant_*` (warp-shuffle tree
  reduction instead of the canonical serial order) and the decode attention
  `scores/reduce` pair (reordered f32 accumulation, wider warps, fewer
  syncs). GEMMs are OUT of scope (integer accumulation is already order-free;
  their limits are warp occupancy, not sum order).
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

rmsnorm + attention ≈ 50% of a ~52µs decode step. Latency-bound kernels with
sync-count reductions of 2–4× typically recover 30–60% of their time; the
honest projection band is **+10–20% e2e decode** at M=1, additive with Track
A's memory work (different kernels). The measurement plan holds the tier to
that: any kernel whose fast variant wins <3% e2e is dropped (complexity not
carried for noise).

## Measurement & acceptance plan

1. `rmsnorm_fast` first (smallest, biggest single share): implement, gate
   (token-parity over the 256-token acceptance horizon + ppl bound), bench.
   Decision point: proceed to attention only if ≥3% e2e.
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
