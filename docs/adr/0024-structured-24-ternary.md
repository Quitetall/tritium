# ADR 0024 — 2:4 structured ternary sparsity (sparse-tensor-core co-design)

Status: **PROPOSED** (2026-07-10). Trainer-hook spec handed to the plan-0039
Fisher track; kernel work gated on a 2:4-trained checkpoint existing.

## Context

Ternary's zero state is 42.2% of BitNet 2B4T's weights (measured,
`tritium report sparsity`), but unstructured zeros are unexploitable in the
regimes that matter: the M=1 decode GEMMs are warp-starved (not
compute-bound), index-based sparse formats lose to dense entropy packing at
this density (census: bitmap+signs 1.578 b/w vs TQ1_0's real 1.625 vs floor
1.560), and all-zero 256-blocks are only 2.44% (block-skip wired in A1b,
honest ~noise on this model). Where sparsity genuinely pays is the
**compute-bound regimes** — prefill (IMMA tensor-core path) and batched M=N
decode — and the hardware primitive for that is `mma.sp`: **2:4 structured
sparsity doubles i8 tensor-core throughput** on sm_80+.

No published work applies 2:4 to ternary weights (same novelty class as
ternary spec-decode). The fit is unusually good: ternary training already
produces 42% zeros; the question is only their PLACEMENT.

## Measured feasibility (BitNet 2B4T, full-model 4-group census)

| zeros in 4-group | fraction |
|---|---|
| 0 | 12.9% |
| 1 | 32.7% |
| 2 | 32.9% |
| 3 | 15.7% |
| 4 | 5.8% |

- **54.4% of 4-groups already satisfy 2:4.**
- Post-hoc forcing (zeroing the 2−z smallest per violating group) would zero
  **14.63% of all weights** on top of the natural 42.2% — far beyond any PTQ
  quality budget. **Conclusion: 2:4 ternary must be trained in, not
  retrofitted.** This ADR therefore specifies a trainer constraint, not a
  repack.
- Post-2:4 zero rate ≈ 56.8% — as a bonus, bitmap+signs (A4) at p=0.57 packs
  at ~1.43 b/w, so the structured checkpoint also SHRINKS.

## Decision (proposed)

1. **Trainer hook (plan 0039/0040 — the SALT distillation loop)**: a
   projected-STE mask — after each optimizer step, per consecutive 4-group
   along k, keep the 2 largest-saliency weights eligible for ±1 and force the
   rest's ternary projection to 0. Saliency = the 0039 Fisher estimate where
   available, |w| otherwise. The constraint binds only the ~45% of groups
   below 2 zeros, and the trained network redistributes — the 14.6% PTQ
   distortion number is the ceiling the trainer must beat, not pay.
2. **Gate**: the 2:4 student's perplexity vs an UNCONSTRAINED student of the
   same recipe/bpw; budget ≤ the KV-rung bar (~0.3% rel). If training can't
   meet it, the ADR is rejected with the measurement recorded.
3. **Kernel (after a checkpoint exists)**: `mma.sp`-based i8 prefill/batch
   GEMM twins — ternary 2:4 compressed operand = per 4-group, 2 trit values
   + the standard 2-bit sparse-metadata indices. Expected: ~2× the
   compute-bound regimes (prefill tok/s, batched aggregate at high M). The
   ADR 0022 twin-sync rule + drift test extend to the new family; Track B's
   codec consolidation should treat "2:4" as a codec axis.
4. **Interchange**: a 2:4-trained model remains a VALID ordinary ternary
   model (the zeros are just placed) — every existing kernel/gate runs it
   unchanged; the sparse kernels are a pure fast path. No format fork.

## Verification plan

- Census extension already measurable per checkpoint
  (`tritium report sparsity` + the 4-group histogram) — the trainer's
  compliance is auditable from the GGUF alone.
- Kernel gates: conformance vectors for the mma.sp path (i8-exact),
  prefill/batch bench before/after, acceptance suite on the 2:4 student.

## Consequences

- The 2× is conditional on training success — honestly gated, cheaply
  falsifiable (one student run with the mask vs without).
- Nothing ships until a checkpoint exists; the only near-term cost is the
  0039 trainer hook (small: a mask in the quantize-forward).
- If accepted end-to-end, ternary's "sparsity advantage" claim becomes
  concrete: structured placement, tensor-core execution, smaller packing —
  all from zeros the training already produces.
