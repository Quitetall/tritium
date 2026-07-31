# ADR 0035 — Frontier-methods integration: executing ADR 0034 inside the frozen campaign

Status: **PROPOSED** (2026-07-30)

- **Deciders:** Brian Lam
- **Relates:** executes the adoption decisions of
  [ADR 0034](./0034-next-gen-ternary-research.md) (research intake, 2026-07-30);
  operates strictly inside [ADR 0028](./0028-salt-v2-additive-ternarization.md)'s
  frozen gates via [plan 0043](../plans/0043-salt-v2-sota-campaign.md)'s amendment
  rules; implementation plan is
  [plan 0054](../plans/0054-frontier-methods-integration.md); amends the *target
  specification* (not the gate) of [ADR 0024](./0024-structured-24-ternary.md) and
  the drafter recipe of [ADR 0021](./0021-drafter-architecture.md)/[ADR 0032](./0032-spec-decode-cost-model-and-next-levers.md).
- **User decisions bound into this ADR (2026-07-30):** full sweep of ADR 0034's
  executable items; **native Rust ports** (Python implementations serve only as
  reference oracles); Stage-7 grid **enlarged by dated amendment**, not by a
  separate track.

> **Claim boundary.** This ADR authorizes implementation work and one preregistration
> amendment. It marks no empirical gate complete and predicts no ablation outcome.
> Every adopted mechanism must win its bracket under plan 0043's unchanged
> successive-halving protocol and thresholds, or be recorded as a negative result.

## Context

ADR 0034 identified the strongest known mechanism at each stage of the ternary
pipeline and decided to adopt them as ablation-gated candidates. Exploration of the
codebase (2026-07-30) established the actual integration surfaces, and three findings
reshape naive "port the paper" plans into something smaller and sharper:

1. **The Kronecker curvature backend already exists.** The July 22 series shipped the
   complete K-FAC capture pipeline: `InputGramAccumulator`/`OutputFisherAccumulator`
   (`tritium-quantize/src/salt_v2_curvature.rs`), the `S2KF` durable artifact
   (`salt_v2_evidence.rs`), the PyO3/PyTorch capture adapter (`tritium-py`
   `kronecker.rs` + `torch/ptq.py`, three frozen objective IDs), and the resumable
   506-tensor Qwen session (`tritium-salt` `ptq_driver/capture.rs`). What ADR 0034
   called "evaluate KronQ as the capture backend" resolves to a *cost* problem: the
   dominant term is **one full calibration replay per tensor** (506 replays at 27B).
   There is no YAQA power-iteration baseline in this repo, so KronQ's reported 10×
   cannot be inherited; the honest local lever is **shared-forward capture** (one
   replay covers every projection consuming the same layer input), whose win must be
   measured, not assumed.
2. **CAT-Q's two-stage schedule is already frozen here.** `SaltV2Refinement::PvKl`'s
   validator enforces a ≥20% hard tail (`salt_v2_model.rs:2148`) and
   `RecoverySchedule` is an exact 80/20 soft/hard split — CAT-Q's γ=0.8 by
   construction. The genuinely new CAT-Q content is (a) the **softened-relay
   continuous fit** as an initialization basin for the exact PTQ solver, (b)
   **learnable modulation** shaping that basin, and (c) the **sliding-window
   output-reconstruction optimizer** — for which the measurement half
   (`OutputReconstructionSchedule::SlidingWindows`, `OutputReconstructionAccumulator`,
   `select_output_reconstruction`) already exists but is called by nothing outside
   tests, with block outputs explicitly walled off in production
   (`execution.rs` rejects `has_block_outputs()`).
3. **HESTIA does not exist anywhere in the tree, but its skeleton does.** `lsq_ste`
   is the two-input-quantize-op precedent, `FsqSte::SoftRound{alpha}` the annealed
   gradchecked surrogate precedent, Python `AnnealedSTE` the τ-ramp precedent, and
   `RecoverySchedule`/`BypassSchedule` the refined-track policy seam. One property
   makes the port unusually clean: the softmax relaxation is smooth in *both* weights
   and temperature, so — unlike STE — **both gradients are finite-differenceable**,
   which strengthens Gate C rather than weakening it. The missing per-tensor
   sensitivity input for τ-scheduling is exactly what the S2KF Kronecker evidence
   already provides (Gram trace × output-Fisher mean); no new Hutch++ estimator is
   needed.

## Decision

Execute ADR 0034's full executable sweep as seven workstreams (plan 0054 carries the
implementation detail; gates summarized here). Native Rust is the product surface for
every mechanism; any Python twin exists only as a golden oracle for the Rust port.

### WS-A — Curvature capture cost (27B critical path)

Measure the existing per-tensor capture cost on the 1.7B rung (the denominator —
plan 0043 forbids extrapolated forecasts), then implement **shared-forward capture**:
one calibration replay feeds the S2KF builders of every projection sharing a layer
input. The invariants that make this safe are already designed in: canonical dyadic
reduction over *global sample ordinals* deliberately excludes batch boundaries and
shard order from evidence identity, so shared-forward records must be — and are
gated to be — **byte-identical** to per-tensor records on the same frozen corpus.
S2KF layout, `CurvatureSourceId` binding, atomic batch semantics, and
PSD-after-damping-with-explicit-failure are unchanged.

**Gates:** byte-identity golden; measured ≥3× replay reduction at 1.7B (honest number
reported whatever it is); Stage-2 perturbation-ordering gate green on shared-forward
artifacts; measured cost receipt feeds the 27B spend forecast.

### WS-B — CAT-Q mechanisms into the PTQ track

1. **Softened-relay initialization basin** in `deterministic_initial_scales`: a
   continuous fit of scales under the two-sided tanh relay
   `f(W;s,Δ) = [tanh(s(W−Δ)) + tanh(s(W+Δ))]/[2·tanh(s)]` with annealed sharpness
   (s₀=30 per CAT-Q), projected through the existing exact E/M solver. The
   accept-only-if-improves guard makes the basin strictly safe: it can only win.
2. **Learnable modulation (δμ, δα, δΔ) as basin-internal parameters.** Nothing is
   stored: δα is what `solve_scales` already computes; δμ/δΔ shape the soft fit and
   vanish at projection. This satisfies ADR 0028's zero-point ban with zero format
   or runtime impact — deliberately cheaper than the `SaltV2Transform` seam, which
   remains available if a *stored* transform ever proves necessary (that would be a
   new preregistration).
3. **Sliding-window output-reconstruction optimizer** behind the declared seam
   (`SaltV2ExternalStage::BlockOutputReconstruction` +
   `SaltV2ModelStageDriver::run_stage`): per window, generate scale-refit candidates
   against cached activations, score with the existing accumulator, select with the
   existing selector; un-wall block outputs in `execution.rs` for transcripts that
   declare window coverage. Windows refit **scales against fixed trits**, preserving
   the bounded-memory one-tensor master-fit invariant (trit refits stay per-tensor).

**Gates:** small-matrix goldens vs the CAT-Q reference (BitTern repo) for the relay
math; every variant is a Stage-7 grid value that must survive successive halving;
`recipe_digest` extended for every new config field; ablation rows added to the
frozen `BASELINES` tuple in `run-baseline-ablation.py` **and**
`verify-estimator-refinement-receipt.py` in lockstep; E/M-non-increasing and
P-monotonicity property tests stay green.

### WS-C — HESTIA differentiable ternarization into the refined track

1. **`hestia_relax` CPU op** (`tritium-train/src/ops/hestia.rs`): forward is the
   expectation over `π_τ(q|w) ∝ exp(−(w/γ−q)²/τ)`, `q ∈ {−1,0,+1}`; vjp in both
   inputs; tape method beside `lsq_ste`; gradcheck on both inputs.
2. **`TempSchedule`** (LrSchedule-shaped) with per-tensor scaling
   `τ_i(t) = τ̄(t)·e^(α·s_i)`, where `s_i` derives from S2KF evidence; policy wiring
   as a `BypassSchedule` sibling in `salt_v2_recovery.rs`; τ reaches its floor at the
   80% Hard boundary, so the hard-export contract (`convert_qat_hard`) is untouched.
3. **CUDA + portable manifest V3** adding `graph.hestia_relax` (the V2 precedent:
   one new op = one new manifest version), CPU reference + CUDA implementation +
   bit-exact vectors. The soft phase runs **dense matmul on the expectation** — the
   packed `salt_matmul` path cannot and need not consume soft weights.
4. **Python `hestia` estimator** in the estimator catalog as the golden oracle only,
   with its own projection helper (the `_projection` hard-forward contract does not
   admit a soft-expectation forward), flagged non-exportable until τ→0.

**Gates:** gradcheck both inputs; portable-V3 conformance green on CPU + CUDA;
refined-track A/B (HESTIA soft phase vs current STE soft phase) on the rung-2 recipe
under the unchanged Stage-7 selection threshold.

### WS-D — GDN-sensitivity preflight gate (pre-Stage-8, 27B critical path)

Freeze the probe protocol *in the plan-0043 amendment before any probe runs*:
matched-bpw PTQ probes on DeltaNet-block vs full-attention-block matrices from the
pinned checkpoint; the binding metric is **divergence growth along the recurrence**
(rollout via the content-bound Qwen3.5-family reference adapter with only the probed
block ternarized), because per-layer MSE is a proven-failed proxy for recurrent
error accumulation (Ternary Mamba). The frozen routing rule: excessive GDN divergence
routes those tensor classes to refined-track evidence expectations; coverage policy
(all 506 matrices in scope) is unchanged either way. This gate adjudicates the
VBQ/Bonsai-versus-Ternary-Mamba/NVFP4 conflict with measurement instead of belief.

**Gates:** protocol + threshold committed before execution; probe receipts in the
evidence directory; routing decision recorded before Stage-8 spend.

### WS-E — Evaluation adoptions

Bonsai-27B-ternary (`prism-ml/Ternary-Bonsai-27B-gguf`) becomes a required Stage-8
baseline row, reproduced in plan 0043's sense (local artifact digest + local eval
outputs). All thinking-mode evals adopt **score@budget** reporting (accuracy +
cap-rate + stated token budget); task thresholds unchanged.

### WS-F — Interop and transport

Q2_0-g64 import (`tritium-format/src/q2_0.rs`, modeled on the TQ2_0 row functions);
CompactV1 P=1 → Q2_0-g64 export behind a profile-compatibility check (exact: trits
unchanged, each G128 scale duplicated into two G64 groups); ANS/rANS seekable outer
transport with expanded-bytes accounting unchanged. Rationale: every published
quality result becomes runnable in the ecosystem's standardizing runtime.

**Gates:** round-trips bit-exact; a Q2_0-g64 artifact decodes token-identical to its
TQ2_0 sibling on the acceptance prompts.

### WS-G — Documentation revisions

ADR 0024: target ratio 2:4 → **6:8 via SlideSparse** sliding-window decomposition;
the no-kernels-before-a-trained-checkpoint gate stands verbatim; plan-0039's trainer
hook targets 6:8 masks. ADR 0021/0032: drafter objective becomes per-position KL to
target weighted by measured target margins (acceptance-theory certificates); tree
width m selected from measured margins instead of grid search; the novelty claim is
reframed to **tree-verified** ternary spec-decode (Neutrino-1 took single-path
argmax-match).

## Preregistration mechanics (the amendment this ADR authorizes)

One dated amendment section in plan 0043, landing **before any rung-2 scoring run**:

- Stage-7 grid: `solver ablation` axis gains `+softened-relay-basin` and
  `+modulated-basin`; the refined-track mechanism set gains `hestia-relaxation`
  (vs `ste-soft`) as an A/B; the `curvature` axis is unchanged (Kronecker variants
  already preregistered) — WS-A changes capture *cost*, not the artifact.
- New pre-Stage-8 GDN-sensitivity preflight gate (WS-D protocol + threshold).
- Stage-8 baseline matrix gains the Bonsai-27B-ternary row.
- Reporting schema gains score@budget fields.
- Explicit sentence: successive-halving protocol, freeze-gate thresholds, PTQ/refined
  separation, token caps, and spend policy are unchanged; anything touching those
  requires a new preregistration.

## Sequencing

```
Docs first (authorize):   ADR 0035 → plan 0054 → 0043 amendment → ADR 0024/0021/0032 notes
27B critical path:        A1 (cost baseline) → A2/A3 (shared-forward + golden) → A4 (forecast)
                          D1 (freeze protocol) → D2 (probes + routing)          → Stage 8
Rung-2 grid (pre-Stage-7 scoring):  B1/B2 → B3 → B4 bookkeeping
Refined track (pre-Stage-9):        C4 (oracle) → C1 → C2 → C3
Parallel low-risk:                  E1/E2, F1 → F2, F3, WS-G
```

## Consequences

- **Positive:** the campaign inherits the frontier's strongest mechanisms without
  reopening a single frozen threshold; the 27B capture cost gets measured and
  attacked before the spend request; the riskiest tensor class gets a
  measurement-backed routing decision; Gate C is strengthened (both-input
  gradchecks); published artifacts become ecosystem-runnable.
- **Negative / risk:** the SALT V2 paper's novelty must now be argued as the
  combination (output-aware additive planes + exact joint solve + allocation +
  native kernels), since individual mechanisms are shared with CAT-Q/HESTIA — the
  ablation table is the defense, and B4's bookkeeping makes it auditable. The
  shared-forward capture's byte-identity claim depends on the dyadic-reduction
  design holding under the multi-writer path; if it fails, WS-A falls back to
  per-tensor capture with the measured cost and the 27B forecast reflects it.
  Portable-manifest V3 obligates every backend (CPU, CUDA) plus vectors — the
  known price of a new op, paid once.
- **Sequencing risk:** B3 (window optimizer) is the largest new code surface;
  it is deliberately scoped to scale-refits-only so a slip degrades to "grid value
  absent from the bracket," not a campaign blocker.

## Definition of done

- [ ] Plan 0054 committed; 0043 amendment landed before any rung-2 scoring run.
- [ ] WS-A: cost receipt at 1.7B (per-tensor and shared-forward), byte-identity
      golden green, Stage-2 perturbation gate green, 27B forecast row filled.
- [ ] WS-B: relay goldens vs BitTern reference; basins + window optimizer in the
      grid; recipe-digest + BASELINES lockstep updated; property tests green.
- [ ] WS-C: gradcheck (both inputs) green; portable V3 conformance CPU+CUDA;
      refined-track A/B run on rung 2 recorded (win, tie, or loss).
- [ ] WS-D: protocol frozen before probes; probe receipts + routing decision
      recorded before Stage-8 spend.
- [ ] WS-E: Bonsai-27B baseline reproduced under the 0043 harness; score@budget in
      the report schema.
- [ ] WS-F: Q2_0-g64 round-trip + token-identity gates green.
- [ ] WS-G: ADR 0024/0021/0032 amendment notes committed.
- [ ] U1–U9 and the plan-0043 per-commit review protocol green throughout.
