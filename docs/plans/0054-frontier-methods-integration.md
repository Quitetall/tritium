# 0054 — Frontier-methods integration (execute ADR 0035)

Status: **PLANNED** (2026-07-30)

- **Decision:** [ADR 0035](../adr/0035-frontier-methods-integration.md) (executing
  [ADR 0034](../adr/0034-next-gen-ternary-research.md)'s adoption decisions)
- **Constraint:** [plan 0043](./0043-salt-v2-sota-campaign.md)'s frozen thresholds,
  successive-halving protocol, token caps, PTQ/refined separation, and spend policy
  are untouched. This plan adds candidates, one preflight gate, baselines, and
  interop — nothing else.
- **User decisions bound:** native Rust ports (Python = oracle only); full sweep;
  grid enlarged by dated amendment.
- **Spend:** local hardware only, inherited from plan 0043. Nothing here authorizes
  paid compute.

## Workstream A — Curvature capture cost (27B critical path)

The K-FAC pipeline is shipped (S2KF records, PyO3/PyTorch capture, resumable
session). The open cost problem: **one calibration replay per tensor** — 506 replays
at 27B. Lever: shared-forward capture.

### A1. Cost baseline at rung 2 (SmolLM2-1.7B)

- Run the existing per-tensor capture flow (`torch/ptq.py::capture_kronecker_module`
  driven per-tensor, session-style) over the 1.7B tensor set on the frozen
  128-sequence rung-1 prefix first (smoke), then the full frozen pack.
- Record into a receipt: wall time total + per tensor class, peak host/device bytes,
  artifact bytes, replay count. This is the measured denominator plan 0043's
  cost-forecast table requires (no extrapolation).
- Files: `crates/tritium-py/python/tritium/torch/ptq.py` (timing hooks only, no
  behavior change), receipt schema beside the existing capture receipts.

### A2. Shared-forward capture session

One forward pass over the calibration set feeds the S2KF builders of **every
projection consuming the same layer input** (q/k/v share the attention-norm output;
gate/up share the FFN-norm output; per block up to 7 linears collapse to ~3 distinct
input streams). Design:

- New session mode beside `Qwen36PtqEvidenceCaptureSession`
  (`crates/tritium-salt/src/qwen36_tensor_work/ptq_driver/capture.rs`): a
  *layer-group* request that names N tensors sharing one input stream; per-tensor
  S2KF specs and writers are unchanged.
- Multi-writer variant in `crates/tritium-py/src/kronecker.rs`
  (`KroneckerEvidenceBuilder` is already per-tensor — the new piece is a
  `SharedForwardCapture` that fans one activation batch into N builders and N
  output-factor VJPs from one forward) and `torch/ptq.py` (one forward hook per
  input stream instead of per module).
- **Invariants preserved (hard constraints from exploration):** S2KF payload layout
  and G128 grouping unchanged; canonical dyadic reduction keyed on global sample
  ordinals (batch boundaries/shard order stay outside identity); transactional
  atomic batch semantics per builder; `CurvatureSourceId` triple-digest binding;
  PSD-after-damping with explicit failure, never implicit diagonal fallback;
  frozen `objective_id` strings unchanged.
- Guided-Fisher note: output-factor VJPs for N tensors from one forward require N
  output-grad requests against one graph — retain the graph across the N VJP calls
  or batch them; parameter `.grad` buffers stay unallocated either way.

### A3. Equivalence golden

Shared-forward records must be **byte-identical** to per-tensor records for the same
frozen corpus and seeds. The dyadic-reduction design makes this provable; the golden
makes it enforced. Test beside
`crates/tritium-quantize/tests/kronecker_evidence_producer.rs` +
`crates/tritium-py/tests/test_kronecker_capture.py`.

### A4. Cost report → 27B forecast

A1-vs-A2 measured ratio at 1.7B fills the plan-0043 spend-forecast row for the 27B
capture (with the honest caveat that 27B layer geometry differs; the forecast states
the scaling assumption explicitly).

### A1/A4 measured results (2026-07-30, SmolLM2-1.7B, RTX 4090)

Receipt: `docs/receipts-ws-a1-cost-baseline-17b.json`; harness:
`tools/ws_a1_cost_baseline.py`. 24 tensors (8 layers x q/k/v), 64x512 = 32,768
calibration tokens, input-hessian, f32.

| path | replays | wall | peak dev bytes | artifact bytes |
|---|---:|---:|---:|---:|
| per-tensor | 24 | 605.3 s | 11.85 GB | 50,732,380 |
| shared-forward (per-layer groups) | 8 | 516.0 s | 12.06 GB | 50,732,380 |

- **Byte-identity: TRUE at 24 tensors** (A3 gate holds on the real model).
- **Replay reduction 3.0x** (q/k/v per attention stream) — the >=3x gate is MET
  exactly; wall speedup is **1.17x**, because at this scale the forward is only
  ~22% of wall (~5.6 s/replay); the replay-invariant Gram accumulation + writer
  path is ~78%. The honest A4 statement: shared-forward eliminates the
  replay-scaling term; the accumulation term is what the 27B forecast must
  price, and it does not shrink with grouping.
- **Measured constraint:** an all-24-in-one capture exceeds the bounded-snapshot
  budget (needs 302 MB vs the 256 MiB `max_capture_bytes` default) and fails
  closed — empirical confirmation that the grouping planner's residency budget
  is load-bearing. Production shape = per-layer input-stream groups.
- 27B forecast note (scaling assumption stated per plan-0043 rules): at 27B the
  forward is far more expensive per replay while the Gram cost per tensor grows
  ~linearly in columns; the forward share therefore RISES with model size, so
  the 3x replay reduction is worth strictly more at 27B than the 1.17x measured
  here. A same-harness run on a sliced 27B layer set must replace this note
  before any spend request.

**Gates (A):** A3 byte-identity green; A2 ≥3× measured replay reduction at 1.7B
(expected more from per-block sharing; report the honest number regardless);
Stage-2 perturbation-ordering gate re-run green on shared-forward artifacts; A4 row
committed. Failure mode: if byte-identity cannot hold under the multi-writer path,
record why, fall back to per-tensor capture, and let A1's measured cost drive the
27B forecast.

## Workstream B — CAT-Q mechanisms into the PTQ track

### B1. Softened-relay initialization basin

- New restart basin in `deterministic_initial_scales`
  (`crates/tritium-quantize/src/salt_v2.rs:791`): continuous soft fit of per-plane
  scales under the two-sided tanh relay
  `f(W;s,Δ) = [tanh(s(W−Δ)) + tanh(s(W+Δ))]/[2·tanh(s)]`, sharpness annealed from
  s₀=30 (CAT-Q defaults), coordinate-descent or fixed small step count
  (deterministic, seedless — basin code must be bit-repeatable), then **projected
  through the existing exact E/M solver** (`optimize_start`). The
  accept-only-if-improves guard makes the basin strictly safe.
- The relay math gets a small pure module (`salt_v2_relay.rs` or inline mod) with
  goldens against the CAT-Q reference implementation
  ([IntelChina-AI/BitTern](https://github.com/IntelChina-AI/BitTern)) on tiny
  matrices (tolerance goldens — their torch f32 vs our f32).

### B2. Learnable modulation as basin parameters

- δμ (mean shift), δα (scale refinement), δΔ (threshold refinement) exist **only
  inside the soft fit**: they shape the basin's starting point; the projected result
  is plain trits + non-negative f16 scales. Nothing stored → ADR 0028's zero-point
  ban satisfied with zero format/runtime impact. (The `SaltV2Transform` seam remains
  the documented alternative if a *stored* transform is ever justified — that is a
  new preregistration, out of scope here.)
- δα in the exact solver is `solve_scales`' job already; the basin's δα only seeds it.

### B3. Sliding-window output-reconstruction optimizer

The missing optimizer for the shipped measurement layer:

- Candidate generator behind the declared seam
  `SaltV2ExternalStage::BlockOutputReconstruction` +
  `SaltV2ModelStageDriver::run_stage`
  (`crates/tritium-quantize/src/salt_v2_model.rs:646/671`).
- Per window from `OutputReconstructionSchedule::SlidingWindows`
  (`salt_v2_output.rs`): load the window's cached activations (`ActivationCache`),
  generate **scale-refit candidates against fixed trits** for the window's tensors
  (deterministic multi-start, reusing `solve_scales` with the window-output
  objective), score each candidate via `OutputReconstructionAccumulator::observe`,
  select via `select_output_reconstruction`. Trits never change here → the
  bounded-memory one-tensor master-fit invariant survives; window residency is
  bounded by `window_size` (2–3 layers).
- **Un-wall block outputs:** in
  `crates/tritium-salt/src/qwen36_tensor_work/execution.rs`, set
  `BLOCK_OUTPUT_COVERAGE` for transcripts that declare window coverage and lift the
  `has_block_outputs()` rejection (:404/:765/:779) for exactly that declared case;
  undeclared block outputs still fail closed.

### B4. Bookkeeping (mandatory, every variant)

- Extend `recipe_digest` (`salt_v2_model.rs:4055`) with every new `SaltV2Config`
  field (basin toggles, relay constants, window parameters) — otherwise two variants
  collide on recipe ID.
- Add ablation rows to the frozen `BASELINES` tuple in **both**
  `scripts/run-baseline-ablation.py` and
  `scripts/verify-estimator-refinement-receipt.py` (they verify in lockstep; a
  one-sided edit fails closed).
- E/M-non-increasing, P-monotonicity, and bit-identical-rerun property tests stay
  green; `NonMonotoneCandidate` rejection unchanged.

**Gates (B):** relay goldens vs BitTern; each mechanism enters the amended Stage-7
grid and must survive successive halving (thresholds unchanged); B4 lockstep
verified by the release-evidence scripts; property tests green.

## Workstream C — HESTIA into the refined track

### C1. `hestia_relax` CPU op

- New `crates/tritium-train/src/ops/hestia.rs`:
  - forward: `w̃ = Σ_q q·π_τ(q|w)`, `π_τ(q|w) = exp(−(w/γ−q)²/τ) / Σ_k exp(−(w/γ−k)²/τ)`,
    `q ∈ {−1,0,+1}`, γ = per-row AbsMean (reuse `absmean_scale_per_row`); output
    scaled back by γ.
  - vjp: analytic in both `wf` and `τ` (smooth everywhere).
- Tape method `hestia_relax(&mut self, wf, tau, rows, cols)` beside `lsq_ste`
  (`crates/tritium-train/src/tape.rs`, two-input pattern).
- Gradcheck `crates/tritium-train/tests/gradcheck_hestia.rs` copying
  `gradcheck_lsq.rs` — **both** inputs finite-differenced (no kink placement
  needed; this is a Gate-C strengthening worth a comment).

### C2. `TempSchedule` + per-tensor sensitivity

- `crates/tritium-train/src/temp.rs` (or into `lr.rs`'s module): pure
  `TempSchedule::new(tau0, tau_floor, total_steps)` with exponential decay,
  `tau(step)`; per-tensor scaling `τ_i(t) = τ̄(t)·e^(α·s_i)` with `s_i` derived from
  S2KF evidence (input-Gram trace × output-Fisher mean, normalized per HESTIA's
  standardized-sigmoid recipe). No new estimator: WS-A's artifact is the input.
- Policy wiring: `TemperatureSchedule` sibling of `BypassSchedule` in
  `crates/tritium-train/src/salt_v2_recovery.rs`; validation: τ reaches `tau_floor`
  at or before the `RecoverySchedule` Hard boundary (80%); Hard phase is untouched
  (hard exported trits + narrowed scales, `convert_qat_hard` contract unchanged).

### C3. CUDA + portable manifest V3

- Kernels `hestia_relax_forward` / `hestia_relax_backward_{w,tau}` in
  `crates/tritium-cuda/kernels/train_grad.cu`; host wrappers in
  `cuda/backend.rs`; names in `cuda/consts.rs`.
- Soft phase consumes the expectation via **dense `matmul`** on the DeviceTape (the
  packed `salt_matmul` path cannot represent soft weights; soft phase is
  training-time only, so no serving-kernel work).
- Portable manifest **V3**: `OPERATIONS_V3` adds `graph.hestia_relax`
  (`crates/tritium-spec/src/training.rs`), `data/training/v3/manifest.json` +
  `vectors/v3.json` (bit-exact f32 bit-pattern vectors incl. forward, vjp-w,
  vjp-tau, and error cases), CPU implementation in
  `crates/tritium-train/src/portable.rs`, CUDA in
  `crates/tritium-cuda/src/train/portable.rs`; conformance tests extended
  (`training_vectors.rs`, `portable_training.rs`, `portable_cpu.rs`).

### C4. Python oracle (first, cheap)

- `hestia` estimator in `crates/tritium-py/python/tritium/torch/estimators.py`
  registered beside `annealed-ste`, with its **own projection helper** (the
  `_projection` hard-forward contract does not admit a soft-expectation forward);
  flagged non-exportable while τ > floor. Used to generate the golden tensors for
  C1/C3 vectors; never the product path (native-Rust decision).

**Gates (C):** gradcheck both inputs; V3 conformance CPU+CUDA green; refined-track
A/B (hestia-relaxation vs ste-soft) on the rung-2 recipe under the unchanged
selection threshold — win, tie, or loss recorded.

## Workstream D — GDN-sensitivity preflight gate (pre-Stage-8)

### D1. Freeze the probe protocol (in the 0043 amendment, before running)

- Probe set: N DeltaNet-block matrices vs N full-attention-block matrices from the
  pinned checkpoint (N and the exact tensor list frozen in the amendment), each
  PTQ-ternarized at matched bpw with the frozen recipe while the rest of the model
  stays at source precision.
- Binding metric: **divergence growth along the recurrence** — roll the
  content-bound Qwen3.5-family reference adapter (`tritium-nn`) over frozen
  calibration sequences, record state/output divergence as a function of
  depth-in-sequence for probed vs control blocks. Per-layer weight/output MSE is
  recorded but explicitly non-binding (Ternary Mamba proxy-failure finding).
- Threshold: frozen in the amendment before any probe executes.

### D2. Routing rule (frozen with the protocol)

If GDN divergence exceeds the threshold at matched bpw: the PTQ track records the
negative honestly (plan 0043: "no is a valid result") and DeltaNet tensor classes
carry refined-track evidence expectations. Coverage policy — all 506 matrices in
scope — is unchanged either way. Probe receipts land in the evidence directory;
the routing decision is recorded before Stage-8 spend.

## Workstream E — Evaluation adoptions

### E1. Bonsai-27B-ternary baseline row

- Add `prism-ml/Ternary-Bonsai-27B-gguf` to the Stage-8 baseline matrix (amendment):
  pinned artifact digest, actual bytes, evaluated under the 0043 frozen harness.
  Reproduced-baseline rules apply (local artifact + local eval outputs; PrismML fork
  runtime documented as its serving path). F1's Q2_0 import enables the
  Tritium-side cross-check.

### E2. score@budget reporting

- Extend the frozen-harness report schema with `cap_rate` and `token_budget` fields
  alongside accuracy for thinking-mode evals (report-side only; task thresholds and
  the six-task aggregate definition unchanged).

## Workstream F — Interop + transport

### F1. Q2_0 g64 import

- `crates/tritium-format/src/q2_0.rs`: `pack_q2_0_row` / `unpack_q2_0_row`
  (group-64, f16 scale per group, 2-bit codes `{0,1,2}` → `{−1,0,+1}`), modeled on
  `rows.rs`'s TQ2_0 pair; block constants; proptest round-trip; re-export in
  `lib.rs`. Loader wiring so official llama.cpp Q2_0 artifacts run through the
  existing GGUF reader path (new tensor type ID, no byte sniffing — explicit type).

### F2. CompactV1 P=1 → Q2_0 g64 export

- Exact for P=1 profiles: trits unchanged; each G128 scale duplicates into two G64
  groups. Behind a profile-compatibility check that fails closed for P>1 or
  non-uniform plane maps. Bit-exact round-trip gate (export → import →
  reconstruction identical).

### F3. Entropy-coded outer transport

- ANS/rANS seekable outer container per plan 0043's existing scoping: transport
  only; expanded fixed-codec bytes remain the accounted resident weights; no bpw
  claim changes. (Field-proven by Neutrino's ~55% lossless ternary-lane
  compression; Tritium's zero fractions will differ — measure, don't assume.)

**Gates (F):** F1/F2 round-trips bit-exact; a Q2_0-g64 artifact decodes
token-identical to its TQ2_0 sibling on the acceptance prompt set; F3 accounting
rules verified by the physical-size report tests.

## Workstream G — Documentation revisions (no code)

- ADR 0024 amendment note: target ratio 2:4 → 6:8 via SlideSparse sliding-window
  decomposition (ICML 2026); checkpoint gate stands; plan-0039 trainer hook spec
  updated to 6:8 masks.
- ADR 0021 + ADR 0032 revision notes: drafter training objective = per-position KL
  to target weighted by measured target margins (acceptance-theory certificates,
  arXiv 2606.30265); tree width m selected from measured margins via the ADR 0032
  cost model; novelty claim reframed as **tree-verified** ternary spec-decode.

## Sequencing

```
Docs (authorize first):  ADR 0035 ✓ → this plan → 0043 amendment → G notes
27B critical path:       A1 → A2 → A3 → A4 ─┐
                         D1 ────────→ D2 ────┴→ Stage 8 (existing plan-0043 flow)
Rung-2 grid entry:       B1 → B2 → B3 → B4  (all before Stage-7 scoring)
Refined track:           C4 → C1 → C2 → C3  (before Stage-9 refinement selection)
Parallel:                E1, E2, F1 → F2, F3
```

Estimated effort (honest ranges, local hardware): A 4–7d (A2 is the bulk), B 6–10d
(B3 dominates), C 5–8d (V3 manifest + CUDA is half), D 2–3d + probe runtime,
E 2–4d (E1 needs the PrismML fork built locally), F 3–5d, G 1d.

## Verification protocol

Per-commit: plan 0043's protocol verbatim (narrow tests, `cargo fmt --check`,
clippy `-D warnings` on affected crates, CPU/CUDA parity + sanitizer gates where
touched, required commit reviewer with verify-before-fix, FP skips recorded).

End-to-end smoke before any rung-2 scoring: rung-1 (SmolLM2-135M) run of the full
amended pipeline — shared-forward capture → fit (new basins available) → allocate →
package → eval — green.

## Out of scope

Reopening any frozen 0043 threshold; Stage 9/10 execution; BLUT drafter training
(G documents the objective only); SlideSparse 6:8 kernels (ADR 0024's checkpoint
gate); vision/multimodal; any paid compute.
