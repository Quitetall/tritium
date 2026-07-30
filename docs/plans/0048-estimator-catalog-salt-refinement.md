# 0048 — Estimator catalog and SALT refinement

Status: **IN PROGRESS** (2026-07-20)

- **Decision:** [ADR 0033](../adr/0033-v11-full-public-release.md)
- **Parent:** [plan 0044](./0044-v11-full-public-release.md)
- **Parallel dependency:** [plan 0043](./0043-salt-v2-sota-campaign.md)

## Goal

Make Tritium the authoritative implementation surface for ternary estimators
and production additive PTQ/refinement. Every algorithm has a hard exportable
forward, explicit backward policy, versioned identity, coverage/state
accounting and matched-byte evaluation.

## Public seams under test

- `tritium.torch.create_estimator`
- `tritium.torch.register_estimator`
- `tritium.torch.registered_estimators`
- `AnnealedSTE`, `LSQEstimator`, `TWNEstimator`, `TTQEstimator`, and
  `SparseTernaryEstimator`
- `TernaryConfig.qat(estimator=...)`
- `RefinementConfig.scale_only(...)` and `RefinementConfig.hard_pv(...)`
- `tritium.torch.refine`, `RefinementResult`, and artifact ancestry

## Step 1 — differentiable estimator catalog and plugins

- Built-in AbsMean/SALT STE remain the conformance base.
- Add annealed STE, LSQ, TWN, TTQ and explicit sparse-ternary estimators.
- Hard detached forward values must decode exactly from `{-1,0,+1}` trits and
  finite nonnegative scales; only backward surrogates may differ.
- Learned scale/threshold state is an ordinary `Parameter`, included exactly
  once in coverage and safetensors state.
- Modules sharing a latent master share one estimator instance.
- External factories register explicitly; duplicate registration fails closed.

Gate: literal hard decode, finite gradients for masters and estimator
parameters, config conversion, tied estimator identity, state/HF round-trip,
coverage and duplicate-plugin tests.

## Step 2 — production SALT reconstruction

- [x] Consume guided-Fisher/input-Hessian/forward-KL Kronecker factors without
  row-wise dense expansion; prove bit-exact solver and tensor-master parity with
  the materialized metric.
- [x] Canonically persist and bounded-reopen those per-tensor factors with
  source/evidence identity and adversarial corruption/count rejection.
- [x] Connect complete Qwen evidence namespaces to admitted checkpoint matrices
  and the resumable pure-PTQ master campaign with campaign-wide token-stream
  consistency.
- [x] Preserve standard SwiGLU QKV bias and conventional per-head Q/K RMSNorm
  constants through semantic identity, exact dense reconstruction, and
  replayable resident/packed CUDA training graphs. Qwen3.6 hybrid DeltaNet and
  gated-attention training remain a separate open integration.
- [x] Add a bounded-memory block/sliding-window evaluator with exact teacher and
  student stream identities, final-logit teacher CE/KL, complete deterministic
  multi-start selection and a strict canonical `TSV2OUT` v2 receipt. Legacy v1
  remains strictly inspectable but cannot claim runtime binding.
  - Edit `crates/tritium-quantize/src/salt_v2_output.rs` for the public frozen
    schedule/spec, streamed accumulator, metric calculation and deterministic
    restart-selection seams.
  - Edit `crates/tritium-quantize/src/salt_v2_output/codec.rs` for the bounded,
    canonical receipt encoder and fail-closed strict reopen path.
  - Edit `crates/tritium-quantize/tests/output_reconstruction.rs` for literal
    metric, traversal, teacher-drift, corruption, allocation-amplification and
    unreachable-state coverage.
  - Expected output: `cargo test -p tritium-quantize --test
    output_reconstruction` passes; `cargo clippy -p
    tritium-quantize --all-targets -- -D warnings` exits zero.
- [x] Reject activation-cache and token-stream drift across every reopened Qwen
  curvature record before planning the immutable master campaign. Expected
  output: `cargo test -p tritium-salt
  record_slot_mismatches_fail_before_source_widening` passes.
- [x] Bind the selected output candidate's final-logit evidence to its immutable
  tensor-master set through campaign-owned execution, not a caller-provided
  candidate label.
  - First vertical slice: edit
    `crates/tritium-nn/src/model/qwen35_execution.rs` to stream final logits
    directly from `Qwen35SaltV2LanguageMtpModel`, bind exact loaded-package and
    self-asserted backend identities, and emit an explicitly non-admissible
    `TSQ35EX` v1 transcript with block coverage absent and an untrusted-backend
    evidence class. Reopen must re-execute; no raw-logit constructor exists.
    A hostile caller backend may create only this lower transcript, never an
    admitted campaign receipt.
    Expected output: `cargo test -p tritium-nn
    complete_bundle_loads_and_executes_language` passes.
  - [x] Second vertical slice: the SALT package-admission capability constructs
    a built-in CPU/CUDA backend internally, reconstructs the authoritative
    preserved companion, strictly loads the selected package, and emits
    canonical `TSQ36EX` v1 evidence binding completion/campaign/admission/
    selection/source IDs, ordered master-set ID, exact package/preserved/
    manifest/config IDs, backend capabilities and physical identity, tokens,
    final logits and explicit scope/count coverage. Reopen re-executes; no
    caller backend, raw-logit or lower-transcript promotion seam exists.
    Forged lineage, consistently relabeled preserved substitution, changed
    tokens and corrupt replay evidence fail closed. Expected output: `cargo test
    -p tritium-salt
    admitted_qwen_execution_binds_campaign_packages_backend_tokens_and_outputs`
    passes. This structural test uses an explicitly test-only small, non-pinned
    Qwen configuration; successful production construction through the pinned
    Qwen3.6-27B loader remains part of the checkpoint-scale evidence item below
    and is not yet qualified.
  - [x] Bridge foundation: move ordered final-logit hashing into a shared format
    contract used byte-for-byte by both Qwen execution and output reconstruction.
    Extend each canonical `TSV2OUT` v2 candidate with its runtime-comparable
    final-logit digest and exact batch/value counts while retaining the distinct
    aggregate block-plus-logit student digest. V1 remains strictly inspectable
    but bridge-ineligible because absent runtime evidence is never synthesized.
    Freeze both evidence-domain and complete v2 goldens; strict reopen rejects
    malformed counters.
  - [x] Third vertical slice: canonical `TSQ36OB` v1 evidence strictly reopens
    `TSV2OUT` v2, derives the admissible candidate identity from immutable
    completion/campaign/admission/selection/source/master/package/preserved/
    profile/spec lineage, revalidates the current `TSQ36EX` authority and admits
    final logits only when digest plus exact batch/value counts agree. Candidate
    relabeling, changed logits, token drift, corrupt bytes and bridge-ineligible
    legacy v1 receipts fail closed. The frozen receipt reports block coverage
    absent and never treats aggregate student-output identity as a runtime-logit
    identity. Persisted bytes strictly reopen only after canonical decoding and
    a fresh bind against current admission/execution authority. Expected output:
    `cargo test -p tritium-salt
    admitted_qwen_execution_binds_campaign_packages_backend_tokens_and_outputs`
    and `cargo test -p tritium-salt
    canonical_final_logits_binding_layout_is_frozen` pass.
  - Full output-aware campaign admission remains open until the runtime seals
    the exact block/window outputs represented by the selected candidate; this
    final-logit bridge must not be presented as block reconstruction proof.
- Execute the bound output-reconstruction workflow on checkpoint-scale evidence.
- [x] Provide the real calibration producer an immutable resumable `S2KF`
  publication namespace: bounded records install atomically without overwrite,
  crash staging stays inside a managed reserved directory and outside the exact
  record namespace, byte-identical retries are idempotent, and conflicts and
  replaced roots fail closed where stable directory identity is available.
  Supported platforms durably sync both publication and staging cleanup;
  unsupported directory-sync platforms reject publication instead of claiming
  durability. Collection of the actual 506 checkpoint records remains open and
  this storage seam is not empirical proof.
- [x] Provide bounded grouped-curvature production: source-bound activation and
  output-factor batches update atomically, G128 input groups and output-row
  Fisher/KL factors preserve canonical sample selection, out-of-order worker
  shards merge to byte-identical evidence, fallible transaction copies preserve
  allocation errors, canonical dyadic state is observable and capped at 64
  segments per accumulator, and exact record geometry is preflighted before
  producer allocation. Consuming finalization installs directly into the durable
  namespace. Runtime capture from the pinned model and collection of 506 real
  records remain open.
- [x] Add pinned-Qwen capture orchestration that exposes canonical tensor
  ordinal, name, geometry, scope, role, and source identity to one runtime
  callback at a time; resumes strictly valid records without replay; rejects
  hostile or stale evidence before overwrite; publishes no active tensor after
  callback failure; validates the exact namespace; and derives an ordered
  evidence-set digest. The PyTorch/CUDA hook and empirical 506-record run remain
  open.
- [x] Expose that same orchestration through a stateful abi3 session: reopening
  preflights every present record before returning a missing task, repeated
  `next_request()` calls are idempotent, advancement requires strict durable
  acceptance of the pending ordinal, drop/reopen resumes without replay, and
  finalization freshly revalidates all 506 records before returning the native
  ordered evidence-set digest. The original callback collector now uses this
  state machine rather than maintaining a second scheduler.
- [x] Expose bounded abi3/PyTorch grouped-evidence ingestion: tensor batches are
  shape-checked and converted to canonical f32le/f64le buffers, token masks are
  strictly boolean, native accumulation releases the GIL, failed parsing or
  accumulation leaves prior state unchanged, and explicit finish atomically
  publishes one record.
- [x] Add the model-aware one-projection capture pass: forward-call order is
  retained for reused modules; input-Gram, explicitly normalized model-loss
  guided-Fisher, and a stateless single-probe softmax-Fisher Rademacher estimate
  use only output VJPs and never allocate parameter gradients. The exact
  objective ID is domain-bound into the activation-cache provenance, probe
  identity uses the global sample ordinal rather than batch boundaries,
  activation snapshots and forward-KL factor workspace have explicit byte
  ceilings, and sharded-device masks move only through bounded chunks.
  Attention masks, or shifted non-ignored labels for the causal-loss
  convention, bind the selected sample set; partial failures abort unpublished
  state; and completed state survives only retryable publication/resource
  failures. The high-level PyTorch model/module resolver and calibration-replay
  loop over the native pinned-Qwen session remain open.
- Keep calibration, reconstruction and validation datasets source-bound and
  disjoint where the recipe requires.

## Step 3 — separate refinement tracks

Status: **IN PROGRESS** — `convert(prepared_qat)` now emits a separately typed
inference-only `QatHardResult`, preflights every unique projection before graph
mutation, packs tied Linear/Embedding masters once and binds source-checkpoint,
recipe and compact-state identities. Built-in hard forwards now decode exact
stored f16 scales. TTQ's asymmetric positive/negative values are represented as
two honest row-scale planes and therefore require `planes=2`; the prior
per-element-scale pseudo-plane could not enter SALT. Generic QAT-hard state now
exports through the shared `export` seam into a canonical SHA-256-ledgered
safetensors bundle, strict-reloads evidence without a source model, or binds to
an explicit dense/QAT shell with tied compact storage and exact hard-state
parity. Rehashed ancestry, latent masters, estimator state, invalid B3/scales,
unknown files, corrupt payloads and wrong shells fail closed. Convolution
lowering and public HF checkpoint registration remain open; none are relabeled
as PTQ. The post-PTQ refinement lifecycle is implemented: versioned scale-only
and hard-PV/S34 recipes consume an immutable PTQ/refinement parent, use bounded
row-wise second-order fitting, optimize hard candidates against held-out
teacher loss, and emit a strict child conversion plus complete ancestry.
Training/validation batches are individually hashed, aggregate-bound, and
cross-split overlap is rejected. Strict reload and atomic export rehash the
parent, evidence, child algorithm/recipe, S34 topology, payloads, and directory
publication. Every deployment-aligned G128 child additionally carries a native
seek-backed SALT package bound to the child conversion identity; deliberately
unaligned research fixtures retain the hard conversion but cannot claim a SALT
package.

- [x] Remove refinement from `TernaryConfig.ptq`; reject ambiguous legacy non-none
  refinement values instead of silently migrating them.
- [x] Implement `refine(parent, teacher, training, validation, config, work_dir)`.
  Scale-only, true alternating hard PV and S34 refinement are distinct child
  work identities and result discriminants.
- [x] Bind every child to its immediate parent and complete ancestry. Scale-only
  freezes trits/allocation; hard PV may change assignments only under its
  declared structure and frozen nested-prefix constraints.
- [x] Export the G128 hard package and prove reload parity; no latent residual may
  enter a PTQ claim. QAT checkpoint, QAT export, PTQ, scale-only and hard-PV
  artifacts remain separately typed. Unaligned research-only children are
  explicitly non-package artifacts.
- [x] Implement a fail-closed matched-physical-byte execution worker for
  RTN/AbsMean, GPTQ/AWQ-style second-order baselines and frozen SALT variants.
  It runs the exact retained argv recipes without a shell, measures thirty
  same-device evaluation samples, hashes every physical artifact, and rejects
  byte mismatch, identity drift, cross-device samples or artifact mutation.
- Execute that worker against final refined Qwen3.6 candidate and frozen
  evaluation ledger.

Release admission now has three separate fail-closed contracts.
`tritium.estimator-catalog-qualification.v1` binds the exact candidate wheel and
all seven built-ins, hard trits/scales, gradients/state/ties/coverage and
external-plugin rejection behavior. The installed-wheel worker emits the
retained `tritium.estimator-catalog-execution.v1` trace; the qualifier binds its
wheel hash, CPU environment and exact results into the release receipt.
`tritium.refinement-qualification.v1`
requires disjoint source-bound splits and distinct scale-only, hard-PV and S34
children with exact ancestry, held-out hard candidates, G128 native SALT
packages, strict reload and no latent residuals; it must descend from the
flagship NearLossless PTQ artifact and the estimator receipt.
`tritium.baseline-ablation-qualification.v1` freezes RTN/AbsMean,
GPTQ/AWQ-style, SALT V1 and three mechanism ablations, recomputes matched-byte
claim eligibility, and binds the exact refined model/evaluation lineage. These
contracts and producers are implemented. Raw ablation evidence now retains
each publishable recipe body, build/evaluation argv, artifact path and physical
artifact SHA-256; verifier recomputes recipe identities instead of trusting
claimed digests. Refinement receipts are derived from retained
dataset/candidate ledgers, hard-structure deltas, package identities, reload
samples and validation losses. Ablation receipts are derived from retained
matched-byte timing/residency samples and frozen recipe identities.
Executing them against the final candidate wheel and checkpoint-scale campaign
artifacts remains open.

## Verification

```bash
PYTHONPATH=crates/tritium-py/python pytest -q crates/tritium-py/tests
python -m unittest scripts.tests.test_qualify_estimator_catalog
python -m unittest scripts.tests.test_qualify_refinement_campaign
python -m unittest scripts.tests.test_run_baseline_ablation
python -m unittest scripts.tests.test_qualify_baseline_ablation
python -m unittest scripts.tests.test_verify_estimator_refinement_receipt
python -m compileall -q crates/tritium-py/python/tritium
cargo fmt --check
git diff --check
```

## Done criterion

All catalog/plugin and production reconstruction/refinement gates pass; hard
artifact receipts feed plan 0043 without relabeling refined results as PTQ.
