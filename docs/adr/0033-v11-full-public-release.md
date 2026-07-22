# ADR 0033 — Tritium v1.1 full public release: ternary research platform

Status: **ACCEPTED** (2026-07-22; installed-observability evidence amendment)

- **Decider:** Brian Lam
- **Research cutoff:** 2026-07-20, inclusive
- **Execution:** [plan 0044](../plans/0044-v11-full-public-release.md)
- **Relates:** preserves the immutable v1.0 infrastructure milestone in
  [ADR 0012](./0012-v100-release.md); retains the Qwen3.6-27B language-plus-MTP
  public-proof gate in amended [ADR 0020](./0020-v1x-salt-distillation-capstone.md),
  [ADR 0028](./0028-salt-v2-additive-ternarization.md), and
  [plan 0043](../plans/0043-salt-v2-sota-campaign.md); consumes the training and
  performance work in [ADR 0031](./0031-reduced-precision-optimizer-and-step1-next-steps.md).

> **Claim boundary.** “PyTorch for ternary” is the product direction, not a
> claim earned by this ADR. Tritium may make that claim only after the release
> gates below pass and at least one external researcher reproduces the documented
> workflow. The existing `v1.0.0` tag remains the frozen infrastructure/API
> milestone. Current post-tag source will publish as `v1.1.0`; the tag must not
> move or be reused.

## Context

Tritium already has more than an inference-kernel skeleton:

- a Rust eager reverse-mode `Tape`, CUDA `DeviceTape`/`DeviceTrainer`, optimizers,
  FSDP/ZeRO and distributed-checkpoint substrate;
- PyTorch autograd wrappers for ternary Conv1d, FSQ, masked STE and LSQ;
- SALT V2 additive PTQ, physical-rate accounting, content-addressed campaign
  state, and native packed inference formats;
- CPU, CUDA, Metal, ROCm, wgpu, WASI, C, Candle, Burn, ONNX, Python and serving
  integration substrate.

Those capabilities did not form a public research product when this decision
started. The legacy Conv1d/FSQ wrappers still cross host lists, but the first
v1.1 slices have since landed a device-resident reference `TernaryLinear`,
recursive QAT conversion, direct ternary Conv1d/Conv2d/Embedding modules,
Hugging Face QAT checkpoint integration, Trainer/Accelerate gates and the core
estimator catalog. Native fused dispatch, the phased PTQ/refinement facade,
portable whole-Tape training, browser training, public packages and the
production/community surface remain open. Plan 0043 has not yet produced the
pinned 27B artifact or its empirical evidence.

PyTorch won research adoption through an imperative graph, familiar modules,
debugging, packaging and a five-minute path to a real experiment. Tritium should
provide those properties without spending the v1.1 cycle cloning PyTorch's
general tensor, optimizer, data and distributed ecosystems.

## Decision

### 1. Product architecture and fixed release sequence

For v1.1:

- **PyTorch owns** tensors, eager dynamic graphs, `nn.Module`, optimizers, AMP,
  DDP/FSDP, data loading and Python control flow.
- **Tritium owns** ternary projection semantics, estimator recipes, SALT
  fitting/refinement, packed kernels, artifact formats, exact physical
  accounting, conversion coverage and evidence receipts.
- Rust `Tape` and `DeviceTape` remain production campaign/reference engines.
  They are not described as missing and are not replaced by Python.

The release sequence is binding:

- **v1.1:** PyTorch/Hugging Face research frontend, whole-Tape semantic parity
  on every inference backend, compiled TypeScript/WebGPU training, ONNX
  inference, and the complete production/community surface in this ADR.
- **v1.2:** safe typed native Rust `Tensor`/`Module`/dynamic-autograd frontend
  plus Python bindings that run without PyTorch, both at v1.1 semantic parity.
- **v1.3:** trainable whole-model ONNX import for supported graphs.

v1.1 therefore freezes language-neutral recipe, artifact, coverage, error,
backend-capability and training-operation schemas now. Native and ONNX
frontends consume those contracts instead of inventing separate ternary
ecosystems.

This architecture deliberately rejects a complete `torch.nn` mirror. Tritium
ships deep modules for supported ternary operations and reuses ordinary PyTorch
modules everywhere else.

### 2. Public Python and Hugging Face interface

Distribution name is **`tritium-torch`**. Import namespace remains
**`tritium`**, with research APIs under `tritium.torch` and direct layers under
`tritium.nn`. PyTorch is an optional dependency of the Rust/Python runtime, not
a dependency of frozen core crates.

Stable primitive workflow follows PyTorch/JAX-style explicit phases:

```python
import tritium.torch as tt

prepared = tt.prepare(
    dense_model,
    tt.TernaryConfig.qat(estimator="salt-ste", planes=1),
    strict=True,
    inplace=True,
)
loss = prepared.model(**batch).loss
loss.backward()
optimizer.step()
qat_export = tt.convert(prepared)
qat_receipt = tt.export(qat_export, "./model.qat.tsalt2")

prepared = tt.prepare(
    dense_model,
    tt.TernaryConfig.ptq(profile="near-lossless-v1"),
    strict=True,
    inplace=True,
)
calibration = tt.calibrate(prepared, calibration_data, work_dir="./ptq-work")
ptq = tt.convert(prepared, calibration=calibration)
ptq_receipt = tt.export(ptq, "./model.ptq.tsalt2")

refined = tt.refine(
    ptq,
    teacher=dense_model,
    training=refinement_data,
    validation=validation_data,
    config=tt.RefinementConfig.hard_pv(structure="dense"),
    work_dir="./refine-work",
)
refined_receipt = tt.export(refined, "./model.refined.tsalt2")
reloaded = tt.load("./model.refined.tsalt2", device="cuda")
coverage = tt.inspect(reloaded)
```

Required stable types and behavior:

- `prepare`, `calibrate`, `convert`, `refine`, `export`, `load`, and `inspect`
  are the primitive contracts. `prepare_qat` and one-call `quantize` remain
  convenience compositions over those phases, not separate implementations.
- `prepare` validates the whole graph before mutation. `inplace` is mandatory,
  not defaulted: `True` consumes the supplied view without cloning 27B masters;
  `False` must preserve independent ownership. Callers rebind the return value
  because a root module may be replaced.
- `TernaryConfig.qat(...)` and `TernaryConfig.ptq(...)` describe versioned
  recipes. QAT supports one to three additive planes. `profile`, `planes` and
  measured `target_bpw` replace misleading `bits=2` controls.
- `TernaryConfig.ptq` has no refinement field. `RefinementConfig` owns
  scale-only and hard-PV choices. QAT exports, PTQ artifacts, scale-only
  children and hard-PV children have distinct discriminants, ancestry, costs
  and claim labels; none may be relabeled as another.
- `PreparedModel`, `CalibrationReceipt`, `ConversionResult`,
  `QuantizationResult`, `RefinementResult`, `ArtifactRef`, `ExportReceipt`, and
  `CoverageReport` have versioned schemas shared by Rust, Python and
  TypeScript. Non-replayable calibration iterables are materialized and hashed
  before fitting or resume.
- `load(...)` accepts a Tritium artifact or Hugging Face directory/ID and
  returns a supported `nn.Module` using the selected Tritium runtime.
- `inspect(...) -> CoverageReport` accounts for every tensor exactly once.
- `tritium.nn` includes `TernaryLinear`, `TernaryEmbedding`,
  `TernaryConv1d`, `TernaryConv2d`, and existing FSQ modules. Each implements
  `from_float`, state-dict round-trip, train/eval semantics and explicit export.

Hugging Face integration registers the same serializable `TernaryConfig` plus a
quantizer adapter usable from `from_pretrained`,
`save_pretrained`, generation and Trainer/Accelerate workflows. Module paths,
tied/shared parameters, tokenizer/config files, device and dtype must survive
conversion.

Advanced research interface lives below the facade:

```python
class Estimator(torch.nn.Module):
    algorithm_id: str
    schema_version: int

    def project(self, master, *, context) -> TernaryProjection: ...
```

`TernaryProjection` contains a non-empty tuple of one to three
`TernaryPlane { trits, scales, group_size, structure }` values plus the
canonical hard decoded tensor. Trits are exactly `{-1,0,+1}`; scales are finite
and nonnegative; structure is `dense` or `s34`; decoded shape equals the latent
master. No residual, bias, zero point, codebook or decoder field exists.
Tritium validates the contract, provides pure-Torch reference execution,
selects optimized adapters, and rejects unregistered or unexportable
projections. Researchers may add estimators without rebuilding Tritium;
custom fused kernels remain a separate runtime-adapter seam.

Public-interface maturity is explicit. All rows block v1.1:

| Phase | Required surface |
|---|---|
| Core | modules, configuration, structured errors, coverage and schemas |
| Eager research | QAT, estimator catalog/plugins, planes 1–3 and state round-trip |
| Ecosystem | Hugging Face Trainer/Accelerate, DDP/FSDP and diagnostics |
| Conversion | phased PTQ/refinement, load/export and seek-backed artifacts |
| Optimized | native dispatch, AMP/vmap/compile and performance/no-copy gates |

### 3. Correctness, performance and failure contracts

- Conversion is two-phase: inspect/validate the entire module graph, then
  commit replacements. Failure cannot expose a partially converted model.
- Strict coverage is default. Every unique tensor is receipted exactly once
  with all aliases, scope, role, shape, source bytes, disposition and packed
  bytes. Unsupported modules, duplicate/shared ownership and targeted dense
  fallbacks fail closed. Explicitly required preserved norms/biases and deferred
  vision tensors are allowed only under named policy and remain visible.
- QAT never destroys latent masters during `eval()`. Export performs explicit
  hard conversion from a consistent parameter snapshot.
- Export is transactional and content-addressed. Canonical SALT V2 uses
  `.tsalt2`; seek-backed SALT-GGUF is an explicit export adapter admitted only
  when it preserves recipe, coverage, lineage and physical-ledger identity.
  Reload binds recipe version, source identity, tensor dispositions, physical
  bytes and runtime profile.
- Parameter version changes invalidate packed caches before next forward.
- No Rust panic crosses Python/JS/C boundaries. `TritiumError` carries stable
  `code`, `stage`, qualified module/tensor and safe partial receipt fields.
- Supported steady-state CPU/GPU paths perform no `.tolist()`, implicit CPU
  move, dense dequantized shadow or global synchronization.
- PyTorch operators register real kernels, autograd, fake/meta behavior,
  autocast and `torch.compile` support through the dispatcher. CPU/CUDA tensor
  ownership and CUDA stream ordering are explicit and adversarially tested.
- Optimized and composite reference adapters must match forward and first-order
  gradients. Python wrapper overhead is at most 5% versus the direct Tritium
  backend at representative ternary-linear shapes after warmup.

### 4. Algorithmic platform and SOTA work

v1.1 ships a versioned core estimator catalog:

- AbsMean masked STE and annealed STE;
- LSQ;
- TWN and TTQ compatibility/baseline recipes;
- sparse ternary projection, including the S34 structural profile;
- SALT additive STE, SALT V2 PTQ, scale-only refinement and hard discrete
  refinement;
- validated external estimators through the interface above.

SALT remains the flagship method. ADR 0028's zero-point-free additive format,
nested profiles and physical accounting remain binding. v1.1 closes these
production research gaps:

- checkpoint-native layer, block and sliding-window reconstruction;
- final-block/LM-head teacher-logit CE/KL objectives, not layer MSE alone;
- output-aware multi-start initialization and matched basin ablations;
- true PV-style alternating continuous-scale and discrete-trit refinement;
- S34 annealed sparse refinement whose training residual reaches exactly zero
  and is absent from the artifact;
- streamed/blockwise fitting, large-K acceleration, durable resume and no
  dense whole-model inverse-Hessian assumption;
- matched reproduced ternary/additive and global low-bit baselines.

Relevant transferable methods include
[PV-Tuning](https://arxiv.org/abs/2405.14852),
[GuidedQuant](https://proceedings.mlr.press/v267/kim25d.html),
[YAQA](https://arxiv.org/abs/2505.22988),
[PTQTP](https://arxiv.org/abs/2509.16989),
[BPDQ](https://arxiv.org/abs/2602.04163),
[OA-EM](https://arxiv.org/abs/2604.08118),
[Sherry](https://arxiv.org/abs/2601.07892),
[CAT-Q](https://arxiv.org/abs/2606.26650), and
[LFQ](https://arxiv.org/abs/2605.29756). Paper-only results remain visibly
labeled until official code can be reproduced. Representation-incompatible
codebooks, trellises, dense residuals and row biases remain baselines or sources
of optimization ideas, never silent SALT features.

### 5. High-level training and diagnostics

v1.1 does not clone `DataLoader` or optimizers. It provides:

- Hugging Face Trainer/Accelerate integration and tested DDP/FSDP paths;
- `TritiumTrainer` as a thin orchestration facade over standard PyTorch/HF
  components, with PTQ, QAT, refinement, evaluation, resume and export stages;
- TensorBoard, Weights & Biases and OpenTelemetry adapters for trit histograms,
  zero rate, plane/scales, saturation, gradients, reconstruction error,
  teacher KL, physical bpw, runtime and resident memory;
- a deterministic calibration-data seam, dataset fingerprints and replayable
  experiment receipts.

### 6. Whole-Tape portable training and browser product

v1.1 freezes a language-neutral `TrainingOpManifestV1` and a fallible internal
`TrainBackendV1` conformance seam. This is not the public native Tensor frontend
scheduled for v1.2. The manifest covers every current public `Tape` operation
and first-order VJP:

- STE surrogate, SALT STE, LSQ STE and FSQ;
- dense and ternary matmul, transpose, embedding gather, column slice/concat,
  detach, constant scale, bias, add and multiply;
- Conv1d plus the v1.1 Conv2d extension;
- ReLU-squared, SiLU, RMSNorm, softmax, causal mask, RoPE and composed
  attention;
- MSE and softmax cross-entropy;
- SGD plus AdamW, CautiousAdamW, Int8AdamW and Muon steps;
- checkpoint/resume and canonical artifact export/reload.

CPU, CUDA, ROCm, Metal, native wgpu, WASI/WASM and MCU must pass the same
versioned semantic vectors on actual targets. Browser WebGPU consumes the same
manifest. F32 is mandatory; additional dtypes are capability-receipted.
Bounded shapes and memory ceilings are allowed on constrained targets, but an
operation cannot silently become inference-only or a host round trip. Semantic
parity is mandatory; performance is tiered by hardware. Failure on any declared
backend blocks v1.1 until this ADR is explicitly amended.

Browser support is a product, not the current WASI scalar build. Intended npm
distribution is `@tritium-ai/web`; name reservation and publication remain
separately authorized external actions. The public surface is a compiled
`WebTrainingSession`, not arbitrary JavaScript dynamic autograd:

```typescript
const session = await tritium.prepareTraining(model, config);
await session.forward(batch);
await session.backward();
await session.step();
const checkpoint = await session.checkpoint();
await session.resume(checkpoint);
await session.export("model.tsalt2");
```

WebGPU owns accelerator execution; WASM owns validation, orchestration and a
separately labeled deterministic fallback. `npm pack` must install into an empty
strict-TypeScript project, pass `tsc --noEmit`, produce a release bundle and run
load through native reload without a per-step GPU-to-CPU tensor transfer.
Current Chrome, Firefox and Safari must each prove real WebGPU execution. A
fallback run never satisfies a WebGPU gate.

### 7. Interop, packaging, production and community

v1.1 requirements:

- whole-model ONNX import/export for supported ternary graphs and a real ORT
  inference session; arbitrary trainable ONNX is binding v1.3 work;
- reference Candle/Burn paths upgraded or clearly capability-tiered—no host
  round trip advertised as accelerated training;
- trusted PyPI publication, ordered crates.io publication, CPU/CUDA wheels,
  supported Python/Torch compatibility matrix, clean-wheel tests, hashes,
  signatures, SBOM and provenance;
- one-command Colab install with no compiler, completing load → PTQ → QAT
  backward/optimizer step → export → reload → generation in five minutes on
  the tutorial model, excluding first model download;
- hardened OpenAI-compatible serving, authentication, limits, backpressure,
  graceful shutdown, Prometheus/OpenTelemetry metrics and health/readiness;
- OCI image, Helm chart, KEDA autoscaling, Knative serverless example, Grafana
  dashboard, OpenTelemetry traces and documented GPU scheduling/cache/storage
  patterns;
- current README/book/API reference, definitive ternary guide, PTQ/QAT/estimator
  tutorials, architecture/format/benchmark documentation and release migration
  guide;
- `CONTRIBUTING`, Code of Conduct, governance, citation, support policy,
  security process, GitHub templates/Discussions and a moderated Discord. GitHub
  remains the authoritative archive for decisions and support outcomes.

Release validation has three non-circular stages:

1. **Local RC:** build candidate wheels, crates, npm archives, OCI images,
   signatures, SBOMs and attestations; install and test them from clean local
   environments without registry publication.
2. **Authorized activation:** publish immutable registry artifacts, create
   public community surfaces and deploy public references only after explicit
   approval.
3. **Post-publication smoke:** install exact registry artifacts and regenerate
   smoke receipts. A failure does not rewrite `v1.1.0`; it advances to a
   corrective `v1.1.1`.

### 8. Audited model zoo and launch evidence

v1.1 ships three audited tiers:

1. **Five-minute tier:** pinned SmolLM2-135M for CPU/Colab/browser end-to-end
   PTQ/QAT/export smoke; SmolLM2-1.7B freezes the scalable recipe.
2. **Native-ternary reference:** pinned BitNet b1.58 2B4T with existing
   correctness/perplexity/performance evidence retained.
3. **Flagship:** pinned Qwen3.6-27B language core plus bundled MTP drafter,
   with separate Compact PTQ, NearLossless PTQ and NearLossless refined
   artifacts. Vision remains explicitly deferred.

Every model card binds source/tokenizer/data/harness revisions, conversion
recipe, exact serialized and resident bytes, preserved tensors, peak memory,
runtime, energy where available, GPU-hours and quality confidence intervals.
No logical `log2(3)` rate supports a compression claim. “More than 10x” is
allowed only when an exact whole-artifact or resident measurement proves it.

The Qwen artifact retains plan 0043's gates and adds long-generation, code and
instruction-following retention; MTP acceptance and speedup; prefill/decode at
multiple contexts/batches; export/reload identity; and second-machine
reproduction. Additive-ternary SOTA and global low-bit Pareto are separate
verdicts.

### Implementation amendment: installed tutorial result

The source-free PyTorch QAT tutorial emits
`tritium.installed-qat-tutorial.v3`. Version 3 retains v2's latent safetensors,
optimizer checkpoint identities and exact resumed step, replaces runner-local
absolute paths with portable contained paths, hashes every hard-artifact file,
and binds the exact candidate wheel bytes, source revision, release and run ID.
The release-evidence registry independently revalidates those bindings and
counts the result as `installed-qat-tutorial` evidence; it does not waive the
remaining frontend, distributed, export, machine or compatibility gates.
Earlier v1/v2 developer results are not admitted v1.1 receipts. Unknown or
older tutorial-result schemas fail closed.

The installed Hugging Face gate emits `tritium.hf-lifecycle.v1`. It binds a
two-plane tied-Llama forward/backward/AdamW step, safe `save_pretrained`, native
`AutoModelForCausalLM` reload, exact logits and current alias-aware coverage to
the candidate wheel, source, release, run and complete checkpoint tree. The
registry admits this only as `frontend-lifecycle`; distributed training and
whole-model export/reload remain independent requirements.

Whole-model hard-artifact admission uses `tritium.hf-export-reload.v1`. A
source-free installed wheel trains and hard-converts the complete tiny tied
Llama fixture, atomically exports its immutable QAT-hard tree, strict-reloads a
fresh Hugging Face shell, and proves exact logits, greedy token generation,
shared packed input/output storage and absence of persistent dense shadows for
converted weights. The receipt binds every artifact byte plus the exact wheel,
source, release and run. The registry counts it only as `export-reload`; it is a
representative language-model lifecycle gate, not flagship-Qwen or optimized
native-kernel evidence.

Installed diagnostics admission uses `tritium.installed-observability.v1` and
the registry kind `observability`. It is a mandatory part of the `pytorch-hf`
gate. A source-free, compiler-free installed candidate wheel must exercise real
TensorBoard, offline Weights & Biases and OpenTelemetry adapters over one fixed
tied-weight QAT fixture. Retained evidence must cover trit histograms, zero rate,
planes, scales, saturation, gradients, reconstruction error, teacher KL,
physical bpw, runtime and resident memory. Admission revalidates the candidate
wheel `METADATA`/`RECORD`, the executed installation inventory, exact metric
names and finite values, telemetry formats and every retained byte. A
source-tree smoke, mocked adapter, online W&B run, copied counter, missing
metric family or structurally substituted receipt cannot satisfy this kind.
This receipt does not substitute for distributed training, optimized native
dispatch, flagship quality/performance, serving telemetry or second-machine
reproduction.

Production output-aware search uses the canonical `TSV2OUT` version-1 receipt.
Its source model, activation cache, token stream, held-out validation set,
block/sliding-window schedule, objective weights, temperature, batch count and
restart count form one immutable specification identity. Candidate evaluation
streams one output batch at a time, binds exact teacher/student bytes, reports
block MSE plus final-logit teacher cross-entropy and temperature-scaled KL, and
selects a complete deterministic restart set with content-ID tie breaking.
Strict reopen rejects corruption, noncanonical candidate order, missing basins
and teacher drift between basins. This core receipt is not flagship evidence
until plan 0043 binds its selected candidate to the corresponding immutable
master campaign and executes it on source-bound checkpoint data.

The generic Qwen runtime emits an opaque, non-constructible `TSQ35EX` v1
transcript binding loaded profile/package/config/preserved bytes, self-asserted
backend claims, exact token-batch boundaries and runtime-produced output bytes.
Because the public backend trait is caller-implementable, this lower transcript
is explicitly untrusted and cannot satisfy campaign admission. Its record
reserves separate block-output and final-logit coverage/digests; absent scopes
remain explicit. Durable comparison re-executes the loaded model over the same
tokens and requires byte-identical canonical evidence. A separate SALT-owned
sealed session must construct a built-in backend internally, bind the
authoritative completion/master/package lineage, and re-execute before it can
mint an admitted execution receipt. Caller-provided logits, candidate labels,
backend identities, or lower transcript bytes never mint campaign provenance.

Multi-device admission uses
`tritium.hf-distributed-qualification.v1`. The validator requires ordered DDP
and FSDP NCCL/fp16 runs on two distinct physical GPU UUIDs, exact checkpoint
and RNG continuation, zero profiled ternary-op host transfers, candidate-wheel
and model identities, internally consistent token throughput, measured peak
memory, a workload of at least 100M parameters, 20 measured steps at sequence
length 128 or greater, and scaling efficiency of at least 70% for DDP and 55% for FSDP against
the bound single-device baseline. Shared-device ranks, missing modes, dirty
source, stale artifacts, copied arithmetic or lower efficiency fail closed.
This schema and registry dispatch are structural until a qualified two-GPU run
produces the receipt.
The producer runs an isolated installed-wheel worker under two-rank `torchrun`
for each mode, benchmarks the same 127,943,680-parameter two-plane Llama against
a rank-zero single-device baseline, profiles host transfers, exercises DDP
rank checkpoints and FSDP distributed-checkpoint restore including RNG state,
and atomically publishes the receipt plus all four content-bound rank checkpoint
files. The producer refuses a dirty or mismatched checkout and fewer than two
visible, distinct GPU UUIDs. It creates an isolated runtime, installs only the
named candidate wheel without dependency resolution, and runs the frozen worker
under `python -I`; worker and interpreter substitution are not release seams.

## v1.1 release gate

`v1.1.0` may publish only when all boxes are green:

- [ ] Plan 0043 produces the pinned Qwen3.6-27B language-plus-MTP PTQ and
      separately labeled refined evidence; refined NearLossless is within 1%
      relative held-out perplexity, task-retention gates pass, and native
      runtime/memory/physical-byte receipts exist.
- [ ] Explicit prepare/calibrate-or-train/convert/refine/export/load phases and
      their one-call PyTorch/HF facades preserve tied weights/state, compile,
      distribute, export and reload. PTQ and refined lineages remain distinct.
- [ ] Steady-state optimized CPU/CUDA paths show no hidden host transfer or
      dense runtime shadow; `TrainingOpManifestV1` passes every declared native
      backend and performance tiers are receipted.
- [ ] Core estimator catalog, external-estimator validation, production SALT
      reconstruction/refinement and reproduced baseline harness pass.
- [ ] Compiled TypeScript `WebTrainingSession` passes the whole manifest in
      Chrome, Firefox and Safari WebGPU, including checkpoint/export/native
      reload; WASM fallback is reported separately.
      Admission uses `tritium.browser-training-qualification.v1`: three ordered
      physical-engine lanes bound to the exact npm candidate, all 114 vectors,
      complete lifecycle and fault injection, and content-hashed traces with no
      steady-state readback or hidden WASM dispatch. Structural emulation cannot
      satisfy this box.
- [ ] Whole-model ONNX inference interoperability passes; trainable ONNX remains
      labeled v1.3.
- [ ] Local RC archives install without a source checkout or compiler; after
      explicit activation, exact PyPI/crates/npm/container artifacts pass the
      same post-publication smoke.
- [ ] Hardened serving, OCI/Kubernetes/serverless/observability artifacts pass
      deployment and failure-injection gates.
- [ ] Three-tier model zoo, guides, governance/community and compatibility docs
      are public and generated claims match evidence receipts.
      Admission is fail-closed through `tritium.model-zoo-admission.v1`,
      `tritium.generated-claims.v1` and `tritium.governance-docs.v1`. The frozen
      four-model ladder binds immutable model/tokenizer revisions, model cards,
      candidate artifacts and admitted evidence ancestry. Generated README,
      model-zoo, benchmark and compatibility claims bind those same receipts;
      the governance inventory must pass link, contact and independent policy
      review and cannot advertise unstaffed channels.
- [ ] Independent fresh-environment reproduction passes on a second machine;
      release revision is clean, reviewed, signed and tagged `v1.1.0`.
      `tritium.second-machine-reproduction.v1` must bind the complete candidate
      inventory, distinct machine/operator, fixed reproduction commands,
      regenerated tables and zero divergences. Final
      `tritium.independent-release-review.v1` must cover every other evidence
      receipt with code/security/evidence review and zero open findings.

## Binding post-v1.1 work

- **v1.2:** safe typed native Rust `Tensor`/`Module`/dynamic-autograd API with
  no panic contracts, plus Python bindings that run without PyTorch. Both must
  consume v1.1 schemas and match the PyTorch reference frontend.
- **v1.3:** arbitrary trainable ONNX graph import for supported operations,
  including gradients, optimizer/checkpoint lifecycle and artifact identity.
- **Ordered but unversioned:** Qwen3.6 vision/multimodal completion, then
  Qwen3.6-35B-A3B MoE and an independent-family flagship. Higher-order
  gradients and broader compiler/graph IR remain evidence-gated.

## Rejected alternatives

- **Move/reissue `v1.0.0`:** destroys release provenance.
- **Native-first v1.1:** delays researcher access while duplicating solved graph,
  optimizer, data and distributed machinery.
- **SALT-only closed API:** prevents researchers from testing new estimators and
  makes Tritium a converter, not a research platform.
- **Five shallow model conversions:** weaker evidence than one audited 27B
  flagship plus reproducible ladder models.
- **Trainable ONNX as v1.1 gate:** ONNX remains deployment-first; build after
  supported inference import/export and core training contracts stabilize.

## Consequences and risks

- v1.1 is a large release. Browser training, all-backend semantics and full
  deployment/community work are real blockers by decision, not marketing
  stretch goals.
- PyTorch/Torch ABI compatibility increases wheel and CI cost.
- Backend semantic parity may expose infeasible constrained-target requirements.
  Such a result requires an explicit ADR amendment; it cannot be relabeled green.
- Qwen3.6-27B evidence needs substantial storage, checkpoint access and GPU time.
  Existing local-first/no-paid-compute policy remains in force until Brian Lam
  separately approves a frozen paid-run recipe, ceiling and stop gate.
