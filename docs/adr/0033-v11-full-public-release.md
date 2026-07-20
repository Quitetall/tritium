# ADR 0033 — Tritium v1.1 full public release: ternary research platform

Status: **PROPOSED** (2026-07-20)

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

Those capabilities do not form a public research product. The Python wrappers
currently convert tensors to host lists and reconstruct new tensors, so CUDA
training crosses a synchronous device-to-host-to-device boundary. There is no
`TernaryLinear`, recursive `nn.Module` conversion, Hugging Face quantizer,
stable state/export contract, published CUDA wheel, browser training SDK, or
production-scale Python PTQ/refinement facade. Plan 0043 has not yet produced
the pinned 27B artifact or its empirical evidence.

PyTorch won research adoption through an imperative graph, familiar modules,
debugging, packaging and a five-minute path to a real experiment. Tritium should
provide those properties without spending the v1.1 cycle cloning PyTorch's
general tensor, optimizer, data and distributed ecosystems.

## Decision

### 1. Product architecture: PyTorch first, Tritium-native next

For v1.1:

- **PyTorch owns** tensors, eager dynamic graphs, `nn.Module`, optimizers, AMP,
  DDP/FSDP, data loading and Python control flow.
- **Tritium owns** ternary projection semantics, estimator recipes, SALT
  fitting/refinement, packed kernels, artifact formats, exact physical
  accounting, conversion coverage and evidence receipts.
- Rust `Tape` and `DeviceTape` remain production campaign/reference engines.
  They are not described as missing and are not replaced by Python.

Native Tritium becomes an equally first-class research frontend during the 1.x
line. v1.1 therefore freezes language-neutral recipe, artifact, coverage and
error schemas now. A safe Rust `Tensor`/`Module` frontend later consumes those
same contracts instead of inventing a second ternary ecosystem.

This architecture deliberately rejects a complete `torch.nn` mirror. Tritium
ships deep modules for supported ternary operations and reuses ordinary PyTorch
modules everywhere else.

### 2. Public Python and Hugging Face interface

Distribution name is **`tritium-torch`**. Import namespace remains
**`tritium`**, with research APIs under `tritium.torch` and direct layers under
`tritium.nn`. PyTorch is an optional dependency of the Rust/Python runtime, not
a dependency of frozen core crates.

Stable golden-path API:

```python
from tritium.torch import TernaryConfig, inspect, load, prepare_qat, quantize

qat_model = prepare_qat(
    dense_model,
    TernaryConfig.qat(
        estimator="salt-ste",
        target_modules=("Linear", "Embedding", "Conv1d"),
        planes=1,
    ),
)

loss = qat_model(**batch).loss
loss.backward()
optimizer.step()

result = quantize(
    dense_model,
    calibration=calibration_data,
    config=TernaryConfig.ptq(profile="near-lossless-v1"),
    work_dir="./tritium-work",
)
receipt = result.export("./model.salt.gguf")
reloaded = load("./model.salt.gguf", device="cuda")
coverage = inspect(reloaded)
```

Required stable types and behavior:

- `TernaryConfig.qat(...)` and `TernaryConfig.ptq(...)` describe versioned
  recipes. `profile`, `planes` and measured `target_bpw` replace misleading
  `bits=2` controls.
- `prepare_qat(model, config) -> torch.nn.Module` validates the whole graph,
  then replaces supported leaves in the supplied model before optimizer
  creation. It does not clone 27B masters; after conversion the caller must not
  use the pre-conversion view concurrently. FP16/FP32 latent masters remain
  ordinary `Parameter`s; hard ternary projections run in forward; registered
  estimators define backward behavior.
- `quantize(model_or_id, calibration, config, work_dir) -> QuantizationResult`
  drives resumable SALT PTQ and separately labeled refinement tracks.
- `QuantizationResult` exposes `model`, `coverage`, `report`, `export(...)` and
  `save_pretrained(...)`.
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

`TernaryProjection` contains exact `{-1,0,+1}` planes, finite nonnegative
scales, canonical hard decoded value and grouping metadata. Tritium validates
the contract, provides pure-Torch reference execution, selects optimized
adapters, and rejects unexportable projections. Researchers may add estimators
without rebuilding Tritium; custom deployment kernels remain an explicit
separate adapter.

### 3. Correctness, performance and failure contracts

- Conversion is two-phase: inspect/validate the entire module graph, then
  commit replacements. Failure cannot expose a partially converted model.
- Strict coverage is default. Unsupported modules, duplicate/shared ownership,
  dense fallbacks and preserved tensors fail closed unless a policy explicitly
  permits them; every permitted exception and byte is recorded.
- QAT never destroys latent masters during `eval()`. Export performs explicit
  hard conversion from a consistent parameter snapshot.
- Export is transactional and content-addressed. Reload binds recipe version,
  source identity, tensor dispositions, physical bytes and runtime profile.
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

### 6. Portable training and browser product

v1.1 targets the same **release-core training semantics** across CPU, CUDA,
ROCm, Metal, wgpu/WebGPU and WASI. The portable core is:

- Linear, Embedding, Conv1d/Conv2d, normalization and required transformer
  reshapes/activations;
- supported losses, STE/LSQ/SALT projection, SGD/AdamW and checkpoint/resume;
- forward, first-order backward, optimizer step and artifact export/reload.

Semantic parity is mandatory; performance is tiered by hardware. CUDA, ROCm,
Metal and WebGPU carry native accelerator performance gates. CPU/WASI and any
MCU training profile carry bounded-shape correctness and memory gates. A backend
that cannot implement the core must remain explicitly inference-only and blocks
v1.1 until this ADR is amended; documentation cannot call inference parity
training parity.

Browser support is a product, not the current WASI scalar build:

- an npm package exposes a TypeScript high-level Module/Trainer API;
- WebGPU executes the portable training core, including optimizer/checkpoint;
- WASM provides orchestration, validation and fallback;
- browser and native paths import/export the same versioned Tritium artifact;
- conformance runs in Chrome and Firefox-capable CI, with feature detection and
  deterministic CPU fallback;
- an install-free tutorial performs forward, backward, optimizer step,
  checkpoint/resume and inference entirely in-browser.

### 7. Interop, packaging, production and community

v1.1 requirements:

- whole-model ONNX import/export for supported ternary graphs and a real ORT
  inference session; arbitrary trainable ONNX is binding later-1.x work;
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
  dashboard and documented GPU scheduling/cache/storage patterns;
- current README/book/API reference, definitive ternary guide, PTQ/QAT/estimator
  tutorials, architecture/format/benchmark documentation and release migration
  guide;
- `CONTRIBUTING`, Code of Conduct, governance, citation, support policy,
  security process, GitHub templates/Discussions and a documented community
  chat channel.

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

## v1.1 release gate

`v1.1.0` may publish only when all boxes are green:

- [ ] Plan 0043 produces the pinned Qwen3.6-27B language-plus-MTP PTQ and
      separately labeled refined evidence; refined NearLossless is within 1%
      relative held-out perplexity, task-retention gates pass, and native
      runtime/memory/physical-byte receipts exist.
- [ ] One-call PyTorch/HF PTQ plus QAT works through ordinary `loss.backward()`
      and optimizer step, preserves tied weights/state, compiles, distributes,
      exports and reloads.
- [ ] Steady-state optimized CPU/CUDA paths show no hidden host transfer or
      dense runtime shadow; backend conformance and performance tiers pass.
- [ ] Core estimator catalog, external-estimator validation, production SALT
      reconstruction/refinement and reproduced baseline harness pass.
- [ ] Portable training core passes every declared backend, including browser
      WebGPU forward/backward/optimizer/checkpoint/export.
- [ ] Whole-model ONNX inference interoperability passes; trainable ONNX remains
      labeled later-1.x.
- [ ] PyPI/crates/wheels/Colab/browser packages install from published artifacts
      without a source checkout.
- [ ] Hardened serving, OCI/Kubernetes/serverless/observability artifacts pass
      deployment and failure-injection gates.
- [ ] Three-tier model zoo, guides, governance/community and compatibility docs
      are public and generated claims match evidence receipts.
- [ ] Independent fresh-environment reproduction passes on a second machine;
      release revision is clean, reviewed, signed and tagged `v1.1.0`.

## Binding post-v1.1 work

These are required in the 1.x line, designed against v1.1 schemas, but do not
block `v1.1.0`:

1. safe, typed, first-class native Rust `Tensor`/`Module`/dynamic-autograd API
   with no panic contracts and parity against the PyTorch reference frontend;
2. arbitrary trainable ONNX graph import where supported ops retain gradients;
3. Qwen3.6 vision/multimodal completion, then MoE and independent-family
   flagship expansion;
4. higher-order gradients and broader compiler/graph IR only after two real
   frontends and two execution consumers justify the seam.

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
