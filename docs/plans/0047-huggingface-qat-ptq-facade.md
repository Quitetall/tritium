# 0047 — Hugging Face QAT/PTQ facade

Status: **IN PROGRESS** (2026-07-20)

- **Decision:** [ADR 0033](../adr/0033-v11-full-public-release.md)
- **Parent:** [plan 0044](./0044-v11-full-public-release.md)
- **Dependency:** [plan 0046](./0046-torch-dispatcher-zero-copy.md) — semantic
  dispatcher complete; native fused performance gate remains open

## Goal

Make Tritium models ordinary Hugging Face models for loading, training,
checkpointing and generation while preserving exact conversion coverage and
recipe identity. PTQ remains a separate resumable SALT workflow and cannot be
represented by a QAT checkpoint.

## Public seams under test

- `tritium.nn.TernaryEmbedding`
- `tritium.nn.TernaryConv1d` and `tritium.nn.TernaryConv2d`
- `tritium.torch.HfTritiumConfig`
- primitive `tritium.torch.prepare`, `calibrate`, `convert`, `export`, `load`,
  and `inspect`
- convenience `tritium.torch.prepare_qat` and `quantize` facades
- Hugging Face `AutoModelForCausalLM.from_pretrained`
- `PreparedModel`, `CalibrationReceipt`, `ConversionResult`,
  `QuantizationResult`, and `ExportReceipt`

## Step 1 — trainable Hugging Face QAT checkpoint

- Add tensor-only estimator identity state so `safetensors` never receives a
  Python object.
- Convert `torch.nn.Embedding` without cloning or breaking tied input/output
  weights.
- Register a lazy external Hugging Face quantization config and quantizer under
  `quant_method="tritium"`.
- Attach the exact Tritium recipe to a converted `PreTrainedModel` so native
  `save_pretrained` writes it into `config.json`.
- Rebuild ternary modules before Hugging Face loads checkpoint tensors.

Gate: a tiny tied-weight Llama model performs forward/backward/optimizer step,
saves with safe serialization, automatically reloads through
`AutoModelForCausalLM`, retains weight tying and matches logits exactly.

## Step 2 — Trainer/Accelerate and distributed state

- [x] Verify Trainer and direct Accelerate steps use ordinary optimizers.
- [x] Verify single-process gradient accumulation and checkpoint/resume.
- [x] Verify CPU bf16 mixed precision and exact RNG continuation through
  Accelerate state resume. CUDA fp16 remains in the dispatcher gate and the
  distinct-device distributed lane.
- [x] Run real two-rank DDP and FSDP semantic tests on CPU/Gloo; skipped
  distributed tests are failures in the release lane. FSDP uses native sharded
  distributed-checkpoint resume; PyTorch 2.11 CPU full-state materialization
  currently segfaults on rank 0 and remains a red export gate.
- [ ] Run accelerator DDP/FSDP mixed-precision and throughput gates on distinct
  physical devices.

## Step 3 — resumable PTQ facade

- [x] Add a bounded-memory Rust producer seam that plans each global tensor
  master before fitting and streams its canonical Pmax payload tile by tile.
  Independent streams are byte-exact with the whole-model reference solver.
- [x] Add a canonical, bounded-reopen factorized-curvature record that binds
  source provenance, global tensor identity, geometry, and all Kronecker
  factors, then reproduces the exact tensor-master stream after restart.
- [x] Resume a pure-PTQ tensor fit from those bound identities without retaining
  or rebuilding the complete activation cache.
- [x] Reconcile the admitted 506-matrix Qwen source and exact evidence namespace
  into the resumable content-addressed PTQ campaign, fitting only missing
  masters and sealing only after strict canonical reopen.
- [x] Expose that expert master-campaign primitive and immutable structural
  receipt through the abi3 Python wheel without a Python-list weight bridge.
- [x] Stream selected Compact/NearLossless parent prefixes into byte-exact
  seek-backed packages and exact package admission without constructing a
  whole-model semantic package or duplicating the caller source artifact.
- [x] Convert canonical full-tile package and indexed-runtime ceilings into an
  exact plane-cardinality budget, solve verified Hessian prefix curves with a
  callback-driven compact allocator, and durably bind nested two-bit maps. The
  solver retains exact non-concave global optimality and deterministic ties;
  ragged or runtime-unindexable geometry fails closed.
- [x] Bind the Rust SALT pipeline to explicit `prepare` → `calibrate` → `convert`
  phases without a Python-list weight bridge. `quantize(model_or_id, ...)`
  composes those exact primitives. The Qwen seam admits canonical precomputed
  evidence. Ordinary PyTorch modules now stream bounded, source/data-bound
  diagonal second moments into an atomically published, strict-reload evidence
  namespace. Public `convert` now fits source/recipe/coverage-bound hard
  additive planes from that evidence, including stored-scale-correct residual
  fitting. Generic conversion now checkpoints each fitted weight independently,
  row-tiles the float64 fitter under an explicit auxiliary-tensor byte budget,
  streams plane chunks directly to durable files, strict-reloads every
  digest/ledger, lazily loads one weight at a time, resumes missing weights after
  restart and atomically seals recipe-bound state. Schema-v2 writes retain a
  tested strict reader for schema-v1 artifacts. Generic reload now binds the
  source digest and exact module aliases into inference-only Linear modules
  holding B3-packed additive trits plus f16 row scales, with no latent master or
  persistent dense decoded weight. This Python path is a semantic reference
  fallback and its transient decode is not native fused/no-shadow evidence.
  G128-aligned fitted weights now stream one tensor at a time into a strict,
  seek-backed SALT V2 package; conversion sources and packed semantics are
  reverified before atomic publication. Complete-model preservation, native
  fused dispatch and stronger factorized curvature remain open.
- [x] Expose immutable, exact-ledger `ArtifactRef`, `QuantizationResult`, and
  `ExportReceipt` records plus atomic matrix-bundle export. `complete_model`
  remains false because vision is intentionally outside the language-plus-MTP
  artifact; it is not used as a proxy for runtime readiness.
- [x] Strictly reload the two exact Tritium matrix packages by re-parsing,
  re-hashing, and re-deriving indexed-runtime bytes through the native reader.
- [x] Promote preserved tensors and the pinned, bounded, content-bound HF
  language asset catalog into the same atomically published schema-v3 directory.
- [x] Implement atomic `save_pretrained`, strict evidence-only reload, and direct
  device load of schema-v3 directories into the Qwen3.6 packed runtime. The
  native loader derives identities from `tritium.json`, verifies all eight HF
  assets and exact physical ledgers, assembles language plus MTP without dense
  matrix shadows, and supports token-ID greedy generation. CPU and feature-gated
  CUDA abi3 wheels were built and imported from fresh environments. On an RTX
  4090 the synthetic full graph streamed every matrix into final CUDA SALT V2
  allocations, emitted `device_resident_salt=true`, and generated successfully;
  the CPU wheel rejects CUDA placement explicitly. This is synthetic
  graph/packaging/device evidence; the pinned 27B run, optimized large-K kernels,
  and MTP production-oracle promotion remain open.
- [x] Remove refinement from `TernaryConfig.ptq`. PTQ can produce only a PTQ
  result; `RefinementConfig` owns separately versioned scale-only and hard-PV
  recipes. Plan 0048's `refine` primitive and bound child ancestry remain open.

## Step 4 — phase and schema closure

- Freeze canonical Rust/Python representations for prepared state,
  calibration receipts, conversion results, artifact references, export
  receipts, coverage and errors; TypeScript fixtures land before plan 0050.
- Require explicit `inplace` on `prepare`; validate before mutation and preserve
  rollback. QAT supports additive plane counts 1, 2 and 3 before v1.1.
- Prove primitive and convenience paths emit identical recipe, coverage and
  artifact identities.
- Treat existing `prepare_qat` as a convenience compatibility seam, not the
  sole stable lifecycle.

## Verification

```bash
PYTHONPATH=crates/tritium-py/python pytest -q crates/tritium-py/tests
python -m compileall -q crates/tritium-py/python/tritium
cargo fmt --check
git diff --check
```

## Review

After each commit, call lamu `review_commit` with this plan as `plan_file` and
verify every finding before applying it.

## Done criterion

Tiny and representative Hugging Face causal LMs pass independent phased QAT
and PTQ lineages, Trainer/Accelerate, DDP/FSDP, resume, atomic export/reload and
generation gates. Refinement remains a separately typed plan-0048 lineage.
