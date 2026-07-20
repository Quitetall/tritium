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
- `tritium.torch.prepare_qat`
- Hugging Face `AutoModelForCausalLM.from_pretrained`
- `tritium.torch.quantize`, `QuantizationResult`, and `load`

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
- Bind the Rust SALT pipeline to `quantize(model_or_id, calibration, config,
  work_dir)` without a Python-list weight bridge.
- Expose immutable `QuantizationResult` model/coverage/report plus atomic
  `export` and `save_pretrained`.
- Implement `load` for exact Tritium packages and supported HF directories.
- Keep `refinement="none"` PTQ and refined runs in distinct work identities,
  receipts and claims.

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

Tiny and representative Hugging Face causal LMs pass QAT, Trainer/Accelerate,
DDP/FSDP, resume, PTQ/refinement, atomic export/reload and generation gates.
