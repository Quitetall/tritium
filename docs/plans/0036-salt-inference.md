# 0036 — Run a SALT-quantized model (`load_salt` → dequant-to-dense)  (serves: ADR 0020 step 2)

## Goal
Bridge the producer→consumer gap: `read_salt_bundle`/`read_salt_gguf` → a runnable model. Load
what `tritium quantize` emits (a SALT bundle of residual ternary planes) and run it. Unblocks the
quantize→run round-trip and the reusable dequant-SALT-weights building block the distillation
student forward (0038) needs.

## Key facts (exploration)
- SALT bundle = `Vec<SaltTensor>{name, rows, k, salt_rows}`; each `SaltRow.planes` = T=1..3 TQ2_0
  planes with **per-256-block** f16 scales. `dequant_salt_row` = `Σ_p Σ_block scale·trit`.
- Native multi-plane matmul exists on **GPU** (`CudaBackend::upload_salt`); on **CPU** only
  dequant-to-dense. dequant-to-dense + int8 activations is **bit-equal** to native multi-plane
  (the weight scale is a post-multiply that distributes over the plane sum).
- Bundles carry **only 2D ternary weights — no norms, no config.** The loader sources 1D norms +
  `config.json` from the original model dir.

## What shipped
- `tritium_format::salt_rows_to_dense(&[SaltRow]) -> Vec<f32>` (crates/tritium-format/src/salt.rs):
  row-major `[rows, k]` dense, concatenating `dequant_salt_row`. Reused for both a bundle's
  `SaltTensor` and a live `QuantizedTensor.salt_rows` (0038).
- `build_standard_model(config, spec, provider: Fn(&str)->Vec<f32>, exact_fp)` extracted from
  `load_hf` (crates/tritium-nn/src/model/hf.rs) — shared assembly; `exact_fp` picks
  `DenseLinear::new_exact` (fp) vs `new` (A8 int8-activation, deployed semantics).
- `ModelWeights::load_salt(model_dir, bundle)` / `ModelRunner::from_salt(...)`: ternary 2D weights
  from the bundle (dequant, A8), 1D norms from `model_dir`'s safetensors; same rope_scaling /
  qk_norm / qkv_bias guards as `load_hf`.

## Gate (met)
`load_salt_dequants_bundle_and_runs` (lib test): quantize a tiny model's 2D weights → `write_salt_bundle`
→ `from_salt` → asserts the loaded `q_proj` weights equal `salt_rows_to_dense` of the bundle (correct
wiring) and a forward yields finite vocab-length logits (runs). `salt_rows_to_dense` unit test.
No `load_hf`/BitNet regression (SmolLM2 conformance still 16/16); fmt + clippy `-D --all-targets` clean.

## Non-goals (follow-ons)
- **GPU native multi-plane wiring** (`read_salt_bundle` → `upload_salt` → resident SALT projection) —
  the VRAM win; matters at 32B → plan 0040. CPU dequant-to-dense proves correctness now.
- **Self-contained bundle** (quantize also emits norms + config) → `from_salt` needs only the bundle.
- **Eager dequant** of all bundle tensors into RAM — fine for small models; 32B streaming is plan 0040.

## Post-plan hardening (2026-07-14)

The CPU inference baseline now retains projection weights as packed additive planes instead of
keeping an `N × K` fp32 matrix:

- `SaltBundleIndex` validates the complete TSLB once, provides O(1) named lookup, and decodes only
  the requested tensor. Duplicate names, invalid UTF-8, overflowing lengths, and corrupt payloads
  in unselected tensors fail closed.
- `SaltLinear` reconstructs one 256-weight block at a time and contracts through the existing A8
  activation path. Its output is bit-exact to `salt_rows_to_dense → DenseLinear::new`, including
  ragged rows and zero-plane rows, while retaining only packed planes.
- `ModelWeights::load_salt` uses `Projection::Salt` for every attention/MLP projection and an
  untied LM head for both TSLB and SALT-GGUF. The embedding and tied head remain dense.

This does not yet make 32B loading or serving production-ready. The loader still reads the whole
bundle and fp master shards, SALT-GGUF decoding is eager, progressive sparse planes expand into the
dense TQ2 runtime representation, and SALT projections do not enter the resident CUDA decoder.
