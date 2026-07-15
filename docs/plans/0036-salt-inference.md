# 0036 — Run a SALT-quantized model with packed additive weights  (serves: ADR 0020 step 2)

## Goal
Bridge the producer→consumer gap: `read_salt_bundle`/`read_salt_gguf` → a runnable model. Load
what `tritium quantize` emits (a SALT bundle of residual ternary planes) and run it. Unblocks the
quantize→run round-trip and the reusable packed-SALT execution building block the distillation
student forward (0038) and 32B scale path need.

## Key facts (exploration)
- SALT bundle = `Vec<SaltTensor>{name, rows, k, salt_rows}`; each `SaltRow.planes` = T=1..3 TQ2_0
  planes with **per-256-block** f16 scales. `dequant_salt_row` = `Σ_p Σ_block scale·trit`.
- Native multi-plane matmul exists on **GPU** (`CudaBackend::upload_salt`). The CPU packed
  reference reconstructs one 256-weight block at a time and is bit-equal to the former dense
  oracle under both exact tied-head and A8 projection semantics.
- Bundles carry **only 2D ternary weights — no norms, no config.** The loader sources 1D norms +
  `config.json` from the original model dir.

## What shipped
- `tritium_format::salt_rows_to_dense(&[SaltRow]) -> Vec<f32>` (crates/tritium-format/src/salt.rs):
  row-major `[rows, k]` dense, concatenating `dequant_salt_row`. Reused for both a bundle's
  `SaltTensor` and a live `QuantizedTensor.salt_rows` (0038).
- `build_standard_model(config, spec, dense, projection)` extracted from `load_hf`
  (crates/tritium-nn/src/model/hf.rs) — shared assembly. The dense provider receives expected
  vector lengths; the projection provider selects exact fp, A8, packed SALT, or backend storage.
- `ModelWeights::load_salt(model_dir, bundle)` / `ModelRunner::from_salt(...)`: packed ternary 2D
  weights from the bundle and 1D norms from `model_dir`'s safetensors. Rope-scaling guards match
  `load_hf`; QK-norm and QKV-bias are detected in the master shards and rejected until the SALT
  Qwen extension in plan 0037 lands.

## Gate (met)
`load_salt_retains_packed_weights_and_runs` exercises both TSLB and SALT-GGUF, tied and untied
heads, and compares full logits with the dense oracle. Dedicated format, `SaltLinear`, and
`TokenEmbedding` tests cover sparse/dense preservation, ragged K, malformed geometry, gather, and
exact tied-head accumulation. The real-model CPU fidelity and greedy-decoding acceptance tests
remain the end-to-end regression gate.

## Non-goals (follow-ons)
- **GPU native multi-plane wiring** (`read_salt_bundle` → `upload_salt` → resident SALT projection) —
  the VRAM win; matters at 32B → plan 0040. The packed CPU reference proves correctness now.
- **Self-contained bundle** (quantize also emits norms + config) → `from_salt` needs only the bundle.
- **Mapped/streamed bundle input** — still required for 32B; plan 0040.

## Post-plan hardening (2026-07-14)

The CPU inference baseline now retains projection weights as packed additive planes instead of
keeping an `N × K` fp32 matrix:

- `SaltBundleIndex` validates the complete TSLB once, provides O(1) named lookup, and decodes only
  the requested tensor. Duplicate names, invalid UTF-8, overflowing lengths, and corrupt payloads
  in unselected tensors fail closed.
- `SaltLinear` reconstructs one 256-weight block at a time and contracts through the existing A8
  activation path. Its output is bit-exact to `salt_rows_to_dense → DenseLinear::new`, including
  ragged rows and zero-plane rows, while retaining only packed planes.
- `ModelWeights::load_salt` uses `Projection::Salt` for every attention/MLP projection and any
  untied LM head for both TSLB and SALT-GGUF.
- `PackedSaltRow` readers preserve progressive dense/sparse plane choices from either TSLB or
  SALT-GGUF when present. `PackedSaltMatrix` flattens them into private dense-byte, sparse-scale,
  and signed-index arenas; sparse residuals no longer expand to dense TQ2_0 at runtime. The current
  GGUF quantize writer still emits dense planes, so sparse GGUF production remains follow-on work.
- `TokenEmbedding` retains one dense-or-packed table. SALT embedding gather and the exact tied LM
  head share that packed allocation, removing the retained `vocab × hidden × fp32` matrix. The
  exact path reconstructs combined weights in plane order and accumulates in global-K order;
  non-finite runtime scales fail closed during matrix construction.
- The current resident CUDA decoder accepts only dense tied token tables and now explicitly rejects
  untied heads. Packed SALT tables remain on the correct host path until resident kernels land.

The file-backed `load_salt` fp-master userspace floor is now removed:

- `SafeTensorsReader<R: Read + Seek>` parses and retains only header metadata beyond its caller-owned
  reader. Requested BF16/F16/F32 tensors are absolute-seek-read through at most 64 KiB of raw
  staging and widened directly into their final fp32 vector. The file-backed HF adapter therefore
  does not retain complete master shards or request unselected payload ranges. Bounded header and
  output reservations plus header, layout, offset, shape, seek, and short-read failures are typed;
  JSON parser-internal allocation remains allocator-controlled.
- Both borrowed and seek-backed parsers reject duplicate names, invalid metadata, holes, overlaps,
  malformed unselected tensors, and trailing unindexed bytes during construction.
- `HfShardSet` indexes actual tensor names across standard HF shard indexes and powers both
  `load_hf` and `load_salt`. Index bytes, total tensors, tensor-name bytes, shape dimensions, rank,
  and shard count are bounded; every index mapping is verified against the selected shard.
  Duplicate tensor names, non-string entries, and absolute or parent-traversing shard paths fail
  closed. Operator-provided symlinks remain trusted so standard Hugging Face cache snapshots
  continue to work.
- The model builder uses named dense requests carrying each vector's expected length. The HF
  adapter checks exact metadata rank and shape before allocation or payload I/O, closing the prior
  wrong-rank/oversized-norm hole. Projection matrices are likewise checked against `[n_out, k_in]`
  before reading.

This still does not make 32B loading or serving production-ready. The loader reads the whole SALT
bundle; TSLB decoding transiently holds each requested packed tensor before flattening it into final
arenas; SALT-GGUF eagerly decodes every packed tensor; and SALT projections/token tables do not enter
the resident CUDA decoder. The next memory-floor slice is direct-arena or mapped bundle input,
followed by resident large-K SALT CUDA execution.
