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

At this checkpoint, 32B loading was not production-ready: both TSLB and SALT-GGUF still had eager
copies and SALT projections/token tables did not enter the resident CUDA decoder. The next two
sections record removal of the TSLB and legacy SALT-GGUF copies; the CUDA limitation remains.

## Direct-arena TSLB loading (2026-07-15)

The TSLB model path now removes both remaining host-side bundle copies:

- `SaltBundleReader<R: Read + Seek>` strictly scans every tensor once, retaining only bounded owned
  metadata, per-tensor BLAKE3 payload digests, and exact final-arena element counts. It rejects
  duplicate names, malformed selected or unselected rows, inconsistent shape, truncation, trailing
  bytes, arithmetic overflow, and explicit tensor/index/name/row/plane/encoded-row resource-limit
  violations. Metadata lives in one fallibly reserved, name-sorted vector with binary search; there
  is no infallibly growing tree index. Strict construction deliberately performs total payload I/O;
  it is bounded-memory, not lazy-I/O. The model-safe policy permits at most eight planes per row and
  a 16 MiB encoded row, so it can reject low-level writer output outside the inference envelope with
  a typed `LimitExceeded` error.
- Named tensor visits use absolute seeks and one reusable encoded-row buffer. Individual `Read`
  requests are capped at 64 KiB; file-backed loading adds a 64 KiB `BufReader` to avoid one syscall
  per row header. A complete visit rechecks the payload digest, detecting valid same-length mutation
  through the already-open source handle. Callbacks can precede a late read or digest failure and
  are not rolled back; transactional callers stage the destination until the visit returns `Ok`.
- Allocation-free `PackedSaltRowRef` and `SparsePlaneRef` views let `PackedSaltMatrixBuilder` copy
  each row directly into pre-sized final dense-byte, sparse-scale, sparse-entry, row-metadata, and
  plane-metadata arenas. It preflights validation and exact element requirements before mutation,
  requests final capacity up front, performs no per-row owned decode, and fails if construction-time
  requirements differ from the second pass.
- `PackedSaltMatrix` clones now share immutable arenas through `Arc`; tied or otherwise cloned views
  do not duplicate model weights. `resident_bytes()` reports the shared backing size and therefore
  must not be summed across clones.
- `config.json` reads are regular-file-only and capped at 16 MiB. HF/GGUF dimension conversion,
  bounded/even geometry, finite positive RoPE/RMS scalars, mistyped optional fields, unsupported
  architectures, and unsupported activations fail closed. Layer metadata reserves fallibly.
- A dedicated allocation-tracking regression strictly scans a borrowed bundle larger than 20 MB,
  including a large unselected tensor, and requires less than 4 MiB of allocations. This catches
  reintroduction of a whole-bundle copy; direct-builder tests separately require stable final-arena
  pointers and shared `Arc` backing.

The current dense TQ2_0 plane costs 2.0625 bits per weight, so one plane for 32 billion weights is
about 8.25 GB of payload. With an illustrative 4.8 million rows, current row/plane metadata adds
about 0.23 GB, for roughly 8.48 GB resident and 7.55× compression versus 64 GB FP16. Additive models
scale with retained plane count; sparse residuals can be smaller. The current format therefore does
not support a physical “more than 10×” claim. Crossing that threshold requires radix-3 or other
entropy coding, better scale amortization, and compact or implicit dense-plane metadata, with quality
measured separately.

## Direct-arena legacy SALT-GGUF loading (2026-07-15)

The legacy SALT-GGUF compatibility path now has the same bounded-memory consumer contract:

- `SaltGgufReader<R: Read + Seek>` streams the GGUF metadata and tensor table, validates canonical
  offsets and zero alignment gaps, and strictly scans every private SALT tensor. Sized standard
  tensors are row/block-layout-validated and ignored; unknown unsized types fail closed. Homogeneous
  nested metadata arrays are supported consistently by eager, writer, and seek paths with shared
  depth/element bounds. BOOL bytes are canonical. A declared `general.alignment` must be a nonzero
  U32 multiple of eight; only an absent key defaults to 32. Any non-finite, negative, or signed-zero
  SALT scale fails construction even in an unselected tensor. Explicit bounds cover header,
  alignment, tensor, metadata, string, name, dimension, row, plane, and encoded-row resources.
- Construction retains only name-sorted SALT metadata, exact final-arena requirements, and BLAKE3
  payload digests. Unselected SALT tensors are fully parsed, so corruption cannot hide behind named
  lookup. Named visits use absolute seeks, one reusable row buffer, and at most 64 KiB per underlying
  `Read` request, then recheck exact length and digest. Callbacks can precede a late error and are not
  rolled back; transactional callers stage the destination until the visit returns `Ok`.
- `ModelWeights::load_salt` passes both TSLB and SALT-GGUF row references directly into
  `PackedSaltMatrixBuilder`; the GGUF path no longer calls `read_to_end`, retains a whole-artifact
  buffer, constructs per-row owned `PackedSaltRow`s, or keeps an all-tensor `HashMap`.

Both containers still read all SALT payload bytes twice when every tensor is selected, and warmed
file cache can appear in cgroup physical-memory peaks even though anonymous heap stays row-bounded.

Host fp32 K/V vectors now start empty and grow fallibly in 64-row chunks, so an artifact's declared
maximum context no longer becomes an unchecked eager allocation. The full-context physical cost is
unchanged: for 64 layers and KV width 1024 it is 17.18 GB at 32K context and 68.72 GB at 128K.
Production 32B serving therefore still requires, in order: compact SALT payload/metadata, resident
large-K CUDA SALT kernels, and a paged fp16 or int8 KV cache with a runtime context override.
