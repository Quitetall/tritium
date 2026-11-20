# Candidate durable calibration-cache transport

Proposed ADR0060 continuation on 139b3fcff25e0873570afacdefb56c9374865add.
The existing `.trit` envelope owns physical layout, deduplication, canonical CBOR,
blob verification and PackageId. Schema owns CalibrationCacheManifest and its
derived projection; format owns the typed canonical manifest codec. Quantization
owns transport of actual immutable payloads, reconstructed through the same
CalibrationCacheBuilder and existing identity/reducer implementation.

The two-item manifest profile is `["tritium.calibration-cache.v1", manifest]`.
Metadata is capped at 1 MiB. The manifest holds the actual claimed binding and
ordered nonzero context sample counts. Exact tokenizer/objective bytes, tokens,
f32 inputs/dense gradients or interleaved u64 row/f32 indexed factors, f64 weights
and byte 0/1 masks are named Asset blobs. Scalars are little-endian; signed zero
bits are preserved. Gather omits inputs and input-only omits output blobs.

Decode supplements original producer admission with receiving package/payload/
sample/window limits before payload copies. Complete named inventory, lengths,
scalars, context identities and stored binding are checked; a caller-provided
expected cache identity must agree with actual reconstructed payloads. Raw envelope
hash verification alone is insufficient: correctly rehashed but swapped buffers,
context metadata or recipe bytes must still reject.

The four candidate tests cover all four supported operation/output combinations,
multiple context windows, full/grouped/ragged same-cache chunked replay, exact
bit/identity round-trip, actual file I/O with owned-file cleanup, receiving limits,
unknown profile, truncation, actual envelope corruption and 18 valid-envelope
semantic/inventory/scalar tamper cases. The first check passed these tests but
failed strict Clippy on two constant chunks_exact iterators; its exit-101 log is
retained. After terminal confirmation only, both were changed to as_chunks without
suppression or semantic/bound/tolerance changes. Corrected invocation
732800f4ab304075bed5cdc3bbff33f5 completed exit 0 on unchanged source: four transport,
four existing cache and eight filtered format cases passed, as did manifest drift,
strict scoped Clippy/docs, bare-metal alloc/serde schema and fmt. These scoped
software results do not establish combined committed gates or qualification.

This initial whole-package transport retains staging/output/caller buffers and
does not establish spooled/mmap/streaming or aggregate metadata/allocator/RSS
admission. It adds no binary magic, copied hash/reduction algorithm, independent
receipt or empirical claim. Required next work remains bounded streaming/spooled
production/replay, immutable source/data/typed recipe verification, registered
row-specific fitting/metric observations and actual package replay, full tensor
inventory/surface migration, and the complete v1.1 hardware/quality/reproduction/
serving/packaging/zoo/community/security and human activation gates.
