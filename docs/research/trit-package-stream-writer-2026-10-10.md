# Shared deterministic `.trit` streaming writer candidate

Proposed ADR0058 continuation on 139b3fcff25e0873570afacdefb56c9374865add.
The existing Vec writer and new caller-owned Write adapter share one preparation
and chunk emission module: original canonical metadata, content deduplication,
ordering, offsets, alignment, padding, header and footer. The streaming adapter
uses the existing PackageHasher and returns identity only after complete emission.
No complete output Vec is materialized by the adapter. Borrowed payloads and
directory/layout metadata are still retained; this is not full RSS or spooling.

Complete-size and structural admission finish before the first destination write.
Short writes and Interrupted are handled by write_all; WriteZero and other faults
return explicit partial-I/O errors and no identity. It does not seek/flush/sync,
overwrite, clean up or publish for callers. Production adoption must remain behind
existing owned pending files and no-clobber publication/binding checks.

Five software tests assert buffered/stream parity, dedup/order/empty/alignment,
one-byte writes, Interrupted recovery, exact pre-write bound rejection, partial
prefix failures, an 8 MiB borrowed payload to an incremental non-retaining sink,
and actual owned file I/O with cleanup. A separate probe includes the exact
pre-change writer source from commit 139b3fcf (not a hand-written old encoder),
comparing actual bytes and PackageId against both adapters over 17 cases.
Probe source, baseline and dependency lock are retained as verification controls.
Corrected guarded invocation 0ac3cf25c00f4ba3a4d4a307546b3297 is terminal success
(exit 0), with all three production-source and four probe-control hashes unchanged.
Five streaming tests and eight filtered format tests passed, as did all 17 actual
pre-change writer byte/PackageId comparisons, strict all-target format Clippy,
strict Rustdoc and workspace formatting. The initial probe check failed because
its temporary Cargo manifest omitted the direct schema dependency; the original
failure and corrected controls/results are preserved separately. Production
source was unchanged between those checks. No tolerance or gate was weakened.

Remaining: exact committed public compatibility/commit gates, actual engine/cache
adoption, bounded/spooled/mmap source and encoded payload access, independent full
RSS admission, row-specific calibrated fitting/evidence/package replay and all
v1.1 frontier/backend/browser/Qwen/MTP/quality/reproduction/serving/packaging/zoo/
security/community/human activation obligations. No wire/profile/quality cap or
golden tolerance is weakened and no empirical release verdict is assigned.
