# Calibration cache writer export candidate

Implements proposed ADR0060's export adapter over ADR0058's shared `.trit`
writer. This is software work, not full-process memory, model quality,
performance or release qualification.

`CalibrationCache::write_package_to(max_package_bytes, destination)` shares
manifest/scalar/blob preparation with `encode_package`; it calls the existing
format writer, not a copied layout or hash. It returns the exact physical
PackageId only after successful emission. The immutable cache, scalar staging
and metadata are retained, but no complete output package is allocated by this
adapter. Buffered export retains its existing contract and bytes.

Logical payload and complete package admission happen before any destination
write. A typed `CalibrationCacheError::Write` exposes common package admission
or I/O failure; the latter may leave partial caller output. It does not seek,
flush, sync, publish or clean up caller files. Production callers must keep
output unpublished until their existing no-clobber commit point.

Candidate checks cover all four cache encodings, short writes and Interrupted,
byte/PackageId/reconstructed-binding identity, complete-bound rejection without
mutation, BrokenPipe/WriteZero at multiple actual prefixes, a nonretaining
destination and real owned-file export/readback/cleanup. Guarded invocation
a857c2b98bae4308a3b0f324c182c3f1 is terminal exit0 on base
d5ec6a3c1156098389e5b4df469f8edca86067a5, with six source hashes and HEAD
unchanged at both guards. Seven transport, four finalized-cache and five format
stream tests passed, as did strict all-target format/quantize Clippy, strict
Rustdoc and workspace formatting. The shared writer's earlier 17 comparisons
against actual pre-change source were separately executed successfully before
integration. These are scoped software checks, not release qualifications.

Exact committed-tree broad verification and normal public push remain pending.

Source/window spooling, streaming decode, mmap, aggregate RSS admission,
registered row-specific calibrated metric fitting/log/package replay and all
empirical release obligations remain open. No legacy cache identity, envelope,
recipe, numeric tolerance or release gate changes.
