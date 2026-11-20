# Original-source content identity candidate

Continuation of proposed ADR0060 on `3fa7b54b381704e6c0cfc68d59ef4099815a3763`.
This is an additive implementation candidate, not a frozen profile or release gate.

`OriginalSourceId` is distinct from fitted ModelId/PackageId. One implementation
in evidence validates and incrementally hashes the complete sorted original
inventory: architecture, exact names/geometry/stored precision, original tensor
raw-byte digests and exact asset bytes. It requires no encoded metadata allocation
and adds no dependency to the schema/evidence crate graph. The candidate hash
domain/preimage is documented beside that implementation and in proposed ADR0060.

`Engine::observe_source` reads actual source bytes under the declared logical
source/asset limits without fitting, widening or retaining tensor payloads.
Conversion and receipt replay derive the same identity from actual engine-recorded
source bindings using the common implementation. Replay verifies recorded content
bindings, not independently accessible original source bytes. Existing event
versions, sealed roots, package formats and fitted identities are unchanged.

Changing original shape, precision, name, tensor bits or exact asset bytes changes
the identity. Container paths/offsets/shard layout and fitting recipes/levels do
not. A pre-calibration observation does not lock mutable files: observation/run
identity equality, immutable-source checks and actual calibration admission must
still be enforced before publication. This change intentionally does not relabel
legacy raw Gram requests as audited calibration.

Remaining ADR0060 work includes objective/cache/token/selection verification,
source equality admission, typed metric observations, row-specific KFAC fitting,
sealed package replay/tamper rejection and complete adapter/fixture migration.
Hybrid/MTP coverage, full RSS/streaming/GPU/out-of-core, frozen projections and
all physical/model-quality/reproduction/public activation gates remain open.

The guarded candidate check completed exit 0: 36 evidence/schema/model-engine
tests passed, including both original source adapters and byte/interpretation/
inventory/limit mismatches; generated projection/drift, scoped strict Clippy/docs
and bare-metal alloc/serde schema checks passed. This does not freeze the domain
or promote the candidate to calibration/package admission. Full committed gates
and publication remain separate checks.
