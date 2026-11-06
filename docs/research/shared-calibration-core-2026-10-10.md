# Shared calibration collector candidate

This implements step 1 of proposed ADR0060 (shared calibration and metric
binding), on parent source `b1b9cce5c0bc5a820fe1ab3af9e8a44ebd3a4e41`.
It is not full P3 completion, an accepted profile freeze, or a release receipt.

## Implementation

`tritium-schema` owns the candidate plan and factor binding vocabulary; its
existing Rust generator derives seven JSON projections. Numerical arrays stay
inside `tritium-quantize`. `CalibrationCollector` orchestrates the existing
source-bound InputGram and dense/indexed OutputFisher reducers and the existing
row-specific K-FAC builder. It does not copy those numerical reductions,
selection leaves, content digest domains, or K-FAC mathematics.

The collector supports full original-input covariance or explicitly grouped
block-diagonal statistics, including ragged groups and widths above the matrix
width. Statistics groups do not impose legacy S2KF G128 or aligned-column
restrictions. Matmul input and an explicit gather identity approximation are
distinct; gather requires indexed output factors without vocabulary one-hots.
Absent Fisher is explicit and cannot silently become Identity or invented
gradients. Diagonal folding reads the same finalized Gram matrices.

Paired accumulation and shard merging are transactional. Invalid masked
activations, gradients, weights, shapes, ordinals, overlapping ranges, source
identities, or segment limits leave both input and output factors unchanged.
Canonical finish retains the existing batch/resume/shard-order-independent
identities. Read-only finalized factors expose the request actually used and
the actual existing numerical content and sample-selection identities.

## Admission and provenance limits

The numerical workspace bound accounts conservatively for finite reduction
segments, old/transactional/inner copies, merge/finalization buffers, grouped
batch scratch and one row's metric workspace. It rejects insufficient budgets
before state allocation or batch mutation; it never changes a full request to
a grouped approximation. It is deliberately **not** full RSS admission: caller
buffers, metadata/allocator overhead, source/cache/tokenizer, other fitter or
Rayon state, device/driver, KV, and matrices retained by the caller are outside
this bound. No measured peak-memory or memory-reduction claim is made.

The plan binds declared finalized source/cache/token identities but does not
independently verify their underlying bytes. Existing reducers incorporate the
source identity into each selection leaf. A live producer therefore must first
finalize and verify a bounded immutable cache/spool and replay it; it may not
invent a provisional identity or rebind completed factors afterward.
The serialized factor view is not a registered evidence event, a sealed
package, or independently cleared provenance. Private finalized runtime fields
prevent callers from substituting factor values behind that view.

## Checks and remaining integration

The focused test seam is the real public collector and actual existing reducers,
not a copied oracle:

```sh
cargo test --locked -p tritium-quantize --test calibration_collector
cargo clippy --locked -p tritium-schema -p tritium-quantize --all-targets -- -D warnings
cargo run --locked -p tritium-schema --features schema-gen --bin tritium-schema-projections -- --check
```

The 16 focused cases cover bit-exact factor/diagonal/row-KFAC parity,
full/grouped/ragged/oversized partitions, dense/indexed factors, gather,
weights/masks, invalid paired updates, resume/reverse shard order, source and
overlap rejection, budget/geometry/ordinal overflow, zero weight, destination
atomicity, nested-field JSON rejection and the finite shard-state limit.
Committed-tree regression, strict docs, no_std serialization and canonical
prepush are separate checks; terminal records belong in the project archive.
Passing these software checks would not establish model quality, physical GPU
or browser behavior, performance, total memory, independent reproduction, or
public activation.

Next is ADR0060 step 2: one actual caller-owned forward/tape tap mechanism with
OProj/Head and real post-objective-backward gradients, preserving old logits
and diagonal parity. Step 3 connects original source-byte inventory, calibration
binding, row-aware direct fitting, registered observations and sealed package
replay, with swap-rejection tests. Step 4 migrates CLI/Qwen/Python/HF adapters
through those same interfaces and verifies legacy fixtures before retirement.
No legacy S2KF bytes, producers, fixtures or campaign identities change here.

Shared shipping-ladder feedback parity, full f64 joint fitting/direct allocation,
frontier solver/refinement/GPU/out-of-core, frozen profiles/all projections/full
RSS/streaming/mmap/shards/entropy and migration remain. Native backend/resident
execution/Qwen hybrid/MTP; Stage7 and official language+MTP quality/runtime/bytes/
independent reproduction; physical backend/browser/WASM/MCU/distributed training;
serving/deployment/security; audited zoo/RC/ONNX/Colab/docs/community/independent
signing and explicit human public activation remain release obligations.
