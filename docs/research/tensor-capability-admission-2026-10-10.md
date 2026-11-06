# Semantic tensor capability admission — 2026-10-10

Code: `25592183245411c4d85439c867f74107d93314e8`.
Base: `a818bdfa2a89ca5dfb6414b5cb062350bc821e2a`.
Scope: ADR 0044 D8/D9 prerequisite implementation, not P2 or release clearance.

## Shared contract and real adapters

Rust schema types define `TensorExecution::{Native, Emulated}`, `TensorCaps`
and `TensorUploadPolicy`. Their JSON Schema projections are generator-owned.
No metadata encoding, scale law, plane limit, numerical default or existing
physical tensor payload changes. The lockfile adds one existing workspace
schema dependency edge to the spec crate; no dependency versions change.

`TernaryBackend::tensor_caps(view)` queries the actual tensor combination;
its default returns `None`, irrespective of ISA flags or legacy packed support.
`upload_tensor_checked` applies one shared, object-safe guard: reject undeclared
support, reject non-native execution under native-only, and enforce the caller's
payload budget before upload. After upload, the actual buffer byte count must
match the declared size. The returned tier is explicit for evidence consumers.

CPU and reference declare direct scalar execution for the current laws and
G32/G128. Native means direct additive execution (or originally dense input),
**not** packed SIMD, accelerator performance or empirical qualification. The
schema's execution-group query keeps serializable G64/G256 out of this checked
path. Other backends retain an undeclared/fail-closed default.

Canonical additive scale-precision validation is now reusable without allocating.
Queries reject inexact F16 scales and negative zero before payload reservation;
`from_view` also validates before copying. Both owned constructors retain the
same exact stored-scale checks and ternary ownership, without dense conversion.
Dense payload sizing validates geometry and checked products in the shared spec.
The codebase-design skill guided shared policy/validation rather than copies in
each backend or surface.

## Verification and boundaries

```sh
cargo test --locked -p tritium-schema -p tritium-core -p tritium-spec \
  -p tritium-format -p tritium-testkit -p tritium-cpu -p tritium-runtime --all-targets
cargo test --locked -p tritium-spec -p tritium-schema -p tritium-format \
  -p tritium-cpu --doc
scripts/verify-gates.sh prepush
```

Committed regression suite: **492 passed, 0 failed, 6 ignored**. The skips remain
one CPU microbenchmark and five real-Qwen/artifact-dependent probes. One format
doctest passes; the other three named crates contain zero doctests.
Scoped Clippy passes with warnings denied.

Three new CPU consumer tests include 60 frozen combinations on each of CPU and
reference (120 executions): all current laws and their plane counts, G32/G128,
Identity/Hadamard/SignedRht, two rows, multiple groups and a ragged last group.
Widths are 68/260 and transformed blocks are 4 with signed seed 7/domain 9.
These are decoded D2-declared views, not new all-codec physical residency or
arbitrary basis/block/seed qualification. Budget equality/zero-byte boundaries,
unknown groups, invalid dense geometry and invalid exact scale precision are
also checked. Two spec policy tests use controlled adapters to establish zero
uploads on rejection and explicit tier/size-mismatch behavior; they are not an
implemented or qualified accelerator emulation tier.

The first probe fails at absent contract types/methods. Two fixture defects are
retained: repeated SignedRht signs instead of whole-axis indexing, and an f32
expected-value reduction losing 0.001 at approximately 849. Corrected frozen
sign bits and f64 expected reduction fix the test oracle; production arithmetic
and the test's 1e-3 absolute threshold are unchanged. An initial collapsible-if
Clippy failure is fixed without weakening warnings.

Canonical prepush invocation `7cea752bab38410691d062389ad5d03a` terminates with
`MainPID=0`, `SubState=exited`, `Result=success`, `ExecMainStatus=0`. Formatting,
generated projection drift, actual bare-metal compilation, default/all-features
Clippy, Windows GNU cross-check and actionlint pass. Local ShellCheck is absent
and explicitly warned, not claimed passing.
All five hosted workflows at the previous source `a818bdfa` succeed. Fresh
same-source hosted checks remain required for this change.

Archive: `/home/brianklam/Projects/Tritium/archive/verification/tensor-admission-25592183-20261010`.
Retained-byte/source custody does not establish empirical release clearance.

## Remaining full goal

The raw upload method stays policy-unchecked during migration. CLI/NN/serving
do not yet call this checked interface. Dense emulation implementations, typed
evidence events, aggregate physical-memory admission and all-backend conformance
remain; a payload budget excludes allocator/handle overhead, scratch, KV and
other tensors. Do not present it as complete model-memory admission.

Complete packed/per-tile execution and SIMD residency, all schema/projections/
identities, streaming/shards/fuzzing, unified engine/calibration/direct fits,
loader/resident runner/surface adapters and legacy migration/deletion. P0–P7,
full Stage-7, Qwen language/MTP quality/runtime/physical-byte/reproduction,
physical backend/distributed/browser training, serving/security, final packages,
audited zoo/docs/community, independent clearance and explicit human release
authorization remain open. No large campaign, cloud job or activation occurs.
