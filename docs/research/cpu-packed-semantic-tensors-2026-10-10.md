# CPU packed semantic tensor execution

## Scope and source

ADR 0044 D2/D8/D9 requires native overrides behind the shared tensor contract,
with truthful payload accounting and frozen conformance. Base:
`f32fbacb7e64e923a43e60890425d12bb8293ed3`. Code:
`c68fb6545bf63e0c2a53c78e2c5996a622459e31`.

The production CPU tensor operations now retain canonical D2/B3/S34 plane bytes
and exact stored-precision f16/f32 scale bits in one owned byte vector. No full
decoded-trit or dense-weight shadow is retained. Upload reserves the final
payload fallibly, uses the format codec's caller-owned `pack_into` seam, and
does not create a second decoded tensor or a per-plane allocating pack buffer.
Existing allocating codec functions share these encoders; their wire bytes and
canonical padding remain unchanged. This implements the accepted codec/native
execution architecture, not a new format, schema, law or admission combination.

Matmul reads packed symbols directly and performs add/subtract/skip in the same
plane/group/coefficient order as the scalar core reference. Independent output
rows use rayon; per-row arithmetic does not change with thread count. Small
compile-time B3/S34 symbol tables are shared code data, not per-tensor shadows.
Forward/inverse basis transforms remain the shared core functions. Gather reads
packed coefficients, combines exact scales, then applies the inverse input-axis
basis. S34 source structure is checked before capability/admission/allocation;
raw additive upload also rejects unadmitted groups. Dense semantic execution and
legacy packed ISA kernels remain unchanged. No SIMD or speed claim is made.

`tensor_caps().payload_bytes` and the opaque handle's `len_bytes()` count packed
planes plus stored-precision scales. Codec units continue across row boundaries
and reset at each plane, including canonical terminal padding. Outer entropy
transport is not resident storage: inputs here are already decoded semantic
views. Per-tensor and shared lifetime admission therefore use the retained
payload, while typed upload evidence keeps decoded-source bytes separately.

Cargo only promotes the existing schema dependency into CPU production and adds
the existing evidence crate for CPU integration tests. Lockfile changes only the
CPU dependency list; no dependency version changes. Shared schema, core reference,
testkit vectors, numerical tolerances, CUDA/NN code and resident kernels are
unchanged.

## Verification

The new production-seam tests cover 31,080 codec-symbol/law/group/plane/basis
cases, 180 additional ragged multi-group/plane cases and 1,320 larger rotation
cases in one- and four-thread pools. Different planes carry different signs;
scale sets include zero and nonzero values. Output and gathered-row f32 bits
must exactly match the unchanged core reference; basis scratch also matches.
Each case checks exact canonical payload length and exact/one-byte-short budget
behavior. The larger matrices have 37 rows and 256 columns, including basis
blocks 4/32/128/256 independently of G32/G128. These are tensor fixtures, not
language-model quality or runtime measurements.

Additional tests cover invalid S34 structure, bad gather IDs, shape/overflow
rejection before mutation, empty calls and actual CPU execution through admitted
tied handles. The fixed SignedRht fixture has 16 decoded-source bytes but 10
resident payload bytes. Two handles fill a 20-byte shared ledger; a third is
rejected, tied ownership counts once, released capacity is reusable and all
leases return to zero. Three typed successful-upload observations verify through
the registered hash-chain verifier; rejection emits no successful observation.
This is payload-accounting evidence, not physical process-memory qualification.

The caller-owned codec test preserves exact canonical byte vectors, checks
incorrect destination lengths without mutation and validates all S34 source
groups before writing. Existing 180-combination frozen CPU/reference suites
remain unchanged except the older CPU residency expectation now names the
actual packed payload (20 bytes rather than 32 decoded bytes).

Focused tests passed while iterating. An initial new S34 expected-byte literal
was corrected from `b4 01` to `34 01` by the documented zero-position/sign
encoding; no pre-existing vector or tolerance changed. The first locked test
after adding the test dependency refused the necessary lockfile-list update;
an offline run updated that list, and final verification uses `--locked`.

Committed verification invocation `d209bb929cc54583aa8df02231e244ce`
reached terminal success: MainPID zero, SubState exited, Result success,
ExecMainStatus zero. Commands (shared SSD target, two build jobs):

```sh
cargo test --locked -p tritium-schema -p tritium-core -p tritium-spec \
  -p tritium-format -p tritium-testkit -p tritium-cpu -p tritium-runtime \
  -p tritium-evidence -p tritium-cuda -p tritium-wgpu --all-targets
cargo clippy --locked -p tritium-cpu -p tritium-format --all-targets -- -D warnings
sh scripts/verify-gates.sh prepush
```

Regression: **524 passed, zero failed, six ignored** (CPU A8 microbenchmark and
five Qwen artifact/master-dependent tests). Default feature-gated GPU bodies
execute zero cases; this is not GPU qualification. Scoped Clippy passes.
Canonical prepush passes format/projections, actual bare-metal compile,
default/all-feature Clippy, Windows GNU cross-check and actionlint. ShellCheck
is locally absent and explicitly warned, not passed. All five hosted workflows
on the previous `f32fbacb` source pass; new-source hosted clearance is separate.

Source/log/script/hardware custody archive:
`/home/brianklam/Projects/Tritium/archive/verification/cpu-packed-c68fb654-20261010`.
Its standalone SHA-256/original-Git verifier establishes custody only.

## Qualification boundary and remaining work

This is P2 native CPU prerequisite progress, not P2 or public-release completion.
Byte counts exclude vector/handle metadata, allocator overhead, shared lookup
tables, source-view storage, output/activation scratch, rayon state, KV and driver
state. This change does not establish physical RSS reduction, model-wide memory
preflight, fitting peak memory, decode throughput or a 10x whole-model claim.
Existing HF/SALT production loaders still use legacy projections and do not
automatically adopt this tensor path or its shared evidence/admission facade.

Remaining: SIMD/native GPU and other-backend execution; full `.trit` typed
manifest, per-tile/stream/shard/migration work; one engine/calibration/direct-fit
allocator; one loader/runner and thin surfaces; production full-memory admission;
Stage-7 solver/auxiliary evidence and official Qwen language/MTP quality/runtime/
bytes/reproduction gates; physical/distributed/browser training; serving/security;
same-source RC/packages/model zoo/docs/community; independent empirical
clearance and explicit human activation. No release verdict, signature or human
authorization is produced here.
