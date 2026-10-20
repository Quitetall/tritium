# Explicit CUDA additive emulation — 2026-10-10

ADR 0044 D8/D9/D12 implementation progress, not P2 or release clearance.
Code: `c4ce8d3114faec3a2eaf8bc2f9d69254ad80aef9`; base:
`28eb0b0f022f5a60d9890896835a9eec239027fe`.

## Execution and admission

CUDA's shared semantic tensor interface supports currently admitted additive
laws at G32/G128 via **Emulated** execution. Capability queries validate exact
stored scale precision and declared geometry without materializing weights.
They report `rows * cols * sizeof(f32)` owned payload bytes, not packed bytes.
G64/G256 remain undeclared; native-only and payload budgets reject before upload.

Upload expands canonical decoded trits/scales once into f32 weights in the
stored basis, transfers them to CUDA, and drops the temporary host weight copy.
No host shadow remains. Allocation reservations are fallible. Nonfinite expanded
weights are refused before device transfer rather than being executed as NaNs.
This is a fail-closed representability limit, not a proof over extreme inputs.
The device tensor retains its basis and exact owning stream/context.

Matmul applies the shared core forward basis to host activation scratch, then
runs the existing exact-f32 GPU GEMM with unit scales. Gather downloads selected
device rows and applies the shared inverse basis. SignedRht inverse is `D H`,
not forward `H D`; neither transform is reimplemented. Host geometry/IDs are
validated before writes. Driver errors need not be transactional. Existing
packed and resident CUDA kernels remain byte-identical; no TF32, CPU GEMM
fallback, packed ternary speed claim or current Qwen chat acceleration is added.

The codebase-design skill kept execution behind the existing shared interface
and moved the frozen scalar vectors into one testkit module, replacing the
CPU-only copy rather than adding a backend-specific arithmetic oracle.

## Typed upload observations

`tritium_runtime::upload_tensor_logged` combines checked upload and registered
typed evidence for future loading surfaces. `TensorUploaded` lives in the Rust
schema and projects to generated JSON Schema. It records tensor/backend/device,
execution tier, actual checked payload bytes and decoded scalar source bytes.
That source count is **not disk size or compression savings**. The event ID is
`tritium.runtime.tensor_uploaded`, version 1. Unknown observation/capability
fields are refused by typed deserialization and reflected in the projections.

Policy/backend rejection emits no successful event. If recording fails after
upload, the handle is dropped and the caller receives an evidence error. Two
controlled runtime tests establish this failure behavior, not hardware claims.
The physical fixture verifies successful event content, registered hash-chain
replay, strict typed parsing and equal roots from two actual upload observations
with identical deterministic context.

Raw and checked backend uploads still exist during migration. **Production
CLI/NN/serve loaders have not yet adopted the logged facade.** Thus this slice
does not establish mandatory evidence on every production upload or complete
D9/D12 enforcement. Per-upload budgets exclude temporary host expansion,
activation/output/scaling scratch, allocator/driver overhead, KV, other tensors
and total model residency. Aggregate physical-memory admission remains open.

## Source-bound checks

Machine: `onyx-maurader-BrianBigPC`, Linux, RTX 4090,
GPU `GPU-1790118a-a6d7-4eaf-fcac-dcacac5f4351`, driver `615.71.09`.
Only tiny synthetic fixtures run; no fitting campaign, cloud job or model payload
is modified. Physical commands explicitly set `TRITIUM_CHECK_ONLY=0`:

```sh
cargo test --locked -p tritium-cuda --features cuda \
  --test semantic_additive --test semantic_dense -- --test-threads=1
compute-sanitizer --tool memcheck --target-processes all --error-exitcode 99 \
  cargo test --locked -p tritium-cuda --features cuda \
  --test semantic_additive --test semantic_dense -- --test-threads=1
cargo test --locked -p tritium-schema -p tritium-core -p tritium-spec \
  -p tritium-format -p tritium-testkit -p tritium-cpu -p tritium-runtime \
  -p tritium-evidence -p tritium-cuda --all-targets
cargo clippy --locked -p tritium-cuda --features cuda --all-targets -- -D warnings
scripts/verify-gates.sh prepush
```

Physical tests: **6 passed, 0 failed, 0 ignored**, never device-self-skipped.
Compute Sanitizer: the same six execute, **0 errors**. Default regression:
**511 passed, 0 failed, 6 ignored** (one CPU microbenchmark, five Qwen/artifact
tests). Feature-gated CUDA bodies do not execute in that default sweep; physical
execution is the separate command above. Scoped physical-CUDA Clippy passes.

The shared frozen harness executes 180 combinations on each of CPU, reference
and CUDA (540 total): all current law/plane counts, G32/G128, three bases and
three codec metadata variants, multiple/ragged groups. These are canonical
decoded views, **not 540 packed-file roundtrips or native codec kernel tests**.
The original embedding 2e-6 and matmul 1e-3 thresholds are unchanged. Dense
vectors retain 1e-4. The signed upload fixture independently checks exact logical
rows, transformed activations, ownership after dropping source data, repeated
gathers, invalid shapes/IDs, admission boundaries and expanded-weight overflow.

The initial physical test fails at undeclared additive capability; an early
import compile failure is retained. Neither gates nor numerical kernels change.
Cargo.lock adds only two edges to the existing evidence crate, no dependency
version changes. Verification unit: `tritium-cuda-emulation-verify-20261010`,
invocation `0445afabbcff44efbbe928947cb51320`. Terminal canonical status and
publication/hosted snapshots are retained separately in the custody archive.
That verification unit terminates with MainPID zero, SubState exited, Result
success and ExecMainStatus zero. Canonical prepush passes format/projection drift,
bare-metal compilation, default/all-feature Clippy, Windows GNU cross-check and
actionlint. **Local ShellCheck is absent and explicitly warned**, not passed.

Archive:
`/home/brianklam/Projects/Tritium/archive/verification/cuda-emulation-c4ce8d31-20261010`.
Source is exported from original Git objects and checked by the standalone
inventory verifier. Custody is not independent numerical/release approval.

## Remaining

Previous-source `28eb0b0f` hosted CI/wheels/CodeQL/docs/capstone all pass; current
source checks remain separate. No fresh D14 decode/perplexity performance pass
is claimed: the recorded BitNet fixture path is absent. GPU availability is not
a quiet-box measurement. No Qwen quality, memory reduction or SOTA result follows.

Remaining: production logged-upload wiring and aggregate memory admission;
native packed semantic kernels and other-backend emulation/conformance;
schema/container/engine/loader/resident runner/surface/migration/evidence P0–P7;
full Stage-7 and official Qwen language/MTP quality/runtime/physical-byte/
reproduction; physical browser/distributed/backend training; serving/security;
fresh RC/packages, audited zoo/docs/community; independent clearance and human
activation. No phase or full goal is self-approved.
