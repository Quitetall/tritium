# Production CPU semantic tensors — 2026-10-10

Code: `ee03b86faa346a70b6707e3a384717ed0015d62e`.
Base: `89e244be9dc16e110ac6ec1cfd9593f1764161e5`.
Scope: part of accepted ADR 0044 D3/D8/P2, not phase clearance.

## Production path and shared implementation

The production `CpuBackend` now implements `upload_tensor`, `matmul` and
`embed_rows` for additive and dense f32 tensors. The initial three actual
backend tests failed at the old default upload method (exit 101), rather than
testing only the reference backend. The final tests pass through the public
backend interface.

Additive uploads reuse the canonical owned `AdditiveTensor::from_view` with
fallible payload reservations and exact stored-scale precision checks. Trits
remain ternary; there is no implicit dense materialization. CPU operations
delegate to the core's consumer-bound additive matmul/gather, so SignedRht
inverse order is not reimplemented in a backend. Payload reservation failure
maps to `BackendError::OutOfMemory` with requested bytes.

Dense validation, scalar f32 dot order and gather now live in allocation-free
core `DenseView`; both CPU and the testkit reference backend use it. This
removes duplicate dense execution loops. Shape/ID validation precedes all
scratch/output writes. Zero-width/zero-row tensors are legal, with zero-row
matmul returning without iterating an arbitrarily large empty batch.
The codebase-design skill guided this shared core seam and canonical ownership
rather than adding backend-specific arithmetic copies.

CPU handles own their payload independently of callers and reject foreign or
legacy buffer kinds for semantic operations. `len_bytes()` reports decoded
i8-trit/f32-scale or dense-f32 payload bytes. It is not packed artifact size,
allocator/metadata accounting or complete model/KV residency.

This path is **scalar semantic execution**, not the existing packed SIMD
dispatch. CPU crate docs explicitly distinguish the two. Legacy packed kernels,
CUDA kernels, format bytes, law admission and numerical precision defaults
are unchanged. The sole lockfile change adds an existing workspace schema
crate as a CPU test dependency; no dependency version changes.

## Verification

```sh
cargo test --locked -p tritium-core -p tritium-spec -p tritium-format \
  -p tritium-testkit -p tritium-cpu --all-targets
cargo test --locked -p tritium-core -p tritium-cpu --doc
scripts/verify-gates.sh prepush
```

Regression suite: **474 passed, 0 failed, 6 ignored**. The skips are one CPU
microbenchmark and five artifact/FP-master-dependent Qwen allocation/error/
scale-census probes; they do not become passing model measurements.
The four CPU semantic tests cover:

- Actual D2/B3/S34 encode→decode→CPU upload→matmul/gather over frozen logical
  weights for Identity, Hadamard and SignedRht (nine codec/basis combinations).
- Every currently admitted scale law at its supported plane count.
- Dense execution, invalid/overflowing shapes, empty matrices and fail-closed
  output/scratch on malformed calls.
- Upload ownership, foreign handles and inexact F16 scale rejection.

These are small matrices (`N=2`, `K=4`, group 32 for additive vectors), not
G64/G256 admission, ragged/per-tile/large-model conformance or speed evidence.
Doctests: two consumer-mismatch compile-fail tests pass; CPU has zero doctests.

Canonical prepush terminates with `MainPID=0`, `SubState=exited`,
`Result=success`, `ExecMainStatus=0`, invocation
`9a9094c33ac446eaa13a7c602bb5b3a5`: formatting, projections, actual bare-metal
compile, default/all-features Clippy, Windows GNU cross-check and actionlint
pass. Local ShellCheck is absent and explicitly warned, not claimed passing.

Previous-source CI `38066337747` and wheels `38066337661` at `89e244be`
succeed. They qualify their own source checks, not this new CPU source.
Fresh same-source hosted results are required after this push.

Archive: `/home/brianklam/Projects/Tritium/archive/verification/cpu-semantic-ee03b86f-20261010`.
Retained-byte/source custody is not empirical release or phase approval.

## Remaining full goal

Per-combination capabilities, native-only refusal, explicit dense-emulation
events/admission and all other backend implementations remain. CPU packed
semantic residency/SIMD, production load-time consumer binding, reference
registration on wasm/MCU and complete conformance still need work. Finish
schema/projections/identities, packed/per-tile views, streaming/shards/fuzzing,
bounded tracing evidence, unified engine/loader/runner/adapters and migration
before legacy deletion. P0–P7 gates are not declared complete.

Full Stage-7, Qwen language/MTP quality/runtime/physical-byte/reproduction,
physical backend/distributed/browser training, serving/security, final RC
packages, audited zoo, docs/community, independent clearance and explicit
human release authorization remain required. No large campaign or cloud job
was started in this slice.
