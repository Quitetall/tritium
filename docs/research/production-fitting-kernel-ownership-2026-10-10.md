# Production fitting kernel ownership — 2026-10-10

## Scope and identities

Implements the production ownership consolidation in accepted ADR0044 D3/D11
and private plan0055 P3. Candidate ADR0055 remains **PROPOSED**; no owner freeze,
full phase clearance or public release authorization is recorded.

- Original source: `e7bc4f9fcff4cc87ad36e121111277808029453b`.
- Kernel relocation: `96c037443f9beb34fc53c43d7b85d29166f021cd`.
- Strict documentation repair: `6265f32aeef5147a875da732c21a6f50ac8ec5e7`.
- Final checked code, including portable optional serde:
  `d8bcfa056bbbfd0a5ef9ef278e0db77d3f109f2d`.
- Verifier unit: `tritium-fit-engine-verify-v5-20261010`, invocation
  `2989f8889b6047db83db7c750f624295`; terminal success at
  2026-10-10 19:02:37 EDT, MainPID 0, SubState exited, ExecMainStatus 0.
- Machine/device metadata and exact commands are retained in the archive below.
  Physical CUDA tests used RTX 4090,
  `GPU-1790118a-a6d7-4eaf-fcac-dcacac5f4351`, driver 615.71.09.

## Implementation

The actual shipping f32 working-weight / f64 metric fitter now lives in
`tritium-quantize::engine`: damping/inverse-Cholesky, feedback, decay/ramp,
scale refit, discrete search, encoding and artifact-rounded reconstruction.
NN retains calibration collection and source-compatible public reexports.
The numerical implementation and config block were relocated byte-for-byte;
working precision, loop ordering, fallback behavior and numerical defaults remain.

Training now uses the same ladder step search, rounding and group fitting.
Quarter-octave constants and first-candidate tie-breaking are unchanged. The
training public helpers remain compatible aliases/adapters, not duplicate fitters.
Training and NN's existing test-only quantize dependencies become normal edges;
quantize does not depend on training.

The f32 Hadamard helper is owned by the existing core basis implementation.
Training's old public path reexports it. The legacy ragged group-tail rotation
rule is retained and explicitly **does not** admit a ragged typed Hadamard basis.

The extended portability check found that core's optional serde dependency
incorrectly enabled std. It now uses the existing no-std workspace dependency.
Public feature names and all three legacy format tokens are unchanged. The
canonical gate separately checks bare-metal serde with and without alloc.
Strict rustdoc failures were fixed as documentation, without suppressing warnings.

## Exact-tree software checks

The final tree was clean before and after verification. Counts are per lane,
not unique tests across repeated feature configurations.

| Check | Result |
|---|---|
| Core + quantize + training, all targets | 624 passed, 0 failed, 10 declared ignores |
| Legacy format serialization conformance | 8 passed, 0 failed |
| Default NN, fixed fixtures, legacy decay, tensor loading | 179 passed, 0 failed |
| CLI ladder artifact roundtrip | 2 passed, 0 failed |
| Tokenizer NN and same fitting/loading surfaces | 184 passed, 0 failed |
| Physical CUDA geometric ladder tests | 2 passed; kernel covers 288 bit-exact configurations |
| CUDA/tokenizer NN fixed fixtures and loading | 17 passed, 0 failed |
| Warning-denied core/quantize/training/NN docs; tokenizer docs | PASS |
| Canonical `sh scripts/verify-gates.sh prepush` | PASS, with ShellCheck unavailable warning |
| Actual `wasm32-unknown-unknown` and `wasm32-wasip1` checks | PASS |
| Actual core bare-metal `serde` and `alloc,serde` checks | PASS |

The canonical gate includes formatting, generated-projection drift, actual
no-std foundation builds, default/all-feature workspace Clippy and Windows
cross-target checks. All-feature toolkit-free compilation is not hardware
execution. ShellCheck was unavailable: workflow run blocks were not linted
locally; this warning is not promoted to a pass. Hosted CI is separate.

Four fitter scale-bit/trit hashes and the Hadamard output-bit hash were captured
on original source before moving math and asserted there before relocation.
Those goldens were not changed. A separately compiled probe includes the actual
original STE source from the pinned Git object, and matches 1,620 ladder cases
(planes 1–9, multiple grids/shapes/distributions, rotation and ragged groups)
in exact anchor bits/trits, plus 13 Hadamard lengths in exact output bits.
Core also checks normalization against the original sqrt-reciprocal bits for
all powers of two representable as a usize and preserves invalid-length panics.

Retained failed attempts: a wrong-crate roundtrip test target; a nocapture result
counter expecting inline `ok`; strict-rustdoc private/broken/redundant links;
and the original bare-metal optional-serde std dependency failure. No numerical
golden, precision default or frozen gate was weakened to resolve them.

## Boundaries and remaining work

This is real numerical ownership consolidation, **not completion of P3 or
`Engine::run(QuantPlan, WeightSource, TensorSink, Evidence)`**. The f64 feedback
oracle remains a distinct reference, not a silently selected production default.
No model, recipe, candidate identity preimage, artifact, law/group/basis/backend
admission or CUDA resident kernel was changed.

This does not establish model quality, near-losslessness, native packed additive
CUDA, speed, physical bytes, full memory bounds, security, SOTA or release readiness.
The BitNet fixture needed for quiet decode/per-kernel performance remains absent;
that gate is UNKNOWN, not waived. Browser/MCU cross-compilation is not physical
browser/MCU execution. Existing independent and human release gates remain.

Full remaining scope, with software before expensive model compute:

1. Typed QuantPlan/source/sink/evidence engine; unified calibration, direct-fit
   allocator and fitter selection; SALT/QTEA/ExTernD/TWLA/frontier GPU/out-of-core
   fitting/refinement; thin PTQ/QAT/Python/HF/CLI/ONNX/FFI/Burn/Candle/browser/MCU.
2. Accepted/frozen profile and all Rust-derived projections; HF content identity,
   full source/caps/memory admission; shards/per-tile/transport/stream/mmap;
   exact legacy/Qwen migration and gated legacy deletion.
3. Native CPU SIMD/GPU/backend conformance and resident Executor/Runner/
   ArchitectureSpec, hybrid DeltaNet/MTP; full scratch/KV/source/tokenizer/
   allocator/driver accounting and quiet/per-kernel performance.
4. Stage-7 recipe/solver/refinement/auxiliary evidence and authorized freeze;
   official Qwen language + MTP ternary/refined quality, task retention, runtime,
   physical bytes and independently executed reproduction.
5. Physical ROCm/Metal/all backend and Chrome/Firefox/Safari operation/lifecycle/
   fault coverage; distinct WASM/MCU/distributed training qualification.
6. Serving residency/cancellation/KV/concurrency/fault/latency/plugins;
   OCI/K8s/serverless/security/observability and explicit human acceptance.
7. Fresh same-source RC archives/wheels/ONNX/Colab, compatibility/docs/tutorials,
   community and audited model zoo; independent second-machine/operator
   reproduction/security/signing clearance and explicit public activation.

Native Rust frontend v1.2 and arbitrary trainable ONNX v1.3 stay distinct from
the v1.1 goal. No complete phase or empirical release gate is cleared here.

## Custody and cleanup

Durable source/fixture/probe/log/private-record archive:
`/home/brianklam/Projects/Tritium/archive/verification/production-fitting-d8bcfa05-20261010`.
Its standalone verifier checks exact retained-byte inventory and changed source
against externally selected original Git commits; custody is not qualification.
Only this run's scratch/probe outputs are cleanup scope after verification.
Shared caches, older caches, campaign outputs and unrelated work are preserved.
