# CUDA dense semantic tensors — 2026-10-10

ADR 0044 D8/D9 prerequisite implementation, not P2 or release clearance.
Code: `f85e4aadf1e3e5ad2a9fc8349af710b2d9c69c55`; base:
`a86c406ee274a53880f8071710d32827a7e55294`.

## Implementation and limits

CUDA implements the shared `tensor_caps`, `upload_tensor`, `matmul` and
`embed_rows` interface for originally dense f32 views. Upload retains only a
device-owned payload, with no host weight shadow. Handles retain their owning
stream/context and are rejected by another backend instance, even on the same
physical device. Legacy and CPU handle types are also refused.

Matmul reuses `matmul_forward_dev` with unit scales. Its existing arbitrary-f32,
sequential-reduction, `--fmad=false` kernel is unchanged; no cuBLASLt TF32,
ternary clipping, STE or CPU fallback is introduced. Input/output/scratch lengths,
overflow and int32 launch geometry are validated before device work or host
writes. Zero operations do not allocate or launch. Gather validates all IDs
before copying device row views, preserving duplicates and order. It is not a
batched/fused gather kernel.

Dense capability is `Native`, meaning direct execution of the originally dense
input. **Additive views remain unsupported**, rather than being advertised as
working emulation. The dense path is needed before an explicit, basis-correct
additive-to-dense upload adapter can execute on GPU. It does not make the current
Qwen loader use this interface or accelerate current Qwen chat.

Payload bytes count the stored f32 weights only; activation, scale and output
scratch, driver/allocator overhead, KV and other tensors remain outside the
per-upload budget. Raw upload remains policy-unchecked during migration; callers
must use `upload_tensor_checked` for native-only/budget admission. No aggregate
physical-memory qualification is claimed. Invalid host inputs are rejected
before writes; a driver failure is not promised to be transactional.

## Source-bound execution evidence

Physical machine: `onyx-maurader-BrianBigPC`, Linux, NVIDIA RTX 4090,
GPU UUID `GPU-1790118a-a6d7-4eaf-fcac-dcacac5f4351`, driver `615.71.09`.
Only tiny synthetic fixtures are used; no model fitting or cloud job is started.
The device availability snapshot is not quiet-box performance evidence.

All Cargo commands use the existing SSD target, `RUSTC_WRAPPER=`, two build jobs,
and `--locked`. Physical commands explicitly use `TRITIUM_CHECK_ONLY=0`:

```sh
cargo test --locked -p tritium-cuda --features cuda --test semantic_dense -- --test-threads=1
compute-sanitizer --tool memcheck --target-processes all --error-exitcode 99 \
  cargo test --locked -p tritium-cuda --features cuda --test semantic_dense -- --test-threads=1
cargo test --locked -p tritium-schema -p tritium-core -p tritium-spec \
  -p tritium-format -p tritium-testkit -p tritium-cpu -p tritium-runtime \
  -p tritium-cuda --all-targets
cargo clippy --locked -p tritium-cuda --features cuda --all-targets -- -D warnings
scripts/verify-gates.sh prepush
```

The physical test fails if device initialization fails; it never self-skips into
a green result. Initial two tests fail with `UnsupportedTensor`. Final committed
physical execution: **3 passed, 0 failed, 0 ignored**. Frozen small values plus
eight shape cases exercise arbitrary fractional weights, K=31/33/129, multiple
activation rows, zero dimensions/batches, repeated gathers, ownership, malformed
shapes/IDs, overflow, exact budget boundaries and CPU/foreign handles. The shared
core DenseView is the numerical oracle; the absolute tolerance remains 1e-4.
This is bounded fixture conformance, not a proof over every dense f32 matrix.

Compute Sanitizer executes the same three tests and reports **0 errors**.
Default foundation/CPU/CUDA regression suite: **502 passed, 0 failed, 6 ignored**
(one CPU microbenchmark and five Qwen/artifact-dependent tests). Default CUDA
semantic integration bodies are feature-gated and therefore do not execute in
that CPU regression sweep. Physical execution is the separate command above.
Scoped real-CUDA Clippy passes.

Verification unit: `tritium-cuda-semantic-verify-20261010.service`, invocation
`ceaf2acd56914a8d876de3ddc895e5dc` terminates with MainPID zero, SubState exited,
Result success and ExecMainStatus zero. Canonical prepush passes format,
projection drift, bare-metal compilation, default/all-feature Clippy, Windows
GNU cross-check and actionlint. **Local ShellCheck is absent**; actionlint warns
that shell workflow blocks were not linted locally. Terminal status and hosted
snapshots are retained in the custody archive.

Archive destination:
`/home/brianklam/Projects/Tritium/archive/verification/cuda-semantic-f85e4aad-20261010`.
Source is exported from original Git objects. Inventory/source-byte custody is
verified separately; it is not numerical, performance, model or release approval.

## Open obligations

The recorded BitNet decode fixture path
`/home/brianklam/.cache/tritium-models/bitnet-2b4t-gguf/ggml-model-i2_s.gguf`
is absent in this turn. No fresh D14 end-to-end decode/perplexity performance pass
is claimed. Kernel source and dependency versions are unchanged. Previous-source
hosted CI/wheels/CodeQL/docs/capstone at `a86c406e` all pass; they do not qualify
this new source.

Remaining: explicit additive emulation plus typed events and memory admission;
native packed CPU/GPU semantics and all-backend vectors; production load-time
basis binding; unified loader/resident executor/runner and surface wiring;
remaining schema/container/engine/migration/evidence P0–P7 gates; full Stage-7
solver grid and official-source Qwen language/MTP quality, runtime, physical-byte
and reproduction gates; physical multi-device/browser training; serving/security;
fresh RC/packages, audited model zoo/docs/community; independent clearance and
explicit human release activation. No phase or full milestone is self-approved.
