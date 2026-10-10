# WebGPU semantic tensor execution — 2026-10-10

Implementation progress under accepted ADR 0044 D8/D9 and plan 0055 P2.
This is not P2 completion, browser qualification, model-quality evidence,
physical-memory qualification, a performance result or public-release approval.

## Source and execution identity

- Base: `e00c26e5ed6a72cc5ef3fb4a57f27121469dced8`.
- Code: `89b4535e1175038b571b3acf6ca6b2137f1c570f`.
- Six source/test/manifest files; Cargo.lock adds one existing evidence dependency
  edge, with no package-version changes.
- Linux `7.2.9-1-cachyos`, x86_64; NVIDIA RTX 4090, driver `615.71.09`, UUID
  `GPU-1790118a-a6d7-4eaf-fcac-dcacac5f4351`; Intel Graphics RPL-S integrated GPU.
- Physical commands select the exact adapter-name substring with
  `TRITIUM_WGPU_ADAPTER`; a nonmatching substring is an error, not fallback.
- Verification unit: `tritium-wgpu-semantic-verify-20261010.service`, invocation
  `b3e2e9f3c3714d278a08cfda54d1d8e2`.

The backend's recorded physical-device descriptor now includes the actual wgpu
family, vendor/device IDs and adapter name instead of the default `"wgpu"`.
This is a descriptor, not a UUID or a unique identity between identical cards.
Source, commands, machine metadata and logs are retained outside scratch at:
`/home/brianklam/Projects/Tritium/archive/verification/wgpu-semantic-89b4535e-20261010`.

## What changed

The shared `TernaryBackend` interface now executes dense and additive tensors on
wgpu. Private semantic handles retain an owned f32 device buffer, geometry,
basis and exact backend-instance ownership token, with no host weight shadow.
Another instance on the same adapter, a reference handle or a legacy handle
cannot be mistaken for the owner. Empty dense payloads do not allocate buffers.

Dense tensors report `Native`. Additive tensors report `Emulated`, validate the
existing admitted laws and G32/G128 combinations, and expand once at upload into
f32 coefficients in their stored basis. Precision-inexact/negative-zero scales,
unadmitted G64/G256, malformed geometry and nonfinite expansion fail closed.
The latter is an expansion-representability limit, not a proof over all extreme
numeric inputs. Capability queries do not promise every finite source can be
expanded without overflow.

A private WGSL shader computes plain f32 GEMM with arbitrary stored coefficients,
not clipping, STE or a CPU arithmetic fallback. Matmul transforms activations
on the host through the shared core. Gather copies selected rows from the GPU
and applies the shared inverse basis on the host, including SignedRht's correct
inverse ordering. Caller input/ID/shape errors are checked before output writes.
Device buffer, binding, u32-index and 2-D dispatch limits are validated before
allocation/dispatch. GPU validation/allocation failures surface as typed errors;
semantic GPU error scopes/readbacks serialize within an executor.

Checked upload enforces native-only and payload budgets before allocation.
The runtime logged-upload facade records explicit emulation and expanded bytes;
successful upload roots replay deterministically in the fixture. Production
CLI/NN/serve callers are not yet migrated to that facade. Payload accounting is
not aggregate model residency: temporary host expansion, activation/output/
readback allocations, driver overhead, KV and concurrent requests are excluded.
Expanded f32 residency earns no packed ternary memory-reduction credit.

Legacy WebGPU shaders, training arithmetic, CUDA kernels and NN source are
unchanged. This does not accelerate the current Qwen loader or resident decoder.
The implementation uses the existing synchronous native wgpu backend; browser
async acquisition/execution and physical browser tests remain separate work.

## Checks and numerical scope

Strict physical tests do not self-skip and reject software rasterizers. On each
of NVIDIA Vulkan and Intel Vulkan, all six semantic tests pass, zero failures or
ignores. Shared frozen additive vectors execute **180 decoded-view combinations
per adapter**, covering all current law/plane counts, G32/G128, three bases and
D2/B3/S34 codec metadata. These are not 360 packed-file roundtrips or native
packed-codec kernels. Original tolerances stay unchanged: 2e-6 gather and 1e-3
additive matmul; dense cases use 1e-4 against the shared core.

Tests also check frozen fractional dense outputs, eight dense shape cases,
source-payload lifetime, inverse gather with duplicate IDs, zero dimensions,
overflow/shape/foreign-instance rejection without caller-buffer mutation,
native-only/budget rejection without success events, deterministic upload
evidence replay and three concurrent callers repeatedly using the same handle.
The initial absent-capability failure and subsequent generic-device-identity
failure are retained; the latter was repaired in implementation, not hidden by
changing the expected evidence to the generic name.

Commands run on the clean code commit, with the shared SSD Cargo cache and two
build jobs:

```sh
TRITIUM_WGPU_ADAPTER='NVIDIA GeForce RTX 4090' cargo test --locked -p tritium-wgpu --features register --all-targets -- --nocapture
TRITIUM_WGPU_ADAPTER='Intel(R) Graphics (RPL-S)' cargo test --locked -p tritium-wgpu --features register --test semantic_tensor -- --nocapture
TRITIUM_WGPU_ADAPTER='Intel(R) Graphics (RPL-S)' cargo test --locked -p tritium-wgpu --features register --test portable_training -- --nocapture
cargo test --locked -p tritium-schema -p tritium-core -p tritium-spec -p tritium-format -p tritium-testkit -p tritium-cpu -p tritium-runtime -p tritium-evidence -p tritium-wgpu --all-targets
cargo clippy --locked -p tritium-wgpu --features register --all-targets -- -D warnings
scripts/verify-gates.sh prepush
```

NVIDIA all-targets: 12 tests pass, including the existing physical training
test's 117 supported conformance cases; zero failures/ignores. A separately
executed Intel training test also passes those 117 cases, zero failures/ignores.
These bounded operation fixtures do not qualify model training or performance.
Default regression:
501 pass, zero failures, six declared ignores (one CPU microbenchmark and five
Qwen/artifact-dependent tests). Feature-gated wgpu bodies do not execute in the
default run; the physical commands above are the execution evidence.
Scoped Clippy and canonical prepush pass. The verification invocation is terminal
success (MainPID zero, SubState exited, Result success, ExecMainStatus zero).
Canonical format, generated projections, actual bare-metal compilation,
default/all-feature Clippy, Windows GNU cross-check and actionlint pass.
Local ShellCheck is absent and explicitly warned, not passed.
Hosted checks on the new source remain separate. All five previous-source hosted
workflows at `e00c26e5` completed successfully. Custody verification checks original
Git bytes/inventory only and does not approve empirical or release claims.

## Remaining release work

1. Production logged-upload/memory-policy adoption, aggregate residency and
   native packed semantic execution; remaining backend conformance and physical
   browser/Metal/ROCm/MCU coverage.
2. ADR 0044 P0–P7: final schema/container/sharding/identity projections, unified
   loader/resident runner, engine/calibration/direct-fit levels, thin surfaces,
   migration/deletion and evidence views, with phase gates still open.
3. Full Stage-7 solver/refinement/auxiliary-producer grid and official Qwen
   language/MTP quality, runtime, physical bytes, residency and reproduction.
4. Distributed/physical training matrix, serving concurrency/KV/cancellation,
   security/OCI/deployment/observability, fresh RC/wheels/ONNX/Colab checks.
5. Audited model zoo, docs/community, second-machine/operator independent
   clearance, signing and explicit human public-release activation.

No campaign/model data, release signature or human authorization is generated.
