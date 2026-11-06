# Lifetime-bound shared tensor payload admission — 2026-10-10

ADR 0044 D8/D9/D12 prerequisite progress, not complete production memory
admission, model-quality/performance evidence, P2/P4 completion or release approval.

## Identities and scope

- Base: `927909630ca59dbb203fecafab00b4d19ec80184`.
- Code: `6c7343c2b51f02f2cd33af1158eb9a3c6646bd74`.
- Six source/test files; no dependency, schema, numerical threshold, model,
  CUDA/NN source, resident kernel or legacy format change.
- Verification invocation: `c7f19c3538c24a5f837808b0a1da1cf0`, unit
  `tritium-tensor-budget-verify-20261010.service`.
- Linux `7.2.9-1-cachyos`, x86_64, NVIDIA RTX 4090 driver `615.71.09`, UUID
  `GPU-1790118a-a6d7-4eaf-fcac-dcacac5f4351`; Intel RPL-S integrated GPU via Vulkan.
- Archive: `/home/brianklam/Projects/Tritium/archive/verification/tensor-budget-6c7343c2-20261010`.

The existing serving admission module governs principals/request rates, not
model allocations. Production HF/SALT loaders still use legacy projection types
rather than the shared tensor interface. This change supplies the shared
ownership guard they need; it does not claim those loaders now use it.

## Implementation

`TensorPayloadBudget` is a process-local ledger with a fixed limit and an atomic
reserved-byte counter. Clones share the same ledger. Callers choose whether a
ledger covers one model, a device, or several concurrent models; separately
created ledgers do not coordinate. Checked addition and compare/exchange reserve
the entire advertised payload before backend upload. Overflow and insufficient
capacity fail closed without a successful upload event.

`upload_tensor_logged_admitted` reuses the existing checked upload/evidence
implementation. Native-only/per-tensor policy remains unchanged, with one shared
`validate_tensor_upload_policy` helper also used by backend checked upload.
Successful upload returns a private `DeviceBuffer` wrapper retaining its lease;
backend downcasts and ownership checks still see the inner concrete handle.
The wrapper drops the owned buffer before releasing admission. Sharing that
handle with `Arc` for tied embedding/head consumers counts its allocation once
until the final owner drops it. No public unwrap method detaches the reservation.

Upload failure, actual/declaration payload mismatch, changing capabilities and
failed evidence recording all drop any buffer and release the lease. Capability
drift is detected before emitting a success observation. Earlier upload events
remain observations of those uploads, not a claim that a whole model loaded.
The original unbudgeted facade remains compatible during migration.

This is **owned-payload accounting**, not physical-memory qualification. It
excludes host expansion, activations, outputs/readback, temporary fitting work,
KV, allocator/driver overhead, other ledgers and unguarded uploads. In-flight
admitted uploads count against the ledger. A backend that misreports bytes is
detected after upload, not physically prevented from allocating them. Expanded
f32 emulation receives no packed ternary memory-reduction credit. Complete model
preflight and production CLI/NN/serve adoption are still required.

## Verification

The initial public-interface test fails at absent budget/facade imports; that
failure is retained. Five controlled facade tests cover final-owner lifetime,
clone sharing, aggregate refusal without evicting live tensors, capacity reuse,
backend/recording failures, mismatch/drift, unsupported/native-only/per-tensor/
empty-name/shape rejection, and eight concurrent uploads sharing one four-byte
ledger. A drop observer checks that buffer destruction precedes lease release.
One arithmetic unit test covers u64 overflow and zero reservations without
pretending to allocate such a physical payload. These are controlled tests,
not independent release or physical-memory evidence.

On each of NVIDIA and Intel Vulkan, seven strict semantic tests pass with no
device self-skip or ignores. The aggregate case uploads two 32-byte emulated
handles into a 64-byte ledger, refuses an additional eight-byte dense upload,
executes GPU matmul/gather through a tied/shared admitted handle, then verifies
capacity reuse and release. The original 180 decoded-view additive combinations
per adapter remain at unchanged tolerances; these are not packed-file roundtrips.
The existing training test executes 117 supported cases on each adapter.

Real CUDA executes the unchanged three dense and three additive tests, zero
failures/ignores. This checks the shared policy refactor, not CUDA use of the new
aggregate facade or a fresh decode-performance result. Default foundation/
CPU/runtime/evidence/CUDA/wgpu regression: 517 passed, zero failed, six declared
ignores (one CPU microbenchmark and five Qwen/artifact-dependent cases).
Feature-gated GPU tests execute only in
the separate physical commands below.

```sh
TRITIUM_WGPU_ADAPTER='NVIDIA GeForce RTX 4090' cargo test --locked -p tritium-wgpu --features register --all-targets -- --nocapture
TRITIUM_WGPU_ADAPTER='Intel(R) Graphics (RPL-S)' cargo test --locked -p tritium-wgpu --features register --test semantic_tensor --test portable_training -- --nocapture
TRITIUM_CHECK_ONLY=0 cargo test --locked -p tritium-cuda --features cuda --test semantic_dense --test semantic_additive -- --nocapture
cargo test --locked -p tritium-schema -p tritium-core -p tritium-spec -p tritium-format -p tritium-testkit -p tritium-cpu -p tritium-runtime -p tritium-evidence -p tritium-cuda -p tritium-wgpu --all-targets
cargo clippy --locked -p tritium-runtime -p tritium-spec -p tritium-wgpu --features tritium-wgpu/register --all-targets -- -D warnings
scripts/verify-gates.sh prepush
```

Commands use the existing shared SSD Cargo cache and two build jobs. Scoped
Clippy and canonical prepush pass. The verification invocation is terminal
success (MainPID zero, SubState exited, Result success, ExecMainStatus zero).
Format/projection drift, actual bare-metal compile, default/all-feature Clippy,
Windows GNU cross-check and actionlint pass. Local ShellCheck is absent and
explicitly warned, not passed. Terminal results are retained before publication.
All five previous-source hosted workflows at `92790963` pass. New-source hosted
results and independent phase/empirical release clearance remain separate.
The archive verifier checks retained hashes/original Git bytes, not model,
numerical, physical-memory, performance or SOTA qualification.

## Remaining release work

1. Wire production loaders/surfaces to shared logged and admitted uploads, with
   complete model preflight and KV/scratch/driver reservations; native packed
   semantic execution and remaining backend/browser hardware conformance.
2. ADR 0044 P0–P7: schema/container/shards/identity, unified loader/resident runner,
   engine/calibration/direct-fit levels, thin surfaces, migration/deletion and
   evidence views, with all phase gates still open.
3. Full Stage-7 solver/refinement/auxiliary grid; official Qwen language/MTP
   quality/runtime/physical bytes/residency/reproduction and NearLossless gates.
4. Distributed/physical training, serving concurrency/KV/cancellation/security,
   fresh RC/packages/ONNX/Colab, audited zoo, docs/community.
5. Independent second-machine/operator clearance, signing and explicit human
   public-release activation. No campaign/model payload or authorization changed.
