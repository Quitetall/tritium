# Recorded model-load identities and receipt views

## Scope and source

Base: `8cd2335651acf4349a3cec9b06e406b6e191ffdf`.
Verified implementation: `952855de1a2d2867b6a29e8f9f7bd2ec9bcc5135`.
Authority: ADR 0044 D1/D7/D10/D12; detailed ADR 0054 remains **PROPOSED**,
without owner acceptance or profile freeze. This is P1/P4 prerequisite progress,
not phase, pretrained-model, physical-memory/performance or release clearance.

The actual shared text `tritium_nn::load` now returns a `ModelLoadReceiptView`
alongside its runner, real tokenizer and architecture variation data. It records
one schema-owned `ModelLoaded` event after tokenizer, weights, tensor coverage,
vocabulary and runner construction succeed. The event binds initial source,
architecture, backend/device labels, vocabulary rows and standard layer count.
It contains no release/quality verdict, completed-run root or memory/speed claim.

`LoadedSourceIdentity::Trit` carries the already validated reader's exact package
ID, candidate selected-level model ID and explicit level label. It does not
reread bytes, substitute a BlobId for ModelId, resolve legacy IDs or verify direct-
fit ancestry. The transitional format PackageId wrapper is copied by its exact
32 bytes into the shared schema wrapper; old hash domains/wire spellings and
pinned receipts are unchanged. `HuggingFace { reason: MissingCapability }` explicitly
has no semantic identity. A path is neither logged nor invented as an identity.

The immutable receipt view borrows its own recorded envelope/typed observation
through a two-getter interface. It is not another serialized receipt file. It
describes initial load, not later mutable runner/weight state. Replay folding
rechecks every chain and supplied root even if public `VerifiedLog` fields were
modified, then checks ModelLoaded version and strict payload. Other registrations
remain the caller's full-registry verification obligation. Hashing a producer's
declaration does not independently prove an artifact or execution claim true.

CLI `evidence verify` and `evidence view` use the same fold for recognized load
events, retaining their existing output shape and generic integrity scope. A
correctly hashed but malformed recognized load event fails both commands.
Payload errors expose only the decoder category, without echoing untrusted data.
Schema types are alloc/no_std compatible; their two JSON Schemas are generated
and drift-checked, not edited by hand.

Final event-recording failure rejects the load and drops the unpublished runner/
weight leases. Earlier tensor-upload observations remain, without a successful
ModelLoaded event. Required registration is explicit:

```rust,ignore
let recorder = tritium_evidence::EvidenceRecorder::new(
    "caller-stable-run",
    [tritium_evidence::TENSOR_UPLOADED_REGISTRATION,
     tritium_evidence::MODEL_LOADED_REGISTRATION],
)?;
```

Weights-only legacy/admitted helpers remain token-level operations without this
text-load event. Complete before-upload preflight, HF immutable source/content
identity, streaming evidence/tracing, full-memory admission and all-surface
adoption remain work. The receipt stores small metadata/envelope copies, not
source bytes or a full retained dense model. It is not a peak-memory measurement.

## Exact-commit verification

Invocation: `46308a8a060440108c544069bf8613d5`.
Terminal state: MainPID zero, SubState exited, Result success, ExecMainStatus zero;
exit 2026-10-10 17:44:23 America/New_York. Source was clean and unchanged before
and after the detached driver.

- Foundation/backend: **552 passed, 0 failed, 6 declared ignores**. Feature-
  disabled accelerator bodies are not physical-device results.
- NN default: **166 unit plus 8 integration passed**.
- NN tokenizer: **167 unit plus 12 integration passed**.
- Strict CUDA/tokenizer: **14 integration passed**. Actual CUDA cases require
  `TRITIUM_CHECK_ONLY=0`, `TRITIUM_REQUIRE_CUDA=1`; not every case is GPU-only.
- Full CLI binary suite: **60 passed, 0 failed, 1 declared external-BitNet ignore**.
- Scoped real-feature Clippy, warning-denied default/tokenizer rustdoc, actual
  bare-metal alloc+serde compile and unchanged cross-encoder identity vectors pass.
- Canonical prepush passes fmt/projection drift, bare-metal foundations, default/
  check-only all-feature workspace Clippy, Windows GNU compile and actionlint.
  ShellCheck remains absent/warned; shell-body linting is not reported passed.

Six new public evidence tests cover recorded/replayed view agreement, no verdict,
mutable-log/root tampering, malformed top-level/nested payloads with valid hashes,
future versions, registration/time failure without sequence advancement and no
invented receipt in upload-only logs. Two CLI tests cover actual command-module
verify/view success and malformed typed payload rejection.

Existing text execution tests now check exact source/device/config binding and
registered receipt replay on CPU, reference and actual CUDA. Two new production-
load tests cover nine selected-level loads across D2/B3/S34: content aliases keep
ModelId, distinct levels differ, package IDs differ across codecs, and each
receipt binds the selected level. HF and typed-package final-registration failure
occurs after successful uploads, then releases all unpublished payloads while
preserving valid upload observations. Existing invalid-asset/vocabulary tests
assert no successful load event. Numerical vectors/tolerances are unchanged.

These are small synthetic standard-transformer/tokenizer fixtures, not pretrained
Qwen language/MTP, fitting, task retention, throughput or model-wide physical-byte
evidence. Real CUDA device: RTX 4090
`GPU-1790118a-a6d7-4eaf-fcac-dcacac5f4351`, driver 615.71.09. Additive CUDA remains
explicit dense emulation; kernels/resident execution and legacy numerical paths
are unchanged. No native additive CUDA or all-backend/browser qualification follows.

Initial missing-interface, distinct PackageId-wrapper, test import and test-only
clone-lint failures are retained, along with the corrected projection-bin invocation.
No warnings, admission laws, numeric bounds or frozen gates were relaxed. All five
hosted workflows for prior publication `8cd23356` pass; new-publication hosted
checks and independent empirical release/security clearance remain separate.

```sh
cargo test --locked -p tritium-schema -p tritium-core -p tritium-spec \
  -p tritium-format -p tritium-testkit -p tritium-cpu -p tritium-runtime \
  -p tritium-evidence -p tritium-cuda -p tritium-wgpu --all-targets
python scripts/verify-trit-semantic-vectors.py
cargo check --locked -p tritium-schema --no-default-features --features alloc,serde \
  --target thumbv7em-none-eabihf
cargo test --locked -p tritium-nn --lib --test tensor_loading
cargo test --locked -p tritium-nn --features tokenizer --lib --test tensor_loading
TRITIUM_CHECK_ONLY=0 TRITIUM_REQUIRE_CUDA=1 cargo test --locked \
  -p tritium-nn --features cuda,tokenizer --test tensor_loading
cargo test --locked -p tritium-cli --bin tritium
cargo clippy --locked -p tritium-schema -p tritium-evidence -p tritium-nn -p tritium-cli \
  --all-targets --features tritium-nn/cuda,tritium-nn/tokenizer -- -D warnings
RUSTDOCFLAGS=-Dwarnings cargo doc --locked -p tritium-schema -p tritium-evidence \
  -p tritium-nn --no-deps
RUSTDOCFLAGS=-Dwarnings cargo doc --locked -p tritium-schema -p tritium-evidence \
  -p tritium-nn --features tritium-nn/tokenizer --no-deps
sh scripts/verify-gates.sh prepush
```

The configured BitNet fixture remains absent; repack qualification, quiet five-run
decode and kernel trace gates are **UNKNOWN**, not waived. No campaign/refit,
cloud compute, model download or old-cache deletion occurred. Custody destination:
`/home/brianklam/Projects/Tritium/archive/verification/load-evidence-952855de-20261010`.
Original-Git/SHA verification proves retained bytes, not qualification.

## Remaining full-release work

No P0-P7 phase or full goal is complete.

1. Owner-accepted/frozen typed profile/identities and all CDDL/layout/Python/TS
   projections; full preflight/load identity, shards/per-tile/transport/stream/mmap,
   exact fixture/Qwen migration and eventual legacy deletion.
2. Unified fitting engine/calibration/direct-fit allocator, SALT/QTEA/ExTernD/TWLA/
   frontier GPU/out-of-core/refinement, thin PTQ/QAT/Python/HF/CLI/ONNX/FFI/Burn/
   Candle/browser/MCU surfaces and bounded/tracing shared evidence/views.
3. Native CPU SIMD/GPU/backends, resident executor/architecture/runner, hybrid
   DeltaNet/MTP, full source/vector/tokenizer/scratch/KV/driver memory preflight
   and quiet decode/per-kernel performance gates.
4. Stage-7 recipe/solver/refinement/auxiliary evidence and authorized freeze;
   official Qwen language/MTP ternary/refined quality/task-retention, runtime,
   physical bytes and independent reproduction.
5. Physical backend/browser/distributed training matrix; Chrome/Firefox/Safari
   117-op/lifecycle/fault checks without hidden readbacks; distinct WASM/MCU gates.
6. Serving cancellation/residency/KV/concurrency/fault/latency, plugins, deployment,
   security/observability and explicit human acceptance.
7. Fresh same-source RC/packages/ONNX/Colab, docs/tutorials/community, audited zoo,
   second-machine/operator replay, independent release/security/signing clearance,
   human activation and publication smoke.

Only this milestone's own scratch is cleaned after durable custody verification.
The reused checkout/shared build cache, unrelated edits and old campaigns remain.
