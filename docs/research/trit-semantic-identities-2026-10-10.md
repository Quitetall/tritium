# Candidate logical tensor and model identities

## Scope and source

Base: `285c05690347a329fa3c1ebd50ed1ffab05c7cc4`.
Implementation: `cd6660b4bbef12a1e03466697120984d4fac4f25`.
Verified code tree: `e82c7d9fdce010bce05098fc9571b552ce97c095`.
Architecture authority: ADR 0044 D1/D4/D7/D10/D13.
Detailed profile: ADR 0054 **PROPOSED**, not owner-accepted or release-frozen.
This is P1/P4 prerequisite implementation, not phase clearance, a legacy identity
rewrite, model-quality qualification or public-release activation.

The checked `.trit` reader now binds candidate tensor and per-level model
identities during existing all-level payload validation. `tensor_digest(level,
name)` and `model_id(level)` expose those immutable bindings; missing names
return errors. No caller-supplied unvalidated manifest is promoted to an identity.
Invalid payloads or imported-prefix claims still prevent the reader from returning
a model. Content identity remains distinct from exact-byte PackageId, BlobId,
fitting provenance, evidence roots, architecture execution and qualification.

Rust owns `TensorSemanticMeta`, `NamedAssetDigest`, `NamedTensorDigest` and
`ModelSemanticMeta`; their JSON Schemas are generated through the existing drift
checker. No generated projection is hand-edited. The types retain alloc/no_std
compatibility and strict serde object-field handling.

## Candidate preimage

Each domain uses BLAKE3 derive-key hashing. Metadata is the existing Rust-derived
canonical CBOR spelling, framed by its u64 little-endian byte length. Domains:
`tritium.trit.additive.v1`, `tritium.trit.dense.v1`, `tritium.trit.model.v1`.

- Additive metadata binds rows/cols, tile/group geometry, actual uniform planes,
  scale law/precision and basis, including signed-RHT block/seed/domain. It is
  followed by a u64-le logical trit count, plane-major one-byte codes
  (negative=0, zero=1, positive=2), then a u64-le scale-byte length and exact
  little-endian stored scale bits. Tied laws hash stored anchor scales, not
  widened or recomputed derived scales. Physical codec/transport and unused
  maximum capacity do not participate.
- Dense metadata binds shape and stored precision, followed by a u64-le payload
  byte length and exact finite stored bytes. Signed zero and stored precision
  remain distinct. Mathematical equivalence across precision or dense/additive
  representations is not claimed.
- One complete model level binds exact architecture spelling, every sorted named
  asset-content digest, and sorted logical tensor names with semantic digests.
  Asset JSON is exact bytes, not JSON-equivalence canonicalization. Display-level
  labels, fit/evidence annotations, other levels, legacy aliases, shard locations,
  packed tensor BlobIds and physical placement are excluded. Identical content
  aliases have the same ModelId without acquiring each other's provenance.

Logical additive hashing reuses the tensor already decoded for validation. Trit
and scale staging is bounded to 1024 bytes; it does not materialize a complete
preimage or dense weight matrix. Metadata serialization and digest inventories
still allocate; retained digests use 32 bytes per level/tensor binding plus
container overhead. Validation still temporarily decodes one additive tensor,
two for prefix comparisons. This is not a whole-model peak-memory measurement,
mmap/streaming reader, physical-memory admission or a 10x reduction result.

Single-file uniform/raw payload support is unchanged. Per-tile/shard/entropy
payload reading remains explicitly unsupported. The standalone calculator may
ignore an owned logical tensor's transport label; it does not make that transport
readable. LoadedModel receipts, HF identities and legacy receipt identities are
not rewritten or synthesized by this slice.

## Verification and claim limits

Exact-tree verification invocation: `cbf051a96eeb42149e848929bd36b8f0`.
Terminal state checked on 2026-10-10: MainPID zero, SubState exited, Result success,
ExecMainStatus zero; exit time 17:12:20 America/New_York.

- Foundation/backend: **546 passed, 0 failed, 6 declared ignores**. Disabled GPU
  feature bodies do not establish physical accelerator coverage.
- NN default: **166 unit plus 8 integration passed**.
- NN tokenizer: **167 unit plus 10 integration passed**.
- CUDA/tokenizer: **12 integration passed**, with real CUDA cases required by
  `TRITIUM_REQUIRE_CUDA=1`, `TRITIUM_CHECK_ONLY=0`.
- Scoped schema/format and real-feature NN Clippy, warning-denied rustdoc,
  actual thumbv7em-none-eabihf alloc+serde compilation and canonical prepush pass.
  Prepush covers fmt/projection drift, bare-metal foundations, default/check-only
  all-feature workspace Clippy, Windows GNU compile and actionlint. ShellCheck
  remains missing/warned; shell-body linting is not reported as passed.

Five additional public-interface tests bring the manifest suite to 22. They check
D2/B3/S34 repacking with stable logical IDs and different physical package IDs;
label/provenance aliases; exact trit/scale/law/precision/group/geometry/plane/basis
changes; individual signed-RHT block/seed/domain changes; dense shape/signed-zero;
architecture/tensor/asset names and asset bytes; physical inventory order; unused
capacity/transport exclusions; independent levels; absent binding rejection; and
fixed preimages. Existing vectors, numerical tolerances and admission laws are
unchanged. Tiny `{}` fixture assets are structural, not executable model configs.

The separately executed `scripts/verify-trit-semantic-vectors.py` assembles its
own metadata and framed preimages using cbor2's canonical encoder, without the
Rust encoder/calculator. Installed libraries: cbor2 6.1.5, blake3 1.0.9. Four
candidate vectors agree with the Rust public-interface tests:

```text
additive fc19a72d5206c01988147cd77700b8d6a7f1ae8977c0321d16763220e89b69cd
dense    b19646f6931fcd0dab8b73d14f624cdc542865cddd9a6a40d9c798727fa7a568
model    a396a7331dadad1cf59f96df549cf551784ca5a19f5c6e9990c1f66d6fcbaf5f
tied     54b0d7099eb3e68572e14a45160b666e70e9d945a827d685f0d331bbe599bdf4
```

This establishes cross-encoder candidate agreement, not independent human
acceptance, profile freeze or empirical release clearance. Real CUDA regression
uses RTX 4090 `GPU-1790118a-a6d7-4eaf-fcac-dcacac5f4351`, driver 615.71.09.
Its synthetic dense/emulated-additive execution is not pretrained Qwen quality,
native packed CUDA performance, model-wide physical bytes or throughput evidence.

Initial absent-interface and fixture-root-type failures are retained. The first
exact-tree job failed strict rustdoc because the pre-existing public GGUF error
documentation linked private `MAX_STRING_LEN`; final code changes only that
wording. The original failure log/unit state and passing rerun remain distinct.
No warning gate, parser bound or numeric threshold was weakened. All five hosted
workflows of previous publication `285c0569` pass; new-publication hosted checks
and independent empirical release clearance remain separate.

```sh
cargo test --locked -p tritium-schema -p tritium-core -p tritium-spec \
  -p tritium-format -p tritium-testkit -p tritium-cpu -p tritium-runtime \
  -p tritium-evidence -p tritium-cuda -p tritium-wgpu --all-targets
python scripts/verify-trit-semantic-vectors.py
cargo clippy --locked -p tritium-schema -p tritium-format --all-targets -- -D warnings
cargo check --locked -p tritium-schema --no-default-features --features alloc,serde \
  --target thumbv7em-none-eabihf
RUSTDOCFLAGS=-Dwarnings cargo doc --locked -p tritium-schema -p tritium-format --no-deps
cargo test --locked -p tritium-nn --lib --test tensor_loading
cargo test --locked -p tritium-nn --features tokenizer --lib --test tensor_loading
TRITIUM_CHECK_ONLY=0 TRITIUM_REQUIRE_CUDA=1 cargo test --locked \
  -p tritium-nn --features cuda,tokenizer --test tensor_loading
cargo clippy --locked -p tritium-nn --all-targets --features cuda,tokenizer -- -D warnings
RUSTDOCFLAGS=-Dwarnings cargo doc --locked -p tritium-nn --no-deps
RUSTDOCFLAGS=-Dwarnings cargo doc --locked -p tritium-nn --features tokenizer --no-deps
sh scripts/verify-gates.sh prepush
```

The configured BitNet decode fixture is still absent; quiet five-run decode/trace
gates remain **UNKNOWN**, not waived. No fitting campaign, cloud compute, model
download or old-cache deletion occurred. Durable custody destination:
`/home/brianklam/Projects/Tritium/archive/verification/semantic-identities-e82c7d9f-20261010`.
Its original-Git/SHA verifier proves retained-byte custody only.

## Remaining full-release work

No P0-P7 phase or full goal is complete.

1. Owner-accepted profile/frozen independently verified identities and all CDDL/
   layout/Python/TS projections; full loader/receipt, shards/per-tile/transport/
   stream/mmap, exact fixture/Qwen migration and eventual legacy deletion.
2. Unified fitting engine, calibration/direct-fit allocator, SALT/QTEA/ExTernD/
   TWLA/frontier GPU/out-of-core/refinement; thin PTQ/QAT/Python/HF/CLI/ONNX/FFI/
   Burn/Candle/browser/MCU surfaces and shared evidence views.
3. Native CPU SIMD/GPU/backends, resident executor/architecture/runner, hybrid
   DeltaNet/MTP, full source/vector/tokenizer/scratch/KV/driver memory preflight,
   quiet decode and kernel-trace performance gates.
4. Stage-7 recipe/solver/refinement/auxiliary evidence and authorized freeze;
   official Qwen language/MTP ternary/refined quality/task-retention, runtime,
   physical bytes and independent reproduction. Synthetic tests are not these gates.
5. Physical backend/browser/distributed training matrix, Chrome/Firefox/Safari
   117-op/lifecycle/fault checks without hidden readbacks, distinct WASM/MCU gates.
6. Serving cancellation/residency/KV/concurrency/fault/latency, plugins, deployment,
   security/observability and explicit human acceptance.
7. Fresh same-source RC/packages/ONNX/Colab, docs/tutorials/community, audited zoo,
   second-machine/operator replay, independent release/security/signing clearance,
   human activation and publication smoke.

The full goal remains active. Only this milestone's own scratch is cleaned after
durable custody verification; the reused checkout and shared build cache remain.
