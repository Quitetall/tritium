# Typed model quantization engine candidate

This implements the first complete source-to-package path in ADR0044 D11,
following the proposed ADR0058 profile. It is not a public release, an owner
profile freeze, a Qwen quality result or a replacement for empirical gates.

## Actual production path

`tritium_quantize::engine::Engine::run` takes a complete Rust-owned `QuantPlan`,
an original-byte `WeightSource`, a `TensorSink` and a fresh registered recorder.
It admits all structural/numerical requests before payload fitting; checks exact
source/asset inventories; hashes original raw F16/BF16/F32 bytes itself; and
widens those same bytes through the shared format decoder. Dense tensors retain
their exact original precision and bits, including signed zero. A requested
Gram must exist, match its original f64-bit digest and pass the unchanged fitter.
The in-memory and already-open multi-shard safetensors adapters use this same
controller. This is observed tensor content, not immutable HF source identity.

Every level is independently fitted from original weights by the existing
production controller, not sliced from a larger level. Original assets and
encoded tensor blobs are deduplicated by actual byte content while preserving
all named directory interpretations. The bounded package writer computes the
complete aligned output size before reserving the output buffer; the legacy
writer delegates and retains identical output bytes.

Registered start, per-tensor fit and completion observations feed a consuming
evidence seal. The immutable complete receipt view rechecks the real JSONL
chain/root, strict payload versions, request/source/output inventories, fit
plans/metrics and dense preservation. Only this actual completed log supplies
the manifest's run/span/root Direct ancestry. Dense-only conversion is Imported.
The log seals before packaging, avoiding a package/manifest/evidence hash cycle.
This is fit-stage completion, not a claim that a sink published successfully.

Before publication, the actual typed package reader validates every level and
its payloads. Package/log binding checks compare the complete manifest and each
fitted semantic digest against the sealed observations. They do not prove a
producer's declaration true by independently refitting original weights.
CLI evidence verify/view use the same recognized conversion fold; their output
shape stays unchanged, and generic integrity is still not a full registry gate.

The memory sink refuses a second artifact. The directory sink exclusively creates
a new destination, writes/syncs the actual log and private pending package, then
uses a no-clobber hard link to publish `model.trit`. Pre-commit failures remove
only known owned files, not injected foreign data. Post-commit pending cleanup
or directory-sync failure is returned as an explicit warning, not an error
pretending publication did not happen. OS sync is not power-loss/security proof.

## Software checks

The candidate has narrow tests for original precisions, both source adapters,
real multi-shard/file I/O, both sinks, exact dense preservation, actual fitter
parity, level/source/Gram changes, dedup, writer byte parity and exact-size
rejection, malformed requests, missing/changed metrics, bad chains/roots,
structural lies with valid hash chains, partial logs and existing destination
preservation. The actual NN tokenizer/text loader consumes an engine-produced
complete ternary Qwen3-shaped synthetic fixture, comparing generated tokens
against the existing reference backend without copying neural math.

These fixtures are software correctness evidence only: no official Qwen weights,
GPU performance, model-quality, complete memory or release qualification.
Final exact-code test/gate identities are recorded after verification.

## Full remaining release scope

1. Common precision-aware feedback/fitter selection, unified calibration and
   direct-fit allocator; SALT/QTEA/ExTernD/TWLA/frontier GPU/out-of-core/refinement;
   thin convert/quantize/Qwen/Python entrypoint migration with parity.
2. Owner-frozen profile and all generated projections; immutable HF/source/caps
   and full-memory admission; physical shards/per-tile/entropy/stream/mmap;
   exact legacy migration and only then gated deletion.
3. Native CPU SIMD/GPU/all-backend conformance; resident Executor/Runner,
   ArchitectureSpec/hybrid DeltaNet/MTP; full RSS/scratch/KV/source/tokenizer/
   driver accounting and quiet physical performance checks.
4. Stage7 authorized recipe/solver/refinement/auxiliary evidence and official
   Qwen language+MTP ternary/refined quality/task retention/runtime/physical-byte
   and independent reproduction gates.
5. Physical ROCm/Metal/all backends; browser operation/lifecycle/fault/no-hidden-
   readback matrix on Chrome/Firefox/Safari; WASM/MCU/distributed training gates.
6. Serving residency/cancellation/KV/concurrency/fault/latency; plugins/OCI/K8s/
   serverless/security/observability and human acceptance.
7. Same-source RC archives/wheels/ONNX/Colab/compatibility; docs/tutorials/
   community/audited model zoo; second-machine/operator reproduction, security/
   signing clearance and explicit human public activation.

The initial engine retains unique encoded blobs and the full final package in
memory. Logical source/asset/retained-blob/package limits are not full process
RSS, solver/Rayon/source-header/tokenizer/device/KV admission or out-of-core
support. Native Rust frontend v1.2 and arbitrary trainable ONNX v1.3 remain
distinct. No full P0-P7 phase or empirical release gate is declared complete.
