# Typed `.trit` model manifests and checked payload bindings

## Source and contract scope

Base: `e22c1bf5118ee4c88cae4aa83fb0ad14249cee01`.
Code: `025435674fb83dde5763c485bda5d6fb4d79705c`.
Accepted architecture authority: ADR 0044 D1/D4/D5/D7/D10/D13.
The detailed field profile is ADR 0054 **PROPOSED**, without owner acceptance.
This is a candidate typed consumer, not a frozen public model-profile admission.

`tritium-schema` now owns model/tensor metadata, complete levels, direct/imported
fit ancestry, named assets/shards, legacy references and stored-byte blob
references. Structural validation checks inventories, required config/tokenizer
assets, safe asset/shard basenames, shape/overflow/count limits, declared shard and
legacy references, additive law/geometry/plane allocation, complete cross-level
tensor names/shapes and acyclic imported-prefix relationships. Recorded direct
ancestry does not verify that evidence exists, is independent or is accepted.

The format module derives canonical CBOR from Rust serde types and the existing
canonical-envelope codec. Encoding normalizes discovery order without collapsing
duplicates; decoding requires the exact typed spelling and ordering. Unknown
fields/variants/versions/tags, optional-field aliases and noncanonical bytes reject.
There is no handwritten second model-manifest encoder.

`read_trit_model` validates the real package directory/hashes and every complete
level before returning a borrowed typed view. Blob kind/dtype and asset names must
match, not merely the digest. Dense F16/BF16/F32 require exact lengths and finite
values; canonical uniform/raw additive payloads reuse the existing codec and
exact-scale validator. Imported prefixes compare logical trits and exact scale
bits, so repacking with another codec does not invalidate a true prefix; direct or
unqualified imported levels are never automatically described as prefixes.

Metadata indexes retain original immutable directory positions. Named references
borrow checked original source bytes rather than rehashing every duplicate level
reference or retaining decoded weights. Additive validation temporarily decodes
one tensor, or two during prefix checking; this is not mmap/streaming/out-of-core
or whole-model peak-memory admission. PackageId is the existing exact-byte identity,
never a substitute for codec-independent semantic tensor digest or per-level ModelId.

Eleven JSON Schemas are generator-produced and drift-checked. Enabling serde for
generation also makes seven existing layout/law/basis/transport projections reject
unknown object properties, matching the strict typed consumer. Recognized fields,
tags, admitted laws, numeric payloads and frozen vectors do not change. JSON Schema
does not encode every invariant in `validate`, nor establish a canonical JSON view.
Normative CDDL/layout/Python/TypeScript and semantic identity vectors remain work.

Per-tile, shard and entropy metadata roundtrips, but the current payload reader
returns explicit unsupported-capability errors. It never treats structural
serializability as execution support, substitutes dense data or produces an empty
model. Config/tokenizer bytes are authenticated assets, not yet architecture/
tokenizer-admitted or executable through this typed view. The generic container,
existing HF/GGUF/SALT readers and immutable evidence/model artifacts are unchanged.

## Test scope and verification

Seventeen public-interface tests exercise real container writing/reading, source
borrowing and cross-level blob sharing; all dense precisions; 18 group/free codec-
precision-basis combinations plus tied-group and tensor-wide laws; original law
caps; exact scale/code/length corruption with fresh correct blob hashes; logical
prefixes across codecs and false trit/scale/dense-prefix claims; all-level checking;
unknown/nested/duplicate/missing fields/tags/variants, optional spelling aliases,
trailing/deep CBOR; unsafe names, shapes/overflow/required assets, complete-level
and prefix-cycle/legacy/shard references; explicit unsupported payload capabilities.
A property test feeds arbitrary short byte strings to the public decoder. This is
bounded randomized testing, not a completed fuzzing/security qualification gate.

Initial red failed on absent types/module. Test authoring then corrected an S34
fixture lacking mandatory zeros and a three-plane tensor-wide fixture exceeding
the existing one-plane I2_S cap. Both failures were retained; validators, admitted
caps, existing expected vectors and numerical tolerances were not weakened.

Verification invocation: `f3859b8e1e7a49e98fe51cb82afc70a3`.
The detached exact-commit job reached terminal success: MainPID zero, SubState
exited, Result success, ExecMainStatus zero. Shared SSD target and two build jobs
were used; no new per-run Cargo cache was created.

Foundation/backend regression: **541 passed, zero failed, six declared ignores**
(CPU microbenchmark and model/master-dependent Qwen tests). Default feature-gated
GPU bodies in that command execute zero cases, not physical backend qualification.
NN regression: **166 unit plus four integration tests passed**. Strict real CUDA
HF-loading regression: **five passed**, with `TRITIUM_REQUIRE_CUDA=1` preventing
an unavailable-device skip. It checks unchanged synthetic HF execution, not a new
typed `.trit` GPU loader or a pretrained model. Device: RTX 4090,
`GPU-1790118a-a6d7-4eaf-fcac-dcacac5f4351`, driver 615.71.09.

Scoped Clippy, actual bare-metal `alloc,serde` compile and canonical prepush pass.
Canonical prepush covers format/projection drift, bare-metal core/schema compile,
default and check-only all-feature workspace Clippy, Windows GNU compile and
actionlint. ShellCheck remains absent/warned, not passed. Cross-target/check-only
compilation is not hardware/backend/browser execution evidence.

```sh
cargo test --locked -p tritium-schema -p tritium-core -p tritium-spec \
  -p tritium-format -p tritium-testkit -p tritium-cpu -p tritium-runtime \
  -p tritium-evidence -p tritium-cuda -p tritium-wgpu --all-targets
cargo test --locked -p tritium-nn --lib --test tensor_loading
TRITIUM_CHECK_ONLY=0 TRITIUM_REQUIRE_CUDA=1 cargo test --locked \
  -p tritium-nn --features cuda --test tensor_loading
cargo clippy --locked -p tritium-schema -p tritium-format --all-targets -- -D warnings
cargo check --locked -p tritium-schema --no-default-features \
  --features alloc,serde --target thumbv7em-none-eabihf
sh scripts/verify-gates.sh prepush
```

All five hosted workflows at previous source `e22c1bf5` pass. New-source hosted CI
and independent empirical release clearance remain separate. The BitNet decode
fixture is still absent; its required five quiet runs and trace baseline are
UNKNOWN, not waived. No large download, campaign, model fitting or cloud job ran.

Custody archive:
`/home/brianklam/Projects/Tritium/archive/verification/model-manifest-02543567-20261010`.
Original-Git/hash verification establishes custody only, not empirical admission.

## Remaining release work

This is P1/P4 prerequisite progress, not completion of either phase or v1.1.

1. Owner-accepted typed profile, semantic identities/frozen vectors, all generated
   projections, typed whole-model loader, shards/per-tile/entropy/stream/mmap,
   verified fixture/Qwen migration and eventual legacy deletion.
2. Unified fitting engine, calibration and direct-fit allocator; frontier solvers,
   GPU/out-of-core/refinement and thin PTQ/QAT/Python/HF/CLI/ONNX/FFI/Burn/Candle/
   browser/MCU surfaces over the same core and evidence views.
3. Native/SIMD backends, resident executor/architecture/runner consolidation,
   hybrid DeltaNet/MTP and complete memory preflight including source, vectors,
   scratch, KV, allocator and driver overhead; quiet decode/trace performance gate.
4. Stage-7 recipe/solver/refinement/auxiliary evidence and authorized freeze,
   official Qwen language plus MTP quality/task retention, runtime, physical-byte
   and independent reproduction gates. No synthetic test establishes these.
5. Physical backend/browser/distributed training matrix, browser 117-op/lifecycle/
   failure checks without hidden readback, and distinct WASM/MCU qualification.
6. Serving cancellation/residency/KV/concurrency/fault/latency, plugins, production
   deployment, security and observability; human acceptance remains explicit.
7. Fresh same-source RC archives/wheels/ONNX/Colab compatibility, docs/tutorials/
   community, audited three-tier/four-model zoo; second-machine/operator replay,
   independent release/security/signing gates, human activation, publication smoke.

The full public goal remains active. No campaign data or old scratch was deleted;
the user's old-cache cleanup confirmation is still pending.
