# Source-bound direct-fit observations

Status: software candidate verified; no full phase or empirical release gate cleared.

Base: `3572cf4a995fd961ecf5480b791bed7234fadee5`.
Code: `107fb0b36ec85a7d8254c796c76fc1ee3b206a74`.
Contract: accepted ADR0044 D5/D11/D12 and proposed private ADR0057;
ADR0054/0056 identity/fitting profiles remain proposed, not owner-frozen.

## Implemented

`fit_direct_levels_recorded` checks the recorder context, calls the unchanged
production `fit_direct_levels`, binds actual original inputs and all outputs,
and records one typed `tritium.quantize.direct_levels_fitted@1` observation.
It returns owned validated tensors and an immutable log-derived receipt view.
No successful event or fit result escapes on fitting/hashing/recording failure.

Rust schema types own the full executed per-tensor plan, logical tensor name,
exact original f32 source-bit digest, explicit identity or original f64 Gram-bit
digest, actual Rayon pool size, and every returned level's label/count/shared
additive semantic digest. Four JSON Schema projections are generated, not edited.
No weights or Gram are copied into the log. This source-f32 identity is **not**
original BF16/safetensors/HF identity or calibration dataset identity. Output
semantic identity remains distinct from stored blob/package bytes.

The view rechecks chains, root, event version, strict payload and plan/output
bindings. It normalizes transient serde numeric representation to the existing
JCS wire spelling: `1000000.0` and `1000000` must not produce unequal views of
the same event. Source/Gram bit digests preserve signed zero. CLI evidence
verify/view use this same typed fold for recognized fit observations without
changing their output shape or inventing a registry/qualification verdict.
Replay verifies recorded integrity/bindings, not producer truth by refitting.

The module design keeps input hashing, output semantic binding and recording in
one shared controller interface rather than duplicating them in each surface.
Existing numerical controller/kernel source and all goldens remain unchanged.

## Exact-code verification

Detached invocation `13d9abb132164e51a50858af3c29c2a7` terminated successfully:
MainPID 0, Result success, ExecMainStatus 0, at 2026-10-10 23:56:44 UTC.
Linux CPU execution; machine/compiler metadata and complete logs are retained.

Commands, with the existing SSD Cargo cache:

```sh
cargo test --locked -p tritium-schema -p tritium-format -p tritium-evidence -p tritium-quantize --all-targets
cargo test --locked -p tritium-cli evidence::tests
cargo test --locked -p tritium-nn --test activation_fit_parity --test feedback_decay_legacy
RUSTDOCFLAGS='-D warnings' cargo doc --locked -p tritium-schema -p tritium-evidence -p tritium-quantize --no-deps --features tritium-schema/schema-gen
cargo check --locked -p tritium-schema --no-default-features --features alloc,serde --target thumbv7em-none-eabihf
sh scripts/verify-gates.sh prepush
```

- Schema/format/evidence/quantizer: **724 passed, 0 failed, 15 declared ignored**.
  Six new recorded-controller tests are included, not counted again.
- Filtered CLI evidence checks: **7 passed**; unrelated CLI tests remain filtered.
- Legacy numerical parity/feedback fixtures: **4 passed**.
- Total executed test passes: **735**, zero failures. The 15 manual/artifact
  ignores are not qualified results and no missing model fixture was downloaded.
- Strict docs, alloc+serde bare-metal schema and canonical fmt/projection drift/
  foundation no-std/default and toolkit-free all-feature Clippy/Windows cfg pass.
  These are compilation/lint checks, not physical hardware results.
- Initial detached PATH missed actionlint. The canonical gate was rerun with
  `/home/brianklam/go/bin` on PATH and passed actionlint. ShellCheck is absent;
  shell workflow lint is explicitly skipped/warned, not reported passed.
- A separately executed Python `struct.pack`/blake3 preimage calculator agrees
  on pinned f32-source and f64-Gram vectors. This is candidate cross-language
  agreement, not independent owner acceptance or a release identity freeze.

Tests compare actual output payloads to the unchanged controller, replay identical
roots/views, check source/metric/plan/name changes and signed-zero bits, observe
real one/two-thread Rayon pools, and reject invalid contexts/sources, modified
roots/events, unsupported versions, unknown fields, incomplete/mismatched/duplicate
level inventories and zero pool size. Generic hashing of a structurally false
observation does not make its typed replay valid.

Initial failure retained: three tests passed, one failed because serde Number
distinguished float/integer payload spellings after JCS replay. The view was
corrected; no numerical threshold, golden or existing gate was weakened.

Custody destination:
`/home/brianklam/Projects/Tritium/archive/verification/recorded-direct-fit-107fb0b3-20261010`.
It retains committed source/projections, numerical support source, private ADR/
full plan, report, scripts, independent calculator, machine data, initial failure,
verification and publication logs, with a separately executed custody check.

## Remaining full scope

This observation does not assign `LevelFit::Direct`, invent a completed-run
reference, write a model package, bind original HF source files, or clear P3.
Logical element limits are not total RSS/solver/source/tokenizer/KV/device memory.
No model-quality, compression, performance, SOTA or public release verdict follows.

1. Complete model QuantPlan/WeightSource/TensorSink/evidence engine, exact stored
   source dtype/content and asset/inventory binding, sealed fit ancestry, real
   `.trit` writing/dedup and consumer execution; common feedback/fitters/calibration/
   allocator; migrate convert/quantize/Qwen/Python to thin surfaces with parity.
   SALT/QTEA/ExTernD/TWLA/frontier GPU/out-of-core/refinement remain required.
2. Owner-frozen profile/all projections; HF content/capability/full-memory admission;
   shards/per-tile/entropy/stream/mmap; exact legacy migration then gated deletion.
3. Native SIMD/GPU and all-backend conformance; resident Executor/Runner/
   ArchitectureSpec, hybrid DeltaNet/MTP, complete memory and quiet performance.
4. Stage7 authorized recipe/solver/refinement/auxiliary evidence; official Qwen
   language+MTP ternary/refined quality, task retention, runtime, bytes and reproduction.
5. Physical backend/browser/WASM/MCU/distributed training matrix, lifecycle/fault
   behavior and no hidden readbacks; compile-only results do not satisfy these.
6. Serving residency/KV/cancellation/concurrency/fault/latency/plugins, deployment,
   security/observability and explicit human acceptance.
7. Same-source RC/packages/wheels/ONNX/Colab/compatibility, audited zoo, docs/tutorials/
   community, independent second-machine/operator/security/signing clearance,
   and explicit human public activation. Native Rust frontend v1.2 and arbitrary
   trainable ONNX v1.3 remain distinct. Fresh BitNet quiet decode/repack/nsys is UNKNOWN.

No older caches, campaign/model payloads or foreign WIP were changed. Per-run
scratch and the owned branch are removed after verified durable custody/merge;
the reused checkout, existing shared build/hook caches and durable archive remain.
