# Shared HF and typed `.trit` model execution

## Scope and source

Base: `3d6f8e8992543cdbaae6e474aeaf621698f8a017`.
Implementation: `29570d06acdc90748d5df09b279a7db570eb5e82`.
Verified code tree: `81d244c53849a5561ea154225e50b4f029d526ae`.
Architecture authority: accepted ADR 0044 D8-D10/D12.
Detailed typed profile: ADR 0054 **PROPOSED**, without owner acceptance.
This is experimental standard-transformer execution, not a profile freeze,
P4 clearance, pretrained-model qualification or public-release activation.

`ModelSource::{HuggingFace,Trit}` selects an indexed HF directory or immutable
typed package bytes with an explicit complete level. `ModelWeights::load_admitted`
and `ModelRunner::from_source_admitted` share admission, upload evidence and the
existing standard-model builder. Existing HF admitted methods delegate to this
selector. Attention feature detection is shared, not a second architecture oracle.

With the `tokenizer` feature, `tritium_nn::load` returns a `LoadedModel` containing
the actual HF tokenizer, architecture variation data and the shared runner. It
parses `tokenizer.json` and `tokenizer_config.json`, resolves real special tokens,
and verifies every vocabulary ID fits the embedding rows. Weights-only helpers
remain explicitly token-level; they do not fabricate a tokenizer or load receipt.

Typed loading authenticates the package once and uses its borrowed assets and
tensor bindings. Config must match manifest architecture and existing supported
geometry/features. F32/F16/BF16 stored dense values decode to an exact f32 upload
view. Uniform/raw additive sources reuse the existing codec and backend upload;
embedding and tied-head ownership/basis binding use the same tensor handles.
Norm/bias vectors remain dense host data. Temporary decoded tensor sources are
dropped; no retained dense model shadow or filesystem asset extraction is added.

Missing/wrong shapes, unexpected selected-level tensors, upload/budget/evidence
failure and invalid tokenizer vocabulary reject the load and release unpublished
handles. Successful earlier upload observations remain in the evidence log; they
are not a whole-model verdict. Unknown tensor coverage is checked after assembly,
and vocabulary bounds after weights load, not before every allocation. Tokenizer
parsing happens before uploads. Input limits are 256 MiB tokenizer JSON and 16 MiB
tokenizer config; HF reads check regular files, metadata and actual bounded bytes.
These limits do not account for parsed tokenizer memory or establish security.

The package reader still explicitly rejects unsupported shard/per-tile/entropy
payloads. Caller-owned package bytes are not mmap/streaming. This slice does not
remove legacy loaders, backend downcasts or resident runner specialization; it
does not add hybrid DeltaNet/MTP execution, semantic ModelId, a load receipt view,
full source/vector/scratch/KV/driver memory admission or native additive CUDA.

## Verification and claim limits

Detached exact-tree verification invocation:
`cf94241ff6cb444f8632b320996a82ce`.
Terminal state: MainPID zero, SubState exited, Result success, ExecMainStatus zero.

- Foundation/backend regression: **541 passed, 0 failed, 6 declared ignores**.
  Feature-disabled accelerator bodies in this command are not physical results.
- NN default: **166 unit and 8 integration tests passed**.
- NN tokenizer: **167 unit and 10 integration tests passed**.
- CUDA plus tokenizer: **12 integration tests passed**, with actual CUDA cases
  enforced by `TRITIUM_REQUIRE_CUDA=1`, `TRITIUM_CHECK_ONLY=0`.
- Scoped real-feature Clippy, default/tokenizer rustdoc with warnings denied,
  and canonical prepush passed. The latter includes format/projection drift,
  bare-metal foundations, default/check-only all-feature workspace Clippy,
  Windows GNU compile and actionlint. ShellCheck is missing/warned, not passed.

New public-interface tests use real package writing/reading, existing synthetic
two-shard HF fixtures and the unchanged legacy HF runner as numerical oracle.
Additive oracle materialization uses core gather semantics; tests do not duplicate
attention, normalization or rotation math. CPU/reference coverage includes dense
Llama/Qwen2/Qwen3 with tied/untied heads, Qwen3 independent head dimensions,
F16/BF16 stored sources, D2/B3/S34, Identity/Hadamard/SignedRht, multi-token prefill
and repeated cached decode. Dense results compare exact f32 bits. New additive
comparisons use a 1e-5 absolute/relative bound; old vectors/tolerances are unchanged.

Real CUDA checks cover dense and explicitly emulated additive package execution,
real-tokenizer loading/generation and unchanged HF parity. Successful events assert
the execution tier; additive CUDA rejects native-only before upload. Actual device:
RTX 4090, `GPU-1790118a-a6d7-4eaf-fcac-dcacac5f4351`, driver 615.71.09.
These small synthetic models do not establish pretrained language quality,
throughput, physical peak memory, native additive CUDA speed or SOTA status.

Failure tests cover absent levels/tensors, extra coverage, incompatible geometry,
architecture mismatch, exhausted budget, unregistered evidence, corrupted package
bytes, missing/invalid tokenizer assets, absent EOS, out-of-range vocabulary and
oversized config. Every failed load releases the payload budget. Tests also verify
registered JSONL hash chains and tied-handle lifetime behavior.

Initial test-authoring failures (ID integer type and inferred tokenizer map type)
and Clippy's large enum finding are retained. Strict rustdoc exposed a pre-existing
unqualified `SaltLinear::set_activation_group` link; the final code qualifies it.
The original failed job and final rerun are both retained; no warning/tolerance/gate
was weakened. All five hosted workflows for prior source `3d6f8e89` pass. Hosted
checks of the new publication and independent empirical clearance are separate.

```sh
cargo test --locked -p tritium-schema -p tritium-core -p tritium-spec \
  -p tritium-format -p tritium-testkit -p tritium-cpu -p tritium-runtime \
  -p tritium-evidence -p tritium-cuda -p tritium-wgpu --all-targets
cargo test --locked -p tritium-nn --lib --test tensor_loading
cargo test --locked -p tritium-nn --features tokenizer --lib --test tensor_loading
TRITIUM_CHECK_ONLY=0 TRITIUM_REQUIRE_CUDA=1 cargo test --locked \
  -p tritium-nn --features cuda,tokenizer --test tensor_loading
cargo clippy --locked -p tritium-nn --all-targets --features cuda,tokenizer -- -D warnings
RUSTDOCFLAGS=-Dwarnings cargo doc --locked -p tritium-nn --no-deps
RUSTDOCFLAGS=-Dwarnings cargo doc --locked -p tritium-nn --features tokenizer --no-deps
sh scripts/verify-gates.sh prepush
```

The configured BitNet decode fixture remains absent; required quiet runs and
kernel-trace baseline are **UNKNOWN**, not waived. No large model download,
campaign/refitting or cloud compute ran. Existing campaign artifacts are unchanged.
Custody archive:
`/home/brianklam/Projects/Tritium/archive/verification/model-loading-81d244c5-20261010`.
Its original-Git/hash verifier proves retained-byte custody, not qualification.

## Remaining full-release work

This is P1/P4 prerequisite progress; all P0-P7 phase gates remain incomplete.

1. Owner-accepted typed profile; semantic identities/frozen vectors and all
   projections; full loader/receipt, shards/per-tile/entropy/stream/mmap,
   verified fixture/Qwen migration and eventual legacy deletion.
2. Unified fitting engine, calibration/direct-fit allocator and frontier solvers,
   GPU/out-of-core/refinement; thin PTQ/QAT/Python/HF/CLI/ONNX/FFI/Burn/Candle/
   browser/MCU surfaces over one core and evidence views.
3. Native/SIMD backends, resident executor/architecture/runner consolidation,
   hybrid DeltaNet/MTP, full-memory preflight and quiet decode/trace performance.
4. Stage-7 recipe/solver/refinement/auxiliary evidence and authorized freeze;
   official Qwen language/MTP quality, task retention, runtime, physical bytes
   and independent reproduction. Synthetic tests do not satisfy these gates.
5. Physical backend/browser/distributed-training matrix, browser 117-op/lifecycle/
   fault checks without hidden readback, distinct WASM/MCU qualification.
6. Serving cancellation/residency/KV/concurrency/fault/latency, plugins, deployment,
   security/observability and explicit human acceptance.
7. Fresh same-source RC/packages/ONNX/Colab compatibility, docs/tutorials/community,
   audited model zoo, second-machine/operator replay, independent release/security/
   signing gates, human activation and publication smoke.

The full goal stays active. Old-cache cleanup confirmation is pending; none of
those caches or campaign outputs were deleted. Only this run's scratch is cleaned
after durable custody verification; the existing pinned checkout/cache is retained.
