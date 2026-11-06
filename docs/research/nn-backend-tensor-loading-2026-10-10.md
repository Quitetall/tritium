# Backend-owned tensors in HF model execution

## Source and implementation scope

ADR 0044 D8-D10 calls for every linear/embedding consumer to use the same
backend tensor contract, with truthful capability reporting, payload admission
and registered evidence. Base: `9186200df14013adc69f0e8387eccb1e38d2745a`.
Code: `762054f52e2b7d04dd5a1bbad858e77f41088e36`.

`TensorLoader` binds a backend, upload policy, shared lifetime payload budget
and evidence span. It validates nonempty geometry and finite dense weights or
the admitted additive schema/exact scales, then calls the runtime's existing
logged/admitted upload. `Linear` retains private validated geometry, an opaque
shared backend handle and its capability declaration. Clones share one payload
and reservation. Matmul and original-coordinate gather use the backend contract,
including the existing forward/inverse basis semantics; no backend downcast or
host fallback is introduced by this module. Repeated callers can provide basis
scratch; the convenience forward allocates scratch fallibly.

The existing projection and token-table consumers can now hold this handle.
Embedding gather and tied logits share one handle; backend-owned weights never
silently become inspectable dense masters. Legacy training/calibration extraction
continues to require the old inspectable master, rather than copying weights
back or pretending an opaque device tensor is trainable in that adapter.

`ModelWeights::load_hf_admitted` and `ModelRunner::from_hf_admitted` connect these
weights to real seek-backed HF safetensors loading and the existing model runner.
The canonical standard-model assembly and HF config/feature detection are reused.
Every 2D source tensor is read/uploaded in turn and its host source released;
1D norms/bias/QK-norm remain validated host vectors. Untied heads upload their
own weights; tied heads share the token table. Failed model assembly drops every
unpublished handle and lease, retaining earlier successful tensor observations
without calling them a whole-model success.

This is an explicit transitional path toward the single loader/weights/runner,
not a claim that the legacy loaders have been consolidated. Existing `load_hf`
and `from_hf` still build the inspectable dense master used by fitting/training
and the unchanged numerical oracle. Existing activation modes, format/schema
contracts, admitted laws, core/backend arithmetic, resident CUDA decoder and
kernel sources are unchanged. No dependency versions change; Cargo only adds
the existing evidence/schema edges to the NN dependency list.

## Tests and evidence boundaries

The fixed additive SignedRht tensor executes through a real NN `Linear`, token
gather and tied head. CPU payload is ten bytes, while both consumers share one
ten-byte reservation until final-owner drop. Invalid/nonfinite weights, shapes,
overflow, bad IDs, foreign backend handles and budget exhaustion reject without
output mutation or successful-upload observations. Backend-free token calls
reject opaque storage rather than falling back to a dense copy.

File-backed fixtures have config plus two actual indexed safetensors shards.
Twelve standard-transformer configurations cover Llama, Qwen2 QKV bias and
Qwen3 QK normalization, tied/untied heads, and head dimensions four/six (including
query width distinct from hidden width). CPU and reference execute all twelve,
with bit-for-bit last-position logits against the unchanged dense runner for
three-token prefill and two cached decode steps: 24 loaded models, 72 comparisons.
Exact total payload admission, upload counts and registered hash-chain verification
are checked; dropping each runner releases all payload reservations.

Additional file-backed failures cover a budget exhausted after the embedding,
a missing later projection, non-finite final norm and an unregistered evidence
schema. All unpublished weights are released. Earlier successful observations
remain observations, not a model receipt or qualification claim.

The CUDA test is feature-gated and must fail instead of skip when
`TRITIUM_REQUIRE_CUDA=1`. It executes six file-backed configurations with explicit
head dimension six, checking the same unchanged oracle bit-for-bit. Fixture
files are owned per-test scratch and removed; these are synthetic small model
execution checks, not actual pretrained Qwen quality, throughput or memory gates.

The initial test failed on the missing loading interface and embedding
constructor. Test authoring then corrected a `usize`/`u64` budget type mismatch;
no existing expected numerical vector or tolerance changed.

## Verification status

Committed verification invocation `954c6dfc8b784bd6ab31d60deb465d2b`
reached terminal success: MainPID zero, SubState exited, Result success and
ExecMainStatus zero. Shared SSD target and two build jobs were used.

```sh
cargo test --locked -p tritium-nn --lib --test tensor_loading
TRITIUM_CHECK_ONLY=0 TRITIUM_REQUIRE_CUDA=1 cargo test --locked \
  -p tritium-nn --features cuda --test tensor_loading
cargo test --locked -p tritium-schema -p tritium-core -p tritium-spec \
  -p tritium-format -p tritium-testkit -p tritium-cpu -p tritium-runtime \
  -p tritium-evidence -p tritium-cuda -p tritium-wgpu --all-targets
cargo clippy --locked -p tritium-nn --all-targets -- -D warnings
sh scripts/verify-gates.sh prepush
```

NN regression: 166 unit plus four integration tests passed, zero failed.
Real CUDA integration: five passed, zero failed; six small GPU model executions
produce 18 bit-exact logit comparisons. Device: RTX 4090,
`GPU-1790118a-a6d7-4eaf-fcac-dcacac5f4351`, driver 615.71.09. The strict CUDA
requirement prevents an unavailable-device skip from passing this invocation.
Backend/foundation regression: 524 passed, zero failed, six declared ignores
(CPU microbenchmark and model/master-dependent Qwen tests). Default GPU bodies
in that separate command execute zero cases; they are not new GPU qualification.
Scoped Clippy and canonical prepush pass format/projection drift, actual
bare-metal compile, default/all-feature workspace Clippy, Windows GNU and
actionlint. ShellCheck is absent/warned, not passed. Check-only optional-feature
Clippy is compilation evidence, not physical backend execution.

Source/log/script/hardware custody archive:
`/home/brianklam/Projects/Tritium/archive/verification/nn-tensor-loading-762054f5-20261010`.
The standalone SHA-256/original-Git verifier establishes custody only.

All five hosted workflows on the previous `9186200d` revision pass. New-source
hosted CI and independent empirical clearance are separate obligations.

The P0/P4 BitNet decode benchmark fixture is absent at
`/home/brianklam/.cache/tritium-models/bitnet-2b4t-gguf/ggml-model-i2_s.gguf`.
The required five-run quiet-box performance/trace gate is **UNKNOWN**, not passed.
No large fixture download, campaign, cloud job or expensive model fitting was run.

## Remaining release work

This is P4 prerequisite implementation, not completion of P4 or the public goal.
The budget governs owned weight payload only: source decoding, host vectors,
activation/basis scratch, KV, allocator/handle/driver overhead and whole-model
peak preflight remain separate. This host-orchestrated path makes no resident
decode-speed, no-transfer, physical-memory or whole-model compression claim.

Remaining: `.trit` typed manifest/per-tile/stream/shard/migration and canonical
whole-model loading; one quantization engine/calibration/direct-fit allocator;
native/SIMD backend execution and resident executor/architecture consolidation;
thin CLI/Python/serve/FFI/ONNX/browser/adapters and legacy deletion; full-memory
preflight; Stage-7 recipe freeze and official Qwen language-plus-MTP PTQ/refined
quality/runtime/physical-byte/reproduction gates; physical browser/backend/
distributed training; hardened serving/deployment/security; same-source RC
packages, model zoo, documentation/community; second-machine reproduction,
independent release clearance and explicit human activation/publication smoke.
No phase, empirical verdict, signature or human authorization is produced here.
