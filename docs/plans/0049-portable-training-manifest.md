# 0049 — Portable training manifest and native backend conformance

Status: **IN PROGRESS** (2026-07-20)

- **Decision:** [ADR 0033](../adr/0033-v11-full-public-release.md)
- **Parent:** [plan 0044](./0044-v11-full-public-release.md)
- **Dependencies:** [plan 0045](./0045-torch-reference-conversion.md) schema
  vocabulary; portable Conv2d reference in `tritium-train`
- **Successor:** plan 0050 browser `WebTrainingSession`

## Goal

Freeze one language-neutral `TrainingOpManifestV1`, one fallible
`TrainBackendV1` seam and one canonical vector corpus. Every declared training
backend must execute that same corpus without hidden host execution. Receipts,
not compile success or skipped tests, prove support.

This plan owns semantic portability. Plan 0050 owns browser product packaging
and WebGPU session orchestration; plan 0051 owns public package publication.

## Non-goals

- No public native dynamic `Tensor`/`Module` interface; ADR 0033 reserves that
  for v1.2.
- No arbitrary JavaScript autograd graph; plan 0050 exposes a compiled session.
- No inference-only fallback counted as training support.
- No relaxed operation set for constrained targets. Shape ceilings may differ;
  semantics may not.

## Contract placement

`tritium-spec` owns language-neutral descriptors and strict schema parsing. It
is the seam shared by training and all backend adapters; no backend
implementation enters that crate. `tritium-train` owns reference semantics and
the CPU reference adapter. Device crates own their adapters.

Canonical fixtures live under `spec/training/v1/`. Source-tree tests read that
one copy; packaged copies must be byte-identical and digest-verified. Divergent
generated fixtures are forbidden.

Stable interface:

```rust
pub struct TrainingOpManifestV1;

impl TrainingOpManifestV1 {
    pub const SCHEMA_ID: &'static str;
    pub const SCHEMA_VERSION: u32;
    pub fn operations() -> &'static [TrainingOpDescriptorV1];
    pub fn parse_json(bytes: &[u8]) -> Result<Self, TrainingManifestError>;
    pub fn canonical_json() -> &'static [u8];
}

pub trait TrainBackendV1 {
    fn capabilities(&self) -> TrainCapabilitiesV1;
    fn execute(
        &self,
        request: TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<TrainReceiptV1, TrainBackendError>;
}
```

`execute` is intentionally one deep interface. Operation dispatch, validation,
buffer ownership, first-order VJP, optimizer state mutation and receipt
construction remain behind it. Backend-specific convenience methods stay
private.

Manifest JSON uses UTF-8, LF, two-space indentation, one terminal newline and
fixed field/descriptor order. It contains no JSON floats. Its identity is the
BLAKE3 digest of exact canonical bytes. Each descriptor contains exactly
`id`, `category`, `forward`, `vjp`, `mutates`, and `checkpoint_planes`; parsers
reject unknown or missing fields rather than normalize them.

## Frozen operation registry

Every ID is lowercase ASCII and permanent within schema v1. Descriptor order is
canonical. Graph operations declare `forward` and `first_order_vjp`; stateful
operations declare mutation and checkpoint planes explicitly.

### Graph

- `graph.ste_surrogate`, `graph.salt_ste`, `graph.lsq_ste`, `graph.fsq`
- `graph.dense_matmul`, `graph.ternary_matmul`, `graph.transpose`
- `graph.embedding_gather`, `graph.slice_cols`, `graph.concat_cols`
- `graph.detach`, `graph.scale_const`, `graph.bias`, `graph.add`, `graph.mul`
- `graph.conv1d`, `graph.conv2d`
- `graph.relu2`, `graph.silu`, `graph.rmsnorm`, `graph.softmax`
- `graph.causal_mask`, `graph.rope`, `graph.attention`
- `loss.mse`, `loss.softmax_cross_entropy`

### Optimizer and lifecycle

- `optimizer.sgd`, `optimizer.adamw`, `optimizer.cautious_adamw`
- `optimizer.int8_adamw`, `optimizer.muon`
- `lifecycle.checkpoint`, `lifecycle.resume`
- `lifecycle.export`, `lifecycle.reload`

`graph.attention` is a projection-free scaled-dot-product attention boundary:
Q/K/V projection and RoPE remain separate graph operations, while causal or
noncausal GQA forward/VJP can map directly to a fused backend kernel. SGD
receives a portable reference implementation before registry closure.
`optimizer.sgd` is stateless
plain SGD with `parameter -= lr * gradient` in f32 order; momentum and weight
decay require distinct future operation identities.

## Slice 1 — schema, registry and exhaustive audit

Status: **DONE** — the Rust registry/parser and exhaustive source audit are
landed, and Python plus dependency-free TypeScript parsers accept the same
semantics, reject duplicate/unknown/type/order drift, and re-emit byte-identical
canonical JSON. Python passed the complete 92-test suite; Deno type-check and
three parser tests passed. Slice 2 and the CPU semantic matrix are complete;
accelerator adapters are next.

The Slice 2 tracer corpus now carries 114 cases across all 35 operations. In
addition to the primitive and shape clusters, it covers `graph.ste_surrogate`,
bounded multi-plane `graph.salt_ste`, `graph.lsq_ste`, configurable `graph.fsq`,
grouped/depthwise asymmetric `graph.conv1d` and `graph.conv2d`, `graph.dense_matmul`,
scale-bearing `graph.ternary_matmul`, `graph.rmsnorm`, row-wise `graph.softmax`,
`graph.causal_mask`, zero-scratch `graph.rope` with target-independent `u32`
positions, and projection-free causal/noncausal grouped-query
`graph.attention`, plus `loss.softmax_cross_entropy` forward/VJP semantics,
`graph.transpose`, repeated
`graph.embedding_gather`, `graph.slice_cols`, and dynamic-role
`graph.concat_cols`, each in forward and VJP phases. Stateful vectors exercise
resumed AdamW, masked Cautious AdamW, block-boundary quiet/spike Int8 AdamW,
and rectangular Muon state transitions. Int8 optimizer planes use canonical
two's-complement bytes for q8 state and f32 block scales. Lifecycle vectors bind
multi-leaf TOPT checkpoint/resume and strict SALT V2 package export/reload.
Error sentinels cover shape
mismatch, invalid quantizer geometry/configuration, out-of-range tokens, slice
bounds, concat geometry, non-finite input, and an intentionally malformed
duplicate-input request.
A generic testkit runner poisons success outputs, preserves error sentinels,
grades structured error identity, independently recomputes request/output
receipt digests, and binds every successful receipt to the exact corpus bytes.
This closes Slice 2 corpus coverage and the CPU semantic matrix; accelerator and
constrained-target receipts remain required for release closure.

CUDA adapter work has started against the same seam. Current actual-RTX-4090
evidence covers all 114 canonical cases across 35/35 operations: STE surrogate, SALT
STE, LSQ, FSQ, dense and scale-bearing ternary matmul, transpose, embedding,
column slice/concat, detach,
scale, bias, add, multiply, ReLU2, SiLU, RMSNorm, softmax, causal mask, and RoPE.
MSE and softmax cross-entropy forward/VJP plus SGD, AdamW, cautious AdamW, and
blockwise int8 AdamW steps are resident as well. Cautious AdamW uses a resident
two-pass masked update; int8 AdamW keeps blockwise moments and scales resident.
Muon uses a deterministic resident Newton--Schulz kernel with bounded global
scratch matching the frozen ledger. Grouped Conv1d supports asymmetric padding,
stride, dilation, deterministic forward/VJP reductions, and bounded resident
scratch. Grouped Conv2d matches the frozen 32-row tiling order across asymmetric
NCHW forward/VJP geometry. Grouped-query attention preserves the canonical
stable-softmax order, reverse-head shared-KV VJP accumulation, causal masking,
and bounded resident probability scratch. The four lifecycle operations share
the canonical host-visible control-plane serializer and strict parser; this is
byte artifact handling, not CPU tensor execution or an accelerator fallback.
STE, LSQ, and FSQ use dedicated forward/VJP kernels, including deterministic
row-order LSQ alpha reduction and seeded stochastic FSQ. SALT supports the full
1--64-plane contract with row-sequential reductions and one-row scratch. Bias
and ReLU2 use dedicated resident kernels; training RoPE positions are unsigned
end-to-end, matching the frozen `u32` contract rather than narrowing to signed
indices. Each success emits a physical-device-bound receipt; the adapter
advertises only this proved subset and never delegates to CPU. This is
development evidence until the receipt artifact is sealed on actual hardware;
the CUDA semantic matrix itself is complete.

The constrained WASI/WASM adapter is complete for the frozen corpus. It compiles
the deterministic scalar semantic executor into the guest rather than calling a
host fallback, advertises all 35 operations, enforces 8 MiB per-buffer and
64 MiB aggregate caller-payload ceilings before mutation, and rebinds receipts
to the actual engine identity. Native structural tests and an actual
`wasm32-wasip1` run under wasmtime 46.0.0/Cranelift both pass all 114 cases.
This closes WASI/WASM semantic parity; browser WASM packaging remains plan 0050,
and MCU constrained-target parity remains open.

Native wgpu work has begun on the RTX 4090 Vulkan adapter. Current evidence
covers 77 vectors across STE surrogate, multi-plane SALT STE, LSQ, dense and scale-bearing ternary matmul, embedding gather, transpose, column slice/concat, detach, constant scale, bias, add, multiply, ReLU2,
SiLU, RMSNorm, causal masking, row softmax, MSE loss, SGD, AdamW, cautious AdamW, int8 AdamW, Muon, and all four lifecycle operations. Pointwise tensor work executes through resident WGSL
storage buffers; lifecycle uses the shared canonical control-plane byte
implementation. AdamW uses ordered storage passes to preserve the frozen f32
rounding points without a host tensor roundtrip; cautious AdamW adds a native
atomic aligned-element reduction and device-side rescaling. Int8 AdamW keeps
packed state and 256-element scale reductions on device; integer
guard/round/sticky division removes driver-dependent WGSL quotient drift.
Muon uses one packed device workspace, serial reference-order folds, and
storage rounding barriers through Newton-Schulz orthogonalization.
SALT preserves ascending-column AbsMean folds and the bounded one-row scratch
contract while keeping all residual planes device-resident.
Receipts bind the physical NVIDIA adapter, and no additional
tensor operation is advertised before its WGSL path passes the corpus. ROCm
implementation/evidence is currently target-blocked on this host: no AMD device
or ROCm compiler/runtime tools are installed.

Edits:

- Add strict manifest types/parser to `tritium-spec/src/training.rs`.
- Add `spec/training/v1/manifest.json` as canonical bytes.
- Add Python dataclasses/parser under
  `crates/tritium-py/python/tritium/portable/manifest.py`.
- Add dependency-free strict TypeScript parser under
  `bindings/typescript/src/training_manifest.ts` with a strict `tsconfig.json`.
- Add a source audit in `tritium-train` mapping every public Tape operation,
  optimizer and lifecycle seam to one manifest ID. Additions without registry
  entries fail CI.

Failure gates:

- unknown schema ID/version, fields, operation IDs, duplicates or order fail;
- missing/extra registry entries fail;
- malformed UTF-8, JSON, booleans or capability fields fail;
- Rust/Python/TypeScript re-emit byte-identical canonical JSON.

## Slice 2 — canonical semantic vectors

Add `spec/training/v1/vectors/` with small, adversarial f32 cases for every
operation. Each case binds:

- manifest digest, operation ID and deterministic case ID;
- exact input/output shapes and little-endian f32 payload digests;
- forward output, every first-order input gradient and mutated state;
- tolerance policy (`bit_exact` or fixed absolute/relative bounds);
- expected structured error for invalid cases;
- peak temporary-byte ceiling where bounded scratch is contractual.

Vectors include zeros, ties, non-finite rejection, ragged groups, grouped and
depthwise convolution, asymmetric padding, stride/dilation, repeated embedding
indices, masked rows, optimizer quiet/spike blocks, resume after mutation and
export/reload parity. Vector generation is a checked-in tool; fixture review
uses its human-readable source plus output digest.

## Slice 3 — fallible seam and CPU reference adapter

- Add owned/borrowed tensor descriptors with checked dtype, rank, shape, byte
  count, aliasing and mutability rules.
- Add structured `TrainBackendError`; no public request path may panic.
- Route all current Tape forward/VJP semantics through a CPU adapter without
  changing numeric order.
- Add portable SGD. Adapt AdamW, CautiousAdamW, Int8AdamW and Muon through the
  same request interface.
- Adapt `TOPT` checkpoint/resume and canonical hard-artifact export/reload.
- Emit receipts binding backend build, physical device, manifest/vector digest,
  executed operation set, dtype, shape ceilings, host-transfer counters, peak
  resident/scratch bytes and result digest.

Gate: CPU executes every valid and invalid vector. Direct existing reference
tests and adapter results match. Catch-unwind tests prove malformed requests
return errors.

## Slice 4 — accelerator adapters

Implement adapters in this order:

1. CUDA
2. ROCm
3. Metal
4. native wgpu

No adapter may call CPU reference execution after setup. Profilers must show
zero steady-state device-to-host/host-to-device tensor transfers and zero
global synchronization not required by receipt finalization. Unsupported dtype
or shape returns a capability error; fallback never greens a device gate.

Each adapter runs full vectors on actual hardware and emits a signed or
content-addressed receipt. CI artifact presence without device identity fails.

## Slice 5 — constrained adapters

- WASI/WASM: deterministic f32 fallback, bounded arena, no filesystem or clock
  dependence inside execution.
- MCU: no allocator during prepared execution; declared static arena and shape
  ceilings; full operation set exercised at bounded shapes.

Cross-compile or emulator evidence is labeled structural only. v1.1 requires
actual target receipts. Zero-test or skipped lanes fail.

## Slice 6 — capability and release evidence

- Validate all receipts against exact manifest/vector digests.
- Generate capability/performance tables solely from admitted receipts.
- Reject duplicate backend identities, stale builds, fallback identities,
  partial operation sets and missing physical-memory counters.
- Feed canonical schema and backend contract into plan 0050 without widening
  it from TypeScript.

## Verification cadence

```bash
cargo fmt --check
cargo test -p tritium-spec -p tritium-train
cargo clippy -p tritium-spec -p tritium-train --all-targets -- -D warnings
PYTHONPATH=crates/tritium-py/python pytest -q \
  crates/tritium-py/tests/test_training_manifest.py
npx tsc -p bindings/typescript/tsconfig.json --noEmit
git diff --check
```

Backend lanes additionally run their feature-gated conformance binary on real
hardware. Final gate admits receipts with one exact manifest/vector digest.

## Review and commits

Every commit receives mandatory lamu `review_commit` with this file as
`plan_file`; findings are verified before fixes.

Expected commit series:

```text
docs(plan-0049): freeze portable training work order
feat(spec): freeze training operation manifest v1
test(train): publish canonical training semantic vectors
feat(train): add portable CPU training adapter
feat(cuda): conform portable training backend
feat(rocm): conform portable training backend
feat(metal): conform portable training backend
feat(wgpu): conform portable training backend
feat(wasm): conform portable training backend
feat(mcu): conform portable training backend
docs(training): generate backend receipts and capability table
```

## Done criterion

Rust, Python and TypeScript accept and re-emit one canonical schema; exhaustive
audit covers every frozen operation; all valid/error vectors pass through the
fallible seam; CPU, CUDA, ROCm, Metal, wgpu, WASI/WASM and MCU produce actual
target receipts for every operation; generated capability tables match those
receipts; plan 0050 consumes the frozen contract unchanged.
