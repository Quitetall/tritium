# Direct packed QTEA CUDA primitives (experimental)

`tritium-cuda`'s opt-in `cuda` feature provides `upload_qtea`,
`qtea_matmul_ordered_into` and `qtea_gather_into` on `CudaBackend` (proposed ADR0172).
The handle owns exactly the canonical checked V1 wire bytes on the device. It
does not keep a dense weight shadow, change wire/numerical identities, run a
fitter, select a method automatically or silently fall back to CPU.

```rust,ignore
let resident = backend.upload_qtea(checked_tensor.view(), encoded_cap)?;
let limits = tritium_cuda::QteaCudaLimits {
    max_device_staging_bytes: device_staging_cap,
    max_host_staging_bytes: host_staging_cap,
};
let ledger = backend.qtea_matmul_ordered_into(
    &resident, &inputs, batch, &mut output, limits)?;
```

Matmul consumes `[batch,columns]` and writes `[batch,rows]`; gather consumes u32
original row IDs and writes `[ids.len(),columns]`. Positive caps are required,
including for empty calls. Shapes, indices, finite inputs, u32 kernel output
counts and caps are checked before device allocation. Geometry/offsets come from
the format's checked layout, never caller-supplied offsets. Foreign-context
handles refuse. Private fallible host staging makes output unchanged on refusal,
including kernel/readback errors and arithmetic overflow. Successful calls
synchronize; input/index, device output and host staging are not retained.

The ordered kernel uses separate RN f32 operations and accumulates products in
original column order. Gather reconstructs selected weights, including packed
1:4 FP8 residuals. Unaligned fields are read bytewise. Compile flags preserve
subnormals and disable multiply/add contraction. This is an ordered reference
path, **not a tuned tensor-core kernel or a performance promise**.

## Limits and evidence boundaries

The upload cap covers only one handle's encoded allocation request. Per-call
device staging is input/index bytes plus output bytes. Host staging bounds the
actual Vec element capacity. The returned logical ledger includes the encoded
bytes, per-call requests, their sum, host capacity and zero dense-weight bytes.
It excludes driver/context/module/events, shared memory, allocation rounding,
caller buffers and other handles. It is not physical VRAM/RSS measurement,
whole-model peak admission or a compression/release receipt.

The primitives also support explicit host-orchestrated NN/embedding/model loading
and CLI/loopback serving (proposed ADR0173), described below. Existing host profiles
remain CPU-only. Resident activation chaining, GPU/out-of-core fitting, large-model
quality/MTP, speed, other devices and independent release gates remain open. No
experimental method is promoted by inference support.

## Explicit host-orchestrated models

`tritium-nn` with `host-qtea,cuda` adds `CudaQteaLinear`, `Projection::CudaQtea`,
`TokenEmbedding::from_cuda_qtea`, `ModelWeights::load_qtea_cuda` and, with
`tokenizer`, `LoadedModel::load_qtea_cuda` / `load_qtea_hybrid_cuda`.
One encoded device allocation per unique blob is shared across projections and
embedding/tied unembedding. Loading streams one encoded host tensor at a time;
there is no complete retained CPU matrix set or dense matrix shadow. All QTEA
matrix operations execute on the concrete CUDA backend; a CPU/wrong-context
backend refuses. Existing CPU constructors and numerical identities are unchanged.

`CudaQteaModelLimits` separates aggregate unique encoded `max_device_weight_bytes`
and per-call `max_device_staging_bytes`. Its `host` field carries existing source/
per-owner limits, but `host.max_weight_bytes` now explicitly bounds retained
preserved F32 Vec capacities plus the one current encoded upload Vec. It does not
count the aggregate device weights again. The complete preserved-payload sum plus
largest encoded tensor is planned before device weight allocation, and actual
capacities are checked while loading. `host.max_staging_bytes` bounds private
per-call host output staging. Metadata, transient preserved-vector byte decoding,
attention/state/activation/cache memory, concurrent calls/models and driver/
allocator overhead remain separate and are not physical admission.

The shared CLI/server JSON request keeps existing fields and adds a required
nonnull `cuda` object only in the new profiles:

```json
{
  "profile": "tritium.qtea-cuda-ordered-host-load.v1",
  "cuda": {"device": 0, "max_device_weight_bytes": 1073741824,
           "max_device_staging_bytes": 16777216}
}
```

This is a fragment, not a complete load request. Supply all existing path/source/
read/host-load/staging/stream fields. The hybrid profile is
`tritium.qtea-hybrid-cuda-ordered-host-load.v1` and also requires the existing
positive `max_preserved_tensor_elements`. Device caps must be positive. Host
profiles reject the CUDA object; missing/null CUDA selection refuses. Builds
without CUDA refuse the new profiles before model/device loading. Both CLI and
server require `--backend cuda` to match a CUDA profile; host profiles require
`--backend cpu`. No automatic selection or fallback is available. Serving remains
loopback-only and unqualified, without speculative/batching/resident modes.

ModelLoaded observations say `qtea-cuda-ordered-host-orchestrated`: projections,
gather and unembedding execute on GPU, but attention/norms/state/orchestration stay
host-side and activations cross the host boundary per operation. This is **not**
a fully resident graph or speed promise. MTP remains Unverified; explicit
draft-only synthetic parity never promotes it or qualifies the real Qwen model.

## Explicit physical checks

Tiny checks require a real device and fail if CUDA cannot initialize. They are
ignored in ordinary runs so check-only builds cannot masquerade as hardware runs.
Never execute artifacts built with `TRITIUM_CHECK_ONLY=1`.

```sh
timeout 600 env TRITIUM_CHECK_ONLY=0 CARGO_BUILD_JOBS=1 \
  cargo test --locked -p tritium-cuda --features cuda --test qtea \
  -- --ignored --test-threads=1 --nocapture
```

The cold-fit test additionally requires `TRITIUM_QTEA_CUDA_CASES_DIR` naming the
immutable eight-case admitted QTEA corpus from the independent CPU workflow. The
separate explicit library test `foreign_context_refuses_even_empty_calls` checks
context isolation without resetting or poisoning any device. Record source/run,
toolkit/driver/device identities and exact commands when preserving observations;
test success alone is not independent physical-byte, quality or release approval.
