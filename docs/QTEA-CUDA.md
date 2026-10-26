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

Full NN/embedding/serving dispatch and resident activation chaining are not wired
by this primitive. Existing QTEA model loading/serving remains CPU-only. GPU and
out-of-core fitting, large-model quality/MTP, speed, other devices and independent
release gates remain open. No experimental method is promoted by this API.

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
