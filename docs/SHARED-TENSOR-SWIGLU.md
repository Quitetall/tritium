# Shared-tensor device SwiGLU (experimental)

`CudaBackend::with_device_swiglu()` opts compatible typed-tensor MLP regions into
device-f32 SiLU. Default/unsupported regions still decline without changing output.
The selected numerical mode is unchanged: device expf is not host exp bit parity.
This is not full-graph residency, measured speed, model quality or release approval.

The `TensorSwiGlu` / `try_tensor_swiglu` success domain is finite input and finite
computed intermediates/output (proposed ADR0175). Enabled CUDA checks input before
allocating/uploading execution scratch. A private device status word records bad
gate/up/activated results and basis-transformed inputs, so sparse/pruned matrices
cannot hide an invalid intermediate behind finite final output. Status and final
readback are checked before caller output is copied. Refusal is an error, never
silent fallback; caller output is unchanged. Workspace can be dirty after numeric
refusal, but status is reset for each nonempty call and valid retries work.
Driver/context failures remain errors, without a promised recovery or device reset.
Backend scratch diagnostics show geometry/capacities, not private activation arrays.

This deliberately changes invalid-domain behavior: NaN/Inf or finite overflow may
previously have been reported as successful output; selected execution now refuses
them. Valid finite arithmetic, basis order, existing fixture bounds, backend mode
identity and opt-in/default selection are unchanged. Callers using nonfinite values
as probe data must handle the error. Ordinary host fallback and unrelated backend
operations are not certified as finite-safe by this change.

The existing geometry-keyed private workspace gains one four-byte device flag and
a four-byte host status read. It remains outside weight-only upload budgets. It is
not full workspace/model memory admission or a physical VRAM/RSS bound; allocator,
driver, model/cache and other calls remain separate. No new weight copy, global
toggle, source-authority claim or qualification receipt is introduced.

Explicit [workspace resources](SWIGLU-WORKSPACE.md) can separately admit scratch
and coordinate it with other logical domains. Legacy execution stays outside
weight-only admission unless those resources are selected.

Tiny explicit physical regressions are ignored in ordinary runs and never silently
skip without a device. The full physical target also retains its previous dense,
additive, rotated, compact, mixed, empty and repeated-call tests with unchanged
numeric bounds:

```sh
timeout 600 env TRITIUM_CHECK_ONLY=0 CARGO_BUILD_JOBS=1 \
  cargo test --locked -p tritium-nn --features cuda --test swiglu_cuda \
  -- --include-ignored --test-threads=1 --nocapture
```

Do not execute images built with `TRITIUM_CHECK_ONLY=1`. Preserve exact source,
model/data fixtures, compiler/device and run identities for observations. These
developer tests do not clear independent quality/performance/physical/security or
public activation gates. Packed QTEA uses its separate bounded region described
in [QTEA-CUDA](QTEA-CUDA.md); that region's arithmetic/profile remains unchanged.
