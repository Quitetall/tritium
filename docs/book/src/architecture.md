# Architecture

Tritium is a **hexagonal / ports-and-adapters** workspace. Dependencies point
**inward** only:

```text
foundation  (core · spec · format · testkit)
   ↑
backends    (cpu · cuda · wgpu · wasm · …)   — each implements tritium-spec
   ↑
runtime     (runtime · quantize · nn · train)
   ↑
frontends   (cli · serve · ffi · py · candle · burn · onnx)
```

Two rules make cross-backend correctness *structural* rather than a matter of
discipline:

1. **A frontend never depends on a concrete backend.** A CLI or server talks to
   the `TernaryBackend` trait and the runtime registry, never to `tritium-cuda`
   directly.
2. **A backend never depends on another backend.** Each backend independently
   implements the same trait and is graded against the same reference vectors.

## The backend contract: `TernaryBackend`

The contract lives in `tritium-spec` as the **object-safe** trait
`TernaryBackend`. Object safety is deliberate: the runtime holds a
`Box<dyn TernaryBackend>` registry of heterogeneous devices. That rules out an
associated `Buffer` type (which would make `dyn TernaryBackend` impossible), so
device memory is carried as a boxed `DeviceBuffer` trait object and each backend
downcasts it to its concrete buffer via `core::any::Any`.

The trait's load-bearing methods:

- `device_id(&self) -> &str` — a stable identifier such as `"cpu"` or `"cuda:0"`.
- `capabilities(&self) -> DeviceCaps` — what the device can do (see below).
- `upload_weights(packed, shape, format) -> Box<dyn DeviceBuffer>` — upload
  host-side **already-packed** weight bytes (`TQ1_0`/`TQ2_0`) and get back an
  opaque handle reused across calls. Packing is *not* part of the trait — it
  lives host-side in `tritium-format`, the single source of truth.
- `mpgemm(act, weights, scales, shape, format, out)` — the plain ternary
  matmul: `out[m,n] = scale[n] · Σ_k act[m,k] · w[n,k]`.
- the fused **W1.58A8** path — quantize `act` to per-token int8 (absmax,
  `Qp = 127`), run the ternary contraction, and fold both the per-token
  activation scale and the per-channel weight scale into the `f32` output. This
  is the BitNet linear-layer primitive.

Every implementation must match `tritium_core::reference_mpgemm`. For the
floating-point path the bar is **relative error ≤ 1e-4** (fp32 accumulation
reorders across backends, so bit-exactness is not required there); the
packing/integer paths are graded **bit-exact**.

## `DeviceCaps`

`DeviceCaps` is how the runtime decides which backend can run a problem. It is
`#[non_exhaustive]` so later milestones can add fields without a breaking change.
Today it carries:

- `backend` — family, e.g. `"cpu"` or `"cuda"`.
- `device_name` — human-readable, e.g. `"x86_64 (avx2)"` or `"NVIDIA RTX 4090"`.
- `features` — detected ISA flags, e.g. `["avx2"]` or `["sm_89"]`.
- `total_memory_bytes` — `0` if unknown.
- `supports_imma` — int8 tensor-core (IMMA) path available.
- `supports_fp8` — fp8 path available.

A backend whose hardware lacks a capability (no fp8, no IMMA) must still produce
a correct result by falling back — see [Backends](./backends.md).

## The registry: `linkme` self-registration

Backends **self-register** with `tritium-runtime` through the `BACKENDS`
`linkme::distributed_slice`. A backend crate places one `BackendEntry` into that
slice; the linker gathers every entry, and `Registry::init()` walks them and
constructs each backend. The consequence: *linking* a backend crate into a
binary makes its device appear in the registry with **no central edit** — that
is exactly what the `tritium` CLI's `list-backends` reports. (`wasm32` is the one
target where `linkme` is unavailable, so `tritium-wasm` is constructed
explicitly rather than self-registered.)

## The frozen-vector conformance model

From v0.70 on, "correct" for a backend means: reproduces a **committed,
immutable** set of conformance vectors — not a set regenerated from a seed at
test time. The set is `tritium-testkit`'s `vectors/v070.jsonl`, surfaced by
`frozen_vectors()`; `VECTOR_SET_VERSION` names the version (`"v070"`).

Each `ConformanceVector` is a self-contained mpGEMM case — `m, n, k`,
activations, ternary weights, per-channel scales, the packing `format`, and the
`expected` output computed once from `tritium_core::reference_mpgemm`. A backend
passes a vector iff its output is within `Tolerance` of `expected`. The default
`Tolerance` is `relative = 1e-4`, not bit-exact (the fp32-accumulate matmul bar
from the [release-roadmap ADR](../../adr/0002-release-roadmap.md)); packing
paths set `bit_exact = true`.

Freezing matters because backend breadth (`tritium-wgpu`, `tritium-wasm`, and the
planned `tritium-metal` / `tritium-rocm`), every interop "matches the native
reference" gate, and every release re-run all grade against this one artifact —
so the reference must not drift underneath them. A drift gate
(`frozen_set_matches_pinned_generator`) turns any accidental change to the
generator, the reference kernel, or the committed file into a hard test failure.
Widening coverage is a deliberate **re-freeze**: regenerate via the
`freeze_vectors` example, commit a new `vectors/<ver>.jsonl`, and bump
`VECTOR_SET_VERSION`.

See [Conformance](./conformance.md) for how a backend author wires the harness
in, and the [backend-breadth ADR](../../adr/0009-v070-backend-breadth.md) for the
rationale.
