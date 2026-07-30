# 0046 — PyTorch dispatcher and zero-copy CPU/CUDA execution

Status: **IN PROGRESS** (2026-07-20)

- **Decision:** [ADR 0033](../adr/0033-v11-full-public-release.md)
- **Parent:** [plan 0044](./0044-v11-full-public-release.md)
- **Dependency:** [plan 0045](./0045-torch-reference-conversion.md) — done

## Goal

Replace Python-list interop on the new ternary-linear path with a first-class
PyTorch dispatcher operator. Success means CPU/CUDA tensors remain resident,
autograd/fake/meta/autocast/`torch.compile` work, and the optimized adapter can
later replace the composite device kernel without changing public behavior.

## Public seams under test

- `tritium.torch.ternary_linear(input, master, bias=None)`
- `tritium.nn.TernaryLinear.forward`

Reference estimator and conversion behavior from 0045 remains the oracle.

## Step 1 — Dispatcher-visible device-resident composite operator

- Register `tritium::ternary_linear` with `torch.library.custom_op` for CPU and
  CUDA.
- Forward performs hard per-row AbsMean projection and linear algebra using
  tensors on the input device/current stream. No list, NumPy, host buffer or
  explicit synchronization is allowed.
- Register custom backward matching the strict Rust STE mask for activation,
  latent master and optional bias.
- Register fake/meta and CPU/CUDA autocast behavior.
- Route built-in `AbsMeanSTE`/single-plane `SaltSTE` modules through the op;
  external estimators retain the validated composite reference adapter.

Gate:

```bash
PYTHONPATH=crates/tritium-py/python pytest -q \
  crates/tritium-py/tests/test_torch_dispatch.py
```

Required tests: literal forward/backward parity, `torch.library.opcheck`,
`torch.compile(fullgraph=True)`, arbitrary leading dimensions, optional bias,
CPU autocast and CUDA residency/autocast when available.

## Step 2 — Native fused adapter

- Add a stream-aware native CPU/CUDA implementation behind the same dispatcher
  schema. DLPack/capsule ownership must be single-owner and lifetime-safe.
- Cache packed ternary weights by parameter version; invalidate before the next
  forward after optimizer mutation.
- Keep composite operator as exact conformance fallback and custom-estimator
  path.

Gate: profiler records zero H2D/D2H copies or global synchronizations after
setup; direct-adapter forward/backward parity passes; wrapper overhead is within
5% of direct Tritium execution at frozen representative Linear shapes.

CPU vertical slice landed:

- compact CPU `float32` inputs and detached masters enter Rust through
  single-consumer DLPack capsules without list, NumPy, or tensor copies;
- TQ2_0 weights and per-row scales are retained in a bounded 4096-entry weak-owner cache
  keyed by parameter identity, mutation version, storage identity, data pointer,
  byte offset, and shape; both ordinary optimizer mutation and storage replacement
  invalidate before the next forward;
- output storage is Rust-owned and transferred to PyTorch through DLPack;
- first-order CPU backward reuses the same packed-cache entry, computes
  activation/projected-weight/bias VJPs in Rust, applies the strict STE mask
  natively through the backend-neutral `mpgemm_projected_vjp` port, and returns
  three Rust-owned gradients through DLPack;
- non-compact upstream gradients are compacted on-device before the native call;
  higher-order-gradient recording retains the composite fallback;
- unsupported dtype/layout/device cases retain the exact composite fallback;
- no persistent dense weight shadow or transposed packed copy is cached; current
  CPU backend unpack scratch remains transient per call.

Native CPU backward gate: dispatcher suite passes 18 tests; 40 frozen-seed spot
checks bound observed forward/gradient drift to `4.77e-6`; full wheel suite
passes 241 tests plus 9 subtests (one prerequisite skip). Warm backward profiles
contain no Torch projection or matrix-multiply operators.

Remaining before Step 2 closes: stream-aware CUDA input/output, CUDA packed-cache
residency, sanitizer evidence, CPU performance optimization/qualification, and
retained representative-shape performance receipts. Exploratory release-build
timings are not a gate receipt: native forward+backward remained 2.6–3.7× slower
than this host's MKL-backed composite at `M=32`, so no CPU speedup claim is
permitted yet.

## Verification

```bash
PYTHONPATH=crates/tritium-py/python pytest -q crates/tritium-py/tests
python -m compileall -q crates/tritium-py/python/tritium
cargo fmt --check
git diff --check
```

Skipped CUDA/opcheck/compile tests are failures when their prerequisites are
present.

## Review

After each commit, call lamu `review_commit` with this plan as `plan_file` and
verify every finding before applying it.

## Commit

First slice:

```text
feat(torch): register device-resident ternary linear op
```

## Done criterion

Both dispatcher-visible composite and native fused adapters pass their gates.
The public op schema is unchanged between them; plan 0047 may consume it.
