# Conformance

A backend is "correct" in Tritium iff it reproduces the **frozen, versioned
conformance vector set** within tolerance. This is the mechanism that makes
cross-backend parity structural — see the
[Architecture](./architecture.md#the-frozen-vector-conformance-model) chapter for
the rationale and the [backend-breadth ADR](../../adr/0009-v070-backend-breadth.md)
for why freezing came first.

## The vector set

The committed set lives at `crates/tritium-testkit/vectors/v070.jsonl` (JSONL,
one `ConformanceVector` per line). It is surfaced by:

- `frozen_vectors() -> Vec<ConformanceVector>` — load the committed set.
- `frozen_vectors_path() -> PathBuf` — its absolute path, resolved from the
  testkit's `CARGO_MANIFEST_DIR` so any consuming crate's test process finds the
  one canonical file.
- `VECTOR_SET_VERSION` — the version tag (`"v070"`).
- `FROZEN_SEED` (`0xC0FFEE`) and `FROZEN_COUNT` (`64`) — the pinned seed and
  random-vector count the set was generated from; a fixed boundary set is appended
  unconditionally, so the file holds more than `FROZEN_COUNT` vectors.

Each `ConformanceVector` carries `m, n, k`, the `[M, K]` activations, the `[N, K]`
ternary weights (each in `{-1, 0, +1}`), the `[N]` per-output-channel scales, the
packing `format` (`"tq2_0"` / `"tq1_0"`), and the `[M, N]` `expected` output —
computed once from `tritium_core::reference_mpgemm`.

## Grading a backend

`run_conformance(&backend, &vectors, Tolerance::default())` runs the backend over
every vector and reports how many it reproduced within `Tolerance`. The default
`Tolerance` is `relative = 1e-4` (the fp32-accumulate matmul bar — fp32
accumulation reorders across backends, so bit-exactness is not required for the
float path); set `bit_exact = true` to grade the packing/integer paths with `==`.

`run_fused_fallback_contract(...)` is the companion gate for the fused W1.58A8
path on backends that cannot take a hardware-accelerated route — see
[capability fallback](./backends.md#capability-fallback).

## The drift gate

The set is deliberately **immutable**. The `frozen_set_matches_pinned_generator`
test makes any accidental drift — a changed generator, a changed reference kernel,
or a hand-edited file — a hard failure. To widen coverage (a new format, a
non-block `K`, more shapes) you re-freeze as a **new version**: regenerate via the
testkit's `freeze_vectors` example, commit a new `vectors/<ver>.jsonl`, and bump
`VECTOR_SET_VERSION` in a reviewed change. Bumping the version without committing
the matching file (or vice versa) fails the gate.

## Running it

```sh
cargo test -p tritium-cpu                       # CPU conformance, every push
cargo test -p tritium-cuda --features cuda      # CUDA, on a GPU box
cargo test --target wasm32-wasip1 -p tritium-wasm   # inside wasmtime
cargo test -p tritium-wgpu --features register      # on a Vulkan adapter
```

The CPU and wasm lanes run on every push; the CUDA and wgpu lanes are
hardware-gated (they self-skip or are manual where the hardware is absent).
