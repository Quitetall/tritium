# Conformance

A backend is "correct" in Tritium iff it reproduces the **frozen, versioned
conformance vector set** within tolerance. This is the mechanism that makes
cross-backend parity structural — see the
[Architecture](./architecture.md#the-frozen-vector-conformance-model) chapter for
the rationale and the backend-breadth ADR (see the [research repository](https://github.com/Quitetall/tritium-research))
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

## Numerics domains

Tritium's end-to-end gates assert three DIFFERENT strengths of "matches", and
every gate's doc comment says which one it pins. When a doc claim and an
assertion disagree, the assertion is the contract — fix the doc.

| domain | contract | why not stronger | gated by |
|---|---|---|---|
| **Single-sequence decode** | **bit-exact within a launch mechanism**: graph replay reproduces its own capture `to_bits()`-equal; kernel-level host↔device gates assert `to_bits()`; greedy token IDs equal the committed `transformers` reference over the full horizon. ACROSS mechanisms the gates are: CPU↔CUDA = greedy-token exact + ≤2e-3 relative logits; eager `step` vs graph = greedy-token exact + top-16 logits ≤2e-3 (the graph reads the f16 LM-head table, a designed difference) | kernels compiled `--fmad=false` with one reduction order (ADR 0018) make the per-kernel results bit-exact; the fp32 residual stream across 30 layers and the f16 graph head bound the cross-mechanism claims | `acceptance.rs` greedy/parity/graph gates, `cuda/tests.rs` per-kernel `to_bits` gates |
| **M=N batch decode** (vs the M=1 path) | **token parity**: greedy decisions match single-sequence over a short lockstep horizon (`cuda_batch_decode_matches_single`, 4 steps asserted); at the serve level, first token exact + early-divergence guard (`batch_serve.rs` G1 asserts token 0 and agreement ≥ 2; the longer prefix is reported, not asserted). WITHIN a batch, sequences are bit-exact independent, graph==eager is bit-exact, and the same request set reproduces exactly across pools/slot orders (G2) | split-KV attention reorders the f32 sum vs the M=1 warp kernel — logit bits differ at ulp level, and near-tie argmaxes can flip at long horizons | `acceptance.rs::cuda_batch_decode_matches_single`, `batch_serve.rs` G1+G2, kernel-level 1e-4 equivalence gate |
| **KV-cache rungs below f32** (f16 / i8; ADR 0020) | **perplexity-gated**: e2e perplexity relative error inside the quality bar (~0.16% f16, ~0.26% i8); no token or logit claim | every written K/V rounds once by design; opt-in per model instance, f32 default stays in the bit-exact domain | perplexity gates + long-context quality probe (ADR 0020) |

Two corollaries worth internalizing:

- A batched server (`--batch-slots N`) is its own reproducibility domain: the
  same request re-run through the SAME domain reproduces exactly (slot
  assignment and admission order don't matter — gated), but it is not
  bit-comparable to the single-request server beyond the greedy-token horizon.
- Combining domains multiplies caveats: batch + f16 KV inherits BOTH the token
  parity bound and the perplexity bound. The f32 single-sequence path is the
  only fully bit-exact end-to-end configuration, which is why it is the
  reference every other domain is measured against.
