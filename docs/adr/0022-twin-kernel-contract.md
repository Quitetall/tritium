# ADR 0022 — Twin kernels: duplication is the contract (for now)

Status: **ACCEPTED** (2026-07-10). Revisit trigger defined below.

## Context

`decode.cu` holds 65 `__global__` kernels. Nine families exist in 2–4 dtype
variants — the ADR 0020 KV precision ladder multiplied the KV-touching
families by the rung count:

| family | variants |
|---|---|
| `rope_kv_fused` | `_g` f32, `_h` f16, `_q8` i8, `_t2` ternary-append |
| `kv_append`, `kv_append_batch` | 4 each (same axes) |
| `gqa_attention_{scores,reduce,batch,tree_scores,tree_reduce}` | 3 each (f32/f16/i8) |
| `lm_head_warp` | 2 (f32/f16 table) |

That is ~22 kernels that are "the same kernel at a different KV dtype". The
Phase-2 architecture question: consolidate via C++ templates with
`extern "C"` shims (one body, N instantiations), or keep the explicit
duplication?

### Measured evidence

The twins are **not** mechanical dtype swaps:

- `gqa_attention_scores_g` (82 lines) vs `_h` (78 lines): ~39 diff lines —
  roughly half the body. The f32 kernel reads K rows as `float4`; the f16
  kernel converts `__half2` pairs; the load width, the unroll shape and the
  tail handling all differ.
- The `_q8` variants are structurally different, not just retyped: the inner
  loop carries a **per-(token, head, group) scale stream** (the i8 rung's
  dynamic dequant), an extra global-memory operand the f32/f16 kernels do
  not have. A shared template body would be a *codec abstraction* (store/load
  functor + optional scale plumbing), not a `typename KV` substitution.
- ADR 0020 measured that scale-stream details dominate the i8 rung's
  performance (a shared-weight-bank variant was 386 vs 340 µs — WORSE);
  per-dtype tuning freedom in the inner loop is currently load-bearing.

### What the numerics contract demands

The f32 single-sequence path is the repo's only fully bit-exact end-to-end
domain (book: Conformance → "Numerics domains"). Every kernel is compiled
`--fmad=false` and gate-pinned `to_bits()`-equal against the host oracle. Any
consolidation must leave the f32 instantiation **provably byte-identical** —
in practice a SASS diff of the f32 instantiation against the current kernel,
plus the full gate suite, per toolchain bump. That proof is cheap once but
must be *re-established at every CUDA toolkit upgrade*, because template
instantiation only guarantees identical SASS while nvcc's inliner treats the
functor-abstracted body identically to the hand-written one — an assumption,
not a contract.

Note the host side is already deduplicated where dedup is free: the tree
graph capture reuses the batched graph builders (`gb_*`, P2a split made this
explicit — `record_graph_tree` calls into `batch.rs`). Device-side bodies are
the only duplication left, and they are exactly the part where the variants
genuinely differ.

## Decision

**Keep explicit per-dtype kernels. Duplication is the contract**, with three
codified guardrails:

1. **Twin-sync rule (review discipline).** A change to any member of a twin
   family must state in the commit message either the matching change to its
   twins or why the twins are exempt. The family table above is the
   checklist.
2. **Cross-dtype oracles stay mandatory.** Every rung ships behind its gate
   (f32: bit-exact; f16/i8: kernel-level equivalence + e2e perplexity bars
   per ADR 0020). A twin drifting semantically from its family fails a gate,
   not a code review.
3. **The codec seam stays host-side.** New KV-touching launch paths go
   through the `KvDtype` dispatch (cuda/kv.rs) so the *selection* logic never
   duplicates; only kernel bodies may.

## Revisit trigger

Consolidate into codec-templated bodies (store/load functor + scale-arena
plumbing, `extern "C"` shims preserving every current symbol) when **either**:

- a 4th KV rung ships (the ladder says ternary "KVTQ" may return with a real
  Hessian), or
- a new attention family lands (e.g. paged KV for batching phase 2),

because at that point the twin matrix grows multiplicatively (families ×
rungs) and the sync burden overtakes the template-proof burden. The
consolidation PR must include: SASS diff of the f32 instantiation vs the
retired hand-written kernel (byte-identical or justified), the full gate
suite green, and a CI step that re-diffs SASS on toolchain bumps.

## Consequences

- No behavior or performance change now; per-dtype inner-loop tuning freedom
  is preserved (the i8 rung depends on it).
- The cost is process, not code: the twin-sync rule adds reviewer burden per
  KV-touching change (~22 kernels across 9 families).
- The decision is explicitly reversible and the reversal condition is
  mechanical, not a judgment call.
