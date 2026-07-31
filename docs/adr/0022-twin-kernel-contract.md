# ADR 0022 — Twin kernels: duplication is the contract (for now)

Status: **ACCEPTED** (2026-07-10); **REVISIT TRIGGER PULLED + CONSOLIDATION
EXECUTED for the f32/f16 axis** (2026-07-11, Track B — user-directed, ahead
of paged KV which will add a new family). See "Executed consolidation" below.
The q8/t2 axis REMAINS duplicated per the original evidence (structurally
different scale-stream contractions, tuning-sensitive).

## Context

`decode.cu` holds 71 `__global__` kernels (65 at this ADR's writing; −1
`gqa_attention_mdecode_f32` retired + +2 paged-KV twins, ADR 0025; +2
`gqa_attention_batch_v2` f32/f16 twins, 2026-07-17 — an order-preserving
prefill-attention rewrite, bit-identical to rev 1 by `to_bits` gate; +2
`gqa_attention_batch_v3` Q-blocked twins, 2026-07-18 — same bit-identity
gate, BQ rows amortize the staged K/V; +1 `draft_chain_advance` 2026-07-30 —
the ADR 0032 L1' chained-draft glue, single-instance (no dtype twins) — the
drift test pins the count with the cause chain). Eleven families exist in
2–4 dtype variants — the ADR 0020 KV precision ladder multiplied the KV-touching
families by the rung count:

| family | variants |
|---|---|
| `rope_kv_fused` | `_g` f32, `_h` f16, `_q8` i8, `_t2` ternary-append |
| `kv_append`, `kv_append_batch` | 4 each (same axes) |
| `gqa_attention_{scores,reduce,batch,tree_scores,tree_reduce}` | 3 each (f32/f16/i8) |
| `gqa_attention_batch_v2` | 2 (f32/f16; i8/t2 fall back to rev 1) |
| `gqa_attention_batch_v3` | 2 (f32/f16; i8/t2 fall back to rev 1) |
| `lm_head_warp` | 2 (f32/f16 table) |

That is 33 kernel symbols across these families — 22 duplicates beyond one
canonical body each. The
Phase-2 architecture question: consolidate via C++ templates with
`extern "C"` shims (one body, N instantiations), or keep the explicit
duplication?

### Measured evidence

The twins are **not** mechanical dtype swaps:

- `gqa_attention_scores_g` (82 lines) vs `_h` (78 lines): 28 of 82 source
  lines differ (~34%). The f32 kernel reads K rows as `float4`; the f16
  kernel converts `__half2` pairs; the load width and the tail-path types
  differ (the unroll pragma is shared).
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
domain (book: Conformance → "Numerics domains"). The decode and training
kernels (`decode.cu`, `train_grad.cu`) are compiled `--fmad=false` and
gate-pinned `to_bits()`-equal against the host oracle (the add/IMMA mpgemm
kernels accumulate in integers, where fmad is moot). Any
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

**Keep explicit per-dtype kernels. Duplication is the contract**, with four
codified guardrails:

1. **Twin-sync rule (review discipline).** A change to any member of a twin
   family must state in the commit message either the matching change to its
   twins or why the twins are exempt. The family table above is the
   checklist.
2. **Cross-dtype oracles stay mandatory.** Every rung ships behind its gate
   (f32: bit-exact; f16/i8: kernel-level equivalence + e2e perplexity bars
   per ADR 0020). A twin drifting semantically from its family fails a gate,
   not a code review.
3. **The codec seam stays host-side.** New KV-touching launch paths select
   symbols through `KvDtype::pick` (cuda/kv.rs) — the one dispatch point —
   so the *selection* logic never duplicates; only kernel bodies may.
4. **Table drift has teeth.** `adr_0022_twin_family_table_matches_decode_cu`
   (cuda/tests.rs, no GPU needed) pins this ADR's family table and the total
   kernel count against `decode.cu`; a new variant or family fails the suite
   until both the ADR and the revisit-trigger assessment are updated.

Accepted debt, recorded: the shared host-side builders live in two places by
two conventions — `gb_*` graph builders are `pub(super)` in batch.rs (tree.rs
depends on a sibling's internals), while the `bl_*` raw-launch helpers sit in
mod.rs (used by prefill, tree and batch alike). Coherent enough to ship;
co-locating them in one builders module is a candidate cleanup if the seam
grows.

## Executed consolidation (2026-07-11)

The f32↔f16 axis is now codec-templated in-tree, under this ADR's proof
obligations:

- **Codecs**: `KvStoreF32/F16` (store axis: kv_append, kv_append_batch,
  rope_kv_fused) and `KvLoadF32/F16` (load axis: the five gqa_attention
  families + lm_head_warp). The load codec is **Row-typed** — it hands the
  body the SAME pointer shape the hand-written kernels used (float4* indexed
  per d for f32; __half* + kvh_load4 for f16), which is what makes the f32
  instantiations compile byte-identically (a plain `T*`+cast-per-load shape
  produced 1.4k lines of scheduling drift before this was found).
- **Templates live outside the file's `extern "C"` block** (C linkage cannot
  template); shims inside it preserve every launch symbol — host code and
  the drift test are untouched. Dynamic-shared arrays inside template bodies
  need unique names (C++ vs C linkage of `extern __shared__` collide).
- **Proof status (tools/sass_diff.sh, sm_89, CUDA 13; counts as of Track B)**: 63/65 kernels SASS
  **byte-identical** after the refactor, including all twelve templated
  attention/lm_head instantiations. The two exceptions — kv_append_batch
  f32/h — are **justified**: identical 152-instruction opcode streams,
  register-allocation permutation only (13 sites), plus the batch
  bit-exactness gates.
- **Still duplicated, with cause**: every `_q8`/`_t2` variant (row-granular
  grouped-scale quant/dequant — an element codec cannot express them; the
  appends already delegate to shared `kv_quant_row_*` helpers), the mdecode
  family (f32-only today; its direct-attention member
  `gqa_attention_mdecode_f32` was RETIRED by ADR 0025 step 2 — the split
  partial+combine pair is the only batch attention), and
  `gqa_attention_decode_warp_g` (legacy geometry fallback, f32-only).

### Re-proof procedure (toolchain bumps)

Byte-identity is toolchain-relative, so a pinned SASS hash would false-alarm
on every CUDA update. Instead, on a toolchain bump re-run the comparison
UNDER THE NEW TOOLCHAIN:

```sh
git show <pre-consolidation>:crates/tritium-cuda/kernels/decode.cu > /tmp/old.cu
nvcc -arch=sm_89 -ptx --fmad=false /tmp/old.cu -o /tmp/old.ptx
nvcc -arch=sm_89 -ptx --fmad=false crates/tritium-cuda/kernels/decode.cu -o /tmp/new.ptx
tools/sass_diff.sh /tmp/old.ptx /tmp/before && tools/sass_diff.sh /tmp/new.ptx /tmp/after
diff -rq /tmp/before /tmp/after   # expect: only the two justified kernels
                                  # (plus the all.sass/all.cubin container
                                  # files, which always differ — not kernels)
```

`<pre-consolidation>` = the parent of the first Track B commit (4f3c566^).
Run alongside the GPU lane's gate suite; a NEW divergence means the new
compiler treats the template differently — re-justify or fix before shipping.

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
  KV-touching change (29 kernel symbols across 9 families).
- The decision is explicitly reversible and the reversal condition is
  mechanical, not a judgment call.
