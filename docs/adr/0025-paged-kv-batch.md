# ADR 0025 — Paged KV for the M=N batch (batching P2, C3)

Status: **PROPOSED** (2026-07-11). C1 (chunked prefill) and C2 (per-row
masks / dead-rows-touch-nothing) are shipped and gated; this is the third and
last planned P2 step.

## Context

The batch KV arenas are dense `[n, max_ctx, kv_width]` per layer: every slot
pre-pays `max_ctx` tokens of VRAM whether its request uses 60 tokens or 4000.
At n=8, max_ctx=4096, kv_width=640, 30 layers, f32 that is ~2.5 GB of which a
typical chat workload touches a few percent. Slots-per-GB — the batching
capacity number — is bounded by the worst case, not the actual load.

C2 established the contract paging needs: a dead row owns no write slot and
touches no arena bytes (`set_live`, device position `-1`, gated by
`cuda_batch_dead_row_touches_nothing`).

## Decision (proposed)

1. **Page pool + per-slot page table.** One pool per layer pair (K and V):
   `[pool_pages, PAGE_TOKENS, kv_width]` f32, `PAGE_TOKENS = 256` (power of
   two: in-kernel index math is shift/mask; 256 tokens × 640 f32 = 640 KB per
   K or V page per layer at BitNet geometry). A per-slot page table
   `d_page_table: [n, max_pages_per_slot] i32` maps logical token `j` to
   physical row `page_table[row][j >> 8] * 256 + (j & 255)`. `-1` entries are
   unmapped (touching one is a bug, not a fallback).
2. **Codec-templated kernels (the Track B pattern).** The two KV-indexing
   kernels — `kv_append_mdecode` (write) and `gqa_attention_split_partial`
   (read) — become one body each with a mapping codec:
   `MapDense::kv_row(base, row, j)` = `base + (row·max_ctx + j)·kv_width`
   (today's arithmetic, pointer-shape-preserved); `MapPaged::kv_row` does the
   table lookup. **Proof obligation: the dense instantiation is SASS
   byte-identical to the retired hand-written kernel** (tools/sass_diff.sh,
   the ADR 0022 procedure); the paged instantiation is the new member.
   `combine`, `rope`, and everything else never touch KV — unchanged.
3. **Bit-exactness by construction.** Paging changes ADDRESSES, never values
   or reduction order: for the same logical contents, paged attention output
   is bit-identical to dense. Gate: a paged batch and a dense batch fed the
   same requests must produce bit-equal logits per row per step (the
   dead-row-gate pattern), plus G1/G2 and the 5 batch acceptance gates
   unchanged on the dense default.
4. **Allocation policy (v1): reservation, no eviction.** Host-side free list
   (`Vec<i32>`); admission reserves `ceil((prompt_len + max_new) / 256)`
   pages per K and V; if the pool can't cover it the job waits in queue (the
   C1 admission machine already serializes admissions). Retirement returns
   pages. Nothing can OOM mid-decode and there is no preemption machinery.
   Memory becomes `sum(per-request footprints)` instead of `n × max_ctx` —
   the win scales with how far typical requests sit below max_ctx.
   vLLM-style oversubscription + preemption is an explicitly separate,
   later decision.
5. **Graph compatibility.** The pool and page-table buffers are allocated at
   `new_batch` (stable pointers, captured once); page-table CONTENT is
   per-step data uploaded next to `d_positions`. Allocation changes flow
   through uploads — no recapture, same mechanism C2 uses for liveness.
6. **Adoption.** `copy_kv_into_batch_row` becomes a per-page loop of dtod
   copies (rare path, host-driven, no new kernel). Debug readers translate
   through the host copy of the page table.
7. **Scope guards.** Dense stays the default (`--kv-pages <pool_tokens>` or
   equivalent opts in); tree/spec and the single-sequence path are untouched;
   batch arenas stay f32 (the KV-rung question is orthogonal and still
   rejected loudly).

## Verification plan

- SASS byte-identity for the dense instantiations (ADR 0022 procedure,
  archived diffs).
- New gate: paged == dense bit-equal logits, same requests, several steps,
  including a retire + re-admit cycle (page reuse) and a dead row (C2
  contract composes: dead rows own zero pages).
- Serve: G1/G2 + C1 interleave on a paged pool; slots-per-GB before/after
  recorded in OPTIMIZATION-LOG with the honest caveat that the win depends
  on the workload's length distribution.
- Allocator unit tests (host-only): reserve/release/reuse, exhaustion
  rejects, no double-grant.

## Consequences

- KV capacity stops scaling with max_ctx pessimism; C4 (batched spec-decode
  coexistence) gets its natural substrate (a tree session = its own paged
  region).
- One more codec axis on two kernels, paid for by the Track B consolidation;
  the dense path stays provably untouched.
- The reservation policy leaves oversubscription upside on the table — the
  recorded trade for zero eviction complexity in v1.
