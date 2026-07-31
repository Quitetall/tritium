# ADR 0032 — Spec-decode cost model: why it sits at parity, and the ranked levers past it

Status: **PROPOSED** (2026-07-19)

- **Deciders:** Brian Lam
- **Relates:** closes the loop on [ADR 0021](./0021-drafter-architecture.md)
  (drafter architecture) with measured e2e; consumes the prefill wins of
  [ADR 0026](./0026-sota-campaign.md) Track P (rev-4 IMMA + v2/v3 attention);
  sequences after the shipped drafter arc (stages 1–3, adaptive draft-k,
  device-argmax drafter steps). The batch-slot item touches
  [ADR 0025](./0025-paged-kv-batch.md).

## Context: the drafter works, but spec decode is stuck at parity

Three stages of drafter training plus two engine changes are shipped and
gated (all bit-identical, 12/12 lossless on prose):

| lever | result |
|---|---|
| drafter stage 1 (4.9M teacher tokens) | τ = 3.76 tok/verify |
| stage 2 (same data, more epochs) | τ = 3.84 — **flat** (generalization-bound) |
| stage 3 (15.3M diverse teacher, from scratch) | τ = 4.23 |
| adaptive draft-k (`DraftPolicy` EWMA) | e2e 0.891× → 0.993× of plain |
| device-argmax drafter steps (4-byte readback) | e2e ~0.97× — **flat** |

τ climbs sub-linearly with teacher data (+12% τ per 3× tokens), and the two
engine changes moved e2e from a clear loss to parity — but **not past it**.
A ternary 2B4T verifier + a 30M-param ternary drafter that is greedy-lossless
by construction *should* be a 1.5–2× decode win at τ = 4.2. It is not. This
ADR records why, from a profile of the live spec loop, and ranks what closes
the gap — so the next session starts from the cost model, not a fresh guess.

## The measurement: a profiled spec loop is LM-head-bound

`nsys` over `tritium-serve --draft-model` (BitNet 2B4T target + stage-3
drafter, 6 prose requests, f32 KV, RTX 4090), GPU-kernel time share:

| kernel | share | what it is |
|---|---:|---|
| `lm_head_tiled_f16` | **44.5%** | the **verifier's** tree LM head — one full 128256-vocab projection per verify |
| `tq2_0_imma_mpgemm` | 20.2% | ternary GEMMs (both runners, all layers) |
| `lm_head_warp_f16` | **16.2%** | the **drafter's** per-token full-vocab head (k projections per cycle) |
| `gqa_attention_batch_v3_f32` | 7.6% | attention (both runners) |
| `argmax_rows_partial_f32` | 3.0% | the device argmax this session added |
| rmsnorm / act_quant / rope / residual | ~7% | elementwise |

**The two LM heads are 60.7% of spec-loop GPU time.** Both are full
128256-row projections, and both are **memory-bandwidth-bound on the
embedding-table read**, not launch- or compute-bound:

- verifier head: 128256 × 2560 f16 ≈ 657 MB/projection; at ~800 GB/s ≈ 0.82 ms,
  matching the observed 0.73 ms median.
- drafter head: 128256 × 768 f16 ≈ 197 MB/projection (3.3× smaller table — the
  drafter shares the target *tokenizer* but has its own narrow embedding), so
  its per-token head is ~⅓ the verifier's; k of them per cycle.

This is why the device-argmax change (ADR-referenced commit) was flat: it
removed the 513 KB logits **readback** per drafted token, but the readback was
never the cost — the full-vocab table **read** is, and argmax still reads every
row. Correct change, wrong bottleneck; recorded as a no-win.

## The cost model (why τ = 4.2 is not enough here)

Per spec cycle: one verify (fixed cost `V`) commits `τ` tokens, but first pays
`k` drafter steps (cost `d` each). Plain decode does one step (cost `P`) per
token. Spec wins per token iff

```
(V + k·d) / τ  <  P
```

Measured (warm, this box): `V ≈ 10 ms`, `P ≈ 3.9 ms`, `d ≈ 1.5–2 ms`,
`τ ≈ 4.2`, and adaptive-k drives `k ≈ τ` (≈ 5–6 on strong prose). Plug in:

```
(10 + 5·1.7) / 4.2  =  18.5 / 4.2  =  4.4 ms/token   vs   3.9 ms/token plain
```

So spec is ~0.89× on committed tokens — and the **`k·d` drafter term is the
killer**, not the verify. The verify alone (`V/τ = 2.4 ms/token`) already
beats plain; the `k` eager drafter steps, each reading the drafter's 197 MB
table, erase the win and then some. Adaptive-k and device-argmax shaved this
to ~parity; the levers below attack `d`, `V`, and the amortization of both.

## Decision: the ranked levers (each falsifiable, measure-first)

### L1 — Ternary drafter LM head — **REFUTED BY MEASUREMENT (2026-07-30)**

Tested end-to-end: untied ternary-head 8L/768 trained from scratch on the
stage-3 mix (same data/epochs as the tied τ=4.23 run; final loss 2.34 vs
1.81 tied), exported with `output.weight` I2_S, served via the loader's new
untied path, measured on the standard prose set:

- **τ = 3.23 vs 4.23 tied — a −24% acceptance hit** (12/12 lossless, as
  structurally guaranteed). Far past this ADR's "<10% still wins" line.
- The `d` saving is smaller than projected: the head's table read (~0.23 ms
  warm) is only ~15% of the ~1.7 ms drafter step — the GPU-share numbers
  above are shares of GPU-busy time, but the step's WALL time is dominated
  by per-token host orchestration (two stream syncs + ctrl H2D + readback
  ≈ 1.2 ms), which a ternary head does not touch.
- Cost model, both idealized on-device: f16 head (10 + 4.2·0.45)/4.23 =
  2.81 ms/tok (1.39×) vs ternary head (10 + 3.2·0.22)/3.23 = 3.31 ms/tok
  (1.18×). **The f16 head wins in every regime — τ dominates.**

Consequence: do NOT build the resident ternary-head GEMV (the premise-first
sequencing saved that effort). The loader half stays (untied ternary heads
now load and serve correctly — useful substrate, zero regression). The wall
decomposition above REDIRECTS the drafter lever: the true dominant `d` term
is the per-token host round-trip, so the "graph-capturing the k-step draft"
rejection below is retracted — a CHAINED k-step draft (k tokens in one
device-side loop/graph, argmax fed back to embed on-device, one host
round-trip per k instead of per token) attacks ~1.2 ms × k directly and is
the new L1'.

**L1' BUILT + MEASURED (2026-07-30): the first real spec-decode speedup.**
`draft_chain` replays the captured M=1 graph k times back-to-back with a
single-thread `draft_chain_advance` kernel feeding each device argmax into
the next replay's control block on-device — one host round-trip per chain.
EOS semantics match the host loop exactly (drafted, never fed; post-halt
replays are idempotent rewrites). Gated: `cuda_draft_chain_matches_per_step`
(token-exact vs the per-step ladder at k=1/4/7/16, back-to-back chains, EOS
truncation + watermark consistency). Same-session three-way interleaved A/B
(stage-3 f16-head drafter, 12×256 greedy prose, 12/12 output-identical):

| config | vs plain decode |
|---|---:|
| per-step ladder (`TRITIUM_DRAFT_CHAIN=0`) | 0.913× |
| **chained draft (default)** | **1.144×** |
| chain vs per-step | **1.253×** |

Spec decode beats plain decode for the first time in the campaign. The
remaining gap to the cost model's ~1.34× warm-clock projection is the
verify cycle's own host overhead (V ≈ 10-20 ms wall vs ~2 ms GPU) — L2's
territory, now the top open lever alongside L3 batch-slots.

### ~~L1 (original projection, kept for the record)~~ — attacks `d`, the dominant term

The drafter's per-token head reads a 197 MB **f16** table. Make it **ternary
I2_S** (the substrate this whole engine is built on) and the read drops ~8× to
~25 MB → the drafter step becomes near-free → the `k·d` term collapses. This is
the "ternary has more to go" lever in its purest form: the drafter proposes,
so its head precision only affects *acceptance* (τ), never correctness (the
verifier is the source of truth — losslessness is untouched).

- **Risk:** a ternary head shares weights with the tied ternary *input*
  embedding, which may cost τ. Mitigation: **untie** the drafter head from its
  embedding and ternarize only the head (the head is ~⅓ of the drafter's
  params; untying + ternary head is a small net add). Or accept the tied
  ternary head and measure the τ hit — if τ drops < ~10% it still wins on the
  `d` collapse.
- **Where:** a drafter-only export flag in `export_i2s_gguf.py` (emit
  `output.weight` as I2_S, untied) + the runner already serves ternary heads
  (the target's is f16 only by choice, not limitation — the resident decoder's
  `lm_head` path is dense; a ternary-head variant reuses the decode GEMV).
- **Gate:** greedy losslessness unchanged (structural); τ delta measured;
  e2e ABBA vs the f16-head drafter. Win condition: e2e ≥ 1.3×.

### L2 — Fused head→argmax/accept for the verifier (attacks `V`)

The verifier materializes `m × 128256` logits per tree, but the greedy accept
walk needs only two scalars per node: the argmax id, and the drafted token's
logit (for the accept comparison; the bonus draw needs the argmax row only at
the final accepted node). A head kernel that keeps a running (max_val, max_idx)
over vocab tiles and probes the drafted-token column — never writing the full
row — saves the `m × 128256 × 4` logits write + readback. **Table read is still
paid** (fundamental), so this is a smaller win than L1 (the write/readback is
~½ MB × m, not the 657 MB table) — but it also removes the separate
`argmax_rows_*` pass (3%) and the tree logits D2H. Measure before building; may
not clear the 3% bar (cf. ADR 0023's rmsnorm rejection).

### L3 — Batch-slot spec decode (amortizes `V` and `d` across sequences)

The largest architectural item, deferred here but named: run the two-runner
spec loop **per batch slot** (ADR 0025 paged KV + C4 tree coexistence make the
KV side possible). N concurrent sequences share neither `V` nor `d` today
(`--draft-model` forces `--batch-slots 1`). Batched spec amortizes the fixed
verify overhead and lets the drafter's k steps overlap across slots — the win
compounds with L1. This is a multi-week track, not a follow-up; it wants its
own ADR when L1/L2 land.

### L4 — τ via drafter scale/data (attacks the inequality's numerator)

Deprioritized by the response curve: τ ≥ 6 by data alone needs ~50–150M teacher
tokens (40–80 h generation) for a projected +40% τ. A larger drafter (12L/1024)
trades `d` for τ — wrong direction while `d` is the bottleneck. Revisit only
after L1 makes `d` cheap, at which point a bigger, higher-τ drafter is affordable.

## Non-goals / rejected here

- ~~**Graph-capturing the k-step draft as one replay**~~ — RETRACTED
  2026-07-30: the ternary-head experiment decomposed `d` and showed the
  per-token host round-trip (~1.2 ms), not the table read (~0.23 ms), is the
  dominant term. A chained k-step device-side draft is now L1' (see the L1
  refutation above). The original "table-read-bound" claim conflated
  GPU-busy-share with wall share.
- **Reducing the drafter's proposal vocab (top-N frequency prune):** argmax
  over a pruned table saves read proportionally, but a wrong-vocab argmax can
  propose a token the target would never rank first, dropping τ unpredictably;
  L1 (ternary full-vocab) gets the same read saving without the τ risk.

## Consequences

- The spec-decode claim is now honestly bounded: **at parity on single-stream
  greedy prose, gated lossless, with the win path identified and quantified**
  (L1 collapses the dominant `k·d` term). No "2× spec decode" claim ships until
  L1 is measured.
- The cost model (`(V + k·d)/τ < P`) is the reusable lens for every future
  drafter/verifier change — record `V`, `d`, `k`, `τ` per experiment.
- Losslessness is invariant across all four levers (the verifier is the sole
  source of truth); none of them needs a numerics RFC.

## Amendment 1 — tree width from measured margins (2026-07-30)

Authorized by [ADR 0035](./0035-frontier-methods-integration.md), from the
[ADR 0034](./0034-next-gen-ternary-research.md) research intake:

Acceptance theory (arXiv 2606.30265) closes a free parameter in this cost model.
The required target margin for guaranteed greedy acceptance falls with tree width
as `√(4ε/(m+1))` (ε = drafter KL bound), so **m is selectable from the measured
margin distribution of the target on the serving corpus** instead of grid search:
choose the smallest m whose implied margin threshold covers the desired acceptance
mass, then price it with this ADR's existing `(V + k·d)/τ < P` lens (larger m
raises k and V; the margin curve says what τ it buys). Record the margin
distribution alongside `V`, `d`, `k`, `τ` per experiment. The parity verdict and
lever ranking of this ADR are unchanged; this only replaces how m is picked when
the ADR 0021 drafter lands.
