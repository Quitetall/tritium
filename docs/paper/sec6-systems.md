# 6. Systems: executing packed planes

Sections 2–5 argued that the SALT representation earns its constraints at fitting and recovery
time. This section reports the serving half of the bargain: a native CUDA engine that executes
packed ternary planes competitively, and — equally load-bearing for this paper — the ledger
discipline under which every performance number was recorded. All measurements are on a single
RTX 4090; the served model is BitNet b1.58 2B4T [REF:bitnet2b4t] in the one-plane ternary I2_S
container — the $P_g = 1$ special case of $\hat{W}_{g,i} = \sum_p s_{g,p} T_{g,p,i}$ (§2.1). A
$P$-plane SALT weight is $P$ passes of the same multiply-free kernels; no multi-plane serving
numbers are claimed here.
<!-- receipt: docs/BENCHMARKS.md:42-48 (model/box); execution model §2.1 / docs/adr/0028 §6 -->

## 6.1 Kernel families, each behind a gate

Four kernel families carry the engine; each was admitted under an exact-equality gate rather than
a tolerance hope, within a twin-kernel contract that pins the kernel population and its cause
chain in a drift test.
<!-- receipt: docs/adr/0022-twin-kernel-contract.md (71 __global__ kernels, drift test, per-twin to_bits gates) -->

*Decode.* Single-stream generation runs int8 `dp4a` dot-product GEMMs directly over packed trits,
dispatched as a CUDA graph of ≈370 kernel nodes per token. Decode is memory-bound — the per-token
cost is streaming the resident weight bytes once — so effective weight-stream bandwidth (decode
tok/s × weight bytes) is the honest self-metric and the cross-model metric of §6.4.
<!-- receipt: docs/adr/0019-persistent-megakernel-decode.md:7-10 (dp4a i8 GEMMs, ~370-node graph); docs/BENCHMARKS.md:26-27 (metric definition) -->

*Prefill.* Prompt processing runs IMMA tensor-core GEMMs on the same packed operands. The
admission gate is bit-identity to the scalar dp4a path: `to_bits` equality over all 128,256
logits of a real-model forward, one-shot $M{=}512$ equal to $4\times128$-chunked equal to dp4a.
Passing it required unifying the epilogue association with dp4a's
(`(float)acc · wscale · act_scale`, a pure multiply chain admitting no contraction); under the
prior association 32.8% of elements ULP-diverged — the gate had teeth before it had a pass.
Autotune acceptance is likewise bit-strict: a faster but non-identical tile cannot win.
<!-- receipt: docs/OPTIMIZATION-LOG.md:1080-1086 (round 22 step 1), 1098-1103 (cuda_imma_prefill_matches_dp4a_bit_exact) -->

*Attention.* Decode attention is a split-KV partial+combine online-softmax in the flash-decoding
style [REF:flashdecoding]. The prefill rewrites (v2, then Q-blocked v3) are order-preserving
mechanics rewrites — every pinned per-row reduction order is the reference kernel's verbatim —
and bit-identical to their predecessors by per-(row, head) `to_bits` gates. A flash-style
reordering is deliberately off the table: summation orders are pinned by decode parity and the
tree-verify gate, and changing them is a numerics RFC, not an optimization.
<!-- receipt: docs/adr/0014-spec-decode-bastion.md:13,55 (split-KV); docs/OPTIMIZATION-LOG.md:1143-1151 (v2 order-preserving, flash off the table), 1177-1188 (v3 pinned orders, bit-identity gate) -->

*Tree verify.* The speculative-decoding verifier is an ancestor-masked sibling of split-KV: $N$
tree nodes share a prefix KV and each attends only its ancestors. Its gate is losslessness, not
speed: `cuda_tree_verify_greedy_lossless` requires chains, branches, and partial and full rejects
to commit exactly the plain-greedy stream. The verifier is the sole source of truth, so
losslessness is structural and survives every drafter change in §6.6.
<!-- receipt: docs/adr/0014-spec-decode-bastion.md:3-10,76-79; docs/adr/0032-spec-decode-cost-model-and-next-levers.md:209-210 -->

## 6.2 The ledger as method

Performance claims in this space routinely fail to specify what was run, on what, next to what.
The ledger legislates against this: every number reproduces from one committed command, recorded
beside an environment capture (GPU, driver, VRAM, co-resident compute processes, date, git
commit); competitor invocations are documented, not wrapped; the ledger updates only by
re-running the harness. The capture is not decorative: its first run disclosed a co-resident
`llama-server` contending the box.
<!-- receipt: docs/BENCHMARKS.md:1-23; docs/OPTIMIZATION-LOG.md:1125-1127 (first-run disclosure) -->

Three further rules were each bought with a documented mistake:

1. **Order-alternated (ABBA) interleaving.** Numbers move ±10% under desktop load, so close
   comparisons interleave A/B/B/A rather than running blocks. Origin: a bench run showing a
   monotonic 302→206 tok/s decay — thermal drift masquerading as a result — was discarded; the
   ABBA protocol is the keeper.
   <!-- receipt: docs/BENCHMARKS.md:30-32; docs/OPTIMIZATION-LOG.md:1066-1068 -->
2. **The stale-binary trap.** One session benched a release binary built before the commit under
   test and read flat. The ABBA matrix caught it — A equaled B on a toggle that could not be a
   no-op — and `nsys` showed stale kernel names and tune keys. An `nsys` kernel-name check is now
   the staleness gate before any ledger entry.
   <!-- receipt: docs/OPTIMIZATION-LOG.md:1169-1173; docs/BENCHMARKS.md:115-117 -->
3. **Quiet-box and clock-state rules.** Short runs at idle clocks (1.1 of 3.1 GHz) produced a 3×
   spread; the reportable numbers on a desktop box are sustained multi-run p50s and in-session
   relative ratios, with contended entries labeled as such and a quiet-box rerun owed before
   publication.
   <!-- receipt: docs/OPTIMIZATION-LOG.md:1199-1201; docs/BENCHMARKS.md:62-63 -->

## 6.3 Worked example: the prefill arc

The prefill campaign is the discipline operating end-to-end: four measured states, each with a
command, an environment line, a bit-identity gate, and a profile that chose the next lever.

**Table 6.1 — pp512 prefill, BitNet 2B4T, RTX 4090, canonical ledger command.**
<!-- receipt: docs/BENCHMARKS.md:69-137,139-153; docs/OPTIMIZATION-LOG.md:1074-1201 (rounds 22-24) -->

| date | change | pp512 tok/s | gate |
|---|---|---:|---|
| 2026-07-11 | dp4a baseline | 1,068 | — (reference) |
| 2026-07-12 | IMMA wired to prefill | 1,969.7 (+84%) | bit-identical to dp4a, 128,256 logits |
| 2026-07-18 | rev-4 IMMA (cp.async, packed smem) + v2 attention | 8,440–9,069 | both legs bit-identical to predecessors |
| 2026-07-18 | v3 Q-blocked attention (8 rows/block) | **12,274.7** (ABBA span 12.1–16.9K across clock states) | bit-identical to rev-1 per (row, head) |

Profile corrections steered the arc. After the wiring win, `nsys` refuted the launch-overhead
hypothesis — kernel time ≈ wall time, so graphing prefill would buy nothing — and split the
remaining wall 63% GEMM / 35% attention, naming the levers. Rev-4 IMMA staging then delivered 22×
on the GEMM share (197 → 8.8 ms) against a 5.6× ask, flipping the profile to 74% attention; v2's
coalesced staging took attention 2.4×; v3's 8-row Q-blocking added 2.2–2.8× (the in-session ABBA
ratio being the trustworthy number), landing a balanced 45%/43% profile. Decode was unchanged
throughout — v3 on/off ABBA pairs are statistically indistinguishable, as a prefill-only dispatch
must be. The cumulative 11.5× required zero numerics re-gating: every kernel along the arc is
bit-identical to its predecessor — the campaign's design constraint, not its accident.
<!-- receipt: docs/OPTIMIZATION-LOG.md:1105-1123 (round 22 profile + refuted hypothesis), 1135-1160 (round 23), 1177-1197 (round 24); docs/BENCHMARKS.md:102-117 (config-isolation ABBA table), 126-128 (decode indistinguishable) -->

## 6.4 Decode, and a same-box comparison on the day the gap closed

Steady-state decode on 2B4T sits at 264–303 tok/s across the documented box states: 301.4–302.8
under lighter desktop load (256 steps), 273.2–275.9 on the quiet-box re-baseline (512 steps),
264–281 (median ≈277) on the contended comparison day below — all inside the ledger's ±10%
contention spread. At the comparison-day median, the engine streams the 1.71 GiB of resident
ternary weights at ≈474 GiB/s effective.
<!-- receipt: docs/BENCHMARKS.md:146,151-153 (301.4-302.8; quiet-box 273.2-275.9), 48 (264-281, ~474 GiB/s) -->

On 2026-07-30, llama.cpp merged a CUDA path for its Q2_0 format [REF:llamacpp-q2_0] — the
one-plane container of §2.7 — and the ledger recorded a same-day, same-box head-to-head. We
reproduce the entry with its caveats intact, because the caveats are the method.

**Table 6.2 — same-box decode, 2026-07-30 (RTX 4090; ~5.3 GB co-resident desktop load —
CONTENDED, both engines equally; ABBA interleaved; llama.cpp upstream master `5f55650a7`,
build 1212, fresh CUDA build; Tritium at HEAD via the ledger command).**
<!-- receipt: docs/BENCHMARKS.md:39-64 -->

| engine | model | weights | decode tok/s (spread) | eff. weight stream |
|---|---|---|---:|---:|
| Tritium CUDA | BitNet 2B4T ternary I2_S | 1.71 GiB | 264–281 (median ≈277) | ≈474 GiB/s |
| llama.cpp CUDA (new Q2_0) | Qwen3.5-4B Q2_0 g64 | 1.42 GiB | 223–264 (mean-of-runs 248) | ≈352 GiB/s |
| llama.cpp CUDA (prior ledger line) | Qwen3.5-4B TQ2_0 | 1.35 GiB | 24.0 | ≈32 GiB/s |

The models differ (2.4B BitNet vs 4B Qwen), so raw tok/s is not the comparison; effective
weight-stream bandwidth is — single-stream decode is memory-bound in both engines, and
normalization divides out model size. Read honestly: llama.cpp's new path is real — a
~10× jump over its own TQ2_0 line, within ~25–35% of Tritium's bandwidth-normalized efficiency
on day one — and on this box, on this day, Tritium's bandwidth-normalized efficiency measured
higher (~474 vs ~352 GiB/s), while "only ternary CUDA engine" retired as a claim as of that
date. Prefill is already a parity band (Q2_0 12,199 ± 1,109 vs
Tritium's 12,275 sustained). Two debts are recorded, not hidden: the quiet-box rerun owed before
this table is publication-final (a pre-submission TODO), and the same-model follow-up — one Q2_0
artifact served by both engines via Tritium's Q2_0 import (§2.7) — removing the architecture
confound.
<!-- receipt: docs/BENCHMARKS.md:52-63; quiet-box rerun docs/paper/salt-whitepaper-outline.md:95 -->

## 6.5 A measured negative: the megakernel deferral

The classic systems argument for many-small-kernel decoders is the persistent megakernel: fuse
the ~370-node per-token graph into one cooperative launch and eliminate dispatch boundaries. The
discipline required measuring the premise before any design work. A microbenchmark of 370 stage
boundaries at decode-like grids priced the alternatives — CUDA-graph node 0.77–1.22 µs,
cooperative `grid.sync()` 0.64–0.98 µs, eager launch 1.3–1.4 µs per boundary — and the decode
profile added two facts: the sum of per-kernel execution medians (≈3.0 ms) already matches the
measured wall per token (≈2.9 ms), so boundaries are nearly free; and the dominant kernels sit at
internal floors a megakernel cannot move (a 656 MB/token f16 LM-head read at bandwidth
speed-of-light, DRAM-bound dp4a GEMMs, latency-chain-bound norms). Projected upside: ≈74 µs/token
— about 3% — against the largest structural rewrite in the codebase. Deferred, with revisit
conditions written down. The measurement cost a scratch harness; the rewrite it prevented would
have carried a re-verification of the whole bit-match discipline in one translation unit.
<!-- receipt: docs/adr/0019-persistent-megakernel-decode.md (premise experiment, projected 3%, decision) -->

## 6.6 Speculative decoding: a cost model instead of a claim

Speculative decoding [REF:leviathan2023spec] multiplies tokens per verifier forward, and the
tree-verify primitive of §6.1 (BASTION-style [REF:bastion2026]) makes Tritium a lossless
verifier. The results divide by drafter. A model-free prompt-lookup drafter [REF:promptlookup]
reached **1.19× end-to-end, lossless,** on repetitive text — itself only after three measured
passes dragged the verify from an initial 0.61×, slower than plain decode.
<!-- receipt: docs/OPTIMIZATION-LOG.md:544-565 (round 9) -->

The trained-drafter track is reported as a cost model, because for most of the campaign its
verdict was *parity*. Per cycle, one verify (cost $t_V$) commits $\tau$ tokens after $k$ drafter
steps (cost $t_d$ each); against plain decode at $t_P$ per token, spec wins iff
$(t_V + k\,t_d)/\tau < t_P$. Measured: $t_V \approx 10$ ms, $t_P \approx 3.9$ ms,
$t_d \approx 1.5$–$2$ ms,
$\tau \approx 4.2$, $k \approx \tau$ — 4.4 ms per committed token vs 3.9 plain, ≈0.89×, shaved to
≈parity by adaptive draft length and device argmax. A lossless drafter at $\tau = 4.2$ *should*
win; the model says why it did not: the $k\,t_d$ drafter term, not the verify, is the killer.
<!-- receipt: docs/adr/0032-spec-decode-cost-model-and-next-levers.md:15-31,62-83 -->

The model then earned its keep by being wrong in a measurable place. The top-ranked lever — a
ternary drafter LM head, collapsing the drafter's 197 MB f16 table read — was built and refuted:
τ fell 24% (4.23 → 3.23), and the wall decomposition showed the table read (~0.23 ms) was only
~15% of the ~1.7 ms drafter step, whose wall time is dominated by per-token host orchestration
(≈1.2 ms of host round-trip). Wrong bottleneck, recorded as a refutation — like the earlier
device-argmax change, a "correct change, wrong bottleneck" no-win that removed a readback that
was never the cost. The redirected lever — a chained $k$-step device-side draft,
one host round-trip per chain — measured **1.144× vs plain decode** (1.253× vs the per-step
ladder) under a same-session interleaved A/B, 12/12 outputs identical: the campaign's first real
spec-decode win, honestly bounded. No "2× spec decode" claim ships from this paper; the cost
model, with $t_V$, $t_d$, $k$, $\tau$ recorded per experiment, is the deliverable.
<!-- receipt: docs/adr/0032-spec-decode-cost-model-and-next-levers.md:87-134 (L1 refuted, L1' measured), 56-60 (device-argmax no-win) -->

The pattern across §6.3–6.6 is this section's actual claim: the same machinery that admitted an
11.5× prefill arc and a 474 GiB/s decode stream also deferred a fashionable rewrite at a
projected 3%, discarded contaminated runs, and published its competitor's good day with the
contention disclosed. Every number here regenerates from a committed command; Section 9 states
that contract in full.

<!-- UNSOURCED (removed from prose; restore only with a receipt):
- §6.5 previously said the deferred megakernel rewrite "would have cost weeks" — no duration
  estimate exists anywhere in docs/adr/0019-persistent-megakernel-decode.md; the ADR prices the
  cost as the largest structural rewrite in the codebase plus re-verifying the whole ADR 0018 /
  bit-match discipline inside one translation unit. Prose now states the ADR's cost verbatim. -->
