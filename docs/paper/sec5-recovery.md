# 5. Recovery: distillation defeats post-training ternarization

Sections 2–4 established the SALT representation, the exact fitting machinery, and the curvature
evidence that allocates planes. This section reports what those components are *for*: post-training
ternarization (PTQ) of a small dense model is catastrophic, and end-to-end distillation from the
full-precision teacher recovers it by more than four orders of magnitude on a recognised held-out
corpus. We also report two findings from the recovery campaign that revise our own earlier
interpretation of the data, and close with the negative results that demonstrate the acceptance
discipline of Section 1 has teeth.

## 5.1 Protocol

The recovery loop is SALT-aware straight-through distillation (SALT-STE). The student holds fp32
latent masters $W$ for every 2-D projection; each forward pass quantizes them through the $T$-plane
residual ternarizer of Section 2, $\hat{W} = Q_T(W)$, and the backward pass applies the
straight-through estimator [REF:bengio2013ste]: because the reconstruction tracks the latent, the
gradient with respect to $\hat{W}$ is passed to $W$ unchanged,
$\partial\mathcal{L}/\partial W := \partial\mathcal{L}/\partial\hat{W}$.
The training signal is full-vocabulary logit distillation [REF:hinton2015distillation] from the fp
teacher: with teacher soft targets $p^{(t)}_{ij}$ over vocabulary $V$ and student logits
$z^{(s)}_i$ at each of $S$ positions,

$$\mathcal{L} = -\frac{1}{S}\sum_{i=1}^{S}\sum_{j\in V} p^{(t)}_{ij}\,\log\,\mathrm{softmax}\big(z^{(s)}_i\big)_j,$$

which equals the KL divergence to the teacher up to the (constant) teacher entropy. Masters are
updated with AdamW [REF:loshchilov2019adamw]; all runs reported here use the bit-exact f32 training
path (the reduced-precision optimizer variants of the training engine are validated but disabled
for these numbers). <!-- receipt: docs/adr/0029-training-throughput-tensor-cores.md:16-17,102-106,136-143; crates/tritium-nn/tests/salt_distill_heldout.rs:119-173 -->

The mechanism was gated bottom-up before any model-level claim: the atomic SALT-STE loop — one
projection, one target, the toy layerwise distillation of the training-plan gate — recovers 92.6%
of the PTQ-induced output error, establishing that the biased STE gradient learns through the
$T$-plane quantizer at all. <!-- receipt: docs/plans/0038-salt-distillation.md:46 -->

Corpus discipline is the load-bearing part of the protocol. Early in-sample experiments (train and
evaluate on the same tokens) measure memorization, not recovery; every number below is held-out.
The corpus generator tokenizes WikiText-2-raw [REF:merity2016wikitext] with the student's own
tokenizer and emits a train pool and an evaluation set drawn from the *disjoint test split* — a
stronger separation than a tail split of one document. The runs below use a 500k-token train pool
and a 4,096-token held-out set. <!-- receipt: tools/gen_corpus.py:96-102; docs/adr/0029-training-throughput-tensor-cores.md:165-167 -->
Held-out perplexity is scored in non-overlapping 512-token windows; the window bound exists because
tape-based evaluation memory is quadratic in evaluated length, and the window was chosen so the
committed regression-gate numbers are unchanged. <!-- receipt: crates/tritium-nn/tests/salt_distill_heldout.rs:30-36 -->
A single training run traces the whole recovery-vs-tokens curve via periodic checkpoint evaluation
(`TRITIUM_DISTILL_CURVE`), rather than retraining per budget; a data-order seed knob
(`TRITIUM_DISTILL_SEED`) rotates the training-window order and is the stochasticity axis for the
error-bar runs. <!-- receipt: crates/tritium-nn/tests/salt_distill_heldout.rs:442-479 -->

## 5.2 The headline arc

The student is SmolLM2-135M [REF:smollm2] at $T{=}2$ SALT planes. Direct PTQ at this scale is not
merely lossy but catastrophic: held-out perplexity rises from 23.827 (fp teacher) to $3.281\times
10^6$ — a factor of roughly $1.4\times 10^5$, i.e. the quantized model retains essentially no
predictive structure. <!-- receipt: docs/adr/0029-training-throughput-tensor-cores.md:168 -->
Table 5.1 gives the recovery arc; we write recovery $R = \mathrm{ppl}_{\text{PTQ}} /
\mathrm{ppl}_{\text{student}}$ and gap $G = \mathrm{ppl}_{\text{student}} /
\mathrm{ppl}_{\text{fp}}$.

**Table 5.1 — WikiText-2 held-out perplexity, SmolLM2-135M, $T{=}2$.**
<!-- receipt: docs/adr/0029-training-throughput-tensor-cores.md:168-189 -->

| configuration | held-out ppl | gap to fp $G$ | recovery vs PTQ $R$ |
|---|---|---|---|
| fp teacher | 23.827 | 1× | — |
| ternary PTQ | $3.281\times 10^6$ | $\sim 1.4\times 10^5$× | 1× |
| distilled, 160k tok, constant LR $2\times 10^{-3}$ | 563.3 (best 431.2) | 23.6× | 5,824× |
| distilled, 160k tok, warmup 200 + cosine | 266.9 (best 265.8) | 11.2× | 12,292× |
| distilled, 480k tok (full pool), warmup 500 + cosine | 148.5 (**best 139.6** @14,500 steps) | 6.2× (**best 5.9×**) | 22,097× (**best 23,493×**) |

The headline: end-to-end SALT-STE distillation takes the ternary student from $3.28\times 10^6$ to
139.6 held-out perplexity — a 23,493× recovery — landing at 5.9× the fp teacher on 480k training
tokens. <!-- receipt: docs/adr/0029-training-throughput-tensor-cores.md:186 --> We make no claim
that 5.9× is where the method stops; Section 5.4 explains why.

**Figure 5.1 (placeholder) — held-out perplexity vs training tokens, constant-LR vs scheduled.**
Three-seed error-bar reruns (rotating the data-order seed) are planned — the top-ranked
pre-submission TODO — and will replace the
single-seed curves here; the single-seed data plotted in the interim is the committed curve
$1286 \to 365 \to 194 \to 140$ at steps $500/5000/11000/14500$.
<!-- receipt: docs/adr/0029-training-throughput-tensor-cores.md:189; seed knob crates/tritium-nn/tests/salt_distill_heldout.rs:474-479; rerun docs/paper/salt-whitepaper-outline.md:93-94 -->

## 5.3 Finding 1: the plateau was an optimization artifact — a correction

Our first convergence study, on a small committed 8k-token fixture, plateaued at 224.8 perplexity
(11.4× that corpus's fp reference of 19.73) after nine epochs, oscillating in a 220–290 band, and
we recorded it at the time as the data's ceiling.
<!-- receipt: docs/adr/0029-training-throughput-tensor-cores.md:157-163; docs/adr/0031-reduced-precision-optimizer-and-step1-next-steps.md:32-35 -->
That reading was wrong, and the WikiText-2 runs show why. The constant-LR baseline
($2\times 10^{-3}$, the regression-gate default) descended from 3,250 to 431 by step 2,800 and then
oscillated flat (±25% tail) for a further 2,200 steps. Adding a linear-warmup-plus-cosine schedule
[REF:loshchilov2017sgdr] — already implemented in the trainer but unused — beat the constant-LR run
at *every* checkpoint, broke through the 431 floor at step 2,400, and settled at 266.9: a 2.1×
better final perplexity, 2.1× more recovery, and 3× less tail oscillation, with no other variable
changed. <!-- receipt: docs/adr/0029-training-throughput-tensor-cores.md:170-178 -->
What we had reported as a data or method ceiling was substantially the optimizer failing to anneal.
We state this as a correction of our own earlier interpretation rather than a new result: the
lesson for measurement discipline is that a plateau is not evidence of a floor until the learning
rate schedule has been eliminated as the cause.

## 5.4 Finding 2: token-limited, therefore no floor claim

The second finding is what the full-pool run did *not* show. Tripling the training tokens (160k to
480k) took the gap from 11.2× to 5.9× fp, and the curve was still descending when the pool ran out
— 1,286 at step 500, 365 at 5,000, 194 at 11,000, 140 at 14,500 — never flattening.
<!-- receipt: docs/adr/0029-training-throughput-tensor-cores.md:186-189 -->
Each lever applied so far (the schedule, more tokens) roughly halved the gap, and neither is
exhausted. The honest consequence cuts both ways: we cannot claim the method converges to fp, and
we equally cannot claim a "small-model ternary floor" exists at this scale — nothing in this data
has plateaued. The recovery curves in this paper are token-limited measurements, not asymptotes,
and we make no floor claim of either sign. Untested levers recorded for follow-up: a larger corpus
slice (WikiText-103/C4), $T > 2$ planes, and larger students.
<!-- receipt: docs/adr/0029-training-throughput-tensor-cores.md:190-192 -->

## 5.5 Negative results: the acceptance discipline binds

A measurement discipline is only credible if it sometimes says no. Two rejections from the same
campaign period are part of this paper's record precisely because they went against our hopes.

*Uniform ternary KV cache, rejected by measurement.* Extending ternarization from weights to the KV
cache (per-64-group symmetric three-level quantization of both K and V, scale $1.5\times$
group-absmean, run through the unmodified int8 attention kernels) degraded perplexity by a relative
error of $3.7\times 10^{-1}$ — catastrophic against the quality bar — while delivering no speedup
at 4K context. The rung was rejected; KV activations are not weights, and nothing makes them
ternary by nature. The experiment harness remains selectable, and the untested asymmetric variants
(V-only ternary, smaller groups) are recorded, not claimed.
<!-- receipt: docs/OPTIMIZATION-LOG.md:740-753 (ADR 0020 rung 3) -->

*Fast rmsnorm tier, rejected below the improvement bar.* A relaxed-reduction rmsnorm kernel
(fused stages, reordered accumulation) measured a median +1.75% decode throughput under an
order-alternated A/B protocol — below the pre-registered ≥3% acceptance bar under every reading —
and was deleted by its own decision rule, despite passing all correctness gates.
<!-- receipt: docs/OPTIMIZATION-LOG.md:1041-1068; docs/adr/0023-relaxed-reduction-tier.md:3 -->

Both rejections were cheap by design (three append kernels; one kernel variant) and both produced
information: the KV result bounds where ternarization applies, and the rmsnorm result refuted a
profiler attribution that had suggested a 32% optimization target. The same acceptance machinery
that admits the 23,493× recovery number is the machinery that discarded these — which is, we
suggest, the reason to believe the former.
