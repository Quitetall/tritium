# 7. Related Work

A note on citation discipline before the survey. Every external number in this
section is cited *as reported by its authors*, under their protocols and their
storage boundaries; per Section 2.5, rates and quality figures are not assumed
comparable unless the storage boundary is stated, and none of these results has
been reproduced under our matched protocol unless explicitly said so. The
subset we intend to reproduce is frozen as the baseline set of the
preregistered, still-unrun campaign of Section 8, and our own verification
sweeps found that a nontrivial
fraction of headline claims in this literature do not survive independent
checking — where a mechanism's headline number failed verification we cite the
mechanism and omit the number.
<!-- receipt: docs/adr/0028-salt-v2-additive-ternarization.md §Research position ("Rates in the table are not assumed comparable unless the same evaluation and storage boundary is stated"); docs/adr/0034-next-gen-ternary-research.md claim boundary (adversarial-verification run; vendor-reported labeling) -->

## 7.1 Quantization-aware training for ternary

The BitNet b1.58 line [REF:bitnet158] [REF:bitnet2b4t] established that flat
ternary weights trained natively can match full-precision quality at scale,
with reported budgets of 100B–4T training tokens. ParetoQ [REF:paretoq]
sharpened the recipe — pretrain in full precision, spend roughly the final 10%
of tokens on QAT — and reported a ternary 600M model outperforming an earlier
ternary 3B. Tequila [REF:tequila] identified the STE deadzone (saturated
weights receive zero gradient and are trapped) and reported a +2.6% average
gain on a 1B model from a direct gradient path for those weights. HESTIA
[REF:hestia] replaces the STE with a temperature-controlled softmax relaxation
over $\{-1,0,+1\}$ and — its actual innovation — schedules per-tensor
temperature by Hessian-trace sensitivity; it reports beating Tequila by
roughly 2.5–2.8 points at a matched 10B-token budget while staying
competitive with 100B-token ternary baselines. LC-QAT [REF:lcqat] couples
a strong PTQ initialization to a smooth differentiable code estimator
(reported Qwen3-8B perplexity 9.72 → 10.23 after a matched 4B-token run).
BCJR-QAT [REF:bcjr-qat] contributes a result we treat as a design constraint
rather than a competitor: cases where per-layer reconstruction MSE improves
while full-model perplexity regresses — the proxy-failure finding behind our
end-to-end acceptance gates (Sections 3 and 5).
<!-- receipt: docs/research-ternary-ecosystem-and-tools.md §3.1, §5.2, §5.3; docs/research-ternary-sota-mid2026.md §8.2; docs/adr/0034-next-gen-ternary-research.md §1.2; docs/adr/0028-salt-v2-additive-ternarization.md §Research position (LC-QAT, BCJR-QAT rows) -->

Closest in spirit to Section 5 is the QAT-plus-distillation conversion route:
Bonsai 27B [REF:bonsai27b] ships a ternary build of the same flagship
checkpoint our preregistered campaign targets, with vendor-reported 94.6%
quality retention at 5.9 GB and one published independent
matched-VRAM reproduction. Reproducing it under our harness is a preregistered
baseline obligation of the unrun 27B campaign, not evidence offered
here.
<!-- receipt: docs/adr/0034-next-gen-ternary-research.md §2.1 (vendor-reported retention; Astezelex reproduction; Stage-8 baseline verdict) -->

## 7.2 Post-training quantization to ternary and sub-2-bit

PT²-LLM [REF:pt2llm] demonstrated that training-free ternarization is viable
at all, reporting a 7B ternarized in 32 minutes on one A800 from 128
calibration samples while matching or beating 2-bit PTQ baselines at lower
memory. PTQTP [REF:ptqtp] is the closest representational relative: two
ternary planes fitted by exact pair assignment with a ridge-regularized scale
solve — machinery Section 3 generalizes to $3^P$ assignment and a conditioned
$P\le 3$ solve — reporting Qwen3-32B perplexity 8.64 → 10.06; its physical
rate at G128 is 4.25 bpw under the accounting of Section 2.5, not the nominal
1.58. CAT-Q [REF:catq] reaches reported QAT-class ternary quality by PTQ from
512 calibration samples, via learnable per-group modulation, a softened
two-sided ternarization relay, and sliding-window output reconstruction; we
adopted the relay as deterministic initialization basins projected through the
exact solver (Section 3.5), and the remaining mechanisms are preregistered
ablation candidates. OA-EM [REF:oaem] showed the initialization basin alone
can dominate: a reported 3B 2-bit pre-refinement perplexity of 352.39 → 16.82
at the same nominal rate — the strongest published evidence against greedy
residual fitting, and a reason our fitter is multi-start by design.
<!-- receipt: docs/research-ternary-sota-mid2026.md §3 (PT²-LLM row); docs/adr/0028-salt-v2-additive-ternarization.md §Research position (PTQTP, OA-EM rows) and §2 (4.25 bpw correction); docs/adr/0034-next-gen-ternary-research.md §1.1; docs/adr/0035-frontier-methods-integration.md §WS-B -->

A parallel line improves PTQ by substituting better curvature without touching
the stored format: GuidedQuant [REF:guidedquant] reports improving QTIP's
Llama2-7B perplexity 6.82 → 6.11 with end-loss curvature, and YAQA [REF:yaqa]
reports 9.39 → 8.39 on Llama3.1-8B with forward-KL Fisher sketches. KronQ
[REF:kronq] and Fisher-Kronecker quantization [REF:fisher-kron] replace
power-iteration sketches with Kronecker-factored approximations at a reported
$\sim 10\times$ lower capture cost. Section 4's evidence layer sits in this
Kronecker family; what it adds is not the factorization but the apparatus
around it — content-addressed evidence identity, the dyadic-reduction design
that makes capture topology provably irrelevant, and fail-closed PSD
validation. BPDQ [REF:bpdq] is our closest optimization peer (exact coordinate
search, Hessian refit, error propagation, all representation-transferable) but
stores binary planes plus a group bias, which Section 2.2 rejects. UniSVQ
[REF:unisvq] similarly reaches reported strong 2-bit Qwen3-32B results
(7.61 → 9.26) through quaternary codes and a dense affine decoder that the
format disallows. Finally, VBQ's learned per-group precision allocation
[REF:vbq] provides convergent — though small-scale and probe-only — evidence
for the premise of Section 2.6: sensitivity is strongly non-uniform across
tensors, and allocation is where the bytes should go.
<!-- receipt: docs/adr/0028-salt-v2-additive-ternarization.md §Research position (GuidedQuant, YAQA, BPDQ, UniSVQ rows); docs/adr/0034-next-gen-ternary-research.md §1.3, §3.1 (VBQ probe caveat: "a deliberately short 1B from-scratch probe") -->

## 7.3 Vector, lattice, and trellis quantization at matched bytes

The strongest sub-2-bit quality numbers belong to methods whose storage the
SALT format deliberately disallows (Section 2.1). AQLM [REF:aqlm] (additive
floating codebooks), VPTQ [REF:vptq] (Hessian-weighted vector quantization),
QuIP\# [REF:quip-sharp] (E8 lattice), and QTIP [REF:qtip] (trellis coding)
report Llama2-70B perplexities of 3.83/3.93/3.91/3.70 at
approximately 2-bit payloads (the QuIP\# and QTIP figures with
fine-tuning); QTIP also reports the measured runtime bar,
188 tok/s (7B) and 23.5 tok/s (70B) on an RTX 6000 Ada. LLVQ [REF:llvq]
(Leech lattice with spherical gain coding) reports Llama2-7B 6.83 → 5.48 after
scale-only tuning. PV-Tuning [REF:pvtuning] contributes the
representation-agnostic piece — alternating continuous/discrete optimization —
that the preregistered campaign's refined track inherits. At the sub-ternary
extreme, LittleBit
[REF:littlebit] reports 0.1 bpw via low-rank binarized factorization; its
dense low-rank matmuls violate the execution invariant, but it marks where the
matched-bytes frontier continues below ternary. These methods are matched-byte
comparison targets and donors of optimization machinery, not admissible
representations here.
<!-- receipt: docs/adr/0028-salt-v2-additive-ternarization.md §Research position (AQLM, VPTQ, QuIP#, QTIP, LLVQ, PV-Tuning rows); docs/adr/0034-next-gen-ternary-research.md §1.5 (LittleBit: "the matched-bytes comparison belongs in the paper's related work") -->

The Ordentlich–Polyanskiy quantized-matmul series [REF:op2026] supplies the
frame in which that comparison should be conducted: the quantity to preserve
is the distortion of the *product* $Wx$, not of the weights; universal
lattice quantizers achieve the matching-lower-bounded optimal product
distortion; and properly scaled ternary is competitive with lattice
constructions for well-behaved weight distributions, with lattices winning
under heavy tails and fine-grained sub-2-bit rate control. This is the
information-theoretic justification for the output-aware objective of
Section 4, and it predicts exactly where the residual lattice advantage lives
— tail shaping — which is why incoherence-processing front ends remain a
preregistered ablation rather than a format change.
<!-- receipt: docs/adr/0034-next-gen-ternary-research.md §4.1 (O-P frame, ternary-vs-lattice regimes, rotation-ablation verdict) -->

## 7.4 Systems and kernels

On CPU, the LUT lineage — T-MAC [REF:tmac] and its parallel-inference
successor Vec-LUT [REF:veclut] — is the reference; Vec-LUT reports up to
$4.2\times$ over scalar-LUT stacks on edge CPUs and a 1.60 bpw lossless
ternary packing, a figure comparable to our codecs only at a stated storage
boundary (Section 2.5). Microsoft's BitNet GPU kernels [REF:bitnetcpp]
(register-resident W2A8 DP4A) report $3.17\times$ over BF16 on an A100 GEMV
shape. The mainstream container is converging on llama.cpp's 2-bit
ternary-capable Q2_0 [REF:llamacpp-q2_0], whose ternary subset's relationship
to SALT — the base plane of a nested refinement, exportable at the base rate
— Section 2.7 details. We
also note FairyFuse [REF:fairyfuse], a multiply-free CPU kernel whose reported
claim that ternary regresses $130\times$ on GPUs rests on a GPU baseline that
ports the CPU bit-extraction algorithm; GPU-native ternary paths, including
Section 6's, are counterexamples. On structured sparsity, SlideSparse
[REF:slidesparse] reports that strict 2:4 pruning collapses reasoning quality
while 6:8 — executed on existing sparse tensor cores via a lossless
sliding-window decomposition — retains it at a measured $1.33\times$
end-to-end; this is directly relevant to the S34 codec's train-under-the-
constraint stance (Section 2.4). For activations rather than weights,
Block-GTQ [REF:blockgtq] reports that structure-aware non-uniform allocation,
not more bits, is what makes $\sim$2-bit KV viable (NIAH 70.6 → 97.4 at a
matched budget) — consistent with our measured rejection of *uniform* ternary
KV in Section 5.5 — and DF-SSM [REF:dfssm] opens the recurrent-state analogue
for linear-attention layers.
<!-- receipt: docs/research-ternary-sota-mid2026.md §1.2 (Vec-LUT), §1.4 (FairyFuse; the GPU-baseline critique verified in §6 item 9); docs/research-ternary-ecosystem-and-tools.md §1.4, §2.1 (T-MAC, BitNet kernels); docs/adr/0034-next-gen-ternary-research.md §2.3, §3.4, §3.5 -->

## 7.5 Positioning

Each of SALT's mechanisms, taken alone, has contemporaries: additive
multi-plane fitting (AQLM, BPDQ, PTQTP), softened-relay ternarization (CAT-Q
— adopted as initialization basins, Section 3.5), differentiable relaxation
with sensitivity-scheduled annealing (HESTIA — adopted as a preregistered
candidate for the campaign's refined track), Kronecker-factored curvature (KronQ,
Fisher-Kronecker), sensitivity-directed allocation (HAWQ [REF:hawq],
SqueezeLLM [REF:squeezellm], VBQ), and distillation-driven recovery (the
BitNet training lineage). We claim novelty for none of them in isolation, and
where we adopted a contemporary's mechanism we cite it at the point of use.
The contribution is the combination held together by two things the
individual works do not share. First, an execution invariant: every mechanism
must emit pure scales-and-trits, $\hat{W} = \sum_p s_p T_p$, executable by
add/subtract/skip without dense materialization — which is why relay
parameters are basin-internal, why curvature changes the objective but never
the format, and why the lattice methods above are baselines rather than
options. Second, an evidence apparatus: exact physical-byte accounting, exact
joint solves with oracle-checked optimality on tractable cases,
content-addressed curvature identity, preregistered ablation brackets, and
acceptance gates that have demonstrably said no (Section 5.5). Convergent
work elsewhere reinforces individual design choices — MoTE's
all-routed-experts-ternary recipe [REF:mote] is architecture-level
convergence, spending capacity on more low-precision experts rather than
fewer high-precision ones; it is cited here for its recipe
only, its headline iso-memory comparison having failed our independent
verification — but we know of no contemporary system that combines the
invariant, the joint solver, curvature-directed allocation, and the
measurement discipline in one auditable pipeline. That combination, and the
discipline that makes it falsifiable, is what this paper contributes.
<!-- receipt: docs/adr/0035-frontier-methods-integration.md §Consequences ("the SALT V2 paper's novelty must now be argued as the combination"); docs/adr/0034-next-gen-ternary-research.md §3.2 (MoTE recipe verified 3-0; iso-memory number REFUTED 0-3 — "do not cite it"), §Consequences -->

<!-- UNSOURCED:
- LC-QAT "at 2 bits" (§7.1, removed from prose): the repo sources for the
  Qwen3-8B 9.72 → 10.23 pair (docs/adr/0028 §Research position; docs/
  research-ternary-sota-mid2026.md §8.2) give the perplexities and the matched
  4B-token budget but never state LC-QAT's bit width. Restore only after
  checking arXiv 2606.10531 directly.
-->

