# SALT: Sensitivity-Allocated Layered Ternarization with Exact Byte Accounting

**Brian K. Lam**
briankhanglam@gmail.com

*Draft 0 — 2026-07-30. arXiv category: cs.LG (cross-list: cs.AR).*
*Companion artifact: the Tritium repository (Apache-2.0). Every number in this
paper regenerates from a committed command.*
<!-- receipt: docs/paper/salt-whitepaper-outline.md header (category, companion-artifact discipline); docs/BENCHMARKS.md "Every number in this file reproduces from one command" -->

## Abstract

We present SALT (Sensitivity-Allocated Layered Ternarization), which represents
each weight group as an additive sum of at most three ternary planes with
non-negative scales, $\hat{W} = \sum_p s_p T_p$ — zero-point-free by
construction, with per-group plane counts allocated from measured curvature.
<!-- receipt: docs/adr/0028-salt-v2-additive-ternarization.md §1; crates/tritium-format/src/salt_v2_package.rs (SALT_V2_MAX_PLANES = 3) -->
Fitting is a deterministic joint solver: an exact $3^P$ assignment step under
separable metrics, a conditioned closed-form scale solve, and
accept-only-on-improvement alternation; softened-relay initialization basins
widen the start set and are never worse by construction, improving 7.4% of
sampled groups at $P=3$.
<!-- receipt: crates/tritium-quantize/src/salt_v2.rs (fit_joint_ternary, exact_ternary_assignment, solve_scales, relay_basins_never_worsen_the_final_objective); docs/receipts-ws-b-relay-basin-ab.txt (57/768 = 7.4% at P=3) -->
Curvature evidence is Kronecker-factored (input Gram $\times$ output Fisher)
with a content-addressed identity that provably excludes capture topology;
shared-forward capture cuts calibration replays $3.0\times$ at 1.7B while
producing byte-identical records.
<!-- receipt: crates/tritium-quantize/src/salt_v2_curvature.rs (CurvatureSourceId, canonical dyadic reduction); docs/receipts-ws-a1-cost-baseline-17b.json (replay_reduction=3.0, byte_identity=true) -->
End-to-end straight-through distillation defeats catastrophic post-training
ternarization: on held-out WikiText-2, a 135M student recovers from
$3.28\times 10^6$ (PTQ) to 139.6 perplexity — a 23,493$\times$ recovery — with
the curve still descending when the 480k-token pool ran out; the result is
token-limited and we claim no floor.
<!-- receipt: docs/adr/0029-training-throughput-tensor-cores.md:168-192 (fp 23.827, PTQ 3.281e6, best 139.6 @14,500 steps, 23,493x, curve 1286→365→194→140 never flattening) -->
A native engine executes the packed planes without dense materialization:
$\approx$273–303 tok/s BitNet-2B4T decode on one RTX 4090 ($\sim$474 GiB/s
effective weight stream at the ledger median) and 12.3K tok/s 512-token
prefill through bit-identical kernels.
<!-- receipt: docs/BENCHMARKS.md (2026-07-30 entry: 264-281 median ~277 contended, ~474 GiB/s = 1.71 GiB × ~277; 2026-07-11 reference table 301.4-302.8; quiet-box re-baseline 273.2-275.9 — session medians span ~273-303 within the documented ±10% contention spread); pp512 12,274.7 sustained (2026-07-18 "v3 Q-blocked attention" entry); bit-identity gates at lines 81, 111-113, 133-134 (IMMA == dp4a, v2/v3 attention == rev-1, by to_bits) -->
Every artifact reports exact physical bytes — logical bits-per-weight is banned
from claims — every reported number regenerates from a committed command, and a
preregistered larger-scale campaign with frozen acceptance gates is future
work.
<!-- receipt: docs/adr/0028 §2 (physical-byte accounting, logical-bpw ban); docs/BENCHMARKS.md ledger discipline; docs/plans/0043-salt-v2-sota-campaign.md (preregistered campaign, frozen gates) -->

**Keywords:** ternary quantization; additive quantization; knowledge
distillation; Kronecker-factored curvature; reproducibility; efficient LLM
inference
