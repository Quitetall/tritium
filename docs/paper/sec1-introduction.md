# 1. Introduction

Ternary weight representations — every weight constrained to $\{-1, 0, +1\}$, so
that matrix multiplication degenerates into add, subtract, or skip — have moved
from provocation to shipping model class. BitNet b1.58 [REF:bitnet158]
established that ternary language models can be trained to competitive quality;
an open 2.4B-parameter ternary checkpoint followed [REF:bitnet2b4t]; a
1B/3B family demonstrated that one pretraining run can yield both ternary and
bfloat16 variants of the same model [REF:falconedge]; and a recent Apache-2.0
release ships 1.7B/4B/8B checkpoints that are ternary throughout, embeddings
and language-model head included [REF:bonsai-family]. On the post-training side,
ternarization without retraining has reached peer-reviewed viability at 7B
scale [REF:pt2llm]. The appeal is structural: single-stream decode on commodity
GPUs is bounded by the weight stream, and ternary attacks that bound directly
while replacing multiplies with additions and admitting a zero-state skip.

The execution layer, meanwhile, is commoditizing. On 2026-07-30 — the day this
draft was frozen — the mainstream llama.cpp runtime merged CUDA kernels for its
2-bit ternary-capable Q2_0 format [REF:llamacpp-q2_0], and in a same-box,
same-day, interleaved measurement the day-one upstream path already lands
within 25–35% of the bandwidth-normalized decode efficiency of the engine
described in this paper, with prefill at parity.
<!-- receipt: docs/BENCHMARKS.md 2026-07-30 ledger entry (llama.cpp master 5f55650a7, PR #25707; ~352 vs ~474 GiB/s effective weight stream; pp512 12,199 vs 12,275; contention disclosed, quiet-box rerun owed) -->
This is progress we welcome and had no hand in, and it changes what a paper in
this area can durably contribute: the one-plane container and the fast kernel
are becoming community property, so whatever is worth writing down must live
above them.

What has not kept pace with the artifacts is the claims made about them. Three
failure modes recur. **First, nominal-rate accounting.** The information
content of one trit, $\log_2 3 \approx 1.585$ bits, is a lower bound on
storage, not an achieved rate; yet "1.58-bit" appears routinely as a label on
artifacts whose physical storage is roughly 2.1 bits per weight,
<!-- receipt: docs/research-ternary-sota-mid2026.md §2.1 ("1.58-bit" marketing = ~2.1 bpw storage at Q2_0 g128) -->
and two independently stored ternary planes announced under the same label cost
4.25 physical bits per weight at 128-coefficient scale groups once their scales
are counted [REF:ptqtp].
<!-- receipt: docs/adr/0028-salt-v2-additive-ternarization.md §2 ("PTQTP-style two-plane storage is 4.25 bpw at G128") and §Alternatives ("Call two planes '1.58-bit'": 3.1699-bit information floor for two trits) -->
Conflating the bound with the rate makes every cross-method comparison silently
favorable to whoever conflates hardest. **Second, headline claims that do not
survive scrutiny.** In a systematic adversarial-verification pass we ran over
the mid-2026 ternary landscape, several prominent headline claims did not
survive independent verification — among them a flagship release's
benchmark-superiority table [REF:refuted-quality-table], the headline
comparison of a peer-reviewed post-training method
[REF:refuted-ptq-comparison], and every load-bearing performance claim of a
circulated sparse-ternary kernel [REF:refuted-kernel-claims].
<!-- receipt: docs/research-ternary-sota-mid2026.md (3-vote adversarial verification key; refutations documented in §1.1, §2.1, §3) -->
We deliberately name no vendors in this prose; the point is not any individual
actor but the base rate, which is high enough that an unverified number in this
space should be treated as marketing until reproduced. **Third, numbers without
provenance.** Benchmark figures are routinely published without the command
that produced them, the environment they ran in, or the contention on the box —
which makes them irreproducible in the strict sense that no third party can
even attempt the reproduction. And spanning all three: no published system we
are aware of ships quantization, recovery training, and serving as one
lifecycle whose every stage is auditable from committed artifacts; mainstream
serving stacks, for their part, carried no ternary support at all — in the
most prominent case through fourteen months of open requests.
<!-- receipt: docs/research-ternary-sota-mid2026.md §5 (serving-stack table; vLLM PR closed unmerged Nov 2025, Jan 2026 request unanswered) -->

This paper is a response to that situation, in the form of a method plus the
evidence apparatus the method is embedded in. The method is SALT —
sensitivity-allocated layered ternarization: a weight is represented as a sum
of at most three ternary planes with non-negative per-group scales,
$\hat{W} = \sum_p s_p T_p$, zero-point-free, with the number of planes
allocated per group by measured curvature rather than uniformly. Around the
representation sit four layers of apparatus: an exact, deterministic joint
fitter whose every acceptance is a measured strict improvement; a curvature
capture whose records are content-addressed so that operationally different
captures either agree byte-for-byte or refuse to combine; a distillation
recovery recipe that repairs what post-training ternarization destroys; and a
measurement ledger in which every published number regenerates from a committed
command with its environment and contention recorded beside it.
<!-- receipt: docs/BENCHMARKS.md header ("Every number in this file reproduces from one command"; ADR 0026 Track R) -->
The quiet thesis, argued by construction throughout, is that the field's
comparison-hygiene problem is fixable *mechanically*: exact byte accounting,
content-addressed evidence, and command-regenerable tables are engineering
artifacts, not virtues, and they are cheap once the formats are designed to
carry them. All of it ships as the Tritium repository (Apache-2.0), which is
the companion artifact to this paper rather than a supplement to it.

## Contributions

**C1 — The SALT representation** (Section 2). Additive ternary planes
$\hat{W} = \sum_p s_p T_p$ with non-negative f16 scales per 128-coefficient
group and at most three planes allocated per 256-coefficient macrotile;
zero-point-free by design so the kernel contract stays pure
add/subtract/skip. One content-addressed semantic tensor admits three physical
codecs (D2, B3, S34) whose rates are reported as exact bytes, and the base
plane remains exportable to the mainstream Q2_0 container.

**C2 — Exact joint fitting** (Section 3). An E/M fitter with exact $3^P$
assignment under separable metrics, a closed-form conditioned scale solve, and
acceptance only on strict measured improvement — bitwise deterministic, matched
against brute-force oracles on tractable instances. Deterministic multi-start
includes softened-relay initialization basins that are never worse by
construction; their measured effect is stated plainly: at $P{=}3$ they improve
7.4% of sampled groups, and at $P \le 2$ they add essentially nothing.
<!-- receipt: docs/receipts-ws-b-relay-basin-ab.txt (768 groups per plane setting, SmolLM2-1.7B, identity metric) -->

**C3 — Kronecker curvature evidence** (Section 4). Input-Gram × output-Fisher
curvature records with content-addressed identity, accumulated by a canonical
dyadic reduction that provably excludes capture topology — batch boundaries,
resume points, sharding — from the evidence identity. Shared-forward capture is
measured at 1.7B: a 3.0× replay reduction with byte-identical published
records, an honest 1.17× wall-clock speedup, and the Amdahl decomposition that
explains the gap stated as such.
<!-- receipt: docs/receipts-ws-a1-cost-baseline-17b.json (replay_reduction=3.0, byte_identity=true, speedup_wall=1.173) -->

**C4 — Distillation recovery** (Section 5). Post-training ternarization of a
135M-parameter model is catastrophic — held-out WikiText-2 perplexity rises
from 23.8 to $3.28 \times 10^6$ — and SALT-aware straight-through distillation
recovers it to 139.6, a 23,493× recovery landing at 5.9× the fp teacher, with
the honest finding that the curve is token-limited rather than floored: nothing
in the data has plateaued, so we claim neither convergence to fp nor a ternary
floor.
<!-- receipt: docs/adr/0029-training-throughput-tensor-cores.md:168-192 -->
The same section reports two negative results (ternary KV cache, a relaxed
rmsnorm kernel) rejected by the same acceptance machinery that admits the
headline number.

**C5 — Physical-byte accounting as a first-class invariant** (Sections 2.5
and 6). Every artifact reports exact serialized and resident bytes; logical
bits-per-weight is banned from claims and may be printed only beside the
physical figures. The ban is enforced mechanically: encoder byte counters are
tested to equal actual file lengths and steady-state device allocations, so a
reported rate is an observable, not an estimate.
<!-- receipt: docs/adr/0028-salt-v2-additive-ternarization.md preregistered correctness gates ("byte counters equal actual file lengths and steady-state device allocations") -->

**C6 — Execution without dense weights** (Section 6). A native engine executes
packed planes directly, with no dense dequantized shadow — an audited quantity,
not an implementation hope. On a consumer RTX 4090 the dated ledger entries
record single-stream decode medians between roughly 273 and 303 tok/s on a 2.4B
ternary model — an effective weight stream of ~467–517 GiB/s, at
bandwidth-efficiency parity with a mature mainstream 4-bit CUDA path
(llama.cpp Q4_K_M) — and 12.3K tok/s
pp512 prefill through an IMMA path gated bit-identical to its dp4a
predecessor.
<!-- receipt: docs/BENCHMARKS.md ledger entries 2026-07-30 (median ~277, ~474 GiB/s, contended), quiet-box re-baseline 273.2-275.9 (~467 GiB/s = 1.71 GiB x ~273), and 2026-07-18 (pp512 12,274.7, bit-identity gates); docs/research-ternary-sota-mid2026.md §7 (301.4-302.8, ~517 GiB/s, vs llama.cpp Q4_K_M ~508 GiB/s) -->

## What this paper does not claim

The claims above are deliberately bounded, and the boundary is part of the
method. We make **no state-of-the-art quality claim**: the method evidence in this
paper spans 135M to 1.7B parameters, and downstream-benchmark evaluation of
SALT artifacts against the verified post-training baselines is open work, not
a result we hold. We report **no results at 27B scale**: a campaign at that
scale exists, but as a preregistration — frozen rate points, coverage policy,
promotion gates, and forecast rules committed before any fitting run — and it
is unrun. We regard stating this as a feature rather than a concession: the
preregistration subjects our own future claims to exactly the discipline this
paper asks of the field, and its results will be reported against the frozen
gates whatever they show.
<!-- receipt: docs/plans/0043-salt-v2-sota-campaign.md (frozen campaign structure, rate points, gates) -->
Finally, we advance **no byte-optimality thesis** — the conjecture that
ternary representations dominate floating-point ones per physical byte at
matched held-out quality is the subject of a separate planned paper and
requires multi-scale held-out curves this paper does not contain. What this
paper claims is narrower and, we believe, more durable: a representation, an
exact solver, curvature machinery, a recovery recipe, and a measurement
discipline under which every number in it can be regenerated, and under which
two of our own optimizations were rejected and one of our own published
interpretations corrected (Section 5).

## Organization

Section 2 defines the SALT representation, its physical codecs, and the
exact-byte accounting rules, including interoperability with the emerging
2-bit standard. Section 3 presents the joint fitter: exact assignment,
conditioned scale solves, monotone acceptance, and deterministic multi-start
with the relay basins' measured effect. Section 4 develops the Kronecker
curvature evidence — the identity design that makes capture topology a free
variable — and the measured cost of shared-forward capture at 1.7B. Section 5
reports the recovery arc: catastrophic PTQ, distillation recovery, the
learning-rate correction, the token-limited finding, and the negative results.
Section 6 describes the execution engine and the benchmark-ledger method,
including the same-box comparison against the day-one mainstream ternary CUDA
path. Section 7 positions SALT against the QAT, PTQ, and theoretical
literature; Section 8 states limitations, which we treat as load-bearing;
Section 9 is the reproducibility statement.
