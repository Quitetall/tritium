# 4. Curvature evidence and shared-forward capture

The joint fitter of Section 3 minimises a metric-weighted quadratic per weight
group. This section describes where that metric comes from, why its identity
design is the load-bearing idea, and what a shared-forward capture optimisation
costs and saves when measured rather than forecast.

## 4.1 Kronecker-factored group metrics

The information-theoretic frame for quantised matrix multiplication holds that
the quantity to preserve is the *product* $Wx$, not the weights themselves:
weight-space MSE is the wrong distortion measure, and output-aware objectives
are the principled ones [REF:op2026]. For a linear layer $y = Wx$, a
second-order expansion of any twice-differentiable downstream loss in a weight
perturbation $\Delta W$ gives, under the standard K-FAC independence
approximation [REF:kfac2015], a per-output-row curvature that factorises as

$$
H_r \;\approx\; f_r \cdot \mathbb{E}\!\left[x x^\top\right],
\qquad
f_r \;=\; \mathbb{E}\!\left[\left(\partial \mathcal{L}/\partial y_r\right)^2\right],
$$

i.e. an input Gram matrix shared across rows, scaled by one empirical-Fisher
scalar per output row. This factorisation is what makes curvature-aware
ternarisation affordable at all: SALT fits weights in groups of 128 along the
input dimension (G128, Section 2), so only the group-diagonal $128 \times 128$
blocks $G_g$ of the input Gram are ever consulted. The metric supplied to the
fitter for row $r$, group $g$ is

$$
M_{r,g} \;=\; f_r\, G_g \;+\; \lambda I ,
$$

with damping $\lambda \ge 0$; $M_{r,g}$ is the matrix supplied to Section 3's
fitter as its metric $H$. A tensor's complete curvature evidence is
therefore its group-diagonal Gram blocks plus one $f_r$ per row — about
2.11 MB for a $2048 \times 2048$ projection in the S2KF record format
<!-- receipt: docs/receipts-ws-a1-cost-baseline-17b.json (artifact_bytes 50,732,380 / 24 tensors); format constants in crates/tritium-quantize/src/salt_v2_evidence.rs (GROUP_SIZE=128, GROUP_PAYLOAD_BYTES) -->
— rather than anything resembling a dense per-row Hessian.

One contract is worth stating because the field's habit is the opposite: the
damped matrix $M_{r,g}$ must pass an explicit dense positive-semidefinite
validation, and failure is an error, not a silent fallback to a diagonal
approximation
<!-- receipt: crates/tritium-quantize/src/salt_v2_curvature.rs (build_kfac_metric, CurvatureError::InvalidKfacMetric); invariant restated in docs/plans/0054-frontier-methods-integration.md §A2 -->.
A metric that cannot be certified PSD stops the pipeline; it does not quietly
degrade the objective.

## 4.2 Three curvature objectives

The capture layer supports three objectives, each pinned by a frozen
`objective_id` string that is hashed into the evidence identity
(§4.3), so records produced under different objectives can never be
conflated
<!-- receipt: crates/tritium-py/python/tritium/torch/ptq.py (objective strings); crates/tritium-quantize/src/salt_v2_evidence.rs (SaltV2Curvature variants) -->:

1. **`tritium.input-gram@1`** — activation-only: $G_g$ accumulated from the
   forward pass, $f_r \equiv 1$, no autograd. This is the layer-wise proxy of
   the GPTQ family [REF:gptq2022], the cheapest and least output-aware option.
2. **`tritium.model-loss-guided-fisher.{sum, mean-attention-mask,
   mean-valid-causal-labels}@1`** — the empirical Fisher of the model's own
   training loss. Gradients are requested only with respect to the selected
   module outputs via vector-Jacobian products, so parameter `.grad` buffers
   are never allocated. The loss reduction is part of the identifier because
   it changes the numbers; leaving it implicit would be a reproducibility hole.
3. **`tritium.softmax-fisher-rademacher.single-probe@1`** — the forward-KL
   (softmax-Fisher) objective via a single Rademacher probe. Writing
   $p = \mathrm{softmax}(z)$ for the model's output distribution, the matrix
   $S = \mathrm{diag}(\sqrt{p}) - p\,\sqrt{p}^{\top}$ satisfies
   $S S^\top = \mathrm{diag}(p) - p p^\top$, the Fisher of the softmax in
   logit space. The capture draws one sign vector
   $\varepsilon \in \{\pm 1\}^{V}$ and back-propagates the factor

   $$
   g_i \;=\; \varepsilon_i \sqrt{p_i} \;-\; p_i \sum_{j} \varepsilon_j \sqrt{p_j},
   \qquad
   \mathbb{E}_{\varepsilon}\!\left[g g^\top\right] = \mathrm{diag}(p) - p p^\top,
   $$

   a Hutchinson-style single-probe estimator [REF:hutchinson1990]. The signs
   are not sampled from mutable RNG state: they are a stateless hash of the
   global sample ordinal and vocabulary index
   <!-- receipt: crates/tritium-py/python/tritium/torch/ptq.py (_forward_kl_factors) -->,
   so the probe is bit-replayable and introduces no hidden state into the
   evidence identity.

## 4.3 Evidence identity: capture topology provably outside it

The design decision this section exists to argue for is a partition of the
capture process into what *is* identity and what *provably is not*.

Every evidence stream is bound to a `CurvatureSourceId`: three 32-byte content
digests naming the source checkpoint, the exact activation-cache contents, and
the ordered calibration token stream; an all-zero component is rejected as
missing provenance rather than accepted as a default
<!-- receipt: crates/tritium-quantize/src/salt_v2_curvature.rs (CurvatureSourceId::new) -->.
An input Gram may be paired with an output Fisher only when all three digests,
the per-sample selection traces, sample counts, and total weights agree
bit-for-bit; anything less is a hard error
<!-- receipt: crates/tritium-quantize/src/salt_v2_curvature.rs (build_kfac_metric: SourceMismatch, SelectionMismatch) -->.

The subtler half concerns arithmetic. Each calibration sample is bound to a
*global sample ordinal*, and accumulation proceeds by a canonical dyadic
reduction: per-sample leaves merge only as aligned power-of-two sibling
intervals, so the `f64` summation tree is a function of the ordinals alone
<!-- receipt: crates/tritium-quantize/src/salt_v2_curvature.rs (canonical_siblings, push_input_segment); bounded to 64 retained segments per accumulator, crates/tritium-quantize/src/salt_v2_evidence.rs (MAX_KRONECKER_REDUCTION_SEGMENTS) -->.
Floating-point addition is not associative, so a naive running sum would leak
the batch partition into the low-order bits of the result — an accidental
side channel that would make any two operationally different captures
byte-distinct. The dyadic tree closes that channel by construction: API batch
boundaries, resume points after interruption, and shard processing order can
affect neither the accumulated values nor the identity digest, while any change
to sample content, ordering, weights, masks, or source provenance necessarily
does. Overlapping, gapped, or non-canonically aligned sample ranges are
rejected at merge time rather than silently absorbed.

The payoff is that *capture topology becomes a free variable*. Any
orchestration — per-tensor replay, resumable sessions, sharded workers, or the
shared-forward grouping below — must produce byte-identical records for the
same frozen inputs, as a consequence of the design rather than as an
aspiration. An equivalence golden enforces on disk what the reduction
guarantees in principle
<!-- receipt: docs/plans/0054-frontier-methods-integration.md §A3; tests beside crates/tritium-quantize/tests/kronecker_evidence_producer.rs -->.

## 4.4 Shared-forward capture

The naive capture cost model is one full calibration replay per tensor — 506
replays for the preregistered 27B catalogue
<!-- receipt: docs/plans/0043-salt-v2-sota-campaign.md (506-matrix coverage policy) -->.
But within a transformer block, up to seven linear projections consume roughly
three distinct input streams: q/k/v share the attention-norm output, and
gate/up share the FFN-norm output
<!-- receipt: docs/plans/0054-frontier-methods-integration.md §A2 -->.
One forward pass can therefore feed the evidence builders of every projection
on the same stream.

The implementation fans one validated activation batch into $N$ per-tensor
builders (`SharedForwardGroupCore`), with two properties worth recording.
First, all contract checks — shapes, finiteness, weights, masks — run in a
preflight before any member mutates, so contract violations reject atomically
for the whole group. Second, a resource failure after the first member has
advanced *poisons* the group terminally: member ordinals that may have diverged
can never be published
<!-- receipt: crates/tritium-py/src/kronecker.rs (SharedForwardGroupCore::preflight/append, SharedForwardAppendError::Poisoned) -->.
For gradient-requiring objectives, the output factors for all $N$ member
outputs are obtained in a single `torch.autograd.grad` call against the shared
forward graph; the Python driver registers a geometry-preflight pre-hook and a
capture hook on each target module and drives one forward pass per calibration
batch, instead of one full-calibration replay per tensor
<!-- receipt: crates/tritium-py/python/tritium/torch/ptq.py (capture_kronecker_module_group: one torch.autograd.grad over flat_outputs; per-module register_forward_pre_hook/register_forward_hook). Corrected from plan 0054 §A2's design sketch ("one forward hook per input stream", "N VJP calls against a retained graph"): the shipped driver hooks each module and batches the VJPs into one autograd.grad call -->.

## 4.5 Measured cost at 1.7B

We ran the same frozen calibration inputs through both paths on
SmolLM2-1.7B (RTX 4090): 24 attention projections (8 layers $\times$ q/k/v,
each $2048 \times 2048$), $64 \times 512 = 32{,}768$ calibration tokens,
input-Gram objective, f32
<!-- receipt: docs/receipts-ws-a1-cost-baseline-17b.json; harness tools/ws_a1_cost_baseline.py (dtype=torch.float32); the GPU model (RTX 4090) is recorded in docs/plans/0054-frontier-methods-integration.md §A1/A4 measured-results header, not in the JSON receipt -->.

| capture path | replays | wall (s) | peak device bytes | artifact bytes |
|---|---:|---:|---:|---:|
| per-tensor | 24 | 605.3 | 11.85 GB | 50,732,380 |
| shared-forward (per-layer groups) | 8 | 516.0 | 12.06 GB | 50,732,380 |

<!-- receipt: docs/receipts-ws-a1-cost-baseline-17b.json (per_tensor.wall_s=605.316, shared_forward.wall_s=515.969, peak_bytes, artifact_bytes, replay_reduction=3.0, speedup_wall=1.173, byte_identity=true) -->

The replay reduction is exactly $3.0\times$ (q/k/v per attention stream) and
the published records are byte-identical across the two paths — the §4.3
guarantee holding on a real model, not a toy. The wall-clock speedup, however,
is only $1.17\times$, and the honest decomposition matters more than the
headline. Writing $t_f$ for forward cost per replay and $t_a$ for the
replay-invariant Gram-accumulation-and-writer cost per tensor, the two
measured walls give two equations,
$24\,(t_f + t_a) = 605.3$ and $8\,t_f + 24\,t_a = 516.0$, hence
$t_f \approx 5.6$ s: the forward is only $\sim$22% of per-tensor wall at this
scale, and the remaining $\sim$78% does not shrink with grouping
<!-- receipt: derived from docs/receipts-ws-a1-cost-baseline-17b.json wall times; decomposition stated in docs/plans/0054-frontier-methods-integration.md §A1/A4 -->.
Shared-forward capture eliminates the replay-*scaling* term; the accumulation
term is what any larger-scale cost forecast must price.

**Scaling argument, stated as assumption.** At 27B the forward is far more
expensive per replay, while Gram accumulation per tensor grows roughly linearly
in columns; the forward's share of wall should therefore rise with model size,
making the $3\times$ replay reduction worth strictly more than the $1.17\times$
measured here. This is an assumption, not a measurement, and per the
preregistration rules a same-harness run on a sliced 27B layer set must replace
it before it informs any spend decision
<!-- receipt: docs/plans/0054-frontier-methods-integration.md §A4 (forecast rules per plan 0043) -->.

**A fail-closed finding.** An all-24-tensors-in-one capture group exceeds the
bounded-snapshot budget — it requires 302 MB of pending activation snapshot
against the 256 MiB `max_capture_bytes` default — and fails closed rather than
spilling
<!-- receipt: docs/plans/0054-frontier-methods-integration.md §A1/A4; default in crates/tritium-py/src/kronecker.rs (DEFAULT_MAX_BATCH_BYTES = 256 MiB) -->.
We report this not as an inconvenience but as empirical confirmation that the
grouping planner's residency budget is load-bearing: the production shape is
per-layer input-stream groups, and the budget is what keeps host memory bounded
when it is tempting to group more aggressively.

The cost table above covers attention projections at 1.7B only; FFN streams
and larger geometries inherit the same identity guarantees but not, yet, a
measured cost row.
