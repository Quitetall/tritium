# 3. Fitting: exact joint solve and initialization basins

Given the representation of Section 2, fitting is a per-group problem. For a
group of $d$ weights $w \in \mathbb{R}^d$ (here $d = 128$), $P \le 3$ planes,
and a supplied curvature metric $H \succeq 0$, the fitter solves

$$
\min_{\substack{s \in \mathbb{R}^P_{\ge 0} \\ T_p \in \{-1,0,+1\}^d}}
\; \Big(w - \textstyle\sum_{p=1}^{P} s_p T_p\Big)^{\!\top} H
\Big(w - \textstyle\sum_{p=1}^{P} s_p T_p\Big),
$$

where $H$ is the identity, a non-negative diagonal, or a validated dense
symmetric PSD matrix (the per-group K-FAC block of Section 4
[REF:kfac]). <!-- receipt: crates/tritium-quantize/src/salt_v2.rs (fit_joint_ternary, JointFitMetric, DensePsdMetric) -->
This is a mixed-integer non-convex problem; we do not claim to solve it
globally. What we do claim — and test — is that the assignment subproblem is
solved exactly under separable metrics, that the scale subproblem is solved in
closed form through a conditioned system, that the alternation is a descent
method by construction (every proposal is re-scored and accepted only on
strict improvement), and that the entire procedure is bitwise deterministic,
so every emitted artifact is reproducible from its inputs alone. Additive multi-codebook
fitting is well studied [REF:aqlm]; the constraint that distinguishes this
solver is that the codebook is fixed to $\{-1,0,+1\}$ scaled by non-negative
per-plane scalars, with no zero point and no stored transform — the exact
representation the execution engine consumes.

## 3.1 E step: exact $3^P$ assignment

For fixed scales the objective under a separable (identity or diagonal) metric
decomposes coordinate-wise, and each coordinate ranges over only $3^P \le 27$
additive states. The E step enumerates all of them per weight and takes the
exact minimizer; ties prefer $0$, then $-1$, then $+1$ in plane order, which
deterministically canonicalizes toward the sparse (skippable) state.
<!-- receipt: crates/tritium-quantize/src/salt_v2.rs (exact_ternary_assignment) -->
A golden test checks this against an independent oracle that enumerates all
$3^{Pd}$ complete ternary matrices on tractable sizes; the resulting errors are
bit-identical.
<!-- receipt: salt_v2.rs test exact_assignment_matches_global_exhaustive_oracle -->

Under a dense metric the coordinates couple and per-coordinate enumeration is
no longer globally exact. The fitter then uses the separable assignment as an
initial point and runs deterministic cyclic coordinate descent in which each
coordinate update is still the exact $3^P$ minimizer with all other
coordinates fixed; the dense quadratic is updated analytically after each
accepted move, and sweeps stop at quiescence or a fixed cap of eight. Each
coordinate move strictly decreases the dense objective, so the dense E step is
monotone even though it is no longer provably optimal.
<!-- receipt: salt_v2.rs (assignment_for_metric, dense coordinate-descent branch) -->

## 3.2 M step: conditioned scale solve

For fixed trits, collect the planes as $G = [T_1 \cdots T_P] \in
\{-1,0,+1\}^{d \times P}$. The scale subproblem is a $P$-dimensional convex
quadratic solved through the weighted normal equations

$$ (G^\top H G + \lambda I)\, s = G^\top H w, \qquad P \le 3 . $$

The accumulated normal matrix is symmetrized deterministically, its extremal
eigenvalues are computed by classical (largest-off-diagonal-pivot) Jacobi
rotations, and the spectral condition
number is measured before regularization. If it exceeds the configured limit
$\kappa_{\lim}$, the ridge is raised in closed form to
$\lambda = \max\!\big(\lambda_0,\, (\mu_{\max} - \kappa_{\lim}\mu_{\min}) /
(\kappa_{\lim} - 1)\big)$, which caps the post-ridge condition number at the
limit. Every attempted solve records its before/after condition numbers, the
ridge actually used, and whether it was adaptively increased.
<!-- receipt: salt_v2.rs (solve_scales, ScaleSolveTelemetry); test singular_scale_system_uses_reported_adaptive_ridge -->

Two canonicalizations follow. Plane sign is a representation symmetry
($s_p T_p = (-s_p)(-T_p)$), so any negative solved coefficient is flipped into
its trit plane, restoring $s \ge 0$; planes are then sorted by descending
scale with an index-stable tie-break. A round-trip test confirms
canonicalization preserves the reconstruction exactly.
<!-- receipt: salt_v2.rs test scale_sign_and_plane_order_canonicalization_preserve_reconstruction -->
When f16 scoring is selected, every fitted scale is rounded through the
deployment representation *before* scoring, so acceptance decisions are made at
the precision that will actually ship rather than on an idealized f32 fit.
<!-- receipt: salt_v2.rs (ScalePrecision, deployment_scale); test f16_scoring_returns_deployment_representable_scales -->

## 3.3 Monotone acceptance

Neither subproblem's proposal is trusted. The ridge biases the scale solve,
f16 rounding perturbs it, and the dense E step is heuristic past its first
move; any of these can propose a candidate that is worse under the true
metric. The alternation therefore re-scores every proposal under the
*unregularized* objective and accepts it only on strict improvement. This makes
the loop a descent method by construction, independent of solver internals.
The iteration terminates at a configured cap (default 16) or at the first full
iteration in which neither phase improves. Every accepted update is recorded
with its before/after objectives, and a property test asserts the accepted
objective sequence is strictly decreasing and gap-free.
<!-- receipt: salt_v2.rs (optimize_start); test every_accepted_e_and_m_update_is_strictly_monotone -->

## 3.4 Determinism and multi-start

The fitter contains no randomness: no RNG, no time-dependent state, no
parallel reduction whose order could vary. Initial scales come from a family
of deterministic basins: restart 0 performs a residual absmean recursion
(fit a plane at the metric-weighted mean absolute residual, subtract the
nearest-trit reconstruction, repeat); restarts $k \ge 1$ anchor at the
metric-weighted absolute-value quantile $0.5 + 0.45\,k/R$ with per-plane dyadic
division and a $\pm 12.5\%$ modulation; and for $P = 2$ one restart is reserved
for a max-minus-min decomposition that exactly represents groups such as
$[\varepsilon, -a, -a]$, where residual-mean starts leave a dead second plane.
<!-- receipt: salt_v2.rs (deterministic_initial_scales); test default_p2_uses_both_planes_for_exact_difference_solution -->
A further basin embeds the best $(P{-}1)$-plane fit with a zero final plane,
guaranteeing that the fitted objective is monotone non-increasing in $P$
without pretending the non-convex solver is globally optimal.
<!-- receipt: salt_v2.rs (LowerPlaneFallback branch of fit_joint_ternary) -->
All basins — the output-aware E/M (OA-EM) restarts, the relay basins below,
and the fallback — are optimized independently and the minimum selected under
a total order on f64 objectives; the complete per-basin optimization evidence
is retained in the result. Bitwise repeatability is tested directly: two runs
must produce equal result structures, receipts included.
<!-- receipt: salt_v2.rs tests fitting_is_bitwise_deterministic, oa_em_evaluates_every_configured_restart_for_p1_through_p3 -->

On tractable instances the whole pipeline is checked against a joint
brute-force oracle over a scale grid crossed with all complete trit matrices;
the fitted objective matches within $10^{-12}$, and $P{=}2$ fits dominate a
greedy residual baseline on deterministic groups.
<!-- receipt: salt_v2.rs tests tiny_joint_fit_matches_full_trit_and_scale_grid_oracle, accepted_iterations_are_monotone_and_p2_dominates_baselines -->

## 3.5 Softened-relay initialization basins

CAT-Q [REF:catq] ternarizes through a smooth two-sided relay,

$$ f(v; \sigma, \Delta) = \frac{\tanh\!\big(\sigma (v - \Delta)\big) +
\tanh\!\big(\sigma (v + \Delta)\big)}{2 \tanh \sigma}, $$

which is odd, vanishes at zero, is bounded by $|f| \le 1$ on $|v| \le 1$ for
$\Delta \in [0, 1]$, and
approaches the hard ternary indicator with threshold $\Delta$ as
$\sigma \to \infty$ — all four properties are held by property tests rather
than assumed.
<!-- receipt: salt_v2.rs (relay module); tests two_sided_relay_is_odd_zero_at_zero_and_bounded, two_sided_relay_sharp_limit_matches_hard_ternary_off_threshold -->
We adopt the relay not as a representation but as two extra deterministic
initialization basins. Each soft-fits sequential residual planes in
absmean-normalized units by twelve analytic gradient steps of fixed size, with
sharpness annealed from $\sigma_0 = 30$, doubling every four steps, and all
parameters clamped to fixed bounds. The *softened* basin descends the scale
only at fixed normalized threshold $\Delta = 0.5$; the *modulated* basin also
descends the threshold and a mean shift $\mu$. After each plane the hard
projection of the soft fit is subtracted from the residual.
<!-- receipt: salt_v2.rs (relay::basin_scales, relay::descend); commit d34fceb -->

The design constraint is that $\Delta$ and $\mu$ are basin-internal: they
shape which scale magnitudes emerge and are neither stored nor returned. The
basin donates only a scale vector, which is then projected through the exact
E/M solver of §3.1–3.3; the emitted representation remains pure
scales-and-trits, so the zero-point ban of the format (ADR 0028) holds by
construction rather than by audit. The basins are appended after the OA-EM
restarts, leaving restart indices and receipts byte-identical when disabled,
and — because extra accept-only-if-improves basins can only widen the
minimized start set — the fitted objective with basins on is never worse than
with basins off. This never-worse property is asserted per group in both a
property test and the measurement harness itself.
<!-- receipt: salt_v2.rs test relay_basins_never_worsen_the_final_objective; crates/tritium-quantize/examples/relay_basin_ab.rs (per-group assert) -->

## 3.6 Measured effect, stated plainly

An A/B harness samples G128 groups from projection and MLP tensors of a real
checkpoint and fits each group with basins off and on, identical configuration
otherwise.
<!-- receipt: crates/tritium-quantize/examples/relay_basin_ab.rs -->
On SmolLM2-1.7B (12 tensors $\times$ 64 groups $=$ 768 groups per plane
setting, identity metric) the result is:
<!-- receipt: docs/receipts-ws-b-relay-basin-ab.txt message (relay-basin A/B harness, plan 0054 WS-B evidence) -->

| $P$ | groups improved | win rate | median rel. improvement (wins) | max |
|-----|-----------------|----------|-------------------------------|-----|
| 1   | 0 / 768         | 0.0%     | —                             | —   |
| 2   | 2 / 768         | 0.3%     | —                             | —   |
| 3   | 57 / 768        | 7.4%     | 3.1%                          | 32% |

<!-- receipt: docs/receipts-ws-b-relay-basin-ab.txt message; P=2 per-win statistics not reported (n=2) -->

At $P = 3$ the basins add $1.58\times$ fitter wall-clock overhead.
<!-- receipt: docs/receipts-ws-b-relay-basin-ab.txt (checkpoint-pinned rerun, bit-identical to the 04ef528 first run) -->

The pattern is informative. At $P = 1$ the deterministic default starts
already reach the selected minimum on every sampled group; at $P = 2$ the
reserved max-minus-min basin appears to cover most of what a relay start could
add. Only the $P = 3$ landscape is rough enough for the extra basins to
matter, and there they act on 7.4% of groups with a modest median gain and an
occasional large one. These are group-level objective improvements under an
identity metric on one checkpoint — not a quality claim. Whether they
translate into any model-level effect is exactly what the preregistered
Stage-7 successive-halving bracket (plan 0043, thresholds unchanged) is for;
the harness reports, it does not decide.
<!-- receipt: docs/plans/0054-frontier-methods-integration.md (Workstream B, Gates B) -->
Because the basins are never worse by construction and default-off, the safe
deployment posture costs nothing: they can be enabled per-recipe, the recipe
digest records the toggles, and the fitted artifact carries the full per-basin
receipt trail either way.
<!-- receipt: commit d34fceb message (recipe digest extension, RECIPE_HASH_CONTEXT v1->v2) -->
