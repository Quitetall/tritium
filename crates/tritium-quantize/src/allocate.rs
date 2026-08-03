//! Rate-distortion plane allocation (ADR 0001 §4) — the corrected "multiplier".
//!
//! Mode-finding sets *where* levels sit; it does **not** set *how many* planes a
//! group earns. That is a sensitivity + rate-distortion question: spend the
//! bits-per-weight budget where the model is most sensitive. Given each group's
//! loss sensitivity `H_g` (Hessian-diagonal / Fisher, reused from GPTQ) and its
//! per-plane reconstruction-error curve `err_g(T)`, solve
//!
//! ```text
//! minimize  Σ_g H_g · err_g(T_g)   subject to   Σ_g |g| · log2(3) · T_g ≤ Budget
//! ```
//!
//! by greedy water-filling: repeatedly hand the next plane to whichever group
//! buys the largest loss-drop-per-bit `H_g · [err_g(T) − err_g(T+1)] / (|g|·log2 3)`,
//! until the budget is exhausted or every group reaches `T_max`. Low-sensitivity
//! groups settle at `T_min`; the budget floor (`Budget = base`) leaves every group
//! at `T = 1` — flat AbsMean, the BitNet special case.

use core::fmt;

use crate::{PlaneStack, recon_error, residual_expand};

/// Information content of one ternary digit, in bits: `log2(3)`. The ADR writes
/// this `1.585`; we carry full precision so the budget accounting is exact.
pub const TRIT_BITS: f64 = 1.584_962_500_721_156_2; // log2(3)

/// Relative tolerance on the base-feasibility check, absorbing the float-rounding
/// gap between a per-group `base` sum and a one-multiply bpw-derived budget so the
/// exact floor (`budget == base`) stays feasible. Additions never use this slack.
const BUDGET_FEASIBILITY_SLACK_REL: f64 = 1e-9;

/// One weight group presented to the allocator.
#[derive(Clone, Copy, Debug)]
pub struct GroupInput<'a> {
    /// The group's fp weights (a kernel tile: an output channel or 128-block).
    pub weights: &'a [f32],
    /// Loss sensitivity `H_g` — larger means more loss-critical, so more planes.
    /// Must be finite and `≥ 0`.
    pub sensitivity: f64,
}

/// One group presented with a **precomputed** error curve, for callers whose fitter is not the
/// greedy residual expansion [`GroupInput`] assumes.
///
/// This exists because the allocation rule and the fitter are independent choices, and hardwiring
/// [`error_curve`] to [`residual_expand`] conflated them. The balanced-ternary ladder, for one,
/// produces a materially different `err(T)` — measured at ~9.5 dB per plane against the greedy
/// expansion's ~2.8 — so allocating from greedy curves would spend the budget as if planes bought
/// far less than they do.
#[derive(Clone, Copy, Debug)]
pub struct GroupCurve<'a> {
    /// `err(0..=T)`: reconstruction error after each plane count, index `0` meaning no planes.
    /// Must be non-empty; values are used as given (monotonicity is the caller's contract, and a
    /// non-decreasing step simply reads as "no gain" and is skipped).
    pub curve: &'a [f64],
    /// Number of weights in the group — sets the bit cost of one plane.
    pub weights: usize,
    /// Loss sensitivity `H_g`, as in [`GroupInput`].
    pub sensitivity: f64,
}

/// Allocation knobs.
#[derive(Clone, Copy, Debug)]
pub struct AllocConfig {
    /// Planes every group gets unconditionally (the dense base). Default `1` —
    /// one base plane everywhere (ADR 0001 hardware constraint). `0` permits
    /// pruning a tile to nothing.
    pub t_min: usize,
    /// Hard cap on planes per group. Default `3` (tile-uniform `{1,2,3}`).
    pub t_max: usize,
    /// Global budget, in bits.
    pub budget_bits: f64,
}

impl AllocConfig {
    /// A budget expressed as a target **average bits-per-weight** over
    /// `total_weights`, the natural knob (`1.585` ≈ all-base … `~4.75` at `T=3`).
    pub fn from_bpw(target_bpw: f64, total_weights: usize, t_min: usize, t_max: usize) -> Self {
        AllocConfig {
            t_min,
            t_max,
            budget_bits: target_bpw * total_weights as f64,
        }
    }
}

impl Default for AllocConfig {
    fn default() -> Self {
        AllocConfig {
            t_min: 1,
            t_max: 3,
            budget_bits: f64::INFINITY,
        }
    }
}

/// Why an allocation could not be produced.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum AllocError {
    /// `t_min > t_max` — empty feasible range.
    BadRange {
        /// Requested minimum planes per group.
        t_min: usize,
        /// Requested maximum planes per group.
        t_max: usize,
    },
    /// The base allocation (`t_min` planes on every group) already exceeds the
    /// budget, so no feasible allocation exists.
    BudgetTooSmall {
        /// Bits used by the base allocation (`t_min` planes on every group).
        base_bits: f64,
        /// The bit budget that the base allocation already exceeds.
        budget_bits: f64,
    },
    /// Group `group`'s `sensitivity` is not finite-and-nonnegative — it would
    /// corrupt the greedy objective (a negative `H_g` spends budget to *raise*
    /// the weighted loss; `NaN` poisons the max-selection so no group is chosen).
    InvalidSensitivity {
        /// Index of the offending group.
        group: usize,
        /// The non-finite-or-negative sensitivity value that was rejected.
        value: f64,
    },
}

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AllocError::BadRange { t_min, t_max } => {
                write!(f, "t_min ({t_min}) exceeds t_max ({t_max})")
            }
            AllocError::BudgetTooSmall {
                base_bits,
                budget_bits,
            } => write!(
                f,
                "base allocation needs {base_bits:.1} bits but budget is {budget_bits:.1}"
            ),
            AllocError::InvalidSensitivity { group, value } => {
                write!(
                    f,
                    "group {group} has invalid sensitivity {value} (want finite, ≥ 0)"
                )
            }
        }
    }
}

impl std::error::Error for AllocError {}

/// The result: a realized plane count per group, index-aligned with the input.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Allocation {
    /// `plane_counts[g]` ∈ `[t_min, t_max]` is the number of planes group `g` got.
    pub plane_counts: Vec<usize>,
}

impl Allocation {
    /// Total stored bits: `Σ_g |g| · log2(3) · T_g`.
    pub fn total_bits(&self, group_sizes: &[usize]) -> f64 {
        self.plane_counts
            .iter()
            .zip(group_sizes)
            .map(|(&t, &s)| s as f64 * TRIT_BITS * t as f64)
            .sum()
    }

    /// Achieved average bits-per-weight over all groups (`0` if no weights).
    pub fn avg_bpw(&self, group_sizes: &[usize]) -> f64 {
        let total: usize = group_sizes.iter().sum();
        if total == 0 {
            return 0.0;
        }
        self.total_bits(group_sizes) / total as f64
    }
}

/// `err_g(0..=realized)` for one group: the sum-of-squared reconstruction error
/// after each plane. Index `0` is `‖w‖²` (no planes); monotonic non-increasing.
fn error_curve(w: &[f32], t_max: usize) -> Vec<f64> {
    let stack = residual_expand(w, t_max);
    let realized = stack.plane_count();
    (0..=realized)
        .map(|t| {
            let prefix = PlaneStack {
                planes: stack.planes[..t].to_vec(),
            };
            recon_error(w, &prefix)
        })
        .collect()
}

/// `err_g(t)`, clamped to the last realized plane (the residual collapsed to zero
/// past that point, so error is flat — adding planes buys nothing).
#[inline]
fn curve_at(curve: &[f64], t: usize) -> f64 {
    debug_assert!(!curve.is_empty(), "error_curve always yields err(0)");
    curve[t.min(curve.len() - 1)]
}

/// Allocate planes to groups by greedy rate-distortion water-filling.
///
/// Every group starts at `t_min`; the budget must cover that base or the call
/// fails. Then planes are handed out one at a time to the group with the highest
/// `H_g · Δerr_g / (|g|·log2 3)`, skipping groups that are at `T_max`, can't
/// afford another plane, or gain nothing (`Δerr = 0`). Ties break toward the
/// lower group index, so the allocation is deterministic.
pub fn allocate(groups: &[GroupInput], cfg: &AllocConfig) -> Result<Allocation, AllocError> {
    if cfg.t_min > cfg.t_max {
        return Err(AllocError::BadRange {
            t_min: cfg.t_min,
            t_max: cfg.t_max,
        });
    }
    for (g, grp) in groups.iter().enumerate() {
        if !(grp.sensitivity.is_finite() && grp.sensitivity >= 0.0) {
            return Err(AllocError::InvalidSensitivity {
                group: g,
                value: grp.sensitivity,
            });
        }
    }
    let curves: Vec<Vec<f64>> = groups
        .iter()
        .map(|g| error_curve(g.weights, cfg.t_max))
        .collect();
    let curved: Vec<GroupCurve<'_>> = groups
        .iter()
        .zip(&curves)
        .map(|(g, c)| GroupCurve {
            curve: c,
            weights: g.weights.len(),
            sensitivity: g.sensitivity,
        })
        .collect();
    allocate_with_curves(&curved, cfg)
}

/// [`allocate`], driven by precomputed error curves instead of the greedy residual expansion.
///
/// This is the actual water-filling; [`allocate`] builds greedy curves and delegates here, so both
/// entry points share one rule and cannot drift. See [`GroupCurve`] for why a caller would want to
/// supply its own curves.
///
/// # Errors
/// [`AllocError::BadRange`] if `t_min > t_max`; [`AllocError::InvalidSensitivity`] for a negative or
/// non-finite sensitivity; [`AllocError::BudgetTooSmall`] if the budget cannot cover `t_min`
/// everywhere.
pub fn allocate_with_curves(
    groups: &[GroupCurve],
    cfg: &AllocConfig,
) -> Result<Allocation, AllocError> {
    if cfg.t_min > cfg.t_max {
        return Err(AllocError::BadRange {
            t_min: cfg.t_min,
            t_max: cfg.t_max,
        });
    }
    for (g, grp) in groups.iter().enumerate() {
        if !(grp.sensitivity.is_finite() && grp.sensitivity >= 0.0) {
            return Err(AllocError::InvalidSensitivity {
                group: g,
                value: grp.sensitivity,
            });
        }
    }
    let n = groups.len();
    let bits_per_plane: Vec<f64> = groups
        .iter()
        .map(|g| g.weights as f64 * TRIT_BITS)
        .collect();
    let curves: Vec<&[f64]> = groups.iter().map(|g| g.curve).collect();

    let base_bits: f64 = bits_per_plane.iter().sum::<f64>() * cfg.t_min as f64;
    // The exact floor (budget == base) must stay feasible: `base` sums per group
    // while a bpw-derived budget is one multiply, so they can disagree by float
    // rounding. Tolerate that here; plane *additions* below stay strict, so the
    // budget ceiling itself is never breached.
    let feasibility_slack = BUDGET_FEASIBILITY_SLACK_REL * base_bits.max(1.0);
    if base_bits > cfg.budget_bits + feasibility_slack {
        return Err(AllocError::BudgetTooSmall {
            base_bits,
            budget_bits: cfg.budget_bits,
        });
    }

    let mut t = vec![cfg.t_min; n];
    let mut spent = base_bits;

    // Gain of group `g`'s NEXT plane, per bit — `None` when the group is capped or the step buys
    // nothing. Affordability is deliberately NOT checked here: it depends on `spent`, which moves.
    let next_gain = |g: usize, t_g: usize| -> Option<f64> {
        if t_g >= cfg.t_max {
            return None;
        }
        let drop = curve_at(curves[g], t_g) - curve_at(curves[g], t_g + 1);
        if drop <= 0.0 {
            return None; // residual exhausted — adding a plane wastes bits
        }
        // An empty group has a flat (zero) curve → drop == 0 → returned above, so any group
        // reaching here has size > 0, hence cost > 0.
        debug_assert!(bits_per_plane[g] > 0.0);
        Some(groups[g].sensitivity * drop / bits_per_plane[g])
    };

    // A max-heap over `(gain, group)`. The linear scan this replaces was O(additions × groups),
    // which is ~2.6e12 operations on a 135M model's 1.14M groups — it does not finish. Each pop
    // here re-pushes only the popped group's next step, so it is O(additions · log groups).
    //
    // `Ord` must reproduce the scan's tie-break EXACTLY or this is a different allocator: highest
    // gain wins, and equal gains go to the LOWEST group index (the scan took strictly-greater while
    // walking g ascending). Hence gain compared normally, index compared REVERSED, since
    // `BinaryHeap` pops the maximum. `total_cmp` gives floats the total order `Ord` requires.
    #[derive(PartialEq)]
    struct Candidate {
        gain: f64,
        group: usize,
    }
    impl Eq for Candidate {}
    impl Ord for Candidate {
        fn cmp(&self, other: &Self) -> core::cmp::Ordering {
            self.gain
                .total_cmp(&other.gain)
                .then_with(|| other.group.cmp(&self.group))
        }
    }
    impl PartialOrd for Candidate {
        fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    let mut heap: std::collections::BinaryHeap<Candidate> = (0..n)
        .filter_map(|g| next_gain(g, t[g]).map(|gain| Candidate { gain, group: g }))
        .collect();

    loop {
        let best = loop {
            let Some(c) = heap.pop() else {
                break None;
            };
            // An unaffordable group can never become affordable — `spent` only grows — so drop it
            // rather than re-pushing. This is what the scan's `continue` amounted to.
            if spent + bits_per_plane[c.group] > cfg.budget_bits {
                continue;
            }
            break Some(c.group);
        };
        match best {
            Some(g) => {
                t[g] += 1;
                spent += bits_per_plane[g];
                if let Some(gain) = next_gain(g, t[g]) {
                    heap.push(Candidate { gain, group: g });
                }
            }
            None => break,
        }
    }

    Ok(Allocation { plane_counts: t })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn sizes_of(groups: &[GroupInput]) -> Vec<usize> {
        groups.iter().map(|g| g.weights.len()).collect()
    }

    // ── Gate (ADR 0006): the allocator never exceeds the budget. ─────────────
    #[test]
    fn budget_floor_leaves_everything_at_t1() {
        // Budget == base ⇒ flat AbsMean everywhere (the BitNet floor), T=1.
        let a = [1.0f32, -0.5, 0.25, 2.0];
        let b = [3.0f32, 0.1, -1.5];
        let groups = [
            GroupInput {
                weights: &a,
                sensitivity: 5.0,
            },
            GroupInput {
                weights: &b,
                sensitivity: 1.0,
            },
        ];
        let sizes = sizes_of(&groups);
        let base = (sizes.iter().sum::<usize>() as f64) * TRIT_BITS;
        let alloc = allocate(
            &groups,
            &AllocConfig {
                t_min: 1,
                t_max: 3,
                budget_bits: base,
            },
        )
        .unwrap();
        assert_eq!(alloc.plane_counts, vec![1, 1]);
        assert!(alloc.total_bits(&sizes) <= base + 1e-9);
    }

    #[test]
    fn budget_below_base_errors() {
        let a = [1.0f32, 2.0, 3.0];
        let groups = [GroupInput {
            weights: &a,
            sensitivity: 1.0,
        }];
        let base = 3.0 * TRIT_BITS;
        let err = allocate(
            &groups,
            &AllocConfig {
                t_min: 1,
                t_max: 3,
                budget_bits: base - 0.5,
            },
        );
        assert!(matches!(err, Err(AllocError::BudgetTooSmall { .. })));
    }

    #[test]
    fn invalid_sensitivity_errors() {
        let a = [1.0f32, 2.0];
        // negative H would spend budget worsening the objective
        let neg = [GroupInput {
            weights: &a,
            sensitivity: -1.0,
        }];
        assert!(matches!(
            allocate(&neg, &AllocConfig::default()),
            Err(AllocError::InvalidSensitivity { group: 0, .. })
        ));
        // NaN H would poison the greedy max-selection
        let nan = [GroupInput {
            weights: &a,
            sensitivity: f64::NAN,
        }];
        assert!(matches!(
            allocate(&nan, &AllocConfig::default()),
            Err(AllocError::InvalidSensitivity { group: 0, .. })
        ));
    }

    #[test]
    fn bad_range_errors() {
        let a = [1.0f32];
        let groups = [GroupInput {
            weights: &a,
            sensitivity: 1.0,
        }];
        let err = allocate(
            &groups,
            &AllocConfig {
                t_min: 3,
                t_max: 2,
                budget_bits: 100.0,
            },
        );
        assert!(matches!(err, Err(AllocError::BadRange { .. })));
    }

    #[test]
    fn generous_budget_fills_high_sensitivity_first() {
        // Two equal-size, equal-curve groups; budget buys exactly ONE extra
        // plane. It must go to the higher-sensitivity group.
        let w = [1.0f32, 0.6, 0.31, 0.14, -0.8, 0.45];
        let groups = [
            GroupInput {
                weights: &w,
                sensitivity: 1.0,
            }, // low
            GroupInput {
                weights: &w,
                sensitivity: 9.0,
            }, // high
        ];
        let sizes = sizes_of(&groups);
        let base = (sizes.iter().sum::<usize>() as f64) * TRIT_BITS;
        let one_plane = w.len() as f64 * TRIT_BITS;
        let cfg = AllocConfig {
            t_min: 1,
            t_max: 3,
            budget_bits: base + one_plane,
        };
        let alloc = allocate(&groups, &cfg).unwrap();
        assert_eq!(
            alloc.plane_counts,
            vec![1, 2],
            "extra plane → higher-H group"
        );
        assert!(alloc.total_bits(&sizes) <= cfg.budget_bits + 1e-9);
    }

    proptest! {
        // ── Gate (ADR 0006): budget respected; every T in [t_min, t_max]. ────
        #[test]
        fn budget_never_exceeded(
            // heterogeneous groups: varied sizes, weights, sensitivities
            specs in prop::collection::vec(
                (prop::collection::vec(-4.0f32..4.0, 1..40), 0.0f64..10.0),
                1..12,
            ),
            extra_bpw in 0.0f64..4.0,
        ) {
            let groups: Vec<GroupInput> = specs.iter()
                .map(|(w, h)| GroupInput { weights: w, sensitivity: *h })
                .collect();
            let sizes = sizes_of(&groups);
            let total_w: usize = sizes.iter().sum();
            // budget = base (t_min=1) + a slice of extra capacity
            let base = total_w as f64 * TRIT_BITS;
            let cfg = AllocConfig {
                t_min: 1,
                t_max: 3,
                budget_bits: base + extra_bpw * total_w as f64,
            };
            let alloc = allocate(&groups, &cfg).unwrap();
            prop_assert_eq!(alloc.plane_counts.len(), groups.len());
            for &t in &alloc.plane_counts {
                prop_assert!((1..=3).contains(&t));
            }
            // never over budget (small slack for f64 accumulation order)
            let slack = 1e-6 * cfg.budget_bits.max(1.0);
            prop_assert!(alloc.total_bits(&sizes) <= cfg.budget_bits + slack);
        }

        // ── Gate (ADR 0006): ordering invariant. ─────────────────────────────
        // Groups with identical weights (⇒ identical size + error curve) must be
        // ranked purely by sensitivity: higher H ⇒ ≥ planes.
        #[test]
        fn ordering_invariant_equal_curves(
            w in prop::collection::vec(-4.0f32..4.0, 2..40),
            hs in prop::collection::vec(0.0f64..10.0, 2..8),
            extra_bpw in 0.0f64..3.0,
        ) {
            let groups: Vec<GroupInput> = hs.iter()
                .map(|h| GroupInput { weights: &w, sensitivity: *h })
                .collect();
            let sizes = sizes_of(&groups);
            let total_w: usize = sizes.iter().sum();
            let base = total_w as f64 * TRIT_BITS;
            let cfg = AllocConfig {
                t_min: 1, t_max: 3,
                budget_bits: base + extra_bpw * total_w as f64,
            };
            let alloc = allocate(&groups, &cfg).unwrap();
            // For every pair, the higher-sensitivity group has ≥ planes.
            for i in 0..groups.len() {
                for j in 0..groups.len() {
                    if hs[i] > hs[j] {
                        prop_assert!(
                            alloc.plane_counts[i] >= alloc.plane_counts[j],
                            "H[{i}]={} > H[{j}]={} but planes {} < {}",
                            hs[i], hs[j], alloc.plane_counts[i], alloc.plane_counts[j],
                        );
                    }
                }
            }
        }

        // ── Gate (ADR 0006): determinism — same input ⇒ same allocation. ─────
        #[test]
        fn allocation_is_deterministic(
            specs in prop::collection::vec(
                (prop::collection::vec(-4.0f32..4.0, 1..32), 0.0f64..10.0),
                1..10,
            ),
            extra_bpw in 0.0f64..3.0,
        ) {
            let groups: Vec<GroupInput> = specs.iter()
                .map(|(w, h)| GroupInput { weights: w, sensitivity: *h })
                .collect();
            let total_w: usize = groups.iter().map(|g| g.weights.len()).sum();
            let cfg = AllocConfig {
                t_min: 1, t_max: 3,
                budget_bits: total_w as f64 * TRIT_BITS + extra_bpw * total_w as f64,
            };
            let a = allocate(&groups, &cfg).unwrap();
            let b = allocate(&groups, &cfg).unwrap();
            prop_assert_eq!(a, b);
        }
    }
}

#[cfg(test)]
mod curve_tests {
    use super::*;
    use crate::{PlaneStack, recon_error, residual_expand};

    /// The same greedy-residual curve `allocate` builds internally, rebuilt here so the equivalence
    /// gate below compares the two entry points on identical inputs rather than on a re-derivation.
    fn greedy_curve(w: &[f32], t_max: usize) -> Vec<f64> {
        let stack = residual_expand(w, t_max);
        (0..=stack.plane_count())
            .map(|t| {
                recon_error(
                    w,
                    &PlaneStack {
                        planes: stack.planes[..t].to_vec(),
                    },
                )
            })
            .collect()
    }

    /// `allocate_with_curves` must reproduce `allocate` exactly when handed the curves `allocate`
    /// would have built. This is what makes the refactor safe: one water-filling rule, two ways in,
    /// and no chance of the ladder path silently drifting from the shipped behaviour.
    #[test]
    fn curve_entry_point_reproduces_the_weights_entry_point() {
        let a = [1.0f32, -0.5, 0.25, 2.0, 0.75, -3.0, 0.1, 0.9];
        let b = [3.0f32, 0.1, -1.5, 0.05, 2.2, -0.3];
        let groups = [
            GroupInput {
                weights: &a,
                sensitivity: 5.0,
            },
            GroupInput {
                weights: &b,
                sensitivity: 1.0,
            },
        ];
        let cfg = AllocConfig::from_bpw(3.2, a.len() + b.len(), 1, 3);
        let want = allocate(&groups, &cfg).expect("allocate");

        let ca = greedy_curve(&a, cfg.t_max);
        let cb = greedy_curve(&b, cfg.t_max);
        let curved = [
            GroupCurve {
                curve: &ca,
                weights: a.len(),
                sensitivity: 5.0,
            },
            GroupCurve {
                curve: &cb,
                weights: b.len(),
                sensitivity: 1.0,
            },
        ];
        let got = allocate_with_curves(&curved, &cfg).expect("allocate_with_curves");
        assert_eq!(got.plane_counts, want.plane_counts);
    }

    /// The whole point of the sensitivity term: at a budget that cannot buy every group a plane,
    /// the loss-critical group must win it. With identical curves, sensitivity is the only
    /// tiebreaker, so this isolates it from the error-curve shape.
    #[test]
    fn sensitivity_decides_which_group_gets_the_scarce_plane() {
        // Identical curves ⇒ identical Δerr and identical cost; only `sensitivity` differs.
        let curve = [100.0f64, 40.0, 20.0, 10.0];
        let mk = |s: f64| GroupCurve {
            curve: &curve,
            weights: 128,
            sensitivity: s,
        };
        // Base (T=1 everywhere) plus exactly one more plane's worth of bits.
        let base = 2.0 * 128.0 * TRIT_BITS;
        let cfg = AllocConfig {
            t_min: 1,
            t_max: 3,
            budget_bits: base + 128.0 * TRIT_BITS,
        };
        let got = allocate_with_curves(&[mk(1.0), mk(9.0)], &cfg).expect("allocate");
        assert_eq!(
            got.plane_counts,
            vec![1, 2],
            "the scarce plane must go to the more sensitive group"
        );
        // ... and it must flip when the sensitivity does, so this is not a group-order artifact.
        let flipped = allocate_with_curves(&[mk(9.0), mk(1.0)], &cfg).expect("allocate");
        assert_eq!(flipped.plane_counts, vec![2, 1]);
    }

    /// A group whose curve is already flat gains nothing from another plane, so the allocator must
    /// spend the budget elsewhere rather than burning it on a zero-Δerr group.
    #[test]
    fn flat_curves_never_take_a_plane() {
        let flat = [50.0f64, 50.0, 50.0, 50.0];
        let useful = [50.0f64, 10.0, 5.0, 1.0];
        let cfg = AllocConfig {
            t_min: 1,
            t_max: 3,
            budget_bits: f64::INFINITY,
        };
        let got = allocate_with_curves(
            &[
                GroupCurve {
                    curve: &flat,
                    weights: 64,
                    sensitivity: 1.0,
                },
                GroupCurve {
                    curve: &useful,
                    weights: 64,
                    sensitivity: 1.0,
                },
            ],
            &cfg,
        )
        .expect("allocate");
        assert_eq!(got.plane_counts, vec![1, 3]);
    }
}

#[cfg(test)]
mod scale_tests {
    use super::*;

    /// The linear-scan water-filling this module shipped before the heap, kept as the reference the
    /// fast path is gated against. Semantics that must be preserved exactly: skip groups at `t_max`,
    /// skip groups that cannot afford another plane, skip zero-gain steps, and break ties toward the
    /// LOWER group index.
    fn allocate_reference(groups: &[GroupCurve], cfg: &AllocConfig) -> Allocation {
        let n = groups.len();
        let bits: Vec<f64> = groups
            .iter()
            .map(|g| g.weights as f64 * TRIT_BITS)
            .collect();
        let mut t = vec![cfg.t_min; n];
        let mut spent: f64 = bits.iter().sum::<f64>() * cfg.t_min as f64;
        loop {
            let mut best: Option<(f64, usize)> = None;
            for g in 0..n {
                if t[g] >= cfg.t_max || spent + bits[g] > cfg.budget_bits {
                    continue;
                }
                let drop = curve_at(groups[g].curve, t[g]) - curve_at(groups[g].curve, t[g] + 1);
                if drop <= 0.0 {
                    continue;
                }
                let gain = groups[g].sensitivity * drop / bits[g];
                if best.is_none_or(|(b, _)| gain > b) {
                    best = Some((gain, g));
                }
            }
            match best {
                Some((_, g)) => {
                    t[g] += 1;
                    spent += bits[g];
                }
                None => break,
            }
        }
        Allocation { plane_counts: t }
    }

    fn seeded(seed: u64, n: usize) -> Vec<f64> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % 10_000) as f64 / 10_000.0
            })
            .collect()
    }

    /// The heap must reproduce the linear scan EXACTLY — including its tie-breaking — across shapes
    /// that exercise ties, zero-gain steps, mixed group sizes and a binding budget. Anything less
    /// and the fast path is a different allocator wearing the same name.
    #[test]
    fn heap_water_filling_matches_the_linear_reference() {
        for seed in 0..24u64 {
            let n = 3 + (seed as usize * 7) % 40;
            let t_max = 1 + (seed as usize) % 5;
            let raw = seeded(seed * 31 + 1, n * (t_max + 1));
            let sens = seeded(seed * 17 + 3, n);
            // Monotone non-increasing curves, with deliberate flat runs (equal consecutive values)
            // so the zero-gain skip is exercised, and deliberate duplicates across groups so ties are.
            let curves: Vec<Vec<f64>> = (0..n)
                .map(|g| {
                    let mut c = vec![100.0f64];
                    for k in 1..=t_max {
                        let step = if (g + k) % 3 == 0 {
                            0.0 // flat run
                        } else {
                            raw[g * (t_max + 1) + k] * 20.0
                        };
                        c.push((c[k - 1] - step).max(0.0));
                    }
                    c
                })
                .collect();
            let sizes: Vec<usize> = (0..n).map(|g| if g % 4 == 0 { 64 } else { 128 }).collect();
            let groups: Vec<GroupCurve<'_>> = (0..n)
                .map(|g| GroupCurve {
                    curve: &curves[g],
                    weights: sizes[g],
                    // Quantised so ties genuinely occur rather than being broken by float noise.
                    sensitivity: (sens[g] * 4.0).round().max(1.0),
                })
                .collect();

            let total: usize = sizes.iter().sum();
            for bpw_mult in [1.0f64, 1.7, 2.5, 100.0] {
                let cfg = AllocConfig::from_bpw(TRIT_BITS * bpw_mult, total, 1, t_max);
                let want = allocate_reference(&groups, &cfg);
                let got = allocate_with_curves(&groups, &cfg).expect("allocate");
                assert_eq!(
                    got.plane_counts, want.plane_counts,
                    "seed {seed} bpw_mult {bpw_mult}: heap diverged from the linear reference"
                );
            }
        }
    }

    /// Model scale. SmolLM2-135M is 1,144,320 groups at g128, and the linear scan is
    /// O(additions x groups) = ~2.6e12 operations there — it does not finish. The heap is
    /// O(additions x log groups). This gate is the reason the fast path exists, and it fails loudly
    /// if anyone reintroduces a full scan per plane.
    #[test]
    fn allocates_over_a_million_groups_quickly() {
        let n = 1_000_000usize;
        let curve = [100.0f64, 40.0, 18.0, 9.0];
        let groups: Vec<GroupCurve<'_>> = (0..n)
            .map(|g| GroupCurve {
                curve: &curve,
                weights: 128,
                // Vary sensitivity so the heap actually reorders rather than running in index order.
                sensitivity: 1.0 + (g % 97) as f64,
            })
            .collect();
        let cfg = AllocConfig::from_bpw(TRIT_BITS * 3.0, n * 128, 1, 3);

        let t0 = std::time::Instant::now();
        let alloc = allocate_with_curves(&groups, &cfg).expect("allocate");
        let secs = t0.elapsed().as_secs_f64();

        assert_eq!(alloc.plane_counts.len(), n);
        let mean: f64 = alloc.plane_counts.iter().sum::<usize>() as f64 / n as f64;
        assert!(
            (2.5..=3.0).contains(&mean),
            "a T=3 budget should be nearly exhausted, got mean T {mean:.3}"
        );
        // Generous: the linear scan would need hours here.
        assert!(secs < 30.0, "1M-group allocation took {secs:.1}s");
    }
}
