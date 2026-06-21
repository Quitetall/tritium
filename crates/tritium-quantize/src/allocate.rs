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
    let n = groups.len();
    let sizes: Vec<usize> = groups.iter().map(|g| g.weights.len()).collect();
    let bits_per_plane: Vec<f64> = sizes.iter().map(|&s| s as f64 * TRIT_BITS).collect();
    let curves: Vec<Vec<f64>> = groups
        .iter()
        .map(|g| error_curve(g.weights, cfg.t_max))
        .collect();

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

    loop {
        // Best (highest loss-drop-per-bit) affordable, beneficial plane to add.
        let mut best: Option<(f64, usize)> = None;
        for g in 0..n {
            if t[g] >= cfg.t_max {
                continue;
            }
            let cost = bits_per_plane[g];
            if spent + cost > cfg.budget_bits {
                continue; // can't afford another plane on this group
            }
            let drop = curve_at(&curves[g], t[g]) - curve_at(&curves[g], t[g] + 1);
            if drop <= 0.0 {
                continue; // residual exhausted — adding a plane wastes bits
            }
            // An empty group has a flat (zero) curve → drop == 0 → skipped above,
            // so any group reaching here has size > 0, hence cost > 0.
            debug_assert!(cost > 0.0);
            let gain_per_bit = groups[g].sensitivity * drop / cost;
            if best.is_none_or(|(b, _)| gain_per_bit > b) {
                best = Some((gain_per_bit, g));
            }
        }
        match best {
            Some((_, g)) => {
                t[g] += 1;
                spent += bits_per_plane[g];
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
