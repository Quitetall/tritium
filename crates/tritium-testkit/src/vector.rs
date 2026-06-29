//! The conformance vector and the tolerance that grades a backend against it.

use serde::{Deserialize, Serialize};

/// One self-contained conformance case: inputs to a ternary mpGEMM plus the
/// reference output every backend must reproduce.
///
/// The `expected` field is computed once, host-side, from
/// [`tritium_core::reference_mpgemm`] (see
/// [`crate::generate_vectors`]). A backend is "correct" on this vector iff its
/// `mpgemm` output is within [`Tolerance`] of `expected`. Serializing a set of
/// these to JSONL (one object per line) gives the committed, versioned
/// conformance suite that makes cross-backend parity structural rather than
/// manual.
///
/// All matrices are row-major and follow the [`tritium_core::GemmShape`]
/// convention: `act` is `[M, K]`, `weights` is `[N, K]` (output-major), `scales`
/// is `[N]` per-output-channel, and `expected` is `[M, N]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConformanceVector {
    /// Stable, human-readable identifier (e.g. `"rand-7"` or `"boundary-allpos"`).
    pub id: String,
    /// Activation rows (`M`).
    pub m: usize,
    /// Output features / weight rows (`N`).
    pub n: usize,
    /// Contraction dimension (`K`).
    pub k: usize,
    /// `[M, K]` row-major activations.
    pub activation: Vec<f32>,
    /// `[N, K]` row-major ternary weights, each element in `{-1, 0, +1}`.
    pub weights: Vec<i8>,
    /// `[N]` per-output-channel scales applied after the ternary contraction.
    pub scales: Vec<f32>,
    /// Packing scheme. Serializes to its canonical lowercase tag (`"tq2_0"` /
    /// `"tq1_0"`) via the `serde(rename = …)` attributes on
    /// [`tritium_core::TernaryFormat`], so the on-disk JSONL is unchanged.
    pub format: tritium_core::TernaryFormat,
    /// `[M, N]` reference output from [`tritium_core::reference_mpgemm`].
    pub expected: Vec<f32>,
}

/// How close a backend's output must be to [`ConformanceVector::expected`].
///
/// The float mpGEMM path tolerates a small relative error because fp32
/// accumulation reorders across backends (ADR 0002: `≤ 1e-4` relative for
/// fp32-accumulate matmul). Integer/packing paths are graded bit-exact by
/// setting [`bit_exact`](Tolerance::bit_exact); there the comparison is `==`
/// and `relative` is ignored.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Tolerance {
    /// Maximum permitted relative error `|got - want| / max(1, |want|)`.
    pub relative: f32,
    /// When `true`, require exact equality (`==`) and ignore [`relative`](Self::relative).
    pub bit_exact: bool,
}

impl Default for Tolerance {
    /// The fp32-accumulate matmul tolerance from ADR 0002: `relative = 1e-4`,
    /// not bit-exact.
    fn default() -> Self {
        Tolerance::relative(1e-4)
    }
}

impl Tolerance {
    /// A bit-exact tolerance: outputs must compare `==` ([`relative`](Self::relative)
    /// is ignored). Use for integer / packing paths.
    #[must_use]
    pub fn bit_exact() -> Self {
        Tolerance {
            relative: 0.0,
            bit_exact: true,
        }
    }

    /// A relative-error tolerance with the given bound (not bit-exact). Use for
    /// fp32-accumulate matmul, where accumulation reorders across backends.
    #[must_use]
    pub fn relative(rel: f32) -> Self {
        Tolerance {
            relative: rel,
            bit_exact: false,
        }
    }

    /// `true` if `got` is acceptably close to `want` under this tolerance.
    ///
    /// In bit-exact mode this is `got == want`. Otherwise NaNs and infinities
    /// must match exactly (a finite result can never approximate `NaN`/`±inf`),
    /// and finite values must satisfy
    /// `|got - want| <= relative * max(1, |want|)`. The `max(1, …)` floor keeps
    /// near-zero references from demanding impossible absolute precision.
    #[must_use]
    pub fn accepts(&self, got: f32, want: f32) -> bool {
        self.accepts_with_floor(got, want, 1.0)
    }

    /// Like [`accepts`](Self::accepts) but with an explicit near-zero `floor`
    /// replacing the default `1.0`: the finite-value bound is
    /// `|got - want| <= relative * max(floor, |want|)`.
    ///
    /// This exists for graded outputs that carry a per-row multiplier. The plain
    /// mpGEMM tolerance floors at `max(1, |P|)` on the *unscaled* product `P`; if
    /// the output is then scaled by a per-token factor `s` (as the fused
    /// W1.58A8 path scales by `act_scale = γ/127`), the sound floor is `s`, not
    /// `1` — because `s · max(1, |P|) = max(s, |s·P|) = max(s, |want|)`. Passing
    /// `floor = s` keeps the tolerance equivalent to the mpGEMM contract for any
    /// `s`, where the fixed `1.0` floor would falsely reject a conformant backend
    /// once `s ≥ 1` (γ > 127, routine in real activations).
    #[must_use]
    pub fn accepts_with_floor(&self, got: f32, want: f32, floor: f32) -> bool {
        if self.bit_exact {
            return got == want;
        }
        if want.is_nan() || got.is_nan() {
            return want.is_nan() && got.is_nan();
        }
        if want.is_infinite() || got.is_infinite() {
            return got == want;
        }
        let denom = want.abs().max(floor);
        (got - want).abs() <= self.relative * denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_relative_1e_4_not_bit_exact() {
        let t = Tolerance::default();
        assert_eq!(t.relative, 1e-4);
        assert!(!t.bit_exact);
    }

    #[test]
    fn relative_tolerance_grades_against_magnitude() {
        let t = Tolerance::default();
        // Within 1e-4 of a large value: accepted.
        assert!(t.accepts(1000.0, 1000.05));
        // Outside 1e-4 of the same value: rejected.
        assert!(!t.accepts(1000.0, 1001.0));
        // Near zero, the max(1, |want|) floor allows a 1e-4 absolute slack.
        assert!(t.accepts(0.0, 5e-5));
        assert!(!t.accepts(0.0, 5e-3));
    }

    #[test]
    fn nan_and_inf_must_match_exactly() {
        let t = Tolerance::default();
        assert!(t.accepts(f32::NAN, f32::NAN));
        assert!(!t.accepts(f32::NAN, 1.0));
        assert!(!t.accepts(1.0, f32::NAN));
        assert!(t.accepts(f32::INFINITY, f32::INFINITY));
        assert!(!t.accepts(f32::INFINITY, f32::NEG_INFINITY));
        assert!(!t.accepts(f32::INFINITY, 1.0));
    }

    #[test]
    fn floor_lifts_tolerance_through_a_per_row_scale() {
        let t = Tolerance::default();
        // Unscaled products P_ref=1.25, P_sub differing by 2e-4 are within the
        // mpGEMM 1e-4 tolerance once floored at the row's act_scale. Here the
        // output is scaled by s=2.5 (γ>127): want=1.25, got=1.2502, |Δ|=2e-4.
        let (s, want, got) = (2.5_f32, 1.25_f32, 1.2502_f32);
        // The fixed 1.0 floor falsely rejects this conformant backend
        // (bound = 1e-4·max(1.25,1) = 1.25e-4 < 2e-4)...
        assert!(!t.accepts(got, want));
        // ...the scale-aware floor accepts it (bound = 1e-4·max(1.25,2.5) = 2.5e-4).
        assert!(t.accepts_with_floor(got, want, s));
    }

    #[test]
    fn bit_exact_requires_equality() {
        let t = Tolerance::bit_exact();
        assert!(t.accepts(1.5, 1.5));
        // Even a tiny difference is rejected, and relative is ignored.
        assert!(!t.accepts(1.5, 1.5000001));
        // bit_exact compares NaN with ==, which is always false.
        assert!(!t.accepts(f32::NAN, f32::NAN));
    }
}
