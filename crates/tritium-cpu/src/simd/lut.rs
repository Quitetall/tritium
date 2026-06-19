//! T-MAC lookup-table ternary mpGEMM (v0.30, WF-C).
//!
//! The Microsoft **T-MAC** scheme: instead of walking the weights and
//! conditionally adding/subtracting each activation, precompute — for one short
//! group of `g` consecutive activations — the partial sum that *every possible*
//! ternary sub-pattern over those `g` positions would produce, then look the
//! answer up by index. A group of `g` ternary weights `(w_0, …, w_{g-1})`,
//! each in `{-1, 0, +1}`, encodes a base-3 index
//!
//! ```text
//! idx = Σ_{j=0}^{g-1} (w_j + 1) · 3^j           // w_j + 1 ∈ {0, 1, 2}
//! ```
//!
//! into a table of `3^g` precomputed partial sums
//!
//! ```text
//! table[idx] = Σ_{j=0}^{g-1} sign(w_j) · act_j
//! ```
//!
//! so the inner loop becomes one gather per group instead of `g` branchy
//! add/sub/skip ops. This is the table-build half of the scheme — pure, safe,
//! ISA-agnostic arithmetic over the activation row. The per-ISA *gather*
//! (`vpermb`/`vpshufb` on x86, `vqtbl` on NEON) lives in [`super::avx512`] /
//! [`super::neon`]; both index this same table.
//!
//! ## Accumulation order and the parity bar
//!
//! Two distinct correctness claims, deliberately kept separate:
//!
//! 1. **Per-group table — bit-exact.** [`build_group_table`] folds one group's
//!    activations from a fresh `0.0` accumulator in increasing `j` order, the
//!    *same* fold a direct add/sub/skip over that group performs. So a gathered
//!    partial is **bit-for-bit** equal to the direct signed sum of the group —
//!    this is what the task's parity tests assert and what the per-ISA gather
//!    must reproduce.
//!
//! 2. **Full-row contraction — within ADR 0002 tolerance, not bit-exact.** A
//!    full row folds each group's partial *from a fresh accumulator* and then
//!    adds those partials together. That **re-associates** the `K` additions
//!    relative to the reference's strict single-accumulator left-fold
//!    (`((a₀ ± a₁) ± a₂) …`), so the last-bit rounding differs in general — the
//!    grouped sum is a different (equally valid) summation order. The result
//!    stays within the `1e-4` relative tolerance of ADR 0002, the bar a kernel
//!    is actually held to, but it is **not** the bit-exact reproduction the
//!    AVX2/AVX-512/NEON kernels achieve (those fold per-element signed
//!    contributions without an intermediate group sum, so they keep the exact
//!    k-order). The LUT trades that last-bit reproducibility for the lookup.
//!
//! `acc + 0.0 == acc` (the skip) and `acc + (-a) == acc - a` (the subtract) are
//! exact in IEEE round-to-nearest, so neither claim loses anything to the
//! skip/subtract themselves; the only divergence in (2) is the grouped
//! re-association.
//!
//! The default group width is [`LUT_GROUP`] = 4 (`3^4 = 81` entries/table),
//! the T-MAC sweet spot for a `vpshufb`-class 16-or-fewer-byte gather budget
//! while keeping the table small enough to rebuild per activation row.
//!
//! ## Wiring status
//!
//! This module is fully implemented and unit-tested, but its *production*
//! consumer is the per-ISA SIMD gather (`vpermb`/`vpshufb` on x86, `vqtbl` on
//! NEON), which is a later step. Until that lands the table-build + portable
//! gather is not on the [`crate::kernel`] dispatch hot path (the bit-exact scalar
//! reference is the terminal fallback, to keep cross-host output byte-identical),
//! so in a non-test build these items are not yet called — hence the
//! `cfg_attr(not(test), allow(dead_code))` below, scoped to this module and
//! lifted automatically once the gather wires `lut_mpgemm` into dispatch.
#![cfg_attr(not(test), allow(dead_code))]

use tritium_core::{GemmShape, Trit};

use tritium_spec::BackendError;

use crate::kernel::scalar_mpgemm;

/// Ternary group width: how many consecutive weights share one lookup table.
///
/// `3^4 = 81` partial sums per table — small enough to rebuild cheaply per
/// activation row, wide enough to amortise the per-group gather. The build and
/// gather helpers are generic over the width via [`pow3`]; this constant only
/// fixes the default used by the kernels.
pub(crate) const LUT_GROUP: usize = 4;

/// `3^g` — the number of partial sums in a width-`g` ternary table.
///
/// Computed with a plain `u32` accumulator (no `pow` on `usize`) so it is a
/// `const fn` usable in array sizing and so the intent — base-3, one digit per
/// weight — is explicit. Widths above 20 would overflow `u32`; the kernels only
/// ever pass [`LUT_GROUP`] = 4, far below that.
#[must_use]
pub(crate) const fn pow3(g: u32) -> usize {
    let mut acc: usize = 1;
    let mut i = 0;
    while i < g {
        acc *= 3;
        i += 1;
    }
    acc
}

/// Base-3 index of a ternary group: `Σ_j (w_j + 1) · 3^j`, low weight first.
///
/// Each trit `w_j ∈ {-1, 0, +1}` maps to a base-3 digit `w_j + 1 ∈ {0, 1, 2}`,
/// so the index ranges over `0..pow3(g)` and is the slot the matching partial
/// sum lives in. Digit `j = 0` is the lowest, matching the `3^j` weighting in
/// [`build_group_table`], so a table built for `weights[0..g]` is gathered with
/// `group_index(&weights[0..g])`.
///
/// `weights.len()` is the group width `g`; callers pass exactly one group.
#[must_use]
pub(crate) fn group_index(weights: &[Trit]) -> usize {
    let mut idx = 0usize;
    let mut place = 1usize;
    for &w in weights {
        // w.get() ∈ {-1,0,1}  ⇒  digit ∈ {0,1,2}
        let digit = (w.get() + 1) as usize;
        idx += digit * place;
        place *= 3;
    }
    idx
}

/// Build the width-`g` T-MAC partial-sum table for one group of `g`
/// activations.
///
/// `act` is the group's `g` activations (`act.len()` is the width `g`); `table`
/// is the caller-provided scratch of length `pow3(g)` that is overwritten with
///
/// ```text
/// table[idx] = Σ_{j=0}^{g-1} sign(decode(idx, j)) · act[j]
/// ```
///
/// where `decode(idx, j)` is the `j`-th base-3 digit of `idx` mapped back to
/// `{-1, 0, +1}`. The build folds `j` in increasing order (`j = 0` first) so a
/// gathered partial is the exact `f32` sub-sum the reference would compute by
/// walking those `g` positions in k-order.
///
/// # Panics
/// Debug-asserts `table.len() == pow3(act.len() as u32)`; in release the loop
/// simply writes the first `pow3(g)` slots, so an over-long `table` is harmless
/// and a short one is caught by the indexing.
pub(crate) fn build_group_table(act: &[f32], table: &mut [f32]) {
    let g = act.len() as u32;
    let n = pow3(g);
    debug_assert_eq!(
        table.len(),
        n,
        "LUT table length {} must equal 3^{g} = {n}",
        table.len()
    );

    // Enumerate every base-3 pattern `idx ∈ 0..3^g`. For each, decode its `g`
    // digits low-to-high and fold the matching signed activation — the same
    // increasing-`j` add/sub/skip the reference performs, so a gathered partial
    // is bit-identical to the reference's k-order sub-sum.
    for (idx, slot) in table.iter_mut().enumerate().take(n) {
        let mut acc = 0.0f32;
        let mut rem = idx;
        for &a in act {
            let digit = rem % 3; // {0,1,2}
            rem /= 3;
            // digit 0 → -1 (subtract), 1 → 0 (skip), 2 → +1 (add).
            match digit {
                0 => acc -= a,
                2 => acc += a,
                _ => {}
            }
        }
        *slot = acc;
    }
}

/// T-MAC row contraction: `Σ_k act[k] · sign(weights[k])`, computed
/// group-by-group through the lookup table, folding partials in increasing group
/// order.
///
/// This is the ISA-agnostic kernel the SIMD gathers accelerate: it builds one
/// width-[`LUT_GROUP`] table per group of activations, gathers the group's
/// partial sum by [`group_index`], and folds those partials in k-order. The
/// ragged tail (`k % LUT_GROUP` positions that do not fill a group) is folded
/// with the same add/sub/skip in k-order.
///
/// Each gathered partial is bit-exact vs the direct signed sum of *its* group
/// (claim 1 in the module note); the full-row fold re-associates the `K`
/// additions relative to [`tritium_core::reference_mpgemm`]'s strict left-fold,
/// so the row result agrees with the reference **within the `1e-4` tolerance of
/// ADR 0002**, not bit-for-bit (claim 2). That is the correctness bar for a
/// kernel; the bit-exact reproduction is reserved for the per-element SIMD
/// kernels.
///
/// `act` and `weights` are one row each of length `k`; the return value is the
/// unscaled dot product (the caller applies the per-channel scale, exactly as
/// the other kernels do).
#[must_use]
pub(crate) fn lut_row_dot(act: &[f32], weights: &[Trit]) -> f32 {
    debug_assert_eq!(act.len(), weights.len(), "act and weights rows must match");
    let k = act.len();
    let g = LUT_GROUP;
    let k_groups = k - (k % g);

    // One reusable table per row; rebuilt per group (the activations change, the
    // 3^g pattern space does not).
    let mut table = [0.0f32; pow3(LUT_GROUP as u32)];

    let mut acc = 0.0f32;
    let mut base = 0;
    while base < k_groups {
        let act_g = &act[base..base + g];
        let w_g = &weights[base..base + g];
        build_group_table(act_g, &mut table);
        let idx = group_index(w_g);
        // Fold this group's gathered partial in increasing group order.
        acc += table[idx];
        base += g;
    }

    // Ragged tail: the final `k % g` positions, same add/sub/skip in k-order.
    for kt in k_groups..k {
        match weights[kt].get() {
            1 => acc += act[kt],
            -1 => acc -= act[kt],
            _ => {}
        }
    }

    acc
}

/// T-MAC lookup-table ternary mpGEMM over an already-unpacked `[N, K]` trit
/// matrix — the portable (ISA-agnostic) realisation of the scheme.
///
/// Computes `out[m, n] = scale[n] · Σ_k act[m, k] · sign(weights[n, k])` by
/// driving [`lut_row_dot`] over every `(m, n)` pair: one width-[`LUT_GROUP`]
/// table per group, gathered by base-3 index, partials folded in k-order. The
/// grouped fold re-associates the `K` additions relative to the reference's
/// strict left-fold, so the output agrees with
/// [`tritium_core::reference_mpgemm`] / [`crate::kernel::scalar_mpgemm`] **within
/// the `1e-4` tolerance of ADR 0002** (the kernel correctness bar), deterministic
/// run-to-run, but not bit-for-bit (see [`lut_row_dot`]).
///
/// This is the kernel the per-ISA SIMD gathers (`vpermb`/`vpshufb` on x86,
/// `vqtbl` on NEON) will accelerate without changing its arithmetic. It is
/// implemented and unit-tested but **not yet on the [`crate::kernel`] dispatch
/// path**: the bit-exact scalar reference is the terminal non-SIMD fallback (to
/// keep cross-host output byte-reproducible), and the LUT goes live once its SIMD
/// gather is wired in — see this module's "Wiring status" note.
///
/// # Errors
/// [`BackendError::Core`] if a buffer length disagrees with `shape`; in that case
/// no table is built (the call defers to the scalar reference for its typed
/// error).
pub(crate) fn lut_mpgemm(
    act: &[f32],
    weights: &[Trit],
    scales: &[f32],
    shape: GemmShape,
    out: &mut [f32],
) -> Result<(), BackendError> {
    let GemmShape { m, n, k } = shape;
    // Validate up front; on mismatch defer to the reference for its typed error.
    if act.len() != m * k || weights.len() != n * k || scales.len() != n || out.len() != m * n {
        return scalar_mpgemm(act, weights, scales, shape, out);
    }

    for mi in 0..m {
        let arow = &act[mi * k..mi * k + k];
        for ni in 0..n {
            let wrow = &weights[ni * k..ni * k + k];
            out[mi * n + ni] = lut_row_dot(arow, wrow) * scales[ni];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use tritium_core::Trit;

    /// Direct add/sub/skip over a group — the thing the table must reproduce.
    fn direct_group_sum(act: &[f32], weights: &[Trit]) -> f32 {
        let mut acc = 0.0f32;
        for (j, &w) in weights.iter().enumerate() {
            match w.get() {
                1 => acc += act[j],
                -1 => acc -= act[j],
                _ => {}
            }
        }
        acc
    }

    fn trit_from_digit(d: usize) -> Trit {
        // 0 → -1, 1 → 0, 2 → +1  (the encoding `build_group_table` decodes).
        Trit::from_sign(d as i8 - 1)
    }

    #[test]
    fn pow3_matches_base3() {
        assert_eq!(pow3(0), 1);
        assert_eq!(pow3(1), 3);
        assert_eq!(pow3(2), 9);
        assert_eq!(pow3(3), 27);
        assert_eq!(pow3(4), 81);
        assert_eq!(pow3(5), 243);
    }

    #[test]
    fn group_index_round_trips_every_pattern() {
        // For width 4, every index 0..81 must decode to a group whose
        // `group_index` is that same index — the table slot and the gather agree.
        let g = LUT_GROUP;
        for idx in 0..pow3(g as u32) {
            let mut rem = idx;
            let mut grp = Vec::with_capacity(g);
            for _ in 0..g {
                grp.push(trit_from_digit(rem % 3));
                rem /= 3;
            }
            assert_eq!(group_index(&grp), idx, "index round-trip failed at {idx}");
        }
    }

    #[test]
    fn table_is_bit_exact_vs_direct_for_every_pattern() {
        // A fixed activation group; the table entry for every one of the 81
        // ternary patterns must equal the direct signed sum, bit-for-bit.
        let act = [1.5f32, -2.25, 0.125, 7.0];
        let mut table = [0.0f32; 81];
        build_group_table(&act, &mut table);
        let g = LUT_GROUP;
        for (idx, &entry) in table.iter().enumerate().take(pow3(g as u32)) {
            let mut rem = idx;
            let mut grp = Vec::with_capacity(g);
            for _ in 0..g {
                grp.push(trit_from_digit(rem % 3));
                rem /= 3;
            }
            let direct = direct_group_sum(&act, &grp);
            assert_eq!(
                entry.to_bits(),
                direct.to_bits(),
                "table[{idx}] {entry} != direct {direct} for {grp:?}"
            );
        }
    }

    proptest! {
        /// Random activation groups × random ternary patterns: the gathered
        /// partial must be bit-exact vs the direct add/sub/skip.
        #[test]
        fn prop_table_gather_bit_exact(
            act in prop::array::uniform4(-1000.0f32..1000.0),
            digits in prop::array::uniform4(0usize..3),
        ) {
            let mut table = [0.0f32; 81];
            build_group_table(&act, &mut table);
            let grp: Vec<Trit> = digits.iter().map(|&d| trit_from_digit(d)).collect();
            let idx = group_index(&grp);
            let direct = direct_group_sum(&act, &grp);
            prop_assert_eq!(table[idx].to_bits(), direct.to_bits());
        }

        /// Random full rows: the grouped LUT contraction must agree with the
        /// reference's single-accumulator k-order add/sub/skip **within the ADR
        /// 0002 `1e-4` relative tolerance**. The grouping re-associates the adds,
        /// so this is a tolerance check, not bit-equality (that bar belongs to the
        /// per-element SIMD kernels). Covers ragged tails (k not a multiple of the
        /// group width) and large-magnitude activations where the f32
        /// cancellation — and hence the re-association gap — is worst.
        #[test]
        fn prop_lut_row_within_tolerance_of_reference(
            k in 1usize..300,
            seed in any::<u64>(),
        ) {
            let mut s = seed | 1;
            let mut next = || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s
            };
            let act: Vec<f32> = (0..k)
                .map(|_| (next() % 20000) as f32 / 100.0 - 100.0)
                .collect();
            let w: Vec<Trit> = (0..k)
                .map(|_| Trit::from_sign((next() % 3) as i8 - 1))
                .collect();

            // Reference: a single f32 accumulator, k-order add/sub/skip.
            let mut want = 0.0f32;
            for ki in 0..k {
                match w[ki].get() {
                    1 => want += act[ki],
                    -1 => want -= act[ki],
                    _ => {}
                }
            }

            let got = lut_row_dot(&act, &w);
            // ADR 0002: relative tolerance 1e-4 with a unit-magnitude floor.
            let tol = 1e-4 * want.abs().max(1.0);
            prop_assert!(
                (got - want).abs() <= tol,
                "lut {got} vs reference {want} (k={k}, tol={tol})"
            );
        }

        /// Determinism: the LUT contraction is byte-identical run-to-run for the
        /// same input (no thread- or order-dependence inside a row).
        #[test]
        fn prop_lut_row_is_deterministic(
            k in 1usize..300,
            seed in any::<u64>(),
        ) {
            let mut s = seed | 1;
            let mut next = || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s
            };
            let act: Vec<f32> = (0..k)
                .map(|_| (next() % 20000) as f32 / 100.0 - 100.0)
                .collect();
            let w: Vec<Trit> = (0..k)
                .map(|_| Trit::from_sign((next() % 3) as i8 - 1))
                .collect();
            prop_assert_eq!(lut_row_dot(&act, &w).to_bits(), lut_row_dot(&act, &w).to_bits());
        }
    }

    /// Relative-tolerance check matching ADR 0002 (`1e-4`).
    fn close(got: f32, want: f32) -> bool {
        (got - want).abs() <= 1e-4 * want.abs().max(1.0)
    }

    #[test]
    fn lut_row_all_zero_and_all_one() {
        let act: Vec<f32> = (0..40).map(|i| (i as f32) * 0.5 - 7.0).collect();
        // All-zero weights → every group partial is exactly 0.0 and the fold of
        // zeros is exactly 0.0, so this case *is* bit-exact (no re-association
        // gap when every term is zero).
        let zeros = vec![Trit::ZERO; 40];
        assert_eq!(lut_row_dot(&act, &zeros), 0.0);
        // All +1 → the row sum, within tolerance (the grouping re-associates the
        // adds vs the reference's strict left-fold).
        let pos = vec![Trit::POS; 40];
        let mut want = 0.0f32;
        for &a in &act {
            want += a;
        }
        assert!(close(lut_row_dot(&act, &pos), want));
        // All -1 → the negated row sum, within tolerance.
        let neg = vec![Trit::NEG; 40];
        let mut want_neg = 0.0f32;
        for &a in &act {
            want_neg -= a;
        }
        assert!(close(lut_row_dot(&act, &neg), want_neg));
    }

    #[test]
    fn lut_row_handles_ragged_tail() {
        // k = 11 with group width 4 → two full groups + a 3-wide tail. Small
        // exactly-representable integer activations: the sum is an exact integer,
        // so re-association cannot change the bits here — a bit-exact check.
        let act = [
            1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0, 11.0,
        ];
        let w: Vec<Trit> = [1i8, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1]
            .iter()
            .map(|&v| Trit::from_sign(v))
            .collect();
        let mut want = 0.0f32;
        for ki in 0..11 {
            match w[ki].get() {
                1 => want += act[ki],
                -1 => want -= act[ki],
                _ => {}
            }
        }
        assert_eq!(lut_row_dot(&act, &w).to_bits(), want.to_bits());
    }

    /// End-to-end `lut_mpgemm` over a multi-row, multi-output shape must agree
    /// with [`tritium_core::reference_mpgemm`] within the ADR 0002 tolerance,
    /// including the per-channel scale and a ragged tail. Exercises the full
    /// portable LUT kernel the per-ISA SIMD gather will later accelerate.
    #[test]
    fn lut_mpgemm_matches_reference_within_tolerance() {
        use tritium_core::{GemmShape, reference_mpgemm};
        let mut s = 0x5EED_1234u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for &(m, n, k) in &[
            (3usize, 5usize, 19usize),
            (2, 4, 256),
            (1, 1, 11),
            (4, 2, 512),
        ] {
            let act: Vec<f32> = (0..m * k)
                .map(|_| (next() % 20000) as f32 / 100.0 - 100.0)
                .collect();
            let w: Vec<Trit> = (0..n * k)
                .map(|_| Trit::from_sign((next() % 3) as i8 - 1))
                .collect();
            let scales: Vec<f32> = (0..n)
                .map(|_| (next() % 200) as f32 / 100.0 + 0.1)
                .collect();
            let shape = GemmShape::new(m, n, k);

            let mut want = vec![0.0f32; m * n];
            reference_mpgemm(&act, &w, &scales, shape, &mut want).unwrap();
            let mut got = vec![0.0f32; m * n];
            lut_mpgemm(&act, &w, &scales, shape, &mut got).unwrap();

            for (g, e) in got.iter().zip(&want) {
                assert!(close(*g, *e), "lut {g} vs reference {e} (shape {shape:?})");
            }
        }
    }

    /// A wrong-length buffer is rejected with the reference's typed error, no
    /// table built.
    #[test]
    fn lut_mpgemm_rejects_bad_lengths() {
        use tritium_core::GemmShape;
        let shape = GemmShape::new(2, 1, 8);
        let act = vec![0.0f32; 8]; // should be 2*8
        let w = vec![Trit::ZERO; 8];
        let scales = vec![1.0f32; 1];
        let mut out = vec![0.0f32; 2];
        let err = lut_mpgemm(&act, &w, &scales, shape, &mut out).unwrap_err();
        assert!(
            matches!(err, BackendError::Core(_)),
            "expected typed Core(ShapeMismatch), got {err:?}"
        );
    }
}
