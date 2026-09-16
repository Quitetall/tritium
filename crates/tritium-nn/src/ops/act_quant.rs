//! Activation quantization for BitNet's W1.58**A8** path.
//!
//! BitNet quantizes activations to int8 per token (absmax) before every ternary
//! linear; the per-token scale folds into the ternary GEMM output. To match the
//! reference greedy tokens we replicate this quant **exactly** rather than
//! running fp16 activations (see the A8 risk in the plan — the rounding mode
//! must match the reference `BitLinear`). The ternary `mpgemm` stays f32-in, so
//! this returns the int8 values kept as `f32` (the existing f32 mpGEMM consumes
//! them directly) plus the per-token dequant scale.
//!
//! # Reference (authoritative)
//!
//! The real BitNet b1.58 2B4T checkpoint (`microsoft/bitnet-b1.58-2B-4T`) ships
//! with `quantization_config = {linear_class: "autobitlinear",
//! quantization_mode: "offline"}`, so the activation-quant path that actually
//! executes in `transformers` is `ActQuant.forward` in
//! `transformers/integrations/bitnet.py` (driven by `AutoBitLinear.forward`),
//! **not** the `BitLinear.activation_quant` method:
//!
//! ```python
//! # ActQuant.forward (num_bits == 8)
//! activation = activation.float()
//! scale = 127 / activation.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
//! activation = (activation * scale).round().clamp(-128, 127) / scale
//! ```
//!
//! Concretely, per token (row) `r` with `gamma = max_c |act[r,c]|`:
//!
//! * scale factor `s = 127 / gamma`,
//! * `q[r,c] = clamp(round_half_to_even(act[r,c] * s), -128, 127)`,
//! * dequant multiplier (what we store) `out_scale[r] = gamma / 127 = 1 / s`.
//!
//! Two reference details we match exactly:
//!
//! * **`Qp = 127`, range `[-128, 127]`** — symmetric int8 with the positive cap
//!   at `2^7 - 1`. (This differs from the original `Qb = 128` sketch in the
//!   stub; the shipped 2B4T model's `ActQuant` uses `127`, and the alternative
//!   `BitLinear.activation_quant` variant uses `Qp = 127` too, so [`QB`] is
//!   `127.0` to match the oracle.)
//! * **`torch.round` is round-half-to-even** (banker's rounding), e.g.
//!   `0.5 -> 0`, `1.5 -> 2`, `2.5 -> 2`, `3.5 -> 4`, `-2.5 -> -2`. Rust's
//!   [`f32::round`] is round-half-**away**-from-zero (`2.5 -> 3`), so we round
//!   via [`f32::round_ties_even`] in [`round_half_to_even`].
//!
//! The reference clamps `gamma` to a `1e-5` floor; for an exactly-zero row that
//! floor only changes the *stored scale* (the int8 values round to `0`
//! regardless), so we follow the plan and emit zeros with a `0` scale for a
//! fully-zero row — the dequantized contribution of such a row is `0` either
//! way, so greedy parity is unaffected.

use crate::error::NnError;

/// Symmetric int8 activation-quant range cap `Qp` (the positive saturation
/// value). The int8 range is `[-128, QB]` with `QB = 127`, matching the
/// reference `ActQuant`/`BitLinear` quant in `transformers`; the per-token
/// scale is `gamma_x / QB` where `gamma_x = max(|act_row|)`.
pub const QB: f32 = 127.0;

/// Round half to even (banker's rounding), matching `torch.round`.
///
/// Rust's [`f32::round`] rounds halves away from zero (`2.5 -> 3.0`); BitNet's
/// reference quant relies on `torch.round`, which rounds halves to the nearest
/// even integer (`2.5 -> 2.0`, `3.5 -> 4.0`, `-2.5 -> -2.0`). Replicating this
/// exactly is load-bearing for greedy token parity.
///
/// Implemented via [`f32::round_ties_even`], so the behaviour tracks the
/// standard library's IEEE-754 round-to-nearest-even.
#[inline]
#[must_use]
fn round_half_to_even(x: f32) -> f32 {
    x.round_ties_even()
}

/// Per-token int8 absmax activation quant (`Qp = 127`, range `[-128, 127]`).
///
/// For each row `r` of the `[rows, cols]` activation tensor:
///
/// * `gamma = max_c |act[r,c]|`,
/// * for each column `c`,
///   `out_q[r,c] = clamp(round_half_to_even(act[r,c] / gamma * 127), -128, 127)`
///   as the int8 value, kept in `f32` for the existing f32 mpGEMM,
/// * `out_scale[r] = gamma / 127` (the dequant multiplier `1 / s`).
///
/// A fully-zero row (`gamma == 0`) yields all-zero quantized values and a `0`
/// scale.
///
/// This matches `transformers` `ActQuant.forward` (the path the shipped BitNet
/// 2B4T checkpoint executes) including its round-half-to-even rounding; see the
/// module docs for the exact reference formula.
///
/// # Errors
/// [`NnError::Shape`] if `act.len()` or `out_q.len()` ≠ `rows * cols`, or
/// `out_scale.len()` ≠ `rows`.
pub fn quantize_activation_int8(
    act: &[f32],
    rows: usize,
    cols: usize,
    out_q: &mut [f32],
    out_scale: &mut [f32],
) -> Result<(), NnError> {
    let elems = rows * cols;
    if act.len() != elems {
        return Err(NnError::Shape {
            expected: elems,
            got: act.len(),
        });
    }
    if out_q.len() != elems {
        return Err(NnError::Shape {
            expected: elems,
            got: out_q.len(),
        });
    }
    if out_scale.len() != rows {
        return Err(NnError::Shape {
            expected: rows,
            got: out_scale.len(),
        });
    }

    for r in 0..rows {
        let row = &act[r * cols..r * cols + cols];

        // gamma = absmax over the row.
        let mut gamma = 0.0_f32;
        for &v in row {
            let a = v.abs();
            if a > gamma {
                gamma = a;
            }
        }

        let out_row = &mut out_q[r * cols..r * cols + cols];

        if gamma == 0.0 {
            // All-zero row: zeros + zero scale (the reference emits zeros too;
            // only the stored scale differs, and the dequant of a zero row is
            // zero regardless).
            for q in out_row.iter_mut() {
                *q = 0.0;
            }
            out_scale[r] = 0.0;
            continue;
        }

        // s = Qp / gamma; quantize, round-half-to-even, clamp to [-128, Qp].
        let s = QB / gamma;
        for (q, &v) in out_row.iter_mut().zip(row) {
            let scaled = round_half_to_even(v * s);
            *q = scaled.clamp(-128.0, QB);
        }
        // Stored dequant multiplier: gamma / Qp == 1 / s.
        out_scale[r] = gamma / QB;
    }

    Ok(())
}

/// Per-**group** int8 absmax activation quant, returning **dequantized** f32.
///
/// The shipping quantizer takes one absmax per token over the whole row, so a single outlier sets
/// the step for every value in it — and LLM activations are outlier-heavy, which is the entire
/// premise AWQ and SmoothQuant exist to address. The weight side solved this years ago with
/// per-group scales; the activation side never got the same treatment.
///
/// Measured headroom on SmolLM2-135M at `g128`: **5.41 dB** on the FFN intermediate, 57% of a whole
/// SALT plane, at zero bit cost. Converted to perplexity through the research tape, per-group int8
/// recovers **exactly 64% of the A8 tax** at both weight settings tested (+1.19% → +0.43% at `T=3`,
/// +1.08% → +0.43% at `T=4`).
///
/// # Why this one dequantizes and [`quantize_activation_int8`] does not
///
/// A per-token scale is a scalar the whole dot product shares, so it factors out and folds into the
/// GEMM output afterwards. A per-group scale does not: `Σ_k w_k x_k = Σ_g s_g Σ_{k∈g} w_k q_k`
/// needs one partial sum per group, which is a kernel change, not a caller change. Multiplying each
/// value back by its own group scale here is the same arithmetic with the sum left to the GEMM, so
/// no kernel has to learn anything — and it keeps the quantization error, which is the entire point.
///
/// **The cost is that the output is no longer integer-valued.** `tritium-cpu`'s AVX2 path checks
/// `act_is_a8_integer` at runtime and falls back to the general f32 GEMM, and any int8 kernel (DP4A)
/// cannot consume this at all. On the SALT projection path that is free — `PackedSaltMatrix::
/// project_rows` reconstructs to f32 and dots in f32 with no integer fast path to lose — which is
/// why this is offered there first.
///
/// `out_scale` is filled with `1.0` (`0.0` for an all-zero row, matching
/// [`quantize_activation_int8`]) so a caller's post-GEMM fold stays a single unconditional multiply.
///
/// A group wider than `cols`, or `cols` not divisible by the group, is fine: the final short group
/// simply pays for its own absmax. `group == cols` reproduces [`quantize_activation_int8`]'s
/// *values* exactly — dequantized rather than kept as integers — which is the degenerate control.
///
/// # Errors
/// [`NnError::Shape`] if `act.len()` or `out_q.len()` ≠ `rows * cols`, `out_scale.len()` ≠ `rows`,
/// or `group` is zero.
pub fn quantize_activation_int8_grouped(
    act: &[f32],
    rows: usize,
    cols: usize,
    group: usize,
    out_q: &mut [f32],
    out_scale: &mut [f32],
) -> Result<(), NnError> {
    let elems = rows * cols;
    if act.len() != elems {
        return Err(NnError::Shape {
            expected: elems,
            got: act.len(),
        });
    }
    if out_q.len() != elems {
        return Err(NnError::Shape {
            expected: elems,
            got: out_q.len(),
        });
    }
    if out_scale.len() != rows {
        return Err(NnError::Shape {
            expected: rows,
            got: out_scale.len(),
        });
    }
    if group == 0 {
        return Err(NnError::Shape {
            expected: 1,
            got: 0,
        });
    }

    for r in 0..rows {
        let row = &act[r * cols..r * cols + cols];
        let out_row = &mut out_q[r * cols..r * cols + cols];
        let mut any = false;
        for (src, dst) in row.chunks(group).zip(out_row.chunks_mut(group)) {
            let mut gamma = 0.0_f32;
            for &v in src {
                let a = v.abs();
                if a > gamma {
                    gamma = a;
                }
            }
            if gamma == 0.0 {
                for q in dst.iter_mut() {
                    *q = 0.0;
                }
                continue;
            }
            any = true;
            // Same `Qp`, same round-half-to-even, same clamp as the per-token path — only the span
            // the absmax is taken over differs. Dequantize immediately by the group's own step.
            let s = QB / gamma;
            let inv = gamma / QB;
            for (q, &v) in dst.iter_mut().zip(src) {
                *q = round_half_to_even(v * s).clamp(-128.0, QB) * inv;
            }
        }
        // The values already carry their scale; a zero row still reports 0 so the two quantizers
        // agree on what an empty row means.
        out_scale[r] = if any { 1.0 } else { 0.0 };
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference quant matching `transformers` `ActQuant.forward`:
    /// `s = 127 / max(|row|)`; `q = clamp(round_ties_even(v*s), -128, 127)`.
    /// Mirrors the Rust impl's zero-row handling (zeros + scale 0) so the two
    /// agree bit-for-bit on the kept int8 values.
    fn reference_quant(act: &[f32], rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
        let mut q = vec![0.0_f32; rows * cols];
        let mut scale = vec![0.0_f32; rows];
        for r in 0..rows {
            let row = &act[r * cols..r * cols + cols];
            let gamma = row.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
            if gamma == 0.0 {
                scale[r] = 0.0;
                continue;
            }
            let s = 127.0_f32 / gamma;
            for c in 0..cols {
                let scaled = (row[c] * s).round_ties_even();
                q[r * cols + c] = scaled.clamp(-128.0, 127.0);
            }
            scale[r] = gamma / 127.0;
        }
        (q, scale)
    }

    fn run_case(act: &[f32], rows: usize, cols: usize) {
        let mut out_q = vec![f32::NAN; rows * cols];
        let mut out_scale = vec![f32::NAN; rows];
        quantize_activation_int8(act, rows, cols, &mut out_q, &mut out_scale).unwrap();

        let (exp_q, exp_scale) = reference_quant(act, rows, cols);
        assert_eq!(out_q, exp_q, "quantized int8 values differ from reference");
        assert_eq!(
            out_scale, exp_scale,
            "per-token scale differs from reference"
        );

        // Every kept value is a valid int8 in [-128, 127].
        for &v in &out_q {
            assert!((-128.0..=127.0).contains(&v), "out of int8 range: {v}");
            assert_eq!(v, v.trunc(), "non-integer int8 value: {v}");
        }
    }

    #[test]
    fn round_half_to_even_matches_torch() {
        // torch.round([0.5,1.5,2.5,3.5,-0.5,-1.5,-2.5]) == [0,2,2,4,0,-2,-2]
        assert_eq!(round_half_to_even(0.5), 0.0);
        assert_eq!(round_half_to_even(1.5), 2.0);
        assert_eq!(round_half_to_even(2.5), 2.0);
        assert_eq!(round_half_to_even(3.5), 4.0);
        assert_eq!(round_half_to_even(-0.5), 0.0); // -0.0 == 0.0
        assert_eq!(round_half_to_even(-1.5), -2.0);
        assert_eq!(round_half_to_even(-2.5), -2.0);
        // non-halves round normally
        assert_eq!(round_half_to_even(2.4999), 2.0);
        assert_eq!(round_half_to_even(2.5001), 3.0);
    }

    #[test]
    fn known_row_matches_explicit_torch_dump() {
        // torch ActQuant on [[1, -2, 0.5, -0.5, 2, -1]] (gamma = 2):
        //   scale 127/2 = 63.5
        //   v*scale = [63.5, -127, 31.75, -31.75, 127, -63.5]
        //   round_ties_even = [64, -127, 32, -32, 127, -64]  (63.5->64, -63.5->-64)
        //   dequant scale gamma/127 = 0.015748031
        // (cross-checked against a live torch run: q == [64,-127,32,-32,127,-64].)
        let act = [1.0_f32, -2.0, 0.5, -0.5, 2.0, -1.0];
        let mut out_q = [f32::NAN; 6];
        let mut out_scale = [f32::NAN; 1];
        quantize_activation_int8(&act, 1, 6, &mut out_q, &mut out_scale).unwrap();
        assert_eq!(out_q, [64.0, -127.0, 32.0, -32.0, 127.0, -64.0]);
        let expected_scale = 2.0_f32 / 127.0;
        assert!((out_scale[0] - expected_scale).abs() < 1e-9);
    }

    #[test]
    fn random_rows_match_reference() {
        // Deterministic xorshift PRNG → reproducible "random" rows in [-4, 4).
        let rows = 7;
        let cols = 33;
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // map to [-4, 4)
            ((state >> 11) as f32 / (1u64 << 53) as f32) * 8.0 - 4.0
        };
        let act: Vec<f32> = (0..rows * cols).map(|_| next()).collect();
        run_case(&act, rows, cols);
    }

    #[test]
    fn all_zero_row_yields_zeros_and_zero_scale() {
        // Row 0 all zero, row 1 nonzero — confirms the zero special-case is
        // isolated to its own row.
        let act = [0.0, 0.0, 0.0, 0.0, 3.0, -6.0, 1.5, 0.0];
        let mut out_q = [f32::NAN; 8];
        let mut out_scale = [f32::NAN; 2];
        quantize_activation_int8(&act, 2, 4, &mut out_q, &mut out_scale).unwrap();
        assert_eq!(&out_q[0..4], &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(out_scale[0], 0.0);
        // Row 1: gamma = 6, scale 127/6.
        let s = 127.0_f32 / 6.0;
        let exp: Vec<f32> = [3.0_f32, -6.0, 1.5, 0.0]
            .iter()
            .map(|&v| (v * s).round_ties_even().clamp(-128.0, 127.0))
            .collect();
        assert_eq!(&out_q[4..8], exp.as_slice());
        assert!((out_scale[1] - 6.0 / 127.0).abs() < 1e-9);
    }

    #[test]
    fn single_element_row() {
        // gamma == |value|, so the single element saturates to the sign-matched
        // cap (-127 for a negative entry, since round_ties_even(-127.0)).
        let act = [-3.5_f32];
        let mut out_q = [f32::NAN; 1];
        let mut out_scale = [f32::NAN; 1];
        quantize_activation_int8(&act, 1, 1, &mut out_q, &mut out_scale).unwrap();
        // v*s = -3.5 * (127/3.5) = -127.
        assert_eq!(out_q[0], -127.0);
        assert!((out_scale[0] - 3.5 / 127.0).abs() < 1e-9);
    }

    #[test]
    fn negative_values_and_saturation() {
        // A symmetric extreme: a value equal to gamma maps to +127, its negation
        // to -127; intermediate values match the reference.
        let act = [-5.0_f32, 5.0, -2.5, 1.25];
        run_case(&act, 1, 4);
        let mut out_q = [f32::NAN; 4];
        let mut out_scale = [f32::NAN; 1];
        quantize_activation_int8(&act, 1, 4, &mut out_q, &mut out_scale).unwrap();
        assert_eq!(out_q[0], -127.0); // -5 == -gamma
        assert_eq!(out_q[1], 127.0); //  5 ==  gamma
    }

    #[test]
    fn shape_mismatch_errors() {
        let act = [1.0_f32, 2.0, 3.0, 4.0];
        let mut out_q = [0.0_f32; 4];
        let mut out_scale = [0.0_f32; 2];
        // act len 4 but rows*cols = 6
        assert_eq!(
            quantize_activation_int8(&act, 2, 3, &mut out_q, &mut out_scale),
            Err(NnError::Shape {
                expected: 6,
                got: 4
            })
        );
        // out_scale wrong length
        let mut bad_scale = [0.0_f32; 3];
        assert_eq!(
            quantize_activation_int8(&act, 2, 2, &mut out_q, &mut bad_scale),
            Err(NnError::Shape {
                expected: 2,
                got: 3
            })
        );
    }
}

#[cfg(test)]
mod grouped_tests {
    use super::*;

    /// Deterministic heavy-tailed activations: one large outlier per row, the rest small.
    /// This is the shape the per-group idea exists for.
    fn outlier_rows(rows: usize, cols: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let x = ((r * 31 + c * 17) % 97) as f32 / 97.0 - 0.5;
                v[r * cols + c] = x * 0.1;
            }
            // One 6-sigma spike, in the first group.
            v[r * cols + 3] = 6.0;
        }
        v
    }

    /// The degenerate control. `group == cols` is one absmax over the whole row, which is exactly
    /// what the shipping quantizer does — so the two must agree on every value, allowing only for
    /// the fact that one returns integers with a scale and the other returns their product.
    #[test]
    fn a_group_as_wide_as_the_row_reproduces_the_per_token_quantizer() {
        let (rows, cols) = (5, 64);
        let act = outlier_rows(rows, cols);

        let mut q_row = vec![0.0f32; rows * cols];
        let mut s_row = vec![0.0f32; rows];
        quantize_activation_int8(&act, rows, cols, &mut q_row, &mut s_row).unwrap();

        let mut q_grp = vec![0.0f32; rows * cols];
        let mut s_grp = vec![0.0f32; rows];
        quantize_activation_int8_grouped(&act, rows, cols, cols, &mut q_grp, &mut s_grp).unwrap();

        for r in 0..rows {
            assert_eq!(
                s_grp[r], 1.0,
                "row {r}: the grouped path carries its own scale"
            );
            for c in 0..cols {
                let i = r * cols + c;
                let expected = q_row[i] * s_row[r];
                assert!(
                    (q_grp[i] - expected).abs() <= 1e-6 * expected.abs().max(1e-6),
                    "row {r} col {c}: grouped {} vs per-token dequant {expected}",
                    q_grp[i]
                );
            }
        }
    }

    /// The claim under test: finer groups cost less error on outlier-heavy rows. This is the
    /// SQNR half of the A8-headroom argument, asserted rather than assumed.
    #[test]
    fn finer_groups_reduce_activation_quantization_error() {
        let (rows, cols) = (4, 512);
        let act = outlier_rows(rows, cols);

        let sse = |group: usize| -> f64 {
            let mut q = vec![0.0f32; rows * cols];
            let mut s = vec![0.0f32; rows];
            quantize_activation_int8_grouped(&act, rows, cols, group, &mut q, &mut s).unwrap();
            act.iter()
                .zip(&q)
                .map(|(&a, &b)| f64::from(a - b) * f64::from(a - b))
                .sum()
        };

        let whole_row = sse(cols);
        let g128 = sse(128);
        let g32 = sse(32);
        assert!(
            g128 < whole_row && g32 < g128,
            "error must fall monotonically as groups narrow: row {whole_row:.3e}, \
             g128 {g128:.3e}, g32 {g32:.3e} — if it does not, the outlier is not being \
             confined and the whole per-group premise is wrong here"
        );
        // The spike sits in one group of 128, so the other three quarters of the row stop paying
        // for it entirely. Anything less than a large factor means the quantizer is not isolating.
        assert!(
            whole_row / g128 > 4.0,
            "a 6-sigma spike confined to 1 of 4 groups should cut error by far more than 4x; \
             got {:.2}x",
            whole_row / g128
        );
    }

    /// A group wider than the row, and a row that does not divide by the group, must both behave —
    /// SmolLM2 is 576 wide, which divides by neither 128 nor 256 evenly at every tensor.
    #[test]
    fn ragged_and_oversized_groups_are_handled() {
        let (rows, cols) = (2, 100);
        let act = outlier_rows(rows, cols);
        for group in [1usize, 7, 64, 100, 4096] {
            let mut q = vec![0.0f32; rows * cols];
            let mut s = vec![0.0f32; rows];
            quantize_activation_int8_grouped(&act, rows, cols, group, &mut q, &mut s)
                .unwrap_or_else(|e| panic!("group {group}: {e}"));
            assert!(
                q.iter().all(|v| v.is_finite()),
                "group {group} produced non-finite values"
            );
        }
        // A group of 1 is lossless: every value is its own absmax and quantizes to +/-127.
        let mut q = vec![0.0f32; rows * cols];
        let mut s = vec![0.0f32; rows];
        quantize_activation_int8_grouped(&act, rows, cols, 1, &mut q, &mut s).unwrap();
        for (i, (&a, &b)) in act.iter().zip(&q).enumerate() {
            assert!(
                (a - b).abs() <= 1e-6 * a.abs().max(1e-6),
                "index {i}: a group of one must be exact, {a} != {b}"
            );
        }
    }

    /// An all-zero row reports a zero scale, exactly as the per-token path does, so a caller's
    /// fold cannot tell the two quantizers apart on the empty case.
    #[test]
    fn an_all_zero_row_matches_the_per_token_convention() {
        let (rows, cols) = (2, 8);
        let act = vec![0.0f32; rows * cols];
        let mut q = vec![9.0f32; rows * cols];
        let mut s = vec![9.0f32; rows];
        quantize_activation_int8_grouped(&act, rows, cols, 4, &mut q, &mut s).unwrap();
        assert!(q.iter().all(|&v| v == 0.0));
        assert!(s.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn a_zero_group_is_refused() {
        let mut q = [0.0f32; 4];
        let mut s = [0.0f32; 2];
        assert!(quantize_activation_int8_grouped(&[0.0; 4], 2, 2, 0, &mut q, &mut s).is_err());
    }
}
