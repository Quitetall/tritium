//! Reconstruction fidelity of a SALT-quantized tensor against its fp source.
//!
//! The arch-agnostic measurement behind a SALT bpw/sensitivity sweep: how far the
//! dequantized planes drift from the original weights. MSE alone hides *direction*
//! error (a uniformly shrunk tensor keeps a high cosine but a large MSE), so this
//! reports the relative Frobenius error and cosine similarity alongside it — the
//! closest weight-space proxies for downstream output divergence without a forward
//! pass. True output KL is a model-forward measurement layered on top of this.

use tritium_format::{FormatError, dequant_salt_row};

use crate::QuantizedTensor;

/// Fidelity of a dequantized SALT tensor vs the original fp weights. All metrics
/// are over the flattened row-major matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReconStats {
    /// Mean squared error, `mean((W − Ŵ)²)`.
    pub mse: f64,
    /// Root mean squared error, `√mse`.
    pub rmse: f64,
    /// Mean absolute error, `mean(|W − Ŵ|)`.
    pub mae: f64,
    /// Largest absolute element error, `max|W − Ŵ|` (worst-case weight miss).
    pub max_abs: f64,
    /// Relative Frobenius error, `‖W − Ŵ‖₂ / ‖W‖₂` (0 = perfect; ≈1 = no better
    /// than the zero matrix). Scale-invariant, so it is comparable across tensors.
    pub frob_rel: f64,
    /// Cosine similarity of flattened `W` and `Ŵ` (1 = identical direction). Pure
    /// direction error — insensitive to a global magnitude miss the way `frob_rel`
    /// is not.
    pub cosine: f64,
}

/// Raw reconstruction moments, accumulated over one or many tensors. Folding tensors
/// into one accumulator and then [`finish`](ReconAccum::finish)ing yields a *true*
/// whole-model metric — `frob_rel` and `cosine` are ratios of summed moments and
/// cannot be recovered by averaging per-tensor [`ReconStats`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ReconAccum {
    /// Elements folded in so far.
    pub count: u64,
    /// `Σ (W − Ŵ)²`.
    pub sq_err: f64,
    /// `Σ |W − Ŵ|`.
    pub abs_err: f64,
    /// `max |W − Ŵ|`.
    pub max_abs: f64,
    /// `Σ W·Ŵ`.
    pub dot: f64,
    /// `Σ W²`.
    pub norm_orig: f64,
    /// `Σ Ŵ²`.
    pub norm_approx: f64,
}

impl ReconAccum {
    /// Fold the original-vs-dequantized error of one SALT tensor into this accumulator.
    ///
    /// # Errors
    /// [`ReconError::ShapeMismatch`] if `original.len() != q.rows · q.k`; propagates
    /// [`FormatError`] from row dequant.
    pub fn accumulate(&mut self, original: &[f32], q: &QuantizedTensor) -> Result<(), ReconError> {
        let n = q.rows.saturating_mul(q.k);
        if original.len() != n {
            return Err(ReconError::ShapeMismatch {
                expected: n,
                got: original.len(),
            });
        }
        for (r, row) in q.salt_rows.iter().enumerate() {
            let approx = dequant_salt_row(row)?;
            // A dequantized row must be exactly `k` wide; guard so a future SaltRow/format
            // drift surfaces as an error rather than silently truncating the zip (which
            // would deflate every moment while `count` still tallies the full `rows·k`).
            if approx.len() != q.k {
                return Err(ReconError::RowLen {
                    expected: q.k,
                    got: approx.len(),
                });
            }
            let orig = &original[r * q.k..r * q.k + q.k];
            for (&a, &b) in orig.iter().zip(&approx) {
                let (a, b) = (a as f64, b as f64);
                let d = a - b;
                self.sq_err += d * d;
                self.abs_err += d.abs();
                self.max_abs = self.max_abs.max(d.abs());
                self.dot += a * b;
                self.norm_orig += a * a;
                self.norm_approx += b * b;
            }
        }
        self.count += n as u64;
        Ok(())
    }

    /// Merge another accumulator's moments (associative — order-independent).
    pub fn merge(&mut self, other: &ReconAccum) {
        self.count += other.count;
        self.sq_err += other.sq_err;
        self.abs_err += other.abs_err;
        self.max_abs = self.max_abs.max(other.max_abs);
        self.dot += other.dot;
        self.norm_orig += other.norm_orig;
        self.norm_approx += other.norm_approx;
    }

    /// Reduce the moments to reported [`ReconStats`]. An empty/all-zero original is
    /// defined as a perfect fit (`cosine = 1`, `frob_rel = 0`) to avoid a 0/0.
    pub fn finish(&self) -> ReconStats {
        if self.count == 0 {
            return ReconStats {
                mse: 0.0,
                rmse: 0.0,
                mae: 0.0,
                max_abs: 0.0,
                frob_rel: 0.0,
                cosine: 1.0,
            };
        }
        let n = self.count as f64;
        let mse = self.sq_err / n;
        let frob_rel = if self.norm_orig > 0.0 {
            (self.sq_err / self.norm_orig).sqrt()
        } else {
            0.0
        };
        let cosine = if self.norm_orig > 0.0 && self.norm_approx > 0.0 {
            self.dot / (self.norm_orig.sqrt() * self.norm_approx.sqrt())
        } else if self.norm_orig == 0.0 && self.norm_approx == 0.0 {
            1.0
        } else {
            0.0
        };
        ReconStats {
            mse,
            rmse: mse.sqrt(),
            mae: self.abs_err / n,
            max_abs: self.max_abs,
            frob_rel,
            cosine,
        }
    }
}

/// Why reconstruction stats could not be computed.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ReconError {
    /// `original.len()` ≠ `q.rows · q.k`.
    ShapeMismatch {
        /// Expected element count (`rows · k`).
        expected: usize,
        /// Actual `original.len()`.
        got: usize,
    },
    /// A dequantized SALT row was not `k` wide (format/contract drift).
    RowLen {
        /// Expected row width (`k`).
        expected: usize,
        /// Actual dequantized row length.
        got: usize,
    },
    /// A SALT row failed to dequantize.
    Format(FormatError),
}

impl core::fmt::Display for ReconError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ReconError::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {expected} weights, got {got}")
            }
            ReconError::RowLen { expected, got } => {
                write!(f, "dequant row length: expected {expected} (k), got {got}")
            }
            ReconError::Format(e) => write!(f, "dequant: {e}"),
        }
    }
}

impl std::error::Error for ReconError {}

impl From<FormatError> for ReconError {
    fn from(e: FormatError) -> Self {
        ReconError::Format(e)
    }
}

/// Reconstruction fidelity of `q` (dequantized via [`dequant_salt_row`]) against the
/// original row-major `q.rows × q.k` `original` weights.
///
/// An all-zero original yields `cosine = 1`, `frob_rel = 0` (the zero matrix is
/// reconstructed exactly), avoiding a 0/0.
///
/// # Errors
/// [`ReconError::ShapeMismatch`] if `original.len() != q.rows · q.k`; propagates
/// [`FormatError`] from row dequant.
pub fn reconstruction_stats(
    original: &[f32],
    q: &QuantizedTensor,
) -> Result<ReconStats, ReconError> {
    let mut accum = ReconAccum::default();
    accum.accumulate(original, q)?;
    Ok(accum.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{QuantConfig, quantize_tensor};

    fn stats_at(w: &[f32], rows: usize, k: usize, cfg: &QuantConfig) -> ReconStats {
        let qt = quantize_tensor(w, rows, k, cfg).expect("quantize_tensor");
        reconstruction_stats(w, &qt).expect("reconstruction_stats")
    }

    /// Weights already on a ternary lattice with no zeros (`±s`) reconstruct exactly at
    /// the T=1 floor: AbsMean = `s`, every trit is `±1`, dequant = `±s` (only the f16
    /// scale rounds). So error ≈ 0 and cosine ≈ 1 — pins the metric's correctness.
    #[test]
    fn ternary_input_reconstructs_at_base() {
        let (rows, k) = (2usize, 256usize);
        let w: Vec<f32> = (0..rows * k)
            .map(|i| if i % 2 == 0 { 3.0 } else { -3.0 })
            .collect();
        let cfg = QuantConfig {
            budget_bpw: crate::TRIT_BITS,
            ..Default::default()
        };
        let s = stats_at(&w, rows, k, &cfg);
        assert!(s.mse < 1e-3, "near-exact mse, got {}", s.mse);
        assert!(s.frob_rel < 1e-2, "near-exact frob_rel, got {}", s.frob_rel);
        assert!(s.cosine > 0.999, "near-exact cosine, got {}", s.cosine);
    }

    /// More bpw ⇒ strictly better reconstruction: lower MSE / frob_rel, higher cosine.
    /// Mirrors the monotonicity gate but on the dequant path the report actually uses.
    #[test]
    fn higher_bpw_improves_fidelity() {
        let (rows, k) = (4usize, 512usize);
        let w: Vec<f32> = (0..rows * k)
            .map(|i| (i as f32 * 0.013).sin() * ((i % 7) as f32 - 3.0))
            .collect();
        let low = stats_at(
            &w,
            rows,
            k,
            &QuantConfig {
                budget_bpw: crate::TRIT_BITS,
                ..Default::default()
            },
        );
        let high = stats_at(
            &w,
            rows,
            k,
            &QuantConfig {
                budget_bpw: 4.0,
                ..Default::default()
            },
        );
        assert!(high.mse < low.mse, "mse {} !< {}", high.mse, low.mse);
        assert!(
            high.frob_rel < low.frob_rel,
            "frob_rel {} !< {}",
            high.frob_rel,
            low.frob_rel
        );
        assert!(
            high.cosine > low.cosine,
            "cosine {} !> {}",
            high.cosine,
            low.cosine
        );
        assert!(low.cosine > 0.0 && high.cosine <= 1.0 + 1e-9);
    }

    /// An all-zero tensor reconstructs exactly: 0 error, cosine defined as 1 (no 0/0).
    #[test]
    fn zero_tensor_is_perfect() {
        let (rows, k) = (2usize, 256usize);
        let w = vec![0.0f32; rows * k];
        let s = stats_at(
            &w,
            rows,
            k,
            &QuantConfig {
                budget_bpw: crate::TRIT_BITS,
                ..Default::default()
            },
        );
        assert_eq!(s.mse, 0.0);
        assert_eq!(s.frob_rel, 0.0);
        assert_eq!(s.cosine, 1.0);
    }

    /// Folding two tensors into one accumulator equals merging two separate
    /// accumulators — the associativity the per-shard model report depends on.
    #[test]
    fn accum_merge_matches_sequential() {
        let cfg = QuantConfig {
            budget_bpw: 2.5,
            ..Default::default()
        };
        let (r1, k1) = (3usize, 256usize);
        let w1: Vec<f32> = (0..r1 * k1).map(|i| (i as f32 * 0.07).cos()).collect();
        let q1 = quantize_tensor(&w1, r1, k1, &cfg).unwrap();
        let (r2, k2) = (2usize, 512usize);
        let w2: Vec<f32> = (0..r2 * k2)
            .map(|i| (i as f32 * 0.03).sin() * 2.0)
            .collect();
        let q2 = quantize_tensor(&w2, r2, k2, &cfg).unwrap();

        let mut seq = ReconAccum::default();
        seq.accumulate(&w1, &q1).unwrap();
        seq.accumulate(&w2, &q2).unwrap();

        let mut a = ReconAccum::default();
        a.accumulate(&w1, &q1).unwrap();
        let mut b = ReconAccum::default();
        b.accumulate(&w2, &q2).unwrap();
        a.merge(&b);

        // count + max_abs are order-independent (exact); the summed moments match to
        // within floating-point reassociation (fp `+` is not associative, so summing
        // into one total vs merging two subtotals differs by a few ULP).
        assert_eq!(seq.count, (r1 * k1 + r2 * k2) as u64);
        assert_eq!(seq.count, a.count);
        assert_eq!(seq.max_abs, a.max_abs);
        let close = |x: f64, y: f64| (x - y).abs() <= 1e-9 * x.abs().max(1.0);
        assert!(
            close(seq.sq_err, a.sq_err),
            "{} vs {}",
            seq.sq_err,
            a.sq_err
        );
        assert!(close(seq.abs_err, a.abs_err));
        assert!(close(seq.dot, a.dot));
        assert!(close(seq.norm_orig, a.norm_orig));
        assert!(close(seq.norm_approx, a.norm_approx));
        // And the reduced stats agree to the same tolerance.
        assert!(close(seq.finish().frob_rel, a.finish().frob_rel));
        assert!(close(seq.finish().cosine, a.finish().cosine));
    }

    /// Shape guard fires before any dequant.
    #[test]
    fn shape_mismatch_errors() {
        let (rows, k) = (2usize, 256usize);
        let w: Vec<f32> = (0..rows * k).map(|i| (i % 5) as f32 - 2.0).collect();
        let qt = quantize_tensor(&w, rows, k, &QuantConfig::default()).unwrap();
        let err = reconstruction_stats(&w[..rows * k - 1], &qt).unwrap_err();
        assert_eq!(
            err,
            ReconError::ShapeMismatch {
                expected: rows * k,
                got: rows * k - 1
            }
        );
    }
}
