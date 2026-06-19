//! `DenseLinear`: an fp32 linear run through BitNet's **A8 activation path** — the
//! reference projection for the SALT accuracy curve (ADR 0006).
//!
//! It quantizes activations to int8 per token exactly as [`TernaryLinear`] does
//! ([`quantize_activation_int8`]), then contracts against a **dense fp32** weight
//! instead of a ternary one:
//!
//! ```text
//! y[m, n] = act_scale[m] · Σ_k q_act[m, k] · W[n, k]
//! ```
//!
//! Swapping a ternary projection for this one therefore isolates the *weight*
//! quantization effect: the activation path is identical, so a perplexity
//! difference is attributable purely to the weights. `W` is either the fp master
//! weights (the upper-bound reference) or a SALT plane-stack dequantized via
//! [`tritium_format::dequant_salt_row`] (the SALT point at a given bpw).

use crate::error::NnError;
use crate::ops::quantize_activation_int8;

/// A dense fp32 projection through the A8 activation path. `W` is row-major
/// `[n_out, k_in]`.
#[derive(Clone, Debug)]
pub struct DenseLinear {
    /// Output feature count (`N`).
    pub n_out: usize,
    /// Input feature count (`K`).
    pub k_in: usize,
    /// Row-major `[n_out, k_in]` fp32 weights.
    pub weights: Vec<f32>,
}

impl DenseLinear {
    /// Build from row-major `[n_out, k_in]` fp32 weights.
    ///
    /// # Errors
    /// [`NnError::Shape`] if `weights.len() != n_out * k_in`.
    pub fn new(weights: Vec<f32>, n_out: usize, k_in: usize) -> Result<Self, NnError> {
        let expected = n_out.checked_mul(k_in).ok_or(NnError::Shape {
            expected: usize::MAX,
            got: weights.len(),
        })?;
        if weights.len() != expected {
            return Err(NnError::Shape {
                expected,
                got: weights.len(),
            });
        }
        Ok(Self {
            n_out,
            k_in,
            weights,
        })
    }

    /// GEMM-free forward: quantize `act` (`[m, k_in]`) to int8 per token, contract
    /// against the dense weights, and fold the per-token dequant scale into `out`
    /// (`[m, n_out]`).
    ///
    /// # Errors
    /// [`NnError::Shape`] on buffer-length mismatch.
    pub fn forward(&self, act: &[f32], m: usize, out: &mut [f32]) -> Result<(), NnError> {
        let k = self.k_in;
        let act_len = m * k;
        if act.len() != act_len {
            return Err(NnError::Shape {
                expected: act_len,
                got: act.len(),
            });
        }
        let out_len = m * self.n_out;
        if out.len() != out_len {
            return Err(NnError::Shape {
                expected: out_len,
                got: out.len(),
            });
        }

        // Same per-token int8 absmax quant as `TernaryLinear` — the apples-to-apples
        // activation path. `q_act` is the int8 values kept as f32.
        let mut q_act = vec![0.0f32; act_len];
        let mut act_scale = vec![0.0f32; m];
        quantize_activation_int8(act, m, k, &mut q_act, &mut act_scale)?;

        for r in 0..m {
            let qrow = &q_act[r * k..r * k + k];
            let s = act_scale[r];
            for n in 0..self.n_out {
                let wn = &self.weights[n * k..n * k + k];
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += qrow[kk] * wn[kk];
                }
                out[r * self.n_out + n] = acc * s;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_bad_shape() {
        assert!(matches!(
            DenseLinear::new(vec![0.0; 5], 2, 3),
            Err(NnError::Shape { .. })
        ));
        assert!(DenseLinear::new(vec![0.0; 6], 2, 3).is_ok());
    }

    // The forward must equal an independent int8-quant + matmul + fold reference
    // built from the same `quantize_activation_int8` (validates the matmul indexing
    // and the per-token scale fold — the logic DenseLinear owns).
    #[test]
    fn forward_matches_reference() {
        let (m, n, k) = (2usize, 3usize, 5usize);
        // deterministic weights + activations
        let weights: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.13).sin()).collect();
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.27).cos() * 3.0).collect();
        let lin = DenseLinear::new(weights.clone(), n, k).unwrap();

        let mut out = vec![0.0f32; m * n];
        lin.forward(&act, m, &mut out).unwrap();

        let mut q = vec![0.0f32; m * k];
        let mut sc = vec![0.0f32; m];
        quantize_activation_int8(&act, m, k, &mut q, &mut sc).unwrap();
        for r in 0..m {
            for nn in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += q[r * k + kk] * weights[nn * k + kk];
                }
                let want = acc * sc[r];
                assert_eq!(out[r * n + nn].to_bits(), want.to_bits(), "r={r} n={nn}");
            }
        }
    }
}
