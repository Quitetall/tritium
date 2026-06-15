//! Scaling schemes. Ternary weights are dimensionless `{-1,0,1}`; a per-block /
//! per-channel scale restores magnitude.

/// How a scale factor is shared across a weight tensor.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum ScaleGranularity {
    /// One scale for the whole tensor (BitNet b1.58 default).
    PerTensor,
    /// One scale per output channel (row of the `[N, K]` weight).
    PerChannel,
    /// One scale per contiguous group of `n` weights along `K`.
    PerGroup(u32),
}

impl ScaleGranularity {
    /// Number of scale factors needed for an `[n_out, k_in]` weight tensor.
    pub const fn scale_count(self, n_out: usize, k_in: usize) -> usize {
        match self {
            ScaleGranularity::PerTensor => 1,
            ScaleGranularity::PerChannel => n_out,
            ScaleGranularity::PerGroup(g) => {
                let g = g as usize;
                // ceil(k_in / g) groups per row, one scale each.
                n_out * k_in.div_ceil(g)
            }
        }
    }
}

/// AbsMean scale — the BitNet b1.58 quantization scale for a slice of weights:
/// `scale = mean(|w|)`. Quantize with `round(w / scale)` clamped to `{-1,0,1}`;
/// dequantize with `trit * scale`.
///
/// Returns `0.0` for an empty slice (degenerate; caller should guard).
pub fn absmean(weights: &[f32]) -> f32 {
    if weights.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for &w in weights {
        sum += w.abs();
    }
    sum / weights.len() as f32
}
