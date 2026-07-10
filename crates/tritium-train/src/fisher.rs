//! Diagonal-Fisher accumulation from reverse-mode gradients (plan 0039).
//!
//! The diagonal of the Fisher information, `F_i = E[(∂L/∂w_i)²]`, is the loss-curvature signal the
//! SALT allocator wants for `Sensitivity::Custom` (via `tritium_quantize::fisher::tile_sensitivity`).
//! Accumulate one squared gradient per data sample from [`Tape::backward`](crate::Tape::backward),
//! then take the mean.

/// Accumulates the diagonal Fisher `F_i = E[(∂L/∂w_i)²]` for one parameter tensor across samples.
#[derive(Clone, Debug)]
pub struct FisherAccumulator {
    sum: Vec<f64>,
    count: u64,
}

impl FisherAccumulator {
    /// A zeroed accumulator for a `len`-element parameter tensor.
    pub fn new(len: usize) -> Self {
        Self {
            sum: vec![0.0; len],
            count: 0,
        }
    }

    /// Add one sample's gradient `g` (accumulates `g_i²`). Panics if `g.len()` ≠ the tensor length.
    pub fn accumulate(&mut self, g: &[f32]) {
        assert_eq!(
            g.len(),
            self.sum.len(),
            "gradient length {} ≠ tensor length {}",
            g.len(),
            self.sum.len()
        );
        for (s, &gi) in self.sum.iter_mut().zip(g) {
            *s += f64::from(gi) * f64::from(gi);
        }
        self.count += 1;
    }

    /// Number of accumulated samples.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// The diagonal Fisher estimate `E[g²]` (mean of squared gradients). All-zero if no samples.
    pub fn into_diag(self) -> Vec<f64> {
        if self.count == 0 {
            return self.sum;
        }
        let n = self.count as f64;
        self.sum.into_iter().map(|s| s / n).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulates_mean_of_squared_gradients() {
        let mut f = FisherAccumulator::new(3);
        f.accumulate(&[1.0, 2.0, 3.0]); // g² = [1, 4, 9]
        f.accumulate(&[1.0, 0.0, 1.0]); // g² = [1, 0, 1]  → sums [2, 4, 10]
        assert_eq!(f.count(), 2);
        let diag = f.into_diag();
        assert_eq!(diag, vec![1.0, 2.0, 5.0]); // means
    }

    #[test]
    fn empty_accumulator_is_all_zero() {
        assert_eq!(FisherAccumulator::new(4).into_diag(), vec![0.0; 4]);
    }

    #[test]
    #[should_panic(expected = "gradient length")]
    fn rejects_mismatched_gradient_length() {
        FisherAccumulator::new(3).accumulate(&[1.0, 2.0]);
    }
}
