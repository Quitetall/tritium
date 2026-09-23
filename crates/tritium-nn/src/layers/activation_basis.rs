//! A tensor's input basis: the signed block Walsh–Hadamard rotation a folded ternary tensor was
//! fitted under.
//!
//! A rotated tensor stores `W·Bᵀ`, with `B = H·D` block-diagonal: `D` is a fixed ±1 sign vector
//! across the full input width and `H` the normalized Sylvester–Walsh–Hadamard transform applied
//! independently to each `block`-wide slice ([`tritium_core::fwht_normalized`]). Feeding such a
//! tensor the rotated activation `B·x` recovers `W·x` exactly in exact arithmetic.
//!
//! The same object answers the two consumers a tensor can have, and they need opposite
//! directions:
//!
//! - a **projection** rotates its input: [`SignedBlockHadamard::rotate`];
//! - a **gather** (a token embedding) reads a stored row `B·w` and must hand the residual stream
//!   `w`, so it un-rotates the row it read: [`SignedBlockHadamard::unrotate`].
//!
//! Holding the basis on the tensor, rather than as a width passed alongside it, is what makes the
//! wrong direction unrepresentable (ADR 0044 D3).

use std::sync::Arc;

use tritium_core::fwht_normalized;

use crate::error::NnError;

/// Signed, block-diagonal, normalized Walsh–Hadamard basis over one input width.
#[derive(Clone, Debug, PartialEq)]
pub struct SignedBlockHadamard {
    block: usize,
    signs: Arc<[f32]>,
}

impl SignedBlockHadamard {
    /// Build a basis from its block width and a ±1 sign per input feature.
    ///
    /// # Errors
    /// [`NnError::Backend`] if `block` is not a power of two, the width is empty or not a whole
    /// number of blocks, or any sign is not exactly `+1` or `-1`.
    pub fn new(block: usize, signs: Vec<f32>) -> Result<Self, NnError> {
        if !block.is_power_of_two() {
            return Err(NnError::Backend(format!(
                "Hadamard block {block} is not a power of two"
            )));
        }
        if signs.is_empty() || !signs.len().is_multiple_of(block) {
            return Err(NnError::Backend(format!(
                "Hadamard width {} is not a whole number of {block}-wide blocks",
                signs.len()
            )));
        }
        if let Some(index) = signs.iter().position(|s| *s != 1.0 && *s != -1.0) {
            return Err(NnError::Backend(format!(
                "Hadamard sign {index} is {}, not +/-1",
                signs[index]
            )));
        }
        Ok(Self {
            block,
            signs: signs.into(),
        })
    }

    /// Input width this basis covers.
    #[must_use]
    pub fn width(&self) -> usize {
        self.signs.len()
    }

    /// Transform block width.
    #[must_use]
    pub const fn block(&self) -> usize {
        self.block
    }

    /// Rotate one input row in place: `x ← H·(D·x)`. This is what a projection feeds its tensor.
    ///
    /// # Errors
    /// [`NnError::Shape`] if `row` is not exactly [`Self::width`] long.
    pub fn rotate(&self, row: &mut [f32]) -> Result<(), NnError> {
        self.check(row.len())?;
        for (value, sign) in row.iter_mut().zip(self.signs.iter()) {
            *value *= sign;
        }
        for slice in row.chunks_exact_mut(self.block) {
            fwht_normalized(slice);
        }
        Ok(())
    }

    /// Undo [`Self::rotate`] in place: `w ← D·(H·w)`. This is what a gather applies to a stored
    /// row before it enters the residual stream.
    ///
    /// # Errors
    /// [`NnError::Shape`] if `row` is not exactly [`Self::width`] long.
    pub fn unrotate(&self, row: &mut [f32]) -> Result<(), NnError> {
        self.check(row.len())?;
        for slice in row.chunks_exact_mut(self.block) {
            fwht_normalized(slice);
        }
        for (value, sign) in row.iter_mut().zip(self.signs.iter()) {
            *value *= sign;
        }
        Ok(())
    }

    fn check(&self, len: usize) -> Result<(), NnError> {
        if len == self.signs.len() {
            Ok(())
        } else {
            Err(NnError::Shape {
                expected: self.signs.len(),
                got: len,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signs(width: usize) -> Vec<f32> {
        (0..width)
            .map(|i| if (i * 7 + 3) % 5 < 2 { -1.0 } else { 1.0 })
            .collect()
    }

    #[test]
    fn unrotate_inverts_rotate() {
        let basis = SignedBlockHadamard::new(8, signs(24)).unwrap();
        let original: Vec<f32> = (0..24).map(|i| (i as f32 - 11.5) / 3.0).collect();
        let mut row = original.clone();
        basis.rotate(&mut row).unwrap();
        assert_ne!(row, original, "a nontrivial basis must move the row");
        basis.unrotate(&mut row).unwrap();
        for (a, b) in row.iter().zip(&original) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    /// The contract a projection relies on: a tensor stored as `W·Bᵀ` and fed `B·x` gives `W·x`.
    #[test]
    fn a_folded_dot_product_recovers_the_original() {
        let basis = SignedBlockHadamard::new(4, signs(8)).unwrap();
        let w: Vec<f32> = (0..8).map(|i| (i as f32) * 0.5 - 1.0).collect();
        let x: Vec<f32> = (0..8).map(|i| 1.0 - (i as f32) * 0.25).collect();
        let want: f32 = w.iter().zip(&x).map(|(a, b)| a * b).sum();
        // B is orthogonal, so the stored row B·w (as a row vector, w·Bᵀ) pairs with B·x.
        let mut stored = w.clone();
        basis.rotate(&mut stored).unwrap();
        let mut fed = x.clone();
        basis.rotate(&mut fed).unwrap();
        let got: f32 = stored.iter().zip(&fed).map(|(a, b)| a * b).sum();
        assert!((got - want).abs() < 1e-5, "{got} vs {want}");
    }

    #[test]
    fn malformed_bases_are_rejected() {
        assert!(SignedBlockHadamard::new(6, signs(12)).is_err());
        assert!(SignedBlockHadamard::new(8, signs(12)).is_err());
        assert!(SignedBlockHadamard::new(4, vec![1.0, -1.0, 0.5, 1.0]).is_err());
        let basis = SignedBlockHadamard::new(4, signs(8)).unwrap();
        assert!(basis.rotate(&mut [0.0; 4]).is_err());
    }
}
