//! Token sampling: greedy (argmax) now; temperature top-k / top-p in WF-2.
//!
//! Sampling that draws from a distribution needs randomness. To keep
//! `tritium-nn` dependency-light and deterministic for tests, the stochastic
//! samplers take an explicit `seed` and use a small inline PRNG (no `rand`
//! crate). The exact PRNG + tie-break rules are fixed in WF-2 so they match the
//! reference oracle.

use core::cmp::Ordering;

/// Greedy sampling: the index of the maximum logit (NaN-tolerant; NaNs lose).
/// Returns `None` for empty logits.
#[must_use]
pub fn sample_greedy(logits: &[f32]) -> Option<u32> {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Less))
        .map(|(i, _)| i as u32)
}

/// Temperature top-k sampling.
///
/// Restricts the candidate set to the `k` highest logits, applies temperature
/// `temp` (logits divided by `temp` before the softmax), then draws one token
/// using a deterministic PRNG seeded by `seed`. Returns `None` for empty
/// `logits`. With `temp == 0.0` this degenerates to greedy.
///
/// `k == 0` or `k >= logits.len()` means "no top-k restriction" (consider all
/// tokens). Implementation (and the precise PRNG) lands in WF-2.
#[must_use]
pub fn sample_top_k(logits: &[f32], k: usize, temp: f32, seed: u64) -> Option<u32> {
    let _ = (logits, k, temp, seed);
    todo!("WF-2: temperature top-k sampling with an inline deterministic PRNG")
}

/// Temperature top-p (nucleus) sampling.
///
/// Sorts tokens by probability, keeps the smallest prefix whose cumulative
/// probability mass first reaches `p ∈ (0, 1]`, applies temperature `temp`, then
/// draws one token using a deterministic PRNG seeded by `seed`. Returns `None`
/// for empty `logits`. With `temp == 0.0` this degenerates to greedy.
///
/// Implementation (and the precise PRNG) lands in WF-2.
#[must_use]
pub fn sample_top_p(logits: &[f32], p: f32, temp: f32, seed: u64) -> Option<u32> {
    let _ = (logits, p, temp, seed);
    todo!("WF-2: temperature top-p (nucleus) sampling with an inline deterministic PRNG")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        assert_eq!(sample_greedy(&[0.1, 0.9, 0.3, 0.2]), Some(1));
        assert_eq!(sample_greedy(&[]), None);
        // NaN must not win.
        assert_eq!(sample_greedy(&[f32::NAN, 0.5]), Some(1));
    }
}
