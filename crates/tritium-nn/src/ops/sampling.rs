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

/// `splitmix64`: a tiny, fast, well-distributed PRNG seeded by a single `u64`.
///
/// One step yields a fresh 64-bit value; we derive a uniform `f32` in `[0, 1)`
/// from its top 24 bits. This keeps the samplers reproducible (a given `seed`
/// always picks the same token) without pulling in the `rand` crate.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Draw a uniform `f32` in `[0, 1)` from a single PRNG step.
#[inline]
fn next_unit(state: &mut u64) -> f32 {
    // Top 24 bits give a value in [0, 2^24); scale into [0, 1).
    let bits = splitmix64(state) >> 40;
    bits as f32 / (1u64 << 24) as f32
}

/// Index of the maximum logit, NaN-tolerant (NaNs lose). Caller guarantees a
/// non-empty slice, so the `expect` is unreachable in practice.
#[inline]
fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Less))
        .map(|(i, _)| i as u32)
        .expect("argmax called on non-empty slice")
}

/// Softmax over the selected `(index, logit)` candidates, with temperature
/// already applied to the logits. Returns parallel `(indices, probs)` vectors.
/// Uses the max-subtraction trick for numerical stability.
fn softmax_over(candidates: &[(u32, f32)]) -> (Vec<u32>, Vec<f32>) {
    let max = candidates
        .iter()
        .map(|&(_, l)| l)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    let mut probs = Vec::with_capacity(candidates.len());
    let mut indices = Vec::with_capacity(candidates.len());
    for &(idx, logit) in candidates {
        let e = (logit - max).exp();
        sum += e;
        probs.push(e);
        indices.push(idx);
    }
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }
    (indices, probs)
}

/// Categorical draw from `(indices, probs)` using one PRNG step. Assumes the
/// probabilities are non-negative and sum to (approximately) one; the final
/// index is returned as a fallback against floating-point round-off.
/// Draw one index from parallel `(indices, probs)` with the deterministic
/// PRNG the samplers use. Public for the speculative-sampling resample step.
#[must_use]
pub fn sample_categorical(indices: &[u32], probs: &[f32], seed: u64) -> u32 {
    let mut state = seed;
    let r = next_unit(&mut state);
    let mut acc = 0.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r < acc {
            return indices[i];
        }
    }
    // Round-off fallback: return the last candidate (probs nominally sum to 1).
    *indices.last().unwrap_or(&0)
}

/// Collect the finite `(index, scaled_logit)` candidates after applying
/// temperature. Returns `None` if there are no finite logits to consider.
fn scaled_candidates(logits: &[f32], temp: f32) -> Option<Vec<(u32, f32)>> {
    let cands: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .filter(|(_, l)| l.is_finite())
        .map(|(i, &l)| (i as u32, l / temp))
        .collect();
    if cands.is_empty() { None } else { Some(cands) }
}

/// Temperature top-k sampling.
///
/// Restricts the candidate set to the `k` highest logits, applies temperature
/// `temp` (logits divided by `temp` before the softmax), then draws one token
/// using a deterministic PRNG seeded by `seed`. Returns `None` for empty
/// `logits`. With `temp <= 0.0` this degenerates to greedy (argmax).
///
/// `k == 0` or `k >= logits.len()` means "no top-k restriction" (consider all
/// tokens).
#[must_use]
pub fn sample_top_k(logits: &[f32], k: usize, temp: f32, seed: u64) -> Option<u32> {
    if logits.is_empty() {
        return None;
    }
    // temp <= 0 (or non-finite, e.g. NaN) => deterministic argmax. Written so
    // NaN takes this branch too, since NaN fails every ordered comparison.
    if !temp.is_finite() || temp <= 0.0 {
        return Some(argmax(logits));
    }

    let (indices, probs) = truncated_top_k(logits, k, temp)?;
    Some(sample_categorical(&indices, &probs, seed))
}

/// The exact truncated, temperature-scaled, renormalized distribution
/// [`sample_top_k`] draws from, as parallel `(indices, probs)`. Public so the
/// speculative-sampling accept rule (serve's spec-decode path) evaluates the
/// SAME distribution the plain sampler uses — lossless-in-distribution by
/// construction, not by re-implementation. `temp <= 0` (or non-finite)
/// degenerates to the single argmax candidate at probability 1, exactly
/// mirroring the sampler's greedy branch.
#[must_use]
pub fn truncated_top_k(logits: &[f32], k: usize, temp: f32) -> Option<(Vec<u32>, Vec<f32>)> {
    if logits.is_empty() {
        return None;
    }
    if !temp.is_finite() || temp <= 0.0 {
        return Some((vec![argmax(logits)], vec![1.0]));
    }
    let mut cands = scaled_candidates(logits, temp)?;

    // Restrict to the k highest logits. k == 0 or k >= len means "keep all".
    let keep = if k == 0 {
        cands.len()
    } else {
        k.min(cands.len())
    };
    if keep < cands.len() {
        // Partition so the `keep` highest-logit candidates come first, then
        // sort that prefix descending for a stable, deterministic order.
        cands.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        cands.truncate(keep);
    }

    Some(softmax_over(&cands))
}

/// Temperature top-p (nucleus) sampling.
///
/// Sorts tokens by probability, keeps the smallest prefix whose cumulative
/// probability mass first reaches `p`, applies temperature `temp`, then draws
/// one token using a deterministic PRNG seeded by `seed`. Returns `None` for
/// empty `logits`. With `temp <= 0.0` this degenerates to greedy (argmax).
///
/// `p` is clamped to `(0, 1]`; `p >= 1.0` keeps every nonzero-probability token.
#[must_use]
pub fn sample_top_p(logits: &[f32], p: f32, temp: f32, seed: u64) -> Option<u32> {
    if logits.is_empty() {
        return None;
    }
    // temp <= 0 (or non-finite, e.g. NaN) => deterministic argmax.
    if !temp.is_finite() || temp <= 0.0 {
        return Some(argmax(logits));
    }

    let (kept_indices, kept_probs) = truncated_top_p(logits, p, temp)?;
    Some(sample_categorical(&kept_indices, &kept_probs, seed))
}

/// The exact nucleus distribution [`sample_top_p`] draws from, as parallel
/// `(indices, probs)` — see [`truncated_top_k`] for why this is public.
/// `temp <= 0` (or non-finite) degenerates to the single argmax candidate.
#[must_use]
pub fn truncated_top_p(logits: &[f32], p: f32, temp: f32) -> Option<(Vec<u32>, Vec<f32>)> {
    if logits.is_empty() {
        return None;
    }
    if !temp.is_finite() || temp <= 0.0 {
        return Some((vec![argmax(logits)], vec![1.0]));
    }
    let cands = scaled_candidates(logits, temp)?;

    // Full softmax over all finite candidates, then sort by descending prob.
    let (indices, probs) = softmax_over(&cands);
    let mut ranked: Vec<(u32, f32)> = indices.into_iter().zip(probs).collect();
    ranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    // Keep the smallest prefix whose cumulative mass first reaches p. A
    // non-positive p falls back to a single token (the most probable); p >= 1.0
    // keeps everything. Always keep at least one candidate.
    let threshold = p.clamp(0.0, 1.0);
    let mut cum = 0.0_f32;
    let mut cutoff = ranked.len();
    for (i, &(_, prob)) in ranked.iter().enumerate() {
        cum += prob;
        if cum >= threshold {
            cutoff = i + 1;
            break;
        }
    }
    ranked.truncate(cutoff.max(1));

    // Renormalize the kept nucleus.
    let kept_indices: Vec<u32> = ranked.iter().map(|&(i, _)| i).collect();
    let mut kept_probs: Vec<f32> = ranked.iter().map(|&(_, pr)| pr).collect();
    let sum: f32 = kept_probs.iter().sum();
    if sum > 0.0 {
        for pr in &mut kept_probs {
            *pr /= sum;
        }
    }
    Some((kept_indices, kept_probs))
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
