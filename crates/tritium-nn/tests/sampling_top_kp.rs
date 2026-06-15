//! Behavioural tests for the stochastic samplers (`sample_top_k`,
//! `sample_top_p`). There is no torch golden for sampling, so these are
//! constructed cases: argmax equivalences, temperature limits, determinism,
//! statistical frequency on a peaked distribution, and empty-input handling.

use tritium_nn::{sample_greedy, sample_top_k, sample_top_p};

/// A clearly-peaked logit vector: index 2 dominates.
fn peaked() -> Vec<f32> {
    vec![0.1, 0.2, 5.0, 0.3, 0.05]
}

#[test]
fn empty_logits_return_none() {
    assert_eq!(sample_top_k(&[], 1, 1.0, 0), None);
    assert_eq!(sample_top_k(&[], 0, 0.0, 7), None);
    assert_eq!(sample_top_p(&[], 1.0, 1.0, 0), None);
    assert_eq!(sample_top_p(&[], 0.9, 0.0, 7), None);
}

#[test]
fn top_k_one_equals_greedy() {
    // k == 1 leaves a single candidate (the argmax) regardless of seed/temp.
    let logits = peaked();
    let want = sample_greedy(&logits);
    for seed in 0..32u64 {
        assert_eq!(sample_top_k(&logits, 1, 1.0, seed), want);
        assert_eq!(sample_top_k(&logits, 1, 0.7, seed), want);
    }
    // Also with a different peak location.
    let other = vec![3.0, -1.0, 0.0, 2.9];
    let want2 = sample_greedy(&other);
    for seed in 0..16u64 {
        assert_eq!(sample_top_k(&other, 1, 1.3, seed), want2);
    }
}

#[test]
fn very_low_temperature_is_argmax() {
    let logits = peaked();
    let want = sample_greedy(&logits);
    // A tiny positive temperature sharpens the softmax onto the peak.
    for seed in 0..64u64 {
        assert_eq!(sample_top_k(&logits, 0, 1e-4, seed), want);
        assert_eq!(sample_top_p(&logits, 1.0, 1e-4, seed), want);
    }
}

#[test]
fn nonpositive_temperature_degenerates_to_greedy() {
    let logits = peaked();
    let want = sample_greedy(&logits);
    // temp <= 0 must be a deterministic argmax with no PRNG involvement.
    assert_eq!(sample_top_k(&logits, 0, 0.0, 123), want);
    assert_eq!(sample_top_k(&logits, 3, -1.0, 999), want);
    assert_eq!(sample_top_p(&logits, 0.9, 0.0, 123), want);
    assert_eq!(sample_top_p(&logits, 0.5, -2.0, 999), want);
}

#[test]
fn determinism_same_seed_same_token() {
    let logits = vec![0.5, 0.5, 0.6, 0.55, 0.45, 0.5];
    for seed in [0u64, 1, 42, 7, 1_000_000, u64::MAX] {
        let a = sample_top_k(&logits, 4, 1.0, seed);
        let b = sample_top_k(&logits, 4, 1.0, seed);
        assert_eq!(a, b, "top_k not deterministic for seed {seed}");

        let c = sample_top_p(&logits, 0.9, 1.0, seed);
        let d = sample_top_p(&logits, 0.9, 1.0, seed);
        assert_eq!(c, d, "top_p not deterministic for seed {seed}");
    }
}

#[test]
fn different_seeds_can_vary() {
    // On a flat-ish distribution, sweeping seeds should produce >1 distinct
    // outcome (the PRNG actually varies by seed, not a constant).
    let logits = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut seen = std::collections::HashSet::new();
    for seed in 0..256u64 {
        if let Some(t) = sample_top_k(&logits, 0, 1.0, seed) {
            seen.insert(t);
        }
    }
    assert!(
        seen.len() > 1,
        "top_k produced only {} distinct token(s) across 256 seeds",
        seen.len()
    );

    let mut seen_p = std::collections::HashSet::new();
    for seed in 0..256u64 {
        if let Some(t) = sample_top_p(&logits, 1.0, 1.0, seed) {
            seen_p.insert(t);
        }
    }
    assert!(
        seen_p.len() > 1,
        "top_p produced only {} distinct token(s) across 256 seeds",
        seen_p.len()
    );
}

#[test]
fn top_p_full_can_sample_any_nonzero_token() {
    // p == 1.0 keeps the whole nucleus, so every finite-logit token is reachable.
    let logits = vec![0.0, 0.1, 0.2, 0.3, 0.05]; // 5 tokens, all plausible
    let mut seen = std::collections::HashSet::new();
    for seed in 0..512u64 {
        if let Some(t) = sample_top_p(&logits, 1.0, 1.0, seed) {
            seen.insert(t);
        }
    }
    assert_eq!(
        seen.len(),
        logits.len(),
        "top_p=1.0 should be able to sample every token; saw {:?}",
        seen
    );
}

#[test]
fn top_p_small_restricts_to_top_token() {
    // A very peaked distribution with small p keeps only the dominant token.
    let logits = vec![10.0, 0.0, -1.0, 0.5];
    let want = sample_greedy(&logits);
    for seed in 0..64u64 {
        assert_eq!(sample_top_p(&logits, 0.1, 1.0, seed), want);
    }
}

#[test]
fn peaked_distribution_samples_peak_with_high_frequency() {
    // Statistical: over many seeds, the dominant token should win the large
    // majority of draws while still leaving room for the tail.
    let logits = peaked(); // index 2 has logit 5.0, the rest near 0.
    let peak = 2u32;
    let trials = 2000u64;
    let mut hits = 0u64;
    for seed in 0..trials {
        if sample_top_k(&logits, 0, 1.0, seed) == Some(peak) {
            hits += 1;
        }
    }
    // softmax mass on index 2 is ~0.97; require a comfortable lower bound to
    // avoid flakiness while still proving the peak dominates.
    let frac = hits as f64 / trials as f64;
    assert!(
        frac > 0.85,
        "peak sampled only {frac:.3} of the time (expected ~0.97)"
    );
    assert!(
        frac < 1.0,
        "peak sampled every single time; tail should occasionally appear"
    );
}

#[test]
fn top_k_larger_than_len_keeps_all() {
    // k >= len behaves like the unrestricted set; the reachable tokens match
    // the full top_p=1.0 nucleus.
    let logits = vec![0.0, 1.0, 0.5, 0.2];
    let mut seen_k = std::collections::HashSet::new();
    let mut seen_all = std::collections::HashSet::new();
    for seed in 0..512u64 {
        if let Some(t) = sample_top_k(&logits, 999, 1.0, seed) {
            seen_k.insert(t);
        }
        if let Some(t) = sample_top_k(&logits, 0, 1.0, seed) {
            seen_all.insert(t);
        }
    }
    assert_eq!(seen_k, seen_all);
    assert_eq!(seen_k.len(), logits.len());
}

#[test]
fn ignores_non_finite_logits() {
    // -inf logits (e.g. masked tokens) must never be selected.
    let logits = vec![f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY, 2.0];
    for seed in 0..128u64 {
        let t = sample_top_k(&logits, 0, 1.0, seed).unwrap();
        assert!(t == 1 || t == 3, "selected a masked token: {t}");
        let p = sample_top_p(&logits, 1.0, 1.0, seed).unwrap();
        assert!(p == 1 || p == 3, "selected a masked token: {p}");
    }
}
