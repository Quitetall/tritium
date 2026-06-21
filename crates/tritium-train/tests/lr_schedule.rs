//! LR-schedule gate (plan 0013): linear warmup → cosine decay. Pure function of `step`; must be
//! finite, non-negative, peak at the end of warmup, and monotone-decreasing through the decay.

use proptest::prelude::*;
use tritium_train::LrSchedule;

#[test]
fn warmup_ramps_up_then_peaks() {
    let s = LrSchedule::new(1.0, 0.1, 10, 100);
    // Strictly increasing across the warmup, reaching peak at the last warmup step.
    let mut prev = 0.0;
    for step in 0..10 {
        let lr = s.lr(step);
        assert!(
            lr > prev,
            "warmup not increasing at step {step}: {lr} <= {prev}"
        );
        assert!(lr <= 1.0 + 1e-6);
        prev = lr;
    }
    assert!(
        (s.lr(9) - 1.0).abs() < 1e-6,
        "warmup should reach peak at its last step"
    );
}

#[test]
fn peak_at_warmup_boundary_then_decays() {
    let s = LrSchedule::new(1.0, 0.1, 10, 100);
    assert!(
        (s.lr(10) - 1.0).abs() < 1e-6,
        "lr at the warmup boundary should be the peak"
    );
    // Strictly decreasing across the decay phase.
    let mut prev = s.lr(10);
    for step in 11..=100 {
        let lr = s.lr(step);
        assert!(
            lr <= prev + 1e-7,
            "decay not monotone at step {step}: {lr} > {prev}"
        );
        prev = lr;
    }
    assert!((s.lr(100) - 0.1).abs() < 1e-5, "decay should end at min_lr");
}

#[test]
fn no_warmup_starts_at_peak() {
    let s = LrSchedule::new(2.0, 0.0, 0, 50);
    assert!((s.lr(0) - 2.0).abs() < 1e-6);
    assert!(s.lr(50) < 1e-5);
}

#[test]
fn clamps_past_total_to_min_lr() {
    let s = LrSchedule::new(1.0, 0.05, 10, 100);
    assert!(
        (s.lr(1000) - 0.05).abs() < 1e-6,
        "past total should hold at min_lr"
    );
}

#[test]
fn degenerate_warmup_equals_total_is_min_after_warmup() {
    // total == warmup: the whole run is warmup; the post-warmup value is min_lr (no decay window).
    let s = LrSchedule::new(1.0, 0.1, 5, 5);
    assert!((s.lr(5) - 0.1).abs() < 1e-6);
    assert!(s.lr(4) > 0.0 && s.lr(4) <= 1.0 + 1e-6);
}

proptest! {
    /// Never NaN/Inf/negative, never above peak, for any step and any sane config.
    #[test]
    fn always_finite_nonneg_bounded(
        peak in 1e-6f32..10.0, min_frac in 0.0f32..1.0,
        total in 1u64..700, warmup_frac in 0.0f64..=1.0, step in 0u64..2000,
    ) {
        let min_lr = peak * min_frac;
        // warmup ∈ [0, total]; warmup_frac == 1.0 reaches the degenerate window==0 (warmup==total).
        let warmup = (warmup_frac * total as f64) as u64;
        let s = LrSchedule::new(peak, min_lr, warmup, total);
        let lr = s.lr(step);
        prop_assert!(lr.is_finite(), "lr not finite at step {}: {}", step, lr);
        prop_assert!(lr >= -1e-9, "lr negative at step {}: {}", step, lr);
        prop_assert!(lr <= peak + 1e-4, "lr {} exceeds peak {}", lr, peak);
    }
}
