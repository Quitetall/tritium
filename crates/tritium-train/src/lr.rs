//! Learning-rate schedule (plan 0013): linear **warmup** then **cosine decay** — the standard
//! pretraining schedule, as a pure function of the step index.
//!
//! Kept out of the [`Optimizer`](crate::Optimizer) trait on purpose (the trait stays minimal — plan
//! 0008): the loop reads `lr(step)` and configures the optimizer's `lr` each step, so the schedule
//! is reusable by any optimizer and any loop (CPU or GPU) without widening the trait.

use std::f32::consts::PI;

/// Linear-warmup-then-cosine-decay learning-rate schedule.
///
/// - Steps `0..warmup`: `lr` ramps linearly `peak·(step+1)/warmup`, reaching `peak` at the last
///   warmup step (`warmup-1`).
/// - Steps `warmup..=total`: `lr` follows a half-cosine from `peak` down to `min_lr` (decay begins
///   at `progress 0`, also `peak` — so `peak` is held across the two adjacent steps `warmup-1` and
///   `warmup` before the cosine descends).
/// - Past `total`: held at `min_lr`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LrSchedule {
    peak: f32,
    min_lr: f32,
    warmup: u64,
    total: u64,
}

impl LrSchedule {
    /// Build a schedule. `peak >= min_lr >= 0`, `total >= 1`, `warmup <= total`.
    ///
    /// # Panics
    /// If those bounds are violated (a misconfigured schedule is a programming error, caught early).
    #[must_use]
    pub fn new(peak: f32, min_lr: f32, warmup: u64, total: u64) -> Self {
        assert!(
            peak.is_finite() && min_lr.is_finite(),
            "lr bounds must be finite"
        );
        assert!(min_lr >= 0.0, "min_lr must be >= 0");
        assert!(peak >= min_lr, "peak ({peak}) must be >= min_lr ({min_lr})");
        assert!(total >= 1, "total must be >= 1");
        assert!(
            warmup <= total,
            "warmup ({warmup}) must be <= total ({total})"
        );
        Self {
            peak,
            min_lr,
            warmup,
            total,
        }
    }

    /// The learning rate at `step` (0-based).
    #[must_use]
    pub fn lr(&self, step: u64) -> f32 {
        if self.warmup > 0 && step < self.warmup {
            // Linear warmup: peak·(step+1)/warmup ⇒ peak at step == warmup-1.
            return self.peak * (step + 1) as f32 / self.warmup as f32;
        }
        // Cosine decay over [warmup, total]. `step >= warmup` here, so the subtraction is safe.
        let window = self.total.saturating_sub(self.warmup);
        if window == 0 {
            // Degenerate: no decay window (total == warmup) — the run is pure warmup, so the
            // post-warmup value is the floor.
            return self.min_lr;
        }
        let progress = ((step - self.warmup) as f32 / window as f32).clamp(0.0, 1.0);
        let cosine = (PI * progress).cos(); // 1 at progress 0 → -1 at progress 1
        self.min_lr + 0.5 * (self.peak - self.min_lr) * (1.0 + cosine)
    }
}
