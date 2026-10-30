//! Deterministic tied-scale ladder grid, rounding and balanced-ternary encoding.

/// `3^n` for `n ≤ 8`, exact in `f32` (3^8 = 6561 ≪ 2^24).
fn pow3(n: usize) -> f32 {
    let mut v = 1.0f32;
    for _ in 0..n {
        v *= 3.0;
    }
    v
}

/// Largest representable multiple of `Δ` at `t` planes: `(3^t − 1)/2`.
fn ladder_kmax(t: usize) -> i32 {
    (3_i32.saturating_pow(t.min(9) as u32) - 1) / 2
}

/// `2^(-j/4)`, the grid's step factor, built so a device mirror reproduces it **exactly**.
///
/// The obvious `(-(j as f32) / 4.0).exp2()` is not usable across host and device: CUDA's `exp2f` and
/// Rust's `f32::exp2` are each accurate to a couple of ULP but not to the same couple, and a 1-ULP
/// difference in `Δ` changes which grid candidate wins, which changes the whole fit. So the factor is
/// assembled from four exact quarter-octave constants and repeated exact halving — no transcendental
/// on either side, and every operation correctly rounded.
fn ladder_grid_factor(j: usize) -> f32 {
    /// `2^(-r/4)` for `r ∈ 0..4`, as f32.
    // These four f32 values must stay BIT-IDENTICAL to `QUARTER_OCTAVE` in
    // `crates/tritium-cuda/kernels/train_grad.cu`, or the device fitter picks a different grid
    // candidate than the host and the parity gate fails. `2^-1/2` is spelled as the std constant
    // because clippy recognises the literal; it is the same f32 (0x3f3504f3) either way.
    const QUARTER: [f32; 4] = [1.0, 0.8408964, core::f32::consts::FRAC_1_SQRT_2, 0.59460356];
    let mut f = QUARTER[j % 4];
    for _ in 0..(j / 4) {
        f *= 0.5; // exact: halving only decrements the exponent
    }
    f
}

/// The ladder's grid-searched step `Δ` for one group at `t` planes.
///
/// The reachable values are exactly `Δ·k` for integer `k ∈ ±(3^t−1)/2` — a uniform grid — so `Δ`
/// and `t` are the complete description of the quantizer for that group. The base-3 digits are the
/// *encoding* of `k`, not part of the rounding decision.
///
/// Exposed so an activation-aware fitter (GPTQ-style error compensation) can reuse the project's
/// own step selection instead of reimplementing it and drifting from it.
#[must_use]
pub fn ladder_step(bs: &[f32], t: usize, grid: usize) -> f32 {
    let t = t.max(1);
    let peak = bs.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if peak <= 0.0 || !peak.is_finite() {
        return 0.0;
    }
    let delta0 = peak / ladder_kmax(t) as f32;
    let mut best = delta0;
    let mut best_sse = ladder_sse(bs, t, delta0);
    for j in 1..grid {
        let delta = delta0 * ladder_grid_factor(j);
        let sse = ladder_sse(bs, t, delta);
        if sse < best_sse {
            best_sse = sse;
            best = delta;
        }
    }
    best
}

/// Round one value onto the ladder's grid at step `delta`: `clamp(round(w/Δ), ±(3^t−1)/2)·Δ`.
///
/// The single definition of what the ladder does to a scalar, shared with the
/// balanced-ternary encoder and SSE metric. An activation-aware fitter must round
/// with exactly this or it is fitting a different quantizer than the packer will encode.
#[must_use]
pub fn ladder_quantize_at(w: f32, t: usize, delta: f32) -> f32 {
    if delta <= 0.0 || !delta.is_finite() {
        return 0.0;
    }
    let kmax = ladder_kmax(t.max(1)) as f32;
    (w / delta).round().clamp(-kmax, kmax) * delta
}

/// Assign one group to the ladder with step `delta`, returning plane-major trits.
///
/// `k = clamp(round(w/Δ))` is the nearest lattice point — and because the ladder is a uniform grid,
/// nearest-lattice-point IS the minimum-SSE ternary assignment, exactly. No search, no iteration.
fn ladder_digits(bs: &[f32], t: usize, delta: f32) -> Vec<Vec<i8>> {
    let kmax = ladder_kmax(t);
    let mut planes = vec![vec![0i8; bs.len()]; t];
    for (i, &w) in bs.iter().enumerate() {
        let mut k = if delta > 0.0 && delta.is_finite() {
            (w / delta).round().clamp(-(kmax as f32), kmax as f32) as i32
        } else {
            0
        };
        // Balanced-ternary digits, most significant plane first: plane 0 carries 3^(t-1).
        for p in (0..t).rev() {
            let d = (k + 1).rem_euclid(3) - 1;
            planes[p][i] = d as i8;
            k = (k - d) / 3;
        }
    }
    planes
}

/// Sum of squared error of the ladder reconstruction at step `delta`, ascending order (ADR 0018).
fn ladder_sse(bs: &[f32], t: usize, delta: f32) -> f32 {
    let kmax = ladder_kmax(t) as f32;
    let mut e = 0.0f32;
    for &w in bs {
        let k = if delta > 0.0 && delta.is_finite() {
            (w / delta).round().clamp(-kmax, kmax)
        } else {
            0.0
        };
        let d = w - k * delta;
        e += d * d;
    }
    e
}

/// Fit one group to the ladder, returning `(s₀, plane-major trits)`.
///
/// The only free parameter is the step `Δ`, searched over a **deterministic** log grid:
/// `Δ_j = Δ₀·2^(-j/4)` for `j ∈ 0..grid`, where `Δ₀ = max|w| / kmax` is the exactly clipping-free
/// step. Increasing `j` trades range for resolution monotonically. Fixed candidate set, argmin SSE,
/// lowest `j` wins ties — so there is no data-dependent early exit to mirror on a device, unlike
/// ITF's accept guard.
///
/// `grid = 0` (or 1) uses `j = 0` alone: the clipping-free ladder, the cheapest useful configuration.
#[must_use]
pub fn fit_ladder_group(bs: &[f32], t: usize, grid: usize) -> (f32, Vec<Vec<i8>>) {
    let t = t.max(1);
    let peak = bs.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if peak <= 0.0 || !peak.is_finite() {
        return (0.0, vec![vec![0i8; bs.len()]; t]);
    }
    let delta = ladder_step(bs, t, grid);
    (delta * pow3(t - 1), ladder_digits(bs, t, delta))
}
