//! Finite scalar quantization (FSQ) forward + straight-through backward for the autograd tape.
//!
//! FSQ rounds each activation channel to one of `L` levels on a fixed grid — the codec's rate knob.
//! Like [`ste`](super::ste), the QAT **forward** is the rounded value ([`forward`]) and the
//! **straight-through backward** is the exact gradient of a differentiable **surrogate** ([`surrogate`]
//! / [`surrogate_soft`]); `round` is piecewise-constant (0 derivative a.e.) and only a forward detail.
//!
//! Grid (parity-robust, endpoints-inclusive): with `L` levels the code is
//! `c = round((bound(x)+1)/2·(L−1))` clamped to `0..L−1`, reconstructed as `q = c/(L−1)·2 − 1`. This
//! emits exactly `L` codes for any parity (`L=2 → {−1,+1}`, `L=3 → {−1,0,1}`), unlike the naive
//! `round(x/step)·step, step=2/L` which yields `L+1` points for even `L`. The rounding is
//! **round-half-away-from-zero implemented explicitly** — never `rintf`/`nearbyint` (half-to-even) — so
//! the grid is bit-identical across CPU/CUDA/MCU (the clinical byte-identity requirement). For the
//! byte-exact deploy path use the [`FsqBound::Clamp`] bound; `tanh` is not bit-identical across
//! libm/CUDA/CMSIS and is training-only unless a shared LUT is provided.
//!
//! This op is also the trainable **activation-quant STE node** the substrate otherwise lacks.

use core::f32::consts::PI;

/// Bounding nonlinearity applied before the grid round.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FsqBound {
    /// `tanh(x)` — smooth, training-only for byte-exact deploy (not bit-identical across libms).
    Tanh,
    /// `clamp(x, −1, 1)` — bit-identical everywhere; the deploy bound.
    Clamp,
}

/// Straight-through estimator variant (the backward, and — for `Stochastic` — the forward rounding).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FsqSte {
    /// Pass the gradient of the bound straight through (`round` invisible).
    Hard,
    /// Annealed soft-round surrogate; `alpha ∈ [0,1]` (0 = identity passthrough, 1 = full soft-round).
    SoftRound { alpha: f32 },
    /// Unbiased stochastic rounding in the forward (`E[q] = bound(x)`), hard gradient in the backward.
    Stochastic { seed: u64 },
}

/// FSQ geometry: `[channels, len]` row-major activation, one `L` per channel (`levels.len()==channels`).
#[derive(Clone, Debug, PartialEq)]
pub struct FsqCfg {
    /// Number of channels (rows).
    pub channels: usize,
    /// Per-channel length (cols).
    pub len: usize,
    /// Levels `L` per channel (each ≥ 2), e.g. `{2,3,5,8,16,32}`.
    pub levels: Vec<u32>,
    /// Bounding nonlinearity.
    pub bound: FsqBound,
}

impl FsqCfg {
    /// Whether the geometry is well-formed for a buffer of length `x_len`.
    #[must_use]
    pub fn buffers_fit(&self, x_len: usize) -> bool {
        self.levels.len() == self.channels
            && self.levels.iter().all(|&l| l >= 2)
            && x_len == self.channels * self.len
    }
}

fn bound_apply(x: f32, b: FsqBound) -> f32 {
    match b {
        FsqBound::Tanh => x.tanh(),
        FsqBound::Clamp => x.clamp(-1.0, 1.0),
    }
}

fn bound_deriv(x: f32, b: FsqBound) -> f32 {
    match b {
        FsqBound::Tanh => {
            let t = x.tanh();
            1.0 - t * t
        }
        FsqBound::Clamp => {
            if x.abs() < 1.0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Round half away from zero — explicit, backend-portable (never `rintf`, which rounds half-to-even).
fn round_half_away(v: f32) -> f32 {
    if v >= 0.0 {
        (v + 0.5).floor()
    } else {
        (v - 0.5).ceil()
    }
}

/// Deterministic uniform in `[0,1)` from a seed and flat index (xorshift64 idiom, forced non-zero).
fn uniform01(seed: u64, idx: usize) -> f32 {
    let mut s = (seed ^ (idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) | 1;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    (s % 1_000_000) as f32 / 1_000_000.0
}

/// Hard grid quantize of a bounded value `b ∈ [−1,1]` to `L` levels.
fn quantize_level(b: f32, l: u32) -> f32 {
    let lm1 = (l - 1) as f32;
    let code = round_half_away((b + 1.0) * 0.5 * lm1).clamp(0.0, lm1);
    code / lm1 * 2.0 - 1.0
}

/// Stochastic grid quantize: round up with probability = fractional code position (unbiased).
fn quantize_level_stochastic(b: f32, l: u32, seed: u64, idx: usize) -> f32 {
    let lm1 = (l - 1) as f32;
    let z = ((b + 1.0) * 0.5 * lm1).clamp(0.0, lm1);
    let floor_z = z.floor();
    let up = if uniform01(seed, idx) < (z - floor_z) {
        1.0
    } else {
        0.0
    };
    (floor_z + up).clamp(0.0, lm1) / lm1 * 2.0 - 1.0
}

/// Soft-round surrogate value of a bounded `b`: `soft_z = z − sin(2πz)/(2π)·alpha`, reconstructed.
fn quantize_level_soft(b: f32, l: u32, alpha: f32) -> f32 {
    let lm1 = (l - 1) as f32;
    let z = (b + 1.0) * 0.5 * lm1;
    let soft_z = z - (2.0 * PI * z).sin() / (2.0 * PI) * alpha;
    soft_z / lm1 * 2.0 - 1.0
}

/// Forward: the rounded QAT value. `Hard`/`SoftRound` round hard (their backwards differ); `Stochastic`
/// rounds stochastically from `(seed, index)`.
#[must_use]
pub fn forward(x: &[f32], cfg: &FsqCfg, ste: FsqSte) -> Vec<f32> {
    debug_assert!(cfg.buffers_fit(x.len()));
    x.iter()
        .enumerate()
        .map(|(i, &xi)| {
            let l = cfg.levels[i / cfg.len];
            let b = bound_apply(xi, cfg.bound);
            match ste {
                FsqSte::Stochastic { seed } => quantize_level_stochastic(b, l, seed, i),
                _ => quantize_level(b, l),
            }
        })
        .collect()
}

/// Hard/stochastic straight-through surrogate: `bound(x)` (round invisible). Its exact gradient is
/// [`vjp_hard`] — the finite-difference oracle for Gate C.
#[must_use]
pub fn surrogate(x: &[f32], cfg: &FsqCfg) -> Vec<f32> {
    x.iter().map(|&xi| bound_apply(xi, cfg.bound)).collect()
}

/// Annealed soft-round surrogate; its exact gradient is [`vjp_soft`].
#[must_use]
pub fn surrogate_soft(x: &[f32], cfg: &FsqCfg, alpha: f32) -> Vec<f32> {
    x.iter()
        .enumerate()
        .map(|(i, &xi)| {
            quantize_level_soft(bound_apply(xi, cfg.bound), cfg.levels[i / cfg.len], alpha)
        })
        .collect()
}

/// vjp for hard passthrough (and stochastic, whose expectation is `bound`): `gX = grad_out · bound'(x)`.
#[must_use]
pub fn vjp_hard(x: &[f32], cfg: &FsqCfg, grad_out: &[f32]) -> Vec<Vec<f32>> {
    let g_x = x
        .iter()
        .zip(grad_out)
        .map(|(&xi, &g)| g * bound_deriv(xi, cfg.bound))
        .collect();
    vec![g_x]
}

/// vjp for the annealed soft-round surrogate: `gX = grad_out · (1 − alpha·cos(2πz)) · bound'(x)`,
/// `z = (bound(x)+1)/2·(L−1)`.
#[must_use]
pub fn vjp_soft(x: &[f32], cfg: &FsqCfg, alpha: f32, grad_out: &[f32]) -> Vec<Vec<f32>> {
    let g_x = x
        .iter()
        .zip(grad_out)
        .enumerate()
        .map(|(i, (&xi, &g))| {
            let l = cfg.levels[i / cfg.len];
            let b = bound_apply(xi, cfg.bound);
            let z = (b + 1.0) * 0.5 * (l - 1) as f32;
            g * (1.0 - alpha * (2.0 * PI * z).cos()) * bound_deriv(xi, cfg.bound)
        })
        .collect();
    vec![g_x]
}

/// Dispatch the straight-through backward for the chosen [`FsqSte`] (used by `Tape::fsq`).
#[must_use]
pub fn vjp(x: &[f32], cfg: &FsqCfg, ste: FsqSte, grad_out: &[f32]) -> Vec<Vec<f32>> {
    match ste {
        FsqSte::Hard | FsqSte::Stochastic { .. } => vjp_hard(x, cfg, grad_out),
        FsqSte::SoftRound { alpha } => vjp_soft(x, cfg, alpha, grad_out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(channels: usize, len: usize, levels: &[u32], bound: FsqBound) -> FsqCfg {
        FsqCfg {
            channels,
            len,
            levels: levels.to_vec(),
            bound,
        }
    }

    #[test]
    fn grid_emits_exactly_l_codes_all_parities() {
        // L=2 → {−1,+1}; L=3 → {−1,0,1}; L=5 → {−1,−.5,0,.5,1}. Sweep bounded inputs, collect codes.
        for &l in &[2u32, 3, 5, 8] {
            let c = cfg(1, 1, &[l], FsqBound::Clamp);
            let mut codes = std::collections::BTreeSet::new();
            for i in 0..=1000 {
                let x = -1.0 + 2.0 * i as f32 / 1000.0;
                let q = forward(&[x], &c, FsqSte::Hard)[0];
                codes.insert((q * 1e6).round() as i64);
            }
            assert_eq!(codes.len(), l as usize, "L={l} must emit exactly L codes");
        }
        // L=2 endpoints.
        let c2 = cfg(1, 1, &[2], FsqBound::Clamp);
        assert_eq!(forward(&[-1.0], &c2, FsqSte::Hard)[0], -1.0);
        assert_eq!(forward(&[1.0], &c2, FsqSte::Hard)[0], 1.0);
        assert_eq!(forward(&[-0.4], &c2, FsqSte::Hard)[0], -1.0); // (b+1)/2=0.3 → code 0
        assert_eq!(forward(&[0.4], &c2, FsqSte::Hard)[0], 1.0); // 0.7 → code 1
    }

    #[test]
    fn round_half_away_not_half_even() {
        // 0.5 → 1 (away), 1.5 → 2 (away), −0.5 → −1. Half-to-even would give 0, 2, 0.
        assert_eq!(round_half_away(0.5), 1.0);
        assert_eq!(round_half_away(1.5), 2.0);
        assert_eq!(round_half_away(-0.5), -1.0);
        assert_eq!(round_half_away(2.5), 3.0);
    }

    #[test]
    fn stochastic_is_unbiased() {
        // Average of stochastic rounding over many seeds ≈ the bounded value (the hard surrogate).
        let b = 0.3f32; // some non-grid point
        let mean: f32 = (0..20_000)
            .map(|s| quantize_level_stochastic(b, 5, s as u64, 0))
            .sum::<f32>()
            / 20_000.0;
        assert!((mean - b).abs() < 0.02, "stochastic mean {mean} vs {b}");
    }

    #[test]
    fn per_channel_levels() {
        // Two channels, different L, distinct grids applied per row.
        let c = cfg(2, 3, &[2, 5], FsqBound::Clamp);
        let x = [0.9, -0.9, 0.1, 0.9, -0.9, 0.1];
        let q = forward(&x, &c, FsqSte::Hard);
        // ch0 L=2: {−1,+1}; ch1 L=5 grid {−1,−.5,0,.5,1}.
        assert_eq!(&q[0..3], &[1.0, -1.0, 1.0]); // L=2 rounds 0.1→(0.55)→code1→+1
        assert_eq!(q[3], 1.0);
        assert_eq!(q[4], -1.0);
        assert_eq!(q[5], 0.0); // L=5: (0.1+1)/2*4=2.2 → code 2 → 0.0
    }
}
