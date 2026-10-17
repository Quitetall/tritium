//! Forward-only, noise-credited gradient estimate for `ternary_carry --grad noise` (ladder axis G,
//! fused with the carry on axis F). All integer.
//!
//! For each layer separately (Dust perturbs layer types in separate passes), integer antithetic
//! noise `a` is added to the layer's raw pre-activation sums and the true quantized forward is
//! re-run from that point; each draw is credited by its exact integer loss difference
//! `d = ℓ(+a) − ℓ(−a)`. Since E[d·a_j] = 2·E[a²]·∂ℓ/∂raw_j to first order, the per-unit estimate is
//! `ê_j = Σ_draws d·a_j / (2·K·E[a²])`, and the weight estimate is `ê ⊗ input` — expressed in the
//! exact gradient's 2⁻¹¹ units so that the carry's lr shift keeps its meaning. As K grows the
//! estimate approaches the exact STE gradient's direction (`ternary_estimator`, rung G1).
use crate::model::{Example, Model};
use crate::numeric::Rng;

/// Noise distribution for each raw-sum perturbation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Noise {
    /// a ∈ {−σ, 0, +σ} uniform: E[a²] = 2σ²/3.
    Ternary,
    /// a = round(σ·N(0,1)) by an integer CLT sum of 12 uniforms: E[a²] ≈ σ².
    Gauss,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Config {
    pub(crate) noise: Noise,
    /// Noise scale in raw units for the hidden and output layers (σ in quantization steps × 2^shift).
    pub(crate) sigma_hidden: i64,
    pub(crate) sigma_output: i64,
    /// Antithetic draws per example per layer.
    pub(crate) draws: u32,
}

/// Fixed-point fraction bits carried by the per-unit estimate before the outer product.
const FRAC: u32 = 16;
const TARGET: i64 = 256;

fn loss(m: &Model, out_raw: &[i64], label: usize) -> i64 {
    out_raw
        .iter()
        .enumerate()
        .map(|(k, &v)| {
            let e = v / (1 << m.output_shift) - if k == label { TARGET } else { 0 };
            e * e
        })
        .sum()
}

fn output_raw(m: &Model, hidden: &[i64]) -> Vec<i64> {
    m.trits[m.inputs * m.hidden..]
        .chunks_exact(m.hidden)
        .map(|row| row.iter().zip(hidden).map(|(&w, &v)| i64::from(w) * v).sum())
        .collect()
}

fn draw(rng: &mut Rng, noise: Noise, sigma: i64) -> i64 {
    match noise {
        Noise::Ternary => (rng.below(3) as i64 - 1) * sigma,
        Noise::Gauss => {
            // Σ of 12 uniforms in [0, 2^16) has mean 6·2^16 and SD 2^16: an integer CLT normal.
            let s: i64 = (0..12).map(|_| rng.below(1 << 16) as i64).sum::<i64>() - 6 * (1 << 16);
            (s * sigma + (1 << 15)) >> 16
        }
    }
}

/// (numerator, denominator) of E[a²]/σ² for the noise: ternary 2/3, Gauss 1.
fn second_moment(noise: Noise) -> (i64, i64) {
    match noise {
        Noise::Ternary => (2, 3),
        Noise::Gauss => (1, 1),
    }
}

/// Per-unit estimates in 2^(11 + FRAC) units of ∂ℓ/∂raw, for one layer.
fn unit_estimate(
    m: &Model,
    x: &Example,
    hidden_raw: &[i64],
    out_raw: &[i64],
    layer: usize,
    c: &Config,
    rng: &mut Rng,
) -> Vec<i64> {
    let (units, sigma) = if layer == 0 {
        (m.hidden, c.sigma_hidden)
    } else {
        (m.outputs, c.sigma_output)
    };
    let mut sum = vec![0i128; units];
    let mut a = vec![0i64; units];
    for _ in 0..c.draws {
        for v in &mut a {
            *v = draw(rng, c.noise, sigma);
        }
        let l = |sign: i64| -> i64 {
            if layer == 0 {
                let h: Vec<i64> = hidden_raw
                    .iter()
                    .zip(&a)
                    .map(|(&r, &n)| ((r + sign * n) / (1 << m.input_shift)).max(0))
                    .collect();
                loss(m, &output_raw(m, &h), x.label)
            } else {
                let o: Vec<i64> = out_raw.iter().zip(&a).map(|(&r, &n)| r + sign * n).collect();
                loss(m, &o, x.label)
            }
        };
        let d = i128::from(l(1) - l(-1));
        if d == 0 {
            continue;
        }
        for (s, &n) in sum.iter_mut().zip(&a) {
            *s += d * i128::from(n);
        }
    }
    // ê = Σ d·a / (2·K·E[a²]) = Σ d·a · den / (2·K·num·σ²), scaled by 2^(11 + FRAC).
    let (num, den) = second_moment(c.noise);
    let divisor = 2 * i128::from(c.draws) * i128::from(num) * i128::from(sigma) * i128::from(sigma);
    sum.iter()
        .map(|&s| {
            let n = s * i128::from(den) << (11 + FRAC);
            // Round half away from zero.
            ((n + n.signum() * divisor / 2) / divisor) as i64
        })
        .collect()
}

/// Noise-credited weight-gradient estimate in the exact gradient's 2⁻¹¹ units and layout.
pub(crate) fn gradient(m: &Model, x: &Example, c: &Config, rng: &mut Rng) -> Vec<i64> {
    let split = m.inputs * m.hidden;
    let hidden_raw: Vec<i64> = m.trits[..split]
        .chunks_exact(m.inputs)
        .map(|row| row.iter().zip(&x.pixels).map(|(&w, &v)| i64::from(w) * i64::from(v)).sum())
        .collect();
    let hidden: Vec<i64> = hidden_raw.iter().map(|v| (v / (1 << m.input_shift)).max(0)).collect();
    let out_raw = output_raw(m, &hidden);
    let e_hidden = unit_estimate(m, x, &hidden_raw, &out_raw, 0, c, rng);
    let e_out = unit_estimate(m, x, &hidden_raw, &out_raw, 1, c, rng);
    let round = |v: i128| ((v + (1 << (FRAC - 1))) >> FRAC) as i64;
    let mut g = vec![0i64; m.trits.len()];
    for j in 0..m.hidden {
        if e_hidden[j] == 0 {
            continue;
        }
        for (i, &p) in x.pixels.iter().enumerate() {
            g[j * m.inputs + i] = round(i128::from(e_hidden[j]) * i128::from(p));
        }
    }
    for k in 0..m.outputs {
        for j in 0..m.hidden {
            g[split + k * m.hidden + j] = round(i128::from(e_out[k]) * i128::from(hidden[j]));
        }
    }
    g
}

/// Partial forwards per example per step: 2 antithetic evaluations per draw per layer.
pub(crate) fn noisy_evaluations(c: &Config) -> u64 {
    4 * u64::from(c.draws)
}
