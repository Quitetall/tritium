//! Residual-carry control for EAT-O: the master weight moved into the optimizer.
//!
//! EAT-O accumulates per-coordinate evidence and flips a trit when evidence clears a threshold,
//! then CLEARS that coordinate's evidence; its two EMAs also forget. A latent ("master") weight
//! under STE is the same accumulator with neither loss: writing it as `trit·S + a`, an update
//! `a ← a − η·g` that pushes `|a|` past `S/2` moves the trit one step and keeps the remainder.
//! This example implements exactly that, in integers, with the latent living in optimizer state —
//! there is no separate floating master tensor anywhere, the model itself is only trits.
//!
//! Two optimizers, both full-coverage (every weight, every example) like the STE baseline:
//!
//! - `carry-sgd`: one i32 latent per weight, `latent -= g >> shift`.
//! - `carry-sgd16`: the same with an **i16** latent at 4,096 units per trit (2 bytes/weight), with
//!   round-to-nearest or integer stochastic rounding of each update (`--stochastic yes`), which keeps
//!   updates smaller than one latent unit unbiased instead of rounding them away.
//! - `carry-adam`: an i32 latent (clamped to ±8 trit steps), i32 first moment and u32 second moment
//!   per weight — 12 bytes, EAT-O's state budget and the baseline's fp32 master + Adam moments:
//!   Adam with β1 = 1 − 2⁻³, β2 = 1 − 2⁻¹⁰, bias correction in Q30 fixed point, integer sqrt.
//!
//! The credit is the model's own backprop sensitivity (`Model::sensitivity`) taken BEFORE its
//! truncating division: the exact rational STE gradient, expressed in units of 2⁻¹¹. Every
//! operation in training is an integer operation.
//!
//! Same network, scales, targets, sampling order, splits and evaluation as `ternary_lab`, so the
//! numbers sit in the same table as the EAT-O campaign and its STE+Adam reference.
// Shared with `ternary_lab`; this example uses only part of each module.
#[allow(dead_code)]
#[path = "ternary_lab/model.rs"]
mod model;
#[allow(dead_code)]
#[path = "ternary_lab/numeric.rs"]
mod numeric;

use model::{Example, Model};
use numeric::Rng;
use std::path::Path;

/// Latent units per trit step.
const S: i64 = 1 << 16;
/// Gradient fixed point: every gradient is an integer multiple of 2⁻¹¹.
const GRAD_SHIFT: u32 = 11;
/// Q30 for the bias-correction factors.
const Q30: i64 = 1 << 30;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Sgd,
    Sgd16,
    Adam,
}

/// Latent units per trit step for the 16-bit latent: ±8 trit steps fit i16.
const S16: i64 = 1 << 12;

struct Carry {
    kind: Kind,
    /// SGD: latent += -(g >> shift). Adam: step size in latent units per normalized unit.
    lr: i64,
    latent: Vec<i32>,
    latent16: Vec<i16>,
    stochastic: bool,
    rng: Rng,
    m: Vec<i32>,
    v: Vec<u32>,
    /// β1^t and β2^t in Q30, for Adam's bias correction.
    b1t: i64,
    b2t: i64,
    transitions: u64,
    reversals: u64,
    last_dir: Vec<i8>,
}

/// Floor square root by integer Newton iteration (no floating point anywhere).
fn isqrt(x: u64) -> u64 {
    if x < 2 {
        return x;
    }
    // Start above the root: 2^ceil(bits/2) ≥ sqrt(x); Newton then decreases monotonically.
    let mut r = 1u64 << (64 - x.leading_zeros()).div_ceil(2);
    loop {
        let next = (r + x / r) / 2;
        if next >= r {
            return r;
        }
        r = next;
    }
}

impl Carry {
    fn new(m: &Model, kind: Kind, lr: i64) -> Self {
        let n = m.trits.len();
        Self {
            kind,
            lr,
            latent: if kind == Kind::Sgd16 {
                Vec::new()
            } else {
                m.trits.iter().map(|&t| i32::from(t) * S as i32).collect()
            },
            latent16: if kind == Kind::Sgd16 {
                m.trits.iter().map(|&t| i16::from(t) * S16 as i16).collect()
            } else {
                Vec::new()
            },
            stochastic: false,
            rng: Rng(0x5EED_CA22),
            m: if kind == Kind::Adam {
                vec![0; n]
            } else {
                Vec::new()
            },
            v: if kind == Kind::Adam {
                vec![0; n]
            } else {
                Vec::new()
            },
            b1t: Q30,
            b2t: Q30,
            transitions: 0,
            reversals: 0,
            last_dir: vec![0; n],
        }
    }

    /// Allocated optimizer state (latent + moments), from actual capacities.
    fn state_bytes(&self) -> usize {
        4 * (self.latent.capacity() + self.m.capacity() + self.v.capacity())
            + 2 * self.latent16.capacity()
    }

    /// Full integer gradient in 2⁻¹¹ units: the numerator of `Model::sensitivity` without its
    /// truncating division, scaled to a common fixed point.
    fn gradient(model: &Model, x: &Example) -> Vec<i64> {
        let a = model.forward(x);
        let split = model.inputs * model.hidden;
        let mut g = vec![0i64; model.trits.len()];
        let err: Vec<i64> = (0..model.outputs)
            .map(|k| a.output[k] - if k == x.label { 256 } else { 0 })
            .collect();
        // Output layer: 2·e·h / 2^output_shift  →  2·e·h · 2^(11 − output_shift).
        let out_scale = GRAD_SHIFT - u32::from(model.output_shift);
        for k in 0..model.outputs {
            for j in 0..model.hidden {
                g[split + k * model.hidden + j] = (2 * err[k] * a.hidden[j]) << out_scale;
            }
        }
        // Input layer: Σ_k 2·e_k·w2[k,j] · pixel / 2^(input_shift + output_shift) = 2⁻¹¹ units.
        assert_eq!(
            u32::from(model.input_shift) + u32::from(model.output_shift),
            GRAD_SHIFT
        );
        for j in 0..model.hidden {
            if a.hidden[j] == 0 {
                continue; // ReLU gated at the quantized activation, as Model::sensitivity does.
            }
            let delta: i64 = (0..model.outputs)
                .map(|k| 2 * err[k] * i64::from(model.trits[split + k * model.hidden + j]))
                .sum();
            if delta == 0 {
                continue;
            }
            let row = &mut g[j * model.inputs..(j + 1) * model.inputs];
            for (gi, &p) in row.iter_mut().zip(&x.pixels) {
                *gi = delta * i64::from(p);
            }
        }
        g
    }

    fn step(&mut self, model: &mut Model, x: &Example) {
        let g = Self::gradient(model, x);
        if self.kind == Kind::Adam {
            // β1 = 1 − 2⁻³, β2 = 1 − 2⁻¹⁰.
            self.b1t -= self.b1t >> 3;
            self.b2t -= self.b2t >> 10;
        }
        let (c1, c2) = (Q30 - self.b1t, Q30 - self.b2t);
        for (i, &grad) in g.iter().enumerate() {
            if self.kind == Kind::Sgd16 {
                let shift = self.lr.clamp(1, 62) as u32;
                // Round-to-nearest, or stochastic: add a uniform integer in [0, 2^shift) before the
                // floor shift, so E[update] = g / 2^shift exactly and sub-unit updates survive.
                let offset = if self.stochastic {
                    self.rng.below(1u64 << shift) as i64
                } else {
                    1i64 << (shift - 1)
                };
                let delta = -((grad + offset) >> shift);
                if delta == 0 {
                    continue;
                }
                let latent = (i64::from(self.latent16[i]) + delta)
                    .clamp(i64::from(i16::MIN) + 1, i64::from(i16::MAX));
                self.latent16[i] = latent as i16;
                let t = ((latent + latent.signum() * (S16 / 2)) / S16).clamp(-1, 1) as i8;
                self.apply(model, i, t);
                continue;
            }
            let delta = match self.kind {
                Kind::Sgd16 => unreachable!("handled above"),
                Kind::Sgd => -(grad >> self.lr.clamp(0, 62)),
                Kind::Adam => {
                    let gi = grad.clamp(i64::from(i32::MIN) / 2, i64::from(i32::MAX) / 2);
                    let m = i64::from(self.m[i]);
                    self.m[i] = (m + ((gi - m) >> 3)) as i32;
                    // v stores g²/2¹⁴ so that |g| ≤ 5·10⁶ fits u32.
                    let g2 = ((gi as i128 * gi as i128) >> 14).min(i128::from(u32::MAX)) as i64;
                    let v = i64::from(self.v[i]);
                    self.v[i] = (v + ((g2 - v) >> 10)).clamp(0, i64::from(u32::MAX)) as u32;
                    // m̂ = m/c1, v̂ = v/c2 (Q30); step = lr · m̂ / (sqrt(v̂·2¹⁴) + ε).
                    let m_hat = (i128::from(self.m[i]) * i128::from(Q30)) / i128::from(c1);
                    let v_hat = (u128::from(self.v[i]) * Q30 as u128) / c2 as u128;
                    let denom =
                        i128::from(isqrt((v_hat << 14).min(u128::from(u64::MAX)) as u64)) + 1;
                    -((i128::from(self.lr) * m_hat) / denom) as i64
                }
            };
            if delta == 0 {
                continue;
            }
            let bound = match self.kind {
                Kind::Sgd => i64::from(i32::MAX),
                Kind::Adam | Kind::Sgd16 => 8 * S,
            };
            let latent = (i64::from(self.latent[i]) + delta).clamp(-bound, bound);
            self.latent[i] = latent as i32;
            // The trit is the rounded latent: the carried remainder is never discarded.
            let t = ((latent + latent.signum() * (S / 2)) / S).clamp(-1, 1) as i8;
            self.apply(model, i, t);
        }
    }

    /// Set weight `i` to trit `t`, counting the transition and any immediate reversal.
    fn apply(&mut self, model: &mut Model, i: usize, t: i8) {
        let old = model.trits[i];
        if t != old {
            let dir = (t - old).signum();
            if self.last_dir[i] == -dir {
                self.reversals += 1;
            }
            self.last_dir[i] = dir;
            self.transitions += 1;
            model.trits[i] = t;
            model.version += 1;
        }
    }
}

fn evaluate(m: &Model, data: &[Example]) -> (i128, usize) {
    (
        data.iter().map(|x| m.loss(x)).sum(),
        data.iter().filter(|x| m.correct(x)).count(),
    )
}

fn arg<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let data = arg(&args, "--data").ok_or("--data MNIST_IDX_DIR required")?;
    let kind = match arg(&args, "--optimizer").unwrap_or("carry-adam") {
        "carry-sgd" => Kind::Sgd,
        "carry-sgd16" => Kind::Sgd16,
        "carry-adam" => Kind::Adam,
        other => return Err(format!("unknown optimizer {other}")),
    };
    let num = |k: &str, d: i64| -> Result<i64, String> {
        arg(&args, k).map_or(Ok(d), |v| v.parse().map_err(|_| format!("invalid {k}")))
    };
    let seed = num("--seed", 1)? as u64;
    let steps = num("--steps", 20_000)? as u64;
    // SGD: right shift of the gradient. Adam: latent units per normalized step; S = 65,536 per trit,
    // so the STE baseline's lr grid {1e-4, 1e-3, 1e-2, 1e-1} is {7, 66, 655, 6554}.
    let lr = num("--lr", if kind == Kind::Adam { 655 } else { 20 })?;
    let eval_offset = num("--eval-offset", 1000)? as usize;
    let eval_limit = num("--eval-limit", 4000)? as usize;
    let p = Path::new(data);
    let all = model::mnist(
        &p.join("train-images-idx3-ubyte"),
        &p.join("train-labels-idx1-ubyte"),
    )?;
    let (train, eval) = all.split_at(55_000);
    let eval = &eval[eval_offset..eval_offset + eval_limit];
    // Same initial trits and fresh-example order as ternary_lab for this seed.
    let mut model = Model::new(784, 128, 10, &mut Rng(seed));
    let mut opt = Carry::new(&model, kind, lr);
    opt.stochastic = arg(&args, "--stochastic") == Some("yes");
    let initial = evaluate(&model, eval);
    let start = std::time::Instant::now();
    for step in 0..steps {
        let mut sampler = Rng(seed
            .wrapping_add(step.wrapping_mul(0x9e37_79b9_7f4a_7c15))
            .max(1));
        let i = sampler.below(train.len() as u64) as usize;
        opt.step(&mut model, &train[i]);
    }
    let elapsed = start.elapsed().as_millis();
    model.validate()?;
    let (loss, correct) = evaluate(&model, eval);
    let n = model.trits.len();
    let report = serde_json::json!({
        "optimizer": match kind {
            Kind::Sgd => "carry-sgd",
            Kind::Sgd16 => "carry-sgd16",
            Kind::Adam => "carry-adam",
        },
        "stochastic_rounding": opt.stochastic,
        "seed": seed, "steps": steps, "lr": lr, "latent_units_per_trit": if kind == Kind::Sgd16 { S16 } else { S },
        "evaluation_offset": eval_offset, "evaluation_count": eval.len(),
        "initial_loss_sum": initial.0, "initial_correct": initial.1,
        "loss_sum": loss, "correct": correct,
        "accuracy": correct as f64 / eval.len() as f64,
        "mean_loss": loss as f64 / eval.len() as f64,
        "transitions": opt.transitions, "reversals": opt.reversals,
        "zero_weights": model.trits.iter().filter(|&&t| t == 0).count(),
        "optimizer_state_bytes": opt.state_bytes(),
        "state_bytes_per_parameter": opt.state_bytes() as f64 / n as f64,
        // Same work proxy as the STE baseline: forward + backward contraction terms per step.
        "contraction_terms": steps * (2 * (784 * 128 + 128 * 10) + 128 * 10) as u64,
        "elapsed_training_ms": elapsed,
        "qualification": "UNKNOWN",
    });
    println!(
        "{}",
        serde_json::to_string(&report).map_err(|e| e.to_string())?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> (Model, Example) {
        let mut m = Model::new(4, 3, 2, &mut Rng(7));
        m.input_shift = 8;
        m.output_shift = 3;
        (
            m,
            Example {
                pixels: vec![200, 17, 0, 255],
                label: 1,
            },
        )
    }

    /// The exact gradient agrees with `Model::sensitivity` up to that function's truncation:
    /// sensitivity = trunc(g / 2¹¹) for every coordinate.
    #[test]
    fn gradient_is_the_untruncated_sensitivity() {
        let (m, x) = tiny();
        let a = m.forward(&x);
        let g = Carry::gradient(&m, &x);
        for (i, gi) in g.iter().enumerate() {
            let s = m.sensitivity(&x, &a, i);
            assert_eq!(i128::from(*gi) / (1 << GRAD_SHIFT), s, "coordinate {i}");
        }
    }

    /// A latent that stays inside its rounding cell never changes the trit; the remainder carries.
    #[test]
    fn remainder_carries_across_steps() {
        let (mut m, x) = tiny();
        let mut opt = Carry::new(&m, Kind::Sgd, 62); // shift so large every update is zero
        let before = m.trits.clone();
        opt.step(&mut m, &x);
        assert_eq!(m.trits, before);
        assert!(
            opt.latent
                .iter()
                .zip(&before)
                .all(|(&l, &t)| l == i32::from(t) * S as i32)
        );
    }

    #[test]
    fn isqrt_is_exact() {
        for x in [0u64, 1, 2, 3, 4, 15, 16, 17, 1 << 40, u64::MAX] {
            let r = isqrt(x);
            assert!(
                r * r <= x && (r + 1).checked_mul(r + 1).is_none_or(|v| v > x),
                "{x}"
            );
        }
    }
}
