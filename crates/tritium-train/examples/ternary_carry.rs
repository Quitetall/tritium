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
    /// `carry-sgd` plus any of the rich-state options in [`Rich`].
    Rich,
}

/// Extra per-weight optimizer state for `--optimizer rich`, each part optional and composable.
/// Ternary weights cost ~0.2 B, so a mixed-precision AdamW footprint (~16 B/weight) leaves room
/// for all of these at once.
#[derive(Clone, Copy, Debug, Default)]
struct Rich {
    /// Momentum EMA of the step, β = 1 − 2^-k (+4 B/weight).
    momentum: Option<u32>,
    /// Exact diagonal Gauss-Newton curvature, EMA 2^-k over examples, and damping δ = layer mean
    /// curvature >> d. Step = Newton fraction 2^-lr of g/(c + δ) (+8 B/weight).
    curvature: Option<(u32, u32)>,
    /// Halve a weight's step on each reversal; recover one level every 1024 steps (+1 B/weight).
    dampen: bool,
    /// Polyak/EMA average of the latent, 2^-k; the DEPLOYED trits are round(average) (+4 B/weight).
    average: Option<u32>,
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
    rich: Rich,
    mom: Vec<i32>,
    curv: Vec<u64>,
    damp: Vec<u8>,
    avg: Vec<i32>,
    steps: u64,
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
            rich: Rich::default(),
            mom: Vec::new(),
            curv: Vec::new(),
            damp: Vec::new(),
            avg: Vec::new(),
            steps: 0,
        }
    }

    fn with_rich(mut self, rich: Rich) -> Self {
        let n = self.latent.len();
        self.rich = rich;
        if rich.momentum.is_some() {
            self.mom = vec![0; n];
        }
        if rich.curvature.is_some() {
            self.curv = vec![0; n];
        }
        if rich.dampen {
            self.damp = vec![0; n];
        }
        if rich.average.is_some() {
            self.avg = self.latent.clone();
        }
        self
    }

    /// Trits from the averaged latent (the deployed model when averaging is on).
    fn averaged_trits(&self) -> Option<Vec<i8>> {
        self.rich.average?;
        Some(
            self.avg
                .iter()
                .map(|&a| {
                    let a = i64::from(a);
                    ((a + a.signum() * (S / 2)) / S).clamp(-1, 1) as i8
                })
                .collect(),
        )
    }

    /// Exact per-example diagonal Gauss-Newton of the squared loss, in 2⁻²² units:
    /// output weight (k, j): 2·(h_j/8)² = h_j²·2¹⁷;
    /// input weight (j, i), active ReLU: 2·Σ_k (w2[k,j]·p_i / 2¹¹)² = 2·p_i²·Σ_k w2[k,j]².
    fn gauss_newton(model: &Model, x: &Example) -> Vec<u64> {
        let a = model.forward(x);
        let split = model.inputs * model.hidden;
        let mut c = vec![0u64; model.trits.len()];
        for k in 0..model.outputs {
            for j in 0..model.hidden {
                let h = a.hidden[j].unsigned_abs();
                c[split + k * model.hidden + j] = (h * h) << 17;
            }
        }
        for j in 0..model.hidden {
            if a.hidden[j] == 0 {
                continue;
            }
            let s2: u64 = (0..model.outputs)
                .map(|k| u64::from(model.trits[split + k * model.hidden + j] != 0))
                .sum();
            if s2 == 0 {
                continue;
            }
            for (ci, &p) in c[j * model.inputs..(j + 1) * model.inputs]
                .iter_mut()
                .zip(&x.pixels)
            {
                *ci = 2 * u64::from(p) * u64::from(p) * s2;
            }
        }
        c
    }

    /// Allocated optimizer state (latent + moments), from actual capacities.
    fn state_bytes(&self) -> usize {
        4 * (self.latent.capacity() + self.m.capacity() + self.v.capacity())
            + 2 * self.latent16.capacity()
            + 4 * (self.mom.capacity() + self.avg.capacity())
            + 8 * self.curv.capacity()
            + self.damp.capacity()
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
        if self.kind == Kind::Rich {
            self.step_rich(model, x);
            return;
        }
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
                Kind::Sgd16 | Kind::Rich => unreachable!("handled above"),
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
                Kind::Sgd | Kind::Rich => i64::from(i32::MAX),
                Kind::Adam | Kind::Sgd16 => 8 * S,
            };
            let latent = (i64::from(self.latent[i]) + delta).clamp(-bound, bound);
            self.latent[i] = latent as i32;
            // The trit is the rounded latent: the carried remainder is never discarded.
            let t = ((latent + latent.signum() * (S / 2)) / S).clamp(-1, 1) as i8;
            self.apply(model, i, t);
        }
    }

    fn step_rich(&mut self, model: &mut Model, x: &Example) {
        let g = Self::gradient(model, x);
        let n = g.len();
        let split = model.inputs * model.hidden;
        // Curvature: EMA of the exact per-example Gauss-Newton diagonal, and per-layer damping.
        let mut delta_layer = [0u128; 2];
        if let Some((ema, dshift)) = self.rich.curvature {
            let gn = Self::gauss_newton(model, x);
            for (c, &e) in self.curv.iter_mut().zip(&gn) {
                let cur = i128::from(*c);
                *c = (cur + ((i128::from(e) - cur) >> ema)).max(0) as u64;
            }
            for (layer, range) in [(0usize, 0..split), (1, split..n)] {
                let len = range.len() as u128;
                let mean = self.curv[range]
                    .iter()
                    .map(|&c| u128::from(c))
                    .sum::<u128>()
                    / len;
                delta_layer[layer] = (mean >> dshift).max(1);
            }
        }
        for (i, &grad) in g.iter().enumerate() {
            // Step direction, in latent units for curvature, in gradient units otherwise.
            let mut step: i64 = if self.rich.curvature.is_some() {
                let layer = usize::from(i >= split);
                let denom = u128::from(self.curv[i]) + delta_layer[layer];
                // Newton fraction 2^-lr of g/(c+δ): Δlatent = g·2^(27−lr) / (c + δ).
                let num = i128::from(grad) << (27 - self.lr.clamp(0, 27) as u32);
                (num / denom as i128).clamp(i128::from(i32::MIN) / 2, i128::from(i32::MAX) / 2)
                    as i64
            } else {
                grad
            };
            if let Some(k) = self.rich.momentum {
                let m = i64::from(self.mom[i]);
                let next = (m + ((step - m) >> k)).clamp(i64::from(i32::MIN), i64::from(i32::MAX));
                self.mom[i] = next as i32;
                step = next;
            }
            let mut delta = if self.rich.curvature.is_some() {
                -step
            } else {
                -(step >> self.lr.clamp(0, 62))
            };
            if self.rich.dampen {
                delta >>= self.damp[i];
            }
            if delta != 0 {
                // Same bound as carry-sgd, so `rich` with no options is carry-sgd exactly.
                let bound = i64::from(i32::MAX);
                let latent = (i64::from(self.latent[i]) + delta).clamp(-bound, bound);
                self.latent[i] = latent as i32;
                let t = ((latent + latent.signum() * (S / 2)) / S).clamp(-1, 1) as i8;
                let reversals = self.reversals;
                self.apply(model, i, t);
                if self.rich.dampen && self.reversals > reversals {
                    self.damp[i] = (self.damp[i] + 1).min(8);
                }
            }
            if let Some(k) = self.rich.average {
                let a = i64::from(self.avg[i]);
                self.avg[i] = (a + ((i64::from(self.latent[i]) - a) >> k)) as i32;
            }
        }
        self.steps += 1;
        if self.rich.dampen && self.steps.is_multiple_of(1024) {
            for d in &mut self.damp {
                *d = d.saturating_sub(1);
            }
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
        "rich" => Kind::Rich,
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
    let opt_shift = |k: &str| -> Result<Option<u32>, String> {
        arg(&args, k)
            .map(|v| v.parse().map_err(|_| format!("invalid {k}")))
            .transpose()
    };
    let rich = Rich {
        momentum: opt_shift("--momentum")?,
        curvature: match (opt_shift("--curv-ema")?, opt_shift("--curv-damp")?) {
            (Some(e), Some(d)) => Some((e, d)),
            (None, None) => None,
            _ => return Err("--curv-ema and --curv-damp go together".into()),
        },
        dampen: arg(&args, "--dampen") == Some("yes"),
        average: opt_shift("--average")?,
    };
    if kind != Kind::Rich
        && (rich.momentum.is_some()
            || rich.curvature.is_some()
            || rich.dampen
            || rich.average.is_some())
    {
        return Err("rich-state options need --optimizer rich".into());
    }
    let mut opt = Carry::new(&model, kind, lr).with_rich(rich);
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
    // With averaging on, the deployed model is round(average); the fast latent is reported too.
    let (fast_loss, fast_correct) = evaluate(&model, eval);
    let (loss, correct) = match opt.averaged_trits() {
        Some(trits) => {
            let mut deployed = model.clone();
            deployed.trits = trits;
            evaluate(&deployed, eval)
        }
        None => (fast_loss, fast_correct),
    };
    let n = model.trits.len();
    let report = serde_json::json!({
        "optimizer": match kind {
            Kind::Sgd => "carry-sgd",
            Kind::Sgd16 => "carry-sgd16",
            Kind::Adam => "carry-adam",
            Kind::Rich => "rich",
        },
        "stochastic_rounding": opt.stochastic,
        "rich": format!("{:?}", opt.rich),
        "fast_accuracy": fast_correct as f64 / eval.len() as f64,
        "fast_mean_loss": fast_loss as f64 / eval.len() as f64,
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

    /// `rich` with every option off is `carry-sgd`, bit for bit.
    #[test]
    fn rich_without_options_is_carry_sgd() {
        let (m0, _) = tiny();
        let xs: Vec<Example> = (0..40u8)
            .map(|k| Example {
                pixels: vec![k.wrapping_mul(37), 200 - k, k, 255 - k],
                label: usize::from(k % 2),
            })
            .collect();
        let (mut a, mut b) = (m0.clone(), m0.clone());
        let mut sgd = Carry::new(&a, Kind::Sgd, 2);
        let mut rich = Carry::new(&b, Kind::Rich, 2).with_rich(Rich::default());
        for x in &xs {
            sgd.step(&mut a, x);
            rich.step(&mut b, x);
        }
        assert_eq!(a.trits, b.trits);
        assert_eq!(sgd.latent, rich.latent);
    }

    /// Gauss-Newton diagonal agrees with a finite second difference of the unrounded output map
    /// for an output-layer weight: d²L/dw² = 2·(h/8)².
    #[test]
    fn gauss_newton_output_layer_is_exact() {
        let (m, x) = tiny();
        let a = m.forward(&x);
        let c = Carry::gauss_newton(&m, &x);
        let split = m.inputs * m.hidden;
        for j in 0..m.hidden {
            let h = a.hidden[j] as u64;
            assert_eq!(c[split + j], (h * h) << 17);
        }
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
