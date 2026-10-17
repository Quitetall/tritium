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
#[path = "ternary_carry/noise.rs"]
mod noise;

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

/// How a `carry-sgd` step `g / 2^shift` becomes an integer latent change.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Round {
    /// Arithmetic right shift (floor), the original `carry-sgd` rule.
    Floor,
    Nearest,
    /// Add a uniform integer in [0, 2^shift) before the floor: unbiased, sub-unit steps survive.
    Stochastic,
}

/// Update-rule rungs on `carry-sgd` (ladder axis U). Every default reproduces the original
/// `carry-sgd` bit for bit: keep 100%, no leak, floor rounding, 2^16 units per trit, i32 storage.
#[derive(Clone, Copy, Debug)]
struct Rule {
    /// Percent of the in-cell remainder kept when the trit changes (U3); 0 = reset to the new
    /// cell centre, as BOLD and Bop reset their accumulators.
    keep_pct: i64,
    /// Leak the in-cell remainder toward the cell centre by r >> k every step (U4).
    leak: Option<u32>,
    round: Round,
    /// log2 of latent units per trit (U8); the lr shift keeps its meaning in 2^16-unit terms.
    units_log2: u32,
    /// Emulated storage width in bits (32, 16 or 8): the latent saturates at that range (U8).
    width: u32,
}

impl Rule {
    fn is_original(&self) -> bool {
        self.keep_pct == 100
            && self.leak.is_none()
            && self.round == Round::Floor
            && self.units_log2 == 16
            && self.width == 32
    }
}

impl Default for Rule {
    fn default() -> Self {
        Self {
            keep_pct: 100,
            leak: None,
            round: Round::Floor,
            units_log2: 16,
            width: 32,
        }
    }
}

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
    rule: Rule,
    /// Nonzero gradients whose latent change rounded to zero (sub-unit credit lost).
    underflows: u64,
    /// Latent updates clipped at the storage range.
    saturations: u64,
    /// Gradient source: exact STE (None) or the forward-only noise estimate (ladder axes G/F).
    noise: Option<noise::Config>,
    /// Separate stream for noise draws, so stochastic rounding's stream is unchanged by it.
    noise_rng: Rng,
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
            rule: Rule::default(),
            underflows: 0,
            saturations: 0,
            noise: None,
            noise_rng: Rng(0x0015_EC0D),
        }
    }

    /// Switch `carry-sgd` to an update rule; the latent is re-expressed in the rule's units.
    fn with_rule(mut self, model: &Model, rule: Rule) -> Self {
        assert_eq!(self.kind, Kind::Sgd, "update-rule rungs apply to carry-sgd");
        assert!((1..=16).contains(&rule.units_log2) && [8, 16, 32].contains(&rule.width));
        let s = 1i32 << rule.units_log2;
        self.latent = model.trits.iter().map(|&t| i32::from(t) * s).collect();
        self.rule = rule;
        self
    }

    /// The `carry-sgd` step under `self.rule`, for every coordinate.
    fn step_rule(&mut self, model: &mut Model, g: &[i64]) {
        let r = self.rule;
        let s = 1i64 << r.units_log2;
        // The lr shift is defined in 2^16 units/trit; coarser units shift further.
        let shift = (self.lr + i64::from(16 - r.units_log2)).clamp(0, 62) as u32;
        let bound = match r.width {
            8 => i64::from(i8::MAX),
            16 => i64::from(i16::MAX),
            _ => i64::from(i32::MAX),
        };
        for (i, &grad) in g.iter().enumerate() {
            let step = match r.round {
                Round::Floor => grad >> shift,
                Round::Nearest if shift > 0 => (grad + (1i64 << (shift - 1))) >> shift,
                Round::Nearest => grad,
                Round::Stochastic => (grad + self.rng.below(1u64 << shift) as i64) >> shift,
            };
            let mut latent = i64::from(self.latent[i]);
            if step == 0 && grad != 0 {
                self.underflows += 1;
            }
            if step != 0 {
                let raw = latent - step;
                latent = raw.clamp(-bound, bound);
                if latent != raw {
                    self.saturations += 1;
                }
            }
            let old = i64::from(model.trits[i]);
            let t = ((latent + latent.signum() * (s / 2)) / s).clamp(-1, 1);
            let centre = t * s;
            if t != old && r.keep_pct != 100 {
                latent = centre + (latent - centre) * r.keep_pct / 100;
            }
            if let Some(k) = r.leak {
                let rem = latent - centre;
                latent -= rem >> k;
            }
            self.latent[i] = latent as i32;
            self.apply(model, i, t as i8);
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
        (self.rule.width as usize / 8) * self.latent.capacity()
            + 4 * (self.m.capacity() + self.v.capacity())
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
        let g = self.gradient_of(model, x);
        self.update(model, &g);
    }

    /// The configured gradient source at `model` (the frozen model when flips are deferred).
    fn gradient_of(&mut self, model: &Model, x: &Example) -> Vec<i64> {
        match &self.noise {
            None => Self::gradient(model, x),
            Some(c) => noise::gradient(model, x, c, &mut self.noise_rng),
        }
    }

    /// Apply one gradient to the latent and set `model`'s trits from it.
    fn update(&mut self, model: &mut Model, g: &[i64]) {
        assert_ne!(self.kind, Kind::Rich, "rich computes its own gradient");
        if self.kind == Kind::Sgd && !self.rule.is_original() {
            self.step_rule(model, g);
            return;
        }
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
                if grad != 0 {
                    self.underflows += 1;
                }
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
    let rule = Rule {
        keep_pct: num("--keep-pct", 100)?.clamp(0, 100),
        leak: opt_shift("--leak")?,
        round: match arg(&args, "--round").unwrap_or("floor") {
            "floor" => Round::Floor,
            "nearest" => Round::Nearest,
            "stochastic" => Round::Stochastic,
            other => return Err(format!("unknown --round {other}")),
        },
        units_log2: num("--units-log2", 16)? as u32,
        width: num("--width", 32)? as u32,
    };
    let mut opt = Carry::new(&model, kind, lr).with_rich(rich);
    if !rule.is_original() {
        if kind != Kind::Sgd {
            return Err("--keep-pct/--leak/--round/--units-log2/--width need --optimizer carry-sgd".into());
        }
        opt = opt.with_rule(&model, rule);
    }
    opt.stochastic = arg(&args, "--stochastic") == Some("yes");
    if arg(&args, "--grad").unwrap_or("exact") == "noise" {
        if kind != Kind::Sgd {
            return Err("--grad noise needs --optimizer carry-sgd".into());
        }
        let sigma: f64 = arg(&args, "--sigma")
            .unwrap_or("2")
            .parse()
            .map_err(|_| "invalid --sigma")?;
        // σ in quantization steps → raw units, once, at configuration time.
        let raw = |shift: u8| ((sigma * f64::from(1u32 << shift)).round() as i64).max(1);
        opt.noise = Some(noise::Config {
            noise: match arg(&args, "--noise").unwrap_or("ternary") {
                "ternary" => noise::Noise::Ternary,
                "gauss" => noise::Noise::Gauss,
                other => return Err(format!("unknown --noise {other}")),
            },
            sigma_hidden: raw(model.input_shift),
            sigma_output: raw(model.output_shift),
            draws: num("--draws", 1)?.max(1) as u32,
        });
    } else if arg(&args, "--grad").is_some_and(|g| g != "exact") {
        return Err("--grad is exact or noise".into());
    }
    // Deferred flips (rung F1): the forward model's trits stay frozen for `defer` steps while the
    // latent accumulates; then they are re-derived from the latent. defer = 1 is immediate.
    let defer = num("--defer", 1)?.max(1) as u64;
    if defer > 1 && kind == Kind::Rich {
        return Err("--defer does not apply to rich".into());
    }
    let initial = evaluate(&model, eval);
    let start = std::time::Instant::now();
    let mut frozen = model.clone();
    for step in 0..steps {
        let mut sampler = Rng(seed
            .wrapping_add(step.wrapping_mul(0x9e37_79b9_7f4a_7c15))
            .max(1));
        let i = sampler.below(train.len() as u64) as usize;
        if defer == 1 {
            opt.step(&mut model, &train[i]);
        } else {
            let g = opt.gradient_of(&frozen, &train[i]);
            opt.update(&mut model, &g);
            if (step + 1) % defer == 0 {
                frozen.trits.clone_from(&model.trits);
                frozen.version += 1;
            }
        }
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
    if let Some(path) = arg(&args, "--save-trits") {
        // The deployed trits (averaged when averaging is on): a frozen checkpoint for the
        // offline estimator diagnostic (`ternary_estimator`).
        let trits = opt.averaged_trits().unwrap_or_else(|| model.trits.clone());
        std::fs::write(path, serde_json::to_string(&trits).map_err(|e| e.to_string())?)
            .map_err(|e| format!("{path}: {e}"))?;
    }
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
        "rule": format!("{:?}", opt.rule),
        "underflows": opt.underflows, "saturations": opt.saturations,
        "gradient": match &opt.noise { None => "exact".to_string(), Some(c) => format!("{c:?}") },
        "defer": defer,
        "noisy_evaluations": opt.noise.as_ref().map_or(0, |c| steps * noise::noisy_evaluations(c)),
        "fast_accuracy": fast_correct as f64 / eval.len() as f64,
        "fast_mean_loss": fast_loss as f64 / eval.len() as f64,
        "seed": seed, "steps": steps, "lr": lr, "latent_units_per_trit": if kind == Kind::Sgd16 { S16 } else { 1i64 << opt.rule.units_log2 },
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
        // Fixed trits with every hidden unit able to fire: Rng(7)'s draw leaves all three ReLUs
        // dead on every input, which made every gradient zero and the tests below vacuous.
        m.trits = vec![1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, -1, 0, 1, -1, 1];
        m.version += 1;
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
    /// At a huge shift the floor rule still moves a negative gradient's latent by one unit
    /// (floor(g / 2^62) = −1 for g < 0) — the floor bias the `--round` rung measures.
    #[test]
    fn remainder_carries_across_steps() {
        let (mut m, x) = tiny();
        let g = Carry::gradient(&m, &x);
        let mut opt = Carry::new(&m, Kind::Sgd, 62);
        let before = m.trits.clone();
        opt.step(&mut m, &x);
        assert_eq!(m.trits, before);
        for i in 0..g.len() {
            let expected = i32::from(before[i]) * S as i32 + i32::from(g[i] < 0);
            assert_eq!(opt.latent[i], expected, "coordinate {i}");
        }
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

    fn stream() -> Vec<Example> {
        (0..60u8)
            .map(|k| Example {
                pixels: vec![k.wrapping_mul(37), 200 - k, k, 255 - k],
                label: usize::from(k % 2),
            })
            .collect()
    }

    fn run(kind: Kind, lr: i64, rule: Option<Rule>, stochastic: bool) -> (Vec<i8>, Carry) {
        let (mut m, _) = tiny();
        let mut opt = Carry::new(&m, kind, lr);
        if let Some(r) = rule {
            opt = opt.with_rule(&m, r);
        }
        opt.stochastic = stochastic;
        for x in &stream() {
            opt.step(&mut m, x);
        }
        (m.trits, opt)
    }

    /// The generic rule path with every knob at its identity value is `carry-sgd` bit for bit
    /// (forcing the generic path, which the dispatcher would otherwise skip).
    #[test]
    fn rule_identity_is_carry_sgd() {
        let (a, sgd) = run(Kind::Sgd, 2, None, false);
        let (mut m, _) = tiny();
        let mut opt = Carry::new(&m, Kind::Sgd, 2);
        for x in &stream() {
            let g = Carry::gradient(&m, x);
            opt.step_rule(&mut m, &g);
        }
        assert_eq!(a, m.trits);
        assert_eq!(sgd.latent, opt.latent);
    }

    /// 2^12 units/trit in i16 storage with nearest rounding is `carry-sgd16`'s round-to-nearest
    /// bit for bit, at the equivalent shift (16 − 12 = 4 coarser).
    #[test]
    fn rule_i16_nearest_is_carry_sgd16() {
        let rule = Rule {
            round: Round::Nearest,
            units_log2: 12,
            width: 16,
            ..Rule::default()
        };
        let (a, ra) = run(Kind::Sgd, 1, Some(rule), false);
        let (b, rb) = run(Kind::Sgd16, 5, None, false);
        assert_eq!(a, b);
        let la: Vec<i64> = ra.latent.iter().map(|&v| i64::from(v)).collect();
        let lb: Vec<i64> = rb.latent16.iter().map(|&v| i64::from(v)).collect();
        assert_eq!(la, lb);
    }

    /// keep 0% puts the latent exactly on the new cell centre after every transition.
    #[test]
    fn keep_zero_resets_to_cell_centre() {
        let rule = Rule {
            keep_pct: 0,
            ..Rule::default()
        };
        let (mut m, _) = tiny();
        let mut opt = Carry::new(&m, Kind::Sgd, 0).with_rule(&m, rule);
        let mut checked = 0;
        for x in &stream() {
            let before = m.trits.clone();
            opt.step(&mut m, x);
            for i in 0..m.trits.len() {
                if m.trits[i] != before[i] {
                    assert_eq!(opt.latent[i], i32::from(m.trits[i]) * S as i32);
                    checked += 1;
                }
            }
        }
        assert!(checked > 0, "no transition exercised");
    }

    /// The leak shrinks the in-cell remainder toward the centre (r −= r >> k) after the step,
    /// and never changes a trit by itself.
    #[test]
    fn leak_shrinks_remainder_toward_centre() {
        let (m0, x) = tiny();
        let start = |m: &Model, rule: Rule| {
            let mut opt = Carry::new(m, Kind::Sgd, 62).with_rule(m, rule);
            opt.latent[0] = i32::from(m.trits[0]) * S as i32 + 3 * (S as i32) / 8;
            opt
        };
        let (mut a, mut b) = (m0.clone(), m0.clone());
        let mut plain = start(&a, Rule::default());
        let mut leaky = start(
            &b,
            Rule {
                leak: Some(1),
                ..Rule::default()
            },
        );
        let g = Carry::gradient(&a, &x);
        plain.step_rule(&mut a, &g);
        leaky.step_rule(&mut b, &g);
        assert_eq!(a.trits, b.trits);
        assert_eq!(a.trits, m0.trits);
        for i in 0..g.len() {
            let centre = i64::from(a.trits[i]) * S;
            let rem = i64::from(plain.latent[i]) - centre;
            assert_eq!(i64::from(leaky.latent[i]) - centre, rem - (rem >> 1), "coordinate {i}");
        }
    }

    /// i8 storage saturates at ±127 and counts it.
    #[test]
    fn i8_width_saturates_and_counts() {
        let rule = Rule {
            units_log2: 4,
            width: 8,
            ..Rule::default()
        };
        let (_, opt) = run(Kind::Sgd, 0, Some(rule), false);
        assert!(opt.latent.iter().all(|&l| (-127..=127).contains(&l)));
        assert!(opt.saturations > 0);
    }

    /// The test streams must exercise nonzero gradients and transitions, or the identity tests
    /// above prove nothing.
    #[test]
    fn test_stream_is_not_vacuous() {
        let (m, _) = tiny();
        let active = stream()
            .iter()
            .filter(|x| Carry::gradient(&m, x).iter().any(|&g| g != 0))
            .count();
        assert!(active > 40, "{active}");
        let (_, sgd) = run(Kind::Sgd, 2, None, false);
        assert!(sgd.transitions > 0);
    }

    /// defer = 1 through the split gradient/update path is the immediate step, bit for bit.
    #[test]
    fn split_step_is_step() {
        let (mut a, _) = tiny();
        let mut b = a.clone();
        let mut oa = Carry::new(&a, Kind::Sgd, 2);
        let mut ob = Carry::new(&b, Kind::Sgd, 2);
        for x in &stream() {
            oa.step(&mut a, x);
            let g = ob.gradient_of(&b, x);
            ob.update(&mut b, &g);
        }
        assert_eq!(a.trits, b.trits);
        assert_eq!(oa.latent, ob.latent);
    }

    fn cos(a: &[i64], b: &[i64]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
        let n = |v: &[i64]| v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        dot / (n(a) * n(b))
    }

    /// The integer noise estimate approaches the exact gradient's direction with many draws, at a
    /// comparable scale (same 2⁻¹¹ units), for ternary and Gaussian noise.
    #[test]
    fn noise_gradient_approaches_exact() {
        let (m, _) = tiny();
        for kind in [noise::Noise::Ternary, noise::Noise::Gauss] {
            let c = noise::Config {
                noise: kind,
                // The tiny model's output divides by 8, so a ±2-step hidden perturbation seldom
                // moves any output and every loss difference is zero: use 8 steps here.
                sigma_hidden: 8 << m.input_shift,
                sigma_output: 2 << m.output_shift,
                draws: 4000,
            };
            let mut rng = Rng(9);
            let (mut ex, mut est) = (vec![0i64; m.trits.len()], vec![0i64; m.trits.len()]);
            for x in &stream() {
                for (e, v) in ex.iter_mut().zip(Carry::gradient(&m, x)) {
                    *e += v;
                }
                for (e, v) in est.iter_mut().zip(noise::gradient(&m, x, &c, &mut rng)) {
                    *e += v;
                }
            }
            let split = m.inputs * m.hidden;
            for (lo, hi) in [(0, split), (split, m.trits.len())] {
                let c = cos(&est[lo..hi], &ex[lo..hi]);
                assert!(c > 0.9, "{kind:?} layer {lo}: cosine {c}");
                let ratio = est[lo..hi].iter().map(|v| v.abs()).sum::<i64>() as f64
                    / ex[lo..hi].iter().map(|v| v.abs()).sum::<i64>() as f64;
                // Scale is only loosely tied to the STE surrogate: large noise on this coarse model
                // crosses ReLU kinks and truncation, so the finite difference's slope is smaller.
                assert!((0.3..3.0).contains(&ratio), "{kind:?} layer {lo}: scale {ratio}");
            }
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
