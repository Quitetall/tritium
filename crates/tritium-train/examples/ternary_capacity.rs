//! Capacity levers for integer ternary training: learned per-row scales, more trit planes per
//! weight, and more width — on the carried-latent + averaging optimizer (`ternary_carry --optimizer
//! rich --average 10`), which matches its fp32 STE counterpart and stalls at the model's ceiling.
//!
//! The network is `ternary_lab`'s 784→H→10 integer ReLU net, generalized:
//!
//! ```text
//! h_j   = max(0, trunc(Σ_i W1_ji·p_i · s1_j / (Q·2⁸·3^(T−1))))
//! out_k =        trunc(Σ_j W2_kj·h_j · s2_k / (Q·2³·3^(T−1)))
//! ```
//!
//! `W` are integers in `[−K, K]`, `K = (3^T − 1)/2` — a sum of T trit planes, so every dot product
//! is still add/subtract; `s` are per-row Q12 scales (`Q = 4096` is 1.0), fixed at Q unless
//! learned, applied once per row. With T = 1, H = 128 and fixed scales this IS `Model::forward`
//! (unit-tested). Training is integer-only: carried latents in real-weight units (2¹⁶ per 1.0),
//! STE gradients as exact integer rationals in 2⁻²⁰ units, latent EMA averaging for the deployed
//! weights and scales.
#[allow(dead_code)]
#[path = "ternary_lab/model.rs"]
mod model;
#[allow(dead_code)]
#[path = "ternary_lab/numeric.rs"]
mod numeric;

use model::Example;
use numeric::Rng;
use std::path::Path;

const Q: i64 = 1 << 12;
/// Latent units per 1.0 of real weight.
const S: i64 = 1 << 16;

#[derive(Clone, Debug)]
struct Net {
    hidden: usize,
    planes: u32,
    /// Integer weights in [−K, K], real value W / 3^(T−1).
    w1: Vec<i64>,
    w2: Vec<i64>,
    s1: Vec<i64>,
    s2: Vec<i64>,
}

struct Fwd {
    raw1: Vec<i64>,
    h: Vec<i64>,
    raw2: Vec<i64>,
    out: Vec<i64>,
}

fn pow3(e: u32) -> i64 {
    3i64.pow(e)
}

impl Net {
    fn mult(&self) -> i64 {
        pow3(self.planes - 1)
    }
    fn kmax(&self) -> i64 {
        (pow3(self.planes) - 1) / 2
    }
    fn forward(&self, x: &Example) -> Fwd {
        let m = self.mult();
        let (d1, d2) = (i128::from(Q * 256 * m), i128::from(Q * 8 * m));
        let raw1: Vec<i64> = self
            .w1
            .as_chunks::<784>()
            .0
            .iter()
            .map(|row| {
                row.iter()
                    .zip(&x.pixels)
                    .map(|(&w, &p)| w * i64::from(p))
                    .sum()
            })
            .collect();
        let h: Vec<i64> = raw1
            .iter()
            .zip(&self.s1)
            .map(|(&r, &s)| ((i128::from(r) * i128::from(s)) / d1).max(0) as i64)
            .collect();
        let raw2: Vec<i64> = self
            .w2
            .chunks_exact(self.hidden)
            .map(|row| row.iter().zip(&h).map(|(&w, &v)| w * v).sum())
            .collect();
        let out = raw2
            .iter()
            .zip(&self.s2)
            .map(|(&r, &s)| ((i128::from(r) * i128::from(s)) / d2) as i64)
            .collect();
        Fwd { raw1, h, raw2, out }
    }
    fn loss_correct(&self, x: &Example) -> (i128, bool) {
        let f = self.forward(x);
        let loss = f
            .out
            .iter()
            .enumerate()
            .map(|(k, &o)| {
                let e = i128::from(o) - if k == x.label { 256 } else { 0 };
                e * e
            })
            .sum();
        let best = (0..10)
            .max_by_key(|&k| (f.out[k], std::cmp::Reverse(k)))
            .unwrap();
        (loss, best == x.label)
    }
}

/// STE gradients in 2⁻²⁰ units: weights w.r.t. their REAL value (W / 3^(T−1)), scales w.r.t. one
/// Q12 unit. Identity through truncation; ReLU gated at the quantized activation.
struct Grad {
    w1: Vec<i64>,
    w2: Vec<i64>,
    s1: Vec<i64>,
    s2: Vec<i64>,
}

#[allow(clippy::needless_range_loop)]
fn gradient(net: &Net, x: &Example) -> Grad {
    let f = net.forward(x);
    let hdim = net.hidden;
    let m = i128::from(net.mult());
    let e: Vec<i128> = (0..10)
        .map(|k| i128::from(f.out[k]) - if k == x.label { 256 } else { 0 })
        .collect();
    let mut g = Grad {
        w1: vec![0; net.w1.len()],
        w2: vec![0; net.w2.len()],
        s1: vec![0; hdim],
        s2: vec![0; 10],
    };
    // dout_k/dW2real_kj = h_j·s2_k/(Q·8): grad·2²⁰ = 2e·h·s2·2⁵.
    for k in 0..10 {
        let s2 = i128::from(net.s2[k]);
        for j in 0..hdim {
            g.w2[k * hdim + j] = (2 * e[k] * i128::from(f.h[j]) * s2 * 32) as i64;
        }
        // dL/ds2 per unit: 2e·raw2/(Q·8·3^(T−1)) → ·2²⁰ = 2e·raw2·2⁵/3^(T−1).
        g.s2[k] = (2 * e[k] * i128::from(f.raw2[k]) * 32 / m) as i64;
    }
    for j in 0..hdim {
        if f.h[j] == 0 {
            continue;
        }
        // dL/dh_j = Nh / (Q·8·3^(T−1)), Nh = Σ_k 2e_k·W2_kj·s2_k.
        let nh: i128 = (0..10)
            .map(|k| 2 * e[k] * i128::from(net.w2[k * hdim + j]) * i128::from(net.s2[k]))
            .sum();
        if nh == 0 {
            continue;
        }
        let s1 = i128::from(net.s1[j]);
        // dL/dW1real = dL/dh·p·s1/(Q·256) → ·2²⁰ = Nh·p·s1 / (2¹⁵·3^(T−1)).
        let den = (1i128 << 15) * m;
        for (gi, &p) in g.w1[j * 784..(j + 1) * 784].iter_mut().zip(&x.pixels) {
            *gi = (nh * i128::from(p) * s1 / den) as i64;
        }
        // dL/ds1 per unit = dL/dh·raw1/(Q·256·3^(T−1)) → ·2²⁰ = Nh·raw1 / (2¹⁵·9^(T−1)).
        g.s1[j] = (nh * i128::from(f.raw1[j]) / ((1i128 << 15) * m * m)) as i64;
    }
    g
}

struct Trainer {
    net: Net,
    lat1: Vec<i64>,
    lat2: Vec<i64>,
    avg1: Vec<i64>,
    avg2: Vec<i64>,
    avg_s1: Vec<i64>,
    avg_s2: Vec<i64>,
    w_shift: u32,
    /// `None`: scales fixed at Q.
    s_shift: Option<u32>,
    avg_k: u32,
}

impl Trainer {
    fn new(net: Net, w_shift: u32, s_shift: Option<u32>, avg_k: u32) -> Self {
        let m = net.mult();
        let lat = |w: &[i64]| -> Vec<i64> { w.iter().map(|&v| v * S / m).collect() };
        let (lat1, lat2) = (lat(&net.w1), lat(&net.w2));
        Self {
            avg1: lat1.clone(),
            avg2: lat2.clone(),
            avg_s1: net.s1.clone(),
            avg_s2: net.s2.clone(),
            lat1,
            lat2,
            net,
            w_shift,
            s_shift,
            avg_k,
        }
    }

    fn quantize(latent: i64, m: i64, k: i64) -> i64 {
        let v = latent * m;
        ((v + v.signum() * (S / 2)) / S).clamp(-k, k)
    }

    #[allow(clippy::needless_range_loop)]
    fn step(&mut self, x: &Example) {
        let g = gradient(&self.net, x);
        let (m, kmax) = (self.net.mult(), self.net.kmax());
        let (ws, ak) = (self.w_shift, self.avg_k);
        for (lat, avg, w, gw) in [
            (&mut self.lat1, &mut self.avg1, &mut self.net.w1, &g.w1),
            (&mut self.lat2, &mut self.avg2, &mut self.net.w2, &g.w2),
        ] {
            for i in 0..lat.len() {
                lat[i] -= gw[i] >> ws;
                w[i] = Self::quantize(lat[i], m, kmax);
                avg[i] += (lat[i] - avg[i]) >> ak;
            }
        }
        if let Some(ss) = self.s_shift {
            for (s, avg, gs) in [
                (&mut self.net.s1, &mut self.avg_s1, &g.s1),
                (&mut self.net.s2, &mut self.avg_s2, &g.s2),
            ] {
                for i in 0..s.len() {
                    // Round to nearest: a floor shift turns every small negative gradient into −1
                    // and makes the scales drift up one unit per step regardless of the signal.
                    let step = (gs[i] + (1i64 << (ss - 1))) >> ss;
                    s[i] = (s[i] - step).clamp(1, 1 << 24);
                    avg[i] += (s[i] - avg[i]) >> ak;
                }
            }
        }
    }

    /// The deployed network: averaged latents rounded, averaged scales.
    fn deployed(&self) -> Net {
        let (m, kmax) = (self.net.mult(), self.net.kmax());
        let mut net = self.net.clone();
        net.w1 = self
            .avg1
            .iter()
            .map(|&l| Self::quantize(l, m, kmax))
            .collect();
        net.w2 = self
            .avg2
            .iter()
            .map(|&l| Self::quantize(l, m, kmax))
            .collect();
        if self.s_shift.is_some() {
            net.s1.clone_from(&self.avg_s1);
            net.s2.clone_from(&self.avg_s2);
        }
        net
    }
}

/// Same initial trits as `Model::new` for the same seed (identical draw order at H = 128); extra
/// planes start at the same values (W = trit·3^(T−1)).
fn init(hidden: usize, planes: u32, seed: u64) -> Net {
    let mut rng = Rng(seed);
    let m = pow3(planes - 1);
    let mut draw =
        |n: usize| -> Vec<i64> { (0..n).map(|_| (rng.below(3) as i64 - 1) * m).collect() };
    let w1 = draw(784 * hidden);
    let w2 = draw(hidden * 10);
    Net {
        hidden,
        planes,
        w1,
        w2,
        s1: vec![Q; hidden],
        s2: vec![Q; 10],
    }
}

fn evaluate(net: &Net, data: &[Example]) -> (i128, usize) {
    data.iter().fold((0, 0), |(l, c), x| {
        let (loss, ok) = net.loss_correct(x);
        (l + loss, c + usize::from(ok))
    })
}

fn arg<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let num = |k: &str, d: i64| -> Result<i64, String> {
        arg(&args, k).map_or(Ok(d), |v| v.parse().map_err(|_| format!("invalid {k}")))
    };
    let data = arg(&args, "--data").ok_or("--data MNIST_IDX_DIR required")?;
    let seed = num("--seed", 1)? as u64;
    let steps = num("--steps", 20_000)? as u64;
    let hidden = num("--hidden", 128)? as usize;
    let planes = num("--planes", 1)? as u32;
    // Base: carry-sgd shift 4 on 2⁻¹¹ gradients = shift 13 on 2⁻²⁰.
    let w_shift = num("--w-shift", 13)? as u32;
    let s_shift = arg(&args, "--s-shift")
        .map(|v| {
            v.parse::<u32>()
                .map_err(|_| "invalid --s-shift".to_string())
        })
        .transpose()?;
    let avg_k = num("--average", 10)? as u32;
    // Fixed global scale multiplier for BOTH layers in Q12 (control: is a learned per-row scale
    // more than a better global scale?).
    let global_scale = num("--global-scale", Q)?;
    let eval_offset = num("--eval-offset", 1000)? as usize;
    let eval_limit = num("--eval-limit", 4000)? as usize;
    if !(1..=4).contains(&planes) || hidden == 0 {
        return Err("planes must be 1..=4 and hidden > 0".into());
    }
    let p = Path::new(data);
    let all = model::mnist(
        &p.join("train-images-idx3-ubyte"),
        &p.join("train-labels-idx1-ubyte"),
    )?;
    let (train, eval) = all.split_at(55_000);
    let eval = &eval[eval_offset..eval_offset + eval_limit];
    let mut net0 = init(hidden, planes, seed);
    net0.s1 = vec![global_scale; hidden];
    net0.s2 = vec![global_scale; 10];
    let mut t = Trainer::new(net0, w_shift, s_shift, avg_k);
    let start = std::time::Instant::now();
    for step in 0..steps {
        let mut sampler = Rng(seed
            .wrapping_add(step.wrapping_mul(0x9e37_79b9_7f4a_7c15))
            .max(1));
        let i = sampler.below(train.len() as u64) as usize;
        t.step(&train[i]);
    }
    let elapsed = start.elapsed().as_millis();
    let deployed = t.deployed();
    let (loss, correct) = evaluate(&deployed, eval);
    let (fast_loss, fast_correct) = evaluate(&t.net, eval);
    let params = deployed.w1.len() + deployed.w2.len();
    let scale_rows = if s_shift.is_some() { hidden + 10 } else { 0 };
    let weight_bits = params as f64 * f64::from(planes) * 3f64.log2() + scale_rows as f64 * 16.0;
    let report = serde_json::json!({
        "seed": seed, "steps": steps, "hidden": hidden, "planes": planes,
        "learned_scales": s_shift.is_some(), "global_scale": global_scale, "w_shift": w_shift, "s_shift": s_shift,
        "average": avg_k, "evaluation_offset": eval_offset, "evaluation_count": eval.len(),
        "accuracy": correct as f64 / eval.len() as f64,
        "mean_loss": loss as f64 / eval.len() as f64,
        "fast_accuracy": fast_correct as f64 / eval.len() as f64,
        "fast_mean_loss": fast_loss as f64 / eval.len() as f64,
        "parameters": params, "model_kib": weight_bits / 8.0 / 1024.0,
        "optimizer_state_bytes_per_parameter": 16,
        "contraction_terms_per_step": 2 * params as u64 * u64::from(planes) + 10 * hidden as u64,
        "elapsed_training_ms": elapsed, "qualification": "UNKNOWN",
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
    use model::Model;

    fn example(k: u64) -> Example {
        let mut r = Rng(k + 1);
        Example {
            pixels: (0..784).map(|_| r.below(256) as u8).collect(),
            label: (k % 10) as usize,
        }
    }

    /// T = 1, H = 128, fixed scales: identical to `Model::forward` for the same seed.
    #[test]
    fn base_configuration_is_the_lab_model() {
        let model = Model::new(784, 128, 10, &mut Rng(5));
        let net = init(128, 1, 5);
        let trits: Vec<i64> = model.trits.iter().map(|&t| i64::from(t)).collect();
        assert_eq!([net.w1.clone(), net.w2.clone()].concat(), trits);
        for k in 0..4 {
            let x = example(k);
            let a = model.forward(&x);
            let f = net.forward(&x);
            assert_eq!(a.hidden, f.h);
            assert_eq!(a.output, f.out);
        }
    }

    /// The base-configuration gradient is exactly 2⁹ × `Model::sensitivity`'s numerator (the
    /// 2⁻¹¹ carry gradient), i.e. the same optimizer trajectory up to the shift.
    #[test]
    fn base_gradient_is_the_carry_gradient() {
        let model = Model::new(784, 128, 10, &mut Rng(9));
        let net = init(128, 1, 9);
        let x = example(3);
        let a = model.forward(&x);
        let g = gradient(&net, &x);
        let all: Vec<i64> = [g.w1, g.w2].concat();
        for (i, &gi) in all.iter().enumerate() {
            assert_eq!(gi % (1 << 9), 0, "coordinate {i} not a multiple of 2⁹");
            assert_eq!(
                i128::from(gi >> 9) / (1 << 11),
                model.sensitivity(&x, &a, i)
            );
        }
    }

    /// Extra planes represent the base trits exactly at initialization.
    #[test]
    fn more_planes_start_at_the_same_function() {
        for planes in 2..=3 {
            let base = init(32, 1, 4);
            let net = init(32, planes, 4);
            let x = example(1);
            assert_eq!(base.forward(&x).out, net.forward(&x).out, "T={planes}");
        }
    }
}
