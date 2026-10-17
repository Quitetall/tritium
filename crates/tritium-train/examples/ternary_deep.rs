//! Width, depth and timing study for fused forward-only training (`ternary_carry --grad noise`
//! generalized): an integer ternary MLP of any widths, trained by the carry (carry-sgd, i32 latent,
//! remainder kept) from either the exact STE gradient or the forward-only noise estimate.
//!
//! - `--sizes 784,128,10` with shifts 8,3 is the `ternary_lab` network: same init, same forward,
//!   same gradient (unit tests). Other sizes take shift ⌊log2(fan_in)/2⌋ per later layer unless
//!   `--shifts` is given.
//! - `--grad noise`: each layer's raw sums are perturbed separately with integer ternary antithetic
//!   noise (σ = 2 quantization steps); the K draws × 2 signs are pushed through the later layers
//!   **as one batch**, so each later weight row is read once per batch of draws (the "fat forward").
//! - `--buffer W`: per-example update vectors (unit error, layer input) are buffered and applied to
//!   the carry in one pass every W examples — one carry read/write per W examples. The forward's
//!   trits stay frozen in between. W = 1 is the immediate step, bit for bit.
//! - `--time N`: single-thread wall-clock per example, split into forward, credit (backprop's error
//!   propagation or the noisy draws) and update (outer product + carry), over N examples.
#[allow(dead_code)]
#[path = "ternary_lab/model.rs"]
mod model;
#[allow(dead_code)]
#[path = "ternary_lab/numeric.rs"]
mod numeric;

use model::Example;
use numeric::Rng;
use std::path::Path;
use std::time::Instant;

/// Latent units per trit step.
const S: i64 = 1 << 16;
/// The lr shift is expressed in the lab's 2⁻¹¹ gradient units, so grids are comparable.
const REF_UNITS: u32 = 11;
const TARGET: i64 = 256;
/// Fixed-point fraction bits of the per-unit noise estimate.
const FRAC: u32 = 16;

#[derive(Clone)]
struct Net {
    /// Layer widths, input first.
    sizes: Vec<usize>,
    /// Right shift after each layer's raw sum.
    shifts: Vec<u32>,
    /// Row-major `[out][in]` trits per layer.
    w: Vec<Vec<i8>>,
}

/// Raw sums and activations of one forward pass: `acts[0]` is the input.
struct Pass {
    raws: Vec<Vec<i64>>,
    acts: Vec<Vec<i64>>,
}

impl Net {
    /// Same draw order as `Model::new` (first layer, then the next, row-major), so 784,128,10
    /// with one seed reproduces the lab model's initial trits.
    fn new(sizes: &[usize], shifts: &[u32], rng: &mut Rng) -> Self {
        let w = sizes
            .windows(2)
            .map(|p| (0..p[0] * p[1]).map(|_| rng.below(3) as i8 - 1).collect())
            .collect();
        Self { sizes: sizes.to_vec(), shifts: shifts.to_vec(), w }
    }

    fn layers(&self) -> usize {
        self.w.len()
    }

    fn act(&self, l: usize, raw: i64) -> i64 {
        let v = raw / (1i64 << self.shifts[l]);
        if l + 1 == self.layers() { v } else { v.max(0) }
    }

    fn forward(&self, x: &[i64]) -> Pass {
        let mut acts = vec![x.to_vec()];
        let mut raws = Vec::new();
        for l in 0..self.layers() {
            let (n_in, n_out) = (self.sizes[l], self.sizes[l + 1]);
            let input = &acts[l];
            let raw: Vec<i64> = (0..n_out)
                .map(|j| {
                    self.w[l][j * n_in..(j + 1) * n_in]
                        .iter()
                        .zip(input)
                        .map(|(&t, &v)| i64::from(t) * v)
                        .sum()
                })
                .collect();
            acts.push(raw.iter().map(|&r| self.act(l, r)).collect());
            raws.push(raw);
        }
        Pass { raws, acts }
    }

    fn loss(out: &[i64], label: usize) -> i64 {
        out.iter()
            .enumerate()
            .map(|(k, &v)| {
                let e = v - if k == label { TARGET } else { 0 };
                e * e
            })
            .sum()
    }

    /// Losses of a batch of perturbed raw vectors at layer `l`, pushed through layers l..L with
    /// every weight row read once per batch.
    fn batch_losses(&self, l: usize, raws: &[Vec<i64>], label: usize) -> Vec<i64> {
        let mut cur: Vec<Vec<i64>> = raws.iter().map(|r| r.iter().map(|&v| self.act(l, v)).collect()).collect();
        for m in l + 1..self.layers() {
            let (n_in, n_out) = (self.sizes[m], self.sizes[m + 1]);
            let mut next = vec![vec![0i64; n_out]; cur.len()];
            for j in 0..n_out {
                let row = &self.w[m][j * n_in..(j + 1) * n_in];
                for (b, v) in cur.iter().enumerate() {
                    let raw: i64 = row.iter().zip(v).map(|(&t, &a)| i64::from(t) * a).sum();
                    next[b][j] = self.act(m, raw);
                }
            }
            cur = next;
        }
        cur.iter().map(|o| Self::loss(o, label)).collect()
    }

    /// Denominator exponent of ∂ℓ/∂raw_l: D_l = Σ_{m ≥ l} shift_m.
    fn denom(&self, l: usize) -> u32 {
        self.shifts[l..].iter().sum()
    }
}

/// Per-layer unit error in fixed point: ∂ℓ/∂raw_l scaled by 2^(D_0 + FRAC), the common unit.
type UnitErrors = Vec<Vec<i128>>;

/// Exact STE unit errors (ReLU gated at the quantized activation, as `Model::sensitivity`).
fn exact_units(net: &Net, p: &Pass, label: usize) -> UnitErrors {
    let big = net.denom(0);
    let last = net.layers() - 1;
    // Numerators N_l with ∂ℓ/∂raw_l = N_l / 2^(D_l).
    let mut num = vec![Vec::new(); net.layers()];
    num[last] = p.acts[last + 1]
        .iter()
        .enumerate()
        .map(|(k, &o)| 2 * i128::from(o - if k == label { TARGET } else { 0 }))
        .collect();
    for l in (0..last).rev() {
        let n_mid = net.sizes[l + 1];
        // Wᵀδ by rows: each weight row is read once, contiguously, and scattered into the sum —
        // the backward pass at the same memory pattern as the forward.
        let mut acc = vec![0i128; n_mid];
        for (k, &d) in num[l + 1].iter().enumerate() {
            if d == 0 {
                continue;
            }
            for (a, &t) in acc.iter_mut().zip(&net.w[l + 1][k * n_mid..(k + 1) * n_mid]) {
                *a += i128::from(t) * d;
            }
        }
        for (a, &h) in acc.iter_mut().zip(&p.acts[l + 1]) {
            if h == 0 {
                *a = 0; // ReLU gated at the quantized activation.
            }
        }
        num[l] = acc;
    }
    num.into_iter()
        .enumerate()
        .map(|(l, v)| v.into_iter().map(|n| n << (big - net.denom(l) + FRAC)).collect())
        .collect()
}

/// Forward-only unit errors: per layer, K ternary antithetic draws pushed through as one batch.
fn noise_units(net: &Net, p: &Pass, label: usize, draws: usize, sigma_steps: i64, rng: &mut Rng) -> UnitErrors {
    let big = net.denom(0);
    (0..net.layers())
        .map(|l| {
            let n = net.sizes[l + 1];
            let sigma = sigma_steps << net.shifts[l];
            let noise: Vec<Vec<i64>> = (0..draws)
                .map(|_| (0..n).map(|_| (rng.below(3) as i64 - 1) * sigma).collect())
                .collect();
            let batch: Vec<Vec<i64>> = noise
                .iter()
                .flat_map(|a| {
                    [1i64, -1].map(|s| p.raws[l].iter().zip(a).map(|(&r, &v)| r + s * v).collect())
                })
                .collect();
            let losses = net.batch_losses(l, &batch, label);
            let mut sum = vec![0i128; n];
            for (d, a) in noise.iter().enumerate() {
                let diff = i128::from(losses[2 * d] - losses[2 * d + 1]);
                if diff != 0 {
                    for (s, &v) in sum.iter_mut().zip(a) {
                        *s += diff * i128::from(v);
                    }
                }
            }
            // ê = Σ d·a / (2K·E[a²]), E[a²] = 2σ²/3  →  3·Σ d·a / (4Kσ²); scale 2^(D_0 + FRAC).
            let divisor = 4 * draws as i128 * i128::from(sigma) * i128::from(sigma);
            sum.into_iter()
                .map(|s| {
                    let v = 3 * s << (big + FRAC);
                    (v + v.signum() * divisor / 2) / divisor
                })
                .collect()
        })
        .collect()
}

struct Carry {
    latent: Vec<Vec<i32>>,
    /// Effective right shift on gradients in 2^-(D_0) units.
    shift: u32,
    transitions: u64,
    /// Buffered (unit errors, layer inputs) awaiting one carry pass.
    pending: Vec<(UnitErrors, Vec<Vec<i64>>)>,
}

impl Carry {
    fn new(net: &Net, lr: u32) -> Self {
        Self {
            latent: net.w.iter().map(|w| w.iter().map(|&t| i32::from(t) * S as i32).collect()).collect(),
            shift: lr + net.denom(0) - REF_UNITS,
            transitions: 0,
            pending: Vec::new(),
        }
    }

    /// Apply every buffered example's outer products to the carry in one pass, then set trits.
    fn flush(&mut self, net: &mut Net) {
        if self.pending.is_empty() {
            return;
        }
        for l in 0..net.layers() {
            let (n_in, n_out) = (net.sizes[l], net.sizes[l + 1]);
            for j in 0..n_out {
                let row = &mut self.latent[l][j * n_in..(j + 1) * n_in];
                let wrow = &mut net.w[l][j * n_in..(j + 1) * n_in];
                for i in 0..n_in {
                    // Gradient in 2^-(D_0) units: unit error (2^(D_0+FRAC)) × input, >> FRAC.
                    let g: i128 = self.pending.iter().map(|(e, x)| e[l][j] * i128::from(x[l][i])).sum();
                    let g = (g >> FRAC) as i64;
                    let step = g >> self.shift;
                    if step == 0 {
                        continue;
                    }
                    let latent = (i64::from(row[i]) - step).clamp(-i64::from(i32::MAX), i64::from(i32::MAX));
                    row[i] = latent as i32;
                    let t = ((latent + latent.signum() * (S / 2)) / S).clamp(-1, 1) as i8;
                    if t != wrow[i] {
                        wrow[i] = t;
                        self.transitions += 1;
                    }
                }
            }
        }
        self.pending.clear();
    }
}

fn pixels(x: &Example) -> Vec<i64> {
    x.pixels.iter().map(|&p| i64::from(p)).collect()
}

fn evaluate(net: &Net, data: &[Example]) -> (f64, f64) {
    let (mut loss, mut correct) = (0i128, 0usize);
    for x in data {
        let out = net.forward(&pixels(x)).acts.pop().expect("output");
        loss += i128::from(Net::loss(&out, x.label));
        let best = (0..out.len()).max_by_key(|&k| (out[k], std::cmp::Reverse(k))).expect("outputs");
        correct += usize::from(best == x.label);
    }
    (correct as f64 / data.len() as f64, loss as f64 / data.len() as f64)
}

fn arg<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).map(String::as_str)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let data = arg(&args, "--data").ok_or("--data MNIST_IDX_DIR required")?;
    let num = |k: &str, d: i64| -> Result<i64, String> {
        arg(&args, k).map_or(Ok(d), |v| v.parse().map_err(|_| format!("invalid {k}")))
    };
    let sizes: Vec<usize> = arg(&args, "--sizes")
        .unwrap_or("784,128,10")
        .split(',')
        .map(|v| v.parse().map_err(|_| "invalid --sizes"))
        .collect::<Result<_, _>>()?;
    if sizes.len() < 2 || sizes[0] != 784 || *sizes.last().unwrap() != 10 {
        return Err("--sizes must start at 784 and end at 10".into());
    }
    let shifts: Vec<u32> = match arg(&args, "--shifts") {
        Some(s) => s.split(',').map(|v| v.parse().map_err(|_| "invalid --shifts")).collect::<Result<_, _>>()?,
        None => (0..sizes.len() - 1)
            .map(|l| if l == 0 { 8 } else { sizes[l].ilog2() / 2 })
            .collect(),
    };
    if shifts.len() != sizes.len() - 1 {
        return Err("--shifts needs one shift per layer".into());
    }
    let seed = num("--seed", 1)? as u64;
    let steps = num("--steps", 20_000)? as u64;
    let lr = num("--lr", 4)? as u32;
    let buffer = num("--buffer", 1)?.max(1) as usize;
    let noise = arg(&args, "--grad").unwrap_or("exact") == "noise";
    let draws = num("--draws", 1)?.max(1) as usize;
    let sigma = num("--sigma", 2)?.max(1);
    let time_n = num("--time", 0)? as u64;
    let eval_offset = num("--eval-offset", 1000)? as usize;
    let eval_limit = num("--eval-limit", 4000)? as usize;
    let p = Path::new(data);
    let all = model::mnist(&p.join("train-images-idx3-ubyte"), &p.join("train-labels-idx1-ubyte"))?;
    let (train, eval) = all.split_at(55_000);
    let eval = &eval[eval_offset..eval_offset + eval_limit];
    let mut net = Net::new(&sizes, &shifts, &mut Rng(seed));
    let mut carry = Carry::new(&net, lr);
    let mut noise_rng = Rng(0x0015_EC0D ^ seed);
    let total = if time_n > 0 { time_n } else { steps };
    // Timing mode: warm up (clocks, caches) on examples outside the timed range, then reset.
    let warmup = if time_n > 0 { buffer as u64 * 2 } else { 0 };
    let (mut t_fwd, mut t_credit, mut t_update) = (0u128, 0u128, 0u128);
    let mut start = Instant::now();
    for step in 0..warmup + total {
        if step == warmup && warmup > 0 {
            carry.flush(&mut net);
            (t_fwd, t_credit, t_update) = (0, 0, 0);
            start = Instant::now();
        }
        let mut sampler = Rng(seed.wrapping_add(step.wrapping_mul(0x9e37_79b9_7f4a_7c15)).max(1));
        let x = &train[sampler.below(train.len() as u64) as usize];
        let t0 = Instant::now();
        let pass = net.forward(&pixels(x));
        let t1 = Instant::now();
        let units = if noise {
            noise_units(&net, &pass, x.label, draws, sigma, &mut noise_rng)
        } else {
            exact_units(&net, &pass, x.label)
        };
        let t2 = Instant::now();
        let inputs: Vec<Vec<i64>> = pass.acts[..net.layers()].to_vec();
        carry.pending.push((units, inputs));
        if carry.pending.len() == buffer {
            carry.flush(&mut net);
        }
        let t3 = Instant::now();
        t_fwd += (t1 - t0).as_nanos();
        t_credit += (t2 - t1).as_nanos();
        t_update += (t3 - t2).as_nanos();
    }
    // The final partial buffer is part of the work: time it as update.
    let tf = Instant::now();
    carry.flush(&mut net);
    t_update += tf.elapsed().as_nanos();
    let elapsed = start.elapsed().as_millis();
    let per = |t: u128| t as f64 / total as f64 / 1000.0;
    let (accuracy, mean_loss) = if time_n > 0 { (f64::NAN, f64::NAN) } else { evaluate(&net, eval) };
    let params: usize = net.w.iter().map(Vec::len).sum();
    let report = serde_json::json!({
        "sizes": sizes, "shifts": shifts, "params": params,
        "gradient": if noise { format!("noise K={draws} sigma={sigma}") } else { "exact".into() },
        "seed": seed, "steps": total, "lr": lr, "buffer": buffer,
        "accuracy": accuracy, "mean_loss": mean_loss,
        "transitions": carry.transitions,
        "noisy_layer_evaluations": if noise { total as usize * 2 * draws * net.layers() } else { 0 },
        "us_per_example": {"forward": per(t_fwd), "credit": per(t_credit), "update": per(t_update),
                            "total": per(t_fwd + t_credit + t_update)},
        "elapsed_ms": elapsed, "qualification": "UNKNOWN",
    });
    println!("{}", serde_json::to_string(&report).map_err(|e| e.to_string())?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use model::Model;

    fn xs(n: usize, width: usize) -> Vec<Example> {
        (0..n)
            .map(|k| Example {
                pixels: (0..width).map(|i| ((k * 37 + i * 101) % 256) as u8).collect(),
                label: k % 10,
            })
            .collect()
    }

    /// 784,128,10 with shifts 8,3 is the lab model: same init, forward and gradient.
    #[test]
    fn lab_network_is_reproduced() {
        let mut m = Model::new(784, 128, 10, &mut Rng(5));
        let net = Net::new(&[784, 128, 10], &[8, 3], &mut Rng(5));
        let flat: Vec<i8> = net.w.concat();
        assert_eq!(flat, m.trits);
        m.version += 1;
        for x in &xs(30, 784) {
            let a = m.forward(x);
            let p = net.forward(&pixels(x));
            assert_eq!(p.acts[1], a.hidden);
            assert_eq!(p.acts[2], a.output);
            // Exact gradient numerator in 2^-11 units = lab sensitivity before its truncation.
            let u = exact_units(&net, &p, x.label);
            let split = 784 * 128;
            for idx in (0..m.trits.len()).step_by(997) {
                let g = if idx < split {
                    u[0][idx / 784] * i128::from(p.acts[0][idx % 784])
                } else {
                    let (k, j) = ((idx - split) / 128, (idx - split) % 128);
                    u[1][k] * i128::from(p.acts[1][j])
                } >> FRAC;
                assert_eq!(g / (1 << 11), m.sensitivity(x, &a, idx), "index {idx}");
            }
        }
    }

    fn cosine(a: &[i128], b: &[i128]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
        let n = |v: &[i128]| v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        dot / (n(a) * n(b))
    }

    /// On a deeper net the batched noise estimate approaches the exact unit errors in every layer.
    #[test]
    fn noise_units_approach_exact_on_deep_net() {
        let net = Net::new(&[784, 32, 16, 10], &[8, 2, 2], &mut Rng(3));
        let mut rng = Rng(11);
        let mut ex = vec![vec![0i128; 32], vec![0; 16], vec![0; 10]];
        let mut est = ex.clone();
        for x in &xs(40, 784) {
            let p = net.forward(&pixels(x));
            let (e, n) = (exact_units(&net, &p, x.label), noise_units(&net, &p, x.label, 3000, 2, &mut rng));
            for l in 0..3 {
                // Sum the per-unit errors over examples; the outer product with the (exact) layer
                // input is common to both, so agreement here is agreement of the weight gradients.
                for j in 0..ex[l].len() {
                    ex[l][j] += e[l][j];
                    est[l][j] += n[l][j];
                }
            }
        }
        for l in 0..3 {
            let c = cosine(&est[l], &ex[l]);
            assert!(c > 0.8, "layer {l}: cosine {c}");
        }
    }

    /// buffer = 1 applies each example as it arrives; a batched forward of zero noise equals the
    /// plain forward's loss.
    #[test]
    fn batch_forward_matches_plain() {
        let net = Net::new(&[784, 24, 12, 10], &[8, 2, 1], &mut Rng(4));
        for x in &xs(10, 784) {
            let p = net.forward(&pixels(x));
            let plain = Net::loss(p.acts.last().unwrap(), x.label);
            for l in 0..3 {
                let got = net.batch_losses(l, &[p.raws[l].clone()], x.label);
                assert_eq!(got, vec![plain]);
            }
        }
    }

    /// Per-example carry training on the exact gradient moves trits and lowers the training loss.
    #[test]
    fn exact_training_moves_trits_and_lowers_loss() {
        let mut net = Net::new(&[784, 32, 10], &[8, 2], &mut Rng(6));
        let data = xs(200, 784);
        let before: i64 = data.iter().map(|x| Net::loss(net.forward(&pixels(x)).acts.last().unwrap(), x.label)).sum();
        let mut carry = Carry::new(&net, 4);
        for _ in 0..3 {
            for x in &data {
                let p = net.forward(&pixels(x));
                carry.pending.push((exact_units(&net, &p, x.label), p.acts[..2].to_vec()));
                carry.flush(&mut net);
            }
        }
        let after: i64 = data.iter().map(|x| Net::loss(net.forward(&pixels(x)).acts.last().unwrap(), x.label)).sum();
        assert!(carry.transitions > 0);
        assert!(after < before, "{after} >= {before}");
    }
}
