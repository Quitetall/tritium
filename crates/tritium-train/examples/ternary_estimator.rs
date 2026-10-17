//! Offline estimator diagnostic (ladder rung G1): how well does a forward-only, noise-credited
//! gradient estimate agree with the exact STE gradient on a FROZEN ternary checkpoint?
//!
//! No training happens here, so estimator quality is measured apart from any optimizer.
//! For one layer at a time (Dust perturbs layer types in separate passes), integer noise `a` is
//! added to that layer's raw pre-activation sums, the true quantized forward is re-run from that
//! point, and each draw is credited by its exact integer loss change:
//!
//! - `gauss`: Gaussian noise rounded to raw units, credit ℓ(a) − ℓ(0) (one extra forward);
//! - `anti`: the same noise as antithetic pairs, credit (ℓ(+a) − ℓ(−a)) / 2 (two forwards);
//! - `ternary`: a ∈ {−σ, 0, +σ} uniform, antithetic;
//! - `sign`: ternary antithetic with the credit reduced to its sign (ElasticZO / EGGROLL).
//!
//! The unit error estimate Σ r·a becomes a weight-gradient estimate by the outer product with
//! the layer input, exactly as Dust does, and is compared with the exact STE gradient (the
//! numerator of `Model::sensitivity`, as in `ternary_carry`) summed over the same examples:
//! cosine per layer, and the |exact|-weighted fraction of coordinates whose sign agrees.
//! σ is given in units of the layer's quantization step (2^shift raw units): noise below one
//! step is mostly truncated away by the integer forward — the coarse-noise landmine.
#[allow(dead_code)]
#[path = "ternary_lab/model.rs"]
mod model;
#[allow(dead_code)]
#[path = "ternary_lab/numeric.rs"]
mod numeric;

use model::{Example, Model};
use numeric::Rng;
use rayon::prelude::*;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Noise {
    Gauss,
    Anti,
    Ternary,
    Sign,
}

const TARGET: i64 = 256;

/// Clean forward pieces for one example: raw hidden sums, hidden activations, raw output sums.
struct Clean {
    hidden_raw: Vec<i64>,
    hidden: Vec<i64>,
    output_raw: Vec<i64>,
}

fn clean(m: &Model, x: &Example) -> Clean {
    let hidden_raw: Vec<i64> = m.trits[..m.inputs * m.hidden]
        .chunks_exact(m.inputs)
        .map(|row| row.iter().zip(&x.pixels).map(|(&w, &v)| i64::from(w) * i64::from(v)).sum())
        .collect();
    let hidden: Vec<i64> = hidden_raw.iter().map(|v| (v / (1 << m.input_shift)).max(0)).collect();
    let output_raw = output_raw(m, &hidden);
    Clean { hidden_raw, hidden, output_raw }
}

fn output_raw(m: &Model, hidden: &[i64]) -> Vec<i64> {
    m.trits[m.inputs * m.hidden..]
        .chunks_exact(m.hidden)
        .map(|row| row.iter().zip(hidden).map(|(&w, &v)| i64::from(w) * v).sum())
        .collect()
}

/// Exact integer loss from raw output sums, as `Model::loss`.
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

/// Loss with noise `a` on layer 0 (hidden raw sums) or layer 1 (output raw sums).
fn noisy_loss(m: &Model, c: &Clean, layer: usize, a: &[i64], sign: i64, label: usize) -> i64 {
    if layer == 0 {
        let h: Vec<i64> = c
            .hidden_raw
            .iter()
            .zip(a)
            .map(|(&r, &n)| ((r + sign * n) / (1 << m.input_shift)).max(0))
            .collect();
        loss(m, &output_raw(m, &h), label)
    } else {
        let o: Vec<i64> = c.output_raw.iter().zip(a).map(|(&r, &n)| r + sign * n).collect();
        loss(m, &o, label)
    }
}

fn gauss(rng: &mut Rng) -> f64 {
    // Box-Muller from two uniforms in (0, 1].
    let u1 = (rng.next() >> 11) as f64 / (1u64 << 53) as f64 + f64::EPSILON;
    let u2 = (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Exact STE gradient over the batch for one layer, in that layer's weight layout.
fn exact(m: &Model, batch: &[(&Example, Clean)], layer: usize) -> Vec<f64> {
    let split = m.inputs * m.hidden;
    let mut g = vec![0f64; if layer == 0 { split } else { m.trits.len() - split }];
    for (x, c) in batch {
        let err: Vec<i64> = (0..m.outputs)
            .map(|k| c.output_raw[k] / (1 << m.output_shift) - if k == x.label { TARGET } else { 0 })
            .collect();
        if layer == 1 {
            for k in 0..m.outputs {
                for j in 0..m.hidden {
                    g[k * m.hidden + j] += (2 * err[k] * c.hidden[j]) as f64;
                }
            }
        } else {
            for j in 0..m.hidden {
                if c.hidden[j] == 0 {
                    continue;
                }
                let d: i64 = (0..m.outputs)
                    .map(|k| 2 * err[k] * i64::from(m.trits[split + k * m.hidden + j]))
                    .sum();
                for (i, &p) in x.pixels.iter().enumerate() {
                    g[j * m.inputs + i] += (d * i64::from(p)) as f64;
                }
            }
        }
    }
    g
}

/// Noise-credited estimate over the batch, K draws per example, one seed per example.
fn estimate(
    m: &Model,
    batch: &[(&Example, Clean)],
    layer: usize,
    noise: Noise,
    sigma_steps: f64,
    k: usize,
    seed: u64,
) -> Vec<f64> {
    let (units, step) = if layer == 0 {
        (m.hidden, 1i64 << m.input_shift)
    } else {
        (m.outputs, 1i64 << m.output_shift)
    };
    let sigma = sigma_steps * step as f64;
    let split = m.inputs * m.hidden;
    let partial: Vec<Vec<f64>> = batch
        .par_iter()
        .enumerate()
        .map(|(b, (x, c))| {
            let mut rng = Rng((seed ^ (b as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)).max(1));
            let mut unit = vec![0f64; units];
            let base = loss(m, &c.output_raw, x.label);
            for _ in 0..k {
                let a: Vec<i64> = (0..units)
                    .map(|_| match noise {
                        Noise::Gauss | Noise::Anti => (gauss(&mut rng) * sigma).round() as i64,
                        Noise::Ternary | Noise::Sign => {
                            (rng.below(3) as i64 - 1) * sigma.round() as i64
                        }
                    })
                    .collect();
                let r = match noise {
                    Noise::Gauss => (noisy_loss(m, c, layer, &a, 1, x.label) - base) as f64,
                    Noise::Anti | Noise::Ternary => {
                        (noisy_loss(m, c, layer, &a, 1, x.label)
                            - noisy_loss(m, c, layer, &a, -1, x.label)) as f64
                            / 2.0
                    }
                    Noise::Sign => (noisy_loss(m, c, layer, &a, 1, x.label)
                        - noisy_loss(m, c, layer, &a, -1, x.label))
                    .signum() as f64,
                };
                for (u, &n) in unit.iter_mut().zip(&a) {
                    *u += r * n as f64;
                }
            }
            // Outer product with the layer input.
            if layer == 0 {
                let mut g = vec![0f64; split];
                for j in 0..m.hidden {
                    for (i, &p) in x.pixels.iter().enumerate() {
                        g[j * m.inputs + i] = unit[j] * f64::from(p);
                    }
                }
                g
            } else {
                let mut g = vec![0f64; m.trits.len() - split];
                for kk in 0..m.outputs {
                    for j in 0..m.hidden {
                        g[kk * m.hidden + j] = unit[kk] * c.hidden[j] as f64;
                    }
                }
                g
            }
        })
        .collect();
    let mut g = vec![0f64; partial[0].len()];
    for p in partial {
        for (gi, pi) in g.iter_mut().zip(p) {
            *gi += pi;
        }
    }
    g
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// Fraction of |exact| mass whose estimated sign matches (a zero estimate counts as a miss).
fn sign_agreement(est: &[f64], ex: &[f64]) -> f64 {
    let total: f64 = ex.iter().map(|v| v.abs()).sum();
    let hit: f64 = est
        .iter()
        .zip(ex)
        .filter(|(e, x)| **x != 0.0 && e.signum() == x.signum() && **e != 0.0)
        .map(|(_, x)| x.abs())
        .sum();
    if total == 0.0 { 0.0 } else { hit / total }
}

fn arg<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).map(String::as_str)
}

fn list<T: std::str::FromStr>(args: &[String], key: &str, default: &str) -> Result<Vec<T>, String> {
    arg(args, key)
        .unwrap_or(default)
        .split(',')
        .map(|v| v.parse().map_err(|_| format!("invalid {key}")))
        .collect()
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let data = arg(&args, "--data").ok_or("--data MNIST_IDX_DIR required")?;
    let ckpt = arg(&args, "--trits").ok_or("--trits CHECKPOINT.json required")?;
    let p = Path::new(data);
    let all = model::mnist(&p.join("train-images-idx3-ubyte"), &p.join("train-labels-idx1-ubyte"))?;
    let train = &all[..55_000];
    let mut m = Model::new(784, 128, 10, &mut Rng(1));
    m.trits = serde_json::from_str(&std::fs::read_to_string(ckpt).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    m.version += 1;
    m.validate()?;
    let batches: Vec<usize> = list(&args, "--batch", "16,256")?;
    let draws: Vec<usize> = list(&args, "--draws", "1,4,16,64")?;
    let sigmas: Vec<f64> = list(&args, "--sigma", "0.5,1,2,4")?;
    let seed: u64 = arg(&args, "--seed").unwrap_or("7").parse().map_err(|_| "invalid --seed")?;
    let max_b = *batches.iter().max().ok_or("empty --batch")?;
    let mut pick = Rng(seed);
    let batch_all: Vec<(&Example, Clean)> = (0..max_b)
        .map(|_| {
            let x = &train[pick.below(train.len() as u64) as usize];
            (x, clean(&m, x))
        })
        .collect();
    for &bsz in &batches {
        let batch = &batch_all[..bsz];
        for layer in 0..2 {
            let ex = exact(&m, batch, layer);
            for noise in [Noise::Gauss, Noise::Anti, Noise::Ternary, Noise::Sign] {
                for &sigma in &sigmas {
                    for &k in &draws {
                        let est = estimate(&m, batch, layer, noise, sigma, k, seed);
                        let forwards = bsz * k * if noise == Noise::Gauss { 1 } else { 2 };
                        println!(
                            "{}",
                            serde_json::json!({
                                "checkpoint": ckpt, "layer": layer, "noise": format!("{noise:?}"),
                                "sigma_steps": sigma, "batch": bsz, "draws": k,
                                "extra_forwards": forwards,
                                "cosine": cosine(&est, &ex),
                                "sign_agreement": sign_agreement(&est, &ex),
                            })
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (Model, Vec<Example>) {
        let mut m = Model::new(6, 4, 3, &mut Rng(3));
        m.trits = vec![
            1, 1, 0, 1, -1, 1, 0, 1, 1, 1, 1, 0, 1, -1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, // input
            1, -1, 0, 1, -1, 1, 1, 0, 0, 1, -1, 1, // output
        ];
        m.version += 1;
        let xs = (0..40u8)
            .map(|k| Example {
                pixels: (0..6).map(|i| k.wrapping_mul(29).wrapping_add(i * 41)).collect(),
                label: usize::from(k % 3),
            })
            .collect();
        (m, xs)
    }

    /// The clean pieces reproduce `Model::forward` and `Model::loss` exactly.
    #[test]
    fn clean_forward_matches_model() {
        let (m, xs) = setup();
        for x in &xs {
            let c = clean(&m, x);
            let a = m.forward(x);
            assert_eq!(c.hidden, a.hidden);
            assert_eq!(loss(&m, &c.output_raw, x.label) as i128, m.loss(x));
        }
    }

    /// Zero noise changes nothing; the estimate of the output layer converges toward the exact
    /// gradient as draws grow (a loose check that the credit sign and outer product are right).
    #[test]
    fn output_layer_estimate_aligns_with_exact() {
        let (m, xs) = setup();
        let batch: Vec<(&Example, Clean)> = xs.iter().map(|x| (x, clean(&m, x))).collect();
        let c = &batch[0].1;
        assert_eq!(noisy_loss(&m, c, 1, &[0, 0, 0], 1, 0), loss(&m, &c.output_raw, 0));
        let ex = exact(&m, &batch, 1);
        let est = estimate(&m, &batch, 1, Noise::Anti, 2.0, 400, 11);
        assert!(cosine(&est, &ex) > 0.9, "{}", cosine(&est, &ex));
    }
}
