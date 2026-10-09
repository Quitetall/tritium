//! **Is the ternary plane the best additive structure? Coded trit-planes (H8, first-principles).**
//!
//! A SALT ladder with `T` planes is a uniform scalar quantizer with `3^T` levels per group. The
//! structure audit (`docs/plans/0056`) measured, on SmolLM2's real Hadamard-rotated weights, that
//! the level placement is already right — free per-plane scales and Lloyd codebooks gain ≤ 0.2 dB —
//! and that the rotated weights are Gaussian to within 0.03 dB. What a scalar structure cannot
//! have is the granular (cell-shape) gain of a vector quantizer, up to 1.53 dB.
//!
//! Ternary structure can reach it without giving up multiply-free arithmetic. Construction A over
//! `F₃`: constrain the FINEST trit plane of every 12-weight block to be a codeword of the extended
//! ternary Golay code `[12, 6, 6]`. The reconstruction is `Δ·(3k + c)` with `k` the usual `T` coarse
//! planes and `c ∈ C`, i.e. an ordinary `(T+1)`-plane ternary tensor whose last plane happens to be a
//! codeword — so the kernel is unchanged, one more plane of add/subtract. Storage is `T + ½` trits a
//! weight (6 message trits per 12-block; the parity trits are mod-3 sums of them). The lattice
//! `C + 3ℤ¹²` has normalized second moment 0.0705 (0.73 dB over ℤ, better than E₈'s 0.65).
//!
//! Decoding is exact: per block and per codeword the coarse digits round independently, so the
//! nearest point is a minimum over 729 candidates. Blocks run along the row and may straddle g128
//! scale groups; each coordinate uses its own group's step (a diagonal-metric decode), and the steps
//! are refit by least squares after the first decode.
//!
//! Arms, all in the g128 group-Hadamard basis on the folded model, fp activations: the ladder at
//! `T`, Golay `T+½`, and the only uncoded way to reach `T+½` trits — time-sharing, alternate rows at
//! `T` and `T+1`. Scored with the antithetic split (`even` = curvature cost, sample term cancelled).
//!
//! # Measured 2026-09-24 — SmolLM2-135M, full WikiText-2 eval (32,768 tokens), fp 22.675
//!
//! ```text
//! structure               bits/w       ppl       even (curvature)   SNR dB
//! ladder T=1               1.710  22506.406      +8.09769            8.40
//! uniform 5-level scalar   2.447    219.137      +2.28345           12.18
//! Golay-coded 1+½          2.502    129.231      +1.79460           13.10
//! time-share 1/2           2.502   2321.279      +5.64457           10.81
//! ladder T=2               3.295     46.405      +0.71623           16.52
//! uniform 15-level scalar  4.032     28.396      +0.23924           20.76
//! Golay-coded 2+½          4.087     27.362      +0.19744           21.38
//! time-share 2/3           4.087     32.683      +0.41632           19.01
//! ladder T=3               4.880     24.278      +0.07746           25.11
//! ```
//!
//! **Coding the finest plane beats every uncoded structure at the same rate.** Against the ternary
//! fallback (time-sharing) it is 18x better in perplexity at 2.50 bpw and -16% at 4.09 bpw (curvature
//! cost -68% / -53%). Against the fairest uncoded rival — the best single scalar at that rate, which
//! is not trit-plane storable — it is -41% / -3.6% perplexity and -21% / -17% curvature cost while
//! spending 0.055 bpw more; net of that premium the gain is ~0.3-0.6 dB, which is what a lattice with
//! 0.73 dB of granular gain should deliver. The finest plane stays a ternary plane, so the kernel is
//! unchanged (one more add/subtract plane); only storage changes (6 message trits per 12 weights).
//! Plain nearest-point decoding only — no GPTQ/BlockLDLQ yet. The odd (sample) term is small at this
//! eval size (|odd| ≤ 0.05 nats except the destroyed T=1 arms).
//!
//! # Measured 2026-10-07 — Golay + block-LDLQ (`TRITIUM_H8_ARMS=ldlq`), 16k-token Grams, auto decay
//!
//! ```text
//! structure                 bits/w      ppl    even (curvature)   odd      SNR dB
//! Golay 1+½ (plain)          2.502   129.231      +1.79460      -0.05427    13.10
//! Golay 1+½ + BlockLDLQ      2.502    82.633      +1.56199      -0.26885    12.24
//! Golay 2+½ (plain)          4.087    27.362      +0.19744      -0.00956    21.38
//! Golay 2+½ + BlockLDLQ      4.087    26.520      +0.16777      -0.01115    20.50
//! ladder T=2 + GPTQ (H6)     3.295    40.263      +0.61521
//! ladder T=3 + GPTQ (H6)     4.880    24.177      +0.06620
//! ```
//!
//! Block feedback composes with the code: −36% ppl at 1+½ and −15% curvature at 2+½ over the plain
//! Golay fit (SNR falls, as it must — the fit now minimizes the Gram metric, not MSE). At 4.09 bpw the
//! curvature cost (0.168) is half of what ladder+GPTQ reaches by interpolating its T=2 and T=3
//! points at the same bytes (~0.34). The 1+½ arm carries a large calibrated first-order term
//! (odd −0.27), the same domain-alignment effect H6 measured for calibrated fits.
//!
//! # Measured 2026-10-07 — Kronecker Golay (`TRITIUM_H8_ARMS=kron`), A ⊗ G from 16k tokens, λ_G = 0.5
//!
//! ```text
//! structure                  bits/w      ppl    even (curvature)   odd      SNR dB
//! Golay 1+½ + BlockLDLQ       2.502    82.633      +1.56199      -0.26885    12.24
//! Golay 1+½ + Kron LDLQ       2.502    76.719      +1.43840      -0.21953    11.84
//! Golay 2+½ + BlockLDLQ       4.087    26.520      +0.16777      -0.01115    20.50
//! Golay 2+½ + Kron LDLQ       4.087    26.570      +0.15976      -0.00125    20.11
//! ```
//!
//! The output side composes with the code exactly as it did with the ladder (H6): −7.2% ppl and
//! −7.9% curvature at 1+½; at 2+½ the ppl change is inside the noise floor and the curvature falls
//! 4.8%. The best 4.09-bpw fit is now Golay 2+½ + Kronecker LDLQ (curvature 0.160 against ~0.34 for
//! ladder+GPTQ interpolated at the same bytes). Cost: about 2.5× the BlockLDLQ fit time (row loop).
//!
//! # Trellis-coded trit-planes (2026-10-09) — same model, plain nearest-path (Viterbi) fits
//!
//! `T` coarse planes plus one bit per weight choosing 2 of the 3 finest-trit cosets along a
//! shift-register trellis (rate `T·log₂3 + 1`): every reconstruction is still a sum of `T + 1`
//! trit planes. Tables searched on seeded Gaussian rows.
//!
//! ```text
//! structure                 bits/w      ppl    even (curvature)   odd      SNR dB
//! trellis 1+1bit S=64        2.710    68.944      +1.23159      +0.02081    14.12
//! trellis 1+1bit S=256       2.710    71.071      +1.23538      +0.04741    14.09
//! trellis 2+1bit S=64        4.295    22.939      +0.15112      +0.00083    22.47
//! trellis 2+1bit S=256       4.295    22.850      +0.14749      +0.00057    22.62
//! ```
//!
//! **The trellis ties the Golay code; it does not beat it.** Different rates, so both are read
//! against the ladder curve (SNR linear in bits, curvature log-linear, between T and T+1):
//! SNR gain +0.60 dB (S=64) / +0.57 (S=256) at 1+1bit and +0.53 / +0.68 at 2+1bit, against Golay
//! +0.64 / +0.57; curvature 0.70× / 0.84× the interpolated ladder against Golay 0.74× / 0.84×.
//! The prototype's +0.1 dB edge over Golay (on sampled groups) does not survive the whole model,
//! and 256 states buy nothing over 64 at T=1 (ppl delta is inside the odd-term noise). Golay keeps
//! the advantage that matters for hardware: independent 12-blocks (random access, one table
//! lookup) versus a sequential Viterbi path per row. Golay stays the coded structure.
//!
//! ```text
//! TRITIUM_CORPUS=$HOME/.cache/tritium-corpora/wikitext2_400k_32k.json \
//!   cargo test -p tritium-nn --release --test coded_planes -- --ignored --nocapture
//! ```
//! `TRITIUM_H8_ARMS=trellis` runs the trellis-coded arms.
//! Knobs: `TRITIUM_H8_TS` (coarse planes for the coded arms, default `1,2`),
//! `TRITIUM_H8_EVAL_WINDOWS` (default 64), `TRITIUM_H8_ARMS=uniform` (single-scalar comparators).

mod common;

use std::path::PathBuf;
use std::time::Instant;

use common::golay::{
    BLOCK, GRID, GROUP, golay_codewords, golay_decode_block, golay_fit, golay_row, hadamard_blocks,
    kmax,
};
use common::{Calib, calibrate, extract, fold, output_fishers, perplexity_windowed};
use rayon::prelude::*;
use tritium_nn::ModelRunner;
use tritium_nn::salt_fit::{TapGrams, auto_decay};
use tritium_train::ops::ste::{self, RotationPolicy};

const WINDOW: usize = 512;
const CALIB_WINDOWS: usize = 4;

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(
        std::env::var("TRITIUM_MODEL_DIR")
            .unwrap_or_else(|_| format!("{home}/.cache/tritium-models/smollm2-135m")),
    )
}

fn corpus() -> (Vec<u32>, Vec<u32>) {
    let path = std::env::var("TRITIUM_CORPUS").expect("set TRITIUM_CORPUS (WikiText-2 basis)");
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("corpus")).expect("json");
    let ids = |k: &str| -> Vec<u32> {
        v[k].as_array()
            .expect(k)
            .iter()
            .map(|x| x.as_u64().expect("id") as u32)
            .collect()
    };
    (ids("train_ids"), ids("eval_ids"))
}

/// Normalized Hadamard on each 128-block of an f64 row (ragged power-of-two tail included), the
/// f64 twin of [`hadamard_blocks`] for rotating a Gram into the codes' basis.
fn hadamard_blocks_f64(row: &mut [f64]) {
    for b in row.chunks_mut(GROUP) {
        let n = b.len();
        if !(n.is_power_of_two() && n > 1) {
            continue;
        }
        let mut len = 1;
        while len < n {
            for start in (0..n).step_by(len * 2) {
                for i in start..start + len {
                    let (a, c) = (b[i], b[i + len]);
                    b[i] = a + c;
                    b[i + len] = a - c;
                }
            }
            len *= 2;
        }
        let scale = 1.0 / (n as f64).sqrt();
        for v in b.iter_mut() {
            *v *= scale;
        }
    }
}

/// `R·H·Rᵀ` for the block-Hadamard `R`: the Gram in the basis the codes are fitted in.
fn rotate_gram(h: &[f64], n: usize) -> Vec<f64> {
    let mut out = h.to_vec();
    for row in out.chunks_mut(n) {
        hadamard_blocks_f64(row);
    }
    let mut col = vec![0.0f64; n];
    for c in 0..n {
        for r in 0..n {
            col[r] = out[r * n + c];
        }
        hadamard_blocks_f64(&mut col);
        for r in 0..n {
            out[r * n + c] = col[r];
        }
    }
    out
}

/// Block-LDLQ feedback matrix for a row-vector objective `e·H·eᵀ` processed in 12-blocks.
///
/// With `H + λI = U·Uᵀ`, `U` upper triangular (the Cholesky of the index-reversed matrix, reversed
/// back), `e·H·eᵀ = ‖e·U‖²` and block `b`'s component depends only on blocks `≤ b`. Nearest-plane
/// in block order then rounds block `b` at the target `w_b + Σ_{a<b} e_a·Ũ_ab` with
/// `Ũ_ab = U_ab·U_bb⁻¹` — LDLQ when the block is one column, QuIP#'s BlockLDLQ here. Returns `Ũ`
/// (`n × n`, only the strictly block-upper part is meaningful), or `None` if `H + λI` is not PD.
fn block_ldlq_factor(h: &[f64], n: usize, damp: f64, block: usize) -> Option<Vec<f64>> {
    let mean = (0..n).map(|i| h[i * n + i]).sum::<f64>() / n as f64;
    let lambda = damp * mean.max(1e-12);
    // Lower Cholesky of the reversed matrix P·H·P.
    let rev = |i: usize| n - 1 - i;
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = h[rev(i) * n + rev(j)] + if i == j { lambda } else { 0.0 };
            for p in 0..j {
                sum -= l[i * n + p] * l[j * n + p];
            }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    return None;
                }
                l[i * n + i] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    // U = P·L·P is upper triangular with H + λI = U·Uᵀ.
    let u = |i: usize, j: usize| l[rev(i) * n + rev(j)];
    let mut ut = vec![0.0f64; n * n];
    for b0 in (0..n).step_by(block) {
        let bs = block.min(n - b0);
        // Inverse of the upper-triangular diagonal block U_bb.
        let mut inv = vec![0.0f64; bs * bs];
        for c in 0..bs {
            for r in (0..=c).rev() {
                let rhs = if r == c { 1.0 } else { 0.0 };
                let s: f64 = (r + 1..=c)
                    .map(|m| u(b0 + r, b0 + m) * inv[m * bs + c])
                    .sum();
                inv[r * bs + c] = (rhs - s) / u(b0 + r, b0 + r);
            }
        }
        // Ũ[:b0, block] = U[:b0, block] · U_bb⁻¹.
        for a in 0..b0 {
            for c in 0..bs {
                ut[a * n + b0 + c] = (0..=c).map(|m| u(a, b0 + m) * inv[m * bs + c]).sum();
            }
        }
    }
    Some(ut)
}

/// Golay-coded `T + ½` with block-LDLQ error feedback over 12-blocks, at fixed steps. `y` is the
/// rotated row, `steps` its per-group steps, `ut` from [`block_ldlq_factor`], `decay` the fraction
/// of each block's error pushed forward (`1.0` = pure LDLQ, the shipped `auto_decay` otherwise).
fn golay_ldlq_row(
    y: &[f32],
    steps: &[f64],
    t: usize,
    ut: &[f64],
    decay: f64,
    code: &[[i8; BLOCK]],
) -> Vec<f32> {
    let n = y.len();
    let k = kmax(t) as f64;
    let mut acc = vec![0.0f64; n];
    let mut out = vec![0.0f32; n];
    for b0 in (0..n).step_by(BLOCK) {
        let target: Vec<f64> = (0..BLOCK)
            .map(|i| f64::from(y[b0 + i]) + acc[b0 + i])
            .collect();
        let (units, _) = golay_decode_block(&target, b0, steps, k, code);
        let mut err = [0.0f64; BLOCK];
        for i in 0..BLOCK {
            let q = steps[(b0 + i) / GROUP] * units[i] as f64;
            out[b0 + i] = q as f32;
            err[i] = f64::from(y[b0 + i]) - q;
        }
        for c in b0 + BLOCK..n {
            let push: f64 = (0..BLOCK).map(|i| err[i] * ut[(b0 + i) * n + c]).sum();
            acc[c] += decay * push;
        }
    }
    out
}

/// Golay-coded `T + ½` fitted against a Kronecker curvature `A ⊗ G` (input Gram ⊗ output Fisher)
/// by nearest-plane in (12-column block, row) order. With `U = U_A ⊗ U_G` upper triangular, row `i`
/// of block `b` is decoded at
/// `y_{i,b} + Σ_{j<i} e_{j,b}·Ũ_G[j,i] + Σ_{a<b} r_{i,a}·Ũ_A[a,b]`, `r_{i,a} = Σ_{j≤i} e_{j,a}·Ũ_G[j,i]`,
/// with `e = y − q` against the original targets (the LDLQ form). `ut_a` is the 12-block factor of
/// the rotated input Gram, `ut_g` the 1-block factor of the output Fisher; `decay_g` scales the
/// row coupling (`0` switches it off), `decay_a` the column push. With `G = I` this is
/// [`golay_ldlq_row`] on every row.
#[allow(clippy::too_many_arguments)]
fn golay_kron_tensor(
    y: &[f32],
    rows: usize,
    cols: usize,
    steps: &[Vec<f64>],
    t: usize,
    ut_a: &[f64],
    ut_g: &[f64],
    decay_a: f64,
    decay_g: f64,
    code: &[[i8; BLOCK]],
) -> Vec<f32> {
    let k = kmax(t) as f64;
    let mut acc_a = vec![0.0f64; rows * cols];
    let mut out = vec![0.0f32; rows * cols];
    for b0 in (0..cols).step_by(BLOCK) {
        // Row-sequential decode with output-side feedback inside the block.
        let mut acc_g = vec![[0.0f64; BLOCK]; rows];
        let mut err = vec![[0.0f64; BLOCK]; rows];
        for i in 0..rows {
            let target: Vec<f64> = (0..BLOCK)
                .map(|m| f64::from(y[i * cols + b0 + m]) + acc_a[i * cols + b0 + m] + acc_g[i][m])
                .collect();
            let (units, _) = golay_decode_block(&target, b0, &steps[i], k, code);
            for m in 0..BLOCK {
                let q = steps[i][(b0 + m) / GROUP] * units[m] as f64;
                out[i * cols + b0 + m] = q as f32;
                err[i][m] = f64::from(y[i * cols + b0 + m]) - q;
            }
            if decay_g != 0.0 {
                let e = err[i];
                acc_g[i + 1..]
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(off, a)| {
                        let c = decay_g * ut_g[i * rows + i + 1 + off];
                        if c != 0.0 {
                            for m in 0..BLOCK {
                                a[m] += e[m] * c;
                            }
                        }
                    });
            }
        }
        // r_{i,b} = e_{i,b} + decay_g·Σ_{j<i} e_{j,b}·Ũ_G[j,i] — exactly e + acc_g.
        acc_a
            .par_chunks_mut(cols)
            .zip(err.par_iter().zip(acc_g.par_iter()))
            .for_each(|(acc_row, (e, ag))| {
                let r: Vec<f64> = (0..BLOCK).map(|m| e[m] + ag[m]).collect();
                for c in b0 + BLOCK..cols {
                    let push: f64 = (0..BLOCK).map(|m| r[m] * ut_a[(b0 + m) * cols + c]).sum();
                    acc_row[c] += decay_a * push;
                }
            });
    }
    out
}

/// A shift-register trellis over the three cosets of the finest trit: from state `s`, branch bit
/// `b` moves to `(2s + b) mod S` and allows coset `table[s][b]` (the two branches allow different
/// cosets). One bit per weight selects the branch, so the finest plane costs 1 bit, not 1 trit.
#[derive(Clone, Debug)]
struct Trellis {
    states: usize,
    table: Vec<[i8; 2]>,
}

impl Trellis {
    /// A random table from a seeded stream: two distinct cosets per state.
    fn random(states: usize, seed: u64) -> Self {
        let mut s = seed | 1;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let table = (0..states)
            .map(|_| {
                let a = (next() % 3) as i8 - 1;
                let b = ((i64::from(a + 1) + 1 + (next() % 2) as i64) % 3) as i8 - 1;
                [a, b]
            })
            .collect();
        Self { states, table }
    }

    /// The best of `tries` seeded tables by total MSE on synthetic Gaussian rows — a deterministic
    /// design step, independent of any model.
    fn search(states: usize, t: usize, tries: u64) -> Self {
        let mut s = 0x007E_1115_u64;
        let mut gauss = move || {
            let mut u = || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
            };
            let (a, b) = (u(), u());
            ((-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()) as f32
        };
        let rows: Vec<Vec<f32>> = (0..48)
            .map(|_| (0..GROUP * 2).map(|_| gauss()).collect())
            .collect();
        (0..tries)
            .map(|k| Trellis::random(states, 0xC0DE + k))
            .map(|tr| {
                let err: f64 = rows.par_iter().map(|r| tcq_fit(r, t, &tr).2).sum();
                (err, tr)
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .expect("tries > 0")
            .1
    }
}

/// Viterbi over one row: the minimum-error path through the trellis at fixed per-group steps.
/// Returns the codes (in step units, values on the T+1-plane grid) and the squared error.
fn tcq_decode(y: &[f32], steps: &[f64], t: usize, tr: &Trellis) -> (Vec<i64>, f64) {
    let n = y.len();
    let k = kmax(t) as f64;
    let ns = tr.states;
    let mut cost = vec![f64::INFINITY; ns];
    cost[0] = 0.0;
    // back[j * ns + s'] = (previous state, branch) that reached s' at step j.
    let mut back = vec![(0u16, 0u8); n * ns];
    for j in 0..n {
        let d = steps[j / GROUP];
        let x = f64::from(y[j]);
        // Best coarse digit and error for each coset c ∈ {−1, 0, 1}.
        let per_coset: [(f64, i64); 3] = std::array::from_fn(|ci| {
            let c = ci as f64 - 1.0;
            if d <= 0.0 {
                return (x * x, 0);
            }
            let kk = ((x / d - c) / 3.0).round().clamp(-k, k);
            let e = x - d * (3.0 * kk + c);
            (e * e, 3 * kk as i64 + ci as i64 - 1)
        });
        let mut next = vec![f64::INFINITY; ns];
        for (st, &c0) in cost.iter().enumerate() {
            if !c0.is_finite() {
                continue;
            }
            for b in 0..2u8 {
                let to = (2 * st + usize::from(b)) % ns;
                let coset = tr.table[st][usize::from(b)];
                let total = c0 + per_coset[(coset + 1) as usize].0;
                if total < next[to] {
                    next[to] = total;
                    back[j * ns + to] = (st as u16, b);
                }
            }
        }
        cost = next;
    }
    let (mut st, err) = cost
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map(|(s, &e)| (s, e))
        .expect("states > 0");
    let mut cosets = vec![0i8; n];
    for j in (0..n).rev() {
        let (prev, b) = back[j * ns + st];
        cosets[j] = tr.table[usize::from(prev)][usize::from(b)];
        st = usize::from(prev);
    }
    let units = (0..n)
        .map(|j| {
            let d = steps[j / GROUP];
            if d <= 0.0 {
                return 0;
            }
            let c = f64::from(cosets[j]);
            let kk = ((f64::from(y[j]) / d - c) / 3.0).round().clamp(-k, k);
            3 * kk as i64 + i64::from(cosets[j])
        })
        .collect();
    (units, err)
}

/// Step fit for the trellis, as [`golay_fit`] does for the code: a per-row multiplier on the
/// `T+1`-plane ladder step, then two least-squares step refits. Returns (steps, codes, error).
fn tcq_fit(y: &[f32], t: usize, tr: &Trellis) -> (Vec<f64>, Vec<i64>, f64) {
    let cols = y.len();
    let base: Vec<f64> = y
        .chunks(GROUP)
        .map(|g| f64::from(ste::ladder_step(g, t + 1, GRID)))
        .collect();
    let (mut steps, mut units, mut err) = [1.0, 1.15, 1.3, 1.45, 1.6]
        .iter()
        .map(|&f| {
            let steps: Vec<f64> = base.iter().map(|d| d * f).collect();
            let (u, e) = tcq_decode(y, &steps, t, tr);
            (steps, u, e)
        })
        .min_by(|a, b| a.2.total_cmp(&b.2))
        .expect("non-empty grid");
    for _ in 0..2 {
        let mut refit = steps.clone();
        for (g, step) in refit.iter_mut().enumerate() {
            let (lo, hi) = (g * GROUP, ((g + 1) * GROUP).min(cols));
            let num: f64 = (lo..hi).map(|j| f64::from(y[j]) * units[j] as f64).sum();
            let den: f64 = (lo..hi).map(|j| (units[j] as f64).powi(2)).sum();
            if den > 0.0 && num > 0.0 {
                *step = num / den;
            }
        }
        let (u, e) = tcq_decode(y, &refit, t, tr);
        if e < err {
            (steps, units, err) = (refit, u, e);
        }
    }
    (steps, units, err)
}

/// One searched trellis per (states, planes), shared by every tensor of a run.
fn trellis_for(states: usize, t: usize) -> Trellis {
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<(usize, usize), Trellis>>,
    > = std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    if let Some(tr) = cache.lock().expect("trellis cache").get(&(states, t)) {
        return tr.clone();
    }
    let tr = Trellis::search(states, t, 32);
    cache
        .lock()
        .expect("trellis cache")
        .insert((states, t), tr.clone());
    tr
}

#[derive(Clone, Copy)]
enum Arm {
    Ladder(usize),
    Golay(usize),
    TimeShare(usize),
    /// A uniform scalar quantizer with `L` levels (odd), packed at `log2 L` bits: the best uncoded
    /// SINGLE scalar at a fractional ternary rate (5 levels ≈ 1+½ trits, 15 ≈ 2+½).
    Uniform(usize),
    /// Golay-coded `T + ½` fitted with block-LDLQ (GPTQ over 12-blocks) against the input Gram.
    GolayLdlq(usize),
    /// [`Arm::GolayLdlq`] with the output side too: Kronecker `A ⊗ G`, λ_G = 0.5 (H6's best).
    GolayKron(usize),
    /// Trellis-coded `T` planes + 1 bit: the finest trit restricted by a `states`-state trellis.
    Trellis(usize, usize),
}

impl Arm {
    fn label(self) -> String {
        match self {
            Arm::Ladder(t) => format!("ladder T={t}"),
            Arm::Golay(t) => format!("Golay-coded {t}+½"),
            Arm::GolayLdlq(t) => format!("Golay {t}+½ + BlockLDLQ"),
            Arm::GolayKron(t) => format!("Golay {t}+½ + Kron LDLQ"),
            Arm::TimeShare(t) => format!("time-share {t}/{}", t + 1),
            Arm::Uniform(l) => format!("uniform {l}-level scalar"),
            Arm::Trellis(t, s) => format!("trellis {t}+1bit S={s}"),
        }
    }
    /// Stored bits per weight: trits plus one fp16 step per 128-group.
    fn bits(self) -> f64 {
        let payload = match self {
            Arm::Ladder(t) => t as f64 * 3f64.log2(),
            Arm::Golay(t) | Arm::GolayLdlq(t) | Arm::GolayKron(t) | Arm::TimeShare(t) => {
                (t as f64 + 0.5) * 3f64.log2()
            }
            Arm::Uniform(l) => (l as f64).log2(),
            Arm::Trellis(t, _) => t as f64 * 3f64.log2() + 1.0,
        };
        payload + 16.0 / GROUP as f64
    }
}

fn quantize(
    w: &[f32],
    rows: usize,
    cols: usize,
    arm: Arm,
    code: &[[i8; BLOCK]],
    gram: Option<(&[f64], f64)>,
    fisher: Option<&[f64]>,
) -> Vec<f32> {
    let ladder = |t: usize, x: &[f32], r: usize| {
        ste::salt_quantize_forward_grouped_geometric(
            x,
            r,
            cols,
            t,
            GROUP,
            GRID,
            RotationPolicy::Always,
        )
    };
    match arm {
        Arm::Ladder(t) => ladder(t, w, rows),
        Arm::TimeShare(t) => {
            let mut out = vec![0.0f32; w.len()];
            out.par_chunks_mut(cols)
                .zip(w.par_chunks(cols))
                .enumerate()
                .for_each(|(r, (o, x))| o.copy_from_slice(&ladder(t + r % 2, x, 1)));
            out
        }
        Arm::Uniform(levels) => {
            let k = ((levels - 1) / 2) as f32;
            let mut out = vec![0.0f32; w.len()];
            out.par_chunks_mut(cols)
                .zip(w.par_chunks(cols))
                .for_each(|(o, x)| {
                    let mut y = x.to_vec();
                    hadamard_blocks(&mut y);
                    for (og, g) in o.chunks_mut(GROUP).zip(y.chunks(GROUP)) {
                        let peak = g.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                        let mut best = (f32::INFINITY, 0.0f32);
                        for i in 0..36 {
                            let d = peak / k * (0.35 + 0.02 * i as f32);
                            if d <= 0.0 {
                                break;
                            }
                            let e: f32 = g
                                .iter()
                                .map(|v| (v - (v / d).round().clamp(-k, k) * d).powi(2))
                                .sum();
                            if e < best.0 {
                                best = (e, d);
                            }
                        }
                        let d = best.1;
                        for (q, v) in og.iter_mut().zip(g) {
                            *q = if d > 0.0 {
                                (v / d).round().clamp(-k, k) * d
                            } else {
                                0.0
                            };
                        }
                    }
                    hadamard_blocks(o);
                });
            out
        }
        Arm::GolayLdlq(t) => {
            // No Gram (the tied embedding) or a singular one: the plain Golay fit.
            let Some(ut) =
                gram.and_then(|(h, _)| block_ldlq_factor(&rotate_gram(h, cols), cols, 0.01, BLOCK))
            else {
                return quantize(w, rows, cols, Arm::Golay(t), code, None, None);
            };
            let decay = gram.map_or(1.0, |(_, d)| d);
            let mut out = vec![0.0f32; w.len()];
            out.par_chunks_mut(cols)
                .zip(w.par_chunks(cols))
                .for_each(|(o, x)| {
                    let mut y = x.to_vec();
                    hadamard_blocks(&mut y);
                    let (steps, _) = golay_fit(&y, t, code);
                    let mut rec = golay_ldlq_row(&y, &steps, t, &ut, decay, code);
                    hadamard_blocks(&mut rec);
                    o.copy_from_slice(&rec);
                });
            out
        }
        Arm::GolayKron(t) => {
            let (Some((h, decay_a)), Some(g)) = (gram, fisher) else {
                return quantize(w, rows, cols, Arm::Golay(t), code, None, None);
            };
            let (Some(ut_a), Some(ut_g)) = (
                block_ldlq_factor(&rotate_gram(h, cols), cols, 0.01, BLOCK),
                block_ldlq_factor(g, rows, 0.01, 1),
            ) else {
                return quantize(w, rows, cols, Arm::GolayLdlq(t), code, gram, None);
            };
            let mut y = w.to_vec();
            y.par_chunks_mut(cols).for_each(hadamard_blocks);
            let steps: Vec<Vec<f64>> = y
                .par_chunks(cols)
                .map(|r| golay_fit(r, t, code).0)
                .collect();
            let mut out =
                golay_kron_tensor(&y, rows, cols, &steps, t, &ut_a, &ut_g, decay_a, 0.5, code);
            out.par_chunks_mut(cols).for_each(hadamard_blocks);
            out
        }
        Arm::Trellis(t, states) => {
            let tr = trellis_for(states, t);
            let mut out = vec![0.0f32; w.len()];
            out.par_chunks_mut(cols)
                .zip(w.par_chunks(cols))
                .for_each(|(o, x)| {
                    let mut y = x.to_vec();
                    hadamard_blocks(&mut y);
                    let (steps, units, _) = tcq_fit(&y, t, &tr);
                    for (j, (v, &u)) in o.iter_mut().zip(&units).enumerate() {
                        *v = (steps[j / GROUP] * u as f64) as f32;
                    }
                    hadamard_blocks(o);
                });
            out
        }
        Arm::Golay(t) => {
            let mut out = vec![0.0f32; w.len()];
            out.par_chunks_mut(cols)
                .zip(w.par_chunks(cols))
                .for_each(|(o, x)| {
                    let mut y = x.to_vec();
                    hadamard_blocks(&mut y);
                    let mut rec = golay_row(&y, t, code);
                    hadamard_blocks(&mut rec); // normalized H is its own inverse
                    o.copy_from_slice(&rec);
                });
            out
        }
    }
}

#[test]
fn the_code_is_the_extended_ternary_golay_code() {
    let code = golay_codewords();
    assert_eq!(code.len(), 729);
    let mut weights = [0usize; BLOCK + 1];
    for w in &code {
        weights[w.iter().filter(|&&d| d != 0).count()] += 1;
    }
    // Weight enumerator of the [12, 6, 6] extended ternary Golay code.
    assert_eq!(weights[0], 1);
    assert_eq!(weights[6], 264);
    assert_eq!(weights[9], 440);
    assert_eq!(weights[12], 24);
}

fn xorshift(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn spd(n: usize, ridge: f64, seed: u64) -> Vec<f64> {
    let mut next = xorshift(seed);
    let m: Vec<f64> = (0..n * n).map(|_| next()).collect();
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = (0..n).map(|p| m[i * n + p] * m[j * n + p]).sum::<f64>() / n as f64;
        }
        out[i * n + i] += ridge;
    }
    out
}

/// The factor is right: `U·Uᵀ` reproduces `H + λI`, and an identity metric pushes nothing.
#[test]
fn block_ldlq_factor_reduces_to_plain_decoding_on_an_identity_metric() {
    let n = 48;
    let mut eye = vec![0.0f64; n * n];
    for i in 0..n {
        eye[i * n + i] = 1.0;
    }
    let ut = block_ldlq_factor(&eye, n, 0.0, BLOCK).expect("identity is PD");
    assert!(
        ut.iter().all(|v| v.abs() < 1e-12),
        "identity metric must not couple blocks"
    );
    let code = golay_codewords();
    let mut next = xorshift(9);
    let y: Vec<f32> = (0..n).map(|_| next() as f32).collect();
    let (steps, units) = golay_fit(&y, 1, &code);
    let plain: Vec<f32> = units
        .iter()
        .enumerate()
        .map(|(j, &r)| (steps[j / GROUP] * r as f64) as f32)
        .collect();
    assert_eq!(golay_ldlq_row(&y, &steps, 1, &ut, 1.0, &code), plain);
}

/// Viterbi finds the minimum over ALL branch sequences from state 0 (brute force on short rows).
#[test]
fn trellis_viterbi_is_the_exhaustive_minimum() {
    let tr = Trellis::random(4, 99);
    let mut next = xorshift(7);
    for t in 1..=2 {
        let y: Vec<f32> = (0..7).map(|_| (next() * 3.0) as f32).collect();
        let steps = vec![0.37f64];
        let (_, err) = tcq_decode(&y, &steps, t, &tr);
        let k = kmax(t) as f64;
        let mut best = f64::INFINITY;
        for bits in 0u32..(1 << y.len()) {
            let (mut st, mut total) = (0usize, 0.0f64);
            for (j, &x) in y.iter().enumerate() {
                let b = ((bits >> j) & 1) as usize;
                let c = f64::from(tr.table[st][b]);
                let kk = ((f64::from(x) / steps[0] - c) / 3.0).round().clamp(-k, k);
                total += (f64::from(x) - steps[0] * (3.0 * kk + c)).powi(2);
                st = (2 * st + b) % tr.states;
            }
            best = best.min(total);
        }
        assert!(
            (err - best).abs() < 1e-9,
            "T={t}: viterbi {err} vs exhaustive {best}"
        );
    }
}

/// With `G = I` the Kronecker fit couples no rows: it is BlockLDLQ on every row, bit for bit.
#[test]
fn kronecker_golay_with_identity_output_metric_is_block_ldlq() {
    let (rows, cols) = (6, 48);
    let h = spd(cols, 0.05, 21);
    let ut_a = block_ldlq_factor(&h, cols, 0.0, BLOCK).expect("PD");
    let mut eye = vec![0.0f64; rows * rows];
    for i in 0..rows {
        eye[i * rows + i] = 1.0;
    }
    let ut_g = block_ldlq_factor(&eye, rows, 0.0, 1).expect("PD");
    let code = golay_codewords();
    let mut next = xorshift(33);
    let y: Vec<f32> = (0..rows * cols).map(|_| next() as f32).collect();
    let steps: Vec<Vec<f64>> = y.chunks(cols).map(|r| golay_fit(r, 1, &code).0).collect();
    let kron = golay_kron_tensor(&y, rows, cols, &steps, 1, &ut_a, &ut_g, 0.75, 0.5, &code);
    for (i, row) in y.chunks(cols).enumerate() {
        let ldlq = golay_ldlq_row(row, &steps[i], 1, &ut_a, 0.75, &code);
        assert_eq!(&kron[i * cols..(i + 1) * cols], ldlq.as_slice(), "row {i}");
    }
}

/// On a correlated metric, block feedback must lower the objective it targets, `e·H·eᵀ`.
#[test]
fn block_ldlq_lowers_the_metric_objective() {
    let n = 96;
    let h = spd(n, 0.02, 4);
    let ut = block_ldlq_factor(&h, n, 0.0, BLOCK).expect("PD");
    let code = golay_codewords();
    let objective = |e: &[f64]| -> f64 {
        (0..n)
            .map(|i| (0..n).map(|j| e[i] * h[i * n + j] * e[j]).sum::<f64>())
            .sum()
    };
    let (mut plain_total, mut ldlq_total) = (0.0, 0.0);
    for seed in 0..8 {
        let mut next = xorshift(100 + seed);
        let y: Vec<f32> = (0..n).map(|_| next() as f32).collect();
        let (steps, units) = golay_fit(&y, 1, &code);
        let plain: Vec<f64> = (0..n)
            .map(|j| f64::from(y[j]) - steps[j / GROUP] * units[j] as f64)
            .collect();
        let rec = golay_ldlq_row(&y, &steps, 1, &ut, 1.0, &code);
        let ldlq: Vec<f64> = (0..n).map(|j| f64::from(y[j] - rec[j])).collect();
        plain_total += objective(&plain);
        ldlq_total += objective(&ldlq);
    }
    assert!(
        ldlq_total < plain_total,
        "BlockLDLQ objective {ldlq_total} not below plain {plain_total}"
    );
}

#[test]
#[ignore = "H8 coded trit-planes; needs SmolLM2-135M; run explicitly"]
fn golay_coded_planes_against_the_ladder() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let ts: Vec<usize> = std::env::var("TRITIUM_H8_TS")
        .unwrap_or_else(|_| "1,2".into())
        .split(',')
        .map(|s| s.trim().parse().expect("T"))
        .collect();
    let n_eval: usize = std::env::var("TRITIUM_H8_EVAL_WINDOWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let code = golay_codewords();
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, fp, shapes) = extract(&runner);
    let (train, eval) = corpus();
    let eval = &eval[..(n_eval * WINDOW).min(eval.len())];
    let mut calib = Calib::new(&arch);
    for w in 0..CALIB_WINDOWS {
        calibrate(&fp, &arch, &train[w * WINDOW..(w + 1) * WINDOW], &mut calib);
    }
    let (folded, arch) = fold(&fp, &shapes, &arch, &calib, 0.75);
    let nll = |w: &[Vec<f32>]| perplexity_windowed(w, &arch, eval, WINDOW).ln();
    let base = nll(&folded);
    println!(
        "SmolLM2-135M | fold α=0.75 | g{GROUP} group-Hadamard | fp activations | {} eval tokens | fp ppl {:.3}\n",
        eval.len(),
        base.exp()
    );
    println!(
        "{:<22} {:>7} {:>9} {:>10} {:>10} {:>10} {:>10}  {:>11}",
        "structure", "bits/w", "ppl", "ΔL(+E)", "ΔL(−E)", "even", "odd", "SNR dB"
    );
    let mut arms = Vec::new();
    for &t in &ts {
        if arms
            .last()
            .is_none_or(|a| !matches!(a, Arm::Ladder(x) if *x == t))
        {
            arms.push(Arm::Ladder(t));
        }
        arms.extend([Arm::Golay(t), Arm::TimeShare(t), Arm::Ladder(t + 1)]);
    }
    // `TRITIUM_H8_ARMS=uniform` runs only the single-scalar comparators at the Golay rates.
    if std::env::var("TRITIUM_H8_ARMS").is_ok_and(|v| v == "uniform") {
        arms = ts
            .iter()
            .map(|&t| Arm::Uniform(if t == 1 { 5 } else { 15 }))
            .collect();
    }
    // `TRITIUM_H8_ARMS=ldlq`: Golay-coded planes with block-LDLQ feedback, against the plain Golay
    // fit, on the same 16k-token Grams and auto decay the shipping GPTQ uses (H6's comparison basis).
    let ldlq = std::env::var("TRITIUM_H8_ARMS").is_ok_and(|v| v == "ldlq");
    if ldlq {
        arms = ts.iter().flat_map(|&t| [Arm::GolayLdlq(t)]).collect();
    }
    // `TRITIUM_H8_ARMS=kron`: the Kronecker (input Gram ⊗ output Fisher) version of the same fit.
    let kron = std::env::var("TRITIUM_H8_ARMS").is_ok_and(|v| v == "kron");
    if kron {
        arms = ts.iter().map(|&t| Arm::GolayKron(t)).collect();
    }
    let ldlq = ldlq || kron;
    // `TRITIUM_H8_ARMS=trellis`: trellis-coded T planes + 1 bit at 64 and 256 states.
    if std::env::var("TRITIUM_H8_ARMS").is_ok_and(|v| v == "trellis") {
        arms = ts
            .iter()
            .flat_map(|&t| [Arm::Trellis(t, 64), Arm::Trellis(t, 256)])
            .collect();
    }
    let calib_tokens = 32 * WINDOW;
    let grams = ldlq.then(|| {
        let windows: Vec<&[u32]> = (0..32)
            .map(|w| &train[w * WINDOW..(w + 1) * WINDOW])
            .collect();
        TapGrams::collect(&folded, &arch, &windows)
    });
    let fishers = kron.then(|| {
        let windows: Vec<&[u32]> = (0..32)
            .map(|w| &train[w * WINDOW..(w + 1) * WINDOW])
            .collect();
        output_fishers(&folded, &arch, &windows, 0xF15E_A5ED)
    });
    for arm in arms {
        let started = Instant::now();
        let q: Vec<Vec<f32>> = folded
            .iter()
            .zip(&shapes)
            .enumerate()
            .map(|(i, (w, &(rows, cols)))| {
                let gram = grams.as_ref().filter(|_| i > 0).map(|g| {
                    (
                        g.for_slot((i - 1) / 7, (i - 1) % 7),
                        auto_decay(calib_tokens, cols),
                    )
                });
                let fisher = fishers
                    .as_ref()
                    .filter(|_| i > 0)
                    .map(|f| f[i - 1].as_slice());
                quantize(w, rows, cols, arm, &code, gram, fisher)
            })
            .collect();
        let (mut sig, mut err) = (0.0f64, 0.0f64);
        for (w, qv) in folded.iter().zip(&q) {
            for (a, b) in w.iter().zip(qv) {
                sig += f64::from(*a) * f64::from(*a);
                err += f64::from(a - b) * f64::from(a - b);
            }
        }
        let anti: Vec<Vec<f32>> = folded
            .iter()
            .zip(&q)
            .map(|(w, qv)| w.iter().zip(qv).map(|(a, b)| 2.0 * a - b).collect())
            .collect();
        let (plus, minus) = (nll(&q) - base, nll(&anti) - base);
        println!(
            "{:<22} {:>7.3} {:>9.3} {:>+10.5} {:>+10.5} {:>+10.5} {:>+10.5}  {:>11.2}   ({:.0}s)",
            arm.label(),
            arm.bits(),
            (base + plus).exp(),
            plus,
            minus,
            0.5 * (plus + minus),
            0.5 * (plus - minus),
            10.0 * (sig / err).log10(),
            started.elapsed().as_secs_f64()
        );
    }
}
