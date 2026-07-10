//! Plan 0039 step 2 — the HONEST gate: diagonal-Fisher SALT plane allocation beats Energy/Uniform
//! on FORWARD loss at a fixed bits-per-weight budget.
//!
//! Why this is non-tautological and non-rigged:
//!  - The allocator minimizes the *weight-space* objective `Σ_g H_g·Σ_i Δw_i²`. We do NOT gate on
//!    that. We gate on the **forward task loss** of the quantized model — a different quantity, and
//!    (crucially) a *nonlinear* one: we quantize `W1`, the layer BEHIND the `relu²`, so the loss is
//!    not a quadratic in `W1`. Fisher winning here means it *predicts forward-loss sensitivity*,
//!    not that it minimizes the thing it's defined to minimize.
//!  - The decorrelation is the canonical mechanism by which magnitude is blind and Fisher is not
//!    (the same reason gradient/Fisher pruning beats magnitude pruning): every `W1` row is built to
//!    the SAME L2 norm (so **Energy is flat → Energy ≡ Uniform**, reproducing the shipped finding),
//!    but half the hidden units are driven negative on the (non-negative) data and die through the
//!    `relu²` — they carry weight magnitude but ~zero output sensitivity. Energy spreads planes
//!    evenly and wastes half of them on dead units; Fisher concentrates planes on the live,
//!    output-bearing units. At equal bits the Fisher-quantized `W1` reconstructs the part the output
//!    depends on better → strictly lower forward loss.
//!
//! The task is **distillation**: targets are the fp model's own output, so quantization strictly
//! degrades the loss and faithfulness to fp is what's rewarded. The empirical loss-gradient is zero
//! at the teacher (perfect fit), so we use the **true Fisher = output sensitivity**
//! `F_i = E_x Σ_o (∂y_o/∂W1_i)²` — the curvature of the MSE head, nonzero even at a perfect fit and
//! the correct quantization-sensitivity signal. It is computed exactly (no sampling) via a per-output
//! unit-residual backward. We quantize only `W1` (W2 kept fp) to isolate the honest, non-quadratic
//! effect — quantizing the output layer `W2` would be linear in the loss (tautological).

use tritium_format::salt_rows_to_dense;
use tritium_quantize::fisher::tile_sensitivity;
use tritium_quantize::{QuantConfig, Sensitivity, quantize_tensor};
use tritium_train::ops::{act, dense};
use tritium_train::{FisherAccumulator, Tape};

const K: usize = 64; // inputs
const H: usize = 64; // hidden units (half live, half dead)
const M: usize = 16; // outputs
const D: usize = 48; // data samples
const BPW: f64 = 2.0; // plane budget (avg 2 planes/group; room for the allocator to concentrate)

/// Deterministic xorshift fill in `[lo, hi)`.
fn seeded(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            lo + (s % 1000) as f32 / 1000.0 * (hi - lo)
        })
        .collect()
}

/// `W1 [H,K]`: every row normalized to unit L2 norm (so per-row Energy is IDENTICAL → Energy≡Uniform).
/// Even rows are made all-positive (live: `relu²` fires on non-negative data); odd rows all-negative
/// (dead: `relu²` → 0). Magnitude is thus fully decorrelated from loss-relevance.
fn build_w1() -> Vec<f32> {
    let mut w = vec![0.0f32; H * K];
    for j in 0..H {
        let raw = seeded(100 + j as u64, K, -1.0, 1.0);
        let live = j % 2 == 0;
        let mut row: Vec<f32> = raw
            .iter()
            .map(|&v| if live { v.abs() } else { -v.abs() })
            .collect();
        let norm = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
        for v in &mut row {
            *v /= norm;
        }
        w[j * K..j * K + K].copy_from_slice(&row);
    }
    w
}

/// Plain fp forward `y = W2 · relu²(W1 · x)`, returning mean squared error to `targets`.
fn task_loss(w1: &[f32], w2: &[f32], x: &[f32], targets: &[f32]) -> f64 {
    let mut se = 0.0f64;
    for d in 0..D {
        let z = dense::forward(&x[d * K..d * K + K], w1, 1, H, K); // [1,H]
        let z2 = act::relu2_forward(&z);
        let y = dense::forward(&z2, w2, 1, M, H); // [1,M]
        for o in 0..M {
            let e = f64::from(y[o]) - f64::from(targets[d * M + o]);
            se += e * e;
        }
    }
    se / (D * M) as f64
}

/// SALT-quantize `W1` at `bpw` under `sens`, returning the dense reconstruction and per-group planes.
fn quant_recon(w1: &[f32], bpw: f64, sens: Sensitivity) -> (Vec<f32>, Vec<usize>) {
    let cfg = QuantConfig {
        budget_bpw: bpw,
        sensitivity: sens,
        ..Default::default()
    };
    let q = quantize_tensor(w1, H, K, &cfg).expect("quantize W1");
    let dense = salt_rows_to_dense(&q.salt_rows).expect("dequant");
    (dense, q.plane_counts)
}

/// Distillation targets = the fp model's own output `y = W2·relu²(W1·x)` over the batch.
fn fp_targets(w1: &[f32], w2: &[f32], x: &[f32]) -> Vec<f32> {
    let mut t = vec![0.0f32; D * M];
    for d in 0..D {
        let z2 = act::relu2_forward(&dense::forward(&x[d * K..d * K + K], w1, 1, H, K));
        t[d * M..d * M + M].copy_from_slice(&dense::forward(&z2, w2, 1, M, H));
    }
    t
}

/// The TRUE diagonal Fisher (output sensitivity) of `W1`, reduced to per-(row,block) tiles:
/// `F_i = E_x Σ_o (∂y_o/∂W1_i)²`, computed exactly via a per-(sample,output) unit-residual backward
/// (target = y − e_o ⇒ ∂mse/∂y = (2/M)·e_o ⇒ grad ∝ ∂y_o/∂W1). The common scale is rank-irrelevant.
fn true_fisher_tiles(w1: &[f32], w2: &[f32], x: &[f32]) -> Vec<f64> {
    let mut fisher = FisherAccumulator::new(H * K);
    for d in 0..D {
        for o in 0..M {
            let mut t = Tape::new();
            let w1l = t.leaf(w1.to_vec());
            let w2l = t.leaf(w2.to_vec());
            let xd = t.leaf(x[d * K..d * K + K].to_vec());
            let z = t.dense_matmul(xd, w1l, 1, H, K);
            let z2 = t.relu2(z);
            let y = t.dense_matmul(z2, w2l, 1, M, H);
            let mut tgt = t.value(y).to_vec();
            tgt[o] -= 1.0; // residual = +e_o at output o, zero elsewhere
            let tg = t.leaf(tgt);
            let loss = t.mse(y, tg);
            let grads = t.backward(loss);
            fisher.accumulate(&grads[w1l]);
        }
    }
    tile_sensitivity(&fisher.into_diag(), H, K)
}

#[test]
fn fisher_allocation_beats_energy_and_uniform_on_forward_loss() {
    let w1 = build_w1();
    let w2 = seeded(2, M * H, -0.5, 0.5);
    let x = seeded(1, D * K, 0.0, 1.0); // non-negative data
    let targets = fp_targets(&w1, &w2, &x);
    let tiles = true_fisher_tiles(&w1, &w2, &x); // one H_g per row (K < 256 → 1 block/row)

    // Sanity: Fisher must actually be concentrated on the live (even) rows, near-zero on the dead
    // ones — otherwise the gate would prove nothing about decorrelation.
    let live_f: f64 = (0..H).step_by(2).map(|j| tiles[j]).sum();
    let dead_f: f64 = (1..H).step_by(2).map(|j| tiles[j]).sum();
    assert!(
        live_f > 50.0 * dead_f.max(1e-30),
        "Fisher should sit on live rows: live {live_f:.3e} vs dead {dead_f:.3e}"
    );

    let l_fp = task_loss(&w1, &w2, &x, &targets);
    let (u, pu) = quant_recon(&w1, BPW, Sensitivity::Uniform);
    let (e, pe) = quant_recon(&w1, BPW, Sensitivity::Energy);
    let (f, pf) = quant_recon(&w1, BPW, Sensitivity::Custom(tiles));
    let l_uniform = task_loss(&u, &w2, &x, &targets);
    let l_energy = task_loss(&e, &w2, &x, &targets);
    let l_fisher = task_loss(&f, &w2, &x, &targets);

    // Planes on live vs dead rows, to show the reallocation concretely.
    let plane_split = |p: &[usize]| {
        let live: usize = (0..H).step_by(2).map(|j| p[j]).sum();
        let dead: usize = (1..H).step_by(2).map(|j| p[j]).sum();
        (live, dead)
    };
    println!(
        "0039 Fisher gate (quantize W1 @ {BPW} bpw): fp loss {l_fp:.5} | Uniform {l_uniform:.5} \
         (planes live/dead {:?}) | Energy {l_energy:.5} ({:?}) | Fisher {l_fisher:.5} ({:?}). \
         Fisher cuts the quant-loss gap by {:.1}% vs Energy.",
        plane_split(&pu),
        plane_split(&pe),
        plane_split(&pf),
        100.0 * (1.0 - (l_fisher - l_fp) / (l_energy - l_fp))
    );

    assert!(
        l_uniform > l_fp && l_energy > l_fp,
        "quantization must degrade the forward loss"
    );
    assert!(
        l_fisher < l_energy,
        "Fisher allocation must beat Energy on forward loss: {l_fisher:.6} vs {l_energy:.6}"
    );
    assert!(
        l_fisher < l_uniform,
        "Fisher allocation must beat Uniform on forward loss: {l_fisher:.6} vs {l_uniform:.6}"
    );
    // The allocator genuinely re-ranked: Fisher gave live rows more planes than dead; flat-Energy did not.
    assert!(
        plane_split(&pf).0 > plane_split(&pf).1,
        "Fisher must spend more planes on live rows than dead: {:?}",
        plane_split(&pf)
    );
}

/// Plan 0039 step 3 — adaptive plane growth: the DUAL of step 2. Instead of "lower loss at fixed
/// bits", show Fisher reaches a fixed QUALITY target at **lower average bits** than uniform growth —
/// the user's thesis "add ternary planes only up to where accuracy needs them". Growth is realized
/// through the allocator's monotone-in-budget property (higher bpw only adds planes to a tile), so
/// stepping a rising bpw grid IS incremental plane growth; the honest signal is bits-to-target.
#[test]
fn fisher_reaches_a_quality_target_at_lower_bpw_than_uniform() {
    let w1 = build_w1();
    let w2 = seeded(2, M * H, -0.5, 0.5);
    let x = seeded(1, D * K, 0.0, 1.0);
    let targets = fp_targets(&w1, &w2, &x);
    let tiles = true_fisher_tiles(&w1, &w2, &x);

    // Ascending budget grid from the ternary floor (log2 3 ≈ 1.585) toward the 3-plane ceiling.
    let grid = [1.6, 1.9, 2.2, 2.5, 2.8, 3.1, 3.4, 3.7, 4.0, 4.3, 4.6];
    let loss_at = |bpw: f64, sens: Sensitivity| {
        let (recon, _) = quant_recon(&w1, bpw, sens);
        task_loss(&recon, &w2, &x, &targets)
    };
    // Smallest grid bpw at which `sens` meets `target` (curves are monotone-decreasing in bpw).
    let min_bpw = |sens: &dyn Fn() -> Sensitivity, target: f64| {
        grid.iter().copied().find(|&b| loss_at(b, sens()) <= target)
    };

    // Quality bar = the loss uniform allocation buys at a mid-grid budget (not the plane ceiling, so
    // there is headroom for Fisher to reach it cheaper). Fisher must hit the SAME quality for less.
    let ref_bpw = grid[5]; // 3.1
    let uniform: &dyn Fn() -> Sensitivity = &|| Sensitivity::Uniform;
    let fisher: &dyn Fn() -> Sensitivity = &|| Sensitivity::Custom(tiles.clone());
    let target = loss_at(ref_bpw, Sensitivity::Uniform);

    let bpw_uniform = min_bpw(uniform, target).expect("uniform meets its own target");
    let bpw_fisher = min_bpw(fisher, target).expect("fisher meets the target");

    println!(
        "0039 adaptive-growth gate: quality target = Uniform@{ref_bpw} bpw (loss {target:.4}). \
         Uniform reaches it at {bpw_uniform} bpw; Fisher reaches it at {bpw_fisher} bpw \
         → {:.0}% of the bits for equal quality.",
        100.0 * bpw_fisher / bpw_uniform
    );

    assert!(
        (bpw_uniform - ref_bpw).abs() < 1e-9,
        "uniform should first meet its own {ref_bpw}-bpw quality exactly at {ref_bpw}: got {bpw_uniform}"
    );
    assert!(
        bpw_fisher < bpw_uniform,
        "Fisher must reach the same quality at strictly lower bpw (adaptive growth is more bit-efficient): \
         Fisher {bpw_fisher} vs Uniform {bpw_uniform}"
    );
}
