//! SALT-STE distillation (plan 0038 step 1): the multi-plane SALT quantizer wired as a
//! straight-through estimator lets AdamW train an fp32 latent so the ternary reconstruction
//! recovers the quantization gap that plain PTQ leaves.

use tritium_train::ops::ste::{absmean_scale_per_row, quantize_forward, salt_quantize_forward};
use tritium_train::{AdamW, Optimizer, Tape};

/// `Y[m,n] = Σ_k X[m,k]·W[n,k]` (`W` is `[n,k]` row-major).
fn matmul(x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += x[mi * k + ki] * w[ni * k + ki];
            }
            y[mi * n + ni] = acc;
        }
    }
    y
}

fn mse(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = f64::from(x) - f64::from(y);
            d * d
        })
        .sum::<f64>()
        / a.len() as f64
}

/// Deterministic pseudo-random values in ~[-0.5, 0.5) (LCG). Uses the top 32 bits (`>> 32`)
/// so the range is symmetric — a `>> 33` numerator would be all-negative and never exercise
/// the `+1` ternary state.
fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 32) as u32 as f32) / (u32::MAX as f32) - 0.5
        })
        .collect()
}

#[test]
#[allow(clippy::needless_range_loop)]
fn salt_forward_t1_is_absmean_ternary_and_monotonic_in_planes() {
    let (rows, cols) = (4usize, 300usize);
    let w = rand_vec(rows * cols, 7);

    // T=1: Ŵ = s·round(clamp(w/s)) = s·trit.
    let s1 = absmean_scale_per_row(&w, rows, cols);
    let trits = quantize_forward(&w, &s1, rows, cols);
    let salt1 = salt_quantize_forward(&w, rows, cols, 1);
    for r in 0..rows {
        for c in 0..cols {
            let i = r * cols + c;
            assert!(
                (salt1[i] - s1[r] * trits[i]).abs() < 1e-6,
                "T=1 must equal s·trit at [{r},{c}]"
            );
        }
    }

    // More planes ⇒ strictly better reconstruction of the latent.
    let e1 = mse(&salt1, &w);
    let e2 = mse(&salt_quantize_forward(&w, rows, cols, 2), &w);
    let e3 = mse(&salt_quantize_forward(&w, rows, cols, 3), &w);
    assert!(
        e2 < e1 && e3 < e2,
        "recon error must drop with T: {e1} {e2} {e3}"
    );

    // Both signs are exercised (guards against a sign-broken RNG).
    assert!(
        w.iter().any(|&x| x > 0.0) && w.iter().any(|&x| x < 0.0),
        "test weights must span both signs"
    );

    // Large T ⇒ near-exact reconstruction (Ŵ → Wf). This is the property the identity STE
    // backward rests on (dŴ/dWf → I as planes accumulate).
    let e8 = mse(&salt_quantize_forward(&w, rows, cols, 8), &w);
    assert!(
        e8 < 0.02 * e1,
        "T=8 must reconstruct near-exactly (Ŵ→Wf): {e8} vs {e1}"
    );
}

#[test]
fn salt_ste_distillation_recovers_the_quant_gap() {
    let (m, n, k) = (8usize, 6usize, 32usize);
    let t = 2usize;
    let w_teacher = rand_vec(n * k, 11);
    let x = rand_vec(m * k, 13);
    let target = matmul(&x, &w_teacher, m, n, k);

    // Naive PTQ: SALT-quantize the teacher weights, no training.
    let w_ptq = salt_quantize_forward(&w_teacher, n, k, t);
    let mse_ptq = mse(&matmul(&x, &w_ptq, m, n, k), &target);

    // Distill: the latent starts at the teacher weights; train it (STE through salt_ste) so the
    // ternary reconstruction matches the teacher OUTPUT better.
    let mut latent = w_teacher.clone();
    let opt = AdamW::new(3e-3);
    let mut state = opt.init_state(latent.len());
    let (mut first_loss, mut last_loss) = (f32::NAN, f32::NAN);
    for step in 1..=300u64 {
        let mut tape = Tape::new();
        let wf = tape.leaf(latent.clone());
        let xi = tape.leaf(x.clone());
        let tg = tape.leaf(target.clone());
        let w_hat = tape.salt_ste(wf, n, k, t);
        let y = tape.dense_matmul(xi, w_hat, m, n, k);
        let loss = tape.mse(y, tg);
        let loss_val = tape.value(loss)[0];
        if step == 1 {
            first_loss = loss_val;
        }
        last_loss = loss_val;
        let grads = tape.backward(loss);
        opt.step(step, &mut latent, &grads[wf], &mut state);
    }
    let w_distilled = salt_quantize_forward(&latent, n, k, t);
    let mse_distilled = mse(&matmul(&x, &w_distilled, m, n, k), &target);

    let recovered = 100.0 * (1.0 - mse_distilled / mse_ptq);
    println!(
        "SALT-STE distill: PTQ mse {mse_ptq:.3e} → {mse_distilled:.3e} ({recovered:.1}% recovered)"
    );
    // The training loss must actually descend...
    assert!(
        last_loss < first_loss,
        "distillation loss must decrease: {first_loss} → {last_loss}"
    );
    // ...and recover most of the gap (observed ~93%; a tight threshold gives regression teeth).
    assert!(
        mse_distilled < 0.3 * mse_ptq,
        "distillation must recover ≥70% of the quant gap: {mse_distilled:.3e} vs PTQ {mse_ptq:.3e}"
    );
}
