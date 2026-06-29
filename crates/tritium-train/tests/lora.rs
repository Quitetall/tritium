//! LoRA on a frozen ternary base (ADR 0007, plan 0009): frozen base receives zero
//! gradient, the adapter's A/B gradients match finite difference (Gate C), the adapter
//! merges into a dense weight correctly, and the rank edges r=1 and r=full pass.

use proptest::prelude::*;
use tritium_testkit::Tolerance;
use tritium_train::Lora;
use tritium_train::ops::{dense, loss, matmul};
use tritium_train::tape::{Tape, ValueId};

/// Deterministic xorshift64 fixture in `[lo, hi)`.
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

/// Build the frozen-base + LoRA-adapter layer on `tape`:
/// `Y = detach(base_matmul(act, trits, scale)) + (α/r)·(act·Aᵀ)·Bᵀ`.
/// Returns the output id and the leaf ids `[trits, scale, a, b, act]`.
#[allow(clippy::too_many_arguments)]
fn build_layer(
    tape: &mut Tape,
    act: Vec<f32>,
    trits: Vec<f32>,
    scale: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    m: usize,
    n: usize,
    k: usize,
    rank: usize,
    alpha: f32,
) -> (ValueId, [ValueId; 5]) {
    let act_id = tape.leaf(act);
    let trits_id = tape.leaf(trits);
    let scale_id = tape.leaf(scale);
    let a_id = tape.leaf(a);
    let b_id = tape.leaf(b);

    let base = tape.matmul(act_id, trits_id, scale_id, m, n, k);
    let base_frozen = tape.detach(base);
    let u = tape.dense_matmul(act_id, a_id, m, rank, k); // [m, rank]
    let dy = tape.dense_matmul(u, b_id, m, n, rank); // [m, n]
    let dy_scaled = tape.scale_const(dy, alpha / rank as f32);
    let y = tape.add(base_frozen, dy_scaled);
    (y, [trits_id, scale_id, a_id, b_id, act_id])
}

/// Plain (non-tape) scalar loss for finite differencing: same composition as
/// [`build_layer`] followed by MSE to `target`.
#[allow(clippy::too_many_arguments)]
fn lora_loss(
    act: &[f32],
    trits: &[f32],
    scale: &[f32],
    a: &[f32],
    b: &[f32],
    target: &[f32],
    m: usize,
    n: usize,
    k: usize,
    rank: usize,
    alpha: f32,
) -> f32 {
    let base = matmul::forward(act, trits, scale, m, n, k);
    let u = dense::forward(act, a, m, rank, k);
    let dy = dense::forward(&u, b, m, n, rank);
    let s = alpha / rank as f32;
    let y: Vec<f32> = base.iter().zip(&dy).map(|(&bb, &d)| bb + s * d).collect();
    loss::mse_forward(&y, target)[0]
}

fn gate_c_tol() -> Tolerance {
    Tolerance::relative(2e-3)
}

/// Tape analytic grads for A and B vs per-element central finite difference of
/// [`lora_loss`] — at the given rank. Small smooth fixture (no kinks in the adapter
/// path; the base is detached so its round/STE never enters the gradient).
fn check_adapter_gradient_at_rank(rank: usize, m: usize, n: usize, k: usize) {
    let alpha = 1.5f32;
    let act = seeded(1, m * k, -0.8, 0.8);
    let trits = seeded(2, n * k, -1.0, 1.0); // arbitrary frozen base weights
    let scale = seeded(3, n, 0.2, 1.2);
    let a = seeded(4, rank * k, -0.6, 0.6);
    let b = seeded(5, n * rank, -0.6, 0.6);
    let target = seeded(6, m * n, -0.5, 0.5);

    let mut tape = Tape::new();
    let (y, ids) = build_layer(
        &mut tape,
        act.clone(),
        trits.clone(),
        scale.clone(),
        a.clone(),
        b.clone(),
        m,
        n,
        k,
        rank,
        alpha,
    );
    let tg = tape.leaf(target.clone());
    let l = tape.mse(y, tg);
    let grads = tape.backward(l);
    let [_trits_id, _scale_id, a_id, b_id, _act_id] = ids;

    let h = 1e-3f32;
    let tol = gate_c_tol();
    // wrt A
    for i in 0..a.len() {
        let mut ap = a.clone();
        ap[i] += h;
        let lp = lora_loss(&act, &trits, &scale, &ap, &b, &target, m, n, k, rank, alpha);
        ap[i] -= 2.0 * h;
        let lm = lora_loss(&act, &trits, &scale, &ap, &b, &target, m, n, k, rank, alpha);
        let numeric = (lp - lm) / (2.0 * h);
        assert!(
            tol.accepts(grads[a_id][i], numeric),
            "rank {rank} A[{i}]: analytic {} vs numeric {numeric}",
            grads[a_id][i]
        );
    }
    // wrt B
    for i in 0..b.len() {
        let mut bp = b.clone();
        bp[i] += h;
        let lp = lora_loss(&act, &trits, &scale, &a, &bp, &target, m, n, k, rank, alpha);
        bp[i] -= 2.0 * h;
        let lm = lora_loss(&act, &trits, &scale, &a, &bp, &target, m, n, k, rank, alpha);
        let numeric = (lp - lm) / (2.0 * h);
        assert!(
            tol.accepts(grads[b_id][i], numeric),
            "rank {rank} B[{i}]: analytic {} vs numeric {numeric}",
            grads[b_id][i]
        );
    }
}

#[test]
fn adapter_gradient_matches_finite_difference() {
    check_adapter_gradient_at_rank(2, 3, 4, 5);
}

#[test]
fn rank_edges_r1_and_rfull_pass_gradient_check() {
    // r=1 (minimal) and r=full = min(N,K) (the factorization can represent any [N,K]
    // delta). Both must produce A/B gradients matching finite difference.
    let (n, k) = (4usize, 5usize);
    check_adapter_gradient_at_rank(1, 3, n, k);
    check_adapter_gradient_at_rank(n.min(k), 3, n, k);
}

#[test]
fn lora_delta_weights_matches_hand_computation() {
    // An integer ground-truth anchor for scaling() + delta_weights(), independent of
    // the tape path (which uses a literal α/r). n=1, k=2, rank=2, α=4 ⇒ scaling=4/2=2.
    // A=[[1,2],[3,4]] ([rank,k]); B=[[5,6]] ([n,rank]).
    //   ΔW[0,0] = 2·(5·1 + 6·3) = 46 ;  ΔW[0,1] = 2·(5·2 + 6·4) = 68
    let lora = Lora {
        a: vec![1.0, 2.0, 3.0, 4.0],
        b: vec![5.0, 6.0],
        rank: 2,
        n: 1,
        k: 2,
        alpha: 4.0,
    };
    assert_eq!(lora.scaling(), 2.0);
    assert_eq!(lora.delta_weights(), vec![46.0, 68.0]);
    // merge folds ΔW onto a dense base.
    assert_eq!(lora.merge(&[10.0, 20.0]), vec![56.0, 88.0]);
}

#[test]
fn scale_const_forward_and_backward() {
    // Pin the scale_const op directly: forward c·x and backward c·g.
    let c = 2.5f32;
    let x0 = vec![1.0f32, -2.0, 3.0];
    let target = vec![0.5f32, 0.5, 0.5];
    let mut tape = Tape::new();
    let x = tape.leaf(x0.clone());
    let y = tape.scale_const(x, c);
    assert_eq!(tape.value(y), &[2.5, -5.0, 7.5]);
    let tg = tape.leaf(target.clone());
    let l = tape.mse(y, tg);
    let grads = tape.backward(l);
    // loss = mean((c·x − t)²) ⇒ dloss/dx_i = (2/N)·(c·x_i − t_i)·c
    let nn = x0.len() as f32;
    for i in 0..x0.len() {
        let want = (2.0 / nn) * (c * x0[i] - target[i]) * c;
        assert!(
            (grads[x][i] - want).abs() < 1e-5,
            "grad[{i}] = {} want {want}",
            grads[x][i]
        );
    }
}

proptest! {
    /// The frozen base receives EXACTLY zero gradient (the `detach` cuts it), while the
    /// adapter A/B receive gradient.
    #[test]
    fn frozen_base_receives_zero_gradient(
        m in 1usize..=5,
        n in 1usize..=5,
        k in 1usize..=5,
        seed in 1u64..10_000,
        rsel in 0usize..50,
    ) {
        let rank = 1 + rsel % n.min(k);
        let act = seeded(seed.wrapping_add(1), m * k, -1.0, 1.0);
        let trits = seeded(seed.wrapping_add(2), n * k, -1.0, 1.0);
        let scale = seeded(seed.wrapping_add(3), n, 0.2, 1.5);
        let a = seeded(seed.wrapping_add(4), rank * k, -0.5, 0.5);
        let b = seeded(seed.wrapping_add(5), n * rank, -0.5, 0.5);
        let target = seeded(seed.wrapping_add(6), m * n, -1.0, 1.0);

        let mut tape = Tape::new();
        let (y, ids) = build_layer(&mut tape, act, trits, scale, a, b, m, n, k, rank, 2.0);
        let tg = tape.leaf(target);
        let l = tape.mse(y, tg);
        let grads = tape.backward(l);
        let [trits_id, scale_id, _a_id, _b_id, _act_id] = ids;

        // The frozen base receives EXACTLY zero gradient — the property under test, and
        // it holds for every input (detach cuts the path unconditionally).
        prop_assert!(
            grads[trits_id].iter().all(|&g| g == 0.0),
            "frozen base trits received a nonzero gradient"
        );
        prop_assert!(
            grads[scale_id].iter().all(|&g| g == 0.0),
            "frozen base scale received a nonzero gradient"
        );
        // (That the adapter A/B each receive the *correct* gradient is pinned by
        // `adapter_gradient_matches_finite_difference` at a non-degenerate fixture. We do
        // not assert non-zero adapter grads here: at tiny random shapes an input can
        // legitimately be 0 — e.g. act[0]=0 zeros both factors' grads — which is not a
        // freeze failure.)
    }

    /// Merging the adapter into a dense weight reproduces base + adapter forward.
    #[test]
    fn merge_matches_base_plus_adapter(
        m in 1usize..=5,
        n in 1usize..=5,
        k in 1usize..=5,
        seed in 1u64..10_000,
        rsel in 0usize..50,
    ) {
        let rank = 1 + rsel % n.min(k);
        let alpha = 2.0f32;
        let act = seeded(seed.wrapping_add(1), m * k, -1.0, 1.0);
        let trits = seeded(seed.wrapping_add(2), n * k, -1.0, 1.0);
        let scale = seeded(seed.wrapping_add(3), n, 0.2, 1.5);
        let a = seeded(seed.wrapping_add(4), rank * k, -0.5, 0.5);
        let b = seeded(seed.wrapping_add(5), n * rank, -0.5, 0.5);

        let lora = Lora { a, b, rank, n, k, alpha };

        // base_dense[n,k] = scale[n]·trits[n,k]
        let base_dense: Vec<f32> = (0..n * k).map(|i| scale[i / k] * trits[i]).collect();
        let merged = lora.merge(&base_dense);

        // Inference via the merged dense weight (this side exercises Lora::merge ->
        // delta_weights -> scaling).
        let y_merged = dense::forward(&act, &merged, m, n, k);
        // Reference: base + adapter delta computed via the INDEPENDENT B·A composition
        // (act·Aᵀ then ·Bᵀ, scaled by α/r as a literal) — deliberately NOT via
        // delta_weights(), so a wrong scaling() or delta_weights() index cannot cancel
        // on both sides and the merge gate actually constrains them.
        let base = matmul::forward(&act, &trits, &scale, m, n, k);
        let u = dense::forward(&act, &lora.a, m, rank, k); // act·Aᵀ → [m, rank]
        let dy = dense::forward(&u, &lora.b, m, n, rank); // → [m, n]
        let s = alpha / rank as f32;
        let y_composed: Vec<f32> = base
            .iter()
            .zip(&dy)
            .map(|(&p, &q)| p + s * q)
            .collect();

        let tol = Tolerance::relative(1e-4);
        for i in 0..m * n {
            prop_assert!(
                tol.accepts(y_merged[i], y_composed[i]),
                "merged {} vs base+adapter {} at {i}",
                y_merged[i], y_composed[i]
            );
        }
    }
}
