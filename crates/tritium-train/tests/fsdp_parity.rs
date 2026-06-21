//! ZeRO-3 / FSDP loss-parity gate (ADR 0008 / plan 0015): a tiny model sharded across `world`
//! simulated ranks — params [`all_gather`](tritium_train::ProcessGroup::all_gather)ed before the
//! forward, grads [`reduce_scatter`](tritium_train::ProcessGroup::reduce_scatter)ed after the backward,
//! optimizer state sharded — produces a loss curve that tracks a single-process full-batch reference.
//!
//! The reference [`baseline`] is an *independent* single-threaded loop (no sharding, no collectives),
//! so a coherent bug copied into both paths cannot pass the gate (the 0013 verification-gap lesson).
//! Parity is **bit-exact** where the float arithmetic permits (world=1; world=2 with replicated data,
//! where `(G+G)/2 == G` exactly) and **within a measured tolerance** for data-parallel partitioning,
//! where per-rank partial sums fold the examples in a different order than the single-process reference.

use std::thread;
use tritium_train::optim::{AdamState, AdamW, Optimizer};
use tritium_train::tape::Tape;
use tritium_train::{FlatShardPlan, ProcessGroup, ReduceOp, SimProcessGroup};

// Tiny dense 2-layer MLP: x[B,D_IN] → W1ᵀ → squared-ReLU → W2ᵀ → MSE(target[B,D_OUT]).
const D_IN: usize = 4;
const H: usize = 6;
const D_OUT: usize = 3;
const B: usize = 8; // global batch; divisible by the worlds under test (1,2,4)
const STEPS: u64 = 20;
const LR: f32 = 0.05;

// Flat parameter = [W1 (H*D_IN) ++ W2 (D_OUT*H)] = 24 + 18 = 42 elements (in this leaf order).
fn leaf_lens() -> [usize; 2] {
    [H * D_IN, D_OUT * H]
}

/// splitmix64-derived deterministic f32 vector in `[lo, hi)` (matches the repo seed idiom).
fn seeded(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut z = seed.wrapping_add(0x1234_5678);
    (0..n)
        .map(|_| {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x ^= x >> 31;
            lo + ((x >> 40) as f32 / (1u64 << 24) as f32) * (hi - lo)
        })
        .collect()
}

/// Seeded initial leaves `[W1, W2]` — small magnitudes keep the loss tame and training stable.
fn init_leaves() -> Vec<Vec<f32>> {
    vec![
        seeded(1, H * D_IN, -0.3, 0.3),
        seeded(2, D_OUT * H, -0.3, 0.3),
    ]
}

/// Seeded global batch `(X[B,D_IN], Y[B,D_OUT])`.
fn data() -> (Vec<f32>, Vec<f32>) {
    (
        seeded(3, B * D_IN, -1.0, 1.0),
        seeded(4, B * D_OUT, -1.0, 1.0),
    )
}

/// Forward + backward of the tiny MLP on `b` examples (`x[b,D_IN]`, `y[b,D_OUT]`). Returns the scalar
/// loss and the per-leaf gradients `[gW1, gW2]` (the trusted tape, gradient-checked in 0011).
fn forward_backward(leaves: &[Vec<f32>], x: &[f32], y: &[f32], b: usize) -> (f32, Vec<Vec<f32>>) {
    let mut t = Tape::new();
    let w1 = t.leaf(leaves[0].clone());
    let w2 = t.leaf(leaves[1].clone());
    let xid = t.leaf(x.to_vec());
    let yid = t.leaf(y.to_vec());
    let h_pre = t.dense_matmul(xid, w1, b, H, D_IN);
    let h = t.relu2(h_pre);
    let pred = t.dense_matmul(h, w2, b, D_OUT, H);
    let lid = t.mse(pred, yid);
    let loss = t.value(lid)[0];
    let grads = t.backward(lid);
    (loss, vec![grads[w1].clone(), grads[w2].clone()])
}

/// Independent single-process reference: full-batch, per-leaf AdamW, no sharding / collectives.
/// Returns the loss curve and the final leaves.
fn baseline(steps: u64) -> (Vec<f32>, Vec<Vec<f32>>) {
    let mut leaves = init_leaves();
    let (x, y) = data();
    let opt = AdamW::new(LR);
    let mut states: Vec<AdamState> = leaves.iter().map(|l| opt.init_state(l.len())).collect();
    let mut curve = Vec::with_capacity(steps as usize);
    for step in 0..steps {
        let (loss, grads) = forward_backward(&leaves, &x, &y, B);
        for ((leaf, grad), st) in leaves.iter_mut().zip(&grads).zip(&mut states) {
            opt.step(step + 1, leaf, grad, st);
        }
        curve.push(loss);
    }
    (curve, leaves)
}

/// One rank's FSDP training loop. Owns its param shard + sharded AdamW state; each step gathers the
/// full params, runs forward/backward on its local data slice, reduce_scatters the grad, steps its
/// shard, and all_reduces the loss. Returns `(loss_curve, final_param_shard)`.
fn rank_loop(
    pg: SimProcessGroup,
    plan: &FlatShardPlan,
    init_flat: &[f32],
    x_full: &[f32],
    y_full: &[f32],
    steps: u64,
    replicated: bool,
) -> (Vec<f32>, Vec<f32>) {
    let rank = pg.rank();
    let world = pg.world_size();
    let (lo, hi) = plan.shard_range(rank);
    let mut shard = init_flat[lo..hi].to_vec();

    // Local data: the full batch (replicated) or this rank's equal contiguous slice (partitioned).
    let (x_local, y_local, b_local) = if replicated {
        (x_full.to_vec(), y_full.to_vec(), B)
    } else {
        let bl = B / world;
        let xs = rank * bl * D_IN;
        let ys = rank * bl * D_OUT;
        (
            x_full[xs..xs + bl * D_IN].to_vec(),
            y_full[ys..ys + bl * D_OUT].to_vec(),
            bl,
        )
    };

    let opt = AdamW::new(LR);
    let mut state = AdamState {
        m: vec![0.0; shard.len()],
        v: vec![0.0; shard.len()],
    };
    let mut curve = Vec::with_capacity(steps as usize);
    for step in 0..steps {
        // 1. all_gather the full parameters from every rank's shard.
        let mut full = vec![0.0f32; plan.padded_len()];
        pg.all_gather(&shard, &mut full).expect("all_gather");
        let leaves = plan.unflatten(&full);
        // 2. forward + backward on the local data.
        let (loss, grads) = forward_backward(&leaves, &x_local, &y_local, b_local);
        // 3. reduce_scatter (Avg) the full grad → this rank's grad shard.
        let grad_flat = plan.flatten(&grads);
        let mut grad_shard = vec![0.0f32; plan.chunk()];
        pg.reduce_scatter(&grad_flat, &mut grad_shard, ReduceOp::Avg)
            .expect("reduce_scatter");
        // 4. AdamW step on the local param shard (sharded state).
        opt.step(step + 1, &mut shard, &grad_shard, &mut state);
        // 5. all_reduce (Avg) the scalar loss → the global loss for the curve.
        let mut lbuf = [loss];
        pg.all_reduce(&mut lbuf, ReduceOp::Avg).expect("all_reduce");
        curve.push(lbuf[0]);
    }
    (curve, shard)
}

/// Run the FSDP loop across `world` simulated ranks (one thread each). Returns the (shared) loss
/// curve and the reassembled final leaves.
fn run_fsdp(world: usize, steps: u64, replicated: bool) -> (Vec<f32>, Vec<Vec<f32>>) {
    let plan = FlatShardPlan::new(&leaf_lens(), world);
    let init_flat = plan.flatten(&init_leaves());
    let (x_full, y_full) = data();
    let pgs = SimProcessGroup::world(world);

    let plan_ref = &plan;
    let init_ref = &init_flat;
    let x_ref = &x_full;
    let y_ref = &y_full;
    let per_rank: Vec<(Vec<f32>, Vec<f32>)> = thread::scope(|s| {
        let handles: Vec<_> = pgs
            .into_iter()
            .map(|pg| {
                s.spawn(move || rank_loop(pg, plan_ref, init_ref, x_ref, y_ref, steps, replicated))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Every rank holds the same all-reduced curve; assert they agree, then take rank 0's.
    for (rank, (curve, _)) in per_rank.iter().enumerate() {
        assert_eq!(curve, &per_rank[0].0, "rank {rank} loss curve disagrees");
    }
    // Reassemble the final flat params from the shards in rank order.
    let mut full = vec![0.0f32; plan.padded_len()];
    for (rank, (_, shard)) in per_rank.iter().enumerate() {
        let (lo, hi) = plan.shard_range(rank);
        full[lo..hi].copy_from_slice(shard);
    }
    (per_rank[0].0.clone(), plan.unflatten(&full))
}

fn assert_finite(curve: &[f32], label: &str) {
    assert!(
        curve.iter().all(|x| x.is_finite()),
        "{label}: non-finite loss in curve {curve:?}"
    );
}

#[test]
fn baseline_trains_and_is_finite() {
    let (curve, _) = baseline(STEPS);
    assert_finite(&curve, "baseline");
    // Sanity: the reference actually learns (loss drops meaningfully over the budget).
    assert!(
        curve[curve.len() - 1] < curve[0] * 0.5,
        "baseline did not train: {} → {}",
        curve[0],
        curve[curve.len() - 1]
    );
}

#[test]
fn world1_fsdp_is_bit_exact_to_baseline() {
    let (base_curve, base_leaves) = baseline(STEPS);
    // world=1: one shard = the whole flat buffer, gather/reduce_scatter are identity — the FSDP
    // orchestration must reproduce the plain loop exactly (it is *different code*).
    let (curve, leaves) = run_fsdp(1, STEPS, false);
    assert_eq!(curve, base_curve, "world=1 loss curve not bit-exact");
    assert_eq!(leaves, base_leaves, "world=1 final params not bit-exact");
}

#[test]
fn world2_replicated_fsdp_is_bit_exact_to_baseline() {
    let (base_curve, base_leaves) = baseline(STEPS);
    // Replicated data + world=2: each rank computes the identical full grad G; reduce_scatter(Avg)
    // yields (G+G)/2 == G exactly (power-of-two scaling never rounds), and AdamW is element-wise —
    // so the sharded run is bit-exact to the full-buffer baseline. Isolates the sharding mechanics.
    let (curve, leaves) = run_fsdp(2, STEPS, true);
    assert_eq!(
        curve, base_curve,
        "world=2 replicated loss curve not bit-exact"
    );
    assert_eq!(
        leaves, base_leaves,
        "world=2 replicated final params not bit-exact"
    );
}

#[test]
fn data_parallel_partition_parity_within_tolerance() {
    let (base_curve, base_leaves) = baseline(STEPS);
    // Data-parallel: partition the global batch into equal per-rank slices, Avg-reduce. Mathematically
    // equal to the full-batch grad, but the per-rank partial sums reorder the float adds, so parity is
    // within a measured tolerance, not bit-exact. Measured max |Δloss| over STEPS steps is ~4.5e-8
    // (≈1 ULP at this loss magnitude) for both world=2 and world=4 — gate at 1e-4 absolute / 1e-3
    // relative leaves a ~2000× margin (tight enough to catch a wrong reduce op, loose enough for f32).
    const ABS_TOL: f32 = 1e-4;
    const REL_TOL: f32 = 1e-3;
    for world in [2usize, 4] {
        let (curve, leaves) = run_fsdp(world, STEPS, false);
        assert_finite(&curve, &format!("world={world}"));
        assert_eq!(curve.len(), base_curve.len());
        let mut max_abs = 0.0f32;
        for (t, (&got, &want)) in curve.iter().zip(&base_curve).enumerate() {
            let abs = (got - want).abs();
            let rel = abs / want.abs().max(1e-6);
            max_abs = max_abs.max(abs);
            assert!(
                abs <= ABS_TOL || rel <= REL_TOL,
                "world={world} step {t}: loss {got} vs baseline {want} (abs {abs}, rel {rel})"
            );
        }
        // Final params also track within tolerance (looser: accumulated drift over the curve).
        for (li, (lf, lb)) in leaves.iter().zip(&base_leaves).enumerate() {
            for (i, (&g, &w)) in lf.iter().zip(lb).enumerate() {
                let abs = (g - w).abs();
                assert!(
                    abs <= 1e-3,
                    "world={world} leaf {li}[{i}]: param {g} vs baseline {w} (abs {abs})"
                );
            }
        }
        assert!(
            max_abs <= ABS_TOL,
            "world={world}: max |Δloss| {max_abs} exceeds {ABS_TOL}"
        );
    }
}

#[test]
fn fsdp_loss_curve_is_deterministic_across_schedulings() {
    // The fixed-order collective fold (0014) must make the whole training loop schedule-independent:
    // repeated runs are bit-identical, no matter how the rank threads interleave.
    for world in [2usize, 4] {
        let (a, pa) = run_fsdp(world, STEPS, false);
        for _ in 0..6 {
            let (b, pb) = run_fsdp(world, STEPS, false);
            assert_eq!(a, b, "world={world}: FSDP loss curve not deterministic");
            assert_eq!(pa, pb, "world={world}: FSDP final params not deterministic");
        }
    }
}
