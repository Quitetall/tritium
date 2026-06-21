//! Distributed-checkpoint gate (ADR 0008 / plan 0016): a checkpoint saved with `K` shards loads,
//! reshards to `J`, and yields an identical forward; a mid-run save + restore continues the loss curve
//! bit-exactly; and an interrupted / corrupted save never replaces or silently loads a torn checkpoint.
//!
//! Builds on [`FlatShardPlan`] (0015) and the thread-simulated [`SimProcessGroup`] (0014). The DCP
//! global state is world-agnostic, so resharding is just `FlatShardPlan::new(leaf_lens, J)` on the
//! loaded global buffers — and resume bit-exactness (same world) proves the optimizer state (m, v) is
//! checkpointed and reassembled exactly, not just the parameters.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use tritium_train::dcp::{self, DcpError, DistCheckpoint};
use tritium_train::optim::{AdamState, AdamW, Optimizer};
use tritium_train::tape::Tape;
use tritium_train::{FlatShardPlan, ProcessGroup, ReduceOp, SimProcessGroup};

// Same tiny dense MLP as the 0015 parity gate: x[B,D_IN] → W1ᵀ → squared-ReLU → W2ᵀ → MSE.
const D_IN: usize = 4;
const H: usize = 6;
const D_OUT: usize = 3;
const B: usize = 8;
const STEPS: u64 = 20;
const LR: f32 = 0.05;
const LEAF_LENS: [usize; 2] = [H * D_IN, D_OUT * H]; // [24, 18]
const TOTAL: usize = H * D_IN + D_OUT * H; // 42

/// A unique scratch directory, removed on drop (best-effort).
struct TmpDir(PathBuf);
impl TmpDir {
    fn new(tag: &str) -> Self {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("tritium_dcp_{}_{tag}_{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("create tmp dir");
        TmpDir(p)
    }
    fn path(&self) -> &Path {
        &self.0
    }
}
impl Drop for TmpDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// splitmix64-derived deterministic f32 vector in `[lo, hi)`.
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

fn init_flat() -> Vec<f32> {
    let mut p = seeded(1, H * D_IN, -0.3, 0.3);
    p.extend(seeded(2, D_OUT * H, -0.3, 0.3));
    p
}

fn data() -> (Vec<f32>, Vec<f32>) {
    (
        seeded(3, B * D_IN, -1.0, 1.0),
        seeded(4, B * D_OUT, -1.0, 1.0),
    )
}

/// Split a flat global parameter (length `TOTAL`) into per-leaf buffers.
fn split_leaves(flat: &[f32]) -> Vec<Vec<f32>> {
    FlatShardPlan::new(&LEAF_LENS, 1).unflatten(flat)
}

/// Forward + backward of the MLP on `b` examples. Returns the scalar loss and `[gW1, gW2]`.
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

/// Full-batch forward loss at a flat global parameter (a concrete "forward").
fn mlp_loss(param: &[f32]) -> f32 {
    let (x, y) = data();
    forward_backward(&split_leaves(param), &x, &y, B).0
}

/// This rank's equal contiguous slice of the global batch.
fn local_slice(
    rank: usize,
    world: usize,
    x_full: &[f32],
    y_full: &[f32],
) -> (Vec<f32>, Vec<f32>, usize) {
    assert_eq!(B % world, 0, "B={B} must be divisible by world={world}");
    let bl = B / world;
    let xs = rank * bl * D_IN;
    let ys = rank * bl * D_OUT;
    (
        x_full[xs..xs + bl * D_IN].to_vec(),
        y_full[ys..ys + bl * D_OUT].to_vec(),
        bl,
    )
}

/// The full global training state: step + flat param + AdamW moments (each length `TOTAL`).
#[derive(Clone, Debug, PartialEq)]
struct GlobalState {
    step: u64,
    param: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
}

fn to_dist(g: &GlobalState) -> DistCheckpoint {
    DistCheckpoint {
        step: g.step,
        leaf_lens: LEAF_LENS.to_vec(),
        param: g.param.clone(),
        planes: vec![g.m.clone(), g.v.clone()],
    }
}

fn from_dist(d: DistCheckpoint) -> GlobalState {
    assert_eq!(d.planes.len(), 2, "expected AdamW [m, v] planes");
    let mut planes = d.planes.into_iter();
    GlobalState {
        step: d.step,
        param: d.param,
        m: planes.next().unwrap(),
        v: planes.next().unwrap(),
    }
}

/// One rank's training outcome: its loss curve and its final param / AdamW-moment shards.
struct RankOutcome {
    curve: Vec<f32>,
    param: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
}

/// FSDP training for `extra` steps starting from a global state `g`, across `world` simulated ranks.
/// Shards `g`'s param + AdamW moments by `FlatShardPlan`, trains (step counter continues from
/// `g.step`), and reassembles the final global state. Returns the (shared) loss curve.
fn fsdp_train_from(world: usize, g: &GlobalState, extra: u64) -> (Vec<f32>, GlobalState) {
    let plan = FlatShardPlan::new(&LEAF_LENS, world);
    let padded = plan.padded_len();
    let mut param = g.param.clone();
    param.resize(padded, 0.0);
    let mut m = g.m.clone();
    m.resize(padded, 0.0);
    let mut v = g.v.clone();
    v.resize(padded, 0.0);
    let (x_full, y_full) = data();
    let start = g.step;

    let plan_ref = &plan;
    let pr = &param;
    let mr = &m;
    let vr = &v;
    let xr = &x_full;
    let yr = &y_full;
    let per_rank: Vec<RankOutcome> = thread::scope(|s| {
        let handles: Vec<_> = SimProcessGroup::world(world)
            .into_iter()
            .map(|pg| {
                s.spawn(move || {
                    let rank = pg.rank();
                    let world = pg.world_size();
                    let (lo, hi) = plan_ref.shard_range(rank);
                    let mut shard = pr[lo..hi].to_vec();
                    let mut state = AdamState {
                        m: mr[lo..hi].to_vec(),
                        v: vr[lo..hi].to_vec(),
                    };
                    let (xl, yl, bl) = local_slice(rank, world, xr, yr);
                    let opt = AdamW::new(LR);
                    let mut curve = Vec::with_capacity(extra as usize);
                    for i in 0..extra {
                        let mut full = vec![0.0f32; plan_ref.padded_len()];
                        pg.all_gather(&shard, &mut full).expect("all_gather");
                        let leaves = plan_ref.unflatten(&full);
                        let (loss, grads) = forward_backward(&leaves, &xl, &yl, bl);
                        let grad_flat = plan_ref.flatten(&grads);
                        let mut grad_shard = vec![0.0f32; plan_ref.chunk()];
                        pg.reduce_scatter(&grad_flat, &mut grad_shard, ReduceOp::Avg)
                            .expect("reduce_scatter");
                        opt.step(start + 1 + i, &mut shard, &grad_shard, &mut state);
                        let mut lbuf = [loss];
                        pg.all_reduce(&mut lbuf, ReduceOp::Avg).expect("all_reduce");
                        curve.push(lbuf[0]);
                    }
                    RankOutcome {
                        curve,
                        param: shard,
                        m: state.m,
                        v: state.v,
                    }
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    for (r, out) in per_rank.iter().enumerate() {
        assert_eq!(
            out.curve, per_rank[0].curve,
            "rank {r} loss curve disagrees"
        );
    }
    let mut fp = vec![0.0f32; padded];
    let mut fm = vec![0.0f32; padded];
    let mut fv = vec![0.0f32; padded];
    for (r, out) in per_rank.iter().enumerate() {
        let (lo, hi) = plan.shard_range(r);
        fp[lo..hi].copy_from_slice(&out.param);
        fm[lo..hi].copy_from_slice(&out.m);
        fv[lo..hi].copy_from_slice(&out.v);
    }
    fp.truncate(TOTAL);
    fm.truncate(TOTAL);
    fv.truncate(TOTAL);
    (
        per_rank[0].curve.clone(),
        GlobalState {
            step: start + extra,
            param: fp,
            m: fm,
            v: fv,
        },
    )
}

/// A synthetic global state (arbitrary but deterministic param + moments) for round-trip tests.
fn synthetic_global(step: u64) -> GlobalState {
    GlobalState {
        step,
        param: seeded(10, TOTAL, -0.5, 0.5),
        m: seeded(11, TOTAL, -0.2, 0.2),
        v: seeded(12, TOTAL, 0.0, 0.3),
    }
}

#[test]
fn save_load_round_trips_global_state_for_any_world() {
    // The DCP global state is world-agnostic: saving with K shards and loading reassembles the exact
    // same global (param + both moments + step), for every K. This is the heart of resharding.
    let g = synthetic_global(5);
    for k in [1usize, 2, 4] {
        let dir = TmpDir::new(&format!("rt_k{k}"));
        dcp::save(dir.path(), &to_dist(&g), k).unwrap();
        let loaded = from_dist(dcp::load(dir.path()).unwrap());
        assert_eq!(
            loaded, g,
            "save world={k} did not round-trip the global state"
        );
    }
}

#[test]
fn save_k_reshard_j_gives_identical_forward() {
    // Save with K shards, load, then RE-SAVE with J shards into a fresh dir and reload — a real K→J
    // reshard through J-sharded files on disk (param AND optimizer planes). The round-tripped global is
    // identical for every (K, J), and the forward matches — so no J≠K slicing/padding/plane-drop bug
    // can hide (each is materialized as real shard files and read back).
    let g = synthetic_global(9);
    let reference = mlp_loss(&g.param);
    for k in [1usize, 2, 4] {
        let dir_k = TmpDir::new(&format!("rf_k{k}"));
        dcp::save(dir_k.path(), &to_dist(&g), k).unwrap();
        let loaded = from_dist(dcp::load(dir_k.path()).unwrap());
        assert_eq!(loaded, g, "load (save world={k}) changed the global state");
        for j in [1usize, 2, 4] {
            let dir_j = TmpDir::new(&format!("rf_k{k}_j{j}"));
            dcp::save(dir_j.path(), &to_dist(&loaded), j).unwrap();
            let resharded = from_dist(dcp::load(dir_j.path()).unwrap());
            assert_eq!(
                resharded, g,
                "K={k} J={j}: resharded global differs (param/m/v/step)"
            );
            assert_eq!(
                mlp_loss(&resharded.param),
                reference,
                "K={k} J={j}: forward differs"
            );
        }
    }
}

#[test]
fn distributed_resume_continues_the_curve_bit_exact() {
    // Save mid-run via DCP, restore, and continue — the resumed loss curve must equal the uninterrupted
    // curve bit-for-bit (same world). This proves the optimizer state (m, v) and step counter survive
    // the distributed-checkpoint round-trip exactly, not just the parameters.
    const HALF: u64 = 8;
    let init = GlobalState {
        step: 0,
        param: init_flat(),
        m: vec![0.0; TOTAL],
        v: vec![0.0; TOTAL],
    };
    for world in [1usize, 2, 4] {
        let (curve_full, _) = fsdp_train_from(world, &init, STEPS);
        let (curve_a, mid) = fsdp_train_from(world, &init, HALF);
        let dir = TmpDir::new(&format!("resume_w{world}"));
        dcp::save(dir.path(), &to_dist(&mid), world).unwrap();
        let loaded = from_dist(dcp::load(dir.path()).unwrap());
        assert_eq!(
            loaded, mid,
            "world={world}: DCP round-trip changed the mid-run state"
        );
        let (curve_b, _) = fsdp_train_from(world, &loaded, STEPS - HALF);
        let mut resumed = curve_a;
        resumed.extend(curve_b);
        assert_eq!(
            resumed, curve_full,
            "world={world}: resumed curve != uninterrupted (not bit-exact)"
        );
    }
}

/// Overwrite `len` bytes of `path` starting at `offset` with `repl`.
fn patch_file(path: &Path, offset: usize, repl: &[u8]) {
    let mut b = std::fs::read(path).unwrap();
    b[offset..offset + repl.len()].copy_from_slice(repl);
    std::fs::write(path, b).unwrap();
}

#[test]
fn uncommitted_newer_shards_do_not_shadow_committed_checkpoint() {
    // The crux of "the manifest is the single commit point": a committed checkpoint, then a fully
    // written but UNCOMMITTED newer save (real, valid step-20 shard files copied in; manifest NOT
    // updated). load() must still return the committed step-10 checkpoint — complete new shards are
    // inert until their manifest commits. This is the real rank-killed-mid-save scenario (shards
    // landed, manifest not yet committed), not a garbage-bytes stand-in.
    let dir = TmpDir::new("fault_uncommitted");
    let g1 = synthetic_global(10);
    dcp::save(dir.path(), &to_dist(&g1), 2).unwrap();

    // Stage a genuine, different step-20 checkpoint elsewhere, then copy ONLY its shards into `dir`.
    let staging = TmpDir::new("fault_staging");
    let g2 = GlobalState {
        step: 20,
        param: seeded(99, TOTAL, -1.0, 1.0),
        m: seeded(98, TOTAL, -1.0, 1.0),
        v: seeded(97, TOTAL, 0.0, 1.0),
    };
    dcp::save(staging.path(), &to_dist(&g2), 2).unwrap();
    for rank in [0usize, 1] {
        let name = format!("step_20_shard_{rank:04}.tdcp");
        std::fs::copy(staging.path().join(&name), dir.path().join(&name)).unwrap();
    }
    // Plus a half-written manifest .tmp from the interrupted commit, to confirm load ignores it.
    std::fs::write(dir.path().join("manifest.tdcp.tmp"), b"half-written").unwrap();

    let loaded = from_dist(dcp::load(dir.path()).unwrap());
    assert_eq!(
        loaded, g1,
        "uncommitted newer shards shadowed the committed checkpoint"
    );
}

#[test]
fn save_rejects_non_monotonic_step() {
    // Re-saving an already-committed step would overwrite live shard files in place (tearing risk);
    // save() rejects it before writing anything. A strictly-greater step is accepted.
    let dir = TmpDir::new("monotonic");
    dcp::save(dir.path(), &to_dist(&synthetic_global(10)), 2).unwrap();
    assert_eq!(
        dcp::save(dir.path(), &to_dist(&synthetic_global(10)), 2),
        Err(DcpError::NonMonotonicSave {
            committed: 10,
            attempted: 10
        })
    );
    assert_eq!(
        dcp::save(dir.path(), &to_dist(&synthetic_global(5)), 2),
        Err(DcpError::NonMonotonicSave {
            committed: 10,
            attempted: 5
        })
    );
    // A greater step is fine, and load() returns the newest committed step.
    dcp::save(dir.path(), &to_dist(&synthetic_global(11)), 2).unwrap();
    assert_eq!(dcp::load(dir.path()).unwrap().step, 11);
}

#[test]
fn load_never_panics_on_corrupt_manifest_fields() {
    // The never-panic-on-untrusted-bytes contract for the structural manifest fields: a corrupt
    // world / n_planes / version must return a clean DcpError, never panic FlatShardPlan or an alloc.
    // Manifest layout: magic[0..4] ver[4] step[5..13] world[13..21] n_planes[21..29] ...
    let make = |tag: &str| {
        let dir = TmpDir::new(tag);
        dcp::save(dir.path(), &to_dist(&synthetic_global(2)), 2).unwrap();
        dir
    };
    // world = 0 → InvalidManifest (would otherwise assert in FlatShardPlan::new).
    let d = make("corrupt_world");
    patch_file(&d.path().join("manifest.tdcp"), 13, &0u64.to_le_bytes());
    assert_eq!(
        dcp::load(d.path()),
        Err(DcpError::InvalidManifest("world size is zero"))
    );
    // n_planes = u64::MAX → TooManyPlanes (would otherwise drive an unbounded Vec allocation).
    let d = make("corrupt_planes");
    patch_file(&d.path().join("manifest.tdcp"), 21, &u64::MAX.to_le_bytes());
    assert_eq!(
        dcp::load(d.path()),
        Err(DcpError::TooManyPlanes {
            got: usize::MAX,
            max: tritium_train::dcp::MAX_STATE_PLANES
        })
    );
    // version byte → UnsupportedVersion.
    let d = make("corrupt_version");
    patch_file(&d.path().join("manifest.tdcp"), 4, &[99u8]);
    assert_eq!(dcp::load(d.path()), Err(DcpError::UnsupportedVersion(99)));
}

#[test]
fn swapped_shards_are_detected() {
    // Swap two shard files' contents on disk: the file load() reads for rank 0 now carries rank=1 in
    // its header → ShardMismatch{rank}, not a silently-permuted global.
    let dir = TmpDir::new("swap");
    dcp::save(dir.path(), &to_dist(&synthetic_global(7)), 2).unwrap();
    let p0 = dir.path().join("step_7_shard_0000.tdcp");
    let p1 = dir.path().join("step_7_shard_0001.tdcp");
    let b0 = std::fs::read(&p0).unwrap();
    let b1 = std::fs::read(&p1).unwrap();
    std::fs::write(&p0, &b1).unwrap();
    std::fs::write(&p1, &b0).unwrap();
    assert!(
        matches!(
            dcp::load(dir.path()),
            Err(DcpError::ShardMismatch { field: "rank", .. })
        ),
        "swapped shards not detected"
    );
}

#[test]
fn truncated_shard_is_detected_not_loaded() {
    // External corruption (a shard the live manifest names is truncated) must error cleanly, never
    // load garbage or panic.
    let dir = TmpDir::new("fault_torn");
    let g = synthetic_global(3);
    dcp::save(dir.path(), &to_dist(&g), 2).unwrap();
    let shard = dir.path().join("step_3_shard_0000.tdcp");
    let bytes = std::fs::read(&shard).unwrap();
    std::fs::write(&shard, &bytes[..bytes.len() / 2]).unwrap();
    assert!(
        matches!(dcp::load(dir.path()), Err(DcpError::Truncated { .. })),
        "truncated shard not detected"
    );
}

#[test]
fn missing_shard_is_detected() {
    let dir = TmpDir::new("fault_missing");
    let g = synthetic_global(4);
    dcp::save(dir.path(), &to_dist(&g), 2).unwrap();
    std::fs::remove_file(dir.path().join("step_4_shard_0001.tdcp")).unwrap();
    assert_eq!(dcp::load(dir.path()), Err(DcpError::MissingShard(1)));
}

#[test]
fn corrupt_manifest_is_detected() {
    let dir = TmpDir::new("fault_manifest");
    let g = synthetic_global(1);
    dcp::save(dir.path(), &to_dist(&g), 1).unwrap();
    let mp = dir.path().join("manifest.tdcp");
    let mut b = std::fs::read(&mp).unwrap();
    b[0] ^= 0xFF;
    std::fs::write(&mp, b).unwrap();
    assert_eq!(dcp::load(dir.path()), Err(DcpError::BadMagic));
}
