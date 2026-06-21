//! Collective-correctness gate (ADR 0008 / plan 0014): the thread-simulated `ProcessGroup` runs N
//! logical ranks in N threads and its collectives are deterministic — bit-identical to a
//! single-process reference that folds in the same fixed rank order, regardless of thread scheduling.

use proptest::prelude::*;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use tritium_train::dist::{DistError, ProcessGroup, ReduceOp, SimProcessGroup};

/// Run `f` on each rank in its own thread; collect the per-rank return values (indexed by rank).
fn run_ranks<T, F>(world: usize, f: F) -> Vec<T>
where
    F: Fn(SimProcessGroup) -> T + Sync,
    T: Send,
{
    let pgs = SimProcessGroup::world(world);
    let f = &f;
    thread::scope(|s| {
        let handles: Vec<_> = pgs.into_iter().map(|pg| s.spawn(move || f(pg))).collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

/// Deterministic per-rank vector (distinct per rank), splitmix64-derived.
fn vec_for(rank: usize, n: usize) -> Vec<f32> {
    let mut z = (rank as u64).wrapping_add(0x1234_5678);
    (0..n)
        .map(|_| {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x ^= x >> 31;
            ((x >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
        })
        .collect()
}

/// Single-process reference all-reduce: fold rank order 0..world, element-wise (same order the sim uses).
fn ref_sum(world: usize, n: usize) -> Vec<f32> {
    let mut acc = vec![0.0f32; n];
    for r in 0..world {
        let v = vec_for(r, n);
        for i in 0..n {
            acc[i] += v[i];
        }
    }
    acc
}

#[test]
fn all_reduce_sum_matches_single_process_reference() {
    for world in [1usize, 2, 4, 8] {
        let n = 17;
        let reference = ref_sum(world, n);
        let results = run_ranks(world, move |pg| {
            let mut buf = vec_for(pg.rank(), n);
            pg.all_reduce(&mut buf, ReduceOp::Sum).unwrap();
            buf
        });
        for (r, got) in results.iter().enumerate() {
            assert_eq!(
                got, &reference,
                "rank {r} (world {world}) all_reduce(Sum) mismatch"
            );
        }
    }
}

#[test]
fn all_reduce_avg_is_sum_over_world() {
    let world = 4;
    let n = 11;
    let mut reference = ref_sum(world, n);
    reference.iter_mut().for_each(|x| *x /= world as f32);
    let results = run_ranks(world, move |pg| {
        let mut buf = vec_for(pg.rank(), n);
        pg.all_reduce(&mut buf, ReduceOp::Avg).unwrap();
        buf
    });
    for got in &results {
        assert_eq!(got, &reference);
    }
}

#[test]
fn reduce_scatter_sum_gives_each_rank_its_chunk() {
    let world = 4;
    let chunk = 3;
    let full = world * chunk; // each rank's input length
    let results = run_ranks(world, move |pg| {
        let input = vec_for(pg.rank(), full);
        let mut out = vec![0.0f32; chunk];
        pg.reduce_scatter(&input, &mut out, ReduceOp::Sum).unwrap();
        out
    });
    // Reference: rank r's output[i] = Σ_src input_src[r*chunk + i].
    for r in 0..world {
        let mut expect = vec![0.0f32; chunk];
        for src in 0..world {
            let v = vec_for(src, full);
            for i in 0..chunk {
                expect[i] += v[r * chunk + i];
            }
        }
        assert_eq!(results[r], expect, "reduce_scatter rank {r}");
    }
}

#[test]
fn all_gather_concatenates_in_rank_order() {
    let world = 4;
    let n = 5;
    let results = run_ranks(world, move |pg| {
        let input = vec_for(pg.rank(), n);
        let mut out = vec![0.0f32; world * n];
        pg.all_gather(&input, &mut out).unwrap();
        out
    });
    let mut expect = Vec::new();
    for r in 0..world {
        expect.extend(vec_for(r, n));
    }
    for (r, got) in results.iter().enumerate() {
        assert_eq!(got, &expect, "all_gather rank {r}");
    }
}

#[test]
fn broadcast_sends_root_buffer_to_all() {
    let world = 4;
    let root = 2;
    let n = 6;
    let expect = vec_for(root, n);
    let results = run_ranks(world, move |pg| {
        // Non-root ranks start with their own (different) data; after broadcast all hold root's.
        let mut buf = vec_for(pg.rank(), n);
        pg.broadcast(&mut buf, root).unwrap();
        buf
    });
    for (r, got) in results.iter().enumerate() {
        assert_eq!(got, &expect, "broadcast rank {r}");
    }
}

#[test]
fn all_reduce_is_deterministic_across_schedulings() {
    // Repeat with freshly-spawned threads; the bit pattern must never vary (fixed-order fold).
    let world = 8;
    let n = 23;
    let first = {
        let r = run_ranks(world, move |pg| {
            let mut b = vec_for(pg.rank(), n);
            pg.all_reduce(&mut b, ReduceOp::Sum).unwrap();
            b
        });
        r[0].clone()
    };
    for _ in 0..50 {
        let r = run_ranks(world, move |pg| {
            let mut b = vec_for(pg.rank(), n);
            pg.all_reduce(&mut b, ReduceOp::Sum).unwrap();
            b
        });
        for got in &r {
            assert_eq!(got, &first, "all_reduce result varied across schedulings");
        }
    }
}

#[test]
fn mismatched_sizes_error_not_panic() {
    // world=1: a local size-relation violation returns Err cleanly (no other rank to desync).
    let pg = SimProcessGroup::world(1).pop().unwrap();
    let input = vec![1.0f32; 6];
    let mut bad_out = vec![0.0f32; 4]; // reduce_scatter needs input.len() == world*out.len() = 4
    assert!(matches!(
        pg.reduce_scatter(&input, &mut bad_out, ReduceOp::Sum),
        Err(DistError::LengthMismatch { .. })
    ));
    let mut bad_gather = vec![0.0f32; 5]; // all_gather needs out.len() == world*in.len() = 6
    assert!(matches!(
        pg.all_gather(&input, &mut bad_gather),
        Err(DistError::LengthMismatch { .. })
    ));
    // broadcast root out of range (world=1 ⇒ only root 0 is valid).
    let mut buf = vec![0.0f32; 3];
    assert!(matches!(
        pg.broadcast(&mut buf, 1),
        Err(DistError::InvalidRoot {
            root: 1,
            world_size: 1
        })
    ));
}

proptest! {
    /// For any world∈[1,8] and random per-rank vectors, every rank's all_reduce(Sum) is bit-exact
    /// to the fixed-order single-process reference.
    #[test]
    fn all_reduce_matches_ordered_reference(world in 1usize..=8, n in 1usize..40) {
        let reference = ref_sum(world, n);
        let results = run_ranks(world, move |pg| {
            let mut buf = vec_for(pg.rank(), n);
            pg.all_reduce(&mut buf, ReduceOp::Sum).unwrap();
            buf
        });
        for got in &results {
            prop_assert_eq!(got, &reference);
        }
    }
}

/// THE teeth for the deadlock fix. Two ranks DISAGREE on the reduce_scatter size relation:
/// rank0 has a valid (need == input.len()) call, rank1's is invalid. Under the OLD pre-publish
/// `return Err(...)` code, rank1 returned without ever crossing barrier #1, so rank0 (which
/// reached barrier #1) blocked there forever → process-wide hang. Under the fix both ranks publish
/// first and always reach both barriers, so both return.
///
/// We spawn DETACHED threads (not `thread::scope`, which would itself hang if a rank hangs) and
/// collect results over an mpsc channel with a per-`recv` timeout. A hang is converted into a clean
/// test failure (and a small leaked thread, which is acceptable on regression).
#[test]
fn reduce_scatter_size_disagreement_does_not_deadlock() {
    let world = 2;
    let pgs = SimProcessGroup::world(world);
    let (tx, rx) = mpsc::channel::<(usize, Result<(), DistError>)>();
    for pg in pgs {
        let tx = tx.clone();
        thread::spawn(move || {
            let rank = pg.rank();
            // input.len() == 6 on both ranks. output.len() differs: rank0 -> 3 (need=6, valid),
            // rank1 -> 4 (need=8 != 6, invalid). Both must still return (no hang).
            let input = vec![1.0f32; 6];
            let out_len = if rank == 0 { 3 } else { 4 };
            let mut out = vec![0.0f32; out_len];
            let res = pg.reduce_scatter(&input, &mut out, ReduceOp::Sum);
            // Send may fail only if the receiver already gave up (timeout); ignore that.
            let _ = tx.send((rank, res));
        });
    }
    drop(tx); // so the channel closes once all spawned senders finish

    let mut results: Vec<Option<Result<(), DistError>>> = vec![None; world];
    for _ in 0..world {
        match rx.recv_timeout(Duration::from_secs(10)) {
            Ok((rank, res)) => results[rank] = Some(res),
            Err(_) => panic!("deadlock: rank did not return within timeout"),
        }
    }

    // rank0's call is valid (need == input.len() == 6) -> Ok.
    assert_eq!(
        results[0],
        Some(Ok(())),
        "rank0 reduce_scatter should succeed"
    );
    // rank1's call violates the size relation -> clean Err, not a hang.
    assert!(
        matches!(results[1], Some(Err(DistError::LengthMismatch { .. }))),
        "rank1 reduce_scatter should be LengthMismatch, got {:?}",
        results[1]
    );
}

/// world=2 with each rank invoking a DIFFERENT collective at the same step. The op-tag guard
/// detects the desync after barrier #1; both ranks reach barrier #2 and return a clean
/// `CollectiveMismatch` (no hang). Timeout-guarded with detached threads, as above.
#[test]
fn divergent_collectives_return_clean_mismatch_no_hang() {
    let world = 2;
    let pgs = SimProcessGroup::world(world);
    let (tx, rx) = mpsc::channel::<(usize, Result<(), DistError>)>();
    for pg in pgs {
        let tx = tx.clone();
        thread::spawn(move || {
            let rank = pg.rank();
            let res = if rank == 0 {
                // rank0 invokes all_gather (output.len() == world*input.len()).
                let input = vec![1.0f32; 3];
                let mut out = vec![0.0f32; world * 3];
                pg.all_gather(&input, &mut out)
            } else {
                // rank1 invokes broadcast at the same step — a cross-collective desync.
                let mut buf = vec![2.0f32; 3];
                pg.broadcast(&mut buf, 0)
            };
            let _ = tx.send((rank, res));
        });
    }
    drop(tx);

    let mut results: Vec<Option<Result<(), DistError>>> = vec![None; world];
    for _ in 0..world {
        match rx.recv_timeout(Duration::from_secs(10)) {
            Ok((rank, res)) => results[rank] = Some(res),
            Err(_) => panic!("deadlock: rank did not return within timeout"),
        }
    }

    // Both ranks see a tag disagreement and return CollectiveMismatch (no hang).
    for (rank, res) in results.iter().enumerate() {
        assert!(
            matches!(res, Some(Err(DistError::CollectiveMismatch { .. }))),
            "rank {rank} should be CollectiveMismatch, got {res:?}"
        );
    }
}

/// reduce_scatter with `Avg`: world divides cleanly, so each rank's chunk[i] == (fixed-order sum
/// over ranks of input_src[rank*chunk + i]) / world.
#[test]
fn reduce_scatter_avg_is_sum_over_world() {
    let world = 4;
    let chunk = 3;
    let full = world * chunk;
    let results = run_ranks(world, move |pg| {
        let input = vec_for(pg.rank(), full);
        let mut out = vec![0.0f32; chunk];
        pg.reduce_scatter(&input, &mut out, ReduceOp::Avg).unwrap();
        out
    });
    for r in 0..world {
        let mut expect = vec![0.0f32; chunk];
        for src in 0..world {
            let v = vec_for(src, full);
            for i in 0..chunk {
                expect[i] += v[r * chunk + i];
            }
        }
        // Same fixed-order sum the sim computes, then the same /world the sim applies.
        expect.iter_mut().for_each(|x| *x /= world as f32);
        assert_eq!(results[r], expect, "reduce_scatter(Avg) rank {r}");
    }
}

/// reduce_scatter is deterministic across schedulings: rerun across freshly-spawned threads; the
/// bit pattern must never vary (fixed-order fold).
#[test]
fn reduce_scatter_is_deterministic_across_schedulings() {
    let world = 8;
    let chunk = 3;
    let full = world * chunk;
    let run = || {
        run_ranks(world, move |pg| {
            let input = vec_for(pg.rank(), full);
            let mut out = vec![0.0f32; chunk];
            pg.reduce_scatter(&input, &mut out, ReduceOp::Sum).unwrap();
            out
        })
    };
    let first = run();
    for _ in 0..40 {
        let r = run();
        assert_eq!(r, first, "reduce_scatter result varied across schedulings");
    }
}

/// all_gather is deterministic across schedulings.
#[test]
fn all_gather_is_deterministic_across_schedulings() {
    let world = 8;
    let n = 5;
    let run = || {
        run_ranks(world, move |pg| {
            let input = vec_for(pg.rank(), n);
            let mut out = vec![0.0f32; world * n];
            pg.all_gather(&input, &mut out).unwrap();
            out
        })
    };
    let first = run();
    for _ in 0..40 {
        let r = run();
        assert_eq!(r, first, "all_gather result varied across schedulings");
    }
}

/// broadcast is deterministic across schedulings.
#[test]
fn broadcast_is_deterministic_across_schedulings() {
    let world = 8;
    let root = 3;
    let n = 7;
    let run = || {
        run_ranks(world, move |pg| {
            let mut buf = vec_for(pg.rank(), n);
            pg.broadcast(&mut buf, root).unwrap();
            buf
        })
    };
    let first = run();
    for _ in 0..40 {
        let r = run();
        assert_eq!(r, first, "broadcast result varied across schedulings");
    }
}
