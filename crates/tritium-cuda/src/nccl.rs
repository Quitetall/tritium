//! Real NCCL collective backend (plan 0017, the ≥2-GPU wall) — feature `nccl`.
//!
//! [`NcclProcessGroup`] implements the **same** [`ProcessGroup`](tritium_train::dist::ProcessGroup)
//! trait as the thread-simulated backend (0014), backed by `cudarc::nccl`. The FSDP loop (0015) and
//! distributed checkpoint (0016) therefore run unchanged over either backend — the sim is the CI
//! oracle, this is the real one validated against it on a multi-GPU box.
//!
//! **One rank per GPU.** Each rank opens its own [`CudaContext`] + stream, then joins the group's NCCL
//! communicator via the shared [`NcclId`] (rank 0 creates it; the others reconstruct it from its bytes,
//! shipped over a channel / file / env). `from_rank` is a rendezvous — every rank must call
//! [`NcclProcessGroup::init`] concurrently or it blocks. A collective uploads the host buffer to the
//! device, runs the NCCL op on the rank's stream, synchronizes, and copies the result back — so the
//! trait's host-`[f32]` signature is preserved (the per-call h2d/d2h is immaterial for the
//! correctness gate; a device-resident path is the perf concern of the deferred 2B engine).
//!
//! **Contract.** As with the sim and with MPI/NCCL generally, all ranks must invoke the same collective
//! sequence with consistent sizes; the size-relation checks here mirror the sim's exactly
//! ([`DistError::LengthMismatch`] / [`LengthOverflow`](DistError::LengthOverflow) /
//! [`InvalidRoot`](DistError::InvalidRoot)) so a mis-sized call fails identically on both backends.
//! NCCL ring/tree reductions are **not** bit-identical to the sim's fixed-order CPU fold (float adds
//! reorder), so the wire-correctness gate compares within a tolerance, exactly like the 0015
//! data-parallel parity gate.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, DriverError};
use cudarc::nccl::result::NcclError;
use cudarc::nccl::{Comm, Id, ReduceOp as NcclOp};
use tritium_train::dist::{DistError, ProcessGroup, ReduceOp};

/// The NCCL rendezvous token. Rank 0 creates one with [`NcclId::new`] and ships its [`bytes`](Self::bytes)
/// to every other rank (channel / file / env); each rank passes it to [`NcclProcessGroup::init`].
#[derive(Clone, Copy)]
pub struct NcclId {
    bytes: [core::ffi::c_char; 128],
}

impl core::fmt::Debug for NcclId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NcclId").finish_non_exhaustive()
    }
}

impl NcclId {
    /// Create a fresh unique id — call **once**, on rank 0.
    ///
    /// # Errors
    /// [`DistError::Backend`] if NCCL fails to produce an id.
    pub fn new() -> Result<Self, DistError> {
        let id = Id::new().map_err(nccl_err)?;
        Ok(Self {
            bytes: *id.internal(),
        })
    }

    /// The 128 raw bytes to ship to peer ranks.
    #[must_use]
    pub fn bytes(&self) -> [core::ffi::c_char; 128] {
        self.bytes
    }

    /// Reconstruct an id from the bytes received from rank 0.
    #[must_use]
    pub fn from_bytes(bytes: [core::ffi::c_char; 128]) -> Self {
        Self { bytes }
    }
}

/// One rank's handle to a real NCCL communicator. Implements [`ProcessGroup`] by uploading each host
/// buffer, running the device collective on the rank's stream, and copying the result back.
pub struct NcclProcessGroup {
    comm: Comm,
}

impl core::fmt::Debug for NcclProcessGroup {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NcclProcessGroup")
            .field("rank", &self.comm.rank())
            .field("world_size", &self.comm.world_size())
            .finish()
    }
}

impl NcclProcessGroup {
    /// Initialise this rank's communicator on `device_ordinal`, joining the `world_size`-rank group
    /// identified by `id`. Every rank must call this **concurrently** (NCCL rendezvous blocks until all
    /// `world_size` ranks have joined). For one-GPU-per-rank pass `device_ordinal == rank`.
    ///
    /// # Errors
    /// [`DistError::Backend`] if opening the device or the NCCL rendezvous fails.
    pub fn init(
        device_ordinal: usize,
        rank: usize,
        world_size: usize,
        id: &NcclId,
    ) -> Result<Self, DistError> {
        let ctx = CudaContext::new(device_ordinal).map_err(driver_err)?;
        let stream = ctx.default_stream();
        let comm =
            Comm::from_rank(stream, rank, world_size, Id::uninit(id.bytes)).map_err(nccl_err)?;
        Ok(Self { comm })
    }

    fn stream(&self) -> Arc<CudaStream> {
        self.comm.stream()
    }
}

impl ProcessGroup for NcclProcessGroup {
    fn rank(&self) -> usize {
        self.comm.rank()
    }

    fn world_size(&self) -> usize {
        self.comm.world_size()
    }

    fn all_reduce(&self, buf: &mut [f32], op: ReduceOp) -> Result<(), DistError> {
        let stream = self.stream();
        let send = stream.clone_htod(buf).map_err(driver_err)?;
        let mut recv = stream.alloc_zeros::<f32>(buf.len()).map_err(driver_err)?;
        self.comm
            .all_reduce(&send, &mut recv, &to_nccl(op))
            .map_err(nccl_err)?;
        stream.synchronize().map_err(driver_err)?;
        stream.memcpy_dtoh(&recv, buf).map_err(driver_err)?;
        stream.synchronize().map_err(driver_err)?;
        Ok(())
    }

    fn reduce_scatter(
        &self,
        input: &[f32],
        output: &mut [f32],
        op: ReduceOp,
    ) -> Result<(), DistError> {
        let world = self.world_size();
        let chunk = output.len();
        let need = world
            .checked_mul(chunk)
            .ok_or(DistError::LengthOverflow { world, chunk })?;
        if input.len() != need {
            return Err(DistError::LengthMismatch {
                expected: need,
                got: input.len(),
            });
        }
        let stream = self.stream();
        let send = stream.clone_htod(input).map_err(driver_err)?;
        let mut recv = stream.alloc_zeros::<f32>(chunk).map_err(driver_err)?;
        self.comm
            .reduce_scatter(&send, &mut recv, &to_nccl(op))
            .map_err(nccl_err)?;
        stream.synchronize().map_err(driver_err)?;
        stream.memcpy_dtoh(&recv, output).map_err(driver_err)?;
        stream.synchronize().map_err(driver_err)?;
        Ok(())
    }

    fn all_gather(&self, input: &[f32], output: &mut [f32]) -> Result<(), DistError> {
        let world = self.world_size();
        let n = input.len();
        let need = world
            .checked_mul(n)
            .ok_or(DistError::LengthOverflow { world, chunk: n })?;
        if output.len() != need {
            return Err(DistError::LengthMismatch {
                expected: need,
                got: output.len(),
            });
        }
        let stream = self.stream();
        let send = stream.clone_htod(input).map_err(driver_err)?;
        let mut recv = stream.alloc_zeros::<f32>(need).map_err(driver_err)?;
        self.comm.all_gather(&send, &mut recv).map_err(nccl_err)?;
        stream.synchronize().map_err(driver_err)?;
        stream.memcpy_dtoh(&recv, output).map_err(driver_err)?;
        stream.synchronize().map_err(driver_err)?;
        Ok(())
    }

    fn broadcast(&self, buf: &mut [f32], root: usize) -> Result<(), DistError> {
        let world = self.world_size();
        if root >= world {
            return Err(DistError::InvalidRoot {
                root,
                world_size: world,
            });
        }
        let root_i32 = i32::try_from(root).map_err(|_| DistError::InvalidRoot {
            root,
            world_size: world,
        })?;
        let stream = self.stream();
        let mut dbuf = stream.clone_htod(buf).map_err(driver_err)?;
        self.comm
            .broadcast_in_place(&mut dbuf, root_i32)
            .map_err(nccl_err)?;
        stream.synchronize().map_err(driver_err)?;
        stream.memcpy_dtoh(&dbuf, buf).map_err(driver_err)?;
        stream.synchronize().map_err(driver_err)?;
        Ok(())
    }
}

fn to_nccl(op: ReduceOp) -> NcclOp {
    match op {
        ReduceOp::Sum => NcclOp::Sum,
        ReduceOp::Avg => NcclOp::Avg,
    }
}

fn driver_err(e: DriverError) -> DistError {
    DistError::Backend(format!("cuda: {e}"))
}

fn nccl_err(e: NcclError) -> DistError {
    // NcclError is Debug (it wraps the raw ncclResult_t), not Display.
    DistError::Backend(format!("nccl: {e:?}"))
}

/// The number of visible CUDA devices (0 if the driver can't be reached). The launcher uses this to
/// size the world (one rank per GPU).
#[must_use]
pub fn device_count() -> usize {
    CudaContext::device_count().map(|n| n as usize).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run `f` on each rank in its own thread (rank `r` on device `r`), sharing one NCCL id. Returns
    /// the per-rank results indexed by rank. All ranks rendezvous in `init`.
    fn nccl_world<T, F>(world: usize, f: F) -> Vec<T>
    where
        F: Fn(NcclProcessGroup) -> T + Sync,
        T: Send,
    {
        let id = NcclId::new().expect("nccl id");
        let f = &f;
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..world)
                .map(|rank| {
                    s.spawn(move || {
                        let pg = NcclProcessGroup::init(rank, rank, world, &id).expect("nccl init");
                        f(pg)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        })
    }

    /// world=1 on a single GPU (runs on any CUDA box, incl. the 4090): every collective is the identity
    /// over one rank. Exercises the full upload → collective → download plumbing + the trait wiring +
    /// the NCCL link, with no second GPU.
    #[test]
    fn world1_collectives_smoke() {
        if device_count() == 0 {
            eprintln!("skip world1_collectives_smoke: no CUDA device");
            return;
        }
        let id = NcclId::new().unwrap();
        let pg = NcclProcessGroup::init(0, 0, 1, &id).unwrap();
        assert_eq!(pg.rank(), 0);
        assert_eq!(pg.world_size(), 1);

        let mut a = vec![1.0f32, 2.0, 3.0];
        pg.all_reduce(&mut a, ReduceOp::Sum).unwrap();
        assert_eq!(
            a,
            vec![1.0, 2.0, 3.0],
            "all_reduce Sum world=1 must be identity"
        );

        let mut b = vec![4.0f32, 8.0];
        pg.all_reduce(&mut b, ReduceOp::Avg).unwrap();
        assert_eq!(b, vec![4.0, 8.0], "all_reduce Avg world=1 must be identity");

        let inp = vec![5.0f32, 6.0];
        let mut out = vec![0.0f32; 2];
        pg.all_gather(&inp, &mut out).unwrap();
        assert_eq!(out, inp, "all_gather world=1 must be identity");

        let mut rs = vec![0.0f32; 3];
        pg.reduce_scatter(&[7.0, 8.0, 9.0], &mut rs, ReduceOp::Sum)
            .unwrap();
        assert_eq!(
            rs,
            vec![7.0, 8.0, 9.0],
            "reduce_scatter world=1 must be identity"
        );

        let mut bc = vec![10.0f32, 11.0];
        pg.broadcast(&mut bc, 0).unwrap();
        assert_eq!(
            bc,
            vec![10.0, 11.0],
            "broadcast root=0 world=1 must be identity"
        );
    }

    /// world=1 size-contract errors mirror the sim exactly (no panic).
    #[test]
    fn world1_size_contract_errors() {
        if device_count() == 0 {
            eprintln!("skip world1_size_contract_errors: no CUDA device");
            return;
        }
        let id = NcclId::new().unwrap();
        let pg = NcclProcessGroup::init(0, 0, 1, &id).unwrap();
        // reduce_scatter: input.len() must equal world*output.len() (== output.len() at world=1).
        let mut out = vec![0.0f32; 3];
        assert!(matches!(
            pg.reduce_scatter(&[1.0, 2.0], &mut out, ReduceOp::Sum),
            Err(DistError::LengthMismatch {
                expected: 3,
                got: 2
            })
        ));
        // all_gather: output.len() must equal world*input.len().
        let mut bad = vec![0.0f32; 5];
        assert!(matches!(
            pg.all_gather(&[1.0, 2.0], &mut bad),
            Err(DistError::LengthMismatch {
                expected: 2,
                got: 5
            })
        ));
        // broadcast root out of range.
        let mut buf = vec![0.0f32; 2];
        assert!(matches!(
            pg.broadcast(&mut buf, 1),
            Err(DistError::InvalidRoot {
                root: 1,
                world_size: 1
            })
        ));
    }

    // ── The ≥2-GPU wall gates (plan 0017). Self-skip on a single-GPU box; the real run is on the
    //    rented 2×GPU machine. NCCL reductions reorder float adds, so the comparison to the
    //    single-process reference is within a tolerance (like the 0015 data-parallel gate). ──

    fn vec_for(rank: usize, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (rank as f32 * 10.0) + (i as f32) * 0.25 - 1.0)
            .collect()
    }

    #[test]
    fn nccl_all_reduce_matches_sum_reference_multi_gpu() {
        let world = device_count();
        if world < 2 {
            eprintln!("skip nccl_all_reduce (needs >=2 GPUs, have {world})");
            return;
        }
        let n = 17;
        let results = nccl_world(world, move |pg| {
            let mut buf = vec_for(pg.rank(), n);
            pg.all_reduce(&mut buf, ReduceOp::Sum).unwrap();
            buf
        });
        let mut reference = vec![0.0f32; n];
        for r in 0..world {
            let v = vec_for(r, n);
            for i in 0..n {
                reference[i] += v[i];
            }
        }
        for (r, got) in results.iter().enumerate() {
            for i in 0..n {
                let tol = 1e-3 * reference[i].abs().max(1.0);
                assert!(
                    (got[i] - reference[i]).abs() <= tol,
                    "rank {r}[{i}]: nccl {} vs sum reference {}",
                    got[i],
                    reference[i]
                );
            }
        }
    }

    #[test]
    fn nccl_all_gather_matches_concat_reference_multi_gpu() {
        let world = device_count();
        if world < 2 {
            eprintln!("skip nccl_all_gather (needs >=2 GPUs, have {world})");
            return;
        }
        let n = 5;
        let results = nccl_world(world, move |pg| {
            let inp = vec_for(pg.rank(), n);
            let mut out = vec![0.0f32; world * n];
            pg.all_gather(&inp, &mut out).unwrap();
            out
        });
        // Reference: ranks' inputs concatenated in rank order (all_gather is exact — a copy).
        let mut reference = Vec::with_capacity(world * n);
        for r in 0..world {
            reference.extend(vec_for(r, n));
        }
        for (r, got) in results.iter().enumerate() {
            assert_eq!(got, &reference, "rank {r} all_gather != concat reference");
        }
    }

    #[test]
    fn nccl_broadcast_matches_root_multi_gpu() {
        let world = device_count();
        if world < 2 {
            eprintln!("skip nccl_broadcast (needs >=2 GPUs, have {world})");
            return;
        }
        let n = 9;
        let root = 1.min(world - 1);
        let results = nccl_world(world, move |pg| {
            let mut buf = vec_for(pg.rank(), n);
            pg.broadcast(&mut buf, root).unwrap();
            buf
        });
        let reference = vec_for(root, n); // broadcast is exact — every rank ends with root's buffer.
        for (r, got) in results.iter().enumerate() {
            assert_eq!(
                got, &reference,
                "rank {r} broadcast != root {root}'s buffer"
            );
        }
    }

    // ── 0018: 2-GPU FSDP loss-parity. The SAME tiny-MLP FSDP loop the sim proved correct in 0015,
    //    run over real NCCL across `device_count` GPUs, checked against a single-process reference.
    //    world=1 (any single GPU) is bit-exact (collectives are identities — a local teeth check);
    //    world>=2 (the box) is within a tolerance (NCCL reductions reorder float adds). ──
    use tritium_train::FlatShardPlan;
    use tritium_train::optim::{AdamState, AdamW, Optimizer};
    use tritium_train::tape::Tape;

    const D_IN: usize = 4;
    const H: usize = 6;
    const D_OUT: usize = 3;
    const B: usize = 8;
    const STEPS: u64 = 20;
    const LR: f32 = 0.05;
    const LEAF_LENS: [usize; 2] = [H * D_IN, D_OUT * H];

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

    fn forward_backward(
        leaves: &[Vec<f32>],
        x: &[f32],
        y: &[f32],
        b: usize,
    ) -> (f32, Vec<Vec<f32>>) {
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

    fn reference_curve() -> Vec<f32> {
        let mut leaves = vec![
            seeded(1, H * D_IN, -0.3, 0.3),
            seeded(2, D_OUT * H, -0.3, 0.3),
        ];
        let (x, y) = data();
        let opt = AdamW::new(LR);
        let mut states: Vec<AdamState> = leaves.iter().map(|l| opt.init_state(l.len())).collect();
        let mut curve = Vec::with_capacity(STEPS as usize);
        for step in 0..STEPS {
            let (loss, grads) = forward_backward(&leaves, &x, &y, B);
            for ((leaf, grad), st) in leaves.iter_mut().zip(&grads).zip(&mut states) {
                opt.step(step + 1, leaf, grad, st);
            }
            curve.push(loss);
        }
        curve
    }

    /// One rank's FSDP loop over any `ProcessGroup` (identical body to 0015's sim-validated loop).
    fn fsdp_curve(pg: &dyn ProcessGroup, plan: &FlatShardPlan, init_padded: &[f32]) -> Vec<f32> {
        let rank = pg.rank();
        let world = pg.world_size();
        let (lo, hi) = plan.shard_range(rank);
        let mut shard = init_padded[lo..hi].to_vec();
        let bl = B / world;
        let (x_full, y_full) = data();
        let xs = rank * bl * D_IN;
        let ys = rank * bl * D_OUT;
        let x_local = x_full[xs..xs + bl * D_IN].to_vec();
        let y_local = y_full[ys..ys + bl * D_OUT].to_vec();
        let opt = AdamW::new(LR);
        let mut state = AdamState {
            m: vec![0.0; shard.len()],
            v: vec![0.0; shard.len()],
        };
        let mut curve = Vec::with_capacity(STEPS as usize);
        for step in 0..STEPS {
            let mut full = vec![0.0f32; plan.padded_len()];
            pg.all_gather(&shard, &mut full).expect("all_gather");
            let leaves = plan.unflatten(&full);
            let (loss, grads) = forward_backward(&leaves, &x_local, &y_local, bl);
            let grad_flat = plan.flatten(&grads);
            let mut grad_shard = vec![0.0f32; plan.chunk()];
            pg.reduce_scatter(&grad_flat, &mut grad_shard, ReduceOp::Avg)
                .expect("reduce_scatter");
            opt.step(step + 1, &mut shard, &grad_shard, &mut state);
            let mut lbuf = [loss];
            pg.all_reduce(&mut lbuf, ReduceOp::Avg).expect("all_reduce");
            curve.push(lbuf[0]);
        }
        curve
    }

    fn nccl_fsdp_curve(world: usize) -> Vec<f32> {
        let plan = FlatShardPlan::new(&LEAF_LENS, world);
        let mut init_padded = init_flat();
        init_padded.resize(plan.padded_len(), 0.0);
        let plan_ref = &plan;
        let init_ref = &init_padded;
        // ONE shared id for the whole group (created once; Copy into each rank's closure). Creating it
        // per-rank would give each rank a different id and the NCCL rendezvous would never complete.
        let id = NcclId::new().expect("nccl id");
        let curves: Vec<Vec<f32>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..world)
                .map(|rank| {
                    s.spawn(move || {
                        let pg = NcclProcessGroup::init(rank, rank, world, &id).expect("nccl init");
                        fsdp_curve(&pg, plan_ref, init_ref)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (r, c) in curves.iter().enumerate() {
            assert_eq!(c, &curves[0], "rank {r} FSDP loss curve disagrees");
        }
        curves.into_iter().next().unwrap()
    }

    #[test]
    fn nccl_fsdp_loss_parity() {
        let world = device_count();
        if world == 0 {
            eprintln!("skip nccl_fsdp_loss_parity: no CUDA device");
            return;
        }
        let reference = reference_curve();
        let world = world.min(B);
        if world == 1 {
            // Single GPU: collectives are identities → bit-exact to the reference (harness teeth check).
            assert_eq!(
                nccl_fsdp_curve(1),
                reference,
                "world=1 NCCL FSDP curve must be bit-exact to the reference"
            );
            eprintln!("nccl_fsdp_loss_parity world=1: bit-exact");
            return;
        }
        // >=2 GPUs (the box): within tolerance (NCCL reductions reorder); tighten from observed.
        const ABS_TOL: f32 = 1e-3;
        const REL_TOL: f32 = 1e-2;
        let curve = nccl_fsdp_curve(world);
        assert_eq!(curve.len(), reference.len());
        let mut max_abs = 0.0f32;
        for (t, (&got, &want)) in curve.iter().zip(&reference).enumerate() {
            assert!(got.is_finite(), "world={world} step {t}: non-finite loss");
            let abs = (got - want).abs();
            let rel = abs / want.abs().max(1e-6);
            max_abs = max_abs.max(abs);
            assert!(
                abs <= ABS_TOL || rel <= REL_TOL,
                "world={world} step {t}: nccl {got} vs reference {want} (abs {abs}, rel {rel})"
            );
        }
        eprintln!("nccl_fsdp_loss_parity world={world}: max |Δloss| = {max_abs:e}");
    }
}
