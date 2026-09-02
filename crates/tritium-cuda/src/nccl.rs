//! Real NCCL collective backend (plan 0017, the ≥2-GPU wall) — feature `nccl`.
//!
//! [`NcclProcessGroup`] implements the **same** [`ProcessGroup`](tritium_train::dist::ProcessGroup)
//! trait as the thread-simulated backend (0014), backed by `cudarc::nccl`. The FSDP loop (0015) and
//! distributed checkpoint (0016) therefore run unchanged over either backend — the sim is the CI
//! oracle on CPU; this backend is validated on a multi-GPU box against **single-process references**
//! (the same targets the sim proves), not against a live `SimProcessGroup` instance.
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
//! sequence with consistent sizes. The *intra-rank* size-relation checks here mirror the sim's
//! ([`DistError::LengthMismatch`] / [`LengthOverflow`](DistError::LengthOverflow) /
//! [`InvalidRoot`](DistError::InvalidRoot)) so a locally-mis-sized call fails identically on both
//! backends. NCCL does **not**, however, enforce the *cross-rank* length agreement the sim additionally
//! checks: if ranks pass different-length buffers, the sim returns a clean `LengthMismatch` while NCCL
//! hangs/aborts (an unguardable NCCL contract violation). NCCL ring/tree reductions are also not
//! bit-identical to the sim's fixed-order CPU fold (float adds reorder), so the wire-correctness gate
//! compares within a tolerance, exactly like the 0015 data-parallel parity gate.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DriverError};
use cudarc::nccl::result::NcclError;
use cudarc::nccl::{Comm, Id, ReduceOp as NcclOp};
use tritium_spec::BackendError;
use tritium_train::dist::{DistError, ProcessGroup, ReduceOp};

use crate::CudaBackend;
use crate::train::{
    DeviceTape, DeviceTensor, FinalizedGradientTransform, GradientEmission, GradientLeafBinding,
    GradientStreamReport, HostOffloadTrainer,
};

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

    /// Join a communicator on the CUDA backend's own stream.
    ///
    /// Resident training buffers and subsequent optimizer launches therefore
    /// share one ordered stream: the collective neither stages through host
    /// memory nor needs a host-side synchronization between reduction and use.
    /// Every rank must call this concurrently, as for [`Self::init`].
    ///
    /// # Errors
    /// [`DistError::Backend`] if the NCCL rendezvous fails.
    pub fn init_on_backend(
        backend: &CudaBackend,
        rank: usize,
        world_size: usize,
        id: &NcclId,
    ) -> Result<Self, DistError> {
        let comm = Comm::from_rank(
            backend.nccl_stream(),
            rank,
            world_size,
            Id::uninit(id.bytes),
        )
        .map_err(nccl_err)?;
        Ok(Self { comm })
    }

    /// Synchronize every rank at a campaign lifecycle boundary.
    ///
    /// NCCL has no standalone barrier operation. A one-word all-gather is a
    /// stream-synchronized barrier here and also verifies that the communicator
    /// reports every rank exactly once in rank order.
    ///
    /// # Errors
    /// [`DistError::Backend`] if the collective fails or the communicator rank
    /// map is inconsistent.
    pub fn barrier(&self) -> Result<(), DistError> {
        let local_rank = u64::try_from(self.comm.rank())
            .map_err(|_| DistError::Backend("NCCL rank exceeds u64".into()))?;
        let gathered = self.all_gather_u64(&[local_rank])?;
        for (rank, &reported) in gathered.iter().enumerate() {
            let expected = u64::try_from(rank)
                .map_err(|_| DistError::Backend("NCCL rank exceeds u64".into()))?;
            if reported != expected {
                return Err(DistError::Backend(format!(
                    "NCCL barrier rank map differs: slot {rank} reported rank {reported}"
                )));
            }
        }
        Ok(())
    }

    /// Fail symmetrically when immutable campaign contract words differ.
    ///
    /// The fixed-size length exchange happens before the value exchange. Ranks
    /// may therefore supply independently constructed slices without violating
    /// NCCL's equal-count requirement: a length disagreement returns on every
    /// rank before any variably sized collective is entered. Callers should
    /// encode the plan/input/growth/job identities into stable `u64` words.
    ///
    /// # Errors
    /// [`DistError::LengthMismatch`] if contract lengths differ;
    /// [`DistError::Backend`] if values differ or a collective fails.
    pub fn verify_u64_consensus(&self, local: &[u64]) -> Result<(), DistError> {
        let local_len = u64::try_from(local.len())
            .map_err(|_| DistError::Backend("campaign contract length exceeds u64".into()))?;
        let lengths = self.all_gather_u64(&[local_len])?;
        let canonical_len = lengths
            .first()
            .copied()
            .ok_or_else(|| DistError::Backend("NCCL communicator has no ranks".into()))?;
        if let Some(&different) = lengths.iter().find(|&&length| length != canonical_len) {
            return Err(DistError::LengthMismatch {
                expected: usize::try_from(canonical_len).unwrap_or(usize::MAX),
                got: usize::try_from(different).unwrap_or(usize::MAX),
            });
        }
        if local.is_empty() {
            return Ok(());
        }
        let gathered = self.all_gather_u64(local)?;
        let canonical = gathered.get(..local.len()).ok_or_else(|| {
            DistError::Backend("NCCL contract gather returned a short buffer".into())
        })?;
        if let Some((rank, _)) = gathered
            .chunks_exact(local.len())
            .enumerate()
            .find(|(_, contract)| *contract != canonical)
        {
            return Err(DistError::Backend(format!(
                "immutable campaign contract differs on NCCL rank {rank}"
            )));
        }
        Ok(())
    }

    /// Enqueue an in-place all-reduce over a resident f32 buffer.
    ///
    /// `logical_len` must equal the allocation length. Making that contract
    /// explicit prevents an oversized reusable arena from silently reducing
    /// unrelated trailing values. All ranks must provide the same length; as
    /// with the host NCCL path, cross-rank disagreement is an NCCL contract
    /// violation that cannot be checked locally without another collective.
    ///
    /// # Errors
    /// [`DistError::LengthMismatch`] if `logical_len` differs from the buffer;
    /// [`DistError::Backend`] if the buffer belongs to another CUDA context or
    /// NCCL rejects the operation.
    pub fn all_reduce_f32_in_place(
        &self,
        buffer: &mut CudaSlice<f32>,
        logical_len: usize,
        op: ReduceOp,
    ) -> Result<(), DistError> {
        self.validate_device_buffer(buffer, logical_len)?;
        if logical_len == 0 {
            return Ok(());
        }
        self.comm
            .all_reduce_in_place(buffer, &to_nccl(op))
            .map_err(nccl_err)?;
        Ok(())
    }

    /// Enqueue an in-place broadcast over a resident f32 buffer.
    ///
    /// # Errors
    /// [`DistError::InvalidRoot`] for a root outside the group;
    /// [`DistError::LengthMismatch`] if `logical_len` differs from the buffer;
    /// [`DistError::Backend`] for context or NCCL failures.
    pub fn broadcast_f32_in_place(
        &self,
        buffer: &mut CudaSlice<f32>,
        logical_len: usize,
        root: usize,
    ) -> Result<(), DistError> {
        self.broadcast_device_in_place(buffer, logical_len, root)
    }

    /// Enqueue an in-place broadcast over resident packed bytes.
    ///
    /// This is the plane/metadata companion to [`Self::broadcast_f32_in_place`].
    ///
    /// # Errors
    /// Same contract as [`Self::broadcast_f32_in_place`].
    pub fn broadcast_u8_in_place(
        &self,
        buffer: &mut CudaSlice<u8>,
        logical_len: usize,
        root: usize,
    ) -> Result<(), DistError> {
        self.broadcast_device_in_place(buffer, logical_len, root)
    }

    /// Run a streamed resident backward whose finalized leaf gradients are
    /// averaged across ranks before host-offloaded AdamW consumes them.
    ///
    /// A fixed-size validity/count exchange followed by an exact manifest
    /// exchange happens before backward. Every rank therefore either enters the
    /// same sequence of equal-length collectives or fails before the first one;
    /// local shape/configuration drift cannot silently hang its peers.
    ///
    /// # Errors
    /// Returns [`BackendError`] for invalid local inputs, cross-rank manifest
    /// disagreement, CUDA/NCCL failure, or optimizer failure.
    #[allow(clippy::too_many_arguments)]
    pub fn xent_backward_into<'backend, 'leaf>(
        &self,
        tape: DeviceTape<'backend, 'leaf>,
        logits: usize,
        target: &DeviceTensor,
        rows: usize,
        cols: usize,
        bindings: &[GradientLeafBinding],
        trainer: &mut HostOffloadTrainer<'backend>,
        step: u64,
    ) -> Result<GradientStreamReport, BackendError> {
        let optimizer_manifest = trainer.distributed_optimizer_manifest();
        let local_manifest = tape
            .xent_gradient_stream_manifest(logits, target, rows, cols, bindings, trainer, step)
            .and_then(|manifest| {
                self.validate_device_buffer(target.resident_buffer(), target.len())
                    .map_err(backend_dist_err)?;
                Ok(manifest)
            });
        self.preflight_gradient_manifest(
            local_manifest.as_deref(),
            rows,
            cols,
            step,
            &optimizer_manifest,
        )?;
        let manifest = local_manifest?;
        let mut transform = DistributedGradientTransform {
            group: self,
            manifest: &manifest,
            next: 0,
        };
        let report = tape.xent_backward_into_with_transform(
            logits,
            target,
            rows,
            cols,
            bindings,
            trainer,
            step,
            &mut transform,
        )?;
        transform.finish()?;
        Ok(report)
    }

    fn preflight_gradient_manifest(
        &self,
        local: Result<&[GradientEmission], &BackendError>,
        rows: usize,
        cols: usize,
        step: u64,
        optimizer_manifest: &[u64],
    ) -> Result<(), BackendError> {
        let rows = u64::try_from(rows)
            .map_err(|_| BackendError::InvalidInput("local batch row count exceeds u64".into()))?;
        let cols = u64::try_from(cols).map_err(|_| {
            BackendError::InvalidInput("softmax-xent column count exceeds u64".into())
        })?;
        let local_count = match local {
            Ok(manifest) => u64::try_from(manifest.len()).map_err(|_| {
                BackendError::InvalidInput("gradient manifest length exceeds u64".into())
            })?,
            Err(_) => 0,
        };
        let header = [u64::from(local.is_ok()), local_count, rows, cols, step];
        let headers = self.all_gather_u64(&header).map_err(backend_dist_err)?;
        let first = [headers[0], headers[1], headers[2], headers[3], headers[4]];
        if headers.as_chunks::<5>().0.iter().any(|rank| rank[0] == 0) {
            return match local {
                Err(error) => Err(BackendError::InvalidInput(format!(
                    "local gradient-stream preflight failed: {error}"
                ))),
                Ok(_) => Err(BackendError::InvalidInput(
                    "a peer rejected its gradient-stream inputs before backward".into(),
                )),
            };
        }
        if headers
            .as_chunks::<5>()
            .0
            .iter()
            .any(|rank| rank[1] != first[1])
        {
            return Err(BackendError::InvalidInput(
                "gradient manifest counts differ across NCCL ranks".into(),
            ));
        }
        if headers
            .as_chunks::<5>()
            .0
            .iter()
            .any(|rank| rank[2] != first[2])
        {
            return Err(BackendError::InvalidInput(
                "local batch row counts differ across NCCL ranks; streamed Avg requires equal batches"
                    .into(),
            ));
        }
        if headers
            .as_chunks::<5>()
            .0
            .iter()
            .any(|rank| rank[3] != first[3])
        {
            return Err(BackendError::InvalidInput(
                "softmax-xent column counts differ across NCCL ranks".into(),
            ));
        }
        if headers
            .as_chunks::<5>()
            .0
            .iter()
            .any(|rank| rank[4] != first[4])
        {
            return Err(BackendError::InvalidInput(
                "optimizer steps differ across NCCL ranks".into(),
            ));
        }

        let manifest = local.expect("all preflight validity headers were successful");
        let mut encoded = Vec::with_capacity(manifest.len().saturating_mul(4));
        for emission in manifest {
            for value in [
                emission.sequence,
                emission.leaf_id,
                emission.parameter_index,
                emission.elements,
            ] {
                encoded.push(u64::try_from(value).map_err(|_| {
                    BackendError::InvalidInput("gradient manifest value exceeds u64".into())
                })?);
            }
        }
        encoded.extend_from_slice(optimizer_manifest);
        if encoded.is_empty() {
            return Ok(());
        }
        let gathered = self.all_gather_u64(&encoded).map_err(backend_dist_err)?;
        if gathered
            .chunks_exact(encoded.len())
            .any(|rank| rank != encoded)
        {
            return Err(BackendError::InvalidInput(
                "gradient stream manifests differ across NCCL ranks".into(),
            ));
        }
        Ok(())
    }

    /// Gather an equal-length u64 record from every rank in rank order.
    ///
    /// Campaign orchestration uses this for bounded control-plane evidence;
    /// tensor gradients continue through the resident f32 collectives.
    pub fn all_gather_u64(&self, input: &[u64]) -> Result<Vec<u64>, DistError> {
        let world = self.comm.world_size();
        let output_len = world
            .checked_mul(input.len())
            .ok_or(DistError::LengthOverflow {
                world,
                chunk: input.len(),
            })?;
        let stream = self.stream();
        let send = stream.clone_htod(input).map_err(driver_err)?;
        let mut recv = stream.alloc_zeros::<u64>(output_len).map_err(driver_err)?;
        self.comm.all_gather(&send, &mut recv).map_err(nccl_err)?;
        stream.synchronize().map_err(driver_err)?;
        let mut output = vec![0u64; output_len];
        stream.memcpy_dtoh(&recv, &mut output).map_err(driver_err)?;
        stream.synchronize().map_err(driver_err)?;
        Ok(output)
    }

    fn broadcast_device_in_place<T: cudarc::nccl::NcclType>(
        &self,
        buffer: &mut CudaSlice<T>,
        logical_len: usize,
        root: usize,
    ) -> Result<(), DistError> {
        let root_i32 = self.validate_root(root)?;
        self.validate_device_buffer(buffer, logical_len)?;
        if logical_len == 0 {
            return Ok(());
        }
        self.comm
            .broadcast_in_place(buffer, root_i32)
            .map_err(nccl_err)?;
        Ok(())
    }

    fn validate_device_buffer<T>(
        &self,
        buffer: &CudaSlice<T>,
        logical_len: usize,
    ) -> Result<(), DistError> {
        if buffer.len() != logical_len {
            return Err(DistError::LengthMismatch {
                expected: logical_len,
                got: buffer.len(),
            });
        }
        if self.comm.context() != buffer.context() {
            return Err(DistError::Backend(
                "device collective buffer belongs to a different CUDA context".into(),
            ));
        }
        Ok(())
    }

    fn validate_root(&self, root: usize) -> Result<i32, DistError> {
        let world = self.comm.world_size();
        if root >= world {
            return Err(DistError::InvalidRoot {
                root,
                world_size: world,
            });
        }
        i32::try_from(root).map_err(|_| DistError::InvalidRoot {
            root,
            world_size: world,
        })
    }

    fn stream(&self) -> Arc<CudaStream> {
        self.comm.stream()
    }
}

struct DistributedGradientTransform<'group, 'manifest> {
    group: &'group NcclProcessGroup,
    manifest: &'manifest [GradientEmission],
    next: usize,
}

impl DistributedGradientTransform<'_, '_> {
    fn finish(&self) -> Result<(), BackendError> {
        if self.next == self.manifest.len() {
            Ok(())
        } else {
            Err(BackendError::InvalidInput(format!(
                "distributed gradient stream emitted {} of {} manifest entries",
                self.next,
                self.manifest.len()
            )))
        }
    }
}

impl FinalizedGradientTransform for DistributedGradientTransform<'_, '_> {
    fn transform(
        &mut self,
        emission: GradientEmission,
        gradient: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        let expected = self.manifest.get(self.next).ok_or_else(|| {
            BackendError::InvalidInput(
                "distributed gradient stream emitted more entries than its manifest".into(),
            )
        })?;
        if &emission != expected {
            return Err(BackendError::InvalidInput(format!(
                "distributed gradient emission {emission:?} differs from manifest {expected:?}"
            )));
        }
        self.group
            .all_reduce_f32_in_place(gradient, emission.elements, ReduceOp::Avg)
            .map_err(backend_dist_err)?;
        self.next += 1;
        Ok(())
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

fn backend_dist_err(error: DistError) -> BackendError {
    BackendError::Backend(format!("distributed gradient: {error}"))
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
    use crate::train::DeviceTrainParam;

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

    #[test]
    fn world1_campaign_consensus_and_barrier_smoke() {
        if device_count() == 0 {
            eprintln!("skip world1_campaign_consensus_and_barrier_smoke: no CUDA device");
            return;
        }
        let id = NcclId::new().expect("nccl id");
        let pg = NcclProcessGroup::init(0, 0, 1, &id).expect("nccl init");

        pg.verify_u64_consensus(&[0xfeed_beef, 17, 99])
            .expect("world-one contract consensus");
        pg.verify_u64_consensus(&[])
            .expect("empty contract consensus");
        pg.barrier().expect("world-one barrier");
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

    /// The resident seam must stay on the backend's stream/context and must not
    /// round-trip through host memory. World size one gives an exact identity
    /// oracle while exercising the real NCCL device-buffer call.
    #[test]
    fn world1_device_all_reduce_is_local_identity() {
        if device_count() == 0 {
            eprintln!("skip world1_device_all_reduce_is_local_identity: no CUDA device");
            return;
        }
        let backend = CudaBackend::new(0).expect("cuda backend");
        let id = NcclId::new().expect("nccl id");
        let pg = NcclProcessGroup::init_on_backend(&backend, 0, 1, &id).expect("nccl init");
        let expected = vec![-3.5f32, 0.0, 2.25, 17.0];
        let mut device = backend.dev_upload(&expected).expect("upload");

        pg.all_reduce_f32_in_place(&mut device, expected.len(), ReduceOp::Sum)
            .expect("resident all reduce");

        let mut got = vec![0.0f32; expected.len()];
        backend.dev_download(&device, &mut got).expect("download");
        assert_eq!(got, expected);
    }

    #[test]
    fn world1_streamed_trainer_reduces_each_finalized_gradient_before_adam() {
        if device_count() == 0 {
            eprintln!(
                "skip world1_streamed_trainer_reduces_each_finalized_gradient_before_adam: no CUDA device"
            );
            return;
        }
        use tritium_train::ops::{loss, matmul};
        use tritium_train::optim::{AdamW, Optimizer};

        let backend = CudaBackend::new(0).expect("cuda backend");
        let id = NcclId::new().expect("nccl id");
        let pg = NcclProcessGroup::init_on_backend(&backend, 0, 1, &id).expect("nccl init");
        let (batch, rows, cols) = (2usize, 2usize, 3usize);
        let input = vec![0.5f32, -1.0, 0.25, -0.75, 0.4, 1.2];
        let target = vec![1.0f32, 0.0, 0.0, 1.0];
        let master = vec![0.2f32, -0.3, 0.7, -0.4, 0.8, 0.1];
        let optimizer = AdamW {
            lr: 0.03,
            beta1: 0.8,
            beta2: 0.9,
            eps: 0.2,
            weight_decay: 0.01,
        };

        let logits = matmul::forward(&input, &master, &vec![1.0; rows], batch, rows, cols);
        let grad_logits = loss::softmax_xent_vjp(&logits, &target, batch, rows, &[1.0]).remove(0);
        let grad = matmul::vjp(
            &input,
            &master,
            &vec![1.0; rows],
            batch,
            rows,
            cols,
            &grad_logits,
        )
        .remove(1);
        let mut expected = master.clone();
        let mut state = optimizer.init_state(expected.len());
        optimizer.step(1, &mut expected, &grad, &mut state);

        let mut trainer = HostOffloadTrainer::new(
            &backend,
            &[DeviceTrainParam {
                master: &master,
                rows,
                cols,
                salt_planes: 1,
                optimizer,
            }],
        )
        .expect("host trainer");
        let target = DeviceTensor::upload(&backend, &target).expect("target upload");
        let mut tape = DeviceTape::new(&backend, rows).expect("device tape");
        let input_leaf = tape.leaf(&input).expect("input leaf");
        let weight_leaf = tape.leaf(&master).expect("weight leaf");
        let logits = tape
            .matmul(input_leaf, weight_leaf, batch, rows, cols)
            .expect("matmul");
        let report = pg
            .xent_backward_into(
                tape,
                logits,
                &target,
                batch,
                rows,
                &[GradientLeafBinding {
                    leaf_id: weight_leaf,
                    parameter_index: 0,
                }],
                &mut trainer,
                1,
            )
            .expect("distributed streamed backward");

        assert_eq!(report.emissions.len(), 1);
        assert_eq!(trainer.completed_step(), 1);
        for (&actual, &expected) in trainer.master(0).unwrap().iter().zip(&expected) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn world1_device_broadcasts_are_local_identity() {
        if device_count() == 0 {
            eprintln!("skip world1_device_broadcasts_are_local_identity: no CUDA device");
            return;
        }
        let backend = CudaBackend::new(0).expect("cuda backend");
        let id = NcclId::new().expect("nccl id");
        let pg = NcclProcessGroup::init_on_backend(&backend, 0, 1, &id).expect("nccl init");

        let f32_expected = vec![1.25f32, -9.0, 4.5];
        let mut f32_device = backend.dev_upload(&f32_expected).expect("upload f32");
        pg.broadcast_f32_in_place(&mut f32_device, f32_expected.len(), 0)
            .expect("resident f32 broadcast");
        let mut f32_got = vec![0.0f32; f32_expected.len()];
        backend
            .dev_download(&f32_device, &mut f32_got)
            .expect("download f32");
        assert_eq!(f32_got, f32_expected);

        let u8_expected = vec![0u8, 1, 127, 128, 255];
        let stream = pg.stream();
        let mut u8_device = stream.clone_htod(&u8_expected).expect("upload u8");
        pg.broadcast_u8_in_place(&mut u8_device, u8_expected.len(), 0)
            .expect("resident u8 broadcast");
        let mut u8_got = vec![0u8; u8_expected.len()];
        stream
            .memcpy_dtoh(&u8_device, &mut u8_got)
            .expect("download u8");
        assert_eq!(u8_got, u8_expected);
    }

    #[test]
    fn device_collective_rejects_logical_length_mismatch() {
        if device_count() == 0 {
            eprintln!("skip device_collective_rejects_logical_length_mismatch: no CUDA device");
            return;
        }
        let backend = CudaBackend::new(0).expect("cuda backend");
        let id = NcclId::new().expect("nccl id");
        let pg = NcclProcessGroup::init_on_backend(&backend, 0, 1, &id).expect("nccl init");
        let mut device = backend.dev_upload(&[1.0f32, 2.0, 3.0]).expect("upload");

        assert!(matches!(
            pg.all_reduce_f32_in_place(&mut device, 2, ReduceOp::Sum),
            Err(DistError::LengthMismatch {
                expected: 2,
                got: 3
            })
        ));
    }

    #[test]
    fn device_collective_rejects_foreign_context() {
        if device_count() < 2 {
            eprintln!("skip device_collective_rejects_foreign_context: needs >=2 GPUs");
            return;
        }
        let backend = CudaBackend::new(0).expect("cuda backend 0");
        let foreign = CudaBackend::new(1).expect("cuda backend 1");
        let id = NcclId::new().expect("nccl id");
        let pg = NcclProcessGroup::init_on_backend(&backend, 0, 1, &id).expect("nccl init");
        let mut device = foreign.dev_upload(&[1.0f32, 2.0]).expect("upload");

        assert!(matches!(
            pg.all_reduce_f32_in_place(&mut device, 2, ReduceOp::Sum),
            Err(DistError::Backend(message))
                if message.contains("different CUDA context")
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

    /// The largest divisor of `b` that is `<= cap` (and `>= 1`). For B=8: cap 1→1, 2→2, 3→2, 4→4,
    /// 5..7→4, ≥8→8 — so each rank always gets an equal, complete batch slice.
    fn largest_divisor_le(b: usize, cap: usize) -> usize {
        (1..=cap.min(b))
            .rev()
            .find(|d| b.is_multiple_of(*d))
            .unwrap_or(1)
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
        // Cover both the rank-0 root and a non-zero root (the in-place path is the same; this guards a
        // regression that only manifests at one root).
        for root in [0usize, world - 1] {
            let results = nccl_world(world, move |pg| {
                let mut buf = vec_for(pg.rank(), n);
                pg.broadcast(&mut buf, root).unwrap();
                buf
            });
            let reference = vec_for(root, n); // exact — every rank ends with root's buffer.
            for (r, got) in results.iter().enumerate() {
                assert_eq!(
                    got, &reference,
                    "rank {r} broadcast != root {root}'s buffer"
                );
            }
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

    /// The scale-sensitive distributed gate: average the per-rank gradients
    /// while they are resident and compare the reduced values directly with a
    /// full-batch CPU reference. Unlike an AdamW loss curve, this catches Sum
    /// accidentally substituted for Avg.
    #[test]
    fn device_all_reduce_gradient_matches_full_batch_reference_multi_gpu() {
        let visible = device_count();
        if visible < 2 {
            eprintln!(
                "skip device_all_reduce_gradient_matches_full_batch_reference_multi_gpu: \
                 needs >=2 GPUs, have {visible}"
            );
            return;
        }
        let world = largest_divisor_le(B, visible);
        let leaves = vec![
            seeded(1, H * D_IN, -0.3, 0.3),
            seeded(2, D_OUT * H, -0.3, 0.3),
        ];
        let (x, y) = data();
        let (_, full_grad_parts) = forward_backward(&leaves, &x, &y, B);
        let full_grad: Vec<f32> = full_grad_parts.into_iter().flatten().collect();
        let id = NcclId::new().expect("nccl id");

        let reduced: Vec<Vec<f32>> = std::thread::scope(|scope| {
            let leaves = &leaves;
            let x = &x;
            let y = &y;
            let handles: Vec<_> = (0..world)
                .map(|rank| {
                    scope.spawn(move || {
                        let backend = CudaBackend::new(rank).expect("cuda backend");
                        let pg = NcclProcessGroup::init_on_backend(&backend, rank, world, &id)
                            .expect("nccl init");
                        let local_batch = B / world;
                        let x_start = rank * local_batch * D_IN;
                        let y_start = rank * local_batch * D_OUT;
                        let (_, grad_parts) = forward_backward(
                            leaves,
                            &x[x_start..x_start + local_batch * D_IN],
                            &y[y_start..y_start + local_batch * D_OUT],
                            local_batch,
                        );
                        let local_grad: Vec<f32> = grad_parts.into_iter().flatten().collect();
                        let grad_len = local_grad.len();
                        let mut device = backend.dev_upload(&local_grad).expect("upload gradient");
                        pg.all_reduce_f32_in_place(&mut device, grad_len, ReduceOp::Avg)
                            .expect("resident gradient all reduce");
                        let mut host = vec![0.0f32; grad_len];
                        backend
                            .dev_download(&device, &mut host)
                            .expect("download gradient");
                        host
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("rank thread"))
                .collect()
        });

        for (rank, got) in reduced.iter().enumerate() {
            assert_eq!(got.len(), full_grad.len());
            for (index, (&actual, &expected)) in got.iter().zip(&full_grad).enumerate() {
                let abs = (actual - expected).abs();
                let rel = abs / expected.abs().max(1e-6);
                assert!(
                    abs <= 1e-5 || rel <= 1e-4,
                    "rank {rank} gradient[{index}]: reduced {actual} vs full-batch {expected} \
                     (abs {abs}, rel {rel})"
                );
            }
        }
    }

    /// End-to-end trainer-path gate: each rank builds its local device tape,
    /// the stream averages finalized gradients in manifest order, and every
    /// host-offloaded Adam state lands on the full-batch CPU reference.
    #[test]
    fn streamed_trainer_matches_full_batch_reference_multi_gpu() {
        let visible = device_count();
        if visible < 2 {
            eprintln!(
                "skip streamed_trainer_matches_full_batch_reference_multi_gpu: \
                 needs >=2 GPUs, have {visible}"
            );
            return;
        }
        use tritium_train::ops::{loss, matmul};

        let world = largest_divisor_le(B, visible);
        let input = seeded(0x27A0, B * D_IN, -1.0, 1.0);
        let mut target = vec![0.0f32; B * D_OUT];
        for row in 0..B {
            target[row * D_OUT + row % D_OUT] = 1.0;
        }
        let masters = vec![
            seeded(0x27A1, H * D_IN, -0.4, 0.4),
            seeded(0x27A2, H * H, -0.4, 0.4),
            seeded(0x27A3, D_OUT * H, -0.4, 0.4),
        ];
        let optimizer = AdamW {
            lr: 0.03,
            beta1: 0.8,
            beta2: 0.9,
            eps: 0.2,
            weight_decay: 0.01,
        };
        let hidden_1 = matmul::forward(&input, &masters[0], &[1.0; H], B, H, D_IN);
        let hidden_2 = matmul::forward(&hidden_1, &masters[1], &[1.0; H], B, H, H);
        let logits = matmul::forward(&hidden_2, &masters[2], &[1.0; D_OUT], B, D_OUT, H);
        let grad_logits = loss::softmax_xent_vjp(&logits, &target, B, D_OUT, &[1.0]).remove(0);
        let grad_3 = matmul::vjp(
            &hidden_2,
            &masters[2],
            &[1.0; D_OUT],
            B,
            D_OUT,
            H,
            &grad_logits,
        );
        let grad_2 = matmul::vjp(&hidden_1, &masters[1], &[1.0; H], B, H, H, &grad_3[0]);
        let grad_1 = matmul::vjp(&input, &masters[0], &[1.0; H], B, H, D_IN, &grad_2[0]);
        let gradients = [&grad_1[1], &grad_2[1], &grad_3[1]];
        let mut expected = masters.clone();
        let mut expected_states: Vec<_> = expected
            .iter()
            .map(|master| optimizer.init_state(master.len()))
            .collect();
        for ((master, state), gradient) in
            expected.iter_mut().zip(&mut expected_states).zip(gradients)
        {
            optimizer.step(1, master, gradient, state);
        }
        let id = NcclId::new().expect("nccl id");

        let rank_masters: Vec<Vec<Vec<f32>>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..world)
                .map(|rank| {
                    let input = &input;
                    let target = &target;
                    let masters = &masters;
                    scope.spawn(move || {
                        let backend = CudaBackend::new(rank).expect("cuda backend");
                        let pg = NcclProcessGroup::init_on_backend(&backend, rank, world, &id)
                            .expect("nccl init");
                        let local_batch = B / world;
                        let input_start = rank * local_batch * D_IN;
                        let target_start = rank * local_batch * D_OUT;
                        let local_input = &input[input_start..input_start + local_batch * D_IN];
                        let local_target =
                            &target[target_start..target_start + local_batch * D_OUT];
                        let specs = [
                            DeviceTrainParam {
                                master: &masters[0],
                                rows: H,
                                cols: D_IN,
                                salt_planes: 1,
                                optimizer,
                            },
                            DeviceTrainParam {
                                master: &masters[1],
                                rows: H,
                                cols: H,
                                salt_planes: 1,
                                optimizer,
                            },
                            DeviceTrainParam {
                                master: &masters[2],
                                rows: D_OUT,
                                cols: H,
                                salt_planes: 1,
                                optimizer,
                            },
                        ];
                        let mut trainer =
                            HostOffloadTrainer::new(&backend, &specs).expect("host trainer");
                        let target =
                            DeviceTensor::upload(&backend, local_target).expect("target upload");
                        let mut tape =
                            DeviceTape::new(&backend, H.max(D_OUT)).expect("device training tape");
                        let input_leaf = tape.leaf(local_input).expect("input leaf");
                        let weight_1 = tape.leaf(&masters[0]).expect("weight 1");
                        let weight_2 = tape.leaf(&masters[1]).expect("weight 2");
                        let weight_3 = tape.leaf(&masters[2]).expect("weight 3");
                        let hidden_1 = tape
                            .matmul(input_leaf, weight_1, local_batch, H, D_IN)
                            .expect("matmul 1");
                        let hidden_2 = tape
                            .matmul(hidden_1, weight_2, local_batch, H, H)
                            .expect("matmul 2");
                        let logits = tape
                            .matmul(hidden_2, weight_3, local_batch, D_OUT, H)
                            .expect("matmul 3");
                        let bindings = [weight_1, weight_2, weight_3]
                            .into_iter()
                            .enumerate()
                            .map(|(parameter_index, leaf_id)| GradientLeafBinding {
                                leaf_id,
                                parameter_index,
                            })
                            .collect::<Vec<_>>();
                        let report = pg
                            .xent_backward_into(
                                tape,
                                logits,
                                &target,
                                local_batch,
                                D_OUT,
                                &bindings,
                                &mut trainer,
                                1,
                            )
                            .expect("distributed streamed backward");
                        assert_eq!(
                            report
                                .emissions
                                .iter()
                                .map(|emission| emission.parameter_index)
                                .collect::<Vec<_>>(),
                            vec![2, 1, 0]
                        );
                        (0..trainer.len())
                            .map(|index| trainer.master(index).unwrap().to_vec())
                            .collect()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("rank thread"))
                .collect()
        });

        for (rank, actual_masters) in rank_masters.iter().enumerate() {
            for (parameter, (actual, expected)) in actual_masters.iter().zip(&expected).enumerate()
            {
                for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
                    let abs = (actual - expected).abs();
                    let rel = abs / expected.abs().max(1e-6);
                    assert!(
                        abs <= 1e-5 || rel <= 1e-4,
                        "rank {rank} parameter {parameter} master[{index}]: distributed {actual} \
                         vs CPU {expected} (abs {abs}, rel {rel})"
                    );
                }
            }
        }
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
        // Use the largest divisor of B that is <= the GPU count, so every rank gets an EQUAL, complete
        // slice of the batch. `Avg`-reduced grads equal the full-batch grad only for equal shards — a
        // non-divisor world (e.g. 3 or 6 GPUs, common rental SKUs) would silently drop the tail rows
        // and diverge from the full-batch reference for a NON-NCCL reason (a phantom failure that burns
        // GPU hours). world∈{1,2,4,8} are the divisors of 8 that run.
        let world = largest_divisor_le(B, world);
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
        // >=2 GPUs (the box): within tolerance. NOTE this is an end-to-end *convergence* check, not the
        // wrong-reduce-op gate — AdamW's update is scale-invariant in the gradient (the 0015 lesson), so
        // a loss curve cannot reliably catch a `world`×-scaled reduction. The reduce-op TEETH live in
        // `nccl_all_reduce_matches_sum_reference_multi_gpu` (compares reduced *values* directly). This
        // bound is a placeholder; on the box, read the printed `max |Δloss|` and tighten ABS/REL to ~10×
        // it (and AND them) before tagging v0.60.0 — do not tag on the first loose-band green.
        const ABS_TOL: f32 = 1e-3;
        const REL_TOL: f32 = 1e-3;
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
