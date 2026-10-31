//! # tritium-cpu
//!
//! The x86-64 CPU execution backend. A runtime-dispatched AVX2 kernel with a
//! scalar fallback (the scalar path delegates to
//! [`tritium_core::reference_mpgemm`], so it is correct by construction). The
//! backend self-registers with `tritium-runtime` through the `BACKENDS` `linkme`
//! distributed slice, so linking this crate into a binary makes a `"cpu"` backend
//! appear in the runtime registry with no central edit.
//!
//! ## Path
//!
//! 1. [`CpuBackend::upload_weights`] validates the packed byte length against the
//!    `[N, K]` shape and the format's block size, then stores the bytes verbatim
//!    in a [`CpuBuffer`].
//! 2. [`CpuBackend::mpgemm`] downcasts the buffer, unpacks the `[N, K]` weights to
//!    `Vec<Trit>` with the matching `tritium-format` row unpacker, and runs the
//!    contraction `out[m, n] = scale[n] · Σ_k act[m, k] · trit[n, k]`.
//!
//! The packed block scale is fixed to `1.0` by the host packer, so the unpacked
//! trits are the raw weights; the per-output-channel `scales` are the only thing
//! applied to the contraction (matching the testkit's packing-vs-scaling split).
//!
//! ## Kernels
//!
//! Several kernels back [`CpuBackend::mpgemm`], chosen at runtime by
//! [`kernel::dispatch_mpgemm`]. On x86-64 it prefers the widest the host can
//! execute — AVX-512 (`avx512f`+`bw`+`vl`), then AVX2 (`avx2`), else the scalar
//! reference; on aarch64 the NEON kernel. The three **per-element** SIMD kernels
//! (AVX2, AVX-512, NEON) fold their SIMD-decoded signed contributions
//! sequentially in `f32` k-order, reproducing the scalar reference accumulation
//! **bit-for-bit** — comfortably inside the `1e-4` relative tolerance of ADR
//! 0002 (in fact exact). The scalar reference is kept as the terminal fallback
//! so cross-host output stays byte-reproducible. A portable T-MAC lookup-table
//! kernel ([`simd::lut`]) is implemented and unit-tested but not yet on the
//! dispatch path — its production form is the per-ISA SIMD gather (deferred). The
//! independent `M` rows are spread across `rayon`'s thread pool without changing
//! per-row arithmetic, so results are deterministic regardless of thread count.
//!
//! ## Unified semantic tensors
//!
//! [`CpuBackend::upload_tensor`], [`CpuBackend::matmul`] and
//! [`CpuBackend::embed_rows`] accept additive and dense semantic tensors. This
//! path retains canonical D2/B3/S34 plane bytes and exact f16/f32 scales, or dense
//! f32 values. Additive matmul executes add/subtract/skip directly over packed
//! coefficients, with independent output rows parallelized. Basis transforms
//! use the shared core; gather applies the inverse transform. Buffer byte counts
//! include owned packed coefficients and stored-precision scales, not container bytes.
//! The ISA flags above describe legacy packed kernel availability, not SIMD
//! acceleration for these semantic operations; the packed arithmetic is scalar.
//! [`CpuBackend::additive_matmul_input_vjp`] supplies the shared-reference input
//! adjoint for additive handles with caller-owned row workspace. It is scalar,
//! CPU-only and distinct from complete backend training or owner refinement.
//! Packed stored-scale adjoints share the core geometry/gradient loop through
//! [`CpuBackend::accumulate_additive_matmul_scale_vjp`] and
//! [`CpuBackend::accumulate_additive_embed_scale_vjp`], without expanding weights.
//
// `linkme`'s `distributed_slice` expands to a static with a custom
// `#[link_section]`, and the AVX2 kernel needs hand-written `unsafe` for its
// intrinsics. We therefore `deny` (not `forbid`) unsafe code at the crate level
// and grant `#[allow(unsafe_code)]` narrowly on the registration static; the
// kernel module carries its own scoped `unsafe` with `// SAFETY:` notes.
#![deny(unsafe_code)]

use core::any::Any;

use tritium_core::{AdditiveAdjointSpan, GemmShape, TernaryFormat, Trit};
use tritium_format::{
    TQ1_0_BLOCK_BYTES, TQ2_0_BLOCK_BYTES, num_blocks, unpack_tq1_0_row, unpack_tq2_0_row,
};
use tritium_runtime::BackendEntry;
use tritium_spec::{
    BackendError, DeviceBuffer, DeviceCaps, MpGemm, MpGemmProjectedVjp, TensorMatmul, TensorView,
    TernaryBackend,
};

#[cfg(test)]
std::thread_local! {
    static UNPACK_WEIGHT_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

// The kernel module holds the only hand-written `unsafe` in the crate: the AVX2
// intrinsic kernel (its `unsafe fn` plus the raw-pointer loads/stores). The
// crate-level `deny(unsafe_code)` is relaxed for exactly this module; each unsafe
// block inside carries a `// SAFETY:` note.
#[allow(unsafe_code)]
mod kernel;

// SIMD kernel variants for v0.30 (AVX-512 / VNNI, ARM NEON, T-MAC LUT). WF-C
// implements them (ADR 0005) and wires the selection into
// `kernel::dispatch_mpgemm` behind the existing `is_x86_feature_detected!` /
// `target_arch` dispatch, gated by the cross-ISA conformance parity gate. The
// `avx512`/`neon` submodules carry hand-written `unsafe` for their intrinsics
// (each block with a `// SAFETY:` note), so — exactly like `mod kernel` — the
// crate-level `deny(unsafe_code)` is relaxed for this module tree; the `lut`
// submodule is pure safe arithmetic and uses no `unsafe`.
#[allow(unsafe_code)]
mod simd;

/// Scalar packed residual-bearing QTEA reference, independent of the offline fitter.
pub mod qtea;
/// Exact, allocation-accounted SALT V2 CPU semantic reference.
pub mod salt_v2;
mod semantic;
pub use semantic::CpuScaleSnapshotBytes;

/// Owned host-memory buffer of packed weight bytes.
///
/// [`CpuBackend::upload_weights`] copies the caller's packed bytes into one of
/// these; [`CpuBackend::mpgemm`] downcasts the [`DeviceBuffer`] back to it and
/// unpacks on the fly. Storing the packed (not unpacked) bytes keeps the upload
/// cheap and the buffer the same size as the on-disk weight.
#[derive(Debug, Clone)]
pub struct CpuBuffer {
    bytes: Vec<u8>,
    n: usize,
    k: usize,
    format: TernaryFormat,
    tq2_invalid_row: Option<usize>,
}

impl CpuBuffer {
    /// Wrap already-packed bytes in a buffer.
    #[must_use]
    fn new(
        bytes: Vec<u8>,
        n: usize,
        k: usize,
        format: TernaryFormat,
        tq2_invalid_row: Option<usize>,
    ) -> Self {
        Self {
            bytes,
            n,
            k,
            format,
            tq2_invalid_row,
        }
    }

    /// The packed bytes this buffer holds.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    fn validate_identity(
        &self,
        shape: GemmShape,
        format: TernaryFormat,
    ) -> Result<(), BackendError> {
        if self.n != shape.n || self.k != shape.k || self.format != format {
            return Err(BackendError::InvalidInput(format!(
                "buffer identity N={} K={} format={:?} does not match N={} K={} format={format:?}",
                self.n, self.k, self.format, shape.n, shape.k
            )));
        }
        Ok(())
    }
}

impl DeviceBuffer for CpuBuffer {
    fn len_bytes(&self) -> usize {
        self.bytes.len()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Packed bytes per block for a format this backend supports.
///
/// # Errors
/// [`BackendError::UnsupportedFormat`] for any format the CPU backend does not
/// pack (everything but TQ2_0 / TQ1_0 today).
fn block_bytes(format: TernaryFormat) -> Result<usize, BackendError> {
    match format {
        TernaryFormat::Tq2_0 => Ok(TQ2_0_BLOCK_BYTES),
        TernaryFormat::Tq1_0 => Ok(TQ1_0_BLOCK_BYTES),
        other => Err(BackendError::UnsupportedFormat(other)),
    }
}

/// The x86-64 CPU backend.
///
/// Stateless: every call re-derives what it needs from its arguments, so a single
/// `CpuBackend` is shared freely across threads (`Send + Sync`).
#[derive(Debug, Default, Clone)]
pub struct CpuBackend;

impl CpuBackend {
    /// Construct the CPU backend.
    #[must_use]
    pub fn new() -> Self {
        CpuBackend
    }

    /// Create an immutable exact-scale version sharing an original packed root.
    ///
    /// Requires an owning Arc so the source handle and any original admission
    /// lease survive old/new graphs. Snapshots of snapshots flatten to the
    /// original root, retaining no obsolete overlay. Codes/geometry are not copied.
    /// Scales must already be finite, nonnegative (not negative zero) and exactly
    /// representable in their declared precision; no silent narrowing occurs.
    ///
    /// Validates before reserving/allocating new stored-scale bytes from `budget`.
    /// This charges only the new scale buffer and retains, but does not create or
    /// authenticate, parent admission. Independent budgets do not coordinate.
    /// Returned byte fields distinguish shared root (including original scales),
    /// owned scale bytes and reachable sum. `len_bytes()` reports the reachable
    /// sum, not the incremental charge; summing snapshots double-counts roots.
    /// Metadata/allocator capacity/RSS are excluded. No format/model identity,
    /// optimizer action, complete refinement or Python training is produced.
    ///
    /// # Errors
    /// Rejects dense/foreign/legacy or misreported handles, invalid scale values,
    /// size overflow, allocation failure and insufficient budget without changing
    /// the source or leaking a new reservation.
    pub fn snapshot_additive_scales(
        &self,
        parent: &std::sync::Arc<dyn DeviceBuffer>,
        scales: &[f32],
        budget: &tritium_runtime::TensorPayloadBudget,
    ) -> Result<(std::sync::Arc<dyn DeviceBuffer>, CpuScaleSnapshotBytes), BackendError> {
        semantic::CpuTensor::snapshot_scales(parent, scales, budget)
    }

    /// Create a complete immutable packed version with new codes and scales.
    ///
    /// Candidate must retain the parent's exact layout, basis, law/precision,
    /// codec, transport, plane count and per-tile counts. Full canonical symbols
    /// and exact nonnegative stored scales are checked before allocating. Reuses
    /// the packed uploader; no decoded or dense shadow is retained.
    ///
    /// Reserves the entire new packed payload, including compact auxiliary
    /// buffers, from `budget` before allocation. The final owning handle releases
    /// that reservation. `len_bytes()` is this exact owned charge, not shared
    /// parent bytes or physical RSS. Caller candidate/optimizer/graph scratch and
    /// allocator/metadata overhead are excluded; independent budgets do not
    /// coordinate. Old graphs retain their old handles. The new handle retains
    /// no parent or obsolete overlay, and supports ordinary inference/adjoints
    /// and later scale/full snapshots. No optimizer, checkpoint, parent identity
    /// authentication or complete-training qualification is produced.
    ///
    /// # Errors
    /// Rejects foreign/dense/legacy/misreported or changed-root parents,
    /// incompatible or noncanonical candidates, overflow, allocation failure or
    /// exhausted budgets without changing old handles or leaking admission.
    pub fn snapshot_additive_weights(
        &self,
        parent: &std::sync::Arc<dyn DeviceBuffer>,
        candidate: tritium_core::AdditiveView<'_>,
        budget: &tritium_runtime::TensorPayloadBudget,
    ) -> Result<std::sync::Arc<dyn DeviceBuffer>, BackendError> {
        semantic::CpuTensor::snapshot_weights(parent, candidate, budget)
    }

    /// Original-coordinate input adjoint for an uploaded CPU additive matrix.
    ///
    /// Matches [`tritium_core::reference_ternary_matmul_input_vjp`] using packed
    /// symbols/scales directly. `output_gradients` has `batch * rows` elements,
    /// `row_workspace` exactly `cols`, and `out` exactly `batch * cols`. Both
    /// products and all lengths are checked before mutation. Zero batch leaves
    /// the still-required row workspace untouched. No internal heap scratch or
    /// dense matrix is allocated; finite-value policy remains the caller's.
    /// This CPU-only input adjoint is not a weight/scale/code update or a claim
    /// of complete backend training support. Existing inference caps are unchanged.
    ///
    /// # Errors
    /// [`BackendError::InvalidInput`] for foreign/legacy handles, shape mismatch
    /// or dimension overflow; [`BackendError::UnsupportedTensor`] for dense handles.
    pub fn additive_matmul_input_vjp(
        &self,
        tensor: &dyn DeviceBuffer,
        output_gradients: &[f32],
        batch: usize,
        row_workspace: &mut [f32],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        semantic::CpuTensor::additive_input_vjp(tensor, output_gradients, batch, row_workspace, out)
    }

    /// Actual stored-scale count of an admitted CPU additive handle.
    ///
    /// Metadata only, excluding derived tied scales; no payload decoding or
    /// allocation. This is not a complete backend-training capability report.
    /// # Errors
    /// Foreign/legacy handles return InvalidInput; dense returns UnsupportedTensor.
    pub fn additive_stored_scale_count(
        &self,
        tensor: &dyn DeviceBuffer,
    ) -> Result<usize, BackendError> {
        semantic::CpuTensor::additive_scale_count(tensor)
    }

    /// Compare actual immutable packed state with an exact decoded view.
    ///
    /// Checks layout/codec/basis/counts, stored scale bits and every coefficient
    /// without allocating a decoded master. Linear in coefficient count; intended
    /// for initial binding, not every optimizer step. This is numeric equality,
    /// not package/source/model authentication or a qualification receipt.
    /// # Errors
    /// Invalid views/foreign/legacy/changed or misreported handles are refused;
    /// dense handles return UnsupportedTensor.
    pub fn additive_equals_view(
        &self,
        tensor: &dyn DeviceBuffer,
        view: tritium_core::AdditiveView<'_>,
    ) -> Result<bool, BackendError> {
        semantic::CpuTensor::additive_equals_view(tensor, view)
    }

    /// Accumulate packed matmul cotangents into actual stored-scale gradients.
    ///
    /// Matches [`tritium_core::accumulate_ternary_matmul_scale_vjp`] through its
    /// shared checked plan, reading canonical symbols directly from this handle.
    /// Inputs have batch*cols and batch*rows elements, row workspace exactly cols,
    /// and f64 output exactly the stored-scale count. Adds in shared reference
    /// order. No heap, dense or decoded-trit scratch; empty batch leaves buffers
    /// untouched. Finite/overflow policy belongs to the caller. The immutable
    /// handle may be shared with callers using independent buffers. No scale
    /// update, code STE or full training/other-backend qualification is implied.
    /// # Errors
    /// InvalidInput for foreign/legacy handles, shapes or overflow;
    /// UnsupportedTensor for dense handles. Refusal precedes buffer mutation.
    pub fn accumulate_additive_matmul_scale_vjp(
        &self,
        tensor: &dyn DeviceBuffer,
        activations: &[f32],
        output_gradients: &[f32],
        batch: usize,
        row_workspace: &mut [f32],
        out: &mut [f64],
    ) -> Result<(), BackendError> {
        semantic::CpuTensor::additive_matmul_scale_vjp(
            tensor,
            activations,
            output_gradients,
            batch,
            row_workspace,
            out,
        )
    }

    /// Accumulate packed embedding cotangents into actual stored-scale gradients.
    ///
    /// Matches [`tritium_core::accumulate_embed_scale_vjp`]: ids retain occurrence
    /// order (including duplicates), cotangents are ids.len()*cols, workspace cols,
    /// f64 output the stored-scale count. Uses forward basis on cotangents and
    /// directly reads admitted packed symbols without allocation or mutation of
    /// the handle. Every shape/index is checked before output/workspace mutation;
    /// empty ids leave buffers untouched. Finite/overflow policy is the caller's.
    /// # Errors
    /// Same handle/shape refusals as the packed matmul scale adjoint, including
    /// out-of-range ids. Existing inference capability reports remain unchanged.
    pub fn accumulate_additive_embed_scale_vjp(
        &self,
        tensor: &dyn DeviceBuffer,
        ids: &[usize],
        output_gradients: &[f32],
        row_workspace: &mut [f32],
        out: &mut [f64],
    ) -> Result<(), BackendError> {
        semantic::CpuTensor::additive_embed_scale_vjp(
            tensor,
            ids,
            output_gradients,
            row_workspace,
            out,
        )
    }

    /// Count present base-plane weight coordinates, including zero trits.
    ///
    /// Excludes pruned tiles and additional planes. No payload decode/allocation.
    /// # Errors
    /// Foreign, legacy, changed-root or misreported handles are InvalidInput;
    /// dense handles are UnsupportedTensor.
    pub fn additive_present_weight_count(
        &self,
        tensor: &dyn DeviceBuffer,
    ) -> Result<usize, BackendError> {
        semantic::CpuTensor::additive_weight_count(tensor)
    }

    /// Accumulate `dy * (B x)` into present stored-weight cotangents.
    ///
    /// Uses [`tritium_core::AdditiveWeightVjpPlan::matmul`] without reading codes
    /// or scales. Output has additive_present_weight_count f64 values in compact
    /// base-plane order; workspace is exactly columns f32. No heap or dense
    /// master. This is not a per-trit STE or complete training qualification.
    /// Finite/overflow policy belongs to the caller. Empty uses change nothing.
    /// # Errors
    /// Handle and complete shape/product preflight precede buffer mutation.
    /// Handle refusals match additive_present_weight_count.
    pub fn accumulate_additive_matmul_weight_vjp(
        &self,
        tensor: &dyn DeviceBuffer,
        activations: &[f32],
        output_gradients: &[f32],
        batch: usize,
        row_workspace: &mut [f32],
        out: &mut [f64],
    ) -> Result<(), BackendError> {
        semantic::CpuTensor::additive_matmul_weight_vjp(
            tensor,
            activations,
            output_gradients,
            batch,
            row_workspace,
            out,
        )
    }

    /// Accumulate embedding `B dy` in occurrence order into present weights.
    ///
    /// Uses [`tritium_core::AdditiveWeightVjpPlan::embed`], including duplicate
    /// indices and forward signed RHT. Scratch/output/count and numerical policy
    /// match the matmul weight adjoint. No allocation or handle mutation.
    /// # Errors
    /// All indices/shapes and handle/root/claimed bytes are checked before
    /// buffer mutation. Empty uses still require the exact workspace/output.
    pub fn accumulate_additive_embed_weight_vjp(
        &self,
        tensor: &dyn DeviceBuffer,
        ids: &[usize],
        output_gradients: &[f32],
        row_workspace: &mut [f32],
        out: &mut [f64],
    ) -> Result<(), BackendError> {
        semantic::CpuTensor::additive_embed_weight_vjp(
            tensor,
            ids,
            output_gradients,
            row_workspace,
            out,
        )
    }

    /// Accumulate one contiguous stored-scale span of a packed matmul VJP.
    /// Per-coordinate order, basis and tied-plane factors match the full VJP;
    /// only caller row/output buffers are used. No handle or payload mutation.
    /// # Errors
    /// InvalidInput for foreign/misreported handles, shapes or invalid spans;
    /// UnsupportedTensor for dense handles. Refusal precedes buffer mutation.
    pub fn accumulate_additive_matmul_scale_vjp_span(
        &self,
        tensor: &dyn DeviceBuffer,
        activations: &[f32],
        output_gradients: &[f32],
        batch: usize,
        row_workspace: &mut [f32],
        out: AdditiveAdjointSpan<'_>,
    ) -> Result<(), BackendError> {
        semantic::CpuTensor::additive_matmul_scale_vjp_span(
            tensor,
            activations,
            output_gradients,
            batch,
            row_workspace,
            out,
        )
    }

    /// Accumulate one contiguous stored-scale span of a packed embedding VJP.
    /// Duplicate ids retain occurrence order; forward basis matches the full VJP.
    /// No allocation or handle mutation; this is not a new training operation.
    /// # Errors
    /// Same refusals as the matmul span, including all out-of-range indices.
    pub fn accumulate_additive_embed_scale_vjp_span(
        &self,
        tensor: &dyn DeviceBuffer,
        ids: &[usize],
        output_gradients: &[f32],
        row_workspace: &mut [f32],
        out: AdditiveAdjointSpan<'_>,
    ) -> Result<(), BackendError> {
        semantic::CpuTensor::additive_embed_scale_vjp_span(
            tensor,
            ids,
            output_gradients,
            row_workspace,
            out,
        )
    }

    /// Accumulate one contiguous present-weight span of a packed matmul VJP.
    /// Uses the full VJP's compact base-plane coordinates and per-coordinate
    /// accumulation order, without reading codes/scales or allocating a master.
    /// # Errors
    /// Handle/shape/span preflight precedes mutation, matching scale spans.
    pub fn accumulate_additive_matmul_weight_vjp_span(
        &self,
        tensor: &dyn DeviceBuffer,
        activations: &[f32],
        output_gradients: &[f32],
        batch: usize,
        row_workspace: &mut [f32],
        out: AdditiveAdjointSpan<'_>,
    ) -> Result<(), BackendError> {
        semantic::CpuTensor::additive_matmul_weight_vjp_span(
            tensor,
            activations,
            output_gradients,
            batch,
            row_workspace,
            out,
        )
    }

    /// Accumulate one contiguous present-weight span of a packed embedding VJP.
    /// Repeated ids and signed forward basis match the full VJP. No allocation.
    /// # Errors
    /// Handle/shape/index/span preflight precedes output/workspace mutation.
    pub fn accumulate_additive_embed_weight_vjp_span(
        &self,
        tensor: &dyn DeviceBuffer,
        ids: &[usize],
        output_gradients: &[f32],
        row_workspace: &mut [f32],
        out: AdditiveAdjointSpan<'_>,
    ) -> Result<(), BackendError> {
        semantic::CpuTensor::additive_embed_weight_vjp_span(
            tensor,
            ids,
            output_gradients,
            row_workspace,
            out,
        )
    }

    /// Unpack the `[N, K]` packed weights in `buf` to a flat `Vec<Trit>` (one
    /// `N·K` row-major buffer), validating the byte length against `shape`.
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] if the byte length disagrees with the shape
    /// and format; [`BackendError::Backend`] if `tritium-format` rejects a row.
    fn unpack_weights(
        buf: &CpuBuffer,
        shape: GemmShape,
        format: TernaryFormat,
    ) -> Result<Vec<Trit>, BackendError> {
        #[cfg(test)]
        UNPACK_WEIGHT_CALLS.with(|calls| calls.set(calls.get() + 1));

        buf.validate_identity(shape, format)?;
        let GemmShape { n, k, .. } = shape;
        let nb = num_blocks(k);
        let row_bytes = nb * block_bytes(format)?;
        let expected = n * row_bytes;
        if buf.bytes.len() != expected {
            return Err(BackendError::ShapeMismatch {
                expected,
                got: buf.bytes.len(),
            });
        }

        let mut trits = vec![Trit::ZERO; n * k];
        // Per-block scale scratch: the host packer fixed these to 1.0 and the
        // per-channel scales are applied in `mpgemm`, so the unpacked block scales
        // are discarded.
        let mut scratch = vec![half::f16::ONE; nb];
        for ni in 0..n {
            let row = &buf.bytes[ni * row_bytes..(ni + 1) * row_bytes];
            let trits_row = &mut trits[ni * k..ni * k + k];
            let res = match format {
                TernaryFormat::Tq2_0 => unpack_tq2_0_row(row, trits_row, &mut scratch),
                TernaryFormat::Tq1_0 => unpack_tq1_0_row(row, trits_row, &mut scratch),
                other => return Err(BackendError::UnsupportedFormat(other)),
            };
            res.map_err(|e| BackendError::Backend(format!("unpack row {ni}: {e}")))?;
        }
        Ok(trits)
    }
}

impl TernaryBackend for CpuBackend {
    fn tensor_stream_plan(
        &self,
        metadata: tritium_spec::TensorMetadata<'_>,
        encoding: tritium_spec::TensorStreamEncoding,
    ) -> Result<Option<tritium_spec::TensorStreamPlan>, BackendError> {
        semantic::CpuTensor::stream_plan(metadata, encoding)
    }
    fn upload_tensor_stream(
        &self,
        source: &dyn tritium_spec::TensorStreamSource,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        semantic::CpuTensor::upload_stream(source)
    }
    fn tensor_upload_plan(
        &self,
        metadata: tritium_spec::TensorMetadata<'_>,
    ) -> Result<Option<tritium_spec::TensorUploadPlan>, BackendError> {
        semantic::CpuTensor::plan(metadata)
    }
    fn tensor_caps(
        &self,
        tensor: TensorView<'_>,
    ) -> Result<Option<tritium_spec::TensorCaps>, BackendError> {
        semantic::CpuTensor::caps(tensor)
    }

    // Unified additive operations retain canonical packed planes; dense and
    // shared basis arithmetic remain exact scalar f32. No SIMD claim follows.
    fn upload_tensor(&self, tensor: TensorView<'_>) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        semantic::CpuTensor::upload(tensor)
    }

    fn matmul(&self, p: TensorMatmul<'_>) -> Result<(), BackendError> {
        semantic::CpuTensor::matmul(p)
    }

    fn embed_rows(
        &self,
        tensor: &dyn DeviceBuffer,
        ids: &[usize],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        semantic::CpuTensor::embed(tensor, ids, out)
    }

    fn device_id(&self) -> &str {
        "cpu"
    }

    fn capabilities(&self) -> DeviceCaps {
        // `mut` is used on x86_64 and aarch64 (both push detected SIMD features);
        // on any other architecture neither block compiles, leaving `features`
        // empty and unmutated, so the `mut` is dead there only.
        #[cfg_attr(
            not(any(target_arch = "x86_64", target_arch = "aarch64")),
            allow(unused_mut)
        )]
        let mut features: Vec<String> = Vec::new();
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                features.push("avx2".to_owned());
            }
            // AVX-512: the kernel needs the F + BW + VL subsets (BW for the
            // byte-lane trit compares, VL to emit a mask from a 128-bit operand).
            if is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vl")
            {
                features.push("avx512".to_owned());
            }
            if is_x86_feature_detected!("fma") {
                features.push("fma".to_owned());
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // Baseline NEON (`Advanced SIMD`) is mandatory on aarch64, so the
            // NEON kernel is always live there.
            features.push("neon".to_owned());
        }
        // `features` is left empty on architectures with no SIMD kernel (the
        // scalar / LUT fallback runs); `with_features` accepts an empty list.
        DeviceCaps::new("cpu", host_arch_name()).with_features(features)
    }

    fn upload_weights(
        &self,
        packed: &[u8],
        shape: GemmShape,
        format: TernaryFormat,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        let GemmShape { n, k, .. } = shape;
        let nb = num_blocks(k);
        let row_bytes = nb.checked_mul(block_bytes(format)?).ok_or_else(|| {
            BackendError::InvalidInput("packed row byte count overflows".to_owned())
        })?;
        let expected = n.checked_mul(row_bytes).ok_or_else(|| {
            BackendError::InvalidInput("packed weight byte count overflows".to_owned())
        })?;
        if packed.len() != expected {
            return Err(BackendError::InvalidInput(format!(
                "packed len {} != expected {expected} for shape {shape:?} format {format:?}",
                packed.len()
            )));
        }
        let tq2_invalid_row = if format == TernaryFormat::Tq2_0 && row_bytes != 0 {
            packed.chunks_exact(row_bytes).position(|row| {
                row.as_chunks::<TQ2_0_BLOCK_BYTES>().0.iter().any(|block| {
                    block[..tritium_format::QK_K / 4]
                        .iter()
                        .any(|&byte| byte & (byte >> 1) & 0x55 != 0)
                })
            })
        } else {
            None
        };
        Ok(Box::new(CpuBuffer::new(
            packed.to_vec(),
            n,
            k,
            format,
            tq2_invalid_row,
        )))
    }

    fn mpgemm(&self, p: tritium_spec::MpGemm<'_>) -> Result<(), BackendError> {
        let MpGemm {
            act,
            weights,
            scales,
            shape,
            format,
            out,
        } = p;
        let buf = weights
            .as_any()
            .downcast_ref::<CpuBuffer>()
            .ok_or_else(|| BackendError::InvalidInput("buffer is not a CpuBuffer".to_owned()))?;
        buf.validate_identity(shape, format)?;

        let GemmShape { m, n, k } = shape;
        // Validate operand lengths against the shape before any kernel runs.
        let act_len = m
            .checked_mul(k)
            .ok_or_else(|| BackendError::InvalidInput("M*K element count overflows".to_owned()))?;
        let out_len = m
            .checked_mul(n)
            .ok_or_else(|| BackendError::InvalidInput("M*N element count overflows".to_owned()))?;
        if act.len() != act_len {
            return Err(BackendError::ShapeMismatch {
                expected: act_len,
                got: act.len(),
            });
        }
        if scales.len() != n {
            return Err(BackendError::ShapeMismatch {
                expected: n,
                got: scales.len(),
            });
        }
        if out.len() != out_len {
            return Err(BackendError::ShapeMismatch {
                expected: out_len,
                got: out.len(),
            });
        }

        // Unpack the [N, K] weights; this also validates the packed byte length.
        let trits = Self::unpack_weights(buf, shape, format)?;

        // WF-C dispatch hook: `dispatch_mpgemm` picks AVX2-vs-scalar today; the
        // v0.30 `simd::{avx512,neon,lut}` paths slot in here behind feature
        // detection (results must stay bit-parity with the scalar reference).
        kernel::dispatch_mpgemm(act, &trits, scales, shape, out)
    }

    fn mpgemm_projected_vjp(&self, p: MpGemmProjectedVjp<'_>) -> Result<(), BackendError> {
        let MpGemmProjectedVjp {
            act,
            weights,
            scales,
            grad_output,
            shape,
            format,
            grad_act,
            grad_projected_weight,
            grad_bias,
        } = p;
        let buffer = weights
            .as_any()
            .downcast_ref::<CpuBuffer>()
            .ok_or_else(|| BackendError::InvalidInput("buffer is not a CpuBuffer".to_owned()))?;
        buffer.validate_identity(shape, format)?;
        match format {
            TernaryFormat::Tq2_0 => {
                if let Some(ni) = buffer.tq2_invalid_row {
                    return Err(BackendError::Backend(format!(
                        "unpack row {ni}: decoded value 2 outside ternary range"
                    )));
                }
                kernel::dispatch_tq2_projected_vjp(
                    act,
                    buffer.bytes(),
                    scales,
                    grad_output,
                    shape,
                    grad_act,
                    grad_projected_weight,
                    grad_bias,
                )
            }
            TernaryFormat::Tq1_0 => {
                let trits = Self::unpack_weights(buffer, shape, format)?;
                kernel::dispatch_mpgemm_projected_vjp(
                    act,
                    &trits,
                    scales,
                    grad_output,
                    shape,
                    grad_act,
                    grad_projected_weight,
                    grad_bias,
                )
            }
            other => Err(BackendError::UnsupportedFormat(other)),
        }
    }
}

/// A human-readable host arch name with the SIMD tier appended when known.
fn host_arch_name() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            "x86_64 (avx2)".to_owned()
        } else {
            "x86_64".to_owned()
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        std::env::consts::ARCH.to_owned()
    }
}

/// `init` constructor for the runtime registry. The CPU backend is always
/// available, so this never fails.
///
/// # Errors
/// Never returns `Err`; the signature matches [`BackendEntry::init`].
fn init_cpu() -> Result<Box<dyn TernaryBackend>, BackendError> {
    Ok(Box::new(CpuBackend::new()))
}

// `linkme`'s `distributed_slice` expands to a static with a custom
// `#[link_section]`, which the `unsafe_code` lint flags. Grant a narrowly-scoped
// allow on exactly this registration static (the crate otherwise `deny`s unsafe
// outside the kernel module).
#[allow(unsafe_code)]
#[linkme::distributed_slice(tritium_runtime::BACKENDS)]
static CPU: BackendEntry = BackendEntry {
    name: "cpu",
    init: init_cpu,
};

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use tritium_format::{pack_tq1_0_row, pack_tq2_0_row};
    use tritium_testkit::{
        Tolerance, frozen_vectors, generate_vectors, run_conformance, run_fused_fallback_contract,
    };

    /// Pack an `[N, K]` trit matrix into the format's row layout, block scale
    /// fixed to `1.0` (the testkit convention), ready for `upload_weights`.
    fn pack(trits: &[Trit], shape: GemmShape, format: TernaryFormat) -> Vec<u8> {
        let GemmShape { n, k, .. } = shape;
        let nb = num_blocks(k);
        let unit = vec![f16::ONE; nb];
        let row_bytes = nb * block_bytes(format).unwrap();
        let mut packed = vec![0u8; n * row_bytes];
        for ni in 0..n {
            let row = &trits[ni * k..ni * k + k];
            let out = &mut packed[ni * row_bytes..(ni + 1) * row_bytes];
            match format {
                TernaryFormat::Tq2_0 => pack_tq2_0_row(row, &unit, out).unwrap(),
                TernaryFormat::Tq1_0 => pack_tq1_0_row(row, &unit, out).unwrap(),
                _ => unreachable!("only TQ2_0/TQ1_0 are packed in tests"),
            }
        }
        packed
    }

    /// Run one mpGEMM end-to-end through the backend (upload + compute).
    fn run(
        trits: &[Trit],
        act: &[f32],
        scales: &[f32],
        shape: GemmShape,
        format: TernaryFormat,
    ) -> Vec<f32> {
        let backend = CpuBackend::new();
        let packed = pack(trits, shape, format);
        let buf = backend.upload_weights(&packed, shape, format).unwrap();
        let mut out = vec![0.0f32; shape.m * shape.n];
        backend
            .mpgemm(MpGemm {
                act,
                weights: buf.as_ref(),
                scales,
                shape,
                format,
                out: &mut out,
            })
            .unwrap();
        out
    }

    fn trits(vals: &[i8]) -> Vec<Trit> {
        vals.iter().map(|&v| Trit::from_i8(v).unwrap()).collect()
    }

    /// Relative-tolerance assertion matching ADR 0002 (`1e-4`).
    fn close(got: f32, want: f32) -> bool {
        (got - want).abs() <= 1e-4 * want.abs().max(1.0)
    }

    #[test]
    fn packed_projected_vjp_matches_dense_contract() {
        let shape = GemmShape::new(2, 2, 3);
        let weights = trits(&[1, 0, -1, -1, 1, 1]);
        let packed = pack(&weights, shape, TernaryFormat::Tq2_0);
        let backend = CpuBackend::new();
        let buffer = backend
            .upload_weights(&packed, shape, TernaryFormat::Tq2_0)
            .unwrap();
        let act = [1.0, 2.0, 3.0, -1.0, 4.0, 0.5];
        let scales = [2.0, 0.5];
        let grad_output = [3.0, -2.0, 4.0, 1.0];
        let mut grad_act = vec![f32::NAN; shape.m * shape.k];
        let mut grad_weight = vec![f32::NAN; shape.n * shape.k];
        let mut grad_bias = vec![f32::NAN; shape.n];

        backend
            .mpgemm_projected_vjp(MpGemmProjectedVjp {
                act: &act,
                weights: buffer.as_ref(),
                scales: &scales,
                grad_output: &grad_output,
                shape,
                format: TernaryFormat::Tq2_0,
                grad_act: &mut grad_act,
                grad_projected_weight: &mut grad_weight,
                grad_bias: Some(&mut grad_bias),
            })
            .unwrap();

        assert_eq!(grad_act, vec![7.0, -1.0, -7.0, 7.5, 0.5, -7.5]);
        assert_eq!(grad_weight, vec![-1.0, 22.0, 11.0, -3.0, 0.0, -5.5]);
        assert_eq!(grad_bias, vec![7.0, -1.0]);
    }

    #[test]
    fn tq2_projected_vjp_decodes_packed_weights_without_dense_unpack() {
        let shape = GemmShape::new(2, 3, 257);
        let weights: Vec<Trit> = (0..shape.n * shape.k)
            .map(|i| Trit::from_i8(((i * 17 + 5) % 3) as i8 - 1).unwrap())
            .collect();
        let packed = pack(&weights, shape, TernaryFormat::Tq2_0);
        let backend = CpuBackend::new();
        let buffer = backend
            .upload_weights(&packed, shape, TernaryFormat::Tq2_0)
            .unwrap();
        let act: Vec<f32> = (0..shape.m * shape.k)
            .map(|i| (i % 23) as f32 * 0.125 - 1.0)
            .collect();
        let scales = [0.25, -1.5, 2.0];
        let grad_output = [0.5, -2.0, 1.25, -0.75, 3.0, -0.5];
        let mut want_grad_act = vec![0.0; shape.m * shape.k];
        let mut want_grad_weight = vec![0.0; shape.n * shape.k];
        kernel::dispatch_mpgemm_projected_vjp(
            &act,
            &weights,
            &scales,
            &grad_output,
            shape,
            &mut want_grad_act,
            &mut want_grad_weight,
            None,
        )
        .unwrap();

        let mut got_grad_act = vec![f32::NAN; shape.m * shape.k];
        let mut got_grad_weight = vec![f32::NAN; shape.n * shape.k];
        UNPACK_WEIGHT_CALLS.with(|calls| calls.set(0));
        backend
            .mpgemm_projected_vjp(MpGemmProjectedVjp {
                act: &act,
                weights: buffer.as_ref(),
                scales: &scales,
                grad_output: &grad_output,
                shape,
                format: TernaryFormat::Tq2_0,
                grad_act: &mut got_grad_act,
                grad_projected_weight: &mut got_grad_weight,
                grad_bias: None,
            })
            .unwrap();

        UNPACK_WEIGHT_CALLS.with(|calls| {
            assert_eq!(
                calls.get(),
                0,
                "TQ2_0 projected VJP must decode directly from packed bytes"
            );
        });
        for (got, want) in got_grad_act.iter().zip(&want_grad_act) {
            assert!(close(*got, *want), "grad_act: got {got}, want {want}");
        }
        for (got, want) in got_grad_weight.iter().zip(&want_grad_weight) {
            assert!(
                close(*got, *want),
                "grad_projected_weight: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn tq2_projected_vjp_rejects_reserved_code_before_writing_outputs() {
        let shape = GemmShape::new(1, 1, 1);
        let backend = CpuBackend::new();
        for corrupt_byte in [0, 1] {
            let mut packed = pack(&[Trit::POS], shape, TernaryFormat::Tq2_0);
            // Byte 0/lane 0 is the logical trit; byte 1/lane 0 is tail padding.
            packed[corrupt_byte] = packed[corrupt_byte] & !0b11 | 0b11;
            let buffer = backend
                .upload_weights(&packed, shape, TernaryFormat::Tq2_0)
                .unwrap();
            let mut grad_act = [41.0];
            let mut grad_weight = [42.0];
            let mut grad_bias = [43.0];

            let error = backend
                .mpgemm_projected_vjp(MpGemmProjectedVjp {
                    act: &[1.0],
                    weights: buffer.as_ref(),
                    scales: &[1.0],
                    grad_output: &[1.0],
                    shape,
                    format: TernaryFormat::Tq2_0,
                    grad_act: &mut grad_act,
                    grad_projected_weight: &mut grad_weight,
                    grad_bias: Some(&mut grad_bias),
                })
                .unwrap_err();

            assert_eq!(
                error,
                BackendError::Backend(
                    "unpack row 0: decoded value 2 outside ternary range".to_owned()
                )
            );
            assert_eq!(grad_act, [41.0]);
            assert_eq!(grad_weight, [42.0]);
            assert_eq!(grad_bias, [43.0]);
        }
    }

    #[test]
    fn tq2_projected_vjp_handles_zero_inner_dimension() {
        let shape = GemmShape::new(2, 3, 0);
        let backend = CpuBackend::new();
        let buffer = backend
            .upload_weights(&[], shape, TernaryFormat::Tq2_0)
            .unwrap();
        let mut grad_act = [];
        let mut grad_weight = [];
        let mut grad_bias = [f32::NAN; 3];

        backend
            .mpgemm_projected_vjp(MpGemmProjectedVjp {
                act: &[],
                weights: buffer.as_ref(),
                scales: &[1.0, 2.0, 3.0],
                grad_output: &[1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
                shape,
                format: TernaryFormat::Tq2_0,
                grad_act: &mut grad_act,
                grad_projected_weight: &mut grad_weight,
                grad_bias: Some(&mut grad_bias),
            })
            .unwrap();

        assert_eq!(grad_bias, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn packed_projected_vjp_rejects_bad_output_shape() {
        let shape = GemmShape::new(1, 1, 1);
        let packed = pack(&[Trit::POS], shape, TernaryFormat::Tq2_0);
        let backend = CpuBackend::new();
        let buffer = backend
            .upload_weights(&packed, shape, TernaryFormat::Tq2_0)
            .unwrap();
        let mut grad_act = [];
        let mut grad_weight = [0.0];

        let error = backend
            .mpgemm_projected_vjp(MpGemmProjectedVjp {
                act: &[1.0],
                weights: buffer.as_ref(),
                scales: &[1.0],
                grad_output: &[1.0],
                shape,
                format: TernaryFormat::Tq2_0,
                grad_act: &mut grad_act,
                grad_projected_weight: &mut grad_weight,
                grad_bias: None,
            })
            .unwrap_err();

        assert!(matches!(
            error,
            BackendError::ShapeMismatch {
                expected: 1,
                got: 0
            }
        ));
    }

    // ---- the conformance gate ------------------------------------------------

    /// The CPU backend reproduces the committed, frozen conformance set
    /// (`tritium-testkit`'s `vectors/v070.jsonl`) with zero failures. Grades
    /// against the immutable artifact rather than a regenerated seed, so this is
    /// the exact same reference every other backend runs (ADR 0009 freeze). The
    /// frozen set is `generate_vectors(0xC0FFEE, 64)`, so this changed nothing
    /// about what CPU validates — only locked it.
    #[test]
    fn conformance_zero_failures() {
        let vectors = frozen_vectors();
        let report = run_conformance(&CpuBackend::new(), &vectors, Tolerance::default());
        assert!(
            report.is_ok(),
            "{} conformance failures: {:?}",
            report.failed.len(),
            report.failed
        );
        assert_eq!(report.passed, vectors.len(), "all vectors must pass");
    }

    #[test]
    fn conformance_zero_failures_other_seeds() {
        for seed in [1u64, 7, 42, 0xDEAD_BEEF] {
            let vectors = generate_vectors(seed, 24);
            let report = run_conformance(&CpuBackend::new(), &vectors, Tolerance::default());
            assert!(
                report.is_ok(),
                "seed {seed}: {} failures: {:?}",
                report.failed.len(),
                report.failed
            );
        }
    }

    /// Capability fallback: the CPU backend advertises **no fp8 / IMMA** yet must
    /// still serve the fused `mpgemm_with_act_quant` path — degrading to the host
    /// quant and matching the host-A8 reference, never erroring. This pins the
    /// no-panic-degrade contract that every backend (wasm, wgpu, …) is held to.
    #[test]
    fn fused_path_degrades_without_fp8() {
        let caps = CpuBackend::new().capabilities();
        assert!(!caps.supports_fp8, "precondition: CPU advertises no fp8");
        assert!(!caps.supports_imma, "precondition: CPU advertises no IMMA");
        assert!(
            !caps.has_feature("i2s_int8"),
            "precondition: CPU advertises no on-device int8"
        );

        let vectors = frozen_vectors();
        let report =
            run_fused_fallback_contract(&CpuBackend::new(), &vectors, Tolerance::default());
        assert!(
            report.is_ok(),
            "{} fused-fallback failures: {:?}",
            report.failed.len(),
            report.failed
        );
        assert_eq!(
            report.passed,
            vectors.len(),
            "all vectors must degrade cleanly"
        );
    }

    // ---- device_id / capabilities -------------------------------------------

    #[test]
    fn device_id_is_cpu() {
        assert_eq!(CpuBackend::new().device_id(), "cpu");
    }

    #[test]
    fn capabilities_report_cpu_backend() {
        let caps = CpuBackend::new().capabilities();
        assert_eq!(caps.backend, "cpu");
        assert!(
            !caps.device_name.is_empty(),
            "device name should be set: {:?}",
            caps.device_name
        );
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            assert!(caps.has_feature("avx2"), "avx2 host must advertise avx2");
        }
    }

    // ---- registration --------------------------------------------------------

    #[test]
    fn registers_with_runtime() {
        // Force this crate's linkme static to be referenced so the entry is kept.
        let _ = &CPU;
        let backend = init_cpu().expect("cpu init never fails");
        assert_eq!(backend.device_id(), "cpu");
    }

    // ---- upload_weights validation ------------------------------------------

    #[test]
    fn upload_rejects_wrong_length() {
        let backend = CpuBackend::new();
        let shape = GemmShape::new(1, 2, 256);
        // One byte short of the expected packed size.
        let nb = num_blocks(256);
        let short = vec![0u8; 2 * nb * TQ2_0_BLOCK_BYTES - 1];
        // `Box<dyn DeviceBuffer>` is not `Debug`, so `unwrap_err` is unavailable;
        // match on the result instead.
        match backend.upload_weights(&short, shape, TernaryFormat::Tq2_0) {
            Err(BackendError::InvalidInput(_)) => {}
            other => panic!(
                "expected InvalidInput, got {:?}",
                other.map(|_| "ok-buffer")
            ),
        }
    }

    #[test]
    fn upload_accepts_correct_length() {
        let backend = CpuBackend::new();
        let shape = GemmShape::new(4, 3, 256);
        let w = vec![Trit::ZERO; 3 * 256];
        let packed = pack(&w, shape, TernaryFormat::Tq2_0);
        let buf = backend
            .upload_weights(&packed, shape, TernaryFormat::Tq2_0)
            .unwrap();
        assert_eq!(buf.len_bytes(), packed.len());
    }

    #[test]
    fn mpgemm_rejects_foreign_buffer() {
        // A DeviceBuffer that is not a CpuBuffer must be rejected.
        #[derive(Debug)]
        struct Foreign;
        impl DeviceBuffer for Foreign {
            fn len_bytes(&self) -> usize {
                0
            }
            fn as_any(&self) -> &dyn Any {
                self
            }
        }
        let backend = CpuBackend::new();
        let shape = GemmShape::new(1, 1, 256);
        let act = vec![0.0f32; 256];
        let scales = vec![1.0f32; 1];
        let mut out = vec![0.0f32; 1];
        let err = backend
            .mpgemm(MpGemm {
                act: &act,
                weights: &Foreign,
                scales: &scales,
                shape,
                format: TernaryFormat::Tq2_0,
                out: &mut out,
            })
            .unwrap_err();
        assert!(matches!(err, BackendError::InvalidInput(_)), "{err:?}");
    }

    #[test]
    fn mpgemm_rejects_bad_operand_lengths() {
        let backend = CpuBackend::new();
        let shape = GemmShape::new(2, 1, 256);
        let w = vec![Trit::ZERO; 256];
        let packed = pack(&w, shape, TernaryFormat::Tq2_0);
        let buf = backend
            .upload_weights(&packed, shape, TernaryFormat::Tq2_0)
            .unwrap();
        // act too short (should be 2*256).
        let act = vec![0.0f32; 256];
        let scales = vec![1.0f32; 1];
        let mut out = vec![0.0f32; 2];
        let err = backend
            .mpgemm(MpGemm {
                act: &act,
                weights: buf.as_ref(),
                scales: &scales,
                shape,
                format: TernaryFormat::Tq2_0,
                out: &mut out,
            })
            .unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch { .. }), "{err:?}");
    }

    #[test]
    fn mpgemm_rejects_activation_count_overflow() {
        let backend = CpuBackend::new();
        let shape = GemmShape::new(usize::MAX / 2 + 1, 1, 2);
        let packed = pack(&[Trit::ZERO; 2], shape, TernaryFormat::Tq2_0);
        let weights = backend
            .upload_weights(&packed, shape, TernaryFormat::Tq2_0)
            .unwrap();
        let mut out = [7.0];
        let result = backend.mpgemm(MpGemm {
            act: &[],
            weights: weights.as_ref(),
            scales: &[1.0],
            shape,
            format: TernaryFormat::Tq2_0,
            out: &mut out,
        });
        assert!(matches!(result, Err(BackendError::InvalidInput(_))));
        assert_eq!(out, [7.0]);
    }

    #[test]
    fn mpgemm_rejects_output_count_overflow() {
        let backend = CpuBackend::new();
        let shape = GemmShape::new(usize::MAX / 2 + 1, 2, 0);
        let weights = backend
            .upload_weights(&[], shape, TernaryFormat::Tq2_0)
            .unwrap();
        let mut out = [7.0];
        let result = backend.mpgemm(MpGemm {
            act: &[],
            weights: weights.as_ref(),
            scales: &[1.0; 2],
            shape,
            format: TernaryFormat::Tq2_0,
            out: &mut out,
        });
        assert!(matches!(result, Err(BackendError::InvalidInput(_))));
        assert_eq!(out, [7.0]);
    }

    // ---- boundary cases ------------------------------------------------------

    fn assert_zero_geometry_matches_reference(shape: GemmShape, fused: bool) {
        let backend = CpuBackend::new();
        let trits = vec![Trit::ZERO; shape.n * shape.k];
        let act = vec![0.0; shape.m * shape.k];
        let scales: Vec<f32> = (0..shape.n)
            .map(|index| if index % 2 == 0 { -2.0 } else { 1.5 })
            .collect();
        let mut expected = vec![17.0; shape.m * shape.n];
        tritium_core::reference_mpgemm(&act, &trits, &scales, shape, &mut expected).unwrap();
        for format in [TernaryFormat::Tq1_0, TernaryFormat::Tq2_0] {
            let packed = pack(&trits, shape, format);
            let weights = backend.upload_weights(&packed, shape, format).unwrap();
            let mut out = vec![17.0; expected.len()];
            let request = MpGemm {
                act: &act,
                weights: weights.as_ref(),
                scales: &scales,
                shape,
                format,
                out: &mut out,
            };
            if fused {
                backend.mpgemm_with_act_quant(request).unwrap();
            } else {
                backend.mpgemm(request).unwrap();
            }
            assert_eq!(
                out.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "{shape:?} {format:?} fused={fused}"
            );
        }
    }

    #[test]
    fn zero_geometry_mpgemm_contraction_matches_reference() {
        assert_zero_geometry_matches_reference(GemmShape::new(1, 1, 0), false);
    }

    #[test]
    fn zero_geometry_fused_contraction_matches_reference() {
        assert_zero_geometry_matches_reference(GemmShape::new(2, 3, 0), true);
    }

    #[test]
    fn zero_geometry_fused_columns_matches_reference() {
        assert_zero_geometry_matches_reference(GemmShape::new(1, 0, 1), true);
    }

    #[test]
    fn zero_geometry_fused_batch_matches_reference() {
        assert_zero_geometry_matches_reference(GemmShape::new(0, 3, 2), true);
    }

    #[test]
    fn zero_geometry_retains_weight_identity_error() {
        let backend = CpuBackend::new();
        let weight_shape = GemmShape::new(1, 1, 3);
        let packed = pack(&[Trit::ZERO; 3], weight_shape, TernaryFormat::Tq2_0);
        let weights = backend
            .upload_weights(&packed, weight_shape, TernaryFormat::Tq2_0)
            .unwrap();
        let error = backend
            .mpgemm_with_act_quant(MpGemm {
                act: &[0.0; 6],
                weights: weights.as_ref(),
                scales: &[],
                shape: GemmShape::new(2, 0, 3),
                format: TernaryFormat::Tq2_0,
                out: &mut [],
            })
            .unwrap_err();
        assert!(matches!(error, BackendError::InvalidInput(_)), "{error:?}");
    }

    #[test]
    fn boundary_all_zero_weights_give_zero() {
        let shape = GemmShape::new(4, 3, 256);
        let w = vec![Trit::ZERO; 3 * 256];
        let act: Vec<f32> = (0..4 * 256).map(|i| (i as f32 % 7.0) - 3.0).collect();
        let scales = vec![1.5f32; 3];
        let out = run(&w, &act, &scales, shape, TernaryFormat::Tq2_0);
        assert!(
            out.iter().all(|&x| x == 0.0),
            "all-zero weights -> zero out"
        );
    }

    #[test]
    fn boundary_m_equals_one() {
        let shape = GemmShape::new(1, 5, 256);
        let w: Vec<Trit> = (0..5 * 256)
            .map(|i| Trit::from_sign((i % 3) as i8 - 1))
            .collect();
        let act: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01) - 1.0).collect();
        let scales = vec![1.0f32; 5];
        let out = run(&w, &act, &scales, shape, TernaryFormat::Tq2_0);
        let mut want = vec![0.0f32; 5];
        tritium_core::reference_mpgemm(&act, &w, &scales, shape, &mut want).unwrap();
        for (g, e) in out.iter().zip(&want) {
            assert!(close(*g, *e));
        }
    }

    #[test]
    fn boundary_m_large() {
        let shape = GemmShape::new(64, 2, 256);
        let w: Vec<Trit> = (0..2 * 256)
            .map(|i| Trit::from_sign((i % 3) as i8 - 1))
            .collect();
        let act: Vec<f32> = (0..64 * 256)
            .map(|i| ((i * 31 % 200) as f32) / 50.0 - 2.0)
            .collect();
        let scales = vec![0.7f32, 1.3];
        let out = run(&w, &act, &scales, shape, TernaryFormat::Tq2_0);
        let mut want = vec![0.0f32; 64 * 2];
        tritium_core::reference_mpgemm(&act, &w, &scales, shape, &mut want).unwrap();
        for (g, e) in out.iter().zip(&want) {
            assert!(close(*g, *e));
        }
    }

    #[test]
    fn boundary_n_equals_one() {
        let shape = GemmShape::new(3, 1, 256);
        let w: Vec<Trit> = (0..256)
            .map(|i| Trit::from_sign((i % 3) as i8 - 1))
            .collect();
        let act: Vec<f32> = (0..3 * 256).map(|i| (i as f32 % 5.0) - 2.0).collect();
        let scales = vec![2.0f32];
        let out = run(&w, &act, &scales, shape, TernaryFormat::Tq2_0);
        let mut want = vec![0.0f32; 3];
        tritium_core::reference_mpgemm(&act, &w, &scales, shape, &mut want).unwrap();
        for (g, e) in out.iter().zip(&want) {
            assert!(close(*g, *e));
        }
    }

    #[test]
    fn boundary_all_pos_and_all_neg() {
        let shape = GemmShape::new(2, 2, 256);
        let act: Vec<f32> = (0..2 * 256).map(|i| (i as f32 % 11.0) - 5.0).collect();
        let scales = vec![1.0f32, 1.0];

        // All +1: each output is the row-sum of the activation.
        let allpos = vec![Trit::POS; 2 * 256];
        let out_pos = run(&allpos, &act, &scales, shape, TernaryFormat::Tq2_0);
        for (mi, _) in (0..2).enumerate() {
            let row_sum: f32 = act[mi * 256..mi * 256 + 256].iter().sum();
            for ni in 0..2 {
                let g = out_pos[mi * 2 + ni];
                assert!(close(g, row_sum));
            }
        }

        // All -1: each output is the negated row-sum.
        let allneg = vec![Trit::NEG; 2 * 256];
        let out_neg = run(&allneg, &act, &scales, shape, TernaryFormat::Tq1_0);
        for (mi, _) in (0..2).enumerate() {
            let row_sum: f32 = act[mi * 256..mi * 256 + 256].iter().sum();
            for ni in 0..2 {
                let g = out_neg[mi * 2 + ni];
                assert!(close(g, -row_sum));
            }
        }
    }

    #[test]
    fn nan_inf_activation_propagates_without_ub() {
        // A NaN/Inf activation hitting a non-zero trit must propagate, not crash.
        let shape = GemmShape::new(1, 1, 256);
        let w = vec![Trit::POS; 256];
        let mut act = vec![1.0f32; 256];
        act[0] = f32::NAN;
        let scales = vec![1.0f32];
        let out = run(&w, &act, &scales, shape, TernaryFormat::Tq2_0);
        assert!(
            out[0].is_nan(),
            "NaN activation must propagate to the output"
        );

        let mut act_inf = vec![1.0f32; 256];
        act_inf[5] = f32::INFINITY;
        let out_inf = run(&w, &act_inf, &scales, shape, TernaryFormat::Tq2_0);
        assert!(
            out_inf[0].is_infinite(),
            "Inf activation must propagate to the output"
        );
    }

    // ---- AVX2 vs scalar parity ----------------------------------------------

    #[test]
    fn avx2_vs_scalar_parity() {
        #[cfg(target_arch = "x86_64")]
        {
            if !is_x86_feature_detected!("avx2") {
                eprintln!("avx2 not detected on this host — skipping AVX2/scalar parity test");
                return;
            }
            let mut s = 0xBADC_0FFE_u64;
            let mut next = || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s
            };
            for trial in 0..24 {
                let m = 1 + (next() % 8) as usize;
                let n = 1 + (next() % 8) as usize;
                // Mix ragged Ks (with an AVX2 scalar tail) and large block-aligned
                // Ks (256/512) where the reference's own f32 cancellation is worst.
                let k = match trial % 3 {
                    0 => 17 + (next() % 100) as usize, // ragged tail
                    1 => 256,                          // one block
                    _ => 512,                          // two blocks
                };
                let shape = GemmShape::new(m, n, k);
                // Large-magnitude activations make partial sums cancel hard, so the
                // reference's f32 order is the only one within tolerance — and the
                // kernel reproduces it bit-for-bit.
                let act: Vec<f32> = (0..m * k)
                    .map(|_| (next() % 20000) as f32 / 100.0 - 100.0)
                    .collect();
                let w: Vec<Trit> = (0..n * k)
                    .map(|_| Trit::from_sign((next() % 3) as i8 - 1))
                    .collect();
                let scales: Vec<f32> = (0..n)
                    .map(|_| (next() % 200) as f32 / 100.0 + 0.1)
                    .collect();

                let mut scal = vec![0.0f32; m * n];
                kernel::scalar_mpgemm(&act, &w, &scales, shape, &mut scal).unwrap();
                let mut avx = vec![0.0f32; m * n];
                // SAFETY: avx2 availability was just confirmed by the guard above.
                #[allow(unsafe_code)]
                unsafe {
                    kernel::avx2_mpgemm(&act, &w, &scales, shape, &mut avx).unwrap();
                }
                // The kernel reproduces the reference's exact f32 accumulation
                // order, so parity is bit-exact, not merely within 1e-4.
                for (a, e) in avx.iter().zip(&scal) {
                    assert_eq!(
                        a.to_bits(),
                        e.to_bits(),
                        "trial {trial} shape {shape:?}: avx2 {a} vs scalar {e}"
                    );
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            eprintln!("not x86_64 — AVX2/scalar parity test is a no-op");
        }
    }

    // ---- determinism ---------------------------------------------------------

    #[test]
    fn determinism_same_input_same_bytes() {
        let shape = GemmShape::new(8, 4, 512);
        let w: Vec<Trit> = (0..4 * 512)
            .map(|i| Trit::from_sign((i % 3) as i8 - 1))
            .collect();
        let act: Vec<f32> = (0..8 * 512)
            .map(|i| ((i * 17 % 300) as f32) / 30.0 - 5.0)
            .collect();
        let scales: Vec<f32> = (0..4).map(|i| 0.5 + i as f32 * 0.25).collect();

        let a = run(&w, &act, &scales, shape, TernaryFormat::Tq2_0);
        let b = run(&w, &act, &scales, shape, TernaryFormat::Tq2_0);
        // Compare the exact bit patterns, not just within tolerance.
        let ab: Vec<u32> = a.iter().map(|x| x.to_bits()).collect();
        let bb: Vec<u32> = b.iter().map(|x| x.to_bits()).collect();
        assert_eq!(ab, bb, "same input must produce byte-identical output");
    }

    #[test]
    fn parallel_matches_reference_and_is_deterministic() {
        // M=64, N=32, K=512 splits into multiple rayon chunks (per-row op count
        // forces several tasks), so this exercises the parallel dispatch path. The
        // per-row accumulation order is unchanged, so the result must still be
        // bit-identical run-to-run and within tolerance of the reference.
        let shape = GemmShape::new(64, 32, 512);
        let w: Vec<Trit> = (0..32 * 512)
            .map(|i| Trit::from_sign((i % 3) as i8 - 1))
            .collect();
        let act: Vec<f32> = (0..64 * 512)
            .map(|i| ((i * 13 % 4000) as f32) / 20.0 - 100.0)
            .collect();
        let scales: Vec<f32> = (0..32).map(|i| 0.3 + (i % 7) as f32 * 0.2).collect();

        let a = run(&w, &act, &scales, shape, TernaryFormat::Tq2_0);
        let b = run(&w, &act, &scales, shape, TernaryFormat::Tq2_0);
        assert_eq!(
            a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "parallel run must be deterministic"
        );

        let mut want = vec![0.0f32; 64 * 32];
        tritium_core::reference_mpgemm(&act, &w, &scales, shape, &mut want).unwrap();
        for (g, e) in a.iter().zip(&want) {
            assert!(close(*g, *e), "parallel vs reference: {g} vs {e}");
        }
    }

    // ---- format round-trip through both packers -----------------------------

    #[test]
    fn both_formats_agree_with_reference() {
        let shape = GemmShape::new(2, 3, 256);
        let w = trits(&(0..3 * 256).map(|i| (i % 3) as i8 - 1).collect::<Vec<_>>());
        let act: Vec<f32> = (0..2 * 256).map(|i| (i as f32 % 9.0) - 4.0).collect();
        let scales = vec![1.1f32, 0.9, 1.0];

        let mut want = vec![0.0f32; 2 * 3];
        tritium_core::reference_mpgemm(&act, &w, &scales, shape, &mut want).unwrap();

        for format in [TernaryFormat::Tq2_0, TernaryFormat::Tq1_0] {
            let got = run(&w, &act, &scales, shape, format);
            for (g, e) in got.iter().zip(&want) {
                assert!(close(*g, *e), "format {format:?}: {g} vs {e}");
            }
        }
    }
}
