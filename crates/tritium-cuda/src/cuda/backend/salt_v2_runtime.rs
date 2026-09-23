use super::*;
use tritium_format::salt_v2_package::SALT_V2_ALLOCATION_TILE_SIZE;

/// Words per fused-launch descriptor: the C `SaltStreamTensor` in `salt_v2.cu` is
/// four device pointers and eight `u32`s, 64 bytes.
pub(super) const SALT_STREAM_DESCRIPTOR_WORDS: usize = 8;

/// Pack one tensor of a fused row-stream launch into the kernel's descriptor.
///
/// The words mirror `SaltStreamTensor` field for field on a little-endian device:
/// four pointers, then the eight `u32` fields paired into four words. `output` is
/// the tensor's own destination, so one launch can fill several buffers.
pub(super) fn salt_stream_descriptor(
    tensor: &SaltV2ResidentTensor,
    stream: &CudaStream,
    output: sys::CUdeviceptr,
    first_row: u32,
) -> Result<[u64; SALT_STREAM_DESCRIPTOR_WORDS], BackendError> {
    let narrow = |value: usize, name: &str| {
        u32::try_from(value)
            .map_err(|_| BackendError::InvalidInput(format!("{name} exceeds the u32 kernel ABI")))
    };
    let pair = |low: u32, high: u32| u64::from(low) | (u64::from(high) << 32);
    let index_metadata = tensor.index_metadata.as_ref().unwrap_or(&tensor.payload);
    Ok([
        crate::cuda::graph_raw::dptr(&tensor.payload, stream),
        crate::cuda::graph_raw::dptr(&tensor.scales, stream),
        crate::cuda::graph_raw::dptr(index_metadata, stream),
        output,
        pair(
            narrow(tensor.rows, "SALT V2 rows")?,
            narrow(tensor.tile_count, "SALT V2 tile count")?,
        ),
        pair(
            narrow(tensor.plane_count, "SALT V2 plane count")?,
            tensor.allocation_map_bytes,
        ),
        pair(tensor.rank_prefix_count, tensor.terminal_map_value),
        pair(first_row, 0),
    ])
}

/// Launch a fused row-stream GEMV over `tensor_count` descriptors (m = 1).
///
/// # Errors
/// Returns a driver failure.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_salt_v2_stream_multi_on(
    stream: &CudaStream,
    function: &CudaFunction,
    descriptors: &CudaSlice<u64>,
    tensor_count: u32,
    total_rows: u32,
    k: u32,
    table_bytes: u32,
    input: &CudaSlice<f32>,
) -> Result<(), BackendError> {
    let cfg = LaunchConfig {
        grid_dim: (total_rows.div_ceil(SALT_V2_STREAM_WARPS), 1, 1),
        block_dim: (SALT_V2_STREAM_WARPS * 32, 1, 1),
        shared_mem_bytes: SALT_V2_B3_TABLE_BYTES + table_bytes * SALT_V2_STREAM_WARPS,
    };
    let mut launch = stream.launch_builder(function);
    launch
        .arg(input)
        .arg(descriptors)
        .arg(&tensor_count)
        .arg(&total_rows)
        .arg(&k)
        .arg(&table_bytes);
    // SAFETY: every descriptor was built from a validated resident tensor this
    // backend owns and an output buffer sized for that tensor's rows; the kernel
    // writes each fused row once, into its own tensor's buffer.
    #[allow(unsafe_code)]
    unsafe {
        launch
            .launch(cfg)
            .map(|_| ())
            .map_err(|error| driver_err("launch SALT V2 fused row-stream forward", &error))
    }
}

/// Quantize `groups` 128-coefficient groups of f32 activations to int8 with one
/// scale per group, for the A8 row-stream GEMV.
///
/// # Errors
/// Returns a driver failure.
pub(super) fn launch_salt_v2_quant_act_on(
    stream: &CudaStream,
    function: &CudaFunction,
    input: &CudaSlice<f32>,
    quantized: &mut CudaSlice<i8>,
    group_scale: &mut CudaSlice<f32>,
    groups: u32,
) -> Result<(), BackendError> {
    let warps = 8u32;
    let cfg = LaunchConfig {
        grid_dim: (groups.div_ceil(warps), 1, 1),
        block_dim: (warps * 32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(function);
    launch
        .arg(input)
        .arg(quantized)
        .arg(group_scale)
        .arg(&groups);
    // SAFETY: the caller sized `input`/`quantized` for `groups * 128` values and
    // `group_scale` for `groups`; one warp writes each group once.
    #[allow(unsafe_code)]
    unsafe {
        launch
            .launch(cfg)
            .map(|_| ())
            .map_err(|error| driver_err("launch SALT V2 activation quantizer", &error))
    }
}

/// Launch the A8 row-stream GEMV between device buffers (m = 1).
///
/// # Errors
/// Rejects a tensor the kernel cannot serve, or returns a driver failure.
pub(super) fn launch_salt_v2_stream_i8_on(
    stream: &CudaStream,
    function: &CudaFunction,
    tensor: &SaltV2ResidentTensor,
    quantized: &CudaSlice<i8>,
    group_scale: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
) -> Result<(), BackendError> {
    let table_bytes =
        salt_v2_stream_dispatch(tensor.columns, tensor.scale_group_size, tensor.codec_tag)
            .ok_or_else(|| {
                BackendError::InvalidInput("A8 row-stream GEMV cannot serve this tensor".into())
            })?;
    let narrow = |value: usize, name: &str| {
        u32::try_from(value)
            .map_err(|_| BackendError::InvalidInput(format!("{name} exceeds the u32 kernel ABI")))
    };
    let m = 1u32;
    let n = narrow(tensor.rows, "SALT V2 rows")?;
    let k = narrow(tensor.columns, "SALT V2 columns")?;
    let tile_count = narrow(tensor.tile_count, "SALT V2 tile count")?;
    let plane_count = narrow(tensor.plane_count, "SALT V2 plane count")?;
    let index_metadata = tensor.index_metadata.as_ref().unwrap_or(&tensor.payload);
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(SALT_V2_STREAM_WARPS), 1, 1),
        block_dim: (SALT_V2_STREAM_WARPS * 32, 1, 1),
        shared_mem_bytes: SALT_V2_B3_INT8_TABLE_BYTES + table_bytes * SALT_V2_STREAM_WARPS,
    };
    let mut launch = stream.launch_builder(function);
    launch
        .arg(quantized)
        .arg(group_scale)
        .arg(&tensor.payload)
        .arg(&tensor.scales)
        .arg(index_metadata)
        .arg(output)
        .arg(&m)
        .arg(&n)
        .arg(&k)
        .arg(&tile_count)
        .arg(&plane_count)
        .arg(&tensor.allocation_map_bytes)
        .arg(&tensor.rank_prefix_count)
        .arg(&tensor.terminal_map_value)
        .arg(&table_bytes);
    // SAFETY: validated resident handle; the caller sized the quantized input for
    // `columns` and the output for `rows`.
    #[allow(unsafe_code)]
    unsafe {
        launch
            .launch(cfg)
            .map(|_| ())
            .map_err(|error| driver_err("launch SALT V2 A8 row-stream forward", &error))
    }
}

/// Launch a fused A8 row-stream GEMV over descriptors (m = 1).
///
/// # Errors
/// Returns a driver failure.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_salt_v2_stream_i8_multi_on(
    stream: &CudaStream,
    function: &CudaFunction,
    descriptors: &CudaSlice<u64>,
    tensor_count: u32,
    total_rows: u32,
    k: u32,
    table_bytes: u32,
    quantized: &CudaSlice<i8>,
    group_scale: &CudaSlice<f32>,
) -> Result<(), BackendError> {
    let cfg = LaunchConfig {
        grid_dim: (total_rows.div_ceil(SALT_V2_STREAM_WARPS), 1, 1),
        block_dim: (SALT_V2_STREAM_WARPS * 32, 1, 1),
        shared_mem_bytes: SALT_V2_B3_INT8_TABLE_BYTES + table_bytes * SALT_V2_STREAM_WARPS,
    };
    let mut launch = stream.launch_builder(function);
    launch
        .arg(quantized)
        .arg(group_scale)
        .arg(descriptors)
        .arg(&tensor_count)
        .arg(&total_rows)
        .arg(&k)
        .arg(&table_bytes);
    // SAFETY: as `launch_salt_v2_stream_multi_on`.
    #[allow(unsafe_code)]
    unsafe {
        launch
            .launch(cfg)
            .map(|_| ())
            .map_err(|error| driver_err("launch SALT V2 fused A8 row-stream forward", &error))
    }
}

/// Launch the row-stream GEMV between two device buffers, with no host transfer.
///
/// The resident executor's projection primitive: `input` is `[m, columns]` and
/// `output` `[m, rows]`, both already on the device. The caller has checked that the
/// tensor is one [`salt_v2_stream_dispatch`] accepts and belongs to `stream`'s context.
///
/// # Errors
/// Rejects a tensor the kernel cannot serve or a buffer too small for the shape, or
/// returns a driver failure.
pub(super) fn launch_salt_v2_stream_on(
    stream: &CudaStream,
    function: &CudaFunction,
    tensor: &SaltV2ResidentTensor,
    input: &CudaSlice<f32>,
    m: u32,
    output: &mut CudaSlice<f32>,
) -> Result<(), BackendError> {
    let table_bytes =
        salt_v2_stream_dispatch(tensor.columns, tensor.scale_group_size, tensor.codec_tag)
            .ok_or_else(|| {
                BackendError::InvalidInput(
                    "row-stream GEMV cannot serve this SALT V2 tensor".into(),
                )
            })?;
    let m_usize = m as usize;
    if input.len() < m_usize * tensor.columns || output.len() < m_usize * tensor.rows {
        return Err(BackendError::ShapeMismatch {
            expected: m_usize * tensor.rows,
            got: output.len(),
        });
    }
    let narrow = |value: usize, name: &str| {
        u32::try_from(value)
            .map_err(|_| BackendError::InvalidInput(format!("{name} exceeds the u32 kernel ABI")))
    };
    let n = narrow(tensor.rows, "SALT V2 rows")?;
    let k = narrow(tensor.columns, "SALT V2 columns")?;
    let tile_count = narrow(tensor.tile_count, "SALT V2 tile count")?;
    let plane_count = narrow(tensor.plane_count, "SALT V2 plane count")?;
    let outputs = narrow(m_usize * tensor.rows, "SALT V2 outputs")?;
    let index_metadata = tensor.index_metadata.as_ref().unwrap_or(&tensor.payload);
    let cfg = LaunchConfig {
        grid_dim: (outputs.div_ceil(SALT_V2_STREAM_WARPS), 1, 1),
        block_dim: (SALT_V2_STREAM_WARPS * 32, 1, 1),
        shared_mem_bytes: SALT_V2_B3_TABLE_BYTES + table_bytes * SALT_V2_STREAM_WARPS,
    };
    let mut launch = stream.launch_builder(function);
    launch
        .arg(input)
        .arg(&tensor.payload)
        .arg(&tensor.scales)
        .arg(index_metadata)
        .arg(output)
        .arg(&m)
        .arg(&n)
        .arg(&k)
        .arg(&tile_count)
        .arg(&plane_count)
        .arg(&tensor.allocation_map_bytes)
        .arg(&tensor.rank_prefix_count)
        .arg(&tensor.terminal_map_value)
        .arg(&table_bytes);
    // SAFETY: validated resident handle, buffers checked against the shape above,
    // one write per `[m, rows]` element, whole aligned B3 plane-tiles per the
    // eligibility check.
    #[allow(unsafe_code)]
    unsafe {
        launch
            .launch(cfg)
            .map(|_| ())
            .map_err(|error| driver_err("launch SALT V2 row-stream forward", &error))
    }
}

/// Per-warp shared bytes for the row-streaming kernel, or `None` when it cannot
/// serve the tensor.
///
/// The kernel reads B3 plane-tiles as thirteen 32-bit words and splits each word
/// at the 128-trit scale boundary, so it needs codec B3, scale group 128, and a
/// width that is a whole number of 256-coefficient tiles. Its per-row table maps
/// each of up to three plane-tiles per tile to a one-byte tile index, which caps
/// a row at 256 tiles.
pub(super) fn salt_v2_stream_dispatch(
    columns: usize,
    scale_group_size: u32,
    codec_tag: u32,
) -> Option<u32> {
    const B3: u32 = 1;
    if codec_tag != B3 || scale_group_size != 128 {
        return None;
    }
    if columns == 0 || !columns.is_multiple_of(SALT_V2_ALLOCATION_TILE_SIZE) {
        return None;
    }
    let tiles_per_row = columns / SALT_V2_ALLOCATION_TILE_SIZE;
    if tiles_per_row > 256 {
        return None;
    }
    // Up to three plane-tiles per tile, one byte each, rounded to whole words.
    u32::try_from((tiles_per_row * 3).div_ceil(4) * 4).ok()
}

/// Geometry for the warp-per-row SALT V2 kernel, or `None` when the scalar
/// kernel must handle the shape.
///
/// The warp kernel places group `i` of a row at column `i * scale_group_size`.
/// That holds only when `k` is a whole number of allocation tiles and a scale
/// group divides a tile: otherwise a row's groups straddle tile boundaries and
/// segments come out shorter than one group, which is the variable-stride walk
/// the scalar kernel does. A zero or over-large scale group is rejected outright.
///
/// Returns `(groups_per_row, warps_per_block, slot_bytes_per_warp)`. Slots are
/// the ordered contribution buffer: three per group, because
/// `plane_count_for_tile` yields at most three planes for a tile.
pub(super) fn salt_v2_warp_dispatch(
    columns: usize,
    scale_group_size: u32,
) -> Option<(u32, u32, u32)> {
    let group = usize::try_from(scale_group_size).ok()?;
    if group == 0 || !columns.is_multiple_of(SALT_V2_ALLOCATION_TILE_SIZE) {
        return None;
    }
    if !SALT_V2_ALLOCATION_TILE_SIZE.is_multiple_of(group) {
        return None;
    }
    let groups_per_row = u32::try_from(columns / group).ok()?;
    let slot_bytes = groups_per_row
        .checked_mul(3)?
        .checked_mul(core::mem::size_of::<f32>() as u32)?;
    if slot_bytes == 0 {
        return None;
    }
    // The B3 digit table is block-wide, so it comes out of the budget once
    // rather than per warp.
    let warp_budget = SALT_V2_WARP_SHARED_BYTES.checked_sub(SALT_V2_B3_TABLE_BYTES)?;
    let warps_per_block = (warp_budget / slot_bytes).min(SALT_V2_WARP_MAX_WARPS);
    if warps_per_block == 0 {
        // One warp's slots alone exceed the shared-memory budget.
        return None;
    }
    Some((groups_per_row, warps_per_block, slot_bytes))
}

#[cfg(feature = "device-loss-qualification")]
fn qualification_fatal_driver_error(error: &DriverError) -> bool {
    matches!(
        error.0,
        sys::CUresult::CUDA_ERROR_ILLEGAL_ADDRESS
            | sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED
            | sys::CUresult::CUDA_ERROR_ASSERT
            | sys::CUresult::CUDA_ERROR_HARDWARE_STACK_ERROR
            | sys::CUresult::CUDA_ERROR_ILLEGAL_INSTRUCTION
            | sys::CUresult::CUDA_ERROR_MISALIGNED_ADDRESS
            | sys::CUresult::CUDA_ERROR_INVALID_ADDRESS_SPACE
            | sys::CUresult::CUDA_ERROR_INVALID_PC
            | sys::CUresult::CUDA_ERROR_LAUNCH_FAILED
    )
}

#[cfg(all(test, feature = "device-loss-qualification"))]
mod qualification_error_tests {
    use super::*;

    #[test]
    fn fatal_cuda_execution_error_set_is_complete_and_bounded() {
        for result in [
            sys::CUresult::CUDA_ERROR_ILLEGAL_ADDRESS,
            sys::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED,
            sys::CUresult::CUDA_ERROR_ASSERT,
            sys::CUresult::CUDA_ERROR_HARDWARE_STACK_ERROR,
            sys::CUresult::CUDA_ERROR_ILLEGAL_INSTRUCTION,
            sys::CUresult::CUDA_ERROR_MISALIGNED_ADDRESS,
            sys::CUresult::CUDA_ERROR_INVALID_ADDRESS_SPACE,
            sys::CUresult::CUDA_ERROR_INVALID_PC,
            sys::CUresult::CUDA_ERROR_LAUNCH_FAILED,
        ] {
            assert!(qualification_fatal_driver_error(&DriverError(result)));
        }
        assert!(!qualification_fatal_driver_error(&DriverError(
            sys::CUresult::CUDA_ERROR_INVALID_VALUE
        )));
    }
}

/// Checked requested-device-allocation ledger for one SALT V2 row gather.
///
/// The persistent component is the original encoded tensor. Per-call bytes are
/// only the selected row IDs and reconstructed output; no full dense table or
/// other dense weight shadow is created.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SaltV2GatherReceipt {
    resident: SaltV2ResidentAllocationReceipt,
    row_index_bytes: u64,
    output_bytes: u64,
    peak_resident_bytes: u64,
}

impl SaltV2GatherReceipt {
    fn new(
        resident: SaltV2ResidentAllocationReceipt,
        selected_rows: usize,
        output_elements: usize,
    ) -> Result<Self, BackendError> {
        let checked_bytes = |elements: usize, element_bytes: usize, field: &str| {
            elements
                .checked_mul(element_bytes)
                .and_then(|value| u64::try_from(value).ok())
                .ok_or_else(|| {
                    BackendError::InvalidInput(format!(
                        "SALT V2 gather {field} byte count overflows u64"
                    ))
                })
        };
        let row_index_bytes =
            checked_bytes(selected_rows, core::mem::size_of::<u32>(), "row-index")?;
        let output_bytes = checked_bytes(output_elements, core::mem::size_of::<f32>(), "output")?;
        let peak_resident_bytes = resident
            .steady_resident_bytes()
            .checked_add(row_index_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                BackendError::InvalidInput(
                    "SALT V2 gather peak resident byte count overflows u64".into(),
                )
            })?;
        Ok(Self {
            resident,
            row_index_bytes,
            output_bytes,
            peak_resident_bytes,
        })
    }

    /// Persistent encoded-weight and compact-index bytes used by this launch.
    #[must_use]
    pub fn resident_allocation(self) -> SaltV2ResidentAllocationReceipt {
        self.resident
    }

    /// Persistent encoded-weight and compact-index bytes.
    #[must_use]
    pub fn steady_resident_bytes(self) -> u64 {
        self.resident.steady_resident_bytes()
    }

    /// Per-call selected-row index bytes uploaded to the device.
    #[must_use]
    pub fn row_index_bytes(self) -> u64 {
        self.row_index_bytes
    }

    /// Per-call reconstructed output bytes allocated on the device.
    #[must_use]
    pub fn output_bytes(self) -> u64 {
        self.output_bytes
    }

    /// Dense dequantized weight bytes, always zero.
    #[must_use]
    pub fn dense_weight_bytes(self) -> u64 {
        0
    }

    /// Persistent bytes plus row-index and output allocations live at launch.
    #[must_use]
    pub fn peak_resident_bytes(self) -> u64 {
        self.peak_resident_bytes
    }
}

impl CudaBackend {
    #[cfg(feature = "device-loss-qualification")]
    fn maybe_poison_context_for_qualification(&self) -> Result<(), BackendError> {
        if !take_destructive_context_loss_qualification_request() {
            return Ok(());
        }
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&self.func_qualification_poison);
        #[allow(unsafe_code)]
        // SAFETY: qualification kernel has no parameters and exactly one thread.
        // Its intentional `trap` is destructive to this CUDA context, not host
        // memory. Caller must replace the serving process after this returns.
        unsafe { launch.launch(cfg) }.map_err(|error| {
            BackendError::Backend(format!(
                "destructive CUDA context-loss qualification trap did not launch: {error}"
            ))
        })?;
        let failure = match self.stream.synchronize() {
            Ok(()) => {
                return Err(BackendError::Backend(
                    "destructive CUDA context-loss qualification trap returned CUDA success".into(),
                ));
            }
            Err(error) => error,
        };
        if !qualification_fatal_driver_error(&failure) {
            return Err(BackendError::Backend(format!(
                "destructive CUDA context-loss qualification observed non-fatal sync error: {failure}"
            )));
        }
        let follow_up = match self.stream.synchronize() {
            Ok(()) => {
                return Err(BackendError::Backend(
                    "destructive CUDA context-loss qualification context accepted follow-up synchronization"
                        .into(),
                ));
            }
            Err(error) => error,
        };
        if !qualification_fatal_driver_error(&follow_up) {
            return Err(BackendError::Backend(format!(
                "destructive CUDA context-loss qualification follow-up was not sticky: {follow_up}"
            )));
        }
        Err(BackendError::Backend(format!(
            "destructive CUDA context-loss qualification observed sticky driver failure: initial={failure}; follow_up={follow_up}"
        )))
    }

    pub(super) fn validate_salt_v2_resident_context(
        &self,
        tensor: &SaltV2ResidentTensor,
    ) -> Result<(), BackendError> {
        if !self.same_context(&tensor.payload)
            || !self.same_context(&tensor.scales)
            || tensor
                .index_metadata
                .as_ref()
                .is_some_and(|metadata| !self.same_context(metadata))
        {
            return Err(BackendError::InvalidInput(
                "SALT V2 resident tensor belongs to a different CUDA context".into(),
            ));
        }
        Ok(())
    }

    /// Execute the deterministic SALT V2 projection into caller-owned host memory.
    ///
    /// Model runners can reuse the published output slice. Results are downloaded
    /// into private host staging and validated before publication, so `output`
    /// remains unchanged on every error. The encoded tensor remains resident
    /// without a dense shadow; transient device activation/output allocations are
    /// reported by the returned receipt.
    ///
    /// # Errors
    /// Returns the errors documented by [`Self::salt_v2_forward_exact`] and a
    /// [`BackendError::ShapeMismatch`] unless `output` is exactly `[M, N]`.
    pub fn salt_v2_forward_exact_into(
        &self,
        tensor: &SaltV2ResidentTensor,
        activation: &[f32],
        m: usize,
        output: &mut [f32],
    ) -> Result<SaltV2ForwardReceipt, BackendError> {
        let (receipt, output_elements) = self.salt_v2_forward_preflight(
            tensor,
            activation,
            m,
            Some(output.len()),
            SaltV2ForwardMode::Exact,
        )?;
        let staged =
            self.salt_v2_forward_launch(tensor, activation, m, output_elements, receipt)?;
        output.copy_from_slice(&staged);
        Ok(receipt)
    }

    /// Execute the fast SALT V2 projection into caller-owned host memory.
    ///
    /// As [`Self::salt_v2_forward_exact_into`], but reduced by warp shuffle
    /// rather than by replaying the scalar kernel's addition order, so results
    /// are close to the CPU reference rather than equal to it. The returned
    /// receipt names the kernel that actually ran: a shape the fast kernel
    /// cannot serve answers with the exact image and says
    /// [`SaltV2ForwardMode::FastAliasesExact`].
    ///
    /// # Errors
    /// Returns the errors documented by [`Self::salt_v2_forward_exact_into`].
    pub fn salt_v2_forward_fast_into(
        &self,
        tensor: &SaltV2ResidentTensor,
        activation: &[f32],
        m: usize,
        output: &mut [f32],
    ) -> Result<SaltV2ForwardReceipt, BackendError> {
        let (receipt, output_elements) = self.salt_v2_forward_preflight(
            tensor,
            activation,
            m,
            Some(output.len()),
            SaltV2ForwardMode::FastWarpReduce,
        )?;
        let staged =
            self.salt_v2_forward_launch(tensor, activation, m, output_elements, receipt)?;
        output.copy_from_slice(&staged);
        Ok(receipt)
    }

    /// A8 row-stream projection with host input and output, for gates and probes.
    ///
    /// Quantizes `activation` on the device to int8 with one scale per 128-wide
    /// group (`q = round_ties_even(x * 127 / absmax)`, scale `absmax / 127`) and runs
    /// the dp4a row-stream GEMV. Returns the output and the device's quantized
    /// activations and scales, so a caller can build an exact reference from the
    /// same inputs the kernel saw. `m` must be 1.
    ///
    /// # Errors
    /// Rejects a tensor the kernel cannot serve or `m != 1`, or returns a driver
    /// failure.
    #[allow(clippy::type_complexity)]
    pub fn salt_v2_forward_a8_probe(
        &self,
        tensor: &SaltV2ResidentTensor,
        activation: &[f32],
        m: usize,
    ) -> Result<(Vec<f32>, Vec<i8>, Vec<f32>), BackendError> {
        self.validate_salt_v2_resident_context(tensor)?;
        if m != 1 || activation.len() != tensor.columns || !tensor.columns.is_multiple_of(128) {
            return Err(BackendError::ShapeMismatch {
                expected: tensor.columns,
                got: activation.len(),
            });
        }
        let groups = tensor.columns / 128;
        let input = self
            .stream
            .clone_htod(activation)
            .map_err(|error| driver_err("upload A8 probe activation", &error))?;
        let mut quantized = self
            .stream
            .alloc_zeros::<i8>(tensor.columns)
            .map_err(|error| driver_err("allocate A8 probe activations", &error))?;
        let mut scale = self
            .stream
            .alloc_zeros::<f32>(groups)
            .map_err(|error| driver_err("allocate A8 probe scales", &error))?;
        let mut output = self
            .stream
            .alloc_zeros::<f32>(tensor.rows)
            .map_err(|error| driver_err("allocate A8 probe output", &error))?;
        launch_salt_v2_quant_act_on(
            &self.stream,
            &self.func_salt_v2_quant_act,
            &input,
            &mut quantized,
            &mut scale,
            u32::try_from(groups).map_err(|_| {
                BackendError::InvalidInput("A8 probe group count exceeds u32".into())
            })?,
        )?;
        launch_salt_v2_stream_i8_on(
            &self.stream,
            &self.func_salt_v2_stream_i8,
            tensor,
            &quantized,
            &scale,
            &mut output,
        )?;
        let mut host_output = vec![0.0f32; tensor.rows];
        let mut host_quantized = vec![0i8; tensor.columns];
        let mut host_scale = vec![0.0f32; groups];
        self.stream
            .memcpy_dtoh(&output, &mut host_output)
            .map_err(|error| driver_err("read A8 probe output", &error))?;
        self.stream
            .memcpy_dtoh(&quantized, &mut host_quantized)
            .map_err(|error| driver_err("read A8 probe activations", &error))?;
        self.stream
            .memcpy_dtoh(&scale, &mut host_scale)
            .map_err(|error| driver_err("read A8 probe scales", &error))?;
        Ok((host_output, host_quantized, host_scale))
    }

    /// Whether the warp kernels can serve this tensor's geometry.
    ///
    /// Mirrors the launch's own choice, including the tiled opt-in, so the
    /// receipt cannot claim a kernel the launch will not run.
    fn salt_v2_warp_eligible(&self, tensor: &SaltV2ResidentTensor) -> bool {
        let tiled = env_flag_on("TRITIUM_SALT_V2_TILED") && tensor.columns.is_multiple_of(256);
        !tiled && salt_v2_warp_dispatch(tensor.columns, tensor.scale_group_size).is_some()
    }

    pub(super) fn salt_v2_forward_preflight(
        &self,
        tensor: &SaltV2ResidentTensor,
        activation: &[f32],
        m: usize,
        output_len: Option<usize>,
        mode: SaltV2ForwardMode,
    ) -> Result<(SaltV2ForwardReceipt, usize), BackendError> {
        self.validate_salt_v2_resident_context(tensor)?;
        let activation_elements = m.checked_mul(tensor.columns).ok_or_else(|| {
            BackendError::InvalidInput("SALT V2 activation length overflows usize".into())
        })?;
        if activation.len() != activation_elements {
            return Err(BackendError::ShapeMismatch {
                expected: activation_elements,
                got: activation.len(),
            });
        }
        if let Some((index, value)) = activation
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(BackendError::InvalidInput(format!(
                "SALT V2 activation {index} is non-finite ({:#010x})",
                value.to_bits()
            )));
        }
        let output_elements = m.checked_mul(tensor.rows).ok_or_else(|| {
            BackendError::InvalidInput("SALT V2 output length overflows usize".into())
        })?;
        if let Some(got) = output_len
            && got != output_elements
        {
            return Err(BackendError::ShapeMismatch {
                expected: output_elements,
                got,
            });
        }
        // Label the receipt with the mode that will actually run. The fast kernel
        // is a variant of the warp kernel, so it serves exactly the shapes the
        // warp kernel serves; anything else answers with the exact image, and a
        // receipt that claimed otherwise would be the only record a caller has.
        // The fast entry point takes the best fast kernel the geometry allows:
        // the row-streaming GEMV where it applies (opt out with
        // TRITIUM_SALT_V2_STREAM=0), else the warp kernel's shuffle variant, else
        // the exact image under its own name.
        let resolved = match mode {
            SaltV2ForwardMode::FastWarpReduce => {
                let stream = env_flag_default_on("TRITIUM_SALT_V2_STREAM")
                    && salt_v2_stream_dispatch(
                        tensor.columns,
                        tensor.scale_group_size,
                        tensor.codec_tag,
                    )
                    .is_some();
                if stream {
                    SaltV2ForwardMode::FastRowStream
                } else if self.salt_v2_warp_eligible(tensor) {
                    SaltV2ForwardMode::FastWarpReduce
                } else {
                    SaltV2ForwardMode::FastAliasesExact
                }
            }
            other => other,
        };
        let receipt = SaltV2ForwardReceipt::new(
            resolved,
            tensor.receipt,
            activation_elements,
            output_elements,
        )?;
        Ok((receipt, output_elements))
    }

    pub(super) fn salt_v2_forward_launch(
        &self,
        tensor: &SaltV2ResidentTensor,
        activation: &[f32],
        m: usize,
        output_elements: usize,
        receipt: SaltV2ForwardReceipt,
    ) -> Result<Vec<f32>, BackendError> {
        #[cfg(feature = "device-loss-qualification")]
        self.maybe_poison_context_for_qualification()?;
        if output_elements == 0 {
            return Ok(Vec::new());
        }

        let m_u32 = u32::try_from(m).map_err(|_| {
            BackendError::InvalidInput("SALT V2 batch rows exceed the u32 kernel ABI".into())
        })?;
        let n_u32 = u32::try_from(tensor.rows).map_err(|_| {
            BackendError::InvalidInput("SALT V2 output rows exceed the u32 kernel ABI".into())
        })?;
        let k_u32 = u32::try_from(tensor.columns).map_err(|_| {
            BackendError::InvalidInput("SALT V2 columns exceed the u32 kernel ABI".into())
        })?;
        let tile_count = u32::try_from(tensor.tile_count).map_err(|_| {
            BackendError::InvalidInput("SALT V2 tile count exceeds the u32 kernel ABI".into())
        })?;
        let plane_count = u32::try_from(tensor.plane_count).map_err(|_| {
            BackendError::InvalidInput("SALT V2 plane count exceeds the u32 kernel ABI".into())
        })?;
        let total_outputs = u32::try_from(output_elements).map_err(|_| {
            BackendError::InvalidInput("SALT V2 output elements exceed the u32 launch grid".into())
        })?;
        let payload_bytes = tensor.receipt.payload_bytes();
        let scale_count = tensor.receipt.scale_bytes() / core::mem::size_of::<u16>() as u64;
        let index_metadata = tensor.index_metadata.as_ref().unwrap_or(&tensor.payload);
        let activation_bytes = usize::try_from(receipt.activation_bytes()).map_err(|_| {
            BackendError::InvalidInput("SALT V2 activation bytes exceed host usize".into())
        })?;
        let output_bytes = usize::try_from(receipt.output_bytes()).map_err(|_| {
            BackendError::InvalidInput("SALT V2 output bytes exceed host usize".into())
        })?;

        // Qwen prompt/decode executes hundreds of projections per request. Keep
        // transient activation/output allocations in a small backend-owned pool;
        // this removes repeated device allocation/free synchronization while
        // preserving exact launch and readback semantics.
        let mut workspace = self
            .salt_v2_workspace
            .lock()
            .map_err(|_| BackendError::Backend("SALT V2 workspace mutex poisoned".to_owned()))?;
        let mut d_activation = workspace
            .take(&self.stream, activation.len())
            .map_err(|error| {
                alloc_or_backend("allocate SALT V2 activation", &error, activation_bytes)
            })?;
        self.stream
            .memcpy_htod(activation, &mut d_activation)
            .map_err(|error| driver_err("upload SALT V2 activation", &error))?;
        let mut d_output = workspace
            .take(&self.stream, output_elements)
            .map_err(|error| alloc_or_backend("allocate SALT V2 output", &error, output_bytes))?;
        if receipt.mode() == SaltV2ForwardMode::FastRowStream {
            let table_bytes =
                salt_v2_stream_dispatch(tensor.columns, tensor.scale_group_size, tensor.codec_tag)
                    .ok_or_else(|| {
                        BackendError::InvalidInput(
                            "SALT V2 row-stream receipt on a tensor the kernel cannot serve".into(),
                        )
                    })?;
            let cfg = LaunchConfig {
                grid_dim: (total_outputs.div_ceil(SALT_V2_STREAM_WARPS), 1, 1),
                block_dim: (SALT_V2_STREAM_WARPS * 32, 1, 1),
                shared_mem_bytes: SALT_V2_B3_TABLE_BYTES + table_bytes * SALT_V2_STREAM_WARPS,
            };
            let mut launch = self.stream.launch_builder(&self.func_salt_v2_stream);
            launch
                .arg(&d_activation)
                .arg(&tensor.payload)
                .arg(&tensor.scales)
                .arg(index_metadata)
                .arg(&mut d_output)
                .arg(&m_u32)
                .arg(&n_u32)
                .arg(&k_u32)
                .arg(&tile_count)
                .arg(&plane_count)
                .arg(&tensor.allocation_map_bytes)
                .arg(&tensor.rank_prefix_count)
                .arg(&tensor.terminal_map_value)
                .arg(&table_bytes);
            // SAFETY: as for the other SALT V2 kernels -- validated resident
            // handle, checked operand lengths, one write per `[M, N]` element.
            // The kernel reads whole 52-byte B3 plane-tiles as aligned words,
            // which the eligibility check guarantees.
            #[allow(unsafe_code)]
            unsafe {
                launch
                    .launch(cfg)
                    .map_err(|error| driver_err("launch SALT V2 row-stream forward", &error))?;
            }
        } else {
            // Qwen's hidden/intermediate widths are 256-aligned. For those
            // matrices, stage each activation tile once per output-row block;
            // irregular shapes retain scalar exact dispatch.
            // Experimental until a broad shape sweep proves a win. The current
            // 4090 benchmark favors scalar dispatch for short prompts; keep this
            // opt-in so production latency never regresses by default.
            let use_tiled =
                env_flag_on("TRITIUM_SALT_V2_TILED") && tensor.columns.is_multiple_of(256);

            // Warp-per-row dispatch; see `salt_v2_warp_dispatch`.
            let warp_dispatch = salt_v2_warp_dispatch(tensor.columns, tensor.scale_group_size);
            let (warp_groups_per_row, warps_per_block, warp_slot_bytes) =
                warp_dispatch.unwrap_or((0, 0, 0));
            let use_warp = !use_tiled && warps_per_block > 0;
            let warp_groups_u32 = warp_groups_per_row;
            // The fast kernel is a variant of the warp kernel, so it serves exactly
            // the shapes the warp kernel serves. Anything else keeps the exact
            // image, and the receipt already says `FastAliasesExact` for that.
            let use_fast = use_warp && receipt.mode() == SaltV2ForwardMode::FastWarpReduce;

            let (grid_x, grid_y, block_x, shared_mem_bytes) = if use_warp {
                (
                    total_outputs.div_ceil(warps_per_block),
                    1,
                    warps_per_block * 32,
                    // The fast kernel keeps no contribution slots; only the table.
                    if use_fast {
                        SALT_V2_B3_TABLE_BYTES
                    } else {
                        SALT_V2_B3_TABLE_BYTES + warp_slot_bytes * warps_per_block
                    },
                )
            } else if use_tiled {
                (
                    n_u32.div_ceil(SALT_V2_TILED_THREADS),
                    m_u32,
                    SALT_V2_TILED_THREADS,
                    256 * core::mem::size_of::<f32>() as u32,
                )
            } else {
                (
                    total_outputs.div_ceil(THREADS_PER_BLOCK),
                    1,
                    THREADS_PER_BLOCK,
                    0,
                )
            };
            let cfg = LaunchConfig {
                grid_dim: (grid_x, grid_y, 1),
                block_dim: (block_x, 1, 1),
                shared_mem_bytes,
            };
            let kernel = if use_fast {
                &self.func_salt_v2_warp_fast
            } else if use_warp {
                &self.func_salt_v2_warp
            } else if use_tiled {
                &self.func_salt_v2_tiled
            } else {
                &self.func_salt_v2_exact
            };
            let mut launch = self.stream.launch_builder(kernel);
            launch
                .arg(&d_activation)
                .arg(&tensor.payload)
                .arg(&tensor.scales)
                .arg(index_metadata)
                .arg(&mut d_output)
                .arg(&m_u32)
                .arg(&n_u32)
                .arg(&k_u32)
                .arg(&tensor.codec_tag)
                .arg(&tensor.scale_group_size)
                .arg(&tile_count)
                .arg(&plane_count)
                .arg(&payload_bytes)
                .arg(&scale_count)
                .arg(&tensor.allocation_map_bytes)
                .arg(&tensor.rank_prefix_count)
                .arg(&tensor.terminal_map_value);
            // Only the warp kernel takes the group count; the other two derive their
            // own geometry from `k`.
            if use_warp {
                launch.arg(&warp_groups_u32);
            }
            // SAFETY: the private handle owns codec payload/scales/index metadata
            // validated at upload. Input/output lengths and every scalar ABI bound
            // are checked above, and the kernel writes each `[M, N]` element once.
            #[allow(unsafe_code)]
            unsafe {
                launch
                    .launch(cfg)
                    .map_err(|error| driver_err("launch SALT V2 exact forward", &error))?;
            }
        }
        let mut staged = Vec::new();
        staged.try_reserve_exact(output_elements).map_err(|error| {
            BackendError::Backend(format!(
                "allocate SALT V2 host output staging for {output_elements} f32 values: {error}"
            ))
        })?;
        staged.resize(output_elements, 0.0f32);
        {
            let d_output_view = d_output.slice(..output_elements);
            self.stream
                .memcpy_dtoh(&d_output_view, &mut staged)
                .map_err(|error| driver_err("download SALT V2 output", &error))?;
        }
        workspace.put(d_activation);
        workspace.put(d_output);
        if let Some((index, value)) = staged
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(BackendError::InvalidInput(format!(
                "SALT V2 output {index} is non-finite ({:#010x})",
                value.to_bits()
            )));
        }
        Ok(staged)
    }

    /// Reconstruct selected semantic rows into caller-owned host memory.
    ///
    /// `rows` is ordered and may contain duplicates, matching token-embedding
    /// gather semantics. The kernel reads D2/B3/S34 payloads and declared-group scales
    /// directly. It never creates or retains the full dense table. Private host
    /// staging makes publication transactional: `output` is unchanged on error.
    ///
    /// # Errors
    /// Rejects an output length other than `rows.len() * tensor.columns()`, any
    /// row outside `[0, tensor.rows())`, a foreign CUDA context, launch-bound
    /// overflow, non-finite reconstructed output, or a CUDA driver failure.
    pub fn salt_v2_gather_rows(
        &self,
        tensor: &SaltV2ResidentTensor,
        rows: &[u32],
        output: &mut [f32],
    ) -> Result<SaltV2GatherReceipt, BackendError> {
        #[cfg(feature = "device-loss-qualification")]
        self.maybe_poison_context_for_qualification()?;
        self.validate_salt_v2_resident_context(tensor)?;
        let output_elements = rows.len().checked_mul(tensor.columns).ok_or_else(|| {
            BackendError::InvalidInput("SALT V2 gather output length overflows usize".into())
        })?;
        if output.len() != output_elements {
            return Err(BackendError::ShapeMismatch {
                expected: output_elements,
                got: output.len(),
            });
        }
        if let Some((selection, row)) = rows
            .iter()
            .copied()
            .enumerate()
            .find(|(_, row)| *row as u64 >= tensor.rows as u64)
        {
            return Err(BackendError::InvalidInput(format!(
                "SALT V2 gather row {selection} is {row}, but the tensor has {} rows",
                tensor.rows
            )));
        }
        let receipt = SaltV2GatherReceipt::new(tensor.receipt, rows.len(), output_elements)?;
        if output_elements == 0 {
            return Ok(receipt);
        }

        let selected_rows = u32::try_from(rows.len()).map_err(|_| {
            BackendError::InvalidInput("SALT V2 selected row count exceeds the u32 ABI".into())
        })?;
        let n_u32 = u32::try_from(tensor.rows).map_err(|_| {
            BackendError::InvalidInput("SALT V2 row count exceeds the u32 kernel ABI".into())
        })?;
        let k_u32 = u32::try_from(tensor.columns).map_err(|_| {
            BackendError::InvalidInput("SALT V2 columns exceed the u32 kernel ABI".into())
        })?;
        let tile_count = u32::try_from(tensor.tile_count).map_err(|_| {
            BackendError::InvalidInput("SALT V2 tile count exceeds the u32 kernel ABI".into())
        })?;
        let plane_count = u32::try_from(tensor.plane_count).map_err(|_| {
            BackendError::InvalidInput("SALT V2 plane count exceeds the u32 kernel ABI".into())
        })?;
        let total_outputs = u32::try_from(output_elements).map_err(|_| {
            BackendError::InvalidInput(
                "SALT V2 gather output elements exceed the u32 launch grid".into(),
            )
        })?;
        let payload_bytes = tensor.receipt.payload_bytes();
        let scale_count = tensor.receipt.scale_bytes() / core::mem::size_of::<u16>() as u64;
        let index_metadata = tensor.index_metadata.as_ref().unwrap_or(&tensor.payload);
        let row_index_bytes = usize::try_from(receipt.row_index_bytes()).map_err(|_| {
            BackendError::InvalidInput("SALT V2 row-index bytes exceed host usize".into())
        })?;
        let output_bytes = usize::try_from(receipt.output_bytes()).map_err(|_| {
            BackendError::InvalidInput("SALT V2 gather output bytes exceed host usize".into())
        })?;

        let d_rows = self.stream.clone_htod(rows).map_err(|error| {
            alloc_or_backend("upload SALT V2 row indices", &error, row_index_bytes)
        })?;
        let mut d_output = self
            .stream
            .alloc_zeros::<f32>(output_elements)
            .map_err(|error| {
                alloc_or_backend("allocate SALT V2 gather output", &error, output_bytes)
            })?;
        let cfg = LaunchConfig {
            grid_dim: (total_outputs.div_ceil(THREADS_PER_BLOCK), 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&self.func_salt_v2_gather);
        launch
            .arg(&tensor.payload)
            .arg(&tensor.scales)
            .arg(index_metadata)
            .arg(&d_rows)
            .arg(&mut d_output)
            .arg(&selected_rows)
            .arg(&n_u32)
            .arg(&k_u32)
            .arg(&tensor.codec_tag)
            .arg(&tensor.scale_group_size)
            .arg(&tile_count)
            .arg(&plane_count)
            .arg(&payload_bytes)
            .arg(&scale_count)
            .arg(&tensor.allocation_map_bytes)
            .arg(&tensor.rank_prefix_count)
            .arg(&tensor.terminal_map_value);
        // SAFETY: row IDs are bounds-checked before upload; the private handle
        // owns validated payload/scales/index metadata, output is exactly
        // `selected_rows * K`, and every scalar matches the kernel's u32/u64 ABI.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|error| driver_err("launch SALT V2 row gather", &error))?;
        }
        let mut staged = vec![0.0f32; output_elements];
        self.stream
            .memcpy_dtoh(&d_output, &mut staged)
            .map_err(|error| driver_err("download SALT V2 gathered rows", &error))?;
        if let Some((index, value)) = staged
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(BackendError::InvalidInput(format!(
                "SALT V2 gathered output {index} is non-finite ({:#010x})",
                value.to_bits()
            )));
        }
        output.copy_from_slice(&staged);
        Ok(receipt)
    }
}

#[cfg(test)]
mod warp_dispatch_tests {
    use super::salt_v2_warp_dispatch;

    #[test]
    fn the_warp_parity_gate_actually_reaches_the_warp_kernel() {
        // `tests/salt_v2_warp.rs` builds a 512-column, 64-group tensor. If this
        // shape were ineligible that file would silently be testing the scalar
        // kernel a second time and asserting nothing about the warp kernel.
        let (groups, warps, slot_bytes) =
            salt_v2_warp_dispatch(512, 64).expect("512 columns at group 64 must use the warp path");
        assert_eq!(groups, 8);
        assert_eq!(slot_bytes, 8 * 3 * 4);
        assert_eq!(warps, super::SALT_V2_WARP_MAX_WARPS);
    }

    #[test]
    fn shapes_that_straddle_a_tile_stay_on_the_scalar_kernel() {
        // 576 is the width the other SALT tests use: 2.25 allocation tiles, so a
        // row's groups do not start at `i * scale_group_size` and the warp
        // kernel's precondition fails.
        assert!(salt_v2_warp_dispatch(576, 64).is_none());
        // A scale group that does not divide a 256-coefficient tile.
        assert!(salt_v2_warp_dispatch(512, 96).is_none());
        // Degenerate widths.
        assert!(salt_v2_warp_dispatch(512, 0).is_none());
        assert!(salt_v2_warp_dispatch(0, 64).is_none());
    }

    #[test]
    fn a_row_too_wide_for_one_warps_slots_falls_back() {
        // Slots are 3 floats per group, and the block-wide B3 digit table takes
        // 512 B off the 48 KiB before any warp gets a share: 48640 / 12 = 4053
        // groups. A width must also be a whole number of 256-coefficient tiles,
        // which at group 64 means a multiple of 4 groups, so 4052 fits and the
        // next admissible width, 4056, does not.
        assert!(salt_v2_warp_dispatch(4052 * 64, 64).is_some());
        assert!(salt_v2_warp_dispatch(4056 * 64, 64).is_none());
    }
}
