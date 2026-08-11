use super::*;

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

    fn validate_salt_v2_resident_context(
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
        let receipt =
            SaltV2ForwardReceipt::new(mode, tensor.receipt, activation_elements, output_elements)?;
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

        let d_activation = self.stream.clone_htod(activation).map_err(|error| {
            alloc_or_backend("upload SALT V2 activation", &error, activation_bytes)
        })?;
        let mut d_output = self
            .stream
            .alloc_zeros::<f32>(output_elements)
            .map_err(|error| alloc_or_backend("allocate SALT V2 output", &error, output_bytes))?;
        let cfg = LaunchConfig {
            grid_dim: (total_outputs.div_ceil(THREADS_PER_BLOCK), 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&self.func_salt_v2_exact);
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
        // SAFETY: the private handle owns codec payload/scales/index metadata
        // validated at upload. Input/output lengths and every scalar ABI bound
        // are checked above, and the kernel writes each `[M, N]` element once.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|error| driver_err("launch SALT V2 exact forward", &error))?;
        }
        let mut staged = Vec::new();
        staged.try_reserve_exact(output_elements).map_err(|error| {
            BackendError::Backend(format!(
                "allocate SALT V2 host output staging for {output_elements} f32 values: {error}"
            ))
        })?;
        staged.resize(output_elements, 0.0f32);
        self.stream
            .memcpy_dtoh(&d_output, &mut staged)
            .map_err(|error| driver_err("download SALT V2 output", &error))?;
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
