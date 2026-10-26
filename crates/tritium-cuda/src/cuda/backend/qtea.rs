//! Direct checked-wire CUDA execution. No dense shadow or fallback (ADR0172).
use super::*;
use tritium_format::qtea::{QteaExecutionLayout, QteaTensorView};

/// Positive per-call logical staging caps, not whole-model/physical admission.
#[derive(Clone, Copy, Debug)]
pub struct QteaCudaLimits {
    /// Sum of requested global device input/index and output allocation bytes.
    pub max_device_staging_bytes: u64,
    /// Actual element capacity bytes of the private output Vec.
    pub max_host_staging_bytes: usize,
}

/// Logical owned/requested bytes, NOT physical memory or a release receipt.
/// Excludes driver/context/module/events, shared memory, allocator overhead,
/// caller buffers, other handles and allocations. No pooled staging is retained.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct QteaCudaAllocationLedger {
    /// Canonical checked wire bytes resident in this one tensor handle.
    pub encoded_resident_bytes: u64,
    /// Per-call device activations (matmul) or u32 row indices (gather).
    pub input_or_index_bytes: u64,
    /// Per-call device output allocation request.
    pub output_bytes: u64,
    /// Actual private host output Vec element capacity bytes.
    pub host_staging_capacity_bytes: usize,
    /// Encoded residency plus per-call input/index and output requests.
    pub requested_device_peak_bytes: u64,
    /// Always zero: no auxiliary reconstructed weight shadow is allocated.
    pub dense_weight_bytes: u64,
}

/// Immutable canonical QTEA wire bytes, retained without a dense weight shadow.
/// Private geometry/layout are derived only from a format-checked view.
#[derive(Debug)]
pub struct QteaResidentTensor {
    bytes: CudaSlice<u8>,
    geometry: Geometry,
    content_digest: [u8; 32],
    kernels: Arc<QteaKernels>,
}
impl QteaResidentTensor {
    /// Exact canonical encoded allocation request, including header/trailer.
    pub fn encoded_bytes(&self) -> usize {
        self.bytes.len()
    }
    /// Original weight geometry, not an execution or quality claim.
    pub fn shape(&self) -> (usize, usize) {
        (self.geometry.rows as usize, self.geometry.columns as usize)
    }
    /// Domain-separated wire integrity identity, not source authentication.
    pub fn content_digest(&self) -> [u8; 32] {
        self.content_digest
    }
}

#[derive(Debug)]
pub(super) struct QteaKernels {
    _module: Arc<CudaModule>,
    matmul: CudaFunction,
    gather: CudaFunction,
}
#[derive(Clone, Copy, Debug)]
struct Geometry {
    rows: u32,
    columns: u32,
    group: u32,
    residual_count: u32,
    trits: u64,
    params: u64,
    multipliers: u64,
    residuals: u64,
    record_bytes: u64,
    position_bytes: u64,
}
impl Geometry {
    fn checked(view: QteaTensorView<'_>) -> Result<Self, BackendError> {
        let meta = view.metadata();
        let QteaExecutionLayout {
            trits_offset,
            group_parameters_offset,
            column_multipliers_offset,
            residuals_offset,
            residual_record_bytes,
            residual_position_bytes,
            residual_count,
            ..
        } = view.execution_layout();
        let word = |x| u32::try_from(x).map_err(|_| invalid("QTEA geometry exceeds u32"));
        let offset = |x| u64::try_from(x).map_err(|_| invalid("QTEA offset exceeds u64"));
        Ok(Self {
            rows: word(meta.rows)?,
            columns: word(meta.columns)?,
            group: word(meta.recipe.group_size)?,
            residual_count: word(residual_count)?,
            trits: offset(trits_offset)?,
            params: offset(group_parameters_offset)?,
            multipliers: offset(column_multipliers_offset)?,
            residuals: offset(residuals_offset)?,
            record_bytes: offset(residual_record_bytes)?,
            position_bytes: offset(residual_position_bytes)?,
        })
    }
}

fn invalid(message: &str) -> BackendError {
    BackendError::InvalidInput(message.into())
}
fn byte_count(elements: usize) -> Result<usize, BackendError> {
    elements
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| invalid("QTEA staging byte count overflow"))
}
fn plan(
    resident: usize,
    input_elements: usize,
    output_elements: usize,
    limits: QteaCudaLimits,
) -> Result<QteaCudaAllocationLedger, BackendError> {
    if limits.max_device_staging_bytes == 0 || limits.max_host_staging_bytes == 0 {
        return Err(invalid("QTEA staging limits must be positive"));
    }
    let input = u64::try_from(byte_count(input_elements)?)
        .map_err(|_| invalid("QTEA input bytes exceed u64"))?;
    let output = byte_count(output_elements)?;
    let device_output =
        u64::try_from(output).map_err(|_| invalid("QTEA output bytes exceed u64"))?;
    let staging = input
        .checked_add(device_output)
        .ok_or_else(|| invalid("QTEA device staging overflow"))?;
    if staging > limits.max_device_staging_bytes || output > limits.max_host_staging_bytes {
        return Err(invalid("QTEA staging exceeds caller limit"));
    }
    let resident =
        u64::try_from(resident).map_err(|_| invalid("QTEA encoded residency exceeds u64"))?;
    Ok(QteaCudaAllocationLedger {
        encoded_resident_bytes: resident,
        input_or_index_bytes: input,
        output_bytes: device_output,
        host_staging_capacity_bytes: output,
        requested_device_peak_bytes: resident
            .checked_add(staging)
            .ok_or_else(|| invalid("QTEA requested peak overflow"))?,
        dense_weight_bytes: 0,
    })
}
fn stage(elements: usize, cap: usize) -> Result<Vec<f32>, BackendError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(elements)
        .map_err(|_| BackendError::OutOfMemory {
            requested: byte_count(elements).unwrap_or(usize::MAX),
        })?;
    if byte_count(output.capacity())? > cap {
        return Err(invalid("QTEA host staging capacity exceeds caller limit"));
    }
    output.resize(elements, 0.0);
    Ok(output)
}

impl CudaBackend {
    fn qtea_kernels(&self) -> Result<Arc<QteaKernels>, BackendError> {
        let mut cache = self
            .qtea_kernels
            .lock()
            .map_err(|_| invalid("QTEA kernel cache poisoned"))?;
        if let Some(kernels) = &*cache {
            return Ok(Arc::clone(kernels));
        }
        let module = self
            .stream
            .context()
            .load_module(Ptx::from_src(include_str!(concat!(
                env!("OUT_DIR"),
                "/qtea.ptx"
            ))))
            .map_err(|e| driver_err("load QTEA PTX", &e))?;
        let kernels = Arc::new(QteaKernels {
            matmul: module
                .load_function("qtea_matmul_ordered")
                .map_err(|e| driver_err("load QTEA ordered matmul", &e))?,
            gather: module
                .load_function("qtea_gather")
                .map_err(|e| driver_err("load QTEA gather", &e))?,
            _module: module,
        });
        *cache = Some(Arc::clone(&kernels));
        Ok(kernels)
    }

    /// Upload exactly one checked canonical QTEA tensor; no dense expansion.
    /// Cap covers only this handle's requested encoded allocation, not driver
    /// modules/context or other handles. Completes the upload before returning.
    /// # Errors
    /// Positive residency cap, representable geometry, module or allocation/copy failures.
    pub fn upload_qtea(
        &self,
        tensor: QteaTensorView<'_>,
        max_resident_bytes: u64,
    ) -> Result<QteaResidentTensor, BackendError> {
        let geometry = Geometry::checked(tensor)?;
        let encoded = u64::try_from(tensor.as_bytes().len())
            .map_err(|_| invalid("QTEA encoded bytes exceed u64"))?;
        if max_resident_bytes == 0 || encoded > max_resident_bytes {
            return Err(invalid("QTEA encoded residency exceeds caller limit"));
        }
        let kernels = self.qtea_kernels()?;
        let bytes = self
            .stream
            .clone_htod(tensor.as_bytes())
            .map_err(|e| alloc_or_backend("upload QTEA wire bytes", &e, tensor.as_bytes().len()))?;
        self.stream
            .synchronize()
            .map_err(|e| driver_err("finish QTEA upload", &e))?;
        Ok(QteaResidentTensor {
            bytes,
            geometry,
            content_digest: tensor.content_digest(),
            kernels,
        })
    }

    fn check_qtea_context(&self, tensor: &QteaResidentTensor) -> Result<(), BackendError> {
        if !self.same_context(&tensor.bytes) {
            return Err(invalid("QTEA tensor belongs to a foreign CUDA context"));
        }
        Ok(())
    }

    /// Ordered f32 `[batch,columns] * weight^T` from direct packed QTEA bytes.
    /// Separate operations and original-column reduction; no fast-mode alias.
    /// Output is unchanged on every refusal, including nonfinite device results.
    /// # Errors
    /// Foreign context, shape/ABI/limit overflow, nonfinite input/output, allocation or CUDA error.
    pub fn qtea_matmul_ordered_into(
        &self,
        tensor: &QteaResidentTensor,
        inputs: &[f32],
        batch: usize,
        output: &mut [f32],
        limits: QteaCudaLimits,
    ) -> Result<QteaCudaAllocationLedger, BackendError> {
        self.check_qtea_context(tensor)?;
        let (rows, columns) = tensor.shape();
        if batch.checked_mul(columns) != Some(inputs.len())
            || batch.checked_mul(rows) != Some(output.len())
        {
            return Err(invalid("QTEA matmul shape mismatch"));
        }
        let count = u32::try_from(output.len())
            .map_err(|_| invalid("QTEA output count exceeds kernel u32 ABI"))?;
        if inputs.iter().any(|x| !x.is_finite()) {
            return Err(invalid("QTEA input is nonfinite"));
        }
        let mut ledger = plan(tensor.encoded_bytes(), inputs.len(), output.len(), limits)?;
        if count == 0 {
            return Ok(ledger);
        }
        let mut staged = stage(output.len(), limits.max_host_staging_bytes)?;
        ledger.host_staging_capacity_bytes = byte_count(staged.capacity())?;
        let device_input = self.stream.clone_htod(inputs).map_err(|e| {
            alloc_or_backend(
                "upload QTEA activations",
                &e,
                byte_count(inputs.len()).unwrap_or(usize::MAX),
            )
        })?;
        let mut device_output = self.stream.alloc_zeros::<f32>(output.len()).map_err(|e| {
            alloc_or_backend(
                "allocate QTEA output",
                &e,
                byte_count(output.len()).unwrap_or(usize::MAX),
            )
        })?;
        let g = tensor.geometry;
        let mut launch = self.stream.launch_builder(&tensor.kernels.matmul);
        launch
            .arg(&tensor.bytes)
            .arg(&device_input)
            .arg(&mut device_output)
            .arg(&count)
            .arg(&g.rows)
            .arg(&g.columns)
            .arg(&g.group)
            .arg(&g.residual_count)
            .arg(&g.trits)
            .arg(&g.params)
            .arg(&g.multipliers)
            .arg(&g.residuals)
            .arg(&g.record_bytes)
            .arg(&g.position_bytes);
        // SAFETY: private checked wire/layout and same-context buffers; input/output
        // lengths and u32 count validated above. Four whole warps per block match
        // the disjoint shared-product arrays and full-mask synchronization.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(LaunchConfig {
                    grid_dim: (count.div_ceil(4), 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| driver_err("launch QTEA ordered matmul", &e))?;
        }
        self.finish_qtea(&device_output, &mut staged, output)?;
        Ok(ledger)
    }

    /// Reconstruct only requested original rows into `[row_ids.len(),columns]`.
    /// No complete weight copy, CPU fallback or retained per-call workspace.
    /// Output is unchanged on every refusal.
    /// # Errors
    /// Foreign context, invalid row, shape/ABI/limit overflow, nonfinite output or CUDA error.
    pub fn qtea_gather_into(
        &self,
        tensor: &QteaResidentTensor,
        row_ids: &[u32],
        output: &mut [f32],
        limits: QteaCudaLimits,
    ) -> Result<QteaCudaAllocationLedger, BackendError> {
        self.check_qtea_context(tensor)?;
        let (rows, columns) = tensor.shape();
        if row_ids.len().checked_mul(columns) != Some(output.len()) {
            return Err(invalid("QTEA gather shape mismatch"));
        }
        let count = u32::try_from(output.len())
            .map_err(|_| invalid("QTEA output count exceeds kernel u32 ABI"))?;
        if row_ids.iter().any(|&row| row as usize >= rows) {
            return Err(invalid("QTEA gather row outside vocabulary"));
        }
        let mut ledger = plan(tensor.encoded_bytes(), row_ids.len(), output.len(), limits)?;
        if count == 0 {
            return Ok(ledger);
        }
        let mut staged = stage(output.len(), limits.max_host_staging_bytes)?;
        ledger.host_staging_capacity_bytes = byte_count(staged.capacity())?;
        let device_rows = self.stream.clone_htod(row_ids).map_err(|e| {
            alloc_or_backend(
                "upload QTEA row IDs",
                &e,
                byte_count(row_ids.len()).unwrap_or(usize::MAX),
            )
        })?;
        let mut device_output = self.stream.alloc_zeros::<f32>(output.len()).map_err(|e| {
            alloc_or_backend(
                "allocate QTEA gathered rows",
                &e,
                byte_count(output.len()).unwrap_or(usize::MAX),
            )
        })?;
        let g = tensor.geometry;
        let mut launch = self.stream.launch_builder(&tensor.kernels.gather);
        launch
            .arg(&tensor.bytes)
            .arg(&device_rows)
            .arg(&mut device_output)
            .arg(&count)
            .arg(&g.rows)
            .arg(&g.columns)
            .arg(&g.group)
            .arg(&g.residual_count)
            .arg(&g.trits)
            .arg(&g.params)
            .arg(&g.multipliers)
            .arg(&g.residuals)
            .arg(&g.record_bytes)
            .arg(&g.position_bytes);
        // SAFETY: checked private wire/layout, same context, valid original row
        // IDs and exact bounded output. Each thread writes one distinct element.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(LaunchConfig {
                    grid_dim: (count.div_ceil(128), 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| driver_err("launch QTEA gather", &e))?;
        }
        self.finish_qtea(&device_output, &mut staged, output)?;
        Ok(ledger)
    }

    fn finish_qtea(
        &self,
        device: &CudaSlice<f32>,
        staged: &mut [f32],
        output: &mut [f32],
    ) -> Result<(), BackendError> {
        self.stream
            .memcpy_dtoh(device, staged)
            .map_err(|e| driver_err("download QTEA output", &e))?;
        self.stream
            .synchronize()
            .map_err(|e| driver_err("finish QTEA execution", &e))?;
        if staged.iter().any(|x| !x.is_finite()) {
            return Err(invalid("QTEA device arithmetic produced nonfinite output"));
        }
        output.copy_from_slice(staged);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn logical_caps_overflows_and_exact_boundary_without_cuda() {
        let limits = QteaCudaLimits {
            max_device_staging_bytes: 40,
            max_host_staging_bytes: 24,
        };
        let ledger = plan(100, 4, 6, limits).unwrap();
        assert_eq!(ledger.requested_device_peak_bytes, 140);
        assert_eq!(ledger.input_or_index_bytes, 16);
        assert_eq!(ledger.output_bytes, 24);
        assert_eq!(ledger.dense_weight_bytes, 0);
        assert!(
            plan(
                100,
                4,
                6,
                QteaCudaLimits {
                    max_device_staging_bytes: 39,
                    ..limits
                }
            )
            .is_err()
        );
        assert!(
            plan(
                100,
                4,
                6,
                QteaCudaLimits {
                    max_host_staging_bytes: 23,
                    ..limits
                }
            )
            .is_err()
        );
        assert!(
            plan(
                0,
                0,
                0,
                QteaCudaLimits {
                    max_host_staging_bytes: 0,
                    ..limits
                }
            )
            .is_err()
        );
        assert!(
            plan(
                0,
                0,
                0,
                QteaCudaLimits {
                    max_device_staging_bytes: 0,
                    ..limits
                }
            )
            .is_err()
        );
        assert!(plan(0, usize::MAX, 0, limits).is_err());
        assert!(plan(usize::MAX, 4, 6, limits).is_err());
        assert_eq!(
            plan(100, 0, 0, limits).unwrap().requested_device_peak_bytes,
            100
        );
    }

    #[test]
    #[ignore = "explicit physical CUDA context isolation test; no silent skip"]
    fn foreign_context_refuses_even_empty_calls() {
        let backend = CudaBackend::new(0).expect("physical CUDA required");
        let _restore = CurrentContextRestore::capture().unwrap();
        let device = result::device::get(0).unwrap();
        let mut raw = std::ptr::null_mut();
        // SAFETY: driver initialized by backend, valid device, writable output
        // pointer and null optional parameters. Transfer the newly created,
        // independent context exactly once to cudarc's owning RAII wrapper.
        // cudarc 0.19.9 omits its safe constructor for the cuda-13020 feature;
        // this test-only binding does not change the dependency feature policy.
        #[allow(unsafe_code)]
        let foreign = unsafe {
            sys::cuCtxCreate_v4(&mut raw, std::ptr::null_mut(), 0, device)
                .result()
                .expect("create separate CUDA context");
            CudaContext::from_raw_context(0, device, raw).expect("own separate CUDA context")
        };
        let stream = foreign.default_stream();
        let tensor = QteaResidentTensor {
            bytes: stream.clone_htod(&[0u8]).unwrap(),
            geometry: Geometry {
                rows: 4,
                columns: 2,
                group: 2,
                residual_count: 0,
                trits: 0,
                params: 0,
                multipliers: 0,
                residuals: 0,
                record_bytes: 0,
                position_bytes: 0,
            },
            content_digest: [0; 32],
            kernels: backend.qtea_kernels().unwrap(),
        };
        let limits = QteaCudaLimits {
            max_device_staging_bytes: 1,
            max_host_staging_bytes: 1,
        };
        assert!(
            backend
                .qtea_matmul_ordered_into(&tensor, &[], 0, &mut [], limits)
                .is_err()
        );
        assert!(
            backend
                .qtea_gather_into(&tensor, &[], &mut [], limits)
                .is_err()
        );
        foreign.synchronize().unwrap();
    }
}
