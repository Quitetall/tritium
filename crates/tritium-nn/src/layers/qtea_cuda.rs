//! Host-orchestrated packed CUDA matrix execution (proposed ADR0173).
use crate::NnError;
use tritium_cuda::{CudaBackend, QteaCudaLimits, QteaResidentTensor};
use tritium_format::qtea::QteaTensorView;
use tritium_spec::TernaryBackend;

/// One immutable checked CUDA QTEA allocation shared by projection and embedding.
/// No host matrix mirror; per-call limits are logical, not physical/model admission.
#[derive(Debug)]
pub struct CudaQteaLinear {
    tensor: QteaResidentTensor,
    staging: QteaCudaLimits,
}
impl CudaQteaLinear {
    pub(crate) fn try_device_swiglu(
        &self,
        up: &Self,
        down: &Self,
        backend: &dyn TernaryBackend,
        input: &[f32],
        batch: usize,
        output: &mut [f32],
    ) -> Result<bool, NnError> {
        let backend = Self::backend(backend)?;
        if !backend.device_swiglu_enabled() {
            return Ok(false);
        }
        let staging = QteaCudaLimits {
            max_device_staging_bytes: self
                .staging
                .max_device_staging_bytes
                .min(up.staging.max_device_staging_bytes)
                .min(down.staging.max_device_staging_bytes),
            max_host_staging_bytes: self
                .staging
                .max_host_staging_bytes
                .min(up.staging.max_host_staging_bytes)
                .min(down.staging.max_host_staging_bytes),
        };
        backend
            .qtea_swiglu_device_into(
                &self.tensor,
                &up.tensor,
                &down.tensor,
                input,
                batch,
                output,
                staging,
            )
            .map(|_| true)
            .map_err(|e| NnError::Backend(e.to_string()))
    }
    /// Upload checked canonical bytes with positive residency and staging caps.
    /// The caller retains its source; this handle retains only encoded CUDA bytes.
    /// # Errors
    /// Invalid caps or the checked CUDA upload refusal; never falls back to CPU.
    pub fn upload(
        backend: &CudaBackend,
        tensor: QteaTensorView<'_>,
        max_resident_bytes: u64,
        staging: QteaCudaLimits,
    ) -> Result<Self, NnError> {
        if staging.max_device_staging_bytes == 0 || staging.max_host_staging_bytes == 0 {
            return Err(NnError::Backend(
                "QTEA CUDA staging limits must be positive".into(),
            ));
        }
        let tensor = backend
            .upload_qtea(tensor, max_resident_bytes)
            .map_err(|e| NnError::Backend(e.to_string()))?;
        Ok(Self { tensor, staging })
    }
    /// Output rows.
    pub fn rows(&self) -> usize {
        self.tensor.shape().0
    }
    /// Input columns.
    pub fn columns(&self) -> usize {
        self.tensor.shape().1
    }
    /// Exact encoded device allocation request, not physical VRAM.
    pub fn encoded_bytes(&self) -> usize {
        self.tensor.encoded_bytes()
    }
    fn backend(backend: &dyn TernaryBackend) -> Result<&CudaBackend, NnError> {
        backend
            .as_concrete()
            .and_then(|b| b.downcast_ref::<CudaBackend>())
            .ok_or_else(|| {
                NnError::Backend(
                    "QTEA CUDA execution requires its concrete CUDA backend; no host fallback"
                        .into(),
                )
            })
    }
    /// Ordered f32 matrix contraction; output unchanged on refusal.
    /// # Errors
    /// Wrong backend/context, shape, finite/cap/allocation or device refusal.
    pub fn forward(
        &self,
        backend: &dyn TernaryBackend,
        input: &[f32],
        batch: usize,
        output: &mut [f32],
    ) -> Result<(), NnError> {
        Self::backend(backend)?
            .qtea_matmul_ordered_into(&self.tensor, input, batch, output, self.staging)
            .map(|_| ())
            .map_err(|e| NnError::Backend(e.to_string()))
    }
    /// Original-row gather; output unchanged on refusal, no dense weight shadow.
    /// # Errors
    /// Wrong backend/context, row/shape, cap/allocation or device refusal.
    pub fn gather(
        &self,
        backend: &dyn TernaryBackend,
        rows: &[u32],
        output: &mut [f32],
    ) -> Result<(), NnError> {
        Self::backend(backend)?
            .qtea_gather_into(&self.tensor, rows, output, self.staging)
            .map(|_| ())
            .map_err(|e| NnError::Backend(e.to_string()))
    }
}
