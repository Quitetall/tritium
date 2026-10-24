//! Explicit host-only packed QTEA projection; no dense weight shadow.

use tritium_format::qtea::{QteaLimits, QteaTensor};

use crate::NnError;

/// Immutable residual-bearing QTEA weights executed by the scalar CPU kernel.
///
/// This is not an additive SALT tensor, device upload or qualification receipt.
/// The byte ceiling covers per-call output staging only, not aggregate model
/// state, allocator overhead, concurrent calls or physical process memory.
pub struct HostQteaLinear {
    tensor: QteaTensor,
    max_staging_bytes: usize,
}

impl core::fmt::Debug for HostQteaLinear {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HostQteaLinear")
            .field("rows", &self.rows())
            .field("columns", &self.columns())
            .field("encoded_capacity_bytes", &self.encoded_capacity_bytes())
            .field("max_staging_bytes", &self.max_staging_bytes)
            .finish_non_exhaustive()
    }
}

impl HostQteaLinear {
    /// Move already checked packed storage without copying or decoding weights.
    /// A zero staging ceiling permits only empty batches.
    #[must_use]
    pub fn from_tensor(tensor: QteaTensor, max_staging_bytes: usize) -> Self {
        Self {
            tensor,
            max_staging_bytes,
        }
    }

    /// Read exactly one bounded canonical tensor, requiring EOF after it.
    /// Embedded source digests remain assertions, not authenticated model custody.
    ///
    /// # Errors
    /// Refuses invalid bytes, read/allocation failures or encoded/validation caps.
    pub fn read_from(
        reader: &mut impl std::io::Read,
        limits: QteaLimits,
        max_staging_bytes: usize,
    ) -> Result<Self, NnError> {
        QteaTensor::read_from(reader, limits)
            .map(|tensor| Self::from_tensor(tensor, max_staging_bytes))
            .map_err(|error| NnError::Backend(error.to_string()))
    }

    /// Output feature count.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.tensor.view().metadata().rows
    }

    /// Input feature count.
    #[must_use]
    pub fn columns(&self) -> usize {
        self.tensor.view().metadata().columns
    }

    /// Retained encoded Vec capacity, excluding handle/allocator overhead.
    #[must_use]
    pub fn encoded_capacity_bytes(&self) -> usize {
        self.tensor.buffer_capacity_bytes()
    }

    fn check_geometry(&self, input: &[f32], batch: usize, output: &[f32]) -> Result<(), NnError> {
        if batch.checked_mul(self.columns()) != Some(input.len())
            || batch.checked_mul(self.rows()) != Some(output.len())
        {
            return Err(NnError::Backend("host QTEA batch geometry mismatch".into()));
        }
        if output
            .len()
            .checked_mul(size_of::<f32>())
            .is_none_or(|bytes| bytes > self.max_staging_bytes)
        {
            return Err(NnError::Backend(
                "host QTEA staging ceiling exceeded".into(),
            ));
        }
        Ok(())
    }

    /// Execute F32 activations with one fallible output-sized staging allocation.
    /// No dense weights, activation quantization or backend dispatch is used.
    ///
    /// # Errors
    /// Refuses shape/count overflow, staging cap/allocation failure, nonfinite
    /// input or arithmetic overflow without changing output.
    pub fn forward(&self, input: &[f32], batch: usize, output: &mut [f32]) -> Result<(), NnError> {
        self.check_geometry(input, batch, output)?;
        let mut staging = Vec::<f32>::new();
        staging
            .try_reserve_exact(output.len())
            .map_err(|error| NnError::Backend(format!("host QTEA staging allocation: {error}")))?;
        if staging
            .capacity()
            .checked_mul(size_of::<f32>())
            .is_none_or(|bytes| bytes > self.max_staging_bytes)
        {
            return Err(NnError::Backend(
                "host QTEA staging capacity exceeded".into(),
            ));
        }
        staging.resize(output.len(), 0.0);
        self.forward_into(input, batch, output, &mut staging)
    }

    /// Execute with caller-owned exact output-sized staging and no heap request.
    /// Staging may change on error; output never does. The ceiling covers the
    /// supplied slice, not any larger allocation owned by its caller.
    ///
    /// # Errors
    /// Same numeric/shape/cap refusals as [`Self::forward`], plus staging mismatch.
    pub fn forward_into(
        &self,
        input: &[f32],
        batch: usize,
        output: &mut [f32],
        staging: &mut [f32],
    ) -> Result<(), NnError> {
        self.check_geometry(input, batch, output)?;
        tritium_cpu::qtea::matmul_into(self.tensor.view(), input, batch, output, staging)
            .map_err(|error| NnError::Backend(error.to_string()))
    }
}
