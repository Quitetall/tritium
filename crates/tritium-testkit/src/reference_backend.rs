//! A trivial-but-correct [`TernaryBackend`] that the harness can validate itself
//! against: unpack the format with `tritium-format`, then delegate to
//! [`tritium_core::reference_mpgemm`].
//!
//! This is `#[doc(hidden)]` and exists so the crate's own doctests and tests have
//! a known-good backend to exercise [`crate::run_conformance`] with. Real
//! backends live in `tritium-cpu` / `tritium-cuda`; this one is deliberately the
//! reference, so a passing run proves the *harness* is correct, not a kernel.

use core::any::Any;

use tritium_core::{GemmShape, TernaryFormat, Trit, reference_mpgemm};
use tritium_format::{
    TQ1_0_BLOCK_BYTES, TQ2_0_BLOCK_BYTES, num_blocks, unpack_tq1_0_row, unpack_tq2_0_row,
};
use tritium_spec::{BackendError, DeviceBuffer, DeviceCaps, TernaryBackend};

/// Device buffer for [`ReferenceBackend`]: the unpacked trits plus the shape they
/// came from, so `mpgemm` needs no re-derivation.
#[derive(Debug)]
pub(crate) struct RefBuffer {
    trits: Vec<Trit>,
    n: usize,
    k: usize,
    bytes: usize,
}

impl DeviceBuffer for RefBuffer {
    fn len_bytes(&self) -> usize {
        self.bytes
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A reference backend: unpacks with `tritium-format`, runs `reference_mpgemm`.
#[derive(Debug, Default)]
pub struct ReferenceBackend;

impl ReferenceBackend {
    /// Construct the reference backend.
    #[must_use]
    pub fn new() -> Self {
        ReferenceBackend
    }
}

impl TernaryBackend for ReferenceBackend {
    fn device_id(&self) -> &str {
        "reference"
    }

    fn capabilities(&self) -> DeviceCaps {
        DeviceCaps::new("reference", "tritium-testkit reference backend")
    }

    fn upload_weights(
        &self,
        packed: &[u8],
        shape: GemmShape,
        format: TernaryFormat,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        let GemmShape { n, k, .. } = shape;
        let nb = num_blocks(k);
        let block_bytes = match format {
            TernaryFormat::Tq2_0 => TQ2_0_BLOCK_BYTES,
            TernaryFormat::Tq1_0 => TQ1_0_BLOCK_BYTES,
            other => return Err(BackendError::UnsupportedFormat(other)),
        };
        let row_bytes = nb * block_bytes;
        let expected = n * row_bytes;
        if packed.len() != expected {
            return Err(BackendError::InvalidInput(format!(
                "packed len {} != expected {expected} for shape {shape:?}",
                packed.len()
            )));
        }

        let mut trits = vec![Trit::ZERO; n * k];
        // Scratch scales (one per block per row); the reference applies the
        // per-channel scales separately, so these unpacked block scales are
        // discarded — they were fixed to 1.0 at pack time by the harness.
        let mut scratch = vec![half::f16::ONE; nb];
        for ni in 0..n {
            let row = &packed[ni * row_bytes..(ni + 1) * row_bytes];
            let trits_row = &mut trits[ni * k..ni * k + k];
            let res = match format {
                TernaryFormat::Tq2_0 => unpack_tq2_0_row(row, trits_row, &mut scratch),
                TernaryFormat::Tq1_0 => unpack_tq1_0_row(row, trits_row, &mut scratch),
                other => return Err(BackendError::UnsupportedFormat(other)),
            };
            res.map_err(|e| BackendError::Backend(format!("unpack row {ni}: {e}")))?;
        }

        Ok(Box::new(RefBuffer {
            trits,
            n,
            k,
            bytes: packed.len(),
        }))
    }

    fn mpgemm(
        &self,
        act: &[f32],
        weights: &dyn DeviceBuffer,
        scales: &[f32],
        shape: GemmShape,
        _format: TernaryFormat,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let buf = weights
            .as_any()
            .downcast_ref::<RefBuffer>()
            .ok_or_else(|| BackendError::InvalidInput("buffer is not a RefBuffer".into()))?;
        if buf.n != shape.n || buf.k != shape.k {
            return Err(BackendError::ShapeMismatch {
                expected: buf.n * buf.k,
                got: shape.n * shape.k,
            });
        }
        reference_mpgemm(act, &buf.trits, scales, shape, out).map_err(BackendError::Core)
    }
}

/// Build a [`ReferenceBackend`] for use in this crate's doctests.
#[doc(hidden)]
#[must_use]
pub fn reference_backend_for_doctest() -> ReferenceBackend {
    ReferenceBackend::new()
}
