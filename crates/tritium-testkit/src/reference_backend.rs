//! A trivial-but-correct [`TernaryBackend`] that the harness can validate itself
//! against: unpack the format with `tritium-format`, then delegate to
//! [`tritium_core::reference_mpgemm`].
//!
//! This is `#[doc(hidden)]` and exists so the crate's own doctests and tests have
//! a known-good backend to exercise [`crate::run_conformance`] with. Real
//! backends live in `tritium-cpu` / `tritium-cuda`; this one is deliberately the
//! reference, so a passing run proves the *harness* is correct, not a kernel.

use core::any::Any;

use tritium_core::{
    DenseView, GemmShape, TernaryFormat, Trit, reference_embed, reference_mpgemm,
    reference_ternary_matmul,
};
use tritium_format::AdditiveTensor;
use tritium_format::{
    TQ1_0_BLOCK_BYTES, TQ2_0_BLOCK_BYTES, num_blocks, unpack_tq1_0_row, unpack_tq2_0_row,
};
use tritium_spec::{
    BackendError, DeviceBuffer, DeviceCaps, MpGemm, TensorCaps, TensorExecution, TensorMatmul,
    TensorView, TernaryBackend, admitted_execution_group,
};

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

/// Owned semantic additive tensor held by the additive reference path.
#[derive(Debug)]
struct RefAdditiveBuffer {
    tensor: AdditiveTensor,
}

impl DeviceBuffer for RefAdditiveBuffer {
    fn len_bytes(&self) -> usize {
        self.tensor.trits().len() + core::mem::size_of_val(self.tensor.scales())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Owned dense f32 tensor held by the reference backend.
#[derive(Debug)]
struct RefDenseBuffer {
    rows: usize,
    cols: usize,
    values: Vec<f32>,
}

impl DeviceBuffer for RefDenseBuffer {
    fn len_bytes(&self) -> usize {
        self.values.len() * core::mem::size_of::<f32>()
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

    fn tensor_caps(&self, tensor: TensorView<'_>) -> Result<Option<TensorCaps>, BackendError> {
        if let TensorView::Additive(view) = tensor {
            let layout = view.layout();
            if !admitted_execution_group(layout.law, layout.group) {
                return Ok(None);
            }
            AdditiveTensor::validate_view(view).map_err(|e| {
                BackendError::InvalidInput(format!(
                    "reference additive capability input failed: {e:?}"
                ))
            })?;
        }
        Ok(Some(TensorCaps {
            execution: TensorExecution::Native,
            payload_bytes: tensor.decoded_payload_bytes()?,
        }))
    }

    fn upload_tensor(&self, tensor: TensorView<'_>) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        match tensor {
            TensorView::Additive(view) => {
                let tensor = AdditiveTensor::from_view(view).map_err(|error| {
                    BackendError::InvalidInput(format!("invalid additive tensor: {error:?}"))
                })?;
                Ok(Box::new(RefAdditiveBuffer { tensor }))
            }
            TensorView::Dense { rows, cols, values } => {
                let expected = rows.checked_mul(cols).ok_or_else(|| {
                    BackendError::InvalidInput("dense tensor dimensions overflow".into())
                })?;
                if values.len() != expected {
                    return Err(BackendError::ShapeMismatch {
                        expected,
                        got: values.len(),
                    });
                }
                Ok(Box::new(RefDenseBuffer {
                    rows,
                    cols,
                    values: values.to_vec(),
                }))
            }
        }
    }

    fn matmul(&self, p: TensorMatmul<'_>) -> Result<(), BackendError> {
        if let Some(buf) = p.tensor.as_any().downcast_ref::<RefAdditiveBuffer>() {
            let view = buf.tensor.view();
            return reference_ternary_matmul(
                p.act,
                &view.bind_matmul(),
                p.batch,
                p.transformed_act,
                p.out,
            )
            .map_err(|e| BackendError::InvalidInput(format!("reference matmul failed: {e:?}")));
        }
        let buf = p
            .tensor
            .as_any()
            .downcast_ref::<RefDenseBuffer>()
            .ok_or_else(|| BackendError::InvalidInput("unknown reference tensor kind".into()))?;
        DenseView::new(buf.rows, buf.cols, &buf.values)
            .and_then(|view| view.matmul(p.act, p.batch, p.transformed_act, p.out))
            .map_err(|e| {
                BackendError::InvalidInput(format!("reference dense matmul failed: {e:?}"))
            })
    }

    fn embed_rows(
        &self,
        tensor: &dyn DeviceBuffer,
        ids: &[usize],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        if let Some(buf) = tensor.as_any().downcast_ref::<RefAdditiveBuffer>() {
            return reference_embed(&buf.tensor.view().bind_gather(), ids, out).map_err(|e| {
                BackendError::InvalidInput(format!("reference gather failed: {e:?}"))
            });
        }
        let buf = tensor
            .as_any()
            .downcast_ref::<RefDenseBuffer>()
            .ok_or_else(|| BackendError::InvalidInput("unknown reference tensor kind".into()))?;
        DenseView::new(buf.rows, buf.cols, &buf.values)
            .and_then(|view| view.embed(ids, out))
            .map_err(|e| {
                BackendError::InvalidInput(format!("reference dense gather failed: {e:?}"))
            })
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

    fn mpgemm(&self, p: MpGemm<'_>) -> Result<(), BackendError> {
        let MpGemm {
            act,
            weights,
            scales,
            shape,
            format: _format,
            out,
        } = p;
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
