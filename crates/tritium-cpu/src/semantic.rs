//! Packed semantic execution; legacy packed SIMD dispatch is unchanged.

use core::any::Any;
use tritium_core::DenseView;
use tritium_format::AdditiveTensor;
use tritium_spec::{
    BackendError, DeviceBuffer, TENSOR_STREAM_CHUNK_BYTES, TensorCaps, TensorExecution,
    TensorMatmul, TensorMetadata, TensorStreamEncoding, TensorStreamPlan, TensorStreamSource,
    TensorUploadPlan, TensorView, admitted_execution_group,
};

mod packed;
use packed::PackedAdditive;

#[derive(Debug)]
// Device handles already box this enum once. Keep sealed compact geometry
// inline rather than adding a second infallible allocation/indirection. All
// payload/index buffers reserve fallibly; handle overhead is not payload bytes.
#[allow(clippy::large_enum_variant)]
pub(crate) enum CpuTensor {
    Additive(PackedAdditive),
    Dense {
        rows: usize,
        cols: usize,
        values: Vec<f32>,
    },
}

impl DeviceBuffer for CpuTensor {
    fn len_bytes(&self) -> usize {
        match self {
            Self::Additive(tensor) => tensor.len_bytes(),
            Self::Dense { values, .. } => core::mem::size_of_val(values.as_slice()),
        }
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl CpuTensor {
    pub(crate) fn stream_plan(
        metadata: TensorMetadata<'_>,
        encoding: TensorStreamEncoding,
    ) -> Result<Option<TensorStreamPlan>, BackendError> {
        let encoded = match (metadata, encoding) {
            (TensorMetadata::Additive(a), TensorStreamEncoding::CanonicalRawAdditive)
                if matches!(a.layout().transport, tritium_schema::Transport::Raw) =>
            {
                PackedAdditive::stream_encoded_bytes(a)?
            }
            (
                TensorMetadata::Dense { rows, cols },
                TensorStreamEncoding::DenseLittleEndian(precision),
            ) => {
                let width = match precision {
                    tritium_schema::ScalePrecision::F16 | tritium_schema::ScalePrecision::Bf16 => 2,
                    tritium_schema::ScalePrecision::F32 => 4,
                    _ => return Ok(None),
                };
                TensorMetadata::dense_bytes(rows, cols)? / 4 * width
            }
            _ => return Ok(None),
        };
        let Some(mut upload) = Self::plan(metadata)? else {
            return Ok(None);
        };
        // Canonical validation uses sequential fixed codec/scale scratch. Dense
        // widening uses only a scalar carry. Metadata/allocator overhead excluded.
        upload.host_workspace_bytes = TENSOR_STREAM_CHUNK_BYTES as u64;
        Ok(Some(TensorStreamPlan {
            upload,
            encoded_bytes: encoded as u64,
        }))
    }

    pub(crate) fn upload_stream(
        source: &dyn TensorStreamSource,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        let metadata = source.metadata();
        let plan = Self::stream_plan(metadata, source.encoding())?
            .ok_or(BackendError::UnsupportedTensor)?;
        if source.encoded_bytes() != plan.encoded_bytes {
            return Err(BackendError::InvalidInput(
                "CPU stream encoded length differs from geometry".into(),
            ));
        }
        let tensor = match (metadata, source.encoding()) {
            (TensorMetadata::Additive(a), TensorStreamEncoding::CanonicalRawAdditive) => {
                Self::Additive(PackedAdditive::upload_stream(a, source)?)
            }
            (
                TensorMetadata::Dense { rows, cols },
                TensorStreamEncoding::DenseLittleEndian(precision),
            ) => {
                let requested = TensorMetadata::dense_bytes(rows, cols)?;
                let mut values = Vec::new();
                values
                    .try_reserve_exact(requested / 4)
                    .map_err(|_| BackendError::OutOfMemory { requested })?;
                let width = if precision == tritium_schema::ScalePrecision::F32 {
                    4
                } else {
                    2
                };
                let mut carry = [0u8; 4];
                let mut filled = 0;
                let mut bytes = 0u64;
                source.visit(&mut |chunk| {
                    if chunk.len() > TENSOR_STREAM_CHUNK_BYTES {
                        return Err(BackendError::InvalidInput(
                            "source window exceeds stream bound".into(),
                        ));
                    }
                    bytes = bytes
                        .checked_add(chunk.len() as u64)
                        .ok_or_else(|| BackendError::InvalidInput("stream size overflow".into()))?;
                    if bytes > plan.encoded_bytes {
                        return Err(BackendError::InvalidInput(
                            "dense source exceeds stream plan".into(),
                        ));
                    }
                    for &byte in chunk {
                        carry[filled] = byte;
                        filled += 1;
                        if filled == width {
                            let value = match precision {
                                tritium_schema::ScalePrecision::F16 => {
                                    half::f16::from_bits(u16::from_le_bytes([carry[0], carry[1]]))
                                        .to_f32()
                                }
                                tritium_schema::ScalePrecision::Bf16 => {
                                    half::bf16::from_bits(u16::from_le_bytes([carry[0], carry[1]]))
                                        .to_f32()
                                }
                                tritium_schema::ScalePrecision::F32 => f32::from_le_bytes(carry),
                                _ => return Err(BackendError::UnsupportedTensor),
                            };
                            if !value.is_finite() {
                                return Err(BackendError::InvalidInput(
                                    "non-finite streamed model weight".into(),
                                ));
                            }
                            values.push(value);
                            filled = 0;
                        }
                    }
                    Ok(())
                })?;
                if bytes != plan.encoded_bytes || filled != 0 || values.len() != requested / 4 {
                    return Err(BackendError::InvalidInput(
                        "dense source shorter than stream plan".into(),
                    ));
                }
                Self::Dense { rows, cols, values }
            }
            _ => return Err(BackendError::UnsupportedTensor),
        };
        Ok(Box::new(tensor))
    }
    pub(crate) fn plan(
        metadata: TensorMetadata<'_>,
    ) -> Result<Option<TensorUploadPlan>, BackendError> {
        let bytes = match metadata {
            TensorMetadata::Dense { rows, cols } => TensorMetadata::dense_bytes(rows, cols)?,
            TensorMetadata::Additive(additive) => {
                let layout = additive.layout();
                if !matches!(
                    layout.allocation,
                    tritium_schema::PlaneAllocation::Uniform
                        | tritium_schema::PlaneAllocation::PerTile
                ) || !admitted_execution_group(layout.law, layout.group)
                {
                    return Ok(None);
                }
                PackedAdditive::payload_metadata(additive)?
            }
        };
        Ok(Some(TensorUploadPlan {
            caps: TensorCaps {
                execution: TensorExecution::Native,
                payload_bytes: bytes as u64,
            },
            host_workspace_bytes: 0,
            device_workspace_bytes: 0,
        }))
    }
    pub(crate) fn caps(view: TensorView<'_>) -> Result<Option<TensorCaps>, BackendError> {
        if let TensorView::Additive(additive) = view {
            let layout = additive.layout();
            if !matches!(
                layout.allocation,
                tritium_schema::PlaneAllocation::Uniform | tritium_schema::PlaneAllocation::PerTile
            ) || !admitted_execution_group(layout.law, layout.group)
            {
                return Ok(None);
            }
            AdditiveTensor::validate_view(additive).map_err(|e| {
                BackendError::InvalidInput(format!("CPU additive capability input failed: {e:?}"))
            })?;
            PackedAdditive::validate_symbols(additive)?;
        }
        let plan = Self::plan(TensorMetadata::from_view(view)?)?;
        Ok(plan.map(|p| p.caps))
    }

    pub(crate) fn upload(view: TensorView<'_>) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        let tensor = match view {
            TensorView::Additive(view) => {
                Self::caps(TensorView::Additive(view))?.ok_or(BackendError::UnsupportedTensor)?;
                Self::Additive(PackedAdditive::upload(view)?)
            }
            TensorView::Dense { rows, cols, values } => {
                DenseView::new(rows, cols, values).map_err(|e| {
                    BackendError::InvalidInput(format!("CPU dense upload failed: {e:?}"))
                })?;
                let requested = values
                    .len()
                    .checked_mul(core::mem::size_of::<f32>())
                    .ok_or_else(|| {
                        BackendError::InvalidInput("dense storage size overflows".into())
                    })?;
                let mut owned = Vec::new();
                owned
                    .try_reserve_exact(values.len())
                    .map_err(|_| BackendError::OutOfMemory { requested })?;
                owned.extend_from_slice(values);
                Self::Dense {
                    rows,
                    cols,
                    values: owned,
                }
            }
        };
        Ok(Box::new(tensor))
    }

    pub(crate) fn matmul(p: TensorMatmul<'_>) -> Result<(), BackendError> {
        let tensor = Self::checked(p.tensor)?;
        match tensor {
            Self::Additive(tensor) => tensor.matmul(p),
            Self::Dense { rows, cols, values } => DenseView::new(*rows, *cols, values)
                .and_then(|view| view.matmul(p.act, p.batch, p.transformed_act, p.out))
                .map_err(|e| BackendError::InvalidInput(format!("CPU dense matmul failed: {e:?}"))),
        }
    }

    pub(crate) fn embed(
        tensor: &dyn DeviceBuffer,
        ids: &[usize],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        match Self::checked(tensor)? {
            Self::Additive(tensor) => tensor.embed(ids, out),
            Self::Dense { rows, cols, values } => DenseView::new(*rows, *cols, values)
                .and_then(|view| view.embed(ids, out))
                .map_err(|e| BackendError::InvalidInput(format!("CPU dense gather failed: {e:?}"))),
        }
    }

    fn checked(tensor: &dyn DeviceBuffer) -> Result<&Self, BackendError> {
        tensor
            .as_any()
            .downcast_ref::<Self>()
            .ok_or_else(|| BackendError::InvalidInput("not a CPU semantic tensor".into()))
    }
}
