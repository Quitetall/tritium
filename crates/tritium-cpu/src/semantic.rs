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
mod snapshot;
use packed::{PackedAdditive, PackedScaleView};
pub use snapshot::CpuScaleSnapshotBytes;
use snapshot::ScaleSnapshot;

#[derive(Debug)]
// Device handles already box this enum once. Keep sealed compact geometry
// inline rather than adding a second infallible allocation/indirection. All
// payload/index buffers reserve fallibly; handle overhead is not payload bytes.
#[allow(clippy::large_enum_variant)]
pub(crate) enum CpuTensor {
    Additive(PackedAdditive),
    ScaleSnapshot(ScaleSnapshot),
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
            Self::ScaleSnapshot(snapshot) => snapshot.bytes().reachable_bytes,
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
                if !matches!(
                    precision,
                    tritium_schema::ScalePrecision::F16
                        | tritium_schema::ScalePrecision::Bf16
                        | tritium_schema::ScalePrecision::F32
                ) {
                    return Ok(None);
                }
                tritium_format::trit_model::DensePayloadDecoder::byte_len(
                    precision,
                    TensorMetadata::dense_bytes(rows, cols)? / 4,
                )
                .map_err(|e| BackendError::InvalidInput(e.to_string()))?
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
                values.resize(requested / 4, 0.);
                let mut decoder =
                    tritium_format::trit_model::DensePayloadDecoder::new(precision, &mut values)
                        .map_err(|e| BackendError::InvalidInput(e.to_string()))?;
                source.visit(&mut |chunk| {
                    if chunk.len() > TENSOR_STREAM_CHUNK_BYTES {
                        return Err(BackendError::InvalidInput(
                            "source window exceeds stream bound".into(),
                        ));
                    }
                    decoder
                        .push(chunk)
                        .map_err(|e| BackendError::InvalidInput(e.to_string()))
                })?;
                decoder
                    .finish()
                    .map_err(|e| BackendError::InvalidInput(e.to_string()))?;
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
            Self::ScaleSnapshot(snapshot) => snapshot.view()?.matmul(p),
            Self::Dense { rows, cols, values } => DenseView::new(*rows, *cols, values)
                .and_then(|view| view.matmul(p.act, p.batch, p.transformed_act, p.out))
                .map_err(|e| BackendError::InvalidInput(format!("CPU dense matmul failed: {e:?}"))),
        }
    }

    pub(crate) fn additive_input_vjp(
        tensor: &dyn DeviceBuffer,
        output_gradients: &[f32],
        batch: usize,
        row_workspace: &mut [f32],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        Self::checked_additive_view(tensor)?.input_vjp(output_gradients, batch, row_workspace, out)
    }

    pub(crate) fn additive_scale_count(tensor: &dyn DeviceBuffer) -> Result<usize, BackendError> {
        Ok(Self::checked_additive(tensor)?
            .scale_vjp_plan()?
            .stored_scale_count())
    }

    pub(crate) fn additive_equals_view(
        tensor: &dyn DeviceBuffer,
        view: tritium_core::AdditiveView<'_>,
    ) -> Result<bool, BackendError> {
        let selected = Self::checked(tensor)?;
        if tensor.len_bytes() != selected.len_bytes() {
            return Err(BackendError::InvalidInput(
                "CPU handle payload bytes differ from backend storage".into(),
            ));
        }
        match selected {
            Self::Additive(tensor) => tensor.view().equals_view(view),
            Self::ScaleSnapshot(snapshot) => snapshot.view()?.equals_view(view),
            Self::Dense { .. } => Err(BackendError::UnsupportedTensor),
        }
    }

    pub(crate) fn additive_matmul_scale_vjp(
        tensor: &dyn DeviceBuffer,
        activations: &[f32],
        output_gradients: &[f32],
        batch: usize,
        row_workspace: &mut [f32],
        out: &mut [f64],
    ) -> Result<(), BackendError> {
        Self::checked_additive(tensor)?.matmul_scale_vjp(
            activations,
            output_gradients,
            batch,
            row_workspace,
            out,
        )
    }

    pub(crate) fn additive_embed_scale_vjp(
        tensor: &dyn DeviceBuffer,
        ids: &[usize],
        output_gradients: &[f32],
        row_workspace: &mut [f32],
        out: &mut [f64],
    ) -> Result<(), BackendError> {
        Self::checked_additive(tensor)?.embed_scale_vjp(ids, output_gradients, row_workspace, out)
    }

    fn weight_vjp_plan(
        tensor: &dyn DeviceBuffer,
    ) -> Result<tritium_core::AdditiveWeightVjpPlan<'_>, BackendError> {
        let selected = Self::checked(tensor)?;
        if tensor.len_bytes() != selected.len_bytes() {
            return Err(BackendError::InvalidInput(
                "CPU handle payload bytes differ from backend storage".into(),
            ));
        }
        let packed = match selected {
            Self::Additive(packed) => packed,
            Self::ScaleSnapshot(snapshot) => snapshot.base()?,
            Self::Dense { .. } => return Err(BackendError::UnsupportedTensor),
        };
        Ok(packed.scale_vjp_plan()?.into())
    }

    pub(crate) fn additive_weight_count(tensor: &dyn DeviceBuffer) -> Result<usize, BackendError> {
        Ok(Self::weight_vjp_plan(tensor)?.present_weight_count())
    }

    pub(crate) fn additive_matmul_weight_vjp(
        tensor: &dyn DeviceBuffer,
        activations: &[f32],
        output_gradients: &[f32],
        batch: usize,
        row_workspace: &mut [f32],
        out: &mut [f64],
    ) -> Result<(), BackendError> {
        Self::weight_vjp_plan(tensor)?
            .matmul(activations, output_gradients, batch, row_workspace, out)
            .map_err(|e| BackendError::InvalidInput(format!("CPU weight adjoint failed: {e:?}")))
    }

    pub(crate) fn additive_embed_weight_vjp(
        tensor: &dyn DeviceBuffer,
        ids: &[usize],
        output_gradients: &[f32],
        row_workspace: &mut [f32],
        out: &mut [f64],
    ) -> Result<(), BackendError> {
        Self::weight_vjp_plan(tensor)?
            .embed(ids, output_gradients, row_workspace, out)
            .map_err(|e| BackendError::InvalidInput(format!("CPU weight adjoint failed: {e:?}")))
    }

    fn checked_additive(tensor: &dyn DeviceBuffer) -> Result<&PackedAdditive, BackendError> {
        match Self::checked(tensor)? {
            Self::Additive(tensor) => Ok(tensor),
            Self::ScaleSnapshot(snapshot) => snapshot.base(),
            Self::Dense { .. } => Err(BackendError::UnsupportedTensor),
        }
    }

    fn checked_additive_view(
        tensor: &dyn DeviceBuffer,
    ) -> Result<PackedScaleView<'_>, BackendError> {
        match Self::checked(tensor)? {
            Self::Additive(tensor) => Ok(tensor.view()),
            Self::ScaleSnapshot(snapshot) => snapshot.view(),
            Self::Dense { .. } => Err(BackendError::UnsupportedTensor),
        }
    }

    pub(crate) fn embed(
        tensor: &dyn DeviceBuffer,
        ids: &[usize],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        match Self::checked(tensor)? {
            Self::Additive(tensor) => tensor.embed(ids, out),
            Self::ScaleSnapshot(snapshot) => snapshot.view()?.embed(ids, out),
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
