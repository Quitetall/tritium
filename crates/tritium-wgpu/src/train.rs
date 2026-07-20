//! Portable-training lifecycle adapter for the native wgpu device.

use tritium_spec::{
    BackendError, TrainBackendError, TrainBackendV1, TrainBufferDataMutV1, TrainBufferDataRefV1,
    TrainCapabilitiesV1, TrainDTypeV1, TrainLimitsV1, TrainOutputV1, TrainReceiptV1,
    TrainRequestV1, TrainingOpManifestV1, train_output_digest_v1, train_request_digest_v1,
};

use crate::WgpuBackend;

const OPERATIONS: &[&str] = &[
    "lifecycle.checkpoint",
    "lifecycle.resume",
    "lifecycle.export",
    "lifecycle.reload",
];
const LIMITS: TrainLimitsV1 = TrainLimitsV1 {
    max_rank: 4,
    max_elements: i32::MAX as u64,
    max_bytes: u32::MAX as u64,
};

/// Native-wgpu implementation of the frozen portable-training seam.
///
/// The initial proved slice contains lifecycle operations only. These operate
/// on canonical host-visible checkpoint/artifact bytes and never invoke CPU
/// tensor execution. Tensor operations are advertised only as their WGSL
/// kernels pass the frozen corpus on an actual adapter.
#[derive(Debug)]
pub struct WgpuTrainBackendV1 {
    _backend: WgpuBackend,
    physical_device: String,
}

impl WgpuTrainBackendV1 {
    /// Open the selected native wgpu adapter.
    ///
    /// # Errors
    /// Returns a backend error when no compatible native adapter is available.
    pub fn new() -> Result<Self, BackendError> {
        let backend = WgpuBackend::new()?;
        let physical_device = backend.physical_device().to_owned();
        Ok(Self {
            _backend: backend,
            physical_device,
        })
    }
}

impl TrainBackendV1 for WgpuTrainBackendV1 {
    fn capabilities(&self) -> TrainCapabilitiesV1 {
        TrainCapabilitiesV1 {
            backend_id: "wgpu.portable.v1:wgpu".to_owned(),
            manifest_digest: TrainingOpManifestV1::digest(),
            supported_operations: OPERATIONS
                .iter()
                .map(|operation| (*operation).to_owned())
                .collect(),
            dtypes: vec![TrainDTypeV1::Bytes],
            limits: LIMITS,
            device_resident: true,
        }
    }

    fn execute(
        &self,
        request: TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<TrainReceiptV1, TrainBackendError> {
        request.validate_with_limits(output, LIMITS)?;
        if !OPERATIONS.contains(&request.operation) {
            return Err(TrainBackendError::UnsupportedOperation(
                request.operation.to_owned(),
            ));
        }
        let input_digest = train_request_digest_v1(&request);
        tritium_train::portable::execute_lifecycle_control_plane(&request, output)?;
        let scratch_bytes =
            tritium_train::portable::lifecycle_control_plane_scratch_bytes(&request)?;
        Ok(TrainReceiptV1 {
            backend_id: "wgpu.portable.v1:wgpu".to_owned(),
            backend_build: format!("tritium-wgpu-{}", env!("CARGO_PKG_VERSION")),
            physical_device: Some(self.physical_device.clone()),
            manifest_digest: TrainingOpManifestV1::digest(),
            vector_digest: request.vector_digest,
            operation: request.operation.to_owned(),
            execution: request.execution,
            dtype: TrainDTypeV1::Bytes,
            limits: LIMITS,
            input_digest,
            output_digest: train_output_digest_v1(output),
            peak_resident_bytes: resident_bytes(&request, output)?,
            scratch_bytes,
            host_transfers: 0,
            device_resident: true,
        })
    }
}

fn resident_bytes(
    request: &TrainRequestV1<'_>,
    output: &TrainOutputV1<'_>,
) -> Result<u64, TrainBackendError> {
    request
        .inputs
        .iter()
        .map(|buffer| match buffer.data {
            TrainBufferDataRefV1::F32(data) => (data.len() as u64).checked_mul(4),
            TrainBufferDataRefV1::U32(data) => (data.len() as u64).checked_mul(4),
            TrainBufferDataRefV1::Bytes(data) => Some(data.len() as u64),
        })
        .chain(output.buffers.iter().map(|buffer| match &buffer.data {
            TrainBufferDataMutV1::F32(data) => (data.len() as u64).checked_mul(4),
            TrainBufferDataMutV1::U32(data) => (data.len() as u64).checked_mul(4),
            TrainBufferDataMutV1::Bytes(data) => Some(data.len() as u64),
        }))
        .try_fold(0_u64, |total, bytes| {
            total
                .checked_add(bytes.ok_or_else(shape_error)?)
                .ok_or_else(shape_error)
        })
}

fn shape_error() -> TrainBackendError {
    TrainBackendError::InvalidOperation(tritium_spec::TrainOperationErrorV1::Shape)
}
