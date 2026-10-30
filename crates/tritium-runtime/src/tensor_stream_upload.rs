//! Mandatory bounded encoded upload, with no decoded compatibility fallback.
use crate::{TensorUploadContext, TensorUploadError, TensorUploadResources};
use std::cell::Cell;
use tritium_spec::{
    BackendError, DeviceBuffer, TensorCaps, TensorStreamSource, TensorUploadPolicy, TernaryBackend,
    validate_tensor_upload_policy,
};

/// Reserve planned payload/workspaces before visiting an encoded source. Unknown
/// streaming support and length/caps/plan changes refuse without fallback. Source
/// and backend numerical/canonical errors drop private handles and all leases.
/// Successful observation keeps scalar decoded_source_bytes semantics without
/// asserting a decoded master was allocated. Enclosing package custody and other
/// memory domains remain the caller's responsibility; not physical qualification.
pub fn upload_tensor_stream_logged_with_resources(
    backend: &dyn TernaryBackend,
    source: &dyn TensorStreamSource,
    policy: TensorUploadPolicy,
    name: &str,
    context: TensorUploadContext<'_>,
    resources: TensorUploadResources<'_>,
) -> Result<(Box<dyn DeviceBuffer>, TensorCaps), TensorUploadError> {
    if name.is_empty() {
        return Err(TensorUploadError::EmptyTensorName);
    }
    context
        .recorder
        .check_emit::<tritium_evidence::TensorUploaded>(
            context.span,
            context.parent,
            context.logical_time,
        )
        .map_err(TensorUploadError::Evidence)?;
    let metadata = source.metadata();
    let plan = backend
        .tensor_stream_plan(metadata, source.encoding())
        .map_err(TensorUploadError::Backend)?
        .ok_or(TensorUploadError::Backend(BackendError::UnsupportedTensor))?;
    if source.encoded_bytes() != plan.encoded_bytes {
        return Err(TensorUploadError::Backend(BackendError::InvalidInput(
            "stream source length differs from plan".into(),
        )));
    }
    validate_tensor_upload_policy(plan.upload.caps, policy).map_err(TensorUploadError::Backend)?;
    let decoded = metadata
        .decoded_payload_bytes()
        .map_err(TensorUploadError::Backend)?;
    let _host = resources
        .host_workspace
        .reserve(plan.upload.host_workspace_bytes)
        .map_err(TensorUploadError::Backend)?;
    let _device = resources
        .device_workspace
        .reserve(plan.upload.device_workspace_bytes)
        .map_err(TensorUploadError::Backend)?;
    let payload = resources
        .payload
        .reserve(plan.upload.caps.payload_bytes)
        .map_err(TensorUploadError::Backend)?;
    let checked = CheckedSource {
        source,
        completed: Cell::new(false),
    };
    let buffer = backend
        .upload_tensor_stream(&checked)
        .map_err(TensorUploadError::Backend)?;
    if !checked.completed.get() {
        return Err(TensorUploadError::Backend(BackendError::InvalidInput(
            "backend did not complete a checked source visit".into(),
        )));
    }
    let actual = u64::try_from(buffer.len_bytes()).map_err(|_| {
        TensorUploadError::Backend(BackendError::InvalidInput(
            "streamed payload size exceeds u64".into(),
        ))
    })?;
    if actual != plan.upload.caps.payload_bytes {
        return Err(TensorUploadError::Backend(
            BackendError::TensorPayloadMismatch {
                advertised: plan.upload.caps.payload_bytes,
                actual,
            },
        ));
    }
    if backend
        .tensor_stream_plan(source.metadata(), source.encoding())
        .map_err(TensorUploadError::Backend)?
        != Some(plan)
        || source.encoded_bytes() != plan.encoded_bytes
    {
        return Err(TensorUploadError::Backend(BackendError::InvalidInput(
            "stream plan changed during upload".into(),
        )));
    }
    super::tensor_upload::record_uploaded(backend, name, plan.upload.caps, decoded, context)?;
    Ok((payload.attach(buffer), plan.upload.caps))
}

struct CheckedSource<'a> {
    source: &'a dyn TensorStreamSource,
    completed: Cell<bool>,
}
impl TensorStreamSource for CheckedSource<'_> {
    fn metadata(&self) -> tritium_spec::TensorMetadata<'_> {
        self.source.metadata()
    }
    fn encoding(&self) -> tritium_spec::TensorStreamEncoding {
        self.source.encoding()
    }
    fn encoded_bytes(&self) -> u64 {
        self.source.encoded_bytes()
    }
    fn visit(
        &self,
        visitor: &mut dyn FnMut(&[u8]) -> Result<(), BackendError>,
    ) -> Result<(), BackendError> {
        self.completed.set(false);
        let mut bytes = 0u64;
        let mut rejected = false;
        self.source.visit(&mut |chunk| {
            let result = (|| {
                if chunk.len() > tritium_spec::TENSOR_STREAM_CHUNK_BYTES {
                    return Err(BackendError::InvalidInput(
                        "stream window exceeds bound".into(),
                    ));
                }
                bytes = bytes
                    .checked_add(chunk.len() as u64)
                    .ok_or_else(|| BackendError::InvalidInput("stream length overflow".into()))?;
                if bytes > self.encoded_bytes() {
                    return Err(BackendError::InvalidInput(
                        "stream exceeds declared length".into(),
                    ));
                }
                visitor(chunk)
            })();
            rejected |= result.is_err();
            result
        })?;
        if rejected {
            return Err(BackendError::InvalidInput(
                "source ignored a rejected stream window".into(),
            ));
        }
        if bytes != self.encoded_bytes() {
            return Err(BackendError::InvalidInput(
                "stream is shorter than declared length".into(),
            ));
        }
        self.completed.set(true);
        Ok(())
    }
}
