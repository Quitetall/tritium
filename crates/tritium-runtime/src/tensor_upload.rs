//! Checked upload plus registered evidence, shared by model-loading surfaces.

use crate::TensorPayloadBudget;
use tritium_evidence::{EvidenceError, EvidenceRecorder, TensorUploaded};
use tritium_spec::{
    BackendError, DeviceBuffer, TensorCaps, TensorUploadPolicy, TensorView, TernaryBackend,
    validate_tensor_upload_policy,
};

/// Run/span context for one typed successful-upload observation.
#[derive(Debug)]
pub struct TensorUploadContext<'a> {
    /// Recorder with the tensor-upload schema registered.
    pub recorder: &'a mut EvidenceRecorder,
    /// Span within that run.
    pub span: &'a str,
    /// Optional parent span.
    pub parent: Option<&'a str>,
    /// Deterministic logical time, not wall-clock time.
    pub logical_time: u64,
}

/// Upload/admission or evidence failure; neither becomes a successful load.
#[derive(Debug)]
pub enum TensorUploadError {
    /// Empty caller-selected tensor name.
    EmptyTensorName,
    /// Backend validation, admission, allocation or transfer failure.
    Backend(BackendError),
    /// Evidence emission failed; the uploaded handle was dropped.
    Evidence(EvidenceError),
}

impl core::fmt::Display for TensorUploadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyTensorName => f.write_str("tensor upload requires a nonempty name"),
            Self::Backend(error) => write!(f, "tensor upload: {error}"),
            Self::Evidence(error) => write!(f, "tensor upload evidence: {error}"),
        }
    }
}

impl std::error::Error for TensorUploadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::EmptyTensorName => None,
            Self::Backend(error) => Some(error),
            Self::Evidence(error) => Some(error),
        }
    }
}

/// Upload with explicit tier/budget policy and append a typed successful event.
///
/// Policy/backend rejection emits no successful event. If emission fails after
/// upload, the handle is dropped and the caller receives an error. This observes
/// checked payload bytes only, not aggregate residency or an empirical verdict.
/// Raw/checked backend uploads remain available during migration; production
/// loaders must adopt this function to obtain the D9/D12 evidence guarantee.
pub fn upload_tensor_logged(
    backend: &dyn TernaryBackend,
    view: TensorView<'_>,
    policy: TensorUploadPolicy,
    name: &str,
    context: TensorUploadContext<'_>,
) -> Result<(Box<dyn DeviceBuffer>, TensorCaps), TensorUploadError> {
    upload_logged(backend, view, policy, name, context, None)
}

/// Logged upload with shared lifetime-bound owned-payload admission.
///
/// Native-only and per-tensor policy remain unchanged. The shared ledger is
/// reserved before backend upload and retained by the returned buffer; every
/// error drops the reservation and any uploaded buffer before returning. If
/// capabilities change during upload, no successful event is emitted. Already
/// recorded observations of earlier uploads are not model-load success claims.
///
/// This bounds declared/checked owned payloads only. A backend that misreports
/// its allocation is rejected after upload, not physically prevented from
/// allocating. Model preflight, KV/scratch/driver admission and production loader
/// adoption remain separate obligations (ADR 0044 D9/D10).
pub fn upload_tensor_logged_admitted(
    backend: &dyn TernaryBackend,
    view: TensorView<'_>,
    policy: TensorUploadPolicy,
    name: &str,
    context: TensorUploadContext<'_>,
    budget: &TensorPayloadBudget,
) -> Result<(Box<dyn DeviceBuffer>, TensorCaps), TensorUploadError> {
    upload_logged(backend, view, policy, name, context, Some(budget))
}

fn upload_logged(
    backend: &dyn TernaryBackend,
    view: TensorView<'_>,
    policy: TensorUploadPolicy,
    name: &str,
    context: TensorUploadContext<'_>,
    budget: Option<&TensorPayloadBudget>,
) -> Result<(Box<dyn DeviceBuffer>, TensorCaps), TensorUploadError> {
    if name.is_empty() {
        return Err(TensorUploadError::EmptyTensorName);
    }
    let decoded_source_bytes = view
        .decoded_payload_bytes()
        .map_err(TensorUploadError::Backend)?;
    let reserved = budget
        .map(|budget| {
            let caps = backend
                .tensor_caps(view)?
                .ok_or(BackendError::UnsupportedTensor)?;
            validate_tensor_upload_policy(caps, policy)?;
            Ok::<_, BackendError>((caps, budget.reserve(caps.payload_bytes)?))
        })
        .transpose()
        .map_err(TensorUploadError::Backend)?;
    let (buffer, caps) = backend
        .upload_tensor_checked(view, policy)
        .map_err(TensorUploadError::Backend)?;
    if let Some((expected, _)) = &reserved
        && *expected != caps
    {
        return Err(TensorUploadError::Backend(BackendError::InvalidInput(
            "tensor capabilities changed during admitted upload".into(),
        )));
    }
    context
        .recorder
        .emit(
            context.span,
            context.parent,
            context.logical_time,
            &TensorUploaded {
                tensor: name.to_owned(),
                backend: backend.device_id().to_owned(),
                physical_device: backend.physical_device_id().to_owned(),
                caps,
                decoded_source_bytes,
            },
        )
        .map_err(TensorUploadError::Evidence)?;
    let buffer = match reserved {
        Some((_, reservation)) => reservation.attach(buffer),
        None => buffer,
    };
    Ok((buffer, caps))
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::any::Any;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use tritium_evidence::TENSOR_UPLOADED_REGISTRATION;
    use tritium_spec::{DeviceCaps, GemmShape, MpGemm, TensorExecution, TernaryFormat};

    struct Backend(Arc<AtomicUsize>);
    struct Buffer(Arc<AtomicUsize>);
    impl Drop for Buffer {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    impl DeviceBuffer for Buffer {
        fn len_bytes(&self) -> usize {
            4
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }
    impl TernaryBackend for Backend {
        fn device_id(&self) -> &str {
            "controlled"
        }
        fn capabilities(&self) -> DeviceCaps {
            DeviceCaps::new("controlled", "fixture")
        }
        fn tensor_caps(&self, _: TensorView<'_>) -> Result<Option<TensorCaps>, BackendError> {
            Ok(Some(TensorCaps {
                execution: TensorExecution::Emulated,
                payload_bytes: 4,
            }))
        }
        fn upload_tensor(&self, _: TensorView<'_>) -> Result<Box<dyn DeviceBuffer>, BackendError> {
            Ok(Box::new(Buffer(Arc::clone(&self.0))))
        }
        fn upload_weights(
            &self,
            _: &[u8],
            _: GemmShape,
            _: TernaryFormat,
        ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
            unreachable!()
        }
        fn mpgemm(&self, _: MpGemm<'_>) -> Result<(), BackendError> {
            unreachable!()
        }
    }

    #[test]
    fn failed_emission_drops_uploaded_handle_and_does_not_claim_success() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let backend = Backend(Arc::clone(&dropped));
        let mut recorder = EvidenceRecorder::new("fixture", []).unwrap();
        let result = upload_tensor_logged(
            &backend,
            TensorView::Dense {
                rows: 1,
                cols: 1,
                values: &[1.],
            },
            TensorUploadPolicy::default(),
            "weight",
            TensorUploadContext {
                recorder: &mut recorder,
                span: "load",
                parent: None,
                logical_time: 0,
            },
        );
        assert!(matches!(
            result,
            Err(TensorUploadError::Evidence(
                EvidenceError::UnregisteredSchema(_)
            ))
        ));
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        assert!(recorder.events().is_empty());
    }

    #[test]
    fn policy_rejection_emits_no_success_and_empty_name_never_uploads() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let backend = Backend(Arc::clone(&dropped));
        let mut recorder =
            EvidenceRecorder::new("fixture", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
        for name in ["weight", ""] {
            let result = upload_tensor_logged(
                &backend,
                TensorView::Dense {
                    rows: 1,
                    cols: 1,
                    values: &[1.],
                },
                TensorUploadPolicy {
                    native_only: true,
                    max_payload_bytes: None,
                },
                name,
                TensorUploadContext {
                    recorder: &mut recorder,
                    span: "load",
                    parent: None,
                    logical_time: 0,
                },
            );
            assert!(result.is_err());
            assert!(recorder.events().is_empty());
            assert_eq!(dropped.load(Ordering::SeqCst), 0);
        }
    }
}
