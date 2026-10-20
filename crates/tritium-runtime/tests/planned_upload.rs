use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tritium_evidence::{EvidenceRecorder, TENSOR_UPLOADED_REGISTRATION};
use tritium_runtime::{
    TensorPayloadBudget, TensorUploadContext, TensorUploadResources,
    upload_tensor_logged_with_resources,
};
use tritium_spec::*;

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
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}
struct Backend {
    case: usize,
    payload: TensorPayloadBudget,
    host: TensorPayloadBudget,
    device: TensorPayloadBudget,
    calls: Arc<AtomicUsize>,
    dropped: Arc<AtomicUsize>,
}
impl Backend {
    fn reserved(&self) {
        assert_eq!(self.payload.used_bytes(), 4);
        assert_eq!(self.host.used_bytes(), 8);
        assert_eq!(self.device.used_bytes(), 12);
    }
}
impl TernaryBackend for Backend {
    fn device_id(&self) -> &str {
        "planned-test"
    }
    fn capabilities(&self) -> DeviceCaps {
        DeviceCaps::new("planned-test", "fixture")
    }
    fn tensor_upload_plan(
        &self,
        _: TensorMetadata<'_>,
    ) -> Result<Option<TensorUploadPlan>, BackendError> {
        Ok((self.case != 3).then_some(TensorUploadPlan {
            caps: TensorCaps {
                execution: TensorExecution::Native,
                payload_bytes: 4,
            },
            host_workspace_bytes: 8,
            device_workspace_bytes: 12,
        }))
    }
    fn tensor_caps(&self, _: TensorView<'_>) -> Result<Option<TensorCaps>, BackendError> {
        self.reserved();
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.case == 4 {
            return Err(BackendError::InvalidInput("numerical rejection".into()));
        }
        Ok(Some(TensorCaps {
            execution: if self.case == 6 {
                TensorExecution::Emulated
            } else {
                TensorExecution::Native
            },
            payload_bytes: 4,
        }))
    }
    fn upload_tensor(&self, _: TensorView<'_>) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        self.reserved();
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.case == 5 {
            return Err(BackendError::Backend("transfer failure".into()));
        }
        Ok(Box::new(Buffer(self.dropped.clone())))
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
fn all_upload_domains_reserve_before_validation_and_release_on_every_fault() {
    for case in 0..9 {
        let payload = TensorPayloadBudget::new(if case == 0 { 3 } else { 4 });
        let host = TensorPayloadBudget::new(if case == 1 { 7 } else { 8 });
        let device = TensorPayloadBudget::new(if case == 2 { 11 } else { 12 });
        let calls = Arc::new(AtomicUsize::new(0));
        let dropped = Arc::new(AtomicUsize::new(0));
        let backend = Backend {
            case,
            payload: payload.clone(),
            host: host.clone(),
            device: device.clone(),
            calls: calls.clone(),
            dropped: dropped.clone(),
        };
        let registrations = if case == 7 {
            vec![]
        } else {
            vec![TENSOR_UPLOADED_REGISTRATION]
        };
        let mut recorder = EvidenceRecorder::new("planned-upload", registrations).unwrap();
        let result = upload_tensor_logged_with_resources(
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
            TensorUploadResources {
                payload: &payload,
                host_workspace: &host,
                device_workspace: &device,
            },
        );
        if case == 8 {
            let (handle, caps) = result.unwrap();
            assert_eq!(caps.payload_bytes, 4);
            assert_eq!(payload.used_bytes(), 4);
            assert_eq!(host.used_bytes(), 0);
            assert_eq!(device.used_bytes(), 0);
            assert_eq!(recorder.events().len(), 1);
            drop(handle);
            assert_eq!(dropped.load(Ordering::SeqCst), 1);
        } else {
            assert!(result.is_err(), "case {case}");
            assert!(recorder.events().is_empty(), "case {case}");
            if case < 4 {
                assert_eq!(calls.load(Ordering::SeqCst), 0);
            }
            if case == 6 || case == 7 {
                assert_eq!(dropped.load(Ordering::SeqCst), 1);
            }
        }
        assert_eq!(payload.used_bytes(), 0, "case {case}");
        assert_eq!(host.used_bytes(), 0, "case {case}");
        assert_eq!(device.used_bytes(), 0, "case {case}");
    }
}
