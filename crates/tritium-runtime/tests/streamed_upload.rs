use std::{
    cell::Cell,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use tritium_evidence::{EvidenceRecorder, TENSOR_UPLOADED_REGISTRATION};
use tritium_runtime::{
    TensorPayloadBudget, TensorUploadContext, TensorUploadResources,
    upload_tensor_stream_logged_with_resources,
};
use tritium_schema::ScalePrecision;
use tritium_spec::*;

struct Source {
    case: usize,
    calls: Cell<usize>,
}
impl TensorStreamSource for Source {
    fn metadata(&self) -> TensorMetadata<'_> {
        TensorMetadata::Dense { rows: 1, cols: 1 }
    }
    fn encoding(&self) -> TensorStreamEncoding {
        TensorStreamEncoding::DenseLittleEndian(ScalePrecision::F16)
    }
    fn encoded_bytes(&self) -> u64 {
        if self.case == 4 { 3 } else { 2 }
    }
    fn visit(
        &self,
        visitor: &mut dyn FnMut(&[u8]) -> Result<(), BackendError>,
    ) -> Result<(), BackendError> {
        self.calls.set(self.calls.get() + 1);
        if self.case == 6 {
            visitor(&vec![0; TENSOR_STREAM_CHUNK_BYTES + 1])?;
        }
        let result = visitor(if self.case == 5 { &[0] } else { &[0, 0] });
        if self.case != 14 {
            result?;
        }
        if self.case == 7 {
            return Err(BackendError::Backend("terminal source fault".into()));
        }
        Ok(())
    }
}
struct Buffer {
    bytes: usize,
    dropped: Arc<AtomicUsize>,
}
impl Drop for Buffer {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}
impl DeviceBuffer for Buffer {
    fn len_bytes(&self) -> usize {
        self.bytes
    }
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}
struct Backend {
    case: usize,
    budgets: [TensorPayloadBudget; 3],
    plans: AtomicUsize,
    dropped: Arc<AtomicUsize>,
}
impl TernaryBackend for Backend {
    fn device_id(&self) -> &str {
        "stream-test"
    }
    fn capabilities(&self) -> DeviceCaps {
        DeviceCaps::new("stream-test", "fixture")
    }
    fn tensor_stream_plan(
        &self,
        _: TensorMetadata<'_>,
        _: TensorStreamEncoding,
    ) -> Result<Option<TensorStreamPlan>, BackendError> {
        let call = self.plans.fetch_add(1, Ordering::SeqCst);
        Ok((self.case != 3).then_some(TensorStreamPlan {
            upload: TensorUploadPlan {
                caps: TensorCaps {
                    execution: if self.case == 10 && call > 0 {
                        TensorExecution::Emulated
                    } else {
                        TensorExecution::Native
                    },
                    payload_bytes: 4,
                },
                host_workspace_bytes: 8,
                device_workspace_bytes: 12,
            },
            encoded_bytes: 2,
        }))
    }
    fn upload_tensor_stream(
        &self,
        source: &dyn TensorStreamSource,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        assert_eq!(self.budgets[0].used_bytes(), 4);
        assert_eq!(self.budgets[1].used_bytes(), 8);
        assert_eq!(self.budgets[2].used_bytes(), 12);
        if self.case == 8 {
            return Err(BackendError::Backend("backend stream fault".into()));
        }
        if self.case != 12 {
            source.visit(&mut |_| {
                if self.case == 14 {
                    Err(BackendError::Backend("callback rejected".into()))
                } else {
                    Ok(())
                }
            })?;
        }
        Ok(Box::new(Buffer {
            bytes: if self.case == 9 { 8 } else { 4 },
            dropped: self.dropped.clone(),
        }))
    }
    fn upload_tensor(&self, _: TensorView<'_>) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        panic!("decoded fallback forbidden")
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
fn streamed_reservations_source_checks_no_fallback_faults_and_final_owner_are_observable() {
    for case in 0..15 {
        let budgets = [
            TensorPayloadBudget::new(if case == 0 { 3 } else { 4 }),
            TensorPayloadBudget::new(if case == 1 { 7 } else { 8 }),
            TensorPayloadBudget::new(if case == 2 { 11 } else { 12 }),
        ];
        let dropped = Arc::new(AtomicUsize::new(0));
        let backend = Backend {
            case,
            budgets: budgets.clone(),
            plans: AtomicUsize::new(0),
            dropped: dropped.clone(),
        };
        let source = Source {
            case,
            calls: Cell::new(0),
        };
        let registrations = if case == 11 {
            vec![]
        } else {
            vec![TENSOR_UPLOADED_REGISTRATION]
        };
        let mut recorder = EvidenceRecorder::new("stream-faults", registrations).unwrap();
        let result = upload_tensor_stream_logged_with_resources(
            &backend,
            &source,
            TensorUploadPolicy::default(),
            "weight",
            TensorUploadContext {
                recorder: &mut recorder,
                span: "load",
                parent: None,
                logical_time: 0,
            },
            TensorUploadResources {
                payload: &budgets[0],
                host_workspace: &budgets[1],
                device_workspace: &budgets[2],
            },
        );
        if case == 13 {
            let (buffer, caps) = result.unwrap();
            assert_eq!(caps.payload_bytes, 4);
            assert_eq!(budgets[0].used_bytes(), 4);
            assert!(budgets[1..].iter().all(|b| b.used_bytes() == 0));
            assert_eq!(recorder.events().len(), 1);
            assert_eq!(recorder.events()[0].payload["decoded_source_bytes"], 4);
            let handle: Arc<dyn DeviceBuffer> = Arc::from(buffer);
            let last = handle.clone();
            drop(handle);
            assert_eq!(budgets[0].used_bytes(), 4);
            drop(last);
            assert_eq!(dropped.load(Ordering::SeqCst), 1);
        } else {
            assert!(result.is_err(), "case {case}");
            assert!(recorder.events().is_empty(), "case {case}");
            if matches!(case, 0..=4 | 8 | 11 | 12) {
                assert_eq!(source.calls.get(), 0, "case {case}");
            }
            if matches!(case, 9 | 10 | 12) {
                assert_eq!(dropped.load(Ordering::SeqCst), 1, "case {case}");
            }
        }
        assert!(budgets.iter().all(|b| b.used_bytes() == 0), "case {case}");
    }
}
