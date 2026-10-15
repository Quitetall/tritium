//! Real CUDA graph with an audit adapter; not physical-memory qualification.
use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use tritium_evidence::{EvidenceEvent, ModelLoaded};
use tritium_spec::*;

const LIMIT: usize = 1_000_000;
struct AuditBackend {
    inner: Arc<tritium_cuda::CudaBackend>,
    ledger: TensorPayloadBudget,
    selected: Arc<AtomicUsize>,
    drift: bool,
    report: Option<usize>,
    calls: Arc<AtomicUsize>,
    dropped: Arc<AtomicBool>,
}
impl TernaryBackend for AuditBackend {
    fn device_id(&self) -> &str {
        self.inner.device_id()
    }
    fn physical_device_id(&self) -> &str {
        self.inner.physical_device_id()
    }
    fn capabilities(&self) -> DeviceCaps {
        self.inner.capabilities()
    }
    fn resident_decoder_requested(&self) -> bool {
        true
    }
    fn resident_decoder_payload_limit(&self) -> Option<usize> {
        Some(self.selected.load(Ordering::SeqCst))
    }
    fn tensor_caps(&self, view: TensorView<'_>) -> Result<Option<TensorCaps>, BackendError> {
        self.inner.tensor_caps(view)
    }
    fn tensor_upload_plan(
        &self,
        metadata: TensorMetadata<'_>,
    ) -> Result<Option<TensorUploadPlan>, BackendError> {
        self.inner.tensor_upload_plan(metadata)
    }
    fn tensor_stream_plan(
        &self,
        metadata: TensorMetadata<'_>,
        encoding: TensorStreamEncoding,
    ) -> Result<Option<TensorStreamPlan>, BackendError> {
        self.inner.tensor_stream_plan(metadata, encoding)
    }
    fn upload_tensor_stream(
        &self,
        source: &dyn TensorStreamSource,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        let result = self.inner.upload_tensor_stream(source)?;
        if self.drift {
            self.selected.store(LIMIT + 1, Ordering::SeqCst);
        }
        Ok(result)
    }
    fn upload_tensor(&self, view: TensorView<'_>) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        self.inner.upload_tensor(view)
    }
    fn mpgemm(&self, p: MpGemm<'_>) -> Result<(), BackendError> {
        self.inner.mpgemm(p)
    }
    fn upload_weights(
        &self,
        packed: &[u8],
        shape: tritium_core::GemmShape,
        format: tritium_core::TernaryFormat,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        self.inner.upload_weights(packed, shape, format)
    }
    fn resident_decoder(
        self: Arc<Self>,
        graph: DecoderGraph<'_>,
        limits: DecoderLimits,
    ) -> Result<Option<Box<dyn ResidentDecoder>>, BackendError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let Some(inner) = self.inner.clone().resident_decoder(graph, limits)? else {
            panic!("CUDA factory unexpectedly declined")
        };
        Ok(Some(Box::new(AuditDecoder {
            inner: Some(inner),
            ledger: self.ledger.clone(),
            report: self.report,
            dropped: self.dropped.clone(),
        })))
    }
}
struct AuditDecoder {
    inner: Option<Box<dyn ResidentDecoder>>,
    ledger: TensorPayloadBudget,
    report: Option<usize>,
    dropped: Arc<AtomicBool>,
}
impl Drop for AuditDecoder {
    fn drop(&mut self) {
        // All graph-owned matrix/state/scratch handles have now been dropped.
        // Runner must still hold the complete parent envelope at this point.
        drop(self.inner.take());
        assert_eq!(self.ledger.used_bytes(), LIMIT as u64);
        self.dropped.store(true, Ordering::SeqCst);
    }
}
impl ResidentDecoder for AuditDecoder {
    fn limits(&self) -> DecoderLimits {
        self.inner.as_ref().unwrap().limits()
    }
    fn payload_bytes(&self) -> usize {
        self.report
            .unwrap_or_else(|| self.inner.as_ref().unwrap().payload_bytes())
    }
    fn len(&self) -> usize {
        self.inner.as_ref().unwrap().len()
    }
    fn reset(&mut self) {
        self.inner.as_mut().unwrap().reset();
    }
    fn forward(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Option<Vec<f32>>, BackendError> {
        self.inner
            .as_mut()
            .unwrap()
            .forward(tokens, positions, cancelled)
    }
}

#[test]
fn physical_resident_reservation_outlives_graph_drop_and_rejects_selection_or_payload_drift() {
    for tied in [false, true] {
        for additive in [
            None,
            Some((
                PlaneCodec::D2,
                Basis::SignedRht {
                    block: 128,
                    seed: 7,
                    domain: 9,
                },
            )),
        ] {
            let (_hf, bytes) = fixture("llama", tied, additive);
            for (drift, report) in [
                (false, None),
                (true, None),
                (false, Some(0)),
                (false, Some(LIMIT + 1)),
            ] {
                let ledgers: [TensorPayloadBudget; 5] =
                    std::array::from_fn(|_| TensorPayloadBudget::new(LIMIT as u64));
                let dropped = Arc::new(AtomicBool::new(false));
                let calls = Arc::new(AtomicUsize::new(0));
                let backend = AuditBackend {
                    inner: Arc::new(
                        tritium_cuda::CudaBackend::new(0)
                            .expect("requires physical CUDA")
                            .with_resident_decoder(LIMIT),
                    ),
                    ledger: ledgers[0].clone(),
                    selected: Arc::new(AtomicUsize::new(LIMIT)),
                    drift,
                    report,
                    calls: calls.clone(),
                    dropped: dropped.clone(),
                };
                let mut evidence = EvidenceRecorder::new(
                    "resident-envelope-audit",
                    [
                        TENSOR_UPLOADED_REGISTRATION,
                        tritium_evidence::MODEL_LOADED_REGISTRATION,
                    ],
                )
                .unwrap();
                let result = tritium_nn::load_streamed_with_resources(
                    ModelSource::Trit {
                        bytes: &bytes,
                        level: "standard",
                    },
                    Box::new(backend),
                    TensorUploadPolicy::default(),
                    tritium_nn::ModelLoadResources {
                        payload: &ledgers[0],
                        source_workspace: &ledgers[1],
                        host_workspace: &ledgers[2],
                        device_workspace: &ledgers[3],
                        retained_host: &ledgers[4],
                    },
                    StreamLimits {
                        max_context: 7,
                        max_segment_tokens: 3,
                        max_host_cache_payload_bytes: 1_000_000,
                    },
                    TensorUploadContext {
                        recorder: &mut evidence,
                        span: "load",
                        parent: None,
                        logical_time: 0,
                    },
                );
                if drift || report.is_some() {
                    let error = result.err().unwrap().to_string();
                    assert!(
                        error.contains(if drift {
                            "envelope changed"
                        } else {
                            "differs from envelope"
                        }),
                        "{error}"
                    );
                    assert!(
                        !evidence
                            .events()
                            .iter()
                            .any(|e| e.schema == ModelLoaded::SCHEMA_ID)
                    );
                } else {
                    let mut loaded = result.unwrap();
                    let mut cpu = load(
                        &bytes,
                        Box::new(tritium_cpu::CpuBackend::new()),
                        &TensorPayloadBudget::new(LIMIT as u64),
                    )
                    .unwrap();
                    compare(
                        &loaded.runner.prefill(&[1, 2]).unwrap(),
                        &cpu.runner.prefill(&[1, 2]).unwrap(),
                    );
                    let runner = loaded.runner;
                    assert_eq!(ledgers[0].used_bytes(), LIMIT as u64);
                    drop(runner);
                }
                assert_eq!(calls.load(Ordering::SeqCst), usize::from(!drift));
                assert_eq!(dropped.load(Ordering::SeqCst), !drift);
                assert!(ledgers.iter().all(|l| l.used_bytes() == 0));
            }
        }
    }
}
