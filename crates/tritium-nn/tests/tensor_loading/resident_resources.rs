//! Real CUDA graph with an audit adapter; not physical-memory qualification.
use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use tritium_evidence::{EvidenceEvent, ModelLoaded};
use tritium_spec::*;

const LIMIT: usize = 1_000_000;

#[test]
fn source_plans_match_real_standard_variants_ties_and_rotation() {
    let mut cases = 0;
    for arch in ["llama", "qwen2", "qwen3", "bitnet"] {
        for tied in [false, true] {
            for additive in [
                None,
                Some((PlaneCodec::B3, Basis::Hadamard { block: 128 })),
                Some((
                    PlaneCodec::D2,
                    Basis::SignedRht {
                        block: 128,
                        seed: 19,
                        domain: 27,
                    },
                )),
                Some((PlaneCodec::S34, Basis::Identity)),
            ] {
                let (_hf, bytes) = fixture(arch, tied, additive);
                let ledgers: [TensorPayloadBudget; 5] =
                    std::array::from_fn(|_| TensorPayloadBudget::new(LIMIT as u64));
                let mut evidence = EvidenceRecorder::new(
                    "source-resident-variants",
                    [
                        TENSOR_UPLOADED_REGISTRATION,
                        tritium_evidence::MODEL_LOADED_REGISTRATION,
                    ],
                )
                .unwrap();
                let mut loaded = tritium_nn::load_streamed_with_resources(
                    ModelSource::Trit {
                        bytes: &bytes,
                        level: "standard",
                    },
                    Box::new(cuda(Some(LIMIT)).expect("physical CUDA")),
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
                        max_host_cache_payload_bytes: LIMIT as u64,
                    },
                    TensorUploadContext {
                        recorder: &mut evidence,
                        span: "load",
                        parent: None,
                        logical_time: 0,
                    },
                )
                .unwrap();
                let admission = loaded.runner.resource_admission().unwrap();
                let plan = admission.resident_device_payload_plan.unwrap();
                assert_eq!(plan.matrix_payload_bytes() as u64, admission.payload_bytes);
                assert_eq!(
                    Some(plan.payload_bytes()),
                    loaded.runner.resident_payload_bytes()
                );
                assert_eq!(ledgers[0].used_bytes(), LIMIT as u64);
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
                compare(
                    &loaded.runner.step(3).unwrap(),
                    &cpu.runner.step(3).unwrap(),
                );
                drop(loaded);
                assert!(ledgers.iter().all(|l| l.used_bytes() == 0));
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 32);
    eprintln!("SOURCE_PLANNED_STANDARD_GRAPHS={cases}");
}

#[derive(Clone, Copy)]
enum PlanFault {
    None,
    Unknown,
    Matrix,
    Large,
    Drift,
}
struct AuditBackend {
    inner: Arc<tritium_cuda::CudaBackend>,
    ledger: TensorPayloadBudget,
    selected: Arc<AtomicUsize>,
    drift: bool,
    report: Option<usize>,
    calls: Arc<AtomicUsize>,
    dropped: Arc<AtomicBool>,
    plan_fault: PlanFault,
    plan_calls: AtomicUsize,
    uploads: Arc<AtomicUsize>,
    underreport: bool,
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
    fn decoder_payload_plan(
        &self,
        metadata: DecoderAllocationMetadata<'_>,
        limits: DecoderLimits,
    ) -> Result<Option<DecoderPayloadPlan>, BackendError> {
        let call = self.plan_calls.fetch_add(1, Ordering::SeqCst);
        let Some(plan) = self.inner.decoder_payload_plan(metadata, limits)? else {
            panic!("known CUDA planner")
        };
        if matches!(self.plan_fault, PlanFault::Unknown) {
            return Ok(None);
        }
        let matrix = plan.matrix_payload_bytes()
            + usize::from(matches!(self.plan_fault, PlanFault::Matrix)) * 4;
        let extra = match self.plan_fault {
            PlanFault::Large => LIMIT,
            PlanFault::Drift if call > 0 => 4,
            _ => 0,
        };
        DecoderPayloadPlan::new(
            matrix,
            plan.preserved_payload_bytes(),
            plan.state_payload_bytes(),
            plan.workspace_payload_bytes() + extra,
        )
        .map(Some)
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
        self.uploads.fetch_add(1, Ordering::SeqCst);
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
            underreport: self.underreport,
        })))
    }
}
struct AuditDecoder {
    inner: Option<Box<dyn ResidentDecoder>>,
    ledger: TensorPayloadBudget,
    report: Option<usize>,
    dropped: Arc<AtomicBool>,
    underreport: bool,
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
        self.report.unwrap_or_else(|| {
            self.inner.as_ref().unwrap().payload_bytes() - if self.underreport { 4 } else { 0 }
        })
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
                    plan_fault: PlanFault::None,
                    plan_calls: AtomicUsize::new(0),
                    uploads: Arc::default(),
                    underreport: false,
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

#[test]
fn resident_structural_declarations_reject_early_or_rollback_before_success() {
    let (_hf, bytes) = fixture("llama", true, None);
    for (fault, underreport, early, factory, expected) in [
        (
            PlanFault::Unknown,
            false,
            true,
            0,
            "resident structural plan is unknown",
        ),
        (PlanFault::Matrix, false, true, 0, "matrix payload differs"),
        (
            PlanFault::Large,
            false,
            true,
            0,
            "exceeds configured envelope",
        ),
        (PlanFault::Drift, false, false, 0, "structural plan changed"),
        (PlanFault::None, true, false, 1, "differs from envelope"),
    ] {
        let ledgers: [TensorPayloadBudget; 5] =
            std::array::from_fn(|_| TensorPayloadBudget::new(LIMIT as u64));
        let uploads = Arc::new(AtomicUsize::new(0));
        let calls = Arc::new(AtomicUsize::new(0));
        let dropped = Arc::new(AtomicBool::new(false));
        let backend = AuditBackend {
            inner: Arc::new(
                tritium_cuda::CudaBackend::new(0)
                    .expect("requires physical CUDA")
                    .with_resident_decoder(LIMIT),
            ),
            ledger: ledgers[0].clone(),
            selected: Arc::new(AtomicUsize::new(LIMIT)),
            drift: false,
            report: None,
            calls: calls.clone(),
            dropped: dropped.clone(),
            plan_fault: fault,
            plan_calls: AtomicUsize::new(0),
            uploads: uploads.clone(),
            underreport,
        };
        let mut evidence = EvidenceRecorder::new(
            "structural-plan-refusal",
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
                max_host_cache_payload_bytes: LIMIT as u64,
            },
            TensorUploadContext {
                recorder: &mut evidence,
                span: "load",
                parent: None,
                logical_time: 0,
            },
        );
        let Err(error) = result else {
            panic!("bad declaration must refuse")
        };
        assert!(error.to_string().contains(expected), "{error}");
        assert_eq!(uploads.load(Ordering::SeqCst) == 0, early);
        assert_eq!(calls.load(Ordering::SeqCst), factory);
        assert_eq!(dropped.load(Ordering::SeqCst), factory != 0);
        assert!(
            !evidence
                .events()
                .iter()
                .any(|e| e.schema == ModelLoaded::SCHEMA_ID)
        );
        assert!(ledgers.iter().all(|b| b.used_bytes() == 0));
    }
}
