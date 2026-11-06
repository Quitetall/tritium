//! Controlled ownership/admission tests, not physical-memory qualification.

use core::any::Any;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tritium_evidence::{EvidenceRecorder, TENSOR_UPLOADED_REGISTRATION};
use tritium_runtime::{
    TensorPayloadBudget, TensorUploadContext, TensorUploadError, upload_tensor_logged_admitted,
};
use tritium_spec::{
    BackendError, DeviceBuffer, DeviceCaps, GemmShape, MpGemm, TensorCaps, TensorExecution,
    TensorUploadPolicy, TensorView, TernaryBackend, TernaryFormat,
};

struct Backend {
    uploads: AtomicUsize,
    drops: Arc<AtomicUsize>,
    caps_calls: AtomicUsize,
    scenario: Scenario,
    drop_observer: Option<TensorPayloadBudget>,
}
#[derive(Clone, Copy)]
enum Scenario {
    Normal,
    FailUpload,
    WrongBytes,
    DriftingCaps,
    Emulated,
    Unsupported,
}
struct Buffer(Arc<AtomicUsize>, usize, Option<TensorPayloadBudget>);
impl Drop for Buffer {
    fn drop(&mut self) {
        if let Some(budget) = &self.2 {
            assert!(
                budget.used_bytes() >= self.1 as u64,
                "payload admission released before buffer destruction"
            );
        }
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
impl DeviceBuffer for Buffer {
    fn len_bytes(&self) -> usize {
        self.1
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
        let call = self.caps_calls.fetch_add(1, Ordering::SeqCst);
        Ok(match self.scenario {
            Scenario::Unsupported => None,
            Scenario::Emulated => Some(TensorCaps {
                execution: TensorExecution::Emulated,
                payload_bytes: 4,
            }),
            Scenario::DriftingCaps if call != 0 => Some(TensorCaps {
                execution: TensorExecution::Native,
                payload_bytes: 8,
            }),
            _ => Some(TensorCaps {
                execution: TensorExecution::Native,
                payload_bytes: 4,
            }),
        })
    }
    fn upload_tensor(&self, _: TensorView<'_>) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        if matches!(self.scenario, Scenario::FailUpload) {
            return Err(BackendError::Backend("controlled failure".into()));
        }
        let bytes = if matches!(self.scenario, Scenario::WrongBytes | Scenario::DriftingCaps) {
            8
        } else {
            4
        };
        Ok(Box::new(Buffer(
            Arc::clone(&self.drops),
            bytes,
            self.drop_observer.clone(),
        )))
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
fn shared_budget_is_held_until_last_tensor_owner_drops() {
    let budget = TensorPayloadBudget::new(4);
    let mut backend = backend(Scenario::Normal);
    backend.drop_observer = Some(budget.clone());
    let mut recorder = EvidenceRecorder::new("fixture", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
    let (tensor, caps) = upload_tensor_logged_admitted(
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
        &budget,
    )
    .unwrap();
    assert_eq!(budget.used_bytes(), caps.payload_bytes);
    assert_eq!(budget.limit_bytes(), 4);
    assert!(
        tensor.as_any().is::<Buffer>(),
        "admission must preserve backend downcasts"
    );
    let shared: Arc<dyn DeviceBuffer> = Arc::from(tensor);
    let tied_head = Arc::clone(&shared);
    drop(shared);
    assert_eq!(budget.used_bytes(), 4);
    assert_eq!(backend.drops.load(Ordering::SeqCst), 0);
    drop(tied_head);
    assert_eq!(budget.used_bytes(), 0);
    assert_eq!(backend.drops.load(Ordering::SeqCst), 1);
}

fn backend(scenario: Scenario) -> Backend {
    Backend {
        uploads: AtomicUsize::new(0),
        drops: Arc::new(AtomicUsize::new(0)),
        caps_calls: AtomicUsize::new(0),
        scenario,
        drop_observer: None,
    }
}

fn upload(
    backend: &Backend,
    budget: &TensorPayloadBudget,
    recorder: &mut EvidenceRecorder,
) -> Result<(Box<dyn DeviceBuffer>, TensorCaps), TensorUploadError> {
    upload_tensor_logged_admitted(
        backend,
        TensorView::Dense {
            rows: 1,
            cols: 1,
            values: &[1.],
        },
        TensorUploadPolicy::default(),
        "weight",
        TensorUploadContext {
            recorder,
            span: "load",
            parent: None,
            logical_time: 0,
        },
        budget,
    )
}

#[test]
fn aggregate_limit_rejects_before_upload_without_evicting_live_tensors() {
    let backend = backend(Scenario::Normal);
    let budget = TensorPayloadBudget::new(8);
    let clone = budget.clone();
    let mut recorder = EvidenceRecorder::new("fixture", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
    let first = upload(&backend, &budget, &mut recorder).unwrap();
    let second = upload(&backend, &clone, &mut recorder).unwrap();
    let root = recorder.root_digest().unwrap();
    assert!(matches!(
        upload(&backend, &budget, &mut recorder),
        Err(TensorUploadError::Backend(
            BackendError::TensorPayloadBudgetExceeded {
                requested: 12,
                limit: 8
            }
        ))
    ));
    assert_eq!(budget.used_bytes(), 8);
    assert_eq!(backend.uploads.load(Ordering::SeqCst), 2);
    assert_eq!(backend.drops.load(Ordering::SeqCst), 0);
    assert_eq!(recorder.root_digest().unwrap(), root);
    drop(first);
    assert_eq!(clone.used_bytes(), 4);
    let replacement = upload(&backend, &budget, &mut recorder).unwrap();
    assert_eq!(budget.used_bytes(), 8);
    drop(second);
    drop(replacement);
    assert_eq!(clone.used_bytes(), 0);
    assert_eq!(backend.drops.load(Ordering::SeqCst), 3);
}

#[test]
fn every_post_reservation_failure_releases_admission_and_emits_no_success() {
    for scenario in [
        Scenario::FailUpload,
        Scenario::WrongBytes,
        Scenario::DriftingCaps,
        Scenario::Normal,
    ] {
        let backend = backend(scenario);
        let budget = TensorPayloadBudget::new(8);
        let mut recorder = if matches!(scenario, Scenario::Normal) {
            EvidenceRecorder::new("fixture", []).unwrap()
        } else {
            EvidenceRecorder::new("fixture", [TENSOR_UPLOADED_REGISTRATION]).unwrap()
        };
        let result = upload(&backend, &budget, &mut recorder);
        match scenario {
            Scenario::FailUpload => assert!(matches!(
                result,
                Err(TensorUploadError::Backend(BackendError::Backend(_)))
            )),
            Scenario::WrongBytes => assert!(matches!(
                result,
                Err(TensorUploadError::Backend(
                    BackendError::TensorPayloadMismatch {
                        advertised: 4,
                        actual: 8
                    }
                ))
            )),
            Scenario::DriftingCaps => assert!(matches!(
                result,
                Err(TensorUploadError::Backend(BackendError::InvalidInput(_)))
            )),
            Scenario::Normal => assert!(matches!(result, Err(TensorUploadError::Evidence(_)))),
            _ => unreachable!(),
        }
        assert_eq!(budget.used_bytes(), 0);
        assert!(recorder.events().is_empty());
        assert_eq!(backend.uploads.load(Ordering::SeqCst), 1);
        assert_eq!(
            backend.drops.load(Ordering::SeqCst),
            usize::from(!matches!(scenario, Scenario::FailUpload))
        );
    }
}

#[test]
fn policy_unsupported_empty_name_and_invalid_geometry_never_upload() {
    for (scenario, policy, name, values) in [
        (
            Scenario::Emulated,
            TensorUploadPolicy {
                native_only: true,
                max_payload_bytes: None,
            },
            "weight",
            &[1.][..],
        ),
        (
            Scenario::Normal,
            TensorUploadPolicy {
                native_only: false,
                max_payload_bytes: Some(3),
            },
            "weight",
            &[1.][..],
        ),
        (
            Scenario::Unsupported,
            TensorUploadPolicy::default(),
            "weight",
            &[1.][..],
        ),
        (
            Scenario::Normal,
            TensorUploadPolicy::default(),
            "",
            &[1.][..],
        ),
        (
            Scenario::Normal,
            TensorUploadPolicy::default(),
            "weight",
            &[][..],
        ),
    ] {
        let backend = backend(scenario);
        let budget = TensorPayloadBudget::new(8);
        let mut recorder =
            EvidenceRecorder::new("fixture", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
        assert!(
            upload_tensor_logged_admitted(
                &backend,
                TensorView::Dense {
                    rows: 1,
                    cols: 1,
                    values
                },
                policy,
                name,
                TensorUploadContext {
                    recorder: &mut recorder,
                    span: "load",
                    parent: None,
                    logical_time: 0
                },
                &budget
            )
            .is_err()
        );
        assert_eq!(budget.used_bytes(), 0);
        assert_eq!(backend.uploads.load(Ordering::SeqCst), 0);
        assert!(recorder.events().is_empty());
    }
}

#[test]
fn concurrent_uploads_cannot_oversubscribe_a_shared_ledger() {
    let backend = backend(Scenario::Normal);
    let budget = TensorPayloadBudget::new(4);
    let barrier = std::sync::Barrier::new(8);
    let results = std::thread::scope(|scope| {
        let workers = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    let mut recorder =
                        EvidenceRecorder::new("fixture", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
                    barrier.wait();
                    let result = upload(&backend, &budget, &mut recorder);
                    assert_eq!(recorder.events().len(), usize::from(result.is_ok()));
                    result
                })
            })
            .collect::<Vec<_>>();
        workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
    assert_eq!(budget.used_bytes(), 4);
    assert_eq!(backend.uploads.load(Ordering::SeqCst), 1);
    drop(results);
    assert_eq!(budget.used_bytes(), 0);
    assert_eq!(backend.drops.load(Ordering::SeqCst), 1);
}
