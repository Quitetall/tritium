//! Real hybrid source/runner limits over the existing frozen small graph fixture.
use super::*;
use tritium_nn::{NnError, StreamLimits};

fn bounds() -> StreamLimits {
    // FA: 2 * context4 * KV-head1 * head-dim4 * f32 =128.
    // GDN: 2 * (conv-width8 * kernel4 + value-heads2 * dk2 * dv2) * f32 =320.
    StreamLimits {
        max_context: 4,
        max_segment_tokens: 2,
        max_host_cache_payload_bytes: 448,
    }
}
fn bounded(
    source: ModelSource<'_>,
    backend: Box<dyn TernaryBackend>,
    budget: &TensorPayloadBudget,
    policy: StreamLimits,
    evidence: &mut EvidenceRecorder,
) -> Result<tritium_nn::LoadedModel, NnError> {
    tritium_nn::load_with_limits(
        source,
        backend,
        TensorUploadPolicy::default(),
        budget,
        policy,
        TensorUploadContext {
            recorder: evidence,
            span: "load",
            parent: None,
            logical_time: 0,
        },
    )
}

#[test]
fn hybrid_capacity_is_admitted_before_any_weight_upload_or_cache_creation() {
    let root = Scratch::new("bounded-hybrid-reject");
    let package = text_package(false, false);
    let bytes = package.bytes();
    let path = root.0.join("model.trit");
    std::fs::write(&path, &bytes).unwrap();
    for source in [
        ModelSource::Trit {
            bytes: &bytes,
            level: "typed",
        },
        file_source(&path),
    ] {
        for policy in [
            StreamLimits {
                max_host_cache_payload_bytes: 447,
                ..bounds()
            },
            StreamLimits {
                max_context: 33,
                max_host_cache_payload_bytes: u64::MAX,
                ..bounds()
            },
        ] {
            let uploads = Arc::new(AtomicUsize::new(0));
            let budget = TensorPayloadBudget::new(1_000_000);
            let mut evidence = recorder();
            let result = bounded(
                source,
                Box::new(Probe {
                    count: uploads.clone(),
                    fail_at: None,
                    mutate: None,
                }),
                &budget,
                policy,
                &mut evidence,
            );
            assert!(matches!(result, Err(NnError::ResourceExhausted(_))));
            assert_eq!(uploads.load(Ordering::SeqCst), 0);
            assert_eq!(budget.used_bytes(), 0);
            assert!(evidence.events().is_empty());
        }
    }
}

#[test]
fn bounded_hybrid_executes_identical_prefill_and_continuation_with_private_cache_limits() {
    let root = Scratch::new("bounded-hybrid-execute");
    for uniform in [false, true] {
        let package = text_package(uniform, false);
        let bytes = package.bytes();
        let path = root.0.join("model.trit");
        std::fs::write(&path, &bytes).unwrap();
        for source in [
            ModelSource::Trit {
                bytes: &bytes,
                level: "typed",
            },
            file_source(&path),
        ] {
            let budget = TensorPayloadBudget::new(1_000_000);
            let mut loaded = bounded(
                source,
                Box::new(tritium_cpu::CpuBackend::new()),
                &budget,
                bounds(),
                &mut recorder(),
            )
            .unwrap();
            let mut oracle = load(
                source,
                Box::new(tritium_cpu::CpuBackend::new()),
                &budget,
                &mut recorder(),
            )
            .unwrap();
            let admission = loaded.runner.cache_admission().unwrap();
            assert_eq!(admission.limits(), bounds());
            assert_eq!(admission.attention_kv_bytes(), 128);
            assert_eq!(admission.delta_cache_bytes(), 320);
            assert_eq!(admission.logical_host_cache_payload_bytes(), 448);
            assert_eq!(loaded.runner.max_context(), 4);
            assert_eq!(loaded.runner.architecture().max_context(), 32);
            assert_eq!(loaded.runner.mtp_status(), oracle.runner.mtp_status());
            assert_eq!(
                loaded.runner.prefill(&[1, 2]).unwrap(),
                oracle.runner.prefill(&[1, 2]).unwrap()
            );
            assert!(loaded.runner.forward(&[3, 4, 5], &[2, 3, 4]).is_err());
            assert_eq!(loaded.runner.cache_len(), 2);
            for token in [4, 5] {
                assert_eq!(
                    loaded.runner.step(token).unwrap(),
                    oracle.runner.step(token).unwrap()
                );
            }
            assert!(loaded.runner.step(6).is_err());
            assert_eq!(loaded.runner.cache_len(), 4);
            loaded.runner.reset();
            oracle.runner.reset();
            assert_eq!(loaded.runner.cache_admission(), Some(admission));
            assert!(loaded.runner.prefill(&[1, 2, 3]).is_err());
            assert_eq!(loaded.runner.cache_len(), 0);
            assert_eq!(
                loaded.runner.prefill(&[1, 2]).unwrap(),
                oracle.runner.prefill(&[1, 2]).unwrap()
            );
            drop(loaded);
            drop(oracle);
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}
