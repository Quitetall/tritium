//! Public mixer transaction tests, not numerical or physical-device evidence.

use super::*;
#[path = "../fixtures/resident_deltanet.rs"]
mod adapter;
use adapter::{ALLOC, Adapter, OUTPUT, STAGE, STEP, WRITE, WRONG_GEOMETRY};

#[test]
fn public_dispatch_reuses_owner_without_readback_and_reset_releases_it() {
    let backend = Adapter::new();
    let mixer = official_layer();
    let mut cache = mixer.new_cache().unwrap();
    let mut out = [0.0; 8];
    mixer
        .forward(&backend, &[0.2; 8], 2, &mut cache, &mut out)
        .unwrap();
    assert!(cache.is_device_resident());
    assert_eq!(cache.len(), 2);
    let mut step = [0.0; 4];
    mixer
        .forward(&backend, &[0.3; 4], 1, &mut cache, &mut step)
        .unwrap();
    {
        let log = backend.record.lock().unwrap();
        assert_eq!(
            (log.creations, log.stages, log.steps, log.reads),
            (1, 2, 3, 0)
        );
        assert!(log.writes.is_empty());
        assert_eq!(log.committed, vec![3.0; 8]);
    }
    cache.reset();
    assert!(!cache.is_device_resident());
    assert_eq!(cache.len(), 0);
    assert_eq!(backend.record.lock().unwrap().drops, 1);
    assert!(cache.recurrent_state().iter().all(|n| *n == 0.0));
    mixer
        .forward(&backend, &[0.3; 4], 1, &mut cache, &mut step)
        .unwrap();
    assert_eq!(backend.record.lock().unwrap().committed, vec![1.0; 8]);
}

#[test]
fn allocation_or_wrong_geometry_is_an_error_not_host_fallback() {
    for point in [ALLOC, WRONG_GEOMETRY] {
        let backend = Adapter::new();
        backend.fault.store(point, Ordering::SeqCst);
        let mixer = official_layer();
        let mut cache = mixer.new_cache().unwrap();
        let mut out = [123.0; 4];
        assert!(
            mixer
                .forward(&backend, &[0.3; 4], 1, &mut cache, &mut out)
                .is_err()
        );
        assert_eq!(out, [123.0; 4]);
        assert_eq!(cache.len(), 0);
        assert!(!cache.is_device_resident());
        assert!(cache.recurrent_state().iter().all(|n| *n == 0.0));
    }
}

#[test]
fn stage_step_and_downstream_errors_preserve_prefix_and_retry_from_commit() {
    for point in [STAGE, STEP, OUTPUT] {
        let backend = Adapter::new();
        // The real output projection calls the backend after resident step.
        let weights = Qwen35DeltaNetWeights::new(
            dense_a8(&QKV_PROJ, 8, 4),
            dense_a8(&Z_PROJ, 4, 4),
            dense_a8(&B_PROJ, 2, 4),
            dense_a8(&A_PROJ, 2, 4),
            Projection::Ternary(TernaryLinear::new(&backend, &[Trit::POS; 16], 4, 4, 1.0).unwrap()),
            CONV.to_vec(),
            vec![1.10, 0.90],
            vec![-0.40, 0.20],
            A_LOG.to_vec(),
        );
        let mixer = Qwen35DeltaNet::new(&text_config(), weights).unwrap();
        let mut cache = mixer.new_cache().unwrap();
        let mut out = [0.0; 4];
        mixer
            .forward(&backend, &[0.3; 4], 1, &mut cache, &mut out)
            .unwrap();
        let conv = cache.conv_state().to_vec();
        backend.fault.store(point, Ordering::SeqCst);
        out.fill(123.0);
        assert!(
            mixer
                .forward(&backend, &[0.4; 4], 1, &mut cache, &mut out)
                .is_err()
        );
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.conv_state(), conv);
        assert_eq!(out, [123.0; 4]);
        assert_eq!(backend.record.lock().unwrap().committed, vec![1.0; 8]);
        backend.fault.store(0, Ordering::SeqCst);
        mixer
            .forward(&backend, &[0.4; 4], 1, &mut cache, &mut out)
            .unwrap();
        assert_eq!(cache.len(), 2);
        assert_eq!(backend.record.lock().unwrap().committed, vec![2.0; 8]);
    }
}

#[test]
fn same_device_label_does_not_authorize_another_owner_or_stale_host_fallback() {
    let backend = Adapter::new();
    let other = Adapter::new();
    let mixer = official_layer();
    let mut cache = mixer.new_cache().unwrap();
    let mut out = [0.0; 4];
    mixer
        .forward(&backend, &[0.3; 4], 1, &mut cache, &mut out)
        .unwrap();
    for wrong in [&other as &dyn TernaryBackend, &backend.cpu] {
        out.fill(123.0);
        assert!(
            mixer
                .forward(wrong, &[0.4; 4], 1, &mut cache, &mut out)
                .is_err()
        );
        assert_eq!(out, [123.0; 4]);
        assert_eq!(cache.len(), 1);
        assert_eq!(backend.record.lock().unwrap().committed, vec![1.0; 8]);
    }
    cache.reset();
    mixer
        .forward(&backend.cpu, &[0.4; 4], 1, &mut cache, &mut out)
        .unwrap();
    assert!(!cache.is_device_resident());
}

#[test]
fn unsupported_then_late_residency_transfers_exact_host_prefix_once() {
    for fail_initialization in [false, true] {
        let backend = Adapter::new();
        backend.supported.store(false, Ordering::SeqCst);
        let mixer = official_layer();
        let mut cache = mixer.new_cache().unwrap();
        let mut out = [0.0; 4];
        mixer
            .forward(&backend, &[0.3; 4], 1, &mut cache, &mut out)
            .unwrap();
        assert!(!cache.is_device_resident());
        let prefix = cache.recurrent_state().to_vec();
        assert!(prefix.iter().any(|n| *n != 0.0));
        backend.supported.store(true, Ordering::SeqCst);
        if fail_initialization {
            backend.fault.store(WRITE, Ordering::SeqCst);
            out.fill(123.0);
            assert!(
                mixer
                    .forward(&backend, &[0.4; 4], 1, &mut cache, &mut out)
                    .is_err()
            );
            assert_eq!(out, [123.0; 4]);
            assert_eq!(cache.len(), 1);
            assert_eq!(cache.recurrent_state(), prefix);
            assert!(!cache.is_device_resident());
            backend.fault.store(0, Ordering::SeqCst);
        }
        mixer
            .forward(&backend, &[0.4; 4], 1, &mut cache, &mut out)
            .unwrap();
        let log = backend.record.lock().unwrap();
        assert!(log.writes.iter().all(|v| *v == prefix));
        assert_eq!(log.writes.len(), if fail_initialization { 2 } else { 1 });
        assert_eq!(
            log.committed,
            prefix.iter().map(|n| n + 1.0).collect::<Vec<_>>()
        );
        assert_eq!(log.reads, 0);
        assert_eq!(cache.len(), 2);
    }
}
