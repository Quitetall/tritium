//! Full packed-version ownership and numerical fixtures, not release acceptance.
use std::sync::Arc;
use tritium_core::{AdditiveView, PerTilePlan, Trit, UniformPlan};
use tritium_cpu::CpuBackend;
use tritium_format::AdditiveTensor;
use tritium_runtime::TensorPayloadBudget;
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, ScalePrecision, Transport,
};
use tritium_spec::{
    AdditiveMetadata, BackendError, DeviceBuffer, TensorMetadata, TensorStreamEncoding, TensorView,
    TernaryBackend,
};
#[path = "common/snapshot_fixture.rs"]
mod snapshot_fixture;
#[path = "common/stream_source.rs"]
mod stream_source;
use snapshot_fixture::{check, tensor};

fn changed(source: &AdditiveTensor) -> AdditiveTensor {
    let mut result = source.clone();
    let mut trits = source.trits().to_vec();
    if source.layout().codec == PlaneCodec::S34 {
        for quartet in trits.as_chunks_mut::<4>().0 {
            quartet.rotate_left(1);
        }
    } else {
        for trit in &mut trits {
            *trit = Trit::from_i8(-trit.get()).unwrap();
        }
    }
    result.replace_trits(trits).unwrap();
    let exact = if source.layout().law.precision == ScalePrecision::F32 {
        f32::from_bits(0x3f00_0001)
    } else {
        0.75
    };
    result
        .replace_scales(
            (0..source.scales().len())
                .map(|i| [exact, 0.0, 0.25, 1.25][i % 4])
                .collect(),
        )
        .unwrap();
    result
}

#[test]
fn full_and_scale_versions_compose_without_retaining_obsolete_parents() {
    let cpu = CpuBackend::new();
    let mut cases = 0;
    for (law, admitted) in ADMITTED_LAWS.iter().enumerate() {
        for planes in 1..=admitted.max_planes {
            for group in [32, 128] {
                for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
                    for basis in [
                        Basis::Identity,
                        Basis::Hadamard { block: 128 },
                        Basis::SignedRht {
                            block: 128,
                            seed: 19,
                            domain: 7,
                        },
                    ] {
                        for mode in 0..3 {
                            let source = tensor(law, codec, basis, mode, group, planes);
                            let a = changed(&source);
                            let b = changed(&a);
                            if !source.trits().is_empty() {
                                assert_ne!(source.trits(), a.trits());
                            }
                            let encoded = source.encode_payload().unwrap();
                            let stream = stream_source::Source::new(
                                TensorMetadata::Additive(AdditiveMetadata::from_view(
                                    source.view(),
                                )),
                                TensorStreamEncoding::CanonicalRawAdditive,
                                &encoded,
                            );
                            let direct: Arc<dyn DeviceBuffer> = Arc::from(
                                cpu.upload_tensor(TensorView::Additive(source.view()))
                                    .unwrap(),
                            );
                            let streamed: Arc<dyn DeviceBuffer> =
                                Arc::from(cpu.upload_tensor_stream(&stream).unwrap());
                            for root in [direct, streamed] {
                                let bytes = root.len_bytes() as u64;
                                let scales = (a.scales().len()
                                    * if source.layout().law.precision == ScalePrecision::F16 {
                                        2
                                    } else {
                                        4
                                    }) as u64;
                                let budget = TensorPayloadBudget::new(2 * bytes + scales);
                                let sa = cpu
                                    .snapshot_additive_weights(&root, a.view(), &budget)
                                    .unwrap();
                                assert_eq!(sa.len_bytes() as u64, bytes);
                                assert_eq!(budget.used_bytes(), bytes);
                                assert_eq!(Arc::strong_count(&root), 1);
                                check(&*root, source.view());
                                check(&*sa, a.view());
                                let (overlay, _) = cpu
                                    .snapshot_additive_scales(&sa, b.scales(), &budget)
                                    .unwrap();
                                let mut overlay_values = a.clone();
                                overlay_values.replace_scales(b.scales().to_vec()).unwrap();
                                check(&*overlay, overlay_values.view());
                                let sb = cpu
                                    .snapshot_additive_weights(&overlay, b.view(), &budget)
                                    .unwrap();
                                assert_eq!(budget.used_bytes(), 2 * bytes + scales);
                                check(&*root, source.view());
                                check(&*sa, a.view());
                                check(&*sb, b.view());
                                let old = Arc::downgrade(&sa);
                                let old_overlay = Arc::downgrade(&overlay);
                                drop(sa);
                                assert!(old.upgrade().is_some());
                                drop(overlay);
                                assert!(old.upgrade().is_none());
                                assert!(old_overlay.upgrade().is_none());
                                assert_eq!(budget.used_bytes(), bytes);
                                let old_root = Arc::downgrade(&root);
                                drop(root);
                                assert!(old_root.upgrade().is_none());
                                // Even an unchanged complete candidate owns a new
                                // packed root; no hidden shared parent admission.
                                let sc = cpu
                                    .snapshot_additive_weights(&sb, b.view(), &budget)
                                    .unwrap();
                                let old_b = Arc::downgrade(&sb);
                                drop(sb);
                                assert!(old_b.upgrade().is_none());
                                assert_eq!(budget.used_bytes(), bytes);
                                check(&*sc, b.view());
                                let alias = Arc::clone(&sc);
                                drop(sc);
                                assert_eq!(budget.used_bytes(), bytes);
                                drop(alias);
                                assert_eq!(budget.used_bytes(), 0);
                                cases += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases, 1080);
}

fn with_geometry(layout: AdditiveLayout, planes: u8, counts: Option<Vec<u8>>) -> AdditiveTensor {
    let (n, s) = if let Some(counts) = &counts {
        let p = PerTilePlan::new(layout, counts).unwrap();
        (p.trit_count(), p.scale_count())
    } else {
        let p = UniformPlan::new(layout, planes).unwrap();
        (p.trit_count(), p.scale_count())
    };
    let trits = (0..n)
        .map(|i| [Trit::ZERO, Trit::POS, Trit::NEG, Trit::POS][i % 4])
        .collect();
    if let Some(counts) = counts {
        AdditiveTensor::new_per_tile(layout, counts, trits, vec![0.5; s]).unwrap()
    } else {
        AdditiveTensor::new(layout, planes, trits, vec![0.5; s]).unwrap()
    }
}

#[test]
fn geometry_mismatches_and_late_noncanonical_values_do_not_charge_or_change_parent() {
    let cpu = CpuBackend::new();
    let source = tensor(0, PlaneCodec::D2, Basis::Identity, 0, 32, 3);
    let parent: Arc<dyn DeviceBuffer> = Arc::from(
        cpu.upload_tensor(TensorView::Additive(source.view()))
            .unwrap(),
    );
    let layout = source.layout();
    let budget = TensorPayloadBudget::new(u64::MAX);
    let incompatible = [
        with_geometry(AdditiveLayout { rows: 4, ..layout }, 3, None),
        with_geometry(
            AdditiveLayout {
                cols: 264,
                ..layout
            },
            3,
            None,
        ),
        with_geometry(
            AdditiveLayout {
                group: 128,
                ..layout
            },
            3,
            None,
        ),
        with_geometry(
            AdditiveLayout {
                basis: Basis::Hadamard { block: 4 },
                ..layout
            },
            3,
            None,
        ),
        with_geometry(
            AdditiveLayout {
                codec: PlaneCodec::B3,
                ..layout
            },
            3,
            None,
        ),
        with_geometry(
            AdditiveLayout {
                transport: Transport::JointHuffman { block: 256 },
                ..layout
            },
            3,
            None,
        ),
        with_geometry(
            AdditiveLayout {
                law: ADMITTED_LAWS[1].law,
                ..layout
            },
            3,
            None,
        ),
        with_geometry(layout, 2, None),
        with_geometry(
            AdditiveLayout {
                allocation: PlaneAllocation::PerTile,
                ..layout
            },
            3,
            Some(vec![3; 6]),
        ),
    ];
    for candidate in &incompatible {
        assert!(matches!(
            cpu.snapshot_additive_weights(&parent, candidate.view(), &budget),
            Err(BackendError::InvalidInput(_))
        ));
        assert_eq!(budget.used_bytes(), 0);
        assert_eq!(Arc::strong_count(&parent), 1);
    }
    let mut scales = source.scales().to_vec();
    *scales.last_mut().unwrap() = 0.50001;
    let inexact = AdditiveView::new(layout, 3, source.trits(), &scales).unwrap();
    assert!(
        cpu.snapshot_additive_weights(&parent, inexact, &budget)
            .is_err()
    );
    *scales.last_mut().unwrap() = -0.0;
    let negative_zero = AdditiveView::new(layout, 3, source.trits(), &scales).unwrap();
    assert!(
        cpu.snapshot_additive_weights(&parent, negative_zero, &budget)
            .is_err()
    );
    assert_eq!(budget.used_bytes(), 0);
    check(&*parent, source.view());

    let sparse = tensor(0, PlaneCodec::S34, Basis::Identity, 0, 32, 3);
    let parent: Arc<dyn DeviceBuffer> = Arc::from(
        cpu.upload_tensor(TensorView::Additive(sparse.view()))
            .unwrap(),
    );
    let mut malformed = sparse.trits().to_vec();
    *malformed.last_mut().unwrap() = Trit::ZERO;
    let invalid = AdditiveView::new(sparse.layout(), 3, &malformed, sparse.scales()).unwrap();
    assert!(
        cpu.snapshot_additive_weights(&parent, invalid, &budget)
            .is_err()
    );
    assert_eq!(budget.used_bytes(), 0);
    check(&*parent, sparse.view());

    let compact = tensor(1, PlaneCodec::D2, Basis::Identity, 1, 32, 3);
    let parent: Arc<dyn DeviceBuffer> = Arc::from(
        cpu.upload_tensor(TensorView::Additive(compact.view()))
            .unwrap(),
    );
    let mut counts = compact.view().tile_counts().unwrap().to_vec();
    counts.swap(0, 4);
    let candidate = with_geometry(compact.layout(), 3, Some(counts));
    assert_eq!(candidate.trits().len(), compact.trits().len());
    assert_eq!(
        cpu.tensor_caps(TensorView::Additive(candidate.view()))
            .unwrap()
            .unwrap()
            .payload_bytes,
        parent.len_bytes() as u64
    );
    assert!(
        cpu.snapshot_additive_weights(&parent, candidate.view(), &budget)
            .is_err()
    );
    assert_eq!(budget.used_bytes(), 0);
    check(&*parent, compact.view());
}

#[test]
fn exact_and_concurrent_payload_admission_and_source_drop_are_lifetime_bound() {
    let cpu = CpuBackend::new();
    let source = tensor(1, PlaneCodec::B3, Basis::Identity, 1, 128, 3);
    let candidate = changed(&source);
    let parent: Arc<dyn DeviceBuffer> = Arc::from(
        cpu.upload_tensor(TensorView::Additive(source.view()))
            .unwrap(),
    );
    let bytes = parent.len_bytes() as u64;
    let short = TensorPayloadBudget::new(bytes - 1);
    assert!(matches!(
        cpu.snapshot_additive_weights(&parent, candidate.view(), &short),
        Err(BackendError::TensorPayloadBudgetExceeded { .. })
    ));
    assert_eq!(short.used_bytes(), 0);
    let budget = TensorPayloadBudget::new(bytes);
    let start = std::sync::Barrier::new(9);
    let finish = std::sync::Barrier::new(9);
    std::thread::scope(|scope| {
        let jobs: Vec<_> = (0..8)
            .map(|_| {
                let (parent, candidate, budget, start, finish) =
                    (&parent, &candidate, &budget, &start, &finish);
                scope.spawn(move || {
                    start.wait();
                    let result = CpuBackend::new().snapshot_additive_weights(
                        parent,
                        candidate.view(),
                        budget,
                    );
                    finish.wait();
                    result.is_ok()
                })
            })
            .collect();
        start.wait();
        finish.wait();
        assert_eq!(
            jobs.into_iter()
                .map(|j| j.join().unwrap())
                .filter(|ok| *ok)
                .count(),
            1
        );
    });
    assert_eq!(budget.used_bytes(), 0);
    check(&*parent, source.view());
    let snapshot = cpu
        .snapshot_additive_weights(&parent, candidate.view(), &budget)
        .unwrap();
    drop(parent);
    drop(source);
    drop(candidate);
    assert_eq!(snapshot.len_bytes() as u64, bytes);
    let mut rows = vec![0.0; 260];
    cpu.embed_rows(&*snapshot, &[2], &mut rows).unwrap();
    assert!(rows.iter().all(|v| v.is_finite()));
    assert!(rows.iter().any(|&v| v != 0.0));
    drop(snapshot);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn ragged_sparse_quartets_cross_rows_and_declared_transport_is_preserved() {
    let cpu = CpuBackend::new();
    let layout = AdditiveLayout {
        rows: 4,
        cols: 257,
        tile: 256,
        group: 32,
        max_planes: 3,
        allocation: PlaneAllocation::Uniform,
        codec: PlaneCodec::S34,
        law: ADMITTED_LAWS[1].law,
        basis: Basis::Identity,
        transport: Transport::Raw,
    };
    let compact_layout = AdditiveLayout {
        allocation: PlaneAllocation::PerTile,
        ..layout
    };
    let declared_transport = AdditiveLayout {
        codec: PlaneCodec::B3,
        transport: Transport::JointHuffman { block: 256 },
        ..layout
    };
    for source in [
        with_geometry(layout, 3, None),
        with_geometry(compact_layout, 3, Some(vec![0, 3, 1, 3, 2, 3, 3, 3])),
        with_geometry(declared_transport, 3, None),
    ] {
        let candidate = changed(&source);
        let parent: Arc<dyn DeviceBuffer> = Arc::from(
            cpu.upload_tensor(TensorView::Additive(source.view()))
                .unwrap(),
        );
        let budget = TensorPayloadBudget::new(parent.len_bytes() as u64);
        let snapshot = cpu
            .snapshot_additive_weights(&parent, candidate.view(), &budget)
            .unwrap();
        check(&*parent, source.view());
        check(&*snapshot, candidate.view());
        drop(snapshot);
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn admitted_parent_releases_independently_and_new_scale_overlay_retains_only_new_root() {
    use tritium_evidence::{EvidenceRecorder, TENSOR_UPLOADED_REGISTRATION};
    use tritium_runtime::{TensorUploadContext, upload_tensor_logged_admitted};
    let cpu = CpuBackend::new();
    let source = tensor(1, PlaneCodec::B3, Basis::Identity, 1, 128, 3);
    let candidate = changed(&source);
    let bytes = cpu
        .tensor_caps(TensorView::Additive(source.view()))
        .unwrap()
        .unwrap()
        .payload_bytes;
    let scale_bytes = (candidate.scales().len() * 4) as u64;
    let budget = TensorPayloadBudget::new(2 * bytes + scale_bytes);
    let mut recorder =
        EvidenceRecorder::new("full-snapshot-fixture", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
    let (root, _) = upload_tensor_logged_admitted(
        &cpu,
        TensorView::Additive(source.view()),
        tritium_schema::TensorUploadPolicy {
            native_only: true,
            max_payload_bytes: None,
        },
        "owner",
        TensorUploadContext {
            recorder: &mut recorder,
            span: "upload",
            parent: None,
            logical_time: 0,
        },
        &budget,
    )
    .unwrap();
    let root: Arc<dyn DeviceBuffer> = Arc::from(root);
    let original_log = recorder.to_jsonl().unwrap();
    let full = cpu
        .snapshot_additive_weights(&root, candidate.view(), &budget)
        .unwrap();
    assert_eq!(budget.used_bytes(), 2 * bytes);
    let old_root = Arc::downgrade(&root);
    drop(root);
    assert!(old_root.upgrade().is_none());
    assert_eq!(budget.used_bytes(), bytes);
    let (overlay, info) = cpu
        .snapshot_additive_scales(&full, candidate.scales(), &budget)
        .unwrap();
    assert_eq!(info.shared_root_bytes as u64, bytes);
    assert_eq!(budget.used_bytes(), bytes + scale_bytes);
    let new_root = Arc::downgrade(&full);
    drop(full);
    assert!(new_root.upgrade().is_some());
    check(&*overlay, candidate.view());
    assert_eq!(
        recorder.to_jsonl().unwrap(),
        original_log,
        "snapshots do not forge upload/refinement receipts"
    );
    drop(overlay);
    assert!(new_root.upgrade().is_none());
    assert_eq!(budget.used_bytes(), 0);
}
