//! Exact CPU snapshot/version/lease fixtures; not model-quality or memory acceptance.
use std::sync::Arc;
use tritium_core::{
    AdditiveView, PerTilePlan, Trit, UniformPlan, accumulate_embed_scale_vjp,
    accumulate_ternary_matmul_scale_vjp, reference_embed, reference_ternary_matmul,
    reference_ternary_matmul_input_vjp,
};
use tritium_cpu::CpuBackend;
use tritium_format::AdditiveTensor;
use tritium_runtime::TensorPayloadBudget;
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, ScalePrecision, Transport,
};
use tritium_spec::{
    AdditiveMetadata, BackendError, DeviceBuffer, TensorMatmul, TensorMetadata,
    TensorStreamEncoding, TensorView, TernaryBackend,
};
#[path = "common/stream_source.rs"]
mod stream_source;

fn tensor(law: usize, codec: PlaneCodec, basis: Basis, mode: u8) -> AdditiveTensor {
    let admitted = ADMITTED_LAWS[law];
    let layout = AdditiveLayout {
        rows: 3,
        cols: if basis == Basis::Identity { 260 } else { 512 },
        tile: 256,
        group: 128,
        max_planes: admitted.max_planes,
        allocation: if mode == 0 {
            PlaneAllocation::Uniform
        } else {
            PlaneAllocation::PerTile
        },
        codec,
        law: admitted.law,
        basis,
        transport: Transport::Raw,
    };
    let counts = if mode == 2 {
        vec![0; 6]
    } else {
        [0, 1, admitted.max_planes, 2, 2, 0]
            .map(|n| n.min(admitted.max_planes))
            .to_vec()
    };
    let (n, s) = if mode == 0 {
        let p = UniformPlan::new(layout, admitted.max_planes).unwrap();
        (p.trit_count(), p.scale_count())
    } else {
        let p = PerTilePlan::new(layout, &counts).unwrap();
        (p.trit_count(), p.scale_count())
    };
    let trits = (0..n)
        .map(|i| [Trit::ZERO, Trit::POS, Trit::NEG, Trit::POS][i % 4])
        .collect();
    if mode == 0 {
        AdditiveTensor::new(layout, admitted.max_planes, trits, vec![0.5; s]).unwrap()
    } else {
        AdditiveTensor::new_per_tile(layout, counts, trits, vec![0.5; s]).unwrap()
    }
}
fn values(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i % 7) as f32 * 0.125 - 0.375).collect()
}
fn bits32(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}
fn bits64(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}
fn check(handle: &dyn DeviceBuffer, view: AdditiveView<'_>) {
    let cpu = CpuBackend::new();
    let rows = view.layout().rows as usize;
    let cols = view.layout().cols as usize;
    assert_eq!(
        cpu.additive_stored_scale_count(handle).unwrap(),
        view.scales().len()
    );
    for batch in [0, 2] {
        let x = values(batch * cols);
        let dy = values(batch * rows);
        let ids = if batch == 0 { vec![] } else { vec![2, 0, 2] };
        let de = values(ids.len() * cols);
        let mut actual = vec![19.0; dy.len()];
        let mut expected = actual.clone();
        let mut arow = vec![17.0; x.len()];
        let mut erow = arow.clone();
        cpu.matmul(TensorMatmul {
            act: &x,
            tensor: handle,
            batch,
            transformed_act: &mut arow,
            out: &mut actual,
        })
        .unwrap();
        reference_ternary_matmul(&x, &view.bind_matmul(), batch, &mut erow, &mut expected).unwrap();
        assert_eq!(bits32(&actual), bits32(&expected));
        assert_eq!(bits32(&arow), bits32(&erow));
        let mut actual = vec![19.0; de.len()];
        let mut expected = actual.clone();
        cpu.embed_rows(handle, &ids, &mut actual).unwrap();
        reference_embed(&view.bind_gather(), &ids, &mut expected).unwrap();
        assert_eq!(bits32(&actual), bits32(&expected));
        let mut actual = vec![19.0; x.len()];
        let mut expected = actual.clone();
        let mut arow = vec![17.0; cols];
        let mut erow = arow.clone();
        cpu.additive_matmul_input_vjp(handle, &dy, batch, &mut arow, &mut actual)
            .unwrap();
        reference_ternary_matmul_input_vjp(
            &dy,
            &view.bind_matmul(),
            batch,
            &mut erow,
            &mut expected,
        )
        .unwrap();
        assert_eq!(bits32(&actual), bits32(&expected));
        assert_eq!(bits32(&arow), bits32(&erow));
        let mut actual = vec![0.25; view.scales().len()];
        let mut expected = actual.clone();
        cpu.accumulate_additive_matmul_scale_vjp(handle, &x, &dy, batch, &mut arow, &mut actual)
            .unwrap();
        accumulate_ternary_matmul_scale_vjp(
            &x,
            &view.bind_matmul(),
            &dy,
            batch,
            &mut erow,
            &mut expected,
        )
        .unwrap();
        cpu.accumulate_additive_embed_scale_vjp(handle, &ids, &de, &mut arow, &mut actual)
            .unwrap();
        accumulate_embed_scale_vjp(&view.bind_gather(), &ids, &de, &mut erow, &mut expected)
            .unwrap();
        assert_eq!(bits64(&actual), bits64(&expected));
        assert_eq!(bits32(&arow), bits32(&erow));
    }
}

#[test]
fn direct_and_stream_snapshots_preserve_old_versions_and_match_exact_new_references() {
    let mut cases = 0;
    for law in 0..ADMITTED_LAWS.len() {
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
                    let source = tensor(law, codec, basis, mode);
                    let cpu = CpuBackend::new();
                    let encoded = source.encode_payload().unwrap();
                    let stream = stream_source::Source::new(
                        TensorMetadata::Additive(AdditiveMetadata::from_view(source.view())),
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
                        let mut a = source.clone();
                        let new = if source.layout().law.precision == ScalePrecision::F32 {
                            f32::from_bits(0x3f00_0001)
                        } else {
                            0.75
                        };
                        a.replace_scales(vec![new; source.scales().len()]).unwrap();
                        let mut b = source.clone();
                        b.replace_scales(vec![0.25; source.scales().len()]).unwrap();
                        let size = source.scales().len()
                            * if source.layout().law.precision == ScalePrecision::F16 {
                                2
                            } else {
                                4
                            };
                        let budget = TensorPayloadBudget::new((2 * size) as u64);
                        let (sa, info) = cpu
                            .snapshot_additive_scales(&root, a.scales(), &budget)
                            .unwrap();
                        assert_eq!(info.shared_root_bytes, root.len_bytes());
                        assert_eq!(info.owned_scale_bytes, size);
                        assert_eq!(info.reachable_bytes, root.len_bytes() + size);
                        assert_eq!(sa.len_bytes(), info.reachable_bytes);
                        let (sb, second) = cpu
                            .snapshot_additive_scales(&sa, b.scales(), &budget)
                            .unwrap();
                        assert_eq!(info, second);
                        assert_eq!(budget.used_bytes(), (2 * size) as u64);
                        check(&*root, source.view());
                        check(&*sa, a.view());
                        check(&*sb, b.view());
                        let weak_a = Arc::downgrade(&sa);
                        drop(sa);
                        assert!(weak_a.upgrade().is_none());
                        assert_eq!(budget.used_bytes(), size as u64);
                        let weak_root = Arc::downgrade(&root);
                        drop(root);
                        assert!(weak_root.upgrade().is_some());
                        check(&*sb, b.view());
                        let last = Arc::clone(&sb);
                        drop(sb);
                        assert_eq!(budget.used_bytes(), size as u64);
                        drop(last);
                        assert_eq!(budget.used_bytes(), 0);
                        assert!(weak_root.upgrade().is_none());
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 216);
}

#[test]
fn invalid_values_and_budget_failures_do_not_mutate_parent_or_retain_new_storage() {
    let source = tensor(0, PlaneCodec::D2, Basis::Identity, 0);
    let cpu = CpuBackend::new();
    let parent: Arc<dyn DeviceBuffer> = Arc::from(
        cpu.upload_tensor(TensorView::Additive(source.view()))
            .unwrap(),
    );
    let count = source.scales().len();
    let budget = TensorPayloadBudget::new((count * 4) as u64);
    for value in [f32::NAN, f32::INFINITY, -1.0, -0.0, 0.50001, 65536.0, 1e-30] {
        assert!(
            cpu.snapshot_additive_scales(&parent, &vec![value; count], &budget)
                .is_err()
        );
        assert_eq!(budget.used_bytes(), 0);
        assert_eq!(Arc::strong_count(&parent), 1);
    }
    assert!(cpu.snapshot_additive_scales(&parent, &[], &budget).is_err());
    let too_small = TensorPayloadBudget::new((count * 2 - 1) as u64);
    assert!(matches!(
        cpu.snapshot_additive_scales(&parent, &vec![0.75; count], &too_small),
        Err(BackendError::TensorPayloadBudgetExceeded { .. })
    ));
    assert_eq!(too_small.used_bytes(), 0);
    check(&*parent, source.view());
    assert_eq!(Arc::strong_count(&parent), 1);
}

#[test]
fn admitted_root_lease_is_retained_once_and_obsolete_overlays_release() {
    use tritium_evidence::{EvidenceRecorder, TENSOR_UPLOADED_REGISTRATION};
    use tritium_runtime::{TensorUploadContext, upload_tensor_logged_admitted};
    let source = tensor(1, PlaneCodec::B3, Basis::Identity, 1);
    let cpu = CpuBackend::new();
    let count = source.scales().len();
    let caps = cpu
        .tensor_caps(TensorView::Additive(source.view()))
        .unwrap()
        .unwrap();
    let budget = TensorPayloadBudget::new(caps.payload_bytes + (count * 8) as u64);
    let mut recorder =
        EvidenceRecorder::new("snapshot-fixture", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
    let (handle, _) = upload_tensor_logged_admitted(
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
    let root: Arc<dyn DeviceBuffer> = Arc::from(handle);
    let root_bytes = root.len_bytes();
    let before = recorder.to_jsonl().unwrap();
    let (a, _) = cpu
        .snapshot_additive_scales(&root, &vec![0.75; count], &budget)
        .unwrap();
    let (b, bytes) = cpu
        .snapshot_additive_scales(&a, &vec![0.25; count], &budget)
        .unwrap();
    assert_eq!(budget.used_bytes(), root_bytes as u64 + (count * 8) as u64);
    assert_eq!(bytes.shared_root_bytes, root_bytes);
    drop(root);
    drop(a);
    assert_eq!(budget.used_bytes(), root_bytes as u64 + (count * 4) as u64);
    assert_eq!(recorder.to_jsonl().unwrap(), before);
    let mut expected = source;
    expected.replace_scales(vec![0.25; count]).unwrap();
    check(&*b, expected.view());
    drop(b);
    assert_eq!(budget.used_bytes(), 0);
}

struct Foreign;
impl DeviceBuffer for Foreign {
    fn len_bytes(&self) -> usize {
        0
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
struct Misreported {
    inner: Arc<dyn DeviceBuffer>,
}
impl DeviceBuffer for Misreported {
    fn len_bytes(&self) -> usize {
        usize::MAX
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self.inner.as_any()
    }
}

#[test]
fn foreign_dense_legacy_and_misreported_handles_are_refused() {
    let cpu = CpuBackend::new();
    let budget = TensorPayloadBudget::new(1000);
    let dense: Arc<dyn DeviceBuffer> = Arc::from(
        cpu.upload_tensor(TensorView::Dense {
            rows: 1,
            cols: 4,
            values: &[1.0; 4],
        })
        .unwrap(),
    );
    let foreign: Arc<dyn DeviceBuffer> = Arc::new(Foreign);
    let mut raw = vec![0; tritium_format::TQ2_0_BLOCK_BYTES];
    tritium_format::pack_tq2_0_row(&[Trit::POS; 4], &[half::f16::ONE], &mut raw).unwrap();
    let legacy: Arc<dyn DeviceBuffer> = Arc::from(
        cpu.upload_weights(
            &raw,
            tritium_core::GemmShape { m: 1, n: 1, k: 4 },
            tritium_core::TernaryFormat::Tq2_0,
        )
        .unwrap(),
    );
    let source = tensor(1, PlaneCodec::D2, Basis::Identity, 0);
    let inner: Arc<dyn DeviceBuffer> = Arc::from(
        cpu.upload_tensor(TensorView::Additive(source.view()))
            .unwrap(),
    );
    let bad: Arc<dyn DeviceBuffer> = Arc::new(Misreported { inner });
    for (i, parent) in [dense, foreign, legacy, bad].into_iter().enumerate() {
        let result = cpu.snapshot_additive_scales(&parent, &[0.5], &budget);
        let error = match result {
            Err(e) => e,
            Ok(_) => panic!("invalid parent accepted"),
        };
        assert!(if i == 0 {
            matches!(error, BackendError::UnsupportedTensor)
        } else {
            matches!(error, BackendError::InvalidInput(_))
        });
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn concurrent_snapshot_creation_cannot_overcommit_incremental_budget() {
    let source = tensor(3, PlaneCodec::D2, Basis::Identity, 0);
    let root: Arc<dyn DeviceBuffer> = Arc::from(
        CpuBackend::new()
            .upload_tensor(TensorView::Additive(source.view()))
            .unwrap(),
    );
    let budget = TensorPayloadBudget::new(4);
    let start = std::sync::Barrier::new(9);
    let finish = std::sync::Barrier::new(9);
    std::thread::scope(|scope| {
        let jobs = (0..8)
            .map(|_| {
                let root = &root;
                let budget = &budget;
                let start = &start;
                let finish = &finish;
                scope.spawn(move || {
                    start.wait();
                    let snapshot =
                        CpuBackend::new().snapshot_additive_scales(root, &[0.75], budget);
                    finish.wait();
                    snapshot.is_ok()
                })
            })
            .collect::<Vec<_>>();
        start.wait();
        finish.wait();
        assert_eq!(
            jobs.into_iter()
                .map(|job| job.join().unwrap())
                .filter(|ok| *ok)
                .count(),
            1
        );
    });
    assert_eq!(budget.used_bytes(), 0);
    assert_eq!(Arc::strong_count(&root), 1);
}
