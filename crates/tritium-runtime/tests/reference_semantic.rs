//! Production reference semantics/admission, not physical performance claims.

use core::any::Any;
use std::sync::Arc;
use tritium_core::{PerTileRowSpan, Trit};
use tritium_evidence::{EvidenceRecorder, TENSOR_UPLOADED_REGISTRATION};
use tritium_format::AdditiveTensor;
use tritium_runtime::{
    ReferenceBackend, Registry, TensorPayloadBudget, TensorUploadContext, TensorUploadError,
    upload_tensor_logged_admitted,
};
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, Transport,
};
use tritium_spec::{
    BackendError, DeviceBuffer, GemmShape, MpGemm, TensorExecution, TensorMatmul,
    TensorUploadPolicy, TensorView, TernaryBackend, TernaryFormat,
};

fn compact(pruned: bool) -> AdditiveTensor {
    let law = ADMITTED_LAWS[0];
    let layout = AdditiveLayout {
        rows: 2,
        cols: 257,
        tile: 256,
        group: 128,
        max_planes: law.max_planes,
        allocation: PlaneAllocation::PerTile,
        codec: PlaneCodec::D2,
        law: law.law,
        basis: Basis::Identity,
        transport: Transport::Raw,
    };
    let counts = if pruned { vec![0; 4] } else { vec![1, 3, 2, 0] };
    let plan = tritium_core::PerTilePlan::new(layout, &counts).unwrap();
    let trits = vec![Trit::POS; plan.trit_count()];
    let scales = vec![0.5; plan.scale_count()];
    AdditiveTensor::new_per_tile(layout, counts, trits, scales).unwrap()
}

#[test]
fn registered_reference_owns_compact_payload_and_exact_scale_semantics() {
    let registry = Registry::init();
    let backend = registry.get("reference").unwrap();
    let owned = compact(false);
    let view = TensorView::Additive(owned.view());
    assert_eq!(view.decoded_payload_bytes().unwrap(), 807);
    let bytes = 807 + 4 + 6 * core::mem::size_of::<PerTileRowSpan>();
    let caps = backend.tensor_caps(view).unwrap().unwrap();
    assert_eq!(caps.execution, TensorExecution::Native);
    assert_eq!(caps.payload_bytes, bytes as u64);
    assert!(matches!(
        backend.upload_tensor_checked(
            view,
            TensorUploadPolicy {
                native_only: true,
                max_payload_bytes: Some(bytes as u64 - 1)
            }
        ),
        Err(BackendError::TensorPayloadBudgetExceeded { .. })
    ));
    let (tensor, actual) = backend
        .upload_tensor_checked(
            view,
            TensorUploadPolicy {
                native_only: true,
                max_payload_bytes: Some(bytes as u64),
            },
        )
        .unwrap();
    assert_eq!(actual, caps);
    assert_eq!(tensor.len_bytes(), bytes);
    drop(owned);
    let act: Vec<_> = [vec![1.; 257], vec![2.; 257]].concat();
    let mut scratch = [99.; 514];
    let mut out = [99.; 4];
    backend
        .matmul(TensorMatmul {
            act: &act,
            tensor: &*tensor,
            batch: 2,
            transformed_act: &mut scratch,
            out: &mut out,
        })
        .unwrap();
    assert_eq!(scratch.as_slice(), act);
    assert_eq!(out, [129.5, 256., 259., 512.]);
    let mut rows = [99.; 514];
    backend.embed_rows(&*tensor, &[0, 1], &mut rows).unwrap();
    assert_eq!(&rows[..256], &[0.5; 256]);
    assert_eq!(rows[256], 1.5);
    assert_eq!(&rows[257..513], &[1.; 256]);
    assert_eq!(rows[513], 0.);
}

#[test]
fn all_pruned_reference_retains_counts_and_checks_shapes_before_mutation() {
    let backend = ReferenceBackend::new();
    let owned = compact(true);
    let view = TensorView::Additive(owned.view());
    assert_eq!(view.decoded_payload_bytes().unwrap(), 0);
    let (tensor, caps) = backend
        .upload_tensor_checked(
            view,
            TensorUploadPolicy {
                native_only: true,
                max_payload_bytes: Some(4),
            },
        )
        .unwrap();
    assert_eq!(caps.payload_bytes, 4);
    assert_eq!(tensor.len_bytes(), 4);
    drop(owned);
    let mut scratch = [99.; 257];
    let mut out = [99.; 2];
    assert!(
        backend
            .matmul(TensorMatmul {
                act: &[1.; 256],
                tensor: &*tensor,
                batch: 1,
                transformed_act: &mut scratch,
                out: &mut out
            })
            .is_err()
    );
    assert_eq!(scratch, [99.; 257]);
    assert_eq!(out, [99.; 2]);
    backend
        .matmul(TensorMatmul {
            act: &[1.; 257],
            tensor: &*tensor,
            batch: 1,
            transformed_act: &mut scratch,
            out: &mut out,
        })
        .unwrap();
    assert_eq!(scratch, [1.; 257]);
    assert_eq!(out, [0.; 2]);
    let mut rows = [99.; 514];
    assert!(backend.embed_rows(&*tensor, &[0, 2], &mut rows).is_err());
    assert_eq!(rows, [99.; 514]);
    backend.embed_rows(&*tensor, &[1, 0], &mut rows).unwrap();
    assert_eq!(rows, [0.; 514]);
}

#[test]
fn production_logged_reference_holds_budget_and_releases_on_emission_failure() {
    let backend = ReferenceBackend::new();
    let owned = compact(false);
    let view = TensorView::Additive(owned.view());
    let bytes = backend.tensor_caps(view).unwrap().unwrap().payload_bytes;
    let budget = TensorPayloadBudget::new(bytes);
    let clone = budget.clone();
    let mut recorder =
        EvidenceRecorder::new("reference-fixture", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
    let upload = |recorder: &mut EvidenceRecorder| {
        upload_tensor_logged_admitted(
            &backend,
            view,
            TensorUploadPolicy {
                native_only: true,
                max_payload_bytes: Some(bytes),
            },
            "weight",
            TensorUploadContext {
                recorder,
                span: "load",
                parent: None,
                logical_time: 0,
            },
            &budget,
        )
    };
    let (tensor, caps) = upload(&mut recorder).unwrap();
    assert_eq!(budget.used_bytes(), bytes);
    assert_eq!(tensor.len_bytes() as u64, caps.payload_bytes);
    assert_eq!(recorder.events().len(), 1);
    let payload = &recorder.events()[0].payload;
    assert_eq!(payload["backend"], "reference");
    assert_eq!(payload["caps"]["payload_bytes"], bytes);
    assert_eq!(payload["decoded_source_bytes"], 807);
    let root = recorder.root_digest().unwrap();
    assert!(matches!(
        upload(&mut recorder),
        Err(TensorUploadError::Backend(
            BackendError::TensorPayloadBudgetExceeded { .. }
        ))
    ));
    assert_eq!(recorder.root_digest().unwrap(), root);
    assert_eq!(clone.used_bytes(), bytes);
    let shared: Arc<dyn DeviceBuffer> = Arc::from(tensor);
    let tied = Arc::clone(&shared);
    drop(shared);
    assert_eq!(clone.used_bytes(), bytes);
    drop(tied);
    assert_eq!(clone.used_bytes(), 0);
    let mut unregistered = EvidenceRecorder::new("reference-fixture", []).unwrap();
    assert!(matches!(
        upload(&mut unregistered),
        Err(TensorUploadError::Evidence(_))
    ));
    assert_eq!(budget.used_bytes(), 0);
    assert!(unregistered.events().is_empty());
    drop(upload(&mut recorder).unwrap());
    assert_eq!(budget.used_bytes(), 0);
}

struct ForeignBuffer;
impl DeviceBuffer for ForeignBuffer {
    fn len_bytes(&self) -> usize {
        0
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[test]
fn dense_and_foreign_reference_errors_do_not_mutate_outputs() {
    let backend = ReferenceBackend::new();
    assert!(
        backend
            .upload_tensor(TensorView::Dense {
                rows: 2,
                cols: 3,
                values: &[1.; 5]
            })
            .is_err()
    );
    let tensor = backend
        .upload_tensor(TensorView::Dense {
            rows: 2,
            cols: 3,
            values: &[1., 2., 3., -1., 0.5, 2.],
        })
        .unwrap();
    assert_eq!(tensor.len_bytes(), 24);
    let mut scratch = [99.; 3];
    let mut out = [99.; 2];
    for buffer in [&*tensor, &ForeignBuffer as &dyn DeviceBuffer] {
        assert!(
            backend
                .matmul(TensorMatmul {
                    act: &[1.; 2],
                    tensor: buffer,
                    batch: 1,
                    transformed_act: &mut scratch,
                    out: &mut out
                })
                .is_err()
        );
        assert_eq!(scratch, [99.; 3]);
        assert_eq!(out, [99.; 2]);
        let mut rows = [99.; 6];
        assert!(backend.embed_rows(buffer, &[0, 2], &mut rows).is_err());
        assert_eq!(rows, [99.; 6]);
    }
    backend
        .matmul(TensorMatmul {
            act: &[2., -1., 4.],
            tensor: &*tensor,
            batch: 1,
            transformed_act: &mut scratch,
            out: &mut out,
        })
        .unwrap();
    assert_eq!(scratch, [2., -1., 4.]);
    assert_eq!(out, [12., 5.5]);
    let mut rows = [0.; 6];
    backend.embed_rows(&*tensor, &[1, 0], &mut rows).unwrap();
    assert_eq!(rows, [-1., 0.5, 2., 1., 2., 3.]);
}

#[test]
fn production_reference_never_discards_legacy_block_scales() {
    let backend = ReferenceBackend::new();
    let shape = GemmShape { m: 1, n: 1, k: 4 };
    for format in [TernaryFormat::Tq1_0, TernaryFormat::Tq2_0] {
        assert!(matches!(backend.upload_weights(&[], shape, format),
            Err(BackendError::UnsupportedFormat(rejected)) if rejected == format));
        let mut out = [99.];
        assert!(
            backend
                .mpgemm(MpGemm {
                    act: &[1.; 4],
                    weights: &ForeignBuffer,
                    scales: &[1.],
                    shape,
                    format,
                    out: &mut out
                })
                .is_err()
        );
        assert_eq!(out, [99.]);
    }
}
