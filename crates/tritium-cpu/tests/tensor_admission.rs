//! Checked production upload and frozen scalar execution combinations.

use tritium_core::{AdditiveView, Trit};
use tritium_cpu::CpuBackend;
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, TensorExecution,
    TensorUploadPolicy, Transport,
};
use tritium_spec::{BackendError, TensorView, TernaryBackend};
use tritium_testkit::ReferenceBackend;

#[test]
fn production_capabilities_execute_admitted_combinations() {
    let cpu = CpuBackend::new();
    let reference = ReferenceBackend::new();
    for backend in [&cpu as &dyn TernaryBackend, &reference] {
        tritium_testkit::assert_additive_upload_rejections(backend);
        assert_eq!(
            tritium_testkit::assert_additive_tensor_conformance(backend, TensorExecution::Native),
            180
        );
    }
}

#[test]
fn cpu_checked_upload_enforces_budget_and_keeps_unadmitted_groups_unknown() {
    let cpu = CpuBackend::new();
    let empty = TensorView::Dense {
        rows: 0,
        cols: 0,
        values: &[],
    };
    let zero_budget = TensorUploadPolicy {
        native_only: true,
        max_payload_bytes: Some(0),
    };
    assert_eq!(
        cpu.upload_tensor_checked(empty, zero_budget)
            .unwrap()
            .1
            .payload_bytes,
        0
    );
    let dense = TensorView::Dense {
        rows: 1,
        cols: 2,
        values: &[1., 2.],
    };
    let too_small = TensorUploadPolicy {
        native_only: true,
        max_payload_bytes: Some(7),
    };
    assert!(matches!(
        cpu.upload_tensor_checked(dense, too_small),
        Err(BackendError::TensorPayloadBudgetExceeded {
            requested: 8,
            limit: 7
        })
    ));
    let exact = TensorUploadPolicy {
        max_payload_bytes: Some(8),
        ..too_small
    };
    assert_eq!(
        cpu.upload_tensor_checked(dense, exact)
            .unwrap()
            .1
            .payload_bytes,
        8
    );
    for group in [64, 256] {
        let layout = AdditiveLayout {
            rows: 1,
            cols: 4,
            tile: 256,
            group,
            max_planes: 1,
            allocation: PlaneAllocation::Uniform,
            codec: PlaneCodec::D2,
            law: ADMITTED_LAWS[1].law,
            basis: Basis::Identity,
            transport: Transport::Raw,
        };
        let view =
            TensorView::Additive(AdditiveView::new(layout, 1, &[Trit::POS; 4], &[1.]).unwrap());
        assert_eq!(cpu.tensor_caps(view).unwrap(), None);
        assert!(matches!(
            cpu.upload_tensor_checked(view, exact),
            Err(BackendError::UnsupportedTensor)
        ));
    }
}

#[test]
fn capability_query_rejects_invalid_dense_and_inexact_scale_inputs() {
    let cpu = CpuBackend::new();
    assert!(
        cpu.tensor_caps(TensorView::Dense {
            rows: usize::MAX,
            cols: 2,
            values: &[]
        })
        .is_err()
    );
    assert!(
        cpu.tensor_caps(TensorView::Dense {
            rows: 1,
            cols: 2,
            values: &[1.]
        })
        .is_err()
    );
    let layout = AdditiveLayout {
        rows: 1,
        cols: 4,
        tile: 256,
        group: 32,
        max_planes: 1,
        allocation: PlaneAllocation::Uniform,
        codec: PlaneCodec::D2,
        law: ADMITTED_LAWS[0].law,
        basis: Basis::Identity,
        transport: Transport::Raw,
    };
    for scales in [[0.1], [-0.]] {
        let view =
            TensorView::Additive(AdditiveView::new(layout, 1, &[Trit::POS; 4], &scales).unwrap());
        assert!(cpu.tensor_caps(view).is_err());
        assert!(
            cpu.upload_tensor_checked(view, TensorUploadPolicy::default())
                .is_err()
        );
    }
}
