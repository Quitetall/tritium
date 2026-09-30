//! Frozen additive-layout reference vectors for ADR 0044 P1.
//!
//! These vectors intentionally live outside `tritium-core` so the consumer
//! crate checks the public schema/core boundary rather than private helpers.

use tritium_core::{AdditiveView, Trit, apply_basis, reference_ternary_matmul};
use tritium_schema::{
    AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, PlaneRelation, ScaleAnchor, ScaleLaw,
    ScalePrecision, Transport,
};

fn layout(basis: Basis, law: ScaleLaw) -> AdditiveLayout {
    AdditiveLayout {
        rows: 1,
        cols: 4,
        tile: 256,
        group: 32,
        max_planes: 3,
        allocation: PlaneAllocation::Uniform,
        codec: PlaneCodec::D2,
        law,
        basis,
        transport: Transport::Raw,
    }
}

fn law(anchor: ScaleAnchor, relation: PlaneRelation) -> ScaleLaw {
    ScaleLaw {
        anchor,
        relation,
        precision: ScalePrecision::F32,
    }
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1.0e-6,
            "value {index}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn frozen_free_group_two_plane_reference_vector() {
    let trits = [
        Trit::POS,
        Trit::NEG,
        Trit::ZERO,
        Trit::POS,
        Trit::ZERO,
        Trit::POS,
        Trit::NEG,
        Trit::POS,
    ];
    let view = AdditiveView::new(
        layout(
            Basis::Identity,
            law(ScaleAnchor::Group, PlaneRelation::Free),
        ),
        2,
        &trits,
        &[0.5, 0.25],
    )
    .expect("frozen vector layout is valid");
    let mut decoded = [0.0; 4];
    view.dequant_row_into(0, &mut decoded)
        .expect("frozen vector row is valid");
    assert_close(&decoded, &[0.5, -0.25, -0.25, 0.75]);

    let mut transformed = [0.0; 4];
    let mut output = [0.0; 1];
    reference_ternary_matmul(
        &[1.0, 2.0, 3.0, 4.0],
        &view,
        1,
        &mut transformed,
        &mut output,
    )
    .expect("frozen vector matmul is valid");
    assert_eq!(transformed, [1.0, 2.0, 3.0, 4.0]);
    assert_close(&output, &[2.25]);
}

#[test]
fn frozen_tied_group_three_plane_reference_vector() {
    let trits = [
        Trit::POS,
        Trit::ZERO,
        Trit::POS,
        Trit::ZERO,
        Trit::ZERO,
        Trit::POS,
        Trit::ZERO,
        Trit::POS,
        Trit::POS,
        Trit::POS,
        Trit::ZERO,
        Trit::ZERO,
    ];
    let view = AdditiveView::new(
        layout(
            Basis::Identity,
            ScaleLaw {
                anchor: ScaleAnchor::Group,
                relation: PlaneRelation::Tied { num: 1, den: 3 },
                precision: ScalePrecision::F16,
            },
        ),
        3,
        &trits,
        &[3.0],
    )
    .expect("frozen vector layout is valid");
    let mut transformed = [0.0; 4];
    let mut output = [0.0; 1];
    reference_ternary_matmul(
        &[1.0, 2.0, 3.0, 4.0],
        &view,
        1,
        &mut transformed,
        &mut output,
    )
    .expect("frozen vector matmul is valid");
    assert_close(&output, &[19.0]);
}

#[test]
fn frozen_hadamard_basis_reference_vector() {
    let trits = [Trit::POS, Trit::NEG, Trit::ZERO, Trit::POS];
    let view = AdditiveView::new(
        layout(
            Basis::Hadamard { block: 4 },
            law(ScaleAnchor::Group, PlaneRelation::Free),
        ),
        1,
        &trits,
        &[1.0],
    )
    .expect("frozen vector layout is valid");
    let mut transformed = [0.0; 4];
    let mut output = [0.0; 1];
    reference_ternary_matmul(
        &[1.0, 2.0, 3.0, 4.0],
        &view,
        1,
        &mut transformed,
        &mut output,
    )
    .expect("frozen vector matmul is valid");
    assert_eq!(transformed, [5.0, -1.0, -2.0, 0.0]);
    assert_close(&output, &[6.0]);
}

#[test]
fn frozen_signed_rht_basis_vector() {
    let mut activation = [1.0, 2.0, 3.0, 4.0];
    apply_basis(
        &mut activation,
        Basis::SignedRht {
            block: 4,
            seed: 7,
            domain: 9,
        },
    )
    .expect("frozen signed RHT basis is valid");
    assert_eq!(activation, [-4.0, 2.0, 3.0, 1.0]);
}
