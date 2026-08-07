//! S2KF-to-HESTIA sensitivity proxy contract (ADR 0035 / plan 0054 WS-C2).

use tritium_quantize::{
    CurvatureSourceId, DensePsdMetric, SaltV2Curvature, SaltV2KroneckerEvidence,
};

fn diagonal_metric(diagonal: f64) -> DensePsdMetric {
    let mut values = vec![0.0; 128 * 128];
    for index in 0..128 {
        values[index * 128 + index] = diagonal;
    }
    DensePsdMetric::new(128, &values).expect("positive diagonal PSD metric")
}

fn source_id() -> CurvatureSourceId {
    CurvatureSourceId::new([1; 32], [2; 32], [3; 32]).expect("nonzero source identities")
}

#[test]
fn guided_fisher_proxy_is_input_gram_trace_times_output_fisher_mean() {
    let evidence = SaltV2KroneckerEvidence::new(
        SaltV2Curvature::GuidedFisher,
        source_id(),
        [4; 32],
        0,
        "model.layers.0.self_attn.q_proj.weight",
        2,
        256,
        vec![diagonal_metric(2.0), diagonal_metric(4.0)],
        vec![2.0, 4.0],
        0.01,
    )
    .expect("valid guided-Fisher evidence");

    // Input trace = 128*2 + 128*4 = 768. Output mean = 3. Proxy = 2304.
    assert_eq!(evidence.hestia_trace_proxy().expect("valid proxy"), 2304.0);
}

#[test]
fn input_hessian_without_output_fisher_cannot_claim_hestia_sensitivity() {
    let evidence = SaltV2KroneckerEvidence::new(
        SaltV2Curvature::InputHessian,
        source_id(),
        [4; 32],
        0,
        "model.layers.0.self_attn.q_proj.weight",
        2,
        128,
        vec![diagonal_metric(1.0)],
        vec![1.0, 1.0],
        0.01,
    )
    .expect("valid input-Hessian evidence");

    assert!(evidence.hestia_trace_proxy().is_err());
}
