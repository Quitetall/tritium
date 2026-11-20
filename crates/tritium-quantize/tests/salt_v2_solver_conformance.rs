//! Public fitter conformance checks for optimized diagonal solver paths.

use tritium_quantize::{
    JointFitConfig, JointFitMetric, JointTernaryFit, RelayBasins, ScalePrecision, fit_joint_ternary,
};

fn fit(metric: JointFitMetric<'_>, weights: &[f32]) -> JointTernaryFit {
    fit_joint_ternary(
        weights,
        metric,
        JointFitConfig {
            planes: 3,
            max_iterations: 16,
            ridge: 1e-8,
            em_restarts: 4,
            ridge_condition_limit: 1e6,
            scale_precision: ScalePrecision::F16,
            relay_basins: RelayBasins {
                softened: true,
                modulated: true,
            },
        },
    )
    .expect("finite public joint fit")
}

#[test]
fn diagonal_f64_and_identity_affine_curvature_are_bit_identical() {
    let weights = (0..128)
        .map(|index| ((index * 7919 % 257) as f32 - 128.0) / 91.0)
        .collect::<Vec<_>>();
    let diagonal = (0..128)
        .map(|index| 0.125 + ((index * 3571 % 1021) as f64 / 733.0))
        .collect::<Vec<_>>();

    let direct = fit(JointFitMetric::DiagonalF64(&diagonal), &weights);
    let affine = fit(
        JointFitMetric::DiagonalAffine {
            values: &diagonal,
            scale: 1.0,
            shift: 0.0,
        },
        &weights,
    );

    assert_eq!(direct, affine);
}
