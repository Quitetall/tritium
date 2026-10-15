//! Actual cache replay and common fitting, with explicit analytical row oracles.
use std::sync::Arc;
use tritium_quantize::engine::*;
use tritium_schema::*;

fn cache(cols: usize, input: f32, gather: f32, adjoint: f32, token: i64) -> OwnerScoreCache {
    let inputs = vec![input; cols];
    let adjoints = [adjoint, 2.0 * adjoint];
    let gathered = vec![gather; cols];
    let indices = [0_i64];
    let uses = [
        OwnerScoreUse::Matmul {
            samples: 1,
            inputs: &inputs,
            adjoints: &adjoints,
        },
        OwnerScoreUse::Gather {
            indices: &indices,
            adjoints: &gathered,
        },
    ];
    let tokens = token.to_le_bytes();
    OwnerScoreCache::new(
        OwnerScoreCacheInput {
            source: OriginalSourceId::from_bytes([7; 32]),
            tensor: QuantSourceTensor {
                name: "weight".into(),
                shape: vec![2, cols as u64],
                precision: ScalePrecision::F32,
            },
            tokenizer: b"actual tokenizer",
            objective: b"scalar window mean",
            meta: OwnerScoreWindowMeta {
                context: vec![OwnerContextMeta {
                    name: "input_ids".into(),
                    dtype: OwnerContextDtype::I64,
                    shape: vec![1],
                    conjugated: false,
                    negated: false,
                }],
                invocations: vec![
                    OwnerInvocationMeta {
                        ordinal: 0,
                        paths: vec!["head".into()],
                        role: OwnerScoreRole::Matmul,
                        samples: 1,
                    },
                    OwnerInvocationMeta {
                        ordinal: 1,
                        paths: vec!["embed".into()],
                        role: OwnerScoreRole::Gather,
                        samples: 1,
                    },
                ],
            },
            context: &[&tokens],
            observations: Arc::new(
                OwnerScoreWindow::new(2, cols, &uses, 100_000)
                    .unwrap()
                    .snapshot()
                    .unwrap(),
            ),
        },
        OwnerScoreCacheLimits {
            max_payload_bytes: 100_000,
            max_context_bytes: 1000,
            max_context_tensors: 4,
            max_invocations: 8,
            max_samples: 16,
        },
    )
    .unwrap()
}
fn limits() -> OwnerScoreMetricLimits {
    OwnerScoreMetricLimits {
        max_windows: 32,
        max_payload_bytes: 1_000_000,
        max_row_bytes: 10_000,
        max_reduction_entries: 1_000_000,
    }
}
fn windows(caches: &[&OwnerScoreCache], weights: &[f64]) -> Vec<WeightedOwnerWindow> {
    caches
        .iter()
        .zip(weights)
        .map(|(cache, &weight)| WeightedOwnerWindow {
            cache: cache.id(),
            weight,
        })
        .collect()
}

#[test]
fn sums_uses_before_outer_product_and_normalizes_windows_not_tokens() {
    let reinforce = cache(2, 1., 1., 1., 0);
    let cancel = cache(2, 1., -1., 1., 1);
    let refs = [&reinforce, &cancel];
    let metrics =
        OwnerScoreMetrics::new(&refs, &windows(&refs, &[1., 3.]), 0.25, limits()).unwrap();
    // First row window gradients: [2,2] and [0,0], not separate use Grams.
    assert_eq!(
        metrics.row_metric(0).unwrap().as_slice(),
        &[1.25, 1., 1., 1.25]
    );
    // Second row has no gather contribution, both window gradients [2,2].
    assert_eq!(
        metrics.row_metric(1).unwrap().as_slice(),
        &[4.25, 4., 4., 4.25]
    );
    assert!(metrics.row_metric(2).is_err());
}

#[test]
fn preserves_f64_reconstructed_products_beyond_f32_range() {
    let a = cache(1, 1e20, 0., 1e20, 0);
    let refs = [&a];
    let m = OwnerScoreMetrics::new(&refs, &windows(&refs, &[1.]), 0., limits()).unwrap();
    let score = f64::from(1e20_f32) * f64::from(1e20_f32);
    assert!(score > f64::from(f32::MAX));
    assert_eq!(m.row_metric(0).unwrap().as_slice(), &[score * score]);
}

#[test]
fn actual_package_replay_has_identical_metric_and_identity() {
    let a = cache(3, 0.3, 0.7, 0.2, 0);
    let b = cache(3, -0.2, 0.1, 0.4, 1);
    let bytes = b.encode_package(65536).unwrap();
    let reopened = OwnerScoreCache::decode_package(&bytes, b.binding(), b.limits(), 65536).unwrap();
    let refs = [&a, &b];
    let replay = [&a, &reopened];
    let expected = windows(&refs, &[0.5, 2.]);
    let before = OwnerScoreMetrics::new(&refs, &expected, 0.1, limits()).unwrap();
    let after = OwnerScoreMetrics::new(&replay, &expected, 0.1, limits()).unwrap();
    for row in 0..2 {
        assert_eq!(
            before.row_metric(row).unwrap().as_slice(),
            after.row_metric(row).unwrap().as_slice()
        );
        assert_eq!(
            before.row_metric(row).unwrap().digest(),
            after.row_metric(row).unwrap().digest()
        );
    }
}

#[test]
fn identity_binds_order_weights_damping_and_row_but_not_limits_or_signed_zero() {
    let a = cache(2, 1., 0., 1., 0);
    let b = cache(2, 1., 0., 1., 1);
    let refs = [&a, &b];
    let swapped = [&b, &a];
    let expected = windows(&refs, &[1., 0.]);
    let baseline = OwnerScoreMetrics::new(&refs, &expected, 0., limits())
        .unwrap()
        .row_metric(0)
        .unwrap();
    for changed in [
        OwnerScoreMetrics::new(&swapped, &windows(&swapped, &[1., 0.]), 0., limits()).unwrap(),
        OwnerScoreMetrics::new(&refs, &windows(&refs, &[2., 0.]), 0., limits()).unwrap(),
        OwnerScoreMetrics::new(&refs, &expected, 0.01, limits()).unwrap(),
    ] {
        assert_ne!(baseline.digest(), changed.row_metric(0).unwrap().digest());
    }
    let mut larger = limits();
    larger.max_windows *= 2;
    let neg_zero = OwnerScoreMetrics::new(&refs, &windows(&refs, &[1., -0.]), -0., larger).unwrap();
    assert_eq!(baseline.digest(), neg_zero.row_metric(0).unwrap().digest());
    assert_ne!(baseline.digest(), neg_zero.row_metric(1).unwrap().digest());
    let repeats = [&a, &a];
    let repeated =
        OwnerScoreMetrics::new(&repeats, &windows(&repeats, &[1., 1.]), 0., limits()).unwrap();
    assert_eq!(
        baseline.as_slice(),
        repeated.row_metric(0).unwrap().as_slice()
    );
    assert_ne!(baseline.digest(), repeated.row_metric(0).unwrap().digest());
}

#[test]
fn expected_binding_configuration_and_each_limit_fail_closed() {
    let a = cache(2, 1., 0., 1., 0);
    let b = cache(2, 2., 0., 1., 0);
    let refs = [&a];
    assert!(OwnerScoreMetrics::new(&refs, &windows(&[&b], &[1.]), 0., limits()).is_err());
    assert!(OwnerScoreMetrics::new(&[], &[], 0., limits()).is_err());
    for weight in [f64::NAN, f64::INFINITY, -1., 0.] {
        assert!(OwnerScoreMetrics::new(&refs, &windows(&refs, &[weight]), 0., limits()).is_err());
    }
    for damping in [f64::NAN, f64::INFINITY, -1.] {
        assert!(OwnerScoreMetrics::new(&refs, &windows(&refs, &[1.]), damping, limits()).is_err());
    }
    for l in [
        OwnerScoreMetricLimits {
            max_windows: 0,
            ..limits()
        },
        OwnerScoreMetricLimits {
            max_payload_bytes: 1,
            ..limits()
        },
        OwnerScoreMetricLimits {
            max_row_bytes: 15,
            ..limits()
        },
        OwnerScoreMetricLimits {
            max_reduction_entries: 1,
            ..limits()
        },
    ] {
        assert!(matches!(
            OwnerScoreMetrics::new(&refs, &windows(&refs, &[1.]), 0., l),
            Err(OwnerScoreMetricError::Limit)
        ));
    }
    let repeated = [&a, &a];
    assert!(
        OwnerScoreMetrics::new(
            &repeated,
            &windows(&repeated, &[f64::MAX, f64::MAX]),
            0.,
            limits()
        )
        .is_err()
    );
}

#[test]
fn fitting_uses_actual_row_metrics_through_the_same_direct_controller() {
    for (cols, basis) in [
        (35, Basis::Identity),
        (
            64,
            Basis::SignedRht {
                seed: 11,
                block: 32,
                domain: 9,
            },
        ),
    ] {
        let a = cache(cols, 0.3, 0.7, 0.2, 0);
        let b = cache(cols, -0.2, 0.1, 0.4, 1);
        let refs = [&a, &b];
        let metrics =
            OwnerScoreMetrics::new(&refs, &windows(&refs, &[1., 2.]), 0.1, limits()).unwrap();
        let weights: Vec<f32> = (0..2 * cols)
            .map(|i| ((i * 17 % 43) as f32 - 21.) / 13.)
            .collect();
        let plan = DirectFitPlan {
            layout: AdditiveLayout {
                rows: 2,
                cols: cols as u64,
                tile: ADDITIVE_TILE_SIZE,
                group: 32,
                max_planes: 3,
                allocation: PlaneAllocation::Uniform,
                codec: PlaneCodec::B3,
                law: ScaleLaw {
                    anchor: ScaleAnchor::Group,
                    precision: ScalePrecision::F32,
                    relation: PlaneRelation::Free,
                },
                basis,
                transport: Transport::Raw,
            },
            profile: DirectFitProfile::JointIndependentGroups(JointGroupFitOptions {
                max_iterations: 4,
                ridge: 1e-8,
                em_restarts: 1,
                ridge_condition_limit: 1e6,
                softened_relay: false,
                modulated_relay: false,
            }),
            levels: vec![
                DirectFitLevel {
                    name: "three".into(),
                    planes: 3,
                },
                DirectFitLevel {
                    name: "one".into(),
                    planes: 1,
                },
            ],
            limits: DirectFitLimits {
                max_weights: 2 * cols as u64,
                max_metric_entries: (cols * cols) as u64,
                max_output_trits: 8 * cols as u64,
            },
        };
        let actual = fit_owner_score_levels(
            &plan,
            metrics.tensor(),
            metrics.source(),
            &weights,
            &metrics,
        )
        .unwrap();
        let mut row_plan = plan.clone();
        row_plan.layout.rows = 1;
        for row in 0..2 {
            let metric = metrics.row_metric(row).unwrap();
            let expected = fit_direct_levels(
                &row_plan,
                &weights[row * cols..(row + 1) * cols],
                Some(metric.as_slice()),
            )
            .unwrap();
            for (full, one) in actual.iter().zip(expected) {
                for (plane, digits) in one.tensor.trits().chunks_exact(cols).enumerate() {
                    assert_eq!(
                        &full.tensor.trits()
                            [plane * 2 * cols + row * cols..plane * 2 * cols + (row + 1) * cols],
                        digits
                    );
                }
                for plane in 0..one.tensor.trits().len() / cols {
                    assert_eq!(
                        &full.tensor.scales()[plane * 4 + row * 2..plane * 4 + (row + 1) * 2],
                        &one.tensor.scales()[plane * 2..(plane + 1) * 2]
                    );
                }
            }
        }
        let mut wrong = metrics.tensor().clone();
        wrong.name.push('x');
        assert!(
            fit_owner_score_levels(&plan, &wrong, metrics.source(), &weights, &metrics).is_err()
        );
        assert!(
            fit_owner_score_levels(
                &plan,
                metrics.tensor(),
                OriginalSourceId::from_bytes([8; 32]),
                &weights,
                &metrics
            )
            .is_err()
        );
        let mut too_small = plan.clone();
        too_small.limits.max_weights -= 1;
        assert!(matches!(
            fit_owner_score_levels(
                &too_small,
                metrics.tensor(),
                metrics.source(),
                &weights,
                &metrics
            ),
            Err(OwnerScoreMetricError::Fit(_))
        ));
    }
}
