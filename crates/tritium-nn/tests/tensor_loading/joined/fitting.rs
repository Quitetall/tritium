//! Shared actual calibrated fitter requests for both legacy and original-captured tests.
use super::*;
use tritium_quantize::engine::CalibrationCache;
use tritium_schema::{
    CalibratedFitPlan, CalibratedQuantAction, CalibratedQuantPlan, CalibratedQuantTensorPlan,
    CalibrationPartition,
};

pub(super) fn calibrated_plan(
    stage: &StagedSalienceSource<'_>,
    caches: &[CalibrationCache],
) -> CalibratedQuantPlan {
    CalibratedQuantPlan {
        architecture: stage.manifest().observation.plan.architecture.clone(),
        original_source: stage.source_id(),
        tensors: stage
            .inventory()
            .unwrap()
            .into_iter()
            .map(|source| {
                let action = if let Some(cache) =
                    caches.iter().find(|c| c.binding().request.tensor == source)
                {
                    let [rows, cols] = source.shape.as_slice() else {
                        panic!("matrix cache required")
                    };
                    CalibratedQuantAction::CalibratedFit {
                        plan: Box::new(CalibratedFitPlan {
                            direct: DirectFitPlan {
                                layout: AdditiveLayout {
                                    rows: *rows,
                                    cols: *cols,
                                    tile: 256,
                                    group: 32,
                                    max_planes: 3,
                                    allocation: PlaneAllocation::Uniform,
                                    codec: PlaneCodec::B3,
                                    law: ADMITTED_LAWS[1].law,
                                    basis: Basis::Identity,
                                    transport: Transport::Raw,
                                },
                                profile: DirectFitProfile::JointIndependentGroups(
                                    JointGroupFitOptions {
                                        max_iterations: 4,
                                        ridge: 1e-8,
                                        em_restarts: 1,
                                        ridge_condition_limit: 1e6,
                                        softened_relay: false,
                                        modulated_relay: false,
                                    },
                                ),
                                levels: vec![DirectFitLevel {
                                    name: "standard".into(),
                                    planes: 3,
                                }],
                                limits: DirectFitLimits {
                                    max_weights: rows * cols,
                                    max_metric_entries: cols * cols,
                                    max_output_trits: 3 * rows * cols,
                                },
                            },
                            cache: cache.binding().clone(),
                            partition: CalibrationPartition::Full,
                            damping: 0.125,
                            max_numeric_bytes: 8_000_000,
                        }),
                    }
                } else {
                    CalibratedQuantAction::KeepDense
                };
                CalibratedQuantTensorPlan { source, action }
            })
            .collect(),
        assets: stage.asset_names(),
        levels: vec!["standard".into()],
        limits: QuantLimits {
            max_source_bytes: 64_000,
            max_asset_bytes: 16_000,
            max_retained_blob_bytes: 64_000,
            max_package_bytes: 1_000_000,
        },
        max_cache_payload_bytes: caches.iter().map(|c| c.binding().payload_bytes).sum(),
        max_cache_package_bytes: 1_000_000,
    }
}

pub(super) fn calibrated_recorder() -> EvidenceRecorder {
    use tritium_evidence::{
        CALIBRATED_LEVELS_FITTED_REGISTRATION, CALIBRATED_QUANTIZATION_COMPLETED_REGISTRATION,
        CALIBRATED_QUANTIZATION_STARTED_REGISTRATION,
    };
    EvidenceRecorder::new(
        "actual-b-capture-fit",
        [
            CALIBRATED_LEVELS_FITTED_REGISTRATION,
            CALIBRATED_QUANTIZATION_STARTED_REGISTRATION,
            CALIBRATED_QUANTIZATION_COMPLETED_REGISTRATION,
        ],
    )
    .unwrap()
}
