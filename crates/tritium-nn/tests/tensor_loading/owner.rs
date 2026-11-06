//! Owner-cache model conversion exercised by the existing real text loader test.
//! Synthetic captured observations test integration, not qualified calibration.
use std::{collections::BTreeMap, sync::Arc};
use tritium_evidence::*;
use tritium_quantize::engine::*;
use tritium_schema::*;

pub(super) fn convert(
    plan: &QuantPlan,
    source: &mut dyn WeightSource,
    sink: &mut dyn TensorSink,
) -> Result<QuantizedModel, QuantEngineError> {
    let original_source = Engine::observe_source(plan, source)?;
    let selected = plan
        .tensors
        .iter()
        .find(|t| t.source.name == "model.embed_tokens.weight")
        .unwrap();
    let rows = selected.source.shape[0] as usize;
    let cols = selected.source.shape[1] as usize;
    let input = vec![0.25; cols];
    let head_adjoint = vec![0.5; rows];
    let gather_adjoint = vec![0.125; cols];
    let uses = [
        OwnerScoreUse::Matmul {
            samples: 1,
            inputs: &input,
            adjoints: &head_adjoint,
        },
        OwnerScoreUse::Gather {
            indices: &[1],
            adjoints: &gather_adjoint,
        },
    ];
    let tokenizer = source.asset("tokenizer.json", 100_000)?.into_owned();
    let cache = OwnerScoreCache::new(
        OwnerScoreCacheInput {
            source: original_source,
            tensor: selected.source.clone(),
            tokenizer: &tokenizer,
            objective: b"synthetic tied-owner integration fixture",
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
                        paths: vec!["lm_head".into()],
                        role: OwnerScoreRole::Matmul,
                        samples: 1,
                    },
                    OwnerInvocationMeta {
                        ordinal: 1,
                        paths: vec!["model.embed_tokens".into()],
                        role: OwnerScoreRole::Gather,
                        samples: 1,
                    },
                ],
            },
            context: &[&1_i64.to_le_bytes()],
            observations: Arc::new(
                OwnerScoreWindow::new(rows, cols, &uses, 100_000)
                    .unwrap()
                    .snapshot()
                    .unwrap(),
            ),
        },
        OwnerScoreCacheLimits {
            max_payload_bytes: 100_000,
            max_context_bytes: 1024,
            max_context_tensors: 8,
            max_invocations: 8,
            max_samples: 32,
        },
    )
    .unwrap();
    let QuantAction::Fit { plan: direct, .. } = &selected.action else {
        unreachable!()
    };
    let fit = OwnerScoreFitPlan {
        direct: (**direct).clone(),
        windows: vec![WeightedOwnerCache {
            binding: cache.binding().clone(),
            weight: 1.,
        }],
        damping: 0.1,
        limits: OwnerScoreMetricLimits {
            max_windows: 8,
            max_payload_bytes: 100_000,
            max_row_bytes: 1024,
            max_reduction_entries: 100_000,
        },
        max_row_bindings: rows as u64,
    };
    let request = OwnerQuantPlan {
        architecture: plan.architecture.clone(),
        original_source,
        tensors: plan
            .tensors
            .iter()
            .map(|tensor| OwnerQuantTensorPlan {
                source: tensor.source.clone(),
                action: if tensor.source == selected.source {
                    OwnerQuantAction::OwnerFit {
                        plan: Box::new(fit.clone()),
                    }
                } else {
                    match &tensor.action {
                        QuantAction::KeepDense => OwnerQuantAction::KeepDense,
                        QuantAction::Fit { plan, metric } => OwnerQuantAction::Fit {
                            plan: plan.clone(),
                            metric: metric.clone(),
                        },
                        _ => unreachable!(),
                    }
                },
            })
            .collect(),
        assets: plan.assets.clone(),
        levels: plan.levels.clone(),
        limits: plan.limits,
        max_cache_payload_bytes: 100_000,
        max_cache_package_bytes: 100_000,
    };
    let mut caches = InMemoryOwnerCalibrationSource::new(BTreeMap::from([(
        selected.source.name.clone(),
        vec![cache],
    )]))?;
    Engine::run_owner(
        &request,
        source,
        &mut caches,
        None,
        sink,
        EvidenceRecorder::new(
            "nn-model-fit",
            [
                DIRECT_LEVELS_FITTED_REGISTRATION,
                OWNER_SCORE_LEVELS_FITTED_REGISTRATION,
                OWNER_QUANTIZATION_STARTED_REGISTRATION,
                OWNER_QUANTIZATION_COMPLETED_REGISTRATION,
            ],
        )?,
    )
}
