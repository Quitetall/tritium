//! Actual mixed-fit owner files through both CLI adapters and the same engine.
use super::*;
use std::sync::Arc;

#[test]
fn owner_convert_and_quantize_match_engine_and_reject_substitution() {
    for precision in [
        ScalePrecision::F32,
        ScalePrecision::F16,
        ScalePrecision::Bf16,
    ] {
        let dir = Scratch::new();
        let mut raw = source(precision);
        raw.tensors
            .insert("owner".into(), raw.tensors["weight"].clone());
        raw.tensors
            .insert("plain".into(), raw.tensors["weight"].clone());
        let (legacy_plan, legacy) = calibrated(&mut raw);
        let original = legacy_plan
            .tensors
            .iter()
            .find(|t| t.source.name == "owner")
            .unwrap()
            .source
            .clone();
        let uses = [OwnerScoreUse::Matmul {
            samples: 1,
            inputs: &[0.25; 32],
            adjoints: &[0.5],
        }];
        let cache = OwnerScoreCache::new(
            OwnerScoreCacheInput {
                source: legacy_plan.original_source,
                tensor: original,
                tokenizer: b"{}",
                objective: b"synthetic CLI owner fixture",
                meta: OwnerScoreWindowMeta {
                    context: vec![OwnerContextMeta {
                        name: "input_ids".into(),
                        dtype: OwnerContextDtype::I64,
                        shape: vec![1],
                        conjugated: false,
                        negated: false,
                    }],
                    invocations: vec![OwnerInvocationMeta {
                        ordinal: 0,
                        paths: vec!["owner".into()],
                        role: OwnerScoreRole::Matmul,
                        samples: 1,
                    }],
                },
                context: &[&0_i64.to_le_bytes()],
                observations: Arc::new(
                    OwnerScoreWindow::new(1, 32, &uses, 4096)
                        .unwrap()
                        .snapshot()
                        .unwrap(),
                ),
            },
            OwnerScoreCacheLimits {
                max_payload_bytes: 8192,
                max_context_bytes: 64,
                max_context_tensors: 4,
                max_invocations: 4,
                max_samples: 4,
            },
        )
        .unwrap();
        let id = cache
            .id()
            .as_bytes()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        std::fs::write(
            dir.0.join("owner.trit"),
            cache.encode_package(100_000).unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.0.join("legacy.trit"),
            legacy.encode_package(100_000).unwrap(),
        )
        .unwrap();
        let owner = OwnerScoreFitPlan {
            direct: direct(),
            windows: vec![
                WeightedOwnerCache {
                    binding: cache.binding().clone(),
                    weight: 1.,
                },
                WeightedOwnerCache {
                    binding: cache.binding().clone(),
                    weight: 0.5,
                },
            ],
            damping: 0.1,
            limits: OwnerScoreMetricLimits {
                max_windows: 4,
                max_payload_bytes: 16384,
                max_row_bytes: 256,
                max_reduction_entries: 100_000,
            },
            max_row_bindings: 1,
        };
        let request = OwnerQuantPlan {
            architecture: legacy_plan.architecture,
            original_source: legacy_plan.original_source,
            assets: legacy_plan.assets,
            levels: legacy_plan.levels,
            limits: legacy_plan.limits,
            max_cache_payload_bytes: 100_000,
            max_cache_package_bytes: 100_000,
            tensors: legacy_plan
                .tensors
                .into_iter()
                .map(|t| OwnerQuantTensorPlan {
                    action: match t.source.name.as_str() {
                        "owner" => OwnerQuantAction::OwnerFit {
                            plan: Box::new(owner.clone()),
                        },
                        "plain" => OwnerQuantAction::Fit {
                            plan: Box::new(direct()),
                            metric: DirectFitMetric::Identity,
                        },
                        _ => match t.action {
                            CalibratedQuantAction::KeepDense => OwnerQuantAction::KeepDense,
                            CalibratedQuantAction::CalibratedFit { plan } => {
                                OwnerQuantAction::CalibratedFit { plan }
                            }
                            _ => unreachable!(),
                        },
                    },
                    source: t.source,
                })
                .collect(),
        };
        std::fs::write(
            dir.0.join("plan.json"),
            serde_json::to_vec(&request).unwrap(),
        )
        .unwrap();
        let mut owners =
            InMemoryOwnerCalibrationSource::new([("owner".into(), vec![cache])]).unwrap();
        let mut legacy = InMemoryCalibrationSource::new([legacy]).unwrap();
        let mut expected = MemoryTensorSink::default();
        Engine::run_owner(
            &request,
            &mut raw,
            &mut owners,
            Some(&mut legacy),
            &mut expected,
            recorder(),
        )
        .unwrap();
        let mut args = arguments(&dir.0, &raw, false, "converted");
        args[1] = "--owner-plan".into();
        let binding = format!("owner={id}={}", dir.0.join("owner.trit").display());
        args.extend([
            "--owner-cache".into(),
            binding.clone().into(),
            "--calibration-cache".into(),
            format!("weight={}", dir.0.join("legacy.trit").display()).into(),
        ]);
        check(&dir.0, &args, &expected);
        args[0] = "quantize".into();
        let out = args.iter().position(|a| a == "--out").unwrap() + 1;
        args[out] = dir.0.join("quantized").into();
        check(&dir.0, &args, &expected);
        for (flag, value) in [
            ("--owner-cache", binding),
            ("--owner-cache", "extra=00=missing".into()),
            ("--plan", dir.0.join("plan.json").display().to_string()),
            ("--max-plan-bytes", "1".into()),
        ] {
            let mut bad = args.clone();
            bad[out] = dir.0.join("rejected").into();
            bad.extend([flag.into(), value.into()]);
            assert!(!run(&bad).status.success());
            assert!(!dir.0.join("rejected").exists());
        }
        args[out] = dir.0.join("changed").into();
        let mut changed = raw.tensors["owner"].clone();
        changed.bytes[0] ^= 1;
        std::fs::write(dir.0.join("owner.safetensors"), shard("owner", &changed)).unwrap();
        assert!(!run(&args).status.success());
        assert!(!dir.0.join("changed").exists());
    }
}
