//! Actual mandatory B source -> shared graph -> actual projection VJPs.
use super::*;
use tritium_nn::calibrate::{StagedCalibrationLimits, StagedCalibrationModel};
use tritium_schema::ScalePrecision;
use tritium_train::Tape;

#[cfg(feature = "tokenizer")]
#[path = "staged/text.rs"]
mod text;

fn limits() -> StagedCalibrationLimits {
    StagedCalibrationLimits {
        max_config_bytes: 16_000,
        max_dense_bytes: 64_000,
        max_window_tokens: 4,
    }
}

fn capture_limits() -> tritium_nn::calibrate::StagedCaptureLimits {
    tritium_nn::calibrate::StagedCaptureLimits {
        per_cache: tritium_schema::CalibrationCacheLimits {
            max_payload_bytes: 64_000,
            max_samples: 8,
            max_windows: 3,
        },
        max_total_payload_bytes: 512_000,
    }
}

#[test]
fn actual_b_next_token_capture_matches_dense_objective_cache_and_replay() {
    use tritium_quantize::engine::{
        CalibrationBatch, CalibrationCache, CalibrationCacheBuilder, CalibrationInput,
        CalibrationOutputBatch,
    };
    use tritium_schema::{CalibrationConsumer, CalibrationPartition};
    let windows = [&[1, 4, 2][..], &[3, 7][..]];
    for precision in [
        ScalePrecision::F32,
        ScalePrecision::F16,
        ScalePrecision::Bf16,
    ] {
        let hf = HfFixture::new("qwen3", true, 6, None);
        let mut input = TritFixture::new(&hf, None);
        if precision != ScalePrecision::F32 {
            input.dense_precision(precision);
        }
        let fixture = produced(&input, false);
        let mut stage =
            StagedSalienceSource::open(&fixture.b, &fixture.transform, stage_limits()).unwrap();
        let model = StagedCalibrationModel::new(&mut stage, limits()).unwrap();
        let caches = model
            .capture_next_token(&mut stage, &windows, capture_limits())
            .unwrap();
        let repeated = model
            .capture_next_token(&mut stage, &windows, capture_limits())
            .unwrap();
        assert_eq!(caches.len(), 7 * model.arch().n_layers + 1);
        let names = tritium_nn::calibrate::weight_names(model.arch());
        let inventory = stage.inventory().unwrap();
        let mut oracle: Vec<_> = caches
            .iter()
            .map(|cache| {
                assert_eq!(cache.binding().request.model, stage.source_id());
                assert_eq!(cache.binding().samples, 3);
                assert_eq!(cache.binding().windows, 2);
                assert_eq!(
                    cache.binding().request.consumer,
                    CalibrationConsumer::Matmul
                );
                assert_eq!(
                    cache.tokenizer(),
                    stage.asset("tokenizer.json", 64_000).unwrap().as_ref()
                );
                assert_eq!(
                    &cache.binding().request.tensor,
                    inventory
                        .iter()
                        .find(|t| t.name == cache.binding().request.tensor.name)
                        .unwrap()
                );
                CalibrationCacheBuilder::new(
                    cache.binding().request.clone(),
                    cache.tokenizer(),
                    cache.objective(),
                )
                .unwrap()
            })
            .collect();
        for window in windows {
            let tokens = &window[..window.len() - 1];
            let mut tape = Tape::new();
            let mut taps = Vec::new();
            let logits = model
                .forward_observed(&mut tape, tokens, &mut |_, tap| {
                    if tap.input.is_some() {
                        taps.push(tap);
                    }
                })
                .unwrap();
            // Separately execute the existing dense one-hot CE, not a copied loss.
            let mut target = vec![0.0; tokens.len() * model.arch().vocab];
            for (i, &id) in window[1..].iter().enumerate() {
                target[i * model.arch().vocab + id as usize] = 1.0;
            }
            let target = tape.leaf(target);
            let loss = tape.softmax_xent(logits, target, tokens.len(), model.arch().vocab);
            let gradients = tape.backward(loss);
            for ((builder, cache), tap) in oracle.iter_mut().zip(&caches).zip(taps) {
                assert_eq!(cache.binding().request.tensor.name, names[tap.weight_index]);
                builder
                    .capture(
                        tokens,
                        CalibrationBatch {
                            samples: tokens.len(),
                            input: CalibrationInput::Activations(tape.value(tap.input.unwrap())),
                            output: CalibrationOutputBatch::Dense(&gradients[tap.output]),
                            weights: None,
                            mask: None,
                        },
                    )
                    .unwrap();
            }
        }
        for ((cache, repeated), oracle) in caches.iter().zip(&repeated).zip(oracle) {
            let oracle = oracle.finish().unwrap();
            assert_eq!(cache.id(), repeated.id());
            assert_eq!(cache.id(), oracle.id());
            let full = cache
                .collect(CalibrationPartition::Full, 0.125, 8_000_000)
                .unwrap();
            let chunked = cache
                .collect_chunked(CalibrationPartition::Full, 0.125, 8_000_000, 1)
                .unwrap();
            assert_eq!(full.factors(), chunked.factors());
            assert_eq!(
                full.factors(),
                oracle
                    .collect(CalibrationPartition::Full, 0.125, 8_000_000)
                    .unwrap()
                    .factors()
            );
            let bytes = cache.encode_package(1_000_000).unwrap();
            let decoded = CalibrationCache::decode_package(
                &bytes,
                cache.id(),
                capture_limits().per_cache,
                1_000_000,
            )
            .unwrap();
            assert_eq!(decoded.binding(), cache.binding());
            assert_eq!(
                decoded
                    .collect(CalibrationPartition::Full, 0.125, 8_000_000)
                    .unwrap()
                    .factors(),
                full.factors()
            );
            let mut changed = bytes.clone();
            changed[256] ^= 1;
            assert!(
                CalibrationCache::decode_package(
                    &changed,
                    cache.id(),
                    capture_limits().per_cache,
                    1_000_000
                )
                .is_err()
            );
        }
        assert_eq!(
            caches.last().unwrap().binding().request.tensor.precision,
            precision
        );
        let changed = model
            .capture_next_token(&mut stage, &[&[1, 4, 3], &[3, 7]], capture_limits())
            .unwrap();
        for (old, new) in caches.iter().zip(changed) {
            assert_eq!(old.binding().source.tokens, new.binding().source.tokens);
            assert_ne!(old.binding().objective, new.binding().objective);
            assert_ne!(old.id(), new.id());
        }
    }
}

#[test]
fn b_capture_admits_exact_payload_limits_and_rejects_foreign_stage_and_bad_windows() {
    let hf = HfFixture::new("llama", true, 4, None);
    let input = TritFixture::new(&hf, None);
    let fixture = produced(&input, false);
    let mut stage =
        StagedSalienceSource::open(&fixture.b, &fixture.transform, stage_limits()).unwrap();
    let model = StagedCalibrationModel::new(&mut stage, limits()).unwrap();
    let windows = [&[1, 4, 2][..], &[3, 7][..]];
    let caches = model
        .capture_next_token(&mut stage, &windows, capture_limits())
        .unwrap();
    let total = caches
        .iter()
        .map(|c| c.binding().payload_bytes)
        .sum::<u64>();
    let maximum = caches
        .iter()
        .map(|c| c.binding().payload_bytes)
        .max()
        .unwrap();
    let exact = tritium_nn::calibrate::StagedCaptureLimits {
        per_cache: tritium_schema::CalibrationCacheLimits {
            max_payload_bytes: maximum,
            max_samples: 3,
            max_windows: 2,
        },
        max_total_payload_bytes: total,
    };
    assert!(
        model
            .capture_next_token(&mut stage, &windows, exact)
            .is_ok()
    );
    for bad in [
        tritium_nn::calibrate::StagedCaptureLimits {
            max_total_payload_bytes: total - 1,
            ..exact
        },
        tritium_nn::calibrate::StagedCaptureLimits {
            per_cache: tritium_schema::CalibrationCacheLimits {
                max_payload_bytes: maximum - 1,
                ..exact.per_cache
            },
            ..exact
        },
        tritium_nn::calibrate::StagedCaptureLimits {
            per_cache: tritium_schema::CalibrationCacheLimits {
                max_samples: 2,
                ..exact.per_cache
            },
            ..exact
        },
        tritium_nn::calibrate::StagedCaptureLimits {
            per_cache: tritium_schema::CalibrationCacheLimits {
                max_windows: 1,
                ..exact.per_cache
            },
            ..exact
        },
    ] {
        assert!(model.capture_next_token(&mut stage, &windows, bad).is_err());
    }
    for bad in [&[][..], &[1][..], &[1, 99][..], &[1, 2, 3, 4, 5, 6][..]] {
        assert!(
            model
                .capture_next_token(&mut stage, &[bad], capture_limits())
                .is_err()
        );
    }
    assert!(
        model
            .capture_next_token(&mut stage, &[], capture_limits())
            .is_err()
    );
    let mut foreign_input = TritFixture::new(&hf, None);
    foreign_input.asset("tokenizer.json", b"different-tokenizer".to_vec());
    let foreign = produced(&foreign_input, false);
    let mut foreign_stage =
        StagedSalienceSource::open(&foreign.b, &foreign.transform, stage_limits()).unwrap();
    assert_ne!(stage.source_id(), foreign_stage.source_id());
    assert!(
        model
            .capture_next_token(&mut foreign_stage, &windows, capture_limits())
            .is_err()
    );
    assert_eq!(model.source_id(), stage.source_id());
    assert!(
        model
            .capture_next_token(&mut stage, &windows, exact)
            .is_ok()
    );
}

#[test]
fn admitted_b_graph_matches_actual_joined_dense_runtime_and_vjps_after_source_drop() {
    for precision in [
        ScalePrecision::F32,
        ScalePrecision::F16,
        ScalePrecision::Bf16,
    ] {
        for head_dim in [2, 6] {
            let mut hf = HfFixture::new("qwen3", true, head_dim, None);
            // Actual original package values, not a second normalization oracle.
            for (name, _, values) in &mut hf.tensors {
                if name.ends_with("q_norm.weight") || name.ends_with("k_norm.weight") {
                    for (i, value) in values.iter_mut().enumerate() {
                        *value = if name.ends_with("q_norm.weight") {
                            0.7 + i as f32 * 0.13
                        } else {
                            1.2 - i as f32 * 0.07
                        };
                    }
                }
            }
            let mut input = TritFixture::new(&hf, None);
            if precision != ScalePrecision::F32 {
                input.dense_precision(precision);
            }
            let fixture = produced(&input, false);
            let mut stage =
                StagedSalienceSource::open(&fixture.b, &fixture.transform, stage_limits()).unwrap();
            let id = stage.source_id();
            assert_ne!(id, stage.manifest().observation.original_source);
            let required: u64 = stage
                .inventory()
                .unwrap()
                .iter()
                .map(|t| t.shape.iter().product::<u64>() * 4)
                .sum();
            assert!(
                StagedCalibrationModel::new(
                    &mut stage,
                    StagedCalibrationLimits {
                        max_dense_bytes: required - 1,
                        ..limits()
                    }
                )
                .is_err()
            );
            let model = StagedCalibrationModel::new(
                &mut stage,
                StagedCalibrationLimits {
                    max_dense_bytes: required,
                    ..limits()
                },
            )
            .unwrap();
            assert_eq!(model.source_id(), id);
            assert!(model.arch().qk_norms[0].is_some());
            let joined =
                JoinedSalienceModel::open(&fixture.c, &stage, &fixture.fit, conversion_limits())
                    .unwrap();
            let budget = TensorPayloadBudget::new(100_000);
            let mut recorder =
                EvidenceRecorder::new("staged-runtime", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
            let mut runner = ModelRunner::from_source_admitted(
                tritium_nn::ModelSource::JoinedSalience {
                    model: &joined,
                    level: "standard",
                },
                Box::new(tritium_cpu::CpuBackend::new()),
                TensorUploadPolicy::default(),
                &budget,
                TensorUploadContext {
                    recorder: &mut recorder,
                    span: "load",
                    parent: None,
                    logical_time: 0,
                },
            )
            .unwrap();
            drop(joined);
            drop(stage);
            drop(fixture);
            for tokens in [&[1][..], &[1, 4, 2][..], &[1, 4, 2, 7][..]] {
                runner.reset();
                let expected = runner
                    .forward(tokens, &(0..tokens.len()).collect::<Vec<_>>())
                    .unwrap();
                let mut tape = Tape::new();
                let mut taps = Vec::new();
                let logits = model
                    .forward_observed(&mut tape, tokens, &mut |_, tap| taps.push(tap))
                    .unwrap();
                let actual = &tape.value(logits)[(tokens.len() - 1) * model.arch().vocab..];
                assert_eq!(actual.len(), expected.len());
                for (a, b) in actual.iter().zip(&expected) {
                    assert!((a - b).abs() < 2e-5, "{precision:?}/{head_dim}: {a} != {b}");
                }
                assert_eq!(taps.len(), 2 + 7 * model.arch().n_layers);
                let target = tape.leaf(vec![0.25; tape.value(logits).len()]);
                let loss = tape.mse(logits, target);
                let gradients = tape.backward(loss);
                for tap in &taps[1..taps.len() - 1] {
                    let expected = tritium_train::ops::dense::vjp(
                        tape.value(tap.input.unwrap()),
                        tape.value(tap.weight),
                        tap.samples,
                        tap.rows,
                        tap.columns,
                        &gradients[tap.output],
                    );
                    assert_eq!(gradients[tap.weight], expected[1]);
                    assert!(gradients[tap.weight].iter().all(|g| g.is_finite()));
                }
                assert_eq!(model.source_id(), id);
            }
            drop(runner);
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

#[test]
fn staged_limits_and_tokens_reject_before_tape_mutation() {
    let hf = HfFixture::new("llama", true, 4, None);
    let fixture = produced(&TritFixture::new(&hf, None), false);
    let mut stage =
        StagedSalienceSource::open(&fixture.b, &fixture.transform, stage_limits()).unwrap();
    let id = stage.source_id();
    for bad in [
        StagedCalibrationLimits {
            max_config_bytes: 1,
            ..limits()
        },
        StagedCalibrationLimits {
            max_dense_bytes: 1,
            ..limits()
        },
        StagedCalibrationLimits {
            max_window_tokens: 0,
            ..limits()
        },
    ] {
        assert!(StagedCalibrationModel::new(&mut stage, bad).is_err());
        assert_eq!(stage.source_id(), id);
    }
    let model = StagedCalibrationModel::new(&mut stage, limits()).unwrap();
    for tokens in [&[][..], &[10][..], &[1, 2, 3, 4, 5][..]] {
        let mut tape = Tape::new();
        let before = tape.leaf(vec![1.]);
        let mut seen = false;
        assert!(
            model
                .forward_observed(&mut tape, tokens, &mut |_, _| seen = true)
                .is_err()
        );
        assert!(!seen);
        assert_eq!(tape.leaf(vec![2.]), before + 1);
    }
}

#[test]
fn unsupported_staged_graphs_and_false_coverage_never_become_calibration_models() {
    for (arch, tied) in [("qwen2", true), ("qwen3", false)] {
        let hf = HfFixture::new(arch, tied, 6, None);
        let fixture = produced(&TritFixture::new(&hf, None), false);
        let mut stage =
            StagedSalienceSource::open(&fixture.b, &fixture.transform, stage_limits()).unwrap();
        assert!(StagedCalibrationModel::new(&mut stage, limits()).is_err());
    }
    for bad in [
        "extra",
        "missing-output",
        "output-shape",
        "architecture",
        "relu2",
        "hybrid",
        "huge-layers",
    ] {
        let mut hf = HfFixture::new(
            "llama",
            true,
            4,
            if bad == "extra" { Some("extra") } else { None },
        );
        if bad == "missing-output" {
            hf.tensors
                .retain(|(name, _, _)| !name.ends_with("o_proj.weight"));
        }
        if bad == "output-shape" {
            hf.tensors
                .iter_mut()
                .find(|(name, _, _)| name.ends_with("o_proj.weight"))
                .unwrap()
                .1 = vec![4, 16];
        }
        let mut input = TritFixture::new(&hf, None);
        if bad == "architecture" {
            input.manifest.architecture = "not-llama".into();
        }
        if matches!(bad, "relu2" | "hybrid" | "huge-layers") {
            let mut config: serde_json::Value =
                serde_json::from_slice(&std::fs::read(hf.dir.join("config.json")).unwrap())
                    .unwrap();
            if bad == "relu2" {
                config["hidden_act"] = "relu2".into();
            }
            if bad == "hybrid" {
                config["model_type"] = "qwen3_5".into();
                input.manifest.architecture = "qwen3_5".into();
            }
            if bad == "huge-layers" {
                config["num_hidden_layers"] = u32::MAX.into();
            }
            input.asset("config.json", serde_json::to_vec(&config).unwrap());
        }
        let fixture = produced(&input, false);
        let mut stage =
            StagedSalienceSource::open(&fixture.b, &fixture.transform, stage_limits()).unwrap();
        assert!(
            StagedCalibrationModel::new(&mut stage, limits()).is_err(),
            "{bad}"
        );
    }
}
