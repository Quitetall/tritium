//! Actual mandatory B source -> shared graph -> actual projection VJPs.
use super::*;
use tritium_nn::calibrate::{StagedCalibrationLimits, StagedCalibrationModel};
use tritium_schema::ScalePrecision;
use tritium_train::Tape;

fn limits() -> StagedCalibrationLimits {
    StagedCalibrationLimits {
        max_config_bytes: 16_000,
        max_dense_bytes: 64_000,
        max_window_tokens: 4,
    }
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
