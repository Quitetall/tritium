//! Actual graph-produced A/B inputs cross the command's filesystem seam.
//! Tiny CPU software evidence, not Qwen quality or release qualification.
use super::*;
use tritium_nn::calibrate::{
    SourceCalibrationModel, StagedCalibrationLimits, StagedCalibrationModel,
};

#[path = "original_stage/generate.rs"]
mod generation;

fn stage_limits() -> SalienceStageLimits {
    SalienceStageLimits {
        max_package_bytes: 1_000_000,
        max_manifest_bytes: 256_000,
        max_tensor_bytes: 64_000,
        max_asset_bytes: 16_000,
        max_statistics_bytes: 512,
    }
}
fn conversion_limits() -> SalienceConversionLimits {
    SalienceConversionLimits {
        max_package_bytes: 1_000_000,
        max_manifest_bytes: 256_000,
    }
}
fn graph_limits() -> StagedCalibrationLimits {
    StagedCalibrationLimits {
        max_config_bytes: 16_000,
        max_dense_bytes: 64_000,
        max_window_tokens: 4,
    }
}
fn raw(precision: ScalePrecision) -> InMemoryWeightSource {
    let mut tensors = BTreeMap::new();
    let mut add = |name: &str, shape: Vec<usize>| {
        let bytes = (0..shape.iter().product())
            .flat_map(|i| {
                let value = if shape.len() == 1 {
                    1.
                } else {
                    ((i * 17 % 31) as f32 - 15.) * 0.01
                };
                match precision {
                    ScalePrecision::F32 => value.to_le_bytes().to_vec(),
                    ScalePrecision::F16 => {
                        half::f16::from_f32(value).to_bits().to_le_bytes().to_vec()
                    }
                    ScalePrecision::Bf16 => {
                        half::bf16::from_f32(value).to_bits().to_le_bytes().to_vec()
                    }
                    _ => unreachable!(),
                }
            })
            .collect();
        tensors.insert(
            name.into(),
            RawTensor {
                shape: shape
                    .into_iter()
                    .map(|n| u64::try_from(n).unwrap())
                    .collect(),
                precision,
                bytes,
            },
        );
    };
    add("model.embed_tokens.weight", vec![10, 8]);
    for name in [
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
    ] {
        add(name, vec![8]);
    }
    for (name, rows, cols) in [
        ("self_attn.q_proj.weight", 12, 8),
        ("self_attn.k_proj.weight", 6, 8),
        ("self_attn.v_proj.weight", 6, 8),
        ("self_attn.o_proj.weight", 8, 12),
        ("mlp.gate_proj.weight", 16, 8),
        ("mlp.up_proj.weight", 16, 8),
        ("mlp.down_proj.weight", 8, 16),
    ] {
        add(&format!("model.layers.0.{name}"), vec![rows, cols]);
    }
    for name in ["q", "k"] {
        add(
            &format!("model.layers.0.self_attn.{name}_norm.weight"),
            vec![6],
        );
    }
    InMemoryWeightSource {
        tensors,
        assets: BTreeMap::from([
            (
                "config.json".into(),
                serde_json::to_vec(&serde_json::json!({
                    "model_type":"qwen3", "hidden_size":8, "num_hidden_layers":1,
                    "num_attention_heads":2, "num_key_value_heads":1, "head_dim":6,
                    "intermediate_size":16, "max_position_embeddings":16,
                    "rms_norm_eps":1e-5, "hidden_act":"silu", "tie_word_embeddings":true,
                    "rope_theta":10000., "vocab_size":10,
                }))
                .unwrap(),
            ),
            ("tokenizer.json".into(), b"{}".to_vec()),
        ]),
        grams: BTreeMap::new(),
    }
}

struct Fixture {
    dir: Scratch,
    b: Vec<u8>,
    original: VerifiedLog,
    transform: VerifiedLog,
    capture: VerifiedLog,
    plan: CalibratedQuantPlan,
    expected: Vec<u8>,
    fit: String,
}
impl Fixture {
    fn new(precision: ScalePrecision) -> Self {
        Self::from_raw(raw(precision))
    }
    fn with_tokenizer() -> Self {
        let mut source = raw(ScalePrecision::Bf16);
        let vocab = [
            "[UNK]", "hello", "partner", "[EOS]", "[BOS]", "t5", "t6", "t7", "t8", "t9",
        ]
        .into_iter()
        .enumerate()
        .map(|(id, word)| (word.to_owned(), u32::try_from(id).unwrap()))
        .collect();
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
        source.assets.insert(
            "tokenizer.json".into(),
            tokenizer.to_string(false).unwrap().into_bytes(),
        );
        source.assets.insert(
            "tokenizer_config.json".into(),
            br#"{"bos_token":"[BOS]","eos_token":{"content":"[EOS]"}}"#.to_vec(),
        );
        Self::from_raw(source)
    }
    fn from_raw(mut raw: InMemoryWeightSource) -> Self {
        let dir = Scratch::new();
        let request = QuantPlan {
            architecture: "qwen3".into(),
            tensors: raw
                .inventory()
                .unwrap()
                .into_iter()
                .map(|source| QuantTensorPlan {
                    source,
                    action: QuantAction::KeepDense,
                })
                .collect(),
            assets: raw.asset_names(),
            levels: vec!["standard".into()],
            limits: QuantLimits {
                max_source_bytes: 64_000,
                max_asset_bytes: 16_000,
                max_retained_blob_bytes: 64_000,
                max_package_bytes: 1_000_000,
            },
        };
        let mut observed = ObservedWeightSource::new(&request, &mut raw).unwrap();
        let model = SourceCalibrationModel::new(&mut observed, graph_limits()).unwrap();
        let captured = model
            .capture_salience_recorded(
                &mut observed,
                &[&[1, 3, 2], &[4]],
                OriginalSalienceCaptureLimits {
                    max_source_bytes: 64_000,
                    max_asset_bytes: 16_000,
                    max_config_bytes: 16_000,
                    max_dense_bytes: 64_000,
                    max_window_tokens: 4,
                    max_windows: 4,
                    max_total_tokens: 8,
                    max_moment_bytes: 512,
                    max_text_bytes: 0,
                    max_tokenizer_config_bytes: 0,
                },
                EvidenceRecorder::new(
                    "cli-original",
                    [
                        ORIGINAL_SALIENCE_CAPTURE_STARTED_REGISTRATION,
                        ORIGINAL_SALIENCE_CAPTURE_COMPLETED_REGISTRATION,
                    ],
                )
                .unwrap(),
            )
            .unwrap();
        drop(model);
        let view = OriginalSalienceCaptureView::fold(captured.evidence().log())
            .unwrap()
            .remove(0);
        let original = &view.started().plan;
        let transform_plan = SalienceTransformPlan {
            architecture: original.architecture.clone(),
            original_source: original.source,
            tensors: original.tensors.iter().map(|b| b.source.clone()).collect(),
            assets: original.assets.iter().map(|a| a.name.clone()).collect(),
            operations: original
                .pairs
                .iter()
                .map(|p| SalienceTransformOperation {
                    name: p.name.clone(),
                    projections: p.projections.clone(),
                    inverse: p.inverse.clone(),
                    inverse_axis: p.inverse_axis,
                    moments: captured.statistics()[&p.name].binding().unwrap(),
                    alpha_bits: 0.75f64.to_le_bytes(),
                })
                .collect(),
            numerics: SalienceNumerics::LegacyF64ScaleF32V1,
            statistics_kind: SalienceStatisticsKind::Declared,
            limits: SalienceTransformLimits {
                max_source_bytes: 64_000,
                max_asset_bytes: 16_000,
                max_derived_bytes: 64_000,
                max_statistics_bytes: 512,
                max_pair_working_bytes: 64_000,
            },
        };
        let recorded = captured
            .transform_recorded(
                &transform_plan,
                &mut observed,
                EvidenceRecorder::new(
                    "cli-transform",
                    [
                        SALIENCE_TRANSFORM_STARTED_REGISTRATION,
                        SALIENCE_TRANSFORM_COMPLETED_REGISTRATION,
                    ],
                )
                .unwrap(),
            )
            .unwrap();
        let b = recorded.encode_stage_package(stage_limits()).unwrap();
        let original = verify_jsonl(recorded.original_capture_evidence().jsonl()).unwrap();
        let transform = verify_jsonl(recorded.transform_evidence().jsonl()).unwrap();
        std::fs::write(dir.0.join("stage.trit"), &b).unwrap();
        std::fs::write(
            dir.0.join("original.jsonl"),
            recorded.original_capture_evidence().jsonl(),
        )
        .unwrap();
        std::fs::write(
            dir.0.join("transform.jsonl"),
            recorded.transform_evidence().jsonl(),
        )
        .unwrap();
        let mut stage =
            StagedSalienceSource::open_original_captured(&b, &transform, &original, stage_limits())
                .unwrap();
        let model = StagedCalibrationModel::new(&mut stage, graph_limits()).unwrap();
        let captured = model
            .capture_next_token_recorded(
                &mut stage,
                &[&[1, 4, 2], &[3, 7]],
                tritium_nn::calibrate::StagedCaptureLimits {
                    per_cache: CalibrationCacheLimits {
                        max_payload_bytes: 64_000,
                        max_samples: 8,
                        max_windows: 3,
                    },
                    max_total_payload_bytes: 512_000,
                },
                EvidenceRecorder::new(
                    "cli-b-capture",
                    [
                        CALIBRATION_CAPTURE_STARTED_REGISTRATION,
                        CALIBRATION_CAPTURE_COMPLETED_REGISTRATION,
                    ],
                )
                .unwrap(),
            )
            .unwrap();
        let capture = verify_jsonl(captured.evidence.jsonl()).unwrap();
        std::fs::write(dir.0.join("capture.jsonl"), captured.evidence.jsonl()).unwrap();
        let caches: Vec<_> = captured
            .caches
            .into_iter()
            .filter(|c| c.binding().request.tensor.name != "model.embed_tokens.weight")
            .collect();
        let plan = CalibratedQuantPlan {
            architecture: "qwen3".into(),
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
                            panic!("matrix required")
                        };
                        let mut direct = direct();
                        direct.layout.rows = *rows;
                        direct.layout.cols = *cols;
                        direct.layout.max_planes = 3;
                        direct.layout.law = ADMITTED_LAWS[1].law;
                        direct.levels = vec![DirectFitLevel {
                            name: "standard".into(),
                            planes: 3,
                        }];
                        direct.limits = DirectFitLimits {
                            max_weights: rows * cols,
                            max_metric_entries: cols * cols,
                            max_output_trits: 3 * rows * cols,
                        };
                        CalibratedQuantAction::CalibratedFit {
                            plan: Box::new(CalibratedFitPlan {
                                direct,
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
            limits: request.limits,
            max_cache_payload_bytes: caches.iter().map(|c| c.binding().payload_bytes).sum(),
            max_cache_package_bytes: 1_000_000,
        };
        for cache in &caches {
            std::fs::write(
                dir.0
                    .join(format!("{}.trit", cache.binding().request.tensor.name)),
                cache.encode_package(1_000_000).unwrap(),
            )
            .unwrap();
        }
        for (name, bytes) in [
            ("plan.json", serde_json::to_vec(&plan).unwrap()),
            (
                "stage-limits.json",
                serde_json::to_vec(&stage_limits()).unwrap(),
            ),
            (
                "conversion-limits.json",
                serde_json::to_vec(&conversion_limits()).unwrap(),
            ),
        ] {
            std::fs::write(dir.0.join(name), bytes).unwrap();
        }
        let mut sink = MemoryTensorSink::default();
        Engine::run_original_captured_salience(
            &plan,
            &mut stage,
            &capture,
            &mut InMemoryCalibrationSource::new(caches).unwrap(),
            &mut sink,
            recorder(),
            conversion_limits(),
        )
        .unwrap();
        let (expected, fit) = sink.artifact().unwrap();
        Self {
            dir,
            b,
            original,
            transform,
            capture,
            plan,
            expected: expected.to_vec(),
            fit: fit.jsonl().into(),
        }
    }
    fn args(&self, destination: &str) -> Vec<OsString> {
        let mut args: Vec<OsString> = vec!["convert".into()];
        for (flag, file) in [
            ("--calibrated-plan", "plan.json"),
            ("--original-captured-stage", "stage.trit"),
            ("--transform-evidence", "transform.jsonl"),
            ("--original-salience-evidence", "original.jsonl"),
            ("--capture-evidence", "capture.jsonl"),
            ("--stage-limits", "stage-limits.json"),
            ("--conversion-limits", "conversion-limits.json"),
            ("--out", destination),
        ] {
            args.push(flag.into());
            args.push(self.dir.0.join(file).into());
        }
        args.extend(["--conversion-run-id".into(), "filesystem-conversion".into()]);
        for tensor in &self.plan.tensors {
            if matches!(tensor.action, CalibratedQuantAction::CalibratedFit { .. }) {
                args.extend([
                    "--calibration-cache".into(),
                    format!(
                        "{}={}",
                        tensor.source.name,
                        self.dir
                            .0
                            .join(format!("{}.trit", tensor.source.name))
                            .display()
                    )
                    .into(),
                ]);
            }
        }
        args
    }
    fn rejected(&self, args: &[OsString]) {
        let result = run(args);
        assert!(
            !result.status.success(),
            "unexpected success: {}",
            String::from_utf8_lossy(&result.stdout)
        );
        assert!(
            !self.dir.0.join("rejected").exists(),
            "failed input admitted a destination"
        );
    }
}
fn replace(args: &mut [OsString], flag: &str, value: impl Into<OsString>) {
    let i = args.iter().position(|a| a == flag).unwrap();
    args[i + 1] = value.into();
}
fn logits(model: &JoinedSalienceModel<'_>) -> Vec<Vec<f32>> {
    let budget = tritium_runtime::TensorPayloadBudget::new(100_000);
    let mut uploads = EvidenceRecorder::new("cli-loaded", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
    let mut runner = tritium_nn::ModelRunner::from_source_admitted(
        tritium_nn::ModelSource::JoinedSalience {
            model,
            level: "standard",
        },
        Box::new(tritium_cpu::CpuBackend::new()),
        TensorUploadPolicy::default(),
        &budget,
        tritium_runtime::TensorUploadContext {
            recorder: &mut uploads,
            span: "load",
            parent: None,
            logical_time: 0,
        },
    )
    .unwrap();
    let result = vec![
        runner.forward(&[1, 2], &[0, 1]).unwrap(),
        runner.forward(&[1], &[2]).unwrap(),
    ];
    drop(runner);
    assert_eq!(budget.used_bytes(), 0);
    result
}

#[test]
fn original_captured_command_matches_actual_fitter_and_loaded_decode_in_all_precisions() {
    for precision in [
        ScalePrecision::F32,
        ScalePrecision::F16,
        ScalePrecision::Bf16,
    ] {
        let f = Fixture::new(precision);
        let args = f.args("converted");
        let result = run(&args);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let c = std::fs::read(f.dir.0.join("converted/model.trit")).unwrap();
        let fit = std::fs::read_to_string(f.dir.0.join("converted/evidence.jsonl")).unwrap();
        assert_eq!(c, f.expected);
        assert_eq!(fit, f.fit);
        let stage = StagedSalienceSource::open_original_captured(
            &f.b,
            &f.transform,
            &f.original,
            stage_limits(),
        )
        .unwrap();
        let log = verify_jsonl(&fit).unwrap();
        let expected_log = verify_jsonl(&f.fit).unwrap();
        let model = JoinedSalienceModel::open_original_captured(
            &c,
            &stage,
            &f.capture,
            &log,
            conversion_limits(),
        )
        .unwrap();
        let expected = JoinedSalienceModel::open_original_captured(
            &f.expected,
            &stage,
            &f.capture,
            &expected_log,
            conversion_limits(),
        )
        .unwrap();
        assert_eq!(logits(&model), logits(&expected));
        if precision == ScalePrecision::Bf16
            && let Some(directory) = std::env::var_os("TRITIUM_ORIGINAL_CAPTURED_CLI_OUTPUT")
        {
            let directory = PathBuf::from(directory);
            for entry in std::fs::read_dir(&f.dir.0).unwrap() {
                let entry = entry.unwrap();
                if entry.file_type().unwrap().is_file() {
                    std::fs::copy(entry.path(), directory.join(entry.file_name())).unwrap();
                }
            }
            std::fs::write(directory.join("cli-model.trit"), &c).unwrap();
            std::fs::write(directory.join("engine-model.trit"), &f.expected).unwrap();
            std::fs::write(directory.join("cli-fit.jsonl"), &fit).unwrap();
            std::fs::write(directory.join("engine-fit.jsonl"), &f.fit).unwrap();
        }
        let summary: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
        assert_eq!(
            summary["evidence_root"],
            serde_json::to_value(log.root_digest).unwrap()
        );
        assert!(!run(&args).status.success());
        assert_eq!(
            std::fs::read(f.dir.0.join("converted/model.trit")).unwrap(),
            c
        );
        assert_eq!(
            std::fs::read_to_string(f.dir.0.join("converted/evidence.jsonl")).unwrap(),
            fit
        );
    }
}

#[test]
fn original_captured_command_rejects_conflicts_missing_inputs_and_file_bounds() {
    let f = Fixture::new(ScalePrecision::F32);
    let args = f.args("rejected");
    let sizes: Vec<_> = ["transform.jsonl", "original.jsonl", "capture.jsonl"]
        .into_iter()
        .map(|name| std::fs::metadata(f.dir.0.join(name)).unwrap().len())
        .collect();
    let per_file = *sizes.iter().max().unwrap();
    assert!(per_file < sizes.iter().sum());
    let mut aggregate = args.clone();
    aggregate.extend(["--max-evidence-bytes".into(), per_file.to_string().into()]);
    f.rejected(&aggregate);
    for (flag, value) in [
        ("--source-shard", "raw.safetensors"),
        ("--asset", "config.json=raw"),
        ("--input-gram", "weight=raw"),
        ("--plan", "plan.json"),
        ("--model", "raw"),
        ("--max-evidence-bytes", "0"),
        ("--max-evidence-bytes", "1"),
        ("--max-plan-bytes", "1"),
    ] {
        let mut a = args.clone();
        a.extend([flag.into(), value.into()]);
        f.rejected(&a);
    }
    for flag in [
        "--original-captured-stage",
        "--transform-evidence",
        "--original-salience-evidence",
        "--capture-evidence",
        "--stage-limits",
        "--conversion-limits",
        "--calibrated-plan",
    ] {
        let mut a = args.clone();
        let i = a.iter().position(|v| v == flag).unwrap();
        a.drain(i..i + 2);
        f.rejected(&a);
    }
    let mut a = args.clone();
    let i = a.iter().position(|v| v == "--calibration-cache").unwrap();
    a.drain(i..i + 2);
    f.rejected(&a);
    let mut a = args.clone();
    a.extend(args[args.len() - 2..].iter().cloned());
    f.rejected(&a);
    for flag in [
        "--original-captured-stage",
        "--transform-evidence",
        "--stage-limits",
        "--conversion-limits",
    ] {
        for file in [f.dir.0.join("missing"), f.dir.0.clone()] {
            let mut a = args.clone();
            replace(&mut a, flag, file);
            f.rejected(&a);
        }
    }
    for (file, bytes) in [
        (
            "stage-limits.json",
            serde_json::to_vec(&SalienceStageLimits {
                max_package_bytes: 1,
                ..stage_limits()
            })
            .unwrap(),
        ),
        (
            "conversion-limits.json",
            serde_json::to_vec(&SalienceConversionLimits {
                max_package_bytes: 1,
                ..conversion_limits()
            })
            .unwrap(),
        ),
        ("stage-limits.json", b"{\"unknown\":1}".to_vec()),
        ("conversion-limits.json", b"{\"unknown\":1}".to_vec()),
    ] {
        let path = f.dir.0.join(file);
        let original = std::fs::read(&path).unwrap();
        std::fs::write(&path, bytes).unwrap();
        f.rejected(&args);
        std::fs::write(path, original).unwrap();
    }
}

#[test]
fn original_captured_command_rejects_rehashed_roots_old_profile_and_changed_payloads() {
    let f = Fixture::new(ScalePrecision::F32);
    let args = f.args("rejected");
    let original_view = OriginalSalienceCaptureView::fold(&f.original)
        .unwrap()
        .remove(0);
    let capture_view = CalibrationCaptureView::fold(&f.capture).unwrap().remove(0);
    for complete in [false, true] {
        let mut r = EvidenceRecorder::new(
            "foreign-original",
            [
                ORIGINAL_SALIENCE_CAPTURE_STARTED_REGISTRATION,
                ORIGINAL_SALIENCE_CAPTURE_COMPLETED_REGISTRATION,
            ],
        )
        .unwrap();
        r.emit("capture", None, 0, original_view.started()).unwrap();
        if complete {
            r.emit("capture", None, 1, original_view.completed())
                .unwrap();
        }
        let path = f.dir.0.join("foreign.jsonl");
        std::fs::write(&path, r.to_jsonl().unwrap()).unwrap();
        let mut a = args.clone();
        replace(&mut a, "--original-salience-evidence", path.clone());
        f.rejected(&a);
        let mut r = EvidenceRecorder::new(
            "foreign-b",
            [
                CALIBRATION_CAPTURE_STARTED_REGISTRATION,
                CALIBRATION_CAPTURE_COMPLETED_REGISTRATION,
            ],
        )
        .unwrap();
        r.emit("capture", None, 0, capture_view.started()).unwrap();
        if complete {
            r.emit("capture", None, 1, capture_view.completed())
                .unwrap();
        }
        std::fs::write(&path, r.to_jsonl().unwrap()).unwrap();
        replace(
            &mut a,
            "--original-salience-evidence",
            f.dir.0.join("original.jsonl"),
        );
        replace(&mut a, "--capture-evidence", path.clone());
        if complete {
            // A complete semantically identical B log under a new run is a new
            // producer input, not an attempted substitution into existing C.
            // Prove that the CLI consumes and binds it, and that cold admission
            // of that new C rejects the original B-capture reference.
            let mut candidate = a.clone();
            replace(&mut candidate, "--out", f.dir.0.join("rerooted"));
            let result = run(&candidate);
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            let c = std::fs::read(f.dir.0.join("rerooted/model.trit")).unwrap();
            assert_ne!(c, f.expected);
            let fit = verify_jsonl(
                &std::fs::read_to_string(f.dir.0.join("rerooted/evidence.jsonl")).unwrap(),
            )
            .unwrap();
            let supplied = verify_jsonl(&r.to_jsonl().unwrap()).unwrap();
            let stage = StagedSalienceSource::open_original_captured(
                &f.b,
                &f.transform,
                &f.original,
                stage_limits(),
            )
            .unwrap();
            assert!(
                JoinedSalienceModel::open_original_captured(
                    &c,
                    &stage,
                    &supplied,
                    &fit,
                    conversion_limits()
                )
                .is_ok()
            );
            assert!(
                JoinedSalienceModel::open_original_captured(
                    &c,
                    &stage,
                    &f.capture,
                    &fit,
                    conversion_limits()
                )
                .is_err()
            );
            // Changing a label without recomputing the actual caches is instead
            // a foreign semantic capture and must reject before publication.
            let mut completed = capture_view.completed().clone();
            let last = completed.windows[0].len() - 1;
            completed.windows[0][last] = (completed.windows[0][last] + 1) % 10;
            let mut changed = EvidenceRecorder::new(
                "foreign-labels",
                [
                    CALIBRATION_CAPTURE_STARTED_REGISTRATION,
                    CALIBRATION_CAPTURE_COMPLETED_REGISTRATION,
                ],
            )
            .unwrap();
            changed
                .emit("capture", None, 0, capture_view.started())
                .unwrap();
            changed.emit("capture", None, 1, &completed).unwrap();
            std::fs::write(&path, changed.to_jsonl().unwrap()).unwrap();
        }
        f.rejected(&a);
    }
    let package = tritium_format::read_trit_package(&f.b).unwrap();
    let manifest = tritium_format::trit_salience::decode_original_captured_salience_stage_manifest(
        package.manifest(),
        256_000,
    )
    .unwrap();
    let old =
        tritium_format::trit_salience::encode_salience_stage_manifest(&manifest.lineage, 256_000)
            .unwrap();
    let blobs: Vec<_> = package
        .blobs()
        .iter()
        .map(|b| tritium_format::TritBlob {
            kind: b.kind.clone(),
            bytes: package.blob(b).unwrap(),
        })
        .collect();
    let path = f.dir.0.join("stage.trit");
    std::fs::write(
        &path,
        tritium_format::write_trit_package(&old, &blobs).unwrap(),
    )
    .unwrap();
    f.rejected(&args);
    let mut changed = f.b.clone();
    let last = changed.len() - 1;
    changed[last] ^= 1;
    std::fs::write(&path, changed).unwrap();
    f.rejected(&args);
    std::fs::write(path, &f.b).unwrap();
    let binding = &f
        .plan
        .tensors
        .iter()
        .find(|t| matches!(t.action, CalibratedQuantAction::CalibratedFit { .. }))
        .unwrap()
        .source
        .name;
    let path = f.dir.0.join(format!("{binding}.trit"));
    std::fs::write(path, b"foreign cache").unwrap();
    f.rejected(&args);
}
