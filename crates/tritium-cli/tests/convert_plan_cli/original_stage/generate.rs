//! Cold joined command loading/generation compared with the actual shared runner.
use super::*;
use tritium_nn::Tokenizer;

fn arguments(f: &Fixture, text: Option<&str>, max_new: usize, eos: u32) -> Vec<OsString> {
    std::fs::write(f.dir.0.join("model.trit"), &f.expected).unwrap();
    std::fs::write(f.dir.0.join("fit.jsonl"), &f.fit).unwrap();
    std::fs::write(f.dir.0.join("tokens.json"), b"[1,2]").unwrap();
    let mut args = vec!["generate".into()];
    for (flag, file) in [
        ("--model", "model.trit"),
        ("--original-captured-stage", "stage.trit"),
        ("--transform-evidence", "transform.jsonl"),
        ("--original-salience-evidence", "original.jsonl"),
        ("--capture-evidence", "capture.jsonl"),
        ("--conversion-evidence", "fit.jsonl"),
        ("--stage-limits", "stage-limits.json"),
        ("--conversion-limits", "conversion-limits.json"),
    ] {
        args.push(flag.into());
        args.push(f.dir.0.join(file).into());
    }
    args.extend([
        "--level".into(),
        "standard".into(),
        "--max-payload-bytes".into(),
        "100000".into(),
        "--max-new".into(),
        max_new.to_string().into(),
        "--eos".into(),
        eos.to_string().into(),
    ]);
    if let Some(text) = text {
        args.extend(["--prompt".into(), text.into()]);
    } else {
        args.extend(["--tokens".into(), f.dir.0.join("tokens.json").into()]);
    }
    args
}
fn report(result: &Output) -> Vec<u32> {
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    String::from_utf8_lossy(&result.stdout)
        .lines()
        .find_map(|line| serde_json::from_str(line).ok())
        .expect("generated token report")
}
fn reject(args: &[OsString]) {
    let result = run(args);
    assert!(
        !result.status.success(),
        "unexpected success: {}",
        String::from_utf8_lossy(&result.stdout)
    );
    assert!(
        result.stdout.is_empty(),
        "failed invocation emitted generated output"
    );
    assert!(
        !String::from_utf8_lossy(&result.stderr).contains("panicked"),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}
fn runner(f: &Fixture, native_only: bool) -> Result<tritium_nn::ModelRunner, tritium_nn::NnError> {
    let stage = StagedSalienceSource::open_original_captured(
        &f.b,
        &f.transform,
        &f.original,
        stage_limits(),
    )
    .unwrap();
    let fit = verify_jsonl(&f.fit).unwrap();
    let model = JoinedSalienceModel::open_original_captured(
        &f.expected,
        &stage,
        &f.capture,
        &fit,
        conversion_limits(),
    )
    .unwrap();
    let budget = tritium_runtime::TensorPayloadBudget::new(100_000);
    let mut uploads = EvidenceRecorder::new(
        "joined-library",
        [TENSOR_UPLOADED_REGISTRATION, MODEL_LOADED_REGISTRATION],
    )
    .unwrap();
    tritium_nn::ModelRunner::from_source_admitted(
        tritium_nn::ModelSource::JoinedSalience {
            model: &model,
            level: "standard",
        },
        Box::new(tritium_cpu::CpuBackend::new()),
        TensorUploadPolicy {
            native_only,
            max_payload_bytes: Some(100_000),
        },
        &budget,
        tritium_runtime::TensorUploadContext {
            recorder: &mut uploads,
            span: "load",
            parent: None,
            logical_time: 0,
        },
    )
}

#[test]
fn joined_generate_matches_library_and_keeps_ids_tokenizer_free_in_all_precisions() {
    for precision in [
        ScalePrecision::F32,
        ScalePrecision::F16,
        ScalePrecision::Bf16,
    ] {
        let f = Fixture::new(precision);
        let args = arguments(&f, None, 2, u32::MAX);
        let expected = runner(&f, false)
            .unwrap()
            .generate(&[1, 2], 2, u32::MAX)
            .unwrap();
        assert_eq!(expected.len(), 2);
        assert_eq!(report(&run(&args)), expected);
        let zero = arguments(&f, None, 0, u32::MAX);
        assert!(report(&run(&zero)).is_empty());
        // This fixture's tokenizer is deliberately {}, with no config. IDs must
        // work without parsing it; text must reject rather than fabricate one.
        reject(&arguments(&f, Some("hello partner"), 2, 3));
        let mut native = args;
        native.push("--native-only".into());
        match runner(&f, true) {
            Ok(mut runner) => assert_eq!(
                report(&run(&native)),
                runner.generate(&[1, 2], 2, u32::MAX).unwrap()
            ),
            Err(_) => reject(&native),
        }
    }
}

#[test]
fn joined_generate_text_uses_real_admitted_tokenizer_and_exact_encoded_ids() {
    let f = Fixture::with_tokenizer();
    let stage = StagedSalienceSource::open_original_captured(
        &f.b,
        &f.transform,
        &f.original,
        stage_limits(),
    )
    .unwrap();
    let fit = verify_jsonl(&f.fit).unwrap();
    let model = JoinedSalienceModel::open_original_captured(
        &f.expected,
        &stage,
        &f.capture,
        &fit,
        conversion_limits(),
    )
    .unwrap();
    let budget = tritium_runtime::TensorPayloadBudget::new(100_000);
    let mut uploads = EvidenceRecorder::new(
        "joined-text-library",
        [TENSOR_UPLOADED_REGISTRATION, MODEL_LOADED_REGISTRATION],
    )
    .unwrap();
    let mut loaded = tritium_nn::load(
        tritium_nn::ModelSource::JoinedSalience {
            model: &model,
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
    let ids = loaded.tokenizer.encode("hello partner").unwrap();
    assert_eq!(ids, [1, 2]);
    let expected = loaded.runner.generate(&ids, 3, 3).unwrap();
    let result = run(&arguments(&f, Some("hello partner"), 3, 3));
    assert_eq!(report(&result), expected);
    assert_eq!(
        String::from_utf8_lossy(&result.stdout)
            .lines()
            .next()
            .unwrap(),
        loaded.tokenizer.decode(&expected).unwrap()
    );
    assert_eq!(report(&run(&arguments(&f, None, 3, 3))), expected);
    reject(&arguments(&f, Some(""), 3, 3));
    if let Some(directory) = std::env::var_os("TRITIUM_JOINED_GENERATION_OUTPUT") {
        let directory = PathBuf::from(directory);
        for entry in std::fs::read_dir(&f.dir.0).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_file() {
                std::fs::copy(entry.path(), directory.join(entry.file_name())).unwrap();
            }
        }
        std::fs::write(
            directory.join("generated-ids.json"),
            serde_json::to_vec(&expected).unwrap(),
        )
        .unwrap();
        std::fs::write(directory.join("generated-report.txt"), &result.stdout).unwrap();
    }
}

#[test]
fn joined_generate_rejects_partial_modes_files_levels_and_all_admission_caps() {
    let f = Fixture::new(ScalePrecision::F32);
    let args = arguments(&f, None, 2, u32::MAX);
    for flag in [
        "--original-captured-stage",
        "--transform-evidence",
        "--original-salience-evidence",
        "--capture-evidence",
        "--conversion-evidence",
        "--stage-limits",
        "--conversion-limits",
        "--level",
        "--max-payload-bytes",
        "--eos",
    ] {
        let mut a = args.clone();
        let i = a.iter().position(|v| v == flag).unwrap();
        a.drain(i..i + 2);
        reject(&a);
    }
    for (flag, value) in [
        ("--max-payload-bytes", "0"),
        ("--max-payload-bytes", "1"),
        ("--level", "missing"),
    ] {
        let mut a = args.clone();
        replace(&mut a, flag, value);
        reject(&a);
    }
    for (flag, value) in [
        ("--max-limits-bytes", "0"),
        ("--max-limits-bytes", "1"),
        ("--max-evidence-bytes", "0"),
        ("--max-evidence-bytes", "1"),
        ("--greedy", "false"),
    ] {
        let mut a = args.clone();
        a.extend([flag.into(), value.into()]);
        reject(&a);
    }
    let first_three: u64 = ["transform.jsonl", "original.jsonl", "capture.jsonl"]
        .into_iter()
        .map(|name| std::fs::metadata(f.dir.0.join(name)).unwrap().len())
        .sum();
    let mut a = args.clone();
    a.extend([
        "--max-evidence-bytes".into(),
        first_three.to_string().into(),
    ]);
    reject(&a);
    for flag in [
        "--model",
        "--original-captured-stage",
        "--conversion-evidence",
        "--stage-limits",
        "--conversion-limits",
    ] {
        for path in [f.dir.0.join("missing"), f.dir.0.clone()] {
            let mut a = args.clone();
            replace(&mut a, flag, path);
            reject(&a);
        }
    }
    for limits in [
        SalienceConversionLimits {
            max_package_bytes: 1,
            ..conversion_limits()
        },
        SalienceConversionLimits {
            max_manifest_bytes: 1,
            ..conversion_limits()
        },
    ] {
        std::fs::write(
            f.dir.0.join("conversion-limits.json"),
            serde_json::to_vec(&limits).unwrap(),
        )
        .unwrap();
        reject(&args);
    }
    std::fs::write(
        f.dir.0.join("conversion-limits.json"),
        serde_json::to_vec(&conversion_limits()).unwrap(),
    )
    .unwrap();
    std::fs::write(
        f.dir.0.join("stage-limits.json"),
        serde_json::to_vec(&SalienceStageLimits {
            max_package_bytes: 1,
            ..stage_limits()
        })
        .unwrap(),
    )
    .unwrap();
    reject(&args);
}

#[test]
fn joined_generate_rejects_both_rerooted_captures_and_foreign_conversion_log() {
    let f = Fixture::new(ScalePrecision::F32);
    let args = arguments(&f, None, 2, u32::MAX);
    let a_view = OriginalSalienceCaptureView::fold(&f.original)
        .unwrap()
        .remove(0);
    let b_view = CalibrationCaptureView::fold(&f.capture).unwrap().remove(0);
    for complete in [false, true] {
        let mut a = EvidenceRecorder::new(
            "foreign-a",
            [
                ORIGINAL_SALIENCE_CAPTURE_STARTED_REGISTRATION,
                ORIGINAL_SALIENCE_CAPTURE_COMPLETED_REGISTRATION,
            ],
        )
        .unwrap();
        a.emit("capture", None, 0, a_view.started()).unwrap();
        if complete {
            a.emit("capture", None, 1, a_view.completed()).unwrap();
        }
        let path = f.dir.0.join("foreign.jsonl");
        std::fs::write(&path, a.to_jsonl().unwrap()).unwrap();
        let mut command = args.clone();
        replace(&mut command, "--original-salience-evidence", path.clone());
        reject(&command);
        let mut b = EvidenceRecorder::new(
            "foreign-b",
            [
                CALIBRATION_CAPTURE_STARTED_REGISTRATION,
                CALIBRATION_CAPTURE_COMPLETED_REGISTRATION,
            ],
        )
        .unwrap();
        b.emit("capture", None, 0, b_view.started()).unwrap();
        if complete {
            b.emit("capture", None, 1, b_view.completed()).unwrap();
        }
        std::fs::write(&path, b.to_jsonl().unwrap()).unwrap();
        replace(
            &mut command,
            "--original-salience-evidence",
            f.dir.0.join("original.jsonl"),
        );
        replace(&mut command, "--capture-evidence", path);
        reject(&command);
    }
    let log = verify_jsonl(&f.fit).unwrap();
    // Re-emit the same actual typed payloads under a complete foreign run;
    // changing only custody still cannot open a previously frozen C.
    let mut foreign = EvidenceRecorder::new(
        "foreign-fit",
        [
            CALIBRATED_QUANTIZATION_STARTED_REGISTRATION,
            CALIBRATED_LEVELS_FITTED_REGISTRATION,
            CALIBRATED_QUANTIZATION_COMPLETED_REGISTRATION,
        ],
    )
    .unwrap();
    for event in &log.events {
        if event.schema == CalibratedQuantizationStarted::SCHEMA_ID {
            foreign
                .emit(
                    &event.span,
                    event.parent.as_deref(),
                    event.t,
                    &serde_json::from_value::<CalibratedQuantizationStarted>(event.payload.clone())
                        .unwrap(),
                )
                .unwrap();
        } else if event.schema == CalibratedLevelsFitted::SCHEMA_ID {
            foreign
                .emit(
                    &event.span,
                    event.parent.as_deref(),
                    event.t,
                    &serde_json::from_value::<CalibratedLevelsFitted>(event.payload.clone())
                        .unwrap(),
                )
                .unwrap();
        } else {
            foreign
                .emit(
                    &event.span,
                    event.parent.as_deref(),
                    event.t,
                    &serde_json::from_value::<CalibratedQuantizationCompleted>(
                        event.payload.clone(),
                    )
                    .unwrap(),
                )
                .unwrap();
        }
    }
    std::fs::write(f.dir.0.join("fit.jsonl"), foreign.to_jsonl().unwrap()).unwrap();
    reject(&args);
}

#[test]
fn joined_generate_rejects_changed_payloads_and_ordinary_unwrapped_profiles() {
    let f = Fixture::new(ScalePrecision::F32);
    let args = arguments(&f, None, 2, u32::MAX);
    for (name, original) in [("model.trit", &f.expected), ("stage.trit", &f.b)] {
        let package = tritium_format::read_trit_package(original).unwrap();
        let mut changed = original.clone();
        changed[usize::try_from(package.blobs()[0].offset).unwrap()] ^= 1;
        std::fs::write(f.dir.0.join(name), changed).unwrap();
        reject(&args);
        std::fs::write(f.dir.0.join(name), original).unwrap();
    }
    let package = tritium_format::read_trit_package(&f.expected).unwrap();
    let manifest = tritium_format::trit_salience_conversion::decode_original_captured_salience_conversion_manifest(package.manifest(),256_000).unwrap();
    let ordinary =
        tritium_format::trit_model::encode_model_manifest(&manifest.lineage.lineage.model).unwrap();
    let blobs: Vec<_> = package
        .blobs()
        .iter()
        .map(|b| tritium_format::TritBlob {
            kind: b.kind.clone(),
            bytes: package.blob(b).unwrap(),
        })
        .collect();
    std::fs::write(
        f.dir.0.join("model.trit"),
        tritium_format::write_trit_package(&ordinary, &blobs).unwrap(),
    )
    .unwrap();
    reject(&args);
}
