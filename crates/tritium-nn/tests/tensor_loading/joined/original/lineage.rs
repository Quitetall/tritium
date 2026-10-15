//! Actual A salience -> mandatory B -> B VJPs -> fitted C -> existing runtime.
//! Toy software and numerical replay, never real-model/physical release qualification.
use super::super::fitting::{calibrated_plan, calibrated_recorder};
use super::super::staged::{capture_limits as b_limits, recorded::recorder as b_recorder};
use super::captured::{capture_limits, recorder, rehashed};
use super::*;
use tritium_evidence::{CalibrationCaptureView, OriginalSalienceCaptureView};
use tritium_format::trit_salience::{
    decode_original_captured_salience_stage_manifest, decode_salience_stage_manifest,
    encode_original_captured_salience_stage_manifest,
};
use tritium_format::trit_salience_conversion::{
    decode_captured_salience_conversion_manifest,
    decode_original_captured_salience_conversion_manifest, decode_salience_conversion_manifest,
    encode_original_captured_salience_conversion_manifest,
};
use tritium_format::{TritBlob, read_trit_package, write_trit_package};
use tritium_nn::calibrate::{
    RecordedOriginalSalienceCapture, StagedCalibrationModel, replay_staged_capture,
    verify_original_captured_stage,
};
use tritium_quantize::engine::{
    DirectoryTensorSink, InMemoryCalibrationSource, PublicationWarning,
};

fn transform_recorder() -> EvidenceRecorder {
    EvidenceRecorder::new(
        "nn-joined-transform",
        [
            SALIENCE_TRANSFORM_STARTED_REGISTRATION,
            SALIENCE_TRANSFORM_COMPLETED_REGISTRATION,
        ],
    )
    .unwrap()
}

fn transform_plan(capture: &RecordedOriginalSalienceCapture) -> SalienceTransformPlan {
    let view = OriginalSalienceCaptureView::fold(capture.evidence().log())
        .unwrap()
        .remove(0);
    let original = &view.started().plan;
    SalienceTransformPlan {
        architecture: original.architecture.clone(),
        original_source: original.source,
        tensors: original.tensors.iter().map(|b| b.source.clone()).collect(),
        assets: original.assets.iter().map(|a| a.name.clone()).collect(),
        operations: original
            .pairs
            .iter()
            .map(|pair| SalienceTransformOperation {
                name: pair.name.clone(),
                projections: pair.projections.clone(),
                inverse: pair.inverse.clone(),
                inverse_axis: pair.inverse_axis,
                moments: capture.statistics()[&pair.name].binding().unwrap(),
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
    }
}

fn repack(bytes: &[u8], manifest: &[u8]) -> Vec<u8> {
    let package = read_trit_package(bytes).unwrap();
    let blobs: Vec<_> = package
        .blobs()
        .iter()
        .map(|b| TritBlob {
            kind: b.kind.clone(),
            bytes: package.blob(b).unwrap(),
        })
        .collect();
    write_trit_package(manifest, &blobs).unwrap()
}

#[test]
fn actual_original_captured_chain_cold_replays_and_decodes_in_all_precisions() {
    for precision in [
        ScalePrecision::F32,
        ScalePrecision::F16,
        ScalePrecision::Bf16,
    ] {
        for architecture in ["llama", "qwen3"] {
            let hf = HfFixture::new(
                architecture,
                true,
                if architecture == "qwen3" { 6 } else { 4 },
                None,
            );
            let mut input = TritFixture::new(&hf, None);
            if precision != ScalePrecision::F32 {
                input.dense_precision(precision);
            }
            let mut raw = raw_fixture_source(&input);
            let request = request(architecture, &raw);
            let mut observed = ObservedWeightSource::new(&request, &mut raw).unwrap();
            let model = SourceCalibrationModel::new(&mut observed, limits()).unwrap();
            let capture = model
                .capture_salience_recorded(
                    &mut observed,
                    &[&[1, 3, 2], &[4]],
                    capture_limits(),
                    recorder(),
                )
                .unwrap();
            drop(model);
            let plan = transform_plan(&capture);
            // Same actual payloads/old profile remain usable without an ancestry upgrade.
            let legacy = SourceTransform::run_recorded(
                &plan,
                &mut observed,
                capture.statistics(),
                transform_recorder(),
            )
            .unwrap();
            let old_b = legacy
                .encode_stage_package(capture.statistics(), stage_limits())
                .unwrap();
            let old_transform = verify_jsonl(legacy.evidence.jsonl()).unwrap();
            let recorded = capture
                .transform_recorded(&plan, &mut observed, transform_recorder())
                .unwrap();
            assert_eq!(
                recorded.transform_evidence().jsonl(),
                legacy.evidence.jsonl()
            );
            drop(legacy);
            drop(observed);
            drop(raw);
            let b = recorded.encode_stage_package(stage_limits()).unwrap();
            let mut written = Vec::new();
            let package_id = recorded
                .write_stage_package_to(stage_limits(), &mut written)
                .unwrap();
            assert_eq!(written, b);
            let destination = hf.dir.join("original-captured-stage");
            let mut sink = DirectoryTensorSink::new(&destination);
            let published = recorded.publish_stage(stage_limits(), &mut sink).unwrap();
            assert_eq!(published.package_id, package_id);
            // As in the shared sink's existing publication tests, directory
            // sync is an observed OS/filesystem capability, not guaranteed by
            // a successful no-clobber commit. Other warnings still reject.
            let expected_warnings = if std::fs::File::open(&destination)
                .and_then(|directory| directory.sync_all())
                .is_err()
            {
                vec![PublicationWarning::DirectorySync]
            } else {
                Vec::new()
            };
            assert_eq!(published.warnings, expected_warnings);
            assert!(!destination.join(".pending.trit").exists());
            assert_eq!(std::fs::read(destination.join("model.trit")).unwrap(), b);
            assert_eq!(
                std::fs::read_to_string(destination.join("evidence.jsonl")).unwrap(),
                recorded.transform_evidence().jsonl()
            );
            assert!(recorded.publish_stage(stage_limits(), &mut sink).is_err());
            assert_eq!(std::fs::read(destination.join("model.trit")).unwrap(), b);
            let original = verify_jsonl(recorded.original_capture_evidence().jsonl()).unwrap();
            let transform = verify_jsonl(recorded.transform_evidence().jsonl()).unwrap();
            let original_jsonl = recorded.original_capture_evidence().jsonl().to_owned();
            let transform_jsonl = recorded.transform_evidence().jsonl().to_owned();
            drop(recorded);
            let original_view = OriginalSalienceCaptureView::fold(&original)
                .unwrap()
                .remove(0);
            let mut old_stage =
                StagedSalienceSource::open(&old_b, &old_transform, stage_limits()).unwrap();
            let mut stage = StagedSalienceSource::open_original_captured(
                &b,
                &transform,
                &original,
                stage_limits(),
            )
            .unwrap();
            assert_eq!(stage.source_id(), old_stage.source_id());
            assert_ne!(stage.package_id(), old_stage.package_id());
            assert_eq!(stage.package_id(), package_id);
            assert_eq!(
                stage.original_capture_reference(),
                Some(original_view.reference())
            );
            assert!(StagedSalienceSource::open(&b, &transform, stage_limits()).is_err());
            assert!(
                StagedSalienceSource::open_original_captured(
                    &old_b,
                    &old_transform,
                    &original,
                    stage_limits()
                )
                .is_err()
            );
            let new_package = read_trit_package(&b).unwrap();
            let old_package = read_trit_package(&old_b).unwrap();
            assert_eq!(new_package.blobs(), old_package.blobs());
            for blob in new_package.blobs() {
                assert_eq!(
                    new_package.blob(blob).unwrap(),
                    old_package.blob(blob).unwrap()
                );
            }
            let carrier =
                decode_original_captured_salience_stage_manifest(new_package.manifest(), 256_000)
                    .unwrap();
            assert_eq!(
                encode_original_captured_salience_stage_manifest(&carrier, 256_000).unwrap(),
                new_package.manifest()
            );
            assert!(decode_salience_stage_manifest(new_package.manifest(), 256_000).is_err());
            let mut actual_a = raw_fixture_source(&input);
            verify_original_captured_stage(&stage, &original, &mut actual_a, capture_limits())
                .unwrap();
            drop(actual_a);
            let model = StagedCalibrationModel::new(&mut stage, limits()).unwrap();
            let captured = model
                .capture_next_token_recorded(
                    &mut stage,
                    &[&[1, 4, 2], &[3, 7]],
                    b_limits(),
                    b_recorder(),
                )
                .unwrap();
            drop(model);
            let b_capture = verify_jsonl(captured.evidence.jsonl()).unwrap();
            let b_view = CalibrationCaptureView::fold(&b_capture).unwrap().remove(0);
            assert_eq!(b_view.started().plan.stage_package, package_id);
            let caches: Vec<_> =
                replay_staged_capture(&b_capture, &mut stage, b_view.started().plan.limits)
                    .unwrap()
                    .into_iter()
                    .filter(|c| c.binding().request.tensor.name != "model.embed_tokens.weight")
                    .collect();
            let plan = calibrated_plan(&stage, &caches);
            let mut provider = InMemoryCalibrationSource::new(caches).unwrap();
            let mut sink = MemoryTensorSink::default();
            let mut incomplete = b_recorder();
            incomplete
                .emit("capture", None, 0, b_view.started())
                .unwrap();
            let incomplete = incomplete.seal().unwrap();
            assert!(
                Engine::run_original_captured_salience(
                    &plan,
                    &mut stage,
                    incomplete.log(),
                    &mut provider,
                    &mut sink,
                    calibrated_recorder(),
                    conversion_limits(),
                )
                .is_err()
            );
            assert!(sink.artifact().is_none());
            assert!(
                Engine::run_original_captured_salience(
                    &plan,
                    &mut old_stage,
                    &b_capture,
                    &mut provider,
                    &mut sink,
                    calibrated_recorder(),
                    conversion_limits(),
                )
                .is_err()
            );
            assert!(sink.artifact().is_none());
            assert!(
                Engine::run_captured_salience(
                    &plan,
                    &mut stage,
                    &b_capture,
                    &mut provider,
                    &mut sink,
                    calibrated_recorder(),
                    conversion_limits()
                )
                .is_err()
            );
            assert!(sink.artifact().is_none());
            let result = Engine::run_original_captured_salience(
                &plan,
                &mut stage,
                &b_capture,
                &mut provider,
                &mut sink,
                calibrated_recorder(),
                conversion_limits(),
            )
            .unwrap();
            let (bytes, evidence) = sink.artifact().unwrap();
            let c = bytes.to_vec();
            let fit = verify_jsonl(evidence.jsonl()).unwrap();
            let fit_jsonl = evidence.jsonl().to_owned();
            if architecture == "qwen3"
                && precision == ScalePrecision::Bf16
                && let Some(directory) = std::env::var_os("TRITIUM_ORIGINAL_CAPTURED_CHAIN_OUTPUT")
            {
                let directory = std::path::PathBuf::from(directory);
                std::fs::write(directory.join("stage.trit"), &b).unwrap();
                std::fs::write(directory.join("model.trit"), &c).unwrap();
                std::fs::write(directory.join("original.jsonl"), &original_jsonl).unwrap();
                std::fs::write(directory.join("transform.jsonl"), &transform_jsonl).unwrap();
                std::fs::write(directory.join("capture.jsonl"), captured.evidence.jsonl()).unwrap();
                std::fs::write(directory.join("fit.jsonl"), evidence.jsonl()).unwrap();
            }
            drop(sink);
            drop(provider);
            drop(captured);
            drop(stage);
            let mut stage = StagedSalienceSource::open_original_captured(
                &b,
                &transform,
                &original,
                stage_limits(),
            )
            .unwrap();
            // Reconstruct all actual B caches and rerun the one fitter/writer cold.
            // A structural C opener cannot stand in for this numerical replay.
            let cold_caches =
                replay_staged_capture(&b_capture, &mut stage, b_view.started().plan.limits)
                    .unwrap()
                    .into_iter()
                    .filter(|cache| {
                        cache.binding().request.tensor.name != "model.embed_tokens.weight"
                    })
                    .collect::<Vec<_>>();
            let cold_plan = calibrated_plan(&stage, &cold_caches);
            assert_eq!(cold_plan, plan);
            let mut cold_provider = InMemoryCalibrationSource::new(cold_caches).unwrap();
            let mut cold_sink = MemoryTensorSink::default();
            let cold_result = Engine::run_original_captured_salience(
                &cold_plan,
                &mut stage,
                &b_capture,
                &mut cold_provider,
                &mut cold_sink,
                calibrated_recorder(),
                conversion_limits(),
            )
            .unwrap();
            assert_eq!(cold_result.package_id, result.package_id);
            assert_eq!(cold_result.source_id, result.source_id);
            let (cold_bytes, cold_evidence) = cold_sink.artifact().unwrap();
            assert_eq!(cold_bytes, c);
            assert_eq!(cold_evidence.jsonl(), fit_jsonl);
            drop(cold_provider);
            drop(cold_sink);
            let joined = JoinedSalienceModel::open_original_captured(
                &c,
                &stage,
                &b_capture,
                &fit,
                conversion_limits(),
            )
            .unwrap();
            assert_eq!(
                joined.original_capture_reference(),
                Some(original_view.reference())
            );
            assert_eq!(joined.capture_reference(), Some(b_view.reference()));
            assert_eq!(joined.manifest().derived_source, result.source_id);
            assert!(
                joined.manifest().model.levels[0]
                    .tensors
                    .iter()
                    .any(|t| matches!(t.storage, TensorStorage::Additive { .. }))
            );
            assert!(JoinedSalienceModel::open(&c, &stage, &fit, conversion_limits()).is_err());
            assert!(
                JoinedSalienceModel::open_captured(
                    &c,
                    &stage,
                    &b_capture,
                    &fit,
                    conversion_limits()
                )
                .is_err()
            );
            assert!(
                JoinedSalienceModel::open_original_captured(
                    &c,
                    &old_stage,
                    &b_capture,
                    &fit,
                    conversion_limits()
                )
                .is_err()
            );
            assert!(tritium_format::trit_model::read_trit_model(&b).is_err());
            assert!(tritium_format::trit_model::read_trit_model(&c).is_err());
            let package = read_trit_package(&c).unwrap();
            let carrier =
                decode_original_captured_salience_conversion_manifest(package.manifest(), 256_000)
                    .unwrap();
            assert_eq!(&carrier.original_capture, original_view.reference());
            let mut unknown = serde_json::to_value(&carrier).unwrap();
            unknown["optional_original_capture"] = serde_json::json!(true);
            assert!(serde_json::from_value::<tritium_schema::OriginalCapturedSalienceConversionManifest>(unknown).is_err());
            for which in 0..4 {
                let mut bad = carrier.clone();
                match which {
                    0 => bad.original_capture.root = [0; 32],
                    1 => bad.lineage.capture.root = [0; 32],
                    2 => bad.lineage.lineage.transform.root = [0; 32],
                    _ => bad.lineage.lineage.schema.minor = 1,
                }
                assert!(
                    encode_original_captured_salience_conversion_manifest(&bad, 256_000).is_err()
                );
            }
            assert_eq!(&carrier.lineage.capture, b_view.reference());
            assert_eq!(
                encode_original_captured_salience_conversion_manifest(&carrier, 256_000).unwrap(),
                package.manifest()
            );
            assert!(decode_salience_conversion_manifest(package.manifest(), 256_000).is_err());
            assert!(
                decode_captured_salience_conversion_manifest(package.manifest(), 256_000).is_err()
            );
            for original_root in [true, false] {
                let mut bad = carrier.clone();
                if original_root {
                    bad.original_capture.root = [17; 32];
                } else {
                    bad.lineage.capture.root = [18; 32];
                }
                let manifest =
                    encode_original_captured_salience_conversion_manifest(&bad, 256_000).unwrap();
                let changed = repack(&c, &manifest);
                assert!(read_trit_package(&changed).is_ok());
                assert!(
                    JoinedSalienceModel::open_original_captured(
                        &changed,
                        &stage,
                        &b_capture,
                        &fit,
                        conversion_limits()
                    )
                    .is_err()
                );
            }
            assert!(
                JoinedSalienceModel::open_original_captured(
                    &c,
                    &stage,
                    &b_capture,
                    &fit,
                    SalienceConversionLimits {
                        max_package_bytes: c.len() as u64 - 1,
                        ..conversion_limits()
                    }
                )
                .is_err()
            );
            // The admitted proof-bearing C can outlive raw A, B and both private captures.
            drop(stage);
            drop(old_stage);
            drop(b_capture);
            drop(original);
            drop(transform);
            drop(fit);
            drop(b_view);
            drop(original_view);
            drop(input);
            for reference in [false, true] {
                let backend: Box<dyn TernaryBackend> = if reference {
                    Box::new(tritium_testkit::ReferenceBackend::new())
                } else {
                    Box::new(tritium_cpu::CpuBackend::new())
                };
                let mut oracle = oracle_fixture(&joined, &c).dense_oracle();
                let budget = TensorPayloadBudget::new(100_000);
                let mut uploads = EvidenceRecorder::new(
                    "original-captured-runtime",
                    [TENSOR_UPLOADED_REGISTRATION],
                )
                .unwrap();
                let mut runner = ModelRunner::from_source_admitted(
                    tritium_nn::ModelSource::JoinedSalience {
                        model: &joined,
                        level: "standard",
                    },
                    backend,
                    TensorUploadPolicy::default(),
                    &budget,
                    TensorUploadContext {
                        recorder: &mut uploads,
                        span: "load",
                        parent: None,
                        logical_time: 0,
                    },
                )
                .unwrap();
                for (tokens, positions) in [(&[1, 2][..], &[0, 1][..]), (&[1][..], &[2][..])] {
                    let actual = runner.forward(tokens, positions).unwrap();
                    let expected = oracle.forward(tokens, positions).unwrap();
                    assert!(
                        actual
                            .iter()
                            .zip(expected)
                            .all(|(a, b)| (a - b).abs() < 2e-5)
                    );
                }
                drop(runner);
                assert_eq!(budget.used_bytes(), 0);
            }
        }
    }
}

#[test]
fn original_captured_stage_rejects_substitutions_caps_and_publication_faults() {
    let hf = HfFixture::new("qwen3", true, 6, None);
    let input = TritFixture::new(&hf, None);
    let mut raw = raw_fixture_source(&input);
    let request = request("qwen3", &raw);
    let mut observed = ObservedWeightSource::new(&request, &mut raw).unwrap();
    let model = SourceCalibrationModel::new(&mut observed, limits()).unwrap();
    let capture = model
        .capture_salience_recorded(
            &mut observed,
            &[&[1, 3, 2], &[4]],
            capture_limits(),
            recorder(),
        )
        .unwrap();
    drop(model);
    let plan = transform_plan(&capture);
    let view = OriginalSalienceCaptureView::fold(capture.evidence().log())
        .unwrap()
        .remove(0);
    for which in 0..5 {
        let mut bad = plan.clone();
        match which {
            0 => bad.operations[0].projections.reverse(),
            1 => bad.operations[0].moments.rows += 1,
            2 => bad.operations.swap(0, 1),
            3 => {
                bad.assets.pop();
            }
            _ => bad.original_source = tritium_schema::OriginalSourceId::from_bytes([9; 32]),
        }
        assert!(
            SourceTransform::run_original_captured_recorded(
                &bad,
                &mut observed,
                capture.statistics(),
                capture.evidence().log(),
                transform_recorder()
            )
            .is_err()
        );
    }
    let mut unfinished = recorder();
    unfinished
        .emit("salience", None, 0, view.started())
        .unwrap();
    let unfinished = unfinished.seal().unwrap();
    assert!(
        SourceTransform::run_original_captured_recorded(
            &plan,
            &mut observed,
            capture.statistics(),
            unfinished.log(),
            transform_recorder()
        )
        .is_err()
    );
    let owned = capture
        .transform_recorded(&plan, &mut observed, transform_recorder())
        .unwrap();
    let b = owned.encode_stage_package(stage_limits()).unwrap();
    let original = verify_jsonl(owned.original_capture_evidence().jsonl()).unwrap();
    let transform = verify_jsonl(owned.transform_evidence().jsonl()).unwrap();
    let package = read_trit_package(&b).unwrap();
    let carrier =
        decode_original_captured_salience_stage_manifest(package.manifest(), 256_000).unwrap();
    let mut unknown = serde_json::to_value(&carrier).unwrap();
    unknown["optional_original_capture"] = serde_json::json!(true);
    assert!(
        serde_json::from_value::<tritium_schema::OriginalCapturedSalienceSourceStageManifest>(
            unknown
        )
        .is_err()
    );
    for which in 0..3 {
        let mut bad = carrier.clone();
        match which {
            0 => bad.original_capture.root = [0; 32],
            1 => bad.lineage.transform.root = [0; 32],
            _ => bad.lineage.schema.minor = 1,
        }
        assert!(encode_original_captured_salience_stage_manifest(&bad, 256_000).is_err());
    }
    let mut bad = carrier.clone();
    bad.original_capture.root = [22; 32];
    let manifest = encode_original_captured_salience_stage_manifest(&bad, 256_000).unwrap();
    let changed = repack(&b, &manifest);
    assert!(
        StagedSalienceSource::open_original_captured(
            &changed,
            &transform,
            &original,
            stage_limits()
        )
        .is_err()
    );
    assert!(
        StagedSalienceSource::open_original_captured(
            &b,
            &transform,
            unfinished.log(),
            stage_limits()
        )
        .is_err()
    );
    let mut completed = view.completed().clone();
    completed.outputs[0].moments.rows += 1;
    // A valid chain is not allowed to replace mandatory ancestry or actual moments.
    let foreign = rehashed(view.started().clone(), completed);
    assert!(verify_jsonl(foreign.jsonl()).is_ok());
    assert!(
        StagedSalienceSource::open_original_captured(&b, &transform, foreign.log(), stage_limits())
            .is_err()
    );
    let mut corrupted = b.clone();
    *corrupted.last_mut().unwrap() ^= 1;
    assert!(
        StagedSalienceSource::open_original_captured(
            &corrupted,
            &transform,
            &original,
            stage_limits()
        )
        .is_err()
    );
    assert!(
        decode_original_captured_salience_stage_manifest(
            package.manifest(),
            package.manifest().len() - 1
        )
        .is_err()
    );
    let small = SalienceStageLimits {
        max_package_bytes: b.len() as u64 - 1,
        ..stage_limits()
    };
    let path = hf.dir.join("rejected-stage");
    assert!(
        owned
            .publish_stage(small, &mut DirectoryTensorSink::new(&path))
            .is_err()
    );
    assert!(!path.exists());
    assert!(
        StagedSalienceSource::open_original_captured(&b, &transform, &original, small).is_err()
    );
    struct BrokenWriter;
    impl std::io::Write for BrokenWriter {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("injected carrier write fault"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    assert!(
        owned
            .write_stage_package_to(stage_limits(), &mut BrokenWriter)
            .is_err()
    );
    let stage =
        StagedSalienceSource::open_original_captured(&b, &transform, &original, stage_limits())
            .unwrap();
    drop(observed);
    let mut tiny = capture_limits();
    tiny.max_total_tokens = 1;
    assert!(verify_original_captured_stage(&stage, &original, &mut raw, tiny).is_err());
    let mut foreign_a = raw_fixture_source(&input);
    foreign_a
        .tensors
        .get_mut("model.layers.0.mlp.up_proj.weight")
        .unwrap()
        .bytes[0] ^= 1;
    assert!(
        verify_original_captured_stage(&stage, &original, &mut foreign_a, capture_limits())
            .is_err()
    );
}

#[cfg(feature = "tokenizer")]
#[test]
fn original_captured_text_aliases_replay_but_cannot_replace_b_or_c_ancestry() {
    let hf = HfFixture::new("qwen3", true, 6, None);
    let mut input = TritFixture::new(&hf, None);
    let (tokenizer, config) = tokenizer_assets(false);
    input.asset("tokenizer.json", tokenizer);
    input.asset("tokenizer_config.json", config);
    let receiving = tritium_schema::OriginalSalienceCaptureLimits {
        max_text_bytes: 1024,
        max_tokenizer_config_bytes: 16_000,
        ..capture_limits()
    };
    let mut raw = raw_fixture_source(&input);
    let request = request("qwen3", &raw);
    let mut observed = ObservedWeightSource::new(&request, &mut raw).unwrap();
    let model = SourceCalibrationModel::new(&mut observed, limits()).unwrap();
    let actual = model
        .capture_salience_text_recorded(
            &mut observed,
            &["hello", "partner hello"],
            receiving,
            recorder(),
        )
        .unwrap();
    let alias = model
        .capture_salience_text_recorded(
            &mut observed,
            &["hello ", "partner hello"],
            receiving,
            recorder(),
        )
        .unwrap();
    drop(model);
    let plan = transform_plan(&actual);
    assert_eq!(plan, transform_plan(&alias));
    let actual = actual
        .transform_recorded(&plan, &mut observed, transform_recorder())
        .unwrap();
    let alias = alias
        .transform_recorded(&plan, &mut observed, transform_recorder())
        .unwrap();
    assert_eq!(
        actual.transform_evidence().jsonl(),
        alias.transform_evidence().jsonl()
    );
    let b = actual.encode_stage_package(stage_limits()).unwrap();
    let alias_b = alias.encode_stage_package(stage_limits()).unwrap();
    let original = verify_jsonl(actual.original_capture_evidence().jsonl()).unwrap();
    let alias_log = verify_jsonl(alias.original_capture_evidence().jsonl()).unwrap();
    let transform = verify_jsonl(actual.transform_evidence().jsonl()).unwrap();
    assert_ne!(original.root_digest, alias_log.root_digest);
    drop(actual);
    drop(alias);
    drop(observed);
    let mut stage =
        StagedSalienceSource::open_original_captured(&b, &transform, &original, stage_limits())
            .unwrap();
    let alias_stage = StagedSalienceSource::open_original_captured(
        &alias_b,
        &transform,
        &alias_log,
        stage_limits(),
    )
    .unwrap();
    assert_eq!(stage.source_id(), alias_stage.source_id());
    assert_ne!(stage.package_id(), alias_stage.package_id());
    verify_original_captured_stage(&stage, &original, &mut raw, receiving).unwrap();
    verify_original_captured_stage(&alias_stage, &alias_log, &mut raw, receiving).unwrap();
    assert!(
        StagedSalienceSource::open_original_captured(&b, &transform, &alias_log, stage_limits())
            .is_err()
    );
    assert!(verify_original_captured_stage(&stage, &alias_log, &mut raw, receiving).is_err());
    let captured = StagedCalibrationModel::new(&mut stage, limits())
        .unwrap()
        .capture_next_token_recorded(&mut stage, &[&[1, 4, 2], &[3, 7]], b_limits(), b_recorder())
        .unwrap();
    let view = CalibrationCaptureView::fold(captured.evidence.log())
        .unwrap()
        .remove(0);
    let caches = replay_staged_capture(
        captured.evidence.log(),
        &mut stage,
        view.started().plan.limits,
    )
    .unwrap()
    .into_iter()
    .filter(|c| c.binding().request.tensor.name != "model.embed_tokens.weight")
    .collect::<Vec<_>>();
    let plan = calibrated_plan(&stage, &caches);
    let mut provider = InMemoryCalibrationSource::new(caches).unwrap();
    let mut sink = MemoryTensorSink::default();
    Engine::run_original_captured_salience(
        &plan,
        &mut stage,
        captured.evidence.log(),
        &mut provider,
        &mut sink,
        calibrated_recorder(),
        conversion_limits(),
    )
    .unwrap();
    let (c, fit) = sink.artifact().unwrap();
    assert!(
        JoinedSalienceModel::open_original_captured(
            c,
            &stage,
            captured.evidence.log(),
            fit.log(),
            conversion_limits()
        )
        .is_ok()
    );
    assert!(
        JoinedSalienceModel::open_original_captured(
            c,
            &alias_stage,
            captured.evidence.log(),
            fit.log(),
            conversion_limits()
        )
        .is_err()
    );
}
