//! Actual original graph/reducer and cold source replay; toy software, not model qualification.
use super::*;
use tritium_evidence::{
    ORIGINAL_SALIENCE_CAPTURE_COMPLETED_REGISTRATION,
    ORIGINAL_SALIENCE_CAPTURE_STARTED_REGISTRATION, OriginalSalienceCaptureView, SealedEvidence,
    verify_jsonl,
};
use tritium_nn::calibrate::{
    Calib, calibrate, extract, fold, norm_tensors, replay_original_salience_capture, weight_names,
};
use tritium_schema::{
    BlobId, OriginalSalienceCaptureCompleted, OriginalSalienceCaptureInput,
    OriginalSalienceCaptureLimits, OriginalSalienceCaptureStarted,
};

pub(super) fn capture_limits() -> OriginalSalienceCaptureLimits {
    OriginalSalienceCaptureLimits {
        max_source_bytes: 64_000,
        max_asset_bytes: 16_000,
        max_config_bytes: limits().max_config_bytes,
        max_dense_bytes: limits().max_dense_bytes,
        max_window_tokens: 4,
        max_windows: 4,
        max_total_tokens: 8,
        max_moment_bytes: 512,
        max_text_bytes: 0,
        max_tokenizer_config_bytes: 0,
    }
}
pub(super) fn recorder() -> EvidenceRecorder {
    EvidenceRecorder::new(
        "actual-original-salience",
        [
            ORIGINAL_SALIENCE_CAPTURE_STARTED_REGISTRATION,
            ORIGINAL_SALIENCE_CAPTURE_COMPLETED_REGISTRATION,
        ],
    )
    .unwrap()
}
pub(super) fn rehashed(
    start: OriginalSalienceCaptureStarted,
    completed: OriginalSalienceCaptureCompleted,
) -> SealedEvidence {
    let mut r = recorder();
    r.emit("salience", None, 0, &start).unwrap();
    r.emit("salience", None, 1, &completed).unwrap();
    r.seal().unwrap()
}
fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn original_salience_capture_matches_legacy_statistics_fold_and_cold_replay_in_all_precisions() {
    let windows = [&[1, 3, 2][..], &[4][..]];
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
            let mut source = raw_fixture_source(&input);
            let request = request(architecture, &source);
            let mut observed = ObservedWeightSource::new(&request, &mut source).unwrap();
            let model = SourceCalibrationModel::new(&mut observed, limits()).unwrap();
            let actual = model
                .capture_salience_recorded(&mut observed, &windows, capture_limits(), recorder())
                .unwrap();
            let repeated = model
                .capture_salience_recorded(&mut observed, &windows, capture_limits(), recorder())
                .unwrap();
            assert_eq!(actual.evidence().jsonl(), repeated.evidence().jsonl());
            let view = OriginalSalienceCaptureView::fold(actual.evidence().log())
                .unwrap()
                .remove(0);
            assert_eq!(view.started().plan.source, model.source_id());
            assert_eq!(view.completed().windows, [vec![1, 3, 2], vec![4]]);
            assert_eq!(view.completed().outputs.len(), model.arch().n_layers * 3);
            assert!(view.completed().outputs.iter().all(|o| o.moments.rows == 4)); // final tokens included
            let legacy = input.dense_oracle();
            let (arch, weights, shapes) = extract(&legacy);
            let mut expected = Calib::new(&arch);
            for window in windows {
                calibrate(&weights, &arch, window, &mut expected);
            }
            for (index, pair) in view.started().plan.pairs.iter().enumerate() {
                let layer = index / 3;
                let moments = match index % 3 {
                    0 => &expected.attn_in[layer],
                    1 => &expected.ffn_in[layer],
                    _ => &expected.down_in[layer],
                };
                let stats = &actual.statistics()[&pair.name];
                assert_eq!(bits(&stats.moments), bits(moments));
                assert_eq!(stats.rows, expected.rows as u64);
                assert_eq!(
                    stats.binding().unwrap(),
                    view.completed().outputs[index].moments
                );
            }
            drop(model);
            drop(observed);
            let cold = verify_jsonl(actual.evidence().jsonl()).unwrap();
            let replayed =
                replay_original_salience_capture(&cold, &mut source, capture_limits()).unwrap();
            for (name, stats) in actual.statistics() {
                assert_eq!(bits(&replayed[name].moments), bits(&stats.moments));
                assert_eq!(replayed[name].rows, stats.rows);
            }
            // Actual vectors feed the existing declared transform; this does not promote B/C ancestry.
            let plan = SalienceTransformPlan {
                architecture: architecture.into(),
                original_source: view.started().plan.source,
                tensors: source.inventory().unwrap(),
                assets: source.asset_names(),
                operations: view
                    .started()
                    .plan
                    .pairs
                    .iter()
                    .map(|pair| SalienceTransformOperation {
                        name: pair.name.clone(),
                        projections: pair.projections.clone(),
                        inverse: pair.inverse.clone(),
                        inverse_axis: pair.inverse_axis,
                        moments: actual.statistics()[&pair.name].binding().unwrap(),
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
            let mut derived =
                SourceTransform::run(&plan, &mut source, actual.statistics()).unwrap();
            let (folded, folded_arch) = fold(&weights, &shapes, &arch, &expected, 0.75);
            let inventory = derived.inventory().unwrap();
            for (name, values) in weight_names(&arch)
                .into_iter()
                .zip(folded.iter().map(Vec::as_slice))
                .chain(norm_tensors(&folded_arch))
            {
                let meta = inventory.iter().find(|t| t.name == name).unwrap();
                let raw = derived
                    .tensor(&name, meta.byte_len().unwrap() as usize)
                    .unwrap();
                let decoded = tritium_format::trit_model::decode_dense_f32(
                    &name,
                    &meta.shape,
                    meta.precision,
                    &raw,
                )
                .unwrap();
                assert_eq!(
                    decoded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
            if architecture == "qwen3"
                && precision == ScalePrecision::Bf16
                && let Some(path) = std::env::var_os("TRITIUM_ORIGINAL_SALIENCE_CAPTURE_OUTPUT")
            {
                std::fs::write(path, actual.evidence().jsonl()).unwrap();
            }
        }
    }
}

#[test]
fn original_salience_preflight_rejects_invalid_windows_caps_and_changed_actual_shards() {
    let hf = HfFixture::new("qwen3", true, 6, None);
    let input = TritFixture::new(&hf, None);
    let raw = raw_fixture_source(&input);
    let path = hf.dir.join("weights.safetensors");
    let mut source = SafeTensorsWeightSource::new(
        [
            std::fs::File::open(&path).unwrap(),
            std::fs::File::open(hf.dir.join("vectors.safetensors")).unwrap(),
        ],
        raw.assets,
        BTreeMap::new(),
    )
    .unwrap();
    let request = request("qwen3", &source);
    let mut observed = ObservedWeightSource::new(&request, &mut source).unwrap();
    let model = SourceCalibrationModel::new(&mut observed, limits()).unwrap();
    for windows in [
        &[][..],
        &[&[][..]][..],
        &[&[1, 2, 3, 4, 5][..]][..],
        &[&[999][..]][..],
    ] {
        assert!(
            model
                .capture_salience_recorded(&mut observed, windows, capture_limits(), recorder())
                .is_err()
        );
    }
    for which in 0..8 {
        let mut bad = capture_limits();
        match which {
            0 => bad.max_source_bytes = 1,
            1 => bad.max_asset_bytes = 1,
            2 => bad.max_config_bytes = 1,
            3 => bad.max_dense_bytes = 1,
            4 => bad.max_window_tokens = 1,
            5 => bad.max_windows = 1,
            6 => bad.max_total_tokens = 1,
            _ => bad.max_moment_bytes = 1,
        }
        assert!(
            model
                .capture_salience_recorded(&mut observed, &[&[1, 3, 2], &[4]], bad, recorder())
                .is_err()
        );
    }
    let unregistered = EvidenceRecorder::new("unregistered-original", []).unwrap();
    assert!(
        model
            .capture_salience_recorded(&mut observed, &[&[1]], capture_limits(), unregistered)
            .is_err()
    );
    let original = std::fs::read(&path).unwrap();
    let offset = original.len() as u64 - 1;
    let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(SeekFrom::Start(offset)).unwrap();
    file.write_all(&[original[offset as usize] ^ 1]).unwrap();
    file.flush().unwrap();
    assert!(
        model
            .capture_salience_recorded(&mut observed, &[&[1]], capture_limits(), recorder())
            .is_err()
    );
    file.seek(SeekFrom::Start(offset)).unwrap();
    file.write_all(&original[offset as usize..]).unwrap();
    file.flush().unwrap();
    model
        .capture_salience_recorded(&mut observed, &[&[1]], capture_limits(), recorder())
        .unwrap();
}

#[test]
fn original_salience_rejects_nonfinite_actual_graph_and_foreign_observed_source() {
    let hf = HfFixture::new("qwen3", true, 6, None);
    let input = TritFixture::new(&hf, None);
    let mut source = raw_fixture_source(&input);
    let request = request("qwen3", &source);
    let mut observed = ObservedWeightSource::new(&request, &mut source).unwrap();
    let model = SourceCalibrationModel::new(&mut observed, limits()).unwrap();
    let mut foreign = raw_fixture_source(&input);
    foreign.assets.get_mut("config.json").unwrap().push(b' ');
    let foreign_request = super::request("qwen3", &foreign);
    let mut foreign_observed = ObservedWeightSource::new(&foreign_request, &mut foreign).unwrap();
    assert!(
        model
            .capture_salience_recorded(&mut foreign_observed, &[&[1]], capture_limits(), recorder())
            .is_err()
    );
    drop(foreign_observed);
    drop(model);
    drop(observed);
    // Finite stored weights are admitted, but actual forward overflow is not a capture result.
    let norm = source
        .tensors
        .get_mut("model.layers.0.input_layernorm.weight")
        .unwrap();
    for value in norm.bytes.as_chunks_mut::<4>().0 {
        value.copy_from_slice(&f32::MAX.to_le_bytes());
    }
    let request = super::request("qwen3", &source);
    let mut observed = ObservedWeightSource::new(&request, &mut source).unwrap();
    let model = SourceCalibrationModel::new(&mut observed, limits()).unwrap();
    assert!(
        model
            .capture_salience_recorded(&mut observed, &[&[1, 2, 3]], capture_limits(), recorder())
            .is_err()
    );
}

#[cfg(not(feature = "tokenizer"))]
#[test]
fn original_text_replay_without_tokenizer_rejects_before_any_actual_source_call() {
    use std::{borrow::Cow, cell::Cell};
    use tritium_quantize::engine::QuantEngineError;
    use tritium_schema::QuantSourceTensor;

    struct Probe<'a> {
        source: &'a mut InMemoryWeightSource,
        calls: &'a Cell<usize>,
    }
    impl Probe<'_> {
        fn record(&self) {
            self.calls.set(self.calls.get() + 1);
        }
    }
    impl WeightSource for Probe<'_> {
        fn inventory(&self) -> Result<Vec<QuantSourceTensor>, QuantEngineError> {
            self.record();
            self.source.inventory()
        }
        fn asset_names(&self) -> Vec<String> {
            self.record();
            self.source.asset_names()
        }
        fn tensor(&mut self, name: &str, limit: usize) -> Result<Cow<'_, [u8]>, QuantEngineError> {
            self.record();
            self.source.tensor(name, limit)
        }
        fn asset(&mut self, name: &str, limit: usize) -> Result<Cow<'_, [u8]>, QuantEngineError> {
            self.record();
            self.source.asset(name, limit)
        }
        fn input_gram(
            &mut self,
            name: &str,
            limit: usize,
        ) -> Result<Cow<'_, [f64]>, QuantEngineError> {
            self.record();
            self.source.input_gram(name, limit)
        }
    }
    let hf = HfFixture::new("qwen3", true, 6, None);
    let input = TritFixture::new(&hf, None);
    let mut source = raw_fixture_source(&input);
    source
        .assets
        .insert("tokenizer_config.json".into(), b"{}".to_vec());
    let request = request("qwen3", &source);
    let mut observed = ObservedWeightSource::new(&request, &mut source).unwrap();
    let model = SourceCalibrationModel::new(&mut observed, limits()).unwrap();
    let actual = model
        .capture_salience_recorded(&mut observed, &[&[1]], capture_limits(), recorder())
        .unwrap();
    let view = OriginalSalienceCaptureView::fold(actual.evidence().log())
        .unwrap()
        .remove(0);
    let mut start = view.started().clone();
    start.plan.input = OriginalSalienceCaptureInput::RawText {
        windows: vec!["hello".into()],
    };
    start.plan.limits.max_text_bytes = 1024;
    start.plan.limits.max_tokenizer_config_bytes = 16_000;
    let receiving = start.plan.limits;
    let text_claim = rehashed(start, view.completed().clone());
    assert_eq!(
        OriginalSalienceCaptureView::fold(text_claim.log())
            .unwrap()
            .len(),
        1
    );
    drop(model);
    drop(observed);
    let calls = Cell::new(0);
    let mut probe = Probe {
        source: &mut source,
        calls: &calls,
    };
    let error =
        replay_original_salience_capture(text_claim.log(), &mut probe, receiving).unwrap_err();
    assert!(error.to_string().contains("requires tokenizer feature"));
    assert_eq!(calls.get(), 0);
    let mut denied = receiving;
    denied.max_source_bytes -= 1;
    let error = replay_original_salience_capture(text_claim.log(), &mut probe, denied).unwrap_err();
    assert!(error.to_string().contains("receiver limit escalation"));
    assert_eq!(calls.get(), 0);
}

#[test]
fn original_salience_replay_rejects_rehashed_moments_consumers_raw_source_and_caps() {
    let hf = HfFixture::new("qwen3", true, 6, None);
    let input = TritFixture::new(&hf, None);
    let mut source = raw_fixture_source(&input);
    let request = request("qwen3", &source);
    let mut observed = ObservedWeightSource::new(&request, &mut source).unwrap();
    let model = SourceCalibrationModel::new(&mut observed, limits()).unwrap();
    let actual = model
        .capture_salience_recorded(&mut observed, &[&[1, 3, 2]], capture_limits(), recorder())
        .unwrap();
    let view = OriginalSalienceCaptureView::fold(actual.evidence().log())
        .unwrap()
        .remove(0);
    drop(model);
    drop(observed);
    let mut wrong = view.completed().clone();
    wrong.outputs[0].moments.content = BlobId::from_bytes([7; 32]);
    let forged = rehashed(view.started().clone(), wrong);
    assert_eq!(
        OriginalSalienceCaptureView::fold(forged.log())
            .unwrap()
            .len(),
        1
    );
    assert!(replay_original_salience_capture(forged.log(), &mut source, capture_limits()).is_err());
    let mut wrong = view.started().clone();
    wrong.plan.pairs[0].projections.pop();
    let forged = rehashed(wrong, view.completed().clone());
    assert_eq!(
        OriginalSalienceCaptureView::fold(forged.log())
            .unwrap()
            .len(),
        1
    );
    assert!(replay_original_salience_capture(forged.log(), &mut source, capture_limits()).is_err());
    let mut changed = source.clone();
    changed.tensors.first_entry().unwrap().get_mut().bytes[0] ^= 1;
    assert!(
        replay_original_salience_capture(actual.evidence().log(), &mut changed, capture_limits())
            .is_err()
    );
    let mut changed = source.clone();
    changed.assets.get_mut("config.json").unwrap().push(b' ');
    assert!(
        replay_original_salience_capture(actual.evidence().log(), &mut changed, capture_limits())
            .is_err()
    );
    for which in 0..8 {
        let mut receiver = capture_limits();
        match which {
            0 => receiver.max_source_bytes -= 1,
            1 => receiver.max_asset_bytes -= 1,
            2 => receiver.max_config_bytes -= 1,
            3 => receiver.max_dense_bytes -= 1,
            4 => receiver.max_window_tokens -= 1,
            5 => receiver.max_windows -= 1,
            6 => receiver.max_total_tokens -= 1,
            _ => receiver.max_moment_bytes -= 1,
        }
        assert!(
            replay_original_salience_capture(actual.evidence().log(), &mut source, receiver)
                .is_err()
        );
    }
    let mut r = recorder();
    r.emit("salience", None, 0, view.started()).unwrap();
    assert!(
        replay_original_salience_capture(r.seal().unwrap().log(), &mut source, capture_limits())
            .is_err()
    );
}

#[cfg(feature = "tokenizer")]
#[test]
fn original_salience_text_runs_actual_tokenizer_and_cold_replay_with_distinct_raw_contexts() {
    let hf = HfFixture::new("qwen3", true, 6, None);
    let mut input = TritFixture::new(&hf, None);
    let (tokenizer, config) = tokenizer_assets(false);
    input.asset("tokenizer.json", tokenizer);
    input.asset("tokenizer_config.json", config);
    let mut source = raw_fixture_source(&input);
    let request = request("qwen3", &source);
    let mut observed = ObservedWeightSource::new(&request, &mut source).unwrap();
    let model = SourceCalibrationModel::new(&mut observed, limits()).unwrap();
    let text_limits = OriginalSalienceCaptureLimits {
        max_text_bytes: 1024,
        max_tokenizer_config_bytes: 16_000,
        ..capture_limits()
    };
    let actual = model
        .capture_salience_text_recorded(
            &mut observed,
            &["hello", "partner hello"],
            text_limits,
            recorder(),
        )
        .unwrap();
    let tokens = model
        .capture_salience_recorded(
            &mut observed,
            &[&[1], &[2, 1]],
            capture_limits(),
            recorder(),
        )
        .unwrap();
    for (name, stats) in actual.statistics() {
        assert_eq!(
            stats.binding().unwrap(),
            tokens.statistics()[name].binding().unwrap()
        );
    }
    let alias = model
        .capture_salience_text_recorded(
            &mut observed,
            &["hello ", "partner hello"],
            text_limits,
            recorder(),
        )
        .unwrap();
    assert_ne!(
        actual.evidence().log().root_digest,
        alias.evidence().log().root_digest
    );
    assert_eq!(
        OriginalSalienceCaptureView::fold(alias.evidence().log()).unwrap()[0].completed(),
        OriginalSalienceCaptureView::fold(actual.evidence().log()).unwrap()[0].completed()
    );
    drop(model);
    drop(observed);
    replay_original_salience_capture(actual.evidence().log(), &mut source, text_limits).unwrap();
    replay_original_salience_capture(alias.evidence().log(), &mut source, text_limits).unwrap(); // numerical coherence is not old-root authentication
    let view = OriginalSalienceCaptureView::fold(actual.evidence().log())
        .unwrap()
        .remove(0);
    let mut changed = view.started().clone();
    let OriginalSalienceCaptureInput::RawText { windows } = &mut changed.plan.input else {
        panic!()
    };
    windows[0] = "partner".into();
    let forged = rehashed(changed, view.completed().clone());
    assert_eq!(
        OriginalSalienceCaptureView::fold(forged.log())
            .unwrap()
            .len(),
        1
    );
    assert!(replay_original_salience_capture(forged.log(), &mut source, text_limits).is_err());
    for which in 0..2 {
        let mut receiver = text_limits;
        if which == 0 {
            receiver.max_text_bytes -= 1;
        } else {
            receiver.max_tokenizer_config_bytes -= 1;
        }
        assert!(
            replay_original_salience_capture(actual.evidence().log(), &mut source, receiver)
                .is_err()
        );
    }
}
