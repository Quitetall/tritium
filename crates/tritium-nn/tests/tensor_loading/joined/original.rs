//! Original raw source uses the same masters/loader/tape as the mandatory B stage.
use super::*;
use std::io::{Seek, SeekFrom, Write};
use tritium_nn::calibrate::{SourceCalibrationModel, StagedCalibrationLimits};
use tritium_quantize::engine::{ObservedWeightSource, SafeTensorsWeightSource};
use tritium_schema::ScalePrecision;
use tritium_train::Tape;

#[path = "original/captured.rs"]
mod captured;
#[path = "original/lineage.rs"]
mod lineage;

fn limits() -> StagedCalibrationLimits {
    StagedCalibrationLimits {
        max_config_bytes: 16_000,
        max_dense_bytes: 64_000,
        max_window_tokens: 4,
    }
}

fn request(architecture: &str, source: &dyn WeightSource) -> QuantPlan {
    QuantPlan {
        architecture: architecture.into(),
        tensors: source
            .inventory()
            .unwrap()
            .into_iter()
            .map(|source| QuantTensorPlan {
                source,
                action: QuantAction::KeepDense,
            })
            .collect(),
        assets: source.asset_names(),
        levels: vec!["standard".into()],
        limits: QuantLimits {
            max_source_bytes: 64_000,
            max_asset_bytes: 16_000,
            max_retained_blob_bytes: 64_000,
            max_package_bytes: 1_000_000,
        },
    }
}

#[test]
fn actual_original_masters_execute_after_source_is_dropped_for_all_storage_precisions() {
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
            let plan = request(architecture, &source);
            let original = Engine::observe_source(&plan, &mut source).unwrap();
            let mut guarded = ObservedWeightSource::new(&plan, &mut source).unwrap();
            let model = SourceCalibrationModel::new(&mut guarded, limits()).unwrap();
            assert_eq!(model.source_id(), original);
            guarded.revalidate().unwrap();
            drop(guarded);
            drop(source);
            let tokens = [1, 3, 2];
            let mut legacy = input.dense_oracle();
            let expected = legacy.forward(&tokens, &[0, 1, 2]).unwrap();
            let mut tape = Tape::new();
            let mut taps = Vec::new();
            let logits = model
                .forward_observed(&mut tape, &tokens, &mut |_, tap| taps.push(tap))
                .unwrap();
            let actual = &tape.value(logits)[(tokens.len() - 1) * model.arch().vocab..];
            assert_eq!(actual.len(), expected.len());
            for (a, b) in actual.iter().zip(expected) {
                assert!(
                    (a - b).abs() < 2e-5,
                    "{architecture} {precision:?}: {a} != {b}"
                );
            }
            assert_eq!(taps.len(), model.arch().n_layers * 7 + 2);
            // Actual existing CPU and reference typed loaders, not a copied forward.
            for reference in [false, true] {
                let bytes = input.bytes();
                let budget = TensorPayloadBudget::new(100_000);
                let mut recorder =
                    EvidenceRecorder::new("original-runtime", [TENSOR_UPLOADED_REGISTRATION])
                        .unwrap();
                let backend: Box<dyn TernaryBackend> = if reference {
                    Box::new(tritium_testkit::ReferenceBackend::new())
                } else {
                    Box::new(tritium_cpu::CpuBackend::new())
                };
                let mut runner = ModelRunner::from_source_admitted(
                    tritium_nn::ModelSource::Trit {
                        bytes: &bytes,
                        level: "standard",
                    },
                    backend,
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
                let expected = runner.forward(&tokens, &[0, 1, 2]).unwrap();
                for (a, b) in actual.iter().zip(expected) {
                    assert!((a - b).abs() < 2e-5);
                }
                drop(runner);
                assert_eq!(budget.used_bytes(), 0);
            }
            for bad in [
                &[][..],
                &[1, 2, 3, 4, 5][..],
                &[model.arch().vocab as u32][..],
            ] {
                let before = tape.leaf(vec![7.0]);
                assert!(
                    model
                        .forward_observed(&mut tape, bad, &mut |_, _| panic!("unexpected tap"))
                        .is_err()
                );
                assert_eq!(tape.leaf(vec![8.0]), before + 1);
            }
        }
    }
}

#[test]
fn original_model_rejects_actual_file_mutation_and_limits_before_returning_masters() {
    let hf = HfFixture::new("qwen3", true, 6, None);
    let input = TritFixture::new(&hf, None);
    let source = raw_fixture_source(&input);
    let path = hf.dir.join("weights.safetensors");
    let mut source = SafeTensorsWeightSource::new(
        [
            std::fs::File::open(&path).unwrap(),
            std::fs::File::open(hf.dir.join("vectors.safetensors")).unwrap(),
        ],
        source.assets,
        BTreeMap::new(),
    )
    .unwrap();
    let plan = request("qwen3", &source);
    let mut guarded = ObservedWeightSource::new(&plan, &mut source).unwrap();
    let original = std::fs::read(&path).unwrap();
    let offset = original.len() as u64 - 1;
    let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(SeekFrom::Start(offset)).unwrap();
    file.write_all(&[original[offset as usize] ^ 1]).unwrap();
    file.flush().unwrap();
    assert!(SourceCalibrationModel::new(&mut guarded, limits()).is_err());
    file.seek(SeekFrom::Start(offset)).unwrap();
    file.write_all(&original[offset as usize..]).unwrap();
    file.flush().unwrap();
    SourceCalibrationModel::new(&mut guarded, limits()).unwrap();
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
        assert!(SourceCalibrationModel::new(&mut guarded, bad).is_err());
    }
}
