//! Actual raw source -> shared fitter -> sealed log -> checked package -> sinks.
#[path = "model_engine/calibrated.rs"]
mod calibrated;
#[path = "model_engine/observed.rs"]
mod observed;
#[path = "model_engine/owner.rs"]
mod owner;
#[path = "model_engine/spool_publish.rs"]
mod spool_publish;
use std::{collections::BTreeMap, io::Cursor};
use tritium_evidence::{
    DIRECT_LEVELS_FITTED_REGISTRATION, EvidenceRecorder, QUANTIZATION_COMPLETED_REGISTRATION,
    QUANTIZATION_STARTED_REGISTRATION, QuantizationReceiptView, verify_jsonl,
};
use tritium_format::trit_model::read_trit_model;
use tritium_format::{
    AdditiveTensor, TritBlob, TritBlobKind, TritPackageError, read_trit_package,
    write_trit_package, write_trit_package_bounded,
};
use tritium_quantize::engine::{
    DirectoryTensorSink, Engine, InMemoryWeightSource, MemoryTensorSink, PublicationWarning,
    QuantEngineError, RawTensor, SafeTensorsWeightSource, TensorSink, WeightSource,
    fit_direct_levels, verify_package_with_fit_log,
};
use tritium_schema::*;

struct ChangingSource {
    inner: InMemoryWeightSource,
    tensor_reads: usize,
    asset_reads: usize,
}
impl WeightSource for ChangingSource {
    fn inventory(&self) -> Result<Vec<QuantSourceTensor>, QuantEngineError> {
        self.inner.inventory()
    }
    fn asset_names(&self) -> Vec<String> {
        self.inner.asset_names()
    }
    fn tensor(
        &mut self,
        name: &str,
        max_bytes: usize,
    ) -> Result<std::borrow::Cow<'_, [u8]>, QuantEngineError> {
        self.tensor_reads += 1;
        if self.tensor_reads == 3 {
            // Actual source mutation at conversion's first read, after preparation.
            self.inner.tensors.get_mut("weight").unwrap().bytes[..4]
                .copy_from_slice(&0.25f32.to_le_bytes());
        }
        self.inner.tensor(name, max_bytes)
    }
    fn asset(
        &mut self,
        name: &str,
        max_bytes: usize,
    ) -> Result<std::borrow::Cow<'_, [u8]>, QuantEngineError> {
        self.asset_reads += 1;
        self.inner.asset(name, max_bytes)
    }
    fn input_gram(
        &mut self,
        name: &str,
        max_entries: usize,
    ) -> Result<std::borrow::Cow<'_, [f64]>, QuantEngineError> {
        self.inner.input_gram(name, max_entries)
    }
}

fn recorder() -> EvidenceRecorder {
    EvidenceRecorder::new(
        "engine-run",
        [
            DIRECT_LEVELS_FITTED_REGISTRATION,
            QUANTIZATION_STARTED_REGISTRATION,
            QUANTIZATION_COMPLETED_REGISTRATION,
        ],
    )
    .unwrap()
}
fn direct() -> DirectFitPlan {
    DirectFitPlan {
        layout: AdditiveLayout {
            rows: 1,
            cols: 32,
            tile: 256,
            group: 32,
            max_planes: 3,
            allocation: PlaneAllocation::Uniform,
            codec: PlaneCodec::B3,
            law: ScaleLaw {
                anchor: ScaleAnchor::Group,
                relation: PlaneRelation::Free,
                precision: ScalePrecision::F32,
            },
            basis: Basis::Identity,
            transport: Transport::Raw,
        },
        profile: DirectFitProfile::JointIndependentGroups(JointGroupFitOptions {
            max_iterations: 8,
            ridge: 1e-8,
            em_restarts: 2,
            ridge_condition_limit: 1e6,
            softened_relay: false,
            modulated_relay: false,
        }),
        levels: [3, 1, 2]
            .map(|planes| DirectFitLevel {
                name: format!("l{planes}"),
                planes,
            })
            .into(),
        limits: DirectFitLimits {
            max_weights: 32,
            max_metric_entries: 1024,
            max_output_trits: 192,
        },
    }
}
fn source(precision: ScalePrecision) -> InMemoryWeightSource {
    let values = (0..32)
        .map(|i| (i as f32 - 16.0) / 16.0)
        .collect::<Vec<_>>();
    let encode = |values: &[f32]| {
        values
            .iter()
            .flat_map(|&v| match precision {
                ScalePrecision::F32 => v.to_le_bytes().to_vec(),
                ScalePrecision::F16 => half::f16::from_f32(v).to_bits().to_le_bytes().to_vec(),
                ScalePrecision::Bf16 => half::bf16::from_f32(v).to_bits().to_le_bytes().to_vec(),
                _ => unreachable!(),
            })
            .collect::<Vec<_>>()
    };
    InMemoryWeightSource {
        tensors: BTreeMap::from([
            (
                "norm".into(),
                RawTensor {
                    shape: vec![2],
                    precision,
                    bytes: encode(&[-0.0, 1.0]),
                },
            ),
            (
                "weight".into(),
                RawTensor {
                    shape: vec![1, 32],
                    precision,
                    bytes: encode(&values),
                },
            ),
        ]),
        assets: BTreeMap::from([
            ("config.json".into(), b"{}".to_vec()),
            ("tokenizer.json".into(), b"{}".to_vec()),
        ]),
        grams: BTreeMap::new(),
    }
}

#[test]
fn original_source_content_identity_oracle() {
    let mut output = Vec::new();
    for precision in [
        ScalePrecision::F32,
        ScalePrecision::F16,
        ScalePrecision::Bf16,
    ] {
        for change in 0..4 {
            let mut input = source(precision);
            match change {
                1 => {
                    input.assets.insert("config.json".into(), b"{ }".to_vec());
                }
                2 => {
                    input.tensors.get_mut("norm").unwrap().bytes.fill(0);
                }
                3 => {
                    input.assets.insert("extra.json".into(), b"extra".to_vec());
                }
                _ => {}
            }
            let request = plan(&input);
            let id = Engine::observe_source(&request, &mut input).unwrap();
            assert_eq!(id, Engine::observe_source(&request, &mut input).unwrap());
            output.extend_from_slice(id.as_bytes());
        }
    }
    if let Some(path) = std::env::var_os("TRITIUM_OBSERVED_SOURCE_ORACLE_OUTPUT") {
        std::fs::write(path, output).unwrap();
    }
}
fn plan(source: &dyn WeightSource) -> QuantPlan {
    QuantPlan {
        architecture: "fixture".into(),
        tensors: source
            .inventory()
            .unwrap()
            .into_iter()
            .map(|source| {
                let action = if source.shape.len() == 2 {
                    QuantAction::Fit {
                        plan: Box::new(direct()),
                        metric: DirectFitMetric::Identity,
                    }
                } else {
                    QuantAction::KeepDense
                };
                QuantTensorPlan { source, action }
            })
            .collect(),
        assets: source.asset_names(),
        levels: vec!["l1".into(), "l2".into(), "l3".into()],
        limits: QuantLimits {
            max_source_bytes: 1024,
            max_asset_bytes: 100,
            max_retained_blob_bytes: 1024,
            max_package_bytes: 100_000,
        },
    }
}
fn shard(tensors: &BTreeMap<String, RawTensor>) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, t) in tensors {
        let start = data.len();
        data.extend_from_slice(&t.bytes);
        let dtype = match t.precision {
            ScalePrecision::F32 => "F32",
            ScalePrecision::F16 => "F16",
            ScalePrecision::Bf16 => "BF16",
            _ => unreachable!(),
        };
        header.insert(
            name.clone(),
            serde_json::json!({"dtype":dtype,"shape":t.shape,"data_offsets":[start,data.len()]}),
        );
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(data);
    bytes
}

#[test]
fn both_sources_preserve_original_bits_fit_each_level_and_replay() {
    for precision in [
        ScalePrecision::F32,
        ScalePrecision::F16,
        ScalePrecision::Bf16,
    ] {
        let mut memory = source(precision);
        let originals = memory.clone();
        let request = plan(&memory);
        let observed_source = Engine::observe_source(&request, &mut memory).unwrap();
        let mut sink = MemoryTensorSink::default();
        let result = Engine::run(&request, &mut memory, &mut sink, recorder()).unwrap();
        assert_eq!(result.source_id, observed_source);
        assert!(result.warnings.is_empty());
        assert_eq!(
            originals.tensors["norm"].bytes,
            memory.tensors["norm"].bytes
        );
        let (package, evidence) = sink.artifact().unwrap();
        verify_package_with_fit_log(package, &verify_jsonl(evidence.jsonl()).unwrap()).unwrap();
        let model = read_trit_model(package).unwrap();
        let raw = &originals.tensors["weight"];
        let values = tritium_format::trit_model::decode_dense_f32(
            "weight",
            &raw.shape,
            raw.precision,
            &raw.bytes,
        )
        .unwrap();
        for fit in fit_direct_levels(&direct(), &values, None).unwrap() {
            let (meta, bytes) = model.tensor(&fit.level, "weight").unwrap();
            let TensorStorage::Additive {
                layout,
                planes: PlaneCounts::Uniform(count),
            } = meta.storage
            else {
                panic!("not additive")
            };
            assert_eq!(
                AdditiveTensor::decode_payload(layout, count, bytes).unwrap(),
                fit.tensor
            );
            assert_eq!(
                model.tensor(&fit.level, "norm").unwrap().1,
                originals.tensors["norm"].bytes
            );
            assert!(matches!(
                model
                    .manifest()
                    .levels
                    .iter()
                    .find(|l| l.name == fit.level)
                    .unwrap()
                    .fit,
                LevelFit::Direct { .. }
            ));
        }
        assert_eq!(result.evidence.root(), evidence.root());
        assert_eq!(
            QuantizationReceiptView::fold(evidence.log()).unwrap()[0].source_id(),
            observed_source
        );
        assert_eq!(
            QuantizationReceiptView::fold(evidence.log()).unwrap().len(),
            1
        );
        // Equal exact asset bytes, distinct named directory bindings, one stored offset.
        let outer = read_trit_package(package).unwrap();
        let asset_offsets = outer
            .blobs()
            .iter()
            .filter(|b| matches!(b.kind, TritBlobKind::Asset(_)))
            .map(|b| b.offset)
            .collect::<Vec<_>>();
        assert_eq!(asset_offsets.len(), 2);
        assert_eq!(asset_offsets[0], asset_offsets[1]);
        let shards = originals
            .tensors
            .iter()
            .map(|(name, t)| Cursor::new(shard(&BTreeMap::from([(name.clone(), t.clone())]))))
            .collect::<Vec<_>>();
        let mut seek =
            SafeTensorsWeightSource::new(shards, originals.assets.clone(), BTreeMap::new())
                .unwrap();
        let mut seek_sink = MemoryTensorSink::default();
        let other = Engine::run(&request, &mut seek, &mut seek_sink, recorder()).unwrap();
        assert_eq!(other.source_id, result.source_id);
        assert_eq!(other.package_id, result.package_id);
        assert_eq!(other.evidence, result.evidence);
        assert_eq!(seek_sink.artifact().unwrap().0, package);
    }
}

#[test]
fn original_source_identity_is_independent_of_fitting_but_detects_actual_byte_changes() {
    let mut input = source(ScalePrecision::F32);
    let request = plan(&input);
    let original = Engine::observe_source(&request, &mut input).unwrap();
    let mut dense_request = request.clone();
    dense_request.levels = vec!["dense".into()];
    for tensor in &mut dense_request.tensors {
        tensor.action = QuantAction::KeepDense;
    }
    let dense = Engine::run(
        &dense_request,
        &mut input,
        &mut MemoryTensorSink::default(),
        recorder(),
    )
    .unwrap();
    assert_eq!(dense.source_id, original);
    let fitted = Engine::run(
        &request,
        &mut input,
        &mut MemoryTensorSink::default(),
        recorder(),
    )
    .unwrap();
    assert_eq!(fitted.source_id, original);
    assert_ne!(dense.package_id, fitted.package_id);
    // Exact bits matter even when numeric +0 and -0 are equal.
    input.tensors.get_mut("norm").unwrap().bytes[..4].copy_from_slice(&0f32.to_le_bytes());
    assert_ne!(
        Engine::observe_source(&request, &mut input).unwrap(),
        original
    );
    input = source(ScalePrecision::F32);
    input.assets.insert("config.json".into(), b"{ }".to_vec());
    assert_ne!(
        Engine::observe_source(&request, &mut input).unwrap(),
        original
    );
}

#[test]
fn original_source_observation_enforces_inventory_payload_and_logical_limits() {
    let mut input = source(ScalePrecision::F32);
    let request = plan(&input);
    let mut limited = request.clone();
    limited.limits.max_source_bytes = 1;
    assert!(matches!(
        Engine::observe_source(&limited, &mut input),
        Err(QuantEngineError::Limit("source bytes"))
    ));
    limited = request.clone();
    limited.limits.max_asset_bytes = 1;
    assert!(Engine::observe_source(&limited, &mut input).is_err());
    input.tensors.get_mut("norm").unwrap().bytes.pop();
    assert!(Engine::observe_source(&request, &mut input).is_err());
    input = source(ScalePrecision::F32);
    input.assets.insert("extra.json".into(), vec![]);
    assert!(matches!(
        Engine::observe_source(&request, &mut input),
        Err(QuantEngineError::Binding(
            "complete original inventory changed"
        ))
    ));
}

#[test]
fn source_guard_checks_consumed_content_before_any_publication_without_changing_v1_evidence() {
    let mut input = source(ScalePrecision::F32);
    let request = plan(&input);
    let expected = Engine::observe_source(&request, &mut input).unwrap();
    let mut original_sink = MemoryTensorSink::default();
    let original = Engine::run(&request, &mut input, &mut original_sink, recorder()).unwrap();
    let mut guarded_sink = MemoryTensorSink::default();
    let guarded = Engine::run_with_source_guard(
        &request,
        &mut input,
        &mut guarded_sink,
        recorder(),
        expected,
    )
    .unwrap();
    assert_eq!(guarded.source_id, expected);
    assert_eq!(guarded.package_id, original.package_id);
    assert_eq!(guarded.evidence, original.evidence);
    assert_eq!(
        guarded_sink.artifact().unwrap().0,
        original_sink.artifact().unwrap().0
    );

    for change in 0..3 {
        let mut changed = source(ScalePrecision::F32);
        match change {
            0 => changed.tensors.get_mut("weight").unwrap().bytes[..4]
                .copy_from_slice(&0.25f32.to_le_bytes()),
            1 => changed.tensors.get_mut("norm").unwrap().bytes[..4]
                .copy_from_slice(&0f32.to_le_bytes()),
            _ => {
                changed
                    .assets
                    .insert("tokenizer.json".into(), b"{ }".to_vec());
            }
        }
        let mut sink = MemoryTensorSink::default();
        assert!(matches!(
            Engine::run_with_source_guard(&request, &mut changed, &mut sink, recorder(), expected),
            Err(QuantEngineError::Binding(
                "actual original source differs from preparation"
            ))
        ));
        assert!(sink.artifact().is_none());
    }
    let mut sink = MemoryTensorSink::default();
    assert!(
        Engine::run_with_source_guard(
            &request,
            &mut input,
            &mut sink,
            recorder(),
            OriginalSourceId::from_bytes([0; 32])
        )
        .is_err()
    );
    assert!(sink.artifact().is_none());
}

#[test]
fn source_guard_detects_mutation_during_conversion_without_rereading_the_source() {
    let mut input = ChangingSource {
        inner: source(ScalePrecision::F32),
        tensor_reads: 0,
        asset_reads: 0,
    };
    let request = plan(&input);
    let expected = Engine::observe_source(&request, &mut input).unwrap();
    assert_eq!(input.tensor_reads, 2);
    assert_eq!(input.asset_reads, 2);
    let mut sink = MemoryTensorSink::default();
    assert!(matches!(
        Engine::run_with_source_guard(&request, &mut input, &mut sink, recorder(), expected),
        Err(QuantEngineError::Binding(
            "actual original source differs from preparation"
        ))
    ));
    assert!(sink.artifact().is_none());
    assert_eq!(input.tensor_reads, 4);
    assert_eq!(input.asset_reads, 4);
}

#[test]
fn dense_only_is_imported_and_exactly_preserved() {
    let mut source = source(ScalePrecision::Bf16);
    let mut request = plan(&source);
    for t in &mut request.tensors {
        t.action = QuantAction::KeepDense;
    }
    let mut sink = MemoryTensorSink::default();
    let result = Engine::run(&request, &mut source, &mut sink, recorder()).unwrap();
    let model = read_trit_model(sink.artifact().unwrap().0).unwrap();
    for l in &model.manifest().levels {
        assert_eq!(l.fit, LevelFit::Imported);
        for (name, t) in &source.tensors {
            assert_eq!(model.tensor(&l.name, name).unwrap().1, t.bytes);
        }
    }
    assert_eq!(result.evidence.log().events.len(), 2);
}

#[test]
fn admission_and_actual_source_failure_never_publish() {
    for failure in [
        "source-limit",
        "asset-limit",
        "retained-limit",
        "package-limit",
        "extra",
        "nan",
        "registration",
        "profile",
        "missing-gram",
        "changed-gram",
    ] {
        let mut source = source(ScalePrecision::F32);
        let mut request = plan(&source);
        match failure {
            "source-limit" => request.limits.max_source_bytes = 1,
            "asset-limit" => request.limits.max_asset_bytes = 1,
            "retained-limit" => request.limits.max_retained_blob_bytes = 1,
            "package-limit" => request.limits.max_package_bytes = 1,
            "extra" => {
                source.assets.insert("extra.json".into(), vec![]);
            }
            "nan" => source.tensors.get_mut("weight").unwrap().bytes[..4]
                .copy_from_slice(&f32::NAN.to_le_bytes()),
            "profile" => {
                let QuantAction::Fit { plan, .. } = &mut request.tensors[1].action else {
                    unreachable!()
                };
                let DirectFitProfile::JointIndependentGroups(ref mut opts) = plan.profile else {
                    unreachable!()
                };
                opts.max_iterations = 0;
            }
            "missing-gram" | "changed-gram" => {
                let QuantAction::Fit { metric, .. } = &mut request.tensors[1].action else {
                    unreachable!()
                };
                *metric = DirectFitMetric::InputGramF64 {
                    content: FitInputDigest([0; 32]),
                };
                if failure == "changed-gram" {
                    source.grams.insert("weight".into(), vec![1.; 1024]);
                }
            }
            _ => {}
        }
        let mut sink = MemoryTensorSink::default();
        let evidence = if failure == "registration" {
            EvidenceRecorder::new("unregistered", []).unwrap()
        } else {
            recorder()
        };
        assert!(
            Engine::run(&request, &mut source, &mut sink, evidence).is_err(),
            "{failure}"
        );
        assert!(sink.artifact().is_none(), "{failure}");
    }
}

#[test]
fn bounded_writer_is_byte_identical_and_checks_complete_size() {
    let blobs = [TritBlob {
        kind: TritBlobKind::Asset("a".into()),
        bytes: b"data",
    }];
    let legacy = write_trit_package(&[0x80], &blobs).unwrap();
    assert_eq!(
        legacy,
        write_trit_package_bounded(&[0x80], &blobs, legacy.len()).unwrap()
    );
    assert_eq!(
        write_trit_package_bounded(&[0x80], &blobs, legacy.len() - 1).unwrap_err(),
        TritPackageError::PackageLimitExceeded {
            requested: legacy.len(),
            limit: legacy.len() - 1
        }
    );
}

#[test]
fn package_and_log_tampering_are_rejected() {
    let mut source = source(ScalePrecision::F32);
    let request = plan(&source);
    let mut sink = MemoryTensorSink::default();
    Engine::run(&request, &mut source, &mut sink, recorder()).unwrap();
    let (package, evidence) = sink.artifact().unwrap();
    let mut bytes = package.to_vec();
    bytes[256] ^= 1;
    assert!(verify_package_with_fit_log(&bytes, evidence.log()).is_err());
    let mut log = evidence.log().clone();
    log.root_digest.replace_range(..1, "z");
    assert!(verify_package_with_fit_log(package, &log).is_err());
    let mut log = evidence.log().clone();
    log.events.last_mut().unwrap().payload["sources"][0]["source"]["name"] = "changed".into();
    assert!(verify_package_with_fit_log(package, &log).is_err());
    let mut source2 = source.clone();
    source2.tensors.get_mut("weight").unwrap().bytes[..4].copy_from_slice(&3.0f32.to_le_bytes());
    let mut sink2 = MemoryTensorSink::default();
    let result2 = Engine::run(&request, &mut source2, &mut sink2, recorder()).unwrap();
    assert!(verify_package_with_fit_log(package, result2.evidence.log()).is_err());
}

#[test]
fn duplicate_cross_shard_names_are_rejected() {
    let source = source(ScalePrecision::F32);
    let bytes = shard(&source.tensors);
    assert!(
        SafeTensorsWeightSource::new(
            [Cursor::new(bytes.clone()), Cursor::new(bytes)],
            source.assets,
            BTreeMap::new()
        )
        .is_err()
    );
}

#[derive(Debug)]
struct FailingSink;
impl TensorSink for FailingSink {
    fn publish(
        &mut self,
        _: &[u8],
        _: &tritium_evidence::SealedEvidence,
    ) -> Result<Vec<PublicationWarning>, QuantEngineError> {
        Err(QuantEngineError::Io(std::io::Error::other(
            "injected precommit failure",
        )))
    }
}
#[test]
fn sink_failure_is_not_reported_as_success() {
    let mut source = source(ScalePrecision::F32);
    let request = plan(&source);
    assert!(Engine::run(&request, &mut source, &mut FailingSink, recorder()).is_err());
}

#[test]
fn real_file_source_and_directory_sink_never_overwrite() {
    let root = std::env::temp_dir().join(format!("tritium-engine-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    {
        let original = source(ScalePrecision::F16);
        let request = plan(&original);
        let input = root.join("source.safetensors");
        std::fs::write(&input, shard(&original.tensors)).unwrap();
        let mut source = SafeTensorsWeightSource::new(
            [std::fs::File::open(&input).unwrap()],
            original.assets,
            BTreeMap::new(),
        )
        .unwrap();
        let out = root.join("out");
        let mut sink = DirectoryTensorSink::new(&out);
        let result = Engine::run(&request, &mut source, &mut sink, recorder()).unwrap();
        // Publication warnings describe post-commit operations. Directory
        // handles cannot be opened/synced on every supported OS/filesystem;
        // assert the observed contract instead of assuming Unix semantics.
        let expected_warnings = if std::fs::File::open(&out)
            .and_then(|directory| directory.sync_all())
            .is_err()
        {
            vec![PublicationWarning::DirectorySync]
        } else {
            Vec::new()
        };
        assert_eq!(result.warnings, expected_warnings);
        let bytes = std::fs::read(out.join("model.trit")).unwrap();
        let log = std::fs::read_to_string(out.join("evidence.jsonl")).unwrap();
        verify_package_with_fit_log(&bytes, &verify_jsonl(&log).unwrap()).unwrap();
        assert!(!out.join(".pending.trit").exists());
        assert!(matches!(
            Engine::run(&request, &mut source, &mut sink, recorder()),
            Err(QuantEngineError::DestinationExists)
        ));
        assert_eq!(std::fs::read(out.join("model.trit")).unwrap(), bytes);
        std::fs::remove_file(out.join("model.trit")).unwrap();
        std::fs::remove_file(out.join("evidence.jsonl")).unwrap();
        std::fs::remove_dir(out).unwrap();
        std::fs::remove_file(input).unwrap();
    }
    std::fs::remove_dir(root).unwrap();
}

#[test]
fn requested_real_gram_is_used_and_input_changes_rebind_provenance() {
    let mut source = source(ScalePrecision::F32);
    let curvature = (0..1024)
        .map(|i| if i / 32 == i % 32 { 2.0 } else { 0.01 })
        .collect::<Vec<f64>>();
    let raw = &source.tensors["weight"];
    let values = tritium_format::trit_model::decode_dense_f32(
        "weight",
        &raw.shape,
        raw.precision,
        &raw.bytes,
    )
    .unwrap();
    let mut per_tensor =
        EvidenceRecorder::new("metric-source", [DIRECT_LEVELS_FITTED_REGISTRATION]).unwrap();
    let fit = tritium_quantize::engine::fit_direct_levels_recorded(
        "weight",
        &direct(),
        &values,
        Some(&curvature),
        &mut per_tensor,
        "fit",
        None,
        0,
    )
    .unwrap();
    let mut request = plan(&source);
    let QuantAction::Fit { metric, .. } = &mut request.tensors[1].action else {
        unreachable!()
    };
    *metric = fit.observation.observation().metric.clone();
    source.grams.insert("weight".into(), curvature);
    let mut sink = MemoryTensorSink::default();
    let result = Engine::run(&request, &mut source, &mut sink, recorder()).unwrap();
    let views = QuantizationReceiptView::fold(result.evidence.log()).unwrap();
    assert_eq!(
        views[0].fits()[0].observation().metric,
        fit.observation.observation().metric
    );
    let model = read_trit_model(sink.artifact().unwrap().0).unwrap();
    for level in fit.levels {
        assert_eq!(
            model.tensor(&level.level, "weight").unwrap().1,
            level.tensor.encode_payload().unwrap()
        );
    }
}

#[test]
fn joint_feedback_profile_is_source_bound_replayed_and_published_deterministically() {
    for (precision, basis) in [
        (WorkPrecision::F32Accum64, Basis::Hadamard { block: 64 }),
        (WorkPrecision::F64, Basis::Hadamard { block: 64 }),
        (
            WorkPrecision::F32Accum64,
            Basis::SignedRht {
                block: 16,
                seed: 42,
                domain: 9,
            },
        ),
        (
            WorkPrecision::F64,
            Basis::SignedRht {
                block: 16,
                seed: 42,
                domain: 9,
            },
        ),
    ] {
        let mut source = source(ScalePrecision::F32);
        let values = (0..64)
            .map(|i| ((i * 29 % 101) as f32 - 50.0) / 37.0)
            .collect::<Vec<_>>();
        source.tensors.insert(
            "weight".into(),
            RawTensor {
                shape: vec![1, 64],
                precision: ScalePrecision::F32,
                bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            },
        );
        let mut fit_plan = direct();
        let DirectFitProfile::JointIndependentGroups(fitter) = fit_plan.profile else {
            unreachable!()
        };
        fit_plan.profile = DirectFitProfile::JointFeedbackGroups(JointFeedbackFitOptions {
            fitter,
            damp: 0.01,
            precision,
            decay: 0.6,
            decay_ramp: true,
        });
        fit_plan.layout.cols = 64;
        fit_plan.layout.basis = basis;
        fit_plan.limits.max_weights = 64;
        fit_plan.limits.max_metric_entries = 4096;
        fit_plan.limits.max_output_trits = 384;
        let curvature = (0..4096)
            .map(|i| if i / 64 == i % 64 { 1.2 } else { 0.2 })
            .collect::<Vec<f64>>();
        let mut recording =
            EvidenceRecorder::new("metric-source", [DIRECT_LEVELS_FITTED_REGISTRATION]).unwrap();
        let actual = tritium_quantize::engine::fit_direct_levels_recorded(
            "weight",
            &fit_plan,
            &values,
            Some(&curvature),
            &mut recording,
            "fit",
            None,
            0,
        )
        .unwrap();
        let mut request = plan(&source);
        request.tensors[1].action = QuantAction::Fit {
            plan: Box::new(fit_plan.clone()),
            metric: actual.observation.observation().metric.clone(),
        };
        source.grams.insert("weight".into(), curvature);
        let before = source.clone();
        let mut first = MemoryTensorSink::default();
        let result = Engine::run(&request, &mut source, &mut first, recorder()).unwrap();
        let replay = verify_jsonl(result.evidence.jsonl()).unwrap();
        let views = QuantizationReceiptView::fold(&replay).unwrap();
        assert_eq!(views[0].fits()[0].observation().plan, fit_plan);
        verify_package_with_fit_log(first.artifact().unwrap().0, &replay).unwrap();
        let model = read_trit_model(first.artifact().unwrap().0).unwrap();
        for level in actual.levels {
            assert_eq!(
                model.tensor(&level.level, "weight").unwrap().1,
                level.tensor.encode_payload().unwrap()
            );
        }
        let mut second = MemoryTensorSink::default();
        let repeated = Engine::run(&request, &mut source, &mut second, recorder()).unwrap();
        assert_eq!(result.package_id, repeated.package_id);
        assert_eq!(result.evidence.jsonl(), repeated.evidence.jsonl());
        assert_eq!(first.artifact().unwrap().0, second.artifact().unwrap().0);
        assert_eq!(
            source.tensors["weight"].bytes,
            before.tensors["weight"].bytes
        );
        let mut json = serde_json::to_value(&fit_plan).unwrap();
        json["profile"]["JointFeedbackGroups"]["precision"] =
            serde_json::json!("ImaginaryPrecision");
        assert!(serde_json::from_value::<DirectFitPlan>(json).is_err());
    }
}

#[derive(serde::Serialize)]
#[serde(transparent)]
struct CompletionPayload(serde_json::Value);
impl tritium_evidence::EvidenceEvent for CompletionPayload {
    const SCHEMA_ID: &'static str = "tritium.quantize.quantization_completed";
    const VERSION: u16 = 1;
}

#[test]
fn valid_hash_chains_do_not_make_structural_lies_or_partial_runs_complete() {
    let mut source = source(ScalePrecision::F32);
    let request = plan(&source);
    let mut sink = MemoryTensorSink::default();
    let result = Engine::run(&request, &mut source, &mut sink, recorder()).unwrap();
    let events = &result.evidence.log().events;
    let start: QuantizationStarted = serde_json::from_value(events[0].payload.clone()).unwrap();
    let fit: DirectLevelsFitted = serde_json::from_value(events[1].payload.clone()).unwrap();
    let complete = events[2].payload.clone();
    let mut partial = recorder();
    partial.emit("quantize", None, 0, &start).unwrap();
    let seal = partial.seal().unwrap();
    assert_eq!(seal.run(), "engine-run");
    assert!(
        QuantizationReceiptView::fold(seal.log())
            .unwrap()
            .is_empty()
    );
    for lie in [
        "unknown-field",
        "missing-source",
        "missing-fit",
        "dense-content",
        "output-planes",
        "extra-level",
    ] {
        let mut forged = recorder();
        forged.emit("quantize", None, 0, &start).unwrap();
        if lie != "missing-fit" {
            forged.emit("tensor-1", Some("quantize"), 0, &fit).unwrap();
        }
        let mut payload = complete.clone();
        match lie {
            "unknown-field" => payload["unexpected"] = true.into(),
            "missing-source" => {
                payload["sources"].as_array_mut().unwrap().pop();
            }
            "dense-content" => {
                payload["sources"][0]["content"] =
                    serde_json::to_value(BlobId::from_bytes([1; 32])).unwrap()
            }
            "output-planes" => {
                payload["levels"][0]["tensors"][1]["storage"]["Additive"]["planes"]["Uniform"] =
                    3.into()
            }
            "extra-level" => {
                let extra = payload["levels"][0].clone();
                payload["levels"].as_array_mut().unwrap().push(extra);
            }
            _ => {}
        }
        forged
            .emit("quantize", None, 1, &CompletionPayload(payload))
            .unwrap();
        let log = verify_jsonl(&forged.to_jsonl().unwrap()).unwrap();
        assert!(QuantizationReceiptView::fold(&log).is_err(), "{lie}");
    }
}
