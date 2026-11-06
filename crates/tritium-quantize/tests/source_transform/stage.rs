//! Tests cross the actual stage, package, source, replay and publication interfaces.
use super::*;
use ciborium::value::{CanonicalValue, Value};
use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
};
use tritium_core::DType;
use tritium_format::trit_salience::{
    decode_salience_stage_manifest, encode_salience_stage_manifest,
};
use tritium_format::{
    TritBlob, TritBlobKind, read_trit_package, read_trit_package_with_manifest_limit,
    write_trit_package_bounded,
};
use tritium_quantize::engine::{
    DirectoryTensorSink, PublicationWarning, QuantEngineError, RecordedSalienceSource,
    StagedSalienceSource,
};

pub(super) fn limits() -> SalienceStageLimits {
    SalienceStageLimits {
        max_package_bytes: 100_000,
        max_manifest_bytes: 50_000,
        max_tensor_bytes: 20_000,
        max_asset_bytes: 1024,
        max_statistics_bytes: 768,
    }
}

pub(super) fn recorded() -> (
    InMemoryWeightSource,
    RecordedSalienceSource,
    BTreeMap<String, DeclaredSalienceStatistics>,
) {
    let mut a = fixture(true);
    let (plan, stats) = plan(&mut a);
    let recorded =
        SourceTransform::run_recorded(&plan, &mut a, &stats, transform_recorder()).unwrap();
    (a, recorded, stats)
}

pub(super) fn entries(bytes: &[u8]) -> Vec<(TritBlobKind, Vec<u8>)> {
    let package = read_trit_package(bytes).unwrap();
    package
        .blobs()
        .iter()
        .map(|entry| (entry.kind.clone(), package.blob(entry).unwrap().to_vec()))
        .collect()
}

pub(super) fn repack(manifest: &[u8], entries: &[(TritBlobKind, Vec<u8>)]) -> Vec<u8> {
    let blobs: Vec<_> = entries
        .iter()
        .map(|(kind, bytes)| TritBlob {
            kind: kind.clone(),
            bytes,
        })
        .collect();
    write_trit_package_bounded(manifest, &blobs, 100_000).unwrap()
}

#[test]
fn cold_stage_preserves_bits_moments_assets_and_borrowed_source_and_replays_actual_a() {
    let (mut a, recorded, stats) = recorded();
    let bytes = recorded.encode_stage_package(&stats, limits()).unwrap();
    let log = verify_jsonl(recorded.evidence.jsonl()).unwrap();
    let observation = recorded.source.observation().clone();
    let mut streamed = Vec::new();
    let package_id = recorded
        .write_stage_package_to(&stats, limits(), &mut streamed)
        .unwrap();
    assert_eq!(streamed, bytes);
    assert_eq!(
        package_id.as_bytes(),
        read_trit_package(&bytes).unwrap().package_id().as_bytes()
    );
    drop(recorded); // No in-memory B is available to the reopened source.
    let mut stage = StagedSalienceSource::open(&bytes, &log, limits()).unwrap();
    assert_eq!(stage.manifest().observation, observation);
    assert_eq!(stage.package_id(), package_id);
    assert_eq!(
        Engine::observe_source(&dense_plan(&stage), &mut stage).unwrap(),
        stage.source_id()
    );
    assert!(matches!(
        stage.tensor("embedding", 4).unwrap(),
        std::borrow::Cow::Borrowed(_)
    ));
    assert_eq!(
        stage.tensor("embedding", 4).unwrap().as_ref(),
        a.tensors["embedding"].bytes
    );
    for (name, bytes) in &a.assets {
        assert_eq!(stage.asset(name, 1024).unwrap().as_ref(), bytes);
    }
    for (name, values) in &stats {
        assert_eq!(
            stage.statistics().unwrap()[name].binding().unwrap(),
            values.binding().unwrap()
        );
    }
    assert!(stage.input_gram("q", 1024).is_err());
    assert!(stage.tensor("embedding", 3).is_err());
    stage.verify_against_original(&mut a).unwrap();
    a.tensors.get_mut("q").unwrap().bytes[0] ^= 1;
    assert!(stage.verify_against_original(&mut a).is_err());
    assert!(
        read_trit_model(&bytes).is_err(),
        "ordinary model reader must not ignore stage lineage"
    );
}

#[test]
fn cold_stage_runs_the_actual_guarded_b_to_c_fitter_not_a() {
    let (_, recorded, stats) = recorded();
    let bytes = recorded.encode_stage_package(&stats, limits()).unwrap();
    let log = verify_jsonl(recorded.evidence.jsonl()).unwrap();
    let a = recorded.source.observation().plan.original_source;
    drop(recorded);
    let mut stage = StagedSalienceSource::open(&bytes, &log, limits()).unwrap();
    let b = stage.source_id();
    let norm_b = stage.tensor("attn_norm", 128).unwrap().into_owned();
    let mut request = dense_plan(&stage);
    request
        .tensors
        .iter_mut()
        .find(|t| t.source.name == "q")
        .unwrap()
        .action = QuantAction::Fit {
        plan: Box::new(direct()),
        metric: DirectFitMetric::Identity,
    };
    let mut rejected = MemoryTensorSink::default();
    assert!(
        Engine::run_with_source_guard(&request, &mut stage, &mut rejected, recorder(), a).is_err()
    );
    assert!(rejected.artifact().is_none());
    let mut sink = MemoryTensorSink::default();
    let result =
        Engine::run_with_source_guard(&request, &mut stage, &mut sink, recorder(), b).unwrap();
    assert_eq!(result.source_id, b);
    let (package, evidence) = sink.artifact().unwrap();
    let fit_log = verify_jsonl(evidence.jsonl()).unwrap();
    verify_package_with_fit_log(package, &fit_log).unwrap();
    assert_eq!(
        QuantizationReceiptView::fold(&fit_log).unwrap()[0].source_id(),
        b
    );
    let model = read_trit_model(package).unwrap();
    assert_eq!(model.tensor("l1", "attn_norm").unwrap().1, norm_b);
    assert!(matches!(
        model.tensor("l1", "q").unwrap().0.storage,
        TensorStorage::Additive { .. }
    ));
    // This is an existing B->C artifact, not the future mandatory joined C carrier.
}

#[test]
fn receiving_and_producer_caps_reject_without_writer_output() {
    let (_, recorded, stats) = recorded();
    let bytes = recorded.encode_stage_package(&stats, limits()).unwrap();
    let log = verify_jsonl(recorded.evidence.jsonl()).unwrap();
    for case in 0..10 {
        let mut caps = limits();
        let fields = [
            &mut caps.max_package_bytes,
            &mut caps.max_manifest_bytes,
            &mut caps.max_tensor_bytes,
            &mut caps.max_asset_bytes,
            &mut caps.max_statistics_bytes,
        ];
        *fields[case % 5] = if case < 5 { 1 } else { 0 };
        assert!(
            StagedSalienceSource::open(&bytes, &log, caps).is_err(),
            "receiver case {case}"
        );
        let mut output = Vec::new();
        assert!(
            recorded
                .write_stage_package_to(&stats, caps, &mut output)
                .is_err(),
            "producer case {case}"
        );
        assert!(output.is_empty());
    }
    assert!(read_trit_package_with_manifest_limit(&bytes, 1).is_err());
    let mut changed = stats.clone();
    changed.get_mut("attn").unwrap().rows += 1;
    assert!(recorded.encode_stage_package(&changed, limits()).is_err());
    changed = stats.clone();
    changed.remove("attn");
    assert!(recorded.encode_stage_package(&changed, limits()).is_err());
    changed = stats;
    changed.insert(
        "extra".into(),
        DeclaredSalienceStatistics {
            moments: vec![1.0; 32],
            rows: 2,
        },
    );
    assert!(recorded.encode_stage_package(&changed, limits()).is_err());
}

#[test]
fn rehashed_missing_extra_wrong_kind_and_changed_payloads_are_not_admitted() {
    let (_, recorded, stats) = recorded();
    let bytes = recorded.encode_stage_package(&stats, limits()).unwrap();
    let log = verify_jsonl(recorded.evidence.jsonl()).unwrap();
    let package = read_trit_package(&bytes).unwrap();
    for case in 0..6 {
        let mut changed = entries(&bytes);
        match case {
            0 => {
                changed.remove(0);
            }
            1 => changed.push((TritBlobKind::Asset("unbound.json".into()), b"{}".to_vec())),
            2 => {
                changed
                    .iter_mut()
                    .find(|(k, _)| matches!(k, TritBlobKind::DenseTensor(_)))
                    .unwrap()
                    .0 = TritBlobKind::AdditiveTensor;
            }
            3 => {
                changed
                    .iter_mut()
                    .find(|(k, _)| matches!(k, TritBlobKind::DenseTensor(_)))
                    .unwrap()
                    .1[0] ^= 1;
            }
            4 => {
                changed
                    .iter_mut()
                    .find(|(k, _)| *k == TritBlobKind::Asset(salience_moment_asset(0)))
                    .unwrap()
                    .1[0] ^= 1;
            }
            _ => {
                changed
                    .iter_mut()
                    .find(|(k, _)| *k == TritBlobKind::Asset("config.json".into()))
                    .unwrap()
                    .1
                    .push(b' ');
            }
        }
        let candidate = repack(package.manifest(), &changed);
        assert!(
            read_trit_package(&candidate).is_ok(),
            "valid generic rehash case {case}"
        );
        assert!(
            StagedSalienceSource::open(&candidate, &log, limits()).is_err(),
            "stage case {case}"
        );
    }
}

#[test]
fn foreign_incomplete_and_ambiguous_logs_fail_closed() {
    let (mut a, recorded, stats) = recorded();
    let bytes = recorded.encode_stage_package(&stats, limits()).unwrap();
    let mut other = recorded.source.observation().clone();
    other.operations[0].scale_bits[0] = 1.0f32.to_le_bytes();
    let foreign = claim(&other.plan, &other);
    let foreign_log = verify_jsonl(foreign.jsonl()).unwrap();
    assert!(StagedSalienceSource::open(&bytes, &foreign_log, limits()).is_err());
    let mut unfinished = transform_recorder();
    unfinished
        .emit(
            "transform",
            None,
            0,
            &SalienceTransformStarted {
                plan: other.plan.clone(),
            },
        )
        .unwrap();
    let unfinished = unfinished.seal().unwrap();
    assert!(
        StagedSalienceSource::open(&bytes, &verify_jsonl(unfinished.jsonl()).unwrap(), limits())
            .is_err()
    );
    let mut twice = transform_recorder();
    for span in ["first", "second"] {
        twice
            .emit(
                span,
                None,
                0,
                &SalienceTransformStarted {
                    plan: other.plan.clone(),
                },
            )
            .unwrap();
        twice
            .emit(
                span,
                None,
                1,
                &SalienceTransformCompleted {
                    observation: other.clone(),
                },
            )
            .unwrap();
    }
    let twice = twice.seal().unwrap();
    assert!(
        StagedSalienceSource::open(&bytes, &verify_jsonl(twice.jsonl()).unwrap(), limits())
            .is_err()
    );
    // A self-consistent malicious observation can be structural but not arithmetic truth.
    let original_log = verify_jsonl(recorded.evidence.jsonl()).unwrap();
    let mut manifest = StagedSalienceSource::open(&bytes, &original_log, limits())
        .unwrap()
        .manifest()
        .clone();
    manifest.observation = other;
    manifest.transform = SalienceTransformView::fold(&foreign_log).unwrap()[0]
        .reference()
        .clone();
    let metadata = encode_salience_stage_manifest(&manifest, 50_000).unwrap();
    let malicious = repack(&metadata, &entries(&bytes));
    let stage = StagedSalienceSource::open(&malicious, &foreign_log, limits()).unwrap();
    assert!(matches!(
        stage.verify_against_original(&mut a),
        Err(SalienceSourceError::Replay)
    ));
}

fn cbor(value: &Value) -> Vec<u8> {
    let mut bytes = Vec::new();
    ciborium::ser::into_writer(value, &mut bytes).unwrap();
    bytes
}

#[test]
fn strict_codec_rejects_profiles_versions_fields_order_and_caps() {
    let (_, recorded, stats) = recorded();
    let bytes = recorded.encode_stage_package(&stats, limits()).unwrap();
    let log = verify_jsonl(recorded.evidence.jsonl()).unwrap();
    let stage = StagedSalienceSource::open(&bytes, &log, limits()).unwrap();
    let metadata = read_trit_package(&bytes).unwrap().manifest();
    let original: Value = ciborium::de::from_reader(metadata).unwrap();
    for case in 0..4 {
        let mut changed = original.clone();
        let Value::Array(tuple) = &mut changed else {
            panic!("tuple")
        };
        if case == 0 {
            tuple[0] = Value::Text("tritium.salience-source-stage.v2".into());
        } else {
            let Value::Map(fields) = &mut tuple[1] else {
                panic!("manifest")
            };
            if case == 1 {
                fields.push((Value::Text("unexpected".into()), Value::Bool(true)));
                fields.sort_by_cached_key(|(key, _)| CanonicalValue::from(key.clone()));
            } else if case == 2 {
                fields.reverse();
            } else {
                tuple.push(Value::Null);
            }
        }
        assert!(
            decode_salience_stage_manifest(&cbor(&changed), 50_000).is_err(),
            "codec case {case}"
        );
    }
    assert!(decode_salience_stage_manifest(metadata, metadata.len() - 1).is_err());
    let mut future = stage.manifest().clone();
    future.schema.minor = 1;
    assert!(encode_salience_stage_manifest(&future, 50_000).is_err());
    let mut json = serde_json::to_value(stage.manifest()).unwrap();
    json["unexpected"] = serde_json::json!(true);
    assert!(serde_json::from_value::<SalienceSourceStageManifest>(json).is_err());
    future = stage.manifest().clone();
    future.observation.assets[0].name = format!("{SALIENCE_MOMENT_ASSET_PREFIX}reserved.json");
    assert!(future.validate().is_err());
}

#[test]
fn identity_stage_accepts_equal_a_b_and_shared_dense_payloads() {
    let mut a = fixture(false);
    a.tensors
        .insert("duplicate_embedding".into(), a.tensors["embedding"].clone());
    let (mut plan, stats) = plan(&mut a);
    for operation in &mut plan.operations {
        operation.alpha_bits = 0.0f64.to_le_bytes();
    }
    let recorded =
        SourceTransform::run_recorded(&plan, &mut a, &stats, transform_recorder()).unwrap();
    assert_eq!(
        recorded.source.observation().derived_source,
        plan.original_source
    );
    let bytes = recorded.encode_stage_package(&stats, limits()).unwrap();
    let log = verify_jsonl(recorded.evidence.jsonl()).unwrap();
    let mut stage = StagedSalienceSource::open(&bytes, &log, limits()).unwrap();
    let first = stage.tensor("embedding", 8).unwrap().as_ptr();
    let second = stage.tensor("duplicate_embedding", 8).unwrap().as_ptr();
    assert_eq!(
        first, second,
        "common writer physically deduplicates identical dense payloads"
    );
    stage.verify_against_original(&mut a).unwrap();
}

struct Scratch(PathBuf);
impl Scratch {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "tritium-source-stage-{name}-{}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn real_directory_stage_reopens_and_never_overwrites_existing_outputs() {
    let (mut a, recorded, stats) = recorded();
    let root = Scratch::new("publish");
    let out = root.0.join("stage");
    let mut sink = DirectoryTensorSink::new(&out);
    let published = recorded.publish_stage(&stats, limits(), &mut sink).unwrap();
    let expected_warnings = if fs::File::open(&out).and_then(|f| f.sync_all()).is_err() {
        vec![PublicationWarning::DirectorySync]
    } else {
        Vec::new()
    };
    assert_eq!(published.warnings, expected_warnings);
    let bytes = fs::read(out.join("model.trit")).unwrap();
    let jsonl = fs::read_to_string(out.join("evidence.jsonl")).unwrap();
    assert!(recorded.publish_stage(&stats, limits(), &mut sink).is_err());
    assert_eq!(fs::read(out.join("model.trit")).unwrap(), bytes);
    assert_eq!(
        fs::read_to_string(out.join("evidence.jsonl")).unwrap(),
        jsonl
    );
    assert!(!out.join(".pending.trit").exists());
    drop(recorded);
    let log = verify_jsonl(&jsonl).unwrap();
    let stage = StagedSalienceSource::open(&bytes, &log, limits()).unwrap();
    assert_eq!(stage.package_id(), published.package_id);
    stage.verify_against_original(&mut a).unwrap();
}

struct FailAfter {
    prefix: Vec<u8>,
    remaining: usize,
}
impl Write for FailAfter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.remaining == 0 {
            return Err(io::Error::other("injected partial writer failure"));
        }
        let count = self.remaining.min(bytes.len());
        self.prefix.extend_from_slice(&bytes[..count]);
        self.remaining -= count;
        Ok(count)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn partial_writer_has_no_success_identity_and_common_publisher_cleans_only_owned_paths() {
    let (_, recorded, stats) = recorded();
    let mut writer = FailAfter {
        prefix: Vec::new(),
        remaining: 70,
    };
    assert!(
        recorded
            .write_stage_package_to(&stats, limits(), &mut writer)
            .is_err()
    );
    assert_eq!(writer.prefix.len(), 70);
    assert!(read_trit_package(&writer.prefix).is_err());
    let root = Scratch::new("failure");
    for foreign in [false, true] {
        let out = root.0.join(if foreign { "foreign" } else { "empty" });
        let mut sink = DirectoryTensorSink::new(&out);
        let result: Result<((), _), _> = sink.publish_with_writer(&recorded.evidence, |file| {
            file.write_all(b"partial")?;
            if foreign {
                fs::write(out.join("foreign.txt"), b"preserve me")?;
            }
            Err(QuantEngineError::Io(io::Error::other(
                "injected publication failure",
            )))
        });
        assert!(result.is_err());
        assert!(!out.join("model.trit").exists());
        assert!(!out.join(".pending.trit").exists());
        assert!(!out.join("evidence.jsonl").exists());
        assert_eq!(out.exists(), foreign);
        if foreign {
            assert_eq!(fs::read(out.join("foreign.txt")).unwrap(), b"preserve me");
        }
    }
}

#[test]
fn unsupported_dtype_even_with_unchanged_bytes_is_not_an_admitted_stage() {
    let (_, recorded, stats) = recorded();
    let bytes = recorded.encode_stage_package(&stats, limits()).unwrap();
    let log = verify_jsonl(recorded.evidence.jsonl()).unwrap();
    let mut changed = entries(&bytes);
    let entry = changed
        .iter_mut()
        .find(|(kind, _)| *kind == TritBlobKind::DenseTensor(DType::BF16))
        .unwrap();
    entry.0 = TritBlobKind::DenseTensor(DType::F16);
    let candidate = repack(read_trit_package(&bytes).unwrap().manifest(), &changed);
    assert!(StagedSalienceSource::open(&candidate, &log, limits()).is_err());
}

#[test]
fn stored_moment_bits_preserve_negative_zero_and_reject_self_consistent_invalid_scalars() {
    let mut a = fixture(true);
    let (mut request, mut stats) = plan(&mut a);
    stats.get_mut("attn").unwrap().moments[0] = -0.0;
    request.operations[0].moments = stats["attn"].binding().unwrap();
    let recorded =
        SourceTransform::run_recorded(&request, &mut a, &stats, transform_recorder()).unwrap();
    let bytes = recorded.encode_stage_package(&stats, limits()).unwrap();
    let log = verify_jsonl(recorded.evidence.jsonl()).unwrap();
    let stage = StagedSalienceSource::open(&bytes, &log, limits()).unwrap();
    assert_eq!(
        stage.statistics().unwrap()["attn"].moments[0].to_bits(),
        (-0.0f64).to_bits()
    );
    stage.verify_against_original(&mut a).unwrap();
    for value in [f64::NAN, f64::INFINITY, -1.0] {
        let mut entries = entries(&bytes);
        let entry = entries
            .iter_mut()
            .find(|(kind, _)| *kind == TritBlobKind::Asset(salience_moment_asset(0)))
            .unwrap();
        entry.1[..8].copy_from_slice(&value.to_le_bytes());
        let mut manifest = stage.manifest().clone();
        manifest.observation.plan.operations[0].moments.content =
            BlobId::from_bytes(*blake3::hash(&entry.1).as_bytes());
        let malicious = claim(&manifest.observation.plan, &manifest.observation);
        let malicious_log = verify_jsonl(malicious.jsonl()).unwrap();
        // The generic package and recorded structure are valid; scalar admission must reject.
        manifest.transform = SalienceTransformView::fold(&malicious_log).unwrap()[0]
            .reference()
            .clone();
        let metadata = encode_salience_stage_manifest(&manifest, 50_000).unwrap();
        let candidate = repack(&metadata, &entries);
        assert!(read_trit_package(&candidate).is_ok());
        assert!(matches!(
            StagedSalienceSource::open(&candidate, &malicious_log, limits()),
            Err(SalienceSourceError::Statistics)
        ));
    }
}
