//! Shared retained-reader assembly and real migration caller. Toy graph evidence only.
use super::*;
use std::{
    fs::File,
    io::{Cursor, Read},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use tritium_core::{DType, Trit};
use tritium_evidence::{
    EvidenceRecorder, LEGACY_MODEL_IMPORT_PREPARED_REGISTRATION, TENSOR_UPLOADED_REGISTRATION,
};
use tritium_format::trit_model::{TritModelReader, encode_model_manifest};
use tritium_format::{
    AdditiveTensor, SaltTensorSpoolLimits, TritBlob, TritBlobKind, TritPackageReadLimits,
    write_trit_package,
};
use tritium_nn::{
    Qwen35CheckpointConfig, Qwen35TensorSchemaRole, Qwen35TextRunner, Qwen35TextWeights,
    Qwen35TritReadLimits, TensorLoader, UnverifiedQwen35Mtp, qwen35_language_mtp_tensor_schema,
};
use tritium_quantize::engine::{
    DirectoryTensorSink, InMemoryWeightSource, RawTensor, SaltV2ModelImportRequest, WeightSource,
    import_salt_v2_model,
};
use tritium_runtime::{TensorPayloadBudget, TensorUploadContext};
use tritium_schema::*;
use tritium_spec::{BackendError, DeviceBuffer, TensorView, TernaryBackend};

struct Scratch(PathBuf);
impl Scratch {
    fn new(label: &str) -> Self {
        let path = fixture_dir(&format!("typed-{label}"));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.0) {
            eprintln!("fixture cleanup {}: {error}", self.0.display());
        }
    }
}
fn tensors() -> Vec<TensorFixture> {
    let config = Qwen35CheckpointConfig::from_hf_config(&config_json()).unwrap();
    let schema = qwen35_language_mtp_tensor_schema(&config.text).unwrap();
    oracle_tensors_with_exact_mtp()
        .into_iter()
        .filter(|t| !t.name.starts_with("model.visual."))
        .map(|mut t| {
            if schema
                .iter()
                .any(|s| s.name == t.name && s.role == Qwen35TensorSchemaRole::Matrix)
            {
                // Exactly representable in three balanced radix-three ternary planes.
                for v in &mut t.values {
                    *v = v.clamp(-13.0 / 32.0, 13.0 / 32.0);
                }
            }
            t
        })
        .collect()
}
fn digits(value: f32) -> [i8; 3] {
    let mut n = (value * 32.0) as i32;
    std::array::from_fn(|_| {
        let d = (n + 1).rem_euclid(3) - 1;
        n = (n - d) / 3;
        d as i8
    })
}
fn limits() -> Qwen35TritReadLimits {
    Qwen35TritReadLimits {
        max_config_bytes: 16_000,
        max_tensor_bytes: 16_000,
        max_decoded_tensor_bytes: 16_000,
    }
}
fn receiving() -> TritPackageReadLimits {
    TritPackageReadLimits {
        max_source_bytes: 1_000_000,
        max_manifest_bytes: 128_000,
        max_directory_bytes: 128_000,
    }
}
fn raw(t: &TensorFixture, precision: ScalePrecision) -> RawTensor {
    RawTensor {
        shape: t.shape.iter().map(|&n| n as u64).collect(),
        precision,
        bytes: t
            .values
            .iter()
            .flat_map(|&v| match precision {
                ScalePrecision::F32 => v.to_le_bytes().to_vec(),
                ScalePrecision::F16 => half::f16::from_f32(v).to_bits().to_le_bytes().to_vec(),
                ScalePrecision::Bf16 => half::bf16::from_f32(v).to_bits().to_le_bytes().to_vec(),
                _ => panic!("fixture precision"),
            })
            .collect(),
    }
}
struct Package {
    manifest: ModelManifest,
    blobs: Vec<(TritBlobKind, Vec<u8>)>,
}
impl Package {
    fn new(ts: Vec<TensorFixture>, uniform: bool) -> Self {
        let config = config_json().into_bytes();
        let reference = |bytes: &[u8]| BlobReference {
            id: BlobId::from_bytes(*blake3::hash(bytes).as_bytes()),
            shard: None,
        };
        let schema = qwen35_language_mtp_tensor_schema(
            &Qwen35CheckpointConfig::from_hf_config(&config_json())
                .unwrap()
                .text,
        )
        .unwrap();
        let mut blobs = vec![
            (TritBlobKind::Asset("config.json".into()), config.clone()),
            (TritBlobKind::Asset("tokenizer.json".into()), b"{}".to_vec()),
        ];
        let mut metas = Vec::new();
        for t in ts {
            let (kind, storage, bytes) = if uniform
                && schema
                    .iter()
                    .any(|s| s.name == t.name && s.role == Qwen35TensorSchemaRole::Matrix)
            {
                let layout = AdditiveLayout {
                    rows: t.shape[0] as u64,
                    cols: t.shape[1] as u64,
                    tile: 256,
                    group: 32,
                    max_planes: 3,
                    allocation: PlaneAllocation::Uniform,
                    codec: PlaneCodec::D2,
                    law: ADMITTED_LAWS[0].law,
                    basis: Basis::Identity,
                    transport: Transport::Raw,
                };
                let trits = (0..3)
                    .flat_map(|p| {
                        t.values
                            .iter()
                            .map(move |&v| Trit::from_i8(digits(v)[p]).unwrap())
                    })
                    .collect();
                let scales = [1.0, 3.0, 9.0]
                    .into_iter()
                    .flat_map(|n| vec![n / 32.0; t.shape[0] * t.shape[1].div_ceil(32)])
                    .collect();
                let tensor = AdditiveTensor::new(layout, 3, trits, scales).unwrap();
                (
                    TritBlobKind::AdditiveTensor,
                    TensorStorage::Additive {
                        layout,
                        planes: PlaneCounts::Uniform(3),
                    },
                    tensor.encode_payload().unwrap(),
                )
            } else {
                let r = raw(&t, ScalePrecision::F32);
                (
                    TritBlobKind::DenseTensor(DType::F32),
                    TensorStorage::Dense {
                        shape: r.shape,
                        precision: r.precision,
                    },
                    r.bytes,
                )
            };
            metas.push(TensorMeta {
                name: t.name,
                blob: reference(&bytes),
                storage,
            });
            blobs.push((kind, bytes));
        }
        Self {
            manifest: ModelManifest {
                schema: SchemaId { major: 1, minor: 0 },
                architecture: "qwen3_5".into(),
                assets: vec![
                    ManifestAsset {
                        name: "config.json".into(),
                        blob: reference(&config),
                    },
                    ManifestAsset {
                        name: "tokenizer.json".into(),
                        blob: reference(b"{}"),
                    },
                ],
                levels: vec![Level {
                    name: "typed".into(),
                    fit: LevelFit::Imported,
                    tensors: metas,
                }],
                shards: vec![],
                legacy_ids: vec![],
            },
            blobs,
        }
    }
    fn bytes(&self) -> Vec<u8> {
        let referenced = self
            .manifest
            .assets
            .iter()
            .map(|a| a.blob.id)
            .chain(self.manifest.levels[0].tensors.iter().map(|t| t.blob.id))
            .collect::<Vec<_>>();
        let blobs = self
            .blobs
            .iter()
            .filter(|(_, b)| referenced.contains(&BlobId::from_bytes(*blake3::hash(b).as_bytes())))
            .map(|(kind, bytes)| TritBlob {
                kind: kind.clone(),
                bytes,
            })
            .collect::<Vec<_>>();
        write_trit_package(&encode_model_manifest(&self.manifest).unwrap(), &blobs).unwrap()
    }
}
fn assembled<R: Read + Seek>(
    reader: &mut TritModelReader<R>,
    level: &str,
    backend: &dyn TernaryBackend,
    budget: &TensorPayloadBudget,
    recorder: &mut EvidenceRecorder,
    caps: Qwen35TritReadLimits,
) -> Result<
    (
        Qwen35CheckpointConfig,
        Qwen35TextWeights,
        tritium_nn::Qwen35MtpWeights,
    ),
    tritium_nn::NnError,
> {
    Qwen35TextWeights::load_trit_admitted(
        reader,
        level,
        &mut TensorLoader::new(
            backend,
            TensorUploadPolicy::default(),
            budget,
            TensorUploadContext {
                recorder,
                span: "qwen-load",
                parent: None,
                logical_time: 0,
            },
        ),
        caps,
    )
}
fn compare<R: Read + Seek>(
    reader: TritModelReader<R>,
    level: &str,
    backend: Box<dyn TernaryBackend>,
    label: &str,
) {
    compare_oracle(reader, level, backend, label, tensors());
}
fn compare_oracle<R: Read + Seek>(
    mut reader: TritModelReader<R>,
    level: &str,
    backend: Box<dyn TernaryBackend>,
    label: &str,
    oracle_tensors: Vec<TensorFixture>,
) {
    let root = Scratch::new(label);
    write_fixture_tensors(&root.0, oracle_tensors);
    let oracle = Qwen35HfSource::open(&root.0)
        .unwrap()
        .verify_semantic_identity()
        .unwrap()
        .load_language_mtp(Box::new(tritium_cpu::CpuBackend::new()))
        .unwrap();
    let budget = TensorPayloadBudget::new(u64::MAX);
    let mut evidence =
        EvidenceRecorder::new("typed-qwen-execution", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
    let (config, weights, mtp_weights) = assembled(
        &mut reader,
        level,
        &*backend,
        &budget,
        &mut evidence,
        limits(),
    )
    .unwrap();
    assert!(weights.embedding.as_dense().is_none());
    assert!(matches!(weights.lm_head, tritium_nn::Projection::Tensor(_)));
    let runner = Qwen35TextRunner::new(&config.text, weights, backend).unwrap();
    let mtp = UnverifiedQwen35Mtp::new(&runner, mtp_weights).unwrap();
    assert_eq!(mtp.status(), Qwen35MtpStatus::Unverified);
    let mut expected_names = qwen35_language_mtp_tensor_schema(&config.text)
        .unwrap()
        .into_iter()
        .filter(|s| s.role == Qwen35TensorSchemaRole::Matrix)
        .map(|s| s.name)
        .collect::<Vec<_>>();
    let mut observed_names = evidence
        .events()
        .iter()
        .map(|e| e.payload["tensor"].as_str().unwrap().to_owned())
        .collect::<Vec<_>>();
    expected_names.sort();
    observed_names.sort();
    assert_eq!(observed_names, expected_names);
    let expected_payload = evidence
        .events()
        .iter()
        .map(|e| e.payload["caps"]["payload_bytes"].as_u64().unwrap())
        .sum::<u64>();
    assert_eq!(budget.used_bytes(), expected_payload);
    drop(reader); // No source reader or encoded/dense matrix shadow needed for execution.
    let mut cache = runner.new_cache(16).unwrap();
    let mut oracle_cache = oracle.runner().new_cache(16).unwrap();
    let mut max_error = 0.0f32;
    for ids in [&[1, 3, 2][..], &[4][..], &[5][..]] {
        let actual = runner.forward(ids, &mut cache).unwrap();
        let expected = oracle.runner().forward(ids, &mut oracle_cache).unwrap();
        for (a, b) in actual
            .last_logits()
            .iter()
            .zip(expected.last_logits())
            .chain(
                actual
                    .final_hidden_states()
                    .iter()
                    .zip(expected.final_hidden_states()),
            )
        {
            assert!(a.is_finite() && b.is_finite());
            max_error = max_error.max((a - b).abs());
        }
        assert_eq!(cache.len(), oracle_cache.len());
        assert_eq!(actual.position_start(), expected.position_start());
    }
    assert!(max_error <= 1e-5, "shared graph error {max_error}");
    eprintln!(
        "{label}: cached hybrid graph max_abs_error={max_error}, payload_bytes={expected_payload}"
    );
    drop(mtp);
    drop(runner);
    assert_eq!(budget.used_bytes(), 0);
}

struct Short(Cursor<Vec<u8>>);
impl Read for Short {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        let n = out.len().min(7);
        self.0.read(&mut out[..n])
    }
}
impl Seek for Short {
    fn seek(&mut self, p: SeekFrom) -> std::io::Result<u64> {
        self.0.seek(p)
    }
}
#[test]
fn retained_dense_and_uniform_models_execute_cached_hybrid_graph() {
    for uniform in [false, true] {
        let bytes = Package::new(tensors(), uniform).bytes();
        for cpu in [true, false] {
            let reader =
                TritModelReader::open(Short(Cursor::new(bytes.clone())), receiving()).unwrap();
            let backend: Box<dyn TernaryBackend> = if cpu {
                Box::new(tritium_cpu::CpuBackend::new())
            } else {
                Box::new(tritium_testkit::ReferenceBackend::new())
            };
            compare(
                reader,
                "typed",
                backend,
                &format!("ordinary-{uniform}-{cpu}"),
            );
        }
    }
}

#[test]
fn signed_basis_binding_survives_hybrid_embedding_and_projection_execution() {
    let mut package = Package::new(tensors(), true);
    for meta in &mut package.manifest.levels[0].tensors {
        if let TensorStorage::Additive { layout, .. } = &mut meta.storage {
            layout.basis = Basis::SignedRht {
                block: 2,
                seed: 7,
                domain: 9,
            };
        }
    }
    let bytes = package.bytes();
    let model = tritium_format::trit_model::read_trit_model(&bytes).unwrap();
    let mut oracle_tensors = tensors();
    for fixture in &mut oracle_tensors {
        let (meta, payload) = model.tensor("typed", &fixture.name).unwrap();
        if let TensorStorage::Additive { layout, .. } = &meta.storage {
            let tensor =
                tritium_format::trit_model::decode_additive_payload(meta, payload).unwrap();
            let ids = (0..layout.rows as usize).collect::<Vec<_>>();
            tritium_core::reference_embed(&tensor.view().bind_gather(), &ids, &mut fixture.values)
                .unwrap();
        }
    }
    for cpu in [true, false] {
        let backend: Box<dyn TernaryBackend> = if cpu {
            Box::new(tritium_cpu::CpuBackend::new())
        } else {
            Box::new(tritium_testkit::ReferenceBackend::new())
        };
        compare_oracle(
            TritModelReader::open(Short(Cursor::new(bytes.clone())), receiving()).unwrap(),
            "typed",
            backend,
            &format!("signed-basis-{cpu}"),
            oracle_tensors.clone(),
        );
    }
}

#[test]
fn actual_salt_import_files_feed_shared_qwen_graph_without_refitting() {
    use tritium_format::salt_v2::SaltV2Codec;
    use tritium_format::salt_v2_package::*;
    let root = Scratch::new("actual-import");
    let schema = qwen35_language_mtp_tensor_schema(
        &Qwen35CheckpointConfig::from_hf_config(&config_json())
            .unwrap()
            .text,
    )
    .unwrap();
    let ts = tensors();
    let matrices = ts
        .iter()
        .filter(|t| {
            schema
                .iter()
                .any(|s| s.name == t.name && s.role == Qwen35TensorSchemaRole::Matrix)
        })
        .map(|t| {
            SaltV2Tensor::new_with_layout(
                t.name.clone(),
                t.shape.iter().map(|&n| n as u64).collect(),
                SaltV2Transform::None,
                128,
                vec![
                    SaltV2Tile::new(
                        (0..3)
                            .map(|p| {
                                SaltV2Plane::new_with_scale_group_size(
                                    t.values.iter().map(|&v| digits(v)[p]).collect(),
                                    vec![
                                        half::f16::from_f32([1.0, 3.0, 9.0][p] / 32.0);
                                        t.values.len().div_ceil(128)
                                    ],
                                    128,
                                )
                                .unwrap()
                            })
                            .collect(),
                    )
                    .unwrap(),
                ],
            )
            .unwrap()
        })
        .collect();
    let legacy = write_salt_v2_package(&SaltV2Package::new(SaltV2Codec::B3, matrices).unwrap())
        .unwrap()
        .bytes;
    std::fs::write(root.0.join("original.tsalt2"), legacy).unwrap();
    for precision in [
        ScalePrecision::F32,
        ScalePrecision::F16,
        ScalePrecision::Bf16,
    ] {
        for codec in [PlaneCodec::D2, PlaneCodec::B3] {
            for group in [32, 128] {
                let label = format!("import-{precision:?}-{codec:?}-{group}");
                let mut companions = InMemoryWeightSource {
                    tensors: ts
                        .iter()
                        .filter(|t| {
                            schema.iter().any(|s| {
                                s.name == t.name && s.role == Qwen35TensorSchemaRole::Preserved
                            })
                        })
                        .map(|t| (t.name.clone(), raw(t, precision)))
                        .collect(),
                    assets: BTreeMap::from([
                        ("config.json".into(), config_json().into_bytes()),
                        ("tokenizer.json".into(), b"{}".to_vec()),
                    ]),
                    grams: BTreeMap::new(),
                };
                let request = SaltV2ModelImportRequest {
                    companions: QuantPlan {
                        architecture: "qwen3_5".into(),
                        tensors: companions
                            .inventory()
                            .unwrap()
                            .into_iter()
                            .map(|source| QuantTensorPlan {
                                source,
                                action: QuantAction::KeepDense,
                            })
                            .collect(),
                        assets: companions.asset_names(),
                        levels: vec!["imported".into()],
                        limits: QuantLimits {
                            max_source_bytes: 1_000_000,
                            max_asset_bytes: 16_000,
                            max_retained_blob_bytes: 1_000_000,
                            max_package_bytes: 1_000_000,
                        },
                    },
                    codec,
                    group,
                    tensor_limits: SaltTensorSpoolLimits {
                        max_source_bytes: 1_000_000,
                        max_trits: 1_000_000,
                        max_scales: 1_000_000,
                        max_working_bytes: 2_000_000,
                        max_payload_bytes: 1_000_000,
                    },
                    max_spool_bytes: 1_000_000,
                    receiving: receiving(),
                };
                let mut original = SaltV2PackageReader::new_strict(
                    File::open(root.0.join("original.tsalt2")).unwrap(),
                )
                .unwrap();
                let mut spool = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create_new(true)
                    .open(root.0.join(format!("{label}.spool")))
                    .unwrap();
                let mut sink = DirectoryTensorSink::new(root.0.join(&label));
                let mut imported = import_salt_v2_model(
                    &request,
                    &mut original,
                    &mut companions,
                    &mut spool,
                    &mut sink,
                    EvidenceRecorder::new(&label, [LEGACY_MODEL_IMPORT_PREPARED_REGISTRATION])
                        .unwrap(),
                )
                .unwrap();
                let budget = TensorPayloadBudget::new(u64::MAX);
                let mut recorder =
                    EvidenceRecorder::new("per-tile-cap", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
                assert!(
                    assembled(
                        &mut imported.model,
                        "imported",
                        &tritium_cpu::CpuBackend::new(),
                        &budget,
                        &mut recorder,
                        Qwen35TritReadLimits {
                            max_decoded_tensor_bytes: 1,
                            ..limits()
                        }
                    )
                    .is_err()
                );
                assert!(recorder.events().is_empty());
                assert_eq!(budget.used_bytes(), 0);
                assert!(
                    imported.model.manifest().levels[0]
                        .tensors
                        .iter()
                        .filter(|m| matches!(m.storage, TensorStorage::Additive { .. }))
                        .all(|m| matches!(
                            m.storage,
                            TensorStorage::Additive {
                                planes: PlaneCounts::PerTile(_),
                                ..
                            }
                        ))
                );
                compare(
                    imported.model,
                    "imported",
                    Box::new(tritium_cpu::CpuBackend::new()),
                    &label,
                );
            }
        }
    }
}

#[derive(Debug)]
struct Probe {
    count: Arc<AtomicUsize>,
    fail_at: Option<usize>,
    mutate: Option<PathBuf>,
}
impl TernaryBackend for Probe {
    fn device_id(&self) -> &str {
        "cpu-qwen-assembly-probe"
    }
    fn capabilities(&self) -> tritium_spec::DeviceCaps {
        tritium_cpu::CpuBackend::new().capabilities()
    }
    fn tensor_caps(
        &self,
        t: TensorView<'_>,
    ) -> Result<Option<tritium_spec::TensorCaps>, BackendError> {
        tritium_cpu::CpuBackend::new().tensor_caps(t)
    }
    fn upload_tensor(&self, t: TensorView<'_>) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        let n = self.count.fetch_add(1, Ordering::SeqCst) + 1;
        if let Some(path) = self.mutate.as_ref().filter(|_| n == 1) {
            let bytes = std::fs::read(path).unwrap();
            let config = config_json();
            let offset = bytes
                .windows(config.len())
                .position(|w| w == config.as_bytes())
                .unwrap();
            let mut f = OpenOptions::new().write(true).open(path).unwrap();
            f.seek(SeekFrom::Start(offset as u64)).unwrap();
            f.write_all(b"[").unwrap();
            f.sync_all().unwrap();
        }
        if self.fail_at == Some(n) {
            return Err(BackendError::Backend("injected upload fault".into()));
        }
        tritium_cpu::CpuBackend::new().upload_tensor(t)
    }
    fn upload_weights(
        &self,
        p: &[u8],
        s: tritium_core::GemmShape,
        f: tritium_core::TernaryFormat,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        tritium_cpu::CpuBackend::new().upload_weights(p, s, f)
    }
    fn mpgemm(&self, p: tritium_spec::MpGemm<'_>) -> Result<(), BackendError> {
        tritium_cpu::CpuBackend::new().mpgemm(p)
    }
}
fn rejection(package: &Package, level: &str, caps: Qwen35TritReadLimits, registered: bool) {
    let mut reader = TritModelReader::open(Cursor::new(package.bytes()), receiving()).unwrap();
    let count = Arc::new(AtomicUsize::new(0));
    let backend = Probe {
        count: count.clone(),
        fail_at: None,
        mutate: None,
    };
    let budget = TensorPayloadBudget::new(u64::MAX);
    let mut recorder = EvidenceRecorder::new(
        "typed-qwen-reject",
        if registered {
            vec![TENSOR_UPLOADED_REGISTRATION]
        } else {
            vec![]
        },
    )
    .unwrap();
    assert!(assembled(&mut reader, level, &backend, &budget, &mut recorder, caps).is_err());
    assert_eq!(count.load(Ordering::SeqCst), 0);
    assert_eq!(budget.used_bytes(), 0);
    assert!(recorder.events().is_empty());
}
#[test]
fn inventory_shape_storage_config_and_level_reject_before_any_upload() {
    let mut missing = tensors();
    missing.pop();
    rejection(&Package::new(missing, false), "typed", limits(), true);
    let mut extra = tensors();
    extra.push(tensor("model.visual.pos_embed.weight", &[1, H], 29));
    rejection(&Package::new(extra, false), "typed", limits(), true);
    let mut renamed = tensors();
    renamed.last_mut().unwrap().name = "unknown.weight".into();
    rejection(&Package::new(renamed, false), "typed", limits(), true);
    let mut wrong = tensors();
    let matrix = wrong
        .iter_mut()
        .find(|t| t.name == "mtp.fc.weight")
        .unwrap();
    matrix.shape.reverse();
    rejection(&Package::new(wrong, false), "typed", limits(), true);
    let mut wrong = tensors();
    wrong
        .iter_mut()
        .find(|t| t.name == "mtp.norm.weight")
        .unwrap()
        .shape = vec![1, H];
    rejection(&Package::new(wrong, false), "typed", limits(), true);
    let mut architecture = Package::new(tensors(), false);
    architecture.manifest.architecture = "qwen3".into();
    rejection(&architecture, "typed", limits(), true);
    let mut storage = Package::new(tensors(), true);
    let matrix = storage.manifest.levels[0]
        .tensors
        .iter()
        .find(|m| matches!(m.storage, TensorStorage::Additive { .. }))
        .unwrap()
        .clone();
    let norm = storage.manifest.levels[0]
        .tensors
        .iter_mut()
        .find(|m| m.name == "mtp.norm.weight")
        .unwrap();
    norm.blob = matrix.blob;
    norm.storage = matrix.storage;
    rejection(&storage, "typed", limits(), true);
    let mut config = Package::new(tensors(), false);
    let bytes = b"{\"model_type\":\"qwen3_5\"}".to_vec();
    config.manifest.assets[0].blob.id = BlobId::from_bytes(*blake3::hash(&bytes).as_bytes());
    config.blobs[0].1 = bytes;
    rejection(&config, "typed", limits(), true);
    rejection(&Package::new(tensors(), false), "absent", limits(), true);
}
#[test]
fn all_source_decode_caps_and_missing_registration_reject_before_upload() {
    for uniform in [false, true] {
        let package = Package::new(tensors(), uniform);
        for caps in [
            Qwen35TritReadLimits {
                max_config_bytes: 1,
                ..limits()
            },
            Qwen35TritReadLimits {
                max_tensor_bytes: 1,
                ..limits()
            },
            Qwen35TritReadLimits {
                max_decoded_tensor_bytes: 1,
                ..limits()
            },
            Qwen35TritReadLimits {
                max_config_bytes: 0,
                ..limits()
            },
            Qwen35TritReadLimits {
                max_tensor_bytes: 0,
                ..limits()
            },
            Qwen35TritReadLimits {
                max_decoded_tensor_bytes: 0,
                ..limits()
            },
        ] {
            rejection(&package, "typed", caps, true);
        }
        rejection(&package, "typed", limits(), false);
    }
}
#[test]
fn budget_and_backend_fault_drop_every_private_payload_lease() {
    for fault in [false, true] {
        let mut reader = TritModelReader::open(
            Cursor::new(Package::new(tensors(), true).bytes()),
            receiving(),
        )
        .unwrap();
        let count = Arc::new(AtomicUsize::new(0));
        let backend = Probe {
            count: count.clone(),
            fail_at: fault.then_some(3),
            mutate: None,
        };
        let budget = TensorPayloadBudget::new(if fault { u64::MAX } else { 1 });
        let mut recorder =
            EvidenceRecorder::new("typed-qwen-upload-fault", [TENSOR_UPLOADED_REGISTRATION])
                .unwrap();
        let error = assembled(
            &mut reader,
            "typed",
            &backend,
            &budget,
            &mut recorder,
            limits(),
        )
        .err()
        .unwrap()
        .to_string();
        assert!(
            error.contains(if fault {
                "injected upload fault"
            } else {
                "budget"
            }),
            "{error}"
        );
        assert_eq!(count.load(Ordering::SeqCst), if fault { 3 } else { 0 });
        assert_eq!(recorder.events().len(), if fault { 2 } else { 0 });
        assert_eq!(budget.used_bytes(), 0);
    }
}
#[test]
fn original_file_mutation_after_config_read_fails_terminal_assembly() {
    let root = Scratch::new("terminal-mutation");
    let path = root.0.join("model.trit");
    std::fs::write(&path, Package::new(tensors(), true).bytes()).unwrap();
    let mut reader = TritModelReader::open(File::open(&path).unwrap(), receiving()).unwrap();
    let count = Arc::new(AtomicUsize::new(0));
    let backend = Probe {
        count: count.clone(),
        fail_at: None,
        mutate: Some(path),
    };
    let budget = TensorPayloadBudget::new(u64::MAX);
    let mut recorder =
        EvidenceRecorder::new("typed-qwen-source-fault", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
    assert!(
        assembled(
            &mut reader,
            "typed",
            &backend,
            &budget,
            &mut recorder,
            limits()
        )
        .is_err()
    );
    assert!(count.load(Ordering::SeqCst) > 0);
    assert!(!recorder.events().is_empty());
    assert_eq!(budget.used_bytes(), 0);
    assert!(reader.verify().is_err());
}
