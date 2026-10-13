//! Actual whole-model caller, strict original source and ordinary Imported log.
use half::f16;
use std::{
    borrow::Cow,
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Cursor, Read, Seek, SeekFrom, Write},
};
use tritium_evidence::{
    EventRegistration, EvidenceEvent, EvidenceRecorder, LEGACY_MODEL_IMPORT_PREPARED_REGISTRATION,
    LegacyModelImportView, verify_jsonl,
};
use tritium_format::salt_v2::SaltV2Codec;
use tritium_format::salt_v2_package::{
    SaltV2Package, SaltV2PackageReadError, SaltV2PackageReader, SaltV2Plane, SaltV2Tensor,
    SaltV2Tile, SaltV2Transform, write_salt_v2_package,
};
use tritium_format::trit_model::{TritModelReader, TritModelStageError};
use tritium_format::{AdditiveTensor, SaltTensorSpoolLimits, TritPackageReadLimits};
use tritium_quantize::engine::*;
use tritium_schema::*;

struct Scratch(std::path::PathBuf);
impl Scratch {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "tritium-model-import-{name}-{}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn tensor(name: &str, dims: Vec<u64>, transform: SaltV2Transform) -> SaltV2Tensor {
    let length = dims.iter().product::<u64>() as usize;
    SaltV2Tensor::new_with_layout(
        name,
        dims,
        transform,
        128,
        (0..length.div_ceil(256))
            .map(|tile| {
                let n = (length - tile * 256).min(256);
                SaltV2Tile::new(
                    (0..1 + tile % 3)
                        .map(|plane| {
                            SaltV2Plane::new_with_scale_group_size(
                                (0..n).map(|i| [0, 1, -1, 1][(i + plane) % 4]).collect(),
                                vec![f16::from_bits([1, 0x3555, 0x3c00][plane]); n.div_ceil(128)],
                                128,
                            )
                            .unwrap()
                        })
                        .collect(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap()
}
fn legacy(source: SaltV2Codec, tensors: Vec<SaltV2Tensor>) -> Vec<u8> {
    write_salt_v2_package(&SaltV2Package::new(source, tensors).unwrap())
        .unwrap()
        .bytes
}
fn originals(precision: ScalePrecision) -> InMemoryWeightSource {
    let bytes = match precision {
        ScalePrecision::F32 => [1.0f32, -0.25]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect(),
        ScalePrecision::F16 => [1.0, -0.25]
            .into_iter()
            .flat_map(|f| f16::from_f32(f).to_bits().to_le_bytes())
            .collect(),
        ScalePrecision::Bf16 => [1.0, -0.25]
            .into_iter()
            .flat_map(|f| half::bf16::from_f32(f).to_bits().to_le_bytes())
            .collect(),
        _ => panic!("fixture precision"),
    };
    InMemoryWeightSource {
        tensors: BTreeMap::from([(
            "norm".into(),
            RawTensor {
                shape: vec![2],
                precision,
                bytes,
            },
        )]),
        assets: BTreeMap::from([
            (
                "config.json".into(),
                b"{ \"model_type\": \"fixture\" }\n".to_vec(),
            ),
            ("tokenizer.json".into(), b"{}\n".to_vec()),
            ("chat_template.jinja".into(), b"{{ messages }}\n".to_vec()),
        ]),
        grams: BTreeMap::new(),
    }
}
fn request(source: &dyn WeightSource, codec: PlaneCodec, group: u16) -> SaltV2ModelImportRequest {
    SaltV2ModelImportRequest {
        companions: QuantPlan {
            architecture: "fixture".into(),
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
            levels: vec!["imported".into()],
            limits: QuantLimits {
                max_source_bytes: 1_000_000,
                max_asset_bytes: 1_000_000,
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
        receiving: TritPackageReadLimits {
            max_source_bytes: 1_000_000,
            max_manifest_bytes: 65536,
            max_directory_bytes: 65536,
        },
    }
}
fn recorder() -> EvidenceRecorder {
    EvidenceRecorder::new(
        "actual-legacy-import",
        [LEGACY_MODEL_IMPORT_PREPARED_REGISTRATION],
    )
    .unwrap()
}
fn check_tensor(model: &mut TritModelReader<File>, source: &SaltV2Tensor) {
    let meta = model.manifest().levels[0]
        .tensors
        .iter()
        .find(|t| t.name == source.name())
        .unwrap()
        .clone();
    let TensorStorage::Additive {
        layout,
        planes: PlaneCounts::PerTile(counts),
    } = meta.storage
    else {
        panic!("compact actual metadata")
    };
    assert_eq!([layout.rows, layout.cols], source.dims());
    let mut bytes = Vec::new();
    model
        .visit_tensor("imported", source.name(), |chunk| -> Result<(), ()> {
            bytes.extend_from_slice(chunk);
            Ok(())
        })
        .unwrap();
    let recovered = AdditiveTensor::decode_per_tile_payload(layout, &counts, &bytes).unwrap();
    let index = tritium_core::OwnedPerTileIndex::new(layout, counts).unwrap();
    for plane in 0..usize::from(layout.max_planes) {
        for row in 0..layout.rows as usize {
            for tile in index.as_index().tiles_in_row(plane, row).unwrap() {
                let columns = tile.columns();
                for col in columns.clone() {
                    let flat = row * layout.cols as usize + col;
                    let original = &source.tiles()[flat / 256].planes()[plane];
                    assert_eq!(
                        recovered.trits()[tile.trit_offset() + col - columns.start],
                        original.trits()[flat % 256]
                    );
                    assert_eq!(
                        f16::from_f32(
                            recovered.scales()[tile.scale_offset()
                                + (col - columns.start) / usize::from(layout.group)]
                        )
                        .to_bits(),
                        original.scales()[(flat % 256) / 128].to_bits()
                    );
                }
            }
        }
    }
}

#[test]
fn actual_files_import_all_source_codecs_dense_precisions_groups_and_exact_original_bits() {
    let root = Scratch::new("parity");
    let mut cases = 0;
    for source_codec in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
        for precision in [
            ScalePrecision::F16,
            ScalePrecision::Bf16,
            ScalePrecision::F32,
        ] {
            for codec in [PlaneCodec::D2, PlaneCodec::B3] {
                for group in [32, 128] {
                    let sources = vec![
                        tensor("weight", vec![2, 256], SaltV2Transform::None),
                        tensor("other", vec![1, 256], SaltV2Transform::None),
                    ];
                    let legacy_path = root.0.join(format!("{cases}.tsalt2"));
                    fs::write(&legacy_path, legacy(source_codec, sources.clone())).unwrap();
                    let mut original =
                        SaltV2PackageReader::new_strict(File::open(&legacy_path).unwrap()).unwrap();
                    let legacy_id = original.package_id().to_string();
                    let mut companions = originals(precision);
                    let expected = companions.clone();
                    let plan = request(&companions, codec, group);
                    let mut spool = OpenOptions::new()
                        .read(true)
                        .write(true)
                        .create_new(true)
                        .open(root.0.join(format!("{cases}.spool")))
                        .unwrap();
                    let out = root.0.join(format!("{cases}.out"));
                    let mut sink = DirectoryTensorSink::new(&out);
                    let mut result = import_salt_v2_model(
                        &plan,
                        &mut original,
                        &mut companions,
                        &mut spool,
                        &mut sink,
                        recorder(),
                    )
                    .unwrap();
                    let manifest = result.model.manifest();
                    assert_eq!(
                        manifest.legacy_ids,
                        vec![LegacyModelIdentity {
                            id: legacy_id,
                            level: "imported".into()
                        }]
                    );
                    assert_eq!(manifest.levels[0].fit, LevelFit::Imported);
                    assert_eq!(manifest.levels[0].tensors.len(), 3);
                    for source in &sources {
                        check_tensor(&mut result.model, source);
                    }
                    let mut dense = Vec::new();
                    result
                        .model
                        .visit_tensor("imported", "norm", |chunk| -> Result<(), ()> {
                            dense.extend_from_slice(chunk);
                            Ok(())
                        })
                        .unwrap();
                    assert_eq!(dense, expected.tensors["norm"].bytes);
                    for (name, bytes) in &expected.assets {
                        let mut actual = Vec::new();
                        result
                            .model
                            .visit_asset(name, |chunk| -> Result<(), ()> {
                                actual.extend_from_slice(chunk);
                                Ok(())
                            })
                            .unwrap();
                        assert_eq!(&actual, bytes);
                    }
                    let log = fs::read_to_string(out.join("evidence.jsonl")).unwrap();
                    assert_eq!(log, result.evidence.jsonl());
                    let verified = verify_jsonl(&log).unwrap();
                    let view =
                        verify_imported_model_with_log(&mut result.model, &verified).unwrap();
                    assert_eq!(view.observation().manifest, *result.model.manifest());
                    assert_eq!(
                        view.observation().companion_source,
                        Engine::observe_source(&plan.companions, &mut companions).unwrap()
                    );
                    assert_eq!(verified.events.len(), 1);
                    assert!(!out.join(".pending.trit").exists());
                    let mut reopened = TritModelReader::open(
                        File::open(out.join("model.trit")).unwrap(),
                        plan.receiving,
                    )
                    .unwrap();
                    verify_imported_model_with_log(&mut reopened, &verified).unwrap();
                    assert_eq!(reopened.package_id(), result.model.package_id());
                    assert_eq!(
                        reopened.model_id("imported").unwrap(),
                        result.model.model_id("imported").unwrap()
                    );
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 36);
}

#[test]
fn nonempty_spool_name_collision_rank_and_unregistered_evidence_never_publish() {
    let root = Scratch::new("preflight");
    for refusal in 0..6 {
        let source = match refusal {
            1 => tensor("norm", vec![2, 256], SaltV2Transform::None),
            2 => tensor("weight", vec![512], SaltV2Transform::None),
            3 => tensor("weight", vec![1, 2, 256], SaltV2Transform::None),
            5 => tensor(
                "weight",
                vec![2, 256],
                SaltV2Transform::SignedRht { seed: 7, domain: 9 },
            ),
            _ => tensor("weight", vec![2, 256], SaltV2Transform::None),
        };
        let mut reader =
            SaltV2PackageReader::new_strict(Cursor::new(legacy(SaltV2Codec::B3, vec![source])))
                .unwrap();
        let mut companions = originals(ScalePrecision::F32);
        let plan = request(&companions, PlaneCodec::B3, 128);
        let mut spool = Cursor::new(if refusal == 0 { vec![42] } else { vec![] });
        let out = root.0.join(format!("{refusal}.out"));
        let mut sink = DirectoryTensorSink::new(&out);
        let rec = if refusal == 4 {
            EvidenceRecorder::new("unregistered", []).unwrap()
        } else {
            recorder()
        };
        assert!(
            import_salt_v2_model(
                &plan,
                &mut reader,
                &mut companions,
                &mut spool,
                &mut sink,
                rec
            )
            .is_err()
        );
        assert_eq!(
            spool.get_ref().as_slice(),
            if refusal == 0 { &[42][..] } else { &[][..] }
        );
        assert!(!out.exists());
    }
}

#[test]
fn spool_package_metadata_and_companion_caps_refuse_without_published_success() {
    let root = Scratch::new("caps");
    for refusal in 0..7 {
        let mut reader = SaltV2PackageReader::new_strict(Cursor::new(legacy(
            SaltV2Codec::B3,
            vec![tensor("weight", vec![2, 256], SaltV2Transform::None)],
        )))
        .unwrap();
        let mut companions = originals(ScalePrecision::F32);
        let mut plan = request(&companions, PlaneCodec::B3, 128);
        match refusal {
            0 => plan.max_spool_bytes = 1,
            1 => plan.tensor_limits.max_working_bytes = 1,
            2 => plan.receiving.max_source_bytes = 1,
            3 => plan.receiving.max_manifest_bytes = 0,
            4 => plan.receiving.max_directory_bytes = 0,
            5 => plan.companions.limits.max_asset_bytes = 1,
            _ => plan.companions.limits.max_source_bytes = 1,
        }
        let mut spool = Cursor::new(Vec::new());
        let out = root.0.join(format!("{refusal}.out"));
        let mut sink = DirectoryTensorSink::new(&out);
        assert!(
            import_salt_v2_model(
                &plan,
                &mut reader,
                &mut companions,
                &mut spool,
                &mut sink,
                recorder()
            )
            .is_err()
        );
        assert!(!out.exists());
    }
}

#[test]
fn library_manifest_lower_bound_refuses_before_any_spool_write() {
    let root = Scratch::new("manifest-bound");
    let mut reader = SaltV2PackageReader::new_strict(Cursor::new(legacy(
        SaltV2Codec::B3,
        vec![tensor("weight", vec![2, 256], SaltV2Transform::None)],
    )))
    .unwrap();
    let mut companions = originals(ScalePrecision::F32);
    let mut plan = request(&companions, PlaneCodec::B3, 128);
    let inspected = inspect_salt_v2_model_import(&plan, &reader, &mut companions).unwrap();
    assert!(inspected.manifest_lower_bound_bytes > 1);
    plan.receiving.max_manifest_bytes = (inspected.manifest_lower_bound_bytes - 1) as usize;
    let mut spool = Cursor::new(Vec::new());
    let out = root.0.join("output");
    let mut sink = DirectoryTensorSink::new(&out);
    let error = import_salt_v2_model(
        &plan,
        &mut reader,
        &mut companions,
        &mut spool,
        &mut sink,
        recorder(),
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("manifest storage lower bound exceeds receiving cap")
    );
    assert!(spool.get_ref().is_empty());
    assert!(!out.exists());
}

struct ChangingCompanion {
    inner: InMemoryWeightSource,
    reads: usize,
    change_on: usize,
    legacy_path: Option<std::path::PathBuf>,
}
impl WeightSource for ChangingCompanion {
    fn inventory(&self) -> Result<Vec<QuantSourceTensor>, QuantEngineError> {
        self.inner.inventory()
    }
    fn asset_names(&self) -> Vec<String> {
        self.inner.asset_names()
    }
    fn tensor(&mut self, name: &str, cap: usize) -> Result<Cow<'_, [u8]>, QuantEngineError> {
        self.reads += 1;
        if self.reads == self.change_on {
            if let Some(path) = &self.legacy_path {
                let mut file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(path)
                    .unwrap();
                let mut byte = [0];
                file.read_exact(&mut byte).unwrap();
                byte[0] ^= 1;
                file.seek(SeekFrom::Start(0)).unwrap();
                file.write_all(&byte).unwrap();
            } else {
                self.inner.tensors.get_mut(name).unwrap().bytes[..4]
                    .copy_from_slice(&2.0f32.to_le_bytes());
            }
        }
        self.inner.tensor(name, cap)
    }
    fn asset(&mut self, name: &str, cap: usize) -> Result<Cow<'_, [u8]>, QuantEngineError> {
        self.inner.asset(name, cap)
    }
    fn input_gram(&mut self, name: &str, cap: usize) -> Result<Cow<'_, [f64]>, QuantEngineError> {
        self.inner.input_gram(name, cap)
    }
}
#[test]
fn actual_companion_and_original_file_mutations_preserve_typed_refusals_and_clean_output() {
    let root = Scratch::new("mutation");
    for case in 0..4 {
        let path = root.0.join(format!("{case}.tsalt2"));
        fs::write(
            &path,
            legacy(
                SaltV2Codec::B3,
                vec![tensor("weight", vec![2, 256], SaltV2Transform::None)],
            ),
        )
        .unwrap();
        let mut reader = SaltV2PackageReader::new_strict(File::open(&path).unwrap()).unwrap();
        let mut companions = ChangingCompanion {
            inner: originals(ScalePrecision::F32),
            reads: 0,
            change_on: if case == 1 || case == 3 { 3 } else { 2 },
            legacy_path: (case >= 2).then_some(path),
        };
        let plan = request(&companions, PlaneCodec::B3, 128);
        let mut spool = Cursor::new(Vec::new());
        let out = root.0.join(format!("{case}.out"));
        let mut sink = DirectoryTensorSink::new(&out);
        let error = import_salt_v2_model(
            &plan,
            &mut reader,
            &mut companions,
            &mut spool,
            &mut sink,
            recorder(),
        )
        .unwrap_err();
        match case {
            0 => assert!(matches!(
                error,
                SaltV2ModelImportError::Companion(QuantEngineError::Binding(
                    "observed tensor content changed"
                ))
            )),
            1 => assert!(matches!(
                error,
                SaltV2ModelImportError::Publication(TritModelPublicationError::Staging(
                    TritModelStageError::Guard(SaltV2ModelGuardError::Companion(
                        QuantEngineError::Binding("observed tensor content changed")
                    ))
                ))
            )),
            _ => assert!(matches!(
                error,
                SaltV2ModelImportError::Publication(TritModelPublicationError::Staging(
                    TritModelStageError::Guard(SaltV2ModelGuardError::Legacy(
                        SaltV2PackageReadError::PackageChanged
                    ))
                ))
            )),
        }
        assert!(!out.exists());
    }
}

#[derive(serde::Serialize)]
#[serde(transparent)]
struct RawImport(serde_json::Value);
impl EvidenceEvent for RawImport {
    const SCHEMA_ID: &'static str = "tritium.migrate.legacy_model_import_prepared";
    const VERSION: u16 = 1;
}
#[derive(serde::Serialize)]
#[serde(transparent)]
struct FutureImport<'a>(&'a LegacyModelImportPrepared);
impl EvidenceEvent for FutureImport<'_> {
    const SCHEMA_ID: &'static str = "tritium.migrate.legacy_model_import_prepared";
    const VERSION: u16 = 2;
}
#[test]
fn independently_resealed_bad_import_bindings_or_fit_logs_cannot_replace_preparation() {
    let root = Scratch::new("replay");
    let mut reader = SaltV2PackageReader::new_strict(Cursor::new(legacy(
        SaltV2Codec::B3,
        vec![tensor("weight", vec![2, 256], SaltV2Transform::None)],
    )))
    .unwrap();
    let mut companions = originals(ScalePrecision::F32);
    let plan = request(&companions, PlaneCodec::B3, 128);
    let mut result = import_salt_v2_model(
        &plan,
        &mut reader,
        &mut companions,
        &mut Cursor::new(Vec::new()),
        &mut DirectoryTensorSink::new(root.0.join("out")),
        recorder(),
    )
    .unwrap();
    let valid = LegacyModelImportView::fold(result.evidence.log())
        .unwrap()
        .remove(0)
        .observation()
        .clone();
    for refusal in 0..5 {
        let mut bad = valid.clone();
        match refusal {
            0 => bad.companion_source = OriginalSourceId::from_bytes([0; 32]),
            1 => bad.legacy_id = "wrong-original".into(),
            2 => {
                bad.manifest.levels[0]
                    .tensors
                    .iter_mut()
                    .find(|t| t.name == "norm")
                    .unwrap()
                    .storage = TensorStorage::Dense {
                    shape: vec![1, 2],
                    precision: ScalePrecision::F32,
                }
            }
            3 => bad.manifest.assets[0].blob.id = BlobId::from_bytes([0; 32]),
            _ => {
                bad.manifest.levels[0].fit = LevelFit::ImportedPrefixOf {
                    level: "imported".into(),
                }
            }
        }
        let mut rec = recorder();
        rec.emit("migration", None, 0, &bad).unwrap();
        let resealed = rec.seal().unwrap(); // chains/root genuinely valid, typed bindings false
        assert!(LegacyModelImportView::fold(resealed.log()).is_err());
        assert!(verify_imported_model_with_log(&mut result.model, resealed.log()).is_err());
    }
    let mut future = EvidenceRecorder::new(
        "future",
        [EventRegistration {
            schema: FutureImport::SCHEMA_ID,
            version: 2,
        }],
    )
    .unwrap();
    future
        .emit("migration", None, 0, &FutureImport(&valid))
        .unwrap();
    assert!(LegacyModelImportView::fold(future.seal().unwrap().log()).is_err());
    let mut value = serde_json::to_value(&valid).unwrap();
    value["unknown"] = true.into();
    let mut rec = EvidenceRecorder::new(
        "raw",
        [EventRegistration {
            schema: RawImport::SCHEMA_ID,
            version: 1,
        }],
    )
    .unwrap();
    rec.emit("migration", None, 0, &RawImport(value)).unwrap();
    assert!(LegacyModelImportView::fold(rec.seal().unwrap().log()).is_err());
    let ordinary = Engine::run(
        &plan.companions,
        &mut companions,
        &mut MemoryTensorSink::default(),
        EvidenceRecorder::new(
            "ordinary-conversion",
            [
                tritium_evidence::DIRECT_LEVELS_FITTED_REGISTRATION,
                tritium_evidence::QUANTIZATION_STARTED_REGISTRATION,
                tritium_evidence::QUANTIZATION_COMPLETED_REGISTRATION,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    assert!(verify_imported_model_with_log(&mut result.model, ordinary.evidence.log()).is_err());
}

struct ShortIo<R>(R);
impl<R: Read> Read for ShortIo<R> {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        let length = bytes.len().min(3);
        self.0.read(&mut bytes[..length])
    }
}
impl<R: Write> Write for ShortIo<R> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.write(&bytes[..bytes.len().min(3)])
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()
    }
}
impl<R: Seek> Seek for ShortIo<R> {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.0.seek(position)
    }
}
#[test]
fn actual_safetensors_companions_and_short_file_io_preserve_shapes_assets_and_no_clobber() {
    let root = Scratch::new("safetensors");
    let expected = originals(ScalePrecision::F32);
    let header = serde_json::to_vec(
        &serde_json::json!({"norm": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}),
    )
    .unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&expected.tensors["norm"].bytes);
    let raw = root.0.join("preserved.safetensors");
    fs::write(&raw, bytes).unwrap();
    let mut companions = SafeTensorsWeightSource::new(
        [File::open(raw).unwrap()],
        expected.assets.clone(),
        BTreeMap::new(),
    )
    .unwrap();
    let plan = request(&companions, PlaneCodec::B3, 128);
    let source = tensor("weight", vec![2, 256], SaltV2Transform::None);
    let original_path = root.0.join("original.tsalt2");
    fs::write(
        &original_path,
        legacy(SaltV2Codec::B3, vec![source.clone()]),
    )
    .unwrap();
    let mut reader =
        SaltV2PackageReader::new_strict(ShortIo(File::open(original_path).unwrap())).unwrap();
    let mut spool = ShortIo(
        OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(root.0.join("pending.spool"))
            .unwrap(),
    );
    let out = root.0.join("out");
    let mut sink = DirectoryTensorSink::new(&out);
    let mut result = import_salt_v2_model(
        &plan,
        &mut reader,
        &mut companions,
        &mut spool,
        &mut sink,
        recorder(),
    )
    .unwrap();
    check_tensor(&mut result.model, &source);
    verify_imported_model_with_log(&mut result.model, result.evidence.log()).unwrap();
    let retained = fs::read(out.join("model.trit")).unwrap();
    let log = fs::read_to_string(out.join("evidence.jsonl")).unwrap();
    let mut retry = Cursor::new(Vec::new());
    assert!(matches!(
        import_salt_v2_model(
            &plan,
            &mut reader,
            &mut companions,
            &mut retry,
            &mut sink,
            recorder()
        ),
        Err(SaltV2ModelImportError::Publication(
            TritModelPublicationError::Publication(QuantEngineError::DestinationExists)
        ))
    ));
    assert_eq!(fs::read(out.join("model.trit")).unwrap(), retained);
    assert_eq!(fs::read_to_string(out.join("evidence.jsonl")).unwrap(), log);
}

struct WriteZeroSpool(Cursor<Vec<u8>>);
impl Read for WriteZeroSpool {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(bytes)
    }
}
impl Seek for WriteZeroSpool {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.0.seek(position)
    }
}
impl Write for WriteZeroSpool {
    fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
        Ok(0)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
#[test]
fn original_spool_write_zero_error_never_creates_published_output() {
    let root = Scratch::new("io");
    let mut reader = SaltV2PackageReader::new_strict(Cursor::new(legacy(
        SaltV2Codec::B3,
        vec![tensor("weight", vec![2, 256], SaltV2Transform::None)],
    )))
    .unwrap();
    let mut companions = originals(ScalePrecision::F32);
    let plan = request(&companions, PlaneCodec::B3, 128);
    let out = root.0.join("out");
    let result = import_salt_v2_model(
        &plan,
        &mut reader,
        &mut companions,
        &mut WriteZeroSpool(Cursor::new(Vec::new())),
        &mut DirectoryTensorSink::new(&out),
        recorder(),
    );
    assert!(
        matches!(result, Err(SaltV2ModelImportError::Tensor(tritium_format::SaltTensorSpoolError::Io(ref error))) if error.kind() == std::io::ErrorKind::WriteZero)
    );
    assert!(!out.exists());
}
