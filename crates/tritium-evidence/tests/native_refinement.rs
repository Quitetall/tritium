//! Synthetic recorded lifecycle with real typed packages. No recovery/quality run.
use std::io::Cursor;
use tritium_core::{DType, Trit};
use tritium_evidence::*;
use tritium_format::trit_model::{TritModelReader, decode_model_manifest, encode_model_manifest};
use tritium_format::{
    AdditiveTensor, TritBlob, TritBlobKind, TritPackageReadLimits, write_trit_package,
};
use tritium_schema::*;

fn id(bytes: &[u8]) -> BlobId {
    BlobId::from_bytes(*blake3::hash(bytes).as_bytes())
}
fn reference(bytes: &[u8]) -> BlobReference {
    BlobReference {
        id: id(bytes),
        shard: None,
    }
}
const CONFIG: &[u8] = b"{\"fixture\":1}";
const TOKENIZER: &[u8] = b"{\"fixture\":2}";
fn payload(scale: f32) -> (TensorStorage, Vec<u8>) {
    let layout = AdditiveLayout {
        rows: 1,
        cols: 4,
        tile: 256,
        group: 32,
        max_planes: 1,
        allocation: PlaneAllocation::Uniform,
        codec: PlaneCodec::D2,
        law: ADMITTED_LAWS[1].law,
        basis: Basis::Identity,
        transport: Transport::Raw,
    };
    let tensor = AdditiveTensor::new(
        layout,
        1,
        vec![Trit::POS, Trit::NEG, Trit::ZERO, Trit::POS],
        vec![scale],
    )
    .unwrap();
    (
        TensorStorage::Additive {
            layout,
            planes: PlaneCounts::Uniform(1),
        },
        tensor.encode_payload().unwrap(),
    )
}
fn package(manifest: &ModelManifest, weight: &[u8]) -> Vec<u8> {
    let bias = 0.25f32.to_le_bytes();
    write_trit_package(
        &encode_model_manifest(manifest).unwrap(),
        &[
            TritBlob {
                kind: TritBlobKind::AdditiveTensor,
                bytes: weight,
            },
            TritBlob {
                kind: TritBlobKind::DenseTensor(DType::F32),
                bytes: &bias,
            },
            TritBlob {
                kind: TritBlobKind::Asset("config.json".into()),
                bytes: CONFIG,
            },
            TritBlob {
                kind: TritBlobKind::Asset("tokenizer.json".into()),
                bytes: TOKENIZER,
            },
        ],
    )
    .unwrap()
}
fn reader(bytes: Vec<u8>) -> TritModelReader<Cursor<Vec<u8>>> {
    TritModelReader::open(
        Cursor::new(bytes),
        TritPackageReadLimits {
            max_source_bytes: 100_000,
            max_manifest_bytes: 50_000,
            max_directory_bytes: 10_000,
        },
    )
    .unwrap()
}
struct Fixture {
    reader: TritModelReader<Cursor<Vec<u8>>>,
    parent: NativeRefinementParentView,
    plan: NativeRefinementPlan,
    updated: Vec<u8>,
    bytes: Vec<u8>,
    original: SealedEvidence,
}
impl Fixture {
    fn new() -> Self {
        let (storage, weight) = payload(0.5);
        let bias = 0.25f32.to_le_bytes();
        let manifest = ModelManifest::imported(
            "synthetic-only".into(),
            "parent".into(),
            "legacy:fixture".into(),
            vec![
                ManifestAsset {
                    name: "config.json".into(),
                    blob: reference(CONFIG),
                },
                ManifestAsset {
                    name: "tokenizer.json".into(),
                    blob: reference(TOKENIZER),
                },
            ],
            vec![
                TensorMeta {
                    name: "bias".into(),
                    blob: reference(&bias),
                    storage: TensorStorage::Dense {
                        shape: vec![1],
                        precision: ScalePrecision::F32,
                    },
                },
                TensorMeta {
                    name: "weight".into(),
                    blob: reference(&weight),
                    storage,
                },
            ],
        );
        let sources = vec![QuantSourceBinding {
            source: QuantSourceTensor {
                name: "bias".into(),
                shape: vec![1],
                precision: ScalePrecision::F32,
            },
            content: id(&bias),
        }];
        let assets = vec![
            NamedAssetDigest {
                name: "config.json".into(),
                content: id(CONFIG),
            },
            NamedAssetDigest {
                name: "tokenizer.json".into(),
                content: id(TOKENIZER),
            },
        ];
        let source = original_source_id(&manifest.architecture, &sources, &assets).unwrap();
        let prepared = LegacyModelImportBoundView::prepare(
            &manifest,
            "legacy:fixture".into(),
            source,
            sources,
            assets,
        )
        .unwrap();
        let mut recorder = EvidenceRecorder::new(
            "original-import",
            [LEGACY_MODEL_IMPORT_BOUND_PREPARED_REGISTRATION],
        )
        .unwrap();
        recorder.emit("import", None, 0, &prepared).unwrap();
        let original = recorder.seal().unwrap();
        let bytes = package(&manifest, &weight);
        let mut reader = reader(bytes.clone());
        let parent =
            NativeRefinementParentView::from_import(&mut reader, "parent", original.log()).unwrap();
        let plan = NativeRefinementPlan {
            parent: parent.binding().clone(),
            child_level: "child".into(),
            owners: vec!["weight".into()],
            recipe: NativeRecoveryRecipe::ScaleOnly {
                optimizer: NativeAdamWRecipe {
                    lr: 0.001,
                    beta1: 0.9,
                    beta2: 0.99,
                    eps: 1e-8,
                    weight_decay: 0.0,
                },
            },
            objective: [3; 32],
            teacher_source: None,
            training: vec![
                NativeRefinementWindow {
                    content: [4; 32],
                    weight: 1.0,
                },
                NativeRefinementWindow {
                    content: [5; 32],
                    weight: 0.5,
                },
            ],
            validation: vec![NativeRefinementWindow {
                content: [6; 32],
                weight: 1.0,
            }],
            limits: NativeRefinementLimits {
                max_state_bytes: 1024,
                max_workspace_bytes: 1024,
                max_checkpoint_bytes: 4096,
                max_package_bytes: 100_000,
            },
        };
        Self {
            reader,
            parent,
            plan,
            updated: payload(0.75).1,
            bytes,
            original,
        }
    }
    fn checkpoint(&self, step: u64) -> NativeRefinementCheckpointed {
        // Synthetic bytes deliberately are NOT a native optimizer checkpoint.
        let raw = step.to_le_bytes();
        let checkpoint = id(&raw);
        let plan = native_refinement_plan_id(&self.plan).unwrap();
        NativeRefinementCheckpointed {
            plan,
            completed_updates: step,
            state: native_refinement_state_id(plan, step, checkpoint),
            checkpoint,
            checkpoint_bytes: 8,
            state_payload_bytes: 64,
            owners: vec![NativeRefinementTensorOutput {
                name: "weight".into(),
                content: id(&self.updated),
            }],
        }
    }
    fn completed(&self) -> NativeRefinementCompleted {
        NativeRefinementCompleted {
            plan: native_refinement_plan_id(&self.plan).unwrap(),
            checkpoint: self.checkpoint(2).checkpoint,
            tensors: self.reader.manifest().levels[0]
                .tensors
                .iter()
                .map(|t| NativeRefinementTensorOutput {
                    name: t.name.clone(),
                    content: if t.name == "weight" {
                        id(&self.updated)
                    } else {
                        t.blob.id
                    },
                })
                .collect(),
            validation: vec![NativeRefinementValidation {
                content: [6; 32],
                parent_loss: 1.0,
                child_loss: 0.9,
            }],
        }
    }
    fn fold(
        &mut self,
        log: &VerifiedLog,
    ) -> Result<Vec<NativeRefinementReceiptView>, EvidenceError> {
        NativeRefinementReceiptView::fold(log, &mut self.reader, &self.parent, &self.plan)
    }
}
fn recorder() -> EvidenceRecorder {
    EvidenceRecorder::new(
        "recorded-only-recovery",
        [
            NATIVE_REFINEMENT_STARTED_REGISTRATION,
            NATIVE_REFINEMENT_CHECKPOINTED_REGISTRATION,
            NATIVE_REFINEMENT_COMPLETED_REGISTRATION,
            QUANTIZATION_COMPLETED_REGISTRATION,
        ],
    )
    .unwrap()
}
fn started(plan: &NativeRefinementPlan) -> NativeRefinementStarted {
    NativeRefinementStarted {
        plan: plan.clone(),
        execution: NativeRefinementExecution {
            backend: "synthetic-not-executed".into(),
            device: "not-a-physical-device".into(),
            source_revision: "fixture:no-numerical-run".into(),
        },
    }
}
fn record(
    plan: &NativeRefinementPlan,
    checkpoints: &[NativeRefinementCheckpointed],
    completed: Option<&NativeRefinementCompleted>,
) -> SealedEvidence {
    let mut r = recorder();
    r.emit("recover", None, 0, &started(plan)).unwrap();
    for (i, c) in checkpoints.iter().enumerate() {
        r.emit("recover", None, i as u64 + 1, c).unwrap();
    }
    if let Some(c) = completed {
        r.emit("recover", None, checkpoints.len() as u64 + 1, c)
            .unwrap();
    }
    r.seal().unwrap()
}

#[test]
fn complete_record_builds_one_distinct_refined_child_and_chains_parent_admission() {
    let mut f = Fixture::new();
    let log = record(
        &f.plan,
        &[f.checkpoint(1), f.checkpoint(2)],
        Some(&f.completed()),
    );
    let view = f.fold(log.log()).unwrap().remove(0);
    assert_eq!(view.reference().root, log.root());
    assert_eq!(view.checkpoint().completed_updates, 2);
    let child = view.manifest();
    assert_eq!(child.schema, SchemaId { major: 1, minor: 1 });
    assert_eq!(child.levels.len(), 1);
    assert_eq!(child.assets, f.reader.manifest().assets);
    assert_eq!(
        child.levels[0].tensors[0],
        f.reader.manifest().levels[0].tensors[0]
    );
    assert_eq!(
        child.levels[0].tensors[1].storage,
        f.reader.manifest().levels[0].tensors[1].storage
    );
    assert!(matches!(child.levels[0].fit, LevelFit::Refined { .. }));
    assert!(QuantizationReceiptView::fold(log.log()).unwrap().is_empty());
    assert_eq!(
        decode_model_manifest(&encode_model_manifest(child).unwrap()).unwrap(),
        *child
    );
    let mut child_reader = reader(package(child, &f.updated));
    let next = NativeRefinementParentView::from_refined(&mut child_reader, "child", &view).unwrap();
    assert_ne!(next.binding().package, f.parent.binding().package);
    assert_eq!(next.binding().source, f.parent.binding().source);
    assert_ne!(next.binding().model, f.parent.binding().model);
}

#[test]
fn unfinished_records_never_produce_a_complete_view() {
    let mut f = Fixture::new();
    for checkpoints in [vec![], vec![f.checkpoint(1)], vec![f.checkpoint(2)]] {
        let log = record(&f.plan, &checkpoints, None);
        assert!(f.fold(log.log()).unwrap().is_empty());
    }
    let no_checkpoint = record(&f.plan, &[], Some(&f.completed()));
    assert!(f.fold(no_checkpoint.log()).is_err());
    let early = record(&f.plan, &[f.checkpoint(1)], Some(&f.completed()));
    assert!(f.fold(early.log()).is_err());
}

#[test]
fn frozen_request_rejects_changed_parent_level_source_recipe_and_data() {
    let mut f = Fixture::new();
    for change in 0..8 {
        let mut request = f.plan.clone();
        match change {
            0 => request.parent.package = PackageId::from_bytes([71; 32]),
            1 => request.parent.model = ModelId::from_bytes([72; 32]),
            2 => request.parent.level = "other".into(),
            3 => request.parent.source = OriginalSourceId::from_bytes([73; 32]),
            4 => request.parent.evidence.root = [74; 32],
            5 => request.objective = [75; 32],
            6 => request.training.reverse(),
            _ => {
                let NativeRecoveryRecipe::ScaleOnly { optimizer } = &mut request.recipe else {
                    panic!()
                };
                optimizer.lr = 0.02;
            }
        }
        let log = record(&request, &[f.checkpoint(2)], Some(&f.completed()));
        assert!(f.fold(log.log()).is_err(), "mutation {change}");
    }
}

#[test]
fn checkpoint_completion_and_validation_contradictions_reject() {
    let mut f = Fixture::new();
    for change in 0..10 {
        let mut checkpoint = f.checkpoint(2);
        let mut completed = f.completed();
        match change {
            0 => checkpoint.owners.clear(),
            1 => checkpoint.state = NativeRefinementStateId::from_bytes([7; 32]),
            2 => checkpoint.checkpoint_bytes = 4097,
            3 => checkpoint.state_payload_bytes = 1025,
            4 => checkpoint.completed_updates = 3,
            5 => completed.checkpoint = BlobId::from_bytes([8; 32]),
            6 => completed.tensors[0].content = BlobId::from_bytes([9; 32]),
            7 => completed.tensors.pop().map(|_| ()).unwrap(),
            8 => completed.validation[0].content = [10; 32],
            _ => completed.validation.clear(),
        }
        let log = record(&f.plan, &[checkpoint], Some(&completed));
        assert!(f.fold(log.log()).is_err(), "mutation {change}");
    }
    let repeated = record(
        &f.plan,
        &[f.checkpoint(1), f.checkpoint(1)],
        Some(&f.completed()),
    );
    assert!(f.fold(repeated.log()).is_err());
}

#[test]
fn request_validation_and_manifest_versions_are_fail_closed() {
    let mut f = Fixture::new();
    for change in 0..7 {
        let mut plan = f.plan.clone();
        match change {
            0 => plan.owners.clear(),
            1 => plan.owners.push("weight".into()),
            2 => plan.owners = vec!["bias".into()],
            3 => plan.validation[0].content = plan.training[0].content,
            4 => plan.training[0].weight = 0.0,
            5 => plan.limits.max_checkpoint_bytes = 0,
            _ => {
                let NativeRecoveryRecipe::ScaleOnly { optimizer } = &mut plan.recipe else {
                    panic!()
                };
                optimizer.beta1 = -0.0;
            }
        }
        assert!(
            plan.validate_parent(f.reader.manifest(), f.parent.binding())
                .is_err()
        );
    }
    let log = record(&f.plan, &[f.checkpoint(2)], Some(&f.completed()));
    let view = f.fold(log.log()).unwrap().remove(0);
    let child = view.manifest();
    let mut old = child.clone();
    old.schema.minor = 0;
    assert!(old.validate().is_err());
    let mut multi = child.clone();
    let mut extra = multi.levels[0].clone();
    extra.name = "z-other".into();
    multi.levels.push(extra);
    assert!(multi.validate().is_err());
    let mut relabeled = child.clone();
    relabeled.levels[0].fit = LevelFit::Imported;
    assert!(relabeled.validate().is_err());
    let mut future = child.clone();
    future.schema.minor = 2;
    assert!(future.validate().is_err());
    let mut unknown = serde_json::to_value(&f.plan).unwrap();
    unknown["unexpected"] = true.into();
    assert!(serde_json::from_value::<NativeRefinementPlan>(unknown).is_err());
}

#[test]
fn request_id_matches_actual_recorded_jcs_and_noop_model_id_is_not_invented() {
    let mut f = Fixture::new();
    f.updated = payload(0.5).1;
    let log = record(&f.plan, &[f.checkpoint(2)], Some(&f.completed()));
    let payload = &log.log().events[0].payload["plan"];
    let mut h = blake3::Hasher::new();
    h.update(b"tritium.native-refinement.plan.v1\0");
    h.update(&serde_json_canonicalizer::to_vec(payload).unwrap());
    assert_eq!(
        native_refinement_plan_id(&f.plan).unwrap().as_bytes(),
        h.finalize().as_bytes()
    );
    let view = f.fold(log.log()).unwrap().remove(0);
    let mut child = reader(package(view.manifest(), &f.updated));
    let admitted = NativeRefinementParentView::from_refined(&mut child, "child", &view).unwrap();
    assert_eq!(admitted.binding().model, f.parent.binding().model);
    assert_ne!(admitted.binding().package, f.parent.binding().package);
    assert!(!log.jsonl().contains("\"verdict\""));
}

#[derive(serde::Serialize)]
struct WrongLifecycle(NativeRefinementCompleted);
impl EvidenceEvent for WrongLifecycle {
    const SCHEMA_ID: &'static str = "tritium.quantize.quantization_completed";
    const VERSION: u16 = 1;
}
#[test]
fn ptq_label_cannot_stand_in_for_native_completion() {
    let mut f = Fixture::new();
    let mut r = recorder();
    r.emit("recover", None, 0, &started(&f.plan)).unwrap();
    r.emit("recover", None, 1, &f.checkpoint(2)).unwrap();
    r.emit("recover", None, 2, &WrongLifecycle(f.completed()))
        .unwrap();
    assert!(f.fold(r.seal().unwrap().log()).is_err());
    let original = f.original.log().clone();
    assert!(f.fold(&original).unwrap().is_empty());
}

struct MutableSource {
    bytes: std::rc::Rc<std::cell::RefCell<Vec<u8>>>,
    position: u64,
}
impl std::io::Read for MutableSource {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        let bytes = self.bytes.borrow();
        let mut cursor = Cursor::new(bytes.as_slice());
        cursor.set_position(self.position);
        let count = std::io::Read::read(&mut cursor, output)?;
        self.position = cursor.position();
        Ok(count)
    }
}
impl std::io::Seek for MutableSource {
    fn seek(&mut self, position: std::io::SeekFrom) -> std::io::Result<u64> {
        let bytes = self.bytes.borrow();
        let mut cursor = Cursor::new(bytes.as_slice());
        cursor.set_position(self.position);
        self.position = std::io::Seek::seek(&mut cursor, position)?;
        Ok(self.position)
    }
}
#[test]
fn changed_parent_source_and_resealed_parent_reference_do_not_rebind_request() {
    let f = Fixture::new();
    let bytes = std::rc::Rc::new(std::cell::RefCell::new(f.bytes.clone()));
    let mut live = TritModelReader::open(
        MutableSource {
            bytes: bytes.clone(),
            position: 0,
        },
        TritPackageReadLimits {
            max_source_bytes: 100_000,
            max_manifest_bytes: 50_000,
            max_directory_bytes: 10_000,
        },
    )
    .unwrap();
    let admitted =
        NativeRefinementParentView::from_import(&mut live, "parent", f.original.log()).unwrap();
    let log = record(&f.plan, &[f.checkpoint(2)], Some(&f.completed()));
    let end = bytes.borrow().len() - 1;
    bytes.borrow_mut()[end] ^= 1;
    assert!(NativeRefinementReceiptView::fold(log.log(), &mut live, &admitted, &f.plan).is_err());
    let mut original_reader = reader(f.bytes.clone());
    let observation: LegacyModelImportBoundPrepared =
        serde_json::from_value(f.original.log().events[0].payload.clone()).unwrap();
    let mut resealed = EvidenceRecorder::new(
        "different-original-root",
        [LEGACY_MODEL_IMPORT_BOUND_PREPARED_REGISTRATION],
    )
    .unwrap();
    resealed.emit("import", None, 0, &observation).unwrap();
    let resealed = resealed.seal().unwrap();
    let changed =
        NativeRefinementParentView::from_import(&mut original_reader, "parent", resealed.log())
            .unwrap();
    assert_ne!(changed.binding().evidence, f.parent.binding().evidence);
    assert!(
        NativeRefinementReceiptView::fold(log.log(), &mut original_reader, &changed, &f.plan)
            .is_err()
    );
}

#[test]
fn typed_native_recipes_preserve_exact_parameters_and_reject_invalid_bounds() {
    let f = Fixture::new();
    let NativeRecoveryRecipe::ScaleOnly { optimizer } = f.plan.recipe else {
        panic!()
    };
    let mut plan = f.plan.clone();
    plan.recipe = NativeRecoveryRecipe::AlternatingPv {
        continuous_optimizer: optimizer,
        code_optimizer: optimizer,
        max_code_change_fraction: 0.25,
        max_relative_code_change: Some(0.5),
    };
    assert!(plan.validate().is_ok());
    let value = serde_json::to_value(&plan).unwrap();
    let roundtrip: NativeRefinementPlan = serde_json::from_value(value).unwrap();
    assert_eq!(
        native_refinement_plan_id(&plan).unwrap(),
        native_refinement_plan_id(&roundtrip).unwrap()
    );
    for fraction in [0.0, -0.0, -0.1, 1.1, f32::INFINITY, f32::NAN] {
        let mut bad = plan.clone();
        let NativeRecoveryRecipe::AlternatingPv {
            max_code_change_fraction,
            ..
        } = &mut bad.recipe
        else {
            panic!()
        };
        *max_code_change_fraction = fraction;
        assert!(bad.validate().is_err());
    }
    for invalid in 0..5 {
        let mut bad = optimizer;
        match invalid {
            0 => bad.lr = 0.0,
            1 => bad.beta2 = 1.0,
            2 => bad.eps = f32::NAN,
            3 => bad.weight_decay = -0.0,
            _ => bad.weight_decay = 2.0 / bad.lr,
        };
        assert!(bad.validate().is_err());
    }
    let mut repeat = plan.clone();
    repeat.training[1].content = repeat.training[0].content;
    assert!(
        repeat.validate().is_ok(),
        "intentional ordered repetitions are not train/validation overlap"
    );
}

#[derive(serde::Serialize)]
struct ExtraStart(serde_json::Value);
impl EvidenceEvent for ExtraStart {
    const SCHEMA_ID: &'static str = "tritium.refine.native_started";
    const VERSION: u16 = 1;
}
#[test]
fn coherent_unknown_fields_and_events_after_completion_are_rejected() {
    let mut f = Fixture::new();
    let mut r = recorder();
    let mut start = serde_json::to_value(started(&f.plan)).unwrap();
    start["unexpected"] = true.into();
    r.emit("recover", None, 0, &ExtraStart(start)).unwrap();
    assert!(f.fold(r.seal().unwrap().log()).is_err());
    let mut r = recorder();
    r.emit("recover", None, 0, &started(&f.plan)).unwrap();
    r.emit("recover", None, 1, &f.checkpoint(2)).unwrap();
    r.emit("recover", None, 2, &f.completed()).unwrap();
    r.emit("recover", None, 3, &f.checkpoint(2)).unwrap();
    assert!(f.fold(r.seal().unwrap().log()).is_err());
    let mut completed = f.completed();
    completed.validation[0].child_loss = f64::NAN;
    let bad = record(&f.plan, &[f.checkpoint(2)], Some(&completed));
    assert!(f.fold(bad.log()).is_err());
    let mut record_only = f.completed();
    record_only.validation[0].child_loss = 2.0;
    let observed = record(&f.plan, &[f.checkpoint(2)], Some(&record_only));
    assert_eq!(
        f.fold(observed.log()).unwrap()[0].completed().validation[0].child_loss,
        2.0,
        "finite recorded observations are not an invented quality-pass verdict"
    );
}
