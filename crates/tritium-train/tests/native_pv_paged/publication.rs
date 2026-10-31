//! Complete file-backed paged record/replay/publication/audit, not model qualification.
use super::*;
use tritium_evidence::*;
use tritium_format::trit_model::{TritModelReader, encode_model_manifest};
use tritium_format::{TritBlob, TritBlobKind, TritPackageReadLimits, write_trit_package};
use tritium_quantize::engine::DirectoryTensorSink;
use tritium_train::pv_tuning::*;

fn receiving() -> TritPackageReadLimits {
    TritPackageReadLimits {
        max_source_bytes: 128 << 20,
        max_manifest_bytes: 1 << 20,
        max_directory_bytes: 1 << 20,
    }
}
fn owner_limits() -> NativeRefinementOwnerReadLimits {
    NativeRefinementOwnerReadLimits {
        max_working_bytes: 2 << 20,
        extra_payload_bytes: 0,
    }
}
fn paging() -> Limits {
    Limits {
        max_resident_payload_bytes: 32 << 20,
    }
}
fn recording() -> RecordingLimits {
    RecordingLimits {
        max_journal_bytes: 128 << 20,
        max_frame_bytes: 128 << 20,
        max_working_bytes: 16384,
    }
}
fn replaying() -> NativePvPagedReplayLimits {
    NativePvPagedReplayLimits {
        max_journal_bytes: 128 << 20,
        max_working_bytes: 16384,
        paging: paging(),
    }
}
pub(super) fn original(
    files: &Files,
) -> (
    TritModelReader<File>,
    NativeRefinementParentView,
    NativeRefinementPlan,
) {
    let law = ADMITTED_LAWS[3];
    let layout = AdditiveLayout {
        rows: 1024,
        cols: 260,
        tile: 256,
        group: 128,
        max_planes: law.max_planes,
        allocation: PlaneAllocation::Uniform,
        codec: PlaneCodec::D2,
        law: law.law,
        basis: Basis::Identity,
        transport: Transport::Raw,
    };
    let geometry = UniformPlan::new(layout, law.max_planes).unwrap();
    let weight = AdditiveTensor::new(
        layout,
        law.max_planes,
        (0..geometry.trit_count())
            .map(|i| [Trit::ZERO, Trit::NEG, Trit::POS, Trit::NEG][i % 4])
            .collect(),
        vec![0.5; geometry.scale_count()],
    )
    .unwrap();
    let payload = weight.encode_payload().unwrap();
    let norm = 0.25f32.to_le_bytes();
    let config = b"{}";
    let tokenizer = b"{ }";
    let id = |b: &[u8]| BlobId::from_bytes(*blake3::hash(b).as_bytes());
    let reference = |b: &[u8]| BlobReference {
        id: id(b),
        shard: None,
    };
    let mut p = plan(8);
    p.training.truncate(1);
    let mut tensors = vec![TensorMeta {
        name: "norm".into(),
        blob: reference(&norm),
        storage: TensorStorage::Dense {
            shape: vec![1],
            precision: ScalePrecision::F32,
        },
    }];
    tensors.extend(p.owners.iter().map(|name| TensorMeta {
        name: name.clone(),
        blob: reference(&payload),
        storage: TensorStorage::Additive {
            layout,
            planes: PlaneCounts::Uniform(law.max_planes),
        },
    }));
    let manifest = ModelManifest::imported(
        "synthetic-paged-file".into(),
        "parent".into(),
        "legacy:synthetic".into(),
        vec![
            ManifestAsset {
                name: "config.json".into(),
                blob: reference(config),
            },
            ManifestAsset {
                name: "tokenizer.json".into(),
                blob: reference(tokenizer),
            },
        ],
        tensors,
    );
    let companions = vec![QuantSourceBinding {
        source: QuantSourceTensor {
            name: "norm".into(),
            shape: vec![1],
            precision: ScalePrecision::F32,
        },
        content: id(&norm),
    }];
    let assets = vec![
        NamedAssetDigest {
            name: "config.json".into(),
            content: id(config),
        },
        NamedAssetDigest {
            name: "tokenizer.json".into(),
            content: id(tokenizer),
        },
    ];
    let source = original_source_id(&manifest.architecture, &companions, &assets).unwrap();
    let event = LegacyModelImportBoundView::prepare(
        &manifest,
        "legacy:synthetic".into(),
        source,
        companions,
        assets,
    )
    .unwrap();
    let mut record = EvidenceRecorder::new(
        "synthetic-file-parent",
        [LEGACY_MODEL_IMPORT_BOUND_PREPARED_REGISTRATION],
    )
    .unwrap();
    record.emit("import", None, 0, &event).unwrap();
    let record = record.seal().unwrap();
    let bytes = write_trit_package(
        &encode_model_manifest(&manifest).unwrap(),
        &[
            TritBlob {
                kind: TritBlobKind::DenseTensor(tritium_core::DType::F32),
                bytes: &norm,
            },
            TritBlob {
                kind: TritBlobKind::AdditiveTensor,
                bytes: &payload,
            },
            TritBlob {
                kind: TritBlobKind::Asset("config.json".into()),
                bytes: config,
            },
            TritBlob {
                kind: TritBlobKind::Asset("tokenizer.json".into()),
                bytes: tokenizer,
            },
        ],
    )
    .unwrap();
    std::fs::write(files.0.join("parent.trit"), bytes).unwrap();
    let mut reader = TritModelReader::open(
        File::open(files.0.join("parent.trit")).unwrap(),
        receiving(),
    )
    .unwrap();
    let view =
        NativeRefinementParentView::from_import(&mut reader, "parent", record.log()).unwrap();
    p.parent = view.binding().clone();
    p.limits = NativeRefinementLimits {
        max_state_bytes: 1 << 30,
        max_workspace_bytes: 1 << 30,
        max_checkpoint_bytes: 1 << 30,
        max_package_bytes: 128 << 20,
    };
    (reader, view, p)
}

#[test]
fn complete_file_backed_pipeline_publishes_and_numerically_reaudits_with_owner_bounded_heap() {
    let _serial = SERIAL.lock().unwrap();
    rayon::join(|| (), || ());
    let files = Files::new();
    let (mut parent, view, p) = original(&files);
    let mut initial =
        NativeRefinementOwnerSource::with(&mut parent, &view, &p, owner_limits(), |source| {
            Paged::initialize(
                p.clone(),
                paging(),
                |i, n| source.load(i, n),
                files.create("initial.tnpv"),
            )
        })
        .unwrap();
    let mut writer =
        PagedWriter::new(files.create("gradients.tnpj"), &mut initial, recording()).unwrap();
    let ready =
        NativeRefinementOwnerSource::with(&mut parent, &view, &p, owner_limits(), |source| {
            initial.stage_recorded(
                (files.create("next.tnpv"), files.create("frame.tmp")),
                |i, n| source.load(i, n),
                |i, _, w| {
                    let n = w.view().plane_trits(0).unwrap().len();
                    Ok((
                        vec![0.125 + i as f32 * 0.01; w.scales().len()],
                        vec![0.125; n],
                    ))
                },
                |_, _, _| Ok(()),
                recording(),
            )
        })
        .unwrap();
    let (mut next, _, _) = ready.commit(&mut initial, &mut writer).unwrap();
    let (journal, summary) = writer.finish(&mut next).unwrap();
    drop(journal);
    drop(initial);
    std::fs::remove_file(files.0.join("initial.tnpv")).unwrap();
    std::fs::remove_file(files.0.join("frame.tmp")).unwrap();
    let mut expected = File::open(files.0.join("next.tnpv")).unwrap();
    let (mut current, proof) =
        NativeRefinementOwnerSource::with(&mut parent, &view, &p, owner_limits(), |source| {
            replay_native_pv_paged_journal(
                NativePvPagedJournalReplay {
                    plan: p.clone(),
                    source: File::open(files.0.join("gradients.tnpj")).unwrap(),
                    expected_journal: summary.journal(),
                    checkpoint: &mut expected,
                },
                |i, n| source.load(i, n),
                |i| Ok(files.create(&format!("replay-{i}.tnpv"))),
                replaying(),
            )
        })
        .unwrap();
    drop(expected);
    drop(next);
    std::fs::remove_file(files.0.join("next.tnpv")).unwrap();
    std::fs::remove_file(files.0.join("replay-0.tnpv")).unwrap();
    let checkpoint_bytes = current.size_ledger().logical.checkpoint_bytes;
    let validation: Vec<_> = p
        .validation
        .iter()
        .map(|w| NativeRefinementValidation {
            content: w.content,
            parent_loss: 1.0,
            child_loss: 1.0,
        })
        .collect();
    let out = files.0.join("published");
    let mut spool = files.create("publisher.spool");
    let cap = NativePvPagedChildLimits {
        max_spool_bytes: 128 << 20,
        max_working_bytes: 2 << 20,
        max_evidence_bytes: 1 << 20,
        max_companion_bytes: 128 << 20,
        owners: owner_limits(),
        receiving: receiving(),
    };
    let (result, peak, retained) = measured(|| {
        let recorder = EvidenceRecorder::new(
            "synthetic-file-child",
            [
                NATIVE_REFINEMENT_STARTED_REGISTRATION,
                NATIVE_REFINEMENT_CHECKPOINTED_REGISTRATION,
                NATIVE_REFINEMENT_COMPLETED_REGISTRATION,
                NATIVE_PV_REPLAY_OBSERVED_REGISTRATION,
            ],
        )
        .unwrap();
        publish_native_pv_paged_child(
            &mut current,
            NativePvChildSource {
                parent: &mut parent,
                parent_evidence: &view,
                replay: &proof,
                journal: &mut File::open(files.0.join("gradients.tnpj")).unwrap(),
                machine: "synthetic-file-fixture",
                validation: &validation,
            },
            &mut spool,
            &mut DirectoryTensorSink::new(&out),
            cap,
            recorder,
        )
    });
    let mut child = result.unwrap();
    assert!(
        (peak as u64) < checkpoint_bytes,
        "publisher buffered checkpoint {peak}"
    );
    assert!(
        peak as u64
            <= child.sizes.owner_resident_payload_bytes
                + child.sizes.owner_decode_working_bytes
                + child.sizes.working_payload_bytes
                + (1 << 20)
    );
    assert!(retained < (1 << 20));
    assert!(child.warnings.is_empty());
    assert_eq!(
        std::fs::metadata(out.join(NATIVE_PV_CHECKPOINT_FILE))
            .unwrap()
            .len(),
        checkpoint_bytes
    );
    drop(current);
    drop(spool);
    for name in ["replay-1.tnpv", "gradients.tnpj", "publisher.spool"] {
        std::fs::remove_file(files.0.join(name)).unwrap();
    }
    let checked = verify_native_pv_paged_child(
        NativePvPagedChildVerification {
            parent: &mut parent,
            parent_evidence: &view,
            child: &mut child.model,
            log: child.evidence.log(),
            plan: &p,
            journal: &mut File::open(out.join(NATIVE_PV_JOURNAL_FILE)).unwrap(),
            checkpoint: &mut File::open(out.join(NATIVE_PV_CHECKPOINT_FILE)).unwrap(),
        },
        NativePvPagedChildVerificationLimits {
            owners: owner_limits(),
            replay: replaying(),
            max_encoded_owner_bytes: 2 << 20,
        },
        |i| Ok::<_, &str>(files.create(&format!("audit-{i}.tnpv"))),
    )
    .unwrap();
    assert_eq!(checked.package(), child.model.package_id());
    assert_eq!(checked.replay().summary(), proof.summary());
    for i in 0..=p.training.len() {
        std::fs::remove_file(files.0.join(format!("audit-{i}.tnpv"))).unwrap();
    }
    eprintln!(
        "PAGED_PUBLICATION owners=8 checkpoint_bytes={checkpoint_bytes} spool_bytes={} publisher_working_bytes={} peak_live_allocator_requests={peak} retained_allocator_requests={retained}",
        child.sizes.spool_payload_bytes, child.sizes.working_payload_bytes
    );
}
