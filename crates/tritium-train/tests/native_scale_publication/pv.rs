//! P/V exercises the same guarded publisher with distinct numerical evidence.
use super::*;
struct Scratch(PathBuf);
impl Scratch {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("native-pv-child-{name}-{}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
struct Fixture {
    parent: TritModelReader<Shared>,
    bytes: Rc<RefCell<Vec<u8>>>,
    original: Vec<u8>,
    view: NativeRefinementParentView,
    plan: NativeRefinementPlan,
    parents: Vec<AdditiveTensor>,
    session: NativePvRefinementSession,
    replay: VerifiedNativePvReplay,
    journal: Vec<u8>,
    checkpoint: Vec<u8>,
}
fn pv_recipe() -> NativeRecoveryRecipe {
    let optimizer = NativeAdamWRecipe {
        lr: 0.01,
        beta1: 0.9,
        beta2: 0.99,
        eps: 1e-8,
        weight_decay: 0.0,
    };
    NativeRecoveryRecipe::AlternatingPv {
        continuous_optimizer: optimizer,
        code_optimizer: NativeAdamWRecipe {
            lr: 0.6,
            ..optimizer
        },
        max_code_change_fraction: 1.0,
        max_relative_code_change: None,
    }
}
impl Fixture {
    fn new(
        precision: ScalePrecision,
        basis: Basis,
        codec: PlaneCodec,
        level: &str,
        zero: bool,
    ) -> Self {
        let super::Fixture {
            parent,
            bytes,
            original,
            view,
            mut plan,
            parents,
            ..
        } = super::Fixture::new(precision, basis, codec, level, zero);
        plan.recipe = pv_recipe();
        let (session, journal, checkpoint, replay) =
            trained(&plan, &parents, if zero { 0.0 } else { 0.25 });
        Self {
            parent,
            bytes,
            original,
            view,
            plan,
            parents,
            session,
            journal,
            checkpoint,
            replay,
        }
    }
}
fn journal_limits() -> NativePvJournalLimits {
    NativePvJournalLimits {
        max_journal_bytes: 1_000_000,
        max_working_bytes: 1_000_000,
    }
}
fn paged_verify_limits() -> NativePvPagedChildVerificationLimits {
    NativePvPagedChildVerificationLimits {
        owners: NativeRefinementOwnerReadLimits {
            max_working_bytes: 2_000_000,
            extra_payload_bytes: 0,
        },
        replay: NativePvPagedReplayLimits {
            max_journal_bytes: 2_000_000,
            max_working_bytes: 16384,
            paging: NativePvPagedLimits {
                max_resident_payload_bytes: 2_000_000,
            },
        },
        max_encoded_owner_bytes: 1_000_000,
    }
}
fn paged_verify<P: Read + Seek, C: Read + Seek, J: Read + Seek, K: Read + Seek>(
    source: NativePvPagedChildVerification<'_, P, C, J, K>,
) -> Result<VerifiedNativePvChild, NativePvPagedChildVerificationError<&'static str>> {
    verify_native_pv_paged_child(source, paged_verify_limits(), |_| {
        Ok::<_, &str>(Cursor::new(Vec::new()))
    })
}
#[path = "pv_paged.rs"]
mod paged;
fn recorder(run: &str) -> EvidenceRecorder {
    EvidenceRecorder::new(
        run,
        [
            NATIVE_REFINEMENT_STARTED_REGISTRATION,
            NATIVE_REFINEMENT_CHECKPOINTED_REGISTRATION,
            NATIVE_REFINEMENT_COMPLETED_REGISTRATION,
            NATIVE_PV_REPLAY_OBSERVED_REGISTRATION,
        ],
    )
    .unwrap()
}
fn trained(
    plan: &NativeRefinementPlan,
    parents: &[AdditiveTensor],
    gradient: f32,
) -> (
    NativePvRefinementSession,
    Vec<u8>,
    Vec<u8>,
    VerifiedNativePvReplay,
) {
    let mut s = NativePvRefinementSession::new(plan.clone(), parents.to_vec()).unwrap();
    let mut j = NativePvJournalWriter::new(Cursor::new(Vec::new()), &s, journal_limits()).unwrap();
    let scales: Vec<_> = parents
        .iter()
        .map(|w| vec![gradient; w.scales().len()])
        .collect();
    let weights: Vec<_> = parents
        .iter()
        .map(|w| {
            vec![
                gradient;
                if w.plane_count() == 0 {
                    0
                } else {
                    w.view().plane_trits(0).unwrap().len()
                }
            ]
        })
        .collect();
    for _ in 0..plan.training.len() {
        s.step_recorded(
            &scales.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            &weights.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            &mut j,
        )
        .unwrap();
    }
    let (stream, summary) = j.finish(&s).unwrap();
    let journal = stream.into_inner();
    let cp = s.checkpoint_bytes().unwrap();
    let replay = replay_native_pv_journal(
        plan.clone(),
        parents.to_vec(),
        Cursor::new(&journal),
        summary.journal(),
        &cp,
        journal_limits(),
    )
    .unwrap();
    (s, journal, cp, replay)
}
fn publish<S: Read + Write + Seek>(
    f: &mut Fixture,
    out: &Path,
    spool: &mut S,
    limits: NativePvChildLimits,
    recorder: EvidenceRecorder,
) -> Result<PublishedNativePvChild, NativePvChildError> {
    publish_native_pv_child(
        &f.session,
        NativePvChildSource {
            parent: &mut f.parent,
            parent_evidence: &f.view,
            replay: &f.replay,
            journal: &mut Cursor::new(&f.journal),
            machine: "synthetic-pv-fixture",
            validation: &validation(),
        },
        spool,
        &mut DirectoryTensorSink::new(out),
        limits,
        recorder,
    )
}
#[test]
fn actual_ptq_parent_recorded_recovery_and_complete_single_level_child_reopen() {
    let root = Scratch::new("profiles");
    let mut cases = 0;
    for precision in [ScalePrecision::F16, ScalePrecision::F32] {
        for codec in [PlaneCodec::D2, PlaneCodec::B3] {
            for basis in [
                Basis::Identity,
                Basis::SignedRht {
                    block: 32,
                    seed: 19,
                    domain: 7,
                },
            ] {
                for level in ["compact", "full"] {
                    let mut f = Fixture::new(precision, basis, codec, level, false);
                    let old_id = f.parent.model_id(level).unwrap();
                    assert_ne!(
                        f.session.weights().next().unwrap().trits(),
                        f.parents[0].trits(),
                        "P/V fixtures must actually change codes"
                    );
                    let preserved = tensor_bytes(&mut f.parent, level, "frozen.weight");
                    let norm = tensor_bytes(&mut f.parent, level, "norm");
                    let old_storage = f
                        .parent
                        .tensor_meta(level, "weight")
                        .unwrap()
                        .storage
                        .clone();
                    let assets: Vec<_> = f
                        .parent
                        .manifest()
                        .assets
                        .iter()
                        .map(|a| a.name.clone())
                        .collect();
                    let expected: Vec<_> = assets
                        .iter()
                        .map(|n| asset_bytes(&mut f.parent, n))
                        .collect();
                    let out = root.0.join(format!("case-{cases}"));
                    let mut spool = Cursor::new(Vec::new());
                    let mut child =
                        publish(&mut f, &out, &mut spool, limits(), recorder("native-child"))
                            .unwrap();
                    let proof = verify_native_pv_paged_child(
                        NativePvPagedChildVerification {
                            parent: &mut f.parent,
                            parent_evidence: &f.view,
                            child: &mut child.model,
                            log: child.evidence.log(),
                            plan: &f.plan,
                            journal: &mut fs::File::open(out.join(NATIVE_PV_JOURNAL_FILE)).unwrap(),
                            checkpoint: &mut fs::File::open(out.join(NATIVE_PV_CHECKPOINT_FILE))
                                .unwrap(),
                        },
                        paged_verify_limits(),
                        |i| {
                            fs::OpenOptions::new()
                                .read(true)
                                .write(true)
                                .create_new(true)
                                .open(out.join(format!("verify-{i}.tnpv")))
                        },
                    )
                    .unwrap();
                    for i in 0..=f.plan.training.len() {
                        fs::remove_file(out.join(format!("verify-{i}.tnpv"))).unwrap();
                    }
                    assert_eq!(proof.package(), child.model.package_id());
                    assert_eq!(proof.replay().summary(), f.replay.summary());
                    assert_eq!(child.model.manifest().levels.len(), 1);
                    assert_eq!(
                        child.model.manifest().schema,
                        SchemaId { major: 1, minor: 1 }
                    );
                    assert_eq!(child.model.manifest().levels[0].name, "refined");
                    assert!(matches!(
                        child.model.manifest().levels[0].fit,
                        LevelFit::Refined { .. }
                    ));
                    assert_eq!(
                        child
                            .model
                            .tensor_meta("refined", "weight")
                            .unwrap()
                            .storage,
                        old_storage
                    );
                    assert_ne!(child.model_id, old_id);
                    assert_ne!(
                        child.model.package_id().as_bytes(),
                        f.view.binding().package.as_bytes()
                    );
                    assert_eq!(
                        tensor_bytes(&mut child.model, "refined", "frozen.weight"),
                        preserved
                    );
                    assert_eq!(tensor_bytes(&mut child.model, "refined", "norm"), norm);
                    for (name, bytes) in assets.iter().zip(&expected) {
                        assert_eq!(asset_bytes(&mut child.model, name), *bytes);
                    }
                    assert_eq!(
                        fs::read(out.join(NATIVE_PV_CHECKPOINT_FILE)).unwrap(),
                        f.checkpoint
                    );
                    assert_eq!(
                        fs::read(out.join(NATIVE_PV_JOURNAL_FILE)).unwrap(),
                        f.journal
                    );
                    assert_eq!(
                        fs::read_to_string(out.join("evidence.jsonl")).unwrap(),
                        child.evidence.jsonl()
                    );
                    assert_eq!(fs::read_dir(&out).unwrap().count(), 4);
                    let log =
                        verify_jsonl(&fs::read_to_string(out.join("evidence.jsonl")).unwrap())
                            .unwrap();
                    let receipt =
                        NativePvChildReceiptView::fold(&log, &mut f.parent, &f.view, &f.plan)
                            .unwrap()
                            .remove(0);
                    receipt.verify_child(&mut child.model).unwrap();
                    assert_eq!(receipt.replay().journal, f.replay.summary().journal());
                    assert_eq!(*f.bytes.borrow(), f.original);
                    assert!(
                        tritium_quantize::engine::verify_package_with_fit_log(
                            &fs::read(out.join("model.trit")).unwrap(),
                            &log
                        )
                        .is_err(),
                        "native child must not become a completed PTQ conversion"
                    );
                    drop(child);
                    drop(f);
                    let mut standalone = TritModelReader::open(
                        fs::File::open(out.join("model.trit")).unwrap(),
                        receiving(),
                    )
                    .unwrap();
                    standalone.verify().unwrap();
                    assert_eq!(tensor_bytes(&mut standalone, "refined", "norm"), norm);
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 16);
    println!("NATIVE_PV_CHILD_PUBLICATION_PROFILES={cases}");
}

#[test]
fn unchanged_state_keeps_semantic_model_id_but_not_parent_package_identity() {
    let root = Scratch::new("unchanged");
    let mut f = Fixture::new(
        ScalePrecision::F32,
        Basis::Identity,
        PlaneCodec::B3,
        "full",
        true,
    );
    let old = f.parent.model_id("full").unwrap();
    let child = publish(
        &mut f,
        &root.0.join("out"),
        &mut Cursor::new(Vec::new()),
        limits(),
        recorder("unchanged-child"),
    )
    .unwrap();
    assert_eq!(child.model_id, old);
    assert_ne!(
        child.model.package_id().as_bytes(),
        f.view.binding().package.as_bytes()
    );
    assert_eq!(
        child.receipt.refinement().completed().validation[0].child_loss,
        3.0,
        "regressing declared observations are not automatically quality approval"
    );
}

#[derive(Debug)]
struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        panic!("no spool read")
    }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        panic!("no spool write")
    }
    fn flush(&mut self) -> io::Result<()> {
        panic!("no spool flush")
    }
}
impl Seek for NoIo {
    fn seek(&mut self, _: SeekFrom) -> io::Result<u64> {
        panic!("no spool seek before admission")
    }
}
#[test]
fn payload_caps_incomplete_and_stale_state_refuse_before_spool_io() {
    let root = Scratch::new("admission");
    let mut f = Fixture::new(
        ScalePrecision::F32,
        Basis::Identity,
        PlaneCodec::B3,
        "full",
        false,
    );
    let size = inspect_native_pv_child(&f.session, &f.parent, &f.replay).unwrap();
    for mode in 0..4 {
        let mut cap = limits();
        match mode {
            0 => cap.max_spool_bytes = size.spool_payload_bytes - 1,
            1 => cap.max_working_bytes = size.working_payload_bytes - 1,
            2 => cap.max_companion_bytes = size.companion_payload_bytes - 1,
            _ => cap.max_evidence_bytes = 0,
        };
        let out = root.0.join(format!("cap-{mode}"));
        assert!(publish(&mut f, &out, &mut NoIo, cap, recorder("reject-cap")).is_err());
        assert!(!out.exists());
    }
    let completed = std::mem::replace(
        &mut f.session,
        NativePvRefinementSession::new(f.plan.clone(), f.parents.clone()).unwrap(),
    );
    assert!(
        publish(
            &mut f,
            &root.0.join("incomplete"),
            &mut NoIo,
            limits(),
            recorder("incomplete")
        )
        .is_err()
    );
    f.session = completed;
    let (changed, _, _, _) = trained(&f.plan, &f.parents, 0.125);
    f.session = changed;
    assert!(
        publish(
            &mut f,
            &root.0.join("stale"),
            &mut NoIo,
            limits(),
            recorder("stale")
        )
        .is_err()
    );
}

#[test]
fn actual_original_parent_state_is_checked_beyond_declared_parent_and_replay() {
    let root = Scratch::new("fake-state");
    let mut f = Fixture::new(
        ScalePrecision::F32,
        Basis::Identity,
        PlaneCodec::B3,
        "full",
        false,
    );
    let mut fake = f.parents.clone();
    let scales = fake[0].scales().iter().map(|s| s + 0.125).collect();
    fake[0].replace_scales(scales).unwrap();
    let (session, journal, checkpoint, replay) = trained(&f.plan, &fake, 0.25);
    f.session = session;
    f.journal = journal;
    f.checkpoint = checkpoint;
    f.replay = replay;
    let mut spool = Cursor::new(Vec::new());
    let error = publish(
        &mut f,
        &root.0.join("out"),
        &mut spool,
        limits(),
        recorder("fake-source"),
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("actual original parent state differs")
    );
    assert!(spool.into_inner().is_empty());
    assert!(!root.0.join("out").exists());
}

#[test]
fn mutated_parent_and_unregistered_producer_reject_without_publication() {
    let root = Scratch::new("parent-change");
    let mut f = Fixture::new(
        ScalePrecision::F32,
        Basis::Identity,
        PlaneCodec::B3,
        "full",
        false,
    );
    let incomplete = EvidenceRecorder::new(
        "missing-replay-registration",
        [
            NATIVE_REFINEMENT_STARTED_REGISTRATION,
            NATIVE_REFINEMENT_CHECKPOINTED_REGISTRATION,
            NATIVE_REFINEMENT_COMPLETED_REGISTRATION,
        ],
    )
    .unwrap();
    assert!(
        publish(
            &mut f,
            &root.0.join("missing"),
            &mut NoIo,
            limits(),
            incomplete
        )
        .is_err()
    );
    f.bytes.borrow_mut()[100] ^= 1;
    assert!(
        publish(
            &mut f,
            &root.0.join("changed"),
            &mut NoIo,
            limits(),
            recorder("changed-parent")
        )
        .is_err()
    );
    assert!(!root.0.join("changed").exists());
}

#[test]
fn journal_corruption_and_container_evidence_caps_leave_no_committed_output() {
    let root = Scratch::new("late-refusal");
    for mode in 0..5 {
        let mut f = Fixture::new(
            ScalePrecision::F32,
            Basis::Identity,
            PlaneCodec::B3,
            "full",
            false,
        );
        let mut cap = limits();
        match mode {
            0 => f.journal[0] ^= 1,
            1 => {
                f.journal.pop();
            }
            2 => f.journal.push(0),
            3 => cap.receiving.max_source_bytes = 1,
            _ => cap.max_evidence_bytes = 1,
        };
        let out = root.0.join(format!("bad-{mode}"));
        let mut spool = Cursor::new(Vec::new());
        assert!(publish(&mut f, &out, &mut spool, cap, recorder("late-refusal")).is_err());
        assert!(!out.exists());
        assert!(
            !spool.get_ref().is_empty(),
            "failed private spool tail remains caller-owned"
        );
        assert_eq!(*f.bytes.borrow(), f.original);
    }
}

#[test]
fn exact_builder_caps_and_no_clobber_preserve_original_files() {
    let root = Scratch::new("exact");
    let mut f = Fixture::new(
        ScalePrecision::F32,
        Basis::Identity,
        PlaneCodec::B3,
        "compact",
        false,
    );
    let size = inspect_native_pv_child(&f.session, &f.parent, &f.replay).unwrap();
    let mut cap = limits();
    cap.max_spool_bytes = size.spool_payload_bytes;
    cap.max_working_bytes = size.working_payload_bytes;
    cap.max_companion_bytes = size.companion_payload_bytes;
    let out = root.0.join("out");
    let child = publish(
        &mut f,
        &out,
        &mut Cursor::new(Vec::new()),
        cap,
        recorder("exact"),
    )
    .unwrap();
    drop(child);
    let original: Vec<_> = [
        "model.trit",
        "evidence.jsonl",
        NATIVE_PV_CHECKPOINT_FILE,
        NATIVE_PV_JOURNAL_FILE,
    ]
    .map(|n| (n, fs::read(out.join(n)).unwrap()))
    .into();
    assert!(
        publish(
            &mut f,
            &out,
            &mut Cursor::new(Vec::new()),
            cap,
            recorder("no-clobber")
        )
        .is_err()
    );
    for (name, bytes) in original {
        assert_eq!(fs::read(out.join(name)).unwrap(), bytes);
    }
    assert_eq!(fs::read_dir(out).unwrap().count(), 4);
}

#[test]
fn parent_changed_during_journal_copy_fails_final_guard_and_cleans_owned_files() {
    let root = Scratch::new("guard");
    let mut f = Fixture::new(
        ScalePrecision::F32,
        Basis::Identity,
        PlaneCodec::B3,
        "full",
        false,
    );
    let out = root.0.join("out");
    struct Mutator {
        data: Cursor<Vec<u8>>,
        parent: Rc<RefCell<Vec<u8>>>,
        done: bool,
    }
    impl Read for Mutator {
        fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
            if !self.done {
                self.parent.borrow_mut()[100] ^= 1;
                self.done = true;
            }
            self.data.read(b)
        }
    }
    impl Seek for Mutator {
        fn seek(&mut self, s: SeekFrom) -> io::Result<u64> {
            self.data.seek(s)
        }
    }
    let mut journal = Mutator {
        data: Cursor::new(f.journal.clone()),
        parent: f.bytes.clone(),
        done: false,
    };
    let error = publish_native_pv_child(
        &f.session,
        NativePvChildSource {
            parent: &mut f.parent,
            parent_evidence: &f.view,
            replay: &f.replay,
            journal: &mut journal,
            machine: "fixture",
            validation: &validation(),
        },
        &mut Cursor::new(Vec::new()),
        &mut DirectoryTensorSink::new(&out),
        limits(),
        recorder("changed-after-copy"),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        NativePvChildError::Publication(
            tritium_quantize::engine::TritModelPublicationError::Staging(
                tritium_format::trit_model::TritModelStageError::Guard(
                    NativePvChildGuardError::Parent(_)
                )
            )
        )
    ));
    assert!(!out.exists());
}

#[test]
fn resealed_replay_record_mismatches_and_ptq_relabeling_never_admit_the_original_child() {
    let root = Scratch::new("records");
    let mut f = Fixture::new(
        ScalePrecision::F32,
        Basis::Identity,
        PlaneCodec::B3,
        "full",
        false,
    );
    let mut child = publish(
        &mut f,
        &root.0.join("out"),
        &mut Cursor::new(Vec::new()),
        limits(),
        recorder("records"),
    )
    .unwrap();
    let original = child.evidence.log().clone();
    for mode in 0..10 {
        let mut record = recorder("resealed");
        for event in &original.events {
            match event.schema.as_str() {
                "tritium.refine.native_started" => {
                    let mut value: NativeRefinementStarted =
                        serde_json::from_value(event.payload.clone()).unwrap();
                    if mode == 8 {
                        value.plan.objective = [99; 32];
                    }
                    if mode == 9 {
                        value.plan.parent.level = "compact".into();
                    }
                    record
                        .emit(&event.span, event.parent.as_deref(), event.t, &value)
                        .unwrap();
                }
                "tritium.refine.pv_replay_observed" => {
                    if mode == 0 {
                        continue;
                    }
                    let mut value: NativePvReplayObserved =
                        serde_json::from_value(event.payload.clone()).unwrap();
                    match mode {
                        1 => value.plan = NativeRefinementPlanId::from_bytes([99; 32]),
                        2 => value.checkpoint = BlobId::from_bytes([99; 32]),
                        3 => value.checkpoint_bytes += 1,
                        4 => value.completed_updates -= 1,
                        5 => value.journal_bytes = 0,
                        6 => value.journal = BlobId::from_bytes([99; 32]),
                        _ => {}
                    }
                    record
                        .emit(
                            &event.span,
                            if mode == 7 {
                                None
                            } else {
                                event.parent.as_deref()
                            },
                            event.t,
                            &value,
                        )
                        .unwrap();
                }
                "tritium.refine.native_checkpointed" => {
                    let value: NativeRefinementCheckpointed =
                        serde_json::from_value(event.payload.clone()).unwrap();
                    record
                        .emit(&event.span, event.parent.as_deref(), event.t, &value)
                        .unwrap();
                }
                "tritium.refine.native_completed" => {
                    let value: NativeRefinementCompleted =
                        serde_json::from_value(event.payload.clone()).unwrap();
                    record
                        .emit(&event.span, event.parent.as_deref(), event.t, &value)
                        .unwrap();
                }
                _ => panic!("unexpected event"),
            }
        }
        let changed = record.seal().unwrap();
        let result = NativePvChildReceiptView::fold(changed.log(), &mut f.parent, &f.view, &f.plan);
        if mode == 6 {
            let view = result.unwrap().remove(0);
            assert!(view.verify_child(&mut child.model).is_err());
            assert!(
                replay_native_pv_journal(
                    f.plan.clone(),
                    f.parents.clone(),
                    Cursor::new(&f.journal),
                    view.replay().journal,
                    &f.checkpoint,
                    journal_limits()
                )
                .is_err(),
                "recorded journal id alone is not numerical proof"
            );
        } else {
            assert!(result.is_err(), "resealed mode{mode} must reject");
        }
    }
    child.receipt.verify_child(&mut child.model).unwrap();
}

#[test]
fn spool_write_error_and_journal_read_error_leave_parent_and_destination_untouched() {
    let root = Scratch::new("io");
    let mut f = Fixture::new(
        ScalePrecision::F32,
        Basis::Identity,
        PlaneCodec::B3,
        "full",
        false,
    );
    struct FailWrite(Cursor<Vec<u8>>);
    impl Read for FailWrite {
        fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
            self.0.read(b)
        }
    }
    impl Seek for FailWrite {
        fn seek(&mut self, s: SeekFrom) -> io::Result<u64> {
            self.0.seek(s)
        }
    }
    impl Write for FailWrite {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::other("original native spool write"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let error = publish(
        &mut f,
        &root.0.join("spool"),
        &mut FailWrite(Cursor::new(Vec::new())),
        limits(),
        recorder("spool-failure"),
    )
    .unwrap_err();
    assert!(error.to_string().contains("original native spool write"));
    struct FailRead(Cursor<Vec<u8>>);
    impl Read for FailRead {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::other("original native journal read"))
        }
    }
    impl Seek for FailRead {
        fn seek(&mut self, s: SeekFrom) -> io::Result<u64> {
            self.0.seek(s)
        }
    }
    let error = publish_native_pv_child(
        &f.session,
        NativePvChildSource {
            parent: &mut f.parent,
            parent_evidence: &f.view,
            replay: &f.replay,
            journal: &mut FailRead(Cursor::new(f.journal.clone())),
            machine: "fixture",
            validation: &validation(),
        },
        &mut Cursor::new(Vec::new()),
        &mut DirectoryTensorSink::new(root.0.join("journal")),
        limits(),
        recorder("journal-failure"),
    )
    .unwrap_err();
    assert!(error.to_string().contains("original native journal read"));
    assert!(!root.0.join("spool").exists());
    assert!(!root.0.join("journal").exists());
    assert_eq!(*f.bytes.borrow(), f.original);
}

#[test]
fn synthetic_imported_raw_profiles_preserve_all_four_laws_codecs_bases_and_allocations() {
    use tritium_core::{PerTilePlan, Trit, UniformPlan};
    use tritium_format::trit_model::encode_model_manifest;
    use tritium_format::{TritBlob, TritBlobKind, write_trit_package};
    let root = Scratch::new("imported-profiles");
    let template = Fixture::new(
        ScalePrecision::F32,
        Basis::Identity,
        PlaneCodec::B3,
        "full",
        false,
    );
    let mut cases = 0;
    let id = |b: &[u8]| BlobId::from_bytes(*blake3::hash(b).as_bytes());
    let reference = |b: &[u8]| BlobReference {
        id: id(b),
        shard: None,
    };
    for admitted in ADMITTED_LAWS {
        for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
            for basis in [
                Basis::Identity,
                Basis::Hadamard { block: 128 },
                Basis::SignedRht {
                    block: 128,
                    seed: u64::MAX - 5,
                    domain: u64::MAX - 3,
                },
            ] {
                for mode in 0..3 {
                    for group in [32, 128] {
                        let layout = AdditiveLayout {
                            rows: 3,
                            cols: 256,
                            tile: 256,
                            group,
                            max_planes: admitted.max_planes,
                            allocation: if mode == 0 {
                                PlaneAllocation::Uniform
                            } else {
                                PlaneAllocation::PerTile
                            },
                            codec,
                            law: admitted.law,
                            basis,
                            transport: Transport::Raw,
                        };
                        let counts = if mode == 2 {
                            vec![0; 3]
                        } else {
                            vec![0, 1, admitted.max_planes]
                        };
                        let (n, s) = if mode == 0 {
                            let p = UniformPlan::new(layout, admitted.max_planes).unwrap();
                            (p.trit_count(), p.scale_count())
                        } else {
                            let p = PerTilePlan::new(layout, &counts).unwrap();
                            (p.trit_count(), p.scale_count())
                        };
                        let codes = (0..n)
                            .map(|i| [Trit::ZERO, Trit::NEG, Trit::POS, Trit::NEG][i % 4])
                            .collect();
                        let tensor = if mode == 0 {
                            AdditiveTensor::new(layout, admitted.max_planes, codes, vec![0.5; s])
                                .unwrap()
                        } else {
                            AdditiveTensor::new_per_tile(
                                layout,
                                counts.clone(),
                                codes,
                                vec![0.5; s],
                            )
                            .unwrap()
                        };
                        let weight = tensor.encode_payload().unwrap();
                        let norm = 0.25f32.to_le_bytes();
                        let config = b"{}";
                        let tokenizer = b"{ }";
                        let manifest = ModelManifest::imported(
                            "synthetic-import-fixture".into(),
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
                            vec![
                                TensorMeta {
                                    name: "norm".into(),
                                    blob: reference(&norm),
                                    storage: TensorStorage::Dense {
                                        shape: vec![1],
                                        precision: ScalePrecision::F32,
                                    },
                                },
                                TensorMeta {
                                    name: "weight".into(),
                                    blob: reference(&weight),
                                    storage: TensorStorage::Additive {
                                        layout,
                                        planes: if mode == 0 {
                                            PlaneCounts::Uniform(admitted.max_planes)
                                        } else {
                                            PlaneCounts::PerTile(counts)
                                        },
                                    },
                                },
                            ],
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
                        let original_source =
                            original_source_id(&manifest.architecture, &companions, &assets)
                                .unwrap();
                        let event = LegacyModelImportBoundView::prepare(
                            &manifest,
                            "legacy:synthetic".into(),
                            original_source,
                            companions,
                            assets,
                        )
                        .unwrap();
                        let mut original = EvidenceRecorder::new(
                            "synthetic-import",
                            [LEGACY_MODEL_IMPORT_BOUND_PREPARED_REGISTRATION],
                        )
                        .unwrap();
                        original.emit("import", None, 0, &event).unwrap();
                        let original = original.seal().unwrap();
                        let bytes = write_trit_package(
                            &encode_model_manifest(&manifest).unwrap(),
                            &[
                                TritBlob {
                                    kind: TritBlobKind::DenseTensor(tritium_core::DType::F32),
                                    bytes: &norm,
                                },
                                TritBlob {
                                    kind: TritBlobKind::AdditiveTensor,
                                    bytes: &weight,
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
                        let mut parent =
                            TritModelReader::open(Cursor::new(bytes), receiving()).unwrap();
                        let view = NativeRefinementParentView::from_import(
                            &mut parent,
                            "parent",
                            original.log(),
                        )
                        .unwrap();
                        let mut plan = template.plan.clone();
                        plan.parent = view.binding().clone();
                        let (session, journal, checkpoint, replay) =
                            trained(&plan, std::slice::from_ref(&tensor), 0.125);
                        let out = root.0.join(format!("case-{cases}"));
                        let mut spool = Cursor::new(Vec::new());
                        let mut child = publish_native_pv_child(
                            &session,
                            NativePvChildSource {
                                parent: &mut parent,
                                parent_evidence: &view,
                                replay: &replay,
                                journal: &mut Cursor::new(&journal),
                                machine: "synthetic-import",
                                validation: &validation(),
                            },
                            &mut spool,
                            &mut DirectoryTensorSink::new(&out),
                            limits(),
                            recorder("imported-child"),
                        )
                        .unwrap();
                        let proof = paged_verify(NativePvPagedChildVerification {
                            parent: &mut parent,
                            parent_evidence: &view,
                            child: &mut child.model,
                            log: child.evidence.log(),
                            plan: &plan,
                            journal: &mut Cursor::new(&journal),
                            checkpoint: &mut Cursor::new(&checkpoint),
                        })
                        .unwrap();
                        assert_eq!(proof.replay().summary(), replay.summary());
                        assert_eq!(proof.package(), child.model.package_id());
                        let raw = tensor_bytes(&mut child.model, "refined", "weight");
                        let actual = decode_additive_payload(
                            child.model.tensor_meta("refined", "weight").unwrap(),
                            &raw,
                        )
                        .unwrap();
                        assert_eq!(actual, *session.weights().next().unwrap());
                        assert_eq!(tensor_bytes(&mut child.model, "refined", "norm"), norm);
                        assert_eq!(
                            fs::read(out.join(NATIVE_PV_CHECKPOINT_FILE)).unwrap(),
                            checkpoint
                        );
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 216);
    println!("NATIVE_PV_CHILD_SYNTHETIC_RAW_PROFILES={cases}");
}

#[test]
fn numerical_verifier_rejects_record_consistent_forged_encoded_owner_state() {
    use tritium_format::trit_model::encode_model_manifest;
    use tritium_format::{TritBlob, TritBlobKind, write_trit_package};
    let root = Scratch::new("numeric-binding");
    let mut f = Fixture::new(
        ScalePrecision::F32,
        Basis::Identity,
        PlaneCodec::B3,
        "full",
        false,
    );
    let mut child = publish(
        &mut f,
        &root.0.join("child"),
        &mut Cursor::new(Vec::new()),
        limits(),
        recorder("original"),
    )
    .unwrap();
    let accepted = verify_native_pv_child(
        NativePvChildVerification {
            parent: &mut f.parent,
            parent_evidence: &f.view,
            child: &mut child.model,
            log: child.evidence.log(),
            plan: &f.plan,
            journal: &mut Cursor::new(&f.journal),
            checkpoint: &f.checkpoint,
        },
        NativePvParentReadLimits {
            max_working_bytes: 2_000_000,
            extra_payload_bytes: 0,
        },
        journal_limits(),
    )
    .unwrap();
    assert_eq!(accepted.package(), child.model.package_id());
    let paged = paged_verify(NativePvPagedChildVerification {
        parent: &mut f.parent,
        parent_evidence: &f.view,
        child: &mut child.model,
        log: child.evidence.log(),
        plan: &f.plan,
        journal: &mut Cursor::new(&f.journal),
        checkpoint: &mut Cursor::new(&f.checkpoint),
    })
    .unwrap();
    assert_eq!(paged.package(), accepted.package());
    assert_eq!(paged.replay().summary(), accepted.replay().summary());
    let (other, _, _, _) = trained(&f.plan, &f.parents, 0.125);
    let changed = other.weights().next().unwrap().encode_payload().unwrap();
    let changed_id = BlobId::from_bytes(*blake3::hash(&changed).as_bytes());
    assert_ne!(
        changed_id,
        child
            .model
            .tensor_meta("refined", "weight")
            .unwrap()
            .blob
            .id
    );
    let mut fake_log = recorder("forged-but-record-consistent");
    for event in &child.evidence.log().events {
        match event.schema.as_str() {
            "tritium.refine.native_started" => {
                let value: NativeRefinementStarted =
                    serde_json::from_value(event.payload.clone()).unwrap();
                fake_log
                    .emit(&event.span, event.parent.as_deref(), event.t, &value)
                    .unwrap();
            }
            "tritium.refine.pv_replay_observed" => {
                let value: NativePvReplayObserved =
                    serde_json::from_value(event.payload.clone()).unwrap();
                fake_log
                    .emit(&event.span, event.parent.as_deref(), event.t, &value)
                    .unwrap();
            }
            "tritium.refine.native_checkpointed" => {
                let mut value: NativeRefinementCheckpointed =
                    serde_json::from_value(event.payload.clone()).unwrap();
                value.owners[0].content = changed_id;
                fake_log
                    .emit(&event.span, event.parent.as_deref(), event.t, &value)
                    .unwrap();
            }
            "tritium.refine.native_completed" => {
                let mut value: NativeRefinementCompleted =
                    serde_json::from_value(event.payload.clone()).unwrap();
                value
                    .tensors
                    .iter_mut()
                    .find(|t| t.name == "weight")
                    .unwrap()
                    .content = changed_id;
                fake_log
                    .emit(&event.span, event.parent.as_deref(), event.t, &value)
                    .unwrap();
            }
            _ => panic!("unexpected event"),
        }
    }
    let fake_log = fake_log.seal().unwrap();
    let view = NativePvChildReceiptView::fold(fake_log.log(), &mut f.parent, &f.view, &f.plan)
        .unwrap()
        .remove(0);
    let mut payloads = Vec::new();
    for meta in &view.manifest().levels[0].tensors {
        let data = if meta.name == "weight" {
            changed.clone()
        } else {
            tensor_bytes(&mut child.model, "refined", &meta.name)
        };
        let kind = match &meta.storage {
            TensorStorage::Additive { .. } => TritBlobKind::AdditiveTensor,
            TensorStorage::Dense { precision, .. } => TritBlobKind::DenseTensor(match precision {
                ScalePrecision::F32 => tritium_core::DType::F32,
                ScalePrecision::Bf16 => tritium_core::DType::BF16,
                _ => tritium_core::DType::F16,
            }),
            _ => panic!("unsupported"),
        };
        payloads.push((kind, data));
    }
    for asset in &view.manifest().assets {
        payloads.push((
            TritBlobKind::Asset(asset.name.clone()),
            asset_bytes(&mut child.model, &asset.name),
        ));
    }
    let blobs: Vec<_> = payloads
        .iter()
        .map(|(kind, bytes)| TritBlob {
            kind: kind.clone(),
            bytes,
        })
        .collect();
    let bytes =
        write_trit_package(&encode_model_manifest(view.manifest()).unwrap(), &blobs).unwrap();
    let mut fake = TritModelReader::open(Cursor::new(bytes), receiving()).unwrap();
    NativePvChildRecordView::from_child(fake_log.log(), &mut fake).unwrap();
    view.verify_child(&mut fake).unwrap();
    let error = verify_native_pv_child(
        NativePvChildVerification {
            parent: &mut f.parent,
            parent_evidence: &f.view,
            child: &mut fake,
            log: fake_log.log(),
            plan: &f.plan,
            journal: &mut Cursor::new(&f.journal),
            checkpoint: &f.checkpoint,
        },
        NativePvParentReadLimits {
            max_working_bytes: 2_000_000,
            extra_payload_bytes: 0,
        },
        journal_limits(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("replayed owner state"));
    let error = paged_verify(NativePvPagedChildVerification {
        parent: &mut f.parent,
        parent_evidence: &f.view,
        child: &mut fake,
        log: fake_log.log(),
        plan: &f.plan,
        journal: &mut Cursor::new(&f.journal),
        checkpoint: &mut Cursor::new(&f.checkpoint),
    })
    .unwrap_err();
    assert!(error.to_string().contains("replayed owner state"));
}

#[test]
fn profile_events_and_recipes_cannot_be_relabelled_as_each_other() {
    let root = Scratch::new("profile-separation");
    let mut f = Fixture::new(
        ScalePrecision::F32,
        Basis::Identity,
        PlaneCodec::B3,
        "full",
        false,
    );
    let mut child = publish(
        &mut f,
        &root.0.join("pv"),
        &mut Cursor::new(Vec::new()),
        limits(),
        recorder("pv-profile"),
    )
    .unwrap();
    assert!(
        NativeScaleChildRecordView::from_child(child.evidence.log(), &mut child.model).is_err()
    );
    assert!(
        NativeScaleChildReceiptView::fold(child.evidence.log(), &mut f.parent, &f.view, &f.plan)
            .is_err()
    );
    let record =
        NativePvChildRecordView::from_child(child.evidence.log(), &mut child.model).unwrap();
    record.verify_child(&mut child.model).unwrap();
    let mut wrong = super::recorder("wrong-profile-event");
    for e in &child.evidence.log().events {
        match e.schema.as_str() {
            "tritium.refine.native_started" => {
                let v: NativeRefinementStarted = serde_json::from_value(e.payload.clone()).unwrap();
                wrong.emit(&e.span, e.parent.as_deref(), e.t, &v).unwrap();
            }
            "tritium.refine.pv_replay_observed" => {
                let v: NativeScaleReplayObserved =
                    serde_json::from_value(e.payload.clone()).unwrap();
                wrong.emit(&e.span, e.parent.as_deref(), e.t, &v).unwrap();
            }
            "tritium.refine.native_checkpointed" => {
                let v: NativeRefinementCheckpointed =
                    serde_json::from_value(e.payload.clone()).unwrap();
                wrong.emit(&e.span, e.parent.as_deref(), e.t, &v).unwrap();
            }
            "tritium.refine.native_completed" => {
                let v: NativeRefinementCompleted =
                    serde_json::from_value(e.payload.clone()).unwrap();
                wrong.emit(&e.span, e.parent.as_deref(), e.t, &v).unwrap();
            }
            _ => panic!("unexpected event"),
        }
    }
    let wrong = wrong.seal().unwrap();
    assert!(NativePvChildReceiptView::fold(wrong.log(), &mut f.parent, &f.view, &f.plan).is_err());
    assert!(NativePvChildRecordView::from_child(wrong.log(), &mut child.model).is_err());
    let mut scale = super::Fixture::new(
        ScalePrecision::F32,
        Basis::Identity,
        PlaneCodec::B3,
        "full",
        false,
    );
    let mut other = super::publish(
        &mut scale,
        &root.0.join("scale"),
        &mut Cursor::new(Vec::new()),
        super::limits(),
        super::recorder("scale-profile"),
    )
    .unwrap();
    assert!(NativePvChildRecordView::from_child(other.evidence.log(), &mut other.model).is_err());
    assert!(
        NativePvChildReceiptView::fold(
            other.evidence.log(),
            &mut scale.parent,
            &scale.view,
            &scale.plan
        )
        .is_err()
    );
}

#[test]
fn pv_children_supply_explicit_immediate_parents_to_pv_and_scale_rounds() {
    let root = Scratch::new("chaining");
    let mut f = Fixture::new(
        ScalePrecision::F32,
        Basis::Identity,
        PlaneCodec::B3,
        "full",
        false,
    );
    let mut child = publish(
        &mut f,
        &root.0.join("first"),
        &mut Cursor::new(Vec::new()),
        limits(),
        recorder("first-pv"),
    )
    .unwrap();
    let record =
        NativePvChildRecordView::from_child(child.evidence.log(), &mut child.model).unwrap();
    let parent =
        NativeRefinementParentView::from_refined_pv_record(&mut child.model, "refined", &record)
            .unwrap();
    assert!(
        NativeRefinementParentView::from_refined_pv_record(&mut child.model, "full", &record)
            .is_err()
    );
    assert_eq!(parent.binding().source, f.plan.parent.source);
    assert_ne!(parent.binding().package, f.plan.parent.package);
    let mut plan = f.plan.clone();
    plan.parent = parent.binding().clone();
    plan.child_level = "pv-again".into();
    drop(f);
    let parents = decode_native_pv_parents(
        &mut child.model,
        &parent,
        &plan,
        NativePvParentReadLimits {
            max_working_bytes: 1_000_000,
            extra_payload_bytes: 0,
        },
    )
    .unwrap();
    let (session, journal, checkpoint, replay) = trained(&plan, &parents, -0.125);
    let mut next = publish_native_pv_child(
        &session,
        NativePvChildSource {
            parent: &mut child.model,
            parent_evidence: &parent,
            replay: &replay,
            journal: &mut Cursor::new(&journal),
            machine: "fixture",
            validation: &validation(),
        },
        &mut Cursor::new(Vec::new()),
        &mut DirectoryTensorSink::new(root.0.join("pv-again")),
        limits(),
        recorder("second-pv"),
    )
    .unwrap();
    verify_native_pv_child(
        NativePvChildVerification {
            parent: &mut child.model,
            parent_evidence: &parent,
            child: &mut next.model,
            log: next.evidence.log(),
            plan: &plan,
            journal: &mut Cursor::new(&journal),
            checkpoint: &checkpoint,
        },
        NativePvParentReadLimits {
            max_working_bytes: 1_000_000,
            extra_payload_bytes: 0,
        },
        journal_limits(),
    )
    .unwrap();
    let mut scale = plan.clone();
    let paged = paged_verify(NativePvPagedChildVerification {
        parent: &mut child.model,
        parent_evidence: &parent,
        child: &mut next.model,
        log: next.evidence.log(),
        plan: &plan,
        journal: &mut Cursor::new(&journal),
        checkpoint: &mut Cursor::new(&checkpoint),
    })
    .unwrap();
    assert_eq!(paged.replay().summary(), replay.summary());
    scale.child_level = "scale-again".into();
    let NativeRecoveryRecipe::AlternatingPv {
        continuous_optimizer,
        ..
    } = scale.recipe
    else {
        unreachable!()
    };
    scale.recipe = NativeRecoveryRecipe::ScaleOnly {
        optimizer: continuous_optimizer,
    };
    let (s, j, _, r) = super::trained(&scale, &parents, 0.125);
    let output = publish_native_scale_child(
        &s,
        NativeScaleChildSource {
            parent: &mut child.model,
            parent_evidence: &parent,
            replay: &r,
            journal: &mut Cursor::new(j),
            machine: "fixture",
            validation: &validation(),
        },
        &mut Cursor::new(Vec::new()),
        &mut DirectoryTensorSink::new(root.0.join("scale-again")),
        limits(),
        super::recorder("scale-after-pv"),
    )
    .unwrap();
    assert_eq!(
        output.receipt.refinement().started().plan.parent,
        parent.binding().clone()
    );
}

#[test]
fn parent_decode_admission_accounts_for_present_coordinate_code_moments() {
    let mut f = Fixture::new(
        ScalePrecision::F32,
        Basis::Identity,
        PlaneCodec::B3,
        "full",
        false,
    );
    let state = f.session.size_ledger().state_payload_bytes;
    let mut too_small = f.plan.clone();
    too_small.limits.max_state_bytes = state - 1;
    let before = f.bytes.borrow().clone();
    f.bytes.borrow_mut()[100] ^= 1;
    assert!(matches!(
        decode_native_pv_parents(
            &mut f.parent,
            &f.view,
            &too_small,
            NativePvParentReadLimits {
                max_working_bytes: 1_000_000,
                extra_payload_bytes: 0
            }
        ),
        Err(NativePvChildError::Limit(_))
    ));
    *f.bytes.borrow_mut() = before;
    assert!(
        decode_native_scale_parents(
            &mut f.parent,
            &f.view,
            &f.plan,
            NativeScaleParentReadLimits {
                max_working_bytes: 1_000_000,
                extra_payload_bytes: 0
            }
        )
        .is_err()
    );
    let decoded = decode_native_pv_parents(
        &mut f.parent,
        &f.view,
        &f.plan,
        NativePvParentReadLimits {
            max_working_bytes: 1_000_000,
            extra_payload_bytes: 0,
        },
    )
    .unwrap();
    assert_eq!(decoded, f.parents);
}

#[test]
fn standalone_record_cannot_rebind_to_a_different_compatible_model_container() {
    use tritium_format::trit_model::encode_model_manifest;
    use tritium_format::{TritBlob, TritBlobKind, write_trit_package};
    let root = Scratch::new("chain-rebind");
    let mut fixture = Fixture::new(
        ScalePrecision::F32,
        Basis::Identity,
        PlaneCodec::B3,
        "full",
        false,
    );
    let mut child = publish(
        &mut fixture,
        &root.0.join("child"),
        &mut Cursor::new(Vec::new()),
        limits(),
        recorder("child"),
    )
    .unwrap();
    let record =
        NativePvChildRecordView::from_child(child.evidence.log(), &mut child.model).unwrap();
    let mut manifest = child.model.manifest().clone();
    manifest.architecture = "different-compatible-architecture".into();
    let mut payloads = Vec::new();
    for meta in &manifest.levels[0].tensors {
        let bytes = tensor_bytes(&mut child.model, "refined", &meta.name);
        let kind = match &meta.storage {
            TensorStorage::Additive { .. } => TritBlobKind::AdditiveTensor,
            TensorStorage::Dense { precision, .. } => TritBlobKind::DenseTensor(match precision {
                ScalePrecision::F32 => tritium_core::DType::F32,
                ScalePrecision::Bf16 => tritium_core::DType::BF16,
                _ => tritium_core::DType::F16,
            }),
            _ => panic!("unsupported fixture"),
        };
        payloads.push((kind, bytes));
    }
    for asset in &manifest.assets {
        payloads.push((
            TritBlobKind::Asset(asset.name.clone()),
            asset_bytes(&mut child.model, &asset.name),
        ));
    }
    let blobs: Vec<_> = payloads
        .iter()
        .map(|(kind, bytes)| TritBlob {
            kind: kind.clone(),
            bytes,
        })
        .collect();
    let bytes = write_trit_package(&encode_model_manifest(&manifest).unwrap(), &blobs).unwrap();
    let mut other = TritModelReader::open(Cursor::new(bytes), receiving()).unwrap();
    // A fresh standalone observation can be byte/record-consistent in its narrow scope.
    NativePvChildRecordView::from_child(child.evidence.log(), &mut other).unwrap();
    // The old observation must never silently bind those different actual bytes.
    assert!(record.verify_child(&mut other).is_err());
    assert!(
        NativeRefinementParentView::from_refined_pv_record(&mut other, "refined", &record).is_err()
    );
    record.verify_child(&mut child.model).unwrap();
    let path = root.0.join("child/model.trit");
    let mut damaged = fs::read(&path).unwrap();
    let end = damaged.len() - 1;
    damaged[end] ^= 1;
    fs::write(&path, damaged).unwrap();
    assert!(record.verify_child(&mut child.model).is_err());
}

#[test]
fn failed_publication_keeps_unexpected_files_and_removes_only_owned_outputs() {
    let root = Scratch::new("foreign-cleanup");
    let out = root.0.join("child");
    let mut f = Fixture::new(
        ScalePrecision::F32,
        Basis::Identity,
        PlaneCodec::B3,
        "full",
        false,
    );
    struct Mutator {
        inner: Cursor<Vec<u8>>,
        parent: Rc<RefCell<Vec<u8>>>,
        out: PathBuf,
        done: bool,
    }
    impl Read for Mutator {
        fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
            if !self.done {
                fs::write(self.out.join("unexpected"), b"keep")?;
                self.parent.borrow_mut()[100] ^= 1;
                self.done = true;
            }
            self.inner.read(b)
        }
    }
    impl Seek for Mutator {
        fn seek(&mut self, p: SeekFrom) -> io::Result<u64> {
            self.inner.seek(p)
        }
    }
    let mut journal = Mutator {
        inner: Cursor::new(f.journal.clone()),
        parent: f.bytes.clone(),
        out: out.clone(),
        done: false,
    };
    let error = publish_native_pv_child(
        &f.session,
        NativePvChildSource {
            parent: &mut f.parent,
            parent_evidence: &f.view,
            replay: &f.replay,
            journal: &mut journal,
            machine: "fixture",
            validation: &validation(),
        },
        &mut Cursor::new(Vec::new()),
        &mut DirectoryTensorSink::new(&out),
        limits(),
        recorder("foreign-sentinel"),
    )
    .unwrap_err();
    assert!(matches!(error, NativePvChildError::Publication(_)));
    assert_eq!(fs::read(out.join("unexpected")).unwrap(), b"keep");
    for name in [
        "model.trit",
        "evidence.jsonl",
        NATIVE_PV_CHECKPOINT_FILE,
        NATIVE_PV_JOURNAL_FILE,
    ] {
        assert!(
            !out.join(name).exists(),
            "owned {name} survived failed publication"
        );
    }
}
