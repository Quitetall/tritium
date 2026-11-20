//! Paged preparation through the common immutable publication transaction.
use super::*;

fn publish_page<K: Read + Seek, S: Read + Write + Seek>(
    f: &mut Fixture,
    session: &mut NativePvPagedSession<K>,
    out: &Path,
    spool: &mut S,
    cap: NativePvPagedChildLimits,
    record: EvidenceRecorder,
) -> Result<PublishedNativePvPagedChild, NativePvChildError> {
    publish_native_pv_paged_child(
        session,
        NativePvChildSource {
            parent: &mut f.parent,
            parent_evidence: &f.view,
            replay: &f.replay,
            journal: &mut Cursor::new(&f.journal),
            machine: "paged-fixture",
            validation: &validation(),
        },
        spool,
        &mut DirectoryTensorSink::new(out),
        cap,
        record,
    )
}

#[test]
fn caps_precede_spool_io_and_success_never_clobbers_any_existing_output() {
    let root = Scratch::new("paged-publication-caps");
    let mut f = Fixture::new(
        ScalePrecision::F32,
        Basis::Identity,
        PlaneCodec::B3,
        "full",
        false,
    );
    let mut session = paged_session(
        &mut f.parent,
        &f.view,
        &f.plan,
        Cursor::new(f.checkpoint.clone()),
    );
    let sizes =
        inspect_native_pv_paged_child(&session, &mut f.parent, &f.view, &f.replay, 0).unwrap();
    assert_eq!(
        sizes.companion_payload_bytes,
        (f.checkpoint.len() + f.journal.len()) as u64
    );
    assert_eq!(
        sizes.working_payload_bytes,
        f.parent.tensor_byte_len("full", "weight").unwrap() + 65536
    );
    for mode in 0..5 {
        let mut cap = paged_publish_limits();
        match mode {
            0 => cap.max_spool_bytes = sizes.spool_payload_bytes - 1,
            1 => cap.max_working_bytes = sizes.working_payload_bytes - 1,
            2 => cap.max_companion_bytes = sizes.companion_payload_bytes - 1,
            3 => cap.owners.max_working_bytes = sizes.owner_decode_working_bytes - 1,
            _ => cap.max_evidence_bytes = 0,
        }
        let out = root.0.join(format!("bad-{mode}"));
        assert!(matches!(
            publish_page(&mut f, &mut session, &out, &mut NoIo, cap, recorder("caps")),
            Err(NativePvChildError::Limit(_))
        ));
        assert!(!out.exists());
        assert_eq!(*f.bytes.borrow(), f.original);
    }
    let out = root.0.join("child");
    let child = publish_page(
        &mut f,
        &mut session,
        &out,
        &mut Cursor::new(Vec::new()),
        paged_publish_limits(),
        recorder("valid"),
    )
    .unwrap();
    assert_eq!(child.sizes, sizes);
    assert!(child.warnings.is_empty());
    let original = fs::read(out.join("model.trit")).unwrap();
    fs::write(out.join("foreign.keep"), b"keep").unwrap();
    assert!(
        publish_page(
            &mut f,
            &mut session,
            &out,
            &mut Cursor::new(Vec::new()),
            paged_publish_limits(),
            recorder("again")
        )
        .is_err()
    );
    assert_eq!(fs::read(out.join("model.trit")).unwrap(), original);
    assert_eq!(fs::read(out.join("foreign.keep")).unwrap(), b"keep");
    assert!(!out.join(".pending.trit").exists());
}

#[test]
fn changed_parent_checkpoint_journal_and_stale_replay_refuse_before_spool_io() {
    let root = Scratch::new("paged-early-sources");
    for mode in 0..4 {
        let mut f = Fixture::new(
            ScalePrecision::F32,
            Basis::Identity,
            PlaneCodec::B3,
            "full",
            false,
        );
        let cp = Rc::new(RefCell::new(f.checkpoint.clone()));
        let mut state = paged_session(
            &mut f.parent,
            &f.view,
            &f.plan,
            super::super::Shared {
                bytes: cp.clone(),
                position: 0,
            },
        );
        match mode {
            0 => {
                let n = f.bytes.borrow().len();
                f.bytes.borrow_mut()[n - 1] ^= 1;
            }
            1 => {
                let n = cp.borrow().len();
                cp.borrow_mut()[n - 1] ^= 1;
            }
            2 => f.journal[0] ^= 1,
            _ => {
                f.replay = trained(&f.plan, &f.parents, 0.125).3;
            }
        }
        let out = root.0.join(format!("bad-{mode}"));
        assert!(
            publish_page(
                &mut f,
                &mut state,
                &out,
                &mut NoIo,
                paged_publish_limits(),
                recorder("early-source")
            )
            .is_err()
        );
        assert!(!out.exists());
    }
}

struct MutatingSpool {
    data: Cursor<Vec<u8>>,
    target: Rc<RefCell<Vec<u8>>>,
    out: PathBuf,
    foreign: bool,
    done: bool,
}
impl Read for MutatingSpool {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        assert!(b.len() <= 65536);
        if !self.done {
            assert!(self.out.join(NATIVE_PV_CHECKPOINT_FILE).exists());
            assert!(self.out.join(NATIVE_PV_JOURNAL_FILE).exists());
            let n = self.target.borrow().len();
            self.target.borrow_mut()[n - 1] ^= 1;
            self.done = true;
            if self.foreign {
                fs::write(self.out.join("foreign.keep"), b"keep")?;
            }
        }
        self.data.read(b)
    }
}
impl Seek for MutatingSpool {
    fn seek(&mut self, s: SeekFrom) -> io::Result<u64> {
        self.data.seek(s)
    }
}
impl Write for MutatingSpool {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        assert!(b.len() <= 65536);
        self.data.write(b)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.data.flush()
    }
}

#[test]
fn companion_changes_after_copy_fail_precommit_and_remove_only_owned_paths() {
    let root = Scratch::new("paged-companion-guard");
    for mode in 0..2 {
        let mut f = Fixture::new(
            ScalePrecision::F32,
            Basis::Identity,
            PlaneCodec::D2,
            "full",
            false,
        );
        let cp = Rc::new(RefCell::new(f.checkpoint.clone()));
        let j = Rc::new(RefCell::new(f.journal.clone()));
        let mut session = paged_session(
            &mut f.parent,
            &f.view,
            &f.plan,
            super::super::Shared {
                bytes: cp.clone(),
                position: 0,
            },
        );
        let mut journal = super::super::Shared {
            bytes: j.clone(),
            position: 0,
        };
        let out = root.0.join(format!("child-{mode}"));
        let mut spool = MutatingSpool {
            data: Cursor::new(Vec::new()),
            target: if mode == 0 { cp.clone() } else { j.clone() },
            out: out.clone(),
            foreign: mode == 1,
            done: false,
        };
        let error = publish_native_pv_paged_child(
            &mut session,
            NativePvChildSource {
                parent: &mut f.parent,
                parent_evidence: &f.view,
                replay: &f.replay,
                journal: &mut journal,
                machine: "guard-fixture",
                validation: &validation(),
            },
            &mut spool,
            &mut DirectoryTensorSink::new(&out),
            paged_publish_limits(),
            recorder("late-companion"),
        )
        .unwrap_err();
        match error {
            NativePvChildError::Publication(
                tritium_quantize::engine::TritModelPublicationError::Staging(
                    tritium_format::trit_model::TritModelStageError::Guard(
                        NativePvChildGuardError::Companion(e),
                    ),
                ),
            ) => {
                assert_eq!(e.kind(), io::ErrorKind::InvalidData);
            }
            other => panic!("wrong guard error: {other:?}"),
        }
        for name in [
            "model.trit",
            ".pending.trit",
            "evidence.jsonl",
            NATIVE_PV_CHECKPOINT_FILE,
            NATIVE_PV_JOURNAL_FILE,
        ] {
            assert!(!out.join(name).exists());
        }
        if mode == 1 {
            assert_eq!(fs::read(out.join("foreign.keep")).unwrap(), b"keep");
        } else {
            assert!(!out.exists());
        }
        let changed = if mode == 0 { &cp } else { &j };
        let n = changed.borrow().len();
        changed.borrow_mut()[n - 1] ^= 1;
        assert_eq!(*cp.borrow(), f.checkpoint);
        assert_eq!(*j.borrow(), f.journal);
        assert_eq!(*f.bytes.borrow(), f.original);
    }
}

struct FaultSpool {
    data: Cursor<Vec<u8>>,
    mode: usize,
}
impl Read for FaultSpool {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        if self.mode == 2 {
            return Err(io::Error::other("original spool read"));
        }
        self.data.read(b)
    }
}
impl Seek for FaultSpool {
    fn seek(&mut self, s: SeekFrom) -> io::Result<u64> {
        if self.mode == 3 {
            return Err(io::Error::other("original spool seek"));
        }
        self.data.seek(s)
    }
}
impl Write for FaultSpool {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        if self.mode == 0 {
            return Err(io::Error::other("original spool write"));
        }
        self.data.write(b)
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.mode == 1 {
            return Err(io::Error::other("original spool flush"));
        }
        self.data.flush()
    }
}
#[test]
fn original_spool_errors_and_late_receiving_evidence_caps_leave_no_committed_child() {
    let root = Scratch::new("paged-spool-faults");
    let mut f = Fixture::new(
        ScalePrecision::F32,
        Basis::Identity,
        PlaneCodec::B3,
        "full",
        false,
    );
    let mut session = paged_session(
        &mut f.parent,
        &f.view,
        &f.plan,
        Cursor::new(f.checkpoint.clone()),
    );
    for mode in 0..7 {
        let mut cap = paged_publish_limits();
        match mode {
            4 => cap.max_evidence_bytes = 1,
            5 => cap.receiving.max_source_bytes = 1,
            6 => cap.receiving.max_manifest_bytes = 1,
            _ => {}
        }
        let out = root.0.join(format!("bad-{mode}"));
        let mut spool = FaultSpool {
            data: Cursor::new(Vec::new()),
            mode,
        };
        let error = publish_page(
            &mut f,
            &mut session,
            &out,
            &mut spool,
            cap,
            recorder("fault"),
        )
        .unwrap_err();
        if mode < 4 {
            assert!(error.to_string().contains("original spool"), "{error}");
        }
        assert!(!out.exists());
        assert_eq!(*f.bytes.borrow(), f.original);
        f.replay.verify_paged(&mut session).unwrap();
    }
}
