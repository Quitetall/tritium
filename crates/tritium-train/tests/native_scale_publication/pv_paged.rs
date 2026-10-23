//! Existing deployed-child capability through actual-parent owner paging.
use super::*;
use std::cell::Cell;

#[derive(Debug)]
struct FaultIo {
    data: Cursor<Vec<u8>>,
    read: bool,
    seek: bool,
}
impl Read for FaultIo {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        assert!(b.len() <= 8192, "unbounded checkpoint/journal read");
        if self.read {
            return Err(io::Error::other("original read refusal"));
        }
        self.data.read(b)
    }
}
impl Seek for FaultIo {
    fn seek(&mut self, s: SeekFrom) -> io::Result<u64> {
        if self.seek {
            return Err(io::Error::other("original seek refusal"));
        }
        self.data.seek(s)
    }
}

#[test]
fn caps_original_factory_and_checkpoint_journal_io_fail_without_child_capability() {
    let root = Scratch::new("paged-refusals");
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
        recorder("paged-refusals"),
    )
    .unwrap();
    let encoded = f.parent.tensor_byte_len("full", "weight").unwrap();
    for mode in 0..10 {
        let mut cap = paged_verify_limits();
        match mode {
            0 => cap.max_encoded_owner_bytes = 0,
            1 => cap.max_encoded_owner_bytes = encoded - 1,
            2 => cap.owners.max_working_bytes = 1,
            3 => cap.replay.max_working_bytes = 1,
            _ => {}
        }
        let mut cp = FaultIo {
            data: Cursor::new(f.checkpoint.clone()),
            read: mode == 6,
            seek: mode == 8,
        };
        if mode == 5 {
            let n = cp.data.get_ref().len();
            cp.data.get_mut()[n - 1] ^= 1;
        }
        let before = cp.data.get_ref().clone();
        let mut journal = FaultIo {
            data: Cursor::new(f.journal.clone()),
            read: mode == 7,
            seek: mode == 9,
        };
        let called = Cell::new(0);
        let error = verify_native_pv_paged_child(
            NativePvPagedChildVerification {
                parent: &mut f.parent,
                parent_evidence: &f.view,
                child: &mut child.model,
                log: child.evidence.log(),
                plan: &f.plan,
                journal: &mut journal,
                checkpoint: &mut cp,
            },
            cap,
            |i| {
                called.set(called.get() + 1);
                if mode == 4 && i == 1 {
                    Err("original factory refusal")
                } else {
                    Ok(Cursor::new(Vec::new()))
                }
            },
        )
        .unwrap_err();
        if mode == 4 {
            assert!(matches!(
                error,
                NativePvPagedChildVerificationError::Replay(NativePvPagedReplayError::Workspace(
                    "original factory refusal"
                ))
            ));
            assert!(called.get() > 0);
        } else {
            assert_eq!(called.get(), 0, "mode{mode} must refuse before workspace");
        }
        if mode >= 6 {
            assert!(error.to_string().contains("original"));
        }
        assert_eq!(*cp.data.get_ref(), before);
        assert_eq!(*journal.data.get_ref(), f.journal);
        assert_eq!(*f.bytes.borrow(), f.original);
    }
}

#[test]
fn original_parent_child_checkpoint_and_journal_mutation_during_factory_never_gain_proof() {
    let root = Scratch::new("paged-drift");
    let mut f = Fixture::new(
        ScalePrecision::F32,
        Basis::Identity,
        PlaneCodec::D2,
        "full",
        false,
    );
    let child = publish(
        &mut f,
        &root.0.join("child"),
        &mut Cursor::new(Vec::new()),
        limits(),
        recorder("paged-drift"),
    )
    .unwrap();
    let child_bytes = Rc::new(RefCell::new(
        fs::read(root.0.join("child/model.trit")).unwrap(),
    ));
    let mut actual = TritModelReader::open(
        super::super::Shared {
            bytes: child_bytes.clone(),
            position: 0,
        },
        receiving(),
    )
    .unwrap();
    let checkpoint = Rc::new(RefCell::new(f.checkpoint.clone()));
    let journal = Rc::new(RefCell::new(f.journal.clone()));
    for mode in 0..4 {
        let mut cp = super::super::Shared {
            bytes: checkpoint.clone(),
            position: 0,
        };
        let mut j = super::super::Shared {
            bytes: journal.clone(),
            position: 0,
        };
        let changed = match mode {
            0 => f.bytes.clone(),
            1 => child_bytes.clone(),
            2 => checkpoint.clone(),
            _ => journal.clone(),
        };
        let original = changed.borrow().clone();
        let at = original.len() - 1;
        let result = verify_native_pv_paged_child(
            NativePvPagedChildVerification {
                parent: &mut f.parent,
                parent_evidence: &f.view,
                child: &mut actual,
                log: child.evidence.log(),
                plan: &f.plan,
                journal: &mut j,
                checkpoint: &mut cp,
            },
            paged_verify_limits(),
            |i| {
                if i == 0 {
                    changed.borrow_mut()[at] ^= 1;
                }
                Ok::<_, &str>(Cursor::new(Vec::new()))
            },
        );
        assert!(result.is_err(), "mode{mode} source drift must refuse");
        changed.borrow_mut()[at] ^= 1;
        assert_eq!(*changed.borrow(), original);
        f.view.verify(&mut f.parent).unwrap();
        actual.verify().unwrap();
    }
}

#[test]
fn two_selected_original_owners_produce_the_same_verified_child_as_resident_audit() {
    let root = Scratch::new("paged-multiple");
    let mut f = Fixture::new(
        ScalePrecision::F32,
        Basis::Identity,
        PlaneCodec::B3,
        "full",
        false,
    );
    f.plan.owners = vec!["frozen.weight".into(), "weight".into()];
    f.parents = decode_native_pv_parents(
        &mut f.parent,
        &f.view,
        &f.plan,
        NativePvParentReadLimits {
            max_working_bytes: 2_000_000,
            extra_payload_bytes: 0,
        },
    )
    .unwrap();
    (f.session, f.journal, f.checkpoint, f.replay) = trained(&f.plan, &f.parents, 0.125);
    let mut child = publish(
        &mut f,
        &root.0.join("child"),
        &mut Cursor::new(Vec::new()),
        limits(),
        recorder("paged-multiple"),
    )
    .unwrap();
    let expected = verify_native_pv_child(
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
    let actual = paged_verify(NativePvPagedChildVerification {
        parent: &mut f.parent,
        parent_evidence: &f.view,
        child: &mut child.model,
        log: child.evidence.log(),
        plan: &f.plan,
        journal: &mut Cursor::new(&f.journal),
        checkpoint: &mut Cursor::new(&f.checkpoint),
    })
    .unwrap();
    assert_eq!(actual.package(), expected.package());
    assert_eq!(actual.replay().summary(), expected.replay().summary());
}
