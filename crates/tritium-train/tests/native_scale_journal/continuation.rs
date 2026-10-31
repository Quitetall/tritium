//! Continuation reproduces actual history, not just a resealed checkpoint.
use super::*;
use tritium_train::pv_tuning::{
    NativeScaleJournalResume as Resume, NativeScaleResumeTail as Tail,
    resume_native_scale_journal as resume,
};

fn saved(parents: &[AdditiveTensor], cursor: usize) -> (Vec<u8>, Vec<u8>, NativeScaleResumePoint) {
    let mut session = Session::new(plan(parents.len()), parents.to_vec()).unwrap();
    let io = FaultIo::new();
    let mut writer = Writer::new(io.clone(), &session, limits()).unwrap();
    for step in 0..cursor {
        let g = gradients(&session, step);
        session
            .step_recorded(
                &g.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                &mut writer,
            )
            .unwrap();
    }
    let (checkpoint, point) = writer.checkpoint(&session).unwrap();
    let prefix = io.data.borrow().get_ref().clone();
    assert_eq!(point.journal_prefix_bytes, prefix.len() as u64);
    assert_eq!(point.journal_prefix, digest(&prefix));
    assert_eq!(point.checkpoint, digest(&checkpoint));
    assert_eq!(point.completed_updates, cursor as u64);
    (prefix, checkpoint, point)
}

fn continue_from(parents: &[AdditiveTensor], cursor: usize) {
    let (prefix, checkpoint, point) = saved(parents, cursor);
    let (mut session, mut writer) = resume(
        Resume {
            plan: plan(parents.len()),
            parents: parents.to_vec(),
            source: Cursor::new(&prefix),
            point: &point,
            checkpoint: &checkpoint,
            tail: Tail::ExactPrefix,
        },
        Cursor::new(Vec::new()),
        limits(),
    )
    .unwrap();
    assert_eq!(session.checkpoint_bytes().unwrap(), checkpoint);
    assert_eq!(writer.checkpoint(&session).unwrap().1, point);
    for step in cursor..4 {
        let g = gradients(&session, step);
        session
            .step_recorded(
                &g.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                &mut writer,
            )
            .unwrap();
    }
    let (output, summary) = writer.finish(&session).unwrap();
    let (_, expected, bytes, wanted) = run(parents.to_vec());
    assert_eq!(
        session.checkpoint_bytes().unwrap(),
        expected.checkpoint_bytes().unwrap()
    );
    assert_eq!(output.into_inner(), bytes);
    assert_eq!(summary, wanted);
}

#[test]
fn every_cursor_and_648_profiles_match_uninterrupted_bytes() {
    for cursor in 0..=4 {
        let mut second = simple();
        second
            .replace_scales(vec![0.75; second.scales().len()])
            .unwrap();
        continue_from(&[simple(), second], cursor);
    }
    let mut cases = 0;
    for law in 0..ADMITTED_LAWS.len() {
        for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
            for basis in [
                Basis::Identity,
                Basis::Hadamard { block: 128 },
                Basis::SignedRht {
                    block: 128,
                    seed: 17,
                    domain: 9,
                },
            ] {
                for mode in 0..3 {
                    for transport in [
                        Transport::Raw,
                        Transport::JointHuffman { block: 128 },
                        Transport::JointHuffman { block: 256 },
                    ] {
                        for group in [32, 128] {
                            continue_from(&[tensor(law, codec, basis, mode, transport, group)], 2);
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases, 648);
    println!("NATIVE_SCALE_CONTINUATION_PROFILES={cases}");
}

#[test]
fn suffix_is_refused_unless_explicitly_branched_and_never_modified() {
    assert_eq!(Tail::default(), Tail::ExactPrefix);
    let parents = [simple()];
    let (prefix, checkpoint, point) = saved(&parents, 2);
    let (_, _, full, _) = run(parents.to_vec());
    assert_eq!(&full[..prefix.len()], &prefix);
    for source in [full, [prefix.clone(), vec![1, 2, 3]].concat()] {
        let original = source.clone();
        for tail in [Tail::ExactPrefix, Tail::CheckpointPrefix] {
            let mut input = Cursor::new(source.clone());
            let result = resume(
                Resume {
                    plan: plan(1),
                    parents: parents.to_vec(),
                    source: &mut input,
                    point: &point,
                    checkpoint: &checkpoint,
                    tail,
                },
                Cursor::new(Vec::new()),
                limits(),
            );
            assert_eq!(input.into_inner(), original);
            match tail {
                Tail::ExactPrefix => assert!(result.is_err()),
                Tail::CheckpointPrefix => {
                    let (mut session, mut writer) = result.unwrap();
                    for step in 2..4 {
                        let g = gradients(&session, step);
                        session
                            .step_recorded(
                                &g.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                                &mut writer,
                            )
                            .unwrap();
                    }
                    assert_eq!(
                        writer.finish(&session).unwrap().0.into_inner(),
                        run(parents.to_vec()).2
                    );
                }
                _ => unreachable!(),
            }
        }
    }
}

#[test]
fn wrong_observations_parents_and_every_changed_prefix_byte_fail() {
    let parents = [simple()];
    let (prefix, checkpoint, point) = saved(&parents, 2);
    let check = |source: Vec<u8>,
                 checkpoint: &[u8],
                 point: &NativeScaleResumePoint,
                 parents: Vec<AdditiveTensor>| {
        resume(
            Resume {
                plan: plan(1),
                parents,
                source: Cursor::new(source),
                point,
                checkpoint,
                tail: Tail::ExactPrefix,
            },
            Cursor::new(Vec::new()),
            limits(),
        )
    };
    for i in 0..prefix.len() {
        let mut changed = prefix.clone();
        changed[i] ^= 1;
        assert!(check(changed, &checkpoint, &point, parents.to_vec()).is_err());
    }
    for n in [0, 1, prefix.len() - 1] {
        assert!(check(prefix[..n].to_vec(), &checkpoint, &point, parents.to_vec()).is_err());
    }
    for i in 0..6 {
        let mut wrong = point.clone();
        match i {
            0 => wrong.plan = NativeRefinementPlanId::from_bytes([0; 32]),
            1 => wrong.completed_updates = 5,
            2 => wrong.journal_prefix = digest(b"wrong"),
            3 => wrong.journal_prefix_bytes += 1,
            4 => wrong.checkpoint = digest(b"wrong"),
            5 => wrong.checkpoint_bytes += 1,
            _ => unreachable!(),
        }
        assert!(check(prefix.clone(), &checkpoint, &wrong, parents.to_vec()).is_err());
    }
    let mut other = simple();
    other
        .replace_scales(vec![0.75; other.scales().len()])
        .unwrap();
    assert!(check(prefix.clone(), &checkpoint, &point, vec![other]).is_err());
    // A syntactically valid, correctly resealed checkpoint still needs actual history.
    let mut forged = checkpoint.clone();
    let n = forged.len();
    forged[n - 36..n - 32].copy_from_slice(&0.125f32.to_le_bytes());
    reseal(&mut forged);
    Session::resume(plan(1), parents.to_vec(), &forged).unwrap();
    let mut wrong = point.clone();
    wrong.checkpoint = digest(&forged);
    assert!(
        check(prefix.clone(), &forged, &wrong, parents.to_vec())
            .unwrap_err()
            .to_string()
            .contains("replayed checkpoint")
    );
    // Rehashing a malformed frame/header does not bypass the shared strict parser.
    for offset in [0, 8, 40, 72, 80, 88, 96, 104, 136] {
        let mut changed = prefix.clone();
        changed[offset] ^= 1;
        let mut wrong = point.clone();
        wrong.journal_prefix = digest(&changed);
        assert!(check(changed, &checkpoint, &wrong, parents.to_vec()).is_err());
    }
}

#[test]
fn caps_are_admitted_before_io_and_nonempty_output_is_preserved() {
    let parents = [simple()];
    let (prefix, checkpoint, point) = saved(&parents, 2);
    let ledger = Session::new(plan(1), parents.to_vec())
        .unwrap()
        .journal_size_ledger()
        .unwrap();
    for cap in [
        Limits {
            max_journal_bytes: ledger.journal_bytes - 1,
            max_working_bytes: u64::MAX,
        },
        Limits {
            max_journal_bytes: u64::MAX,
            max_working_bytes: ledger.replay_working_bytes - 1,
        },
    ] {
        assert!(matches!(
            resume(
                Resume {
                    plan: plan(1),
                    parents: parents.to_vec(),
                    source: NoIo,
                    point: &point,
                    checkpoint: &checkpoint,
                    tail: Tail::ExactPrefix
                },
                NoIo,
                cap
            ),
            Err(Error::Limit(_))
        ));
    }
    let mut output = Cursor::new(vec![17, 23]);
    assert!(
        resume(
            Resume {
                plan: plan(1),
                parents: parents.to_vec(),
                source: Cursor::new(&prefix),
                point: &point,
                checkpoint: &checkpoint,
                tail: Tail::ExactPrefix
            },
            &mut output,
            limits()
        )
        .is_err()
    );
    assert_eq!(output.into_inner(), vec![17, 23]);
    resume(
        Resume {
            plan: plan(1),
            parents: parents.to_vec(),
            source: Cursor::new(&prefix),
            point: &point,
            checkpoint: &checkpoint,
            tail: Tail::ExactPrefix,
        },
        Cursor::new(Vec::new()),
        Limits {
            max_journal_bytes: ledger.journal_bytes,
            max_working_bytes: ledger.replay_working_bytes,
        },
    )
    .unwrap();
}

struct MutatingSource {
    source: Cursor<Vec<u8>>,
    starts: usize,
    mutate_at: usize,
}
impl Read for MutatingSource {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        assert!(bytes.len() <= 8192);
        self.source.read(bytes)
    }
}
impl Seek for MutatingSource {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        if from == SeekFrom::Start(0) {
            self.starts += 1;
            if self.starts == self.mutate_at {
                self.source.get_mut()[136] ^= 1;
            }
        }
        self.source.seek(from)
    }
}

#[test]
fn changed_source_and_partial_output_or_flush_fail_without_a_usable_writer() {
    let parents = [simple()];
    let (prefix, checkpoint, point) = saved(&parents, 2);
    for mutate_at in [2, 3] {
        assert!(
            resume(
                Resume {
                    plan: plan(1),
                    parents: parents.to_vec(),
                    source: MutatingSource {
                        source: Cursor::new(prefix.clone()),
                        starts: 0,
                        mutate_at
                    },
                    point: &point,
                    checkpoint: &checkpoint,
                    tail: Tail::ExactPrefix
                },
                Cursor::new(Vec::new()),
                limits()
            )
            .is_err()
        );
    }
    for flush in [false, true] {
        let io = FaultIo::new();
        if flush {
            io.flush.set(true);
        } else {
            io.remaining.set(5);
        }
        let result = resume(
            Resume {
                plan: plan(1),
                parents: parents.to_vec(),
                source: Cursor::new(&prefix),
                point: &point,
                checkpoint: &checkpoint,
                tail: Tail::ExactPrefix,
            },
            io.clone(),
            limits(),
        );
        assert!(matches!(result, Err(Error::Io(_))));
        assert_eq!(
            io.data.borrow().get_ref().len(),
            if flush { prefix.len() } else { 5 }
        );
    }
}

#[test]
fn failed_snapshot_flush_preserves_state_and_poison_is_not_a_resume_point() {
    let session = Session::new(plan(1), vec![simple()]).unwrap();
    let io = FaultIo::new();
    let mut writer = Writer::new(io.clone(), &session, limits()).unwrap();
    let before = session.checkpoint_bytes().unwrap();
    io.flush.set(true);
    assert!(matches!(writer.checkpoint(&session), Err(Error::Io(_))));
    assert_eq!(session.checkpoint_bytes().unwrap(), before);
    io.flush.set(false);
    assert!(
        writer
            .checkpoint(&session)
            .unwrap_err()
            .to_string()
            .contains("poisoned")
    );
}

struct FailingSource {
    source: Cursor<Vec<u8>>,
    reads: usize,
    seeks: usize,
}
impl Read for FailingSource {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        if self.reads == 0 {
            return Err(io::Error::other("injected resume read"));
        }
        let available = bytes.len().min(self.reads);
        let n = self.source.read(&mut bytes[..available])?;
        self.reads -= n;
        Ok(n)
    }
}
impl Seek for FailingSource {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        if self.seeks == 0 {
            return Err(io::Error::other("injected resume seek"));
        }
        self.seeks -= 1;
        self.source.seek(from)
    }
}

#[test]
fn source_read_and_seek_failures_preserve_original_errors_across_all_passes() {
    let parents = [simple()];
    let (prefix, checkpoint, point) = saved(&parents, 2);
    for (reads, seeks, expected) in [
        (0, usize::MAX, "injected resume read"),
        (prefix.len(), usize::MAX, "injected resume read"),
        (prefix.len() * 2, usize::MAX, "injected resume read"),
        (usize::MAX, 0, "injected resume seek"),
        (usize::MAX, 1, "injected resume seek"),
        (usize::MAX, 2, "injected resume seek"),
        (usize::MAX, 3, "injected resume seek"),
        (usize::MAX, 4, "injected resume seek"),
        (usize::MAX, 5, "injected resume seek"),
    ] {
        let error = resume(
            Resume {
                plan: plan(1),
                parents: parents.to_vec(),
                source: FailingSource {
                    source: Cursor::new(prefix.clone()),
                    reads,
                    seeks,
                },
                point: &point,
                checkpoint: &checkpoint,
                tail: Tail::ExactPrefix,
            },
            Cursor::new(Vec::new()),
            limits(),
        )
        .unwrap_err();
        assert!(matches!(error, Error::Io(_)));
        assert_eq!(error.to_string(), expected);
    }
}
