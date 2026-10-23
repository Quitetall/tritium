//! Actual numerical replay/recovery from originals, not model qualification.
use super::*;
use tritium_train::pv_tuning::{NativePvJournalSummary, NativePvPagedReplayError};
fn cap() -> NativePvPagedReplayLimits {
    NativePvPagedReplayLimits {
        max_journal_bytes: 1 << 30,
        max_working_bytes: 16384,
        paging: limits(),
    }
}
type Saved = (Vec<u8>, NativePvResumePoint);
fn history(
    p: &NativeRefinementPlan,
    weights: &[AdditiveTensor],
) -> (Vec<u8>, NativePvJournalSummary, Vec<Saved>, Resident) {
    let mut current = Resident::new(p.clone(), weights.to_vec()).unwrap();
    let mut journal = NativePvJournalWriter::new(
        output(),
        &current,
        NativePvJournalLimits {
            max_journal_bytes: 1 << 30,
            max_working_bytes: 1 << 30,
        },
    )
    .unwrap();
    let mut saved = vec![journal.checkpoint(&current).unwrap()];
    for step in 0..p.training.len() {
        let g: Vec<_> = current
            .weights()
            .enumerate()
            .map(|(i, w)| grad(i + step, w))
            .collect();
        current
            .step_recorded(
                &g.iter().map(|(s, _)| s.as_slice()).collect::<Vec<_>>(),
                &g.iter().map(|(_, w)| w.as_slice()).collect::<Vec<_>>(),
                &mut journal,
            )
            .unwrap();
        saved.push(journal.checkpoint(&current).unwrap());
    }
    let (journal, summary) = journal.finish(&current).unwrap();
    (journal.into_inner(), summary, saved, current)
}

#[test]
fn complete_and_every_prefix_replay_reproduce_exact_state_and_continued_journal_all_profiles() {
    let _serial = SERIAL.lock().unwrap();
    let mut cases = 0;
    for law in 0..ADMITTED_LAWS.len() {
        for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
            for basis in [
                Basis::Identity,
                Basis::Hadamard { block: 128 },
                Basis::SignedRht {
                    block: 128,
                    seed: u64::MAX,
                    domain: 9,
                },
            ] {
                let p = plan(3);
                let weights: Vec<_> = (0..3).map(|i| tensor(law, codec, basis, i, 2)).collect();
                let (journal, summary, saved, reference) = history(&p, &weights);
                let mut checkpoint = Cursor::new(saved[4].0.clone());
                let (mut actual, proof) = replay_native_pv_paged_journal(
                    NativePvPagedJournalReplay {
                        plan: p.clone(),
                        source: Cursor::new(&journal),
                        expected_journal: summary.journal(),
                        checkpoint: &mut checkpoint,
                    },
                    parents(&weights),
                    |_| Ok(output()),
                    cap(),
                )
                .unwrap();
                assert_eq!(proof.summary(), &summary);
                proof.verify_paged(&mut actual).unwrap();
                proof.verify_session(&reference).unwrap();
                assert_eq!(actual.into_checkpoint().into_inner(), saved[4].0);
                for cursor in 0..=4 {
                    for tail in [
                        NativePvResumeTail::ExactPrefix,
                        NativePvResumeTail::CheckpointPrefix,
                    ] {
                        let (cp, point) = &saved[cursor];
                        let prefix = point.journal_prefix_bytes as usize;
                        let source = if tail == NativePvResumeTail::ExactPrefix {
                            &journal[..prefix]
                        } else {
                            &journal[..]
                        };
                        let mut target = Cursor::new(cp.clone());
                        let (mut current, mut writer) = resume_native_pv_paged_journal(
                            NativePvPagedJournalResume {
                                plan: p.clone(),
                                source: Cursor::new(source),
                                point,
                                checkpoint: &mut target,
                                tail,
                            },
                            output(),
                            parents(&weights),
                            |_| Ok(output()),
                            cap(),
                        )
                        .unwrap();
                        assert_eq!(writer.checkpoint(&mut current).unwrap(), *point);
                        for step in cursor..4 {
                            let ready = current
                                .stage_recorded(
                                    (output(), output()),
                                    parents(&weights),
                                    |i, _, w| Ok(grad(i + step, w)),
                                    |_, _, _| Ok(()),
                                    RecordingLimits {
                                        max_journal_bytes: 1 << 30,
                                        max_frame_bytes: 1 << 29,
                                        max_working_bytes: 16384,
                                    },
                                )
                                .unwrap();
                            let (next, _, _) = ready.commit(&mut current, &mut writer).unwrap();
                            current = next;
                        }
                        let (output, observed) = writer.finish(&mut current).unwrap();
                        assert_eq!(observed, summary);
                        assert_eq!(output.into_inner(), journal);
                        assert_eq!(current.into_checkpoint().into_inner(), saved[4].0);
                        assert_eq!(target.into_inner(), *cp);
                    }
                }
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 36);
}

#[derive(Clone)]
struct Input {
    io: Shared,
    pass: Rc<Cell<u64>>,
    alter_pass: Rc<Cell<u64>>,
    alter_at: usize,
}
impl Input {
    fn new(bytes: Vec<u8>, at: usize) -> Self {
        Self {
            io: Shared::new(bytes),
            pass: Rc::new(Cell::new(0)),
            alter_pass: Rc::new(Cell::new(0)),
            alter_at: at,
        }
    }
}
impl Read for Input {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        let at = self.io.inner.borrow().position() as usize;
        let n = self.io.read(b)?;
        if self.pass.get() == self.alter_pass.get() && self.alter_at >= at && self.alter_at < at + n
        {
            b[self.alter_at - at] ^= 1;
        }
        Ok(n)
    }
}
impl Seek for Input {
    fn seek(&mut self, s: SeekFrom) -> io::Result<u64> {
        if matches!(s, SeekFrom::Start(0)) {
            self.pass.set(self.pass.get() + 1);
        }
        self.io.seek(s)
    }
}

#[test]
fn corrupt_or_resealed_forged_state_wrong_originals_caps_and_tails_never_gain_replay_or_writer() {
    let _serial = SERIAL.lock().unwrap();
    let p = plan(2);
    let weights: Vec<_> = (0..2)
        .map(|_| tensor(0, PlaneCodec::B3, Basis::Identity, 0, 2))
        .collect();
    let (journal, summary, saved, _) = history(&p, &weights);
    for mode in 0..6 {
        let mut source = journal.clone();
        if mode == 0 {
            source[0] ^= 1;
        }
        if mode == 1 {
            source.truncate(source.len() - 1);
        }
        if mode == 2 {
            source[88] ^= 1;
        }
        let mut cp = saved[4].0.clone();
        if mode == 3 {
            let n = cp.len();
            cp[n - 36..n - 32].copy_from_slice(&0.001f32.to_le_bytes());
            let hash = blake3::hash(&cp[..n - 32]);
            cp[n - 32..].copy_from_slice(hash.as_bytes());
        }
        let mut originals = weights.clone();
        if mode == 4 {
            let mut scales = originals[1].scales().to_vec();
            scales[0] = 0.25;
            originals[1].replace_scales(scales).unwrap();
        }
        let id = if mode <= 2 {
            BlobId::from_bytes(*blake3::hash(&source).as_bytes())
        } else {
            summary.journal()
        };
        let mut target = Cursor::new(cp);
        let mut limits = cap();
        if mode == 5 {
            limits.max_working_bytes = 16383;
        }
        let workspace_called = Cell::new(false);
        assert!(
            replay_native_pv_paged_journal(
                NativePvPagedJournalReplay {
                    plan: p.clone(),
                    source: Cursor::new(source),
                    expected_journal: id,
                    checkpoint: &mut target
                },
                parents(&originals),
                |_| {
                    workspace_called.set(true);
                    Ok(output())
                },
                limits
            )
            .is_err()
        );
        if mode != 3 {
            assert!(
                !workspace_called.get(),
                "mode {mode} must fail in preflight"
            );
        }
    }
    let (cp, point) = &saved[2];
    let mut target = Cursor::new(cp.clone());
    assert!(
        resume_native_pv_paged_journal(
            NativePvPagedJournalResume {
                plan: p.clone(),
                source: Cursor::new(&journal),
                point,
                checkpoint: &mut target,
                tail: NativePvResumeTail::ExactPrefix
            },
            output(),
            parents(&weights),
            |_| Ok(output()),
            cap()
        )
        .is_err()
    );
    let mut forged = cp.clone();
    let n = forged.len();
    forged[n - 36..n - 32].copy_from_slice(&0.001f32.to_le_bytes());
    let hash = blake3::hash(&forged[..n - 32]);
    forged[n - 32..].copy_from_slice(hash.as_bytes());
    let mut fake_point = point.clone();
    fake_point.checkpoint = BlobId::from_bytes(*blake3::hash(&forged).as_bytes());
    let mut fake = Cursor::new(forged);
    assert!(
        resume_native_pv_paged_journal(
            NativePvPagedJournalResume {
                plan: p.clone(),
                source: Cursor::new(&journal),
                point: &fake_point,
                checkpoint: &mut fake,
                tail: NativePvResumeTail::CheckpointPrefix
            },
            output(),
            parents(&weights),
            |_| Ok(output()),
            cap()
        )
        .is_err()
    );
    let mut target = Cursor::new(cp.clone());
    assert!(
        resume_native_pv_paged_journal(
            NativePvPagedJournalResume {
                plan: p.clone(),
                source: Cursor::new(&journal),
                point,
                checkpoint: &mut target,
                tail: NativePvResumeTail::CheckpointPrefix
            },
            Cursor::new(vec![1]),
            parents(&weights),
            |_| Ok(output()),
            cap()
        )
        .is_err()
    );
    for bad in [
        NativePvPagedReplayLimits {
            max_working_bytes: 0,
            ..cap()
        },
        NativePvPagedReplayLimits {
            max_journal_bytes: 1,
            ..cap()
        },
    ] {
        let called = Cell::new(false);
        let mut target = Cursor::new(saved[4].0.clone());
        assert!(
            replay_native_pv_paged_journal(
                NativePvPagedJournalReplay {
                    plan: p.clone(),
                    source: Cursor::new(&journal),
                    expected_journal: summary.journal(),
                    checkpoint: &mut target
                },
                |i, _| {
                    called.set(true);
                    Ok::<_, &str>(weights[i].clone())
                },
                |_| Ok(output()),
                bad
            )
            .is_err()
        );
        assert!(!called.get());
    }
}

#[test]
fn transient_traversal_or_prefix_copy_changes_and_original_factory_errors_preserve_sources() {
    let _serial = SERIAL.lock().unwrap();
    let p = plan(2);
    let weights: Vec<_> = (0..2)
        .map(|_| tensor(0, PlaneCodec::D2, Basis::Identity, 0, 2))
        .collect();
    let (journal, summary, saved, _) = history(&p, &weights);
    let at = 88 + 16 * 2 + 40;
    let source = Input::new(journal.clone(), at);
    source.alter_pass.set(4); // numerical traversal, after canonical preflight
    let mut cp = Cursor::new(saved[4].0.clone());
    assert!(
        replay_native_pv_paged_journal(
            NativePvPagedJournalReplay {
                plan: p.clone(),
                source: source.clone(),
                expected_journal: summary.journal(),
                checkpoint: &mut cp
            },
            parents(&weights),
            |_| Ok(output()),
            cap()
        )
        .is_err()
    );
    assert_eq!(source.io.contents(), journal);
    let source = Input::new(journal.clone(), at);
    source.alter_pass.set(8); // actual prefix transfer, after numerical replay
    let (checkpoint, point) = &saved[2];
    let mut cp = Cursor::new(checkpoint.clone());
    assert!(
        resume_native_pv_paged_journal(
            NativePvPagedJournalResume {
                plan: p.clone(),
                source: source.clone(),
                point,
                checkpoint: &mut cp,
                tail: NativePvResumeTail::CheckpointPrefix
            },
            output(),
            parents(&weights),
            |_| Ok(output()),
            cap()
        )
        .is_err()
    );
    assert_eq!(source.io.contents(), journal);
    assert_eq!(cp.into_inner(), *checkpoint);
    let mut cp = Cursor::new(saved[4].0.clone());
    let result = replay_native_pv_paged_journal(
        NativePvPagedJournalReplay {
            plan: p,
            source: Cursor::new(&journal),
            expected_journal: summary.journal(),
            checkpoint: &mut cp,
        },
        parents(&weights),
        |i| {
            if i == 2 {
                Err("original factory refusal")
            } else {
                Ok(output())
            }
        },
        cap(),
    );
    assert!(matches!(
        result,
        Err(NativePvPagedReplayError::Workspace(
            "original factory refusal"
        ))
    ));
    assert_eq!(cp.into_inner(), saved[4].0);
}

#[test]
fn source_target_workspace_and_copy_io_errors_preserve_original_evidence() {
    let _serial = SERIAL.lock().unwrap();
    let p = plan(2);
    let weights: Vec<_> = (0..2)
        .map(|_| tensor(0, PlaneCodec::B3, Basis::Identity, 0, 2))
        .collect();
    let (journal, _, saved, _) = history(&p, &weights);
    let (checkpoint, point) = &saved[2];
    for mode in 0..10 {
        let source = Shared::new(journal.clone());
        let mut target = Shared::new(checkpoint.clone());
        let destination = Shared::new(Vec::new());
        source.fail_read.set(mode == 0);
        source.fail_seek.set(mode == 1);
        target.fail_read.set(mode == 2);
        target.fail_seek.set(mode == 3);
        if mode == 8 {
            destination.fail_write.set(Some(1));
        }
        destination.fail_flush.set(mode == 9);
        let result = resume_native_pv_paged_journal(
            NativePvPagedJournalResume {
                plan: p.clone(),
                source: source.clone(),
                point,
                checkpoint: &mut target,
                tail: NativePvResumeTail::CheckpointPrefix,
            },
            destination,
            parents(&weights),
            |ordinal| {
                let work = Shared::new(Vec::new());
                if ordinal == 1 {
                    if mode == 4 {
                        work.fail_write.set(Some(90));
                    }
                    work.fail_flush.set(mode == 5);
                    work.fail_read.set(mode == 6);
                    work.fail_seek.set(mode == 7);
                }
                Ok(work)
            },
            cap(),
        );
        let error = result.expect_err("I/O fault must not return live state/writer");
        assert!(
            error.to_string().contains("injected"),
            "mode {mode}: {error}"
        );
        assert!(matches!(
            error,
            NativePvPagedReplayError::Journal(tritium_train::pv_tuning::NativePvJournalError::Io(
                _
            )) | NativePvPagedReplayError::Paged(NativePvPagedError::Io(_))
        ));
        assert_eq!(source.contents(), journal);
        assert_eq!(target.contents(), *checkpoint);
    }
}

#[test]
fn file_backed_replay_residency_is_owner_bounded_not_checkpoint_or_gradient_window_sized() {
    let _serial = SERIAL.lock().unwrap();
    rayon::join(|| (), || ());
    let files = Files::new();
    let mut p = plan(8);
    p.training.truncate(1);
    let mut initial = Paged::initialize(
        p.clone(),
        limits(),
        |_, _| Ok::<_, &str>(big()),
        files.create("record0.tnpv"),
    )
    .unwrap();
    let record_cap = RecordingLimits {
        max_journal_bytes: 1 << 30,
        max_frame_bytes: 1 << 29,
        max_working_bytes: 16384,
    };
    let mut writer =
        PagedWriter::new(files.create("journal.tnpj"), &mut initial, record_cap).unwrap();
    let ready = initial
        .stage_recorded(
            (files.create("record1.tnpv"), files.create("frame.tmp")),
            |_, _| Ok::<_, &str>(big()),
            |i, _, w| Ok(grad(i, w)),
            |_, _, _| Ok(()),
            record_cap,
        )
        .unwrap();
    let (mut final_state, _, _) = ready.commit(&mut initial, &mut writer).unwrap();
    let point = writer.checkpoint(&mut final_state).unwrap();
    let (_, summary) = writer.finish(&mut final_state).unwrap();
    let checkpoint_bytes = final_state.size_ledger().logical.checkpoint_bytes;
    drop(final_state);
    drop(initial);
    let mut expected = File::open(files.0.join("record1.tnpv")).unwrap();
    let source = File::open(files.0.join("journal.tnpj")).unwrap();
    let (result, peak, retained) = measured(|| {
        replay_native_pv_paged_journal(
            NativePvPagedJournalReplay {
                plan: p.clone(),
                source,
                expected_journal: summary.journal(),
                checkpoint: &mut expected,
            },
            |_, _| Ok::<_, &str>(big()),
            |i| Ok(files.create(&format!("replay{i}.tnpv"))),
            cap(),
        )
    });
    let (mut replayed, proof) = result.unwrap();
    proof.verify_paged(&mut replayed).unwrap();
    assert_eq!(proof.summary(), &summary);
    assert!(retained < 65536, "retained replay payload {retained}");
    assert!(
        peak as u64 <= replayed.size_ledger().max_owner_payload_bytes + 2 * 8192 + 65536,
        "replay buffered window/checkpoint {peak}"
    );
    assert!((peak as u64) < checkpoint_bytes);
    eprintln!(
        "PAGED_REPLAY owners=8 checkpoint_bytes={checkpoint_bytes} peak_live_allocator_requests={peak} retained_allocator_requests={retained}"
    );
    let output = files.create("recovered.tnpj");
    let source = File::open(files.0.join("journal.tnpj")).unwrap();
    let (result, prefix_peak, prefix_retained) = measured(|| {
        resume_native_pv_paged_journal(
            NativePvPagedJournalResume {
                plan: p.clone(),
                source,
                point: &point,
                checkpoint: &mut expected,
                tail: NativePvResumeTail::CheckpointPrefix,
            },
            output,
            |_, _| Ok::<_, &str>(big()),
            |i| Ok(files.create(&format!("recover{i}.tnpv"))),
            cap(),
        )
    });
    let (mut recovered, writer) = result.unwrap();
    assert!(
        (prefix_peak as u64) <= recovered.size_ledger().max_owner_payload_bytes + 2 * 8192 + 65536
    );
    assert!(prefix_retained < 65536);
    let (_, observed) = writer.finish(&mut recovered).unwrap();
    assert_eq!(observed, summary);
    eprintln!(
        "PAGED_PREFIX_RECOVERY owners=8 prefix_bytes={} peak_live_allocator_requests={prefix_peak} retained_allocator_requests={prefix_retained}",
        point.journal_prefix_bytes
    );
}
