//! Existing-wire parity and atomic staged recording, not model qualification.
use super::*;
fn cap() -> RecordingLimits {
    RecordingLimits {
        max_journal_bytes: 1 << 30,
        max_frame_bytes: 1 << 29,
        max_working_bytes: 16384,
    }
}

#[test]
fn all_pruned_owners_record_empty_adjoints_without_skipping_declared_windows() {
    let _serial = SERIAL.lock().unwrap();
    let p = plan(2);
    let weights: Vec<_> = (0..2)
        .map(|_| tensor(0, PlaneCodec::B3, Basis::Identity, 2, 2))
        .collect();
    let mut reference = Resident::new(p.clone(), weights.clone()).unwrap();
    let mut resident = NativePvJournalWriter::new(
        output(),
        &reference,
        NativePvJournalLimits {
            max_journal_bytes: 1 << 30,
            max_working_bytes: 1 << 30,
        },
    )
    .unwrap();
    let mut current = Paged::initialize(p.clone(), limits(), parents(&weights), output()).unwrap();
    assert_eq!(current.recording_size_ledger().unwrap().frame_bytes, 72);
    let mut writer = PagedWriter::new(output(), &mut current, cap()).unwrap();
    for _ in 0..4 {
        reference
            .step_recorded(&[&[], &[]], &[&[], &[]], &mut resident)
            .unwrap();
        let ready = current
            .stage_recorded(
                (output(), output()),
                parents(&weights),
                |_, _, _| Ok((Vec::new(), Vec::new())),
                |_, _, _| Ok(()),
                cap(),
            )
            .unwrap();
        let (next, _, _) = ready.commit(&mut current, &mut writer).unwrap();
        current = next;
        assert_eq!(
            resident.checkpoint(&reference).unwrap().1,
            writer.checkpoint(&mut current).unwrap()
        );
    }
    let (actual, summary) = writer.finish(&mut current).unwrap();
    let (expected, other) = resident.finish(&reference).unwrap();
    assert_eq!(actual.into_inner(), expected.into_inner());
    assert_eq!(summary, other);
    assert_eq!(
        current.into_checkpoint().into_inner(),
        reference.checkpoint_bytes().unwrap()
    );
}

#[test]
fn recorded_paged_bytes_points_statistics_and_standard_replay_match_all_native_profiles() {
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
                let mut reference = Resident::new(p.clone(), weights.clone()).unwrap();
                let mut resident_writer = NativePvJournalWriter::new(
                    output(),
                    &reference,
                    NativePvJournalLimits {
                        max_journal_bytes: 1 << 30,
                        max_working_bytes: 1 << 30,
                    },
                )
                .unwrap();
                let mut current =
                    Paged::initialize(p.clone(), limits(), parents(&weights), output()).unwrap();
                let mut writer = PagedWriter::new(output(), &mut current, cap()).unwrap();
                assert_eq!(
                    current.recording_size_ledger().unwrap().journal_bytes,
                    reference.journal_size_ledger().unwrap().journal_bytes
                );
                for step in 0..4 {
                    let (before, point) = resident_writer.checkpoint(&reference).unwrap();
                    assert_eq!(writer.checkpoint(&mut current).unwrap(), point);
                    let g: Vec<_> = reference
                        .weights()
                        .enumerate()
                        .map(|(i, w)| grad(i + step, w))
                        .collect();
                    let expected = reference
                        .step_recorded(
                            &g.iter().map(|(s, _)| s.as_slice()).collect::<Vec<_>>(),
                            &g.iter().map(|(_, w)| w.as_slice()).collect::<Vec<_>>(),
                            &mut resident_writer,
                        )
                        .unwrap();
                    let ready = current
                        .stage_recorded(
                            (output(), output()),
                            parents(&weights),
                            |i, _, _| Ok(g[i].clone()),
                            |_, _, _| Ok(()),
                            cap(),
                        )
                        .unwrap();
                    assert_eq!(writer.checkpoint(&mut current).unwrap(), point);
                    let (next, stats, resources) = ready.commit(&mut current, &mut writer).unwrap();
                    assert_eq!(stats, expected);
                    assert_eq!(resources.len(), 3);
                    assert_eq!(current.into_checkpoint().into_inner(), before);
                    current = next;
                    let (_, point) = resident_writer.checkpoint(&reference).unwrap();
                    assert_eq!(writer.checkpoint(&mut current).unwrap(), point);
                }
                let (actual, summary) = writer.finish(&mut current).unwrap();
                let (expected, expected_summary) = resident_writer.finish(&reference).unwrap();
                assert_eq!(actual.into_inner(), expected.clone().into_inner());
                assert_eq!(summary, expected_summary);
                let checkpoint = current.into_checkpoint().into_inner();
                let (replayed, _) = replay_native_pv_journal_with_state(
                    p,
                    weights,
                    expected,
                    summary.journal(),
                    &checkpoint,
                    NativePvJournalLimits {
                        max_journal_bytes: 1 << 30,
                        max_working_bytes: 1 << 30,
                    },
                )
                .unwrap();
                assert_eq!(replayed.checkpoint_bytes().unwrap(), checkpoint);
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 36);
}

type Action = Rc<RefCell<Option<Box<dyn FnOnce()>>>>;
#[derive(Clone)]
struct Controlled {
    io: Shared,
    pass: Rc<Cell<u64>>,
    fail_copy: Rc<Cell<bool>>,
    alter_copy: Rc<Cell<bool>>,
    panic_write: Rc<Cell<bool>>,
    on_write: Action,
    on_flush: Action,
}
impl Controlled {
    fn new() -> Self {
        Self {
            io: Shared::new(Vec::new()),
            pass: Rc::new(Cell::new(0)),
            fail_copy: Rc::new(Cell::new(false)),
            alter_copy: Rc::new(Cell::new(false)),
            panic_write: Rc::new(Cell::new(false)),
            on_write: Rc::new(RefCell::new(None)),
            on_flush: Rc::new(RefCell::new(None)),
        }
    }
}
impl Read for Controlled {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        if self.fail_copy.get() && self.pass.get() == 2 && self.io.inner.borrow().position() >= 8192
        {
            return Err(io::Error::other("injected live-copy read"));
        }
        let n = self.io.read(b)?;
        if self.alter_copy.get() && self.pass.get() == 2 && n > 0 {
            b[0] ^= 1;
        }
        Ok(n)
    }
}
impl Write for Controlled {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        assert!(!self.panic_write.get(), "injected append unwind");
        if let Some(action) = self.on_write.borrow_mut().take() {
            action();
        }
        self.io.write(b)
    }
    fn flush(&mut self) -> io::Result<()> {
        if let Some(action) = self.on_flush.borrow_mut().take() {
            action();
        }
        self.io.flush()
    }
}
impl Seek for Controlled {
    fn seek(&mut self, s: SeekFrom) -> io::Result<u64> {
        if matches!(s, SeekFrom::Start(0)) {
            self.pass.set(self.pass.get() + 1);
        }
        self.io.seek(s)
    }
}

#[test]
fn caps_staging_refusals_and_changed_spools_never_touch_live_history() {
    let _serial = SERIAL.lock().unwrap();
    let p = plan(2);
    let weights: Vec<_> = (0..2)
        .map(|_| tensor(0, PlaneCodec::D2, Basis::Identity, 0, 2))
        .collect();
    let source = Shared::new(Vec::new());
    let mut current =
        Paged::initialize(p.clone(), limits(), parents(&weights), source.clone()).unwrap();
    let live = Shared::new(Vec::new());
    let mut writer = PagedWriter::new(live.clone(), &mut current, cap()).unwrap();
    let original = source.contents();
    let prefix = live.contents();
    let point = writer.checkpoint(&mut current).unwrap();
    let sizes = current.recording_size_ledger().unwrap();
    for mode in 0..3 {
        let bad = match mode {
            0 => RecordingLimits {
                max_journal_bytes: sizes.journal_bytes - 1,
                ..cap()
            },
            1 => RecordingLimits {
                max_frame_bytes: sizes.frame_bytes - 1,
                ..cap()
            },
            _ => RecordingLimits {
                max_working_bytes: 16383,
                ..cap()
            },
        };
        let calls = Cell::new(0);
        let target = Shared::new(Vec::new());
        let frame = Shared::new(Vec::new());
        let result = current.stage_recorded(
            (target.clone(), frame.clone()),
            |i, _| {
                calls.set(calls.get() + 1);
                Ok::<_, &str>(weights[i].clone())
            },
            |i, _, w| Ok(grad(i, w)),
            |_, _, _| Ok(()),
            bad,
        );
        assert!(matches!(result, Err(RecordingError::Journal(_))));
        assert_eq!(calls.get(), 0);
        assert!(target.contents().is_empty());
        assert!(frame.contents().is_empty());
    }
    for mode in 0..6 {
        let drops = Rc::new(Cell::new(0));
        let target = Shared::new(Vec::new());
        let frame = Shared::new(Vec::new());
        if mode == 0 {
            frame.fail_write.set(Some(50));
        }
        if mode == 1 {
            frame.fail_flush.set(true);
        }
        if mode == 2 {
            frame.corrupt.set(true);
        }
        let ready = current.stage_recorded(
            (target.clone(), frame.clone()),
            parents(&weights),
            |i, _, w| Ok(grad(i, w)),
            |_, _, _| Ok(Resource(drops.clone())),
            cap(),
        );
        if mode < 3 {
            assert!(ready.is_err());
        } else {
            let ready = ready.unwrap();
            if mode == 3 {
                frame.inner.borrow_mut().get_mut()[40] ^= 1;
            }
            if mode == 4 {
                target.inner.borrow_mut().get_mut()[90] ^= 1;
            }
            if mode == 5 {
                source.inner.borrow_mut().get_mut()[90] ^= 1;
            }
            assert!(ready.commit(&mut current, &mut writer).is_err());
            source
                .inner
                .borrow_mut()
                .get_mut()
                .copy_from_slice(&original);
        }
        assert_eq!(current.completed_updates(), 0);
        assert_eq!(source.contents(), original);
        assert_eq!(live.contents(), prefix);
        assert_eq!(writer.checkpoint(&mut current).unwrap(), point);
    }
    let ready = current
        .stage_recorded(
            (output(), output()),
            parents(&weights),
            |i, _, w| Ok(grad(i, w)),
            |_, _, _| Ok(()),
            cap(),
        )
        .unwrap();
    let (mut next, _, _) = ready.commit(&mut current, &mut writer).unwrap();
    let mut other = Resident::new(p.clone(), weights.clone()).unwrap();
    let g: Vec<_> = weights
        .iter()
        .map(|w| {
            let (s, mut g) = grad(0, w);
            g.fill(0.9);
            (s, g)
        })
        .collect();
    other
        .step(
            &g.iter().map(|(s, _)| s.as_slice()).collect::<Vec<_>>(),
            &g.iter().map(|(_, w)| w.as_slice()).collect::<Vec<_>>(),
        )
        .unwrap();
    let mut wrong = Paged::reopen(
        p,
        limits(),
        parents(&weights),
        Cursor::new(other.checkpoint_bytes().unwrap()),
    )
    .unwrap();
    assert!(writer.checkpoint(&mut wrong).is_err());
    assert!(writer.checkpoint(&mut next).is_ok());
}

#[test]
fn partial_copy_io_transferred_substitution_and_unwind_poison_without_exposing_next_state() {
    let _serial = SERIAL.lock().unwrap();
    let p = plan(2);
    let weights: Vec<_> = (0..2)
        .map(|_| tensor(0, PlaneCodec::D2, Basis::Identity, 0, 64))
        .collect();
    for mode in 0..7 {
        let source = Shared::new(Vec::new());
        let mut current =
            Paged::initialize(p.clone(), limits(), parents(&weights), source.clone()).unwrap();
        let live = Controlled::new();
        let mut writer = PagedWriter::new(live.clone(), &mut current, cap()).unwrap();
        let before = source.contents();
        let frame = Controlled::new();
        let target = Shared::new(Vec::new());
        let drops = Rc::new(Cell::new(0));
        let ready = current
            .stage_recorded(
                (target.clone(), frame.clone()),
                parents(&weights),
                |i, _, w| Ok(grad(i, w)),
                |_, _, _| Ok(Resource(drops.clone())),
                cap(),
            )
            .unwrap();
        frame.pass.set(0);
        if mode == 0 {
            frame.fail_copy.set(true);
        }
        if mode == 1 {
            frame.alter_copy.set(true);
        }
        if mode == 2 {
            live.io
                .fail_write
                .set(Some(live.io.contents().len() as u64 + 8192));
        }
        if mode == 3 {
            live.io.fail_flush.set(true);
        }
        if mode == 4 {
            live.panic_write.set(true);
        }
        if mode == 5 {
            let changed = source.clone();
            *live.on_write.borrow_mut() = Some(Box::new(move || {
                changed.inner.borrow_mut().get_mut()[90] ^= 1;
            }));
        }
        if mode == 6 {
            let changed = target.clone();
            *live.on_write.borrow_mut() = Some(Box::new(move || {
                changed.inner.borrow_mut().get_mut()[90] ^= 1;
            }));
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ready.commit(&mut current, &mut writer)
        }));
        if mode == 4 {
            assert!(result.is_err());
        } else {
            assert!(result.unwrap().is_err());
        }
        assert_eq!(drops.get(), 2);
        if mode == 5 {
            source.inner.borrow_mut().get_mut().copy_from_slice(&before);
        }
        assert_eq!(source.contents(), before);
        assert_eq!(current.completed_updates(), 0);
        assert!(
            writer
                .checkpoint(&mut current)
                .unwrap_err()
                .to_string()
                .contains("poisoned")
        );
    }
}

#[test]
fn file_recording_snapshot_and_finish_use_no_complete_frame_or_checkpoint_buffer() {
    let _serial = SERIAL.lock().unwrap();
    rayon::join(|| (), || ());
    let files = Files::new();
    let mut p = plan(8);
    p.training.truncate(1);
    let mut current = Paged::initialize(
        p.clone(),
        limits(),
        |_, _| Ok::<_, &str>(big()),
        files.create("old.tnpv"),
    )
    .unwrap();
    let (result, peak, retained) =
        measured(|| PagedWriter::new(files.create("journal.tnpj"), &mut current, cap()));
    let mut writer = result.unwrap();
    assert!(peak < 65536);
    assert!(retained < 65536);
    let (result, peak, retained) = measured(|| {
        current.stage_recorded(
            (files.create("next.tnpv"), files.create("frame.tmp")),
            |_, _| Ok::<_, &str>(big()),
            |i, _, w| Ok(grad(i, w)),
            |_, _, _| Ok(()),
            cap(),
        )
    });
    let ready = result.unwrap();
    assert!(retained < 65536);
    assert!((peak as u64) < current.size_ledger().max_owner_payload_bytes + 65536);
    let (result, commit_peak, commit_retained) =
        measured(|| ready.commit(&mut current, &mut writer));
    let (mut next, _, _) = result.unwrap();
    assert!(
        commit_peak < 65536,
        "commit buffered frame/checkpoint: {commit_peak}"
    );
    assert!(commit_retained < 65536);
    let (result, snapshot_peak, _) = measured(|| writer.checkpoint(&mut next));
    let point = result.unwrap();
    assert!(snapshot_peak < 65536);
    assert_eq!(
        point.checkpoint_bytes,
        next.size_ledger().logical.checkpoint_bytes
    );
    let (result, finish_peak, _) = measured(|| writer.finish(&mut next));
    let (_, summary) = result.unwrap();
    assert_eq!(summary.checkpoint(), point.checkpoint);
    assert!(
        finish_peak < 65536,
        "finish buffered checkpoint: {finish_peak}"
    );
    eprintln!(
        "PAGED_RECORDING owners=8 frame_bytes={} checkpoint_bytes={} stage_peak_live_requests={peak} stage_retained_requests={retained} commit_peak_live_requests={commit_peak} snapshot_peak_live_requests={snapshot_peak}",
        next.recording_size_ledger().unwrap().frame_bytes,
        point.checkpoint_bytes
    );
    eprintln!("PAGED_RECORDING_FINISH peak_live_requests={finish_peak}");
}

#[test]
fn initialization_snapshot_and_finish_refuse_checkpoint_mutation_during_output_effects() {
    let _serial = SERIAL.lock().unwrap();
    let mut p = plan(1);
    p.training.truncate(1);
    let weights = vec![tensor(0, PlaneCodec::D2, Basis::Identity, 0, 2)];
    for phase in 0..3 {
        let source = Shared::new(Vec::new());
        let mut current =
            Paged::initialize(p.clone(), limits(), parents(&weights), source.clone()).unwrap();
        let live = Controlled::new();
        if phase == 0 {
            let changed = source.clone();
            *live.on_write.borrow_mut() = Some(Box::new(move || {
                changed.inner.borrow_mut().get_mut()[90] ^= 1;
            }));
            assert!(PagedWriter::new(live, &mut current, cap()).is_err());
            continue;
        }
        let mut writer = PagedWriter::new(live.clone(), &mut current, cap()).unwrap();
        let target = Shared::new(Vec::new());
        let ready = current
            .stage_recorded(
                (target.clone(), output()),
                parents(&weights),
                |i, _, w| Ok(grad(i, w)),
                |_, _, _| Ok(()),
                cap(),
            )
            .unwrap();
        let (mut next, _, _) = ready.commit(&mut current, &mut writer).unwrap();
        let unused = Shared::new(Vec::new());
        assert!(PagedWriter::new(unused.clone(), &mut next, cap()).is_err());
        assert!(unused.contents().is_empty());
        let changed = target.clone();
        let action: Box<dyn FnOnce()> = Box::new(move || {
            changed.inner.borrow_mut().get_mut()[90] ^= 1;
        });
        if phase == 1 {
            *live.on_flush.borrow_mut() = Some(action);
            assert!(writer.checkpoint(&mut next).is_err());
        } else {
            *live.on_write.borrow_mut() = Some(action);
            assert!(writer.finish(&mut next).is_err());
        }
    }
}
