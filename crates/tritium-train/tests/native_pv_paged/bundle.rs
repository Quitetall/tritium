//! Standard bundle parity and bounded seek-backed save/recovery, not qualification.
use super::*;
use tritium_train::pv_tuning::{
    NativePvPagedResumeBundleLimits as Caps, NativePvPagedResumeBundleReader as Bundle,
    NativePvResumeBundleLimits as ResidentCaps, publish_native_pv_paged_resume_bundle as publish,
    stage_native_pv_paged_resume_bundle as stage, stage_native_pv_resume_bundle as resident_stage,
};
fn caps() -> Caps {
    Caps {
        max_bundle_bytes: 1 << 30,
        max_manifest_bytes: 100_000,
        max_checkpoint_bytes: 1 << 29,
        max_working_bytes: 108192,
    }
}
fn replay_caps() -> NativePvPagedReplayLimits {
    NativePvPagedReplayLimits {
        max_journal_bytes: 1 << 30,
        max_working_bytes: 16384,
        paging: limits(),
    }
}
fn recording_caps() -> RecordingLimits {
    RecordingLimits {
        max_journal_bytes: 1 << 30,
        max_frame_bytes: 1 << 29,
        max_working_bytes: 16384,
    }
}
fn advance<R: Read + Seek, W: Read + Write + Seek>(
    current: &mut Paged<R>,
    writer: &mut PagedWriter<W>,
    weights: &[AdditiveTensor],
) -> Paged<Cursor<Vec<u8>>> {
    let step = current.completed_updates() as usize;
    let ready = current
        .stage_recorded(
            (output(), output()),
            parents(weights),
            |i, _, w| Ok(grad(i + step, w)),
            |_, _, _| Ok(()),
            recording_caps(),
        )
        .unwrap();
    ready.commit(current, writer).unwrap().0
}
struct Fixture {
    session: Paged<Shared>,
    journal: PagedWriter<Shared>,
    cp: Shared,
    prefix: Shared,
    resident: Resident,
    resident_journal: NativePvJournalWriter<Cursor<Vec<u8>>>,
}
fn fixture(p: &NativeRefinementPlan, weights: &[AdditiveTensor], cursor: usize) -> Fixture {
    let resident = Resident::new(p.clone(), weights.to_vec()).unwrap();
    let resident_journal = NativePvJournalWriter::new(
        output(),
        &resident,
        NativePvJournalLimits {
            max_journal_bytes: 1 << 30,
            max_working_bytes: 1 << 30,
        },
    )
    .unwrap();
    let cp = Shared::new(Vec::new());
    let mut session = Paged::initialize(p.clone(), limits(), parents(weights), cp.clone()).unwrap();
    let prefix = Shared::new(Vec::new());
    let journal = PagedWriter::new(prefix.clone(), &mut session, recording_caps()).unwrap();
    let mut f = Fixture {
        session,
        journal,
        cp,
        prefix,
        resident,
        resident_journal,
    };
    for step in 0..cursor {
        let g: Vec<_> = f
            .resident
            .weights()
            .enumerate()
            .map(|(i, w)| grad(i + step, w))
            .collect();
        f.resident
            .step_recorded(
                &g.iter().map(|x| x.0.as_slice()).collect::<Vec<_>>(),
                &g.iter().map(|x| x.1.as_slice()).collect::<Vec<_>>(),
                &mut f.resident_journal,
            )
            .unwrap();
        let next = Shared::new(Vec::new());
        let ready = f
            .session
            .stage_recorded(
                (next.clone(), output()),
                parents(weights),
                |i, _, w| Ok(grad(i + step, w)),
                |_, _, _| Ok(()),
                recording_caps(),
            )
            .unwrap();
        f.session = ready.commit(&mut f.session, &mut f.journal).unwrap().0;
        f.cp = next;
    }
    f
}
fn encoded(f: &mut Fixture) -> (Vec<u8>, BlobId) {
    let out = Shared::new(Vec::new());
    let b = stage(&mut f.session, &mut f.journal, out.clone(), caps()).unwrap();
    (out.contents(), b.id())
}

#[test]
fn every_cursor_all_native_profiles_has_exact_resident_bundle_and_continued_journal_bytes() {
    let _serial = SERIAL.lock().unwrap();
    let mut profiles = 0;
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
                for cursor in 0..=4 {
                    let mut f = fixture(&p, &weights, cursor);
                    let before = f.cp.contents();
                    let (actual, id) = encoded(&mut f);
                    let expected = Shared::new(Vec::new());
                    let b = resident_stage(
                        &f.resident,
                        &mut f.resident_journal,
                        expected.clone(),
                        ResidentCaps {
                            max_bundle_bytes: 1 << 30,
                            max_manifest_bytes: 100_000,
                            max_checkpoint_bytes: 1 << 29,
                            max_working_bytes: 1 << 30,
                        },
                    )
                    .unwrap();
                    assert_eq!(actual, expected.contents());
                    assert_eq!(id, b.id());
                    assert_eq!(f.cp.contents(), before);
                    let source = Shared::new(actual);
                    let b = Bundle::open(source.clone(), id, caps()).unwrap();
                    assert_eq!(b.manifest().point.completed_updates, cursor as u64);
                    let (mut s, mut j) = b
                        .resume(output(), parents(&weights), |_| Ok(output()), replay_caps())
                        .unwrap();
                    for step in cursor..4 {
                        let g: Vec<_> = f
                            .resident
                            .weights()
                            .enumerate()
                            .map(|(i, w)| grad(i + step, w))
                            .collect();
                        f.resident
                            .step_recorded(
                                &g.iter().map(|x| x.0.as_slice()).collect::<Vec<_>>(),
                                &g.iter().map(|x| x.1.as_slice()).collect::<Vec<_>>(),
                                &mut f.resident_journal,
                            )
                            .unwrap();
                        s = advance(&mut s, &mut j, &weights);
                    }
                    let (actual, summary) = j.finish(&mut s).unwrap();
                    let (expected, expected_summary) =
                        f.resident_journal.finish(&f.resident).unwrap();
                    assert_eq!(actual.into_inner(), expected.into_inner());
                    assert_eq!(summary, expected_summary);
                    assert_eq!(
                        s.into_checkpoint().into_inner(),
                        f.resident.checkpoint_bytes().unwrap()
                    );
                }
                profiles += 1;
            }
        }
    }
    assert_eq!(profiles, 36);
}

#[test]
fn caps_corruption_resealed_moments_and_whole_source_drift_never_gain_numerical_recovery() {
    let _serial = SERIAL.lock().unwrap();
    let p = plan(2);
    let weights: Vec<_> = (0..2)
        .map(|_| tensor(0, PlaneCodec::B3, Basis::Identity, 0, 2))
        .collect();
    let mut f = fixture(&p, &weights, 2);
    let (bytes, id) = encoded(&mut f);
    for mode in 0..6 {
        let mut input = bytes.clone();
        match mode {
            0 => input[0] ^= 1,
            1 => input.truncate(input.len() - 1),
            2 => input.push(0),
            3 => input[16..24].copy_from_slice(&u64::MAX.to_le_bytes()),
            4 => input[40] ^= 1,
            _ => {
                let at = input.len() - 40;
                input[at] ^= 1;
            }
        }
        assert!(Bundle::open(Cursor::new(input), id, caps()).is_err());
    }
    for bad in [
        Caps {
            max_bundle_bytes: bytes.len() as u64 - 1,
            ..caps()
        },
        Caps {
            max_working_bytes: 8191,
            ..caps()
        },
        Caps {
            max_checkpoint_bytes: 1,
            ..caps()
        },
        Caps {
            max_manifest_bytes: 1,
            ..caps()
        },
    ] {
        assert!(Bundle::open(Cursor::new(&bytes), id, bad).is_err());
        let out = Shared::new(Vec::new());
        assert!(stage(&mut f.session, &mut f.journal, out.clone(), bad).is_err());
        assert!(out.contents().is_empty());
        assert!(f.journal.checkpoint(&mut f.session).is_ok());
    }
    let m = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let exact = Caps {
        max_working_bytes: m as u64 + 8192,
        ..caps()
    };
    Bundle::open(Cursor::new(&bytes), id, exact).unwrap();
    assert!(
        Bundle::open(
            Cursor::new(&bytes),
            id,
            Caps {
                max_working_bytes: exact.max_working_bytes - 1,
                ..exact
            }
        )
        .is_err()
    );
    let c = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
    let mut checkpoint = bytes[32 + m..32 + m + c].to_vec();
    let n = checkpoint.len();
    checkpoint[n - 36..n - 32].copy_from_slice(&0.001f32.to_le_bytes());
    let hash = blake3::hash(&checkpoint[..n - 32]);
    checkpoint[n - 32..].copy_from_slice(hash.as_bytes());
    let mut manifest: NativePvResumeManifest = serde_json::from_slice(&bytes[32..32 + m]).unwrap();
    manifest.point.checkpoint = BlobId::from_bytes(*blake3::hash(&checkpoint).as_bytes());
    let metadata = serde_json::to_vec(&manifest).unwrap();
    let prefix = &bytes[32 + m + c..bytes.len() - 32];
    let mut forged = b"TNPB\0\0\0\x01".to_vec();
    for n in [metadata.len(), checkpoint.len(), prefix.len()] {
        forged.extend_from_slice(&(n as u64).to_le_bytes());
    }
    forged.extend(metadata);
    forged.extend(checkpoint);
    forged.extend(prefix);
    let hash = blake3::hash(&forged);
    forged.extend_from_slice(hash.as_bytes());
    let id2 = BlobId::from_bytes(*blake3::hash(&forged).as_bytes());
    let b = Bundle::open(Cursor::new(forged), id2, caps()).unwrap();
    assert!(
        b.resume(output(), parents(&weights), |_| Ok(output()), replay_caps())
            .is_err()
    );
    let source = Shared::new(bytes.clone());
    let b = Bundle::open(source.clone(), id, caps()).unwrap();
    let result = b.resume(
        output(),
        parents(&weights),
        |i| {
            if i == 1 {
                source.inner.borrow_mut().get_mut()[32] ^= 1;
            }
            Ok(output())
        },
        replay_caps(),
    );
    assert!(
        result.is_err(),
        "whole bundle must be checked after region replay"
    );
    source.inner.borrow_mut().get_mut()[32] ^= 1;
    assert_eq!(source.contents(), bytes);
    let b = Bundle::open(source.clone(), id, caps()).unwrap();
    assert!(
        b.resume(
            LateFlush {
                io: Shared::new(Vec::new()),
                source: source.clone(),
                flushes: 0
            },
            parents(&weights),
            |_| Ok(output()),
            replay_caps()
        )
        .is_err()
    );
    source.inner.borrow_mut().get_mut()[32] ^= 1;
    assert_eq!(source.contents(), bytes);
}

struct LateFlush {
    io: Shared,
    source: Shared,
    flushes: usize,
}
impl Seek for LateFlush {
    fn seek(&mut self, s: SeekFrom) -> io::Result<u64> {
        self.io.seek(s)
    }
}
impl Write for LateFlush {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        self.io.write(b)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.flushes += 1;
        if self.flushes == 2 {
            self.source.inner.borrow_mut().get_mut()[32] ^= 1;
        }
        self.io.flush()
    }
}

struct Effect {
    out: Shared,
    cp: Shared,
    prefix: Shared,
    at: u64,
    mode: usize,
    fired: bool,
}
impl Read for Effect {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        self.out.read(b)
    }
}
impl Seek for Effect {
    fn seek(&mut self, s: SeekFrom) -> io::Result<u64> {
        self.out.seek(s)
    }
}
impl Write for Effect {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        let n = self.out.write(b)?;
        if !self.fired && self.out.inner.borrow().position() >= self.at {
            self.fired = true;
            match self.mode {
                7 => self.cp.inner.borrow_mut().get_mut()[100] ^= 1,
                8 => self.prefix.inner.borrow_mut().get_mut()[0] ^= 1,
                9 | 10 => panic!("injected bundle output unwind"),
                _ => {}
            }
        }
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.out.flush()
    }
}

#[test]
fn destination_refusal_restores_retry_but_source_uncertainty_and_unwind_poison() {
    let _serial = SERIAL.lock().unwrap();
    let p = plan(2);
    let weights: Vec<_> = (0..2)
        .map(|_| tensor(0, PlaneCodec::D2, Basis::Identity, 0, 2))
        .collect();
    for mode in 1..=11 {
        let mut f = fixture(&p, &weights, 2);
        let (bytes, _) = encoded(&mut f);
        let m = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let c = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let cp_start = 32 + m;
        let prefix_start = cp_start + c;
        let out = Shared::new(Vec::new());
        if mode <= 3 {
            out.fail_write.set(Some(match mode {
                1 => 1,
                2 => cp_start + 1,
                _ => prefix_start + 1,
            }));
        }
        out.fail_flush.set(mode == 4);
        f.cp.fail_read.set(mode == 5);
        f.prefix.fail_read.set(mode == 6);
        f.cp.fail_seek.set(mode == 11);
        let destination = Effect {
            out: out.clone(),
            cp: f.cp.clone(),
            prefix: f.prefix.clone(),
            at: if mode == 9 {
                cp_start + 1
            } else {
                prefix_start + 1
            },
            mode,
            fired: false,
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            stage(&mut f.session, &mut f.journal, destination, caps())
        }));
        if mode == 9 || mode == 10 {
            assert!(result.is_err());
        } else {
            assert!(result.unwrap().is_err(), "mode{mode}");
        }
        f.cp.fail_read.set(false);
        f.cp.fail_seek.set(false);
        f.prefix.fail_read.set(false);
        if mode == 7 {
            f.cp.inner.borrow_mut().get_mut()[100] ^= 1;
        }
        if mode == 8 {
            f.prefix.inner.borrow_mut().get_mut()[0] ^= 1;
        }
        let point = f.journal.checkpoint(&mut f.session);
        if mode <= 4 {
            point.unwrap();
            stage(&mut f.session, &mut f.journal, output(), caps()).unwrap();
        } else {
            assert!(
                point.is_err(),
                "uncertain source/unwind must poison mode{mode}"
            );
        }
    }
}

#[test]
fn file_bundle_stage_open_publish_and_recovery_do_not_buffer_checkpoint_or_prefix() {
    let _serial = SERIAL.lock().unwrap();
    rayon::join(|| (), || ());
    let files = Files::new();
    let mut p = plan(8);
    p.training.truncate(1);
    let mut initial = Paged::initialize(
        p.clone(),
        limits(),
        |_, _| Ok::<_, &str>(big()),
        files.create("old.tnpv"),
    )
    .unwrap();
    let mut journal =
        PagedWriter::new(files.create("live.tnpj"), &mut initial, recording_caps()).unwrap();
    let ready = initial
        .stage_recorded(
            (files.create("next.tnpv"), files.create("frame.tmp")),
            |_, _| Ok::<_, &str>(big()),
            |i, _, w| Ok(grad(i, w)),
            |_, _, _| Ok(()),
            recording_caps(),
        )
        .unwrap();
    let (mut session, _, _) = ready.commit(&mut initial, &mut journal).unwrap();
    let cp_bytes = session.size_ledger().logical.checkpoint_bytes;
    let (result, stage_peak, stage_retained) = measured(|| {
        stage(
            &mut session,
            &mut journal,
            files.create("staged.tnpb"),
            caps(),
        )
    });
    let b = result.unwrap();
    let id = b.id();
    let total = b.bytes();
    drop(b);
    assert!(stage_peak < 65536, "stage buffered state {stage_peak}");
    assert!(stage_retained < 65536);
    let (result, open_peak, open_retained) =
        measured(|| Bundle::open(File::open(files.0.join("staged.tnpb")).unwrap(), id, caps()));
    let b = result.unwrap();
    assert_eq!(b.bytes(), total);
    drop(b);
    assert!(open_peak < 65536, "open buffered checkpoint {open_peak}");
    assert!(open_retained < 65536);
    let directory = files.0.join("published");
    let published = publish(&mut session, &mut journal, &directory, caps()).unwrap();
    assert_eq!(published.bundle.id(), id);
    assert!(published.warnings.is_empty());
    assert!(publish(&mut session, &mut journal, &directory, caps()).is_err());
    drop(initial);
    drop(session);
    drop(journal);
    for name in [
        "old.tnpv",
        "next.tnpv",
        "live.tnpj",
        "frame.tmp",
        "staged.tnpb",
    ] {
        std::fs::remove_file(files.0.join(name)).unwrap();
    }
    drop(published.bundle);
    let (result, resume_peak, resume_retained) = measured(|| {
        Bundle::open(
            File::open(directory.join("resume.tnpb")).unwrap(),
            id,
            caps(),
        )
        .unwrap()
        .resume(
            files.create("recovered.tnpj"),
            |_, _| Ok::<_, &str>(big()),
            |i| Ok(files.create(&format!("recover{i}.tnpv"))),
            replay_caps(),
        )
    });
    let (mut s, j) = result.unwrap();
    assert!(resume_peak as u64 <= s.size_ledger().max_owner_payload_bytes + 2 * 8192 + 65536);
    assert!((resume_peak as u64) < cp_bytes);
    assert!(resume_retained < 65536);
    let (_, summary) = j.finish(&mut s).unwrap();
    assert_eq!(summary.updates(), 1);
    eprintln!(
        "PAGED_BUNDLE checkpoint_bytes={cp_bytes} bundle_bytes={total} stage_peak={stage_peak} stage_retained={stage_retained} open_peak={open_peak} open_retained={open_retained} resume_peak={resume_peak} resume_retained={resume_retained}"
    );
}
