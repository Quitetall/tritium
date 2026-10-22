//! Actual common optimizer replay fixtures, not model/objective/release approval.
use std::{
    cell::{Cell, RefCell},
    io::{self, Cursor, Read, Seek, SeekFrom, Write},
    rc::Rc,
};
use tritium_core::{PerTilePlan, Trit, UniformPlan};
use tritium_format::AdditiveTensor;
use tritium_schema::*;
use tritium_train::pv_tuning::{
    NativeScaleJournalError as Error, NativeScaleJournalLimits as Limits,
    NativeScaleJournalSummary as Summary, NativeScaleJournalWriter as Writer,
    NativeScaleRefinementSession as Session, NativeScaleStageError,
    replay_native_scale_journal as replay,
};

fn plan(owners: usize) -> NativeRefinementPlan {
    NativeRefinementPlan {
        parent: NativeRefinementParent {
            package: PackageId::from_bytes([1; 32]),
            level: "parent".into(),
            model: ModelId::from_bytes([2; 32]),
            source: OriginalSourceId::from_bytes([3; 32]),
            evidence: EvidenceReference {
                run: "fixture".into(),
                span: "parent".into(),
                root: [4; 32],
            },
        },
        child_level: "child".into(),
        owners: (0..owners).map(|i| format!("owner-{i:04}")).collect(),
        recipe: NativeRecoveryRecipe::ScaleOnly {
            optimizer: NativeAdamWRecipe {
                lr: 0.01,
                beta1: 0.9,
                beta2: 0.99,
                eps: 1e-8,
                weight_decay: 0.01,
            },
        },
        objective: [5; 32],
        teacher_source: None,
        training: (0..4)
            .map(|i| NativeRefinementWindow {
                content: [10 + i % 3; 32],
                weight: if i == 0 { 2.0 } else { 1.0 },
            })
            .collect(),
        validation: vec![NativeRefinementWindow {
            content: [20; 32],
            weight: 1.0,
        }],
        limits: NativeRefinementLimits {
            max_state_bytes: 1_000_000,
            max_workspace_bytes: 1_000_000,
            max_checkpoint_bytes: 1_000_000,
            max_package_bytes: 1_000_000,
        },
    }
}
fn tensor(
    law: usize,
    codec: PlaneCodec,
    basis: Basis,
    mode: u8,
    transport: Transport,
    group: u16,
) -> AdditiveTensor {
    let admitted = ADMITTED_LAWS[law];
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
        transport,
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
    let trits = (0..n)
        .map(|i| [Trit::ZERO, Trit::NEG, Trit::POS, Trit::NEG][i % 4])
        .collect();
    let scale = if layout.law.precision == ScalePrecision::F32 {
        f32::from_bits(0x3f00_0001)
    } else {
        0.5
    };
    if mode == 0 {
        AdditiveTensor::new(layout, admitted.max_planes, trits, vec![scale; s]).unwrap()
    } else {
        AdditiveTensor::new_per_tile(layout, counts, trits, vec![scale; s]).unwrap()
    }
}
fn simple() -> AdditiveTensor {
    tensor(1, PlaneCodec::D2, Basis::Identity, 0, Transport::Raw, 32)
}
fn limits() -> Limits {
    Limits {
        max_journal_bytes: 10_000_000,
        max_working_bytes: 10_000_000,
    }
}
fn digest(bytes: &[u8]) -> BlobId {
    BlobId::from_bytes(*blake3::hash(bytes).as_bytes())
}
fn gradients(session: &Session, step: usize) -> Vec<Vec<f32>> {
    session
        .weights()
        .enumerate()
        .map(|(o, w)| {
            (0..w.scales().len())
                .map(|i| ((i + o + step) % 7) as f32 * 0.125 - 0.375)
                .collect()
        })
        .collect()
}
fn run(parents: Vec<AdditiveTensor>) -> (NativeRefinementPlan, Session, Vec<u8>, Summary) {
    let request = plan(parents.len());
    let mut session = Session::new(request.clone(), parents).unwrap();
    let mut writer = Writer::new(Cursor::new(Vec::new()), &session, limits()).unwrap();
    for step in 0..request.training.len() {
        let g = gradients(&session, step);
        session
            .step_recorded(
                &g.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                &mut writer,
            )
            .unwrap();
    }
    let (bytes, summary) = writer.finish(&session).unwrap();
    (request, session, bytes.into_inner(), summary)
}
fn reseal(bytes: &mut [u8]) {
    let n = bytes.len() - 32;
    let hash = blake3::hash(&bytes[..n]);
    bytes[n..].copy_from_slice(hash.as_bytes());
}
fn reseal_frame(bytes: &mut [u8], start: usize, frame_bytes: usize) {
    let mut hash = blake3::Hasher::new();
    hash.update(b"tritium.native-scale.journal-frame.v1\0");
    hash.update(&bytes[start..start + frame_bytes - 32]);
    bytes[start + frame_bytes - 32..start + frame_bytes]
        .copy_from_slice(hash.finalize().as_bytes());
}

#[test]
fn recorded_steps_and_streamed_replay_match_648_reference_profiles() {
    let mut cases = 0;
    for law in 0..ADMITTED_LAWS.len() {
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
                    for transport in [
                        Transport::Raw,
                        Transport::JointHuffman { block: 128 },
                        Transport::JointHuffman { block: 256 },
                    ] {
                        for group in [32, 128] {
                            let parent = tensor(law, codec, basis, mode, transport, group);
                            let (request, session, bytes, summary) = run(vec![parent.clone()]);
                            let checkpoint = session.checkpoint_bytes().unwrap();
                            let scales = parent.scales().len() as u64;
                            assert_eq!(bytes.len() as u64, 176 + 8 + 4 * (72 + 4 * scales));
                            assert_eq!(summary.bytes(), bytes.len() as u64);
                            assert_eq!(summary.journal(), digest(&bytes));
                            assert_eq!(summary.checkpoint(), digest(&checkpoint));
                            assert_eq!(summary.checkpoint_bytes(), checkpoint.len() as u64);
                            let ledger = session.journal_size_ledger().unwrap();
                            assert_eq!(ledger.journal_bytes, summary.bytes());
                            let mut direct =
                                Session::new(request.clone(), vec![parent.clone()]).unwrap();
                            for step in 0..4 {
                                let g = gradients(&direct, step);
                                direct
                                    .step(&g.iter().map(Vec::as_slice).collect::<Vec<_>>())
                                    .unwrap();
                            }
                            assert_eq!(direct.checkpoint_bytes().unwrap(), checkpoint);
                            let verified = replay(
                                request,
                                vec![parent.clone()],
                                Cursor::new(&bytes),
                                summary.journal(),
                                &checkpoint,
                                limits(),
                            )
                            .unwrap();
                            verified.verify_session(&session).unwrap();
                            assert_eq!(verified.summary(), &summary);
                            assert_eq!(session.weights().next().unwrap().trits(), parent.trits());
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases, 648);
    println!("NATIVE_SCALE_JOURNAL_PROFILES={cases}");
}

#[test]
fn multiowner_order_and_current_state_are_bound_not_just_cursor() {
    let a = simple();
    let mut b = a.clone();
    b.replace_scales(vec![0.75; b.scales().len()]).unwrap();
    let parents = vec![a, b];
    let (request, session, bytes, summary) = run(parents.clone());
    let checkpoint = session.checkpoint_bytes().unwrap();
    let verified = replay(
        request.clone(),
        parents.clone(),
        Cursor::new(&bytes),
        summary.journal(),
        &checkpoint,
        limits(),
    )
    .unwrap();
    let mut swapped = parents.clone();
    swapped.swap(0, 1);
    assert!(
        replay(
            request.clone(),
            swapped,
            Cursor::new(&bytes),
            summary.journal(),
            &checkpoint,
            limits()
        )
        .is_err()
    );
    let mut changed = request.clone();
    changed.training.swap(0, 1);
    assert!(
        replay(
            changed,
            parents.clone(),
            Cursor::new(&bytes),
            summary.journal(),
            &checkpoint,
            limits()
        )
        .is_err()
    );
    let mut different = Session::new(request, parents).unwrap();
    for _ in 0..4 {
        let g: Vec<_> = different
            .weights()
            .map(|w| vec![0.0; w.scales().len()])
            .collect();
        different
            .step(&g.iter().map(Vec::as_slice).collect::<Vec<_>>())
            .unwrap();
    }
    assert_eq!(different.completed_updates(), session.completed_updates());
    assert!(verified.verify_session(&different).is_err());
    verified.verify_session(&session).unwrap();
}

#[test]
fn every_byte_change_truncation_and_trailing_data_fail_original_identity() {
    let parent = simple();
    let (request, session, bytes, summary) = run(vec![parent.clone()]);
    let checkpoint = session.checkpoint_bytes().unwrap();
    for i in 0..bytes.len() {
        let mut changed = bytes.clone();
        changed[i] ^= 1;
        assert!(
            replay(
                request.clone(),
                vec![parent.clone()],
                Cursor::new(changed),
                summary.journal(),
                &checkpoint,
                limits()
            )
            .is_err(),
            "byte{i}"
        );
    }
    for n in [0, 1, bytes.len() / 2, bytes.len() - 1] {
        assert!(
            replay(
                request.clone(),
                vec![parent.clone()],
                Cursor::new(&bytes[..n]),
                summary.journal(),
                &checkpoint,
                limits()
            )
            .is_err()
        );
    }
    let mut longer = bytes.clone();
    longer.push(0);
    assert!(
        replay(
            request,
            vec![parent],
            Cursor::new(longer),
            summary.journal(),
            &checkpoint,
            limits()
        )
        .is_err()
    );
}

#[test]
fn resealed_malformed_header_frames_and_trailer_do_not_gain_replay() {
    let parent = simple();
    let scales = parent.scales().len();
    let (request, session, bytes, _) = run(vec![parent.clone()]);
    let checkpoint = session.checkpoint_bytes().unwrap();
    let header = 96;
    let frame = 72 + 4 * scales;
    let trailer = bytes.len() - 88;
    for offset in [
        0,
        8,
        40,
        72,
        80,
        88,
        header,
        header + 8,
        header + frame - 1,
        trailer,
        trailer + 8,
        trailer + 16,
        trailer + 24,
        bytes.len() - 1,
    ] {
        let mut changed = bytes.clone();
        changed[offset] ^= 1;
        if offset != bytes.len() - 1 {
            reseal(&mut changed);
        }
        let id = digest(&changed);
        assert!(
            replay(
                request.clone(),
                vec![parent.clone()],
                Cursor::new(changed),
                id,
                &checkpoint,
                limits()
            )
            .is_err(),
            "offset{offset}"
        );
    }
    for value in [f32::NAN, f32::INFINITY, 0.75] {
        let mut changed = bytes.clone();
        changed[header + 40..header + 44].copy_from_slice(&value.to_le_bytes());
        reseal_frame(&mut changed, header, frame);
        reseal(&mut changed);
        let id = digest(&changed);
        assert!(
            replay(
                request.clone(),
                vec![parent.clone()],
                Cursor::new(changed),
                id,
                &checkpoint,
                limits()
            )
            .is_err()
        );
    }
    let mut swapped = bytes.clone();
    let first = swapped[header..header + frame].to_vec();
    swapped.copy_within(header + frame..header + 2 * frame, header);
    swapped[header + frame..header + 2 * frame].copy_from_slice(&first);
    reseal(&mut swapped);
    let id = digest(&swapped);
    assert!(
        replay(
            request,
            vec![parent],
            Cursor::new(swapped),
            id,
            &checkpoint,
            limits()
        )
        .is_err()
    );
}

#[test]
fn valid_resealed_checkpoint_with_forged_moments_is_not_numerical_replay() {
    let parent = simple();
    let (request, session, mut bytes, _) = run(vec![parent.clone()]);
    let mut checkpoint = session.checkpoint_bytes().unwrap();
    let n = checkpoint.len();
    checkpoint[n - 36..n - 32].copy_from_slice(&0.125f32.to_le_bytes());
    reseal(&mut checkpoint);
    let forged = Session::resume(request.clone(), vec![parent.clone()], &checkpoint).unwrap();
    assert_eq!(forged.completed_updates(), 4);
    let n = bytes.len();
    bytes[n - 64..n - 32].copy_from_slice(digest(&checkpoint).as_bytes());
    reseal(&mut bytes);
    let id = digest(&bytes);
    let error = replay(
        request,
        vec![parent],
        Cursor::new(bytes),
        id,
        &checkpoint,
        limits(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("exact checkpoint"));
}

#[derive(Debug)]
struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        panic!("no read before admission")
    }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        panic!("no write before admission")
    }
    fn flush(&mut self) -> io::Result<()> {
        panic!("no flush")
    }
}
impl Seek for NoIo {
    fn seek(&mut self, _: SeekFrom) -> io::Result<u64> {
        panic!("no seek before admission")
    }
}

#[test]
fn exact_logical_caps_refuse_before_io_and_accept_at_the_boundary() {
    let parent = simple();
    let (request, session, bytes, summary) = run(vec![parent.clone()]);
    let checkpoint = session.checkpoint_bytes().unwrap();
    let initial = Session::new(request.clone(), vec![parent.clone()]).unwrap();
    let ledger = initial.size_ledger();
    let writer_work = ledger.checkpoint_bytes + 8192;
    assert_eq!(
        initial.journal_size_ledger().unwrap().writer_working_bytes,
        writer_work
    );
    for cap in [
        Limits {
            max_journal_bytes: summary.bytes() - 1,
            max_working_bytes: u64::MAX,
        },
        Limits {
            max_journal_bytes: u64::MAX,
            max_working_bytes: writer_work - 1,
        },
    ] {
        assert!(matches!(
            Writer::new(NoIo, &initial, cap),
            Err(Error::Limit(_))
        ));
    }
    let mut writer = Writer::new(
        Cursor::new(Vec::new()),
        &initial,
        Limits {
            max_journal_bytes: summary.bytes(),
            max_working_bytes: writer_work,
        },
    )
    .unwrap();
    let mut exact = initial;
    for i in 0..4 {
        let g = gradients(&exact, i);
        exact
            .step_recorded(
                &g.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                &mut writer,
            )
            .unwrap();
    }
    assert_eq!(writer.finish(&exact).unwrap().0.into_inner(), bytes);
    let replay_work = ledger.state_payload_bytes
        + ledger.update_workspace_bytes
        + 4 * parent.scales().len() as u64
        + ledger.checkpoint_bytes
        + 8192;
    assert_eq!(
        session.journal_size_ledger().unwrap().replay_working_bytes,
        replay_work
    );
    for cap in [
        Limits {
            max_journal_bytes: summary.bytes() - 1,
            max_working_bytes: u64::MAX,
        },
        Limits {
            max_journal_bytes: u64::MAX,
            max_working_bytes: replay_work - 1,
        },
    ] {
        assert!(matches!(
            replay(
                request.clone(),
                vec![parent.clone()],
                NoIo,
                summary.journal(),
                &checkpoint,
                cap
            ),
            Err(Error::Limit(_))
        ));
    }
    replay(
        request,
        vec![parent],
        Cursor::new(bytes),
        summary.journal(),
        &checkpoint,
        Limits {
            max_journal_bytes: summary.bytes(),
            max_working_bytes: replay_work,
        },
    )
    .unwrap();
}

#[derive(Clone, Debug)]
struct FaultIo {
    data: Rc<RefCell<Cursor<Vec<u8>>>>,
    remaining: Rc<Cell<usize>>,
    flush: Rc<Cell<bool>>,
    unwind: Rc<Cell<bool>>,
}
impl FaultIo {
    fn new() -> Self {
        Self {
            data: Rc::new(RefCell::new(Cursor::new(Vec::new()))),
            remaining: Rc::new(Cell::new(usize::MAX)),
            flush: Rc::new(Cell::new(false)),
            unwind: Rc::new(Cell::new(false)),
        }
    }
}
impl Write for FaultIo {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.remaining.get() == 0 {
            if self.unwind.get() {
                panic!("injected journal unwind");
            }
            return Err(io::Error::other("injected journal write"));
        }
        let n = bytes.len().min(self.remaining.get());
        let n = self.data.borrow_mut().write(&bytes[..n])?;
        self.remaining.set(self.remaining.get() - n);
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.flush.get() {
            return Err(io::Error::other("injected journal flush"));
        }
        Ok(())
    }
}
impl Seek for FaultIo {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        self.data.borrow_mut().seek(from)
    }
}

#[test]
fn numerical_failure_writes_nothing_and_valid_retry_matches_uninterrupted() {
    let parent = simple();
    let mut session = Session::new(plan(1), vec![parent.clone()]).unwrap();
    let io = FaultIo::new();
    let mut writer = Writer::new(io.clone(), &session, limits()).unwrap();
    let before = session.checkpoint_bytes().unwrap();
    let len = io.data.borrow().get_ref().len();
    let mut g = gradients(&session, 0);
    g[0][0] = f32::NAN;
    assert!(matches!(
        session.step_recorded(
            &g.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            &mut writer
        ),
        Err(NativeScaleStageError::Update(_))
    ));
    assert_eq!(session.checkpoint_bytes().unwrap(), before);
    assert_eq!(io.data.borrow().get_ref().len(), len);
    for i in 0..4 {
        let g = gradients(&session, i);
        session
            .step_recorded(
                &g.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                &mut writer,
            )
            .unwrap();
    }
    let (_, summary) = writer.finish(&session).unwrap();
    let (_, expected, bytes, wanted) = run(vec![parent]);
    assert_eq!(
        session.checkpoint_bytes().unwrap(),
        expected.checkpoint_bytes().unwrap()
    );
    assert_eq!(summary, wanted);
    assert_eq!(*io.data.borrow().get_ref(), bytes);
}

#[test]
fn partial_write_flush_error_and_unwind_preserve_state_and_poison_the_writer() {
    for fault in 0..3 {
        let mut session = Session::new(plan(1), vec![simple()]).unwrap();
        let io = FaultIo::new();
        let mut writer = Writer::new(io.clone(), &session, limits()).unwrap();
        let checkpoint = session.checkpoint_bytes().unwrap();
        if fault == 1 {
            io.flush.set(true);
        } else {
            io.remaining.set(5);
            io.unwind.set(fault == 2);
        }
        let g = gradients(&session, 0);
        let refs: Vec<_> = g.iter().map(Vec::as_slice).collect();
        if fault == 2 {
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                    || session.step_recorded(&refs, &mut writer)
                ))
                .is_err()
            );
        } else {
            assert!(matches!(
                session.step_recorded(&refs, &mut writer),
                Err(NativeScaleStageError::Stage(Error::Io(_)))
            ));
        }
        assert_eq!(session.completed_updates(), 0);
        assert_eq!(session.checkpoint_bytes().unwrap(), checkpoint);
        io.remaining.set(usize::MAX);
        io.flush.set(false);
        io.unwind.set(false);
        assert!(
            session
                .step_recorded(&refs, &mut writer)
                .unwrap_err()
                .to_string()
                .contains("poisoned")
        );
        assert!(
            writer
                .finish(&session)
                .unwrap_err()
                .to_string()
                .contains("poisoned")
        );
    }
}

#[test]
fn nonempty_zero_origin_incomplete_and_unrecorded_cursor_cannot_be_sealed() {
    let mut session = Session::new(plan(1), vec![simple()]).unwrap();
    assert!(
        Writer::new(Cursor::new(vec![9]), &session, limits())
            .unwrap_err()
            .to_string()
            .contains("not empty")
    );
    let writer = Writer::new(Cursor::new(Vec::new()), &session, limits()).unwrap();
    assert!(writer.finish(&session).is_err());
    let mut writer = Writer::new(Cursor::new(Vec::new()), &session, limits()).unwrap();
    let g = gradients(&session, 0);
    let refs: Vec<_> = g.iter().map(Vec::as_slice).collect();
    session.step(&refs).unwrap();
    assert!(session.step_recorded(&refs, &mut writer).is_err());
    assert!(
        Writer::new(NoIo, &session, limits())
            .unwrap_err()
            .to_string()
            .contains("zero-origin")
    );
}

struct BoundedRead {
    source: Cursor<Vec<u8>>,
    starts: usize,
    change: bool,
}
impl Read for BoundedRead {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        assert!(b.len() <= 8192);
        self.source.read(b)
    }
}
impl Seek for BoundedRead {
    fn seek(&mut self, s: SeekFrom) -> io::Result<u64> {
        if s == SeekFrom::Start(0) {
            self.starts += 1;
            if self.change && self.starts == 2 {
                self.source.get_mut()[96 + 40] ^= 1;
            }
        }
        self.source.seek(s)
    }
}
#[test]
fn bounded_two_pass_source_rejects_mutation_between_identity_check_and_replay() {
    let parent = simple();
    let (request, session, bytes, summary) = run(vec![parent.clone()]);
    let checkpoint = session.checkpoint_bytes().unwrap();
    replay(
        request.clone(),
        vec![parent.clone()],
        BoundedRead {
            source: Cursor::new(bytes.clone()),
            starts: 0,
            change: false,
        },
        summary.journal(),
        &checkpoint,
        limits(),
    )
    .unwrap();
    assert!(
        replay(
            request,
            vec![parent],
            BoundedRead {
                source: Cursor::new(bytes),
                starts: 0,
                change: true
            },
            summary.journal(),
            &checkpoint,
            limits()
        )
        .is_err()
    );
}

#[test]
fn real_file_journal_reopens_with_multichunk_owner_gradients() {
    use std::fs::{self, File, OpenOptions};
    let root =
        std::env::temp_dir().join(format!("native-scale-journal-file-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(root.clone());
    let layout = AdditiveLayout {
        rows: 1,
        cols: 131072,
        tile: 256,
        group: 32,
        max_planes: 3,
        allocation: PlaneAllocation::Uniform,
        codec: PlaneCodec::B3,
        law: ADMITTED_LAWS[1].law,
        basis: Basis::SignedRht {
            block: 128,
            seed: 17,
            domain: 9,
        },
        transport: Transport::Raw,
    };
    let parent = AdditiveTensor::new(layout, 1, vec![Trit::POS; 131072], vec![0.5; 4096]).unwrap();
    let request = plan(1);
    let mut session = Session::new(request.clone(), vec![parent.clone()]).unwrap();
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .read(true)
        .open(root.join("journal.tnsj"))
        .unwrap();
    let mut writer = Writer::new(file, &session, limits()).unwrap();
    for i in 0..4 {
        let g = gradients(&session, i);
        session
            .step_recorded(
                &g.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                &mut writer,
            )
            .unwrap();
    }
    let (file, summary) = writer.finish(&session).unwrap();
    file.sync_all().unwrap();
    drop(file);
    assert!(summary.bytes() > 8192);
    let checkpoint = session.checkpoint_bytes().unwrap();
    struct Observed {
        file: File,
        maximum: Rc<Cell<usize>>,
        read_bytes: Rc<Cell<u64>>,
    }
    impl Read for Observed {
        fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
            self.maximum.set(self.maximum.get().max(bytes.len()));
            let n = self.file.read(bytes)?;
            self.read_bytes.set(self.read_bytes.get() + n as u64);
            Ok(n)
        }
    }
    impl Seek for Observed {
        fn seek(&mut self, s: SeekFrom) -> io::Result<u64> {
            self.file.seek(s)
        }
    }
    let maximum = Rc::new(Cell::new(0));
    let read_bytes = Rc::new(Cell::new(0));
    let source = Observed {
        file: File::open(root.join("journal.tnsj")).unwrap(),
        maximum: maximum.clone(),
        read_bytes: read_bytes.clone(),
    };
    let verified = replay(
        request,
        vec![parent],
        source,
        summary.journal(),
        &checkpoint,
        limits(),
    )
    .unwrap();
    verified.verify_session(&session).unwrap();
    assert_eq!(maximum.get(), 8192);
    assert_eq!(read_bytes.get(), summary.bytes() * 2);
    println!(
        "NATIVE_SCALE_FILE_JOURNAL bytes={} maximum_read={}",
        summary.bytes(),
        maximum.get()
    );
}

#[test]
fn wire_bytes_follow_the_declared_little_endian_frames_not_only_roundtrip() {
    let parent = simple();
    let scales = parent.scales().len();
    let (request, session, bytes, summary) = run(vec![parent]);
    let checkpoint = session.checkpoint_bytes().unwrap();
    let mut expected = b"TNSJ\0\0\0\x01".to_vec();
    expected.extend_from_slice(session.plan_id().as_bytes());
    expected.extend_from_slice(&checkpoint[37..69]); // Existing TNSR exact-parent fingerprint.
    expected.extend_from_slice(&4u64.to_le_bytes());
    expected.extend_from_slice(&1u64.to_le_bytes());
    expected.extend_from_slice(&(scales as u64).to_le_bytes());
    for ordinal in 0..4 {
        let mut frame = (ordinal as u64).to_le_bytes().to_vec();
        frame.extend_from_slice(&request.training[ordinal].content);
        for index in 0..scales {
            frame
                .extend_from_slice(&(((index + ordinal) % 7) as f32 * 0.125 - 0.375).to_le_bytes());
        }
        let mut hash = blake3::Hasher::new();
        hash.update(b"tritium.native-scale.journal-frame.v1\0");
        hash.update(&frame);
        expected.extend_from_slice(&frame);
        expected.extend_from_slice(hash.finalize().as_bytes());
    }
    expected.extend_from_slice(b"TNSJEND1");
    expected.extend_from_slice(&4u64.to_le_bytes());
    expected.extend_from_slice(&(checkpoint.len() as u64).to_le_bytes());
    expected.extend_from_slice(digest(&checkpoint).as_bytes());
    let checksum = blake3::hash(&expected);
    expected.extend_from_slice(checksum.as_bytes());
    assert_eq!(bytes, expected);
    assert_eq!(summary.journal(), digest(&expected));
}

#[test]
fn signed_zero_gradient_bits_are_retained_without_inventing_a_state_difference() {
    let parent = simple();
    let request = plan(1);
    let mut results = Vec::new();
    for value in [0.0, -0.0] {
        let mut session = Session::new(request.clone(), vec![parent.clone()]).unwrap();
        let mut writer = Writer::new(Cursor::new(Vec::new()), &session, limits()).unwrap();
        let gradient = vec![value; parent.scales().len()];
        for _ in 0..4 {
            session.step_recorded(&[&gradient], &mut writer).unwrap();
        }
        let (source, summary) = writer.finish(&session).unwrap();
        let checkpoint = session.checkpoint_bytes().unwrap();
        replay(
            request.clone(),
            vec![parent.clone()],
            source,
            summary.journal(),
            &checkpoint,
            limits(),
        )
        .unwrap();
        results.push((summary, checkpoint));
    }
    assert_ne!(results[0].0.journal(), results[1].0.journal());
    assert_eq!(results[0].1, results[1].1);
}
