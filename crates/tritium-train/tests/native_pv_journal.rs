//! Recorded native P/V numerical replay, not objective or release approval.
use std::{
    cell::{Cell, RefCell},
    io::{self, Cursor, Read, Seek, SeekFrom, Write},
    rc::Rc,
};
use tritium_core::{PerTilePlan, Trit, UniformPlan};
use tritium_format::AdditiveTensor;
use tritium_schema::*;
use tritium_train::pv_tuning::{
    NativePvJournalError as Error, NativePvJournalLimits as Limits, NativePvJournalResume,
    NativePvJournalSummary as Summary, NativePvJournalWriter as Writer,
    NativePvRefinementSession as Session, NativePvResumeTail as Tail, NativePvStageError,
    replay_native_pv_journal_with_state as replay, resume_native_pv_journal as resume,
};
#[path = "native_pv_journal/bundle.rs"]
mod bundle;
fn plan(owners: usize) -> NativeRefinementPlan {
    NativeRefinementPlan {
        parent: NativeRefinementParent {
            package: PackageId::from_bytes([1; 32]),
            level: "explicit-parent".into(),
            model: ModelId::from_bytes([2; 32]),
            source: OriginalSourceId::from_bytes([3; 32]),
            evidence: EvidenceReference {
                run: "synthetic-parent".into(),
                span: "import".into(),
                root: [4; 32],
            },
        },
        child_level: "one-child".into(),
        owners: (0..owners).map(|i| format!("owner-{i:04}")).collect(),
        recipe: NativeRecoveryRecipe::AlternatingPv {
            continuous_optimizer: NativeAdamWRecipe {
                lr: 0.01,
                beta1: 0.9,
                beta2: 0.99,
                eps: 1e-8,
                weight_decay: 0.01,
            },
            code_optimizer: NativeAdamWRecipe {
                lr: 0.6,
                beta1: 0.9,
                beta2: 0.99,
                eps: 1e-8,
                weight_decay: 0.01,
            },
            max_code_change_fraction: 0.5,
            max_relative_code_change: Some(0.9),
        },
        objective: [5; 32],
        teacher_source: None,
        training: (0..4)
            .map(|i| NativeRefinementWindow {
                content: [i + 10; 32],
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
) -> AdditiveTensor {
    tensor_group(law, codec, basis, mode, transport, 32)
}
fn tensor_group(
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
        cols: if basis == Basis::Identity { 260 } else { 512 },
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
        vec![0; 6]
    } else {
        [0, 1, admitted.max_planes, 2, 2, 0]
            .map(|n| n.min(admitted.max_planes))
            .to_vec()
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
    let scale = if admitted.law.precision == ScalePrecision::F32 {
        f32::from_bits(0x3f00_0001)
    } else {
        0.5
    };
    if mode == 0 {
        AdditiveTensor::new(layout, admitted.max_planes, codes, vec![scale; s]).unwrap()
    } else {
        AdditiveTensor::new_per_tile(layout, counts, codes, vec![scale; s]).unwrap()
    }
}

fn active(w: &AdditiveTensor) -> usize {
    if w.plane_count() == 0 {
        0
    } else {
        w.view().plane_trits(0).unwrap().len()
    }
}
fn limits() -> Limits {
    Limits {
        max_journal_bytes: 100_000_000,
        max_working_bytes: 100_000_000,
    }
}
fn id(bytes: &[u8]) -> BlobId {
    BlobId::from_bytes(*blake3::hash(bytes).as_bytes())
}
fn gradients(s: &Session, step: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let scale = s
        .weights()
        .enumerate()
        .map(|(o, w)| {
            (0..w.scales().len())
                .map(|i| ((i + o + step) % 7) as f32 * 0.125 - 0.375)
                .collect()
        })
        .collect();
    let weight = s
        .weights()
        .enumerate()
        .map(|(o, w)| {
            (0..active(w))
                .map(|i| ((i * 3 + o + step) % 9) as f32 * 0.125 - 0.5)
                .collect()
        })
        .collect();
    (scale, weight)
}
fn recorded<W: Write + Seek>(s: &mut Session, w: &mut Writer<W>) {
    let (a, b) = gradients(s, s.completed_updates() as usize);
    s.step_recorded(
        &a.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        &b.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        w,
    )
    .unwrap();
}
fn tiny() -> AdditiveTensor {
    AdditiveTensor::new(
        AdditiveLayout {
            rows: 1,
            cols: 4,
            tile: 256,
            group: 32,
            max_planes: 1,
            allocation: PlaneAllocation::Uniform,
            codec: PlaneCodec::D2,
            law: ADMITTED_LAWS[3].law,
            basis: Basis::Identity,
            transport: Transport::Raw,
        },
        1,
        vec![Trit::POS; 4],
        vec![1.0],
    )
    .unwrap()
}
fn run(parents: Vec<AdditiveTensor>) -> (NativeRefinementPlan, Session, Vec<u8>, Summary) {
    let p = plan(parents.len());
    let mut s = Session::new(p.clone(), parents).unwrap();
    let mut w = Writer::new(Cursor::new(Vec::new()), &s, limits()).unwrap();
    for _ in 0..p.training.len() {
        recorded(&mut s, &mut w);
    }
    let (bytes, summary) = w.finish(&s).unwrap();
    (p, s, bytes.into_inner(), summary)
}
fn reseal(bytes: &mut [u8]) {
    let n = bytes.len() - 32;
    let hash = blake3::hash(&bytes[..n]);
    bytes[n..].copy_from_slice(hash.as_bytes());
}

#[test]
fn all_profiles_replay_and_every_continuation_cursor_match_uninterrupted_bytes() {
    let mut cases = 0;
    let mut continuations = 0;
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
                for mode in 0..3 {
                    for transport in [
                        Transport::Raw,
                        Transport::JointHuffman { block: 128 },
                        Transport::JointHuffman { block: 256 },
                    ] {
                        for group in [32, 128] {
                            let parent = tensor_group(law, codec, basis, mode, transport, group);
                            let (p, final_state, full, summary) = run(vec![parent.clone()]);
                            let cp = final_state.checkpoint_bytes().unwrap();
                            assert_eq!(summary.bytes(), full.len() as u64);
                            assert_eq!(summary.journal(), id(&full));
                            assert_eq!(summary.checkpoint(), id(&cp));
                            let ledger = final_state.journal_size_ledger().unwrap();
                            assert_eq!(ledger.journal_bytes, full.len() as u64);
                            let (again, verified) = replay(
                                p.clone(),
                                vec![parent.clone()],
                                Cursor::new(&full),
                                summary.journal(),
                                &cp,
                                limits(),
                            )
                            .unwrap();
                            assert_eq!(verified.summary(), &summary);
                            verified.verify_session(&final_state).unwrap();
                            assert_eq!(again.checkpoint_bytes().unwrap(), cp);
                            let mut s = Session::new(p.clone(), vec![parent.clone()]).unwrap();
                            let mut w = Writer::new(Cursor::new(Vec::new()), &s, limits()).unwrap();
                            for cursor in 0..=p.training.len() {
                                let (saved, point) = w.checkpoint(&s).unwrap();
                                let prefix = full[..point.journal_prefix_bytes as usize].to_vec();
                                assert_eq!(point.journal_prefix, id(&prefix));
                                let (mut recovered, mut out) = resume(
                                    NativePvJournalResume {
                                        plan: p.clone(),
                                        parents: vec![parent.clone()],
                                        source: Cursor::new(prefix.clone()),
                                        point: &point,
                                        checkpoint: &saved,
                                        tail: Tail::ExactPrefix,
                                    },
                                    Cursor::new(Vec::new()),
                                    limits(),
                                )
                                .unwrap();
                                assert_eq!(recovered.checkpoint_bytes().unwrap(), saved);
                                for _ in cursor..p.training.len() {
                                    recorded(&mut recovered, &mut out);
                                }
                                let (out, observed) = out.finish(&recovered).unwrap();
                                assert_eq!(out.into_inner(), full);
                                assert_eq!(observed, summary);
                                assert_eq!(recovered.checkpoint_bytes().unwrap(), cp);
                                continuations += 1;
                                if cursor < p.training.len() {
                                    recorded(&mut s, &mut w);
                                }
                            }
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases, 648);
    assert_eq!(continuations, 3240);
    eprintln!("NATIVE_PV_JOURNAL profiles={cases} continuation_cursors={continuations}");
}

#[test]
fn corruption_wrong_inputs_and_well_formed_forged_state_never_gain_replay() {
    let parent = tiny();
    let (p, s, full, summary) = run(vec![parent.clone()]);
    let cp = s.checkpoint_bytes().unwrap();
    for i in 0..full.len() {
        let mut bad = full.clone();
        bad[i] ^= 1;
        assert!(
            replay(
                p.clone(),
                vec![parent.clone()],
                Cursor::new(bad),
                summary.journal(),
                &cp,
                limits()
            )
            .is_err()
        );
    }
    for i in 0..full.len() {
        assert!(
            replay(
                p.clone(),
                vec![parent.clone()],
                Cursor::new(&full[..i]),
                summary.journal(),
                &cp,
                limits()
            )
            .is_err()
        );
    }
    let mut extra = full.clone();
    extra.push(0);
    assert!(
        replay(
            p.clone(),
            vec![parent.clone()],
            Cursor::new(extra),
            summary.journal(),
            &cp,
            limits()
        )
        .is_err()
    );
    // A positive finite first moment with a valid checkpoint checksum is legal
    // resumable state, but not the state actually obtained from this journal.
    let mut forged = cp.clone();
    forged[125..129].copy_from_slice(&0.125f32.to_le_bytes());
    reseal(&mut forged);
    let forged_session = Session::resume(p.clone(), vec![parent.clone()], &forged).unwrap();
    let (_, proof) = replay(
        p.clone(),
        vec![parent.clone()],
        Cursor::new(&full),
        summary.journal(),
        &cp,
        limits(),
    )
    .unwrap();
    assert!(proof.verify_session(&forged_session).is_err());
    let mut changed = full.clone();
    let n = changed.len();
    changed[n - 64..n - 32].copy_from_slice(id(&forged).as_bytes());
    reseal(&mut changed);
    assert!(
        replay(
            p.clone(),
            vec![parent.clone()],
            Cursor::new(&changed),
            id(&changed),
            &forged,
            limits()
        )
        .is_err()
    );
    for offset in [0, 8, 40, 72, 80, 88, 96, 104, 112] {
        let mut changed = full.clone();
        changed[offset] ^= 127;
        reseal(&mut changed);
        assert!(
            replay(
                p.clone(),
                vec![parent.clone()],
                Cursor::new(&changed),
                id(&changed),
                &cp,
                limits()
            )
            .is_err()
        );
    }
    let mut other = p.clone();
    other.objective = [99; 32];
    assert!(
        replay(
            other,
            vec![parent.clone()],
            Cursor::new(&full),
            summary.journal(),
            &cp,
            limits()
        )
        .is_err()
    );
    let mut other = parent.clone();
    other.replace_scales(vec![0.5]).unwrap();
    assert!(
        replay(
            p.clone(),
            vec![other],
            Cursor::new(&full),
            summary.journal(),
            &cp,
            limits()
        )
        .is_err()
    );
    assert!(
        replay(
            p,
            vec![parent],
            Cursor::new(&full),
            BlobId::from_bytes([99; 32]),
            &cp,
            limits()
        )
        .is_err()
    );
}

struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        panic!("I/O before admission")
    }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        panic!("I/O before admission")
    }
    fn flush(&mut self) -> io::Result<()> {
        panic!("I/O before admission")
    }
}
impl Seek for NoIo {
    fn seek(&mut self, _: SeekFrom) -> io::Result<u64> {
        panic!("I/O before admission")
    }
}

#[test]
fn exact_journal_caps_reject_before_io_and_new_writer_requires_zero_empty_state() {
    let parent = tiny();
    let s = Session::new(plan(1), vec![parent.clone()]).unwrap();
    let ledger = s.journal_size_ledger().unwrap();
    for l in [
        Limits {
            max_journal_bytes: ledger.journal_bytes - 1,
            max_working_bytes: u64::MAX,
        },
        Limits {
            max_journal_bytes: ledger.journal_bytes,
            max_working_bytes: ledger.writer_working_bytes - 1,
        },
    ] {
        assert!(matches!(Writer::new(NoIo, &s, l), Err(Error::Limit(_))));
    }
    assert!(Writer::new(Cursor::new(vec![1]), &s, limits()).is_err());
    let mut zero = Writer::new(Cursor::new(Vec::new()), &s, limits()).unwrap();
    let (initial, point) = zero.checkpoint(&s).unwrap();
    assert!(zero.finish(&s).is_err());
    assert!(matches!(
        resume(
            NativePvJournalResume {
                plan: plan(1),
                parents: vec![parent.clone()],
                source: NoIo,
                point: &point,
                checkpoint: &initial,
                tail: Tail::ExactPrefix
            },
            NoIo,
            Limits {
                max_journal_bytes: ledger.journal_bytes,
                max_working_bytes: ledger.replay_working_bytes - 1
            }
        ),
        Err(Error::Limit(_))
    ));
    let (p, done, full, summary) = run(vec![parent.clone()]);
    let cp = done.checkpoint_bytes().unwrap();
    assert!(Writer::new(Cursor::new(Vec::new()), &done, limits()).is_err());
    for l in [
        Limits {
            max_journal_bytes: ledger.journal_bytes - 1,
            max_working_bytes: u64::MAX,
        },
        Limits {
            max_journal_bytes: ledger.journal_bytes,
            max_working_bytes: ledger.replay_working_bytes - 1,
        },
    ] {
        assert!(matches!(
            replay(
                p.clone(),
                vec![parent.clone()],
                NoIo,
                summary.journal(),
                &cp,
                l
            ),
            Err(Error::Limit(_))
        ));
    }
    let exact = Limits {
        max_journal_bytes: ledger.journal_bytes,
        max_working_bytes: ledger.replay_working_bytes,
    };
    replay(
        p,
        vec![parent],
        Cursor::new(full),
        summary.journal(),
        &cp,
        exact,
    )
    .unwrap();
}

#[test]
fn prefix_tail_policies_wrong_points_and_nonempty_outputs_preserve_recovery_source() {
    let parent = tiny();
    let p = plan(1);
    let mut s = Session::new(p.clone(), vec![parent.clone()]).unwrap();
    let mut w = Writer::new(Cursor::new(Vec::new()), &s, limits()).unwrap();
    recorded(&mut s, &mut w);
    let (saved, point) = w.checkpoint(&s).unwrap();
    for _ in 1..4 {
        recorded(&mut s, &mut w);
    }
    let (full, _) = w.finish(&s).unwrap();
    let full = full.into_inner();
    let occupied = FaultIo::new();
    occupied
        .data
        .borrow_mut()
        .get_mut()
        .extend_from_slice(b"occupied");
    assert!(
        resume(
            NativePvJournalResume {
                plan: p.clone(),
                parents: vec![parent.clone()],
                source: Cursor::new(&full),
                point: &point,
                checkpoint: &saved,
                tail: Tail::CheckpointPrefix
            },
            occupied.clone(),
            limits()
        )
        .is_err()
    );
    assert_eq!(occupied.data.borrow().get_ref().as_slice(), b"occupied");
    assert!(
        resume(
            NativePvJournalResume {
                plan: p.clone(),
                parents: vec![parent.clone()],
                source: Cursor::new(&full),
                point: &point,
                checkpoint: &saved,
                tail: Tail::ExactPrefix
            },
            Cursor::new(Vec::new()),
            limits()
        )
        .is_err()
    );
    let (state, out) = resume(
        NativePvJournalResume {
            plan: p.clone(),
            parents: vec![parent.clone()],
            source: Cursor::new(&full),
            point: &point,
            checkpoint: &saved,
            tail: Tail::CheckpointPrefix,
        },
        Cursor::new(Vec::new()),
        limits(),
    )
    .unwrap();
    assert_eq!(state.checkpoint_bytes().unwrap(), saved);
    assert_eq!(
        out.into_inner().into_inner(),
        full[..point.journal_prefix_bytes as usize]
    );
    for mut bad in [point.clone(), point.clone(), point.clone(), point.clone()]
        .into_iter()
        .enumerate()
    {
        match bad.0 {
            0 => bad.1.plan = NativeRefinementPlanId::from_bytes([99; 32]),
            1 => bad.1.completed_updates += 1,
            2 => bad.1.journal_prefix_bytes -= 1,
            _ => bad.1.checkpoint_bytes += 1,
        };
        assert!(
            resume(
                NativePvJournalResume {
                    plan: p.clone(),
                    parents: vec![parent.clone()],
                    source: NoIo,
                    point: &bad.1,
                    checkpoint: &saved,
                    tail: Tail::CheckpointPrefix
                },
                NoIo,
                limits()
            )
            .is_err()
        );
    }
    assert!(
        resume(
            NativePvJournalResume {
                plan: p,
                parents: vec![parent],
                source: Cursor::new(&full),
                point: &point,
                checkpoint: &saved,
                tail: Tail::CheckpointPrefix
            },
            Cursor::new(vec![7]),
            limits()
        )
        .is_err()
    );
}

#[test]
fn heterogeneous_owners_keep_channel_inventory_and_state_through_replay_and_resume() {
    let parents = vec![
        tiny(),
        tensor(
            2,
            PlaneCodec::S34,
            Basis::SignedRht {
                block: 128,
                seed: 91,
                domain: 3,
            },
            1,
            Transport::Raw,
        ),
    ];
    let (p, done, full, summary) = run(parents.clone());
    let cp = done.checkpoint_bytes().unwrap();
    let (_, proof) = replay(
        p.clone(),
        parents.clone(),
        Cursor::new(&full),
        summary.journal(),
        &cp,
        limits(),
    )
    .unwrap();
    proof.verify_session(&done).unwrap();
    let mut s = Session::new(p.clone(), parents.clone()).unwrap();
    let mut w = Writer::new(Cursor::new(Vec::new()), &s, limits()).unwrap();
    recorded(&mut s, &mut w);
    recorded(&mut s, &mut w);
    let (saved, point) = w.checkpoint(&s).unwrap();
    let (mut continued, mut output) = resume(
        NativePvJournalResume {
            plan: p,
            parents,
            source: Cursor::new(&full),
            point: &point,
            checkpoint: &saved,
            tail: Tail::CheckpointPrefix,
        },
        Cursor::new(Vec::new()),
        limits(),
    )
    .unwrap();
    for _ in 2..4 {
        recorded(&mut continued, &mut output);
    }
    let (bytes, again) = output.finish(&continued).unwrap();
    assert_eq!(bytes.into_inner(), full);
    assert_eq!(again, summary);
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
                panic!("injected P/V write unwind")
            };
            return Err(io::Error::from_raw_os_error(28));
        }
        let n = bytes.len().min(self.remaining.get());
        self.remaining.set(self.remaining.get() - n);
        self.data.borrow_mut().write(&bytes[..n])
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.flush.get() {
            Err(io::Error::from_raw_os_error(5))
        } else {
            Ok(())
        }
    }
}
impl Seek for FaultIo {
    fn seek(&mut self, p: SeekFrom) -> io::Result<u64> {
        self.data.borrow_mut().seek(p)
    }
}

#[test]
fn numerical_refusal_writes_nothing_and_io_failures_poison_without_committing() {
    let parent = tensor(1, PlaneCodec::B3, Basis::Identity, 0, Transport::Raw);
    let p = plan(1);
    for failure in 0..3 {
        let mut s = Session::new(p.clone(), vec![parent.clone()]).unwrap();
        let io = FaultIo::new();
        let mut w = Writer::new(io.clone(), &s, limits()).unwrap();
        let cp = s.checkpoint_bytes().unwrap();
        let header = io.data.borrow().get_ref().clone();
        let (a, b) = gradients(&s, 0);
        let mut bad = b.clone();
        bad[0][0] = f32::NAN;
        assert!(matches!(
            s.step_recorded(
                &a.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                &bad.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                &mut w
            ),
            Err(NativePvStageError::Update(_))
        ));
        assert_eq!(*io.data.borrow().get_ref(), header);
        assert_eq!(s.checkpoint_bytes().unwrap(), cp);
        match failure {
            0 => io.remaining.set(7),
            1 => io.flush.set(true),
            _ => {
                io.remaining.set(9);
                io.unwind.set(true);
            }
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            s.step_recorded(
                &a.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                &b.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                &mut w,
            )
        }));
        if failure == 2 {
            assert!(result.is_err());
        } else {
            match result.unwrap() {
                Err(NativePvStageError::Stage(Error::Io(error))) => assert_eq!(
                    error.raw_os_error(),
                    Some(if failure == 0 { 28 } else { 5 })
                ),
                other => panic!("unexpected {other:?}"),
            }
        }
        assert_eq!(s.checkpoint_bytes().unwrap(), cp);
        io.remaining.set(usize::MAX);
        io.flush.set(false);
        io.unwind.set(false);
        assert!(
            w.checkpoint(&s)
                .unwrap_err()
                .to_string()
                .contains("poisoned")
        );
        assert!(w.finish(&s).unwrap_err().to_string().contains("poisoned"));
    }
    let mut s = Session::new(p, vec![parent]).unwrap();
    let io = FaultIo::new();
    let mut w = Writer::new(io.clone(), &s, limits()).unwrap();
    io.flush.set(true);
    assert!(w.checkpoint(&s).is_err());
    io.flush.set(false);
    let (a, b) = gradients(&s, 0);
    assert!(
        s.step_recorded(
            &a.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            &b.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            &mut w
        )
        .is_err()
    );
    assert_eq!(s.completed_updates(), 0);
}

struct Changing {
    inner: Cursor<Vec<u8>>,
    starts: usize,
    change_on: usize,
    fail: Option<(usize, bool)>,
}
impl Read for Changing {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        if self.fail == Some((self.starts, false)) {
            return Err(io::Error::from_raw_os_error(5));
        }
        self.inner.read(bytes)
    }
}
impl Seek for Changing {
    fn seek(&mut self, p: SeekFrom) -> io::Result<u64> {
        if p == SeekFrom::Start(0) {
            self.starts += 1;
            if self.starts == self.change_on {
                self.inner.get_mut()[8] ^= 1;
            }
            if self.fail == Some((self.starts, true)) {
                return Err(io::Error::from_raw_os_error(29));
            }
        }
        self.inner.seek(p)
    }
}

#[test]
fn changed_sources_and_original_read_seek_copy_flush_errors_fail_closed() {
    let parent = tiny();
    let (p, done, full, summary) = run(vec![parent.clone()]);
    let cp = done.checkpoint_bytes().unwrap();
    assert!(
        replay(
            p.clone(),
            vec![parent.clone()],
            Changing {
                inner: Cursor::new(full.clone()),
                starts: 0,
                change_on: 2,
                fail: None
            },
            summary.journal(),
            &cp,
            limits()
        )
        .is_err()
    );
    let mut s = Session::new(p.clone(), vec![parent.clone()]).unwrap();
    let mut w = Writer::new(Cursor::new(Vec::new()), &s, limits()).unwrap();
    recorded(&mut s, &mut w);
    let (saved, point) = w.checkpoint(&s).unwrap();
    let prefix = full[..point.journal_prefix_bytes as usize].to_vec();
    for pass in 1..=3 {
        let source = Changing {
            inner: Cursor::new(prefix.clone()),
            starts: 0,
            change_on: pass,
            fail: None,
        };
        assert!(
            resume(
                NativePvJournalResume {
                    plan: p.clone(),
                    parents: vec![parent.clone()],
                    source,
                    point: &point,
                    checkpoint: &saved,
                    tail: Tail::ExactPrefix
                },
                Cursor::new(Vec::new()),
                limits()
            )
            .is_err()
        );
        for seek in [false, true] {
            let source = Changing {
                inner: Cursor::new(prefix.clone()),
                starts: 0,
                change_on: usize::MAX,
                fail: Some((pass, seek)),
            };
            match resume(
                NativePvJournalResume {
                    plan: p.clone(),
                    parents: vec![parent.clone()],
                    source,
                    point: &point,
                    checkpoint: &saved,
                    tail: Tail::ExactPrefix,
                },
                Cursor::new(Vec::new()),
                limits(),
            ) {
                Err(Error::Io(error)) => {
                    assert_eq!(error.raw_os_error(), Some(if seek { 29 } else { 5 }))
                }
                _ => panic!("original I/O error was lost"),
            }
        }
    }
    for flush in [false, true] {
        let out = FaultIo::new();
        if flush {
            out.flush.set(true)
        } else {
            out.remaining.set(7)
        }
        assert!(matches!(
            resume(
                NativePvJournalResume {
                    plan: p.clone(),
                    parents: vec![parent.clone()],
                    source: Cursor::new(prefix.clone()),
                    point: &point,
                    checkpoint: &saved,
                    tail: Tail::ExactPrefix
                },
                out.clone(),
                limits()
            ),
            Err(Error::Io(_))
        ));
        assert_eq!(full[..point.journal_prefix_bytes as usize], prefix);
    }
    // Updating the saved hash does not make forged moments numerically true.
    let mut forged = saved.clone();
    forged[125..129].copy_from_slice(&0.125f32.to_le_bytes());
    reseal(&mut forged);
    Session::resume(p.clone(), vec![parent.clone()], &forged).unwrap();
    let mut bad = point.clone();
    bad.checkpoint = id(&forged);
    assert!(
        resume(
            NativePvJournalResume {
                plan: p,
                parents: vec![parent],
                source: Cursor::new(prefix),
                point: &bad,
                checkpoint: &forged,
                tail: Tail::ExactPrefix
            },
            Cursor::new(Vec::new()),
            limits()
        )
        .is_err()
    );
}

#[derive(Debug)]
struct Bounded {
    inner: Cursor<Vec<u8>>,
    maximum: Rc<Cell<usize>>,
}
impl Read for Bounded {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        assert!(bytes.len() <= 8192);
        self.maximum.set(self.maximum.get().max(bytes.len()));
        self.inner.read(bytes)
    }
}
impl Write for Bounded {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        assert!(bytes.len() <= 8192);
        self.maximum.set(self.maximum.get().max(bytes.len()));
        self.inner.write(bytes)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
impl Seek for Bounded {
    fn seek(&mut self, p: SeekFrom) -> io::Result<u64> {
        self.inner.seek(p)
    }
}

#[test]
fn multichunk_record_replay_and_continuation_bound_every_io_request() {
    let layout = AdditiveLayout {
        rows: 1,
        cols: 16384,
        tile: 256,
        group: 128,
        max_planes: 1,
        allocation: PlaneAllocation::Uniform,
        codec: PlaneCodec::D2,
        law: ADMITTED_LAWS[3].law,
        basis: Basis::Identity,
        transport: Transport::Raw,
    };
    let parent = AdditiveTensor::new(layout, 1, vec![Trit::POS; 16384], vec![0.5]).unwrap();
    let mut p = plan(1);
    p.limits.max_state_bytes = 10_000_000;
    p.limits.max_workspace_bytes = 10_000_000;
    p.limits.max_checkpoint_bytes = 10_000_000;
    let maximum = Rc::new(Cell::new(0));
    let mut s = Session::new(p.clone(), vec![parent.clone()]).unwrap();
    let mut w = Writer::new(
        Bounded {
            inner: Cursor::new(Vec::new()),
            maximum: maximum.clone(),
        },
        &s,
        limits(),
    )
    .unwrap();
    recorded(&mut s, &mut w);
    let (saved, point) = w.checkpoint(&s).unwrap();
    for _ in 1..4 {
        recorded(&mut s, &mut w);
    }
    let (out, summary) = w.finish(&s).unwrap();
    let full = out.inner.into_inner();
    assert_eq!(maximum.get(), 8192);
    let cp = s.checkpoint_bytes().unwrap();
    let (_, proof) = replay(
        p.clone(),
        vec![parent.clone()],
        Bounded {
            inner: Cursor::new(full.clone()),
            maximum: maximum.clone(),
        },
        summary.journal(),
        &cp,
        limits(),
    )
    .unwrap();
    proof.verify_session(&s).unwrap();
    let (mut continued, mut writer) = resume(
        NativePvJournalResume {
            plan: p,
            parents: vec![parent],
            source: Bounded {
                inner: Cursor::new(full.clone()),
                maximum: maximum.clone(),
            },
            point: &point,
            checkpoint: &saved,
            tail: Tail::CheckpointPrefix,
        },
        Bounded {
            inner: Cursor::new(Vec::new()),
            maximum: maximum.clone(),
        },
        limits(),
    )
    .unwrap();
    for _ in 1..4 {
        recorded(&mut continued, &mut writer);
    }
    let (out, again) = writer.finish(&continued).unwrap();
    assert_eq!(again, summary);
    assert_eq!(out.inner.into_inner(), full);
    eprintln!(
        "NATIVE_PV_JOURNAL_IO bytes={} maximum_request={}",
        full.len(),
        maximum.get()
    );
}

#[test]
fn wire_bytes_encode_owner_channel_order_raw_weighting_and_strict_point_schema() {
    let p = plan(2);
    let mut s = Session::new(p.clone(), vec![tiny(), tiny()]).unwrap();
    let mut w = Writer::new(Cursor::new(Vec::new()), &s, limits()).unwrap();
    let a = [-0.0f32];
    let b = [-0.0f32, 0.125, -0.25, 0.375];
    s.step_recorded(&[&a, &[0.25]], &[&b, &b], &mut w).unwrap();
    let (cp, point) = w.checkpoint(&s).unwrap();
    let value = serde_json::to_value(&point).unwrap();
    assert_eq!(
        serde_json::from_value::<NativePvResumePoint>(value.clone()).unwrap(),
        point
    );
    let mut bad = value;
    bad.as_object_mut()
        .unwrap()
        .insert("ignored".into(), true.into());
    assert!(serde_json::from_value::<NativePvResumePoint>(bad).is_err());
    let bytes = w.into_inner().into_inner();
    assert_eq!(&bytes[..8], b"TNPJ\0\0\0\x01");
    assert_eq!(&cp[..4], b"TNPV");
    assert_eq!(point.journal_prefix_bytes, 120 + 72 + 40);
    assert_eq!(&bytes[88..96], &1u64.to_le_bytes());
    assert_eq!(&bytes[96..104], &4u64.to_le_bytes());
    assert_eq!(&bytes[120..128], &0u64.to_le_bytes());
    assert_eq!(&bytes[128..160], &p.training[0].content);
    for (i, value) in a
        .iter()
        .chain(&b)
        .chain([0.25f32].iter())
        .chain(&b)
        .enumerate()
    {
        assert_eq!(&bytes[160 + 4 * i..164 + 4 * i], &value.to_le_bytes());
    }
    let mut hash = blake3::Hasher::new();
    hash.update(b"tritium.native-pv.journal-frame.v1\0");
    hash.update(&bytes[120..200]);
    assert_eq!(&bytes[200..232], hash.finalize().as_bytes());
}
