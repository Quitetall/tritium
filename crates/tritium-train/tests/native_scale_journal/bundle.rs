//! Real persistence and adversarial receiving through the public recovery interface.
use super::*;
use tritium_train::pv_tuning::{
    NativeScaleResumeBundleError as BundleError, NativeScaleResumeBundleLimits as BundleLimits,
    NativeScaleResumeBundleReader as Bundle, publish_native_scale_resume_bundle as publish,
    stage_native_scale_resume_bundle as stage,
};
impl Read for FaultIo {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        self.data.borrow_mut().read(b)
    }
}
fn caps() -> BundleLimits {
    BundleLimits {
        max_bundle_bytes: 10_000_000,
        max_manifest_bytes: 100_000,
        max_checkpoint_bytes: 1_000_000,
        max_working_bytes: 2_000_000,
    }
}
fn parents() -> Vec<AdditiveTensor> {
    let mut b = simple();
    b.replace_scales(vec![0.75; b.scales().len()]).unwrap();
    vec![simple(), b]
}
fn advance<W: Write + Seek>(s: &mut Session, w: &mut Writer<W>, end: usize) {
    for i in s.completed_updates() as usize..end {
        let g = gradients(s, i);
        s.step_recorded(&g.iter().map(Vec::as_slice).collect::<Vec<_>>(), w)
            .unwrap();
    }
}
fn saved(k: usize) -> (Session, Writer<FaultIo>, FaultIo) {
    let mut s = Session::new(plan(2), parents()).unwrap();
    let io = FaultIo::new();
    let mut w = Writer::new(io.clone(), &s, limits()).unwrap();
    advance(&mut s, &mut w, k);
    (s, w, io)
}
fn encoded(k: usize) -> (Vec<u8>, BlobId) {
    let (s, mut w, _) = saved(k);
    let out = FaultIo::new();
    let reader = stage(&s, &mut w, out.clone(), caps()).unwrap();
    let bytes = out.data.borrow().get_ref().clone();
    (bytes, reader.id())
}
fn regions(b: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let m = u64::from_le_bytes(b[8..16].try_into().unwrap()) as usize;
    let c = u64::from_le_bytes(b[16..24].try_into().unwrap()) as usize;
    let p = u64::from_le_bytes(b[24..32].try_into().unwrap()) as usize;
    (
        b[32..32 + m].to_vec(),
        b[32 + m..32 + m + c].to_vec(),
        b[32 + m + c..32 + m + c + p].to_vec(),
    )
}
fn pack(m: &[u8], c: &[u8], p: &[u8]) -> Vec<u8> {
    let mut out = b"TNSB\0\0\0\x01".to_vec();
    for n in [m.len(), c.len(), p.len()] {
        out.extend_from_slice(&(n as u64).to_le_bytes());
    }
    out.extend_from_slice(m);
    out.extend_from_slice(c);
    out.extend_from_slice(p);
    let body = blake3::hash(&out);
    out.extend_from_slice(body.as_bytes());
    out
}

#[test]
fn all_cursors_are_self_contained_and_live_writer_continues_identically() {
    for k in 0..=4 {
        let (mut s, mut w, original) = saved(k);
        let before = s.checkpoint_bytes().unwrap();
        let prefix = original.data.borrow().get_ref().clone();
        let out = FaultIo::new();
        let reader = stage(&s, &mut w, out.clone(), caps()).unwrap();
        assert_eq!(reader.manifest().plan, s.plan().clone());
        assert_eq!(reader.manifest().point.completed_updates, k as u64);
        assert_eq!(original.data.borrow().position(), prefix.len() as u64);
        assert_eq!(*original.data.borrow().get_ref(), prefix);
        assert_eq!(s.checkpoint_bytes().unwrap(), before);
        let encoded = out.data.borrow().get_ref().clone();
        assert_eq!(reader.id(), digest(&encoded));
        assert_eq!(reader.bytes(), encoded.len() as u64);
        let (m, c, p) = regions(&encoded);
        assert_eq!(pack(&m, &c, &p), encoded);
        assert_eq!(p, prefix);
        let (mut resumed, mut continued) = reader
            .resume(parents(), Cursor::new(Vec::new()), limits())
            .unwrap();
        advance(&mut resumed, &mut continued, 4);
        advance(&mut s, &mut w, 4);
        let (output, summary) = continued.finish(&resumed).unwrap();
        let (_, wanted) = w.finish(&s).unwrap();
        assert_eq!(summary, wanted);
        assert_eq!(*original.data.borrow().get_ref(), output.into_inner());
        assert_eq!(
            s.checkpoint_bytes().unwrap(),
            resumed.checkpoint_bytes().unwrap()
        );
    }
}

#[test]
fn every_byte_truncation_and_tail_refuses_the_original_identity() {
    let (b, id) = encoded(2);
    for i in 0..b.len() {
        let mut changed = b.clone();
        changed[i] ^= 1;
        assert!(
            Bundle::open(Cursor::new(changed), id, caps()).is_err(),
            "byte{i}"
        );
    }
    for n in 0..b.len() {
        assert!(
            Bundle::open(Cursor::new(&b[..n]), id, caps()).is_err(),
            "length{n}"
        );
    }
    let mut trailing = b;
    trailing.push(0);
    let id = digest(&trailing);
    assert!(Bundle::open(Cursor::new(trailing), id, caps()).is_err());
}

#[test]
fn resealed_bad_manifest_bindings_and_forged_moments_are_not_numerical_proof() {
    let (b, _) = encoded(2);
    let (m, mut c, p) = regions(&b);
    let original: NativeScaleResumeManifest = serde_json::from_slice(&m).unwrap();
    let mut undersized = original.clone();
    undersized.plan.limits.max_checkpoint_bytes = 1;
    undersized.point.plan = tritium_evidence::native_refinement_plan_id(&undersized.plan).unwrap();
    let contradictory = pack(&serde_json::to_vec(&undersized).unwrap(), &c, &p);
    assert!(Bundle::open(Cursor::new(&contradictory), digest(&contradictory), caps()).is_err());
    for mutation in 0..7 {
        let mut value = serde_json::from_slice::<serde_json::Value>(&m).unwrap();
        match mutation {
            0 => value["unexpected"] = true.into(),
            1 => value["point"]["unexpected"] = true.into(),
            2 => value["point"]["completed_updates"] = 5.into(),
            3 => value["point"]["checkpoint_bytes"] = 1.into(),
            4 => value["point"]["journal_prefix_bytes"] = 1.into(),
            5 => value["plan"]["parent"]["level"] = "different".into(),
            6 => {
                value["point"]["plan"] =
                    serde_json::to_value(NativeRefinementPlanId::from_bytes([0; 32])).unwrap()
            }
            _ => unreachable!(),
        }
        let b = pack(&serde_json::to_vec(&value).unwrap(), &c, &p);
        assert!(
            Bundle::open(Cursor::new(&b), digest(&b), caps()).is_err(),
            "mutation{mutation}"
        );
    }
    let duplicate = format!(
        "{{\"plan\":{},\"plan\":{},\"point\":{}}}",
        serde_json::to_string(&original.plan).unwrap(),
        serde_json::to_string(&original.plan).unwrap(),
        serde_json::to_string(&original.point).unwrap()
    );
    let b = pack(duplicate.as_bytes(), &c, &p);
    assert!(Bundle::open(Cursor::new(&b), digest(&b), caps()).is_err());
    let n = c.len();
    c[n - 36..n - 32].copy_from_slice(&0.125f32.to_le_bytes());
    reseal(&mut c);
    Session::resume(original.plan.clone(), parents(), &c).unwrap();
    let mut forged = original;
    forged.point.checkpoint = digest(&c);
    let b = pack(&serde_json::to_vec(&forged).unwrap(), &c, &p);
    // This scope is deliberately only byte consistency. Actual replay refuses the forged state.
    let reader = Bundle::open(Cursor::new(&b), digest(&b), caps()).unwrap();
    assert!(
        reader
            .resume(parents(), Cursor::new(Vec::new()), limits())
            .unwrap_err()
            .to_string()
            .contains("replayed checkpoint")
    );
}

#[test]
fn exact_caps_fail_before_output_io_or_snapshot_flush() {
    let (encoded, id) = encoded(2);
    let (m, c, _) = regions(&encoded);
    let exact = BundleLimits {
        max_bundle_bytes: encoded.len() as u64,
        max_manifest_bytes: m.len(),
        max_checkpoint_bytes: c.len() as u64,
        max_working_bytes: (m.len() + c.len() + 8192) as u64,
    };
    for field in 0..4 {
        let (s, mut w, io) = saved(2);
        io.flush.set(true);
        let mut cap = exact;
        match field {
            0 => cap.max_bundle_bytes -= 1,
            1 => cap.max_manifest_bytes -= 1,
            2 => cap.max_checkpoint_bytes -= 1,
            3 => cap.max_working_bytes -= 1,
            _ => unreachable!(),
        }
        assert!(matches!(
            stage(&s, &mut w, NoIo, cap),
            Err(BundleError::Limit(_))
        ));
        io.flush.set(false);
        w.checkpoint(&s).unwrap();
    }
    let (s, mut w, _) = saved(2);
    stage(&s, &mut w, Cursor::new(Vec::new()), exact).unwrap();
    Bundle::open(Cursor::new(&encoded), id, exact).unwrap();
    let mut cap = exact;
    cap.max_bundle_bytes = 0;
    assert!(matches!(
        Bundle::open(NoIo, id, cap),
        Err(BundleError::Limit(_))
    ));
    let sentinel = FaultIo::new();
    sentinel
        .data
        .borrow_mut()
        .get_mut()
        .extend_from_slice(&[17, 23]);
    assert!(stage(&s, &mut w, sentinel.clone(), exact).is_err());
    assert_eq!(*sentinel.data.borrow().get_ref(), vec![17, 23]);
}

#[test]
fn destination_failures_restore_live_journal_and_allow_new_output_retry() {
    let (b, _) = encoded(2);
    let (m, c, _) = regions(&b);
    for remaining in [0, 5, 32 + m.len() + c.len() + 5, b.len() - 16, usize::MAX] {
        let (mut s, mut w, input) = saved(2);
        let prefix = input.data.borrow().get_ref().clone();
        let out = FaultIo::new();
        out.remaining.set(remaining);
        out.flush.set(remaining == usize::MAX);
        assert!(matches!(
            stage(&s, &mut w, out, caps()),
            Err(BundleError::Io(_))
        ));
        assert_eq!(*input.data.borrow().get_ref(), prefix);
        assert_eq!(input.data.borrow().position(), prefix.len() as u64);
        let retry = stage(&s, &mut w, Cursor::new(Vec::new()), caps()).unwrap();
        assert_eq!(retry.manifest().point.completed_updates, 2);
        advance(&mut s, &mut w, 4);
        let (_, summary) = w.finish(&s).unwrap();
        assert_eq!(summary, run(parents()).3);
    }
}

#[test]
fn prefix_copy_unwind_preserves_state_and_cannot_leave_a_usable_writer() {
    let (b, _) = encoded(2);
    let (m, c, _) = regions(&b);
    let (s, mut w, _) = saved(2);
    let before = s.checkpoint_bytes().unwrap();
    let out = FaultIo::new();
    out.remaining.set(32 + m.len() + c.len() + 5);
    out.unwind.set(true);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| stage(
            &s,
            &mut w,
            out,
            caps()
        )))
        .is_err()
    );
    assert_eq!(s.checkpoint_bytes().unwrap(), before);
    assert!(
        w.checkpoint(&s)
            .unwrap_err()
            .to_string()
            .contains("poisoned")
    );
}

struct Source {
    io: FaultIo,
    read_failure: bool,
    seeks: usize,
    fail_seek: Option<usize>,
    inject: Option<std::path::PathBuf>,
}
impl Read for Source {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        if let Some(path) = self.inject.take() {
            std::fs::write(path, b"foreign sentinel")?;
        }
        if self.read_failure {
            return Err(io::Error::other("injected bundle source read"));
        }
        self.io.read(b)
    }
}
impl Write for Source {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        self.io.write(b)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.io.flush()
    }
}
impl Seek for Source {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        let n = self.seeks;
        self.seeks += 1;
        if self.fail_seek == Some(n) {
            return Err(io::Error::other("injected bundle source seek"));
        }
        self.io.seek(from)
    }
}
#[test]
fn uncertain_source_poison_preserves_session_and_original_io_error() {
    for failure in 0..4 {
        let mut s = Session::new(plan(2), parents()).unwrap();
        let io = FaultIo::new();
        let source = Source {
            io: io.clone(),
            read_failure: false,
            seeks: 0,
            fail_seek: None,
            inject: None,
        };
        let mut w = Writer::new(source, &s, limits()).unwrap();
        advance(&mut s, &mut w, 2);
        // Source seek counts: new writer2; stream_position, start0, end0, restore.
        // External shared storage can change without replacing the retained adapter.
        match failure {
            0 => io.data.borrow_mut().get_mut()[0] ^= 1,
            1 => {
                io.data.borrow_mut().set_position(1);
            }
            2 => {
                io.flush.set(true);
            }
            _ => {
                io.data.borrow_mut().get_mut().truncate(10);
            }
        }
        let before = s.checkpoint_bytes().unwrap();
        assert!(stage(&s, &mut w, Cursor::new(Vec::new()), caps()).is_err());
        assert_eq!(s.checkpoint_bytes().unwrap(), before);
        io.flush.set(false);
        assert!(
            w.checkpoint(&s)
                .unwrap_err()
                .to_string()
                .contains("poisoned")
        );
    }
    for fail_seek in [Some(2), Some(3), Some(4), Some(5), None] {
        let s = Session::new(plan(2), parents()).unwrap();
        let io = FaultIo::new();
        let source = Source {
            io,
            read_failure: fail_seek.is_none(),
            seeks: 0,
            fail_seek,
            inject: None,
        };
        let mut w = Writer::new(source, &s, limits()).unwrap();
        let error = stage(&s, &mut w, Cursor::new(Vec::new()), caps()).unwrap_err();
        assert!(error.to_string().contains("injected bundle source"));
        assert!(
            w.checkpoint(&s)
                .unwrap_err()
                .to_string()
                .contains("poisoned")
        );
    }
}

struct Root(std::path::PathBuf);
impl Root {
    fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "native-scale-bundle-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Root {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn real_publication_recovers_without_old_recording_work_and_never_clobbers() {
    use std::fs::{self, File, OpenOptions};
    let root = Root::new();
    let mut s = Session::new(plan(2), parents()).unwrap();
    let path = root.0.join("live.tnsj");
    let file = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    let mut w = Writer::new(file, &s, limits()).unwrap();
    advance(&mut s, &mut w, 2);
    let directory = root.0.join("saved");
    let saved = publish(&s, &mut w, &directory, caps()).unwrap();
    assert!(saved.warnings.is_empty());
    let id = saved.bundle.id();
    drop(saved);
    let bytes = fs::read(directory.join("resume.tnsb")).unwrap();
    assert!(!directory.join(".pending.tnsb").exists());
    assert!(publish(&s, &mut w, &directory, caps()).is_err());
    assert_eq!(fs::read(directory.join("resume.tnsb")).unwrap(), bytes);
    advance(&mut s, &mut w, 4);
    drop(w.finish(&s).unwrap());
    let expected = fs::read(&path).unwrap();
    fs::remove_file(&path).unwrap();
    let bundle = Bundle::open(
        File::open(directory.join("resume.tnsb")).unwrap(),
        id,
        caps(),
    )
    .unwrap();
    let out = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(root.0.join("continued.tnsj"))
        .unwrap();
    let (mut resumed, mut w) = bundle.resume(parents(), out, limits()).unwrap();
    advance(&mut resumed, &mut w, 4);
    drop(w.finish(&resumed).unwrap());
    assert_eq!(fs::read(root.0.join("continued.tnsj")).unwrap(), expected);
    assert_eq!(
        s.checkpoint_bytes().unwrap(),
        resumed.checkpoint_bytes().unwrap()
    );
}

#[test]
fn failed_publication_removes_only_owned_files_and_preserves_unexpected_sentinel() {
    let root = Root::new();
    let directory = root.0.join("failed");
    let s = Session::new(plan(2), parents()).unwrap();
    let source = Source {
        io: FaultIo::new(),
        read_failure: true,
        seeks: 0,
        fail_seek: None,
        inject: Some(directory.join("foreign.txt")),
    };
    let mut w = Writer::new(source, &s, limits()).unwrap();
    assert!(publish(&s, &mut w, &directory, caps()).is_err());
    assert_eq!(
        std::fs::read(directory.join("foreign.txt")).unwrap(),
        b"foreign sentinel"
    );
    assert!(!directory.join(".pending.tnsb").exists());
    assert!(!directory.join("resume.tnsb").exists());
}

#[test]
fn bundle_recovery_matches_all_648_admitted_reference_profiles() {
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
                            let parent = tensor(law, codec, basis, mode, transport, group);
                            let mut s = Session::new(plan(1), vec![parent.clone()]).unwrap();
                            let mut w = Writer::new(Cursor::new(Vec::new()), &s, limits()).unwrap();
                            advance(&mut s, &mut w, 2);
                            let bundle =
                                stage(&s, &mut w, Cursor::new(Vec::new()), caps()).unwrap();
                            let (mut recovered, mut journal) = bundle
                                .resume(vec![parent.clone()], Cursor::new(Vec::new()), limits())
                                .unwrap();
                            advance(&mut recovered, &mut journal, 4);
                            let (out, summary) = journal.finish(&recovered).unwrap();
                            let (_, expected, bytes, wanted) = run(vec![parent]);
                            assert_eq!(out.into_inner(), bytes);
                            assert_eq!(summary, wanted);
                            assert_eq!(
                                recovered.checkpoint_bytes().unwrap(),
                                expected.checkpoint_bytes().unwrap()
                            );
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases, 648);
}

#[derive(Clone)]
struct Bounded {
    io: FaultIo,
    read: Rc<Cell<usize>>,
    write: Rc<Cell<usize>>,
}
impl Bounded {
    fn new() -> Self {
        Self {
            io: FaultIo::new(),
            read: Rc::new(Cell::new(0)),
            write: Rc::new(Cell::new(0)),
        }
    }
}
impl Read for Bounded {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        assert!(b.len() <= 8192);
        self.read.set(self.read.get().max(b.len()));
        self.io.read(b)
    }
}
impl Write for Bounded {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        assert!(b.len() <= 8192);
        self.write.set(self.write.get().max(b.len()));
        self.io.write(b)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.io.flush()
    }
}
impl Seek for Bounded {
    fn seek(&mut self, s: SeekFrom) -> io::Result<u64> {
        self.io.seek(s)
    }
}

#[test]
fn large_checkpoint_prefix_and_receiving_copy_are_bounded_to_8192_byte_calls() {
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
    let mut s = Session::new(plan(1), vec![parent.clone()]).unwrap();
    let io = Bounded::new();
    let mut w = Writer::new(io.clone(), &s, limits()).unwrap();
    advance(&mut s, &mut w, 2);
    let out = Bounded::new();
    let bundle = stage(&s, &mut w, out.clone(), caps()).unwrap();
    assert!(bundle.bytes() > 8192);
    assert_eq!(io.read.get(), 8192);
    assert_eq!(out.read.get(), 8192);
    assert_eq!(out.write.get(), 8192);
    let output = Bounded::new();
    let (mut resumed, mut journal) = bundle
        .resume(vec![parent], output.clone(), limits())
        .unwrap();
    advance(&mut resumed, &mut journal, 4);
    advance(&mut s, &mut w, 4);
    let (_, summary) = journal.finish(&resumed).unwrap();
    let (_, expected) = w.finish(&s).unwrap();
    assert_eq!(summary, expected);
    assert_eq!(
        *output.io.data.borrow().get_ref(),
        *io.io.data.borrow().get_ref()
    );
}

struct Changed {
    io: FaultIo,
    starts: usize,
    at: usize,
}
impl Read for Changed {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        self.io.read(b)
    }
}
impl Seek for Changed {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        if from == SeekFrom::Start(0) {
            self.starts += 1;
            if self.starts == self.at {
                self.io.data.borrow_mut().get_mut()[40] ^= 1;
            }
        }
        self.io.seek(from)
    }
}
#[test]
fn changes_between_raw_and_parsed_passes_or_before_resume_are_rejected() {
    let (b, id) = encoded(2);
    let io = FaultIo::new();
    io.data.borrow_mut().get_mut().extend_from_slice(&b);
    assert!(
        Bundle::open(
            Changed {
                io: io.clone(),
                starts: 0,
                at: 3
            },
            id,
            caps()
        )
        .is_err()
    );
    *io.data.borrow_mut() = Cursor::new(b);
    let bundle = Bundle::open(io.clone(), id, caps()).unwrap();
    io.data.borrow_mut().get_mut()[40] ^= 1;
    assert!(
        bundle
            .resume(parents(), Cursor::new(Vec::new()), limits())
            .is_err()
    );
}

struct HeaderOnly(Cursor<Vec<u8>>);
impl Read for HeaderOnly {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        assert!(self.0.position() < 32);
        self.0.read(b)
    }
}
impl Seek for HeaderOnly {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        assert_eq!(from, SeekFrom::Start(0));
        self.0.seek(from)
    }
}
#[test]
fn hostile_header_lengths_refuse_before_body_io_or_allocation() {
    for lengths in [
        [u64::MAX, 1, 1],
        [1, u64::MAX, 1],
        [1, 1, u64::MAX],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ] {
        let mut b = b"TNSB\0\0\0\x01".to_vec();
        for n in lengths {
            b.extend_from_slice(&n.to_le_bytes());
        }
        assert!(matches!(
            Bundle::open(HeaderOnly(Cursor::new(b)), digest(b"unused"), caps()),
            Err(BundleError::Limit(_))
        ));
    }
}
