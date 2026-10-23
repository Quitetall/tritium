//! Actual P/V bundle transport/replay, not model or durability qualification.
use super::*;
use tritium_train::pv_tuning::{
    NativePvResumeBundleError as BundleError, NativePvResumeBundleLimits as Caps,
    NativePvResumeBundleReader as Bundle, NativeScaleResumeBundleReader as ScaleBundle,
    publish_native_pv_resume_bundle as publish, stage_native_pv_resume_bundle as stage,
};
impl Read for FaultIo {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        self.data.borrow_mut().read(b)
    }
}
fn caps() -> Caps {
    Caps {
        max_bundle_bytes: 10_000_000,
        max_manifest_bytes: 100_000,
        max_checkpoint_bytes: 1_000_000,
        max_working_bytes: 2_000_000,
    }
}
fn saved(k: usize) -> (Session, Writer<FaultIo>, FaultIo) {
    let mut s = Session::new(plan(1), vec![tiny()]).unwrap();
    let input = FaultIo::new();
    let mut w = Writer::new(input.clone(), &s, limits()).unwrap();
    for _ in 0..k {
        recorded(&mut s, &mut w);
    }
    (s, w, input)
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
    (
        b[32..32 + m].to_vec(),
        b[32 + m..32 + m + c].to_vec(),
        b[32 + m + c..b.len() - 32].to_vec(),
    )
}
fn pack(m: &[u8], c: &[u8], p: &[u8]) -> Vec<u8> {
    let mut out = b"TNPB\0\0\0\x01".to_vec();
    for n in [m.len(), c.len(), p.len()] {
        out.extend_from_slice(&(n as u64).to_le_bytes());
    }
    out.extend_from_slice(m);
    out.extend_from_slice(c);
    out.extend_from_slice(p);
    let hash = blake3::hash(&out);
    out.extend_from_slice(hash.as_bytes());
    out
}

#[test]
fn every_cursor_is_self_contained_and_live_and_recovered_writers_continue_identically() {
    for cursor in 0..=4 {
        let (mut s, mut w, input) = saved(cursor);
        let before = s.checkpoint_bytes().unwrap();
        let prefix = input.data.borrow().get_ref().clone();
        let out = FaultIo::new();
        let bundle = stage(&s, &mut w, out.clone(), caps()).unwrap();
        assert_eq!(bundle.manifest().plan, *s.plan());
        assert_eq!(bundle.manifest().point.completed_updates, cursor as u64);
        let b = out.data.borrow().get_ref().clone();
        let (m, c, p) = regions(&b);
        assert_eq!(b, pack(&m, &c, &p));
        assert_eq!(c, before);
        assert_eq!(p, prefix);
        assert_eq!(input.data.borrow().position(), prefix.len() as u64);
        let (mut r, mut j) = bundle
            .resume(vec![tiny()], Cursor::new(Vec::new()), limits())
            .unwrap();
        for _ in cursor..4 {
            recorded(&mut s, &mut w);
            recorded(&mut r, &mut j);
        }
        let (expected, summary) = w.finish(&s).unwrap();
        let (actual, other) = j.finish(&r).unwrap();
        assert_eq!(expected.data.borrow().get_ref(), actual.get_ref());
        assert_eq!(summary, other);
        assert_eq!(s.checkpoint_bytes().unwrap(), r.checkpoint_bytes().unwrap());
    }
}

#[test]
fn bundle_recovery_matches_all_648_native_profiles() {
    let mut profiles = 0;
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
                            let parent = tensor_group(law, codec, basis, mode, transport, group);
                            let mut s = Session::new(plan(1), vec![parent.clone()]).unwrap();
                            let mut w = Writer::new(Cursor::new(Vec::new()), &s, limits()).unwrap();
                            for _ in 0..2 {
                                recorded(&mut s, &mut w);
                            }
                            let bundle =
                                stage(&s, &mut w, Cursor::new(Vec::new()), caps()).unwrap();
                            let (mut r, mut j) = bundle
                                .resume(vec![parent], Cursor::new(Vec::new()), limits())
                                .unwrap();
                            for _ in 2..4 {
                                recorded(&mut s, &mut w);
                                recorded(&mut r, &mut j);
                            }
                            let (expected, a) = w.finish(&s).unwrap();
                            let (actual, b) = j.finish(&r).unwrap();
                            assert_eq!(actual, expected);
                            assert_eq!(a, b);
                            assert_eq!(
                                r.checkpoint_bytes().unwrap(),
                                s.checkpoint_bytes().unwrap()
                            );
                            profiles += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(profiles, 648);
}

#[test]
fn every_byte_truncation_tail_strict_manifest_and_family_mismatch_refuse() {
    let (b, expected) = encoded(1);
    for i in 0..b.len() {
        let mut changed = b.clone();
        changed[i] ^= 1;
        assert!(Bundle::open(Cursor::new(changed), expected, caps()).is_err());
    }
    for n in 0..b.len() {
        assert!(Bundle::open(Cursor::new(&b[..n]), expected, caps()).is_err());
    }
    let mut tail = b.clone();
    tail.push(0);
    assert!(Bundle::open(Cursor::new(&tail), id(&tail), caps()).is_err());
    assert!(ScaleBundle::open(Cursor::new(&b), expected, caps()).is_err());
    let mut relabeled = b.clone();
    relabeled[..8].copy_from_slice(b"TNSB\0\0\0\x01");
    reseal(&mut relabeled);
    assert!(ScaleBundle::open(Cursor::new(&relabeled), id(&relabeled), caps()).is_err());
    let (m, c, p) = regions(&b);
    for change in 0..5 {
        let mut value: serde_json::Value = serde_json::from_slice(&m).unwrap();
        match change {
            0 => value["extra"] = true.into(),
            1 => value["point"]["extra"] = true.into(),
            2 => value["point"]["completed_updates"] = 5.into(),
            3 => value["point"]["checkpoint_bytes"] = 1.into(),
            _ => {
                value["plan"]["recipe"] = serde_json::json!({"ScaleOnly":{"optimizer":{"lr":0.01,"beta1":0.9,"beta2":0.99,"eps":1e-8,"weight_decay":0.0}}})
            }
        }
        let changed = pack(&serde_json::to_vec(&value).unwrap(), &c, &p);
        assert!(Bundle::open(Cursor::new(&changed), id(&changed), caps()).is_err());
    }
}

#[test]
fn resealed_finite_code_moments_are_byte_consistent_but_not_numerical_replay() {
    let (b, _) = encoded(2);
    let (m, mut c, p) = regions(&b);
    assert_eq!(&c[..4], b"TNPV");
    c[125..129].copy_from_slice(&0.123f32.to_le_bytes());
    reseal(&mut c);
    assert!(Session::resume(plan(1), vec![tiny()], &c).is_ok());
    let mut manifest: NativePvResumeManifest = serde_json::from_slice(&m).unwrap();
    manifest.point.checkpoint = id(&c);
    let forged = pack(&serde_json::to_vec(&manifest).unwrap(), &c, &p);
    let bundle = Bundle::open(Cursor::new(&forged), id(&forged), caps()).unwrap();
    assert!(matches!(
        bundle.resume(vec![tiny()], Cursor::new(Vec::new()), limits()),
        Err(BundleError::PvJournal(_))
    ));
}

#[test]
fn exact_caps_precede_output_io_and_failed_destination_restores_live_append() {
    let (b, _) = encoded(2);
    let (m, c, _) = regions(&b);
    let exact = Caps {
        max_bundle_bytes: b.len() as u64,
        max_manifest_bytes: m.len(),
        max_checkpoint_bytes: c.len() as u64,
        max_working_bytes: (m.len() + c.len() + 8192) as u64,
    };
    assert!(matches!(
        Bundle::open(
            NoIo,
            id(&b),
            Caps {
                max_bundle_bytes: 0,
                ..exact
            }
        ),
        Err(BundleError::Limit(_))
    ));
    for field in 0..4 {
        let (s, mut w, input) = saved(2);
        input.flush.set(true);
        let mut cap = exact;
        match field {
            0 => cap.max_bundle_bytes -= 1,
            1 => cap.max_manifest_bytes -= 1,
            2 => cap.max_checkpoint_bytes -= 1,
            _ => cap.max_working_bytes -= 1,
        }
        assert!(matches!(
            stage(&s, &mut w, NoIo, cap),
            Err(BundleError::Limit(_))
        ));
        input.flush.set(false);
        w.checkpoint(&s).unwrap();
    }
    let (s, mut w, _) = saved(2);
    stage(&s, &mut w, Cursor::new(Vec::new()), exact).unwrap();
    for fail in [0, 32 + m.len() + c.len() + 3, b.len() - 5, usize::MAX] {
        let (s, mut w, input) = saved(2);
        let prefix = input.data.borrow().get_ref().clone();
        let out = FaultIo::new();
        out.remaining.set(fail);
        out.flush.set(fail == usize::MAX);
        assert!(matches!(
            stage(&s, &mut w, out, caps()),
            Err(BundleError::Io(_))
        ));
        assert_eq!(input.data.borrow().get_ref(), &prefix);
        assert_eq!(input.data.borrow().position(), prefix.len() as u64);
        stage(&s, &mut w, Cursor::new(Vec::new()), caps()).unwrap();
    }
    let (s, mut w, input) = saved(2);
    input.data.borrow_mut().get_mut()[0] ^= 1;
    assert!(stage(&s, &mut w, Cursor::new(Vec::new()), caps()).is_err());
    assert!(w.checkpoint(&s).is_err());
    let (s, mut w, _) = saved(2);
    let out = FaultIo::new();
    out.remaining.set(32 + m.len() + c.len() + 3);
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
    assert!(w.checkpoint(&s).is_err());
}

#[test]
fn real_new_directory_save_recovers_after_original_journal_is_removed() {
    use std::{
        fs::{self, File},
        sync::atomic::{AtomicU64, Ordering},
    };
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let root = std::env::temp_dir().join(format!(
        "tritium-pv-bundle-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&root).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(root.clone());
    let work = root.join("work.tnpj");
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&work)
        .unwrap();
    let mut s = Session::new(plan(1), vec![tiny()]).unwrap();
    let mut w = Writer::new(file, &s, limits()).unwrap();
    for _ in 0..2 {
        recorded(&mut s, &mut w);
    }
    let directory = root.join("saved");
    let saved = publish(&s, &mut w, &directory, caps()).unwrap();
    let id = saved.bundle.id();
    let original = fs::read(directory.join("resume.tnpb")).unwrap();
    assert!(publish(&s, &mut w, &directory, caps()).is_err());
    assert_eq!(fs::read(directory.join("resume.tnpb")).unwrap(), original);
    drop(saved);
    drop(w);
    fs::remove_file(work).unwrap();
    let reopened = Bundle::open(
        File::open(directory.join("resume.tnpb")).unwrap(),
        id,
        caps(),
    )
    .unwrap();
    let (mut recovered, mut continued) = reopened
        .resume(vec![tiny()], Cursor::new(Vec::new()), limits())
        .unwrap();
    for _ in 2..4 {
        recorded(&mut recovered, &mut continued);
    }
    let (actual, summary) = continued.finish(&recovered).unwrap();
    let (_, expected, bytes, wanted) = run(vec![tiny()]);
    assert_eq!(actual.into_inner(), bytes);
    assert_eq!(summary, wanted);
    assert_eq!(
        recovered.checkpoint_bytes().unwrap(),
        expected.checkpoint_bytes().unwrap()
    );
}

#[derive(Clone)]
struct Bounded {
    inner: FaultIo,
    maximum_read: Rc<Cell<usize>>,
    maximum_write: Rc<Cell<usize>>,
}
impl Bounded {
    fn new() -> Self {
        Self {
            inner: FaultIo::new(),
            maximum_read: Rc::new(Cell::new(0)),
            maximum_write: Rc::new(Cell::new(0)),
        }
    }
}
impl Read for Bounded {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        assert!(b.len() <= 8192);
        self.maximum_read.set(self.maximum_read.get().max(b.len()));
        self.inner.read(b)
    }
}
impl Write for Bounded {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        assert!(b.len() <= 8192);
        self.maximum_write
            .set(self.maximum_write.get().max(b.len()));
        self.inner.write(b)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}
impl Seek for Bounded {
    fn seek(&mut self, p: SeekFrom) -> io::Result<u64> {
        self.inner.seek(p)
    }
}
#[test]
fn large_pv_prefix_checkpoint_and_resume_use_at_most_8192_byte_stream_windows() {
    let mut layout = tiny().layout();
    layout.cols = 16384;
    let parent = AdditiveTensor::new(layout, 1, vec![Trit::POS; 16384], vec![1.0]).unwrap();
    let mut s = Session::new(plan(1), vec![parent.clone()]).unwrap();
    let input = Bounded::new();
    let output = Bounded::new();
    let mut w = Writer::new(input.clone(), &s, limits()).unwrap();
    recorded(&mut s, &mut w);
    let bundle = stage(&s, &mut w, output.clone(), caps()).unwrap();
    assert!(
        bundle.manifest().point.checkpoint_bytes > 8192
            && bundle.manifest().point.journal_prefix_bytes > 8192
    );
    let continued = Bounded::new();
    let (r, _) = bundle
        .resume(vec![parent], continued.clone(), limits())
        .unwrap();
    assert_eq!(r.checkpoint_bytes().unwrap(), s.checkpoint_bytes().unwrap());
    assert_eq!(input.maximum_read.get(), 8192);
    assert_eq!(output.maximum_read.get(), 8192);
    assert_eq!(input.maximum_write.get(), 8192);
    assert_eq!(output.maximum_write.get(), 8192);
    assert_eq!(continued.maximum_write.get(), 8192);
}
