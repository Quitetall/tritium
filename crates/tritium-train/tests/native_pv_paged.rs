//! Real numerical/seek-backed paging fixtures; not physical/model qualification.
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::{Cell, RefCell},
    fs::{File, OpenOptions},
    io::{self, Cursor, Read, Seek, SeekFrom, Write},
    path::PathBuf,
    rc::Rc,
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicIsize, AtomicUsize, Ordering},
    },
};
use tritium_core::{PerTilePlan, Trit, UniformPlan};
use tritium_format::AdditiveTensor;
use tritium_schema::*;
use tritium_train::pv_tuning::{
    NativePvJournalLimits, NativePvJournalWriter, NativePvPagedError, NativePvPagedJournalReplay,
    NativePvPagedJournalResume, NativePvPagedJournalWriter as PagedWriter,
    NativePvPagedLimits as Limits, NativePvPagedRecordingError as RecordingError,
    NativePvPagedRecordingLimits as RecordingLimits, NativePvPagedReplayLimits,
    NativePvPagedSession as Paged, NativePvRefinementSession as Resident, NativePvResumeTail,
    replay_native_pv_journal_with_state, replay_native_pv_paged_journal,
    resume_native_pv_paged_journal,
};
#[path = "native_pv_paged/bundle.rs"]
mod bundle;
#[path = "native_pv_paged/publication.rs"]
mod publication;
#[path = "native_pv_paged/recording.rs"]
mod recording;
#[path = "native_pv_paged/replay.rs"]
mod replay;
#[path = "native_pv_paged/runtime.rs"]
#[cfg(feature = "native-cpu")]
mod runtime;
#[path = "native_pv_paged/visit.rs"]
mod visit;

static SERIAL: Mutex<()> = Mutex::new(());
struct Allocator;
static TRACK: AtomicBool = AtomicBool::new(false);
static LIVE: AtomicIsize = AtomicIsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
fn allocated(size: usize) {
    if TRACK.load(Ordering::Relaxed) {
        let live = LIVE.fetch_add(size as isize, Ordering::Relaxed) + size as isize;
        PEAK.fetch_max(live.max(0) as usize, Ordering::Relaxed);
    }
}
// SAFETY: all requests forward unchanged to System; counters do not allocate.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        // SAFETY: GlobalAlloc supplies a valid layout.
        let ptr = unsafe { System.alloc(l) };
        if !ptr.is_null() {
            allocated(l.size());
        }
        ptr
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        if TRACK.load(Ordering::Relaxed) {
            LIVE.fetch_sub(l.size() as isize, Ordering::Relaxed);
        }
        // SAFETY: caller supplies the original allocation and layout.
        unsafe { System.dealloc(p, l) };
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        // SAFETY: caller supplies a live pointer/layout and valid new size.
        let ptr = unsafe { System.realloc(p, l, n) };
        if !ptr.is_null() {
            if TRACK.load(Ordering::Relaxed) {
                LIVE.fetch_sub(l.size() as isize, Ordering::Relaxed);
            }
            allocated(n);
        }
        ptr
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;
fn measured<T>(f: impl FnOnce() -> T) -> (T, usize, isize) {
    LIVE.store(0, Ordering::Relaxed);
    PEAK.store(0, Ordering::Relaxed);
    TRACK.store(true, Ordering::Relaxed);
    let result = f();
    TRACK.store(false, Ordering::Relaxed);
    (
        result,
        PEAK.load(Ordering::Relaxed),
        LIVE.load(Ordering::Relaxed),
    )
}
fn plan(n: usize) -> NativeRefinementPlan {
    let adam = NativeAdamWRecipe {
        lr: 0.01,
        beta1: 0.9,
        beta2: 0.99,
        eps: 1e-8,
        weight_decay: 0.01,
    };
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
        owners: (0..n).map(|i| format!("owner-{i:03}")).collect(),
        recipe: NativeRecoveryRecipe::AlternatingPv {
            continuous_optimizer: adam,
            code_optimizer: NativeAdamWRecipe { lr: 0.6, ..adam },
            max_code_change_fraction: 0.03125,
            max_relative_code_change: Some(0.9),
        },
        objective: [5; 32],
        teacher_source: None,
        training: (0..4)
            .map(|i| NativeRefinementWindow {
                content: [10 + i; 32],
                weight: if i == 0 { 2.0 } else { 0.3 },
            })
            .collect(),
        validation: vec![NativeRefinementWindow {
            content: [20; 32],
            weight: 1.0,
        }],
        limits: NativeRefinementLimits {
            max_state_bytes: 1 << 30,
            max_workspace_bytes: 1 << 30,
            max_checkpoint_bytes: 1 << 30,
            max_package_bytes: 1 << 30,
        },
    }
}
fn limits() -> Limits {
    Limits {
        max_resident_payload_bytes: 1 << 28,
    }
}
fn tensor(law: usize, codec: PlaneCodec, basis: Basis, mode: usize, rows: u64) -> AdditiveTensor {
    let law = ADMITTED_LAWS[law];
    let layout = AdditiveLayout {
        rows,
        cols: if basis == Basis::Identity { 260 } else { 512 },
        tile: 256,
        group: 128,
        max_planes: law.max_planes,
        allocation: if mode == 0 {
            PlaneAllocation::Uniform
        } else {
            PlaneAllocation::PerTile
        },
        codec,
        law: law.law,
        basis,
        transport: Transport::JointHuffman { block: 128 },
    };
    let counts: Vec<_> = (0..rows * 2)
        .map(|i| {
            if mode == 2 {
                0
            } else {
                [0, 1, law.max_planes, 2.min(law.max_planes)][i as usize % 4]
            }
        })
        .collect();
    let (t, s) = if mode == 0 {
        let p = UniformPlan::new(layout, law.max_planes).unwrap();
        (p.trit_count(), p.scale_count())
    } else {
        let p = PerTilePlan::new(layout, &counts).unwrap();
        (p.trit_count(), p.scale_count())
    };
    let trits = (0..t)
        .map(|i| [Trit::ZERO, Trit::NEG, Trit::POS, Trit::NEG][i % 4])
        .collect();
    let scales = vec![0.5; s];
    if mode == 0 {
        AdditiveTensor::new(layout, law.max_planes, trits, scales).unwrap()
    } else {
        AdditiveTensor::new_per_tile(layout, counts, trits, scales).unwrap()
    }
}
fn grad(i: usize, w: &AdditiveTensor) -> (Vec<f32>, Vec<f32>) {
    let a = if w.plane_count() == 0 {
        0
    } else {
        w.view().plane_trits(0).unwrap().len()
    };
    (
        (0..w.scales().len())
            .map(|n| ((n + i) % 5) as f32 * 0.125 - 0.25)
            .collect(),
        (0..a)
            .map(|n| ((n + i) % 7) as f32 * 0.125 - 0.375)
            .collect(),
    )
}
fn parents(
    weights: &[AdditiveTensor],
) -> impl FnMut(usize, &str) -> Result<AdditiveTensor, &'static str> + '_ {
    |i, _| Ok(weights[i].clone())
}
fn output() -> Cursor<Vec<u8>> {
    Cursor::new(Vec::new())
}

#[test]
fn all_profiles_match_resident_checkpoints_statistics_and_every_reopen_cursor() {
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
                    domain: 7,
                },
            ] {
                let p = plan(3);
                let weights: Vec<_> = (0..3).map(|i| tensor(law, codec, basis, i, 2)).collect();
                let mut resident = Resident::new(p.clone(), weights.clone()).unwrap();
                let journal_limits = NativePvJournalLimits {
                    max_journal_bytes: 1 << 30,
                    max_working_bytes: 1 << 30,
                };
                let mut journal =
                    NativePvJournalWriter::new(output(), &resident, journal_limits).unwrap();
                let mut paged =
                    Paged::initialize(p.clone(), limits(), parents(&weights), output()).unwrap();
                assert_eq!(paged.size_ledger().logical, resident.size_ledger());
                for cursor in 0..=4 {
                    let bytes = paged.into_checkpoint().into_inner();
                    assert_eq!(bytes, resident.checkpoint_bytes().unwrap());
                    assert_eq!(
                        Resident::resume(p.clone(), weights.clone(), &bytes)
                            .unwrap()
                            .completed_updates(),
                        cursor
                    );
                    paged =
                        Paged::reopen(p.clone(), limits(), parents(&weights), Cursor::new(bytes))
                            .unwrap();
                    assert_eq!(paged.completed_updates(), cursor);
                    let owners = paged
                        .visit_weights(parents(&weights), |i, name, current| {
                            assert_eq!(name, p.owners[i]);
                            assert_eq!(current, resident.weights().nth(i).unwrap());
                            Ok(())
                        })
                        .unwrap();
                    assert_eq!(owners.len(), weights.len());
                    if cursor == 4 {
                        break;
                    }
                    let old: Vec<_> = resident.weights().cloned().collect();
                    let grads: Vec<_> = old
                        .iter()
                        .enumerate()
                        .map(|(i, w)| grad(i + cursor as usize, w))
                        .collect();
                    let s: Vec<_> = grads.iter().map(|(s, _)| s.as_slice()).collect();
                    let w: Vec<_> = grads.iter().map(|(_, w)| w.as_slice()).collect();
                    let expected = resident.step_recorded(&s, &w, &mut journal).unwrap();
                    let (next, observed, staged) = paged
                        .step_staged(
                            output(),
                            parents(&weights),
                            |i, name, current| {
                                assert_eq!(name, p.owners[i]);
                                assert_eq!(current, &old[i]);
                                Ok(grads[i].clone())
                            },
                            |i, _, candidate| {
                                assert_eq!(candidate, resident.weights().nth(i).unwrap());
                                Ok(())
                            },
                        )
                        .unwrap();
                    assert_eq!(expected, observed);
                    assert_eq!(staged.len(), 3);
                    assert_eq!(paged.completed_updates(), cursor);
                    paged = next;
                }
                assert!(
                    paged
                        .step_staged(
                            output(),
                            parents(&weights),
                            |i, _, w| Ok(grad(i, w)),
                            |_, _, _| Ok(())
                        )
                        .is_err()
                );
                let (journal, summary) = journal.finish(&resident).unwrap();
                let bytes = paged.into_checkpoint().into_inner();
                let (replayed, _) = replay_native_pv_journal_with_state(
                    p.clone(),
                    weights.clone(),
                    journal,
                    summary.journal(),
                    &bytes,
                    journal_limits,
                )
                .unwrap();
                assert_eq!(replayed.checkpoint_bytes().unwrap(), bytes);
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 36);
}

#[derive(Clone)]
struct Shared {
    inner: Rc<RefCell<Cursor<Vec<u8>>>>,
    max: Rc<Cell<usize>>,
    fail_write: Rc<Cell<Option<u64>>>,
    fail_flush: Rc<Cell<bool>>,
    corrupt: Rc<Cell<bool>>,
    fail_read: Rc<Cell<bool>>,
    fail_seek: Rc<Cell<bool>>,
}
impl Shared {
    fn new(bytes: Vec<u8>) -> Self {
        Self {
            inner: Rc::new(RefCell::new(Cursor::new(bytes))),
            max: Rc::new(Cell::new(0)),
            fail_write: Rc::new(Cell::new(None)),
            fail_flush: Rc::new(Cell::new(false)),
            corrupt: Rc::new(Cell::new(false)),
            fail_read: Rc::new(Cell::new(false)),
            fail_seek: Rc::new(Cell::new(false)),
        }
    }
    fn request(&self, n: usize) {
        assert!(n <= 8192, "unbounded I/O {n}");
        self.max.set(self.max.get().max(n));
    }
    fn contents(&self) -> Vec<u8> {
        self.inner.borrow().get_ref().clone()
    }
}
impl Read for Shared {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        self.request(b.len());
        if self.fail_read.get() {
            return Err(io::Error::other("injected read"));
        }
        self.inner.borrow_mut().read(b)
    }
}
impl Seek for Shared {
    fn seek(&mut self, s: SeekFrom) -> io::Result<u64> {
        if self.fail_seek.get() {
            return Err(io::Error::other("injected seek"));
        }
        self.inner.borrow_mut().seek(s)
    }
}
impl Write for Shared {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        self.request(b.len());
        if self
            .fail_write
            .get()
            .is_some_and(|limit| self.inner.borrow().position() + b.len() as u64 > limit)
        {
            return Err(io::Error::other("injected write"));
        }
        let start = self.inner.borrow().position() as usize;
        let n = self.inner.borrow_mut().write(b)?;
        if self.corrupt.get() && start >= 85 && n > 0 {
            self.inner.borrow_mut().get_mut()[start] ^= 1;
        }
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.fail_flush.get() {
            Err(io::Error::other("injected flush"))
        } else {
            Ok(())
        }
    }
}
struct Resource(Rc<Cell<usize>>);
impl Drop for Resource {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

#[test]
fn late_failure_unwind_wrong_parents_and_io_never_advance_old_or_leak_staged_resources() {
    let _serial = SERIAL.lock().unwrap();
    let p = plan(3);
    let weights: Vec<_> = (0..3)
        .map(|_| tensor(0, PlaneCodec::D2, Basis::Identity, 0, 2))
        .collect();
    let source = Shared::new(Vec::new());
    let mut paged =
        Paged::initialize(p.clone(), limits(), parents(&weights), source.clone()).unwrap();
    let before = source.contents();
    for mode in 0..10 {
        let target = Shared::new(Vec::new());
        if mode == 3 {
            target.fail_write.set(Some(before.len() as u64 / 2));
        }
        if mode == 4 {
            target.fail_flush.set(true);
        }
        if mode == 5 {
            target.corrupt.set(true);
        }
        let drops = Rc::new(Cell::new(0));
        let mut made = 0;
        let result = paged.step_staged(
            target,
            |i, _| {
                if mode == 0 && i == 2 {
                    Err("original parent failure")
                } else {
                    Ok(weights[i].clone())
                }
            },
            |i, _, w| {
                if mode == 7 && i == 0 {
                    source.fail_read.set(true);
                }
                if mode == 1 && i == 2 {
                    Err("gradient failure")
                } else {
                    let mut gradients = grad(i, w);
                    if mode == 9 && i == 2 {
                        gradients.1[0] = f32::NAN;
                    }
                    Ok(gradients)
                }
            },
            |i, _, _| {
                if mode == 8 && i == 2 {
                    source.fail_seek.set(true);
                }
                if mode == 2 && i == 2 {
                    Err("stage failure")
                } else {
                    made += 1;
                    Ok(Resource(drops.clone()))
                }
            },
        );
        source.fail_read.set(false);
        source.fail_seek.set(false);
        if mode == 6 {
            let (next, _, resources) = result.unwrap();
            assert_eq!(next.completed_updates(), 1);
            drop(resources);
        } else {
            let error = result.err().expect("fault must reject");
            match mode {
                0 => assert!(matches!(
                    error,
                    NativePvPagedError::Provider("original parent failure")
                )),
                1 => assert!(matches!(
                    error,
                    NativePvPagedError::Provider("gradient failure")
                )),
                2 => assert!(matches!(
                    error,
                    NativePvPagedError::Provider("stage failure")
                )),
                3 | 4 | 7 | 8 => assert!(matches!(error, NativePvPagedError::Io(_))),
                5 | 9 => assert!(matches!(error, NativePvPagedError::State(_))),
                _ => unreachable!(),
            }
        }
        assert_eq!(drops.get(), made);
        assert_eq!(source.contents(), before);
        assert_eq!(paged.completed_updates(), 0);
    }
    let drops = Rc::new(Cell::new(0));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = paged.step_staged(
            output(),
            parents(&weights),
            |i, _, w| Ok(grad(i, w)),
            |i, _, _| {
                if i == 2 {
                    panic!("injected stage unwind");
                }
                Ok(Resource(drops.clone()))
            },
        );
    }));
    assert!(result.is_err());
    assert_eq!(drops.get(), 2);
    assert_eq!(source.contents(), before);
    let mut changed = weights.clone();
    let mut scales = changed[2].scales().to_vec();
    scales[0] = 0.25;
    changed[2].replace_scales(scales).unwrap();
    assert!(
        paged
            .step_staged(
                output(),
                parents(&changed),
                |i, _, w| Ok(grad(i, w)),
                |_, _, _| Ok(())
            )
            .is_err()
    );
    assert!(Paged::reopen(p.clone(), limits(), parents(&changed), source.clone()).is_err());
    assert!(
        paged
            .step_staged(
                Cursor::new(vec![1]),
                parents(&weights),
                |i, _, w| Ok(grad(i, w)),
                |_, _, _| Ok(())
            )
            .is_err()
    );
    for n in [0, 1, 84, before.len() - 1] {
        assert!(
            Paged::reopen(
                p.clone(),
                limits(),
                parents(&weights),
                Cursor::new(before[..n].to_vec())
            )
            .is_err()
        );
    }
    source.inner.borrow_mut().get_mut()[90] ^= 1;
    let error = paged
        .step_staged(
            output(),
            parents(&weights),
            |i, _, w| Ok(grad(i, w)),
            |_, _, _| Ok(()),
        )
        .unwrap_err();
    assert!(matches!(error, NativePvPagedError::State(_)));
    source.inner.borrow_mut().get_mut().copy_from_slice(&before);
    assert_eq!(source.contents(), before);
}

#[test]
fn checksum_valid_temporary_source_substitution_is_detected_during_traversal() {
    let _serial = SERIAL.lock().unwrap();
    let p = plan(2);
    let weights: Vec<_> = (0..2)
        .map(|_| tensor(0, PlaneCodec::D2, Basis::Identity, 0, 2))
        .collect();
    let mut resident = Resident::new(p.clone(), weights.clone()).unwrap();
    let grads: Vec<_> = weights
        .iter()
        .enumerate()
        .map(|(i, w)| grad(i, w))
        .collect();
    resident
        .step(
            &grads.iter().map(|(s, _)| s.as_slice()).collect::<Vec<_>>(),
            &grads.iter().map(|(_, w)| w.as_slice()).collect::<Vec<_>>(),
        )
        .unwrap();
    let before = resident.checkpoint_bytes().unwrap();
    let source = Shared::new(before.clone());
    let mut paged = Paged::reopen(p.clone(), limits(), parents(&weights), source.clone()).unwrap();
    let owner = &weights[0];
    let a = owner.view().plane_trits(0).unwrap().len();
    let owner_len = 24 + owner.trits().len() + 10 * owner.scales().len() + 8 * a;
    let offset = 85 + owner_len + 8 + weights[1].trits().len() + 8;
    let original = before[offset];
    let result = paged.step_staged(
        output(),
        |i, _| {
            if i == 1 {
                source.inner.borrow_mut().get_mut()[offset] ^= 1;
            }
            Ok::<_, &str>(weights[i].clone())
        },
        |i, _, w| {
            if i == 1 {
                source.inner.borrow_mut().get_mut()[offset] = original;
            }
            Ok(grad(i, w))
        },
        |_, _, _| Ok(()),
    );
    assert!(result.unwrap_err().to_string().contains("body"));
    assert_eq!(source.contents(), before);
    assert_eq!(paged.completed_updates(), 1);
}

struct Files(PathBuf);

#[test]
fn actual_packed_resources_stage_atomically_and_late_budget_failure_releases_every_new_lease() {
    use std::sync::Arc;
    use tritium_cpu::CpuBackend;
    use tritium_runtime::TensorPayloadBudget;
    use tritium_spec::{DeviceBuffer, TensorView, TernaryBackend};
    let _serial = SERIAL.lock().unwrap();
    let cpu = CpuBackend::new();
    let p = plan(3);
    let weights: Vec<_> = (0..3)
        .map(|_| {
            let w = tensor(0, PlaneCodec::D2, Basis::Identity, 0, 2);
            AdditiveTensor::new(
                AdditiveLayout {
                    transport: Transport::Raw,
                    ..w.layout()
                },
                w.plane_count(),
                w.trits().to_vec(),
                w.scales().to_vec(),
            )
            .unwrap()
        })
        .collect();
    let old: Vec<Arc<dyn DeviceBuffer>> = weights
        .iter()
        .map(|w| Arc::from(cpu.upload_tensor(TensorView::Additive(w.view())).unwrap()))
        .collect();
    let sizes: Vec<_> = weights
        .iter()
        .map(|w| {
            cpu.tensor_caps(TensorView::Additive(w.view()))
                .unwrap()
                .unwrap()
                .payload_bytes
        })
        .collect();
    let mut paged = Paged::initialize(p.clone(), limits(), parents(&weights), output()).unwrap();
    let digest = paged.checkpoint_digest();
    let short = TensorPayloadBudget::new(sizes.iter().sum::<u64>() - 1);
    let error = paged
        .step_staged(
            output(),
            |i, _| Ok(weights[i].clone()),
            |i, _, w| Ok(grad(i, w)),
            |i, _, w| cpu.snapshot_additive_weights(&old[i], w.view(), &short),
        )
        .err()
        .expect("late packed budget must refuse");
    assert!(matches!(error, NativePvPagedError::Provider(_)));
    assert_eq!(short.used_bytes(), 0);
    assert_eq!(paged.checkpoint_digest(), digest);
    assert_eq!(paged.completed_updates(), 0);
    for (handle, w) in old.iter().zip(&weights) {
        assert!(cpu.additive_equals_view(&**handle, w.view()).unwrap());
    }
    let budget = TensorPayloadBudget::new(sizes.iter().sum());
    let (next, observed, handles) = paged
        .step_staged(
            output(),
            |i, _| Ok(weights[i].clone()),
            |i, _, w| Ok(grad(i, w)),
            |i, _, w| cpu.snapshot_additive_weights(&old[i], w.view(), &budget),
        )
        .unwrap();
    assert!(observed.owners.iter().any(|s| s.changed_units > 0));
    let mut reference = Resident::new(p.clone(), weights.clone()).unwrap();
    let g: Vec<_> = weights
        .iter()
        .enumerate()
        .map(|(i, w)| grad(i, w))
        .collect();
    reference
        .step(
            &g.iter().map(|(s, _)| s.as_slice()).collect::<Vec<_>>(),
            &g.iter().map(|(_, w)| w.as_slice()).collect::<Vec<_>>(),
        )
        .unwrap();
    assert_eq!(
        next.into_checkpoint().into_inner(),
        reference.checkpoint_bytes().unwrap()
    );
    for ((new, old), w) in handles.iter().zip(&old).zip(reference.weights()) {
        assert!(cpu.additive_equals_view(&**new, w.view()).unwrap());
        assert!(!Arc::ptr_eq(new, old));
    }
    assert_eq!(budget.used_bytes(), sizes.iter().sum::<u64>());
    drop(handles);
    assert_eq!(budget.used_bytes(), 0);
}

impl Files {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "tritium-pv-paged-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        Self(root)
    }
    fn create(&self, name: &str) -> File {
        OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(self.0.join(name))
            .unwrap()
    }
}
impl Drop for Files {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
fn big() -> AdditiveTensor {
    let mut w = tensor(3, PlaneCodec::D2, Basis::Identity, 0, 1024);
    // Keep valid admitted geometry; no prebuilt all-owner vector is captured.
    let mut values = w.scales().to_vec();
    values.fill(0.5);
    w.replace_scales(values).unwrap();
    w
}
#[test]
fn file_backed_residency_tracks_largest_owner_not_owner_count_and_caps_precede_engine_allocations()
{
    let _serial = SERIAL.lock().unwrap();
    rayon::join(|| (), || ());
    let files = Files::new();
    let mut first_peak = 0;
    for owners in [1, 8] {
        let p = plan(owners);
        let destination = files.create(&format!("init-{owners}"));
        let (result, peak, retained) = measured(|| {
            Paged::initialize(
                p.clone(),
                limits(),
                |_, _| Ok::<_, &str>(big()),
                destination,
            )
        });
        let mut paged = result.unwrap();
        let ledger = paged.size_ledger();
        assert!(retained < 65536, "owner payload retained {retained}");
        if owners == 1 {
            first_peak = peak;
        } else {
            assert!(
                peak < first_peak + 65536,
                "owner-count residency grew {first_peak}->{peak}"
            );
            assert!(ledger.logical.state_payload_bytes > ledger.max_owner_payload_bytes);
        }
        let destination = files.create(&format!("step-{owners}"));
        let (result, step_peak, step_retained) = measured(|| {
            paged.step_staged(
                destination,
                |_, _| Ok::<_, &str>(big()),
                |i, _, w| Ok(grad(i, w)),
                |_, _, _| Ok(()),
            )
        });
        let (next, _, _) = result.unwrap();
        assert!(
            step_retained < 65536,
            "next session retained owner payload {step_retained}"
        );
        assert!(
            step_peak as u64 <= ledger.max_owner_payload_bytes + 65536,
            "peak {step_peak} exceeds ledger {ledger:?}"
        );
        let source = next.into_checkpoint();
        let reopened =
            Paged::reopen(p.clone(), limits(), |_, _| Ok::<_, &str>(big()), source).unwrap();
        assert_eq!(reopened.completed_updates(), 1);
        let cap = Limits {
            max_resident_payload_bytes: ledger.max_owner_payload_bytes,
        };
        let exact = Paged::initialize(
            p.clone(),
            cap,
            |_, _| Ok::<_, &str>(big()),
            files.create(&format!("exact-{owners}")),
        )
        .unwrap();
        assert_eq!(exact.size_ledger(), ledger);
        let (error, rejected_peak, _) = measured(|| {
            Paged::initialize(
                p,
                Limits {
                    max_resident_payload_bytes: cap.max_resident_payload_bytes - 1,
                },
                |_, _| Ok::<_, &str>(big()),
                files.create(&format!("short-{owners}")),
            )
            .unwrap_err()
        });
        assert!(error.to_string().contains("cap"));
        assert!(
            rejected_peak < 1_000_000,
            "rejected cap allocated owner moments: {rejected_peak}"
        );
        eprintln!(
            "OWNER_PAGED owners={owners} init_peak_live_requests={peak} retained_requests={retained} step_peak_live_requests={step_peak} step_retained_requests={step_retained} max_owner_payload={} total_logical_state={} checkpoint_bytes={}",
            ledger.max_owner_payload_bytes,
            ledger.logical.state_payload_bytes,
            ledger.logical.checkpoint_bytes
        );
    }
}
