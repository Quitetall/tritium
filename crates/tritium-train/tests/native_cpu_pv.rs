#![cfg(feature = "native-cpu")]
//! Packed P/V transactions; synthetic fixtures, not source/model qualification.
use std::{
    cell::{Cell, RefCell},
    io::{self, Cursor, Seek, SeekFrom, Write},
    rc::Rc,
    sync::Arc,
};
use tritium_cpu::CpuBackend;
use tritium_format::AdditiveTensor;
use tritium_runtime::TensorPayloadBudget;
use tritium_schema::*;
use tritium_spec::{BackendError, DeviceBuffer, TensorMatmul, TensorView, TernaryBackend};
use tritium_train::pv_tuning::{
    NativeCpuPvJournalStageError as Stage, NativeCpuPvSession as Bound,
    NativePvContribution as Use, NativePvJournalLimits as Limits, NativePvJournalWriter as Writer,
    NativePvRefinementSession as Session, NativePvStageError as Error,
    replay_native_pv_journal_with_state as replay,
};
#[path = "common/native_pv_consumer.rs"]
mod fixture;
#[path = "native_cpu_pv/paged.rs"]
mod paged;
use fixture::{active, manual, plan, tensor, values};
fn limits() -> Limits {
    Limits {
        max_journal_bytes: 10_000_000,
        max_working_bytes: 10_000_000,
    }
}
fn roots(weights: &[AdditiveTensor]) -> Vec<(String, Arc<dyn DeviceBuffer>)> {
    weights
        .iter()
        .enumerate()
        .map(|(i, w)| {
            (
                format!("owner-{i:04}"),
                Arc::from(
                    CpuBackend::new()
                        .upload_tensor(TensorView::Additive(w.view()))
                        .unwrap(),
                ),
            )
        })
        .collect()
}
fn packed_bytes(weights: &[AdditiveTensor]) -> u64 {
    weights
        .iter()
        .map(|w| {
            CpuBackend::new()
                .tensor_caps(TensorView::Additive(w.view()))
                .unwrap()
                .unwrap()
                .payload_bytes
        })
        .sum()
}
fn check(bound: &Bound) {
    for ((_, handle), weight) in bound.handles().zip(bound.session().weights()) {
        assert!(
            CpuBackend::new()
                .additive_equals_view(&**handle, weight.view())
                .unwrap()
        );
    }
}

#[test]
fn actual_packed_consumers_match_decoded_manual_journals_and_replay_across_profiles() {
    let mut profiles = 0;
    let mut moves = 0;
    for law in 0..ADMITTED_LAWS.len() {
        for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
            for basis in [
                Basis::Identity,
                Basis::Hadamard { block: 128 },
                Basis::SignedRht {
                    block: 128,
                    seed: 19,
                    domain: 7,
                },
            ] {
                for mode in 0..3 {
                    for group in [32, 128] {
                        let cols = if basis == Basis::Identity { 260 } else { 512 };
                        let parents = vec![tensor(law, codec, basis, mode, group, cols)];
                        let p = plan(1);
                        let budget = TensorPayloadBudget::new(3 * packed_bytes(&parents));
                        let mut bound = Bound::bind(
                            Session::new(p.clone(), parents.clone()).unwrap(),
                            roots(&parents),
                            budget.clone(),
                        )
                        .unwrap();
                        let mut direct = Session::new(p.clone(), parents.clone()).unwrap();
                        let mut writer =
                            Writer::new(Cursor::new(Vec::new()), bound.session(), limits())
                                .unwrap();
                        let mut other =
                            Writer::new(Cursor::new(Vec::new()), &direct, limits()).unwrap();
                        let x = values(2 * cols as usize, 0);
                        let dy = values(6, 2);
                        let de = values(4 * cols as usize, 3);
                        let uses = [
                            Use::Matmul {
                                activations: &x,
                                output_gradients: &dy,
                                batch: 2,
                            },
                            Use::Embed {
                                ids: &[2, 0, 2, 1],
                                output_gradients: &de,
                            },
                            Use::Matmul {
                                activations: &x,
                                output_gradients: &dy,
                                batch: 2,
                            },
                        ];
                        let old = Arc::clone(bound.handles().next().unwrap().1);
                        for step in 0..2 {
                            let (s, a) = manual(direct.weights().next().unwrap(), &uses);
                            let expected = direct.step_recorded(&[&s], &[&a], &mut other).unwrap();
                            let actual = bound
                                .step_from_consumers_recorded(&[&uses], &mut writer)
                                .unwrap();
                            moves += actual.owners[0].changed_units;
                            assert_eq!(actual, expected);
                            assert_eq!(
                                bound.session().checkpoint_bytes().unwrap(),
                                direct.checkpoint_bytes().unwrap()
                            );
                            check(&bound);
                            assert!(
                                CpuBackend::new()
                                    .additive_equals_view(&*old, parents[0].view())
                                    .unwrap()
                            );
                            if step == 0 {
                                let checkpoint = bound.session().checkpoint_bytes().unwrap();
                                let current: Vec<_> = bound.session().weights().cloned().collect();
                                let restored = Bound::bind(
                                    Session::resume(p.clone(), parents.clone(), &checkpoint)
                                        .unwrap(),
                                    roots(&current),
                                    budget.clone(),
                                )
                                .unwrap();
                                bound = restored;
                            }
                        }
                        let cp = bound.session().checkpoint_bytes().unwrap();
                        let (out, summary) = writer.finish(bound.session()).unwrap();
                        let (expected, _) = other.finish(&direct).unwrap();
                        assert_eq!(out.get_ref(), expected.get_ref());
                        let (replayed, proof) = replay(
                            p.clone(),
                            parents.clone(),
                            Cursor::new(out.into_inner()),
                            summary.journal(),
                            &cp,
                            limits(),
                        )
                        .unwrap();
                        proof.verify_session(bound.session()).unwrap();
                        let current: Vec<_> = replayed.weights().cloned().collect();
                        let resumed = Bound::bind(
                            Session::resume(p, parents, &cp).unwrap(),
                            roots(&current),
                            TensorPayloadBudget::new(packed_bytes(&current)),
                        )
                        .unwrap();
                        check(&resumed);
                        assert_eq!(resumed.session().checkpoint_bytes().unwrap(), cp);
                        drop(bound);
                        drop(old);
                        assert_eq!(budget.used_bytes(), 0);
                        profiles += 1;
                    }
                }
            }
        }
    }
    assert_eq!(profiles, 216);
    assert!(moves > 0);
}

#[derive(Clone, Default)]
struct Fault {
    data: Rc<RefCell<Cursor<Vec<u8>>>>,
    mode: Rc<Cell<u8>>,
}
impl Write for Fault {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        match self.mode.get() {
            1 => return Err(io::Error::other("injected PV write")),
            3 => panic!("injected PV write unwind"),
            _ => {}
        }
        self.data.borrow_mut().write(bytes)
    }
    fn flush(&mut self) -> io::Result<()> {
        match self.mode.get() {
            2 => Err(io::Error::other("injected PV flush")),
            4 => panic!("injected PV flush unwind"),
            _ => Ok(()),
        }
    }
}
impl Seek for Fault {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        self.data.borrow_mut().seek(from)
    }
}

#[test]
fn late_owner_budget_failure_writes_nothing_and_same_writer_retries_atomically() {
    let parent = tensor(1, PlaneCodec::B3, Basis::Identity, 0, 32, 4);
    let parents = vec![parent.clone(), parent];
    let payload = packed_bytes(&parents);
    let budget = TensorPayloadBudget::new(2 * payload);
    let mut bound = Bound::bind(
        Session::new(plan(2), parents.clone()).unwrap(),
        roots(&parents),
        budget.clone(),
    )
    .unwrap();
    let old: Vec<_> = bound.handles().map(|(_, h)| Arc::clone(h)).collect();
    let before = bound.session().checkpoint_bytes().unwrap();
    let stream = Fault::default();
    let mut writer = Writer::new(stream.clone(), bound.session(), limits()).unwrap();
    let header = stream.data.borrow().get_ref().clone();
    let invalid = [Use::Embed {
        ids: &[3],
        output_gradients: &[0.0; 4],
    }];
    assert!(matches!(
        bound.step_from_consumers_recorded(&[&[], &invalid], &mut writer),
        Err(Error::Update(_))
    ));
    assert_eq!(stream.data.borrow().get_ref(), &header);
    assert_eq!(bound.session().checkpoint_bytes().unwrap(), before);
    // Leaves room for only the first next-owner snapshot, not the second.
    let blocker = budget.reserve_storage(payload / 2).unwrap();
    let s: Vec<_> = parents
        .iter()
        .map(|w| vec![0.1; w.scales().len()])
        .collect();
    let a: Vec<_> = parents.iter().map(|w| vec![0.5; active(w)]).collect();
    let ss: Vec<_> = s.iter().map(Vec::as_slice).collect();
    let aa: Vec<_> = a.iter().map(Vec::as_slice).collect();
    let mut nonfinite = s.clone();
    nonfinite[1][0] = f32::NAN;
    assert!(matches!(
        bound.step_recorded(
            &nonfinite.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            &aa,
            &mut writer
        ),
        Err(Error::Update(_))
    ));
    assert_eq!(stream.data.borrow().get_ref(), &header);
    assert_eq!(bound.session().checkpoint_bytes().unwrap(), before);
    assert!(matches!(
        bound.step_recorded(&ss, &aa, &mut writer),
        Err(Error::Stage(Stage::Packed(
            BackendError::TensorPayloadBudgetExceeded { .. }
        )))
    ));
    assert_eq!(stream.data.borrow().get_ref(), &header);
    assert_eq!(bound.session().checkpoint_bytes().unwrap(), before);
    assert_eq!(budget.used_bytes(), payload + payload / 2);
    for ((_, h), old) in bound.handles().zip(&old) {
        assert!(Arc::ptr_eq(h, old));
    }
    drop(blocker);
    bound.step_recorded(&ss, &aa, &mut writer).unwrap();
    check(&bound);
    for (h, w) in old.iter().zip(&parents) {
        assert!(
            CpuBackend::new()
                .additive_equals_view(&**h, w.view())
                .unwrap()
        );
    }
    drop(old);
    assert_eq!(budget.used_bytes(), payload);
    bound.step_recorded(&ss, &aa, &mut writer).unwrap();
    let (_, summary) = writer.finish(bound.session()).unwrap();
    assert_eq!(summary.updates(), 2);
}

#[test]
fn journal_write_flush_failure_and_unwind_release_all_staged_versions_and_poison_writer() {
    for mode in 1..=4 {
        let parents = vec![tensor(1, PlaneCodec::D2, Basis::Identity, 0, 32, 4)];
        let payload = packed_bytes(&parents);
        let budget = TensorPayloadBudget::new(2 * payload);
        let mut bound = Bound::bind(
            Session::new(plan(1), parents.clone()).unwrap(),
            roots(&parents),
            budget.clone(),
        )
        .unwrap();
        let before = bound.session().checkpoint_bytes().unwrap();
        let old = Arc::clone(bound.handles().next().unwrap().1);
        let stream = Fault::default();
        let mut writer = Writer::new(stream.clone(), bound.session(), limits()).unwrap();
        stream.mode.set(mode);
        let s = vec![0.1; parents[0].scales().len()];
        let a = vec![0.5; active(&parents[0])];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            bound.step_recorded(&[&s], &[&a], &mut writer)
        }));
        if mode <= 2 {
            assert!(matches!(
                result.unwrap(),
                Err(Error::Stage(Stage::Journal(_)))
            ));
        } else {
            assert!(result.is_err());
        }
        assert_eq!(bound.session().checkpoint_bytes().unwrap(), before);
        assert_eq!(budget.used_bytes(), payload);
        assert!(Arc::ptr_eq(bound.handles().next().unwrap().1, &old));
        check(&bound);
        stream.mode.set(0);
        assert!(writer.checkpoint(bound.session()).is_err());
        assert!(writer.finish(bound.session()).is_err());
        drop(bound);
        drop(old);
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn binding_rejects_wrong_names_scales_codes_and_stale_checkpoint_handles_without_leaks() {
    let parent = tensor(1, PlaneCodec::D2, Basis::Identity, 0, 32, 4);
    for case in 0..4 {
        let mut owners = roots(std::slice::from_ref(&parent));
        if case == 0 {
            owners[0].0 = "wrong-owner".into();
        }
        if case == 1 {
            let mut changed = parent.clone();
            changed
                .replace_scales(vec![0.25; parent.scales().len()])
                .unwrap();
            owners = roots(&[changed]);
        }
        if case == 2 {
            let mut changed = parent.clone();
            changed
                .replace_trits(
                    parent
                        .trits()
                        .iter()
                        .map(|t| tritium_core::Trit::from_i8(-t.get()).unwrap())
                        .collect(),
                )
                .unwrap();
            owners = roots(&[changed]);
        }
        if case == 3 {
            owners.clear();
        }
        let budget = TensorPayloadBudget::new(100_000);
        assert!(
            Bound::bind(
                Session::new(plan(1), vec![parent.clone()]).unwrap(),
                owners,
                budget.clone()
            )
            .is_err()
        );
        assert_eq!(budget.used_bytes(), 0);
    }
    let parents = vec![parent.clone(), parent.clone()];
    let original_roots = roots(&parents);
    let budget = TensorPayloadBudget::new(packed_bytes(&parents) / 2);
    assert!(matches!(
        Bound::bind(
            Session::new(plan(2), parents.clone()).unwrap(),
            original_roots.clone(),
            budget.clone()
        ),
        Err(Error::Stage(
            BackendError::TensorPayloadBudgetExceeded { .. }
        ))
    ));
    assert_eq!(budget.used_bytes(), 0);
    for ((_, h), w) in original_roots.iter().zip(&parents) {
        assert!(
            CpuBackend::new()
                .additive_equals_view(&**h, w.view())
                .unwrap()
        );
    }
    let mut changed = parent.clone();
    changed
        .replace_scales(vec![0.25; parent.scales().len()])
        .unwrap();
    let wrong = roots(&[parent.clone(), changed]);
    let zero_budget = TensorPayloadBudget::new(0);
    assert!(
        matches!(Bound::bind(Session::new(plan(2), parents).unwrap(), wrong, zero_budget.clone()),
        Err(Error::Stage(BackendError::InvalidInput(message))) if message.contains("current common state"))
    );
    assert_eq!(zero_budget.used_bytes(), 0);
    let p = plan(1);
    let mut state = Session::new(p.clone(), vec![parent.clone()]).unwrap();
    state
        .step(
            &[&vec![0.0; parent.scales().len()]],
            &[&vec![1.0; active(&parent)]],
        )
        .unwrap();
    assert_ne!(state.weights().next().unwrap().trits(), parent.trits());
    let cp = state.checkpoint_bytes().unwrap();
    let budget = TensorPayloadBudget::new(100_000);
    assert!(
        Bound::bind(
            Session::resume(p, vec![parent.clone()], &cp).unwrap(),
            roots(&[parent]),
            budget.clone()
        )
        .is_err()
    );
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn actual_packed_tied_consumer_loss_decreases_and_old_graph_stays_exact() {
    let parent = tensor(
        1,
        PlaneCodec::D2,
        Basis::SignedRht {
            block: 4,
            seed: 19,
            domain: 7,
        },
        0,
        32,
        4,
    );
    let budget = TensorPayloadBudget::new(3 * packed_bytes(std::slice::from_ref(&parent)));
    let mut bound = Bound::bind(
        Session::new(plan(1), vec![parent.clone()]).unwrap(),
        roots(&[parent]),
        budget,
    )
    .unwrap();
    let old = Arc::clone(bound.handles().next().unwrap().1);
    let x = values(8, 0);
    let forward = |handle: &dyn DeviceBuffer| {
        let cpu = CpuBackend::new();
        let mut y = vec![0.0; 6];
        let mut e = vec![0.0; 16];
        cpu.matmul(TensorMatmul {
            act: &x,
            tensor: handle,
            batch: 2,
            transformed_act: &mut [0.0; 8],
            out: &mut y,
        })
        .unwrap();
        cpu.embed_rows(handle, &[2, 0, 2, 1], &mut e).unwrap();
        (y, e)
    };
    let initial = forward(&*old);
    let loss = |v: &(Vec<f32>, Vec<f32>)| {
        v.0.iter()
            .chain(&v.1)
            .map(|&x| f64::from(x).powi(2) * 0.5)
            .sum::<f64>()
    };
    let mut moves = 0;
    for _ in 0..2 {
        let (dy, de) = forward(&**bound.handles().next().unwrap().1);
        let uses = [
            Use::Matmul {
                activations: &x,
                output_gradients: &dy,
                batch: 2,
            },
            Use::Embed {
                ids: &[2, 0, 2, 1],
                output_gradients: &de,
            },
        ];
        moves += bound.step_from_consumers(&[&uses]).unwrap().owners[0].changed_units;
    }
    assert!(moves > 0);
    let initial_loss = loss(&initial);
    let final_loss = loss(&forward(&**bound.handles().next().unwrap().1));
    assert!(final_loss < initial_loss);
    eprintln!(
        "NATIVE_PV_PACKED_TIED_LOSS initial={initial_loss} final={final_loss} changed_units={moves}"
    );
    assert_eq!(forward(&*old), initial);
    check(&bound);
}
