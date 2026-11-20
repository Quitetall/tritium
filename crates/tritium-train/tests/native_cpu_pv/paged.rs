//! Actual live packed controller with seek-backed optimizer state, not qualification.
use super::*;
use tritium_train::pv_tuning::{
    NativeCpuPvPagedError as PagedError, NativeCpuPvPagedSession as Paged,
    NativeCpuPvPagedStepLimits as StepLimits, NativePvPagedError, NativePvPagedLimits,
    NativePvPagedRecordingLimits, NativePvPagedSession,
};
fn recording() -> NativePvPagedRecordingLimits {
    NativePvPagedRecordingLimits {
        max_journal_bytes: 10_000_000,
        max_frame_bytes: 10_000_000,
        max_working_bytes: 16384,
    }
}
fn cap() -> StepLimits {
    StepLimits {
        recording: recording(),
        max_consumer_working_bytes: 10_000_000,
    }
}
fn out() -> Cursor<Vec<u8>> {
    Cursor::new(Vec::new())
}
fn bound(
    p: NativeRefinementPlan,
    weights: &[AdditiveTensor],
    budget: TensorPayloadBudget,
) -> Paged<Cursor<Vec<u8>>> {
    let state = NativePvPagedSession::initialize(
        p,
        NativePvPagedLimits {
            max_resident_payload_bytes: 10_000_000,
        },
        |i, _| Ok::<_, &str>(weights[i].clone()),
        out(),
    )
    .unwrap();
    Paged::bind(state, roots(weights), budget, |i, _| {
        Ok::<_, &str>(weights[i].clone())
    })
    .unwrap()
}
fn check_packed<R: std::io::Read + Seek>(paged: &Paged<R>, reference: &Session) {
    for ((_, handle), weight) in paged.handles().zip(reference.weights()) {
        assert!(
            CpuBackend::new()
                .additive_equals_view(&**handle, weight.view())
                .unwrap()
        );
    }
    assert_eq!(
        paged.session().checkpoint_digest(),
        *blake3::hash(&reference.checkpoint_bytes().unwrap()).as_bytes()
    );
}

#[test]
fn packed_paged_steps_match_manual_checkpoints_stats_and_standard_journal_across_profiles() {
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
                        let mut live = bound(p.clone(), &parents, budget.clone());
                        let mut direct = Session::new(p, parents.clone()).unwrap();
                        let mut journal = live.new_recording(out(), recording()).unwrap();
                        let mut other = Writer::new(out(), &direct, limits()).unwrap();
                        let old = live.handles().next().unwrap().1.clone();
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
                        for step in 0..2 {
                            let before = live.session().checkpoint_digest();
                            let (s, a) = manual(direct.weights().next().unwrap(), &uses);
                            let expected = direct.step_recorded(&[&s], &[&a], &mut other).unwrap();
                            let ready = live
                                .stage_from_consumers_recorded(
                                    (out(), out()),
                                    |i, _| Ok::<_, &str>(parents[i].clone()),
                                    &[&uses],
                                    cap(),
                                )
                                .unwrap();
                            assert_eq!(live.session().completed_updates(), step);
                            assert_eq!(live.session().checkpoint_digest(), before);
                            let (next, actual) = ready.commit(&mut live, &mut journal).unwrap();
                            assert_eq!(actual, expected);
                            moves += actual.owners[0].changed_units;
                            live = next;
                            check_packed(&live, &direct);
                            assert_eq!(
                                live.checkpoint(&mut journal).unwrap(),
                                other.checkpoint(&direct).unwrap().1
                            );
                            assert!(
                                CpuBackend::new()
                                    .additive_equals_view(&*old, parents[0].view())
                                    .unwrap()
                            );
                        }
                        let (journal, summary) = live.finish_recording(journal).unwrap();
                        let (other, other_summary) = other.finish(&direct).unwrap();
                        assert_eq!(journal.into_inner(), other.into_inner());
                        assert_eq!(summary, other_summary);
                        assert_eq!(
                            live.into_session().into_checkpoint().into_inner(),
                            direct.checkpoint_bytes().unwrap()
                        );
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
    println!("NATIVE_CPU_PAGED_PROFILES={profiles} changed_units={moves}");
}

#[test]
fn late_budget_provider_gradient_and_consumer_refusals_preserve_live_cohort_and_journal() {
    let parent = tensor(1, PlaneCodec::B3, Basis::Identity, 0, 32, 4);
    let parents = vec![parent.clone(), parent];
    let payload = packed_bytes(&parents);
    let budget = TensorPayloadBudget::new(2 * payload);
    let mut live = bound(plan(2), &parents, budget.clone());
    let current: Vec<_> = live.handles().map(|(_, h)| h.clone()).collect();
    let before = live.session().checkpoint_digest();
    let stream = Fault::default();
    let mut j = live.new_recording(stream.clone(), recording()).unwrap();
    let header = stream.data.borrow().get_ref().clone();
    let baseline = budget.used_bytes();
    for mode in 0..5 {
        let blocker = if mode == 0 {
            Some(budget.reserve_storage(payload / 2).unwrap())
        } else {
            None
        };
        let prior = budget.used_bytes();
        let result = live.stage_recorded(
            (out(), out()),
            |i, _| {
                if mode == 1 && i == 1 {
                    Err("original provider refusal")
                } else {
                    Ok(parents[i].clone())
                }
            },
            |i, _, _, w| {
                if mode == 2 && i == 1 {
                    return Err("original gradient refusal");
                }
                let mut s = vec![0.1; w.scales().len()];
                if mode == 3 && i == 1 {
                    s[0] = f32::NAN;
                }
                Ok((s, vec![0.5; active(w)]))
            },
            if mode == 4 {
                NativePvPagedRecordingLimits {
                    max_frame_bytes: 1,
                    ..recording()
                }
            } else {
                recording()
            },
        );
        let error = result.unwrap_err();
        match mode {
            0 => assert!(matches!(
                error,
                PagedError::Packed(BackendError::TensorPayloadBudgetExceeded { .. })
            )),
            1 => assert!(matches!(
                error,
                PagedError::Paged(NativePvPagedError::Provider("original provider refusal"))
            )),
            2 => assert!(matches!(
                error,
                PagedError::Paged(NativePvPagedError::Provider("original gradient refusal"))
            )),
            3 => {
                assert!(error.to_string().contains("adjoints differ"));
                assert!(matches!(error, PagedError::Journal(_)));
            }
            _ => assert!(matches!(error, PagedError::Journal(_))),
        }
        assert_eq!(budget.used_bytes(), prior);
        drop(blocker);
        assert_eq!(live.session().checkpoint_digest(), before);
        assert_eq!(live.session().completed_updates(), 0);
        assert_eq!(*stream.data.borrow().get_ref(), header);
        assert!(
            live.handles()
                .zip(&current)
                .all(|((_, h), old)| Arc::ptr_eq(h, old))
        );
    }
    let invalid = [Use::Embed {
        ids: &[3],
        output_gradients: &[0.0; 4],
    }];
    assert!(
        live.stage_from_consumers_recorded(
            (out(), out()),
            |i, _| Ok::<_, &str>(parents[i].clone()),
            &[&[], &invalid],
            cap()
        )
        .is_err()
    );
    assert!(
        live.stage_from_consumers_recorded(
            (out(), out()),
            |i, _| Ok::<_, &str>(parents[i].clone()),
            &[&[], &[]],
            StepLimits {
                max_consumer_working_bytes: 1,
                ..cap()
            }
        )
        .is_err()
    );
    assert_eq!(budget.used_bytes(), baseline);
    assert_eq!(*stream.data.borrow().get_ref(), header);
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ready = live.stage_recorded(
            (out(), out()),
            |i, _| Ok::<_, &str>(parents[i].clone()),
            |i, _, _, w| {
                assert!(i != 1, "injected paged gradient unwind");
                Ok((vec![0.1; w.scales().len()], vec![0.5; active(w)]))
            },
            recording(),
        );
    }));
    assert!(unwind.is_err());
    assert_eq!(budget.used_bytes(), baseline);
    assert_eq!(live.session().checkpoint_digest(), before);
    assert_eq!(*stream.data.borrow().get_ref(), header);
    let ready = live
        .stage_from_consumers_recorded(
            (out(), out()),
            |i, _| Ok::<_, &str>(parents[i].clone()),
            &[&[], &[]],
            cap(),
        )
        .unwrap();
    let (next, _) = ready.commit(&mut live, &mut j).unwrap();
    assert_eq!(next.session().completed_updates(), 1);
}

#[test]
fn append_write_flush_and_unwind_poison_without_exposing_new_handles() {
    for mode in 1..=4 {
        let parents = vec![tensor(1, PlaneCodec::D2, Basis::Identity, 0, 32, 4)];
        let budget = TensorPayloadBudget::new(3 * packed_bytes(&parents));
        let mut live = bound(plan(1), &parents, budget.clone());
        let before = live.session().checkpoint_digest();
        let handle = live.handles().next().unwrap().1.clone();
        let stream = Fault::default();
        let mut j = live.new_recording(stream.clone(), recording()).unwrap();
        let baseline = budget.used_bytes();
        let ready = live
            .stage_recorded(
                (out(), out()),
                |i, _| Ok::<_, &str>(parents[i].clone()),
                |_, _, _, w| Ok((vec![0.1; w.scales().len()], vec![0.5; active(w)])),
                recording(),
            )
            .unwrap();
        stream.mode.set(mode);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ready.commit(&mut live, &mut j)
        }));
        if mode >= 3 {
            assert!(result.is_err());
        } else {
            assert!(matches!(result.unwrap(), Err(Stage::Journal(_))));
        }
        stream.mode.set(0);
        assert_eq!(budget.used_bytes(), baseline);
        assert_eq!(live.session().checkpoint_digest(), before);
        assert_eq!(live.session().completed_updates(), 0);
        assert!(Arc::ptr_eq(live.handles().next().unwrap().1, &handle));
        assert!(live.checkpoint(&mut j).is_err());
    }
}

#[test]
fn other_driver_and_equal_value_fork_cannot_redirect_staged_packed_cohort() {
    let parents = vec![tensor(1, PlaneCodec::B3, Basis::Identity, 0, 32, 4)];
    let budget = TensorPayloadBudget::new(8 * packed_bytes(&parents));
    let mut original = bound(plan(1), &parents, budget.clone());
    let mut other = bound(
        plan(1),
        &parents,
        TensorPayloadBudget::new(8 * packed_bytes(&parents)),
    );
    let mut j1 = original.new_recording(out(), recording()).unwrap();
    let mut j2 = original.new_recording(out(), recording()).unwrap();
    let stage = |live: &mut Paged<Cursor<Vec<u8>>>| {
        live.stage_recorded(
            (out(), out()),
            |i, _| Ok::<_, &str>(parents[i].clone()),
            |_, _, _, w| Ok((vec![0.1; w.scales().len()], vec![0.5; active(w)])),
            recording(),
        )
        .unwrap()
    };
    let ready = stage(&mut original);
    assert!(matches!(
        ready.commit(&mut other, &mut j1),
        Err(Stage::Packed(_))
    ));
    let a = stage(&mut original);
    let b = stage(&mut original);
    let (mut a, _) = a.commit(&mut original, &mut j1).unwrap();
    let (mut b, _) = b.commit(&mut original, &mut j2).unwrap();
    assert_eq!(
        a.session().checkpoint_digest(),
        b.session().checkpoint_digest()
    );
    let ready = stage(&mut a);
    let point = b.checkpoint(&mut j2).unwrap();
    assert!(matches!(
        ready.commit(&mut b, &mut j2),
        Err(Stage::Packed(_))
    ));
    assert_eq!(b.checkpoint(&mut j2).unwrap(), point);
    assert_eq!(b.session().completed_updates(), 1);
}

#[test]
fn actual_packed_matmul_embed_loss_decreases_and_old_graph_remains_exact() {
    let parents = vec![tensor(
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
    )];
    let budget = TensorPayloadBudget::new(3 * packed_bytes(&parents));
    let mut live = bound(plan(1), &parents, budget);
    let mut journal = live.new_recording(out(), recording()).unwrap();
    let old = live.handles().next().unwrap().1.clone();
    let x = values(8, 0);
    let forward = |h: &dyn DeviceBuffer| {
        let cpu = CpuBackend::new();
        let mut y = vec![0.0; 6];
        let mut e = vec![0.0; 16];
        cpu.matmul(TensorMatmul {
            act: &x,
            tensor: h,
            batch: 2,
            transformed_act: &mut [0.0; 8],
            out: &mut y,
        })
        .unwrap();
        cpu.embed_rows(h, &[2, 0, 2, 1], &mut e).unwrap();
        (y, e)
    };
    let loss = |v: &(Vec<f32>, Vec<f32>)| {
        v.0.iter()
            .chain(&v.1)
            .map(|&x| f64::from(x).powi(2) * 0.5)
            .sum::<f64>()
    };
    let initial = forward(&*old);
    let mut moves = 0;
    for _ in 0..2 {
        let (dy, de) = forward(&**live.handles().next().unwrap().1);
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
        let ready = live
            .stage_from_consumers_recorded(
                (out(), out()),
                |i, _| Ok::<_, &str>(parents[i].clone()),
                &[&uses],
                cap(),
            )
            .unwrap();
        let (next, stats) = ready.commit(&mut live, &mut journal).unwrap();
        moves += stats.owners[0].changed_units;
        live = next;
    }
    let initial_loss = loss(&initial);
    let final_loss = loss(&forward(&**live.handles().next().unwrap().1));
    assert!(moves > 0);
    assert!(final_loss < initial_loss);
    assert_eq!(forward(&*old), initial);
    eprintln!(
        "NATIVE_CPU_PAGED_LOSS initial={initial_loss} final={final_loss} changed_units={moves}"
    );
}

#[test]
fn exact_binding_rejects_names_code_scale_and_budget_without_leaking_admission() {
    let parents = vec![tensor(1, PlaneCodec::B3, Basis::Identity, 0, 32, 4)];
    for mode in 0..5 {
        let mut weights = parents.clone();
        if mode == 1 {
            weights[0]
                .replace_scales(vec![0.25; parents[0].scales().len()])
                .unwrap();
        }
        if mode == 2 {
            weights[0]
                .replace_trits(
                    parents[0]
                        .trits()
                        .iter()
                        .map(|t| tritium_core::Trit::from_i8(-t.get()).unwrap())
                        .collect(),
                )
                .unwrap();
        }
        let mut packed = roots(&weights);
        if mode == 0 {
            packed[0].0 = "wrong-owner".into();
        }
        if mode == 3 {
            packed.clear();
        }
        let budget = TensorPayloadBudget::new(if mode == 4 { 0 } else { 100_000 });
        let state = NativePvPagedSession::initialize(
            plan(1),
            NativePvPagedLimits {
                max_resident_payload_bytes: 10_000_000,
            },
            |i, _| Ok::<_, &str>(parents[i].clone()),
            out(),
        )
        .unwrap();
        assert!(matches!(
            Paged::bind(state, packed, budget.clone(), |i, _| Ok::<_, &str>(
                parents[i].clone()
            )),
            Err(PagedError::Packed(_))
        ));
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn heterogeneous_and_empty_owner_uses_keep_exact_decay_and_moments() {
    let parents = vec![
        tensor(0, PlaneCodec::D2, Basis::Identity, 0, 32, 260),
        tensor(
            1,
            PlaneCodec::B3,
            Basis::SignedRht {
                block: 128,
                seed: 3,
                domain: 4,
            },
            1,
            128,
            512,
        ),
        tensor(
            2,
            PlaneCodec::S34,
            Basis::Hadamard { block: 128 },
            2,
            32,
            512,
        ),
    ];
    let mut live = bound(
        plan(3),
        &parents,
        TensorPayloadBudget::new(3 * packed_bytes(&parents)),
    );
    let mut direct = Session::new(plan(3), parents.clone()).unwrap();
    let mut j = live.new_recording(out(), recording()).unwrap();
    let x = values(520, 0);
    let dy = values(6, 2);
    let e = values(1024, 3);
    let a = [Use::Matmul {
        activations: &x,
        output_gradients: &dy,
        batch: 2,
    }];
    let b = [Use::Embed {
        ids: &[2, 2],
        output_gradients: &e,
    }];
    for _ in 0..2 {
        let uses: [&[Use<'_>]; 3] = [&a, &b, &[]];
        let gradients: Vec<_> = direct
            .weights()
            .zip(&uses)
            .map(|(w, u)| manual(w, u))
            .collect();
        let expected = direct
            .step(
                &gradients.iter().map(|g| g.0.as_slice()).collect::<Vec<_>>(),
                &gradients.iter().map(|g| g.1.as_slice()).collect::<Vec<_>>(),
            )
            .unwrap();
        let ready = live
            .stage_from_consumers_recorded(
                (out(), out()),
                |i, _| Ok::<_, &str>(parents[i].clone()),
                &uses,
                cap(),
            )
            .unwrap();
        let (next, actual) = ready.commit(&mut live, &mut j).unwrap();
        assert_eq!(expected, actual);
        live = next;
        check_packed(&live, &direct);
    }
}
