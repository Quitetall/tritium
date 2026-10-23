//! Actual recorded PTQ package through scoped source and paged numerical state.
use super::*;
use std::cell::Cell;
fn read_limits() -> NativeRefinementOwnerReadLimits {
    NativeRefinementOwnerReadLimits {
        max_working_bytes: 1_000_000,
        extra_payload_bytes: 0,
    }
}
fn paged_limits() -> NativePvPagedLimits {
    NativePvPagedLimits {
        max_resident_payload_bytes: 1_000_000,
    }
}
fn request(f: &Fixture) -> NativeRefinementPlan {
    let mut plan = f.plan.clone();
    plan.owners = vec!["frozen.weight".into(), "weight".into()];
    let NativeRecoveryRecipe::ScaleOnly { optimizer } = plan.recipe else {
        unreachable!()
    };
    plan.recipe = NativeRecoveryRecipe::AlternatingPv {
        continuous_optimizer: optimizer,
        code_optimizer: NativeAdamWRecipe {
            lr: 0.6,
            ..optimizer
        },
        max_code_change_fraction: 0.5,
        max_relative_code_change: Some(0.9),
    };
    plan
}
fn gradients(weight: &AdditiveTensor, i: usize) -> (Vec<f32>, Vec<f32>) {
    let a = if weight.plane_count() == 0 {
        0
    } else {
        weight.view().plane_trits(0).unwrap().len()
    };
    (
        vec![0.125; weight.scales().len()],
        (0..a)
            .map(|n| ((n + i) % 5) as f32 * 0.125 - 0.25)
            .collect(),
    )
}
#[test]
fn original_ptq_packages_feed_complete_paged_recovery_and_standard_replay_without_parent_cache() {
    let mut cases = 0;
    for precision in [ScalePrecision::F16, ScalePrecision::F32] {
        // The existing recorded direct-PTQ producer admits D2/B3, not S34.
        // S34 remains covered by the format decoder and imported paging tests.
        for codec in [PlaneCodec::D2, PlaneCodec::B3] {
            for basis in [
                Basis::Identity,
                Basis::Hadamard { block: 32 },
                Basis::SignedRht {
                    block: 32,
                    seed: 19,
                    domain: 7,
                },
            ] {
                for level in ["compact", "full"] {
                    let mut f = Fixture::new(precision, basis, codec, level, false);
                    let plan = request(&f);
                    let originals = decode_native_pv_parents(
                        &mut f.parent,
                        &f.view,
                        &plan,
                        NativePvParentReadLimits {
                            max_working_bytes: 1_000_000,
                            extra_payload_bytes: 0,
                        },
                    )
                    .unwrap();
                    let mut reference =
                        NativePvRefinementSession::new(plan.clone(), originals.clone()).unwrap();
                    let mut journal = NativePvJournalWriter::new(
                        Cursor::new(Vec::new()),
                        &reference,
                        NativePvJournalLimits {
                            max_journal_bytes: 1_000_000,
                            max_working_bytes: 1_000_000,
                        },
                    )
                    .unwrap();
                    let mut paged = NativeRefinementOwnerSource::with(
                        &mut f.parent,
                        &f.view,
                        &plan,
                        read_limits(),
                        |source| {
                            NativePvPagedSession::initialize(
                                plan.clone(),
                                paged_limits(),
                                |i, name| source.load(i, name),
                                Cursor::new(Vec::new()),
                            )
                        },
                    )
                    .unwrap();
                    for step in 0..4 {
                        let g: Vec<_> = reference
                            .weights()
                            .enumerate()
                            .map(|(i, w)| gradients(w, i + step))
                            .collect();
                        let expected = reference
                            .step_recorded(
                                &g.iter().map(|(s, _)| s.as_slice()).collect::<Vec<_>>(),
                                &g.iter().map(|(_, w)| w.as_slice()).collect::<Vec<_>>(),
                                &mut journal,
                            )
                            .unwrap();
                        let (next, stats, resources) = NativeRefinementOwnerSource::with(
                            &mut f.parent,
                            &f.view,
                            &plan,
                            read_limits(),
                            |source| {
                                paged.step_staged(
                                    Cursor::new(Vec::new()),
                                    |i, name| source.load(i, name),
                                    |i, _, w| Ok(gradients(w, i + step)),
                                    |_, _, _| Ok(()),
                                )
                            },
                        )
                        .unwrap();
                        assert_eq!(stats, expected);
                        assert_eq!(resources.len(), 2);
                        assert_eq!(paged.completed_updates(), step as u64);
                        let bytes = next.into_checkpoint().into_inner();
                        assert_eq!(bytes, reference.checkpoint_bytes().unwrap());
                        paged = NativeRefinementOwnerSource::with(
                            &mut f.parent,
                            &f.view,
                            &plan,
                            read_limits(),
                            |source| {
                                NativePvPagedSession::reopen(
                                    plan.clone(),
                                    paged_limits(),
                                    |i, name| source.load(i, name),
                                    Cursor::new(bytes),
                                )
                            },
                        )
                        .unwrap();
                    }
                    let (journal, summary) = journal.finish(&reference).unwrap();
                    let checkpoint = paged.into_checkpoint().into_inner();
                    let replay = replay_native_pv_journal(
                        plan,
                        originals,
                        journal,
                        summary.journal(),
                        &checkpoint,
                        NativePvJournalLimits {
                            max_journal_bytes: 1_000_000,
                            max_working_bytes: 1_000_000,
                        },
                    )
                    .unwrap();
                    replay.verify_session(&reference).unwrap();
                    assert_eq!(*f.bytes.borrow(), f.original);
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 24);
}

struct Dropped(Rc<Cell<usize>>);

#[test]
fn nested_scope_swap_cannot_redirect_terminal_verification() {
    let mut first = Fixture::new(
        ScalePrecision::F32,
        Basis::Identity,
        PlaneCodec::B3,
        "compact",
        false,
    );
    let mut second = Fixture::new(
        ScalePrecision::F32,
        Basis::Identity,
        PlaneCodec::B3,
        "full",
        false,
    );
    let a = request(&first);
    let b = request(&second);
    let result = NativeRefinementOwnerSource::with(
        &mut first.parent,
        &first.view,
        &a,
        read_limits(),
        |left| {
            for (i, name) in a.owners.iter().enumerate() {
                left.load(i, name).unwrap();
            }
            let inner = NativeRefinementOwnerSource::with(
                &mut second.parent,
                &second.view,
                &b,
                read_limits(),
                |right| {
                    for (i, name) in b.owners.iter().enumerate() {
                        right.load(i, name).unwrap();
                    }
                    std::mem::swap(left, right);
                    Ok::<_, ()>(())
                },
            );
            assert!(inner.is_err());
            Ok::<_, ()>(())
        },
    );
    assert!(matches!(
        result,
        Err(NativeRefinementOwnerSourceError::Source(_))
    ));
}

#[test]
fn retained_parent_file_to_paged_files_reopens_exact_common_state() {
    use std::fs::{File, OpenOptions};
    let f = Fixture::new(
        ScalePrecision::F32,
        Basis::Identity,
        PlaneCodec::B3,
        "full",
        false,
    );
    let plan = request(&f);
    let root = Scratch::new("source-paged-files");
    let parent_path = root.0.join("parent.trit");
    fs::write(&parent_path, &f.original).unwrap();
    let mut reader = TritModelReader::open(File::open(&parent_path).unwrap(), receiving()).unwrap();
    let make = |name: &str| {
        OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(root.0.join(name))
            .unwrap()
    };
    let mut current =
        NativeRefinementOwnerSource::with(&mut reader, &f.view, &plan, read_limits(), |source| {
            NativePvPagedSession::initialize(
                plan.clone(),
                paged_limits(),
                |i, name| source.load(i, name),
                make("initial.tnpv"),
            )
        })
        .unwrap();
    let (next, stats, _) =
        NativeRefinementOwnerSource::with(&mut reader, &f.view, &plan, read_limits(), |source| {
            current.step_staged(
                make("next.tnpv"),
                |i, name| source.load(i, name),
                |i, _, w| Ok(gradients(w, i)),
                |_, _, _| Ok(()),
            )
        })
        .unwrap();
    assert_eq!(stats.completed_updates, 1);
    assert_eq!(current.completed_updates(), 0);
    let observed = next.checkpoint_digest();
    drop(next);
    let reopened =
        NativeRefinementOwnerSource::with(&mut reader, &f.view, &plan, read_limits(), |source| {
            NativePvPagedSession::reopen(
                plan.clone(),
                paged_limits(),
                |i, name| source.load(i, name),
                File::open(root.0.join("next.tnpv")).unwrap(),
            )
        })
        .unwrap();
    assert_eq!(reopened.completed_updates(), 1);
    assert_eq!(reopened.checkpoint_digest(), observed);
    let originals = decode_native_pv_parents(
        &mut reader,
        &f.view,
        &plan,
        NativePvParentReadLimits {
            max_working_bytes: 1_000_000,
            extra_payload_bytes: 0,
        },
    )
    .unwrap();
    let mut resident = NativePvRefinementSession::new(plan, originals.clone()).unwrap();
    let g: Vec<_> = originals
        .iter()
        .enumerate()
        .map(|(i, w)| gradients(w, i))
        .collect();
    resident
        .step(
            &g.iter().map(|(s, _)| s.as_slice()).collect::<Vec<_>>(),
            &g.iter().map(|(_, w)| w.as_slice()).collect::<Vec<_>>(),
        )
        .unwrap();
    assert_eq!(
        fs::read(root.0.join("next.tnpv")).unwrap(),
        resident.checkpoint_bytes().unwrap()
    );
    assert_eq!(fs::read(parent_path).unwrap(), f.original);
    drop(reopened);
    drop(current);
    drop(reader);
}

impl Drop for Dropped {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}
#[test]
fn strict_scoped_order_caps_poisoning_consumer_failure_and_terminal_source_drift_refuse() {
    let mut f = Fixture::new(
        ScalePrecision::F32,
        Basis::Identity,
        PlaneCodec::B3,
        "compact",
        false,
    );
    let plan = request(&f);
    let working =
        NativeRefinementOwnerSource::with(&mut f.parent, &f.view, &plan, read_limits(), |source| {
            let needed = source.working_bytes();
            for (i, name) in plan.owners.iter().enumerate() {
                source.load(i, name)?;
            }
            Ok::<_, NativePvChildError>(needed)
        })
        .unwrap();
    for extra in [0, 777] {
        assert_eq!(
            NativeRefinementOwnerSource::required_working_bytes(
                &mut f.parent,
                &f.view,
                &plan,
                extra
            )
            .unwrap(),
            working + extra
        );
        let exact = NativeRefinementOwnerReadLimits {
            max_working_bytes: working + extra,
            extra_payload_bytes: extra,
        };
        NativeRefinementOwnerSource::with(&mut f.parent, &f.view, &plan, exact, |source| {
            assert_eq!(source.working_bytes(), working + extra);
            for _ in 0..2 {
                for (i, name) in plan.owners.iter().enumerate() {
                    source.load(i, name)?;
                }
            }
            Ok::<_, NativePvChildError>(())
        })
        .unwrap();
        let called = Cell::new(false);
        assert!(matches!(
            NativeRefinementOwnerSource::with(
                &mut f.parent,
                &f.view,
                &plan,
                NativeRefinementOwnerReadLimits {
                    max_working_bytes: exact.max_working_bytes - 1,
                    ..exact
                },
                |_| {
                    called.set(true);
                    Ok::<_, ()>(())
                }
            ),
            Err(NativeRefinementOwnerSourceError::Source(_))
        ));
        assert!(!called.get());
    }
    for mode in 0..4 {
        let result = NativeRefinementOwnerSource::with(
            &mut f.parent,
            &f.view,
            &plan,
            read_limits(),
            |source| {
                match mode {
                    0 => {}
                    1 => {
                        source.load(0, &plan.owners[0]).unwrap();
                    }
                    2 => {
                        assert!(source.load(1, &plan.owners[1]).is_err());
                    }
                    _ => {
                        assert!(source.load(0, "wrong").is_err());
                    }
                }
                Ok::<_, ()>(())
            },
        );
        assert!(matches!(
            result,
            Err(NativeRefinementOwnerSourceError::Source(_))
        ));
    }
    assert!(matches!(
        NativeRefinementOwnerSource::with(&mut f.parent, &f.view, &plan, read_limits(), |_| Err::<
            (),
            _,
        >(
            "original consumer error"
        )),
        Err(NativeRefinementOwnerSourceError::Consumer(
            "original consumer error"
        ))
    ));
    let drops = Rc::new(Cell::new(0));
    let bytes = f.bytes.clone();
    let result =
        NativeRefinementOwnerSource::with(&mut f.parent, &f.view, &plan, read_limits(), |source| {
            for (i, name) in plan.owners.iter().enumerate() {
                source.load(i, name)?;
            }
            let last = bytes.borrow().len() - 1;
            bytes.borrow_mut()[last] ^= 1;
            Ok::<_, NativePvChildError>(Dropped(drops.clone()))
        });
    assert!(matches!(
        result,
        Err(NativeRefinementOwnerSourceError::Source(_))
    ));
    assert_eq!(drops.get(), 1);
    f.bytes.borrow_mut().copy_from_slice(&f.original);
    let mut wrong = plan.clone();
    wrong.parent.source = OriginalSourceId::from_bytes([99; 32]);
    assert!(
        NativeRefinementOwnerSource::with(&mut f.parent, &f.view, &wrong, read_limits(), |_| Ok::<
            _,
            (),
        >(
            ()
        ))
        .is_err()
    );
    let mut short = plan.clone();
    short.limits.max_state_bytes = 1;
    let called = Cell::new(false);
    assert!(
        NativeRefinementOwnerSource::with(&mut f.parent, &f.view, &short, read_limits(), |_| {
            called.set(true);
            Ok::<_, ()>(())
        })
        .is_err()
    );
    assert!(!called.get());
}
