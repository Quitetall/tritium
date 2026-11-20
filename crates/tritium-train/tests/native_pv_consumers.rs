//! Real consumer reduction through native P/V; synthetic numerical inputs only.
use std::io::Cursor;
use tritium_schema::*;
use tritium_train::pv_tuning::{
    NativePvContribution as Use, NativePvJournalLimits as Limits, NativePvJournalWriter as Writer,
    NativePvRefinementSession as Session, replay_native_pv_journal_with_state as replay,
};
#[path = "common/native_pv_consumer.rs"]
mod fixture;
use fixture::{active, manual, plan, tensor, values};
fn limits() -> Limits {
    Limits {
        max_journal_bytes: 10_000_000,
        max_working_bytes: 10_000_000,
    }
}

#[test]
fn ordered_repeated_consumers_match_direct_adjoints_and_checkpoint_restart_all_profiles() {
    let mut profiles = 0;
    let mut code_moves = 0;
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
                        let parent = tensor(law, codec, basis, mode, group, cols);
                        let p = plan(1);
                        let mut actual = Session::new(p.clone(), vec![parent.clone()]).unwrap();
                        let mut direct = Session::new(p.clone(), vec![parent.clone()]).unwrap();
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
                            let (s, a) = manual(direct.weights().next().unwrap(), &uses);
                            let expected = direct.step(&[&s], &[&a]).unwrap();
                            assert_eq!(actual.step_from_consumers(&[&uses]).unwrap(), expected);
                            assert_eq!(
                                actual.checkpoint_bytes().unwrap(),
                                direct.checkpoint_bytes().unwrap()
                            );
                            code_moves += expected.owners[0].changed_units;
                            if step == 0 {
                                actual = Session::resume(
                                    p.clone(),
                                    vec![parent.clone()],
                                    &actual.checkpoint_bytes().unwrap(),
                                )
                                .unwrap();
                            }
                        }
                        assert!(actual.step_from_consumers(&[&uses]).is_err());
                        profiles += 1;
                    }
                }
            }
        }
    }
    assert_eq!(profiles, 216);
    assert!(code_moves > 0);
}

#[test]
fn consumer_journal_records_exact_unweighted_bits_and_replays_original_parents() {
    let parents = vec![
        tensor(0, PlaneCodec::D2, Basis::Identity, 0, 32, 4),
        tensor(1, PlaneCodec::B3, Basis::Identity, 0, 32, 4),
    ];
    let p = plan(2);
    let mut state = Session::new(p.clone(), parents.clone()).unwrap();
    let mut direct = Session::new(p.clone(), parents.clone()).unwrap();
    let mut writer = Writer::new(Cursor::new(Vec::new()), &state, limits()).unwrap();
    let mut other = Writer::new(Cursor::new(Vec::new()), &direct, limits()).unwrap();
    let x = values(8, 1);
    let dy = values(6, 2);
    let de = values(12, 3);
    let uses = [
        Use::Matmul {
            activations: &x,
            output_gradients: &dy,
            batch: 2,
        },
        Use::Embed {
            ids: &[2, 0, 2],
            output_gradients: &de,
        },
    ];
    for _ in 0..2 {
        let gradients: Vec<_> = direct.weights().map(|w| manual(w, &uses)).collect();
        let s: Vec<_> = gradients.iter().map(|(s, _)| s.as_slice()).collect();
        let a: Vec<_> = gradients.iter().map(|(_, a)| a.as_slice()).collect();
        assert_eq!(
            state
                .step_from_consumers_recorded(&[&uses, &uses], &mut writer)
                .unwrap(),
            direct.step_recorded(&s, &a, &mut other).unwrap()
        );
    }
    let cp = state.checkpoint_bytes().unwrap();
    let (out, summary) = writer.finish(&state).unwrap();
    let (expected, _) = other.finish(&direct).unwrap();
    assert_eq!(out.get_ref(), expected.get_ref());
    let (replayed, proof) = replay(
        p,
        parents,
        Cursor::new(out.into_inner()),
        summary.journal(),
        &cp,
        limits(),
    )
    .unwrap();
    assert_eq!(replayed.checkpoint_bytes().unwrap(), cp);
    proof.verify_session(&state).unwrap();
}

#[test]
fn exact_workspace_empty_uses_and_all_owner_preflight_are_atomic() {
    let parents = vec![
        tensor(0, PlaneCodec::D2, Basis::Identity, 0, 32, 4),
        tensor(1, PlaneCodec::B3, Basis::Identity, 0, 32, 260),
    ];
    let p = plan(2);
    let state = Session::new(p.clone(), parents.clone()).unwrap();
    let g: usize = parents.iter().map(|w| w.scales().len() + active(w)).sum();
    let m = parents
        .iter()
        .map(|w| w.scales().len() + active(w))
        .max()
        .unwrap();
    let required = (4 * g as u64 + 8 * m as u64 + 4 * 260)
        .max(4 * g as u64 + state.size_ledger().update_workspace_bytes);
    let mut short = p.clone();
    short.limits.max_workspace_bytes = required - 1;
    let mut rejected = Session::new(short, parents.clone()).unwrap();
    let before = rejected.checkpoint_bytes().unwrap();
    assert!(
        rejected
            .step_from_consumers(&[&[], &[]])
            .unwrap_err()
            .to_string()
            .contains("workspace")
    );
    assert_eq!(rejected.checkpoint_bytes().unwrap(), before);
    let mut exact = p;
    exact.limits.max_workspace_bytes = required;
    let mut actual = Session::new(exact.clone(), parents.clone()).unwrap();
    let mut direct = Session::new(exact.clone(), parents.clone()).unwrap();
    let zeros: Vec<_> = parents
        .iter()
        .map(|w| (vec![0.0; w.scales().len()], vec![0.0; active(w)]))
        .collect();
    direct
        .step(
            &zeros.iter().map(|(s, _)| s.as_slice()).collect::<Vec<_>>(),
            &zeros.iter().map(|(_, a)| a.as_slice()).collect::<Vec<_>>(),
        )
        .unwrap();
    actual.step_from_consumers(&[&[], &[]]).unwrap();
    assert_eq!(
        actual.checkpoint_bytes().unwrap(),
        direct.checkpoint_bytes().unwrap()
    );
    for uses in [
        vec![Use::Embed {
            ids: &[3],
            output_gradients: &[0.0; 260],
        }],
        vec![Use::Matmul {
            activations: &[f32::NAN; 260],
            output_gradients: &[0.0; 3],
            batch: 1,
        }],
        vec![Use::Matmul {
            activations: &[],
            output_gradients: &[],
            batch: usize::MAX,
        }],
        vec![
            Use::Embed {
                ids: &[0],
                output_gradients: &[f32::MAX; 260],
            },
            Use::Embed {
                ids: &[0],
                output_gradients: &[f32::MAX; 260],
            },
        ],
    ] {
        let before = actual.checkpoint_bytes().unwrap();
        assert!(actual.step_from_consumers(&[&[], &uses]).is_err());
        assert_eq!(actual.checkpoint_bytes().unwrap(), before);
    }
}
