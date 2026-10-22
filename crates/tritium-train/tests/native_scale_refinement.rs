//! Small CPU numerical/checkpoint fixtures, not model-quality or release evidence.
use half::f16;
use tritium_core::{
    PerTilePlan, Trit, UniformPlan, accumulate_embed_scale_vjp,
    accumulate_ternary_matmul_scale_vjp, reference_embed, reference_ternary_matmul,
};
use tritium_format::AdditiveTensor;
use tritium_schema::*;
use tritium_train::optim::{AdamW, Optimizer};
use tritium_train::pv_tuning::{
    NativeScaleContribution as Use, NativeScaleRefinementSession as Session,
};

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
    let admitted = ADMITTED_LAWS[law];
    let layout = AdditiveLayout {
        rows: 3,
        cols: if basis == Basis::Identity { 260 } else { 512 },
        tile: 256,
        group: 32,
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

fn simple() -> AdditiveTensor {
    tensor(1, PlaneCodec::D2, Basis::Identity, 0, Transport::Raw)
}
fn step(session: &mut Session, value: f32) {
    let g: Vec<_> = session
        .weights()
        .map(|w| vec![value; w.scales().len()])
        .collect();
    session
        .step(&g.iter().map(Vec::as_slice).collect::<Vec<_>>())
        .unwrap();
}
fn checkpoint(session: &Session) -> Vec<u8> {
    session.checkpoint_bytes().unwrap()
}
fn reseal(bytes: &mut [u8]) {
    let n = bytes.len() - 32;
    let hash = blake3::hash(&bytes[..n]);
    bytes[n..].copy_from_slice(hash.as_bytes());
}

#[test]
fn exact_update_resume_all_admitted_laws_codecs_allocations_bases_and_transports() {
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
                        let parent = tensor(law, codec, basis, mode, transport);
                        let mut a = Session::new(plan(1), vec![parent.clone()]).unwrap();
                        let address = a.weights().next().unwrap().trits().as_ptr();
                        step(&mut a, 0.125);
                        assert_eq!(
                            address,
                            a.weights().next().unwrap().trits().as_ptr(),
                            "update must not replace coefficients"
                        );
                        let bytes = checkpoint(&a);
                        assert_eq!(bytes.len() as u64, a.size_ledger().checkpoint_bytes);
                        let mut b = Session::resume(plan(1), vec![parent.clone()], &bytes).unwrap();
                        assert_eq!(checkpoint(&b), bytes);
                        for gradient in [-0.25, 0.5, 0.0] {
                            step(&mut a, gradient);
                            step(&mut b, gradient);
                            assert_eq!(checkpoint(&a), checkpoint(&b));
                        }
                        let current = a.weights().next().unwrap();
                        assert_eq!(current.trits(), parent.trits());
                        assert_eq!(current.tile_counts(), parent.tile_counts());
                        assert_eq!(current.layout(), parent.layout());
                        let before = checkpoint(&a);
                        assert!(a.step(&[&[]]).is_err());
                        assert_eq!(checkpoint(&a), before);
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 324);
}

#[test]
fn updates_use_shared_adam_actual_precision_and_declared_window_weight() {
    for law in 0..ADMITTED_LAWS.len() {
        let parent = tensor(law, PlaneCodec::D2, Basis::Identity, 0, Transport::Raw);
        let mut expected = parent.scales().to_vec();
        let optimizer = AdamW {
            lr: 0.01,
            beta1: 0.9,
            beta2: 0.99,
            eps: 1e-8,
            weight_decay: 0.01,
        };
        let mut state = optimizer.init_state(expected.len());
        let mut session = Session::new(plan(1), vec![parent.clone()]).unwrap();
        for (i, gradient) in [0.125, -0.25, 0.5, 0.0].into_iter().enumerate() {
            let weighted = gradient * if i == 0 { 2.0 } else { 1.0 };
            optimizer.step(
                i as u64 + 1,
                &mut expected,
                &vec![weighted; parent.scales().len()],
                &mut state,
            );
            for scale in &mut expected {
                *scale = scale.max(0.0);
                if parent.layout().law.precision == ScalePrecision::F16 {
                    *scale = f16::from_f32(*scale).to_f32();
                }
            }
            step(&mut session, gradient);
            assert_eq!(session.weights().next().unwrap().scales(), expected);
        }
        if law == 1 {
            assert!(
                session
                    .weights()
                    .next()
                    .unwrap()
                    .scales()
                    .iter()
                    .any(|v| f16::from_f32(*v).to_f32().to_bits() != v.to_bits())
            );
        }
    }
}

#[test]
fn all_owners_commit_atomically_and_retry_matches_uninterrupted() {
    let parents = vec![simple(), simple()];
    let mut session = Session::new(plan(2), parents.clone()).unwrap();
    step(&mut session, 0.125);
    let good = checkpoint(&session);
    let n = parents[0].scales().len();
    for bad in [f32::NAN, f32::INFINITY, f32::MAX, 1e20] {
        assert!(session.step(&[&vec![0.25; n], &vec![bad; n]]).is_err());
        assert_eq!(checkpoint(&session), good);
    }
    assert!(session.step(&[&vec![0.25; n]]).is_err());
    assert!(session.step(&[&vec![0.25; n], &[]]).is_err());
    assert_eq!(checkpoint(&session), good);
    let mut resumed = Session::resume(plan(2), parents, &good).unwrap();
    step(&mut session, -0.125);
    step(&mut resumed, -0.125);
    assert_eq!(checkpoint(&session), checkpoint(&resumed));
}

#[test]
fn weighted_overflow_and_storage_overflow_leave_last_good_state() {
    let mut p = plan(1);
    p.training[0].weight = f64::MAX;
    let mut session = Session::new(p, vec![simple()]).unwrap();
    let good = checkpoint(&session);
    let n = session.weights().next().unwrap().scales().len();
    assert!(session.step(&[&vec![2.0; n]]).is_err());
    assert_eq!(checkpoint(&session), good);
    let mut parent = tensor(0, PlaneCodec::D2, Basis::Identity, 0, Transport::Raw);
    parent
        .replace_scales(vec![65504.0; parent.scales().len()])
        .unwrap();
    let mut p = plan(1);
    if let NativeRecoveryRecipe::ScaleOnly { optimizer } = &mut p.recipe {
        optimizer.lr = 100.0;
        optimizer.weight_decay = 0.0;
    }
    let mut session = Session::new(p, vec![parent]).unwrap();
    let good = checkpoint(&session);
    assert!(session.step(&[&vec![-1.0; n]]).is_err());
    assert_eq!(checkpoint(&session), good);
}

#[test]
fn limits_are_exact_and_alternating_pv_cannot_silently_use_scale_only() {
    let parents = vec![simple()];
    let ledger = Session::new(plan(1), parents.clone())
        .unwrap()
        .size_ledger();
    let mut p = plan(1);
    p.limits.max_state_bytes = ledger.state_payload_bytes;
    p.limits.max_workspace_bytes = ledger.update_workspace_bytes;
    p.limits.max_checkpoint_bytes = ledger.checkpoint_bytes;
    assert!(Session::new(p.clone(), parents.clone()).is_ok());
    for field in 0..3 {
        let mut bad = p.clone();
        match field {
            0 => bad.limits.max_state_bytes -= 1,
            1 => bad.limits.max_workspace_bytes -= 1,
            _ => bad.limits.max_checkpoint_bytes -= 1,
        };
        assert!(Session::new(bad, parents.clone()).is_err());
    }
    let NativeRecoveryRecipe::ScaleOnly { optimizer } = p.recipe else {
        unreachable!()
    };
    p.recipe = NativeRecoveryRecipe::AlternatingPv {
        continuous_optimizer: optimizer,
        code_optimizer: optimizer,
        max_code_change_fraction: 0.1,
        max_relative_code_change: None,
    };
    assert!(Session::new(p, parents.clone()).is_err());
    assert!(Session::new(plan(2), parents).is_err());
}

#[test]
fn exact_parent_plan_binding_and_all_checkpoint_bytes_are_checked() {
    let parent = simple();
    let mut a = Session::new(plan(1), vec![parent.clone()]).unwrap();
    step(&mut a, 0.125);
    let bytes = checkpoint(&a);
    for i in 0..bytes.len() {
        let mut bad = bytes.clone();
        bad[i] ^= 1;
        assert!(
            Session::resume(plan(1), vec![parent.clone()], &bad).is_err(),
            "byte {i}"
        );
    }
    for end in 0..bytes.len() {
        assert!(Session::resume(plan(1), vec![parent.clone()], &bytes[..end]).is_err());
    }
    let mut bad = bytes.clone();
    bad.push(0);
    assert!(Session::resume(plan(1), vec![parent.clone()], &bad).is_err());
    let mut changed = parent.clone();
    changed
        .replace_scales(vec![0.75; changed.scales().len()])
        .unwrap();
    assert!(Session::resume(plan(1), vec![changed], &bytes).is_err());
    let mut p = plan(1);
    p.parent.level = "other-explicit-level".into();
    assert!(Session::resume(p, vec![parent.clone()], &bytes).is_err());
    let mut p = plan(1);
    p.training.swap(0, 1);
    assert!(Session::resume(p, vec![parent], &bytes).is_err());
}

#[test]
fn resealed_invalid_format_cursor_counts_values_and_moments_are_rejected() {
    let parent = simple();
    let mut a = Session::new(plan(1), vec![parent.clone()]).unwrap();
    step(&mut a, 0.125);
    let good = checkpoint(&a);
    let n = parent.scales().len();
    // offsets: magic4, version1, plan32, parent32, cursor8, owners8, scale-count8.
    for (offset, replacement) in [
        (0, b"TPV1".to_vec()),
        (4, vec![2]),
        (69, 5u64.to_le_bytes().to_vec()),
        (77, 2u64.to_le_bytes().to_vec()),
        (85, (n as u64 + 1).to_le_bytes().to_vec()),
        (93, f32::NAN.to_bits().to_le_bytes().to_vec()),
        (93, (-0.0f32).to_bits().to_le_bytes().to_vec()),
        (93 + 4 * n, f32::INFINITY.to_bits().to_le_bytes().to_vec()),
        (93 + 8 * n, (-1.0f32).to_bits().to_le_bytes().to_vec()),
    ] {
        let mut bad = good.clone();
        bad[offset..offset + replacement.len()].copy_from_slice(&replacement);
        reseal(&mut bad);
        assert!(
            Session::resume(plan(1), vec![parent.clone()], &bad).is_err(),
            "offset {offset}"
        );
    }
    let zero = Session::new(plan(1), vec![parent.clone()]).unwrap();
    let mut bad = checkpoint(&zero);
    bad[93..97].copy_from_slice(&0.75f32.to_bits().to_le_bytes());
    reseal(&mut bad);
    assert!(Session::resume(plan(1), vec![parent.clone()], &bad).is_err());
    let mut bad = checkpoint(&zero);
    bad[93 + 4 * n..97 + 4 * n].copy_from_slice(&0.5f32.to_bits().to_le_bytes());
    reseal(&mut bad);
    assert!(Session::resume(plan(1), vec![parent], &bad).is_err());
}

#[test]
fn scale_replacement_is_exact_and_transactional_and_s34_is_not_bypassed() {
    let mut parent = tensor(0, PlaneCodec::S34, Basis::Identity, 0, Transport::Raw);
    let before = parent.clone();
    let n = parent.scales().len();
    for value in [f32::NAN, f32::INFINITY, -1.0, -0.0, 0.50001] {
        assert!(parent.replace_scales(vec![value; n]).is_err());
        assert_eq!(parent, before);
    }
    assert!(parent.replace_scales(vec![]).is_err());
    assert_eq!(parent, before);
    parent.replace_scales(vec![0.0; n]).unwrap();
    assert_eq!(parent.trits(), before.trits());
    let invalid = AdditiveTensor::new(
        before.layout(),
        before.plane_count(),
        vec![Trit::POS; before.trits().len()],
        before.scales().to_vec(),
    )
    .unwrap();
    assert!(Session::new(plan(1), vec![invalid]).is_err());
}

#[test]
fn exact_parent_binding_includes_codes_counts_layout_basis_codec_and_transport() {
    let parent = simple();
    let bytes = checkpoint(&Session::new(plan(1), vec![parent.clone()]).unwrap());
    let mut changed = Vec::new();
    for choice in 0..3 {
        let mut layout = parent.layout();
        match choice {
            0 => layout.basis = Basis::Hadamard { block: 4 },
            1 => layout.codec = PlaneCodec::B3,
            _ => layout.transport = Transport::JointHuffman { block: 128 },
        }
        changed.push(
            AdditiveTensor::new(
                layout,
                parent.plane_count(),
                parent.trits().to_vec(),
                parent.scales().to_vec(),
            )
            .unwrap(),
        );
    }
    let mut codes = parent.trits().to_vec();
    codes[0] = Trit::POS;
    changed.push(
        AdditiveTensor::new(
            parent.layout(),
            parent.plane_count(),
            codes,
            parent.scales().to_vec(),
        )
        .unwrap(),
    );
    changed.push(tensor(
        1,
        PlaneCodec::D2,
        Basis::Identity,
        1,
        Transport::Raw,
    ));
    for weight in changed {
        assert!(Session::resume(plan(1), vec![weight], &bytes).is_err());
    }
}

#[test]
fn nonnegative_projection_permits_exact_zero_without_mutating_codes() {
    let parent = simple();
    let codes = parent.trits().to_vec();
    let mut p = plan(1);
    if let NativeRecoveryRecipe::ScaleOnly { optimizer } = &mut p.recipe {
        optimizer.lr = 2.0;
        optimizer.weight_decay = 0.0;
    }
    let mut session = Session::new(p.clone(), vec![parent.clone()]).unwrap();
    step(&mut session, 1.0);
    assert!(
        session
            .weights()
            .next()
            .unwrap()
            .scales()
            .iter()
            .all(|v| v.to_bits() == 0)
    );
    assert_eq!(session.weights().next().unwrap().trits(), codes);
    let bytes = checkpoint(&session);
    assert_eq!(
        checkpoint(&Session::resume(p, vec![parent], &bytes).unwrap()),
        bytes
    );
}

#[test]
fn large_u64_rotation_seed_and_domain_bits_are_not_rounded_in_parent_binding() {
    let seed = 1u64 << 60;
    let parent = tensor(
        1,
        PlaneCodec::D2,
        Basis::SignedRht {
            block: 128,
            seed,
            domain: seed,
        },
        0,
        Transport::Raw,
    );
    let bytes = checkpoint(&Session::new(plan(1), vec![parent.clone()]).unwrap());
    for basis in [
        Basis::SignedRht {
            block: 128,
            seed: seed + 1,
            domain: seed,
        },
        Basis::SignedRht {
            block: 128,
            seed,
            domain: seed + 1,
        },
    ] {
        let mut layout = parent.layout();
        layout.basis = basis;
        let other = AdditiveTensor::new(
            layout,
            parent.plane_count(),
            parent.trits().to_vec(),
            parent.scales().to_vec(),
        )
        .unwrap();
        assert!(Session::resume(plan(1), vec![other], &bytes).is_err());
    }
}

fn consumer_values(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i % 7) as f32 * 0.125 - 0.375).collect()
}

#[test]
fn tied_repeated_consumers_match_direct_scale_step_and_resume_across_profiles() {
    let mut cases = 0;
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
                    let parent = tensor(law, codec, basis, mode, Transport::Raw);
                    let cols = parent.layout().cols as usize;
                    let x = consumer_values(2 * cols);
                    let dy = consumer_values(6);
                    let ids = [2, 0, 2];
                    let de = consumer_values(3 * cols);
                    let uses = [
                        Use::Matmul {
                            activations: &x,
                            output_gradients: &dy,
                            batch: 2,
                        },
                        Use::Embed {
                            ids: &ids,
                            output_gradients: &de,
                        },
                        Use::Matmul {
                            activations: &x[..cols],
                            output_gradients: &dy[..3],
                            batch: 1,
                        },
                    ];
                    let view = parent.view();
                    let mut g = vec![0.0; parent.scales().len()];
                    let mut row = vec![0.0; cols];
                    accumulate_ternary_matmul_scale_vjp(
                        &x,
                        &view.bind_matmul(),
                        &dy,
                        2,
                        &mut row,
                        &mut g,
                    )
                    .unwrap();
                    accumulate_embed_scale_vjp(&view.bind_gather(), &ids, &de, &mut row, &mut g)
                        .unwrap();
                    accumulate_ternary_matmul_scale_vjp(
                        &x[..cols],
                        &view.bind_matmul(),
                        &dy[..3],
                        1,
                        &mut row,
                        &mut g,
                    )
                    .unwrap();
                    let g: Vec<f32> = g.iter().map(|&v| v as f32).collect();
                    let mut actual = Session::new(plan(1), vec![parent.clone()]).unwrap();
                    let mut direct = Session::new(plan(1), vec![parent.clone()]).unwrap();
                    let mut resumed: Option<Session> = None;
                    for window in 0..4 {
                        actual.step_from_consumers(&[&uses]).unwrap();
                        direct.step(&[&g]).unwrap();
                        assert_eq!(checkpoint(&actual), checkpoint(&direct));
                        if let Some(state) = &mut resumed {
                            state.step_from_consumers(&[&uses]).unwrap();
                            assert_eq!(checkpoint(state), checkpoint(&actual));
                        }
                        if window == 1 {
                            resumed = Some(
                                Session::resume(
                                    plan(1),
                                    vec![parent.clone()],
                                    &checkpoint(&actual),
                                )
                                .unwrap(),
                            );
                        }
                    }
                    assert!(actual.step_from_consumers(&[&uses]).is_err());
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 108);
}

#[test]
fn consumer_workspace_bound_is_exact_and_empty_uses_mean_zero_gradient_not_skip() {
    let parents = vec![
        simple(),
        tensor(3, PlaneCodec::D2, Basis::Identity, 0, Transport::Raw),
    ];
    let total: usize = parents.iter().map(|w| w.scales().len()).sum();
    let max = parents.iter().map(|w| w.scales().len()).max().unwrap();
    let required = (4 * total + 8 * max + 4 * 260).max(20 * total) as u64;
    let mut p = plan(2);
    p.limits.max_workspace_bytes = required - 1;
    let mut insufficient = Session::new(p, parents.clone()).unwrap();
    let good = checkpoint(&insufficient);
    assert!(insufficient.step_from_consumers(&[&[], &[]]).is_err());
    assert_eq!(checkpoint(&insufficient), good);
    let mut p = plan(2);
    p.limits.max_workspace_bytes = required;
    let mut actual = Session::new(p.clone(), parents.clone()).unwrap();
    let mut direct = Session::new(p, parents.clone()).unwrap();
    actual.step_from_consumers(&[&[], &[]]).unwrap();
    let zero: Vec<_> = parents
        .iter()
        .map(|w| vec![0.0; w.scales().len()])
        .collect();
    direct
        .step(&zero.iter().map(Vec::as_slice).collect::<Vec<_>>())
        .unwrap();
    assert_eq!(checkpoint(&actual), checkpoint(&direct));
    assert_ne!(
        actual.weights().next().unwrap().scales(),
        parents[0].scales()
    );
}

#[test]
fn invalid_or_overflowing_consumer_contributions_preserve_all_owner_state() {
    let parents = vec![simple(), simple()];
    let mut actual = Session::new(plan(2), parents).unwrap();
    step(&mut actual, 0.125);
    let good = checkpoint(&actual);
    let x = vec![0.125; 260];
    let dy = [0.25; 3];
    let valid = [Use::Matmul {
        activations: &x,
        output_gradients: &dy,
        batch: 1,
    }];
    let invalid = [Use::Embed {
        ids: &[0, 3],
        output_gradients: &[0.0; 520],
    }];
    assert!(actual.step_from_consumers(&[&valid, &invalid]).is_err());
    assert_eq!(checkpoint(&actual), good);
    let invalid = [Use::Matmul {
        activations: &[],
        output_gradients: &[],
        batch: usize::MAX,
    }];
    assert!(actual.step_from_consumers(&[&valid, &invalid]).is_err());
    assert_eq!(checkpoint(&actual), good);
    assert!(actual.step_from_consumers(&[&valid]).is_err());
    let mut nonfinite = x.clone();
    nonfinite[259] = f32::NAN;
    let invalid = [Use::Matmul {
        activations: &nonfinite,
        output_gradients: &dy,
        batch: 1,
    }];
    assert!(actual.step_from_consumers(&[&valid, &invalid]).is_err());
    assert_eq!(checkpoint(&actual), good);
    let invalid = [Use::Matmul {
        activations: &x,
        output_gradients: &[f32::INFINITY; 3],
        batch: 1,
    }];
    assert!(actual.step_from_consumers(&[&valid, &invalid]).is_err());
    assert_eq!(checkpoint(&actual), good);
    let mut huge = vec![0.0; 260];
    huge[2] = f32::MAX;
    let invalid = [Use::Matmul {
        activations: &huge,
        output_gradients: &[f32::MAX; 3],
        batch: 1,
    }];
    assert!(actual.step_from_consumers(&[&valid, &invalid]).is_err());
    assert_eq!(checkpoint(&actual), good);
}

fn fixture_outputs(weight: &AdditiveTensor, x: &[f32], ids: &[usize]) -> (Vec<f32>, Vec<f32>) {
    let cols = weight.layout().cols as usize;
    let batch = x.len() / cols;
    let mut y = vec![0.0; batch * weight.layout().rows as usize];
    reference_ternary_matmul(
        x,
        &weight.view().bind_matmul(),
        batch,
        &mut vec![0.0; x.len()],
        &mut y,
    )
    .unwrap();
    let mut e = vec![0.0; ids.len() * cols];
    reference_embed(&weight.view().bind_gather(), ids, &mut e).unwrap();
    (y, e)
}
fn fixture_loss_and_cotangents(
    weight: &AdditiveTensor,
    x: &[f32],
    ids: &[usize],
    target: &(Vec<f32>, Vec<f32>),
) -> (f64, Vec<f32>, Vec<f32>) {
    let (y, e) = fixture_outputs(weight, x, ids);
    let mut loss = 0.0;
    let mut gradients = |values: &[f32], target: &[f32]| {
        values
            .iter()
            .zip(target)
            .map(|(&v, &t)| {
                let residual = v - t;
                loss += f64::from(residual).powi(2);
                2.0 * residual
            })
            .collect::<Vec<_>>()
    };
    let dy = gradients(&y, &target.0);
    let de = gradients(&e, &target.1);
    (loss, dy, de)
}

#[test]
fn actual_tied_matmul_embedding_loss_decreases_and_resumes_exactly() {
    let layout = AdditiveLayout {
        rows: 3,
        cols: 4,
        max_planes: 1,
        basis: Basis::SignedRht {
            block: 4,
            seed: 19,
            domain: 7,
        },
        ..simple().layout()
    };
    let parent = AdditiveTensor::new(
        layout,
        1,
        [Trit::POS, Trit::ZERO, Trit::NEG, Trit::POS].repeat(3),
        vec![0.5; 3],
    )
    .unwrap();
    let mut teacher = parent.clone();
    teacher.replace_scales(vec![0.8, 0.65, 0.9]).unwrap();
    let x = consumer_values(8);
    let ids = [2, 0, 2];
    let target = fixture_outputs(&teacher, &x, &ids);
    let mut p = plan(1);
    if let NativeRecoveryRecipe::ScaleOnly { optimizer } = &mut p.recipe {
        optimizer.lr = 0.02;
        optimizer.weight_decay = 0.0;
    }
    let mut actual = Session::new(p.clone(), vec![parent.clone()]).unwrap();
    let initial = fixture_loss_and_cotangents(&parent, &x, &ids, &target).0;
    let mut resumed: Option<Session> = None;
    for window in 0..4 {
        let (_, dy, de) =
            fixture_loss_and_cotangents(actual.weights().next().unwrap(), &x, &ids, &target);
        let uses = [
            Use::Matmul {
                activations: &x,
                output_gradients: &dy,
                batch: 2,
            },
            Use::Embed {
                ids: &ids,
                output_gradients: &de,
            },
        ];
        actual.step_from_consumers(&[&uses]).unwrap();
        if let Some(state) = &mut resumed {
            state.step_from_consumers(&[&uses]).unwrap();
            assert_eq!(checkpoint(state), checkpoint(&actual));
        }
        if window == 1 {
            resumed = Some(
                Session::resume(p.clone(), vec![parent.clone()], &checkpoint(&actual)).unwrap(),
            );
        }
    }
    let final_loss =
        fixture_loss_and_cotangents(actual.weights().next().unwrap(), &x, &ids, &target).0;
    assert!(
        final_loss < initial,
        "fixture loss: {initial} -> {final_loss}"
    );
    assert_eq!(actual.weights().next().unwrap().trits(), parent.trits());
    eprintln!("NATIVE_TIED_FIXTURE_LOSS initial={initial} final={final_loss}");
}

// Closed-loop common recovery + actual packed execution. The common session
// still owns decoded oracle state; this is not production packed/out-of-core QAT.
fn packed_outputs(
    handle: &dyn tritium_spec::DeviceBuffer,
    x: &[f32],
    ids: &[usize],
) -> (Vec<f32>, Vec<f32>) {
    use tritium_spec::TernaryBackend;
    let cpu = tritium_cpu::CpuBackend::new();
    let mut y = vec![0.0; x.len() / 4 * 3];
    cpu.matmul(tritium_spec::TensorMatmul {
        act: x,
        tensor: handle,
        batch: x.len() / 4,
        transformed_act: &mut vec![0.0; x.len()],
        out: &mut y,
    })
    .unwrap();
    let mut e = vec![0.0; ids.len() * 4];
    cpu.embed_rows(handle, ids, &mut e).unwrap();
    (y, e)
}
fn packed_recovery_step(
    state: &mut Session,
    current: &std::sync::Arc<dyn tritium_spec::DeviceBuffer>,
    x: &[f32],
    ids: &[usize],
    target: &(Vec<f32>, Vec<f32>),
    budget: &tritium_runtime::TensorPayloadBudget,
) -> std::sync::Arc<dyn tritium_spec::DeviceBuffer> {
    let (y, e) = packed_outputs(&**current, x, ids);
    let dy: Vec<_> = y
        .iter()
        .zip(&target.0)
        .map(|(&y, &t)| 2.0 * (y - t))
        .collect();
    let de: Vec<_> = e
        .iter()
        .zip(&target.1)
        .map(|(&y, &t)| 2.0 * (y - t))
        .collect();
    let cpu = tritium_cpu::CpuBackend::new();
    let mut gradient = vec![0.0; cpu.additive_stored_scale_count(&**current).unwrap()];
    let mut row = [0.0; 4];
    cpu.accumulate_additive_matmul_scale_vjp(
        &**current,
        x,
        &dy,
        x.len() / 4,
        &mut row,
        &mut gradient,
    )
    .unwrap();
    cpu.accumulate_additive_embed_scale_vjp(&**current, ids, &de, &mut row, &mut gradient)
        .unwrap();
    let gradient: Vec<_> = gradient.iter().map(|&v| v as f32).collect();
    state.step(&[&gradient]).unwrap();
    cpu.snapshot_additive_scales(current, state.weights().next().unwrap().scales(), budget)
        .unwrap()
        .0
}

#[test]
fn common_scale_recovery_uses_actual_packed_snapshots_with_exact_restart_and_old_graph_isolation() {
    use std::sync::Arc;
    use tritium_spec::{DeviceBuffer, TensorView, TernaryBackend};
    let layout = AdditiveLayout {
        rows: 3,
        cols: 4,
        max_planes: 1,
        basis: Basis::SignedRht {
            block: 4,
            seed: 19,
            domain: 7,
        },
        ..simple().layout()
    };
    let parent = AdditiveTensor::new(
        layout,
        1,
        [Trit::POS, Trit::ZERO, Trit::NEG, Trit::POS].repeat(3),
        vec![0.5; 3],
    )
    .unwrap();
    let mut teacher = parent.clone();
    teacher.replace_scales(vec![0.8, 0.65, 0.9]).unwrap();
    let x = consumer_values(8);
    let ids = [2, 0, 2];
    let target = fixture_outputs(&teacher, &x, &ids);
    let root: Arc<dyn DeviceBuffer> = Arc::from(
        tritium_cpu::CpuBackend::new()
            .upload_tensor(TensorView::Additive(parent.view()))
            .unwrap(),
    );
    let mut current = Arc::clone(&root);
    let budget = tritium_runtime::TensorPayloadBudget::new(36);
    let fork_budget = tritium_runtime::TensorPayloadBudget::new(24);
    let mut p = plan(1);
    if let NativeRecoveryRecipe::ScaleOnly { optimizer } = &mut p.recipe {
        optimizer.lr = 0.02;
        optimizer.weight_decay = 0.0;
    }
    let mut state = Session::new(p.clone(), vec![parent.clone()]).unwrap();
    let mut resumed: Option<(Session, Arc<dyn DeviceBuffer>)> = None;
    let mut old_graph = None;
    let loss = |out: &(Vec<f32>, Vec<f32>)| {
        out.0
            .iter()
            .chain(&out.1)
            .zip(target.0.iter().chain(&target.1))
            .map(|(&y, &t)| f64::from(y - t).powi(2))
            .sum::<f64>()
    };
    let initial = loss(&packed_outputs(&*current, &x, &ids));
    for window in 0..4 {
        current = packed_recovery_step(&mut state, &current, &x, &ids, &target, &budget);
        if let Some((fork, handle)) = &mut resumed {
            *handle = packed_recovery_step(fork, handle, &x, &ids, &target, &fork_budget);
            assert_eq!(checkpoint(fork), checkpoint(&state));
            assert_eq!(
                packed_outputs(&**handle, &x, &ids),
                packed_outputs(&*current, &x, &ids)
            );
        }
        if window == 0 {
            old_graph = Some((Arc::clone(&current), packed_outputs(&*current, &x, &ids)));
        }
        if window == 1 {
            let fork =
                Session::resume(p.clone(), vec![parent.clone()], &checkpoint(&state)).unwrap();
            let handle = tritium_cpu::CpuBackend::new()
                .snapshot_additive_scales(
                    &root,
                    fork.weights().next().unwrap().scales(),
                    &fork_budget,
                )
                .unwrap()
                .0;
            resumed = Some((fork, handle));
        }
    }
    let final_loss = loss(&packed_outputs(&*current, &x, &ids));
    assert!(final_loss < initial);
    let (old, expected) = old_graph.unwrap();
    assert_eq!(packed_outputs(&*old, &x, &ids), expected);
    assert_eq!(
        packed_outputs(&*root, &x, &ids),
        fixture_outputs(&parent, &x, &ids)
    );
    eprintln!("PACKED_SNAPSHOT_RECOVERY_LOSS initial={initial} final={final_loss}");
    drop(old);
    drop(current);
    drop(resumed);
    assert_eq!(budget.used_bytes(), 0);
    assert_eq!(fork_budget.used_bytes(), 0);
}

#[test]
fn staged_scale_error_rolls_back_and_success_matches_direct_step_bits() {
    use tritium_train::pv_tuning::NativeScaleStageError;
    let parents = vec![simple(), simple()];
    let mut actual = Session::new(plan(2), parents.clone()).unwrap();
    let mut direct = Session::new(plan(2), parents).unwrap();
    let before = checkpoint(&actual);
    let n = actual.weights().next().unwrap().scales().len();
    let g = vec![0.25; n];
    let expected_plan = actual.plan_id();
    let error = actual
        .step_staged(&[&g, &g], |candidate| {
            assert_eq!(candidate.plan_id(), expected_plan);
            assert_eq!(candidate.proposed_update(), 1);
            assert_eq!(
                candidate.owners().map(|(name, _)| name).collect::<Vec<_>>(),
                vec!["owner-0000", "owner-0001"]
            );
            assert!(candidate.owners().all(|(_, scales)| {
                scales
                    .iter()
                    .any(|v| v.to_bits() != f32::from_bits(0x3f00_0001).to_bits())
            }));
            Err::<(), _>("original staging error")
        })
        .unwrap_err();
    assert_eq!(
        error,
        NativeScaleStageError::Stage("original staging error")
    );
    assert_eq!(checkpoint(&actual), before);
    let (changed, staged) = actual
        .step_staged(&[&g, &g], |candidate| {
            Ok::<_, String>(
                candidate
                    .owners()
                    .map(|(_, scales)| scales.to_vec())
                    .collect::<Vec<_>>(),
            )
        })
        .unwrap();
    assert_eq!(changed, direct.step(&[&g, &g]).unwrap());
    assert_eq!(checkpoint(&actual), checkpoint(&direct));
    assert_eq!(
        actual
            .weights()
            .map(|w| w.scales().to_vec())
            .collect::<Vec<_>>(),
        staged
    );
}

#[test]
fn stage_unwind_and_preflight_errors_preserve_the_last_good_state() {
    let mut actual = Session::new(plan(1), vec![simple()]).unwrap();
    let before = checkpoint(&actual);
    let n = actual.weights().next().unwrap().scales().len();
    let gradient = vec![0.25; n];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        actual.step_staged(&[&gradient], |_| -> Result<(), String> {
            panic!("injected staging unwind")
        })
    }));
    assert!(result.is_err());
    assert_eq!(checkpoint(&actual), before);
    let called = std::cell::Cell::new(false);
    for bad in [f32::NAN, 1e20] {
        assert!(
            actual
                .step_staged(&[&vec![bad; n]], |_| {
                    called.set(true);
                    Ok::<_, String>(())
                })
                .is_err()
        );
        assert!(!called.get());
        assert_eq!(checkpoint(&actual), before);
    }
}
