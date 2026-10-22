//! Small CPU numerical/checkpoint fixtures, not model-quality or release evidence.
use half::f16;
use tritium_core::{PerTilePlan, Trit, UniformPlan};
use tritium_format::AdditiveTensor;
use tritium_schema::*;
use tritium_train::optim::{AdamW, Optimizer};
use tritium_train::pv_tuning::NativeScaleRefinementSession as Session;

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
