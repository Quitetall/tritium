//! Actual native numerical state tests; not model or release qualification.
use tritium_core::{PerTilePlan, Trit, UniformPlan};
use tritium_format::AdditiveTensor;
use tritium_schema::*;
use tritium_train::pv_tuning::{
    NativePvRefinementSession as Session, NativePvStageError, NativeScaleRefinementSession,
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

fn active(weight: &AdditiveTensor) -> usize {
    if weight.plane_count() == 0 {
        0
    } else {
        weight.view().plane_trits(0).unwrap().len()
    }
}
fn step(session: &mut Session, value: f32) {
    let scales: Vec<_> = session
        .weights()
        .map(|w| vec![value; w.scales().len()])
        .collect();
    let weights: Vec<_> = session.weights().map(|w| vec![value; active(w)]).collect();
    let stats = session
        .step(
            &scales.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            &weights.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        )
        .unwrap();
    for owner in stats.owners {
        assert!(owner.surrogate_after <= owner.surrogate_before);
        assert!(owner.relative_change <= 0.900_001);
    }
}
fn reseal(bytes: &mut [u8]) {
    let n = bytes.len() - 32;
    let hash = blake3::hash(&bytes[..n]);
    bytes[n..].copy_from_slice(hash.as_bytes());
}

#[test]
fn exact_resume_and_real_code_updates_across_native_profiles() {
    let mut cases = 0;
    let mut changed = 0;
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
                            let original = parent.clone();
                            let mut a = Session::new(plan(1), vec![parent.clone()]).unwrap();
                            let initial = a.checkpoint_bytes().unwrap();
                            assert_eq!(initial.len() as u64, a.size_ledger().checkpoint_bytes);
                            for value in [0.125, -0.25, 0.5, 0.0] {
                                let saved = a.checkpoint_bytes().unwrap();
                                let mut b =
                                    Session::resume(plan(1), vec![parent.clone()], &saved).unwrap();
                                assert_eq!(b.checkpoint_bytes().unwrap(), saved);
                                step(&mut a, value);
                                step(&mut b, value);
                                assert_eq!(
                                    a.checkpoint_bytes().unwrap(),
                                    b.checkpoint_bytes().unwrap()
                                );
                                let current = a.weights().next().unwrap();
                                assert_eq!(current.layout(), original.layout());
                                assert_eq!(current.tile_counts(), original.tile_counts());
                                if codec == PlaneCodec::S34 {
                                    assert!(current.trits().as_chunks::<4>().0.iter().all(
                                        |block| {
                                            block.iter().filter(|code| code.is_zero()).count() == 1
                                        }
                                    ));
                                }
                            }
                            changed += usize::from(
                                a.weights().next().unwrap().trits() != original.trits(),
                            );
                            assert_eq!(parent, original);
                            assert_eq!(a.completed_updates(), 4);
                            let saved = a.checkpoint_bytes().unwrap();
                            let mut resumed =
                                Session::resume(plan(1), vec![parent], &saved).unwrap();
                            assert!(resumed.step(&[&[]], &[&[]]).is_err());
                            assert_eq!(resumed.checkpoint_bytes().unwrap(), saved);
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(changed > 0);
    assert_eq!(cases, 648);
    eprintln!("NATIVE_PV_NATIVE_PROFILES cases={cases} changed_final_codes={changed}");
}

fn tiny(sparse: bool) -> AdditiveTensor {
    AdditiveTensor::new(
        AdditiveLayout {
            rows: 1,
            cols: 4,
            tile: 256,
            group: 32,
            max_planes: 1,
            allocation: PlaneAllocation::Uniform,
            codec: if sparse {
                PlaneCodec::S34
            } else {
                PlaneCodec::D2
            },
            law: ADMITTED_LAWS[3].law,
            basis: Basis::Identity,
            transport: Transport::Raw,
        },
        1,
        if sparse {
            vec![Trit::ZERO, Trit::POS, Trit::POS, Trit::POS]
        } else {
            vec![Trit::POS; 4]
        },
        vec![1.0],
    )
    .unwrap()
}
fn decode(weight: &AdditiveTensor) -> Vec<f32> {
    let mut row = vec![0.0; 4];
    weight.view().dequant_row_into(0, &mut row).unwrap();
    row
}

#[test]
fn actual_tiny_dense_and_sparse_losses_improve_without_dense_master() {
    for sparse in [false, true] {
        let parent = tiny(sparse);
        let target = if sparse {
            [1.0, 0.0, 1.0, 1.0]
        } else {
            [0.0; 4]
        };
        let before = decode(&parent);
        let g: Vec<_> = before
            .iter()
            .zip(target)
            .map(|(a, b)| 2.0 * (a - b))
            .collect();
        let scale: f32 = g
            .iter()
            .zip(parent.trits())
            .map(|(g, code)| g * code.to_f32())
            .sum();
        let mut p = plan(1);
        p.training[0].weight = 1.0;
        if let NativeRecoveryRecipe::AlternatingPv {
            code_optimizer,
            max_code_change_fraction,
            max_relative_code_change,
            ..
        } = &mut p.recipe
        {
            code_optimizer.lr = if sparse { 1.0 } else { 0.6 };
            *max_code_change_fraction = 1.0;
            *max_relative_code_change = None;
        }
        let mut s = Session::new(p, vec![parent]).unwrap();
        let stats = s.step(&[&[scale]], &[&g]).unwrap();
        assert!(stats.owners[0].changed_units > 0);
        let after = decode(s.weights().next().unwrap());
        let loss = |values: &[f32]| {
            values
                .iter()
                .zip(target)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
        };
        assert!(loss(&after) < loss(&before));
    }
}

#[test]
fn all_owner_numeric_errors_stage_refusal_and_unwind_are_atomic() {
    let p = plan(2);
    let parents = vec![tiny(false), tiny(true)];
    let mut s = Session::new(p.clone(), parents.clone()).unwrap();
    let before = s.checkpoint_bytes().unwrap();
    assert!(
        s.step(&[&[0.0], &[0.0]], &[&[0.0; 4], &[f32::NAN; 4]])
            .is_err()
    );
    assert_eq!(s.checkpoint_bytes().unwrap(), before);
    assert!(
        s.step(&[&[0.0], &[0.0]], &[&[0.0; 4], &[f32::MAX; 4]])
            .is_err()
    );
    assert_eq!(s.checkpoint_bytes().unwrap(), before);
    let error = s
        .step_staged(
            &[&[0.125], &[0.125]],
            &[&[0.25; 4], &[0.25; 4]],
            |candidate| {
                assert_eq!(candidate.plan_id(), s_plan_id(&p));
                assert_eq!(candidate.owners().len(), 2);
                assert_eq!(candidate.stats().completed_updates, 1);
                Err::<(), _>("staging disk error")
            },
        )
        .unwrap_err();
    assert_eq!(error, NativePvStageError::Stage("staging disk error"));
    assert_eq!(s.checkpoint_bytes().unwrap(), before);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        s.step_staged(
            &[&[0.0], &[0.0]],
            &[&[0.0; 4], &[0.0; 4]],
            |_| -> Result<(), ()> { panic!("injected stage unwind") },
        )
    }));
    assert!(result.is_err());
    assert_eq!(s.checkpoint_bytes().unwrap(), before);
    let mut expected = Session::new(p, parents).unwrap();
    step(&mut s, 0.125);
    step(&mut expected, 0.125);
    assert_eq!(
        s.checkpoint_bytes().unwrap(),
        expected.checkpoint_bytes().unwrap()
    );
}
fn s_plan_id(p: &NativeRefinementPlan) -> NativeRefinementPlanId {
    tritium_evidence::native_refinement_plan_id(p).unwrap()
}

#[test]
fn exact_caps_compact_state_and_recipe_separation() {
    let parent = tensor(1, PlaneCodec::D2, Basis::Identity, 1, Transport::Raw);
    let s = Session::new(plan(1), vec![parent.clone()]).unwrap();
    let ledger = s.size_ledger();
    let a = active(&parent) as u64;
    let scales = parent.scales().len() as u64;
    assert_eq!(
        ledger.state_payload_bytes,
        parent.trits().len() as u64
            + parent.view().auxiliary_buffer_len_bytes().unwrap() as u64
            + 12 * scales
            + 8 * a
    );
    assert!(a < parent.layout().rows * parent.layout().cols);
    for field in 0..3 {
        let mut p = plan(1);
        p.limits.max_state_bytes = ledger.state_payload_bytes;
        p.limits.max_workspace_bytes = ledger.update_workspace_bytes;
        p.limits.max_checkpoint_bytes = ledger.checkpoint_bytes;
        assert!(Session::new(p.clone(), vec![parent.clone()]).is_ok());
        match field {
            0 => p.limits.max_state_bytes -= 1,
            1 => p.limits.max_workspace_bytes -= 1,
            _ => p.limits.max_checkpoint_bytes -= 1,
        };
        assert!(Session::new(p, vec![parent.clone()]).is_err());
    }
    assert!(NativeScaleRefinementSession::new(plan(1), vec![parent.clone()]).is_err());
    let mut p = plan(1);
    if let NativeRecoveryRecipe::AlternatingPv {
        continuous_optimizer,
        ..
    } = p.recipe
    {
        p.recipe = NativeRecoveryRecipe::ScaleOnly {
            optimizer: continuous_optimizer,
        };
    }
    assert!(Session::new(p.clone(), vec![parent.clone()]).is_err());
    let scale = NativeScaleRefinementSession::new(p, vec![parent.clone()])
        .unwrap()
        .checkpoint_bytes()
        .unwrap();
    assert!(Session::resume(plan(1), vec![parent], &scale).is_err());
    let pruned = tensor(1, PlaneCodec::S34, Basis::Identity, 2, Transport::Raw);
    assert_eq!(active(&pruned), 0);
    let mut s = Session::new(plan(1), vec![pruned]).unwrap();
    let stats = s.step(&[&[]], &[&[]]).unwrap();
    assert_eq!(stats.owners[0].selected_units, 0);
}

#[test]
fn checkpoint_checks_every_byte_and_resealed_invalid_states() {
    let parent = tiny(true);
    let mut s = Session::new(plan(1), vec![parent.clone()]).unwrap();
    let initial = s.checkpoint_bytes().unwrap();
    let mut valid_but_changed = initial.clone();
    valid_but_changed[93..97].copy_from_slice(&[1, 0, 1, 1]);
    reseal(&mut valid_but_changed);
    assert!(Session::resume(plan(1), vec![parent.clone()], &valid_but_changed).is_err());
    // Header ends at85. Trit-count8 then four codes. Scale-count8 then f32 scale.
    for (offset, bytes) in [
        (93, vec![2]),
        (93, vec![1]),
        (105, (-0.0f32).to_bits().to_le_bytes().to_vec()),
        (109, (1.0f32).to_bits().to_le_bytes().to_vec()),
    ] {
        let mut forged = initial.clone();
        forged[offset..offset + bytes.len()].copy_from_slice(&bytes);
        reseal(&mut forged);
        assert!(Session::resume(plan(1), vec![parent.clone()], &forged).is_err());
    }
    step(&mut s, 0.125);
    let saved = s.checkpoint_bytes().unwrap();
    for (offset, value) in [
        (109, f32::MAX),
        (113, f32::MAX),
        (125, f32::NAN),
        (141, -0.0),
        (141, -1.0),
        (141, f32::MAX),
    ] {
        let mut forged = saved.clone();
        forged[offset..offset + 4].copy_from_slice(&value.to_bits().to_le_bytes());
        reseal(&mut forged);
        assert!(Session::resume(plan(1), vec![parent.clone()], &forged).is_err());
    }
    for i in 0..saved.len() {
        let mut corrupt = saved.clone();
        corrupt[i] ^= 1;
        assert!(Session::resume(plan(1), vec![parent.clone()], &corrupt).is_err());
    }
    for i in 0..saved.len() {
        assert!(Session::resume(plan(1), vec![parent.clone()], &saved[..i]).is_err());
    }
    let mut extra = saved.clone();
    extra.push(0);
    assert!(Session::resume(plan(1), vec![parent.clone()], &extra).is_err());
    for offset in [4, 5, 37, 69, 77, 85, 97, 117] {
        let mut forged = saved.clone();
        forged[offset] ^= 127;
        reseal(&mut forged);
        assert!(Session::resume(plan(1), vec![parent.clone()], &forged).is_err());
    }
    let mut other = plan(1);
    other.training[1].weight = 2.0;
    assert!(Session::resume(other, vec![parent.clone()], &saved).is_err());
    assert!(Session::resume(plan(1), vec![tiny(false)], &saved).is_err());
}

#[test]
fn format_code_replacement_checks_full_structure_and_preserves_original_on_error() {
    let mut weight = tiny(true);
    let original = weight.clone();
    assert!(weight.replace_trits(vec![Trit::POS; 4]).is_err());
    assert_eq!(weight, original);
    assert!(weight.replace_trits(vec![Trit::POS; 3]).is_err());
    assert_eq!(weight, original);
    weight
        .replace_trits(vec![Trit::POS, Trit::ZERO, Trit::POS, Trit::POS])
        .unwrap();
    assert_eq!(weight.scales(), original.scales());
    assert_eq!(weight.layout(), original.layout());
    let mut weight = tensor(0, PlaneCodec::S34, Basis::Identity, 1, Transport::Raw);
    let original = weight.clone();
    let mut codes = weight.trits().to_vec();
    let last = codes.len() - 1;
    codes[last] = Trit::ZERO;
    assert!(weight.replace_trits(codes).is_err());
    assert_eq!(weight, original);
}

#[test]
fn uniform_f16_native_pv_matches_common_legacy_algorithm() {
    use half::f16;
    use tritium_train::{
        AdamW, PvTernaryPlane, PvTernaryStructure, PvTernaryWeight, PvTuningConfig, PvTuningSession,
    };
    for sparse in [false, true] {
        let codes: Vec<_> = (0..96)
            .map(|i| {
                if sparse {
                    [Trit::ZERO, Trit::NEG, Trit::POS, Trit::NEG][i % 4]
                } else {
                    [Trit::NEG, Trit::ZERO, Trit::POS][i % 3]
                }
            })
            .collect();
        let scales = vec![0.5, 0.25, 0.125];
        let native = AdditiveTensor::new(
            AdditiveLayout {
                rows: 1,
                cols: 32,
                tile: 256,
                group: 32,
                max_planes: 3,
                allocation: PlaneAllocation::Uniform,
                codec: if sparse {
                    PlaneCodec::S34
                } else {
                    PlaneCodec::D2
                },
                law: ADMITTED_LAWS[0].law,
                basis: Basis::Identity,
                transport: Transport::Raw,
            },
            3,
            codes.clone(),
            scales.clone(),
        )
        .unwrap();
        let legacy = PvTernaryWeight::new(
            1,
            32,
            32,
            if sparse {
                PvTernaryStructure::S34
            } else {
                PvTernaryStructure::Dense
            },
            (0..3)
                .map(|p| {
                    PvTernaryPlane::new(
                        codes[p * 32..(p + 1) * 32]
                            .iter()
                            .map(|c| c.get())
                            .collect(),
                        vec![f16::from_f32(scales[p])],
                    )
                })
                .collect(),
        )
        .unwrap();
        let mut p = plan(1);
        for window in &mut p.training {
            window.weight = 1.0;
        }
        let convert = |r: NativeAdamWRecipe| AdamW {
            lr: r.lr,
            beta1: r.beta1,
            beta2: r.beta2,
            eps: r.eps,
            weight_decay: r.weight_decay,
        };
        let NativeRecoveryRecipe::AlternatingPv {
            continuous_optimizer,
            code_optimizer,
            max_code_change_fraction,
            max_relative_code_change,
            ..
        } = &mut p.recipe
        else {
            unreachable!()
        };
        *max_relative_code_change = None;
        let cfg = PvTuningConfig::builder(convert(*continuous_optimizer), convert(*code_optimizer))
            .max_code_change_fraction(*max_code_change_fraction)
            .build()
            .unwrap();
        let mut a = Session::new(p, vec![native]).unwrap();
        let mut b = PvTuningSession::new(legacy, cfg).unwrap();
        for t in 1..=4 {
            let g: Vec<_> = (0..32)
                .map(|i| (((i * 3 + t) % 7) as f32 - 3.0) * 0.125)
                .collect();
            let scales: Vec<f32> = a
                .weights()
                .next()
                .unwrap()
                .trits()
                .chunks(32)
                .map(|codes| {
                    codes
                        .iter()
                        .zip(&g)
                        .map(|(c, &g)| f64::from(c.get()) * f64::from(g))
                        .sum::<f64>() as f32
                })
                .collect();
            let actual = a.step(&[&scales], &[&g]).unwrap();
            let expected = b.step(&g, t as u64).unwrap();
            let current = a.weights().next().unwrap();
            assert_eq!(
                current.trits().iter().map(|c| c.get()).collect::<Vec<_>>(),
                b.weight()
                    .planes()
                    .iter()
                    .flat_map(|p| p.trits().iter().copied())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                current.scales(),
                b.weight()
                    .planes()
                    .iter()
                    .flat_map(|p| p.scales().iter().map(|s| s.to_f32()))
                    .collect::<Vec<_>>()
            );
            let got = actual.owners[0];
            assert_eq!(got.changed_units, expected.changed_code_units());
            assert_eq!(got.surrogate_before, expected.v_surrogate_before());
            assert_eq!(got.surrogate_after, expected.v_surrogate_after());
        }
    }
}

#[test]
fn window_weight_applies_once_and_nonfinite_native_proposals_roll_back() {
    let mut p = plan(1);
    p.training[0].weight = 2.0;
    let mut q = p.clone();
    q.training[0].weight = 1.0;
    let mut a = Session::new(p, vec![tiny(false)]).unwrap();
    let mut b = Session::new(q, vec![tiny(false)]).unwrap();
    assert_eq!(
        a.step(&[&[0.25]], &[&[0.125; 4]]).unwrap(),
        b.step(&[&[0.5]], &[&[0.25; 4]]).unwrap()
    );
    let left = a.checkpoint_bytes().unwrap();
    let right = b.checkpoint_bytes().unwrap();
    assert_eq!(&left[85..left.len() - 32], &right[85..right.len() - 32]);
    // Checksums differ because plan/parent identities differ; compare body only.
    let mut p = plan(1);
    if let NativeRecoveryRecipe::AlternatingPv { code_optimizer, .. } = &mut p.recipe {
        code_optimizer.lr = f32::MAX;
        code_optimizer.eps = f32::MIN_POSITIVE;
        code_optimizer.beta1 = 0.0;
        code_optimizer.beta2 = 0.0;
        code_optimizer.weight_decay = 0.0;
    }
    let mut s = Session::new(p, vec![tiny(false)]).unwrap();
    let before = s.checkpoint_bytes().unwrap();
    assert!(s.step(&[&[0.0]], &[&[1e-30; 4]]).is_err());
    assert_eq!(s.checkpoint_bytes().unwrap(), before);
}

#[test]
fn corrected_code_variance_overflow_and_zero_adjoint_decay_are_not_silenced() {
    let mut s = Session::new(plan(1), vec![tiny(false)]).unwrap();
    let before = s.checkpoint_bytes().unwrap();
    let error = s.step(&[&[0.0]], &[&[2e19; 4]]).unwrap_err();
    assert!(error.to_string().contains("bias-corrected"));
    assert_eq!(s.checkpoint_bytes().unwrap(), before);
    let stats = s.step(&[&[0.0]], &[&[0.0; 4]]).unwrap();
    assert_eq!(stats.changed_scales, 1);
    assert!(s.weights().next().unwrap().scales()[0] < 1.0);
}
