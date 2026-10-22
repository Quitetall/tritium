#![cfg(feature = "native-cpu")]
//! Real packed binding/rollback fixtures; not model/source/release acceptance.
use std::sync::Arc;
use tritium_core::Trit;
use tritium_cpu::CpuBackend;
use tritium_format::AdditiveTensor;
use tritium_runtime::TensorPayloadBudget;
use tritium_schema::*;
use tritium_spec::{BackendError, DeviceBuffer, TensorMatmul, TensorView, TernaryBackend};
use tritium_train::pv_tuning::{
    NativeCpuScaleSession as Bound, NativeScaleContribution as Use,
    NativeScaleRefinementSession as Session, NativeScaleStageError,
};

fn plan(owners: usize) -> NativeRefinementPlan {
    NativeRefinementPlan {
        parent: NativeRefinementParent {
            package: PackageId::from_bytes([1; 32]),
            level: "explicit-parent".into(),
            model: ModelId::from_bytes([2; 32]),
            source: OriginalSourceId::from_bytes([3; 32]),
            evidence: EvidenceReference {
                run: "synthetic".into(),
                span: "parent".into(),
                root: [4; 32],
            },
        },
        child_level: "child".into(),
        owners: (0..owners).map(|i| format!("owner-{i:04}")).collect(),
        recipe: NativeRecoveryRecipe::ScaleOnly {
            optimizer: NativeAdamWRecipe {
                lr: 0.02,
                beta1: 0.9,
                beta2: 0.99,
                eps: 1e-8,
                weight_decay: 0.0,
            },
        },
        objective: [5; 32],
        teacher_source: None,
        training: (0..4)
            .map(|i| NativeRefinementWindow {
                content: [10 + i; 32],
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
fn tensor() -> AdditiveTensor {
    AdditiveTensor::new(
        AdditiveLayout {
            rows: 3,
            cols: 4,
            tile: 256,
            group: 32,
            max_planes: 1,
            allocation: PlaneAllocation::Uniform,
            codec: PlaneCodec::D2,
            law: ADMITTED_LAWS[1].law,
            basis: Basis::SignedRht {
                block: 4,
                seed: 19,
                domain: 7,
            },
            transport: Transport::Raw,
        },
        1,
        [Trit::POS, Trit::ZERO, Trit::NEG, Trit::POS].repeat(3),
        vec![0.5; 3],
    )
    .unwrap()
}
fn roots(parents: &[AdditiveTensor]) -> Vec<(String, Arc<dyn DeviceBuffer>)> {
    parents
        .iter()
        .enumerate()
        .map(|(i, p)| {
            (
                format!("owner-{i:04}"),
                Arc::from(
                    CpuBackend::new()
                        .upload_tensor(TensorView::Additive(p.view()))
                        .unwrap(),
                ),
            )
        })
        .collect()
}
fn bytes(binding: &Bound) -> Vec<u8> {
    binding.session().checkpoint_bytes().unwrap()
}
fn outputs(handle: &dyn DeviceBuffer, x: &[f32], ids: &[usize]) -> (Vec<f32>, Vec<f32>) {
    let cpu = CpuBackend::new();
    let mut y = vec![0.0; x.len() / 4 * 3];
    let mut e = vec![0.0; ids.len() * 4];
    cpu.matmul(TensorMatmul {
        act: x,
        tensor: handle,
        batch: x.len() / 4,
        transformed_act: &mut vec![0.0; x.len()],
        out: &mut y,
    })
    .unwrap();
    cpu.embed_rows(handle, ids, &mut e).unwrap();
    (y, e)
}

#[test]
fn late_owner_budget_failure_preserves_all_state_handles_and_reservations() {
    let parents = vec![tensor(), tensor()];
    let state = Session::new(plan(2), parents.clone()).unwrap();
    let budget = TensorPayloadBudget::new(36); //24initial +12room:owner0 succeeds,owner1 must fail.
    let mut bound = Bound::bind(state, roots(&parents), budget.clone()).unwrap();
    let before = bytes(&bound);
    let old: Vec<_> = bound.handles().map(|(_, h)| Arc::clone(h)).collect();
    assert_eq!(budget.used_bytes(), 24);
    let result = bound.step(&[&[0.25; 3], &[0.25; 3]]);
    assert!(matches!(
        result,
        Err(NativeScaleStageError::Stage(
            BackendError::TensorPayloadBudgetExceeded {
                requested: 48,
                limit: 36
            }
        ))
    ));
    assert_eq!(bytes(&bound), before);
    assert_eq!(budget.used_bytes(), 24);
    for ((_, current), old) in bound.handles().zip(&old) {
        assert!(Arc::ptr_eq(current, old));
    }
    drop(bound);
    assert_eq!(budget.used_bytes(), 24);
    drop(old);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn binding_checks_names_actual_codes_layout_and_scale_versions() {
    let parent = tensor();
    let state = || Session::new(plan(1), vec![parent.clone()]).unwrap();
    let budget = TensorPayloadBudget::new(1000);
    let mut named = roots(std::slice::from_ref(&parent));
    named[0].0 = "wrong-owner".into();
    assert!(Bound::bind(state(), named, budget.clone()).is_err());
    assert_eq!(budget.used_bytes(), 0);
    let mut changed = parent.clone();
    changed.replace_scales(vec![0.75; 3]).unwrap(); //Binding pins the declared common scales,not an arbitrary supplied overlay.
    let bound = Bound::bind(state(), roots(&[changed]), budget.clone()).unwrap();
    assert_eq!(bound.handles().len(), 1);
    drop(bound);
    assert_eq!(budget.used_bytes(), 0);
    let changed =
        AdditiveTensor::new(parent.layout(), 1, vec![Trit::NEG; 12], vec![0.5; 3]).unwrap();
    assert!(Bound::bind(state(), roots(&[changed]), budget.clone()).is_err());
    assert_eq!(budget.used_bytes(), 0);
    let changed = AdditiveTensor::new(
        AdditiveLayout {
            basis: Basis::Identity,
            ..parent.layout()
        },
        1,
        parent.trits().to_vec(),
        vec![0.5; 3],
    )
    .unwrap();
    assert!(Bound::bind(state(), roots(&[changed]), budget.clone()).is_err());
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn actual_packed_consumers_commit_with_state_restart_exactly_and_keep_old_graphs() {
    let parent = tensor();
    let mut teacher = parent.clone();
    teacher.replace_scales(vec![0.8, 0.65, 0.9]).unwrap();
    let p = plan(1);
    let named = roots(std::slice::from_ref(&parent));
    let teacher_handle = roots(&[teacher]).pop().unwrap().1;
    let x = [0.25, -0.5, 0.125, 1.0, 0.5, 0.125, -0.25, 0.75];
    let ids = [2, 0, 2];
    let target = outputs(&*teacher_handle, &x, &ids);
    let loss = |out: &(Vec<f32>, Vec<f32>)| {
        out.0
            .iter()
            .chain(&out.1)
            .zip(target.0.iter().chain(&target.1))
            .map(|(&a, &b)| f64::from(a - b).powi(2))
            .sum::<f64>()
    };
    let budget = TensorPayloadBudget::new(36);
    let fork_budget = TensorPayloadBudget::new(24);
    let mut bound = Bound::bind(
        Session::new(p.clone(), vec![parent.clone()]).unwrap(),
        named,
        budget.clone(),
    )
    .unwrap();
    let mut fork = None;
    let old = Arc::clone(bound.handles().next().unwrap().1);
    let old_out = outputs(&*old, &x, &ids);
    let initial = loss(&old_out);
    for window in 0..4 {
        let out = outputs(&**bound.handles().next().unwrap().1, &x, &ids);
        let dy: Vec<_> = out
            .0
            .iter()
            .zip(&target.0)
            .map(|(&a, &b)| 2.0 * (a - b))
            .collect();
        let de: Vec<_> = out
            .1
            .iter()
            .zip(&target.1)
            .map(|(&a, &b)| 2.0 * (a - b))
            .collect();
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
        bound.step_from_consumers(&[&uses]).unwrap();
        if let Some(other) = &mut fork {
            let other: &mut Bound = other;
            other.step_from_consumers(&[&uses]).unwrap();
            assert_eq!(bytes(other), bytes(&bound));
            assert_eq!(
                outputs(&**other.handles().next().unwrap().1, &x, &ids),
                outputs(&**bound.handles().next().unwrap().1, &x, &ids)
            );
        }
        if window == 1 {
            fork = Some(
                Bound::bind(
                    Session::resume(p.clone(), vec![parent.clone()], &bytes(&bound)).unwrap(),
                    roots(std::slice::from_ref(&parent)),
                    fork_budget.clone(),
                )
                .unwrap(),
            );
        }
    }
    let final_loss = loss(&outputs(&**bound.handles().next().unwrap().1, &x, &ids));
    assert!(final_loss < initial);
    assert_eq!(outputs(&*old, &x, &ids), old_out);
    eprintln!("ATOMIC_PACKED_RECOVERY_LOSS initial={initial} final={final_loss}");
    drop(old);
    drop(bound);
    drop(fork);
    assert_eq!(budget.used_bytes(), 0);
    assert_eq!(fork_budget.used_bytes(), 0);
}

#[test]
fn direct_atomic_step_matches_common_checkpoint_and_numeric_failures_do_not_stage() {
    let parent = tensor();
    let mut direct = Session::new(plan(1), vec![parent.clone()]).unwrap();
    let budget = TensorPayloadBudget::new(24);
    let mut bound = Bound::bind(
        Session::new(plan(1), vec![parent.clone()]).unwrap(),
        roots(&[parent]),
        budget.clone(),
    )
    .unwrap();
    let before = bytes(&bound);
    let used = budget.used_bytes();
    let old = Arc::clone(bound.handles().next().unwrap().1);
    for bad in [f32::NAN, 1e20] {
        assert!(bound.step(&[&[bad; 3]]).is_err());
        assert_eq!(bytes(&bound), before);
        assert_eq!(budget.used_bytes(), used);
        assert!(Arc::ptr_eq(&old, bound.handles().next().unwrap().1));
    }
    drop(old);
    for gradient in [0.25, -0.5, 0.125, 0.0] {
        assert_eq!(
            bound.step(&[&[gradient; 3]]).unwrap(),
            direct.step(&[&[gradient; 3]]).unwrap()
        );
        assert_eq!(bytes(&bound), direct.checkpoint_bytes().unwrap());
    }
    let before = bytes(&bound);
    let used = budget.used_bytes();
    assert!(bound.step(&[&[0.25; 3]]).is_err());
    assert_eq!(bytes(&bound), before);
    assert_eq!(budget.used_bytes(), used);
}

#[test]
fn packed_consumer_atomic_steps_match_reference_across_cpu_profiles() {
    use tritium_core::{PerTilePlan, UniformPlan};
    let mut cases = 0;
    for admitted in ADMITTED_LAWS {
        for group in [32, 128] {
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
                            transport: Transport::Raw,
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
                            .map(|i| [Trit::ZERO, Trit::POS, Trit::NEG, Trit::POS][i % 4])
                            .collect();
                        let parent = if mode == 0 {
                            AdditiveTensor::new(layout, admitted.max_planes, codes, vec![0.5; s])
                                .unwrap()
                        } else {
                            AdditiveTensor::new_per_tile(layout, counts, codes, vec![0.5; s])
                                .unwrap()
                        };
                        let mut direct = Session::new(plan(1), vec![parent.clone()]).unwrap();
                        let budget = TensorPayloadBudget::new(1_000_000);
                        let mut bound = Bound::bind(
                            Session::new(plan(1), vec![parent.clone()]).unwrap(),
                            roots(&[parent]),
                            budget.clone(),
                        )
                        .unwrap();
                        let cols = layout.cols as usize;
                        let x: Vec<_> = (0..2 * cols)
                            .map(|i| (i % 7) as f32 * 0.125 - 0.375)
                            .collect();
                        let dy = [0.25, -0.5, 0.75, 0.5, 0.25, -0.125];
                        let ids = [2, 0, 2];
                        let de = vec![0.25; 3 * cols];
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
                        for _ in 0..4 {
                            assert_eq!(
                                bound.step_from_consumers(&[&uses]).unwrap(),
                                direct.step_from_consumers(&[&uses]).unwrap()
                            );
                            assert_eq!(bytes(&bound), direct.checkpoint_bytes().unwrap());
                            assert!(
                                CpuBackend::new()
                                    .additive_equals_view(
                                        &**bound.handles().next().unwrap().1,
                                        direct.weights().next().unwrap().view()
                                    )
                                    .unwrap()
                            );
                        }
                        drop(bound);
                        assert_eq!(budget.used_bytes(), 0);
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 216);
}

struct Switching {
    a: Arc<dyn DeviceBuffer>,
    b: Arc<dyn DeviceBuffer>,
    flag: Arc<std::sync::atomic::AtomicBool>,
}
impl DeviceBuffer for Switching {
    fn len_bytes(&self) -> usize {
        self.a.len_bytes()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        if self.flag.load(std::sync::atomic::Ordering::Acquire) {
            self.b.as_any()
        } else {
            self.a.as_any()
        }
    }
}
#[test]
fn changed_forwarded_root_cannot_commit_bound_state_or_handle_set() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let parent = tensor();
    let changed = AdditiveTensor::new(
        parent.layout(),
        1,
        vec![Trit::NEG; 12],
        parent.scales().to_vec(),
    )
    .unwrap();
    let a = roots(std::slice::from_ref(&parent)).pop().unwrap().1;
    let b = roots(&[changed]).pop().unwrap().1;
    assert_eq!(a.len_bytes(), b.len_bytes());
    let flag = Arc::new(AtomicBool::new(false));
    let root: Arc<dyn DeviceBuffer> = Arc::new(Switching {
        a,
        b,
        flag: flag.clone(),
    });
    let budget = TensorPayloadBudget::new(24);
    let mut bound = Bound::bind(
        Session::new(plan(1), vec![parent]).unwrap(),
        vec![("owner-0000".into(), root)],
        budget.clone(),
    )
    .unwrap();
    let before = bytes(&bound);
    let old = Arc::clone(bound.handles().next().unwrap().1);
    flag.store(true, Ordering::Release);
    assert!(matches!(
        bound.step(&[&[0.25; 3]]),
        Err(NativeScaleStageError::Stage(BackendError::InvalidInput(_)))
    ));
    assert_eq!(bytes(&bound), before);
    assert!(Arc::ptr_eq(&old, bound.handles().next().unwrap().1));
    assert_eq!(budget.used_bytes(), 12);
    let uses = [Use::Matmul {
        activations: &[0.25; 4],
        output_gradients: &[0.25; 3],
        batch: 1,
    }];
    assert!(bound.step_from_consumers(&[&uses]).is_err());
    assert_eq!(bytes(&bound), before);
    assert_eq!(budget.used_bytes(), 12);
    flag.store(false, Ordering::Release);
    bound.step(&[&[0.25; 3]]).unwrap();
    assert_eq!(bound.session().completed_updates(), 1);
    drop(old);
    drop(bound);
    assert_eq!(budget.used_bytes(), 0);
}
