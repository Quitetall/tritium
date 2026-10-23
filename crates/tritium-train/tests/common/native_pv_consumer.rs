//! Small synthetic native P/V fixtures, not authenticated training data.
use tritium_core::{
    PerTilePlan, Trit, UniformPlan, accumulate_embed_scale_vjp, accumulate_embed_weight_vjp,
    accumulate_ternary_matmul_scale_vjp, accumulate_ternary_matmul_weight_vjp,
};
use tritium_format::AdditiveTensor;
use tritium_schema::*;
use tritium_train::pv_tuning::NativePvContribution as Use;

pub(super) fn plan(owners: usize) -> NativeRefinementPlan {
    let optimizer = NativeAdamWRecipe {
        lr: 0.01,
        beta1: 0.9,
        beta2: 0.99,
        eps: 1e-8,
        weight_decay: 0.01,
    };
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
            continuous_optimizer: optimizer,
            code_optimizer: NativeAdamWRecipe {
                lr: 0.6,
                ..optimizer
            },
            max_code_change_fraction: 0.5,
            max_relative_code_change: Some(0.9),
        },
        objective: [5; 32],
        teacher_source: None,
        training: [2.0, 0.5]
            .into_iter()
            .enumerate()
            .map(|(i, weight)| NativeRefinementWindow {
                content: [10 + i as u8; 32],
                weight,
            })
            .collect(),
        validation: vec![NativeRefinementWindow {
            content: [20; 32],
            weight: 1.0,
        }],
        limits: NativeRefinementLimits {
            max_state_bytes: 10_000_000,
            max_workspace_bytes: 10_000_000,
            max_checkpoint_bytes: 10_000_000,
            max_package_bytes: 10_000_000,
        },
    }
}
pub(super) fn tensor(
    law: usize,
    codec: PlaneCodec,
    basis: Basis,
    mode: u8,
    group: u16,
    cols: u64,
) -> AdditiveTensor {
    let admitted = ADMITTED_LAWS[law];
    let layout = AdditiveLayout {
        rows: 3,
        cols,
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
    let tiles = (cols as usize).div_ceil(256);
    let counts: Vec<_> = (0..3 * tiles)
        .map(|i| {
            if mode == 2 || i % 3 == 0 {
                0
            } else if i % 3 == 1 {
                admitted.max_planes
            } else {
                1
            }
        })
        .collect();
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
pub(super) fn active(w: &AdditiveTensor) -> usize {
    if w.plane_count() == 0 {
        0
    } else {
        w.view().plane_trits(0).unwrap().len()
    }
}
pub(super) fn values(n: usize, phase: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 13 + i / 7 + phase) % 19) as f32 / 16.0 - 0.5)
        .collect()
}
pub(super) fn manual(w: &AdditiveTensor, uses: &[Use<'_>]) -> (Vec<f32>, Vec<f32>) {
    let mut row = vec![0.0; w.layout().cols as usize];
    let mut s = vec![0.0; w.scales().len()];
    let mut a = vec![0.0; active(w)];
    for u in uses {
        match u {
            Use::Matmul {
                activations,
                output_gradients,
                batch,
            } => {
                accumulate_ternary_matmul_scale_vjp(
                    activations,
                    &w.view().bind_matmul(),
                    output_gradients,
                    *batch,
                    &mut row,
                    &mut s,
                )
                .unwrap();
                accumulate_ternary_matmul_weight_vjp(
                    activations,
                    &w.view().bind_matmul(),
                    output_gradients,
                    *batch,
                    &mut row,
                    &mut a,
                )
                .unwrap();
            }
            Use::Embed {
                ids,
                output_gradients,
            } => {
                accumulate_embed_scale_vjp(
                    &w.view().bind_gather(),
                    ids,
                    output_gradients,
                    &mut row,
                    &mut s,
                )
                .unwrap();
                accumulate_embed_weight_vjp(
                    &w.view().bind_gather(),
                    ids,
                    output_gradients,
                    &mut row,
                    &mut a,
                )
                .unwrap();
            }
            _ => unreachable!("fixture supplies only known consumers"),
        }
    }
    (
        s.into_iter().map(|v| v as f32).collect(),
        a.into_iter().map(|v| v as f32).collect(),
    )
}
