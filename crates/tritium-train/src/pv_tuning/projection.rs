use super::representation::{unit_start, unit_width};
use super::selection::squared_error;
use super::{PvTernaryStructure, PvTernaryWeight};

#[derive(Clone, Copy, Debug)]
pub(super) struct ProjectionStats {
    pub(super) changed_units: usize,
    pub(super) trust_limited_units: usize,
    pub(super) relative_change: f64,
}

pub(super) fn project_units<F>(
    weight: &mut PvTernaryWeight,
    max_relative_code_change: Option<f32>,
    units: &[usize],
    mut proposal_at: F,
) -> (ProjectionStats, f64, f64)
where
    F: FnMut(usize, f32) -> f32,
{
    let representation_norm_squared = (0..weight.len())
        .map(|index| f64::from(weight.decode_element(index)).powi(2))
        .sum::<f64>()
        .max(f64::from(f32::EPSILON));
    let trust_limit = max_relative_code_change
        .map(|ratio| f64::from(ratio).powi(2) * representation_norm_squared);
    let mut accepted_delta = 0.0f64;
    let mut changed_units = 0;
    let mut trust_limited_units = 0;
    let mut surrogate_before = 0.0f64;
    let mut surrogate_after = 0.0f64;
    for &unit in units {
        let start = unit_start(weight, unit);
        let width = unit_width(weight.structure);
        let original = (start..start + width)
            .map(|index| weight.decode_element(index))
            .collect::<Vec<_>>();
        let target = original
            .iter()
            .enumerate()
            .map(|(offset, &decoded)| proposal_at(start + offset, decoded))
            .collect::<Vec<_>>();
        let before = original
            .iter()
            .zip(&target)
            .map(|(&value, &target)| squared_error(value, target))
            .sum::<f64>();
        surrogate_before += before;
        let candidate = best_candidate(weight, unit, &target);
        let Some(candidate) = candidate else {
            surrogate_after += before;
            continue;
        };
        let delta = candidate
            .decoded
            .iter()
            .zip(&original)
            .map(|(&candidate, &original)| (f64::from(candidate) - f64::from(original)).powi(2))
            .sum::<f64>();
        if trust_limit.is_some_and(|limit| accepted_delta + delta > limit) {
            trust_limited_units += 1;
            surrogate_after += before;
            continue;
        }
        let after = candidate
            .decoded
            .iter()
            .zip(&target)
            .map(|(&value, &target)| squared_error(value, target))
            .sum::<f64>();
        if candidate.apply(weight) {
            accepted_delta += delta;
            changed_units += 1;
            surrogate_after += after;
        } else {
            surrogate_after += before;
        }
    }
    (
        ProjectionStats {
            changed_units,
            trust_limited_units,
            relative_change: (accepted_delta / representation_norm_squared).sqrt(),
        },
        surrogate_before,
        surrogate_after,
    )
}

// Both native layouts and legacy f16 PV use the same discrete search. The
// legacy adapter widens one unit only, preserving its original f32 arithmetic
// and outer selection/trust accounting. No dense owner conversion is involved.
fn best_candidate(weight: &PvTernaryWeight, unit: usize, target: &[f32]) -> Option<CodeCandidate> {
    use tritium_core::{AdditiveView, Trit, project_additive_code_unit};
    use tritium_schema::{
        ADMITTED_LAWS, AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, Transport,
    };
    let width = unit_width(weight.structure);
    let start = unit_start(weight, unit);
    let planes = weight.planes.len();
    let mut input = [Trit::ZERO; 12];
    let mut scales = [0.0f32; 3];
    let scale_index =
        (start / weight.cols) * weight.groups_per_row() + (start % weight.cols) / weight.group_size;
    for (p, plane) in weight.planes.iter().enumerate() {
        scales[p] = f32::from(plane.scales[scale_index]);
        for offset in 0..width {
            input[p * width + offset] = Trit::from_i8(plane.trits[start + offset])
                .expect("validated legacy ternary coefficient");
        }
    }
    let layout = AdditiveLayout {
        rows: 1,
        cols: width as u64,
        tile: 256,
        group: 32,
        max_planes: 3,
        allocation: PlaneAllocation::Uniform,
        codec: if weight.structure == PvTernaryStructure::S34 {
            PlaneCodec::S34
        } else {
            PlaneCodec::D2
        },
        law: ADMITTED_LAWS[1].law,
        basis: Basis::Identity,
        transport: Transport::Raw,
    };
    let length = planes * width;
    let view = AdditiveView::new(layout, planes as u8, &input[..length], &scales[..planes])
        .expect("lossless one-unit legacy projection adapter");
    let mut output = input;
    let stats = project_additive_code_unit(view, 0, target, None, &mut output[..length])
        .expect("finite validated legacy unit and proposal");
    if !stats.changed {
        return None;
    }
    let mut decoded = vec![0.0; width];
    let codes_by_plane = (0..planes)
        .map(|p| {
            (0..width)
                .map(|offset| {
                    let code = output[p * width + offset];
                    decoded[offset] += code.to_f32() * scales[p];
                    code.get()
                })
                .collect()
        })
        .collect();
    Some(CodeCandidate {
        start,
        codes_by_plane,
        decoded,
    })
}

#[derive(Debug)]
struct CodeCandidate {
    start: usize,
    codes_by_plane: Vec<Vec<i8>>,
    decoded: Vec<f32>,
}

impl CodeCandidate {
    fn apply(self, weight: &mut PvTernaryWeight) -> bool {
        let mut changed = false;
        for (plane, codes) in weight.planes.iter_mut().zip(self.codes_by_plane) {
            for (offset, code) in codes.into_iter().enumerate() {
                changed |= plane.trits[self.start + offset] != code;
                plane.trits[self.start + offset] = code;
            }
        }
        changed
    }
}
