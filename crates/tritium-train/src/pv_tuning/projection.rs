use super::selection::squared_error;
use super::{PvTernaryStructure, PvTernaryWeight, PvTuningSession};

#[derive(Clone, Copy, Debug)]
pub(super) struct ProjectionStats {
    pub(super) changed_units: usize,
    pub(super) trust_limited_units: usize,
    pub(super) relative_change: f64,
}

impl PvTuningSession {
    pub(super) fn v_step(
        &mut self,
        units: &[usize],
        decoded: &[f32],
        proposal: &[f32],
    ) -> ProjectionStats {
        let representation_norm_squared = decoded
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .max(f64::from(f32::EPSILON));
        let trust_limit = self
            .config
            .max_relative_code_change
            .map(|ratio| f64::from(ratio).powi(2) * representation_norm_squared);
        let mut accepted_delta = 0.0f64;
        let mut changed_units = 0;
        let mut trust_limited_units = 0;
        for &unit in units {
            let candidate = match self.weight.structure {
                PvTernaryStructure::Dense => self.best_dense_candidate(unit, proposal[unit]),
                PvTernaryStructure::S34 => self.best_s34_candidate(unit, proposal),
            };
            let Some(candidate) = candidate else {
                continue;
            };
            let delta = candidate
                .decoded
                .iter()
                .enumerate()
                .map(|(offset, value)| {
                    let index = candidate.start + offset;
                    (f64::from(*value) - f64::from(decoded[index])).powi(2)
                })
                .sum::<f64>();
            if trust_limit.is_some_and(|limit| accepted_delta + delta > limit) {
                trust_limited_units += 1;
                continue;
            }
            if candidate.apply(&mut self.weight) {
                accepted_delta += delta;
                changed_units += 1;
            }
        }
        ProjectionStats {
            changed_units,
            trust_limited_units,
            relative_change: (accepted_delta / representation_norm_squared).sqrt(),
        }
    }

    fn best_dense_candidate(&self, index: usize, target: f32) -> Option<CodeCandidate> {
        let current: Vec<i8> = self
            .weight
            .planes
            .iter()
            .map(|plane| plane.trits[index])
            .collect();
        let scales: Vec<f32> = self
            .weight
            .planes
            .iter()
            .map(|plane| {
                let row = index / self.weight.cols;
                let col = index % self.weight.cols;
                let scale_index = row * self.weight.groups_per_row() + col / self.weight.group_size;
                f32::from(plane.scales[scale_index])
            })
            .collect();
        let current_decoded = decode_tuple(&current, &scales);
        let mut best_codes = current.clone();
        let mut best_decoded = current_decoded;
        let mut best_loss = squared_error(current_decoded, target);
        let combinations = 3usize.pow(self.weight.planes.len() as u32);
        for encoded in 0..combinations {
            let mut cursor = encoded;
            let mut codes = Vec::with_capacity(self.weight.planes.len());
            for &scale in &scales {
                let code = match cursor % 3 {
                    0 => -1,
                    1 => 0,
                    _ => 1,
                };
                cursor /= 3;
                codes.push(if scale == 0.0 { 0 } else { code });
            }
            let value = decode_tuple(&codes, &scales);
            let loss = squared_error(value, target);
            if loss < best_loss {
                best_loss = loss;
                best_decoded = value;
                best_codes = codes;
            }
        }
        (best_codes != current).then_some(CodeCandidate {
            start: index,
            codes_by_plane: best_codes.into_iter().map(|code| vec![code]).collect(),
            decoded: vec![best_decoded],
        })
    }

    fn best_s34_candidate(&self, unit: usize, proposal: &[f32]) -> Option<CodeCandidate> {
        let blocks_per_row = self.weight.cols / 4;
        let start = (unit / blocks_per_row) * self.weight.cols + (unit % blocks_per_row) * 4;
        let target = &proposal[start..start + 4];
        let original: Vec<Vec<i8>> = self
            .weight
            .planes
            .iter()
            .map(|plane| plane.trits[start..start + 4].to_vec())
            .collect();
        let mut codes = original.clone();
        let scale_index = (start / self.weight.cols) * self.weight.groups_per_row()
            + (start % self.weight.cols) / self.weight.group_size;
        let scales: Vec<f32> = self
            .weight
            .planes
            .iter()
            .map(|plane| f32::from(plane.scales[scale_index]))
            .collect();
        for plane_index in 0..self.weight.planes.len() {
            let mut best = codes[plane_index].clone();
            let mut best_loss = block_loss(&codes, &scales, target);
            for zero in 0..4 {
                for signs in 0..8usize {
                    let candidate = s34_pattern(zero, signs);
                    codes[plane_index].copy_from_slice(&candidate);
                    let loss = block_loss(&codes, &scales, target);
                    if loss < best_loss {
                        best_loss = loss;
                        best.copy_from_slice(&candidate);
                    }
                }
            }
            codes[plane_index] = best;
        }
        if codes == original {
            return None;
        }
        let mut decoded = vec![0.0; 4];
        for (plane_codes, scale) in codes.iter().zip(scales) {
            for offset in 0..4 {
                decoded[offset] += scale * f32::from(plane_codes[offset]);
            }
        }
        Some(CodeCandidate {
            start,
            codes_by_plane: codes,
            decoded,
        })
    }
}

fn s34_pattern(zero: usize, signs: usize) -> [i8; 4] {
    let mut candidate = [0; 4];
    let mut sign_index = 0;
    for (offset, code) in candidate.iter_mut().enumerate() {
        if offset != zero {
            *code = if (signs >> sign_index) & 1 == 0 {
                -1
            } else {
                1
            };
            sign_index += 1;
        }
    }
    candidate
}

fn decode_tuple(codes: &[i8], scales: &[f32]) -> f32 {
    codes
        .iter()
        .zip(scales)
        .map(|(&code, &scale)| f32::from(code) * scale)
        .sum()
}

fn block_loss(codes: &[Vec<i8>], scales: &[f32], target: &[f32]) -> f64 {
    (0..4)
        .map(|offset| {
            let value = codes
                .iter()
                .zip(scales)
                .map(|(plane, &scale)| f32::from(plane[offset]) * scale)
                .sum();
            squared_error(value, target[offset])
        })
        .sum()
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
