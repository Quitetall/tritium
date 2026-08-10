use half::f16;

use crate::optim::{AdamState, Optimizer};

use super::{PvTernaryWeight, PvTuningConfig, PvTuningError, PvTuningSession};

impl PvTuningSession {
    pub(super) fn p_step(
        &mut self,
        gradient: &[f32],
        optimizer_step: u64,
    ) -> Result<usize, PvTuningError> {
        let groups_per_row = self.weight.groups_per_row();
        let scales_per_plane = self.weight.scale_count_per_plane();
        let mut scale_gradient = Vec::with_capacity(self.weight.total_scale_count());
        for plane in &self.weight.planes {
            for scale_index in 0..scales_per_plane {
                let row = scale_index / groups_per_row;
                let group = scale_index % groups_per_row;
                let start = row * self.weight.cols + group * self.weight.group_size;
                let end = (start + self.weight.group_size).min((row + 1) * self.weight.cols);
                let mut derivative = 0.0f64;
                for (&element_gradient, &trit) in
                    gradient[start..end].iter().zip(&plane.trits[start..end])
                {
                    derivative += f64::from(element_gradient) * f64::from(trit);
                }
                let derivative = derivative as f32;
                if !derivative.is_finite() {
                    return Err(PvTuningError::step("scale gradient overflowed f32"));
                }
                scale_gradient.push(f64::from(derivative));
            }
        }
        apply_accumulated_scale_gradient(
            &mut self.weight,
            &mut self.scale_state,
            self.config,
            &scale_gradient,
            optimizer_step,
        )
    }
}

pub(super) fn apply_accumulated_scale_gradient(
    weight: &mut PvTernaryWeight,
    scale_state: &mut AdamState,
    config: PvTuningConfig,
    accumulated: &[f64],
    optimizer_step: u64,
) -> Result<usize, PvTuningError> {
    if accumulated.len() != weight.total_scale_count() {
        return Err(PvTuningError::step("scale gradient length mismatch"));
    }
    let mut scale_gradient = Vec::with_capacity(accumulated.len());
    for &derivative in accumulated {
        let derivative = derivative as f32;
        if !derivative.is_finite() {
            return Err(PvTuningError::step("scale gradient overflowed f32"));
        }
        scale_gradient.push(derivative);
    }
    let mut values = weight
        .planes
        .iter()
        .flat_map(|plane| plane.scales.iter().map(|scale| f32::from(*scale)))
        .collect::<Vec<_>>();
    let mut active = Vec::with_capacity(weight.total_scale_count());
    let groups_per_row = weight.groups_per_row();
    for plane in &weight.planes {
        for scale_index in 0..weight.scale_count_per_plane() {
            let row = scale_index / groups_per_row;
            let group = scale_index % groups_per_row;
            let start = row * weight.cols + group * weight.group_size;
            let end = (start + weight.group_size).min((row + 1) * weight.cols);
            active.push(plane.trits[start..end].iter().any(|&trit| trit != 0));
        }
    }
    let old_bits: Vec<u16> = weight
        .planes
        .iter()
        .flat_map(|plane| plane.scales.iter().map(|scale| scale.to_bits()))
        .collect();
    config
        .continuous_optimizer
        .step(optimizer_step, &mut values, &scale_gradient, scale_state);
    let mut flat_index = 0;
    for plane in &mut weight.planes {
        for scale in &mut plane.scales {
            let mut value = values[flat_index];
            if !value.is_finite() {
                return Err(PvTuningError::step(
                    "continuous optimizer produced a non-finite scale",
                ));
            }
            value = value.max(if active[flat_index] {
                f32::from(f16::from_bits(1))
            } else {
                0.0
            });
            let narrowed = f16::from_f32(value);
            if !f32::from(narrowed).is_finite() {
                return Err(PvTuningError::step("scale update overflowed f16"));
            }
            *scale = narrowed;
            flat_index += 1;
        }
    }
    Ok(old_bits
        .iter()
        .zip(weight.planes.iter().flat_map(|plane| &plane.scales))
        .filter(|(old, new)| **old != new.to_bits())
        .count())
}
