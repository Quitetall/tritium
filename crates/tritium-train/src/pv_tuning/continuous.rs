use half::f16;

use crate::optim::Optimizer;

use super::{PvTuningError, PvTuningSession};

impl PvTuningSession {
    pub(super) fn p_step(
        &mut self,
        gradient: &[f32],
        optimizer_step: u64,
    ) -> Result<usize, PvTuningError> {
        let groups_per_row = self.weight.groups_per_row();
        let scales_per_plane = self.weight.scale_count_per_plane();
        let mut values = Vec::with_capacity(self.weight.total_scale_count());
        let mut scale_gradient = Vec::with_capacity(self.weight.total_scale_count());
        let mut active = Vec::with_capacity(self.weight.total_scale_count());
        for plane in &self.weight.planes {
            for scale_index in 0..scales_per_plane {
                values.push(f32::from(plane.scales[scale_index]));
                let row = scale_index / groups_per_row;
                let group = scale_index % groups_per_row;
                let start = row * self.weight.cols + group * self.weight.group_size;
                let end = (start + self.weight.group_size).min((row + 1) * self.weight.cols);
                let mut derivative = 0.0f64;
                let mut has_nonzero_code = false;
                for (&element_gradient, &trit) in
                    gradient[start..end].iter().zip(&plane.trits[start..end])
                {
                    has_nonzero_code |= trit != 0;
                    derivative += f64::from(element_gradient) * f64::from(trit);
                }
                let derivative = derivative as f32;
                if !derivative.is_finite() {
                    return Err(PvTuningError::step("scale gradient overflowed f32"));
                }
                scale_gradient.push(derivative);
                active.push(has_nonzero_code);
            }
        }
        let old_bits: Vec<u16> = self
            .weight
            .planes
            .iter()
            .flat_map(|plane| plane.scales.iter().map(|scale| scale.to_bits()))
            .collect();
        self.config.continuous_optimizer.step(
            optimizer_step,
            &mut values,
            &scale_gradient,
            &mut self.scale_state,
        );
        let mut flat_index = 0;
        for plane in &mut self.weight.planes {
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
            .zip(self.weight.planes.iter().flat_map(|plane| &plane.scales))
            .filter(|(old, new)| **old != new.to_bits())
            .count())
    }
}
