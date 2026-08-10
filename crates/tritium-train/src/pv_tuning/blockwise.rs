use core::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use super::continuous::apply_accumulated_scale_gradient;
use super::projection::project_units;
use super::representation::unit_width;
use super::{PvBlockwiseState, PvStepReceipt, PvTuningError, PvTuningSession};

/// Durable position inside one strict, contiguous blockwise P/V gradient stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PvBlockwiseCursor {
    optimizer_step: u64,
    next_offset: usize,
    total_elements: usize,
    max_block_elements: usize,
}

impl PvBlockwiseCursor {
    #[must_use]
    pub const fn optimizer_step(self) -> u64 {
        self.optimizer_step
    }

    #[must_use]
    pub const fn next_offset(self) -> usize {
        self.next_offset
    }

    #[must_use]
    pub const fn total_elements(self) -> usize {
        self.total_elements
    }

    #[must_use]
    pub const fn max_block_elements(self) -> usize {
        self.max_block_elements
    }
}

impl PvTuningSession {
    /// Begin one exact next P/V step whose gradient arrives in bounded contiguous blocks.
    pub fn begin_blockwise_step(
        &mut self,
        optimizer_step: u64,
        max_block_elements: usize,
    ) -> Result<(), PvTuningError> {
        if self.blockwise.is_some() {
            return Err(PvTuningError::step("a blockwise step is already active"));
        }
        let expected = self
            .completed_step
            .checked_add(1)
            .ok_or_else(|| PvTuningError::step("step counter overflow"))?;
        if optimizer_step != expected {
            return Err(PvTuningError::step(format!(
                "expected optimizer step {expected}, got {optimizer_step}"
            )));
        }
        if max_block_elements == 0 {
            return Err(PvTuningError::step("max_block_elements must be non-zero"));
        }
        let scale_count = self.weight.total_scale_count();
        let mut scale_gradient = Vec::new();
        scale_gradient
            .try_reserve_exact(scale_count)
            .map_err(|_| PvTuningError::step("scale-gradient allocation failed"))?;
        scale_gradient.resize(scale_count, 0.0);
        self.blockwise = Some(PvBlockwiseState {
            optimizer_step,
            max_block_elements,
            next_offset: 0,
            scale_gradient,
        });
        Ok(())
    }

    /// Current in-flight blockwise position, or `None` between steps.
    #[must_use]
    pub fn blockwise_cursor(&self) -> Option<PvBlockwiseCursor> {
        self.blockwise.as_ref().map(|state| PvBlockwiseCursor {
            optimizer_step: state.optimizer_step,
            next_offset: state.next_offset,
            total_elements: self.weight.len(),
            max_block_elements: state.max_block_elements,
        })
    }

    /// Consume one finite gradient block at the exact next flattened offset.
    pub fn apply_gradient_block(
        &mut self,
        offset: usize,
        gradient: &[f32],
    ) -> Result<(), PvTuningError> {
        let state = self
            .blockwise
            .as_ref()
            .ok_or_else(|| PvTuningError::step("no blockwise step is active"))?;
        if offset != state.next_offset {
            return Err(PvTuningError::step(format!(
                "expected gradient block offset {}, got {offset}",
                state.next_offset
            )));
        }
        if gradient.is_empty() || gradient.len() > state.max_block_elements {
            return Err(PvTuningError::step(
                "gradient block must be non-empty and within max_block_elements",
            ));
        }
        let end = offset
            .checked_add(gradient.len())
            .ok_or_else(|| PvTuningError::step("gradient block range overflow"))?;
        if end > self.weight.len() {
            return Err(PvTuningError::step("gradient block exceeds weight length"));
        }
        if gradient.iter().any(|value| !value.is_finite()) {
            return Err(PvTuningError::step(
                "gradient block contains a non-finite value",
            ));
        }

        let optimizer = self.config.code_optimizer;
        let mut next_m = Vec::with_capacity(gradient.len());
        let mut next_v = Vec::with_capacity(gradient.len());
        for (local, &value) in gradient.iter().enumerate() {
            let index = offset + local;
            let first =
                optimizer.beta1 * self.code_state.m[index] + (1.0 - optimizer.beta1) * value;
            let second = optimizer.beta2 * self.code_state.v[index]
                + (1.0 - optimizer.beta2) * value * value;
            if !first.is_finite() || !second.is_finite() {
                return Err(PvTuningError::step(
                    "code optimizer moment update became non-finite",
                ));
            }
            next_m.push(first);
            next_v.push(second);
        }

        let scale_count = self.weight.scale_count_per_plane();
        let groups_per_row = self.weight.groups_per_row();
        let mut scale_updates = Vec::new();
        for (plane_index, plane) in self.weight.planes.iter().enumerate() {
            let plane_base = plane_index * scale_count;
            let mut current_scale = None;
            let mut accumulated = 0.0f64;
            for (local, &value) in gradient.iter().enumerate() {
                let index = offset + local;
                let row = index / self.weight.cols;
                let col = index % self.weight.cols;
                let scale_index = plane_base + row * groups_per_row + col / self.weight.group_size;
                if current_scale != Some(scale_index) {
                    if let Some(previous) = current_scale {
                        scale_updates.push((previous, accumulated));
                    }
                    current_scale = Some(scale_index);
                    accumulated = state.scale_gradient[scale_index];
                }
                accumulated += f64::from(value) * f64::from(plane.trits[index]);
            }
            if let Some(scale_index) = current_scale {
                scale_updates.push((scale_index, accumulated));
            }
        }
        if scale_updates.iter().any(|(_, value)| !value.is_finite()) {
            return Err(PvTuningError::step(
                "scale gradient accumulation became non-finite",
            ));
        }

        self.code_state.m[offset..end].copy_from_slice(&next_m);
        self.code_state.v[offset..end].copy_from_slice(&next_v);
        let state = self
            .blockwise
            .as_mut()
            .expect("blockwise state was validated above");
        for (scale_index, value) in scale_updates {
            state.scale_gradient[scale_index] = value;
        }
        state.next_offset = end;
        Ok(())
    }

    /// Commit a fully consumed blockwise gradient as one alternating P/V step.
    pub fn finish_blockwise_step(&mut self) -> Result<PvStepReceipt, PvTuningError> {
        let state = self
            .blockwise
            .as_ref()
            .ok_or_else(|| PvTuningError::step("no blockwise step is active"))?;
        if state.next_offset != self.weight.len() {
            return Err(PvTuningError::step(format!(
                "blockwise gradient is incomplete: consumed {}, expected {}",
                state.next_offset,
                self.weight.len()
            )));
        }
        let optimizer_step = state.optimizer_step;
        let mut next_weight = self.weight.clone();
        let mut next_scale_state = self.scale_state.clone();
        let changed_scales = apply_accumulated_scale_gradient(
            &mut next_weight,
            &mut next_scale_state,
            self.config,
            &state.scale_gradient,
            optimizer_step,
        )?;
        let units = select_units_from_updated_moments(
            &next_weight,
            self.config.code_optimizer,
            &self.code_state,
            optimizer_step,
            self.config.max_code_change_fraction,
        );
        let optimizer = self.config.code_optimizer;
        let code_state = &self.code_state;
        let (projection, v_surrogate_before, v_surrogate_after) = project_units(
            &mut next_weight,
            self.config.max_relative_code_change,
            &units,
            |index, decoded| {
                proposal_from_updated_moments(
                    optimizer,
                    optimizer_step,
                    decoded,
                    code_state.m[index],
                    code_state.v[index],
                )
            },
        );
        next_weight.validate()?;
        if v_surrogate_after > v_surrogate_before {
            return Err(PvTuningError::step(
                "V projection increased its discrete surrogate",
            ));
        }
        let representation_digest = next_weight.digest();
        self.weight = next_weight;
        self.scale_state = next_scale_state;
        self.completed_step = optimizer_step;
        self.blockwise = None;
        Ok(PvStepReceipt {
            optimizer_step,
            selected_code_units: units.len(),
            changed_code_units: projection.changed_units,
            trust_limited_code_units: projection.trust_limited_units,
            changed_scales,
            v_surrogate_before,
            v_surrogate_after,
            relative_code_change: projection.relative_change,
            representation_digest,
        })
    }
}

fn proposal_from_updated_moments(
    optimizer: crate::optim::AdamW,
    step: u64,
    parameter: f32,
    first_moment: f32,
    second_moment: f32,
) -> f32 {
    let exponent = i32::try_from(step).unwrap_or(i32::MAX);
    let first_correction = 1.0 - optimizer.beta1.powi(exponent);
    let second_correction = 1.0 - optimizer.beta2.powi(exponent);
    let shrink = 1.0 - optimizer.lr * optimizer.weight_decay;
    parameter * shrink
        - optimizer.lr
            * (first_moment
                / first_correction
                / ((second_moment / second_correction).sqrt() + optimizer.eps))
}

fn select_units_from_updated_moments(
    weight: &super::PvTernaryWeight,
    optimizer: crate::optim::AdamW,
    code_state: &crate::optim::AdamState,
    step: u64,
    fraction: f32,
) -> Vec<usize> {
    let width = unit_width(weight.structure);
    let unit_count = weight.len() / width;
    let keep = ((unit_count as f64) * f64::from(fraction)).ceil() as usize;
    let keep = keep.max(1).min(unit_count);
    let mut selected = BinaryHeap::with_capacity(keep);
    for unit in 0..unit_count {
        let start = unit * width;
        let mut magnitude = 0.0f64;
        for index in start..start + width {
            let decoded = weight.decode_element(index);
            let proposal = proposal_from_updated_moments(
                optimizer,
                step,
                decoded,
                code_state.m[index],
                code_state.v[index],
            );
            magnitude += (f64::from(proposal) - f64::from(decoded)).powi(2);
        }
        let candidate = RankedUnit { unit, magnitude };
        if selected.len() < keep {
            selected.push(Reverse(candidate));
        } else if candidate > selected.peek().expect("heap is non-empty").0 {
            selected.pop();
            selected.push(Reverse(candidate));
        }
    }
    let mut ranked = selected
        .into_iter()
        .map(|Reverse(candidate)| candidate)
        .collect::<Vec<_>>();
    ranked.sort_unstable_by(|left, right| right.cmp(left));
    ranked.into_iter().map(|candidate| candidate.unit).collect()
}

#[derive(Clone, Copy, Debug)]
struct RankedUnit {
    unit: usize,
    magnitude: f64,
}

impl PartialEq for RankedUnit {
    fn eq(&self, other: &Self) -> bool {
        self.unit == other.unit && self.magnitude.to_bits() == other.magnitude.to_bits()
    }
}

impl Eq for RankedUnit {}

impl PartialOrd for RankedUnit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RankedUnit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.magnitude
            .total_cmp(&other.magnitude)
            .then_with(|| other.unit.cmp(&self.unit))
    }
}
