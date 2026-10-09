//! Bounded, owned observations of authoritative committed Qwen cache state.

use crate::NnError;

/// Maximum FP32 value-payload bytes in one native cache observation request.
pub const QWEN35_REFERENCE_STATE_MAX_BYTES: usize = 256 * 1024 * 1024;

/// Immutable numeric observation, not cache ownership or a qualification receipt.
///
/// Values are flattened in the named ONNX graph output's layout. Runner methods
/// validate provenance before creating these copies; later decode/reset cannot
/// mutate them. Observations do not promote unverified MTP execution.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35ReferenceState {
    name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
}

impl Qwen35ReferenceState {
    /// Canonical graph output name, including its layer index.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Canonical graph output axes.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Finite committed values, owned independently of the runner cache.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

#[derive(Debug)]
pub(super) struct ReferenceStateView<'a> {
    pub name: String,
    pub shape: Vec<usize>,
    pub values: &'a [f32],
}

pub(super) fn snapshot_states(
    views: Vec<ReferenceStateView<'_>>,
    max_state_bytes: usize,
) -> Result<Vec<Qwen35ReferenceState>, NnError> {
    if max_state_bytes == 0 || max_state_bytes > QWEN35_REFERENCE_STATE_MAX_BYTES {
        return Err(NnError::Backend(format!(
            "Qwen cache observation budget must be in 1..={QWEN35_REFERENCE_STATE_MAX_BYTES} bytes"
        )));
    }
    if views.is_empty() {
        return Err(NnError::Backend(
            "Qwen cache observation is empty".to_owned(),
        ));
    }
    // Validate all geometry and the aggregate budget before any value copies.
    let mut bytes = 0usize;
    for view in &views {
        let elements = view.shape.iter().try_fold(1usize, |size, &axis| {
            if axis == 0 {
                None
            } else {
                size.checked_mul(axis)
            }
        });
        if view.name.is_empty() || view.shape.is_empty() || elements != Some(view.values.len()) {
            return Err(NnError::Backend(format!(
                "Qwen cache observation {} has invalid geometry",
                view.name
            )));
        }
        bytes = view
            .values
            .len()
            .checked_mul(size_of::<f32>())
            .and_then(|size| bytes.checked_add(size))
            .ok_or_else(|| NnError::Backend("Qwen cache observation size overflow".to_owned()))?;
    }
    if bytes > max_state_bytes {
        return Err(NnError::Backend(format!(
            "Qwen cache observation requires {bytes} value bytes, budget is {max_state_bytes}"
        )));
    }
    if views
        .iter()
        .any(|view| view.values.iter().any(|value| !value.is_finite()))
    {
        return Err(NnError::Backend(
            "Qwen cache observation contains non-finite values".to_owned(),
        ));
    }
    let mut snapshots = Vec::new();
    snapshots.try_reserve_exact(views.len()).map_err(|error| {
        NnError::Backend(format!("allocate Qwen cache observation table: {error}"))
    })?;
    for view in views {
        let mut values = Vec::new();
        values
            .try_reserve_exact(view.values.len())
            .map_err(|error| {
                NnError::Backend(format!(
                    "allocate Qwen cache observation {}: {error}",
                    view.name
                ))
            })?;
        values.extend_from_slice(view.values);
        snapshots.push(Qwen35ReferenceState {
            name: view.name,
            shape: view.shape,
            values,
        });
    }
    Ok(snapshots)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observations_reject_nonfinite_empty_overflow_and_mismatched_geometry() {
        for (shape, values) in [
            (vec![], vec![1.0]),
            (vec![0], vec![]),
            (vec![2], vec![1.0]),
            (vec![usize::MAX, 2], vec![1.0]),
            (vec![1], vec![f32::NAN]),
            (vec![1], vec![f32::INFINITY]),
        ] {
            assert!(
                snapshot_states(
                    vec![ReferenceStateView {
                        name: "state".to_owned(),
                        shape,
                        values: &values,
                    }],
                    32
                )
                .is_err()
            );
        }
        assert!(snapshot_states(vec![], 32).is_err());
    }

    #[test]
    fn budget_covers_all_states_not_each_one_individually() {
        let views = || {
            (0..2)
                .map(|index| ReferenceStateView {
                    name: format!("state.{index}"),
                    shape: vec![2],
                    values: &[1.0, 2.0],
                })
                .collect()
        };
        assert!(snapshot_states(views(), 15).is_err());
        assert_eq!(snapshot_states(views(), 16).unwrap().len(), 2);
    }
}
