//! Pure shared-forward capture planning under a resident-accumulator budget.
//!
//! Plan 0054 WS-A2: one forward pass over a calibration batch can feed the
//! S2KF evidence builders of every tensor in one group, so the planner's job
//! is to partition a tensor catalog into shared-input groups whose resident
//! G128 Gram accumulators fit a caller byte budget. Wiring these groups into
//! [`super::Qwen36PtqEvidenceCaptureSession`] is deliberately out of scope
//! here; this module is a pure function over catalog geometry.

use core::fmt;

const GROUP_SIZE: u64 = 128;
// One resident f64 G128 Gram block per column group, doubled because every
// transactional batch append clones the accumulator before mutating it.
const RESIDENT_BYTES_PER_COLUMN_GROUP: u64 = GROUP_SIZE * GROUP_SIZE * 8 * 2;

/// One additive tensor eligible for shared-forward evidence capture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SharedForwardTensor {
    tensor_index: u64,
    input_stream: u64,
    columns: usize,
}

impl SharedForwardTensor {
    /// Describe one catalog tensor by ordinal, shared-input key, and columns.
    ///
    /// `input_stream` is an opaque equality key: two tensors share a forward
    /// activation stream exactly when their keys are equal.
    ///
    /// # Errors
    /// Rejects zero or non-G128-aligned column counts.
    pub fn new(
        tensor_index: u64,
        input_stream: u64,
        columns: usize,
    ) -> Result<Self, SharedForwardPlanError> {
        if columns == 0 || !columns.is_multiple_of(GROUP_SIZE as usize) {
            return Err(SharedForwardPlanError::MalformedColumns { tensor_index });
        }
        Ok(Self {
            tensor_index,
            input_stream,
            columns,
        })
    }

    /// Global additive-catalog tensor ordinal.
    #[must_use]
    pub const fn tensor_index(&self) -> u64 {
        self.tensor_index
    }

    /// Opaque shared-input activation-stream key.
    #[must_use]
    pub const fn input_stream(&self) -> u64 {
        self.input_stream
    }

    /// G128-aligned input column count.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Exact resident accumulator bytes this tensor holds while capturing.
    ///
    /// Alignment was validated at construction, so the division is exact.
    /// Unaddressably wide geometry saturates, which fails closed against any
    /// real budget.
    #[must_use]
    pub fn resident_bytes(&self) -> u64 {
        (self.columns as u64 / GROUP_SIZE).saturating_mul(RESIDENT_BYTES_PER_COLUMN_GROUP)
    }
}

/// One planned capture group: tensors resident through one calibration pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SharedForwardCaptureGroup {
    tensor_indices: Vec<u64>,
    resident_bytes: u64,
}

impl SharedForwardCaptureGroup {
    /// Catalog ordinals captured together, in catalog order.
    #[must_use]
    pub fn tensor_indices(&self) -> &[u64] {
        &self.tensor_indices
    }

    /// Exact resident accumulator bytes while this group captures.
    #[must_use]
    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

/// Failure while validating or partitioning a shared-forward catalog.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SharedForwardPlanError {
    /// A tensor's columns were zero or not G128-aligned.
    MalformedColumns {
        /// Offending catalog ordinal.
        tensor_index: u64,
    },
    /// Two catalog entries repeated one global tensor ordinal.
    DuplicateTensor {
        /// Repeated catalog ordinal.
        tensor_index: u64,
    },
    /// A single tensor's resident accumulators exceed the whole budget.
    TensorExceedsBudget {
        /// Offending catalog ordinal.
        tensor_index: u64,
        /// Exact bytes that tensor holds resident.
        required_bytes: u64,
        /// Caller-authorized resident ceiling.
        max_resident_bytes: u64,
    },
}

impl fmt::Display for SharedForwardPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MalformedColumns { tensor_index } => write!(
                formatter,
                "shared-forward tensor {tensor_index} columns must be a positive multiple of 128"
            ),
            Self::DuplicateTensor { tensor_index } => write!(
                formatter,
                "shared-forward catalog repeats tensor ordinal {tensor_index}"
            ),
            Self::TensorExceedsBudget {
                tensor_index,
                required_bytes,
                max_resident_bytes,
            } => write!(
                formatter,
                "shared-forward tensor {tensor_index} needs {required_bytes} resident bytes, \
                 budget is {max_resident_bytes}"
            ),
        }
    }
}

impl std::error::Error for SharedForwardPlanError {}

/// Partition a catalog into shared-input capture groups under a byte budget.
///
/// Grouping is deterministic: streams are visited in first-appearance catalog
/// order and packed first-fit, keeping every tensor of one input stream in
/// one group whenever the stream fits the budget. A stream larger than the
/// budget is split; correctness is unaffected because evidence identity is
/// keyed on global sample ordinals, so a re-run forward pass feeds the split
/// remainder identically. An empty catalog plans zero groups.
///
/// # Errors
/// Rejects duplicate ordinals or any single tensor larger than the budget
/// (which subsumes a zero budget over a nonempty catalog).
pub fn plan_shared_forward_groups(
    tensors: &[SharedForwardTensor],
    max_resident_bytes: u64,
) -> Result<Vec<SharedForwardCaptureGroup>, SharedForwardPlanError> {
    for (position, tensor) in tensors.iter().enumerate() {
        if tensors[..position]
            .iter()
            .any(|earlier| earlier.tensor_index == tensor.tensor_index)
        {
            return Err(SharedForwardPlanError::DuplicateTensor {
                tensor_index: tensor.tensor_index,
            });
        }
        let required_bytes = tensor.resident_bytes();
        if required_bytes > max_resident_bytes {
            return Err(SharedForwardPlanError::TensorExceedsBudget {
                tensor_index: tensor.tensor_index,
                required_bytes,
                max_resident_bytes,
            });
        }
    }

    // Streams in first-appearance order, members in catalog order.
    let mut streams: Vec<(u64, Vec<&SharedForwardTensor>)> = Vec::new();
    for tensor in tensors {
        match streams
            .iter_mut()
            .find(|(stream, _)| *stream == tensor.input_stream)
        {
            Some((_, members)) => members.push(tensor),
            None => streams.push((tensor.input_stream, vec![tensor])),
        }
    }

    let mut groups = Vec::new();
    let mut current_indices: Vec<u64> = Vec::new();
    let mut current_bytes = 0_u64;
    for (_, members) in streams {
        let stream_bytes = members.iter().fold(0_u64, |total, tensor| {
            total.saturating_add(tensor.resident_bytes())
        });
        if !current_indices.is_empty()
            && current_bytes.saturating_add(stream_bytes) > max_resident_bytes
        {
            groups.push(SharedForwardCaptureGroup {
                tensor_indices: core::mem::take(&mut current_indices),
                resident_bytes: current_bytes,
            });
            current_bytes = 0;
        }
        for tensor in members {
            let required_bytes = tensor.resident_bytes();
            if !current_indices.is_empty()
                && current_bytes.saturating_add(required_bytes) > max_resident_bytes
            {
                groups.push(SharedForwardCaptureGroup {
                    tensor_indices: core::mem::take(&mut current_indices),
                    resident_bytes: current_bytes,
                });
                current_bytes = 0;
            }
            current_indices.push(tensor.tensor_index);
            current_bytes += required_bytes;
        }
    }
    if !current_indices.is_empty() {
        groups.push(SharedForwardCaptureGroup {
            tensor_indices: current_indices,
            resident_bytes: current_bytes,
        });
    }
    Ok(groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(index: u64, stream: u64, columns: usize) -> SharedForwardTensor {
        SharedForwardTensor::new(index, stream, columns).unwrap()
    }

    #[test]
    fn resident_cost_is_exact_and_alignment_is_enforced() {
        // One G128 group costs 128*128*8 bytes, doubled for the
        // transactional clone: 256 KiB.
        assert_eq!(tensor(0, 0, 128).resident_bytes(), 262_144);
        assert_eq!(tensor(0, 0, 512).resident_bytes(), 4 * 262_144);
        assert!(matches!(
            SharedForwardTensor::new(1, 0, 0),
            Err(SharedForwardPlanError::MalformedColumns { tensor_index: 1 })
        ));
        assert!(matches!(
            SharedForwardTensor::new(2, 0, 130),
            Err(SharedForwardPlanError::MalformedColumns { tensor_index: 2 })
        ));
    }

    #[test]
    fn whole_streams_pack_first_fit_in_catalog_order() {
        let catalog = [
            tensor(0, 10, 128),
            tensor(1, 10, 128),
            tensor(2, 20, 128),
            tensor(3, 20, 128),
            tensor(4, 30, 128),
        ];
        // Budget fits exactly one two-tensor stream per group.
        let plan = plan_shared_forward_groups(&catalog, 2 * 262_144).unwrap();
        assert_eq!(plan.len(), 3);
        assert_eq!(plan[0].tensor_indices(), [0, 1]);
        assert_eq!(plan[1].tensor_indices(), [2, 3]);
        assert_eq!(plan[2].tensor_indices(), [4]);
        assert_eq!(plan[0].resident_bytes(), 2 * 262_144);
        assert_eq!(plan[2].resident_bytes(), 262_144);

        // A larger budget keeps stream 20 whole next to stream 30.
        let plan = plan_shared_forward_groups(&catalog, 3 * 262_144).unwrap();
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].tensor_indices(), [0, 1]);
        assert_eq!(plan[1].tensor_indices(), [2, 3, 4]);
    }

    #[test]
    fn oversized_streams_split_and_oversized_tensors_reject() {
        let catalog = [tensor(0, 10, 256), tensor(1, 10, 256), tensor(2, 10, 256)];
        let per_tensor = 2 * 262_144;
        let plan = plan_shared_forward_groups(&catalog, 2 * per_tensor).unwrap();
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].tensor_indices(), [0, 1]);
        assert_eq!(plan[1].tensor_indices(), [2]);

        assert!(matches!(
            plan_shared_forward_groups(&catalog, per_tensor - 1),
            Err(SharedForwardPlanError::TensorExceedsBudget {
                tensor_index: 0,
                ..
            })
        ));
    }

    #[test]
    fn duplicate_ordinals_reject_and_empty_catalog_plans_nothing() {
        assert_eq!(
            plan_shared_forward_groups(&[], 0).unwrap(),
            Vec::<SharedForwardCaptureGroup>::new()
        );
        assert!(matches!(
            plan_shared_forward_groups(&[tensor(7, 0, 128), tensor(7, 1, 128)], u64::MAX),
            Err(SharedForwardPlanError::DuplicateTensor { tensor_index: 7 })
        ));
    }
}
