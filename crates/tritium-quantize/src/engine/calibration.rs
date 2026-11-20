//! One candidate collector over the existing source-bound numerical reducers.
//! The caller must verify/finalize source/cache identities before accumulation.
//! This is not a cache verifier, registered evidence event or full RSS admission.
use crate::{
    CurvatureError, CurvatureSourceId, IndexedOutputFisherAccumulator, InputGram,
    InputGramAccumulator, KfacMetric, OutputFisher, OutputFisherAccumulator, build_kfac_metric,
};
use tritium_schema::{
    CalibrationConsumer, CalibrationFactor, CalibrationFactors, CalibrationOutput,
    CalibrationPartition, CalibrationPlan, CalibrationPlanError, FitInputDigest,
};

const MAX_SEGMENTS: usize = crate::salt_v2_evidence::MAX_KRONECKER_REDUCTION_SEGMENTS;
// Old state + transactional state + inner reducer clone, with spare matrices
// for merging/finalization/canonicalization and one row's K-FAC/PSD workspace.
const NUMERIC_SLOTS: u64 = 3 * MAX_SEGMENTS as u64 + 16;

/// Failure at the common calibration seam; no estimator fallback is performed.
#[derive(Debug)]
#[non_exhaustive]
pub enum CalibrationError {
    /// Invalid candidate request.
    Plan(CalibrationPlanError),
    /// Address/dimension or allocation arithmetic overflow.
    Allocation,
    /// Numerical workspace exceeds its explicit budget.
    Budget {
        /// Conservative required numerical bytes.
        required: u64,
        /// Caller-declared limit.
        limit: u64,
    },
    /// Supplied batch encoding/shape or consumer does not match the request.
    Batch,
    /// Shard contract differs, including source/cache/token binding.
    Shard,
    /// Retained reduction state exceeds the declared finite segment bound.
    Segments,
    /// Input/output source or selection no longer agrees.
    Selection,
    /// Requested K-FAC without an actual output factor.
    MissingFisher,
    /// Invalid group or diagonal destination length.
    Shape,
    /// The existing reducer/metric rejects numerical content.
    Curvature(CurvatureError),
}
impl From<CurvatureError> for CalibrationError {
    fn from(error: CurvatureError) -> Self {
        Self::Curvature(error)
    }
}
impl core::fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "calibration rejected: {self:?}")
    }
}
impl std::error::Error for CalibrationError {}

/// Actual original-source values for one observed batch.
#[derive(Clone, Copy, Debug)]
pub enum CalibrationInput<'a> {
    /// Row-major `[samples, original columns]` linear inputs.
    Activations(&'a [f32]),
    /// Explicit gather identity approximation; no invented dense activations.
    GatherIdentity,
}
/// Actual output encoding, which must exactly match the plan.
#[derive(Clone, Copy, Debug)]
pub enum CalibrationOutputBatch<'a> {
    /// No output factor and no Fisher claim.
    None,
    /// Row-major `[samples, original rows]` output gradients.
    Dense(&'a [f32]),
    /// One named factor row per sample; all other row factors are zero.
    Indexed {
        /// Original output-row indices.
        indices: &'a [usize],
        /// Actual factors, not inferred squared activations.
        factors: &'a [f32],
    },
}
/// A paired input/output batch sharing exact sample ordinals, weights and masks.
#[derive(Clone, Copy, Debug)]
pub struct CalibrationBatch<'a> {
    /// Number of observed samples, including masked samples.
    pub samples: usize,
    /// Actual original-source inputs or declared gather approximation.
    pub input: CalibrationInput<'a>,
    /// Actual output factors, if requested.
    pub output: CalibrationOutputBatch<'a>,
    /// Optional nonnegative weights, one per observed sample.
    pub weights: Option<&'a [f64]>,
    /// Optional selection mask, one per observed sample.
    pub mask: Option<&'a [bool]>,
}

#[derive(Debug)]
enum Fisher {
    Dense(OutputFisherAccumulator),
    Indexed(IndexedOutputFisherAccumulator),
}
impl Fisher {
    fn try_clone(&self) -> Result<Self, CurvatureError> {
        match self {
            Self::Dense(f) => f.try_clone_transactional().map(Self::Dense),
            Self::Indexed(f) => f.try_clone_transactional().map(Self::Indexed),
        }
    }
    fn segments(&self) -> usize {
        match self {
            Self::Dense(f) => f.retained_reduction_segments(),
            Self::Indexed(f) => f.retained_reduction_segments(),
        }
    }
    fn finish(&self) -> Result<OutputFisher, CurvatureError> {
        match self {
            Self::Dense(f) => f.finish(),
            Self::Indexed(f) => f.finish(),
        }
    }
}

/// Source-bound full/grouped Gram and dense/indexed Fisher orchestration.
/// All numerical reductions and their existing identities remain shared.
#[derive(Debug)]
pub struct CalibrationCollector {
    plan: CalibrationPlan,
    columns: usize,
    width: usize,
    base_bytes: u64,
    inputs: Vec<InputGramAccumulator>,
    output: Option<Fisher>,
}
impl CalibrationCollector {
    /// Conservative numeric-array/encoded-factor bound for this plan.
    /// Excludes metadata/allocator overhead, borrowed caller buffers and RSS.
    pub fn required_numeric_bytes(plan: &CalibrationPlan) -> Result<u64, CalibrationError> {
        plan.validate().map_err(CalibrationError::Plan)?;
        let columns = plan.tensor.shape[1];
        let width = match plan.partition {
            CalibrationPartition::Full => columns,
            CalibrationPartition::Groups { columns: width } => u64::from(width).min(columns),
            _ => return Err(CalibrationError::Plan(CalibrationPlanError::Partition)),
        };
        let full = columns / width;
        let tail = columns % width;
        let entries = width
            .checked_mul(width)
            .and_then(|n| n.checked_mul(full))
            .and_then(|n| tail.checked_mul(tail).and_then(|t| n.checked_add(t)))
            .ok_or(CalibrationError::Allocation)?;
        let row_bytes = match plan.output {
            CalibrationOutput::None => 0,
            CalibrationOutput::DenseFisher | CalibrationOutput::IndexedFisher => plan.tensor.shape
                [0]
            .checked_mul(16)
            .ok_or(CalibrationError::Allocation)?,
            _ => return Err(CalibrationError::Batch),
        };
        entries
            .checked_mul(8)
            .and_then(|n| n.checked_add(row_bytes))
            .and_then(|n| n.checked_mul(NUMERIC_SLOTS))
            .and_then(|n| {
                columns
                    .div_ceil(width)
                    .checked_add(2)
                    .and_then(|groups| groups.checked_mul(2048))
                    .and_then(|headers| n.checked_add(headers))
            })
            .ok_or(CalibrationError::Allocation)
    }

    /// Admit numerical state before allocating any factor matrices.
    /// Declared nonzero source/cache identities are not independently verified here.
    pub fn new(plan: CalibrationPlan) -> Result<Self, CalibrationError> {
        let base_bytes = Self::required_numeric_bytes(&plan)?;
        admit(base_bytes, plan.max_numeric_bytes)?;
        let rows =
            usize::try_from(plan.tensor.shape[0]).map_err(|_| CalibrationError::Allocation)?;
        let columns =
            usize::try_from(plan.tensor.shape[1]).map_err(|_| CalibrationError::Allocation)?;
        let width = match plan.partition {
            CalibrationPartition::Full => columns,
            CalibrationPartition::Groups { columns: width } => usize::try_from(width)
                .map_err(|_| CalibrationError::Allocation)?
                .min(columns),
            _ => return Err(CalibrationError::Plan(CalibrationPlanError::Partition)),
        };
        let source = CurvatureSourceId::new(
            plan.source.model.0,
            plan.source.cache.0,
            plan.source.tokens.0,
        )?;
        let mut inputs = Vec::new();
        inputs
            .try_reserve_exact(columns.div_ceil(width))
            .map_err(|_| CalibrationError::Allocation)?;
        for start in (0..columns).step_by(width) {
            inputs.push(InputGramAccumulator::new_bound(
                (columns - start).min(width),
                source,
                plan.sample_start,
            )?);
        }
        let output = match plan.output {
            CalibrationOutput::None => None,
            CalibrationOutput::DenseFisher => Some(Fisher::Dense(
                OutputFisherAccumulator::new_bound(rows, source, plan.sample_start)?,
            )),
            CalibrationOutput::IndexedFisher => Some(Fisher::Indexed(
                IndexedOutputFisherAccumulator::new_bound(rows, source, plan.sample_start)?,
            )),
            _ => return Err(CalibrationError::Batch),
        };
        Ok(Self {
            plan,
            columns,
            width,
            base_bytes,
            inputs,
            output,
        })
    }

    fn try_state(&self) -> Result<(Vec<InputGramAccumulator>, Option<Fisher>), CalibrationError> {
        let mut inputs = Vec::new();
        inputs
            .try_reserve_exact(self.inputs.len())
            .map_err(|_| CalibrationError::Allocation)?;
        for input in &self.inputs {
            inputs.push(input.try_clone_transactional()?);
        }
        let output = self.output.as_ref().map(Fisher::try_clone).transpose()?;
        Ok((inputs, output))
    }

    /// Accumulate a complete pair atomically, including malformed masked values.
    /// Any failure leaves both original factors and ordinals unchanged.
    pub fn accumulate(&mut self, batch: CalibrationBatch<'_>) -> Result<(), CalibrationError> {
        match (&self.output, batch.output) {
            (None, CalibrationOutputBatch::None)
            | (Some(Fisher::Dense(_)), CalibrationOutputBatch::Dense(_))
            | (Some(Fisher::Indexed(_)), CalibrationOutputBatch::Indexed { .. }) => {}
            _ => return Err(CalibrationError::Batch),
        }
        let activations = match (self.plan.consumer, batch.input) {
            (CalibrationConsumer::Matmul, CalibrationInput::Activations(values)) => {
                let expected = batch
                    .samples
                    .checked_mul(self.columns)
                    .ok_or(CalibrationError::Allocation)?;
                if values.len() != expected {
                    return Err(CalibrationError::Batch);
                }
                Some(values)
            }
            (CalibrationConsumer::GatherIdentity, CalibrationInput::GatherIdentity) => None,
            _ => return Err(CalibrationError::Batch),
        };
        let scratch_len = if activations.is_some() && self.inputs.len() > 1 {
            batch
                .samples
                .checked_mul(self.width)
                .ok_or(CalibrationError::Allocation)?
        } else {
            0
        };
        let scratch_bytes = u64::try_from(scratch_len)
            .ok()
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| self.base_bytes.checked_add(n))
            .ok_or(CalibrationError::Allocation)?;
        admit(scratch_bytes, self.plan.max_numeric_bytes)?;
        let mut scratch = Vec::new();
        scratch
            .try_reserve_exact(scratch_len)
            .map_err(|_| CalibrationError::Allocation)?;
        let (mut inputs, mut output) = self.try_state()?;
        for (group, input) in inputs.iter_mut().enumerate() {
            if let Some(values) = activations {
                if self.inputs.len() == 1 {
                    input.accumulate_batch(values, batch.samples, batch.weights, batch.mask)?;
                } else {
                    let start = group * self.width;
                    let width = (self.columns - start).min(self.width);
                    scratch.clear();
                    for sample in values.chunks_exact(self.columns) {
                        scratch.extend_from_slice(&sample[start..start + width]);
                    }
                    input.accumulate_batch(&scratch, batch.samples, batch.weights, batch.mask)?;
                }
            } else {
                input.accumulate_identity_batch(batch.samples, batch.weights, batch.mask)?;
            }
        }
        match (&mut output, batch.output) {
            (Some(Fisher::Dense(f)), CalibrationOutputBatch::Dense(values)) => {
                f.accumulate_batch(values, batch.samples, batch.weights, batch.mask)?
            }
            (Some(Fisher::Indexed(f)), CalibrationOutputBatch::Indexed { indices, factors }) => {
                f.accumulate_batch(indices, factors, batch.samples, batch.weights, batch.mask)?
            }
            (None, CalibrationOutputBatch::None) => {}
            _ => return Err(CalibrationError::Batch),
        }
        ensure_segments(&inputs, output.as_ref())?;
        self.inputs = inputs;
        self.output = output;
        Ok(())
    }

    /// Atomically merge a same-contract shard at its actual declared start ordinal.
    /// Existing numerical reducers reject overlaps, gaps at finish and source mismatch.
    pub fn merge_shard(&mut self, shard: &Self) -> Result<(), CalibrationError> {
        let mut expected = self.plan.clone();
        expected.sample_start = shard.plan.sample_start;
        if expected != shard.plan {
            return Err(CalibrationError::Shard);
        }
        let (mut inputs, mut output) = self.try_state()?;
        for (input, other) in inputs.iter_mut().zip(&shard.inputs) {
            input.merge_shard(shard.plan.sample_start, other)?;
        }
        match (&mut output, &shard.output) {
            (Some(Fisher::Dense(f)), Some(Fisher::Dense(other))) => {
                f.merge_shard(shard.plan.sample_start, other)?
            }
            (Some(Fisher::Indexed(f)), Some(Fisher::Indexed(other))) => {
                f.merge_shard(shard.plan.sample_start, other)?
            }
            (None, None) => {}
            _ => return Err(CalibrationError::Shard),
        }
        ensure_segments(&inputs, output.as_ref())?;
        self.inputs = inputs;
        self.output = output;
        Ok(())
    }

    /// Finalize actual normalized factors and their typed immutable view.
    /// Zero selected weight or mismatched source/selection is rejected.
    pub fn finish(&self) -> Result<CalibratedTensor, CalibrationError> {
        let mut inputs = Vec::new();
        inputs
            .try_reserve_exact(self.inputs.len())
            .map_err(|_| CalibrationError::Allocation)?;
        for input in &self.inputs {
            inputs.push(input.finish()?);
        }
        let output = self.output.as_ref().map(Fisher::finish).transpose()?;
        let first = &inputs[0];
        for input in &inputs {
            if input.source_id() != first.source_id()
                || input.selection_digest() != first.selection_digest()
                || input.sample_count() != first.sample_count()
                || input.selected_count() != first.selected_count()
                || input.total_weight().to_bits() != first.total_weight().to_bits()
            {
                return Err(CalibrationError::Selection);
            }
        }
        if let Some(f) = &output
            && (f.source_id() != first.source_id()
                || f.selection_digest() != first.selection_digest()
                || f.sample_count() != first.sample_count()
                || f.selected_count() != first.selected_count()
                || f.total_weight().to_bits() != first.total_weight().to_bits())
        {
            return Err(CalibrationError::Selection);
        }
        let mut bindings = Vec::new();
        bindings
            .try_reserve_exact(inputs.len())
            .map_err(|_| CalibrationError::Allocation)?;
        for input in &inputs {
            bindings.push(CalibrationFactor {
                dimension: input.dimension() as u64,
                samples: input.sample_count(),
                selected: input.selected_count(),
                total_weight: input.total_weight(),
                selection: FitInputDigest(*input.selection_digest()),
                content: FitInputDigest(input.digest()),
            });
        }
        let output_binding = output.as_ref().map(|f| CalibrationFactor {
            dimension: f.output_rows() as u64,
            samples: f.sample_count(),
            selected: f.selected_count(),
            total_weight: f.total_weight(),
            selection: FitInputDigest(*f.selection_digest()),
            content: FitInputDigest(f.digest()),
        });
        let factors = CalibrationFactors {
            plan: self.plan.clone(),
            inputs: bindings,
            output: output_binding,
        };
        Ok(CalibratedTensor {
            factors,
            inputs,
            output,
        })
    }
}

fn admit(required: u64, limit: u64) -> Result<(), CalibrationError> {
    if required > limit {
        Err(CalibrationError::Budget { required, limit })
    } else {
        Ok(())
    }
}
fn ensure_segments(
    inputs: &[InputGramAccumulator],
    output: Option<&Fisher>,
) -> Result<(), CalibrationError> {
    if inputs
        .iter()
        .any(|g| g.retained_reduction_segments() > MAX_SEGMENTS)
        || output.is_some_and(|f| f.segments() > MAX_SEGMENTS)
    {
        Err(CalibrationError::Segments)
    } else {
        Ok(())
    }
}

/// Actual factors and read-only binding view; no mutable factor/identity access.
#[derive(Debug)]
pub struct CalibratedTensor {
    factors: CalibrationFactors,
    inputs: Vec<InputGram>,
    output: Option<OutputFisher>,
}
impl CalibratedTensor {
    /// Typed actual request/content view, not independently cleared evidence.
    pub fn factors(&self) -> &CalibrationFactors {
        &self.factors
    }
    /// Normalized original-input groups in declared order.
    pub fn inputs(&self) -> &[InputGram] {
        &self.inputs
    }
    /// Actual output factor, if requested and observed.
    pub fn output(&self) -> Option<&OutputFisher> {
        self.output.as_ref()
    }
    /// Fold the diagonal from these same actual normalized Grams, without allocation.
    /// Invalid destination length fails before any output write.
    pub fn diagonal_into(&self, destination: &mut [f64]) -> Result<(), CalibrationError> {
        if destination.len() as u64 != self.factors.plan.tensor.shape[1] {
            return Err(CalibrationError::Shape);
        }
        let mut start = 0;
        for input in &self.inputs {
            let width = input.dimension();
            for i in 0..width {
                destination[start + i] = input.as_slice()[i * width + i];
            }
            start += width;
        }
        Ok(())
    }
    /// Build one actual output row's group K-FAC using the existing validated builder.
    /// Only one matrix is allocated per call; retained caller matrices are outside
    /// collector admission. Missing factors and invalid group/row/PSD are errors.
    pub fn kfac_metric(&self, group: usize, row: usize) -> Result<KfacMetric, CalibrationError> {
        let input = self.inputs.get(group).ok_or(CalibrationError::Shape)?;
        let output = self
            .output
            .as_ref()
            .ok_or(CalibrationError::MissingFisher)?;
        Ok(build_kfac_metric(
            input,
            output,
            row,
            self.factors.plan.damping,
        )?)
    }
}
