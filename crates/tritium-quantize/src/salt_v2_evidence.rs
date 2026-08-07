//! Durable factorized curvature evidence for bounded-memory SALT V2 fitting.

use core::{fmt, mem::size_of};
use std::io::{self, Read, Write};

use crate::{
    CurvatureArtifact, CurvatureError, CurvatureSourceId, DensePsdMetric,
    IndexedOutputFisherAccumulator, InputGram, InputGramAccumulator, OutputFisher,
    OutputFisherAccumulator, SaltV2Curvature, SaltV2TensorFitInput,
};

const MAGIC: [u8; 4] = *b"S2KF";
const VERSION: u16 = 1;
const CHECKSUM_CONTEXT: &str = "tritium salt v2 kronecker evidence checksum v1";
const CHECKSUM_BYTES: usize = 32;
const MAX_NAME_BYTES: usize = 1024 * 1024;
const GROUP_SIZE: usize = 128;
const FIXED_PAYLOAD_BYTES: usize = 184;
const GROUP_PAYLOAD_BYTES: usize = size_of::<u32>() + GROUP_SIZE * GROUP_SIZE * size_of::<f64>();
const BUILDER_DIGEST_CONTEXT: &str = "tritium salt v2 kronecker evidence builder v1";
/// Default hard ceiling for one canonical grouped-curvature record.
pub const DEFAULT_MAX_KRONECKER_EVIDENCE_BYTES: u64 = 64 * 1024 * 1024;
/// Maximum dyadic reduction segments retained by any one factor accumulator.
pub const MAX_KRONECKER_REDUCTION_SEGMENTS: usize = 64;

/// Immutable tensor and provenance contract for streamed Kronecker evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct SaltV2KroneckerEvidenceSpec {
    kind: SaltV2Curvature,
    source_id: CurvatureSourceId,
    tensor_index: u64,
    tensor_name: String,
    rows: usize,
    columns: usize,
    damping: f64,
    canonical_bytes: u64,
}

impl SaltV2KroneckerEvidenceSpec {
    /// Validate one grouped evidence-production contract.
    ///
    /// # Errors
    /// Rejects unsupported curvature, invalid names or geometry, and invalid damping.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kind: SaltV2Curvature,
        source_id: CurvatureSourceId,
        tensor_index: u64,
        tensor_name: impl Into<String>,
        rows: usize,
        columns: usize,
        damping: f64,
    ) -> Result<Self, SaltV2KroneckerEvidenceBuildError> {
        Self::new_bounded(
            kind,
            source_id,
            tensor_index,
            tensor_name,
            rows,
            columns,
            damping,
            DEFAULT_MAX_KRONECKER_EVIDENCE_BYTES,
        )
    }

    /// Validate one contract under an explicit canonical-record byte ceiling.
    ///
    /// # Errors
    /// Rejects unsupported curvature, invalid names or geometry, invalid
    /// damping, zero limit, arithmetic overflow, or record size above limit.
    #[allow(clippy::too_many_arguments)]
    pub fn new_bounded(
        kind: SaltV2Curvature,
        source_id: CurvatureSourceId,
        tensor_index: u64,
        tensor_name: impl Into<String>,
        rows: usize,
        columns: usize,
        damping: f64,
        max_record_bytes: u64,
    ) -> Result<Self, SaltV2KroneckerEvidenceBuildError> {
        if !matches!(
            kind,
            SaltV2Curvature::InputHessian
                | SaltV2Curvature::GuidedFisher
                | SaltV2Curvature::ForwardKlKronecker
        ) {
            return Err(SaltV2KroneckerEvidenceBuildError::Malformed(
                "curvature kind",
            ));
        }
        let tensor_name = tensor_name.into();
        if tensor_name.is_empty() || tensor_name.len() > MAX_NAME_BYTES {
            return Err(SaltV2KroneckerEvidenceBuildError::Malformed("tensor name"));
        }
        if rows == 0 || columns == 0 || !columns.is_multiple_of(GROUP_SIZE) {
            return Err(SaltV2KroneckerEvidenceBuildError::Malformed(
                "tensor geometry",
            ));
        }
        if !damping.is_finite() || damping < 0.0 {
            return Err(SaltV2KroneckerEvidenceBuildError::Malformed("damping"));
        }
        if max_record_bytes == 0 {
            return Err(SaltV2KroneckerEvidenceBuildError::Malformed(
                "record byte limit",
            ));
        }
        let group_count = columns / GROUP_SIZE;
        let canonical_bytes = payload_len(tensor_name.len(), group_count, rows)
            .and_then(|bytes| bytes.checked_add(CHECKSUM_BYTES))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(SaltV2KroneckerEvidenceBuildError::Malformed(
                "canonical record length",
            ))?;
        if canonical_bytes > max_record_bytes {
            return Err(SaltV2KroneckerEvidenceBuildError::SizeLimitExceeded {
                required_bytes: canonical_bytes,
                max_bytes: max_record_bytes,
            });
        }
        Ok(Self {
            kind,
            source_id,
            tensor_index,
            tensor_name,
            rows,
            columns,
            damping: canonical_zero(damping),
            canonical_bytes,
        })
    }

    /// Curvature estimator represented by produced evidence.
    #[must_use]
    pub const fn kind(&self) -> SaltV2Curvature {
        self.kind
    }

    /// Immutable checkpoint/cache/token-stream identity.
    #[must_use]
    pub const fn source_id(&self) -> CurvatureSourceId {
        self.source_id
    }

    /// Global architecture tensor ordinal.
    #[must_use]
    pub const fn tensor_index(&self) -> u64 {
        self.tensor_index
    }

    /// Canonical source tensor name.
    #[must_use]
    pub fn tensor_name(&self) -> &str {
        &self.tensor_name
    }

    /// Output row count.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// G128-aligned input column count.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Diagonal damping applied after Kronecker scaling.
    #[must_use]
    pub const fn damping(&self) -> f64 {
        self.damping
    }

    /// Exact canonical S2KF bytes implied by this contract.
    #[must_use]
    pub const fn canonical_bytes(&self) -> u64 {
        self.canonical_bytes
    }
}

/// Streaming, source-bound producer for one factorized S2KF tensor record.
///
/// Producer keeps one G128 Gram accumulator per input group plus one output-row
/// accumulator. Batch mutation is atomic: any malformed activation, gradient,
/// weight, mask, or allocation leaves prior evidence unchanged.
#[derive(Debug)]
pub struct SaltV2KroneckerEvidenceBuilder {
    spec: SaltV2KroneckerEvidenceSpec,
    input_groups: Vec<InputGramAccumulator>,
    output_fisher: Option<OutputFisherBuilder>,
}

#[derive(Debug)]
enum OutputFisherBuilder {
    Dense(OutputFisherAccumulator),
    Indexed(IndexedOutputFisherAccumulator),
}

impl OutputFisherBuilder {
    fn retained_reduction_segments(&self) -> usize {
        match self {
            Self::Dense(accumulator) => accumulator.retained_reduction_segments(),
            Self::Indexed(accumulator) => accumulator.retained_reduction_segments(),
        }
    }

    fn try_clone_transactional(&self) -> Result<Self, CurvatureError> {
        match self {
            Self::Dense(accumulator) => accumulator.try_clone_transactional().map(Self::Dense),
            Self::Indexed(accumulator) => accumulator.try_clone_transactional().map(Self::Indexed),
        }
    }

    fn finish(&self) -> Result<OutputFisher, CurvatureError> {
        match self {
            Self::Dense(accumulator) => accumulator.finish(),
            Self::Indexed(accumulator) => accumulator.finish(),
        }
    }

    fn into_fisher(self) -> Result<OutputFisher, CurvatureError> {
        match self {
            Self::Dense(accumulator) => accumulator.finish(),
            Self::Indexed(accumulator) => accumulator.finish(),
        }
    }
}

/// Exact retained reduction-tree state for one producer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SaltV2KroneckerEvidenceResidency {
    input_segments: usize,
    output_segments: usize,
}

impl SaltV2KroneckerEvidenceResidency {
    /// Total segments across all G128 input accumulators.
    #[must_use]
    pub const fn input_segments(self) -> usize {
        self.input_segments
    }

    /// Segments in the optional output-row accumulator.
    #[must_use]
    pub const fn output_segments(self) -> usize {
        self.output_segments
    }
}

impl SaltV2KroneckerEvidenceBuilder {
    /// Start evidence production at global sample ordinal zero.
    ///
    /// # Errors
    /// Rejects accumulator geometry or allocation failure.
    pub fn new(
        spec: SaltV2KroneckerEvidenceSpec,
    ) -> Result<Self, SaltV2KroneckerEvidenceBuildError> {
        Self::new_at(spec, 0)
    }

    /// Start one independently mergeable shard at a declared sample ordinal.
    ///
    /// # Errors
    /// Rejects accumulator geometry or allocation failure.
    pub fn new_at(
        spec: SaltV2KroneckerEvidenceSpec,
        sample_start: u64,
    ) -> Result<Self, SaltV2KroneckerEvidenceBuildError> {
        Self::new_at_with_output_encoding(spec, sample_start, false)
    }

    /// Start sparse indexed output-factor production at ordinal zero.
    ///
    /// The indexed contract is intended for embedding tables: every sample
    /// names one output row and one scalar factor, avoiding a dense vocabulary
    /// vector while producing numerically identical Fisher values under a
    /// domain-separated sparse provenance identity.
    ///
    /// # Errors
    /// Rejects input-Hessian curvature, invalid geometry, or allocation failure.
    pub fn new_indexed_output(
        spec: SaltV2KroneckerEvidenceSpec,
    ) -> Result<Self, SaltV2KroneckerEvidenceBuildError> {
        Self::new_indexed_output_at(spec, 0)
    }

    /// Start one independently mergeable indexed-output shard.
    ///
    /// # Errors
    /// Rejects input-Hessian curvature, invalid geometry, or allocation failure.
    pub fn new_indexed_output_at(
        spec: SaltV2KroneckerEvidenceSpec,
        sample_start: u64,
    ) -> Result<Self, SaltV2KroneckerEvidenceBuildError> {
        if matches!(spec.kind, SaltV2Curvature::InputHessian) {
            return Err(SaltV2KroneckerEvidenceBuildError::WrongOutputFactorEncoding);
        }
        Self::new_at_with_output_encoding(spec, sample_start, true)
    }

    fn new_at_with_output_encoding(
        spec: SaltV2KroneckerEvidenceSpec,
        sample_start: u64,
        indexed_output: bool,
    ) -> Result<Self, SaltV2KroneckerEvidenceBuildError> {
        let group_count = spec.columns / GROUP_SIZE;
        let mut input_groups = Vec::new();
        input_groups
            .try_reserve_exact(group_count)
            .map_err(|_| SaltV2KroneckerEvidenceBuildError::AllocationFailed)?;
        for _ in 0..group_count {
            input_groups.push(InputGramAccumulator::new_bound(
                GROUP_SIZE,
                spec.source_id,
                sample_start,
            )?);
        }
        let output_fisher = if matches!(spec.kind, SaltV2Curvature::InputHessian) {
            None
        } else if indexed_output {
            Some(OutputFisherBuilder::Indexed(
                IndexedOutputFisherAccumulator::new_bound(spec.rows, spec.source_id, sample_start)?,
            ))
        } else {
            Some(OutputFisherBuilder::Dense(
                OutputFisherAccumulator::new_bound(spec.rows, spec.source_id, sample_start)?,
            ))
        };
        Ok(Self {
            spec,
            input_groups,
            output_fisher,
        })
    }

    /// Immutable tensor/provenance contract.
    #[must_use]
    pub const fn spec(&self) -> &SaltV2KroneckerEvidenceSpec {
        &self.spec
    }

    /// Report exact retained dyadic segments without materializing factors.
    #[must_use]
    pub fn residency(&self) -> SaltV2KroneckerEvidenceResidency {
        SaltV2KroneckerEvidenceResidency {
            input_segments: self
                .input_groups
                .iter()
                .map(InputGramAccumulator::retained_reduction_segments)
                .sum(),
            output_segments: self
                .output_fisher
                .as_ref()
                .map_or(0, OutputFisherBuilder::retained_reduction_segments),
        }
    }

    /// Atomically add one row-major activation/output-factor batch.
    ///
    /// Input-Hessian evidence requires no output factors. Guided-Fisher and
    /// forward-KL Kronecker evidence require one factor per sample and output row.
    ///
    /// # Errors
    /// Rejects wrong factor presence, malformed shapes, non-finite values,
    /// invalid masks/weights, sample overflow, or allocation failure.
    pub fn accumulate_batch(
        &mut self,
        activations: &[f32],
        output_factors: Option<&[f32]>,
        samples: usize,
        token_weights: Option<&[f64]>,
        token_mask: Option<&[bool]>,
    ) -> Result<(), SaltV2KroneckerEvidenceBuildError> {
        match (&self.output_fisher, output_factors) {
            (Some(OutputFisherBuilder::Dense(_)), None) => {
                return Err(SaltV2KroneckerEvidenceBuildError::MissingOutputFactors);
            }
            (Some(OutputFisherBuilder::Indexed(_)), _) => {
                return Err(SaltV2KroneckerEvidenceBuildError::WrongOutputFactorEncoding);
            }
            (None, Some(_)) => {
                return Err(SaltV2KroneckerEvidenceBuildError::UnexpectedOutputFactors);
            }
            _ => {}
        }
        let expected_activations = samples
            .checked_mul(self.spec.columns)
            .ok_or(SaltV2KroneckerEvidenceBuildError::AllocationFailed)?;
        if activations.len() != expected_activations {
            return Err(SaltV2KroneckerEvidenceBuildError::BatchLengthMismatch {
                field: "activations",
                expected: expected_activations,
                got: activations.len(),
            });
        }
        if samples == 0 {
            return Err(CurvatureError::EmptyBatch.into());
        }
        if let Some(weights) = token_weights {
            if weights.len() != samples {
                return Err(CurvatureError::TokenWeightLengthMismatch {
                    expected: samples,
                    got: weights.len(),
                }
                .into());
            }
            if let Some(sample) = weights
                .iter()
                .position(|weight| !weight.is_finite() || *weight < 0.0)
            {
                return Err(CurvatureError::InvalidTokenWeight { sample }.into());
            }
        }
        if let Some(mask) = token_mask
            && mask.len() != samples
        {
            return Err(CurvatureError::TokenMaskLengthMismatch {
                expected: samples,
                got: mask.len(),
            }
            .into());
        }
        if let Some(index) = activations.iter().position(|value| !value.is_finite()) {
            return Err(CurvatureError::NonFiniteActivation {
                sample: index / self.spec.columns,
                feature: index % self.spec.columns,
            }
            .into());
        }
        if let Some(factors) = output_factors {
            let expected_factors = samples
                .checked_mul(self.spec.rows)
                .ok_or(SaltV2KroneckerEvidenceBuildError::AllocationFailed)?;
            if factors.len() != expected_factors {
                return Err(SaltV2KroneckerEvidenceBuildError::BatchLengthMismatch {
                    field: "output factors",
                    expected: expected_factors,
                    got: factors.len(),
                });
            }
            if let Some(index) = factors.iter().position(|value| !value.is_finite()) {
                return Err(CurvatureError::NonFiniteGradient {
                    sample: index / self.spec.rows,
                    output_row: index % self.spec.rows,
                }
                .into());
            }
        }

        let mut next_output = try_clone_output_accumulator(self.output_fisher.as_ref())?;
        if let (Some(OutputFisherBuilder::Dense(accumulator)), Some(factors)) =
            (&mut next_output, output_factors)
        {
            accumulator.accumulate_batch(factors, samples, token_weights, token_mask)?;
        }
        let mut next_inputs = try_clone_input_accumulators(&self.input_groups)?;
        let scratch_len = samples
            .checked_mul(GROUP_SIZE)
            .ok_or(SaltV2KroneckerEvidenceBuildError::AllocationFailed)?;
        let mut scratch = Vec::new();
        scratch
            .try_reserve_exact(scratch_len)
            .map_err(|_| SaltV2KroneckerEvidenceBuildError::AllocationFailed)?;
        for (group_index, accumulator) in next_inputs.iter_mut().enumerate() {
            scratch.clear();
            let column_start = group_index * GROUP_SIZE;
            for sample in 0..samples {
                let row_start = sample * self.spec.columns + column_start;
                scratch.extend_from_slice(&activations[row_start..row_start + GROUP_SIZE]);
            }
            accumulator.accumulate_batch(&scratch, samples, token_weights, token_mask)?;
        }
        ensure_segment_limits(&next_inputs, next_output.as_ref())?;
        self.input_groups = next_inputs;
        self.output_fisher = next_output;
        Ok(())
    }

    /// Atomically add one sparse indexed output-factor batch.
    ///
    /// Each sample names exactly one output row and scalar factor. The input
    /// activations retain the normal row-major `[samples, columns]` layout.
    /// Its Fisher values match a dense one-hot output-factor batch, while its
    /// provenance digest explicitly records sparse indexed capture.
    ///
    /// # Errors
    /// Rejects a dense/input-Hessian builder, malformed arrays, out-of-range
    /// output rows, non-finite values, invalid masks/weights, or allocation
    /// failure without changing prior evidence.
    #[allow(clippy::too_many_arguments)]
    pub fn accumulate_indexed_output_batch(
        &mut self,
        activations: &[f32],
        output_indices: &[usize],
        output_factors: &[f32],
        samples: usize,
        token_weights: Option<&[f64]>,
        token_mask: Option<&[bool]>,
    ) -> Result<(), SaltV2KroneckerEvidenceBuildError> {
        if !matches!(
            self.output_fisher.as_ref(),
            Some(OutputFisherBuilder::Indexed(_))
        ) {
            return Err(SaltV2KroneckerEvidenceBuildError::WrongOutputFactorEncoding);
        }
        let expected_activations = samples
            .checked_mul(self.spec.columns)
            .ok_or(SaltV2KroneckerEvidenceBuildError::AllocationFailed)?;
        if activations.len() != expected_activations {
            return Err(SaltV2KroneckerEvidenceBuildError::BatchLengthMismatch {
                field: "activations",
                expected: expected_activations,
                got: activations.len(),
            });
        }
        if samples == 0 {
            return Err(CurvatureError::EmptyBatch.into());
        }
        if let Some(index) = activations.iter().position(|value| !value.is_finite()) {
            return Err(CurvatureError::NonFiniteActivation {
                sample: index / self.spec.columns,
                feature: index % self.spec.columns,
            }
            .into());
        }

        let mut next_output = try_clone_output_accumulator(self.output_fisher.as_ref())?;
        let Some(OutputFisherBuilder::Indexed(accumulator)) = &mut next_output else {
            return Err(SaltV2KroneckerEvidenceBuildError::WrongOutputFactorEncoding);
        };
        accumulator.accumulate_batch(
            output_indices,
            output_factors,
            samples,
            token_weights,
            token_mask,
        )?;
        let mut next_inputs = try_clone_input_accumulators(&self.input_groups)?;
        let scratch_len = samples
            .checked_mul(GROUP_SIZE)
            .ok_or(SaltV2KroneckerEvidenceBuildError::AllocationFailed)?;
        let mut scratch = Vec::new();
        scratch
            .try_reserve_exact(scratch_len)
            .map_err(|_| SaltV2KroneckerEvidenceBuildError::AllocationFailed)?;
        for (group_index, input) in next_inputs.iter_mut().enumerate() {
            scratch.clear();
            let column_start = group_index * GROUP_SIZE;
            for sample in 0..samples {
                let row_start = sample * self.spec.columns + column_start;
                scratch.extend_from_slice(&activations[row_start..row_start + GROUP_SIZE]);
            }
            input.accumulate_batch(&scratch, samples, token_weights, token_mask)?;
        }
        ensure_segment_limits(&next_inputs, next_output.as_ref())?;
        self.input_groups = next_inputs;
        self.output_fisher = next_output;
        Ok(())
    }

    /// Atomically merge one independently accumulated canonical sample shard.
    ///
    /// Shards may arrive out of processing order. `declared_order` must equal
    /// shard's first global sample ordinal; finalization restores canonical order.
    ///
    /// # Errors
    /// Rejects tensor/provenance drift, wrong shard order, gaps, overlaps,
    /// source mismatch, or allocation failure without changing prior evidence.
    pub fn merge_shard(
        &mut self,
        declared_order: u64,
        shard: &Self,
    ) -> Result<(), SaltV2KroneckerEvidenceBuildError> {
        if self.spec != shard.spec {
            return Err(SaltV2KroneckerEvidenceBuildError::SpecMismatch);
        }
        let mut next_inputs = try_clone_input_accumulators(&self.input_groups)?;
        let mut next_output = try_clone_output_accumulator(self.output_fisher.as_ref())?;
        for (destination, source) in next_inputs.iter_mut().zip(&shard.input_groups) {
            destination.merge_shard(declared_order, source)?;
        }
        match (&mut next_output, &shard.output_fisher) {
            (
                Some(OutputFisherBuilder::Dense(destination)),
                Some(OutputFisherBuilder::Dense(source)),
            ) => {
                destination.merge_shard(declared_order, source)?;
            }
            (
                Some(OutputFisherBuilder::Indexed(destination)),
                Some(OutputFisherBuilder::Indexed(source)),
            ) => destination.merge_shard(declared_order, source)?,
            (None, None) => {}
            _ => return Err(SaltV2KroneckerEvidenceBuildError::SpecMismatch),
        }
        ensure_segment_limits(&next_inputs, next_output.as_ref())?;
        self.input_groups = next_inputs;
        self.output_fisher = next_output;
        Ok(())
    }

    /// Finalize exact factorized curvature and its upstream builder identity.
    ///
    /// # Errors
    /// Rejects empty/misaligned evidence, invalid PSD groups, invalid output
    /// curvature, allocation failure, or canonical record failure.
    pub fn finish(&self) -> Result<SaltV2KroneckerEvidence, SaltV2KroneckerEvidenceBuildError> {
        let mut grams = Vec::new();
        grams
            .try_reserve_exact(self.input_groups.len())
            .map_err(|_| SaltV2KroneckerEvidenceBuildError::AllocationFailed)?;
        for accumulator in &self.input_groups {
            grams.push(accumulator.finish()?);
        }
        let fisher = self
            .output_fisher
            .as_ref()
            .map(OutputFisherBuilder::finish)
            .transpose()?;
        finalize_builder_evidence(&self.spec, grams, fisher)
    }

    /// Consume producer state and finalize one canonical record at lower peak residency.
    ///
    /// Each accumulator is released after its normalized factor is materialized,
    /// before canonical record encoding begins.
    ///
    /// # Errors
    /// Rejects same malformed or incomplete evidence as [`Self::finish`].
    pub fn into_evidence(
        self,
    ) -> Result<SaltV2KroneckerEvidence, SaltV2KroneckerEvidenceBuildError> {
        let Self {
            spec,
            input_groups,
            output_fisher,
        } = self;
        let mut grams = Vec::new();
        grams
            .try_reserve_exact(input_groups.len())
            .map_err(|_| SaltV2KroneckerEvidenceBuildError::AllocationFailed)?;
        for accumulator in input_groups {
            grams.push(accumulator.finish()?);
        }
        let fisher = output_fisher
            .map(OutputFisherBuilder::into_fisher)
            .transpose()?;
        finalize_builder_evidence(&spec, grams, fisher)
    }
}

fn ensure_segment_limits(
    input_groups: &[InputGramAccumulator],
    output_fisher: Option<&OutputFisherBuilder>,
) -> Result<(), SaltV2KroneckerEvidenceBuildError> {
    let retained = input_groups
        .iter()
        .map(InputGramAccumulator::retained_reduction_segments)
        .chain(
            output_fisher
                .into_iter()
                .map(OutputFisherBuilder::retained_reduction_segments),
        )
        .max()
        .unwrap_or(0);
    if retained > MAX_KRONECKER_REDUCTION_SEGMENTS {
        return Err(
            SaltV2KroneckerEvidenceBuildError::ReductionSegmentLimitExceeded {
                retained,
                max_segments: MAX_KRONECKER_REDUCTION_SEGMENTS,
            },
        );
    }
    Ok(())
}

fn try_clone_input_accumulators(
    accumulators: &[InputGramAccumulator],
) -> Result<Vec<InputGramAccumulator>, SaltV2KroneckerEvidenceBuildError> {
    let mut cloned = Vec::new();
    cloned
        .try_reserve_exact(accumulators.len())
        .map_err(|_| SaltV2KroneckerEvidenceBuildError::AllocationFailed)?;
    for accumulator in accumulators {
        cloned.push(
            accumulator
                .try_clone_transactional()
                .map_err(|_| SaltV2KroneckerEvidenceBuildError::AllocationFailed)?,
        );
    }
    Ok(cloned)
}

fn try_clone_output_accumulator(
    accumulator: Option<&OutputFisherBuilder>,
) -> Result<Option<OutputFisherBuilder>, SaltV2KroneckerEvidenceBuildError> {
    accumulator
        .map(|accumulator| {
            accumulator
                .try_clone_transactional()
                .map_err(|_| SaltV2KroneckerEvidenceBuildError::AllocationFailed)
        })
        .transpose()
}

fn finalize_builder_evidence(
    spec: &SaltV2KroneckerEvidenceSpec,
    grams: Vec<InputGram>,
    fisher: Option<OutputFisher>,
) -> Result<SaltV2KroneckerEvidence, SaltV2KroneckerEvidenceBuildError> {
    let first = grams
        .first()
        .ok_or(SaltV2KroneckerEvidenceBuildError::Malformed(
            "input group count",
        ))?;
    for gram in &grams[1..] {
        ensure_matching_selection(first, gram)?;
    }
    if let Some(fisher) = fisher.as_ref() {
        ensure_fisher_selection(first, fisher)?;
    }

    let mut input_metrics = Vec::new();
    input_metrics
        .try_reserve_exact(grams.len())
        .map_err(|_| SaltV2KroneckerEvidenceBuildError::AllocationFailed)?;
    for gram in &grams {
        input_metrics.push(
            DensePsdMetric::new(GROUP_SIZE, gram.as_slice())
                .map_err(|_| SaltV2KroneckerEvidenceBuildError::Malformed("input group metric"))?,
        );
    }
    let mut output_weights = Vec::new();
    output_weights
        .try_reserve_exact(spec.rows)
        .map_err(|_| SaltV2KroneckerEvidenceBuildError::AllocationFailed)?;
    match fisher.as_ref() {
        Some(fisher) => output_weights.extend_from_slice(fisher.as_slice()),
        None => output_weights.resize(spec.rows, 1.0),
    }
    let upstream_evidence_digest = builder_digest(spec, &grams, fisher.as_ref());
    SaltV2KroneckerEvidence::new(
        spec.kind,
        spec.source_id,
        upstream_evidence_digest,
        spec.tensor_index,
        spec.tensor_name.clone(),
        spec.rows,
        spec.columns,
        input_metrics,
        output_weights,
        spec.damping,
    )
    .map_err(SaltV2KroneckerEvidenceBuildError::Evidence)
}

fn ensure_matching_selection(
    expected: &InputGram,
    got: &InputGram,
) -> Result<(), SaltV2KroneckerEvidenceBuildError> {
    if expected.source_id() != got.source_id()
        || expected.sample_count() != got.sample_count()
        || expected.selected_count() != got.selected_count()
        || expected.total_weight().to_bits() != got.total_weight().to_bits()
        || expected.selection_digest() != got.selection_digest()
    {
        return Err(SaltV2KroneckerEvidenceBuildError::SelectionMismatch);
    }
    Ok(())
}

fn ensure_fisher_selection(
    expected: &InputGram,
    got: &OutputFisher,
) -> Result<(), SaltV2KroneckerEvidenceBuildError> {
    if expected.source_id() != got.source_id()
        || expected.sample_count() != got.sample_count()
        || expected.selected_count() != got.selected_count()
        || expected.total_weight().to_bits() != got.total_weight().to_bits()
        || expected.selection_digest() != got.selection_digest()
    {
        return Err(SaltV2KroneckerEvidenceBuildError::SelectionMismatch);
    }
    Ok(())
}

fn builder_digest(
    spec: &SaltV2KroneckerEvidenceSpec,
    grams: &[InputGram],
    fisher: Option<&OutputFisher>,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key(BUILDER_DIGEST_CONTEXT);
    hasher.update(&[kind_tag(spec.kind)]);
    hasher.update(&spec.source_id.digest());
    hasher.update(&spec.tensor_index.to_le_bytes());
    hasher.update(&(spec.tensor_name.len() as u64).to_le_bytes());
    hasher.update(spec.tensor_name.as_bytes());
    hasher.update(&(spec.rows as u64).to_le_bytes());
    hasher.update(&(spec.columns as u64).to_le_bytes());
    hasher.update(&spec.damping.to_bits().to_le_bytes());
    hasher.update(&(grams.len() as u64).to_le_bytes());
    for gram in grams {
        hasher.update(&gram.digest());
    }
    match fisher {
        Some(fisher) => {
            hasher.update(&[1]);
            hasher.update(&fisher.digest());
        }
        None => {
            hasher.update(&[0]);
        }
    };
    *hasher.finalize().as_bytes()
}

/// Failure while producing one streamed factorized-curvature record.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SaltV2KroneckerEvidenceBuildError {
    /// Static producer contract was malformed.
    Malformed(&'static str),
    /// Input-Hessian evidence unexpectedly received output factors.
    UnexpectedOutputFactors,
    /// Fisher/KL evidence omitted required output factors.
    MissingOutputFactors,
    /// The builder and append operation used different output-factor encodings.
    WrongOutputFactorEncoding,
    /// Row-major batch storage did not match declared samples and geometry.
    BatchLengthMismatch {
        /// Malformed input field.
        field: &'static str,
        /// Required scalar count.
        expected: usize,
        /// Supplied scalar count.
        got: usize,
    },
    /// Exact canonical record exceeds configured resource ceiling.
    SizeLimitExceeded {
        /// Exact bytes required by geometry and name.
        required_bytes: u64,
        /// Maximum admitted bytes.
        max_bytes: u64,
    },
    /// Out-of-order reduction state exceeded its fixed residency ceiling.
    ReductionSegmentLimitExceeded {
        /// Segments the operation would retain in one accumulator.
        retained: usize,
        /// Maximum admitted segments per accumulator.
        max_segments: usize,
    },
    /// Shard tensor, estimator, geometry, or provenance differed.
    SpecMismatch,
    /// Input groups and output factors selected different samples.
    SelectionMismatch,
    /// Curvature accumulation failed.
    Curvature(CurvatureError),
    /// Canonical S2KF construction failed.
    Evidence(SaltV2KroneckerEvidenceError),
    /// Bounded owned storage could not be allocated.
    AllocationFailed,
}

impl From<CurvatureError> for SaltV2KroneckerEvidenceBuildError {
    fn from(error: CurvatureError) -> Self {
        Self::Curvature(error)
    }
}

impl fmt::Display for SaltV2KroneckerEvidenceBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Malformed(field) => write!(formatter, "malformed Kronecker builder {field}"),
            Self::UnexpectedOutputFactors => {
                formatter.write_str("input-Hessian builder rejects output factors")
            }
            Self::MissingOutputFactors => {
                formatter.write_str("Fisher/KL builder requires output factors")
            }
            Self::WrongOutputFactorEncoding => formatter.write_str(
                "Kronecker builder output-factor encoding does not match append operation",
            ),
            Self::BatchLengthMismatch {
                field,
                expected,
                got,
            } => write!(
                formatter,
                "Kronecker builder {field} length mismatch: expected {expected}, got {got}"
            ),
            Self::SizeLimitExceeded {
                required_bytes,
                max_bytes,
            } => write!(
                formatter,
                "Kronecker evidence requires {required_bytes} bytes, limit is {max_bytes}"
            ),
            Self::ReductionSegmentLimitExceeded {
                retained,
                max_segments,
            } => write!(
                formatter,
                "Kronecker reduction would retain {retained} segments, limit is {max_segments}"
            ),
            Self::SpecMismatch => formatter.write_str("Kronecker builder shard spec mismatch"),
            Self::SelectionMismatch => {
                formatter.write_str("Kronecker builder sample selection mismatch")
            }
            Self::Curvature(error) => write!(formatter, "Kronecker curvature failed: {error}"),
            Self::Evidence(error) => write!(formatter, "Kronecker evidence failed: {error}"),
            Self::AllocationFailed => formatter.write_str("Kronecker builder allocation failed"),
        }
    }
}

impl std::error::Error for SaltV2KroneckerEvidenceBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Curvature(error) => Some(error),
            Self::Evidence(error) => Some(error),
            Self::Malformed(_)
            | Self::UnexpectedOutputFactors
            | Self::MissingOutputFactors
            | Self::WrongOutputFactorEncoding
            | Self::BatchLengthMismatch { .. }
            | Self::SizeLimitExceeded { .. }
            | Self::ReductionSegmentLimitExceeded { .. }
            | Self::SpecMismatch
            | Self::SelectionMismatch
            | Self::AllocationFailed => None,
        }
    }
}

/// Canonical, source-bound factorized curvature for one additive tensor.
#[derive(Clone, Debug)]
pub struct SaltV2KroneckerEvidence {
    kind: SaltV2Curvature,
    source_id: CurvatureSourceId,
    upstream_evidence_digest: [u8; 32],
    tensor_index: u64,
    tensor_name: String,
    rows: usize,
    columns: usize,
    input_groups: Vec<DensePsdMetric>,
    output_weights: Vec<f64>,
    damping: f64,
    record_digest: [u8; 32],
}

impl SaltV2KroneckerEvidence {
    /// Validate and canonicalize one factorized evidence record.
    ///
    /// Zero values are normalized to positive zero before identity is derived.
    /// `columns` must be G128-aligned, with one input block per column group and
    /// one output scalar per row.
    ///
    /// # Errors
    /// Rejects unsupported curvature kinds, empty or oversized names, malformed
    /// geometry, missing evidence identity, non-finite/negative factors, zero
    /// effective row metrics, invalid PSD input blocks, or allocation overflow.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kind: SaltV2Curvature,
        source_id: CurvatureSourceId,
        upstream_evidence_digest: [u8; 32],
        tensor_index: u64,
        tensor_name: impl Into<String>,
        rows: usize,
        columns: usize,
        input_groups: Vec<DensePsdMetric>,
        output_weights: Vec<f64>,
        damping: f64,
    ) -> Result<Self, SaltV2KroneckerEvidenceError> {
        if !matches!(
            kind,
            SaltV2Curvature::InputHessian
                | SaltV2Curvature::GuidedFisher
                | SaltV2Curvature::ForwardKlKronecker
        ) {
            return Err(SaltV2KroneckerEvidenceError::Malformed("curvature kind"));
        }
        if upstream_evidence_digest == [0; 32] {
            return Err(SaltV2KroneckerEvidenceError::Malformed(
                "upstream evidence digest",
            ));
        }
        let tensor_name = tensor_name.into();
        if tensor_name.is_empty() || tensor_name.len() > MAX_NAME_BYTES {
            return Err(SaltV2KroneckerEvidenceError::Malformed("tensor name"));
        }
        let expected_groups = columns
            .checked_div(GROUP_SIZE)
            .filter(|_| rows > 0 && columns > 0 && columns.is_multiple_of(GROUP_SIZE));
        if expected_groups != Some(input_groups.len()) || output_weights.len() != rows {
            return Err(SaltV2KroneckerEvidenceError::Malformed("factor geometry"));
        }
        if !damping.is_finite() || damping < 0.0 {
            return Err(SaltV2KroneckerEvidenceError::Malformed("damping"));
        }
        let damping = canonical_zero(damping);

        let mut canonical_groups = Vec::new();
        canonical_groups
            .try_reserve_exact(input_groups.len())
            .map_err(|_| SaltV2KroneckerEvidenceError::AllocationFailed)?;
        for group in input_groups {
            if group.dimension() != GROUP_SIZE {
                return Err(SaltV2KroneckerEvidenceError::Malformed(
                    "input group dimension",
                ));
            }
            let values = group
                .as_slice()
                .iter()
                .map(|value| canonical_zero(*value))
                .collect::<Vec<_>>();
            canonical_groups.push(
                DensePsdMetric::new(GROUP_SIZE, &values)
                    .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("input group metric"))?,
            );
        }

        let mut canonical_outputs = Vec::new();
        canonical_outputs
            .try_reserve_exact(output_weights.len())
            .map_err(|_| SaltV2KroneckerEvidenceError::AllocationFailed)?;
        for output in output_weights {
            if !output.is_finite() || output < 0.0 || (output == 0.0 && damping == 0.0) {
                return Err(SaltV2KroneckerEvidenceError::Malformed("output curvature"));
            }
            canonical_outputs.push(canonical_zero(output));
        }

        let mut record = Self {
            kind,
            source_id,
            upstream_evidence_digest,
            tensor_index,
            tensor_name,
            rows,
            columns,
            input_groups: canonical_groups,
            output_weights: canonical_outputs,
            damping,
            record_digest: [0; 32],
        };
        let payload = record.encode_payload()?;
        record.record_digest = checksum(&payload);
        Ok(record)
    }

    /// Curvature algorithm represented by this record.
    #[must_use]
    pub const fn kind(&self) -> SaltV2Curvature {
        self.kind
    }

    /// Immutable source-model/cache/token-stream identity.
    #[must_use]
    pub const fn source_id(&self) -> CurvatureSourceId {
        self.source_id
    }

    /// Digest of the upstream accumulator or builder evidence.
    #[must_use]
    pub const fn upstream_evidence_digest(&self) -> [u8; 32] {
        self.upstream_evidence_digest
    }

    /// Global architecture-adapter tensor ordinal.
    #[must_use]
    pub const fn tensor_index(&self) -> u64 {
        self.tensor_index
    }

    /// Canonical source tensor name.
    #[must_use]
    pub fn tensor_name(&self) -> &str {
        &self.tensor_name
    }

    /// Matrix output rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Matrix input columns.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Shared input-side G128 PSD blocks.
    #[must_use]
    pub fn input_groups(&self) -> &[DensePsdMetric] {
        &self.input_groups
    }

    /// Output-side Fisher/KL scalars.
    #[must_use]
    pub fn output_weights(&self) -> &[f64] {
        &self.output_weights
    }

    /// Diagonal damping applied after Kronecker scaling.
    #[must_use]
    pub const fn damping(&self) -> f64 {
        self.damping
    }

    /// Digest of the complete canonical record payload.
    #[must_use]
    pub const fn record_digest(&self) -> [u8; 32] {
        self.record_digest
    }

    /// Derive HESTIA's tensor sensitivity proxy from this S2KF record.
    ///
    /// The frozen proxy is `input-Gram trace * output-Fisher mean`. Damping is
    /// excluded: it stabilizes fitting but is not observed curvature. Input-only
    /// Hessian evidence cannot supply the required output-Fisher signal.
    ///
    /// # Errors
    /// Rejects input-only evidence or non-positive/non-finite derived arithmetic.
    pub fn hestia_trace_proxy(&self) -> Result<f64, SaltV2KroneckerEvidenceError> {
        if !matches!(
            self.kind,
            SaltV2Curvature::GuidedFisher | SaltV2Curvature::ForwardKlKronecker
        ) {
            return Err(SaltV2KroneckerEvidenceError::Malformed(
                "HESTIA sensitivity requires output-Fisher evidence",
            ));
        }
        let mut input_trace = 0.0_f64;
        for group in &self.input_groups {
            let dimension = group.dimension();
            for index in 0..dimension {
                input_trace += group.as_slice()[index * dimension + index];
            }
        }
        let output_mean =
            self.output_weights.iter().sum::<f64>() / self.output_weights.len() as f64;
        let proxy = input_trace * output_mean;
        if !input_trace.is_finite()
            || !output_mean.is_finite()
            || !proxy.is_finite()
            || input_trace <= 0.0
            || output_mean <= 0.0
            || proxy <= 0.0
        {
            return Err(SaltV2KroneckerEvidenceError::Malformed(
                "HESTIA sensitivity proxy",
            ));
        }
        Ok(proxy)
    }

    /// Exact canonical identity and encoded length without materializing bytes.
    ///
    /// # Errors
    /// Returns a checked-length failure if the validated geometry cannot be
    /// represented by the canonical record layout.
    pub fn receipt(&self) -> Result<SaltV2KroneckerEvidenceReceipt, SaltV2KroneckerEvidenceError> {
        let bytes = payload_len(
            self.tensor_name.len(),
            self.input_groups.len(),
            self.output_weights.len(),
        )
        .and_then(|bytes| bytes.checked_add(CHECKSUM_BYTES))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(SaltV2KroneckerEvidenceError::Malformed("encoded length"))?;
        Ok(SaltV2KroneckerEvidenceReceipt {
            record_digest: self.record_digest,
            bytes,
        })
    }

    /// Verify exact canonical bytes previously written for this record.
    ///
    /// Verification uses fixed memory: it hashes the expected payload length,
    /// checks both the computed and terminal checksum against this record's
    /// identity, and rejects truncation or trailing bytes without decoding or
    /// re-encoding a second record-sized buffer.
    ///
    /// # Errors
    /// Rejects I/O failure, truncation, trailing bytes, or any payload/checksum
    /// mismatch with this validated record.
    pub fn verify_written(
        &self,
        mut reader: impl Read,
    ) -> Result<SaltV2KroneckerEvidenceReceipt, SaltV2KroneckerEvidenceError> {
        const VERIFY_BUFFER_BYTES: usize = 8 * 1024;
        let receipt = self.receipt()?;
        let payload_bytes = receipt
            .bytes
            .checked_sub(CHECKSUM_BYTES as u64)
            .ok_or(SaltV2KroneckerEvidenceError::Malformed("encoded length"))?;
        let mut payload = (&mut reader).take(payload_bytes);
        let mut buffer = [0_u8; VERIFY_BUFFER_BYTES];
        let mut hasher = blake3::Hasher::new_derive_key(CHECKSUM_CONTEXT);
        let mut consumed = 0_u64;
        loop {
            let count = payload
                .read(&mut buffer)
                .map_err(|error| evidence_io("verify written evidence", error))?;
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
            consumed = consumed
                .checked_add(count as u64)
                .ok_or(SaltV2KroneckerEvidenceError::Malformed("encoded length"))?;
        }
        if consumed != payload_bytes {
            return Err(SaltV2KroneckerEvidenceError::Malformed(
                "truncated written record",
            ));
        }
        let mut terminal = [0_u8; CHECKSUM_BYTES];
        read_exact_evidence(&mut reader, &mut terminal)?;
        let computed = *hasher.finalize().as_bytes();
        if computed != self.record_digest || terminal != self.record_digest {
            return Err(SaltV2KroneckerEvidenceError::Malformed(
                "written record identity",
            ));
        }
        let mut trailing = [0_u8; 1];
        if reader
            .read(&mut trailing)
            .map_err(|error| evidence_io("verify written evidence", error))?
            != 0
        {
            return Err(SaltV2KroneckerEvidenceError::Malformed(
                "trailing written record",
            ));
        }
        Ok(receipt)
    }

    /// Reconstruct the borrowed fit-time curvature artifact.
    #[must_use]
    pub fn artifact(&self) -> CurvatureArtifact<'_> {
        let factors =
            crate::KroneckerCurvature::new(&self.input_groups, &self.output_weights, self.damping);
        match self.kind {
            SaltV2Curvature::InputHessian => CurvatureArtifact::input_hessian_kronecker(
                self.source_id,
                self.record_digest,
                factors,
            ),
            SaltV2Curvature::GuidedFisher => CurvatureArtifact::guided_fisher_kronecker(
                self.source_id,
                self.record_digest,
                factors,
            ),
            SaltV2Curvature::ForwardKlKronecker => CurvatureArtifact::forward_kl_kronecker_factors(
                self.source_id,
                self.record_digest,
                factors,
            ),
            SaltV2Curvature::DiagonalFisher => {
                unreachable!("constructor rejects diagonal Fisher")
            }
        }
    }

    /// Join this evidence to one caller-owned widened source matrix.
    ///
    /// # Errors
    /// Rejects a weight slice whose length differs from the record's exact
    /// matrix geometry.
    pub fn tensor_fit_input<'a>(
        &'a self,
        weights: &'a [f32],
    ) -> Result<SaltV2TensorFitInput<'a>, SaltV2KroneckerEvidenceError> {
        let expected = self
            .rows
            .checked_mul(self.columns)
            .ok_or(SaltV2KroneckerEvidenceError::Malformed("tensor geometry"))?;
        if weights.len() != expected {
            return Err(SaltV2KroneckerEvidenceError::WeightLengthMismatch {
                expected,
                got: weights.len(),
            });
        }
        Ok(SaltV2TensorFitInput {
            name: &self.tensor_name,
            weights,
            rows: self.rows,
            cols: self.columns,
            curvature: self.artifact(),
        })
    }

    /// Encode the exact canonical record and terminal checksum.
    ///
    /// # Errors
    /// Returns a checked length or allocation failure.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, SaltV2KroneckerEvidenceError> {
        let mut bytes = self.encode_payload()?;
        let digest = checksum(&bytes);
        if digest != self.record_digest {
            return Err(SaltV2KroneckerEvidenceError::Malformed("record identity"));
        }
        bytes
            .try_reserve_exact(CHECKSUM_BYTES)
            .map_err(|_| SaltV2KroneckerEvidenceError::AllocationFailed)?;
        bytes.extend_from_slice(&digest);
        Ok(bytes)
    }

    /// Decode and verify one complete canonical record.
    ///
    /// # Errors
    /// Rejects truncation, corruption, trailing/noncanonical bytes, invalid
    /// counts, geometry, factors, provenance, or allocation overflow.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, SaltV2KroneckerEvidenceError> {
        if bytes.len() < MAGIC.len() + 2 + 1 + 1 + CHECKSUM_BYTES {
            return Err(SaltV2KroneckerEvidenceError::Malformed("truncated record"));
        }
        let checksum_offset = bytes.len() - CHECKSUM_BYTES;
        let (payload, recorded_checksum) = bytes.split_at(checksum_offset);
        if checksum(payload).as_slice() != recorded_checksum {
            return Err(SaltV2KroneckerEvidenceError::Malformed("checksum"));
        }
        let mut cursor = Cursor::new(payload);
        if cursor.take(MAGIC.len())? != MAGIC {
            return Err(SaltV2KroneckerEvidenceError::Malformed("magic"));
        }
        if cursor.u16()? != VERSION || cursor.u8()? != 0 {
            return Err(SaltV2KroneckerEvidenceError::Malformed(
                "version or reserved byte",
            ));
        }
        let kind = kind_from_tag(cursor.u8()?)?;
        let tensor_index = cursor.u64()?;
        let rows = usize::try_from(cursor.u64()?)
            .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("rows"))?;
        let columns = usize::try_from(cursor.u64()?)
            .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("columns"))?;
        let name_len = usize::try_from(cursor.u32()?)
            .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("name length"))?;
        let group_count = usize::try_from(cursor.u32()?)
            .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("group count"))?;
        let output_count = usize::try_from(cursor.u64()?)
            .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("output count"))?;
        let damping = f64::from_bits(cursor.u64()?);
        let source_id =
            CurvatureSourceId::new(cursor.digest()?, cursor.digest()?, cursor.digest()?)
                .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("source identity"))?;
        let upstream_evidence_digest = cursor.digest()?;
        if name_len == 0 || name_len > MAX_NAME_BYTES {
            return Err(SaltV2KroneckerEvidenceError::Malformed("name length"));
        }
        let expected_groups = columns
            .checked_div(GROUP_SIZE)
            .filter(|_| rows > 0 && columns > 0 && columns.is_multiple_of(GROUP_SIZE));
        if expected_groups != Some(group_count) || output_count != rows {
            return Err(SaltV2KroneckerEvidenceError::Malformed("factor counts"));
        }
        let expected_payload_len = payload_len(name_len, group_count, output_count)
            .ok_or(SaltV2KroneckerEvidenceError::Malformed("encoded length"))?;
        if expected_payload_len != payload.len() {
            return Err(SaltV2KroneckerEvidenceError::Malformed("encoded length"));
        }
        let name = std::str::from_utf8(cursor.take(name_len)?)
            .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("tensor name utf8"))?;
        let mut tensor_name = String::new();
        tensor_name
            .try_reserve_exact(name_len)
            .map_err(|_| SaltV2KroneckerEvidenceError::AllocationFailed)?;
        tensor_name.push_str(name);

        let mut input_groups = Vec::new();
        input_groups
            .try_reserve_exact(group_count)
            .map_err(|_| SaltV2KroneckerEvidenceError::AllocationFailed)?;
        for _ in 0..group_count {
            let dimension = usize::try_from(cursor.u32()?)
                .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("group dimension"))?;
            if dimension != GROUP_SIZE {
                return Err(SaltV2KroneckerEvidenceError::Malformed("group dimension"));
            }
            let value_count = dimension
                .checked_mul(dimension)
                .ok_or(SaltV2KroneckerEvidenceError::Malformed("group size"))?;
            let mut values = Vec::new();
            values
                .try_reserve_exact(value_count)
                .map_err(|_| SaltV2KroneckerEvidenceError::AllocationFailed)?;
            for _ in 0..value_count {
                values.push(f64::from_bits(cursor.u64()?));
            }
            input_groups.push(
                DensePsdMetric::new(dimension, &values)
                    .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("input group metric"))?,
            );
        }
        let mut output_weights = Vec::new();
        output_weights
            .try_reserve_exact(output_count)
            .map_err(|_| SaltV2KroneckerEvidenceError::AllocationFailed)?;
        for _ in 0..output_count {
            output_weights.push(f64::from_bits(cursor.u64()?));
        }
        if cursor.remaining() != 0 {
            return Err(SaltV2KroneckerEvidenceError::Malformed("trailing bytes"));
        }
        let record = Self::new(
            kind,
            source_id,
            upstream_evidence_digest,
            tensor_index,
            tensor_name,
            rows,
            columns,
            input_groups,
            output_weights,
            damping,
        )?;
        if record.canonical_bytes()? != bytes {
            return Err(SaltV2KroneckerEvidenceError::Malformed(
                "noncanonical record",
            ));
        }
        Ok(record)
    }

    /// Read one record through a hard byte ceiling.
    ///
    /// # Errors
    /// Rejects a zero limit, I/O failure, input exceeding `max_bytes`, or any
    /// canonical decode failure.
    pub fn read_from(
        reader: impl Read,
        max_bytes: u64,
    ) -> Result<Self, SaltV2KroneckerEvidenceError> {
        if max_bytes == 0 {
            return Err(SaltV2KroneckerEvidenceError::SizeLimitExceeded { max_bytes });
        }
        let read_limit = max_bytes
            .checked_add(1)
            .ok_or(SaltV2KroneckerEvidenceError::SizeLimitExceeded { max_bytes })?;
        let mut bytes = Vec::new();
        let reserve = usize::try_from(max_bytes.min(16 * 1024 * 1024))
            .map_err(|_| SaltV2KroneckerEvidenceError::AllocationFailed)?;
        bytes
            .try_reserve(reserve)
            .map_err(|_| SaltV2KroneckerEvidenceError::AllocationFailed)?;
        reader
            .take(read_limit)
            .read_to_end(&mut bytes)
            .map_err(|error| evidence_io("read evidence", error))?;
        if bytes.len() as u64 > max_bytes {
            return Err(SaltV2KroneckerEvidenceError::SizeLimitExceeded { max_bytes });
        }
        Self::from_canonical_bytes(&bytes)
    }

    /// Write one canonical record and return its exact content receipt.
    ///
    /// # Errors
    /// Returns an encoding or output I/O failure.
    pub fn write_to(
        &self,
        mut writer: impl Write,
    ) -> Result<SaltV2KroneckerEvidenceReceipt, SaltV2KroneckerEvidenceError> {
        let bytes = self.canonical_bytes()?;
        writer
            .write_all(&bytes)
            .map_err(|error| evidence_io("write evidence", error))?;
        let receipt = self.receipt()?;
        debug_assert_eq!(receipt.bytes, bytes.len() as u64);
        Ok(receipt)
    }

    fn encode_payload(&self) -> Result<Vec<u8>, SaltV2KroneckerEvidenceError> {
        let name_len = u32::try_from(self.tensor_name.len())
            .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("name length"))?;
        let group_count = u32::try_from(self.input_groups.len())
            .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("group count"))?;
        let output_count = u64::try_from(self.output_weights.len())
            .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("output count"))?;
        let encoded_len = payload_len(
            self.tensor_name.len(),
            self.input_groups.len(),
            self.output_weights.len(),
        )
        .ok_or(SaltV2KroneckerEvidenceError::Malformed("encoded length"))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(encoded_len)
            .map_err(|_| SaltV2KroneckerEvidenceError::AllocationFailed)?;
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.push(0);
        bytes.push(kind_tag(self.kind));
        bytes.extend_from_slice(&self.tensor_index.to_le_bytes());
        bytes.extend_from_slice(
            &u64::try_from(self.rows)
                .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("rows"))?
                .to_le_bytes(),
        );
        bytes.extend_from_slice(
            &u64::try_from(self.columns)
                .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("columns"))?
                .to_le_bytes(),
        );
        bytes.extend_from_slice(&name_len.to_le_bytes());
        bytes.extend_from_slice(&group_count.to_le_bytes());
        bytes.extend_from_slice(&output_count.to_le_bytes());
        bytes.extend_from_slice(&self.damping.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.source_id.source_model_digest());
        bytes.extend_from_slice(&self.source_id.activation_cache_digest());
        bytes.extend_from_slice(&self.source_id.token_stream_digest());
        bytes.extend_from_slice(&self.upstream_evidence_digest);
        bytes.extend_from_slice(self.tensor_name.as_bytes());
        for group in &self.input_groups {
            bytes.extend_from_slice(
                &u32::try_from(group.dimension())
                    .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("group dimension"))?
                    .to_le_bytes(),
            );
            for value in group.as_slice() {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        for output in &self.output_weights {
            bytes.extend_from_slice(&output.to_bits().to_le_bytes());
        }
        debug_assert_eq!(bytes.len(), encoded_len);
        Ok(bytes)
    }
}

/// Exact identity and length of one written evidence record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SaltV2KroneckerEvidenceReceipt {
    record_digest: [u8; 32],
    bytes: u64,
}

impl SaltV2KroneckerEvidenceReceipt {
    /// Canonical record payload digest.
    #[must_use]
    pub const fn record_digest(self) -> [u8; 32] {
        self.record_digest
    }

    /// Exact bytes written, including the terminal checksum.
    #[must_use]
    pub const fn bytes(self) -> u64 {
        self.bytes
    }
}

/// Failure while creating, reopening, or writing factorized curvature evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SaltV2KroneckerEvidenceError {
    /// A stable schema, identity, geometry, or numerical invariant failed.
    Malformed(&'static str),
    /// Caller-owned source weights did not match the record geometry.
    WeightLengthMismatch {
        /// Required number of row-major weights.
        expected: usize,
        /// Supplied number of weights.
        got: usize,
    },
    /// Bounded input exceeded the caller-authorized byte ceiling.
    SizeLimitExceeded {
        /// Maximum admitted bytes.
        max_bytes: u64,
    },
    /// A bounded allocation failed.
    AllocationFailed,
    /// Portable input/output failure.
    Io {
        /// Stable operation label.
        operation: &'static str,
        /// Portable I/O category.
        kind: io::ErrorKind,
    },
}

impl fmt::Display for SaltV2KroneckerEvidenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Malformed(field) => write!(formatter, "malformed Kronecker evidence: {field}"),
            Self::WeightLengthMismatch { expected, got } => write!(
                formatter,
                "Kronecker evidence needs {expected} source weights, received {got}"
            ),
            Self::SizeLimitExceeded { max_bytes } => write!(
                formatter,
                "Kronecker evidence exceeds the {max_bytes}-byte input limit"
            ),
            Self::AllocationFailed => formatter.write_str("Kronecker evidence allocation failed"),
            Self::Io { operation, kind } => {
                write!(formatter, "Kronecker evidence {operation} failed: {kind:?}")
            }
        }
    }
}

impl std::error::Error for SaltV2KroneckerEvidenceError {}

fn kind_tag(kind: SaltV2Curvature) -> u8 {
    match kind {
        SaltV2Curvature::InputHessian => 1,
        SaltV2Curvature::GuidedFisher => 2,
        SaltV2Curvature::ForwardKlKronecker => 3,
        SaltV2Curvature::DiagonalFisher => 0,
    }
}

fn kind_from_tag(tag: u8) -> Result<SaltV2Curvature, SaltV2KroneckerEvidenceError> {
    match tag {
        1 => Ok(SaltV2Curvature::InputHessian),
        2 => Ok(SaltV2Curvature::GuidedFisher),
        3 => Ok(SaltV2Curvature::ForwardKlKronecker),
        _ => Err(SaltV2KroneckerEvidenceError::Malformed("curvature kind")),
    }
}

fn checksum(payload: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key(CHECKSUM_CONTEXT);
    hasher.update(payload);
    *hasher.finalize().as_bytes()
}

fn payload_len(name_len: usize, group_count: usize, output_count: usize) -> Option<usize> {
    group_count
        .checked_mul(GROUP_PAYLOAD_BYTES)
        .and_then(|groups| FIXED_PAYLOAD_BYTES.checked_add(groups))
        .and_then(|length| length.checked_add(name_len))
        .and_then(|length| {
            output_count
                .checked_mul(size_of::<f64>())
                .and_then(|outputs| length.checked_add(outputs))
        })
}

fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

fn read_exact_evidence(
    reader: &mut impl Read,
    mut output: &mut [u8],
) -> Result<(), SaltV2KroneckerEvidenceError> {
    while !output.is_empty() {
        match reader.read(output) {
            Ok(0) => {
                return Err(SaltV2KroneckerEvidenceError::Malformed(
                    "truncated written record",
                ));
            }
            Ok(count) => output = &mut output[count..],
            Err(error) => return Err(evidence_io("verify written evidence", error)),
        }
    }
    Ok(())
}

fn evidence_io(operation: &'static str, error: io::Error) -> SaltV2KroneckerEvidenceError {
    SaltV2KroneckerEvidenceError::Io {
        operation,
        kind: error.kind(),
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], SaltV2KroneckerEvidenceError> {
        let end = self
            .offset
            .checked_add(count)
            .filter(|end| *end <= self.bytes.len())
            .ok_or(SaltV2KroneckerEvidenceError::Malformed("truncated field"))?;
        let value = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, SaltV2KroneckerEvidenceError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, SaltV2KroneckerEvidenceError> {
        let mut bytes = [0; 2];
        bytes.copy_from_slice(self.take(2)?);
        Ok(u16::from_le_bytes(bytes))
    }

    fn u32(&mut self) -> Result<u32, SaltV2KroneckerEvidenceError> {
        let mut bytes = [0; 4];
        bytes.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, SaltV2KroneckerEvidenceError> {
        let mut bytes = [0; 8];
        bytes.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(bytes))
    }

    fn digest(&mut self) -> Result<[u8; 32], SaltV2KroneckerEvidenceError> {
        let mut bytes = [0; 32];
        bytes.copy_from_slice(self.take(32)?);
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ActivationCache, ActivationCacheBuilder, ActivationCacheSpec, ActivationChunk,
        ActivationDType, ActivationDigest, PhysicalRateTarget, SaltV2Config, SaltV2Packing,
        SaltV2TensorMasterFitInput, fit_salt_v2_tensor_master,
    };
    use tritium_format::ModelId;

    fn source_id() -> CurvatureSourceId {
        CurvatureSourceId::new([1; 32], [2; 32], [3; 32]).unwrap()
    }

    fn identity_group() -> DensePsdMetric {
        let values = (0..GROUP_SIZE * GROUP_SIZE)
            .map(|index| {
                if index / GROUP_SIZE == index % GROUP_SIZE {
                    1.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        DensePsdMetric::new(GROUP_SIZE, &values).unwrap()
    }

    fn evidence_for(source_id: CurvatureSourceId) -> SaltV2KroneckerEvidence {
        SaltV2KroneckerEvidence::new(
            SaltV2Curvature::GuidedFisher,
            source_id,
            [4; 32],
            17,
            "model.layers.3.mlp.down_proj.weight",
            2,
            GROUP_SIZE,
            vec![identity_group()],
            vec![0.5, 1.5],
            0.125,
        )
        .unwrap()
    }

    fn evidence() -> SaltV2KroneckerEvidence {
        evidence_for(source_id())
    }

    fn activation_cache() -> ActivationCache {
        let spec = ActivationCacheSpec::new(
            0,
            "x",
            1,
            1,
            ActivationDType::Float32,
            ActivationDigest::from_bytes([3; 32]),
            1,
        )
        .unwrap();
        let mut builder = ActivationCacheBuilder::new(spec.clone());
        builder
            .ingest(ActivationChunk::new(&spec, 0, 1, vec![1.0], vec![true], vec![1]).unwrap())
            .unwrap();
        builder.finalize().unwrap()
    }

    #[test]
    fn canonical_record_round_trips_and_binds_every_factor() {
        let original = evidence();
        let bytes = original.canonical_bytes().unwrap();
        let reopened = SaltV2KroneckerEvidence::from_canonical_bytes(&bytes).unwrap();
        let receipt = original.receipt().unwrap();
        assert_eq!(receipt.record_digest(), original.record_digest());
        assert_eq!(receipt.bytes(), bytes.len() as u64);
        assert_eq!(reopened.canonical_bytes().unwrap(), bytes);
        assert_eq!(reopened.receipt().unwrap(), receipt);
        assert_eq!(reopened.record_digest(), original.record_digest());
        assert_eq!(reopened.tensor_index(), 17);
        assert_eq!(reopened.tensor_name(), original.tensor_name());
        assert_eq!(reopened.artifact().digest(), original.artifact().digest());

        let changed = SaltV2KroneckerEvidence::new(
            original.kind(),
            original.source_id(),
            original.upstream_evidence_digest(),
            original.tensor_index(),
            original.tensor_name(),
            original.rows(),
            original.columns(),
            original.input_groups().to_vec(),
            vec![0.5, 1.75],
            original.damping(),
        )
        .unwrap();
        assert_ne!(changed.record_digest(), original.record_digest());
        assert_ne!(changed.artifact().digest(), original.artifact().digest());
    }

    #[test]
    fn bounded_reader_and_corruption_fail_closed() {
        let record = evidence();
        let bytes = record.canonical_bytes().unwrap();
        assert_eq!(
            record.verify_written(bytes.as_slice()).unwrap(),
            record.receipt().unwrap()
        );
        assert!(matches!(
            SaltV2KroneckerEvidence::read_from(bytes.as_slice(), bytes.len() as u64 - 1),
            Err(SaltV2KroneckerEvidenceError::SizeLimitExceeded { .. })
        ));
        assert!(SaltV2KroneckerEvidence::read_from(bytes.as_slice(), bytes.len() as u64).is_ok());
        for index in [0, bytes.len() / 2, bytes.len() - 1] {
            let mut corrupt = bytes.clone();
            corrupt[index] ^= 1;
            assert!(SaltV2KroneckerEvidence::from_canonical_bytes(&corrupt).is_err());
            assert!(record.verify_written(corrupt.as_slice()).is_err());
        }
        for length in 0..bytes.len().min(256) {
            assert!(SaltV2KroneckerEvidence::from_canonical_bytes(&bytes[..length]).is_err());
            assert!(record.verify_written(&bytes[..length]).is_err());
        }
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(record.verify_written(trailing.as_slice()).is_err());

        let mut forged = bytes;
        forged[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
        forged[40..48].copy_from_slice(&u64::MAX.to_le_bytes());
        let checksum_offset = forged.len() - CHECKSUM_BYTES;
        let digest = checksum(&forged[..checksum_offset]);
        forged[checksum_offset..].copy_from_slice(&digest);
        assert!(matches!(
            SaltV2KroneckerEvidence::from_canonical_bytes(&forged),
            Err(SaltV2KroneckerEvidenceError::Malformed("encoded length"))
        ));
    }

    #[test]
    fn reopened_record_drives_the_same_tensor_master_bytes() {
        let cache = activation_cache();
        let source_id = CurvatureSourceId::new(
            [1; 32],
            cache.digest().into_bytes(),
            cache.spec().source_digest().into_bytes(),
        )
        .unwrap();
        let original = evidence_for(source_id);
        let bytes = original.canonical_bytes().unwrap();
        let reopened = SaltV2KroneckerEvidence::from_canonical_bytes(&bytes).unwrap();
        let weights = (0..2 * GROUP_SIZE)
            .map(|index| (index as f32 - 127.0) / 61.0)
            .collect::<Vec<_>>();
        let mut recipe = SaltV2Config {
            curvature: SaltV2Curvature::GuidedFisher,
            packing: SaltV2Packing::B3,
            rate: PhysicalRateTarget {
                max_matrix_bytes: 100_000,
                max_artifact_bytes: 100_000,
                max_resident_bytes: None,
            },
            ..SaltV2Config::default()
        };
        recipe.coordinate_sweeps = 2;
        recipe.em_restarts = 1;
        let fit = |evidence: &SaltV2KroneckerEvidence, sink: &mut Vec<u8>| {
            fit_salt_v2_tensor_master(
                SaltV2TensorMasterFitInput {
                    tensor: evidence.tensor_fit_input(&weights).unwrap(),
                    activations: &cache,
                    source_model_id: ModelId::from_digest([1; 32]),
                    tensor_index: evidence.tensor_index(),
                    source_tensor_digest: [5; 32],
                },
                &recipe,
                sink,
            )
            .unwrap();
        };
        let mut left = Vec::new();
        let mut right = Vec::new();
        fit(&original, &mut left);
        fit(&reopened, &mut right);
        assert_eq!(left, right);
    }

    #[test]
    fn writer_receipt_matches_canonical_record() {
        let evidence = evidence();
        let mut bytes = Vec::new();
        let receipt = evidence.write_to(&mut bytes).unwrap();
        assert_eq!(receipt.record_digest(), evidence.record_digest());
        assert_eq!(receipt.bytes(), bytes.len() as u64);
        assert_eq!(bytes, evidence.canonical_bytes().unwrap());
    }
}
