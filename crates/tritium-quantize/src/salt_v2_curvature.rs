//! Deterministic streaming curvature evidence for SALT V2 joint fitting.
//!
//! Input activations produce a normalized Gram matrix. Output gradients produce one normalized
//! empirical-Fisher scalar per output row. Evidence is bound to immutable source/cache/token-stream
//! identities and global sample ordinals. API batch boundaries, resume points, and shard processing
//! order are deliberately excluded from identity.

use super::salt_v2::DensePsdMetric;

const INPUT_GRAM_MAGIC: [u8; 4] = *b"SIG2";
const OUTPUT_FISHER_MAGIC: [u8; 4] = *b"SOF2";
const KFAC_MAGIC: [u8; 4] = *b"SKF2";
const CANONICAL_VERSION: u8 = 2;
const INPUT_GRAM_DIGEST_CONTEXT: &str = "tritium salt v2 input gram v1";
const OUTPUT_FISHER_DIGEST_CONTEXT: &str = "tritium salt v2 output fisher v1";
const KFAC_DIGEST_CONTEXT: &str = "tritium salt v2 kfac metric v1";
const SOURCE_ID_CONTEXT: &str = "tritium salt v2 curvature source id v1";
const SELECTION_SAMPLE_CONTEXT: &str = "tritium salt v2 curvature selection sample v1";
const INPUT_SAMPLE_CONTEXT: &str = "tritium salt v2 input curvature sample v1";
const FISHER_SAMPLE_CONTEXT: &str = "tritium salt v2 fisher curvature sample v1";

/// Immutable identities that make a curvature sample stream auditable.
///
/// `source_model_digest` binds the source checkpoint, `activation_cache_digest`
/// binds the exact cached tensor values, masks, boundaries, and shard manifest,
/// and `token_stream_digest` binds canonical ordered calibration/token
/// provenance and tokenizer output. Input activations and output gradients may
/// be paired only when all three identities agree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CurvatureSourceId {
    source_model_digest: [u8; 32],
    activation_cache_digest: [u8; 32],
    token_stream_digest: [u8; 32],
}

impl CurvatureSourceId {
    /// Bind curvature evidence to exact source, cache, and token-stream digests.
    ///
    /// # Errors
    /// Rejects an all-zero component because it represents missing rather than
    /// content-addressed provenance.
    pub fn new(
        source_model_digest: [u8; 32],
        activation_cache_digest: [u8; 32],
        token_stream_digest: [u8; 32],
    ) -> Result<Self, CurvatureError> {
        if source_model_digest == [0; 32] {
            return Err(CurvatureError::MissingSourceModelDigest);
        }
        if activation_cache_digest == [0; 32] {
            return Err(CurvatureError::MissingActivationCacheDigest);
        }
        if token_stream_digest == [0; 32] {
            return Err(CurvatureError::MissingTokenStreamDigest);
        }
        Ok(Self {
            source_model_digest,
            activation_cache_digest,
            token_stream_digest,
        })
    }

    /// Exact source-model digest.
    pub const fn source_model_digest(self) -> [u8; 32] {
        self.source_model_digest
    }

    /// Exact activation-cache manifest/content digest.
    pub const fn activation_cache_digest(self) -> [u8; 32] {
        self.activation_cache_digest
    }

    /// Exact ordered token-stream digest.
    pub const fn token_stream_digest(self) -> [u8; 32] {
        self.token_stream_digest
    }

    /// Domain-separated digest of all three binding identities.
    pub fn digest(self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new_derive_key(SOURCE_ID_CONTEXT);
        hasher.update(&self.source_model_digest);
        hasher.update(&self.activation_cache_digest);
        hasher.update(&self.token_stream_digest);
        *hasher.finalize().as_bytes()
    }
}

/// Failure while accumulating or materializing SALT V2 curvature evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CurvatureError {
    /// The source-model content digest was all zeroes.
    MissingSourceModelDigest,
    /// The activation-cache content digest was all zeroes.
    MissingActivationCacheDigest,
    /// The ordered token-stream content digest was all zeroes.
    MissingTokenStreamDigest,
    /// An input feature or output-row dimension was zero.
    InvalidDimension,
    /// A sample count was zero.
    EmptyBatch,
    /// A dimension multiplication overflowed `usize`.
    DimensionTooLarge,
    /// Row-major values did not match the declared sample count and dimension.
    ShapeMismatch {
        /// Name of the malformed value array.
        field: &'static str,
        /// Required number of scalar values.
        expected: usize,
        /// Supplied number of scalar values.
        got: usize,
    },
    /// Optional token weights did not contain one value per sample.
    TokenWeightLengthMismatch {
        /// Required token-weight count.
        expected: usize,
        /// Supplied token-weight count.
        got: usize,
    },
    /// Optional token mask did not contain one value per sample.
    TokenMaskLengthMismatch {
        /// Required mask length.
        expected: usize,
        /// Supplied mask length.
        got: usize,
    },
    /// A token weight was negative, NaN, or infinite.
    InvalidTokenWeight {
        /// Sample containing the invalid weight.
        sample: usize,
    },
    /// An activation was NaN or infinite.
    NonFiniteActivation {
        /// Sample containing the invalid activation.
        sample: usize,
        /// Feature containing the invalid activation.
        feature: usize,
    },
    /// An output gradient was NaN or infinite.
    NonFiniteGradient {
        /// Sample containing the invalid gradient.
        sample: usize,
        /// Output row containing the invalid gradient.
        output_row: usize,
    },
    /// A weighted sum overflowed finite `f64` arithmetic.
    NonFiniteAccumulation,
    /// The accumulated unmasked token weight was zero.
    ZeroTotalWeight,
    /// A shard dimension did not match its destination accumulator.
    ShardDimensionMismatch {
        /// Dimension required by the destination.
        expected: usize,
        /// Dimension carried by the shard.
        got: usize,
    },
    /// A shard was merged outside the required contiguous zero-based order.
    MergeOrderMismatch {
        /// Shard's canonical first-sample ordinal.
        expected: u64,
        /// Supplied first-sample ordinal.
        got: u64,
    },
    /// Source checkpoint, activation cache, or token-stream identity differed.
    SourceMismatch,
    /// Evidence without explicit source/cache/token-stream identity reached a solve boundary.
    UnboundSource,
    /// A global sample ordinal or sample range overflowed `u64`.
    SampleOrdinalOverflow,
    /// Merged canonical sample ranges overlapped or repeated evidence.
    OverlappingSampleRange,
    /// Canonical sample coverage had a gap.
    NonContiguousSampleRange {
        /// First sample ordinal required after the preceding range.
        expected: u64,
        /// First sample ordinal actually present.
        got: u64,
    },
    /// An internal reduction node was not an aligned power-of-two sample interval.
    NonCanonicalSampleRange {
        /// Inclusive node start ordinal.
        start: u64,
        /// Exclusive node end ordinal.
        end: u64,
    },
    /// Input and output accumulators did not use the same ordered token selection.
    SelectionMismatch,
    /// The requested output row did not exist.
    OutputRowOutOfRange {
        /// Number of available output rows.
        rows: usize,
        /// Requested output row.
        got: usize,
    },
    /// K-FAC damping was negative, NaN, or infinite.
    InvalidDamping,
    /// The scaled and damped matrix failed the dense PSD contract.
    InvalidKfacMetric,
}

impl core::fmt::Display for CurvatureError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingSourceModelDigest => {
                write!(f, "curvature source-model digest must be present")
            }
            Self::MissingActivationCacheDigest => {
                write!(f, "curvature activation-cache digest must be present")
            }
            Self::MissingTokenStreamDigest => {
                write!(f, "curvature token-stream digest must be present")
            }
            Self::InvalidDimension => write!(f, "curvature dimension must be greater than zero"),
            Self::EmptyBatch => write!(f, "curvature batch must contain at least one sample"),
            Self::DimensionTooLarge => {
                write!(f, "curvature dimensions overflow addressable storage")
            }
            Self::ShapeMismatch {
                field,
                expected,
                got,
            } => write!(
                f,
                "{field} shape mismatch: expected {expected} values, got {got}"
            ),
            Self::TokenWeightLengthMismatch { expected, got } => write!(
                f,
                "token-weight length mismatch: expected {expected}, got {got}"
            ),
            Self::TokenMaskLengthMismatch { expected, got } => {
                write!(
                    f,
                    "token-mask length mismatch: expected {expected}, got {got}"
                )
            }
            Self::InvalidTokenWeight { sample } => {
                write!(f, "token weight at sample {sample} is invalid")
            }
            Self::NonFiniteActivation { sample, feature } => write!(
                f,
                "activation at sample {sample}, feature {feature} is not finite"
            ),
            Self::NonFiniteGradient { sample, output_row } => write!(
                f,
                "gradient at sample {sample}, output row {output_row} is not finite"
            ),
            Self::NonFiniteAccumulation => write!(f, "curvature accumulation overflowed"),
            Self::ZeroTotalWeight => write!(f, "curvature evidence has zero total weight"),
            Self::ShardDimensionMismatch { expected, got } => write!(
                f,
                "curvature shard dimension mismatch: expected {expected}, got {got}"
            ),
            Self::MergeOrderMismatch { expected, got } => write!(
                f,
                "curvature shard start mismatch: expected sample {expected}, got {got}"
            ),
            Self::SourceMismatch => write!(
                f,
                "curvature source, cache, or token-stream identity mismatch"
            ),
            Self::UnboundSource => write!(
                f,
                "curvature evidence lacks source, cache, and token-stream identity"
            ),
            Self::SampleOrdinalOverflow => write!(f, "curvature sample ordinal overflowed"),
            Self::OverlappingSampleRange => {
                write!(f, "curvature sample ranges overlap or repeat evidence")
            }
            Self::NonContiguousSampleRange { expected, got } => write!(
                f,
                "curvature sample coverage expected ordinal {expected}, got {got}"
            ),
            Self::NonCanonicalSampleRange { start, end } => write!(
                f,
                "curvature reduction node [{start}, {end}) is not canonically aligned"
            ),
            Self::SelectionMismatch => write!(
                f,
                "input Gram and output Fisher used different ordered token selections"
            ),
            Self::OutputRowOutOfRange { rows, got } => {
                write!(f, "output row {got} is outside 0..{rows}")
            }
            Self::InvalidDamping => write!(f, "K-FAC damping must be finite and non-negative"),
            Self::InvalidKfacMetric => write!(f, "scaled and damped K-FAC matrix is not valid PSD"),
        }
    }
}

impl std::error::Error for CurvatureError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SampleRange {
    start: u64,
    end: u64,
}

impl SampleRange {
    fn overlaps(self, other: Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

#[derive(Clone, Debug)]
struct InputSegment {
    range: SampleRange,
    sums: Vec<f64>,
    total_weight: f64,
    selected_count: u64,
    selection_trace: [u8; 32],
    data_trace: [u8; 32],
}

#[derive(Clone, Debug)]
struct FisherSegment {
    range: SampleRange,
    sums: Vec<f64>,
    total_weight: f64,
    selected_count: u64,
    selection_trace: [u8; 32],
    data_trace: [u8; 32],
}

fn zeroed_f64_values(length: usize) -> Result<Vec<f64>, CurvatureError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| CurvatureError::DimensionTooLarge)?;
    values.resize(length, 0.0);
    Ok(values)
}

fn try_clone_f64_values(values: &[f64]) -> Result<Vec<f64>, CurvatureError> {
    let mut cloned = Vec::new();
    cloned
        .try_reserve_exact(values.len())
        .map_err(|_| CurvatureError::DimensionTooLarge)?;
    cloned.extend_from_slice(values);
    Ok(cloned)
}

fn try_clone_input_segments(
    segments: &[InputSegment],
) -> Result<Vec<InputSegment>, CurvatureError> {
    let mut cloned = Vec::new();
    cloned
        .try_reserve_exact(segments.len())
        .map_err(|_| CurvatureError::DimensionTooLarge)?;
    for segment in segments {
        cloned.push(InputSegment {
            range: segment.range,
            sums: try_clone_f64_values(&segment.sums)?,
            total_weight: segment.total_weight,
            selected_count: segment.selected_count,
            selection_trace: segment.selection_trace,
            data_trace: segment.data_trace,
        });
    }
    Ok(cloned)
}

fn try_clone_fisher_segments(
    segments: &[FisherSegment],
) -> Result<Vec<FisherSegment>, CurvatureError> {
    let mut cloned = Vec::new();
    cloned
        .try_reserve_exact(segments.len())
        .map_err(|_| CurvatureError::DimensionTooLarge)?;
    for segment in segments {
        cloned.push(FisherSegment {
            range: segment.range,
            sums: try_clone_f64_values(&segment.sums)?,
            total_weight: segment.total_weight,
            selected_count: segment.selected_count,
            selection_trace: segment.selection_trace,
            data_trace: segment.data_trace,
        });
    }
    Ok(cloned)
}

fn canonical_sample_range(range: SampleRange) -> bool {
    let length = range.end.saturating_sub(range.start);
    length.is_power_of_two() && range.start.is_multiple_of(length)
}

fn canonical_siblings(left: SampleRange, right: SampleRange) -> bool {
    let left_length = left.end - left.start;
    let right_length = right.end - right.start;
    left_length == right_length
        && left.end == right.start
        && left_length
            .checked_mul(2)
            .is_some_and(|parent_length| left.start.is_multiple_of(parent_length))
}

fn merge_input_segments(
    mut left: InputSegment,
    right: InputSegment,
) -> Result<InputSegment, CurvatureError> {
    debug_assert!(canonical_siblings(left.range, right.range));
    add_slice_finite(&mut left.sums, &right.sums)?;
    left.total_weight = finite_add(left.total_weight, right.total_weight)?;
    left.selected_count = left
        .selected_count
        .checked_add(right.selected_count)
        .ok_or(CurvatureError::DimensionTooLarge)?;
    xor_digest(&mut left.selection_trace, right.selection_trace);
    xor_digest(&mut left.data_trace, right.data_trace);
    left.range.end = right.range.end;
    Ok(left)
}

fn merge_fisher_segments(
    mut left: FisherSegment,
    right: FisherSegment,
) -> Result<FisherSegment, CurvatureError> {
    debug_assert!(canonical_siblings(left.range, right.range));
    add_slice_finite(&mut left.sums, &right.sums)?;
    left.total_weight = finite_add(left.total_weight, right.total_weight)?;
    left.selected_count = left
        .selected_count
        .checked_add(right.selected_count)
        .ok_or(CurvatureError::DimensionTooLarge)?;
    xor_digest(&mut left.selection_trace, right.selection_trace);
    xor_digest(&mut left.data_trace, right.data_trace);
    left.range.end = right.range.end;
    Ok(left)
}

fn push_input_segment(
    stack: &mut Vec<InputSegment>,
    segment: InputSegment,
) -> Result<(), CurvatureError> {
    stack
        .try_reserve(1)
        .map_err(|_| CurvatureError::DimensionTooLarge)?;
    stack.push(segment);
    while stack.len() >= 2 {
        let right_index = stack.len() - 1;
        let left_index = right_index - 1;
        if !canonical_siblings(stack[left_index].range, stack[right_index].range) {
            break;
        }
        let right = stack.pop().expect("length checked");
        let left = stack.pop().expect("length checked");
        stack.push(merge_input_segments(left, right)?);
    }
    Ok(())
}

fn push_fisher_segment(
    stack: &mut Vec<FisherSegment>,
    segment: FisherSegment,
) -> Result<(), CurvatureError> {
    stack
        .try_reserve(1)
        .map_err(|_| CurvatureError::DimensionTooLarge)?;
    stack.push(segment);
    while stack.len() >= 2 {
        let right_index = stack.len() - 1;
        let left_index = right_index - 1;
        if !canonical_siblings(stack[left_index].range, stack[right_index].range) {
            break;
        }
        let right = stack.pop().expect("length checked");
        let left = stack.pop().expect("length checked");
        stack.push(merge_fisher_segments(left, right)?);
    }
    Ok(())
}

fn canonicalize_input_segments(
    mut segments: Vec<InputSegment>,
    sample_start: u64,
) -> Result<Vec<InputSegment>, CurvatureError> {
    segments.sort_unstable_by_key(|segment| segment.range.start);
    let mut canonical = Vec::new();
    canonical
        .try_reserve_exact(segments.len())
        .map_err(|_| CurvatureError::DimensionTooLarge)?;
    let mut expected = sample_start;
    for segment in segments {
        if !canonical_sample_range(segment.range) {
            return Err(CurvatureError::NonCanonicalSampleRange {
                start: segment.range.start,
                end: segment.range.end,
            });
        }
        if segment.range.start < expected {
            return Err(CurvatureError::OverlappingSampleRange);
        }
        if segment.range.start != expected {
            return Err(CurvatureError::NonContiguousSampleRange {
                expected,
                got: segment.range.start,
            });
        }
        expected = segment.range.end;
        push_input_segment(&mut canonical, segment)?;
    }
    Ok(canonical)
}

fn canonicalize_fisher_segments(
    mut segments: Vec<FisherSegment>,
    sample_start: u64,
) -> Result<Vec<FisherSegment>, CurvatureError> {
    segments.sort_unstable_by_key(|segment| segment.range.start);
    let mut canonical = Vec::new();
    canonical
        .try_reserve_exact(segments.len())
        .map_err(|_| CurvatureError::DimensionTooLarge)?;
    let mut expected = sample_start;
    for segment in segments {
        if !canonical_sample_range(segment.range) {
            return Err(CurvatureError::NonCanonicalSampleRange {
                start: segment.range.start,
                end: segment.range.end,
            });
        }
        if segment.range.start < expected {
            return Err(CurvatureError::OverlappingSampleRange);
        }
        if segment.range.start != expected {
            return Err(CurvatureError::NonContiguousSampleRange {
                expected,
                got: segment.range.start,
            });
        }
        expected = segment.range.end;
        push_fisher_segment(&mut canonical, segment)?;
    }
    Ok(canonical)
}

/// Streaming accumulator for `E[x x^T]` over row-major activation samples.
///
/// Batch updates are atomic: malformed input does not change sums or content traces. Each sample
/// is bound to its global ordinal, and merged shard ranges are canonicalized independently of
/// processing order.
#[derive(Clone, Debug)]
pub struct InputGramAccumulator {
    dimension: usize,
    source_id: Option<CurvatureSourceId>,
    sample_start: u64,
    next_sample_ordinal: u64,
    sample_count: u64,
    local_segments: Vec<InputSegment>,
    merged_segments: Vec<InputSegment>,
}

impl InputGramAccumulator {
    /// Start an unbound diagnostic accumulator at sample ordinal zero.
    ///
    /// Unbound evidence can be inspected, but [`build_kfac_metric`] rejects it.
    /// Campaign code should use [`Self::new_bound`].
    ///
    /// # Errors
    /// Rejects zero or unaddressably large dimensions.
    pub fn new(dimension: usize) -> Result<Self, CurvatureError> {
        Self::new_inner(dimension, None, 0)
    }

    /// Start a source-bound accumulator at a canonical global sample ordinal.
    ///
    /// `sample_start` is part of every sample leaf. This makes actual sample
    /// order observable while permitting arbitrary API batch boundaries.
    ///
    /// # Errors
    /// Rejects zero or unaddressably large dimensions.
    pub fn new_bound(
        dimension: usize,
        source_id: CurvatureSourceId,
        sample_start: u64,
    ) -> Result<Self, CurvatureError> {
        Self::new_inner(dimension, Some(source_id), sample_start)
    }

    fn new_inner(
        dimension: usize,
        source_id: Option<CurvatureSourceId>,
        sample_start: u64,
    ) -> Result<Self, CurvatureError> {
        if dimension == 0 {
            return Err(CurvatureError::InvalidDimension);
        }
        dimension
            .checked_mul(dimension)
            .ok_or(CurvatureError::DimensionTooLarge)?;
        Ok(Self {
            dimension,
            source_id,
            sample_start,
            next_sample_ordinal: sample_start,
            sample_count: 0,
            local_segments: Vec::new(),
            merged_segments: Vec::new(),
        })
    }

    pub(crate) fn try_clone_transactional(&self) -> Result<Self, CurvatureError> {
        Ok(Self {
            dimension: self.dimension,
            source_id: self.source_id,
            sample_start: self.sample_start,
            next_sample_ordinal: self.next_sample_ordinal,
            sample_count: self.sample_count,
            local_segments: try_clone_input_segments(&self.local_segments)?,
            merged_segments: try_clone_input_segments(&self.merged_segments)?,
        })
    }

    pub(crate) fn retained_reduction_segments(&self) -> usize {
        self.local_segments.len() + self.merged_segments.len()
    }

    /// Add one row-major activation batch.
    ///
    /// `token_weights`, when absent, means unit weight. `token_mask`, when absent, selects every
    /// sample. Supplied weights and every activation are validated even for masked samples so the
    /// content identity cannot hide malformed values.
    ///
    /// # Errors
    /// Rejects empty or malformed batches, non-finite values, negative weights, and overflow.
    pub fn accumulate_batch(
        &mut self,
        activations: &[f32],
        samples: usize,
        token_weights: Option<&[f64]>,
        token_mask: Option<&[bool]>,
    ) -> Result<(), CurvatureError> {
        validate_batch_shape(
            "activations",
            activations.len(),
            samples,
            self.dimension,
            token_weights,
            token_mask,
        )?;
        validate_activations(activations, samples, self.dimension)?;

        let added_samples =
            u64::try_from(samples).map_err(|_| CurvatureError::SampleOrdinalOverflow)?;
        let next_sample_ordinal = self
            .next_sample_ordinal
            .checked_add(added_samples)
            .ok_or(CurvatureError::SampleOrdinalOverflow)?;
        let added_range = SampleRange {
            start: self.next_sample_ordinal,
            end: next_sample_ordinal,
        };
        if self
            .merged_segments
            .iter()
            .any(|segment| segment.range.overlaps(added_range))
        {
            return Err(CurvatureError::OverlappingSampleRange);
        }

        let entries = self
            .dimension
            .checked_mul(self.dimension)
            .ok_or(CurvatureError::DimensionTooLarge)?;
        let mut next_segments = try_clone_input_segments(&self.local_segments)?;
        for sample in 0..samples {
            let ordinal = self
                .next_sample_ordinal
                .checked_add(
                    u64::try_from(sample).map_err(|_| CurvatureError::SampleOrdinalOverflow)?,
                )
                .ok_or(CurvatureError::SampleOrdinalOverflow)?;
            let weight = sample_weight(token_weights, sample);
            let base = sample * self.dimension;
            let is_selected = selected(token_mask, sample);
            let selection_leaf =
                selection_sample_digest(self.source_id, ordinal, weight, is_selected);
            let data_trace = input_sample_digest(
                self.dimension,
                &activations[base..base + self.dimension],
                selection_leaf,
            );
            let mut sums = zeroed_f64_values(entries)?;
            if is_selected {
                for row in 0..self.dimension {
                    let left = f64::from(activations[base + row]);
                    for col in row..self.dimension {
                        let right = f64::from(activations[base + col]);
                        let contribution = finite_product3(weight, left, right)?;
                        let index = row * self.dimension + col;
                        sums[index] = contribution;
                        if row != col {
                            let symmetric = col * self.dimension + row;
                            sums[symmetric] = contribution;
                        }
                    }
                }
            }
            push_input_segment(
                &mut next_segments,
                InputSegment {
                    range: SampleRange {
                        start: ordinal,
                        end: ordinal + 1,
                    },
                    sums,
                    total_weight: if is_selected { weight } else { 0.0 },
                    selected_count: u64::from(is_selected),
                    selection_trace: selection_leaf,
                    data_trace,
                },
            )?;
        }

        let sample_count = checked_sample_add(self.sample_count, samples)?;
        self.sample_count = sample_count;
        self.local_segments = next_segments;
        self.next_sample_ordinal = next_sample_ordinal;
        Ok(())
    }

    /// Merge a shard identified by its canonical first-sample ordinal.
    ///
    /// Shards may arrive in any processing order. Finalization sorts their
    /// immutable ranges and folds them in canonical sample order.
    ///
    /// # Errors
    /// Rejects source/dimension mismatch, an incorrect start ordinal,
    /// overlapping/gapped shard evidence, or count overflow.
    pub fn merge_shard(&mut self, declared_order: u64, shard: &Self) -> Result<(), CurvatureError> {
        validate_merge(
            self.dimension,
            shard.dimension,
            shard.sample_start,
            declared_order,
        )?;
        if self.source_id != shard.source_id {
            return Err(CurvatureError::SourceMismatch);
        }
        let segments = shard.canonical_segments()?;
        let local_range = (self.sample_count != 0).then_some(SampleRange {
            start: self.sample_start,
            end: self.next_sample_ordinal,
        });
        for segment in &segments {
            if local_range.is_some_and(|range| range.overlaps(segment.range))
                || self
                    .merged_segments
                    .iter()
                    .any(|existing| existing.range.overlaps(segment.range))
            {
                return Err(CurvatureError::OverlappingSampleRange);
            }
        }
        self.merged_segments
            .try_reserve_exact(segments.len())
            .map_err(|_| CurvatureError::DimensionTooLarge)?;
        self.merged_segments.extend(segments);
        Ok(())
    }

    /// Normalize the accumulated Gram matrix by selected sample weight.
    ///
    /// # Errors
    /// Rejects evidence whose total selected weight is zero.
    pub fn finish(&self) -> Result<InputGram, CurvatureError> {
        let segments = self.canonical_segments()?;
        let entries = self
            .dimension
            .checked_mul(self.dimension)
            .ok_or(CurvatureError::DimensionTooLarge)?;
        let mut sums = zeroed_f64_values(entries)?;
        let mut total_weight = 0.0;
        let mut sample_count = 0_u64;
        let mut selected_count = 0_u64;
        let mut selection_trace = [0; 32];
        let mut data_trace = [0; 32];
        for segment in segments {
            add_slice_finite(&mut sums, &segment.sums)?;
            total_weight = finite_add(total_weight, segment.total_weight)?;
            sample_count = sample_count
                .checked_add(segment.range.end - segment.range.start)
                .ok_or(CurvatureError::DimensionTooLarge)?;
            selected_count = selected_count
                .checked_add(segment.selected_count)
                .ok_or(CurvatureError::DimensionTooLarge)?;
            xor_digest(&mut selection_trace, segment.selection_trace);
            xor_digest(&mut data_trace, segment.data_trace);
        }
        if total_weight == 0.0 {
            return Err(CurvatureError::ZeroTotalWeight);
        }
        let values = normalized_values(&sums, total_weight)?;
        Ok(InputGram {
            dimension: self.dimension,
            values,
            total_weight,
            sample_count,
            selected_count,
            source_id: self.source_id,
            selection_trace,
            data_trace,
        })
    }

    fn canonical_segments(&self) -> Result<Vec<InputSegment>, CurvatureError> {
        let mut segments = try_clone_input_segments(&self.merged_segments)?;
        segments
            .try_reserve_exact(self.local_segments.len())
            .map_err(|_| CurvatureError::DimensionTooLarge)?;
        segments.extend(try_clone_input_segments(&self.local_segments)?);
        canonicalize_input_segments(segments, self.sample_start)
    }
}

/// Normalized input Gram evidence compatible with row-major [`DensePsdMetric`] storage.
#[derive(Clone, Debug, PartialEq)]
pub struct InputGram {
    dimension: usize,
    values: Vec<f64>,
    total_weight: f64,
    sample_count: u64,
    selected_count: u64,
    source_id: Option<CurvatureSourceId>,
    selection_trace: [u8; 32],
    data_trace: [u8; 32],
}

impl InputGram {
    /// Number of input features.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Normalized row-major Gram values.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    /// Sum of selected token weights used as the normalization denominator.
    #[must_use]
    pub fn total_weight(&self) -> f64 {
        self.total_weight
    }

    /// Number of observed samples, including masked samples.
    #[must_use]
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// Number of unmasked samples, including selected zero-weight samples.
    #[must_use]
    pub fn selected_count(&self) -> u64 {
        self.selected_count
    }

    /// Immutable source/cache/token-stream binding, when supplied.
    #[must_use]
    pub const fn source_id(&self) -> Option<CurvatureSourceId> {
        self.source_id
    }

    /// Digest of global sample ordinals, canonical weights, and masks.
    #[must_use]
    pub fn selection_digest(&self) -> &[u8; 32] {
        &self.selection_trace
    }

    /// Versioned canonical evidence bytes.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + 1 + 8 * 4 + 1 + 96 + 64 + self.values.len() * 8);
        out.extend_from_slice(&INPUT_GRAM_MAGIC);
        out.push(CANONICAL_VERSION);
        write_usize(&mut out, self.dimension);
        out.extend_from_slice(&self.sample_count.to_le_bytes());
        out.extend_from_slice(&self.selected_count.to_le_bytes());
        out.extend_from_slice(&canonical_f64_bits(self.total_weight).to_le_bytes());
        write_source_id(&mut out, self.source_id);
        out.extend_from_slice(&self.selection_trace);
        out.extend_from_slice(&self.data_trace);
        write_f64_values(&mut out, &self.values);
        out
    }

    /// Domain-separated identity of [`Self::canonical_bytes`].
    #[must_use]
    pub fn digest(&self) -> [u8; 32] {
        domain_hash(INPUT_GRAM_DIGEST_CONTEXT, &self.canonical_bytes())
    }
}

/// Streaming accumulator for normalized empirical-Fisher scalars per output row.
#[derive(Clone, Debug)]
pub struct OutputFisherAccumulator {
    output_rows: usize,
    source_id: Option<CurvatureSourceId>,
    sample_start: u64,
    next_sample_ordinal: u64,
    sample_count: u64,
    local_segments: Vec<FisherSegment>,
    merged_segments: Vec<FisherSegment>,
}

impl OutputFisherAccumulator {
    /// Start an unbound diagnostic accumulator at sample ordinal zero.
    ///
    /// Unbound evidence can be inspected, but [`build_kfac_metric`] rejects it.
    /// Campaign code should use [`Self::new_bound`].
    ///
    /// # Errors
    /// Rejects a zero row count.
    pub fn new(output_rows: usize) -> Result<Self, CurvatureError> {
        Self::new_inner(output_rows, None, 0)
    }

    /// Start a source-bound Fisher accumulator at a global sample ordinal.
    ///
    /// # Errors
    /// Rejects a zero output-row count.
    pub fn new_bound(
        output_rows: usize,
        source_id: CurvatureSourceId,
        sample_start: u64,
    ) -> Result<Self, CurvatureError> {
        Self::new_inner(output_rows, Some(source_id), sample_start)
    }

    fn new_inner(
        output_rows: usize,
        source_id: Option<CurvatureSourceId>,
        sample_start: u64,
    ) -> Result<Self, CurvatureError> {
        if output_rows == 0 {
            return Err(CurvatureError::InvalidDimension);
        }
        Ok(Self {
            output_rows,
            source_id,
            sample_start,
            next_sample_ordinal: sample_start,
            sample_count: 0,
            local_segments: Vec::new(),
            merged_segments: Vec::new(),
        })
    }

    pub(crate) fn try_clone_transactional(&self) -> Result<Self, CurvatureError> {
        Ok(Self {
            output_rows: self.output_rows,
            source_id: self.source_id,
            sample_start: self.sample_start,
            next_sample_ordinal: self.next_sample_ordinal,
            sample_count: self.sample_count,
            local_segments: try_clone_fisher_segments(&self.local_segments)?,
            merged_segments: try_clone_fisher_segments(&self.merged_segments)?,
        })
    }

    pub(crate) fn retained_reduction_segments(&self) -> usize {
        self.local_segments.len() + self.merged_segments.len()
    }

    /// Add one row-major output-gradient batch.
    ///
    /// Each row accumulator receives `weight * gradient²`. Mask and weight behavior matches
    /// [`InputGramAccumulator::accumulate_batch`].
    ///
    /// # Errors
    /// Rejects empty or malformed batches, non-finite values, negative weights, and overflow.
    pub fn accumulate_batch(
        &mut self,
        gradients: &[f32],
        samples: usize,
        token_weights: Option<&[f64]>,
        token_mask: Option<&[bool]>,
    ) -> Result<(), CurvatureError> {
        validate_batch_shape(
            "gradients",
            gradients.len(),
            samples,
            self.output_rows,
            token_weights,
            token_mask,
        )?;
        validate_gradients(gradients, samples, self.output_rows)?;

        let added_samples =
            u64::try_from(samples).map_err(|_| CurvatureError::SampleOrdinalOverflow)?;
        let next_sample_ordinal = self
            .next_sample_ordinal
            .checked_add(added_samples)
            .ok_or(CurvatureError::SampleOrdinalOverflow)?;
        let added_range = SampleRange {
            start: self.next_sample_ordinal,
            end: next_sample_ordinal,
        };
        if self
            .merged_segments
            .iter()
            .any(|segment| segment.range.overlaps(added_range))
        {
            return Err(CurvatureError::OverlappingSampleRange);
        }

        let mut next_segments = try_clone_fisher_segments(&self.local_segments)?;
        for sample in 0..samples {
            let ordinal = self
                .next_sample_ordinal
                .checked_add(
                    u64::try_from(sample).map_err(|_| CurvatureError::SampleOrdinalOverflow)?,
                )
                .ok_or(CurvatureError::SampleOrdinalOverflow)?;
            let weight = sample_weight(token_weights, sample);
            let base = sample * self.output_rows;
            let is_selected = selected(token_mask, sample);
            let selection_leaf =
                selection_sample_digest(self.source_id, ordinal, weight, is_selected);
            let data_trace = fisher_sample_digest(
                self.output_rows,
                &gradients[base..base + self.output_rows],
                selection_leaf,
            );
            let mut sums = zeroed_f64_values(self.output_rows)?;
            if is_selected {
                for output_row in 0..self.output_rows {
                    let gradient = f64::from(gradients[base + output_row]);
                    let contribution = finite_product3(weight, gradient, gradient)?;
                    sums[output_row] = contribution;
                }
            }
            push_fisher_segment(
                &mut next_segments,
                FisherSegment {
                    range: SampleRange {
                        start: ordinal,
                        end: ordinal + 1,
                    },
                    sums,
                    total_weight: if is_selected { weight } else { 0.0 },
                    selected_count: u64::from(is_selected),
                    selection_trace: selection_leaf,
                    data_trace,
                },
            )?;
        }

        let sample_count = checked_sample_add(self.sample_count, samples)?;
        self.sample_count = sample_count;
        self.local_segments = next_segments;
        self.next_sample_ordinal = next_sample_ordinal;
        Ok(())
    }

    /// Merge a Fisher shard identified by its canonical first-sample ordinal.
    ///
    /// Shards may arrive in any processing order. Finalization folds their
    /// immutable ranges in canonical sample order.
    ///
    /// # Errors
    /// Rejects source/row mismatch, an incorrect start ordinal,
    /// overlapping/gapped evidence, or count overflow.
    pub fn merge_shard(&mut self, declared_order: u64, shard: &Self) -> Result<(), CurvatureError> {
        validate_merge(
            self.output_rows,
            shard.output_rows,
            shard.sample_start,
            declared_order,
        )?;
        if self.source_id != shard.source_id {
            return Err(CurvatureError::SourceMismatch);
        }
        let segments = shard.canonical_segments()?;
        let local_range = (self.sample_count != 0).then_some(SampleRange {
            start: self.sample_start,
            end: self.next_sample_ordinal,
        });
        for segment in &segments {
            if local_range.is_some_and(|range| range.overlaps(segment.range))
                || self
                    .merged_segments
                    .iter()
                    .any(|existing| existing.range.overlaps(segment.range))
            {
                return Err(CurvatureError::OverlappingSampleRange);
            }
        }
        self.merged_segments
            .try_reserve_exact(segments.len())
            .map_err(|_| CurvatureError::DimensionTooLarge)?;
        self.merged_segments.extend(segments);
        Ok(())
    }

    /// Normalize every output-row Fisher scalar by selected sample weight.
    ///
    /// # Errors
    /// Rejects evidence whose total selected weight is zero.
    pub fn finish(&self) -> Result<OutputFisher, CurvatureError> {
        let segments = self.canonical_segments()?;
        let mut sums = zeroed_f64_values(self.output_rows)?;
        let mut total_weight = 0.0;
        let mut sample_count = 0_u64;
        let mut selected_count = 0_u64;
        let mut selection_trace = [0; 32];
        let mut data_trace = [0; 32];
        for segment in segments {
            add_slice_finite(&mut sums, &segment.sums)?;
            total_weight = finite_add(total_weight, segment.total_weight)?;
            sample_count = sample_count
                .checked_add(segment.range.end - segment.range.start)
                .ok_or(CurvatureError::DimensionTooLarge)?;
            selected_count = selected_count
                .checked_add(segment.selected_count)
                .ok_or(CurvatureError::DimensionTooLarge)?;
            xor_digest(&mut selection_trace, segment.selection_trace);
            xor_digest(&mut data_trace, segment.data_trace);
        }
        if total_weight == 0.0 {
            return Err(CurvatureError::ZeroTotalWeight);
        }
        let values = normalized_values(&sums, total_weight)?;
        Ok(OutputFisher {
            output_rows: self.output_rows,
            values,
            total_weight,
            sample_count,
            selected_count,
            source_id: self.source_id,
            selection_trace,
            data_trace,
        })
    }

    fn canonical_segments(&self) -> Result<Vec<FisherSegment>, CurvatureError> {
        let mut segments = try_clone_fisher_segments(&self.merged_segments)?;
        segments
            .try_reserve_exact(self.local_segments.len())
            .map_err(|_| CurvatureError::DimensionTooLarge)?;
        segments.extend(try_clone_fisher_segments(&self.local_segments)?);
        canonicalize_fisher_segments(segments, self.sample_start)
    }
}

/// Normalized output-gradient empirical-Fisher scalars.
#[derive(Clone, Debug, PartialEq)]
pub struct OutputFisher {
    output_rows: usize,
    values: Vec<f64>,
    total_weight: f64,
    sample_count: u64,
    selected_count: u64,
    source_id: Option<CurvatureSourceId>,
    selection_trace: [u8; 32],
    data_trace: [u8; 32],
}

impl OutputFisher {
    /// Number of output rows.
    #[must_use]
    pub fn output_rows(&self) -> usize {
        self.output_rows
    }

    /// Normalized Fisher scalar for each output row.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    /// Sum of selected token weights used as the normalization denominator.
    #[must_use]
    pub fn total_weight(&self) -> f64 {
        self.total_weight
    }

    /// Number of observed samples, including masked samples.
    #[must_use]
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// Number of unmasked samples, including selected zero-weight samples.
    #[must_use]
    pub fn selected_count(&self) -> u64 {
        self.selected_count
    }

    /// Immutable source/cache/token-stream binding, when supplied.
    #[must_use]
    pub const fn source_id(&self) -> Option<CurvatureSourceId> {
        self.source_id
    }

    /// Digest of global sample ordinals, canonical weights, and masks.
    #[must_use]
    pub fn selection_digest(&self) -> &[u8; 32] {
        &self.selection_trace
    }

    /// Versioned canonical evidence bytes.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + 1 + 8 * 4 + 1 + 96 + 64 + self.values.len() * 8);
        out.extend_from_slice(&OUTPUT_FISHER_MAGIC);
        out.push(CANONICAL_VERSION);
        write_usize(&mut out, self.output_rows);
        out.extend_from_slice(&self.sample_count.to_le_bytes());
        out.extend_from_slice(&self.selected_count.to_le_bytes());
        out.extend_from_slice(&canonical_f64_bits(self.total_weight).to_le_bytes());
        write_source_id(&mut out, self.source_id);
        out.extend_from_slice(&self.selection_trace);
        out.extend_from_slice(&self.data_trace);
        write_f64_values(&mut out, &self.values);
        out
    }

    /// Domain-separated identity of [`Self::canonical_bytes`].
    #[must_use]
    pub fn digest(&self) -> [u8; 32] {
        domain_hash(OUTPUT_FISHER_DIGEST_CONTEXT, &self.canonical_bytes())
    }
}

/// One output-row K-FAC block and its receipt-binding metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct KfacMetric {
    metric: DensePsdMetric,
    source_id: CurvatureSourceId,
    output_row: usize,
    output_scalar: f64,
    damping: f64,
    total_weight: f64,
    input_digest: [u8; 32],
    fisher_digest: [u8; 32],
}

impl KfacMetric {
    /// Dense PSD matrix consumed by the SALT V2 joint fitter.
    #[must_use]
    pub fn metric(&self) -> &DensePsdMetric {
        &self.metric
    }

    /// Immutable source-model, activation-cache, and token-stream provenance.
    #[must_use]
    pub const fn source_id(&self) -> CurvatureSourceId {
        self.source_id
    }

    /// Output row whose Fisher scalar scaled the input Gram.
    #[must_use]
    pub fn output_row(&self) -> usize {
        self.output_row
    }

    /// Normalized empirical-Fisher scalar used for this output row.
    #[must_use]
    pub fn output_scalar(&self) -> f64 {
        self.output_scalar
    }

    /// Diagonal damping added after Kronecker scaling.
    #[must_use]
    pub fn damping(&self) -> f64 {
        self.damping
    }

    /// Versioned canonical bytes binding evidence identities, dimensions, weights, values, and
    /// damping.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let values = self.metric.as_slice();
        let mut out = Vec::with_capacity(4 + 1 + 8 * 5 + 64 + values.len() * 8);
        out.extend_from_slice(&KFAC_MAGIC);
        out.push(CANONICAL_VERSION);
        write_usize(&mut out, self.metric.dimension());
        write_usize(&mut out, self.output_row);
        out.extend_from_slice(&canonical_f64_bits(self.output_scalar).to_le_bytes());
        out.extend_from_slice(&canonical_f64_bits(self.damping).to_le_bytes());
        out.extend_from_slice(&canonical_f64_bits(self.total_weight).to_le_bytes());
        out.extend_from_slice(&self.input_digest);
        out.extend_from_slice(&self.fisher_digest);
        write_f64_values(&mut out, values);
        out
    }

    /// Domain-separated identity of [`Self::canonical_bytes`].
    #[must_use]
    pub fn digest(&self) -> [u8; 32] {
        domain_hash(KFAC_DIGEST_CONTEXT, &self.canonical_bytes())
    }
}

/// Form `output_fisher[row] * input_gram + damping * I` as a validated dense PSD metric.
///
/// Input and output evidence must carry identical source/cache/token-stream identities and
/// canonical selection traces. This prevents an input Gram from being silently paired with
/// gradients from different corpora, samples, weights, or masks while excluding operational batch
/// and shard processing boundaries from identity.
///
/// # Errors
/// Rejects misaligned evidence, an out-of-range row, invalid damping, non-finite products, or a
/// matrix that fails the dense PSD contract.
pub fn build_kfac_metric(
    input: &InputGram,
    fisher: &OutputFisher,
    output_row: usize,
    damping: f64,
) -> Result<KfacMetric, CurvatureError> {
    let (Some(input_source), Some(fisher_source)) = (input.source_id, fisher.source_id) else {
        return Err(CurvatureError::UnboundSource);
    };
    if input_source != fisher_source {
        return Err(CurvatureError::SourceMismatch);
    }
    if input.selection_trace != fisher.selection_trace
        || input.sample_count != fisher.sample_count
        || input.selected_count != fisher.selected_count
        || input.total_weight.to_bits() != fisher.total_weight.to_bits()
    {
        return Err(CurvatureError::SelectionMismatch);
    }
    let Some(&output_scalar) = fisher.values.get(output_row) else {
        return Err(CurvatureError::OutputRowOutOfRange {
            rows: fisher.output_rows,
            got: output_row,
        });
    };
    if !damping.is_finite() || damping < 0.0 {
        return Err(CurvatureError::InvalidDamping);
    }

    let mut values = Vec::new();
    values
        .try_reserve_exact(input.values.len())
        .map_err(|_| CurvatureError::DimensionTooLarge)?;
    for row in 0..input.dimension {
        for col in 0..input.dimension {
            let mut value = input.values[row * input.dimension + col] * output_scalar;
            if row == col {
                value += damping;
            }
            if !value.is_finite() {
                return Err(CurvatureError::NonFiniteAccumulation);
            }
            values.push(canonicalize_zero(value));
        }
    }
    let metric = DensePsdMetric::new(input.dimension, &values)
        .map_err(|_| CurvatureError::InvalidKfacMetric)?;
    Ok(KfacMetric {
        metric,
        source_id: input_source,
        output_row,
        output_scalar,
        damping: canonicalize_zero(damping),
        total_weight: input.total_weight,
        input_digest: input.digest(),
        fisher_digest: fisher.digest(),
    })
}

fn validate_batch_shape(
    field: &'static str,
    value_len: usize,
    samples: usize,
    dimension: usize,
    token_weights: Option<&[f64]>,
    token_mask: Option<&[bool]>,
) -> Result<(), CurvatureError> {
    if samples == 0 {
        return Err(CurvatureError::EmptyBatch);
    }
    let expected = samples
        .checked_mul(dimension)
        .ok_or(CurvatureError::DimensionTooLarge)?;
    if value_len != expected {
        return Err(CurvatureError::ShapeMismatch {
            field,
            expected,
            got: value_len,
        });
    }
    if let Some(weights) = token_weights {
        if weights.len() != samples {
            return Err(CurvatureError::TokenWeightLengthMismatch {
                expected: samples,
                got: weights.len(),
            });
        }
        if let Some(sample) = weights
            .iter()
            .position(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(CurvatureError::InvalidTokenWeight { sample });
        }
    }
    if let Some(mask) = token_mask
        && mask.len() != samples
    {
        return Err(CurvatureError::TokenMaskLengthMismatch {
            expected: samples,
            got: mask.len(),
        });
    }
    Ok(())
}

fn validate_activations(
    activations: &[f32],
    samples: usize,
    dimension: usize,
) -> Result<(), CurvatureError> {
    if let Some(index) = activations.iter().position(|value| !value.is_finite()) {
        return Err(CurvatureError::NonFiniteActivation {
            sample: index / dimension,
            feature: index % dimension,
        });
    }
    debug_assert_eq!(activations.len(), samples * dimension);
    Ok(())
}

fn validate_gradients(
    gradients: &[f32],
    samples: usize,
    output_rows: usize,
) -> Result<(), CurvatureError> {
    if let Some(index) = gradients.iter().position(|value| !value.is_finite()) {
        return Err(CurvatureError::NonFiniteGradient {
            sample: index / output_rows,
            output_row: index % output_rows,
        });
    }
    debug_assert_eq!(gradients.len(), samples * output_rows);
    Ok(())
}

fn validate_merge(
    expected_dimension: usize,
    shard_dimension: usize,
    expected_start: u64,
    declared_order: u64,
) -> Result<(), CurvatureError> {
    if expected_dimension != shard_dimension {
        return Err(CurvatureError::ShardDimensionMismatch {
            expected: expected_dimension,
            got: shard_dimension,
        });
    }
    if declared_order != expected_start {
        return Err(CurvatureError::MergeOrderMismatch {
            expected: expected_start,
            got: declared_order,
        });
    }
    Ok(())
}

fn selected(mask: Option<&[bool]>, sample: usize) -> bool {
    mask.is_none_or(|values| values[sample])
}

fn sample_weight(weights: Option<&[f64]>, sample: usize) -> f64 {
    weights.map_or(1.0, |values| values[sample])
}

fn finite_add(left: f64, right: f64) -> Result<f64, CurvatureError> {
    let value = left + right;
    value
        .is_finite()
        .then_some(value)
        .ok_or(CurvatureError::NonFiniteAccumulation)
}

fn finite_product3(a: f64, b: f64, c: f64) -> Result<f64, CurvatureError> {
    let value = a * b * c;
    value
        .is_finite()
        .then_some(value)
        .ok_or(CurvatureError::NonFiniteAccumulation)
}

fn add_slice_finite(destination: &mut [f64], source: &[f64]) -> Result<(), CurvatureError> {
    debug_assert_eq!(destination.len(), source.len());
    for (destination, source) in destination.iter_mut().zip(source) {
        *destination = finite_add(*destination, *source)?;
    }
    Ok(())
}

fn checked_sample_add(current: u64, added: usize) -> Result<u64, CurvatureError> {
    let added = u64::try_from(added).map_err(|_| CurvatureError::DimensionTooLarge)?;
    current
        .checked_add(added)
        .ok_or(CurvatureError::DimensionTooLarge)
}

fn normalized_values(sums: &[f64], total_weight: f64) -> Result<Vec<f64>, CurvatureError> {
    let mut values = zeroed_f64_values(sums.len())?;
    for (value, sum) in values.iter_mut().zip(sums) {
        let normalized = *sum / total_weight;
        *value = normalized
            .is_finite()
            .then_some(canonicalize_zero(normalized))
            .ok_or(CurvatureError::NonFiniteAccumulation)?;
    }
    Ok(values)
}

fn selection_sample_digest(
    source_id: Option<CurvatureSourceId>,
    ordinal: u64,
    weight: f64,
    selected: bool,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key(SELECTION_SAMPLE_CONTEXT);
    match source_id {
        Some(source_id) => {
            hasher.update(&[1]);
            hasher.update(&source_id.digest());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    hasher.update(&ordinal.to_le_bytes());
    hasher.update(&canonical_f64_bits(weight).to_le_bytes());
    hasher.update(&[u8::from(selected)]);
    *hasher.finalize().as_bytes()
}

fn input_sample_digest(
    dimension: usize,
    activations: &[f32],
    selection_sample: [u8; 32],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key(INPUT_SAMPLE_CONTEXT);
    hasher.update(&u64::try_from(dimension).unwrap_or(u64::MAX).to_le_bytes());
    hasher.update(&selection_sample);
    for &activation in activations {
        hasher.update(&canonical_f32_bits(activation).to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

fn fisher_sample_digest(
    output_rows: usize,
    gradients: &[f32],
    selection_sample: [u8; 32],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key(FISHER_SAMPLE_CONTEXT);
    hasher.update(&u64::try_from(output_rows).unwrap_or(u64::MAX).to_le_bytes());
    hasher.update(&selection_sample);
    for &gradient in gradients {
        hasher.update(&canonical_f32_bits(gradient).to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

fn xor_digest(trace: &mut [u8; 32], leaf: [u8; 32]) {
    for (trace, leaf) in trace.iter_mut().zip(leaf) {
        *trace ^= leaf;
    }
}

fn domain_hash(context: &'static str, bytes: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key(context);
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}

fn write_usize(out: &mut Vec<u8>, value: usize) {
    out.extend_from_slice(&u64::try_from(value).unwrap_or(u64::MAX).to_le_bytes());
}

fn write_source_id(out: &mut Vec<u8>, source_id: Option<CurvatureSourceId>) {
    let Some(source_id) = source_id else {
        out.push(0);
        return;
    };
    out.push(1);
    out.extend_from_slice(&source_id.source_model_digest);
    out.extend_from_slice(&source_id.activation_cache_digest);
    out.extend_from_slice(&source_id.token_stream_digest);
}

fn write_f64_values(out: &mut Vec<u8>, values: &[f64]) {
    out.extend_from_slice(
        &u64::try_from(values.len())
            .unwrap_or(u64::MAX)
            .to_le_bytes(),
    );
    for &value in values {
        out.extend_from_slice(&canonical_f64_bits(value).to_le_bytes());
    }
}

fn canonicalize_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

fn canonical_f64_bits(value: f64) -> u64 {
    canonicalize_zero(value).to_bits()
}

fn canonical_f32_bits(value: f32) -> u32 {
    if value == 0.0 {
        0.0_f32.to_bits()
    } else {
        value.to_bits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(seed: u8) -> CurvatureSourceId {
        CurvatureSourceId::new(
            [seed; 32],
            [seed.wrapping_add(1); 32],
            [seed.wrapping_add(2); 32],
        )
        .expect("nonzero curvature source identity")
    }

    #[test]
    fn source_identity_rejects_each_missing_digest_component() {
        assert_eq!(
            CurvatureSourceId::new([0; 32], [1; 32], [2; 32]),
            Err(CurvatureError::MissingSourceModelDigest)
        );
        assert_eq!(
            CurvatureSourceId::new([1; 32], [0; 32], [2; 32]),
            Err(CurvatureError::MissingActivationCacheDigest)
        );
        assert_eq!(
            CurvatureSourceId::new([1; 32], [2; 32], [0; 32]),
            Err(CurvatureError::MissingTokenStreamDigest)
        );
    }

    #[test]
    fn adversarial_curvature_is_bitwise_chunk_resume_and_reverse_merge_invariant() {
        let binding = source(91);
        let mut values = vec![1.0_f32; 130];
        values[0] = 2.0_f32.powi(30);

        let mut one_input = InputGramAccumulator::new_bound(1, binding, 0).expect("one input");
        one_input
            .accumulate_batch(&values, values.len(), None, None)
            .expect("one input batch");
        let one_input = one_input.finish().expect("one input evidence");

        let mut chunked_input =
            InputGramAccumulator::new_bound(1, binding, 0).expect("chunked input");
        for range in [0..1, 1..130] {
            chunked_input
                .accumulate_batch(&values[range.clone()], range.len(), None, None)
                .expect("arbitrary input chunk");
        }
        let chunked_input = chunked_input.finish().expect("chunked input evidence");

        let input_ranges = [0..1, 1..130];
        let mut input_shards = Vec::new();
        for range in &input_ranges {
            let mut shard = InputGramAccumulator::new_bound(1, binding, range.start as u64)
                .expect("input shard");
            shard
                .accumulate_batch(&values[range.clone()], range.len(), None, None)
                .expect("input shard batch");
            input_shards.push(shard);
        }
        let mut merged_input =
            InputGramAccumulator::new_bound(1, binding, 0).expect("merged input");
        for (range, shard) in input_ranges.iter().zip(&input_shards).rev() {
            merged_input
                .merge_shard(range.start as u64, shard)
                .expect("reverse input merge");
        }
        let merged_input = merged_input.finish().expect("merged input evidence");

        assert_eq!(chunked_input, one_input);
        assert_eq!(merged_input, one_input);
        assert_eq!(chunked_input.digest(), one_input.digest());
        assert_eq!(merged_input.digest(), one_input.digest());

        let mut one_fisher = OutputFisherAccumulator::new_bound(1, binding, 0).expect("one Fisher");
        one_fisher
            .accumulate_batch(&values, values.len(), None, None)
            .expect("one Fisher batch");
        let one_fisher = one_fisher.finish().expect("one Fisher evidence");

        let mut chunked_fisher =
            OutputFisherAccumulator::new_bound(1, binding, 0).expect("chunked Fisher");
        for range in [0..1, 1..130] {
            chunked_fisher
                .accumulate_batch(&values[range.clone()], range.len(), None, None)
                .expect("arbitrary Fisher chunk");
        }
        let chunked_fisher = chunked_fisher.finish().expect("chunked Fisher evidence");

        let mut fisher_shards = Vec::new();
        for range in &input_ranges {
            let mut shard = OutputFisherAccumulator::new_bound(1, binding, range.start as u64)
                .expect("Fisher shard");
            shard
                .accumulate_batch(&values[range.clone()], range.len(), None, None)
                .expect("Fisher shard batch");
            fisher_shards.push(shard);
        }
        let mut merged_fisher =
            OutputFisherAccumulator::new_bound(1, binding, 0).expect("merged Fisher");
        for (range, shard) in input_ranges.iter().zip(&fisher_shards).rev() {
            merged_fisher
                .merge_shard(range.start as u64, shard)
                .expect("reverse Fisher merge");
        }
        let merged_fisher = merged_fisher.finish().expect("merged Fisher evidence");

        assert_eq!(chunked_fisher, one_fisher);
        assert_eq!(merged_fisher, one_fisher);
        assert_eq!(chunked_fisher.digest(), one_fisher.digest());
        assert_eq!(merged_fisher.digest(), one_fisher.digest());
        assert_eq!(
            build_kfac_metric(&merged_input, &merged_fisher, 0, 1e-4).expect("merged K-FAC"),
            build_kfac_metric(&one_input, &one_fisher, 0, 1e-4).expect("one-shot K-FAC")
        );
    }

    #[test]
    fn fully_masked_shard_is_mergeable_when_global_evidence_has_weight() {
        let binding = source(101);
        let mut masked = InputGramAccumulator::new_bound(1, binding, 0).expect("masked shard");
        masked
            .accumulate_batch(&[1.0e20], 1, None, Some(&[false]))
            .expect("masked data remains digest-bound");
        let mut selected = InputGramAccumulator::new_bound(1, binding, 1).expect("selected shard");
        selected
            .accumulate_batch(&[2.0], 1, None, Some(&[true]))
            .expect("selected data");

        let mut merged = InputGramAccumulator::new_bound(1, binding, 0).expect("merged input");
        merged.merge_shard(1, &selected).expect("selected merge");
        merged
            .merge_shard(0, &masked)
            .expect("zero-weight shard merge");
        let evidence = merged.finish().expect("global positive evidence");

        assert_eq!(evidence.as_slice(), &[4.0]);
        assert_eq!(evidence.sample_count(), 2);
        assert_eq!(evidence.selected_count(), 1);
    }

    fn analytic_evidence() -> (InputGram, OutputFisher) {
        let weights = [1.0, 3.0];
        let binding = source(1);
        let mut input = InputGramAccumulator::new_bound(2, binding, 0).expect("dimension");
        input
            .accumulate_batch(&[1.0, 2.0, 3.0, 4.0], 2, Some(&weights), None)
            .expect("analytic input batch");
        let mut output = OutputFisherAccumulator::new_bound(2, binding, 0).expect("rows");
        output
            .accumulate_batch(&[1.0, 2.0, 3.0, 4.0], 2, Some(&weights), None)
            .expect("analytic gradient batch");
        (
            input.finish().expect("positive input weight"),
            output.finish().expect("positive output weight"),
        )
    }

    #[test]
    fn analytic_gram_fisher_and_kfac_are_exact() {
        let (input, output) = analytic_evidence();
        assert_eq!(input.as_slice(), &[7.0, 9.5, 9.5, 13.0]);
        assert_eq!(output.as_slice(), &[7.0, 13.0]);
        assert_eq!(input.total_weight(), 4.0);
        assert_eq!(output.total_weight(), 4.0);

        let kfac = build_kfac_metric(&input, &output, 0, 0.25).expect("damped PSD");
        assert_eq!(kfac.output_scalar(), 7.0);
        assert_eq!(kfac.metric().as_slice(), &[49.25, 66.5, 66.5, 91.25]);
    }

    #[test]
    fn ordered_shard_merge_matches_one_shot() {
        let all = [1.0_f32, 2.0, 3.0, 4.0, -1.0, 5.0];
        let weights = [1.0, 2.0, 3.0];
        let mut one_shot = InputGramAccumulator::new(2).expect("dimension");
        one_shot
            .accumulate_batch(&all, 3, Some(&weights), None)
            .expect("one-shot batch");

        let binding = source(2);
        let mut shard_zero = InputGramAccumulator::new_bound(2, binding, 0).expect("dimension");
        shard_zero
            .accumulate_batch(&all[..4], 2, Some(&weights[..2]), None)
            .expect("first shard");
        let mut shard_one = InputGramAccumulator::new_bound(2, binding, 2).expect("dimension");
        shard_one
            .accumulate_batch(&all[4..], 1, Some(&weights[2..]), None)
            .expect("second shard");
        let mut merged = InputGramAccumulator::new_bound(2, binding, 0).expect("dimension");
        merged
            .merge_shard(0, &shard_zero)
            .expect("ordered shard zero");
        merged
            .merge_shard(2, &shard_one)
            .expect("ordered shard one");

        let expected = one_shot.finish().expect("one-shot evidence");
        let actual = merged.finish().expect("merged evidence");
        for (expected, actual) in expected.as_slice().iter().zip(actual.as_slice()) {
            assert!((expected - actual).abs() <= 1e-15);
        }
        assert_eq!(actual.total_weight(), expected.total_weight());
        assert_eq!(actual.sample_count(), expected.sample_count());

        let mut fisher_one_shot = OutputFisherAccumulator::new(2).expect("rows");
        fisher_one_shot
            .accumulate_batch(&all, 3, Some(&weights), None)
            .expect("one-shot Fisher");
        let mut fisher_zero = OutputFisherAccumulator::new_bound(2, binding, 0).expect("rows");
        fisher_zero
            .accumulate_batch(&all[..4], 2, Some(&weights[..2]), None)
            .expect("first Fisher shard");
        let mut fisher_one = OutputFisherAccumulator::new_bound(2, binding, 2).expect("rows");
        fisher_one
            .accumulate_batch(&all[4..], 1, Some(&weights[2..]), None)
            .expect("second Fisher shard");
        let mut fisher_merged = OutputFisherAccumulator::new_bound(2, binding, 0).expect("rows");
        fisher_merged
            .merge_shard(0, &fisher_zero)
            .expect("ordered Fisher shard zero");
        fisher_merged
            .merge_shard(2, &fisher_one)
            .expect("ordered Fisher shard one");
        let expected = fisher_one_shot.finish().expect("one-shot Fisher evidence");
        let actual = fisher_merged.finish().expect("merged Fisher evidence");
        for (expected, actual) in expected.as_slice().iter().zip(actual.as_slice()) {
            assert!((expected - actual).abs() <= 1e-15);
        }
    }

    #[test]
    fn canonical_sample_stream_is_chunk_resume_and_merge_invariant() {
        let binding = source(11);
        let activations = [1.0_f32, 2.0, 3.0, 4.0, -1.0, 5.0];
        let gradients = [2.0_f32, 1.0, 4.0];
        let weights = [1.0, 2.0, 3.0];

        let mut one_input = InputGramAccumulator::new_bound(2, binding, 0).expect("one-shot input");
        one_input
            .accumulate_batch(&activations, 3, Some(&weights), None)
            .expect("one-shot input batch");
        let mut chunked_input =
            InputGramAccumulator::new_bound(2, binding, 0).expect("chunked input");
        chunked_input
            .accumulate_batch(&activations[..4], 2, Some(&weights[..2]), None)
            .expect("first input chunk");
        let mut resumed_input = chunked_input.clone();
        resumed_input
            .accumulate_batch(&activations[4..], 1, Some(&weights[2..]), None)
            .expect("resumed input chunk");

        let mut input_zero =
            InputGramAccumulator::new_bound(2, binding, 0).expect("input shard zero");
        input_zero
            .accumulate_batch(&activations[..4], 2, Some(&weights[..2]), None)
            .expect("input shard zero data");
        let mut input_one =
            InputGramAccumulator::new_bound(2, binding, 2).expect("input shard one");
        input_one
            .accumulate_batch(&activations[4..], 1, Some(&weights[2..]), None)
            .expect("input shard one data");
        let mut merged_input =
            InputGramAccumulator::new_bound(2, binding, 0).expect("merged input");
        merged_input
            .merge_shard(2, &input_one)
            .expect("reverse input shard one");
        merged_input
            .merge_shard(0, &input_zero)
            .expect("reverse input shard zero");

        let expected_input = one_input.finish().expect("one-shot input evidence");
        let resumed_input = resumed_input.finish().expect("resumed input evidence");
        let merged_input = merged_input.finish().expect("merged input evidence");
        assert_eq!(resumed_input.as_slice(), expected_input.as_slice());
        assert_eq!(merged_input.as_slice(), expected_input.as_slice());
        assert_eq!(resumed_input.digest(), expected_input.digest());
        assert_eq!(merged_input.digest(), expected_input.digest());

        let mut one_fisher =
            OutputFisherAccumulator::new_bound(1, binding, 0).expect("one-shot Fisher");
        one_fisher
            .accumulate_batch(&gradients, 3, Some(&weights), None)
            .expect("one-shot Fisher batch");
        let mut fisher_zero =
            OutputFisherAccumulator::new_bound(1, binding, 0).expect("Fisher shard zero");
        fisher_zero
            .accumulate_batch(&gradients[..2], 2, Some(&weights[..2]), None)
            .expect("Fisher shard zero data");
        let mut fisher_one =
            OutputFisherAccumulator::new_bound(1, binding, 2).expect("Fisher shard one");
        fisher_one
            .accumulate_batch(&gradients[2..], 1, Some(&weights[2..]), None)
            .expect("Fisher shard one data");
        let mut merged_fisher =
            OutputFisherAccumulator::new_bound(1, binding, 0).expect("merged Fisher");
        merged_fisher
            .merge_shard(2, &fisher_one)
            .expect("reverse Fisher shard one");
        merged_fisher
            .merge_shard(0, &fisher_zero)
            .expect("reverse Fisher shard zero");
        let expected_fisher = one_fisher.finish().expect("one-shot Fisher evidence");
        let merged_fisher = merged_fisher.finish().expect("merged Fisher evidence");
        assert_eq!(merged_fisher.as_slice(), expected_fisher.as_slice());
        assert_eq!(merged_fisher.digest(), expected_fisher.digest());
        assert_eq!(
            build_kfac_metric(&merged_input, &merged_fisher, 0, 1e-4)
                .expect("merged solve")
                .digest(),
            build_kfac_metric(&expected_input, &expected_fisher, 0, 1e-4)
                .expect("one-shot solve")
                .digest()
        );
    }

    #[test]
    fn sample_order_is_bound_but_batch_boundaries_are_not() {
        let binding = source(21);
        let mut ab = InputGramAccumulator::new_bound(1, binding, 0).expect("ab");
        ab.accumulate_batch(&[1.0, 2.0], 2, None, None)
            .expect("ab data");
        let mut ba = InputGramAccumulator::new_bound(1, binding, 0).expect("ba");
        ba.accumulate_batch(&[2.0], 1, None, None).expect("b");
        ba.accumulate_batch(&[1.0], 1, None, None).expect("a");

        let ab = ab.finish().expect("ab evidence");
        let ba = ba.finish().expect("ba evidence");
        assert_eq!(ab.as_slice(), ba.as_slice());
        assert_ne!(ab.digest(), ba.digest());
    }

    #[test]
    fn source_cache_and_token_stream_identity_prevent_cross_pairing() {
        let mut input = InputGramAccumulator::new_bound(1, source(31), 0).expect("input");
        input
            .accumulate_batch(&[2.0], 1, None, None)
            .expect("input data");
        let mut fisher = OutputFisherAccumulator::new_bound(1, source(32), 0).expect("Fisher");
        fisher
            .accumulate_batch(&[2.0], 1, None, None)
            .expect("Fisher data");

        assert_eq!(
            build_kfac_metric(
                &input.finish().expect("input evidence"),
                &fisher.finish().expect("Fisher evidence"),
                0,
                1e-4,
            )
            .unwrap_err(),
            CurvatureError::SourceMismatch
        );

        let mut destination =
            InputGramAccumulator::new_bound(1, source(31), 0).expect("destination");
        let mut wrong_source =
            InputGramAccumulator::new_bound(1, source(32), 0).expect("wrong source");
        wrong_source
            .accumulate_batch(&[2.0], 1, None, None)
            .expect("wrong-source data");
        assert_eq!(
            destination.merge_shard(0, &wrong_source).unwrap_err(),
            CurvatureError::SourceMismatch
        );
    }

    #[test]
    fn token_mask_excludes_padding_from_gram_and_fisher() {
        let mask = [true, false, true];
        let mut input = InputGramAccumulator::new(2).expect("dimension");
        input
            .accumulate_batch(&[1.0, 2.0, 100.0, 200.0, 3.0, 4.0], 3, None, Some(&mask))
            .expect("masked input");
        let input = input.finish().expect("selected input");
        assert_eq!(input.as_slice(), &[5.0, 7.0, 7.0, 10.0]);
        assert_eq!(input.sample_count(), 3);
        assert_eq!(input.selected_count(), 2);

        let mut output = OutputFisherAccumulator::new(1).expect("rows");
        output
            .accumulate_batch(&[2.0, 100.0, 4.0], 3, None, Some(&mask))
            .expect("masked gradients");
        assert_eq!(
            output.finish().expect("selected output").as_slice(),
            &[10.0]
        );
    }

    #[test]
    fn damping_materializes_psd_metric_from_zero_curvature() {
        let binding = source(3);
        let mut input = InputGramAccumulator::new_bound(2, binding, 0).expect("dimension");
        input
            .accumulate_batch(&[0.0, 0.0], 1, None, None)
            .expect("zero activations are finite");
        let mut output = OutputFisherAccumulator::new_bound(1, binding, 0).expect("rows");
        output
            .accumulate_batch(&[0.0], 1, None, None)
            .expect("zero gradient is finite");
        let metric = build_kfac_metric(
            &input.finish().expect("positive sample weight"),
            &output.finish().expect("positive sample weight"),
            0,
            1e-4,
        )
        .expect("damping supplies positive diagonal");
        assert_eq!(metric.source_id(), binding);
        assert_eq!(metric.metric().as_slice(), &[1e-4, 0.0, 0.0, 1e-4]);
    }

    #[test]
    fn evidence_and_metric_digests_bind_order_data_dimensions_and_damping() {
        let mut ab = InputGramAccumulator::new(1).expect("dimension");
        ab.accumulate_batch(&[1.0], 1, None, None).expect("a");
        ab.accumulate_batch(&[2.0], 1, None, None).expect("b");
        let mut ba = InputGramAccumulator::new(1).expect("dimension");
        ba.accumulate_batch(&[2.0], 1, None, None).expect("b");
        ba.accumulate_batch(&[1.0], 1, None, None).expect("a");
        assert_eq!(
            ab.finish().expect("ab").as_slice(),
            ba.finish().expect("ba").as_slice()
        );
        assert_ne!(
            ab.finish().expect("ab").digest(),
            ba.finish().expect("ba").digest()
        );

        let mut changed = InputGramAccumulator::new(1).expect("dimension");
        changed
            .accumulate_batch(&[3.0, 0.0], 2, None, None)
            .expect("changed data");
        assert_ne!(
            ab.finish().expect("ab").digest(),
            changed.finish().expect("changed").digest()
        );

        let mut weight_one = InputGramAccumulator::new(1).expect("dimension");
        weight_one
            .accumulate_batch(&[1.0], 1, Some(&[1.0]), None)
            .expect("unit weight");
        let mut weight_two = InputGramAccumulator::new(1).expect("dimension");
        weight_two
            .accumulate_batch(&[1.0], 1, Some(&[2.0]), None)
            .expect("double weight");
        let weight_one = weight_one.finish().expect("unit-weight evidence");
        let weight_two = weight_two.finish().expect("double-weight evidence");
        assert_eq!(weight_one.as_slice(), weight_two.as_slice());
        assert_ne!(weight_one.digest(), weight_two.digest());

        let mut dimension_two = InputGramAccumulator::new(2).expect("dimension");
        dimension_two
            .accumulate_batch(&[1.0, 0.0], 1, Some(&[1.0]), None)
            .expect("two-dimensional data");
        assert_ne!(
            weight_one.digest(),
            dimension_two.finish().expect("dimension evidence").digest()
        );

        let (input, fisher) = analytic_evidence();
        let low = build_kfac_metric(&input, &fisher, 1, 1e-4).expect("low damping");
        let high = build_kfac_metric(&input, &fisher, 1, 1e-3).expect("high damping");
        assert_ne!(low.digest(), high.digest());
        assert_ne!(input.digest(), fisher.digest());
    }

    #[test]
    fn malformed_batches_zero_totals_and_order_misuse_are_rejected_atomically() {
        assert_eq!(
            InputGramAccumulator::new(0).unwrap_err(),
            CurvatureError::InvalidDimension
        );
        let mut input = InputGramAccumulator::new(2).expect("dimension");
        assert!(matches!(
            input.accumulate_batch(&[1.0], 1, None, None),
            Err(CurvatureError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            input.accumulate_batch(&[1.0, 2.0], 1, Some(&[]), None),
            Err(CurvatureError::TokenWeightLengthMismatch { .. })
        ));
        assert!(matches!(
            input.accumulate_batch(&[1.0, 2.0], 1, None, Some(&[])),
            Err(CurvatureError::TokenMaskLengthMismatch { .. })
        ));
        assert!(matches!(
            input.accumulate_batch(&[f32::NAN, 2.0], 1, None, None),
            Err(CurvatureError::NonFiniteActivation { .. })
        ));
        assert!(matches!(
            input.accumulate_batch(&[1.0, 2.0], 1, Some(&[-1.0]), None),
            Err(CurvatureError::InvalidTokenWeight { .. })
        ));
        assert_eq!(input.finish().unwrap_err(), CurvatureError::ZeroTotalWeight);

        let mut zero = InputGramAccumulator::new(2).expect("dimension");
        zero.accumulate_batch(&[1.0, 2.0], 1, Some(&[0.0]), None)
            .expect("zero weight is valid");
        assert_eq!(zero.finish().unwrap_err(), CurvatureError::ZeroTotalWeight);

        let mut shard = InputGramAccumulator::new(2).expect("dimension");
        shard
            .accumulate_batch(&[1.0, 2.0], 1, None, None)
            .expect("shard");
        let mut destination = InputGramAccumulator::new(2).expect("dimension");
        assert_eq!(
            destination.merge_shard(1, &shard).unwrap_err(),
            CurvatureError::MergeOrderMismatch {
                expected: 0,
                got: 1
            }
        );
        destination
            .merge_shard(0, &shard)
            .expect("state unchanged after error");

        let mut output = OutputFisherAccumulator::new(1).expect("rows");
        assert!(matches!(
            output.accumulate_batch(&[f32::INFINITY], 1, None, None),
            Err(CurvatureError::NonFiniteGradient { .. })
        ));
    }

    #[test]
    fn kfac_rejects_misaligned_selection_bad_rows_and_bad_damping() {
        let (input, fisher) = analytic_evidence();
        assert_eq!(
            build_kfac_metric(&input, &fisher, 2, 1e-4).unwrap_err(),
            CurvatureError::OutputRowOutOfRange { rows: 2, got: 2 }
        );
        assert_eq!(
            build_kfac_metric(&input, &fisher, 0, -1.0).unwrap_err(),
            CurvatureError::InvalidDamping
        );
        assert_eq!(
            build_kfac_metric(&input, &fisher, 0, f64::NAN).unwrap_err(),
            CurvatureError::InvalidDamping
        );

        let mut mismatched = OutputFisherAccumulator::new_bound(1, source(1), 0).expect("rows");
        mismatched
            .accumulate_batch(&[1.0], 1, None, None)
            .expect("different selection");
        assert_eq!(
            build_kfac_metric(
                &input,
                &mismatched.finish().expect("mismatched evidence"),
                0,
                1e-4,
            )
            .unwrap_err(),
            CurvatureError::SelectionMismatch
        );
    }
}
