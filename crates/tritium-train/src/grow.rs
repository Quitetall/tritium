//! Function-preserving width growth before SALT distillation (ADR 0027, Track F.2).
//!
//! [`Net2WiderPlan`] widens a hidden axis without changing the represented function:
//! every new hidden unit copies one old unit, while every corresponding outgoing
//! weight is divided by the number of copies of that old unit. For an input
//! projection `W_in: [hidden, input]` and output projection
//! `W_out: [output, hidden]`, the transformed tensors are therefore
//! `W_in': [wide_hidden, input]` and `W_out': [output, wide_hidden]`.
//!
//! A transformer SwiGLU block applies the same plan to the rows of both `gate` and
//! `up`, and once to the columns of `down`. Because all hidden operations between
//! them are pointwise, duplicate units remain equal and the divided `down` columns
//! sum to the original contribution. Width-dependent operations such as an
//! intermediate RMSNorm need a separate, architecture-specific transform and are
//! intentionally outside this prototype.
//! This is specifically an `n_ff`/intermediate-width expansion: `n_embd`, attention,
//! embeddings, residual paths, and transformer depth stay unchanged. Expanding the
//! residual-stream width would require a coupled transform across those components.
//!
//! This module only performs the exact fp32 tensor transform. A real growth run must
//! map model tensor names to these axes, verify the pre-SALT forward gate, then pass
//! the grown masters to the existing SALT distillation loop. The 1.7B 4090 run and
//! rented multi-GPU 32B quality/VRAM/step-time campaign remain hardware gates; this
//! pure CPU layout transform does not claim those results.

/// A deterministic Net2Wider transform for one hidden axis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Net2WiderPlan {
    old_width: usize,
    new_width: usize,
    source_for_new: Vec<usize>,
    copies_per_source: Vec<usize>,
}

/// Why a [`Net2WiderPlan`] could not be built or applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrowError {
    /// The source width is zero or the requested width would narrow the tensor.
    InvalidWidths {
        /// Existing hidden width.
        old_width: usize,
        /// Requested hidden width.
        new_width: usize,
    },
    /// The source width cannot be represented by the deterministic 64-bit mapper.
    UnsupportedWidth(usize),
    /// A tensor length did not match the documented row-major shape.
    ShapeMismatch {
        /// Logical tensor role.
        tensor: &'static str,
        /// Required element count.
        expected: usize,
        /// Supplied element count.
        actual: usize,
    },
    /// Computing the required tensor length overflowed `usize`.
    SizeOverflow {
        /// Logical tensor role.
        tensor: &'static str,
    },
}

impl core::fmt::Display for GrowError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GrowError::InvalidWidths {
                old_width,
                new_width,
            } => write!(
                f,
                "Net2Wider requires 0 < old_width <= new_width, got {old_width} -> {new_width}"
            ),
            GrowError::UnsupportedWidth(width) => {
                write!(f, "source width {width} exceeds the 64-bit mapping domain")
            }
            GrowError::ShapeMismatch {
                tensor,
                expected,
                actual,
            } => write!(
                f,
                "{tensor} length mismatch: expected {expected}, got {actual}"
            ),
            GrowError::SizeOverflow { tensor } => {
                write!(f, "{tensor} element count overflows usize")
            }
        }
    }
}

impl std::error::Error for GrowError {}

/// One measured point on Track F's quality-versus-storage curve.
///
/// `model_bytes` is the serialized inference artifact size, not an estimate from
/// parameter count. Lower `held_out_perplexity` and fewer bytes are better.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QualityBytesPoint {
    /// Number of model parameters represented by this artifact.
    pub parameter_count: u64,
    /// Serialized inference artifact size in bytes.
    pub model_bytes: u64,
    /// Held-out perplexity measured by the campaign's fixed evaluation harness.
    pub held_out_perplexity: f64,
}

impl QualityBytesPoint {
    /// Record one measured campaign point.
    #[must_use]
    pub const fn new(parameter_count: u64, model_bytes: u64, held_out_perplexity: f64) -> Self {
        Self {
            parameter_count,
            model_bytes,
            held_out_perplexity,
        }
    }
}

/// Measurements used to select Track F's quality-versus-bytes operating point.
///
/// This report performs no interpolation and makes no claim for unmeasured model
/// sizes. Non-finite perplexity values are excluded from selections.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct QualityBytesReport {
    measurements: Vec<QualityBytesPoint>,
}

impl QualityBytesReport {
    /// Build a report from measured points.
    #[must_use]
    pub fn new(measurements: Vec<QualityBytesPoint>) -> Self {
        Self { measurements }
    }

    /// Measurements in the caller's original order.
    #[must_use]
    pub fn measurements(&self) -> &[QualityBytesPoint] {
        &self.measurements
    }

    /// Non-dominated points, ordered by increasing serialized bytes.
    ///
    /// A point is omitted when another measurement has no more bytes and no worse
    /// perplexity, with at least one strict improvement. Distinct measurements with
    /// identical byte and perplexity coordinates are retained.
    #[must_use]
    pub fn pareto_frontier(&self) -> Vec<QualityBytesPoint> {
        let mut ordered: Vec<_> = self
            .measurements
            .iter()
            .copied()
            .filter(|point| point.held_out_perplexity.is_finite())
            .collect();
        ordered.sort_by(|a, b| {
            a.model_bytes
                .cmp(&b.model_bytes)
                .then_with(|| a.held_out_perplexity.total_cmp(&b.held_out_perplexity))
                .then_with(|| a.parameter_count.cmp(&b.parameter_count))
        });

        let mut best_perplexity = f64::INFINITY;
        let mut best_perplexity_bytes = None;
        ordered
            .into_iter()
            .filter(|point| {
                if point.held_out_perplexity < best_perplexity {
                    best_perplexity = point.held_out_perplexity;
                    best_perplexity_bytes = Some(point.model_bytes);
                    true
                } else {
                    point.held_out_perplexity == best_perplexity
                        && best_perplexity_bytes == Some(point.model_bytes)
                }
            })
            .collect()
    }

    /// Fewest-byte measured point whose perplexity is at most `maximum_perplexity`.
    ///
    /// Ties prefer lower perplexity, then fewer parameters. Returns `None` when no
    /// finite measurement satisfies the quality threshold.
    #[must_use]
    pub fn byte_optimal_at_or_better_than(
        &self,
        maximum_perplexity: f64,
    ) -> Option<QualityBytesPoint> {
        self.measurements
            .iter()
            .copied()
            .filter(|point| {
                point.held_out_perplexity.is_finite()
                    && point.held_out_perplexity <= maximum_perplexity
            })
            .min_by(|a, b| {
                a.model_bytes
                    .cmp(&b.model_bytes)
                    .then_with(|| a.held_out_perplexity.total_cmp(&b.held_out_perplexity))
                    .then_with(|| a.parameter_count.cmp(&b.parameter_count))
            })
    }
}

impl Net2WiderPlan {
    /// Build a deterministic widening map.
    ///
    /// The first `old_width` entries are the identity, guaranteeing every source
    /// unit has at least one copy. Additional entries are selected by SplitMix64,
    /// so the same `(old_width, new_width, seed)` produces the same map on 32- and
    /// 64-bit targets.
    ///
    /// # Errors
    /// [`GrowError::InvalidWidths`] if `old_width == 0` or `new_width < old_width`;
    /// [`GrowError::UnsupportedWidth`] on a target whose `usize` exceeds 64 bits.
    pub fn seeded(old_width: usize, new_width: usize, seed: u64) -> Result<Self, GrowError> {
        if old_width == 0 || new_width < old_width {
            return Err(GrowError::InvalidWidths {
                old_width,
                new_width,
            });
        }
        let old_width_u64 =
            u64::try_from(old_width).map_err(|_| GrowError::UnsupportedWidth(old_width))?;

        let mut source_for_new: Vec<usize> = (0..old_width).collect();
        let mut state = seed;
        source_for_new.reserve(new_width - old_width);
        for _ in old_width..new_width {
            let source_u64 = splitmix64(&mut state) % old_width_u64;
            let source = usize::try_from(source_u64)
                .expect("source index is modulo an existing usize width");
            source_for_new.push(source);
        }

        let mut copies_per_source = vec![0usize; old_width];
        for &source in &source_for_new {
            copies_per_source[source] += 1;
        }

        Ok(Self {
            old_width,
            new_width,
            source_for_new,
            copies_per_source,
        })
    }

    /// Source hidden-unit index for each unit in the widened tensor.
    ///
    /// The slice has length `new_width`; its identity prefix has length
    /// `old_width`. Persisting this mapping lets a model-growth command apply the
    /// exact same transform to every hidden-producing tensor in a block.
    #[must_use]
    pub fn source_indices(&self) -> &[usize] {
        &self.source_for_new
    }

    /// Number of widened units copied from each original hidden unit.
    ///
    /// The slice has length `old_width` and sums to `new_width`.
    #[must_use]
    pub fn replication_counts(&self) -> &[usize] {
        &self.copies_per_source
    }

    /// Duplicate a per-hidden-unit vector, such as an input projection's bias.
    ///
    /// `values` must have shape `[old_width]`; the result has shape
    /// `[new_width]`. A bias on the outgoing projection is not on the widened axis
    /// and must remain unchanged.
    ///
    /// # Errors
    /// [`GrowError::ShapeMismatch`] if `values.len() != old_width`.
    pub fn expand_hidden_vector(&self, values: &[f32]) -> Result<Vec<f32>, GrowError> {
        if values.len() != self.old_width {
            return Err(GrowError::ShapeMismatch {
                tensor: "hidden vector",
                expected: self.old_width,
                actual: values.len(),
            });
        }
        Ok(self
            .source_for_new
            .iter()
            .map(|&source| values[source])
            .collect())
    }

    /// Duplicate rows of an input projection.
    ///
    /// `weights` must be row-major `[old_width, input_width]`. The returned tensor
    /// is row-major `[new_width, input_width]`.
    ///
    /// # Errors
    /// [`GrowError::ShapeMismatch`] if the input length is not
    /// `old_width * input_width`, or [`GrowError::SizeOverflow`] if that product
    /// cannot be represented.
    pub fn expand_incoming_rows(
        &self,
        weights: &[f32],
        input_width: usize,
    ) -> Result<Vec<f32>, GrowError> {
        check_shape("incoming projection", weights, self.old_width, input_width)?;
        let output_len = checked_len("expanded incoming projection", self.new_width, input_width)?;
        let mut expanded = Vec::with_capacity(output_len);
        for &source in &self.source_for_new {
            let start = source * input_width;
            expanded.extend_from_slice(&weights[start..start + input_width]);
        }
        Ok(expanded)
    }

    /// Duplicate and rescale columns of an output projection.
    ///
    /// `weights` must be row-major `[output_width, old_width]`. The returned tensor
    /// is row-major `[output_width, new_width]`. Each copied column is divided by
    /// the number of copies of its source, so their sum is the original column.
    ///
    /// # Errors
    /// [`GrowError::ShapeMismatch`] if the input length is not
    /// `output_width * old_width`, or [`GrowError::SizeOverflow`] if a required
    /// element count cannot be represented.
    pub fn expand_outgoing_columns(
        &self,
        weights: &[f32],
        output_width: usize,
    ) -> Result<Vec<f32>, GrowError> {
        check_shape("outgoing projection", weights, output_width, self.old_width)?;
        let output_len = checked_len("expanded outgoing projection", output_width, self.new_width)?;
        let mut expanded = Vec::with_capacity(output_len);
        for row in weights.chunks_exact(self.old_width) {
            for &source in &self.source_for_new {
                expanded.push(row[source] / self.copies_per_source[source] as f32);
            }
        }
        Ok(expanded)
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn checked_len(tensor: &'static str, rows: usize, cols: usize) -> Result<usize, GrowError> {
    rows.checked_mul(cols)
        .ok_or(GrowError::SizeOverflow { tensor })
}

fn check_shape(
    tensor: &'static str,
    values: &[f32],
    rows: usize,
    cols: usize,
) -> Result<(), GrowError> {
    let expected = checked_len(tensor, rows, cols)?;
    if values.len() != expected {
        return Err(GrowError::ShapeMismatch {
            tensor,
            expected,
            actual: values.len(),
        });
    }
    Ok(())
}
