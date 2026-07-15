//! Checked SALT V2 additive-coefficient growth planning.

use crate::{
    ArchSpec, ModelConfig,
    training::{TiedSwiGluTrainingModel, TrainingAdapterError, semantic_training_model_digest},
};
use tritium_train::{GrowError, Net2WiderPlan};

/// Largest additive plane count in the SALT V2 synthesis representation.
pub const MAX_ADDITIVE_PLANES: u8 = 3;

/// Additive-coefficient targets at or above this boundary must retain an
/// explicit whole-head and hidden-width growth stage. Intermediate-only growth
/// is not a valid endpoint at this scale.
pub const WHOLE_HEAD_AND_HIDDEN_COEFFICIENT_THRESHOLD: u64 = 50_000_000_000;

/// Maximum stage-1 width for an eagerly materialized deterministic Net2Wider receipt.
///
/// The receipt stores multiple vectors proportional to this width. Bounding it
/// before calling the legacy infallible mapping constructor prevents a valid but
/// hostile coefficient target from turning planning into a multi-gigabyte allocation.
pub const MAX_STAGE1_RECEIPT_WIDTH: u32 = 1_048_576;

/// Semantic identity of the exact source checkpoint consumed by a growth plan.
///
/// Callers should construct this from [`crate::semantic_training_model_digest`]
/// after loading and validating the training model. The all-zero value is
/// reserved for "identity not supplied" and is rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GrowthSourceModelId([u8; 32]);

impl GrowthSourceModelId {
    /// Bind a nonzero semantic model digest.
    pub fn new(digest: [u8; 32]) -> Result<Self, GrowthPlanError> {
        if digest == [0; 32] {
            return Err(GrowthPlanError::InvalidSourceModelId);
        }
        Ok(Self(digest))
    }

    /// Compute the semantic identity of a validated training checkpoint.
    pub fn from_training_model(
        config: &ModelConfig,
        spec: &ArchSpec,
        model: &TiedSwiGluTrainingModel,
    ) -> Result<Self, GrowthPlanError> {
        Self::new(semantic_training_model_digest(config, spec, model))
    }

    /// Canonical 32-byte semantic digest.
    #[must_use]
    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

/// Additive ternary plane counts for every transformer core projection.
///
/// Each scalar in a projection contributes one stored ternary coefficient per
/// plane. This type deliberately does not count an fp parameter once when two or
/// three additive ternary planes are stored for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjectionPlaneCounts {
    /// Query projection planes.
    query: u8,
    /// Key projection planes.
    key: u8,
    /// Value projection planes.
    value: u8,
    /// Attention output projection planes.
    attention_output: u8,
    /// SwiGLU gate projection planes.
    gate: u8,
    /// SwiGLU up projection planes.
    up: u8,
    /// SwiGLU down projection planes.
    down: u8,
}

impl ProjectionPlaneCounts {
    /// Build checked SALT V2 plane counts. Every core projection uses one to
    /// three planes; preserving a tensor is represented by the separate fixed
    /// embedding policy, not by a zero-plane core projection.
    pub fn new(
        query: u8,
        key: u8,
        value: u8,
        attention_output: u8,
        gate: u8,
        up: u8,
        down: u8,
    ) -> Result<Self, GrowthPlanError> {
        for (projection, planes) in [
            ("query", query),
            ("key", key),
            ("value", value),
            ("attention_output", attention_output),
            ("gate", gate),
            ("up", up),
            ("down", down),
        ] {
            if !(1..=MAX_ADDITIVE_PLANES).contains(&planes) {
                return Err(GrowthPlanError::InvalidPlaneCount { projection, planes });
            }
        }
        Ok(Self {
            query,
            key,
            value,
            attention_output,
            gate,
            up,
            down,
        })
    }

    /// Plane counts in query, key, value, attention-output, gate, up, and down order.
    #[must_use]
    pub const fn as_array(self) -> [u8; 7] {
        [
            self.query,
            self.key,
            self.value,
            self.attention_output,
            self.gate,
            self.up,
            self.down,
        ]
    }
}

/// Exact number of scalar weights assigned to each additive plane count in one
/// projection role.
///
/// Unlike [`ProjectionPlaneCounts`], this ledger can represent a projection
/// containing a mixture of P1, P2, and P3 scalar weights without rounding the
/// stored-coefficient count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneWeightHistogram {
    /// Scalar weights represented by one ternary coefficient.
    p1_weights: u64,
    /// Scalar weights represented by two additive ternary coefficients.
    p2_weights: u64,
    /// Scalar weights represented by three additive ternary coefficients.
    p3_weights: u64,
    total_weights: u64,
    stored_coefficients: u64,
}

impl PlaneWeightHistogram {
    /// Build a checked exact plane histogram.
    pub fn new(p1_weights: u64, p2_weights: u64, p3_weights: u64) -> Result<Self, GrowthPlanError> {
        let total_weights = count_to_u64(
            "plane-ledger scalar weights",
            u128::from(p1_weights) + u128::from(p2_weights) + u128::from(p3_weights),
        )?;
        let stored_coefficients = count_to_u64(
            "plane-ledger stored coefficients",
            u128::from(p1_weights) + 2 * u128::from(p2_weights) + 3 * u128::from(p3_weights),
        )?;
        Ok(Self {
            p1_weights,
            p2_weights,
            p3_weights,
            total_weights,
            stored_coefficients,
        })
    }

    /// Total scalar weights covered by the histogram.
    #[must_use]
    pub const fn total_weights(self) -> u64 {
        self.total_weights
    }

    /// Scalar-weight counts represented by P1, P2, and P3 respectively.
    #[must_use]
    pub const fn weights_by_plane_count(self) -> [u64; 3] {
        [self.p1_weights, self.p2_weights, self.p3_weights]
    }

    /// Exact number of stored additive ternary coefficients.
    #[must_use]
    pub const fn stored_coefficients(self) -> u64 {
        self.stored_coefficients
    }

    fn uniform(
        projection: &'static str,
        weights: u64,
        planes: u8,
    ) -> Result<Self, GrowthPlanError> {
        validate_fixed_planes(projection, planes)?;
        match planes {
            1 => Self::new(weights, 0, 0),
            2 => Self::new(0, weights, 0),
            3 => Self::new(0, 0, weights),
            _ => unreachable!("plane count was validated"),
        }
    }
}

/// Exact mixed-plane coefficient ledger for all transformer core projections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjectionCoefficientLedger {
    /// Query projection histogram.
    query: PlaneWeightHistogram,
    /// Key projection histogram.
    key: PlaneWeightHistogram,
    /// Value projection histogram.
    value: PlaneWeightHistogram,
    /// Attention output projection histogram.
    attention_output: PlaneWeightHistogram,
    /// SwiGLU gate projection histogram.
    gate: PlaneWeightHistogram,
    /// SwiGLU up projection histogram.
    up: PlaneWeightHistogram,
    /// SwiGLU down projection histogram.
    down: PlaneWeightHistogram,
}

impl ProjectionCoefficientLedger {
    /// Build a checked projection ledger.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        query: PlaneWeightHistogram,
        key: PlaneWeightHistogram,
        value: PlaneWeightHistogram,
        attention_output: PlaneWeightHistogram,
        gate: PlaneWeightHistogram,
        up: PlaneWeightHistogram,
        down: PlaneWeightHistogram,
    ) -> Result<Self, GrowthPlanError> {
        let ledger = Self {
            query,
            key,
            value,
            attention_output,
            gate,
            up,
            down,
        };
        count_to_u64(
            "mixed projection ledger coefficients",
            ledger
                .histograms()
                .into_iter()
                .map(|histogram| u128::from(histogram.stored_coefficients()))
                .sum(),
        )?;
        Ok(ledger)
    }

    /// Histograms in query, key, value, attention-output, gate, up, and down order.
    #[must_use]
    pub const fn as_array(self) -> [PlaneWeightHistogram; 7] {
        self.histograms()
    }

    const fn histograms(self) -> [PlaneWeightHistogram; 7] {
        [
            self.query,
            self.key,
            self.value,
            self.attention_output,
            self.gate,
            self.up,
            self.down,
        ]
    }
}

/// Storage treatment for embeddings and the language-model head, whose geometry
/// remains fixed during intermediate-only growth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixedEmbeddingPolicy {
    /// Keep embedding/head weights dense and exclude them from additive ternary
    /// coefficient counts. `tied_lm_head` records whether one or two dense tensors
    /// are retained in the physical artifact.
    PreservedDense {
        /// Whether the output head aliases the input embedding.
        tied_lm_head: bool,
    },
    /// Store one tied embedding/head matrix as additive ternary planes.
    AdditiveTernaryTied {
        /// Plane count for the shared matrix.
        planes: u8,
    },
    /// Store separate input embedding and output-head matrices as additive
    /// ternary planes.
    AdditiveTernaryUntied {
        /// Input embedding plane count.
        embedding_planes: u8,
        /// Output-head plane count.
        lm_head_planes: u8,
    },
}

impl FixedEmbeddingPolicy {
    fn validate(self) -> Result<(), GrowthPlanError> {
        match self {
            Self::PreservedDense { .. } => Ok(()),
            Self::AdditiveTernaryTied { planes } => {
                validate_fixed_planes("embedding_and_tied_lm_head", planes)
            }
            Self::AdditiveTernaryUntied {
                embedding_planes,
                lm_head_planes,
            } => {
                validate_fixed_planes("embedding", embedding_planes)?;
                validate_fixed_planes("lm_head", lm_head_planes)
            }
        }
    }

    fn tied_lm_head(self) -> bool {
        matches!(
            self,
            Self::PreservedDense { tied_lm_head: true } | Self::AdditiveTernaryTied { .. }
        )
    }

    fn plane_sum(self) -> u8 {
        match self {
            Self::PreservedDense { .. } => 0,
            Self::AdditiveTernaryTied { planes } => planes,
            Self::AdditiveTernaryUntied {
                embedding_planes,
                lm_head_planes,
            } => embedding_planes + lm_head_planes,
        }
    }
}

/// Checked transformer projection geometry used by the growth planner.
///
/// Attention, depth, residual width, and embeddings are held fixed. Only the
/// uniform SwiGLU intermediate width changes in stage 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjectionGeometry {
    /// Transformer block count.
    layers: u32,
    /// Residual-stream width.
    residual_width: u32,
    /// Query projection output width.
    query_width: u32,
    /// Key/value projection output width.
    key_value_width: u32,
    /// Existing SwiGLU intermediate width.
    intermediate_width: u32,
    /// Token vocabulary size.
    vocabulary: u32,
    /// Uniform plane policy used for newly added intermediate units.
    planes: ProjectionPlaneCounts,
    /// Exact P1/P2/P3 allocation across the existing core projections.
    coefficient_ledger: ProjectionCoefficientLedger,
    /// Fixed embedding and language-model-head treatment.
    embedding_policy: FixedEmbeddingPolicy,
}

impl ProjectionGeometry {
    /// Build portable geometry whose axes fit both persisted `u32` receipts and
    /// the current target's `usize` tensor indexing.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        layers: usize,
        residual_width: usize,
        query_width: usize,
        key_value_width: usize,
        intermediate_width: usize,
        vocabulary: usize,
        planes: ProjectionPlaneCounts,
        embedding_policy: FixedEmbeddingPolicy,
    ) -> Result<Self, GrowthPlanError> {
        let axes = CheckedProjectionAxes::new(
            layers,
            residual_width,
            query_width,
            key_value_width,
            intermediate_width,
            vocabulary,
        )?;
        let coefficient_ledger = axes.uniform_ledger(planes)?;
        Self::from_checked_axes(axes, coefficient_ledger, planes, embedding_policy)
    }

    /// Build geometry using an exact mixed P1/P2/P3 allocation for the existing
    /// model and a uniform plane policy for newly added intermediate units.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_ledger(
        layers: usize,
        residual_width: usize,
        query_width: usize,
        key_value_width: usize,
        intermediate_width: usize,
        vocabulary: usize,
        coefficient_ledger: ProjectionCoefficientLedger,
        growth_planes: ProjectionPlaneCounts,
        embedding_policy: FixedEmbeddingPolicy,
    ) -> Result<Self, GrowthPlanError> {
        let axes = CheckedProjectionAxes::new(
            layers,
            residual_width,
            query_width,
            key_value_width,
            intermediate_width,
            vocabulary,
        )?;
        Self::from_checked_axes(axes, coefficient_ledger, growth_planes, embedding_policy)
    }

    fn from_checked_axes(
        axes: CheckedProjectionAxes,
        coefficient_ledger: ProjectionCoefficientLedger,
        planes: ProjectionPlaneCounts,
        embedding_policy: FixedEmbeddingPolicy,
    ) -> Result<Self, GrowthPlanError> {
        embedding_policy.validate()?;
        axes.validate_ledger(coefficient_ledger)?;
        let geometry = Self {
            layers: axes.layers,
            residual_width: axes.residual_width,
            query_width: axes.query_width,
            key_value_width: axes.key_value_width,
            intermediate_width: axes.intermediate_width,
            vocabulary: axes.vocabulary,
            planes,
            coefficient_ledger,
            embedding_policy,
        };
        geometry.fixed_coefficient_count()?;
        geometry.core_coefficient_count(geometry.intermediate_width)?;
        Ok(geometry)
    }

    /// Exact mixed-plane ledger bound to the existing model geometry.
    #[must_use]
    pub const fn coefficient_ledger(&self) -> ProjectionCoefficientLedger {
        self.coefficient_ledger
    }

    /// Portable geometry axes in layers, residual, query, key/value, intermediate, vocabulary order.
    #[must_use]
    pub const fn axes(&self) -> [u32; 6] {
        [
            self.layers,
            self.residual_width,
            self.query_width,
            self.key_value_width,
            self.intermediate_width,
            self.vocabulary,
        ]
    }

    /// Plane policy applied only to newly added intermediate units.
    #[must_use]
    pub const fn growth_planes(&self) -> ProjectionPlaneCounts {
        self.planes
    }

    /// Fixed embedding and language-model-head storage policy.
    #[must_use]
    pub const fn embedding_policy(&self) -> FixedEmbeddingPolicy {
        self.embedding_policy
    }

    /// Stored additive ternary coefficients unaffected by intermediate growth.
    /// Preserved dense embedding/head tensors are intentionally excluded.
    pub fn fixed_coefficient_count(&self) -> Result<u64, GrowthPlanError> {
        let attention = [
            self.coefficient_ledger.query,
            self.coefficient_ledger.key,
            self.coefficient_ledger.value,
            self.coefficient_ledger.attention_output,
        ]
        .into_iter()
        .map(|histogram| u128::from(histogram.stored_coefficients()))
        .sum::<u128>();
        let embedding = u128::from(self.vocabulary)
            * u128::from(self.residual_width)
            * u128::from(self.embedding_policy.plane_sum());
        count_to_u64("fixed projection coefficients", attention + embedding)
    }

    /// Stored additive ternary coefficients for the full core at `width`.
    pub fn core_coefficient_count(&self, width: u32) -> Result<u64, GrowthPlanError> {
        if width < self.intermediate_width {
            return Err(GrowthPlanError::IntermediateNarrowing {
                existing: self.intermediate_width,
                requested: width,
            });
        }
        let added_width = u128::from(width - self.intermediate_width);
        let total = self.base_core_coefficient_count()
            + self.coefficients_per_intermediate_unit() * added_width;
        count_to_u64("core projection coefficients", total)
    }

    /// Compute the minimum uniform intermediate width that reaches the stage-1
    /// coefficient floor without narrowing the existing model.
    pub fn plan(
        &self,
        source_model_id: GrowthSourceModelId,
        target: GrowthTarget,
        seed: u64,
    ) -> Result<IntermediateGrowthPlan, GrowthPlanError> {
        let fixed = self.fixed_coefficient_count()?;
        let base = self.core_coefficient_count(self.intermediate_width)?;
        let floor = target.intermediate_coefficient_floor();
        let required_width = if floor <= base {
            u128::from(self.intermediate_width)
        } else {
            let additional_floor = u128::from(floor - base);
            let slope = self.coefficients_per_intermediate_unit();
            u128::from(self.intermediate_width) + additional_floor.div_ceil(slope)
        };
        let required_width = required_width.max(u128::from(self.intermediate_width));
        let new_width =
            u32::try_from(required_width).map_err(|_| GrowthPlanError::WidthOutOfRange {
                axis: "planned_intermediate_width",
                value: required_width,
            })?;
        if new_width > MAX_STAGE1_RECEIPT_WIDTH {
            return Err(GrowthPlanError::PlanReceiptTooLarge {
                width: new_width,
                maximum: MAX_STAGE1_RECEIPT_WIDTH,
            });
        }
        usize::try_from(new_width).map_err(|_| GrowthPlanError::WidthOutOfRange {
            axis: "planned_intermediate_width",
            value: required_width,
        })?;

        let resulting = self.core_coefficient_count(new_width)?;
        if resulting >= WHOLE_HEAD_AND_HIDDEN_COEFFICIENT_THRESHOLD {
            return Err(GrowthPlanError::WholeHeadAndHiddenRequired { target: resulting });
        }
        let plan = IntermediateGrowthPlan {
            source_model_id,
            geometry: *self,
            target,
            old_width: self.intermediate_width,
            new_width,
            fixed_coefficient_count: fixed,
            base_core_coefficient_count: base,
            resulting_core_coefficient_count: resulting,
            seed,
            stage2_requirement: target.stage2_requirement(),
        };
        // A numerical width is not a usable stage-1 plan unless the existing
        // deterministic Net2Wider implementation can build its replay mapping.
        plan.expected_net2wider_plan()?;
        Ok(plan)
    }

    fn coefficients_per_intermediate_unit(&self) -> u128 {
        u128::from(self.layers)
            * u128::from(self.residual_width)
            * u128::from(self.planes.gate + self.planes.up + self.planes.down)
    }

    fn base_core_coefficient_count(&self) -> u128 {
        let core = self
            .coefficient_ledger
            .histograms()
            .into_iter()
            .map(|histogram| u128::from(histogram.stored_coefficients()))
            .sum::<u128>();
        let embedding = u128::from(self.vocabulary)
            * u128::from(self.residual_width)
            * u128::from(self.embedding_policy.plane_sum());
        core + embedding
    }
}

#[derive(Clone, Copy)]
struct CheckedProjectionAxes {
    layers: u32,
    residual_width: u32,
    query_width: u32,
    key_value_width: u32,
    intermediate_width: u32,
    vocabulary: u32,
}

impl CheckedProjectionAxes {
    fn new(
        layers: usize,
        residual_width: usize,
        query_width: usize,
        key_value_width: usize,
        intermediate_width: usize,
        vocabulary: usize,
    ) -> Result<Self, GrowthPlanError> {
        Ok(Self {
            layers: checked_axis("layers", layers)?,
            residual_width: checked_axis("residual_width", residual_width)?,
            query_width: checked_axis("query_width", query_width)?,
            key_value_width: checked_axis("key_value_width", key_value_width)?,
            intermediate_width: checked_axis("intermediate_width", intermediate_width)?,
            vocabulary: checked_axis("vocabulary", vocabulary)?,
        })
    }

    fn uniform_ledger(
        self,
        planes: ProjectionPlaneCounts,
    ) -> Result<ProjectionCoefficientLedger, GrowthPlanError> {
        let expected = self.expected_projection_weights()?;
        ProjectionCoefficientLedger::new(
            PlaneWeightHistogram::uniform("query", expected[0].1, planes.query)?,
            PlaneWeightHistogram::uniform("key", expected[1].1, planes.key)?,
            PlaneWeightHistogram::uniform("value", expected[2].1, planes.value)?,
            PlaneWeightHistogram::uniform(
                "attention_output",
                expected[3].1,
                planes.attention_output,
            )?,
            PlaneWeightHistogram::uniform("gate", expected[4].1, planes.gate)?,
            PlaneWeightHistogram::uniform("up", expected[5].1, planes.up)?,
            PlaneWeightHistogram::uniform("down", expected[6].1, planes.down)?,
        )
    }

    fn validate_ledger(self, ledger: ProjectionCoefficientLedger) -> Result<(), GrowthPlanError> {
        for ((projection, expected_weights), histogram) in self
            .expected_projection_weights()?
            .into_iter()
            .zip(ledger.histograms())
        {
            let actual_weights = histogram.total_weights();
            if expected_weights != actual_weights {
                return Err(GrowthPlanError::PlaneLedgerShapeMismatch {
                    projection,
                    expected_weights,
                    actual_weights,
                });
            }
        }
        Ok(())
    }

    fn expected_projection_weights(self) -> Result<[(&'static str, u64); 7], GrowthPlanError> {
        let layers = u128::from(self.layers);
        let residual = u128::from(self.residual_width);
        let query = count_to_u64(
            "query projection scalar weights",
            layers * residual * u128::from(self.query_width),
        )?;
        let key_value = count_to_u64(
            "key/value projection scalar weights",
            layers * residual * u128::from(self.key_value_width),
        )?;
        let intermediate = count_to_u64(
            "intermediate projection scalar weights",
            layers * residual * u128::from(self.intermediate_width),
        )?;
        Ok([
            ("query", query),
            ("key", key_value),
            ("value", key_value),
            ("attention_output", query),
            ("gate", intermediate),
            ("up", intermediate),
            ("down", intermediate),
        ])
    }
}

/// A checked additive-coefficient target and its intended architecture scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GrowthTarget {
    intermediate_coefficient_floor: u64,
    stage2_requirement: Stage2Requirement,
}

impl GrowthTarget {
    /// Require at least `coefficients` after intermediate-only growth.
    pub fn intermediate_at_least(coefficients: u64) -> Result<Self, GrowthPlanError> {
        if coefficients == 0 {
            return Err(GrowthPlanError::ZeroTarget);
        }
        if coefficients >= WHOLE_HEAD_AND_HIDDEN_COEFFICIENT_THRESHOLD {
            return Err(GrowthPlanError::WholeHeadAndHiddenRequired {
                target: coefficients,
            });
        }
        Ok(Self {
            intermediate_coefficient_floor: coefficients,
            stage2_requirement: Stage2Requirement::NotRequested,
        })
    }

    /// Plan a stage-1 intermediate floor while explicitly retaining the future
    /// whole-head/hidden stage needed for the final architecture target.
    pub fn staged(
        intermediate_coefficients: u64,
        final_coefficients: u64,
    ) -> Result<Self, GrowthPlanError> {
        if intermediate_coefficients == 0 || final_coefficients == 0 {
            return Err(GrowthPlanError::ZeroTarget);
        }
        if intermediate_coefficients >= WHOLE_HEAD_AND_HIDDEN_COEFFICIENT_THRESHOLD {
            return Err(GrowthPlanError::WholeHeadAndHiddenRequired {
                target: intermediate_coefficients,
            });
        }
        if final_coefficients <= intermediate_coefficients
            || final_coefficients < WHOLE_HEAD_AND_HIDDEN_COEFFICIENT_THRESHOLD
        {
            return Err(GrowthPlanError::InvalidStage2Target {
                intermediate: intermediate_coefficients,
                final_target: final_coefficients,
            });
        }
        Ok(Self {
            intermediate_coefficient_floor: intermediate_coefficients,
            stage2_requirement: Stage2Requirement::WholeHeadAndHidden {
                final_coefficient_floor: final_coefficients,
            },
        })
    }

    /// Stage-1 coefficient floor used by the intermediate-width calculation.
    #[must_use]
    pub const fn intermediate_coefficient_floor(self) -> u64 {
        self.intermediate_coefficient_floor
    }

    /// Explicit follow-on architecture requirement.
    #[must_use]
    pub const fn stage2_requirement(self) -> Stage2Requirement {
        self.stage2_requirement
    }
}

/// Whether coefficient-count stage 1 is also intended to feed whole-model growth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage2Requirement {
    /// The caller requested only an intermediate-width coefficient floor.
    NotRequested,
    /// Stage 1 does not satisfy this final target by itself: residual width and
    /// complete attention heads must be grown by a future architecture-aware
    /// transform and revalidated end to end.
    WholeHeadAndHidden {
        /// Minimum stored additive ternary coefficients after stage 2.
        final_coefficient_floor: u64,
    },
}

/// Replayable stage-1 intermediate-width growth plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntermediateGrowthPlan {
    /// Exact semantic checkpoint consumed by this replayable transform.
    source_model_id: GrowthSourceModelId,
    geometry: ProjectionGeometry,
    target: GrowthTarget,
    /// Existing uniform SwiGLU intermediate width.
    old_width: u32,
    /// Smallest uniform width satisfying the stage-1 coefficient floor.
    new_width: u32,
    /// Fixed attention plus additively ternarized embedding/head coefficients.
    fixed_coefficient_count: u64,
    /// Full stored additive ternary coefficient count before growth.
    base_core_coefficient_count: u64,
    /// Full stored additive ternary coefficient count after stage 1.
    resulting_core_coefficient_count: u64,
    /// Seed binding the deterministic Net2Wider receipt.
    seed: u64,
    /// Typed future architecture work, if requested.
    stage2_requirement: Stage2Requirement,
}

impl IntermediateGrowthPlan {
    /// Semantic source checkpoint bound into planning, application, and receipt validation.
    #[must_use]
    pub const fn source_model_id(&self) -> GrowthSourceModelId {
        self.source_model_id
    }

    /// Geometry bound into this plan.
    #[must_use]
    pub const fn geometry(&self) -> ProjectionGeometry {
        self.geometry
    }

    /// Target bound into this plan.
    #[must_use]
    pub const fn target(&self) -> GrowthTarget {
        self.target
    }

    /// Existing uniform SwiGLU intermediate width.
    #[must_use]
    pub const fn old_width(&self) -> u32 {
        self.old_width
    }

    /// Smallest uniform width satisfying the stage-1 coefficient floor.
    #[must_use]
    pub const fn new_width(&self) -> u32 {
        self.new_width
    }

    /// Fixed attention plus additively ternarized embedding/head coefficients.
    #[must_use]
    pub const fn fixed_coefficient_count(&self) -> u64 {
        self.fixed_coefficient_count
    }

    /// Full stored additive ternary coefficient count before growth.
    #[must_use]
    pub const fn base_core_coefficient_count(&self) -> u64 {
        self.base_core_coefficient_count
    }

    /// Full stored additive ternary coefficient count after stage 1.
    #[must_use]
    pub const fn resulting_core_coefficient_count(&self) -> u64 {
        self.resulting_core_coefficient_count
    }

    /// Seed binding deterministic Net2Wider replay.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Typed follow-on architecture requirement.
    #[must_use]
    pub const fn stage2_requirement(&self) -> Stage2Requirement {
        self.stage2_requirement
    }

    fn expected_net2wider_plan(&self) -> Result<Net2WiderPlan, GrowthPlanError> {
        let old_width =
            usize::try_from(self.old_width).map_err(|_| GrowthPlanError::WidthOutOfRange {
                axis: "old_intermediate_width",
                value: u128::from(self.old_width),
            })?;
        let new_width =
            usize::try_from(self.new_width).map_err(|_| GrowthPlanError::WidthOutOfRange {
                axis: "new_intermediate_width",
                value: u128::from(self.new_width),
            })?;
        Net2WiderPlan::seeded(old_width, new_width, self.seed).map_err(GrowthPlanError::Net2Wider)
    }

    /// Verify that a returned transform receipt is exactly the deterministic
    /// receipt bound by this plan. Receipts can only be issued by [`Self::apply`].
    pub fn validate_receipt(
        &self,
        receipt: &IntermediateGrowthReceipt,
    ) -> Result<(), GrowthPlanError> {
        self.validate_source_model_id(receipt.source_model_id)?;
        if self.expected_net2wider_plan()? != receipt.net2wider {
            return Err(GrowthPlanError::ReceiptMismatch);
        }
        Ok(())
    }

    /// Verify an application target is the exact checkpoint bound during planning.
    pub fn validate_source_model_id(
        &self,
        actual: GrowthSourceModelId,
    ) -> Result<(), GrowthPlanError> {
        if actual != self.source_model_id {
            return Err(GrowthPlanError::SourceModelMismatch {
                expected: self.source_model_id,
                actual,
            });
        }
        Ok(())
    }

    /// Apply this plan to a compatible tied-SwiGLU training model and verify the
    /// receipt returned by the existing Net2Wider transform.
    pub fn apply(
        &self,
        config: &ModelConfig,
        spec: &ArchSpec,
        model: &mut TiedSwiGluTrainingModel,
    ) -> Result<IntermediateGrowthReceipt, GrowthPlanError> {
        let source_model_id = GrowthSourceModelId::from_training_model(config, spec, model)?;
        self.validate_source_model_id(source_model_id)?;
        self.validate_model_geometry(model)?;
        let net2wider = model
            .widen_intermediate(
                usize::try_from(self.new_width).map_err(|_| GrowthPlanError::WidthOutOfRange {
                    axis: "new_intermediate_width",
                    value: u128::from(self.new_width),
                })?,
                self.seed,
            )
            .map_err(GrowthPlanError::TrainingAdapter)?;
        let receipt = IntermediateGrowthReceipt {
            source_model_id,
            net2wider,
        };
        self.validate_receipt(&receipt)?;
        Ok(receipt)
    }

    fn validate_model_geometry(
        &self,
        model: &TiedSwiGluTrainingModel,
    ) -> Result<(), GrowthPlanError> {
        let arch = model.architecture();
        let query_width = arch
            .n_head
            .checked_mul(arch.head_dim)
            .ok_or(GrowthPlanError::ModelGeometryOverflow("query_width"))?;
        let key_value_width = arch
            .n_head_kv
            .checked_mul(arch.head_dim)
            .ok_or(GrowthPlanError::ModelGeometryOverflow("key_value_width"))?;
        for (axis, expected, actual) in [
            ("layers", self.geometry.layers, arch.n_layers),
            ("residual_width", self.geometry.residual_width, arch.n_embd),
            ("query_width", self.geometry.query_width, query_width),
            (
                "key_value_width",
                self.geometry.key_value_width,
                key_value_width,
            ),
            (
                "intermediate_width",
                self.geometry.intermediate_width,
                arch.n_ff,
            ),
            ("vocabulary", self.geometry.vocabulary, arch.vocab),
        ] {
            if usize::try_from(expected).ok() != Some(actual) {
                return Err(GrowthPlanError::ModelGeometryMismatch {
                    axis,
                    expected,
                    actual,
                });
            }
        }
        let expected_tied = self.geometry.embedding_policy.tied_lm_head();
        if expected_tied != model.is_lm_head_tied() {
            return Err(GrowthPlanError::ModelTyingMismatch {
                expected_tied,
                actual_tied: model.is_lm_head_tied(),
            });
        }
        Ok(())
    }
}

/// Source-bound receipt issued after one deterministic intermediate-width transform.
///
/// Its fields are intentionally private: callers can validate or persist a
/// receipt returned by [`IntermediateGrowthPlan::apply`], but cannot mint one
/// from a plan that has not been applied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntermediateGrowthReceipt {
    source_model_id: GrowthSourceModelId,
    net2wider: Net2WiderPlan,
}

impl IntermediateGrowthReceipt {
    /// Semantic checkpoint transformed by this receipt.
    #[must_use]
    pub const fn source_model_id(&self) -> GrowthSourceModelId {
        self.source_model_id
    }

    /// Deterministic mapping and split metadata applied to the checkpoint.
    #[must_use]
    pub const fn net2wider(&self) -> &Net2WiderPlan {
        &self.net2wider
    }
}

/// Why checked SALT V2 growth planning or application failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GrowthPlanError {
    /// A growth plan was not bound to a semantic source-checkpoint digest.
    InvalidSourceModelId,
    /// Application or replay named a different semantic source checkpoint.
    SourceModelMismatch {
        /// Checkpoint digest frozen in the plan.
        expected: GrowthSourceModelId,
        /// Checkpoint digest supplied by the caller or receipt.
        actual: GrowthSourceModelId,
    },
    /// A required geometry axis is zero.
    ZeroAxis(&'static str),
    /// A geometry axis cannot fit the portable `u32` receipt domain.
    AxisOutOfRange {
        /// Axis name.
        axis: &'static str,
        /// Supplied platform-width value.
        value: usize,
    },
    /// A core or fixed projection requested an unsupported plane count.
    InvalidPlaneCount {
        /// Projection role.
        projection: &'static str,
        /// Supplied plane count.
        planes: u8,
    },
    /// An exact additive coefficient count exceeded `u64`.
    CoefficientCountOverflow(&'static str),
    /// An exact mixed-plane ledger does not cover the projection geometry.
    PlaneLedgerShapeMismatch {
        /// Projection role.
        projection: &'static str,
        /// Scalar weights required by the projection geometry.
        expected_weights: u64,
        /// Scalar weights assigned by the ledger.
        actual_weights: u64,
    },
    /// Existing intermediate units cannot be removed by additive growth.
    IntermediateNarrowing {
        /// Existing intermediate width.
        existing: u32,
        /// Requested intermediate width.
        requested: u32,
    },
    /// A target coefficient count must be non-zero.
    ZeroTarget,
    /// This target is too large to use intermediate-only growth as an endpoint.
    WholeHeadAndHiddenRequired {
        /// Rejected intermediate-only target.
        target: u64,
    },
    /// A staged final target must exceed stage 1 and reach the whole-head/hidden frontier.
    InvalidStage2Target {
        /// Stage-1 floor.
        intermediate: u64,
        /// Requested final floor.
        final_target: u64,
    },
    /// A planned width cannot fit the persisted or platform indexing domain.
    WidthOutOfRange {
        /// Width role.
        axis: &'static str,
        /// Exact computed width.
        value: u128,
    },
    /// An eagerly materialized Net2Wider receipt would exceed its bounded plan width.
    PlanReceiptTooLarge {
        /// Computed stage-1 width.
        width: u32,
        /// Largest width admitted by the eager receipt representation.
        maximum: u32,
    },
    /// The existing deterministic Net2Wider implementation rejected the plan.
    Net2Wider(GrowError),
    /// A supplied receipt does not match deterministic replay.
    ReceiptMismatch,
    /// A model-derived width overflowed `usize`.
    ModelGeometryOverflow(&'static str),
    /// A model architecture does not match the geometry bound into the plan.
    ModelGeometryMismatch {
        /// Differing axis.
        axis: &'static str,
        /// Planned value.
        expected: u32,
        /// Model value.
        actual: usize,
    },
    /// Planned tied/untied embedding policy differs from the model.
    ModelTyingMismatch {
        /// Planned tying state.
        expected_tied: bool,
        /// Model tying state.
        actual_tied: bool,
    },
    /// The existing training-model adapter rejected application.
    TrainingAdapter(TrainingAdapterError),
}

impl core::fmt::Display for GrowthPlanError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidSourceModelId => {
                write!(formatter, "growth source-model digest cannot be all zero")
            }
            Self::SourceModelMismatch { expected, actual } => write!(
                formatter,
                "growth source-model digest mismatch: expected {:02x?}, got {:02x?}",
                expected.as_bytes(),
                actual.as_bytes()
            ),
            Self::ZeroAxis(axis) => write!(formatter, "growth geometry axis {axis} is zero"),
            Self::AxisOutOfRange { axis, value } => {
                write!(formatter, "growth geometry axis {axis}={value} exceeds u32")
            }
            Self::InvalidPlaneCount { projection, planes } => write!(
                formatter,
                "projection {projection} has {planes} planes; SALT V2 requires 1..={MAX_ADDITIVE_PLANES}"
            ),
            Self::CoefficientCountOverflow(context) => {
                write!(formatter, "{context} exceed u64")
            }
            Self::PlaneLedgerShapeMismatch {
                projection,
                expected_weights,
                actual_weights,
            } => write!(
                formatter,
                "projection {projection} ledger covers {actual_weights} scalar weights; geometry requires {expected_weights}"
            ),
            Self::IntermediateNarrowing {
                existing,
                requested,
            } => write!(
                formatter,
                "intermediate growth cannot narrow existing width {existing} to {requested}"
            ),
            Self::ZeroTarget => write!(formatter, "growth coefficient target is zero"),
            Self::WholeHeadAndHiddenRequired { target } => write!(
                formatter,
                "coefficient target {target} requires staged whole-head and hidden-width growth"
            ),
            Self::InvalidStage2Target {
                intermediate,
                final_target,
            } => write!(
                formatter,
                "stage-2 target {final_target} must exceed stage-1 floor {intermediate} and be at least {WHOLE_HEAD_AND_HIDDEN_COEFFICIENT_THRESHOLD}"
            ),
            Self::WidthOutOfRange { axis, value } => {
                write!(
                    formatter,
                    "growth width {axis}={value} exceeds u32 or usize"
                )
            }
            Self::PlanReceiptTooLarge { width, maximum } => write!(
                formatter,
                "growth width {width} exceeds eager Net2Wider receipt limit {maximum}"
            ),
            Self::Net2Wider(error) => write!(formatter, "Net2Wider receipt: {error}"),
            Self::ReceiptMismatch => write!(formatter, "Net2Wider receipt does not match plan"),
            Self::ModelGeometryOverflow(axis) => {
                write!(formatter, "training model geometry {axis} overflows usize")
            }
            Self::ModelGeometryMismatch {
                axis,
                expected,
                actual,
            } => write!(
                formatter,
                "training model axis {axis}={actual} does not match planned {expected}"
            ),
            Self::ModelTyingMismatch {
                expected_tied,
                actual_tied,
            } => write!(
                formatter,
                "training model tied-head state {actual_tied} does not match planned {expected_tied}"
            ),
            Self::TrainingAdapter(error) => write!(formatter, "training adapter: {error}"),
        }
    }
}

impl std::error::Error for GrowthPlanError {}

fn validate_fixed_planes(projection: &'static str, planes: u8) -> Result<(), GrowthPlanError> {
    if !(1..=MAX_ADDITIVE_PLANES).contains(&planes) {
        return Err(GrowthPlanError::InvalidPlaneCount { projection, planes });
    }
    Ok(())
}

fn checked_axis(axis: &'static str, value: usize) -> Result<u32, GrowthPlanError> {
    if value == 0 {
        return Err(GrowthPlanError::ZeroAxis(axis));
    }
    u32::try_from(value).map_err(|_| GrowthPlanError::AxisOutOfRange { axis, value })
}

fn count_to_u64(context: &'static str, value: u128) -> Result<u64, GrowthPlanError> {
    u64::try_from(value).map_err(|_| GrowthPlanError::CoefficientCountOverflow(context))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_model_identity_rejects_the_unbound_zero_digest() {
        assert_eq!(
            GrowthSourceModelId::new([0; 32]),
            Err(GrowthPlanError::InvalidSourceModelId)
        );
    }

    #[test]
    fn growth_receipt_cannot_replay_against_a_different_source_checkpoint() {
        let source_a = source_model_id(0x11);
        let source_b = source_model_id(0x22);
        let target = GrowthTarget::intermediate_at_least(120).unwrap();
        let plan_a = tiny_geometry().plan(source_a, target, 17).unwrap();
        let plan_b = tiny_geometry().plan(source_b, target, 17).unwrap();
        let receipt = IntermediateGrowthReceipt {
            source_model_id: source_a,
            net2wider: plan_a.expected_net2wider_plan().unwrap(),
        };

        assert_eq!(plan_a.source_model_id(), source_a);
        assert_eq!(receipt.source_model_id(), source_a);
        assert_eq!(
            plan_a.validate_source_model_id(source_b),
            Err(GrowthPlanError::SourceModelMismatch {
                expected: source_a,
                actual: source_b,
            })
        );
        assert_eq!(
            plan_b.validate_receipt(&receipt),
            Err(GrowthPlanError::SourceModelMismatch {
                expected: source_b,
                actual: source_a,
            })
        );
    }

    #[test]
    fn identity_target_keeps_the_existing_intermediate_width() {
        let geometry = tiny_geometry();
        let target = GrowthTarget::intermediate_at_least(60).expect("valid target");

        let plan = geometry
            .plan(source_model_id(0x11), target, 17)
            .expect("identity plan");

        assert_eq!(plan.old_width, 3);
        assert_eq!(plan.new_width, 3);
        assert_eq!(plan.fixed_coefficient_count, 24);
        assert_eq!(plan.base_core_coefficient_count, 60);
        assert_eq!(plan.resulting_core_coefficient_count, 60);
        assert_eq!(plan.seed, 17);
        assert_eq!(plan.stage2_requirement, Stage2Requirement::NotRequested);
        assert_eq!(
            plan.expected_net2wider_plan().unwrap().algorithm(),
            tritium_train::grow::NET2WIDER_ALGORITHM_V1
        );
    }

    #[test]
    fn qwen_8b_like_geometry_uses_exact_additive_coefficient_arithmetic() {
        let geometry = qwen_8b_like_geometry();

        assert_eq!(geometry.fixed_coefficient_count().unwrap(), 1_509_949_440);
        assert_eq!(
            geometry.core_coefficient_count(12_288).unwrap(),
            6_945_767_424
        );

        let plan = geometry
            .plan(
                source_model_id(0x11),
                GrowthTarget::intermediate_at_least(32_000_000_000).unwrap(),
                0x0028,
            )
            .unwrap();
        assert_eq!(plan.new_width, 68_925);
        assert_eq!(plan.resulting_core_coefficient_count, 32_000_163_840);
    }

    #[test]
    fn planned_width_is_the_minimum_width_that_meets_the_floor() {
        let geometry = tiny_geometry();
        let plan = geometry
            .plan(
                source_model_id(0x11),
                GrowthTarget::intermediate_at_least(61).unwrap(),
                9,
            )
            .unwrap();

        assert_eq!(plan.new_width, 4);
        assert_eq!(plan.resulting_core_coefficient_count, 72);
        assert_eq!(
            geometry.core_coefficient_count(plan.new_width - 1).unwrap(),
            60
        );
    }

    #[test]
    fn fifty_billion_endpoint_requires_whole_head_and_hidden_growth() {
        assert_eq!(
            GrowthTarget::intermediate_at_least(50_000_000_000),
            Err(GrowthPlanError::WholeHeadAndHiddenRequired {
                target: 50_000_000_000,
            })
        );
        assert_eq!(
            GrowthTarget::intermediate_at_least(u64::MAX),
            Err(GrowthPlanError::WholeHeadAndHiddenRequired { target: u64::MAX })
        );
        assert_eq!(
            GrowthTarget::staged(32_000_000_000, 49_000_000_000),
            Err(GrowthPlanError::InvalidStage2Target {
                intermediate: 32_000_000_000,
                final_target: 49_000_000_000,
            })
        );

        let rounded = qwen_8b_like_geometry()
            .plan(
                source_model_id(0x11),
                GrowthTarget::intermediate_at_least(
                    WHOLE_HEAD_AND_HIDDEN_COEFFICIENT_THRESHOLD - 1,
                )
                .expect("floor is below the explicit threshold"),
                0,
            )
            .expect_err("rounded FFN-only result crosses the whole-head threshold");
        assert!(matches!(
            rounded,
            GrowthPlanError::WholeHeadAndHiddenRequired { target }
                if target >= WHOLE_HEAD_AND_HIDDEN_COEFFICIENT_THRESHOLD
        ));
    }

    #[test]
    fn staged_target_keeps_whole_head_and_hidden_growth_explicit() {
        let target = GrowthTarget::staged(32_000_000_000, 50_000_000_000).unwrap();
        let plan = qwen_8b_like_geometry()
            .plan(source_model_id(0x11), target, 11)
            .unwrap();

        assert_eq!(plan.new_width, 68_925);
        assert_eq!(
            plan.stage2_requirement,
            Stage2Requirement::WholeHeadAndHidden {
                final_coefficient_floor: 50_000_000_000
            }
        );
        assert!(plan.resulting_core_coefficient_count < 50_000_000_000);
    }

    #[test]
    fn mixed_plane_ledger_counts_every_projection_exactly() {
        let ledger = ProjectionCoefficientLedger::new(
            PlaneWeightHistogram::new(1, 1, 2).unwrap(),
            PlaneWeightHistogram::new(1, 1, 0).unwrap(),
            PlaneWeightHistogram::new(0, 2, 0).unwrap(),
            PlaneWeightHistogram::new(4, 0, 0).unwrap(),
            PlaneWeightHistogram::new(3, 3, 0).unwrap(),
            PlaneWeightHistogram::new(2, 2, 2).unwrap(),
            PlaneWeightHistogram::new(0, 0, 6).unwrap(),
        )
        .unwrap();
        let growth_planes = ProjectionPlaneCounts::new(2, 1, 3, 2, 1, 2, 3).unwrap();
        let geometry = ProjectionGeometry::new_with_ledger(
            1,
            2,
            2,
            1,
            3,
            10,
            ledger,
            growth_planes,
            FixedEmbeddingPolicy::PreservedDense { tied_lm_head: true },
        )
        .unwrap();

        assert_eq!(geometry.coefficient_ledger(), ledger);
        assert_eq!(geometry.fixed_coefficient_count().unwrap(), 20);
        assert_eq!(geometry.core_coefficient_count(3).unwrap(), 59);
        assert_eq!(geometry.core_coefficient_count(4).unwrap(), 71);
        assert_eq!(
            geometry.core_coefficient_count(2),
            Err(GrowthPlanError::IntermediateNarrowing {
                existing: 3,
                requested: 2,
            })
        );
    }

    #[test]
    fn mixed_plane_ledger_must_cover_each_projection_geometry_exactly() {
        let too_short_query = ProjectionCoefficientLedger::new(
            PlaneWeightHistogram::new(3, 0, 0).unwrap(),
            PlaneWeightHistogram::new(2, 0, 0).unwrap(),
            PlaneWeightHistogram::new(2, 0, 0).unwrap(),
            PlaneWeightHistogram::new(4, 0, 0).unwrap(),
            PlaneWeightHistogram::new(6, 0, 0).unwrap(),
            PlaneWeightHistogram::new(6, 0, 0).unwrap(),
            PlaneWeightHistogram::new(6, 0, 0).unwrap(),
        )
        .unwrap();
        let error = ProjectionGeometry::new_with_ledger(
            1,
            2,
            2,
            1,
            3,
            10,
            too_short_query,
            ProjectionPlaneCounts::new(1, 1, 1, 1, 1, 1, 1).unwrap(),
            FixedEmbeddingPolicy::PreservedDense { tied_lm_head: true },
        )
        .unwrap_err();

        assert_eq!(
            error,
            GrowthPlanError::PlaneLedgerShapeMismatch {
                projection: "query",
                expected_weights: 4,
                actual_weights: 3,
            }
        );
    }

    #[test]
    fn coefficient_and_width_overflow_fail_closed() {
        let planes = ProjectionPlaneCounts::new(3, 3, 3, 3, 3, 3, 3).unwrap();
        let count_error = ProjectionGeometry::new(
            u32::MAX as usize,
            u32::MAX as usize,
            u32::MAX as usize,
            u32::MAX as usize,
            1,
            1,
            planes,
            FixedEmbeddingPolicy::PreservedDense { tied_lm_head: true },
        )
        .unwrap_err();
        assert!(matches!(
            count_error,
            GrowthPlanError::CoefficientCountOverflow(_)
        ));

        let narrow_geometry = ProjectionGeometry::new(
            1,
            1,
            1,
            1,
            1,
            1,
            ProjectionPlaneCounts::new(1, 1, 1, 1, 1, 1, 1).unwrap(),
            FixedEmbeddingPolicy::PreservedDense { tied_lm_head: true },
        )
        .unwrap();
        let width_error = narrow_geometry
            .plan(
                source_model_id(0x11),
                GrowthTarget::intermediate_at_least(49_999_999_999).unwrap(),
                0,
            )
            .unwrap_err();
        assert!(matches!(
            width_error,
            GrowthPlanError::WidthOutOfRange {
                axis: "planned_intermediate_width",
                ..
            }
        ));

        let receipt_error = narrow_geometry
            .plan(
                source_model_id(0x11),
                GrowthTarget::intermediate_at_least(12_000_000_000).unwrap(),
                0,
            )
            .unwrap_err();
        assert!(matches!(
            receipt_error,
            GrowthPlanError::PlanReceiptTooLarge {
                width,
                maximum: MAX_STAGE1_RECEIPT_WIDTH,
            } if width > MAX_STAGE1_RECEIPT_WIDTH
        ));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn geometry_axis_must_fit_the_portable_u32_receipt() {
        let error = ProjectionGeometry::new(
            u32::MAX as usize + 1,
            1,
            1,
            1,
            1,
            1,
            ProjectionPlaneCounts::new(1, 1, 1, 1, 1, 1, 1).unwrap(),
            FixedEmbeddingPolicy::PreservedDense { tied_lm_head: true },
        )
        .unwrap_err();

        assert_eq!(
            error,
            GrowthPlanError::AxisOutOfRange {
                axis: "layers",
                value: u32::MAX as usize + 1
            }
        );
    }

    #[test]
    fn deterministic_receipt_is_bound_to_widths_and_seed() {
        let plan = tiny_geometry()
            .plan(
                source_model_id(0x11),
                GrowthTarget::intermediate_at_least(120).unwrap(),
                17,
            )
            .unwrap();
        let first = plan.expected_net2wider_plan().unwrap();
        let replay = plan.expected_net2wider_plan().unwrap();

        assert_eq!(first, replay);
        let first = IntermediateGrowthReceipt {
            source_model_id: plan.source_model_id(),
            net2wider: first,
        };
        plan.validate_receipt(&first).unwrap();

        let wrong_seed = tritium_train::Net2WiderPlan::seeded(3, 8, 18).unwrap();
        let wrong_seed = IntermediateGrowthReceipt {
            source_model_id: plan.source_model_id(),
            net2wider: wrong_seed,
        };
        assert_eq!(
            plan.validate_receipt(&wrong_seed),
            Err(GrowthPlanError::ReceiptMismatch)
        );
    }

    fn tiny_geometry() -> ProjectionGeometry {
        ProjectionGeometry::new(
            1,
            2,
            2,
            1,
            3,
            10,
            ProjectionPlaneCounts::new(2, 1, 3, 2, 1, 2, 3).expect("valid planes"),
            FixedEmbeddingPolicy::PreservedDense { tied_lm_head: true },
        )
        .expect("valid geometry")
    }

    fn source_model_id(tag: u8) -> GrowthSourceModelId {
        GrowthSourceModelId::new([tag; 32]).expect("nonzero source id")
    }

    fn qwen_8b_like_geometry() -> ProjectionGeometry {
        ProjectionGeometry::new(
            36,
            4_096,
            4_096,
            1_024,
            12_288,
            151_936,
            ProjectionPlaneCounts::new(1, 1, 1, 1, 1, 1, 1).unwrap(),
            FixedEmbeddingPolicy::PreservedDense { tied_lm_head: true },
        )
        .unwrap()
    }
}
