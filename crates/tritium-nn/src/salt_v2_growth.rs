//! Checked SALT V2 additive-coefficient growth planning.

use crate::{
    ArchSpec, ModelConfig, ModelRunner,
    training::{TiedSwiGluTrainingModel, TrainingAdapterError, semantic_training_model_digest},
};
use tritium_spec::{
    BackendError, DeviceBuffer, DeviceCaps, GemmShape, MpGemm, TernaryBackend, TernaryFormat,
};
use tritium_train::{GrowError, Net2WiderPlan};

const APPLIED_GROWTH_RECEIPT_MAGIC: [u8; 4] = *b"TGR1";
const APPLIED_GROWTH_RECEIPT_VERSION: u16 = 1;
const APPLIED_GROWTH_RECEIPT_DIGEST_CONTEXT: &str =
    "tritium.salt-v2.applied-intermediate-growth-receipt.v1";
const DENSE_GROWTH_ORACLE_DIGEST_CONTEXT: &str = "tritium.salt-v2.dense-growth-oracle-logits.v1";
const DENSE_GROWTH_ORACLE_MAX_TOKENS: usize = 4;
const DENSE_GROWTH_ORACLE_MAX_LOGITS: u64 = 16_000_000;

/// Versioned dense-logit oracle executed by checked G1 growth application.
pub const DENSE_GROWTH_ORACLE_ALGORITHM_V1: &str = "dense-logits.fixed-tokens.v1";

/// Maximum absolute logit delta admitted by the deterministic G1 dense oracle.
pub const DENSE_GROWTH_ORACLE_TOLERANCE: f32 = 2.0e-6;

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

    /// Validate the descriptors and compute the semantic identity of a training checkpoint.
    pub fn from_training_model(
        config: &ModelConfig,
        spec: &ArchSpec,
        model: &TiedSwiGluTrainingModel,
    ) -> Result<Self, GrowthPlanError> {
        validate_source_descriptors(config, spec, model)?;
        Self::new(semantic_training_model_digest(config, spec, model))
    }

    /// Canonical 32-byte semantic digest.
    #[must_use]
    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

/// Semantic identity of the exact widened checkpoint produced by checked growth.
///
/// This is deliberately distinct from [`GrowthSourceModelId`], preventing a
/// source digest from being accidentally installed in the result field. The
/// all-zero value is reserved and rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GrowthResultModelId([u8; 32]);

impl GrowthResultModelId {
    /// Bind a nonzero widened-model semantic digest.
    pub fn new(digest: [u8; 32]) -> Result<Self, GrowthPlanError> {
        if digest == [0; 32] {
            return Err(GrowthPlanError::InvalidResultModelId);
        }
        Ok(Self(digest))
    }

    /// Validate descriptors and identify a widened training checkpoint.
    pub fn from_training_model(
        config: &ModelConfig,
        spec: &ArchSpec,
        model: &TiedSwiGluTrainingModel,
    ) -> Result<Self, GrowthPlanError> {
        validate_result_descriptors(config, spec, model)?;
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

/// Arithmetic preflight for the current non-streaming checked-growth path.
///
/// The estimate covers retained fp32 model/logit payloads, not allocator
/// metadata, model-runner activations, or backend workspaces. The current oracle
/// reconstructs owned dense weights while transactional widening retains a
/// rollback source, so the reported blocker is conservatively modeled as one
/// source payload plus two widened payloads and two vocabulary-logit vectors.
/// This is intentionally exposed rather than pretending the oracle is streaming.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GrowthTrackedFp32PayloadEstimate {
    source_model_bytes: u64,
    widened_model_bytes: u64,
    oracle_logits_bytes: u64,
    tracked_peak_bytes: u64,
}

impl GrowthTrackedFp32PayloadEstimate {
    /// Complete source training-model fp32 payload used by the estimate.
    #[must_use]
    pub const fn source_model_bytes(self) -> u64 {
        self.source_model_bytes
    }

    /// Complete widened training-model fp32 payload used by the estimate.
    #[must_use]
    pub const fn widened_model_bytes(self) -> u64 {
        self.widened_model_bytes
    }

    /// Two final-token vocabulary-logit vectors retained during comparison.
    #[must_use]
    pub const fn oracle_logits_bytes(self) -> u64 {
        self.oracle_logits_bytes
    }

    /// Conservative retained-payload estimate: `source + 2*widened + logits`.
    #[must_use]
    pub const fn tracked_peak_bytes(self) -> u64 {
        self.tracked_peak_bytes
    }
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

    /// Preflight the retained fp32 payload required by the current owned oracle.
    ///
    /// This reports the explicit model-scale blocker before any oracle clone or
    /// model mutation. It is not a memory-admission guarantee because runtime
    /// activations, backend workspaces, and allocator overhead are excluded.
    pub fn tracked_fp32_payload_estimate(
        &self,
        model: &TiedSwiGluTrainingModel,
    ) -> Result<GrowthTrackedFp32PayloadEstimate, GrowthPlanError> {
        self.validate_model_geometry(model)?;
        growth_tracked_fp32_payload_estimate(model, self.new_width)
    }

    /// Check whether the modeled retained fp32 payload fits a caller-supplied
    /// tracked-payload limit.
    ///
    /// Callers must reserve additional headroom for activations, backend
    /// workspaces, allocator metadata, and receipt/plan storage, which are not
    /// represented by [`GrowthTrackedFp32PayloadEstimate`].
    pub fn preflight_tracked_fp32_payload_limit(
        &self,
        model: &TiedSwiGluTrainingModel,
        maximum_bytes: u64,
    ) -> Result<GrowthTrackedFp32PayloadEstimate, GrowthPlanError> {
        let estimate = self.tracked_fp32_payload_estimate(model)?;
        if estimate.tracked_peak_bytes > maximum_bytes {
            return Err(GrowthPlanError::TrackedFp32PayloadLimitExceeded {
                required: estimate.tracked_peak_bytes,
                maximum: maximum_bytes,
            });
        }
        Ok(estimate)
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
        receipt: &AppliedIntermediateGrowthReceipt,
    ) -> Result<(), GrowthPlanError> {
        self.validate_source_model_id(receipt.source_model_id)?;
        if receipt.target != self.target
            || receipt.resulting_core_coefficient_count != self.resulting_core_coefficient_count
            || receipt.seed != self.seed
            || receipt.old_width != self.old_width
            || receipt.new_width != self.new_width
        {
            return Err(GrowthPlanError::ReceiptMismatch);
        }
        receipt.validate_mapping(&self.expected_net2wider_plan()?)?;
        receipt.function_preservation.validate()?;
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
    ///
    /// # Model-scale blocker
    /// The current dense oracle is owned rather than borrowed/streaming. Its
    /// retained fp32 payload can reach `source + 2*widened + logits`, before
    /// activations and allocator overhead. Production callers should gate this
    /// path with [`Self::preflight_tracked_fp32_payload_limit`]; this method
    /// performs only the overflow-safe estimate and does not infer available
    /// system memory.
    pub fn apply(
        &self,
        config: &ModelConfig,
        spec: &ArchSpec,
        model: &mut TiedSwiGluTrainingModel,
    ) -> Result<AppliedIntermediateGrowthReceipt, GrowthPlanError> {
        self.apply_with_validation_hooks(config, spec, model, || Ok(()), |_| Ok(()))
    }

    /// Apply only when the current non-streaming retained-payload estimate fits
    /// `maximum_bytes`.
    ///
    /// Budget rejection occurs before semantic hashing, dense oracle cloning,
    /// plan allocation, or model mutation. The limit still needs external
    /// headroom for the unmodeled runtime costs documented by
    /// [`Self::preflight_tracked_fp32_payload_limit`].
    pub fn apply_with_tracked_fp32_payload_limit(
        &self,
        config: &ModelConfig,
        spec: &ArchSpec,
        model: &mut TiedSwiGluTrainingModel,
        maximum_bytes: u64,
    ) -> Result<AppliedIntermediateGrowthReceipt, GrowthPlanError> {
        self.preflight_tracked_fp32_payload_limit(model, maximum_bytes)?;
        self.apply_with_validation_hooks(config, spec, model, || Ok(()), |_| Ok(()))
    }

    fn apply_with_validation_hooks<OracleHook, ReceiptHook>(
        &self,
        config: &ModelConfig,
        spec: &ArchSpec,
        model: &mut TiedSwiGluTrainingModel,
        mut before_grown_oracle: OracleHook,
        mut before_receipt_validation: ReceiptHook,
    ) -> Result<AppliedIntermediateGrowthReceipt, GrowthPlanError>
    where
        OracleHook: FnMut() -> Result<(), GrowthPlanError>,
        ReceiptHook: FnMut(&mut AppliedIntermediateGrowthReceipt) -> Result<(), GrowthPlanError>,
    {
        // Cheap structural checks precede the full-checkpoint semantic hash and
        // dense oracle, which are intentionally expensive at model scale.
        self.validate_model_geometry(model)?;
        let _tracked_payload_estimate = self.tracked_fp32_payload_estimate(model)?;
        validate_source_descriptors(config, spec, model)?;
        let source_model_id =
            GrowthSourceModelId::new(semantic_training_model_digest(config, spec, model))?;
        self.validate_source_model_id(source_model_id)?;

        let oracle_vocabulary = u32::try_from(model.architecture().vocab)
            .map_err(|_| GrowthPlanError::InvalidOracleEvidence("vocabulary exceeds u32"))?;
        let oracle_context_length = u32::try_from(model.architecture().n_ctx)
            .map_err(|_| GrowthPlanError::InvalidOracleEvidence("context length exceeds u32"))?;
        let oracle_tokens = dense_growth_oracle_tokens(oracle_vocabulary, oracle_context_length)?;
        let source_logits = dense_growth_oracle_logits(config, model, &oracle_tokens)?;
        let widening = model
            .begin_intermediate_widening(
                usize::try_from(self.new_width).map_err(|_| GrowthPlanError::WidthOutOfRange {
                    axis: "new_intermediate_width",
                    value: u128::from(self.new_width),
                })?,
                self.seed,
            )
            .map_err(GrowthPlanError::TrainingAdapter)?;

        // Everything after provisional installation can fail. The transaction
        // owns the complete source MLP plane and restores it on every early
        // return, including oracle and receipt validation failures.
        before_grown_oracle()?;
        let mut grown_config = config.clone();
        grown_config.n_ff = self.new_width;
        let grown_logits =
            dense_growth_oracle_logits(&grown_config, widening.model(), &oracle_tokens)?;
        let function_preservation = GrowthFunctionPreservationEvidence::from_logits(
            oracle_vocabulary,
            oracle_context_length,
            oracle_tokens,
            &source_logits,
            &grown_logits,
        )?;
        let result_model_id =
            GrowthResultModelId::from_training_model(&grown_config, spec, widening.model())?;
        let mut receipt = AppliedIntermediateGrowthReceipt::from_applied_plan(
            source_model_id,
            result_model_id,
            widening.plan(),
            self.target,
            self.resulting_core_coefficient_count,
            self.seed,
            function_preservation,
        )?;
        before_receipt_validation(&mut receipt)?;
        receipt.validate_result_model(&grown_config, spec, widening.model())?;
        self.validate_receipt(&receipt)?;
        widening.commit();
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

/// Content identity of a canonical applied-growth receipt.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GrowthReceiptDigest([u8; 32]);

impl GrowthReceiptDigest {
    /// Rebuild a digest stored by a durable evidence index.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Canonical digest bytes.
    #[must_use]
    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

/// Measured dense-logit preservation evidence produced during checked growth.
///
/// The fields are private and no public constructor exists. The only live
/// producer is [`IntermediateGrowthPlan::apply`], which executes the oracle; a
/// durable instance can only be recovered through strict canonical decoding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GrowthFunctionPreservationEvidence {
    vocabulary: u32,
    context_length: u32,
    tokens: Vec<u32>,
    token_count: u32,
    logit_count: u64,
    tolerance_bits: u32,
    max_absolute_error_bits: u32,
    worst_logit_index: u64,
    source_worst_bits: u32,
    grown_worst_bits: u32,
    source_logits_digest: [u8; 32],
    grown_logits_digest: [u8; 32],
}

impl GrowthFunctionPreservationEvidence {
    /// Versioned oracle algorithm.
    #[must_use]
    pub const fn algorithm(&self) -> &'static str {
        DENSE_GROWTH_ORACLE_ALGORITHM_V1
    }

    /// Vocabulary bound into the fixed-token oracle descriptor.
    #[must_use]
    pub const fn vocabulary(&self) -> u32 {
        self.vocabulary
    }

    /// Context length bound into the fixed-token oracle descriptor.
    #[must_use]
    pub const fn context_length(&self) -> u32 {
        self.context_length
    }

    /// Deterministic token sequence evaluated at positions `0..tokens.len()`.
    #[must_use]
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Number of logits compared.
    #[must_use]
    pub const fn logit_count(&self) -> u64 {
        self.logit_count
    }

    /// Frozen maximum-absolute-error threshold.
    #[must_use]
    pub const fn tolerance(&self) -> f32 {
        f32::from_bits(self.tolerance_bits)
    }

    /// Largest measured absolute logit delta.
    #[must_use]
    pub const fn max_absolute_error(&self) -> f32 {
        f32::from_bits(self.max_absolute_error_bits)
    }

    /// Index of one logit witnessing the recorded maximum absolute error.
    #[must_use]
    pub const fn worst_logit_index(&self) -> u64 {
        self.worst_logit_index
    }

    /// Source logit at [`Self::worst_logit_index`].
    #[must_use]
    pub const fn source_worst_logit(&self) -> f32 {
        f32::from_bits(self.source_worst_bits)
    }

    /// Grown logit at [`Self::worst_logit_index`].
    #[must_use]
    pub const fn grown_worst_logit(&self) -> f32 {
        f32::from_bits(self.grown_worst_bits)
    }

    /// Digest of the source-model oracle logits and exact oracle tokens.
    #[must_use]
    pub const fn source_logits_digest(&self) -> [u8; 32] {
        self.source_logits_digest
    }

    /// Digest of the grown-model oracle logits and exact oracle tokens.
    #[must_use]
    pub const fn grown_logits_digest(&self) -> [u8; 32] {
        self.grown_logits_digest
    }

    fn from_logits(
        vocabulary: u32,
        context_length: u32,
        tokens: Vec<u32>,
        source_logits: &[f32],
        grown_logits: &[f32],
    ) -> Result<Self, GrowthPlanError> {
        if source_logits.len() != grown_logits.len() || source_logits.is_empty() {
            return Err(GrowthPlanError::OracleLogitLengthMismatch {
                source: source_logits.len(),
                grown: grown_logits.len(),
            });
        }
        let token_count = u32::try_from(tokens.len())
            .map_err(|_| GrowthPlanError::InvalidOracleEvidence("token count exceeds u32"))?;
        let logit_count = u64::try_from(source_logits.len())
            .map_err(|_| GrowthPlanError::InvalidOracleEvidence("logit count exceeds u64"))?;
        let mut maximum = 0.0_f32;
        let mut worst_index = 0_usize;
        for (index, (&source, &grown)) in source_logits.iter().zip(grown_logits).enumerate() {
            if !source.is_finite() {
                return Err(GrowthPlanError::NonFiniteOracleLogit {
                    side: "source",
                    index,
                    value_bits: source.to_bits(),
                });
            }
            if !grown.is_finite() {
                return Err(GrowthPlanError::NonFiniteOracleLogit {
                    side: "grown",
                    index,
                    value_bits: grown.to_bits(),
                });
            }
            let delta = (source - grown).abs();
            if delta > maximum {
                maximum = delta;
                worst_index = index;
            }
        }
        if maximum > DENSE_GROWTH_ORACLE_TOLERANCE {
            return Err(GrowthPlanError::FunctionPreservationFailed {
                maximum_bits: maximum.to_bits(),
                tolerance_bits: DENSE_GROWTH_ORACLE_TOLERANCE.to_bits(),
            });
        }
        let source_logits_digest = dense_oracle_logits_digest(
            "source",
            vocabulary,
            context_length,
            &tokens,
            source_logits,
        );
        let grown_logits_digest =
            dense_oracle_logits_digest("grown", vocabulary, context_length, &tokens, grown_logits);
        let evidence = Self {
            vocabulary,
            context_length,
            tokens,
            token_count,
            logit_count,
            tolerance_bits: DENSE_GROWTH_ORACLE_TOLERANCE.to_bits(),
            max_absolute_error_bits: maximum.to_bits(),
            worst_logit_index: u64::try_from(worst_index).map_err(|_| {
                GrowthPlanError::InvalidOracleEvidence("worst logit index exceeds u64")
            })?,
            source_worst_bits: source_logits[worst_index].to_bits(),
            grown_worst_bits: grown_logits[worst_index].to_bits(),
            source_logits_digest,
            grown_logits_digest,
        };
        evidence.validate()?;
        Ok(evidence)
    }

    fn validate(&self) -> Result<(), GrowthPlanError> {
        if self.vocabulary == 0
            || u64::from(self.vocabulary) > DENSE_GROWTH_ORACLE_MAX_LOGITS
            || self.context_length == 0
        {
            return Err(GrowthPlanError::InvalidOracleEvidence(
                "oracle vocabulary or context length is outside the supported domain",
            ));
        }
        let token_count = usize::try_from(self.token_count)
            .map_err(|_| GrowthPlanError::InvalidOracleEvidence("token count exceeds usize"))?;
        if token_count == 0
            || token_count > DENSE_GROWTH_ORACLE_MAX_TOKENS
            || token_count != self.tokens.len()
        {
            return Err(GrowthPlanError::InvalidOracleEvidence(
                "oracle token count is not canonical",
            ));
        }
        if self.tokens.iter().any(|&token| token >= self.vocabulary) {
            return Err(GrowthPlanError::InvalidOracleEvidence(
                "oracle token is outside the vocabulary",
            ));
        }
        let expected_tokens = dense_growth_oracle_tokens(self.vocabulary, self.context_length)?;
        if self.tokens != expected_tokens {
            return Err(GrowthPlanError::InvalidOracleEvidence(
                "oracle tokens do not match the fixed descriptor",
            ));
        }
        if self.logit_count != u64::from(self.vocabulary) {
            return Err(GrowthPlanError::InvalidOracleEvidence(
                "oracle logit count does not match the vocabulary",
            ));
        }
        if self.source_logits_digest == [0; 32] || self.grown_logits_digest == [0; 32] {
            return Err(GrowthPlanError::InvalidOracleEvidence(
                "oracle logit digests cannot be all zero",
            ));
        }
        if self.tolerance_bits != DENSE_GROWTH_ORACLE_TOLERANCE.to_bits() {
            return Err(GrowthPlanError::InvalidOracleEvidence(
                "oracle tolerance does not match the frozen algorithm",
            ));
        }
        let maximum = self.max_absolute_error();
        if !maximum.is_finite() || maximum < 0.0 || maximum > self.tolerance() {
            return Err(GrowthPlanError::InvalidOracleEvidence(
                "oracle maximum error is invalid or exceeds tolerance",
            ));
        }
        if self.worst_logit_index >= self.logit_count {
            return Err(GrowthPlanError::InvalidOracleEvidence(
                "oracle worst-logit index is outside the output",
            ));
        }
        let source_worst = f32::from_bits(self.source_worst_bits);
        let grown_worst = f32::from_bits(self.grown_worst_bits);
        if !source_worst.is_finite()
            || !grown_worst.is_finite()
            || (source_worst - grown_worst).abs().to_bits() != self.max_absolute_error_bits
        {
            return Err(GrowthPlanError::InvalidOracleEvidence(
                "oracle worst-logit witness does not match the maximum error",
            ));
        }
        Ok(())
    }
}

/// Source-bound receipt issued only after a checked intermediate-width transform.
///
/// It binds the complete expected transform, exact coefficient target/result,
/// explicit seed, and the dense-logit preservation evidence executed by
/// [`IntermediateGrowthPlan::apply`]. Planning cannot construct this type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AppliedIntermediateGrowthReceipt {
    source_model_id: GrowthSourceModelId,
    result_model_id: GrowthResultModelId,
    target: GrowthTarget,
    resulting_core_coefficient_count: u64,
    seed: u64,
    old_width: u32,
    new_width: u32,
    source_indices: Vec<u32>,
    replication_counts: Vec<u32>,
    split_denominator_log2: Option<u32>,
    split_numerators: Option<Vec<u32>>,
    function_preservation: GrowthFunctionPreservationEvidence,
}

impl AppliedIntermediateGrowthReceipt {
    /// Semantic checkpoint transformed by this receipt.
    #[must_use]
    pub const fn source_model_id(&self) -> GrowthSourceModelId {
        self.source_model_id
    }

    /// Semantic checkpoint produced by this receipt.
    #[must_use]
    pub const fn result_model_id(&self) -> GrowthResultModelId {
        self.result_model_id
    }

    /// Verify that a materialized widened checkpoint is the result bound here.
    pub fn validate_result_model(
        &self,
        config: &ModelConfig,
        spec: &ArchSpec,
        model: &TiedSwiGluTrainingModel,
    ) -> Result<(), GrowthPlanError> {
        let actual = GrowthResultModelId::from_training_model(config, spec, model)?;
        if actual != self.result_model_id {
            return Err(GrowthPlanError::ResultModelMismatch {
                expected: self.result_model_id,
                actual,
            });
        }
        Ok(())
    }

    /// Exact coefficient target used by the planner.
    #[must_use]
    pub const fn target(&self) -> GrowthTarget {
        self.target
    }

    /// Exact stored coefficient count after applying this transform.
    #[must_use]
    pub const fn resulting_core_coefficient_count(&self) -> u64 {
        self.resulting_core_coefficient_count
    }

    /// Explicit deterministic mapping seed.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Source intermediate width.
    #[must_use]
    pub const fn old_width(&self) -> u32 {
        self.old_width
    }

    /// Applied intermediate width.
    #[must_use]
    pub const fn new_width(&self) -> u32 {
        self.new_width
    }

    /// Versioned Net2Wider algorithm applied to the source model.
    #[must_use]
    pub fn net2wider_algorithm(&self) -> &'static str {
        if self.split_numerators.is_some() {
            tritium_train::grow::NET2WIDER_ALGORITHM_V2
        } else {
            tritium_train::grow::NET2WIDER_ALGORITHM_V1
        }
    }

    /// Source index selected for every widened unit.
    #[must_use]
    pub fn source_indices(&self) -> &[u32] {
        &self.source_indices
    }

    /// Number of widened copies assigned to each source unit.
    #[must_use]
    pub fn replication_counts(&self) -> &[u32] {
        &self.replication_counts
    }

    /// Base-two split denominator for actual growth.
    #[must_use]
    pub const fn split_denominator_log2(&self) -> Option<u32> {
        self.split_denominator_log2
    }

    /// Exact outgoing split numerators for actual growth.
    #[must_use]
    pub fn split_numerators(&self) -> Option<&[u32]> {
        self.split_numerators.as_deref()
    }

    /// Executed dense function-preservation evidence.
    #[must_use]
    pub const fn function_preservation(&self) -> &GrowthFunctionPreservationEvidence {
        &self.function_preservation
    }

    /// Reconstruct the deterministic Net2Wider transform for replay.
    pub fn replay_plan(&self) -> Result<Net2WiderPlan, GrowthPlanError> {
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

    /// Versioned canonical receipt bytes.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, GrowthPlanError> {
        let mut out = Vec::new();
        let encoding_capacity = receipt_encoding_capacity(self)?;
        out.try_reserve_exact(encoding_capacity).map_err(|_| {
            growth_allocation_failed::<u8>("growth receipt encoding", encoding_capacity)
        })?;
        out.extend_from_slice(&APPLIED_GROWTH_RECEIPT_MAGIC);
        out.extend_from_slice(&APPLIED_GROWTH_RECEIPT_VERSION.to_le_bytes());
        out.extend_from_slice(&0_u16.to_le_bytes());
        out.extend_from_slice(&self.source_model_id.as_bytes());
        out.extend_from_slice(&self.result_model_id.as_bytes());
        out.extend_from_slice(&self.target.intermediate_coefficient_floor.to_le_bytes());
        out.extend_from_slice(&stage2_final_floor(self.target).to_le_bytes());
        out.extend_from_slice(&self.resulting_core_coefficient_count.to_le_bytes());
        out.extend_from_slice(&self.seed.to_le_bytes());
        out.extend_from_slice(&self.old_width.to_le_bytes());
        out.extend_from_slice(&self.new_width.to_le_bytes());
        out.push(net2wider_algorithm_tag(self.split_numerators.is_some()));
        out.extend_from_slice(&[0; 3]);
        out.extend_from_slice(&self.new_width.to_le_bytes());
        for &source in &self.source_indices {
            out.extend_from_slice(&source.to_le_bytes());
        }
        out.extend_from_slice(&self.old_width.to_le_bytes());
        for &copies in &self.replication_counts {
            out.extend_from_slice(&copies.to_le_bytes());
        }
        out.extend_from_slice(&self.split_denominator_log2.unwrap_or(0).to_le_bytes());
        out.extend_from_slice(
            &self
                .split_numerators
                .as_ref()
                .map_or(0, |_| self.new_width)
                .to_le_bytes(),
        );
        if let Some(numerators) = &self.split_numerators {
            for &numerator in numerators {
                out.extend_from_slice(&numerator.to_le_bytes());
            }
        }
        out.extend_from_slice(&1_u16.to_le_bytes());
        out.extend_from_slice(&0_u16.to_le_bytes());
        out.extend_from_slice(&self.function_preservation.vocabulary.to_le_bytes());
        out.extend_from_slice(&self.function_preservation.context_length.to_le_bytes());
        out.extend_from_slice(&self.function_preservation.tolerance_bits.to_le_bytes());
        out.extend_from_slice(
            &self
                .function_preservation
                .max_absolute_error_bits
                .to_le_bytes(),
        );
        out.extend_from_slice(&self.function_preservation.worst_logit_index.to_le_bytes());
        out.extend_from_slice(&self.function_preservation.source_worst_bits.to_le_bytes());
        out.extend_from_slice(&self.function_preservation.grown_worst_bits.to_le_bytes());
        out.extend_from_slice(&self.function_preservation.token_count.to_le_bytes());
        for &token in &self.function_preservation.tokens {
            out.extend_from_slice(&token.to_le_bytes());
        }
        out.extend_from_slice(&self.function_preservation.logit_count.to_le_bytes());
        out.extend_from_slice(&self.function_preservation.source_logits_digest);
        out.extend_from_slice(&self.function_preservation.grown_logits_digest);
        Ok(out)
    }

    /// Domain-separated content identity of [`Self::canonical_bytes`].
    pub fn digest(&self) -> Result<GrowthReceiptDigest, GrowthPlanError> {
        Ok(GrowthReceiptDigest(applied_growth_receipt_digest(
            &self.canonical_bytes()?,
        )))
    }

    /// Strictly reopen canonical bytes, including all structural invariants.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, GrowthPlanError> {
        decode_applied_growth_receipt(bytes)
    }

    /// Reopen canonical bytes only when their content digest matches the index.
    pub fn from_canonical_bytes_verified(
        bytes: &[u8],
        expected: GrowthReceiptDigest,
    ) -> Result<Self, GrowthPlanError> {
        let actual = GrowthReceiptDigest(applied_growth_receipt_digest(bytes));
        if actual != expected {
            return Err(GrowthPlanError::ReceiptDigestMismatch { expected, actual });
        }
        Self::from_canonical_bytes(bytes)
    }

    fn from_applied_plan(
        source_model_id: GrowthSourceModelId,
        result_model_id: GrowthResultModelId,
        net2wider: &Net2WiderPlan,
        target: GrowthTarget,
        resulting_core_coefficient_count: u64,
        seed: u64,
        function_preservation: GrowthFunctionPreservationEvidence,
    ) -> Result<Self, GrowthPlanError> {
        let old_width = u32::try_from(net2wider.replication_counts().len())
            .map_err(|_| GrowthPlanError::InvalidReceiptField("old width exceeds u32"))?;
        let new_width = u32::try_from(net2wider.source_indices().len())
            .map_err(|_| GrowthPlanError::InvalidReceiptField("new width exceeds u32"))?;
        let source_indices =
            checked_u32_vector("source index exceeds u32", net2wider.source_indices())?;
        let replication_counts = checked_u32_vector(
            "replication count exceeds u32",
            net2wider.replication_counts(),
        )?;
        let split_numerators = net2wider
            .split_numerators()
            .map(copy_u32_vector)
            .transpose()?;
        let receipt = Self {
            source_model_id,
            result_model_id,
            target,
            resulting_core_coefficient_count,
            seed,
            old_width,
            new_width,
            source_indices,
            replication_counts,
            split_denominator_log2: net2wider.split_denominator_log2(),
            split_numerators,
            function_preservation,
        };
        receipt.validate_invariants()?;
        Ok(receipt)
    }

    fn validate_mapping(&self, expected: &Net2WiderPlan) -> Result<(), GrowthPlanError> {
        let sources_match = expected
            .source_indices()
            .iter()
            .zip(&self.source_indices)
            .all(|(&expected, &actual)| u32::try_from(expected).ok() == Some(actual))
            && expected.source_indices().len() == self.source_indices.len();
        let replications_match = expected
            .replication_counts()
            .iter()
            .zip(&self.replication_counts)
            .all(|(&expected, &actual)| u32::try_from(expected).ok() == Some(actual))
            && expected.replication_counts().len() == self.replication_counts.len();
        if !sources_match
            || !replications_match
            || expected.split_denominator_log2() != self.split_denominator_log2
            || expected.split_numerators() != self.split_numerators.as_deref()
        {
            return Err(GrowthPlanError::ReceiptMismatch);
        }
        Ok(())
    }

    fn validate_invariants(&self) -> Result<(), GrowthPlanError> {
        if self.old_width == 0
            || self.new_width < self.old_width
            || self.new_width > MAX_STAGE1_RECEIPT_WIDTH
        {
            return Err(GrowthPlanError::InvalidReceiptField(
                "growth widths are outside the supported domain",
            ));
        }
        if self.resulting_core_coefficient_count < self.target.intermediate_coefficient_floor()
            || self.resulting_core_coefficient_count >= WHOLE_HEAD_AND_HIDDEN_COEFFICIENT_THRESHOLD
        {
            return Err(GrowthPlanError::InvalidReceiptField(
                "resulting coefficient count does not satisfy stage 1",
            ));
        }
        self.validate_mapping(&self.replay_plan()?)?;
        self.function_preservation.validate()?;
        Ok(())
    }
}

/// Why checked SALT V2 growth planning or application failed.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum GrowthPlanError {
    /// A growth plan was not bound to a semantic source-checkpoint digest.
    InvalidSourceModelId,
    /// An applied receipt was not bound to a semantic result-checkpoint digest.
    InvalidResultModelId,
    /// Supplied model descriptors do not describe the exact training model.
    SourceDescriptorMismatch(&'static str),
    /// Supplied result descriptors do not describe the exact widened model.
    ResultDescriptorMismatch(&'static str),
    /// Application or replay named a different semantic source checkpoint.
    SourceModelMismatch {
        /// Checkpoint digest frozen in the plan.
        expected: GrowthSourceModelId,
        /// Checkpoint digest supplied by the caller or receipt.
        actual: GrowthSourceModelId,
    },
    /// A materialized widened model differs from the result bound into the receipt.
    ResultModelMismatch {
        /// Checkpoint digest frozen in the applied receipt.
        expected: GrowthResultModelId,
        /// Semantic digest recomputed from the supplied widened model.
        actual: GrowthResultModelId,
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
    /// A persisted receipt has malformed or truncated bytes.
    ReceiptEncoding(&'static str),
    /// A persisted receipt uses an unsupported version.
    UnsupportedReceiptVersion(u16),
    /// Receipt bytes decode but are not the unique canonical representation.
    NonCanonicalReceipt,
    /// Reopened bytes do not match their recorded content identity.
    ReceiptDigestMismatch {
        /// Digest recorded by the durable index.
        expected: GrowthReceiptDigest,
        /// Digest of the bytes supplied for reopen.
        actual: GrowthReceiptDigest,
    },
    /// A receipt field violates the versioned replay contract.
    InvalidReceiptField(&'static str),
    /// A bounded growth-path allocation could not be reserved.
    AllocationFailed {
        /// Logical allocation site.
        allocation: &'static str,
        /// Requested payload bytes, excluding allocator metadata.
        requested_bytes: usize,
    },
    /// Arithmetic for the exposed tracked-payload estimate overflowed.
    PayloadEstimateOverflow(&'static str),
    /// The modeled retained fp32 payload exceeds the caller's admission budget.
    TrackedFp32PayloadLimitExceeded {
        /// Modeled bytes required by the current non-streaming implementation.
        required: u64,
        /// Caller-supplied maximum retained-payload budget.
        maximum: u64,
    },
    /// Dense oracle execution failed before evidence could be issued.
    OracleExecution(String),
    /// Source and grown oracle outputs have incompatible lengths.
    OracleLogitLengthMismatch {
        /// Source-model logit count.
        source: usize,
        /// Grown-model logit count.
        grown: usize,
    },
    /// A dense oracle produced a non-finite logit.
    NonFiniteOracleLogit {
        /// Oracle side (`source` or `grown`).
        side: &'static str,
        /// Logit index.
        index: usize,
        /// Exact non-finite fp32 bits.
        value_bits: u32,
    },
    /// The measured dense-logit delta exceeded the frozen tolerance.
    FunctionPreservationFailed {
        /// Maximum absolute-error fp32 bits.
        maximum_bits: u32,
        /// Frozen tolerance fp32 bits.
        tolerance_bits: u32,
    },
    /// Persisted oracle evidence violates its fixed-version invariants.
    InvalidOracleEvidence(&'static str),
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
            Self::InvalidResultModelId => {
                write!(formatter, "growth result-model digest cannot be all zero")
            }
            Self::SourceDescriptorMismatch(axis) => {
                write!(
                    formatter,
                    "growth source descriptor {axis} does not match the model"
                )
            }
            Self::ResultDescriptorMismatch(axis) => {
                write!(
                    formatter,
                    "growth result descriptor {axis} does not match the model"
                )
            }
            Self::SourceModelMismatch { expected, actual } => write!(
                formatter,
                "growth source-model digest mismatch: expected {:02x?}, got {:02x?}",
                expected.as_bytes(),
                actual.as_bytes()
            ),
            Self::ResultModelMismatch { expected, actual } => write!(
                formatter,
                "growth result-model digest mismatch: expected {:02x?}, got {:02x?}",
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
            Self::ReceiptEncoding(reason) => {
                write!(formatter, "growth receipt encoding: {reason}")
            }
            Self::UnsupportedReceiptVersion(version) => {
                write!(formatter, "unsupported growth receipt version {version}")
            }
            Self::NonCanonicalReceipt => {
                write!(formatter, "growth receipt bytes are not canonical")
            }
            Self::ReceiptDigestMismatch { expected, actual } => write!(
                formatter,
                "growth receipt digest mismatch: expected {:02x?}, got {:02x?}",
                expected.as_bytes(),
                actual.as_bytes()
            ),
            Self::InvalidReceiptField(reason) => {
                write!(formatter, "invalid growth receipt field: {reason}")
            }
            Self::AllocationFailed {
                allocation,
                requested_bytes,
            } => write!(
                formatter,
                "growth allocation failed: {allocation} ({requested_bytes} requested bytes)"
            ),
            Self::PayloadEstimateOverflow(component) => {
                write!(
                    formatter,
                    "growth tracked-payload estimate overflows at {component}"
                )
            }
            Self::TrackedFp32PayloadLimitExceeded { required, maximum } => write!(
                formatter,
                "growth tracked fp32 payload {required} bytes exceeds budget {maximum} bytes"
            ),
            Self::OracleExecution(reason) => {
                write!(formatter, "dense growth oracle execution failed: {reason}")
            }
            Self::OracleLogitLengthMismatch { source, grown } => write!(
                formatter,
                "dense growth oracle logit count mismatch: source {source}, grown {grown}"
            ),
            Self::NonFiniteOracleLogit {
                side,
                index,
                value_bits,
            } => write!(
                formatter,
                "dense growth oracle {side} logit {index} is non-finite (bits {value_bits:#010x})"
            ),
            Self::FunctionPreservationFailed {
                maximum_bits,
                tolerance_bits,
            } => write!(
                formatter,
                "dense growth oracle maximum error {} exceeds tolerance {}",
                f32::from_bits(*maximum_bits),
                f32::from_bits(*tolerance_bits)
            ),
            Self::InvalidOracleEvidence(reason) => {
                write!(formatter, "invalid dense growth oracle evidence: {reason}")
            }
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

fn validate_source_descriptors(
    config: &ModelConfig,
    spec: &ArchSpec,
    model: &TiedSwiGluTrainingModel,
) -> Result<(), GrowthPlanError> {
    TiedSwiGluTrainingModel::validate_config(config, spec)
        .map_err(GrowthPlanError::TrainingAdapter)?;
    let arch = model.architecture();
    for (axis, configured, actual) in [
        ("layers", config.n_layers, arch.n_layers),
        ("residual_width", config.n_embd, arch.n_embd),
        ("query_heads", config.n_head, arch.n_head),
        ("key_value_heads", config.n_head_kv, arch.n_head_kv),
        ("head_dimension", config.head_dim, arch.head_dim),
        ("intermediate_width", config.n_ff, arch.n_ff),
        ("context_length", config.n_ctx, arch.n_ctx),
    ] {
        if usize::try_from(configured).ok() != Some(actual) {
            return Err(GrowthPlanError::SourceDescriptorMismatch(axis));
        }
    }
    if config.rope_theta.to_bits() != arch.rope_theta.to_bits() {
        return Err(GrowthPlanError::SourceDescriptorMismatch("rope_theta"));
    }
    if config.rms_eps.to_bits() != arch.rms_eps.to_bits() {
        return Err(GrowthPlanError::SourceDescriptorMismatch("rms_epsilon"));
    }
    if spec.tied_embeddings != model.is_lm_head_tied() {
        return Err(GrowthPlanError::SourceDescriptorMismatch("tied_embeddings"));
    }
    Ok(())
}

fn validate_result_descriptors(
    config: &ModelConfig,
    spec: &ArchSpec,
    model: &TiedSwiGluTrainingModel,
) -> Result<(), GrowthPlanError> {
    match validate_source_descriptors(config, spec, model) {
        Err(GrowthPlanError::SourceDescriptorMismatch(axis)) => {
            Err(GrowthPlanError::ResultDescriptorMismatch(axis))
        }
        result => result,
    }
}

fn dense_growth_oracle_tokens(
    vocabulary: u32,
    context_length: u32,
) -> Result<Vec<u32>, GrowthPlanError> {
    if vocabulary == 0
        || u64::from(vocabulary) > DENSE_GROWTH_ORACLE_MAX_LOGITS
        || context_length == 0
    {
        return Err(GrowthPlanError::InvalidOracleEvidence(
            "oracle vocabulary or context is outside the supported domain",
        ));
    }
    let token_count = usize::try_from(context_length)
        .map_err(|_| GrowthPlanError::InvalidOracleEvidence("context length exceeds usize"))?
        .min(DENSE_GROWTH_ORACLE_MAX_TOKENS);
    let three_quarters = u32::try_from(u64::from(vocabulary) * 3 / 4)
        .map_err(|_| GrowthPlanError::InvalidOracleEvidence("oracle token exceeds u32"))?;
    let candidates = [0, vocabulary - 1, vocabulary / 2, three_quarters];
    let mut tokens = Vec::new();
    tokens
        .try_reserve_exact(token_count)
        .map_err(|_| growth_allocation_failed::<u32>("dense oracle tokens", token_count))?;
    tokens.extend_from_slice(&candidates[..token_count]);
    Ok(tokens)
}

fn dense_growth_oracle_logits(
    config: &ModelConfig,
    model: &TiedSwiGluTrainingModel,
    tokens: &[u32],
) -> Result<Vec<f32>, GrowthPlanError> {
    let weights = model
        .to_dense_weights()
        .map_err(GrowthPlanError::TrainingAdapter)?;
    let mut runner =
        ModelRunner::try_from_weights(config.clone(), weights, Box::new(DenseOracleBackend))
            .map_err(|error| GrowthPlanError::OracleExecution(error.to_string()))?;
    let mut positions = Vec::new();
    positions
        .try_reserve_exact(tokens.len())
        .map_err(|_| growth_allocation_failed::<usize>("dense oracle positions", tokens.len()))?;
    positions.extend(0..tokens.len());
    runner
        .forward(tokens, &positions)
        .map_err(|error| GrowthPlanError::OracleExecution(error.to_string()))
}

fn dense_oracle_logits_digest(
    label: &str,
    vocabulary: u32,
    context_length: u32,
    tokens: &[u32],
    logits: &[f32],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key(DENSE_GROWTH_ORACLE_DIGEST_CONTEXT);
    hasher.update(&(label.len() as u64).to_le_bytes());
    hasher.update(label.as_bytes());
    hasher.update(&vocabulary.to_le_bytes());
    hasher.update(&context_length.to_le_bytes());
    hasher.update(&(tokens.len() as u64).to_le_bytes());
    for token in tokens {
        hasher.update(&token.to_le_bytes());
    }
    hasher.update(&(logits.len() as u64).to_le_bytes());
    for logit in logits {
        hasher.update(&logit.to_bits().to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

fn applied_growth_receipt_digest(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key(APPLIED_GROWTH_RECEIPT_DIGEST_CONTEXT);
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}

fn stage2_final_floor(target: GrowthTarget) -> u64 {
    match target.stage2_requirement() {
        Stage2Requirement::NotRequested => 0,
        Stage2Requirement::WholeHeadAndHidden {
            final_coefficient_floor,
        } => final_coefficient_floor,
    }
}

fn growth_tracked_fp32_payload_estimate(
    model: &TiedSwiGluTrainingModel,
    new_width: u32,
) -> Result<GrowthTrackedFp32PayloadEstimate, GrowthPlanError> {
    let arch = model.architecture();
    let mut source_elements = 0_u128;
    for parameter in model.parameters() {
        let expected = parameter.rows.checked_mul(parameter.cols).ok_or(
            GrowthPlanError::PayloadEstimateOverflow("source parameter geometry"),
        )?;
        if parameter.master.len() != expected {
            return Err(GrowthPlanError::TrainingAdapter(
                TrainingAdapterError::InvalidInput(format!(
                    "{} master is drained or shape-inconsistent",
                    parameter.name
                )),
            ));
        }
        source_elements = source_elements
            .checked_add(parameter.master.len() as u128)
            .ok_or(GrowthPlanError::PayloadEstimateOverflow(
                "source parameter payload",
            ))?;
    }
    for norm in arch
        .attn_norms
        .iter()
        .chain(&arch.ffn_norms)
        .chain(core::iter::once(&arch.output_norm))
    {
        source_elements = source_elements.checked_add(norm.len() as u128).ok_or(
            GrowthPlanError::PayloadEstimateOverflow("source norm payload"),
        )?;
    }

    let new_width = usize::try_from(new_width).map_err(|_| GrowthPlanError::WidthOutOfRange {
        axis: "new_intermediate_width",
        value: u128::from(new_width),
    })?;
    let added_width =
        new_width
            .checked_sub(arch.n_ff)
            .ok_or(GrowthPlanError::IntermediateNarrowing {
                existing: u32::try_from(arch.n_ff).unwrap_or(u32::MAX),
                requested: u32::try_from(new_width).unwrap_or(u32::MAX),
            })?;
    let added_mlp_elements = (arch.n_layers as u128)
        .checked_mul(3)
        .and_then(|value| value.checked_mul(arch.n_embd as u128))
        .and_then(|value| value.checked_mul(added_width as u128))
        .ok_or(GrowthPlanError::PayloadEstimateOverflow(
            "widened MLP payload",
        ))?;
    let widened_elements = source_elements.checked_add(added_mlp_elements).ok_or(
        GrowthPlanError::PayloadEstimateOverflow("widened model payload"),
    )?;
    let source_model_bytes = checked_memory_bytes(source_elements, "source model bytes")?;
    let widened_model_bytes = checked_memory_bytes(widened_elements, "widened model bytes")?;
    let oracle_logits_bytes = checked_memory_bytes(
        (arch.vocab as u128)
            .checked_mul(2)
            .ok_or(GrowthPlanError::PayloadEstimateOverflow(
                "oracle logit elements",
            ))?,
        "oracle logit bytes",
    )?;
    let tracked_peak_bytes = source_model_bytes
        .checked_add(widened_model_bytes.checked_mul(2).ok_or(
            GrowthPlanError::PayloadEstimateOverflow("two widened model payloads"),
        )?)
        .and_then(|value| value.checked_add(oracle_logits_bytes))
        .ok_or(GrowthPlanError::PayloadEstimateOverflow(
            "retained fp32 peak",
        ))?;
    Ok(GrowthTrackedFp32PayloadEstimate {
        source_model_bytes,
        widened_model_bytes,
        oracle_logits_bytes,
        tracked_peak_bytes,
    })
}

fn checked_memory_bytes(elements: u128, component: &'static str) -> Result<u64, GrowthPlanError> {
    elements
        .checked_mul(core::mem::size_of::<f32>() as u128)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(GrowthPlanError::PayloadEstimateOverflow(component))
}

fn growth_allocation_failed<T>(allocation: &'static str, elements: usize) -> GrowthPlanError {
    GrowthPlanError::AllocationFailed {
        allocation,
        requested_bytes: elements.saturating_mul(core::mem::size_of::<T>()),
    }
}

fn receipt_encoding_capacity(
    receipt: &AppliedIntermediateGrowthReceipt,
) -> Result<usize, GrowthPlanError> {
    let dynamic_values = receipt
        .source_indices
        .len()
        .checked_add(receipt.replication_counts.len())
        .and_then(|count| count.checked_add(receipt.split_numerators.as_ref().map_or(0, Vec::len)))
        .and_then(|count| count.checked_add(receipt.function_preservation.tokens.len()))
        .ok_or(GrowthPlanError::ReceiptEncoding(
            "encoding element count overflow",
        ))?;
    dynamic_values
        .checked_mul(core::mem::size_of::<u32>())
        .and_then(|bytes| bytes.checked_add(256))
        .ok_or(GrowthPlanError::ReceiptEncoding(
            "encoding byte count overflow",
        ))
}

const fn net2wider_algorithm_tag(has_splits: bool) -> u8 {
    if has_splits { 2 } else { 1 }
}

fn checked_u32_vector(reason: &'static str, values: &[usize]) -> Result<Vec<u32>, GrowthPlanError> {
    let mut result = Vec::new();
    result.try_reserve_exact(values.len()).map_err(|_| {
        growth_allocation_failed::<u32>("growth receipt integer vector", values.len())
    })?;
    for &value in values {
        result
            .push(u32::try_from(value).map_err(|_| GrowthPlanError::InvalidReceiptField(reason))?);
    }
    Ok(result)
}

fn copy_u32_vector(values: &[u32]) -> Result<Vec<u32>, GrowthPlanError> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(values.len())
        .map_err(|_| growth_allocation_failed::<u32>("growth split numerators", values.len()))?;
    result.extend_from_slice(values);
    Ok(result)
}

fn decode_applied_growth_receipt(
    bytes: &[u8],
) -> Result<AppliedIntermediateGrowthReceipt, GrowthPlanError> {
    let mut cursor = GrowthReceiptCursor::new(bytes);
    if cursor.take(4)? != APPLIED_GROWTH_RECEIPT_MAGIC {
        return Err(GrowthPlanError::ReceiptEncoding("bad magic"));
    }
    let version = cursor.u16()?;
    if version != APPLIED_GROWTH_RECEIPT_VERSION {
        return Err(GrowthPlanError::UnsupportedReceiptVersion(version));
    }
    if cursor.u16()? != 0 {
        return Err(GrowthPlanError::NonCanonicalReceipt);
    }
    let source_model_id = GrowthSourceModelId::new(cursor.digest()?)?;
    let result_model_id = GrowthResultModelId::new(cursor.digest()?)?;
    let intermediate_floor = cursor.u64()?;
    let stage2_final = cursor.u64()?;
    let target = if stage2_final == 0 {
        GrowthTarget::intermediate_at_least(intermediate_floor)?
    } else {
        GrowthTarget::staged(intermediate_floor, stage2_final)?
    };
    let resulting_core_coefficient_count = cursor.u64()?;
    let seed = cursor.u64()?;
    let old_width = cursor.u32()?;
    let new_width = cursor.u32()?;
    let algorithm_tag = cursor.u8()?;
    if cursor.take(3)? != [0; 3] {
        return Err(GrowthPlanError::NonCanonicalReceipt);
    }
    let source_count = cursor.u32()?;
    if source_count != new_width || new_width > MAX_STAGE1_RECEIPT_WIDTH {
        return Err(GrowthPlanError::InvalidReceiptField(
            "source-index count does not match new width",
        ));
    }
    let source_indices = cursor.u32_vector(source_count, MAX_STAGE1_RECEIPT_WIDTH)?;
    let replication_count = cursor.u32()?;
    if replication_count != old_width || old_width > MAX_STAGE1_RECEIPT_WIDTH {
        return Err(GrowthPlanError::InvalidReceiptField(
            "replication count does not match old width",
        ));
    }
    let replication_counts = cursor.u32_vector(replication_count, MAX_STAGE1_RECEIPT_WIDTH)?;
    let split_denominator = cursor.u32()?;
    let split_count = cursor.u32()?;
    let (split_denominator_log2, split_numerators) = if split_count == 0 {
        if split_denominator != 0 {
            return Err(GrowthPlanError::NonCanonicalReceipt);
        }
        (None, None)
    } else {
        if split_count != new_width || split_denominator == 0 {
            return Err(GrowthPlanError::InvalidReceiptField(
                "split metadata does not match new width",
            ));
        }
        (
            Some(split_denominator),
            Some(cursor.u32_vector(split_count, MAX_STAGE1_RECEIPT_WIDTH)?),
        )
    };
    if algorithm_tag != net2wider_algorithm_tag(split_numerators.is_some()) {
        return Err(GrowthPlanError::InvalidReceiptField(
            "Net2Wider algorithm tag disagrees with split metadata",
        ));
    }
    if cursor.u16()? != 1 || cursor.u16()? != 0 {
        return Err(GrowthPlanError::InvalidOracleEvidence(
            "unsupported oracle algorithm or nonzero reserved field",
        ));
    }
    let vocabulary = cursor.u32()?;
    let context_length = cursor.u32()?;
    let tolerance_bits = cursor.u32()?;
    let max_absolute_error_bits = cursor.u32()?;
    let worst_logit_index = cursor.u64()?;
    let source_worst_bits = cursor.u32()?;
    let grown_worst_bits = cursor.u32()?;
    let token_count = cursor.u32()?;
    let tokens = cursor.u32_vector(
        token_count,
        u32::try_from(DENSE_GROWTH_ORACLE_MAX_TOKENS).unwrap_or(u32::MAX),
    )?;
    let logit_count = cursor.u64()?;
    let source_logits_digest = cursor.digest()?;
    let grown_logits_digest = cursor.digest()?;
    if cursor.remaining() != 0 {
        return Err(GrowthPlanError::NonCanonicalReceipt);
    }
    let function_preservation = GrowthFunctionPreservationEvidence {
        vocabulary,
        context_length,
        tokens,
        token_count,
        logit_count,
        tolerance_bits,
        max_absolute_error_bits,
        worst_logit_index,
        source_worst_bits,
        grown_worst_bits,
        source_logits_digest,
        grown_logits_digest,
    };
    let receipt = AppliedIntermediateGrowthReceipt {
        source_model_id,
        result_model_id,
        target,
        resulting_core_coefficient_count,
        seed,
        old_width,
        new_width,
        source_indices,
        replication_counts,
        split_denominator_log2,
        split_numerators,
        function_preservation,
    };
    receipt.validate_invariants()?;
    if receipt.canonical_bytes()?.as_slice() != bytes {
        return Err(GrowthPlanError::NonCanonicalReceipt);
    }
    Ok(receipt)
}

struct GrowthReceiptCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> GrowthReceiptCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], GrowthPlanError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(GrowthPlanError::ReceiptEncoding("offset overflow"))?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(GrowthPlanError::ReceiptEncoding("truncated input"))?;
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, GrowthPlanError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, GrowthPlanError> {
        let mut bytes = [0; 2];
        bytes.copy_from_slice(self.take(2)?);
        Ok(u16::from_le_bytes(bytes))
    }

    fn u32(&mut self) -> Result<u32, GrowthPlanError> {
        let mut bytes = [0; 4];
        bytes.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, GrowthPlanError> {
        let mut bytes = [0; 8];
        bytes.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(bytes))
    }

    fn digest(&mut self) -> Result<[u8; 32], GrowthPlanError> {
        let mut digest = [0; 32];
        digest.copy_from_slice(self.take(32)?);
        Ok(digest)
    }

    fn u32_vector(&mut self, count: u32, maximum: u32) -> Result<Vec<u32>, GrowthPlanError> {
        if count > maximum {
            return Err(GrowthPlanError::ReceiptEncoding(
                "vector count exceeds versioned bound",
            ));
        }
        let count = usize::try_from(count)
            .map_err(|_| GrowthPlanError::ReceiptEncoding("vector count exceeds usize"))?;
        if count > self.remaining() / 4 {
            return Err(GrowthPlanError::ReceiptEncoding("truncated vector"));
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| growth_allocation_failed::<u32>("decoded receipt vector", count))?;
        for _ in 0..count {
            values.push(self.u32()?);
        }
        Ok(values)
    }
}

struct DenseOracleBackend;

impl TernaryBackend for DenseOracleBackend {
    fn device_id(&self) -> &str {
        "dense-growth-oracle"
    }

    fn capabilities(&self) -> DeviceCaps {
        DeviceCaps::new("oracle", "dense-only growth oracle")
    }

    fn upload_weights(
        &self,
        _packed: &[u8],
        _shape: GemmShape,
        format: TernaryFormat,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        Err(BackendError::UnsupportedFormat(format))
    }

    fn mpgemm(&self, parameters: MpGemm<'_>) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedFormat(parameters.format))
    }
}

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
    use crate::{
        DenseLinear, Mlp, MlpKind, ModelWeights, Projection, SwiGluMlp, TokenEmbedding,
        TransformerBlock,
    };

    #[test]
    fn post_widen_oracle_failure_restores_digest_geometry_and_weights() {
        let (config, spec, mut model, plan) = checked_growth_fixture();
        let source_digest = semantic_training_model_digest(&config, &spec, &model);
        let source_architecture = model.architecture().clone();
        let source_parameters = model.parameters().to_vec();

        let error = plan
            .apply_with_validation_hooks(
                &config,
                &spec,
                &mut model,
                || {
                    Err(GrowthPlanError::OracleExecution(
                        "injected post-widen oracle failure".to_owned(),
                    ))
                },
                |_| Ok(()),
            )
            .expect_err("injected oracle failure must reject growth");

        assert!(matches!(error, GrowthPlanError::OracleExecution(_)));
        assert_eq!(
            semantic_training_model_digest(&config, &spec, &model),
            source_digest
        );
        assert_eq!(model.architecture(), &source_architecture);
        assert_eq!(model.parameters(), source_parameters);
    }

    #[test]
    fn post_widen_receipt_failure_restores_digest_geometry_and_weights() {
        let (config, spec, mut model, plan) = checked_growth_fixture();
        let source_digest = semantic_training_model_digest(&config, &spec, &model);
        let source_architecture = model.architecture().clone();
        let source_parameters = model.parameters().to_vec();

        let error = plan
            .apply_with_validation_hooks(
                &config,
                &spec,
                &mut model,
                || Ok(()),
                |_| Err(GrowthPlanError::ReceiptMismatch),
            )
            .expect_err("injected receipt failure must reject growth");

        assert_eq!(error, GrowthPlanError::ReceiptMismatch);
        assert_eq!(
            semantic_training_model_digest(&config, &spec, &model),
            source_digest
        );
        assert_eq!(model.architecture(), &source_architecture);
        assert_eq!(model.parameters(), source_parameters);
    }

    #[test]
    fn post_widen_result_validation_failure_restores_digest_geometry_and_weights() {
        let (config, spec, mut model, plan) = checked_growth_fixture();
        let source_digest = semantic_training_model_digest(&config, &spec, &model);
        let source_architecture = model.architecture().clone();
        let source_parameters = model.parameters().to_vec();
        let injected_result = GrowthResultModelId::new([0x7f; 32]).expect("nonzero result id");

        let error = plan
            .apply_with_validation_hooks(
                &config,
                &spec,
                &mut model,
                || Ok(()),
                |receipt| {
                    receipt.result_model_id = injected_result;
                    Ok(())
                },
            )
            .expect_err("result-model mismatch must reject growth");

        assert!(matches!(
            error,
            GrowthPlanError::ResultModelMismatch {
                expected,
                actual: _
            } if expected == injected_result
        ));
        assert_eq!(
            semantic_training_model_digest(&config, &spec, &model),
            source_digest
        );
        assert_eq!(model.architecture(), &source_architecture);
        assert_eq!(model.parameters(), source_parameters);
    }

    #[test]
    fn canonical_oracle_descriptor_rejects_recomputed_semantic_tampering() {
        let (config, spec, mut model, plan) = checked_growth_fixture();
        let receipt = plan.apply(&config, &spec, &mut model).expect("growth");
        assert_eq!(receipt.function_preservation.vocabulary, 4);
        assert_eq!(receipt.function_preservation.context_length, 4);
        assert_eq!(receipt.function_preservation.tokens, [0, 3, 2, 3]);
        assert_eq!(receipt.function_preservation.logit_count, 4);

        let mut altered_token = receipt.clone();
        altered_token.function_preservation.tokens[1] = 1;
        assert_recomputed_canonical_rejected(altered_token);

        let mut out_of_vocabulary = receipt.clone();
        out_of_vocabulary.function_preservation.tokens[0] = 4;
        assert_recomputed_canonical_rejected(out_of_vocabulary);

        let mut altered_vocabulary = receipt.clone();
        altered_vocabulary.function_preservation.vocabulary = 5;
        assert_recomputed_canonical_rejected(altered_vocabulary);

        let mut altered_context = receipt.clone();
        altered_context.function_preservation.context_length = 2;
        assert_recomputed_canonical_rejected(altered_context);

        let mut altered_count = receipt.clone();
        altered_count.function_preservation.logit_count = 3;
        assert_recomputed_canonical_rejected(altered_count);

        let mut zero_source_digest = receipt.clone();
        zero_source_digest
            .function_preservation
            .source_logits_digest = [0; 32];
        assert_recomputed_canonical_rejected(zero_source_digest);

        let mut zero_grown_digest = receipt;
        zero_grown_digest.function_preservation.grown_logits_digest = [0; 32];
        assert_recomputed_canonical_rejected(zero_grown_digest);
    }

    #[test]
    fn canonical_result_identity_is_nonzero_and_requires_the_materialized_model() {
        let (config, spec, mut model, plan) = checked_growth_fixture();
        let receipt = plan.apply(&config, &spec, &mut model).expect("growth");
        let mut grown_config = config;
        grown_config.n_ff = receipt.new_width;
        receipt
            .validate_result_model(&grown_config, &spec, &model)
            .expect("bound result");

        let mut zero_result = receipt.clone();
        zero_result.result_model_id = GrowthResultModelId([0; 32]);
        let zero_bytes = zero_result
            .canonical_bytes()
            .expect("encode invalid fixture");
        let zero_digest = GrowthReceiptDigest(applied_growth_receipt_digest(&zero_bytes));
        assert_eq!(
            AppliedIntermediateGrowthReceipt::from_canonical_bytes_verified(
                &zero_bytes,
                zero_digest
            ),
            Err(GrowthPlanError::InvalidResultModelId)
        );

        let mut altered_result = receipt;
        altered_result.result_model_id = GrowthResultModelId::new([0x55; 32]).unwrap();
        let altered_bytes = altered_result.canonical_bytes().unwrap();
        let altered_digest = GrowthReceiptDigest(applied_growth_receipt_digest(&altered_bytes));
        let reopened = AppliedIntermediateGrowthReceipt::from_canonical_bytes_verified(
            &altered_bytes,
            altered_digest,
        )
        .expect("nonzero result identity is structurally canonical");
        assert!(matches!(
            reopened.validate_result_model(&grown_config, &spec, &model),
            Err(GrowthPlanError::ResultModelMismatch { .. })
        ));
    }

    #[test]
    fn tracked_payload_preflight_exposes_the_non_streaming_model_blocker() {
        let (config, spec, mut model, plan) = checked_growth_fixture();
        let estimate = plan
            .tracked_fp32_payload_estimate(&model)
            .expect("tracked payload estimate");

        assert_eq!(estimate.source_model_bytes(), 192);
        assert_eq!(estimate.widened_model_bytes(), 216);
        assert_eq!(estimate.oracle_logits_bytes(), 32);
        assert_eq!(estimate.tracked_peak_bytes(), 656);
        assert_eq!(
            plan.preflight_tracked_fp32_payload_limit(&model, 655),
            Err(GrowthPlanError::TrackedFp32PayloadLimitExceeded {
                required: 656,
                maximum: 655,
            })
        );
        assert_eq!(
            plan.preflight_tracked_fp32_payload_limit(&model, 656),
            Ok(estimate)
        );

        let before = model.clone();
        assert_eq!(
            plan.apply_with_tracked_fp32_payload_limit(&config, &spec, &mut model, 655),
            Err(GrowthPlanError::TrackedFp32PayloadLimitExceeded {
                required: 656,
                maximum: 655,
            })
        );
        assert_eq!(model, before);
    }

    fn assert_recomputed_canonical_rejected(receipt: AppliedIntermediateGrowthReceipt) {
        let bytes = receipt
            .canonical_bytes()
            .expect("invalid semantic fixture remains encodable");
        let digest = GrowthReceiptDigest(applied_growth_receipt_digest(&bytes));
        assert!(matches!(
            AppliedIntermediateGrowthReceipt::from_canonical_bytes_verified(&bytes, digest),
            Err(GrowthPlanError::InvalidOracleEvidence(_))
        ));
    }

    fn checked_growth_fixture() -> (
        ModelConfig,
        ArchSpec,
        TiedSwiGluTrainingModel,
        IntermediateGrowthPlan,
    ) {
        let config = ModelConfig {
            arch: "llama".to_owned(),
            n_layers: 1,
            n_embd: 2,
            n_head: 1,
            n_head_kv: 1,
            head_dim: 2,
            n_ff: 3,
            n_ctx: 4,
            rope_theta: 10_000.0,
            rms_eps: 1e-5,
        };
        let spec = ArchSpec {
            mlp: MlpKind::SwiGlu,
            attn_sub_norm: false,
            ffn_sub_norm: false,
            qk_norm: false,
            qkv_bias: false,
            tied_embeddings: true,
        };
        let weights = ModelWeights {
            token_embd: TokenEmbedding::from_dense(
                vec![0.25, -0.125, 0.5, 0.375, -0.25, 0.75, 0.125, -0.5],
                4,
                2,
            )
            .expect("test embedding"),
            vocab: 4,
            n_embd: 2,
            layers: vec![TransformerBlock {
                attn_norm: vec![1.0, 0.875],
                q_proj: test_projection(2, 2, 1),
                k_proj: test_projection(2, 2, 2),
                v_proj: test_projection(2, 2, 3),
                o_proj: test_projection(2, 2, 4),
                attn_sub_norm: Vec::new(),
                q_bias: Vec::new(),
                k_bias: Vec::new(),
                v_bias: Vec::new(),
                q_norm: Vec::new(),
                k_norm: Vec::new(),
                ffn_norm: vec![0.75, 1.125],
                mlp: Mlp::SwiGlu(SwiGluMlp {
                    gate: test_projection(3, 2, 5),
                    up: test_projection(3, 2, 6),
                    down: test_projection(2, 3, 7),
                }),
            }],
            output_norm: vec![1.0, 0.9375],
            lm_head: None,
        };
        let model = TiedSwiGluTrainingModel::extract(&config, &spec, &weights)
            .expect("test training model");
        let source_model_id = GrowthSourceModelId::from_training_model(&config, &spec, &model)
            .expect("source identity");
        let geometry = ProjectionGeometry::new(
            1,
            2,
            2,
            2,
            3,
            4,
            ProjectionPlaneCounts::new(1, 1, 1, 1, 1, 1, 1).expect("plane counts"),
            FixedEmbeddingPolicy::PreservedDense { tied_lm_head: true },
        )
        .expect("growth geometry");
        let target = GrowthTarget::intermediate_at_least(
            geometry
                .core_coefficient_count(4)
                .expect("target coefficient count"),
        )
        .expect("growth target");
        let plan = geometry
            .plan(source_model_id, target, 0x27)
            .expect("growth plan");
        (config, spec, model, plan)
    }

    fn test_projection(rows: usize, cols: usize, seed: usize) -> Projection {
        let weights = (0..rows * cols)
            .map(|index| ((index * 7 + seed * 3) % 17) as f32 / 64.0 - 0.125)
            .collect();
        Projection::Dense(
            DenseLinear::new_exact(weights, rows, cols).expect("test dense projection"),
        )
    }

    #[test]
    fn source_model_identity_rejects_the_unbound_zero_digest() {
        assert_eq!(
            GrowthSourceModelId::new([0; 32]),
            Err(GrowthPlanError::InvalidSourceModelId)
        );
    }

    #[test]
    fn growth_plan_cannot_replay_against_a_different_source_checkpoint() {
        let source_a = source_model_id(0x11);
        let source_b = source_model_id(0x22);
        let target = GrowthTarget::intermediate_at_least(120).unwrap();
        let plan_a = tiny_geometry().plan(source_a, target, 17).unwrap();
        let plan_b = tiny_geometry().plan(source_b, target, 17).unwrap();

        assert_eq!(plan_a.source_model_id(), source_a);
        assert_eq!(
            plan_a.validate_source_model_id(source_b),
            Err(GrowthPlanError::SourceModelMismatch {
                expected: source_a,
                actual: source_b,
            })
        );
        assert_eq!(
            plan_b.validate_source_model_id(source_a),
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
    fn deterministic_expected_mapping_is_bound_to_widths_and_seed() {
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
        let wrong_seed = tritium_train::Net2WiderPlan::seeded(3, 8, 18).unwrap();
        assert_ne!(first, wrong_seed);
        assert_eq!(plan.seed(), 17);
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
