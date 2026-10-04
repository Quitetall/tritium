//! Streamed block-output and teacher-logit reconstruction objectives.

use core::fmt;
use std::collections::BTreeSet;

use tritium_format::{
    ModelId, RuntimeEvidenceError, RuntimeFinalLogitsAccumulator,
    RuntimeOutputReconstructionAccumulator, RuntimeOutputScope, RuntimeOutputScopeAccumulator,
    RuntimeOutputScopeEvidence,
};

mod codec;

const SPEC_HASH_CONTEXT: &str = "tritium salt v2 output reconstruction spec v1";
const TEACHER_HASH_CONTEXT: &str = "tritium salt v2 output reconstruction teacher v1";
const CANDIDATE_HASH_CONTEXT: &str = "tritium salt v2 output reconstruction candidate v1";
const CANDIDATE_HASH_CONTEXT_V3: &str = "tritium salt v2 output reconstruction candidate v2";
const RECEIPT_HASH_CONTEXT: &str = "tritium salt v2 output reconstruction receipt v1";
const MAX_OUTPUT_RECONSTRUCTION_SCOPES: usize = 1 << 20;
const MAX_FIXED_TRIT_REFIT_GROUPS: usize = 1024;
const MAX_FIXED_TRIT_REFIT_SWEEPS: usize = 100_000;

/// Ordered model region evaluated by output reconstruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum OutputReconstructionScope {
    /// Inclusive/exclusive transformer-block range.
    Block {
        /// First block in the reconstructed region.
        start: u32,
        /// Exclusive block bound.
        end: u32,
    },
    /// Final LM-head logits evaluated with teacher cross-entropy and KL.
    FinalLogits,
}

/// Frozen block traversal used for one output-aware fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputReconstructionSchedule {
    /// Evaluate each transformer block independently.
    Blocks {
        /// Number of transformer blocks.
        block_count: u32,
    },
    /// Evaluate deterministic overlapping block windows and a tail-covering window.
    SlidingWindows {
        /// Number of transformer blocks.
        block_count: u32,
        /// Blocks evaluated together.
        window_size: u32,
        /// Start-position stride before the mandatory tail window.
        stride: u32,
    },
}

/// Weights and temperature for candidate selection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutputObjectiveWeights {
    block_mse: f64,
    teacher_cross_entropy: f64,
    teacher_kl: f64,
    temperature: f64,
}

impl OutputObjectiveWeights {
    /// Construct finite, non-negative objective weights and positive temperature.
    ///
    /// # Errors
    /// Rejects non-finite/negative weights, a non-positive temperature, or an all-zero objective.
    pub fn new(
        block_mse: f64,
        teacher_cross_entropy: f64,
        teacher_kl: f64,
        temperature: f64,
    ) -> Result<Self, OutputReconstructionError> {
        let weights = [block_mse, teacher_cross_entropy, teacher_kl];
        if weights
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
            || !temperature.is_finite()
            || temperature <= 0.0
            || weights.iter().all(|value| *value == 0.0)
        {
            return Err(OutputReconstructionError::InvalidObjective);
        }
        Ok(Self {
            block_mse: canonical_zero(block_mse),
            teacher_cross_entropy: canonical_zero(teacher_cross_entropy),
            teacher_kl: canonical_zero(teacher_kl),
            temperature,
        })
    }

    /// Block-output MSE selection weight.
    #[must_use]
    pub const fn block_mse(self) -> f64 {
        self.block_mse
    }

    /// Teacher-distribution cross-entropy selection weight.
    #[must_use]
    pub const fn teacher_cross_entropy(self) -> f64 {
        self.teacher_cross_entropy
    }

    /// Teacher KL selection weight.
    #[must_use]
    pub const fn teacher_kl(self) -> f64 {
        self.teacher_kl
    }

    /// Distillation temperature.
    #[must_use]
    pub const fn temperature(self) -> f64 {
        self.temperature
    }
}

/// Immutable provenance, schedule, and objective for one output-aware search.
#[derive(Clone, Debug, PartialEq)]
pub struct OutputReconstructionSpec {
    source_model_id: ModelId,
    activation_digest: [u8; 32],
    token_stream_digest: [u8; 32],
    validation_digest: [u8; 32],
    schedule: OutputReconstructionSchedule,
    scopes: Vec<OutputReconstructionScope>,
    objective: OutputObjectiveWeights,
    batches_per_scope: u32,
    restarts: usize,
    spec_id: [u8; 32],
}

impl OutputReconstructionSpec {
    /// Build a source/data-bound block or sliding-window reconstruction search.
    ///
    /// Final logits are always appended after all block scopes. Every candidate
    /// must observe exactly `batches_per_scope` batches for every scope.
    ///
    /// # Errors
    /// Rejects missing provenance, malformed schedules, or zero batch/restart counts.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source_model_id: ModelId,
        activation_digest: [u8; 32],
        token_stream_digest: [u8; 32],
        validation_digest: [u8; 32],
        schedule: OutputReconstructionSchedule,
        objective: OutputObjectiveWeights,
        batches_per_scope: u32,
        restarts: usize,
    ) -> Result<Self, OutputReconstructionError> {
        if source_model_id.as_bytes() == &[0; 32]
            || activation_digest == [0; 32]
            || token_stream_digest == [0; 32]
            || validation_digest == [0; 32]
        {
            return Err(OutputReconstructionError::MissingProvenance);
        }
        if batches_per_scope == 0 || restarts == 0 {
            return Err(OutputReconstructionError::InvalidCount);
        }
        let scopes = schedule_scopes(schedule)?;
        let mut spec = Self {
            source_model_id,
            activation_digest,
            token_stream_digest,
            validation_digest,
            schedule,
            scopes,
            objective,
            batches_per_scope,
            restarts,
            spec_id: [0; 32],
        };
        spec.spec_id = spec.derive_id();
        Ok(spec)
    }

    /// Canonical ordered block regions followed by final logits.
    #[must_use]
    pub fn scopes(&self) -> &[OutputReconstructionScope] {
        &self.scopes
    }

    /// Source-model semantic identity evaluated by this specification.
    #[must_use]
    pub const fn source_model_id(&self) -> ModelId {
        self.source_model_id
    }

    /// Exact token-stream identity required by this specification.
    #[must_use]
    pub const fn token_stream_digest(&self) -> &[u8; 32] {
        &self.token_stream_digest
    }

    /// Required batches for each scope.
    #[must_use]
    pub const fn batches_per_scope(&self) -> u32 {
        self.batches_per_scope
    }

    /// Required deterministic initialization count.
    #[must_use]
    pub const fn restarts(&self) -> usize {
        self.restarts
    }

    /// Objective weights and temperature.
    #[must_use]
    pub const fn objective(&self) -> OutputObjectiveWeights {
        self.objective
    }

    /// Content identity of provenance, schedule, and objective.
    #[must_use]
    pub const fn spec_id(&self) -> &[u8; 32] {
        &self.spec_id
    }

    fn derive_id(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new_derive_key(SPEC_HASH_CONTEXT);
        hasher.update(self.source_model_id.as_bytes());
        hasher.update(&self.activation_digest);
        hasher.update(&self.token_stream_digest);
        hasher.update(&self.validation_digest);
        match self.schedule {
            OutputReconstructionSchedule::Blocks { block_count } => {
                hasher.update(&[1]);
                hasher.update(&block_count.to_le_bytes());
            }
            OutputReconstructionSchedule::SlidingWindows {
                block_count,
                window_size,
                stride,
            } => {
                hasher.update(&[2]);
                hasher.update(&block_count.to_le_bytes());
                hasher.update(&window_size.to_le_bytes());
                hasher.update(&stride.to_le_bytes());
            }
        }
        for value in [
            self.objective.block_mse,
            self.objective.teacher_cross_entropy,
            self.objective.teacher_kl,
            self.objective.temperature,
        ] {
            hasher.update(&value.to_bits().to_le_bytes());
        }
        hasher.update(&self.batches_per_scope.to_le_bytes());
        hasher.update(&(self.restarts as u64).to_le_bytes());
        *hasher.finalize().as_bytes()
    }

    fn expected_observations(&self) -> Result<u64, OutputReconstructionError> {
        u64::try_from(self.scopes.len())
            .map_err(|_| OutputReconstructionError::CountOverflow)?
            .checked_mul(u64::from(self.batches_per_scope))
            .ok_or(OutputReconstructionError::CountOverflow)
    }

    fn objective_for(&self, block_mse: f64, teacher_cross_entropy: f64, teacher_kl: f64) -> f64 {
        canonical_zero(
            self.objective.block_mse * block_mse
                + self.objective.teacher_cross_entropy * teacher_cross_entropy
                + self.objective.teacher_kl * teacher_kl,
        )
    }
}

/// Streaming accumulator for one deterministic output-aware initialization.
#[derive(Clone, Debug)]
pub struct OutputReconstructionAccumulator {
    spec: OutputReconstructionSpec,
    candidate_id: [u8; 32],
    initialization_seed: u64,
    scope_index: usize,
    batch_index: u32,
    observations: u64,
    block_squared_error: f64,
    block_elements: u64,
    teacher_cross_entropy_sum: f64,
    teacher_kl_sum: f64,
    final_tokens: u64,
    teacher_hasher: blake3::Hasher,
    student_outputs: RuntimeOutputReconstructionAccumulator,
    student_scope_outputs: Vec<RuntimeOutputScopeAccumulator>,
    runtime_final_logits: RuntimeFinalLogitsAccumulator,
}

impl OutputReconstructionAccumulator {
    /// Begin one candidate without retaining any activation or logit batch.
    ///
    /// # Errors
    /// Rejects a zero candidate identity.
    pub fn new(
        spec: &OutputReconstructionSpec,
        candidate_id: [u8; 32],
        initialization_seed: u64,
    ) -> Result<Self, OutputReconstructionError> {
        if candidate_id == [0; 32] {
            return Err(OutputReconstructionError::MissingCandidateIdentity);
        }
        let mut teacher_hasher = blake3::Hasher::new_derive_key(TEACHER_HASH_CONTEXT);
        teacher_hasher.update(spec.spec_id());
        let student_outputs = RuntimeOutputReconstructionAccumulator::new(
            spec.spec_id(),
            &candidate_id,
            initialization_seed,
        )
        .map_err(map_runtime_evidence_error)?;
        let mut student_scope_outputs = Vec::new();
        student_scope_outputs
            .try_reserve_exact(spec.scopes().len())
            .map_err(|_| OutputReconstructionError::ReceiptAllocationFailed)?;
        for scope in spec.scopes() {
            student_scope_outputs.push(
                RuntimeOutputScopeAccumulator::new(
                    spec.spec_id(),
                    &candidate_id,
                    initialization_seed,
                    match scope {
                        OutputReconstructionScope::Block { start, end } => {
                            RuntimeOutputScope::Block {
                                start: *start,
                                end: *end,
                            }
                        }
                        OutputReconstructionScope::FinalLogits => RuntimeOutputScope::FinalLogits,
                    },
                )
                .map_err(map_runtime_evidence_error)?,
            );
        }
        Ok(Self {
            spec: spec.clone(),
            candidate_id,
            initialization_seed,
            scope_index: 0,
            batch_index: 0,
            observations: 0,
            block_squared_error: 0.0,
            block_elements: 0,
            teacher_cross_entropy_sum: 0.0,
            teacher_kl_sum: 0.0,
            final_tokens: 0,
            teacher_hasher,
            student_outputs,
            student_scope_outputs,
            runtime_final_logits: RuntimeFinalLogitsAccumulator::new(),
        })
    }

    /// Consume one canonical teacher/student output batch.
    ///
    /// `mask` selects rows/tokens. Storage remains caller-owned and can be
    /// released immediately after return.
    ///
    /// # Errors
    /// Rejects out-of-order scopes/batches, invalid geometry, empty selections,
    /// non-finite outputs, or count overflow.
    #[allow(clippy::too_many_arguments)]
    pub fn observe(
        &mut self,
        scope: OutputReconstructionScope,
        batch_index: u32,
        rows: usize,
        columns: usize,
        mask: &[bool],
        teacher: &[f32],
        student: &[f32],
    ) -> Result<(), OutputReconstructionError> {
        let expected = self
            .spec
            .scopes
            .get(self.scope_index)
            .copied()
            .ok_or(OutputReconstructionError::ExtraObservation)?;
        if scope != expected || batch_index != self.batch_index {
            return Err(OutputReconstructionError::ScopeOrder {
                expected,
                expected_batch: self.batch_index,
                got: scope,
                got_batch: batch_index,
            });
        }
        let values = rows
            .checked_mul(columns)
            .ok_or(OutputReconstructionError::InvalidGeometry)?;
        if rows == 0
            || columns == 0
            || mask.len() != rows
            || teacher.len() != values
            || student.len() != values
        {
            return Err(OutputReconstructionError::InvalidGeometry);
        }
        if teacher.iter().any(|value| !value.is_finite()) {
            return Err(OutputReconstructionError::NonFiniteOutput { teacher: true });
        }
        if student.iter().any(|value| !value.is_finite()) {
            return Err(OutputReconstructionError::NonFiniteOutput { teacher: false });
        }
        let selected = mask.iter().filter(|selected| **selected).count();
        if selected == 0 {
            return Err(OutputReconstructionError::EmptyTokenSelection);
        }
        if scope == OutputReconstructionScope::FinalLogits && columns < 2 {
            return Err(OutputReconstructionError::InvalidGeometry);
        }
        hash_observation(
            &mut self.teacher_hasher,
            scope,
            batch_index,
            rows,
            columns,
            mask,
            teacher,
        );
        self.student_outputs
            .observe(
                match scope {
                    OutputReconstructionScope::Block { start, end } => {
                        RuntimeOutputScope::Block { start, end }
                    }
                    OutputReconstructionScope::FinalLogits => RuntimeOutputScope::FinalLogits,
                },
                batch_index,
                rows,
                columns,
                mask,
                student,
            )
            .map_err(map_runtime_evidence_error)?;
        self.student_scope_outputs[self.scope_index]
            .observe(batch_index, rows, columns, mask, student)
            .map_err(map_runtime_evidence_error)?;
        match scope {
            OutputReconstructionScope::Block { .. } => {
                for (row, selected) in mask.iter().copied().enumerate() {
                    if !selected {
                        continue;
                    }
                    let start = row * columns;
                    for index in start..start + columns {
                        let residual = f64::from(teacher[index]) - f64::from(student[index]);
                        self.block_squared_error += residual * residual;
                    }
                }
                self.block_elements = self
                    .block_elements
                    .checked_add(
                        u64::try_from(
                            selected
                                .checked_mul(columns)
                                .ok_or(OutputReconstructionError::CountOverflow)?,
                        )
                        .map_err(|_| OutputReconstructionError::CountOverflow)?,
                    )
                    .ok_or(OutputReconstructionError::CountOverflow)?;
            }
            OutputReconstructionScope::FinalLogits => {
                for (row, selected) in mask.iter().copied().enumerate() {
                    if !selected {
                        continue;
                    }
                    let start = row * columns;
                    self.runtime_final_logits
                        .observe(&student[start..start + columns])
                        .map_err(map_runtime_evidence_error)?;
                    let (cross_entropy, kl) = distillation_losses(
                        &teacher[start..start + columns],
                        &student[start..start + columns],
                        self.spec.objective.temperature,
                    );
                    self.teacher_cross_entropy_sum += cross_entropy;
                    self.teacher_kl_sum += kl;
                }
                self.final_tokens = self
                    .final_tokens
                    .checked_add(
                        u64::try_from(selected)
                            .map_err(|_| OutputReconstructionError::CountOverflow)?,
                    )
                    .ok_or(OutputReconstructionError::CountOverflow)?;
            }
        }
        self.observations = self
            .observations
            .checked_add(1)
            .ok_or(OutputReconstructionError::CountOverflow)?;
        self.batch_index += 1;
        if self.batch_index == self.spec.batches_per_scope {
            self.batch_index = 0;
            self.scope_index += 1;
        }
        Ok(())
    }

    /// Seal exact aggregate losses and aggregate plus per-scope output evidence.
    ///
    /// # Errors
    /// Rejects incomplete scope coverage or missing block/logit measurements.
    pub fn finish(self) -> Result<OutputCandidateReceipt, OutputReconstructionError> {
        if self.scope_index != self.spec.scopes.len() || self.batch_index != 0 {
            return Err(OutputReconstructionError::IncompleteCandidate);
        }
        if self.block_elements == 0 || self.final_tokens == 0 {
            return Err(OutputReconstructionError::IncompleteCandidate);
        }
        let block_output_mse = self.block_squared_error / self.block_elements as f64;
        let teacher_cross_entropy = self.teacher_cross_entropy_sum / self.final_tokens as f64;
        let teacher_kl = self.teacher_kl_sum / self.final_tokens as f64;
        let objective =
            self.spec
                .objective_for(block_output_mse, teacher_cross_entropy, teacher_kl);
        if !objective.is_finite() {
            return Err(OutputReconstructionError::NonFiniteObjective);
        }
        let teacher_evidence_digest = *self.teacher_hasher.finalize().as_bytes();
        let student_outputs = self
            .student_outputs
            .finish()
            .map_err(map_runtime_evidence_error)?;
        let student_output_digest = *student_outputs.digest();
        let runtime_final_logits = self
            .runtime_final_logits
            .finish()
            .map_err(map_runtime_evidence_error)?;
        let mut scope_evidence = Vec::new();
        scope_evidence
            .try_reserve_exact(self.student_scope_outputs.len())
            .map_err(|_| OutputReconstructionError::ReceiptAllocationFailed)?;
        for output in self.student_scope_outputs {
            scope_evidence.push(output.finish().map_err(map_runtime_evidence_error)?);
        }
        let mut receipt = OutputCandidateReceipt {
            spec_id: self.spec.spec_id,
            candidate_id: self.candidate_id,
            initialization_seed: self.initialization_seed,
            teacher_evidence_digest,
            student_output_digest,
            runtime_final_logits_digest: *runtime_final_logits.digest(),
            runtime_batch_count: runtime_final_logits.batch_count(),
            runtime_logit_count: runtime_final_logits.logit_count(),
            observations: self.observations,
            block_elements: self.block_elements,
            final_tokens: self.final_tokens,
            block_output_mse: canonical_zero(block_output_mse),
            teacher_cross_entropy: canonical_zero(teacher_cross_entropy),
            teacher_kl: canonical_zero(teacher_kl),
            objective: canonical_zero(objective),
            scope_evidence,
            receipt_id: [0; 32],
        };
        receipt.receipt_id = receipt.derive_id();
        Ok(receipt)
    }
}

/// Immutable metrics and streamed evidence for one deterministic initialization.
#[derive(Clone, Debug, PartialEq)]
pub struct OutputCandidateReceipt {
    spec_id: [u8; 32],
    candidate_id: [u8; 32],
    initialization_seed: u64,
    teacher_evidence_digest: [u8; 32],
    student_output_digest: [u8; 32],
    runtime_final_logits_digest: [u8; 32],
    runtime_batch_count: u64,
    runtime_logit_count: u64,
    observations: u64,
    block_elements: u64,
    final_tokens: u64,
    block_output_mse: f64,
    teacher_cross_entropy: f64,
    teacher_kl: f64,
    objective: f64,
    scope_evidence: Vec<RuntimeOutputScopeEvidence>,
    receipt_id: [u8; 32],
}

impl OutputCandidateReceipt {
    /// Candidate content identity supplied by the production initializer.
    #[must_use]
    pub const fn candidate_id(&self) -> &[u8; 32] {
        &self.candidate_id
    }

    /// Seed that binds this restart's student-output stream identity.
    #[must_use]
    pub const fn initialization_seed(&self) -> u64 {
        self.initialization_seed
    }

    /// Exact teacher stream identity shared by every valid restart.
    #[must_use]
    pub const fn teacher_evidence_digest(&self) -> &[u8; 32] {
        &self.teacher_evidence_digest
    }

    /// Aggregate block-plus-logit student stream identity used by reconstruction.
    #[must_use]
    pub const fn student_output_digest(&self) -> &[u8; 32] {
        &self.student_output_digest
    }

    /// Final-logit identity in the shared runtime execution domain.
    #[must_use]
    pub const fn runtime_final_logits_digest(&self) -> &[u8; 32] {
        &self.runtime_final_logits_digest
    }

    /// Runtime-comparable final-logit batch count.
    #[must_use]
    pub const fn runtime_batch_count(&self) -> u64 {
        self.runtime_batch_count
    }

    /// Runtime-comparable final-logit value count.
    #[must_use]
    pub const fn runtime_logit_count(&self) -> u64 {
        self.runtime_logit_count
    }

    /// Scope digests committed by `TSV2OUT` v3 for block-major runtime matching.
    #[must_use]
    pub fn scope_evidence(&self) -> &[RuntimeOutputScopeEvidence] {
        &self.scope_evidence
    }

    /// Mean squared error across selected block outputs.
    #[must_use]
    pub const fn block_output_mse(&self) -> f64 {
        self.block_output_mse
    }

    /// Mean teacher-distribution cross-entropy at configured temperature.
    #[must_use]
    pub const fn teacher_cross_entropy(&self) -> f64 {
        self.teacher_cross_entropy
    }

    /// Mean teacher KL, multiplied by temperature squared.
    #[must_use]
    pub const fn teacher_kl(&self) -> f64 {
        self.teacher_kl
    }

    /// Frozen weighted selection objective.
    #[must_use]
    pub const fn objective(&self) -> f64 {
        self.objective
    }

    /// Content identity of all candidate fields.
    #[must_use]
    pub const fn receipt_id(&self) -> &[u8; 32] {
        &self.receipt_id
    }

    fn derive_id(&self) -> [u8; 32] {
        if self.scope_evidence.is_empty() {
            return self.derive_v2_id();
        }
        let mut hasher = blake3::Hasher::new_derive_key(CANDIDATE_HASH_CONTEXT_V3);
        hasher.update(&self.spec_id);
        hasher.update(&self.candidate_id);
        hasher.update(&self.initialization_seed.to_le_bytes());
        hasher.update(&self.teacher_evidence_digest);
        hasher.update(&self.student_output_digest);
        hasher.update(&self.runtime_final_logits_digest);
        hasher.update(&self.runtime_batch_count.to_le_bytes());
        hasher.update(&self.runtime_logit_count.to_le_bytes());
        hasher.update(&self.observations.to_le_bytes());
        hasher.update(&self.block_elements.to_le_bytes());
        hasher.update(&self.final_tokens.to_le_bytes());
        for value in [
            self.block_output_mse,
            self.teacher_cross_entropy,
            self.teacher_kl,
            self.objective,
        ] {
            hasher.update(&value.to_bits().to_le_bytes());
        }
        hasher.update(&(self.scope_evidence.len() as u64).to_le_bytes());
        for evidence in &self.scope_evidence {
            hash_scope_evidence(&mut hasher, evidence);
        }
        *hasher.finalize().as_bytes()
    }

    fn derive_v2_id(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new_derive_key(CANDIDATE_HASH_CONTEXT);
        hasher.update(&self.spec_id);
        hasher.update(&self.candidate_id);
        hasher.update(&self.initialization_seed.to_le_bytes());
        hasher.update(&self.teacher_evidence_digest);
        hasher.update(&self.student_output_digest);
        hasher.update(&self.runtime_final_logits_digest);
        hasher.update(&self.runtime_batch_count.to_le_bytes());
        hasher.update(&self.runtime_logit_count.to_le_bytes());
        hasher.update(&self.observations.to_le_bytes());
        hasher.update(&self.block_elements.to_le_bytes());
        hasher.update(&self.final_tokens.to_le_bytes());
        for value in [
            self.block_output_mse,
            self.teacher_cross_entropy,
            self.teacher_kl,
            self.objective,
        ] {
            hasher.update(&value.to_bits().to_le_bytes());
        }
        *hasher.finalize().as_bytes()
    }
}

/// Selected output-aware restart and complete matched-basin evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct OutputReconstructionReceipt {
    spec_id: [u8; 32],
    teacher_evidence_digest: [u8; 32],
    candidates: Vec<OutputCandidateReceipt>,
    selected_candidate_id: [u8; 32],
    receipt_id: [u8; 32],
}

/// Streaming non-negative least-squares fit for one output row with fixed trits.
///
/// Each observed value is the contribution of every fixed-trit scale group for
/// one valid calibration token. Only the group Gram matrix and target products
/// are retained, so callers can release activation batches immediately.
#[derive(Clone, Debug)]
pub struct FixedTritScaleRefitAccumulator {
    gram: Vec<f64>,
    target_products: Vec<f64>,
    target_squared: f64,
    observations: u64,
    coordinate_sweeps: usize,
}

/// Non-negative scale solution for one output row and its calibration error.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedTritScaleRefit {
    scales: Vec<f64>,
    squared_error: f64,
    observations: u64,
}

impl FixedTritScaleRefit {
    /// Fitted non-negative scale for each fixed-trit group.
    #[must_use]
    pub fn scales(&self) -> &[f64] {
        &self.scales
    }

    /// Sum of squared output error over observed calibration rows.
    #[must_use]
    pub const fn squared_error(&self) -> f64 {
        self.squared_error
    }

    /// Number of valid calibration rows consumed by the fit.
    #[must_use]
    pub const fn observations(&self) -> u64 {
        self.observations
    }
}

impl FixedTritScaleRefitAccumulator {
    /// Start a bounded-memory refit for one output row.
    ///
    /// `group_count` is the number of fixed-trit scale groups, and
    /// `coordinate_sweeps` controls deterministic cyclic coordinate descent.
    ///
    /// # Errors
    /// Rejects zero or excessive dimensions, zero sweeps, or allocation failure.
    pub fn new(
        group_count: usize,
        coordinate_sweeps: usize,
    ) -> Result<Self, OutputReconstructionError> {
        if group_count == 0
            || group_count > MAX_FIXED_TRIT_REFIT_GROUPS
            || coordinate_sweeps == 0
            || coordinate_sweeps > MAX_FIXED_TRIT_REFIT_SWEEPS
        {
            return Err(OutputReconstructionError::InvalidScaleRefit);
        }
        let gram_len = group_count
            .checked_mul(group_count)
            .ok_or(OutputReconstructionError::CountOverflow)?;
        let mut gram = Vec::new();
        gram.try_reserve_exact(gram_len)
            .map_err(|_| OutputReconstructionError::ReceiptAllocationFailed)?;
        gram.resize(gram_len, 0.0);
        let mut target_products = Vec::new();
        target_products
            .try_reserve_exact(group_count)
            .map_err(|_| OutputReconstructionError::ReceiptAllocationFailed)?;
        target_products.resize(group_count, 0.0);
        Ok(Self {
            gram,
            target_products,
            target_squared: 0.0,
            observations: 0,
            coordinate_sweeps,
        })
    }

    /// Add one row of fixed-trit group outputs and the matching teacher output.
    ///
    /// # Errors
    /// Rejects the wrong group count, non-finite values, or counter overflow.
    pub fn observe(
        &mut self,
        group_outputs: &[f64],
        teacher_output: f64,
    ) -> Result<(), OutputReconstructionError> {
        let groups = self.target_products.len();
        if group_outputs.len() != groups {
            return Err(OutputReconstructionError::InvalidGeometry);
        }
        if !teacher_output.is_finite() {
            return Err(OutputReconstructionError::NonFiniteOutput { teacher: true });
        }
        if group_outputs.iter().any(|value| !value.is_finite()) {
            return Err(OutputReconstructionError::NonFiniteOutput { teacher: false });
        }
        let observations = self
            .observations
            .checked_add(1)
            .ok_or(OutputReconstructionError::CountOverflow)?;
        let target_squared = self.target_squared + teacher_output * teacher_output;
        if !target_squared.is_finite() {
            return Err(OutputReconstructionError::NonFiniteScaleRefit);
        }
        for group in 0..groups {
            if !(self.target_products[group] + group_outputs[group] * teacher_output).is_finite() {
                return Err(OutputReconstructionError::NonFiniteScaleRefit);
            }
            for other in 0..groups {
                let index = group * groups + other;
                if !(self.gram[index] + group_outputs[group] * group_outputs[other]).is_finite() {
                    return Err(OutputReconstructionError::NonFiniteScaleRefit);
                }
            }
        }
        for group in 0..groups {
            self.target_products[group] += group_outputs[group] * teacher_output;
            for other in 0..groups {
                let index = group * groups + other;
                self.gram[index] += group_outputs[group] * group_outputs[other];
            }
        }
        self.target_squared = target_squared;
        self.observations = observations;
        Ok(())
    }

    /// Solve the accumulated non-negative least-squares problem.
    ///
    /// Coordinates are visited in a fixed order. A zero diagonal produces a
    /// zero scale; every coordinate update is bounded below by zero and cannot
    /// increase the quadratic objective.
    ///
    /// # Errors
    /// Rejects an empty stream or an invalid/non-finite accumulated solution.
    pub fn finish(self) -> Result<FixedTritScaleRefit, OutputReconstructionError> {
        if self.observations == 0 {
            return Err(OutputReconstructionError::InvalidScaleRefit);
        }
        let groups = self.target_products.len();
        let mut scales = vec![0.0; groups];
        for _ in 0..self.coordinate_sweeps {
            for group in 0..groups {
                let diagonal = self.gram[group * groups + group];
                if diagonal <= 0.0 {
                    scales[group] = 0.0;
                    continue;
                }
                let fitted_product = (0..groups)
                    .map(|other| self.gram[group * groups + other] * scales[other])
                    .sum::<f64>();
                if !fitted_product.is_finite() {
                    return Err(OutputReconstructionError::NonFiniteScaleRefit);
                }
                let updated = (scales[group]
                    + (self.target_products[group] - fitted_product) / diagonal)
                    .max(0.0);
                if !updated.is_finite() {
                    return Err(OutputReconstructionError::NonFiniteScaleRefit);
                }
                scales[group] = updated;
            }
        }
        let linear = scales
            .iter()
            .zip(&self.target_products)
            .map(|(scale, product)| scale * product)
            .sum::<f64>();
        let quadratic = scales
            .iter()
            .enumerate()
            .map(|(row, scale)| {
                scales
                    .iter()
                    .enumerate()
                    .map(|(column, other)| scale * self.gram[row * groups + column] * other)
                    .sum::<f64>()
            })
            .sum::<f64>();
        let raw_squared_error = self.target_squared - 2.0 * linear + quadratic;
        if !linear.is_finite() || !quadratic.is_finite() || !raw_squared_error.is_finite() {
            return Err(OutputReconstructionError::NonFiniteScaleRefit);
        }
        let squared_error = raw_squared_error.max(0.0);
        Ok(FixedTritScaleRefit {
            scales,
            squared_error,
            observations: self.observations,
        })
    }
}

/// Strictly validated legacy `TSV2OUT` v1 identity.
///
/// Version 1 predates runtime-comparable final-logit fields. It remains
/// inspectable for audit continuity but cannot satisfy execution admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LegacyOutputReconstructionReceipt {
    spec_id: [u8; 32],
    teacher_evidence_digest: [u8; 32],
    selected_candidate_id: [u8; 32],
    candidate_count: u32,
    receipt_id: [u8; 32],
}

impl LegacyOutputReconstructionReceipt {
    /// Frozen reconstruction specification identity.
    #[must_use]
    pub const fn spec_id(&self) -> &[u8; 32] {
        &self.spec_id
    }

    /// Shared teacher stream identity across the legacy restart set.
    #[must_use]
    pub const fn teacher_evidence_digest(&self) -> &[u8; 32] {
        &self.teacher_evidence_digest
    }

    /// Legacy winning candidate identity.
    #[must_use]
    pub const fn selected_candidate_id(&self) -> &[u8; 32] {
        &self.selected_candidate_id
    }

    /// Number of complete deterministic legacy candidates.
    #[must_use]
    pub const fn candidate_count(self) -> u32 {
        self.candidate_count
    }

    /// Legacy v1 receipt content identity.
    #[must_use]
    pub const fn receipt_id(&self) -> &[u8; 32] {
        &self.receipt_id
    }
}

impl OutputReconstructionReceipt {
    /// Frozen reconstruction specification identity.
    #[must_use]
    pub const fn spec_id(&self) -> &[u8; 32] {
        &self.spec_id
    }

    /// All candidates sorted by content identity, independent of evaluation order.
    #[must_use]
    pub fn candidates(&self) -> &[OutputCandidateReceipt] {
        &self.candidates
    }

    /// Winning candidate identity.
    #[must_use]
    pub const fn selected_candidate_id(&self) -> &[u8; 32] {
        &self.selected_candidate_id
    }

    /// Winning candidate receipt.
    #[must_use]
    pub fn selected(&self) -> &OutputCandidateReceipt {
        self.candidates
            .iter()
            .find(|candidate| candidate.candidate_id == self.selected_candidate_id)
            .expect("validated output-reconstruction receipt retains selected candidate")
    }

    /// Content identity of spec, teacher stream, all basins, and selection.
    #[must_use]
    pub const fn receipt_id(&self) -> &[u8; 32] {
        &self.receipt_id
    }
}

/// Select the lowest frozen objective across a complete deterministic restart set.
///
/// Evaluation order never affects candidate ordering or selection. Exact objective
/// ties resolve by candidate content identity.
///
/// # Errors
/// Rejects incomplete restart counts, provenance drift, duplicate candidates/seeds,
/// or candidate receipts produced from another specification.
pub fn select_output_reconstruction(
    spec: &OutputReconstructionSpec,
    mut candidates: Vec<OutputCandidateReceipt>,
) -> Result<OutputReconstructionReceipt, OutputReconstructionError> {
    if candidates.len() != spec.restarts {
        return Err(OutputReconstructionError::RestartCount {
            expected: spec.restarts,
            got: candidates.len(),
        });
    }
    let teacher_evidence_digest = candidates
        .first()
        .map(|candidate| candidate.teacher_evidence_digest)
        .ok_or(OutputReconstructionError::InvalidCount)?;
    let scope_evidence_mode = candidates
        .first()
        .is_some_and(|candidate| !candidate.scope_evidence.is_empty());
    let mut ids = BTreeSet::new();
    let mut seeds = BTreeSet::new();
    for candidate in &candidates {
        if candidate.spec_id != spec.spec_id
            || candidate.receipt_id != candidate.derive_id()
            || (!candidate.scope_evidence.is_empty()) != scope_evidence_mode
            || (!candidate.scope_evidence.is_empty()
                && !scope_evidence_matches_spec(spec, candidate))
        {
            return Err(OutputReconstructionError::CandidateSpecMismatch);
        }
        if candidate.teacher_evidence_digest != teacher_evidence_digest {
            return Err(OutputReconstructionError::TeacherEvidenceMismatch);
        }
        if !ids.insert(candidate.candidate_id) {
            return Err(OutputReconstructionError::DuplicateCandidate);
        }
        if !seeds.insert(candidate.initialization_seed) {
            return Err(OutputReconstructionError::DuplicateInitializationSeed);
        }
    }
    candidates.sort_by_key(|candidate| candidate.candidate_id);
    let selected_candidate_id = candidates
        .iter()
        .min_by(|left, right| {
            left.objective
                .total_cmp(&right.objective)
                .then_with(|| left.candidate_id.cmp(&right.candidate_id))
        })
        .map(|candidate| candidate.candidate_id)
        .ok_or(OutputReconstructionError::InvalidCount)?;
    let mut hasher = blake3::Hasher::new_derive_key(RECEIPT_HASH_CONTEXT);
    hasher.update(&spec.spec_id);
    hasher.update(&teacher_evidence_digest);
    hasher.update(&(candidates.len() as u64).to_le_bytes());
    for candidate in &candidates {
        hasher.update(&candidate.receipt_id);
    }
    hasher.update(&selected_candidate_id);
    let receipt_id = *hasher.finalize().as_bytes();
    Ok(OutputReconstructionReceipt {
        spec_id: spec.spec_id,
        teacher_evidence_digest,
        candidates,
        selected_candidate_id,
        receipt_id,
    })
}

fn scope_evidence_matches_spec(
    spec: &OutputReconstructionSpec,
    candidate: &OutputCandidateReceipt,
) -> bool {
    candidate.scope_evidence.len() == spec.scopes.len()
        && candidate
            .scope_evidence
            .iter()
            .zip(&spec.scopes)
            .all(|(evidence, expected)| {
                let expected = match expected {
                    OutputReconstructionScope::Block { start, end } => RuntimeOutputScope::Block {
                        start: *start,
                        end: *end,
                    },
                    OutputReconstructionScope::FinalLogits => RuntimeOutputScope::FinalLogits,
                };
                evidence.scope() == expected
                    && evidence.spec_id() == &candidate.spec_id
                    && evidence.candidate_id() == &candidate.candidate_id
                    && evidence.initialization_seed() == candidate.initialization_seed
                    && evidence.observation_count() == u64::from(spec.batches_per_scope)
                    && evidence.value_count() > 0
                    && evidence.digest() != &[0; 32]
            })
}

fn schedule_scopes(
    schedule: OutputReconstructionSchedule,
) -> Result<Vec<OutputReconstructionScope>, OutputReconstructionError> {
    let mut scopes = Vec::new();
    match schedule {
        OutputReconstructionSchedule::Blocks { block_count } => {
            if block_count == 0 {
                return Err(OutputReconstructionError::InvalidSchedule);
            }
            let scope_count = usize::try_from(block_count)
                .map_err(|_| OutputReconstructionError::CountOverflow)?
                .checked_add(1)
                .ok_or(OutputReconstructionError::CountOverflow)?;
            if scope_count > MAX_OUTPUT_RECONSTRUCTION_SCOPES {
                return Err(OutputReconstructionError::CountOverflow);
            }
            scopes
                .try_reserve_exact(scope_count)
                .map_err(|_| OutputReconstructionError::CountOverflow)?;
            for start in 0..block_count {
                scopes.push(OutputReconstructionScope::Block {
                    start,
                    end: start + 1,
                });
            }
        }
        OutputReconstructionSchedule::SlidingWindows {
            block_count,
            window_size,
            stride,
        } => {
            if block_count == 0 || window_size == 0 || window_size > block_count || stride == 0 {
                return Err(OutputReconstructionError::InvalidSchedule);
            }
            let tail_start = block_count - window_size;
            let window_count = if tail_start == 0 {
                1
            } else {
                u64::from(tail_start)
                    .checked_add(u64::from(stride) - 1)
                    .ok_or(OutputReconstructionError::CountOverflow)?
                    / u64::from(stride)
                    + 1
            };
            let scope_count = usize::try_from(
                window_count
                    .checked_add(1)
                    .ok_or(OutputReconstructionError::CountOverflow)?,
            )
            .map_err(|_| OutputReconstructionError::CountOverflow)?;
            if scope_count > MAX_OUTPUT_RECONSTRUCTION_SCOPES {
                return Err(OutputReconstructionError::CountOverflow);
            }
            scopes
                .try_reserve_exact(scope_count)
                .map_err(|_| OutputReconstructionError::CountOverflow)?;
            let mut start = 0;
            loop {
                scopes.push(OutputReconstructionScope::Block {
                    start,
                    end: start + window_size,
                });
                if start == tail_start {
                    break;
                }
                let next = start.saturating_add(stride);
                start = next.min(tail_start);
            }
        }
    }
    scopes.push(OutputReconstructionScope::FinalLogits);
    Ok(scopes)
}

fn hash_observation(
    hasher: &mut blake3::Hasher,
    scope: OutputReconstructionScope,
    batch_index: u32,
    rows: usize,
    columns: usize,
    mask: &[bool],
    values: &[f32],
) {
    match scope {
        OutputReconstructionScope::Block { start, end } => {
            hasher.update(&[1]);
            hasher.update(&start.to_le_bytes());
            hasher.update(&end.to_le_bytes());
        }
        OutputReconstructionScope::FinalLogits => {
            hasher.update(&[2]);
        }
    }
    hasher.update(&batch_index.to_le_bytes());
    hasher.update(&(rows as u64).to_le_bytes());
    hasher.update(&(columns as u64).to_le_bytes());
    for selected in mask {
        hasher.update(&[u8::from(*selected)]);
    }
    for value in values {
        hasher.update(&value.to_bits().to_le_bytes());
    }
}

fn hash_scope_evidence(hasher: &mut blake3::Hasher, evidence: &RuntimeOutputScopeEvidence) {
    match evidence.scope() {
        RuntimeOutputScope::Block { start, end } => {
            hasher.update(&[1]);
            hasher.update(&start.to_le_bytes());
            hasher.update(&end.to_le_bytes());
        }
        RuntimeOutputScope::FinalLogits => {
            hasher.update(&[2]);
            hasher.update(&0_u32.to_le_bytes());
            hasher.update(&0_u32.to_le_bytes());
        }
    }
    hasher.update(&evidence.observation_count().to_le_bytes());
    hasher.update(&evidence.value_count().to_le_bytes());
    hasher.update(evidence.digest());
}

fn distillation_losses(teacher: &[f32], student: &[f32], temperature: f64) -> (f64, f64) {
    let teacher_max = teacher
        .iter()
        .map(|value| f64::from(*value) / temperature)
        .fold(f64::NEG_INFINITY, f64::max);
    let student_max = student
        .iter()
        .map(|value| f64::from(*value) / temperature)
        .fold(f64::NEG_INFINITY, f64::max);
    let teacher_sum = teacher
        .iter()
        .map(|value| (f64::from(*value) / temperature - teacher_max).exp())
        .sum::<f64>();
    let student_sum = student
        .iter()
        .map(|value| (f64::from(*value) / temperature - student_max).exp())
        .sum::<f64>();
    let teacher_log_partition = teacher_max + teacher_sum.ln();
    let student_log_partition = student_max + student_sum.ln();
    let mut cross_entropy = 0.0;
    let mut kl = 0.0;
    for (teacher, student) in teacher.iter().zip(student) {
        let teacher_log_probability = f64::from(*teacher) / temperature - teacher_log_partition;
        let student_log_probability = f64::from(*student) / temperature - student_log_partition;
        let probability = teacher_log_probability.exp();
        cross_entropy -= probability * student_log_probability;
        kl += probability * (teacher_log_probability - student_log_probability);
    }
    (
        cross_entropy,
        canonical_zero((kl * temperature * temperature).max(0.0)),
    )
}

const fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

const fn map_runtime_evidence_error(error: RuntimeEvidenceError) -> OutputReconstructionError {
    match error {
        RuntimeEvidenceError::EmptyBatch => OutputReconstructionError::InvalidGeometry,
        RuntimeEvidenceError::NonFiniteOutput => {
            OutputReconstructionError::NonFiniteOutput { teacher: false }
        }
        RuntimeEvidenceError::CountOverflow => OutputReconstructionError::CountOverflow,
        RuntimeEvidenceError::EmptyStream => OutputReconstructionError::IncompleteCandidate,
        RuntimeEvidenceError::InvalidGeometry => OutputReconstructionError::InvalidGeometry,
        RuntimeEvidenceError::MissingIdentity => {
            OutputReconstructionError::MissingCandidateIdentity
        }
        RuntimeEvidenceError::EmptySelection => OutputReconstructionError::EmptyTokenSelection,
        RuntimeEvidenceError::InvalidBatchOrder => OutputReconstructionError::InvalidGeometry,
    }
}

/// Invalid output-reconstruction specification, stream, or candidate set.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum OutputReconstructionError {
    /// Objective weights or temperature are invalid.
    InvalidObjective,
    /// Source, activation, token, or validation identity is zero.
    MissingProvenance,
    /// Schedule geometry is invalid.
    InvalidSchedule,
    /// Batch or restart count is zero.
    InvalidCount,
    /// Candidate identity is zero.
    MissingCandidateIdentity,
    /// Scope or batch arrived outside canonical order.
    ScopeOrder {
        /// Required scope.
        expected: OutputReconstructionScope,
        /// Required batch ordinal.
        expected_batch: u32,
        /// Observed scope.
        got: OutputReconstructionScope,
        /// Observed batch ordinal.
        got_batch: u32,
    },
    /// Observation followed complete scheduled coverage.
    ExtraObservation,
    /// Rows, columns, mask, or tensor lengths disagree.
    InvalidGeometry,
    /// One teacher or student value was not finite.
    NonFiniteOutput {
        /// True for teacher output, false for student output.
        teacher: bool,
    },
    /// Observation selected no token rows.
    EmptyTokenSelection,
    /// Count or geometry arithmetic overflowed.
    CountOverflow,
    /// Candidate did not observe every required scope and batch.
    IncompleteCandidate,
    /// Weighted objective was not finite.
    NonFiniteObjective,
    /// Candidate count differs from frozen restart count.
    RestartCount {
        /// Required restart count.
        expected: usize,
        /// Supplied candidate count.
        got: usize,
    },
    /// Candidate belongs to another spec or its receipt identity changed.
    CandidateSpecMismatch,
    /// Teacher bytes differ across restart evaluations.
    TeacherEvidenceMismatch,
    /// Candidate content identity is duplicated.
    DuplicateCandidate,
    /// Initialization seed is duplicated.
    DuplicateInitializationSeed,
    /// Canonical receipt exceeds its bounded format.
    ReceiptTooLarge,
    /// Canonical receipt allocation failed.
    ReceiptAllocationFailed,
    /// Canonical receipt bytes are malformed or noncanonical.
    MalformedReceipt(&'static str),
    /// A valid legacy v1 receipt lacks runtime-comparable final-logit evidence.
    LegacyReceiptMissingRuntimeEvidence,
    /// Fixed-trit scale refit dimensions are empty, excessive, or have no observations.
    InvalidScaleRefit,
    /// Fixed-trit scale refit accumulated a non-finite intermediate or result.
    NonFiniteScaleRefit,
}

impl fmt::Display for OutputReconstructionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidObjective => {
                formatter.write_str("output-reconstruction objective is invalid")
            }
            Self::MissingProvenance => {
                formatter.write_str("output-reconstruction provenance is missing")
            }
            Self::InvalidSchedule => {
                formatter.write_str("output-reconstruction schedule is invalid")
            }
            Self::InvalidCount => formatter.write_str("output-reconstruction count is invalid"),
            Self::MissingCandidateIdentity => {
                formatter.write_str("output-reconstruction candidate identity is missing")
            }
            Self::ScopeOrder { .. } => {
                formatter.write_str("output-reconstruction observation order differs")
            }
            Self::ExtraObservation => {
                formatter.write_str("output-reconstruction has an extra observation")
            }
            Self::InvalidGeometry => {
                formatter.write_str("output-reconstruction observation geometry differs")
            }
            Self::NonFiniteOutput { teacher } => write!(
                formatter,
                "output-reconstruction {} output is not finite",
                if *teacher { "teacher" } else { "student" }
            ),
            Self::EmptyTokenSelection => {
                formatter.write_str("output-reconstruction batch selects no tokens")
            }
            Self::CountOverflow => formatter.write_str("output-reconstruction count overflow"),
            Self::IncompleteCandidate => {
                formatter.write_str("output-reconstruction candidate is incomplete")
            }
            Self::NonFiniteObjective => {
                formatter.write_str("output-reconstruction objective is not finite")
            }
            Self::RestartCount { expected, got } => write!(
                formatter,
                "output-reconstruction needs {expected} restarts, received {got}"
            ),
            Self::CandidateSpecMismatch => {
                formatter.write_str("output-reconstruction candidate spec differs")
            }
            Self::TeacherEvidenceMismatch => formatter
                .write_str("output-reconstruction teacher evidence differs across restarts"),
            Self::DuplicateCandidate => {
                formatter.write_str("output-reconstruction candidate is duplicated")
            }
            Self::DuplicateInitializationSeed => {
                formatter.write_str("output-reconstruction initialization seed is duplicated")
            }
            Self::ReceiptTooLarge => {
                formatter.write_str("output-reconstruction receipt is too large")
            }
            Self::ReceiptAllocationFailed => {
                formatter.write_str("output-reconstruction receipt allocation failed")
            }
            Self::MalformedReceipt(field) => {
                write!(
                    formatter,
                    "output-reconstruction receipt {field} is malformed"
                )
            }
            Self::LegacyReceiptMissingRuntimeEvidence => formatter.write_str(
                "legacy TSV2OUT v1 receipt has no runtime-comparable final-logit evidence",
            ),
            Self::InvalidScaleRefit => formatter.write_str("fixed-trit scale refit is invalid"),
            Self::NonFiniteScaleRefit => {
                formatter.write_str("fixed-trit scale refit became non-finite")
            }
        }
    }
}

impl std::error::Error for OutputReconstructionError {}
