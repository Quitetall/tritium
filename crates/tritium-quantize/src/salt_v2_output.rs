//! Streamed block-output and teacher-logit reconstruction objectives.

use core::fmt;
use std::collections::BTreeSet;

use tritium_format::ModelId;

mod codec;

const SPEC_HASH_CONTEXT: &str = "tritium salt v2 output reconstruction spec v1";
const TEACHER_HASH_CONTEXT: &str = "tritium salt v2 output reconstruction teacher v1";
const STUDENT_HASH_CONTEXT: &str = "tritium salt v2 output reconstruction student v1";
const CANDIDATE_HASH_CONTEXT: &str = "tritium salt v2 output reconstruction candidate v1";
const RECEIPT_HASH_CONTEXT: &str = "tritium salt v2 output reconstruction receipt v1";
const MAX_OUTPUT_RECONSTRUCTION_SCOPES: usize = 1 << 20;

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
    student_hasher: blake3::Hasher,
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
        let mut student_hasher = blake3::Hasher::new_derive_key(STUDENT_HASH_CONTEXT);
        teacher_hasher.update(spec.spec_id());
        student_hasher.update(spec.spec_id());
        student_hasher.update(&candidate_id);
        student_hasher.update(&initialization_seed.to_le_bytes());
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
            student_hasher,
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
        hash_observation(
            &mut self.student_hasher,
            scope,
            batch_index,
            rows,
            columns,
            mask,
            student,
        );
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

    /// Seal exact aggregate losses and streamed evidence identities.
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
        let student_output_digest = *self.student_hasher.finalize().as_bytes();
        let mut receipt = OutputCandidateReceipt {
            spec_id: self.spec.spec_id,
            candidate_id: self.candidate_id,
            initialization_seed: self.initialization_seed,
            teacher_evidence_digest,
            student_output_digest,
            observations: self.observations,
            block_elements: self.block_elements,
            final_tokens: self.final_tokens,
            block_output_mse: canonical_zero(block_output_mse),
            teacher_cross_entropy: canonical_zero(teacher_cross_entropy),
            teacher_kl: canonical_zero(teacher_kl),
            objective: canonical_zero(objective),
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
    observations: u64,
    block_elements: u64,
    final_tokens: u64,
    block_output_mse: f64,
    teacher_cross_entropy: f64,
    teacher_kl: f64,
    objective: f64,
    receipt_id: [u8; 32],
}

impl OutputCandidateReceipt {
    /// Candidate content identity supplied by the production initializer.
    #[must_use]
    pub const fn candidate_id(&self) -> &[u8; 32] {
        &self.candidate_id
    }

    /// Exact teacher stream identity shared by every valid restart.
    #[must_use]
    pub const fn teacher_evidence_digest(&self) -> &[u8; 32] {
        &self.teacher_evidence_digest
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
        let mut hasher = blake3::Hasher::new_derive_key(CANDIDATE_HASH_CONTEXT);
        hasher.update(&self.spec_id);
        hasher.update(&self.candidate_id);
        hasher.update(&self.initialization_seed.to_le_bytes());
        hasher.update(&self.teacher_evidence_digest);
        hasher.update(&self.student_output_digest);
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

impl OutputReconstructionReceipt {
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
    let mut ids = BTreeSet::new();
    let mut seeds = BTreeSet::new();
    for candidate in &candidates {
        if candidate.spec_id != spec.spec_id || candidate.receipt_id != candidate.derive_id() {
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
        }
    }
}

impl std::error::Error for OutputReconstructionError {}
