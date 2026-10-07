//! Streamed block-output and teacher-logit reconstruction objectives.

use core::fmt;
use std::collections::BTreeSet;

use crate::salt_v2_activation::{ActivationCache, ActivationCacheError, ActivationWindow};
use tritium_core::Trit;
use tritium_format::{
    ModelId, RuntimeEvidenceError, RuntimeFinalLogitsAccumulator,
    RuntimeOutputReconstructionAccumulator, RuntimeOutputScope, RuntimeOutputScopeAccumulator,
    RuntimeOutputScopeEvidence,
    salt_v2::SaltV2Codec,
    salt_v2_package::{
        PackedSaltV2PlaneRef, SALT_V2_ALLOCATION_TILE_SIZE, SALT_V2_SCALE_GROUP_SIZE,
        SALT_V2_SCALE_GROUP_SIZE_64, SALT_V2_SCALE_GROUP_SIZE_256, SaltV2ScaleUpdate,
        unpack_salt_v2_plane,
    },
};

mod codec;

const SPEC_HASH_CONTEXT: &str = "tritium salt v2 output reconstruction spec v1";
const TEACHER_HASH_CONTEXT: &str = "tritium salt v2 output reconstruction teacher v1";
const CANDIDATE_HASH_CONTEXT: &str = "tritium salt v2 output reconstruction candidate v1";
const CANDIDATE_HASH_CONTEXT_V3: &str = "tritium salt v2 output reconstruction candidate v2";
const SCALE_UPDATE_HASH_CONTEXT: &str = "tritium salt v2 output reconstruction scale updates v1";
const SCALE_CANDIDATE_HASH_CONTEXT: &str =
    "tritium salt v2 output reconstruction scale candidate v1";
const SCALE_REFIT_START_CONTEXT: &str = "tritium salt v2 fixed-trit scale-refit start v1";
const ACTIVATION_SET_HASH_CONTEXT: &str = "tritium salt v2 output reconstruction activation set v1";
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

    /// Content identity of the exact ordered layer activation-cache set.
    #[must_use]
    pub const fn activation_digest(&self) -> &[u8; 32] {
        &self.activation_digest
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

    /// Derive an output-reconstruction candidate identity from its exact SALT
    /// V2 scale updates, the exact parent-package digest, this frozen spec, and
    /// its deterministic initialization seed. Updates must be non-empty and
    /// strictly ordered by `(tensor, tile, plane)`; every scale must be finite
    /// and positive.
    ///
    /// The returned identity can be passed to [`OutputReconstructionAccumulator::new`]
    /// so the resulting output receipt names the exact scale mutation evaluated.
    ///
    /// # Errors
    /// Rejects empty, unordered, duplicate, or invalid updates and overflowing
    /// platform-sized indices/counts.
    pub fn candidate_id_for_scale_updates(
        &self,
        parent_package_digest: &[u8; 32],
        initialization_seed: u64,
        updates: &[SaltV2ScaleUpdate],
    ) -> Result<[u8; 32], OutputReconstructionError> {
        if parent_package_digest == &[0; 32] {
            return Err(OutputReconstructionError::MissingPackageIdentity);
        }
        if updates.is_empty() {
            return Err(OutputReconstructionError::EmptyScaleUpdateSet);
        }

        let mut update_hasher = blake3::Hasher::new_derive_key(SCALE_UPDATE_HASH_CONTEXT);
        update_hasher.update(
            &u64::try_from(updates.len())
                .map_err(|_| OutputReconstructionError::CountOverflow)?
                .to_le_bytes(),
        );
        let mut previous_target = None;
        for update in updates {
            let target = (
                update.tensor_index(),
                update.tile_index(),
                update.plane_index(),
            );
            if previous_target.is_some_and(|previous| target <= previous) {
                return Err(OutputReconstructionError::NonCanonicalScaleUpdateOrder);
            }
            previous_target = Some(target);

            update_hasher.update(
                &u64::try_from(target.0)
                    .map_err(|_| OutputReconstructionError::CountOverflow)?
                    .to_le_bytes(),
            );
            update_hasher.update(
                &u64::try_from(target.1)
                    .map_err(|_| OutputReconstructionError::CountOverflow)?
                    .to_le_bytes(),
            );
            update_hasher.update(
                &u64::try_from(target.2)
                    .map_err(|_| OutputReconstructionError::CountOverflow)?
                    .to_le_bytes(),
            );
            update_hasher.update(
                &u64::try_from(update.scales().len())
                    .map_err(|_| OutputReconstructionError::CountOverflow)?
                    .to_le_bytes(),
            );
            for scale in update.scales() {
                let value = scale.to_f32();
                if !value.is_finite() || value <= 0.0 {
                    return Err(OutputReconstructionError::InvalidScaleUpdate);
                }
                update_hasher.update(&scale.to_bits().to_le_bytes());
            }
        }

        let update_id = update_hasher.finalize();
        let mut candidate_hasher = blake3::Hasher::new_derive_key(SCALE_CANDIDATE_HASH_CONTEXT);
        candidate_hasher.update(&self.spec_id);
        candidate_hasher.update(parent_package_digest);
        candidate_hasher.update(&initialization_seed.to_le_bytes());
        candidate_hasher.update(update_id.as_bytes());
        let candidate_id = *candidate_hasher.finalize().as_bytes();
        if candidate_id == [0; 32] {
            return Err(OutputReconstructionError::MissingCandidateIdentity);
        }
        Ok(candidate_id)
    }

    /// Bind a scale-update slice to its parent package and output-evaluation spec.
    /// The returned borrowed candidate keeps the exact updates paired with the
    /// identity used by the output receipt without copying the update payload.
    ///
    /// # Errors
    /// Returns the same errors as [`Self::candidate_id_for_scale_updates`].
    pub fn scale_update_candidate<'a>(
        &self,
        parent_package_digest: &[u8; 32],
        initialization_seed: u64,
        updates: &'a [SaltV2ScaleUpdate],
    ) -> Result<OutputReconstructionScaleCandidate<'a>, OutputReconstructionError> {
        let candidate_id = self.candidate_id_for_scale_updates(
            parent_package_digest,
            initialization_seed,
            updates,
        )?;
        Ok(OutputReconstructionScaleCandidate {
            spec_id: self.spec_id,
            parent_package_digest: *parent_package_digest,
            initialization_seed,
            updates,
            candidate_id,
        })
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

/// Content metadata for one per-layer activation cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OutputReconstructionActivationLayer {
    /// Zero-based transformer block index.
    pub layer_index: u32,
    /// Digest of the exact canonical cache bytes.
    pub cache_digest: [u8; 32],
    /// Digest of cache layer, tensor, shape, dtype, source, and shard policy.
    pub schema_digest: [u8; 32],
    /// Shared token-source provenance envelope.
    pub source_digest: [u8; 32],
    /// Number of token rows in this cache.
    pub total_tokens: u64,
    /// Feature width of each token row.
    pub feature_width: u64,
}

/// Storage seam for reading layer activation windows without loading all caches.
///
/// A disk-backed adapter can retain only this layer metadata and open/decode one
/// cache at a time in `read_layer_window`; it need not keep the model-wide cache
/// set resident in memory.
pub trait OutputReconstructionActivationSource {
    /// Number of layer caches represented by the source.
    fn layer_count(&self) -> usize;

    /// Metadata for one canonical zero-based layer index.
    fn layer_metadata(&self, layer_index: u32) -> Option<OutputReconstructionActivationLayer>;

    /// Read one bounded interval from one layer's cache.
    fn read_layer_window(
        &self,
        layer_index: u32,
        token_start: u64,
        token_count: u64,
        max_decoded_bytes: u64,
    ) -> Result<ActivationWindow, OutputReconstructionError>;
}

/// Compute the stable digest of canonical layer activation metadata.
///
/// This does not load cache payloads. Metadata must describe layers in exactly
/// `0..layer_count`, with matching token count, feature width, and provenance.
///
/// # Errors
/// Rejects empty, unordered, duplicated, or shape/provenance-inconsistent sets.
pub fn output_reconstruction_activation_digest<S: OutputReconstructionActivationSource + ?Sized>(
    source: &S,
) -> Result<[u8; 32], OutputReconstructionError> {
    validate_activation_source(source)?;
    let mut hasher = blake3::Hasher::new_derive_key(ACTIVATION_SET_HASH_CONTEXT);
    hasher.update(
        &u64::try_from(source.layer_count())
            .map_err(|_| OutputReconstructionError::CountOverflow)?
            .to_le_bytes(),
    );
    for ordinal in 0..source.layer_count() {
        let layer_index =
            u32::try_from(ordinal).map_err(|_| OutputReconstructionError::CountOverflow)?;
        let metadata = source
            .layer_metadata(layer_index)
            .ok_or(OutputReconstructionError::InvalidActivationCacheSet)?;
        hasher.update(&metadata.layer_index.to_le_bytes());
        hasher.update(&metadata.schema_digest);
        hasher.update(&metadata.source_digest);
        hasher.update(&metadata.total_tokens.to_le_bytes());
        hasher.update(&metadata.feature_width.to_le_bytes());
        hasher.update(&metadata.cache_digest);
    }
    Ok(*hasher.finalize().as_bytes())
}

impl OutputReconstructionActivationSource for [ActivationCache] {
    fn layer_count(&self) -> usize {
        self.len()
    }

    fn layer_metadata(&self, layer_index: u32) -> Option<OutputReconstructionActivationLayer> {
        let cache = self.get(usize::try_from(layer_index).ok()?)?;
        Some(OutputReconstructionActivationLayer {
            layer_index: cache.spec().layer_index(),
            cache_digest: cache.digest().into_bytes(),
            schema_digest: cache.spec().schema_digest().into_bytes(),
            source_digest: cache.spec().source_digest().into_bytes(),
            total_tokens: cache.spec().total_tokens(),
            feature_width: cache.spec().feature_width(),
        })
    }

    fn read_layer_window(
        &self,
        layer_index: u32,
        token_start: u64,
        token_count: u64,
        max_decoded_bytes: u64,
    ) -> Result<ActivationWindow, OutputReconstructionError> {
        self.get(
            usize::try_from(layer_index)
                .map_err(|_| OutputReconstructionError::InvalidActivationCacheSet)?,
        )
        .ok_or(OutputReconstructionError::InvalidActivationCacheSet)?
        .read_window(token_start, token_count, max_decoded_bytes)
        .map_err(OutputReconstructionError::ActivationCache)
    }
}

/// Ordered, provenance-checked activation source for B3 evaluation.
pub struct OutputReconstructionActivationSet<'a, S: OutputReconstructionActivationSource + ?Sized> {
    spec: &'a OutputReconstructionSpec,
    source: &'a S,
}

impl<S: OutputReconstructionActivationSource + ?Sized> fmt::Debug
    for OutputReconstructionActivationSet<'_, S>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OutputReconstructionActivationSet")
            .field("spec_id", self.spec.spec_id())
            .field("layer_count", &self.source.layer_count())
            .finish()
    }
}

impl<'a, S: OutputReconstructionActivationSource + ?Sized>
    OutputReconstructionActivationSet<'a, S>
{
    /// Bind an activation source to a frozen output-reconstruction spec.
    ///
    /// The source can be disk-backed, so only a sliding window of activation
    /// payloads needs to be decoded while all-layer provenance remains bound.
    ///
    /// # Errors
    /// Rejects mismatched layer count, activation identity, or token provenance.
    pub fn new(
        spec: &'a OutputReconstructionSpec,
        source: &'a S,
    ) -> Result<OutputReconstructionActivationSet<'a, S>, OutputReconstructionError> {
        validate_activation_source(source)?;
        if u64::try_from(source.layer_count()).ok()
            != Some(u64::from(schedule_block_count(spec.schedule)))
        {
            return Err(OutputReconstructionError::InvalidActivationCacheSet);
        }
        let first_layer = source
            .layer_metadata(0)
            .ok_or(OutputReconstructionError::InvalidActivationCacheSet)?;
        if &first_layer.source_digest != spec.token_stream_digest() {
            return Err(OutputReconstructionError::ActivationSetIdentityMismatch);
        }
        if output_reconstruction_activation_digest(source)? != *spec.activation_digest() {
            return Err(OutputReconstructionError::ActivationSetIdentityMismatch);
        }
        Ok(Self { spec, source })
    }

    /// Read one scheduled block/window under one total decoded-payload budget.
    ///
    /// Every layer cache must report identical token masks and sequence
    /// boundaries for the requested interval. The source is asked for one layer
    /// at a time; a file-backed source may release each encoded cache immediately.
    ///
    /// # Errors
    /// Rejects an unscheduled/non-block scope, inconsistent layer windows,
    /// exceeded budget, invalid range, or a corrupt cache.
    pub fn read_window(
        &self,
        scope: OutputReconstructionScope,
        token_start: u64,
        token_count: u64,
        max_decoded_bytes: u64,
    ) -> Result<OutputReconstructionActivationWindows, OutputReconstructionError> {
        let OutputReconstructionScope::Block { start, end } = scope else {
            return Err(OutputReconstructionError::InvalidActivationWindowScope);
        };
        if !self.spec.scopes().contains(&scope) || start >= end {
            return Err(OutputReconstructionError::InvalidActivationWindowScope);
        }
        let block_count =
            usize::try_from(end - start).map_err(|_| OutputReconstructionError::CountOverflow)?;
        let mut windows = Vec::new();
        windows
            .try_reserve_exact(block_count)
            .map_err(|_| OutputReconstructionError::ReceiptAllocationFailed)?;
        let mut decoded_bytes = 0_u64;
        for layer_index in start..end {
            let remaining = max_decoded_bytes
                .checked_sub(decoded_bytes)
                .ok_or(OutputReconstructionError::ActivationWindowBudgetExceeded)?;
            let window =
                self.source
                    .read_layer_window(layer_index, token_start, token_count, remaining)?;
            if let Some(first) = windows.first()
                && !activation_windows_align(first, &window)
            {
                return Err(OutputReconstructionError::ActivationWindowMismatch);
            }
            decoded_bytes = decoded_bytes
                .checked_add(window.decoded_byte_estimate())
                .ok_or(OutputReconstructionError::CountOverflow)?;
            windows.push(window);
        }
        Ok(OutputReconstructionActivationWindows {
            first_block: start,
            windows,
            decoded_bytes,
        })
    }
}

/// Bounded decoded activation windows for one scheduled block range.
#[derive(Clone, Debug, PartialEq)]
pub struct OutputReconstructionActivationWindows {
    first_block: u32,
    windows: Vec<ActivationWindow>,
    decoded_bytes: u64,
}

impl OutputReconstructionActivationWindows {
    /// First block represented by this window set.
    #[must_use]
    pub const fn first_block(&self) -> u32 {
        self.first_block
    }

    /// Number of blocks represented by this window set.
    #[must_use]
    pub fn block_count(&self) -> u32 {
        u32::try_from(self.windows.len()).unwrap_or(u32::MAX)
    }

    /// Decoded payload estimate charged against the caller's total budget.
    #[must_use]
    pub const fn decoded_bytes(&self) -> u64 {
        self.decoded_bytes
    }

    /// Activation rows for one absolute block index in this range.
    #[must_use]
    pub fn layer(&self, block_index: u32) -> Option<&ActivationWindow> {
        let offset = block_index.checked_sub(self.first_block)?;
        self.windows.get(usize::try_from(offset).ok()?)
    }
}

fn validate_activation_source<S: OutputReconstructionActivationSource + ?Sized>(
    source: &S,
) -> Result<(), OutputReconstructionError> {
    let count = source.layer_count();
    if count == 0 {
        return Err(OutputReconstructionError::InvalidActivationCacheSet);
    }
    let first = source
        .layer_metadata(0)
        .ok_or(OutputReconstructionError::InvalidActivationCacheSet)?;
    for ordinal in 0..count {
        let layer_index =
            u32::try_from(ordinal).map_err(|_| OutputReconstructionError::CountOverflow)?;
        let metadata = source
            .layer_metadata(layer_index)
            .ok_or(OutputReconstructionError::InvalidActivationCacheSet)?;
        if metadata.layer_index != layer_index
            || metadata.source_digest != first.source_digest
            || metadata.total_tokens == 0
            || metadata.total_tokens != first.total_tokens
            || metadata.feature_width == 0
            || metadata.feature_width != first.feature_width
        {
            return Err(OutputReconstructionError::InvalidActivationCacheSet);
        }
    }
    Ok(())
}

fn schedule_block_count(schedule: OutputReconstructionSchedule) -> u32 {
    match schedule {
        OutputReconstructionSchedule::Blocks { block_count }
        | OutputReconstructionSchedule::SlidingWindows { block_count, .. } => block_count,
    }
}

fn activation_windows_align(first: &ActivationWindow, other: &ActivationWindow) -> bool {
    first.token_start() == other.token_start()
        && first.token_count() == other.token_count()
        && first.feature_width() == other.feature_width()
        && first.token_mask() == other.token_mask()
        && first.sequence_ends() == other.sequence_ends()
}

/// Borrowed, content-bound SALT scale candidate for output evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutputReconstructionScaleCandidate<'a> {
    spec_id: [u8; 32],
    parent_package_digest: [u8; 32],
    initialization_seed: u64,
    updates: &'a [SaltV2ScaleUpdate],
    candidate_id: [u8; 32],
}

impl OutputReconstructionScaleCandidate<'_> {
    /// Frozen output-reconstruction spec identity for this candidate.
    #[must_use]
    pub const fn spec_id(&self) -> &[u8; 32] {
        &self.spec_id
    }

    /// Exact-byte identity of the package to which the updates apply.
    #[must_use]
    pub const fn parent_package_digest(&self) -> &[u8; 32] {
        &self.parent_package_digest
    }

    /// Deterministic initialization seed included in the candidate identity.
    #[must_use]
    pub const fn initialization_seed(&self) -> u64 {
        self.initialization_seed
    }

    /// Canonical scale updates whose exact f16 payloads are hashed into this candidate.
    #[must_use]
    pub fn updates(&self) -> &[SaltV2ScaleUpdate] {
        self.updates
    }

    /// Candidate identity to use when recording output-reconstruction evidence.
    #[must_use]
    pub const fn candidate_id(&self) -> &[u8; 32] {
        &self.candidate_id
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
    /// Begin evaluating a content-bound scale candidate.
    ///
    /// # Errors
    /// Rejects candidates created for another output-reconstruction spec.
    pub fn for_scale_candidate(
        spec: &OutputReconstructionSpec,
        candidate: &OutputReconstructionScaleCandidate<'_>,
    ) -> Result<Self, OutputReconstructionError> {
        if candidate.spec_id != *spec.spec_id() {
            return Err(OutputReconstructionError::CandidateSpecMismatch);
        }
        Self::new(spec, candidate.candidate_id, candidate.initialization_seed)
    }

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
    projection_scratch: Vec<f64>,
    initial_scales: Vec<f64>,
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

/// Bounded streaming refit for one fixed-trit tile-plane across activation windows.
///
/// The accumulator retains only the small scale-group Gram system. Callers can
/// read one activation/teacher-output window, observe it, and release its dense
/// rows before reading the next window.
#[derive(Clone, Debug)]
pub struct FixedTritTileScaleRefitAccumulator {
    output_width: usize,
    tile_index: usize,
    trits: Vec<Trit>,
    scale_group_size: usize,
    input_width: Option<usize>,
    fit: FixedTritScaleRefitAccumulator,
}

impl FixedTritScaleRefit {
    /// Fitted non-negative scale for each fixed-trit group.
    #[must_use]
    pub fn scales(&self) -> &[f64] {
        &self.scales
    }

    /// Convert fitted scales to the package's stored f16 precision.
    ///
    /// Package application still validates group-specific zero-scale
    /// constraints against the fixed trits.
    ///
    /// # Errors
    /// Rejects an invalid or non-finite f64 scale or one that overflows f16.
    pub fn to_f16_scales(&self) -> Result<Vec<half::f16>, OutputReconstructionError> {
        let mut scales = Vec::new();
        scales
            .try_reserve_exact(self.scales.len())
            .map_err(|_| OutputReconstructionError::ReceiptAllocationFailed)?;
        for &scale in &self.scales {
            if !scale.is_finite() || scale < 0.0 {
                return Err(OutputReconstructionError::NonFiniteScaleRefit);
            }
            let stored = half::f16::from_f64(scale);
            if !stored.is_finite() || stored.to_bits() & 0x8000 != 0 {
                return Err(OutputReconstructionError::ScaleNotRepresentable);
            }
            scales.push(stored);
        }
        Ok(scales)
    }

    /// Convert scales to strictly positive f16 values for output candidates.
    ///
    /// Exact zeros become the smallest positive f16 subnormal. This preserves
    /// the candidate identity contract; downstream package application and
    /// output scoring still validate the resulting artifact.
    ///
    /// # Errors
    /// Rejects an invalid or non-finite f64 scale or one that overflows f16.
    pub fn to_positive_f16_scales(&self) -> Result<Vec<half::f16>, OutputReconstructionError> {
        let mut scales = self.to_f16_scales()?;
        let smallest_positive = half::f16::from_bits(1);
        for scale in &mut scales {
            if *scale == half::f16::ZERO {
                *scale = smallest_positive;
            }
        }
        Ok(scales)
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
        Self::new_with_initial_scales(group_count, coordinate_sweeps, None)
    }

    fn new_with_initial_scales(
        group_count: usize,
        coordinate_sweeps: usize,
        initial_scales: Option<&[f64]>,
    ) -> Result<Self, OutputReconstructionError> {
        if group_count == 0
            || group_count > MAX_FIXED_TRIT_REFIT_GROUPS
            || coordinate_sweeps == 0
            || coordinate_sweeps > MAX_FIXED_TRIT_REFIT_SWEEPS
            || initial_scales.is_some_and(|scales| {
                scales.len() != group_count
                    || scales
                        .iter()
                        .any(|scale| !scale.is_finite() || *scale < 0.0)
            })
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
        let mut projection_scratch = Vec::new();
        projection_scratch
            .try_reserve_exact(group_count)
            .map_err(|_| OutputReconstructionError::ReceiptAllocationFailed)?;
        projection_scratch.resize(group_count, 0.0);
        let mut initial_scales_owned = Vec::new();
        initial_scales_owned
            .try_reserve_exact(group_count)
            .map_err(|_| OutputReconstructionError::ReceiptAllocationFailed)?;
        if let Some(scales) = initial_scales {
            initial_scales_owned.extend_from_slice(scales);
        } else {
            initial_scales_owned.resize(group_count, 0.0);
        }
        Ok(Self {
            gram,
            target_products,
            projection_scratch,
            initial_scales: initial_scales_owned,
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

    /// Stream one activation row through the fixed trits and fit its teacher output.
    ///
    /// The trit and activation slices are row-major for one output channel;
    /// each `scale_group_size`-wide segment contributes one feature to the
    /// non-negative scale fit. Scratch storage is allocated once by the
    /// accumulator and reused for every observation.
    ///
    /// # Errors
    /// Rejects mismatched dimensions, invalid group width, non-finite values,
    /// or counter/accumulator overflow without partially accepting the row.
    pub fn observe_fixed_trit_projection(
        &mut self,
        trits: &[Trit],
        activations: &[f32],
        scale_group_size: usize,
        teacher_output: f64,
    ) -> Result<(), OutputReconstructionError> {
        if trits.is_empty()
            || trits.len() != activations.len()
            || scale_group_size == 0
            || trits.len().div_ceil(scale_group_size) != self.target_products.len()
        {
            return Err(OutputReconstructionError::InvalidGeometry);
        }
        if !teacher_output.is_finite() {
            return Err(OutputReconstructionError::NonFiniteOutput { teacher: true });
        }
        if activations.iter().any(|value| !value.is_finite()) {
            return Err(OutputReconstructionError::NonFiniteOutput { teacher: false });
        }

        let mut projected = std::mem::take(&mut self.projection_scratch);
        let result = (|| {
            for (group, output) in projected.iter_mut().enumerate() {
                let start = group * scale_group_size;
                let end = (start + scale_group_size).min(trits.len());
                *output = trits[start..end]
                    .iter()
                    .zip(&activations[start..end])
                    .map(|(trit, activation)| f64::from(trit.get()) * f64::from(*activation))
                    .sum();
                if !output.is_finite() {
                    return Err(OutputReconstructionError::NonFiniteScaleRefit);
                }
            }
            self.observe(&projected, teacher_output)
        })();
        self.projection_scratch = projected;
        result
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
        let mut scales = self.initial_scales;
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

impl FixedTritTileScaleRefitAccumulator {
    /// Start a streaming fit for one package allocation tile-plane.
    ///
    /// # Errors
    /// Rejects invalid dimensions, unsupported scale-group widths, or allocation
    /// failure.
    pub fn new(
        output_width: usize,
        tile_index: usize,
        trits: &[Trit],
        scale_group_size: usize,
        coordinate_sweeps: usize,
    ) -> Result<Self, OutputReconstructionError> {
        Self::new_with_initial_scales(
            output_width,
            tile_index,
            trits,
            scale_group_size,
            coordinate_sweeps,
            None,
        )
    }

    fn new_with_initial_scales(
        output_width: usize,
        tile_index: usize,
        trits: &[Trit],
        scale_group_size: usize,
        coordinate_sweeps: usize,
        initial_scales: Option<&[f64]>,
    ) -> Result<Self, OutputReconstructionError> {
        if output_width == 0
            || trits.is_empty()
            || trits.len() > SALT_V2_ALLOCATION_TILE_SIZE
            || !matches!(
                scale_group_size,
                SALT_V2_SCALE_GROUP_SIZE_64
                    | SALT_V2_SCALE_GROUP_SIZE
                    | SALT_V2_SCALE_GROUP_SIZE_256
            )
        {
            return Err(OutputReconstructionError::InvalidGeometry);
        }
        let group_count = trits.len().div_ceil(scale_group_size);
        let fit = FixedTritScaleRefitAccumulator::new_with_initial_scales(
            group_count,
            coordinate_sweeps,
            initial_scales,
        )?;
        let mut owned_trits = Vec::new();
        owned_trits
            .try_reserve_exact(trits.len())
            .map_err(|_| OutputReconstructionError::ReceiptAllocationFailed)?;
        owned_trits.extend_from_slice(trits);
        Ok(Self {
            output_width,
            tile_index,
            trits: owned_trits,
            scale_group_size,
            input_width: None,
            fit,
        })
    }

    /// Add one bounded activation window and its teacher residual outputs.
    ///
    /// Windows for one accumulator must share the same feature width. The
    /// observation is atomic: malformed values or numeric overflow leave the
    /// accumulated fit unchanged.
    ///
    /// # Errors
    /// Rejects shape/provenance geometry drift or non-finite values.
    pub fn observe_window(
        &mut self,
        activations: &ActivationWindow,
        residual_outputs: &[f32],
    ) -> Result<(), OutputReconstructionError> {
        let input_width = usize::try_from(activations.feature_width())
            .map_err(|_| OutputReconstructionError::InvalidGeometry)?;
        let token_count = usize::try_from(activations.token_count())
            .map_err(|_| OutputReconstructionError::InvalidGeometry)?;
        let total_coefficients = self
            .output_width
            .checked_mul(input_width)
            .ok_or(OutputReconstructionError::InvalidGeometry)?;
        let target_values = token_count
            .checked_mul(self.output_width)
            .ok_or(OutputReconstructionError::InvalidGeometry)?;
        let activation_values = token_count
            .checked_mul(input_width)
            .ok_or(OutputReconstructionError::InvalidGeometry)?;
        if input_width == 0
            || token_count == 0
            || activations.values().len() != activation_values
            || activations.token_mask().len() != token_count
            || residual_outputs.len() != target_values
        {
            return Err(OutputReconstructionError::InvalidGeometry);
        }
        if self.input_width.is_some_and(|width| width != input_width) {
            return Err(OutputReconstructionError::InvalidGeometry);
        }
        if activations.values().iter().any(|value| !value.is_finite()) {
            return Err(OutputReconstructionError::NonFiniteOutput { teacher: false });
        }
        if residual_outputs.iter().any(|value| !value.is_finite()) {
            return Err(OutputReconstructionError::NonFiniteOutput { teacher: true });
        }

        let tile_start = self
            .tile_index
            .checked_mul(SALT_V2_ALLOCATION_TILE_SIZE)
            .ok_or(OutputReconstructionError::InvalidGeometry)?;
        let tile_end = tile_start
            .checked_add(self.trits.len())
            .ok_or(OutputReconstructionError::InvalidGeometry)?;
        let expected_tile_len = total_coefficients
            .checked_sub(tile_start)
            .ok_or(OutputReconstructionError::InvalidGeometry)?
            .min(SALT_V2_ALLOCATION_TILE_SIZE);
        if tile_start >= total_coefficients
            || tile_end > total_coefficients
            || self.trits.len() != expected_tile_len
        {
            return Err(OutputReconstructionError::InvalidGeometry);
        }

        let mut candidate_fit = self.fit.clone();
        let group_count = self.trits.len().div_ceil(self.scale_group_size);
        let mut group_outputs = Vec::new();
        group_outputs
            .try_reserve_exact(group_count)
            .map_err(|_| OutputReconstructionError::ReceiptAllocationFailed)?;
        group_outputs.resize(group_count, 0.0);
        let first_output_row = tile_start / input_width;
        let last_output_row = (tile_end - 1) / input_width;
        for token in 0..token_count {
            if !activations.token_mask()[token] {
                continue;
            }
            let input_offset = token
                .checked_mul(input_width)
                .ok_or(OutputReconstructionError::InvalidGeometry)?;
            let target_offset = token
                .checked_mul(self.output_width)
                .ok_or(OutputReconstructionError::InvalidGeometry)?;
            for output_row in first_output_row..=last_output_row {
                group_outputs.fill(0.0);
                let row_start = output_row
                    .checked_mul(input_width)
                    .ok_or(OutputReconstructionError::InvalidGeometry)?;
                let row_end = row_start
                    .checked_add(input_width)
                    .ok_or(OutputReconstructionError::InvalidGeometry)?;
                let coefficient_start = tile_start.max(row_start);
                let coefficient_end = tile_end.min(row_end);
                for global_index in coefficient_start..coefficient_end {
                    let tile_offset = global_index - tile_start;
                    let group = tile_offset / self.scale_group_size;
                    let input_column = global_index - row_start;
                    group_outputs[group] += f64::from(self.trits[tile_offset].get())
                        * f64::from(activations.values()[input_offset + input_column]);
                }
                candidate_fit.observe(
                    &group_outputs,
                    f64::from(residual_outputs[target_offset + output_row]),
                )?;
            }
        }
        self.fit = candidate_fit;
        self.input_width = Some(input_width);
        Ok(())
    }

    /// Evaluate this fixed-trit tile-plane's exact current contribution for a window.
    ///
    /// The returned rows are zero for masked tokens. Only the requested tile-plane
    /// is evaluated; output storage is bounded by this window.
    ///
    /// # Errors
    /// Rejects invalid geometry, non-finite activations, or malformed base scales.
    pub fn current_tile_plane_outputs(
        &self,
        activations: &ActivationWindow,
        scales: &[half::f16],
    ) -> Result<Vec<f32>, OutputReconstructionError> {
        let input_width = usize::try_from(activations.feature_width())
            .map_err(|_| OutputReconstructionError::InvalidGeometry)?;
        let token_count = usize::try_from(activations.token_count())
            .map_err(|_| OutputReconstructionError::InvalidGeometry)?;
        let total_coefficients = self
            .output_width
            .checked_mul(input_width)
            .ok_or(OutputReconstructionError::InvalidGeometry)?;
        let output_count = token_count
            .checked_mul(self.output_width)
            .ok_or(OutputReconstructionError::InvalidGeometry)?;
        let activation_count = token_count
            .checked_mul(input_width)
            .ok_or(OutputReconstructionError::InvalidGeometry)?;
        let group_count = self.trits.len().div_ceil(self.scale_group_size);
        if input_width == 0
            || token_count == 0
            || activations.values().len() != activation_count
            || activations.token_mask().len() != token_count
            || scales.len() != group_count
        {
            return Err(OutputReconstructionError::InvalidGeometry);
        }
        if activations.values().iter().any(|value| !value.is_finite()) {
            return Err(OutputReconstructionError::NonFiniteOutput { teacher: false });
        }
        if scales
            .iter()
            .any(|scale| !scale.is_finite() || scale.to_bits() & 0x8000 != 0)
            || self
                .trits
                .chunks(self.scale_group_size)
                .zip(scales)
                .any(|(group, scale)| {
                    *scale == half::f16::ZERO && group.iter().any(|trit| !trit.is_zero())
                })
        {
            return Err(OutputReconstructionError::InvalidScaleUpdate);
        }

        let tile_start = self
            .tile_index
            .checked_mul(SALT_V2_ALLOCATION_TILE_SIZE)
            .ok_or(OutputReconstructionError::InvalidGeometry)?;
        let tile_end = tile_start
            .checked_add(self.trits.len())
            .ok_or(OutputReconstructionError::InvalidGeometry)?;
        let expected_tile_len = total_coefficients
            .checked_sub(tile_start)
            .ok_or(OutputReconstructionError::InvalidGeometry)?
            .min(SALT_V2_ALLOCATION_TILE_SIZE);
        if tile_start >= total_coefficients
            || tile_end > total_coefficients
            || self.trits.len() != expected_tile_len
        {
            return Err(OutputReconstructionError::InvalidGeometry);
        }

        let mut outputs = Vec::new();
        outputs
            .try_reserve_exact(output_count)
            .map_err(|_| OutputReconstructionError::ReceiptAllocationFailed)?;
        outputs.resize(output_count, 0.0);
        let mut group_outputs = Vec::new();
        group_outputs
            .try_reserve_exact(group_count)
            .map_err(|_| OutputReconstructionError::ReceiptAllocationFailed)?;
        group_outputs.resize(group_count, 0.0_f64);
        let first_output_row = tile_start / input_width;
        let last_output_row = (tile_end - 1) / input_width;
        for token in 0..token_count {
            if !activations.token_mask()[token] {
                continue;
            }
            let input_offset = token
                .checked_mul(input_width)
                .ok_or(OutputReconstructionError::InvalidGeometry)?;
            let output_offset = token
                .checked_mul(self.output_width)
                .ok_or(OutputReconstructionError::InvalidGeometry)?;
            for output_row in first_output_row..=last_output_row {
                group_outputs.fill(0.0);
                let row_start = output_row
                    .checked_mul(input_width)
                    .ok_or(OutputReconstructionError::InvalidGeometry)?;
                let row_end = row_start
                    .checked_add(input_width)
                    .ok_or(OutputReconstructionError::InvalidGeometry)?;
                let coefficient_start = tile_start.max(row_start);
                let coefficient_end = tile_end.min(row_end);
                for global_index in coefficient_start..coefficient_end {
                    let tile_offset = global_index - tile_start;
                    let group = tile_offset / self.scale_group_size;
                    let input_column = global_index - row_start;
                    group_outputs[group] += f64::from(self.trits[tile_offset].get())
                        * f64::from(activations.values()[input_offset + input_column]);
                }
                let contribution = group_outputs
                    .iter()
                    .zip(scales)
                    .map(|(projected, scale)| projected * f64::from(scale.to_f32()))
                    .sum::<f64>() as f32;
                if !contribution.is_finite() {
                    return Err(OutputReconstructionError::NonFiniteOutput { teacher: false });
                }
                outputs[output_offset + output_row] = contribution;
            }
        }
        Ok(outputs)
    }

    /// Finish the accumulated non-negative scale solution.
    ///
    /// # Errors
    /// Rejects an empty observation stream or a non-finite solution.
    pub fn finish(self) -> Result<FixedTritScaleRefit, OutputReconstructionError> {
        self.fit.finish()
    }

    /// Finish as a canonical package update for the fitted tile-plane.
    ///
    /// Exact-zero fitted scales become the smallest positive f16 value, matching
    /// the output-candidate identity contract.
    ///
    /// # Errors
    /// Rejects an empty observation stream, unrepresentable scales, or invalid
    /// package update geometry.
    pub fn finish_update(
        self,
        tensor_index: usize,
        plane_index: usize,
    ) -> Result<FixedTritTileScaleUpdate, OutputReconstructionError> {
        let tile_index = self.tile_index;
        let fit = self.fit.finish()?;
        let scales = fit.to_positive_f16_scales()?;
        let update = SaltV2ScaleUpdate::new(tensor_index, tile_index, plane_index, scales)
            .map_err(|_| OutputReconstructionError::InvalidScaleUpdate)?;
        Ok(FixedTritTileScaleUpdate {
            update,
            squared_error: fit.squared_error(),
            observations: fit.observations(),
        })
    }
}

/// Refit one SALT tile-plane's shared scales against residual dense outputs.
///
/// `residual_outputs` must contain the teacher layer output minus the current
/// contribution of every coefficient outside this tile-plane. The tile-plane
/// then fits the remaining target without changing any trit. Rows marked false
/// by the activation window are skipped. Work and retained fit state are bounded
/// by one 256-coefficient allocation tile, independent of model size.
///
/// # Errors
/// Rejects noncanonical geometry, inconsistent activation/target shapes, an
/// out-of-range tile, or invalid values.
pub fn fit_fixed_trit_tile_scale_refit(
    activations: &ActivationWindow,
    residual_outputs: &[f32],
    output_width: usize,
    tile_index: usize,
    trits: &[Trit],
    scale_group_size: usize,
    coordinate_sweeps: usize,
) -> Result<FixedTritScaleRefit, OutputReconstructionError> {
    let mut accumulator = FixedTritTileScaleRefitAccumulator::new(
        output_width,
        tile_index,
        trits,
        scale_group_size,
        coordinate_sweeps,
    )?;
    accumulator.observe_window(activations, residual_outputs)?;
    accumulator.finish()
}

/// Tile-local fixed-trit fit paired with its canonical package scale update.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedTritTileScaleUpdate {
    update: SaltV2ScaleUpdate,
    squared_error: f64,
    observations: u64,
}

/// Owned, immutable candidate assembled from streamed fixed-trit scale fits.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedTritScaleUpdateCandidate {
    spec_id: [u8; 32],
    parent_package_digest: [u8; 32],
    initialization_seed: u64,
    updates: Vec<SaltV2ScaleUpdate>,
    candidate_id: [u8; 32],
}

impl FixedTritScaleUpdateCandidate {
    /// Canonical scale updates owned by this candidate.
    #[must_use]
    pub fn updates(&self) -> &[SaltV2ScaleUpdate] {
        &self.updates
    }

    /// Stable identity of the exact candidate contents and provenance.
    #[must_use]
    pub const fn candidate_id(&self) -> &[u8; 32] {
        &self.candidate_id
    }

    /// Re-open this owned candidate through its frozen output-reconstruction spec.
    ///
    /// # Errors
    /// Rejects a different spec or any candidate whose recomputed identity differs.
    pub fn as_scale_candidate<'a>(
        &'a self,
        spec: &OutputReconstructionSpec,
    ) -> Result<OutputReconstructionScaleCandidate<'a>, OutputReconstructionError> {
        if spec.spec_id() != &self.spec_id {
            return Err(OutputReconstructionError::CandidateSpecMismatch);
        }
        let candidate = spec.scale_update_candidate(
            &self.parent_package_digest,
            self.initialization_seed,
            &self.updates,
        )?;
        if candidate.candidate_id() != &self.candidate_id {
            return Err(OutputReconstructionError::MissingCandidateIdentity);
        }
        Ok(candidate)
    }
}

#[derive(Debug)]
struct ActiveTileScaleFit {
    tensor_index: usize,
    plane_index: usize,
    output_width: usize,
    current_scales: Vec<half::f16>,
    accumulator: FixedTritTileScaleRefitAccumulator,
}

fn deterministic_scale_refit_initial_scales(
    spec_id: &[u8; 32],
    parent_package_digest: &[u8; 32],
    initialization_seed: u64,
    tensor_index: usize,
    tile_index: usize,
    plane_index: usize,
    current_scales: &[half::f16],
) -> Vec<f64> {
    current_scales
        .iter()
        .enumerate()
        .map(|(group_index, scale)| {
            let mut hasher = blake3::Hasher::new_derive_key(SCALE_REFIT_START_CONTEXT);
            hasher.update(spec_id);
            hasher.update(parent_package_digest);
            for value in [
                initialization_seed,
                tensor_index as u64,
                tile_index as u64,
                plane_index as u64,
                group_index as u64,
            ] {
                hasher.update(&value.to_le_bytes());
            }
            let digest = hasher.finalize();
            let random = u64::from_le_bytes(
                digest.as_bytes()[..8]
                    .try_into()
                    .expect("BLAKE3 prefix always has eight bytes"),
            );
            let unit = (random >> 11) as f64 / ((1_u64 << 53) as f64);
            f64::from(scale.to_f32()) * (0.75 + 0.5 * unit)
        })
        .collect()
}

/// Bounded-memory builder for an immutable output-aware scale candidate.
///
/// It retains only canonical f16 scale updates and the active tile's compact
/// fit state; activation and residual windows can be released after each call.
#[derive(Debug)]
pub struct FixedTritScaleUpdateCandidateBuilder<'spec> {
    spec: &'spec OutputReconstructionSpec,
    parent_package_digest: [u8; 32],
    initialization_seed: u64,
    updates: Vec<SaltV2ScaleUpdate>,
    active: Option<ActiveTileScaleFit>,
    last_target: Option<(usize, usize, usize)>,
}

impl<'spec> FixedTritScaleUpdateCandidateBuilder<'spec> {
    /// Start a candidate bound to a frozen spec, exact parent, and seed.
    #[must_use]
    pub fn new(
        spec: &'spec OutputReconstructionSpec,
        parent_package_digest: &[u8; 32],
        initialization_seed: u64,
    ) -> Self {
        Self {
            spec,
            parent_package_digest: *parent_package_digest,
            initialization_seed,
            updates: Vec::new(),
            active: None,
            last_target: None,
        }
    }

    /// Whether this in-progress fit is bound to the exact frozen spec and parent.
    #[must_use]
    pub fn is_bound_to(
        &self,
        spec: &OutputReconstructionSpec,
        parent_package_digest: &[u8; 32],
    ) -> bool {
        self.spec.spec_id() == spec.spec_id()
            && &self.parent_package_digest == parent_package_digest
    }

    /// Begin the next canonical tensor/tile/plane fit.
    ///
    /// # Errors
    /// Rejects overlapping fits, noncanonical target order, invalid geometry,
    /// or allocation failure.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_tile_plane(
        &mut self,
        tensor_index: usize,
        tile_index: usize,
        plane_index: usize,
        output_width: usize,
        trits: &[Trit],
        current_scales: &[half::f16],
        scale_group_size: usize,
        coordinate_sweeps: usize,
    ) -> Result<(), OutputReconstructionError> {
        if self.active.is_some() {
            return Err(OutputReconstructionError::ScaleFitAlreadyActive);
        }
        let target = (tensor_index, tile_index, plane_index);
        if self.last_target.is_some_and(|previous| target <= previous) {
            return Err(OutputReconstructionError::NonCanonicalScaleUpdateOrder);
        }
        if trits.is_empty() || scale_group_size == 0 {
            return Err(OutputReconstructionError::InvalidGeometry);
        }
        let expected_scale_count = trits.len().div_ceil(scale_group_size);
        if current_scales.len() != expected_scale_count
            || current_scales
                .iter()
                .any(|scale| !scale.is_finite() || scale.to_bits() & 0x8000 != 0)
            || trits
                .chunks(scale_group_size)
                .zip(current_scales)
                .any(|(group, scale)| {
                    *scale == half::f16::ZERO && group.iter().any(|trit| !trit.is_zero())
                })
        {
            return Err(OutputReconstructionError::InvalidScaleUpdate);
        }
        self.updates
            .try_reserve(1)
            .map_err(|_| OutputReconstructionError::ReceiptAllocationFailed)?;
        let mut current_scales_owned = Vec::new();
        current_scales_owned
            .try_reserve_exact(current_scales.len())
            .map_err(|_| OutputReconstructionError::ReceiptAllocationFailed)?;
        current_scales_owned.extend_from_slice(current_scales);
        let initial_scales = deterministic_scale_refit_initial_scales(
            self.spec.spec_id(),
            &self.parent_package_digest,
            self.initialization_seed,
            tensor_index,
            tile_index,
            plane_index,
            current_scales,
        );
        let accumulator = FixedTritTileScaleRefitAccumulator::new_with_initial_scales(
            output_width,
            tile_index,
            trits,
            scale_group_size,
            coordinate_sweeps,
            Some(&initial_scales),
        )?;
        self.active = Some(ActiveTileScaleFit {
            tensor_index,
            plane_index,
            output_width,
            current_scales: current_scales_owned,
            accumulator,
        });
        Ok(())
    }

    /// Begin a fit directly from one strictly-read SALT V2 parent plane.
    ///
    /// The packed plane is decoded canonically and copied into the bounded
    /// active-tile state, allowing the reader's borrowed buffer to be released
    /// as soon as this method returns. `tensor_index` is campaign-defined; the
    /// caller must bind it to the package tensor name in campaign provenance.
    ///
    /// # Errors
    /// Rejects noncanonical packed data or the same geometry/order errors as
    /// [`Self::begin_tile_plane`].
    #[allow(clippy::too_many_arguments)]
    pub fn begin_packed_tile_plane(
        &mut self,
        tensor_index: usize,
        codec: SaltV2Codec,
        packed_plane: PackedSaltV2PlaneRef<'_>,
        output_width: usize,
        scale_group_size: usize,
        coordinate_sweeps: usize,
    ) -> Result<(), OutputReconstructionError> {
        let trits = unpack_salt_v2_plane(
            codec,
            packed_plane.packed_bytes(),
            packed_plane.logical_len(),
        )
        .map_err(|_| OutputReconstructionError::InvalidPackedTritPlane)?;
        self.begin_tile_plane(
            tensor_index,
            packed_plane.tile_index(),
            packed_plane.plane_index(),
            output_width,
            &trits,
            packed_plane.scales(),
            scale_group_size,
            coordinate_sweeps,
        )
    }

    /// Add one activation/residual window to the active tile-plane fit.
    ///
    /// # Errors
    /// Rejects calls without an active fit or invalid/non-finite window data.
    pub fn observe_window(
        &mut self,
        activations: &ActivationWindow,
        residual_outputs: &[f32],
    ) -> Result<(), OutputReconstructionError> {
        self.active
            .as_mut()
            .ok_or(OutputReconstructionError::NoActiveScaleFit)?
            .accumulator
            .observe_window(activations, residual_outputs)
    }

    /// Derive the fixed-trit residual for one window and observe it.
    ///
    /// For a current full projection output `y` and the active tile-plane's
    /// current contribution `p`, the refit target is
    /// `teacher - (y - p)`: the teacher output minus every contribution except
    /// the tile-plane whose non-negative scales are being refit.
    ///
    /// # Errors
    /// Rejects calls without an active fit, mismatched output geometry, or
    /// non-finite/unrepresentable values.
    pub fn observe_window_with_current_output(
        &mut self,
        activations: &ActivationWindow,
        teacher_outputs: &[f32],
        current_projection_outputs: &[f32],
        active_tile_plane_outputs: &[f32],
    ) -> Result<(), OutputReconstructionError> {
        let active = self
            .active
            .as_mut()
            .ok_or(OutputReconstructionError::NoActiveScaleFit)?;
        let token_count = usize::try_from(activations.token_count())
            .map_err(|_| OutputReconstructionError::InvalidGeometry)?;
        let output_count = token_count
            .checked_mul(active.output_width)
            .ok_or(OutputReconstructionError::InvalidGeometry)?;
        if token_count == 0
            || teacher_outputs.len() != output_count
            || current_projection_outputs.len() != output_count
            || active_tile_plane_outputs.len() != output_count
        {
            return Err(OutputReconstructionError::InvalidGeometry);
        }
        if teacher_outputs.iter().any(|value| !value.is_finite()) {
            return Err(OutputReconstructionError::NonFiniteOutput { teacher: true });
        }
        if current_projection_outputs
            .iter()
            .chain(active_tile_plane_outputs)
            .any(|value| !value.is_finite())
        {
            return Err(OutputReconstructionError::NonFiniteOutput { teacher: false });
        }

        let mut residual_outputs = Vec::new();
        residual_outputs
            .try_reserve_exact(output_count)
            .map_err(|_| OutputReconstructionError::ReceiptAllocationFailed)?;
        for ((teacher, current), active_tile) in teacher_outputs
            .iter()
            .zip(current_projection_outputs)
            .zip(active_tile_plane_outputs)
        {
            let residual = f64::from(*teacher) - (f64::from(*current) - f64::from(*active_tile));
            let residual = residual as f32;
            if !residual.is_finite() {
                return Err(OutputReconstructionError::NonFiniteOutput { teacher: false });
            }
            residual_outputs.push(residual);
        }
        active
            .accumulator
            .observe_window(activations, &residual_outputs)
    }

    /// Derive the active tile-plane contribution from the stored trits and base scales.
    ///
    /// The exact contribution is subtracted from the current full projection
    /// output before the fixed-trit scale fit observes the residual. Base scales
    /// are fixed for the lifetime of this candidate builder.
    ///
    /// # Errors
    /// Rejects calls without an active fit, invalid output geometry, or non-finite values.
    pub fn observe_window_from_current_projection(
        &mut self,
        activations: &ActivationWindow,
        teacher_outputs: &[f32],
        current_projection_outputs: &[f32],
    ) -> Result<(), OutputReconstructionError> {
        let active = self
            .active
            .as_ref()
            .ok_or(OutputReconstructionError::NoActiveScaleFit)?;
        let tile_outputs = active
            .accumulator
            .current_tile_plane_outputs(activations, &active.current_scales)?;
        self.observe_window_with_current_output(
            activations,
            teacher_outputs,
            current_projection_outputs,
            &tile_outputs,
        )
    }

    /// Finish the current tile-plane and append its canonical package update.
    ///
    /// # Errors
    /// Rejects calls without an active fit or an empty/invalid observation stream.
    pub fn finish_tile_plane(&mut self) -> Result<(), OutputReconstructionError> {
        let active = self
            .active
            .take()
            .ok_or(OutputReconstructionError::NoActiveScaleFit)?;
        let update = active
            .accumulator
            .finish_update(active.tensor_index, active.plane_index)?;
        self.updates.push(update.update().clone());
        self.last_target = Some((
            update.update().tensor_index(),
            update.update().tile_index(),
            update.update().plane_index(),
        ));
        Ok(())
    }

    /// Finish the complete candidate and compute its immutable content identity.
    ///
    /// # Errors
    /// Rejects an active or empty fit set, invalid parent identity, or invalid updates.
    pub fn finish(mut self) -> Result<FixedTritScaleUpdateCandidate, OutputReconstructionError> {
        if let Some(active) = self.active.take() {
            let update = active
                .accumulator
                .finish_update(active.tensor_index, active.plane_index)?;
            self.updates.push(update.update().clone());
        }
        let candidate_id = self.spec.candidate_id_for_scale_updates(
            &self.parent_package_digest,
            self.initialization_seed,
            &self.updates,
        )?;
        Ok(FixedTritScaleUpdateCandidate {
            spec_id: *self.spec.spec_id(),
            parent_package_digest: self.parent_package_digest,
            initialization_seed: self.initialization_seed,
            updates: self.updates,
            candidate_id,
        })
    }
}

impl FixedTritTileScaleUpdate {
    /// Canonically indexed replacement scales for this package plane.
    #[must_use]
    pub const fn update(&self) -> &SaltV2ScaleUpdate {
        &self.update
    }

    /// Sum of squared residual-output error before f16 storage rounding.
    #[must_use]
    pub const fn squared_error(&self) -> f64 {
        self.squared_error
    }

    /// Number of selected token/output rows consumed by the fit.
    #[must_use]
    pub const fn observations(&self) -> u64 {
        self.observations
    }
}

/// Fit one tile-plane and return a package-ready canonical scale update.
#[allow(clippy::too_many_arguments)]
pub fn fit_fixed_trit_tile_scale_update(
    activations: &ActivationWindow,
    residual_outputs: &[f32],
    output_width: usize,
    tensor_index: usize,
    tile_index: usize,
    plane_index: usize,
    trits: &[Trit],
    scale_group_size: usize,
    coordinate_sweeps: usize,
) -> Result<FixedTritTileScaleUpdate, OutputReconstructionError> {
    let mut accumulator = FixedTritTileScaleRefitAccumulator::new(
        output_width,
        tile_index,
        trits,
        scale_group_size,
        coordinate_sweeps,
    )?;
    accumulator.observe_window(activations, residual_outputs)?;
    accumulator.finish_update(tensor_index, plane_index)
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
    /// Per-layer activation caches are empty, unordered, or inconsistent.
    InvalidActivationCacheSet,
    /// Activation-cache identity or source provenance differs from the frozen spec.
    ActivationSetIdentityMismatch,
    /// A requested activation scope is not a scheduled block/window.
    InvalidActivationWindowScope,
    /// Layer activation windows differ in token alignment, mask, or sequence boundaries.
    ActivationWindowMismatch,
    /// Decoded activation payload exceeds the caller's total window budget.
    ActivationWindowBudgetExceeded,
    /// A bounded activation cache failed to reopen its requested window.
    ActivationCache(ActivationCacheError),
    /// The parent package has no exact-byte identity.
    MissingPackageIdentity,
    /// A scale candidate contains no updates.
    EmptyScaleUpdateSet,
    /// Scale updates are not strictly ordered by tensor, tile, and plane.
    NonCanonicalScaleUpdateOrder,
    /// A candidate scale is non-finite or not positive.
    InvalidScaleUpdate,
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
    /// A candidate builder already has a tile-plane fit in progress.
    ScaleFitAlreadyActive,
    /// A tile-plane operation was requested without an active fit.
    NoActiveScaleFit,
    /// A fitted f64 scale cannot be represented by the package's f16 scale field.
    ScaleNotRepresentable,
    /// A packed parent-package plane failed canonical ternary decoding.
    InvalidPackedTritPlane,
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
            Self::InvalidActivationCacheSet => {
                formatter.write_str("output-reconstruction activation cache set is invalid")
            }
            Self::ActivationSetIdentityMismatch => {
                formatter.write_str("output-reconstruction activation identity differs")
            }
            Self::InvalidActivationWindowScope => {
                formatter.write_str("output-reconstruction activation window scope is invalid")
            }
            Self::ActivationWindowMismatch => {
                formatter.write_str("output-reconstruction layer windows are not aligned")
            }
            Self::ActivationWindowBudgetExceeded => {
                formatter.write_str("output-reconstruction activation window budget exceeded")
            }
            Self::ActivationCache(error) => write!(formatter, "activation cache: {error}"),
            Self::MissingPackageIdentity => {
                formatter.write_str("output-reconstruction parent package identity is missing")
            }
            Self::EmptyScaleUpdateSet => {
                formatter.write_str("output-reconstruction scale candidate is empty")
            }
            Self::NonCanonicalScaleUpdateOrder => {
                formatter.write_str("output-reconstruction scale updates are not canonical")
            }
            Self::InvalidScaleUpdate => {
                formatter.write_str("output-reconstruction scale update is invalid")
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
            Self::ScaleFitAlreadyActive => {
                formatter.write_str("fixed-trit scale candidate already has an active fit")
            }
            Self::NoActiveScaleFit => {
                formatter.write_str("fixed-trit scale candidate has no active fit")
            }
            Self::ScaleNotRepresentable => {
                formatter.write_str("fixed-trit scale cannot be represented as f16")
            }
            Self::InvalidPackedTritPlane => {
                formatter.write_str("packed parent plane is not canonical ternary data")
            }
        }
    }
}

impl std::error::Error for OutputReconstructionError {}
