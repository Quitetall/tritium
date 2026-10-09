//! Exact Qwen3.5-family hybrid language runner.
//!
//! Qwen3.6 interleaves recurrent Gated DeltaNet and gated full-attention
//! mixers.  That graph, its cache state, and its zero-centered normalization
//! semantics are deliberately kept out of the homogeneous [`ModelRunner`]
//! (`super::ModelRunner`).

use core::convert::Infallible;
use std::sync::Arc;

use tritium_format::salt_v2_package::SaltV2ScaleUpdate;
use tritium_spec::TernaryBackend;

use super::qwen35_reference::{Qwen35ReferenceState, ReferenceStateView, snapshot_states};

use crate::error::NnError;
use crate::layers::{
    Projection, ProjectionActivationMode, Qwen35DeltaNet, Qwen35DeltaNetCache,
    Qwen35DeltaNetWeights, Qwen35FullAttention, Qwen35FullAttentionCache,
    Qwen35FullAttentionWeights, RecurrentStateObserver, SwiGluMlp, TokenEmbedding,
};
use crate::ops::rmsnorm_zero_centered;
use crate::qwen35_config::{Qwen35LayerType, Qwen35NormWeightSemantics, Qwen35TextConfig};

/// Unbound token-mixer weights for one Qwen3.5-family decoder layer.
#[allow(missing_debug_implementations)]
pub enum Qwen35TextMixerWeights {
    /// Gated DeltaNet weights for a `linear_attention` layer.
    DeltaNet(Qwen35DeltaNetWeights),
    /// Gated causal GQA weights for a `full_attention` layer.
    FullAttention(Qwen35FullAttentionWeights),
}

impl Qwen35TextMixerWeights {
    const fn kind(&self) -> Qwen35LayerType {
        match self {
            Self::DeltaNet(_) => Qwen35LayerType::DeltaNet,
            Self::FullAttention(_) => Qwen35LayerType::FullAttention,
        }
    }
}

/// Raw weights for one exact Qwen3.5-family language layer.
#[allow(missing_debug_implementations)]
pub struct Qwen35TextLayerWeights {
    /// Zero-centered RMSNorm before the token mixer, `[hidden]`.
    pub input_norm: Vec<f32>,
    /// Mixer selected by the checkpoint's exact layer schedule.
    pub mixer: Qwen35TextMixerWeights,
    /// Zero-centered RMSNorm before SwiGLU, `[hidden]`.
    pub post_attention_norm: Vec<f32>,
    /// Bias-free Qwen SwiGLU feed-forward network.
    pub mlp: SwiGluMlp,
}

impl Qwen35TextLayerWeights {
    /// Collect one unbound language layer.
    #[must_use]
    pub fn new(
        input_norm: Vec<f32>,
        mixer: Qwen35TextMixerWeights,
        post_attention_norm: Vec<f32>,
        mlp: SwiGluMlp,
    ) -> Self {
        Self {
            input_norm,
            mixer,
            post_attention_norm,
            mlp,
        }
    }
}

/// Raw, untied Qwen3.5-family language-model weights.
#[allow(missing_debug_implementations)]
pub struct Qwen35TextWeights {
    /// Token embedding table `[vocab, hidden]`.
    pub embedding: TokenEmbedding,
    /// Decoder layers in checkpoint order.
    pub layers: Vec<Qwen35TextLayerWeights>,
    /// Final zero-centered RMSNorm parameter `[hidden]`.
    pub final_norm: Vec<f32>,
    /// Mandatory untied language head `[vocab, hidden]`.
    pub lm_head: Projection,
}

impl Qwen35TextWeights {
    /// Collect an unbound exact text graph.
    #[must_use]
    pub fn new(
        embedding: TokenEmbedding,
        layers: Vec<Qwen35TextLayerWeights>,
        final_norm: Vec<f32>,
        lm_head: Projection,
    ) -> Self {
        Self {
            embedding,
            layers,
            final_norm,
            lm_head,
        }
    }
}

enum Qwen35TextMixer {
    DeltaNet(Qwen35DeltaNet),
    FullAttention(Qwen35FullAttention),
}

impl Qwen35TextMixer {
    #[allow(dead_code)] // Consumed by the B3 bounded projection-window adapter.
    fn projection(&self, name: &str) -> Result<&Projection, NnError> {
        match self {
            Self::DeltaNet(layer) => layer.projection(name),
            Self::FullAttention(layer) => layer.projection(name),
        }
    }

    const fn kind(&self) -> Qwen35LayerType {
        match self {
            Self::DeltaNet(_) => Qwen35LayerType::DeltaNet,
            Self::FullAttention(_) => Qwen35LayerType::FullAttention,
        }
    }

    const fn activation_mode(&self) -> ProjectionActivationMode {
        match self {
            Self::DeltaNet(layer) => layer.activation_mode(),
            Self::FullAttention(layer) => layer.activation_mode(),
        }
    }

    fn count_salt_v2_tensor_index(&self, tensor_index: usize) -> usize {
        match self {
            Self::DeltaNet(layer) => layer.count_salt_v2_tensor_index(tensor_index),
            Self::FullAttention(layer) => layer.count_salt_v2_tensor_index(tensor_index),
        }
    }

    fn apply_salt_v2_scale_updates(
        &mut self,
        tensor_index: usize,
        updates: &[SaltV2ScaleUpdate],
    ) -> Result<bool, NnError> {
        match self {
            Self::DeltaNet(layer) => layer.apply_salt_v2_scale_updates(tensor_index, updates),
            Self::FullAttention(layer) => layer.apply_salt_v2_scale_updates(tensor_index, updates),
        }
    }

    #[allow(dead_code)] // Used by the crate-internal paired Qwen measurement path.
    fn replace_projection(
        &mut self,
        name: &str,
        replacement: Projection,
    ) -> Result<Projection, NnError> {
        match self {
            Self::DeltaNet(layer) => layer.replace_projection(name, replacement),
            Self::FullAttention(layer) => layer.replace_projection(name, replacement),
        }
    }
}

struct Qwen35TextLayer {
    input_norm: Vec<f32>,
    mixer: Qwen35TextMixer,
    post_attention_norm: Vec<f32>,
    mlp: SwiGluMlp,
}

#[derive(Debug)]
enum Qwen35TextLayerCache {
    DeltaNet(Qwen35DeltaNetCache),
    FullAttention(Qwen35FullAttentionCache),
}

impl Qwen35TextLayerCache {
    const fn kind(&self) -> Qwen35LayerType {
        match self {
            Self::DeltaNet(_) => Qwen35LayerType::DeltaNet,
            Self::FullAttention(_) => Qwen35LayerType::FullAttention,
        }
    }

    const fn committed_len(&self) -> usize {
        match self {
            Self::DeltaNet(cache) => cache.len(),
            Self::FullAttention(cache) => cache.committed_len(),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::DeltaNet(cache) => cache.reset(),
            Self::FullAttention(cache) => cache.reset(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct RunnerIdentity;

/// One exact-runner hybrid cache with a single committed language cursor.
///
/// Individual mixer states are private so callers cannot advance one layer
/// independently of the rest of the language graph.
#[derive(Debug)]
pub struct Qwen35TextCache {
    runner_identity: Arc<RunnerIdentity>,
    layers: Vec<Qwen35TextLayerCache>,
    committed_len: usize,
    max_context: usize,
}

impl Qwen35TextCache {
    /// Number of tokens committed by every language layer.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.committed_len
    }

    /// Whether the cache contains no committed tokens.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.committed_len == 0
    }

    /// Stream capacity selected when this cache was created.
    #[must_use]
    pub const fn max_context(&self) -> usize {
        self.max_context
    }

    /// Clear every mixer state and the global cursor without freeing capacity.
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
        self.committed_len = 0;
    }
}

/// Successful exact-language forward output.
///
/// `final_hidden_states` are after the model's final zero-centered RMSNorm and
/// are therefore the target hidden rows consumed by Qwen3.5 MTP. They are not
/// the pre-final residual stream.
#[derive(Debug, Clone)]
pub struct Qwen35TextOutput {
    runner_identity: Arc<RunnerIdentity>,
    position_start: usize,
    hidden_size: usize,
    input_token_ids: Vec<u32>,
    final_hidden_states: Vec<f32>,
    last_logits: Vec<f32>,
}

// Equality remains a comparison of published numeric output values. Private
// runner provenance is an execution authority checked separately by MTP.
impl PartialEq for Qwen35TextOutput {
    fn eq(&self, other: &Self) -> bool {
        self.position_start == other.position_start
            && self.hidden_size == other.hidden_size
            && self.input_token_ids == other.input_token_ids
            && self.final_hidden_states == other.final_hidden_states
            && self.last_logits == other.last_logits
    }
}

impl Qwen35TextOutput {
    /// First absolute target position represented by this output.
    #[must_use]
    pub const fn position_start(&self) -> usize {
        self.position_start
    }

    /// Number of input token rows represented by the hidden-state buffer.
    #[must_use]
    pub const fn sequence(&self) -> usize {
        self.input_token_ids.len()
    }

    /// Width of each final-normalized hidden row.
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Final-normalized hidden rows `[sequence, hidden]`, suitable for MTP.
    #[must_use]
    pub fn final_hidden_states(&self) -> &[f32] {
        &self.final_hidden_states
    }

    /// Untied language-head logits for the last input token, `[vocab]`.
    #[must_use]
    pub fn last_logits(&self) -> &[f32] {
        &self.last_logits
    }

    pub(crate) fn runner_identity(&self) -> &Arc<RunnerIdentity> {
        &self.runner_identity
    }

    pub(crate) fn input_token_ids(&self) -> &[u32] {
        &self.input_token_ids
    }
}

/// Exact Qwen3.5/Qwen3.6 hybrid language-core runner.
#[allow(missing_debug_implementations)]
pub struct Qwen35TextRunner {
    identity: Arc<RunnerIdentity>,
    config: Qwen35TextConfig,
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    max_context: usize,
    rms_norm_eps: f32,
    activation_mode: ProjectionActivationMode,
    has_delta_net: bool,
    backend: Box<dyn TernaryBackend>,
    embedding: TokenEmbedding,
    layers: Vec<Qwen35TextLayer>,
    final_norm: Vec<f32>,
    lm_head: Projection,
}

pub(crate) enum Qwen35TextForwardError<E> {
    Runtime(NnError),
    Observer(E),
}

/// Paired raw vectors at one sampled token position for an internal PTQ probe.
#[allow(dead_code)] // Consumed by the receipt-producing Stage-7 probe driver.
pub(crate) struct Qwen35ProjectionProbeDepth<'a> {
    /// Zero-based calibration-sequence ordinal.
    pub(crate) sequence_index: u64,
    /// One-based token position within that calibration sequence.
    pub(crate) token_position: usize,
    /// Dense reference's final-normalized hidden row.
    pub(crate) reference_hidden: &'a [f32],
    /// Single-matrix candidate's final-normalized hidden row.
    pub(crate) candidate_hidden: &'a [f32],
    /// Dense reference recurrent state at the selected DeltaNet layer.
    pub(crate) reference_state: &'a [f32],
    /// Candidate recurrent state at the same DeltaNet layer.
    pub(crate) candidate_state: &'a [f32],
}

/// Failure while collecting one-matrix paired Qwen probe vectors.
#[allow(dead_code)] // Consumed by the receipt-producing Stage-7 probe driver.
#[derive(Debug)]
pub(crate) enum Qwen35ProjectionProbeError<E> {
    Runtime(NnError),
    Observer(E),
}

struct Qwen35ProbeDepthOwned {
    token_position: usize,
    final_hidden: Vec<f32>,
    recurrent_state: Vec<f32>,
}

fn probe_forward_sample(
    runner: &Qwen35TextRunner,
    tokens: &[u32],
    positions: &[usize],
    state_layer: usize,
) -> Result<Vec<Qwen35ProbeDepthOwned>, NnError> {
    let zero_based = positions
        .iter()
        .map(|position| position - 1)
        .collect::<Vec<_>>();
    let mut recurrent = (0..positions.len())
        .map(|_| None)
        .collect::<Vec<Option<Vec<f32>>>>();
    let mut cache = runner.new_cache(tokens.len())?;
    let output = match runner.forward_with_block_and_state_observer(
        tokens,
        &mut cache,
        &zero_based,
        |_, _, _, _| Ok::<_, Infallible>(()),
        |block, token_position, state| {
            if usize::try_from(block).ok() == Some(state_layer)
                && let Ok(sample_index) = zero_based.binary_search(&token_position)
            {
                recurrent[sample_index] = Some(state.to_vec());
            }
        },
    ) {
        Ok(output) => output,
        Err(Qwen35TextForwardError::Runtime(error)) => return Err(error),
        Err(Qwen35TextForwardError::Observer(never)) => match never {},
    };

    let hidden_size = output.hidden_size();
    let hidden = output.final_hidden_states();
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(positions.len())
        .map_err(|error| NnError::Backend(format!("allocate Qwen probe samples: {error}")))?;
    for (sample_index, &token_position) in positions.iter().enumerate() {
        let row_start = (token_position - 1)
            .checked_mul(hidden_size)
            .ok_or(NnError::Shape {
                expected: usize::MAX,
                got: hidden.len(),
            })?;
        let row_end = row_start.checked_add(hidden_size).ok_or(NnError::Shape {
            expected: usize::MAX,
            got: hidden.len(),
        })?;
        let hidden_row = hidden.get(row_start..row_end).ok_or(NnError::Shape {
            expected: row_end,
            got: hidden.len(),
        })?;
        let recurrent_state = recurrent[sample_index].take().ok_or_else(|| {
            NnError::MissingTensor(format!(
                "Qwen probe did not observe recurrent state at layer {state_layer}, token {token_position}"
            ))
        })?;
        if hidden_row
            .iter()
            .chain(&recurrent_state)
            .any(|value| !value.is_finite())
        {
            return Err(NnError::Backend(
                "Qwen probe sample contains a non-finite value".to_owned(),
            ));
        }
        samples.push(Qwen35ProbeDepthOwned {
            token_position,
            final_hidden: hidden_row.to_vec(),
            recurrent_state,
        });
    }
    Ok(samples)
}

impl<E> From<NnError> for Qwen35TextForwardError<E> {
    fn from(error: NnError) -> Self {
        Self::Runtime(error)
    }
}

#[allow(dead_code)] // Used by the crate-internal paired Qwen measurement path.
struct ProjectionRestoreGuard<'a> {
    runner: &'a mut Qwen35TextRunner,
    tensor_name: String,
    original: Option<Projection>,
}

impl Drop for ProjectionRestoreGuard<'_> {
    fn drop(&mut self) {
        if let Some(original) = self.original.take() {
            // The slot was resolved before guard creation and cannot be
            // structurally removed while the callback holds the runner.
            let _ = self
                .runner
                .replace_named_projection(&self.tensor_name, original);
        }
    }
}

#[allow(dead_code)] // Used by the crate-internal paired Qwen measurement path.
fn replace_projection_slot(
    slot: &mut Projection,
    replacement: Projection,
    tensor_name: &str,
) -> Result<Projection, NnError> {
    if replacement.n_out() != slot.n_out() || replacement.k_in() != slot.k_in() {
        return Err(NnError::Shape {
            expected: slot.n_out().saturating_mul(slot.k_in()),
            got: replacement.n_out().saturating_mul(replacement.k_in()),
        });
    }
    if replacement.activation_mode() != slot.activation_mode() {
        return Err(NnError::Backend(format!(
            "replacement projection `{tensor_name}` changes activation arithmetic"
        )));
    }
    Ok(std::mem::replace(slot, replacement))
}

impl Qwen35TextRunner {
    /// Recompute one named projection over a bounded row-major activation window.
    ///
    /// This measurement seam lets PTQ compare the exact deployed projection
    /// against teacher outputs computed from the original dense checkpoint,
    /// without retaining a full-model activation history. `observer` runs
    /// synchronously with one owned output window; it must not treat this local
    /// projection result as model-quality or release evidence by itself.
    #[allow(dead_code)] // Consumed by the B3 candidate-builder adapter.
    pub(crate) fn visit_named_projection_outputs(
        &self,
        tensor_name: &str,
        activations: &[f32],
        rows: usize,
        mut observer: impl FnMut(&[f32]),
    ) -> Result<(), NnError> {
        if rows == 0 {
            return Err(NnError::Shape {
                expected: 1,
                got: 0,
            });
        }
        let projection = self.named_projection(tensor_name)?;
        let input_count = rows.checked_mul(projection.k_in()).ok_or(NnError::Shape {
            expected: usize::MAX,
            got: activations.len(),
        })?;
        let output_count = rows.checked_mul(projection.n_out()).ok_or(NnError::Shape {
            expected: usize::MAX,
            got: rows,
        })?;
        if activations.len() != input_count {
            return Err(NnError::Shape {
                expected: input_count,
                got: activations.len(),
            });
        }
        if activations.iter().any(|value| !value.is_finite()) {
            return Err(NnError::Backend(
                "Qwen projection input contains a non-finite value".to_owned(),
            ));
        }
        let mut outputs = Vec::new();
        outputs.try_reserve_exact(output_count).map_err(|error| {
            NnError::Backend(format!("allocate Qwen projection output window: {error}"))
        })?;
        outputs.resize(output_count, 0.0);
        projection.forward(self.backend.as_ref(), activations, rows, &mut outputs)?;
        if outputs.iter().any(|value| !value.is_finite()) {
            return Err(NnError::Backend(
                "Qwen projection output contains a non-finite value".to_owned(),
            ));
        }
        observer(&outputs);
        Ok(())
    }

    /// Recompute aligned teacher and current-package outputs for one bounded
    /// projection activation window. The teacher must preserve the package's
    /// activation arithmetic (for example, an exact-fp32 dense projection for
    /// SALT V2); callers bind the activation source and teacher identity in
    /// their campaign receipt. Outputs are borrowed only for the synchronous
    /// callback and are not retained by the runner.
    ///
    /// # Errors
    /// Rejects unknown projection names, incompatible teacher geometry or
    /// activation arithmetic, mismatched input shape, non-finite inputs or
    /// outputs, and allocation/backend failures.
    pub fn visit_named_projection_output_pairs(
        &self,
        tensor_name: &str,
        teacher: &Projection,
        activations: &[f32],
        rows: usize,
        mut observer: impl FnMut(&[f32], &[f32]),
    ) -> Result<(), NnError> {
        if rows == 0 {
            return Err(NnError::Shape {
                expected: 1,
                got: 0,
            });
        }
        let current = self.named_projection(tensor_name)?;
        if teacher.k_in() != current.k_in() || teacher.n_out() != current.n_out() {
            return Err(NnError::Shape {
                expected: current.n_out().saturating_mul(current.k_in()),
                got: teacher.n_out().saturating_mul(teacher.k_in()),
            });
        }
        if teacher.activation_mode() != current.activation_mode() {
            return Err(NnError::Backend(
                "Qwen teacher and package projection use different activation arithmetic"
                    .to_owned(),
            ));
        }
        let input_count = rows.checked_mul(current.k_in()).ok_or(NnError::Shape {
            expected: usize::MAX,
            got: activations.len(),
        })?;
        if activations.len() != input_count {
            return Err(NnError::Shape {
                expected: input_count,
                got: activations.len(),
            });
        }
        if activations.iter().any(|value| !value.is_finite()) {
            return Err(NnError::Backend(
                "Qwen projection input contains a non-finite value".to_owned(),
            ));
        }
        let output_count = rows.checked_mul(current.n_out()).ok_or(NnError::Shape {
            expected: usize::MAX,
            got: rows,
        })?;
        let mut teacher_outputs = Vec::new();
        teacher_outputs
            .try_reserve_exact(output_count)
            .map_err(|error| {
                NnError::Backend(format!("allocate Qwen teacher output window: {error}"))
            })?;
        teacher_outputs.resize(output_count, 0.0);
        let mut current_outputs = Vec::new();
        current_outputs
            .try_reserve_exact(output_count)
            .map_err(|error| {
                NnError::Backend(format!("allocate Qwen package output window: {error}"))
            })?;
        current_outputs.resize(output_count, 0.0);
        teacher.forward(
            self.backend.as_ref(),
            activations,
            rows,
            &mut teacher_outputs,
        )?;
        current.forward(
            self.backend.as_ref(),
            activations,
            rows,
            &mut current_outputs,
        )?;
        if teacher_outputs
            .iter()
            .chain(&current_outputs)
            .any(|value| !value.is_finite())
        {
            return Err(NnError::Backend(
                "Qwen paired projection output contains a non-finite value".to_owned(),
            ));
        }
        observer(&teacher_outputs, &current_outputs);
        Ok(())
    }

    #[allow(dead_code)] // Consumed by the B3 candidate-builder adapter.
    fn named_projection(&self, tensor_name: &str) -> Result<&Projection, NnError> {
        let layer_path = tensor_name
            .strip_prefix("model.language_model.layers.")
            .ok_or_else(|| NnError::MissingTensor(tensor_name.to_owned()))?;
        let (index_text, projection_name) = layer_path
            .split_once('.')
            .ok_or_else(|| NnError::MissingTensor(tensor_name.to_owned()))?;
        let index = index_text
            .parse::<usize>()
            .map_err(|_| NnError::MissingTensor(tensor_name.to_owned()))?;
        if index.to_string() != index_text {
            return Err(NnError::MissingTensor(tensor_name.to_owned()));
        }
        let layer = self
            .layers
            .get(index)
            .ok_or_else(|| NnError::MissingTensor(tensor_name.to_owned()))?;
        match projection_name {
            "mlp.gate_proj.weight" => Ok(&layer.mlp.gate),
            "mlp.up_proj.weight" => Ok(&layer.mlp.up),
            "mlp.down_proj.weight" => Ok(&layer.mlp.down),
            _ if projection_name.starts_with("linear_attn.")
                || projection_name.starts_with("self_attn.") =>
            {
                layer.mixer.projection(projection_name)
            }
            _ => Err(NnError::MissingTensor(tensor_name.to_owned())),
        }
    }

    /// Temporarily replace one canonical language projection while executing a
    /// paired measurement. The dense projection is restored on normal return,
    /// error, and panic unwind. The model is single-threaded during the callback.
    #[allow(dead_code)] // The GDN probe producer is the next consumer of this seam.
    pub(crate) fn with_projection_override<T>(
        &mut self,
        tensor_name: &str,
        replacement: Projection,
        execute: impl FnOnce(&Self) -> Result<T, NnError>,
    ) -> Result<T, NnError> {
        let original = self.replace_named_projection(tensor_name, replacement)?;
        let guard = ProjectionRestoreGuard {
            runner: self,
            tensor_name: tensor_name.to_owned(),
            original: Some(original),
        };
        let result = execute(&*guard.runner)?;
        drop(guard);
        Ok(result)
    }

    /// Stream paired reference/candidate vectors for one projection without
    /// retaining sequence histories. Each reference and candidate forward gets
    /// a fresh cache; the candidate projection is restored before observations
    /// are delivered to the caller. This collects raw samples only and does not
    /// define a divergence metric or produce campaign evidence.
    #[allow(dead_code)] // The receipt-producing Stage-7 probe driver will call this.
    pub(crate) fn visit_projection_probe_pairs<'tokens, I, E>(
        &mut self,
        tensor_name: &str,
        replacement: Projection,
        sequences: I,
        one_based_positions: &[usize],
        state_layer: usize,
        mut observer: impl FnMut(Qwen35ProjectionProbeDepth<'_>) -> Result<(), E>,
    ) -> Result<u64, Qwen35ProjectionProbeError<E>>
    where
        I: IntoIterator<Item = &'tokens [u32]>,
    {
        if one_based_positions.is_empty()
            || one_based_positions[0] == 0
            || one_based_positions
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(Qwen35ProjectionProbeError::Runtime(NnError::MissingConfig(
                "Qwen probe positions must be nonempty, one-based, and strictly increasing".into(),
            )));
        }
        if self.config.layer_types.get(state_layer) != Some(&Qwen35LayerType::DeltaNet) {
            return Err(Qwen35ProjectionProbeError::Runtime(NnError::MissingConfig(
                "Qwen probe recurrent-state layer must be DeltaNet".into(),
            )));
        }
        if replacement.clone_salt_v2_resident().is_none() {
            return Err(Qwen35ProjectionProbeError::Runtime(NnError::MissingConfig(
                "Qwen probe candidate must use a resident SALT V2 projection".into(),
            )));
        }

        let mut sequence_index = 0_u64;
        for tokens in sequences {
            if tokens.is_empty()
                || one_based_positions
                    .iter()
                    .any(|position| *position > tokens.len())
            {
                return Err(Qwen35ProjectionProbeError::Runtime(NnError::Shape {
                    expected: one_based_positions.last().copied().unwrap_or(1),
                    got: tokens.len(),
                }));
            }
            let reference = probe_forward_sample(self, tokens, one_based_positions, state_layer)
                .map_err(Qwen35ProjectionProbeError::Runtime)?;
            let candidate_projection = replacement.clone_salt_v2_resident().ok_or_else(|| {
                Qwen35ProjectionProbeError::Runtime(NnError::MissingConfig(
                    "Qwen probe SALT V2 resident could not be shared".into(),
                ))
            })?;
            let candidate = self
                .with_projection_override(tensor_name, candidate_projection, |runner| {
                    probe_forward_sample(runner, tokens, one_based_positions, state_layer)
                })
                .map_err(Qwen35ProjectionProbeError::Runtime)?;
            if reference.len() != candidate.len() {
                return Err(Qwen35ProjectionProbeError::Runtime(NnError::Shape {
                    expected: reference.len(),
                    got: candidate.len(),
                }));
            }
            for (reference, candidate) in reference.iter().zip(&candidate) {
                if reference.token_position != candidate.token_position
                    || reference.final_hidden.len() != candidate.final_hidden.len()
                    || reference.recurrent_state.len() != candidate.recurrent_state.len()
                {
                    return Err(Qwen35ProjectionProbeError::Runtime(NnError::Provenance(
                        "paired Qwen probe sample coordinates or shapes differ".to_owned(),
                    )));
                }
                observer(Qwen35ProjectionProbeDepth {
                    sequence_index,
                    token_position: reference.token_position,
                    reference_hidden: &reference.final_hidden,
                    candidate_hidden: &candidate.final_hidden,
                    reference_state: &reference.recurrent_state,
                    candidate_state: &candidate.recurrent_state,
                })
                .map_err(Qwen35ProjectionProbeError::Observer)?;
            }
            sequence_index = sequence_index.checked_add(1).ok_or_else(|| {
                Qwen35ProjectionProbeError::Runtime(NnError::ResourceExhausted(
                    "Qwen probe sequence count exceeds u64".into(),
                ))
            })?;
        }
        Ok(sequence_index)
    }

    #[allow(dead_code)] // Used by the crate-internal paired Qwen measurement path.
    fn replace_named_projection(
        &mut self,
        tensor_name: &str,
        replacement: Projection,
    ) -> Result<Projection, NnError> {
        let layer_path = tensor_name
            .strip_prefix("model.language_model.layers.")
            .ok_or_else(|| NnError::MissingTensor(tensor_name.to_owned()))?;
        let (index_text, projection_name) = layer_path
            .split_once('.')
            .ok_or_else(|| NnError::MissingTensor(tensor_name.to_owned()))?;
        let index = index_text
            .parse::<usize>()
            .map_err(|_| NnError::MissingTensor(tensor_name.to_owned()))?;
        if index.to_string() != index_text {
            return Err(NnError::MissingTensor(tensor_name.to_owned()));
        }
        let layer = self
            .layers
            .get_mut(index)
            .ok_or_else(|| NnError::MissingTensor(tensor_name.to_owned()))?;

        match projection_name {
            "mlp.gate_proj.weight" => {
                replace_projection_slot(&mut layer.mlp.gate, replacement, tensor_name)
            }
            "mlp.up_proj.weight" => {
                replace_projection_slot(&mut layer.mlp.up, replacement, tensor_name)
            }
            "mlp.down_proj.weight" => {
                replace_projection_slot(&mut layer.mlp.down, replacement, tensor_name)
            }
            _ if projection_name.starts_with("linear_attn.")
                || projection_name.starts_with("self_attn.") =>
            {
                layer.mixer.replace_projection(projection_name, replacement)
            }
            _ => Err(NnError::MissingTensor(tensor_name.to_owned())),
        }
    }

    /// Apply one tensor's scale-only candidate to its uniquely identified SALT V2
    /// projection. The full model graph is scanned before mutation; host and CUDA
    /// residents validate the complete update before publishing changed scales.
    ///
    /// # Errors
    /// Rejects empty or mixed-tensor updates, missing or ambiguous tensor
    /// identity, shared resident storage, and malformed scale candidates.
    pub fn apply_salt_v2_scale_updates(
        &mut self,
        updates: &[SaltV2ScaleUpdate],
    ) -> Result<(), NnError> {
        let tensor_index = updates
            .first()
            .map(SaltV2ScaleUpdate::tensor_index)
            .ok_or_else(|| NnError::Backend("SALT V2 scale update set is empty".into()))?;
        if updates
            .iter()
            .any(|update| update.tensor_index() != tensor_index)
        {
            return Err(NnError::Backend(
                "one model scale-update call must target exactly one package tensor".into(),
            ));
        }

        let mut matches = usize::from(self.embedding.salt_v2_tensor_index() == Some(tensor_index))
            + usize::from(self.lm_head.salt_v2_tensor_index() == Some(tensor_index));
        for layer in &self.layers {
            matches += layer.mixer.count_salt_v2_tensor_index(tensor_index);
            matches += usize::from(layer.mlp.gate.salt_v2_tensor_index() == Some(tensor_index));
            matches += usize::from(layer.mlp.up.salt_v2_tensor_index() == Some(tensor_index));
            matches += usize::from(layer.mlp.down.salt_v2_tensor_index() == Some(tensor_index));
        }
        if matches != 1 {
            return Err(NnError::Backend(format!(
                "Qwen model has {matches} SALT V2 projections for package tensor {tensor_index}; expected exactly one"
            )));
        }

        if self.embedding.salt_v2_tensor_index() == Some(tensor_index) {
            self.embedding
                .apply_salt_v2_scale_updates(tensor_index, updates)?;
            return Ok(());
        }
        if self.lm_head.salt_v2_tensor_index() == Some(tensor_index) {
            return self
                .lm_head
                .apply_salt_v2_scale_updates(tensor_index, updates);
        }
        for layer in &mut self.layers {
            if layer.mixer.count_salt_v2_tensor_index(tensor_index) > 0 {
                layer
                    .mixer
                    .apply_salt_v2_scale_updates(tensor_index, updates)?;
                return Ok(());
            }
            for projection in [&mut layer.mlp.gate, &mut layer.mlp.up, &mut layer.mlp.down] {
                if projection.salt_v2_tensor_index() == Some(tensor_index) {
                    return projection.apply_salt_v2_scale_updates(tensor_index, updates);
                }
            }
        }
        Err(NnError::Backend(
            "Qwen SALT V2 tensor identity disappeared during update".into(),
        ))
    }

    /// Bind the exact mixed schedule and all raw weights to one private runner.
    ///
    /// # Errors
    ///
    /// Returns [`NnError::MissingConfig`] for unsupported or contradictory
    /// schedule/numeric semantics, [`NnError::Shape`] for any weight geometry
    /// mismatch, or [`NnError::Backend`] for non-finite fp32 weights or an
    /// allocation failure.
    pub fn new(
        config: &Qwen35TextConfig,
        weights: Qwen35TextWeights,
        backend: Box<dyn TernaryBackend>,
    ) -> Result<Self, NnError> {
        let hidden_size = axis(config.hidden_size, "hidden_size")?;
        let intermediate_size = axis(config.intermediate_size, "intermediate_size")?;
        let vocab_size = axis(config.vocab_size, "vocab_size")?;
        let max_context = axis(config.max_position_embeddings, "max_position_embeddings")?;
        let layer_count = axis(config.num_hidden_layers, "num_hidden_layers")?;
        let interval = axis(config.full_attention_interval, "full_attention_interval")?;
        let rms_norm_eps = config.rms_norm_eps as f32;
        // Guard both the source value and overflow introduced by f64-to-f32 narrowing.
        if !config.rms_norm_eps.is_finite() || !rms_norm_eps.is_finite() || rms_norm_eps <= 0.0 {
            return Err(invalid_config(
                "Qwen3.5 text RMSNorm epsilon must be a finite positive f32 value",
            ));
        }
        if config.model_type != "qwen3_5_text"
            || !config.use_cache
            || config.tied_embeddings
            || config.full_attention.norm_weight_semantics
                != Qwen35NormWeightSemantics::ZeroCenteredOnePlusWeight
        {
            return Err(invalid_config(
                "unsupported Qwen3.5 text graph or normalization semantics",
            ));
        }
        if config.layer_types.len() != layer_count {
            return Err(NnError::Shape {
                expected: layer_count,
                got: config.layer_types.len(),
            });
        }
        if weights.layers.len() != layer_count {
            return Err(NnError::Shape {
                expected: layer_count,
                got: weights.layers.len(),
            });
        }
        validate_schedule(&config.layer_types, interval)?;
        validate_token_table(&weights.embedding, vocab_size, hidden_size)?;
        validate_projection(&weights.lm_head, vocab_size, hidden_size)?;
        validate_finite_projection(&weights.lm_head, "language head")?;
        validate_finite(&weights.final_norm, hidden_size, "final norm")?;

        let activation_mode = weights.lm_head.activation_mode();
        let mut layers = Vec::new();
        layers.try_reserve_exact(layer_count).map_err(|error| {
            NnError::Backend(format!(
                "allocate Qwen3.5 bound layer table for {layer_count} layers: {error}"
            ))
        })?;
        for (index, (expected_kind, raw)) in config
            .layer_types
            .iter()
            .copied()
            .zip(weights.layers)
            .enumerate()
        {
            if raw.mixer.kind() != expected_kind {
                return Err(invalid_config(format!(
                    "Qwen3.5 layer {index} mixer contradicts the configured schedule"
                )));
            }
            validate_finite(&raw.input_norm, hidden_size, "input norm")?;
            validate_finite(&raw.post_attention_norm, hidden_size, "post-attention norm")?;
            let mlp_mode = raw
                .mlp
                .validate_for_geometry(hidden_size, intermediate_size)?;
            if mlp_mode != activation_mode {
                return Err(invalid_config(
                    "Qwen3.5 language projections must use one activation arithmetic mode",
                ));
            }
            let mixer = match raw.mixer {
                Qwen35TextMixerWeights::DeltaNet(raw) => {
                    Qwen35TextMixer::DeltaNet(Qwen35DeltaNet::new(config, raw)?)
                }
                Qwen35TextMixerWeights::FullAttention(raw) => {
                    Qwen35TextMixer::FullAttention(Qwen35FullAttention::new(config, raw)?)
                }
            };
            if mixer.activation_mode() != activation_mode {
                return Err(invalid_config(
                    "Qwen3.5 language projections must use one activation arithmetic mode",
                ));
            }
            layers.push(Qwen35TextLayer {
                input_norm: raw.input_norm,
                mixer,
                post_attention_norm: raw.post_attention_norm,
                mlp: raw.mlp,
            });
        }

        Ok(Self {
            identity: Arc::new(RunnerIdentity),
            config: config.clone(),
            hidden_size,
            intermediate_size,
            vocab_size,
            max_context,
            rms_norm_eps,
            activation_mode,
            has_delta_net: config.layer_types.contains(&Qwen35LayerType::DeltaNet),
            backend,
            embedding: weights.embedding,
            layers,
            final_norm: weights.final_norm,
            lm_head: weights.lm_head,
        })
    }

    /// Activation arithmetic shared by every language projection.
    #[must_use]
    pub const fn activation_mode(&self) -> ProjectionActivationMode {
        self.activation_mode
    }

    /// Hidden-state width.
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// SwiGLU intermediate width.
    #[must_use]
    pub const fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    /// Vocabulary size shared by embedding and the mandatory untied head.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub(crate) fn config(&self) -> &Qwen35TextConfig {
        &self.config
    }

    pub(crate) fn identity(&self) -> &Arc<RunnerIdentity> {
        &self.identity
    }

    pub(crate) const fn max_context(&self) -> usize {
        self.max_context
    }

    pub(crate) const fn rms_norm_eps(&self) -> f32 {
        self.rms_norm_eps
    }

    pub(crate) fn execution_backend(&self) -> &dyn TernaryBackend {
        self.backend.as_ref()
    }

    pub(crate) fn gather_shared_embedding(
        &self,
        tokens: &[u32],
        output: &mut [f32],
    ) -> Result<(), NnError> {
        self.embedding
            .gather_with_backend(self.backend.as_ref(), tokens, output)
    }

    /// Language-head logits for one hidden row of a bound output, `[vocab]`.
    ///
    /// [`Qwen35TextOutput::last_logits`] covers only the final row, which is all
    /// sequential decoding needs. Verifying a speculative draft needs the rows
    /// underneath it too: a forward over the accepted token plus `k` drafted
    /// ones has to answer what the target itself would have emitted at each
    /// position, and those answers live in rows `0..k`.
    ///
    /// Outputs minted by a different runner are rejected, matching the rest of
    /// this type's provenance handling.
    ///
    /// # Errors
    /// Returns [`NnError::Provenance`] for a foreign output, [`NnError::Shape`]
    /// for a row past the output's sequence, or a projection error.
    pub fn logits_for_row(
        &self,
        output: &Qwen35TextOutput,
        row: usize,
    ) -> Result<Vec<f32>, NnError> {
        if !Arc::ptr_eq(output.runner_identity(), &self.identity) {
            return Err(NnError::Provenance(
                "Qwen3.5 language head received an output from a different runner".to_owned(),
            ));
        }
        let sequence = output.sequence();
        if row >= sequence {
            return Err(NnError::Shape {
                expected: sequence,
                got: row,
            });
        }
        let start = checked_mul(row, self.hidden_size, "language-head hidden row")?;
        let mut logits = zeroed_scratch(self.vocab_size, "language-head row logits")?;
        self.project_shared_head(
            &output.final_hidden_states()[start..start + self.hidden_size],
            1,
            &mut logits,
        )?;
        Ok(logits)
    }

    pub(crate) fn project_shared_head(
        &self,
        hidden_states: &[f32],
        rows: usize,
        output: &mut [f32],
    ) -> Result<(), NnError> {
        self.lm_head
            .forward(self.backend.as_ref(), hidden_states, rows, output)
    }

    /// Allocate one exact-runner hybrid cache.
    ///
    /// # Errors
    ///
    /// Returns [`NnError::Shape`] when `max_context` is zero or exceeds the
    /// model limit, or [`NnError::Backend`] when a layer-cache allocation fails.
    pub fn new_cache(&self, max_context: usize) -> Result<Qwen35TextCache, NnError> {
        if max_context == 0 || max_context > self.max_context {
            return Err(NnError::Shape {
                expected: self.max_context,
                got: max_context,
            });
        }
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(self.layers.len())
            .map_err(|error| {
                NnError::Backend(format!(
                    "allocate Qwen3.5 cache layer table for {} layers: {error}",
                    self.layers.len()
                ))
            })?;
        for layer in &self.layers {
            layers.push(match &layer.mixer {
                Qwen35TextMixer::DeltaNet(mixer) => {
                    Qwen35TextLayerCache::DeltaNet(mixer.new_cache()?)
                }
                Qwen35TextMixer::FullAttention(mixer) => {
                    Qwen35TextLayerCache::FullAttention(mixer.new_cache(max_context)?)
                }
            });
        }
        Ok(Qwen35TextCache {
            runner_identity: Arc::clone(&self.identity),
            layers,
            committed_len: 0,
            max_context,
        })
    }

    /// Observe committed hybrid state in canonical ONNX graph output order.
    ///
    /// Owns immutable FP32 copies, bounded by `max_state_bytes` (1..=256 MiB
    /// of values). Never advances or lends out mutable cache state. Device-owned
    /// DeltaNet recurrence is rejected, not read from stale host buffers.
    /// Observations are numeric evidence, not qualification receipts.
    ///
    /// # Errors
    /// Returns [`NnError::Provenance`] for a foreign cache, or
    /// [`NnError::Backend`] for empty/inconsistent/device-owned/non-finite state,
    /// invalid budgets or allocation errors. No partial observation is returned.
    pub fn reference_states(
        &self,
        cache: &Qwen35TextCache,
        max_state_bytes: usize,
    ) -> Result<Vec<Qwen35ReferenceState>, NnError> {
        if !Arc::ptr_eq(&cache.runner_identity, &self.identity) {
            return Err(NnError::Provenance(
                "Qwen cache observation received a foreign runner cache".to_owned(),
            ));
        }
        self.validate_cache_layers(cache)?;
        if cache.is_empty()
            || cache.len() > cache.max_context
            || cache.max_context > self.max_context
        {
            return Err(NnError::Backend(
                "Qwen cache observation requires a nonempty committed cursor".to_owned(),
            ));
        }
        let mut views = Vec::new();
        views
            .try_reserve_exact(checked_mul(
                cache.layers.len(),
                2,
                "cache observation table",
            )?)
            .map_err(|error| {
                NnError::Backend(format!("allocate Qwen cache observation table: {error}"))
            })?;
        for (index, layer) in cache.layers.iter().enumerate() {
            match layer {
                Qwen35TextLayerCache::DeltaNet(state) => {
                    if state.is_device_resident() {
                        return Err(NnError::Backend(format!(
                            "Qwen cache observation layer {index} is device-owned; host state is not authoritative"
                        )));
                    }
                    views.push(ReferenceStateView {
                        name: format!("next_conv.{index}"),
                        shape: vec![state.conv_width(), state.conv_kernel_dim()],
                        values: state.conv_state(),
                    });
                    views.push(ReferenceStateView {
                        name: format!("next_recurrent.{index}"),
                        shape: vec![
                            state.num_value_heads(),
                            state.key_head_dim(),
                            state.value_head_dim(),
                        ],
                        values: state.recurrent_state(),
                    });
                }
                Qwen35TextLayerCache::FullAttention(state) => {
                    let shape = vec![
                        cache.len(),
                        axis(self.config.full_attention.num_key_value_heads, "KV heads")?,
                        axis(self.config.full_attention.head_dim, "KV head dimension")?,
                    ];
                    views.push(ReferenceStateView {
                        name: format!("present_k.{index}"),
                        shape: shape.clone(),
                        values: state.keys(),
                    });
                    views.push(ReferenceStateView {
                        name: format!("present_v.{index}"),
                        shape,
                        values: state.values(),
                    });
                }
            }
        }
        snapshot_states(views, max_state_bytes)
    }

    /// Run one initial prefill or one-token cached continuation transaction.
    ///
    /// Positions are derived as the contiguous interval beginning at
    /// [`Qwen35TextCache::len`]. Callers cannot provide divergent RoPE positions.
    /// Every DeltaNet layer stages state privately; full-attention KV appends are
    /// provisional until the final norm and untied language head both succeed.
    /// On any error all DeltaNet stages are discarded, every full-attention cache
    /// is restored to the global base, and no output object is returned.
    ///
    /// # Errors
    ///
    /// Returns [`NnError::Shape`] for an empty input or capacity mismatch,
    /// [`NnError::MissingTensor`] for an out-of-vocabulary token,
    /// [`NnError::MissingConfig`] for an unsafe multi-token cached DeltaNet
    /// continuation, [`NnError::Backend`] for foreign/inconsistent cache state,
    /// or an error from a layer operation.
    pub fn forward(
        &self,
        tokens: &[u32],
        cache: &mut Qwen35TextCache,
    ) -> Result<Qwen35TextOutput, NnError> {
        match self.forward_with_block_observer(tokens, cache, |_, _, _, _| Ok::<_, Infallible>(()))
        {
            Ok(output) => Ok(output),
            Err(Qwen35TextForwardError::Runtime(error)) => Err(error),
            Err(Qwen35TextForwardError::Observer(never)) => match never {},
        }
    }

    /// Execute one forward while borrowing each post-block residual matrix to an observer.
    ///
    /// Outputs are emitted in layer order and are valid only for the observer call. No
    /// per-layer activation history is retained by the runner.
    pub(crate) fn forward_with_block_observer<E>(
        &self,
        tokens: &[u32],
        cache: &mut Qwen35TextCache,
        observer: impl FnMut(u32, usize, &[u32], &[f32]) -> Result<(), E>,
    ) -> Result<Qwen35TextOutput, Qwen35TextForwardError<E>> {
        self.forward_with_block_and_state_observer(tokens, cache, &[], observer, |_, _, _| {})
    }

    /// Execute one forward while borrowing block outputs and selected DeltaNet states.
    ///
    /// State callbacks occur only at the requested zero-based token rows and only
    /// for DeltaNet layers. No state or activation history is retained by the runner.
    pub(crate) fn forward_with_block_and_state_observer<E>(
        &self,
        tokens: &[u32],
        cache: &mut Qwen35TextCache,
        state_positions: &[usize],
        mut observer: impl FnMut(u32, usize, &[u32], &[f32]) -> Result<(), E>,
        mut state_observer: impl FnMut(u32, usize, &[f32]),
    ) -> Result<Qwen35TextOutput, Qwen35TextForwardError<E>> {
        let (base, new_len) = self.preflight_forward(tokens, cache)?;
        let sequence = tokens.len();
        let hidden_len = checked_mul(sequence, self.hidden_size, "hidden-state buffer")?;
        let mut residual = zeroed_scratch(hidden_len, "embedding output")?;
        let mut normalized = zeroed_scratch(hidden_len, "normalized hidden state")?;
        let mut branch = zeroed_scratch(hidden_len, "decoder branch output")?;
        let mut positions = Vec::new();
        positions.try_reserve_exact(sequence).map_err(|error| {
            NnError::Backend(format!(
                "allocate Qwen3.5 position vector for {sequence} tokens: {error}"
            ))
        })?;
        positions.extend(base..new_len);
        let mut input_token_ids = Vec::new();
        input_token_ids
            .try_reserve_exact(sequence)
            .map_err(|error| {
                NnError::Backend(format!(
                    "allocate Qwen3.5 output token provenance for {sequence} tokens: {error}"
                ))
            })?;
        input_token_ids.extend_from_slice(tokens);
        self.embedding
            .gather_with_backend(self.backend.as_ref(), tokens, &mut residual)?;

        let result = self.forward_provisional(
            self.backend.as_ref(),
            base,
            input_token_ids,
            &positions,
            cache,
            &mut residual,
            &mut normalized,
            &mut branch,
            state_positions,
            &mut observer,
            &mut state_observer,
        );
        let output = match result {
            Ok(output) => output,
            Err(error) => {
                self.abort_and_rollback(cache, base);
                return Err(error);
            }
        };

        if let Err(error) = self.preflight_commit(cache, new_len) {
            self.abort_and_rollback(cache, base);
            return Err(error.into());
        }
        for layer in &mut cache.layers {
            if let Qwen35TextLayerCache::DeltaNet(cache) = layer {
                cache.commit_staged();
            }
        }
        cache.committed_len = new_len;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_provisional<E>(
        &self,
        backend: &dyn TernaryBackend,
        position_start: usize,
        input_token_ids: Vec<u32>,
        positions: &[usize],
        cache: &mut Qwen35TextCache,
        residual: &mut [f32],
        normalized: &mut [f32],
        branch: &mut [f32],
        state_positions: &[usize],
        observer: &mut impl FnMut(u32, usize, &[u32], &[f32]) -> Result<(), E>,
        state_observer: &mut impl FnMut(u32, usize, &[f32]),
    ) -> Result<Qwen35TextOutput, Qwen35TextForwardError<E>> {
        let sequence = input_token_ids.len();
        for (block_index, (layer, layer_cache)) in
            self.layers.iter().zip(&mut cache.layers).enumerate()
        {
            let block_index = u32::try_from(block_index).map_err(|_| {
                Qwen35TextForwardError::Runtime(NnError::ResourceExhausted(
                    "Qwen3.5 block index exceeds u32".to_owned(),
                ))
            })?;
            normalize_rows(
                residual,
                &layer.input_norm,
                self.rms_norm_eps,
                self.hidden_size,
                normalized,
            )?;
            match (&layer.mixer, layer_cache) {
                (Qwen35TextMixer::DeltaNet(mixer), Qwen35TextLayerCache::DeltaNet(cache)) => {
                    let mut report_state = |position: usize, state: &[f32]| {
                        state_observer(block_index, position, state);
                    };
                    let mut recurrent_observer =
                        RecurrentStateObserver::new(state_positions, &mut report_state);
                    mixer.stage_forward_with_state_observer(
                        backend,
                        normalized,
                        sequence,
                        cache,
                        branch,
                        &mut recurrent_observer,
                    )?
                }
                (
                    Qwen35TextMixer::FullAttention(mixer),
                    Qwen35TextLayerCache::FullAttention(cache),
                ) => mixer.forward(backend, normalized, positions, cache, branch)?,
                _ => {
                    return Err(NnError::Backend(
                        "Qwen3.5 cache layer kind changed after preflight".to_owned(),
                    )
                    .into());
                }
            }
            add_in_place(residual, branch);

            normalize_rows(
                residual,
                &layer.post_attention_norm,
                self.rms_norm_eps,
                self.hidden_size,
                normalized,
            )?;
            layer.mlp.forward(backend, normalized, sequence, branch)?;
            add_in_place(residual, branch);
            observer(block_index, position_start, &input_token_ids, residual)
                .map_err(Qwen35TextForwardError::Observer)?;
        }

        let mut final_hidden_states = zeroed_scratch(residual.len(), "final hidden states")?;
        normalize_rows(
            residual,
            &self.final_norm,
            self.rms_norm_eps,
            self.hidden_size,
            &mut final_hidden_states,
        )?;
        let mut last_logits = zeroed_scratch(self.vocab_size, "last-token logits")?;
        let last_start = checked_mul(sequence - 1, self.hidden_size, "last hidden row")?;
        self.lm_head.forward(
            backend,
            &final_hidden_states[last_start..last_start + self.hidden_size],
            1,
            &mut last_logits,
        )?;
        if final_hidden_states
            .iter()
            .chain(&last_logits)
            .any(|value| !value.is_finite())
        {
            return Err(NnError::Backend(
                "Qwen3.5 text forward produced a non-finite value".to_owned(),
            )
            .into());
        }
        Ok(Qwen35TextOutput {
            runner_identity: Arc::clone(&self.identity),
            position_start,
            hidden_size: self.hidden_size,
            input_token_ids,
            final_hidden_states,
            last_logits,
        })
    }

    fn preflight_forward(
        &self,
        tokens: &[u32],
        cache: &Qwen35TextCache,
    ) -> Result<(usize, usize), NnError> {
        if !Arc::ptr_eq(&cache.runner_identity, &self.identity) {
            return Err(NnError::Backend(
                "Qwen3.5 text cache belongs to a different runner".to_owned(),
            ));
        }
        if tokens.is_empty() {
            return Err(NnError::Shape {
                expected: 1,
                got: 0,
            });
        }
        self.validate_cache_layers(cache)?;
        if cache.committed_len != 0 && self.has_delta_net && tokens.len() != 1 {
            return Err(invalid_config(
                "Qwen3.5 cached DeltaNet continuation must contain exactly one token",
            ));
        }
        if let Some(&token) = tokens
            .iter()
            .find(|&&token| usize::try_from(token).map_or(true, |token| token >= self.vocab_size))
        {
            return Err(NnError::MissingTensor(format!("token_embd row {token}")));
        }
        let new_len = cache
            .committed_len
            .checked_add(tokens.len())
            .ok_or(NnError::Shape {
                expected: cache.max_context,
                got: usize::MAX,
            })?;
        if new_len > cache.max_context {
            return Err(NnError::Shape {
                expected: cache.max_context,
                got: new_len,
            });
        }
        Ok((cache.committed_len, new_len))
    }

    fn validate_cache_layers(&self, cache: &Qwen35TextCache) -> Result<(), NnError> {
        if cache.layers.len() != self.layers.len() {
            return Err(NnError::Backend(
                "Qwen3.5 text cache layer count is inconsistent".to_owned(),
            ));
        }
        for (index, (layer, layer_cache)) in self.layers.iter().zip(&cache.layers).enumerate() {
            if layer.mixer.kind() != layer_cache.kind()
                || layer_cache.committed_len() != cache.committed_len
                || matches!(
                    layer_cache,
                    Qwen35TextLayerCache::DeltaNet(cache) if cache.staged_len().is_some()
                )
            {
                return Err(NnError::Backend(format!(
                    "Qwen3.5 text cache layer {index} is inconsistent with the global cursor"
                )));
            }
        }
        Ok(())
    }

    fn preflight_commit(&self, cache: &Qwen35TextCache, new_len: usize) -> Result<(), NnError> {
        for (index, layer) in cache.layers.iter().enumerate() {
            let valid = match layer {
                Qwen35TextLayerCache::DeltaNet(cache) => cache.staged_len() == Some(new_len),
                Qwen35TextLayerCache::FullAttention(cache) => cache.committed_len() == new_len,
            };
            if !valid {
                return Err(NnError::Backend(format!(
                    "Qwen3.5 text layer {index} failed global commit preflight"
                )));
            }
        }
        Ok(())
    }

    fn abort_and_rollback(&self, cache: &mut Qwen35TextCache, base: usize) {
        for layer in &mut cache.layers {
            match layer {
                Qwen35TextLayerCache::DeltaNet(cache) => cache.abort_staged(),
                Qwen35TextLayerCache::FullAttention(cache) => cache.rollback_to(base),
            }
        }
    }
}

fn validate_schedule(layer_types: &[Qwen35LayerType], interval: usize) -> Result<(), NnError> {
    for (index, &kind) in layer_types.iter().enumerate() {
        let layer_number = index
            .checked_add(1)
            .ok_or_else(|| invalid_config("Qwen3.5 layer schedule index overflow"))?;
        let expected = if layer_number.is_multiple_of(interval) {
            Qwen35LayerType::FullAttention
        } else {
            Qwen35LayerType::DeltaNet
        };
        if kind != expected {
            return Err(invalid_config(format!(
                "Qwen3.5 layer {index} contradicts full_attention_interval"
            )));
        }
    }
    Ok(())
}

fn validate_token_table(
    embedding: &TokenEmbedding,
    rows: usize,
    cols: usize,
) -> Result<(), NnError> {
    if embedding.rows() != rows {
        return Err(NnError::Shape {
            expected: rows,
            got: embedding.rows(),
        });
    }
    if embedding.cols() != cols {
        return Err(NnError::Shape {
            expected: cols,
            got: embedding.cols(),
        });
    }
    if embedding
        .as_dense()
        .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
    {
        return Err(NnError::Backend(
            "Qwen3.5 token embedding contains a non-finite value".to_owned(),
        ));
    }
    Ok(())
}

fn validate_projection(
    projection: &Projection,
    expected_out: usize,
    expected_in: usize,
) -> Result<(), NnError> {
    if projection.n_out() != expected_out {
        return Err(NnError::Shape {
            expected: expected_out,
            got: projection.n_out(),
        });
    }
    if projection.k_in() != expected_in {
        return Err(NnError::Shape {
            expected: expected_in,
            got: projection.k_in(),
        });
    }
    Ok(())
}

fn validate_finite_projection(projection: &Projection, name: &str) -> Result<(), NnError> {
    projection
        .validate_retained_parameters()
        .map_err(|error| match error {
            NnError::Backend(message) => NnError::Backend(format!("Qwen3.5 {name}: {message}")),
            other => other,
        })
}

fn validate_finite(values: &[f32], expected: usize, name: &str) -> Result<(), NnError> {
    if values.len() != expected {
        return Err(NnError::Shape {
            expected,
            got: values.len(),
        });
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(NnError::Backend(format!(
            "Qwen3.5 {name} contains a non-finite value"
        )));
    }
    Ok(())
}

fn normalize_rows(
    input: &[f32],
    weights: &[f32],
    epsilon: f32,
    hidden_size: usize,
    output: &mut [f32],
) -> Result<(), NnError> {
    if input.len() != output.len() {
        return Err(NnError::Shape {
            expected: input.len(),
            got: output.len(),
        });
    }
    for (source, destination) in input
        .chunks_exact(hidden_size)
        .zip(output.chunks_exact_mut(hidden_size))
    {
        rmsnorm_zero_centered(source, weights, epsilon, destination)?;
    }
    Ok(())
}

fn add_in_place(residual: &mut [f32], branch: &[f32]) {
    debug_assert_eq!(residual.len(), branch.len());
    for (residual, &branch) in residual.iter_mut().zip(branch) {
        *residual += branch;
    }
}

fn axis(value: u32, name: &str) -> Result<usize, NnError> {
    if value == 0 {
        return Err(invalid_config(format!("Qwen3.5 {name} must be non-zero")));
    }
    usize::try_from(value).map_err(|_| invalid_config(format!("Qwen3.5 {name} does not fit usize")))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, NnError> {
    left.checked_mul(right)
        .ok_or_else(|| NnError::Backend(format!("Qwen3.5 {name} extent overflow")))
}

fn zeroed_scratch(len: usize, name: &str) -> Result<Vec<f32>, NnError> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|error| {
        NnError::Backend(format!(
            "allocate Qwen3.5 {name} for {len} f32 values: {error}"
        ))
    })?;
    values.resize(len, 0.0);
    Ok(values)
}

fn invalid_config(reason: impl Into<String>) -> NnError {
    NnError::MissingConfig(reason.into())
}
