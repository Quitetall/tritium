//! Production model adapter for SwiGLU training campaigns.
//!
//! The adapter turns the inference model representation into the stable parameter
//! order consumed by the device-resident training stack:
//!
//! `embed`, then `q, k, v, o, gate, up, down` for each layer, followed by an
//! optional untied `lm_head.weight`.
//!
//! It validates every architecture axis and tensor shape before cloning a host
//! master. Unsupported variants fail explicitly instead of being run through a
//! silently incomplete training graph.

use crate::{
    ArchSpec, DenseLinear, Mlp, MlpKind, ModelConfig, ModelWeights, Projection, SwiGluMlp,
    TokenEmbedding, TransformerBlock,
};

/// Failure to adapt or execute a SwiGLU training model.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TrainingAdapterError {
    /// The model uses an architecture axis the packed training graph does not implement.
    UnsupportedArchitecture(String),
    /// A named tensor's scalar count disagrees with the canonical model geometry.
    TensorShape {
        /// Canonical HuggingFace tensor or model-component name.
        name: String,
        /// Required scalar count.
        expected: usize,
        /// Supplied scalar count.
        got: usize,
    },
    /// A runtime caller input is invalid.
    InvalidInput(String),
    /// A bounded host allocation could not be reserved.
    AllocationFailed {
        /// Logical allocation site.
        allocation: &'static str,
        /// Requested payload bytes, excluding allocator metadata.
        requested_bytes: usize,
    },
    /// A device operation failed.
    Backend(String),
}

impl core::fmt::Display for TrainingAdapterError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedArchitecture(reason) => {
                write!(formatter, "unsupported training architecture: {reason}")
            }
            Self::TensorShape {
                name,
                expected,
                got,
            } => write!(
                formatter,
                "training tensor {name} expected {expected} elements, got {got}"
            ),
            Self::InvalidInput(reason) => write!(formatter, "invalid training input: {reason}"),
            Self::AllocationFailed {
                allocation,
                requested_bytes,
            } => write!(
                formatter,
                "training allocation failed: {allocation} ({requested_bytes} requested bytes)"
            ),
            Self::Backend(reason) => write!(formatter, "training backend error: {reason}"),
        }
    }
}

impl std::error::Error for TrainingAdapterError {}

#[cfg(feature = "cuda")]
impl From<tritium_spec::BackendError> for TrainingAdapterError {
    fn from(error: tritium_spec::BackendError) -> Self {
        Self::Backend(error.to_string())
    }
}

/// Fixed transformer geometry and fp32 norm weights used by the training graph.
#[derive(Clone, Debug, PartialEq)]
pub struct TiedSwiGluTrainingArchitecture {
    /// Per-layer pre-attention RMSNorm weights, each `[n_embd]`.
    pub attn_norms: Vec<Vec<f32>>,
    /// Per-layer pre-MLP RMSNorm weights, each `[n_embd]`.
    pub ffn_norms: Vec<Vec<f32>>,
    /// Per-layer fixed Qwen attention vectors, in transformer order.
    pub attention_constants: Vec<TrainingAttentionConstants>,
    /// Final RMSNorm weight, `[n_embd]`.
    pub output_norm: Vec<f32>,
    /// Residual-stream width.
    pub n_embd: usize,
    /// Query-head count.
    pub n_head: usize,
    /// Key/value-head count.
    pub n_head_kv: usize,
    /// Width of one attention head.
    pub head_dim: usize,
    /// SwiGLU intermediate width.
    pub n_ff: usize,
    /// Token vocabulary size.
    pub vocab: usize,
    /// RMSNorm epsilon.
    pub rms_eps: f32,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// Transformer-block count.
    pub n_layers: usize,
    /// Maximum configured sequence length.
    pub n_ctx: usize,
}

/// Preserved non-matrix constants consumed by one standard attention layer.
///
/// Bias vectors are empty or match their projection output width. Q/K norm
/// vectors are empty or `[head_dim]` and are shared across heads. They remain
/// fixed during matrix refinement and therefore do not appear in
/// [`TiedSwiGluTrainingModel::parameters`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrainingAttentionConstants {
    /// Query-projection bias, `[n_head * head_dim]`.
    pub q_bias: Vec<f32>,
    /// Key-projection bias, `[n_head_kv * head_dim]`.
    pub k_bias: Vec<f32>,
    /// Value-projection bias, `[n_head_kv * head_dim]`.
    pub v_bias: Vec<f32>,
    /// Per-head query RMSNorm weight, `[head_dim]`.
    pub q_norm: Vec<f32>,
    /// Per-head key RMSNorm weight, `[head_dim]`.
    pub k_norm: Vec<f32>,
}

/// One canonical HuggingFace 2D trainable tensor and its latent fp32 master.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingParameter {
    /// Canonical HuggingFace tensor name.
    pub name: String,
    /// Row-major latent fp32 master.
    pub master: Vec<f32>,
    /// Matrix output dimension (rows).
    pub rows: usize,
    /// Matrix input dimension (columns).
    pub cols: usize,
}

impl TrainingParameter {
    /// Number of logical scalar elements in this matrix.
    ///
    /// This remains available after the owned master has moved into an
    /// optimizer; the canonical matrix geometry, rather than `master.len()`, is
    /// the graph contract.
    #[must_use]
    pub fn elements(&self) -> usize {
        self.rows.saturating_mul(self.cols)
    }
}

/// Validated, owned inputs for a SwiGLU device-training campaign.
#[derive(Clone, Debug, PartialEq)]
pub struct TiedSwiGluTrainingModel {
    arch: TiedSwiGluTrainingArchitecture,
    parameters: Vec<TrainingParameter>,
    lm_head_tied: bool,
}

struct StagedIntermediateLayer {
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
}

struct OriginalIntermediateLayer {
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
}

/// Provisional intermediate-width mutation that restores the source tensors
/// unless explicitly committed.
///
/// Construction stages every widened MLP tensor before changing the model. Once
/// installed, the original MLP tensors remain owned by this guard so callers can
/// run fallible validation against the widened candidate without cloning the
/// embedding, attention, norms, or output head.
pub(crate) struct IntermediateWideningTransaction<'model> {
    model: &'model mut TiedSwiGluTrainingModel,
    original_width: usize,
    originals: Vec<OriginalIntermediateLayer>,
    plan: Option<tritium_train::Net2WiderPlan>,
    committed: bool,
}

impl IntermediateWideningTransaction<'_> {
    /// Widened candidate visible to post-transform validation.
    pub(crate) fn model(&self) -> &TiedSwiGluTrainingModel {
        self.model
    }

    /// Exact deterministic plan installed in the provisional candidate.
    pub(crate) fn plan(&self) -> &tritium_train::Net2WiderPlan {
        self.plan
            .as_ref()
            .expect("uncommitted widening transaction retains its plan")
    }

    /// Keep the widened tensors and return their deterministic transform plan.
    pub(crate) fn commit(mut self) -> tritium_train::Net2WiderPlan {
        let plan = self
            .plan
            .take()
            .expect("uncommitted widening transaction retains its plan");
        self.committed = true;
        plan
    }

    fn rollback(&mut self) {
        for (layer_index, original) in self.originals.drain(..).enumerate() {
            let base = 1 + 7 * layer_index;
            let gate = &mut self.model.parameters[base + 4];
            gate.master = original.gate;
            gate.rows = self.original_width;

            let up = &mut self.model.parameters[base + 5];
            up.master = original.up;
            up.rows = self.original_width;

            let down = &mut self.model.parameters[base + 6];
            down.master = original.down;
            down.cols = self.original_width;
        }
        self.model.arch.n_ff = self.original_width;
    }
}

impl Drop for IntermediateWideningTransaction<'_> {
    fn drop(&mut self) {
        if !self.committed {
            self.rollback();
        }
    }
}

/// Hash every semantic input consumed by a SwiGLU training graph.
///
/// The digest binds the HuggingFace configuration, supported architecture flags,
/// canonical parameter names/shapes/masters, and fixed norm vectors. It is the
/// authoritative teacher-cache and campaign model identity; callers must not
/// substitute a value-only weight hash because that omits graph semantics.
#[must_use]
pub fn semantic_training_model_digest(
    config: &ModelConfig,
    spec: &ArchSpec,
    model: &TiedSwiGluTrainingModel,
) -> [u8; 32] {
    let mut hash = blake3::Hasher::new();
    hash.update(b"tritium-tied-swiglu-training-model-v1");
    digest_bytes(&mut hash, config.arch.as_bytes());
    for value in [
        config.n_layers,
        config.n_embd,
        config.n_head,
        config.n_head_kv,
        config.head_dim,
        config.n_ff,
        config.n_ctx,
    ] {
        hash.update(&(value as u64).to_le_bytes());
    }
    hash.update(&config.rope_theta.to_bits().to_le_bytes());
    hash.update(&config.rms_eps.to_bits().to_le_bytes());
    hash.update(&[match spec.mlp {
        MlpKind::Relu2 => 0,
        MlpKind::SwiGlu => 1,
    }]);
    hash.update(&[
        u8::from(spec.attn_sub_norm),
        u8::from(spec.ffn_sub_norm),
        u8::from(spec.qk_norm),
        u8::from(spec.qkv_bias),
        u8::from(spec.tied_embeddings),
    ]);
    hash.update(&(model.parameters.len() as u64).to_le_bytes());
    for parameter in &model.parameters {
        digest_bytes(&mut hash, parameter.name.as_bytes());
        hash.update(&(parameter.rows as u64).to_le_bytes());
        hash.update(&(parameter.cols as u64).to_le_bytes());
        digest_f32s(&mut hash, &parameter.master);
    }
    hash.update(&(model.arch.attn_norms.len() as u64).to_le_bytes());
    for norm in &model.arch.attn_norms {
        digest_f32s(&mut hash, norm);
    }
    hash.update(&(model.arch.ffn_norms.len() as u64).to_le_bytes());
    for norm in &model.arch.ffn_norms {
        digest_f32s(&mut hash, norm);
    }
    if model.arch.attention_constants.iter().any(|constants| {
        !constants.q_bias.is_empty()
            || !constants.k_bias.is_empty()
            || !constants.v_bias.is_empty()
            || !constants.q_norm.is_empty()
            || !constants.k_norm.is_empty()
    }) {
        hash.update(b"tritium-standard-qwen-attention-constants-v1");
        hash.update(&(model.arch.attention_constants.len() as u64).to_le_bytes());
        for constants in &model.arch.attention_constants {
            digest_f32s(&mut hash, &constants.q_bias);
            digest_f32s(&mut hash, &constants.k_bias);
            digest_f32s(&mut hash, &constants.v_bias);
            digest_f32s(&mut hash, &constants.q_norm);
            digest_f32s(&mut hash, &constants.k_norm);
        }
    }
    digest_f32s(&mut hash, &model.arch.output_norm);
    *hash.finalize().as_bytes()
}

fn digest_bytes(hash: &mut blake3::Hasher, bytes: &[u8]) {
    hash.update(&(bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

fn digest_f32s(hash: &mut blake3::Hasher, values: &[f32]) {
    hash.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        hash.update(&value.to_bits().to_le_bytes());
    }
}

impl TiedSwiGluTrainingModel {
    /// Fixed architecture and fp32 norm weights used by the training graph.
    #[must_use]
    pub fn architecture(&self) -> &TiedSwiGluTrainingArchitecture {
        &self.arch
    }

    /// Canonically ordered 2D latent masters.
    #[must_use]
    pub fn parameters(&self) -> &[TrainingParameter] {
        &self.parameters
    }

    /// Whether the output projection shares the embedding parameter.
    #[must_use]
    pub const fn is_lm_head_tied(&self) -> bool {
        self.lm_head_tied
    }

    #[cfg(feature = "cuda")]
    fn lm_head_parameter_index(&self) -> usize {
        if self.lm_head_tied {
            0
        } else {
            self.parameters.len() - 1
        }
    }

    /// Move every latent master out in canonical parameter order while
    /// retaining names and matrix geometry for packed graph construction.
    ///
    /// This is the scale-safe handoff to a host-offloaded optimizer: it avoids
    /// retaining a fourth full fp32 model plane beside master, first moment,
    /// and second moment. After this call, [`Self::parameters`] still describes
    /// the complete graph, but each `TrainingParameter::master` is empty.
    #[must_use]
    pub fn take_parameter_masters(&mut self) -> Vec<Vec<f32>> {
        self.parameters
            .iter_mut()
            .map(|parameter| core::mem::take(&mut parameter.master))
            .collect()
    }

    /// Reconstruct exact-fp inference weights from the canonical latent masters.
    ///
    /// Every trainable projection uses [`DenseLinear::new_exact`], so a caller can
    /// score the current student through [`crate::ModelRunner`] without activation
    /// quantization. The current architecture is authoritative, including a widened
    /// [`Self::architecture`]'s `n_ff`; callers must pair the returned weights with a
    /// [`ModelConfig`] carrying that same intermediate width. Tied heads remain
    /// tied to the embedding, while an untied final `lm_head.weight` is reconstructed
    /// as its own dense projection.
    ///
    /// This method clones every master and all fixed norm vectors. It therefore
    /// temporarily retains one additional full dense fp32 model plane and is meant
    /// for bounded evaluation before [`Self::take_parameter_masters`] hands ownership
    /// to an optimizer. It intentionally fails after that scale-safe handoff instead
    /// of silently building incomplete weights.
    ///
    /// # Errors
    /// Returns [`TrainingAdapterError::InvalidInput`] if canonical parameter
    /// geometry is inconsistent or any latent master has been drained, and
    /// [`TrainingAdapterError::TensorShape`] if fixed norm geometry is invalid.
    pub fn to_dense_weights(&self) -> Result<ModelWeights, TrainingAdapterError> {
        let arch = &self.arch;
        let expected_parameters = arch
            .n_layers
            .checked_mul(7)
            .and_then(|count| count.checked_add(1))
            .and_then(|count| count.checked_add(usize::from(!self.lm_head_tied)))
            .ok_or_else(|| invalid_input("parameter count overflows usize"))?;
        if self.parameters.len() != expected_parameters {
            return Err(tensor_mismatch(
                "canonical training parameter count",
                expected_parameters,
                self.parameters.len(),
            ));
        }
        if arch.attn_norms.len() != arch.n_layers {
            return Err(tensor_mismatch(
                "attention norm count",
                arch.n_layers,
                arch.attn_norms.len(),
            ));
        }
        if arch.ffn_norms.len() != arch.n_layers {
            return Err(tensor_mismatch(
                "FFN norm count",
                arch.n_layers,
                arch.ffn_norms.len(),
            ));
        }
        if arch.attention_constants.len() != arch.n_layers {
            return Err(tensor_mismatch(
                "attention constant set count",
                arch.n_layers,
                arch.attention_constants.len(),
            ));
        }
        let q_width = arch
            .n_head
            .checked_mul(arch.head_dim)
            .ok_or_else(|| invalid_input("query projection width overflows usize"))?;
        let kv_width = arch
            .n_head_kv
            .checked_mul(arch.head_dim)
            .ok_or_else(|| invalid_input("key/value projection width overflows usize"))?;
        validate_vector("model.norm.weight", &arch.output_norm, arch.n_embd)?;
        for layer_index in 0..arch.n_layers {
            let constants = &arch.attention_constants[layer_index];
            validate_vector(
                &format!("model.layers.{layer_index}.input_layernorm.weight"),
                &arch.attn_norms[layer_index],
                arch.n_embd,
            )?;
            validate_vector(
                &format!("model.layers.{layer_index}.post_attention_layernorm.weight"),
                &arch.ffn_norms[layer_index],
                arch.n_embd,
            )?;
            validate_optional_vector(
                &format!("model.layers.{layer_index}.self_attn.q_proj.bias"),
                &constants.q_bias,
                q_width,
            )?;
            validate_optional_vector(
                &format!("model.layers.{layer_index}.self_attn.k_proj.bias"),
                &constants.k_bias,
                kv_width,
            )?;
            validate_optional_vector(
                &format!("model.layers.{layer_index}.self_attn.v_proj.bias"),
                &constants.v_bias,
                kv_width,
            )?;
            validate_optional_vector(
                &format!("model.layers.{layer_index}.self_attn.q_norm.weight"),
                &constants.q_norm,
                arch.head_dim,
            )?;
            validate_optional_vector(
                &format!("model.layers.{layer_index}.self_attn.k_norm.weight"),
                &constants.k_norm,
                arch.head_dim,
            )?;
        }

        let expected_geometry = core::iter::once((arch.vocab, arch.n_embd))
            .chain((0..arch.n_layers).flat_map(|_| {
                [
                    (q_width, arch.n_embd),
                    (kv_width, arch.n_embd),
                    (kv_width, arch.n_embd),
                    (arch.n_embd, q_width),
                    (arch.n_ff, arch.n_embd),
                    (arch.n_ff, arch.n_embd),
                    (arch.n_embd, arch.n_ff),
                ]
            }))
            .chain((!self.lm_head_tied).then_some((arch.vocab, arch.n_embd)));
        for (index, (parameter, (rows, cols))) in
            self.parameters.iter().zip(expected_geometry).enumerate()
        {
            validate_parameter_geometry(parameter, rows, cols)?;
            if parameter.master.len() != parameter.elements() {
                return Err(invalid_input(&format!(
                    "{} master at canonical index {index} is drained or shape-inconsistent",
                    parameter.name
                )));
            }
        }

        let dense = |index: usize| -> Result<Projection, TrainingAdapterError> {
            let parameter = &self.parameters[index];
            DenseLinear::new_exact(parameter.master.clone(), parameter.rows, parameter.cols)
                .map(Projection::Dense)
                .map_err(|error| invalid_input(&format!("{}: {error}", parameter.name)))
        };
        let mut layers = Vec::with_capacity(arch.n_layers);
        for layer_index in 0..arch.n_layers {
            let base = 1 + 7 * layer_index;
            let constants = &arch.attention_constants[layer_index];
            layers.push(TransformerBlock {
                attn_norm: arch.attn_norms[layer_index].clone(),
                q_proj: dense(base)?,
                k_proj: dense(base + 1)?,
                v_proj: dense(base + 2)?,
                o_proj: dense(base + 3)?,
                attn_sub_norm: Vec::new(),
                q_bias: constants.q_bias.clone(),
                k_bias: constants.k_bias.clone(),
                v_bias: constants.v_bias.clone(),
                q_norm: constants.q_norm.clone(),
                k_norm: constants.k_norm.clone(),
                ffn_norm: arch.ffn_norms[layer_index].clone(),
                mlp: Mlp::SwiGlu(
                    SwiGluMlp::new(dense(base + 4)?, dense(base + 5)?, dense(base + 6)?).map_err(
                        |error| invalid_input(&format!("layer {layer_index} SwiGLU: {error}")),
                    )?,
                ),
            });
        }
        let lm_head = if self.lm_head_tied {
            None
        } else {
            Some(dense(self.parameters.len() - 1)?)
        };

        Ok(ModelWeights {
            token_embd: TokenEmbedding::from_dense(
                self.parameters[0].master.clone(),
                arch.vocab,
                arch.n_embd,
            )
            .map_err(|error| invalid_input(&format!("model.embed_tokens.weight: {error}")))?,
            vocab: arch.vocab,
            n_embd: arch.n_embd,
            layers,
            output_norm: arch.output_norm.clone(),
            lm_head,
        })
    }

    /// Deterministically widen every SwiGLU intermediate axis in place.
    ///
    /// One [`tritium_train::Net2WiderPlan`] is shared by all transformer layers:
    /// gate and up rows are copied from the selected source units, while down
    /// columns receive the plan's positive unequal dyadic shares. Those shares
    /// sum to one per source unit, preserving the dense fp32 SwiGLU function
    /// before SALT quantization while breaking clone permutation symmetry.
    /// Canonical parameter names and order are unchanged.
    ///
    /// The model is fully preflighted and every widened MLP tensor is staged
    /// before the first source tensor or geometry field is changed. A staging
    /// failure therefore leaves the complete model byte-for-byte unchanged.
    /// Temporary storage is the widened MLP plane; embeddings, attention, norms,
    /// and the output head are never cloned.
    ///
    /// # Errors
    /// Returns [`TrainingAdapterError::InvalidInput`] when `new_width` narrows the
    /// model, a size overflows, parameter geometry is inconsistent, or latent
    /// masters have already moved into an optimizer, and
    /// [`TrainingAdapterError::AllocationFailed`] when staged MLP or rollback
    /// storage cannot be reserved.
    pub fn widen_intermediate(
        &mut self,
        new_width: usize,
        seed: u64,
    ) -> Result<tritium_train::Net2WiderPlan, TrainingAdapterError> {
        Ok(self.begin_intermediate_widening(new_width, seed)?.commit())
    }

    /// Install a provisional widening whose source MLP tensors remain available
    /// for automatic rollback until the caller commits it.
    pub(crate) fn begin_intermediate_widening(
        &mut self,
        new_width: usize,
        seed: u64,
    ) -> Result<IntermediateWideningTransaction<'_>, TrainingAdapterError> {
        self.begin_intermediate_widening_with_hook(new_width, seed, |_| Ok(()))
    }

    fn begin_intermediate_widening_with_hook<F>(
        &mut self,
        new_width: usize,
        seed: u64,
        mut after_staged_projection: F,
    ) -> Result<IntermediateWideningTransaction<'_>, TrainingAdapterError>
    where
        F: FnMut(usize) -> Result<(), TrainingAdapterError>,
    {
        let old_width = self.arch.n_ff;
        let expected_parameters = self
            .arch
            .n_layers
            .checked_mul(7)
            .and_then(|count| count.checked_add(1))
            .and_then(|count| count.checked_add(usize::from(!self.lm_head_tied)))
            .ok_or_else(|| invalid_input("parameter count overflows usize"))?;
        if self.parameters.len() != expected_parameters {
            return Err(tensor_mismatch(
                "canonical training parameter count",
                expected_parameters,
                self.parameters.len(),
            ));
        }

        for (index, parameter) in self.parameters.iter().enumerate() {
            if parameter.master.len() != parameter.elements() {
                return Err(invalid_input(&format!(
                    "{} master at canonical index {index} is drained or shape-inconsistent",
                    parameter.name
                )));
            }
        }
        for layer_index in 0..self.arch.n_layers {
            let base = 1 + 7 * layer_index;
            validate_parameter_geometry(&self.parameters[base + 4], old_width, self.arch.n_embd)?;
            validate_parameter_geometry(&self.parameters[base + 5], old_width, self.arch.n_embd)?;
            validate_parameter_geometry(&self.parameters[base + 6], self.arch.n_embd, old_width)?;
        }
        new_width
            .checked_mul(self.arch.n_embd)
            .ok_or_else(|| invalid_input("widened incoming projection overflows usize"))?;
        self.arch
            .n_embd
            .checked_mul(new_width)
            .ok_or_else(|| invalid_input("widened outgoing projection overflows usize"))?;
        let plan = tritium_train::Net2WiderPlan::seeded(old_width, new_width, seed)
            .map_err(|error| invalid_input(&error.to_string()))?;
        if new_width == old_width {
            return Ok(IntermediateWideningTransaction {
                model: self,
                original_width: old_width,
                originals: Vec::new(),
                plan: Some(plan),
                committed: false,
            });
        }

        // Do every fallible transform before changing a source tensor. An error or
        // unwind during staging therefore observes `self` before mutable commit.
        let mut staged = Vec::new();
        staged.try_reserve_exact(self.arch.n_layers).map_err(|_| {
            allocation_failed::<StagedIntermediateLayer>(
                "widened MLP staging journal",
                self.arch.n_layers,
            )
        })?;
        let mut staged_projection_count = 0;
        for layer_index in 0..self.arch.n_layers {
            let base = 1 + 7 * layer_index;
            let gate = expand_incoming_rows_fallible(
                &plan,
                &self.parameters[base + 4].master,
                self.arch.n_embd,
            )?;
            staged_projection_count += 1;
            after_staged_projection(staged_projection_count)?;

            let up = expand_incoming_rows_fallible(
                &plan,
                &self.parameters[base + 5].master,
                self.arch.n_embd,
            )?;
            staged_projection_count += 1;
            after_staged_projection(staged_projection_count)?;

            let down = expand_outgoing_columns_fallible(
                &plan,
                &self.parameters[base + 6].master,
                self.arch.n_embd,
            )?;
            staged_projection_count += 1;
            after_staged_projection(staged_projection_count)?;
            staged.push(StagedIntermediateLayer { gate, up, down });
        }

        // Reserve the rollback journal before the first swap. From here through
        // transaction construction, all operations are fixed-capacity moves and
        // in-bounds writes established by the preflight above.
        let mut originals = Vec::new();
        originals
            .try_reserve_exact(self.arch.n_layers)
            .map_err(|_| {
                allocation_failed::<OriginalIntermediateLayer>(
                    "growth rollback journal",
                    self.arch.n_layers,
                )
            })?;
        for (layer_index, replacement) in staged.into_iter().enumerate() {
            let base = 1 + 7 * layer_index;
            let gate = &mut self.parameters[base + 4];
            let original_gate = core::mem::replace(&mut gate.master, replacement.gate);
            gate.rows = new_width;

            let up = &mut self.parameters[base + 5];
            let original_up = core::mem::replace(&mut up.master, replacement.up);
            up.rows = new_width;

            let down = &mut self.parameters[base + 6];
            let original_down = core::mem::replace(&mut down.master, replacement.down);
            down.cols = new_width;

            originals.push(OriginalIntermediateLayer {
                gate: original_gate,
                up: original_up,
                down: original_down,
            });
        }
        self.arch.n_ff = new_width;
        Ok(IntermediateWideningTransaction {
            model: self,
            original_width: old_width,
            originals,
            plan: Some(plan),
            committed: false,
        })
    }

    /// Validate that an HF architecture can use the packed training graph.
    ///
    /// This is a cheap preflight that can run immediately after parsing
    /// `config.json`, before loading large tensor shards.
    ///
    /// # Errors
    /// Returns [`TrainingAdapterError::UnsupportedArchitecture`] for unsupported
    /// architecture axes or invalid geometry.
    pub fn validate_config(
        config: &ModelConfig,
        spec: &ArchSpec,
    ) -> Result<(), TrainingAdapterError> {
        if spec.mlp != MlpKind::SwiGlu {
            return Err(unsupported("only SwiGLU MLPs are supported"));
        }
        if spec.attn_sub_norm || spec.ffn_sub_norm {
            return Err(unsupported(
                "attention/FFN sub-norms are not supported by the SwiGLU graph",
            ));
        }
        if config.n_layers == 0 {
            return Err(unsupported("num_hidden_layers must be non-zero"));
        }
        if config.n_embd == 0 {
            return Err(unsupported("hidden_size must be non-zero"));
        }
        if config.n_ff == 0 {
            return Err(unsupported("intermediate_size must be non-zero"));
        }
        if config.n_head == 0 || config.n_head_kv == 0 || config.head_dim() == 0 {
            return Err(unsupported(
                "attention head counts and head_dim must be non-zero",
            ));
        }
        if !config.n_head.is_multiple_of(config.n_head_kv) {
            return Err(unsupported(
                "num_attention_heads must be divisible by num_key_value_heads",
            ));
        }
        if !config.head_dim().is_multiple_of(2) {
            return Err(unsupported("head_dim must be even for RoPE"));
        }
        if config.n_ctx == 0 {
            return Err(unsupported("max_position_embeddings must be non-zero"));
        }
        if !config.rms_eps.is_finite() || config.rms_eps <= 0.0 {
            return Err(unsupported("rms_norm_eps must be finite and positive"));
        }
        if !config.rope_theta.is_finite() || config.rope_theta <= 0.0 {
            return Err(unsupported("rope_theta must be finite and positive"));
        }
        Ok(())
    }

    /// Extract the canonical host masters and fixed norms from loaded HF weights.
    ///
    /// Parameter order is stable: `model.embed_tokens.weight`, followed by each
    /// layer's `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
    /// `down_proj` tensors. An untied `lm_head.weight`, when configured, is
    /// appended last. This is the order returned as gradient-only leaves by
    /// `packed_device_forward` when CUDA is enabled.
    ///
    /// # Errors
    /// Returns an error before producing a model if the configuration is
    /// unsupported, a required projection is not dense fp32, an optional feature
    /// omitted by the training graph is present, or any tensor shape disagrees
    /// with the configuration.
    pub fn extract(
        config: &ModelConfig,
        spec: &ArchSpec,
        weights: &ModelWeights,
    ) -> Result<Self, TrainingAdapterError> {
        Self::validate_config(config, spec)?;

        let n_layers = usize_from_u32(config.n_layers, "num_hidden_layers")?;
        let n_embd = usize_from_u32(config.n_embd, "hidden_size")?;
        let n_head = usize_from_u32(config.n_head, "num_attention_heads")?;
        let n_head_kv = usize_from_u32(config.n_head_kv, "num_key_value_heads")?;
        let head_dim = usize_from_u32(config.head_dim(), "head_dim")?;
        let n_ff = usize_from_u32(config.n_ff, "intermediate_size")?;
        let n_ctx = usize_from_u32(config.n_ctx, "max_position_embeddings")?;
        let q_width = checked_mul(n_head, head_dim, "query projection width")?;
        let kv_width = checked_mul(n_head_kv, head_dim, "key/value projection width")?;

        if weights.layers.len() != n_layers {
            return Err(tensor_mismatch(
                "model.layers",
                n_layers,
                weights.layers.len(),
            ));
        }
        if weights.n_embd != n_embd {
            return Err(tensor_mismatch(
                "model hidden width",
                n_embd,
                weights.n_embd,
            ));
        }
        if weights.vocab == 0 {
            return Err(unsupported("vocabulary must be non-zero"));
        }
        if weights.token_embd.rows() != weights.vocab {
            return Err(tensor_mismatch(
                "model.embed_tokens.weight rows",
                weights.vocab,
                weights.token_embd.rows(),
            ));
        }
        if weights.token_embd.cols() != n_embd {
            return Err(tensor_mismatch(
                "model.embed_tokens.weight columns",
                n_embd,
                weights.token_embd.cols(),
            ));
        }
        let embedding_elements = checked_mul(weights.vocab, n_embd, "embedding shape")?;
        let token_embd = weights.token_embd.as_dense().ok_or_else(|| {
            unsupported("packed token embedding does not contain the latent fp32 training master")
        })?;
        if token_embd.len() != embedding_elements {
            return Err(tensor_mismatch(
                "model.embed_tokens.weight",
                embedding_elements,
                token_embd.len(),
            ));
        }
        validate_vector("model.norm.weight", &weights.output_norm, n_embd)?;
        if spec.tied_embeddings != weights.lm_head.is_none() {
            return Err(unsupported(if spec.tied_embeddings {
                "tie_word_embeddings=true requires no separate lm_head.weight"
            } else {
                "tie_word_embeddings=false requires a separate lm_head.weight"
            }));
        }

        let parameter_capacity = checked_mul(n_layers, 7, "parameter count")?
            .checked_add(1)
            .and_then(|count| count.checked_add(usize::from(!spec.tied_embeddings)))
            .ok_or_else(|| unsupported("parameter count overflows usize"))?;
        let mut parameters = Vec::with_capacity(parameter_capacity);
        parameters.push(TrainingParameter {
            name: "model.embed_tokens.weight".to_owned(),
            master: token_embd.to_vec(),
            rows: weights.vocab,
            cols: n_embd,
        });
        let mut attn_norms = Vec::with_capacity(n_layers);
        let mut ffn_norms = Vec::with_capacity(n_layers);
        let mut attention_constants = Vec::with_capacity(n_layers);

        for (layer_index, layer) in weights.layers.iter().enumerate() {
            let prefix = format!("model.layers.{layer_index}");
            validate_vector(
                &format!("{prefix}.input_layernorm.weight"),
                &layer.attn_norm,
                n_embd,
            )?;
            validate_vector(
                &format!("{prefix}.post_attention_layernorm.weight"),
                &layer.ffn_norm,
                n_embd,
            )?;
            if !layer.attn_sub_norm.is_empty() {
                return Err(unsupported(&format!(
                    "{prefix} contains an attention sub-norm"
                )));
            }
            validate_feature_vector(
                &format!("{prefix}.self_attn.q_proj.bias"),
                &layer.q_bias,
                q_width,
                spec.qkv_bias,
                "QKV bias",
            )?;
            validate_feature_vector(
                &format!("{prefix}.self_attn.k_proj.bias"),
                &layer.k_bias,
                kv_width,
                spec.qkv_bias,
                "QKV bias",
            )?;
            validate_feature_vector(
                &format!("{prefix}.self_attn.v_proj.bias"),
                &layer.v_bias,
                kv_width,
                spec.qkv_bias,
                "QKV bias",
            )?;
            validate_feature_vector(
                &format!("{prefix}.self_attn.q_norm.weight"),
                &layer.q_norm,
                head_dim,
                spec.qk_norm,
                "QK norm",
            )?;
            validate_feature_vector(
                &format!("{prefix}.self_attn.k_norm.weight"),
                &layer.k_norm,
                head_dim,
                spec.qk_norm,
                "QK norm",
            )?;

            let (gate, up, down) = match &layer.mlp {
                Mlp::SwiGlu(mlp) => (&mlp.gate, &mlp.up, &mlp.down),
                Mlp::Relu2(_) => {
                    return Err(unsupported(&format!(
                        "{prefix} uses a ReLU2 MLP instead of SwiGLU"
                    )));
                }
            };
            for (name, projection, rows, cols) in [
                (
                    format!("{prefix}.self_attn.q_proj.weight"),
                    &layer.q_proj,
                    q_width,
                    n_embd,
                ),
                (
                    format!("{prefix}.self_attn.k_proj.weight"),
                    &layer.k_proj,
                    kv_width,
                    n_embd,
                ),
                (
                    format!("{prefix}.self_attn.v_proj.weight"),
                    &layer.v_proj,
                    kv_width,
                    n_embd,
                ),
                (
                    format!("{prefix}.self_attn.o_proj.weight"),
                    &layer.o_proj,
                    n_embd,
                    q_width,
                ),
                (format!("{prefix}.mlp.gate_proj.weight"), gate, n_ff, n_embd),
                (format!("{prefix}.mlp.up_proj.weight"), up, n_ff, n_embd),
                (format!("{prefix}.mlp.down_proj.weight"), down, n_embd, n_ff),
            ] {
                parameters.push(extract_dense(name, projection, rows, cols)?);
            }
            attn_norms.push(layer.attn_norm.clone());
            ffn_norms.push(layer.ffn_norm.clone());
            attention_constants.push(TrainingAttentionConstants {
                q_bias: layer.q_bias.clone(),
                k_bias: layer.k_bias.clone(),
                v_bias: layer.v_bias.clone(),
                q_norm: layer.q_norm.clone(),
                k_norm: layer.k_norm.clone(),
            });
        }

        if let Some(lm_head) = &weights.lm_head {
            parameters.push(extract_dense(
                "lm_head.weight".to_owned(),
                lm_head,
                weights.vocab,
                n_embd,
            )?);
        }

        Ok(Self {
            arch: TiedSwiGluTrainingArchitecture {
                attn_norms,
                ffn_norms,
                attention_constants,
                output_norm: weights.output_norm.clone(),
                n_embd,
                n_head,
                n_head_kv,
                head_dim,
                n_ff,
                vocab: weights.vocab,
                rms_eps: config.rms_eps,
                rope_theta: config.rope_theta,
                n_layers,
                n_ctx,
            },
            parameters,
            lm_head_tied: spec.tied_embeddings,
        })
    }
}

/// Architecture descriptor for a SwiGLU training campaign.
///
/// This alias keeps callers using the original tied-head name source compatible.
pub type SwiGluTrainingArchitecture = TiedSwiGluTrainingArchitecture;

/// Validated SwiGLU training model with a tied or untied output projection.
///
/// This alias keeps callers using the original tied-head name source compatible.
pub type SwiGluTrainingModel = TiedSwiGluTrainingModel;

/// Outputs from one packed device forward graph.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedTrainingForward {
    /// DeviceTape value id for `[sequence, vocabulary]` logits.
    pub logits: usize,
    /// Gradient-only leaf ids in [`TiedSwiGluTrainingModel::parameters`] order.
    pub master_leaves: Vec<usize>,
}

/// Outputs from one dense resident device forward graph.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentTrainingForward {
    /// DeviceTape value id for `[sequence, vocabulary]` logits.
    pub logits: usize,
    /// Borrowed master leaf ids in [`TiedSwiGluTrainingModel::parameters`] order.
    pub master_leaves: Vec<usize>,
}

/// Outputs from one dense HESTIA device forward graph.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HestiaTrainingForward {
    /// DeviceTape value id for `[sequence, vocabulary]` logits.
    pub logits: usize,
    /// Borrowed latent-master leaf ids in canonical parameter order.
    pub master_leaves: Vec<usize>,
}

#[cfg(feature = "cuda")]
fn dense_device_forward_with<'backend, 'leaf, F>(
    tape: &mut tritium_cuda::train::DeviceTape<'backend, 'leaf>,
    model: &TiedSwiGluTrainingModel,
    tokens: &[i32],
    mut weight: F,
) -> Result<usize, TrainingAdapterError>
where
    F: FnMut(
        &mut tritium_cuda::train::DeviceTape<'backend, 'leaf>,
        usize,
    ) -> Result<usize, TrainingAdapterError>,
{
    let arch = &model.arch;
    let sequence = tokens.len();
    let embedding = weight(tape, 0)?;
    let mut hidden = tape.embed(embedding, tokens, sequence, arch.n_embd, arch.vocab)?;
    for layer_index in 0..arch.n_layers {
        let base = 1 + 7 * layer_index;
        let constants = &arch.attention_constants[layer_index];
        let attn_norm = tape.leaf(&arch.attn_norms[layer_index])?;
        let normalized = tape.rmsnorm(hidden, attn_norm, sequence, arch.n_embd, arch.rms_eps)?;
        let q = weight(tape, base)?;
        let k = weight(tape, base + 1)?;
        let v = weight(tape, base + 2)?;
        let o = weight(tape, base + 3)?;
        let attention = tape.attention_with_fixed(
            normalized,
            q,
            k,
            v,
            o,
            sequence,
            arch.n_embd,
            arch.n_head,
            arch.n_head_kv,
            arch.head_dim,
            arch.rope_theta,
            tritium_cuda::train::FixedAttentionParameters {
                q_bias: &constants.q_bias,
                k_bias: &constants.k_bias,
                v_bias: &constants.v_bias,
                q_norm: &constants.q_norm,
                k_norm: &constants.k_norm,
                rms_eps: arch.rms_eps,
            },
        )?;
        hidden = tape.add(hidden, attention)?;

        let ffn_norm = tape.leaf(&arch.ffn_norms[layer_index])?;
        let normalized = tape.rmsnorm(hidden, ffn_norm, sequence, arch.n_embd, arch.rms_eps)?;
        let gate_weight = weight(tape, base + 4)?;
        let up_weight = weight(tape, base + 5)?;
        let down_weight = weight(tape, base + 6)?;
        let gate = tape.matmul(normalized, gate_weight, sequence, arch.n_ff, arch.n_embd)?;
        let up = tape.matmul(normalized, up_weight, sequence, arch.n_ff, arch.n_embd)?;
        let activated_gate = tape.silu(gate)?;
        let gated = tape.mul(activated_gate, up)?;
        let down = tape.matmul(gated, down_weight, sequence, arch.n_embd, arch.n_ff)?;
        hidden = tape.add(hidden, down)?;
        tape.checkpoint_keep(&[hidden])?;
    }

    let output_norm = tape.leaf(&arch.output_norm)?;
    let normalized = tape.rmsnorm(hidden, output_norm, sequence, arch.n_embd, arch.rms_eps)?;
    let head = model.lm_head_parameter_index();
    let head_weight = weight(tape, head)?;
    tape.matmul(normalized, head_weight, sequence, arch.vocab, arch.n_embd)
        .map_err(TrainingAdapterError::from)
}

/// Build a differentiable HESTIA SwiGLU forward from resident latent masters.
///
/// Each master is relaxed with its packed SALT handle's current first-plane
/// AbsMean scale, then consumed by dense training ops. Projections are created
/// inside their checkpoint segment; a tied output head reprojects the same
/// master so gradients still accumulate into one leaf without retaining a
/// vocabulary-sized dense activation. All inputs are validated before the tape
/// changes.
///
/// # Errors
/// Returns [`TrainingAdapterError::InvalidInput`] for token, temperature, or
/// packed-model disagreement, [`TrainingAdapterError::TensorShape`] for resident
/// parameter disagreement, and [`TrainingAdapterError::Backend`] for device errors.
#[cfg(feature = "cuda")]
pub fn hestia_device_forward<'backend, 'leaf>(
    tape: &mut tritium_cuda::train::DeviceTape<'backend, 'leaf>,
    model: &TiedSwiGluTrainingModel,
    masters: &[&'leaf tritium_cuda::train::DeviceTensor],
    packed: &'leaf [tritium_cuda::train::DevicePackedSaltWeight],
    temperatures: &[f32],
    tokens: &[i32],
) -> Result<HestiaTrainingForward, TrainingAdapterError> {
    let arch = &model.arch;
    validate_training_tokens(arch, tokens)?;
    validate_runtime_architecture(arch)?;
    let expected = model.parameters.len();
    for (name, got) in [
        ("resident HESTIA parameter count", masters.len()),
        ("packed HESTIA parameter count", packed.len()),
        ("HESTIA temperature count", temperatures.len()),
    ] {
        if got != expected {
            return Err(tensor_mismatch(name, expected, got));
        }
    }
    for (index, parameter) in model.parameters.iter().enumerate() {
        let master = masters[index];
        let weight = &packed[index];
        tape.validate_device_tensor(master)?;
        if master.len() != parameter.elements() {
            return Err(tensor_mismatch(
                &format!("resident HESTIA {}", parameter.name),
                parameter.elements(),
                master.len(),
            ));
        }
        if weight.rows() != parameter.rows || weight.cols() != parameter.cols {
            return Err(invalid_input(&format!(
                "packed HESTIA {} is [{}, {}], expected [{}, {}]",
                parameter.name,
                weight.rows(),
                weight.cols(),
                parameter.rows,
                parameter.cols
            )));
        }
        weight.validate_current_master(master)?;
        let tau = temperatures[index];
        if !tau.is_finite() || tau < tritium_train::ops::hestia::MIN_DIFFERENTIABLE_TAU {
            return Err(invalid_input(&format!(
                "HESTIA temperature for {} must be finite and at least {}",
                parameter.name,
                tritium_train::ops::hestia::MIN_DIFFERENTIABLE_TAU
            )));
        }
    }

    let master_leaves = masters
        .iter()
        .map(|master| tape.leaf_device(master).map_err(TrainingAdapterError::from))
        .collect::<Result<Vec<_>, _>>()?;
    let logits = dense_device_forward_with(tape, model, tokens, |tape, index| {
        let tau = tape.leaf(&[temperatures[index]])?;
        tape.hestia_relax_packed(master_leaves[index], masters[index], &packed[index], tau)
            .map_err(TrainingAdapterError::from)
    })?;
    Ok(HestiaTrainingForward {
        logits,
        master_leaves,
    })
}

/// Build the SwiGLU packed-SALT forward on a CUDA
/// [`DeviceTape`](tritium_cuda::train::DeviceTape).
///
/// Every packed weight is shape-checked against the canonical parameter map
/// before the tape is changed. A tied head shares master leaf zero; an untied
/// head uses the final canonical leaf. Transformer-block boundaries become
/// activation-checkpoint frontiers.
///
/// # Errors
/// Returns [`TrainingAdapterError::InvalidInput`] for invalid tokens or packed-model
/// shape disagreement and [`TrainingAdapterError::Backend`] for device allocation
/// or kernel errors.
#[cfg(feature = "cuda")]
pub fn packed_device_forward<'backend, 'leaf>(
    tape: &mut tritium_cuda::train::DeviceTape<'backend, 'leaf>,
    model: &TiedSwiGluTrainingModel,
    weights: &'leaf [tritium_cuda::train::DevicePackedSaltWeight],
    tokens: &[i32],
) -> Result<PackedTrainingForward, TrainingAdapterError> {
    let arch = &model.arch;
    validate_training_tokens(arch, tokens)?;
    validate_runtime_architecture(arch)?;
    if weights.len() != model.parameters.len() {
        return Err(tensor_mismatch(
            "packed parameter count",
            model.parameters.len(),
            weights.len(),
        ));
    }
    for (parameter, weight) in model.parameters.iter().zip(weights) {
        if weight.rows() != parameter.rows || weight.cols() != parameter.cols {
            return Err(invalid_input(&format!(
                "packed {} is [{}, {}], expected [{}, {}]",
                parameter.name,
                weight.rows(),
                weight.cols(),
                parameter.rows,
                parameter.cols
            )));
        }
    }

    let masters: Vec<usize> = model
        .parameters
        .iter()
        .map(|parameter| {
            tape.gradient_leaf(parameter.elements())
                .map_err(TrainingAdapterError::from)
        })
        .collect::<Result<_, _>>()?;
    let sequence = tokens.len();
    let mut hidden = tape.salt_embed(masters[0], &weights[0], tokens)?;

    for layer_index in 0..arch.n_layers {
        let base = 1 + 7 * layer_index;
        let constants = &arch.attention_constants[layer_index];
        let attn_norm = tape.leaf(&arch.attn_norms[layer_index])?;
        let normalized = tape.rmsnorm(hidden, attn_norm, sequence, arch.n_embd, arch.rms_eps)?;
        let attention = tape.salt_attention_with_fixed(
            normalized,
            masters[base],
            &weights[base],
            masters[base + 1],
            &weights[base + 1],
            masters[base + 2],
            &weights[base + 2],
            masters[base + 3],
            &weights[base + 3],
            sequence,
            arch.n_embd,
            arch.n_head,
            arch.n_head_kv,
            arch.head_dim,
            arch.rope_theta,
            tritium_cuda::train::FixedAttentionParameters {
                q_bias: &constants.q_bias,
                k_bias: &constants.k_bias,
                v_bias: &constants.v_bias,
                q_norm: &constants.q_norm,
                k_norm: &constants.k_norm,
                rms_eps: arch.rms_eps,
            },
        )?;
        hidden = tape.add(hidden, attention)?;

        let ffn_norm = tape.leaf(&arch.ffn_norms[layer_index])?;
        let normalized = tape.rmsnorm(hidden, ffn_norm, sequence, arch.n_embd, arch.rms_eps)?;
        let gate = tape.salt_matmul(normalized, masters[base + 4], &weights[base + 4], sequence)?;
        let up = tape.salt_matmul(normalized, masters[base + 5], &weights[base + 5], sequence)?;
        let activated_gate = tape.silu(gate)?;
        let gated = tape.mul(activated_gate, up)?;
        let down = tape.salt_matmul(gated, masters[base + 6], &weights[base + 6], sequence)?;
        hidden = tape.add(hidden, down)?;
        tape.checkpoint_keep(&[hidden])?;
    }

    let output_norm = tape.leaf(&arch.output_norm)?;
    let normalized = tape.rmsnorm(hidden, output_norm, sequence, arch.n_embd, arch.rms_eps)?;
    let head = model.lm_head_parameter_index();
    let logits = tape.salt_matmul(normalized, masters[head], &weights[head], sequence)?;
    Ok(PackedTrainingForward {
        logits,
        master_leaves: masters,
    })
}

/// Build the SwiGLU dense forward from already-resident CUDA tensors.
///
/// Every tensor is length-checked before any leaf is added. The tensors are then
/// borrowed through [`tritium_cuda::train::DeviceTape::leaf_device`], which also
/// rejects CUDA-context mismatches without allocating or copying trainable
/// weights. Parameter leaves retain [`TiedSwiGluTrainingModel::parameters`] order.
/// A tied head shares leaf zero; an untied head uses the final canonical leaf.
/// Transformer-block boundaries match [`packed_device_forward`] activation-
/// checkpoint frontiers.
///
/// # Errors
/// Returns [`TrainingAdapterError::InvalidInput`] for invalid tokens,
/// [`TrainingAdapterError::TensorShape`] for parameter count or length
/// disagreement, and [`TrainingAdapterError::Backend`] for CUDA-context,
/// allocation, or kernel errors.
#[cfg(feature = "cuda")]
pub fn resident_device_forward<'backend, 'leaf>(
    tape: &mut tritium_cuda::train::DeviceTape<'backend, 'leaf>,
    model: &TiedSwiGluTrainingModel,
    weights: &[&'leaf tritium_cuda::train::DeviceTensor],
    tokens: &[i32],
) -> Result<ResidentTrainingForward, TrainingAdapterError> {
    let arch = &model.arch;
    validate_training_tokens(arch, tokens)?;
    validate_runtime_architecture(arch)?;
    if weights.len() != model.parameters.len() {
        return Err(tensor_mismatch(
            "resident parameter count",
            model.parameters.len(),
            weights.len(),
        ));
    }
    for (parameter, weight) in model.parameters.iter().zip(weights) {
        if weight.len() != parameter.elements() {
            return Err(tensor_mismatch(
                &format!("resident {}", parameter.name),
                parameter.elements(),
                weight.len(),
            ));
        }
    }

    // Borrow every trainable tensor before constructing the graph. Besides
    // preserving canonical order, leaf_device validates that each allocation
    // belongs to this tape's CUDA context.
    let masters = weights
        .iter()
        .map(|weight| tape.leaf_device(weight).map_err(TrainingAdapterError::from))
        .collect::<Result<Vec<_>, _>>()?;
    let logits = dense_device_forward_with(tape, model, tokens, |_tape, index| Ok(masters[index]))?;
    Ok(ResidentTrainingForward {
        logits,
        master_leaves: masters,
    })
}

#[cfg(feature = "cuda")]
fn validate_training_tokens(
    arch: &TiedSwiGluTrainingArchitecture,
    tokens: &[i32],
) -> Result<(), TrainingAdapterError> {
    if tokens.is_empty() {
        return Err(invalid_input("training sequence must be non-empty"));
    }
    if tokens.len() > arch.n_ctx {
        return Err(invalid_input(
            "training sequence exceeds max_position_embeddings",
        ));
    }
    for (position, &token) in tokens.iter().enumerate() {
        if token < 0 || usize::try_from(token).map_or(true, |token| token >= arch.vocab) {
            return Err(invalid_input(&format!(
                "token at position {position} is outside the vocabulary"
            )));
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn validate_runtime_architecture(
    arch: &TiedSwiGluTrainingArchitecture,
) -> Result<(), TrainingAdapterError> {
    if arch.attn_norms.len() != arch.n_layers {
        return Err(tensor_mismatch(
            "attention norm count",
            arch.n_layers,
            arch.attn_norms.len(),
        ));
    }
    if arch.ffn_norms.len() != arch.n_layers {
        return Err(tensor_mismatch(
            "FFN norm count",
            arch.n_layers,
            arch.ffn_norms.len(),
        ));
    }
    if arch.attention_constants.len() != arch.n_layers {
        return Err(tensor_mismatch(
            "attention constant set count",
            arch.n_layers,
            arch.attention_constants.len(),
        ));
    }
    validate_vector("model.norm.weight", &arch.output_norm, arch.n_embd)?;
    let q_width = checked_mul(arch.n_head, arch.head_dim, "query projection width")?;
    let kv_width = checked_mul(arch.n_head_kv, arch.head_dim, "key/value projection width")?;
    for layer_index in 0..arch.n_layers {
        let prefix = format!("model.layers.{layer_index}");
        validate_vector(
            &format!("{prefix}.input_layernorm.weight"),
            &arch.attn_norms[layer_index],
            arch.n_embd,
        )?;
        validate_vector(
            &format!("{prefix}.post_attention_layernorm.weight"),
            &arch.ffn_norms[layer_index],
            arch.n_embd,
        )?;
        let constants = &arch.attention_constants[layer_index];
        let bias_presence = [
            !constants.q_bias.is_empty(),
            !constants.k_bias.is_empty(),
            !constants.v_bias.is_empty(),
        ];
        if bias_presence.iter().any(|present| *present)
            && !bias_presence.iter().all(|present| *present)
        {
            return Err(invalid_input(&format!(
                "{prefix} QKV bias vectors must be all present or all absent"
            )));
        }
        if constants.q_norm.is_empty() != constants.k_norm.is_empty() {
            return Err(invalid_input(&format!(
                "{prefix} Q/K norm vectors must be both present or both absent"
            )));
        }
        for (name, values, width) in [
            ("q_proj.bias", constants.q_bias.as_slice(), q_width),
            ("k_proj.bias", constants.k_bias.as_slice(), kv_width),
            ("v_proj.bias", constants.v_bias.as_slice(), kv_width),
            ("q_norm.weight", constants.q_norm.as_slice(), arch.head_dim),
            ("k_norm.weight", constants.k_norm.as_slice(), arch.head_dim),
        ] {
            validate_optional_vector(&format!("{prefix}.self_attn.{name}"), values, width)?;
        }
    }
    Ok(())
}

fn expand_incoming_rows_fallible(
    plan: &tritium_train::Net2WiderPlan,
    weights: &[f32],
    input_width: usize,
) -> Result<Vec<f32>, TrainingAdapterError> {
    let old_width = plan.replication_counts().len();
    let expected = old_width
        .checked_mul(input_width)
        .ok_or_else(|| invalid_input("incoming projection size overflows usize"))?;
    if weights.len() != expected {
        return Err(tensor_mismatch(
            "incoming growth projection",
            expected,
            weights.len(),
        ));
    }
    let output_len = plan
        .source_indices()
        .len()
        .checked_mul(input_width)
        .ok_or_else(|| invalid_input("expanded incoming projection overflows usize"))?;
    let mut expanded = Vec::new();
    expanded
        .try_reserve_exact(output_len)
        .map_err(|_| allocation_failed::<f32>("expanded incoming projection", output_len))?;
    for &source in plan.source_indices() {
        let start = source
            .checked_mul(input_width)
            .ok_or_else(|| invalid_input("incoming source row overflows usize"))?;
        let end = start
            .checked_add(input_width)
            .ok_or_else(|| invalid_input("incoming source row end overflows usize"))?;
        let row = weights
            .get(start..end)
            .ok_or_else(|| invalid_input("incoming source row is outside the tensor"))?;
        expanded.extend_from_slice(row);
    }
    Ok(expanded)
}

fn expand_outgoing_columns_fallible(
    plan: &tritium_train::Net2WiderPlan,
    weights: &[f32],
    output_width: usize,
) -> Result<Vec<f32>, TrainingAdapterError> {
    let old_width = plan.replication_counts().len();
    let new_width = plan.source_indices().len();
    let expected = output_width
        .checked_mul(old_width)
        .ok_or_else(|| invalid_input("outgoing projection size overflows usize"))?;
    if weights.len() != expected {
        return Err(tensor_mismatch(
            "outgoing growth projection",
            expected,
            weights.len(),
        ));
    }
    let output_len = output_width
        .checked_mul(new_width)
        .ok_or_else(|| invalid_input("expanded outgoing projection overflows usize"))?;
    let mut expanded = Vec::new();
    expanded
        .try_reserve_exact(output_len)
        .map_err(|_| allocation_failed::<f32>("expanded outgoing projection", output_len))?;
    let denominator = plan
        .split_denominator_log2()
        .map(|exponent| {
            1_u32
                .checked_shl(exponent)
                .ok_or_else(|| invalid_input("outgoing split denominator overflows u32"))
        })
        .transpose()?;
    for row in weights.chunks_exact(old_width) {
        for (new_index, &source) in plan.source_indices().iter().enumerate() {
            let source_weight = *row
                .get(source)
                .ok_or_else(|| invalid_input("outgoing source column is outside the tensor"))?;
            let coefficient = match (plan.split_numerators(), denominator) {
                (Some(numerators), Some(denominator)) => {
                    let numerator = *numerators.get(new_index).ok_or_else(|| {
                        invalid_input("outgoing split numerator is outside the plan")
                    })?;
                    numerator as f32 / denominator as f32
                }
                (None, None) => {
                    let copies = *plan.replication_counts().get(source).ok_or_else(|| {
                        invalid_input("outgoing replication count is outside the plan")
                    })?;
                    if copies == 0 {
                        return Err(invalid_input("outgoing replication count is zero"));
                    }
                    1.0 / copies as f32
                }
                _ => return Err(invalid_input("outgoing split metadata is inconsistent")),
            };
            expanded.push(source_weight * coefficient);
        }
    }
    Ok(expanded)
}

fn extract_dense(
    name: String,
    projection: &Projection,
    rows: usize,
    cols: usize,
) -> Result<TrainingParameter, TrainingAdapterError> {
    let dense = match projection {
        Projection::Dense(dense) => dense,
        Projection::Salt(_) | Projection::Ternary(_) | Projection::Q2(_) => {
            return Err(unsupported(&format!(
                "{name} is already ternary; a latent fp32 master is required"
            )));
        }
        Projection::HostSaltV2(_) => {
            return Err(unsupported(&format!(
                "{name} is already host SALT V2; a latent fp32 master is required"
            )));
        }
        #[cfg(feature = "cuda")]
        Projection::SaltV2(_) => {
            return Err(unsupported(&format!(
                "{name} is already resident SALT V2; a latent fp32 master is required"
            )));
        }
    };
    if dense.n_out != rows || dense.k_in != cols {
        return Err(unsupported(&format!(
            "{name} is [{}, {}], expected [{rows}, {cols}]",
            dense.n_out, dense.k_in
        )));
    }
    let elements = checked_mul(rows, cols, &format!("{name} shape"))?;
    if dense.weights.len() != elements {
        return Err(tensor_mismatch(&name, elements, dense.weights.len()));
    }
    Ok(TrainingParameter {
        name,
        master: dense.weights.clone(),
        rows,
        cols,
    })
}

fn validate_vector(
    name: &str,
    vector: &[f32],
    expected: usize,
) -> Result<(), TrainingAdapterError> {
    if vector.len() != expected {
        return Err(tensor_mismatch(name, expected, vector.len()));
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(invalid_input(&format!(
            "{name} contains a non-finite value"
        )));
    }
    Ok(())
}

fn validate_optional_vector(
    name: &str,
    vector: &[f32],
    expected: usize,
) -> Result<(), TrainingAdapterError> {
    if vector.is_empty() {
        Ok(())
    } else {
        validate_vector(name, vector, expected)
    }
}

fn validate_feature_vector(
    name: &str,
    vector: &[f32],
    expected: usize,
    required: bool,
    feature: &str,
) -> Result<(), TrainingAdapterError> {
    if required {
        validate_vector(name, vector, expected)
    } else if vector.is_empty() {
        Ok(())
    } else {
        Err(unsupported(&format!(
            "{name} contains {feature} while the architecture disables it"
        )))
    }
}

fn validate_parameter_geometry(
    parameter: &TrainingParameter,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<(), TrainingAdapterError> {
    if parameter.rows == expected_rows && parameter.cols == expected_cols {
        Ok(())
    } else {
        Err(invalid_input(&format!(
            "{} is [{}, {}], expected [{expected_rows}, {expected_cols}]",
            parameter.name, parameter.rows, parameter.cols
        )))
    }
}

fn usize_from_u32(value: u32, name: &str) -> Result<usize, TrainingAdapterError> {
    usize::try_from(value).map_err(|_| unsupported(&format!("{name} exceeds usize")))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, TrainingAdapterError> {
    left.checked_mul(right)
        .ok_or_else(|| unsupported(&format!("{name} overflows usize")))
}

fn unsupported(reason: &str) -> TrainingAdapterError {
    TrainingAdapterError::UnsupportedArchitecture(reason.to_owned())
}

fn tensor_mismatch(name: &str, expected: usize, got: usize) -> TrainingAdapterError {
    TrainingAdapterError::TensorShape {
        name: name.to_owned(),
        expected,
        got,
    }
}

fn invalid_input(reason: &str) -> TrainingAdapterError {
    TrainingAdapterError::InvalidInput(reason.to_owned())
}

fn allocation_failed<T>(allocation: &'static str, elements: usize) -> TrainingAdapterError {
    TrainingAdapterError::AllocationFailed {
        allocation,
        requested_bytes: elements.saturating_mul(core::mem::size_of::<T>()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widening_staging_failure_after_an_earlier_layer_is_atomic() {
        let mut model = atomic_growth_fixture();
        let before = model.clone();

        let error = match model.begin_intermediate_widening_with_hook(5, 0x27, |projection| {
            if projection == 4 {
                Err(allocation_failed::<f32>(
                    "injected second-layer staging allocation",
                    1,
                ))
            } else {
                Ok(())
            }
        }) {
            Ok(_) => panic!("injected staging failure must reject widening"),
            Err(error) => error,
        };

        assert_eq!(
            error,
            TrainingAdapterError::AllocationFailed {
                allocation: "injected second-layer staging allocation",
                requested_bytes: 4,
            }
        );
        assert_eq!(model, before);
    }

    fn atomic_growth_fixture() -> TiedSwiGluTrainingModel {
        let n_layers = 2;
        let n_embd = 2;
        let n_ff = 3;
        let mut parameters = Vec::new();
        parameters.push(parameter("embed", 4, n_embd, 0.25));
        for layer in 0..n_layers {
            let marker = layer as f32 + 1.0;
            parameters.extend([
                parameter("q", n_embd, n_embd, marker),
                parameter("k", n_embd, n_embd, marker + 0.1),
                parameter("v", n_embd, n_embd, marker + 0.2),
                parameter("o", n_embd, n_embd, marker + 0.3),
                parameter("gate", n_ff, n_embd, marker + 0.4),
                parameter("up", n_ff, n_embd, marker + 0.5),
                parameter("down", n_embd, n_ff, marker + 0.6),
            ]);
        }
        TiedSwiGluTrainingModel {
            arch: TiedSwiGluTrainingArchitecture {
                attn_norms: vec![vec![1.0; n_embd]; n_layers],
                ffn_norms: vec![vec![1.0; n_embd]; n_layers],
                attention_constants: vec![TrainingAttentionConstants::default(); n_layers],
                output_norm: vec![1.0; n_embd],
                n_embd,
                n_head: 1,
                n_head_kv: 1,
                head_dim: n_embd,
                n_ff,
                vocab: 4,
                rms_eps: 1e-5,
                rope_theta: 10_000.0,
                n_layers,
                n_ctx: 8,
            },
            parameters,
            lm_head_tied: true,
        }
    }

    fn parameter(name: &str, rows: usize, cols: usize, value: f32) -> TrainingParameter {
        TrainingParameter {
            name: name.to_owned(),
            master: vec![value; rows * cols],
            rows,
            cols,
        }
    }
}
