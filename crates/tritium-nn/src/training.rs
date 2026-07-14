//! Production model adapter for bias-free SwiGLU training campaigns.
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

use crate::{ArchSpec, Mlp, MlpKind, ModelConfig, ModelWeights, Projection};

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

    /// Deterministically widen every SwiGLU intermediate axis in place.
    ///
    /// One [`tritium_train::Net2WiderPlan`] is shared by all transformer layers:
    /// gate and up rows are copied from the selected source units, while down
    /// columns are copied and divided by each source unit's replication count.
    /// This preserves the dense fp32 SwiGLU function before SALT quantization.
    /// Canonical parameter names and order are unchanged.
    ///
    /// The model is fully preflighted before mutation. During mutation, only one
    /// source tensor and its expanded replacement are live at a time, so temporary
    /// storage is bounded by the largest individual MLP tensor rather than model
    /// depth.
    ///
    /// # Errors
    /// Returns [`TrainingAdapterError::InvalidInput`] when `new_width` narrows the
    /// model, a size overflows, parameter geometry is inconsistent, or latent
    /// masters have already moved into an optimizer.
    pub fn widen_intermediate(
        &mut self,
        new_width: usize,
        seed: u64,
    ) -> Result<tritium_train::Net2WiderPlan, TrainingAdapterError> {
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
            return Ok(plan);
        }

        for layer_index in 0..self.arch.n_layers {
            let base = 1 + 7 * layer_index;
            let gate = &mut self.parameters[base + 4];
            gate.master = plan
                .expand_incoming_rows(&gate.master, self.arch.n_embd)
                .map_err(|error| invalid_input(&error.to_string()))?;
            gate.rows = new_width;

            let up = &mut self.parameters[base + 5];
            up.master = plan
                .expand_incoming_rows(&up.master, self.arch.n_embd)
                .map_err(|error| invalid_input(&error.to_string()))?;
            up.rows = new_width;

            let down = &mut self.parameters[base + 6];
            down.master = plan
                .expand_outgoing_columns(&down.master, self.arch.n_embd)
                .map_err(|error| invalid_input(&error.to_string()))?;
            down.cols = new_width;
        }
        self.arch.n_ff = new_width;
        Ok(plan)
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
        if spec.qkv_bias {
            return Err(unsupported("QKV bias is not supported"));
        }
        if spec.qk_norm {
            return Err(unsupported("QK norm is not supported"));
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
        let embedding_elements = checked_mul(weights.vocab, n_embd, "embedding shape")?;
        if weights.token_embd.len() != embedding_elements {
            return Err(tensor_mismatch(
                "model.embed_tokens.weight",
                embedding_elements,
                weights.token_embd.len(),
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
            master: weights.token_embd.clone(),
            rows: weights.vocab,
            cols: n_embd,
        });
        let mut attn_norms = Vec::with_capacity(n_layers);
        let mut ffn_norms = Vec::with_capacity(n_layers);

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
            if !layer.q_bias.is_empty() || !layer.k_bias.is_empty() || !layer.v_bias.is_empty() {
                return Err(unsupported(&format!("{prefix} contains QKV bias")));
            }
            if !layer.q_norm.is_empty() || !layer.k_norm.is_empty() {
                return Err(unsupported(&format!("{prefix} contains QK norm")));
            }

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

/// Architecture descriptor for a bias-free SwiGLU training campaign.
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
        let attn_norm = tape.leaf(&arch.attn_norms[layer_index])?;
        let normalized = tape.rmsnorm(hidden, attn_norm, sequence, arch.n_embd, arch.rms_eps)?;
        let attention = tape.salt_attention(
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
    let sequence = tokens.len();
    let mut hidden = tape.embed(masters[0], tokens, sequence, arch.n_embd, arch.vocab)?;

    for layer_index in 0..arch.n_layers {
        let base = 1 + 7 * layer_index;
        let attn_norm = tape.leaf(&arch.attn_norms[layer_index])?;
        let normalized = tape.rmsnorm(hidden, attn_norm, sequence, arch.n_embd, arch.rms_eps)?;
        let attention = tape.attention(
            normalized,
            masters[base],
            masters[base + 1],
            masters[base + 2],
            masters[base + 3],
            sequence,
            arch.n_embd,
            arch.n_head,
            arch.n_head_kv,
            arch.head_dim,
            arch.rope_theta,
        )?;
        hidden = tape.add(hidden, attention)?;

        let ffn_norm = tape.leaf(&arch.ffn_norms[layer_index])?;
        let normalized = tape.rmsnorm(hidden, ffn_norm, sequence, arch.n_embd, arch.rms_eps)?;
        let gate = tape.matmul(
            normalized,
            masters[base + 4],
            sequence,
            arch.n_ff,
            arch.n_embd,
        )?;
        let up = tape.matmul(
            normalized,
            masters[base + 5],
            sequence,
            arch.n_ff,
            arch.n_embd,
        )?;
        let activated_gate = tape.silu(gate)?;
        let gated = tape.mul(activated_gate, up)?;
        let down = tape.matmul(gated, masters[base + 6], sequence, arch.n_embd, arch.n_ff)?;
        hidden = tape.add(hidden, down)?;
        tape.checkpoint_keep(&[hidden])?;
    }

    let output_norm = tape.leaf(&arch.output_norm)?;
    let normalized = tape.rmsnorm(hidden, output_norm, sequence, arch.n_embd, arch.rms_eps)?;
    let head = model.lm_head_parameter_index();
    let logits = tape.matmul(normalized, masters[head], sequence, arch.vocab, arch.n_embd)?;
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

fn extract_dense(
    name: String,
    projection: &Projection,
    rows: usize,
    cols: usize,
) -> Result<TrainingParameter, TrainingAdapterError> {
    let dense = match projection {
        Projection::Dense(dense) => dense,
        Projection::Ternary(_) => {
            return Err(unsupported(&format!(
                "{name} is already ternary; a latent fp32 master is required"
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
    if vector.len() == expected {
        Ok(())
    } else {
        Err(tensor_mismatch(name, expected, vector.len()))
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
