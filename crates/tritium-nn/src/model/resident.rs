//! Compilation adapter from the same admitted weights to an owned backend graph.
use super::{ArchitectureSpec, ModelWeights, unified::Executor};
use crate::{Mlp, NnError, Projection};
use std::sync::Arc;
use tritium_spec::{
    DecoderActivation, DecoderExecution, DecoderGeometry, DecoderGraph, DecoderLayer,
    DecoderLimits, DeviceBuffer, ResidentDecoder, TernaryBackend,
};

pub(crate) fn matrix(projection: &Projection) -> Result<&Arc<dyn DeviceBuffer>, NnError> {
    match projection {
        Projection::Tensor(linear) => Ok(linear.device_arc()),
        _ => Err(NnError::Backend(
            "shared decoder graph requires admitted tensor handles".into(),
        )),
    }
}
pub(super) fn standard(
    architecture: &ArchitectureSpec,
    weights: &ModelWeights,
    backend: Arc<dyn TernaryBackend>,
    limits: DecoderLimits,
) -> Result<Option<Box<dyn Executor>>, NnError> {
    if !backend.resident_decoder_requested() {
        return Ok(None);
    }
    let ArchitectureSpec::Standard { config, .. } = architecture else {
        return Err(NnError::Backend(
            "standard graph compilation requires standard architecture".into(),
        ));
    };
    if weights.layers.len() != architecture.layer_count()
        || weights.token_embd.rows() != architecture.vocab_size()
        || weights.token_embd.cols() != config.n_embd as usize
        || limits.max_context == 0
        || limits.max_context > architecture.max_context()
    {
        return Err(NnError::Backend(
            "resident graph disagrees with admitted architecture".into(),
        ));
    }
    let embedding = weights
        .token_embd
        .shared_linear()
        .ok_or_else(|| NnError::Backend("common decoder requires shared token table".into()))?
        .device_arc();
    let head = weights
        .lm_head
        .as_ref()
        .map(matrix)
        .transpose()?
        .unwrap_or(embedding);
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(weights.layers.len())
        .map_err(|e| NnError::Backend(format!("resident graph view allocation: {e}")))?;
    for layer in &weights.layers {
        let (gate, up, down, activation, sub_norm, eps) = match &layer.mlp {
            Mlp::SwiGlu(mlp) => (
                &mlp.gate,
                &mlp.up,
                &mlp.down,
                DecoderActivation::SwiGlu,
                &[][..],
                config.rms_eps,
            ),
            Mlp::Relu2(mlp) => (
                &mlp.gate,
                &mlp.up,
                &mlp.down,
                DecoderActivation::Relu2,
                mlp.ffn_sub_norm.as_slice(),
                mlp.rms_eps,
            ),
        };
        layers.push(DecoderLayer {
            attention_norm: &layer.attn_norm,
            query: matrix(&layer.q_proj)?,
            key: matrix(&layer.k_proj)?,
            value: matrix(&layer.v_proj)?,
            attention_output: matrix(&layer.o_proj)?,
            attention_sub_norm: &layer.attn_sub_norm,
            query_bias: &layer.q_bias,
            key_bias: &layer.k_bias,
            value_bias: &layer.v_bias,
            query_norm: &layer.q_norm,
            key_norm: &layer.k_norm,
            ffn_norm: &layer.ffn_norm,
            gate: matrix(gate)?,
            up: matrix(up)?,
            down: matrix(down)?,
            activation,
            ffn_sub_norm: sub_norm,
            ffn_sub_eps: eps,
        });
    }
    let graph = DecoderGraph {
        geometry: DecoderGeometry {
            hidden: config.n_embd as usize,
            intermediate: config.n_ff as usize,
            query_heads: config.n_head as usize,
            kv_heads: config.n_head_kv as usize,
            head_dim: config.head_dim() as usize,
            vocabulary: weights.vocab,
            rms_eps: config.rms_eps,
            rope_theta: config.rope_theta,
        },
        embedding,
        head,
        final_norm: &weights.output_norm,
        layers: tritium_spec::DecoderLayers::Standard(&layers),
        semantics: tritium_spec::DecoderSemantics::standard(config.head_dim() as usize),
        retained_matrices: &[],
    };
    let selected = backend
        .resident_decoder(graph, limits)
        .map_err(|e| NnError::Backend(e.to_string()))?
        .ok_or_else(|| {
            NnError::Backend("requested resident decoder declined admitted graph".into())
        })?;
    if selected.limits() != limits || !selected.is_empty() {
        return Err(NnError::Backend(
            "resident decoder changed admitted limits or initial cursor".into(),
        ));
    }
    Ok(Some(Box::new(ResidentAdapter {
        decoder: selected,
        mtp: None,
        _identity: None,
    })))
}
struct ResidentAdapter {
    decoder: Box<dyn ResidentDecoder>,
    mtp: Option<super::UnverifiedQwen35Mtp>,
    _identity: Option<Arc<super::qwen35::RunnerIdentity>>,
}
impl Executor for ResidentAdapter {
    fn forward(
        &mut self,
        _architecture: &ArchitectureSpec,
        tokens: &[u32],
        positions: &[usize],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Option<Vec<f32>>, NnError> {
        self.decoder
            .forward(tokens, positions, cancelled)
            .map_err(|e| NnError::Backend(e.to_string()))
    }
    fn reset(&mut self) {
        self.decoder.reset();
    }
    fn len(&self) -> usize {
        self.decoder.len()
    }
    fn execution(&self) -> DecoderExecution {
        DecoderExecution::ResidentDeviceF32
    }
    fn resident_payload_bytes(&self) -> Option<usize> {
        Some(self.decoder.payload_bytes())
    }
    fn mtp_status(&self) -> Option<crate::Qwen35MtpStatus> {
        self.mtp.as_ref().map(super::UnverifiedQwen35Mtp::status)
    }
}

pub(super) fn hybrid(
    architecture: &ArchitectureSpec,
    weights: &super::host::Weights,
    backend: Arc<dyn TernaryBackend>,
    limits: DecoderLimits,
    mtp: super::UnverifiedQwen35Mtp,
    identity: Arc<super::qwen35::RunnerIdentity>,
) -> Result<Box<dyn Executor>, NnError> {
    let ArchitectureSpec::Hybrid(config) = architecture else {
        return Err(NnError::Backend(
            "hybrid graph architecture mismatch".into(),
        ));
    };
    let c = &config.text;
    if weights.layers.len() != architecture.layer_count()
        || limits.max_context == 0
        || limits.max_context > architecture.max_context()
    {
        return Err(NnError::Backend(
            "hybrid graph disagrees with admitted architecture".into(),
        ));
    }
    let embedding = weights
        .embedding
        .shared_linear()
        .ok_or_else(|| NnError::Backend("resident hybrid requires shared embedding".into()))?
        .device_arc();
    let head = matrix(
        weights
            .head
            .as_ref()
            .ok_or_else(|| NnError::Backend("hybrid head missing".into()))?,
    )?;
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(weights.layers.len())
        .map_err(|e| NnError::Backend(format!("hybrid graph allocation: {e}")))?;
    for layer in &weights.layers {
        let super::host::Layer::Hybrid(layer) = layer else {
            return Err(NnError::Backend("hybrid graph has standard layer".into()));
        };
        layers.push(layer.resident_binding()?);
    }
    let retained = mtp.resident_matrices()?;
    let graph = DecoderGraph {
        geometry: DecoderGeometry {
            hidden: c.hidden_size as usize,
            intermediate: c.intermediate_size as usize,
            query_heads: c.full_attention.num_heads as usize,
            kv_heads: c.full_attention.num_key_value_heads as usize,
            head_dim: c.full_attention.head_dim as usize,
            vocabulary: c.vocab_size as usize,
            rms_eps: c.rms_norm_eps as f32,
            rope_theta: c.rope.theta as f32,
        },
        embedding,
        head,
        final_norm: &weights.final_norm,
        layers: tritium_spec::DecoderLayers::Hybrid(&layers),
        semantics: tritium_spec::DecoderSemantics {
            zero_centered_norm: true,
            rotary_dim: c.rope.rotary_dim as usize,
            qwen_rope_f32: true,
        },
        retained_matrices: &retained,
    };
    let decoder = backend
        .resident_decoder(graph, limits)
        .map_err(|e| NnError::Backend(e.to_string()))?
        .ok_or_else(|| NnError::Backend("requested hybrid resident decoder declined".into()))?;
    if decoder.limits() != limits || !decoder.is_empty() {
        return Err(NnError::Backend(
            "hybrid executor changed limits or initial cursor".into(),
        ));
    }
    Ok(Box::new(ResidentAdapter {
        decoder,
        mtp: Some(mtp),
        _identity: Some(identity),
    }))
}
