//! Owned structural inputs compiled from the same admitted source graph.
use super::{
    ModelLoadPlan,
    qwen35_hf::{LANGUAGE_EMBEDDING, TensorRole, language_schema, mtp_schema},
    retained_plan::Graph,
};
use crate::{NnError, Qwen35LayerType};
use std::io::{Read, Seek};
use tritium_format::trit_model::TritModelReader;
use tritium_spec::*;

pub(super) struct Allocation {
    geometry: DecoderGeometry,
    kind: DecoderAllocationKind,
    semantics: DecoderSemantics,
    final_norm: usize,
    layers: Vec<DecoderAllocationLayer>,
    retained: Vec<(usize, usize)>,
    matrices: usize,
    rotation: usize,
    pub(super) limits: DecoderLimits,
}
impl Allocation {
    pub(super) fn metadata(&self) -> DecoderAllocationMetadata<'_> {
        DecoderAllocationMetadata {
            geometry: self.geometry,
            kind: self.kind,
            semantics: self.semantics,
            final_norm: self.final_norm,
            layers: &self.layers,
            matrix_payload_bytes: self.matrices,
            retained_matrix_shapes: &self.retained,
            rotation_workspace_width: self.rotation,
        }
    }
}
fn invalid(e: impl std::fmt::Display) -> NnError {
    NnError::ResourceExhausted(format!("resident source plan: {e}"))
}
fn rotation<R: Read + Seek>(
    model: &TritModelReader<R>,
    level: &str,
    name: &str,
) -> Result<usize, NnError> {
    let meta = model.tensor_meta(level, name).map_err(invalid)?;
    match super::tensor_stream::metadata(meta).map_err(invalid)? {
        TensorMetadata::Additive(a) if a.layout().basis != Basis::Identity => {
            usize::try_from(a.layout().cols).map_err(invalid)
        }
        _ => Ok(0),
    }
}

pub(super) fn compile<R: Read + Seek>(
    model: &TritModelReader<R>,
    level: &str,
    graph: &Graph,
    source: &ModelLoadPlan,
    matrix_bytes: u64,
) -> Result<Allocation, NnError> {
    let bounds = source
        .cache_admission()
        .ok_or_else(|| invalid("explicit stream limits required"))?
        .limits();
    let limits = DecoderLimits {
        max_context: bounds.max_context,
        max_segment_tokens: bounds.max_segment_tokens,
    };
    let matrices = usize::try_from(matrix_bytes).map_err(invalid)?;
    match graph {
        Graph::Standard(p) => {
            let (c, spec, plan) = p.as_ref();
            let geometry = DecoderGeometry {
                hidden: c.n_embd as usize,
                intermediate: c.n_ff as usize,
                query_heads: c.n_head as usize,
                kv_heads: c.n_head_kv as usize,
                head_dim: c.head_dim() as usize,
                vocabulary: plan.vocab(),
                rms_eps: c.rms_eps,
                rope_theta: c.rope_theta,
            };
            let mut width = 0;
            for request in plan.executed_matrices() {
                width = width.max(rotation(model, level, &request.name)?);
            }
            Ok(Allocation {
                geometry,
                kind: DecoderAllocationKind::Standard,
                semantics: DecoderSemantics::standard(geometry.head_dim),
                final_norm: geometry.hidden,
                layers: plan.resident_layers(spec, c.rms_eps)?,
                retained: Vec::new(),
                matrices,
                rotation: width,
                limits,
            })
        }
        Graph::Hybrid(config) => {
            let c = &config.text;
            let hidden = c.hidden_size as usize;
            let geometry = DecoderGeometry {
                hidden,
                intermediate: c.intermediate_size as usize,
                query_heads: c.full_attention.num_heads as usize,
                kv_heads: c.full_attention.num_key_value_heads as usize,
                head_dim: c.full_attention.head_dim as usize,
                vocabulary: c.vocab_size as usize,
                rms_eps: c.rms_norm_eps as f32,
                rope_theta: c.rope.theta as f32,
            };
            let mut width = 0;
            for (name, entry) in language_schema(c)? {
                if entry.role == TensorRole::Matrix && name != LANGUAGE_EMBEDDING {
                    width = width.max(rotation(model, level, &name)?);
                }
            }
            let mtp = mtp_schema(c)?;
            let mut retained = Vec::new();
            retained.try_reserve_exact(mtp.len()).map_err(invalid)?;
            for entry in mtp.values().filter(|e| e.role == TensorRole::Matrix) {
                retained.push((entry.shape[0], entry.shape[1]));
            }
            let delta = DeltaMixerGeometry {
                recurrence: DeltaNetGeometry {
                    value_heads: c.delta_net.num_value_heads as usize,
                    key_head_dim: c.delta_net.key_head_dim as usize,
                    value_head_dim: c.delta_net.value_head_dim as usize,
                    group_size: (c.delta_net.num_value_heads / c.delta_net.num_key_heads) as usize,
                },
                conv_kernel: c.delta_net.conv_kernel_dim as usize,
                rms_eps: geometry.rms_eps,
            };
            let convolution = delta
                .widths()
                .map_err(invalid)?
                .0
                .checked_mul(delta.conv_kernel)
                .ok_or_else(|| invalid("convolution extent overflow"))?;
            let mut layers = Vec::new();
            layers
                .try_reserve_exact(c.layer_types.len())
                .map_err(invalid)?;
            for kind in &c.layer_types {
                layers.push(DecoderAllocationLayer {
                    input_norm: hidden,
                    post_norm: hidden,
                    activation: DecoderActivation::SwiGlu,
                    ffn_sub_norm: 0,
                    ffn_sub_eps: geometry.rms_eps,
                    mixer: match kind {
                        Qwen35LayerType::FullAttention => DecoderAllocationMixer::Attention {
                            gated: true,
                            vectors: [0, 0, 0, 0, geometry.head_dim, geometry.head_dim],
                        },
                        Qwen35LayerType::DeltaNet => DecoderAllocationMixer::DeltaNet {
                            geometry: delta,
                            weights: DeltaMixerWeightLengths {
                                convolution,
                                norm: delta.recurrence.value_head_dim,
                                dt_bias: delta.recurrence.value_heads,
                                a_log: delta.recurrence.value_heads,
                            },
                        },
                    },
                });
            }
            Ok(Allocation {
                geometry,
                kind: DecoderAllocationKind::Hybrid,
                semantics: DecoderSemantics {
                    zero_centered_norm: true,
                    rotary_dim: c.rope.rotary_dim as usize,
                    qwen_rope_f32: true,
                },
                final_norm: hidden,
                layers,
                retained,
                matrices,
                rotation: width,
                limits,
            })
        }
    }
}
