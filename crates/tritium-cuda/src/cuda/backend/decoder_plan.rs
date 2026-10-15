//! Pure allocation geometry shared by metadata callers and the actual factory.
use super::*;
use tritium_spec::*;

pub(super) struct Plan {
    pub(super) payload: DecoderPayloadPlan,
    pub(super) widths: [usize; 17],
    pub(super) widest: usize,
    pub(super) has_delta: bool,
}
fn invalid(message: &str) -> BackendError {
    BackendError::InvalidInput(message.into())
}
fn mul(a: usize, b: usize) -> Result<usize, BackendError> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("decoder extent overflow"))
}
fn add(a: usize, b: usize) -> Result<usize, BackendError> {
    a.checked_add(b)
        .ok_or_else(|| invalid("decoder payload overflow"))
}
fn vector(len: usize, width: usize, optional: bool) -> Result<usize, BackendError> {
    if len != width && !(optional && len == 0) {
        return Err(invalid("decoder vector geometry or values"));
    }
    Ok(len)
}

pub(super) fn plan(
    metadata: DecoderAllocationMetadata<'_>,
    limits: DecoderLimits,
) -> Result<Plan, BackendError> {
    let g = metadata.geometry;
    let s = metadata.semantics;
    let (q, kv) = AttentionGeometry {
        query_heads: g.query_heads,
        kv_heads: g.kv_heads,
        head_dim: g.head_dim,
    }
    .widths()?;
    if g.hidden == 0
        || g.intermediate == 0
        || g.vocabulary == 0
        || g.head_dim > 1024
        || !g.head_dim.is_multiple_of(2)
        || s.rotary_dim == 0
        || !s.rotary_dim.is_multiple_of(2)
        || s.rotary_dim > g.head_dim
        || !g.rms_eps.is_finite()
        || g.rms_eps <= 0.
        || !g.rope_theta.is_finite()
        || g.rope_theta <= 0.
        || limits.max_context == 0
        || limits.max_segment_tokens == 0
        || limits.max_segment_tokens > limits.max_context
        || u32::try_from(limits.max_context).is_err()
    {
        return Err(invalid("unsupported decoder geometry or limits"));
    }
    match metadata.kind {
        DecoderAllocationKind::Standard if s == DecoderSemantics::standard(g.head_dim) => {}
        DecoderAllocationKind::Hybrid if s.zero_centered_norm && s.qwen_rope_f32 => {}
        _ => return Err(invalid("unsupported decoder family or numerical semantics")),
    }
    if metadata.layers.is_empty() {
        return Err(invalid("decoder has no layers"));
    }
    let m = limits.max_segment_tokens;
    CudaBackend::check_grad_launch_bounds(1, g.vocabulary, g.hidden)?;
    let mut preserved = vector(metadata.final_norm, g.hidden, false)?;
    let mut state_bytes = 0;
    let mut has_attention = false;
    let mut has_delta = false;
    let (mut aux_a, mut aux_b, mut aux_c, mut aux_d) = (0, 0, 0, 0);
    let (mut gate_width, mut conv_width, mut mixed_width) = (0, 0, 0);
    let mut widest = g.vocabulary.max(g.hidden).max(g.intermediate);
    let mut max_input = g.hidden.max(g.intermediate);
    for l in metadata.layers {
        preserved = add(preserved, vector(l.input_norm, g.hidden, false)?)?;
        preserved = add(preserved, vector(l.post_norm, g.hidden, false)?)?;
        preserved = add(preserved, vector(l.ffn_sub_norm, g.intermediate, true)?)?;
        if !matches!(
            l.activation,
            DecoderActivation::SwiGlu | DecoderActivation::Relu2
        ) || (l.activation == DecoderActivation::SwiGlu && l.ffn_sub_norm != 0)
            || (l.ffn_sub_norm != 0 && (!l.ffn_sub_eps.is_finite() || l.ffn_sub_eps <= 0.))
        {
            return Err(invalid("unsupported MLP semantics"));
        }
        for (rows, cols) in [(g.intermediate, g.hidden), (g.hidden, g.intermediate)] {
            CudaBackend::check_grad_launch_bounds(m, rows, cols)?;
        }
        match l.mixer {
            DecoderAllocationMixer::Attention { vectors, gated } => {
                if gated != (metadata.kind == DecoderAllocationKind::Hybrid)
                    || (metadata.kind == DecoderAllocationKind::Hybrid
                        && (l.activation != DecoderActivation::SwiGlu || l.ffn_sub_norm != 0))
                {
                    return Err(invalid("attention family differs from graph"));
                }
                has_attention = true;
                mixed_width = mixed_width.max(q);
                max_input = max_input.max(q);
                widest = widest.max(q).max(kv);
                let query_width = if gated { mul(2, q)? } else { q };
                widest = widest.max(query_width);
                for (rows, cols) in [(query_width, g.hidden), (kv, g.hidden), (g.hidden, q)] {
                    CudaBackend::check_grad_launch_bounds(m, rows, cols)?;
                }
                for (i, (&len, width)) in vectors
                    .iter()
                    .zip([q, q, kv, kv, g.head_dim, g.head_dim])
                    .enumerate()
                {
                    preserved = add(preserved, vector(len, width, !gated || i < 4)?)?;
                    if gated && i < 4 && len != 0 {
                        return Err(invalid("hybrid attention has unsupported optional vectors"));
                    }
                }
                state_bytes = add(state_bytes, mul(mul(mul(2, limits.max_context)?, kv)?, 4)?)?;
                if gated {
                    aux_a = aux_a.max(query_width);
                    gate_width = gate_width.max(q);
                }
            }
            DecoderAllocationMixer::DeltaNet { geometry, weights } => {
                if metadata.kind != DecoderAllocationKind::Hybrid
                    || l.activation != DecoderActivation::SwiGlu
                    || l.ffn_sub_norm != 0
                {
                    return Err(invalid("DeltaNet family differs from graph"));
                }
                has_delta = true;
                let (cv, value) = geometry.widths()?;
                let heads = geometry.recurrence.value_heads;
                if geometry.rms_eps.to_bits() != g.rms_eps.to_bits() {
                    return Err(invalid("DeltaNet epsilon differs from graph"));
                }
                state_bytes = add(
                    state_bytes,
                    delta_mixer::State::structural_payload_bytes(geometry, weights, limits)?,
                )?;
                for (rows, cols) in [
                    (cv, g.hidden),
                    (value, g.hidden),
                    (heads, g.hidden),
                    (g.hidden, value),
                ] {
                    CudaBackend::check_grad_launch_bounds(m, rows, cols)?;
                }
                aux_a = aux_a.max(cv);
                aux_b = aux_b.max(value);
                aux_c = aux_c.max(heads);
                aux_d = aux_d.max(heads);
                conv_width = conv_width.max(cv);
                mixed_width = mixed_width.max(value);
                widest = widest.max(cv).max(value).max(heads);
                max_input = max_input.max(value);
            }
            _ => return Err(BackendError::UnsupportedTensor),
        }
    }
    if metadata.rotation_workspace_width > max_input {
        return Err(invalid(
            "decoder rotation workspace exceeds projection geometry",
        ));
    }
    let widths = [
        g.hidden,
        g.hidden,
        g.hidden,
        if has_attention { q } else { 0 },
        if has_attention { kv } else { 0 },
        if has_attention { kv } else { 0 },
        mixed_width,
        g.intermediate,
        g.intermediate,
        aux_a,
        aux_b,
        aux_c,
        aux_d,
        gate_width,
        conv_width,
        metadata.rotation_workspace_width,
        if has_attention { s.rotary_dim } else { 0 },
    ];
    let mut workspace = 0;
    for width in widths {
        let n = mul(m, width)?;
        u32::try_from(n).map_err(|_| invalid("decoder workspace exceeds u32"))?;
        workspace = add(workspace, n)?;
    }
    if has_attention {
        u32::try_from(mul(m, g.query_heads)?).map_err(|_| invalid("attention grid exceeds u32"))?;
    }
    workspace = add(workspace, add(add(g.hidden, g.vocabulary)?, widest)?)?;
    let payload = DecoderPayloadPlan::new(
        metadata.matrix_payload_bytes,
        mul(preserved, 4)?,
        state_bytes,
        add(mul(workspace, 4)?, mul(m, 4)?)?,
    )?;
    Ok(Plan {
        payload,
        widths,
        widest,
        has_delta,
    })
}
