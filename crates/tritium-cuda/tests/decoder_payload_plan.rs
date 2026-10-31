//! Pure metadata contracts: no CUDA context, transfers or qualification claims.
#![cfg(feature = "cuda")]
use tritium_cuda::CudaBackend;
use tritium_spec::*;

fn geometry() -> DecoderGeometry {
    DecoderGeometry {
        hidden: 8,
        intermediate: 12,
        query_heads: 2,
        kv_heads: 1,
        head_dim: 4,
        vocabulary: 16,
        rms_eps: 1e-6,
        rope_theta: 10_000.,
    }
}
fn limits() -> DecoderLimits {
    DecoderLimits {
        max_context: 7,
        max_segment_tokens: 3,
    }
}
fn attention(gated: bool) -> DecoderAllocationLayer {
    DecoderAllocationLayer {
        input_norm: 8,
        post_norm: 8,
        activation: DecoderActivation::SwiGlu,
        ffn_sub_norm: 0,
        ffn_sub_eps: 1e-6,
        mixer: DecoderAllocationMixer::Attention {
            gated,
            vectors: if gated { [0, 0, 0, 0, 4, 4] } else { [0; 6] },
        },
    }
}
fn delta() -> DecoderAllocationLayer {
    DecoderAllocationLayer {
        mixer: DecoderAllocationMixer::DeltaNet {
            geometry: DeltaMixerGeometry {
                recurrence: DeltaNetGeometry {
                    value_heads: 2,
                    key_head_dim: 4,
                    value_head_dim: 4,
                    group_size: 2,
                },
                conv_kernel: 3,
                rms_eps: 1e-6,
            },
            weights: DeltaMixerWeightLengths {
                convolution: 48,
                norm: 4,
                dt_bias: 2,
                a_log: 2,
            },
        },
        ..attention(true)
    }
}
fn metadata(layers: &[DecoderAllocationLayer], hybrid: bool) -> DecoderAllocationMetadata<'_> {
    DecoderAllocationMetadata {
        geometry: geometry(),
        kind: if hybrid {
            DecoderAllocationKind::Hybrid
        } else {
            DecoderAllocationKind::Standard
        },
        semantics: DecoderSemantics {
            zero_centered_norm: hybrid,
            rotary_dim: 4,
            qwen_rope_f32: hybrid,
        },
        final_norm: 8,
        layers,
        matrix_payload_bytes: 1000,
        retained_matrix_shapes: &[],
        rotation_workspace_width: 0,
    }
}
fn plan(layers: &[DecoderAllocationLayer], hybrid: bool) -> DecoderPayloadPlan {
    CudaBackend::plan_decoder_metadata(metadata(layers, hybrid), limits()).unwrap()
}

#[test]
fn unsupported_adapter_keeps_structural_cost_unknown() {
    let layers = [attention(false)];
    assert_eq!(
        tritium_cpu::CpuBackend::new()
            .decoder_payload_plan(metadata(&layers, false), limits())
            .unwrap(),
        None
    );
}

#[test]
fn pure_standard_and_hybrid_categories_match_independent_allocation_sums() {
    for (layers, hybrid, preserved, state, workspace, total) in [
        (vec![attention(false)], false, 96, 224, 1084, 2404),
        (vec![attention(true)], true, 128, 224, 1372, 2724),
        (vec![delta()], true, 96, 864, 1372, 3332),
        (vec![attention(true), delta()], true, 192, 1088, 1708, 3988),
    ] {
        let got = plan(&layers, hybrid);
        assert_eq!(got.matrix_payload_bytes(), 1000);
        assert_eq!(got.preserved_payload_bytes(), preserved);
        assert_eq!(got.state_payload_bytes(), state);
        assert_eq!(got.workspace_payload_bytes(), workspace);
        assert_eq!(got.payload_bytes(), total);
    }
}

#[test]
fn rotation_retained_matrix_and_stream_costs_remain_distinct() {
    let layers = [attention(true), delta()];
    let base = metadata(&layers, true);
    let plan = |m, l| CudaBackend::plan_decoder_metadata(m, l).unwrap();
    let original = plan(base, limits());
    let retained = plan(
        DecoderAllocationMetadata {
            matrix_payload_bytes: 1256,
            ..base
        },
        limits(),
    );
    assert_eq!(retained.payload_bytes(), original.payload_bytes() + 256);
    assert_eq!(
        retained.workspace_payload_bytes(),
        original.workspace_payload_bytes()
    );
    let rotated = plan(
        DecoderAllocationMetadata {
            rotation_workspace_width: 12,
            ..base
        },
        limits(),
    );
    assert_eq!(
        rotated.payload_bytes(),
        original.payload_bytes() + 3 * 12 * 4
    );
    assert_eq!(
        rotated.matrix_payload_bytes(),
        original.matrix_payload_bytes()
    );
    let context = plan(
        base,
        DecoderLimits {
            max_context: 8,
            ..limits()
        },
    );
    assert_eq!(
        context.state_payload_bytes(),
        original.state_payload_bytes() + 32
    );
    assert_eq!(
        context.workspace_payload_bytes(),
        original.workspace_payload_bytes()
    );
    let segment = plan(
        base,
        DecoderLimits {
            max_segment_tokens: 4,
            ..limits()
        },
    );
    assert_eq!(
        segment.workspace_payload_bytes(),
        original.workspace_payload_bytes() + 516
    );
    assert_eq!(
        segment.state_payload_bytes(),
        original.state_payload_bytes()
    );
}

#[test]
fn structural_planner_rejects_invalid_lengths_semantics_limits_and_overflow() {
    let layers = [attention(false)];
    let base = metadata(&layers, false);
    for limits in [
        DecoderLimits {
            max_context: 0,
            ..limits()
        },
        DecoderLimits {
            max_segment_tokens: 0,
            ..limits()
        },
        DecoderLimits {
            max_segment_tokens: 8,
            ..limits()
        },
    ] {
        assert!(CudaBackend::plan_decoder_metadata(base, limits).is_err());
    }
    for bad in [
        DecoderAllocationMetadata {
            final_norm: 7,
            ..base
        },
        DecoderAllocationMetadata {
            retained_matrix_shapes: &[(0, 1)],
            ..base
        },
        DecoderAllocationMetadata {
            retained_matrix_shapes: &[(usize::MAX, 2)],
            ..base
        },
        DecoderAllocationMetadata {
            layers: &[],
            ..base
        },
        DecoderAllocationMetadata {
            matrix_payload_bytes: usize::MAX,
            ..base
        },
        DecoderAllocationMetadata {
            rotation_workspace_width: 13,
            ..base
        },
        DecoderAllocationMetadata {
            kind: DecoderAllocationKind::Hybrid,
            ..base
        },
        DecoderAllocationMetadata {
            geometry: DecoderGeometry {
                hidden: usize::MAX,
                ..geometry()
            },
            ..base
        },
        DecoderAllocationMetadata {
            geometry: DecoderGeometry {
                rms_eps: f32::NAN,
                ..geometry()
            },
            ..base
        },
        DecoderAllocationMetadata {
            semantics: DecoderSemantics {
                rotary_dim: 3,
                ..base.semantics
            },
            ..base
        },
    ] {
        assert!(CudaBackend::plan_decoder_metadata(bad, limits()).is_err());
    }
    let mut bad_layers = [attention(true), delta()];
    bad_layers[0].input_norm = 7;
    assert!(CudaBackend::plan_decoder_metadata(metadata(&bad_layers, true), limits()).is_err());
    let mut bad_delta = delta();
    if let DecoderAllocationMixer::DeltaNet { weights, .. } = &mut bad_delta.mixer {
        weights.norm = 3;
    }
    assert!(CudaBackend::plan_decoder_metadata(metadata(&[bad_delta], true), limits()).is_err());
    assert!(CudaBackend::plan_decoder_metadata(metadata(&[delta()], false), limits()).is_err());
    assert!(DecoderPayloadPlan::new(usize::MAX, 1, 0, 0).is_err());
}
