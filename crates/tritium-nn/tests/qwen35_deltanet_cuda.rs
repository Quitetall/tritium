//! Host/device parity for the Gated DeltaNet recurrence.
//!
//! The device path exists because `Qwen35DeltaNet::recurrent_forward` was 31.8%
//! of every CPU sample in a decode profile, and it is only usable if it is a
//! relocation rather than a reimplementation. Both paths run through the same
//! public `forward`, with the same `CudaBackend` serving the projections, so the
//! recurrence is the only thing that differs -- and the comparison is
//! `assert_eq!`, not a tolerance, because the kernel takes every transcendental
//! pre-computed from the host precisely so it can be exact.
#![cfg(feature = "cuda")]

use tritium_nn::{
    DenseLinear, Projection, Qwen35DeltaNet, Qwen35DeltaNetConfig, Qwen35DeltaNetWeights,
    Qwen35Dtype, Qwen35FullAttentionConfig, Qwen35LayerType, Qwen35MtpConfig,
    Qwen35NormWeightSemantics, Qwen35OutputGate, Qwen35RopeConfig, Qwen35RopeType,
    Qwen35TextConfig,
};

// Group size 3 matches Qwen3.6-27B's 48 value heads over 16 key heads, which is
// the ratio the kernel's `head / group_size` mapping has to get right.
const HIDDEN: usize = 8;
const KEY_HEADS: usize = 2;
const VALUE_HEADS: usize = 6;
const HEAD_DIM: usize = 8;
const CONV_KERNEL: usize = 4;
const KEY_WIDTH: usize = KEY_HEADS * HEAD_DIM;
const VALUE_WIDTH: usize = VALUE_HEADS * HEAD_DIM;
const CONV_WIDTH: usize = 2 * KEY_WIDTH + VALUE_WIDTH;

/// Deterministic spread-out values; a constant fill would hide index errors.
fn ramp(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = (index as f32 * 0.37 + seed).sin();
            let y = (index as f32 * 0.11 + seed * 1.7).cos();
            0.45 * x + 0.3 * y
        })
        .collect()
}

fn text_config() -> Qwen35TextConfig {
    Qwen35TextConfig {
        model_type: "qwen3_5_text".to_owned(),
        num_hidden_layers: 1,
        hidden_size: HIDDEN as u32,
        intermediate_size: 16,
        vocab_size: 16,
        max_position_embeddings: 64,
        full_attention_interval: 4,
        layer_types: vec![Qwen35LayerType::DeltaNet],
        full_attention: Qwen35FullAttentionConfig {
            num_heads: 1,
            num_key_value_heads: 1,
            head_dim: 8,
            bias: false,
            dropout: 0.0,
            output_gate: Qwen35OutputGate::Sigmoid,
            norm_weight_semantics: Qwen35NormWeightSemantics::ZeroCenteredOnePlusWeight,
        },
        delta_net: Qwen35DeltaNetConfig {
            conv_kernel_dim: CONV_KERNEL as u32,
            num_key_heads: KEY_HEADS as u32,
            num_value_heads: VALUE_HEADS as u32,
            key_head_dim: HEAD_DIM as u32,
            value_head_dim: HEAD_DIM as u32,
            state_arithmetic_dtype: Qwen35Dtype::Float32,
            output_gate: Qwen35OutputGate::Swish,
            gated_norm_weight_semantics: Qwen35NormWeightSemantics::UnitCenteredDirectWeight,
        },
        rope: Qwen35RopeConfig {
            theta: 10_000.0,
            partial_rotary_factor: 1.0,
            rotary_dim: 8,
            rope_type: Qwen35RopeType::Default,
            mrope_interleaved: true,
            mrope_section: [4, 0, 0],
        },
        rms_norm_eps: 1e-6,
        source_dtype: Qwen35Dtype::Bfloat16,
        use_cache: true,
        tied_embeddings: false,
        mtp: Qwen35MtpConfig {
            num_hidden_layers: 1,
            dedicated_embeddings: false,
        },
    }
}

fn dense(values: Vec<f32>, n_out: usize, k_in: usize) -> Projection {
    Projection::Dense(DenseLinear::new_exact(values, n_out, k_in).unwrap())
}

fn layer() -> Qwen35DeltaNet {
    let weights = Qwen35DeltaNetWeights::new(
        dense(ramp(CONV_WIDTH * HIDDEN, 0.3), CONV_WIDTH, HIDDEN),
        dense(ramp(VALUE_WIDTH * HIDDEN, 1.1), VALUE_WIDTH, HIDDEN),
        dense(ramp(VALUE_HEADS * HIDDEN, 2.4), VALUE_HEADS, HIDDEN),
        dense(ramp(VALUE_HEADS * HIDDEN, 3.9), VALUE_HEADS, HIDDEN),
        dense(ramp(HIDDEN * VALUE_WIDTH, 5.2), HIDDEN, VALUE_WIDTH),
        ramp(CONV_WIDTH * CONV_KERNEL, 6.6),
        ramp(HEAD_DIM, 7.3).iter().map(|v| 1.0 + v).collect(),
        ramp(VALUE_HEADS, 8.1),
        ramp(VALUE_HEADS, 9.4),
    );
    Qwen35DeltaNet::new(&text_config(), weights).unwrap()
}

/// Prefill, then three cached single-token continuations, returning every
/// output row produced along the way.
fn run(backend: &dyn tritium_spec::TernaryBackend, prefill: usize) -> Vec<f32> {
    let mixer = layer();
    let mut cache = mixer.new_cache().unwrap();
    let mut collected = Vec::new();

    let input = ramp(prefill * HIDDEN, 11.7);
    let mut out = vec![f32::NAN; prefill * HIDDEN];
    mixer
        .forward(backend, &input, prefill, &mut cache, &mut out)
        .unwrap();
    collected.extend_from_slice(&out);

    for step in 0..3 {
        let token = ramp(HIDDEN, 20.0 + step as f32);
        let mut step_out = vec![f32::NAN; HIDDEN];
        mixer
            .forward(backend, &token, 1, &mut cache, &mut step_out)
            .unwrap();
        collected.extend_from_slice(&step_out);
    }
    collected
}

/// Both paths, and the proof that they were both actually taken.
///
/// This is deliberately ONE test. Dispatch is chosen by a process-global
/// environment variable, so two tests flipping it run in parallel threads and
/// corrupt each other's runs -- which is how this file first "failed": the host
/// pass was partly executed on the device. Worse, the same race can make a split
/// pair pass while proving nothing, by sending both runs down the same path.
/// Keep the flag's lifetime inside a single test body.
#[test]
fn device_recurrence_matches_the_host_reduction_bit_for_bit() {
    let Ok(cuda) = tritium_cuda::CudaBackend::new(0) else {
        eprintln!("skipping DeltaNet device parity: no CUDA device");
        return;
    };

    // SAFETY: the only mutation of this variable in the crate, and this test
    // holds it for the whole body. `0` is the explicit opt-out; the device path
    // is the default, so the host run is the one that needs the flag.
    unsafe { std::env::set_var("TRITIUM_DELTANET_CUDA", "0") };
    let host = run(&cuda, 5);
    assert!(
        !residency_probe(&cuda),
        "the flag was clear but the recurrence still went to the device"
    );

    // SAFETY: as above -- this test is the variable's only writer and no forward
    // is in flight between calls.
    unsafe { std::env::set_var("TRITIUM_DELTANET_CUDA", "1") };
    let device = run(&cuda, 5);
    let device_ran = residency_probe(&cuda);
    // SAFETY: as above.
    unsafe { std::env::remove_var("TRITIUM_DELTANET_CUDA") };
    // With the variable unset the device path must still be the one chosen.
    assert!(
        residency_probe(&cuda),
        "the device recurrence is supposed to be the default on a CUDA backend"
    );

    assert!(
        device_ran,
        "the flag was set and the backend was CUDA, but the recurrence stayed on the host, \
         so this comparison would have compared the host path against itself"
    );
    assert_eq!(device.len(), host.len());
    assert!(
        host.iter().all(|value| value.is_finite()),
        "host reference produced a non-finite value"
    );
    // A constant-output bug would satisfy equality trivially.
    let first = host[0];
    assert!(
        host.iter().any(|value| *value != first),
        "reference output is constant, so this comparison proves nothing"
    );
    assert_eq!(device, host, "device recurrence diverged from the host");
}

/// Run one short forward and report whether it left the state on the device.
fn residency_probe(backend: &dyn tritium_spec::TernaryBackend) -> bool {
    let mixer = layer();
    let mut cache = mixer.new_cache().unwrap();
    let input = ramp(2 * HIDDEN, 4.2);
    let mut out = vec![f32::NAN; 2 * HIDDEN];
    mixer
        .forward(backend, &input, 2, &mut cache, &mut out)
        .unwrap();
    cache.is_device_resident()
}
