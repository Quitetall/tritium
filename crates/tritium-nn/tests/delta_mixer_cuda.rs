//! Actual Qwen DeltaNet module/cache oracle, not model quality or performance.
#![cfg(feature = "cuda")]
use tritium_nn::*;
use tritium_spec::{
    DecoderLimits, DeltaMixerGeometry, DeltaMixerSegment, DeltaMixerWeights, DeltaNetGeometry,
    ResidentDeltaMixer, TernaryBackend,
};

const H: usize = 8;
fn ramp(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 13 + seed * 7) % 43) as f32 / 43. - 0.48)
        .collect()
}
fn projection(values: Vec<f32>, rows: usize, cols: usize) -> Projection {
    Projection::Dense(DenseLinear::new_exact(values, rows, cols).unwrap())
}
struct Fixture {
    geometry: DeltaMixerGeometry,
    convolution: Vec<f32>,
    norm: Vec<f32>,
    dt: Vec<f32>,
    a_log: Vec<f32>,
    qkv: Vec<f32>,
    z: Vec<f32>,
    beta: Vec<f32>,
    decay: Vec<f32>,
    out: Vec<f32>,
}
impl Fixture {
    fn new(heads: usize, dk: usize, dv: usize, group: usize, kernel: usize) -> Self {
        let geometry = DeltaMixerGeometry {
            recurrence: DeltaNetGeometry {
                value_heads: heads,
                key_head_dim: dk,
                value_head_dim: dv,
                group_size: group,
            },
            conv_kernel: kernel,
            rms_eps: 1e-6,
        };
        let (conv, value) = geometry.widths().unwrap();
        let scaled = |n, seed| ramp(n, seed).iter().map(|v| v * 0.2).collect();
        Self {
            geometry,
            convolution: scaled(conv * kernel, 1),
            norm: ramp(dv, 2).iter().map(|v| 1. + v * 0.1).collect(),
            dt: scaled(heads, 3),
            a_log: scaled(heads, 4),
            qkv: scaled(conv * H, 5),
            z: scaled(value * H, 6),
            beta: scaled(heads * H, 7),
            decay: scaled(heads * H, 8),
            out: scaled(H * value, 9),
        }
    }
    fn weights(&self) -> DeltaMixerWeights<'_> {
        DeltaMixerWeights {
            convolution: &self.convolution,
            norm: &self.norm,
            dt_bias: &self.dt,
            a_log: &self.a_log,
        }
    }
    fn layer(&self) -> Qwen35DeltaNet {
        let g = self.geometry.recurrence;
        let (conv, value) = self.geometry.widths().unwrap();
        let config = Qwen35TextConfig {
            model_type: "qwen3_5_text".into(),
            num_hidden_layers: 1,
            hidden_size: H as u32,
            intermediate_size: 16,
            vocab_size: 16,
            max_position_embeddings: 13,
            full_attention_interval: 4,
            layer_types: vec![Qwen35LayerType::DeltaNet],
            full_attention: Qwen35FullAttentionConfig {
                num_heads: 1,
                num_key_value_heads: 1,
                head_dim: 8,
                bias: false,
                dropout: 0.,
                output_gate: Qwen35OutputGate::Sigmoid,
                norm_weight_semantics: Qwen35NormWeightSemantics::ZeroCenteredOnePlusWeight,
            },
            delta_net: Qwen35DeltaNetConfig {
                conv_kernel_dim: self.geometry.conv_kernel as u32,
                num_key_heads: (g.value_heads / g.group_size) as u32,
                num_value_heads: g.value_heads as u32,
                key_head_dim: g.key_head_dim as u32,
                value_head_dim: g.value_head_dim as u32,
                state_arithmetic_dtype: Qwen35Dtype::Float32,
                output_gate: Qwen35OutputGate::Swish,
                gated_norm_weight_semantics: Qwen35NormWeightSemantics::UnitCenteredDirectWeight,
            },
            rope: Qwen35RopeConfig {
                theta: 10000.,
                partial_rotary_factor: 1.,
                rotary_dim: 8,
                rope_type: Qwen35RopeType::Default,
                mrope_interleaved: true,
                mrope_section: [4, 0, 0],
            },
            rms_norm_eps: self.geometry.rms_eps as f64,
            source_dtype: Qwen35Dtype::Bfloat16,
            use_cache: true,
            tied_embeddings: false,
            mtp: Qwen35MtpConfig {
                num_hidden_layers: 1,
                dedicated_embeddings: false,
            },
        };
        Qwen35DeltaNet::new(
            &config,
            Qwen35DeltaNetWeights::new(
                projection(self.qkv.clone(), conv, H),
                projection(self.z.clone(), value, H),
                projection(self.beta.clone(), g.value_heads, H),
                projection(self.decay.clone(), g.value_heads, H),
                projection(self.out.clone(), H, value),
                self.convolution.clone(),
                self.norm.clone(),
                self.dt.clone(),
                self.a_log.clone(),
            ),
        )
        .unwrap()
    }
    fn projected(&self, input: &[f32], tokens: usize) -> Inputs {
        let (conv, value) = self.geometry.widths().unwrap();
        let heads = self.geometry.recurrence.value_heads;
        let forward = |values: &[f32], rows| {
            let mut out = vec![0.; tokens * rows];
            projection(values.to_vec(), rows, H)
                .forward(&tritium_cpu::CpuBackend::new(), input, tokens, &mut out)
                .unwrap();
            out
        };
        Inputs {
            qkv: forward(&self.qkv, conv),
            z: forward(&self.z, value),
            beta: forward(&self.beta, heads),
            decay: forward(&self.decay, heads),
            tokens,
        }
    }
    fn project_output(&self, core: &[f32], tokens: usize) -> Vec<f32> {
        let mut out = vec![0.; tokens * H];
        projection(self.out.clone(), H, self.geometry.widths().unwrap().1)
            .forward(&tritium_cpu::CpuBackend::new(), core, tokens, &mut out)
            .unwrap();
        out
    }
}
struct Inputs {
    qkv: Vec<f32>,
    z: Vec<f32>,
    beta: Vec<f32>,
    decay: Vec<f32>,
    tokens: usize,
}
impl Inputs {
    fn stage(
        &self,
        mixer: &mut dyn ResidentDeltaMixer,
        output: &mut [f32],
    ) -> Result<(), tritium_spec::BackendError> {
        mixer.stage(DeltaMixerSegment {
            qkv: &self.qkv,
            z: &self.z,
            beta_logits: &self.beta,
            decay_logits: &self.decay,
            tokens: self.tokens,
            output,
        })
    }
}
fn close(actual: &[f32], expected: &[f32], absolute: f32, relative: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= absolute + b.abs() * relative,
            "lane {i}: device={a}, host={b}"
        );
    }
}
fn cuda() -> Option<tritium_cuda::CudaBackend> {
    match tritium_cuda::CudaBackend::new(0) {
        Ok(b) => Some(b),
        Err(e) if std::env::var("TRITIUM_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("required CUDA: {e}")
        }
        Err(e) => {
            eprintln!("SKIP DeltaNet full mixer: {e}");
            None
        }
    }
}
fn limits() -> DecoderLimits {
    DecoderLimits {
        max_context: 13,
        max_segment_tokens: 5,
    }
}

#[test]
fn full_device_mixer_matches_actual_qwen_module_and_both_committed_states() {
    let Some(backend) = cuda() else {
        return;
    };
    let fixtures: Vec<_> = [
        (6, 8, 8, 3, 4),
        (4, 7, 5, 2, 1),
        (3, 32, 16, 3, 4),
        (48, 128, 128, 3, 4),
    ]
    .into_iter()
    .map(|(h, k, v, g, c)| Fixture::new(h, k, v, g, c))
    .collect();
    let mut mixers: Vec<_> = fixtures
        .iter()
        .map(|f| {
            let payload = f
                .geometry
                .payload_bytes(f.weights(), limits(), usize::MAX)
                .unwrap();
            let mixer = backend
                .resident_delta_mixer(f.geometry, f.weights(), limits(), payload)
                .unwrap()
                .unwrap();
            assert_eq!(mixer.payload_bytes(), payload);
            mixer
        })
        .collect();
    drop(backend);
    let mut comparisons = 0;
    let mut positive_softplus = 0;
    let mut negative_softplus = 0;
    for (f, mixer) in fixtures.iter().zip(&mut mixers) {
        let oracle = f.layer();
        for prefill in [1, 5] {
            mixer.reset();
            let mut cache = oracle.new_cache().unwrap();
            for (iteration, tokens) in [prefill, 1, 1, 1].into_iter().enumerate() {
                let mut input = ramp(tokens * H, 20 + iteration);
                if iteration == 3 {
                    for value in &mut input {
                        *value *= 1500.;
                    }
                }
                let prepared = f.projected(&input, tokens);
                for (&value, &bias) in prepared.decay.iter().zip(&f.dt) {
                    positive_softplus += usize::from(value + bias > 20.);
                    negative_softplus += usize::from(value + bias < -20.);
                }
                let mut core = vec![f32::NAN; tokens * f.geometry.widths().unwrap().1];
                let base = mixer.len();
                prepared.stage(mixer.as_mut(), &mut core).unwrap();
                assert_eq!(mixer.len(), base);
                assert_eq!(mixer.staged_len(), Some(base + tokens));
                mixer.abort();
                mixer.commit();
                assert_eq!(mixer.len(), base);
                // A different abandoned tail/state must not affect exact retry.
                let alternate = f.projected(&ramp(tokens * H, 99), tokens);
                alternate.stage(mixer.as_mut(), &mut core).unwrap();
                prepared.stage(mixer.as_mut(), &mut core).unwrap();
                let device = f.project_output(&core, tokens);
                let mut expected = vec![f32::NAN; tokens * H];
                oracle
                    .forward(
                        &tritium_cpu::CpuBackend::new(),
                        &input,
                        tokens,
                        &mut cache,
                        &mut expected,
                    )
                    .unwrap();
                assert!(
                    expected.iter().any(|v| v.abs() > 1e-6),
                    "zero module oracle"
                );
                close(&device, &expected, 3e-5, 3e-4);
                mixer.commit();
                assert_eq!(mixer.len(), cache.len());
                let mut conv = vec![0.; f.convolution.len()];
                let mut state = vec![0.; f.geometry.recurrence.elements().unwrap()];
                mixer.read_committed(&mut conv, &mut state).unwrap();
                assert_eq!(conv, cache.conv_state());
                close(&state, cache.recurrent_state(), 2e-6, 2e-4);
                comparisons += 1;
            }
        }
    }
    assert_eq!(comparisons, 32);
    assert!(
        positive_softplus > 0 && negative_softplus > 0,
        "softplus tails were not exercised"
    );
    eprintln!(
        "actual Qwen DeltaNet forward/state comparisons={comparisons}, geometries=4; toy projections, not model qualification"
    );
}

#[test]
fn staging_failures_capacity_reset_and_payload_admission_are_transactional() {
    let Some(backend) = cuda() else {
        return;
    };
    let f = Fixture::new(6, 8, 8, 3, 4);
    let payload = f
        .geometry
        .payload_bytes(f.weights(), limits(), usize::MAX)
        .unwrap();
    assert!(
        backend
            .resident_delta_mixer(f.geometry, f.weights(), limits(), payload - 1)
            .is_err()
    );
    assert!(
        tritium_cpu::CpuBackend::new()
            .resident_delta_mixer(f.geometry, f.weights(), limits(), payload)
            .unwrap()
            .is_none()
    );
    let mut mixer = backend
        .resident_delta_mixer(f.geometry, f.weights(), limits(), payload)
        .unwrap()
        .unwrap();
    let prefix = f.projected(&ramp(2 * H, 1), 2);
    let mut output = vec![0.; 2 * f.geometry.widths().unwrap().1];
    prefix.stage(mixer.as_mut(), &mut output).unwrap();
    mixer.commit();
    let next = f.projected(&ramp(H, 2), 1);
    output.resize(f.geometry.widths().unwrap().1, 0.);
    next.stage(mixer.as_mut(), &mut output).unwrap();
    let baseline = output.clone();
    mixer.abort();
    let mut committed_conv = vec![0.; f.convolution.len()];
    let mut committed_state = vec![0.; f.geometry.recurrence.elements().unwrap()];
    mixer
        .read_committed(&mut committed_conv, &mut committed_state)
        .unwrap();
    for case in 0..6 {
        next.stage(mixer.as_mut(), &mut output).unwrap();
        let mut bad = f.projected(&ramp(H, 3), 1);
        match case {
            0 => {
                bad.qkv.pop();
            }
            1 => {
                bad.z[0] = f32::NAN;
            }
            2 => {
                bad.beta[0] = f32::INFINITY;
            }
            3 => {
                bad.decay[0] = f32::NEG_INFINITY;
            }
            4 => {
                bad = f.projected(&ramp(2 * H, 4), 2);
            }
            5 => {
                bad = f.projected(&[], 0);
            }
            _ => unreachable!(),
        }
        let mut rejected = vec![-1234.; bad.tokens * f.geometry.widths().unwrap().1];
        assert!(
            bad.stage(mixer.as_mut(), &mut rejected).is_err(),
            "case={case}"
        );
        assert!(rejected.iter().all(|&v| v == -1234.));
        assert_eq!(mixer.staged_len(), None);
        mixer.commit();
        assert_eq!(mixer.len(), 2);
        let mut conv = committed_conv.clone();
        let mut state = committed_state.clone();
        mixer.read_committed(&mut conv, &mut state).unwrap();
        assert_eq!(conv, committed_conv);
        assert_eq!(state, committed_state);
        next.stage(mixer.as_mut(), &mut output).unwrap();
        assert_eq!(output, baseline);
        mixer.abort();
    }
    for _ in 2..13 {
        next.stage(mixer.as_mut(), &mut output).unwrap();
        mixer.commit();
    }
    output.fill(-1234.);
    assert!(next.stage(mixer.as_mut(), &mut output).is_err());
    assert!(output.iter().all(|&v| v == -1234.));
    assert_eq!(mixer.len(), 13);
    mixer.reset();
    mixer
        .read_committed(&mut committed_conv, &mut committed_state)
        .unwrap();
    assert!(
        committed_conv
            .iter()
            .chain(&committed_state)
            .all(|&v| v == 0.)
    );
    let mut reset = vec![0.; 2 * f.geometry.widths().unwrap().1];
    prefix.stage(mixer.as_mut(), &mut reset).unwrap();
    mixer.commit();
    assert_eq!(mixer.len(), 2);

    // Valid finite coefficients/operands that force actual convolution overflow,
    // unlike a large-input assumption which can legitimately normalize to zero.
    let strong = vec![2.; f.convolution.len()];
    let weights = DeltaMixerWeights {
        convolution: &strong,
        ..f.weights()
    };
    let mut fault = backend
        .resident_delta_mixer(f.geometry, weights, limits(), payload)
        .unwrap()
        .unwrap();
    let (conv_width, value_width) = f.geometry.widths().unwrap();
    let zero = Inputs {
        qkv: vec![0.; conv_width],
        z: vec![0.; value_width],
        beta: vec![0.; 6],
        decay: vec![0.; 6],
        tokens: 1,
    };
    let mut output = vec![0.; value_width];
    zero.stage(fault.as_mut(), &mut output).unwrap();
    fault.commit();
    let overflowing = Inputs {
        qkv: vec![f32::MAX; conv_width],
        z: vec![f32::MAX; value_width],
        beta: vec![0.; 6],
        decay: vec![0.; 6],
        tokens: 1,
    };
    output.fill(-1234.);
    let error = overflowing.stage(fault.as_mut(), &mut output).unwrap_err();
    assert!(error.to_string().contains("non-finite output"), "{error}");
    assert!(output.iter().all(|&v| v == -1234.));
    assert_eq!(fault.len(), 1);
    assert_eq!(fault.staged_len(), None);
    fault.commit();
    assert_eq!(fault.len(), 1);
    fault
        .read_committed(&mut committed_conv, &mut committed_state)
        .unwrap();
    assert!(
        committed_conv
            .iter()
            .chain(&committed_state)
            .all(|&v| v == 0.)
    );
    zero.stage(fault.as_mut(), &mut output).unwrap();
    assert!(output.iter().all(|&v| v == 0.));
}
