//! Actual common-loader whole standard graph, not model quality/performance.
use super::*;
use tritium_nn::{ModelSource, StreamLimits};
use tritium_spec::{
    DecoderActivation, DecoderExecution, DecoderGeometry, DecoderGraph, DecoderLayer,
    DecoderLayers, DecoderLimits, DecoderSemantics,
};

#[path = "resident_resources.rs"]
mod resources;

fn cuda(budget: Option<usize>) -> Option<tritium_cuda::CudaBackend> {
    match tritium_cuda::CudaBackend::new(0) {
        Ok(backend) => Some(match budget {
            Some(budget) => backend.with_resident_decoder(budget),
            None => backend,
        }),
        Err(error) if std::env::var("TRITIUM_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("required CUDA: {error}")
        }
        Err(error) => {
            eprintln!("SKIP resident decoder: {error}");
            None
        }
    }
}
fn fixture(arch: &str, tied: bool, additive: Option<(PlaneCodec, Basis)>) -> (HfFixture, Vec<u8>) {
    let hf = if additive.is_some() {
        HfFixture::with_dimensions(arch, tied, 64, 128, 128, None)
    } else {
        HfFixture::new(arch, tied, 6, None)
    };
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(hf.dir.join("config.json")).unwrap()).unwrap();
    config["num_hidden_layers"] = 2.into();
    if arch == "bitnet" {
        config["hidden_act"] = "relu2".into();
    }
    std::fs::write(
        hf.dir.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let mut hf = hf;
    let second: Vec<_> = hf
        .tensors
        .iter()
        .filter(|(name, _, _)| name.starts_with("model.layers.0."))
        .map(|(name, shape, values)| {
            (
                name.replace("layers.0.", "layers.1."),
                shape.clone(),
                values.iter().map(|v| v * 0.91).collect(),
            )
        })
        .collect();
    hf.tensors.extend(second);
    let mut package = TritFixture::new(&hf, additive);
    package.asset("tokenizer.json", stream_limits::tokenizer());
    package.asset(
        "tokenizer_config.json",
        br#"{"bos_token":"t8","eos_token":"t9"}"#.to_vec(),
    );
    (hf, package.bytes())
}
fn load(
    bytes: &[u8],
    backend: Box<dyn TernaryBackend>,
    budget: &TensorPayloadBudget,
) -> Result<tritium_nn::LoadedModel, tritium_nn::NnError> {
    let mut recorder = EvidenceRecorder::new(
        "resident-standard",
        [
            TENSOR_UPLOADED_REGISTRATION,
            tritium_evidence::MODEL_LOADED_REGISTRATION,
        ],
    )
    .unwrap();
    tritium_nn::load_with_limits(
        ModelSource::Trit {
            bytes,
            level: "standard",
        },
        backend,
        TensorUploadPolicy::default(),
        budget,
        StreamLimits {
            max_context: 7,
            max_segment_tokens: 3,
            max_host_cache_payload_bytes: u64::MAX,
        },
        TensorUploadContext {
            recorder: &mut recorder,
            span: "load",
            parent: None,
            logical_time: 0,
        },
    )
}
fn compare(got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len());
    assert!(expected.iter().any(|v| v.abs() > 1e-5), "zero oracle");
    for (&a, &b) in got.iter().zip(expected) {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= 2e-4 + b.abs() * 2e-4,
            "resident={a}, host={b}"
        );
    }
}

fn structural_payload(runner: &tritium_nn::Runner, matrices: usize, rotated: bool) -> usize {
    use tritium_spec::{
        DecoderAllocationKind, DecoderAllocationLayer, DecoderAllocationMetadata,
        DecoderAllocationMixer,
    };
    let tritium_nn::ArchitectureSpec::Standard {
        config: c,
        variation: v,
        vocab_size,
    } = runner.architecture()
    else {
        panic!("standard fixture")
    };
    let hidden = c.n_embd as usize;
    let intermediate = c.n_ff as usize;
    let dim = c.head_dim() as usize;
    let q = c.n_head as usize * dim;
    let kv = c.n_head_kv as usize * dim;
    let relu = v.mlp == tritium_nn::MlpKind::Relu2;
    let layer = DecoderAllocationLayer {
        input_norm: hidden,
        post_norm: hidden,
        activation: if relu {
            DecoderActivation::Relu2
        } else {
            DecoderActivation::SwiGlu
        },
        ffn_sub_norm: if relu && v.ffn_sub_norm {
            intermediate
        } else {
            0
        },
        ffn_sub_eps: c.rms_eps,
        mixer: DecoderAllocationMixer::Attention {
            gated: false,
            vectors: [
                if v.attn_sub_norm { q } else { 0 },
                if v.qkv_bias { q } else { 0 },
                if v.qkv_bias { kv } else { 0 },
                if v.qkv_bias { kv } else { 0 },
                if v.qk_norm { dim } else { 0 },
                if v.qk_norm { dim } else { 0 },
            ],
        },
    };
    let layers = vec![layer; c.n_layers as usize];
    tritium_cuda::CudaBackend::plan_decoder_metadata(
        DecoderAllocationMetadata {
            geometry: DecoderGeometry {
                hidden,
                intermediate,
                query_heads: c.n_head as usize,
                kv_heads: c.n_head_kv as usize,
                head_dim: dim,
                vocabulary: *vocab_size,
                rms_eps: c.rms_eps,
                rope_theta: c.rope_theta,
            },
            kind: DecoderAllocationKind::Standard,
            semantics: DecoderSemantics::standard(dim),
            final_norm: hidden,
            layers: &layers,
            matrix_payload_bytes: matrices,
            rotation_workspace_width: if rotated {
                hidden.max(intermediate).max(q)
            } else {
                0
            },
        },
        DecoderLimits {
            max_context: 7,
            max_segment_tokens: 3,
        },
    )
    .unwrap()
    .payload_bytes()
}

#[test]
fn whole_standard_dense_additive_rotated_tied_and_variations_match_host() {
    let Some(probe) = cuda(None) else {
        return;
    };
    assert!(!probe.resident_decoder_requested());
    drop(probe);
    let mut comparisons = 0;
    for arch in ["llama", "qwen2", "qwen3", "bitnet"] {
        for tied in [false, true] {
            for additive in [
                None,
                Some((PlaneCodec::B3, Basis::Hadamard { block: 128 })),
                Some((
                    PlaneCodec::D2,
                    Basis::SignedRht {
                        block: 128,
                        seed: 19,
                        domain: 27,
                    },
                )),
                Some((PlaneCodec::S34, Basis::Identity)),
            ] {
                let (_hf, bytes) = fixture(arch, tied, additive);
                let cpu_budget = TensorPayloadBudget::new(1_000_000);
                let gpu_budget = TensorPayloadBudget::new(1_000_000);
                let mut cpu = load(
                    &bytes,
                    Box::new(tritium_cpu::CpuBackend::new()),
                    &cpu_budget,
                )
                .unwrap();
                let mut gpu = load(
                    &bytes,
                    Box::new(cuda(Some(1_000_000)).unwrap()),
                    &gpu_budget,
                )
                .unwrap();
                assert_eq!(cpu.runner.execution(), DecoderExecution::Host);
                assert_eq!(cpu.runner.resident_payload_bytes(), None);
                assert_eq!(gpu.runner.execution(), DecoderExecution::ResidentDeviceF32);
                assert!(gpu.runner.resident_payload_bytes().unwrap() < 1_000_000);
                assert_eq!(
                    gpu.runner.resident_payload_bytes().unwrap(),
                    structural_payload(
                        &gpu.runner,
                        usize::try_from(gpu_budget.used_bytes()).unwrap(),
                        additive.is_some_and(|(_, b)| b != Basis::Identity)
                    )
                );
                for (tokens, positions) in [
                    (&[1, 3][..], &[0, 1][..]),
                    (&[4, 2, 5][..], &[2, 3, 4][..]),
                    (&[6][..], &[5][..]),
                    (&[7][..], &[6][..]),
                ] {
                    compare(
                        &gpu.runner.forward(tokens, positions).unwrap(),
                        &cpu.runner.forward(tokens, positions).unwrap(),
                    );
                    comparisons += 1;
                }
                assert!(gpu.runner.step(8).is_err());
                assert_eq!(gpu.runner.cache_len(), 7);
                gpu.runner.reset();
                cpu.runner.reset();
                compare(
                    &gpu.runner.prefill(&[3, 1, 2]).unwrap(),
                    &cpu.runner.prefill(&[3, 1, 2]).unwrap(),
                );
                comparisons += 1;
                drop(cpu);
                drop(gpu);
                assert_eq!(gpu_budget.used_bytes(), 0);
                assert_eq!(cpu_budget.used_bytes(), 0);
            }
        }
    }
    assert_eq!(comparisons, 160);
    eprintln!("whole resident decoder host comparisons={comparisons}, two-layer graphs=32");
}

#[test]
fn graph_wide_cancellation_retry_reset_invalid_input_and_exact_payload_budget() {
    let Some(backend) = cuda(Some(1_000_000)) else {
        return;
    };
    let (_hf, bytes) = fixture("qwen3", true, None);
    let budget = TensorPayloadBudget::new(1_000_000);
    let mut gpu = load(&bytes, Box::new(backend), &budget).unwrap();
    let admitted = gpu.runner.resident_payload_bytes().unwrap();
    let baseline_prefix = gpu.runner.prefill(&[1, 3]).unwrap();
    let baseline = gpu.runner.step(4).unwrap();
    let mut cancellations = 0;
    for cut in 1..=8 {
        gpu.runner.reset();
        assert_eq!(gpu.runner.prefill(&[1, 3]).unwrap(), baseline_prefix);
        let calls = std::cell::Cell::new(0);
        let result = gpu
            .runner
            .forward_cancellable(&[4], &[2], &|| {
                calls.set(calls.get() + 1);
                calls.get() >= cut
            })
            .unwrap();
        if let Some(logits) = result {
            assert_eq!(logits, baseline);
            assert_eq!(gpu.runner.cache_len(), 3);
        } else {
            cancellations += 1;
            assert_eq!(gpu.runner.cache_len(), 2);
            assert_eq!(gpu.runner.step(4).unwrap(), baseline);
        }
    }
    assert!(cancellations >= 5, "no late graph cancellation exercised");
    for (tokens, positions) in [
        (&[99][..], &[3][..]),
        (&[4][..], &[0][..]),
        (&[1, 2, 3, 4][..], &[3, 4, 5, 6][..]),
    ] {
        assert!(gpu.runner.forward(tokens, positions).is_err());
        assert_eq!(gpu.runner.cache_len(), 3);
    }
    drop(gpu);
    assert_eq!(budget.used_bytes(), 0);
    let exact = load(&bytes, Box::new(cuda(Some(admitted)).unwrap()), &budget).unwrap();
    assert_eq!(exact.runner.resident_payload_bytes(), Some(admitted));
    drop(exact);
    assert_eq!(budget.used_bytes(), 0);
    let rejected = load(&bytes, Box::new(cuda(Some(admitted - 1)).unwrap()), &budget);
    assert!(
        rejected.is_err(),
        "under-budget resident mode silently fell back"
    );
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn resident_factory_checks_exact_owner_and_deduplicates_shared_payload() {
    let Some(backend) = cuda(Some(1_000_000)) else {
        return;
    };
    let backend = std::sync::Arc::new(backend);
    let values = vec![0.1; 64];
    let handle: std::sync::Arc<dyn tritium_spec::DeviceBuffer> = std::sync::Arc::from(
        backend
            .upload_tensor(TensorView::Dense {
                rows: 8,
                cols: 8,
                values: &values,
            })
            .unwrap(),
    );
    let norm = vec![1.; 8];
    let layer = DecoderLayer {
        attention_norm: &norm,
        ffn_norm: &norm,
        query: &handle,
        key: &handle,
        value: &handle,
        attention_output: &handle,
        gate: &handle,
        up: &handle,
        down: &handle,
        attention_sub_norm: &[],
        query_bias: &[],
        key_bias: &[],
        value_bias: &[],
        query_norm: &[],
        key_norm: &[],
        activation: DecoderActivation::SwiGlu,
        ffn_sub_norm: &[],
        ffn_sub_eps: 1e-5,
    };
    let layers = [layer];
    let graph = || DecoderGraph {
        geometry: DecoderGeometry {
            hidden: 8,
            intermediate: 8,
            query_heads: 1,
            kv_heads: 1,
            head_dim: 8,
            vocabulary: 8,
            rms_eps: 1e-5,
            rope_theta: 10000.,
        },
        embedding: &handle,
        head: &handle,
        final_norm: &norm,
        layers: DecoderLayers::Standard(&layers),
        semantics: DecoderSemantics::standard(8),
        retained_matrices: &[],
    };
    let limits = DecoderLimits {
        max_context: 4,
        max_segment_tokens: 2,
    };
    let foreign = std::sync::Arc::new(cuda(Some(1_000_000)).unwrap());
    assert!(foreign.resident_decoder(graph(), limits).is_err());
    let mut decoder = std::sync::Arc::clone(&backend)
        .resident_decoder(graph(), limits)
        .unwrap()
        .unwrap();
    assert_eq!(decoder.payload_bytes(), 1352);
    let logits = decoder
        .forward(&[1, 2], &[0, 1], &|| false)
        .unwrap()
        .unwrap();
    assert!(logits.iter().all(|v| v.is_finite() && *v > 0.));
    assert_eq!(decoder.len(), 2);

    // A fault-control fixture: zero-valued prefix is finite, but one token
    // overflows query projection after its layers stage KV. This is not the
    // accuracy oracle; the nonzero public-loader comparisons exercise numerics.
    let huge = vec![f32::MAX; 64];
    let huge: std::sync::Arc<dyn tritium_spec::DeviceBuffer> = std::sync::Arc::from(
        backend
            .upload_tensor(TensorView::Dense {
                rows: 8,
                cols: 8,
                values: &huge,
            })
            .unwrap(),
    );
    let mut token_table = vec![0.; 64];
    token_table[32..40].fill(1.);
    let token_table: std::sync::Arc<dyn tritium_spec::DeviceBuffer> = std::sync::Arc::from(
        backend
            .upload_tensor(TensorView::Dense {
                rows: 8,
                cols: 8,
                values: &token_table,
            })
            .unwrap(),
    );
    let fault_layers: [_; 2] = std::array::from_fn(|_| DecoderLayer {
        attention_norm: &norm,
        ffn_norm: &norm,
        query: &huge,
        key: &handle,
        value: &handle,
        attention_output: &handle,
        gate: &handle,
        up: &handle,
        down: &handle,
        attention_sub_norm: &[],
        query_bias: &[],
        key_bias: &[],
        value_bias: &[],
        query_norm: &[],
        key_norm: &[],
        activation: DecoderActivation::SwiGlu,
        ffn_sub_norm: &[],
        ffn_sub_eps: 1e-5,
    });
    let mut fault = backend
        .resident_decoder(
            DecoderGraph {
                geometry: graph().geometry,
                embedding: &token_table,
                head: &handle,
                final_norm: &norm,
                layers: DecoderLayers::Standard(&fault_layers),
                semantics: DecoderSemantics::standard(8),
                retained_matrices: &[],
            },
            limits,
        )
        .unwrap()
        .unwrap();
    assert!(
        fault
            .forward(&[1], &[0], &|| false)
            .unwrap()
            .unwrap()
            .iter()
            .all(|&v| v == 0.)
    );
    let baseline = fault.forward(&[2], &[1], &|| false).unwrap().unwrap();
    fault.reset();
    fault.forward(&[1], &[0], &|| false).unwrap();
    let error = fault.forward(&[4], &[1], &|| false).unwrap_err();
    assert!(
        error.to_string().contains("non-finite final logits"),
        "{error}"
    );
    assert_eq!(fault.len(), 1);
    assert_eq!(
        fault.forward(&[2], &[1], &|| false).unwrap().unwrap(),
        baseline
    );
}
