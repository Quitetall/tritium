//! Production tensor loading and model execution, not model-quality qualification.

use tritium_core::{AdditiveView, Trit};
use tritium_evidence::{
    EvidenceRecorder, TENSOR_UPLOADED_REGISTRATION, verify_jsonl_with_registry,
};
use tritium_nn::{Linear, ModelRunner, TensorLoader, TokenEmbedding};
use tritium_runtime::{TensorPayloadBudget, TensorUploadContext};
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, TensorUploadPolicy,
    Transport,
};
use tritium_spec::{TensorExecution, TensorView, TernaryBackend};

#[test]
fn one_admitted_handle_executes_linear_gather_and_tied_head() {
    let cpu = tritium_cpu::CpuBackend::new();
    let budget = TensorPayloadBudget::new(10);
    let mut recorder = EvidenceRecorder::new("nn-tensor", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
    let layout = AdditiveLayout {
        rows: 2,
        cols: 4,
        tile: 256,
        group: 32,
        max_planes: 1,
        allocation: PlaneAllocation::Uniform,
        codec: PlaneCodec::D2,
        law: ADMITTED_LAWS[1].law,
        basis: Basis::SignedRht {
            block: 4,
            seed: 7,
            domain: 9,
        },
        transport: Transport::Raw,
    };
    let trits = [
        Trit::POS,
        Trit::NEG,
        Trit::ZERO,
        Trit::POS,
        Trit::POS,
        Trit::NEG,
        Trit::ZERO,
        Trit::POS,
    ];
    let view = AdditiveView::new(layout, 1, &trits, &[1., 2.]).unwrap();
    let mut loader = TensorLoader::new(
        &cpu,
        TensorUploadPolicy {
            native_only: true,
            max_payload_bytes: Some(10),
        },
        &budget,
        TensorUploadContext {
            recorder: &mut recorder,
            span: "load",
            parent: None,
            logical_time: 0,
        },
    );
    let linear: Linear = loader
        .upload("embedding", TensorView::Additive(view))
        .unwrap();
    assert_eq!(linear.tensor_caps().execution, TensorExecution::Native);
    assert_eq!(linear.payload_bytes(), 10);
    let embedding = TokenEmbedding::from_linear(linear.clone());
    assert_eq!(budget.used_bytes(), 10);
    let mut out = [0.; 2];
    linear.forward(&cpu, &[1.; 4], 1, &mut out).unwrap();
    assert_eq!(out, [-1., -2.]);
    let mut row = [0.; 4];
    embedding.gather_with_backend(&cpu, &[0], &mut row).unwrap();
    assert_eq!(row, [0.5, -0.5, 0.5, -1.5]);
    embedding
        .unembed_exact_with_backend(&cpu, &[1.; 4], &mut out)
        .unwrap();
    assert_eq!(out, [-1., -2.]);
    assert!(embedding.as_dense().is_none());
    assert_eq!(embedding.resident_bytes(), 10);
    drop(linear);
    assert_eq!(budget.used_bytes(), 10);
    drop(embedding);
    assert_eq!(budget.used_bytes(), 0);
    assert_eq!(recorder.events().len(), 1);
}

#[test]
fn invalid_weights_and_shapes_fail_without_output_or_budget_changes() {
    let cpu = tritium_cpu::CpuBackend::new();
    let budget = TensorPayloadBudget::new(16);
    let mut recorder = EvidenceRecorder::new("nn-invalid", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
    let mut loader = TensorLoader::new(
        &cpu,
        TensorUploadPolicy::default(),
        &budget,
        TensorUploadContext {
            recorder: &mut recorder,
            span: "load",
            parent: None,
            logical_time: 0,
        },
    );
    for values in [[0., f32::NAN, 0., 1.], [0., f32::INFINITY, 0., 1.]] {
        assert!(
            loader
                .upload(
                    "invalid",
                    TensorView::Dense {
                        rows: 2,
                        cols: 2,
                        values: &values
                    }
                )
                .is_err()
        );
        assert_eq!(budget.used_bytes(), 0);
    }
    let linear = loader
        .upload(
            "dense",
            TensorView::Dense {
                rows: 2,
                cols: 2,
                values: &[1., 2., 3., 4.],
            },
        )
        .unwrap();
    assert_eq!(budget.used_bytes(), 16);
    assert!(
        loader
            .upload(
                "too-much",
                TensorView::Dense {
                    rows: 1,
                    cols: 1,
                    values: &[1.]
                }
            )
            .is_err()
    );
    let mut output = [7.; 2];
    assert!(linear.forward(&cpu, &[1.], 1, &mut output).is_err());
    assert!(linear.forward(&cpu, &[], usize::MAX, &mut output).is_err());
    assert!(linear.gather(&cpu, &[0, 2], &mut output).is_err());
    assert!(linear.gather(&cpu, &[2], &mut output).is_err());
    let reference = tritium_testkit::ReferenceBackend::new();
    assert!(
        linear
            .forward(&reference, &[1., 1.], 1, &mut output)
            .is_err()
    );
    assert_eq!(output, [7.; 2]);
    linear.forward(&cpu, &[1., 1.], 1, &mut output).unwrap();
    assert_eq!(output, [3., 7.]);
    let embedding = TokenEmbedding::from_linear(linear.clone());
    assert!(embedding.gather(&[0], &mut output).is_err());
    assert!(embedding.unembed_exact(&[1., 1.], &mut output).is_err());
    assert!(!embedding.is_packed_salt());
    assert!(embedding.tensor_caps().is_some());
    drop(linear);
    drop(embedding);
    assert_eq!(budget.used_bytes(), 0);
    assert_eq!(recorder.events().len(), 1);
}

/// Real local HF files; no duplicated attention, norm or contraction oracle.
struct HfFixture {
    dir: std::path::PathBuf,
    payload_bytes: u64,
    upload_count: usize,
}

impl HfFixture {
    fn new(arch: &str, tied: bool, head_dim: usize, bad: Option<&str>) -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let seq = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("tritium-admitted-hf-{}-{seq}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let mut fixture = Self {
            dir,
            payload_bytes: 0,
            upload_count: 0,
        };
        let mut tensors = Vec::new();
        let mut add = |name: &str, shape: &[usize]| {
            let n = shape.iter().product::<usize>();
            let values: Vec<f32> = if shape.len() == 1 && name.ends_with("weight") {
                vec![1.; n]
            } else {
                (0..n)
                    .map(|i| ((i * 17 % 31) as f32 - 15.) * 0.01)
                    .collect()
            };
            if shape.len() == 2 {
                fixture.payload_bytes += u64::try_from(n * 4).unwrap();
                fixture.upload_count += 1;
            }
            tensors.push((name.to_owned(), shape.to_vec(), values));
        };
        add("model.embed_tokens.weight", &[10, 8]);
        for name in [
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        ] {
            add(name, &[8]);
        }
        for (name, rows, cols) in [
            ("self_attn.q_proj.weight", 2 * head_dim, 8),
            ("self_attn.k_proj.weight", head_dim, 8),
            ("self_attn.v_proj.weight", head_dim, 8),
            ("self_attn.o_proj.weight", 8, 2 * head_dim),
            ("mlp.gate_proj.weight", 16, 8),
            ("mlp.up_proj.weight", 16, 8),
            ("mlp.down_proj.weight", 8, 16),
        ] {
            add(&format!("model.layers.0.{name}"), &[rows, cols]);
        }
        if !tied {
            add("lm_head.weight", &[10, 8]);
        }
        if arch == "qwen2" {
            for (name, n) in [("q", 2 * head_dim), ("k", head_dim), ("v", head_dim)] {
                add(&format!("model.layers.0.self_attn.{name}_proj.bias"), &[n]);
            }
        }
        if arch == "qwen3" {
            for name in ["q", "k"] {
                add(
                    &format!("model.layers.0.self_attn.{name}_norm.weight"),
                    &[head_dim],
                );
            }
        }
        if bad == Some("missing-down") {
            tensors.retain(|(name, _, _)| !name.ends_with("down_proj.weight"));
        }
        if bad == Some("nan-norm") {
            tensors
                .iter_mut()
                .find(|(name, _, _)| name == "model.norm.weight")
                .unwrap()
                .2[0] = f32::NAN;
        }
        let shard = |shape: &[usize]| {
            if shape.len() == 2 {
                "weights.safetensors"
            } else {
                "vectors.safetensors"
            }
        };
        let mut weight_map = serde_json::Map::new();
        for filename in ["weights.safetensors", "vectors.safetensors"] {
            let mut header = serde_json::Map::new();
            let mut data = Vec::new();
            for (name, shape, values) in &tensors {
                if shard(shape) != filename {
                    continue;
                }
                let start = data.len();
                for value in values {
                    data.extend_from_slice(&value.to_le_bytes());
                }
                header.insert(name.clone(), serde_json::json!({"dtype":"F32", "shape":shape, "data_offsets":[start,data.len()]}));
                weight_map.insert(name.clone(), filename.into());
            }
            let mut json = serde_json::to_vec(&header).unwrap();
            while !json.len().is_multiple_of(8) {
                json.push(b' ');
            }
            let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
            bytes.extend(json);
            bytes.extend(data);
            std::fs::write(fixture.dir.join(filename), bytes).unwrap();
        }
        std::fs::write(
            fixture.dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map":weight_map})).unwrap(),
        )
        .unwrap();
        std::fs::write(
            fixture.dir.join("config.json"),
            serde_json::to_vec(&serde_json::json!({
                "model_type":arch, "hidden_size":8, "num_hidden_layers":1,
                "num_attention_heads":2, "num_key_value_heads":1, "head_dim":head_dim,
                "intermediate_size":16, "max_position_embeddings":16,
                "rms_norm_eps":1e-5, "hidden_act":"silu", "tie_word_embeddings":tied,
                "rope_theta":10000., "vocab_size":10,
            }))
            .unwrap(),
        )
        .unwrap();
        fixture
    }
}

impl Drop for HfFixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.dir).unwrap();
    }
}

fn check_hf_parity(backend: Box<dyn TernaryBackend>, fixture: &HfFixture) {
    let mut oracle =
        ModelRunner::from_hf(&fixture.dir, Box::new(tritium_cpu::CpuBackend::new())).unwrap();
    let budget = TensorPayloadBudget::new(fixture.payload_bytes);
    let mut recorder = EvidenceRecorder::new("nn-hf", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
    let mut runner = ModelRunner::from_hf_admitted(
        &fixture.dir,
        backend,
        TensorUploadPolicy {
            native_only: true,
            max_payload_bytes: None,
        },
        &budget,
        TensorUploadContext {
            recorder: &mut recorder,
            span: "load",
            parent: None,
            logical_time: 0,
        },
    )
    .unwrap();
    assert_eq!(budget.used_bytes(), fixture.payload_bytes);
    assert_eq!(recorder.events().len(), fixture.upload_count);
    let verified = verify_jsonl_with_registry(
        &recorder.to_jsonl().unwrap(),
        [TENSOR_UPLOADED_REGISTRATION],
    )
    .unwrap();
    assert_eq!(verified.root_digest, recorder.root_digest().unwrap());
    assert!(runner.weights.token_embd.as_dense().is_none());
    assert!(matches!(
        runner.weights.layers[0].q_proj,
        tritium_nn::Projection::Tensor(_)
    ));
    for (tokens, positions) in [
        (&[1, 3, 2][..], &[0, 1, 2][..]),
        (&[4][..], &[3][..]),
        (&[5][..], &[4][..]),
    ] {
        let expected = oracle.forward(tokens, positions).unwrap();
        let actual = runner.forward(tokens, positions).unwrap();
        assert!(expected.iter().all(|v| v.is_finite()));
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{}",
            runner.backend.device_id()
        );
    }
    drop(runner);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn admitted_hf_cpu_and_reference_match_legacy_logits_bit_for_bit() {
    for arch in ["llama", "qwen2", "qwen3"] {
        for tied in [false, true] {
            for hd in [4, 6] {
                let fixture = HfFixture::new(arch, tied, hd, None);
                check_hf_parity(Box::new(tritium_cpu::CpuBackend::new()), &fixture);
                check_hf_parity(Box::new(tritium_testkit::ReferenceBackend::new()), &fixture);
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn admitted_hf_cuda_matches_legacy_logits_bit_for_bit() {
    if let Err(error) = tritium_cuda::CudaBackend::new(0) {
        assert!(
            std::env::var_os("TRITIUM_REQUIRE_CUDA").is_none(),
            "required CUDA device: {error}"
        );
        eprintln!("UNKNOWN: admitted HF CUDA model execution unavailable: {error}");
        return;
    }
    for arch in ["llama", "qwen2", "qwen3"] {
        for tied in [false, true] {
            let fixture = HfFixture::new(arch, tied, 6, None);
            check_hf_parity(
                Box::new(tritium_cuda::CudaBackend::new(0).unwrap()),
                &fixture,
            );
        }
    }
}

#[test]
fn failed_hf_assembly_releases_all_unpublished_payloads() {
    for (bad, limit, expected_events) in [
        (None, 320, 1),
        (Some("missing-down"), u64::MAX, 3),
        (Some("nan-norm"), u64::MAX, 1),
    ] {
        let fixture = HfFixture::new("llama", true, 4, bad);
        let budget = TensorPayloadBudget::new(limit);
        let mut recorder =
            EvidenceRecorder::new("nn-hf-failure", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
        let result = ModelRunner::from_hf_admitted(
            &fixture.dir,
            Box::new(tritium_cpu::CpuBackend::new()),
            TensorUploadPolicy::default(),
            &budget,
            TensorUploadContext {
                recorder: &mut recorder,
                span: "load",
                parent: None,
                logical_time: 0,
            },
        );
        assert!(result.is_err());
        assert_eq!(budget.used_bytes(), 0);
        // Successful tensor observations are retained, never a whole-model verdict.
        assert_eq!(recorder.events().len(), expected_events);
    }
    let fixture = HfFixture::new("llama", true, 4, None);
    let budget = TensorPayloadBudget::new(fixture.payload_bytes);
    let mut recorder = EvidenceRecorder::new("nn-unregistered", []).unwrap();
    assert!(
        ModelRunner::from_hf_admitted(
            &fixture.dir,
            Box::new(tritium_cpu::CpuBackend::new()),
            TensorUploadPolicy::default(),
            &budget,
            TensorUploadContext {
                recorder: &mut recorder,
                span: "load",
                parent: None,
                logical_time: 0
            }
        )
        .is_err()
    );
    assert_eq!(budget.used_bytes(), 0);
    assert!(recorder.events().is_empty());
}
