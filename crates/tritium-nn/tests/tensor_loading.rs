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

#[path = "tensor_loading/joined.rs"]
mod joined;

#[path = "tensor_loading/calibration.rs"]
mod calibration;

#[path = "tensor_loading/compact.rs"]
mod compact;

#[cfg(feature = "tokenizer")]
#[path = "tensor_loading/stream_limits.rs"]
mod stream_limits;

#[cfg(all(feature = "cuda", feature = "tokenizer"))]
#[path = "tensor_loading/resident.rs"]
mod resident;

#[derive(Debug)]
struct UploadProbe(std::sync::Arc<std::sync::atomic::AtomicUsize>);

impl TernaryBackend for UploadProbe {
    fn device_id(&self) -> &str {
        "cpu-preflight-probe"
    }
    fn capabilities(&self) -> tritium_spec::DeviceCaps {
        tritium_cpu::CpuBackend::new().capabilities()
    }
    fn tensor_caps(
        &self,
        tensor: TensorView<'_>,
    ) -> Result<Option<tritium_spec::TensorCaps>, tritium_spec::BackendError> {
        tritium_cpu::CpuBackend::new().tensor_caps(tensor)
    }
    fn upload_tensor(
        &self,
        tensor: TensorView<'_>,
    ) -> Result<Box<dyn tritium_spec::DeviceBuffer>, tritium_spec::BackendError> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        tritium_cpu::CpuBackend::new().upload_tensor(tensor)
    }
    fn upload_weights(
        &self,
        packed: &[u8],
        shape: tritium_core::GemmShape,
        format: tritium_core::TernaryFormat,
    ) -> Result<Box<dyn tritium_spec::DeviceBuffer>, tritium_spec::BackendError> {
        tritium_cpu::CpuBackend::new().upload_weights(packed, shape, format)
    }
    fn mpgemm(&self, p: tritium_spec::MpGemm<'_>) -> Result<(), tritium_spec::BackendError> {
        tritium_cpu::CpuBackend::new().mpgemm(p)
    }
}

fn upload_probe() -> (
    Box<dyn TernaryBackend>,
    std::sync::Arc<std::sync::atomic::AtomicUsize>,
) {
    let count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    (Box::new(UploadProbe(count.clone())), count)
}

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
    tensors: Vec<(String, Vec<usize>, Vec<f32>)>,
}

impl HfFixture {
    fn new(arch: &str, tied: bool, head_dim: usize, bad: Option<&str>) -> Self {
        Self::with_dimensions(arch, tied, head_dim, 8, 16, bad)
    }

    fn with_dimensions(
        arch: &str,
        tied: bool,
        head_dim: usize,
        width: usize,
        ff: usize,
        bad: Option<&str>,
    ) -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let seq = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("tritium-admitted-hf-{}-{seq}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let mut fixture = Self {
            dir,
            payload_bytes: 0,
            upload_count: 0,
            tensors: Vec::new(),
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
        add("model.embed_tokens.weight", &[10, width]);
        for name in [
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        ] {
            add(name, &[width]);
        }
        for (name, rows, cols) in [
            ("self_attn.q_proj.weight", 2 * head_dim, width),
            ("self_attn.k_proj.weight", head_dim, width),
            ("self_attn.v_proj.weight", head_dim, width),
            ("self_attn.o_proj.weight", width, 2 * head_dim),
            ("mlp.gate_proj.weight", ff, width),
            ("mlp.up_proj.weight", ff, width),
            ("mlp.down_proj.weight", width, ff),
        ] {
            add(&format!("model.layers.0.{name}"), &[rows, cols]);
        }
        if !tied {
            add("lm_head.weight", &[10, width]);
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
        if bad == Some("shape") {
            let (_, shape, _) = tensors
                .iter_mut()
                .find(|(name, _, _)| name.ends_with("q_proj.weight"))
                .unwrap();
            *shape = vec![1, shape.iter().product()];
        }
        if bad == Some("vector-shape") {
            let (_, shape, _) = tensors
                .iter_mut()
                .find(|(name, _, _)| name == "model.norm.weight")
                .unwrap();
            *shape = vec![1, width];
        }
        if bad == Some("extra") {
            tensors.push(("unexpected.weight".into(), vec![1], vec![0.]));
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
                let dtype = if bad == Some("dtype") && name.ends_with("q_proj.weight") {
                    "U32"
                } else {
                    "F32"
                };
                header.insert(name.clone(), serde_json::json!({"dtype":dtype, "shape":shape, "data_offsets":[start,data.len()]}));
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
                "model_type":arch, "hidden_size":width, "num_hidden_layers":1,
                "num_attention_heads":2, "num_key_value_heads":1, "head_dim":head_dim,
                "intermediate_size":ff, "max_position_embeddings":16,
                "rms_norm_eps":1e-5, "hidden_act":"silu", "tie_word_embeddings":tied,
                "rope_theta":10000., "vocab_size":10,
            }))
            .unwrap(),
        )
        .unwrap();
        fixture.tensors = tensors;
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
        (Some("missing-down"), u64::MAX, 0),
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

#[test]
fn hf_complete_inventory_preflight_rejects_before_backend_upload() {
    for arch in ["llama", "qwen2", "qwen3"] {
        for tied in [false, true] {
            for bad in ["missing-down", "shape", "extra", "dtype", "vector-shape"] {
                let fixture = HfFixture::new(arch, tied, 6, Some(bad));
                let budget = TensorPayloadBudget::new(u64::MAX);
                let mut recorder =
                    EvidenceRecorder::new("hf-preflight", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
                let (backend, uploads) = upload_probe();
                let result = ModelRunner::from_hf_admitted(
                    &fixture.dir,
                    backend,
                    TensorUploadPolicy::default(),
                    &budget,
                    TensorUploadContext {
                        recorder: &mut recorder,
                        span: "load",
                        parent: None,
                        logical_time: 0,
                    },
                );
                assert!(result.is_err(), "{arch}/{tied}/{bad}");
                assert!(recorder.events().is_empty());
                assert_eq!(uploads.load(std::sync::atomic::Ordering::SeqCst), 0);
                assert_eq!(budget.used_bytes(), 0);
            }
        }
    }
}

#[test]
fn hf_and_typed_inventory_adapters_reject_the_same_graph_mismatches_before_upload() {
    for arch in ["llama", "qwen2", "qwen3"] {
        for tied in [false, true] {
            for bad in ["missing-down", "shape", "extra", "vector-shape"] {
                let hf = HfFixture::new(arch, tied, 6, Some(bad));
                let fixture = TritFixture::new(&hf, None);
                let bytes = fixture.bytes();
                for source in [
                    tritium_nn::ModelSource::HuggingFace(&hf.dir),
                    tritium_nn::ModelSource::Trit {
                        bytes: &bytes,
                        level: "standard",
                    },
                ] {
                    let budget = TensorPayloadBudget::new(u64::MAX);
                    let mut recorder = EvidenceRecorder::new(
                        "shared-inventory-reject",
                        [TENSOR_UPLOADED_REGISTRATION],
                    )
                    .unwrap();
                    let (backend, uploads) = upload_probe();
                    assert!(
                        ModelRunner::from_source_admitted(
                            source,
                            backend,
                            TensorUploadPolicy::default(),
                            &budget,
                            TensorUploadContext {
                                recorder: &mut recorder,
                                span: "load",
                                parent: None,
                                logical_time: 0,
                            },
                        )
                        .is_err(),
                        "{arch}/{tied}/{bad}"
                    );
                    assert_eq!(uploads.load(std::sync::atomic::Ordering::SeqCst), 0);
                    assert!(recorder.events().is_empty());
                    assert_eq!(budget.used_bytes(), 0);
                }
            }
        }
    }
}

#[test]
fn one_source_loader_keeps_existing_hf_execution_and_budget() {
    let fixture = HfFixture::new("qwen3", true, 6, None);
    let budget = TensorPayloadBudget::new(fixture.payload_bytes);
    let mut recorder = EvidenceRecorder::new("nn-source", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
    let mut runner = ModelRunner::from_source_admitted(
        tritium_nn::ModelSource::HuggingFace(&fixture.dir),
        Box::new(tritium_cpu::CpuBackend::new()),
        TensorUploadPolicy::default(),
        &budget,
        TensorUploadContext {
            recorder: &mut recorder,
            span: "load",
            parent: None,
            logical_time: 0,
        },
    )
    .unwrap();
    let mut oracle =
        ModelRunner::from_hf(&fixture.dir, Box::new(tritium_cpu::CpuBackend::new())).unwrap();
    assert_eq!(
        runner.forward(&[1, 3, 2], &[0, 1, 2]).unwrap(),
        oracle.forward(&[1, 3, 2], &[0, 1, 2]).unwrap()
    );
    assert_eq!(budget.used_bytes(), fixture.payload_bytes);
    drop(runner);
    assert_eq!(budget.used_bytes(), 0);
}

struct TritFixture {
    manifest: tritium_schema::ModelManifest,
    blobs: Vec<(tritium_format::TritBlobKind, Vec<u8>)>,
}

impl TritFixture {
    fn new(hf: &HfFixture, additive: Option<(PlaneCodec, Basis)>) -> Self {
        use tritium_schema::{
            BlobId, BlobReference, Level, LevelFit, ManifestAsset, ModelManifest, PlaneCounts,
            ScalePrecision, SchemaId, TensorMeta, TensorStorage,
        };
        let reference = |bytes: &[u8]| BlobReference {
            id: BlobId::from_bytes(*blake3::hash(bytes).as_bytes()),
            shard: None,
        };
        let config = std::fs::read(hf.dir.join("config.json")).unwrap();
        let architecture =
            serde_json::from_slice::<serde_json::Value>(&config).unwrap()["model_type"]
                .as_str()
                .unwrap()
                .to_owned();
        let mut blobs = vec![
            (
                tritium_format::TritBlobKind::Asset("config.json".into()),
                config,
            ),
            (
                tritium_format::TritBlobKind::Asset("tokenizer.json".into()),
                b"{}".to_vec(),
            ),
        ];
        let assets = blobs
            .iter()
            .map(|(kind, bytes)| {
                let tritium_format::TritBlobKind::Asset(name) = kind else {
                    unreachable!()
                };
                ManifestAsset {
                    name: name.clone(),
                    blob: reference(bytes),
                }
            })
            .collect();
        let mut tensors = Vec::new();
        for (name, shape, values) in &hf.tensors {
            let (kind, storage, bytes) = if let Some((codec, basis)) =
                additive.filter(|_| shape.len() == 2)
            {
                let layout = AdditiveLayout {
                    rows: shape[0] as u64,
                    cols: shape[1] as u64,
                    tile: 256,
                    group: 32,
                    max_planes: 3,
                    allocation: PlaneAllocation::Uniform,
                    codec,
                    law: ADMITTED_LAWS[1].law,
                    basis,
                    transport: Transport::Raw,
                };
                let trits = (0..3)
                    .flat_map(|p| {
                        (0..values.len()).map(move |i| {
                            if i % 4 == 1 {
                                Trit::ZERO
                            } else if (i / shape[1] + p + i) % 2 == 0 {
                                Trit::POS
                            } else {
                                Trit::NEG
                            }
                        })
                    })
                    .collect();
                let scales = (0..3)
                    .flat_map(|p| {
                        (0..shape[0]).flat_map(move |r| {
                            (0..shape[1].div_ceil(layout.group as usize))
                                .map(move |_| (1 + r % 2) as f32 * 0.0625 * 0.5f32.powi(p))
                        })
                    })
                    .collect();
                let tensor = tritium_format::AdditiveTensor::new(layout, 3, trits, scales).unwrap();
                (
                    tritium_format::TritBlobKind::AdditiveTensor,
                    TensorStorage::Additive {
                        layout,
                        planes: PlaneCounts::Uniform(3),
                    },
                    tensor.encode_payload().unwrap(),
                )
            } else {
                (
                    tritium_format::TritBlobKind::DenseTensor(tritium_core::DType::F32),
                    TensorStorage::Dense {
                        shape: shape.iter().map(|&n| n as u64).collect(),
                        precision: ScalePrecision::F32,
                    },
                    values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                )
            };
            tensors.push(TensorMeta {
                name: name.clone(),
                blob: reference(&bytes),
                storage,
            });
            blobs.push((kind, bytes));
        }
        Self {
            manifest: ModelManifest {
                schema: SchemaId { major: 1, minor: 0 },
                architecture,
                assets,
                levels: vec![Level {
                    name: "standard".into(),
                    fit: LevelFit::Imported,
                    tensors,
                }],
                shards: vec![],
                legacy_ids: vec![],
            },
            blobs,
        }
    }

    fn bytes(&self) -> Vec<u8> {
        let blobs = self
            .blobs
            .iter()
            .map(|(kind, bytes)| tritium_format::TritBlob {
                kind: kind.clone(),
                bytes,
            })
            .collect::<Vec<_>>();
        tritium_format::write_trit_package(
            &tritium_format::trit_model::encode_model_manifest(&self.manifest).unwrap(),
            &blobs,
        )
        .unwrap()
    }

    fn dense_oracle(&self) -> ModelRunner {
        // Build the numerical oracle with the unchanged legacy HF loader. Additive
        // fixtures materialize original-coordinate values through the core gather,
        // not a second attention/normalization implementation in this test.
        let hf = HfFixture::new(
            &self.manifest.architecture,
            self.manifest.levels[0]
                .tensors
                .iter()
                .all(|t| t.name != "lm_head.weight"),
            if self.manifest.architecture == "qwen3" {
                6
            } else {
                4
            },
            None,
        );
        let bytes = self.bytes();
        let model = tritium_format::trit_model::read_trit_model(&bytes).unwrap();
        let mut data = Vec::new();
        let mut header = serde_json::Map::new();
        for tensor in &self.manifest.levels[0].tensors {
            let (_, payload) = model.tensor("standard", &tensor.name).unwrap();
            let (shape, values) = match &tensor.storage {
                tritium_schema::TensorStorage::Dense { shape, precision } => {
                    let values = match precision {
                        tritium_schema::ScalePrecision::F32 => payload
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|b| f32::from_le_bytes(*b))
                            .collect::<Vec<_>>(),
                        tritium_schema::ScalePrecision::F16 => payload
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|b| half::f16::from_bits(u16::from_le_bytes(*b)).to_f32())
                            .collect(),
                        tritium_schema::ScalePrecision::Bf16 => payload
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|b| half::bf16::from_bits(u16::from_le_bytes(*b)).to_f32())
                            .collect(),
                        _ => unreachable!(),
                    };
                    (shape.clone(), values)
                }
                tritium_schema::TensorStorage::Additive { layout, .. } => {
                    let tensor =
                        tritium_format::trit_model::decode_additive_payload(tensor, payload)
                            .unwrap();
                    let ids = (0..layout.rows as usize).collect::<Vec<_>>();
                    let mut values = vec![0.; (layout.rows * layout.cols) as usize];
                    tritium_core::reference_embed(&tensor.view().bind_gather(), &ids, &mut values)
                        .unwrap();
                    (vec![layout.rows, layout.cols], values)
                }
                _ => unreachable!(),
            };
            let start = data.len();
            data.extend(values.iter().flat_map(|v| v.to_le_bytes()));
            header.insert(tensor.name.clone(), serde_json::json!({"dtype":"F32", "shape":shape, "data_offsets":[start,data.len()]}));
        }
        let mut json = serde_json::to_vec(&header).unwrap();
        while !json.len().is_multiple_of(8) {
            json.push(b' ');
        }
        let mut packed = (json.len() as u64).to_le_bytes().to_vec();
        packed.extend(json);
        packed.extend(data);
        std::fs::write(hf.dir.join("weights.safetensors"), packed).unwrap();
        let map = self.manifest.levels[0]
            .tensors
            .iter()
            .map(|t| {
                (
                    t.name.clone(),
                    serde_json::Value::String("weights.safetensors".into()),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        std::fs::write(
            hf.dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map":map})).unwrap(),
        )
        .unwrap();
        let (_, config) = self
            .blobs
            .iter()
            .find(|(kind, _)| *kind == tritium_format::TritBlobKind::Asset("config.json".into()))
            .unwrap();
        std::fs::write(hf.dir.join("config.json"), config).unwrap();
        ModelRunner::from_hf(&hf.dir, Box::new(tritium_cpu::CpuBackend::new())).unwrap()
    }

    fn dense_precision(&mut self, precision: tritium_schema::ScalePrecision) {
        use tritium_schema::{BlobId, ScalePrecision, TensorStorage};
        for tensor in &mut self.manifest.levels[0].tensors {
            let TensorStorage::Dense {
                precision: stored_precision,
                ..
            } = &mut tensor.storage
            else {
                continue;
            };
            assert_eq!(*stored_precision, ScalePrecision::F32);
            let original = self
                .blobs
                .iter()
                .find(|(_, b)| blake3::hash(b).as_bytes() == tensor.blob.id.as_bytes())
                .unwrap();
            let values = original
                .1
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b));
            let (dtype, bytes) = match precision {
                ScalePrecision::F16 => (
                    tritium_core::DType::F16,
                    values
                        .flat_map(|v| half::f16::from_f32(v).to_bits().to_le_bytes())
                        .collect::<Vec<_>>(),
                ),
                ScalePrecision::Bf16 => (
                    tritium_core::DType::BF16,
                    values
                        .flat_map(|v| half::bf16::from_f32(v).to_bits().to_le_bytes())
                        .collect(),
                ),
                _ => unreachable!(),
            };
            *stored_precision = precision;
            tensor.blob.id = BlobId::from_bytes(*blake3::hash(&bytes).as_bytes());
            self.blobs
                .push((tritium_format::TritBlobKind::DenseTensor(dtype), bytes));
        }
    }

    fn asset(&mut self, name: &str, bytes: Vec<u8>) {
        use tritium_schema::{BlobId, BlobReference, ManifestAsset};
        let kind = tritium_format::TritBlobKind::Asset(name.into());
        self.blobs.retain(|(k, _)| *k != kind);
        self.manifest.assets.retain(|a| a.name != name);
        self.manifest.assets.push(ManifestAsset {
            name: name.into(),
            blob: BlobReference {
                id: BlobId::from_bytes(*blake3::hash(&bytes).as_bytes()),
                shard: None,
            },
        });
        self.blobs.push((kind, bytes));
    }
}

fn check_trit_parity(
    fixture: &TritFixture,
    backend: Box<dyn TernaryBackend>,
    native_only: bool,
    exact: bool,
) {
    let bytes = fixture.bytes();
    let mut oracle = fixture.dense_oracle();
    let budget = TensorPayloadBudget::new(100_000);
    let mut recorder = EvidenceRecorder::new("nn-trit", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
    let mut runner = ModelRunner::from_source_admitted(
        tritium_nn::ModelSource::Trit {
            bytes: &bytes,
            level: "standard",
        },
        backend,
        TensorUploadPolicy {
            native_only,
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
    assert!(budget.used_bytes() > 0);
    assert!(runner.weights.token_embd.as_dense().is_none());
    assert_eq!(recorder.events().len(), fixture.manifest.levels[0].tensors.iter().filter(|t| !matches!(&t.storage, tritium_schema::TensorStorage::Dense { shape, .. } if shape.len() == 1)).count());
    for event in recorder.events() {
        // These CPU/reference/CUDA fixtures are admitted native combinations.
        // Permitting emulation does not require a native adapter to emulate.
        assert_eq!(event.payload["caps"]["execution"], "Native");
    }
    for (tokens, positions) in [
        (&[1, 3, 2][..], &[0, 1, 2][..]),
        (&[4][..], &[3][..]),
        (&[5][..], &[4][..]),
    ] {
        let expected = oracle.forward(tokens, positions).unwrap();
        let actual = runner.forward(tokens, positions).unwrap();
        if exact {
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        } else {
            for (a, e) in actual.iter().zip(&expected) {
                assert!(
                    a.is_finite() && e.is_finite() && (a - e).abs() <= 1e-5 * e.abs().max(1.),
                    "{a} != {e}"
                );
            }
        }
    }
    verify_jsonl_with_registry(
        &recorder.to_jsonl().unwrap(),
        [TENSOR_UPLOADED_REGISTRATION],
    )
    .unwrap();
    drop(runner);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn typed_dense_and_additive_packages_execute_prefill_and_cached_decode() {
    for arch in ["llama", "qwen2", "qwen3"] {
        for tied in [false, true] {
            let hf = HfFixture::new(arch, tied, if arch == "qwen3" { 6 } else { 4 }, None);
            let fixture = TritFixture::new(&hf, None);
            check_trit_parity(
                &fixture,
                Box::new(tritium_cpu::CpuBackend::new()),
                true,
                true,
            );
            check_trit_parity(
                &fixture,
                Box::new(tritium_testkit::ReferenceBackend::new()),
                true,
                true,
            );
        }
    }
    for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
        for basis in [
            Basis::Identity,
            Basis::Hadamard { block: 4 },
            Basis::SignedRht {
                block: 4,
                seed: 7,
                domain: 9,
            },
        ] {
            for tied in [false, true] {
                let hf = HfFixture::new("qwen3", tied, 6, None);
                let fixture = TritFixture::new(&hf, Some((codec, basis)));
                check_trit_parity(
                    &fixture,
                    Box::new(tritium_cpu::CpuBackend::new()),
                    true,
                    false,
                );
                check_trit_parity(
                    &fixture,
                    Box::new(tritium_testkit::ReferenceBackend::new()),
                    true,
                    false,
                );
            }
        }
    }
}

#[test]
fn typed_f16_and_bf16_sources_match_lossless_f32_decode() {
    for precision in [
        tritium_schema::ScalePrecision::F16,
        tritium_schema::ScalePrecision::Bf16,
    ] {
        let hf = HfFixture::new("qwen3", true, 6, None);
        let mut fixture = TritFixture::new(&hf, None);
        fixture.dense_precision(precision);
        check_trit_parity(
            &fixture,
            Box::new(tritium_cpu::CpuBackend::new()),
            true,
            true,
        );
        check_trit_parity(
            &fixture,
            Box::new(tritium_testkit::ReferenceBackend::new()),
            true,
            true,
        );
    }
}

#[cfg(feature = "tokenizer")]
fn tokenizer_assets(out_of_range: bool) -> (Vec<u8>, Vec<u8>) {
    let mut vocab = [
        "[UNK]", "hello", "partner", "[EOS]", "[BOS]", "t5", "t6", "t7", "t8", "t9",
    ]
    .into_iter()
    .enumerate()
    .map(|(id, word)| (word.to_owned(), id as u32))
    .collect::<Vec<_>>();
    if out_of_range {
        // Even an unused/added vocabulary ID must be admitted against embeddings.
        vocab.push(("not-in-model".into(), 10));
    }
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(vocab.into_iter().collect())
        .unk_token("[UNK]".to_owned())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
    (
        tokenizer.to_string(false).unwrap().into_bytes(),
        br#"{"bos_token":"[BOS]","eos_token":{"content":"[EOS]"}}"#.to_vec(),
    )
}

#[cfg(feature = "tokenizer")]
fn check_text_load(
    source: tritium_nn::ModelSource<'_>,
    mut oracle: ModelRunner,
    backend: Box<dyn TernaryBackend>,
) {
    use tritium_evidence::{MODEL_LOADED_REGISTRATION, ModelLoadReceiptView};
    use tritium_nn::Tokenizer;
    let expected_source = match source {
        tritium_nn::ModelSource::TritFile {
            path,
            level,
            receiving,
            ..
        } => {
            let model = tritium_format::trit_model::TritModelReader::open(
                std::fs::File::open(path).unwrap(),
                receiving,
            )
            .unwrap();
            tritium_schema::LoadedSourceIdentity::Trit {
                package_id: tritium_schema::PackageId::from_bytes(*model.package_id().as_bytes()),
                model_id: model.model_id(level).unwrap(),
                level: level.into(),
            }
        }
        tritium_nn::ModelSource::Trit { bytes, level } => {
            let model = tritium_format::trit_model::read_trit_model(bytes).unwrap();
            tritium_schema::LoadedSourceIdentity::Trit {
                package_id: tritium_schema::PackageId::from_bytes(*model.package_id().as_bytes()),
                model_id: model.model_id(level).unwrap(),
                level: level.into(),
            }
        }
        tritium_nn::ModelSource::HuggingFace(_) => {
            tritium_schema::LoadedSourceIdentity::HuggingFace {
                reason: tritium_schema::UnknownReason::MissingCapability,
            }
        }
        tritium_nn::ModelSource::JoinedSalience { model, level } => {
            tritium_schema::LoadedSourceIdentity::Trit {
                package_id: tritium_schema::PackageId::from_bytes(
                    *model.model().package_id().as_bytes(),
                ),
                model_id: model.model().model_id(level).unwrap(),
                level: level.into(),
            }
        }
        _ => panic!("unexpected test source"),
    };
    let expected_backend = backend.device_id().to_owned();
    if let tritium_nn::ModelSource::TritFile {
        path,
        level,
        receiving,
        limits,
    } = source
    {
        let plan = tritium_nn::inspect_trit_file(path, level, receiving, limits, None).unwrap();
        assert_eq!(plan.source(), &expected_source);
        assert_eq!(plan.architecture().name(), "qwen3");
        assert_eq!(plan.architecture().vocab_size(), 10);
        assert!(plan.backend_payload_bytes().is_none());
        assert!(plan.cache_admission().is_none());
    }
    let expected_device = backend.physical_device_id().to_owned();
    let budget = TensorPayloadBudget::new(100_000);
    let mut recorder = EvidenceRecorder::new(
        "nn-text-load",
        [TENSOR_UPLOADED_REGISTRATION, MODEL_LOADED_REGISTRATION],
    )
    .unwrap();
    let mut loaded = tritium_nn::load(
        source,
        backend,
        TensorUploadPolicy::default(),
        &budget,
        TensorUploadContext {
            recorder: &mut recorder,
            span: "load",
            parent: None,
            logical_time: 0,
        },
    )
    .unwrap();
    assert_eq!(loaded.tokenizer.encode("hello partner").unwrap(), [1, 2]);
    assert_eq!(loaded.tokenizer.decode(&[1, 2]).unwrap(), "hello partner");
    assert_eq!(loaded.tokenizer.bos(), 4);
    assert_eq!(loaded.tokenizer.eos(), 3);
    assert_eq!(loaded.runner.architecture().name(), "qwen3");
    assert_eq!(loaded.receipt.observation().source, expected_source);
    assert_eq!(loaded.receipt.observation().backend, expected_backend);
    assert_eq!(
        loaded.receipt.observation().physical_device,
        expected_device
    );
    assert_eq!(
        loaded.receipt.observation().architecture,
        loaded.runner.architecture().name()
    );
    assert_eq!(loaded.receipt.observation().vocab_size, 10);
    assert_eq!(
        loaded.receipt.observation().layer_count,
        loaded.runner.architecture().layer_count() as u64
    );
    assert_eq!(Some(loaded.receipt.envelope()), recorder.events().last());
    let prompt = loaded.tokenizer.encode("hello partner").unwrap();
    let close = |actual: &[f32], expected: &[f32]| {
        assert_eq!(actual.len(), expected.len());
        for (&a, &e) in actual.iter().zip(expected) {
            assert!(a.is_finite() && e.is_finite() && (a - e).abs() <= 1e-5 * e.abs().max(1.));
        }
    };
    assert!(loaded.runner.step(1).is_err());
    let actual = loaded.runner.prefill(&prompt).unwrap();
    oracle.reset();
    close(&actual, &oracle.forward(&prompt, &[0, 1]).unwrap());
    assert_eq!(loaded.runner.cache_len(), 2);
    assert!(loaded.runner.prefill(&prompt).is_err());
    assert!(loaded.runner.forward(&[1], &[0]).is_err());
    assert!(loaded.runner.step(10).is_err());
    assert!(
        loaded
            .runner
            .forward_cancellable(&[], &[], &|| true)
            .unwrap()
            .is_none()
    );
    assert_eq!(loaded.runner.cache_len(), 2);
    close(
        &loaded.runner.step(1).unwrap(),
        &oracle.forward(&[1], &[2]).unwrap(),
    );
    assert_eq!(
        loaded
            .runner
            .generate(&prompt, 3, loaded.tokenizer.eos())
            .unwrap(),
        oracle.generate(&prompt, 3, 3).unwrap()
    );
    assert!(budget.used_bytes() > 0);
    let verified = verify_jsonl_with_registry(
        &recorder.to_jsonl().unwrap(),
        [TENSOR_UPLOADED_REGISTRATION, MODEL_LOADED_REGISTRATION],
    )
    .unwrap();
    assert_eq!(
        ModelLoadReceiptView::fold(&verified).unwrap().as_slice(),
        std::slice::from_ref(&loaded.receipt)
    );
    drop(loaded);
    assert_eq!(budget.used_bytes(), 0);
}

#[cfg(feature = "tokenizer")]
#[test]
fn engine_produced_ternary_model_executes_through_actual_text_loader() {
    use tritium_evidence::{
        CALIBRATED_LEVELS_FITTED_REGISTRATION, CALIBRATED_QUANTIZATION_COMPLETED_REGISTRATION,
        CALIBRATED_QUANTIZATION_STARTED_REGISTRATION, DIRECT_LEVELS_FITTED_REGISTRATION,
        QUANTIZATION_COMPLETED_REGISTRATION, QUANTIZATION_STARTED_REGISTRATION,
    };
    use tritium_quantize::engine::{
        CalibrationBatch, CalibrationCacheBuilder, CalibrationCollector, CalibrationInput,
        CalibrationOutputBatch, Engine, InMemoryCalibrationSource, MemoryTensorSink,
        SafeTensorsWeightSource, WeightSource,
    };
    use tritium_schema::{
        CalibratedFitPlan, CalibratedQuantAction, CalibratedQuantPlan, CalibratedQuantTensorPlan,
        CalibrationCacheLimits, CalibrationCacheRequest, CalibrationConsumer, CalibrationOutput,
        CalibrationPartition, DirectFitLevel, DirectFitLimits, DirectFitMetric, DirectFitPlan,
        DirectFitProfile, JointGroupFitOptions, QuantAction, QuantLimits, QuantPlan,
        QuantTensorPlan,
    };
    for calibrated in [false, true] {
        let fixture = HfFixture::new("qwen3", true, 6, None);
        let (tokenizer, tokenizer_config) = tokenizer_assets(false);
        let assets = std::collections::BTreeMap::from([
            (
                "config.json".into(),
                std::fs::read(fixture.dir.join("config.json")).unwrap(),
            ),
            ("tokenizer.json".into(), tokenizer),
            ("tokenizer_config.json".into(), tokenizer_config),
        ]);
        let readers = ["weights.safetensors", "vectors.safetensors"]
            .map(|name| std::fs::File::open(fixture.dir.join(name)).unwrap());
        let mut source =
            SafeTensorsWeightSource::new(readers, assets, std::collections::BTreeMap::new())
                .unwrap();
        let plan = QuantPlan {
            architecture: "qwen3".into(),
            assets: source.asset_names(),
            levels: vec!["full".into()],
            limits: QuantLimits {
                max_source_bytes: 100_000,
                max_asset_bytes: 100_000,
                max_retained_blob_bytes: 100_000,
                max_package_bytes: 1_000_000,
            },
            tensors: source
                .inventory()
                .unwrap()
                .into_iter()
                .map(|source| {
                    let action = if source.shape.len() == 2 {
                        QuantAction::Fit {
                            metric: DirectFitMetric::Identity,
                            plan: Box::new(DirectFitPlan {
                                layout: AdditiveLayout {
                                    rows: source.shape[0],
                                    cols: source.shape[1],
                                    tile: 256,
                                    group: 32,
                                    max_planes: 3,
                                    law: ADMITTED_LAWS[1].law,
                                    allocation: PlaneAllocation::Uniform,
                                    codec: PlaneCodec::B3,
                                    basis: Basis::Identity,
                                    transport: Transport::Raw,
                                },
                                profile: DirectFitProfile::JointIndependentGroups(
                                    JointGroupFitOptions {
                                        max_iterations: 8,
                                        ridge: 1e-8,
                                        em_restarts: 2,
                                        ridge_condition_limit: 1e6,
                                        softened_relay: false,
                                        modulated_relay: false,
                                    },
                                ),
                                levels: vec![DirectFitLevel {
                                    name: "full".into(),
                                    planes: 3,
                                }],
                                limits: DirectFitLimits {
                                    max_weights: 100_000,
                                    max_metric_entries: 10_000,
                                    max_output_trits: 300_000,
                                },
                            }),
                        }
                    } else {
                        QuantAction::KeepDense
                    };
                    QuantTensorPlan { source, action }
                })
                .collect(),
        };
        let mut sink = MemoryTensorSink::default();
        let fitted = if calibrated {
            let original_source = Engine::observe_source(&plan, &mut source).unwrap();
            let selected = plan
                .tensors
                .iter()
                .find(|t| t.source.name.contains("q_proj.weight"))
                .unwrap();
            let rows = selected.source.shape[0] as usize;
            let cols = selected.source.shape[1] as usize;
            let mut input = vec![0.; cols * cols];
            for i in 0..cols {
                input[i * cols + i] = 1.;
            }
            let tokenizer = source
                .asset("tokenizer.json", 100_000)
                .unwrap()
                .into_owned();
            let mut builder = CalibrationCacheBuilder::new(
                CalibrationCacheRequest {
                    tensor: selected.source.clone(),
                    model: original_source,
                    consumer: CalibrationConsumer::Matmul,
                    output: CalibrationOutput::DenseFisher,
                    sample_start: 0,
                    limits: CalibrationCacheLimits {
                        max_payload_bytes: 100_000,
                        max_samples: 100,
                        max_windows: 4,
                    },
                },
                &tokenizer,
                b"declared NN fixture objective",
            )
            .unwrap();
            builder
                .capture(
                    &(0..cols as u32).collect::<Vec<_>>(),
                    CalibrationBatch {
                        samples: cols,
                        input: CalibrationInput::Activations(&input),
                        output: CalibrationOutputBatch::Dense(&vec![0.5; cols * rows]),
                        weights: None,
                        mask: None,
                    },
                )
                .unwrap();
            let cache = builder.finish().unwrap();
            let QuantAction::Fit { plan: direct, .. } = &selected.action else {
                unreachable!()
            };
            let mut fit = CalibratedFitPlan {
                direct: (**direct).clone(),
                cache: cache.binding().clone(),
                partition: CalibrationPartition::Groups { columns: 4 },
                damping: 0.01,
                max_numeric_bytes: 1,
            };
            fit.max_numeric_bytes =
                CalibrationCollector::required_numeric_bytes(&fit.calibration_plan().unwrap())
                    .unwrap()
                    + cache.binding().payload_bytes;
            let request = CalibratedQuantPlan {
                architecture: plan.architecture.clone(),
                original_source,
                assets: plan.assets.clone(),
                levels: plan.levels.clone(),
                limits: plan.limits,
                max_cache_payload_bytes: 100_000,
                max_cache_package_bytes: 100_000,
                tensors: plan
                    .tensors
                    .iter()
                    .map(|t| CalibratedQuantTensorPlan {
                        source: t.source.clone(),
                        action: if t.source == selected.source {
                            CalibratedQuantAction::CalibratedFit {
                                plan: Box::new(fit.clone()),
                            }
                        } else {
                            match &t.action {
                                QuantAction::KeepDense => CalibratedQuantAction::KeepDense,
                                QuantAction::Fit { plan, metric } => CalibratedQuantAction::Fit {
                                    plan: plan.clone(),
                                    metric: metric.clone(),
                                },
                                _ => unreachable!(),
                            }
                        },
                    })
                    .collect(),
            };
            Engine::run_calibrated(
                &request,
                &mut source,
                &mut InMemoryCalibrationSource::new([cache]).unwrap(),
                &mut sink,
                EvidenceRecorder::new(
                    "nn-model-fit",
                    [
                        DIRECT_LEVELS_FITTED_REGISTRATION,
                        CALIBRATED_LEVELS_FITTED_REGISTRATION,
                        CALIBRATED_QUANTIZATION_STARTED_REGISTRATION,
                        CALIBRATED_QUANTIZATION_COMPLETED_REGISTRATION,
                    ],
                )
                .unwrap(),
            )
        } else {
            Engine::run(
                &plan,
                &mut source,
                &mut sink,
                EvidenceRecorder::new(
                    "nn-model-fit",
                    [
                        DIRECT_LEVELS_FITTED_REGISTRATION,
                        QUANTIZATION_STARTED_REGISTRATION,
                        QUANTIZATION_COMPLETED_REGISTRATION,
                    ],
                )
                .unwrap(),
            )
        }
        .unwrap();
        let bytes = sink.artifact().unwrap().0;
        tritium_quantize::engine::verify_package_with_fit_log(bytes, fitted.evidence.log())
            .unwrap();
        let mut recorder =
            EvidenceRecorder::new("nn-engine-oracle", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
        let oracle = ModelRunner::from_source_admitted(
            tritium_nn::ModelSource::Trit {
                bytes,
                level: "full",
            },
            Box::new(tritium_testkit::ReferenceBackend::new()),
            TensorUploadPolicy::default(),
            &TensorPayloadBudget::new(1_000_000),
            TensorUploadContext {
                recorder: &mut recorder,
                span: "load",
                parent: None,
                logical_time: 0,
            },
        )
        .unwrap();
        check_text_load(
            tritium_nn::ModelSource::Trit {
                bytes,
                level: "full",
            },
            oracle,
            Box::new(tritium_cpu::CpuBackend::new()),
        );
    }
}

#[cfg(feature = "tokenizer")]
#[test]
fn text_source_loads_real_tokenizers_and_generates_with_shared_runner() {
    let hf = HfFixture::new("qwen3", true, 6, None);
    let (tokenizer, config) = tokenizer_assets(false);
    std::fs::write(hf.dir.join("tokenizer.json"), &tokenizer).unwrap();
    std::fs::write(hf.dir.join("tokenizer_config.json"), &config).unwrap();
    check_text_load(
        tritium_nn::ModelSource::HuggingFace(&hf.dir),
        ModelRunner::from_hf(&hf.dir, Box::new(tritium_cpu::CpuBackend::new())).unwrap(),
        Box::new(tritium_cpu::CpuBackend::new()),
    );
    for additive in [
        None,
        Some((
            PlaneCodec::B3,
            Basis::SignedRht {
                block: 4,
                seed: 7,
                domain: 9,
            },
        )),
    ] {
        let mut fixture = TritFixture::new(&hf, additive);
        fixture.asset("tokenizer.json", tokenizer.clone());
        fixture.asset("tokenizer_config.json", config.clone());
        let bytes = fixture.bytes();
        let path = hf.dir.join("model.trit");
        std::fs::write(&path, &bytes).unwrap();
        for cpu in [true, false] {
            let backend: Box<dyn TernaryBackend> = if cpu {
                Box::new(tritium_cpu::CpuBackend::new())
            } else {
                Box::new(tritium_testkit::ReferenceBackend::new())
            };
            check_text_load(
                tritium_nn::ModelSource::TritFile {
                    path: &path,
                    level: "standard",
                    receiving: tritium_format::TritPackageReadLimits {
                        max_source_bytes: 1_000_000,
                        max_manifest_bytes: 128_000,
                        max_directory_bytes: 128_000,
                    },
                    limits: tritium_nn::Qwen35TritReadLimits {
                        max_config_bytes: 16_000,
                        max_tensor_bytes: 64_000,
                        max_decoded_tensor_bytes: 64_000,
                    },
                },
                fixture.dense_oracle(),
                backend,
            );
        }
        check_text_load(
            tritium_nn::ModelSource::Trit {
                bytes: &bytes,
                level: "standard",
            },
            fixture.dense_oracle(),
            Box::new(tritium_cpu::CpuBackend::new()),
        );
        check_text_load(
            tritium_nn::ModelSource::Trit {
                bytes: &bytes,
                level: "standard",
            },
            fixture.dense_oracle(),
            Box::new(tritium_testkit::ReferenceBackend::new()),
        );
        #[cfg(feature = "cuda")]
        if let Ok(backend) = tritium_cuda::CudaBackend::new(0) {
            check_text_load(
                tritium_nn::ModelSource::Trit {
                    bytes: &bytes,
                    level: "standard",
                },
                fixture.dense_oracle(),
                Box::new(backend),
            );
        } else {
            assert!(
                std::env::var_os("TRITIUM_REQUIRE_CUDA").is_none(),
                "required text-load CUDA device unavailable"
            );
            eprintln!("UNKNOWN text-load CUDA device unavailable");
        }
    }
}

#[cfg(feature = "tokenizer")]
#[test]
fn text_load_rejects_invalid_assets_and_vocab_and_releases_budget() {
    use tritium_evidence::{EvidenceEvent, MODEL_LOADED_REGISTRATION, ModelLoaded};
    for bad in [
        "missing-config",
        "invalid-json",
        "missing-eos",
        "vocab",
        "large-config",
    ] {
        let hf = HfFixture::new("qwen3", true, 6, None);
        let (mut tokenizer, mut config) = tokenizer_assets(bad == "vocab");
        if bad == "invalid-json" {
            tokenizer = b"{}".to_vec();
        }
        if bad == "missing-eos" {
            config = br#"{"eos_token":"not-in-model"}"#.to_vec();
        }
        if bad == "large-config" {
            config.resize(16 * 1024 * 1024 + 1, b' ');
        }
        std::fs::write(hf.dir.join("tokenizer.json"), &tokenizer).unwrap();
        if bad != "missing-config" {
            std::fs::write(hf.dir.join("tokenizer_config.json"), &config).unwrap();
        }
        let mut fixture = TritFixture::new(&hf, None);
        fixture.asset("tokenizer.json", tokenizer);
        if bad != "missing-config" {
            fixture.asset("tokenizer_config.json", config);
        }
        let bytes = fixture.bytes();
        for source in [
            tritium_nn::ModelSource::HuggingFace(&hf.dir),
            tritium_nn::ModelSource::Trit {
                bytes: &bytes,
                level: "standard",
            },
        ] {
            let budget = TensorPayloadBudget::new(100_000);
            let mut recorder = EvidenceRecorder::new(
                "nn-text-failure",
                [TENSOR_UPLOADED_REGISTRATION, MODEL_LOADED_REGISTRATION],
            )
            .unwrap();
            let (backend, uploads) = upload_probe();
            let result = tritium_nn::load(
                source,
                backend,
                TensorUploadPolicy::default(),
                &budget,
                TensorUploadContext {
                    recorder: &mut recorder,
                    span: "load",
                    parent: None,
                    logical_time: 0,
                },
            );
            assert!(result.is_err(), "{bad}");
            assert_eq!(budget.used_bytes(), 0, "{bad}");
            assert_eq!(
                uploads.load(std::sync::atomic::Ordering::SeqCst),
                0,
                "{bad}"
            );
            assert!(
                recorder
                    .events()
                    .iter()
                    .all(|event| event.schema != ModelLoaded::SCHEMA_ID)
            );
            assert!(
                recorder.events().is_empty(),
                "{bad}: upload before preflight"
            );
        }
    }
}

#[cfg(feature = "tokenizer")]
#[test]
fn load_observations_bind_selected_level_and_logical_ids_across_codecs() {
    use tritium_evidence::{MODEL_LOADED_REGISTRATION, ModelLoadReceiptView};
    use tritium_schema::{BlobId, LoadedSourceIdentity};
    let hf = HfFixture::new("qwen3", true, 6, None);
    let (tokenizer, config) = tokenizer_assets(false);
    let mut baseline = None;
    let mut packages = Vec::new();
    for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
        let mut fixture = TritFixture::new(&hf, Some((codec, Basis::Identity)));
        fixture.asset("tokenizer.json", tokenizer.clone());
        fixture.asset("tokenizer_config.json", config.clone());
        let mut alias = fixture.manifest.levels[0].clone();
        alias.name = "alias".into();
        let mut changed = alias.clone();
        changed.name = "changed".into();
        let norm = changed
            .tensors
            .iter_mut()
            .find(|t| t.name == "model.norm.weight")
            .unwrap();
        let mut bytes = fixture
            .blobs
            .iter()
            .find(|(_, bytes)| blake3::hash(bytes).as_bytes() == norm.blob.id.as_bytes())
            .unwrap()
            .1
            .clone();
        bytes[..4].copy_from_slice(&0.5f32.to_le_bytes());
        norm.blob.id = BlobId::from_bytes(*blake3::hash(&bytes).as_bytes());
        fixture.blobs.push((
            tritium_format::TritBlobKind::DenseTensor(tritium_core::DType::F32),
            bytes,
        ));
        fixture.manifest.levels.extend([alias, changed]);
        let bytes = fixture.bytes();
        let checked = tritium_format::trit_model::read_trit_model(&bytes).unwrap();
        let ids = [
            checked.model_id("standard").unwrap(),
            checked.model_id("alias").unwrap(),
            checked.model_id("changed").unwrap(),
        ];
        assert_eq!(ids[0], ids[1]);
        assert_ne!(ids[0], ids[2]);
        if let Some(baseline) = baseline {
            assert_eq!(ids, baseline);
        } else {
            baseline = Some(ids);
        }
        packages.push(checked.package_id());
        for (level, expected) in ["standard", "alias", "changed"].into_iter().zip(ids) {
            let budget = TensorPayloadBudget::new(100_000);
            let mut recorder = EvidenceRecorder::new(
                "level-load",
                [TENSOR_UPLOADED_REGISTRATION, MODEL_LOADED_REGISTRATION],
            )
            .unwrap();
            let loaded = tritium_nn::load(
                tritium_nn::ModelSource::Trit {
                    bytes: &bytes,
                    level,
                },
                Box::new(tritium_cpu::CpuBackend::new()),
                TensorUploadPolicy::default(),
                &budget,
                TensorUploadContext {
                    recorder: &mut recorder,
                    span: "load",
                    parent: None,
                    logical_time: 0,
                },
            )
            .unwrap();
            assert_eq!(
                loaded.receipt.observation().source,
                LoadedSourceIdentity::Trit {
                    package_id: tritium_schema::PackageId::from_bytes(
                        *checked.package_id().as_bytes()
                    ),
                    model_id: expected,
                    level: level.into(),
                }
            );
            let verified = verify_jsonl_with_registry(
                &recorder.to_jsonl().unwrap(),
                [TENSOR_UPLOADED_REGISTRATION, MODEL_LOADED_REGISTRATION],
            )
            .unwrap();
            assert_eq!(
                ModelLoadReceiptView::fold(&verified).unwrap().as_slice(),
                std::slice::from_ref(&loaded.receipt)
            );
            drop(loaded);
            assert_eq!(budget.used_bytes(), 0);
        }
    }
    assert!(
        packages
            .iter()
            .enumerate()
            .all(|(i, id)| !packages[..i].contains(id))
    );
}

#[cfg(feature = "tokenizer")]
#[test]
fn missing_load_registration_rejects_before_backend_upload() {
    use tritium_evidence::{EvidenceEvent, ModelLoaded};
    let hf = HfFixture::new("qwen3", true, 6, None);
    let (tokenizer, config) = tokenizer_assets(false);
    std::fs::write(hf.dir.join("tokenizer.json"), &tokenizer).unwrap();
    std::fs::write(hf.dir.join("tokenizer_config.json"), &config).unwrap();
    let mut fixture = TritFixture::new(&hf, None);
    fixture.asset("tokenizer.json", tokenizer);
    fixture.asset("tokenizer_config.json", config);
    let bytes = fixture.bytes();
    for source in [
        tritium_nn::ModelSource::HuggingFace(&hf.dir),
        tritium_nn::ModelSource::Trit {
            bytes: &bytes,
            level: "standard",
        },
    ] {
        let budget = TensorPayloadBudget::new(100_000);
        // Missing registration is a cheap preflight rejection, not a late failure.
        let mut recorder =
            EvidenceRecorder::new("final-load-failure", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
        let (backend, uploads) = upload_probe();
        let error = tritium_nn::load(
            source,
            backend,
            TensorUploadPolicy::default(),
            &budget,
            TensorUploadContext {
                recorder: &mut recorder,
                span: "load",
                parent: None,
                logical_time: 0,
            },
        )
        .err()
        .unwrap();
        assert!(
            error
                .to_string()
                .contains("model load evidence: UnregisteredSchema")
        );
        assert!(recorder.events().is_empty());
        assert_eq!(uploads.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert!(
            recorder
                .events()
                .iter()
                .all(|event| event.schema != ModelLoaded::SCHEMA_ID)
        );
        assert!(matches!(
            verify_jsonl_with_registry(
                &recorder.to_jsonl().unwrap(),
                [TENSOR_UPLOADED_REGISTRATION],
            ),
            Err(tritium_evidence::EvidenceError::EmptyRun)
        ));
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn typed_model_rejects_false_coverage_geometry_and_releases_failed_uploads() {
    let hf = HfFixture::new("llama", true, 4, None);
    for bad in [
        "missing",
        "extra",
        "shape",
        "architecture",
        "budget",
        "level",
    ] {
        let mut fixture = TritFixture::new(&hf, None);
        match bad {
            "missing" => fixture.manifest.levels[0]
                .tensors
                .retain(|t| !t.name.ends_with("down_proj.weight")),
            "extra" => {
                let mut extra = fixture.manifest.levels[0].tensors[0].clone();
                extra.name = "unexpected.weight".into();
                fixture.manifest.levels[0].tensors.push(extra);
            }
            "shape" => {
                let meta = fixture.manifest.levels[0]
                    .tensors
                    .iter_mut()
                    .find(|t| t.name.ends_with("q_proj.weight"))
                    .unwrap();
                let tritium_schema::TensorStorage::Dense { shape, .. } = &mut meta.storage else {
                    unreachable!()
                };
                *shape = vec![4, 16];
            }
            "architecture" => fixture.manifest.architecture = "not-llama".into(),
            _ => {}
        }
        let bytes = fixture.bytes();
        let budget = TensorPayloadBudget::new(if bad == "budget" { 320 } else { 100_000 });
        let mut recorder =
            EvidenceRecorder::new("nn-trit-failure", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
        let (backend, uploads) = upload_probe();
        let result = ModelRunner::from_source_admitted(
            tritium_nn::ModelSource::Trit {
                bytes: &bytes,
                level: if bad == "level" { "absent" } else { "standard" },
            },
            backend,
            TensorUploadPolicy::default(),
            &budget,
            TensorUploadContext {
                recorder: &mut recorder,
                span: "load",
                parent: None,
                logical_time: 0,
            },
        );
        assert!(result.is_err(), "{bad}");
        assert_eq!(budget.used_bytes(), 0, "{bad}");
        if bad != "budget" {
            assert!(
                recorder.events().is_empty(),
                "{bad}: upload before preflight"
            );
            assert_eq!(
                uploads.load(std::sync::atomic::Ordering::SeqCst),
                0,
                "{bad}"
            );
        }
    }
    for bad in ["unregistered", "corrupt-bytes"] {
        let fixture = TritFixture::new(&hf, None);
        let mut bytes = fixture.bytes();
        if bad == "corrupt-bytes" {
            // Corrupt an authenticated payload, not the metadata gate or oracle.
            bytes[256] ^= 1;
        }
        let budget = TensorPayloadBudget::new(100_000);
        let mut recorder = EvidenceRecorder::new(
            "nn-trit-integrity",
            if bad == "unregistered" {
                vec![]
            } else {
                vec![TENSOR_UPLOADED_REGISTRATION]
            },
        )
        .unwrap();
        assert!(
            ModelRunner::from_source_admitted(
                tritium_nn::ModelSource::Trit {
                    bytes: &bytes,
                    level: "standard"
                },
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
            .is_err(),
            "{bad}"
        );
        assert_eq!(budget.used_bytes(), 0);
        assert!(recorder.events().is_empty());
    }
}

#[cfg(feature = "cuda")]
#[test]
fn typed_packages_execute_on_real_cuda_without_backend_downcasts() {
    if let Err(e) = tritium_cuda::CudaBackend::new(0) {
        assert!(
            std::env::var_os("TRITIUM_REQUIRE_CUDA").is_none(),
            "required CUDA device: {e}"
        );
        eprintln!("UNKNOWN typed CUDA: {e}");
        return;
    }
    for tied in [false, true] {
        let hf = HfFixture::new("qwen3", tied, 6, None);
        for additive in [
            None,
            Some((
                PlaneCodec::D2,
                Basis::SignedRht {
                    block: 4,
                    seed: 7,
                    domain: 9,
                },
            )),
        ] {
            let fixture = TritFixture::new(&hf, additive);
            check_trit_parity(
                &fixture,
                Box::new(tritium_cuda::CudaBackend::new(0).unwrap()),
                true,
                additive.is_none(),
            );
        }
    }
    let hf = HfFixture::new("qwen3", true, 6, None);
    let fixture = TritFixture::new(&hf, Some((PlaneCodec::D2, Basis::Identity)));
    let bytes = fixture.bytes();
    let budget = TensorPayloadBudget::new(100_000);
    let mut recorder =
        EvidenceRecorder::new("nn-trit-native-only", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
    let runner = ModelRunner::from_source_admitted(
        tritium_nn::ModelSource::Trit {
            bytes: &bytes,
            level: "standard",
        },
        Box::new(tritium_cuda::CudaBackend::new(0).unwrap()),
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
    .expect("admitted packed additive CUDA must accept native-only");
    assert!(budget.used_bytes() > 0);
    assert!(!recorder.events().is_empty());
    assert!(
        recorder
            .events()
            .iter()
            .all(|event| event.payload["caps"]["execution"] == "Native")
    );
    drop(runner);
    assert_eq!(budget.used_bytes(), 0);
}
