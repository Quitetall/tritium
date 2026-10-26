use super::*;
use crate::{
    ChatRenderOptions, Qwen35CheckpointConfig, Qwen35MtpStatus, Qwen35TensorSchemaRole, Tokenizer,
};
use std::{collections::BTreeMap, io::Cursor};
use tritium_format::qtea::model::QteaModelTensor;
use tritium_format::{
    TritBlob, TritBlobKind, TritPackageReadLimits,
    qtea::model::{
        QTEA_HYBRID_MODEL_PROFILE, QteaModelManifest, QteaModelReadLimits,
        encode_qtea_hybrid_model_manifest,
    },
    write_trit_package_bounded,
};
use tritium_schema::{BlobId, ScalePrecision};

fn config() -> Qwen35CheckpointConfig {
    Qwen35CheckpointConfig::from_hf_config(&config_json().to_string()).unwrap()
}
fn config_json() -> serde_json::Value {
    serde_json::json!({"architectures":["Qwen3_5ForConditionalGeneration"],"language_model_only":false,
        "model_type":"qwen3_5","tie_word_embeddings":false,"vision_config":{"model_type":"qwen3_5"},
        "text_config":{"attention_bias":false,"attention_dropout":0.0,"attn_output_gate":true,
        "dtype":"bfloat16","full_attention_interval":2,"head_dim":4,"hidden_act":"silu",
        "hidden_size":8,"intermediate_size":12,"layer_types":["linear_attention","full_attention"],
        "linear_conv_kernel_dim":4,"linear_key_head_dim":2,"linear_num_key_heads":2,
        "linear_num_value_heads":4,"linear_value_head_dim":2,"mamba_ssm_dtype":"float32",
        "max_position_embeddings":16,"model_type":"qwen3_5_text","mtp_num_hidden_layers":1,
        "mtp_use_dedicated_embeddings":false,"num_attention_heads":2,"num_hidden_layers":2,
        "num_key_value_heads":1,"output_gate_type":"swish","partial_rotary_factor":0.5,
        "rms_norm_eps":1e-6,"rope_parameters":{"mrope_interleaved":true,"mrope_section":[1,0,0],
        "partial_rotary_factor":0.5,"rope_theta":10000.0,"rope_type":"default"},
        "tie_word_embeddings":false,"use_cache":true,"vocab_size":8}})
}
fn limits() -> HostQteaHybridModelLimits {
    HostQteaHybridModelLimits {
        base: HostQteaModelLimits {
            receiving: TritPackageReadLimits {
                max_source_bytes: 1 << 20,
                max_manifest_bytes: 1 << 16,
                max_directory_bytes: 1 << 16,
            },
            model: QteaModelReadLimits {
                max_tensors: 128,
                max_assets: 8,
                max_asset_bytes: 1 << 16,
                max_config_bytes: 4096,
                max_vector_elements: 32,
                tensor: tritium_format::qtea::QteaLimits {
                    max_tensor_bytes: 1 << 20,
                    max_validation_bytes: 1 << 16,
                },
            },
            max_weight_bytes: 1 << 20,
            max_staging_bytes: 1 << 16,
        },
        max_preserved_tensor_elements: 64,
    }
}
fn stream() -> super::super::StreamLimits {
    super::super::StreamLimits {
        max_context: 8,
        max_segment_tokens: 3,
        max_host_cache_payload_bytes: 4096,
    }
}
fn id(bytes: &[u8]) -> BlobId {
    BlobId::from_bytes(*blake3::hash(bytes).as_bytes())
}
const TEMPLATE: &str = "{% for message in messages %}{% if message['role'] not in ['system','user','assistant','tool'] %}{{ raise_exception('invalid role') }}{% endif %}{% endfor %}{% if enable_thinking is defined and enable_thinking %}reply {% endif %}{{ messages[0]['content'] }} partner";
#[derive(Clone)]
struct Fixture {
    manifest: QteaModelManifest,
    blobs: Vec<(TritBlobKind, Vec<u8>)>,
    decoded: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
}
impl Fixture {
    fn asset(&mut self, name: &str, bytes: Vec<u8>) {
        if name == "config.json" {
            self.manifest.config = id(&bytes);
        } else {
            self.manifest.assets.insert(name.into(), id(&bytes));
        }
        self.blobs
            .retain(|(kind, _)| *kind != TritBlobKind::Asset(name.into()));
        self.blobs.push((TritBlobKind::Asset(name.into()), bytes));
    }
    fn text_config(&mut self, template: bool) {
        let mut config = serde_json::json!({"bos_token":"[BOS]","eos_token":"[EOS]","tokenizer_class":"PreTrainedTokenizerFast"});
        if template {
            config["chat_template"] = TEMPLATE.into();
        }
        self.asset(
            "tokenizer_config.json",
            serde_json::to_vec(&config).unwrap(),
        );
    }
    fn new() -> Self {
        use tritium_quantize::{
            CurvatureSourceId, InputGramAccumulator,
            qtea_reference::{TensorFit, TensorFitConfig},
        };
        let mut f = Self {
            manifest: QteaModelManifest {
                profile: QTEA_HYBRID_MODEL_PROFILE.into(),
                version: 1,
                source_model: [3; 32],
                config: id(b""),
                assets: BTreeMap::new(),
                tensors: BTreeMap::new(),
            },
            blobs: vec![],
            decoded: BTreeMap::new(),
        };
        f.asset("config.json", serde_json::to_vec(&config_json()).unwrap());
        let words = [
            "[UNK]",
            "Testing",
            "partner",
            "[EOS]",
            "[BOS]",
            "user",
            "assistant",
            "reply",
        ];
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                words
                    .into_iter()
                    .enumerate()
                    .map(|(i, w)| (w.into(), i as u32))
                    .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
        f.asset(
            "tokenizer.json",
            tokenizer.to_string(false).unwrap().into_bytes(),
        );
        f.text_config(true);
        for (seed, entry) in crate::qwen35_language_mtp_tensor_schema(&config().text)
            .unwrap()
            .into_iter()
            .enumerate()
        {
            let count = entry.shape.iter().product::<usize>();
            let (tensor, kind, bytes, values) = match entry.role {
                Qwen35TensorSchemaRole::Matrix => {
                    let (rows, cols) = (entry.shape[0], entry.shape[1]);
                    let original = (0..count)
                        .map(|i| {
                            let (r, c) = (i / cols, i % cols);
                            ((((r + 1) * (c + 3) + seed) % 17) as f32 - 8.) * 0.015
                                + c as f32 * 0.001
                                + r as f32 * 0.002
                        })
                        .collect::<Vec<_>>();
                    let mut gram = InputGramAccumulator::new_bound(
                        cols,
                        CurvatureSourceId::new([3; 32], [4; 32], [5; 32]).unwrap(),
                        0,
                    )
                    .unwrap();
                    let mut activation = vec![0.; cols * cols];
                    for c in 0..cols {
                        activation[c * cols + c] = 1.;
                    }
                    gram.accumulate_batch(&activation, cols, None, None)
                        .unwrap();
                    let fit = TensorFit::fit(
                        &original,
                        rows,
                        cols,
                        &gram.finish().unwrap(),
                        TensorFitConfig {
                            group_size: 4,
                            salient_ratio: 0.5,
                            ..TensorFitConfig::default()
                        },
                        TensorFit::required_work_bytes(rows, cols).unwrap(),
                    )
                    .unwrap();
                    let mut decoded = vec![0.; count];
                    fit.decode_into(&mut decoded).unwrap();
                    let bytes = fit
                        .to_qtea_tensor(limits().base.model.tensor)
                        .unwrap()
                        .as_bytes()
                        .to_vec();
                    (
                        QteaModelTensor::Qtea {
                            blob: id(&bytes),
                            rows: rows as u64,
                            columns: cols as u64,
                        },
                        TritBlobKind::QteaTensor,
                        bytes,
                        decoded,
                    )
                }
                Qwen35TensorSchemaRole::Preserved => {
                    let value = if entry.name.ends_with("A_log") {
                        -1.
                    } else if entry.shape.len() > 1 {
                        0.125
                    } else {
                        0.
                    };
                    let values = vec![value; count];
                    let bytes = values
                        .iter()
                        .flat_map(|v| f32::to_le_bytes(*v))
                        .collect::<Vec<_>>();
                    let tensor = if entry.shape.len() == 1 {
                        QteaModelTensor::Vector {
                            blob: id(&bytes),
                            length: count as u64,
                            precision: ScalePrecision::F32,
                        }
                    } else {
                        QteaModelTensor::Preserved {
                            blob: id(&bytes),
                            shape: entry.shape.iter().map(|d| *d as u64).collect(),
                            precision: ScalePrecision::F32,
                        }
                    };
                    (
                        tensor,
                        TritBlobKind::DenseTensor(tritium_core::DType::F32),
                        bytes,
                        values,
                    )
                }
            };
            f.manifest.tensors.insert(entry.name.clone(), tensor);
            f.blobs.push((kind, bytes));
            f.decoded.insert(entry.name, (entry.shape, values));
        }
        f
    }
    fn bytes(&self) -> Vec<u8> {
        let manifest = encode_qtea_hybrid_model_manifest(&self.manifest, 1 << 16).unwrap();
        let blobs = self
            .blobs
            .iter()
            .map(|(kind, bytes)| TritBlob {
                kind: kind.clone(),
                bytes,
            })
            .collect::<Vec<_>>();
        write_trit_package_bounded(&manifest, &blobs, 1 << 20).unwrap()
    }
    fn oracle(&self) -> super::super::Runner {
        struct Decoded<'a>(&'a BTreeMap<String, (Vec<usize>, Vec<f32>)>);
        impl Qwen35HfTensorSource for Decoded<'_> {
            fn tensor_f32_exact(
                &self,
                name: &str,
                expected: &[usize],
            ) -> Result<Vec<f32>, NnError> {
                let (shape, values) = &self.0[name];
                assert_eq!(shape, expected);
                Ok(values.clone())
            }
        }
        let config = config();
        let source = Decoded(&self.decoded);
        let language = super::super::Qwen35TextRunner::new(
            &config.text,
            load_language_weights(&source, &config.text).unwrap(),
            Box::new(tritium_cpu::CpuBackend::new()),
        )
        .unwrap();
        let mtp = super::super::UnverifiedQwen35Mtp::new(
            &language,
            load_mtp_weights(&source, &config.text).unwrap(),
        )
        .unwrap();
        let admission = super::super::CacheAdmission::hybrid(&config, stream()).unwrap();
        super::super::Runner::hybrid(config, language, mtp, Some(admission)).unwrap()
    }
}
fn load(
    f: &Fixture,
    caps: HostQteaHybridModelLimits,
) -> Result<super::super::LoadedModel, NnError> {
    let mut recorder = tritium_evidence::EvidenceRecorder::new(
        "hybrid",
        [tritium_evidence::MODEL_LOADED_REGISTRATION],
    )
    .unwrap();
    let result = super::super::LoadedModel::load_qtea_hybrid(
        Cursor::new(f.bytes()),
        [3; 32],
        caps,
        stream(),
        tritium_runtime::TensorUploadContext {
            recorder: &mut recorder,
            span: "load",
            parent: None,
            logical_time: 0,
        },
    );
    assert_eq!(recorder.events().len(), usize::from(result.is_ok()));
    result
}
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    let mut maximum = 0_f32;
    for (a, b) in actual.iter().zip(expected) {
        maximum = maximum.max((a - b).abs());
        assert!((a - b).abs() <= 1e-5 + 1e-5 * b.abs(), "{a} != {b}");
    }
    eprintln!("synthetic hybrid QTEA/decoded max_abs_error={maximum}; not quality qualification");
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "explicit physical CUDA hybrid language/text and draft-only MTP parity; no silent skip"]
fn cuda_hybrid_model_and_unverified_draft_execute_packed() {
    let _serial = crate::model::chat_renderer::TEST_SERIAL.lock().unwrap();
    let f = Fixture::new();
    let caps = super::super::CudaQteaModelLimits {
        host: limits().base,
        max_device_weight_bytes: 1 << 20,
        max_device_staging_bytes: 1 << 16,
    };
    let mut recorder = tritium_evidence::EvidenceRecorder::new(
        "hybrid-cuda",
        [tritium_evidence::MODEL_LOADED_REGISTRATION],
    )
    .unwrap();
    let mut actual = super::super::LoadedModel::load_qtea_hybrid_cuda(
        Cursor::new(f.bytes()),
        [3; 32],
        caps,
        limits().max_preserved_tensor_elements,
        stream(),
        Box::new(tritium_cuda::CudaBackend::new(0).expect("physical CUDA required")),
        tritium_runtime::TensorUploadContext {
            recorder: &mut recorder,
            span: "load",
            parent: None,
            logical_time: 0,
        },
    )
    .unwrap();
    let mut expected = load(&f, limits()).unwrap();
    assert_eq!(
        actual.receipt.observation().backend,
        "qtea-cuda-ordered-host-orchestrated"
    );
    assert_eq!(
        actual.runner.mtp_status(),
        Some(Qwen35MtpStatus::Unverified)
    );
    let bit = |v: Vec<f32>| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bit(actual.runner.forward(&[1, 2], &[0, 1]).unwrap()),
        bit(expected.runner.forward(&[1, 2], &[0, 1]).unwrap())
    );
    assert_eq!(
        bit(actual.runner.step(4).unwrap()),
        bit(expected.runner.step(4).unwrap())
    );
    actual.runner.reset();
    expected.runner.reset();
    assert_eq!(
        actual.runner.generate_bounded(&[1, 2], 3, 3).unwrap(),
        expected.runner.generate_bounded(&[1, 2], 3, 3).unwrap()
    );
    assert_eq!(recorder.events().len(), 1);

    // Exercise the existing explicitly non-promoting draft-only interface with
    // real assembled packed owners. No fabricated oracle trace or promotion.
    let parts = |gpu: bool| {
        let backend: Box<dyn tritium_spec::TernaryBackend> = if gpu {
            Box::new(tritium_cuda::CudaBackend::new(0).expect("physical CUDA required"))
        } else {
            Box::new(tritium_cpu::CpuBackend::new())
        };
        let mut reader = QteaModelReader::open_hybrid(
            Cursor::new(f.bytes()),
            caps.host.receiving,
            QteaHybridModelReadLimits {
                base: caps.host.model,
                max_preserved_tensor_elements: limits().max_preserved_tensor_elements,
            },
            [3; 32],
        )
        .unwrap();
        let target = if gpu {
            Execution::Cuda(caps)
        } else {
            Execution::Host(caps.host)
        };
        super::super::qtea_owners::plan_target(&reader, target).unwrap();
        let owners = Source(RefCell::new(
            super::super::qtea_owners::read_target(&mut reader, target, Some(backend.as_ref()))
                .unwrap(),
        ));
        let language_weights = load_language_weights(&owners, &config().text).unwrap();
        let mtp_weights = load_mtp_weights(&owners, &config().text).unwrap();
        let language =
            super::super::Qwen35TextRunner::new(&config().text, language_weights, backend).unwrap();
        let mtp = super::super::UnverifiedQwen35Mtp::new(&language, mtp_weights).unwrap();
        reader.verify().unwrap();
        (language, mtp)
    };
    let (gpu, gpu_mtp) = parts(true);
    let (host, host_mtp) = parts(false);
    let mut gc = gpu.new_cache(8).unwrap();
    let mut hc = host.new_cache(8).unwrap();
    let go = gpu.forward(&[1, 2], &mut gc).unwrap();
    let ho = host.forward(&[1, 2], &mut hc).unwrap();
    let gd = gpu_mtp.draft_only_runner();
    let hd = host_mtp.draft_only_runner();
    let mut gdc = gd.new_cache(8).unwrap();
    let mut hdc = hd.new_cache(8).unwrap();
    let got = gd.forward(&gpu, &go, 4, &mut gdc).unwrap();
    let want = hd.forward(&host, &ho, 4, &mut hdc).unwrap();
    assert_eq!(
        got.final_hidden_states()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        want.final_hidden_states()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        got.last_logits()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        want.last_logits()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(gpu_mtp.status(), Qwen35MtpStatus::Unverified);
}
#[test]
fn native_fit_hybrid_language_mtp_loads_packed_and_runs_source_bound() {
    let _serial = crate::model::chat_renderer::TEST_SERIAL.lock().unwrap();
    let f = Fixture::new();
    let mut loaded = load(&f, limits()).unwrap();
    let mut oracle = f.oracle();
    assert_eq!(
        loaded.runner.mtp_status(),
        Some(Qwen35MtpStatus::Unverified)
    );
    assert_eq!(
        loaded.runner.qtea_package_id(),
        Some(tritium_format::PackageId::from_package_bytes(&f.bytes()))
    );
    assert_eq!(
        loaded.architecture.layer_kinds().collect::<Vec<_>>(),
        [crate::LayerKind::DeltaNet, crate::LayerKind::Attention]
    );
    let prompt = loaded
        .tokenizer
        .render_chat_prompt(&[("user", "Testing")], 1024)
        .unwrap()
        .unwrap();
    assert_eq!(prompt, "Testing partner");
    let input = loaded.tokenizer.encode(&prompt).unwrap();
    assert_eq!(input, [1, 2]);
    let controlled = loaded
        .tokenizer
        .render_chat_prompt_with_options(
            &[("user", "Testing")],
            ChatRenderOptions::default().with_thinking(true),
            1024,
        )
        .unwrap()
        .unwrap();
    assert_eq!(controlled, "reply Testing partner");
    let thinking_input = loaded.tokenizer.encode(&controlled).unwrap();
    assert_eq!(thinking_input, [7, 1, 2]);
    close(
        &loaded.runner.forward(&[1, 2], &[0, 1]).unwrap(),
        &oracle.forward(&[1, 2], &[0, 1]).unwrap(),
    );
    close(&loaded.runner.step(4).unwrap(), &oracle.step(4).unwrap());
    loaded.runner.reset();
    assert_eq!(loaded.runner.cache_len(), 0);
    let generated = loaded.runner.generate_bounded(&input, 3, 3).unwrap();
    assert_eq!(generated, oracle.generate_bounded(&[1, 2], 3, 3).unwrap());
    assert_eq!(
        loaded.runner.generate_bounded(&[1, 2, 1, 2], 3, 3).unwrap(),
        oracle.generate_bounded(&[1, 2, 1, 2], 3, 3).unwrap()
    );
    assert!(
        super::super::LoadedModel::load_qtea(
            Cursor::new(f.bytes()),
            [3; 32],
            limits().base,
            stream(),
            tritium_runtime::TensorUploadContext {
                recorder: &mut tritium_evidence::EvidenceRecorder::new(
                    "standard",
                    [tritium_evidence::MODEL_LOADED_REGISTRATION]
                )
                .unwrap(),
                span: "load",
                parent: None,
                logical_time: 0
            }
        )
        .is_err()
    );
    if let Some(root) = std::env::var_os("TRITIUM_QTEA_HYBRID_FIXTURE_OUTPUT") {
        use std::io::Write;
        let root = std::path::PathBuf::from(root).join("hybrid");
        std::fs::create_dir(&root).unwrap();
        let write = |name: &str, bytes: &[u8]| {
            std::fs::File::create_new(root.join(name))
                .unwrap()
                .write_all(bytes)
                .unwrap()
        };
        for name in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
            write(
                name,
                &f.blobs
                    .iter()
                    .find(|(kind, _)| *kind == TritBlobKind::Asset(name.into()))
                    .unwrap()
                    .1,
            );
        }
        write("model.trit", &f.bytes());
        let mut no_template = f.clone();
        no_template.text_config(false);
        write("model-no-template.trit", &no_template.bytes());
        write("actual-text.json",&serde_json::to_vec(&serde_json::json!({"schema":"tritium.qtea.text-software-observation.v1","qualified":false,"package_id":loaded.runner.qtea_package_id().unwrap().as_bytes(),"messages":[{"role":"user","content":"Testing"}],"prompt":prompt,"input_ids":input,"thinking_prompt":controlled,"thinking_input_ids":thinking_input,"generated_ids":generated})).unwrap());
    }
}
#[test]
fn hybrid_inventory_exact_roles_shapes_caps_and_mtp_are_mandatory() {
    let _serial = crate::model::chat_renderer::TEST_SERIAL.lock().unwrap();
    let f = Fixture::new();
    let mut cap = limits();
    cap.max_preserved_tensor_elements = 63;
    assert!(load(&f, cap).is_err());
    let mut cap = limits();
    cap.base.max_weight_bytes = 1;
    assert!(load(&f, cap).is_err());
    let mut cap = limits();
    cap.base.max_staging_bytes = 0;
    assert!(load(&f, cap).is_err());
    for mutation in 0..4 {
        let mut changed = f.clone();
        match mutation {
            0 => {
                changed.manifest.tensors.remove("mtp.fc.weight");
            }
            1 => {
                let entry = changed.manifest.tensors.remove("mtp.fc.weight").unwrap();
                changed
                    .manifest
                    .tensors
                    .insert("mtp.unknown.weight".into(), entry);
            }
            2 => {
                let QteaModelTensor::Preserved { shape, .. } = changed
                    .manifest
                    .tensors
                    .get_mut("model.language_model.layers.0.linear_attn.conv1d.weight")
                    .unwrap()
                else {
                    unreachable!()
                };
                *shape = vec![16, 4, 1];
            }
            _ => {
                let name = "model.language_model.layers.0.linear_attn.conv1d.weight";
                let QteaModelTensor::Preserved {
                    blob, precision, ..
                } = changed.manifest.tensors[name]
                else {
                    unreachable!()
                };
                changed.manifest.tensors.insert(
                    name.into(),
                    QteaModelTensor::Vector {
                        blob,
                        length: 64,
                        precision,
                    },
                );
            }
        }
        let mut caps = limits();
        if mutation == 3 {
            caps.base.model.max_vector_elements = 64;
        }
        let error = load(&changed, caps).err().unwrap().to_string();
        if mutation >= 2 {
            assert!(error.contains("role/shape"), "{error}");
        }
    }
}

#[test]
fn shared_hybrid_request_requires_the_explicit_profile_and_nonnull_positive_cap() {
    let _serial = crate::model::chat_renderer::TEST_SERIAL.lock().unwrap();
    let root = std::env::temp_dir().join(format!("qtea-hybrid-request-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }
    let _cleanup = Cleanup(root.clone());
    let model = root.join("model.trit");
    let path = root.join("load.json");
    let valid = serde_json::json!({"profile":"tritium.qtea-hybrid-host-load.v1","path":model,"expected_source_model":vec![3;32],
    "max_source_bytes":1<<20,"max_manifest_bytes":1<<16,"max_directory_bytes":1<<16,
    "max_tensors":128,"max_assets":8,"max_asset_bytes":1<<16,"max_config_bytes":4096,
    "max_vector_elements":32,"max_tensor_bytes":1<<20,"max_validation_bytes":1<<16,
    "max_weight_bytes":1<<20,"max_staging_bytes":1<<16,"max_preserved_tensor_elements":64,
    "stream_limits":{"max_context":8,"max_segment_tokens":3,"max_host_cache_payload_bytes":4096}});
    for variant in 0..5 {
        let mut request = valid.clone();
        match variant {
            0 => {
                request
                    .as_object_mut()
                    .unwrap()
                    .remove("max_preserved_tensor_elements");
            }
            1 => request["max_preserved_tensor_elements"] = serde_json::Value::Null,
            2 => request["max_preserved_tensor_elements"] = 0.into(),
            3 => request["profile"] = "tritium.qtea-host-load.v1".into(),
            _ => request["max_preserved_tensor_elements"] = true.into(),
        }
        std::fs::write(&path, request.to_string()).unwrap();
        assert!(crate::QteaLoadConfig::read(&path).is_err(), "{request}");
    }
    std::fs::write(&model, Fixture::new().bytes()).unwrap();
    std::fs::write(&path, valid.to_string()).unwrap();
    let mut recorder = tritium_evidence::EvidenceRecorder::new(
        "request",
        [tritium_evidence::MODEL_LOADED_REGISTRATION],
    )
    .unwrap();
    let loaded = crate::QteaLoadConfig::read(&path)
        .unwrap()
        .load(tritium_runtime::TensorUploadContext {
            recorder: &mut recorder,
            span: "load",
            parent: None,
            logical_time: 0,
        })
        .unwrap();
    assert_eq!(
        loaded.runner.mtp_status(),
        Some(Qwen35MtpStatus::Unverified)
    );
    assert_eq!(recorder.events().len(), 1);
}

#[test]
fn hybrid_context_precedes_source_io_and_tokenizer_precedes_numeric_materialization() {
    let _serial = crate::model::chat_renderer::TEST_SERIAL.lock().unwrap();
    struct NoIo;
    impl std::io::Read for NoIo {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            panic!("context refusal must precede Read")
        }
    }
    impl std::io::Seek for NoIo {
        fn seek(&mut self, _: std::io::SeekFrom) -> std::io::Result<u64> {
            panic!("context refusal must precede Seek")
        }
    }
    let mut recorder = tritium_evidence::EvidenceRecorder::new("invalid", []).unwrap();
    assert!(
        crate::LoadedModel::load_qtea_hybrid(
            NoIo,
            [3; 32],
            limits(),
            stream(),
            tritium_runtime::TensorUploadContext {
                recorder: &mut recorder,
                span: "load",
                parent: None,
                logical_time: 0
            }
        )
        .is_err()
    );
    assert!(recorder.events().is_empty());
    let mut f = Fixture::new();
    let name = "lm_head.weight";
    let blob = f.manifest.tensors[name].blob();
    let payload = &mut f
        .blobs
        .iter_mut()
        .find(|(kind, bytes)| *kind == TritBlobKind::QteaTensor && id(bytes) == blob)
        .unwrap()
        .1;
    payload[0] ^= 0xff;
    let replacement = id(payload);
    let QteaModelTensor::Qtea { blob, .. } = f.manifest.tensors.get_mut(name).unwrap() else {
        unreachable!()
    };
    *blob = replacement;
    let good = f
        .blobs
        .iter()
        .find(|(kind, _)| *kind == TritBlobKind::Asset("tokenizer.json".into()))
        .unwrap()
        .1
        .clone();
    f.asset("tokenizer.json", b"{}".to_vec());
    assert!(matches!(
        load(&f, limits()).err().unwrap(),
        NnError::Tokenizer(_)
    ));
    f.asset("tokenizer.json", good);
    assert!(matches!(
        load(&f, limits()).err().unwrap(),
        NnError::InvalidArtifact(_)
    ));
}
