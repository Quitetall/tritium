//! Cold named model loading, not pretrained Qwen or release qualification.
#[path = "qtea/common.rs"]
mod common;
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Cursor,
    sync::Arc,
};
use tritium_format::{
    TritBlob, TritBlobKind, TritPackageReadLimits,
    qtea::model::{
        QTEA_MODEL_PROFILE, QteaModelManifest, QteaModelReadLimits, QteaModelReader,
        QteaModelTensor, encode_qtea_model_manifest,
    },
    write_trit_package_bounded,
};
use tritium_nn::{
    DenseLinear, HostQteaModelLimits, Mlp, ModelConfig, ModelRunner, ModelWeights, Projection,
    SwiGluMlp, TokenEmbedding, TransformerBlock,
};
use tritium_schema::{BlobId, ScalePrecision};

struct Fixture {
    manifest: QteaModelManifest,
    blobs: Vec<(TritBlobKind, Vec<u8>)>,
    decoded: BTreeMap<String, Vec<f32>>,
    tied: bool,
}

#[cfg(feature = "tokenizer")]
impl Fixture {
    fn asset(&mut self, name: &str, bytes: Vec<u8>) {
        self.manifest.assets.insert(name.into(), id(&bytes));
        self.blobs
            .retain(|(kind, _)| *kind != TritBlobKind::Asset(name.into()));
        self.blobs.push((TritBlobKind::Asset(name.into()), bytes));
    }
    fn text(&mut self, extra_id: bool, template: Option<serde_json::Value>) {
        let mut vocab = [
            "[UNK]",
            "Testing",
            "partner",
            "[EOS]",
            "[BOS]",
            "user",
            "assistant",
            "reply",
        ]
        .into_iter()
        .enumerate()
        .map(|(id, word)| (word.to_owned(), id as u32))
        .collect::<Vec<_>>();
        if extra_id {
            vocab.push(("outside-model".into(), 8));
        }
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab.into_iter().collect())
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
        self.asset(
            "tokenizer.json",
            tokenizer.to_string(false).unwrap().into_bytes(),
        );
        let mut config = serde_json::json!({"bos_token":"[BOS]","eos_token":"[EOS]","tokenizer_class":"PreTrainedTokenizerFast"});
        if let Some(template) = template {
            config["chat_template"] = template;
        }
        self.asset(
            "tokenizer_config.json",
            serde_json::to_vec(&config).unwrap(),
        );
    }
}
fn id(bytes: &[u8]) -> BlobId {
    BlobId::from_bytes(*blake3::hash(bytes).as_bytes())
}
fn limits() -> HostQteaModelLimits {
    HostQteaModelLimits {
        receiving: TritPackageReadLimits {
            max_source_bytes: 1 << 20,
            max_manifest_bytes: 1 << 16,
            max_directory_bytes: 1 << 16,
        },
        model: QteaModelReadLimits {
            max_tensors: 64,
            max_assets: 8,
            max_asset_bytes: 1 << 16,
            max_config_bytes: 4096,
            max_vector_elements: 32,
            tensor: common::limits(),
        },
        max_weight_bytes: 1 << 20,
        max_staging_bytes: 1 << 16,
    }
}
fn config() -> ModelConfig {
    ModelConfig {
        arch: "llama".into(),
        n_layers: 2,
        n_embd: 8,
        n_head: 2,
        n_head_kv: 1,
        head_dim: 4,
        n_ff: 12,
        n_ctx: 16,
        rope_theta: 10000.,
        rms_eps: 1e-5,
    }
}
impl Fixture {
    fn new(tied: bool, fitted: bool) -> Self {
        let config = serde_json::to_vec(
            &serde_json::json!({"model_type":"llama", "num_hidden_layers":2,
            "hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
            "intermediate_size":12,"max_position_embeddings":16,"rope_theta":10000.,
            "rms_norm_eps":1e-5,"vocab_size":8,"tie_word_embeddings":tied}),
        )
        .unwrap();
        let token = b"{}".to_vec();
        let mut out = Self {
            manifest: QteaModelManifest {
                profile: QTEA_MODEL_PROFILE.into(),
                version: 1,
                source_model: [3; 32],
                config: id(&config),
                assets: BTreeMap::from([("tokenizer.json".into(), id(&token))]),
                tensors: BTreeMap::new(),
            },
            blobs: vec![
                (TritBlobKind::Asset("config.json".into()), config),
                (TritBlobKind::Asset("tokenizer.json".into()), token),
            ],
            decoded: BTreeMap::new(),
            tied,
        };
        out.matrix("model.embed_tokens.weight", 8, 8, 1, fitted);
        out.vector("model.norm.weight");
        if !tied {
            out.matrix("lm_head.weight", 8, 8, 31, fitted);
        }
        for layer in 0..2 {
            let p = format!("model.layers.{layer}");
            out.vector(&format!("{p}.input_layernorm.weight"));
            out.vector(&format!("{p}.post_attention_layernorm.weight"));
            for (name, rows, cols, seed) in [
                ("self_attn.q_proj.weight", 8, 8, 2),
                ("self_attn.k_proj.weight", 4, 8, 3),
                ("self_attn.v_proj.weight", 4, 8, 4),
                ("self_attn.o_proj.weight", 8, 8, 5),
                ("mlp.gate_proj.weight", 12, 8, 6),
                ("mlp.up_proj.weight", 12, 8, 7),
                ("mlp.down_proj.weight", 8, 12, 8),
            ] {
                out.matrix(&format!("{p}.{name}"), rows, cols, seed + layer * 9, fitted);
            }
        }
        out
    }
    fn matrix(&mut self, name: &str, rows: usize, cols: usize, seed: usize, fitted: bool) {
        let (tensor, mut decoded) = common::fixture(rows, cols, 2, 0.5, seed);
        let tensor = if fitted {
            use tritium_quantize::qtea_reference::{TensorFit, TensorFitConfig};
            use tritium_quantize::{CurvatureSourceId, InputGramAccumulator};
            decoded = (0..rows)
                .flat_map(|r| {
                    (0..cols).map(move |c| {
                        (((r + 1) * (c + 3)) % 17) as f32 * 0.35 - 2.8
                            + c as f32 * 0.19
                            + r as f32 * 0.07
                            + seed as f32 * 0.013
                    })
                })
                .collect();
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
                &decoded,
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
            fit.decode_into(&mut decoded).unwrap();
            fit.to_qtea_tensor(common::limits()).unwrap()
        } else {
            tensor
        };
        let bytes = tensor.as_bytes().to_vec();
        self.manifest.tensors.insert(
            name.into(),
            QteaModelTensor::Qtea {
                blob: id(&bytes),
                rows: rows as u64,
                columns: cols as u64,
            },
        );
        self.blobs.push((TritBlobKind::QteaTensor, bytes));
        self.decoded.insert(name.into(), decoded);
    }
    fn vector(&mut self, name: &str) {
        let bytes = [1_f32; 8]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        self.manifest.tensors.insert(
            name.into(),
            QteaModelTensor::Vector {
                blob: id(&bytes),
                length: 8,
                precision: ScalePrecision::F32,
            },
        );
        self.blobs
            .push((TritBlobKind::DenseTensor(tritium_core::DType::F32), bytes));
    }
    fn bytes(&self) -> Vec<u8> {
        self.bytes_with_limit(1 << 20)
    }
    fn bytes_with_limit(&self, maximum: usize) -> Vec<u8> {
        let manifest = encode_qtea_model_manifest(&self.manifest, 1 << 16).unwrap();
        let blobs = self
            .blobs
            .iter()
            .map(|(kind, bytes)| TritBlob {
                kind: kind.clone(),
                bytes,
            })
            .collect::<Vec<_>>();
        write_trit_package_bounded(&manifest, &blobs, maximum).unwrap()
    }
    fn payload_bytes(&self) -> usize {
        let mut used = BTreeSet::new();
        self.manifest
            .tensors
            .values()
            .map(|tensor| match tensor {
                QteaModelTensor::Qtea { blob, .. } if used.insert(*blob.as_bytes()) => self
                    .blobs
                    .iter()
                    .find(|(_, b)| id(b) == *blob)
                    .unwrap()
                    .1
                    .len(),
                QteaModelTensor::Qtea { .. } => 0,
                QteaModelTensor::Vector { length, .. } => *length as usize * 4,
            })
            .sum()
    }
    fn oracle(&self) -> ModelWeights {
        let dense = |name: &str, rows, cols| {
            Projection::Dense(
                DenseLinear::new_exact(self.decoded[name].clone(), rows, cols).unwrap(),
            )
        };
        let layers = (0..2)
            .map(|layer| {
                let p = format!("model.layers.{layer}");
                TransformerBlock {
                    attn_norm: vec![1.; 8],
                    ffn_norm: vec![1.; 8],
                    q_proj: dense(&format!("{p}.self_attn.q_proj.weight"), 8, 8),
                    k_proj: dense(&format!("{p}.self_attn.k_proj.weight"), 4, 8),
                    v_proj: dense(&format!("{p}.self_attn.v_proj.weight"), 4, 8),
                    o_proj: dense(&format!("{p}.self_attn.o_proj.weight"), 8, 8),
                    q_bias: vec![],
                    k_bias: vec![],
                    v_bias: vec![],
                    q_norm: vec![],
                    k_norm: vec![],
                    attn_sub_norm: vec![],
                    mlp: Mlp::SwiGlu(SwiGluMlp {
                        gate: dense(&format!("{p}.mlp.gate_proj.weight"), 12, 8),
                        up: dense(&format!("{p}.mlp.up_proj.weight"), 12, 8),
                        down: dense(&format!("{p}.mlp.down_proj.weight"), 8, 12),
                    }),
                }
            })
            .collect();
        ModelWeights {
            token_embd: TokenEmbedding::from_dense(
                self.decoded["model.embed_tokens.weight"].clone(),
                8,
                8,
            )
            .unwrap(),
            vocab: 8,
            n_embd: 8,
            layers,
            output_norm: vec![1.; 8],
            lm_head: (!self.tied).then(|| dense("lm_head.weight", 8, 8)),
        }
    }
}
fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[cfg(feature = "tokenizer")]
fn command_config(path: &std::path::Path) -> serde_json::Value {
    serde_json::json!({
        "profile":"tritium.qtea-host-load.v1", "path":path,"expected_source_model":vec![3;32],
        "max_source_bytes":1<<20,"max_manifest_bytes":1<<16,"max_directory_bytes":1<<16,
        "max_tensors":64,"max_assets":8,"max_asset_bytes":1<<16,"max_config_bytes":4096,
        "max_vector_elements":32,"max_tensor_bytes":1<<20,"max_validation_bytes":1<<16,
        "max_weight_bytes":1<<20,"max_staging_bytes":1<<16,
        "stream_limits":{"max_context":8,"max_segment_tokens":3,"max_host_cache_payload_bytes":4096}
    })
}

#[cfg(feature = "tokenizer")]
struct CommandScratch(std::path::PathBuf);
#[cfg(feature = "tokenizer")]
impl CommandScratch {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "qtea-command-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
#[cfg(feature = "tokenizer")]
impl Drop for CommandScratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[cfg(feature = "tokenizer")]
#[test]
fn shared_command_config_is_strict_capped_and_revalidated_before_model_io() {
    use tritium_nn::QteaLoadConfig;
    let root = CommandScratch::new();
    let path = root.0.join("load.json");
    let valid = command_config(&root.0.join("absent.trit"));
    let mut invalid = Vec::new();
    for field in [
        "max_source_bytes",
        "max_manifest_bytes",
        "max_directory_bytes",
        "max_tensors",
        "max_assets",
        "max_asset_bytes",
        "max_config_bytes",
        "max_vector_elements",
        "max_tensor_bytes",
        "max_validation_bytes",
        "max_weight_bytes",
        "max_staging_bytes",
    ] {
        let mut request = valid.clone();
        request[field] = 0.into();
        invalid.push(request);
    }
    for field in [
        "max_context",
        "max_segment_tokens",
        "max_host_cache_payload_bytes",
    ] {
        let mut request = valid.clone();
        request["stream_limits"][field] = 0.into();
        invalid.push(request);
    }
    for (field, value) in [
        ("profile", serde_json::json!("tritium.qtea-host-load.v2")),
        ("path", serde_json::json!("relative.trit")),
        ("expected_source_model", serde_json::json!(vec![0; 32])),
    ] {
        let mut request = valid.clone();
        request[field] = value;
        invalid.push(request);
    }
    let mut overflow = valid.clone();
    overflow["stream_limits"]["max_segment_tokens"] = 9.into();
    invalid.push(overflow);
    for request in invalid {
        std::fs::write(&path, request.to_string()).unwrap();
        assert!(QteaLoadConfig::read(&path).is_err(), "{request}");
        // Even a caller deserializing directly cannot bypass validation at load.
        let config: QteaLoadConfig = serde_json::from_value(request).unwrap();
        let mut recorder = tritium_evidence::EvidenceRecorder::new(
            "revalidate",
            [tritium_evidence::MODEL_LOADED_REGISTRATION],
        )
        .unwrap();
        let error = config
            .load(tritium_runtime::TensorUploadContext {
                recorder: &mut recorder,
                span: "load",
                parent: None,
                logical_time: 0,
            })
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("--qtea"), "{error}");
        assert!(recorder.events().is_empty());
    }
    for raw in [
        valid.to_string().replacen('{', "{\"unexpected\":true,", 1),
        valid.to_string().replacen('{', "{\"max_tensors\":64,", 1),
        valid
            .to_string()
            .replace("\"max_context\":8", "\"max_context\":8,\"unknown\":0"),
    ] {
        std::fs::write(&path, raw).unwrap();
        assert!(QteaLoadConfig::read(&path).is_err());
    }
    std::fs::write(&path, vec![b' '; 16385]).unwrap();
    assert!(
        QteaLoadConfig::read(&path)
            .unwrap_err()
            .to_string()
            .contains("16KiB")
    );
    assert!(
        QteaLoadConfig::read(&root.0)
            .unwrap_err()
            .to_string()
            .contains("regular file")
    );
    std::fs::write(&path, valid.to_string()).unwrap();
    let mut recorder = tritium_evidence::EvidenceRecorder::new("unregistered", []).unwrap();
    let error = QteaLoadConfig::read(&path)
        .unwrap()
        .load(tritium_runtime::TensorUploadContext {
            recorder: &mut recorder,
            span: "load",
            parent: None,
            logical_time: 0,
        })
        .err()
        .unwrap()
        .to_string();
    assert!(
        !error.contains("No such file"),
        "context refusal must precede absent model I/O: {error}"
    );
    assert!(recorder.events().is_empty());
}

#[cfg(feature = "tokenizer")]
#[test]
fn shared_command_config_loads_real_packed_models_and_checks_source_size() {
    use tritium_nn::{QteaLoadConfig, Tokenizer};
    let root = CommandScratch::new();
    let path = root.0.join("load.json");
    let model = root.0.join("model.trit");
    for tied in [true, false] {
        let mut fixture = Fixture::new(tied, true);
        fixture.text(false, Some("{{ messages[0]['content'] }} partner".into()));
        let bytes = fixture.bytes();
        std::fs::write(&model, &bytes).unwrap();
        let mut config = command_config(&model);
        for insufficient in [true, false] {
            config["max_source_bytes"] = if insufficient {
                bytes.len() - 1
            } else {
                bytes.len()
            }
            .into();
            std::fs::write(&path, config.to_string()).unwrap();
            let mut recorder = tritium_evidence::EvidenceRecorder::new(
                "shared",
                [tritium_evidence::MODEL_LOADED_REGISTRATION],
            )
            .unwrap();
            let result =
                QteaLoadConfig::read(&path)
                    .unwrap()
                    .load(tritium_runtime::TensorUploadContext {
                        recorder: &mut recorder,
                        span: "load",
                        parent: None,
                        logical_time: 0,
                    });
            if insufficient {
                assert!(
                    result
                        .err()
                        .unwrap()
                        .to_string()
                        .contains("max_source_bytes")
                );
                assert!(recorder.events().is_empty());
                continue;
            }
            let mut loaded = result.unwrap();
            assert!(loaded.tokenizer.has_source_chat_template());
            assert_eq!(
                loaded.runner.qtea_package_id(),
                Some(tritium_format::PackageId::from_package_bytes(&bytes))
            );
            let mut oracle = ModelRunner::try_from_weights(
                crate::config(),
                fixture.oracle(),
                Box::new(tritium_cpu::CpuBackend::new()),
            )
            .unwrap();
            assert_eq!(
                loaded
                    .runner
                    .generate_bounded(&[1, 2], 3, loaded.tokenizer.eos())
                    .unwrap(),
                oracle.generate(&[1, 2], 3, loaded.tokenizer.eos()).unwrap()
            );
            assert_eq!(recorder.events().len(), 1);
        }
    }
}

#[test]
fn named_cold_models_and_native_fit_exports_run_packed() {
    for tied in [true, false] {
        for fitted in [false, true] {
            let fixture = Fixture::new(tied, fitted);
            let bytes = fixture.bytes();
            assert!(tritium_format::trit_model::read_trit_model(&bytes).is_err());
            let mut bounded = limits();
            bounded.max_weight_bytes = fixture.payload_bytes();
            let (actual_config, spec, weights, package) =
                ModelWeights::load_qtea(Cursor::new(&bytes), [3; 32], bounded).unwrap();
            assert_eq!(
                package,
                tritium_format::PackageId::from_package_bytes(&bytes)
            );
            assert_eq!(spec.tied_embeddings, tied);
            assert!(weights.token_embd.as_dense().is_none());
            assert!(!weights.token_embd.is_packed_salt());
            assert!(weights.token_embd.tensor_caps().is_none());
            assert!(matches!(weights.layers[0].q_proj, Projection::HostQtea(_)));
            assert_eq!(weights.lm_head.is_none(), tied);
            let mut actual = ModelRunner::try_from_weights(
                actual_config,
                weights,
                Box::new(tritium_cpu::CpuBackend::new()),
            )
            .unwrap();
            let mut reference = ModelRunner::try_from_weights(
                config(),
                fixture.oracle(),
                Box::new(tritium_cpu::CpuBackend::new()),
            )
            .unwrap();
            for (tokens, positions) in [
                (&[1, 2, 3][..], &[0, 1, 2][..]),
                (&[4][..], &[3][..]),
                (&[5][..], &[4][..]),
            ] {
                assert_eq!(
                    bits(&actual.forward(tokens, positions).unwrap()),
                    bits(&reference.forward(tokens, positions).unwrap())
                );
            }
            actual.reset();
            reference.reset();
            assert_eq!(
                bits(&actual.forward(&[2], &[0]).unwrap()),
                bits(&reference.forward(&[2], &[0]).unwrap())
            );
            bounded.max_weight_bytes -= 1;
            assert!(ModelWeights::load_qtea(Cursor::new(&bytes), [3; 32], bounded).is_err());
        }
    }
}

#[test]
fn complete_owner_roles_source_and_caps_are_enforced() {
    let fixture = Fixture::new(true, false);
    let bytes = fixture.bytes();
    assert!(ModelWeights::load_qtea(Cursor::new(&bytes), [9; 32], limits()).is_err());
    for mutate in [0, 1, 2, 3] {
        let mut changed = Fixture::new(true, false);
        let key = "model.layers.0.self_attn.q_proj.weight";
        match mutate {
            0 => {
                let entry = changed.manifest.tensors.remove(key).unwrap();
                changed
                    .manifest
                    .tensors
                    .insert("renamed.weight".into(), entry);
            }
            1 => {
                let QteaModelTensor::Qtea { columns, .. } =
                    changed.manifest.tensors.get_mut(key).unwrap()
                else {
                    unreachable!()
                };
                *columns = 10;
            }
            2 => {
                changed.manifest.tensors.insert(
                    "lm_head.weight".into(),
                    changed.manifest.tensors["model.embed_tokens.weight"].clone(),
                );
            }
            _ => {
                let vector = changed.manifest.tensors["model.norm.weight"].clone();
                changed.manifest.tensors.insert(key.into(), vector);
            }
        }
        assert!(ModelWeights::load_qtea(Cursor::new(changed.bytes()), [3; 32], limits()).is_err());
    }
    let mut cap = limits();
    cap.model.max_tensors = 1;
    assert!(ModelWeights::load_qtea(Cursor::new(&bytes), [3; 32], cap).is_err());
    cap = limits();
    cap.model.max_asset_bytes = 1;
    assert!(ModelWeights::load_qtea(Cursor::new(&bytes), [3; 32], cap).is_err());
    cap = limits();
    cap.receiving.max_source_bytes = bytes.len() as u64 - 1;
    assert!(ModelWeights::load_qtea(Cursor::new(&bytes), [3; 32], cap).is_err());
}

#[test]
fn identical_immutable_blobs_share_storage_without_inventing_aliases() {
    let mut fixture = Fixture::new(true, false);
    let q = "model.layers.0.self_attn.q_proj.weight";
    let o = "model.layers.0.self_attn.o_proj.weight";
    let old = fixture.manifest.tensors[o].blob();
    fixture
        .manifest
        .tensors
        .insert(o.into(), fixture.manifest.tensors[q].clone());
    fixture.blobs.retain(|(_, bytes)| id(bytes) != old);
    let mut bounded = limits();
    bounded.max_weight_bytes = fixture.payload_bytes();
    let (_, _, weights, _) =
        ModelWeights::load_qtea(Cursor::new(fixture.bytes()), [3; 32], bounded).unwrap();
    let (Projection::HostQtea(q), Projection::HostQtea(o)) =
        (&weights.layers[0].q_proj, &weights.layers[0].o_proj)
    else {
        panic!("packed storage required")
    };
    assert!(Arc::ptr_eq(q, o));
}

#[test]
fn retained_carrier_assets_and_terminal_mutation_are_checked() {
    use std::{
        cell::RefCell,
        io::{self, Read, Seek, SeekFrom},
        rc::Rc,
    };
    struct Shared {
        bytes: Rc<RefCell<Cursor<Vec<u8>>>>,
    }
    impl Read for Shared {
        fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
            self.bytes.borrow_mut().read(b)
        }
    }
    impl Seek for Shared {
        fn seek(&mut self, p: SeekFrom) -> io::Result<u64> {
            self.bytes.borrow_mut().seek(p)
        }
    }
    let bytes = Rc::new(RefCell::new(Cursor::new(Fixture::new(true, false).bytes())));
    let mut reader = QteaModelReader::open(
        Shared {
            bytes: Rc::clone(&bytes),
        },
        limits().receiving,
        limits().model,
        [3; 32],
    )
    .unwrap();
    assert_eq!(reader.asset("tokenizer.json").unwrap(), b"{}");
    assert!(reader.asset("missing").is_err());
    reader.verify().unwrap();
    let n = bytes.borrow().get_ref().len();
    bytes.borrow_mut().get_mut()[n - 1] ^= 1;
    assert!(reader.verify().is_err());
}

#[cfg(feature = "tokenizer")]
#[test]
fn common_host_runner_loads_the_same_carrier_without_dense_fallback() {
    use tritium_nn::{Runner, StreamLimits};
    let fixture = Fixture::new(true, false);
    let bytes = fixture.bytes();
    let package = tritium_format::PackageId::from_package_bytes(&bytes);
    let mut actual = Runner::load_qtea(
        Cursor::new(bytes),
        [3; 32],
        limits(),
        StreamLimits {
            max_context: 8,
            max_segment_tokens: 3,
            max_host_cache_payload_bytes: 4096,
        },
    )
    .unwrap();
    let mut expected = ModelRunner::try_from_weights(
        config(),
        fixture.oracle(),
        Box::new(tritium_cpu::CpuBackend::new()),
    )
    .unwrap();
    assert_eq!(actual.max_context(), 8);
    assert_eq!(actual.qtea_package_id(), Some(package));
    assert_eq!(
        bits(&actual.forward(&[1, 2, 3], &[0, 1, 2]).unwrap()),
        bits(&expected.forward(&[1, 2, 3], &[0, 1, 2]).unwrap())
    );
    assert_eq!(
        bits(&actual.step(4).unwrap()),
        bits(&expected.forward(&[4], &[3]).unwrap())
    );
    actual.reset();
    assert_eq!(actual.cache_len(), 0);
}

#[cfg(feature = "tokenizer")]
fn stream() -> tritium_nn::StreamLimits {
    tritium_nn::StreamLimits {
        max_context: 8,
        max_segment_tokens: 3,
        max_host_cache_payload_bytes: 4096,
    }
}

#[cfg(feature = "tokenizer")]
#[test]
fn source_bound_text_load_renders_encodes_generates_and_replays() {
    use tritium_evidence::{
        EvidenceRecorder, MODEL_LOADED_REGISTRATION, ModelLoadReceiptView,
        verify_jsonl_with_registry,
    };
    use tritium_nn::{ChatRenderOptions, LoadedModel, Tokenizer};
    let template = concat!(
        "{% for message in messages %}{% if message['role'] not in ['system','user','assistant','tool'] %}{{ raise_exception('invalid role') }}{% endif %}{% endfor %}",
        "{% if enable_thinking is defined and enable_thinking %}reply {% endif %}{{ messages[0]['content'] }} partner"
    );
    for tied in [true, false] {
        let mut fixture = Fixture::new(tied, true);
        fixture.text(false, Some(template.into()));
        let bytes = fixture.bytes();
        let package = tritium_format::PackageId::from_package_bytes(&bytes);
        let mut recorder = EvidenceRecorder::new("qtea-text", [MODEL_LOADED_REGISTRATION]).unwrap();
        let mut loaded = LoadedModel::load_qtea(
            Cursor::new(&bytes),
            [3; 32],
            limits(),
            stream(),
            tritium_runtime::TensorUploadContext {
                recorder: &mut recorder,
                span: "load",
                parent: None,
                logical_time: 0,
            },
        )
        .unwrap();
        assert_eq!(loaded.runner.qtea_package_id(), Some(package));
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
        assert_eq!(loaded.tokenizer.encode(&controlled).unwrap(), [7, 1, 2]);
        assert!(
            loaded
                .tokenizer
                .render_chat_prompt(&[("user", "Testing")], 1)
                .is_err()
        );
        let mut reference = ModelRunner::try_from_weights(
            config(),
            fixture.oracle(),
            Box::new(tritium_cpu::CpuBackend::new()),
        )
        .unwrap();
        assert_eq!(
            bits(&loaded.runner.forward(&input, &[0, 1]).unwrap()),
            bits(&reference.forward(&input, &[0, 1]).unwrap())
        );
        let generated = loaded
            .runner
            .generate(&input, 3, loaded.tokenizer.eos())
            .unwrap();
        assert_eq!(
            generated,
            reference
                .generate(&input, 3, loaded.tokenizer.eos())
                .unwrap()
        );
        loaded.tokenizer.decode(&generated).unwrap();
        if let Some(destination) = std::env::var_os("TRITIUM_QTEA_TEXT_FIXTURE_OUTPUT") {
            use std::io::Write;
            let destination =
                std::path::PathBuf::from(destination).join(if tied { "tied" } else { "untied" });
            std::fs::create_dir(&destination).unwrap();
            let write = |name: &str, data: &[u8]| {
                std::fs::File::create_new(destination.join(name))
                    .unwrap()
                    .write_all(data)
                    .unwrap();
            };
            for name in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
                let (_, data) = fixture
                    .blobs
                    .iter()
                    .find(|(kind, _)| *kind == TritBlobKind::Asset(name.into()))
                    .unwrap();
                write(name, data);
            }
            write("model.trit", &bytes);
            fixture.text(false, None);
            write("model-no-template.trit", &fixture.bytes());
            write("actual-text.json",&serde_json::to_vec(&serde_json::json!({
                "schema":"tritium.qtea.text-software-observation.v1","qualified":false,
                "package_id":package.as_bytes(),"messages":[{"role":"user","content":"Testing"}],
                "prompt":prompt,"input_ids":input,"thinking_prompt":controlled,
                "thinking_input_ids":loaded.tokenizer.encode(&controlled).unwrap(),"generated_ids":generated,
            })).unwrap());
        }
        let observation = loaded.receipt.observation();
        assert_eq!(observation.backend, "qtea-scalar-host");
        assert_eq!(
            observation.source,
            tritium_schema::LoadedSourceIdentity::Qtea {
                package_id: tritium_schema::PackageId::from_bytes(*package.as_bytes()),
                declared_source_model: [3; 32]
            }
        );
        assert_eq!(recorder.events().len(), 1);
        let log =
            verify_jsonl_with_registry(&recorder.to_jsonl().unwrap(), [MODEL_LOADED_REGISTRATION])
                .unwrap();
        assert_eq!(ModelLoadReceiptView::fold(&log).unwrap(), [loaded.receipt]);
    }
}

#[cfg(feature = "tokenizer")]
#[test]
fn bad_tokenizer_assets_fail_before_publishing_a_load_observation() {
    use tritium_evidence::{EvidenceRecorder, MODEL_LOADED_REGISTRATION};
    for bad in [
        "missing-tokenizer",
        "missing-config",
        "json",
        "duplicate-config",
        "ambiguous",
        "syntax",
        "vocab",
    ] {
        let mut fixture = Fixture::new(true, false);
        fixture.text(bad == "vocab", Some("{{ messages[0]['content'] }}".into()));
        match bad {
            "missing-tokenizer" | "missing-config" => {
                let name = if bad == "missing-tokenizer" {
                    "tokenizer.json"
                } else {
                    "tokenizer_config.json"
                };
                fixture.manifest.assets.remove(name);
                fixture
                    .blobs
                    .retain(|(kind, _)| *kind != TritBlobKind::Asset(name.into()));
            }
            "json" => fixture.asset("tokenizer.json", b"{}".to_vec()),
            "duplicate-config" => fixture.asset(
                "tokenizer_config.json",
                br#"{"eos_token":"[EOS]","eos_token":"[BOS]"}"#.to_vec(),
            ),
            "ambiguous" => fixture.text(false, Some(serde_json::json!({"a":"A","b":"B"}))),
            "syntax" => fixture.text(false, Some("{% if".into())),
            _ => {}
        }
        let mut recorder =
            EvidenceRecorder::new("qtea-invalid", [MODEL_LOADED_REGISTRATION]).unwrap();
        let before = recorder.root_digest();
        let result = tritium_nn::LoadedModel::load_qtea(
            Cursor::new(fixture.bytes()),
            [3; 32],
            limits(),
            stream(),
            tritium_runtime::TensorUploadContext {
                recorder: &mut recorder,
                span: "load",
                parent: None,
                logical_time: 0,
            },
        );
        let Err(error) = result else {
            panic!("invalid {bad} accepted")
        };
        assert!(
            matches!(error, tritium_nn::NnError::Tokenizer(_)),
            "{bad}: {error}"
        );
        assert!(recorder.events().is_empty());
        assert_eq!(recorder.root_digest(), before);
    }
}

#[cfg(feature = "tokenizer")]
#[test]
fn tokenizer_validation_precedes_actual_matrix_decode_and_caps_owned_assets() {
    use tritium_evidence::{EvidenceRecorder, MODEL_LOADED_REGISTRATION};
    let mut fixture = Fixture::new(true, false);
    fixture.text(false, Some("{{ messages[0]['content'] }}".into()));
    let name = "model.layers.0.self_attn.q_proj.weight";
    let old = fixture.manifest.tensors[name].blob();
    let (_, bytes) = fixture
        .blobs
        .iter_mut()
        .find(|(_, b)| id(b) == old)
        .unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    let new = id(bytes);
    let QteaModelTensor::Qtea { blob, .. } = fixture.manifest.tensors.get_mut(name).unwrap() else {
        unreachable!()
    };
    *blob = new;
    for malformed in [true, false] {
        if malformed {
            fixture.asset("tokenizer.json", b"{}".to_vec());
        } else {
            fixture.text(false, Some("{{ messages[0]['content'] }}".into()));
        }
        let mut recorder = EvidenceRecorder::new("predecode", [MODEL_LOADED_REGISTRATION]).unwrap();
        let result = tritium_nn::LoadedModel::load_qtea(
            Cursor::new(fixture.bytes()),
            [3; 32],
            limits(),
            stream(),
            tritium_runtime::TensorUploadContext {
                recorder: &mut recorder,
                span: "load",
                parent: None,
                logical_time: 0,
            },
        );
        let Err(error) = result else {
            panic!("bad matrix accepted")
        };
        if malformed {
            assert!(matches!(error, tritium_nn::NnError::Tokenizer(_)));
        } else {
            assert!(matches!(error, tritium_nn::NnError::InvalidArtifact(_)));
        }
        assert!(recorder.events().is_empty());
    }
    let mut fixture = Fixture::new(true, false);
    fixture.text(false, None);
    fixture.asset("tokenizer_config.json", vec![b' '; 1024 * 1024 + 1]);
    let mut cap = limits();
    cap.model.max_asset_bytes = 2 * 1024 * 1024;
    cap.receiving.max_source_bytes = 4 * 1024 * 1024;
    let mut recorder = EvidenceRecorder::new("tokenizer-cap", [MODEL_LOADED_REGISTRATION]).unwrap();
    let result = tritium_nn::LoadedModel::load_qtea(
        Cursor::new(fixture.bytes_with_limit(4 << 20)),
        [3; 32],
        cap,
        stream(),
        tritium_runtime::TensorUploadContext {
            recorder: &mut recorder,
            span: "load",
            parent: None,
            logical_time: 0,
        },
    );
    let Err(error) = result else {
        panic!("oversized config accepted")
    };
    assert!(error.to_string().contains("source size limit"), "{error}");
    assert!(recorder.events().is_empty());
}

#[cfg(feature = "tokenizer")]
#[test]
fn source_absence_stays_absent_and_asset_changes_rebind_package_identity() {
    use tritium_evidence::{EvidenceRecorder, MODEL_LOADED_REGISTRATION};
    use tritium_nn::{LoadedModel, Tokenizer};
    let mut fixture = Fixture::new(true, false);
    fixture.text(false, None);
    let absent = fixture.bytes();
    let absent_id = tritium_format::PackageId::from_package_bytes(&absent);
    let mut recorder =
        EvidenceRecorder::new("absent-template", [MODEL_LOADED_REGISTRATION]).unwrap();
    let loaded = LoadedModel::load_qtea(
        Cursor::new(absent),
        [3; 32],
        limits(),
        stream(),
        tritium_runtime::TensorUploadContext {
            recorder: &mut recorder,
            span: "load",
            parent: None,
            logical_time: 0,
        },
    )
    .unwrap();
    assert!(
        loaded
            .tokenizer
            .render_chat_prompt(&[("user", "Testing")], 1024)
            .unwrap()
            .is_none()
    );
    fixture.text(false, Some("partner".into()));
    let bytes = fixture.bytes();
    let changed_id = tritium_format::PackageId::from_package_bytes(&bytes);
    assert_ne!(changed_id, absent_id);
    let mut recorder =
        EvidenceRecorder::new("changed-template", [MODEL_LOADED_REGISTRATION]).unwrap();
    let changed = LoadedModel::load_qtea(
        Cursor::new(bytes),
        [3; 32],
        limits(),
        stream(),
        tritium_runtime::TensorUploadContext {
            recorder: &mut recorder,
            span: "load",
            parent: None,
            logical_time: 0,
        },
    )
    .unwrap();
    assert_eq!(changed.runner.qtea_package_id(), Some(changed_id));
    assert_eq!(
        changed
            .tokenizer
            .render_chat_prompt(&[("user", "Testing")], 1024)
            .unwrap()
            .as_deref(),
        Some("partner")
    );
    assert_ne!(
        changed.receipt.observation().source,
        loaded.receipt.observation().source
    );
}

#[cfg(feature = "tokenizer")]
#[test]
fn recording_context_is_checked_before_source_io() {
    use std::{
        cell::Cell,
        io::{self, Read, Seek, SeekFrom},
        rc::Rc,
    };
    use tritium_evidence::{EvidenceRecorder, MODEL_LOADED_REGISTRATION};
    struct Count {
        source: Cursor<Vec<u8>>,
        calls: Rc<Cell<usize>>,
    }
    impl Read for Count {
        fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
            self.calls.set(self.calls.get() + 1);
            self.source.read(b)
        }
    }
    impl Seek for Count {
        fn seek(&mut self, p: SeekFrom) -> io::Result<u64> {
            self.calls.set(self.calls.get() + 1);
            self.source.seek(p)
        }
    }
    for registered in [false, true] {
        let calls = Rc::new(Cell::new(0));
        let source = Count {
            source: Cursor::new(vec![]),
            calls: Rc::clone(&calls),
        };
        let registrations = if registered {
            vec![MODEL_LOADED_REGISTRATION]
        } else {
            vec![]
        };
        let mut recorder = EvidenceRecorder::new("preflight", registrations).unwrap();
        let result = tritium_nn::LoadedModel::load_qtea(
            source,
            [3; 32],
            limits(),
            stream(),
            tritium_runtime::TensorUploadContext {
                recorder: &mut recorder,
                span: if registered { "" } else { "load" },
                parent: None,
                logical_time: 0,
            },
        );
        assert!(result.is_err());
        assert_eq!(calls.get(), 0);
        assert!(recorder.events().is_empty());
    }
}
