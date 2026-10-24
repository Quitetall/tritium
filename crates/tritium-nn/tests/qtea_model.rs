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
        let manifest = encode_qtea_model_manifest(&self.manifest, 1 << 16).unwrap();
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
