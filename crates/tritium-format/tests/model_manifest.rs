//! Typed model metadata validation, not architecture or model-quality qualification.

use ciborium::value::{CanonicalValue, Value};
use half::{bf16, f16};
use proptest::prelude::*;
use tritium_core::{DType, Trit};
use tritium_format::trit_model::{
    TritModelError, decode_model_manifest, encode_model_manifest, read_trit_model,
};
use tritium_format::{
    AdditiveTensor, TritBlob, TritBlobKind, read_trit_package, write_trit_package,
};
use tritium_schema::{
    AdditiveLayout, Basis, BlobId, BlobReference, EvidenceReference, LegacyModelIdentity, Level,
    LevelFit, ManifestAsset, ModelManifest, ModelManifestError, ModelShard, PackageId,
    PlaneAllocation, PlaneCodec, PlaneCounts, PlaneRelation, ScaleAnchor, ScaleLaw, ScalePrecision,
    SchemaId, TensorMeta, TensorStorage, Transport,
};

const TENSOR: &str = "model.embed_tokens.weight";
const ASSET: &[u8] = b"{}"; // Structural fixture, not an executable model config/tokenizer.

#[test]
fn candidate_semantic_ids_ignore_physical_codec_and_level_aliases() {
    let mut expected = None;
    let mut physical = Vec::new();
    for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
        let (tensor, payload) = additive(codec, ScalePrecision::F32, Basis::Identity, 2);
        let mut manifest = fixture();
        manifest.levels[0].tensors = vec![tensor];
        let bytes = package(
            &manifest,
            &[TritBlob {
                kind: TritBlobKind::AdditiveTensor,
                bytes: &payload,
            }],
        );
        let model = read_trit_model(&bytes).unwrap();
        let identity = (
            model.tensor_digest("imported", TENSOR).unwrap(),
            model.model_id("imported").unwrap(),
        );
        assert_eq!(
            blake3::Hash::from_bytes(*identity.0.as_bytes())
                .to_hex()
                .as_str(),
            "fc19a72d5206c01988147cd77700b8d6a7f1ae8977c0321d16763220e89b69cd"
        );
        assert_eq!(
            blake3::Hash::from_bytes(*identity.1.as_bytes())
                .to_hex()
                .as_str(),
            "a396a7331dadad1cf59f96df549cf551784ca5a19f5c6e9990c1f66d6fcbaf5f"
        );
        assert_eq!(*expected.get_or_insert(identity), identity);
        physical.push(model.package_id());
        let mut alias = manifest.levels[0].clone();
        alias.name = "same-content".into();
        alias.fit = LevelFit::Direct {
            evidence: vec![EvidenceReference {
                run: "candidate".into(),
                span: "fit".into(),
                root: [3; 32],
            }],
        };
        manifest.levels.push(alias);
        let aliased_bytes = package(
            &manifest,
            &[TritBlob {
                kind: TritBlobKind::AdditiveTensor,
                bytes: &payload,
            }],
        );
        let aliased = read_trit_model(&aliased_bytes).unwrap();
        assert_eq!(aliased.model_id("same-content").unwrap(), identity.1);
        assert_eq!(aliased.model_id("imported").unwrap(), identity.1);
        assert!(aliased.tensor_digest("missing", TENSOR).is_err());
        assert!(aliased.model_id("missing").is_err());
    }
    assert_ne!(physical[0], physical[1]);
    assert_ne!(physical[1], physical[2]);
}

#[test]
fn candidate_additive_digest_binds_content_not_capacity_or_transport() {
    use tritium_format::trit_model::additive_semantic_digest;
    let (meta, bytes) = additive(PlaneCodec::D2, ScalePrecision::F32, Basis::Identity, 2);
    let TensorStorage::Additive { layout, .. } = meta.storage else {
        unreachable!()
    };
    let base = AdditiveTensor::decode_payload(layout, 2, &bytes).unwrap();
    let expected = additive_semantic_digest(&base).unwrap();
    for change in [
        "trit",
        "scale",
        "basis",
        "seed",
        "precision",
        "group",
        "shape",
        "planes",
        "capacity",
        "transport",
    ] {
        let mut layout = layout;
        let mut trits = base.trits().to_vec();
        let mut scales = base.scales().to_vec();
        let mut planes = 2;
        match change {
            "trit" => trits[0] = Trit::NEG,
            "scale" => scales[0] = f32::from_bits(scales[0].to_bits() + 1),
            "basis" => layout.basis = Basis::Hadamard { block: 4 },
            "seed" => {
                layout.basis = Basis::SignedRht {
                    block: 4,
                    seed: 7,
                    domain: 9,
                }
            }
            "precision" => layout.law.precision = ScalePrecision::F16,
            "group" => layout.group = 128,
            "shape" => {
                layout.rows = 4;
                layout.cols = 2;
                scales = vec![0.5; 8];
            }
            "planes" => {
                planes = 1;
                trits.truncate(8);
                scales.truncate(2);
            }
            "capacity" => layout.max_planes = 3,
            "transport" => layout.transport = Transport::JointHuffman { block: 64 },
            _ => unreachable!(),
        }
        let changed = AdditiveTensor::new(layout, planes, trits, scales).unwrap();
        let actual = additive_semantic_digest(&changed).unwrap();
        if matches!(change, "capacity" | "transport") {
            assert_eq!(actual, expected, "{change}");
        } else {
            assert_ne!(actual, expected, "{change}");
        }
    }
    let mut rotated_layout = layout;
    rotated_layout.basis = Basis::SignedRht {
        block: 4,
        seed: 7,
        domain: 9,
    };
    let rotated = AdditiveTensor::new(
        rotated_layout,
        2,
        base.trits().to_vec(),
        base.scales().to_vec(),
    )
    .unwrap();
    let rotated_id = additive_semantic_digest(&rotated).unwrap();
    for basis in [
        Basis::SignedRht {
            block: 2,
            seed: 7,
            domain: 9,
        },
        Basis::SignedRht {
            block: 4,
            seed: 8,
            domain: 9,
        },
        Basis::SignedRht {
            block: 4,
            seed: 7,
            domain: 10,
        },
    ] {
        rotated_layout.basis = basis;
        let changed = AdditiveTensor::new(
            rotated_layout,
            2,
            base.trits().to_vec(),
            base.scales().to_vec(),
        )
        .unwrap();
        assert_ne!(additive_semantic_digest(&changed).unwrap(), rotated_id);
    }
}

#[test]
fn candidate_model_digest_binds_architecture_assets_and_logical_names() {
    let base = fixture();
    let payload = [0; 32];
    let packed = package(
        &base,
        &[TritBlob {
            kind: TritBlobKind::DenseTensor(DType::F32),
            bytes: &payload,
        }],
    );
    let expected = read_trit_model(&packed)
        .unwrap()
        .model_id("imported")
        .unwrap();
    for change in [
        "architecture",
        "tensor-name",
        "asset-bytes",
        "asset-name",
        "shape",
        "dense-bits",
        "legacy-alias",
        "physical-order",
    ] {
        let mut manifest = base.clone();
        let mut payload = payload;
        let mut config = ASSET.to_vec();
        match change {
            "architecture" => manifest.architecture = "qwen3".into(),
            "tensor-name" => manifest.levels[0].tensors[0].name = "different.weight".into(),
            "asset-bytes" => {
                config = b"{ }".to_vec();
                manifest.assets[0].blob = reference(&config);
            }
            "asset-name" => manifest.assets.push(ManifestAsset {
                name: "template.json".into(),
                blob: reference(ASSET),
            }),
            "shape" => {
                manifest.levels[0].tensors[0].storage = TensorStorage::Dense {
                    shape: vec![4, 2],
                    precision: ScalePrecision::F32,
                }
            }
            "dense-bits" => {
                payload[3] = 0x80;
                manifest.levels[0].tensors[0].blob = reference(&payload);
            } // exact negative zero
            "legacy-alias" => manifest.legacy_ids.push(LegacyModelIdentity {
                id: "legacy:fixture".into(),
                level: "imported".into(),
            }),
            _ => {}
        }
        let mut blobs = vec![
            TritBlob {
                kind: TritBlobKind::Asset("config.json".into()),
                bytes: &config,
            },
            TritBlob {
                kind: TritBlobKind::Asset("tokenizer.json".into()),
                bytes: ASSET,
            },
            TritBlob {
                kind: TritBlobKind::DenseTensor(DType::F32),
                bytes: &payload,
            },
        ];
        if change == "asset-name" {
            blobs.push(TritBlob {
                kind: TritBlobKind::Asset("template.json".into()),
                bytes: ASSET,
            });
        }
        if change == "physical-order" {
            blobs.reverse();
        }
        let packed =
            write_trit_package(&encode_model_manifest(&manifest).unwrap(), &blobs).unwrap();
        let actual = read_trit_model(&packed)
            .unwrap()
            .model_id("imported")
            .unwrap();
        if matches!(change, "legacy-alias" | "physical-order") {
            assert_eq!(actual, expected, "{change}");
        } else {
            assert_ne!(actual, expected, "{change}");
        }
    }
}

#[test]
fn candidate_dense_and_tied_scale_vectors_match_independent_preimages() {
    let bytes = package(
        &fixture(),
        &[TritBlob {
            kind: TritBlobKind::DenseTensor(DType::F32),
            bytes: &[0; 32],
        }],
    );
    let model = read_trit_model(&bytes).unwrap();
    assert_eq!(
        blake3::Hash::from_bytes(*model.tensor_digest("imported", TENSOR).unwrap().as_bytes())
            .to_hex()
            .as_str(),
        "b19646f6931fcd0dab8b73d14f624cdc542865cddd9a6a40d9c798727fa7a568"
    );
    let (meta, raw) = additive(PlaneCodec::D2, ScalePrecision::F32, Basis::Identity, 3);
    let TensorStorage::Additive { mut layout, .. } = meta.storage else {
        unreachable!()
    };
    let original = AdditiveTensor::decode_payload(layout, 3, &raw).unwrap();
    layout.law.precision = ScalePrecision::F16;
    layout.law.relation = PlaneRelation::Tied { num: 1, den: 3 };
    layout.basis = Basis::SignedRht {
        block: 4,
        seed: 7,
        domain: 9,
    };
    let tied = AdditiveTensor::new(layout, 3, original.trits().to_vec(), vec![0.5, 1.]).unwrap();
    assert_eq!(
        blake3::Hash::from_bytes(
            *tritium_format::trit_model::additive_semantic_digest(&tied)
                .unwrap()
                .as_bytes()
        )
        .to_hex()
        .as_str(),
        "54b0d7099eb3e68572e14a45160b666e70e9d945a827d685f0d331bbe599bdf4"
    );
}

#[test]
fn candidate_model_identity_is_per_level_not_a_whole_package_digest() {
    let base = fixture();
    let payload = [0; 32];
    let base_bytes = package(
        &base,
        &[TritBlob {
            kind: TritBlobKind::DenseTensor(DType::F32),
            bytes: &payload,
        }],
    );
    let expected = read_trit_model(&base_bytes)
        .unwrap()
        .model_id("imported")
        .unwrap();
    let mut expanded = base;
    let mut other = expanded.levels[0].clone();
    other.name = "different-content".into();
    let other_payload = [0.5f32.to_le_bytes(); 8].concat();
    other.tensors[0].blob = reference(&other_payload);
    expanded.levels.push(other);
    let bytes = package(
        &expanded,
        &[
            TritBlob {
                kind: TritBlobKind::DenseTensor(DType::F32),
                bytes: &payload,
            },
            TritBlob {
                kind: TritBlobKind::DenseTensor(DType::F32),
                bytes: &other_payload,
            },
        ],
    );
    let model = read_trit_model(&bytes).unwrap();
    assert_eq!(model.model_id("imported").unwrap(), expected);
    assert_ne!(model.model_id("different-content").unwrap(), expected);
}

fn reference(bytes: &[u8]) -> BlobReference {
    BlobReference {
        id: BlobId::from_bytes(*blake3::hash(bytes).as_bytes()),
        shard: None,
    }
}

fn fixture() -> ModelManifest {
    ModelManifest {
        schema: SchemaId { major: 1, minor: 0 },
        architecture: "llama".into(),
        assets: vec![
            ManifestAsset {
                name: "config.json".into(),
                blob: reference(ASSET),
            },
            ManifestAsset {
                name: "tokenizer.json".into(),
                blob: reference(ASSET),
            },
        ],
        levels: vec![Level {
            name: "imported".into(),
            fit: LevelFit::Imported,
            tensors: vec![TensorMeta {
                name: TENSOR.into(),
                blob: reference(&[0; 32]),
                storage: TensorStorage::Dense {
                    shape: vec![2, 4],
                    precision: ScalePrecision::F32,
                },
            }],
        }],
        shards: vec![],
        legacy_ids: vec![],
    }
}

#[test]
fn typed_manifest_has_one_canonical_encoding_and_strict_roundtrip() {
    let manifest = fixture();
    let canonical = encode_model_manifest(&manifest).unwrap();
    assert_eq!(decode_model_manifest(&canonical).unwrap(), manifest);
    let mut reordered = manifest.clone();
    reordered.assets.reverse();
    assert_eq!(encode_model_manifest(&reordered).unwrap(), canonical);
    let mut bad = manifest;
    bad.assets.push(bad.assets[0].clone());
    assert!(encode_model_manifest(&bad).is_err());
}

fn package(manifest: &ModelManifest, tensors: &[TritBlob<'_>]) -> Vec<u8> {
    let mut blobs = vec![
        TritBlob {
            kind: TritBlobKind::Asset("config.json".into()),
            bytes: ASSET,
        },
        TritBlob {
            kind: TritBlobKind::Asset("tokenizer.json".into()),
            bytes: ASSET,
        },
    ];
    blobs.extend_from_slice(tensors);
    write_trit_package(&encode_model_manifest(manifest).unwrap(), &blobs).unwrap()
}

fn additive(
    codec: PlaneCodec,
    precision: ScalePrecision,
    basis: Basis,
    planes: u8,
) -> (TensorMeta, Vec<u8>) {
    let layout = AdditiveLayout {
        rows: 2,
        cols: 4,
        tile: 256,
        group: 32,
        max_planes: planes,
        allocation: PlaneAllocation::Uniform,
        codec,
        law: ScaleLaw {
            anchor: ScaleAnchor::Group,
            relation: PlaneRelation::Free,
            precision,
        },
        basis,
        transport: Transport::Raw,
    };
    let trits = [
        Trit::POS,
        Trit::ZERO,
        Trit::NEG,
        Trit::POS,
        Trit::NEG,
        Trit::ZERO,
        Trit::POS,
        Trit::NEG,
    ]
    .repeat(usize::from(planes));
    let bytes = AdditiveTensor::new(layout, planes, trits, vec![0.5; 2 * usize::from(planes)])
        .unwrap()
        .encode_payload()
        .unwrap();
    (
        TensorMeta {
            name: TENSOR.into(),
            blob: reference(&bytes),
            storage: TensorStorage::Additive {
                layout,
                planes: PlaneCounts::Uniform(planes),
            },
        },
        bytes,
    )
}

fn cbor(mut value: Value) -> Vec<u8> {
    // Construct hostile inputs through the public decoder, not private validators.
    fn order(value: &mut Value) {
        match value {
            Value::Map(entries) => {
                for (key, value) in entries.iter_mut() {
                    order(key);
                    order(value);
                }
                entries.sort_by_cached_key(|e| CanonicalValue::from(e.0.clone()));
            }
            Value::Array(items) => items.iter_mut().for_each(order),
            Value::Tag(_, inner) => order(inner),
            _ => {}
        }
    }
    order(&mut value);
    let mut bytes = vec![];
    ciborium::ser::into_writer(&value, &mut bytes).unwrap();
    bytes
}

fn field<'a>(value: &'a mut Value, key: &str) -> &'a mut Value {
    let Value::Map(entries) = value else {
        panic!("expected map")
    };
    &mut entries
        .iter_mut()
        .find(|e| e.0 == Value::Text(key.into()))
        .unwrap()
        .1
}

#[test]
fn dense_precisions_bind_exact_dtype_finite_bytes_and_borrow_source() {
    for (precision, dtype, bytes) in [
        (
            ScalePrecision::F16,
            DType::F16,
            f16::from_f32(-0.5).to_bits().to_le_bytes().repeat(8),
        ),
        (
            ScalePrecision::Bf16,
            DType::BF16,
            bf16::from_f32(-0.5).to_bits().to_le_bytes().repeat(8),
        ),
        (
            ScalePrecision::F32,
            DType::F32,
            (-0.5f32).to_le_bytes().repeat(8),
        ),
    ] {
        let mut manifest = fixture();
        manifest.levels[0].tensors[0].storage = TensorStorage::Dense {
            shape: vec![2, 4],
            precision,
        };
        manifest.levels[0].tensors[0].blob = reference(&bytes);
        let mut shared = manifest.levels[0].clone();
        shared.name = "same".into();
        manifest.levels.push(shared);
        let source = package(
            &manifest,
            &[TritBlob {
                kind: TritBlobKind::DenseTensor(dtype),
                bytes: &bytes,
            }],
        );
        let model = read_trit_model(&source).unwrap();
        let (_, a) = model.tensor("imported", TENSOR).unwrap();
        let (_, b) = model.tensor("same", TENSOR).unwrap();
        assert_eq!(a, bytes);
        assert!(core::ptr::eq(a.as_ptr(), b.as_ptr()));
        assert!(source.as_ptr_range().contains(&a.as_ptr()));
        assert_eq!(model.asset("config.json").unwrap(), ASSET);
        assert_eq!(
            model.package_id(),
            tritium_format::PackageId::from_package_bytes(&source)
        );
        assert_eq!(read_trit_package(&source).unwrap().blobs().len(), 3);
        assert!(matches!(
            model.tensor("absent", TENSOR),
            Err(TritModelError::MissingBinding(_))
        ));
        assert!(matches!(
            model.tensor("imported", "absent"),
            Err(TritModelError::MissingBinding(_))
        ));
        assert!(matches!(
            model.asset("absent"),
            Err(TritModelError::MissingBinding(_))
        ));
    }
}

#[test]
fn missing_or_incompatible_directory_bindings_are_rejected() {
    let manifest = fixture();
    for kind in [
        TritBlobKind::DenseTensor(DType::F16),
        TritBlobKind::AdditiveTensor,
        TritBlobKind::Asset(TENSOR.into()),
    ] {
        let source = package(
            &manifest,
            &[TritBlob {
                kind,
                bytes: &[0; 32],
            }],
        );
        assert!(matches!(
            read_trit_model(&source),
            Err(TritModelError::MissingOrWrongBlob(_))
        ));
    }
    assert!(matches!(
        read_trit_model(&package(&manifest, &[])),
        Err(TritModelError::MissingOrWrongBlob(_))
    ));
    let source = write_trit_package(
        &encode_model_manifest(&manifest).unwrap(),
        &[
            TritBlob {
                kind: TritBlobKind::Asset("wrong.json".into()),
                bytes: ASSET,
            },
            TritBlob {
                kind: TritBlobKind::DenseTensor(DType::F32),
                bytes: &[0; 32],
            },
        ],
    )
    .unwrap();
    assert!(matches!(
        read_trit_model(&source),
        Err(TritModelError::MissingOrWrongBlob(_))
    ));
}

#[test]
fn all_dense_precisions_reject_nonfinite_and_wrong_length_payloads() {
    for (precision, dtype, invalid) in [
        (
            ScalePrecision::F16,
            DType::F16,
            f16::NAN.to_bits().to_le_bytes().to_vec(),
        ),
        (
            ScalePrecision::Bf16,
            DType::BF16,
            bf16::INFINITY.to_bits().to_le_bytes().to_vec(),
        ),
        (
            ScalePrecision::F32,
            DType::F32,
            f32::NEG_INFINITY.to_le_bytes().to_vec(),
        ),
    ] {
        for bytes in [invalid.repeat(8), vec![0; 1], invalid.repeat(9)] {
            let mut manifest = fixture();
            manifest.levels[0].tensors[0].storage = TensorStorage::Dense {
                shape: vec![2, 4],
                precision,
            };
            manifest.levels[0].tensors[0].blob = reference(&bytes);
            let source = package(
                &manifest,
                &[TritBlob {
                    kind: TritBlobKind::DenseTensor(dtype),
                    bytes: &bytes,
                }],
            );
            assert!(matches!(
                read_trit_model(&source),
                Err(TritModelError::DensePayload(_))
            ));
        }
    }
}

#[test]
fn additive_codecs_precisions_and_bases_reuse_canonical_payload_validation() {
    for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
        for precision in [ScalePrecision::F16, ScalePrecision::F32] {
            for basis in [
                Basis::Identity,
                Basis::Hadamard { block: 4 },
                Basis::SignedRht {
                    block: 4,
                    seed: 7,
                    domain: 9,
                },
            ] {
                let (meta, bytes) = additive(codec, precision, basis, 3);
                let mut manifest = fixture();
                manifest.levels[0].tensors[0] = meta.clone();
                let source = package(
                    &manifest,
                    &[TritBlob {
                        kind: TritBlobKind::AdditiveTensor,
                        bytes: &bytes,
                    }],
                );
                let model = read_trit_model(&source).unwrap();
                assert_eq!(
                    model.tensor("imported", TENSOR).unwrap(),
                    (&meta, bytes.as_slice())
                );
                let mut trailing = bytes;
                trailing.push(0);
                manifest.levels[0].tensors[0].blob = reference(&trailing);
                let source = package(
                    &manifest,
                    &[TritBlob {
                        kind: TritBlobKind::AdditiveTensor,
                        bytes: &trailing,
                    }],
                );
                assert!(matches!(
                    read_trit_model(&source),
                    Err(TritModelError::Additive(_))
                ));
            }
        }
    }
}

#[test]
fn additive_rejects_invalid_codes_and_exact_scale_bits_after_hash_rebinding() {
    let (meta, bytes) = additive(PlaneCodec::D2, ScalePrecision::F32, Basis::Identity, 1);
    for scale in [f32::NAN, -0.0, -1.0] {
        let mut corrupt = bytes.clone();
        let end = corrupt.len();
        corrupt[end - 4..].copy_from_slice(&scale.to_le_bytes());
        let mut manifest = fixture();
        manifest.levels[0].tensors[0] = meta.clone();
        manifest.levels[0].tensors[0].blob = reference(&corrupt);
        let source = package(
            &manifest,
            &[TritBlob {
                kind: TritBlobKind::AdditiveTensor,
                bytes: &corrupt,
            }],
        );
        assert!(matches!(
            read_trit_model(&source),
            Err(TritModelError::Additive(_))
        ));
    }
    let mut corrupt = bytes;
    corrupt[0] |= 3;
    let mut manifest = fixture();
    manifest.levels[0].tensors[0] = meta;
    manifest.levels[0].tensors[0].blob = reference(&corrupt);
    let source = package(
        &manifest,
        &[TritBlob {
            kind: TritBlobKind::AdditiveTensor,
            bytes: &corrupt,
        }],
    );
    assert!(matches!(
        read_trit_model(&source),
        Err(TritModelError::Additive(_))
    ));
}

#[test]
fn imported_prefix_compares_logical_planes_and_exact_scales_not_codec_bytes() {
    let (child, child_bytes) = additive(PlaneCodec::B3, ScalePrecision::F16, Basis::Identity, 1);
    let (parent, parent_bytes) = additive(PlaneCodec::D2, ScalePrecision::F16, Basis::Identity, 3);
    let mut manifest = fixture();
    manifest.levels = vec![
        Level {
            name: "child".into(),
            fit: LevelFit::ImportedPrefixOf {
                level: "parent".into(),
            },
            tensors: vec![child],
        },
        Level {
            name: "parent".into(),
            fit: LevelFit::Imported,
            tensors: vec![parent],
        },
    ];
    let source = package(
        &manifest,
        &[
            TritBlob {
                kind: TritBlobKind::AdditiveTensor,
                bytes: &child_bytes,
            },
            TritBlob {
                kind: TritBlobKind::AdditiveTensor,
                bytes: &parent_bytes,
            },
        ],
    );
    assert!(read_trit_model(&source).is_ok());
    for corrupt_scale in [false, true] {
        let TensorStorage::Additive { layout, .. } = manifest.levels[0].tensors[0].storage else {
            unreachable!()
        };
        let original = AdditiveTensor::decode_payload(layout, 1, &child_bytes).unwrap();
        let mut trits = original.trits().to_vec();
        let mut scales = original.scales().to_vec();
        if corrupt_scale {
            scales[0] = 0.25;
        } else {
            trits[0] = Trit::NEG;
        }
        let false_prefix = AdditiveTensor::new(layout, 1, trits, scales)
            .unwrap()
            .encode_payload()
            .unwrap();
        manifest.levels[0].tensors[0].blob = reference(&false_prefix);
        let source = package(
            &manifest,
            &[
                TritBlob {
                    kind: TritBlobKind::AdditiveTensor,
                    bytes: &false_prefix,
                },
                TritBlob {
                    kind: TritBlobKind::AdditiveTensor,
                    bytes: &parent_bytes,
                },
            ],
        );
        assert!(
            matches!(read_trit_model(&source), Err(TritModelError::InvalidPrefix(level)) if level == "child")
        );
    }
}

#[test]
fn dense_prefix_must_be_identical_and_every_complete_level_is_checked() {
    let mut manifest = fixture();
    let mut parent = manifest.levels[0].clone();
    parent.name = "parent".into();
    manifest.levels[0].fit = LevelFit::ImportedPrefixOf {
        level: "parent".into(),
    };
    manifest.levels.push(parent);
    let blobs = [TritBlob {
        kind: TritBlobKind::DenseTensor(DType::F32),
        bytes: &[0; 32],
    }];
    assert!(read_trit_model(&package(&manifest, &blobs)).is_ok());
    let different = 1f32.to_le_bytes().repeat(8);
    manifest.levels[1].tensors[0].blob = reference(&different);
    let mut blobs = blobs.to_vec();
    blobs.push(TritBlob {
        kind: TritBlobKind::DenseTensor(DType::F32),
        bytes: &different,
    });
    assert!(matches!(
        read_trit_model(&package(&manifest, &blobs)),
        Err(TritModelError::InvalidPrefix(_))
    ));
    manifest.levels[0].fit = LevelFit::Imported;
    assert!(read_trit_model(&package(&manifest, &blobs)).is_ok());
    manifest.levels[1].tensors[0].blob = reference(b"absent");
    assert!(matches!(
        read_trit_model(&package(&manifest, &blobs)),
        Err(TritModelError::MissingOrWrongBlob(_))
    ));
}

#[test]
fn invalid_shape_version_inventory_and_required_assets_fail_structurally() {
    let original = fixture();
    for shape in [vec![], vec![0, 4], vec![u64::MAX, 2], vec![1; 33]] {
        let mut manifest = original.clone();
        manifest.levels[0].tensors[0].storage = TensorStorage::Dense {
            shape,
            precision: ScalePrecision::F32,
        };
        assert_eq!(manifest.validate(), Err(ModelManifestError::InvalidShape));
    }
    let mut manifest = original.clone();
    manifest.schema.minor = 1;
    assert!(matches!(
        manifest.validate(),
        Err(ModelManifestError::UnsupportedVersion(_))
    ));
    for name in [
        "",
        "../config.json",
        ".",
        "..",
        "a/b",
        "a\\b",
        "a:b",
        "nul\0.json",
    ] {
        let mut manifest = original.clone();
        manifest.assets.push(ManifestAsset {
            name: name.into(),
            blob: reference(ASSET),
        });
        assert!(encode_model_manifest(&manifest).is_err());
    }
    let mut manifest = original.clone();
    manifest.assets.remove(0);
    assert_eq!(
        manifest.validate(),
        Err(ModelManifestError::MissingAsset("config.json"))
    );
    let mut manifest = original.clone();
    manifest.levels.clear();
    assert_eq!(manifest.validate(), Err(ModelManifestError::InvalidCount));
    let mut manifest = original;
    manifest.levels[0].tensors.clear();
    assert_eq!(manifest.validate(), Err(ModelManifestError::InvalidCount));
}

#[test]
fn complete_levels_noncyclic_prefixes_and_legacy_references_are_required() {
    let original = fixture();
    for parent in ["imported", "absent"] {
        let mut manifest = original.clone();
        manifest.levels[0].fit = LevelFit::ImportedPrefixOf {
            level: parent.into(),
        };
        assert_eq!(manifest.validate(), Err(ModelManifestError::InvalidPrefix));
    }
    let mut manifest = original.clone();
    let mut second = manifest.levels[0].clone();
    second.name = "second".into();
    manifest.levels.push(second);
    manifest.levels[0].fit = LevelFit::ImportedPrefixOf {
        level: "second".into(),
    };
    manifest.levels[1].fit = LevelFit::ImportedPrefixOf {
        level: "imported".into(),
    };
    assert_eq!(manifest.validate(), Err(ModelManifestError::InvalidPrefix));
    manifest
        .levels
        .iter_mut()
        .for_each(|l| l.fit = LevelFit::Imported);
    manifest.levels[1].tensors[0].name = "other".into();
    assert_eq!(
        manifest.validate(),
        Err(ModelManifestError::LevelShapeMismatch)
    );
    manifest.levels[1].tensors[0].name = TENSOR.into();
    manifest.levels[1].tensors[0].storage = TensorStorage::Dense {
        shape: vec![4, 2],
        precision: ScalePrecision::F32,
    };
    assert_eq!(
        manifest.validate(),
        Err(ModelManifestError::LevelShapeMismatch)
    );
    let mut manifest = original;
    manifest.legacy_ids = vec![LegacyModelIdentity {
        id: "old:012345".into(),
        level: "absent".into(),
    }];
    assert_eq!(
        manifest.validate(),
        Err(ModelManifestError::MissingLegacyLevel)
    );
    manifest.legacy_ids[0].level = "imported".into();
    assert!(manifest.validate().is_ok());
}

#[test]
fn direct_fit_requires_canonical_nonempty_ancestry_not_a_verdict() {
    let mut manifest = fixture();
    manifest.levels[0].fit = LevelFit::Direct { evidence: vec![] };
    assert_eq!(manifest.validate(), Err(ModelManifestError::InvalidCount));
    let a = EvidenceReference {
        run: "run-a".into(),
        span: "fit".into(),
        root: [1; 32],
    };
    let b = EvidenceReference {
        run: "run-b".into(),
        span: "fit".into(),
        root: [2; 32],
    };
    manifest.levels[0].fit = LevelFit::Direct {
        evidence: vec![b.clone(), a.clone()],
    };
    let canonical = decode_model_manifest(&encode_model_manifest(&manifest).unwrap()).unwrap();
    assert_eq!(
        canonical.levels[0].fit,
        LevelFit::Direct {
            evidence: vec![a.clone(), b]
        }
    );
    manifest.levels[0].fit = LevelFit::Direct {
        evidence: vec![a.clone(), a],
    };
    assert!(encode_model_manifest(&manifest).is_err());
}

#[test]
fn unsupported_payload_capabilities_are_representable_but_not_silently_loaded() {
    let (meta, bytes) = additive(PlaneCodec::D2, ScalePrecision::F32, Basis::Identity, 3);
    for capability in [
        "per-tile model payloads",
        "entropy model payloads",
        "sharded model payloads",
    ] {
        let mut manifest = fixture();
        manifest.levels[0].tensors[0] = meta.clone();
        match capability {
            "per-tile model payloads" => {
                let TensorStorage::Additive { layout, planes } =
                    &mut manifest.levels[0].tensors[0].storage
                else {
                    unreachable!()
                };
                layout.allocation = PlaneAllocation::PerTile;
                *planes = PlaneCounts::PerTile(vec![0, 3]);
            }
            "entropy model payloads" => {
                let TensorStorage::Additive { layout, .. } =
                    &mut manifest.levels[0].tensors[0].storage
                else {
                    unreachable!()
                };
                layout.transport = Transport::JointHuffman { block: 256 };
            }
            _ => {
                manifest.shards.push(ModelShard {
                    name: "weights-01.trit".into(),
                    package: PackageId::from_bytes([7; 32]),
                });
                manifest.levels[0].tensors[0].blob.shard = Some("weights-01.trit".into());
            }
        }
        assert_eq!(
            decode_model_manifest(&encode_model_manifest(&manifest).unwrap()).unwrap(),
            manifest
        );
        let source = package(
            &manifest,
            &[TritBlob {
                kind: TritBlobKind::AdditiveTensor,
                bytes: &bytes,
            }],
        );
        assert!(
            matches!(read_trit_model(&source), Err(TritModelError::UnsupportedFeature(name)) if name == capability)
        );
    }
}

#[test]
fn plane_allocation_and_shard_references_match_declared_geometry() {
    let (meta, _) = additive(PlaneCodec::D2, ScalePrecision::F16, Basis::Identity, 3);
    let mut manifest = fixture();
    manifest.levels[0].tensors[0] = meta;
    let original = manifest.clone();
    for planes in [
        PlaneCounts::Uniform(0),
        PlaneCounts::Uniform(2),
        PlaneCounts::PerTile(vec![3, 3]),
    ] {
        let TensorStorage::Additive { planes: field, .. } =
            &mut manifest.levels[0].tensors[0].storage
        else {
            unreachable!()
        };
        *field = planes;
        assert_eq!(manifest.validate(), Err(ModelManifestError::InvalidPlanes));
    }
    manifest = original;
    let TensorStorage::Additive { layout, planes } = &mut manifest.levels[0].tensors[0].storage
    else {
        unreachable!()
    };
    layout.allocation = PlaneAllocation::PerTile;
    *planes = PlaneCounts::PerTile(vec![3, 4]);
    assert_eq!(manifest.validate(), Err(ModelManifestError::InvalidPlanes));
    manifest.levels[0].tensors[0].blob.shard = Some("absent.trit".into());
    assert_eq!(manifest.validate(), Err(ModelManifestError::MissingShard));
}

#[test]
fn strict_cbor_rejects_unknown_fields_tags_missing_fields_and_unsorted_lists() {
    let canonical = encode_model_manifest(&fixture()).unwrap();
    let original: Value = ciborium::de::from_reader(canonical.as_slice()).unwrap();
    let mut unknown = original.clone();
    let Value::Map(entries) = &mut unknown else {
        unreachable!()
    };
    entries.push((Value::Text("qualified".into()), Value::Bool(true)));
    assert!(matches!(
        decode_model_manifest(&cbor(unknown)),
        Err(TritModelError::Schema(_))
    ));
    let mut missing = original.clone();
    let Value::Map(entries) = &mut missing else {
        unreachable!()
    };
    entries.retain(|e| e.0 != Value::Text("levels".into()));
    assert!(matches!(
        decode_model_manifest(&cbor(missing)),
        Err(TritModelError::Schema(_))
    ));
    assert!(decode_model_manifest(&cbor(Value::Tag(123, Box::new(original.clone())))).is_err());
    let mut unsorted = original.clone();
    let Value::Array(assets) = field(&mut unsorted, "assets") else {
        unreachable!()
    };
    assets.reverse();
    assert!(matches!(
        decode_model_manifest(&cbor(unsorted)),
        Err(TritModelError::Manifest(
            ModelManifestError::UnsortedOrDuplicate("assets")
        ))
    ));
    let mut duplicate = original;
    let Value::Map(entries) = &mut duplicate else {
        unreachable!()
    };
    entries.push(entries[0].clone());
    assert!(matches!(
        decode_model_manifest(&cbor(duplicate)),
        Err(TritModelError::Package(_))
    ));
    let mut trailing = canonical;
    trailing.push(0);
    assert!(matches!(
        decode_model_manifest(&trailing),
        Err(TritModelError::Package(_))
    ));
}

#[test]
fn remaining_admitted_scale_laws_and_dense_norm_shapes_are_checked() {
    for law in [
        ScaleLaw {
            anchor: ScaleAnchor::Group,
            relation: PlaneRelation::Tied { num: 1, den: 3 },
            precision: ScalePrecision::F16,
        },
        ScaleLaw {
            anchor: ScaleAnchor::Tensor,
            relation: PlaneRelation::Free,
            precision: ScalePrecision::F32,
        },
    ] {
        let planes = tritium_schema::admitted_law(law).unwrap().max_planes.min(3);
        let (mut meta, _) = additive(PlaneCodec::S34, law.precision, Basis::Identity, planes);
        let TensorStorage::Additive { layout, .. } = &mut meta.storage else {
            unreachable!()
        };
        layout.law = law;
        let scales = if matches!(law.relation, PlaneRelation::Tied { .. }) {
            vec![0.5; 2]
        } else {
            vec![0.5; usize::from(planes)]
        };
        let trits = [Trit::POS, Trit::ZERO, Trit::NEG, Trit::POS].repeat(2 * usize::from(planes));
        let bytes = AdditiveTensor::new(*layout, planes, trits, scales)
            .unwrap()
            .encode_payload()
            .unwrap();
        meta.blob = reference(&bytes);
        let mut manifest = fixture();
        manifest.levels[0].tensors[0] = meta;
        let source = package(
            &manifest,
            &[TritBlob {
                kind: TritBlobKind::AdditiveTensor,
                bytes: &bytes,
            }],
        );
        assert!(read_trit_model(&source).is_ok());
    }
    let mut manifest = fixture();
    manifest.levels[0].tensors[0].storage = TensorStorage::Dense {
        shape: vec![8],
        precision: ScalePrecision::F32,
    };
    let source = package(
        &manifest,
        &[TritBlob {
            kind: TritBlobKind::DenseTensor(DType::F32),
            bytes: &[0; 32],
        }],
    );
    assert!(read_trit_model(&source).is_ok());
}

#[test]
fn nested_unknown_fields_variants_and_optional_aliases_fail_closed() {
    let canonical = encode_model_manifest(&fixture()).unwrap();
    let original: Value = ciborium::de::from_reader(canonical.as_slice()).unwrap();
    let mut nested = original.clone();
    let Value::Map(schema) = field(&mut nested, "schema") else {
        unreachable!()
    };
    schema.push((Value::Text("ignored".into()), Value::Bool(true)));
    assert!(matches!(
        decode_model_manifest(&cbor(nested)),
        Err(TritModelError::Schema(_))
    ));
    let mut alias = original.clone();
    let Value::Array(assets) = field(&mut alias, "assets") else {
        unreachable!()
    };
    let Value::Map(blob) = field(&mut assets[0], "blob") else {
        unreachable!()
    };
    blob.retain(|e| e.0 != Value::Text("shard".into()));
    // serde can infer None; canonical typed decoding must still reject that spelling.
    assert!(matches!(
        decode_model_manifest(&cbor(alias)),
        Err(TritModelError::NonCanonical)
    ));
    let mut unknown = original;
    let Value::Array(levels) = field(&mut unknown, "levels") else {
        unreachable!()
    };
    *field(&mut levels[0], "fit") = Value::Text("FutureFit".into());
    assert!(matches!(
        decode_model_manifest(&cbor(unknown)),
        Err(TritModelError::Schema(_))
    ));
    let mut deeply_nested = Value::Null;
    for _ in 0..100 {
        deeply_nested = Value::Array(vec![deeply_nested]);
    }
    assert!(matches!(
        decode_model_manifest(&cbor(deeply_nested)),
        Err(TritModelError::Package(_))
    ));
}

proptest! {
    #[test]
    fn arbitrary_short_metadata_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..4096)) {
        let _ = decode_model_manifest(&bytes);
    }
}
