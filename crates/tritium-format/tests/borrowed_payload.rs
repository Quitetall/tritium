//! Payload/identity compatibility, not model quality or native execution evidence.
use ciborium::value::{CanonicalValue, Value};
use tritium_core::{PerTilePlan, Trit};
use tritium_format::salt_v2::{SaltV2CodecError, unpack_b3, unpack_d2, unpack_s34};
use tritium_format::trit_model::{
    additive_semantic_digest, encode_model_manifest, read_trit_model,
};
use tritium_format::{
    AdditivePayload, AdditiveTensor, AdditiveTensorError, TritBlob, TritBlobKind,
    write_trit_package,
};
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, BlobId, BlobReference, Level, LevelFit, ManifestAsset,
    ModelManifest, PlaneAllocation, PlaneCodec, PlaneCounts, PlaneRelation, ScaleAnchor, SchemaId,
    TensorMeta, TensorStorage, Transport,
};

fn tensor(layout: AdditiveLayout, planes: &PlaneCounts) -> AdditiveTensor {
    let (trits, scales) = match planes {
        PlaneCounts::Uniform(count) => {
            let units = match layout.law.anchor {
                ScaleAnchor::Group => {
                    layout.rows as usize
                        * (layout.cols as usize).div_ceil(usize::from(layout.group))
                }
                ScaleAnchor::Row | ScaleAnchor::Channel => layout.rows as usize,
                ScaleAnchor::Tensor => 1,
                _ => unreachable!(),
            };
            (
                layout.rows as usize * layout.cols as usize * usize::from(*count),
                units
                    * if layout.law.relation == PlaneRelation::Free {
                        usize::from(*count)
                    } else {
                        1
                    },
            )
        }
        PlaneCounts::PerTile(counts) => {
            let plan = PerTilePlan::new(layout, counts).unwrap();
            (plan.trit_count(), plan.scale_count())
        }
        _ => unreachable!(),
    };
    let trits = (0..trits)
        .map(|i| [Trit::NEG, Trit::ZERO, Trit::POS, Trit::NEG][i % 4])
        .collect();
    let scales = vec![0.5; scales];
    match planes {
        PlaneCounts::Uniform(count) => AdditiveTensor::new(layout, *count, trits, scales).unwrap(),
        PlaneCounts::PerTile(counts) => {
            AdditiveTensor::new_per_tile(layout, counts.clone(), trits, scales).unwrap()
        }
        _ => unreachable!(),
    }
}

fn reference(bytes: &[u8]) -> BlobReference {
    BlobReference {
        id: BlobId::from_bytes(*blake3::hash(bytes).as_bytes()),
        shard: None,
    }
}

fn model_digest(owned: &AdditiveTensor, planes: PlaneCounts, payload: &[u8]) {
    // Previous owned digest workflow as a test-only independent metadata oracle.
    let layout = owned.layout();
    let semantic = tritium_schema::TensorSemanticMeta::Additive {
        rows: layout.rows,
        cols: layout.cols,
        tile: layout.tile,
        group: layout.group,
        planes: planes.clone(),
        law: layout.law,
        basis: layout.basis,
    };
    let mut value = Value::serialized(&semantic).unwrap();
    fn canonical(value: &mut Value) {
        match value {
            Value::Array(values) => values.iter_mut().for_each(canonical),
            Value::Map(entries) => {
                for (key, value) in entries.iter_mut() {
                    canonical(key);
                    canonical(value);
                }
                entries.sort_by_cached_key(|(key, _)| CanonicalValue::from(key.clone()));
            }
            _ => {}
        }
    }
    canonical(&mut value);
    let mut encoded = Vec::new();
    ciborium::ser::into_writer(&value, &mut encoded).unwrap();
    let mut reference_hash = blake3::Hasher::new_derive_key("tritium.trit.additive.v1");
    reference_hash.update(&(encoded.len() as u64).to_le_bytes());
    reference_hash.update(&encoded);
    reference_hash.update(&(owned.trits().len() as u64).to_le_bytes());
    reference_hash.update(
        &owned
            .trits()
            .iter()
            .map(|trit| (trit.get() + 1) as u8)
            .collect::<Vec<_>>(),
    );
    let scale_bits: Vec<u8> = owned
        .scales()
        .iter()
        .flat_map(|scale| match layout.law.precision {
            tritium_schema::ScalePrecision::F16 => {
                half::f16::from_f32(*scale).to_bits().to_le_bytes().to_vec()
            }
            tritium_schema::ScalePrecision::Bf16 => half::bf16::from_f32(*scale)
                .to_bits()
                .to_le_bytes()
                .to_vec(),
            tritium_schema::ScalePrecision::F32 => scale.to_bits().to_le_bytes().to_vec(),
            _ => unreachable!(),
        })
        .collect();
    reference_hash.update(&(scale_bits.len() as u64).to_le_bytes());
    reference_hash.update(&scale_bits);
    assert_eq!(
        additive_semantic_digest(owned).unwrap().as_bytes(),
        reference_hash.finalize().as_bytes()
    );
    let manifest = ModelManifest {
        schema: SchemaId { major: 1, minor: 0 },
        architecture: "fixture-only".into(),
        assets: ["config.json", "tokenizer.json"]
            .map(|name| ManifestAsset {
                name: name.into(),
                blob: reference(b"{}"),
            })
            .to_vec(),
        levels: vec![Level {
            name: "imported".into(),
            fit: LevelFit::Imported,
            tensors: vec![TensorMeta {
                name: "weight".into(),
                blob: reference(payload),
                storage: TensorStorage::Additive {
                    layout: owned.layout(),
                    planes,
                },
            }],
        }],
        shards: vec![],
        legacy_ids: vec![],
    };
    let blobs = [
        TritBlob {
            kind: TritBlobKind::AdditiveTensor,
            bytes: payload,
        },
        TritBlob {
            kind: TritBlobKind::Asset("config.json".into()),
            bytes: b"{}",
        },
        TritBlob {
            kind: TritBlobKind::Asset("tokenizer.json".into()),
            bytes: b"{}",
        },
    ];
    let package = write_trit_package(&encode_model_manifest(&manifest).unwrap(), &blobs).unwrap();
    assert_eq!(
        read_trit_model(&package)
            .unwrap()
            .tensor_digest("imported", "weight")
            .unwrap(),
        additive_semantic_digest(owned).unwrap()
    );
}

#[test]
fn borrowed_view_and_actual_model_hash_match_owned_all_profiles() {
    let mut cases = 0;
    for law in ADMITTED_LAWS {
        for group in [32, 64, 128, 256] {
            for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
                for cols in [256u64, 1280, 1536, 257] {
                    if codec == PlaneCodec::S34 && !cols.is_multiple_of(4) {
                        continue;
                    }
                    for basis in [
                        Basis::Identity,
                        Basis::Hadamard { block: 32 },
                        Basis::SignedRht {
                            block: 32,
                            seed: 123,
                            domain: 7,
                        },
                    ] {
                        if basis != Basis::Identity && !cols.is_multiple_of(32) {
                            continue;
                        }
                        for allocation in [PlaneAllocation::Uniform, PlaneAllocation::PerTile] {
                            let layout = AdditiveLayout {
                                rows: 2,
                                cols,
                                tile: 256,
                                group,
                                max_planes: law.max_planes,
                                allocation,
                                codec,
                                law: law.law,
                                basis,
                                transport: Transport::Raw,
                            };
                            let planes = if allocation == PlaneAllocation::Uniform {
                                PlaneCounts::Uniform(law.max_planes)
                            } else {
                                let n = 2 * (cols as usize).div_ceil(256);
                                PlaneCounts::PerTile(
                                    (0..n)
                                        .map(|i| (i % (usize::from(law.max_planes) + 1)) as u8)
                                        .collect(),
                                )
                            };
                            let owned = tensor(layout, &planes);
                            let bytes = owned.encode_payload().unwrap();
                            let view = AdditivePayload::new(layout, &planes, &bytes).unwrap();
                            assert_eq!(view.to_owned().unwrap(), owned);
                            assert_eq!(view.trit_count(), owned.trits().len());
                            let mut visited = Vec::new();
                            view.visit_trits(|chunk| {
                                assert!(chunk.len() <= 640);
                                visited.extend_from_slice(chunk);
                            })
                            .unwrap();
                            assert_eq!(visited, owned.trits());
                            model_digest(&owned, planes, &bytes);
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    println!("{cases} borrowed/owned/complete-model compatibility cases");
}

#[test]
fn codec_errors_after_first_chunk_keep_full_plane_positions() {
    for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
        let cols = if codec == PlaneCodec::S34 { 1284 } else { 1281 };
        let layout = AdditiveLayout {
            rows: 1,
            cols,
            tile: 256,
            group: 128,
            max_planes: 3,
            allocation: PlaneAllocation::Uniform,
            codec,
            law: ADMITTED_LAWS[0].law,
            basis: Basis::Identity,
            transport: Transport::Raw,
        };
        let planes = PlaneCounts::Uniform(1);
        let owned = tensor(layout, &planes);
        let mut bytes = owned.encode_payload().unwrap();
        let packed_len = match codec {
            PlaneCodec::D2 => (cols as usize).div_ceil(4),
            PlaneCodec::B3 => (cols as usize).div_ceil(5),
            PlaneCodec::S34 => (cols as usize / 4 * 5).div_ceil(8),
            _ => unreachable!(),
        };
        let offset = match codec {
            PlaneCodec::D2 => 200,
            PlaneCodec::B3 => 200,
            PlaneCodec::S34 => packed_len - 1,
            _ => unreachable!(),
        };
        bytes[offset] = 255;
        let reference = match codec {
            PlaneCodec::D2 => unpack_d2(&bytes[..packed_len], cols as usize),
            PlaneCodec::B3 => unpack_b3(&bytes[..packed_len], cols as usize),
            PlaneCodec::S34 => unpack_s34(&bytes[..packed_len], cols as usize),
            _ => unreachable!(),
        }
        .unwrap_err();
        assert_eq!(
            AdditivePayload::new(layout, &planes, &bytes).unwrap_err(),
            AdditiveTensorError::Codec(reference)
        );
        if codec == PlaneCodec::D2 {
            assert!(matches!(
                AdditivePayload::new(layout, &planes, &bytes),
                Err(AdditiveTensorError::Codec(
                    SaltV2CodecError::InvalidD2Code {
                        trit_index: 800,
                        ..
                    }
                ))
            ));
        }
    }
}

#[test]
fn empty_compact_scales_and_wrong_lengths_are_strict() {
    for law in ADMITTED_LAWS {
        let layout = AdditiveLayout {
            rows: 2,
            cols: 257,
            tile: 256,
            group: 32,
            max_planes: law.max_planes,
            allocation: PlaneAllocation::PerTile,
            codec: PlaneCodec::B3,
            law: law.law,
            basis: Basis::Identity,
            transport: Transport::Raw,
        };
        let planes = PlaneCounts::PerTile(vec![0; 4]);
        let view = AdditivePayload::new(layout, &planes, &[]).unwrap();
        assert_eq!(view.trit_count(), 0);
        assert!(view.scale_bytes().is_empty());
        model_digest(&view.to_owned().unwrap(), planes.clone(), &[]);
        assert!(AdditivePayload::new(layout, &planes, &[0]).is_err());
        let planes = PlaneCounts::PerTile(vec![1, 0, 0, 0]);
        let owned = tensor(layout, &planes);
        let bytes = owned.encode_payload().unwrap();
        assert!(AdditivePayload::new(layout, &planes, &bytes[..bytes.len() - 1]).is_err());
        let scale_start = bytes.len()
            - owned.scales().len()
                * if law.law.precision == tritium_schema::ScalePrecision::F32 {
                    4
                } else {
                    2
                };
        for bad in [f32::NAN, f32::INFINITY, -0.0, -1.0] {
            let mut corrupt = bytes.clone();
            let bits = if law.law.precision == tritium_schema::ScalePrecision::F32 {
                bad.to_le_bytes().to_vec()
            } else {
                half::f16::from_f32(bad).to_bits().to_le_bytes().to_vec()
            };
            corrupt[scale_start..scale_start + bits.len()].copy_from_slice(&bits);
            assert!(matches!(
                AdditivePayload::new(layout, &planes, &corrupt),
                Err(AdditiveTensorError::InvalidView(
                    tritium_core::AdditiveError::InvalidScale
                ))
            ));
        }
    }
}
