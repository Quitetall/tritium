//! Complete-package tests of the proposed compact model profile (ADR 0066).
//! The tiny assets satisfy container structure; they are not runnable models.

use tritium_core::Trit;
use tritium_format::trit_model::{
    TritModelError, additive_semantic_digest, decode_additive_payload, encode_model_manifest,
    read_trit_model,
};
use tritium_format::{AdditiveTensor, TritBlob, TritBlobKind, write_trit_package};
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, BlobId, BlobReference, Level, LevelFit, ManifestAsset,
    ModelManifest, PlaneAllocation, PlaneCodec, PlaneCounts, PlaneRelation, ScaleAnchor, SchemaId,
    TensorMeta, TensorStorage, Transport,
};

const NAME: &str = "model.embed_tokens.weight";
const ASSET: &[u8] = b"{}";

fn layout(law: tritium_schema::AdmittedLaw, cols: u64, group: u16) -> AdditiveLayout {
    AdditiveLayout {
        rows: 2,
        cols,
        tile: 256,
        group,
        max_planes: law.max_planes,
        allocation: PlaneAllocation::PerTile,
        codec: PlaneCodec::D2,
        law: law.law,
        basis: Basis::Identity,
        transport: Transport::Raw,
    }
}

// Select fixture coefficients/scales by logical coordinates. This is not an
// offset/codec/quantization oracle: the actual package reader validates storage.
fn tensor(mut layout: AdditiveLayout, counts: Option<&[u8]>, planes: u8) -> AdditiveTensor {
    let rows = layout.rows as usize;
    let cols = layout.cols as usize;
    let tiles = cols.div_ceil(256);
    let group = usize::from(layout.group);
    let present = counts.map_or(planes, |c| c.iter().copied().max().unwrap());
    let count = |row: usize, col: usize| counts.map_or(planes, |c| c[row * tiles + col / 256]);
    let mut trits = Vec::new();
    let mut scales = Vec::new();
    for plane in 0..present {
        for row in 0..rows {
            for col in 0..cols {
                if count(row, col) > plane {
                    trits.push(
                        Trit::from_i8(((usize::from(plane) * 13 + row * 7 + col) % 3) as i8 - 1)
                            .unwrap(),
                    );
                }
            }
            if layout.law.anchor == ScaleAnchor::Group
                && (layout.law.relation == PlaneRelation::Free || plane == 0)
            {
                for unit in 0..cols.div_ceil(group) {
                    if count(row, unit * group) > plane {
                        scales.push(
                            (1 + (usize::from(plane) * 7 + row * 3 + unit) % 31) as f32 * 0.125,
                        );
                    }
                }
            }
        }
        if layout.law.anchor == ScaleAnchor::Tensor {
            scales.push(f32::from(plane + 1) * 0.125);
        }
    }
    if let Some(counts) = counts {
        AdditiveTensor::new_per_tile(layout, counts.to_vec(), trits, scales).unwrap()
    } else {
        layout.allocation = PlaneAllocation::Uniform;
        layout.max_planes = planes;
        AdditiveTensor::new(layout, planes, trits, scales).unwrap()
    }
}

fn reference(bytes: &[u8]) -> BlobReference {
    BlobReference {
        id: BlobId::from_bytes(*blake3::hash(bytes).as_bytes()),
        shard: None,
    }
}

fn metadata(tensor: &AdditiveTensor, bytes: &[u8]) -> TensorMeta {
    TensorMeta {
        name: NAME.into(),
        blob: reference(bytes),
        storage: TensorStorage::Additive {
            layout: tensor.layout(),
            planes: tensor.tile_counts().map_or_else(
                || PlaneCounts::Uniform(tensor.plane_count()),
                |counts| PlaneCounts::PerTile(counts.to_vec()),
            ),
        },
    }
}

fn manifest(levels: Vec<Level>) -> ModelManifest {
    ModelManifest {
        schema: SchemaId { major: 1, minor: 0 },
        architecture: "llama".into(),
        assets: ["config.json", "tokenizer.json"]
            .map(|name| ManifestAsset {
                name: name.into(),
                blob: reference(ASSET),
            })
            .to_vec(),
        levels,
        shards: vec![],
        legacy_ids: vec![],
    }
}

fn package(manifest: &ModelManifest, payloads: &[Vec<u8>]) -> Vec<u8> {
    let mut blobs = ["config.json", "tokenizer.json"]
        .map(|name| TritBlob {
            kind: TritBlobKind::Asset(name.into()),
            bytes: ASSET,
        })
        .to_vec();
    blobs.extend(payloads.iter().map(|bytes| TritBlob {
        kind: TritBlobKind::AdditiveTensor,
        bytes,
    }));
    write_trit_package(&encode_model_manifest(manifest).unwrap(), &blobs).unwrap()
}

fn one(tensor: &AdditiveTensor) -> Vec<u8> {
    let payload = tensor.encode_payload().unwrap();
    package(
        &manifest(vec![Level {
            name: "imported".into(),
            fit: LevelFit::Imported,
            tensors: vec![metadata(tensor, &payload)],
        }]),
        &[payload],
    )
}

fn prefix(child: &AdditiveTensor, parent: &AdditiveTensor) -> Vec<u8> {
    let a = child.encode_payload().unwrap();
    let b = parent.encode_payload().unwrap();
    package(
        &manifest(vec![
            Level {
                name: "child".into(),
                fit: LevelFit::ImportedPrefixOf {
                    level: "parent".into(),
                },
                tensors: vec![metadata(child, &a)],
            },
            Level {
                name: "parent".into(),
                fit: LevelFit::Imported,
                tensors: vec![metadata(parent, &b)],
            },
        ]),
        &[a, b],
    )
}

#[test]
fn complete_compact_packages_bind_exact_counts_and_reuse_real_payloads() {
    for &law in ADMITTED_LAWS {
        for group in [32, 128] {
            for cols in [1, 255, 256, 257, 512] {
                let l = layout(law, cols, group);
                let counts: Vec<_> = (0..2 * (cols as usize).div_ceil(256))
                    .map(|i| (i % (usize::from(law.max_planes) + 1)) as u8)
                    .collect();
                for counts in [counts, vec![0; 2 * (cols as usize).div_ceil(256)]] {
                    let original = tensor(l, Some(&counts), 0);
                    let source = one(&original);
                    let model = read_trit_model(&source).unwrap();
                    let (meta, bytes) = model.tensor("imported", NAME).unwrap();
                    assert_eq!(meta.storage, metadata(&original, bytes).storage);
                    let decoded =
                        AdditiveTensor::decode_per_tile_payload(l, &counts, bytes).unwrap();
                    assert_eq!(decoded, original);
                    assert_eq!(
                        model.tensor_digest("imported", NAME).unwrap(),
                        additive_semantic_digest(&original).unwrap()
                    );
                    assert_eq!(model.manifest().levels[0].fit, LevelFit::Imported);
                    let again = one(&decoded);
                    assert_eq!(again, source);
                }
            }
        }
    }
}

#[test]
fn compact_identity_is_codec_independent_but_binds_the_count_partition() {
    let l = layout(ADMITTED_LAWS[0], 512, 128);
    let a = tensor(l, Some(&[1, 3, 2, 0]), 0);
    let b = AdditiveTensor::new_per_tile(
        AdditiveLayout {
            codec: PlaneCodec::B3,
            ..l
        },
        a.tile_counts().unwrap().to_vec(),
        a.trits().to_vec(),
        a.scales().to_vec(),
    )
    .unwrap();
    let source_a = one(&a);
    let source_b = one(&b);
    let model_a = read_trit_model(&source_a).unwrap();
    let model_b = read_trit_model(&source_b).unwrap();
    assert_eq!(
        model_a.tensor_digest("imported", NAME).unwrap(),
        model_b.tensor_digest("imported", NAME).unwrap()
    );
    assert_eq!(
        model_a.model_id("imported").unwrap(),
        model_b.model_id("imported").unwrap()
    );
    assert_ne!(source_a, source_b);
    // Same zero coefficients and scale bytes, different logical tile ownership.
    let a = AdditiveTensor::new_per_tile(l, vec![1, 0, 1, 0], vec![Trit::ZERO; 512], vec![0.5; 4])
        .unwrap();
    let b =
        AdditiveTensor::new_per_tile(l, vec![0, 1, 0, 1], a.trits().to_vec(), a.scales().to_vec())
            .unwrap();
    assert_ne!(
        additive_semantic_digest(&a).unwrap(),
        additive_semantic_digest(&b).unwrap()
    );
    let source_a = one(&a);
    let source_b = one(&b);
    assert_ne!(
        read_trit_model(&source_a)
            .unwrap()
            .model_id("imported")
            .unwrap(),
        read_trit_model(&source_b)
            .unwrap()
            .model_id("imported")
            .unwrap()
    );
    let equal = tensor(l, Some(&[3; 4]), 0);
    let uniform = tensor(l, None, 3);
    assert_eq!(
        equal.encode_payload().unwrap(),
        uniform.encode_payload().unwrap()
    );
    assert_ne!(
        additive_semantic_digest(&equal).unwrap(),
        additive_semantic_digest(&uniform).unwrap()
    );
}

#[test]
fn imported_prefix_checks_logical_coordinates_across_both_allocations() {
    for &law in ADMITTED_LAWS {
        for group in [32, 128] {
            let l = layout(law, 257, group);
            let n = law.max_planes;
            for (child_counts, parent_counts) in [
                (None, None),
                (Some([0, 1, 1, 0]), None),
                (None, Some([1, n, n, 1])),
                (Some([0, 1, 1, 0]), Some([1, n, n, 1])),
                (Some([0, n, 1, 0]), Some([1, n, n, 1])),
                (Some([1, 0, n, 0]), Some([1, n, n, 1])),
                (Some([0; 4]), Some([0; 4])),
            ] {
                let child = tensor(
                    AdditiveLayout {
                        codec: PlaneCodec::B3,
                        ..l
                    },
                    child_counts.as_ref().map(|c| c.as_slice()),
                    1,
                );
                let parent = tensor(l, parent_counts.as_ref().map(|c| c.as_slice()), n);
                assert!(child.view().is_logical_prefix_of(parent.view()));
                let source = prefix(&child, &parent);
                let model = read_trit_model(&source).unwrap();
                assert_eq!(
                    model.manifest().levels[0].fit,
                    LevelFit::ImportedPrefixOf {
                        level: "parent".into()
                    }
                );
                if !child.trits().is_empty() {
                    for corrupt_scale in [false, true] {
                        let mut trits = child.trits().to_vec();
                        let mut scales = child.scales().to_vec();
                        if corrupt_scale {
                            scales[0] *= 2.0;
                        } else {
                            trits[0] = if trits[0] == Trit::POS {
                                Trit::NEG
                            } else {
                                Trit::POS
                            };
                        }
                        let corrupt = if let Some(counts) = child.tile_counts() {
                            AdditiveTensor::new_per_tile(
                                child.layout(),
                                counts.to_vec(),
                                trits,
                                scales,
                            )
                            .unwrap()
                        } else {
                            AdditiveTensor::new(child.layout(), child.plane_count(), trits, scales)
                                .unwrap()
                        };
                        assert!(!corrupt.view().is_logical_prefix_of(parent.view()));
                        assert!(matches!(
                            read_trit_model(&prefix(&corrupt, &parent)),
                            Err(TritModelError::InvalidPrefix(_))
                        ));
                    }
                }
            }
        }
    }
}

#[test]
fn compact_candidate_vectors_match_separate_python_preimages() {
    for (law, cols, counts, trits, scales, expected) in [
        (
            ADMITTED_LAWS[0],
            512,
            [1, 0, 1, 0],
            512,
            4,
            "470b29d42f3c920c4607e6e6f52d6d56d0e9cbe9faba52a682f91faa604dad38",
        ),
        (
            ADMITTED_LAWS[2],
            257,
            [1, 3, 2, 0],
            771,
            5,
            "033c3c21362004d35f3e8bef90197b1080be442e8f596e899729351e69fbf6fa",
        ),
        (
            ADMITTED_LAWS[0],
            257,
            [0; 4],
            0,
            0,
            "613f6359e9c50727588ff82539f2edbb1e34d0748583dbe9dd709204573c51f1",
        ),
    ] {
        let tensor = AdditiveTensor::new_per_tile(
            layout(law, cols, 128),
            counts.to_vec(),
            vec![Trit::ZERO; trits],
            vec![0.5; scales],
        )
        .unwrap();
        let source = one(&tensor);
        let model = read_trit_model(&source).unwrap();
        assert_eq!(
            blake3::Hash::from_bytes(*model.tensor_digest("imported", NAME).unwrap().as_bytes())
                .to_hex()
                .as_str(),
            expected
        );
    }
}

#[test]
fn prefixes_reject_more_planes_and_incompatible_semantics_or_capacity() {
    let l = layout(ADMITTED_LAWS[0], 512, 128);
    let child = tensor(l, Some(&[0, 2, 1, 0]), 0);
    let parent = tensor(l, Some(&[3, 1, 1, 0]), 0);
    assert!(!child.view().is_logical_prefix_of(parent.view()));
    assert!(matches!(
        read_trit_model(&prefix(&child, &parent)),
        Err(TritModelError::InvalidPrefix(_))
    ));
    let child = tensor(l, Some(&[0, 1, 1, 0]), 0);
    // The shared logical comparator excludes unused capacity; the model profile
    // retains its prior metadata/capacity constraint, even for compact children.
    let parent = tensor(l, None, 1);
    assert!(child.view().is_logical_prefix_of(parent.view()));
    assert!(matches!(
        read_trit_model(&prefix(&child, &parent)),
        Err(TritModelError::InvalidPrefix(_))
    ));
    let parent = tensor(l, None, 3);
    for changed in [
        AdditiveLayout { group: 32, ..l },
        AdditiveLayout {
            basis: Basis::Hadamard { block: 32 },
            ..l
        },
        layout(ADMITTED_LAWS[1], 512, 128),
    ] {
        let other = tensor(changed, Some(&[0, 1, 1, 0]), 0);
        assert!(!other.view().is_logical_prefix_of(parent.view()));
        assert!(matches!(
            read_trit_model(&prefix(&other, &parent)),
            Err(TritModelError::InvalidPrefix(_))
        ));
    }
}

#[test]
fn compact_packages_reject_wrong_payload_counts_and_rehashed_invalid_bytes() {
    let l = layout(ADMITTED_LAWS[0], 257, 128);
    let original = tensor(l, Some(&[1, 3, 2, 0]), 0);
    let payload = original.encode_payload().unwrap();
    let mut m = manifest(vec![Level {
        name: "imported".into(),
        fit: LevelFit::Imported,
        tensors: vec![metadata(&original, &payload)],
    }]);
    let TensorStorage::Additive { planes, .. } = &mut m.levels[0].tensors[0].storage else {
        unreachable!()
    };
    *planes = PlaneCounts::PerTile(vec![0; 4]);
    assert!(matches!(
        read_trit_model(&package(&m, std::slice::from_ref(&payload))),
        Err(TritModelError::Additive(_))
    ));
    for change in [0, 1, 2] {
        let mut bad = payload.clone();
        match change {
            0 => {
                bad[0] |= 0b11;
            }
            1 => {
                bad.pop();
            }
            _ => {
                let n = bad.len();
                bad[n - 2..].copy_from_slice(&0x8000u16.to_le_bytes());
            }
        }
        m.levels[0].tensors[0] = metadata(&original, &bad);
        assert!(matches!(
            read_trit_model(&package(&m, &[bad])),
            Err(TritModelError::Additive(_))
        ));
    }
    for counts in [vec![1, 3, 2], vec![1, 4, 2, 0]] {
        let TensorStorage::Additive { planes, .. } = &mut m.levels[0].tensors[0].storage else {
            unreachable!()
        };
        *planes = PlaneCounts::PerTile(counts);
        assert!(encode_model_manifest(&m).is_err());
    }
}

#[test]
fn shared_payload_decoder_owns_both_allocations_but_does_not_authenticate_blob_ids() {
    for counts in [None, Some([1, 3, 2, 0])] {
        let original = tensor(
            layout(ADMITTED_LAWS[0], 257, 128),
            counts.as_ref().map(|c| c.as_slice()),
            3,
        );
        let payload = original.encode_payload().unwrap();
        let mut meta = metadata(&original, &payload);
        // Raw decode is deliberately not container authentication. The complete
        // reader still checks physical identity before any model upload.
        meta.blob.id = BlobId::from_bytes([0; 32]);
        let decoded = decode_additive_payload(&meta, &payload).unwrap();
        drop(payload);
        assert_eq!(decoded, original);
    }
}

#[test]
fn shared_payload_decoder_preserves_typed_errors_before_model_admission() {
    let original = tensor(layout(ADMITTED_LAWS[0], 257, 128), Some(&[1, 3, 2, 0]), 0);
    let payload = original.encode_payload().unwrap();
    let original_meta = metadata(&original, &payload);
    for counts in [vec![1, 3, 2], vec![1, 4, 2, 0], vec![0; 4]] {
        let mut meta = original_meta.clone();
        let TensorStorage::Additive { planes, .. } = &mut meta.storage else {
            unreachable!()
        };
        *planes = PlaneCounts::PerTile(counts);
        assert!(matches!(
            decode_additive_payload(&meta, &payload),
            Err(TritModelError::Additive(_))
        ));
    }
    let mut meta = original_meta.clone();
    meta.storage = TensorStorage::Dense {
        shape: vec![2, 257],
        precision: tritium_schema::ScalePrecision::F32,
    };
    assert!(matches!(
        decode_additive_payload(&meta, &payload),
        Err(TritModelError::UnsupportedFeature("not additive"))
    ));
    let mut meta = original_meta;
    let TensorStorage::Additive { layout, .. } = &mut meta.storage else {
        unreachable!()
    };
    layout.transport = Transport::JointHuffman { block: 256 };
    assert!(matches!(
        decode_additive_payload(&meta, &payload),
        Err(TritModelError::UnsupportedFeature("entropy model payloads"))
    ));
}
