//! Exact prefix parity, not runtime/model quality qualification.
use tritium_core::Trit;
use tritium_format::trit_model::{encode_model_manifest, read_trit_model};
use tritium_format::{AdditivePayload, AdditiveTensor, TritBlob, TritBlobKind, write_trit_package};
#[path = "support/seek_model_parity.rs"]
mod seek_model_parity;
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, BlobId, BlobReference, Level, LevelFit, ManifestAsset,
    ModelManifest, PlaneAllocation, PlaneCodec, PlaneCounts, PlaneRelation, ScaleAnchor, SchemaId,
    TensorMeta, TensorStorage, Transport,
};

fn tensor(mut layout: AdditiveLayout, counts: &PlaneCounts) -> AdditiveTensor {
    let rows = layout.rows as usize;
    let cols = layout.cols as usize;
    let tiles = cols.div_ceil(256);
    let count = |row: usize, col: usize| match counts {
        PlaneCounts::Uniform(n) => *n,
        PlaneCounts::PerTile(counts) => counts[row * tiles + col / 256],
        _ => unreachable!(),
    };
    let present = match counts {
        PlaneCounts::Uniform(n) => *n,
        PlaneCounts::PerTile(counts) => *counts.iter().max().unwrap(),
        _ => unreachable!(),
    };
    let mut trits = Vec::new();
    let mut scales = Vec::new();
    for plane in 0..present {
        for row in 0..rows {
            for col in 0..cols {
                if count(row, col) > plane {
                    trits.push(
                        [Trit::NEG, Trit::ZERO, Trit::POS, Trit::NEG]
                            [(usize::from(plane) + col) % 4],
                    );
                }
            }
            if layout.law.anchor == ScaleAnchor::Group
                && (layout.law.relation == PlaneRelation::Free || plane == 0)
            {
                for unit in 0..cols.div_ceil(usize::from(layout.group)) {
                    if count(row, unit * usize::from(layout.group)) > plane {
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
    match counts {
        PlaneCounts::Uniform(n) => {
            layout.max_planes = *n;
            AdditiveTensor::new(layout, *n, trits, scales).unwrap()
        }
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
fn actual_model(child: &AdditiveTensor, parent: &AdditiveTensor) -> bool {
    let child_bytes = child.encode_payload().unwrap();
    let parent_bytes = parent.encode_payload().unwrap();
    let meta = |tensor: &AdditiveTensor, bytes: &[u8]| TensorMeta {
        name: "weight".into(),
        blob: reference(bytes),
        storage: TensorStorage::Additive {
            layout: tensor.layout(),
            planes: tensor
                .tile_counts()
                .map_or(PlaneCounts::Uniform(tensor.plane_count()), |counts| {
                    PlaneCounts::PerTile(counts.to_vec())
                }),
        },
    };
    let manifest = ModelManifest {
        schema: SchemaId { major: 1, minor: 0 },
        architecture: "fixture-only".into(),
        assets: ["config.json", "tokenizer.json"]
            .map(|name| ManifestAsset {
                name: name.into(),
                blob: reference(b"{}"),
            })
            .to_vec(),
        levels: vec![
            Level {
                name: "compact".into(),
                fit: LevelFit::ImportedPrefixOf {
                    level: "near".into(),
                },
                tensors: vec![meta(child, &child_bytes)],
            },
            Level {
                name: "near".into(),
                fit: LevelFit::Imported,
                tensors: vec![meta(parent, &parent_bytes)],
            },
        ],
        shards: vec![],
        legacy_ids: vec![],
    };
    let blobs = [
        TritBlob {
            kind: TritBlobKind::AdditiveTensor,
            bytes: &child_bytes,
        },
        TritBlob {
            kind: TritBlobKind::AdditiveTensor,
            bytes: &parent_bytes,
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
    let manifest = encode_model_manifest(&manifest).unwrap();
    let bytes = write_trit_package(&manifest, &blobs).unwrap();
    seek_model_parity::assert_matches(&bytes);
    read_trit_model(&bytes).is_ok()
}
fn compare(child: &AdditiveTensor, parent: &AdditiveTensor, model: bool) -> bool {
    let cp = child
        .tile_counts()
        .map_or(PlaneCounts::Uniform(child.plane_count()), |counts| {
            PlaneCounts::PerTile(counts.to_vec())
        });
    let pp = parent
        .tile_counts()
        .map_or(PlaneCounts::Uniform(parent.plane_count()), |counts| {
            PlaneCounts::PerTile(counts.to_vec())
        });
    let cb = child.encode_payload().unwrap();
    let pb = parent.encode_payload().unwrap();
    let c = AdditivePayload::new(child.layout(), &cp, &cb).unwrap();
    let p = AdditivePayload::new(parent.layout(), &pp, &pb).unwrap();
    let expected = child.view().is_logical_prefix_of(parent.view());
    assert_eq!(c.is_logical_prefix_of(&p).unwrap(), expected);
    if model {
        assert_eq!(actual_model(child, parent), expected);
    }
    expected
}

#[test]
fn bounded_prefix_matches_decoded_oracle_and_real_model_all_profiles() {
    let mut cases = 0;
    for &law in ADMITTED_LAWS {
        for group in [32, 64, 128, 256] {
            for cols in [1u64, 257, 1280, 1281] {
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
                    for child_codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
                        for parent_codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
                            if (child_codec == PlaneCodec::S34 || parent_codec == PlaneCodec::S34)
                                && !cols.is_multiple_of(4)
                            {
                                continue;
                            }
                            for ca in [PlaneAllocation::Uniform, PlaneAllocation::PerTile] {
                                for pa in [PlaneAllocation::Uniform, PlaneAllocation::PerTile] {
                                    let layout = AdditiveLayout {
                                        rows: 3,
                                        cols,
                                        tile: 256,
                                        group,
                                        max_planes: law.max_planes,
                                        allocation: pa,
                                        codec: parent_codec,
                                        law: law.law,
                                        basis,
                                        transport: Transport::Raw,
                                    };
                                    let tiles = 3 * (cols as usize).div_ceil(256);
                                    let parent_counts: Vec<_> = (0..tiles)
                                        .map(|i| {
                                            if pa == PlaneAllocation::Uniform {
                                                law.max_planes
                                            } else {
                                                ((i * 7 + 1) % (usize::from(law.max_planes) + 1))
                                                    as u8
                                            }
                                        })
                                        .collect();
                                    let pp = if pa == PlaneAllocation::Uniform {
                                        PlaneCounts::Uniform(law.max_planes)
                                    } else {
                                        PlaneCounts::PerTile(parent_counts.clone())
                                    };
                                    let cp = if ca == PlaneAllocation::Uniform {
                                        PlaneCounts::Uniform(1)
                                    } else {
                                        PlaneCounts::PerTile(
                                            parent_counts
                                                .iter()
                                                .enumerate()
                                                .map(|(i, n)| n.saturating_sub((i % 2) as u8))
                                                .collect(),
                                        )
                                    };
                                    let parent = tensor(layout, &pp);
                                    let child = tensor(
                                        AdditiveLayout {
                                            codec: child_codec,
                                            allocation: ca,
                                            ..layout
                                        },
                                        &cp,
                                    );
                                    compare(&child, &parent, true);
                                    cases += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    println!("{cases} bounded/decoded/actual imported-prefix model cases");
}

#[test]
fn changed_retained_bits_fail_but_extra_parent_content_is_not_compared() {
    let layout = AdditiveLayout {
        rows: 2,
        cols: 1281,
        tile: 256,
        group: 128,
        max_planes: 3,
        allocation: PlaneAllocation::PerTile,
        codec: PlaneCodec::B3,
        law: ADMITTED_LAWS[0].law,
        basis: Basis::Identity,
        transport: Transport::Raw,
    };
    let counts = vec![3; 12];
    let child_counts = vec![0, 0, 1, 3, 0, 1, 2, 0, 0, 1, 0, 1];
    let parent = tensor(layout, &PlaneCounts::PerTile(counts.clone()));
    let child = tensor(layout, &PlaneCounts::PerTile(child_counts));
    assert!(compare(&child, &parent, true));
    let mut trits = parent.trits().to_vec();
    trits[0] = Trit::POS; // Parent tile absent in child.
    let mut extra_scales = parent.scales().to_vec();
    extra_scales[0] += 0.125; // Unretained parent scale is also irrelevant.
    let extra = AdditiveTensor::new_per_tile(layout, counts.clone(), trits, extra_scales).unwrap();
    assert!(compare(&child, &extra, true));
    let mut trits = child.trits().to_vec();
    trits[641] = if trits[641] == Trit::POS {
        Trit::NEG
    } else {
        Trit::POS
    };
    let changed = AdditiveTensor::new_per_tile(
        layout,
        child.tile_counts().unwrap().to_vec(),
        trits,
        child.scales().to_vec(),
    )
    .unwrap();
    assert!(!compare(&changed, &parent, true));
    let mut scales = child.scales().to_vec();
    *scales.last_mut().unwrap() += 0.125;
    let changed = AdditiveTensor::new_per_tile(
        layout,
        child.tile_counts().unwrap().to_vec(),
        child.trits().to_vec(),
        scales,
    )
    .unwrap();
    assert!(!compare(&changed, &parent, true));
    let empty = tensor(layout, &PlaneCounts::PerTile(vec![0; 12]));
    assert!(compare(&empty, &parent, true));
    assert!(compare(&empty, &empty, true));
}
