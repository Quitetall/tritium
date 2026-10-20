//! Synthetic complete typed inventory, not pretrained or runnable model evidence.
use tritium_core::{DType, PerTilePlan, Trit, UniformPlan};
use tritium_format::trit_model::encode_model_manifest;
use tritium_format::{AdditiveTensor, TritBlob, TritBlobKind, write_trit_package};
use tritium_schema::*;

fn reference(bytes: &[u8]) -> BlobReference {
    BlobReference {
        id: BlobId::from_bytes(*blake3::hash(bytes).as_bytes()),
        shard: None,
    }
}
pub(super) fn fixture(cols: usize, dense_bytes: usize) -> Vec<u8> {
    let mut layout = AdditiveLayout {
        rows: 2,
        cols: cols as u64,
        tile: 256,
        group: 128,
        max_planes: 2,
        allocation: PlaneAllocation::Uniform,
        codec: PlaneCodec::B3,
        law: ScaleLaw {
            anchor: ScaleAnchor::Group,
            relation: PlaneRelation::Free,
            precision: ScalePrecision::F32,
        },
        basis: Basis::Identity,
        transport: Transport::Raw,
    };
    let value = |plane: usize, row: usize, col: usize| {
        Trit::from_i8(((plane * 11 + row * 7 + col) % 3) as i8 - 1).unwrap()
    };
    let mut parent_trits = Vec::new();
    for plane in 0..2 {
        for row in 0..2 {
            for col in 0..cols {
                parent_trits.push(value(plane, row, col));
            }
        }
    }
    let plan = UniformPlan::new(layout, 2).unwrap();
    let parent =
        AdditiveTensor::new(layout, 2, parent_trits, vec![0.5; plan.scale_count()]).unwrap();
    layout.allocation = PlaneAllocation::PerTile;
    layout.codec = PlaneCodec::D2;
    let tiles = cols.div_ceil(256);
    let counts: Vec<_> = (0..2 * tiles).map(|i| u8::from(i % 3 != 1)).collect();
    let plan = PerTilePlan::new(layout, &counts).unwrap();
    let mut child_trits = Vec::new();
    for row in 0..2 {
        for col in 0..cols {
            if counts[row * tiles + col / 256] != 0 {
                child_trits.push(value(0, row, col));
            }
        }
    }
    let child = AdditiveTensor::new_per_tile(
        layout,
        counts.clone(),
        child_trits,
        vec![0.5; plan.scale_count()],
    )
    .unwrap();
    let parent_bytes = parent.encode_payload().unwrap();
    let child_bytes = child.encode_payload().unwrap();
    assert!(dense_bytes >= 4 && dense_bytes.is_multiple_of(4));
    let mut dense = vec![0; dense_bytes];
    dense[..4].copy_from_slice(&(-0.0f32).to_le_bytes());
    let dense_meta = TensorMeta {
        name: "norm.weight".into(),
        blob: reference(&dense),
        storage: TensorStorage::Dense {
            shape: vec![(dense_bytes / 4) as u64],
            precision: ScalePrecision::F32,
        },
    };
    let meta = |tensor: &AdditiveTensor, bytes: &[u8], planes| TensorMeta {
        name: "weight".into(),
        blob: reference(bytes),
        storage: TensorStorage::Additive {
            layout: tensor.layout(),
            planes,
        },
    };
    let manifest = ModelManifest {
        schema: SchemaId { major: 1, minor: 0 },
        architecture: "llama".into(),
        assets: ["config.json", "tokenizer.json"]
            .map(|name| ManifestAsset {
                name: name.into(),
                blob: reference(b"{}"),
            })
            .to_vec(),
        levels: vec![
            Level {
                name: "child".into(),
                fit: LevelFit::ImportedPrefixOf {
                    level: "parent".into(),
                },
                tensors: vec![
                    dense_meta.clone(),
                    meta(&child, &child_bytes, PlaneCounts::PerTile(counts)),
                ],
            },
            Level {
                name: "parent".into(),
                fit: LevelFit::Imported,
                tensors: vec![
                    dense_meta,
                    meta(&parent, &parent_bytes, PlaneCounts::Uniform(2)),
                ],
            },
        ],
        shards: vec![],
        legacy_ids: vec![],
    };
    let blobs = [
        TritBlob {
            kind: TritBlobKind::Asset("config.json".into()),
            bytes: b"{}",
        },
        TritBlob {
            kind: TritBlobKind::Asset("tokenizer.json".into()),
            bytes: b"{}",
        },
        TritBlob {
            kind: TritBlobKind::AdditiveTensor,
            bytes: &parent_bytes,
        },
        TritBlob {
            kind: TritBlobKind::AdditiveTensor,
            bytes: &child_bytes,
        },
        TritBlob {
            kind: TritBlobKind::DenseTensor(DType::F32),
            bytes: &dense,
        },
    ];
    write_trit_package(&encode_model_manifest(&manifest).unwrap(), &blobs).unwrap()
}
