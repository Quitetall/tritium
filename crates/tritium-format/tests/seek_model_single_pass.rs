//! Synthetic public-reader read budget, not whole-model latency qualification.
use std::cell::Cell;
use std::io::{self, Cursor, Read, Seek, SeekFrom};
use std::rc::Rc;
use tritium_core::{Trit, UniformPlan};
use tritium_format::trit_model::{TritModelReader, encode_model_manifest, read_trit_model};
use tritium_format::{
    AdditiveTensor, TritBlob, TritBlobKind, TritPackageReadLimits, read_trit_package,
    write_trit_package,
};
use tritium_schema::*;

fn layout() -> AdditiveLayout {
    AdditiveLayout {
        rows: 1,
        cols: 327681,
        tile: 256,
        group: 128,
        max_planes: 3,
        allocation: PlaneAllocation::Uniform,
        codec: PlaneCodec::B3,
        law: ScaleLaw {
            anchor: ScaleAnchor::Group,
            relation: PlaneRelation::Free,
            precision: ScalePrecision::F16,
        },
        basis: Basis::Identity,
        transport: Transport::Raw,
    }
}

fn payload(layout: AdditiveLayout) -> Vec<u8> {
    let plan = UniformPlan::new(layout, 3).unwrap();
    AdditiveTensor::new(
        layout,
        3,
        (0..plan.trit_count())
            .map(|i| Trit::from_i8((i % 3) as i8 - 1).unwrap())
            .collect(),
        vec![0.5; plan.scale_count()],
    )
    .unwrap()
    .encode_payload()
    .unwrap()
}

fn package(layout: AdditiveLayout, payload: &[u8]) -> Vec<u8> {
    let reference = |bytes: &[u8]| BlobReference {
        id: BlobId::from_bytes(*blake3::hash(bytes).as_bytes()),
        shard: None,
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
        levels: vec![Level {
            name: "imported".into(),
            fit: LevelFit::Imported,
            tensors: vec![TensorMeta {
                name: "weight".into(),
                blob: reference(payload),
                storage: TensorStorage::Additive {
                    layout,
                    planes: PlaneCounts::Uniform(3),
                },
            }],
        }],
        shards: vec![],
        legacy_ids: vec![],
    };
    write_trit_package(
        &encode_model_manifest(&manifest).unwrap(),
        &[
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
                bytes: payload,
            },
        ],
    )
    .unwrap()
}

struct Counted<'a> {
    source: Cursor<&'a [u8]>,
    blob: std::ops::Range<u64>,
    read_bytes: Rc<Cell<u64>>,
}
impl Read for Counted<'_> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let start = self.source.position();
        let count = self.source.read(out)?;
        let overlap = (start + count as u64)
            .min(self.blob.end)
            .saturating_sub(start.max(self.blob.start));
        self.read_bytes.set(self.read_bytes.get() + overlap);
        Ok(count)
    }
}
impl Seek for Counted<'_> {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        self.source.seek(from)
    }
}

fn limits(bytes: &[u8]) -> TritPackageReadLimits {
    TritPackageReadLimits {
        max_source_bytes: bytes.len() as u64,
        max_manifest_bytes: bytes.len(),
        max_directory_bytes: bytes.len(),
    }
}

#[test]
fn typed_additive_admission_reuses_one_validated_semantic_payload_pass() {
    let payload = payload(layout());
    let bytes = package(layout(), &payload);
    let raw = read_trit_package(&bytes).unwrap();
    let blob = raw
        .blobs()
        .iter()
        .find(|blob| blob.kind == TritBlobKind::AdditiveTensor)
        .unwrap();
    assert!(blob.length > 2 * 65536);
    let read_bytes = Rc::new(Cell::new(0));
    let source = Counted {
        source: Cursor::new(bytes.as_slice()),
        blob: blob.offset..blob.offset + blob.length,
        read_bytes: read_bytes.clone(),
    };
    let reader = TritModelReader::open(source, limits(&bytes)).unwrap();
    let borrowed = read_trit_model(&bytes).unwrap();
    assert_eq!(reader.package_id(), borrowed.package_id());
    assert_eq!(
        reader.model_id("imported").unwrap(),
        borrowed.model_id("imported").unwrap()
    );
    assert_eq!(
        reader.tensor_digest("imported", "weight").unwrap(),
        borrowed.tensor_digest("imported", "weight").unwrap()
    );
    // Frozen from ebd979a6 before changing the shared validator, not calculated
    // from the new reader on the assertion's expected side.
    assert_eq!(
        blake3::Hash::from_bytes(*reader.model_id("imported").unwrap().as_bytes())
            .to_hex()
            .as_str(),
        "4bda3c974e753fe2075fa06313e87cb666d6cb3ce42ca53318fb7de520dfcb10"
    );
    assert_eq!(
        blake3::Hash::from_bytes(
            *reader
                .tensor_digest("imported", "weight")
                .unwrap()
                .as_bytes()
        )
        .to_hex()
        .as_str(),
        "aec5cc80c67ce77c69cd02ab54ab035a22765a2e6bd9379376e703b51f07ea0a"
    );
    // Two initial full-source identity scans, one blob-integrity scan and the
    // final full-source identity scan are intentionally retained. Typed codec,
    // scale and semantic validation should add only one further payload pass.
    assert_eq!(read_bytes.get(), 5 * blob.length);
}

#[test]
fn shared_semantic_pass_refuses_noncanonical_trits_padding_and_every_scale_window() {
    for cols in [327681, 2000001] {
        let layout = AdditiveLayout { cols, ..layout() };
        let original = payload(layout);
        let plan = UniformPlan::new(layout, 3).unwrap();
        assert_eq!(plan.scale_count() * 2 > 65536, cols == 2000001);
        let scale_start = original.len() - plan.scale_count() * 2;
        for fault in [
            "codec",
            "padding",
            "nan",
            "infinity",
            "negative-zero",
            "late-negative",
        ] {
            let mut changed = original.clone();
            match fault {
                "codec" => changed[0] = 255,
                "padding" => changed[(cols as usize).div_ceil(5) - 1] += 3,
                "nan" => {
                    changed[scale_start..scale_start + 2].copy_from_slice(&0x7e00u16.to_le_bytes())
                }
                "infinity" => {
                    changed[scale_start..scale_start + 2].copy_from_slice(&0x7c00u16.to_le_bytes())
                }
                "negative-zero" => {
                    changed[scale_start..scale_start + 2].copy_from_slice(&0x8000u16.to_le_bytes())
                }
                "late-negative" => {
                    let start = changed.len() - 2;
                    changed[start..].copy_from_slice(&0xb800u16.to_le_bytes());
                }
                _ => unreachable!(),
            }
            // Rebuild stored blob/package hashes so this reaches typed payload
            // validation rather than merely tripping package-integrity checks.
            let bytes = package(layout, &changed);
            let borrowed = read_trit_model(&bytes).unwrap_err();
            let retained = TritModelReader::open(Cursor::new(&bytes), limits(&bytes)).unwrap_err();
            assert_eq!(borrowed.to_string(), retained.to_string(), "{cols}/{fault}");
            // Original ebd979a6 refusal domains/locations, not just agreement
            // between two adapters that now share the modified implementation.
            let expected = match fault {
                "codec" => {
                    "trit model: Additive(Codec(InvalidB3Code { byte_index: 0, code: 255 }))".into()
                }
                "padding" => format!(
                    "trit model: Additive(Codec(NonCanonicalTritPadding {{ codec: B3, trit_index: {cols}, code: 2 }}))"
                ),
                _ => "trit model: Additive(InvalidView(InvalidScale))".into(),
            };
            assert_eq!(borrowed.to_string(), expected, "{cols}/{fault}");
        }
    }
}
