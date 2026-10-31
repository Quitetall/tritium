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

#[test]
fn canonical_hash_buffer_boundaries_preserve_frozen_ids() {
    // Frozen from 4df7d4ee before changing production hashing, not computed
    // using the candidate on the expected side. Three planes give 3 * cols
    // canonical bytes: short, just below, exactly on, and above 16 KiB edges.
    for (cols, expected_model, expected_tensor) in [
        (
            1,
            "8f2d2f2821b39548eb38989b43c7d3d4e29baa40cfc11cc696faa09689cb55c0",
            "49c04aced0313f3b9ae48250911b9567e530a769472215aa21533b682dbf7b2d",
        ),
        (
            16383,
            "e08cb8faab6febd4e14fef240eb48e690bfa3d5b7b1c0d102732f7ddacf63820",
            "0c707e2fa715d2741f24a7af9f5ea30ff5e1ed5f9dc9c3c0148ebddd0686b15f",
        ),
        (
            16384,
            "67b1936f2660c999d49600850bf094b1fcfa484da0237d38bc00eaf9a02ac3c6",
            "ae7e2b8b266fb0cd8a1323064990a62142dae83de020a8d3f20d45cfe393fa4c",
        ),
        (
            16385,
            "1331bf6c518c84f153939602d6e7890b2696b314cfc2aba36b05c4761937e365",
            "1a165faf81075309c9a0827a8e97fb9f6d9ad757a3c3d156fc36cca7e04a2d21",
        ),
    ] {
        let layout = AdditiveLayout { cols, ..layout() };
        let bytes = package(layout, &payload(layout));
        let borrowed = read_trit_model(&bytes).unwrap();
        let retained = TritModelReader::open(Cursor::new(&bytes), limits(&bytes)).unwrap();
        let model = borrowed.model_id("imported").unwrap();
        let tensor = borrowed.tensor_digest("imported", "weight").unwrap();
        assert_eq!(retained.model_id("imported").unwrap(), model);
        assert_eq!(
            retained.tensor_digest("imported", "weight").unwrap(),
            tensor
        );
        assert_eq!(
            blake3::Hash::from_bytes(*model.as_bytes())
                .to_hex()
                .as_str(),
            expected_model,
            "cols={cols}"
        );
        assert_eq!(
            blake3::Hash::from_bytes(*tensor.as_bytes())
                .to_hex()
                .as_str(),
            expected_tensor,
            "cols={cols}"
        );
    }
}

#[test]
#[ignore = "explicit synthetic release-mode CPU reader profile; not a latency or release gate"]
fn profile_public_reader_cpu_costs() {
    use std::hint::black_box;
    use std::time::Instant;
    use tritium_format::{AdditivePayload, TritPackageReader};
    assert!(
        !black_box(cfg!(debug_assertions)),
        "profile requires --release"
    );
    let layout = AdditiveLayout {
        cols: 8_388_611,
        ..layout()
    };
    let raw = payload(layout);
    let bytes = package(layout, &raw);
    let count = UniformPlan::new(layout, 3).unwrap().trit_count();
    // Independent small synthetic canonical-byte oracle, never model weights.
    let canonical: Vec<_> = (0..count).map(|i| (i % 3) as u8).collect();
    let raw_hash = blake3::hash(&raw);
    let canonical_hash = blake3::hash(&canonical);
    let borrowed = AdditivePayload::new_uniform(layout, 3, &raw).unwrap();
    let reference = read_trit_model(&bytes).unwrap();
    let package_id = reference.package_id();
    let model_id = reference.model_id("imported").unwrap();
    let tensor_id = reference.tensor_digest("imported", "weight").unwrap();
    let names = [
        "encoded_hash",
        "canonical_bytes_hash",
        "payload_validation",
        "decoded_visit",
        "decoded_canonical_hash",
        "decoded_canonical_hash_4k",
        "decoded_canonical_hash_16k",
        "decoded_canonical_hash_64k",
        "retained_package",
        "retained_model",
    ];
    let mut samples: [Vec<u64>; 10] = std::array::from_fn(|_| Vec::new());
    // Two warmups, six balanced rotating/reversed rounds. Host CPU elapsed
    // times, in-memory sources, no disk, GPU, exclusivity or clock control.
    for round in 0..8 {
        for offset in 0..names.len() {
            let slot = if round % 2 == 0 {
                offset
            } else {
                names.len() - 1 - offset
            };
            let mode = (round + slot) % names.len();
            let started = Instant::now();
            match mode {
                0 => assert_eq!(blake3::hash(black_box(&raw)), raw_hash),
                1 => assert_eq!(blake3::hash(black_box(&canonical)), canonical_hash),
                2 => {
                    let value = AdditivePayload::new_uniform(layout, 3, black_box(&raw)).unwrap();
                    assert_eq!(black_box(value.trit_count()), count);
                }
                3 => {
                    let mut visited = 0;
                    borrowed
                        .visit_trits(|chunk| {
                            black_box(chunk);
                            visited += chunk.len();
                        })
                        .unwrap();
                    assert_eq!(visited, count);
                }
                4 => {
                    let mut hash = blake3::Hasher::new();
                    let mut staging = [0u8; 640];
                    borrowed
                        .visit_trits(|chunk| {
                            assert!(chunk.len() <= staging.len());
                            for (out, trit) in staging.iter_mut().zip(chunk) {
                                *out = (trit.get() + 1) as u8;
                            }
                            hash.update(&staging[..chunk.len()]);
                        })
                        .unwrap();
                    assert_eq!(hash.finalize(), canonical_hash);
                }
                5 => {
                    assert_eq!(coalesced_trit_hash::<4096>(&borrowed), canonical_hash);
                }
                6 => {
                    assert_eq!(coalesced_trit_hash::<16384>(&borrowed), canonical_hash);
                }
                7 => {
                    assert_eq!(coalesced_trit_hash::<65536>(&borrowed), canonical_hash);
                }
                8 => {
                    let value =
                        TritPackageReader::open(Cursor::new(black_box(&bytes)), limits(&bytes))
                            .unwrap();
                    assert_eq!(value.package_id(), package_id);
                }
                9 => {
                    let value =
                        TritModelReader::open(Cursor::new(black_box(&bytes)), limits(&bytes))
                            .unwrap();
                    assert_eq!(value.package_id(), package_id);
                    assert_eq!(value.model_id("imported").unwrap(), model_id);
                    assert_eq!(
                        value.tensor_digest("imported", "weight").unwrap(),
                        tensor_id
                    );
                }
                _ => unreachable!(),
            }
            if round >= 2 {
                samples[mode].push(u64::try_from(started.elapsed().as_nanos()).unwrap());
            }
        }
    }
    let modes: serde_json::Map<_, _> = names
        .into_iter()
        .zip(samples)
        .map(|(name, values)| {
            let mut sorted = values.clone();
            sorted.sort_unstable();
            let median = sorted[2] + (sorted[3] - sorted[2]) / 2;
            (
                name.to_owned(),
                serde_json::json!({"sample_count":values.len(),
            "sample_nanoseconds":values,"median_nanoseconds":median,
            "median_rule":"sorted midpoint, floor to nanoseconds for even sample counts"}),
            )
        })
        .collect();
    eprintln!(
        "READER_CPU_PROFILE {}",
        serde_json::json!({"qualified":false,
        "profile":"synthetic_cpu_reader_warm_v1","rows":layout.rows,"cols":layout.cols,
        "planes":3,"canonical_trits":count,"encoded_payload_bytes":raw.len(),
        "package_bytes":bytes.len(),"modes":modes,
        "model_id":blake3::Hash::from_bytes(*model_id.as_bytes()).to_hex().to_string(),
        "tensor_id":blake3::Hash::from_bytes(*tensor_id.as_bytes()).to_hex().to_string(),
        "scope":"Private synthetic warm in-memory CPU profile. Source/test-binary identity must be bound by the run wrapper. Not actual-model/disk/phase attribution, quality, physical memory, independent speed or release qualification."})
    );
}

fn coalesced_trit_hash<const N: usize>(
    value: &tritium_format::AdditivePayload<'_>,
) -> blake3::Hash {
    let mut hash = blake3::Hasher::new();
    let mut staging = [0u8; N];
    let mut used = 0;
    value
        .visit_trits(|mut chunk| {
            while !chunk.is_empty() {
                let count = (N - used).min(chunk.len());
                for (out, trit) in staging[used..used + count].iter_mut().zip(&chunk[..count]) {
                    *out = (trit.get() + 1) as u8;
                }
                used += count;
                chunk = &chunk[count..];
                if used == N {
                    hash.update(&staging);
                    used = 0;
                }
            }
        })
        .unwrap();
    hash.update(&staging[..used]);
    hash.finalize()
}
