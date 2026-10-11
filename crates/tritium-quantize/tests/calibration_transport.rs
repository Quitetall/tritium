use std::io::{Read, Write};
use tritium_format::trit_calibration::{
    decode_calibration_cache_manifest, encode_calibration_cache_manifest,
};
use tritium_format::{TritBlob, TritBlobKind, read_trit_package, write_trit_package};
use tritium_quantize::engine::{
    CalibrationBatch, CalibrationCache, CalibrationCacheBuilder, CalibrationInput,
    CalibrationOutputBatch,
};
use tritium_schema::*;

fn cache(consumer: CalibrationConsumer, output: CalibrationOutput) -> CalibrationCache {
    let request = CalibrationCacheRequest {
        tensor: QuantSourceTensor {
            name: "weight".into(),
            shape: vec![2, 3],
            precision: ScalePrecision::Bf16,
        },
        model: OriginalSourceId::from_bytes([7; 32]),
        consumer,
        output,
        sample_start: 9,
        limits: CalibrationCacheLimits {
            max_payload_bytes: 100_000,
            max_samples: 10,
            max_windows: 5,
        },
    };
    let mut b = CalibrationCacheBuilder::new(
        request,
        b"{\"tokenizer\": 1}",
        b"exact declared objective bytes",
    )
    .unwrap();
    for _ in 0..2 {
        b.capture(
            &[10, 11],
            CalibrationBatch {
                samples: 2,
                input: if consumer == CalibrationConsumer::Matmul {
                    CalibrationInput::Activations(&[-0., 1., 2., 3., 4., 5.])
                } else {
                    CalibrationInput::GatherIdentity
                },
                output: match output {
                    CalibrationOutput::None => CalibrationOutputBatch::None,
                    CalibrationOutput::DenseFisher => {
                        CalibrationOutputBatch::Dense(&[0.5, -0., 1., 2.])
                    }
                    _ => CalibrationOutputBatch::Indexed {
                        indices: &[0, 1],
                        factors: &[-0., 2.],
                    },
                },
                weights: Some(&[0.5, 2.]),
                mask: Some(&[true, false]),
            },
        )
        .unwrap();
    }
    b.finish().unwrap()
}
fn read(
    c: &CalibrationCache,
    bytes: &[u8],
) -> Result<CalibrationCache, tritium_quantize::engine::CalibrationCacheError> {
    CalibrationCache::decode_package(bytes, c.id(), c.binding().request.limits, 1_000_000)
}

#[test]
fn durable_roundtrip_all_encodings_preserves_bits_identity_and_common_replay() {
    for (consumer, output) in [
        (CalibrationConsumer::Matmul, CalibrationOutput::DenseFisher),
        (
            CalibrationConsumer::Matmul,
            CalibrationOutput::IndexedFisher,
        ),
        (CalibrationConsumer::Matmul, CalibrationOutput::None),
        (
            CalibrationConsumer::GatherIdentity,
            CalibrationOutput::IndexedFisher,
        ),
    ] {
        let c = cache(consumer, output);
        let bytes = c.encode_package(1_000_000).unwrap();
        let decoded = read(&c, &bytes).unwrap();
        assert_eq!(decoded.id(), c.id());
        assert_eq!(decoded.binding(), c.binding());
        assert_eq!(decoded.encode_package(1_000_000).unwrap(), bytes);
        assert_eq!(decoded.tokenizer(), c.tokenizer());
        assert_eq!(decoded.objective(), c.objective());
        for partition in [
            CalibrationPartition::Full,
            CalibrationPartition::Groups { columns: 2 },
        ] {
            let damping = if output == CalibrationOutput::None {
                0.
            } else {
                0.125
            };
            let original = c.collect(partition, damping, 1_000_000).unwrap();
            let replay = decoded
                .collect_chunked(partition, damping, 1_000_000, 1)
                .unwrap();
            assert_eq!(original.factors(), replay.factors());
            for (a, b) in original.inputs().iter().zip(replay.inputs()) {
                assert_eq!(a.as_slice(), b.as_slice());
            }
        }
    }
}

fn rewrite(
    c: &CalibrationCache,
    change: impl FnOnce(&mut CalibrationCacheManifest, &mut Vec<(TritBlobKind, Vec<u8>)>),
) -> Vec<u8> {
    let bytes = c.encode_package(1_000_000).unwrap();
    let p = read_trit_package(&bytes).unwrap();
    let mut m = decode_calibration_cache_manifest(p.manifest()).unwrap();
    let mut assets = p
        .blobs()
        .iter()
        .map(|i| (i.kind.clone(), p.blob(i).unwrap().to_vec()))
        .collect();
    change(&mut m, &mut assets);
    let manifest = encode_calibration_cache_manifest(&m).unwrap();
    let blobs: Vec<_> = assets
        .iter()
        .map(|(kind, bytes)| TritBlob {
            kind: kind.clone(),
            bytes,
        })
        .collect();
    write_trit_package(&manifest, &blobs).unwrap()
}
fn asset<'a>(assets: &'a mut [(TritBlobKind, Vec<u8>)], wanted: &str) -> &'a mut Vec<u8> {
    &mut assets
        .iter_mut()
        .find(|(kind, _)| *kind == TritBlobKind::Asset(wanted.into()))
        .unwrap()
        .1
}

#[test]
fn valid_envelopes_with_changed_context_payload_or_inventory_fail_closed() {
    let c = cache(CalibrationConsumer::Matmul, CalibrationOutput::DenseFisher);
    let changed = [
        rewrite(&c, |m, _| {
            m.binding.request.model = OriginalSourceId::from_bytes([8; 32])
        }),
        rewrite(&c, |m, _| m.binding.source.tokens = FitInputDigest([8; 32])),
        rewrite(&c, |m, _| m.binding.tokenizer = BlobId::from_bytes([8; 32])),
        rewrite(&c, |m, _| m.binding.objective = BlobId::from_bytes([8; 32])),
        rewrite(&c, |_, a| asset(a, "tokenizer")[0] ^= 1),
        rewrite(&c, |_, a| asset(a, "objective")[0] ^= 1),
        rewrite(&c, |_, a| asset(a, "window/0/tokens")[0] ^= 1),
        rewrite(&c, |_, a| {
            asset(a, "window/0/input")[..4].copy_from_slice(&0f32.to_le_bytes())
        }),
        rewrite(&c, |_, a| {
            asset(a, "window/0/input")[12..16].copy_from_slice(&f32::NAN.to_le_bytes())
        }),
        rewrite(&c, |_, a| {
            asset(a, "window/0/output")[..4].copy_from_slice(&1f32.to_le_bytes())
        }),
        rewrite(&c, |_, a| {
            asset(a, "window/0/weights")[..8].copy_from_slice(&(-1f64).to_le_bytes())
        }),
        rewrite(&c, |_, a| asset(a, "window/0/mask")[0] = 2),
        rewrite(&c, |_, a| {
            asset(a, "window/0/input").pop();
        }),
        rewrite(&c, |_, a| {
            a[0].0 = TritBlobKind::DenseTensor(tritium_core::DType::F32);
        }),
        rewrite(&c, |_, a| {
            a[0].0 = TritBlobKind::Asset("unexpected".into());
        }),
        rewrite(&c, |_, a| {
            a.pop();
        }),
        rewrite(&c, |_, a| {
            a.push((TritBlobKind::Asset("extra".into()), vec![1]))
        }),
    ];
    for (index, bytes) in changed.iter().enumerate() {
        assert!(
            read_trit_package(bytes).is_ok(),
            "valid physical envelope {index}"
        );
        assert!(
            read(&c, bytes).is_err(),
            "must reject semantic tamper {index}"
        );
    }
    let indexed = cache(
        CalibrationConsumer::GatherIdentity,
        CalibrationOutput::IndexedFisher,
    );
    let bad = rewrite(&indexed, |_, a| {
        asset(a, "window/0/output")[..8].copy_from_slice(&2u64.to_le_bytes())
    });
    assert!(read(&indexed, &bad).is_err());
}

#[test]
fn receiving_limits_expected_identity_truncation_and_unknown_profile_reject() {
    let c = cache(CalibrationConsumer::Matmul, CalibrationOutput::DenseFisher);
    let bytes = c.encode_package(1_000_000).unwrap();
    assert!(c.encode_package(1).is_err());
    assert!(
        CalibrationCache::decode_package(
            &bytes,
            CalibrationCacheId::from_bytes([8; 32]),
            c.binding().request.limits,
            bytes.len()
        )
        .is_err()
    );
    for limits in [
        CalibrationCacheLimits {
            max_payload_bytes: c.binding().payload_bytes - 1,
            ..c.binding().request.limits
        },
        CalibrationCacheLimits {
            max_samples: 3,
            ..c.binding().request.limits
        },
        CalibrationCacheLimits {
            max_windows: 1,
            ..c.binding().request.limits
        },
    ] {
        assert!(CalibrationCache::decode_package(&bytes, c.id(), limits, bytes.len()).is_err());
    }
    assert!(
        CalibrationCache::decode_package(
            &bytes,
            c.id(),
            c.binding().request.limits,
            bytes.len() - 1
        )
        .is_err()
    );
    for n in [0, 63, bytes.len() - 1] {
        assert!(read(&c, &bytes[..n]).is_err());
    }
    let p = read_trit_package(&bytes).unwrap();
    let mut m = p.manifest().to_vec();
    let at = m
        .windows(b"tritium.calibration-cache.v1".len())
        .position(|s| s == b"tritium.calibration-cache.v1")
        .unwrap();
    m[at] = b'X';
    let blobs: Vec<_> = p
        .blobs()
        .iter()
        .map(|i| TritBlob {
            kind: i.kind.clone(),
            bytes: p.blob(i).unwrap(),
        })
        .collect();
    let unknown = write_trit_package(&m, &blobs).unwrap();
    assert!(read(&c, &unknown).is_err());
    let mut invalid = decode_calibration_cache_manifest(p.manifest()).unwrap();
    invalid.window_samples[0] = u64::MAX;
    assert!(encode_calibration_cache_manifest(&invalid).is_err());
    let mut corrupted = bytes.clone();
    corrupted[p.blobs()[0].offset as usize] ^= 1;
    assert!(read(&c, &corrupted).is_err());
}

#[test]
fn actual_file_roundtrip_uses_owned_scratch_and_removes_it() {
    struct FileGuard(std::path::PathBuf);
    impl Drop for FileGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let path = std::env::temp_dir().join(format!(
        "tritium-cache-transport-{}.trit",
        std::process::id()
    ));
    let c = cache(CalibrationConsumer::Matmul, CalibrationOutput::DenseFisher);
    let bytes = c.encode_package(1_000_000).unwrap();
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    let guard = FileGuard(path.clone());
    file.write_all(&bytes).unwrap();
    file.sync_all().unwrap();
    drop(file);
    let mut disk = Vec::new();
    std::fs::File::open(&path)
        .unwrap()
        .take(1_000_001)
        .read_to_end(&mut disk)
        .unwrap();
    assert_eq!(read(&c, &disk).unwrap().id(), c.id());
    drop(guard);
    assert!(!path.exists());
}
