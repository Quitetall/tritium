use super::*;
use crate::engine::OwnerScoreWindow;
use tritium_format::{TritBlob, TritBlobKind, read_trit_package, write_trit_package_bounded};
use tritium_schema::{OwnerInvocationMeta, ScalePrecision};

fn limits() -> OwnerScoreCacheLimits {
    OwnerScoreCacheLimits {
        max_payload_bytes: 4096,
        max_context_bytes: 1024,
        max_context_tensors: 8,
        max_invocations: 8,
        max_samples: 32,
    }
}
fn fixture() -> OwnerScoreCache {
    let uses = [
        OwnerScoreUse::Matmul {
            samples: 2,
            inputs: &[1.0, 2.0, 3.0, 4.0],
            adjoints: &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        },
        OwnerScoreUse::Gather {
            indices: &[0, 1, 0],
            adjoints: &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        },
    ];
    let observations = Arc::new(
        OwnerScoreWindow::new(3, 2, &uses, 4096)
            .unwrap()
            .snapshot()
            .unwrap(),
    );
    let tokens = cache_scalar::encode(&[0i64, 1, 0], |v| v.to_le_bytes()).unwrap();
    let meta = OwnerScoreWindowMeta {
        context: vec![
            OwnerContextMeta {
                name: "input_ids".into(),
                dtype: OwnerContextDtype::I64,
                shape: vec![3],
                conjugated: false,
                negated: false,
            },
            OwnerContextMeta {
                name: "mask".into(),
                dtype: OwnerContextDtype::Bool,
                shape: vec![2],
                conjugated: false,
                negated: false,
            },
        ],
        invocations: vec![
            OwnerInvocationMeta {
                ordinal: 3,
                paths: vec!["head".into()],
                role: OwnerScoreRole::Matmul,
                samples: 2,
            },
            OwnerInvocationMeta {
                ordinal: 5,
                paths: vec!["embed".into(), "shared".into()],
                role: OwnerScoreRole::Gather,
                samples: 3,
            },
        ],
    };
    OwnerScoreCache::new(
        OwnerScoreCacheInput {
            source: OriginalSourceId::from_bytes([1; 32]),
            tensor: QuantSourceTensor {
                name: "weight".into(),
                shape: vec![3, 2],
                precision: ScalePrecision::F32,
            },
            tokenizer: b"{}",
            objective: b"loss",
            meta,
            context: &[&tokens, &[1, 0]],
            observations,
        },
        limits(),
    )
    .unwrap()
}

#[test]
fn common_writer_reader_and_row_replay_are_exact() {
    let original = fixture();
    let bytes = original.encode_package(65536).unwrap();
    let reopened =
        OwnerScoreCache::decode_package(&bytes, original.binding(), limits(), 65536).unwrap();
    assert_eq!(reopened.binding(), original.binding());
    assert_eq!(reopened.encode_package(65536).unwrap(), bytes);
    for row in 0..3 {
        assert_eq!(
            original.row_score(row, 16).unwrap(),
            reopened.row_score(row, 16).unwrap()
        );
    }
    let mut streamed = Vec::new();
    let id = original.write_package_to(65536, &mut streamed).unwrap();
    assert_eq!(streamed, bytes);
    assert_eq!(id, read_trit_package(&bytes).unwrap().package_id());
}

#[test]
fn limits_and_expected_binding_reject_before_success() {
    let cache = fixture();
    let bytes = cache.encode_package(65536).unwrap();
    for bound in [
        OwnerScoreCacheLimits {
            max_payload_bytes: 1,
            ..limits()
        },
        OwnerScoreCacheLimits {
            max_context_bytes: 1,
            ..limits()
        },
        OwnerScoreCacheLimits {
            max_context_tensors: 1,
            ..limits()
        },
        OwnerScoreCacheLimits {
            max_invocations: 1,
            ..limits()
        },
        OwnerScoreCacheLimits {
            max_samples: 1,
            ..limits()
        },
    ] {
        assert!(OwnerScoreCache::decode_package(&bytes, cache.binding(), bound, 65536).is_err());
    }
    let mut expected = cache.binding().clone();
    expected.samples += 1;
    assert!(OwnerScoreCache::decode_package(&bytes, &expected, limits(), 65536).is_err());
    assert!(cache.encode_package(1).is_err());
    assert!(OwnerScoreCache::decode_package(&bytes, cache.binding(), limits(), 1).is_err());
}

#[test]
fn rehashed_container_tampering_still_rejects_logical_content() {
    let cache = fixture();
    let bytes = cache.encode_package(65536).unwrap();
    let package = read_trit_package(&bytes).unwrap();
    for target in [
        "tokenizer",
        "objective",
        "context/0",
        "invocation/0/input",
        "invocation/1/adjoint",
    ] {
        let owned: Vec<_> = package
            .blobs()
            .iter()
            .map(|info| {
                let mut value = package.blob(info).unwrap().to_vec();
                if info.kind == TritBlobKind::Asset(target.into()) {
                    value[0] ^= 1;
                }
                (info.kind.clone(), value)
            })
            .collect();
        let blobs: Vec<_> = owned
            .iter()
            .map(|(kind, bytes)| TritBlob {
                kind: kind.clone(),
                bytes,
            })
            .collect();
        let altered = write_trit_package_bounded(package.manifest(), &blobs, 65536).unwrap();
        assert!(read_trit_package(&altered).is_ok());
        assert!(
            OwnerScoreCache::decode_package(&altered, cache.binding(), limits(), 65536).is_err()
        );
    }
}

#[test]
fn context_numeric_boolean_and_token_validation_is_exact() {
    for (dtype, data) in [
        (OwnerContextDtype::F16, 0x7c00u16.to_le_bytes().to_vec()),
        (OwnerContextDtype::Bf16, 0x7f80u16.to_le_bytes().to_vec()),
        (OwnerContextDtype::F32, f32::NAN.to_le_bytes().to_vec()),
        (OwnerContextDtype::F64, f64::INFINITY.to_le_bytes().to_vec()),
        (
            OwnerContextDtype::C64,
            [f32::INFINITY.to_le_bytes(), 0f32.to_le_bytes()].concat(),
        ),
        (OwnerContextDtype::Bool, vec![2]),
    ] {
        let meta = OwnerContextMeta {
            name: "unused".into(),
            dtype,
            shape: vec![1],
            conjugated: false,
            negated: false,
        };
        assert!(validate_context(&meta, &data).is_err());
    }
    let meta = OwnerContextMeta {
        name: "input_ids".into(),
        dtype: OwnerContextDtype::I64,
        shape: vec![1],
        conjugated: false,
        negated: false,
    };
    for value in [-1i64, i64::MAX] {
        assert!(validate_context(&meta, &value.to_le_bytes()).is_err());
    }
    assert!(validate_context(&meta, &(u32::MAX as i64).to_le_bytes()).is_ok());
}

#[test]
fn write_failure_never_reports_a_success_identity() {
    struct Broken;
    impl std::io::Write for Broken {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("injected"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            panic!("cache writer must not flush");
        }
    }
    assert!(matches!(
        fixture().write_package_to(65536, &mut Broken),
        Err(OwnerScoreCacheError::Write(_))
    ));
}
