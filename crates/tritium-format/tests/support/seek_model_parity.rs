//! Production-interface parity, shared by ordinary and compact model fixtures.
use std::io::Cursor;
use tritium_format::TritPackageReadLimits;
use tritium_format::trit_model::{
    TritModelError, TritModelReadError, TritModelReader, read_trit_model,
};

pub(super) fn limits(bytes: &[u8]) -> TritPackageReadLimits {
    TritPackageReadLimits {
        max_source_bytes: bytes.len() as u64,
        max_manifest_bytes: bytes.len(),
        max_directory_bytes: bytes.len(),
    }
}
pub(super) fn assert_matches(bytes: &[u8]) {
    let borrowed = read_trit_model(bytes);
    // Match the existing borrowed reader's global metadata caps. Deliberately
    // smaller explicit receiving caps have different refusal precedence.
    let mut receiving = limits(bytes);
    receiving.max_source_bytes = u64::MAX;
    receiving.max_manifest_bytes = usize::MAX;
    receiving.max_directory_bytes = usize::MAX;
    let retained = TritModelReader::open(Cursor::new(bytes), receiving);
    match (borrowed, retained) {
        (Ok(model), Ok(mut reader)) => {
            assert_eq!(reader.manifest(), model.manifest());
            assert_eq!(reader.package_id(), model.package_id());
            assert_eq!(
                reader
                    .validate_complete_payload_inventory()
                    .map_err(|e| e.to_string()),
                model
                    .validate_complete_payload_inventory()
                    .map_err(|e| e.to_string())
            );
            for level in &model.manifest().levels {
                assert_eq!(
                    reader.model_id(&level.name).unwrap(),
                    model.model_id(&level.name).unwrap()
                );
                for tensor in &level.tensors {
                    let (meta, expected) = model.tensor(&level.name, &tensor.name).unwrap();
                    assert_eq!(reader.tensor_meta(&level.name, &tensor.name).unwrap(), meta);
                    assert_eq!(
                        reader.tensor_byte_len(&level.name, &tensor.name).unwrap(),
                        expected.len() as u64
                    );
                    assert_eq!(
                        reader.tensor_digest(&level.name, &tensor.name).unwrap(),
                        model.tensor_digest(&level.name, &tensor.name).unwrap()
                    );
                    let mut at = 0;
                    reader
                        .visit_tensor(&level.name, &tensor.name, |chunk| -> Result<(), ()> {
                            assert!(chunk.len() <= 64 * 1024);
                            assert_eq!(chunk, &expected[at..at + chunk.len()]);
                            at += chunk.len();
                            Ok(())
                        })
                        .unwrap();
                    assert_eq!(at, expected.len());
                }
            }
            for asset in &model.manifest().assets {
                let expected = model.asset(&asset.name).unwrap();
                assert_eq!(
                    reader.asset_byte_len(&asset.name).unwrap(),
                    expected.len() as u64
                );
                let mut at = 0;
                reader
                    .visit_asset(&asset.name, |chunk| -> Result<(), ()> {
                        assert!(chunk.len() <= 64 * 1024);
                        assert_eq!(chunk, &expected[at..at + chunk.len()]);
                        at += chunk.len();
                        Ok(())
                    })
                    .unwrap();
                assert_eq!(at, expected.len());
            }
            reader.verify().unwrap();
        }
        (Err(expected), Err(TritModelReadError::Model(actual))) => {
            assert_eq!(actual.to_string(), expected.to_string())
        }
        (
            Err(expected),
            Err(TritModelReadError::Source(tritium_format::TritPackageReadError::Format(actual))),
        ) => assert_eq!(
            TritModelError::Package(actual).to_string(),
            expected.to_string()
        ),
        (expected, actual) => panic!("borrowed/retained mismatch: {expected:?} / {actual:?}"),
    }
}
