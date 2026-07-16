use std::io::Write;

use tritium_format::ModelId;
use tritium_salt::{
    ContentId, TensorPutError, TensorRecordReceipt, TensorRecordSpec, TensorWorkError,
    TensorWorkStore,
};

#[derive(Debug, PartialEq, Eq)]
struct ProducerStop;

#[derive(Debug, PartialEq, Eq)]
struct SinkStop;

fn fixture_root(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "tritium-tensor-work-{label}-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ))
}

fn fixture_spec(payload_bytes: u64) -> TensorRecordSpec {
    TensorRecordSpec::new(
        ContentId::of_bytes(b"fixture tensor schema v1"),
        ModelId::from_digest([7; 32]),
        [11; 32],
        "model.layers.0.weight",
        vec![2, 3],
        b"fixture-metadata".to_vec(),
        payload_bytes,
    )
    .expect("valid fixture spec")
}

#[test]
fn record_streams_round_trips_and_reopens_with_bounded_visits() {
    let root = fixture_root("roundtrip");
    let _ = std::fs::remove_dir_all(&root);
    let store = TensorWorkStore::open(&root).expect("open tensor work store");
    let spec = fixture_spec(9);
    let payload = b"abcdefghi";

    let receipt = store
        .put(&spec, |writer| -> Result<(), ProducerStop> {
            writer.write_all(&payload[..4]).unwrap();
            writer.write_all(&payload[4..]).unwrap();
            Ok(())
        })
        .expect("publish record");
    let duplicate = store
        .put(&spec, |writer| -> Result<(), ProducerStop> {
            writer.write_all(payload).unwrap();
            Ok(())
        })
        .expect("idempotent publish");
    assert_eq!(duplicate, receipt);
    assert_eq!(receipt.info().name(), spec.name());
    assert_eq!(receipt.info().payload_bytes(), 9);

    let canonical_receipt = receipt.canonical_bytes().expect("encode receipt");
    let decoded =
        TensorRecordReceipt::from_canonical_bytes(&canonical_receipt).expect("decode receipt");
    assert_eq!(decoded, receipt);

    let mut reader = store.open_verified(&decoded).expect("reopen exact record");
    assert_eq!(reader.info(), receipt.info());
    let mut actual = Vec::new();
    reader
        .try_visit_payload(3, |chunk| -> Result<(), SinkStop> {
            assert!(!chunk.is_empty());
            assert!(chunk.len() <= 3);
            actual.extend_from_slice(chunk);
            Ok(())
        })
        .expect("visit payload");
    assert_eq!(actual, payload);

    let error = reader
        .try_visit_payload(2, |_| Err(SinkStop))
        .expect_err("preserve sink error");
    assert!(matches!(
        error,
        tritium_salt::TensorVisitError::Sink(SinkStop)
    ));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn failed_or_malformed_producers_never_publish_a_record() {
    let root = fixture_root("producer-failure");
    let _ = std::fs::remove_dir_all(&root);
    let store = TensorWorkStore::open(&root).expect("open tensor work store");
    let spec = fixture_spec(9);

    let error = store
        .put(&spec, |writer| {
            writer.write_all(b"abc").unwrap();
            Err(ProducerStop)
        })
        .expect_err("producer failure");
    assert!(matches!(error, TensorPutError::Producer(ProducerStop)));

    let error = store
        .put(&spec, |writer| -> Result<(), ProducerStop> {
            writer.write_all(b"short").unwrap();
            Ok(())
        })
        .expect_err("short producer");
    assert!(matches!(
        error,
        TensorPutError::Store(TensorWorkError::PayloadLengthMismatch {
            expected: 9,
            actual: 5
        })
    ));

    let error = store
        .put(&spec, |writer| -> Result<(), ProducerStop> {
            assert!(writer.write_all(b"payload-too-long").is_err());
            Ok(())
        })
        .expect_err("overrun producer");
    assert!(matches!(
        error,
        TensorPutError::Store(TensorWorkError::PayloadOverrun { expected: 9 })
    ));
    assert_eq!(std::fs::read_dir(store.objects_dir()).unwrap().count(), 0);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn reopened_records_fail_closed_after_exact_byte_tampering() {
    let root = fixture_root("tamper");
    let _ = std::fs::remove_dir_all(&root);
    let store = TensorWorkStore::open(&root).expect("open tensor work store");
    let spec = fixture_spec(9);
    let receipt = store
        .put(&spec, |writer| -> Result<(), ProducerStop> {
            writer.write_all(b"abcdefghi").unwrap();
            Ok(())
        })
        .expect("publish record");
    let path = store.record_path(receipt.record_id());
    let mut bytes = std::fs::read(&path).unwrap();
    let index = bytes.len() - 33;
    bytes[index] ^= 1;
    std::fs::write(&path, bytes).unwrap();

    assert!(matches!(
        store.open_verified(&receipt),
        Err(TensorWorkError::RecordIdMismatch) | Err(TensorWorkError::PayloadDigestMismatch)
    ));
    let _ = std::fs::remove_dir_all(root);
}
