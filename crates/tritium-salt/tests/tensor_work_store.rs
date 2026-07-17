use std::io::Write;

use tritium_format::ModelId;
use tritium_salt::{
    ContentId, TensorPayloadValidator, TensorPutError, TensorRecordReceipt, TensorRecordSpec,
    TensorValidatedPutError, TensorWorkError, TensorWorkStore,
};

#[derive(Debug, PartialEq, Eq)]
struct ProducerStop;

#[derive(Debug, PartialEq, Eq)]
struct SinkStop;

#[derive(Debug, PartialEq, Eq)]
struct ValidatorStop;

#[derive(Debug, Default)]
struct ExactPayloadValidator {
    payload: Vec<u8>,
}

#[derive(Debug, Default)]
struct RejectingPayloadValidator;

impl TensorPayloadValidator for RejectingPayloadValidator {
    type Error = ValidatorStop;
    type Output = ();

    fn try_push(&mut self, bytes: &[u8]) -> Result<(), Self::Error> {
        if bytes.contains(&b'x') {
            Err(ValidatorStop)
        } else {
            Ok(())
        }
    }

    fn finish(self) -> Result<Self::Output, Self::Error> {
        Ok(())
    }
}

impl TensorPayloadValidator for ExactPayloadValidator {
    type Error = ValidatorStop;
    type Output = Vec<u8>;

    fn try_push(&mut self, bytes: &[u8]) -> Result<(), Self::Error> {
        self.payload.extend_from_slice(bytes);
        Ok(())
    }

    fn finish(self) -> Result<Self::Output, Self::Error> {
        if self.payload == b"abcdefghi" {
            Ok(self.payload)
        } else {
            Err(ValidatorStop)
        }
    }
}

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

    let error = store
        .put(&spec, |writer| writer.write_all(b"payload-too-long"))
        .expect_err("writer failure outranks propagated producer error");
    assert!(matches!(
        error,
        TensorPutError::Store(TensorWorkError::PayloadOverrun { expected: 9 })
    ));
    assert_eq!(std::fs::read_dir(store.objects_dir()).unwrap().count(), 0);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn terminal_validation_failure_never_publishes_an_exact_length_record() {
    let root = fixture_root("terminal-validation-failure");
    let _ = std::fs::remove_dir_all(&root);
    let store = TensorWorkStore::open(&root).expect("open tensor work store");
    let spec = fixture_spec(9);

    let error = store
        .put_validated(
            &spec,
            ExactPayloadValidator::default(),
            |writer| -> Result<(), ProducerStop> {
                writer.write_all(b"abcdxfghi").unwrap();
                Ok(())
            },
        )
        .expect_err("terminal validator failure");
    assert!(matches!(
        error,
        TensorValidatedPutError::Validator(ValidatorStop)
    ));
    assert_eq!(std::fs::read_dir(store.objects_dir()).unwrap().count(), 0);
    assert_eq!(
        std::fs::read_dir(store.root().join(".tmp"))
            .unwrap()
            .count(),
        0
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn streaming_validation_failure_preserves_type_and_removes_staging() {
    let root = fixture_root("streaming-validation-failure");
    let _ = std::fs::remove_dir_all(&root);
    let store = TensorWorkStore::open(&root).expect("open tensor work store");
    let spec = fixture_spec(9);

    let error = store
        .put_validated(
            &spec,
            RejectingPayloadValidator,
            |writer| -> Result<(), ProducerStop> {
                writer.write_all(b"abcdxfghi").unwrap();
                Ok(())
            },
        )
        .expect_err("streaming validator failure");
    assert!(matches!(
        error,
        TensorValidatedPutError::Validator(ValidatorStop)
    ));
    assert_eq!(std::fs::read_dir(store.objects_dir()).unwrap().count(), 0);
    assert_eq!(
        std::fs::read_dir(store.root().join(".tmp"))
            .unwrap()
            .count(),
        0
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn validated_put_returns_semantics_and_direct_visit_checks_the_same_record() {
    let root = fixture_root("validated-roundtrip");
    let _ = std::fs::remove_dir_all(&root);
    let store = TensorWorkStore::open(&root).expect("open tensor work store");
    let spec = fixture_spec(9);

    let (receipt, validated) = store
        .put_validated(
            &spec,
            ExactPayloadValidator::default(),
            |writer| -> Result<(), ProducerStop> {
                writer.write_all(b"abc").unwrap();
                writer.write_all(b"defghi").unwrap();
                Ok(())
            },
        )
        .expect("publish validated record");
    assert_eq!(validated, b"abcdefghi");

    let mut reopened = Vec::new();
    store
        .try_visit_verified(&receipt, 2, |chunk| -> Result<(), SinkStop> {
            reopened.extend_from_slice(chunk);
            Ok(())
        })
        .expect("visit exact record once");
    assert_eq!(reopened, b"abcdefghi");
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

#[cfg(unix)]
#[test]
fn store_rejects_symlinked_ancestor_directories() {
    use std::os::unix::fs::symlink;

    let root = fixture_root("ancestor-symlink");
    let _ = std::fs::remove_dir_all(&root);
    let redirect = root.join("redirect");
    std::fs::create_dir_all(&redirect).unwrap();
    let link = root.join("linked");
    symlink(&redirect, &link).unwrap();

    assert!(matches!(
        TensorWorkStore::open(&link.join("nested")),
        Err(TensorWorkError::InvalidPath(_))
    ));
    assert!(!redirect.join("nested").exists());
    let _ = std::fs::remove_dir_all(root);
}
