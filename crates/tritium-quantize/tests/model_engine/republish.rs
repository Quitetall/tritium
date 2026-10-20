//! Exact artifact export reuses the shared verifier and directory commit.
use super::*;
use std::{fs, path::PathBuf};

struct Scratch(PathBuf);
impl Scratch {
    fn new(name: &str) -> Self {
        let root =
            std::env::temp_dir().join(format!("tritium-republish-{name}-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        Self(root)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn exact_original_bytes_and_log_republish_without_new_identity_or_overwrite() {
    let root = Scratch::new("success");
    for precision in [
        ScalePrecision::F16,
        ScalePrecision::Bf16,
        ScalePrecision::F32,
    ] {
        let mut original = source(precision);
        let mut memory = MemoryTensorSink::default();
        let fitted = Engine::run(&plan(&original), &mut original, &mut memory, recorder()).unwrap();
        let (bytes, sealed) = memory.artifact().unwrap();
        let log = sealed.jsonl().to_owned();
        let out = root.0.join(format!("{precision:?}"));
        let mut sink = DirectoryTensorSink::new(&out);
        let warnings = sink.publish_verified_model(bytes, &log).unwrap();
        let expected = if fs::File::open(&out)
            .and_then(|file| file.sync_all())
            .is_err()
        {
            vec![PublicationWarning::DirectorySync]
        } else {
            Vec::new()
        };
        assert_eq!(warnings, expected);
        assert_eq!(fs::read(out.join("model.trit")).unwrap(), bytes);
        assert_eq!(fs::read_to_string(out.join("evidence.jsonl")).unwrap(), log);
        assert!(!out.join(".pending.trit").exists());
        let model = read_trit_model(bytes).unwrap();
        assert_eq!(model.package_id().as_bytes(), fitted.package_id.as_bytes());
        assert_eq!(
            verify_jsonl(&log).unwrap().root_digest,
            fitted.evidence.log().root_digest
        );
        fs::write(out.join("foreign"), b"retain").unwrap();
        assert!(matches!(
            sink.publish_verified_model(bytes, &log),
            Err(QuantEngineError::DestinationExists)
        ));
        assert_eq!(fs::read(out.join("foreign")).unwrap(), b"retain");
        assert_eq!(fs::read(out.join("model.trit")).unwrap(), bytes);
    }
}

#[test]
fn corruption_and_foreign_complete_log_refuse_before_destination_creation() {
    let root = Scratch::new("invalid");
    let mut original = source(ScalePrecision::F32);
    let mut memory = MemoryTensorSink::default();
    Engine::run(&plan(&original), &mut original, &mut memory, recorder()).unwrap();
    let (bytes, sealed) = memory.artifact().unwrap();
    let mut foreign = source(ScalePrecision::F32);
    foreign.tensors.get_mut("weight").unwrap().bytes[0] ^= 1;
    let mut other = MemoryTensorSink::default();
    Engine::run(&plan(&foreign), &mut foreign, &mut other, recorder()).unwrap();
    let mut corrupt = bytes.to_vec();
    *corrupt.last_mut().unwrap() ^= 1;
    let noncanonical = format!(" {}", sealed.jsonl());
    for (name, package, log) in [
        ("corrupt", corrupt.as_slice(), sealed.jsonl()),
        ("malformed", bytes, "not jsonl"),
        ("noncanonical", bytes, noncanonical.as_str()),
        ("foreign", bytes, other.artifact().unwrap().1.jsonl()),
    ] {
        let out = root.0.join(name);
        assert!(
            DirectoryTensorSink::new(&out)
                .publish_verified_model(package, log)
                .is_err()
        );
        assert!(!out.exists());
    }
}

#[test]
fn standalone_fit_does_not_become_a_complete_conversion_during_export() {
    let root = Scratch::new("standalone");
    let mut original = source(ScalePrecision::F32);
    let mut memory = MemoryTensorSink::default();
    Engine::run(&plan(&original), &mut original, &mut memory, recorder()).unwrap();
    let mut standalone =
        EvidenceRecorder::new("not-a-conversion", [DIRECT_LEVELS_FITTED_REGISTRATION]).unwrap();
    let tensor = &original.tensors["weight"];
    let weights = tritium_format::trit_model::decode_dense_f32(
        "weight",
        &tensor.shape,
        tensor.precision,
        &tensor.bytes,
    )
    .unwrap();
    tritium_quantize::engine::fit_direct_levels_recorded(
        "weight",
        &direct(),
        &weights,
        None,
        &mut standalone,
        "fit",
        None,
        0,
    )
    .unwrap();
    let standalone = standalone.seal().unwrap();
    let out = root.0.join("refused");
    assert!(
        DirectoryTensorSink::new(&out)
            .publish_verified_model(memory.artifact().unwrap().0, standalone.jsonl())
            .is_err()
    );
    assert!(!out.exists());
}
