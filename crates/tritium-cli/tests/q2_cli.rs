use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use tritium_core::Trit;
use tritium_format::{
    GGML_TYPE_I2_S, GGML_TYPE_Q2_0, GgufValue, Q2_0_BLOCK_BYTES, TensorOut, pack_q2_0_row,
    write_gguf,
};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn tritium_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_tritium"))
}

fn temp_dir(tag: &str) -> PathBuf {
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "tritium-q2-cli-{tag}-{}-{sequence}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("create test directory");
    path
}

fn i2s_payload(trits: &[Trit], scale: f32) -> Vec<u8> {
    assert!(trits.len().is_multiple_of(128));
    let mut payload = vec![0u8; trits.len() / 4 + tritium_format::I2S_SCALE_BYTES];
    for (block_index, block) in trits.as_chunks::<128>().0.iter().enumerate() {
        for group_position in 0..32 {
            let mut byte = 0u8;
            for group in 0..4 {
                let code = (block[group * 32 + group_position].get() + 1) as u8;
                byte |= code << (6 - 2 * group);
            }
            payload[block_index * 32 + group_position] = byte;
        }
    }
    let scale_offset = trits.len() / 4;
    payload[scale_offset..scale_offset + 4].copy_from_slice(&scale.to_le_bytes());
    payload
}

fn write_i2s_model(path: &Path, scale: f32, metadata: BTreeMap<String, GgufValue>) {
    let trits = vec![Trit::POS; 256];
    let payload = i2s_payload(&trits, scale);
    let tensors = [TensorOut {
        name: "blk.0.w".to_owned(),
        dims: vec![256, 1],
        ggml_type: GGML_TYPE_I2_S,
        data: &payload,
    }];
    let bytes = write_gguf(3, &metadata, &tensors).expect("serialize I2_S model");
    std::fs::write(path, bytes).expect("write I2_S model");
}

fn write_q2_model(path: &Path, scales: [f16; 4], metadata: BTreeMap<String, GgufValue>) {
    let trits = vec![Trit::POS; 256];
    let mut payload = vec![0u8; scales.len() * Q2_0_BLOCK_BYTES];
    pack_q2_0_row(&trits, &scales, &mut payload).expect("pack Q2_0 row");
    let tensors = [TensorOut {
        name: "blk.0.w".to_owned(),
        dims: vec![256, 1],
        ggml_type: GGML_TYPE_Q2_0,
        data: &payload,
    }];
    let bytes = write_gguf(3, &metadata, &tensors).expect("serialize Q2_0 model");
    std::fs::write(path, bytes).expect("write Q2_0 model");
}

fn repack(input: &Path, output: &Path, target: &str) -> std::process::Output {
    Command::new(tritium_bin())
        .arg("repack")
        .arg("--input")
        .arg(input)
        .arg("--output")
        .arg(output)
        .arg("--to")
        .arg(target)
        .output()
        .expect("run tritium repack")
}

#[test]
fn repack_rejects_nonfinite_i2s_scale_without_replacing_destination() {
    let directory = temp_dir("nonfinite-i2s");
    let input = directory.join("input.gguf");
    let output = directory.join("output.gguf");
    write_i2s_model(&input, f32::NAN, BTreeMap::new());
    std::fs::write(&output, b"previous artifact").expect("write destination sentinel");

    let result = repack(&input, &output, "q2");

    assert!(!result.status.success(), "non-finite scale was accepted");
    assert!(
        String::from_utf8_lossy(&result.stderr).contains("non-finite"),
        "unexpected stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        std::fs::read(&output).expect("read preserved destination"),
        b"previous artifact"
    );
    std::fs::remove_dir_all(directory).expect("remove test directory");
}

#[test]
fn repack_rejects_i2s_scales_not_representable_as_nonzero_f16() {
    for (label, scale, expected_error) in [
        ("overflow", f32::MAX, "overflows finite f16"),
        ("underflow", f32::from_bits(1), "underflows to f16 zero"),
    ] {
        let directory = temp_dir(label);
        let input = directory.join("input.gguf");
        let output = directory.join("output.gguf");
        write_i2s_model(&input, scale, BTreeMap::new());

        let result = repack(&input, &output, "q2");

        assert!(!result.status.success(), "{label} scale was accepted");
        assert!(
            String::from_utf8_lossy(&result.stderr).contains(expected_error),
            "unexpected {label} stderr: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(!output.exists(), "{label} repack published an artifact");
        std::fs::remove_dir_all(directory).expect("remove test directory");
    }
}

#[test]
fn repack_rebinds_exporter_scale_metadata_to_i2s_source() {
    let directory = temp_dir("i2s-scale-metadata");
    let input = directory.join("input.gguf");
    let output = directory.join("output.gguf");
    let key = "tritium.i2s_scale.blk.0.w";
    let mut metadata = BTreeMap::new();
    metadata.insert(key.to_owned(), GgufValue::F32(0.125_01));
    write_i2s_model(&input, 0.125, metadata);

    let result = repack(&input, &output, "q2");

    assert!(
        result.status.success(),
        "repack failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let bytes = std::fs::read(&output).expect("read repacked model");
    let file = tritium_format::read_gguf(&bytes).expect("parse repacked model");
    assert_eq!(
        file.metadata.get(key),
        None,
        "f16-exact source scale must remove stale override"
    );
    std::fs::remove_dir_all(directory).expect("remove test directory");
}

#[test]
fn repack_rejects_nonfinite_q2_scale_without_replacing_destination() {
    let directory = temp_dir("nonfinite-q2");
    let input = directory.join("input.gguf");
    let output = directory.join("output.gguf");
    write_q2_model(
        &input,
        [f16::NAN, f16::ONE, f16::ONE, f16::ONE],
        BTreeMap::new(),
    );
    std::fs::write(&output, b"previous artifact").expect("write destination sentinel");

    let result = repack(&input, &output, "q2");

    assert!(!result.status.success(), "non-finite Q2 scale was accepted");
    assert!(
        String::from_utf8_lossy(&result.stderr).contains("non-finite"),
        "unexpected stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        std::fs::read(&output).expect("read preserved destination"),
        b"previous artifact"
    );
    std::fs::remove_dir_all(directory).expect("remove test directory");
}

#[test]
fn repack_rejects_stale_q2_scale_override_without_replacing_destination() {
    let directory = temp_dir("stale-q2-metadata");
    let input = directory.join("input.gguf");
    let output = directory.join("output.gguf");
    let mut metadata = BTreeMap::new();
    metadata.insert("tritium.i2s_scale.blk.0.w".to_owned(), GgufValue::F32(0.25));
    write_q2_model(&input, [f16::from_f32(0.125); 4], metadata);
    std::fs::write(&output, b"previous artifact").expect("write destination sentinel");

    let result = repack(&input, &output, "tq2");

    assert!(!result.status.success(), "stale Q2 override was accepted");
    assert!(
        String::from_utf8_lossy(&result.stderr).contains("stale scale metadata"),
        "unexpected stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        std::fs::read(&output).expect("read preserved destination"),
        b"previous artifact"
    );
    std::fs::remove_dir_all(directory).expect("remove test directory");
}

#[test]
fn repack_preserves_source_bound_q2_scale_override() {
    let directory = temp_dir("bound-q2-metadata");
    let input = directory.join("input.gguf");
    let output = directory.join("output.gguf");
    let key = "tritium.i2s_scale.blk.0.w";
    let exact = 0.125_01;
    let mut metadata = BTreeMap::new();
    metadata.insert(key.to_owned(), GgufValue::F32(exact));
    write_q2_model(&input, [f16::from_f32(exact); 4], metadata);

    let result = repack(&input, &output, "tq2");

    assert!(
        result.status.success(),
        "repack failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let bytes = std::fs::read(&output).expect("read repacked model");
    let file = tritium_format::read_gguf(&bytes).expect("parse repacked model");
    assert_eq!(file.metadata.get(key), Some(&GgufValue::F32(exact)));
    std::fs::remove_dir_all(directory).expect("remove test directory");
}

#[test]
fn repack_can_atomically_replace_its_input_path() {
    let directory = temp_dir("in-place");
    let model = directory.join("model.gguf");
    write_i2s_model(&model, 0.125, BTreeMap::new());

    let result = repack(&model, &model, "q2");

    assert!(
        result.status.success(),
        "in-place repack failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let bytes = std::fs::read(&model).expect("read in-place output");
    let file = tritium_format::read_gguf(&bytes).expect("parse in-place output");
    assert_eq!(
        file.tensor("blk.0.w").expect("repacked tensor").ggml_type,
        GGML_TYPE_Q2_0
    );
    std::fs::remove_dir_all(directory).expect("remove test directory");
}

#[cfg(unix)]
#[test]
fn repack_publication_does_not_follow_destination_symlink() {
    let directory = temp_dir("atomic-symlink");
    let input = directory.join("input.gguf");
    let output = directory.join("output.gguf");
    let victim = directory.join("victim.bin");
    write_i2s_model(&input, 0.125, BTreeMap::new());
    std::fs::write(&victim, b"do not replace").expect("write symlink target");
    std::os::unix::fs::symlink(&victim, &output).expect("create destination symlink");

    let result = repack(&input, &output, "q2");

    assert!(
        result.status.success(),
        "repack failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        std::fs::read(&victim).expect("read symlink target"),
        b"do not replace",
        "publication followed destination symlink"
    );
    assert!(
        !std::fs::symlink_metadata(&output)
            .expect("inspect output")
            .file_type()
            .is_symlink(),
        "output path must become the published regular file"
    );
    let bytes = std::fs::read(&output).expect("read published output");
    tritium_format::read_gguf(&bytes).expect("published output must parse");
    std::fs::remove_dir_all(directory).expect("remove test directory");
}

#[test]
fn sparsity_report_counts_zero_scale_q2_groups_as_semantic_zeros() {
    let directory = temp_dir("q2-semantic-sparsity");
    let model = directory.join("model.gguf");
    write_q2_model(
        &model,
        [f16::ZERO, f16::ONE, f16::ONE, f16::ONE],
        BTreeMap::new(),
    );

    let result = Command::new(tritium_bin())
        .arg("report")
        .arg("sparsity")
        .arg("--model")
        .arg(&model)
        .output()
        .expect("run sparsity report");

    assert!(
        result.status.success(),
        "report failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(
        stdout.contains("TOTAL: 256 weights | element zeros 25.00%"),
        "semantic zero group missing from report: {stdout}"
    );
    std::fs::remove_dir_all(directory).expect("remove test directory");
}
