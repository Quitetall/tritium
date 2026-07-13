//! End-to-end gate for progressive sparse SALT sidecar emission.

use std::path::PathBuf;
use std::process::Command;

use tritium_format::{SALT_MAGIC, SALT_PROGRESSIVE_VERSION, read_salt_bundle};

const ROWS: usize = 8;
const K: usize = 8;

fn tritium_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_tritium"))
}

fn build_safetensors(values: &[f32]) -> Vec<u8> {
    assert_eq!(values.len(), ROWS * K);
    let mut data = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        data.extend_from_slice(&value.to_le_bytes());
    }
    let header = format!(
        r#"{{"w":{{"dtype":"F32","shape":[{ROWS},{K}],"data_offsets":[0,{}]}}}}"#,
        data.len()
    );
    let mut bytes = Vec::with_capacity(8 + header.len() + data.len());
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&data);
    bytes
}

#[test]
fn progressive_sidecar_emits_v2_rows_that_roundtrip() {
    let dir = std::env::temp_dir().join(format!("tritium-progressive-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let weight_path = dir.join("w.safetensors");
    let output_path = dir.join("w-progressive.tslb");
    let weights: Vec<f32> = (0..ROWS * K)
        .map(|i| ((i as f32 + 0.5) * 0.37).sin())
        .collect();
    std::fs::write(&weight_path, build_safetensors(&weights)).expect("write weights");

    let output = Command::new(tritium_bin())
        .args([
            "quantize",
            "--input",
            weight_path.to_str().unwrap(),
            "--output",
            output_path.to_str().unwrap(),
            "--bpw",
            "3.0",
            "--format",
            "sidecar-progressive",
        ])
        .output()
        .expect("run progressive quantize");
    assert!(
        output.status.success(),
        "progressive quantize failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let bytes = std::fs::read(&output_path).expect("read progressive bundle");
    let row_header = bytes
        .windows(SALT_MAGIC.len())
        .position(|window| window == SALT_MAGIC)
        .expect("embedded SALT row");
    assert_eq!(
        bytes[row_header + SALT_MAGIC.len()],
        SALT_PROGRESSIVE_VERSION
    );
    let tensors = read_salt_bundle(&bytes).expect("read progressive bundle");
    assert_eq!(tensors.len(), 1);
    assert_eq!(tensors[0].name, "w");
    assert_eq!(tensors[0].salt_rows.len(), ROWS);

    let _ = std::fs::remove_dir_all(&dir);
}
