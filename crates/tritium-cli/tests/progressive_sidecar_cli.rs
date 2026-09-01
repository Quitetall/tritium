//! End-to-end gate for progressive sparse SALT sidecar emission.

use std::path::{Path, PathBuf};
use std::process::Command;

use tritium_format::{SALT_MAGIC, SALT_PROGRESSIVE_VERSION, read_salt_bundle};

const ROWS: usize = 8;
const K: usize = 8;

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let path =
            std::env::temp_dir().join(format!("tritium-progressive-cli-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("mkdir");
        Self(path)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

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

fn run_quantize(input: &Path, output: &Path, format: &str) {
    let result = Command::new(tritium_bin())
        .args([
            "quantize",
            "--input",
            input.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--bpw",
            "3.0",
            "--ladder",
            "itf",
            "--format",
            format,
        ])
        .output()
        .expect("run quantize");
    assert!(
        result.status.success(),
        "{format} quantize failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
}

#[test]
fn progressive_sidecar_emits_v2_rows_that_roundtrip() {
    let dir = TestDir::new();
    let weight_path = dir.0.join("w.safetensors");
    let progressive_path = dir.0.join("w-progressive.tslb");
    let legacy_path = dir.0.join("w-legacy.tslb");
    let weights: Vec<f32> = (0..ROWS * K)
        .map(|i| ((i as f32 + 0.5) * 0.37).sin())
        .collect();
    std::fs::write(&weight_path, build_safetensors(&weights)).expect("write weights");

    run_quantize(&weight_path, &progressive_path, "sidecar-progressive");
    run_quantize(&weight_path, &legacy_path, "sidecar");

    let progressive_bytes = std::fs::read(&progressive_path).expect("read progressive bundle");
    let legacy_bytes = std::fs::read(&legacy_path).expect("read legacy bundle");
    assert_ne!(
        progressive_bytes, legacy_bytes,
        "progressive output must use distinct row framing"
    );
    let row_header = progressive_bytes
        .windows(SALT_MAGIC.len())
        .position(|window| window == SALT_MAGIC)
        .expect("embedded SALT row");
    assert_eq!(
        progressive_bytes[row_header + SALT_MAGIC.len()],
        SALT_PROGRESSIVE_VERSION
    );
    let progressive = read_salt_bundle(&progressive_bytes).expect("read progressive bundle");
    let legacy = read_salt_bundle(&legacy_bytes).expect("read legacy bundle");
    assert_eq!(progressive, legacy, "v1 and v2 must decode identically");
    assert_eq!(progressive.len(), 1);
    assert_eq!(progressive[0].name, "w");
    assert_eq!(progressive[0].salt_rows.len(), ROWS);
}
