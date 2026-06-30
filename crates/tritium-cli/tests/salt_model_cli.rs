//! End-to-end test of the `tritium report salt-model` subcommand.
//!
//! Synthesizes a tiny fp32 safetensors file (no external deps — the container is a
//! u64-LE header length + JSON header + tensor data), runs the report, and asserts
//! the reconstruction-fidelity invariants on the JSON: every tensor is accounted for,
//! and more bits-per-weight strictly improves fidelity (lower relative-Frobenius
//! error, higher cosine). Exercises the whole path: arg parsing → safetensors parse
//! → per-tensor SALT quantize → reconstruction accumulation → JSON emit.

use std::path::PathBuf;
use std::process::Command;

fn tritium_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_tritium"))
}

/// Build a minimal fp32 safetensors blob: `[u64 header_len][JSON header][f32 data]`.
/// `tensors` is `(name, rows, k, values)` with `values.len() == rows*k`, row-major.
fn build_safetensors(tensors: &[(&str, usize, usize, Vec<f32>)]) -> Vec<u8> {
    let mut data: Vec<u8> = Vec::new();
    let mut header = String::from("{");
    for (i, (tn, rows, k, vals)) in tensors.iter().enumerate() {
        assert_eq!(vals.len(), rows * k, "{tn} value count");
        let start = data.len();
        for &v in vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        if i > 0 {
            header.push(',');
        }
        header.push_str(&format!(
            "\"{tn}\":{{\"dtype\":\"F32\",\"shape\":[{rows},{k}],\"data_offsets\":[{start},{}]}}",
            data.len()
        ));
    }
    header.push('}');
    let hbytes = header.into_bytes();

    let mut buf = Vec::with_capacity(8 + hbytes.len() + data.len());
    buf.extend_from_slice(&(hbytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(&hbytes);
    buf.extend_from_slice(&data);
    buf
}

/// Write a single-file safetensors into the temp dir; returns its path.
fn write_safetensors(name: &str, tensors: &[(&str, usize, usize, Vec<f32>)]) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "tritium-saltmodel-{}-{}.safetensors",
        name,
        std::process::id()
    ));
    std::fs::write(&p, build_safetensors(tensors)).expect("write safetensors");
    p
}

/// Create a fresh temp directory unique to this process + `tag`.
fn temp_dir(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("tritium-saltmodel-{}-{}", tag, std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).expect("mkdir");
    p
}

/// A deterministic non-ternary matrix (so the T=1 floor has real, reducible error).
fn synth(rows: usize, k: usize, seed: f32) -> Vec<f32> {
    (0..rows * k)
        .map(|i| (i as f32 * 0.017 + seed).sin() * 1.3 + (i as f32 * 0.0031).cos() * 0.4)
        .collect()
}

#[test]
fn salt_model_reports_fidelity_and_improves_with_bpw() {
    let path = write_safetensors(
        "fidelity",
        &[
            ("w.a", 8, 512, synth(8, 512, 0.0)),
            ("w.b", 4, 256, synth(4, 256, 1.0)),
        ],
    );
    let out = Command::new(tritium_bin())
        .args(["report", "salt-model", "--input"])
        .arg(&path)
        .args(["--budgets", "1.585,3.0", "--format", "json"])
        .output()
        .expect("spawn tritium");
    let _ = std::fs::remove_file(&path);

    assert!(
        out.status.success(),
        "salt-model failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let json: serde_json::Value =
        serde_json::from_str(&stdout).unwrap_or_else(|e| panic!("parse JSON ({e}): {stdout}"));

    assert_eq!(json["report"], "salt-model");
    assert_eq!(json["tensors"], 2);
    assert_eq!(json["total_params"], (8 * 512 + 4 * 256) as u64);

    let budgets = json["budgets"].as_array().expect("budgets array");
    assert_eq!(budgets.len(), 2);
    let frob = |b: &serde_json::Value| b["frob_rel"].as_f64().unwrap();
    let cos = |b: &serde_json::Value| b["cosine"].as_f64().unwrap();
    // budgets are reported in input order: 1.585 then 3.0.
    assert!(
        frob(&budgets[1]) < frob(&budgets[0]),
        "frob_rel must drop with more bpw: {} !< {}",
        frob(&budgets[1]),
        frob(&budgets[0])
    );
    assert!(
        cos(&budgets[1]) > cos(&budgets[0]),
        "cosine must rise with more bpw: {} !> {}",
        cos(&budgets[1]),
        cos(&budgets[0])
    );
    // The whole-model cosine is a real similarity in [-1, 1].
    assert!(cos(&budgets[0]) > 0.0 && cos(&budgets[1]) <= 1.0 + 1e-9);
}

#[test]
fn salt_model_loads_sharded_via_index_json() {
    // Two shards + a HF-style index.json; the report must process tensors from BOTH.
    let dir = temp_dir("sharded");
    std::fs::write(
        dir.join("shard1.safetensors"),
        build_safetensors(&[("w.a", 8, 512, synth(8, 512, 0.0))]),
    )
    .unwrap();
    std::fs::write(
        dir.join("shard2.safetensors"),
        build_safetensors(&[("w.b", 4, 256, synth(4, 256, 1.0))]),
    )
    .unwrap();
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        r#"{"metadata":{"total_size":0},"weight_map":{"w.a":"shard1.safetensors","w.b":"shard2.safetensors"}}"#,
    )
    .unwrap();

    // Pass the directory: salt-model should discover the index and stream both shards.
    let out = Command::new(tritium_bin())
        .args(["report", "salt-model", "--input"])
        .arg(&dir)
        .args(["--budgets", "2.0", "--format", "json"])
        .output()
        .expect("spawn tritium");
    let _ = std::fs::remove_dir_all(&dir);

    assert!(
        out.status.success(),
        "sharded salt-model failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let json: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&out.stdout)).expect("parse JSON");
    assert_eq!(json["tensors"], 2, "both shards' tensors must be counted");
    assert_eq!(json["total_params"], (8 * 512 + 4 * 256) as u64);
}

#[test]
fn salt_model_errors_on_no_2d_tensors() {
    // Only a 1D tensor — nothing to quantize.
    let path = write_safetensors("only1d", &[("bias", 1, 16, synth(1, 16, 0.0))]);
    let out = Command::new(tritium_bin())
        .args(["report", "salt-model", "--input"])
        .arg(&path)
        .args(["--budgets", "2.0"])
        .output()
        .expect("spawn tritium");
    let _ = std::fs::remove_file(&path);

    assert!(!out.status.success(), "expected failure on a 1D-only model");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("no 2D weight tensors"),
        "expected a no-2D-tensors error, got: {stderr}"
    );
    assert!(!stderr.contains("panicked"), "must not panic: {stderr}");
}

#[test]
fn salt_model_help_lists_flags() {
    let out = Command::new(tritium_bin())
        .args(["report", "salt-model", "--help"])
        .output()
        .expect("spawn tritium");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    for flag in [
        "--input",
        "--budgets",
        "--sensitivity",
        "--scale-group",
        "--per-tensor",
    ] {
        assert!(stdout.contains(flag), "help missing {flag}: {stdout}");
    }
}
