//! End-to-end test of `tritium quantize --fisher <sidecar>` (plan 0039 step 4).
//!
//! Synthesizes a tiny fp32 weight `.safetensors` plus a Fisher sensitivity sidecar (same tensor
//! name/shape) whose per-weight Fisher is large on the SECOND half of the rows and ~zero on the
//! first. Runs the quantizer twice — once with the sidecar, once uniform — reads back the emitted
//! SALT bundle, and asserts the Fisher run concentrated planes on the high-Fisher rows, more than
//! the uniform baseline does. Exercises the whole path: arg parse → sidecar load →
//! `tile_sensitivity` → `Sensitivity::Custom` → allocator → bundle.

use std::path::PathBuf;
use std::process::Command;

use tritium_format::read_salt_bundle;

const ROWS: usize = 8;
const K: usize = 8;

fn tritium_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_tritium"))
}

/// Build a minimal fp32 safetensors blob: `[u64 header_len][JSON header][f32 data]`.
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

/// Per-row plane counts of tensor `w` in a SALT bundle file.
fn plane_counts(path: &PathBuf) -> Vec<usize> {
    let bytes = std::fs::read(path).expect("read bundle");
    let tensors = read_salt_bundle(&bytes).expect("parse bundle");
    let t = tensors.iter().find(|t| t.name == "w").expect("tensor w");
    t.salt_rows.iter().map(|r| r.plane_count()).collect()
}

#[test]
fn fisher_sidecar_concentrates_planes_on_high_fisher_rows() {
    let dir = std::env::temp_dir().join(format!("tritium-fisher-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let weight_path = dir.join("w.safetensors");
    let fisher_path = dir.join("fisher.safetensors");
    let out_fisher = dir.join("out_fisher.tslb");
    let out_uniform = dir.join("out_uniform.tslb");

    // Every row is the same non-trivial pattern → identical reconstruction curves, so only the
    // Fisher sidecar distinguishes the rows. High Fisher on the SECOND half (rows 4..8): the
    // allocator breaks ties toward the LOW index, so a uniform run favors the first half — putting
    // the tie-break AGAINST the concentration we expect, making the gate conservative.
    let row = [0.5f32, -0.3, 0.8, -0.1, 0.6, -0.4, 0.2, -0.7];
    let w: Vec<f32> = (0..ROWS).flat_map(|_| row).collect();
    let fisher: Vec<f32> = (0..ROWS)
        .flat_map(|r| [if r >= ROWS / 2 { 1000.0f32 } else { 1e-4 }; K])
        .collect();

    std::fs::write(&weight_path, build_safetensors(&[("w", ROWS, K, w)])).expect("write w");
    std::fs::write(&fisher_path, build_safetensors(&[("w", ROWS, K, fisher)]))
        .expect("write fisher");

    let run = |extra: &[&str], out: &PathBuf| {
        let mut args = vec![
            "quantize",
            "--input",
            weight_path.to_str().unwrap(),
            "--output",
            out.to_str().unwrap(),
            "--ladder",
            "itf",
            "--bpw",
            "2.0",
        ];
        args.extend_from_slice(extra);
        let status = Command::new(tritium_bin())
            .args(&args)
            .status()
            .expect("run tritium quantize");
        assert!(status.success(), "quantize failed: {args:?}");
    };
    run(&["--fisher", fisher_path.to_str().unwrap()], &out_fisher);
    run(&[], &out_uniform);

    let pf = plane_counts(&out_fisher);
    let pu = plane_counts(&out_uniform);
    // Planes on the high-Fisher half (rows 4..8).
    let f_second: usize = pf[ROWS / 2..].iter().sum();
    let f_first: usize = pf[..ROWS / 2].iter().sum();
    let u_second: usize = pu[ROWS / 2..].iter().sum();
    println!(
        "0039 CLI Fisher gate: Fisher planes/row {pf:?} (hi-Fisher half {f_second}, other {f_first}); \
         Uniform {pu:?} (same half {u_second})"
    );

    // Within the Fisher run, the high-Fisher half wins — despite the tie-break favoring the other half.
    assert!(
        f_second > f_first,
        "Fisher must spend more planes on the high-Fisher (second) half: {pf:?}"
    );
    // And the sidecar redirected planes to that half relative to the uniform baseline (same weights).
    assert!(
        f_second > u_second,
        "Fisher sidecar must shift planes to the high-Fisher half vs uniform: Fisher {f_second} vs Uniform {u_second} ({pf:?} vs {pu:?})"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn mis_shaped_fisher_sidecar_is_rejected() {
    let dir = std::env::temp_dir().join(format!("tritium-fisher-badshape-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let weight_path = dir.join("w.safetensors");
    let fisher_path = dir.join("fisher.safetensors");
    let out = dir.join("out.tslb");

    // Weight is [ROWS, K]; the Fisher sidecar has the WRONG number of rows (would silently mis-map
    // if only element count were checked). Must be a clean hard error, not a bad allocation.
    let w: Vec<f32> = (0..ROWS * K).map(|i| (i as f32 % 5.0) - 2.0).collect();
    let bad_rows = ROWS / 2;
    let fisher: Vec<f32> = vec![1.0; bad_rows * K];
    std::fs::write(&weight_path, build_safetensors(&[("w", ROWS, K, w)])).expect("write w");
    std::fs::write(
        &fisher_path,
        build_safetensors(&[("w", bad_rows, K, fisher)]),
    )
    .expect("write fisher");

    let output = Command::new(tritium_bin())
        .args([
            "quantize",
            "--input",
            weight_path.to_str().unwrap(),
            "--output",
            out.to_str().unwrap(),
            // --fisher allocates planes per group, which only the itf fitter does; the geometric
            // ladder gives every group the same plane count and refuses the flag outright. Ask for
            // itf so this exercises the SHAPE check it is actually about.
            "--ladder",
            "itf",
            "--bpw",
            "2.0",
            "--fisher",
            fisher_path.to_str().unwrap(),
        ])
        .output()
        .expect("run tritium quantize");
    assert!(
        !output.status.success(),
        "a mis-shaped Fisher sidecar must be rejected"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("shape"),
        "error should name the shape mismatch; got: {stderr}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
