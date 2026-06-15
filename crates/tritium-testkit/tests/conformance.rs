//! Integration tests: the harness must validate itself.
//!
//! 1. The reference backend reproduces every generated vector exactly (zero
//!    failures) — proves the harness's pack/upload/mpgemm/compare loop is sound.
//! 2. A deliberately-wrong backend produces failures — proves the harness can
//!    actually *detect* a non-conformant backend (a harness that always passes
//!    would be useless).
//! 3. JSONL save then load is a lossless roundtrip.

use core::any::Any;

use tritium_core::{GemmShape, TernaryFormat, Trit};
use tritium_format::{
    TQ1_0_BLOCK_BYTES, TQ2_0_BLOCK_BYTES, num_blocks, unpack_tq1_0_row, unpack_tq2_0_row,
};
use tritium_spec::{BackendError, DeviceBuffer, DeviceCaps, TernaryBackend};
use tritium_testkit::{
    ReferenceBackend, Tolerance, generate_vectors, load_vectors, run_conformance, save_vectors,
};

#[test]
fn reference_backend_passes_all_vectors() {
    let vectors = generate_vectors(0xA11CE, 64);
    assert!(vectors.len() > 64, "boundary set should be appended");
    let report = run_conformance(&ReferenceBackend::new(), &vectors, Tolerance::default());
    assert!(
        report.is_ok(),
        "reference backend must conform; {} failures, first: {:?}",
        report.failed.len(),
        report.failed.first().map(|f| (&f.id, f.reason.to_string()))
    );
    assert_eq!(report.passed, vectors.len());
    assert_eq!(report.total(), vectors.len());
}

#[test]
fn reference_backend_passes_both_formats() {
    // Spot-check that both tq2_0 and tq1_0 vectors are present and pass.
    let vectors = generate_vectors(7, 20);
    assert!(vectors.iter().any(|v| v.format == "tq2_0"));
    assert!(vectors.iter().any(|v| v.format == "tq1_0"));
    let report = run_conformance(&ReferenceBackend::new(), &vectors, Tolerance::default());
    assert!(report.is_ok());
}

// ---------------------------------------------------------------------------
// A deliberately-wrong backend: it unpacks correctly but adds a bias to the
// output, so every non-trivial case must be flagged.
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct WrongBuffer {
    trits: Vec<Trit>,
    bytes: usize,
}

impl DeviceBuffer for WrongBuffer {
    fn len_bytes(&self) -> usize {
        self.bytes
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug, Default)]
struct WrongBackend;

impl TernaryBackend for WrongBackend {
    fn device_id(&self) -> &str {
        "wrong"
    }
    fn capabilities(&self) -> DeviceCaps {
        DeviceCaps::new("wrong", "deliberately incorrect backend")
    }
    fn upload_weights(
        &self,
        packed: &[u8],
        shape: GemmShape,
        format: TernaryFormat,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        let GemmShape { n, k, .. } = shape;
        let nb = num_blocks(k);
        let block_bytes = match format {
            TernaryFormat::Tq2_0 => TQ2_0_BLOCK_BYTES,
            TernaryFormat::Tq1_0 => TQ1_0_BLOCK_BYTES,
            other => return Err(BackendError::UnsupportedFormat(other)),
        };
        let row_bytes = nb * block_bytes;
        let mut trits = vec![Trit::ZERO; n * k];
        let mut scratch = vec![half::f16::ONE; nb];
        for ni in 0..n {
            let row = &packed[ni * row_bytes..(ni + 1) * row_bytes];
            let trits_row = &mut trits[ni * k..ni * k + k];
            let res = match format {
                TernaryFormat::Tq2_0 => unpack_tq2_0_row(row, trits_row, &mut scratch),
                TernaryFormat::Tq1_0 => unpack_tq1_0_row(row, trits_row, &mut scratch),
                other => return Err(BackendError::UnsupportedFormat(other)),
            };
            res.map_err(|e| BackendError::Backend(e.to_string()))?;
        }
        Ok(Box::new(WrongBuffer {
            trits,
            bytes: packed.len(),
        }))
    }
    fn mpgemm(
        &self,
        act: &[f32],
        weights: &dyn DeviceBuffer,
        scales: &[f32],
        shape: GemmShape,
        _format: TernaryFormat,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let buf = weights
            .as_any()
            .downcast_ref::<WrongBuffer>()
            .ok_or_else(|| BackendError::InvalidInput("not a WrongBuffer".into()))?;
        let GemmShape { m, n, k } = shape;
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0.0f32;
                for ki in 0..k {
                    acc += act[mi * k + ki] * buf.trits[ni * k + ki].to_f32();
                }
                // The bug: a constant bias the reference does not have.
                out[mi * n + ni] = acc * scales[ni] + 1000.0;
            }
        }
        Ok(())
    }
}

#[test]
fn wrong_backend_is_flagged() {
    let vectors = generate_vectors(0xBADF00D, 32);
    let report = run_conformance(&WrongBackend, &vectors, Tolerance::default());
    assert!(
        !report.is_ok(),
        "a backend with a +1000 bias must fail conformance"
    );
    // The bias is large, so essentially every case with a nonzero contraction is
    // caught. At minimum, more than half must fail.
    assert!(
        report.failed.len() > vectors.len() / 2,
        "expected most cases to fail, got {}/{}",
        report.failed.len(),
        vectors.len()
    );
}

// ---------------------------------------------------------------------------
// JSONL roundtrip.
// ---------------------------------------------------------------------------

#[test]
fn jsonl_roundtrips() {
    let vectors = generate_vectors(0x5EED, 24);
    let dir = std::env::temp_dir();
    let path = dir.join(format!("tritium_testkit_rt_{}.jsonl", std::process::id()));

    save_vectors(&path, &vectors).expect("save");
    let loaded = load_vectors(&path).expect("load");
    std::fs::remove_file(&path).ok();

    assert_eq!(loaded, vectors, "JSONL save/load must be lossless");
}

#[test]
fn jsonl_skips_blank_lines() {
    let vectors = generate_vectors(1, 3);
    let dir = std::env::temp_dir();
    let path = dir.join(format!(
        "tritium_testkit_blank_{}.jsonl",
        std::process::id()
    ));

    // Write with extra blank lines interspersed.
    let mut body = String::new();
    for v in &vectors {
        body.push_str(&serde_json::to_string(v).unwrap());
        body.push_str("\n\n"); // double newline => blank line between records
    }
    std::fs::write(&path, body).unwrap();
    let loaded = load_vectors(&path).expect("load");
    std::fs::remove_file(&path).ok();
    assert_eq!(loaded, vectors);
}

#[test]
fn load_missing_file_errors() {
    let path = std::env::temp_dir().join("tritium_testkit_does_not_exist_xyz.jsonl");
    assert!(load_vectors(&path).is_err());
}

#[test]
fn malformed_line_errors_with_line_number() {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("tritium_testkit_bad_{}.jsonl", std::process::id()));
    std::fs::write(&path, "{\"id\":\"ok\"\nnot json at all\n").unwrap();
    let err = load_vectors(&path).unwrap_err();
    std::fs::remove_file(&path).ok();
    // Just assert it is an error and renders; line number is in the Display.
    assert!(!err.to_string().is_empty());
}
