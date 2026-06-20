//! Runnable v0.4.0 SALT example (ADR 0002 U9): quantize an fp32 weight matrix with SALT,
//! write it to both container formats, read them back, and verify the dequantized weights
//! match. Run: `cargo run -p tritium-quantize --example salt_roundtrip`.

use tritium_format::{dequant_salt_row, read_salt_bundle, read_salt_gguf, write_salt_bundle, write_salt_gguf};
use tritium_quantize::{QuantConfig, ScaleGroup, Sensitivity, quantize_tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A small fp32 weight matrix: 4 output rows × 256 input features.
    let (rows, k) = (4usize, 256usize);
    let weights: Vec<f32> = (0..rows * k).map(|i| ((i as f32) * 0.013 - 1.7).sin()).collect();

    // SALT-quantize at 2.0 bits/weight (per-256-block base, residual planes as budget allows).
    let cfg = QuantConfig {
        budget_bpw: 2.0,
        t_min: 1,
        t_max: 3,
        sensitivity: Sensitivity::Uniform,
        scale_group: ScaleGroup::Block,
    };
    let qt = quantize_tensor(&weights, rows, k, &cfg)?;
    println!(
        "quantized {rows}x{k} -> {:.3} logical bpw, {} SALT rows",
        qt.logical_bpw(),
        qt.salt_rows.len(),
    );

    // Write both v0.4.0 containers from the same SALT rows.
    let bundle = write_salt_bundle(&[("weight", &qt.salt_rows)])?;
    let gguf = write_salt_gguf(&[("weight", &qt.salt_rows)])?;
    println!("sidecar bundle: {} bytes; SALT-in-GGUF: {} bytes", bundle.len(), gguf.len());

    // Read both back and confirm they reconstruct identical dequantized weights.
    let from_bundle = read_salt_bundle(&bundle)?;
    let from_gguf = read_salt_gguf(&gguf)?;
    assert_eq!(from_bundle.len(), 1);
    assert_eq!(from_gguf.len(), 1);
    assert_eq!(from_bundle[0].name, "weight");
    assert_eq!(from_bundle[0].rows, rows);
    assert_eq!(from_bundle[0].k, k);

    for r in 0..rows {
        let a = dequant_salt_row(&from_bundle[0].salt_rows[r])?;
        let b = dequant_salt_row(&from_gguf[0].salt_rows[r])?;
        assert_eq!(a, b, "bundle vs gguf dequant mismatch on row {r}");
        assert_eq!(a.len(), k);
    }

    println!("OK: bundle and GGUF round-trip to identical dequantized weights");
    Ok(())
}
