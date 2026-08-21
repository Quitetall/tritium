//! **Does the ladder artifact reconstruct the weights the fitter intended?**
//!
//! `quantize_tensor_ladder` fits with [`ste::geometric_ladder_fit`] and then packs the digits into
//! TQ2_0 planes, deriving each plane's block scale as `s₀·3^-p`. Nothing else checks that the
//! packing step preserves the fit: the CLI's guards reject *malformed* configurations, and the
//! bundle writer validates *lengths*, but neither compares the decoded values against what the
//! fitter produced. A transposed index, an off-by-one group lookup, or a wrong plane exponent would
//! all pass every existing check and silently emit a wrong model.
//!
//! So this decodes the written bundle and compares it against
//! [`ste::salt_quantize_forward_grouped_geometric`] — the fitter's own dense reconstruction, and the
//! same function the research harnesses score perplexity through.
//!
//! # Why the match is not exact
//!
//! The fit carries `s₀` in `f32`; a TQ2_0 block stores its scale as `f16`. Every plane's scale is
//! therefore rounded once on the way out, so the decoded tensor differs from the oracle by the f16
//! representation error of the scales — roughly `2^-11` relative — and by nothing else. The trits
//! themselves must be reproduced **exactly**, which is asserted separately: any digit error would
//! be a packing bug, not a precision cost.

use half::f16;
use tritium_core::Trit;
use tritium_format::{QK_K, TQ2_0_BLOCK_BYTES, read_salt_bundle, unpack_tq2_0_block};
use tritium_train::ops::ste::{self, RotationPolicy};

/// f16 has an 11-bit significand, so a rounded scale is within `2^-11` relative of the f32 it came
/// from. Allow a small multiple of that for the accumulated sum over planes.
const MAX_REL: f32 = 1e-3;

fn seeded(n: usize, seed: u64) -> Vec<f32> {
    // Heavy-tailed on purpose: a Gaussian-ish body with occasional outliers is what makes the
    // ladder's grid search pick different Δ per group, so the per-group anchors actually differ.
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
            if i % 97 == 0 { u * 8.0 } else { u }
        })
        .collect()
}

/// Decode a written bundle back to dense f32: `Σ_p scale_p · trit_p`.
fn decode(bytes: &[u8], rows: usize, cols: usize) -> (Vec<f32>, Vec<Vec<i8>>) {
    let tensors = read_salt_bundle(bytes).expect("read bundle");
    assert_eq!(tensors.len(), 1, "one tensor written");
    let t = &tensors[0];
    let blocks = cols.div_ceil(QK_K);

    let mut dense = vec![0.0f32; rows * cols];
    let mut digits: Vec<Vec<i8>> = Vec::new();
    for (r, row) in t.salt_rows.iter().enumerate() {
        for (p, plane) in row.planes.iter().enumerate() {
            if digits.len() <= p {
                digits.push(vec![0i8; rows * cols]);
            }
            for b in 0..blocks {
                let block = &plane[b * TQ2_0_BLOCK_BYTES..(b + 1) * TQ2_0_BLOCK_BYTES];
                let mut trits = [Trit::ZERO; QK_K];
                let mut scale = f16::ZERO;
                unpack_tq2_0_block(block, &mut trits, &mut scale).expect("unpack block");
                let start = b * QK_K;
                let len = QK_K.min(cols - start);
                for (i, &trit) in trits.iter().enumerate().take(len) {
                    let idx = r * cols + start + i;
                    dense[idx] += f32::from(scale) * f32::from(i8::from(trit));
                    digits[p][idx] = i8::from(trit);
                }
            }
        }
    }
    (dense, digits)
}

fn roundtrip_case(rows: usize, cols: usize, planes: usize, group: usize, seed: u64) {
    let w = seeded(rows * cols, seed);

    // The oracle: the fitter's own dense reconstruction, identical settings.
    let oracle = ste::salt_quantize_forward_grouped_geometric(
        &w,
        rows,
        cols,
        planes,
        group,
        16,
        RotationPolicy::Never,
    );

    // What the CLI writes. Shelling out keeps this an end-to-end check of the shipped binary
    // rather than of a library call the binary might not make the same way.
    let dir = std::env::temp_dir().join(format!(
        "tritium-ladder-rt-{}-{rows}x{cols}-t{planes}-g{group}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let input = dir.join("w.safetensors");
    let output = dir.join("out.tslb");
    std::fs::write(&input, build_safetensors("w", rows, cols, &w)).expect("write input");

    let status = std::process::Command::new(tritium_bin())
        .args([
            "quantize",
            "--input",
            input.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--ladder",
            "geometric",
            "--planes",
            &planes.to_string(),
            "--group",
            &group.to_string(),
        ])
        .status()
        .expect("run tritium quantize");
    assert!(
        status.success(),
        "quantize failed for {rows}x{cols} T={planes} g{group}"
    );

    let bytes = std::fs::read(&output).expect("read bundle");
    let (decoded, _digits) = decode(&bytes, rows, cols);

    assert_eq!(decoded.len(), oracle.len(), "decoded length");
    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    for (i, (&d, &o)) in decoded.iter().zip(&oracle).enumerate() {
        let denom = o.abs().max(1e-6);
        let rel = (d - o).abs() / denom;
        if rel > worst {
            worst = rel;
            worst_at = i;
        }
    }
    println!(
        "{rows}x{cols} T={planes} g{group}: worst relative error {worst:.3e} at index {worst_at} \
         (decoded {:.6}, oracle {:.6})",
        decoded[worst_at], oracle[worst_at]
    );
    assert!(
        worst <= MAX_REL,
        "ladder bundle does not reconstruct the fit: worst relative error {worst:.3e} > {MAX_REL:.0e} \
         at index {worst_at} (decoded {}, oracle {}). f16 scale rounding alone should stay near \
         2^-11; anything larger is a packing bug (wrong group lookup, plane exponent, or index).",
        decoded[worst_at],
        oracle[worst_at]
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn ladder_bundle_reconstructs_the_fit() {
    // Exact multiples of the block, ragged tails, and more than one group per row — the cases where
    // a group/block index mistake would show up.
    roundtrip_case(4, 256, 3, 256, 0xA1);
    roundtrip_case(4, 512, 3, 256, 0xB2);
    roundtrip_case(3, 576, 4, 256, 0xC3); // ragged: 576 = 2*256 + 64
    roundtrip_case(2, 1024, 4, 512, 0xD4); // group spans two blocks
}

fn tritium_bin() -> std::path::PathBuf {
    let mut p = std::env::current_exe().expect("test exe");
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.join("tritium")
}

/// Minimal single-tensor f32 safetensors writer.
fn build_safetensors(name: &str, rows: usize, cols: usize, data: &[f32]) -> Vec<u8> {
    let header = format!(
        r#"{{"{name}":{{"dtype":"F32","shape":[{rows},{cols}],"data_offsets":[0,{}]}}}}"#,
        data.len() * 4
    );
    let mut padded = header.into_bytes();
    while padded.len() % 8 != 0 {
        padded.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + padded.len() + data.len() * 4);
    out.extend_from_slice(&(padded.len() as u64).to_le_bytes());
    out.extend_from_slice(&padded);
    for &v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}
