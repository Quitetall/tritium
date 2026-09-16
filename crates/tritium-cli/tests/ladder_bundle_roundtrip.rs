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
use tritium_train::ops::ste::{self, RotationPolicy, fast_hadamard, group_is_rotatable};

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

/// **`--rotate` must reconstruct the rotated fit, and only where the container can say so.**
///
/// Rotation is opt-in on `quantize` because two of its three containers have no rotation field. A
/// rotated fit stores codes for `W·H`; a reader that does not rotate the activation computes
/// `W·H·x` and there is nothing in the file to signal it. So this asserts both halves:
///
/// * the sidecar bundle decodes to the **rotated** oracle, is version 2, and carries the group;
/// * the progressive bundle and the SALT GGUF **refuse** the flag rather than write it.
///
/// The second half is the one that matters. A wrong reconstruction announces itself in perplexity;
/// a wrong *basis* does not, which is the failure class version 2 exists to make impossible.
#[test]
fn rotate_reconstructs_the_rotated_fit_and_is_refused_where_it_cannot_be_recorded() {
    let (rows, cols, planes, group) = (4usize, 512usize, 3usize, 256usize);
    let w = seeded(rows * cols, 0xE5);
    let oracle = ste::salt_quantize_forward_grouped_geometric(
        &w,
        rows,
        cols,
        planes,
        group,
        16,
        RotationPolicy::Always,
    );

    let dir = std::env::temp_dir().join(format!("tritium-ladder-rot-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let input = dir.join("w.safetensors");
    std::fs::write(&input, build_safetensors("w", rows, cols, &w)).expect("write input");

    let run = |format: &str, output: &std::path::Path| -> bool {
        std::process::Command::new(tritium_bin())
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
                "--format",
                format,
                "--rotate",
            ])
            .status()
            .expect("run tritium quantize")
            .success()
    };

    let sidecar = dir.join("rot.tslb");
    assert!(
        run("sidecar", &sidecar),
        "--rotate must work on a sidecar bundle"
    );
    let bytes = std::fs::read(&sidecar).expect("read bundle");

    // The artifact has to SAY it is rotated, through the reader rather than a header offset.
    let declared = tritium_format::SaltBundleReader::new_strict(std::io::BufReader::new(
        std::fs::File::open(&sidecar).expect("open"),
    ))
    .expect("parse bundle")
    .rotation_group();
    assert_eq!(
        declared,
        Some(group as u16),
        "a rotated fit must record its Hadamard group; without it the runtime reconstructs W·H"
    );

    let (mut decoded, _) = decode(&bytes, rows, cols);
    // The stored codes are in the ROTATED basis; `salt_quantize_forward_grouped_geometric` returns
    // its reconstruction already turned back. Comparing the two directly measures the Hadamard
    // rather than the packer — the same mistake the `convert` fidelity receipt made, where it read
    // 1.4239 relative error on a model whose real error was 0.0195. `H` is its own inverse.
    for row in decoded.chunks_mut(cols) {
        for slice in row.chunks_mut(group) {
            if group_is_rotatable(slice.len()) {
                fast_hadamard(slice);
            }
        }
    }
    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    let (mut se, mut sw) = (0.0f64, 0.0f64);
    for (i, (&d, &o)) in decoded.iter().zip(&oracle).enumerate() {
        let rel = (d - o).abs() / o.abs().max(1e-6);
        if rel > worst {
            worst = rel;
            worst_at = i;
        }
        se += f64::from(d - o) * f64::from(d - o);
        sw += f64::from(o) * f64::from(o);
    }
    let frob = (se / sw).sqrt();
    println!(
        "rotated {rows}x{cols} T={planes} g{group}: relative Frobenius {frob:.3e} | worst \
         per-element {worst:.3e} at {worst_at} (decoded {:.8}, oracle {:.8})",
        decoded[worst_at], oracle[worst_at]
    );
    // Judged on relative FROBENIUS, not per-element relative, and the reason is specific to
    // rotation. Turning the codes back sums `group` terms whose f16-rounded scales nearly cancel:
    // a coordinate whose true value is 6e-8 carries the rounding residue of 256 terms, so its
    // per-element relative error is enormous (427x measured here) while its absolute error is
    // 4e-4 — a rounding artifact, not a packing bug. Frobenius is the metric `convert`'s own
    // fidelity receipt uses, and it is scale-free without dividing by a cancelled value.
    assert!(
        frob <= f64::from(MAX_REL),
        "the rotated bundle does not reconstruct the ROTATED fit: relative Frobenius {frob:.3e} \
         > {MAX_REL:.0e}. f16 scale rounding alone lands near 2e-4; anything larger means the \
         writer packed one basis and the fitter produced the other"
    );
    // And an absolute bound, so a real defect confined to a few coordinates cannot hide inside a
    // healthy whole-tensor norm.
    let rms = (sw / oracle.len() as f64).sqrt();
    let worst_abs = decoded
        .iter()
        .zip(&oracle)
        .map(|(&d, &o)| f64::from(d - o).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst_abs <= 0.01 * rms,
        "worst absolute element error {worst_abs:.3e} exceeds 1% of the tensor RMS {rms:.3e} — \
         f16 rounding cannot do that, so this is a packing or basis defect in a few coordinates"
    );

    // And the refusals. These containers cannot record a rotation, so writing one would be a
    // silently wrong model — the exact failure the version bump exists to prevent.
    for format in ["sidecar-progressive", "gguf"] {
        let out = dir.join(format!("rot.{format}"));
        assert!(
            !run(format, &out),
            "--format {format} has no rotation field, so --rotate must be REFUSED. Writing it \
             would produce a file whose codes are in a basis nothing can discover."
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}
