//! **Does the TQ2_0 f16 block scale destroy the ladder's fine planes on a real model?**
//!
//! `convert_anchor_diagnostic` localised a 2.76% perplexity gap to exactly one step: the fitter's
//! dense reconstruction versus the packed artifact (`D/C = 1.0276`). The evaluator was ruled out
//! (fp scores 18.240 through both the tape and `ModelRunner`), and the fit itself reproduces the
//! published anchor.
//!
//! The ladder assigns plane `p` the scale `s₀·3^-p`. A SALT bundle stores each plane as TQ2_0,
//! which carries **one f16 scale per 256-trit block**. f16's smallest *normal* value is 6.104e-5;
//! below that it degrades into subnormals with progressively fewer significand bits, and it flushes
//! to zero at 5.96e-8. At `T = 4` the finest plane is `s₀/27`, so any group whose anchor is small
//! enough loses that plane — silently, because a zero scale is a perfectly valid TQ2_0 block.
//!
//! `ladder_bundle_roundtrip` cannot see this: it fits synthetic uniform data, where every group's
//! `s₀` is the same order of magnitude and `s₀/27` is nowhere near the f16 floor. A real model's
//! per-group anchors span orders of magnitude.
//!
//! This measures the real distribution and the resulting reconstruction error, so the answer is
//! data rather than argument.
//!
//! ```text
//! TRITIUM_MODEL_DIR=$HOME/.cache/tritium-models/smollm2-360m \
//!   cargo test -p tritium-nn --release --test ladder_f16_scale_underflow -- --ignored --nocapture
//! ```

mod common;

use std::path::PathBuf;

use common::extract;
use half::f16;
use tritium_nn::ModelRunner;
use tritium_train::ops::ste::{self, RotationPolicy};

const PLANES: usize = 4;
const GROUP: usize = 256;
const GRID: usize = 16;

/// Smallest positive **normal** f16. Below this, significand bits are lost one at a time.
const F16_MIN_NORMAL: f64 = 6.103_515_625e-5;

fn relative_frobenius(reference: &[f32], other: &[f32]) -> f64 {
    let mut se = 0.0f64;
    let mut sw = 0.0f64;
    for (&a, &b) in reference.iter().zip(other) {
        let d = f64::from(a) - f64::from(b);
        se += d * d;
        sw += f64::from(a) * f64::from(a);
    }
    if sw > 0.0 { (se / sw).sqrt() } else { 0.0 }
}

#[test]
#[ignore = "needs a real fp master"]
fn f16_block_scales_preserve_every_ladder_plane() {
    let dir = PathBuf::from(
        std::env::var("TRITIUM_MODEL_DIR").expect("set TRITIUM_MODEL_DIR to an fp model directory"),
    );
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("load fp");
    let (_arch, fp, shapes) = extract(&runner);
    drop(runner);

    let mut anchors: Vec<f64> = Vec::new();
    let mut subnormal = [0usize; PLANES];
    let mut flushed = [0usize; PLANES];
    let mut total_groups = 0usize;

    // Reconstruction error introduced purely by rounding each plane's scale to f16, isolated from
    // every other step: same trits, same fit, only the scale representation differs.
    let mut worst_tensor = (String::new(), 0.0f64);
    let mut se_total = 0.0f64;
    let mut sw_total = 0.0f64;

    for (i, (w, &(n, k))) in fp.iter().zip(&shapes).enumerate() {
        let fits = ste::geometric_ladder_fit(w, n, k, PLANES, GROUP, GRID, RotationPolicy::Never);
        let groups_per_row = k.div_ceil(GROUP);

        // Dense reconstruction at f32 scales (what the fitter intends) and at f16 scales (what a
        // TQ2_0 block can actually hold).
        let mut exact = vec![0.0f32; n * k];
        let mut rounded = vec![0.0f32; n * k];
        for r in 0..n {
            for g in 0..groups_per_row {
                let (s0, planes) = &fits[r * groups_per_row + g];
                total_groups += 1;
                anchors.push(f64::from(*s0));
                let mut scale = f64::from(*s0);
                for (p, digits) in planes.iter().enumerate().take(PLANES) {
                    let as_f16 = f16::from_f64(scale);
                    if scale < F16_MIN_NORMAL {
                        subnormal[p] += 1;
                    }
                    if as_f16 == f16::ZERO && scale > 0.0 {
                        flushed[p] += 1;
                    }
                    let start = g * GROUP;
                    let len = GROUP.min(k - start);
                    for (j, &d) in digits.iter().enumerate().take(len) {
                        let idx = r * k + start + j;
                        exact[idx] += (scale * f64::from(d)) as f32;
                        rounded[idx] += f32::from(as_f16) * f32::from(d);
                    }
                    scale /= 3.0;
                }
            }
        }

        for (&a, &b) in exact.iter().zip(&rounded) {
            let d = f64::from(a) - f64::from(b);
            se_total += d * d;
            sw_total += f64::from(a) * f64::from(a);
        }
        let rel = relative_frobenius(&exact, &rounded);
        if rel > worst_tensor.1 {
            worst_tensor = (format!("weights[{i}] {n}x{k}"), rel);
        }
    }

    anchors.sort_by(|a, b| a.partial_cmp(b).expect("finite anchors"));
    let pct = |q: f64| anchors[((anchors.len() - 1) as f64 * q) as usize];
    println!("groups: {total_groups}");
    println!(
        "s0 percentiles: min {:.3e}  p1 {:.3e}  p50 {:.3e}  max {:.3e}",
        anchors[0],
        pct(0.01),
        pct(0.50),
        anchors[anchors.len() - 1]
    );
    println!("f16 min normal: {F16_MIN_NORMAL:.3e}");
    for p in 0..PLANES {
        println!(
            "plane {p} (s0/3^{p}): {:>9} subnormal ({:.3}%), {:>9} flushed to zero ({:.3}%)",
            subnormal[p],
            100.0 * subnormal[p] as f64 / total_groups as f64,
            flushed[p],
            100.0 * flushed[p] as f64 / total_groups as f64,
        );
    }
    let whole = (se_total / sw_total).sqrt();
    println!("\nrelative Frobenius error from f16 SCALE ROUNDING ALONE: {whole:.6}");
    println!("worst tensor: {} at {:.6}", worst_tensor.0, worst_tensor.1);
    println!(
        "for scale: the whole T=4 quantization itself costs ~0.025-0.038 relative error, and the \
         measured fit->artifact perplexity gap is 2.76%."
    );
}

/// **Do the artifact's decoded weights equal the fitter's reconstruction on a REAL model?**
///
/// `ladder_bundle_roundtrip` asks this on synthetic tensors up to 3x576. It passes, which proves
/// the packer is right for those shapes and nothing more. The real model has a 49152x960 tied
/// embedding, 320x960 kv projections, and 960 columns that split into three full 256-trit groups
/// plus a 192-wide tail on *every* tensor.
///
/// This decodes the bundle `tritium convert` actually wrote and compares it against
/// `salt_quantize_forward_grouped_geometric` — the same oracle the perplexity anchor was measured
/// through. It separates two very different diagnoses for the measured 2.76% fit-to-artifact
/// perplexity gap: bytes that decode to the wrong values, versus bytes that are right and a
/// runtime that computes with them differently.
#[test]
#[ignore = "needs a real fp master and a converted artifact"]
fn artifact_decodes_to_the_fit_on_a_real_model() {
    use tritium_format::{read_salt_bundle, salt_rows_to_dense};
    use tritium_nn::calibrate::weight_names;

    let dir = PathBuf::from(std::env::var("TRITIUM_MODEL_DIR").expect("set TRITIUM_MODEL_DIR"));
    let converted =
        PathBuf::from(std::env::var("TRITIUM_CONVERTED_DIR").expect("set TRITIUM_CONVERTED_DIR"));

    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("load fp");
    let (arch, fp, shapes) = extract(&runner);
    drop(runner);
    let names = weight_names(&arch);

    let bytes = std::fs::read(converted.join("model.tslb")).expect("read bundle");
    let tensors = read_salt_bundle(&bytes).expect("parse bundle");
    println!("bundle carries {} tensors", tensors.len());

    let mut worst = (String::new(), 0.0f64);
    let mut se_total = 0.0f64;
    let mut sw_total = 0.0f64;
    let mut checked = 0usize;

    for (i, (name, &(n, k))) in names.iter().zip(&shapes).enumerate() {
        let Some(t) = tensors.iter().find(|t| &t.name == name) else {
            panic!("bundle has no tensor `{name}`");
        };
        assert_eq!((t.rows, t.k), (n, k), "{name} shape");

        let oracle = ste::salt_quantize_forward_grouped_geometric(
            &fp[i],
            n,
            k,
            PLANES,
            GROUP,
            GRID,
            RotationPolicy::Never,
        );
        let decoded = salt_rows_to_dense(&t.salt_rows).expect("decode");
        assert_eq!(decoded.len(), oracle.len(), "{name} length");

        for (&a, &b) in oracle.iter().zip(&decoded) {
            let d = f64::from(a) - f64::from(b);
            se_total += d * d;
            sw_total += f64::from(a) * f64::from(a);
        }
        let rel = relative_frobenius(&oracle, &decoded);
        if rel > worst.1 {
            worst = (format!("{name} [{n}x{k}]"), rel);
        }
        checked += 1;
    }

    let whole = (se_total / sw_total).sqrt();
    println!("checked {checked} tensors");
    println!("whole-model relative Frobenius error, artifact vs fitter: {whole:.6}");
    println!("worst tensor: {} at {:.6}", worst.0, worst.1);
    println!(
        "f16 scale rounding alone accounts for 0.000271. Anything materially above that is a \
         packing defect on real shapes; anything at that level means the bytes are correct and the \
         2.76% perplexity gap lives in the runtime, not the artifact."
    );
}
