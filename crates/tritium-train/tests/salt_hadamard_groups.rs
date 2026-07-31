//! Hadamard rotation front end + finer scale groups for SALT.
//!
//! SALT's failure mode is outliers: a plane adds at most `±mean|residual|`, so a heavy tail is
//! unreachable no matter how the trits fall. A Hadamard rotation is orthogonal (`H·H = I`, norm
//! preserving), so quantizing in the rotated basis and rotating back leaves the error norm unchanged
//! while spreading one large weight's energy across the whole group — turning a heavy-tailed group
//! into a nearly Gaussian one, which is the regime the AbsMean fitter is good at.
//!
//! These gates pin the transform's algebra, the backward compatibility of the grouped path, and the
//! actual reconstruction win on heavy-tailed data.

use tritium_train::ops::ste::{self, RotationPolicy};

fn seeded(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            lo + (s % 1000) as f32 / 1000.0 * (hi - lo)
        })
        .collect()
}

fn sse(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x - y) * f64::from(x - y))
        .sum()
}

/// `H` is an involution at this normalization and preserves the L2 norm — the two properties that make
/// "rotate, quantize, rotate back" error-preserving.
#[test]
fn hadamard_is_an_involution_and_preserves_norm() {
    for n in [2usize, 8, 128, 256] {
        let v = seeded(0x4A11 ^ n as u64, n, -3.0, 3.0);
        let norm = |x: &[f32]| {
            x.iter()
                .map(|&a| f64::from(a) * f64::from(a))
                .sum::<f64>()
                .sqrt()
        };

        let mut r = v.clone();
        ste::fast_hadamard(&mut r);
        assert!(
            (norm(&r) - norm(&v)).abs() < 1e-3 * norm(&v).max(1.0),
            "rotation must preserve the L2 norm (n={n})"
        );

        ste::fast_hadamard(&mut r); // twice == identity
        for (a, b) in r.iter().zip(&v) {
            assert!(
                (a - b).abs() < 1e-3,
                "H·H must be the identity (n={n}): {a} vs {b}"
            );
        }
    }
}

/// The grouped path with a whole-row group, no rotation and no ITF must reproduce the committed
/// quantizer exactly — so adopting it cannot silently move any existing result.
#[test]
fn grouped_defaults_reproduce_the_committed_quantizer() {
    for &(rows, cols) in &[(4usize, 64usize), (3, 128), (2, 576)] {
        let w = seeded(0x6009 ^ cols as u64, rows * cols, -1.5, 1.5);
        for t in 1..=3 {
            let base = ste::salt_quantize_forward(&w, rows, cols, t);
            let grouped = ste::salt_quantize_forward_grouped(
                &w,
                rows,
                cols,
                t,
                cols,
                0,
                RotationPolicy::Never,
            );
            assert_eq!(
                base, grouped,
                "group=cols, no rotate, no ITF must equal the greedy fit"
            );
        }
    }
}

/// The point of the front end: on heavy-tailed groups (a few large weights among a Gaussian bulk),
/// rotation must cut reconstruction error substantially — and it must NOT hurt on clean data.
#[test]
fn rotation_cuts_error_on_heavy_tails_and_is_neutral_on_clean_data() {
    let (rows, cols, group) = (8usize, 128usize, 128usize);

    // Sub-Gaussian (uniform) weights: rotation ADDS tails by CLT, so Always is expected to HURT —
    // measured ~5× worse. Auto must notice and decline to rotate, so it never loses.
    let clean = seeded(0xC1EA, rows * cols, -1.0, 1.0);
    let plain =
        ste::salt_quantize_forward_grouped(&clean, rows, cols, 3, group, 5, RotationPolicy::Never);
    let always =
        ste::salt_quantize_forward_grouped(&clean, rows, cols, 3, group, 5, RotationPolicy::Always);
    let auto =
        ste::salt_quantize_forward_grouped(&clean, rows, cols, 3, group, 5, RotationPolicy::Auto);
    let (e_plain, e_always, e_auto) = (
        sse(&plain, &clean),
        sse(&always, &clean),
        sse(&auto, &clean),
    );
    println!(
        "sub-Gaussian T=3 SSE: plain {e_plain:.4e} | always {e_always:.4e} | auto {e_auto:.4e}"
    );
    assert!(
        e_always > e_plain,
        "blanket rotation is expected to hurt sub-Gaussian data"
    );
    assert!(
        e_auto <= e_plain * (1.0 + 1e-9),
        "Auto must never be worse than not rotating: {e_auto:.4e} vs {e_plain:.4e}"
    );

    // Heavy-tailed: a few 8× outliers per group, which is what real weight tails look like.
    let mut heavy = seeded(0x11EA, rows * cols, -1.0, 1.0);
    for r in 0..rows {
        for k in [7usize, 40, 91] {
            heavy[r * cols + k] *= 8.0;
        }
    }
    let plain_h =
        ste::salt_quantize_forward_grouped(&heavy, rows, cols, 3, group, 5, RotationPolicy::Never);
    let rot_h =
        ste::salt_quantize_forward_grouped(&heavy, rows, cols, 3, group, 5, RotationPolicy::Always);
    let auto_h =
        ste::salt_quantize_forward_grouped(&heavy, rows, cols, 3, group, 5, RotationPolicy::Auto);
    let (e_ph, e_rh) = (sse(&plain_h, &heavy), sse(&rot_h, &heavy));
    let e_ah = sse(&auto_h, &heavy);
    assert!(
        e_ah <= e_ph * (1.0 + 1e-9),
        "Auto must capture the heavy-tail win"
    );
    println!(
        "heavy-tailed T=3 SSE: plain {e_ph:.4e} → rotated {e_rh:.4e} ({:.2}× better)",
        e_ph / e_rh
    );
    assert!(
        e_rh < e_ph * 0.6,
        "rotation must substantially cut heavy-tail error: {e_rh:.4e} vs {e_ph:.4e}"
    );
}

/// Finer scale groups reduce error monotonically at fixed plane count (they buy resolution at
/// `16/group` extra bits per weight per plane), and the bpw accounting reflects the cost.
#[test]
fn finer_groups_reduce_error_and_cost_bits() {
    let (rows, cols) = (4usize, 512usize);
    let w = seeded(0x6005, rows * cols, -1.0, 1.0);
    let mut prev = f64::INFINITY;
    for group in [512usize, 256, 128, 64] {
        let q =
            ste::salt_quantize_forward_grouped(&w, rows, cols, 2, group, 5, RotationPolicy::Never);
        let e = sse(&q, &w);
        assert!(
            e <= prev * 1.02,
            "finer groups should not be worse: {e:.4e} vs {prev:.4e}"
        );
        prev = e;
    }
    // The deployed TQ2_0 figure must fall out of the accounting.
    let bpw_256 = ste::ternary_bits_per_weight(1, 256);
    assert!(
        (bpw_256 - 2.0625).abs() < 1e-9,
        "group 256 must be TQ2_0's 2.0625 bpw, got {bpw_256}"
    );
    assert!(ste::ternary_bits_per_weight(3, 128) > ste::ternary_bits_per_weight(3, 256));
}
