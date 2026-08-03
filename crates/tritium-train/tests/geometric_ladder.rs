//! **The balanced-ternary scale ladder** — `s_p = s₀·3^-(p-1)`.
//!
//! SALT's greedy residual expansion picks each plane's scale from the residual the previous plane
//! left (`s_p = mean|r_p|`, then ITF refines it). That ladder decays far too slowly and then breaks
//! down: measured ratios `s_{p+1}/s_p` run 0.41, 0.42, 0.58, 0.70 and eventually **exceed 1.0** — a
//! later plane taking a *larger* scale than the one before it. The consequence is that each added
//! plane buys roughly **1 dB** of reconstruction where the information-theoretic rate is
//! `1.585 bits × 6.02 = 9.54 dB`, i.e. added planes land on top of the levels already reachable
//! instead of subdividing the gaps between them.
//!
//! It is worth being precise about what is NOT wrong, because the obvious explanation is false:
//! the reachable levels do **not** collide. Enumerating `Σ ε_p s_p` over `ε ∈ {-1,0,+1}^T` for the
//! greedy ladder gives 27/27, 81/81, 729/729 and 6561/6561 *distinct* values on Gaussian, Laplace and
//! t₃ groups. Distinctness is fine; **spacing** is the defect.
//!
//! Fixing the ratio to exactly `1/3` is balanced ternary, and it makes
//! `Σ_p ε_p s_p = Δ·k` with `Δ = s₀·3^-(T-1)` a **bijection onto every integer `k` in `±(3^T-1)/2`** —
//! a uniform grid, all `3^T` levels, evenly spaced. That buys three things at once:
//!
//! 1. reconstruction that actually tracks the 9.54 dB/plane rate;
//! 2. an **O(T)** fit (one `round`, then digit extraction) instead of `3^T` enumeration — which is
//!    what makes the good fitter cheap enough to run inside a training step at all;
//! 3. **one** stored scale per group instead of `T`.
//!
//! These are Gate-C-style forward gates, not gradient gates: the ladder changes the forward
//! reconstruction rule only. [`ste::salt_quantize_vjp`] is untouched and still the identity STE.

use tritium_train::ops::ste::{self, RotationPolicy};

fn seeded(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            lo + (s % 100_000) as f32 / 100_000.0 * (hi - lo)
        })
        .collect()
}

/// Approximately-Gaussian fixture: sum of 12 uniforms, the classic Irwin–Hall trick. Real weight
/// groups are peaked and roughly symmetric, which is the regime the ladder is designed for.
fn gaussian(seed: u64, n: usize) -> Vec<f32> {
    let u = seeded(seed, n * 12, 0.0, 1.0);
    (0..n)
        .map(|i| u[i * 12..(i + 1) * 12].iter().sum::<f32>() - 6.0)
        .collect()
}

fn kmax(t: usize) -> i64 {
    (3_i64.pow(t as u32) - 1) / 2
}

/// **The bijection.** Feed a group that is exactly `{k·Δ}` for every integer `k` the ladder can
/// represent, and every weight must come back *exactly* — the reconstruction is a lattice point and
/// the fit is a nearest-lattice-point search, so there is no rounding left to do.
///
/// This is the property the whole design rests on: `T` ternary planes on a `1/3` ladder address
/// every integer in `±(3^T-1)/2` and nothing else. If it holds, the digit extraction is correct by
/// construction and no separate "are the trits right" test is needed.
#[test]
fn every_representable_integer_reconstructs_exactly() {
    for t in 1..=8usize {
        let km = kmax(t);
        // Δ = 1 keeps the arithmetic exact in f32 for every T ≤ 8 (3^8 = 6561 ≪ 2^24).
        let weights: Vec<f32> = (-km..=km).map(|k| k as f32).collect();
        let n = weights.len();

        // grid = 0 pins s₀ to the clipping-free anchor, so Δ comes back as exactly 1.
        let fit = ste::geometric_ladder_fit(&weights, 1, n, t, n, 0, RotationPolicy::Never);
        assert_eq!(fit.len(), 1, "one group");
        let (s0, planes) = &fit[0];
        assert_eq!(planes.len(), t, "T planes at T={t}");

        let delta = s0 / 3.0f32.powi(t as i32 - 1);
        assert!(
            (delta - 1.0).abs() < 1e-4,
            "T={t}: clipping-free anchor should give Δ=1, got {delta}"
        );

        for (i, &w) in weights.iter().enumerate() {
            let mut recon = 0.0f32;
            for (p, plane) in planes.iter().enumerate() {
                assert!(
                    (-1..=1).contains(&plane[i]),
                    "T={t} plane {p} weight {i}: trit {} out of range",
                    plane[i]
                );
                recon += s0 * 3.0f32.powi(-(p as i32)) * f32::from(plane[i]);
            }
            assert!(
                (recon - w).abs() < 1e-3,
                "T={t} k={w}: reconstructed {recon}, every lattice point must be exact"
            );
        }
    }
}

/// The forward reconstruction must agree with the trits the fit reports. These are two separate
/// code paths (one returns a dense buffer, one returns the symbols a packer would write), and a
/// disagreement between them is exactly the silent-corruption class the `fit_group` comment warns
/// about for the rotation mask.
#[test]
fn dense_forward_matches_the_reported_trits() {
    let (rows, cols, group) = (3usize, 128usize, 64usize);
    let w = gaussian(0xA11CE, rows * cols);
    for t in 1..=4usize {
        for &rot in &[RotationPolicy::Never, RotationPolicy::Always] {
            let dense =
                ste::salt_quantize_forward_grouped_geometric(&w, rows, cols, t, group, 8, rot);
            let fit = ste::geometric_ladder_fit(&w, rows, cols, t, group, 8, rot);
            assert_eq!(fit.len(), rows * cols.div_ceil(group));

            // Unrotated groups reconstruct directly from the trits; rotated ones live in the
            // Hadamard basis, so only compare those in the basis the trits were fitted in.
            if rot == RotationPolicy::Never {
                for (g, (s0, planes)) in fit.iter().enumerate() {
                    let r = g / cols.div_ceil(group);
                    let b = g % cols.div_ceil(group);
                    let base = r * cols + b * group;
                    for i in 0..group.min(cols - b * group) {
                        let mut recon = 0.0f32;
                        for (p, plane) in planes.iter().enumerate() {
                            recon += s0 * 3.0f32.powi(-(p as i32)) * f32::from(plane[i]);
                        }
                        let got = dense[base + i];
                        assert!(
                            (recon - got).abs() <= 1e-5 * recon.abs().max(1.0),
                            "T={t} group {g} elem {i}: trits give {recon}, forward gave {got}"
                        );
                    }
                }
            }
        }
    }
}

/// **The mechanism gate.** The ladder exists because greedy stops improving with `T`. On a Gaussian
/// group the geometric ladder must beat greedy+ITF by a wide and *growing* margin — that widening is
/// the claim, not the single-point win.
///
/// Deliberately loose thresholds: the point is the trend, and a tight bound here would be a fixture
/// tolerance rather than a fact about the method. Held-out perplexity, not this, decides whether the
/// ladder ships (`ste.rs` documents the proxy gap: per-layer MSE can anti-correlate with ppl).
#[test]
fn geometric_ladder_beats_greedy_and_the_gap_widens_with_planes() {
    let (rows, cols, group) = (8usize, 128usize, 128usize);
    let w = gaussian(0xBEEF, rows * cols);
    let sse = |q: &[f32]| -> f64 {
        q.iter()
            .zip(&w)
            .map(|(&a, &b)| f64::from(a - b) * f64::from(a - b))
            .sum()
    };

    let mut gains = Vec::new();
    for t in [3usize, 4, 6] {
        let greedy =
            ste::salt_quantize_forward_grouped(&w, rows, cols, t, group, 5, RotationPolicy::Never);
        let geo = ste::salt_quantize_forward_grouped_geometric(
            &w,
            rows,
            cols,
            t,
            group,
            16,
            RotationPolicy::Never,
        );
        let gain = sse(&greedy) / sse(&geo);
        println!(
            "T={t}: greedy SSE {:.4e}  geometric {:.4e}  gain {gain:.1}x",
            sse(&greedy),
            sse(&geo)
        );
        gains.push(gain);
    }

    assert!(gains[0] > 2.0, "T=3 gain {:.2}x should clear 2x", gains[0]);
    assert!(
        gains[2] > 50.0,
        "T=6 gain {:.2}x should clear 50x",
        gains[2]
    );
    assert!(
        gains[2] > gains[0] * 5.0,
        "the gap must WIDEN with planes: T=3 {:.1}x vs T=6 {:.1}x",
        gains[0],
        gains[2]
    );
}

/// More planes must keep paying. This is the property greedy loses — the thing that makes "why don't
/// more planes help" a real defect rather than a law of nature.
#[test]
fn each_added_plane_still_reduces_error() {
    let (rows, cols) = (4usize, 128usize);
    let w = gaussian(0xC0DE, rows * cols);
    let sse = |q: &[f32]| -> f64 {
        q.iter()
            .zip(&w)
            .map(|(&a, &b)| f64::from(a - b) * f64::from(a - b))
            .sum()
    };

    let mut prev = f64::INFINITY;
    for t in 1..=6usize {
        let e = sse(&ste::salt_quantize_forward_grouped_geometric(
            &w,
            rows,
            cols,
            t,
            128,
            16,
            RotationPolicy::Never,
        ));
        // 3x per plane is a deliberately weak floor; the theoretical rate is 9x in SSE terms.
        assert!(
            e < prev / 3.0,
            "T={t}: SSE {e:.4e} must be at least 3x better than T={} ({prev:.4e})",
            t - 1
        );
        prev = e;
    }
}

/// One `s₀` per group, not `T` scales — the ladder is determined by its anchor. At T=3/g128/B3 that
/// is 4.925 bpw against the free-scale path's 5.176.
#[test]
fn ladder_stores_one_scale_per_group_not_one_per_plane() {
    use tritium_format::salt_v2::SaltV2Codec;
    for t in 1..=4usize {
        for &group in &[128usize, 256] {
            let geo = ste::ternary_bits_per_weight_geometric(t, group, SaltV2Codec::B3, group);
            let free = ste::ternary_bits_per_weight_codec(t, group, SaltV2Codec::B3, group);
            let saved = (t as f64 - 1.0) * 16.0 / group as f64;
            assert!(
                (free - geo - saved).abs() < 1e-9,
                "T={t} g={group}: expected to save {saved} bpw, got {}",
                free - geo
            );
        }
    }
    // The headline configuration, pinned. Note B3 over a 128-trit run is `ceil(128/5) = 26` bytes =
    // **1.625** bits/trit, not the asymptotic 1.6 — so the free-scale rate is `3·(1.625 + 0.125) =
    // 5.25` and the ladder saves the two redundant f16 scales: 5.25 − 0.25 = 5.00.
    let bpw = ste::ternary_bits_per_weight_geometric(3, 128, SaltV2Codec::B3, 128);
    assert!(
        (bpw - 5.00).abs() < 1e-6,
        "T=3 g128 B3 should be 5.00 bpw, got {bpw}"
    );
}

/// A ragged final group (not a power of two) must never be rotated — no zero-padding, no phantom
/// weights — and must still fit. Same contract the ITF path holds.
#[test]
fn ragged_groups_are_fitted_and_never_rotated() {
    let (rows, cols, group) = (2usize, 200usize, 128usize); // 200 = 128 + 72, ragged tail
    let w = gaussian(0xF00D, rows * cols);
    let q = ste::salt_quantize_forward_grouped_geometric(
        &w,
        rows,
        cols,
        3,
        group,
        8,
        RotationPolicy::Always,
    );
    assert_eq!(q.len(), w.len());
    assert!(
        q.iter().all(|v| v.is_finite()),
        "ragged fit produced non-finite values"
    );

    let fit = ste::geometric_ladder_fit(&w, rows, cols, 3, group, 8, RotationPolicy::Always);
    assert_eq!(fit.len(), rows * 2);
    // The ragged group still carries 72 real trits per plane, not 128 padded ones.
    assert_eq!(fit[1].1[0].len(), 72);
}

/// A dead (all-zero) group must produce an all-zero fit rather than a NaN scale — the same early-out
/// the greedy path takes for a dead row.
#[test]
fn all_zero_group_fits_to_zero_without_nan() {
    let w = vec![0.0f32; 256];
    for t in 1..=3usize {
        let q = ste::salt_quantize_forward_grouped_geometric(
            &w,
            1,
            256,
            t,
            128,
            8,
            RotationPolicy::Never,
        );
        assert!(
            q.iter().all(|&v| v == 0.0),
            "T={t}: dead group must stay zero, got {q:?}"
        );
        let fit = ste::geometric_ladder_fit(&w, 1, 256, t, 128, 8, RotationPolicy::Never);
        for (s0, planes) in &fit {
            assert!(
                s0.is_finite(),
                "T={t}: dead group scale must be finite, got {s0}"
            );
            assert!(planes.iter().all(|p| p.iter().all(|&x| x == 0)));
        }
    }
}
