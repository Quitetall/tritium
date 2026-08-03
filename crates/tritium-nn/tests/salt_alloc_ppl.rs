//! **Spend the bit budget where the loss is sensitive, not where the weights are big.**
//!
//! Every SALT number so far gives each group the same `T`. That is only right if every group matters
//! equally, which is exactly what the calibration data says is false. Rate-distortion allocation
//! (`tritium_quantize::allocate_with_curves`) instead hands each group a `T_g` by loss-drop-per-bit
//!
//! ```text
//! minimise  Σ_g H_g · err_g(T_g)   s.t.   Σ_g |g|·log2(3)·T_g ≤ budget
//! ```
//!
//! and the reason to reach for it here rather than a better fitter is that
//! `salt_encoding_headroom.rs` measured the additive-ternary family at ~3× the SSE of a free scalar
//! quantizer at T=3, with the free-scale fitter closing almost none of it — that gap is the price of
//! multiply-free decode and no scale rule escapes it. Allocation attacks a different axis: not where
//! levels sit inside a group, but which groups get levels at all. It is also the only lever measured
//! so far that targets the **proxy gap** directly, since `H_g` is loss curvature rather than weight
//! magnitude.
//!
//! **Three arms at a matched budget**, so the only difference is how the same bits are spent:
//!
//! | arm | `T_g` | `H_g` |
//! |---|---|---|
//! | uniform | fixed `T_REF` everywhere | — |
//! | allocated, flat | water-filled | `1` for every group |
//! | allocated, curvature | water-filled | post-fold input curvature |
//!
//! The flat arm is the control that matters: it isolates *allocation* (some groups have steeper
//! error curves than others) from *curvature* (some groups matter more to the loss). Without it a
//! win cannot be attributed.
//!
//! **The curvature term, and why it is not just `E[x²]`.** The objective for weight-only
//! quantization is `‖(W−Ŵ)X‖²`, so with a diagonal approximation column `j` is weighted by
//! `d_j = E[x_j²]`. But the salience fold has already rescaled column `j` by `s_j`, and an error
//! `e'_j` in the folded basis is `e'_j / s_j` in the original one — so the correct post-fold weight
//! is `d_j / s_j²`. With `s_j ∝ rms_j^α` and `d_j ∝ rms_j²` that is `rms_j^(2−2α)`: at `α = 1` the
//! fold has equalised everything and curvature allocation is a no-op, and at our `α = 0.75` a real
//! `rms_j^0.5` residual remains. Getting this wrong (using `d_j` directly) would double-count the
//! fold and allocate as if it had never been applied.
//!
//! Group sensitivity is the **mean** of that over the group's columns. Per-column weighting inside a
//! group is not available: the Hadamard rotation mixes columns, so a diagonal weight in the original
//! basis becomes dense in the rotated one. The group mean survives rotation, which is why allocation
//! is a per-group lever and `Δ` selection stays plain SSE (a per-group constant cannot change
//! `argmin_Δ`).
//!
//! Run:
//! ```text
//! TRITIUM_ALLOC_T=3 TRITIUM_ALLOC_TMAX=6 \
//!   cargo test -p tritium-nn --release --test salt_alloc_ppl -- --ignored --nocapture
//! ```

mod common;

use std::path::PathBuf;

use common::{Arch, Calib, calibrate, extract, fold, perplexity_windowed, smooth_scales};
use tritium_nn::ModelRunner;
use tritium_quantize::{AllocConfig, GroupCurve, allocate_with_curves};
use tritium_train::ops::ste::{self, RotationPolicy};

const EVAL_WINDOW: usize = 512;
const CALIB_WINDOWS: usize = 8;
const CALIB_SEQ: usize = 512;
const GROUP: usize = 128;
const GRID: usize = 16;
const FOLD_ALPHA: f64 = 0.75;
/// B3 over a 128-trit run: `ceil(128/5) = 26` bytes.
const B3_BITS_PER_TRIT: f64 = 1.625;

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    let dir = std::env::var("TRITIUM_MODEL_DIR")
        .unwrap_or_else(|_| format!("{home}/.cache/tritium-models/smollm2-135m"));
    PathBuf::from(dir)
}

fn corpus() -> (Vec<u32>, Vec<u32>) {
    let path = std::env::var("TRITIUM_CORPUS").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tools/reference/heldout_corpus.json"
        )
        .to_string()
    });
    let j: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).expect("corpus json")).expect("parse");
    let ids = |k: &str| -> Vec<u32> {
        j[k].as_array()
            .expect(k)
            .iter()
            .map(|v| v.as_u64().expect("id") as u32)
            .collect()
    };
    (ids("train_ids"), ids("eval_ids"))
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Post-fold effective input curvature per column, per tensor: `d_j / s_j²`.
///
/// Tensor layout matches `common::extract` / `common::fold`: index 0 is the tied embed/head, then
/// `1 + 7·layer` gives `[q, k, v, o, gate, up, down]`. Only q/k/v, gate/up and down have calibrated
/// inputs; the tied head and `o_proj` get a flat `1.0` because no calibration is collected for them
/// (o_proj's input is the attention output, and the head's is the final hidden state). Flat is the
/// honest default — it says "we do not know", not "this does not matter".
fn column_curvature(a: &Arch, c: &Calib, shapes: &[(usize, usize)], alpha: f64) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = shapes.iter().map(|&(_, k)| vec![1.0f64; k]).collect();
    let eff = |acc: &[f64], rows: usize| -> Vec<f64> {
        let s = smooth_scales(acc, rows, alpha);
        acc.iter()
            .zip(&s)
            .map(|(&d, &sj)| {
                let sj = f64::from(sj).max(1e-12);
                (d / rows as f64) / (sj * sj)
            })
            .collect()
    };
    for li in 0..a.n_layers {
        let base = 1 + 7 * li;
        let attn = eff(&c.attn_in[li], c.rows);
        let ffn = eff(&c.ffn_in[li], c.rows);
        let down = eff(&c.down_in[li], c.rows);
        for k in 0..3 {
            out[base + k].clone_from(&attn);
        }
        out[base + 4].clone_from(&ffn);
        out[base + 5].clone_from(&ffn);
        out[base + 6].clone_from(&down);
    }
    out
}

/// Per-group SSE curve `err(0..=t_max)` for the ladder, plus the group's mean sensitivity.
///
/// `err(0)` is `‖w‖²` — the no-planes error the allocator needs as its starting point.
fn ladder_curves(
    w: &[f32],
    rows: usize,
    cols: usize,
    curvature: &[f64],
    t_max: usize,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<usize>) {
    let per_row = cols.div_ceil(GROUP);
    let n_groups = rows * per_row;
    let mut curves = vec![vec![0.0f64; t_max + 1]; n_groups];
    let mut sens = vec![0.0f64; n_groups];
    let mut sizes = vec![0usize; n_groups];

    for r in 0..rows {
        for b in 0..per_row {
            let lo = r * cols + b * GROUP;
            let hi = (lo + GROUP).min((r + 1) * cols);
            let g = r * per_row + b;
            sizes[g] = hi - lo;
            curves[g][0] = w[lo..hi].iter().map(|&v| f64::from(v) * f64::from(v)).sum();
            let c0 = b * GROUP;
            let c1 = (c0 + GROUP).min(cols);
            sens[g] = curvature[c0..c1].iter().sum::<f64>() / (c1 - c0) as f64;
        }
    }
    for t in 1..=t_max {
        let q = ste::salt_quantize_forward_grouped_geometric(
            w,
            rows,
            cols,
            t,
            GROUP,
            GRID,
            RotationPolicy::Always,
        );
        for r in 0..rows {
            for b in 0..per_row {
                let lo = r * cols + b * GROUP;
                let hi = (lo + GROUP).min((r + 1) * cols);
                curves[r * per_row + b][t] = q[lo..hi]
                    .iter()
                    .zip(&w[lo..hi])
                    .map(|(&a, &x)| f64::from(a - x) * f64::from(a - x))
                    .sum();
            }
        }
    }
    (curves, sens, sizes)
}

/// Real stored bpw for an allocation: trits + one f16 anchor per group + the rotation bit + the
/// plane-count field the decoder needs to know how many planes to read.
fn alloc_bpw(counts: &[u8], sizes: &[usize], plane_bits: f64) -> f64 {
    let total: usize = sizes.iter().sum();
    let trits: f64 = counts
        .iter()
        .zip(sizes)
        .map(|(&t, &s)| f64::from(t) * s as f64 * B3_BITS_PER_TRIT)
        .sum();
    let per_group_side = (16.0 + 1.0 + plane_bits) * sizes.len() as f64;
    (trits + per_group_side) / total as f64
}

#[test]
#[ignore = "PTQ sweep over every tensor; needs SmolLM2-135M; run explicitly"]
fn curvature_allocation_beats_uniform_planes_at_matched_bits() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let t_ref = env_usize("TRITIUM_ALLOC_T", 3);
    let t_max = env_usize("TRITIUM_ALLOC_TMAX", 6).max(t_ref);
    // Plane count per group must be transmitted; ceil(log2(t_max+1)) bits.
    let plane_bits = ((t_max + 1) as f64).log2().ceil();

    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, fp, shapes) = extract(&runner);
    let (train, eval) = corpus();
    let ppl_fp = perplexity_windowed(&fp, &arch, &eval, EVAL_WINDOW);

    let mut calib = Calib::new(&arch);
    for w in 0..CALIB_WINDOWS {
        calibrate(
            &fp,
            &arch,
            &train[w * CALIB_SEQ..(w + 1) * CALIB_SEQ],
            &mut calib,
        );
    }
    let curvature = column_curvature(&arch, &calib, &shapes, FOLD_ALPHA);
    let (fp, arch) = fold(&fp, &shapes, &arch, &calib, FOLD_ALPHA);

    // Curves once; both allocated arms reuse them (they differ only in `H_g`).
    let mut all_curves: Vec<Vec<Vec<f64>>> = Vec::with_capacity(fp.len());
    let mut all_sens: Vec<Vec<f64>> = Vec::with_capacity(fp.len());
    let mut all_sizes: Vec<Vec<usize>> = Vec::with_capacity(fp.len());
    for ((w, &(rows, cols)), curv) in fp.iter().zip(&shapes).zip(&curvature) {
        let (c, s, z) = ladder_curves(w, rows, cols, curv, t_max);
        all_curves.push(c);
        all_sens.push(s);
        all_sizes.push(z);
    }
    let flat_sizes: Vec<usize> = all_sizes.iter().flatten().copied().collect();
    let total_weights: usize = flat_sizes.iter().sum();

    println!(
        "SmolLM2-135M | fp {ppl_fp:.3} | fold α={FOLD_ALPHA} | g{GROUP} | ladder (always rot)\n\
         budget = uniform T={t_ref}; allocator may spend T∈[1,{t_max}] per group\n\
         bpw includes the {plane_bits:.0}-bit per-group plane-count field the decoder needs.\n"
    );
    println!(
        "{:<34} {:>8} {:>9} {:>11} {:>9}",
        "arm", "mean T", "bpw", "ppl", "× fp"
    );
    println!("{}", "-".repeat(76));

    // ── Arm 1: uniform ────────────────────────────────────────────────────────────────────────
    let uniform: Vec<Vec<f32>> = fp
        .iter()
        .zip(&shapes)
        .map(|(w, &(n, k))| {
            ste::salt_quantize_forward_grouped_geometric(
                w,
                n,
                k,
                t_ref,
                GROUP,
                GRID,
                RotationPolicy::Always,
            )
        })
        .collect();
    let counts_uniform: Vec<u8> = vec![t_ref as u8; flat_sizes.len()];
    let ppl_u = perplexity_windowed(&uniform, &arch, &eval, EVAL_WINDOW);
    println!(
        "{:<34} {:>8.3} {:>9.3} {ppl_u:>11.3} {:>8.3}×",
        format!("uniform T={t_ref}"),
        t_ref as f64,
        alloc_bpw(&counts_uniform, &flat_sizes, plane_bits),
        ppl_u / ppl_fp,
    );

    // ── Arms 2 and 3: allocated, flat vs curvature-weighted ───────────────────────────────────
    for (label, use_curv) in [
        ("allocated, flat H=1", false),
        ("allocated, curvature H", true),
    ] {
        let curved: Vec<GroupCurve<'_>> = all_curves
            .iter()
            .zip(&all_sens)
            .zip(&all_sizes)
            .flat_map(|((cs, ss), zs)| {
                cs.iter()
                    .zip(ss)
                    .zip(zs)
                    .map(move |((c, &s), &z)| GroupCurve {
                        curve: c,
                        weights: z,
                        sensitivity: if use_curv { s.max(0.0) } else { 1.0 },
                    })
            })
            .collect();
        let cfg = AllocConfig::from_bpw(
            tritium_quantize::TRIT_BITS * t_ref as f64,
            total_weights,
            1,
            t_max,
        );
        let alloc = allocate_with_curves(&curved, &cfg).expect("allocate");
        let counts: Vec<u8> = alloc.plane_counts.iter().map(|&t| t as u8).collect();

        // Slice the flat allocation back per tensor, in the same order it was flattened.
        let mut cursor = 0usize;
        let q: Vec<Vec<f32>> = fp
            .iter()
            .zip(&shapes)
            .zip(&all_sizes)
            .map(|((w, &(n, k)), zs)| {
                let seg = &counts[cursor..cursor + zs.len()];
                cursor += zs.len();
                ste::salt_quantize_forward_grouped_geometric_alloc(
                    w,
                    n,
                    k,
                    seg,
                    GROUP,
                    GRID,
                    RotationPolicy::Always,
                )
            })
            .collect();
        assert_eq!(
            cursor,
            counts.len(),
            "allocation slicing must consume exactly"
        );

        let mean_t = counts
            .iter()
            .zip(&flat_sizes)
            .map(|(&t, &s)| f64::from(t) * s as f64)
            .sum::<f64>()
            / total_weights as f64;
        let ppl = perplexity_windowed(&q, &arch, &eval, EVAL_WINDOW);
        println!(
            "{label:<34} {mean_t:>8.3} {:>9.3} {ppl:>11.3} {:>8.3}×",
            alloc_bpw(&counts, &flat_sizes, plane_bits),
            ppl / ppl_fp,
        );

        let mut hist = vec![0usize; t_max + 1];
        for &t in &counts {
            hist[usize::from(t)] += 1;
        }
        println!("     plane histogram (groups per T): {hist:?}");
    }

    println!(
        "\nThe flat-H arm is the control. If it already captures the win, the gain is from ALLOCATION\n\
         (error curves differ across groups) and the calibration data is not doing any work; only the\n\
         gap between it and the curvature arm is attributable to H_g. Both are judged on held-out\n\
         perplexity — the point of curvature weighting is precisely that weight-space error is the\n\
         wrong objective, so it would be incoherent to score it on weight-space error."
    );
}
