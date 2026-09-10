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
//! fold has equalised the per-COLUMN shape -- but NOT the per-TENSOR one. `smooth_scales`
//! normalises by `gm`, the geometric mean of `rms` over the tensor's columns, so the weight is
//! really `rms_j^(2-2α)·gm^(2α)` and at `α = 1` it degenerates to `gm²`: flat inside a tensor,
//! tensors ranked by their typical activation magnitude. That is pure per-tensor sensitivity, not a
//! no-op. This comment asserted it WAS a no-op until 2026-09-10, when the α sweep measured the
//! curvature arm at 1.318× fp against the flat arm's 1.363× — different, and ordered the opposite
//! way from `α = 0.75`, where curvature is the worse arm (1.240× vs 1.221×).
//!
//! Getting this wrong (using `d_j` directly) would double-count the fold and allocate as if it had
//! never been applied.
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
/// Linear tensors per transformer block in `extract()`'s layout: q,k,v,o,gate,up,down.
/// Pinned by `calibrate::weight_names_match_extract_layout`; `layer` granularity mis-groups if the
/// two ever disagree, so the assert below fails loudly rather than aggregating the wrong tensors.
const SLOTS_PER_LAYER: usize = 7;
const GRID: usize = 16;
/// AWQ salience-fold strength, overridable with `TRITIUM_ALLOC_ALPHA`.
///
/// Sweepable because the fold and curvature allocation consume overlapping signal. The post-fold
/// column weight is `d_j/s_j^2 = rms_j^(2-2*alpha) * gm^(2*alpha)`, so alpha trades a per-COLUMN
/// shape against a per-TENSOR scale:
///
/// * `alpha = 0` -- full `rms^2` per-column signal, no per-tensor term. Allocation's best shot at
///   the fine-grained sensitivity the objective was written for.
/// * `alpha = 1` -- per-column shape gone, pure per-tensor `gm^2`. Measured 2026-09-10: 1.318x fp,
///   BETTER than the flat arm's 1.363x. Coarse sensitivity beat fine sensitivity.
/// * `alpha = 0.75` -- the shipping fold, and the only setting the 2026-08-03 demotion ever tested.
///   There curvature is the WORSE arm (1.240x vs flat 1.221x).
///
/// The sign flip between those settings is why this is a knob and not a constant: no single alpha
/// stands in for the family.
fn fold_alpha() -> f64 {
    std::env::var("TRITIUM_ALLOC_ALPHA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.75)
}
/// B3 over a 128-trit run: `ceil(128/5) = 26` bytes.
const B3_BITS_PER_TRIT: f64 = 1.625;

/// Floor on planes per group, overridable with `TRITIUM_ALLOC_TMIN` (default `1`).
///
/// The cliff hypothesis: a demoted group falling to `T = 1` drops to flat AbsMean, while a promoted
/// group gains at most 9.54 dB with sharp diminishing returns. The payoff is convex-bad downward and
/// concave-small upward, and a sum-of-terms objective cannot see that asymmetry. The original run
/// allowed `T in [1, 6]`; the 2026-08-03 histogram shows ~90k demotions against ~78k promotions --
/// 8% of groups moving cost 12% perplexity. Raising the floor tests whether the damage lives in the
/// demoted tail.
fn alloc_tmin() -> usize {
    env_usize("TRITIUM_ALLOC_TMIN", 1)
}

/// Allocation granularity, overridable with `TRITIUM_ALLOC_GRAN` = `group` | `tensor` | `layer`.
///
/// `group` is 1.1M independent decisions, each driven by one noisy scalar -- the classic setup for
/// over-fitting a proxy. Layer-wise bit allocation is the form that works in the wider literature,
/// so if allocation only fails at the finest granularity, the defect is decision count rather than
/// the sensitivity signal.
fn granularity() -> String {
    std::env::var("TRITIUM_ALLOC_GRAN").unwrap_or_else(|_| "group".to_owned())
}

/// Pearson correlation, used to ask whether the allocator demotes by *absolute* scale.
///
/// Uniform `T` is uniform RELATIVE precision (each group takes its own `max|w|` as the ladder
/// anchor), and RMSNorm is what makes relative precision the meaningful quantity. If SSE-optimal
/// allocation is really equalising ABSOLUTE error, demotions concentrate in low-norm groups and this
/// correlation is strongly positive. Recorded as the cheap test for that hypothesis.
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return f64::NAN;
    }
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for (&a, &b) in x.iter().zip(y) {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx) * (a - mx);
        syy += (b - my) * (b - my);
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return f64::NAN;
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

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
    // `d_j/s_j²` expands to `d_j^(1−α)·gm^(2α)`, so it carries a per-tensor scale (`gm`, the typical
    // activation magnitude into that tensor) as well as a per-column shape. That makes the
    // uncalibrated tensors dangerous: a hardcoded `1.0` is not "neutral", it is 1.0 in units the
    // calibrated tensors do not share, and whichever way it lands the allocator either starves those
    // tensors or lets them hoover up every plane. Filling them with the MEAN of the calibrated values
    // says "average sensitivity, we don't know better" in the right units.
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
    let mut out: Vec<Vec<f64>> = shapes.iter().map(|_| Vec::new()).collect();
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
    let (sum, cnt) = out
        .iter()
        .flatten()
        .fold((0.0f64, 0usize), |(s, n), &v| (s + v, n + 1));
    let mean = if cnt > 0 { sum / cnt as f64 } else { 1.0 };
    for (o, &(_, k)) in out.iter_mut().zip(shapes) {
        if o.is_empty() {
            *o = vec![mean; k]; // tied embed/head and o_proj: no calibration collected
        }
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
    // `t` indexes each group's own curve (`curves[g][t]`), not `curves` — clippy reads the double
    // subscript as a range loop over `curves` and its `enumerate()` suggestion inverts the meaning.
    // The outer loop must be over `t` because one quantize pass yields the whole model at that T.
    #[allow(clippy::needless_range_loop)]
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
fn allocation_vs_uniform_planes_at_matched_bits() {
    // Renamed from `curvature_allocation_beats_uniform_planes_at_matched_bits`: it does not beat
    // uniform, and a test name that asserts the outcome makes a negative result read as a failure.
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
    let alpha = fold_alpha();
    let curvature = column_curvature(&arch, &calib, &shapes, alpha);
    let (fp, arch) = fold(&fp, &shapes, &arch, &calib, alpha);

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
    let flat_curves: Vec<&Vec<f64>> = all_curves.iter().flatten().collect();
    let flat_sens: Vec<f64> = all_sens.iter().flatten().copied().collect();

    // Total weight-space error the arm's plane counts actually realise. This is the column that
    // makes the table interpretable rather than merely negative: the water-filling MINIMISES this
    // subject to the budget, so if an allocated arm posts lower error and worse perplexity, the
    // objective itself is wrong — that is the proxy gap, not an allocator bug.
    let total_sse = |counts: &[u8]| -> f64 {
        counts
            .iter()
            .zip(&flat_curves)
            .map(|(&t, c)| c[usize::from(t).min(c.len() - 1)])
            .sum()
    };

    // `Sigma_g H_g * err_g(T_g)` -- the curvature arm's OWN objective, which the raw-SSE column
    // above does not report. Without it the claim "the allocator achieved its own objective" is
    // only checkable for the flat arm, where `H_g = 1` makes weighted and raw identical. If the
    // curvature arm posts a HIGHER weighted error than uniform, the water-filling is not solving
    // the problem it was handed and the result is an allocator bug, not a proxy gap.
    let weighted_sse = |counts: &[u8]| -> f64 {
        counts
            .iter()
            .zip(&flat_curves)
            .zip(&flat_sens)
            .map(|((&t, c), h)| h * c[usize::from(t).min(c.len() - 1)])
            .sum()
    };

    println!(
        "SmolLM2-135M | fp {ppl_fp:.3} | fold α={alpha} | g{GROUP} | ladder (always rot)\n\
         budget = uniform T={t_ref}; allocator may spend T∈[1,{t_max}] per group\n\
         bpw includes the {plane_bits:.0}-bit per-group plane-count field the decoder needs.\n"
    );
    println!(
        "{:<34} {:>8} {:>9} {:>12} {:>13} {:>11} {:>9}",
        "arm", "mean T", "bpw", "recon SSE", "weighted SSE", "ppl", "× fp"
    );
    println!("{}", "-".repeat(102));

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
    let sse_u = total_sse(&counts_uniform);
    let wsse_u = weighted_sse(&counts_uniform);
    println!(
        "{:<34} {:>8.3} {:>9.3} {sse_u:>12.5e} {:>13.5e} {ppl_u:>11.3} {:>8.3}×",
        format!("uniform T={t_ref}"),
        t_ref as f64,
        alloc_bpw(&counts_uniform, &flat_sizes, plane_bits),
        wsse_u,
        ppl_u / ppl_fp,
    );

    // ── Allocation units ──────────────────────────────────────────────────────────────────────
    // Groups are the finest unit; `tensor` and `layer` aggregate them so the SAME budget is spent
    // with fewer, better-conditioned decisions. Aggregation is exact: a unit's error curve is the
    // elementwise SUM of its groups' curves (errors are additive), its weight count is the sum, and
    // its sensitivity is the weight-weighted mean of the groups' `max(0, H_g)` -- negatives are
    // zeroed BEFORE summing, matching what the per-group path already did at the GroupCurve site,
    // so one negative group cannot cancel its neighbours. Allocating over units and
    // broadcasting `T` back is therefore the same optimisation problem at a coarser resolution, not
    // a different one.
    let gran = granularity();
    let n_groups = flat_sizes.len();
    let unit_of: Vec<usize> = match gran.as_str() {
        "group" => (0..n_groups).collect(),
        "tensor" | "layer" => {
            let mut v = Vec::with_capacity(n_groups);
            for (ti, zs) in all_sizes.iter().enumerate() {
                // extract() lays tensors out as [token_embd] + per layer x 7 slots
                // (q,k,v,o,gate,up,down) -- pinned by calibrate::weight_names_match_extract_layout.
                let u = if gran == "tensor" {
                    ti
                } else if ti == 0 {
                    0
                } else {
                    1 + (ti - 1) / SLOTS_PER_LAYER
                };
                v.extend(std::iter::repeat_n(u, zs.len()));
            }
            v
        }
        other => panic!("TRITIUM_ALLOC_GRAN must be group|tensor|layer, got {other:?}"),
    };
    let n_units = unit_of.iter().max().map_or(0, |m| m + 1);
    let curve_len = t_max + 1;
    let mut unit_curve = vec![vec![0.0f64; curve_len]; n_units];
    let mut unit_weights = vec![0usize; n_units];
    let mut unit_sens_num = vec![0.0f64; n_units];
    for g in 0..n_groups {
        let u = unit_of[g];
        let c = flat_curves[g];
        // ladder_curves() allocates every curve at exactly `t_max + 1`, so this holds by
        // construction. Asserted rather than clamped: a shorter curve would make the aggregate a
        // sum of repeated tail values -- silently wrong rather than loudly wrong.
        debug_assert_eq!(c.len(), curve_len, "group {g} curve is not t_max+1 long");
        for t in 0..curve_len {
            unit_curve[u][t] += c[t];
        }
        unit_weights[u] += flat_sizes[g];
        unit_sens_num[u] += flat_sens[g].max(0.0) * flat_sizes[g] as f64;
    }
    // Group RMS, for the demotion-vs-scale correlation. `curve[0]` is the group's ||w||^2 by
    // construction, so no second pass over the weights is needed.
    let group_rms: Vec<f64> = (0..n_groups)
        .map(|g| (flat_curves[g][0] / flat_sizes[g] as f64).sqrt())
        .collect();
    let t_min = alloc_tmin();
    println!(
        "granularity = {gran} ({n_units} allocation units over {n_groups} groups), T_min = {t_min}\n"
    );

    // ── Arms 2 and 3: allocated, flat vs curvature-weighted ───────────────────────────────────
    for (label, use_curv) in [
        ("allocated, flat H=1", false),
        ("allocated, curvature H", true),
    ] {
        let curved: Vec<GroupCurve<'_>> = (0..n_units)
            .map(|u| GroupCurve {
                curve: &unit_curve[u],
                weights: unit_weights[u],
                sensitivity: if use_curv {
                    unit_sens_num[u] / unit_weights[u] as f64
                } else {
                    1.0
                },
            })
            .collect();
        let cfg = AllocConfig::from_bpw(
            tritium_quantize::TRIT_BITS * t_ref as f64,
            total_weights,
            t_min,
            t_max,
        );
        let alloc = allocate_with_curves(&curved, &cfg).expect("allocate");
        // Broadcast the unit decision back to every group it covers. At `group` granularity this is
        // the identity, so the default path is unchanged.
        // `t_max` is env-driven, so this cast is not obviously safe: TRITIUM_ALLOC_TMAX=300 would
        // wrap silently and quantize at a plane count nobody asked for.
        let counts: Vec<u8> = (0..n_groups)
            .map(|g| {
                u8::try_from(alloc.plane_counts[unit_of[g]])
                    .expect("plane count exceeds u8 — lower TRITIUM_ALLOC_TMAX")
            })
            .collect();

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
        let sse = total_sse(&counts);
        let wsse = weighted_sse(&counts);
        println!(
            "{label:<34} {mean_t:>8.3} {:>9.3} {sse:>12.5e} {:>13.5e} {ppl:>11.3} {:>8.3}×   (SSE {:+.1}%, wSSE {:+.1}% vs uniform)",
            alloc_bpw(&counts, &flat_sizes, plane_bits),
            wsse,
            ppl / ppl_fp,
            100.0 * (sse - sse_u) / sse_u,
            100.0 * (wsse - wsse_u) / wsse_u,
        );

        let mut hist = vec![0usize; t_max + 1];
        for &t in &counts {
            hist[usize::from(t)] += 1;
        }
        println!("     plane histogram (groups per T): {hist:?}");

        // How much movement, and in which direction. 8% of groups moving cost 12% perplexity on
        // 2026-08-03, so the interesting quantity is not the mean T (pinned by the budget) but the
        // size and asymmetry of the tail.
        let demoted = counts.iter().filter(|&&t| usize::from(t) < t_ref).count();
        let promoted = counts.iter().filter(|&&t| usize::from(t) > t_ref).count();
        let at_floor = counts.iter().filter(|&&t| usize::from(t) == t_min).count();
        let moved = demoted + promoted;
        println!(
            "     moved {moved} groups ({:.2}%): {demoted} demoted, {promoted} promoted; {at_floor} at the T_min={t_min} floor ({:.2}%)",
            100.0 * moved as f64 / n_groups as f64,
            100.0 * at_floor as f64 / n_groups as f64,
        );

        // Does the allocator demote by ABSOLUTE scale? Uniform T is uniform RELATIVE precision, and
        // if SSE-optimal allocation is equalising absolute error it must hand planes to large-norm
        // groups and starve small-norm ones -- a strongly positive correlation here. Near zero
        // refutes that hypothesis and points the failure elsewhere.
        let delta_t: Vec<f64> = counts
            .iter()
            .map(|&t| f64::from(t) - t_ref as f64)
            .collect();
        let r = pearson(&delta_t, &group_rms);
        // Undefined when either column is constant -- which happens for real: at `T_min = T_ref`
        // the budget forces every unit to the same T, so ΔT has zero variance. Say so instead of
        // printing NaN.
        if r.is_nan() {
            println!("     corr(ΔT, group RMS) = n/a   (ΔT is constant — no allocation freedom)");
        } else {
            println!("     corr(ΔT, group RMS) = {r:+.4}   (positive ⇒ demotes low-norm groups)");
        }
    }

    println!(
        "\nThe flat-H arm is the control. If it already captures the win, the gain is from ALLOCATION\n\
         (error curves differ across groups) and the calibration data is not doing any work; only the\n\
         gap between it and the curvature arm is attributable to H_g. Both are judged on held-out\n\
         perplexity — the point of curvature weighting is precisely that weight-space error is the\n\
         wrong objective, so it would be incoherent to score it on weight-space error.\n\n\
         Read the SSE column against the ppl column. The water-filling MINIMISES weighted SSE under\n\
         the budget, so an allocated arm that posts lower SSE and worse perplexity is not a broken\n\
         allocator — it is a demonstration that the objective is wrong."
    );
}
