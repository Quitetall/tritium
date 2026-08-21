//! **The comparison that decides whether ternary is interesting**: bits-per-weight vs perplexity
//! against integer quantization, measured in the *same* harness so there are no eval confounds.
//!
//! Beating Tritium's own naive PTQ proves nothing. Beating **fp16** proves less than it sounds —
//! nobody ships fp16 weights. The baseline that matters is ordinary group-wise integer
//! quantization, because that is what a deployed model is actually stored as.
//!
//! - **RTN** (round-to-nearest, group-wise asymmetric int-`b`): the standard baseline GPTQ/AWQ
//!   improve upon. `bpw = b + 32/group` (f16 scale + f16 zero per group), with
//!   ragged final groups charged at their actual count.
//! - **SALT ladder** `T` planes: balanced-ternary geometric ladder, one f16 anchor per group plus a
//!   rotation bit, packed B3 (5 trits/byte) — `1.625` bpw per plane.
//! - **SALT ITF** `T` planes: the previous free-scale fitter, kept as the "old fitter" control.
//!
//! # Fairness: the fold goes on BOTH sides
//!
//! Every published ladder number is measured under the AWQ-style salience fold (α = 0.75). Scoring
//! a folded ladder against *unfolded* RTN would compare our best configuration against the
//! baseline's weakest, and the fold — not ternary — could be doing the work. So RTN is run **both
//! ways**: unfolded (plain RTN) and folded (activation-aware, i.e. AWQ-like).
//!
//! **The verdict scores against `min(folded, unfolded)` — the best RTN can do at each width — not
//! against folded by default.** Folded-vs-folded looks like the fair choice and at `g128` it is
//! conservative, because the fold HELPS RTN there (int4 23.225 → 21.980). But the fold is not a
//! universal gain: at `g256` it HURTS (int4 26.029 → 28.887, int5 16.737 → 17.236), since it widens
//! within-group dynamic range that RTN's min/max scaling already struggles to cover at that width.
//! Measuring against a handicapped baseline would manufacture a domination out of the baseline's
//! bad configuration. Taking the min can only weaken our own claims, which is the direction an
//! honest verdict errs in. Both rows stay in the table so the fold's contribution is visible on
//! each side rather than hidden inside one column.
//!
//! `#[ignore]`d; run:
//! ```text
//! TRITIUM_CORPUS=$HOME/.cache/tritium-corpora/wikitext2_400k_32k.json \
//! TRITIUM_MODEL_DIR=$HOME/.cache/tritium-models/smollm2-360m \
//!   cargo test -p tritium-nn --release --test baseline_rtn_vs_salt -- --ignored --nocapture
//! ```

mod common;

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

use common::{Calib, Evaluator, calibrate, extract, fold};
use tritium_format::salt_v2::SaltV2Codec;
use tritium_nn::ModelRunner;
use tritium_train::ops::ste::{self, RotationPolicy};

const EVAL_WINDOW: usize = 512;
/// Default group size for both RTN scales and SALT groups — 128 is the standard GPTQ/AWQ
/// reporting width. Override with `TRITIUM_GROUP`; unset reproduces every published number.
const GROUP_DEFAULT: usize = 128;
/// ITF alternations for the legacy free-scale control.
const ITERS: usize = 5;
/// Δ candidates per group for the ladder.
const GRID: usize = 16;
/// Default salience-fold strength — the value every published SALT number was measured under.
/// Override with `TRITIUM_FOLD_ALPHA`. The optimum is known to shift DOWN with model size
/// (0.75 -> 0.50 observed), so a fixed value is very likely wrong away from 135M.
const FOLD_ALPHA_DEFAULT: f64 = 0.75;

/// Group size for this run. Both the RTN scales and the SALT groups use it, so a sweep moves the
/// two baselines together and the comparison stays like-for-like.
fn group() -> usize {
    match std::env::var("TRITIUM_GROUP") {
        Ok(value) => {
            let parsed = value
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("TRITIUM_GROUP must be an integer >= 8, got {value:?}"));
            assert!(parsed >= 8, "TRITIUM_GROUP must be >= 8, got {parsed}");
            parsed
        }
        Err(std::env::VarError::NotPresent) => GROUP_DEFAULT,
        Err(error) => panic!("TRITIUM_GROUP is not valid UTF-8: {error}"),
    }
}

/// Rotation policy for the ternary arms.
///
/// **The sidecar SALT bundle carries NO rotation metadata**, and the ladder fits in the rotated
/// basis, so an artifact written with rotation applied would reconstruct `W·H` instead of `W`.
/// Anything the CLI can actually ship today must therefore be measured at `never`.
fn rotate_policy() -> RotationPolicy {
    match std::env::var("TRITIUM_ROTATE").as_deref() {
        Ok("never") => RotationPolicy::Never,
        Ok("auto") => RotationPolicy::Auto,
        Ok("always") => RotationPolicy::Always,
        Err(std::env::VarError::NotPresent) => RotationPolicy::Never,
        Ok(other) => panic!("TRITIUM_ROTATE must be always|auto|never, got {other:?}"),
        Err(error) => panic!("TRITIUM_ROTATE is not valid UTF-8: {error}"),
    }
}

/// Salience-fold alpha for this run.
fn fold_alpha() -> f64 {
    let alpha = match std::env::var("TRITIUM_FOLD_ALPHA") {
        Ok(value) => value.parse::<f64>().unwrap_or_else(|_| {
            panic!("TRITIUM_FOLD_ALPHA must be finite in [0, 1], got {value:?}")
        }),
        Err(std::env::VarError::NotPresent) => FOLD_ALPHA_DEFAULT,
        Err(error) => panic!("TRITIUM_FOLD_ALPHA is not valid UTF-8: {error}"),
    };
    assert!(
        alpha.is_finite() && (0.0..=1.0).contains(&alpha),
        "TRITIUM_FOLD_ALPHA must be finite in [0, 1], got {alpha}"
    );
    alpha
}
const CALIB_WINDOWS: usize = 8;
const CALIB_SEQ: usize = 512;

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    let dir = std::env::var("TRITIUM_MODEL_DIR")
        .unwrap_or_else(|_| format!("{home}/.cache/tritium-models/smollm2-135m"));
    PathBuf::from(dir)
}

fn corpus_path() -> PathBuf {
    std::env::var("TRITIUM_CORPUS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../tools/reference/heldout_corpus.json"
            ))
        })
}

fn hash_file(path: &Path) -> [u8; 32] {
    let mut file =
        File::open(path).unwrap_or_else(|error| panic!("open {}: {error}", path.display()));
    let mut hasher = blake3::Hasher::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .unwrap_or_else(|error| panic!("hash {}: {error}", path.display()));
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    *hasher.finalize().as_bytes()
}

fn hex_digest(digest: [u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn git_head() -> String {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("repo root");
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(repo)
        .output()
        .expect("git rev-parse");
    assert!(output.status.success(), "git rev-parse failed");
    String::from_utf8(output.stdout)
        .expect("git output")
        .trim()
        .to_owned()
}

fn model_manifest_digest(dir: &Path) -> String {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("read model directory {}: {error}", dir.display()))
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && (path.extension().is_some_and(|ext| ext == "safetensors")
                    || matches!(
                        path.file_name().and_then(|name| name.to_str()),
                        Some(
                            "config.json"
                                | "tokenizer.json"
                                | "tokenizer_config.json"
                                | "model.safetensors.index.json"
                        )
                    ))
        })
        .collect();
    paths.sort();
    assert!(
        !paths.is_empty(),
        "model directory has no identity-bearing files"
    );
    let mut hasher = blake3::Hasher::new();
    for path in paths {
        hasher.update(
            path.file_name()
                .expect("model file name")
                .to_string_lossy()
                .as_bytes(),
        );
        hasher.update(&hash_file(&path));
    }
    hex_digest(*hasher.finalize().as_bytes())
}

fn optional_file_digest(path: &Path) -> String {
    if path.is_file() {
        hex_digest(hash_file(path))
    } else {
        "absent".to_owned()
    }
}

/// `(train_ids, eval_ids)` — training ids feed calibration, eval ids score perplexity.
fn corpus() -> (Vec<u32>, Vec<u32>) {
    let path = corpus_path();
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

/// Group-wise **asymmetric** round-to-nearest integer quantization — the ordinary `int-b` weight
/// quantizer. Each contiguous run of `group()` weights within a row gets its own `(scale, zero)`, the
/// weight is mapped to `[0, 2^b - 1]`, and the returned buffer is the dequantized reconstruction.
fn rtn_quantize(w: &[f32], rows: usize, cols: usize, bits: u32) -> Vec<f32> {
    let levels = ((1u32 << bits) - 1) as f32;
    let mut out = vec![0.0f32; w.len()];
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let dst = &mut out[r * cols..(r + 1) * cols];
        for (g_src, g_dst) in row.chunks(group()).zip(dst.chunks_mut(group())) {
            let lo = g_src.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = g_src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let scale = (hi - lo) / levels;
            if scale <= 0.0 || scale.is_nan() {
                g_dst.fill(lo); // constant group: exactly representable
                continue;
            }
            // Asymmetric: the zero-point lands `lo` on integer 0.
            let zero = (-lo / scale).round();
            for (d, &s) in g_dst.iter_mut().zip(g_src) {
                let q = ((s / scale).round() + zero).clamp(0.0, levels);
                *d = (q - zero) * scale;
            }
        }
    }
    out
}

/// The best RTN configuration at each bit width — folded or unfolded, whichever scores lower.
///
/// The two share a bpw (the fold is a reparameterization, not a rate change), so "best" is simply
/// the lower perplexity. This is the baseline a real deployment would pick, and therefore the only
/// honest thing for a ternary arm to be measured against.
fn best_rtn(rows: &[Row]) -> Vec<Row> {
    let mut best: Vec<Row> = Vec::new();
    for r in rows.iter().filter(|r| !r.ternary) {
        // Rows at equal bpw are the same integer width; keep the better-scoring one.
        match best.iter_mut().find(|b| (b.bpw - r.bpw).abs() < 1e-9) {
            Some(b) if r.ppl < b.ppl => {
                b.name = r.name.clone();
                b.ppl = r.ppl;
            }
            Some(_) => {}
            None => best.push(Row {
                name: r.name.clone(),
                bpw: r.bpw,
                ppl: r.ppl,
                ternary: false,
            }),
        }
    }
    best
}

fn parameter_count(shapes: &[(usize, usize)]) -> usize {
    shapes.iter().map(|&(rows, cols)| rows * cols).sum()
}

fn group_count(shapes: &[(usize, usize)]) -> usize {
    shapes
        .iter()
        .map(|&(rows, cols)| rows * cols.div_ceil(group()))
        .sum()
}

/// Physical B3 payload bits for one ternary plane, including each ragged tail.
fn b3_payload_bits(shapes: &[(usize, usize)]) -> usize {
    shapes
        .iter()
        .map(|&(rows, cols)| {
            (0..rows)
                .map(|_| {
                    (0..cols)
                        .step_by(group())
                        .map(|start| {
                            SaltV2Codec::B3
                                .ledger((cols - start).min(group()))
                                .expect("B3 ledger")
                                .physical_bytes
                                * 8
                        })
                        .sum::<usize>()
                })
                .sum::<usize>()
        })
        .sum()
}

/// bpw for group-wise int-`b`: codes plus one f16 scale and one f16 zero per
/// actual group, including ragged final groups.
fn rtn_bpw(bits: u32, shapes: &[(usize, usize)]) -> f64 {
    f64::from(bits) + 32.0 * group_count(shapes) as f64 / parameter_count(shapes) as f64
}

/// bpw for `T` ladder planes: B3 payload, one f16 anchor per group, plus the
/// rotation bit. Every term uses physical tail-aware counts.
fn ladder_bpw(t: usize, shapes: &[(usize, usize)]) -> f64 {
    (t * b3_payload_bits(shapes) + 16 * group_count(shapes) + group_count(shapes)) as f64
        / parameter_count(shapes) as f64
}

/// bpw for `T` free-scale (ITF) planes: B3 payload plus one f16 scale per
/// plane per actual group.
fn itf_bpw(t: usize, shapes: &[(usize, usize)]) -> f64 {
    (t * (b3_payload_bits(shapes) + 16 * group_count(shapes))) as f64
        / parameter_count(shapes) as f64
}

/// One scored configuration.
struct Row {
    name: String,
    bpw: f64,
    ppl: f64,
    /// `true` for the ternary arms — the ones whose multiply-free execution is the real claim.
    ternary: bool,
}

#[test]
#[ignore = "slow PTQ baseline comparison; set TRITIUM_MODEL_DIR + TRITIUM_CORPUS; run explicitly"]
fn salt_vs_rtn_quality_per_byte() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() && !dir.join("model.safetensors.index.json").exists()
    {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, fp, shapes) = extract(&runner);
    let (train, eval) = corpus();

    let ev = Evaluator::from_env().expect("valid TRITIUM_EVAL_DEVICE");
    let g = group();
    let alpha = fold_alpha();
    let corpus = corpus_path();
    println!(
        "PROVENANCE git={} model_dir={} model_manifest_blake3={} tokenizer_blake3={} corpus={} corpus_blake3={} evaluator={} recipe=group:{g},fold_alpha:{alpha},window:{EVAL_WINDOW},calib_windows:{CALIB_WINDOWS},calib_seq:{CALIB_SEQ},grid:{GRID},iters:{ITERS},rotation:always",
        git_head(),
        dir.display(),
        model_manifest_digest(&dir),
        optional_file_digest(&dir.join("tokenizer.json")),
        corpus.display(),
        hex_digest(hash_file(&corpus)),
        ev.label(),
    );

    // Peak host memory is the binding constraint on a shared workstation, and it is knowable up
    // front. The arms below are ordered so at most TWO f32 copies of the model are live at once
    // (a source and the arm being scored) rather than three — see the `drop(fp)` below. Reported
    // so a run that would evict a co-tenant is visible before it starts rather than after the
    // kernel picks a victim.
    let params = parameter_count(&shapes);
    let copy_gib = params as f64 * 4.0 / (1024.0 * 1024.0 * 1024.0);
    eprintln!(
        "peak host weights ≈ {:.1} GiB ({params} params × f32 × 2 live copies)",
        copy_gib * 2.0
    );

    // The fp reference is scored on the ORIGINAL weights, before the fold reparameterizes them.
    let ppl_fp = ev.ppl(&fp, &arch, &eval, EVAL_WINDOW);

    println!(
        "{} | fp {ppl_fp:.3} ppl | {} held-out tokens | g{g} | B3 bpw | fold α={alpha} | eval={}\n\
         RTN is scored BOTH unfolded and folded; folded-vs-folded is the honest comparison, and \
         the unfolded row shows what the fold itself is worth.\n",
        dir.file_name().unwrap_or_default().to_string_lossy(),
        eval.len(),
        ev.label(),
    );
    println!(
        "{:<30} {:>7}  {:>11}  {:>9}",
        "method", "bpw", "ppl", "× fp"
    );
    println!("{}", "-".repeat(62));

    let mut rows: Vec<Row> = Vec::new();
    let mut score = |name: String, bpw: f64, ppl: f64, ternary: bool| {
        println!("{name:<30} {bpw:>7.3}  {ppl:>11.3}  {:>8.3}×", ppl / ppl_fp);
        rows.push(Row {
            name,
            bpw,
            ppl,
            ternary,
        });
    };

    // ── Baseline: ordinary integer quantization, unfolded then folded ────────────────────────────
    // int5/int6 are NOT optional padding: at g128 they land at 5.25 and 6.25 bpw, straddling the
    // ladder's T=3 (5.008) and T=4 (6.633). Without them the "best RTN at or below budget" rule
    // silently compares those arms against int4 while they spend 18%/56% more bytes, which flatters
    // ternary. int6 is the row that can dominate T=4 outright.
    for bits in [2u32, 3, 4, 5, 6, 8] {
        let q: Vec<Vec<f32>> = fp
            .iter()
            .zip(&shapes)
            .map(|(w, &(n, k))| rtn_quantize(w, n, k, bits))
            .collect();
        let ppl = ev.ppl(&q, &arch, &eval, EVAL_WINDOW);
        score(
            format!("RTN int{bits} g{g}"),
            rtn_bpw(bits, &shapes),
            ppl,
            false,
        );
    }

    // Calibration and the fold happen HERE, after the unfolded arms are done, so that `fp` can be
    // released the moment the folded copy exists. Folding up front (the obvious order) keeps `fp`,
    // `fp_folded` and the arm under test all live — three full f32 copies, 20.5 GiB at 1.7B — and
    // on a shared box that is the difference between finishing and being OOM-killed.
    let mut calib = Calib::new(&arch);
    for w in 0..CALIB_WINDOWS {
        calibrate(
            &fp,
            &arch,
            &train[w * CALIB_SEQ..(w + 1) * CALIB_SEQ],
            &mut calib,
        );
    }
    let (fp_folded, arch_folded) = fold(&fp, &shapes, &arch, &calib, alpha);
    drop(fp);
    drop(calib);

    for bits in [2u32, 3, 4, 5, 6, 8] {
        let q: Vec<Vec<f32>> = fp_folded
            .iter()
            .zip(&shapes)
            .map(|(w, &(n, k))| rtn_quantize(w, n, k, bits))
            .collect();
        let ppl = ev.ppl(&q, &arch_folded, &eval, EVAL_WINDOW);
        score(
            format!("RTN int{bits} g{g} +fold"),
            rtn_bpw(bits, &shapes),
            ppl,
            false,
        );
    }

    // ── SALT: the shipped ladder, plus the legacy free-scale fitter as a control ─────────────────
    for t in 1..=4 {
        let q: Vec<Vec<f32>> = fp_folded
            .iter()
            .zip(&shapes)
            .map(|(w, &(n, k))| {
                ste::salt_quantize_forward_grouped_geometric(
                    w,
                    n,
                    k,
                    t,
                    group(),
                    GRID,
                    rotate_policy(),
                )
            })
            .collect();
        let ppl = ev.ppl(&q, &arch_folded, &eval, EVAL_WINDOW);
        score(
            format!("SALT ladder T={t} +fold"),
            ladder_bpw(t, &shapes),
            ppl,
            true,
        );
    }
    for t in 1..=3 {
        let q: Vec<Vec<f32>> = fp_folded
            .iter()
            .zip(&shapes)
            .map(|(w, &(n, k))| {
                ste::salt_quantize_forward_grouped(w, n, k, t, group(), ITERS, RotationPolicy::Auto)
            })
            .collect();
        let ppl = ev.ppl(&q, &arch_folded, &eval, EVAL_WINDOW);
        score(
            format!("SALT ITF T={t} +fold"),
            itf_bpw(t, &shapes),
            ppl,
            true,
        );
    }

    // ── Verdict: at matched-or-fewer bits, does ternary beat integer? ────────────────────────────
    //
    // The rival is the BEST RTN configuration at each bit width — folded or unfolded, whichever
    // scores lower — not "folded" by default.
    //
    // Scoring against folded-only looked like the fair choice (our arms are folded, so folded-vs-
    // folded is like against like) and at g128 it is conservative: the fold HELPS RTN there
    // (int4 23.225 -> 21.980), so folded is already the baseline's better face. But the fold is not
    // a universal gain. At g256 it HURTS — int4 26.029 -> 28.887, int5 16.737 -> 17.236 — because
    // the fold widens within-group dynamic range that RTN's min/max scaling is already struggling
    // to cover at that width. Comparing against a handicapped baseline would manufacture a
    // domination out of the baseline's bad configuration, precisely where we most want the claim
    // to be trustworthy. Taking the min can only ever weaken our result, which is the direction an
    // honest verdict should err in.
    println!("\n--- quality-per-byte verdict (vs the BEST RTN config: min(folded, unfolded)) ---");
    let rivals: Vec<Row> = best_rtn(&rows);
    let mut ternary_wins = 0usize;
    let mut ternary_total = 0usize;
    for r in rows.iter().filter(|r| r.ternary) {
        ternary_total += 1;
        // The best RTN setting that costs no more than this ternary setting.
        let rival = rivals
            .iter()
            .filter(|x| x.bpw <= r.bpw + 1e-9)
            .min_by(|a, b| a.ppl.partial_cmp(&b.ppl).expect("finite ppl"));
        match rival {
            Some(x) if x.ppl < r.ppl => println!(
                "  {} ({:.3} bpw, {:.2} ppl) LOSES to {} ({:.3} bpw, {:.2} ppl)",
                r.name, r.bpw, r.ppl, x.name, x.bpw, x.ppl
            ),
            Some(x) => {
                ternary_wins += 1;
                println!(
                    "  {} ({:.3} bpw, {:.2} ppl) beats {} ({:.3} bpw, {:.2} ppl)",
                    r.name, r.bpw, r.ppl, x.name, x.bpw, x.ppl
                );
            }
            None => println!(
                "  {} ({:.3} bpw) — no RTN setting at or below this budget",
                r.name, r.bpw
            ),
        }
    }
    println!("\n  ternary arms winning their budget: {ternary_wins}/{ternary_total}");

    // The stricter test. "Beats the best RTN at or below my budget" lets a ternary arm spend more
    // bytes than its rival and still be called a win. Pareto DOMINATION is the claim that survives
    // scrutiny: no cheaper-or-equal point is also better-or-equal. Report both directions, because
    // an arm being dominated is the result that would sink the thesis.
    println!("\n--- strict Pareto check (dominate = no worse on BOTH bpw and ppl) ---");
    let folded_rtn: Vec<&Row> = rivals.iter().collect();
    let eps = 1e-9;
    for r in rows.iter().filter(|r| r.ternary) {
        let dominated_by: Vec<&str> = folded_rtn
            .iter()
            .filter(|x| x.bpw <= r.bpw + eps && x.ppl <= r.ppl + eps)
            .map(|x| x.name.as_str())
            .collect();
        let dominates: Vec<&str> = folded_rtn
            .iter()
            .filter(|x| r.bpw <= x.bpw + eps && r.ppl <= x.ppl + eps)
            .map(|x| x.name.as_str())
            .collect();
        if !dominated_by.is_empty() {
            println!("  {} is DOMINATED by {}", r.name, dominated_by.join(", "));
        } else if !dominates.is_empty() {
            println!("  {} DOMINATES {}", r.name, dominates.join(", "));
        } else {
            println!(
                "  {} — on the Pareto front (no domination either way)",
                r.name
            );
        }
    }

    println!(
        "\nNOTE: pure PTQ — no distillation. Uniform plane counts, not per-group allocation \
         (measured negative: lower weight SSE, worse ppl).\n\
         If ternary LOSES on quality-per-byte here, the honest claim is not \"ternary is smaller\" \
         but \"ternary matches integer quantization per byte AND executes multiply-free\" — a \
         speed/energy argument this harness does not measure."
    );
}
