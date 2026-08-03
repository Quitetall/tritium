//! **Does perplexity still respond to reconstruction, and does the balanced-ternary ladder help?**
//!
//! Two questions, and the first one can close the second.
//!
//! **Q1 (the kill-first question).** SALT's best PTQ configuration sits at 1.079× fp. If perplexity is
//! no longer reconstruction-limited at that point, then no fitter improvement — this one or any
//! other — reaches fp parity, and the only remaining lever is distillation. That is answerable in one
//! run: drive `T` to 8 with a fitter whose reconstruction error genuinely keeps falling, and watch
//! whether ppl follows. If ppl goes flat while MSE drops 30 dB, the "more planes" thesis is dead.
//!
//! **Q2 (the ladder).** SALT's greedy expansion picks `s_p` from the residual plane `p-1` left. The
//! resulting ratios `s_{p+1}/s_p` measure ≈0.41, 0.42, 0.58, 0.70 and eventually exceed 1.0, so each
//! added plane buys ~1 dB against the 9.54 dB the rate allows. Pinning the ratio to 1/3 makes the
//! reachable levels a uniform grid over every integer in `±(3^T−1)/2` — see
//! `tritium_train::ops::ste::salt_quantize_forward_grouped_geometric` and
//! `crates/tritium-train/tests/geometric_ladder.rs`, where the synthetic gain measures 3.2×/16×/737×
//! at T=3/4/6.
//!
//! **Weight-space MSE is not the verdict.** This repo has a documented proxy gap (per-layer MSE can
//! anti-correlate with perplexity; ITF at T=1 without rotation made things worse by minimising SSE),
//! so both quantities are printed side by side and only the ppl column decides anything.
//!
//! Arms (select with `TRITIUM_LADDER_ARMS`, default `AC`):
//!
//! | arm | fitter | rotation | purpose |
//! |---|---|---|---|
//! | `A` | greedy + ITF | Auto | control — what the trainer runs today |
//! | `C` | geometric ladder | Always | the proposal |
//! | `D` | geometric ladder | Auto | does per-group rotation choice help a fixed ladder? |
//! | `E` | geometric ladder | Never | is rotation a precondition for the ladder? |
//!
//! Run:
//! ```text
//! TRITIUM_LADDER_ARMS=AC TRITIUM_LADDER_T=2,3,4,6,8 \
//!   cargo test -p tritium-nn --release --test salt_ladder_ppl -- --ignored --nocapture
//! ```

mod common;

use std::path::PathBuf;

use common::{Calib, calibrate, extract, fold, perplexity_windowed};
use tritium_format::salt_v2::SaltV2Codec;
use tritium_nn::ModelRunner;
use tritium_train::ops::ste::{self, RotationPolicy};

const EVAL_WINDOW: usize = 512;
const CALIB_WINDOWS: usize = 8;
const CALIB_SEQ: usize = 512;
const GROUP: usize = 128;
const ITERS: usize = 5;
/// Δ candidates per group for the ladder. 16 spans a 16× range; the search matters most on
/// heavy-tailed groups, where a clipping-free step wastes resolution on outliers.
const GRID: usize = 16;
const FOLD_ALPHA: f64 = 0.75;

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

/// Total squared reconstruction error over every quantized tensor — the proxy, reported so the ppl
/// column can be read against it. A large MSE drop with a flat ppl column IS the kill signal.
fn total_sse(q: &[Vec<f32>], fp: &[Vec<f32>]) -> f64 {
    q.iter()
        .zip(fp)
        .map(|(a, b)| {
            a.iter()
                .zip(b)
                .map(|(&x, &y)| f64::from(x - y) * f64::from(x - y))
                .sum::<f64>()
        })
        .sum()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    GreedyAuto,
    LadderAlways,
    LadderAuto,
    LadderNever,
}

impl Arm {
    fn from_char(c: char) -> Option<Self> {
        match c {
            'A' => Some(Self::GreedyAuto),
            'C' => Some(Self::LadderAlways),
            'D' => Some(Self::LadderAuto),
            'E' => Some(Self::LadderNever),
            _ => None,
        }
    }

    fn label(self, t: usize) -> String {
        match self {
            Self::GreedyAuto => format!("T={t} greedy+ITF (auto rot)"),
            Self::LadderAlways => format!("T={t} ladder 1/3 (always rot)"),
            Self::LadderAuto => format!("T={t} ladder 1/3 (auto rot)"),
            Self::LadderNever => format!("T={t} ladder 1/3 (no rot)"),
        }
    }

    fn quantize(self, w: &[f32], n: usize, k: usize, t: usize) -> Vec<f32> {
        match self {
            Self::GreedyAuto => {
                ste::salt_quantize_forward_grouped(w, n, k, t, GROUP, ITERS, RotationPolicy::Auto)
            }
            Self::LadderAlways => ste::salt_quantize_forward_grouped_geometric(
                w,
                n,
                k,
                t,
                GROUP,
                GRID,
                RotationPolicy::Always,
            ),
            Self::LadderAuto => ste::salt_quantize_forward_grouped_geometric(
                w,
                n,
                k,
                t,
                GROUP,
                GRID,
                RotationPolicy::Auto,
            ),
            Self::LadderNever => ste::salt_quantize_forward_grouped_geometric(
                w,
                n,
                k,
                t,
                GROUP,
                GRID,
                RotationPolicy::Never,
            ),
        }
    }

    /// Free-scale arms pay one f16 scale per plane per group plus a rotation bit; the ladder pays a
    /// single anchor per group, and arm E pays no rotation bit because it never rotates.
    fn bpw(self, t: usize) -> f64 {
        let rot_bit = 1.0 / GROUP as f64;
        match self {
            Self::GreedyAuto => {
                ste::ternary_bits_per_weight_codec(t, GROUP, SaltV2Codec::B3, GROUP) + rot_bit
            }
            Self::LadderNever => {
                ste::ternary_bits_per_weight_geometric(t, GROUP, SaltV2Codec::B3, GROUP)
            }
            _ => ste::ternary_bits_per_weight_geometric(t, GROUP, SaltV2Codec::B3, GROUP) + rot_bit,
        }
    }
}

fn arms() -> Vec<Arm> {
    std::env::var("TRITIUM_LADDER_ARMS")
        .unwrap_or_else(|_| "AC".to_string())
        .chars()
        .filter_map(Arm::from_char)
        .collect()
}

fn plane_counts() -> Vec<usize> {
    std::env::var("TRITIUM_LADDER_T")
        .unwrap_or_else(|_| "2,3,4,6,8".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .filter(|t| (1..=8).contains(t))
        .collect()
}

#[test]
#[ignore = "slow PTQ sweep over every tensor; needs SmolLM2-135M; run explicitly"]
fn ladder_plane_scaling_vs_greedy_residual() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, fp, shapes) = extract(&runner);
    let (train, eval) = corpus();
    let ppl_fp = perplexity_windowed(&fp, &arch, &eval, EVAL_WINDOW);

    // The salience fold is the strongest known configuration and the basis every published SALT
    // number was measured under, so every arm gets it — otherwise a win here could just be the fold.
    let mut calib = Calib::new(&arch);
    for w in 0..CALIB_WINDOWS {
        calibrate(
            &fp,
            &arch,
            &train[w * CALIB_SEQ..(w + 1) * CALIB_SEQ],
            &mut calib,
        );
    }
    let (fp, arch) = fold(&fp, &shapes, &arch, &calib, FOLD_ALPHA);

    let arms = arms();
    let ts = plane_counts();
    println!(
        "SmolLM2-135M | fp {ppl_fp:.3} ppl | salience fold α={FOLD_ALPHA} | g{GROUP} | B3 bpw | \
         eval window {EVAL_WINDOW}\n\
         MSE is the PROXY and decides nothing — a large MSE drop with a flat ppl column means \
         perplexity is no longer reconstruction-limited.\n"
    );
    println!(
        "{:<32} {:>7} {:>11} {:>9} {:>12} {:>9}",
        "configuration", "bpw", "ppl", "× fp", "recon SSE", "dB"
    );
    println!("{}", "-".repeat(86));

    for &arm in &arms {
        let mut prev_sse: Option<f64> = None;
        for &t in &ts {
            let q: Vec<Vec<f32>> = fp
                .iter()
                .zip(&shapes)
                .map(|(w, &(n, k))| arm.quantize(w, n, k, t))
                .collect();
            let ppl = perplexity_windowed(&q, &arch, &eval, EVAL_WINDOW);
            let sse = total_sse(&q, &fp);
            // dB improvement over this arm's previous plane count — the 9.54 dB/plane target.
            let db = prev_sse.map_or(f64::NAN, |p| 10.0 * (p / sse).log10());
            println!(
                "{:<32} {:>7.3} {ppl:>11.3} {:>8.3}× {sse:>12.4e} {db:>9.2}",
                arm.label(t),
                arm.bpw(t),
                ppl / ppl_fp,
            );
            prev_sse = Some(sse);
        }
        println!();
    }

    println!(
        "Reading it: if the ppl column for the ladder arm keeps falling as T grows, reconstruction \
         still binds and the fitter is worth improving. If ppl flattens while the dB column keeps \
         reporting real gains, more planes cannot reach fp parity by any fitter and distillation is \
         the only remaining lever."
    );
}
