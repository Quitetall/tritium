//! **Is a high-bitrate ternary model a different teacher from the fp model it was quantized from?**
//!
//! The proposal: build a high-quality ternary reference (the ladder at large `T`, measured at
//! 1.000× fp) and distil the low-bitrate student against *that* instead of — or alongside — the fp16
//! teacher. The appeal is real: a ternary teacher only ever demands things a ternary student can
//! represent, whereas an fp teacher can be confident about distinctions no ternary lattice can make,
//! and the student then chases an unreachable target.
//!
//! But knowledge distillation transports the teacher's **output distribution**, not its weights. So
//! the proposal only has content if `p_ternary(· | x)` actually differs from `p_fp(· | x)`. If the
//! high-`T` reference matches fp to within noise — which is what "1.000× fp" suggests — then its
//! soft targets *are* the fp soft targets, and distilling from it is distilling from fp with extra
//! steps.
//!
//! That is a measurement, not an argument. This test reports, per plane count:
//!
//! - **KL(fp ‖ ternary)** in nats per token — the exact quantity a KD loss transports. A KD gradient
//!   is `p_student − p_teacher`, so two teachers whose distributions differ by ~0 give the same
//!   gradient no matter how their weights differ.
//! - **top-1 agreement** with fp — the coarse version, for intuition.
//! - **KL(ternary_T ‖ ternary_1)** — how far a *low*-bitrate student starts from each candidate
//!   teacher. This is the teaching-assistant question (Mirzadeh et al.): a teacher closer in
//!   capacity to the student can distil better than a stronger one, so the useful teacher may be a
//!   mid-`T` ternary model rather than either extreme.
//!
//! What this test cannot answer: whether a smaller KL actually trains better. Capacity-gap effects
//! are about optimisation dynamics, not about distance at initialisation. It bounds the question —
//! a teacher whose KL from fp is ~0 cannot possibly behave differently — and rules out the arm that
//! is provably redundant before anyone spends a training run on it.
//!
//! Run:
//! ```text
//! cargo test -p tritium-nn --release --test salt_teacher_signal -- --ignored --nocapture
//! ```

mod common;

use std::path::PathBuf;

use common::{Arch, Calib, calibrate, extract, fold, logits_of};
use tritium_nn::ModelRunner;
use tritium_train::ops::ste::{self, RotationPolicy};

const EVAL_WINDOW: usize = 512;
const CALIB_WINDOWS: usize = 8;
const CALIB_SEQ: usize = 512;
const GROUP: usize = 128;
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

/// Log-softmax of one logit row, in f64, max-subtracted.
fn log_softmax(row: &[f32]) -> Vec<f64> {
    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = m + row
        .iter()
        .map(|&x| (f64::from(x) - m).exp())
        .sum::<f64>()
        .ln();
    row.iter().map(|&x| f64::from(x) - lse).collect()
}

/// Mean `KL(a ‖ b)` in nats per position, plus top-1 agreement, over `[seq, vocab]` logits.
fn kl_and_agreement(a: &[f32], b: &[f32], vocab: usize) -> (f64, f64) {
    let seq = a.len() / vocab;
    let mut kl = 0.0f64;
    let mut agree = 0usize;
    for t in 0..seq {
        let ra = &a[t * vocab..(t + 1) * vocab];
        let rb = &b[t * vocab..(t + 1) * vocab];
        let la = log_softmax(ra);
        let lb = log_softmax(rb);
        for i in 0..vocab {
            let pa = la[i].exp();
            if pa > 0.0 {
                kl += pa * (la[i] - lb[i]);
            }
        }
        let am = (0..vocab).max_by(|&x, &y| ra[x].total_cmp(&ra[y])).unwrap();
        let bm = (0..vocab).max_by(|&x, &y| rb[x].total_cmp(&rb[y])).unwrap();
        if am == bm {
            agree += 1;
        }
    }
    (kl / seq as f64, agree as f64 / seq as f64)
}

/// Quantize every tensor with the ladder at `t` planes.
fn ladder(fp: &[Vec<f32>], shapes: &[(usize, usize)], t: usize) -> Vec<Vec<f32>> {
    fp.iter()
        .zip(shapes)
        .map(|(w, &(n, k))| {
            ste::salt_quantize_forward_grouped_geometric(
                w,
                n,
                k,
                t,
                GROUP,
                GRID,
                RotationPolicy::Always,
            )
        })
        .collect()
}

/// Concatenated `[seq, vocab]` logits over every scored eval window.
fn logits_over(weights: &[Vec<f32>], arch: &Arch, eval: &[u32]) -> Vec<f32> {
    let mut out = Vec::new();
    for chunk in eval.chunks(EVAL_WINDOW) {
        if chunk.len() < 2 {
            continue;
        }
        out.extend(logits_of(weights, arch, chunk));
    }
    out
}

#[test]
#[ignore = "needs SmolLM2-135M; several forward passes over the eval set; run explicitly"]
fn high_bitrate_ternary_teacher_vs_fp_teacher() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, fp, shapes) = extract(&runner);
    let (train, eval) = corpus();

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

    let vocab = arch.vocab;
    let fp_logits = logits_over(&fp, &arch, &eval);
    let positions = fp_logits.len() / vocab;

    // T=1 is the bitrate the student actually has to reach, so it doubles as the reference point
    // for "how far does the student start from each candidate teacher".
    let student = ladder(&fp, &shapes, 1);
    let student_logits = logits_over(&student, &arch, &eval);
    let (kl_fp_student, _) = kl_and_agreement(&fp_logits, &student_logits, vocab);

    println!(
        "SmolLM2-135M | fold α={FOLD_ALPHA} | g{GROUP} | ladder (always rot) | {positions} scored \
         positions\n\
         KL is nats/token, the exact quantity a KD loss transports. A KD gradient is\n\
         (p_student − p_teacher), so two teachers at KL ≈ 0 from each other give the SAME gradient\n\
         however different their weights are.\n"
    );
    println!(
        "{:<10} {:>12} {:>14} {:>18}",
        "teacher", "KL(fp‖T)", "top-1 vs fp", "KL(T‖T=1 student)"
    );
    println!("{}", "-".repeat(58));
    println!(
        "{:<10} {:>12.3e} {:>13.1}% {:>18.4}",
        "fp16", 0.0, 100.0, kl_fp_student
    );

    for t in [2usize, 3, 4, 6, 8] {
        let q = ladder(&fp, &shapes, t);
        let ql = logits_over(&q, &arch, &eval);
        let (kl_fp, agree) = kl_and_agreement(&fp_logits, &ql, vocab);
        let (kl_to_student, _) = kl_and_agreement(&ql, &student_logits, vocab);
        println!(
            "{:<10} {:>12.3e} {:>13.1}% {:>18.4}",
            format!("ternary T={t}"),
            kl_fp,
            agree * 100.0,
            kl_to_student,
        );
    }

    println!(
        "\nReading it. If KL(fp‖T=8) is orders of magnitude below KL(fp‖T=1), the high-bitrate\n\
         ternary reference carries the same soft targets as fp and is redundant AS A LOGIT TEACHER —\n\
         the cost of running it buys nothing the fp teacher does not already give. The\n\
         KL(T‖T=1 student) column is the teaching-assistant question: a mid-T teacher that is much\n\
         closer to the student than fp is may still distil better even though it is a WORSE model,\n\
         because the student can actually reach it. That one this test cannot settle — it is about\n\
         optimisation dynamics, not distance at initialisation — but it says which pairs are worth\n\
         a training run."
    );
}
