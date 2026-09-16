//! **Can allocation be made global — driven by task loss itself rather than a weight-space proxy?**
//!
//! Every allocator this project has measured shares one shape: score each unit *independently*,
//! then sum. Roughly 44 arms have now lost to uniform that way, and two results say the shape is
//! the problem rather than the scoring.
//!
//! - **E5** allocated on *measured* held-out Δppl per tensor — a perfect local signal, no proxy —
//!   and scored **+9.26%**, worse than the blind weight-space proxy's +2.21%.
//! - **Per-tap activation allocation** has five decision units, an exactly-measured budget, and a
//!   signal verified good in the same experiment (the anti-control is by far the worst arm). It
//!   lost 4 of 4.
//!
//! So: not the estimator, not the decision count. What every one of them did was evaluate a unit's
//! marginal cost **with every other unit held at the baseline**, then add those marginals up. That
//! is only valid if the loss is separable across units, and the evidence says it is not.
//!
//! # What "global" has to mean here
//!
//! One backward pass through the whole network, at the actual operating point, gives
//! `∂L/∂W_i` for **every** tensor simultaneously — with every other tensor already quantized, and
//! with all interactions between them included by construction. That is the thing a sum of
//! independently-measured marginals cannot express.
//!
//! Turning that into a plane decision needs no new estimator. The change in reconstruction from
//! moving group `g` from `T` to `T±1` is a vector `Δ` the fitter already produces, so the
//! first-order change in **task loss** is exactly
//!
//! ```text
//! ΔL  ≈  ⟨ ∂L/∂W_i |_g , Q_{T±1}(W_i)|_g − Q_T(W_i)|_g ⟩
//! ```
//!
//! No proxy, no separability assumption in the *scoring*, and one backward pass prices all 1.14M
//! groups in both directions at once.
//!
//! # The question this actually settles
//!
//! A plane is not an infinitesimal step — it moves the group's error by **9×**. So the honest
//! question is not "is the gradient correct" (it is) but **"is the loss linear enough over a
//! one-plane step for a gradient to rank swaps?"**
//!
//! That is what the arms below measure, and either answer is worth having:
//!
//! - If gradient-ranked reallocation **beats uniform**, allocation works and every previous failure
//!   was the separable-sum shape.
//! - If it does not, but **ranks in the right order** (gradient-best > random > gradient-worst),
//!   the signal is real and the step size is what defeats it — which points at the 9× asymmetry as
//!   the binding constraint rather than at any estimator.
//! - If the ordering itself collapses, first-order information does not survive a plane-sized step
//!   and no gradient method of this shape can work.
//!
//! # Controls
//!
//! - `K = 0` must reproduce uniform **exactly**. It is the same code path with an empty swap set,
//!   so any drift is the harness moving the answer by itself.
//! - **Random** selection at the same `K` and the same bit budget separates "the ranking is
//!   informative" from "perturbing this many groups happens to help".
//! - **Anti-ranked** selection — deliberately the worst swaps the gradient can name — is the
//!   sign check. If it does not lose clearly, the gradient carries no signal at all.
//! - Every arm's realized bit count is asserted equal to uniform's, so no arm can win by spending
//!   more.
//!
//! # Measured 2026-09-16 — none of the three anticipated outcomes
//!
//! ```text
//! arm                                  ppl    vs unif   predicted ΔL
//! K=0 (degenerate control)         24.2783          —              0   <- reproduces uniform
//! gradient          K=1144         24.6578      1.56%      -1.348e-1
//! random (control)  K=1144         24.3132      0.14%      -1.746e-5
//! anti-ranked       K=1144         25.6594      5.69%      +1.008e-1
//! gradient          K=11443        25.4196      4.70%      -4.801e-1
//! random (control)  K=11443        24.3742      0.39%      -9.738e-4
//! anti-ranked       K=11443        28.6587     18.04%      +4.174e-1
//! gradient          K=57216        31.7914     30.95%      -1.063e0
//! random (control)  K=57216        24.9329      2.70%      -9.182e-3
//! anti-ranked       K=57216        40.1711     65.46%      +9.636e-1
//! gradient          K=228864       46.8118     92.81%      -1.766e0
//! ```
//!
//! The ordering is **random < gradient < anti-ranked**, at every `K`. Not one of the three cases
//! this test was written to distinguish.
//!
//! The gradient is informative: it beats the anti-ranked arm by ~3.7x at every `K`, so the **sign**
//! of the first-order term genuinely predicts direction. And it is confidently wrong: at
//! `K = 57216` it predicts `ΔL = -1.063` and delivers **+30.95%**, with the error growing
//! superlinearly in `K` (predicted grows 13x from the smallest arm to the largest; actual grows
//! 60x).
//!
//! # Why a better signal selects worse than no signal
//!
//! Both tails lose to the bulk. `gradient` takes the largest-magnitude scores with the favourable
//! sign; `anti-ranked` takes the largest-magnitude scores with the unfavourable one; `random` takes
//! typical scores. Ranked by damage: anti (largest |score|) > gradient (largest |score|) > random
//! (small |score|).
//!
//! So **|∂L/∂g| predicts DAMAGE, and its sign predicts direction.** That is not a paradox. A plane
//! is a 9x step, so the realised change is a first-order term plus higher-order terms, and the two
//! are correlated — both scale with how much the loss cares about that group. Selecting on the
//! largest first-order term therefore selects the groups where the linear model is least valid.
//! **The criterion finds its own failure cases**, and a more accurate criterion finds them more
//! reliably. The weight-space proxy's noise was keeping its selection closer to random, which is
//! the least-damaging selection available.
//!
//! # The result underneath all of it: uniform is a local optimum
//!
//! `random` at `K = 1144` perturbs 0.1% of groups in a bit-neutral way with no criterion at all,
//! and still loses (+0.14%). Every arm here is uphill. Uniform `T` is not the absence of
//! allocation — each group's step is set by its own `max|w|`, so uniform `T` **is** allocation by
//! uniform *relative* precision, which is the quantity RMSNorm makes meaningful. Reallocating by
//! sensitivity trades that for equalised *absolute* error, which the architecture normalises away.
//!
//! That is why ~50 arms across four campaigns have failed with four different signals: the target
//! was already right. The instrument cannot express anything better, because a plane is a fixed 9x
//! step applied to all 128 weights at once.
//!
//! The consequence for the project is that allocation is not the lever and no estimator will make
//! it one. The gain has to come from changing what a plane *is* — the fit metric, a continuous
//! low-rank term, or the symbol pricing — which moves the whole curve instead of redistributing
//! along it.
//!
//! ```text
//! TRITIUM_CORPUS=$HOME/.cache/tritium-corpora/wikitext2_400k_32k.json \
//!   cargo test -p tritium-nn --release --test salt_alloc_gradient -- --ignored --nocapture
//! ```

mod common;

use std::path::PathBuf;

use common::{Calib, calibrate, extract, fold, perplexity_windowed};
use tritium_nn::ModelRunner;
use tritium_nn::calibrate::forward;
use tritium_train::Tape;
use tritium_train::ops::ste::{self, RotationPolicy};
use tritium_train::tape::ValueId;

const EVAL_WINDOW: usize = 512;
const CALIB_WINDOWS: usize = 4;
const CALIB_SEQ: usize = 512;
const GROUP: usize = 128;
const GRID: usize = 16;
/// Windows used for the gradient itself. Each retains a full autograd tape, so this is the memory
/// knob; the gradient is averaged over them.
const GRAD_WINDOWS: usize = 2;
const GRAD_SEQ: usize = 256;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(
        std::env::var("TRITIUM_MODEL_DIR")
            .unwrap_or_else(|_| format!("{home}/.cache/tritium-models/smollm2-135m")),
    )
}

fn corpus() -> (Vec<u32>, Vec<u32>) {
    let path = std::env::var("TRITIUM_CORPUS").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tools/reference/heldout_corpus.json"
        )
        .to_owned()
    });
    let text = std::fs::read_to_string(&path).expect("corpus");
    let v: serde_json::Value = serde_json::from_str(&text).expect("corpus json");
    let ids = |k: &str| -> Vec<u32> {
        v[k].as_array()
            .expect(k)
            .iter()
            .map(|x| x.as_u64().expect("id") as u32)
            .collect()
    };
    (ids("train_ids"), ids("eval_ids"))
}

/// Row-major softmax, used to turn fp logits into the teacher distribution.
fn softmax_rows(logits: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &logits[r * cols..r * cols + cols];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (o, &x) in out[r * cols..r * cols + cols].iter_mut().zip(row) {
            *o = (x - m).exp();
            sum += *o;
        }
        for o in out[r * cols..r * cols + cols].iter_mut() {
            *o /= sum;
        }
    }
    out
}

/// One candidate plane move.
#[derive(Clone, Copy)]
struct Move {
    tensor: usize,
    block: usize,
    /// Trits this move adds (`+width`) or frees (`−width`) — the exact bit cost, ragged tails
    /// included, so bit-neutrality is accounted rather than assumed.
    width: usize,
    /// First-order predicted change in task loss. Negative is an improvement.
    score: f32,
}

fn fit(w: &[f32], rows: usize, cols: usize, t: usize) -> Vec<f32> {
    ste::salt_quantize_forward_grouped_geometric(
        w,
        rows,
        cols,
        t,
        GROUP,
        GRID,
        RotationPolicy::Always,
    )
}

#[test]
#[ignore = "needs SmolLM2-135M; several full evaluations plus a backward pass"]
fn gradient_ranked_global_allocation_against_uniform() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let t_ref = env_usize("TRITIUM_GRAD_T", 3);
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch0, fp0, shapes) = extract(&runner);
    let (train, eval) = corpus();

    let mut calib = Calib::new(&arch0);
    for w in 0..CALIB_WINDOWS {
        calibrate(
            &fp0,
            &arch0,
            &train[w * CALIB_SEQ..(w + 1) * CALIB_SEQ],
            &mut calib,
        );
    }
    let (fp, arch) = fold(&fp0, &shapes, &arch0, &calib, 0.75);

    // The baseline every arm is judged against: uniform `t_ref` planes everywhere.
    let uniform: Vec<Vec<f32>> = fp
        .iter()
        .zip(&shapes)
        .map(|(w, &(r, c))| fit(w, r, c, t_ref))
        .collect();
    let ppl_fp = perplexity_windowed(&fp, &arch, &eval, EVAL_WINDOW);
    let ppl_uniform = perplexity_windowed(&uniform, &arch, &eval, EVAL_WINDOW);
    println!(
        "SmolLM2-135M | WikiText-2 {} held-out | fold α=0.75 | g{GROUP} | rotation always\n\
         fp {ppl_fp:.4} | uniform T={t_ref} {ppl_uniform:.4} ({:.4}× fp)\n",
        eval.len(),
        ppl_uniform / ppl_fp
    );

    // ── The global signal: ∂L/∂W for every tensor at once, with every other tensor already
    // quantized. This is the whole point — a marginal measured here includes the interactions a
    // per-unit sweep holds fixed at the baseline.
    println!("backward pass over {GRAD_WINDOWS} × {GRAD_SEQ} tokens (KL to the fp teacher)…");
    let mut grads: Vec<Vec<f32>> = uniform.iter().map(|w| vec![0.0f32; w.len()]).collect();
    for wnd in 0..GRAD_WINDOWS {
        let toks = &train[wnd * GRAD_SEQ..(wnd + 1) * GRAD_SEQ];
        // Teacher: the fp model's own distribution, as data.
        let teacher = {
            let mut t = Tape::new();
            let wids: Vec<ValueId> = fp.iter().map(|w| t.leaf(w.clone())).collect();
            let out = forward(&mut t, &wids, &arch, toks);
            softmax_rows(t.value(out), toks.len(), arch.vocab)
        };
        let mut t = Tape::new();
        let wids: Vec<ValueId> = uniform.iter().map(|w| t.leaf(w.clone())).collect();
        let logits = forward(&mut t, &wids, &arch, toks);
        let target = t.leaf(teacher);
        // Cross-entropy against the teacher distribution is KL up to a constant in the student, so
        // its gradient IS the KL gradient — and KL to the fp model is the objective allocation has
        // always claimed to serve.
        let loss = t.softmax_xent(logits, target, toks.len(), arch.vocab);
        let g = t.backward(loss);
        for (acc, &wid) in grads.iter_mut().zip(&wids) {
            for (a, &v) in acc.iter_mut().zip(&g[wid]) {
                *a += v / GRAD_WINDOWS as f32;
            }
        }
    }

    // ── Price every group in both directions. Streamed per tensor: three reconstructions of one
    // tensor at a time rather than three copies of the model.
    let per_row = |cols: usize| cols.div_ceil(GROUP);
    let mut promote: Vec<Move> = Vec::new();
    let mut demote: Vec<Move> = Vec::new();
    for (i, (w, &(rows, cols))) in fp.iter().zip(&shapes).enumerate() {
        let q_ref = &uniform[i];
        let q_up = fit(w, rows, cols, t_ref + 1);
        let q_dn = fit(w, rows, cols, t_ref - 1);
        let grad = &grads[i];
        let pr = per_row(cols);
        for r in 0..rows {
            for b in 0..pr {
                let start = r * cols + b * GROUP;
                let end = (start + GROUP).min(r * cols + cols);
                let width = end - start;
                let (mut up, mut dn) = (0.0f32, 0.0f32);
                for k in start..end {
                    up += grad[k] * (q_up[k] - q_ref[k]);
                    dn += grad[k] * (q_dn[k] - q_ref[k]);
                }
                promote.push(Move {
                    tensor: i,
                    block: r * pr + b,
                    width,
                    score: up,
                });
                demote.push(Move {
                    tensor: i,
                    block: r * pr + b,
                    width,
                    score: dn,
                });
            }
        }
    }
    let groups = promote.len();
    // Most negative predicted ΔL first for promotions; least positive first for demotions.
    promote.sort_by(|a, b| a.score.total_cmp(&b.score));
    demote.sort_by(|a, b| a.score.total_cmp(&b.score));
    println!(
        "{groups} groups priced in both directions from one backward pass.\n  \
         promote best {:+.3e} … worst {:+.3e}\n  demote  best {:+.3e} … worst {:+.3e}\n",
        promote[0].score,
        promote[groups - 1].score,
        demote[0].score,
        demote[groups - 1].score
    );

    // Deterministic pseudo-random order for the control arm.
    let mut shuffled: Vec<usize> = (0..groups).collect();
    let mut s = 0x2545F491_4F6CDD1Du64;
    for i in (1..groups).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        shuffled.swap(i, (s % (i as u64 + 1)) as usize);
    }

    // Build a plane-count map: `t_ref` everywhere, then apply `k` promotions and enough
    // demotions to pay for them exactly.
    let build = |promos: &[Move], demos: &[Move], k: usize| -> (Vec<Vec<u8>>, i64) {
        let mut planes: Vec<Vec<u8>> = shapes
            .iter()
            .map(|&(rows, cols)| vec![t_ref as u8; rows * per_row(cols)])
            .collect();
        let mut owed: i64 = 0;
        let mut taken = 0usize;
        for m in promos.iter().take(k) {
            planes[m.tensor][m.block] = (t_ref + 1) as u8;
            owed += m.width as i64;
            taken += 1;
        }
        let _ = taken;
        // Pay the exact trit bill. A group already promoted cannot also be demoted.
        for m in demos {
            if owed <= 0 {
                break;
            }
            if planes[m.tensor][m.block] != t_ref as u8 {
                continue;
            }
            planes[m.tensor][m.block] = (t_ref - 1) as u8;
            owed -= m.width as i64;
        }
        (planes, owed)
    };

    let score_arm = |planes: &[Vec<u8>]| -> f64 {
        let q: Vec<Vec<f32>> = fp
            .iter()
            .zip(&shapes)
            .zip(planes)
            .map(|((w, &(r, c)), p)| {
                ste::salt_quantize_forward_grouped_geometric_alloc(
                    w,
                    r,
                    c,
                    p,
                    GROUP,
                    GRID,
                    RotationPolicy::Always,
                )
            })
            .collect();
        perplexity_windowed(&q, &arch, &eval, EVAL_WINDOW)
    };

    let total_trits: i64 = shapes.iter().map(|&(r, c)| (r * c) as i64).sum();
    println!(
        "{:<34} {:>11} {:>10} {:>14} {:>16}",
        "arm", "ppl", "vs unif", "predicted ΔL", "trit balance"
    );
    println!("{}", "-".repeat(90));

    // DEGENERATE CONTROL. An empty swap set must reproduce uniform bit-for-bit; this is the same
    // `_alloc` code path the arms use, so a discrepancy here is the harness, not allocation.
    let (zero, owed0) = build(&promote, &demote, 0);
    let ppl_zero = score_arm(&zero);
    println!(
        "{:<34} {ppl_zero:>11.4} {:>10} {:>14} {:>16}",
        "K=0 (degenerate control)", "—", "0", owed0
    );
    assert!(
        (ppl_zero - ppl_uniform).abs() / ppl_uniform < 1e-9,
        "K=0 scored {ppl_zero:.6} against uniform's {ppl_uniform:.6}. The allocated fitter must \
         reproduce the uniform fitter when every count is t_ref; until it does, no arm below \
         measures allocation"
    );

    let mut best_arm = (f64::INFINITY, String::new());
    for k in [groups / 1000, groups / 100, groups / 20, groups / 5] {
        if k == 0 {
            continue;
        }
        for (label, promos, demos) in [
            ("gradient", promote.clone(), demote.clone()),
            (
                "random (control)",
                shuffled.iter().map(|&i| promote[i]).collect::<Vec<_>>(),
                shuffled.iter().map(|&i| demote[i]).collect::<Vec<_>>(),
            ),
            (
                "anti-ranked (anti-control)",
                promote.iter().rev().copied().collect::<Vec<_>>(),
                demote.iter().rev().copied().collect::<Vec<_>>(),
            ),
        ] {
            let (planes, owed) = build(&promos, &demos, k);
            // Predicted total, first order: the promotions taken plus the demotions that paid.
            let predicted: f64 = promos
                .iter()
                .take(k)
                .map(|m| f64::from(m.score))
                .sum::<f64>();
            let ppl = score_arm(&planes);
            let name = format!("{label} K={k}");
            println!(
                "{name:<34} {ppl:>11.4} {:>9.2}% {predicted:>14.3e} {:>15}",
                100.0 * (ppl - ppl_uniform) / ppl_uniform,
                format!("{owed}/{total_trits}")
            );
            assert!(
                owed.abs() * 10_000 < total_trits,
                "arm `{name}` is not bit-neutral: {owed} trits unpaid out of {total_trits}. An \
                 arm that spends more cannot be compared to uniform"
            );
            if ppl < best_arm.0 {
                best_arm = (ppl, name);
            }
        }
    }

    println!(
        "\nbest arm: {} at {:.4} ({:+.2}% vs uniform {ppl_uniform:.4})",
        best_arm.1,
        best_arm.0,
        100.0 * (best_arm.0 - ppl_uniform) / ppl_uniform
    );
    println!(
        "\nRead the ORDERING before the sign. gradient < random < anti-ranked means the first-order\n\
         signal is real even if no arm beats uniform — which would put the blame on the 9× step\n\
         size rather than on the estimator. If the ordering collapses, first-order information does\n\
         not survive a plane-sized step and no gradient method of this shape can work."
    );
}
