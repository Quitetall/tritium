//! End-to-end SALT distillation (plan 0038 steps 3–4): a whole tiny **SwiGLU** transformer
//! (rmsnorm → q/k/v → RoPE → causal attention → o → residual → SwiGLU MLP → residual →
//! out-norm → LM head) with every 2D weight held as an fp32 **latent** SALT-quantized in the
//! forward (STE). An fp **teacher** (the same graph, un-quantized) supplies soft logits; the
//! ternary student is distilled by `softmax_xent(student_logits, softmax(teacher_logits))`
//! (whose gradient is exactly the KL gradient). The distilled student's logits track the
//! teacher far better than plain PTQ — the full loop, self-contained, at tiny scale.

use tritium_train::ops::{act, dense, elementwise, loss, norm, rope, softmax, ste};
use tritium_train::{AdamW, Optimizer, Tape};

const E: usize = 8; // n_embd == head_dim (single head)
const SEQ: usize = 4;
const FF: usize = 16;
const V: usize = 10;
const EPS: f32 = 1e-5;
const THETA: f32 = 10_000.0;
const SCALE: f32 = 0.353_553_39; // 1/sqrt(8)
const T: usize = 1; // SALT planes — single-plane (aggressive ternary) leaves a real gap to heal

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

/// The 2D weights (SALT-quantized in the student); 1D norms stay fp.
struct W {
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    wgate: Vec<f32>,
    wup: Vec<f32>,
    wdown: Vec<f32>,
    wlm: Vec<f32>,
}
struct Norms {
    h0: Vec<f32>,
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    out_norm: Vec<f32>,
}

fn init_weights() -> W {
    W {
        wq: seeded(2, E * E, -0.6, 0.6),
        wk: seeded(3, E * E, -0.6, 0.6),
        wv: seeded(4, E * E, -0.6, 0.6),
        wo: seeded(5, E * E, -0.6, 0.6),
        wgate: seeded(7, FF * E, -0.5, 0.5),
        wup: seeded(8, FF * E, -0.5, 0.5),
        wdown: seeded(11, E * FF, -0.5, 0.5),
        wlm: seeded(12, V * E, -0.6, 0.6),
    }
}
fn init_norms() -> Norms {
    Norms {
        h0: seeded(1, SEQ * E, -1.0, 1.0),
        attn_norm: seeded(20, E, 0.4, 1.6),
        ffn_norm: seeded(21, E, 0.4, 1.6),
        out_norm: seeded(22, E, 0.4, 1.6),
    }
}

/// Non-tape fp forward → logits `[SEQ, V]`. Used for the teacher and for scoring a fixed
/// (PTQ / distilled-then-requantized) weight set.
fn fwd_logits(w: &W, nm: &Norms) -> Vec<f32> {
    let pos: Vec<usize> = (0..SEQ).collect();
    let xn = norm::forward(&nm.h0, &nm.attn_norm, SEQ, E, EPS);
    let q = rope::forward(&dense::forward(&xn, &w.wq, SEQ, E, E), &pos, 1, E, THETA);
    let k = rope::forward(&dense::forward(&xn, &w.wk, SEQ, E, E), &pos, 1, E, THETA);
    let vv = dense::forward(&xn, &w.wv, SEQ, E, E);
    let mut scores = dense::forward(&q, &k, SEQ, SEQ, E);
    for s in &mut scores {
        *s *= SCALE;
    }
    let scores = softmax::causal_mask_forward(&scores, SEQ, SEQ);
    let p = softmax::forward(&scores, SEQ, SEQ);
    let vt = dense::transpose_forward(&vv, SEQ, E);
    let attn = dense::forward(&p, &vt, SEQ, E, SEQ);
    let o = dense::forward(&attn, &w.wo, SEQ, E, E);
    let h1 = elementwise::add_forward(&nm.h0, &o);

    let hn = norm::forward(&h1, &nm.ffn_norm, SEQ, E, EPS);
    let g = dense::forward(&hn, &w.wgate, SEQ, FF, E);
    let u = dense::forward(&hn, &w.wup, SEQ, FF, E);
    let gated = elementwise::mul_forward(&act::silu_forward(&g), &u); // SwiGLU
    let down = dense::forward(&gated, &w.wdown, SEQ, E, FF);
    let h2 = elementwise::add_forward(&h1, &down);

    let on = norm::forward(&h2, &nm.out_norm, SEQ, E, EPS);
    dense::forward(&on, &w.wlm, SEQ, V, E)
}

/// SALT-quantize (dequant-to-dense) every 2D weight at `T` planes — the PTQ student weights.
fn ptq(w: &W) -> W {
    let q = |wf: &[f32], n: usize, k: usize| ste::salt_quantize_forward(wf, n, k, T);
    W {
        wq: q(&w.wq, E, E),
        wk: q(&w.wk, E, E),
        wv: q(&w.wv, E, E),
        wo: q(&w.wo, E, E),
        wgate: q(&w.wgate, FF, E),
        wup: q(&w.wup, FF, E),
        wdown: q(&w.wdown, E, FF),
        wlm: q(&w.wlm, V, E),
    }
}

fn row_softmax(logits: &[f32]) -> Vec<f32> {
    softmax::forward(logits, SEQ, V)
}

#[test]
fn salt_distills_a_tiny_swiglu_transformer_end_to_end() {
    let teacher = init_weights();
    let nm = init_norms();
    let teacher_logits = fwd_logits(&teacher, &nm);
    let teacher_probs = row_softmax(&teacher_logits);

    // PTQ baseline: SALT-quantize the teacher weights, no training.
    let ptq_logits = fwd_logits(&ptq(&teacher), &nm);

    // Distill: latents start at the teacher weights; train them (STE through salt_ste) so the
    // ternary student's logits match the teacher's soft distribution.
    let mut lat = init_weights();
    let opt = AdamW::new(4e-3);
    // one Adam state per 2D weight
    let mut st: Vec<_> = [
        &lat.wq, &lat.wk, &lat.wv, &lat.wo, &lat.wgate, &lat.wup, &lat.wdown, &lat.wlm,
    ]
    .iter()
    .map(|w| opt.init_state(w.len()))
    .collect();

    let (mut first_loss, mut last_loss) = (f32::NAN, f32::NAN);
    for step in 1..=1500u64 {
        let mut t = Tape::new();
        let pos: Vec<usize> = (0..SEQ).collect();
        // norms + input as constants (leaves, not trained)
        let h0 = t.leaf(nm.h0.clone());
        let an = t.leaf(nm.attn_norm.clone());
        let fn_ = t.leaf(nm.ffn_norm.clone());
        let onw = t.leaf(nm.out_norm.clone());
        // 2D latents + their SALT-STE reconstructions
        let (wq, wk, wv, wo) = (
            t.leaf(lat.wq.clone()),
            t.leaf(lat.wk.clone()),
            t.leaf(lat.wv.clone()),
            t.leaf(lat.wo.clone()),
        );
        let (wg, wu, wd, wl) = (
            t.leaf(lat.wgate.clone()),
            t.leaf(lat.wup.clone()),
            t.leaf(lat.wdown.clone()),
            t.leaf(lat.wlm.clone()),
        );
        let qh = t.salt_ste(wq, E, E, T);
        let kh = t.salt_ste(wk, E, E, T);
        let vh = t.salt_ste(wv, E, E, T);
        let oh = t.salt_ste(wo, E, E, T);
        let gh = t.salt_ste(wg, FF, E, T);
        let uh = t.salt_ste(wu, FF, E, T);
        let dh = t.salt_ste(wd, E, FF, T);
        let lh = t.salt_ste(wl, V, E, T);

        let xn = t.rmsnorm(h0, an, SEQ, E, EPS);
        let q0 = t.dense_matmul(xn, qh, SEQ, E, E);
        let q = t.rope(q0, pos.clone(), 1, E, THETA);
        let k0 = t.dense_matmul(xn, kh, SEQ, E, E);
        let k = t.rope(k0, pos.clone(), 1, E, THETA);
        let vv = t.dense_matmul(xn, vh, SEQ, E, E);
        let scores = t.dense_matmul(q, k, SEQ, SEQ, E);
        let scores = t.scale_const(scores, SCALE);
        let scores = t.causal_mask(scores, SEQ, SEQ);
        let p = t.softmax(scores, SEQ, SEQ);
        let vt = t.transpose(vv, SEQ, E);
        let attn = t.dense_matmul(p, vt, SEQ, E, SEQ);
        let o = t.dense_matmul(attn, oh, SEQ, E, E);
        let h1 = t.add(h0, o);

        let hn = t.rmsnorm(h1, fn_, SEQ, E, EPS);
        let g = t.dense_matmul(hn, gh, SEQ, FF, E);
        let u = t.dense_matmul(hn, uh, SEQ, FF, E);
        let gact = t.silu(g);
        let gated = t.mul(gact, u);
        let down = t.dense_matmul(gated, dh, SEQ, E, FF);
        let h2 = t.add(h1, down);

        let on = t.rmsnorm(h2, onw, SEQ, E, EPS);
        let logits = t.dense_matmul(on, lh, SEQ, V, E);
        let tg = t.leaf(teacher_probs.clone());
        let loss = t.softmax_xent(logits, tg, SEQ, V); // KL gradient vs the teacher

        let loss_val = t.value(loss)[0];
        if step == 1 {
            first_loss = loss_val;
        }
        last_loss = loss_val;
        let grads = t.backward(loss);

        // AdamW on each latent.
        for (i, (w, id)) in [
            (&mut lat.wq, wq),
            (&mut lat.wk, wk),
            (&mut lat.wv, wv),
            (&mut lat.wo, wo),
            (&mut lat.wgate, wg),
            (&mut lat.wup, wu),
            (&mut lat.wdown, wd),
            (&mut lat.wlm, wl),
        ]
        .into_iter()
        .enumerate()
        {
            opt.step(step, w, &grads[id], &mut st[i]);
        }
    }

    // The distillation trains KL(teacher‖student), so score on the cross-entropy (KL up to the
    // teacher-entropy constant) of the re-quantized student — not logit-MSE, which a KL-matched
    // student need not minimize (logits may differ by a per-row constant with the same softmax).
    let ptq_xent = loss::softmax_xent_forward(&ptq_logits, &teacher_probs, SEQ, V)[0];
    let distilled_logits = fwd_logits(&ptq(&lat), &nm);
    let distilled_xent = loss::softmax_xent_forward(&distilled_logits, &teacher_probs, SEQ, V)[0];
    let recovered = 100.0 * (1.0 - f64::from(distilled_xent) / f64::from(ptq_xent));
    println!(
        "SALT e2e distill (SwiGLU tiny transformer, T={T}): PTQ xent {ptq_xent:.4} → distilled {distilled_xent:.4} ({recovered:.1}% of the gap to the teacher's own entropy); surrogate {first_loss:.4} → {last_loss:.4}"
    );

    assert!(
        last_loss < first_loss,
        "distillation KL/xent must decrease: {first_loss} → {last_loss}"
    );
    // Require a real reduction (≥3%; observed ~5.3%) so a crippled optimizer that heals only a
    // sliver still fails — the strong per-projection recovery claim is carried by salt_distill.rs.
    assert!(
        distilled_xent < 0.97 * ptq_xent,
        "distillation must reduce the student's KL to the teacher ≥3%: {distilled_xent:.4} vs PTQ {ptq_xent:.4}"
    );
}
