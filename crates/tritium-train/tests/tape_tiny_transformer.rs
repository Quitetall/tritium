//! Gate C (ADR 0008 / plan 0011), full-model level: the reverse-mode tape backprops a
//! whole tiny transformer block (rmsnorm → q/k/v → RoPE → causal masked attention →
//! o_proj → residual → gated squared-ReLU MLP + sub-norm → residual → output-norm → LM
//! head → MSE) end-to-end, with every trainable leaf's analytic gradient matched to a
//! per-element central finite difference. This is the v0.50-deferred "full-model backprop"
//! wall (plan 0010 finding 4). Single attention head (n_head = n_head_kv = 1) keeps the
//! composition flat; GQA multi-head adds only a forward reshape + grad accumulation over
//! shared KV, which the tape's fan-out += already handles (B2). Dense projections (the
//! ternary STE matmul backward is gradient-checked separately in 0005) keep the FD clean.

use tritium_testkit::Tolerance;
use tritium_train::ops::{act, dense, elementwise, loss, norm, rope, softmax};
use tritium_train::tape::Tape;

const E: usize = 4; // n_embd == head_dim (single head)
const SEQ: usize = 3;
const FF: usize = 6;
const V: usize = 4;
const EPS: f32 = 1e-5;
const THETA: f32 = 10_000.0;
const SCALE: f32 = 0.5; // 1/sqrt(E) = 1/2

fn positions() -> Vec<usize> {
    (0..SEQ).collect()
}

/// Leaf layout (index → buffer), shared by the non-tape forward and the tape build.
/// 0 h0[SEQ,E] · 1 attn_norm[E] · 2 wq[E,E] · 3 wk[E,E] · 4 wv[E,E] · 5 wo[E,E] ·
/// 6 ffn_norm[E] · 7 wgate[FF,E] · 8 wup[FF,E] · 9 ffn_sub_norm[FF] · 10 wdown[E,FF] ·
/// 11 out_norm[E] · 12 wlm[V,E] · 13 target[SEQ,V] (data).
fn composed_loss(l: &[Vec<f32>]) -> f32 {
    let pos = positions();
    let xn = norm::forward(&l[0], &l[1], SEQ, E, EPS);
    let q = rope::forward(&dense::forward(&xn, &l[2], SEQ, E, E), &pos, 1, E, THETA);
    let k = rope::forward(&dense::forward(&xn, &l[3], SEQ, E, E), &pos, 1, E, THETA);
    let v = dense::forward(&xn, &l[4], SEQ, E, E);
    let mut scores = dense::forward(&q, &k, SEQ, SEQ, E); // [SEQ,SEQ]
    for s in &mut scores {
        *s *= SCALE;
    }
    let scores = softmax::causal_mask_forward(&scores, SEQ, SEQ);
    let p = softmax::forward(&scores, SEQ, SEQ);
    let vt = dense::transpose_forward(&v, SEQ, E); // [E,SEQ]
    let attn = dense::forward(&p, &vt, SEQ, E, SEQ); // [SEQ,E]
    let o = dense::forward(&attn, &l[5], SEQ, E, E);
    let h1 = elementwise::add_forward(&l[0], &o);

    let hn = norm::forward(&h1, &l[6], SEQ, E, EPS);
    let g = dense::forward(&hn, &l[7], SEQ, FF, E);
    let u = dense::forward(&hn, &l[8], SEQ, FF, E);
    let gated = elementwise::mul_forward(&act::relu2_forward(&g), &u);
    let gated = norm::forward(&gated, &l[9], SEQ, FF, EPS);
    let down = dense::forward(&gated, &l[10], SEQ, E, FF);
    let h2 = elementwise::add_forward(&h1, &down);

    let on = norm::forward(&h2, &l[11], SEQ, E, EPS);
    let logits = dense::forward(&on, &l[12], SEQ, V, E);
    loss::mse_forward(&logits, &l[13])[0]
}

/// The same graph on the tape; returns analytic grads for every leaf (input order).
fn composed_grads(l: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let pos = positions();
    let mut t = Tape::new();
    let id: Vec<_> = l.iter().map(|b| t.leaf(b.clone())).collect();

    let xn = t.rmsnorm(id[0], id[1], SEQ, E, EPS);
    let q0 = t.dense_matmul(xn, id[2], SEQ, E, E);
    let q = t.rope(q0, pos.clone(), 1, E, THETA);
    let k0 = t.dense_matmul(xn, id[3], SEQ, E, E);
    let k = t.rope(k0, pos.clone(), 1, E, THETA);
    let v = t.dense_matmul(xn, id[4], SEQ, E, E);
    let scores = t.dense_matmul(q, k, SEQ, SEQ, E);
    let scores = t.scale_const(scores, SCALE);
    let scores = t.causal_mask(scores, SEQ, SEQ);
    let p = t.softmax(scores, SEQ, SEQ);
    let vt = t.transpose(v, SEQ, E);
    let attn = t.dense_matmul(p, vt, SEQ, E, SEQ);
    let o = t.dense_matmul(attn, id[5], SEQ, E, E);
    let h1 = t.add(id[0], o);

    let hn = t.rmsnorm(h1, id[6], SEQ, E, EPS);
    let g = t.dense_matmul(hn, id[7], SEQ, FF, E);
    let u = t.dense_matmul(hn, id[8], SEQ, FF, E);
    let gr = t.relu2(g);
    let gated0 = t.mul(gr, u);
    let gated = t.rmsnorm(gated0, id[9], SEQ, FF, EPS);
    let down = t.dense_matmul(gated, id[10], SEQ, E, FF);
    let h2 = t.add(h1, down);

    let on = t.rmsnorm(h2, id[11], SEQ, E, EPS);
    let logits = t.dense_matmul(on, id[12], SEQ, V, E);
    let lid = t.mse(logits, id[13]);
    let grads = t.backward(lid);
    id.iter().map(|&i| grads[i].clone()).collect()
}

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

#[test]
fn tape_tiny_transformer_end_to_end_gradient() {
    // Small magnitudes keep the composed loss tame so the central-difference truncation
    // error stays under the Gate-C bar; norm inputs stay clearly nonzero.
    let base: Vec<Vec<f32>> = vec![
        seeded(1, SEQ * E, -1.0, 1.0),  // 0 h0
        seeded(2, E, 0.4, 1.6),         // 1 attn_norm
        seeded(3, E * E, -0.6, 0.6),    // 2 wq
        seeded(4, E * E, -0.6, 0.6),    // 3 wk
        seeded(5, E * E, -0.6, 0.6),    // 4 wv
        seeded(6, E * E, -0.6, 0.6),    // 5 wo
        seeded(7, E, 0.4, 1.6),         // 6 ffn_norm
        seeded(8, FF * E, -0.5, 0.5),   // 7 wgate
        seeded(9, FF * E, -0.5, 0.5),   // 8 wup
        seeded(10, FF, 0.4, 1.6),       // 9 ffn_sub_norm
        seeded(11, E * FF, -0.5, 0.5),  // 10 wdown
        seeded(12, E, 0.4, 1.6),        // 11 out_norm
        seeded(13, V * E, -0.6, 0.6),   // 12 wlm
        seeded(14, SEQ * V, -0.5, 0.5), // 13 target (data)
    ];

    let analytic = composed_grads(&base);
    let h = 1e-3f32;
    let tol = Tolerance {
        relative: 2e-3,
        bit_exact: false,
    };
    // Check every trainable leaf (0..=12); 13 is the data target.
    for leaf in 0..=12 {
        for i in 0..base[leaf].len() {
            let mut lv = base.clone();
            lv[leaf][i] += h;
            let lp = composed_loss(&lv);
            lv[leaf][i] -= 2.0 * h;
            let lm = composed_loss(&lv);
            let numeric = (lp - lm) / (2.0 * h);
            let a = analytic[leaf][i];
            assert!(
                tol.accepts(a, numeric),
                "leaf {leaf}[{i}]: analytic {a} vs numeric {numeric}"
            );
        }
    }
}
