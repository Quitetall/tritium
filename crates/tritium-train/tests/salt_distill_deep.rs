//! End-to-end SALT distillation across DEPTH (plan 0038b step 1) — the bridge from the 1-block e2e
//! mechanism (`salt_distill_e2e.rs`) to the real 30-layer capstone.
//!
//! 0038 step 5 showed on the real SmolLM2 that ternary PTQ is catastrophic because error COMPOUNDS
//! across layers, and that a purely *local* layerwise heal cannot rescue it. The fix is end-to-end
//! distillation — train every latent jointly against the teacher's final output so the gradient
//! flows through the whole depth. This test generalizes the 1-block e2e to N blocks and shows the
//! recovery works when the gradient must flow through the FULL depth jointly — the multi-layer case
//! the local per-projection heal (step 5) could not handle. (The compounding *catastrophe* itself
//! is a trained-real-scale effect, shown on SmolLM2 in step 5; independent tiny random models don't
//! reproduce monotone growth-with-depth, so we report per-depth PTQ but gate only on recovery.)
//!
//! An N-block transformer (rmsnorm→qkv→RoPE→causal attn→o→residual→SwiGLU→residual, ×N →out-norm
//! →lm-head) with every 2D weight an fp32 latent SALT-quantized in the forward (STE, T=1 =
//! aggressive ternary, a real gap). The fp teacher (same graph, un-quantized) supplies soft logits;
//! the student is distilled by `softmax_xent(student, softmax(teacher))` (= the KL gradient) over
//! ALL latents jointly with AdamW.

use tritium_train::ops::{act, dense, elementwise, loss, norm, rope, softmax, ste};
use tritium_train::{AdamW, Optimizer, Tape};

const E: usize = 8; // n_embd == head_dim (single head)
const SEQ: usize = 4;
const FF: usize = 16;
const V: usize = 10;
const EPS: f32 = 1e-5;
const THETA: f32 = 10_000.0;
const SCALE: f32 = 0.353_553_39; // 1/sqrt(8)
const T: usize = 1; // single-plane ternary — leaves a real gap to heal
const N_BLOCKS: usize = 3;

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

/// A tiny N-block transformer. 2D weights are a flat list (index 0 = lm head, then per block
/// `wq,wk,wv,wo,wgate,wup,wdown`); 1D norms stay fp.
struct Model {
    lat: Vec<Vec<f32>>,
    shapes: Vec<(usize, usize)>,
    attn_norms: Vec<Vec<f32>>,
    ffn_norms: Vec<Vec<f32>>,
    out_norm: Vec<f32>,
    h0: Vec<f32>,
    n_blocks: usize,
}

fn build(n_blocks: usize) -> Model {
    let mut lat = vec![seeded(12, V * E, -0.6, 0.6)]; // wlm
    let mut shapes = vec![(V, E)];
    let (mut attn_norms, mut ffn_norms) = (Vec::new(), Vec::new());
    for b in 0..n_blocks {
        let s = b as u64 * 100;
        for (seed, rows, cols) in [
            (2 + s, E, E),   // wq
            (3 + s, E, E),   // wk
            (4 + s, E, E),   // wv
            (5 + s, E, E),   // wo
            (7 + s, FF, E),  // wgate
            (8 + s, FF, E),  // wup
            (11 + s, E, FF), // wdown
        ] {
            let (lo, hi) = if rows == E && cols == E {
                (-0.6, 0.6)
            } else {
                (-0.5, 0.5)
            };
            lat.push(seeded(seed, rows * cols, lo, hi));
            shapes.push((rows, cols));
        }
        attn_norms.push(seeded(20 + s, E, 0.4, 1.6));
        ffn_norms.push(seeded(21 + s, E, 0.4, 1.6));
    }
    Model {
        lat,
        shapes,
        attn_norms,
        ffn_norms,
        out_norm: seeded(22, E, 0.4, 1.6),
        h0: seeded(1, SEQ * E, -1.0, 1.0),
        n_blocks,
    }
}

/// Plain fp forward with the supplied 2D weight set `w` (teacher = raw latents; PTQ/student =
/// SALT-quantized). Returns logits `[SEQ, V]`.
fn fwd(m: &Model, w: &[Vec<f32>]) -> Vec<f32> {
    let pos: Vec<usize> = (0..SEQ).collect();
    let mut h = m.h0.clone();
    for b in 0..m.n_blocks {
        let base = 1 + 7 * b;
        let xn = norm::forward(&h, &m.attn_norms[b], SEQ, E, EPS);
        let q = rope::forward(&dense::forward(&xn, &w[base], SEQ, E, E), &pos, 1, E, THETA);
        let k = rope::forward(
            &dense::forward(&xn, &w[base + 1], SEQ, E, E),
            &pos,
            1,
            E,
            THETA,
        );
        let v = dense::forward(&xn, &w[base + 2], SEQ, E, E);
        let mut scores = dense::forward(&q, &k, SEQ, SEQ, E);
        for sc in &mut scores {
            *sc *= SCALE;
        }
        let scores = softmax::causal_mask_forward(&scores, SEQ, SEQ);
        let p = softmax::forward(&scores, SEQ, SEQ);
        let vt = dense::transpose_forward(&v, SEQ, E);
        let attn = dense::forward(&p, &vt, SEQ, E, SEQ);
        let o = dense::forward(&attn, &w[base + 3], SEQ, E, E);
        let h1 = elementwise::add_forward(&h, &o);
        let hn = norm::forward(&h1, &m.ffn_norms[b], SEQ, E, EPS);
        let g = dense::forward(&hn, &w[base + 4], SEQ, FF, E);
        let u = dense::forward(&hn, &w[base + 5], SEQ, FF, E);
        let gated = elementwise::mul_forward(&act::silu_forward(&g), &u);
        let down = dense::forward(&gated, &w[base + 6], SEQ, E, FF);
        h = elementwise::add_forward(&h1, &down);
    }
    let on = norm::forward(&h, &m.out_norm, SEQ, E, EPS);
    dense::forward(&on, &w[0], SEQ, V, E)
}

/// SALT-quantize (dequant-to-dense) every 2D weight at `T` planes — the PTQ / re-scored student set.
fn quantize(w: &[Vec<f32>], shapes: &[(usize, usize)]) -> Vec<Vec<f32>> {
    w.iter()
        .zip(shapes)
        .map(|(wf, &(r, c))| ste::salt_quantize_forward(wf, r, c, T))
        .collect()
}

fn row_softmax(logits: &[f32]) -> Vec<f32> {
    softmax::forward(logits, SEQ, V)
}

/// Distill all latents end-to-end against `teacher_probs`; return the re-quantized student's xent.
fn distill(m: &Model, teacher_probs: &[f32], steps: u64) -> (f32, f32, f32) {
    let mut lat = m.lat.clone();
    let opt = AdamW::new(4e-3);
    let mut states: Vec<_> = lat.iter().map(|w| opt.init_state(w.len())).collect();
    let pos: Vec<usize> = (0..SEQ).collect();
    let (mut first, mut last) = (f32::NAN, f32::NAN);

    for step in 1..=steps {
        let mut t = Tape::new();
        // Leaf + SALT-STE every latent; keep the leaf ids to update them after backward.
        let mut leaf_ids = Vec::with_capacity(lat.len());
        let mut ste_ids = Vec::with_capacity(lat.len());
        for (i, w) in lat.iter().enumerate() {
            let l = t.leaf(w.clone());
            let (r, c) = m.shapes[i];
            ste_ids.push(t.salt_ste(l, r, c, T));
            leaf_ids.push(l);
        }

        let mut h = t.leaf(m.h0.clone());
        for b in 0..m.n_blocks {
            let base = 1 + 7 * b;
            let an = t.leaf(m.attn_norms[b].clone());
            let fnw = t.leaf(m.ffn_norms[b].clone());
            let xn = t.rmsnorm(h, an, SEQ, E, EPS);
            let q0 = t.dense_matmul(xn, ste_ids[base], SEQ, E, E);
            let q = t.rope(q0, pos.clone(), 1, E, THETA);
            let k0 = t.dense_matmul(xn, ste_ids[base + 1], SEQ, E, E);
            let k = t.rope(k0, pos.clone(), 1, E, THETA);
            let v = t.dense_matmul(xn, ste_ids[base + 2], SEQ, E, E);
            let scores = t.dense_matmul(q, k, SEQ, SEQ, E);
            let scores = t.scale_const(scores, SCALE);
            let scores = t.causal_mask(scores, SEQ, SEQ);
            let p = t.softmax(scores, SEQ, SEQ);
            let vt = t.transpose(v, SEQ, E);
            let attn = t.dense_matmul(p, vt, SEQ, E, SEQ);
            let o = t.dense_matmul(attn, ste_ids[base + 3], SEQ, E, E);
            let h1 = t.add(h, o);
            let hn = t.rmsnorm(h1, fnw, SEQ, E, EPS);
            let g = t.dense_matmul(hn, ste_ids[base + 4], SEQ, FF, E);
            let u = t.dense_matmul(hn, ste_ids[base + 5], SEQ, FF, E);
            let gact = t.silu(g);
            let gated = t.mul(gact, u);
            let down = t.dense_matmul(gated, ste_ids[base + 6], SEQ, E, FF);
            h = t.add(h1, down);
        }
        let onw = t.leaf(m.out_norm.clone());
        let on = t.rmsnorm(h, onw, SEQ, E, EPS);
        let logits = t.dense_matmul(on, ste_ids[0], SEQ, V, E);
        let tg = t.leaf(teacher_probs.to_vec());
        let l = t.softmax_xent(logits, tg, SEQ, V);

        let lv = t.value(l)[0];
        if step == 1 {
            first = lv;
        }
        last = lv;
        let grads = t.backward(l);
        for i in 0..lat.len() {
            opt.step(step, &mut lat[i], &grads[leaf_ids[i]], &mut states[i]);
        }
    }

    let q = quantize(&lat, &m.shapes);
    let distilled_xent = loss::softmax_xent_forward(&fwd(m, &q), teacher_probs, SEQ, V)[0];
    (first, last, distilled_xent)
}

#[test]
fn end_to_end_distillation_recovers_a_deep_multi_block_ptq_gap() {
    // The KL floor is the teacher's own entropy; recovery is measured against the gap above it.
    let entropy =
        |m: &Model, probs: &[f32]| loss::softmax_xent_forward(&fwd(m, &m.lat), probs, SEQ, V)[0];

    // Per-depth PTQ gap above the teacher's entropy — informative only. NOTE: each depth is an
    // INDEPENDENT random-init toy, so per-model variance dominates and the gap need not grow
    // monotonically. The compounding *catastrophe* is a trained-real-scale effect (SmolLM2 ppl
    // 24→3.3M across 30 real layers, 0038 step 5); we don't assert it on tiny random nets.
    let mut ptq_gap_by_depth = Vec::new();
    for n in 1..=N_BLOCKS {
        let m = build(n);
        let tp = row_softmax(&fwd(&m, &m.lat));
        let h = entropy(&m, &tp);
        let ptq =
            loss::softmax_xent_forward(&fwd(&m, &quantize(&m.lat, &m.shapes)), &tp, SEQ, V)[0];
        ptq_gap_by_depth.push(ptq - h);
    }

    // Recovery at full depth: end-to-end distillation with the gradient flowing through ALL blocks.
    let m = build(N_BLOCKS);
    let tp = row_softmax(&fwd(&m, &m.lat));
    let h = entropy(&m, &tp);
    let ptq_xent =
        loss::softmax_xent_forward(&fwd(&m, &quantize(&m.lat, &m.shapes)), &tp, SEQ, V)[0];
    let (first, last, distilled_xent) = distill(&m, &tp, 2500);
    let (ptq_gap, dist_gap) = (ptq_xent - h, distilled_xent - h);
    let recovered = 100.0 * (1.0 - dist_gap / ptq_gap);

    println!(
        "0038b deep e2e (T={T}, {N_BLOCKS} blocks): teacher entropy {h:.4}. Per-depth PTQ gap (informative) = {ptq_gap_by_depth:?}. \
         At depth {N_BLOCKS}: PTQ xent {ptq_xent:.4} (gap {ptq_gap:.4}) → distilled {distilled_xent:.4} (gap {dist_gap:.4}) = {recovered:.1}% of the gap recovered; surrogate {first:.4}→{last:.4}."
    );

    // The distillation surrogate actually trained.
    assert!(
        last < first,
        "distillation KL must decrease: {first} → {last}"
    );
    // End-to-end distillation recovers the large majority (≥60%) of the deep PTQ gap by flowing the
    // gradient through ALL blocks jointly — the multi-layer recovery a local per-projection heal
    // (0038 step 5, which could NOT rescue model ppl) cannot deliver. Observed ~98%.
    assert!(
        dist_gap < 0.4 * ptq_gap,
        "end-to-end distillation must recover ≥60% of the deep PTQ gap: distilled gap {dist_gap:.4} vs PTQ gap {ptq_gap:.4}"
    );
}
