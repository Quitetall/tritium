//! Multi-head + GQA attention on the tape (plan 0040 step 2): the `nn::attention` helper must match
//! an independent reference scaled-dot-product-attention forward, and its gradient (composed from
//! the individually-gradchecked ops) must match a finite difference end-to-end.

use tritium_train::Tape;
use tritium_train::nn::{attention, attention_heads};
use tritium_train::ops::{dense, rope, shape, softmax};

// Real GQA shape: 4 query heads share 2 KV heads (group size 2).
const SEQ: usize = 3;
const N_EMBD: usize = 8;
const N_HEAD: usize = 4;
const N_KV_HEAD: usize = 2;
const HEAD_DIM: usize = 2;
const THETA: f32 = 10_000.0;
const QD: usize = N_HEAD * HEAD_DIM;
const KVD: usize = N_KV_HEAD * HEAD_DIM;

fn seeded(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 1000) as f32 / 500.0 - 1.0 // [-1, 1)
        })
        .collect()
}

/// Independent reference: multi-head GQA causal SDPA, mirroring `nn::attention` via the free ops.
fn ref_attention(x: &[f32], wq: &[f32], wk: &[f32], wv: &[f32], wo: &[f32]) -> Vec<f32> {
    let group = N_HEAD / N_KV_HEAD;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let pos: Vec<usize> = (0..SEQ).collect();

    let q = rope::forward(
        &dense::forward(x, wq, SEQ, QD, N_EMBD),
        &pos,
        N_HEAD,
        HEAD_DIM,
        THETA,
    );
    let k = rope::forward(
        &dense::forward(x, wk, SEQ, KVD, N_EMBD),
        &pos,
        N_KV_HEAD,
        HEAD_DIM,
        THETA,
    );
    let v = dense::forward(x, wv, SEQ, KVD, N_EMBD);

    let mut heads: Vec<Vec<f32>> = Vec::with_capacity(N_HEAD);
    for h in 0..N_HEAD {
        let kv = h / group;
        let qh = shape::slice_cols_forward(&q, SEQ, QD, h * HEAD_DIM, HEAD_DIM);
        let kh = shape::slice_cols_forward(&k, SEQ, KVD, kv * HEAD_DIM, HEAD_DIM);
        let vh = shape::slice_cols_forward(&v, SEQ, KVD, kv * HEAD_DIM, HEAD_DIM);
        let mut sc = dense::forward(&qh, &kh, SEQ, SEQ, HEAD_DIM);
        for s in &mut sc {
            *s *= scale;
        }
        let sc = softmax::causal_mask_forward(&sc, SEQ, SEQ);
        let p = softmax::forward(&sc, SEQ, SEQ);
        let vt = dense::transpose_forward(&vh, SEQ, HEAD_DIM);
        heads.push(dense::forward(&p, &vt, SEQ, HEAD_DIM, SEQ));
    }
    let refs: Vec<&[f32]> = heads.iter().map(Vec::as_slice).collect();
    let cat = shape::concat_cols_forward(&refs, SEQ, &[HEAD_DIM; N_HEAD]);
    dense::forward(&cat, wo, SEQ, N_EMBD, QD)
}

/// `(x, wq, wk, wv, wo)`.
type Weights = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

fn weights() -> Weights {
    (
        seeded(1, SEQ * N_EMBD), // x
        seeded(2, QD * N_EMBD),  // wq
        seeded(3, KVD * N_EMBD), // wk
        seeded(4, KVD * N_EMBD), // wv
        seeded(5, N_EMBD * QD),  // wo
    )
}

#[test]
fn tape_attention_matches_reference_gqa_sdpa() {
    let (x, wq, wk, wv, wo) = weights();
    let reference = ref_attention(&x, &wq, &wk, &wv, &wo);

    let mut t = Tape::new();
    let (xid, wqid, wkid, wvid, woid) = (
        t.leaf(x.clone()),
        t.leaf(wq.clone()),
        t.leaf(wk.clone()),
        t.leaf(wv.clone()),
        t.leaf(wo.clone()),
    );
    let out = attention(
        &mut t, xid, wqid, wkid, wvid, woid, SEQ, N_EMBD, N_HEAD, N_KV_HEAD, HEAD_DIM, THETA,
    );
    let got = t.value(out);
    assert_eq!(got.len(), SEQ * N_EMBD);
    let max = got
        .iter()
        .zip(&reference)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max < 1e-5,
        "tape attention vs reference SDPA max abs diff {max}"
    );
}

#[test]
fn tape_attention_gradient_matches_finite_difference() {
    let (x, wq, wk, wv, wo) = weights();

    // Scalar loss L = Σ out·r for a fixed random cotangent r (a dot-product = matmul to [1,1]).
    let r = seeded(9, SEQ * N_EMBD);
    // Loss as a function of ALL five inputs, so we can perturb any one of them.
    let loss_of = |x: &[f32], wq: &[f32], wk: &[f32], wv: &[f32], wo: &[f32]| -> f64 {
        ref_attention(x, wq, wk, wv, wo)
            .iter()
            .zip(&r)
            .map(|(&y, &ri)| f64::from(y) * f64::from(ri))
            .sum()
    };

    let mut t = Tape::new();
    let (xid, wqid, wkid, wvid, woid) = (
        t.leaf(x.clone()),
        t.leaf(wq.clone()),
        t.leaf(wk.clone()),
        t.leaf(wv.clone()),
        t.leaf(wo.clone()),
    );
    let out = attention(
        &mut t, xid, wqid, wkid, wvid, woid, SEQ, N_EMBD, N_HEAD, N_KV_HEAD, HEAD_DIM, THETA,
    );
    let rid = t.leaf(r.clone());
    let scalar = t.dense_matmul(out, rid, 1, 1, SEQ * N_EMBD); // Σ out·r
    let grads = t.backward(scalar);

    // Probe EVERY input (x + all four projections — wk/wv carry the GQA-shared paths, x carries the
    // multi-slice accumulation into q/k/v). A broken vjp or accumulate on any path fails here.
    let h = 1e-3f64;
    for (name, base, id) in [
        ("x", &x, xid),
        ("wq", &wq, wqid),
        ("wk", &wk, wkid),
        ("wv", &wv, wvid),
        ("wo", &wo, woid),
    ] {
        let n = base.len();
        for &i in &[0usize, n / 3, n - 1] {
            let (mut plus, mut minus) = (base.clone(), base.clone());
            plus[i] += h as f32;
            minus[i] -= h as f32;
            let numeric = match name {
                "x" => loss_of(&plus, &wq, &wk, &wv, &wo) - loss_of(&minus, &wq, &wk, &wv, &wo),
                "wq" => loss_of(&x, &plus, &wk, &wv, &wo) - loss_of(&x, &minus, &wk, &wv, &wo),
                "wk" => loss_of(&x, &wq, &plus, &wv, &wo) - loss_of(&x, &wq, &minus, &wv, &wo),
                "wv" => loss_of(&x, &wq, &wk, &plus, &wo) - loss_of(&x, &wq, &wk, &minus, &wo),
                _ => loss_of(&x, &wq, &wk, &wv, &plus) - loss_of(&x, &wq, &wk, &wv, &minus),
            } / (2.0 * h);
            let analytic = f64::from(grads[id][i]);
            let denom = numeric.abs().max(1.0);
            assert!(
                ((analytic - numeric) / denom).abs() < 3e-3,
                "dL/d{name}[{i}]: analytic {analytic} vs numeric {numeric}"
            );
        }
    }
}

/// The refactor control: `attention` must be `attention_heads` plus one matmul and nothing else.
///
/// `attention_heads` was split out so an activation-precision sweep can reach `o_proj`'s input —
/// the one projection input a block used to hide. Every published tape number flows through
/// `attention`, so equivalence is asserted bit-for-bit rather than assumed: the two paths run the
/// same ops in the same order, and any drift here means the split changed the model.
#[test]
fn attention_is_attention_heads_plus_the_output_projection() {
    let x = seeded(1, SEQ * N_EMBD);
    let wq = seeded(2, QD * N_EMBD);
    let wk = seeded(3, KVD * N_EMBD);
    let wv = seeded(4, KVD * N_EMBD);
    let wo = seeded(5, N_EMBD * QD);

    let whole = {
        let mut t = Tape::new();
        let (xi, q, k, v, o) = (
            t.leaf(x.clone()),
            t.leaf(wq.clone()),
            t.leaf(wk.clone()),
            t.leaf(wv.clone()),
            t.leaf(wo.clone()),
        );
        let out = attention(
            &mut t, xi, q, k, v, o, SEQ, N_EMBD, N_HEAD, N_KV_HEAD, HEAD_DIM, THETA,
        );
        t.value(out).to_vec()
    };

    let split = {
        let mut t = Tape::new();
        let (xi, q, k, v, o) = (t.leaf(x), t.leaf(wq), t.leaf(wk), t.leaf(wv), t.leaf(wo));
        let heads = attention_heads(
            &mut t, xi, q, k, v, SEQ, N_EMBD, N_HEAD, N_KV_HEAD, HEAD_DIM, THETA,
        );
        assert_eq!(
            t.value(heads).len(),
            SEQ * QD,
            "the head outputs are o_proj's input, so they must be [seq, n_head * head_dim]"
        );
        let out = t.dense_matmul(heads, o, SEQ, N_EMBD, QD);
        t.value(out).to_vec()
    };

    assert_eq!(whole.len(), split.len());
    for (i, (a, b)) in whole.iter().zip(&split).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "index {i}: whole {a} vs split {b} — the split must be exact, not merely close"
        );
    }
}
