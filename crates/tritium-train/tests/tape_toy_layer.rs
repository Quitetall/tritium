//! Gate C (ADR 0007), graph level: the reverse-mode tape reproduces the analytic
//! gradient of a composed QAT toy layer, validated element-by-element against the
//! central finite difference. Also isolates fan-out gradient accumulation.

use tritium_testkit::Tolerance;
use tritium_train::ops::{act, bias, elementwise, loss, matmul, ste};
use tritium_train::tape::Tape;

fn gate_c_tol() -> Tolerance {
    Tolerance::relative(2e-3)
}

#[test]
fn tape_accumulates_fanout_gradient() {
    // x fans out into add(x, x) = 2x, then mse(2x, target). A value feeding two
    // consumers must have its grads summed: dloss/dx_i = (2/N)(2x_i - t_i)·2.
    let x = vec![0.5f32, -1.0, 2.0, 0.25];
    let target = vec![0.1f32, 0.2, -0.3, 0.4];
    let n = x.len() as f32;
    let mut tape = Tape::new();
    let xid = tape.leaf(x.clone());
    let tid = tape.leaf(target.clone());
    let d = tape.add(xid, xid); // same leaf twice -> fan-out
    let l = tape.mse(d, tid);
    let grads = tape.backward(l);
    let tol = gate_c_tol();
    for i in 0..x.len() {
        let want = (2.0 / n) * (2.0 * x[i] - target[i]) * 2.0;
        assert!(
            tol.accepts(grads[xid][i], want),
            "x[{i}]: got {} want {want}",
            grads[xid][i]
        );
    }
}

// Toy QAT layer dims. Wf:[N,K] weights, act:[M,K], s/s_q/b:[N], target:[M,N].
const M: usize = 2;
const N: usize = 3;
const K: usize = 4;

// Differentiable forward (STE *surrogate* in the quantizer slot) -> scalar loss.
fn composed_loss(
    wf: &[f32],
    s_q: &[f32],
    act_: &[f32],
    s: &[f32],
    b: &[f32],
    target: &[f32],
) -> f32 {
    let t = ste::quantize_surrogate(wf, s_q, N, K);
    let y = matmul::forward(act_, &t, s, M, N, K);
    let z = bias::forward(&y, b, M, N);
    let h = act::relu2_forward(&z);
    let g = elementwise::add_forward(&h, &h); // fan-out: h feeds add twice
    loss::mse_forward(&g, target)[0]
}

// Same graph on the tape; returns analytic grads for the 6 leaves in input order.
fn composed_grads(
    wf: &[f32],
    s_q: &[f32],
    act_: &[f32],
    s: &[f32],
    b: &[f32],
    target: &[f32],
) -> Vec<Vec<f32>> {
    let mut tape = Tape::new();
    let wf_id = tape.leaf(wf.to_vec());
    let sq_id = tape.leaf(s_q.to_vec());
    let act_id = tape.leaf(act_.to_vec());
    let s_id = tape.leaf(s.to_vec());
    let b_id = tape.leaf(b.to_vec());
    let tg_id = tape.leaf(target.to_vec());
    let t = tape.ste_surrogate(wf_id, sq_id, N, K);
    let y = tape.matmul(act_id, t, s_id, M, N, K);
    let z = tape.bias(y, b_id, M, N);
    let h = tape.relu2(z);
    let g = tape.add(h, h);
    let l = tape.mse(g, tg_id);
    let grads = tape.backward(l);
    vec![
        grads[wf_id].clone(),
        grads[sq_id].clone(),
        grads[act_id].clone(),
        grads[s_id].clone(),
        grads[b_id].clone(),
        grads[tg_id].clone(),
    ]
}

#[test]
fn tape_toy_layer_end_to_end_gradient() {
    // Wf magnitudes kept off the clamp kink |Wf/s_q| = 1; s_q = per-row AbsMean (frozen).
    #[rustfmt::skip]
    let wf: Vec<f32> = vec![
         0.2, -0.4,  1.5, -1.8, // s_q=0.975
        -0.3,  0.5, -1.6,  0.1, // s_q=0.625
         0.4, -0.5,  1.7, -0.2, // s_q=0.700
    ];
    let s_q = ste::absmean_scale_per_row(&wf, N, K);
    // Small magnitudes keep the composed loss (4th-order in z via MSE∘relu²) tame so the
    // central-difference truncation error stays far under the Gate-C bar.
    #[rustfmt::skip]
    let act_: Vec<f32> = vec![
        0.4, -0.2,  0.3, -0.5,
        0.1,  0.3, -0.2,  0.4,
    ];
    let s = vec![0.7f32, 0.5, 0.6]; // matmul scale: a free leaf, independent of s_q
    let b = vec![0.05f32, -0.1, 0.2];
    let target = vec![0.3f32, -0.2, 0.5, 0.1, 0.4, -0.3];

    // Base leaves (input order: wf, s_q, act, s, b, target).
    let base = vec![wf, s_q, act_, s, b, target];
    let analytic = composed_grads(&base[0], &base[1], &base[2], &base[3], &base[4], &base[5]);

    // Compare the tape's seed-1 grads DIRECTLY to per-element central differences — no
    // cotangent indirection, so the check cannot be masked by a lucky scalar cotangent.
    let h = 1e-3f32;
    let tol = gate_c_tol();
    // Wf(0), act(2), matmul-scale(3), bias(4). s_q(1) is stop-gradient; target(5) is data.
    for &leaf in &[0usize, 2, 3, 4] {
        for i in 0..base[leaf].len() {
            let mut lv = base.clone();
            lv[leaf][i] += h;
            let lp = composed_loss(&lv[0], &lv[1], &lv[2], &lv[3], &lv[4], &lv[5]);
            lv[leaf][i] -= 2.0 * h;
            let lm = composed_loss(&lv[0], &lv[1], &lv[2], &lv[3], &lv[4], &lv[5]);
            let numeric = (lp - lm) / (2.0 * h);
            let a = analytic[leaf][i];
            assert!(
                tol.accepts(a, numeric),
                "leaf {leaf}[{i}]: analytic {a} vs numeric {numeric}"
            );
        }
    }
}
