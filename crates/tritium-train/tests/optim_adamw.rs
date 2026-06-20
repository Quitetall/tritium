//! AdamW update correctness (closed-form + decoupled weight decay) and the
//! conformance slice that is CPU-testable: no NaN/Inf over ≥1k steps and loss
//! decreases (ADR 0007, plan 0008).

mod common;

use tritium_train::optim::{AdamW, Optimizer};

#[test]
fn adamw_single_step_closed_form() {
    // t=1, wd=0: m̂ = g, v̂ = g² ⇒ w ← w − lr·g/(|g| + eps). (g=0 ⇒ no move.)
    let lr = 0.01f32;
    let eps = 1e-8f32;
    let opt = AdamW {
        lr,
        beta1: 0.9,
        beta2: 0.999,
        eps,
        weight_decay: 0.0,
    };
    let orig = [1.0f32, -2.0, 0.5, 0.0];
    let grad = [0.5f32, -1.0, 2.0, 0.0];
    let mut param = orig.to_vec();
    let mut st = opt.init_state(param.len());
    opt.step(1, &mut param, &grad, &mut st);
    for i in 0..param.len() {
        let g = grad[i];
        let want = orig[i] - lr * g / (g.abs() + eps);
        assert!(
            (param[i] - want).abs() < 1e-6,
            "param[{i}] = {} want {want}",
            param[i]
        );
    }
}

#[test]
fn adamw_decoupled_weight_decay() {
    // wd>0 adds an extra ·(1 − lr·wd) factor on the param that does NOT pass through
    // the adaptive denominator: w ← w·(1 − lr·wd) − lr·g/(|g| + eps) at t=1.
    let lr = 0.1f32;
    let wd = 0.05f32;
    let eps = 1e-8f32;
    let opt = AdamW {
        lr,
        beta1: 0.9,
        beta2: 0.999,
        eps,
        weight_decay: wd,
    };
    let orig = [1.0f32, -2.0, 3.0];
    let grad = [0.5f32, 0.5, -0.5];
    let mut param = orig.to_vec();
    let mut st = opt.init_state(param.len());
    opt.step(1, &mut param, &grad, &mut st);
    for i in 0..param.len() {
        let g = grad[i];
        let want = orig[i] * (1.0 - lr * wd) - lr * g / (g.abs() + eps);
        assert!(
            (param[i] - want).abs() < 1e-6,
            "param[{i}] = {} want {want}",
            param[i]
        );
    }
}

/// Independent f64 reference for one AdamW step, with `eps` EXPLICITLY outside the
/// square root (`√v̂ + eps`) — the canonical Adam/AdamW denominator. Hand-derived here
/// so the comparison is not a copy of the implementation.
#[allow(clippy::too_many_arguments)]
fn adamw_ref_step(
    t: i32,
    w: &mut [f64],
    g: &[f64],
    m: &mut [f64],
    v: &mut [f64],
    lr: f64,
    b1: f64,
    b2: f64,
    eps: f64,
    wd: f64,
) {
    let bc1 = 1.0 - b1.powi(t);
    let bc2 = 1.0 - b2.powi(t);
    for i in 0..w.len() {
        m[i] = b1 * m[i] + (1.0 - b1) * g[i];
        v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
        let m_hat = m[i] / bc1;
        let v_hat = v[i] / bc2;
        w[i] = w[i] * (1.0 - lr * wd) - lr * (m_hat / (v_hat.sqrt() + eps));
    }
}

#[test]
fn adamw_matches_f64_reference_over_multiple_steps() {
    // The t=1-from-zero closed form cannot see eps placement: bias correction cancels
    // the (1−β) factor, and at eps=1e-8 the inside-vs-outside difference is ~1e-11, far
    // under tolerance. Here we run several steps with accumulated m/v, distinct betas,
    // and a LARGE eps so √v̂ + eps differs from √(v̂ + eps) well above tolerance —
    // pinning eps-outside, the bias-correction divisors at t≥2, the new-m-used-for-m̂
    // ordering, and the β1/β2 roles against an independent f64 reference.
    let opt = AdamW {
        lr: 0.1,
        beta1: 0.8,
        beta2: 0.9,
        eps: 0.5, // large on purpose: makes eps-inside-vs-outside-sqrt observable
        weight_decay: 0.03,
    };
    // Small grads so √v̂ ≈ eps and the placement matters; distinct per step so m/v
    // genuinely accumulate.
    let grads: [[f32; 3]; 4] = [
        [0.10, -0.20, 0.05],
        [-0.15, 0.10, -0.25],
        [0.20, 0.05, -0.10],
        [-0.05, -0.10, 0.15],
    ];
    let mut param = vec![0.5f32, -0.3, 0.8];
    let mut st = opt.init_state(param.len());

    let mut wf: Vec<f64> = param.iter().map(|&x| x as f64).collect();
    let mut mf = vec![0.0f64; param.len()];
    let mut vf = vec![0.0f64; param.len()];

    for (k, g) in grads.iter().enumerate() {
        let t = (k + 1) as u64;
        opt.step(t, &mut param, g, &mut st);
        let gf: Vec<f64> = g.iter().map(|&x| x as f64).collect();
        adamw_ref_step(
            t as i32,
            &mut wf,
            &gf,
            &mut mf,
            &mut vf,
            opt.lr as f64,
            opt.beta1 as f64,
            opt.beta2 as f64,
            opt.eps as f64,
            opt.weight_decay as f64,
        );
        for i in 0..param.len() {
            assert!(
                (param[i] as f64 - wf[i]).abs() < 1e-5,
                "step {t} param[{i}]: f32 impl {} vs f64 ref {}",
                param[i],
                wf[i]
            );
        }
    }
}

#[test]
fn adamw_no_nan_and_loss_decreases_over_1k_steps() {
    let opt = AdamW {
        lr: 5e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
    };
    let s_q = common::linear_region_s_q();
    let act = common::seeded(10, common::M * common::K, -1.0, 1.0);
    let s = vec![0.7f32, 0.5, 0.6];

    // A reachable target: forward at a teacher (Wf*, b*), so min loss is ~0.
    let wf_teacher = common::seeded(11, common::N * common::K, -1.0, 1.0);
    let b_teacher = common::seeded(12, common::N, -0.2, 0.2);
    let d = common::ToyData {
        target: common::forward(&wf_teacher, &s_q, &act, &s, &b_teacher),
        s_q,
        act,
        s,
    };

    // Start off the teacher by a fixed perturbation.
    let nw = common::seeded(13, common::N * common::K, -1.0, 1.0);
    let nb = common::seeded(14, common::N, -1.0, 1.0);
    let mut p = common::ToyParams {
        wf: wf_teacher
            .iter()
            .zip(&nw)
            .map(|(&w, &n)| w + 0.3 * n)
            .collect(),
        b: b_teacher
            .iter()
            .zip(&nb)
            .map(|(&w, &n)| w + 0.3 * n)
            .collect(),
    };
    let mut st = common::ToyState::init(&opt);

    let initial = d.loss(&p);
    assert!(
        initial.is_finite() && initial > 0.0,
        "bad initial loss {initial}"
    );

    for t in 1..=1200u64 {
        let loss = common::train_step(t, &opt, &mut p, &mut st, &d);
        assert!(loss.is_finite(), "loss non-finite at step {t}: {loss}");
        assert!(
            p.wf.iter().all(|x| x.is_finite()),
            "wf non-finite at step {t}"
        );
        assert!(
            p.b.iter().all(|x| x.is_finite()),
            "b non-finite at step {t}"
        );
    }

    let final_loss = d.loss(&p);
    assert!(
        final_loss.is_finite(),
        "final loss non-finite: {final_loss}"
    );
    assert!(
        final_loss < initial,
        "loss did not decrease: {initial} -> {final_loss}"
    );
    // Convex linear-least-squares with a reachable optimum: 1200 steps drives it near 0.
    assert!(
        final_loss < initial * 1e-2,
        "weak convergence: {initial} -> {final_loss}"
    );
}
