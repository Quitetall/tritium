//! SALT reconstruction-error vs bits-per-weight on a **normally-trained fp model** (gpt2),
//! the validation BitNet b1.58 cannot give.
//!
//! SALT's premise (ADR 0001): more planes ⇒ lower reconstruction error ⇒ a smooth
//! ~1.58→3 bpw accuracy curve. On b1.58 the bf16 "master" is *latent* QAT weights, so the
//! curve inverts (higher bpw regresses toward unusable weights — see `tritium-nn`'s
//! `salt_accuracy`). A normal fp model's weights ARE the target, so SALT's residual planes
//! genuinely improve fidelity: this gate quantizes every 2D weight tensor of gpt2 at a bpw
//! sweep (per-256-block, `BaseScaleScope::Block`), dequantizes, and asserts the aggregate
//! relative reconstruction error `‖W − Ŵ‖ / ‖W‖` **decreases monotonically** with bpw.
//!
//! Arch-agnostic (no forward), so it works on any fp16/f32 safetensors. `#[ignore]`d and
//! skips cleanly when the model is absent. Run:
//! ```text
//! cargo test -p tritium-quantize --release --test recon_curve -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use tritium_format::{SafeTensors, dequant_salt_row};
use tritium_quantize::{BaseScaleScope, QuantConfig, Sensitivity, quantize_tensor};

/// gpt2 `model.safetensors` under the HF hub cache (a small, normal fp model). `None` if
/// absent (the test then skips).
fn gpt2_safetensors() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home).join(".cache/huggingface/hub/models--gpt2/snapshots");
    let snap = std::fs::read_dir(&snaps)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.is_dir())?;
    let st = snap.join("model.safetensors");
    st.exists().then_some(st)
}

/// Aggregate relative reconstruction error of SALT at `bpw` over every 2D weight tensor:
/// `sqrt(Σ‖W−Ŵ‖² / Σ‖W‖²)`.
fn recon_error_at(st: &SafeTensors, names: &[String], bpw: f64) -> f64 {
    let cfg = QuantConfig {
        budget_bpw: bpw,
        t_min: 1,
        t_max: 3,
        sensitivity: Sensitivity::Uniform,
        scale_group: BaseScaleScope::Block,
    };
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for name in names {
        let shape = st.shape(name).expect("shape");
        let (rows, k) = (shape[0], shape[1]);
        let w = st.tensor_f32(name).expect("tensor");
        let qt = quantize_tensor(&w, rows, k, &cfg).expect("quantize_tensor");
        for (r, row) in qt.salt_rows.iter().enumerate() {
            let wq = dequant_salt_row(row).expect("dequant");
            for i in 0..k {
                let orig = f64::from(w[r * k + i]);
                let diff = orig - f64::from(wq[i]);
                num += diff * diff;
                den += orig * orig;
            }
        }
    }
    (num / den).sqrt()
}

#[test]
#[ignore = "loads gpt2 safetensors; SALT recon-error vs bpw on a normal fp model"]
fn salt_recon_error_decreases_with_bpw() {
    let Some(path) = gpt2_safetensors() else {
        eprintln!("skipping: gpt2 model.safetensors absent");
        return;
    };
    let bytes = std::fs::read(&path).expect("read gpt2");
    let st = SafeTensors::parse(&bytes).expect("parse safetensors");

    // Every 2D weight tensor (the Linear/embedding matrices); skip 1D norms/biases.
    let names: Vec<String> = st
        .names()
        .filter(|n| {
            st.shape(n)
                .is_some_and(|s| s.len() == 2 && s[0] >= 2 && s[1] >= 2)
        })
        .map(str::to_owned)
        .collect();
    assert!(!names.is_empty(), "no 2D weight tensors found");

    let bpws = [tritium_quantize::TRIT_BITS, 2.0, 2.6, 3.0];
    println!(
        "\nSALT recon-error vs bpw (gpt2, {} weight tensors):",
        names.len()
    );
    println!("  {:>8}  {:>14}", "bpw", "rel recon err");
    let mut prev = f64::INFINITY;
    for &bpw in &bpws {
        let err = recon_error_at(&st, &names, bpw);
        println!("  {bpw:>8.3}  {err:>14.5}");
        // SALT's claim: each extra plane only reduces residual, so error is monotone
        // non-increasing in bpw. (Small float slack for the f16-rounded block scales.)
        assert!(
            err <= prev + 1e-6,
            "recon error rose with bpw: {err} > {prev} at {bpw} bpw"
        );
        prev = err;
    }
    // The floor (all T=1) must be a real lossy ternary error, and the top of the sweep must
    // beat it meaningfully — i.e. residual planes actually buy fidelity on a normal model.
    let floor = recon_error_at(&st, &names, tritium_quantize::TRIT_BITS);
    let top = recon_error_at(&st, &names, 3.0);
    assert!(
        floor > 0.05,
        "T=1 ternary should be visibly lossy, got {floor}"
    );
    assert!(
        top < floor * 0.8,
        "3 bpw should cut recon error >20% vs the floor ({top} vs {floor})"
    );
}
