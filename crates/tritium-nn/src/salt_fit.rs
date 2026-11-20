//! Activation-aware SALT fitting: GPTQ error compensation, plus an optional discrete search over
//! the trit move set and closed-form step refits.
//!
//! The shipping converter rounds each group to the nearest ladder point, minimizing `‖W − Ŵ‖²`. The
//! loss does not see `W`; it sees `W·x`. Fitting in the activation metric `‖(W − Ŵ)·X‖²` — and, more
//! importantly, *acting* on its off-diagonal by pushing each quantized column's residual into the
//! columns not yet quantized — is worth **−0.49%** perplexity at identical bits on SmolLM2-135M, and
//! **−0.64%** with the discrete search and step refits on top.
//!
//! Everything here emits `(s₀, plane-major trits)` per group: the same shape
//! [`tritium_train::ops::ste::geometric_ladder_fit`] returns, so the existing packer consumes it
//! unchanged and the artifact is byte-for-byte the same container.
//!
//! # What the measurements say about using this
//!
//! - **Calibration size binds it.** GPTQ reaches 0.236× of round-to-nearest's objective on the Gram
//!   it fitted and only 0.459× on unseen tokens. Give it as many tokens as can be afforded.
//! - **The discrete search alone overfits** (−0.37%, worse than GPTQ). It only pays with
//!   [`ActivationAwareConfig::refit_scale`], where continuous parameters absorb what discrete moves
//!   chase.
//! - **A Gram narrower than the input width is rank-deficient by construction.** `down_proj` is 1536
//!   wide; a Gram from 1,024 tokens made GPTQ *lose* by 2.98%.

use rayon::prelude::*;
use tritium_train::Tape;
use tritium_train::tape::ValueId;

pub use tritium_quantize::engine::{
    ActivationAwareConfig, ScaleRefitMode, auto_decay, fit_tensor,
    fit_tensor_with_scale_refit_mode, fit_to_dense,
};

use crate::calibrate::{Arch, Tap, forward_aq};

/// Input Gram `E[x·xᵀ]` at each of the four projection inputs, per layer.
///
/// `q`, `k` and `v` share the attention-norm output; `gate` and `up` share the FFN-norm output;
/// `down` sees the FFN intermediate; `o` sees the concatenated attention heads. The tied head has no
/// Gram here — it is an embedding as well as a projection, so a fit that suits one corrupts the
/// other.
#[derive(Debug, Clone)]
pub struct TapGrams {
    /// Per layer, `n_embd × n_embd`.
    pub attn: Vec<Vec<f64>>,
    /// Per layer, `n_embd × n_embd`.
    pub ffn: Vec<Vec<f64>>,
    /// Per layer, `ff × ff`.
    pub down: Vec<Vec<f64>>,
    /// Per layer, `(n_head·head_dim)²`.
    pub o: Vec<Vec<f64>>,
    /// Tokens accumulated.
    pub rows: usize,
}

fn accumulate(h: &mut [f64], k: usize, act: &[f32], seq: usize) {
    h.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
        for r in 0..seq {
            let x = &act[r * k..(r + 1) * k];
            let xi = f64::from(x[i]);
            if xi == 0.0 {
                continue;
            }
            for (j, hij) in row.iter_mut().enumerate().skip(i) {
                *hij += xi * f64::from(x[j]);
            }
        }
    });
}

fn finish(h: &mut [f64], k: usize, rows: usize) {
    for i in 0..k {
        for j in 0..i {
            h[i * k + j] = h[j * k + i];
        }
    }
    let n = rows.max(1) as f64;
    for v in h.iter_mut() {
        *v /= n;
    }
}

impl TapGrams {
    /// Collect every tap's Gram from `weights`, one forward per window.
    ///
    /// The `O(seq·k²)` accumulation is the cost here, not the forwards, which is why it is parallel:
    /// the sample counts this fit needs are otherwise out of reach.
    #[must_use]
    pub fn collect(weights: &[Vec<f32>], arch: &Arch, windows: &[&[u32]]) -> Self {
        let q_width = arch.n_head * arch.head_dim;
        let mut out = Self {
            attn: vec![vec![0.0; arch.n_embd * arch.n_embd]; arch.n_layers],
            ffn: vec![vec![0.0; arch.n_embd * arch.n_embd]; arch.n_layers],
            down: vec![vec![0.0; arch.ff * arch.ff]; arch.n_layers],
            o: vec![vec![0.0; q_width * q_width]; arch.n_layers],
            rows: 0,
        };
        for tokens in windows {
            let mut tape = Tape::new();
            let wids: Vec<ValueId> = weights.iter().map(|w| tape.leaf(w.clone())).collect();
            forward_aq(
                &mut tape,
                &wids,
                arch,
                tokens,
                &mut |kind, li, v, seq, cols| match kind {
                    Tap::AttnIn => accumulate(&mut out.attn[li], cols, v, seq),
                    Tap::FfnIn => accumulate(&mut out.ffn[li], cols, v, seq),
                    Tap::DownIn => accumulate(&mut out.down[li], cols, v, seq),
                    Tap::OProjIn => accumulate(&mut out.o[li], cols, v, seq),
                    Tap::Head => {}
                },
            );
            out.rows += tokens.len();
        }
        for li in 0..arch.n_layers {
            finish(&mut out.attn[li], arch.n_embd, out.rows);
            finish(&mut out.ffn[li], arch.n_embd, out.rows);
            finish(&mut out.down[li], arch.ff, out.rows);
            finish(&mut out.o[li], q_width, out.rows);
        }
        out
    }

    /// The Gram for slot `slot` of layer `li`, in `extract`'s order: q, k, v, o, gate, up, down.
    #[must_use]
    pub fn for_slot(&self, li: usize, slot: usize) -> &[f64] {
        match slot {
            0..=2 => &self.attn[li],
            3 => &self.o[li],
            4..=5 => &self.ffn[li],
            _ => &self.down[li],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tritium_train::ops::ste::{self, RotationPolicy};

    /// The degenerate control. With `H = I` the compensation term vanishes — the Cholesky of a
    /// scaled identity is diagonal — so GPTQ reduces to rounding each column on the ladder's own
    /// grid, which is exactly what the plain fitter does. The codes must match it group for group.
    #[test]
    fn an_identity_gram_reproduces_the_plain_ladder_fit() {
        let (rows, cols, planes, group) = (3usize, 256usize, 3usize, 128usize);
        let mut s = 0x1357_9BDFu64;
        let w: Vec<f32> = (0..rows * cols)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / 8_388_608.0) - 1.0
            })
            .collect();
        let mut gram = vec![0.0f64; cols * cols];
        for i in 0..cols {
            gram[i * cols + i] = 1.0;
        }
        for rotate in [false, true] {
            let cfg = ActivationAwareConfig {
                planes,
                group,
                grid: 16,
                damp: 1e-9,
                search_sweeps: 0,
                refit_scale: false,
                rotate,
                decay: 1.0,
                decay_ramp: false,
            };
            let fits = fit_tensor(&w, rows, cols, &gram, &cfg).expect("fit");
            let oracle = ste::geometric_ladder_fit(
                &w,
                rows,
                cols,
                planes,
                group,
                16,
                if rotate {
                    RotationPolicy::Always
                } else {
                    RotationPolicy::Never
                },
            );
            assert_eq!(fits.len(), oracle.len());
            for (i, ((s_a, t_a), (s_b, t_b))) in fits.iter().zip(&oracle).enumerate() {
                assert!(
                    (s_a - s_b).abs() <= 1e-5 * s_b.abs().max(1e-6),
                    "rotate={rotate} group {i}: anchor {s_a} vs {s_b}"
                );
                assert_eq!(t_a, t_b, "rotate={rotate} group {i}: digits differ");
            }
        }
    }

    /// The search may never raise the objective it minimizes, and on a real (anisotropic) Gram it
    /// must find something to lower.
    #[test]
    fn the_search_lowers_the_activation_objective() {
        let (rows, cols, planes, group) = (4usize, 256usize, 3usize, 128usize);
        let mut s = 0x2468_ACE0u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f64 / 8_388_608.0) - 1.0
        };
        let w: Vec<f32> = (0..rows * cols).map(|_| next() as f32).collect();
        let m = 800;
        let x: Vec<f32> = (0..m * cols)
            .map(|i| (next() * (1.0 + 5.0 * f64::from(u8::from(i % cols % 9 == 0)))) as f32)
            .collect();
        let mut gram = vec![0.0f64; cols * cols];
        accumulate(&mut gram, cols, &x, m);
        finish(&mut gram, cols, m);
        let mut hd = gram.clone();
        let mean: f64 = (0..cols).map(|a| hd[a * cols + a]).sum::<f64>() / cols as f64;
        for a in 0..cols {
            hd[a * cols + a] += 0.01 * mean;
        }
        let objective = |q: &[f32]| -> f64 {
            (0..rows)
                .map(|r| {
                    let e: Vec<f64> = (0..cols)
                        .map(|j| f64::from(w[r * cols + j] - q[r * cols + j]))
                        .collect();
                    (0..cols)
                        .map(|a| e[a] * (0..cols).map(|b| hd[a * cols + b] * e[b]).sum::<f64>())
                        .sum::<f64>()
                })
                .sum()
        };
        let base = ActivationAwareConfig {
            planes,
            group,
            grid: 16,
            damp: 0.01,
            search_sweeps: 0,
            refit_scale: false,
            rotate: false,
            decay: 1.0,
            decay_ramp: false,
        };
        let gptq = fit_to_dense(
            &fit_tensor(&w, rows, cols, &gram, &base).unwrap(),
            rows,
            cols,
            group,
            false,
        );
        let searched = fit_to_dense(
            &fit_tensor(
                &w,
                rows,
                cols,
                &gram,
                &ActivationAwareConfig {
                    search_sweeps: 8,
                    refit_scale: true,
                    ..base
                },
            )
            .unwrap(),
            rows,
            cols,
            group,
            false,
        );
        let (a, b) = (objective(&gptq), objective(&searched));
        println!("activation objective: gptq {a:.6e} search+refit {b:.6e}");
        assert!(
            b < a,
            "the search did not lower the objective: {b:.6e} vs {a:.6e}"
        );
    }

    #[test]
    fn in_loop_scale_refit_changes_the_sequential_candidate_and_keeps_ladder_codes() {
        let weights = [0.91, -0.34, 0.72, -0.58];
        let gram = [
            1.0, 0.72, 0.11, -0.08, 0.72, 1.0, 0.19, -0.04, 0.11, 0.19, 1.0, 0.61, -0.08, -0.04,
            0.61, 1.0,
        ];
        let base = ActivationAwareConfig {
            planes: 2,
            group: 2,
            grid: 16,
            damp: 0.01,
            search_sweeps: 0,
            refit_scale: true,
            rotate: false,
            decay: 1.0,
            decay_ramp: false,
        };
        let post_pass = fit_tensor_with_scale_refit_mode(
            &weights,
            1,
            4,
            &gram,
            &base,
            ScaleRefitMode::PostPass,
        )
        .expect("post-pass control");
        let in_loop =
            fit_tensor_with_scale_refit_mode(&weights, 1, 4, &gram, &base, ScaleRefitMode::InLoop)
                .expect("in-loop fit");

        assert_ne!(
            in_loop, post_pass,
            "the in-loop arm must execute scale refits"
        );
        assert_eq!(in_loop.len(), 2);
        for (anchor, planes) in &in_loop {
            assert!(anchor.is_finite() && *anchor >= 0.0);
            assert_eq!(planes.len(), 2);
            for (&high, &low) in planes[0].iter().zip(&planes[1]) {
                let code = high as i32 * 3 + low as i32;
                assert!((-4..=4).contains(&code), "invalid two-trit code {code}");
            }
        }
    }

    /// A tap that saw no signal must degrade to the plain ladder, not to a wrong fit: with an
    /// all-zero Gram the damping leaves a multiple of the identity, whose compensation term is zero.
    /// A Gram with non-finite entries is a different matter and must be refused.
    #[test]
    fn a_dead_gram_degrades_to_the_ladder_and_a_broken_one_is_refused() {
        let (rows, cols, planes, group) = (2usize, 256usize, 3usize, 128usize);
        let mut s = 0xFACE_B00Cu64;
        let w: Vec<f32> = (0..rows * cols)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / 8_388_608.0) - 1.0
            })
            .collect();
        let cfg = ActivationAwareConfig {
            planes,
            group,
            grid: 16,
            damp: 0.01,
            search_sweeps: 0,
            refit_scale: false,
            rotate: false,
            decay: 1.0,
            decay_ramp: false,
        };
        let dead = fit_tensor(&w, rows, cols, &vec![0.0f64; cols * cols], &cfg).expect("dead gram");
        let oracle =
            ste::geometric_ladder_fit(&w, rows, cols, planes, group, 16, RotationPolicy::Never);
        for (i, ((s_a, t_a), (s_b, t_b))) in dead.iter().zip(&oracle).enumerate() {
            assert!(
                (s_a - s_b).abs() <= 1e-5 * s_b.abs().max(1e-6),
                "group {i} anchor"
            );
            assert_eq!(t_a, t_b, "group {i} digits");
        }
        let mut broken = vec![0.0f64; cols * cols];
        broken[0] = f64::NAN;
        assert!(fit_tensor(&w, rows, cols, &broken, &cfg).is_none());
    }
}

#[cfg(test)]
mod decay_tests {
    use super::*;

    fn synthetic(rows: usize, cols: usize) -> (Vec<f32>, Vec<f64>) {
        // Deterministic heavy-tailed weights and a full-rank, well-conditioned Gram.
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let x = ((i * 7919) % 1000) as f32 / 500.0 - 1.0;
                x * x * x + 0.05 * x
            })
            .collect();
        let mut gram = vec![0.0f64; cols * cols];
        for a in 0..cols {
            for b in 0..cols {
                gram[a * cols + b] =
                    0.3f64.powi((a as i32 - b as i32).abs()) * (1.0 + a as f64 / cols as f64);
            }
        }
        (w, gram)
    }

    #[test]
    fn a_decay_of_one_is_plain_gptq_with_or_without_the_ramp() {
        let (w, gram) = synthetic(4, 128);
        let base = ActivationAwareConfig {
            planes: 2,
            group: 64,
            grid: 8,
            search_sweeps: 0,
            rotate: false,
            ..ActivationAwareConfig::default()
        };
        let plain = fit_tensor(&w, 4, 128, &gram, &base).expect("fit");
        let ramped = fit_tensor(
            &w,
            4,
            128,
            &gram,
            &ActivationAwareConfig {
                decay_ramp: true,
                ..base
            },
        )
        .expect("fit");
        assert_eq!(plain, ramped, "at decay 1.0 the ramp must change nothing");
    }

    #[test]
    fn a_decay_below_one_changes_the_propagation() {
        let (w, gram) = synthetic(4, 128);
        let base = ActivationAwareConfig {
            planes: 2,
            group: 64,
            grid: 8,
            search_sweeps: 0,
            rotate: false,
            ..ActivationAwareConfig::default()
        };
        let plain = fit_tensor(&w, 4, 128, &gram, &base).expect("fit");
        let decayed = fit_tensor(
            &w,
            4,
            128,
            &gram,
            &ActivationAwareConfig { decay: 0.5, ..base },
        )
        .expect("fit");
        assert_ne!(
            plain, decayed,
            "decay 0.5 must alter the codes GPTQ produces"
        );
    }
}
