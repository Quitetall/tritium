//! GPU QAT training step + tiny-model pretrain smoke (plan 0013).
//!
//! This is the first real consumer of the `train_grad.cu` kernels (`ternary_matmul_forward` +
//! `grad_a`/`grad_w`/`grad_s`): it composes them — together with the reused `tritium-train` CPU ops
//! (STE-quantize, squared-ReLU, MSE) and `AdamW` + the [`LrSchedule`](tritium_train::LrSchedule) —
//! into a converging GPU training step for a tiny from-scratch ternary MLP.
//!
//! **Model.** A 2-layer ternary MLP: `x[M,K₀] →[Wq₁,s₁] h_pre[M,N₁] → relu² → [Wq₂,s₂] y[M,N₂]`,
//! MSE against a fixed synthetic target. The matmul (forward + the three gradients) runs on the GPU;
//! the elementwise glue (quantize, relu², loss) is host-side and tiny.
//!
//! **QAT scheme.** Master weights are f32, kept off the 1-bit grid at init to avoid the STE-freeze
//! (plan 0010 + the bitnet-qat scars). The per-row scale is a **learned output gain** on
//! `y = s·(A·Wqᵀ)` — *not* a learned quantizer step size: unlike LSQ (Esser et al. 2020), the
//! quantizer's dependence on `s` is stop-gradiented (`ste::quantize_vjp` returns an all-zero
//! `g_sq`), so `s` is trained *only* by `grad_s` (its gradient from the explicit output scaling).
//! This is a deliberate choice — distinct from the inference-time AbsMean scale of b1.58 — so the
//! optimizer actually exercises `grad_s`. The weight master is updated through the standard
//! straight-through estimator ([`ste::quantize_vjp`](tritium_train::ops::ste::quantize_vjp),
//! `1/s·1[|W/s|<1]`).
//!
//! **Gates.** (1) the composed step's gradients match the `tritium-train` CPU tape vjp within `1e-4`
//! (device == CPU); (2) the loss falls well below its start over the step budget with no NaN
//! (pretrain smoke). Activation-retention / dropping the per-call htod↔dtoh is a perf refinement that
//! is immaterial at this scale (measured in the smoke) and load-bearing only for the deferred full-2B
//! resident training engine.

use std::rc::Rc;

use tritium_core::GemmShape;
use tritium_spec::BackendError;
use tritium_train::ops::{act, loss, matmul, ste};
use tritium_train::{AdamState, AdamW, LrSchedule, Optimizer, TrainGemm};

use crate::cuda::CudaBackend;

/// `(g_a[M,K], g_w[N,K], g_s[N])` — the three matmul gradients returned together by
/// [`GemmEngine::backward`].
pub(crate) type GradTriple = (Vec<f32>, Vec<f32>, Vec<f32>);

/// A ternary GEMM engine: the forward `Y = s·(A·Wᵀ)` and its backward `(gA, gW, gs)`. Implemented
/// twice — on the GPU kernels and on the `tritium-train` CPU oracle — so one training-step code path
/// drives both, and the device↔CPU gate compares them.
pub(crate) trait GemmEngine {
    fn forward(
        &self,
        a: &[f32],
        w: &[f32],
        s: &[f32],
        shape: GemmShape,
    ) -> Result<Vec<f32>, BackendError>;
    /// Returns `(g_a[M,K], g_w[N,K], g_s[N])` for `gy[M,N]`, `a[M,K]`, `w[N,K]`, `s[N]`.
    fn backward(
        &self,
        gy: &[f32],
        a: &[f32],
        w: &[f32],
        s: &[f32],
        shape: GemmShape,
    ) -> Result<GradTriple, BackendError>;
}

/// The GPU engine: the `train_grad.cu` kernels via [`CudaBackend`].
pub(crate) struct GpuEngine<'a>(pub(crate) &'a CudaBackend);

impl GemmEngine for GpuEngine<'_> {
    fn forward(
        &self,
        a: &[f32],
        w: &[f32],
        s: &[f32],
        shape: GemmShape,
    ) -> Result<Vec<f32>, BackendError> {
        let mut y = vec![0.0f32; shape.m * shape.n];
        self.0.train_forward(a, w, s, shape, &mut y)?;
        Ok(y)
    }

    fn backward(
        &self,
        gy: &[f32],
        a: &[f32],
        w: &[f32],
        s: &[f32],
        shape: GemmShape,
    ) -> Result<GradTriple, BackendError> {
        let mut g_a = vec![0.0f32; shape.m * shape.k];
        let mut g_w = vec![0.0f32; shape.n * shape.k];
        let mut g_s = vec![0.0f32; shape.n];
        self.0.grad_a(gy, w, s, shape, &mut g_a)?;
        self.0.grad_w(gy, a, s, shape, &mut g_w)?;
        self.0.grad_s(gy, a, w, shape, &mut g_s)?;
        Ok((g_a, g_w, g_s))
    }
}

/// Owned GPU GEMM engine for injecting into a [`tritium_train::Tape`] via `Tape::with_gemm`
/// (plan 0043): it holds the [`CudaBackend`] (so it is `'static` for `Rc<dyn TrainGemm>`) and
/// delegates each matmul to the borrowed [`GpuEngine`] (the `train_grad.cu` kernels). A device error
/// is fatal to a training step, so the `TrainGemm` methods (which cannot return `Result` — the tape
/// builds the graph eagerly) surface it as a panic.
pub struct GpuGemm(Rc<CudaBackend>);

impl core::fmt::Debug for GpuGemm {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GpuGemm").finish_non_exhaustive()
    }
}

impl GpuGemm {
    /// Open CUDA `device` and build a tape GEMM engine on it.
    ///
    /// # Errors
    /// [`BackendError`] if the device cannot be opened.
    pub fn new(device: usize) -> Result<Self, BackendError> {
        Ok(Self(Rc::new(CudaBackend::new(device)?)))
    }
}

impl TrainGemm for GpuGemm {
    fn dense_forward(&self, x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        // fp dense = the ternary-matmul forward with a unit per-row scale (`s = 1`).
        let ones = vec![1.0f32; n];
        GpuEngine(&self.0)
            .forward(x, w, &ones, GemmShape { m, n, k })
            .expect("GpuGemm dense_forward: device error")
    }

    fn dense_backward(
        &self,
        gy: &[f32],
        x: &[f32],
        w: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        // Only `grad_a`/`grad_w` — the fp path has no per-row scale, so we skip the `grad_s` kernel
        // (a full GEMM's worth of work) that the ternary `GpuEngine::backward` would compute.
        let shape = GemmShape { m, n, k };
        let ones = vec![1.0f32; n];
        let mut g_x = vec![0.0f32; m * k];
        let mut g_w = vec![0.0f32; n * k];
        self.0
            .grad_a(gy, w, &ones, shape, &mut g_x)
            .expect("GpuGemm dense_backward grad_a: device error");
        self.0
            .grad_w(gy, x, &ones, shape, &mut g_w)
            .expect("GpuGemm dense_backward grad_w: device error");
        (g_x, g_w)
    }
}

/// The CPU oracle engine: [`tritium_train::ops::matmul`] forward + vjp. Test-only — it backs the
/// device↔CPU parity gate; the shipping path uses [`GpuEngine`].
#[cfg(test)]
pub(crate) struct CpuEngine;

#[cfg(test)]
impl GemmEngine for CpuEngine {
    fn forward(
        &self,
        a: &[f32],
        w: &[f32],
        s: &[f32],
        shape: GemmShape,
    ) -> Result<Vec<f32>, BackendError> {
        Ok(matmul::forward(a, w, s, shape.m, shape.n, shape.k))
    }

    fn backward(
        &self,
        gy: &[f32],
        a: &[f32],
        w: &[f32],
        s: &[f32],
        shape: GemmShape,
    ) -> Result<GradTriple, BackendError> {
        let mut g = matmul::vjp(a, w, s, shape.m, shape.n, shape.k, gy);
        let g_s = g.remove(2);
        let g_w = g.remove(1);
        let g_a = g.remove(0);
        Ok((g_a, g_w, g_s))
    }
}

/// A tiny 2-layer ternary MLP: f32 master weights + learned per-row scales, with per-leaf AdamW state.
#[derive(Clone)]
pub(crate) struct TinyMlp {
    k0: usize,
    n1: usize,
    n2: usize,
    w1: Vec<f32>, // [N1, K0] master
    s1: Vec<f32>, // [N1] learned scale
    w2: Vec<f32>, // [N2, N1] master
    s2: Vec<f32>, // [N2] learned scale
    st_w1: AdamState,
    st_s1: AdamState,
    st_w2: AdamState,
    st_s2: AdamState,
}

/// Per-step gradients for the four leaves, plus the step's scalar loss.
pub(crate) struct StepGrads {
    g_w1: Vec<f32>,
    g_s1: Vec<f32>,
    g_w2: Vec<f32>,
    g_s2: Vec<f32>,
    loss: f32,
}

impl TinyMlp {
    /// Seeded init: master weights uniform in `[-0.3, 0.3]` (off the 1-bit grid → STE gradient
    /// flows), each row's scale initialised to its AbsMean (the natural BitNet starting scale, then
    /// learned from there).
    pub(crate) fn init(k0: usize, n1: usize, n2: usize, seed: u64) -> Self {
        let w1 = seeded_uniform(seed ^ 0x11, n1 * k0, -0.3, 0.3);
        let w2 = seeded_uniform(seed ^ 0x22, n2 * n1, -0.3, 0.3);
        let s1 = ste::absmean_scale_per_row(&w1, n1, k0);
        let s2 = ste::absmean_scale_per_row(&w2, n2, n1);
        Self {
            st_w1: AdamW::new(0.0).init_state(w1.len()),
            st_s1: AdamW::new(0.0).init_state(s1.len()),
            st_w2: AdamW::new(0.0).init_state(w2.len()),
            st_s2: AdamW::new(0.0).init_state(s2.len()),
            k0,
            n1,
            n2,
            w1,
            s1,
            w2,
            s2,
        }
    }

    /// One forward+backward over a fixed batch `x[M,K₀]` against `target[M,N₂]`, returning the
    /// per-leaf gradients + loss. Uses the supplied [`GemmEngine`] for the matmuls; the STE,
    /// squared-ReLU, and MSE glue is reused from `tritium-train` (identical on both engines, so it
    /// cancels out of the device↔CPU comparison).
    pub(crate) fn forward_backward<E: GemmEngine>(
        &self,
        eng: &E,
        x: &[f32],
        target: &[f32],
        m: usize,
    ) -> Result<StepGrads, BackendError> {
        // The shipping path quantizes with `round` (the real QAT forward).
        self.forward_backward_with(eng, x, target, m, ste::quantize_forward)
    }

    /// Forward+backward parameterized by the weight quantizer. The shipping
    /// [`Self::forward_backward`] passes [`ste::quantize_forward`] (round); the gradient-check test
    /// passes [`ste::quantize_surrogate`] (the differentiable clamp) so a central finite difference
    /// of the loss is meaningful — pinning *this exact wiring* (which buffer feeds `relu2_vjp`,
    /// which cotangent feeds `quantize_vjp`, the layer ordering, the `.remove` routing) against the
    /// analytic gradient, independently of the device↔CPU parity gate (which shares the wiring).
    pub(crate) fn forward_backward_with<E: GemmEngine>(
        &self,
        eng: &E,
        x: &[f32],
        target: &[f32],
        m: usize,
        quant: fn(&[f32], &[f32], usize, usize) -> Vec<f32>,
    ) -> Result<StepGrads, BackendError> {
        let (k0, n1, n2) = (self.k0, self.n1, self.n2);
        // Layer 1: quantize master → ternary, ternary matmul, squared-ReLU.
        let q1 = quant(&self.w1, &self.s1, n1, k0);
        let h_pre = eng.forward(x, &q1, &self.s1, GemmShape { m, n: n1, k: k0 })?;
        let h = act::relu2_forward(&h_pre);
        // Layer 2.
        let q2 = quant(&self.w2, &self.s2, n2, n1);
        let y = eng.forward(&h, &q2, &self.s2, GemmShape { m, n: n2, k: n1 })?;
        let step_loss = loss::mse_forward(&y, target)[0];

        // Backward. MSE → dL/dy.
        let dy = loss::mse_vjp(&y, target, &[1.0]).remove(0);
        // Layer 2: dL/dh (to backprop), dL/dq2 (→ master via STE), dL/ds2 (learned scale).
        let (gh, gq2, g_s2) =
            eng.backward(&dy, &h, &q2, &self.s2, GemmShape { m, n: n2, k: n1 })?;
        let g_w2 = ste::quantize_vjp(&self.w2, &self.s2, n2, n1, &gq2).remove(0);
        // Squared-ReLU backward.
        let dh_pre = act::relu2_vjp(&h_pre, &gh).remove(0);
        // Layer 1: gradient into x is discarded (x is data); dL/dq1, dL/ds1 are kept.
        let (_gx, gq1, g_s1) =
            eng.backward(&dh_pre, x, &q1, &self.s1, GemmShape { m, n: n1, k: k0 })?;
        let g_w1 = ste::quantize_vjp(&self.w1, &self.s1, n1, k0, &gq1).remove(0);

        Ok(StepGrads {
            g_w1,
            g_s1,
            g_w2,
            g_s2,
            loss: step_loss,
        })
    }

    /// Apply one AdamW update to every leaf at 1-based step `t` with learning rate `lr`.
    pub(crate) fn apply(&mut self, lr: f32, t: u64, g: &StepGrads) {
        // Standard decoupled weight decay on the weight masters.
        let opt_w = AdamW::new(lr);
        // The learned per-row scales are output gains, not weights — exclude them from weight decay
        // so it can't pull `s → 0` against `grad_s` (the codebase idiom; cf. qat_heal_gate.rs).
        let opt_s = AdamW {
            weight_decay: 0.0,
            ..AdamW::new(lr)
        };
        opt_w.step(t, &mut self.w1, &g.g_w1, &mut self.st_w1);
        opt_s.step(t, &mut self.s1, &g.g_s1, &mut self.st_s1);
        opt_w.step(t, &mut self.w2, &g.g_w2, &mut self.st_w2);
        opt_s.step(t, &mut self.s2, &g.g_s2, &mut self.st_s2);
    }
}

/// Knobs for [`pretrain_smoke`].
#[derive(Clone, Copy, Debug)]
pub struct SmokeConfig {
    /// CUDA device ordinal.
    pub device: usize,
    /// Batch size `M`, input dim `K₀`, hidden dim `N₁`, output dim `N₂`.
    pub m: usize,
    pub k0: usize,
    pub n1: usize,
    pub n2: usize,
    /// Total optimizer steps and the warmup length for the LR schedule.
    pub steps: u64,
    pub warmup: u64,
    /// Peak / floor learning rate.
    pub peak_lr: f32,
    pub min_lr: f32,
    /// RNG seed for init + synthetic data.
    pub seed: u64,
}

impl Default for SmokeConfig {
    fn default() -> Self {
        Self {
            device: 0,
            m: 32,
            k0: 8,
            n1: 16,
            n2: 4,
            steps: 300,
            warmup: 20,
            peak_lr: 0.05,
            min_lr: 0.005,
            seed: 0xC0FFEE,
        }
    }
}

/// Outcome of a pretrain smoke run.
#[derive(Clone, Copy, Debug)]
pub struct SmokeReport {
    /// Loss at step 0 and after the final step.
    pub initial_loss: f32,
    pub final_loss: f32,
    /// Steps actually run.
    pub steps: u64,
    /// `true` if any step produced a non-finite loss (the run is then invalid).
    pub saw_non_finite: bool,
}

/// Train the tiny ternary MLP on a fixed synthetic batch and report the loss trajectory — the
/// ADR-0008 "from-scratch tiny model reaches target loss" gate on one GPU.
///
/// The target is a fixed small linear map of the input (`target = x · Wₜᵀ`); the ternary 2-layer
/// MLP learns to approximate it. Returns the initial/final loss so the caller can gate on the drop.
///
/// # Errors
/// Propagates [`BackendError`] from opening the device or any kernel launch.
pub fn pretrain_smoke(cfg: &SmokeConfig) -> Result<SmokeReport, BackendError> {
    let backend = CudaBackend::new(cfg.device)?;
    let eng = GpuEngine(&backend);
    let mut mlp = TinyMlp::init(cfg.k0, cfg.n1, cfg.n2, cfg.seed);

    // Fixed synthetic batch: random input, target = a small fixed linear map of it (realizable
    // enough that the loss should fall substantially).
    let x = seeded_uniform(cfg.seed ^ 0xA1, cfg.m * cfg.k0, -1.0, 1.0);
    let teacher = seeded_uniform(cfg.seed ^ 0xB2, cfg.n2 * cfg.k0, -0.5, 0.5);
    let target = matmul::forward(&x, &teacher, &vec![1.0; cfg.n2], cfg.m, cfg.n2, cfg.k0);

    let sched = LrSchedule::new(cfg.peak_lr, cfg.min_lr, cfg.warmup, cfg.steps.max(1));
    let mut initial_loss = f32::NAN;
    let mut last_loss = f32::NAN;
    let mut saw_non_finite = false;
    for step in 0..cfg.steps {
        let g = mlp.forward_backward(&eng, &x, &target, cfg.m)?;
        if step == 0 {
            initial_loss = g.loss;
        }
        last_loss = g.loss;
        if !g.loss.is_finite() {
            saw_non_finite = true;
            break;
        }
        mlp.apply(sched.lr(step), step + 1, &g);
    }

    Ok(SmokeReport {
        initial_loss,
        final_loss: last_loss,
        steps: cfg.steps,
        saw_non_finite,
    })
}

/// Deterministic uniform `[lo, hi)` f32 vector from a splitmix64 stream (no external RNG; matches
/// the integer-only determinism the rest of the project keeps).
fn seeded_uniform(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut z = seed;
    (0..n)
        .map(|_| {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^= x >> 31;
            // top 24 bits → [0,1)
            let unit = (x >> 40) as f32 / (1u64 << 24) as f32;
            lo + unit * (hi - lo)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// Set each row's scale to `1.25·max|w_row|` so every `|w/s| ≤ 0.8` — i.e. the clamp surrogate
    /// is in its linear region at the operating point, with no kink within a finite-difference step.
    fn force_in_band(w: &[f32], s: &mut [f32], rows: usize, cols: usize) {
        for r in 0..rows {
            let row_max = (0..cols)
                .map(|c| w[r * cols + c].abs())
                .fold(0.0f32, f32::max);
            s[r] = (row_max * 1.25).max(1e-3);
        }
    }

    /// Gate (0013 review): a CPU central finite-difference check that the composed step's
    /// weight-master gradients equal the gradient of the smooth STE-surrogate loss. This pins the
    /// composition *wiring* — `relu2_vjp` on the pre-activation, `quantize_vjp` on `grad_w`'s
    /// `dL/dq`, the layer ordering, the `.remove` routing — independently of
    /// [`device_step_matches_cpu_tape`] (which shares the wiring across both engines and so cannot
    /// catch a coherent miswiring). The learned scales are forced in-band so the surrogate is smooth;
    /// the scale-leaf gradients are covered by the op-level `grad_s` check + the device==CPU step.
    #[test]
    fn forward_backward_grads_match_finite_difference() {
        let (m, k0, n1, n2) = (4, 5, 4, 3);
        let mut mlp = TinyMlp::init(k0, n1, n2, 0xF00D);
        force_in_band(&mlp.w1.clone(), &mut mlp.s1, n1, k0);
        force_in_band(&mlp.w2.clone(), &mut mlp.s2, n2, n1);
        let x = seeded_uniform(0x999, m * k0, -1.0, 1.0);
        let target = seeded_uniform(0xAAA, m * n2, -1.0, 1.0);

        // Analytic gradient of the SURROGATE loss (the STE backward IS the surrogate's exact grad).
        let analytic = mlp
            .forward_backward_with(&CpuEngine, &x, &target, m, ste::quantize_surrogate)
            .unwrap();

        let h = 1e-3f32;
        let surrogate_loss = |p: &TinyMlp| {
            p.forward_backward_with(&CpuEngine, &x, &target, m, ste::quantize_surrogate)
                .unwrap()
                .loss
        };
        let check = |analytic_g: &[f32], pick: fn(&mut TinyMlp) -> &mut Vec<f32>| {
            for (i, &a) in analytic_g.iter().enumerate() {
                let mut pp = mlp.clone();
                pick(&mut pp)[i] += h;
                let lp = surrogate_loss(&pp);
                let mut pm = mlp.clone();
                pick(&mut pm)[i] -= h;
                let lm = surrogate_loss(&pm);
                let numeric = (lp - lm) / (2.0 * h);
                // Combined absolute-OR-relative: the abs floor (5e-5) absorbs f32 FD roundoff on
                // tiny gradients; the 1.5% relative bar still catches any structural miswiring
                // (which is off by large factors, not ~1%).
                let abs_err = (numeric - a).abs();
                let rel_err = abs_err / a.abs().max(numeric.abs()).max(1e-12);
                assert!(
                    abs_err <= 5e-5 || rel_err <= 1.5e-2,
                    "weight-grad mismatch at {i}: analytic {a}, numeric {numeric} \
                     (abs {abs_err}, rel {rel_err})"
                );
            }
        };
        check(&analytic.g_w1, |p| &mut p.w1);
        check(&analytic.g_w2, |p| &mut p.w2);
    }

    /// Gate: one composed 2-layer training step's gradients (and loss) computed on the GPU match
    /// the `tritium-train` CPU tape vjp within 1e-4 — device backward == CPU vjp, end to end.
    #[test]
    fn device_step_matches_cpu_tape() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping device_step_matches_cpu_tape: no CUDA device ({e})");
                return;
            }
        };
        let (m, k0, n1, n2) = (5, 7, 6, 3);
        let mlp = TinyMlp::init(k0, n1, n2, 0xABCDEF);
        let x = seeded_uniform(0x1234, m * k0, -1.5, 1.5);
        let target = seeded_uniform(0x5678, m * n2, -1.0, 1.0);

        let gpu = mlp
            .forward_backward(&GpuEngine(&backend), &x, &target, m)
            .expect("gpu step");
        let cpu = mlp
            .forward_backward(&CpuEngine, &x, &target, m)
            .expect("cpu step");

        let tol = 1e-4;
        assert!(
            (gpu.loss - cpu.loss).abs() <= tol,
            "loss {} vs {}",
            gpu.loss,
            cpu.loss
        );
        assert!(max_abs_diff(&gpu.g_w1, &cpu.g_w1) <= tol, "g_w1 mismatch");
        assert!(max_abs_diff(&gpu.g_s1, &cpu.g_s1) <= tol, "g_s1 mismatch");
        assert!(max_abs_diff(&gpu.g_w2, &cpu.g_w2) <= tol, "g_w2 mismatch");
        assert!(max_abs_diff(&gpu.g_s2, &cpu.g_s2) <= tol, "g_s2 mismatch");
    }

    /// Gate (plan 0043 P2.1): a device-**resident** 2-matmul chain (fwd + bwd) matches the CPU
    /// `tritium-train` tape BIT-EXACTLY, and eliminates the per-op host↔device round-trips. Unlike
    /// Phase 1 (each matmul htod→launch→dtoh), the activations `h`, `y` and the backprop grad `gh`
    /// live in VRAM across the whole step — uploaded once (leaves), downloaded once (results). This
    /// is the foundational proof of the device-resident execution model; the glue ops (rmsnorm,
    /// silu, …) chain onto the same buffers in later increments.
    #[test]
    fn resident_matmul_chain_matches_cpu_tape() {
        use std::time::Instant;

        use tritium_train::Tape;

        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping resident_matmul_chain_matches_cpu_tape: no CUDA device ({e})");
                return;
            }
        };
        // Non-trivial sizes so the matmuls dominate and residency (no round-trips) actually shows.
        let (m, k0, n1, n2) = (64usize, 512usize, 1024usize, 512usize);
        let x = seeded_uniform(0x1111, m * k0, -1.0, 1.0);
        let w1 = seeded_uniform(0x2222, n1 * k0, -0.5, 0.5);
        let w2 = seeded_uniform(0x3333, n2 * n1, -0.5, 0.5);
        let cot = seeded_uniform(0x4444, m * n2, -1.0, 1.0); // upstream cotangent: L = Σ y·cot

        // ── CPU reference: the same chain on the tritium-train tape ──
        let mut t = Tape::new();
        let xid = t.leaf(x.clone());
        let w1id = t.leaf(w1.clone());
        let w2id = t.leaf(w2.clone());
        let hid = t.dense_matmul(xid, w1id, m, n1, k0); // [m, n1]
        let yid = t.dense_matmul(hid, w2id, m, n2, n1); // [m, n2]
        let cid = t.leaf(cot.clone());
        let loss = t.dense_matmul(yid, cid, 1, 1, m * n2); // Σ y·cot ⇒ dL/dy = cot
        let cpu_y = t.value(yid).to_vec();
        let grads = t.backward(loss);
        let (cpu_gx, cpu_gw1, cpu_gw2) =
            (grads[xid].clone(), grads[w1id].clone(), grads[w2id].clone());

        // ── Device-resident: upload leaves once, chain on-device, download results once ──
        let shape1 = GemmShape { m, n: n1, k: k0 };
        let shape2 = GemmShape { m, n: n2, k: n1 };
        let ones = vec![1.0f32; n1.max(n2)]; // s = 1 (fp dense); one buffer serves both matmuls
        let resident_step = || -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
            let d_x = backend.dev_upload(&x).unwrap();
            let d_w1 = backend.dev_upload(&w1).unwrap();
            let d_w2 = backend.dev_upload(&w2).unwrap();
            let d_ones = backend.dev_upload(&ones).unwrap();
            // forward — h and y stay resident
            let mut d_h = backend.dev_alloc_zeros(m * n1).unwrap();
            backend
                .matmul_forward_dev(&d_x, &d_w1, &d_ones, shape1, &mut d_h)
                .unwrap();
            let mut d_y = backend.dev_alloc_zeros(m * n2).unwrap();
            backend
                .matmul_forward_dev(&d_h, &d_w2, &d_ones, shape2, &mut d_y)
                .unwrap();
            // backward — gy2 = cot; gh stays resident between the two layers' grads
            let d_gy2 = backend.dev_upload(&cot).unwrap();
            let mut d_gw2 = backend.dev_alloc_zeros(n2 * n1).unwrap();
            backend
                .grad_w_dev(&d_gy2, &d_h, &d_ones, shape2, &mut d_gw2)
                .unwrap();
            let mut d_gh = backend.dev_alloc_zeros(m * n1).unwrap();
            backend
                .grad_a_dev(&d_gy2, &d_w2, &d_ones, shape2, &mut d_gh)
                .unwrap();
            let mut d_gw1 = backend.dev_alloc_zeros(n1 * k0).unwrap();
            backend
                .grad_w_dev(&d_gh, &d_x, &d_ones, shape1, &mut d_gw1)
                .unwrap();
            let mut d_gx = backend.dev_alloc_zeros(m * k0).unwrap();
            backend
                .grad_a_dev(&d_gh, &d_w1, &d_ones, shape1, &mut d_gx)
                .unwrap();
            let (mut y, mut gx, mut gw1, mut gw2) = (
                vec![0.0f32; m * n2],
                vec![0.0f32; m * k0],
                vec![0.0f32; n1 * k0],
                vec![0.0f32; n2 * n1],
            );
            backend.dev_download(&d_y, &mut y).unwrap();
            backend.dev_download(&d_gx, &mut gx).unwrap();
            backend.dev_download(&d_gw1, &mut gw1).unwrap();
            backend.dev_download(&d_gw2, &mut gw2).unwrap();
            (y, gx, gw1, gw2)
        };
        let (dev_y, dev_gx, dev_gw1, dev_gw2) = resident_step();

        // ── Bit-exact gate ──
        assert_eq!(max_abs_diff(&dev_y, &cpu_y), 0.0, "forward y not bit-exact");
        assert_eq!(max_abs_diff(&dev_gx, &cpu_gx), 0.0, "grad x not bit-exact");
        assert_eq!(
            max_abs_diff(&dev_gw1, &cpu_gw1),
            0.0,
            "grad w1 not bit-exact"
        );
        assert_eq!(
            max_abs_diff(&dev_gw2, &cpu_gw2),
            0.0,
            "grad w2 not bit-exact"
        );

        // ── Bench: resident vs host-orchestrated (the same chain via the Phase-1 htod/dtoh methods) ──
        // (The host-slice methods require `s.len() == n` exactly, so slice `ones` per shape.)
        let (ones1, ones2) = (&ones[..n1], &ones[..n2]);
        let host_orch_step = || {
            let mut h = vec![0.0f32; m * n1];
            backend
                .train_forward(&x, &w1, ones1, shape1, &mut h)
                .unwrap();
            let mut y = vec![0.0f32; m * n2];
            backend
                .train_forward(&h, &w2, ones2, shape2, &mut y)
                .unwrap();
            let mut gw2 = vec![0.0f32; n2 * n1];
            backend.grad_w(&cot, &h, ones2, shape2, &mut gw2).unwrap();
            let mut gh = vec![0.0f32; m * n1];
            backend.grad_a(&cot, &w2, ones2, shape2, &mut gh).unwrap();
            let mut gw1 = vec![0.0f32; n1 * k0];
            backend.grad_w(&gh, &x, ones1, shape1, &mut gw1).unwrap();
            let mut gx = vec![0.0f32; m * k0];
            backend.grad_a(&gh, &w1, ones1, shape1, &mut gx).unwrap();
        };
        let iters = 50;
        for _ in 0..5 {
            resident_step();
            host_orch_step();
        } // warm up
        let t0 = Instant::now();
        for _ in 0..iters {
            resident_step();
        }
        let resident_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let t1 = Instant::now();
        for _ in 0..iters {
            host_orch_step();
        }
        let host_ms = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;
        eprintln!(
            "0043 P2.1 resident matmul chain (m={m} k0={k0} n1={n1} n2={n2}): BIT-EXACT vs CPU tape. \
             fwd+bwd step: resident {resident_ms:.3}ms | host-orchestrated {host_ms:.3}ms \
             ({:.2}× fewer round-trips)",
            host_ms / resident_ms.max(1e-9)
        );
    }

    /// Gate: the from-scratch tiny model's loss falls well below its start over the step budget,
    /// with no non-finite loss along the way.
    #[test]
    fn pretrain_smoke_reaches_target_loss() {
        let cfg = SmokeConfig::default();
        let t0 = std::time::Instant::now();
        let report = match pretrain_smoke(&cfg) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping pretrain_smoke_reaches_target_loss: no CUDA device ({e})");
                return;
            }
        };
        let elapsed = t0.elapsed();
        // Diagnostic (visible under `--nocapture`): trajectory + per-step cost. 4 GPU matmuls/step
        // (2 fwd + 2 bwd), each a host↔device round-trip — the measurement behind the plan's
        // "activation-retention is a deferred perf refinement" note.
        eprintln!(
            "pretrain smoke: loss {:.5} -> {:.5} ({:.1}% of initial) over {} steps; \
             {:.2} ms total, {:.3} ms/step ({} round-tripped matmuls/step)",
            report.initial_loss,
            report.final_loss,
            100.0 * report.final_loss / report.initial_loss,
            report.steps,
            elapsed.as_secs_f64() * 1e3,
            elapsed.as_secs_f64() * 1e3 / report.steps as f64,
            4,
        );
        assert!(
            !report.saw_non_finite,
            "training produced a non-finite loss"
        );
        assert!(report.initial_loss.is_finite() && report.initial_loss > 0.0);
        // Substantial decrease — gate at ≥70% reduction; observed ~90.5% (0.261→0.025), so the
        // margin absorbs minor cross-GPU float drift over 300 deterministic steps.
        assert!(
            report.final_loss < report.initial_loss * 0.3,
            "loss did not fall enough: {} -> {}",
            report.initial_loss,
            report.final_loss
        );
    }
}
