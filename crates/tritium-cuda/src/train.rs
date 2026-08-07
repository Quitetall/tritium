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
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};
use cudarc::driver::{CudaSlice, CudaStream, PinnedHostSlice};
use tritium_core::GemmShape;
use tritium_spec::BackendError;
use tritium_train::dcp::{DcpError, StatePlane};
use tritium_train::ops::{act, loss, matmul, ste};
use tritium_train::{AdamState, AdamW, LrSchedule, Optimizer, TrainGemm};

use crate::cuda::{CudaBackend, EmbedSegments, TrainingSaltLinear};

mod portable;

pub use portable::CudaTrainBackendV1;

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

/// Tensor-core training GEMMs via cuBLASLt (Lever 1). All three training GEMMs — forward
/// `Y=X·Wᵀ`, activation grad `gA=gY·W`, weight grad `gW=gYᵀ·X` — run on the 4090's tf32
/// tensor cores (`CUBLAS_COMPUTE_32F_FAST_TF32`, fp32 accumulate, f32 in/out), a drop-in
/// replacement for the naive `--fmad=false` f32 kernels that is ~65× faster on realistic
/// shapes. Not bit-exact vs the CPU oracle (tf32 truncates the mantissa to ~10 bits); the
/// tier is gated on end-to-end distillation recovery, with the f32 kernels kept as the
/// correctness oracle.
///
/// Row-major → column-major cuBLASLt mapping (cuBLASLt is column-major; a row-major
/// `[r,c]` buffer with leading dim `c` is a column-major `[c,r]` buffer with ld `c`):
/// - forward  `Y[m,n]=X[m,k]·Wᵀ`      → `a=W,  b=X,  transa,        m=n, n=m, k=k, lda=k, ldb=k, ldc=n`
/// - grad_a   `gA[m,k]=gY[m,n]·W[n,k]` → `a=W,  b=gY,                m=k, n=m, k=n, lda=k, ldb=n, ldc=k`
/// - grad_w   `gW[n,k]=gYᵀ·X`          → `a=X,  b=gY, transb,        m=k, n=n, k=m, lda=k, ldb=n, ldc=k`
pub struct TensorCoreGemm {
    blas: CudaBlasLT,
}

impl std::fmt::Debug for TensorCoreGemm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorCoreGemm").finish_non_exhaustive()
    }
}

impl TensorCoreGemm {
    /// Build a cuBLASLt handle on the backend's stream. Fails closed if cuBLASLt cannot be
    /// loaded (e.g. `libcublasLt` absent), so callers can fall back to the f32 kernels.
    pub fn new(backend: &CudaBackend) -> Result<Self, BackendError> {
        let blas = CudaBlasLT::new(backend.stream().clone()).map_err(|e| {
            BackendError::InvalidInput(format!("cuBLASLt handle init failed: {e:?}"))
        })?;
        Ok(Self { blas })
    }

    fn run(
        &self,
        cfg: MatmulConfig,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        // SAFETY: every call site fixes `cfg`'s shapes/leading-dims to the row-major→column-major
        // mapping documented on the type, and the buffers are sized by the shape guards below.
        #[allow(unsafe_code)]
        unsafe {
            self.blas
                .matmul(cfg, a, b, c, Option::<&CudaSlice<f32>>::None, None)
                .map_err(|e| BackendError::InvalidInput(format!("cuBLASLt matmul failed: {e:?}")))
        }
    }

    /// `Y[m,n] = X[m,k]·Wᵀ` (`W` is `[n,k]`), tf32 tensor cores. `d_y` preallocated `m*n`.
    pub fn forward(
        &self,
        d_x: &CudaSlice<f32>,
        d_w: &CudaSlice<f32>,
        shape: GemmShape,
        d_y: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        let GemmShape { m, n, k } = shape;
        if d_x.len() < m * k || d_w.len() < n * k || d_y.len() < m * n {
            return Err(BackendError::ShapeMismatch {
                expected: m * n,
                got: d_y.len(),
            });
        }
        if m * n == 0 || k == 0 {
            return Ok(());
        }
        self.run(
            MatmulConfig {
                transa: true,
                transb: false,
                transc: false,
                m: n as u64,
                n: m as u64,
                k: k as u64,
                alpha: 1.0,
                lda: k as i64,
                ldb: k as i64,
                beta: 0.0,
                ldc: n as i64,
                stride_a: None,
                stride_b: None,
                stride_c: None,
                stride_bias: None,
                batch_size: None,
            },
            d_w,
            d_x,
            d_y,
        )
    }

    /// `gA[m,k] = gY[m,n]·W[n,k]` (activation grad), tf32 tensor cores. `d_ga` preallocated `m*k`.
    pub fn grad_a(
        &self,
        d_gy: &CudaSlice<f32>,
        d_w: &CudaSlice<f32>,
        shape: GemmShape,
        d_ga: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        let GemmShape { m, n, k } = shape;
        if d_gy.len() < m * n || d_w.len() < n * k || d_ga.len() < m * k {
            return Err(BackendError::ShapeMismatch {
                expected: m * k,
                got: d_ga.len(),
            });
        }
        if m * k == 0 || n == 0 {
            return Ok(());
        }
        self.run(
            MatmulConfig {
                transa: false,
                transb: false,
                transc: false,
                m: k as u64,
                n: m as u64,
                k: n as u64,
                alpha: 1.0,
                lda: k as i64,
                ldb: n as i64,
                beta: 0.0,
                ldc: k as i64,
                stride_a: None,
                stride_b: None,
                stride_c: None,
                stride_bias: None,
                batch_size: None,
            },
            d_w,
            d_gy,
            d_ga,
        )
    }

    /// `gW[n,k] = Σ_m gY[m,n]·X[m,k]` (weight grad), tf32 tensor cores. `d_gw` preallocated `n*k`.
    pub fn grad_w(
        &self,
        d_gy: &CudaSlice<f32>,
        d_x: &CudaSlice<f32>,
        shape: GemmShape,
        d_gw: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        let GemmShape { m, n, k } = shape;
        if d_gy.len() < m * n || d_x.len() < m * k || d_gw.len() < n * k {
            return Err(BackendError::ShapeMismatch {
                expected: n * k,
                got: d_gw.len(),
            });
        }
        if n * k == 0 || m == 0 {
            return Ok(());
        }
        self.run(
            MatmulConfig {
                transa: false,
                transb: true,
                transc: false,
                m: k as u64,
                n: n as u64,
                k: m as u64,
                alpha: 1.0,
                lda: k as i64,
                ldb: n as i64,
                beta: 0.0,
                ldc: k as i64,
                stride_a: None,
                stride_b: None,
                stride_c: None,
                stride_bias: None,
                batch_size: None,
            },
            d_x,
            d_gy,
            d_gw,
        )
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

/// An op recorded on the [`DeviceTape`] for the reverse pass — input/output value ids + params. The
/// backward reads the output's grad + the saved forward values (`vals`) and accumulates into the
/// inputs' grad buffers (so a value with >1 consumer — residuals, the tied embedding — sums exactly
/// like the CPU tape's `grads[id] += v`).
// Test-exercised (`device_tape_mlp_stack_matches_cpu_tape`) until the distillation loop drives the
// DeviceTape on a real model (plan 0043 P2.5b onward).
#[allow(dead_code)]
enum DevOp<'leaf> {
    HestiaRelax {
        weight: usize,
        scale: usize,
        tau: usize,
        rows: usize,
        cols: usize,
        out: usize,
    },
    Matmul {
        x: usize,
        w: usize,
        m: usize,
        n: usize,
        k: usize,
        out: usize,
    },
    SaltMatmul {
        x: usize,
        master: usize,
        weight: &'leaf DevicePackedSaltWeight,
        m: usize,
        n: usize,
        k: usize,
        out: usize,
    },
    Rmsnorm {
        x: usize,
        w: usize,
        rows: usize,
        cols: usize,
        eps: f32,
        out: usize,
    },
    Silu {
        x: usize,
        n: usize,
        out: usize,
    },
    Mul {
        a: usize,
        b: usize,
        n: usize,
        out: usize,
    },
    Add {
        a: usize,
        b: usize,
        n: usize,
        out: usize,
    },
    Embed {
        w: usize,
        tokens: CudaSlice<i32>,
        segments: EmbedSegments,
        seq: usize,
        dim: usize,
        vocab: usize,
        out: usize,
    },
    SaltEmbed {
        master: usize,
        weight: &'leaf DevicePackedSaltWeight,
        tokens: CudaSlice<i32>,
        segments: EmbedSegments,
        seq: usize,
        dim: usize,
        vocab: usize,
        out: usize,
    },
    Rope {
        x: usize,
        pos: CudaSlice<u32>,
        n_head: usize,
        head_dim: usize,
        theta: f32,
        n_token: usize,
        out: usize,
    },
    SliceCols {
        x: usize,
        rows: usize,
        cols: usize,
        start: usize,
        len: usize,
        out: usize,
    },
    ScaleConst {
        x: usize,
        c: f32,
        n: usize,
        out: usize,
    },
    CausalMask {
        x: usize,
        rows: usize,
        cols: usize,
        out: usize,
    },
    Softmax {
        x: usize,
        rows: usize,
        cols: usize,
        out: usize,
    },
    Transpose {
        x: usize,
        rows: usize,
        cols: usize,
        out: usize,
    },
    Concat {
        parts: Vec<usize>,
        rows: usize,
        lens: Vec<usize>,
        out: usize,
    },
}

impl DevOp<'_> {
    fn output(&self) -> usize {
        match self {
            Self::HestiaRelax { out, .. }
            | Self::Matmul { out, .. }
            | Self::SaltMatmul { out, .. }
            | Self::Rmsnorm { out, .. }
            | Self::Silu { out, .. }
            | Self::Mul { out, .. }
            | Self::Add { out, .. }
            | Self::Embed { out, .. }
            | Self::SaltEmbed { out, .. }
            | Self::Rope { out, .. }
            | Self::SliceCols { out, .. }
            | Self::ScaleConst { out, .. }
            | Self::CausalMask { out, .. }
            | Self::Softmax { out, .. }
            | Self::Transpose { out, .. }
            | Self::Concat { out, .. } => *out,
        }
    }

    fn gradient_inputs(&self) -> Vec<usize> {
        match self {
            Self::HestiaRelax {
                weight, scale, tau, ..
            } => vec![*weight, *scale, *tau],
            Self::Matmul { x, w, .. } => vec![*x, *w],
            Self::SaltMatmul { x, master, .. } => vec![*x, *master],
            Self::Rmsnorm { x, w, .. } => vec![*x, *w],
            Self::Silu { x, .. }
            | Self::Rope { x, .. }
            | Self::SliceCols { x, .. }
            | Self::ScaleConst { x, .. }
            | Self::CausalMask { x, .. }
            | Self::Softmax { x, .. }
            | Self::Transpose { x, .. } => vec![*x],
            Self::Mul { a, b, .. } | Self::Add { a, b, .. } => vec![*a, *b],
            Self::Embed { w, .. } => vec![*w],
            Self::SaltEmbed { master, .. } => vec![*master],
            Self::Concat { parts, .. } => parts.clone(),
        }
    }
}

/// Activation retention policy for [`DeviceTape`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckpointPolicy {
    /// Retain every forward activation until its VJP runs.
    KeepAll,
    /// Materialize one checkpoint after each non-zero number of block markers.
    EveryBlocks(usize),
    /// Choose `ceil(sqrt(total_blocks))` as the checkpoint interval.
    SqrtDepth(usize),
}

/// Arithmetic policy for packed SALT contractions recorded by [`DeviceTape`].
///
/// Exact preserves the dense-order semantic twin. Fast uses the reassociated,
/// plane-grouped multiply-free kernels and is therefore an explicit immutable
/// campaign choice rather than a shape-dependent fallback.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PackedSaltComputePolicy {
    #[default]
    Exact,
    Fast,
}

impl CheckpointPolicy {
    fn interval(self) -> Result<Option<usize>, BackendError> {
        match self {
            Self::KeepAll => Ok(None),
            Self::EveryBlocks(0) => Err(BackendError::InvalidInput(
                "checkpoint interval must be non-zero".into(),
            )),
            Self::EveryBlocks(interval) => Ok(Some(interval)),
            Self::SqrtDepth(0) => Err(BackendError::InvalidInput(
                "checkpoint depth must be non-zero".into(),
            )),
            Self::SqrtDepth(total_blocks) => {
                let floor = (total_blocks as f64).sqrt() as usize;
                let interval = if floor.saturating_mul(floor) < total_blocks {
                    floor + 1
                } else {
                    floor
                };
                Ok(Some(interval))
            }
        }
    }
}

#[derive(Clone, Debug)]
struct CheckpointSegment {
    op_start: usize,
    op_end: usize,
    value_start: usize,
    value_end: usize,
    frontier: Vec<usize>,
    evicted: bool,
}

/// An opaque f32 tensor owned by one CUDA context.
///
/// The allocation can be borrowed by [`DeviceTape::leaf_device`] without a
/// device-to-device copy.  Its contents remain private so callers cannot mix
/// contexts or mutate a tensor while a tape borrows it.
pub struct DeviceTensor {
    buf: CudaSlice<f32>,
    version: Arc<DeviceTensorVersion>,
}

struct DeviceTensorVersion {
    generation: AtomicU64,
}

impl core::fmt::Debug for DeviceTensor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DeviceTensor")
            .field("len", &self.buf.len())
            .field("device", &self.buf.ordinal())
            .finish_non_exhaustive()
    }
}

impl DeviceTensor {
    /// Upload one tensor into `backend`'s CUDA context.
    pub fn upload(backend: &CudaBackend, host: &[f32]) -> Result<Self, BackendError> {
        Ok(Self {
            buf: backend.dev_upload(host)?,
            version: Arc::new(DeviceTensorVersion {
                generation: AtomicU64::new(0),
            }),
        })
    }

    /// Number of f32 elements in this tensor.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Whether this tensor has no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Copy this tensor to host memory for evaluation or checkpointing.
    pub fn download(&self, backend: &CudaBackend) -> Result<Vec<f32>, BackendError> {
        if !backend.same_context(&self.buf) {
            return Err(BackendError::InvalidInput(
                "device tensor belongs to a different CUDA context".into(),
            ));
        }
        let mut host = vec![0.0; self.buf.len()];
        backend.dev_download(&self.buf, &mut host)?;
        Ok(host)
    }

    #[cfg(feature = "nccl")]
    pub(crate) fn resident_buffer(&self) -> &CudaSlice<f32> {
        &self.buf
    }
}

/// Opaque device-resident SALT weight for training-time packed execution.
///
/// The handle owns compact TQ2-addressed plane codes plus external f32 per-row
/// scales. It never owns a dense quantized reconstruction. Latent masters and
/// optimizer state remain separate; callers explicitly repack after updating a
/// host-resident master.
pub struct DevicePackedSaltWeight {
    inner: TrainingSaltLinear,
    prepared: bool,
    source: Option<PackedMasterBinding>,
}

struct PackedMasterBinding {
    version: Arc<DeviceTensorVersion>,
    packed_generation: u64,
}

impl core::fmt::Debug for DevicePackedSaltWeight {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DevicePackedSaltWeight")
            .field("rows", &self.rows())
            .field("cols", &self.cols())
            .field("planes", &self.planes())
            .field("resident_bytes", &self.resident_bytes())
            .field("prepared", &self.prepared)
            .field("source_bound", &self.source.is_some())
            .finish_non_exhaustive()
    }
}

impl DevicePackedSaltWeight {
    /// Upload and greedily SALT-pack one latent host matrix.
    pub fn from_host(
        backend: &CudaBackend,
        master: &[f32],
        rows: usize,
        cols: usize,
        planes: usize,
    ) -> Result<Self, BackendError> {
        let expected = rows.checked_mul(cols).ok_or_else(|| {
            BackendError::InvalidInput("packed SALT shape overflows usize".into())
        })?;
        if master.len() != expected {
            return Err(BackendError::ShapeMismatch {
                expected,
                got: master.len(),
            });
        }
        let d_master = backend.dev_upload(master)?;
        let mut scratch = backend.dev_alloc_zeros(expected)?;
        let inner = backend.pack_training_salt(&d_master, &mut scratch, rows, cols, planes)?;
        Ok(Self {
            inner,
            prepared: true,
            source: None,
        })
    }

    /// Pack an immutable resident tensor and bind the handle to that exact allocation.
    pub fn from_device_master(
        backend: &CudaBackend,
        master: &DeviceTensor,
        rows: usize,
        cols: usize,
        planes: usize,
    ) -> Result<Self, BackendError> {
        if !backend.same_context(&master.buf) {
            return Err(BackendError::InvalidInput(
                "device master belongs to a different CUDA context".into(),
            ));
        }
        let expected = rows.checked_mul(cols).ok_or_else(|| {
            BackendError::InvalidInput("packed SALT shape overflows usize".into())
        })?;
        if master.len() != expected {
            return Err(BackendError::ShapeMismatch {
                expected,
                got: master.len(),
            });
        }
        let mut scratch = backend.dev_alloc_zeros(expected)?;
        let inner = backend.pack_training_salt(&master.buf, &mut scratch, rows, cols, planes)?;
        Ok(Self {
            inner,
            prepared: true,
            source: Some(PackedMasterBinding {
                version: Arc::clone(&master.version),
                packed_generation: master.version.generation.load(Ordering::Acquire),
            }),
        })
    }

    /// Replace all packed codes/scales from a new host master with the same
    /// geometry. The handle becomes stale before validation or device work and
    /// becomes prepared again only after a successful pack.
    pub fn repack_from_host(
        &mut self,
        backend: &CudaBackend,
        master: &[f32],
    ) -> Result<(), BackendError> {
        self.prepared = false;
        let expected = self.rows().checked_mul(self.cols()).ok_or_else(|| {
            BackendError::InvalidInput("packed SALT shape overflows usize".into())
        })?;
        if master.len() != expected {
            return Err(BackendError::ShapeMismatch {
                expected,
                got: master.len(),
            });
        }
        let d_master = backend.dev_upload(master)?;
        let mut scratch = backend.dev_alloc_zeros(expected)?;
        backend.repack_training_salt(&d_master, &mut scratch, &mut self.inner)?;
        self.prepared = true;
        self.source = None;
        Ok(())
    }

    /// Mark codes/scales stale before an out-of-band master update.
    pub fn mark_stale(&mut self) {
        self.prepared = false;
    }

    /// Whether the handle may be inserted into a new device tape.
    #[must_use]
    pub fn is_prepared(&self) -> bool {
        self.ensure_prepared().is_ok()
    }

    /// Validate that this prepared handle was packed from `master`'s current generation.
    pub fn validate_current_master(&self, master: &DeviceTensor) -> Result<(), BackendError> {
        self.ensure_bound_to(master)
    }

    /// Output rows (`N`, or vocabulary size for a tied embedding).
    #[must_use]
    pub fn rows(&self) -> usize {
        self.inner.rows()
    }

    /// Contraction columns (`K`, or embedding dimension).
    #[must_use]
    pub fn cols(&self) -> usize {
        self.inner.cols()
    }

    /// Number of residual ternary planes.
    #[must_use]
    pub fn planes(&self) -> usize {
        self.inner.planes()
    }

    /// Compact 2-bit code bytes, excluding f32 scales.
    #[must_use]
    pub fn packed_bytes(&self) -> usize {
        self.inner.packed_bytes()
    }

    /// External f32 scale bytes.
    #[must_use]
    pub fn scale_bytes(&self) -> usize {
        self.inner.scale_bytes()
    }

    /// Total device bytes owned by codes and scales.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.inner.resident_bytes()
    }

    fn ensure_prepared(&self) -> Result<(), BackendError> {
        if !self.prepared {
            return Err(BackendError::InvalidInput(
                "packed SALT weight is stale; repack from the updated master".into(),
            ));
        }
        if let Some(source) = &self.source
            && source.version.generation.load(Ordering::Acquire) != source.packed_generation
        {
            return Err(BackendError::InvalidInput(
                "packed SALT weight predates the current resident master generation; repack it"
                    .into(),
            ));
        }
        Ok(())
    }

    fn ensure_bound_to(&self, master: &DeviceTensor) -> Result<(), BackendError> {
        self.ensure_prepared()?;
        let source = self.source.as_ref().ok_or_else(|| {
            BackendError::InvalidInput(
                "packed SALT weight is not identity-bound to a resident master".into(),
            )
        })?;
        if !Arc::ptr_eq(&source.version, &master.version) {
            return Err(BackendError::InvalidInput(
                "packed SALT weight is bound to a different resident master".into(),
            ));
        }
        Ok(())
    }
}

enum DeviceValue<'a> {
    Owned(CudaSlice<f32>),
    Borrowed(&'a CudaSlice<f32>),
    GradientOnly,
}

impl DeviceValue<'_> {
    fn as_slice(&self) -> &CudaSlice<f32> {
        match self {
            Self::Owned(buf) => buf,
            Self::Borrowed(buf) => buf,
            Self::GradientOnly => unreachable!("gradient-only leaves have no forward buffer"),
        }
    }
}

/// Gradient-slot memory observed during a device reverse pass.
///
/// Both values count f32 elements in graph-owned gradient slots. Temporary
/// buffers local to one VJP kernel are excluded. The peak includes the current
/// output gradient until its VJP completes, even though that buffer has been
/// taken out of its slot.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DeviceBackwardStats {
    /// Elements needed by the old strategy: one persistent gradient buffer for
    /// every value recorded on the tape.
    pub naive_all_value_grad_elements: usize,
    /// Maximum simultaneously live persistent gradient elements with lazy
    /// slots and output-gradient release.
    pub peak_persistent_grad_elements: usize,
    /// Non-leaf activation elements retained by a naive keep-all tape.
    pub naive_activation_elements: usize,
    /// Maximum simultaneously resident non-leaf activation elements across
    /// the original forward, replay, and reverse pass.
    pub peak_live_activation_elements: usize,
    /// Activation elements held at materialized checkpoint frontiers when the
    /// reverse pass began.
    pub retained_checkpoint_elements: usize,
    /// Forward operations recomputed while replaying evicted segments.
    pub recomputed_ops: usize,
}

impl DeviceBackwardStats {
    /// Persistent gradient elements avoided at the measured peak.
    #[must_use]
    pub fn saved_grad_elements(self) -> usize {
        self.naive_all_value_grad_elements
            .saturating_sub(self.peak_persistent_grad_elements)
    }

    /// Non-leaf activation elements avoided at the measured peak.
    #[must_use]
    pub fn saved_activation_elements(self) -> usize {
        self.naive_activation_elements
            .saturating_sub(self.peak_live_activation_elements)
    }
}

struct DeviceBackwardResult {
    grads: Vec<Option<CudaSlice<f32>>>,
    stats: DeviceBackwardStats,
}

/// Device-resident gradients returned in the exact order requested from
/// [`DeviceTape::xent_backward_device`].
pub struct DeviceGradients {
    bufs: Vec<CudaSlice<f32>>,
    stats: DeviceBackwardStats,
}

impl core::fmt::Debug for DeviceGradients {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DeviceGradients")
            .field("count", &self.bufs.len())
            .field("backward_stats", &self.stats)
            .finish_non_exhaustive()
    }
}

impl DeviceGradients {
    /// Number of requested gradient tensors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bufs.len()
    }

    /// Whether no gradients were requested.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bufs.is_empty()
    }

    /// Gradient-slot liveness diagnostics for the reverse pass that produced
    /// these tensors.
    #[must_use]
    pub fn backward_stats(&self) -> DeviceBackwardStats {
        self.stats
    }

    /// Download one gradient for validation or diagnostics.
    pub fn download(&self, backend: &CudaBackend, index: usize) -> Result<Vec<f32>, BackendError> {
        let buf = self.bufs.get(index).ok_or_else(|| {
            BackendError::InvalidInput(format!("gradient index {index} is out of range"))
        })?;
        if !backend.same_context(buf) {
            return Err(BackendError::InvalidInput(
                "device gradient belongs to a different CUDA context".into(),
            ));
        }
        let mut host = vec![0.0; buf.len()];
        backend.dev_download(buf, &mut host)?;
        Ok(host)
    }
}

/// Map one requested tape leaf to one optimizer-sink parameter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GradientLeafBinding {
    pub leaf_id: usize,
    pub parameter_index: usize,
}

/// One finalized leaf gradient emitted during reverse traversal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GradientEmission {
    pub sequence: usize,
    pub leaf_id: usize,
    pub parameter_index: usize,
    pub elements: usize,
}

/// In-place transform applied to each finalized resident leaf gradient before
/// the host-offloaded optimizer consumes it.
///
/// Implementations must preserve the allocation length and CUDA context. The
/// NCCL training path uses this seam for an `Avg` all-reduce; local training
/// uses the identity transform.
pub(crate) trait FinalizedGradientTransform {
    /// Transform one gradient in deterministic stream-manifest order.
    fn transform(
        &mut self,
        emission: GradientEmission,
        gradient: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError>;
}

struct IdentityGradientTransform;

impl FinalizedGradientTransform for IdentityGradientTransform {
    fn transform(
        &mut self,
        _emission: GradientEmission,
        _gradient: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        Ok(())
    }
}

pub(crate) trait GradientOptimizerSink {
    fn backend(&self) -> &CudaBackend;
    fn parameter_count(&self) -> usize;
    fn parameter_len(&self, index: usize) -> Result<usize, BackendError>;
    fn validate_stream_step(&self, step: u64) -> Result<(), BackendError>;
    fn apply_finalized_gradient(
        &mut self,
        parameter_index: usize,
        gradient: &CudaSlice<f32>,
        step: u64,
    ) -> Result<(), BackendError>;
    fn abort_gradient_stream(&mut self);
    fn finish_gradient_stream(
        &mut self,
        step: u64,
        materialized_gradient_elements: usize,
    ) -> Result<(), BackendError>;
}

/// Result of a streamed backward-and-offload step.
#[derive(Clone, Debug, PartialEq)]
pub struct GradientStreamReport {
    pub emissions: Vec<GradientEmission>,
    pub materialized_collection_elements: usize,
    pub peak_live_requested_gradient_elements: usize,
    pub backward_stats: DeviceBackwardStats,
}

struct GradientCompletionPlan {
    bindings: Vec<GradientLeafBinding>,
    binding_by_leaf: Vec<Option<usize>>,
    complete_at: Vec<Vec<usize>>,
    unused: Vec<usize>,
    remaining_edges: Vec<usize>,
    materialized_collection_elements: usize,
}

impl GradientCompletionPlan {
    fn manifest(&self, lens: &[usize]) -> Vec<GradientEmission> {
        let mut manifest = Vec::with_capacity(self.bindings.len());
        for group in self.complete_at.iter().rev() {
            for &binding_index in group {
                let binding = self.bindings[binding_index];
                manifest.push(GradientEmission {
                    sequence: manifest.len(),
                    leaf_id: binding.leaf_id,
                    parameter_index: binding.parameter_index,
                    elements: lens[binding.leaf_id],
                });
            }
        }
        for &binding_index in &self.unused {
            let binding = self.bindings[binding_index];
            manifest.push(GradientEmission {
                sequence: manifest.len(),
                leaf_id: binding.leaf_id,
                parameter_index: binding.parameter_index,
                elements: lens[binding.leaf_id],
            });
        }
        manifest
    }
}

/// Host description of one trainable SALT weight used by [`DeviceTrainer`] or
/// [`HostOffloadTrainer`].
#[derive(Clone, Copy, Debug)]
pub struct DeviceTrainParam<'a> {
    /// Initial latent f32 master.
    pub master: &'a [f32],
    /// Matrix geometry (`master.len() == rows * cols`).
    pub rows: usize,
    pub cols: usize,
    /// Number of residual ternary planes (`1..=3`).
    pub salt_planes: usize,
    /// Per-parameter AdamW configuration.
    pub optimizer: AdamW,
}

/// Persistent weight storage selected when constructing a [`DeviceTrainer`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DeviceTrainerWeightStorage {
    /// Retain one dense f32 SALT reconstruction per parameter.
    #[default]
    DenseQuantized,
    /// Retain compact packed SALT handles outside the trainer and omit dense
    /// quantized tensors. Packing reuses one largest-leaf residual scratch.
    Packed,
}

/// Precision of the resident AdamW moment state (Lever 5).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MomentPrecision {
    /// Full f32 first/second moments — the bit-exact default.
    #[default]
    F32,
    /// Block-wise int8 moments (signed `m`, sqrt-space unsigned `v`) via `adamw_step_8bit`,
    /// a 4× shrink of the moment state. Mirrors `tritium_train::Int8AdamW`.
    Int8,
}

/// Precision of the resident AdamW latent master (Lever 5).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MasterPrecision {
    /// Full f32 master — the default.
    #[default]
    F32,
    /// The master's *values* are confined to the bf16 grid with stochastic rounding after every step
    /// (numerically identical to storing the master in a u16 bf16 buffer and dequantizing it for the
    /// SALT reconstruction — so this validates a bf16 master's recovery impact without swapping the
    /// storage type through the reconstruction path; the u16 VRAM-halving swap is a mechanical
    /// follow-up gated on this result).
    Bf16,
}

/// Stochastic-rounding base seed for the bf16 master grid; the step index is folded in per step so
/// every round draws fresh dither. Fixed so a run is reproducible.
const BF16_MASTER_SEED: u64 = 0x6D61_7374_6572_5652;

/// Resident block-wise int8 AdamW moment state (Lever 5): the two moments quantized plus their
/// per-block scales. Present only when a [`DeviceTrainer`] is built with [`MomentPrecision::Int8`].
struct Int8Moments {
    m_q: CudaSlice<i8>,
    v_q: CudaSlice<u8>,
    m_scale: CudaSlice<f32>,
    v_scale: CudaSlice<f32>,
}

/// Owned host description of one trainable SALT weight.
///
/// Passing this to [`HostOffloadTrainer::new_owned`] moves the latent master
/// directly into optimizer state. Large campaigns use this seam to avoid an
/// otherwise permanent full-model adapter copy.
#[derive(Debug)]
pub struct HostOffloadTrainParam {
    /// Initial latent f32 master.
    pub master: Vec<f32>,
    /// Matrix geometry (`master.len() == rows * cols`).
    pub rows: usize,
    pub cols: usize,
    /// Number of residual ternary planes (`1..=3`).
    pub salt_planes: usize,
    /// Per-parameter AdamW configuration.
    pub optimizer: AdamW,
}

/// How the plane scales are chosen.
///
/// Two genuinely different fitters, not a tuning knob:
///
/// * [`Itf`](Self::Itf) — greedy residual expansion, each plane's scale fitted to what the previous
///   left, then refined toward its least-squares optimum. The scales are free, and measured on real
///   weights they drift *upward* (0.41, 0.42, 0.58, 0.70, eventually past 1.0), so each added plane
///   buys about 1 dB against the 9.54 dB the rate allows.
/// * [`Geometric`](Self::Geometric) — the scales are pinned to `s_p = s0·3^-p`, i.e. balanced
///   ternary, which makes the reachable levels a uniform grid over every integer in `±(3^T-1)/2`.
///   Measured 9.55–9.58 dB per plane, reaching fp parity at T=4; O(T) per weight instead of a
///   per-plane residual pass; and one stored anchor per group instead of `T` scales.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaltLadder {
    /// Greedy residual expansion with `iters` ITF alternations per plane (`0` = plain AbsMean).
    Itf {
        /// Alternations per plane. Each moves the scale to `s* = <r,t>/<t,t>` and re-rounds,
        /// accepting only on a strict SSE improvement, so more is never worse. `5` is what the PTQ
        /// sweeps used. Also drives the host rotation-mask decision, which must agree with the
        /// device fit.
        iters: usize,
    },
    /// Balanced-ternary ladder. `grid` is the number of deterministic `Δ` candidates swept per
    /// group (`0`/`1` = the single clipping-free step; `16` spans a 16× range and is the usual
    /// choice — the search is load-bearing on heavy-tailed groups).
    Geometric {
        /// Number of `Δ` candidates.
        grid: usize,
    },
}

impl SaltLadder {
    /// Plane counts this fitter admits. The ITF path is capped at 3 by the SALT V2 format and the
    /// `3^T` joint enumeration; the ladder is O(T) and runs to 9.
    #[must_use]
    pub fn plane_range(self) -> core::ops::RangeInclusive<usize> {
        match self {
            SaltLadder::Itf { .. } => 1..=3,
            SaltLadder::Geometric { .. } => 1..=9,
        }
    }
}

/// The SALT fitter configuration a [`DeviceTrainer`] reconstructs with (the PTQ campaign's stack:
/// finer scale groups + per-group Hadamard). Absent means the legacy per-row AbsMean path.
///
/// Measured on SmolLM2-135M PTQ, this stack is worth ~6 200× over the per-row default, so a
/// distillation run that reconstructs the old way starts from a far worse student than it needs to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SaltGrouping {
    /// Which scale rule the fitter uses. See [`SaltLadder`].
    pub ladder: SaltLadder,
    /// Weights per scale. `256` matches the deployed TQ2_0 block; `128` the PTQ convention.
    pub group: usize,
    /// Rotation policy. Decided **once on the host** at construction — see [`ste::rotation_mask`].
    pub rotation: ste::RotationPolicy,
}

struct ResidentTrainParam {
    master: DeviceTensor,
    quantized: Option<DeviceTensor>,
    /// One byte per scale group (`1` = rotate), fixed at construction. `None` when ungrouped.
    ///
    /// Deliberately *not* re-derived per step: a rotation bit that flipped as the master drifted
    /// would make the loss surface discontinuous, and the deployed format carries one fixed bit per
    /// group regardless.
    rotate: Option<CudaSlice<u8>>,
    m: CudaSlice<f32>,
    v: CudaSlice<f32>,
    /// Opt-in block-wise int8 moments (Lever 5). When present, `step` routes through the int8 kernel
    /// and `m`/`v` are unused; when `None`, the f32 `m`/`v` above are the moment state. Kept additive
    /// so the f32 checkpoint/inspection/offload paths stay byte-identical.
    int8: Option<Int8Moments>,
    rows: usize,
    cols: usize,
    salt_planes: usize,
    optimizer: AdamW,
}

#[derive(Clone, Debug)]
struct ResidentLoadState {
    step: u64,
    total: usize,
    next_offsets: [usize; 3],
}

/// Persistent device-state geometry owned by a [`DeviceTrainer`].
///
/// Packed SALT handles, tape activations, and input gradients are intentionally
/// excluded because their lifetimes are controlled by the campaign and tape.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResidentTrainerStats {
    /// Latent f32 master elements.
    pub parameter_elements: usize,
    /// Dense f32 SALT reconstruction elements.
    pub quantized_elements: usize,
    /// Reusable f32 SALT residual scratch elements.
    pub residual_elements: usize,
    /// First- and second-moment f32 elements.
    pub optimizer_elements: usize,
    /// Sum of all persistent f32 elements owned by the trainer.
    pub resident_elements: usize,
    /// Largest individual parameter leaf.
    pub largest_parameter_elements: usize,
}

/// Owns latent masters, SALT reconstructions, and AdamW moments in VRAM across
/// training steps.  The autograd graph remains the separate [`DeviceTape`].
pub struct DeviceTrainer<'a> {
    backend: &'a CudaBackend,
    params: Vec<ResidentTrainParam>,
    residual: CudaSlice<f32>,
    leaf_lens: Vec<usize>,
    quantized_prepared: bool,
    completed_step: u64,
    stats: ResidentTrainerStats,
    poisoned: bool,
    loading: Option<ResidentLoadState>,
    /// Lever 5: when `Bf16`, each step confines the master to the bf16 grid with stochastic rounding.
    master_precision: MasterPrecision,
    /// Sherry annealed fp residual mixed into each reconstruction; `0.0` = pure ternary.
    sherry_alpha: f32,
    /// Grouped/rotated SALT fitter, or `None` for the legacy per-row AbsMean reconstruction.
    grouping: Option<SaltGrouping>,
}

impl core::fmt::Debug for DeviceTrainer<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DeviceTrainer")
            .field("parameter_count", &self.params.len())
            .field("completed_step", &self.completed_step)
            .field("poisoned", &self.poisoned)
            .finish_non_exhaustive()
    }
}

impl<'a> DeviceTrainer<'a> {
    /// Upload all masters once and allocate dense quantized plus optimizer state.
    pub fn new(
        backend: &'a CudaBackend,
        params: &[DeviceTrainParam<'_>],
    ) -> Result<Self, BackendError> {
        Self::new_with_weight_storage(backend, params, DeviceTrainerWeightStorage::DenseQuantized)
    }

    /// Construct a resident trainer with an explicit dense or packed weight
    /// storage contract (f32 moments).
    pub fn new_with_weight_storage(
        backend: &'a CudaBackend,
        params: &[DeviceTrainParam<'_>],
        weight_storage: DeviceTrainerWeightStorage,
    ) -> Result<Self, BackendError> {
        Self::new_with_options(
            backend,
            params,
            weight_storage,
            MomentPrecision::F32,
            MasterPrecision::F32,
        )
    }

    /// Construct a resident trainer choosing the weight storage, the AdamW moment precision, and the
    /// master precision (Lever 5). [`MomentPrecision::Int8`] holds the moments block-wise int8 via
    /// `adamw_step_8bit`; [`MasterPrecision::Bf16`] confines the master to the bf16 grid with
    /// stochastic rounding each step. The f32 `m`/`v` buffers are still allocated (so
    /// checkpoint/inspection stay valid) but unused when int8 moments are active.
    pub fn new_with_options(
        backend: &'a CudaBackend,
        params: &[DeviceTrainParam<'_>],
        weight_storage: DeviceTrainerWeightStorage,
        moment_precision: MomentPrecision,
        master_precision: MasterPrecision,
    ) -> Result<Self, BackendError> {
        Self::new_with_fitter(
            backend,
            params,
            weight_storage,
            moment_precision,
            master_precision,
            None,
        )
    }

    /// As [`new_with_options`](Self::new_with_options), plus the SALT fitter the student is
    /// reconstructed with each step. `None` keeps the legacy per-row AbsMean path bit-for-bit;
    /// `Some` switches to grouped scales with an optional per-group Hadamard rotation.
    ///
    /// The rotation mask is computed **here, once, on the host** from the initial masters (see
    /// [`SaltGrouping::rotation`]) and then only read by the device kernel.
    pub fn new_with_fitter(
        backend: &'a CudaBackend,
        params: &[DeviceTrainParam<'_>],
        weight_storage: DeviceTrainerWeightStorage,
        moment_precision: MomentPrecision,
        master_precision: MasterPrecision,
        grouping: Option<SaltGrouping>,
    ) -> Result<Self, BackendError> {
        if let Some(g) = grouping {
            if g.group == 0 || g.group > 256 {
                return Err(BackendError::InvalidInput(format!(
                    "SALT scale group must be in 1..=256, got {}",
                    g.group
                )));
            }
            match g.ladder {
                SaltLadder::Itf { iters } if iters > 16 => {
                    return Err(BackendError::InvalidInput(format!(
                        "SALT ITF iterations must be <= 16 (they converge in a handful), got {iters}"
                    )));
                }
                // The ladder's grid is a fixed candidate set, and beyond ~16 the steps are finer
                // than the fit can use; keeping a bound stops a caller burning a per-step sweep.
                SaltLadder::Geometric { grid } if grid > 32 => {
                    return Err(BackendError::InvalidInput(format!(
                        "geometric Δ grid must be <= 32, got {grid}"
                    )));
                }
                _ => {}
            }
        }
        let mut leaf_lens = Vec::with_capacity(params.len());
        let mut parameter_elements = 0usize;
        let mut largest_parameter_elements = 0usize;
        for (index, param) in params.iter().enumerate() {
            let len = param.rows.checked_mul(param.cols).ok_or_else(|| {
                BackendError::InvalidInput(format!("parameter {index} shape overflows usize"))
            })?;
            if param.master.len() != len {
                return Err(BackendError::ShapeMismatch {
                    expected: len,
                    got: param.master.len(),
                });
            }
            // The cap is the FITTER's, not a global constant: the ITF path is bounded at 3 by the
            // SALT V2 format and the 3^T joint enumeration, while the ladder is O(T) and runs to 9.
            // T>3 is exactly what the ladder unlocks (it reaches fp parity at T=4).
            let planes_ok = grouping.map_or(1..=3, |g: SaltGrouping| g.ladder.plane_range());
            if !planes_ok.contains(&param.salt_planes) {
                return Err(BackendError::InvalidInput(format!(
                    "parameter {index} SALT planes must be in {:?}..={:?}",
                    planes_ok.start(),
                    planes_ok.end()
                )));
            }
            parameter_elements = parameter_elements.checked_add(len).ok_or_else(|| {
                BackendError::InvalidInput("resident parameter elements overflow usize".into())
            })?;
            largest_parameter_elements = largest_parameter_elements.max(len);
            leaf_lens.push(len);
        }
        let mut resident = Vec::with_capacity(params.len());
        for param in params {
            // Decide rotation once, on the host, from the initial weights. Ordering is
            // load-bearing: this reads `param.master`, the caller's host slice, and must run
            // before the upload below takes ownership of the device copy the kernel will fit.
            // The mask must be chosen with the SAME fitter the device kernel runs, so `g.iters`
            // feeds both this decision and the kernel launch.
            let rotate = match grouping {
                Some(g) if g.rotation != ste::RotationPolicy::Never => {
                    // Same fitter for the mask as for the fit. Deriving the mask from one fitter and
                    // applying it to another is the train/eval quantizer mismatch that invalidated a
                    // published conclusion once already (task #76).
                    let mask = match g.ladder {
                        SaltLadder::Itf { iters } => ste::rotation_mask(
                            param.master,
                            param.rows,
                            param.cols,
                            param.salt_planes,
                            g.group,
                            iters,
                            g.rotation,
                        ),
                        SaltLadder::Geometric { grid } => ste::geometric_rotation_mask(
                            param.master,
                            param.rows,
                            param.cols,
                            param.salt_planes,
                            g.group,
                            grid,
                            g.rotation,
                        ),
                    };
                    Some(backend.dev_upload_u8(&mask)?)
                }
                _ => None,
            };
            resident.push(ResidentTrainParam {
                master: DeviceTensor::upload(backend, param.master)?,
                rotate,
                quantized: match weight_storage {
                    DeviceTrainerWeightStorage::DenseQuantized => Some(DeviceTensor {
                        buf: backend.dev_alloc_zeros(param.master.len())?,
                        version: Arc::new(DeviceTensorVersion {
                            generation: AtomicU64::new(0),
                        }),
                    }),
                    DeviceTrainerWeightStorage::Packed => None,
                },
                m: backend.dev_alloc_zeros(param.master.len())?,
                v: backend.dev_alloc_zeros(param.master.len())?,
                int8: match moment_precision {
                    MomentPrecision::F32 => None,
                    MomentPrecision::Int8 => {
                        let len = param.master.len();
                        let nblocks = len.div_ceil(tritium_train::INT8_ADAM_BLOCK);
                        Some(Int8Moments {
                            m_q: backend.dev_alloc_zeros_i8(len)?,
                            v_q: backend.dev_alloc_zeros_u8(len)?,
                            m_scale: backend.dev_alloc_zeros(nblocks)?,
                            v_scale: backend.dev_alloc_zeros(nblocks)?,
                        })
                    }
                },
                rows: param.rows,
                cols: param.cols,
                salt_planes: param.salt_planes,
                optimizer: param.optimizer,
            });
        }
        if master_precision == MasterPrecision::Bf16 {
            // Confine the initial master to the bf16 grid so the first step's w_old is already on-grid
            // (matching a real bf16 master, which would dequantize from bf16 from the very first read).
            for param in &mut resident {
                backend.sr_round_to_bf16grid_dev(&mut param.master.buf, BF16_MASTER_SEED)?;
            }
        }
        let residual = backend.dev_alloc_zeros(largest_parameter_elements)?;
        let optimizer_elements = parameter_elements.checked_mul(2).ok_or_else(|| {
            BackendError::InvalidInput("resident optimizer elements overflow usize".into())
        })?;
        let quantized_elements = match weight_storage {
            DeviceTrainerWeightStorage::DenseQuantized => parameter_elements,
            DeviceTrainerWeightStorage::Packed => 0,
        };
        let resident_elements = parameter_elements
            .checked_add(quantized_elements)
            .and_then(|elements| elements.checked_add(largest_parameter_elements))
            .and_then(|elements| elements.checked_add(optimizer_elements))
            .ok_or_else(|| {
                BackendError::InvalidInput("resident state elements overflow usize".into())
            })?;
        Ok(Self {
            backend,
            params: resident,
            residual,
            leaf_lens,
            quantized_prepared: false,
            completed_step: 0,
            stats: ResidentTrainerStats {
                parameter_elements,
                quantized_elements,
                residual_elements: largest_parameter_elements,
                optimizer_elements,
                resident_elements,
                largest_parameter_elements,
            },
            poisoned: false,
            loading: None,
            master_precision,
            sherry_alpha: 0.0,
            grouping,
        })
    }

    /// Number of resident parameter leaves.
    #[must_use]
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Whether no parameter leaves are configured.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Completed optimizer step represented by the current masters and moments.
    #[must_use]
    pub fn completed_step(&self) -> u64 {
        self.completed_step
    }

    /// Stable per-parameter flattened lengths used by streaming checkpoints.
    #[must_use]
    pub fn leaf_lens(&self) -> &[usize] {
        &self.leaf_lens
    }

    /// Persistent device-state geometry owned by this trainer.
    #[must_use]
    pub fn resident_stats(&self) -> ResidentTrainerStats {
        self.stats
    }

    /// Matrix and SALT packing metadata for one resident parameter.
    pub fn parameter_metadata(
        &self,
        index: usize,
    ) -> Result<HostOffloadParamMetadata, BackendError> {
        self.ensure_usable()?;
        self.params
            .get(index)
            .map(|param| HostOffloadParamMetadata {
                rows: param.rows,
                cols: param.cols,
                salt_planes: param.salt_planes,
            })
            .ok_or_else(|| {
                BackendError::InvalidInput(format!("parameter index {index} is out of range"))
            })
    }

    /// Build one compact SALT handle directly from a resident latent master.
    ///
    /// The master never crosses the host boundary; one largest-leaf residual
    /// allocation is reused as packing scratch across every parameter.
    pub fn packed_weight(&mut self, index: usize) -> Result<DevicePackedSaltWeight, BackendError> {
        self.ensure_usable()?;
        let param = self.params.get(index).ok_or_else(|| {
            BackendError::InvalidInput(format!("parameter index {index} is out of range"))
        })?;
        let inner = self.backend.pack_training_salt(
            &param.master.buf,
            &mut self.residual,
            param.rows,
            param.cols,
            param.salt_planes,
        )?;
        Ok(DevicePackedSaltWeight {
            inner,
            prepared: true,
            source: Some(PackedMasterBinding {
                version: Arc::clone(&param.master.version),
                packed_generation: param.master.version.generation.load(Ordering::Acquire),
            }),
        })
    }

    /// Refresh a compact SALT handle directly from a resident latent master.
    ///
    /// The handle is marked stale before its geometry or device buffers are
    /// validated and becomes prepared again only after a successful repack.
    pub fn repack_packed_weight(
        &mut self,
        index: usize,
        weight: &mut DevicePackedSaltWeight,
    ) -> Result<(), BackendError> {
        self.ensure_usable()?;
        weight.prepared = false;
        let param = self.params.get(index).ok_or_else(|| {
            BackendError::InvalidInput(format!("parameter index {index} is out of range"))
        })?;
        if (weight.rows(), weight.cols(), weight.planes())
            != (param.rows, param.cols, param.salt_planes)
        {
            return Err(BackendError::InvalidInput(format!(
                "packed SALT geometry does not match resident parameter {index}"
            )));
        }
        self.backend.repack_training_salt(
            &param.master.buf,
            &mut self.residual,
            &mut weight.inner,
        )?;
        weight.source = Some(PackedMasterBinding {
            version: Arc::clone(&param.master.version),
            packed_generation: param.master.version.generation.load(Ordering::Acquire),
        });
        weight.prepared = true;
        Ok(())
    }

    fn invalidate_packed_weights(&self) {
        for parameter in &self.params {
            parameter
                .master
                .version
                .generation
                .fetch_add(1, Ordering::AcqRel);
        }
    }

    /// Borrow one resident latent master without allocation or device transfer.
    ///
    /// The returned tensor remains owned by this trainer. A tape borrowing it
    /// prevents a mutable optimizer step until that tape is dropped.
    pub fn master_tensor(&self, index: usize) -> Result<&DeviceTensor, BackendError> {
        self.ensure_usable()?;
        self.params
            .get(index)
            .map(|parameter| &parameter.master)
            .ok_or_else(|| {
                BackendError::InvalidInput(format!("parameter index {index} is out of range"))
            })
    }

    fn ensure_usable(&self) -> Result<(), BackendError> {
        if self.poisoned {
            Err(BackendError::InvalidInput(
                "resident trainer is poisoned; complete a fresh checkpoint reload or reconstruct it before reuse"
                    .into(),
            ))
        } else {
            Ok(())
        }
    }

    /// Whether an optimizer failure may have left parameter generations mixed.
    #[must_use]
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Reconstruct every resident master into its dense f32 SALT tensor.
    pub fn prepare_quantized(&mut self) -> Result<(), BackendError> {
        self.ensure_usable()?;
        // The grouped kernel has no Sherry term, so silently dropping the fp mix would make a
        // configured anneal a no-op. Refuse instead.
        if self.grouping.is_some() && self.sherry_alpha != 0.0 {
            return Err(BackendError::InvalidInput(
                "Sherry's fp residual is not implemented on the grouped SALT path; set \
                 sherry_alpha to 0 or construct the trainer without a SaltGrouping"
                    .into(),
            ));
        }
        self.quantized_prepared = false;
        for param in &mut self.params {
            let quantized = param.quantized.as_mut().ok_or_else(|| {
                BackendError::InvalidInput(
                    "dense quantized storage is disabled for this resident trainer".into(),
                )
            })?;
            match self.grouping {
                Some(g) => match g.ladder {
                    SaltLadder::Itf { iters } => self.backend.salt_quantize_forward_grouped_dev(
                        &param.master.buf,
                        &mut quantized.buf,
                        param.rotate.as_ref(),
                        param.rows,
                        param.cols,
                        param.salt_planes,
                        g.group,
                        iters,
                    )?,
                    SaltLadder::Geometric { grid } => {
                        self.backend.salt_quantize_forward_grouped_geometric_dev(
                            &param.master.buf,
                            &mut quantized.buf,
                            param.rotate.as_ref(),
                            param.rows,
                            param.cols,
                            param.salt_planes,
                            g.group,
                            grid,
                        )?
                    }
                },
                None => self.backend.salt_quantize_forward_sherry_dev(
                    &param.master.buf,
                    &mut self.residual,
                    &mut quantized.buf,
                    param.rows,
                    param.cols,
                    param.salt_planes,
                    self.sherry_alpha,
                )?,
            }
        }
        self.quantized_prepared = true;
        Ok(())
    }

    /// Set the **Sherry** fp-residual mix used by the next [`prepare_quantized`](Self::prepare_quantized):
    /// the student's weights become `(1-alpha)*Ŵ + alpha*master`. Drive it from
    /// [`ste::sherry_alpha`](tritium_train::ops::ste::sherry_alpha) so it cosine-anneals to 0, leaving a
    /// purely ternary model. `0.0` (the default) is the plain ternary path, bit-for-bit.
    pub fn set_sherry_alpha(&mut self, alpha: f32) {
        self.sherry_alpha = alpha;
    }

    /// Borrow a prepared quantized weight for zero-copy insertion into a tape.
    pub fn quantized(&self, index: usize) -> Result<&DeviceTensor, BackendError> {
        self.ensure_usable()?;
        if !self.quantized_prepared {
            return Err(BackendError::InvalidInput(
                "quantized weights are stale; call prepare_quantized first".into(),
            ));
        }
        self.params
            .get(index)
            .ok_or_else(|| {
                BackendError::InvalidInput(format!("parameter index {index} is out of range"))
            })?
            .quantized
            .as_ref()
            .ok_or_else(|| {
                BackendError::InvalidInput(
                    "dense quantized storage is disabled for this resident trainer".into(),
                )
            })
    }

    /// Apply one 1-based resident AdamW step. Gradients must be in parameter
    /// order, as returned by requesting weight leaf ids in that order. A device
    /// failure after mutation begins poisons the trainer because earlier leaves
    /// may already represent the new generation.
    pub fn step(&mut self, grads: DeviceGradients, step: u64) -> Result<(), BackendError> {
        self.ensure_usable()?;
        let expected_step = self.completed_step.checked_add(1).ok_or_else(|| {
            BackendError::InvalidInput("resident trainer completed-step counter overflowed".into())
        })?;
        if step != expected_step {
            return Err(BackendError::InvalidInput(format!(
                "resident AdamW expected step {expected_step}, got {step}"
            )));
        }
        if grads.bufs.len() != self.params.len() {
            return Err(BackendError::ShapeMismatch {
                expected: self.params.len(),
                got: grads.bufs.len(),
            });
        }
        for (param, grad) in self.params.iter().zip(&grads.bufs) {
            if !self.backend.same_context(grad) {
                return Err(BackendError::InvalidInput(
                    "device gradient belongs to a different CUDA context".into(),
                ));
            }
            if grad.len() != param.master.len() {
                return Err(BackendError::ShapeMismatch {
                    expected: param.master.len(),
                    got: grad.len(),
                });
            }
        }
        self.invalidate_packed_weights();
        self.quantized_prepared = false;
        self.poisoned = true;
        let master_precision = self.master_precision;
        for (param, grad) in self.params.iter_mut().zip(grads.bufs) {
            match &mut param.int8 {
                Some(i8m) => self.backend.adamw_step_8bit_dev(
                    &mut param.master.buf,
                    &grad,
                    &mut i8m.m_q,
                    &mut i8m.v_q,
                    &mut i8m.m_scale,
                    &mut i8m.v_scale,
                    step,
                    &param.optimizer,
                )?,
                None => self.backend.adamw_step_dev(
                    &mut param.master.buf,
                    &grad,
                    &mut param.m,
                    &mut param.v,
                    step,
                    &param.optimizer,
                )?,
            }
            if master_precision == MasterPrecision::Bf16 {
                // Confine the updated master to the bf16 grid with stochastic rounding — numerically a
                // real bf16 master. Fresh dither each step so sub-ULP updates survive in expectation.
                self.backend
                    .sr_round_to_bf16grid_dev(&mut param.master.buf, BF16_MASTER_SEED ^ step)?;
            }
        }
        self.poisoned = false;
        self.completed_step = step;
        Ok(())
    }

    /// Set the learning rate on every resident parameter's optimizer, for the **next** [`step`](Self::step).
    ///
    /// The [`LrSchedule`] (warmup + cosine decay) is deliberately outside the `Optimizer` trait, so a
    /// campaign drives it by calling this with `schedule.lr(step)` before each step. Only `lr` changes;
    /// betas/eps/weight-decay and all moment state are untouched, so a schedule can be introduced
    /// mid-run without disturbing the optimizer's history.
    pub fn set_lr(&mut self, lr: f32) {
        for param in &mut self.params {
            param.optimizer.lr = lr;
        }
    }

    /// Download one latent master for evaluation or checkpointing.
    pub fn download_master(&self, index: usize) -> Result<Vec<f32>, BackendError> {
        self.ensure_usable()?;
        let param = self.params.get(index).ok_or_else(|| {
            BackendError::InvalidInput(format!("parameter index {index} is out of range"))
        })?;
        param.master.download(self.backend)
    }
}

#[derive(Clone, Copy, Debug)]
enum ResidentStatePlane {
    Parameter,
    FirstMoment,
    SecondMoment,
}

fn resident_dcp_plane(plane: StatePlane) -> Result<(ResidentStatePlane, usize), DcpError> {
    match plane {
        StatePlane::Parameter => Ok((ResidentStatePlane::Parameter, 0)),
        StatePlane::Optimizer(0) => Ok((ResidentStatePlane::FirstMoment, 1)),
        StatePlane::Optimizer(1) => Ok((ResidentStatePlane::SecondMoment, 2)),
        StatePlane::Optimizer(_) => Err(DcpError::InvalidState(
            "resident AdamW has exactly two optimizer planes",
        )),
    }
}

fn resident_state(param: &ResidentTrainParam, plane: ResidentStatePlane) -> &CudaSlice<f32> {
    match plane {
        ResidentStatePlane::Parameter => &param.master.buf,
        ResidentStatePlane::FirstMoment => &param.m,
        ResidentStatePlane::SecondMoment => &param.v,
    }
}

fn resident_state_mut(
    param: &mut ResidentTrainParam,
    plane: ResidentStatePlane,
) -> &mut CudaSlice<f32> {
    match plane {
        ResidentStatePlane::Parameter => &mut param.master.buf,
        ResidentStatePlane::FirstMoment => &mut param.m,
        ResidentStatePlane::SecondMoment => &mut param.v,
    }
}

impl CudaBackend {
    fn resident_state_download_range(
        &self,
        device: &CudaSlice<f32>,
        offset: usize,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        if !self.same_context(device) {
            return Err(BackendError::InvalidInput(
                "resident checkpoint source belongs to a different CUDA context".into(),
            ));
        }
        let end = offset.checked_add(out.len()).ok_or_else(|| {
            BackendError::InvalidInput("resident checkpoint source range overflows usize".into())
        })?;
        if end > device.len() {
            return Err(BackendError::ShapeMismatch {
                expected: end,
                got: device.len(),
            });
        }
        if out.is_empty() {
            return Ok(());
        }
        let stream = Arc::clone(device.stream());
        let source = device.try_slice(offset..end).ok_or_else(|| {
            BackendError::InvalidInput("resident checkpoint source range is invalid".into())
        })?;
        stream.memcpy_dtoh(&source, out).map_err(|error| {
            BackendError::Backend(format!("resident checkpoint dtoh failed: {error}"))
        })?;
        stream.synchronize().map_err(|error| {
            BackendError::Backend(format!(
                "resident checkpoint dtoh synchronization failed: {error}"
            ))
        })
    }

    fn resident_state_upload_range(
        &self,
        device: &mut CudaSlice<f32>,
        offset: usize,
        values: &[f32],
    ) -> Result<(), BackendError> {
        if !self.same_context(device) {
            return Err(BackendError::InvalidInput(
                "resident checkpoint sink belongs to a different CUDA context".into(),
            ));
        }
        let end = offset.checked_add(values.len()).ok_or_else(|| {
            BackendError::InvalidInput("resident checkpoint sink range overflows usize".into())
        })?;
        if end > device.len() {
            return Err(BackendError::ShapeMismatch {
                expected: end,
                got: device.len(),
            });
        }
        if values.is_empty() {
            return Ok(());
        }
        let stream = Arc::clone(device.stream());
        let mut destination = device.try_slice_mut(offset..end).ok_or_else(|| {
            BackendError::InvalidInput("resident checkpoint sink range is invalid".into())
        })?;
        stream
            .memcpy_htod(values, &mut destination)
            .map_err(|error| {
                BackendError::Backend(format!("resident checkpoint htod failed: {error}"))
            })?;
        stream.synchronize().map_err(|error| {
            BackendError::Backend(format!(
                "resident checkpoint htod synchronization failed: {error}"
            ))
        })
    }
}

impl GradientOptimizerSink for DeviceTrainer<'_> {
    fn backend(&self) -> &CudaBackend {
        self.backend
    }

    fn parameter_count(&self) -> usize {
        self.params.len()
    }

    fn parameter_len(&self, index: usize) -> Result<usize, BackendError> {
        self.params
            .get(index)
            .map(|parameter| parameter.master.len())
            .ok_or_else(|| {
                BackendError::InvalidInput(format!(
                    "streamed parameter index {index} is out of range"
                ))
            })
    }

    fn validate_stream_step(&self, step: u64) -> Result<(), BackendError> {
        self.ensure_usable()?;
        let expected = self.completed_step.checked_add(1).ok_or_else(|| {
            BackendError::InvalidInput("resident trainer completed-step counter overflowed".into())
        })?;
        if step != expected {
            return Err(BackendError::InvalidInput(format!(
                "resident AdamW expected step {expected}, got {step}"
            )));
        }
        Ok(())
    }

    fn apply_finalized_gradient(
        &mut self,
        parameter_index: usize,
        gradient: &CudaSlice<f32>,
        step: u64,
    ) -> Result<(), BackendError> {
        let parameter = self.params.get(parameter_index).ok_or_else(|| {
            BackendError::InvalidInput(format!(
                "streamed parameter index {parameter_index} is out of range"
            ))
        })?;
        if !self.backend.same_context(gradient) {
            return Err(BackendError::InvalidInput(format!(
                "streamed gradient {parameter_index} belongs to a different CUDA context"
            )));
        }
        if gradient.len() != parameter.master.len() {
            return Err(BackendError::ShapeMismatch {
                expected: parameter.master.len(),
                got: gradient.len(),
            });
        }
        if !self.poisoned {
            self.invalidate_packed_weights();
        }
        self.quantized_prepared = false;
        self.poisoned = true;
        let parameter = &mut self.params[parameter_index];
        self.backend.adamw_step_dev(
            &mut parameter.master.buf,
            gradient,
            &mut parameter.m,
            &mut parameter.v,
            step,
            &parameter.optimizer,
        )
    }

    fn abort_gradient_stream(&mut self) {
        self.poisoned = true;
    }

    fn finish_gradient_stream(
        &mut self,
        step: u64,
        _materialized_gradient_elements: usize,
    ) -> Result<(), BackendError> {
        self.completed_step = step;
        self.poisoned = false;
        Ok(())
    }
}

impl tritium_train::dcp::StateSource for DeviceTrainer<'_> {
    fn step(&self) -> u64 {
        self.completed_step
    }

    fn leaf_lens(&self) -> &[usize] {
        &self.leaf_lens
    }

    fn plane_count(&self) -> usize {
        2
    }

    fn read_chunk(
        &mut self,
        plane: StatePlane,
        offset: usize,
        out: &mut [f32],
    ) -> Result<(), DcpError> {
        if self.poisoned {
            return Err(DcpError::InvalidState(
                "cannot checkpoint a poisoned resident trainer",
            ));
        }
        let (plane, _) = resident_dcp_plane(plane)?;
        let total = self.stats.parameter_elements;
        let end = offset
            .checked_add(out.len())
            .ok_or(DcpError::InvalidState("source range overflows usize"))?;
        if end > total {
            return Err(DcpError::InvalidState("source range out of bounds"));
        }

        let mut global_start = 0usize;
        let mut copied = 0usize;
        let mut position = offset;
        for param in &self.params {
            let state = resident_state(param, plane);
            let global_end = global_start + state.len();
            if position < global_end && copied < out.len() {
                let local_start = position.saturating_sub(global_start);
                let count = (state.len() - local_start).min(out.len() - copied);
                self.backend
                    .resident_state_download_range(
                        state,
                        local_start,
                        &mut out[copied..copied + count],
                    )
                    .map_err(|error| DcpError::Io(error.to_string()))?;
                copied += count;
                position += count;
            }
            global_start = global_end;
        }
        if copied != out.len() {
            return Err(DcpError::InvalidState(
                "source range was not fully supplied",
            ));
        }
        Ok(())
    }
}

impl tritium_train::dcp::StateSink for DeviceTrainer<'_> {
    fn begin(
        &mut self,
        step: u64,
        leaf_lens: &[usize],
        plane_count: usize,
    ) -> Result<(), DcpError> {
        // A load may overwrite part of device state before it fails. Poisoning
        // makes that partial generation unavailable until a complete fresh load.
        self.invalidate_packed_weights();
        self.poisoned = true;
        self.quantized_prepared = false;
        self.loading = None;
        if leaf_lens != self.leaf_lens {
            return Err(DcpError::InvalidState(
                "checkpoint leaf layout does not match resident trainer",
            ));
        }
        if plane_count != 2 {
            return Err(DcpError::InvalidState(
                "resident AdamW requires exactly two optimizer planes",
            ));
        }
        self.loading = Some(ResidentLoadState {
            step,
            total: self.stats.parameter_elements,
            next_offsets: [0; 3],
        });
        Ok(())
    }

    fn write_chunk(
        &mut self,
        plane: StatePlane,
        offset: usize,
        values: &[f32],
    ) -> Result<(), DcpError> {
        self.poisoned = true;
        self.quantized_prepared = false;
        let (plane, plane_index) = resident_dcp_plane(plane)?;
        let loading = self.loading.as_ref().ok_or(DcpError::InvalidState(
            "resident checkpoint load has not begun",
        ))?;
        if offset != loading.next_offsets[plane_index] {
            return Err(DcpError::InvalidState(
                "checkpoint chunks must be contiguous and ordered per plane",
            ));
        }
        let end = offset
            .checked_add(values.len())
            .ok_or(DcpError::InvalidState("sink range overflows usize"))?;
        if end > loading.total {
            return Err(DcpError::InvalidState("sink range out of bounds"));
        }

        let mut global_start = 0usize;
        let mut copied = 0usize;
        let mut position = offset;
        for param in &mut self.params {
            let state = resident_state_mut(param, plane);
            let global_end = global_start + state.len();
            if position < global_end && copied < values.len() {
                let local_start = position.saturating_sub(global_start);
                let count = (state.len() - local_start).min(values.len() - copied);
                self.backend
                    .resident_state_upload_range(
                        state,
                        local_start,
                        &values[copied..copied + count],
                    )
                    .map_err(|error| DcpError::Io(error.to_string()))?;
                copied += count;
                position += count;
            }
            global_start = global_end;
        }
        if copied != values.len() {
            return Err(DcpError::InvalidState("sink range was not fully stored"));
        }
        self.loading
            .as_mut()
            .expect("load state was validated above")
            .next_offsets[plane_index] = end;
        Ok(())
    }

    fn finish(&mut self) -> Result<(), DcpError> {
        self.poisoned = true;
        self.quantized_prepared = false;
        let loading = self.loading.take().ok_or(DcpError::InvalidState(
            "resident checkpoint load has not begun",
        ))?;
        if loading
            .next_offsets
            .iter()
            .any(|&offset| offset != loading.total)
        {
            return Err(DcpError::InvalidState(
                "resident checkpoint load is incomplete",
            ));
        }
        self.completed_step = loading.step;
        self.poisoned = false;
        Ok(())
    }
}

struct HostOffloadParam {
    master: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
    rows: usize,
    cols: usize,
    salt_planes: usize,
    optimizer: AdamW,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PendingOffload {
    parameter_index: usize,
    len: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OffloadSlotPhase {
    Free,
    Computing(PendingOffload),
    Downloading(PendingOffload),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OffloadTransition {
    target_slot: usize,
    reclaim: Option<PendingOffload>,
    download_slot: Option<usize>,
}

/// Pure two-slot state machine shared by collected and streamed gradients.
/// CUDA operations execute in the transition order documented by
/// [`HostOffloadTrainer`]; any failed transition resets and poisons the owner.
#[derive(Debug)]
struct DoubleBufferSchedule {
    phases: [OffloadSlotPhase; 2],
    active_compute: Option<usize>,
    next_slot: usize,
    peak_in_flight: usize,
}

impl Default for DoubleBufferSchedule {
    fn default() -> Self {
        Self {
            phases: [OffloadSlotPhase::Free; 2],
            active_compute: None,
            next_slot: 0,
            peak_in_flight: 0,
        }
    }
}

impl DoubleBufferSchedule {
    fn enqueue(&mut self, pending: PendingOffload) -> Result<OffloadTransition, BackendError> {
        let target_slot = self.next_slot;
        let reclaim = match self.phases[target_slot] {
            OffloadSlotPhase::Free => None,
            OffloadSlotPhase::Downloading(pending) => Some(pending),
            OffloadSlotPhase::Computing(_) => {
                return Err(BackendError::InvalidInput(
                    "host-offload double buffer attempted to reuse an active compute slot".into(),
                ));
            }
        };
        let download_slot = self.active_compute.take();
        if let Some(slot) = download_slot {
            let OffloadSlotPhase::Computing(previous) = self.phases[slot] else {
                return Err(BackendError::InvalidInput(
                    "host-offload active slot is not computing".into(),
                ));
            };
            self.phases[slot] = OffloadSlotPhase::Downloading(previous);
        }
        self.phases[target_slot] = OffloadSlotPhase::Computing(pending);
        self.active_compute = Some(target_slot);
        self.next_slot ^= 1;
        self.peak_in_flight = self.peak_in_flight.max(
            self.phases
                .iter()
                .filter(|phase| !matches!(phase, OffloadSlotPhase::Free))
                .count(),
        );
        Ok(OffloadTransition {
            target_slot,
            reclaim,
            download_slot,
        })
    }

    fn begin_finish(&mut self) -> Result<Option<usize>, BackendError> {
        let Some(slot) = self.active_compute.take() else {
            return Ok(None);
        };
        let OffloadSlotPhase::Computing(pending) = self.phases[slot] else {
            return Err(BackendError::InvalidInput(
                "host-offload finish found an invalid active slot".into(),
            ));
        };
        self.phases[slot] = OffloadSlotPhase::Downloading(pending);
        Ok(Some(slot))
    }

    fn pending_downloads(&self) -> impl Iterator<Item = (usize, PendingOffload)> + '_ {
        self.phases
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(slot, phase)| match phase {
                OffloadSlotPhase::Downloading(pending) => Some((slot, pending)),
                OffloadSlotPhase::Free | OffloadSlotPhase::Computing(_) => None,
            })
    }

    fn reset(&mut self) {
        self.phases = [OffloadSlotPhase::Free; 2];
        self.active_compute = None;
        self.next_slot = 0;
    }
}

struct HostOffloadSlot {
    host_master: PinnedHostSlice<f32>,
    host_m: PinnedHostSlice<f32>,
    host_v: PinnedHostSlice<f32>,
    device_master: CudaSlice<f32>,
    device_m: CudaSlice<f32>,
    device_v: CudaSlice<f32>,
}

impl HostOffloadSlot {
    fn new(backend: &CudaBackend, capacity: usize) -> Result<Self, BackendError> {
        Ok(Self {
            host_master: backend.offload_alloc_pinned_zeros(capacity)?,
            host_m: backend.offload_alloc_pinned_zeros(capacity)?,
            host_v: backend.offload_alloc_pinned_zeros(capacity)?,
            device_master: backend.dev_alloc_zeros(capacity)?,
            device_m: backend.dev_alloc_zeros(capacity)?,
            device_v: backend.dev_alloc_zeros(capacity)?,
        })
    }
}

#[derive(Clone, Copy, Debug)]
enum HostOffloadStatePlane {
    Parameter,
    FirstMoment,
    SecondMoment,
}

#[derive(Clone, Debug)]
struct HostOffloadLoadState {
    step: u64,
    total: usize,
    next_offsets: [usize; 3],
}

fn host_offload_state(param: &HostOffloadParam, plane: HostOffloadStatePlane) -> &[f32] {
    match plane {
        HostOffloadStatePlane::Parameter => &param.master,
        HostOffloadStatePlane::FirstMoment => &param.m,
        HostOffloadStatePlane::SecondMoment => &param.v,
    }
}

fn host_offload_state_mut(
    param: &mut HostOffloadParam,
    plane: HostOffloadStatePlane,
) -> &mut [f32] {
    match plane {
        HostOffloadStatePlane::Parameter => &mut param.master,
        HostOffloadStatePlane::FirstMoment => &mut param.m,
        HostOffloadStatePlane::SecondMoment => &mut param.v,
    }
}

fn host_offload_dcp_plane(plane: StatePlane) -> Result<(HostOffloadStatePlane, usize), DcpError> {
    match plane {
        StatePlane::Parameter => Ok((HostOffloadStatePlane::Parameter, 0)),
        StatePlane::Optimizer(0) => Ok((HostOffloadStatePlane::FirstMoment, 1)),
        StatePlane::Optimizer(1) => Ok((HostOffloadStatePlane::SecondMoment, 2)),
        StatePlane::Optimizer(_) => Err(DcpError::InvalidState(
            "host offload AdamW has exactly two optimizer planes",
        )),
    }
}

/// Matrix and SALT packing metadata retained with an offloaded parameter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostOffloadParamMetadata {
    pub rows: usize,
    pub cols: usize,
    pub salt_planes: usize,
}

/// Static byte geometry for a packed-SALT [`HostOffloadTrainer`] campaign.
///
/// This estimate is independent of a CUDA device and covers the persistent
/// packed representation, host-resident AdamW state, the two optimizer staging
/// slots, and the dense-gradient baseline. Runtime activation and streamed
/// gradient peaks are intentionally excluded because they depend on the graph
/// executed by each training step.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HostOffloadMemoryGeometry {
    /// Packed codes plus external per-row scale bytes resident on the device.
    pub packed_parameter_bytes: usize,
    /// Two-bit SALT code payload bytes, excluding external scales.
    pub packed_code_bytes: usize,
    /// External f32 per-row SALT scale bytes.
    pub packed_scale_bytes: usize,
    /// Dense f32 latent-master bytes across every parameter leaf.
    pub dense_parameter_bytes: usize,
    /// Dense f32 bytes in the largest parameter leaf.
    pub largest_parameter_bytes: usize,
    /// Host master plus first and second Adam moment bytes.
    pub host_optimizer_bytes: usize,
    /// Device staging bytes for two master/moment slots; pinned host staging is equal.
    pub peak_optimizer_staging_bytes: usize,
    /// Full dense-gradient collection baseline used to compare streamed gradients.
    pub materialized_gradient_bytes: usize,
}

/// Compute the static memory geometry for packed-SALT host-offloaded training.
///
/// Packed codes use the resident training layout (two bits per trit, padded to
/// [`tritium_format::QK_K`] columns) and retain one f32 row scale per residual
/// plane. Host AdamW owns the latent master plus two moments. Device and
/// pinned-host optimizer staging are each double buffered, so each class owns
/// six f32 copies of the largest parameter leaf at its measured peak. The
/// returned staging field records the device class; pinned-host staging is equal.
///
/// # Errors
/// Returns [`BackendError::InvalidInput`] for an unsupported SALT plane count or
/// any shape, packed-layout, or byte-count overflow.
pub fn host_offload_memory_geometry(
    parameters: &[HostOffloadParamMetadata],
) -> Result<HostOffloadMemoryGeometry, BackendError> {
    let mut dense_elements = 0usize;
    let mut largest_parameter_elements = 0usize;
    let mut packed_code_bytes = 0usize;
    let mut packed_scale_bytes = 0usize;

    for (index, parameter) in parameters.iter().enumerate() {
        if !(1..=3).contains(&parameter.salt_planes) {
            return Err(BackendError::InvalidInput(format!(
                "parameter {index} SALT planes must be in 1..=3"
            )));
        }
        let elements = parameter.rows.checked_mul(parameter.cols).ok_or_else(|| {
            BackendError::InvalidInput(format!("parameter {index} dense shape overflows usize"))
        })?;
        dense_elements = dense_elements.checked_add(elements).ok_or_else(|| {
            BackendError::InvalidInput("dense parameter elements overflow usize".into())
        })?;
        largest_parameter_elements = largest_parameter_elements.max(elements);

        let blocks_per_row = parameter.cols.div_ceil(tritium_format::QK_K);
        let parameter_code_bytes = parameter
            .salt_planes
            .checked_mul(parameter.rows)
            .and_then(|count| count.checked_mul(blocks_per_row))
            .and_then(|count| count.checked_mul(tritium_format::QK_K / 4))
            .ok_or_else(|| {
                BackendError::InvalidInput(format!(
                    "parameter {index} packed SALT code bytes overflow usize"
                ))
            })?;
        packed_code_bytes = packed_code_bytes
            .checked_add(parameter_code_bytes)
            .ok_or_else(|| {
                BackendError::InvalidInput("packed SALT code bytes overflow usize".into())
            })?;
        let parameter_scale_bytes = parameter
            .salt_planes
            .checked_mul(parameter.rows)
            .and_then(|count| count.checked_mul(core::mem::size_of::<f32>()))
            .ok_or_else(|| {
                BackendError::InvalidInput(format!(
                    "parameter {index} packed SALT scale bytes overflow usize"
                ))
            })?;
        packed_scale_bytes = packed_scale_bytes
            .checked_add(parameter_scale_bytes)
            .ok_or_else(|| {
                BackendError::InvalidInput("packed SALT scale bytes overflow usize".into())
            })?;
    }

    let f32_bytes = |elements: usize, label: &'static str| {
        elements
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| BackendError::InvalidInput(format!("{label} bytes overflow usize")))
    };
    let dense_parameter_bytes = f32_bytes(dense_elements, "dense parameter")?;
    let largest_parameter_bytes = f32_bytes(largest_parameter_elements, "largest parameter")?;
    let host_optimizer_bytes = dense_parameter_bytes
        .checked_mul(3)
        .ok_or_else(|| BackendError::InvalidInput("host optimizer bytes overflow usize".into()))?;
    let peak_optimizer_staging_bytes =
        f32_bytes(largest_parameter_elements, "largest parameter staging")?
            .checked_mul(6)
            .ok_or_else(|| {
                BackendError::InvalidInput("optimizer staging bytes overflow usize".into())
            })?;
    let packed_parameter_bytes = packed_code_bytes
        .checked_add(packed_scale_bytes)
        .ok_or_else(|| {
            BackendError::InvalidInput("packed SALT parameter bytes overflow usize".into())
        })?;

    Ok(HostOffloadMemoryGeometry {
        packed_parameter_bytes,
        packed_code_bytes,
        packed_scale_bytes,
        dense_parameter_bytes,
        largest_parameter_bytes,
        host_optimizer_bytes,
        peak_optimizer_staging_bytes,
        materialized_gradient_bytes: dense_parameter_bytes,
    })
}

/// Deterministic logical memory accounting for [`HostOffloadTrainer`].
///
/// The two persistent slots each hold master plus two Adam moments on both the
/// device and page-locked host memory: each staging count is therefore exactly
/// `6 * largest_parameter_elements`. `resident_input_gradient_elements` is
/// reported separately because [`DeviceGradients`] may own every requested
/// gradient before the optimizer step begins.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HostOffloadStats {
    /// Host-resident master + first moment + second moment elements.
    pub host_optimizer_elements: usize,
    /// Size of the largest parameter leaf.
    pub largest_parameter_elements: usize,
    /// Persistent double-buffered optimizer-state elements on the device.
    pub peak_optimizer_device_elements: usize,
    /// Persistent page-locked optimizer-state staging elements.
    pub pinned_optimizer_host_elements: usize,
    /// Peak parameter updates concurrently computing or downloading.
    pub peak_in_flight_parameters: usize,
    /// Device gradient elements supplied to the most recent successful step.
    pub resident_input_gradient_elements: usize,
}

/// Double-buffered CPU-offloaded AdamW state.
///
/// Masters and moments remain in ordinary host vectors between steps. Each
/// parameter fills one of two persistent page-locked/device slots. A dedicated
/// transfer stream uploads the next leaf while the default stream computes the
/// prior AdamW update, then downloads that prior leaf while compute advances.
/// cudarc buffer events fence every cross-stream dependency and slot reuse.
///
/// This bounds both device and pinned staging to six times the largest leaf.
/// [`Self::step`] accepts a fully materialized [`DeviceGradients`] collection,
/// while [`DeviceTape::xent_backward_into`] emits finalized leaf gradients
/// directly and drops each after its update. Both public paths drain the
/// pipeline before returning, so host masters and moments are always coherent.
pub struct HostOffloadTrainer<'a> {
    backend: &'a CudaBackend,
    params: Vec<HostOffloadParam>,
    leaf_lens: Vec<usize>,
    transfer_stream: Arc<CudaStream>,
    slots: Option<[HostOffloadSlot; 2]>,
    schedule: DoubleBufferSchedule,
    completed_step: u64,
    stats: HostOffloadStats,
    poisoned: bool,
    loading: Option<HostOffloadLoadState>,
}

impl core::fmt::Debug for HostOffloadTrainer<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HostOffloadTrainer")
            .field("parameter_count", &self.params.len())
            .field("completed_step", &self.completed_step)
            .field("stats", &self.stats)
            .field("poisoned", &self.poisoned)
            .finish_non_exhaustive()
    }
}

impl<'a> HostOffloadTrainer<'a> {
    /// Copy initial masters to host-owned optimizer state. Parameter geometry
    /// and SALT-plane metadata are validated identically to [`DeviceTrainer`].
    pub fn new(
        backend: &'a CudaBackend,
        params: &[DeviceTrainParam<'_>],
    ) -> Result<Self, BackendError> {
        let owned = params
            .iter()
            .map(|param| HostOffloadTrainParam {
                master: param.master.to_vec(),
                rows: param.rows,
                cols: param.cols,
                salt_planes: param.salt_planes,
                optimizer: param.optimizer,
            })
            .collect();
        Self::new_owned(backend, owned)
    }

    /// Move initial masters into host-owned optimizer state without cloning.
    ///
    /// This is the production scale path. The borrowed [`Self::new`] remains
    /// available for small callers and tests that need to retain their source
    /// masters.
    pub fn new_owned(
        backend: &'a CudaBackend,
        params: Vec<HostOffloadTrainParam>,
    ) -> Result<Self, BackendError> {
        let mut host_params = Vec::with_capacity(params.len());
        let mut leaf_lens = Vec::with_capacity(params.len());
        let mut total_parameter_elements = 0usize;
        let mut largest_parameter_elements = 0usize;
        for (index, param) in params.into_iter().enumerate() {
            let len = param.rows.checked_mul(param.cols).ok_or_else(|| {
                BackendError::InvalidInput(format!("parameter {index} shape overflows usize"))
            })?;
            if param.master.len() != len {
                return Err(BackendError::ShapeMismatch {
                    expected: len,
                    got: param.master.len(),
                });
            }
            if !(1..=3).contains(&param.salt_planes) {
                return Err(BackendError::InvalidInput(format!(
                    "parameter {index} SALT planes must be in 1..=3"
                )));
            }
            total_parameter_elements =
                total_parameter_elements.checked_add(len).ok_or_else(|| {
                    BackendError::InvalidInput("total parameter elements overflow usize".into())
                })?;
            largest_parameter_elements = largest_parameter_elements.max(len);
            leaf_lens.push(len);
            host_params.push(HostOffloadParam {
                master: param.master,
                m: vec![0.0; len],
                v: vec![0.0; len],
                rows: param.rows,
                cols: param.cols,
                salt_planes: param.salt_planes,
                optimizer: param.optimizer,
            });
        }
        let host_optimizer_elements = total_parameter_elements.checked_mul(3).ok_or_else(|| {
            BackendError::InvalidInput("host optimizer elements overflow usize".into())
        })?;
        let staging_elements = largest_parameter_elements.checked_mul(6).ok_or_else(|| {
            BackendError::InvalidInput("host offload staging elements overflow usize".into())
        })?;
        // Creating the auxiliary stream switches cudarc into event-tracked
        // multi-stream mode before any persistent staging buffer is allocated.
        let transfer_stream = backend.offload_transfer_stream()?;
        let slots = if largest_parameter_elements == 0 {
            None
        } else {
            Some([
                HostOffloadSlot::new(backend, largest_parameter_elements)?,
                HostOffloadSlot::new(backend, largest_parameter_elements)?,
            ])
        };
        Ok(Self {
            backend,
            params: host_params,
            leaf_lens,
            transfer_stream,
            slots,
            schedule: DoubleBufferSchedule::default(),
            completed_step: 0,
            stats: HostOffloadStats {
                host_optimizer_elements,
                largest_parameter_elements,
                peak_optimizer_device_elements: staging_elements,
                pinned_optimizer_host_elements: staging_elements,
                peak_in_flight_parameters: 0,
                resident_input_gradient_elements: 0,
            },
            poisoned: false,
            loading: None,
        })
    }

    /// Number of offloaded parameter leaves.
    #[must_use]
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Whether no parameter leaves are configured.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Completed optimizer step represented by the current master and moments.
    #[must_use]
    pub fn completed_step(&self) -> u64 {
        self.completed_step
    }

    /// Stable per-parameter flattened lengths used by streaming checkpoints.
    #[must_use]
    pub fn leaf_lens(&self) -> &[usize] {
        &self.leaf_lens
    }

    fn ensure_usable(&self) -> Result<(), BackendError> {
        if self.poisoned {
            Err(BackendError::InvalidInput(
                "host offload trainer is poisoned; complete a fresh checkpoint reload before reuse"
                    .into(),
            ))
        } else {
            Ok(())
        }
    }

    fn validate_next_step(&self, step: u64) -> Result<(), BackendError> {
        self.ensure_usable()?;
        if step == 0 {
            return Err(BackendError::InvalidInput(
                "AdamW step is 1-based; got step 0".into(),
            ));
        }
        let expected_step = self.completed_step.checked_add(1).ok_or_else(|| {
            BackendError::InvalidInput(
                "host offload trainer completed-step counter overflowed".into(),
            )
        })?;
        if step != expected_step {
            return Err(BackendError::InvalidInput(format!(
                "host-offload AdamW expected step {expected_step}, got {step}"
            )));
        }
        Ok(())
    }

    /// Borrow one host-resident latent master.
    pub fn master(&self, index: usize) -> Result<&[f32], BackendError> {
        self.ensure_usable()?;
        self.params
            .get(index)
            .map(|param| param.master.as_slice())
            .ok_or_else(|| {
                BackendError::InvalidInput(format!("parameter index {index} is out of range"))
            })
    }

    /// Borrow one host-resident pair of Adam moments.
    pub fn moments(&self, index: usize) -> Result<(&[f32], &[f32]), BackendError> {
        self.ensure_usable()?;
        self.params
            .get(index)
            .map(|param| (param.m.as_slice(), param.v.as_slice()))
            .ok_or_else(|| {
                BackendError::InvalidInput(format!("parameter index {index} is out of range"))
            })
    }

    /// Geometry needed to rebuild or repack the resident SALT representation
    /// after a streamed AdamW update.
    pub fn parameter_metadata(
        &self,
        index: usize,
    ) -> Result<HostOffloadParamMetadata, BackendError> {
        self.ensure_usable()?;
        self.params
            .get(index)
            .map(|param| HostOffloadParamMetadata {
                rows: param.rows,
                cols: param.cols,
                salt_planes: param.salt_planes,
            })
            .ok_or_else(|| {
                BackendError::InvalidInput(format!("parameter index {index} is out of range"))
            })
    }

    #[cfg(feature = "nccl")]
    pub(crate) fn distributed_optimizer_manifest(&self) -> Vec<u64> {
        let mut manifest = Vec::with_capacity(self.params.len().saturating_mul(8));
        for param in &self.params {
            manifest.extend([
                param.rows as u64,
                param.cols as u64,
                param.salt_planes as u64,
                u64::from(param.optimizer.lr.to_bits()),
                u64::from(param.optimizer.beta1.to_bits()),
                u64::from(param.optimizer.beta2.to_bits()),
                u64::from(param.optimizer.eps.to_bits()),
                u64::from(param.optimizer.weight_decay.to_bits()),
            ]);
        }
        manifest
    }

    /// Current logical host/offload memory accounting.
    #[must_use]
    pub fn stats(&self) -> HostOffloadStats {
        self.stats
    }

    /// Whether a streamed step failed after device optimizer mutation began.
    #[must_use]
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Apply one 1-based AdamW step while staging one parameter's optimizer
    /// state at a time. All gradient metadata is validated before any master or
    /// moment is changed. A device failure during the update loop can leave
    /// earlier leaves updated; reconstruct or reload the trainer before retrying.
    /// Reported step statistics are meaningful only after a successful return.
    pub fn step(&mut self, grads: DeviceGradients, step: u64) -> Result<(), BackendError> {
        self.validate_next_step(step)?;
        if grads.bufs.len() != self.params.len() {
            return Err(BackendError::ShapeMismatch {
                expected: self.params.len(),
                got: grads.bufs.len(),
            });
        }
        let mut resident_input_gradient_elements = 0usize;
        for (index, (param, grad)) in self.params.iter().zip(&grads.bufs).enumerate() {
            if !self.backend.same_context(grad) {
                return Err(BackendError::InvalidInput(format!(
                    "gradient {index} belongs to a different CUDA context"
                )));
            }
            if grad.len() != param.master.len() {
                return Err(BackendError::ShapeMismatch {
                    expected: param.master.len(),
                    got: grad.len(),
                });
            }
            resident_input_gradient_elements = resident_input_gradient_elements
                .checked_add(grad.len())
                .ok_or_else(|| {
                    BackendError::InvalidInput("resident gradient elements overflow usize".into())
                })?;
        }

        for (parameter_index, grad) in grads.bufs.into_iter().enumerate() {
            self.apply_streamed_gradient(parameter_index, &grad, step)?;
        }
        self.finish_offload_pipeline()?;
        self.stats.resident_input_gradient_elements = resident_input_gradient_elements;
        self.completed_step = step;
        Ok(())
    }

    fn apply_streamed_gradient(
        &mut self,
        parameter_index: usize,
        grad: &CudaSlice<f32>,
        step: u64,
    ) -> Result<(), BackendError> {
        let param = self.params.get(parameter_index).ok_or_else(|| {
            BackendError::InvalidInput(format!(
                "streamed parameter index {parameter_index} is out of range"
            ))
        })?;
        if !self.backend.same_context(grad) {
            return Err(BackendError::InvalidInput(format!(
                "streamed gradient {parameter_index} belongs to a different CUDA context"
            )));
        }
        if grad.len() != param.master.len() {
            return Err(BackendError::ShapeMismatch {
                expected: param.master.len(),
                got: grad.len(),
            });
        }
        let pending = PendingOffload {
            parameter_index,
            len: param.master.len(),
        };
        // Preserve the pre-existing zero-sized leaf contract without requiring
        // a zero-byte pinned allocation. AdamW is an elementwise no-op here.
        if pending.len == 0 {
            return Ok(());
        }
        let transition = match self.schedule.enqueue(pending) {
            Ok(transition) => transition,
            Err(error) => {
                self.abort_offload_pipeline();
                return Err(error);
            }
        };
        let update = (|| {
            if let Some(reclaim) = transition.reclaim {
                self.commit_slot(transition.target_slot, reclaim)?;
            }
            self.fill_slot(transition.target_slot, pending)?;
            self.enqueue_slot_upload(transition.target_slot, pending.len)?;
            if let Some(previous) = transition.download_slot {
                self.enqueue_slot_download(previous)?;
            }
            self.enqueue_slot_adam(transition.target_slot, pending, grad, step)
        })();
        if let Err(error) = update {
            self.abort_offload_pipeline();
            return Err(error);
        }
        self.stats.peak_in_flight_parameters = self
            .stats
            .peak_in_flight_parameters
            .max(self.schedule.peak_in_flight);
        Ok(())
    }

    fn slot_mut(&mut self, slot: usize) -> Result<&mut HostOffloadSlot, BackendError> {
        self.slots
            .as_mut()
            .and_then(|slots| slots.get_mut(slot))
            .ok_or_else(|| {
                BackendError::InvalidInput("host-offload staging slot is unavailable".into())
            })
    }

    fn commit_slot(
        &mut self,
        slot_index: usize,
        pending: PendingOffload,
    ) -> Result<(), BackendError> {
        let HostOffloadTrainer { slots, params, .. } = self;
        let slot = slots
            .as_mut()
            .and_then(|slots| slots.get_mut(slot_index))
            .ok_or_else(|| {
                BackendError::InvalidInput("host-offload staging slot is unavailable".into())
            })?;
        let param = params.get_mut(pending.parameter_index).ok_or_else(|| {
            BackendError::InvalidInput(format!(
                "streamed parameter index {} is out of range",
                pending.parameter_index
            ))
        })?;
        if param.master.len() != pending.len {
            return Err(BackendError::ShapeMismatch {
                expected: param.master.len(),
                got: pending.len,
            });
        }
        param.master.copy_from_slice(
            &slot.host_master.as_slice().map_err(|e| {
                BackendError::Backend(format!("read host-offload master staging: {e}"))
            })?[..pending.len],
        );
        param.m.copy_from_slice(
            &slot.host_m.as_slice().map_err(|e| {
                BackendError::Backend(format!("read host-offload first moment staging: {e}"))
            })?[..pending.len],
        );
        param.v.copy_from_slice(
            &slot.host_v.as_slice().map_err(|e| {
                BackendError::Backend(format!("read host-offload second moment staging: {e}"))
            })?[..pending.len],
        );
        Ok(())
    }

    fn fill_slot(
        &mut self,
        slot_index: usize,
        pending: PendingOffload,
    ) -> Result<(), BackendError> {
        let HostOffloadTrainer { slots, params, .. } = self;
        let slot = slots
            .as_mut()
            .and_then(|slots| slots.get_mut(slot_index))
            .ok_or_else(|| {
                BackendError::InvalidInput("host-offload staging slot is unavailable".into())
            })?;
        let param = params.get(pending.parameter_index).ok_or_else(|| {
            BackendError::InvalidInput(format!(
                "streamed parameter index {} is out of range",
                pending.parameter_index
            ))
        })?;
        slot.host_master
            .as_mut_slice()
            .map_err(|e| BackendError::Backend(format!("fill host-offload master staging: {e}")))?
            [..pending.len]
            .copy_from_slice(&param.master);
        slot.host_m.as_mut_slice().map_err(|e| {
            BackendError::Backend(format!("fill host-offload first moment staging: {e}"))
        })?[..pending.len]
            .copy_from_slice(&param.m);
        slot.host_v.as_mut_slice().map_err(|e| {
            BackendError::Backend(format!("fill host-offload second moment staging: {e}"))
        })?[..pending.len]
            .copy_from_slice(&param.v);
        Ok(())
    }

    fn enqueue_slot_upload(&mut self, slot_index: usize, len: usize) -> Result<(), BackendError> {
        let backend = self.backend;
        let transfer = Arc::clone(&self.transfer_stream);
        let slot = self.slot_mut(slot_index)?;
        backend.offload_htod_prefix(
            &transfer,
            &mut slot.host_master,
            len,
            &mut slot.device_master,
        )?;
        backend.offload_htod_prefix(&transfer, &mut slot.host_m, len, &mut slot.device_m)?;
        backend.offload_htod_prefix(&transfer, &mut slot.host_v, len, &mut slot.device_v)
    }

    fn enqueue_slot_download(&mut self, slot_index: usize) -> Result<(), BackendError> {
        let pending = match self.schedule.phases[slot_index] {
            OffloadSlotPhase::Downloading(pending) => pending,
            OffloadSlotPhase::Free | OffloadSlotPhase::Computing(_) => {
                return Err(BackendError::InvalidInput(
                    "host-offload download slot is not ready".into(),
                ));
            }
        };
        let backend = self.backend;
        let transfer = Arc::clone(&self.transfer_stream);
        let slot = self.slot_mut(slot_index)?;
        backend.offload_dtoh_prefix(
            &transfer,
            &slot.device_master,
            pending.len,
            &mut slot.host_master,
        )?;
        backend.offload_dtoh_prefix(&transfer, &slot.device_m, pending.len, &mut slot.host_m)?;
        backend.offload_dtoh_prefix(&transfer, &slot.device_v, pending.len, &mut slot.host_v)
    }

    fn enqueue_slot_adam(
        &mut self,
        slot_index: usize,
        pending: PendingOffload,
        grad: &CudaSlice<f32>,
        step: u64,
    ) -> Result<(), BackendError> {
        let optimizer = self.params[pending.parameter_index].optimizer;
        let backend = self.backend;
        let slot = self.slot_mut(slot_index)?;
        backend.adamw_step_dev_prefix(
            &mut slot.device_master,
            grad,
            &mut slot.device_m,
            &mut slot.device_v,
            pending.len,
            step,
            &optimizer,
        )
    }

    fn finish_offload_pipeline(&mut self) -> Result<(), BackendError> {
        let finish = (|| {
            if let Some(slot) = self.schedule.begin_finish()? {
                self.enqueue_slot_download(slot)?;
            }
            self.backend.offload_synchronize(&self.transfer_stream)?;
            let pending: Vec<_> = self.schedule.pending_downloads().collect();
            for (slot, pending) in pending {
                self.commit_slot(slot, pending)?;
            }
            self.schedule.reset();
            Ok(())
        })();
        if let Err(error) = finish {
            self.abort_offload_pipeline();
            return Err(error);
        }
        Ok(())
    }

    fn abort_offload_pipeline(&mut self) {
        let _ = self.backend.offload_synchronize(&self.transfer_stream);
        self.schedule.reset();
        self.poisoned = true;
    }
}

impl GradientOptimizerSink for HostOffloadTrainer<'_> {
    fn backend(&self) -> &CudaBackend {
        self.backend
    }

    fn parameter_count(&self) -> usize {
        self.params.len()
    }

    fn parameter_len(&self, index: usize) -> Result<usize, BackendError> {
        self.params
            .get(index)
            .map(|parameter| parameter.master.len())
            .ok_or_else(|| {
                BackendError::InvalidInput(format!(
                    "streamed parameter index {index} is out of range"
                ))
            })
    }

    fn validate_stream_step(&self, step: u64) -> Result<(), BackendError> {
        self.validate_next_step(step)
    }

    fn apply_finalized_gradient(
        &mut self,
        parameter_index: usize,
        gradient: &CudaSlice<f32>,
        step: u64,
    ) -> Result<(), BackendError> {
        self.apply_streamed_gradient(parameter_index, gradient, step)
    }

    fn abort_gradient_stream(&mut self) {
        self.abort_offload_pipeline();
    }

    fn finish_gradient_stream(
        &mut self,
        step: u64,
        materialized_gradient_elements: usize,
    ) -> Result<(), BackendError> {
        self.finish_offload_pipeline()?;
        self.stats.resident_input_gradient_elements = materialized_gradient_elements;
        self.completed_step = step;
        Ok(())
    }
}

impl tritium_train::dcp::StateSource for HostOffloadTrainer<'_> {
    fn step(&self) -> u64 {
        self.completed_step
    }

    fn leaf_lens(&self) -> &[usize] {
        &self.leaf_lens
    }

    fn plane_count(&self) -> usize {
        2
    }

    fn read_chunk(
        &mut self,
        plane: StatePlane,
        offset: usize,
        out: &mut [f32],
    ) -> Result<(), DcpError> {
        if self.poisoned {
            return Err(DcpError::InvalidState(
                "cannot checkpoint a poisoned host offload trainer",
            ));
        }
        let (plane, _) = host_offload_dcp_plane(plane)?;
        let total = self
            .leaf_lens
            .iter()
            .try_fold(0usize, |sum, &len| sum.checked_add(len))
            .ok_or(DcpError::InvalidState(
                "host offload layout overflows usize",
            ))?;
        let end = offset
            .checked_add(out.len())
            .ok_or(DcpError::InvalidState("source range overflows usize"))?;
        if end > total {
            return Err(DcpError::InvalidState("source range out of bounds"));
        }

        let mut global_start = 0usize;
        let mut copied = 0usize;
        let mut position = offset;
        for param in &self.params {
            let state = host_offload_state(param, plane);
            let global_end = global_start + state.len();
            if position < global_end && copied < out.len() {
                let local_start = position.saturating_sub(global_start);
                let count = (state.len() - local_start).min(out.len() - copied);
                out[copied..copied + count]
                    .copy_from_slice(&state[local_start..local_start + count]);
                copied += count;
                position += count;
            }
            global_start = global_end;
        }
        if copied != out.len() {
            return Err(DcpError::InvalidState(
                "source range was not fully supplied",
            ));
        }
        Ok(())
    }
}

impl tritium_train::dcp::StateSink for HostOffloadTrainer<'_> {
    fn begin(
        &mut self,
        step: u64,
        leaf_lens: &[usize],
        plane_count: usize,
    ) -> Result<(), DcpError> {
        // A load is transactional only at the trainer-availability boundary:
        // partial bytes may overwrite host vectors, but poison prevents their use.
        self.poisoned = true;
        self.loading = None;
        if leaf_lens != self.leaf_lens {
            return Err(DcpError::InvalidState(
                "checkpoint leaf layout does not match host offload trainer",
            ));
        }
        if plane_count != 2 {
            return Err(DcpError::InvalidState(
                "host offload AdamW requires exactly two optimizer planes",
            ));
        }
        let total = self
            .leaf_lens
            .iter()
            .try_fold(0usize, |sum, &len| sum.checked_add(len))
            .ok_or(DcpError::InvalidState(
                "host offload layout overflows usize",
            ))?;
        self.loading = Some(HostOffloadLoadState {
            step,
            total,
            next_offsets: [0; 3],
        });
        Ok(())
    }

    fn write_chunk(
        &mut self,
        plane: StatePlane,
        offset: usize,
        values: &[f32],
    ) -> Result<(), DcpError> {
        self.poisoned = true;
        let (plane, plane_index) = host_offload_dcp_plane(plane)?;
        let loading = self.loading.as_ref().ok_or(DcpError::InvalidState(
            "host offload checkpoint load has not begun",
        ))?;
        if offset != loading.next_offsets[plane_index] {
            return Err(DcpError::InvalidState(
                "checkpoint chunks must be contiguous and ordered per plane",
            ));
        }
        let end = offset
            .checked_add(values.len())
            .ok_or(DcpError::InvalidState("sink range overflows usize"))?;
        if end > loading.total {
            return Err(DcpError::InvalidState("sink range out of bounds"));
        }

        let mut global_start = 0usize;
        let mut copied = 0usize;
        let mut position = offset;
        for param in &mut self.params {
            let state = host_offload_state_mut(param, plane);
            let global_end = global_start + state.len();
            if position < global_end && copied < values.len() {
                let local_start = position.saturating_sub(global_start);
                let count = (state.len() - local_start).min(values.len() - copied);
                state[local_start..local_start + count]
                    .copy_from_slice(&values[copied..copied + count]);
                copied += count;
                position += count;
            }
            global_start = global_end;
        }
        if copied != values.len() {
            return Err(DcpError::InvalidState("sink range was not fully stored"));
        }
        self.loading
            .as_mut()
            .expect("load state was validated above")
            .next_offsets[plane_index] = end;
        Ok(())
    }

    fn finish(&mut self) -> Result<(), DcpError> {
        self.poisoned = true;
        let loading = self.loading.take().ok_or(DcpError::InvalidState(
            "host offload checkpoint load has not begun",
        ))?;
        if loading
            .next_offsets
            .iter()
            .any(|&offset| offset != loading.total)
        {
            return Err(DcpError::InvalidState(
                "host offload checkpoint load is incomplete",
            ));
        }
        self.completed_step = loading.step;
        self.poisoned = false;
        Ok(())
    }
}

struct GradientStream<'sink, 'transform> {
    sink: &'sink mut dyn GradientOptimizerSink,
    transform: &'transform mut dyn FinalizedGradientTransform,
    plan: GradientCompletionPlan,
    step: u64,
    emissions: Vec<GradientEmission>,
    peak_live_requested_gradient_elements: usize,
    mutation_started: bool,
}

impl GradientStream<'_, '_> {
    fn observe_requested(&mut self, grads: &[Option<CudaSlice<f32>>], lens: &[usize]) {
        let live = self
            .plan
            .bindings
            .iter()
            .filter(|binding| grads[binding.leaf_id].is_some())
            .fold(0usize, |total, binding| {
                total.saturating_add(lens[binding.leaf_id])
            });
        self.peak_live_requested_gradient_elements =
            self.peak_live_requested_gradient_elements.max(live);
    }

    #[allow(clippy::too_many_arguments)]
    fn emit(
        &mut self,
        binding_index: usize,
        backend: &CudaBackend,
        grads: &mut [Option<CudaSlice<f32>>],
        lens: &[usize],
        live_elements: &mut usize,
        peak_elements: &mut usize,
    ) -> Result<(), BackendError> {
        let binding = self.plan.bindings[binding_index];
        let len = lens[binding.leaf_id];
        let mut grad = if let Some(grad) = grads[binding.leaf_id].take() {
            grad
        } else {
            let grad = backend.dev_alloc_zeros(len)?;
            *live_elements = live_elements.saturating_add(len);
            *peak_elements = (*peak_elements).max(*live_elements);
            self.peak_live_requested_gradient_elements =
                self.peak_live_requested_gradient_elements.max(len);
            grad
        };
        let emission = GradientEmission {
            sequence: self.emissions.len(),
            leaf_id: binding.leaf_id,
            parameter_index: binding.parameter_index,
            elements: len,
        };
        self.transform.transform(emission, &mut grad)?;
        if let Err(error) =
            self.sink
                .apply_finalized_gradient(binding.parameter_index, &grad, self.step)
        {
            self.sink.abort_gradient_stream();
            return Err(error);
        }
        self.mutation_started = true;
        *live_elements = live_elements.saturating_sub(len);
        drop(grad);
        self.emissions.push(emission);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn after_op(
        &mut self,
        op_index: usize,
        gradient_inputs: &[usize],
        backend: &CudaBackend,
        grads: &mut [Option<CudaSlice<f32>>],
        lens: &[usize],
        live_elements: &mut usize,
        peak_elements: &mut usize,
    ) -> Result<(), BackendError> {
        for &input in gradient_inputs {
            if let Some(binding_index) = self.plan.binding_by_leaf[input] {
                let remaining = &mut self.plan.remaining_edges[binding_index];
                *remaining = remaining.checked_sub(1).ok_or_else(|| {
                    BackendError::InvalidInput(format!(
                        "gradient completion edge underflow for leaf {input}"
                    ))
                })?;
            }
        }
        self.observe_requested(grads, lens);
        let completed = self.plan.complete_at[op_index].clone();
        for binding_index in completed {
            if self.plan.remaining_edges[binding_index] != 0 {
                return Err(BackendError::InvalidInput(format!(
                    "gradient leaf {} reached completion with {} pending edges",
                    self.plan.bindings[binding_index].leaf_id,
                    self.plan.remaining_edges[binding_index]
                )));
            }
            self.emit(
                binding_index,
                backend,
                grads,
                lens,
                live_elements,
                peak_elements,
            )?;
        }
        Ok(())
    }

    fn emit_unused(
        &mut self,
        backend: &CudaBackend,
        grads: &mut [Option<CudaSlice<f32>>],
        lens: &[usize],
        live_elements: &mut usize,
        peak_elements: &mut usize,
    ) -> Result<(), BackendError> {
        let unused = self.plan.unused.clone();
        for binding_index in unused {
            self.emit(
                binding_index,
                backend,
                grads,
                lens,
                live_elements,
                peak_elements,
            )?;
        }
        Ok(())
    }
}

/// Fixed non-matrix vectors applied inside standard Qwen attention.
///
/// All slices are borrowed only for the duration of graph construction. The
/// tape uploads their values into ordinary leaf tensors, so checkpoint replay
/// and reverse mode use the existing `Add` and `Rmsnorm` operations.
#[derive(Clone, Copy, Debug, Default)]
pub struct FixedAttentionParameters<'a> {
    /// Query projection bias, empty or `[n_head * head_dim]`.
    pub q_bias: &'a [f32],
    /// Key projection bias, empty or `[n_kv_head * head_dim]`.
    pub k_bias: &'a [f32],
    /// Value projection bias, empty or `[n_kv_head * head_dim]`.
    pub v_bias: &'a [f32],
    /// Per-head query RMSNorm weight, empty or `[head_dim]`.
    pub q_norm: &'a [f32],
    /// Per-head key RMSNorm weight, empty or `[head_dim]`.
    pub k_norm: &'a [f32],
    /// RMSNorm epsilon used when Q/K norm vectors are present.
    pub rms_eps: f32,
}

/// A device-resident autograd tape (plan 0043 P2.5): the GPU analogue of [`tritium_train::Tape`].
/// Leaves upload once, results download once, and the recorded ops chain the resident kernels
/// ([`super`]'s `*_dev` methods) with no host round-trips. Forward ops append a device buffer to
/// `vals` and record a `DevOp`. Reverse replay allocates input-gradient slots lazily, retains only
/// requested leaves, and releases each output gradient after its VJP. A `dense_matmul` uses
/// `s = ones`; the shared `ones` buffer is sized once at construction.
pub struct DeviceTape<'backend, 'leaf> {
    b: &'backend CudaBackend,
    vals: Vec<Option<DeviceValue<'leaf>>>,
    lens: Vec<usize>,
    leaves: Vec<bool>,
    ops: Vec<DevOp<'leaf>>,
    ones: CudaSlice<f32>,
    checkpoint_policy: CheckpointPolicy,
    packed_compute_policy: PackedSaltComputePolicy,
    /// Optional tensor-core GEMM tier (Lever 1). When set, dense `Matmul` forward + both
    /// backward GEMMs run on tf32 tensor cores instead of the f32 `--fmad=false` kernels
    /// (~65× per GEMM). Borrowed so one cuBLASLt handle is reused across every per-step tape.
    /// `None` ⇒ the bit-exact f32 path (unchanged); the tier is gated on recovery, not
    /// bit-exactness.
    tc: Option<&'backend TensorCoreGemm>,
    checkpoint_interval: Option<usize>,
    checkpoint_markers: usize,
    last_checkpoint_op: usize,
    segments: Vec<CheckpointSegment>,
    segment_op_start: usize,
    segment_value_start: usize,
    live_activation_elements: usize,
    peak_live_activation_elements: usize,
    recomputed_ops: usize,
}

fn validate_attention_geometry(
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
) -> Result<(usize, usize), BackendError> {
    if n_head == 0 || n_kv_head == 0 || head_dim == 0 {
        return Err(BackendError::InvalidInput(
            "attention head counts and head dimension must be non-zero".into(),
        ));
    }
    if !n_head.is_multiple_of(n_kv_head) {
        return Err(BackendError::InvalidInput(format!(
            "attention heads {n_head} must be divisible by KV heads {n_kv_head}"
        )));
    }
    let q_width = n_head
        .checked_mul(head_dim)
        .ok_or_else(|| BackendError::InvalidInput("attention Q width overflows usize".into()))?;
    let kv_width = n_kv_head
        .checked_mul(head_dim)
        .ok_or_else(|| BackendError::InvalidInput("attention KV width overflows usize".into()))?;
    Ok((q_width, kv_width))
}

fn validate_fixed_attention_parameters(
    fixed: FixedAttentionParameters<'_>,
    q_width: usize,
    kv_width: usize,
    head_dim: usize,
) -> Result<(), BackendError> {
    let bias_presence = [
        !fixed.q_bias.is_empty(),
        !fixed.k_bias.is_empty(),
        !fixed.v_bias.is_empty(),
    ];
    if bias_presence.iter().any(|present| *present) && !bias_presence.iter().all(|present| *present)
    {
        return Err(BackendError::InvalidInput(
            "QKV bias vectors must be all present or all absent".into(),
        ));
    }
    let norm_presence = [!fixed.q_norm.is_empty(), !fixed.k_norm.is_empty()];
    if norm_presence[0] != norm_presence[1] {
        return Err(BackendError::InvalidInput(
            "Q/K norm vectors must be both present or both absent".into(),
        ));
    }
    for (name, values, expected) in [
        ("Q bias", fixed.q_bias, q_width),
        ("K bias", fixed.k_bias, kv_width),
        ("V bias", fixed.v_bias, kv_width),
        ("Q norm", fixed.q_norm, head_dim),
        ("K norm", fixed.k_norm, head_dim),
    ] {
        if !values.is_empty() && values.len() != expected {
            return Err(BackendError::InvalidInput(format!(
                "attention {name} has {} elements, expected {expected}",
                values.len()
            )));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(BackendError::InvalidInput(format!(
                "attention {name} contains a non-finite value"
            )));
        }
    }
    if norm_presence[0] && (!fixed.rms_eps.is_finite() || fixed.rms_eps <= 0.0) {
        return Err(BackendError::InvalidInput(
            "attention Q/K norm epsilon must be finite and positive".into(),
        ));
    }
    Ok(())
}

fn attention_positions(sequence: usize) -> Result<Vec<u32>, BackendError> {
    let mut positions = Vec::new();
    positions
        .try_reserve_exact(sequence)
        .map_err(|_| BackendError::OutOfMemory {
            requested: sequence.saturating_mul(core::mem::size_of::<i32>()),
        })?;
    for position in 0..sequence {
        positions.push(u32::try_from(position).map_err(|_| {
            BackendError::InvalidInput("attention position exceeds u32::MAX".into())
        })?);
    }
    Ok(positions)
}

impl core::fmt::Debug for DeviceTape<'_, '_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DeviceTape")
            .field("n_values", &self.vals.len())
            .field("n_ops", &self.ops.len())
            .finish_non_exhaustive()
    }
}

impl<'backend, 'leaf> DeviceTape<'backend, 'leaf> {
    /// New empty tape on `b`. `ones_max` must be ≥ the largest matmul output width `n` used (the
    /// per-row unit scale buffer is allocated once to this size).
    pub fn new(b: &'backend CudaBackend, ones_max: usize) -> Result<Self, BackendError> {
        Self::new_with_checkpoint_policy(b, ones_max, CheckpointPolicy::KeepAll)
    }

    /// New tape with explicit activation-checkpoint scheduling.
    pub fn new_with_checkpoint_policy(
        b: &'backend CudaBackend,
        ones_max: usize,
        checkpoint_policy: CheckpointPolicy,
    ) -> Result<Self, BackendError> {
        Self::new_with_policies(
            b,
            ones_max,
            checkpoint_policy,
            PackedSaltComputePolicy::Exact,
        )
    }

    /// New tape with explicit activation-checkpoint and packed-compute policy.
    pub fn new_with_policies(
        b: &'backend CudaBackend,
        ones_max: usize,
        checkpoint_policy: CheckpointPolicy,
        packed_compute_policy: PackedSaltComputePolicy,
    ) -> Result<Self, BackendError> {
        let checkpoint_interval = checkpoint_policy.interval()?;
        let ones = b.dev_upload(&vec![1.0f32; ones_max.max(1)])?;
        Ok(Self {
            b,
            vals: Vec::new(),
            lens: Vec::new(),
            leaves: Vec::new(),
            ops: Vec::new(),
            ones,
            checkpoint_policy,
            packed_compute_policy,
            tc: None,
            checkpoint_interval,
            checkpoint_markers: 0,
            last_checkpoint_op: 0,
            segments: Vec::new(),
            segment_op_start: 0,
            segment_value_start: 0,
            live_activation_elements: 0,
            peak_live_activation_elements: 0,
            recomputed_ops: 0,
        })
    }

    fn push_activation(&mut self, buf: CudaSlice<f32>, len: usize) -> usize {
        let id = self.vals.len();
        self.vals.push(Some(DeviceValue::Owned(buf)));
        self.lens.push(len);
        self.leaves.push(false);
        self.live_activation_elements = self.live_activation_elements.saturating_add(len);
        self.peak_live_activation_elements = self
            .peak_live_activation_elements
            .max(self.live_activation_elements);
        id
    }

    fn value_slice(&self, id: usize) -> Result<&CudaSlice<f32>, BackendError> {
        let value = self.vals.get(id).ok_or_else(|| {
            BackendError::InvalidInput(format!("device tape value id {id} is out of range"))
        })?;
        match value {
            Some(DeviceValue::GradientOnly) => Err(BackendError::InvalidInput(format!(
                "device tape value id {id} is a gradient-only leaf with no forward value"
            ))),
            Some(value) => Ok(value.as_slice()),
            None => Err(BackendError::InvalidInput(format!(
                "device tape value id {id} was evicted; include it in checkpoint_keep frontier"
            ))),
        }
    }

    fn evict_activation(&mut self, id: usize) -> Result<bool, BackendError> {
        if id >= self.vals.len() {
            return Err(BackendError::InvalidInput(format!(
                "device tape value id {id} is out of range"
            )));
        }
        if self.leaves[id] || self.vals[id].is_none() {
            return Ok(false);
        }
        match self.vals[id].take() {
            Some(DeviceValue::Owned(_)) => {
                self.live_activation_elements =
                    self.live_activation_elements.saturating_sub(self.lens[id]);
                Ok(true)
            }
            Some(value @ DeviceValue::Borrowed(_)) => {
                self.vals[id] = Some(value);
                Err(BackendError::InvalidInput(format!(
                    "non-leaf device tape value id {id} cannot borrow checkpoint storage"
                )))
            }
            Some(DeviceValue::GradientOnly) => Err(BackendError::InvalidInput(format!(
                "non-leaf device tape value id {id} cannot be gradient-only"
            ))),
            None => Ok(false),
        }
    }

    /// Upload a weight/input leaf; returns its value id.
    pub fn leaf(&mut self, host: &[f32]) -> Result<usize, BackendError> {
        let buf = self.b.dev_upload(host)?;
        let id = self.vals.len();
        self.vals.push(Some(DeviceValue::Owned(buf)));
        self.lens.push(host.len());
        self.leaves.push(true);
        Ok(id)
    }

    /// Borrow an existing resident tensor as a leaf without allocating or
    /// copying it. The tensor must belong to this tape's CUDA context.
    pub fn leaf_device(&mut self, tensor: &'leaf DeviceTensor) -> Result<usize, BackendError> {
        self.validate_device_tensor(tensor)?;
        let id = self.vals.len();
        self.vals.push(Some(DeviceValue::Borrowed(&tensor.buf)));
        self.lens.push(tensor.buf.len());
        self.leaves.push(true);
        Ok(id)
    }

    /// Validate a resident tensor against this tape without changing the graph.
    pub fn validate_device_tensor(&self, tensor: &DeviceTensor) -> Result<(), BackendError> {
        if !self.b.same_context(&tensor.buf) {
            return Err(BackendError::InvalidInput(
                "device tensor belongs to a different CUDA context".into(),
            ));
        }
        Ok(())
    }

    fn leaf_borrowed_prefix(
        &mut self,
        buffer: &'leaf CudaSlice<f32>,
        len: usize,
        name: &str,
    ) -> Result<usize, BackendError> {
        if !self.b.same_context(buffer) {
            return Err(BackendError::InvalidInput(format!(
                "{name} belongs to a different CUDA context"
            )));
        }
        if len == 0 || buffer.len() < len {
            return Err(BackendError::ShapeMismatch {
                expected: len.max(1),
                got: buffer.len(),
            });
        }
        let id = self.vals.len();
        self.vals.push(Some(DeviceValue::Borrowed(buffer)));
        self.lens.push(len);
        self.leaves.push(true);
        Ok(id)
    }

    /// Create a leaf that receives gradients but owns no forward f32 buffer.
    /// This is the latent-master identity used by packed SALT ops: their
    /// forward reads compact planes while their STE VJP targets this value id.
    pub fn gradient_leaf(&mut self, len: usize) -> Result<usize, BackendError> {
        if len == 0 {
            return Err(BackendError::InvalidInput(
                "gradient-only leaf length must be non-zero".into(),
            ));
        }
        let id = self.vals.len();
        self.vals.push(Some(DeviceValue::GradientOnly));
        self.lens.push(len);
        self.leaves.push(true);
        Ok(id)
    }

    fn validate_packed_master(
        &self,
        master: usize,
        weight: &DevicePackedSaltWeight,
    ) -> Result<(), BackendError> {
        weight.ensure_prepared()?;
        let expected = weight.rows().checked_mul(weight.cols()).ok_or_else(|| {
            BackendError::InvalidInput("packed SALT master shape overflows usize".into())
        })?;
        let value = self.vals.get(master).ok_or_else(|| {
            BackendError::InvalidInput(format!("packed SALT master id {master} is out of range"))
        })?;
        if !matches!(value, Some(DeviceValue::GradientOnly)) || !self.leaves[master] {
            return Err(BackendError::InvalidInput(format!(
                "packed SALT master id {master} must be a gradient-only leaf"
            )));
        }
        if self.lens[master] != expected {
            return Err(BackendError::ShapeMismatch {
                expected,
                got: self.lens[master],
            });
        }
        Ok(())
    }

    /// Mark a logical block boundary and explicitly name every non-leaf value
    /// that may be used after it. Leaves are retained implicitly. When the
    /// configured interval materializes this boundary, other owned values in
    /// the completed segment are released immediately.
    pub fn checkpoint_keep(&mut self, frontier: &[usize]) -> Result<(), BackendError> {
        if frontier.is_empty() {
            return Err(BackendError::InvalidInput(
                "checkpoint frontier must contain at least one value".into(),
            ));
        }
        for (position, &id) in frontier.iter().enumerate() {
            if id >= self.vals.len() {
                return Err(BackendError::InvalidInput(format!(
                    "checkpoint frontier value id {id} is out of range"
                )));
            }
            if frontier[..position].contains(&id) {
                return Err(BackendError::InvalidInput(format!(
                    "checkpoint frontier value id {id} is duplicated"
                )));
            }
            if self.leaves[id] {
                return Err(BackendError::InvalidInput(format!(
                    "checkpoint frontier value id {id} is a leaf; leaves are retained implicitly"
                )));
            }
            self.value_slice(id)?;
        }

        if self.last_checkpoint_op == self.ops.len() {
            return Err(BackendError::InvalidInput(
                "checkpoint block contains no forward operations".into(),
            ));
        }
        let next_marker = self.checkpoint_markers.checked_add(1).ok_or_else(|| {
            BackendError::InvalidInput("checkpoint marker count overflows usize".into())
        })?;
        if let CheckpointPolicy::SqrtDepth(total_blocks) = self.checkpoint_policy
            && next_marker > total_blocks
        {
            return Err(BackendError::InvalidInput(format!(
                "checkpoint marker count {} exceeds configured depth {total_blocks}",
                next_marker
            )));
        }
        self.checkpoint_markers = next_marker;
        self.last_checkpoint_op = self.ops.len();
        let Some(interval) = self.checkpoint_interval else {
            return Ok(());
        };
        if !self.checkpoint_markers.is_multiple_of(interval) {
            return Ok(());
        }
        if self.segment_op_start == self.ops.len() {
            return Err(BackendError::InvalidInput(
                "checkpoint segment contains no forward operations".into(),
            ));
        }

        let mut keep = vec![false; self.vals.len()];
        for &id in frontier {
            keep[id] = true;
        }
        let value_end = self.vals.len();
        for (id, should_keep) in keep
            .iter()
            .enumerate()
            .take(value_end)
            .skip(self.segment_value_start)
        {
            if !should_keep {
                self.evict_activation(id)?;
            }
        }
        self.segments.push(CheckpointSegment {
            op_start: self.segment_op_start,
            op_end: self.ops.len(),
            value_start: self.segment_value_start,
            value_end,
            frontier: frontier.to_vec(),
            evicted: true,
        });
        self.segment_op_start = self.ops.len();
        self.segment_value_start = value_end;
        Ok(())
    }

    /// Download a value (e.g. the logits) to host.
    pub fn value(&self, id: usize) -> Result<Vec<f32>, BackendError> {
        let len = *self.lens.get(id).ok_or_else(|| {
            BackendError::InvalidInput(format!("device tape value id {id} is out of range"))
        })?;
        let mut h = vec![0.0f32; len];
        self.b.dev_download(self.value_slice(id)?, &mut h)?;
        Ok(h)
    }

    /// `g_logits` of `L = mean_row softmax-xent(logits, target)` — the reverse seed when the loss is
    /// distillation cross-entropy (`grad_out = 1`, so `gscale = 1/rows`).
    pub(crate) fn softmax_xent_grad(
        &self,
        logits: usize,
        target: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<CudaSlice<f32>, BackendError> {
        let d_target = self.b.dev_upload(target)?;
        self.softmax_xent_grad_device(logits, &d_target, rows, cols)
    }

    fn softmax_xent_grad_device(
        &self,
        logits: usize,
        target: &CudaSlice<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<CudaSlice<f32>, BackendError> {
        let mut g = self.b.dev_alloc_zeros(rows * cols)?;
        self.b.softmax_xent_backward_dev(
            self.value_slice(logits)?,
            target,
            &mut g,
            rows,
            cols,
            1.0 / rows as f32,
        )?;
        Ok(g)
    }

    /// `Y[m,n] = X[m,k]·W[n,k]ᵀ` (fp dense).
    /// Route this tape's dense `Matmul` GEMMs (forward + both backward) through the tf32
    /// tensor-core tier. Builder-style so existing constructors are untouched (default `None`
    /// = f32 path). The handle is created once (e.g. per training run) and shared by every
    /// per-step tape.
    #[must_use]
    pub fn with_tensor_core(mut self, tc: &'backend TensorCoreGemm) -> Self {
        self.tc = Some(tc);
        self
    }

    /// Dense-matmul forward `Y = X·Wᵀ`: tf32 tensor cores when the tier is attached, else the
    /// f32 `--fmad=false` kernel (with `s = ones`, so the two are the same operation).
    fn mm_forward(
        &self,
        xs: &CudaSlice<f32>,
        ws: &CudaSlice<f32>,
        shape: GemmShape,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        match self.tc {
            Some(tc) => tc.forward(xs, ws, shape, out),
            None => self.b.matmul_forward_dev(xs, ws, &self.ones, shape, out),
        }
    }

    /// Dense-matmul activation grad `gA = gY·W`: tf32 tier or f32 kernel.
    fn mm_grad_a(
        &self,
        gy: &CudaSlice<f32>,
        ws: &CudaSlice<f32>,
        shape: GemmShape,
        ga: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        match self.tc {
            Some(tc) => tc.grad_a(gy, ws, shape, ga),
            None => self.b.grad_a_dev(gy, ws, &self.ones, shape, ga),
        }
    }

    /// Dense-matmul weight grad `gW = gYᵀ·X`: tf32 tier or f32 kernel.
    fn mm_grad_w(
        &self,
        gy: &CudaSlice<f32>,
        xs: &CudaSlice<f32>,
        shape: GemmShape,
        gw: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        match self.tc {
            Some(tc) => tc.grad_w(gy, xs, shape, gw),
            None => self.b.grad_w_dev(gy, xs, &self.ones, shape, gw),
        }
    }

    /// Smooth HESTIA ternary expectation retained as a dense training weight.
    ///
    /// Compose returned value with [`Self::matmul`]. Packed SALT contractions are
    /// intentionally unavailable until hard export because soft expectations are dense.
    /// Temperatures below `MIN_DIFFERENTIABLE_TAU` fail closed to zero in device kernels.
    pub fn hestia_relax(
        &mut self,
        weight: usize,
        scale: usize,
        tau: usize,
        rows: usize,
        cols: usize,
    ) -> Result<usize, BackendError> {
        let elements = rows.checked_mul(cols).ok_or_else(|| {
            BackendError::InvalidInput("HESTIA matrix shape overflows usize".into())
        })?;
        if rows == 0 || cols == 0 {
            return Err(BackendError::InvalidInput(
                "HESTIA rows and cols must be non-zero".into(),
            ));
        }
        for (id, expected) in [(weight, elements), (scale, rows), (tau, 1)] {
            let got = *self.lens.get(id).ok_or_else(|| {
                BackendError::InvalidInput(format!("device tape value id {id} is out of range"))
            })?;
            if got != expected {
                return Err(BackendError::ShapeMismatch { expected, got });
            }
        }
        let mut out = self.b.dev_alloc_zeros(elements)?;
        self.b.hestia_relax_forward_dev(
            self.value_slice(weight)?,
            self.value_slice(scale)?,
            self.value_slice(tau)?,
            &mut out,
            rows,
            cols,
        )?;
        let id = self.push_activation(out, elements);
        self.ops.push(DevOp::HestiaRelax {
            weight,
            scale,
            tau,
            rows,
            cols,
            out: id,
        });
        Ok(id)
    }

    /// Project one resident master through HESTIA using the packed handle's
    /// current first-plane AbsMean scales.
    ///
    /// SALT packing refreshes these scales from the same master after every
    /// optimizer step. The scale prefix is borrowed directly; no host transfer,
    /// device copy, or dense quantized shadow is created.
    pub fn hestia_relax_packed(
        &mut self,
        weight: usize,
        master: &DeviceTensor,
        packed: &'leaf DevicePackedSaltWeight,
        tau: usize,
    ) -> Result<usize, BackendError> {
        packed.ensure_bound_to(master)?;
        let elements = packed.rows().checked_mul(packed.cols()).ok_or_else(|| {
            BackendError::InvalidInput("HESTIA packed geometry overflows usize".into())
        })?;
        let value = self.vals.get(weight).ok_or_else(|| {
            BackendError::InvalidInput(format!("device tape value id {weight} is out of range"))
        })?;
        if !matches!(
            value,
            Some(DeviceValue::Borrowed(buffer)) if std::ptr::eq(*buffer, &master.buf)
        ) {
            return Err(BackendError::InvalidInput(
                "HESTIA weight id is not the bound resident-master leaf".into(),
            ));
        }
        let got = self.lens[weight];
        if got != elements {
            return Err(BackendError::ShapeMismatch {
                expected: elements,
                got,
            });
        }
        let scale = self.leaf_borrowed_prefix(
            packed.inner.scales(),
            packed.rows(),
            "packed HESTIA scales",
        )?;
        self.hestia_relax(weight, scale, tau, packed.rows(), packed.cols())
    }

    pub fn matmul(
        &mut self,
        x: usize,
        w: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<usize, BackendError> {
        let mut out = self.b.dev_alloc_zeros(m * n)?;
        self.mm_forward(
            self.value_slice(x)?,
            self.value_slice(w)?,
            GemmShape { m, n, k },
            &mut out,
        )?;
        let id = self.push_activation(out, m * n);
        self.ops.push(DevOp::Matmul {
            x,
            w,
            m,
            n,
            k,
            out: id,
        });
        Ok(id)
    }

    /// Packed SALT matrix multiply. Forward reads only compact planes; the
    /// ordinary dense `grad_w` VJP is routed to `master` as the identity STE.
    pub fn salt_matmul(
        &mut self,
        x: usize,
        master: usize,
        weight: &'leaf DevicePackedSaltWeight,
        m: usize,
    ) -> Result<usize, BackendError> {
        self.validate_packed_master(master, weight)?;
        let n = weight.rows();
        let k = weight.cols();
        let x_len = m.checked_mul(k).ok_or_else(|| {
            BackendError::InvalidInput("packed SALT activation shape overflows usize".into())
        })?;
        let out_len = m.checked_mul(n).ok_or_else(|| {
            BackendError::InvalidInput("packed SALT output shape overflows usize".into())
        })?;
        if self.ones.len() < n {
            return Err(BackendError::ShapeMismatch {
                expected: n,
                got: self.ones.len(),
            });
        }
        let got_x = *self.lens.get(x).ok_or_else(|| {
            BackendError::InvalidInput(format!("device tape value id {x} is out of range"))
        })?;
        if got_x != x_len {
            return Err(BackendError::ShapeMismatch {
                expected: x_len,
                got: got_x,
            });
        }
        let mut out = self.b.dev_alloc_zeros(out_len)?;
        match self.packed_compute_policy {
            PackedSaltComputePolicy::Exact => {
                self.b
                    .training_salt_forward(self.value_slice(x)?, &weight.inner, m, &mut out)?
            }
            PackedSaltComputePolicy::Fast => self.b.training_salt_forward_fast(
                self.value_slice(x)?,
                &weight.inner,
                m,
                &mut out,
            )?,
        }
        let id = self.push_activation(out, out_len);
        self.ops.push(DevOp::SaltMatmul {
            x,
            master,
            weight,
            m,
            n,
            k,
            out: id,
        });
        Ok(id)
    }

    pub fn rmsnorm(
        &mut self,
        x: usize,
        w: usize,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<usize, BackendError> {
        let mut out = self.b.dev_alloc_zeros(rows * cols)?;
        self.b.rmsnorm_forward_dev(
            self.value_slice(x)?,
            self.value_slice(w)?,
            &mut out,
            rows,
            cols,
            eps,
        )?;
        let id = self.push_activation(out, rows * cols);
        self.ops.push(DevOp::Rmsnorm {
            x,
            w,
            rows,
            cols,
            eps,
            out: id,
        });
        Ok(id)
    }

    pub fn silu(&mut self, x: usize) -> Result<usize, BackendError> {
        let n = self.lens[x];
        let mut out = self.b.dev_alloc_zeros(n)?;
        self.b.silu_forward_dev(self.value_slice(x)?, &mut out, n)?;
        let id = self.push_activation(out, n);
        self.ops.push(DevOp::Silu { x, n, out: id });
        Ok(id)
    }

    pub fn mul(&mut self, a: usize, b: usize) -> Result<usize, BackendError> {
        debug_assert_eq!(
            self.lens[a], self.lens[b],
            "mul operands must be same length"
        );
        let n = self.lens[a];
        let mut out = self.b.dev_alloc_zeros(n)?;
        self.b
            .ew_mul_forward_dev(self.value_slice(a)?, self.value_slice(b)?, &mut out, n)?;
        let id = self.push_activation(out, n);
        self.ops.push(DevOp::Mul { a, b, n, out: id });
        Ok(id)
    }

    pub fn add(&mut self, a: usize, b: usize) -> Result<usize, BackendError> {
        debug_assert_eq!(
            self.lens[a], self.lens[b],
            "add operands must be same length"
        );
        let n = self.lens[a];
        let mut out = self.b.dev_alloc_zeros(n)?;
        self.b
            .ew_add_forward_dev(self.value_slice(a)?, self.value_slice(b)?, &mut out, n)?;
        let id = self.push_activation(out, n);
        self.ops.push(DevOp::Add { a, b, n, out: id });
        Ok(id)
    }

    /// Tied-embedding gather `y[t,:] = w[tokens[t],:]`. `w` is the `[vocab,dim]` embedding leaf.
    pub fn embed(
        &mut self,
        w: usize,
        tokens: &[i32],
        seq: usize,
        dim: usize,
        vocab: usize,
    ) -> Result<usize, BackendError> {
        let segments = self.b.prepare_embed_segments(tokens, seq, vocab)?;
        let d_tok = self.b.dev_upload_i32(tokens)?;
        let mut out = self.b.dev_alloc_zeros(seq * dim)?;
        self.b
            .embed_gather_forward_dev(self.value_slice(w)?, &d_tok, &mut out, seq, dim)?;
        let id = self.push_activation(out, seq * dim);
        self.ops.push(DevOp::Embed {
            w,
            tokens: d_tok,
            segments,
            seq,
            dim,
            vocab,
            out: id,
        });
        Ok(id)
    }

    /// Gather token rows directly from packed SALT planes. The deterministic
    /// segmented embedding VJP accumulates into the same identity-STE master
    /// leaf that a tied [`Self::salt_matmul`] head uses.
    pub fn salt_embed(
        &mut self,
        master: usize,
        weight: &'leaf DevicePackedSaltWeight,
        tokens: &[i32],
    ) -> Result<usize, BackendError> {
        self.validate_packed_master(master, weight)?;
        let seq = tokens.len();
        let vocab = weight.rows();
        let dim = weight.cols();
        let out_len = seq.checked_mul(dim).ok_or_else(|| {
            BackendError::InvalidInput("packed SALT embedding shape overflows usize".into())
        })?;
        let segments = self.b.prepare_embed_segments(tokens, seq, vocab)?;
        let d_tokens = self.b.dev_upload_i32(tokens)?;
        let mut out = self.b.dev_alloc_zeros(out_len)?;
        self.b
            .training_salt_embed_forward(&weight.inner, &d_tokens, seq, &mut out)?;
        let id = self.push_activation(out, out_len);
        self.ops.push(DevOp::SaltEmbed {
            master,
            weight,
            tokens: d_tokens,
            segments,
            seq,
            dim,
            vocab,
            out: id,
        });
        Ok(id)
    }

    /// RoPE over a `[n_token, n_head, head_dim]` buffer (forward rotation).
    pub(crate) fn rope(
        &mut self,
        x: usize,
        positions: &[u32],
        n_head: usize,
        head_dim: usize,
        theta: f32,
        n_token: usize,
    ) -> Result<usize, BackendError> {
        let n = self.lens[x];
        let pos = self.b.dev_upload_u32(positions)?;
        let mut out = self.b.dev_alloc_zeros(n)?;
        self.b.rope_apply_dev(
            self.value_slice(x)?,
            &mut out,
            &pos,
            n_head,
            head_dim,
            theta,
            n_token,
            1.0,
        )?;
        let id = self.push_activation(out, n);
        self.ops.push(DevOp::Rope {
            x,
            pos,
            n_head,
            head_dim,
            theta,
            n_token,
            out: id,
        });
        Ok(id)
    }

    /// Extract columns `[start, start+len)` from a `[rows, cols]` buffer → `[rows, len]`.
    pub(crate) fn slice_cols(
        &mut self,
        x: usize,
        rows: usize,
        cols: usize,
        start: usize,
        len: usize,
    ) -> Result<usize, BackendError> {
        let mut out = self.b.dev_alloc_zeros(rows * len)?;
        self.b
            .slice_cols_forward_dev(self.value_slice(x)?, &mut out, rows, cols, start, len)?;
        let id = self.push_activation(out, rows * len);
        self.ops.push(DevOp::SliceCols {
            x,
            rows,
            cols,
            start,
            len,
            out: id,
        });
        Ok(id)
    }

    /// Multiply by a constant scalar (attention's `1/√head_dim`).
    pub(crate) fn scale_const(&mut self, x: usize, c: f32) -> Result<usize, BackendError> {
        let n = self.lens[x];
        let mut out = self.b.dev_alloc_zeros(n)?;
        self.b
            .scale_const_dev(self.value_slice(x)?, &mut out, c, n)?;
        let id = self.push_activation(out, n);
        self.ops.push(DevOp::ScaleConst { x, c, n, out: id });
        Ok(id)
    }

    /// Additive causal mask over `[rows=queries, cols=keys]` scores.
    pub(crate) fn causal_mask(
        &mut self,
        x: usize,
        rows: usize,
        cols: usize,
    ) -> Result<usize, BackendError> {
        let mut out = self.b.dev_alloc_zeros(rows * cols)?;
        self.b
            .causal_mask_forward_dev(self.value_slice(x)?, &mut out, rows, cols)?;
        let id = self.push_activation(out, rows * cols);
        self.ops.push(DevOp::CausalMask {
            x,
            rows,
            cols,
            out: id,
        });
        Ok(id)
    }

    /// Row softmax over `[rows, cols]`.
    pub(crate) fn softmax(
        &mut self,
        x: usize,
        rows: usize,
        cols: usize,
    ) -> Result<usize, BackendError> {
        let mut out = self.b.dev_alloc_zeros(rows * cols)?;
        self.b
            .softmax_forward_dev(self.value_slice(x)?, &mut out, rows, cols)?;
        let id = self.push_activation(out, rows * cols);
        self.ops.push(DevOp::Softmax {
            x,
            rows,
            cols,
            out: id,
        });
        Ok(id)
    }

    /// Transpose `[rows, cols]` → `[cols, rows]`.
    pub(crate) fn transpose(
        &mut self,
        x: usize,
        rows: usize,
        cols: usize,
    ) -> Result<usize, BackendError> {
        let mut out = self.b.dev_alloc_zeros(rows * cols)?;
        self.b
            .transpose_forward_dev(self.value_slice(x)?, &mut out, rows, cols)?;
        let id = self.push_activation(out, rows * cols);
        self.ops.push(DevOp::Transpose {
            x,
            rows,
            cols,
            out: id,
        });
        Ok(id)
    }

    /// Concatenate `parts` (each `[rows, lens[i]]`) along columns → `[rows, Σ lens]`.
    pub(crate) fn concat(
        &mut self,
        parts: &[usize],
        rows: usize,
        lens: &[usize],
    ) -> Result<usize, BackendError> {
        let total: usize = lens.iter().sum();
        let mut out = self.b.dev_alloc_zeros(rows * total)?;
        let mut off = 0;
        for (&p, &len) in parts.iter().zip(lens) {
            self.b
                .copy_into_cols_dev(self.value_slice(p)?, &mut out, rows, total, off, len)?;
            off += len;
        }
        let id = self.push_activation(out, rows * total);
        self.ops.push(DevOp::Concat {
            parts: parts.to_vec(),
            rows,
            lens: lens.to_vec(),
            out: id,
        });
        Ok(id)
    }

    /// Extract a contiguous block of `n_rows` rows starting at `start_row` from an
    /// `[total_rows, cols]` row-major activation → `[n_rows, cols]`.
    ///
    /// Row-major rows are contiguous, so a row-block is exactly a column-slice of the
    /// flattened `[1, total_rows*cols]` view. This is the batching primitive: it carves
    /// one sequence out of a `[batch*seq, cols]` stack so attention (which mixes across
    /// the seq dim and must not attend across sequences) can run per-sequence while the
    /// row-independent stages (embed/norm/MLP/head) run once over the whole batch. Its
    /// vjp scatters the grad back into the block's offset — the exact `SliceCols` vjp.
    pub fn slice_rows(
        &mut self,
        x: usize,
        total_rows: usize,
        cols: usize,
        start_row: usize,
        n_rows: usize,
    ) -> Result<usize, BackendError> {
        if start_row + n_rows > total_rows {
            return Err(BackendError::InvalidInput(format!(
                "slice_rows [{start_row}, {}) exceeds {total_rows} rows",
                start_row + n_rows
            )));
        }
        self.slice_cols(x, 1, total_rows * cols, start_row * cols, n_rows * cols)
    }

    /// Stack row-blocks (each `[part_rows[i], cols]`) into `[Σ part_rows, cols]` — the
    /// inverse of [`Self::slice_rows`], recombining per-sequence attention outputs into
    /// the batched `[batch*seq, cols]` activation. Reuses column `concat` on the
    /// flattened single-row view, so its vjp is the exact `Concat` vjp.
    pub fn concat_rows(
        &mut self,
        parts: &[usize],
        part_rows: &[usize],
        cols: usize,
    ) -> Result<usize, BackendError> {
        if parts.len() != part_rows.len() {
            return Err(BackendError::InvalidInput(format!(
                "concat_rows got {} parts but {} row counts",
                parts.len(),
                part_rows.len()
            )));
        }
        let lens: Vec<usize> = part_rows.iter().map(|r| r * cols).collect();
        self.concat(parts, 1, &lens)
    }

    fn add_fixed_attention_bias(
        &mut self,
        value: usize,
        rows: usize,
        width: usize,
        bias: &[f32],
    ) -> Result<usize, BackendError> {
        if bias.is_empty() {
            return Ok(value);
        }
        let elements = rows.checked_mul(width).ok_or_else(|| {
            BackendError::InvalidInput("attention bias expansion overflows usize".into())
        })?;
        let mut repeated = Vec::new();
        repeated
            .try_reserve_exact(elements)
            .map_err(|_| BackendError::OutOfMemory {
                requested: elements.saturating_mul(core::mem::size_of::<f32>()),
            })?;
        for _ in 0..rows {
            repeated.extend_from_slice(bias);
        }
        let bias = self.leaf(&repeated)?;
        self.add(value, bias)
    }

    fn apply_fixed_qk_norm(
        &mut self,
        value: usize,
        sequence: usize,
        heads: usize,
        head_dim: usize,
        weight: &[f32],
        eps: f32,
    ) -> Result<usize, BackendError> {
        if weight.is_empty() {
            return Ok(value);
        }
        let rows = sequence.checked_mul(heads).ok_or_else(|| {
            BackendError::InvalidInput("Q/K norm row count overflows usize".into())
        })?;
        let weight = self.leaf(weight)?;
        self.rmsnorm(value, weight, rows, head_dim, eps)
    }

    /// Multi-head causal self-attention with GQA — the device analogue of `tritium_train::nn::attention`.
    /// `x` is `[seq, n_embd]` (normed). Returns the attention output `[seq, n_embd]`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &mut self,
        x: usize,
        wq: usize,
        wk: usize,
        wv: usize,
        wo: usize,
        seq: usize,
        n_embd: usize,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        theta: f32,
    ) -> Result<usize, BackendError> {
        self.attention_with_fixed(
            x,
            wq,
            wk,
            wv,
            wo,
            seq,
            n_embd,
            n_head,
            n_kv_head,
            head_dim,
            theta,
            FixedAttentionParameters::default(),
        )
    }

    /// Standard causal GQA with optional fixed QKV bias and per-head Q/K RMSNorm.
    ///
    /// The numerical order is projection, bias, Q/K norm, RoPE, attention. All
    /// fixed-vector contracts are validated before the tape is mutated.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_with_fixed(
        &mut self,
        x: usize,
        wq: usize,
        wk: usize,
        wv: usize,
        wo: usize,
        seq: usize,
        n_embd: usize,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        theta: f32,
        fixed: FixedAttentionParameters<'_>,
    ) -> Result<usize, BackendError> {
        let (qd, kvd) = validate_attention_geometry(n_head, n_kv_head, head_dim)?;
        validate_fixed_attention_parameters(fixed, qd, kvd, head_dim)?;
        let group = n_head / n_kv_head;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let pos = attention_positions(seq)?;

        let q = self.matmul(x, wq, seq, qd, n_embd)?;
        let k = self.matmul(x, wk, seq, kvd, n_embd)?;
        let v = self.matmul(x, wv, seq, kvd, n_embd)?;
        let q = self.add_fixed_attention_bias(q, seq, qd, fixed.q_bias)?;
        let k = self.add_fixed_attention_bias(k, seq, kvd, fixed.k_bias)?;
        let v = self.add_fixed_attention_bias(v, seq, kvd, fixed.v_bias)?;
        let q = self.apply_fixed_qk_norm(q, seq, n_head, head_dim, fixed.q_norm, fixed.rms_eps)?;
        let k =
            self.apply_fixed_qk_norm(k, seq, n_kv_head, head_dim, fixed.k_norm, fixed.rms_eps)?;
        let q = self.rope(q, &pos, n_head, head_dim, theta, seq)?;
        let k = self.rope(k, &pos, n_kv_head, head_dim, theta, seq)?;

        let mut head_outs = Vec::with_capacity(n_head);
        for h in 0..n_head {
            let kv = h / group;
            let qh = self.slice_cols(q, seq, qd, h * head_dim, head_dim)?;
            let kh = self.slice_cols(k, seq, kvd, kv * head_dim, head_dim)?;
            let vh = self.slice_cols(v, seq, kvd, kv * head_dim, head_dim)?;
            let scores = self.matmul(qh, kh, seq, seq, head_dim)?; // qh · khᵀ
            let scores = self.scale_const(scores, scale)?;
            let scores = self.causal_mask(scores, seq, seq)?;
            let p = self.softmax(scores, seq, seq)?;
            let vt = self.transpose(vh, seq, head_dim)?;
            head_outs.push(self.matmul(p, vt, seq, head_dim, seq)?); // p · vh
        }
        let cat = self.concat(&head_outs, seq, &vec![head_dim; n_head])?;
        self.matmul(cat, wo, seq, n_embd, qd)
    }

    /// Multi-head causal self-attention whose four trainable projections read
    /// compact SALT planes and route identity-STE gradients to their separate
    /// gradient-only master ids. All attention glue remains the same resident
    /// implementation used by [`Self::attention`].
    #[allow(clippy::too_many_arguments)]
    pub fn salt_attention(
        &mut self,
        x: usize,
        wq_master: usize,
        wq: &'leaf DevicePackedSaltWeight,
        wk_master: usize,
        wk: &'leaf DevicePackedSaltWeight,
        wv_master: usize,
        wv: &'leaf DevicePackedSaltWeight,
        wo_master: usize,
        wo: &'leaf DevicePackedSaltWeight,
        seq: usize,
        n_embd: usize,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        theta: f32,
    ) -> Result<usize, BackendError> {
        self.salt_attention_with_fixed(
            x,
            wq_master,
            wq,
            wk_master,
            wk,
            wv_master,
            wv,
            wo_master,
            wo,
            seq,
            n_embd,
            n_head,
            n_kv_head,
            head_dim,
            theta,
            FixedAttentionParameters::default(),
        )
    }

    /// Packed-SALT causal GQA with optional fixed QKV bias and Q/K RMSNorm.
    ///
    /// Only matrix projections use packed weights. Fixed vectors are uploaded as
    /// ordinary leaves and composed from existing replayable tape operations.
    #[allow(clippy::too_many_arguments)]
    pub fn salt_attention_with_fixed(
        &mut self,
        x: usize,
        wq_master: usize,
        wq: &'leaf DevicePackedSaltWeight,
        wk_master: usize,
        wk: &'leaf DevicePackedSaltWeight,
        wv_master: usize,
        wv: &'leaf DevicePackedSaltWeight,
        wo_master: usize,
        wo: &'leaf DevicePackedSaltWeight,
        seq: usize,
        n_embd: usize,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        theta: f32,
        fixed: FixedAttentionParameters<'_>,
    ) -> Result<usize, BackendError> {
        let (qd, kvd) = validate_attention_geometry(n_head, n_kv_head, head_dim)?;
        validate_fixed_attention_parameters(fixed, qd, kvd, head_dim)?;
        let input_len = seq.checked_mul(n_embd).ok_or_else(|| {
            BackendError::InvalidInput("packed attention input shape overflows usize".into())
        })?;
        let got_input_len = *self.lens.get(x).ok_or_else(|| {
            BackendError::InvalidInput(format!("device tape value id {x} is out of range"))
        })?;
        if got_input_len != input_len {
            return Err(BackendError::ShapeMismatch {
                expected: input_len,
                got: got_input_len,
            });
        }
        for (name, master, weight, rows, cols) in [
            ("q", wq_master, wq, qd, n_embd),
            ("k", wk_master, wk, kvd, n_embd),
            ("v", wv_master, wv, kvd, n_embd),
            ("o", wo_master, wo, n_embd, qd),
        ] {
            self.validate_packed_master(master, weight)?;
            if weight.rows() != rows || weight.cols() != cols {
                return Err(BackendError::InvalidInput(format!(
                    "packed attention {name} projection is [{}, {}], expected [{rows}, {cols}]",
                    weight.rows(),
                    weight.cols()
                )));
            }
        }
        let group = n_head / n_kv_head;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let pos = attention_positions(seq)?;

        let q = self.salt_matmul(x, wq_master, wq, seq)?;
        let k = self.salt_matmul(x, wk_master, wk, seq)?;
        let v = self.salt_matmul(x, wv_master, wv, seq)?;
        let q = self.add_fixed_attention_bias(q, seq, qd, fixed.q_bias)?;
        let k = self.add_fixed_attention_bias(k, seq, kvd, fixed.k_bias)?;
        let v = self.add_fixed_attention_bias(v, seq, kvd, fixed.v_bias)?;
        let q = self.apply_fixed_qk_norm(q, seq, n_head, head_dim, fixed.q_norm, fixed.rms_eps)?;
        let k =
            self.apply_fixed_qk_norm(k, seq, n_kv_head, head_dim, fixed.k_norm, fixed.rms_eps)?;
        let q = self.rope(q, &pos, n_head, head_dim, theta, seq)?;
        let k = self.rope(k, &pos, n_kv_head, head_dim, theta, seq)?;

        let mut head_outs = Vec::with_capacity(n_head);
        for h in 0..n_head {
            let kv = h / group;
            let qh = self.slice_cols(q, seq, qd, h * head_dim, head_dim)?;
            let kh = self.slice_cols(k, seq, kvd, kv * head_dim, head_dim)?;
            let vh = self.slice_cols(v, seq, kvd, kv * head_dim, head_dim)?;
            let scores = self.matmul(qh, kh, seq, seq, head_dim)?;
            let scores = self.scale_const(scores, scale)?;
            let scores = self.causal_mask(scores, seq, seq)?;
            let probabilities = self.softmax(scores, seq, seq)?;
            let vt = self.transpose(vh, seq, head_dim)?;
            head_outs.push(self.matmul(probabilities, vt, seq, head_dim, seq)?);
        }
        let cat = self.concat(&head_outs, seq, &vec![head_dim; n_head])?;
        self.salt_matmul(cat, wo_master, wo, seq)
    }

    fn replay_op(&mut self, op_index: usize) -> Result<(), BackendError> {
        let op = self.ops.get(op_index).ok_or_else(|| {
            BackendError::InvalidInput(format!("replay op index {op_index} is out of range"))
        })?;
        let out_id = op.output();
        if self.vals[out_id].is_some() {
            return Err(BackendError::InvalidInput(format!(
                "replay output value id {out_id} is already resident"
            )));
        }

        let output = match *op {
            DevOp::HestiaRelax {
                weight,
                scale,
                tau,
                rows,
                cols,
                out: _,
            } => {
                let mut output = self.b.dev_alloc_zeros(rows * cols)?;
                self.b.hestia_relax_forward_dev(
                    self.value_slice(weight)?,
                    self.value_slice(scale)?,
                    self.value_slice(tau)?,
                    &mut output,
                    rows,
                    cols,
                )?;
                output
            }
            DevOp::Matmul {
                x,
                w,
                m,
                n,
                k,
                out: _,
            } => {
                let mut output = self.b.dev_alloc_zeros(m * n)?;
                self.mm_forward(
                    self.value_slice(x)?,
                    self.value_slice(w)?,
                    GemmShape { m, n, k },
                    &mut output,
                )?;
                output
            }
            DevOp::SaltMatmul {
                x,
                master: _,
                weight,
                m,
                n,
                k: _,
                out: _,
            } => {
                weight.ensure_prepared()?;
                let mut output = self.b.dev_alloc_zeros(m * n)?;
                match self.packed_compute_policy {
                    PackedSaltComputePolicy::Exact => self.b.training_salt_forward(
                        self.value_slice(x)?,
                        &weight.inner,
                        m,
                        &mut output,
                    )?,
                    PackedSaltComputePolicy::Fast => self.b.training_salt_forward_fast(
                        self.value_slice(x)?,
                        &weight.inner,
                        m,
                        &mut output,
                    )?,
                }
                output
            }
            DevOp::Rmsnorm {
                x,
                w,
                rows,
                cols,
                eps,
                out: _,
            } => {
                let mut output = self.b.dev_alloc_zeros(rows * cols)?;
                self.b.rmsnorm_forward_dev(
                    self.value_slice(x)?,
                    self.value_slice(w)?,
                    &mut output,
                    rows,
                    cols,
                    eps,
                )?;
                output
            }
            DevOp::Silu { x, n, out: _ } => {
                let mut output = self.b.dev_alloc_zeros(n)?;
                self.b
                    .silu_forward_dev(self.value_slice(x)?, &mut output, n)?;
                output
            }
            DevOp::Mul { a, b, n, out: _ } => {
                let mut output = self.b.dev_alloc_zeros(n)?;
                self.b.ew_mul_forward_dev(
                    self.value_slice(a)?,
                    self.value_slice(b)?,
                    &mut output,
                    n,
                )?;
                output
            }
            DevOp::Add { a, b, n, out: _ } => {
                let mut output = self.b.dev_alloc_zeros(n)?;
                self.b.ew_add_forward_dev(
                    self.value_slice(a)?,
                    self.value_slice(b)?,
                    &mut output,
                    n,
                )?;
                output
            }
            DevOp::Embed {
                w,
                ref tokens,
                segments: _,
                seq,
                dim,
                vocab: _,
                out: _,
            } => {
                let mut output = self.b.dev_alloc_zeros(seq * dim)?;
                self.b.embed_gather_forward_dev(
                    self.value_slice(w)?,
                    tokens,
                    &mut output,
                    seq,
                    dim,
                )?;
                output
            }
            DevOp::SaltEmbed {
                master: _,
                weight,
                ref tokens,
                segments: _,
                seq,
                dim,
                vocab: _,
                out: _,
            } => {
                weight.ensure_prepared()?;
                let mut output = self.b.dev_alloc_zeros(seq * dim)?;
                self.b
                    .training_salt_embed_forward(&weight.inner, tokens, seq, &mut output)?;
                output
            }
            DevOp::Rope {
                x,
                ref pos,
                n_head,
                head_dim,
                theta,
                n_token,
                out: _,
            } => {
                let mut output = self.b.dev_alloc_zeros(self.lens[out_id])?;
                self.b.rope_apply_dev(
                    self.value_slice(x)?,
                    &mut output,
                    pos,
                    n_head,
                    head_dim,
                    theta,
                    n_token,
                    1.0,
                )?;
                output
            }
            DevOp::SliceCols {
                x,
                rows,
                cols,
                start,
                len,
                out: _,
            } => {
                let mut output = self.b.dev_alloc_zeros(rows * len)?;
                self.b.slice_cols_forward_dev(
                    self.value_slice(x)?,
                    &mut output,
                    rows,
                    cols,
                    start,
                    len,
                )?;
                output
            }
            DevOp::ScaleConst { x, c, n, out: _ } => {
                let mut output = self.b.dev_alloc_zeros(n)?;
                self.b
                    .scale_const_dev(self.value_slice(x)?, &mut output, c, n)?;
                output
            }
            DevOp::CausalMask {
                x,
                rows,
                cols,
                out: _,
            } => {
                let mut output = self.b.dev_alloc_zeros(rows * cols)?;
                self.b
                    .causal_mask_forward_dev(self.value_slice(x)?, &mut output, rows, cols)?;
                output
            }
            DevOp::Softmax {
                x,
                rows,
                cols,
                out: _,
            } => {
                let mut output = self.b.dev_alloc_zeros(rows * cols)?;
                self.b
                    .softmax_forward_dev(self.value_slice(x)?, &mut output, rows, cols)?;
                output
            }
            DevOp::Transpose {
                x,
                rows,
                cols,
                out: _,
            } => {
                let mut output = self.b.dev_alloc_zeros(rows * cols)?;
                self.b
                    .transpose_forward_dev(self.value_slice(x)?, &mut output, rows, cols)?;
                output
            }
            DevOp::Concat {
                ref parts,
                rows,
                ref lens,
                out: _,
            } => {
                let total: usize = lens.iter().sum();
                let mut output = self.b.dev_alloc_zeros(rows * total)?;
                let mut offset = 0;
                for (&part, &len) in parts.iter().zip(lens) {
                    self.b.copy_into_cols_dev(
                        self.value_slice(part)?,
                        &mut output,
                        rows,
                        total,
                        offset,
                        len,
                    )?;
                    offset += len;
                }
                output
            }
        };
        if output.len() != self.lens[out_id] {
            return Err(BackendError::ShapeMismatch {
                expected: self.lens[out_id],
                got: output.len(),
            });
        }
        self.vals[out_id] = Some(DeviceValue::Owned(output));
        self.live_activation_elements = self
            .live_activation_elements
            .saturating_add(self.lens[out_id]);
        self.peak_live_activation_elements = self
            .peak_live_activation_elements
            .max(self.live_activation_elements);
        self.recomputed_ops = self.recomputed_ops.saturating_add(1);
        Ok(())
    }

    fn backward_segments(&self) -> Result<Vec<CheckpointSegment>, BackendError> {
        if let CheckpointPolicy::SqrtDepth(total_blocks) = self.checkpoint_policy
            && self.checkpoint_markers != total_blocks
        {
            return Err(BackendError::InvalidInput(format!(
                "checkpoint marker count {} does not match configured depth {total_blocks}",
                self.checkpoint_markers
            )));
        }
        let mut segments = self.segments.clone();
        if self.segment_op_start < self.ops.len() {
            segments.push(CheckpointSegment {
                op_start: self.segment_op_start,
                op_end: self.ops.len(),
                value_start: self.segment_value_start,
                value_end: self.vals.len(),
                frontier: Vec::new(),
                evicted: false,
            });
        }
        Ok(segments)
    }

    fn gradient_completion_plan(
        &self,
        bindings: &[GradientLeafBinding],
        sink: &dyn GradientOptimizerSink,
        step: u64,
    ) -> Result<GradientCompletionPlan, BackendError> {
        sink.validate_stream_step(step)?;
        if !sink.backend().same_context(&self.ones) {
            return Err(BackendError::InvalidInput(
                "gradient optimizer sink belongs to a different CUDA context".into(),
            ));
        }
        if bindings.len() != sink.parameter_count() {
            return Err(BackendError::ShapeMismatch {
                expected: sink.parameter_count(),
                got: bindings.len(),
            });
        }
        // Validate checkpoint marker completeness before a sink can mutate.
        self.backward_segments()?;

        let mut binding_by_leaf = vec![None; self.vals.len()];
        let mut parameter_seen = vec![false; sink.parameter_count()];
        let mut materialized_collection_elements = 0usize;
        for (binding_index, &binding) in bindings.iter().enumerate() {
            if binding.leaf_id >= self.vals.len() {
                return Err(BackendError::InvalidInput(format!(
                    "gradient leaf id {} is out of range",
                    binding.leaf_id
                )));
            }
            if !self.leaves[binding.leaf_id] {
                return Err(BackendError::InvalidInput(format!(
                    "gradient value id {} is not a leaf",
                    binding.leaf_id
                )));
            }
            if binding_by_leaf[binding.leaf_id]
                .replace(binding_index)
                .is_some()
            {
                return Err(BackendError::InvalidInput(format!(
                    "gradient leaf id {} is duplicated",
                    binding.leaf_id
                )));
            }
            let parameter_len = sink.parameter_len(binding.parameter_index)?;
            if parameter_seen[binding.parameter_index] {
                return Err(BackendError::InvalidInput(format!(
                    "streamed parameter index {} is duplicated",
                    binding.parameter_index
                )));
            }
            parameter_seen[binding.parameter_index] = true;
            if self.lens[binding.leaf_id] != parameter_len {
                return Err(BackendError::ShapeMismatch {
                    expected: parameter_len,
                    got: self.lens[binding.leaf_id],
                });
            }
            materialized_collection_elements = materialized_collection_elements
                .checked_add(self.lens[binding.leaf_id])
                .ok_or_else(|| {
                    BackendError::InvalidInput(
                        "materialized gradient elements overflow usize".into(),
                    )
                })?;
        }

        let mut remaining_edges = vec![0usize; bindings.len()];
        let mut first_consumer = vec![None; bindings.len()];
        for (op_index, op) in self.ops.iter().enumerate() {
            for input in op.gradient_inputs() {
                if let Some(binding_index) = binding_by_leaf[input] {
                    remaining_edges[binding_index] = remaining_edges[binding_index]
                        .checked_add(1)
                        .ok_or_else(|| {
                            BackendError::InvalidInput(
                                "gradient consumer edge count overflows usize".into(),
                            )
                        })?;
                    first_consumer[binding_index] = Some(
                        first_consumer[binding_index]
                            .map_or(op_index, |earlier: usize| earlier.min(op_index)),
                    );
                }
            }
        }
        let mut complete_at = vec![Vec::new(); self.ops.len()];
        let mut unused = Vec::new();
        for (binding_index, completion) in first_consumer.into_iter().enumerate() {
            if let Some(op_index) = completion {
                complete_at[op_index].push(binding_index);
            } else {
                unused.push(binding_index);
            }
        }
        let by_parameter = |&binding_index: &usize| bindings[binding_index].parameter_index;
        for group in &mut complete_at {
            group.sort_unstable_by_key(by_parameter);
        }
        unused.sort_unstable_by_key(by_parameter);
        Ok(GradientCompletionPlan {
            bindings: bindings.to_vec(),
            binding_by_leaf,
            complete_at,
            unused,
            remaining_edges,
            materialized_collection_elements,
        })
    }

    fn accumulate_grad_slot(
        &self,
        grads: &mut [Option<CudaSlice<f32>>],
        retain: &[bool],
        id: usize,
        source: &CudaSlice<f32>,
        live_elements: &mut usize,
        peak_elements: &mut usize,
    ) -> Result<(), BackendError> {
        // An unrequested leaf has no producer to replay, so its gradient can be
        // discarded instead of consuming a persistent slot. Shared/tied leaves
        // that are requested still accumulate every consumer contribution.
        if self.leaves[id] && !retain[id] {
            return Ok(());
        }
        if grads[id].is_none() {
            grads[id] = Some(self.b.dev_alloc_zeros(self.lens[id])?);
            *live_elements = live_elements.saturating_add(self.lens[id]);
            *peak_elements = (*peak_elements).max(*live_elements);
        }
        self.b.accumulate_dev(
            grads[id]
                .as_mut()
                .expect("gradient slot was allocated above"),
            source,
            self.lens[id],
        )
    }

    fn backward_core(
        &mut self,
        seed_id: usize,
        seed: &CudaSlice<f32>,
        retain_ids: &[usize],
        mut stream: Option<&mut GradientStream<'_, '_>>,
    ) -> Result<DeviceBackwardResult, BackendError> {
        if seed_id >= self.vals.len() {
            return Err(BackendError::InvalidInput(format!(
                "gradient seed value id {seed_id} is out of range"
            )));
        }
        if seed.len() != self.lens[seed_id] {
            return Err(BackendError::ShapeMismatch {
                expected: self.lens[seed_id],
                got: seed.len(),
            });
        }
        if !self.b.same_context(seed) {
            return Err(BackendError::InvalidInput(
                "gradient seed belongs to a different CUDA context".into(),
            ));
        }
        let mut retain = vec![false; self.vals.len()];
        for &id in retain_ids {
            if id >= self.vals.len() {
                return Err(BackendError::InvalidInput(format!(
                    "retained gradient value id {id} is out of range"
                )));
            }
            retain[id] = true;
        }

        let naive_all_value_grad_elements = self
            .lens
            .iter()
            .fold(0usize, |total, &len| total.saturating_add(len));
        let naive_activation_elements = self
            .lens
            .iter()
            .zip(&self.leaves)
            .filter(|(_, leaf)| !**leaf)
            .fold(0usize, |total, (&len, _)| total.saturating_add(len));
        let mut checkpoint_seen = vec![false; self.vals.len()];
        let retained_checkpoint_elements = self
            .segments
            .iter()
            .flat_map(|segment| segment.frontier.iter().copied())
            .filter(|&id| {
                let first = !checkpoint_seen[id];
                checkpoint_seen[id] = true;
                first && !self.leaves[id] && self.vals[id].is_some()
            })
            .fold(0usize, |total, id| total.saturating_add(self.lens[id]));
        let backward_segments = self.backward_segments()?;
        let mut grads: Vec<Option<CudaSlice<f32>>> = (0..self.vals.len()).map(|_| None).collect();
        let mut live_elements = 0usize;
        let mut peak_persistent_grad_elements = 0usize;
        self.accumulate_grad_slot(
            &mut grads,
            &retain,
            seed_id,
            seed,
            &mut live_elements,
            &mut peak_persistent_grad_elements,
        )?;
        if let Some(stream) = stream.as_deref_mut() {
            stream.observe_requested(&grads, &self.lens);
        }
        for segment in backward_segments.iter().rev() {
            if segment.evicted {
                for id in segment.value_start..segment.value_end {
                    self.evict_activation(id)?;
                }
                for op_index in segment.op_start..segment.op_end {
                    self.replay_op(op_index)?;
                }
            }
            for op_index in (segment.op_start..segment.op_end).rev() {
                let op = &self.ops[op_index];
                let out_id = op.output();
                let gradient_inputs = op.gradient_inputs();
                let Some(grad_out) = grads[out_id].take() else {
                    // This op is not on a path from the seed, but its activation
                    // is still dead once reverse replay reaches its producer.
                    self.evict_activation(out_id)?;
                    if let Some(stream) = stream.as_deref_mut() {
                        stream.after_op(
                            op_index,
                            &gradient_inputs,
                            self.b,
                            &mut grads,
                            &self.lens,
                            &mut live_elements,
                            &mut peak_persistent_grad_elements,
                        )?;
                    }
                    continue;
                };
                match *op {
                    DevOp::HestiaRelax {
                        weight,
                        scale,
                        tau,
                        rows,
                        cols,
                        out: _,
                    } => {
                        let mut grad_weight = self.b.dev_alloc_zeros(rows * cols)?;
                        let mut grad_tau = self.b.dev_alloc_zeros(1)?;
                        self.b.hestia_relax_backward_dev(
                            self.value_slice(weight)?,
                            self.value_slice(scale)?,
                            self.value_slice(tau)?,
                            &grad_out,
                            &mut grad_weight,
                            &mut grad_tau,
                            rows,
                            cols,
                        )?;
                        self.accumulate_grad_slot(
                            &mut grads,
                            &retain,
                            weight,
                            &grad_weight,
                            &mut live_elements,
                            &mut peak_persistent_grad_elements,
                        )?;
                        let grad_scale = self.b.dev_alloc_zeros(rows)?;
                        self.accumulate_grad_slot(
                            &mut grads,
                            &retain,
                            scale,
                            &grad_scale,
                            &mut live_elements,
                            &mut peak_persistent_grad_elements,
                        )?;
                        self.accumulate_grad_slot(
                            &mut grads,
                            &retain,
                            tau,
                            &grad_tau,
                            &mut live_elements,
                            &mut peak_persistent_grad_elements,
                        )?;
                    }
                    DevOp::Matmul {
                        x,
                        w,
                        m,
                        n,
                        k,
                        out: _,
                    } => {
                        let shape = GemmShape { m, n, k };
                        let mut gx = self.b.dev_alloc_zeros(m * k)?;
                        self.mm_grad_a(&grad_out, self.value_slice(w)?, shape, &mut gx)?;
                        self.accumulate_grad_slot(
                            &mut grads,
                            &retain,
                            x,
                            &gx,
                            &mut live_elements,
                            &mut peak_persistent_grad_elements,
                        )?;
                        let mut gw = self.b.dev_alloc_zeros(n * k)?;
                        self.mm_grad_w(&grad_out, self.value_slice(x)?, shape, &mut gw)?;
                        self.accumulate_grad_slot(
                            &mut grads,
                            &retain,
                            w,
                            &gw,
                            &mut live_elements,
                            &mut peak_persistent_grad_elements,
                        )?;
                    }
                    DevOp::SaltMatmul {
                        x,
                        master,
                        weight,
                        m,
                        n,
                        k,
                        out: _,
                    } => {
                        weight.ensure_prepared()?;
                        let shape = GemmShape { m, n, k };
                        let mut gx = self.b.dev_alloc_zeros(m * k)?;
                        match self.packed_compute_policy {
                            PackedSaltComputePolicy::Exact => {
                                self.b
                                    .training_salt_grad_a(&grad_out, &weight.inner, m, &mut gx)?
                            }
                            PackedSaltComputePolicy::Fast => self.b.training_salt_grad_a_fast(
                                &grad_out,
                                &weight.inner,
                                m,
                                &mut gx,
                            )?,
                        }
                        self.accumulate_grad_slot(
                            &mut grads,
                            &retain,
                            x,
                            &gx,
                            &mut live_elements,
                            &mut peak_persistent_grad_elements,
                        )?;
                        // Identity SALT STE: dQ/dMaster := I. The master VJP is
                        // therefore the ordinary dense weight gradient X^T·gy;
                        // no dense quantized weight is read or materialized.
                        let mut gmaster = self.b.dev_alloc_zeros(n * k)?;
                        self.b.grad_w_dev(
                            &grad_out,
                            self.value_slice(x)?,
                            &self.ones,
                            shape,
                            &mut gmaster,
                        )?;
                        self.accumulate_grad_slot(
                            &mut grads,
                            &retain,
                            master,
                            &gmaster,
                            &mut live_elements,
                            &mut peak_persistent_grad_elements,
                        )?;
                    }
                    DevOp::Rmsnorm {
                        x,
                        w,
                        rows,
                        cols,
                        eps,
                        out: _,
                    } => {
                        let mut gx = self.b.dev_alloc_zeros(rows * cols)?;
                        let mut gw = self.b.dev_alloc_zeros(cols)?;
                        self.b.rmsnorm_backward_dev(
                            self.value_slice(x)?,
                            self.value_slice(w)?,
                            &grad_out,
                            &mut gx,
                            &mut gw,
                            rows,
                            cols,
                            eps,
                        )?;
                        self.accumulate_grad_slot(
                            &mut grads,
                            &retain,
                            x,
                            &gx,
                            &mut live_elements,
                            &mut peak_persistent_grad_elements,
                        )?;
                        self.accumulate_grad_slot(
                            &mut grads,
                            &retain,
                            w,
                            &gw,
                            &mut live_elements,
                            &mut peak_persistent_grad_elements,
                        )?;
                    }
                    DevOp::Silu { x, n, out: _ } => {
                        let mut gx = self.b.dev_alloc_zeros(n)?;
                        self.b
                            .silu_backward_dev(self.value_slice(x)?, &grad_out, &mut gx, n)?;
                        self.accumulate_grad_slot(
                            &mut grads,
                            &retain,
                            x,
                            &gx,
                            &mut live_elements,
                            &mut peak_persistent_grad_elements,
                        )?;
                    }
                    DevOp::Mul { a, b, n, out: _ } => {
                        let mut ga = self.b.dev_alloc_zeros(n)?;
                        self.b
                            .ew_mul_backward_dev(&grad_out, self.value_slice(b)?, &mut ga, n)?;
                        self.accumulate_grad_slot(
                            &mut grads,
                            &retain,
                            a,
                            &ga,
                            &mut live_elements,
                            &mut peak_persistent_grad_elements,
                        )?;
                        let mut gb = self.b.dev_alloc_zeros(n)?;
                        self.b
                            .ew_mul_backward_dev(&grad_out, self.value_slice(a)?, &mut gb, n)?;
                        self.accumulate_grad_slot(
                            &mut grads,
                            &retain,
                            b,
                            &gb,
                            &mut live_elements,
                            &mut peak_persistent_grad_elements,
                        )?;
                    }
                    DevOp::Add { a, b, n: _, out: _ } => {
                        self.accumulate_grad_slot(
                            &mut grads,
                            &retain,
                            a,
                            &grad_out,
                            &mut live_elements,
                            &mut peak_persistent_grad_elements,
                        )?;
                        self.accumulate_grad_slot(
                            &mut grads,
                            &retain,
                            b,
                            &grad_out,
                            &mut live_elements,
                            &mut peak_persistent_grad_elements,
                        )?;
                    }
                    DevOp::Embed {
                        w,
                        tokens: _,
                        ref segments,
                        seq,
                        dim,
                        vocab,
                        out: _,
                    } => {
                        let mut gw = self.b.dev_alloc_zeros(vocab * dim)?;
                        self.b.embed_gather_backward_segmented_prepared_dev(
                            &grad_out, segments, &mut gw, seq, dim, vocab,
                        )?;
                        self.accumulate_grad_slot(
                            &mut grads,
                            &retain,
                            w,
                            &gw,
                            &mut live_elements,
                            &mut peak_persistent_grad_elements,
                        )?;
                    }
                    DevOp::SaltEmbed {
                        master,
                        weight: _,
                        tokens: _,
                        ref segments,
                        seq,
                        dim,
                        vocab,
                        out: _,
                    } => {
                        let mut gmaster = self.b.dev_alloc_zeros(vocab * dim)?;
                        self.b.embed_gather_backward_segmented_prepared_dev(
                            &grad_out,
                            segments,
                            &mut gmaster,
                            seq,
                            dim,
                            vocab,
                        )?;
                        self.accumulate_grad_slot(
                            &mut grads,
                            &retain,
                            master,
                            &gmaster,
                            &mut live_elements,
                            &mut peak_persistent_grad_elements,
                        )?;
                    }
                    DevOp::Rope {
                        x,
                        ref pos,
                        n_head,
                        head_dim,
                        theta,
                        n_token,
                        out,
                    } => {
                        // vjp = inverse rotation (sign = -1) of the output grad.
                        let n = self.lens[out];
                        let mut gx = self.b.dev_alloc_zeros(n)?;
                        self.b.rope_apply_dev(
                            &grad_out, &mut gx, pos, n_head, head_dim, theta, n_token, -1.0,
                        )?;
                        self.accumulate_grad_slot(
                            &mut grads,
                            &retain,
                            x,
                            &gx,
                            &mut live_elements,
                            &mut peak_persistent_grad_elements,
                        )?;
                    }
                    DevOp::SliceCols {
                        x,
                        rows,
                        cols,
                        start,
                        len,
                        out: _,
                    } => {
                        // vjp = scatter the [rows,len] grad back into a zeroed [rows,cols] at [start,+len).
                        let mut gx = self.b.dev_alloc_zeros(rows * cols)?;
                        self.b
                            .copy_into_cols_dev(&grad_out, &mut gx, rows, cols, start, len)?;
                        self.accumulate_grad_slot(
                            &mut grads,
                            &retain,
                            x,
                            &gx,
                            &mut live_elements,
                            &mut peak_persistent_grad_elements,
                        )?;
                    }
                    DevOp::ScaleConst { x, c, n, out: _ } => {
                        let mut gx = self.b.dev_alloc_zeros(n)?;
                        self.b.scale_const_dev(&grad_out, &mut gx, c, n)?;
                        self.accumulate_grad_slot(
                            &mut grads,
                            &retain,
                            x,
                            &gx,
                            &mut live_elements,
                            &mut peak_persistent_grad_elements,
                        )?;
                    }
                    DevOp::CausalMask {
                        x,
                        rows,
                        cols,
                        out: _,
                    } => {
                        let mut gx = self.b.dev_alloc_zeros(rows * cols)?;
                        self.b
                            .causal_mask_backward_dev(&grad_out, &mut gx, rows, cols)?;
                        self.accumulate_grad_slot(
                            &mut grads,
                            &retain,
                            x,
                            &gx,
                            &mut live_elements,
                            &mut peak_persistent_grad_elements,
                        )?;
                    }
                    DevOp::Softmax { x, rows, cols, out } => {
                        // vjp uses the saved probabilities p = vals[out].
                        let mut gx = self.b.dev_alloc_zeros(rows * cols)?;
                        self.b.softmax_backward_dev(
                            self.value_slice(out)?,
                            &grad_out,
                            &mut gx,
                            rows,
                            cols,
                        )?;
                        self.accumulate_grad_slot(
                            &mut grads,
                            &retain,
                            x,
                            &gx,
                            &mut live_elements,
                            &mut peak_persistent_grad_elements,
                        )?;
                    }
                    DevOp::Transpose {
                        x,
                        rows,
                        cols,
                        out: _,
                    } => {
                        // vjp = transpose the [cols,rows] output grad back to [rows,cols].
                        let mut gx = self.b.dev_alloc_zeros(rows * cols)?;
                        self.b
                            .transpose_forward_dev(&grad_out, &mut gx, cols, rows)?;
                        self.accumulate_grad_slot(
                            &mut grads,
                            &retain,
                            x,
                            &gx,
                            &mut live_elements,
                            &mut peak_persistent_grad_elements,
                        )?;
                    }
                    DevOp::Concat {
                        ref parts,
                        rows,
                        ref lens,
                        out: _,
                    } => {
                        // vjp = slice each part's column range back out of the concatenated grad.
                        let total: usize = lens.iter().sum();
                        let mut off = 0;
                        for (&p, &len) in parts.iter().zip(lens) {
                            let mut gp = self.b.dev_alloc_zeros(rows * len)?;
                            self.b.slice_cols_forward_dev(
                                &grad_out, &mut gp, rows, total, off, len,
                            )?;
                            self.accumulate_grad_slot(
                                &mut grads,
                                &retain,
                                p,
                                &gp,
                                &mut live_elements,
                                &mut peak_persistent_grad_elements,
                            )?;
                            off += len;
                        }
                    }
                }

                if retain[out_id] {
                    grads[out_id] = Some(grad_out);
                } else {
                    // `grad_out` remains alive for the entire VJP above. Account for
                    // its release only after all input slots have been accumulated.
                    live_elements = live_elements.saturating_sub(self.lens[out_id]);
                    drop(grad_out);
                }
                self.evict_activation(out_id)?;
                if let Some(stream) = stream.as_deref_mut() {
                    stream.after_op(
                        op_index,
                        &gradient_inputs,
                        self.b,
                        &mut grads,
                        &self.lens,
                        &mut live_elements,
                        &mut peak_persistent_grad_elements,
                    )?;
                }
            }
        }

        if let Some(stream) = stream.as_deref_mut() {
            stream.emit_unused(
                self.b,
                &mut grads,
                &self.lens,
                &mut live_elements,
                &mut peak_persistent_grad_elements,
            )?;
        }

        // Preserve the old API's zero-gradient behavior for requested values
        // that are disconnected from the seed.
        if stream.is_none() {
            for &id in retain_ids {
                if grads[id].is_none() {
                    grads[id] = Some(self.b.dev_alloc_zeros(self.lens[id])?);
                    live_elements = live_elements.saturating_add(self.lens[id]);
                    peak_persistent_grad_elements =
                        peak_persistent_grad_elements.max(live_elements);
                }
            }
        }
        for (id, slot) in grads.iter_mut().enumerate() {
            if !retain[id] && slot.take().is_some() {
                live_elements = live_elements.saturating_sub(self.lens[id]);
            }
        }
        debug_assert_eq!(
            self.live_activation_elements, 0,
            "all non-leaf activations must be released after reverse replay"
        );
        let expected_live = if stream.is_some() {
            0
        } else {
            retain
                .iter()
                .zip(&self.lens)
                .filter(|(keep, _)| **keep)
                .fold(0usize, |total, (_, &len)| total.saturating_add(len))
        };
        debug_assert_eq!(live_elements, expected_live);
        Ok(DeviceBackwardResult {
            grads,
            stats: DeviceBackwardStats {
                naive_all_value_grad_elements,
                peak_persistent_grad_elements,
                naive_activation_elements,
                peak_live_activation_elements: self.peak_live_activation_elements,
                retained_checkpoint_elements,
                recomputed_ops: self.recomputed_ops,
            },
        })
    }

    /// Reverse pass with lazy gradient slots. `retain_ids` are returned in
    /// their value-id slots; all other gradients are released once their
    /// producing op has consumed them.
    fn backward_retain(
        &mut self,
        seed_id: usize,
        seed: &CudaSlice<f32>,
        retain_ids: &[usize],
    ) -> Result<DeviceBackwardResult, BackendError> {
        self.backward_core(seed_id, seed, retain_ids, None)
    }

    /// One distillation-step gradient, host in / host out: seed the softmax-xent loss grad at the
    /// `logits` value, run the whole device-resident backward, and download the gradients for the
    /// requested value ids. Hides the device buffers entirely — the caller (the distillation loop)
    /// works in host `Vec<f32>` and never touches a `CudaSlice`. `want` is typically the weight-leaf
    /// ids; the returned grads are in `want` order.
    pub fn xent_backward(
        &mut self,
        logits: usize,
        target: &[f32],
        rows: usize,
        cols: usize,
        want: &[usize],
    ) -> Result<Vec<Vec<f32>>, BackendError> {
        let seed = self.softmax_xent_grad(logits, target, rows, cols)?;
        let result = self.backward_retain(logits, &seed, want)?;
        want.iter()
            .map(|&id| {
                let mut h = vec![0.0f32; self.lens[id]];
                self.b.dev_download(
                    result.grads[id]
                        .as_ref()
                        .expect("requested gradient is retained"),
                    &mut h,
                )?;
                Ok(h)
            })
            .collect()
    }

    /// Run a resident softmax-xent reverse pass and stream each requested leaf
    /// gradient into a host-offloaded AdamW parameter as soon as all of that
    /// leaf's reverse-topological consumer edges are complete.
    #[allow(clippy::too_many_arguments)]
    pub fn xent_backward_into(
        self,
        logits: usize,
        target: &DeviceTensor,
        rows: usize,
        cols: usize,
        bindings: &[GradientLeafBinding],
        trainer: &mut HostOffloadTrainer<'_>,
        step: u64,
    ) -> Result<GradientStreamReport, BackendError> {
        let mut transform = IdentityGradientTransform;
        self.xent_backward_into_with_transform(
            logits,
            target,
            rows,
            cols,
            bindings,
            trainer,
            step,
            &mut transform,
        )
    }

    /// Run a resident softmax-xent reverse pass and apply each finalized leaf
    /// gradient directly to device-resident AdamW state. This bounds requested
    /// parameter-gradient residency by reverse-topological liveness instead of
    /// materializing a full-model gradient collection.
    #[allow(clippy::too_many_arguments)]
    pub fn xent_backward_into_resident(
        self,
        logits: usize,
        target: &DeviceTensor,
        rows: usize,
        cols: usize,
        bindings: &[GradientLeafBinding],
        trainer: &mut DeviceTrainer<'_>,
        step: u64,
    ) -> Result<GradientStreamReport, BackendError> {
        let mut transform = IdentityGradientTransform;
        self.xent_backward_into_with_transform(
            logits,
            target,
            rows,
            cols,
            bindings,
            trainer,
            step,
            &mut transform,
        )
    }

    /// Compute the exact finalized-gradient emission sequence without running
    /// backward or mutating optimizer state.
    ///
    /// Distributed callers preflight this fixed manifest across ranks before
    /// entering any variable-length gradient collective.
    #[cfg(feature = "nccl")]
    pub(crate) fn gradient_stream_manifest(
        &self,
        bindings: &[GradientLeafBinding],
        trainer: &HostOffloadTrainer<'_>,
        step: u64,
    ) -> Result<Vec<GradientEmission>, BackendError> {
        Ok(self
            .gradient_completion_plan(bindings, trainer, step)?
            .manifest(&self.lens))
    }

    /// Validate the complete streamed softmax-xent call and return its exact
    /// gradient manifest without launching any kernels or mutating the trainer.
    ///
    /// NCCL callers exchange this result with a fixed-size preflight before any
    /// rank enters backward, so a local shape/configuration error cannot strand
    /// peers inside a gradient collective.
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "nccl")]
    pub(crate) fn xent_gradient_stream_manifest(
        &self,
        logits: usize,
        target: &DeviceTensor,
        rows: usize,
        cols: usize,
        bindings: &[GradientLeafBinding],
        trainer: &HostOffloadTrainer<'_>,
        step: u64,
    ) -> Result<Vec<GradientEmission>, BackendError> {
        self.validate_xent_stream_inputs(logits, target, rows, cols)?;
        self.gradient_stream_manifest(bindings, trainer, step)
    }

    fn validate_xent_stream_inputs(
        &self,
        logits: usize,
        target: &DeviceTensor,
        rows: usize,
        cols: usize,
    ) -> Result<(), BackendError> {
        if rows == 0 || cols == 0 {
            return Err(BackendError::InvalidInput(
                "softmax-xent rows and cols must be non-zero".into(),
            ));
        }
        let expected = rows.checked_mul(cols).ok_or_else(|| {
            BackendError::InvalidInput("softmax-xent shape overflows usize".into())
        })?;
        if logits >= self.vals.len() {
            return Err(BackendError::InvalidInput(format!(
                "logits value id {logits} is out of range"
            )));
        }
        if self.lens[logits] != expected {
            return Err(BackendError::ShapeMismatch {
                expected,
                got: self.lens[logits],
            });
        }
        if target.len() != expected {
            return Err(BackendError::ShapeMismatch {
                expected,
                got: target.len(),
            });
        }
        if !self.b.same_context(&target.buf) {
            return Err(BackendError::InvalidInput(
                "target tensor belongs to a different CUDA context".into(),
            ));
        }
        Ok(())
    }

    /// Stream finalized gradients through an in-place transform before each
    /// optimizer-sink update.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn xent_backward_into_with_transform(
        mut self,
        logits: usize,
        target: &DeviceTensor,
        rows: usize,
        cols: usize,
        bindings: &[GradientLeafBinding],
        sink: &mut dyn GradientOptimizerSink,
        step: u64,
        transform: &mut dyn FinalizedGradientTransform,
    ) -> Result<GradientStreamReport, BackendError> {
        self.validate_xent_stream_inputs(logits, target, rows, cols)?;
        let plan = self.gradient_completion_plan(bindings, sink, step)?;
        let expected_manifest = plan.manifest(&self.lens);
        let retain_ids: Vec<usize> = plan
            .bindings
            .iter()
            .map(|binding| binding.leaf_id)
            .collect();
        let seed = self.softmax_xent_grad_device(logits, &target.buf, rows, cols)?;
        let mut stream = GradientStream {
            sink,
            transform,
            plan,
            step,
            emissions: Vec::with_capacity(bindings.len()),
            peak_live_requested_gradient_elements: 0,
            mutation_started: false,
        };
        let backward = self.backward_core(logits, &seed, &retain_ids, Some(&mut stream));
        let result = match backward {
            Ok(result) => result,
            Err(error) => {
                if stream.mutation_started {
                    stream.sink.abort_gradient_stream();
                }
                return Err(error);
            }
        };
        if stream.emissions != expected_manifest
            || stream.plan.remaining_edges.iter().any(|&count| count != 0)
        {
            if stream.mutation_started {
                stream.sink.abort_gradient_stream();
            }
            return Err(BackendError::InvalidInput(
                "gradient stream finished before every binding was emitted".into(),
            ));
        }
        stream
            .sink
            .finish_gradient_stream(step, stream.plan.materialized_collection_elements)?;
        Ok(GradientStreamReport {
            emissions: stream.emissions,
            materialized_collection_elements: stream.plan.materialized_collection_elements,
            peak_live_requested_gradient_elements: stream.peak_live_requested_gradient_elements,
            backward_stats: result.stats,
        })
    }

    /// One fully resident distillation backward pass. The tape is consumed so
    /// all borrowed parameter tensors are released before a caller mutates
    /// their masters or optimizer state. Requested gradients are moved into the
    /// result without a device-to-device copy and preserve `want` order.
    pub fn xent_backward_device(
        mut self,
        logits: usize,
        target: &DeviceTensor,
        rows: usize,
        cols: usize,
        want: &[usize],
    ) -> Result<DeviceGradients, BackendError> {
        if rows == 0 || cols == 0 {
            return Err(BackendError::InvalidInput(
                "softmax-xent rows and cols must be non-zero".into(),
            ));
        }
        let expected = rows.checked_mul(cols).ok_or_else(|| {
            BackendError::InvalidInput("softmax-xent shape overflows usize".into())
        })?;
        if logits >= self.vals.len() {
            return Err(BackendError::InvalidInput(format!(
                "logits value id {logits} is out of range"
            )));
        }
        if self.lens[logits] != expected {
            return Err(BackendError::ShapeMismatch {
                expected,
                got: self.lens[logits],
            });
        }
        if target.len() != expected {
            return Err(BackendError::ShapeMismatch {
                expected,
                got: target.len(),
            });
        }
        if !self.b.same_context(&target.buf) {
            return Err(BackendError::InvalidInput(
                "target tensor belongs to a different CUDA context".into(),
            ));
        }
        for (position, &id) in want.iter().enumerate() {
            if id >= self.vals.len() {
                return Err(BackendError::InvalidInput(format!(
                    "requested gradient value id {id} is out of range"
                )));
            }
            if want[..position].contains(&id) {
                return Err(BackendError::InvalidInput(format!(
                    "requested gradient value id {id} is duplicated"
                )));
            }
            if !self.leaves[id] {
                return Err(BackendError::InvalidInput(format!(
                    "requested gradient value id {id} is not a leaf"
                )));
            }
        }

        let seed = self.softmax_xent_grad_device(logits, &target.buf, rows, cols)?;
        let mut result = self.backward_retain(logits, &seed, want)?;
        let bufs = want
            .iter()
            .map(|&id| {
                result.grads[id]
                    .take()
                    .expect("want ids were validated unique and retained")
            })
            .collect();
        Ok(DeviceGradients {
            bufs,
            stats: result.stats,
        })
    }
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

    #[test]
    fn host_offload_memory_geometry_covers_packing_optimizer_and_gradients() {
        let parameters = [
            HostOffloadParamMetadata {
                rows: 2,
                cols: 257,
                salt_planes: 2,
            },
            HostOffloadParamMetadata {
                rows: 3,
                cols: 1,
                salt_planes: 2,
            },
        ];

        assert_eq!(
            host_offload_memory_geometry(&parameters).unwrap(),
            HostOffloadMemoryGeometry {
                packed_parameter_bytes: 936,
                packed_code_bytes: 896,
                packed_scale_bytes: 40,
                dense_parameter_bytes: 2_068,
                largest_parameter_bytes: 2_056,
                host_optimizer_bytes: 6_204,
                peak_optimizer_staging_bytes: 12_336,
                materialized_gradient_bytes: 2_068,
            }
        );
    }

    #[test]
    fn host_offload_memory_geometry_rejects_invalid_planes() {
        for salt_planes in [0, 4] {
            let error = host_offload_memory_geometry(&[HostOffloadParamMetadata {
                rows: 1,
                cols: 1,
                salt_planes,
            }])
            .unwrap_err();
            assert!(
                matches!(error, BackendError::InvalidInput(message) if message.contains("planes"))
            );
        }
    }

    #[test]
    fn host_offload_memory_geometry_rejects_shape_overflow() {
        let error = host_offload_memory_geometry(&[HostOffloadParamMetadata {
            rows: usize::MAX,
            cols: 2,
            salt_planes: 1,
        }])
        .unwrap_err();
        assert!(matches!(error, BackendError::InvalidInput(message) if message.contains("shape")));
    }

    #[test]
    fn host_offload_scheduler_overlaps_two_slots_and_drains() {
        let pending = |parameter_index, len| PendingOffload {
            parameter_index,
            len,
        };
        let mut schedule = DoubleBufferSchedule::default();

        assert_eq!(
            schedule.enqueue(pending(4, 17)).unwrap(),
            OffloadTransition {
                target_slot: 0,
                reclaim: None,
                download_slot: None,
            }
        );
        assert_eq!(
            schedule.enqueue(pending(2, 257)).unwrap(),
            OffloadTransition {
                target_slot: 1,
                reclaim: None,
                download_slot: Some(0),
            }
        );
        assert_eq!(
            schedule.enqueue(pending(9, 63)).unwrap(),
            OffloadTransition {
                target_slot: 0,
                reclaim: Some(pending(4, 17)),
                download_slot: Some(1),
            }
        );
        assert_eq!(schedule.peak_in_flight, 2);
        assert_eq!(schedule.begin_finish().unwrap(), Some(0));
        assert_eq!(
            schedule.pending_downloads().collect::<Vec<_>>(),
            vec![(0, pending(9, 63)), (1, pending(2, 257))]
        );
        schedule.reset();
        assert!(schedule.pending_downloads().next().is_none());
        assert_eq!(schedule.active_compute, None);
    }

    #[test]
    fn device_tape_gradient_slots_are_lazy() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping device_tape_gradient_slots_are_lazy: no CUDA device ({e})");
                return;
            }
        };
        let n = 4096;
        let x = seeded_uniform(0xC001, n, -1.0, 1.0);
        let seed = seeded_uniform(0xC002, n, -0.5, 0.5);
        let mut tape = DeviceTape::new(&backend, 1).unwrap();
        let x_id = tape.leaf(&x).unwrap();
        let mut out = x_id;
        for _ in 0..16 {
            out = tape.silu(out).unwrap();
        }
        let d_seed = backend.dev_upload(&seed).unwrap();

        let result = tape.backward_retain(out, &d_seed, &[x_id]).unwrap();

        assert!(result.grads[x_id].is_some());
        assert!(
            result.stats.peak_persistent_grad_elements < result.stats.naive_all_value_grad_elements,
            "lazy gradient slots must beat one persistent gradient per value: {:?}",
            result.stats
        );
    }

    #[test]
    fn device_tape_hestia_relax_uses_dense_matmul_and_multistage_tau_vjp() {
        use tritium_train::ops::{dense, hestia};

        let backend = match CudaBackend::new(0) {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping DeviceTape HESTIA test: no CUDA device ({error})");
                return;
            }
        };
        let (m, rows, cols) = (2usize, 3usize, 769usize);
        let x = seeded_uniform(0xC301, m * cols, -0.75, 0.75);
        let weight = seeded_uniform(0xC302, rows * cols, -1.25, 1.25);
        let scale = [0.8f32, 1.0, 0.6];
        let tau = [0.7f32];
        let seed = seeded_uniform(0xC303, m * rows, -0.5, 0.5);

        let soft = hestia::hestia_forward(&weight, &scale, &tau, rows, cols);
        let expected_output = dense::forward(&x, &soft, m, rows, cols);
        let dense_grad = dense::vjp(&x, &soft, m, rows, cols, &seed);
        let hestia_grad = hestia::hestia_vjp(&weight, &scale, &tau, rows, cols, &dense_grad[1]);

        let mut tape = DeviceTape::new_with_checkpoint_policy(
            &backend,
            rows,
            CheckpointPolicy::EveryBlocks(1),
        )
        .unwrap();
        let x_id = tape.leaf(&x).unwrap();
        let weight_id = tape.leaf(&weight).unwrap();
        let scale_id = tape.leaf(&scale).unwrap();
        let tau_id = tape.leaf(&tau).unwrap();
        let soft_id = tape
            .hestia_relax(weight_id, scale_id, tau_id, rows, cols)
            .unwrap();
        let output_id = tape.matmul(x_id, soft_id, m, rows, cols).unwrap();
        let output_diff = max_abs_diff(&tape.value(output_id).unwrap(), &expected_output);
        assert!(
            output_diff < 1e-5,
            "HESTIA dense matmul forward max abs diff {output_diff}"
        );
        tape.checkpoint_keep(&[output_id]).unwrap();
        let device_seed = backend.dev_upload(&seed).unwrap();
        let result = tape
            .backward_retain(
                output_id,
                &device_seed,
                &[x_id, weight_id, scale_id, tau_id],
            )
            .unwrap();
        let download = |id: usize| {
            let mut host = vec![0.0f32; result.grads[id].as_ref().unwrap().len()];
            backend
                .dev_download(result.grads[id].as_ref().unwrap(), &mut host)
                .unwrap();
            host
        };
        assert!(max_abs_diff(&download(x_id), &dense_grad[0]) < 1e-5);
        assert!(max_abs_diff(&download(weight_id), &hestia_grad[0]) < 1e-5);
        assert_eq!(download(scale_id), hestia_grad[1]);
        assert!(max_abs_diff(&download(tau_id), &hestia_grad[2]) < 1e-4);
        assert!(result.stats.recomputed_ops >= 2);
    }

    #[test]
    fn device_tape_hestia_unrepresentable_tau_fails_closed() {
        use tritium_train::ops::hestia;

        let backend = match CudaBackend::new(0) {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping DeviceTape HESTIA test: no CUDA device ({error})");
                return;
            }
        };
        let weight = [0.25f32, -0.75, 1.0, -1.5];
        let scale = [0.5f32, 1.25];
        let tau = [hestia::MIN_DIFFERENTIABLE_TAU * 0.5];
        let seed = [1.0f32; 4];

        let mut tape = DeviceTape::new(&backend, 2).unwrap();
        let weight_id = tape.leaf(&weight).unwrap();
        let scale_id = tape.leaf(&scale).unwrap();
        let tau_id = tape.leaf(&tau).unwrap();
        let output_id = tape
            .hestia_relax(weight_id, scale_id, tau_id, 2, 2)
            .unwrap();
        assert_eq!(tape.value(output_id).unwrap(), vec![0.0; 4]);

        let device_seed = backend.dev_upload(&seed).unwrap();
        let result = tape
            .backward_retain(output_id, &device_seed, &[weight_id, scale_id, tau_id])
            .unwrap();
        for id in [weight_id, scale_id, tau_id] {
            let gradient = result.grads[id].as_ref().unwrap();
            let mut host = vec![f32::NAN; gradient.len()];
            backend.dev_download(gradient, &mut host).unwrap();
            assert!(host.iter().all(|value| *value == 0.0));
        }
    }

    #[test]
    fn device_tape_hestia_exact_temperature_floor_has_finite_vjp() {
        use tritium_train::ops::hestia;

        let backend = match CudaBackend::new(0) {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping DeviceTape HESTIA test: no CUDA device ({error})");
                return;
            }
        };
        let mut tape = DeviceTape::new(&backend, 1).unwrap();
        let weight_id = tape.leaf(&[1.0]).unwrap();
        let scale_id = tape.leaf(&[1.0]).unwrap();
        let tau_id = tape.leaf(&[hestia::MIN_DIFFERENTIABLE_TAU]).unwrap();
        let output_id = tape
            .hestia_relax(weight_id, scale_id, tau_id, 1, 1)
            .unwrap();
        assert_eq!(tape.value(output_id).unwrap(), vec![1.0]);

        let device_seed = backend.dev_upload(&[1.0]).unwrap();
        let result = tape
            .backward_retain(output_id, &device_seed, &[weight_id, scale_id, tau_id])
            .unwrap();
        for id in [weight_id, scale_id, tau_id] {
            let gradient = result.grads[id].as_ref().unwrap();
            let mut host = vec![f32::NAN; gradient.len()];
            backend.dev_download(gradient, &mut host).unwrap();
            assert_eq!(host, vec![0.0; gradient.len()]);
        }
    }

    #[test]
    fn checkpoint_policy_rejects_zero_interval() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping checkpoint_policy_rejects_zero_interval: no CUDA ({e})");
                return;
            }
        };
        assert!(matches!(
            DeviceTape::new_with_checkpoint_policy(&backend, 1, CheckpointPolicy::EveryBlocks(0)),
            Err(BackendError::InvalidInput(_))
        ));
        assert!(matches!(
            DeviceTape::new_with_checkpoint_policy(&backend, 1, CheckpointPolicy::SqrtDepth(0)),
            Err(BackendError::InvalidInput(_))
        ));
    }

    #[test]
    fn checkpoint_frontier_fails_closed() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping checkpoint_frontier_fails_closed: no CUDA ({e})");
                return;
            }
        };
        let mut tape =
            DeviceTape::new_with_checkpoint_policy(&backend, 1, CheckpointPolicy::EveryBlocks(1))
                .unwrap();
        let x = tape.leaf(&[0.25; 32]).unwrap();
        let internal = tape.silu(x).unwrap();
        let frontier = tape.silu(internal).unwrap();
        tape.checkpoint_keep(&[frontier]).unwrap();

        assert!(matches!(
            tape.value(internal),
            Err(BackendError::InvalidInput(message)) if message.contains("was evicted")
        ));
        assert!(matches!(
            tape.silu(internal),
            Err(BackendError::InvalidInput(message)) if message.contains("was evicted")
        ));
        assert!(matches!(
            tape.checkpoint_keep(&[usize::MAX]),
            Err(BackendError::InvalidInput(_))
        ));
        assert!(matches!(
            tape.checkpoint_keep(&[frontier, frontier]),
            Err(BackendError::InvalidInput(message)) if message.contains("duplicated")
        ));
        assert!(matches!(
            tape.checkpoint_keep(&[x]),
            Err(BackendError::InvalidInput(message)) if message.contains("leaf")
        ));
        assert!(matches!(
            tape.checkpoint_keep(&[frontier]),
            Err(BackendError::InvalidInput(message)) if message.contains("no forward operations")
        ));

        let mut incomplete =
            DeviceTape::new_with_checkpoint_policy(&backend, 1, CheckpointPolicy::SqrtDepth(4))
                .unwrap();
        let input = incomplete.leaf(&[0.1; 32]).unwrap();
        let output = incomplete.silu(input).unwrap();
        incomplete.checkpoint_keep(&[output]).unwrap();
        let seed = backend.dev_upload(&[1.0; 32]).unwrap();
        assert!(matches!(
            incomplete.backward_retain(output, &seed, &[input]),
            Err(BackendError::InvalidInput(message)) if message.contains("marker count")
        ));
    }

    fn run_checkpoint_chain(
        backend: &CudaBackend,
        policy: CheckpointPolicy,
        depth: usize,
        width: usize,
    ) -> (Vec<f32>, Vec<f32>, DeviceBackwardStats) {
        let input = seeded_uniform(0xC027, width, -0.05, 0.05);
        let seed = seeded_uniform(0xC028, width, -0.25, 0.25);
        let mut tape = DeviceTape::new_with_checkpoint_policy(backend, 1, policy).unwrap();
        let input_id = tape.leaf(&input).unwrap();
        let mut hidden = input_id;
        for _ in 0..depth {
            hidden = tape.silu(hidden).unwrap();
            hidden = tape.add(hidden, input_id).unwrap();
            tape.checkpoint_keep(&[hidden]).unwrap();
        }
        let output = tape.value(hidden).unwrap();
        let d_seed = backend.dev_upload(&seed).unwrap();
        let result = tape.backward_retain(hidden, &d_seed, &[input_id]).unwrap();
        let mut gradient = vec![0.0; width];
        backend
            .dev_download(
                result.grads[input_id]
                    .as_ref()
                    .expect("input gradient retained"),
                &mut gradient,
            )
            .unwrap();
        (output, gradient, result.stats)
    }

    #[test]
    fn multi_block_checkpoint_recompute_matches_keep_all() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "skipping multi_block_checkpoint_recompute_matches_keep_all: no CUDA ({e})"
                );
                return;
            }
        };
        let (depth, width) = (9usize, 1024usize);
        let (keep_output, keep_gradient, keep_stats) =
            run_checkpoint_chain(&backend, CheckpointPolicy::KeepAll, depth, width);
        let (checkpoint_output, checkpoint_gradient, checkpoint_stats) =
            run_checkpoint_chain(&backend, CheckpointPolicy::SqrtDepth(depth), depth, width);

        assert!(max_abs_diff(&keep_output, &checkpoint_output) < 1e-6);
        assert!(max_abs_diff(&keep_gradient, &checkpoint_gradient) < 1e-6);
        assert_eq!(keep_stats.recomputed_ops, 0);
        assert_eq!(checkpoint_stats.recomputed_ops, depth * 2);
        assert_eq!(
            checkpoint_stats.naive_activation_elements,
            keep_stats.naive_activation_elements
        );
        assert!(
            checkpoint_stats.peak_live_activation_elements
                < checkpoint_stats.naive_activation_elements,
            "checkpointing must reduce logical activation peak: {checkpoint_stats:?}"
        );
        assert!(
            checkpoint_stats.peak_live_activation_elements
                < keep_stats.peak_live_activation_elements,
            "checkpointing must beat KeepAll: checkpoint={checkpoint_stats:?}, \
             keep={keep_stats:?}"
        );
        eprintln!(
            "0027 Track C multi-block checkpoint: activation peak {}/{} elements, retained {}, \
             recomputed {} ops",
            checkpoint_stats.peak_live_activation_elements,
            checkpoint_stats.naive_activation_elements,
            checkpoint_stats.retained_checkpoint_elements,
            checkpoint_stats.recomputed_ops
        );
    }

    #[test]
    fn sqrt_depth_checkpoint_peak_scales_sublinearly() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping sqrt_depth_checkpoint_peak_scales_sublinearly: no CUDA ({e})");
                return;
            }
        };
        let width = 256usize;
        for depth in [4usize, 16, 64] {
            let input = seeded_uniform(0xD000 + depth as u64, width, -0.1, 0.1);
            let mut tape = DeviceTape::new_with_checkpoint_policy(
                &backend,
                1,
                CheckpointPolicy::SqrtDepth(depth),
            )
            .unwrap();
            let input_id = tape.leaf(&input).unwrap();
            let mut hidden = input_id;
            for _ in 0..depth {
                hidden = tape.silu(hidden).unwrap();
                tape.checkpoint_keep(&[hidden]).unwrap();
            }
            let seed = backend.dev_upload(&vec![1.0; width]).unwrap();
            let stats = tape
                .backward_retain(hidden, &seed, &[input_id])
                .unwrap()
                .stats;
            let interval = (depth as f64).sqrt().ceil() as usize;
            let materialized = depth.div_ceil(interval);
            let upper_bound = (materialized + interval) * width;
            assert!(
                stats.peak_live_activation_elements <= upper_bound,
                "depth {depth} exceeds O(depth/k + k) bound {upper_bound}: {stats:?}"
            );
            assert!(
                stats.peak_live_activation_elements < stats.naive_activation_elements,
                "depth {depth} did not reduce activation peak: {stats:?}"
            );
            assert_eq!(stats.recomputed_ops, depth);
        }
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

    /// Gate (plan 0043 P2.2): the device-resident glue kernels (silu, elementwise mul/add, grad
    /// accumulate) match their `tritium-train` CPU ops. The pure `+`/`*` ops are BIT-EXACT; silu
    /// carries `expf` (device vs host libm ~1 ULP) so it is gated device==CPU within 1e-4.
    /// **The wiring gate.** `DeviceTrainer::new_with_fitter(.., Some(SaltGrouping))` must make
    /// `prepare_quantized` reproduce the host fitter `ste::salt_quantize_forward_grouped(.., Auto)`
    /// on every parameter — that is the entire point of the grouped path, and the previous parity
    /// gate only covered the raw kernel, not the trainer that drives it.
    ///
    /// Also pins the two contracts that make the wiring safe: `None` keeps the legacy per-row
    /// reconstruction bit-for-bit (so no committed result moves), and Sherry is refused rather than
    /// silently ignored on the grouped path.
    #[test]
    fn device_trainer_grouped_fitter_matches_host() {
        let Ok(backend) = CudaBackend::new(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        // A ragged row (576 = 4x128 + 64) exercises the never-rotated tail block.
        for &(rows, cols) in &[(6usize, 256usize), (4, 576)] {
            let master: Vec<f32> = (0..rows * cols)
                .map(|i| {
                    let x = ((i * 2654435761) % 4096) as f32 / 4096.0 - 0.5;
                    // Heavy tails on a few coordinates so Auto genuinely splits both ways.
                    if i % 97 == 0 { x * 9.0 } else { x }
                })
                .collect();
            for planes in 1..=3usize {
                for group in [128usize, 256] {
                    for iters in [0usize, 1, 5] {
                        let grouping = SaltGrouping {
                            ladder: SaltLadder::Itf { iters },
                            group,
                            rotation: ste::RotationPolicy::Auto,
                        };
                        let params = [DeviceTrainParam {
                            master: &master,
                            rows,
                            cols,
                            salt_planes: planes,
                            optimizer: AdamW::new(1e-3),
                        }];
                        let mut trainer = DeviceTrainer::new_with_fitter(
                            &backend,
                            &params,
                            DeviceTrainerWeightStorage::DenseQuantized,
                            MomentPrecision::F32,
                            MasterPrecision::F32,
                            Some(grouping),
                        )
                        .expect("grouped trainer");
                        trainer.prepare_quantized().expect("prepare");
                        let mut got = vec![0.0f32; master.len()];
                        backend
                            .dev_download(&trainer.quantized(0).unwrap().buf, &mut got)
                            .expect("download");
                        let want = ste::salt_quantize_forward_grouped(
                            &master,
                            rows,
                            cols,
                            planes,
                            group,
                            iters,
                            ste::RotationPolicy::Auto,
                        );
                        let delta = got
                            .iter()
                            .zip(&want)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0f32, f32::max);
                        assert!(
                            delta < 2e-4,
                            "grouped trainer {rows}x{cols} g{group} T{planes} itf{iters}: max|delta| {delta:.3e}"
                        );
                        // Vacuity guard: with ITF on, the fit must actually MOVE off the greedy
                        // one, otherwise this comparison could pass with the iterations never firing.
                        if iters > 0 {
                            let greedy = ste::salt_quantize_forward_grouped(
                                &master,
                                rows,
                                cols,
                                planes,
                                group,
                                0,
                                ste::RotationPolicy::Auto,
                            );
                            let moved = got
                                .iter()
                                .zip(&greedy)
                                .map(|(a, b)| (a - b).abs())
                                .fold(0.0f32, f32::max);
                            assert!(
                                moved > 1e-6,
                                "ITF never fired: itf{iters} equals greedy ({rows}x{cols} g{group} T{planes})"
                            );
                        }

                        // Sherry must be refused, not quietly dropped.
                        trainer.set_sherry_alpha(0.25);
                        let err = trainer.prepare_quantized().unwrap_err();
                        assert!(
                            matches!(&err, BackendError::InvalidInput(m) if m.contains("Sherry")),
                            "grouped + Sherry must error, got {err:?}"
                        );
                    }
                }

                // `None` keeps the committed per-row reconstruction bit-for-bit.
                let params = [DeviceTrainParam {
                    master: &master,
                    rows,
                    cols,
                    salt_planes: planes,
                    optimizer: AdamW::new(1e-3),
                }];
                let mut legacy = DeviceTrainer::new(&backend, &params).expect("legacy trainer");
                legacy.prepare_quantized().expect("prepare");
                let mut got = vec![0.0f32; master.len()];
                backend
                    .dev_download(&legacy.quantized(0).unwrap().buf, &mut got)
                    .expect("download");
                let want = ste::salt_quantize_forward(&master, rows, cols, planes);
                let delta = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    delta < 2e-4,
                    "ungrouped must stay on the legacy path {rows}x{cols} T{planes}: {delta:.3e}"
                );
            }
        }
    }

    /// The grouped SALT kernel (finer scale groups + optional per-group Hadamard) must match the CPU
    /// oracle `ste::salt_quantize_forward_grouped`. This is the fitter that took PTQ from 10786× to
    /// 1.74× fp, so the device has to reproduce it before any distillation run can use it.
    #[test]
    fn salt_quantize_grouped_matches_cpu_oracle() {
        use tritium_train::ops::ste::{self, RotationPolicy};
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "skipping salt_quantize_grouped_matches_cpu_oracle: no CUDA device ({e})"
                );
                return;
            }
        };
        // Include a shape whose row is NOT a multiple of the group, to exercise the ragged tail
        // (which must stay unrotated on both sides).
        for &(rows, cols) in &[(4usize, 256usize), (3, 128), (5, 576)] {
            let master = seeded_uniform(0x9A17 ^ cols as u64, rows * cols, -2.0, 2.0);
            let d_master = backend.dev_upload(&master).unwrap();
            let mut d_out = backend.dev_alloc_zeros(rows * cols).unwrap();

            for group in [128usize, 256] {
                let groups_per_row = cols.div_ceil(group);
                let groups = rows * groups_per_row;
                for planes in 1..=3 {
                    // (a) no rotation
                    backend
                        .salt_quantize_forward_grouped_dev(
                            &d_master, &mut d_out, None, rows, cols, planes, group, 0,
                        )
                        .unwrap();
                    let mut got = vec![0.0f32; rows * cols];
                    backend.dev_download(&d_out, &mut got).unwrap();
                    let want = ste::salt_quantize_forward_grouped(
                        &master,
                        rows,
                        cols,
                        planes,
                        group,
                        0,
                        RotationPolicy::Never,
                    );
                    let d = max_abs_diff(&got, &want);
                    assert!(
                        d < 1e-4,
                        "grouped (no rot) {rows}x{cols} g{group} T{planes}: max|delta| {d:.3e}"
                    );

                    // (b) rotate every group -> compare against the host Always policy
                    let mask = vec![1u8; groups];
                    let d_mask = backend.stream().clone_htod(&mask).unwrap();
                    backend
                        .salt_quantize_forward_grouped_dev(
                            &d_master,
                            &mut d_out,
                            Some(&d_mask),
                            rows,
                            cols,
                            planes,
                            group,
                            0,
                        )
                        .unwrap();
                    backend.dev_download(&d_out, &mut got).unwrap();
                    let want_rot = ste::salt_quantize_forward_grouped(
                        &master,
                        rows,
                        cols,
                        planes,
                        group,
                        0,
                        RotationPolicy::Always,
                    );
                    let d = max_abs_diff(&got, &want_rot);
                    assert!(
                        d < 1e-3,
                        "grouped (rotated) {rows}x{cols} g{group} T{planes}: max|delta| {d:.3e}"
                    );
                }
            }
        }
        eprintln!(
            "grouped SALT kernel matches the CPU oracle (rotated and not, incl. ragged tails)"
        );
    }

    /// The trainer driving the ladder end to end: `prepare_quantized` must reproduce the host
    /// fitter bit-for-bit, at plane counts the ITF path cannot reach.
    ///
    /// The kernel gate below proves the kernel; this proves the WIRING — that `SaltGrouping`
    /// dispatches to it, that the rotation mask is built with the ladder's own fitter rather than
    /// the ITF one (deriving it from the wrong fitter is the train/eval mismatch of task #76), and
    /// that the `1..=3` cap is lifted only for the fitter that can honour it.
    #[test]
    fn device_trainer_geometric_ladder_matches_host_and_lifts_the_plane_cap() {
        use tritium_train::ops::ste::{self, RotationPolicy};
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping trainer ladder wiring: no CUDA device ({e})");
                return;
            }
        };
        let (rows, cols) = (4usize, 256usize);
        let master = seeded_uniform(0x1AD_7A11, rows * cols, -2.0, 2.0);
        let bits = |v: &[f32]| -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() };

        // 6 is past the ITF path's hard cap of 3 — the point of the ladder on device.
        for planes in [1usize, 3, 6] {
            let grouping = SaltGrouping {
                ladder: SaltLadder::Geometric { grid: 16 },
                group: 128,
                rotation: RotationPolicy::Auto,
            };
            let params = [DeviceTrainParam {
                master: &master,
                rows,
                cols,
                salt_planes: planes,
                optimizer: AdamW::new(1e-3),
            }];
            let mut trainer = DeviceTrainer::new_with_fitter(
                &backend,
                &params,
                DeviceTrainerWeightStorage::DenseQuantized,
                MomentPrecision::F32,
                MasterPrecision::F32,
                Some(grouping),
            )
            .expect("trainer with the geometric ladder");
            trainer.prepare_quantized().expect("prepare");
            let mut got = vec![0.0f32; rows * cols];
            backend
                .dev_download(&trainer.quantized(0).expect("quantized").buf, &mut got)
                .unwrap();

            // The host oracle must use the SAME frozen mask the trainer built, not a fresh `Auto`
            // decision — `Auto` re-decides per call and would introduce a second mismatch.
            let mask = ste::geometric_rotation_mask(
                &master,
                rows,
                cols,
                planes,
                128,
                16,
                RotationPolicy::Auto,
            );
            let mut want = vec![0.0f32; rows * cols];
            let per_row = cols.div_ceil(128);
            for r in 0..rows {
                for b in 0..per_row {
                    let lo = r * cols + b * 128;
                    let hi = (lo + 128).min((r + 1) * cols);
                    let policy = if mask[r * per_row + b] == 1 {
                        RotationPolicy::Always
                    } else {
                        RotationPolicy::Never
                    };
                    let fit = ste::salt_quantize_forward_grouped_geometric(
                        &master[lo..hi],
                        1,
                        hi - lo,
                        planes,
                        128,
                        16,
                        policy,
                    );
                    want[lo..hi].copy_from_slice(&fit);
                }
            }
            assert_eq!(
                bits(&got),
                bits(&want),
                "trainer ladder T{planes}: device diverged from the host fitter"
            );
        }

        // And the cap is per-fitter, not global: the ITF path must still refuse T=6.
        let itf = SaltGrouping {
            ladder: SaltLadder::Itf { iters: 5 },
            group: 128,
            rotation: RotationPolicy::Auto,
        };
        let params = [DeviceTrainParam {
            master: &master,
            rows,
            cols,
            salt_planes: 6,
            optimizer: AdamW::new(1e-3),
        }];
        assert!(
            DeviceTrainer::new_with_fitter(
                &backend,
                &params,
                DeviceTrainerWeightStorage::DenseQuantized,
                MomentPrecision::F32,
                MasterPrecision::F32,
                Some(itf),
            )
            .is_err(),
            "the ITF path must still reject T=6 — its cap is a format/enumeration limit, not a \
             global one, and silently accepting it would produce a fit the packer cannot store"
        );
        eprintln!("trainer ladder wiring bit-exact at T=1/3/6; ITF cap still enforced");
    }

    /// The **balanced-ternary ladder** kernel against its CPU oracle, asserted BIT-EXACT.
    ///
    /// The ITF gate above has to allow `1e-4`, because its fit runs a least-squares scale solve and
    /// an accept guard whose block reduction cannot reproduce the host's sequential summation order.
    /// The ladder has neither: one `roundf` per weight, then base-3 digit extraction and a sum of
    /// `T` scaled trits. Everything after that round is integer, so there is no float slop left to
    /// tolerate and a tolerance would only hide a real divergence.
    ///
    /// The one float decision is which `Δ` grid candidate wins, and both sides sweep the same fixed
    /// candidate set taking strictly-less. Candidates are `2^(1/4)` apart — far outside reduction
    /// noise — so the argmin agrees; this assertion is what would make a flip visible rather than
    /// silently changing the fit under training.
    #[test]
    fn salt_quantize_geometric_matches_cpu_oracle_bit_exact() {
        use tritium_train::ops::ste::{self, RotationPolicy};
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping geometric ladder parity: no CUDA device ({e})");
                return;
            }
        };
        let bits = |v: &[f32]| -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() };
        let mut checked = 0usize;
        let mut moved_vs_itf = 0usize;

        // 576 = 4*128 + 64 gives a ragged tail that must stay unrotated on both sides.
        for &(rows, cols) in &[(4usize, 256usize), (3, 128), (5, 576)] {
            let master = seeded_uniform(0x1AD_DE12 ^ cols as u64, rows * cols, -2.0, 2.0);
            let d_master = backend.dev_upload(&master).unwrap();
            let mut d_out = backend.dev_alloc_zeros(rows * cols).unwrap();

            for group in [128usize, 256] {
                let groups = rows * cols.div_ceil(group);
                let ones = vec![1u8; groups];
                let d_mask = backend.stream().clone_htod(&ones).unwrap();
                // Planes to 8: the ladder is O(T), so the 1..=3 cap the 3^T enumeration forces on
                // the joint fitter does not apply here, and T>3 is exactly what this unlocks.
                for planes in 1..=8usize {
                    for grid in [0usize, 4, 16] {
                        for rotate in [false, true] {
                            let mask = rotate.then_some(&d_mask);
                            backend
                                .salt_quantize_forward_grouped_geometric_dev(
                                    &d_master, &mut d_out, mask, rows, cols, planes, group, grid,
                                )
                                .unwrap();
                            let mut got = vec![0.0f32; rows * cols];
                            backend.dev_download(&d_out, &mut got).unwrap();
                            let policy = if rotate {
                                RotationPolicy::Always
                            } else {
                                RotationPolicy::Never
                            };
                            let want = ste::salt_quantize_forward_grouped_geometric(
                                &master, rows, cols, planes, group, grid, policy,
                            );
                            assert_eq!(
                                bits(&got),
                                bits(&want),
                                "ladder {rows}x{cols} g{group} T{planes} grid{grid} rot={rotate}: \
                                 device diverged from the host oracle"
                            );
                            checked += 1;

                            // Vacuity guard: a stubbed kernel that echoed the ITF fit would pass a
                            // loose gate. Assert the ladder actually produces a DIFFERENT fit.
                            if !rotate && planes <= 3 {
                                let itf = ste::salt_quantize_forward_grouped(
                                    &master,
                                    rows,
                                    cols,
                                    planes,
                                    group,
                                    5,
                                    RotationPolicy::Never,
                                );
                                if max_abs_diff(&want, &itf) > 1e-6 {
                                    moved_vs_itf += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(checked >= 200, "sweep should cover the grid, got {checked}");
        assert!(
            moved_vs_itf > 0,
            "the ladder never differed from the ITF fit — the gate would pass on a stub"
        );
        eprintln!(
            "geometric ladder: {checked} configurations bit-exact vs the CPU oracle \
             ({moved_vs_itf} confirmed distinct from the ITF fit)"
        );
    }

    #[test]
    fn resident_glue_ops_match_cpu() {
        use tritium_train::ops::{act, elementwise};

        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping resident_glue_ops_match_cpu: no CUDA device ({e})");
                return;
            }
        };
        let n = 4096;
        let x = seeded_uniform(0xAA, n, -4.0, 4.0); // wide range → exercise sigmoid saturation
        let gy = seeded_uniform(0xBB, n, -1.0, 1.0);
        let b = seeded_uniform(0xCC, n, -2.0, 2.0);
        let d_x = backend.dev_upload(&x).unwrap();
        let d_gy = backend.dev_upload(&gy).unwrap();
        let d_b = backend.dev_upload(&b).unwrap();
        let download = |d: &_| {
            let mut h = vec![0.0f32; n];
            backend.dev_download(d, &mut h).unwrap();
            h
        };

        // silu forward + backward — expf ⇒ within 1e-4, not bit-exact.
        let mut d_y = backend.dev_alloc_zeros(n).unwrap();
        backend.silu_forward_dev(&d_x, &mut d_y, n).unwrap();
        assert!(
            max_abs_diff(&download(&d_y), &act::silu_forward(&x)) < 1e-4,
            "silu forward"
        );
        let mut d_gx = backend.dev_alloc_zeros(n).unwrap();
        backend
            .silu_backward_dev(&d_x, &d_gy, &mut d_gx, n)
            .unwrap();
        assert!(
            max_abs_diff(&download(&d_gx), &act::silu_vjp(&x, &gy)[0]) < 1e-4,
            "silu backward"
        );

        // mul forward/backward — bit-exact.
        let mut d_m = backend.dev_alloc_zeros(n).unwrap();
        backend.ew_mul_forward_dev(&d_x, &d_b, &mut d_m, n).unwrap();
        assert_eq!(
            max_abs_diff(&download(&d_m), &elementwise::mul_forward(&x, &b)),
            0.0,
            "mul forward"
        );
        let mul_g = elementwise::mul_vjp(&x, &b, &gy); // [gA = gy⊙b, gB = gy⊙x]
        let mut d_ga = backend.dev_alloc_zeros(n).unwrap();
        backend
            .ew_mul_backward_dev(&d_gy, &d_b, &mut d_ga, n)
            .unwrap();
        assert_eq!(max_abs_diff(&download(&d_ga), &mul_g[0]), 0.0, "mul grad a");
        let mut d_gb = backend.dev_alloc_zeros(n).unwrap();
        backend
            .ew_mul_backward_dev(&d_gy, &d_x, &mut d_gb, n)
            .unwrap();
        assert_eq!(max_abs_diff(&download(&d_gb), &mul_g[1]), 0.0, "mul grad b");

        // add forward + grad accumulate — bit-exact.
        let mut d_add = backend.dev_alloc_zeros(n).unwrap();
        backend
            .ew_add_forward_dev(&d_x, &d_b, &mut d_add, n)
            .unwrap();
        assert_eq!(
            max_abs_diff(&download(&d_add), &elementwise::add_forward(&x, &b)),
            0.0,
            "add forward"
        );
        let mut d_acc = backend.dev_upload(&x).unwrap(); // dst = x; dst += b ⇒ x + b
        backend.accumulate_dev(&mut d_acc, &d_b, n).unwrap();
        assert_eq!(
            max_abs_diff(&download(&d_acc), &elementwise::add_forward(&x, &b)),
            0.0,
            "accumulate"
        );
        eprintln!("0043 P2.2 resident glue ops (silu/mul/add/accumulate, n={n}): match CPU ops");
    }

    /// ADR 0027 Track A: device SALT quantization preserves the existing
    /// per-row f32 AbsMean oracle for every supported plane count. The
    /// 257-column case guards against accidentally switching this first
    /// resident path to the deployed per-256-block format from Track D.
    /// Sherry on the device: `(1-a)*Q + a*master` must match the CPU oracle
    /// `ste::salt_quantize_forward_sherry` for every alpha, and `a = 0` must be **bit-identical** to the
    /// pure-ternary path (the guard in the kernel exists precisely so a==0 cannot perturb `-0.0`).
    #[test]
    fn resident_salt_quantize_sherry_matches_cpu() {
        use tritium_train::ops::ste;

        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "skipping resident_salt_quantize_sherry_matches_cpu: no CUDA device ({e})"
                );
                return;
            }
        };

        for &(rows, cols) in &[(5usize, 7usize), (3, 257), (128, 576)] {
            let mut master =
                seeded_uniform(0x5EA1 ^ rows as u64 ^ cols as u64, rows * cols, -2.0, 2.0);
            master[..cols].fill(0.0); // zero-scale row: the early-stop path must blend correctly too
            let d_master = backend.dev_upload(&master).unwrap();
            let mut d_residual = backend.dev_alloc_zeros(rows * cols).unwrap();
            let mut d_quantized = backend.dev_alloc_zeros(rows * cols).unwrap();

            for planes in 1..=3 {
                // alpha = 0 must reproduce the plain kernel BIT-for-bit.
                backend
                    .salt_quantize_forward_dev(
                        &d_master,
                        &mut d_residual,
                        &mut d_quantized,
                        rows,
                        cols,
                        planes,
                    )
                    .unwrap();
                let mut plain = vec![0.0f32; rows * cols];
                backend.dev_download(&d_quantized, &mut plain).unwrap();
                backend
                    .salt_quantize_forward_sherry_dev(
                        &d_master,
                        &mut d_residual,
                        &mut d_quantized,
                        rows,
                        cols,
                        planes,
                        0.0,
                    )
                    .unwrap();
                let mut at_zero = vec![0.0f32; rows * cols];
                backend.dev_download(&d_quantized, &mut at_zero).unwrap();
                assert_eq!(
                    at_zero, plain,
                    "alpha=0 must be bit-identical to the pure-ternary kernel (rows={rows} cols={cols} planes={planes})"
                );

                for alpha in [0.25f32, 0.5, 1.0] {
                    backend
                        .salt_quantize_forward_sherry_dev(
                            &d_master,
                            &mut d_residual,
                            &mut d_quantized,
                            rows,
                            cols,
                            planes,
                            alpha,
                        )
                        .unwrap();
                    let mut got = vec![0.0f32; rows * cols];
                    backend.dev_download(&d_quantized, &mut got).unwrap();
                    let want =
                        ste::salt_quantize_forward_sherry(&master, rows, cols, planes, alpha);
                    let delta = max_abs_diff(&got, &want);
                    assert!(
                        delta < 1e-4,
                        "sherry rows={rows} cols={cols} planes={planes} alpha={alpha}: max|delta|={delta:.3e}"
                    );
                    // alpha=1 is the fp master itself.
                    if alpha == 1.0 {
                        let d = max_abs_diff(&got, &master);
                        assert!(
                            d < 1e-4,
                            "alpha=1 must return the fp master: max|delta|={d:.3e}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn resident_salt_quantize_matches_cpu() {
        use tritium_train::ops::ste;

        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping resident_salt_quantize_matches_cpu: no CUDA device ({e})");
                return;
            }
        };

        for &(rows, cols) in &[(5usize, 7usize), (3, 257), (576, 576)] {
            let mut master =
                seeded_uniform(0x2700 ^ rows as u64 ^ cols as u64, rows * cols, -2.0, 2.0);
            master[..cols].fill(0.0); // zero-scale row and early-stop behavior
            let d_master = backend.dev_upload(&master).unwrap();
            let mut d_residual = backend.dev_alloc_zeros(rows * cols).unwrap();
            let mut d_quantized = backend.dev_alloc_zeros(rows * cols).unwrap();

            for planes in 1..=3 {
                backend
                    .salt_quantize_forward_dev(
                        &d_master,
                        &mut d_residual,
                        &mut d_quantized,
                        rows,
                        cols,
                        planes,
                    )
                    .unwrap();
                let mut got = vec![0.0f32; rows * cols];
                backend.dev_download(&d_quantized, &mut got).unwrap();
                let want = ste::salt_quantize_forward(&master, rows, cols, planes);
                let delta = max_abs_diff(&got, &want);
                assert!(
                    delta < 1e-4,
                    "rows={rows} cols={cols} planes={planes}: max|delta|={delta:.3e}"
                );
            }
        }

        let d_master = backend.dev_upload(&[1.0, -1.0]).unwrap();
        let mut d_residual = backend.dev_alloc_zeros(2).unwrap();
        let mut d_quantized = backend.dev_alloc_zeros(2).unwrap();
        assert!(matches!(
            backend.salt_quantize_forward_dev(
                &d_master,
                &mut d_residual,
                &mut d_quantized,
                1,
                2,
                0,
            ),
            Err(BackendError::InvalidInput(_))
        ));
        let mut undersized = backend.dev_alloc_zeros(1).unwrap();
        assert!(matches!(
            backend
                .salt_quantize_forward_dev(&d_master, &mut d_residual, &mut undersized, 1, 2, 1,),
            Err(BackendError::ShapeMismatch { .. })
        ));
    }

    /// ADR 0027 Track A: resident AdamW keeps master and moment buffers on
    /// device while preserving the CPU optimizer's operation order, bias
    /// correction, epsilon placement, and decoupled weight decay.
    #[test]
    fn resident_adamw_matches_cpu() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping resident_adamw_matches_cpu: no CUDA device ({e})");
                return;
            }
        };
        let opt = AdamW {
            lr: 0.1,
            beta1: 0.8,
            beta2: 0.9,
            eps: 0.5,
            weight_decay: 0.03,
        };
        let mut param = seeded_uniform(0xA27, 4099, -1.0, 1.0);
        let mut state = opt.init_state(param.len());
        let mut d_param = backend.dev_upload(&param).unwrap();
        let mut d_m = backend.dev_alloc_zeros(param.len()).unwrap();
        let mut d_v = backend.dev_alloc_zeros(param.len()).unwrap();

        for step in 1..=4 {
            let grad = seeded_uniform(0xB27 + step, param.len(), -0.25, 0.25);
            opt.step(step, &mut param, &grad, &mut state);
            let d_grad = backend.dev_upload(&grad).unwrap();
            backend
                .adamw_step_dev(&mut d_param, &d_grad, &mut d_m, &mut d_v, step, &opt)
                .unwrap();

            let mut got_param = vec![0.0; param.len()];
            let mut got_m = vec![0.0; param.len()];
            let mut got_v = vec![0.0; param.len()];
            backend.dev_download(&d_param, &mut got_param).unwrap();
            backend.dev_download(&d_m, &mut got_m).unwrap();
            backend.dev_download(&d_v, &mut got_v).unwrap();
            assert!(max_abs_diff(&got_param, &param) < 1e-5, "param step {step}");
            assert!(max_abs_diff(&got_m, &state.m) < 1e-5, "m step {step}");
            assert!(max_abs_diff(&got_v, &state.v) < 1e-5, "v step {step}");
        }

        let grad = backend.dev_alloc_zeros(param.len()).unwrap();
        assert!(matches!(
            backend.adamw_step_dev(&mut d_param, &grad, &mut d_m, &mut d_v, 0, &opt),
            Err(BackendError::InvalidInput(_))
        ));
        let short_grad = backend.dev_alloc_zeros(param.len() - 1).unwrap();
        assert!(matches!(
            backend.adamw_step_dev(&mut d_param, &short_grad, &mut d_m, &mut d_v, 1, &opt,),
            Err(BackendError::ShapeMismatch { .. })
        ));
    }

    fn max_rel_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y).abs() / x.abs().max(y.abs()).max(1.0))
            .fold(0.0f32, f32::max)
    }

    /// ADR 0027 Track D tape gate: one compact handle serves both tied embedding
    /// gather and LM-head matmul, while their two identity-STE contributions
    /// accumulate into one zero-storage latent-master leaf.
    #[test]
    fn packed_salt_tied_embed_head_matches_dense_oracle() {
        use tritium_train::ops::{dense, embed};

        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping packed_salt_tied_embed_head_matches_dense_oracle: {e}");
                return;
            }
        };
        let (seq, vocab) = (4usize, 5usize);
        let tokens_i32 = [4i32, 0, 4, 2];
        let tokens_u32 = [4u32, 0, 4, 2];

        for dim in [7usize, 257, 576] {
            let mut master = seeded_uniform(0xD000 + dim as u64, vocab * dim, -1.25, 1.25);
            master[..dim].fill(0.0);
            let seed = seeded_uniform(0xD100 + dim as u64, seq * vocab, -0.5, 0.5);
            for planes in 1..=3 {
                let packed =
                    DevicePackedSaltWeight::from_host(&backend, &master, vocab, dim, planes)
                        .unwrap();
                assert_eq!(packed.rows(), vocab);
                assert_eq!(packed.cols(), dim);
                assert_eq!(packed.planes(), planes);
                assert_eq!(
                    packed.packed_bytes(),
                    planes
                        * vocab
                        * dim.div_ceil(tritium_format::QK_K)
                        * (tritium_format::QK_K / 4)
                );
                assert_eq!(
                    packed.scale_bytes(),
                    planes * vocab * core::mem::size_of::<f32>()
                );
                if dim >= tritium_format::QK_K {
                    assert!(packed.resident_bytes() < master.len() * core::mem::size_of::<f32>());
                }

                let dense_w = ste::salt_quantize_forward(&master, vocab, dim, planes);
                let want_emb = embed::gather_forward(&dense_w, &tokens_u32, dim);
                let want_logits = dense::forward(&want_emb, &dense_w, seq, vocab, dim);
                let dense_grads = dense::vjp(&want_emb, &dense_w, seq, vocab, dim, &seed);
                let mut want_master = dense_grads[1].clone();
                let embed_grad = embed::gather_vjp(vocab, &tokens_u32, dim, &dense_grads[0]);
                for (dst, src) in want_master.iter_mut().zip(embed_grad) {
                    *dst += src;
                }

                let mut tape = DeviceTape::new(&backend, vocab).unwrap();
                let master_id = tape.gradient_leaf(vocab * dim).unwrap();
                assert!(matches!(
                    tape.vals[master_id],
                    Some(DeviceValue::GradientOnly)
                ));
                let emb = tape.salt_embed(master_id, &packed, &tokens_i32).unwrap();
                assert!(max_rel_diff(&tape.value(emb).unwrap(), &want_emb) < 1e-4);
                let logits = tape.salt_matmul(emb, master_id, &packed, seq).unwrap();
                assert!(max_rel_diff(&tape.value(logits).unwrap(), &want_logits) < 1e-4);
                let d_seed = backend.dev_upload(&seed).unwrap();
                let result = tape.backward_retain(logits, &d_seed, &[master_id]).unwrap();
                let mut got_master = vec![0.0; vocab * dim];
                backend
                    .dev_download(result.grads[master_id].as_ref().unwrap(), &mut got_master)
                    .unwrap();
                assert!(
                    max_rel_diff(&got_master, &want_master) < 1e-4,
                    "tied master gradient T={planes} dim={dim}"
                );
            }
        }
    }

    fn run_packed_checkpoint_graph(
        backend: &CudaBackend,
        packed: &DevicePackedSaltWeight,
        tokens: &[i32],
        policy: CheckpointPolicy,
        compute: PackedSaltComputePolicy,
    ) -> (Vec<f32>, Vec<f32>, DeviceBackwardStats) {
        let seq = tokens.len();
        let mut tape =
            DeviceTape::new_with_policies(backend, packed.rows(), policy, compute).unwrap();
        let master = tape.gradient_leaf(packed.rows() * packed.cols()).unwrap();
        let emb = tape.salt_embed(master, packed, tokens).unwrap();
        let hidden = tape.salt_matmul(emb, master, packed, seq).unwrap();
        tape.checkpoint_keep(&[hidden]).unwrap();
        let hidden = tape.silu(hidden).unwrap();
        let logits = tape.salt_matmul(hidden, master, packed, seq).unwrap();
        let output = tape.value(logits).unwrap();
        let seed = seeded_uniform(0xD271, output.len(), -0.25, 0.25);
        let d_seed = backend.dev_upload(&seed).unwrap();
        let result = tape.backward_retain(logits, &d_seed, &[master]).unwrap();
        let mut grad = vec![0.0; packed.rows() * packed.cols()];
        backend
            .dev_download(result.grads[master].as_ref().unwrap(), &mut grad)
            .unwrap();
        (output, grad, result.stats)
    }

    #[test]
    fn packed_salt_ops_replay_through_checkpoint() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping packed_salt_ops_replay_through_checkpoint: {e}");
                return;
            }
        };
        // Square so the same tied handle can feed both consecutive matmuls;
        // tail-block coverage lives in the dense-oracle gate above.
        let (vocab, dim) = (5usize, 5usize);
        let master = seeded_uniform(0xD270, vocab * dim, -1.0, 1.0);
        let packed = DevicePackedSaltWeight::from_host(&backend, &master, vocab, dim, 3).unwrap();
        let tokens = [4, 0, 4, 2];
        let keep = run_packed_checkpoint_graph(
            &backend,
            &packed,
            &tokens,
            CheckpointPolicy::KeepAll,
            PackedSaltComputePolicy::Exact,
        );
        let replay = run_packed_checkpoint_graph(
            &backend,
            &packed,
            &tokens,
            CheckpointPolicy::EveryBlocks(1),
            PackedSaltComputePolicy::Exact,
        );
        assert!(max_rel_diff(&keep.0, &replay.0) < 1e-6);
        assert!(max_rel_diff(&keep.1, &replay.1) < 1e-6);
        assert!(
            replay.2.recomputed_ops >= 2,
            "packed ops must replay: {:?}",
            replay.2
        );
    }

    #[test]
    fn packed_fast_policy_covers_forward_replay_activation_vjp_and_tied_master_vjp() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "skipping packed_fast_policy_covers_forward_replay_activation_vjp_and_tied_master_vjp: {e}"
                );
                return;
            }
        };
        let default_tape = DeviceTape::new(&backend, 32).unwrap();
        assert_eq!(
            default_tape.packed_compute_policy,
            PackedSaltComputePolicy::Exact
        );
        drop(default_tape);

        let width = 32usize;
        let master = seeded_uniform(0xD272, width * width, -1.0, 1.0);
        let packed = DevicePackedSaltWeight::from_host(&backend, &master, width, width, 3).unwrap();
        let tokens = [31, 0, 17, 31];
        let exact = run_packed_checkpoint_graph(
            &backend,
            &packed,
            &tokens,
            CheckpointPolicy::KeepAll,
            PackedSaltComputePolicy::Exact,
        );
        let fast = run_packed_checkpoint_graph(
            &backend,
            &packed,
            &tokens,
            CheckpointPolicy::KeepAll,
            PackedSaltComputePolicy::Fast,
        );
        let fast_replay = run_packed_checkpoint_graph(
            &backend,
            &packed,
            &tokens,
            CheckpointPolicy::EveryBlocks(1),
            PackedSaltComputePolicy::Fast,
        );
        assert!(max_rel_diff(&fast.0, &exact.0) < 1e-4);
        assert!(max_rel_diff(&fast.1, &exact.1) < 1e-4);
        assert!(max_rel_diff(&fast_replay.0, &fast.0) < 1e-6);
        assert!(max_rel_diff(&fast_replay.1, &fast.1) < 1e-6);
        assert!(fast_replay.2.recomputed_ops >= 2);
    }

    #[test]
    fn packed_salt_tape_validation_fails_closed() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping packed_salt_tape_validation_fails_closed: {e}");
                return;
            }
        };
        let mut packed = DevicePackedSaltWeight::from_host(&backend, &[0.5; 35], 5, 7, 2).unwrap();
        assert!(matches!(
            packed.repack_from_host(&backend, &[0.0; 34]),
            Err(BackendError::ShapeMismatch { .. })
        ));
        assert!(!packed.is_prepared());
        assert!(packed.repack_from_host(&backend, &[0.25; 35]).is_ok());

        let mut short_ones = DeviceTape::new(&backend, 1).unwrap();
        let short_ones_master = short_ones.gradient_leaf(35).unwrap();
        let short_ones_data = short_ones.leaf(&[1.0; 7]).unwrap();
        assert!(matches!(
            short_ones.salt_matmul(short_ones_data, short_ones_master, &packed, 1),
            Err(BackendError::ShapeMismatch {
                expected: 5,
                got: 1
            })
        ));

        let mut tape = DeviceTape::new(&backend, 5).unwrap();
        let data = tape.leaf(&[1.0; 7]).unwrap();
        let wrong_master = tape.leaf(&[0.0; 35]).unwrap();
        assert!(matches!(
            tape.salt_matmul(data, wrong_master, &packed, 1),
            Err(BackendError::InvalidInput(_))
        ));
        let short_master = tape.gradient_leaf(34).unwrap();
        assert!(matches!(
            tape.salt_matmul(data, short_master, &packed, 1),
            Err(BackendError::ShapeMismatch { .. })
        ));
        let master = tape.gradient_leaf(35).unwrap();
        assert!(matches!(
            tape.value(master),
            Err(BackendError::InvalidInput(message)) if message.contains("gradient-only")
        ));
        assert!(matches!(
            tape.salt_embed(master, &packed, &[0, -1]),
            Err(BackendError::InvalidInput(_))
        ));
        packed.mark_stale();
        let mut tape = DeviceTape::new(&backend, 5).unwrap();
        let master = tape.gradient_leaf(35).unwrap();
        let data = tape.leaf(&[1.0; 7]).unwrap();
        assert!(matches!(
            tape.salt_matmul(data, master, &packed, 1),
            Err(BackendError::InvalidInput(message)) if message.contains("stale")
        ));
        drop(tape);

        // A second physical device, when present, must reject this device-0
        // packed handle before launch. Single-GPU developer machines skip only
        // this conditional branch; all other validation above remains active.
        if let Ok(other) = CudaBackend::new(1) {
            packed.repack_from_host(&backend, &[0.25; 35]).unwrap();
            let mut tape = DeviceTape::new(&other, 5).unwrap();
            let master = tape.gradient_leaf(35).unwrap();
            let data = tape.leaf(&[1.0; 7]).unwrap();
            assert!(matches!(
                tape.salt_matmul(data, master, &packed, 1),
                Err(BackendError::InvalidInput(message)) if message.contains("context")
            ));
        }
    }

    /// ADR 0027 Track A exit seam: masters, quantized weights, optimizer
    /// moments, targets, and returned gradients remain resident for a complete
    /// training step.  The final master must match the CPU SALT + AdamW oracle.
    #[test]
    fn device_trainer_step_matches_host_reference() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping device_trainer_step_matches_host_reference: no CUDA ({e})");
                return;
            }
        };
        let (batch, rows, cols, planes) = (3usize, 5usize, 7usize, 2usize);
        let input = seeded_uniform(0xA270, batch * cols, -1.0, 1.0);
        let target = vec![1.0 / rows as f32; batch * rows];
        let master = seeded_uniform(0xA271, rows * cols, -1.5, 1.5);
        let opt = AdamW {
            lr: 0.03,
            beta1: 0.8,
            beta2: 0.9,
            eps: 0.2,
            weight_decay: 0.01,
        };

        // Independent host oracle: SALT forward, dense matmul/xent VJP, AdamW.
        let quantized = ste::salt_quantize_forward(&master, rows, cols, planes);
        let logits = matmul::forward(&input, &quantized, &vec![1.0; rows], batch, rows, cols);
        let g_logits = loss::softmax_xent_vjp(&logits, &target, batch, rows, &[1.0]).remove(0);
        let grad = matmul::vjp(
            &input,
            &quantized,
            &vec![1.0; rows],
            batch,
            rows,
            cols,
            &g_logits,
        )
        .remove(1);
        let mut expected = master.clone();
        let mut expected_state = opt.init_state(expected.len());
        opt.step(1, &mut expected, &grad, &mut expected_state);

        let mut trainer = DeviceTrainer::new(
            &backend,
            &[DeviceTrainParam {
                master: &master,
                rows,
                cols,
                salt_planes: planes,
                optimizer: opt,
            }],
        )
        .unwrap();
        assert!(matches!(
            trainer.quantized(0),
            Err(BackendError::InvalidInput(message)) if message.contains("prepare_quantized")
        ));
        trainer.prepare_quantized().unwrap();
        let d_input = DeviceTensor::upload(&backend, &input).unwrap();
        let d_target = DeviceTensor::upload(&backend, &target).unwrap();
        let grads = {
            let mut tape = DeviceTape::new(&backend, rows).unwrap();
            let x = tape.leaf_device(&d_input).unwrap();
            let w = tape.leaf_device(trainer.quantized(0).unwrap()).unwrap();
            let logits = tape.matmul(x, w, batch, rows, cols).unwrap();
            tape.xent_backward_device(logits, &d_target, batch, rows, &[w])
                .unwrap()
        };
        let liveness = grads.backward_stats();
        assert!(
            liveness.peak_persistent_grad_elements < liveness.naive_all_value_grad_elements,
            "resident xent exposes lazy-slot diagnostics: {liveness:?}"
        );
        trainer.step(grads, 1).unwrap();
        assert!(matches!(
            trainer.quantized(0),
            Err(BackendError::InvalidInput(message)) if message.contains("stale")
        ));

        let got = trainer.download_master(0).unwrap();
        assert_eq!(d_input.download(&backend).unwrap(), input);
        assert_eq!(d_target.download(&backend).unwrap(), target);
        assert!(
            max_abs_diff(&got, &expected) < 1e-5,
            "resident trainer diverged from host oracle"
        );
    }

    /// Lever 5: a `DeviceTrainer` built with reduced-precision optimizer state — int8 moments and/or a
    /// bf16-grid master — trains the toy SALT-distill problem end-to-end and drives the teacher-forced
    /// xent loss down comparably to the f32 trainer on the same seed. Functional gate for the wired
    /// int8 + bf16-master paths; the kernel↔oracle numerics are gated separately
    /// (`adamw_step_8bit_matches_cpu_oracle`, `adamw_step_bf16_master_matches_host`).
    #[test]
    fn device_trainer_reduced_precision_trains_like_f32() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping device_trainer_int8_moments_train_like_f32: no CUDA ({e})");
                return;
            }
        };
        let (batch, rows, cols, planes) = (4usize, 6usize, 8usize, 2usize);
        let input = seeded_uniform(0x1817, batch * cols, -1.0, 1.0);
        // A fixed soft target the student is distilled toward (teacher-forced).
        let target = {
            let raw = seeded_uniform(0x1818, batch * rows, 0.0, 1.0);
            let mut t = vec![0.0f32; batch * rows];
            for b in 0..batch {
                let s: f32 = raw[b * rows..(b + 1) * rows].iter().sum::<f32>().max(1e-6);
                for r in 0..rows {
                    t[b * rows + r] = raw[b * rows + r] / s;
                }
            }
            t
        };
        let master = seeded_uniform(0x1819, rows * cols, -1.5, 1.5);
        let d_input = DeviceTensor::upload(&backend, &input).unwrap();
        let d_target = DeviceTensor::upload(&backend, &target).unwrap();

        // Run the toy distill for `steps` steps at the given moment precision, returning the loss
        // trajectory (device xent value each step, pre-update).
        let run = |moment: MomentPrecision, master_prec: MasterPrecision| -> Vec<f32> {
            let mut trainer = DeviceTrainer::new_with_options(
                &backend,
                &[DeviceTrainParam {
                    master: &master,
                    rows,
                    cols,
                    salt_planes: planes,
                    optimizer: AdamW::new(0.05),
                }],
                DeviceTrainerWeightStorage::DenseQuantized,
                moment,
                master_prec,
            )
            .unwrap();
            let mut losses = Vec::new();
            for step in 1..=30u64 {
                trainer.prepare_quantized().unwrap();
                let (loss, grads) = {
                    let mut tape = DeviceTape::new(&backend, rows).unwrap();
                    let x = tape.leaf_device(&d_input).unwrap();
                    let w = tape.leaf_device(trainer.quantized(0).unwrap()).unwrap();
                    let logits = tape.matmul(x, w, batch, rows, cols).unwrap();
                    // Teacher-forced cross-entropy H(target, softmax(logits)), averaged over the batch.
                    let logits_h = tape.value(logits).unwrap();
                    let mut loss = 0.0f32;
                    for b in 0..batch {
                        let row = &logits_h[b * rows..(b + 1) * rows];
                        let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let z: f32 = row.iter().map(|&v| (v - mx).exp()).sum();
                        let logz = mx + z.ln();
                        for r in 0..rows {
                            loss -= target[b * rows + r] * (row[r] - logz);
                        }
                    }
                    let loss = loss / batch as f32;
                    let d_tgt = DeviceTensor::upload(&backend, &target).unwrap();
                    let grads = tape
                        .xent_backward_device(logits, &d_tgt, batch, rows, &[w])
                        .unwrap();
                    (loss, grads)
                };
                trainer.step(grads, step).unwrap();
                losses.push(loss);
            }
            let _ = &d_target;
            losses
        };

        let f32_losses = run(MomentPrecision::F32, MasterPrecision::F32);
        let int8_losses = run(MomentPrecision::Int8, MasterPrecision::F32);
        let bf16_losses = run(MomentPrecision::F32, MasterPrecision::Bf16);
        let (f0, f1) = (f32_losses[0], *f32_losses.last().unwrap());
        let (i0, i1) = (int8_losses[0], *int8_losses.last().unwrap());
        let (b0, b1) = (bf16_losses[0], *bf16_losses.last().unwrap());
        eprintln!(
            "reduced-precision DeviceTrainer distill: f32 {f0:.4}→{f1:.4}, int8 {i0:.4}→{i1:.4}, \
             bf16-master {b0:.4}→{b1:.4}"
        );
        assert_eq!(i0, f0, "same seed ⇒ identical starting loss (int8)");
        assert!(
            f1 < 0.95 * f0,
            "f32 trainer must descend: {f0:.4} → {f1:.4}"
        );
        assert!(
            i1 < 0.95 * i0,
            "int8 trainer must descend: {i0:.4} → {i1:.4}"
        );
        assert!(
            b1 < 0.95 * b0,
            "bf16-master trainer must descend: {b0:.4} → {b1:.4}"
        );
        // The point of Lever 5: reduced-precision optimizer state trains *like* f32 — final losses stay
        // close relative to how far f32 travelled (within ~20% of f32's descent, either side).
        let descent = (f0 - f1).abs().max(1e-3);
        assert!(
            (i1 - f1).abs() < 0.2 * descent,
            "int8 final {i1:.4} must track f32 {f1:.4} (descent {:.4} from {f0:.4})",
            f0 - f1
        );
        assert!(
            (b1 - f1).abs() < 0.2 * descent,
            "bf16-master final {b1:.4} must track f32 {f1:.4} (descent {:.4} from {f0:.4})",
            f0 - f1
        );
    }

    #[test]
    fn device_trainer_steps_are_contiguous_and_report_completed_step() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "skipping device_trainer_steps_are_contiguous_and_report_completed_step: \
                     no CUDA ({e})"
                );
                return;
            }
        };
        let master = vec![0.25f32; 8];
        let mut trainer = DeviceTrainer::new(
            &backend,
            &[DeviceTrainParam {
                master: &master,
                rows: 2,
                cols: 4,
                salt_planes: 1,
                optimizer: AdamW::new(1e-2),
            }],
        )
        .unwrap();
        let gradients = || upload_test_gradients(&backend, &[vec![0.1; 8]]);

        assert_eq!(trainer.completed_step(), 0);
        assert!(matches!(
            trainer.step(gradients(), 0),
            Err(BackendError::InvalidInput(message)) if message.contains("expected step 1")
        ));
        assert!(matches!(
            trainer.step(gradients(), 2),
            Err(BackendError::InvalidInput(message)) if message.contains("expected step 1")
        ));
        assert_eq!(trainer.download_master(0).unwrap(), master);

        trainer.step(gradients(), 1).unwrap();
        assert_eq!(trainer.completed_step(), 1);
        assert!(matches!(
            trainer.step(gradients(), 1),
            Err(BackendError::InvalidInput(message)) if message.contains("expected step 2")
        ));
        assert_eq!(trainer.completed_step(), 1);
    }

    #[test]
    fn device_trainer_packs_and_repacks_directly_from_resident_masters() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "skipping device_trainer_packs_and_repacks_directly_from_resident_masters: \
                     no CUDA ({e})"
                );
                return;
            }
        };
        let (rows, cols, planes) = (3usize, 7usize, 2usize);
        let master = seeded_uniform(0xD2A0, rows * cols, -1.0, 1.0);
        let mut trainer = DeviceTrainer::new_with_weight_storage(
            &backend,
            &[DeviceTrainParam {
                master: &master,
                rows,
                cols,
                salt_planes: planes,
                optimizer: AdamW::new(1e-2),
            }],
            DeviceTrainerWeightStorage::Packed,
        )
        .unwrap();
        assert_eq!(
            trainer.resident_stats(),
            ResidentTrainerStats {
                parameter_elements: 21,
                quantized_elements: 0,
                residual_elements: 21,
                optimizer_elements: 42,
                resident_elements: 84,
                largest_parameter_elements: 21,
            }
        );
        assert!(matches!(
            trainer.prepare_quantized(),
            Err(BackendError::InvalidInput(message))
                if message.contains("dense quantized storage is disabled")
        ));
        let mut packed = trainer.packed_weight(0).unwrap();
        let direct_packed = DevicePackedSaltWeight::from_device_master(
            &backend,
            trainer.master_tensor(0).unwrap(),
            rows,
            cols,
            planes,
        )
        .unwrap();
        let input = seeded_uniform(0xD2A1, 2 * cols, -0.5, 0.5);
        let packed_output = |packed: &DevicePackedSaltWeight| {
            let mut tape = DeviceTape::new(&backend, rows).unwrap();
            let x = tape.leaf(&input).unwrap();
            let master_leaf = tape.gradient_leaf(rows * cols).unwrap();
            let output = tape.salt_matmul(x, master_leaf, packed, 2).unwrap();
            tape.value(output).unwrap()
        };

        let host_packed =
            DevicePackedSaltWeight::from_host(&backend, &master, rows, cols, planes).unwrap();
        assert_eq!(packed_output(&packed), packed_output(&host_packed));

        let gradients = vec![seeded_uniform(0xD2A2, rows * cols, -0.2, 0.2)];
        trainer
            .step(upload_test_gradients(&backend, &gradients), 1)
            .unwrap();
        assert!(!packed.is_prepared());
        assert!(!direct_packed.is_prepared());
        assert!(matches!(
            packed.validate_current_master(trainer.master_tensor(0).unwrap()),
            Err(BackendError::InvalidInput(message)) if message.contains("predates")
        ));
        trainer.repack_packed_weight(0, &mut packed).unwrap();
        let updated_master = trainer.download_master(0).unwrap();
        let updated_host =
            DevicePackedSaltWeight::from_host(&backend, &updated_master, rows, cols, planes)
                .unwrap();
        assert_eq!(packed_output(&packed), packed_output(&updated_host));
        assert!(packed.is_prepared());
    }

    #[test]
    fn device_trainer_dcp_roundtrip_resumes_with_update_parity() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "skipping device_trainer_dcp_roundtrip_resumes_with_update_parity: \
                     no CUDA ({e})"
                );
                return;
            }
        };
        let masters = [
            seeded_uniform(0xD2B0, 3, -1.0, 1.0),
            seeded_uniform(0xD2B1, 17, -1.0, 1.0),
            seeded_uniform(0xD2B2, 5, -1.0, 1.0),
        ];
        let specs = [
            DeviceTrainParam {
                master: &masters[0],
                rows: 1,
                cols: 3,
                salt_planes: 1,
                optimizer: AdamW::new(0.03),
            },
            DeviceTrainParam {
                master: &masters[1],
                rows: 1,
                cols: 17,
                salt_planes: 2,
                optimizer: AdamW::new(0.02),
            },
            DeviceTrainParam {
                master: &masters[2],
                rows: 1,
                cols: 5,
                salt_planes: 3,
                optimizer: AdamW::new(0.01),
            },
        ];
        let gradients_1 = [
            seeded_uniform(0xD2B3, 3, -0.2, 0.2),
            seeded_uniform(0xD2B4, 17, -0.2, 0.2),
            seeded_uniform(0xD2B5, 5, -0.2, 0.2),
        ];
        let gradients_2 = [
            seeded_uniform(0xD2B6, 3, -0.2, 0.2),
            seeded_uniform(0xD2B7, 17, -0.2, 0.2),
            seeded_uniform(0xD2B8, 5, -0.2, 0.2),
        ];
        let mut uninterrupted = DeviceTrainer::new(&backend, &specs).unwrap();
        let mut source = DeviceTrainer::new(&backend, &specs).unwrap();
        uninterrupted
            .step(upload_test_gradients(&backend, &gradients_1), 1)
            .unwrap();
        uninterrupted
            .step(upload_test_gradients(&backend, &gradients_2), 2)
            .unwrap();
        source
            .step(upload_test_gradients(&backend, &gradients_1), 1)
            .unwrap();

        let dir = host_offload_dcp_dir("resident-roundtrip");
        tritium_train::dcp::save_from(&dir, &mut source, 2).unwrap();
        let zeros = [vec![0.0; 3], vec![0.0; 17], vec![0.0; 5]];
        let restore_specs = [
            DeviceTrainParam {
                master: &zeros[0],
                ..specs[0]
            },
            DeviceTrainParam {
                master: &zeros[1],
                ..specs[1]
            },
            DeviceTrainParam {
                master: &zeros[2],
                ..specs[2]
            },
        ];
        let mut restored = DeviceTrainer::new(&backend, &restore_specs).unwrap();
        tritium_train::dcp::load_into(&dir, &mut restored).unwrap();
        assert_eq!(restored.completed_step(), 1);
        assert_eq!(restored.len(), 3);
        assert!(!restored.is_empty());
        assert_eq!(restored.leaf_lens(), &[3, 17, 5]);
        assert_eq!(
            restored.parameter_metadata(1).unwrap(),
            HostOffloadParamMetadata {
                rows: 1,
                cols: 17,
                salt_planes: 2,
            }
        );
        assert_eq!(
            restored.resident_stats(),
            ResidentTrainerStats {
                parameter_elements: 25,
                quantized_elements: 25,
                residual_elements: 17,
                optimizer_elements: 50,
                resident_elements: 117,
                largest_parameter_elements: 17,
            }
        );
        restored
            .step(upload_test_gradients(&backend, &gradients_2), 2)
            .unwrap();
        for index in 0..masters.len() {
            assert_eq!(
                restored.download_master(index).unwrap(),
                uninterrupted.download_master(index).unwrap(),
                "restored resident state diverged at leaf {index}"
            );
        }

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn device_trainer_dcp_chunks_cross_leaves_and_failed_loads_stay_poisoned() {
        use tritium_train::dcp::{StateSink, StateSource};

        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "skipping device_trainer_dcp_chunks_cross_leaves_and_failed_loads_stay_poisoned: \
                     no CUDA ({e})"
                );
                return;
            }
        };
        let masters = [
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0],
            vec![6.0, 7.0, 8.0, 9.0],
        ];
        let specs = [
            DeviceTrainParam {
                master: &masters[0],
                rows: 1,
                cols: 3,
                salt_planes: 1,
                optimizer: AdamW::new(0.01),
            },
            DeviceTrainParam {
                master: &masters[1],
                rows: 1,
                cols: 2,
                salt_planes: 1,
                optimizer: AdamW::new(0.01),
            },
            DeviceTrainParam {
                master: &masters[2],
                rows: 1,
                cols: 4,
                salt_planes: 1,
                optimizer: AdamW::new(0.01),
            },
        ];
        let mut trainer = DeviceTrainer::new(&backend, &specs).unwrap();
        let mut crossing = [0.0; 6];
        StateSource::read_chunk(&mut trainer, StatePlane::Parameter, 2, &mut crossing).unwrap();
        assert_eq!(crossing, [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert!(matches!(
            StateSource::read_chunk(
                &mut trainer,
                StatePlane::Optimizer(2),
                0,
                &mut crossing[..1],
            ),
            Err(DcpError::InvalidState(_))
        ));

        assert!(matches!(
            StateSink::begin(&mut trainer, 4, &[3, 3, 3], 2),
            Err(DcpError::InvalidState(_))
        ));
        assert!(trainer.is_poisoned());
        assert!(trainer.download_master(0).is_err());

        StateSink::begin(&mut trainer, 5, &[3, 2, 4], 2).unwrap();
        StateSink::write_chunk(
            &mut trainer,
            StatePlane::Parameter,
            0,
            &[11.0, 12.0, 13.0, 14.0],
        )
        .unwrap();
        assert!(matches!(
            StateSink::finish(&mut trainer),
            Err(DcpError::InvalidState(_))
        ));
        assert!(trainer.is_poisoned());

        StateSink::begin(&mut trainer, 6, &[3, 2, 4], 2).unwrap();
        assert!(matches!(
            StateSink::write_chunk(&mut trainer, StatePlane::Parameter, 1, &[1.0]),
            Err(DcpError::InvalidState(_))
        ));
        assert!(trainer.is_poisoned());

        let parameters = [21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0];
        let first_moment = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let second_moment = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9];
        StateSink::begin(&mut trainer, 7, &[3, 2, 4], 2).unwrap();
        StateSink::write_chunk(&mut trainer, StatePlane::Parameter, 0, &parameters[..5]).unwrap();
        StateSink::write_chunk(&mut trainer, StatePlane::Parameter, 5, &parameters[5..]).unwrap();
        StateSink::write_chunk(&mut trainer, StatePlane::Optimizer(0), 0, &first_moment).unwrap();
        StateSink::write_chunk(&mut trainer, StatePlane::Optimizer(1), 0, &second_moment).unwrap();
        StateSink::finish(&mut trainer).unwrap();
        assert!(!trainer.is_poisoned());
        assert_eq!(trainer.completed_step(), 7);
        assert_eq!(trainer.download_master(0).unwrap(), &parameters[..3]);
        assert_eq!(trainer.download_master(1).unwrap(), &parameters[3..5]);
        assert_eq!(trainer.download_master(2).unwrap(), &parameters[5..]);
        trainer.prepare_quantized().unwrap();
    }

    #[test]
    fn host_offload_rejects_zero_step_before_mutation() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping host_offload_rejects_zero_step_before_mutation: no CUDA ({e})");
                return;
            }
        };
        let master = seeded_uniform(0xE270, 35, -1.0, 1.0);
        let spec = DeviceTrainParam {
            master: &master,
            rows: 5,
            cols: 7,
            salt_planes: 1,
            optimizer: AdamW::new(1e-3),
        };
        let mut trainer = HostOffloadTrainer::new(&backend, &[spec]).unwrap();
        let before = trainer.master(0).unwrap().to_vec();
        let grads = DeviceGradients {
            bufs: vec![backend.dev_alloc_zeros(master.len()).unwrap()],
            stats: DeviceBackwardStats::default(),
        };

        assert!(matches!(
            trainer.step(grads, 0),
            Err(BackendError::InvalidInput(message)) if message.contains("1-based")
        ));
        assert_eq!(trainer.master(0).unwrap(), before);
        assert!(!trainer.is_poisoned());
    }

    #[test]
    fn host_offload_accepts_owned_masters_without_a_clone_at_the_api_seam() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "skipping host_offload_accepts_owned_masters_without_a_clone_at_the_api_seam: no CUDA ({e})"
                );
                return;
            }
        };
        let master = seeded_uniform(0xE271, 35, -1.0, 1.0);
        let expected = master.clone();

        let trainer = HostOffloadTrainer::new_owned(
            &backend,
            vec![HostOffloadTrainParam {
                master,
                rows: 5,
                cols: 7,
                salt_planes: 2,
                optimizer: AdamW::new(1e-3),
            }],
        )
        .unwrap();

        assert_eq!(trainer.master(0).unwrap(), expected);
        assert_eq!(trainer.stats().host_optimizer_elements, 3 * 35);
    }

    #[test]
    fn host_offload_zero_sized_leaf_is_a_noop() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping host_offload_zero_sized_leaf_is_a_noop: no CUDA ({e})");
                return;
            }
        };
        let master = Vec::new();
        let spec = DeviceTrainParam {
            master: &master,
            rows: 0,
            cols: 7,
            salt_planes: 1,
            optimizer: AdamW::new(1e-3),
        };
        let mut trainer = HostOffloadTrainer::new(&backend, &[spec]).unwrap();
        let grads = DeviceGradients {
            bufs: vec![backend.dev_alloc_zeros(0).unwrap()],
            stats: DeviceBackwardStats::default(),
        };

        trainer.step(grads, 1).unwrap();

        assert_eq!(trainer.completed_step(), 1);
        assert!(trainer.master(0).unwrap().is_empty());
        assert!(!trainer.is_poisoned());
        assert_eq!(trainer.stats().peak_optimizer_device_elements, 0);
        assert_eq!(trainer.stats().pinned_optimizer_host_elements, 0);
        assert_eq!(trainer.stats().peak_in_flight_parameters, 0);
    }

    #[test]
    fn host_offload_poison_refuses_reuse_after_device_update_failure() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "skipping host_offload_poison_refuses_reuse_after_device_update_failure: \
                     no CUDA ({e})"
                );
                return;
            }
        };
        let master = seeded_uniform(0xE275, 35, -1.0, 1.0);
        let spec = DeviceTrainParam {
            master: &master,
            rows: 5,
            cols: 7,
            salt_planes: 1,
            optimizer: AdamW::new(1e-3),
        };
        let mut trainer = HostOffloadTrainer::new(&backend, &[spec]).unwrap();
        let grad = backend.dev_alloc_zeros(master.len()).unwrap();
        assert!(trainer.apply_streamed_gradient(0, &grad, 0).is_err());
        assert!(trainer.is_poisoned());
        let grads = DeviceGradients {
            bufs: vec![backend.dev_alloc_zeros(master.len()).unwrap()],
            stats: DeviceBackwardStats::default(),
        };
        assert!(matches!(
            trainer.step(grads, 1),
            Err(BackendError::InvalidInput(message)) if message.contains("poisoned")
        ));
    }

    fn upload_test_gradients(backend: &CudaBackend, grads: &[Vec<f32>]) -> DeviceGradients {
        DeviceGradients {
            bufs: grads
                .iter()
                .map(|grad| backend.dev_upload(grad).unwrap())
                .collect(),
            stats: DeviceBackwardStats::default(),
        }
    }

    #[test]
    fn device_trainer_poisoned_after_partial_optimizer_failure() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "skipping device_trainer_poisoned_after_partial_optimizer_failure: no CUDA ({e})"
                );
                return;
            }
        };
        let masters = [vec![0.5f32; 8], vec![-0.25f32; 8]];
        let specs = [
            DeviceTrainParam {
                master: &masters[0],
                rows: 2,
                cols: 4,
                salt_planes: 1,
                optimizer: AdamW::new(1e-2),
            },
            DeviceTrainParam {
                master: &masters[1],
                rows: 2,
                cols: 4,
                salt_planes: 1,
                optimizer: AdamW::new(1e-2),
            },
        ];
        let mut trainer = DeviceTrainer::new(&backend, &specs).unwrap();
        let before = trainer.params[0].master.download(&backend).unwrap();
        // Corrupt only the second private moment buffer after construction. All
        // public gradient metadata still prevalidates, so leaf 0 is launched
        // before leaf 1 reports the injected optimizer-state shape failure.
        trainer.params[1].m = backend.dev_alloc_zeros(0).unwrap();
        let grads = vec![vec![0.25f32; 8], vec![0.25f32; 8]];

        assert!(matches!(
            trainer.step(upload_test_gradients(&backend, &grads), 1),
            Err(BackendError::ShapeMismatch { .. })
        ));
        assert!(trainer.is_poisoned());
        let after = trainer.params[0].master.download(&backend).unwrap();
        assert!(max_abs_diff(&before, &after) > 0.0);
        assert!(matches!(
            trainer.prepare_quantized(),
            Err(BackendError::InvalidInput(message)) if message.contains("poisoned")
        ));
        assert!(matches!(
            trainer.quantized(0),
            Err(BackendError::InvalidInput(message)) if message.contains("poisoned")
        ));
        assert!(matches!(
            trainer.download_master(0),
            Err(BackendError::InvalidInput(message)) if message.contains("poisoned")
        ));
        assert!(matches!(
            trainer.step(upload_test_gradients(&backend, &grads), 2),
            Err(BackendError::InvalidInput(message)) if message.contains("poisoned")
        ));
    }

    #[test]
    fn host_offload_matches_fully_resident_adamw_over_multiple_steps() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "skipping host_offload_matches_fully_resident_adamw_over_multiple_steps: \
                     no CUDA ({e})"
                );
                return;
            }
        };
        let masters = [
            seeded_uniform(0xE271, 17, -1.0, 1.0),
            seeded_uniform(0xE272, 257, -1.0, 1.0),
            seeded_uniform(0xE273, 63, -1.0, 1.0),
        ];
        let optimizers = [
            AdamW {
                lr: 0.03,
                beta1: 0.8,
                beta2: 0.9,
                eps: 0.2,
                weight_decay: 0.01,
            },
            AdamW {
                lr: 0.01,
                beta1: 0.9,
                beta2: 0.95,
                eps: 0.1,
                weight_decay: 0.03,
            },
            AdamW::new(0.02),
        ];
        let specs = [
            DeviceTrainParam {
                master: &masters[0],
                rows: 1,
                cols: 17,
                salt_planes: 1,
                optimizer: optimizers[0],
            },
            DeviceTrainParam {
                master: &masters[1],
                rows: 1,
                cols: 257,
                salt_planes: 2,
                optimizer: optimizers[1],
            },
            DeviceTrainParam {
                master: &masters[2],
                rows: 7,
                cols: 9,
                salt_planes: 3,
                optimizer: optimizers[2],
            },
        ];
        let mut resident = DeviceTrainer::new(&backend, &specs).unwrap();
        let mut offload = HostOffloadTrainer::new(&backend, &specs).unwrap();
        for (index, spec) in specs.iter().enumerate() {
            assert_eq!(
                offload.parameter_metadata(index).unwrap(),
                HostOffloadParamMetadata {
                    rows: spec.rows,
                    cols: spec.cols,
                    salt_planes: spec.salt_planes,
                }
            );
        }

        for step in 1..=4u64 {
            let grads: Vec<Vec<f32>> = masters
                .iter()
                .enumerate()
                .map(|(index, master)| {
                    seeded_uniform(0xE300 + step * 17 + index as u64, master.len(), -0.25, 0.25)
                })
                .collect();
            resident
                .step(upload_test_gradients(&backend, &grads), step)
                .unwrap();
            offload
                .step(upload_test_gradients(&backend, &grads), step)
                .unwrap();

            for (index, master) in masters.iter().enumerate() {
                let resident_master = resident.download_master(index).unwrap();
                assert!(
                    max_abs_diff(&resident_master, offload.master(index).unwrap()) < 1e-5,
                    "master {index} diverged at step {step}"
                );
                let resident_param = &resident.params[index];
                let mut resident_m = vec![0.0; master.len()];
                let mut resident_v = vec![0.0; master.len()];
                backend
                    .dev_download(&resident_param.m, &mut resident_m)
                    .unwrap();
                backend
                    .dev_download(&resident_param.v, &mut resident_v)
                    .unwrap();
                let (offload_m, offload_v) = offload.moments(index).unwrap();
                assert!(
                    max_abs_diff(&resident_m, offload_m) < 1e-5,
                    "first moment {index} diverged at step {step}"
                );
                assert!(
                    max_abs_diff(&resident_v, offload_v) < 1e-5,
                    "second moment {index} diverged at step {step}"
                );
            }
        }

        let stats = offload.stats();
        let total_elements: usize = masters.iter().map(Vec::len).sum();
        let largest = masters.iter().map(Vec::len).max().unwrap();
        assert_eq!(stats.host_optimizer_elements, total_elements * 3);
        assert_eq!(stats.largest_parameter_elements, largest);
        assert_eq!(stats.peak_optimizer_device_elements, largest * 6);
        assert_eq!(stats.pinned_optimizer_host_elements, largest * 6);
        assert_eq!(stats.peak_in_flight_parameters, 2);
        assert_eq!(stats.resident_input_gradient_elements, total_elements);
        eprintln!(
            "0027 Track E host AdamW offload: device/pinned staging {}/{} elements; host state {}; \
             largest leaf {}; peak in-flight {}; resident gradient input {} elements",
            stats.peak_optimizer_device_elements,
            stats.pinned_optimizer_host_elements,
            stats.host_optimizer_elements,
            stats.largest_parameter_elements,
            stats.peak_in_flight_parameters,
            stats.resident_input_gradient_elements
        );
    }

    #[test]
    fn host_offload_collected_steps_must_be_contiguous_before_mutation() {
        let backend = match CudaBackend::new(0) {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!(
                    "skipping host_offload_collected_steps_must_be_contiguous_before_mutation: \
                     no CUDA ({error})"
                );
                return;
            }
        };
        let master = seeded_uniform(0xE276, 35, -1.0, 1.0);
        let gradients = vec![seeded_uniform(0xE277, master.len(), -0.25, 0.25)];
        let spec = DeviceTrainParam {
            master: &master,
            rows: 5,
            cols: 7,
            salt_planes: 1,
            optimizer: AdamW::new(1e-3),
        };
        let mut trainer = HostOffloadTrainer::new(&backend, &[spec]).unwrap();
        let initial_stats = trainer.stats();

        for invalid_step in [0, 2] {
            assert!(matches!(
                trainer.step(upload_test_gradients(&backend, &gradients), invalid_step),
                Err(BackendError::InvalidInput(_))
            ));
            assert_eq!(trainer.completed_step(), 0);
            assert_eq!(trainer.master(0).unwrap(), master);
            assert_eq!(
                trainer.moments(0).unwrap(),
                (&[0.0; 35][..], &[0.0; 35][..])
            );
            assert_eq!(trainer.stats(), initial_stats);
            assert!(!trainer.is_poisoned());
        }

        trainer
            .step(upload_test_gradients(&backend, &gradients), 1)
            .unwrap();
        let after_master = trainer.master(0).unwrap().to_vec();
        let (after_m, after_v) = trainer.moments(0).unwrap();
        let (after_m, after_v) = (after_m.to_vec(), after_v.to_vec());
        let after_stats = trainer.stats();

        for invalid_step in [1, 3] {
            assert!(matches!(
                trainer.step(upload_test_gradients(&backend, &gradients), invalid_step),
                Err(BackendError::InvalidInput(message)) if message.contains("expected step 2")
            ));
            assert_eq!(trainer.completed_step(), 1);
            assert_eq!(trainer.master(0).unwrap(), after_master);
            assert_eq!(
                trainer.moments(0).unwrap(),
                (after_m.as_slice(), after_v.as_slice())
            );
            assert_eq!(trainer.stats(), after_stats);
            assert!(!trainer.is_poisoned());
        }
    }

    #[test]
    fn host_offload_streamed_steps_must_be_contiguous_before_mutation() {
        let backend = match CudaBackend::new(0) {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!(
                    "skipping host_offload_streamed_steps_must_be_contiguous_before_mutation: \
                     no CUDA ({error})"
                );
                return;
            }
        };
        let (batch, rows, cols) = (3usize, 5usize, 7usize);
        let input = seeded_uniform(0xE278, batch * cols, -1.0, 1.0);
        let master = seeded_uniform(0xE279, rows * cols, -1.0, 1.0);
        let target =
            DeviceTensor::upload(&backend, &vec![1.0 / rows as f32; batch * rows]).unwrap();
        let spec = DeviceTrainParam {
            master: &master,
            rows,
            cols,
            salt_planes: 1,
            optimizer: AdamW::new(1e-3),
        };
        let mut trainer = HostOffloadTrainer::new(&backend, &[spec]).unwrap();
        let build = |weight: &[f32]| {
            let mut tape = DeviceTape::new(&backend, rows).unwrap();
            let x = tape.leaf(&input).unwrap();
            let w = tape.leaf(weight).unwrap();
            let logits = tape.matmul(x, w, batch, rows, cols).unwrap();
            (tape, w, logits)
        };
        let run = |trainer: &mut HostOffloadTrainer<'_>, weight: &[f32], step| {
            let (tape, w, logits) = build(weight);
            tape.xent_backward_into(
                logits,
                &target,
                batch,
                rows,
                &[GradientLeafBinding {
                    leaf_id: w,
                    parameter_index: 0,
                }],
                trainer,
                step,
            )
        };

        assert!(matches!(
            run(&mut trainer, &master, 2),
            Err(BackendError::InvalidInput(message)) if message.contains("expected step 1")
        ));
        assert_eq!(trainer.completed_step(), 0);
        assert_eq!(trainer.master(0).unwrap(), master);
        assert!(!trainer.is_poisoned());

        run(&mut trainer, &master, 1).unwrap();
        let after_master = trainer.master(0).unwrap().to_vec();
        let (after_m, after_v) = trainer.moments(0).unwrap();
        let (after_m, after_v) = (after_m.to_vec(), after_v.to_vec());
        let after_stats = trainer.stats();

        for invalid_step in [1, 3] {
            assert!(matches!(
                run(&mut trainer, &after_master, invalid_step),
                Err(BackendError::InvalidInput(message)) if message.contains("expected step 2")
            ));
            assert_eq!(trainer.completed_step(), 1);
            assert_eq!(trainer.master(0).unwrap(), after_master);
            assert_eq!(
                trainer.moments(0).unwrap(),
                (after_m.as_slice(), after_v.as_slice())
            );
            assert_eq!(trainer.stats(), after_stats);
            assert!(!trainer.is_poisoned());
        }
    }

    #[test]
    fn packed_salt_repack_tracks_host_offload_updates() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping packed_salt_repack_tracks_host_offload_updates: {e}");
                return;
            }
        };
        let (batch, rows, cols, planes) = (3usize, 5usize, 7usize, 2usize);
        let input = seeded_uniform(0xD380, batch * cols, -1.0, 1.0);
        let target = vec![1.0 / rows as f32; batch * rows];
        let initial = seeded_uniform(0xD381, rows * cols, -1.25, 1.25);
        let optimizer = AdamW {
            lr: 0.03,
            beta1: 0.8,
            beta2: 0.9,
            eps: 0.2,
            weight_decay: 0.01,
        };
        let spec = DeviceTrainParam {
            master: &initial,
            rows,
            cols,
            salt_planes: planes,
            optimizer,
        };
        let mut offload = HostOffloadTrainer::new(&backend, &[spec]).unwrap();
        let mut packed =
            DevicePackedSaltWeight::from_host(&backend, &initial, rows, cols, planes).unwrap();
        let d_input = DeviceTensor::upload(&backend, &input).unwrap();
        let d_target = DeviceTensor::upload(&backend, &target).unwrap();
        let mut expected = initial.clone();
        let mut expected_state = optimizer.init_state(expected.len());

        for step in 1..=3u64 {
            let dense_w = ste::salt_quantize_forward(&expected, rows, cols, planes);
            let logits = matmul::forward(&input, &dense_w, &vec![1.0; rows], batch, rows, cols);
            let g_logits = loss::softmax_xent_vjp(&logits, &target, batch, rows, &[1.0]).remove(0);
            let grad = matmul::vjp(
                &input,
                &dense_w,
                &vec![1.0; rows],
                batch,
                rows,
                cols,
                &g_logits,
            )
            .remove(1);
            optimizer.step(step, &mut expected, &grad, &mut expected_state);

            let grads = {
                let mut tape = DeviceTape::new(&backend, rows).unwrap();
                let x = tape.leaf_device(&d_input).unwrap();
                let master = tape.gradient_leaf(rows * cols).unwrap();
                let logits = tape.salt_matmul(x, master, &packed, batch).unwrap();
                tape.xent_backward_device(logits, &d_target, batch, rows, &[master])
                    .unwrap()
            };
            offload.step(grads, step).unwrap();
            assert!(
                max_abs_diff(offload.master(0).unwrap(), &expected) < 1e-5,
                "offloaded master diverged at step {step}"
            );
            packed.mark_stale();
            packed
                .repack_from_host(&backend, offload.master(0).unwrap())
                .unwrap();
        }
    }

    #[test]
    fn host_offload_validates_all_gradients_before_mutation() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "skipping host_offload_validates_all_gradients_before_mutation: no CUDA ({e})"
                );
                return;
            }
        };
        let master = seeded_uniform(0xE274, 35, -1.0, 1.0);
        let spec = DeviceTrainParam {
            master: &master,
            rows: 5,
            cols: 7,
            salt_planes: 1,
            optimizer: AdamW::new(1e-3),
        };
        let mut trainer = HostOffloadTrainer::new(&backend, &[spec]).unwrap();
        let before = trainer.master(0).unwrap().to_vec();

        assert!(matches!(
            trainer.step(
                DeviceGradients {
                    bufs: Vec::new(),
                    stats: DeviceBackwardStats::default(),
                },
                1,
            ),
            Err(BackendError::ShapeMismatch { .. })
        ));
        let short = DeviceGradients {
            bufs: vec![backend.dev_alloc_zeros(master.len() - 1).unwrap()],
            stats: DeviceBackwardStats::default(),
        };
        assert!(matches!(
            trainer.step(short, 1),
            Err(BackendError::ShapeMismatch { .. })
        ));
        assert_eq!(trainer.master(0).unwrap(), before);
        assert_eq!(
            trainer.stats().peak_optimizer_device_elements,
            master.len() * 6
        );
        assert_eq!(trainer.stats().peak_in_flight_parameters, 0);
        assert_eq!(trainer.stats().resident_input_gradient_elements, 0);

        if let Ok(other_backend) = CudaBackend::new(1) {
            let foreign = DeviceGradients {
                bufs: vec![other_backend.dev_alloc_zeros(master.len()).unwrap()],
                stats: DeviceBackwardStats::default(),
            };
            assert!(matches!(
                trainer.step(foreign, 1),
                Err(BackendError::InvalidInput(message)) if message.contains("different CUDA context")
            ));
            assert_eq!(trainer.master(0).unwrap(), before);
        }
    }

    fn host_offload_dcp_dir(label: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "tritium-host-offload-{label}-{}-{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn host_offload_streaming_dcp_roundtrips_unequal_leaves_across_worlds() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "skipping host_offload_streaming_dcp_roundtrips_unequal_leaves_across_worlds: \
                     no CUDA ({e})"
                );
                return;
            }
        };
        let masters = [
            seeded_uniform(0xD270, 3, -1.0, 1.0),
            seeded_uniform(0xD271, 17, -1.0, 1.0),
            seeded_uniform(0xD272, 5, -1.0, 1.0),
        ];
        let optimizer = AdamW {
            lr: 0.03,
            beta1: 0.8,
            beta2: 0.9,
            eps: 0.2,
            weight_decay: 0.01,
        };
        let specs = [
            DeviceTrainParam {
                master: &masters[0],
                rows: 1,
                cols: 3,
                salt_planes: 1,
                optimizer,
            },
            DeviceTrainParam {
                master: &masters[1],
                rows: 1,
                cols: 17,
                salt_planes: 2,
                optimizer,
            },
            DeviceTrainParam {
                master: &masters[2],
                rows: 1,
                cols: 5,
                salt_planes: 3,
                optimizer,
            },
        ];
        let grads: Vec<Vec<f32>> = masters
            .iter()
            .enumerate()
            .map(|(index, master)| seeded_uniform(0xD280 + index as u64, master.len(), -0.25, 0.25))
            .collect();
        let mut source = HostOffloadTrainer::new(&backend, &specs).unwrap();
        for step in 1..=7 {
            source
                .step(upload_test_gradients(&backend, &grads), step)
                .unwrap();
        }
        assert_eq!(source.completed_step(), 7);
        assert_eq!(source.leaf_lens(), &[3, 17, 5]);

        let first = host_offload_dcp_dir("world3");
        let second = host_offload_dcp_dir("world2");
        tritium_train::dcp::save_from(&first, &mut source, 3).unwrap();

        let zeros = [vec![0.0; 3], vec![0.0; 17], vec![0.0; 5]];
        let restore_specs = [
            DeviceTrainParam {
                master: &zeros[0],
                ..specs[0]
            },
            DeviceTrainParam {
                master: &zeros[1],
                ..specs[1]
            },
            DeviceTrainParam {
                master: &zeros[2],
                ..specs[2]
            },
        ];
        let mut restored = HostOffloadTrainer::new(&backend, &restore_specs).unwrap();
        tritium_train::dcp::load_into(&first, &mut restored).unwrap();
        assert_eq!(restored.completed_step(), 7);
        assert!(!restored.is_poisoned());
        for index in 0..masters.len() {
            assert_eq!(
                restored.master(index).unwrap(),
                source.master(index).unwrap()
            );
            assert_eq!(
                restored.moments(index).unwrap(),
                source.moments(index).unwrap()
            );
        }

        let packed_source =
            DevicePackedSaltWeight::from_host(&backend, source.master(1).unwrap(), 1, 17, 2)
                .unwrap();
        let mut packed_restored =
            DevicePackedSaltWeight::from_host(&backend, &zeros[1], 1, 17, 2).unwrap();
        packed_restored.mark_stale();
        packed_restored
            .repack_from_host(&backend, restored.master(1).unwrap())
            .unwrap();
        let input = seeded_uniform(0xD290, 2 * 17, -0.5, 0.5);
        let packed_output = |packed: &DevicePackedSaltWeight| {
            let mut tape = DeviceTape::new(&backend, 1).unwrap();
            let x = tape.leaf(&input).unwrap();
            let master = tape.gradient_leaf(17).unwrap();
            let out = tape.salt_matmul(x, master, packed, 2).unwrap();
            tape.value(out).unwrap()
        };
        assert_eq!(
            packed_output(&packed_restored),
            packed_output(&packed_source),
            "restored host masters must deterministically rebuild packed SALT state"
        );

        tritium_train::dcp::save_from(&second, &mut restored, 2).unwrap();
        let mut resharded = HostOffloadTrainer::new(&backend, &restore_specs).unwrap();
        tritium_train::dcp::load_into(&second, &mut resharded).unwrap();
        for index in 0..masters.len() {
            assert_eq!(
                resharded.master(index).unwrap(),
                source.master(index).unwrap()
            );
            assert_eq!(
                resharded.moments(index).unwrap(),
                source.moments(index).unwrap()
            );
        }

        let _ = std::fs::remove_dir_all(first);
        let _ = std::fs::remove_dir_all(second);
    }

    #[test]
    fn host_offload_dcp_chunks_cross_leaves_and_failed_loads_stay_poisoned() {
        use tritium_train::dcp::{StateSink, StateSource};

        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "skipping host_offload_dcp_chunks_cross_leaves_and_failed_loads_stay_poisoned: \
                     no CUDA ({e})"
                );
                return;
            }
        };
        let masters = [
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0],
            vec![6.0, 7.0, 8.0, 9.0],
        ];
        let specs = [
            DeviceTrainParam {
                master: &masters[0],
                rows: 1,
                cols: 3,
                salt_planes: 1,
                optimizer: AdamW::new(0.01),
            },
            DeviceTrainParam {
                master: &masters[1],
                rows: 1,
                cols: 2,
                salt_planes: 1,
                optimizer: AdamW::new(0.01),
            },
            DeviceTrainParam {
                master: &masters[2],
                rows: 1,
                cols: 4,
                salt_planes: 1,
                optimizer: AdamW::new(0.01),
            },
        ];
        let mut trainer = HostOffloadTrainer::new(&backend, &specs).unwrap();
        let mut crossing = [0.0; 6];
        StateSource::read_chunk(&mut trainer, StatePlane::Parameter, 2, &mut crossing).unwrap();
        assert_eq!(crossing, [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert!(matches!(
            StateSource::read_chunk(
                &mut trainer,
                StatePlane::Optimizer(2),
                0,
                &mut crossing[..1],
            ),
            Err(DcpError::InvalidState(_))
        ));

        assert!(matches!(
            StateSink::begin(&mut trainer, 4, &[3, 3, 3], 2),
            Err(DcpError::InvalidState(_))
        ));
        assert!(trainer.is_poisoned());
        assert!(trainer.master(0).is_err());

        StateSink::begin(&mut trainer, 5, &[3, 2, 4], 2).unwrap();
        StateSink::write_chunk(
            &mut trainer,
            StatePlane::Parameter,
            0,
            &[11.0, 12.0, 13.0, 14.0],
        )
        .unwrap();
        assert!(matches!(
            StateSink::finish(&mut trainer),
            Err(DcpError::InvalidState(_))
        ));
        assert!(trainer.is_poisoned());

        StateSink::begin(&mut trainer, 6, &[3, 2, 4], 2).unwrap();
        assert!(matches!(
            StateSink::write_chunk(&mut trainer, StatePlane::Optimizer(2), 0, &[1.0]),
            Err(DcpError::InvalidState(_))
        ));
        assert!(trainer.is_poisoned());

        let parameters = [21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0];
        let first_moment = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let second_moment = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9];
        StateSink::begin(&mut trainer, 7, &[3, 2, 4], 2).unwrap();
        StateSink::write_chunk(&mut trainer, StatePlane::Parameter, 0, &parameters[..5]).unwrap();
        StateSink::write_chunk(&mut trainer, StatePlane::Parameter, 5, &parameters[5..]).unwrap();
        StateSink::write_chunk(&mut trainer, StatePlane::Optimizer(0), 0, &first_moment).unwrap();
        StateSink::write_chunk(&mut trainer, StatePlane::Optimizer(1), 0, &second_moment).unwrap();
        StateSink::finish(&mut trainer).unwrap();
        assert!(!trainer.is_poisoned());
        assert_eq!(trainer.completed_step(), 7);
        assert_eq!(trainer.master(0).unwrap(), &parameters[..3]);
        assert_eq!(trainer.master(1).unwrap(), &parameters[3..5]);
        assert_eq!(trainer.master(2).unwrap(), &parameters[5..]);
        assert_eq!(trainer.moments(1).unwrap().0, &first_moment[3..5]);
        assert_eq!(trainer.moments(1).unwrap().1, &second_moment[3..5]);
    }

    #[test]
    fn streamed_host_offload_emits_without_collecting_device_gradients() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "skipping streamed_host_offload_emits_without_collecting_device_gradients: \
                     no CUDA ({e})"
                );
                return;
            }
        };
        let (rows, cols, batch) = (5usize, 7usize, 3usize);
        let input = seeded_uniform(0xE400, batch * cols, -1.0, 1.0);
        let master = seeded_uniform(0xE401, rows * cols, -1.0, 1.0);
        let target =
            DeviceTensor::upload(&backend, &vec![1.0 / rows as f32; batch * rows]).unwrap();
        let spec = DeviceTrainParam {
            master: &master,
            rows,
            cols,
            salt_planes: 1,
            optimizer: AdamW::new(1e-3),
        };
        let mut trainer = HostOffloadTrainer::new(&backend, &[spec]).unwrap();
        let mut tape = DeviceTape::new(&backend, rows).unwrap();
        let x = tape.leaf(&input).unwrap();
        let w = tape.leaf(&master).unwrap();
        let logits = tape.matmul(x, w, batch, rows, cols).unwrap();

        let report = tape
            .xent_backward_into(
                logits,
                &target,
                batch,
                rows,
                &[GradientLeafBinding {
                    leaf_id: w,
                    parameter_index: 0,
                }],
                &mut trainer,
                1,
            )
            .unwrap();

        assert_eq!(report.emissions.len(), 1);
        assert_eq!(report.peak_live_requested_gradient_elements, master.len());
        assert!(!trainer.is_poisoned());
        assert_eq!(trainer.completed_step(), 1);
    }

    #[test]
    fn streamed_host_offload_matches_collected_across_steps_and_checkpointing() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "skipping streamed_host_offload_matches_collected_across_steps_and_checkpointing: \
                     no CUDA ({e})"
                );
                return;
            }
        };
        let (batch, input_dim, hidden_dim, output_dim) = (3usize, 5usize, 7usize, 4usize);
        let input = seeded_uniform(0xE410, batch * input_dim, -0.5, 0.5);
        let target =
            DeviceTensor::upload(&backend, &vec![1.0 / output_dim as f32; batch * output_dim])
                .unwrap();
        let initial_w1 = seeded_uniform(0xE411, hidden_dim * input_dim, -0.3, 0.3);
        let initial_w2 = seeded_uniform(0xE412, output_dim * hidden_dim, -0.3, 0.3);
        let optimizer = AdamW {
            lr: 0.02,
            beta1: 0.8,
            beta2: 0.9,
            eps: 0.2,
            weight_decay: 0.01,
        };
        let specs = [
            DeviceTrainParam {
                master: &initial_w1,
                rows: hidden_dim,
                cols: input_dim,
                salt_planes: 1,
                optimizer,
            },
            DeviceTrainParam {
                master: &initial_w2,
                rows: output_dim,
                cols: hidden_dim,
                salt_planes: 1,
                optimizer,
            },
        ];
        let mut collected = HostOffloadTrainer::new(&backend, &specs).unwrap();
        let mut streamed = HostOffloadTrainer::new(&backend, &specs).unwrap();

        for step in 1..=3u64 {
            let collected_w1 = collected.master(0).unwrap().to_vec();
            let collected_w2 = collected.master(1).unwrap().to_vec();
            let mut tape = DeviceTape::new(&backend, hidden_dim.max(output_dim)).unwrap();
            let x = tape.leaf(&input).unwrap();
            let w1 = tape.leaf(&collected_w1).unwrap();
            let w2 = tape.leaf(&collected_w2).unwrap();
            let hidden = tape.matmul(x, w1, batch, hidden_dim, input_dim).unwrap();
            let hidden = tape.silu(hidden).unwrap();
            let logits = tape
                .matmul(hidden, w2, batch, output_dim, hidden_dim)
                .unwrap();
            let grads = tape
                .xent_backward_device(logits, &target, batch, output_dim, &[w1, w2])
                .unwrap();
            collected.step(grads, step).unwrap();

            let streamed_w1 = streamed.master(0).unwrap().to_vec();
            let streamed_w2 = streamed.master(1).unwrap().to_vec();
            let mut tape = DeviceTape::new_with_checkpoint_policy(
                &backend,
                hidden_dim.max(output_dim),
                CheckpointPolicy::EveryBlocks(1),
            )
            .unwrap();
            let x = tape.leaf(&input).unwrap();
            let w1 = tape.leaf(&streamed_w1).unwrap();
            let w2 = tape.leaf(&streamed_w2).unwrap();
            let hidden = tape.matmul(x, w1, batch, hidden_dim, input_dim).unwrap();
            let hidden = tape.silu(hidden).unwrap();
            tape.checkpoint_keep(&[hidden]).unwrap();
            let logits = tape
                .matmul(hidden, w2, batch, output_dim, hidden_dim)
                .unwrap();
            let report = tape
                .xent_backward_into(
                    logits,
                    &target,
                    batch,
                    output_dim,
                    &[
                        GradientLeafBinding {
                            leaf_id: w1,
                            parameter_index: 0,
                        },
                        GradientLeafBinding {
                            leaf_id: w2,
                            parameter_index: 1,
                        },
                    ],
                    &mut streamed,
                    step,
                )
                .unwrap();
            assert_eq!(
                report
                    .emissions
                    .iter()
                    .map(|emission| emission.parameter_index)
                    .collect::<Vec<_>>(),
                vec![1, 0]
            );
            assert!(report.backward_stats.recomputed_ops > 0);
            assert!(
                report.peak_live_requested_gradient_elements
                    < report.materialized_collection_elements,
                "stream must reduce requested-gradient peak: {report:?}"
            );
            if step == 1 {
                eprintln!(
                    "0027 streamed gradients: requested peak {}/{} elements, emission order {:?}",
                    report.peak_live_requested_gradient_elements,
                    report.materialized_collection_elements,
                    report
                        .emissions
                        .iter()
                        .map(|emission| emission.parameter_index)
                        .collect::<Vec<_>>()
                );
            }

            for index in 0..2 {
                assert!(
                    max_abs_diff(
                        collected.master(index).unwrap(),
                        streamed.master(index).unwrap()
                    ) < 1e-5,
                    "master {index} diverged at step {step}"
                );
                let (collected_m, collected_v) = collected.moments(index).unwrap();
                let (streamed_m, streamed_v) = streamed.moments(index).unwrap();
                assert!(max_abs_diff(collected_m, streamed_m) < 1e-5);
                assert!(max_abs_diff(collected_v, streamed_v) < 1e-5);
            }
        }
        assert_eq!(collected.completed_step(), 3);
        assert_eq!(streamed.completed_step(), 3);
    }

    #[test]
    fn streamed_tied_packed_master_emits_once() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping streamed_tied_packed_master_emits_once: no CUDA ({e})");
                return;
            }
        };
        let (seq, vocab, dim, planes) = (4usize, 5usize, 7usize, 2usize);
        let tokens = [4i32, 0, 4, 2];
        let master = seeded_uniform(0xE420, vocab * dim, -1.0, 1.0);
        let target =
            DeviceTensor::upload(&backend, &vec![1.0 / vocab as f32; seq * vocab]).unwrap();
        let spec = DeviceTrainParam {
            master: &master,
            rows: vocab,
            cols: dim,
            salt_planes: planes,
            optimizer: AdamW::new(0.01),
        };
        let mut collected = HostOffloadTrainer::new(&backend, &[spec]).unwrap();
        let mut streamed = HostOffloadTrainer::new(&backend, &[spec]).unwrap();

        let packed = DevicePackedSaltWeight::from_host(
            &backend,
            collected.master(0).unwrap(),
            vocab,
            dim,
            planes,
        )
        .unwrap();
        let mut tape = DeviceTape::new(&backend, vocab).unwrap();
        let master_id = tape.gradient_leaf(vocab * dim).unwrap();
        let emb = tape.salt_embed(master_id, &packed, &tokens).unwrap();
        let logits = tape.salt_matmul(emb, master_id, &packed, seq).unwrap();
        let grads = tape
            .xent_backward_device(logits, &target, seq, vocab, &[master_id])
            .unwrap();
        collected.step(grads, 1).unwrap();

        let packed = DevicePackedSaltWeight::from_host(
            &backend,
            streamed.master(0).unwrap(),
            vocab,
            dim,
            planes,
        )
        .unwrap();
        let mut tape = DeviceTape::new(&backend, vocab).unwrap();
        let master_id = tape.gradient_leaf(vocab * dim).unwrap();
        let emb = tape.salt_embed(master_id, &packed, &tokens).unwrap();
        let logits = tape.salt_matmul(emb, master_id, &packed, seq).unwrap();
        let report = tape
            .xent_backward_into(
                logits,
                &target,
                seq,
                vocab,
                &[GradientLeafBinding {
                    leaf_id: master_id,
                    parameter_index: 0,
                }],
                &mut streamed,
                1,
            )
            .unwrap();

        assert_eq!(report.emissions.len(), 1);
        assert_eq!(report.emissions[0].leaf_id, master_id);
        assert!(max_abs_diff(collected.master(0).unwrap(), streamed.master(0).unwrap()) < 1e-5);
        let (collected_m, collected_v) = collected.moments(0).unwrap();
        let (streamed_m, streamed_v) = streamed.moments(0).unwrap();
        assert!(max_abs_diff(collected_m, streamed_m) < 1e-5);
        assert!(max_abs_diff(collected_v, streamed_v) < 1e-5);
    }

    #[test]
    fn streamed_completion_handles_duplicate_edges_and_unused_leaf() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "skipping streamed_completion_handles_duplicate_edges_and_unused_leaf: \
                     no CUDA ({e})"
                );
                return;
            }
        };
        let p = seeded_uniform(0xE430, 4, -0.5, 0.5);
        let q = seeded_uniform(0xE431, 4, -0.5, 0.5);
        let unused = seeded_uniform(0xE432, 3, -0.5, 0.5);
        let specs = [
            DeviceTrainParam {
                master: &p,
                rows: 1,
                cols: 4,
                salt_planes: 1,
                optimizer: AdamW::new(0.01),
            },
            DeviceTrainParam {
                master: &q,
                rows: 1,
                cols: 4,
                salt_planes: 1,
                optimizer: AdamW::new(0.01),
            },
            DeviceTrainParam {
                master: &unused,
                rows: 1,
                cols: 3,
                salt_planes: 1,
                optimizer: AdamW::new(0.01),
            },
        ];
        let target = DeviceTensor::upload(&backend, &[1.0 / 12.0; 12]).unwrap();
        let build = |p: &[f32], q: &[f32], unused: &[f32]| {
            let mut tape = DeviceTape::new(&backend, 12).unwrap();
            let p_id = tape.leaf(p).unwrap();
            let q_id = tape.leaf(q).unwrap();
            let unused_id = tape.leaf(unused).unwrap();
            let duplicate_concat = tape.concat(&[p_id, p_id], 1, &[4, 4]).unwrap();
            let duplicate_add = tape.add(q_id, q_id).unwrap();
            let logits = tape
                .concat(&[duplicate_concat, duplicate_add], 1, &[8, 4])
                .unwrap();
            (tape, [p_id, q_id, unused_id], logits)
        };

        let mut collected = HostOffloadTrainer::new(&backend, &specs).unwrap();
        let (tape, ids, logits) = build(&p, &q, &unused);
        let grads = tape
            .xent_backward_device(logits, &target, 1, 12, &ids)
            .unwrap();
        collected.step(grads, 1).unwrap();

        let mut streamed = HostOffloadTrainer::new(&backend, &specs).unwrap();
        let (tape, ids, logits) = build(&p, &q, &unused);
        let report = tape
            .xent_backward_into(
                logits,
                &target,
                1,
                12,
                &[
                    GradientLeafBinding {
                        leaf_id: ids[0],
                        parameter_index: 0,
                    },
                    GradientLeafBinding {
                        leaf_id: ids[1],
                        parameter_index: 1,
                    },
                    GradientLeafBinding {
                        leaf_id: ids[2],
                        parameter_index: 2,
                    },
                ],
                &mut streamed,
                1,
            )
            .unwrap();
        assert_eq!(
            report
                .emissions
                .iter()
                .map(|emission| emission.parameter_index)
                .collect::<Vec<_>>(),
            vec![1, 0, 2]
        );
        for index in 0..3 {
            assert!(
                max_abs_diff(
                    collected.master(index).unwrap(),
                    streamed.master(index).unwrap()
                ) < 1e-5
            );
        }
    }

    #[test]
    fn streamed_resident_adam_matches_collected_and_bounds_requested_gradients() {
        let backend = match CudaBackend::new(0) {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping streamed resident Adam gate: no CUDA ({error})");
                return;
            }
        };
        let p = seeded_uniform(0xE431, 4, -0.5, 0.5);
        let q = seeded_uniform(0xE432, 4, -0.5, 0.5);
        let unused = seeded_uniform(0xE433, 3, -0.5, 0.5);
        let specs = [
            DeviceTrainParam {
                master: &p,
                rows: 1,
                cols: 4,
                salt_planes: 1,
                optimizer: AdamW::new(0.01),
            },
            DeviceTrainParam {
                master: &q,
                rows: 1,
                cols: 4,
                salt_planes: 1,
                optimizer: AdamW::new(0.01),
            },
            DeviceTrainParam {
                master: &unused,
                rows: 1,
                cols: 3,
                salt_planes: 1,
                optimizer: AdamW::new(0.01),
            },
        ];
        let target = DeviceTensor::upload(&backend, &[1.0 / 12.0; 12]).unwrap();
        let build = || {
            let mut tape = DeviceTape::new(&backend, 12).unwrap();
            let p_id = tape.leaf(&p).unwrap();
            let q_id = tape.leaf(&q).unwrap();
            let unused_id = tape.leaf(&unused).unwrap();
            let duplicate_concat = tape.concat(&[p_id, p_id], 1, &[4, 4]).unwrap();
            let duplicate_add = tape.add(q_id, q_id).unwrap();
            let logits = tape
                .concat(&[duplicate_concat, duplicate_add], 1, &[8, 4])
                .unwrap();
            (tape, [p_id, q_id, unused_id], logits)
        };

        let mut collected = DeviceTrainer::new_with_weight_storage(
            &backend,
            &specs,
            DeviceTrainerWeightStorage::Packed,
        )
        .unwrap();
        let (tape, ids, logits) = build();
        let gradients = tape
            .xent_backward_device(logits, &target, 1, 12, &ids)
            .unwrap();
        collected.step(gradients, 1).unwrap();

        let mut streamed = DeviceTrainer::new_with_weight_storage(
            &backend,
            &specs,
            DeviceTrainerWeightStorage::Packed,
        )
        .unwrap();
        let (tape, ids, logits) = build();
        let report = tape
            .xent_backward_into_resident(
                logits,
                &target,
                1,
                12,
                &[
                    GradientLeafBinding {
                        leaf_id: ids[0],
                        parameter_index: 0,
                    },
                    GradientLeafBinding {
                        leaf_id: ids[1],
                        parameter_index: 1,
                    },
                    GradientLeafBinding {
                        leaf_id: ids[2],
                        parameter_index: 2,
                    },
                ],
                &mut streamed,
                1,
            )
            .unwrap();
        assert_eq!(streamed.completed_step(), 1);
        assert!(report.peak_live_requested_gradient_elements < 11);
        assert_eq!(report.materialized_collection_elements, 11);
        for index in 0..3 {
            assert!(
                max_abs_diff(
                    &collected.download_master(index).unwrap(),
                    &streamed.download_master(index).unwrap(),
                ) < 1e-5
            );
        }
    }

    #[test]
    fn streamed_plan_errors_precede_host_mutation() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping streamed_plan_errors_precede_host_mutation: no CUDA ({e})");
                return;
            }
        };
        let master = seeded_uniform(0xE440, 35, -0.5, 0.5);
        let input = seeded_uniform(0xE441, 21, -0.5, 0.5);
        let target = DeviceTensor::upload(&backend, &[0.2; 15]).unwrap();
        let spec = DeviceTrainParam {
            master: &master,
            rows: 5,
            cols: 7,
            salt_planes: 1,
            optimizer: AdamW::new(0.01),
        };
        let mut trainer = HostOffloadTrainer::new(&backend, &[spec]).unwrap();
        let before = trainer.master(0).unwrap().to_vec();
        let build = || {
            let mut tape = DeviceTape::new(&backend, 5).unwrap();
            let x = tape.leaf(&input).unwrap();
            let w = tape.leaf(&master).unwrap();
            let logits = tape.matmul(x, w, 3, 5, 7).unwrap();
            (tape, w, logits)
        };

        let (tape, w, logits) = build();
        assert!(matches!(
            tape.xent_backward_into(
                logits,
                &target,
                3,
                5,
                &[GradientLeafBinding {
                    leaf_id: w,
                    parameter_index: 0,
                }],
                &mut trainer,
                0,
            ),
            Err(BackendError::InvalidInput(message)) if message.contains("1-based")
        ));
        let (tape, _, logits) = build();
        assert!(matches!(
            tape.xent_backward_into(logits, &target, 3, 5, &[], &mut trainer, 1),
            Err(BackendError::ShapeMismatch { .. })
        ));
        let (tape, _, logits) = build();
        assert!(matches!(
            tape.xent_backward_into(
                logits,
                &target,
                3,
                5,
                &[GradientLeafBinding {
                    leaf_id: logits,
                    parameter_index: 0,
                }],
                &mut trainer,
                1,
            ),
            Err(BackendError::InvalidInput(message)) if message.contains("not a leaf")
        ));
        assert_eq!(trainer.master(0).unwrap(), before);
        assert!(!trainer.is_poisoned());
        assert_eq!(
            trainer.stats().peak_optimizer_device_elements,
            master.len() * 6
        );
        assert_eq!(trainer.stats().peak_in_flight_parameters, 0);
    }

    /// Two backend handles retaining the same CUDA primary context may share a
    /// tensor even though their Rust `Arc`s and streams differ.
    #[test]
    fn device_tensor_accepts_same_primary_context() {
        let first = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping device_tensor_accepts_same_primary_context: no CUDA ({e})");
                return;
            }
        };
        let second = CudaBackend::new(0).unwrap();
        let tensor = DeviceTensor::upload(&first, &[1.0, 2.0, 3.0]).unwrap();
        let mut tape = DeviceTape::new(&second, 1).unwrap();
        let id = tape.leaf_device(&tensor).unwrap();
        assert_eq!(tape.value(id).unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn resident_backward_rejects_invalid_gradient_requests() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "skipping resident_backward_rejects_invalid_gradient_requests: no CUDA ({e})"
                );
                return;
            }
        };
        let target = DeviceTensor::upload(&backend, &[1.0, 0.0]).unwrap();
        let build = || {
            let mut tape = DeviceTape::new(&backend, 2).unwrap();
            let logits = tape.leaf(&[0.25, -0.5]).unwrap();
            (tape, logits)
        };

        let (tape, logits) = build();
        assert!(matches!(
            tape.xent_backward_device(logits, &target, 1, 2, &[logits, logits]),
            Err(BackendError::InvalidInput(message)) if message.contains("duplicated")
        ));
        let (tape, logits) = build();
        assert!(matches!(
            tape.xent_backward_device(logits, &target, 1, 2, &[usize::MAX]),
            Err(BackendError::InvalidInput(message)) if message.contains("out of range")
        ));
        let mut tape = DeviceTape::new(&backend, 2).unwrap();
        let x = tape.leaf(&[1.0, -1.0]).unwrap();
        let w = tape.leaf(&[0.5, 0.25, -0.5, 0.75]).unwrap();
        let logits = tape.matmul(x, w, 1, 2, 2).unwrap();
        assert!(matches!(
            tape.xent_backward_device(logits, &target, 1, 2, &[logits]),
            Err(BackendError::InvalidInput(message)) if message.contains("not a leaf")
        ));
    }

    /// ADR 0027 Track B performance gate. Timings include stable host grouping,
    /// metadata upload, output zeroing, kernel execution, and synchronization.
    #[test]
    #[ignore = "4090 performance gate; run explicitly with --ignored --nocapture"]
    fn embed_gather_segmented_is_five_times_faster() {
        use std::time::Instant;

        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping embed_gather_segmented_is_five_times_faster: no CUDA ({e})");
                return;
            }
        };
        // seq=512 is the scale where the reference O(vocab*seq*dim) scan is a
        // material training bottleneck; Track A separately reports seq=32.
        let (vocab, dim, seq) = (49_152usize, 576usize, 512usize);
        let tokens: Vec<i32> = (0..seq).map(|i| ((i * 1531) % vocab) as i32).collect();
        let gy = seeded_uniform(0xB270, seq * dim, -1.0, 1.0);
        let d_gy = backend.dev_upload(&gy).unwrap();
        let d_tokens = backend.dev_upload_i32(&tokens).unwrap();
        let mut d_reference = backend.dev_alloc_zeros(vocab * dim).unwrap();
        let mut d_segmented = backend.dev_alloc_zeros(vocab * dim).unwrap();

        let mut reference = || {
            let start = Instant::now();
            backend
                .embed_gather_backward_dev(&d_gy, &d_tokens, &mut d_reference, seq, dim, vocab)
                .unwrap();
            backend.dev_synchronize().unwrap();
            start.elapsed().as_secs_f64() * 1e3
        };
        let mut segmented = || {
            let start = Instant::now();
            backend
                .embed_gather_backward_segmented_dev(
                    &d_gy,
                    &tokens,
                    &mut d_segmented,
                    seq,
                    dim,
                    vocab,
                )
                .unwrap();
            backend.dev_synchronize().unwrap();
            start.elapsed().as_secs_f64() * 1e3
        };
        for _ in 0..10 {
            reference();
            segmented();
        }
        let mut reference_ms = Vec::with_capacity(21);
        let mut segmented_ms = Vec::with_capacity(21);
        for _ in 0..21 {
            reference_ms.push(reference());
            segmented_ms.push(segmented());
        }
        reference_ms.sort_by(f64::total_cmp);
        segmented_ms.sort_by(f64::total_cmp);
        let (reference_ms, segmented_ms) = (reference_ms[10], segmented_ms[10]);
        let speedup = reference_ms / segmented_ms;
        eprintln!(
            "0027 B embedding backward on {} ({vocab}x{dim}, seq {seq}, {seq} unique): reference \
             {reference_ms:.3}ms | segmented-total {segmented_ms:.3}ms | {speedup:.1}x faster",
            backend.dev_name()
        );
        if backend.dev_name().contains("4090") {
            assert!(
                speedup >= 5.0,
                "segmented embedding backward speedup {speedup:.2}x is below 5x gate"
            );
        }
    }

    #[test]
    fn embed_gather_segmented_edge_matrix_is_bit_exact() {
        use tritium_train::ops::embed;

        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "skipping embed_gather_segmented_edge_matrix_is_bit_exact: no CUDA ({e})"
                );
                return;
            }
        };
        let vocab = 8usize;
        let token_cases: &[&[i32]] = &[&[], &[3, 3, 3], &[0, 7], &[0, 2, 4, 6], &[5, 1, 5, 1, 5]];
        for &dim in &[1usize, 255, 256, 257, 576] {
            for &tokens in token_cases {
                let seq = tokens.len();
                let mut gy = seeded_uniform(0xB271 ^ dim as u64 ^ seq as u64, seq * dim, -1.0, 1.0);
                if tokens == [3, 3, 3] {
                    for d in 0..dim {
                        gy[d] = 1e20;
                        gy[dim + d] = -1e20;
                        gy[2 * dim + d] = 3.25;
                    }
                }
                let d_gy = backend.dev_upload(&gy).unwrap();
                let mut d_gw = backend.dev_upload(&vec![7.0; vocab * dim]).unwrap();
                for _ in 0..2 {
                    backend
                        .embed_gather_backward_segmented_dev(
                            &d_gy, tokens, &mut d_gw, seq, dim, vocab,
                        )
                        .unwrap();
                    let mut got = vec![0.0; vocab * dim];
                    backend.dev_download(&d_gw, &mut got).unwrap();
                    let tokens_u32: Vec<u32> = tokens.iter().map(|&token| token as u32).collect();
                    let expected = embed::gather_vjp(vocab, &tokens_u32, dim, &gy);
                    assert!(
                        got.iter()
                            .zip(&expected)
                            .all(|(&a, &b)| a.to_bits() == b.to_bits()),
                        "segmented mismatch for dim={dim}, tokens={tokens:?}"
                    );
                }
            }
        }

        let d_gy = backend.dev_alloc_zeros(1).unwrap();
        let mut d_gw = backend.dev_alloc_zeros(vocab).unwrap();
        for invalid in [&[-1][..], &[vocab as i32][..]] {
            assert!(matches!(
                backend
                    .embed_gather_backward_segmented_dev(&d_gy, invalid, &mut d_gw, 1, 1, vocab,),
                Err(BackendError::InvalidInput(_))
            ));
        }
        assert!(matches!(
            backend.embed_gather_backward_segmented_dev(&d_gy, &[], &mut d_gw, 1, 1, vocab,),
            Err(BackendError::ShapeMismatch { .. })
        ));
    }

    /// Gate (plan 0043 P2.3): the device-resident TRAINING RMSNorm (forward + grad_x + grad_w)
    /// matches `tritium-train`'s `ops::norm` — which uses a sequential sum, so (only +,*,/,sqrt, all
    /// IEEE correctly-rounded, --fmad=false) it is BIT-EXACT, unlike silu's expf.
    #[test]
    fn resident_rmsnorm_matches_cpu() {
        use tritium_train::ops::norm;

        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping resident_rmsnorm_matches_cpu: no CUDA device ({e})");
                return;
            }
        };
        let (rows, cols, eps) = (48usize, 576usize, 1e-5f32); // cols = SmolLM2 n_embd
        let x = seeded_uniform(0x51, rows * cols, -2.0, 2.0);
        let w = seeded_uniform(0x52, cols, 0.5, 1.5);
        let gy = seeded_uniform(0x53, rows * cols, -1.0, 1.0);
        let d_x = backend.dev_upload(&x).unwrap();
        let d_w = backend.dev_upload(&w).unwrap();
        let d_gy = backend.dev_upload(&gy).unwrap();
        let download = |d: &_, n: usize| {
            let mut h = vec![0.0f32; n];
            backend.dev_download(d, &mut h).unwrap();
            h
        };

        // forward
        let mut d_y = backend.dev_alloc_zeros(rows * cols).unwrap();
        backend
            .rmsnorm_forward_dev(&d_x, &d_w, &mut d_y, rows, cols, eps)
            .unwrap();
        let cpu_y = norm::forward(&x, &w, rows, cols, eps);
        let fwd_d = max_abs_diff(&download(&d_y, rows * cols), &cpu_y);

        // backward
        let mut d_gx = backend.dev_alloc_zeros(rows * cols).unwrap();
        let mut d_gw = backend.dev_alloc_zeros(cols).unwrap();
        backend
            .rmsnorm_backward_dev(&d_x, &d_w, &d_gy, &mut d_gx, &mut d_gw, rows, cols, eps)
            .unwrap();
        let cpu_g = norm::vjp(&x, &w, rows, cols, eps, &gy); // [gx, gw]
        let gx_d = max_abs_diff(&download(&d_gx, rows * cols), &cpu_g[0]);
        let gw_d = max_abs_diff(&download(&d_gw, cols), &cpu_g[1]);

        eprintln!(
            "0043 P2.3 resident RMSNorm (rows={rows} cols={cols}): \
             fwd max|Δ| {fwd_d:.3e} | grad_x {gx_d:.3e} | grad_w {gw_d:.3e} (0.0 = bit-exact)"
        );
        assert!(fwd_d < 1e-4, "rmsnorm forward: {fwd_d:.3e}");
        assert!(gx_d < 1e-4, "rmsnorm grad_x: {gx_d:.3e}");
        assert!(gw_d < 1e-4, "rmsnorm grad_w: {gw_d:.3e}");
    }

    /// Gate (plan 0043 P2.4): the device-resident attention glue ops vs their `tritium-train`
    /// oracles. Copy/select ops (mask, slice, insert, transpose, gather) are BIT-EXACT; softmax,
    /// RoPE and softmax-xent (expf/sin/cos) are device==CPU within 1e-4.
    #[test]
    fn resident_attention_ops_match_cpu() {
        use tritium_train::ops::{dense, embed, loss, rope, shape, softmax};

        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping resident_attention_ops_match_cpu: no CUDA device ({e})");
                return;
            }
        };
        let dl = |d: &_, n: usize| {
            let mut h = vec![0.0f32; n];
            backend.dev_download(d, &mut h).unwrap();
            h
        };
        let rel = |dev: &[f32], cpu: &[f32]| {
            let range = cpu.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                - cpu.iter().copied().fold(f32::INFINITY, f32::min);
            max_abs_diff(dev, cpu) / range.max(1e-6)
        };

        // ── softmax fwd/bwd (rows×cols attention scores) ──
        let (rows, cols) = (32usize, 40usize);
        let sx = seeded_uniform(0x81, rows * cols, -3.0, 3.0);
        let sgy = seeded_uniform(0x82, rows * cols, -1.0, 1.0);
        let d_sx = backend.dev_upload(&sx).unwrap();
        let d_sgy = backend.dev_upload(&sgy).unwrap();
        let mut d_p = backend.dev_alloc_zeros(rows * cols).unwrap();
        backend
            .softmax_forward_dev(&d_sx, &mut d_p, rows, cols)
            .unwrap();
        let cpu_p = softmax::forward(&sx, rows, cols);
        assert!(rel(&dl(&d_p, rows * cols), &cpu_p) < 1e-4, "softmax fwd");
        let mut d_sgx = backend.dev_alloc_zeros(rows * cols).unwrap();
        backend
            .softmax_backward_dev(&d_p, &d_sgy, &mut d_sgx, rows, cols)
            .unwrap();
        let cpu_sgx = softmax::vjp(&sx, rows, cols, &sgy);
        assert!(
            rel(&dl(&d_sgx, rows * cols), &cpu_sgx[0]) < 1e-4,
            "softmax bwd"
        );

        // ── causal mask fwd/bwd (bit-exact) ──
        let mut d_mask = backend.dev_alloc_zeros(rows * cols).unwrap();
        backend
            .causal_mask_forward_dev(&d_sx, &mut d_mask, rows, cols)
            .unwrap();
        assert_eq!(
            max_abs_diff(
                &dl(&d_mask, rows * cols),
                &softmax::causal_mask_forward(&sx, rows, cols)
            ),
            0.0,
            "causal mask fwd"
        );
        let mut d_mgx = backend.dev_alloc_zeros(rows * cols).unwrap();
        backend
            .causal_mask_backward_dev(&d_sgy, &mut d_mgx, rows, cols)
            .unwrap();
        assert_eq!(
            max_abs_diff(
                &dl(&d_mgx, rows * cols),
                &softmax::causal_mask_vjp(rows, cols, &sgy)[0]
            ),
            0.0,
            "causal mask bwd"
        );

        // ── RoPE fwd/bwd over [n_token, n_head, head_dim] ──
        let (n_token, n_head, head_dim, theta) = (8usize, 4usize, 16usize, 10000.0f32);
        let rn = n_token * n_head * head_dim;
        let rx = seeded_uniform(0x83, rn, -1.0, 1.0);
        let rgy = seeded_uniform(0x84, rn, -1.0, 1.0);
        let positions: Vec<usize> = (0..n_token).collect();
        let pos_u32: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
        let d_rx = backend.dev_upload(&rx).unwrap();
        let d_pos = backend.dev_upload_u32(&pos_u32).unwrap();
        let mut d_ry = backend.dev_alloc_zeros(rn).unwrap();
        backend
            .rope_apply_dev(
                &d_rx, &mut d_ry, &d_pos, n_head, head_dim, theta, n_token, 1.0,
            )
            .unwrap();
        let cpu_ry = rope::forward(&rx, &positions, n_head, head_dim, theta);
        assert!(rel(&dl(&d_ry, rn), &cpu_ry) < 1e-4, "rope fwd");
        let d_rgy = backend.dev_upload(&rgy).unwrap();
        let mut d_rgx = backend.dev_alloc_zeros(rn).unwrap();
        backend
            .rope_apply_dev(
                &d_rgy, &mut d_rgx, &d_pos, n_head, head_dim, theta, n_token, -1.0,
            )
            .unwrap();
        let cpu_rgx = rope::vjp(&positions, n_head, head_dim, theta, &rgy);
        assert!(rel(&dl(&d_rgx, rn), &cpu_rgx[0]) < 1e-4, "rope bwd");

        // ── slice_cols + copy_into_cols (bit-exact; slice's vjp = insert into a zeroed buffer) ──
        let (sr, sc, start, len) = (6usize, 12usize, 3usize, 5usize);
        let slx = seeded_uniform(0x85, sr * sc, -1.0, 1.0);
        let d_slx = backend.dev_upload(&slx).unwrap();
        let mut d_sl = backend.dev_alloc_zeros(sr * len).unwrap();
        backend
            .slice_cols_forward_dev(&d_slx, &mut d_sl, sr, sc, start, len)
            .unwrap();
        assert_eq!(
            max_abs_diff(
                &dl(&d_sl, sr * len),
                &shape::slice_cols_forward(&slx, sr, sc, start, len)
            ),
            0.0,
            "slice fwd"
        );
        // slice vjp: scatter the [sr,len] grad back into a zeroed [sr,sc] at [start,start+len).
        let slg = seeded_uniform(0x86, sr * len, -1.0, 1.0);
        let d_slg = backend.dev_upload(&slg).unwrap();
        let mut d_scat = backend.dev_alloc_zeros(sr * sc).unwrap();
        backend
            .copy_into_cols_dev(&d_slg, &mut d_scat, sr, sc, start, len)
            .unwrap();
        assert_eq!(
            max_abs_diff(
                &dl(&d_scat, sr * sc),
                &shape::slice_cols_vjp(sr, sc, start, len, &slg)
            ),
            0.0,
            "slice vjp / copy_into_cols"
        );
        // concat: two inserts reproduce concat_cols_forward.
        let (pa, pb) = (
            seeded_uniform(0x87, sr * 4, -1.0, 1.0),
            seeded_uniform(0x88, sr * 3, -1.0, 1.0),
        );
        let d_pa = backend.dev_upload(&pa).unwrap();
        let d_pb = backend.dev_upload(&pb).unwrap();
        let mut d_cat = backend.dev_alloc_zeros(sr * 7).unwrap();
        backend
            .copy_into_cols_dev(&d_pa, &mut d_cat, sr, 7, 0, 4)
            .unwrap();
        backend
            .copy_into_cols_dev(&d_pb, &mut d_cat, sr, 7, 4, 3)
            .unwrap();
        assert_eq!(
            max_abs_diff(
                &dl(&d_cat, sr * 7),
                &shape::concat_cols_forward(&[&pa, &pb], sr, &[4, 3])
            ),
            0.0,
            "concat via copy_into_cols"
        );

        // ── transpose (bit-exact; its own vjp) ──
        let (tr, tc) = (7usize, 5usize);
        let tx = seeded_uniform(0x89, tr * tc, -1.0, 1.0);
        let d_tx = backend.dev_upload(&tx).unwrap();
        let mut d_ty = backend.dev_alloc_zeros(tr * tc).unwrap();
        backend
            .transpose_forward_dev(&d_tx, &mut d_ty, tr, tc)
            .unwrap();
        assert_eq!(
            max_abs_diff(&dl(&d_ty, tr * tc), &dense::transpose_forward(&tx, tr, tc)),
            0.0,
            "transpose"
        );

        // ── embed gather fwd/bwd (bit-exact) ──
        let (vocab, dim, seq) = (50usize, 24usize, 10usize);
        let w = seeded_uniform(0x8A, vocab * dim, -1.0, 1.0);
        let tokens: Vec<u32> = [3u32, 0, 49, 3, 12, 0, 7, 3, 25, 1].to_vec();
        let tok_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let d_w = backend.dev_upload(&w).unwrap();
        let d_tok = backend.dev_upload_i32(&tok_i32).unwrap();
        let mut d_ey = backend.dev_alloc_zeros(seq * dim).unwrap();
        backend
            .embed_gather_forward_dev(&d_w, &d_tok, &mut d_ey, seq, dim)
            .unwrap();
        assert_eq!(
            max_abs_diff(
                &dl(&d_ey, seq * dim),
                &embed::gather_forward(&w, &tokens, dim)
            ),
            0.0,
            "gather fwd"
        );
        let mut egy = seeded_uniform(0x8B, seq * dim, -1.0, 1.0);
        for d in 0..dim {
            egy[d] = 1e20;
            egy[3 * dim + d] = -1e20;
            egy[7 * dim + d] = 3.25;
        }
        let d_egy = backend.dev_upload(&egy).unwrap();
        let mut d_egw = backend.dev_alloc_zeros(vocab * dim).unwrap();
        backend
            .embed_gather_backward_dev(&d_egy, &d_tok, &mut d_egw, seq, dim, vocab)
            .unwrap();
        assert_eq!(
            max_abs_diff(
                &dl(&d_egw, vocab * dim),
                &embed::gather_vjp(vocab, &tokens, dim, &egy)
            ),
            0.0,
            "gather bwd (repeated tokens accumulate)"
        );
        let mut d_egw_segmented = backend.dev_upload(&vec![7.0; vocab * dim]).unwrap();
        backend
            .embed_gather_backward_segmented_dev(
                &d_egy,
                &tok_i32,
                &mut d_egw_segmented,
                seq,
                dim,
                vocab,
            )
            .unwrap();
        let segmented = dl(&d_egw_segmented, vocab * dim);
        let cpu = embed::gather_vjp(vocab, &tokens, dim, &egy);
        assert!(
            segmented
                .iter()
                .zip(&cpu)
                .all(|(&device, &host)| device.to_bits() == host.to_bits()),
            "segmented gather bwd must preserve exact ascending-position additions and zero untouched rows"
        );

        // ── softmax cross-entropy backward (expf ⇒ 1e-4) ──
        let (xr, xc) = (16usize, 32usize);
        let logits = seeded_uniform(0x8C, xr * xc, -2.0, 2.0);
        let target = softmax::forward(&seeded_uniform(0x8D, xr * xc, -1.0, 1.0), xr, xc); // a distribution
        let d_lg = backend.dev_upload(&logits).unwrap();
        let d_tg = backend.dev_upload(&target).unwrap();
        let mut d_xgl = backend.dev_alloc_zeros(xr * xc).unwrap();
        backend
            .softmax_xent_backward_dev(&d_lg, &d_tg, &mut d_xgl, xr, xc, 1.0 / xr as f32)
            .unwrap();
        let cpu_xgl = loss::softmax_xent_vjp(&logits, &target, xr, xc, &[1.0]);
        assert!(
            rel(&dl(&d_xgl, xr * xc), &cpu_xgl[0]) < 1e-4,
            "softmax_xent bwd"
        );

        eprintln!(
            "0043 P2.4 resident attention glue: softmax/mask/rope/slice/copy_into_cols/transpose/\
             gather/xent all match CPU (copy ops bit-exact; softmax/rope/xent within 1e-4)"
        );
    }

    /// Gate + bench (plan 0043 P2.5a): a whole multi-layer MLP stack (embed → N×[rmsnorm → gate/up →
    /// silu → mul → down → residual] → final rmsnorm → tied lm-head → softmax-xent) assembled on the
    /// `DeviceTape` and run forward+backward entirely on-device, vs the same model on the
    /// tritium-train CPU `Tape`. Exercises the generic device tape, multi-layer chaining, and the
    /// tricky **tied-embedding** grad (accumulated from the input gather AND the output head). The
    /// per-weight grads match within 1e-4; the device step is timed against the CPU tape.
    #[test]
    fn device_tape_mlp_stack_matches_cpu_tape() {
        use std::time::Instant;

        use tritium_train::Tape;
        use tritium_train::ops::softmax;

        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping device_tape_mlp_stack_matches_cpu_tape: no CUDA device ({e})");
                return;
            }
        };
        // Realistic transformer dims (SmolLM2-ish) so the multi-layer matmuls dominate and the
        // compounding device-resident speedup shows. Vocab kept moderate: the tied-embedding backward
        // is O(vocab·seq·dim) (a known hotspot for a later perf pass), so a huge vocab would swamp it.
        let (vocab, dim, ff, seq, layers, eps) =
            (2048usize, 576usize, 1536usize, 48usize, 4usize, 1e-5f32);
        let embd = seeded_uniform(0x100, vocab * dim, -0.1, 0.1);
        let out_norm = seeded_uniform(0x101, dim, 0.5, 1.5);
        let ffn_norm: Vec<Vec<f32>> = (0..layers)
            .map(|l| seeded_uniform(0x110 + l as u64, dim, 0.5, 1.5))
            .collect();
        let wg: Vec<Vec<f32>> = (0..layers)
            .map(|l| seeded_uniform(0x120 + l as u64, ff * dim, -0.1, 0.1))
            .collect();
        let wu: Vec<Vec<f32>> = (0..layers)
            .map(|l| seeded_uniform(0x130 + l as u64, ff * dim, -0.1, 0.1))
            .collect();
        let wd: Vec<Vec<f32>> = (0..layers)
            .map(|l| seeded_uniform(0x140 + l as u64, dim * ff, -0.1, 0.1))
            .collect();
        let tokens: Vec<u32> = (0..seq).map(|i| ((i * 37 + 5) % vocab) as u32).collect();
        let tokens_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let target = softmax::forward(&seeded_uniform(0x150, seq * vocab, -1.0, 1.0), seq, vocab);

        // ── CPU reference ──
        let mut t = Tape::new();
        let c_embd = t.leaf(embd.clone());
        let mut c_hidden = t.embed_gather(c_embd, &tokens, vocab, dim);
        let mut c_w = Vec::new(); // (ffn_norm, wg, wu, wd) leaf ids per layer
        for l in 0..layers {
            let fnid = t.leaf(ffn_norm[l].clone());
            let (gid, uid, did) = (
                t.leaf(wg[l].clone()),
                t.leaf(wu[l].clone()),
                t.leaf(wd[l].clone()),
            );
            let hn = t.rmsnorm(c_hidden, fnid, seq, dim, eps);
            let g = t.dense_matmul(hn, gid, seq, ff, dim);
            let u = t.dense_matmul(hn, uid, seq, ff, dim);
            let ga = t.silu(g);
            let gated = t.mul(ga, u);
            let down = t.dense_matmul(gated, did, seq, dim, ff);
            c_hidden = t.add(c_hidden, down);
            c_w.push((fnid, gid, uid, did));
        }
        let c_on = t.leaf(out_norm.clone());
        let c_fn = t.rmsnorm(c_hidden, c_on, seq, dim, eps);
        let c_logits = t.dense_matmul(c_fn, c_embd, seq, vocab, dim); // tied head
        let c_tg = t.leaf(target.clone());
        let c_loss = t.softmax_xent(c_logits, c_tg, seq, vocab);
        let cg = t.backward(c_loss);

        // ── Device tape ──
        #[allow(clippy::type_complexity)]
        let build_device = || -> Result<(Vec<f32>, Vec<Option<CudaSlice<f32>>>, DeviceBackwardStats, Vec<(usize, usize, usize, usize)>, usize, usize), BackendError> {
            let mut dt = DeviceTape::new_with_checkpoint_policy(
                &backend,
                vocab,
                CheckpointPolicy::SqrtDepth(layers),
            )?;
            let d_embd = dt.leaf(&embd)?;
            let mut d_hidden = dt.embed(d_embd, &tokens_i32, seq, dim, vocab)?;
            let mut d_w = Vec::new();
            for l in 0..layers {
                let fnid = dt.leaf(&ffn_norm[l])?;
                let (gid, uid, did) = (dt.leaf(&wg[l])?, dt.leaf(&wu[l])?, dt.leaf(&wd[l])?);
                let hn = dt.rmsnorm(d_hidden, fnid, seq, dim, eps)?;
                let g = dt.matmul(hn, gid, seq, ff, dim)?;
                let u = dt.matmul(hn, uid, seq, ff, dim)?;
                let ga = dt.silu(g)?;
                let gated = dt.mul(ga, u)?;
                let down = dt.matmul(gated, did, seq, dim, ff)?;
                d_hidden = dt.add(d_hidden, down)?;
                dt.checkpoint_keep(&[d_hidden])?;
                d_w.push((fnid, gid, uid, did));
            }
            let d_on = dt.leaf(&out_norm)?;
            let d_fn = dt.rmsnorm(d_hidden, d_on, seq, dim, eps)?;
            let d_logits = dt.matmul(d_fn, d_embd, seq, vocab, dim)?;
            let logits_h = dt.value(d_logits)?;
            let seed = dt.softmax_xent_grad(d_logits, &target, seq, vocab)?;
            let mut retain = vec![d_embd, d_on];
            retain.extend(
                d_w.iter()
                    .flat_map(|&(ffn_norm, gate, up, down)| [ffn_norm, gate, up, down]),
            );
            let result = dt.backward_retain(d_logits, &seed, &retain)?;
            Ok((logits_h, result.grads, result.stats, d_w, d_embd, d_on))
        };
        let (dev_logits, grads, liveness, d_w, d_embd, d_on) = build_device().expect("device tape");

        // download a device grad buffer
        let dl = |g: &Option<CudaSlice<f32>>, n: usize| {
            let mut h = vec![0.0f32; n];
            backend
                .dev_download(g.as_ref().expect("gradient retained"), &mut h)
                .unwrap();
            h
        };
        let rel = |dev: &[f32], cpu: &[f32]| {
            let range = cpu.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                - cpu.iter().copied().fold(f32::INFINITY, f32::min);
            max_abs_diff(dev, cpu) / range.max(1e-6)
        };

        // Compare logits + every weight grad (incl. the tied embedding + norms).
        let mut worst = rel(&dev_logits, t.value(c_logits));
        let mut worst_name = "logits";
        let check = |dev: &[f32],
                     cpu: &[f32],
                     name: &'static str,
                     worst: &mut f32,
                     wn: &mut &'static str| {
            let r = rel(dev, cpu);
            if r > *worst {
                *worst = r;
                *wn = name;
            }
        };
        check(
            &dl(&grads[d_embd], vocab * dim),
            &cg[c_embd],
            "embd(tied)",
            &mut worst,
            &mut worst_name,
        );
        check(
            &dl(&grads[d_on], dim),
            &cg[c_on],
            "out_norm",
            &mut worst,
            &mut worst_name,
        );
        for l in 0..layers {
            let (dfn, dg, du, dd) = d_w[l];
            let (cfn, cgi, cu, cd) = c_w[l];
            check(
                &dl(&grads[dfn], dim),
                &cg[cfn],
                "ffn_norm",
                &mut worst,
                &mut worst_name,
            );
            check(
                &dl(&grads[dg], ff * dim),
                &cg[cgi],
                "wg",
                &mut worst,
                &mut worst_name,
            );
            check(
                &dl(&grads[du], ff * dim),
                &cg[cu],
                "wu",
                &mut worst,
                &mut worst_name,
            );
            check(
                &dl(&grads[dd], dim * ff),
                &cg[cd],
                "wd",
                &mut worst,
                &mut worst_name,
            );
        }
        assert!(worst < 1e-4, "worst grad rel {worst:.3e} at {worst_name}");
        assert!(
            liveness.peak_persistent_grad_elements < liveness.naive_all_value_grad_elements,
            "full-stack lazy gradient slots must reduce persistent VRAM: {liveness:?}"
        );
        assert!(
            liveness.peak_live_activation_elements < liveness.naive_activation_elements,
            "full-stack checkpointing must reduce activation VRAM: {liveness:?}"
        );
        assert!(liveness.recomputed_ops > 0);

        // Bench: device tape vs CPU tape (full rebuild + fwd + bwd).
        let cpu_step = || {
            let mut t = Tape::new();
            let c_embd = t.leaf(embd.clone());
            let mut c_hidden = t.embed_gather(c_embd, &tokens, vocab, dim);
            for l in 0..layers {
                let fnid = t.leaf(ffn_norm[l].clone());
                let (gid, uid, did) = (
                    t.leaf(wg[l].clone()),
                    t.leaf(wu[l].clone()),
                    t.leaf(wd[l].clone()),
                );
                let hn = t.rmsnorm(c_hidden, fnid, seq, dim, eps);
                let g = t.dense_matmul(hn, gid, seq, ff, dim);
                let u = t.dense_matmul(hn, uid, seq, ff, dim);
                let ga = t.silu(g);
                let gated = t.mul(ga, u);
                let down = t.dense_matmul(gated, did, seq, dim, ff);
                c_hidden = t.add(c_hidden, down);
            }
            let c_on = t.leaf(out_norm.clone());
            let c_fn = t.rmsnorm(c_hidden, c_on, seq, dim, eps);
            let c_logits = t.dense_matmul(c_fn, c_embd, seq, vocab, dim);
            let c_tg = t.leaf(target.clone());
            let c_loss = t.softmax_xent(c_logits, c_tg, seq, vocab);
            let _ = t.backward(c_loss);
        };
        for _ in 0..2 {
            build_device().unwrap();
            cpu_step();
        }
        let iters = 10;
        let t0 = Instant::now();
        for _ in 0..iters {
            build_device().unwrap();
        }
        let dev_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let t1 = Instant::now();
        for _ in 0..iters {
            cpu_step();
        }
        let cpu_ms = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;
        eprintln!(
            "0043 P2.5a DeviceTape MLP stack ({layers} layers, vocab={vocab} dim={dim} ff={ff} seq={seq}): \
             matches CPU tape (worst grad rel {worst:.2e} at {worst_name}). \
             gradient slots peak {}/{} elements (saved {}). \
             activation peak {}/{} elements; retained checkpoints {}; replayed {} ops. \
             fwd+bwd step: device-resident {dev_ms:.2}ms | CPU tape {cpu_ms:.2}ms ({:.1}× faster)",
            liveness.peak_persistent_grad_elements,
            liveness.naive_all_value_grad_elements,
            liveness.saved_grad_elements(),
            liveness.peak_live_activation_elements,
            liveness.naive_activation_elements,
            liveness.retained_checkpoint_elements,
            liveness.recomputed_ops,
            cpu_ms / dev_ms.max(1e-9)
        );
    }

    /// Gate (plan 0043 P2.5b): a full **transformer block** — rmsnorm → GQA attention (q/k/v proj →
    /// RoPE → per-head scaled scores → causal mask → softmax → P·V → concat → o_proj) → residual →
    /// rmsnorm → SwiGLU MLP → residual — assembled on the `DeviceTape` (`attention` + MLP ops) vs the
    /// same block on the tritium-train CPU tape (`nn::attention`). This closes the last op gap: every
    /// piece the real standard-transformer forward+backward uses now runs device-resident and matches
    /// the CPU tape (within 1e-4; rope/softmax carry transcendentals). Grouped-query (n_kv_head <
    /// n_head) is exercised.
    #[test]
    fn device_tape_transformer_block_matches_cpu_tape() {
        use tritium_train::Tape;
        use tritium_train::nn::attention;

        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "skipping device_tape_transformer_block_matches_cpu_tape: no CUDA device ({e})"
                );
                return;
            }
        };
        let (n_embd, n_head, n_kv_head, head_dim, seq, ff, eps, theta) = (
            64usize, 4usize, 2usize, 16usize, 8usize, 128usize, 1e-5f32, 10000.0f32,
        );
        let (qd, kvd) = (n_head * head_dim, n_kv_head * head_dim);
        let hidden = seeded_uniform(0x200, seq * n_embd, -1.0, 1.0);
        let attn_norm = seeded_uniform(0x201, n_embd, 0.5, 1.5);
        let ffn_norm = seeded_uniform(0x202, n_embd, 0.5, 1.5);
        let wq = seeded_uniform(0x203, qd * n_embd, -0.1, 0.1);
        let wk = seeded_uniform(0x204, kvd * n_embd, -0.1, 0.1);
        let wv = seeded_uniform(0x205, kvd * n_embd, -0.1, 0.1);
        let wo = seeded_uniform(0x206, n_embd * qd, -0.1, 0.1);
        let wg = seeded_uniform(0x207, ff * n_embd, -0.1, 0.1);
        let wu = seeded_uniform(0x208, ff * n_embd, -0.1, 0.1);
        let wd = seeded_uniform(0x209, n_embd * ff, -0.1, 0.1);
        let cot = seeded_uniform(0x20A, seq * n_embd, -1.0, 1.0); // dL/dout for L = Σ out·cot

        // ── CPU reference (nn::attention) ──
        let mut t = Tape::new();
        let h = t.leaf(hidden.clone());
        let a_n = t.leaf(attn_norm.clone());
        let xn = t.rmsnorm(h, a_n, seq, n_embd, eps);
        let (cwq, cwk, cwv, cwo) = (
            t.leaf(wq.clone()),
            t.leaf(wk.clone()),
            t.leaf(wv.clone()),
            t.leaf(wo.clone()),
        );
        let attn = attention(
            &mut t, xn, cwq, cwk, cwv, cwo, seq, n_embd, n_head, n_kv_head, head_dim, theta,
        );
        let h1 = t.add(h, attn);
        let f_n = t.leaf(ffn_norm.clone());
        let hn = t.rmsnorm(h1, f_n, seq, n_embd, eps);
        let (cwg, cwu, cwd) = (t.leaf(wg.clone()), t.leaf(wu.clone()), t.leaf(wd.clone()));
        let cg = t.dense_matmul(hn, cwg, seq, ff, n_embd);
        let cu = t.dense_matmul(hn, cwu, seq, ff, n_embd);
        let cga = t.silu(cg);
        let cgated = t.mul(cga, cu);
        let cdown = t.dense_matmul(cgated, cwd, seq, n_embd, ff);
        let cout = t.add(h1, cdown);
        let clc = t.leaf(cot.clone());
        let closs = t.dense_matmul(cout, clc, 1, 1, seq * n_embd);
        let cpu_out = t.value(cout).to_vec();
        let cgrads = t.backward(closs);

        // ── Device tape ──
        let ones_max = n_embd.max(ff).max(qd);
        let mut dt = DeviceTape::new_with_checkpoint_policy(
            &backend,
            ones_max,
            CheckpointPolicy::EveryBlocks(1),
        )
        .unwrap();
        let dh = dt.leaf(&hidden).unwrap();
        let dan = dt.leaf(&attn_norm).unwrap();
        let dxn = dt.rmsnorm(dh, dan, seq, n_embd, eps).unwrap();
        let (dwq, dwk, dwv, dwo) = (
            dt.leaf(&wq).unwrap(),
            dt.leaf(&wk).unwrap(),
            dt.leaf(&wv).unwrap(),
            dt.leaf(&wo).unwrap(),
        );
        let dattn = dt
            .attention(
                dxn, dwq, dwk, dwv, dwo, seq, n_embd, n_head, n_kv_head, head_dim, theta,
            )
            .unwrap();
        let dh1 = dt.add(dh, dattn).unwrap();
        let dfn = dt.leaf(&ffn_norm).unwrap();
        let dhn = dt.rmsnorm(dh1, dfn, seq, n_embd, eps).unwrap();
        let (dwg, dwu, dwd) = (
            dt.leaf(&wg).unwrap(),
            dt.leaf(&wu).unwrap(),
            dt.leaf(&wd).unwrap(),
        );
        let dg = dt.matmul(dhn, dwg, seq, ff, n_embd).unwrap();
        let du = dt.matmul(dhn, dwu, seq, ff, n_embd).unwrap();
        let dga = dt.silu(dg).unwrap();
        let dgated = dt.mul(dga, du).unwrap();
        let ddown = dt.matmul(dgated, dwd, seq, n_embd, ff).unwrap();
        let dout = dt.add(dh1, ddown).unwrap();
        dt.checkpoint_keep(&[dout]).unwrap();
        let dev_out = dt.value(dout).unwrap();
        // L = Σ out·cot ⇒ dL/dout = cot; seed the block output's grad and backprop.
        let seed = backend.dev_upload(&cot).unwrap();
        let retain = [dh, dan, dfn, dwq, dwk, dwv, dwo, dwg, dwu, dwd];
        let result = dt.backward_retain(dout, &seed, &retain).unwrap();
        let grads = result.grads;
        let liveness = result.stats;

        let dl = |g: &Option<CudaSlice<f32>>, n: usize| {
            let mut hbuf = vec![0.0f32; n];
            backend
                .dev_download(g.as_ref().expect("gradient retained"), &mut hbuf)
                .unwrap();
            hbuf
        };
        let rel = |dev: &[f32], cpu: &[f32]| {
            let range = cpu.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                - cpu.iter().copied().fold(f32::INFINITY, f32::min);
            max_abs_diff(dev, cpu) / range.max(1e-6)
        };
        let mut worst = rel(&dev_out, &cpu_out);
        let mut wn = "out";
        let ck = |dev: &[f32],
                  cpu: &[f32],
                  name: &'static str,
                  worst: &mut f32,
                  wn: &mut &'static str| {
            let r = rel(dev, cpu);
            if r > *worst {
                *worst = r;
                *wn = name;
            }
        };
        ck(
            &dl(&grads[dh], seq * n_embd),
            &cgrads[h],
            "hidden",
            &mut worst,
            &mut wn,
        );
        ck(
            &dl(&grads[dan], n_embd),
            &cgrads[a_n],
            "attn_norm",
            &mut worst,
            &mut wn,
        );
        ck(
            &dl(&grads[dfn], n_embd),
            &cgrads[f_n],
            "ffn_norm",
            &mut worst,
            &mut wn,
        );
        ck(
            &dl(&grads[dwq], qd * n_embd),
            &cgrads[cwq],
            "wq",
            &mut worst,
            &mut wn,
        );
        ck(
            &dl(&grads[dwk], kvd * n_embd),
            &cgrads[cwk],
            "wk",
            &mut worst,
            &mut wn,
        );
        ck(
            &dl(&grads[dwv], kvd * n_embd),
            &cgrads[cwv],
            "wv",
            &mut worst,
            &mut wn,
        );
        ck(
            &dl(&grads[dwo], n_embd * qd),
            &cgrads[cwo],
            "wo",
            &mut worst,
            &mut wn,
        );
        ck(
            &dl(&grads[dwg], ff * n_embd),
            &cgrads[cwg],
            "wg",
            &mut worst,
            &mut wn,
        );
        ck(
            &dl(&grads[dwu], ff * n_embd),
            &cgrads[cwu],
            "wu",
            &mut worst,
            &mut wn,
        );
        ck(
            &dl(&grads[dwd], n_embd * ff),
            &cgrads[cwd],
            "wd",
            &mut worst,
            &mut wn,
        );
        assert!(
            worst < 1e-4,
            "transformer block worst grad rel {worst:.3e} at {wn}"
        );
        assert!(
            liveness.peak_persistent_grad_elements < liveness.naive_all_value_grad_elements,
            "transformer-block lazy gradient slots must reduce persistent VRAM: {liveness:?}"
        );
        assert!(
            liveness.recomputed_ops > 0,
            "checkpointed transformer block must replay every current DevOp: {liveness:?}"
        );
        eprintln!(
            "0043 P2.5b DeviceTape transformer block (GQA n_head={n_head} n_kv={n_kv_head} \
             head_dim={head_dim} seq={seq} n_embd={n_embd} ff={ff}): full attention+MLP fwd+bwd \
             matches CPU tape (worst grad rel {worst:.2e} at {wn}); gradient slots peak {}/{} \
             elements (saved {}); replayed {} ops",
            liveness.peak_persistent_grad_elements,
            liveness.naive_all_value_grad_elements,
            liveness.saved_grad_elements(),
            liveness.recomputed_ops
        );
    }

    /// Throughput lever (batching): a **batched** attention+MLP block — `[batch*seq, n_embd]`
    /// hidden, one batched rmsnorm/MLP/residual over all `batch*seq` rows, and per-sequence
    /// `attention` carved out with [`DeviceTape::slice_rows`] then restacked with
    /// [`DeviceTape::concat_rows`] — must equal running the `batch` sequences **separately**.
    /// This is the correctness contract that lets one step process `batch×` the tokens by
    /// saturating the linear GEMMs (M = batch*seq) instead of the seq=32/batch=1 starvation
    /// config. Forward is bit-exact per row (every row's compute is independent); shared-weight
    /// gradients match within 1e-4 (the only difference is the float summation order across the
    /// batch dimension — one grad_w over batch*seq rows vs `batch` grad_w sums added on the host).
    #[test]
    fn device_tape_batched_block_matches_per_sequence() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "skipping device_tape_batched_block_matches_per_sequence: no CUDA device ({e})"
                );
                return;
            }
        };
        let (n_embd, n_head, n_kv_head, head_dim, ff, eps, theta) = (
            64usize, 4usize, 2usize, 16usize, 128usize, 1e-5f32, 10000.0f32,
        );
        let (qd, kvd) = (n_head * head_dim, n_kv_head * head_dim);
        let (batch, seq) = (3usize, 5usize);
        let bs = batch * seq;
        let attn_norm = seeded_uniform(0x301, n_embd, 0.5, 1.5);
        let ffn_norm = seeded_uniform(0x302, n_embd, 0.5, 1.5);
        let wq = seeded_uniform(0x303, qd * n_embd, -0.1, 0.1);
        let wk = seeded_uniform(0x304, kvd * n_embd, -0.1, 0.1);
        let wv = seeded_uniform(0x305, kvd * n_embd, -0.1, 0.1);
        let wo = seeded_uniform(0x306, n_embd * qd, -0.1, 0.1);
        let wg = seeded_uniform(0x307, ff * n_embd, -0.1, 0.1);
        let wu = seeded_uniform(0x308, ff * n_embd, -0.1, 0.1);
        let wd = seeded_uniform(0x309, n_embd * ff, -0.1, 0.1);
        let hidden = seeded_uniform(0x300, bs * n_embd, -1.0, 1.0);
        let cot = seeded_uniform(0x30A, bs * n_embd, -1.0, 1.0); // dL/dout for L = Σ out·cot

        // One attention+MLP block, forward+backward on its own tape, returning the block output
        // and the downloaded grads in order [hidden, attn_norm, ffn_norm, wq, wk, wv, wo, wg, wu, wd].
        #[allow(clippy::too_many_arguments)]
        fn run_block(
            backend: &CudaBackend,
            hidden: &[f32],
            cot: &[f32],
            attn_norm: &[f32],
            ffn_norm: &[f32],
            wq: &[f32],
            wk: &[f32],
            wv: &[f32],
            wo: &[f32],
            wg: &[f32],
            wu: &[f32],
            wd: &[f32],
            rows: usize, // batch*seq for the batched path, seq for one sequence
            attn: impl FnOnce(
                &mut DeviceTape,
                usize,
                usize,
                usize,
                usize,
                usize,
            ) -> Result<usize, BackendError>, // (normed hidden, wq, wk, wv, wo) → attention out id
            n_embd: usize,
            qd: usize,
            kvd: usize,
            ff: usize,
            eps: f32,
        ) -> (Vec<f32>, Vec<Vec<f32>>) {
            let ones_max = (rows * n_embd).max(ff).max(qd);
            let mut dt = DeviceTape::new(backend, ones_max).unwrap();
            let dh = dt.leaf(hidden).unwrap();
            let dan = dt.leaf(attn_norm).unwrap();
            let xn = dt.rmsnorm(dh, dan, rows, n_embd, eps).unwrap();
            let dwq = dt.leaf(wq).unwrap();
            let dwk = dt.leaf(wk).unwrap();
            let dwv = dt.leaf(wv).unwrap();
            let dwo = dt.leaf(wo).unwrap();
            let attn_out = attn(&mut dt, xn, dwq, dwk, dwv, dwo).unwrap();
            let h1 = dt.add(dh, attn_out).unwrap();
            let dfn = dt.leaf(ffn_norm).unwrap();
            let hn = dt.rmsnorm(h1, dfn, rows, n_embd, eps).unwrap();
            let dwg = dt.leaf(wg).unwrap();
            let dwu = dt.leaf(wu).unwrap();
            let dwd = dt.leaf(wd).unwrap();
            let g = dt.matmul(hn, dwg, rows, ff, n_embd).unwrap();
            let u = dt.matmul(hn, dwu, rows, ff, n_embd).unwrap();
            let ga = dt.silu(g).unwrap();
            let gated = dt.mul(ga, u).unwrap();
            let down = dt.matmul(gated, dwd, rows, n_embd, ff).unwrap();
            let out = dt.add(h1, down).unwrap();
            let out_vec = dt.value(out).unwrap();
            let seed = backend.dev_upload(cot).unwrap();
            let retain = [dh, dan, dfn, dwq, dwk, dwv, dwo, dwg, dwu, dwd];
            let res = dt.backward_retain(out, &seed, &retain).unwrap();
            let grads = res.grads;
            let dl = |id: usize, n: usize| {
                let mut b = vec![0.0f32; n];
                backend
                    .dev_download(grads[id].as_ref().expect("gradient retained"), &mut b)
                    .unwrap();
                b
            };
            let gv = vec![
                dl(dh, rows * n_embd),
                dl(dan, n_embd),
                dl(dfn, n_embd),
                dl(dwq, qd * n_embd),
                dl(dwk, kvd * n_embd),
                dl(dwv, kvd * n_embd),
                dl(dwo, n_embd * qd),
                dl(dwg, ff * n_embd),
                dl(dwu, ff * n_embd),
                dl(dwd, n_embd * ff),
            ];
            (out_vec, gv)
        }

        // ── Reference: run each of the `batch` sequences separately, concatenate outputs and
        //    hidden grads (per-sequence), sum the shared-weight grads. ──
        let mut ref_out: Vec<f32> = Vec::with_capacity(bs * n_embd);
        let mut ref_grads: Vec<Vec<f32>> = vec![Vec::new(); 10];
        for b in 0..batch {
            let slice = |v: &[f32]| v[b * seq * n_embd..(b + 1) * seq * n_embd].to_vec();
            let (o, g) = run_block(
                &backend,
                &slice(&hidden),
                &slice(&cot),
                &attn_norm,
                &ffn_norm,
                &wq,
                &wk,
                &wv,
                &wo,
                &wg,
                &wu,
                &wd,
                seq,
                |dt, xn, dwq, dwk, dwv, dwo| {
                    dt.attention(
                        xn, dwq, dwk, dwv, dwo, seq, n_embd, n_head, n_kv_head, head_dim, theta,
                    )
                },
                n_embd,
                qd,
                kvd,
                ff,
                eps,
            );
            ref_out.extend_from_slice(&o);
            ref_grads[0].extend_from_slice(&g[0]); // hidden grad is per-sequence → concatenate
            for i in 1..10 {
                if ref_grads[i].is_empty() {
                    ref_grads[i] = g[i].clone();
                } else {
                    for (acc, v) in ref_grads[i].iter_mut().zip(&g[i]) {
                        *acc += v;
                    }
                }
            }
        }

        // ── Batched: one tape, batched norm/MLP over all bs rows, per-sequence attention. ──
        let (bat_out, bat_grads) = run_block(
            &backend,
            &hidden,
            &cot,
            &attn_norm,
            &ffn_norm,
            &wq,
            &wk,
            &wv,
            &wo,
            &wg,
            &wu,
            &wd,
            bs,
            |dt, xn, dwq, dwk, dwv, dwo| {
                // per-sequence attention on row-blocks (same weight leaves), restacked
                let mut parts = Vec::with_capacity(batch);
                for b in 0..batch {
                    let xn_b = dt.slice_rows(xn, bs, n_embd, b * seq, seq)?;
                    parts.push(dt.attention(
                        xn_b, dwq, dwk, dwv, dwo, seq, n_embd, n_head, n_kv_head, head_dim, theta,
                    )?);
                }
                dt.concat_rows(&parts, &vec![seq; batch], n_embd)
            },
            n_embd,
            qd,
            kvd,
            ff,
            eps,
        );

        let rel = |a: &[f32], b: &[f32]| {
            let range = b.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                - b.iter().copied().fold(f32::INFINITY, f32::min);
            max_abs_diff(a, b) / range.max(1e-6)
        };
        let names = [
            "hidden",
            "attn_norm",
            "ffn_norm",
            "wq",
            "wk",
            "wv",
            "wo",
            "wg",
            "wu",
            "wd",
        ];
        let out_rel = rel(&bat_out, &ref_out);
        assert!(
            out_rel < 1e-6,
            "batched forward must be bit-exact per row vs per-sequence: rel {out_rel:.2e}"
        );
        let mut worst = out_rel;
        let mut wn = "out";
        for i in 0..10 {
            let r = rel(&bat_grads[i], &ref_grads[i]);
            if r > worst {
                worst = r;
                wn = names[i];
            }
            assert!(
                r < 1e-4,
                "batched grad[{}] must match per-sequence within 1e-4: rel {r:.2e}",
                names[i]
            );
        }
        eprintln!(
            "batching: batched attention+MLP block (batch={batch} seq={seq} n_embd={n_embd} \
             ff={ff}) == {batch} separate sequences (fwd rel {out_rel:.2e}, worst grad rel \
             {worst:.2e} at {wn}); one step now processes {bs} tokens vs {seq}"
        );
    }

    /// Throughput measurement (batching lever): time batched fwd+bwd at SmolLM2-135M layer
    /// shapes across batch sizes. The linear GEMMs batch (M = batch*seq) but `attention` is
    /// looped per sequence, so this reveals the honest tradeoff — how much the FLOP-dominant
    /// linears + head gain from saturation vs how much the per-sequence attention launches cost
    /// at high batch. Tokens/s is the bottom line; it includes per-step weight-upload
    /// amortization (fewer, larger steps re-upload the same weights fewer times).
    #[test]
    #[ignore = "throughput bench; run with --ignored --nocapture"]
    fn bench_batched_throughput() {
        use std::time::Instant;

        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping bench_batched_throughput: no CUDA device ({e})");
                return;
            }
        };
        let (n_embd, n_head, n_kv_head, head_dim, ff, eps, theta) = (
            576usize, 9usize, 3usize, 64usize, 1536usize, 1e-5f32, 10000.0f32,
        );
        let (qd, kvd) = (n_head * head_dim, n_kv_head * head_dim);
        let n_layers = 4usize;
        let mkw = |salt: u64, n: usize, lo: f32, hi: f32| seeded_uniform(salt, n, lo, hi);
        let attn_norm: Vec<Vec<f32>> = (0..n_layers)
            .map(|l| mkw(0x400 + l as u64, n_embd, 0.5, 1.5))
            .collect();
        let ffn_norm: Vec<Vec<f32>> = (0..n_layers)
            .map(|l| mkw(0x410 + l as u64, n_embd, 0.5, 1.5))
            .collect();
        let wq: Vec<Vec<f32>> = (0..n_layers)
            .map(|l| mkw(0x420 + l as u64, qd * n_embd, -0.1, 0.1))
            .collect();
        let wk: Vec<Vec<f32>> = (0..n_layers)
            .map(|l| mkw(0x430 + l as u64, kvd * n_embd, -0.1, 0.1))
            .collect();
        let wv: Vec<Vec<f32>> = (0..n_layers)
            .map(|l| mkw(0x440 + l as u64, kvd * n_embd, -0.1, 0.1))
            .collect();
        let wo: Vec<Vec<f32>> = (0..n_layers)
            .map(|l| mkw(0x450 + l as u64, n_embd * qd, -0.1, 0.1))
            .collect();
        let wg: Vec<Vec<f32>> = (0..n_layers)
            .map(|l| mkw(0x460 + l as u64, ff * n_embd, -0.1, 0.1))
            .collect();
        let wu: Vec<Vec<f32>> = (0..n_layers)
            .map(|l| mkw(0x470 + l as u64, ff * n_embd, -0.1, 0.1))
            .collect();
        let wd: Vec<Vec<f32>> = (0..n_layers)
            .map(|l| mkw(0x480 + l as u64, n_embd * ff, -0.1, 0.1))
            .collect();

        let time_step = |batch: usize, seq: usize, iters: usize| -> f64 {
            let bs = batch * seq;
            let ones_max = (bs * n_embd).max(ff).max(qd);
            let hidden0 = mkw(0x4F0, bs * n_embd, -1.0, 1.0);
            let cot = mkw(0x4F1, bs * n_embd, -1.0, 1.0);
            let mut total = 0.0;
            for it in 0..(iters + 2) {
                let t0 = Instant::now();
                let mut dt = DeviceTape::new(&backend, ones_max).unwrap();
                let h0 = dt.leaf(&hidden0).unwrap();
                let mut hidden = h0;
                for l in 0..n_layers {
                    let an = dt.leaf(&attn_norm[l]).unwrap();
                    let xn = dt.rmsnorm(hidden, an, bs, n_embd, eps).unwrap();
                    let dwq = dt.leaf(&wq[l]).unwrap();
                    let dwk = dt.leaf(&wk[l]).unwrap();
                    let dwv = dt.leaf(&wv[l]).unwrap();
                    let dwo = dt.leaf(&wo[l]).unwrap();
                    let mut parts = Vec::with_capacity(batch);
                    for b in 0..batch {
                        let xn_b = dt.slice_rows(xn, bs, n_embd, b * seq, seq).unwrap();
                        parts.push(
                            dt.attention(
                                xn_b, dwq, dwk, dwv, dwo, seq, n_embd, n_head, n_kv_head, head_dim,
                                theta,
                            )
                            .unwrap(),
                        );
                    }
                    let attn = dt.concat_rows(&parts, &vec![seq; batch], n_embd).unwrap();
                    hidden = dt.add(hidden, attn).unwrap();
                    let fnw = dt.leaf(&ffn_norm[l]).unwrap();
                    let hn = dt.rmsnorm(hidden, fnw, bs, n_embd, eps).unwrap();
                    let dwg = dt.leaf(&wg[l]).unwrap();
                    let dwu = dt.leaf(&wu[l]).unwrap();
                    let dwd = dt.leaf(&wd[l]).unwrap();
                    let g = dt.matmul(hn, dwg, bs, ff, n_embd).unwrap();
                    let u = dt.matmul(hn, dwu, bs, ff, n_embd).unwrap();
                    let ga = dt.silu(g).unwrap();
                    let gated = dt.mul(ga, u).unwrap();
                    let down = dt.matmul(gated, dwd, bs, n_embd, ff).unwrap();
                    hidden = dt.add(hidden, down).unwrap();
                }
                let seed = backend.dev_upload(&cot).unwrap();
                let res = dt.backward_retain(hidden, &seed, &[h0]).unwrap();
                // Force GPU completion before stopping the clock (dev_download synchronizes).
                let mut sink = vec![0.0f32; bs * n_embd];
                backend
                    .dev_download(res.grads[h0].as_ref().expect("retained h0 grad"), &mut sink)
                    .unwrap();
                std::hint::black_box(&sink);
                if it >= 2 {
                    total += t0.elapsed().as_secs_f64();
                }
            }
            total / iters as f64
        };

        eprintln!(
            "throughput ({n_layers} layers, SmolLM2-135M shapes: n_embd={n_embd} ff={ff} \
             n_head={n_head}); per-sequence attention, batched linears:"
        );
        eprintln!("── batch sweep at seq=32 (many tiny attentions) ──");
        let base = time_step(1, 32, 5); // s/step at 32 tok, the starvation baseline
        let base_tps = 32.0 / base;
        for &(batch, seq) in &[(1, 32), (8, 32), (16, 32), (32, 32), (64, 32)] {
            let s = time_step(batch, seq, 5);
            let toks = (batch * seq) as f64;
            eprintln!(
                "  batch={batch:3} seq={seq:4} ({:5} tok/step): {:8.2} ms/step  →  {:8.0} tok/s  ({:4.1}× vs base)",
                batch * seq,
                s * 1e3,
                toks / s,
                (toks / s) / base_tps
            );
        }
        eprintln!("── seq sweep at batch=1 (one big attention) ──");
        for &(batch, seq) in &[(1, 32), (1, 128), (1, 512), (1, 1024), (1, 2048)] {
            let s = time_step(batch, seq, 3);
            let toks = (batch * seq) as f64;
            eprintln!(
                "  batch={batch:3} seq={seq:4} ({:5} tok/step): {:8.2} ms/step  →  {:8.0} tok/s  ({:4.1}× vs base)",
                batch * seq,
                s * 1e3,
                toks / s,
                (toks / s) / base_tps
            );
        }
        eprintln!("── fixed 2048-token budget: batch × seq trade ──");
        for &(batch, seq) in &[(64, 32), (16, 128), (4, 512), (2, 1024), (1, 2048)] {
            let s = time_step(batch, seq, 3);
            let toks = (batch * seq) as f64;
            eprintln!(
                "  batch={batch:3} seq={seq:4} ({:5} tok/step): {:8.2} ms/step  →  {:8.0} tok/s  ({:4.1}× vs base)",
                batch * seq,
                s * 1e3,
                toks / s,
                (toks / s) / base_tps
            );
        }
    }

    /// Tensor-core training tier (Lever 1) — smoke gate. cuBLASLt's `Matmul<f32>` uses
    /// `CUBLAS_COMPUTE_32F_FAST_TF32`: tf32 tensor cores, fp32 accumulate, **f32 in/out**
    /// (no cast kernels). This proves the whole approach end-to-end — the dlopened
    /// cuBLASLt engages the 4090's tensor cores, the tf32 result matches the f32 kernel
    /// within tf32's ~10-bit-mantissa tolerance, and it is faster on a realistic MLP-gate
    /// GEMM — before wiring it in as a DeviceTape compute policy. Row-major `Y=X·Wᵀ` maps to
    /// column-major cuBLASLt as `a=W, b=X, transa, m=n, n=m, k=k, lda=ldb=k, ldc=n`.
    #[test]
    fn tf32_tensor_core_matmul_matches_f32() {
        use std::time::Instant;

        use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};

        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping tf32_tensor_core_matmul_matches_f32: no CUDA device ({e})");
                return;
            }
        };
        let blas = match CudaBlasLT::new(backend.stream().clone()) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping tf32_tensor_core_matmul_matches_f32: no cuBLASLt ({e:?})");
                return;
            }
        };
        // MLP-gate shape: Y[m,n] = X[m,k]·Wᵀ, X=[batch*seq, n_embd], W=[ff, n_embd].
        let (m, k, n) = (512usize, 576usize, 1536usize);
        let x = seeded_uniform(0x900, m * k, -1.0, 1.0);
        let w = seeded_uniform(0x901, n * k, -0.1, 0.1);
        let ones = vec![1.0f32; n];

        // f32 reference (the bit-exact --fmad=false kernel), s=ones ⇒ plain X·Wᵀ.
        let d_x = backend.dev_upload(&x).unwrap();
        let d_w = backend.dev_upload(&w).unwrap();
        let d_s = backend.dev_upload(&ones).unwrap();
        let mut d_yf = backend.dev_alloc_zeros(m * n).unwrap();
        backend
            .matmul_forward_dev(&d_x, &d_w, &d_s, GemmShape { m, n, k }, &mut d_yf)
            .unwrap();
        let mut y_f32 = vec![0.0f32; m * n];
        backend.dev_download(&d_yf, &mut y_f32).unwrap();

        // tf32 tensor-core path.
        let cfg = MatmulConfig {
            transa: true,
            transb: false,
            transc: false,
            m: n as u64,
            n: m as u64,
            k: k as u64,
            alpha: 1.0,
            lda: k as i64,
            ldb: k as i64,
            beta: 0.0,
            ldc: n as i64,
            stride_a: None,
            stride_b: None,
            stride_c: None,
            stride_bias: None,
            batch_size: None,
        };
        let mut d_yt = backend.dev_alloc_zeros(m * n).unwrap();
        // SAFETY: shapes/leading-dims match the uploaded row-major buffers per the mapping above.
        #[allow(unsafe_code)]
        unsafe {
            blas.matmul(
                cfg,
                &d_w,
                &d_x,
                &mut d_yt,
                Option::<&CudaSlice<f32>>::None,
                None,
            )
            .unwrap();
        }
        backend.stream().synchronize().unwrap();
        let mut y_tf32 = vec![0.0f32; m * n];
        backend.dev_download(&d_yt, &mut y_tf32).unwrap();

        let range = y_f32.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - y_f32.iter().copied().fold(f32::INFINITY, f32::min);
        let rel = max_abs_diff(&y_tf32, &y_f32) / range.max(1e-6);
        assert!(
            rel < 5e-3,
            "tf32 tensor-core GEMM must match f32 within tf32 tolerance: rel {rel:.2e}"
        );

        // Timing: warm, then time many iterations with one sync at the end.
        let time = |iters: usize, run: &mut dyn FnMut()| -> f64 {
            for _ in 0..3 {
                run();
            }
            backend.stream().synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..iters {
                run();
            }
            backend.stream().synchronize().unwrap();
            t0.elapsed().as_secs_f64() / iters as f64
        };
        let iters = 100;
        let f32_s = time(iters, &mut || {
            backend
                .matmul_forward_dev(&d_x, &d_w, &d_s, GemmShape { m, n, k }, &mut d_yf)
                .unwrap();
        });
        let tf32_s = time(iters, &mut || {
            #[allow(unsafe_code)]
            // SAFETY: the device buffers were allocated for this exact GEMM geometry and remain
            // alive until the stream is synchronized below.
            unsafe {
                blas.matmul(
                    cfg,
                    &d_w,
                    &d_x,
                    &mut d_yt,
                    Option::<&CudaSlice<f32>>::None,
                    None,
                )
                .unwrap();
            }
        });
        eprintln!(
            "tf32 tensor-core GEMM [{m}×{k}]·[{n}×{k}]ᵀ: rel {rel:.2e} vs f32 kernel; \
             f32 {:.3} ms, tf32 {:.3} ms → {:.1}× faster",
            f32_s * 1e3,
            tf32_s * 1e3,
            f32_s / tf32_s
        );
    }

    /// Tensor-core tier — all three training GEMMs. Validates `TensorCoreGemm`'s forward,
    /// grad_a, and grad_w against the f32 `--fmad=false` kernels (with `s = ones`, so the
    /// ternary scale is identity). Distinct m/k/n so a transposed/wrong mapping mismatches
    /// dimensions (cuBLASLt errors) or produces values far above the tf32 tolerance — it
    /// cannot pass by luck. grad_a and grad_w are the mappings the forward smoke test does
    /// not cover.
    #[test]
    fn tensor_core_gemm_all_three_match_f32() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping tensor_core_gemm_all_three_match_f32: no CUDA device ({e})");
                return;
            }
        };
        let tc = match TensorCoreGemm::new(&backend) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("skipping tensor_core_gemm_all_three_match_f32: no cuBLASLt ({e:?})");
                return;
            }
        };
        let (m, k, n) = (384usize, 576usize, 1536usize);
        let shape = GemmShape { m, n, k };
        let x = seeded_uniform(0x910, m * k, -1.0, 1.0);
        let w = seeded_uniform(0x911, n * k, -0.1, 0.1);
        let gy = seeded_uniform(0x912, m * n, -1.0, 1.0);
        let ones = vec![1.0f32; n];
        let d_x = backend.dev_upload(&x).unwrap();
        let d_w = backend.dev_upload(&w).unwrap();
        let d_gy = backend.dev_upload(&gy).unwrap();
        let d_s = backend.dev_upload(&ones).unwrap();

        let dl = |d: &CudaSlice<f32>, len: usize| {
            let mut h = vec![0.0f32; len];
            backend.dev_download(d, &mut h).unwrap();
            h
        };
        let rel = |a: &[f32], b: &[f32]| {
            let range = b.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                - b.iter().copied().fold(f32::INFINITY, f32::min);
            max_abs_diff(a, b) / range.max(1e-6)
        };

        // forward Y = X·Wᵀ
        let mut d_yf = backend.dev_alloc_zeros(m * n).unwrap();
        backend
            .matmul_forward_dev(&d_x, &d_w, &d_s, shape, &mut d_yf)
            .unwrap();
        let mut d_yt = backend.dev_alloc_zeros(m * n).unwrap();
        tc.forward(&d_x, &d_w, shape, &mut d_yt).unwrap();
        backend.stream().synchronize().unwrap();
        let r_fwd = rel(&dl(&d_yt, m * n), &dl(&d_yf, m * n));

        // grad_a = gY·W
        let mut d_gaf = backend.dev_alloc_zeros(m * k).unwrap();
        backend
            .grad_a_dev(&d_gy, &d_w, &d_s, shape, &mut d_gaf)
            .unwrap();
        let mut d_gat = backend.dev_alloc_zeros(m * k).unwrap();
        tc.grad_a(&d_gy, &d_w, shape, &mut d_gat).unwrap();
        backend.stream().synchronize().unwrap();
        let r_ga = rel(&dl(&d_gat, m * k), &dl(&d_gaf, m * k));

        // grad_w = gYᵀ·X
        let mut d_gwf = backend.dev_alloc_zeros(n * k).unwrap();
        backend
            .grad_w_dev(&d_gy, &d_x, &d_s, shape, &mut d_gwf)
            .unwrap();
        let mut d_gwt = backend.dev_alloc_zeros(n * k).unwrap();
        tc.grad_w(&d_gy, &d_x, shape, &mut d_gwt).unwrap();
        backend.stream().synchronize().unwrap();
        let r_gw = rel(&dl(&d_gwt, n * k), &dl(&d_gwf, n * k));

        assert!(r_fwd < 5e-3, "tf32 forward rel {r_fwd:.2e}");
        assert!(r_ga < 5e-3, "tf32 grad_a rel {r_ga:.2e}");
        assert!(r_gw < 5e-3, "tf32 grad_w rel {r_gw:.2e}");
        eprintln!(
            "tensor-core tier all 3 GEMMs match f32 (m={m} k={k} n={n}): \
             fwd {r_fwd:.2e}, grad_a {r_ga:.2e}, grad_w {r_gw:.2e}"
        );
    }

    /// Lever 5: the device `adamw_step_8bit` kernel must match the `tritium_train::Int8AdamW` CPU
    /// oracle. Runs 5 steps of block-wise int8 AdamW on both (n = 300 → two blocks with a ragged
    /// 44-element tail, so the reduction, the tail guard, and cross-step requant are all exercised)
    /// and checks the f32 parameters agree tightly and every int8 moment code agrees within ±1 (the
    /// only slack a boundary-straddling requantization could introduce — the update math is the same
    /// correctly-rounded ops built `--fmad=false`).
    #[test]
    fn adamw_step_8bit_matches_cpu_oracle() {
        use tritium_train::{Int8AdamW, Optimizer};
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping adamw_step_8bit_matches_cpu_oracle: no CUDA device ({e})");
                return;
            }
        };
        let n = 300usize;
        let block = tritium_train::INT8_ADAM_BLOCK;
        let nblocks = n.div_ceil(block);
        let lr = 0.01;
        let opt8 = Int8AdamW::new(lr);
        let opt_f = AdamW::new(lr); // same config the device wrapper consumes

        // Deterministic param + a distinct grad per step (varying absmax dynamics per block).
        let param = seeded_uniform(0x8b17, n, -1.0, 1.0);
        let grads: Vec<Vec<f32>> = (0..5)
            .map(|t| seeded_uniform(0x6a00 + t as u64, n, -0.5, 0.5))
            .collect();

        // CPU oracle.
        let mut cpu_param = param.clone();
        let mut cpu_state = opt8.init_state(n);
        // Device state: param + zeroed int8 moments + zeroed per-block scales.
        let stream = backend.stream();
        let mut d_param = stream.clone_htod(&param).unwrap();
        let mut d_mq = stream.clone_htod(&vec![0i8; n]).unwrap();
        let mut d_vq = stream.clone_htod(&vec![0u8; n]).unwrap();
        let mut d_ms = stream.clone_htod(&vec![0.0f32; nblocks]).unwrap();
        let mut d_vs = stream.clone_htod(&vec![0.0f32; nblocks]).unwrap();

        for (t, g) in grads.iter().enumerate() {
            let step = (t + 1) as u64;
            opt8.step(step, &mut cpu_param, g, &mut cpu_state);
            let d_grad = stream.clone_htod(g).unwrap();
            backend
                .adamw_step_8bit_dev(
                    &mut d_param,
                    &d_grad,
                    &mut d_mq,
                    &mut d_vq,
                    &mut d_ms,
                    &mut d_vs,
                    step,
                    &opt_f,
                )
                .unwrap();
        }

        let mut dev_param = vec![0.0f32; n];
        let mut dev_mq = vec![0i8; n];
        let mut dev_vq = vec![0u8; n];
        stream.memcpy_dtoh(&d_param, &mut dev_param).unwrap();
        stream.memcpy_dtoh(&d_mq, &mut dev_mq).unwrap();
        stream.memcpy_dtoh(&d_vq, &mut dev_vq).unwrap();

        for i in 0..n {
            assert!(
                (dev_param[i] - cpu_param[i]).abs() <= 1e-4 * (1.0 + cpu_param[i].abs()),
                "param[{i}] device {} vs cpu {}",
                dev_param[i],
                cpu_param[i]
            );
            assert!(
                (i32::from(dev_mq[i]) - i32::from(cpu_state.m_q[i])).abs() <= 1,
                "m_q[{i}] device {} vs cpu {}",
                dev_mq[i],
                cpu_state.m_q[i]
            );
            assert!(
                (i32::from(dev_vq[i]) - i32::from(cpu_state.v_q[i])).abs() <= 1,
                "v_q[{i}] device {} vs cpu {}",
                dev_vq[i],
                cpu_state.v_q[i]
            );
        }
        eprintln!(
            "adamw_step_8bit matches the Int8AdamW oracle over 5 steps (n={n}, {nblocks} blocks)"
        );
    }

    /// Lever 5: the device `adamw_step_bf16_master` kernel must be bit-identical to a host AdamW loop
    /// that keeps the master in bf16 and stochastic-rounds each update via `tritium_train::bf16`. Same
    /// dither stream (seed ^ step, index), same `--fmad=false` correctly-rounded ops ⇒ exact match on
    /// the bf16 master codes and the f32 moments over 5 steps.
    #[test]
    fn adamw_step_bf16_master_matches_host() {
        use tritium_train::bf16;
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping adamw_step_bf16_master_matches_host: no CUDA device ({e})");
                return;
            }
        };
        let n = 257usize; // spills a warp boundary; not a power of two
        let seed = 0xBF16_5EEDu64;
        let opt = AdamW::new(0.02);
        let master_f32 = seeded_uniform(0xBF01, n, -2.0, 2.0);
        let master0: Vec<u16> = master_f32
            .iter()
            .map(|&w| bf16::from_f32_nearest(w))
            .collect();
        let grads: Vec<Vec<f32>> = (0..5)
            .map(|t| seeded_uniform(0x7b00 + t as u64, n, -0.4, 0.4))
            .collect();

        // Host reference: bf16 master + f32 moments, AdamW with stochastic-rounded write-back.
        let mut hm = master0.clone();
        let (mut h_m, mut h_v) = (vec![0.0f32; n], vec![0.0f32; n]);
        // Device.
        let stream = backend.stream();
        let mut d_master = stream.clone_htod(&master0).unwrap();
        let mut d_m = stream.clone_htod(&vec![0.0f32; n]).unwrap();
        let mut d_v = stream.clone_htod(&vec![0.0f32; n]).unwrap();

        for (t, g) in grads.iter().enumerate() {
            let step = (t + 1) as u64;
            let exp = step as i32;
            let bc1 = 1.0 - opt.beta1.powi(exp);
            let bc2 = 1.0 - opt.beta2.powi(exp);
            let shrink = 1.0 - opt.lr * opt.weight_decay;
            for i in 0..n {
                let w = bf16::to_f32(hm[i]);
                let mi = opt.beta1 * h_m[i] + (1.0 - opt.beta1) * g[i];
                let vi = opt.beta2 * h_v[i] + (1.0 - opt.beta2) * g[i] * g[i];
                h_m[i] = mi;
                h_v[i] = vi;
                let w_new = w * shrink - opt.lr * (mi / bc1 / ((vi / bc2).sqrt() + opt.eps));
                hm[i] = bf16::from_f32_stochastic(w_new, bf16::dither16(seed ^ step, i));
            }
            let d_grad = stream.clone_htod(g).unwrap();
            backend
                .adamw_step_bf16_master_dev(
                    &mut d_master,
                    &d_grad,
                    &mut d_m,
                    &mut d_v,
                    step,
                    seed,
                    &opt,
                )
                .unwrap();
        }

        let mut dev_master = vec![0u16; n];
        let mut dev_m = vec![0.0f32; n];
        let mut dev_v = vec![0.0f32; n];
        stream.memcpy_dtoh(&d_master, &mut dev_master).unwrap();
        stream.memcpy_dtoh(&d_m, &mut dev_m).unwrap();
        stream.memcpy_dtoh(&d_v, &mut dev_v).unwrap();
        assert_eq!(
            dev_master, hm,
            "bf16 master codes must match the host SR loop exactly"
        );
        assert_eq!(dev_m, h_m, "first moment must match exactly");
        assert_eq!(dev_v, h_v, "second moment must match exactly");
        eprintln!("adamw_step_bf16_master bit-matches the host bf16 SR loop over 5 steps (n={n})");
    }

    /// Tensor-core tier — WHOLE MODEL. Builds a multi-layer transformer (embed → GQA attention
    /// + SwiGLU blocks → tied head) twice on a `DeviceTape` — once on the f32 kernels, once with
    ///   `.with_tensor_core(...)` so every dense GEMM (qkv/o, gate/up/down, head) runs on tf32
    ///   tensor cores — and asserts logits and gradients agree within tf32 tolerance accumulated
    ///   across the full depth. This is the wiring gate: the tier changes speed, not the trained
    ///   result. (Embed/norm/softmax stay f32 in both; only the dense matmuls switch.)
    #[test]
    fn device_tape_tf32_matches_f32_whole_model() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping device_tape_tf32_matches_f32_whole_model: no CUDA ({e})");
                return;
            }
        };
        let tc = match TensorCoreGemm::new(&backend) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("skipping device_tape_tf32_matches_f32_whole_model: no cuBLASLt ({e:?})");
                return;
            }
        };

        const VOCAB: usize = 256;
        const N_EMBD: usize = 64;
        const N_HEAD: usize = 4;
        const N_KV: usize = 2;
        const HEAD_DIM: usize = 16;
        const FF: usize = 128;
        const SEQ: usize = 16;
        const N_LAYERS: usize = 2;
        const EPS: f32 = 1e-5;
        const THETA: f32 = 10000.0;
        let qd = N_HEAD * HEAD_DIM;
        let kvd = N_KV * HEAD_DIM;

        let tokens: Vec<i32> = (0..SEQ).map(|i| ((i * 37 + 11) % VOCAB) as i32).collect();
        let cot = seeded_uniform(0xA00, SEQ * VOCAB, -1.0, 1.0);

        // Build + run the model on `dt` (tf32 if `use_tc`), returning logits and the retained
        // grads for [tied embed, layer0 wq, layer0 gate]. Weights are seeded identically per run.
        let run = |use_tc: bool| -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
            let mut dt =
                DeviceTape::new(&backend, (SEQ * N_EMBD).max(FF).max(qd).max(VOCAB)).unwrap();
            if use_tc {
                dt = dt.with_tensor_core(&tc);
            }
            let embed_w = seeded_uniform(0xA01, VOCAB * N_EMBD, -0.1, 0.1);
            let emb = dt.leaf(&embed_w).unwrap();
            let mut hidden = dt.embed(emb, &tokens, SEQ, N_EMBD, VOCAB).unwrap();
            let mut wq0 = 0usize;
            let mut wg0 = 0usize;
            for l in 0..N_LAYERS {
                let an = dt
                    .leaf(&seeded_uniform(0xB00 + l as u64, N_EMBD, 0.5, 1.5))
                    .unwrap();
                let xn = dt.rmsnorm(hidden, an, SEQ, N_EMBD, EPS).unwrap();
                let wq = dt
                    .leaf(&seeded_uniform(0xB10 + l as u64, qd * N_EMBD, -0.1, 0.1))
                    .unwrap();
                let wk = dt
                    .leaf(&seeded_uniform(0xB20 + l as u64, kvd * N_EMBD, -0.1, 0.1))
                    .unwrap();
                let wv = dt
                    .leaf(&seeded_uniform(0xB30 + l as u64, kvd * N_EMBD, -0.1, 0.1))
                    .unwrap();
                let wo = dt
                    .leaf(&seeded_uniform(0xB40 + l as u64, N_EMBD * qd, -0.1, 0.1))
                    .unwrap();
                let attn = dt
                    .attention(
                        xn, wq, wk, wv, wo, SEQ, N_EMBD, N_HEAD, N_KV, HEAD_DIM, THETA,
                    )
                    .unwrap();
                hidden = dt.add(hidden, attn).unwrap();
                let fnw = dt
                    .leaf(&seeded_uniform(0xB50 + l as u64, N_EMBD, 0.5, 1.5))
                    .unwrap();
                let hn = dt.rmsnorm(hidden, fnw, SEQ, N_EMBD, EPS).unwrap();
                let wg = dt
                    .leaf(&seeded_uniform(0xB60 + l as u64, FF * N_EMBD, -0.1, 0.1))
                    .unwrap();
                let wu = dt
                    .leaf(&seeded_uniform(0xB70 + l as u64, FF * N_EMBD, -0.1, 0.1))
                    .unwrap();
                let wd = dt
                    .leaf(&seeded_uniform(0xB80 + l as u64, N_EMBD * FF, -0.1, 0.1))
                    .unwrap();
                let g = dt.matmul(hn, wg, SEQ, FF, N_EMBD).unwrap();
                let u = dt.matmul(hn, wu, SEQ, FF, N_EMBD).unwrap();
                let ga = dt.silu(g).unwrap();
                let gated = dt.mul(ga, u).unwrap();
                let down = dt.matmul(gated, wd, SEQ, N_EMBD, FF).unwrap();
                hidden = dt.add(hidden, down).unwrap();
                if l == 0 {
                    wq0 = wq;
                    wg0 = wg;
                }
            }
            let onw = dt.leaf(&seeded_uniform(0xA02, N_EMBD, 0.5, 1.5)).unwrap();
            let fnorm = dt.rmsnorm(hidden, onw, SEQ, N_EMBD, EPS).unwrap();
            let logits = dt.matmul(fnorm, emb, SEQ, VOCAB, N_EMBD).unwrap(); // tied head
            let logits_v = dt.value(logits).unwrap();
            let seed = backend.dev_upload(&cot).unwrap();
            let res = dt.backward_retain(logits, &seed, &[emb, wq0, wg0]).unwrap();
            let dl = |id: usize, n: usize| {
                let mut h = vec![0.0f32; n];
                backend
                    .dev_download(res.grads[id].as_ref().expect("grad retained"), &mut h)
                    .unwrap();
                h
            };
            (
                logits_v,
                dl(emb, VOCAB * N_EMBD),
                dl(wq0, qd * N_EMBD),
                dl(wg0, FF * N_EMBD),
            )
        };

        let (lf, gef, gqf, ggf) = run(false);
        let (lt, get, gqt, ggt) = run(true);
        let rel = |a: &[f32], b: &[f32]| {
            let range = b.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                - b.iter().copied().fold(f32::INFINITY, f32::min);
            max_abs_diff(a, b) / range.max(1e-6)
        };
        let r_log = rel(&lt, &lf);
        let r_emb = rel(&get, &gef);
        let r_wq = rel(&gqt, &gqf);
        let r_wg = rel(&ggt, &ggf);
        // Lower-bound sentinel: tf32 truncates the mantissa, so it MUST measurably diverge from
        // f32. If `with_tensor_core` ever silently became a no-op, both runs would be bit-identical
        // (rel = 0) and the upper-bound checks below would still pass — this catches that.
        assert!(
            r_log > 1e-7,
            "tf32 must measurably diverge from f32 — else the tier is silently un-wired \
             (logits rel {r_log:.2e})"
        );
        for (name, r) in [
            ("logits", r_log),
            ("embed grad", r_emb),
            ("wq grad", r_wq),
            ("gate grad", r_wg),
        ] {
            assert!(r < 2e-2, "tf32 whole-model {name} rel {r:.2e} exceeds 2e-2");
        }
        eprintln!(
            "tensor-core tier whole model ({N_LAYERS}L, vocab={VOCAB} n_embd={N_EMBD} seq={SEQ}): \
             tf32 vs f32 — logits {r_log:.2e}, embed grad {r_emb:.2e}, wq grad {r_wq:.2e}, \
             gate grad {r_wg:.2e}"
        );
    }

    /// Tensor-core tier — full-step speedup. Times a whole multi-layer transformer fwd+bwd
    /// (embed → GQA+SwiGLU blocks → tied head) on the f32 kernels vs the tf32 tensor-core tier.
    /// This is the end-to-end payoff — Amdahl-limited by the non-GEMM ops (embed/norm/softmax/
    /// elementwise/launch) that stay f32, so it is below the ~65× per-GEMM figure but is the
    /// number that matters for the training step.
    #[test]
    #[ignore = "throughput bench; run with --ignored --nocapture"]
    fn bench_tf32_whole_model_step() {
        use std::time::Instant;

        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping bench_tf32_whole_model_step: no CUDA ({e})");
                return;
            }
        };
        let tc = TensorCoreGemm::new(&backend).expect("cuBLASLt");
        const VOCAB: usize = 8192;
        const N_EMBD: usize = 576;
        const N_HEAD: usize = 9;
        const N_KV: usize = 3;
        const HEAD_DIM: usize = 64;
        const FF: usize = 1536;
        const N_LAYERS: usize = 8;
        const EPS: f32 = 1e-5;
        const THETA: f32 = 10000.0;
        let qd = N_HEAD * HEAD_DIM;
        let kvd = N_KV * HEAD_DIM;

        let step = |seq: usize, use_tc: bool, iters: usize| -> f64 {
            let tokens: Vec<i32> = (0..seq).map(|i| ((i * 37 + 11) % VOCAB) as i32).collect();
            let cot = seeded_uniform(0xC00, seq * VOCAB, -1.0, 1.0);
            let embed_w = seeded_uniform(0xC01, VOCAB * N_EMBD, -0.1, 0.1);
            let mut total = 0.0;
            for it in 0..(iters + 1) {
                let t0 = Instant::now();
                let mut dt =
                    DeviceTape::new(&backend, (seq * N_EMBD).max(FF).max(qd).max(VOCAB)).unwrap();
                if use_tc {
                    dt = dt.with_tensor_core(&tc);
                }
                let emb = dt.leaf(&embed_w).unwrap();
                let mut hidden = dt.embed(emb, &tokens, seq, N_EMBD, VOCAB).unwrap();
                for l in 0..N_LAYERS {
                    let an = dt
                        .leaf(&seeded_uniform(0xD00 + l as u64, N_EMBD, 0.5, 1.5))
                        .unwrap();
                    let xn = dt.rmsnorm(hidden, an, seq, N_EMBD, EPS).unwrap();
                    let wq = dt
                        .leaf(&seeded_uniform(0xD10 + l as u64, qd * N_EMBD, -0.1, 0.1))
                        .unwrap();
                    let wk = dt
                        .leaf(&seeded_uniform(0xD20 + l as u64, kvd * N_EMBD, -0.1, 0.1))
                        .unwrap();
                    let wv = dt
                        .leaf(&seeded_uniform(0xD30 + l as u64, kvd * N_EMBD, -0.1, 0.1))
                        .unwrap();
                    let wo = dt
                        .leaf(&seeded_uniform(0xD40 + l as u64, N_EMBD * qd, -0.1, 0.1))
                        .unwrap();
                    let attn = dt
                        .attention(
                            xn, wq, wk, wv, wo, seq, N_EMBD, N_HEAD, N_KV, HEAD_DIM, THETA,
                        )
                        .unwrap();
                    hidden = dt.add(hidden, attn).unwrap();
                    let fnw = dt
                        .leaf(&seeded_uniform(0xD50 + l as u64, N_EMBD, 0.5, 1.5))
                        .unwrap();
                    let hn = dt.rmsnorm(hidden, fnw, seq, N_EMBD, EPS).unwrap();
                    let wg = dt
                        .leaf(&seeded_uniform(0xD60 + l as u64, FF * N_EMBD, -0.1, 0.1))
                        .unwrap();
                    let wu = dt
                        .leaf(&seeded_uniform(0xD70 + l as u64, FF * N_EMBD, -0.1, 0.1))
                        .unwrap();
                    let wd = dt
                        .leaf(&seeded_uniform(0xD80 + l as u64, N_EMBD * FF, -0.1, 0.1))
                        .unwrap();
                    let g = dt.matmul(hn, wg, seq, FF, N_EMBD).unwrap();
                    let u = dt.matmul(hn, wu, seq, FF, N_EMBD).unwrap();
                    let ga = dt.silu(g).unwrap();
                    let gated = dt.mul(ga, u).unwrap();
                    let down = dt.matmul(gated, wd, seq, N_EMBD, FF).unwrap();
                    hidden = dt.add(hidden, down).unwrap();
                }
                let onw = dt.leaf(&seeded_uniform(0xC02, N_EMBD, 0.5, 1.5)).unwrap();
                let fnorm = dt.rmsnorm(hidden, onw, seq, N_EMBD, EPS).unwrap();
                let logits = dt.matmul(fnorm, emb, seq, VOCAB, N_EMBD).unwrap();
                let seed = backend.dev_upload(&cot).unwrap();
                let res = dt.backward_retain(logits, &seed, &[emb]).unwrap();
                let mut sink = vec![0.0f32; VOCAB * N_EMBD];
                backend
                    .dev_download(res.grads[emb].as_ref().unwrap(), &mut sink)
                    .unwrap();
                std::hint::black_box(&sink);
                if it >= 1 {
                    total += t0.elapsed().as_secs_f64();
                }
            }
            total / iters as f64
        };

        eprintln!(
            "tf32 tier full-step ({N_LAYERS}L SmolLM2-ish: n_embd={N_EMBD} ff={FF} n_head={N_HEAD} \
             vocab={VOCAB}), fwd+bwd:"
        );
        for &seq in &[128usize, 512] {
            let f = step(seq, false, 3);
            let t = step(seq, true, 3);
            eprintln!(
                "  seq={seq:4}: f32 {:8.1} ms  |  tf32 {:7.2} ms  →  {:5.1}× faster  ({:.0} vs {:.0} tok/s)",
                f * 1e3,
                t * 1e3,
                f / t,
                seq as f64 / f,
                seq as f64 / t,
            );
        }
    }

    /// Gate + payoff (plan 0043 P2.4a): a full **device-resident SwiGLU MLP block** — rmsnorm →
    /// gate/up matmul → silu → mul → down matmul → residual — run forward AND backward entirely on
    /// resident VRAM buffers (activations never leave the device; grads of a value with two
    /// consumers — `hn` from gate+up, `h` from norm+residual — summed via `accumulate`). Uses ONLY
    /// the P2.1–P2.3 ops, so it proves the resident execution model end-to-end on a realistic block
    /// BEFORE the attention ops land. Matches the CPU tape within 1e-4 (silu's expf), and the
    /// device-resident block is timed against the CPU tape — the compounding-speedup proof.
    #[test]
    fn resident_swiglu_block_matches_cpu_tape() {
        use std::time::Instant;

        use tritium_train::Tape;

        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping resident_swiglu_block_matches_cpu_tape: no CUDA device ({e})");
                return;
            }
        };
        let (seq, d, ff, eps) = (64usize, 576usize, 1536usize, 1e-5f32); // SmolLM2 MLP shape
        let h = seeded_uniform(0x71, seq * d, -1.0, 1.0);
        let wn = seeded_uniform(0x72, d, 0.5, 1.5);
        let wg = seeded_uniform(0x73, ff * d, -0.1, 0.1);
        let wu = seeded_uniform(0x74, ff * d, -0.1, 0.1);
        let wd = seeded_uniform(0x75, d * ff, -0.1, 0.1);
        let cot = seeded_uniform(0x76, seq * d, -1.0, 1.0); // dL/dout for L = Σ out·cot

        // ── CPU reference on the tape ──
        let mut t = Tape::new();
        let hid = t.leaf(h.clone());
        let (lwn, lwg, lwu, lwd) = (
            t.leaf(wn.clone()),
            t.leaf(wg.clone()),
            t.leaf(wu.clone()),
            t.leaf(wd.clone()),
        );
        let hn = t.rmsnorm(hid, lwn, seq, d, eps);
        let g = t.dense_matmul(hn, lwg, seq, ff, d);
        let u = t.dense_matmul(hn, lwu, seq, ff, d);
        let ga = t.silu(g);
        let gated = t.mul(ga, u);
        let down = t.dense_matmul(gated, lwd, seq, d, ff);
        let out = t.add(hid, down);
        let lc = t.leaf(cot.clone());
        let loss = t.dense_matmul(out, lc, 1, 1, seq * d);
        let cpu_out = t.value(out).to_vec();
        let gr = t.backward(loss);
        let (c_gh, c_gwn, c_gwg, c_gwu, c_gwd) = (
            gr[hid].clone(),
            gr[lwn].clone(),
            gr[lwg].clone(),
            gr[lwu].clone(),
            gr[lwd].clone(),
        );

        // ── Device-resident block ──
        let sh_gu = GemmShape {
            m: seq,
            n: ff,
            k: d,
        }; // gate/up: [seq,d]·[ff,d]ᵀ
        let sh_d = GemmShape {
            m: seq,
            n: d,
            k: ff,
        }; // down:    [seq,ff]·[d,ff]ᵀ
        let ones = vec![1.0f32; ff.max(d)];
        #[allow(clippy::type_complexity)]
        let resident_block = || -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
            let up = |v: &[f32]| backend.dev_upload(v).unwrap();
            let z = |n: usize| backend.dev_alloc_zeros(n).unwrap();
            let (d_h, d_wn, d_wg, d_wu, d_wd, d_ones) =
                (up(&h), up(&wn), up(&wg), up(&wu), up(&wd), up(&ones));
            // forward
            let mut d_hn = z(seq * d);
            backend
                .rmsnorm_forward_dev(&d_h, &d_wn, &mut d_hn, seq, d, eps)
                .unwrap();
            let mut d_g = z(seq * ff);
            backend
                .matmul_forward_dev(&d_hn, &d_wg, &d_ones, sh_gu, &mut d_g)
                .unwrap();
            let mut d_u = z(seq * ff);
            backend
                .matmul_forward_dev(&d_hn, &d_wu, &d_ones, sh_gu, &mut d_u)
                .unwrap();
            let mut d_ga = z(seq * ff);
            backend.silu_forward_dev(&d_g, &mut d_ga, seq * ff).unwrap();
            let mut d_gated = z(seq * ff);
            backend
                .ew_mul_forward_dev(&d_ga, &d_u, &mut d_gated, seq * ff)
                .unwrap();
            let mut d_down = z(seq * d);
            backend
                .matmul_forward_dev(&d_gated, &d_wd, &d_ones, sh_d, &mut d_down)
                .unwrap();
            let mut d_out = z(seq * d);
            backend
                .ew_add_forward_dev(&d_h, &d_down, &mut d_out, seq * d)
                .unwrap();
            // backward: dL/dout = cot; residual sends it to both h and down.
            let d_gout = up(&cot);
            let mut d_gwd = z(d * ff);
            backend
                .grad_w_dev(&d_gout, &d_gated, &d_ones, sh_d, &mut d_gwd)
                .unwrap();
            let mut d_ggated = z(seq * ff);
            backend
                .grad_a_dev(&d_gout, &d_wd, &d_ones, sh_d, &mut d_ggated)
                .unwrap();
            let mut d_gga = z(seq * ff);
            backend
                .ew_mul_backward_dev(&d_ggated, &d_u, &mut d_gga, seq * ff)
                .unwrap(); // g_ga = g_gated·u
            let mut d_gu = z(seq * ff);
            backend
                .ew_mul_backward_dev(&d_ggated, &d_ga, &mut d_gu, seq * ff)
                .unwrap(); // g_u = g_gated·ga
            let mut d_gg = z(seq * ff);
            backend
                .silu_backward_dev(&d_g, &d_gga, &mut d_gg, seq * ff)
                .unwrap();
            let mut d_gwu = z(ff * d);
            backend
                .grad_w_dev(&d_gu, &d_hn, &d_ones, sh_gu, &mut d_gwu)
                .unwrap();
            let mut d_ghn = z(seq * d); // hn's grad: start from the up path…
            backend
                .grad_a_dev(&d_gu, &d_wu, &d_ones, sh_gu, &mut d_ghn)
                .unwrap();
            let mut d_gwg = z(ff * d);
            backend
                .grad_w_dev(&d_gg, &d_hn, &d_ones, sh_gu, &mut d_gwg)
                .unwrap();
            let mut d_ghn_g = z(seq * d);
            backend
                .grad_a_dev(&d_gg, &d_wg, &d_ones, sh_gu, &mut d_ghn_g)
                .unwrap();
            backend
                .accumulate_dev(&mut d_ghn, &d_ghn_g, seq * d)
                .unwrap(); // …+ the gate path
            let mut d_ghnorm = z(seq * d);
            let mut d_gwn = z(d);
            backend
                .rmsnorm_backward_dev(&d_h, &d_wn, &d_ghn, &mut d_ghnorm, &mut d_gwn, seq, d, eps)
                .unwrap();
            let mut d_gh = up(&cot); // residual grad on h = cot…
            backend
                .accumulate_dev(&mut d_gh, &d_ghnorm, seq * d)
                .unwrap(); // …+ the norm path
            let dl = |dv: &_, n: usize| {
                let mut hv = vec![0.0f32; n];
                backend.dev_download(dv, &mut hv).unwrap();
                hv
            };
            (
                dl(&d_out, seq * d),
                dl(&d_gh, seq * d),
                dl(&d_gwn, d),
                dl(&d_gwg, ff * d),
                dl(&d_gwu, ff * d),
                dl(&d_gwd, d * ff),
            )
        };
        let (o, gh, gwn, gwg, gwu, gwd) = resident_block();

        let rel = |dev: &[f32], cpu: &[f32]| {
            let range = cpu.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                - cpu.iter().copied().fold(f32::INFINITY, f32::min);
            max_abs_diff(dev, cpu) / range.max(1e-6)
        };
        let (r_o, r_gh, r_gwn, r_gwg, r_gwu, r_gwd) = (
            rel(&o, &cpu_out),
            rel(&gh, &c_gh),
            rel(&gwn, &c_gwn),
            rel(&gwg, &c_gwg),
            rel(&gwu, &c_gwu),
            rel(&gwd, &c_gwd),
        );
        for (name, r) in [
            ("out", r_o),
            ("g_h", r_gh),
            ("g_wn", r_gwn),
            ("g_wg", r_gwg),
            ("g_wu", r_gwu),
            ("g_wd", r_gwd),
        ] {
            assert!(r < 1e-4, "swiglu block {name} rel {r:.3e}");
        }

        // Payoff bench: resident device block vs the CPU tape (rebuild + fwd + bwd).
        let cpu_block = || {
            let mut t = Tape::new();
            let hid = t.leaf(h.clone());
            let (lwn, lwg, lwu, lwd) = (
                t.leaf(wn.clone()),
                t.leaf(wg.clone()),
                t.leaf(wu.clone()),
                t.leaf(wd.clone()),
            );
            let hn = t.rmsnorm(hid, lwn, seq, d, eps);
            let g = t.dense_matmul(hn, lwg, seq, ff, d);
            let u = t.dense_matmul(hn, lwu, seq, ff, d);
            let ga = t.silu(g);
            let gated = t.mul(ga, u);
            let down = t.dense_matmul(gated, lwd, seq, d, ff);
            let out = t.add(hid, down);
            let lc = t.leaf(cot.clone());
            let loss = t.dense_matmul(out, lc, 1, 1, seq * d);
            let _ = t.backward(loss);
        };
        for _ in 0..3 {
            resident_block();
            cpu_block();
        }
        let iters = 20;
        let t0 = Instant::now();
        for _ in 0..iters {
            resident_block();
        }
        let dev_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let t1 = Instant::now();
        for _ in 0..iters {
            cpu_block();
        }
        let cpu_ms = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;
        eprintln!(
            "0043 P2.4a resident SwiGLU block (seq={seq} d={d} ff={ff}): matches CPU tape (max rel {:.2e}). \
             fwd+bwd block: device-resident {dev_ms:.2}ms | CPU tape {cpu_ms:.2}ms ({:.1}× faster)",
            [r_o, r_gh, r_gwn, r_gwg, r_gwu, r_gwd]
                .iter()
                .copied()
                .fold(0.0f32, f32::max),
            cpu_ms / dev_ms.max(1e-9)
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
