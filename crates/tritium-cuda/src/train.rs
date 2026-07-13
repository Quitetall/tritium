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

use cudarc::driver::CudaSlice;
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

/// An op recorded on the [`DeviceTape`] for the reverse pass — input/output value ids + params. The
/// backward reads the output's grad + the saved forward values (`vals`) and accumulates into the
/// inputs' grad buffers (so a value with >1 consumer — residuals, the tied embedding — sums exactly
/// like the CPU tape's `grads[id] += v`).
// Test-exercised (`device_tape_mlp_stack_matches_cpu_tape`) until the distillation loop drives the
// DeviceTape on a real model (plan 0043 P2.5b onward).
#[allow(dead_code)]
enum DevOp {
    Matmul {
        x: usize,
        w: usize,
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
        seq: usize,
        dim: usize,
        vocab: usize,
        out: usize,
    },
    Rope {
        x: usize,
        pos: CudaSlice<i32>,
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

/// An opaque f32 tensor owned by one CUDA context.
///
/// The allocation can be borrowed by [`DeviceTape::leaf_device`] without a
/// device-to-device copy.  Its contents remain private so callers cannot mix
/// contexts or mutate a tensor while a tape borrows it.
pub struct DeviceTensor {
    buf: CudaSlice<f32>,
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
}

enum DeviceValue<'a> {
    Owned(CudaSlice<f32>),
    Borrowed(&'a CudaSlice<f32>),
}

impl DeviceValue<'_> {
    fn as_slice(&self) -> &CudaSlice<f32> {
        match self {
            Self::Owned(buf) => buf,
            Self::Borrowed(buf) => buf,
        }
    }
}

/// Device-resident gradients returned in the exact order requested from
/// [`DeviceTape::xent_backward_device`].
pub struct DeviceGradients {
    bufs: Vec<CudaSlice<f32>>,
}

impl core::fmt::Debug for DeviceGradients {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DeviceGradients")
            .field("count", &self.bufs.len())
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

/// Host description of one trainable SALT weight uploaded into a
/// [`DeviceTrainer`].
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

struct ResidentTrainParam {
    master: DeviceTensor,
    residual: CudaSlice<f32>,
    quantized: DeviceTensor,
    m: CudaSlice<f32>,
    v: CudaSlice<f32>,
    rows: usize,
    cols: usize,
    salt_planes: usize,
    optimizer: AdamW,
}

/// Owns latent masters, SALT reconstructions, and AdamW moments in VRAM across
/// training steps.  The autograd graph remains the separate [`DeviceTape`].
pub struct DeviceTrainer<'a> {
    backend: &'a CudaBackend,
    params: Vec<ResidentTrainParam>,
    quantized_prepared: bool,
}

impl core::fmt::Debug for DeviceTrainer<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DeviceTrainer")
            .field("parameter_count", &self.params.len())
            .finish_non_exhaustive()
    }
}

impl<'a> DeviceTrainer<'a> {
    /// Upload all masters once and allocate resident quantized and optimizer
    /// state buffers.
    pub fn new(
        backend: &'a CudaBackend,
        params: &[DeviceTrainParam<'_>],
    ) -> Result<Self, BackendError> {
        let mut resident = Vec::with_capacity(params.len());
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
            if !(1..=3).contains(&param.salt_planes) {
                return Err(BackendError::InvalidInput(format!(
                    "parameter {index} SALT planes must be in 1..=3"
                )));
            }
            resident.push(ResidentTrainParam {
                master: DeviceTensor::upload(backend, param.master)?,
                residual: backend.dev_alloc_zeros(len)?,
                quantized: DeviceTensor {
                    buf: backend.dev_alloc_zeros(len)?,
                },
                m: backend.dev_alloc_zeros(len)?,
                v: backend.dev_alloc_zeros(len)?,
                rows: param.rows,
                cols: param.cols,
                salt_planes: param.salt_planes,
                optimizer: param.optimizer,
            });
        }
        Ok(Self {
            backend,
            params: resident,
            quantized_prepared: false,
        })
    }

    /// Reconstruct every resident master into its dense f32 SALT tensor.
    pub fn prepare_quantized(&mut self) -> Result<(), BackendError> {
        self.quantized_prepared = false;
        for param in &mut self.params {
            self.backend.salt_quantize_forward_dev(
                &param.master.buf,
                &mut param.residual,
                &mut param.quantized.buf,
                param.rows,
                param.cols,
                param.salt_planes,
            )?;
        }
        self.quantized_prepared = true;
        Ok(())
    }

    /// Borrow a prepared quantized weight for zero-copy insertion into a tape.
    pub fn quantized(&self, index: usize) -> Result<&DeviceTensor, BackendError> {
        if !self.quantized_prepared {
            return Err(BackendError::InvalidInput(
                "quantized weights are stale; call prepare_quantized first".into(),
            ));
        }
        self.params.get(index).map(|p| &p.quantized).ok_or_else(|| {
            BackendError::InvalidInput(format!("parameter index {index} is out of range"))
        })
    }

    /// Apply one 1-based resident AdamW step. Gradients must be in parameter
    /// order, as returned by requesting weight leaf ids in that order.
    pub fn step(&mut self, grads: DeviceGradients, step: u64) -> Result<(), BackendError> {
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
        self.quantized_prepared = false;
        for (param, grad) in self.params.iter_mut().zip(grads.bufs) {
            self.backend.adamw_step_dev(
                &mut param.master.buf,
                &grad,
                &mut param.m,
                &mut param.v,
                step,
                &param.optimizer,
            )?;
        }
        Ok(())
    }

    /// Download one latent master for evaluation or checkpointing.
    pub fn download_master(&self, index: usize) -> Result<Vec<f32>, BackendError> {
        let param = self.params.get(index).ok_or_else(|| {
            BackendError::InvalidInput(format!("parameter index {index} is out of range"))
        })?;
        param.master.download(self.backend)
    }
}

/// A device-resident autograd tape (plan 0043 P2.5): the GPU analogue of [`tritium_train::Tape`].
/// Every activation and gradient lives in VRAM across the whole fwd+bwd step — leaves upload once,
/// results download once, and the recorded ops chain the resident kernels ([`super`]'s `*_dev`
/// methods) with no host round-trips. Forward ops append a device buffer to `vals` and record a
/// [`DevOp`]; [`backward`](Self::backward) replays them in reverse, accumulating grads on-device.
/// A `dense_matmul` uses `s = ones`; the shared `ones` buffer is sized once at construction.
pub struct DeviceTape<'backend, 'leaf> {
    b: &'backend CudaBackend,
    vals: Vec<DeviceValue<'leaf>>,
    lens: Vec<usize>,
    leaves: Vec<bool>,
    ops: Vec<DevOp>,
    ones: CudaSlice<f32>,
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
        let ones = b.dev_upload(&vec![1.0f32; ones_max.max(1)])?;
        Ok(Self {
            b,
            vals: Vec::new(),
            lens: Vec::new(),
            leaves: Vec::new(),
            ops: Vec::new(),
            ones,
        })
    }

    fn push(&mut self, buf: CudaSlice<f32>, len: usize) -> usize {
        let id = self.vals.len();
        self.vals.push(DeviceValue::Owned(buf));
        self.lens.push(len);
        self.leaves.push(false);
        id
    }

    /// Upload a weight/input leaf; returns its value id.
    pub fn leaf(&mut self, host: &[f32]) -> Result<usize, BackendError> {
        let buf = self.b.dev_upload(host)?;
        let id = self.push(buf, host.len());
        self.leaves[id] = true;
        Ok(id)
    }

    /// Borrow an existing resident tensor as a leaf without allocating or
    /// copying it. The tensor must belong to this tape's CUDA context.
    pub fn leaf_device(&mut self, tensor: &'leaf DeviceTensor) -> Result<usize, BackendError> {
        if !self.b.same_context(&tensor.buf) {
            return Err(BackendError::InvalidInput(
                "device tensor belongs to a different CUDA context".into(),
            ));
        }
        let id = self.vals.len();
        self.vals.push(DeviceValue::Borrowed(&tensor.buf));
        self.lens.push(tensor.buf.len());
        self.leaves.push(true);
        Ok(id)
    }

    /// Download a value (e.g. the logits) to host.
    pub fn value(&self, id: usize) -> Result<Vec<f32>, BackendError> {
        let mut h = vec![0.0f32; self.lens[id]];
        self.b.dev_download(self.vals[id].as_slice(), &mut h)?;
        Ok(h)
    }

    /// `g_logits` of `L = mean_row softmax-xent(logits, target)` — the seed for [`backward`](Self::backward)
    /// when the loss is distillation cross-entropy (`grad_out = 1`, so `gscale = 1/rows`).
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
            self.vals[logits].as_slice(),
            target,
            &mut g,
            rows,
            cols,
            1.0 / rows as f32,
        )?;
        Ok(g)
    }

    /// `Y[m,n] = X[m,k]·W[n,k]ᵀ` (fp dense).
    pub fn matmul(
        &mut self,
        x: usize,
        w: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<usize, BackendError> {
        let mut out = self.b.dev_alloc_zeros(m * n)?;
        self.b.matmul_forward_dev(
            self.vals[x].as_slice(),
            self.vals[w].as_slice(),
            &self.ones,
            GemmShape { m, n, k },
            &mut out,
        )?;
        let id = self.push(out, m * n);
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
            self.vals[x].as_slice(),
            self.vals[w].as_slice(),
            &mut out,
            rows,
            cols,
            eps,
        )?;
        let id = self.push(out, rows * cols);
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
        self.b
            .silu_forward_dev(self.vals[x].as_slice(), &mut out, n)?;
        let id = self.push(out, n);
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
        self.b.ew_mul_forward_dev(
            self.vals[a].as_slice(),
            self.vals[b].as_slice(),
            &mut out,
            n,
        )?;
        let id = self.push(out, n);
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
        self.b.ew_add_forward_dev(
            self.vals[a].as_slice(),
            self.vals[b].as_slice(),
            &mut out,
            n,
        )?;
        let id = self.push(out, n);
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
        let d_tok = self.b.dev_upload_i32(tokens)?;
        let mut out = self.b.dev_alloc_zeros(seq * dim)?;
        self.b
            .embed_gather_forward_dev(self.vals[w].as_slice(), &d_tok, &mut out, seq, dim)?;
        let id = self.push(out, seq * dim);
        self.ops.push(DevOp::Embed {
            w,
            tokens: d_tok,
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
        positions: &[i32],
        n_head: usize,
        head_dim: usize,
        theta: f32,
        n_token: usize,
    ) -> Result<usize, BackendError> {
        let n = self.lens[x];
        let pos = self.b.dev_upload_i32(positions)?;
        let mut out = self.b.dev_alloc_zeros(n)?;
        self.b.rope_apply_dev(
            self.vals[x].as_slice(),
            &mut out,
            &pos,
            n_head,
            head_dim,
            theta,
            n_token,
            1.0,
        )?;
        let id = self.push(out, n);
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
            .slice_cols_forward_dev(self.vals[x].as_slice(), &mut out, rows, cols, start, len)?;
        let id = self.push(out, rows * len);
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
            .scale_const_dev(self.vals[x].as_slice(), &mut out, c, n)?;
        let id = self.push(out, n);
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
            .causal_mask_forward_dev(self.vals[x].as_slice(), &mut out, rows, cols)?;
        let id = self.push(out, rows * cols);
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
            .softmax_forward_dev(self.vals[x].as_slice(), &mut out, rows, cols)?;
        let id = self.push(out, rows * cols);
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
            .transpose_forward_dev(self.vals[x].as_slice(), &mut out, rows, cols)?;
        let id = self.push(out, rows * cols);
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
                .copy_into_cols_dev(self.vals[p].as_slice(), &mut out, rows, total, off, len)?;
            off += len;
        }
        let id = self.push(out, rows * total);
        self.ops.push(DevOp::Concat {
            parts: parts.to_vec(),
            rows,
            lens: lens.to_vec(),
            out: id,
        });
        Ok(id)
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
        let qd = n_head * head_dim;
        let kvd = n_kv_head * head_dim;
        let group = n_head / n_kv_head;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let pos: Vec<i32> = (0..seq as i32).collect();

        let q = self.matmul(x, wq, seq, qd, n_embd)?;
        let k = self.matmul(x, wk, seq, kvd, n_embd)?;
        let v = self.matmul(x, wv, seq, kvd, n_embd)?;
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

    /// Reverse pass: seed `grads[seed_id] += seed` (e.g. the xent `g_logits`), then replay the ops in
    /// reverse, accumulating each input's grad on-device. Returns the grad buffer per value id.
    pub(crate) fn backward(
        &self,
        seed_id: usize,
        seed: &CudaSlice<f32>,
    ) -> Result<Vec<CudaSlice<f32>>, BackendError> {
        let mut grads: Vec<CudaSlice<f32>> = self
            .lens
            .iter()
            .map(|&l| self.b.dev_alloc_zeros(l))
            .collect::<Result<_, _>>()?;
        self.b
            .accumulate_dev(&mut grads[seed_id], seed, self.lens[seed_id])?;
        for op in self.ops.iter().rev() {
            match *op {
                DevOp::Matmul { x, w, m, n, k, out } => {
                    let shape = GemmShape { m, n, k };
                    let mut gx = self.b.dev_alloc_zeros(m * k)?;
                    self.b.grad_a_dev(
                        &grads[out],
                        self.vals[w].as_slice(),
                        &self.ones,
                        shape,
                        &mut gx,
                    )?;
                    self.b.accumulate_dev(&mut grads[x], &gx, m * k)?;
                    let mut gw = self.b.dev_alloc_zeros(n * k)?;
                    self.b.grad_w_dev(
                        &grads[out],
                        self.vals[x].as_slice(),
                        &self.ones,
                        shape,
                        &mut gw,
                    )?;
                    self.b.accumulate_dev(&mut grads[w], &gw, n * k)?;
                }
                DevOp::Rmsnorm {
                    x,
                    w,
                    rows,
                    cols,
                    eps,
                    out,
                } => {
                    let mut gx = self.b.dev_alloc_zeros(rows * cols)?;
                    let mut gw = self.b.dev_alloc_zeros(cols)?;
                    self.b.rmsnorm_backward_dev(
                        self.vals[x].as_slice(),
                        self.vals[w].as_slice(),
                        &grads[out],
                        &mut gx,
                        &mut gw,
                        rows,
                        cols,
                        eps,
                    )?;
                    self.b.accumulate_dev(&mut grads[x], &gx, rows * cols)?;
                    self.b.accumulate_dev(&mut grads[w], &gw, cols)?;
                }
                DevOp::Silu { x, n, out } => {
                    let mut gx = self.b.dev_alloc_zeros(n)?;
                    self.b
                        .silu_backward_dev(self.vals[x].as_slice(), &grads[out], &mut gx, n)?;
                    self.b.accumulate_dev(&mut grads[x], &gx, n)?;
                }
                DevOp::Mul { a, b, n, out } => {
                    let mut ga = self.b.dev_alloc_zeros(n)?;
                    self.b
                        .ew_mul_backward_dev(&grads[out], self.vals[b].as_slice(), &mut ga, n)?;
                    self.b.accumulate_dev(&mut grads[a], &ga, n)?;
                    let mut gb = self.b.dev_alloc_zeros(n)?;
                    self.b
                        .ew_mul_backward_dev(&grads[out], self.vals[a].as_slice(), &mut gb, n)?;
                    self.b.accumulate_dev(&mut grads[b], &gb, n)?;
                }
                DevOp::Add { a, b, n, out } => {
                    // out = a + b ⇒ grad flows unchanged to both. Copy grad_out into a temp first
                    // (accumulate into a zeroed buffer) so the two accumulates don't alias grads[out].
                    let mut t = self.b.dev_alloc_zeros(n)?;
                    self.b.accumulate_dev(&mut t, &grads[out], n)?;
                    self.b.accumulate_dev(&mut grads[a], &t, n)?;
                    self.b.accumulate_dev(&mut grads[b], &t, n)?;
                }
                DevOp::Embed {
                    w,
                    ref tokens,
                    seq,
                    dim,
                    vocab,
                    out,
                } => {
                    let mut gw = self.b.dev_alloc_zeros(vocab * dim)?;
                    self.b.embed_gather_backward_dev(
                        &grads[out],
                        tokens,
                        &mut gw,
                        seq,
                        dim,
                        vocab,
                    )?;
                    self.b.accumulate_dev(&mut grads[w], &gw, vocab * dim)?;
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
                        &grads[out],
                        &mut gx,
                        pos,
                        n_head,
                        head_dim,
                        theta,
                        n_token,
                        -1.0,
                    )?;
                    self.b.accumulate_dev(&mut grads[x], &gx, n)?;
                }
                DevOp::SliceCols {
                    x,
                    rows,
                    cols,
                    start,
                    len,
                    out,
                } => {
                    // vjp = scatter the [rows,len] grad back into a zeroed [rows,cols] at [start,+len).
                    let mut gx = self.b.dev_alloc_zeros(rows * cols)?;
                    self.b
                        .copy_into_cols_dev(&grads[out], &mut gx, rows, cols, start, len)?;
                    self.b.accumulate_dev(&mut grads[x], &gx, rows * cols)?;
                }
                DevOp::ScaleConst { x, c, n, out } => {
                    let mut gx = self.b.dev_alloc_zeros(n)?;
                    self.b.scale_const_dev(&grads[out], &mut gx, c, n)?;
                    self.b.accumulate_dev(&mut grads[x], &gx, n)?;
                }
                DevOp::CausalMask { x, rows, cols, out } => {
                    let mut gx = self.b.dev_alloc_zeros(rows * cols)?;
                    self.b
                        .causal_mask_backward_dev(&grads[out], &mut gx, rows, cols)?;
                    self.b.accumulate_dev(&mut grads[x], &gx, rows * cols)?;
                }
                DevOp::Softmax { x, rows, cols, out } => {
                    // vjp uses the saved probabilities p = vals[out].
                    let mut gx = self.b.dev_alloc_zeros(rows * cols)?;
                    self.b.softmax_backward_dev(
                        self.vals[out].as_slice(),
                        &grads[out],
                        &mut gx,
                        rows,
                        cols,
                    )?;
                    self.b.accumulate_dev(&mut grads[x], &gx, rows * cols)?;
                }
                DevOp::Transpose { x, rows, cols, out } => {
                    // vjp = transpose the [cols,rows] output grad back to [rows,cols].
                    let mut gx = self.b.dev_alloc_zeros(rows * cols)?;
                    self.b
                        .transpose_forward_dev(&grads[out], &mut gx, cols, rows)?;
                    self.b.accumulate_dev(&mut grads[x], &gx, rows * cols)?;
                }
                DevOp::Concat {
                    ref parts,
                    rows,
                    ref lens,
                    out,
                } => {
                    // vjp = slice each part's column range back out of the concatenated grad.
                    let total: usize = lens.iter().sum();
                    let mut off = 0;
                    for (&p, &len) in parts.iter().zip(lens) {
                        let mut gp = self.b.dev_alloc_zeros(rows * len)?;
                        self.b.slice_cols_forward_dev(
                            &grads[out],
                            &mut gp,
                            rows,
                            total,
                            off,
                            len,
                        )?;
                        self.b.accumulate_dev(&mut grads[p], &gp, rows * len)?;
                        off += len;
                    }
                }
            }
        }
        Ok(grads)
    }

    /// One distillation-step gradient, host in / host out: seed the softmax-xent loss grad at the
    /// `logits` value, run the whole device-resident backward, and download the gradients for the
    /// requested value ids. Hides the device buffers entirely — the caller (the distillation loop)
    /// works in host `Vec<f32>` and never touches a `CudaSlice`. `want` is typically the weight-leaf
    /// ids; the returned grads are in `want` order.
    pub fn xent_backward(
        &self,
        logits: usize,
        target: &[f32],
        rows: usize,
        cols: usize,
        want: &[usize],
    ) -> Result<Vec<Vec<f32>>, BackendError> {
        let seed = self.softmax_xent_grad(logits, target, rows, cols)?;
        let grads = self.backward(logits, &seed)?;
        want.iter()
            .map(|&id| {
                let mut h = vec![0.0f32; self.lens[id]];
                self.b.dev_download(&grads[id], &mut h)?;
                Ok(h)
            })
            .collect()
    }

    /// One fully resident distillation backward pass. The tape is consumed so
    /// all borrowed parameter tensors are released before a caller mutates
    /// their masters or optimizer state. Requested gradients are moved into the
    /// result without a device-to-device copy and preserve `want` order.
    pub fn xent_backward_device(
        self,
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
        let mut slots: Vec<Option<CudaSlice<f32>>> = self
            .backward(logits, &seed)?
            .into_iter()
            .map(Some)
            .collect();
        let bufs = want
            .iter()
            .map(|&id| slots[id].take().expect("want ids were validated unique"))
            .collect();
        Ok(DeviceGradients { bufs })
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
        let pos_i32: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        let d_rx = backend.dev_upload(&rx).unwrap();
        let d_pos = backend.dev_upload_i32(&pos_i32).unwrap();
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
        let egy = seeded_uniform(0x8B, seq * dim, -1.0, 1.0);
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

        use tritium_train::ops::softmax;
        use tritium_train::Tape;

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
        let build_device = || -> Result<(Vec<f32>, Vec<CudaSlice<f32>>, Vec<(usize, usize, usize, usize)>, usize, usize), BackendError> {
            let mut dt = DeviceTape::new(&backend, vocab)?;
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
                d_w.push((fnid, gid, uid, did));
            }
            let d_on = dt.leaf(&out_norm)?;
            let d_fn = dt.rmsnorm(d_hidden, d_on, seq, dim, eps)?;
            let d_logits = dt.matmul(d_fn, d_embd, seq, vocab, dim)?;
            let logits_h = dt.value(d_logits)?;
            let seed = dt.softmax_xent_grad(d_logits, &target, seq, vocab)?;
            let grads = dt.backward(d_logits, &seed)?;
            Ok((logits_h, grads, d_w, d_embd, d_on))
        };
        let (dev_logits, grads, d_w, d_embd, d_on) = build_device().expect("device tape");

        // download a device grad buffer
        let dl = |g: &CudaSlice<f32>, n: usize| {
            let mut h = vec![0.0f32; n];
            backend.dev_download(g, &mut h).unwrap();
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
             fwd+bwd step: device-resident {dev_ms:.2}ms | CPU tape {cpu_ms:.2}ms ({:.1}× faster)",
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
        use tritium_train::nn::attention;
        use tritium_train::Tape;

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
        let mut dt = DeviceTape::new(&backend, ones_max).unwrap();
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
        let dev_out = dt.value(dout).unwrap();
        // L = Σ out·cot ⇒ dL/dout = cot; seed the block output's grad and backprop.
        let seed = backend.dev_upload(&cot).unwrap();
        let grads = dt.backward(dout, &seed).unwrap();

        let dl = |g: &CudaSlice<f32>, n: usize| {
            let mut hbuf = vec![0.0f32; n];
            backend.dev_download(g, &mut hbuf).unwrap();
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
        eprintln!(
            "0043 P2.5b DeviceTape transformer block (GQA n_head={n_head} n_kv={n_kv_head} \
             head_dim={head_dim} seq={seq} n_embd={n_embd} ff={ff}): full attention+MLP fwd+bwd \
             matches CPU tape (worst grad rel {worst:.2e} at {wn})"
        );
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
