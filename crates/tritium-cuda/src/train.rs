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

use crate::cuda::{CudaBackend, EmbedSegments, TrainingSaltLinear};

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
enum DevOp<'leaf> {
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

impl DevOp<'_> {
    fn output(&self) -> usize {
        match self {
            Self::Matmul { out, .. }
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

/// Opaque device-resident SALT weight for training-time packed execution.
///
/// The handle owns compact TQ2-addressed plane codes plus external f32 per-row
/// scales. It never owns a dense quantized reconstruction. Latent masters and
/// optimizer state remain separate; callers explicitly repack after updating a
/// host-resident master.
pub struct DevicePackedSaltWeight {
    inner: TrainingSaltLinear,
    prepared: bool,
}

impl core::fmt::Debug for DevicePackedSaltWeight {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DevicePackedSaltWeight")
            .field("rows", &self.rows())
            .field("cols", &self.cols())
            .field("planes", &self.planes())
            .field("resident_bytes", &self.resident_bytes())
            .field("prepared", &self.prepared)
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
        Ok(())
    }

    /// Mark codes/scales stale before an out-of-band master update.
    pub fn mark_stale(&mut self) {
        self.prepared = false;
    }

    /// Whether the handle may be inserted into a new device tape.
    #[must_use]
    pub fn is_prepared(&self) -> bool {
        self.prepared
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
        if self.prepared {
            Ok(())
        } else {
            Err(BackendError::InvalidInput(
                "packed SALT weight is stale; repack from the updated master".into(),
            ))
        }
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

struct HostOffloadParam {
    master: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
    rows: usize,
    cols: usize,
    salt_planes: usize,
    optimizer: AdamW,
}

/// Matrix and SALT packing metadata retained with an offloaded parameter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostOffloadParamMetadata {
    pub rows: usize,
    pub cols: usize,
    pub salt_planes: usize,
}

/// Deterministic logical memory accounting for [`HostOffloadTrainer`].
///
/// `peak_optimizer_device_elements` counts only the streamed master and two
/// Adam moments. `resident_input_gradient_elements` is reported separately
/// because [`DeviceGradients`] currently owns every requested gradient on the
/// device before the optimizer step begins.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HostOffloadStats {
    /// Host-resident master + first moment + second moment elements.
    pub host_optimizer_elements: usize,
    /// Size of the largest parameter leaf.
    pub largest_parameter_elements: usize,
    /// Peak streamed optimizer-state elements resident on the device.
    pub peak_optimizer_device_elements: usize,
    /// Device gradient elements supplied to the most recent successful step.
    pub resident_input_gradient_elements: usize,
}

/// Correctness-first CPU-offloaded AdamW state.
///
/// Masters and moments remain in ordinary host vectors between steps. Each
/// parameter uploads only its master, `m`, and `v`, invokes the same device
/// AdamW kernel as [`DeviceTrainer`], downloads the updated state, and releases
/// those temporary allocations before advancing to the next parameter.
///
/// This bounds optimizer-state staging to three times the largest leaf. It does
/// not yet stream gradient production: the input [`DeviceGradients`] remains a
/// resident collection and is accounted separately in [`HostOffloadStats`]. A
/// gradient sink and transfer overlap are deliberately separate follow-ups.
pub struct HostOffloadTrainer<'a> {
    backend: &'a CudaBackend,
    params: Vec<HostOffloadParam>,
    stats: HostOffloadStats,
}

impl core::fmt::Debug for HostOffloadTrainer<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HostOffloadTrainer")
            .field("parameter_count", &self.params.len())
            .field("stats", &self.stats)
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
        let mut host_params = Vec::with_capacity(params.len());
        let mut total_parameter_elements = 0usize;
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
            host_params.push(HostOffloadParam {
                master: param.master.to_vec(),
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
        Ok(Self {
            backend,
            params: host_params,
            stats: HostOffloadStats {
                host_optimizer_elements,
                largest_parameter_elements,
                peak_optimizer_device_elements: 0,
                resident_input_gradient_elements: 0,
            },
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

    /// Borrow one host-resident latent master.
    pub fn master(&self, index: usize) -> Result<&[f32], BackendError> {
        self.params
            .get(index)
            .map(|param| param.master.as_slice())
            .ok_or_else(|| {
                BackendError::InvalidInput(format!("parameter index {index} is out of range"))
            })
    }

    /// Borrow one host-resident pair of Adam moments.
    pub fn moments(&self, index: usize) -> Result<(&[f32], &[f32]), BackendError> {
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

    /// Current logical host/offload memory accounting.
    #[must_use]
    pub fn stats(&self) -> HostOffloadStats {
        self.stats
    }

    /// Apply one 1-based AdamW step while staging one parameter's optimizer
    /// state at a time. All gradient metadata is validated before any master or
    /// moment is changed. A device failure during the update loop can leave
    /// earlier leaves updated; reconstruct or reload the trainer before retrying.
    /// Reported step statistics are meaningful only after a successful return.
    pub fn step(&mut self, grads: DeviceGradients, step: u64) -> Result<(), BackendError> {
        if step == 0 {
            return Err(BackendError::InvalidInput(
                "AdamW step is 1-based; got step 0".into(),
            ));
        }
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

        for (param, grad) in self.params.iter_mut().zip(grads.bufs) {
            let mut d_master = self.backend.dev_upload(&param.master)?;
            let mut d_m = self.backend.dev_upload(&param.m)?;
            let mut d_v = self.backend.dev_upload(&param.v)?;
            let staged_elements = param.master.len().checked_mul(3).ok_or_else(|| {
                BackendError::InvalidInput("staged optimizer elements overflow usize".into())
            })?;
            self.stats.peak_optimizer_device_elements = self
                .stats
                .peak_optimizer_device_elements
                .max(staged_elements);
            self.backend.adamw_step_dev(
                &mut d_master,
                &grad,
                &mut d_m,
                &mut d_v,
                step,
                &param.optimizer,
            )?;
            self.backend.dev_download(&d_master, &mut param.master)?;
            self.backend.dev_download(&d_m, &mut param.m)?;
            self.backend.dev_download(&d_v, &mut param.v)?;
        }
        self.stats.resident_input_gradient_elements = resident_input_gradient_elements;
        Ok(())
    }
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
        if !self.b.same_context(&tensor.buf) {
            return Err(BackendError::InvalidInput(
                "device tensor belongs to a different CUDA context".into(),
            ));
        }
        let id = self.vals.len();
        self.vals.push(Some(DeviceValue::Borrowed(&tensor.buf)));
        self.lens.push(tensor.buf.len());
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
            self.value_slice(x)?,
            self.value_slice(w)?,
            &self.ones,
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
        self.b
            .training_salt_forward(self.value_slice(x)?, &weight.inner, m, &mut out)?;
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
            DevOp::Matmul {
                x,
                w,
                m,
                n,
                k,
                out: _,
            } => {
                let mut output = self.b.dev_alloc_zeros(m * n)?;
                self.b.matmul_forward_dev(
                    self.value_slice(x)?,
                    self.value_slice(w)?,
                    &self.ones,
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
                self.b.training_salt_forward(
                    self.value_slice(x)?,
                    &weight.inner,
                    m,
                    &mut output,
                )?;
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

    /// Reverse pass with lazy gradient slots. `retain_ids` are returned in
    /// their value-id slots; all other gradients are released once their
    /// producing op has consumed them.
    fn backward_retain(
        &mut self,
        seed_id: usize,
        seed: &CudaSlice<f32>,
        retain_ids: &[usize],
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
                let Some(grad_out) = grads[out_id].take() else {
                    // This op is not on a path from the seed, but its activation
                    // is still dead once reverse replay reaches its producer.
                    self.evict_activation(out_id)?;
                    continue;
                };
                match *op {
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
                        self.b.grad_a_dev(
                            &grad_out,
                            self.value_slice(w)?,
                            &self.ones,
                            shape,
                            &mut gx,
                        )?;
                        self.accumulate_grad_slot(
                            &mut grads,
                            &retain,
                            x,
                            &gx,
                            &mut live_elements,
                            &mut peak_persistent_grad_elements,
                        )?;
                        let mut gw = self.b.dev_alloc_zeros(n * k)?;
                        self.b.grad_w_dev(
                            &grad_out,
                            self.value_slice(x)?,
                            &self.ones,
                            shape,
                            &mut gw,
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
                        self.b
                            .training_salt_grad_a(&grad_out, &weight.inner, m, &mut gx)?;
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
            }
        }

        // Preserve the old API's zero-gradient behavior for requested values
        // that are disconnected from the seed.
        for &id in retain_ids {
            if grads[id].is_none() {
                grads[id] = Some(self.b.dev_alloc_zeros(self.lens[id])?);
                live_elements = live_elements.saturating_add(self.lens[id]);
                peak_persistent_grad_elements = peak_persistent_grad_elements.max(live_elements);
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
        debug_assert_eq!(
            live_elements,
            retain
                .iter()
                .zip(&self.lens)
                .filter(|(keep, _)| **keep)
                .fold(0usize, |total, (_, &len)| total.saturating_add(len))
        );
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
    ) -> (Vec<f32>, Vec<f32>, DeviceBackwardStats) {
        let seq = tokens.len();
        let mut tape =
            DeviceTape::new_with_checkpoint_policy(backend, packed.rows(), policy).unwrap();
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
        let keep =
            run_packed_checkpoint_graph(&backend, &packed, &tokens, CheckpointPolicy::KeepAll);
        let replay = run_packed_checkpoint_graph(
            &backend,
            &packed,
            &tokens,
            CheckpointPolicy::EveryBlocks(1),
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
        assert_eq!(stats.peak_optimizer_device_elements, largest * 3);
        assert_eq!(stats.resident_input_gradient_elements, total_elements);
        assert!(
            stats.peak_optimizer_device_elements < stats.host_optimizer_elements,
            "offload staging must be leaf-bounded, not model-bounded: {stats:?}"
        );
        eprintln!(
            "0027 Track E host AdamW offload: optimizer staging peak {}/{} elements; largest leaf \
             {}; resident gradient input {} elements",
            stats.peak_optimizer_device_elements,
            stats.host_optimizer_elements,
            stats.largest_parameter_elements,
            stats.resident_input_gradient_elements
        );
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
        assert_eq!(trainer.stats().peak_optimizer_device_elements, 0);
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
