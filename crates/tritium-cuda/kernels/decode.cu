// decode.cu — the on-device M=1 decode kernels for the v0.3.1 device-resident
// forward (ADR 0013). Every kernel here is written to reproduce the host f32 op
// (tritium-nn/src/ops/*) **bit-for-bit**, so the fully on-device forward keeps the
// model's greedy 256/256 parity with transformers — moving the math to the GPU
// must not change a single rounding.
//
// ## The bit-match discipline (load-bearing)
//
// Free-running greedy decode is brutally sensitive: a ~1e-6 logit perturbation
// flips a near-tie token within ~75 steps and cascades. So:
//
//   * **Reductions are sequential, single-thread, in the host's order.** A parallel
//     tree reduction sums in a different order → different f32 rounding → divergence.
//     At M=1 the reduced lengths are tiny (n_embd=2560, ctx<=4096), so a one-thread
//     left-fold costs microseconds and is the correct choice.
//   * **No FMA contraction.** nvcc fuses `a*b+c` into a single `fma` (one rounding)
//     by default (`-fmad=true`), whereas the host does a multiply *then* an add (two
//     roundings). We force the host's behaviour with the round-to-nearest intrinsics
//     `__fmul_rn` / `__fadd_rn` / `__fdiv_rn`, so the result is identical regardless
//     of the `-fmad` flag. `sqrtf` is IEEE correctly-rounded (matches Rust `f32::sqrt`).
//
// Elementwise (non-reducing) work is parallelised freely — without a reduction the
// per-element result is order-independent, so any thread assignment is bit-identical.

#include <cuda_runtime.h>

extern "C" {

// rmsnorm_f32 — bit-exact match of tritium_nn::ops::rmsnorm:
//   mean_sq = (Σ_i x[i]*x[i]) / n        (sequential f32 left-fold, no FMA)
//   inv     = 1 / sqrt(mean_sq + eps)
//   out[i]  = (x[i] * inv) * w[i]
//
// One block. Thread 0 does the sequential sum-of-squares (the only reduction);
// after a barrier every thread writes the embarrassingly-parallel elementwise pass.
__global__ void rmsnorm_f32(const float* __restrict__ x,
                            const float* __restrict__ w,
                            const float eps,
                            const int n,
                            float* __restrict__ out) {
  __shared__ float s_inv;

  if (threadIdx.x == 0) {
    // Sequential sum of squares in the host's i=0..n order, multiply-then-add with
    // no FMA contraction (host: `x.iter().map(|v| v*v).sum::<f32>()`).
    float ss = 0.0f;
    for (int i = 0; i < n; ++i) {
      const float xi = x[i];
      ss = __fadd_rn(ss, __fmul_rn(xi, xi));
    }
    const float mean_sq = __fdiv_rn(ss, (float)n);             // host: sum / n as f32
    const float denom = sqrtf(__fadd_rn(mean_sq, eps));        // host: (mean_sq+eps).sqrt()
    s_inv = __fdiv_rn(1.0f, denom);                            // host: 1.0 / denom
  }
  __syncthreads();

  const float inv = s_inv;
  // out[i] = (x[i] * inv) * w[i] — host left-to-right; two plain f32 muls, no add to
  // contract, so this is order-independent and parallel-safe.
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    out[i] = __fmul_rn(__fmul_rn(x[i], inv), w[i]);
  }
}

}  // extern "C"
