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

// rope_apply_f32 — bit-exact match of tritium_nn::ops::rope_apply for ONE token
// (the M=1 decode case). NeoX half-rotated: lane j in [0,half) pairs with j+half,
//   out[j]      = a*cos - b*sin ;  out[j+half] = b*cos + a*sin   (a=x[j], b=x[j+half])
//
// The host computes (cos,sin) as the **f64** `sin_cos(pos * theta^(-2j/d))` cast to
// f32 — data-independent of the activations — so they are PRECOMPUTED host-side
// (identically) and passed in as `cos_t`/`sin_t` (`half` entries for this token's
// position). The forward builds a `[ctx, half]` table once at load and indexes it by
// position, so there is no per-token host work. The rotation itself is plain f32
// mul/add/sub with no FMA (matching the host's three separate roundings).
//
// One thread per (head, j) pair; each pair is read+written by a single thread, so
// the in-place update is race-free.
__global__ void rope_apply_f32(float* __restrict__ x,
                               const float* __restrict__ cos_t,  // [half], this pos
                               const float* __restrict__ sin_t,  // [half], this pos
                               const int n_head,
                               const int head_dim) {
  const int half = head_dim >> 1;
  const int total = n_head * half;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  const int head = idx / half;
  const int j = idx - head * half;
  const int base = head * head_dim;

  const float c = cos_t[j];
  const float s = sin_t[j];
  const float a = x[base + j];
  const float b = x[base + j + half];
  // host: a*cos - b*sin  and  b*cos + a*sin (two muls then add/sub, no FMA).
  x[base + j] = __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
  x[base + j + half] = __fadd_rn(__fmul_rn(b, c), __fmul_rn(a, s));
}

// softmax_f32 — row-wise softmax matching tritium_nn::ops::softmax_rows: one thread
// per row, sequential max → exp(x-max) → sum → divide, in the host's order. The
// reductions are sequential (bit-match), and `__fsub_rn`/`__fadd_rn`/`__fdiv_rn`/
// `__fmul_rn` forbid FMA. The ONE op that may not bit-match host f32 is `expf`:
// CUDA's libm differs from the host's glibc `expf` and can disagree by ~1 ULP. If so,
// softmax (and attention) fall to the perplexity+lockstep gate; everything else stays
// bit-exact. In-place on `x` (`[rows, row_len]`).
__global__ void softmax_f32(float* __restrict__ x, const int row_len, const int rows) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  float* r = x + (long long)row * row_len;

  float m = -INFINITY;  // host: NaN-ignoring max via `v > m`
  for (int i = 0; i < row_len; ++i) {
    if (r[i] > m) m = r[i];
  }
  float sum = 0.0f;
  for (int i = 0; i < row_len; ++i) {
    const float e = expf(__fsub_rn(r[i], m));  // host: (v - max).exp()
    r[i] = e;
    sum = __fadd_rn(sum, e);
  }
  const float inv = __fdiv_rn(1.0f, sum);
  for (int i = 0; i < row_len; ++i) {
    r[i] = __fmul_rn(r[i], inv);
  }
}

}  // extern "C"
