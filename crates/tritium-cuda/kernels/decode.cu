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
//   * **Rounded f32 reductions fold in a documented canonical order shared with
//     the host.** RMSNorm sums use the ADR 0018 tree order (256 strided slots +
//     pow-2 tree — implemented identically by `tritium_nn::ops::rmsnorm`, so
//     cross-backend bits match by construction). Sums with no host-side tree
//     counterpart yet (attention softmax) remain sequential in the host's
//     order on a single thread.
//   * **No FMA contraction.** nvcc fuses `a*b+c` into a single `fma` (one rounding)
//     by default (`-fmad=true`), whereas the host does a multiply *then* an add (two
//     roundings). We force the host's behaviour with the round-to-nearest intrinsics
//     `__fmul_rn` / `__fadd_rn` / `__fdiv_rn`, so the result is identical regardless
//     of the `-fmad` flag. `sqrtf` is IEEE correctly-rounded (matches Rust `f32::sqrt`).
//
// Elementwise (non-reducing) work is parallelised freely — without a reduction the
// per-element result is order-independent, so any thread assignment is bit-identical.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// exp_f32 — the softmax exponential, computed in **double** then rounded to float.
// The host op is Rust `f32::exp`, which lowers to glibc `expf` (correctly rounded to
// ≤0.5 ULP). CUDA's `expf` carries ~2 ULP of error, which is enough to flip a greedy
// near-tie within a few dozen tokens. Computing `exp` in f64 (CUDA's `exp` is ≤1 ULP
// of the true double value) and rounding once to f32 yields the correctly-rounded
// float result — bit-identical to glibc `expf` for essentially every input — so the
// device softmax/attention match the host instead of drifting. f64 exp is slower than
// `expf`, but at M=1 decode the reduced lengths (n_embd, ctx) make it negligible.
__device__ __forceinline__ float exp_f32(float x) {
  return __double2float_rn(exp((double)x));
}

extern "C" {

// rmsnorm_f32 — bit-exact match of tritium_nn::ops::rmsnorm:
//   mean_sq = (Σ_i x[i]*x[i]) / n        (canonical tree order, ADR 0018, no FMA)
//   inv     = 1 / sqrt(mean_sq + eps)
//   out[i]  = (x[i] * inv) * w[i]
//
// One 256-thread block. The sum-of-squares folds in the canonical cross-backend
// order (slot t = thread t strided fold + pow-2 tree) that the host implements
// identically; the elementwise pass is embarrassingly parallel.
__global__ void rmsnorm_f32(const float* __restrict__ x,
                            const float* __restrict__ w,
                            const float eps,
                            const int n,
                            float* __restrict__ out) {
  __shared__ float s_inv;
  __shared__ float s_red[256];

  // Canonical tree sum (ADR 0018): slot t = threadIdx.x folds x[t],
  // x[t+256], ... in ascending order, then a power-of-two tree combines
  // the slots — the documented cross-backend order tritium_nn::ops::rmsnorm
  // implements identically on the host. Requires blockDim.x == 256 (all
  // launches comply). Replaces the sequential thread-0 fold, which was the
  // measured decode bottleneck (a 4-cycle-per-element latency chain).
  {
    float part = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      const float xi = x[i];
      part = __fadd_rn(part, __fmul_rn(xi, xi));
    }
    s_red[threadIdx.x] = part;
    __syncthreads();
    // Levels 128 and 64 in shared, then warp 0 finishes in registers: the
    // shuffle pairing (t, t+off) IS the canonical tree's pairing, so the DAG
    // (and every rounding) is identical — just 6 fewer block barriers.
    for (int off = 128; off >= 64; off >>= 1) {
      if (threadIdx.x < off) {
        s_red[threadIdx.x] = __fadd_rn(s_red[threadIdx.x], s_red[threadIdx.x + off]);
      }
      __syncthreads();
    }
    if (threadIdx.x < 32) {
      float v = __fadd_rn(s_red[threadIdx.x], s_red[threadIdx.x + 32]);  // level 32
      for (int off = 16; off > 0; off >>= 1) {
        v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, off));
      }
      if (threadIdx.x == 0) {
        const float mean_sq = __fdiv_rn(v, (float)n);
        const float denom = sqrtf(__fadd_rn(mean_sq, eps));
        s_inv = __fdiv_rn(1.0f, denom);
      }
    }
  }
  __syncthreads();

  const float inv = s_inv;
  // out[i] = (x[i] * inv) * w[i] — host left-to-right; two plain f32 muls, no add to
  // contract, so this is order-independent and parallel-safe.
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    out[i] = __fmul_rn(__fmul_rn(x[i], inv), w[i]);
  }
}

// rmsnorm_shared_f32 — **bit-identical** to rmsnorm_f32 (same canonical tree
// order, ADR 0018), reading from a COALESCED shared stage of `x` instead of
// global. Dynamic shared = `n * 4` bytes (the launch sets it); n_ff=6912 →
// 27 KiB < 48 KiB. The elementwise pass reads the staged `s_x`.
__global__ void rmsnorm_shared_f32(const float* __restrict__ x,
                                   const float* __restrict__ w,
                                   const float eps,
                                   const int n,
                                   float* __restrict__ out) {
  extern __shared__ float s_x[];
  __shared__ float s_inv;
  __shared__ float s_red[256];
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    s_x[i] = x[i];  // coalesced stage into shared
  }
  __syncthreads();
  // Canonical tree sum (ADR 0018): slot t = threadIdx.x folds s_x[t],
  // s_x[t+256], ... in ascending order, then a power-of-two tree combines
  // the slots — the documented cross-backend order tritium_nn::ops::rmsnorm
  // implements identically on the host. Requires blockDim.x == 256 (all
  // launches comply). Replaces the sequential thread-0 fold, which was the
  // measured decode bottleneck (a 4-cycle-per-element latency chain).
  {
    float part = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      const float xi = s_x[i];
      part = __fadd_rn(part, __fmul_rn(xi, xi));
    }
    s_red[threadIdx.x] = part;
    __syncthreads();
    // Levels 128 and 64 in shared, then warp 0 finishes in registers: the
    // shuffle pairing (t, t+off) IS the canonical tree's pairing, so the DAG
    // (and every rounding) is identical — just 6 fewer block barriers.
    for (int off = 128; off >= 64; off >>= 1) {
      if (threadIdx.x < off) {
        s_red[threadIdx.x] = __fadd_rn(s_red[threadIdx.x], s_red[threadIdx.x + off]);
      }
      __syncthreads();
    }
    if (threadIdx.x < 32) {
      float v = __fadd_rn(s_red[threadIdx.x], s_red[threadIdx.x + 32]);  // level 32
      for (int off = 16; off > 0; off >>= 1) {
        v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, off));
      }
      if (threadIdx.x == 0) {
        const float mean_sq = __fdiv_rn(v, (float)n);
        const float denom = sqrtf(__fadd_rn(mean_sq, eps));
        s_inv = __fdiv_rn(1.0f, denom);
      }
    }
  }
  __syncthreads();
  const float inv = s_inv;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    out[i] = __fmul_rn(__fmul_rn(s_x[i], inv), w[i]);
  }
}

// rmsnorm_quant_f32 — fused RMSNorm + int8 activation quant (v0.7.0 opt).
// Combines rmsnorm_shared_f32 + act_quant_tiled_f32 into one kernel:
//   1. Stage x into shared memory (coalesced load)
//   2. Thread 0: sequential sum-of-squares → inv = 1/sqrt(mean_sq + eps)
//   3. All threads: compute y[i] = x[i] * inv * w[i] (in shared), track absmax
//   4. Block-wide absmax reduction → gamma
//   5. All threads: q_out[i] = clamp(round(y[i] * 127/gamma), -128, 127)
//
// Eliminates one global read + one global write per norm (the rmsnorm output
// stays in shared memory). Saves 1 kernel launch per call (4 per layer).
// Dynamic shared = n * 4 bytes (the launch must set it).
//
// Bit-match contract: the rmsnorm output y[i] is computed identically to
// rmsnorm_shared_f32 (same sum order, same FMA discipline). The absmax uses
// a tree reduction (associative+commutative for max) → same gamma. The quant
// uses rintf + clamp → same as act_quant_tiled_f32.
__global__ void rmsnorm_quant_f32(const float* __restrict__ x,
                                   const float* __restrict__ w,
                                   const float eps,
                                   const int n,
                                   float* __restrict__ q_out,
                                   float* __restrict__ act_scale) {
  extern __shared__ float s_x[];
  __shared__ float s_inv;
  __shared__ float s_gamma;
  __shared__ float s_red[1024];

  // 1. Stage x into shared memory (coalesced).
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    s_x[i] = x[i];
  }
  __syncthreads();

  // Canonical tree sum (ADR 0018): slot t = threadIdx.x folds s_x[t],
  // s_x[t+256], ... in ascending order, then a power-of-two tree combines
  // the slots — the documented cross-backend order tritium_nn::ops::rmsnorm
  // implements identically on the host. Requires blockDim.x == 256 (all
  // launches comply). Replaces the sequential thread-0 fold, which was the
  // measured decode bottleneck (a 4-cycle-per-element latency chain).
  {
    float part = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      const float xi = s_x[i];
      part = __fadd_rn(part, __fmul_rn(xi, xi));
    }
    s_red[threadIdx.x] = part;
    __syncthreads();
    // Levels 128 and 64 in shared, then warp 0 finishes in registers: the
    // shuffle pairing (t, t+off) IS the canonical tree's pairing, so the DAG
    // (and every rounding) is identical — just 6 fewer block barriers.
    for (int off = 128; off >= 64; off >>= 1) {
      if (threadIdx.x < off) {
        s_red[threadIdx.x] = __fadd_rn(s_red[threadIdx.x], s_red[threadIdx.x + off]);
      }
      __syncthreads();
    }
    if (threadIdx.x < 32) {
      float v = __fadd_rn(s_red[threadIdx.x], s_red[threadIdx.x + 32]);  // level 32
      for (int off = 16; off > 0; off >>= 1) {
        v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, off));
      }
      if (threadIdx.x == 0) {
        const float mean_sq = __fdiv_rn(v, (float)n);
        const float denom = sqrtf(__fadd_rn(mean_sq, eps));
        s_inv = __fdiv_rn(1.0f, denom);
      }
    }
  }
  __syncthreads();

  // 3. Compute rmsnorm output y[i] = x[i] * inv * w[i] in shared memory,
  //    and track per-thread absmax for the quant step.
  const float inv = s_inv;
  float local_max = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float yi = __fmul_rn(__fmul_rn(s_x[i], inv), w[i]);
    s_x[i] = yi;  // overwrite x with rmsnorm output (reuses shared mem)
    const float a = fabsf(yi);
    if (a > local_max) local_max = a;
  }
  __syncthreads();

  // 4. Block-wide absmax reduction. Reduce in a SEPARATE static shared buffer
  //    (`s_red`), NOT in `s_x`: s_x[0..n) holds the rmsnorm output y[i] that step 5
  //    must quantize. The previous version reused `s_x[threadIdx.x]` as reduction
  //    scratch, which clobbered the first `blockDim.x` outputs before they were
  //    quantized → garbage activations. `blockDim.x` ≤ 1024 (CUDA max) and is a power
  //    of two (matches act_quant_tiled_f32's reduction).
  s_red[threadIdx.x] = local_max;
  __syncthreads();
  // max is exact under any order; same shared-then-warp-shuffle shape as the
  // sum tree above (6 fewer barriers).
  for (int off = 128; off >= 64; off >>= 1) {
    if (threadIdx.x < off) {
      s_red[threadIdx.x] = fmaxf(s_red[threadIdx.x], s_red[threadIdx.x + off]);
    }
    __syncthreads();
  }
  if (threadIdx.x < 32) {
    float v = fmaxf(s_red[threadIdx.x], s_red[threadIdx.x + 32]);
    for (int off = 16; off > 0; off >>= 1) {
      v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    }
    if (threadIdx.x == 0) {
      s_gamma = v;
      *act_scale = (v == 0.0f) ? 0.0f : __fdiv_rn(v, 127.0f);
    }
  }
  __syncthreads();

  // 5. Quantize: q_out[i] = clamp(round(y[i] * 127/gamma), -128, 127).
  const float gamma = s_gamma;
  if (gamma == 0.0f) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) q_out[i] = 0.0f;
    return;
  }
  const float s = __fdiv_rn(127.0f, gamma);
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float scaled = rintf(__fmul_rn(s_x[i], s));
    q_out[i] = fminf(fmaxf(scaled, -128.0f), 127.0f);
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
// `__fmul_rn` forbid FMA. The exponential uses `exp_f32` (f64 `exp` rounded once to
// f32) so it matches glibc `expf` — the host op — bit-for-bit on essentially every
// input, rather than drifting like CUDA's ~2-ULP `expf`. In-place on `x`
// (`[rows, row_len]`).
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
    const float e = exp_f32(__fsub_rn(r[i], m));  // host: (v - max).exp()
    r[i] = e;
    sum = __fadd_rn(sum, e);
  }
  const float inv = __fdiv_rn(1.0f, sum);
  for (int i = 0; i < row_len; ++i) {
    r[i] = __fmul_rn(r[i], inv);
  }
}

// residual_add_f32 — x[i] += y[i]. A single f32 add per element (no rounding choice,
// no FMA), exact match of the host residual add. Embarrassingly parallel.
__global__ void residual_add_f32(float* __restrict__ x, const float* __restrict__ y,
                                 const int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = __fadd_rn(x[i], y[i]);
  }
}

// embedding_gather_f32 — out[i] = table[tok*n_embd + i]. A pure copy of one row of
// the embedding table (exact match of the host gather). One thread per element.
__global__ void embedding_gather_f32(const float* __restrict__ table, const int tok,
                                     const int n_embd, float* __restrict__ out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_embd) {
    out[i] = table[(long long)tok * n_embd + i];
  }
}

// lm_head_f32 — tied LM head: logits[v] = Σ_k h[k] * embd[v*n_embd + k]. Bit-matches
// the host's sequential `acc += h[k]*row[k]` (mul then add, no FMA) by accumulating
// in the same k=0..n_embd order. One thread per vocab row; the reduction is sequential
// within the thread (parallel across vocab).
__global__ void lm_head_f32(const float* __restrict__ h, const float* __restrict__ embd,
                            const int n_embd, const int vocab, float* __restrict__ logits) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= vocab) return;
  const float* row = embd + (long long)v * n_embd;
  float acc = 0.0f;
  for (int k = 0; k < n_embd; ++k) {
    acc = __fadd_rn(acc, __fmul_rn(h[k], row[k]));
  }
  logits[v] = acc;
}

// gqa_attention_decode_f32 — the M=1 (seq=1) decode case of tritium_nn::ops::
// gqa_attention. One thread per query head: scaled sequential q·k dots → inline
// row softmax → sequential weighted sum Σ w·v, all in the host's index order with
// __fmul_rn/__fadd_rn (no FMA). The softmax exponential uses `exp_f32` (f64 exp →
// f32), so it matches glibc `expf` and the whole op bit-matches the host. `limit` is
// the last visible key index (causal_offset); keys j>limit are masked (-inf).
// `scores` is a [n_head, ctx] scratch the caller provides (resident in the forward).
__global__ void gqa_attention_decode_f32(const float* __restrict__ q,    // [n_head, head_dim]
                                         const float* __restrict__ k,    // [ctx, n_head_kv, head_dim]
                                         const float* __restrict__ v,    // [ctx, n_head_kv, head_dim]
                                         float* __restrict__ out,        // [n_head, head_dim]
                                         float* __restrict__ scores,     // [n_head, ctx] scratch
                                         const int ctx, const int n_head,
                                         const int n_head_kv, const int head_dim,
                                         const float scale, const int limit) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= n_head) return;

  const int n_rep = n_head / n_head_kv;
  const int kv = h / n_rep;
  const float* q_row = q + (long long)h * head_dim;
  float* sc = scores + (long long)h * ctx;

  // Scaled dot scores; masked keys (j > limit) -> -inf.
  for (int j = 0; j < ctx; ++j) {
    if (j > limit) {
      sc[j] = -INFINITY;
      continue;
    }
    const float* k_row = k + ((long long)j * n_head_kv + kv) * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      dot = __fadd_rn(dot, __fmul_rn(q_row[d], k_row[d]));  // host: dot += q[d]*k[d]
    }
    sc[j] = __fmul_rn(dot, scale);  // host: dot * scale
  }

  // Inline row softmax (same sequential math as softmax_f32; exp_f32 matches glibc).
  float m = -INFINITY;
  for (int j = 0; j < ctx; ++j) {
    if (sc[j] > m) m = sc[j];
  }
  float sum = 0.0f;
  for (int j = 0; j < ctx; ++j) {
    const float e = exp_f32(__fsub_rn(sc[j], m));
    sc[j] = e;
    sum = __fadd_rn(sum, e);
  }
  const float inv = __fdiv_rn(1.0f, sum);
  for (int j = 0; j < ctx; ++j) {
    sc[j] = __fmul_rn(sc[j], inv);
  }

  // Weighted sum of v rows, host order (j outer, d inner); skip w==0 like the host.
  float* o_row = out + (long long)h * head_dim;
  for (int d = 0; d < head_dim; ++d) {
    o_row[d] = 0.0f;
  }
  for (int j = 0; j < ctx; ++j) {
    const float w = sc[j];
    if (w == 0.0f) continue;
    const float* v_row = v + ((long long)j * n_head_kv + kv) * head_dim;
    for (int d = 0; d < head_dim; ++d) {
      o_row[d] = __fadd_rn(o_row[d], __fmul_rn(w, v_row[d]));  // host: o[d] += w*v[d]
    }
  }
}

// act_quant_tiled_f32 — bit-match of tritium_nn::ops::quantize_activation_int8 for
// ONE row (M=1 decode), producing the int8 values kept as f32 that the tiled add-only
// GEMM consumes, plus the per-token dequant scale. Per the host (Qp=127):
//   gamma = max_k |act[k]|
//   gamma==0 → q_out = 0, *act_scale = 0
//   else  s = 127/gamma ; q_out[k] = clamp(round_ties_even(act[k]*s), -128, 127)
//         *act_scale = gamma/127
// `rintf` is IEEE round-to-nearest-even (== Rust f32::round_ties_even). Thread 0 does
// the sequential absmax (bit-match); the per-element quant is parallel + order-free.
__global__ void act_quant_tiled_f32(const float* __restrict__ act, const int k,
                                    float* __restrict__ q_out,
                                    float* __restrict__ act_scale) {
  __shared__ float s_red[256];  // block reduction scratch (blockDim.x == 256)
  __shared__ float s_gamma;
  // Parallel absmax: each thread reduces its strided slice, then a shared-memory tree
  // reduction combines them. `max` is associative+commutative, so the result is
  // **bit-identical** to the sequential `a > gamma` fold (unlike a sum, no reordering
  // error) — this stays an exact match of the host quant while using the whole block.
  float local = 0.0f;
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    const float a = fabsf(act[i]);
    if (a > local) local = a;
  }
  s_red[threadIdx.x] = local;
  __syncthreads();
  // max is exact under any order: shared levels 128/64, warp-0 shuffle tail
  // (6 fewer barriers); thread 0 lands the result back in s_red[0] for the
  // existing consumer below.
  for (int off = 128; off >= 64; off >>= 1) {
    if (threadIdx.x < off) {
      const float o = s_red[threadIdx.x + off];
      if (o > s_red[threadIdx.x]) s_red[threadIdx.x] = o;
    }
    __syncthreads();
  }
  if (threadIdx.x < 32) {
    float v = fmaxf(s_red[threadIdx.x], s_red[threadIdx.x + 32]);
    for (int off = 16; off > 0; off >>= 1) {
      v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    }
    if (threadIdx.x == 0) {
      s_red[0] = v;
    }
  }
  if (threadIdx.x == 0) {
    const float gamma = s_red[0];
    s_gamma = gamma;
    *act_scale = (gamma == 0.0f) ? 0.0f : __fdiv_rn(gamma, 127.0f);  // host: gamma/127
  }
  __syncthreads();

  const float gamma = s_gamma;
  if (gamma == 0.0f) {
    for (int i = threadIdx.x; i < k; i += blockDim.x) q_out[i] = 0.0f;
    return;
  }
  const float s = __fdiv_rn(127.0f, gamma);  // host: 127/gamma
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    const float scaled = rintf(__fmul_rn(act[i], s));        // host: (act*s).round_ties_even()
    q_out[i] = fminf(fmaxf(scaled, -128.0f), 127.0f);        // host: .clamp(-128, 127)
  }
}

// ─── i8-emitting quant variants (v1.x decode opt) ────────────────────────────
// Identical math to their `_f32` originals, but `q_out` is **packed int8** — the
// exact same integer values the f32 kernels stored as floats (the final
// f32→int8 conversion of an already-rounded, already-clamped value is exact),
// just 4× smaller. The `tq2_0_add_mpgemm_tiled_i8_scaled*` GEMMs read this
// buffer directly from global/L1 (no shared staging, no f32→i8 convert per
// block), which removes the dominant redundant traffic at M=1: every N-column
// block re-read the whole f32 activation row (e.g. N=2560 → 320 blocks × 10 KiB
// = 3.2 MiB of L2 traffic vs only 1.7 MiB of weights). The f32 originals stay
// for the public host-facing quant helpers and the f64 bit-parity GEMM path.
//
// The GEMMs read the row as `int` (4 bytes at a time) for full 256-trit blocks
// only, so rows must be 4-byte aligned: the host guarantees `k % 4 == 0` on
// every decode shape (all BitNet dims are multiples of 256).

// i8 sibling of `rmsnorm_quant_f32` (fused RMSNorm + A8 quant, M=1).
__global__ void rmsnorm_quant_i8(const float* __restrict__ x,
                                 const float* __restrict__ w,
                                 const float eps,
                                 const int n,
                                 signed char* __restrict__ q_out,
                                 float* __restrict__ act_scale) {
  extern __shared__ float s_x[];
  __shared__ float s_inv;
  __shared__ float s_gamma;
  __shared__ float s_red[1024];

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    s_x[i] = x[i];
  }
  __syncthreads();

  // Canonical tree sum (ADR 0018): slot t (t < 256) folds s_x[t],
  // s_x[t+256], ... in ascending order, then a power-of-two tree combines
  // the slots — the documented cross-backend order tritium_nn::ops::rmsnorm
  // implements identically on the host. Unlike its siblings this kernel pins
  // the fold to 256 slots EXPLICITLY, so it may launch with blockDim.x > 256
  // (multiples of 32, ≤ 1024): the extra threads accelerate the staging /
  // normalize / quant passes of this single-block, latency-bound kernel.
  {
    // Canonical 256-SLOT fold regardless of blockDim (ADR 0018): slot t < 256
    // folds x[t], x[t+256], … — threads ≥ 256 sit out the fold but accelerate
    // every elementwise pass (this kernel is single-block latency-bound; ncu:
    // ~1% of every throughput ceiling at 256 threads).
    float part = 0.0f;
    if (threadIdx.x < 256) {
      for (int i = threadIdx.x; i < n; i += 256) {
        const float xi = s_x[i];
        part = __fadd_rn(part, __fmul_rn(xi, xi));
      }
      s_red[threadIdx.x] = part;
    }
    __syncthreads();
    // Levels 128 and 64 in shared, then warp 0 finishes in registers: the
    // shuffle pairing (t, t+off) IS the canonical tree's pairing, so the DAG
    // (and every rounding) is identical — just 6 fewer block barriers.
    for (int off = 128; off >= 64; off >>= 1) {
      if (threadIdx.x < off) {
        s_red[threadIdx.x] = __fadd_rn(s_red[threadIdx.x], s_red[threadIdx.x + off]);
      }
      __syncthreads();
    }
    if (threadIdx.x < 32) {
      float v = __fadd_rn(s_red[threadIdx.x], s_red[threadIdx.x + 32]);  // level 32
      for (int off = 16; off > 0; off >>= 1) {
        v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, off));
      }
      if (threadIdx.x == 0) {
        const float mean_sq = __fdiv_rn(v, (float)n);
        const float denom = sqrtf(__fadd_rn(mean_sq, eps));
        s_inv = __fdiv_rn(1.0f, denom);
      }
    }
  }
  __syncthreads();

  const float inv = s_inv;
  float local_max = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float yi = __fmul_rn(__fmul_rn(s_x[i], inv), w[i]);
    s_x[i] = yi;
    const float a = fabsf(yi);
    if (a > local_max) local_max = a;
  }
  __syncthreads();

  // absmax slots: every thread parks its max in s_red[threadIdx.x] (s_red is
  // 1024 wide, blockDim ≤ 1024), then slot owners t < 256 fold EVERY upper
  // 256-stride bank in — generic over blockDim (512, 768, 1024 all correct),
  // and exact because max is order-free.
  s_red[threadIdx.x] = local_max;
  __syncthreads();
  if (threadIdx.x < 256) {
    float v = s_red[threadIdx.x];
    for (int base = 256; base + threadIdx.x < blockDim.x; base += 256) {
      v = fmaxf(v, s_red[threadIdx.x + base]);
    }
    s_red[threadIdx.x] = v;
  }
  __syncthreads();
  // max is exact under any order; same shared-then-warp-shuffle shape as the
  // sum tree above (6 fewer barriers).
  for (int off = 128; off >= 64; off >>= 1) {
    if (threadIdx.x < off) {
      s_red[threadIdx.x] = fmaxf(s_red[threadIdx.x], s_red[threadIdx.x + off]);
    }
    __syncthreads();
  }
  if (threadIdx.x < 32) {
    float v = fmaxf(s_red[threadIdx.x], s_red[threadIdx.x + 32]);
    for (int off = 16; off > 0; off >>= 1) {
      v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    }
    if (threadIdx.x == 0) {
      s_gamma = v;
      *act_scale = (v == 0.0f) ? 0.0f : __fdiv_rn(v, 127.0f);
    }
  }
  __syncthreads();

  const float gamma = s_gamma;
  if (gamma == 0.0f) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) q_out[i] = 0;
    return;
  }
  const float s = __fdiv_rn(127.0f, gamma);
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float scaled = rintf(__fmul_rn(s_x[i], s));
    q_out[i] = (signed char)fminf(fmaxf(scaled, -128.0f), 127.0f);
  }
}

// i8 sibling of `act_quant_tiled_f32` (standalone A8 quant, M=1).
__global__ void act_quant_tiled_i8(const float* __restrict__ act, const int k,
                                   signed char* __restrict__ q_out,
                                   float* __restrict__ act_scale) {
  __shared__ float s_red[256];
  __shared__ float s_gamma;
  float local = 0.0f;
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    const float a = fabsf(act[i]);
    if (a > local) local = a;
  }
  s_red[threadIdx.x] = local;
  __syncthreads();
  // max is exact under any order: shared levels 128/64, warp-0 shuffle tail
  // (6 fewer barriers); thread 0 lands the result back in s_red[0] for the
  // existing consumer below.
  for (int off = 128; off >= 64; off >>= 1) {
    if (threadIdx.x < off) {
      const float o = s_red[threadIdx.x + off];
      if (o > s_red[threadIdx.x]) s_red[threadIdx.x] = o;
    }
    __syncthreads();
  }
  if (threadIdx.x < 32) {
    float v = fmaxf(s_red[threadIdx.x], s_red[threadIdx.x + 32]);
    for (int off = 16; off > 0; off >>= 1) {
      v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    }
    if (threadIdx.x == 0) {
      s_red[0] = v;
    }
  }
  if (threadIdx.x == 0) {
    const float gamma = s_red[0];
    s_gamma = gamma;
    *act_scale = (gamma == 0.0f) ? 0.0f : __fdiv_rn(gamma, 127.0f);
  }
  __syncthreads();

  const float gamma = s_gamma;
  if (gamma == 0.0f) {
    for (int i = threadIdx.x; i < k; i += blockDim.x) q_out[i] = 0;
    return;
  }
  const float s = __fdiv_rn(127.0f, gamma);
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    const float scaled = rintf(__fmul_rn(act[i], s));
    q_out[i] = (signed char)fminf(fmaxf(scaled, -128.0f), 127.0f);
  }
}

// i8 sibling of `act_quant_batch_f32` (per-row A8 quant of `[m, k]`).
__global__ void act_quant_batch_i8(const float* __restrict__ act, const int k, const int m,
                                   signed char* __restrict__ q_out,
                                   float* __restrict__ act_scale) {
  const int mi = blockIdx.x;
  if (mi >= m) return;
  __shared__ float s_red[256];
  __shared__ float s_gamma;
  const float* ar = act + (long long)mi * k;
  signed char* qr = q_out + (long long)mi * k;
  float local = 0.0f;
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    const float a = fabsf(ar[i]);
    if (a > local) local = a;
  }
  s_red[threadIdx.x] = local;
  __syncthreads();
  // max is exact under any order: shared levels 128/64, warp-0 shuffle tail
  // (6 fewer barriers); thread 0 lands the result back in s_red[0] for the
  // existing consumer below.
  for (int off = 128; off >= 64; off >>= 1) {
    if (threadIdx.x < off) {
      const float o = s_red[threadIdx.x + off];
      if (o > s_red[threadIdx.x]) s_red[threadIdx.x] = o;
    }
    __syncthreads();
  }
  if (threadIdx.x < 32) {
    float v = fmaxf(s_red[threadIdx.x], s_red[threadIdx.x + 32]);
    for (int off = 16; off > 0; off >>= 1) {
      v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    }
    if (threadIdx.x == 0) {
      s_red[0] = v;
    }
  }
  if (threadIdx.x == 0) {
    const float gamma = s_red[0];
    s_gamma = gamma;
    act_scale[mi] = (gamma == 0.0f) ? 0.0f : __fdiv_rn(gamma, 127.0f);
  }
  __syncthreads();
  const float gamma = s_gamma;
  if (gamma == 0.0f) {
    for (int i = threadIdx.x; i < k; i += blockDim.x) qr[i] = 0;
    return;
  }
  const float s = __fdiv_rn(127.0f, gamma);
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    const float scaled = rintf(__fmul_rn(ar[i], s));
    qr[i] = (signed char)fminf(fmaxf(scaled, -128.0f), 127.0f);
  }
}

// scale_mul_f32 — out[i] *= *s. The per-token activation-dequant fold the host applies
// after the GEMM (`*slot *= act_scale[r]`); a single f32 mul (no FMA), `s` a device
// scalar. Embarrassingly parallel.
__global__ void scale_mul_f32(float* __restrict__ out, const float* __restrict__ s,
                              const int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = __fmul_rn(out[i], *s);
  }
}

// relu2_gate_f32 — BitNet squared-ReLU gating, bit-exact match of the host
// tritium_nn::layers::mlp gating loop:
//   r = gate[i].max(0.0) ;  gate[i] = (r * r) * up[i]
// In place into `gate` (which holds gate_proj output); `up` holds up_proj output.
// `fmaxf(g, 0)` matches Rust `f32::max(0.0)` (both return the non-NaN operand), and
// the two muls are left-associated with no FMA contraction (host: `r * r * u`).
// Elementwise → order-free, fully parallel.
__global__ void relu2_gate_f32(float* __restrict__ gate, const float* __restrict__ up,
                               const int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float r = fmaxf(gate[i], 0.0f);                 // host: g.max(0.0)
    gate[i] = __fmul_rn(__fmul_rn(r, r), up[i]);          // host: r * r * u (left-assoc, no FMA)
  }
}

// ===========================================================================
// v0.3.2 (#29): CUDA-graph "_g" variants. A captured graph bakes by-value kernel
// params, but the decode step's token id / RoPE position / KV write offset /
// attention range change every token. These variants read those per-token values
// from a small **device control block** `ctrl` (int[4] = {token, pos, cache_len, _})
// instead of by-value args, so ONE captured graph replays across tokens (the host
// just rewrites `ctrl` + the input between replays). The math is byte-identical to
// the by-value kernels above given the same control values — the originals stay for
// the eager path + the goldens; these are added alongside (not replacing).
// ===========================================================================

// embedding_gather_f32_g — like embedding_gather_f32 but tok = ctrl[0].
__global__ void embedding_gather_f32_g(const float* __restrict__ table,
                                       const int* __restrict__ ctrl,
                                       const int n_embd, float* __restrict__ out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_embd) {
    const int tok = ctrl[0];
    out[i] = table[(long long)tok * n_embd + i];
  }
}

// rope_apply_f32_g — like rope_apply_f32 but pos = ctrl[1], indexing the FULL
// precomputed cos/sin table `[max_ctx, half]` itself (the eager kernel takes a
// pre-sliced row; a captured graph cannot re-slice per token). Rotation math is
// identical (two muls then sub/add, no FMA).
__global__ void rope_apply_f32_g(float* __restrict__ x,
                                 const float* __restrict__ cos_table,  // [max_ctx*half]
                                 const float* __restrict__ sin_table,  // [max_ctx*half]
                                 const int* __restrict__ ctrl,
                                 const int n_head, const int head_dim) {
  const int half = head_dim >> 1;
  const int total = n_head * half;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  const int head = idx / half;
  const int j = idx - head * half;
  const int base = head * head_dim;
  const int pos = ctrl[1];
  const float c = cos_table[(long long)pos * half + j];
  const float s = sin_table[(long long)pos * half + j];
  const float a = x[base + j];
  const float b = x[base + j + half];
  x[base + j] = __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
  x[base + j + half] = __fadd_rn(__fmul_rn(b, c), __fmul_rn(a, s));
}

// kv_append_f32 — the device half of the KV append: copy this token's `src` row
// (`[kv_width]`) into the layer's KV arena at offset `ctrl[2]*kv_width`. Replaces the
// eager path's `memcpy_dtod` (whose offset would be baked into the graph). Pure copy.
}  // extern "C" — templates cannot carry C linkage; the shims below reopen it.

// ── KV store codecs (ADR 0022 twin consolidation, Track B) ──────────────────
// One templated body per KV-touching family; extern "C" shims preserve every
// launch symbol, so the host side is untouched. PROOF OBLIGATION: every
// instantiation stays SASS byte-identical to the retired hand-written kernel
// or is justified in ADR 0022 (kv_append_batch f32/h: opcode-identical,
// register permutation only). tools/sass_diff.sh; re-diff on toolchain bumps. The q8/t2 rungs keep
// their own signatures — they already delegate to the shared kv_quant_row_*
// helpers (scale-arena axis), i.e. they were born consolidated.
struct KvStoreF32 {
  using T = float;
  static __device__ __forceinline__ void store(float* p, float v) { *p = v; }
};
struct KvStoreF16 {
  using T = __half;
  static __device__ __forceinline__ void store(__half* p, float v) {
    *p = __float2half_rn(v);
  }
};

template <class C>
static __device__ __forceinline__ void kv_append_body(
    const float* __restrict__ src, typename C::T* __restrict__ kv_base,
    const int* __restrict__ ctrl, const int kv_width) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < kv_width) {
    const long long off = (long long)ctrl[2] * kv_width + i;
    C::store(kv_base + off, src[i]);
  }
}

template <class C>
static __device__ __forceinline__ void kv_append_batch_body(
    const float* __restrict__ src, typename C::T* __restrict__ kv_base,
    const int cache_len, const int kv_width, const int m) {
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long long)m * kv_width) return;
  const int row = idx / kv_width;
  const int e = idx - (long long)row * kv_width;
  C::store(&kv_base[((long long)(cache_len + row)) * kv_width + e],
           src[(long long)row * kv_width + e]);
}

template <class C>
static __device__ __forceinline__ void rope_kv_fused_body(
    float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, typename C::T* __restrict__ kv_k_base,
    typename C::T* __restrict__ kv_v_base, const float* __restrict__ cos_table,
    const float* __restrict__ sin_table, const int* __restrict__ ctrl,
    const int n_head, const int n_head_kv, const int head_dim,
    const int kv_width) {
  const int half = head_dim >> 1;
  const int q_total = n_head * half;
  const int k_total = n_head_kv * half;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int pos = ctrl[1];
  const long long row = (long long)ctrl[2] * kv_width;
  if (idx < q_total) {
    const int head = idx / half;
    const int j = idx - head * half;
    const int base = head * head_dim;
    const float c = cos_table[(long long)pos * half + j];
    const float s = sin_table[(long long)pos * half + j];
    const float a = q[base + j];
    const float b = q[base + j + half];
    q[base + j] = __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
    q[base + j + half] = __fadd_rn(__fmul_rn(b, c), __fmul_rn(a, s));
  } else if (idx < q_total + k_total) {
    const int t = idx - q_total;
    const int head = t / half;
    const int j = t - head * half;
    const int base = head * head_dim;
    const float c = cos_table[(long long)pos * half + j];
    const float s = sin_table[(long long)pos * half + j];
    const float a = k[base + j];
    const float b = k[base + j + half];
    C::store(&kv_k_base[row + base + j],
             __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s)));
    C::store(&kv_k_base[row + base + j + half],
             __fadd_rn(__fmul_rn(b, c), __fmul_rn(a, s)));
  } else if (idx < q_total + k_total + kv_width) {
    const int i = idx - q_total - k_total;
    C::store(&kv_v_base[row + i], v[i]);
  }
}

extern "C" {

__global__ void kv_append_f32(const float* __restrict__ src,
                              float* __restrict__ kv_base,   // [max_ctx*kv_width]
                              const int* __restrict__ ctrl,
                              const int kv_width) {
  kv_append_body<KvStoreF32>(src, kv_base, ctrl, kv_width);
}

// rope_kv_fused_g — fused q-RoPE + k-RoPE + K/V append for the M=1 decode graph
// (v1.x node-count opt): replaces FOUR launches per layer (rope q, rope k,
// kv_append k, kv_append v) with ONE, removing ~90 graph nodes per token.
// Values are bit-identical to the unfused sequence:
//   * q pairs rotate exactly as rope_apply_f32_g (same ops, in place);
//   * k pairs rotate identically and the rotated pair is written DIRECTLY to
//     the KV arena row ctrl[2] (the append fused at the value level; the k
//     scratch row is no longer written back — nothing reads it post-append);
//   * v lanes are the kv_append_f32 pure copy.
// Thread space: [0, n_head·half) q-rotate | [+, n_head_kv·half) k-rotate+append
// | [+, kv_width) v-copy.
__global__ void rope_kv_fused_g(float* __restrict__ q,
                                const float* __restrict__ k,
                                const float* __restrict__ v,
                                float* __restrict__ kv_k_base,  // [max_ctx*kv_width]
                                float* __restrict__ kv_v_base,  // [max_ctx*kv_width]
                                const float* __restrict__ cos_table,
                                const float* __restrict__ sin_table,
                                const int* __restrict__ ctrl,
                                const int n_head, const int n_head_kv,
                                const int head_dim, const int kv_width) {
  rope_kv_fused_body<KvStoreF32>(q, k, v, kv_k_base, kv_v_base, cos_table,
                                 sin_table, ctrl, n_head, n_head_kv, head_dim,
                                 kv_width);
}

// gqa_attention_decode_f32_g — like gqa_attention_decode_f32 but cache_len = ctrl[2]
// (so ctx = cache_len+1, limit = cache_len), and `scores` is strided by the FIXED
// `max_ctx` (not the per-token `ctx`), since the scratch layout must be constant
// across replays. Same sequential dots / inline softmax (exp_f32) / weighted sum.
__global__ void gqa_attention_decode_f32_g(const float* __restrict__ q,    // [n_head, head_dim]
                                           const float* __restrict__ k,    // [max_ctx, n_head_kv, head_dim]
                                           const float* __restrict__ v,    // [max_ctx, n_head_kv, head_dim]
                                           float* __restrict__ out,        // [n_head, head_dim]
                                           float* __restrict__ scores,     // [n_head, max_ctx] scratch
                                           const int* __restrict__ ctrl,
                                           const int max_ctx, const int n_head,
                                           const int n_head_kv, const int head_dim,
                                           const float scale) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= n_head) return;

  const int cache_len = ctrl[2];
  const int ctx = cache_len + 1;
  const int limit = cache_len;
  const int n_rep = n_head / n_head_kv;
  const int kv = h / n_rep;
  const float* q_row = q + (long long)h * head_dim;
  float* sc = scores + (long long)h * max_ctx;  // FIXED stride (constant across replays)

  for (int j = 0; j < ctx; ++j) {
    if (j > limit) {
      sc[j] = -INFINITY;
      continue;
    }
    const float* k_row = k + ((long long)j * n_head_kv + kv) * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      dot = __fadd_rn(dot, __fmul_rn(q_row[d], k_row[d]));
    }
    sc[j] = __fmul_rn(dot, scale);
  }

  float m = -INFINITY;
  for (int j = 0; j < ctx; ++j) {
    if (sc[j] > m) m = sc[j];
  }
  float sum = 0.0f;
  for (int j = 0; j < ctx; ++j) {
    const float e = exp_f32(__fsub_rn(sc[j], m));
    sc[j] = e;
    sum = __fadd_rn(sum, e);
  }
  const float inv = __fdiv_rn(1.0f, sum);
  for (int j = 0; j < ctx; ++j) {
    sc[j] = __fmul_rn(sc[j], inv);
  }

  float* o_row = out + (long long)h * head_dim;
  for (int d = 0; d < head_dim; ++d) {
    o_row[d] = 0.0f;
  }
  for (int j = 0; j < ctx; ++j) {
    const float w = sc[j];
    if (w == 0.0f) continue;
    const float* v_row = v + ((long long)j * n_head_kv + kv) * head_dim;
    for (int d = 0; d < head_dim; ++d) {
      o_row[d] = __fadd_rn(o_row[d], __fmul_rn(w, v_row[d]));
    }
  }
}

// gqa_attention_decode_warp_g — ONE WARP per query head (vs the one-thread-per-head
// `_g` kernel, which leaves 31/32 of a warp idle). **Bit-identical** to it (modulo the
// shared `exp_f32`): the parallelism is across keys + output dims, not inside any
// f32-SUM reduction, so no f32 sum is reordered.
//
// v1.x decode opt — this kernel was the measured decode bottleneck (~62% of decode
// GPU time): scores lived in GLOBAL memory and the whole softmax ran sequentially
// on lane 0, including the f64 `exp_f32` (1/64 rate on consumer GPUs) one key at a
// time. Three bit-exactness-preserving fixes:
//   * scores are staged in dynamic SHARED memory (`max_ctx · 4` bytes; the launch
//     is now ONE warp per block, and the host opts the function into the size via
//     CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES — see `attn_shared_opt_in`
//     in cuda.rs — up to the device opt-in limit, ≈ 25K ctx on Ada). The global
//     `scores` scratch parameter is kept for launch/graph ABI compatibility but
//     no longer written.
//   * the max scan is a parallel warp reduction — f32 max is EXACT (no rounding)
//     and NaN-skipping `fmaxf` matches the sequential `>` scan's NaN behaviour,
//     so any reduction order yields the bit-identical maximum.
//   * `exp_f32` is elementwise (not a reduction), so all 32 lanes exponentiate
//     their own keys in parallel — same input, same function, same bits.
// Only the softmax SUM is a rounded f32 fold, so it alone stays sequential on
// lane 0, in the host's j-order, now reading shared instead of global.
//
//   * scores: lane-per-key — each lane runs the FULL sequential `head_dim` dot for its
//     keys (j = lane, lane+32, …), so each dot keeps the host's d-order.
//   * weighted sum: lane-per-output-dim — lane `d` sums `Σ_j w[j]·v[j][d]` in the host's
//     j-order, so each output keeps its order.
// `__syncwarp` separates the phases. ctrl[2] = cache_len (ctx = +1, limit = cache_len).
// Launch contract: grid.x = n_head, block = 32 (one warp), dynamic shared =
// max_ctx · 4 bytes.
__global__ void gqa_attention_decode_warp_g(const float* __restrict__ q,
                                            const float* __restrict__ k,
                                            const float* __restrict__ v,
                                            float* __restrict__ out,
                                            float* __restrict__ scores,
                                            const int* __restrict__ ctrl,
                                            const int max_ctx, const int n_head,
                                            const int n_head_kv, const int head_dim,
                                            const float scale) {
  extern __shared__ float sc[];  // this head's scores, [max_ctx] (ctx used)
  (void)scores;                  // legacy global scratch — ABI-compatible, unused
  const int h = blockIdx.x;      // one warp-block per head
  const int lane = threadIdx.x & 31;
  if (h >= n_head) return;
  const int cache_len = ctrl[2];
  const int ctx = cache_len + 1;
  const int limit = cache_len;
  const int n_rep = n_head / n_head_kv;
  const int kv = h / n_rep;
  const float* q_row = q + (long long)h * head_dim;

  // scores: lane-per-key; each lane's dot is sequential over head_dim (bit-exact).
  // Two keys' chains run interleaved per lane (v1.x ILP): the chains are
  // INDEPENDENT — each key's dot still folds in the host's d-order, so every
  // sc[j] is bit-identical; the interleave only hides the ~4-cycle fadd latency.
  for (int j0 = lane; j0 < ctx; j0 += 64) {
    const int j1 = j0 + 32;
    if (j0 > limit) {  // defensive — unreachable in decode (ctx == limit + 1), kept for parity
      sc[j0] = -INFINITY;
      if (j1 < ctx) sc[j1] = -INFINITY;
      continue;
    }
    const float* k0 = k + ((long long)j0 * n_head_kv + kv) * head_dim;
    if (j1 < ctx && j1 <= limit) {
      const float* k1 = k + ((long long)j1 * n_head_kv + kv) * head_dim;
      float d0 = 0.0f;
      float d1 = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        const float qd = q_row[d];
        d0 = __fadd_rn(d0, __fmul_rn(qd, k0[d]));
        d1 = __fadd_rn(d1, __fmul_rn(qd, k1[d]));
      }
      sc[j0] = __fmul_rn(d0, scale);
      sc[j1] = __fmul_rn(d1, scale);
    } else {
      float d0 = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        d0 = __fadd_rn(d0, __fmul_rn(q_row[d], k0[d]));
      }
      sc[j0] = __fmul_rn(d0, scale);
      if (j1 < ctx) sc[j1] = -INFINITY;  // j1 > limit (defensive, unreachable)
    }
  }
  __syncwarp();

  // max: parallel warp reduction — exact for f32 (max never rounds), and
  // `fmaxf(m, NaN) == m` skips NaN exactly like the sequential `>` scan did.
  float m = -INFINITY;
  for (int j = lane; j < ctx; j += 32) {
    m = fmaxf(m, sc[j]);
  }
  for (int off = 16; off > 0; off >>= 1) {
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
  }

  // exp: elementwise, parallel across lanes — identical `exp_f32` bits per key.
  for (int j = lane; j < ctx; j += 32) {
    sc[j] = exp_f32(__fsub_rn(sc[j], m));
  }
  __syncwarp();

  // sum: the ONE rounded f32 fold — sequential on lane 0 in the host's j-order.
  __shared__ float s_inv;
  if (lane == 0) {
    float sum = 0.0f;
    for (int j = 0; j < ctx; ++j) {
      sum = __fadd_rn(sum, sc[j]);
    }
    s_inv = __fdiv_rn(1.0f, sum);
  }
  __syncwarp();

  // normalize: elementwise, parallel.
  const float inv = s_inv;
  for (int j = lane; j < ctx; j += 32) {
    sc[j] = __fmul_rn(sc[j], inv);
  }
  __syncwarp();

  // weighted sum: lane-per-output-dim; each output sums over j in the host's order.
  // v1.x ILP: a lane owns dims d = lane, lane+32, … — all of them accumulate in ONE
  // pass over j (independent per-dim chains, each still in j-order → bit-identical),
  // instead of head_dim/32 separate passes. Each v row is now touched once per j
  // with a full coalesced read, and the chains interleave to hide fadd latency.
  // `ATTN_MAX_DIMS_PER_LANE` covers head_dim ≤ 256; larger heads take the fallback.
  float* o_row = out + (long long)h * head_dim;
  const int ndims = (head_dim > lane) ? (head_dim - lane + 31) / 32 : 0;
#define ATTN_MAX_DIMS_PER_LANE 8
  if (ndims <= ATTN_MAX_DIMS_PER_LANE) {
    float acc[ATTN_MAX_DIMS_PER_LANE];
    for (int t = 0; t < ndims; ++t) {
      acc[t] = 0.0f;
    }
    for (int j = 0; j < ctx; ++j) {
      const float w = sc[j];
      if (w == 0.0f) continue;
      const float* v_row = v + ((long long)j * n_head_kv + kv) * head_dim;
#pragma unroll
      for (int t = 0; t < ATTN_MAX_DIMS_PER_LANE; ++t) {
        if (t < ndims) {
          acc[t] = __fadd_rn(acc[t], __fmul_rn(w, v_row[lane + 32 * t]));
        }
      }
    }
    for (int t = 0; t < ndims; ++t) {
      o_row[lane + 32 * t] = acc[t];
    }
  } else {
    // head_dim > 256: the original one-dim-at-a-time loop (same order per dim).
    for (int d = lane; d < head_dim; d += 32) {
      float acc = 0.0f;
      for (int j = 0; j < ctx; ++j) {
        const float w = sc[j];
        if (w == 0.0f) continue;
        const float* v_row = v + ((long long)j * n_head_kv + kv) * head_dim;
        acc = __fadd_rn(acc, __fmul_rn(w, v_row[d]));
      }
      o_row[d] = acc;
    }
  }
#undef ATTN_MAX_DIMS_PER_LANE
}

// ─── Split decode attention (v1.x) — the bit-exact ctx-parallel pair ─────────
//
// The single warp-per-head kernel above is latency-bound: 20 warps on a 128-SM
// GPU, each streaming its keys/values behind a dependent f32 chain — measured
// ~0.5µs per context element. The split pair keeps EVERY f32 sum in the host's
// order (per-key dot in d-order, softmax sum + weighted sum in j-order) but
// distributes the independent work:
//
//   * `gqa_attention_scores_g` — grid (n_head, ceil(max_ctx/SCORE_CHUNK)), one
//     warp per block, 4 keys per lane as float4 chains (in-vector order = the
//     host's d-order, so each dot is bit-identical). Keys are mutually
//     independent, so fanning them out across blocks reorders nothing. Scores
//     land in the global `scores` scratch ([n_head, max_ctx], revived).
//     Requires head_dim % 4 == 0 and 16-byte-aligned K rows (head_dim % 4 == 0
//     implies it, rows being head_dim-strided from a 256-byte-aligned base);
//     the host routes other geometries to the legacy warp kernel above.
//   * `gqa_attention_reduce_g` — grid n_head, ONE 128-THREAD BLOCK per head:
//     stages the head's scores into shared (coalesced), block-tree max (exact,
//     `fmaxf` NaN-skip == the sequential scan), parallel `exp_f32`, the ONE
//     rounded fold (softmax sum) sequential on thread 0 in j-order, parallel
//     normalize, then ONE OUTPUT DIM PER THREAD for the weighted sum — each
//     dim's j-order chain unchanged, 4× the outstanding v-loads of one warp.
//     The v loads are hoisted OUT of the w==0 skip (the skip predicates only
//     the fadd — identical accumulation, but the loads batch).
//
// Measured (4090, standalone, both kernels): 2.1× at ctx=64 → 9× at ctx=4096
// vs the single-kernel form, bit-exact at every context length.
// Launch contract: scores → block=32, no dynamic shared; reduce → block=128,
// dynamic shared = max_ctx·4 (same over-48KiB opt-in as the legacy kernel).
#define SCORE_CHUNK 128

}  // extern "C" — KV LOAD codecs + attention template bodies (Track B).

// ── KV load codecs (ADR 0022, the attention families' rung axis) ────────────
// f32 loads a float4 directly; f16 converts __half2 pairs (the former
// kvh_load4 helper, hoisted here). The i8 rung is NOT a load codec: its inner
// loop carries a per-(token, head, group) scale stream (a structurally
// different contraction, tuning-sensitive per ADR 0020) — the _q8 kernels
// stay hand-written, justified in ADR 0022.
__device__ __forceinline__ float4 kvh_load4(const __half* p) {
  const __half2* h2 = (const __half2*)p;
  const float2 lo = __half22float2(h2[0]);
  const float2 hi = __half22float2(h2[1]);
  return make_float4(lo.x, lo.y, hi.x, hi.y);
}

#define TREE_SCORE_CHUNK 128

struct KvLoadF32 {
  using T = float;
  // Row handle + indexed load, shaped EXACTLY like the hand-written f32
  // kernels (float4* declared once, indexed per d) so the instantiation
  // compiles byte-identically.
  using Row = const float4*;
  static __device__ __forceinline__ Row row(const T* base, long long off) {
    return (const float4*)(base + off);
  }
  static __device__ __forceinline__ float4 load4(Row r, int d) { return r[d]; }
  static __device__ __forceinline__ float load(const T* p) { return *p; }
};
struct KvLoadF16 {
  using T = __half;
  using Row = const __half*;
  static __device__ __forceinline__ Row row(const T* base, long long off) {
    return base + off;
  }
  static __device__ __forceinline__ float4 load4(Row r, int d) {
    return kvh_load4(r + 4 * d);
  }
  static __device__ __forceinline__ float load(const T* p) {
    return __half2float(*p);
  }
};

template <class C>
static __device__ __forceinline__ void gqa_attention_scores_body(
    const float* __restrict__ q, const typename C::T* __restrict__ k,
    float* __restrict__ scores, const int* __restrict__ ctrl,
    const int max_ctx, const int n_head, const int n_head_kv,
    const int head_dim, const float scale) {
  const int h = blockIdx.x;
  const int chunk = blockIdx.y * SCORE_CHUNK;
  const int lane = threadIdx.x & 31;
  const int ctx = ctrl[2] + 1;
  if (h >= n_head || chunk >= ctx) return;
  const int n_rep = n_head / n_head_kv;
  const int kv = h / n_rep;
  const float4* qv = (const float4*)(q + (long long)h * head_dim);
  const int hd4 = head_dim >> 2;
  float* sc = scores + (long long)h * max_ctx;

  const int j0 = chunk + lane;
  const int j1 = j0 + 32;
  const int j2 = j0 + 64;
  const int j3 = j0 + 96;
  const typename C::Row k0 = C::row(k, ((long long)j0 * n_head_kv + kv) * head_dim);
  const typename C::Row k1 = C::row(k, ((long long)j1 * n_head_kv + kv) * head_dim);
  const typename C::Row k2 = C::row(k, ((long long)j2 * n_head_kv + kv) * head_dim);
  const typename C::Row k3 = C::row(k, ((long long)j3 * n_head_kv + kv) * head_dim);
  if (j3 < ctx) {
    // Fast path: all four of this lane's keys are live. Four independent
    // chains; each folds x→y→z→w then the next float4 — exactly d-order.
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
#pragma unroll 4
    for (int d = 0; d < hd4; ++d) {
      const float4 qq = qv[d];
      const float4 a = C::load4(k0, d);
      const float4 b = C::load4(k1, d);
      const float4 c = C::load4(k2, d);
      const float4 e = C::load4(k3, d);
      d0 = __fadd_rn(d0, __fmul_rn(qq.x, a.x));
      d1 = __fadd_rn(d1, __fmul_rn(qq.x, b.x));
      d2 = __fadd_rn(d2, __fmul_rn(qq.x, c.x));
      d3 = __fadd_rn(d3, __fmul_rn(qq.x, e.x));
      d0 = __fadd_rn(d0, __fmul_rn(qq.y, a.y));
      d1 = __fadd_rn(d1, __fmul_rn(qq.y, b.y));
      d2 = __fadd_rn(d2, __fmul_rn(qq.y, c.y));
      d3 = __fadd_rn(d3, __fmul_rn(qq.y, e.y));
      d0 = __fadd_rn(d0, __fmul_rn(qq.z, a.z));
      d1 = __fadd_rn(d1, __fmul_rn(qq.z, b.z));
      d2 = __fadd_rn(d2, __fmul_rn(qq.z, c.z));
      d3 = __fadd_rn(d3, __fmul_rn(qq.z, e.z));
      d0 = __fadd_rn(d0, __fmul_rn(qq.w, a.w));
      d1 = __fadd_rn(d1, __fmul_rn(qq.w, b.w));
      d2 = __fadd_rn(d2, __fmul_rn(qq.w, c.w));
      d3 = __fadd_rn(d3, __fmul_rn(qq.w, e.w));
    }
    sc[j0] = __fmul_rn(d0, scale);
    sc[j1] = __fmul_rn(d1, scale);
    sc[j2] = __fmul_rn(d2, scale);
    sc[j3] = __fmul_rn(d3, scale);
  } else {
    // Chunk tail: per-key guarded, same per-key math. j ascends with t, so the
    // break stops exactly at the first dead key.
    const typename C::Row ks[4] = {k0, k1, k2, k3};
    const int js[4] = {j0, j1, j2, j3};
    for (int t = 0; t < 4; ++t) {
      if (js[t] >= ctx) break;
      float dot = 0.0f;
#pragma unroll 4
      for (int d = 0; d < hd4; ++d) {
        const float4 qq = qv[d];
        const float4 a = C::load4(ks[t], d);
        dot = __fadd_rn(dot, __fmul_rn(qq.x, a.x));
        dot = __fadd_rn(dot, __fmul_rn(qq.y, a.y));
        dot = __fadd_rn(dot, __fmul_rn(qq.z, a.z));
        dot = __fadd_rn(dot, __fmul_rn(qq.w, a.w));
      }
      sc[js[t]] = __fmul_rn(dot, scale);
    }
  }
}

template <class C>
static __device__ __forceinline__ void gqa_attention_reduce_body(
    const typename C::T* __restrict__ v,
                                       const float* __restrict__ scores,
                                       float* __restrict__ out,
                                       const int* __restrict__ ctrl,
                                       const int max_ctx, const int n_head,
                                       const int n_head_kv, const int head_dim) {
  extern __shared__ float red_sc[];  // dynamic smem (aliases every extern-shared array)
  __shared__ float s_red[256];  // block max scratch — this kernel REQUIRES
                                // blockDim.x == 128 exactly (hardcoded 64-level
                                // + warp-0 shuffle tail below); keep in sync
                                // with ATTN_REDUCE_THREADS in cuda.rs.
  __shared__ float s_inv;
  const int h = blockIdx.x;
  const int tid = threadIdx.x;
  if (h >= n_head) return;
  const int ctx = ctrl[2] + 1;
  const int n_rep = n_head / n_head_kv;
  const int kv = h / n_rep;

  // Stage this head's raw scores from global (coalesced).
  const float* gsc = scores + (long long)h * max_ctx;
  for (int j = tid; j < ctx; j += blockDim.x) {
    red_sc[j] = gsc[j];
  }
  __syncthreads();

  // max: block tree — exact (f32 max never rounds; fmaxf skips NaN like the
  // sequential `>` scan).
  float m = -INFINITY;
  for (int j = tid; j < ctx; j += blockDim.x) {
    m = fmaxf(m, red_sc[j]);
  }
  s_red[tid] = m;
  __syncthreads();
  // max is exact under any order: one shared level (blockDim == 128), then a
  // warp-0 shuffle tail — 5 fewer block barriers than the full shared tree.
  if (tid < 64) {
    s_red[tid] = fmaxf(s_red[tid], s_red[tid + 64]);
  }
  __syncthreads();
  if (tid < 32) {
    float v = fmaxf(s_red[tid], s_red[tid + 32]);
    for (int off = 16; off > 0; off >>= 1) {
      v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    }
    if (tid == 0) {
      s_red[0] = v;
    }
  }
  __syncthreads();
  m = s_red[0];

  // exp: elementwise, parallel — identical exp_f32 bits per key.
  for (int j = tid; j < ctx; j += blockDim.x) {
    red_sc[j] = exp_f32(__fsub_rn(red_sc[j], m));
  }
  __syncthreads();

  // sum: the ONE rounded f32 fold — sequential on thread 0 in j-order.
  if (tid == 0) {
    float sum = 0.0f;
    for (int j = 0; j < ctx; ++j) {
      sum = __fadd_rn(sum, red_sc[j]);
    }
    s_inv = __fdiv_rn(1.0f, sum);
  }
  __syncthreads();
  const float inv = s_inv;
  for (int j = tid; j < ctx; j += blockDim.x) {
    red_sc[j] = __fmul_rn(red_sc[j], inv);
  }
  __syncthreads();

  // weighted sum: one output dim per thread, j-order chain per dim (bit-exact).
  // The v load is unconditional so the compiler batches loads across the
  // unrolled j iterations; the w==0 skip predicates only the fadd.
  float* o_row = out + (long long)h * head_dim;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
#pragma unroll 8
    for (int j = 0; j < ctx; ++j) {
      const float w = red_sc[j];
      const float vv = C::load(&v[((long long)j * n_head_kv + kv) * head_dim + d]);
      if (w != 0.0f) {
        acc = __fadd_rn(acc, __fmul_rn(w, vv));
      }
    }
    o_row[d] = acc;
  }
}


template <class C>
static __device__ __forceinline__ void gqa_attention_tree_scores_body(
    const float* __restrict__ q, const typename C::T* __restrict__ k,
    float* __restrict__ scores, const int* __restrict__ anc,
    const int* __restrict__ n_anc, const int ctx_max, const int n_head,
    const int n_head_kv, const int head_dim, const float scale,
    const int prefix_len, const int max_anc, const int m) {
  const int rowhead = blockIdx.x;
  const int row = rowhead / n_head;
  const int h = rowhead - row * n_head;
  if (row >= m) return;
  const int ctx = prefix_len + n_anc[row];
  const int chunk = blockIdx.y * TREE_SCORE_CHUNK;
  if (chunk >= ctx) return;
  const int lane = threadIdx.x & 31;
  const int n_rep = n_head / n_head_kv;
  const int kv = h / n_rep;
  const float4* qv = (const float4*)(q + ((long long)row * n_head + h) * head_dim);
  const int hd4 = head_dim >> 2;
  const int* arow = anc + (long long)row * max_anc;
  float* sc = scores + ((long long)row * n_head + h) * ctx_max;

  const int j0 = chunk + lane;
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    const int j = j0 + 32 * t;
    if (j >= ctx) break;
    const int slot = (j < prefix_len) ? j : arow[j - prefix_len];
    const typename C::Row kr = C::row(k, ((long long)slot * n_head_kv + kv) * head_dim);
    float dot = 0.0f;
#pragma unroll 8
    for (int d = 0; d < hd4; ++d) {
      const float4 qq = qv[d];
      const float4 a = C::load4(kr, d);
      dot = __fadd_rn(dot, __fmul_rn(qq.x, a.x));
      dot = __fadd_rn(dot, __fmul_rn(qq.y, a.y));
      dot = __fadd_rn(dot, __fmul_rn(qq.z, a.z));
      dot = __fadd_rn(dot, __fmul_rn(qq.w, a.w));
    }
    sc[j] = __fmul_rn(dot, scale);
  }
}

template <class C>
static __device__ __forceinline__ void gqa_attention_tree_reduce_body(
    const typename C::T* __restrict__ v, const float* __restrict__ scores,
    float* __restrict__ out, const int* __restrict__ anc,
    const int* __restrict__ n_anc, const int ctx_max, const int n_head,
    const int n_head_kv, const int head_dim, const int prefix_len,
    const int max_anc, const int m) {
  extern __shared__ float trd_sc[];  // dynamic smem alias
  __shared__ float s_red[256];
  __shared__ float s_inv;
  const int rowhead = blockIdx.x;
  const int row = rowhead / n_head;
  const int h = rowhead - row * n_head;
  if (row >= m) return;
  const int ctx = prefix_len + n_anc[row];
  const int tid = threadIdx.x;
  const int n_rep = n_head / n_head_kv;
  const int kv = h / n_rep;
  const int* arow = anc + (long long)row * max_anc;

  const float* gsc = scores + ((long long)row * n_head + h) * ctx_max;
  for (int j = tid; j < ctx; j += blockDim.x) {
    trd_sc[j] = gsc[j];
  }
  __syncthreads();

  float mx = -INFINITY;
  for (int j = tid; j < ctx; j += blockDim.x) {
    mx = fmaxf(mx, trd_sc[j]);
  }
  s_red[tid] = mx;
  __syncthreads();
  if (tid < 64) {
    s_red[tid] = fmaxf(s_red[tid], s_red[tid + 64]);
  }
  __syncthreads();
  if (tid < 32) {
    float vv = fmaxf(s_red[tid], s_red[tid + 32]);
    for (int off = 16; off > 0; off >>= 1) {
      vv = fmaxf(vv, __shfl_down_sync(0xffffffffu, vv, off));
    }
    if (tid == 0) {
      s_red[0] = vv;
    }
  }
  __syncthreads();
  mx = s_red[0];

  for (int j = tid; j < ctx; j += blockDim.x) {
    trd_sc[j] = exp_f32(__fsub_rn(trd_sc[j], mx));
  }
  __syncthreads();
  if (tid == 0) {
    float sum = 0.0f;
    for (int j = 0; j < ctx; ++j) {
      sum = __fadd_rn(sum, trd_sc[j]);
    }
    s_inv = __fdiv_rn(1.0f, sum);
  }
  __syncthreads();
  const float inv = s_inv;
  for (int j = tid; j < ctx; j += blockDim.x) {
    trd_sc[j] = __fmul_rn(trd_sc[j], inv);
  }
  __syncthreads();

  float* o_row = out + ((long long)row * n_head + h) * head_dim;
  const int ndims = (head_dim > tid) ? (head_dim - tid + (int)blockDim.x - 1) / (int)blockDim.x : 0;
#define TREE_MAX_DIMS_PER_THREAD 8
  if (ndims <= TREE_MAX_DIMS_PER_THREAD) {
    float acc[TREE_MAX_DIMS_PER_THREAD];
    for (int t = 0; t < ndims; ++t) {
      acc[t] = 0.0f;
    }
#pragma unroll 4
    for (int j = 0; j < ctx; ++j) {
      const float w = trd_sc[j];
      const int slot = (j < prefix_len) ? j : arow[j - prefix_len];
      const typename C::T* v_row = v + ((long long)slot * n_head_kv + kv) * head_dim;
      float vals[TREE_MAX_DIMS_PER_THREAD];
#pragma unroll
      for (int t = 0; t < TREE_MAX_DIMS_PER_THREAD; ++t) {
        if (t < ndims) {
          vals[t] = C::load(&v_row[tid + (int)blockDim.x * t]);
        }
      }
      if (w != 0.0f) {
#pragma unroll
        for (int t = 0; t < TREE_MAX_DIMS_PER_THREAD; ++t) {
          if (t < ndims) {
            acc[t] = __fadd_rn(acc[t], __fmul_rn(w, vals[t]));
          }
        }
      }
    }
    for (int t = 0; t < ndims; ++t) {
      o_row[tid + (int)blockDim.x * t] = acc[t];
    }
  } else {
    for (int d = tid; d < head_dim; d += blockDim.x) {
      float acc = 0.0f;
      for (int j = 0; j < ctx; ++j) {
        const float w = trd_sc[j];
        if (w == 0.0f) continue;
        const int slot = (j < prefix_len) ? j : arow[j - prefix_len];
        const typename C::T* v_row = v + ((long long)slot * n_head_kv + kv) * head_dim;
        acc = __fadd_rn(acc, __fmul_rn(w, C::load(&v_row[d])));
      }
      o_row[d] = acc;
    }
  }
#undef TREE_MAX_DIMS_PER_THREAD
}

template <class C>
static __device__ __forceinline__ void gqa_attention_batch_body(
    const float* __restrict__ q, const typename C::T* __restrict__ k,
                                        const typename C::T* __restrict__ v, float* __restrict__ out,
                                        float* __restrict__ scores, const int ctx_max,
                                        const int n_head, const int n_head_kv, const int head_dim,
                                        const float scale, const int causal_offset, const int m) {
  const long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const long long total = (long long)m * n_head;
  if (warp >= total) return;
  const int row = warp / n_head;
  const int h = warp - (long long)row * n_head;
  const int limit = causal_offset + row;
  const int ctx = limit + 1;
  const int n_rep = n_head / n_head_kv;
  const int kv = h / n_rep;
  const float* q_row = q + ((long long)row * n_head + h) * head_dim;
  float* sc = scores + ((long long)row * n_head + h) * ctx_max;

  for (int j = lane; j < ctx; j += 32) {
    const typename C::T* k_row = k + ((long long)j * n_head_kv + kv) * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      dot = __fadd_rn(dot, __fmul_rn(q_row[d], C::load(&k_row[d])));
    }
    sc[j] = __fmul_rn(dot, scale);
  }
  __syncwarp();
  if (lane == 0) {
    float mx = -INFINITY;
    for (int j = 0; j < ctx; ++j) {
      if (sc[j] > mx) mx = sc[j];
    }
    float sum = 0.0f;
    for (int j = 0; j < ctx; ++j) {
      const float e = exp_f32(__fsub_rn(sc[j], mx));
      sc[j] = e;
      sum = __fadd_rn(sum, e);
    }
    const float inv = __fdiv_rn(1.0f, sum);
    for (int j = 0; j < ctx; ++j) {
      sc[j] = __fmul_rn(sc[j], inv);
    }
  }
  __syncwarp();
  float* o_row = out + ((long long)row * n_head + h) * head_dim;
  for (int d = lane; d < head_dim; d += 32) {
    float acc = 0.0f;
    for (int j = 0; j < ctx; ++j) {
      const float w = sc[j];
      if (w == 0.0f) continue;
      const typename C::T* v_row = v + ((long long)j * n_head_kv + kv) * head_dim;
      acc = __fadd_rn(acc, __fmul_rn(w, C::load(&v_row[d])));
    }
    o_row[d] = acc;
  }
}

// ---- v2 batched prefill attention (order-preserving rewrite; ADR 0022) ----
//
// One ATTN_V2_THREADS-thread block per (query row, q-head): grid (n_head, m).
// Motivation (nsys round 22): gqa_attention_batch_* is 35% of a 512-token
// prefill. Its lane-per-key K walk is fully uncoalesced (32 lanes stride 32
// different KV rows), lane 0 runs the whole softmax alone, and the global
// `scores` scratch round-trips ~4x per (row, head). v2 fixes the mechanics
// WITHOUT touching any pinned fold order:
//
//   * K is staged into padded shared memory in coalesced ATTN_V2_KCHUNK-key
//     chunks (pad kills the bank conflicts a head_dim-stride walk would hit);
//     each staged value is C::load of the same element the rev-1 kernel read,
//     so the per-key d-order dot chain sees identical inputs in identical
//     order.
//   * Scores live in shared (no global scratch traffic).
//   * The max scan is a parallel fmaxf tree — max is EXACT under any order
//     (NaNs included: both the rev-1 sequential `>` scan and fmaxf skip NaNs,
//     yielding the max of the non-NaN elements). exp and the inv scale are
//     elementwise. The softmax SUM keeps the rev-1 sequential j-order fold on
//     a single thread — that order is pinned (decode-parity + ADR 0014 tree
//     gates compare against it).
//   * The V pass keeps the per-dim sequential-j fold and the `w == 0.0f` skip
//     (load-bearing: skipping avoids `-0.0 + +0.0 == +0.0` flips).
//
// Everything above makes v2 BIT-IDENTICAL per (row, head) to
// gqa_attention_batch_{f32,h} — the gate is `to_bits` equality against them.
//
// Host contract (dispatch in mod.rs): head_dim <= ATTN_V2_HDMAX and
// causal_offset + m <= ATTN_V2_MAX_CTX (the shared budget), else the rev-1
// kernel launches. Shared: scores 14,336 + K stage 33,024 + q 512 + red 512
// + inv 4 = 48,388 B, under the 48 KiB static ceiling.
#define ATTN_V2_THREADS 128
#define ATTN_V2_KCHUNK 64
#define ATTN_V2_HDMAX 128
#define ATTN_V2_MAX_CTX 3584

template <class C>
static __device__ __forceinline__ void gqa_attention_batch_v2_body(
    const float* __restrict__ q, const typename C::T* __restrict__ k,
    const typename C::T* __restrict__ v, float* __restrict__ out, const int n_head,
    const int n_head_kv, const int head_dim, const float scale, const int causal_offset,
    const int m) {
  const int h = blockIdx.x;
  const int row = blockIdx.y;
  if (row >= m || h >= n_head) return;
  const int tid = threadIdx.x;
  const int ctx = causal_offset + row + 1;
  const int n_rep = n_head / n_head_kv;
  const int kv = h / n_rep;

  __shared__ float s_scores[ATTN_V2_MAX_CTX];
  __shared__ float s_k[ATTN_V2_KCHUNK * (ATTN_V2_HDMAX + 1)];
  __shared__ float s_q[ATTN_V2_HDMAX];
  __shared__ float s_red[ATTN_V2_THREADS];
  __shared__ float s_inv;

  const float* q_row = q + ((long long)row * n_head + h) * head_dim;
  for (int d = tid; d < head_dim; d += ATTN_V2_THREADS) s_q[d] = q_row[d];
  __syncthreads();

  // Phase 1: scaled dots. Stage a key chunk coalesced (consecutive threads
  // read consecutive d of one KV row), then thread t folds key c0+t with the
  // rev-1 sequential d-order chain from shared. Row pad (+1) spreads the
  // per-thread rows across banks.
  const int kstride = head_dim + 1;
  for (int c0 = 0; c0 < ctx; c0 += ATTN_V2_KCHUNK) {
    const int nk = min(ATTN_V2_KCHUNK, ctx - c0);
    for (int idx = tid; idx < nk * head_dim; idx += ATTN_V2_THREADS) {
      const int j = idx / head_dim;
      const int d = idx - j * head_dim;
      s_k[j * kstride + d] =
          C::load(&k[((long long)(c0 + j) * n_head_kv + kv) * head_dim + d]);
    }
    __syncthreads();
    for (int t = tid; t < nk; t += ATTN_V2_THREADS) {
      const float* kr = &s_k[t * kstride];
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        dot = __fadd_rn(dot, __fmul_rn(s_q[d], kr[d]));
      }
      s_scores[c0 + t] = __fmul_rn(dot, scale);
    }
    __syncthreads();
  }

  // Phase 2: softmax. Parallel max (exact under any order), elementwise exp,
  // ORDER-PINNED sequential sum on thread 0, elementwise scale.
  float local = -INFINITY;
  for (int j = tid; j < ctx; j += ATTN_V2_THREADS) local = fmaxf(local, s_scores[j]);
  s_red[tid] = local;
  __syncthreads();
  for (int off = ATTN_V2_THREADS / 2; off > 0; off >>= 1) {
    if (tid < off) s_red[tid] = fmaxf(s_red[tid], s_red[tid + off]);
    __syncthreads();
  }
  const float mx = s_red[0];

  for (int j = tid; j < ctx; j += ATTN_V2_THREADS) {
    s_scores[j] = exp_f32(__fsub_rn(s_scores[j], mx));
  }
  __syncthreads();
  if (tid == 0) {
    float sum = 0.0f;
    for (int j = 0; j < ctx; ++j) {
      sum = __fadd_rn(sum, s_scores[j]);
    }
    s_inv = __fdiv_rn(1.0f, sum);
  }
  __syncthreads();
  const float inv = s_inv;
  for (int j = tid; j < ctx; j += ATTN_V2_THREADS) {
    s_scores[j] = __fmul_rn(s_scores[j], inv);
  }
  __syncthreads();

  // Phase 3: weighted V sum — per-dim sequential-j fold, coalesced across
  // threads (consecutive d), identical order + zero-skip to rev 1.
  float* o_row = out + ((long long)row * n_head + h) * head_dim;
  for (int d = tid; d < head_dim; d += ATTN_V2_THREADS) {
    float acc = 0.0f;
    for (int j = 0; j < ctx; ++j) {
      const float w = s_scores[j];
      if (w == 0.0f) continue;
      acc = __fadd_rn(
          acc, __fmul_rn(w, C::load(&v[((long long)j * n_head_kv + kv) * head_dim + d])));
    }
    o_row[d] = acc;
  }
}



// ---- v3 Q-blocked prefill attention (order-preserving; round 23 lever) ----
//
// v2 made attention 74% of prefill by fixing everything EXCEPT data reuse:
// each (row, head) block re-reads the full K/V. v3 blocks ATTN_V3_BQ query
// rows per (block, head): K chunks are staged ONCE and feed all BQ rows'
// dot chains; V values are loaded once per (thread, key) and folded into
// all BQ rows' accumulators — K/V traffic drops ~BQ x. Scores return to the
// global [m, n_head, ctx_max] scratch (their traffic is ~2 orders below the
// K/V traffic being amortized, and it lifts v2's ctx <= 3584 shared cap:
// v3 handles ANY ctx).
//
// Every pinned fold order is rev-1's, verbatim per row:
//   * per-key d-order dot chain (inputs staged bit-preserving);
//   * max via fmaxf reduction (exact under any order, NaN/-0.0 corners
//     argued at the v2 body); exp/scale elementwise;
//   * the softmax SUM: sequential j on one lane per row;
//   * the V fold: per (row, d) sequential-j chain across ascending chunks
//     with the `w == 0.0f` skip. Keys past a row's causal limit stage a 0
//     weight and take the same skip — the accumulator chain is untouched,
//     exactly as if the iteration never ran.
//
// => BIT-IDENTICAL per (row, head) to gqa_attention_batch_{f32,h} (and so
// to v2). Gate: to_bits equality, both dtypes, staircase/tail/deep-ctx
// regimes. Host contract: head_dim <= ATTN_V2_HDMAX only (no ctx bound).
#define ATTN_V3_THREADS 128
#define ATTN_V3_BQ 8
#define ATTN_V3_KCH 32

template <class C>
static __device__ __forceinline__ void gqa_attention_batch_v3_body(
    const float* __restrict__ q, const typename C::T* __restrict__ k,
    const typename C::T* __restrict__ v, float* __restrict__ out,
    float* __restrict__ scores, const int ctx_max, const int n_head,
    const int n_head_kv, const int head_dim, const float scale,
    const int causal_offset, const int m) {
  const int h = blockIdx.x;
  const int row0 = blockIdx.y * ATTN_V3_BQ;
  const int tid = threadIdx.x;
  const int nrows = min(ATTN_V3_BQ, m - row0);
  if (nrows <= 0 || h >= n_head) return;  // blockIdx-uniform: no barrier hazard
  const int n_rep = n_head / n_head_kv;
  const int kv = h / n_rep;
  // One past the highest key any row of this block attends.
  const int ctx_top = causal_offset + row0 + nrows;

  // K/V chunk stage (reused between phases 1 and 3), the BQ query rows, and
  // the per-chunk score strip phase 3 reads broadcast from shared.
  __shared__ __align__(16) float s_kv[ATTN_V3_KCH * (ATTN_V2_HDMAX + 1)];
  __shared__ __align__(16) float s_q[ATTN_V3_BQ * ATTN_V2_HDMAX];
  __shared__ float s_w[ATTN_V3_BQ][ATTN_V3_KCH + 1];

  for (int idx = tid; idx < nrows * head_dim; idx += ATTN_V3_THREADS) {
    const int r = idx / head_dim;
    const int d = idx - r * head_dim;
    s_q[r * ATTN_V2_HDMAX + d] =
        q[((long long)(row0 + r) * n_head + h) * head_dim + d];
  }
  __syncthreads();

  const int kstride = head_dim + 1;

  // Phase 1: scaled dots. Each staged KCH-key chunk feeds all BQ rows:
  // thread t owns key t&31 for rows t>>5 and (t>>5)+4 (4 threads share a
  // key row -> broadcast shared reads; 32 threads share a query row ->
  // broadcast s_q reads). Keys past a row's causal limit write nothing.
  for (int c0 = 0; c0 < ctx_top; c0 += ATTN_V3_KCH) {
    const int nk = min(ATTN_V3_KCH, ctx_top - c0);
    for (int idx = tid; idx < nk * head_dim; idx += ATTN_V3_THREADS) {
      const int j = idx / head_dim;
      const int d = idx - j * head_dim;
      s_kv[j * kstride + d] =
          C::load(&k[((long long)(c0 + j) * n_head_kv + kv) * head_dim + d]);
    }
    __syncthreads();
    const int key = tid & 31;
    if (key < nk) {
      const float* kr = &s_kv[key * kstride];
      const int jglob = c0 + key;
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        const int r = (tid >> 5) + rr * 4;
        if (r < nrows && jglob <= causal_offset + row0 + r) {
          const float* qr = &s_q[r * ATTN_V2_HDMAX];
          float dot = 0.0f;
          for (int d = 0; d < head_dim; ++d) {
            dot = __fadd_rn(dot, __fmul_rn(qr[d], kr[d]));
          }
          scores[((long long)(row0 + r) * n_head + h) * ctx_max + jglob] =
              __fmul_rn(dot, scale);
        }
      }
    }
    __syncthreads();
  }

  // Phase 2: per-row softmax; warp w owns rows w and w+4. Butterfly fmaxf
  // (exact any order), elementwise exp, ORDER-PINNED sequential sum on
  // lane 0, elementwise inv scale.
  const int warp = tid >> 5;
  const int lane = tid & 31;
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    const int r = warp + rr * 4;
    if (r < nrows) {
      float* sc = scores + ((long long)(row0 + r) * n_head + h) * ctx_max;
      const int ctx = causal_offset + row0 + r + 1;
      float local = -INFINITY;
      for (int j = lane; j < ctx; j += 32) local = fmaxf(local, sc[j]);
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        local = fmaxf(local, __shfl_xor_sync(0xffffffffu, local, off));
      }
      const float mx = local;
      for (int j = lane; j < ctx; j += 32) {
        sc[j] = exp_f32(__fsub_rn(sc[j], mx));
      }
      __syncwarp();
      float inv = 0.0f;
      if (lane == 0) {
        float sum = 0.0f;
        for (int j = 0; j < ctx; ++j) {
          sum = __fadd_rn(sum, sc[j]);
        }
        inv = __fdiv_rn(1.0f, sum);
      }
      inv = __shfl_sync(0xffffffffu, inv, 0);
      for (int j = lane; j < ctx; j += 32) {
        sc[j] = __fmul_rn(sc[j], inv);
      }
    }
  }
  __syncthreads();

  // Phase 3: V fold. Per chunk: stage V into the reused s_kv, stage the
  // chunk's normalized weights for all rows into s_w (0 past a row's causal
  // limit), then thread d folds every row — one V load feeds BQ chains, each
  // chain j-ascending with the rev-1 zero-skip.
  float acc[ATTN_V3_BQ];
#pragma unroll
  for (int r = 0; r < ATTN_V3_BQ; ++r) acc[r] = 0.0f;
  for (int c0 = 0; c0 < ctx_top; c0 += ATTN_V3_KCH) {
    const int nk = min(ATTN_V3_KCH, ctx_top - c0);
    for (int idx = tid; idx < nk * head_dim; idx += ATTN_V3_THREADS) {
      const int j = idx / head_dim;
      const int d = idx - j * head_dim;
      s_kv[j * kstride + d] =
          C::load(&v[((long long)(c0 + j) * n_head_kv + kv) * head_dim + d]);
    }
    for (int idx = tid; idx < nrows * nk; idx += ATTN_V3_THREADS) {
      const int r = idx / nk;
      const int jj = idx - r * nk;
      const int jglob = c0 + jj;
      s_w[r][jj] = (jglob <= causal_offset + row0 + r)
          ? scores[((long long)(row0 + r) * n_head + h) * ctx_max + jglob]
          : 0.0f;
    }
    __syncthreads();
    if (tid < head_dim) {
      const int d = tid;
      for (int jj = 0; jj < nk; ++jj) {
        const float vv = s_kv[jj * kstride + d];
#pragma unroll
        for (int r = 0; r < ATTN_V3_BQ; ++r) {
          // Rows >= nrows read UNSTAGED s_w garbage here — safe by design:
          // their acc[r] can only poison a register that the nrows-guarded
          // store below never emits (no FP traps on this path).
          const float w = s_w[r][jj];
          if (w != 0.0f) {
            acc[r] = __fadd_rn(acc[r], __fmul_rn(w, vv));
          }
        }
      }
    }
    __syncthreads();  // fold reads done before the next chunk restages s_kv/s_w
  }
  if (tid < head_dim) {
#pragma unroll
    for (int r = 0; r < ATTN_V3_BQ; ++r) {
      if (r < nrows) {
        out[((long long)(row0 + r) * n_head + h) * head_dim + tid] = acc[r];
      }
    }
  }
}


template <class C>
static __device__ __forceinline__ void lm_head_warp_body(
    const float* __restrict__ h, const typename C::T* __restrict__ embd,
                                 const int n_embd, const int vocab, float* __restrict__ logits) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= vocab) return;
  const typename C::T* row = embd + (long long)warp * n_embd;
  float acc = 0.0f;
  for (int k = lane; k < n_embd; k += 32) {
    acc = __fadd_rn(acc, __fmul_rn(h[k], C::load(&row[k])));  // coalesced across the warp
  }
  // Warp-shuffle tree reduction (reorders the sum vs the host's sequential fold).
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffff, acc, off);
  }
  if (lane == 0) logits[warp] = acc;
}

extern "C" {

__global__ void gqa_attention_scores_g(const float* __restrict__ q,
                                       const float* __restrict__ k,
                                       float* __restrict__ scores,
                                       const int* __restrict__ ctrl,
                                       const int max_ctx, const int n_head,
                                       const int n_head_kv, const int head_dim,
                                       const float scale) {
  gqa_attention_scores_body<KvLoadF32>(q, k, scores, ctrl, max_ctx, n_head,
                                       n_head_kv, head_dim, scale);
}

__global__ void gqa_attention_reduce_g(const float* __restrict__ v,
                                       const float* __restrict__ scores,
                                       float* __restrict__ out,
                                       const int* __restrict__ ctrl,
                                       const int max_ctx, const int n_head,
                                       const int n_head_kv, const int head_dim) {
  gqa_attention_reduce_body<KvLoadF32>(v, scores, out, ctrl, max_ctx, n_head,
                                       n_head_kv, head_dim);
}

// ─── Split tree-verify attention (v1.x) — the verify-path twin of the decode
// split pair. The single-kernel gqa_attention_tree_f32 above inherits the
// latency-bound per-warp pattern (sequential scalar dots, softmax on lane 0,
// weighted sum behind the w==0 branch) — measured ~150µs/layer at verify
// context, i.e. ~4.5ms of attention per tree verify. The split keeps every
// f32 fold order (per-key d-order dots via in-vector-ordered float4 chains,
// j-order softmax sum on one thread, j-order weighted chains) so outputs stay
// bit-identical; requires head_dim % 4 == 0 (the host routes other geometries
// to the single kernel). Tree verifies launch eagerly, so grids are exact per
// call — no ctrl-driven early-exit machinery needed.

// Keep in sync with the host-side const in `bl_attn_tree_split` (cuda.rs).

// Scores fan-out: grid (m·n_head, ceil(max_row_ctx/TREE_SCORE_CHUNK)), one
// warp per block; lane owns keys chunk+lane, +32, +64, +96 as independent
// float4 chains. Key j maps to arena slot j < prefix_len ? j : anc[row][j-prefix].
__global__ void gqa_attention_tree_scores_g(
    const float* __restrict__ q, const float* __restrict__ k,
    float* __restrict__ scores, const int* __restrict__ anc,
    const int* __restrict__ n_anc, const int ctx_max, const int n_head,
    const int n_head_kv, const int head_dim, const float scale,
    const int prefix_len, const int max_anc, const int m) {
  gqa_attention_tree_scores_body<KvLoadF32>(q, k, scores, anc, n_anc, ctx_max, n_head, n_head_kv, head_dim, scale, prefix_len,
      max_anc, m);
}

// Per-(row, head) softmax + weighted sum: 128-thread block, scores staged in
// dynamic shared (ctx_max·4 bytes ≤ the same opt-in budget as decode), tree
// max (exact), parallel exp_f32, the ONE rounded fold sequential on thread 0
// in j-order, weighted sum one-dim-per-thread with the slot indirection and
// loads hoisted out of the w==0 skip.
// REQUIRES blockDim.x == 128 (the max-fold levels below hardcode 64/32; the
// host launches comply). s_red[256] is headroom, not configurability.
__global__ void gqa_attention_tree_reduce_g(
    const float* __restrict__ v, const float* __restrict__ scores,
    float* __restrict__ out, const int* __restrict__ anc,
    const int* __restrict__ n_anc, const int ctx_max, const int n_head,
    const int n_head_kv, const int head_dim, const int prefix_len,
    const int max_anc, const int m) {
  gqa_attention_tree_reduce_body<KvLoadF32>(v, scores, out, anc, n_anc, ctx_max, n_head, n_head_kv, head_dim, prefix_len,
      max_anc, m);
}

// ─── Ctrl-driven tree-verify kernels (v1.x) — graph-capturable twins of the
// tree trunk's shape-varying launches. A CUDA graph bakes every scalar arg and
// grid at capture, but a spec-decode loop calls verify with a growing prefix
// and a varying node count; these variants read the per-call values from
// `tree_ctrl` = [prefix_len, real_m, kv_row_base] (device i32[3], uploaded
// before replay) so ONE graph per padded tree size (bucket) serves every
// verify. Pad rows (real_m <= row < m) are early-exited in attention (the
// expensive per-row cost) and left to compute harmless junk elsewhere — pads
// are root-token duplicates, and every trunk op is row-independent, so real
// rows are bit-identical to the eager path.
//
// `kv_row_base` (I2, L3 batch-slot spec decode): a TOKEN-ROW offset added to
// every KV arena row index. 0 = the single-sequence cache (bit-identical to
// the pre-I2 two-word ctrl); r·max_ctx targets dense batch slot r of a
// BatchKv arena, so one captured graph serves EVERY slot (pointers are baked
// at capture; the base is per-replay data). The `anc` table stays REGION-
// LOCAL (slots cache_len+i); the kernels add the base at the KV access.

// kv_append_tree_g — append m provisional K/V rows at arena rows
// [ctrl[2] + ctrl[0], ctrl[2] + ctrl[0] + m). Pad rows write junk into dead
// region space (rows at or above the post-accept watermark are never read and
// get overwritten); the bucket guard keeps ctrl[0] + m inside the region, so
// pads can never spill into a neighbouring batch slot.
__global__ void kv_append_tree_g(const float* __restrict__ src, float* __restrict__ kv_base,
                                 const int* __restrict__ tree_ctrl, const int kv_width,
                                 const int m) {
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long total = (long long)m * kv_width;
  if (idx >= total) return;
  const int prefix_len = tree_ctrl[0];
  const int row_base = tree_ctrl[2];
  const int row = (int)(idx / kv_width);
  const int col = (int)(idx - (long long)row * kv_width);
  kv_base[((long long)row_base + prefix_len + row) * kv_width + col] = src[idx];
}

// gqa_attention_tree_scores_ctrl_g — the scores fan-out with prefix_len from
// ctrl and a FIXED score stride (max_ctx, baked) instead of the eager path's
// per-call ctx_max. The stride only addresses scratch — values are identical.
// grid.y is baked at ceil(max_ctx / TREE_SCORE_CHUNK); chunks past the live
// context exit like the decode split's ctrl-driven grid does.
__global__ void gqa_attention_tree_scores_ctrl_g(
    const float* __restrict__ q, const float* __restrict__ k, float* __restrict__ scores,
    const int* __restrict__ anc, const int* __restrict__ n_anc,
    const int* __restrict__ tree_ctrl, const int score_stride, const int n_head,
    const int n_head_kv, const int head_dim, const float scale, const int max_anc,
    const int m) {
  const int rowhead = blockIdx.x;
  const int row = rowhead / n_head;
  const int h = rowhead - row * n_head;
  if (row >= tree_ctrl[1]) return;  // pad row (uniform per block)
  const int prefix_len = tree_ctrl[0];
  const int row_base = tree_ctrl[2];
  const int ctx = prefix_len + n_anc[row];
  const int chunk = blockIdx.y * TREE_SCORE_CHUNK;
  if (chunk >= ctx) return;
  const int lane = threadIdx.x & 31;
  const int n_rep = n_head / n_head_kv;
  const int kv = h / n_rep;
  const float4* qv = (const float4*)(q + ((long long)row * n_head + h) * head_dim);
  const int hd4 = head_dim >> 2;
  const int* arow = anc + (long long)row * max_anc;
  float* sc = scores + ((long long)row * n_head + h) * score_stride;

  const int j0 = chunk + lane;
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    const int j = j0 + 32 * t;
    if (j >= ctx) break;
    const int slot = row_base + ((j < prefix_len) ? j : arow[j - prefix_len]);
    const float4* kr = (const float4*)(k + ((long long)slot * n_head_kv + kv) * head_dim);
    float dot = 0.0f;
#pragma unroll 8
    for (int d = 0; d < hd4; ++d) {
      const float4 qq = qv[d];
      const float4 a = kr[d];
      dot = __fadd_rn(dot, __fmul_rn(qq.x, a.x));
      dot = __fadd_rn(dot, __fmul_rn(qq.y, a.y));
      dot = __fadd_rn(dot, __fmul_rn(qq.z, a.z));
      dot = __fadd_rn(dot, __fmul_rn(qq.w, a.w));
    }
    sc[j] = __fmul_rn(dot, scale);
  }
}

// gqa_attention_tree_reduce_ctrl_g — the per-(row, head) softmax + weighted
// reduce with prefix_len from ctrl + fixed score stride. REQUIRES
// blockDim.x == 128 (fold levels hardcode 64/32). Every f32 fold keeps the
// canonical order of the non-ctrl twin; the pad-row early return precedes the
// first barrier and is uniform per block.
__global__ void gqa_attention_tree_reduce_ctrl_g(
    const float* __restrict__ v, const float* __restrict__ scores, float* __restrict__ out,
    const int* __restrict__ anc, const int* __restrict__ n_anc,
    const int* __restrict__ tree_ctrl, const int score_stride, const int n_head,
    const int n_head_kv, const int head_dim, const int max_anc, const int m) {
  extern __shared__ float sc[];
  __shared__ float s_red[256];
  __shared__ float s_inv;
  const int rowhead = blockIdx.x;
  const int row = rowhead / n_head;
  const int h = rowhead - row * n_head;
  if (row >= tree_ctrl[1]) return;  // pad row (uniform per block)
  const int prefix_len = tree_ctrl[0];
  const int row_base = tree_ctrl[2];
  const int ctx = prefix_len + n_anc[row];
  const int tid = threadIdx.x;
  const int n_rep = n_head / n_head_kv;
  const int kv = h / n_rep;
  const int* arow = anc + (long long)row * max_anc;

  const float* gsc = scores + ((long long)row * n_head + h) * score_stride;
  for (int j = tid; j < ctx; j += blockDim.x) {
    sc[j] = gsc[j];
  }
  __syncthreads();

  float mx = -INFINITY;
  for (int j = tid; j < ctx; j += blockDim.x) {
    mx = fmaxf(mx, sc[j]);
  }
  s_red[tid] = mx;
  __syncthreads();
  if (tid < 64) {
    s_red[tid] = fmaxf(s_red[tid], s_red[tid + 64]);
  }
  __syncthreads();
  if (tid < 32) {
    float vv = fmaxf(s_red[tid], s_red[tid + 32]);
    for (int off = 16; off > 0; off >>= 1) {
      vv = fmaxf(vv, __shfl_down_sync(0xffffffffu, vv, off));
    }
    if (tid == 0) {
      s_red[0] = vv;
    }
  }
  __syncthreads();
  mx = s_red[0];

  for (int j = tid; j < ctx; j += blockDim.x) {
    sc[j] = exp_f32(__fsub_rn(sc[j], mx));
  }
  __syncthreads();
  if (tid == 0) {
    float sum = 0.0f;
    for (int j = 0; j < ctx; ++j) {
      sum = __fadd_rn(sum, sc[j]);
    }
    s_inv = __fdiv_rn(1.0f, sum);
  }
  __syncthreads();
  const float inv = s_inv;
  for (int j = tid; j < ctx; j += blockDim.x) {
    sc[j] = __fmul_rn(sc[j], inv);
  }
  __syncthreads();

  float* o_row = out + ((long long)row * n_head + h) * head_dim;
  const int ndims = (head_dim > tid) ? (head_dim - tid + (int)blockDim.x - 1) / (int)blockDim.x : 0;
#define TREE_MAX_DIMS_PER_THREAD 8
  if (ndims <= TREE_MAX_DIMS_PER_THREAD) {
    float acc[TREE_MAX_DIMS_PER_THREAD];
    for (int t = 0; t < ndims; ++t) {
      acc[t] = 0.0f;
    }
#pragma unroll 4
    for (int j = 0; j < ctx; ++j) {
      const float w = sc[j];
      const int slot = row_base + ((j < prefix_len) ? j : arow[j - prefix_len]);
      const float* v_row = v + ((long long)slot * n_head_kv + kv) * head_dim;
      float vals[TREE_MAX_DIMS_PER_THREAD];
#pragma unroll
      for (int t = 0; t < TREE_MAX_DIMS_PER_THREAD; ++t) {
        if (t < ndims) {
          vals[t] = v_row[tid + (int)blockDim.x * t];
        }
      }
      if (w != 0.0f) {
#pragma unroll
        for (int t = 0; t < TREE_MAX_DIMS_PER_THREAD; ++t) {
          if (t < ndims) {
            acc[t] = __fadd_rn(acc[t], __fmul_rn(w, vals[t]));
          }
        }
      }
    }
    for (int t = 0; t < ndims; ++t) {
      o_row[tid + (int)blockDim.x * t] = acc[t];
    }
  } else {
    for (int d = tid; d < head_dim; d += blockDim.x) {
      float acc = 0.0f;
      for (int j = 0; j < ctx; ++j) {
        const float w = sc[j];
        if (w == 0.0f) continue;
        const int slot = row_base + ((j < prefix_len) ? j : arow[j - prefix_len]);
        const float* v_row = v + ((long long)slot * n_head_kv + kv) * head_dim;
        acc = __fadd_rn(acc, __fmul_rn(w, v_row[d]));
      }
      o_row[d] = acc;
    }
  }
#undef TREE_MAX_DIMS_PER_THREAD
}

// ─── f16 KV-cache twins (ADR 0020, rung 1) — every kernel that touches the
// single-sequence KV arenas gets an `_h` twin whose ONLY change is the KV
// element type: stores round once via __float2half_rn, loads widen via
// __half2float, and every f32 fold keeps its order. The f32 originals are
// untouched (the default path stays bit-exact); an f16 model instance loads
// the `_h` names into the same function handles, so launch sites don't
// change. Verify trees route to the EAGER path under f16 (the ctrl twins are
// deliberately not duplicated — the graph measured ≈ no wall-clock win).

// 8-byte load of 4 halves → float4 (the f16 twin of a float4 KV read).

__global__ void kv_append_h(const float* __restrict__ src,
                            __half* __restrict__ kv_base,  // [max_ctx*kv_width]
                            const int* __restrict__ ctrl,
                            const int kv_width) {
  kv_append_body<KvStoreF16>(src, kv_base, ctrl, kv_width);
}

__global__ void kv_append_batch_h(const float* __restrict__ src, __half* __restrict__ kv_base,
                                  const int cache_len, const int kv_width, const int m) {
  kv_append_batch_body<KvStoreF16>(src, kv_base, cache_len, kv_width, m);
}

__global__ void rope_kv_fused_h(float* __restrict__ q,
                                const float* __restrict__ k,
                                const float* __restrict__ v,
                                __half* __restrict__ kv_k_base,
                                __half* __restrict__ kv_v_base,
                                const float* __restrict__ cos_table,
                                const float* __restrict__ sin_table,
                                const int* __restrict__ ctrl,
                                const int n_head, const int n_head_kv,
                                const int head_dim, const int kv_width) {
  rope_kv_fused_body<KvStoreF16>(q, k, v, kv_k_base, kv_v_base, cos_table,
                                 sin_table, ctrl, n_head, n_head_kv, head_dim,
                                 kv_width);
}

__global__ void gqa_attention_scores_h(const float* __restrict__ q,
                                       const __half* __restrict__ k,
                                       float* __restrict__ scores,
                                       const int* __restrict__ ctrl,
                                       const int max_ctx, const int n_head,
                                       const int n_head_kv, const int head_dim,
                                       const float scale) {
  gqa_attention_scores_body<KvLoadF16>(q, k, scores, ctrl, max_ctx, n_head,
                                       n_head_kv, head_dim, scale);
}

// REQUIRES blockDim.x == 128 (same contract as gqa_attention_reduce_g).
__global__ void gqa_attention_reduce_h(const __half* __restrict__ v,
                                       const float* __restrict__ scores,
                                       float* __restrict__ out,
                                       const int* __restrict__ ctrl,
                                       const int max_ctx, const int n_head,
                                       const int n_head_kv, const int head_dim) {
  gqa_attention_reduce_body<KvLoadF16>(v, scores, out, ctrl, max_ctx, n_head,
                                       n_head_kv, head_dim);
}

__global__ void gqa_attention_batch_h(const float* __restrict__ q, const __half* __restrict__ k,
                                      const __half* __restrict__ v, float* __restrict__ out,
                                      float* __restrict__ scores, const int ctx_max,
                                      const int n_head, const int n_head_kv, const int head_dim,
                                      const float scale, const int causal_offset, const int m) {
  gqa_attention_batch_body<KvLoadF16>(q, k, v, out, scores, ctx_max, n_head, n_head_kv, head_dim, scale,
      causal_offset, m);
}

// gqa_attention_batch_v2_h — f16-KV twin of gqa_attention_batch_v2_f32 (the
// C::load conversion happens at the shared stage; values are identical).
__global__ void __launch_bounds__(ATTN_V2_THREADS) gqa_attention_batch_v2_h(
    const float* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
    float* __restrict__ out, const int n_head, const int n_head_kv, const int head_dim,
    const float scale, const int causal_offset, const int m) {
  gqa_attention_batch_v2_body<KvLoadF16>(q, k, v, out, n_head, n_head_kv, head_dim, scale,
      causal_offset, m);
}

// gqa_attention_batch_v3_h — f16-KV twin of gqa_attention_batch_v3_f32.
__global__ void __launch_bounds__(ATTN_V3_THREADS) gqa_attention_batch_v3_h(
    const float* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
    float* __restrict__ out, float* __restrict__ scores, const int ctx_max, const int n_head,
    const int n_head_kv, const int head_dim, const float scale, const int causal_offset,
    const int m) {
  gqa_attention_batch_v3_body<KvLoadF16>(q, k, v, out, scores, ctx_max, n_head, n_head_kv,
      head_dim, scale, causal_offset, m);
}

__global__ void gqa_attention_tree_scores_h(
    const float* __restrict__ q, const __half* __restrict__ k,
    float* __restrict__ scores, const int* __restrict__ anc,
    const int* __restrict__ n_anc, const int ctx_max, const int n_head,
    const int n_head_kv, const int head_dim, const float scale,
    const int prefix_len, const int max_anc, const int m) {
  gqa_attention_tree_scores_body<KvLoadF16>(q, k, scores, anc, n_anc, ctx_max, n_head, n_head_kv, head_dim, scale, prefix_len,
      max_anc, m);
}

// REQUIRES blockDim.x == 128 (same contract as gqa_attention_tree_reduce_g).
__global__ void gqa_attention_tree_reduce_h(
    const __half* __restrict__ v, const float* __restrict__ scores,
    float* __restrict__ out, const int* __restrict__ anc,
    const int* __restrict__ n_anc, const int ctx_max, const int n_head,
    const int n_head_kv, const int head_dim, const int prefix_len,
    const int max_anc, const int m) {
  gqa_attention_tree_reduce_body<KvLoadF16>(v, scores, out, anc, n_anc, ctx_max, n_head, n_head_kv, head_dim, prefix_len,
      max_anc, m);
}

// ─── i8 KV-cache twins (ADR 0020, rung 2) — per-(token, kv-head, KV_QGROUP-
// dim group) DYNAMIC scales computed at append (absmax/127, the A8 activation
// recipe); attention loads dequant to f32 (`(float)k8 * scale_g`) and then run
// the IDENTICAL fold chains as the f32/f16 kernels. Scales live in a side
// arena `[max_ctx, n_head_kv, head_dim/KV_QGROUP]` f32, passed as a TRAILING
// param so the dtype-selected launch sites share a common arg prefix.
#define KV_QGROUP 64

// Per-row append + group-quantize: ONE BLOCK PER ROW (grid.x = m); shared
// absmax reduction over each contiguous KV_QGROUP run of the row.
__device__ __forceinline__ void kv_quant_row_q8(const float* __restrict__ src_row,
                                                signed char* __restrict__ dst_row,
                                                float* __restrict__ sc_row,
                                                const int kv_width) {
  __shared__ float s_absmax[64]; // up to 64 groups per row (kv_width ≤ 4096)
  const int n_groups = kv_width / KV_QGROUP;
  for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
    s_absmax[g] = 0.0f;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < kv_width; i += blockDim.x) {
    const float a = fabsf(src_row[i]);
    // Non-negative floats compare correctly as ints — atomicMax on the bits.
    atomicMax((int*)&s_absmax[i / KV_QGROUP], __float_as_int(a));
  }
  __syncthreads();
  for (int i = threadIdx.x; i < kv_width; i += blockDim.x) {
    const int g = i / KV_QGROUP;
    const float am = s_absmax[g];
    if (am == 0.0f) {
      dst_row[i] = 0;
      if (i % KV_QGROUP == 0) sc_row[g] = 0.0f;
    } else {
      const float scale = am / 127.0f;
      float q = roundf(src_row[i] / scale);
      q = fminf(127.0f, fmaxf(-127.0f, q));
      dst_row[i] = (signed char)q;
      if (i % KV_QGROUP == 0) sc_row[g] = scale;
    }
  }
}

// ─── Ternary KV experiment ("KVTQ", ADR 0020 rung 3) — values quantize to
// {-s, 0, +s} per KV_QGROUP group and ride the SAME i8 lattice + scale arena
// as rung 2 (code = trit·127, scale = fl(s/127) → dequant = trit·fl(s/127),
// within a ulp of trit·s), so
// the attention kernels are the unmodified _q8 ones; only the append rounding
// differs. Level s = 1.5·(group absmean) with threshold s/2 — near the
// MSE-optimal 3-level quantizer for zero-mean Gaussian data (level ≈ 1.53·E|v|).
__device__ __forceinline__ void kv_quant_row_t2(const float* __restrict__ src_row,
                                                signed char* __restrict__ dst_row,
                                                float* __restrict__ sc_row,
                                                const int kv_width) {
  // group Σ|v| — NOTE: float atomicAdd is order-nondeterministic, so t2
  // appends are not bit-reproducible run-to-run (near-threshold trits can
  // flip); acceptable for the rejected-experiment harness. kv_width ≤
  // 64·KV_QGROUP is build-guarded.
  __shared__ float s_sum[64];
  const int n_groups = kv_width / KV_QGROUP;
  for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
    s_sum[g] = 0.0f;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < kv_width; i += blockDim.x) {
    atomicAdd(&s_sum[i / KV_QGROUP], fabsf(src_row[i]));
  }
  __syncthreads();
  for (int i = threadIdx.x; i < kv_width; i += blockDim.x) {
    const int g = i / KV_QGROUP;
    const float level = 1.5f * (s_sum[g] / (float)KV_QGROUP);
    if (level == 0.0f) {
      dst_row[i] = 0;
      if (i % KV_QGROUP == 0) sc_row[g] = 0.0f;
    } else {
      const float v = src_row[i];
      const signed char trit = (fabsf(v) > 0.5f * level) ? (v > 0.0f ? 1 : -1) : 0;
      dst_row[i] = (signed char)(trit * 127);
      if (i % KV_QGROUP == 0) sc_row[g] = level / 127.0f;
    }
  }
}

__global__ void kv_append_t2(const float* __restrict__ src,
                             signed char* __restrict__ kv_base,
                             const int* __restrict__ ctrl,
                             const int kv_width,
                             float* __restrict__ scales) {
  const long long row = ctrl[2];
  const int n_groups = kv_width / KV_QGROUP;
  kv_quant_row_t2(src, kv_base + row * kv_width, scales + row * n_groups, kv_width);
}

__global__ void kv_append_batch_t2(const float* __restrict__ src,
                                   signed char* __restrict__ kv_base,
                                   const int cache_len, const int kv_width, const int m,
                                   float* __restrict__ scales) {
  const int r = blockIdx.x;
  if (r >= m) return;
  const long long row = (long long)cache_len + r;
  const int n_groups = kv_width / KV_QGROUP;
  kv_quant_row_t2(src + (long long)r * kv_width, kv_base + row * kv_width,
                  scales + row * n_groups, kv_width);
}

__global__ void rope_kv_fused_t2(float* __restrict__ q,
                                 const float* __restrict__ k,
                                 const float* __restrict__ v,
                                 signed char* __restrict__ kv_k_base,
                                 signed char* __restrict__ kv_v_base,
                                 const float* __restrict__ cos_table,
                                 const float* __restrict__ sin_table,
                                 const int* __restrict__ ctrl,
                                 const int n_head, const int n_head_kv,
                                 const int head_dim, const int kv_width,
                                 float* __restrict__ k_scales,
                                 float* __restrict__ v_scales) {
  extern __shared__ float s_kv[];
  const int half = head_dim >> 1;
  const int pos = ctrl[1];
  const long long row = ctrl[2];
  if (blockIdx.x == 0) {
    const int q_total = n_head * half;
    for (int idx = threadIdx.x; idx < q_total; idx += blockDim.x) {
      const int head = idx / half;
      const int j = idx - head * half;
      const int base = head * head_dim;
      const float c = cos_table[(long long)pos * half + j];
      const float s = sin_table[(long long)pos * half + j];
      const float a = q[base + j];
      const float b = q[base + j + half];
      q[base + j] = __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
      q[base + j + half] = __fadd_rn(__fmul_rn(b, c), __fmul_rn(a, s));
    }
    return;
  }
  float* s_k = s_kv;
  float* s_v = s_kv + kv_width;
  const int k_total = n_head_kv * half;
  for (int t = threadIdx.x; t < k_total; t += blockDim.x) {
    const int head = t / half;
    const int j = t - head * half;
    const int base = head * head_dim;
    const float c = cos_table[(long long)pos * half + j];
    const float s = sin_table[(long long)pos * half + j];
    const float a = k[base + j];
    const float b = k[base + j + half];
    s_k[base + j] = __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
    s_k[base + j + half] = __fadd_rn(__fmul_rn(b, c), __fmul_rn(a, s));
  }
  for (int i = threadIdx.x; i < kv_width; i += blockDim.x) {
    s_v[i] = v[i];
  }
  __syncthreads();
  const int n_groups = kv_width / KV_QGROUP;
  kv_quant_row_t2(s_k, kv_k_base + row * kv_width, k_scales + row * n_groups, kv_width);
  __syncthreads();
  kv_quant_row_t2(s_v, kv_v_base + row * kv_width, v_scales + row * n_groups, kv_width);
}

__global__ void kv_append_q8(const float* __restrict__ src,
                             signed char* __restrict__ kv_base,
                             const int* __restrict__ ctrl,
                             const int kv_width,
                             float* __restrict__ scales) {
  const long long row = ctrl[2];
  const int n_groups = kv_width / KV_QGROUP;
  kv_quant_row_q8(src, kv_base + row * kv_width, scales + row * n_groups, kv_width);
}

__global__ void kv_append_batch_q8(const float* __restrict__ src,
                                   signed char* __restrict__ kv_base,
                                   const int cache_len, const int kv_width, const int m,
                                   float* __restrict__ scales) {
  const int r = blockIdx.x;
  if (r >= m) return;
  const long long row = (long long)cache_len + r;
  const int n_groups = kv_width / KV_QGROUP;
  kv_quant_row_q8(src + (long long)r * kv_width, kv_base + row * kv_width,
                  scales + row * n_groups, kv_width);
}

// grid (2,1,1): block 0 rotates q in place; block 1 stages rotated k + v in
// dynamic shared (2·kv_width floats), then group-quantizes both into the
// arenas. Rotation math identical to rope_kv_fused_g.
__global__ void rope_kv_fused_q8(float* __restrict__ q,
                                 const float* __restrict__ k,
                                 const float* __restrict__ v,
                                 signed char* __restrict__ kv_k_base,
                                 signed char* __restrict__ kv_v_base,
                                 const float* __restrict__ cos_table,
                                 const float* __restrict__ sin_table,
                                 const int* __restrict__ ctrl,
                                 const int n_head, const int n_head_kv,
                                 const int head_dim, const int kv_width,
                                 float* __restrict__ k_scales,
                                 float* __restrict__ v_scales) {
  extern __shared__ float s_kv[]; // [kv_width] rotated k, then [kv_width] v
  const int half = head_dim >> 1;
  const int pos = ctrl[1];
  const long long row = ctrl[2];
  if (blockIdx.x == 0) {
    const int q_total = n_head * half;
    for (int idx = threadIdx.x; idx < q_total; idx += blockDim.x) {
      const int head = idx / half;
      const int j = idx - head * half;
      const int base = head * head_dim;
      const float c = cos_table[(long long)pos * half + j];
      const float s = sin_table[(long long)pos * half + j];
      const float a = q[base + j];
      const float b = q[base + j + half];
      q[base + j] = __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
      q[base + j + half] = __fadd_rn(__fmul_rn(b, c), __fmul_rn(a, s));
    }
    return;
  }
  // Block 1: rotate k into shared, copy v into shared.
  float* s_k = s_kv;
  float* s_v = s_kv + kv_width;
  const int k_total = n_head_kv * half;
  for (int t = threadIdx.x; t < k_total; t += blockDim.x) {
    const int head = t / half;
    const int j = t - head * half;
    const int base = head * head_dim;
    const float c = cos_table[(long long)pos * half + j];
    const float s = sin_table[(long long)pos * half + j];
    const float a = k[base + j];
    const float b = k[base + j + half];
    s_k[base + j] = __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
    s_k[base + j + half] = __fadd_rn(__fmul_rn(b, c), __fmul_rn(a, s));
  }
  for (int i = threadIdx.x; i < kv_width; i += blockDim.x) {
    s_v[i] = v[i];
  }
  __syncthreads();
  const int n_groups = kv_width / KV_QGROUP;
  kv_quant_row_q8(s_k, kv_k_base + row * kv_width, k_scales + row * n_groups, kv_width);
  __syncthreads();
  kv_quant_row_q8(s_v, kv_v_base + row * kv_width, v_scales + row * n_groups, kv_width);
}

// Dequantizing float4-equivalent load: 4 consecutive i8 KV elements (one
// aligned 32-bit read) × the group scale.
__device__ __forceinline__ float4 kvq8_load4(const signed char* p, const float scale) {
  const int w = *(const int*)p;
  return make_float4((float)(signed char)(w & 0xFF) * scale,
                     (float)(signed char)((w >> 8) & 0xFF) * scale,
                     (float)(signed char)((w >> 16) & 0xFF) * scale,
                     (float)(signed char)((w >> 24) & 0xFF) * scale);
}

__global__ void gqa_attention_scores_q8(const float* __restrict__ q,
                                        const signed char* __restrict__ k,
                                        float* __restrict__ scores,
                                        const int* __restrict__ ctrl,
                                        const int max_ctx, const int n_head,
                                        const int n_head_kv, const int head_dim,
                                        const float scale,
                                        const float* __restrict__ k_scales) {
  const int h = blockIdx.x;
  const int chunk = blockIdx.y * SCORE_CHUNK;
  const int lane = threadIdx.x & 31;
  const int ctx = ctrl[2] + 1;
  if (h >= n_head || chunk >= ctx) return;
  const int n_rep = n_head / n_head_kv;
  const int kv = h / n_rep;
  const float4* qv = (const float4*)(q + (long long)h * head_dim);
  const int hd4 = head_dim >> 2;
  const int gph = head_dim / KV_QGROUP; // groups per head
  float* sc = scores + (long long)h * max_ctx;

  const int j0 = chunk + lane;
  const int j1 = j0 + 32;
  const int j2 = j0 + 64;
  const int j3 = j0 + 96;
  const signed char* k0 = k + ((long long)j0 * n_head_kv + kv) * head_dim;
  const signed char* k1 = k + ((long long)j1 * n_head_kv + kv) * head_dim;
  const signed char* k2 = k + ((long long)j2 * n_head_kv + kv) * head_dim;
  const signed char* k3 = k + ((long long)j3 * n_head_kv + kv) * head_dim;
  const float* s0 = k_scales + ((long long)j0 * n_head_kv + kv) * gph;
  const float* s1 = k_scales + ((long long)j1 * n_head_kv + kv) * gph;
  const float* s2 = k_scales + ((long long)j2 * n_head_kv + kv) * gph;
  const float* s3 = k_scales + ((long long)j3 * n_head_kv + kv) * gph;
  if (j3 < ctx) {
    // Fast path: 4 independent per-key chains (memory-level parallelism —
    // the same shape as the f32/f16 decode scores kernels).
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
#pragma unroll 4
    for (int d = 0; d < hd4; ++d) {
      const float4 qq = qv[d];
      const int gi = (4 * d) / KV_QGROUP;
      const float4 a = kvq8_load4(k0 + 4 * d, s0[gi]);
      const float4 b = kvq8_load4(k1 + 4 * d, s1[gi]);
      const float4 c = kvq8_load4(k2 + 4 * d, s2[gi]);
      const float4 e = kvq8_load4(k3 + 4 * d, s3[gi]);
      d0 = __fadd_rn(d0, __fmul_rn(qq.x, a.x));
      d1 = __fadd_rn(d1, __fmul_rn(qq.x, b.x));
      d2 = __fadd_rn(d2, __fmul_rn(qq.x, c.x));
      d3 = __fadd_rn(d3, __fmul_rn(qq.x, e.x));
      d0 = __fadd_rn(d0, __fmul_rn(qq.y, a.y));
      d1 = __fadd_rn(d1, __fmul_rn(qq.y, b.y));
      d2 = __fadd_rn(d2, __fmul_rn(qq.y, c.y));
      d3 = __fadd_rn(d3, __fmul_rn(qq.y, e.y));
      d0 = __fadd_rn(d0, __fmul_rn(qq.z, a.z));
      d1 = __fadd_rn(d1, __fmul_rn(qq.z, b.z));
      d2 = __fadd_rn(d2, __fmul_rn(qq.z, c.z));
      d3 = __fadd_rn(d3, __fmul_rn(qq.z, e.z));
      d0 = __fadd_rn(d0, __fmul_rn(qq.w, a.w));
      d1 = __fadd_rn(d1, __fmul_rn(qq.w, b.w));
      d2 = __fadd_rn(d2, __fmul_rn(qq.w, c.w));
      d3 = __fadd_rn(d3, __fmul_rn(qq.w, e.w));
    }
    sc[j0] = __fmul_rn(d0, scale);
    sc[j1] = __fmul_rn(d1, scale);
    sc[j2] = __fmul_rn(d2, scale);
    sc[j3] = __fmul_rn(d3, scale);
  } else {
    const signed char* krs[4] = {k0, k1, k2, k3};
    const float* kss[4] = {s0, s1, s2, s3};
    const int js[4] = {j0, j1, j2, j3};
    for (int t = 0; t < 4; ++t) {
      if (js[t] >= ctx) break;
      float dot = 0.0f;
#pragma unroll 4
      for (int d = 0; d < hd4; ++d) {
        const float4 qq = qv[d];
        const float4 a = kvq8_load4(krs[t] + 4 * d, kss[t][(4 * d) / KV_QGROUP]);
        dot = __fadd_rn(dot, __fmul_rn(qq.x, a.x));
        dot = __fadd_rn(dot, __fmul_rn(qq.y, a.y));
        dot = __fadd_rn(dot, __fmul_rn(qq.z, a.z));
        dot = __fadd_rn(dot, __fmul_rn(qq.w, a.w));
      }
      sc[js[t]] = __fmul_rn(dot, scale);
    }
  }
}

// REQUIRES blockDim.x == 128 (same contract as gqa_attention_reduce_g).
__global__ void gqa_attention_reduce_q8(const signed char* __restrict__ v,
                                        const float* __restrict__ scores,
                                        float* __restrict__ out,
                                        const int* __restrict__ ctrl,
                                        const int max_ctx, const int n_head,
                                        const int n_head_kv, const int head_dim,
                                        const float* __restrict__ v_scales) {
  extern __shared__ float sc[];
  __shared__ float s_red[256];
  __shared__ float s_inv;
  const int h = blockIdx.x;
  const int tid = threadIdx.x;
  if (h >= n_head) return;
  const int ctx = ctrl[2] + 1;
  const int n_rep = n_head / n_head_kv;
  const int kv = h / n_rep;
  const int gph = head_dim / KV_QGROUP;

  const float* gsc = scores + (long long)h * max_ctx;
  for (int j = tid; j < ctx; j += blockDim.x) {
    sc[j] = gsc[j];
  }
  __syncthreads();

  float mx = -INFINITY;
  for (int j = tid; j < ctx; j += blockDim.x) {
    mx = fmaxf(mx, sc[j]);
  }
  s_red[tid] = mx;
  __syncthreads();
  if (tid < 64) {
    s_red[tid] = fmaxf(s_red[tid], s_red[tid + 64]);
  }
  __syncthreads();
  if (tid < 32) {
    float vv = fmaxf(s_red[tid], s_red[tid + 32]);
    for (int off = 16; off > 0; off >>= 1) {
      vv = fmaxf(vv, __shfl_down_sync(0xffffffffu, vv, off));
    }
    if (tid == 0) {
      s_red[0] = vv;
    }
  }
  __syncthreads();
  mx = s_red[0];

  for (int j = tid; j < ctx; j += blockDim.x) {
    sc[j] = exp_f32(__fsub_rn(sc[j], mx));
  }
  __syncthreads();
  if (tid == 0) {
    float sum = 0.0f;
    for (int j = 0; j < ctx; ++j) {
      sum = __fadd_rn(sum, sc[j]);
    }
    s_inv = __fdiv_rn(1.0f, sum);
  }
  __syncthreads();
  const float inv = s_inv;
  for (int j = tid; j < ctx; j += blockDim.x) {
    sc[j] = __fmul_rn(sc[j], inv);
  }
  __syncthreads();

  // Weighted sum with per-(j, group) dequant. A weight-bank variant (V scales
  // pre-folded into shared per-group weights) was MEASURED SLOWER (386 vs
  // 340 µs at ctx≈4K) — the loop is latency-bound and the extra staging pass
  // cost more than the removed scale loads saved. Keep the simple form.
  float* o_row = out + (long long)h * head_dim;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    const int g = d / KV_QGROUP;
    float acc = 0.0f;
#pragma unroll 8
    for (int j = 0; j < ctx; ++j) {
      const float w = sc[j];
      const float vv = (float)v[((long long)j * n_head_kv + kv) * head_dim + d]
          * v_scales[((long long)j * n_head_kv + kv) * gph + g];
      if (w != 0.0f) {
        acc = __fadd_rn(acc, __fmul_rn(w, vv));
      }
    }
    o_row[d] = acc;
  }
}

__global__ void gqa_attention_batch_q8(const float* __restrict__ q,
                                       const signed char* __restrict__ k,
                                       const signed char* __restrict__ v,
                                       float* __restrict__ out,
                                       float* __restrict__ scores, const int ctx_max,
                                       const int n_head, const int n_head_kv,
                                       const int head_dim, const float scale,
                                       const int causal_offset, const int m,
                                       const float* __restrict__ k_scales,
                                       const float* __restrict__ v_scales) {
  const long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const long long total = (long long)m * n_head;
  if (warp >= total) return;
  const int row = warp / n_head;
  const int h = warp - (long long)row * n_head;
  const int limit = causal_offset + row;
  const int ctx = limit + 1;
  const int n_rep = n_head / n_head_kv;
  const int kv = h / n_rep;
  const int gph = head_dim / KV_QGROUP;
  const float* q_row = q + ((long long)row * n_head + h) * head_dim;
  float* sc = scores + ((long long)row * n_head + h) * ctx_max;

  for (int j = lane; j < ctx; j += 32) {
    const signed char* k_row = k + ((long long)j * n_head_kv + kv) * head_dim;
    const float* ks = k_scales + ((long long)j * n_head_kv + kv) * gph;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      dot = __fadd_rn(dot, __fmul_rn(q_row[d], (float)k_row[d] * ks[d / KV_QGROUP]));
    }
    sc[j] = __fmul_rn(dot, scale);
  }
  __syncwarp();
  if (lane == 0) {
    float mx = -INFINITY;
    for (int j = 0; j < ctx; ++j) {
      if (sc[j] > mx) mx = sc[j];
    }
    float sum = 0.0f;
    for (int j = 0; j < ctx; ++j) {
      const float e = exp_f32(__fsub_rn(sc[j], mx));
      sc[j] = e;
      sum = __fadd_rn(sum, e);
    }
    const float inv = __fdiv_rn(1.0f, sum);
    for (int j = 0; j < ctx; ++j) {
      sc[j] = __fmul_rn(sc[j], inv);
    }
  }
  __syncwarp();
  float* o_row = out + ((long long)row * n_head + h) * head_dim;
  for (int d = lane; d < head_dim; d += 32) {
    const int g = d / KV_QGROUP;
    float acc = 0.0f;
    for (int j = 0; j < ctx; ++j) {
      const float w = sc[j];
      if (w == 0.0f) continue;
      const float vv = (float)v[((long long)j * n_head_kv + kv) * head_dim + d]
          * v_scales[((long long)j * n_head_kv + kv) * gph + g];
      acc = __fadd_rn(acc, __fmul_rn(w, vv));
    }
    o_row[d] = acc;
  }
}

__global__ void gqa_attention_tree_scores_q8(
    const float* __restrict__ q, const signed char* __restrict__ k,
    float* __restrict__ scores, const int* __restrict__ anc,
    const int* __restrict__ n_anc, const int ctx_max, const int n_head,
    const int n_head_kv, const int head_dim, const float scale,
    const int prefix_len, const int max_anc, const int m,
    const float* __restrict__ k_scales) {
  const int rowhead = blockIdx.x;
  const int row = rowhead / n_head;
  const int h = rowhead - row * n_head;
  if (row >= m) return;
  const int ctx = prefix_len + n_anc[row];
  const int chunk = blockIdx.y * TREE_SCORE_CHUNK;
  if (chunk >= ctx) return;
  const int lane = threadIdx.x & 31;
  const int n_rep = n_head / n_head_kv;
  const int kv = h / n_rep;
  const float4* qv = (const float4*)(q + ((long long)row * n_head + h) * head_dim);
  const int hd4 = head_dim >> 2;
  const int gph = head_dim / KV_QGROUP;
  const int* arow = anc + (long long)row * max_anc;
  float* sc = scores + ((long long)row * n_head + h) * ctx_max;

  const int j0 = chunk + lane;
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    const int j = j0 + 32 * t;
    if (j >= ctx) break;
    const int slot = (j < prefix_len) ? j : arow[j - prefix_len];
    const signed char* kr = k + ((long long)slot * n_head_kv + kv) * head_dim;
    const float* ks = k_scales + ((long long)slot * n_head_kv + kv) * gph;
    float dot = 0.0f;
#pragma unroll 8
    for (int d = 0; d < hd4; ++d) {
      const float4 qq = qv[d];
      const float g = ks[(4 * d) / KV_QGROUP];
      const float4 a = kvq8_load4(kr + 4 * d, g);
      dot = __fadd_rn(dot, __fmul_rn(qq.x, a.x));
      dot = __fadd_rn(dot, __fmul_rn(qq.y, a.y));
      dot = __fadd_rn(dot, __fmul_rn(qq.z, a.z));
      dot = __fadd_rn(dot, __fmul_rn(qq.w, a.w));
    }
    sc[j] = __fmul_rn(dot, scale);
  }
}

// REQUIRES blockDim.x == 128 (same contract as gqa_attention_tree_reduce_g).
__global__ void gqa_attention_tree_reduce_q8(
    const signed char* __restrict__ v, const float* __restrict__ scores,
    float* __restrict__ out, const int* __restrict__ anc,
    const int* __restrict__ n_anc, const int ctx_max, const int n_head,
    const int n_head_kv, const int head_dim, const int prefix_len,
    const int max_anc, const int m, const float* __restrict__ v_scales) {
  extern __shared__ float sc[];
  __shared__ float s_red[256];
  __shared__ float s_inv;
  const int rowhead = blockIdx.x;
  const int row = rowhead / n_head;
  const int h = rowhead - row * n_head;
  if (row >= m) return;
  const int ctx = prefix_len + n_anc[row];
  const int tid = threadIdx.x;
  const int n_rep = n_head / n_head_kv;
  const int kv = h / n_rep;
  const int gph = head_dim / KV_QGROUP;
  const int* arow = anc + (long long)row * max_anc;

  const float* gsc = scores + ((long long)row * n_head + h) * ctx_max;
  for (int j = tid; j < ctx; j += blockDim.x) {
    sc[j] = gsc[j];
  }
  __syncthreads();

  float mx = -INFINITY;
  for (int j = tid; j < ctx; j += blockDim.x) {
    mx = fmaxf(mx, sc[j]);
  }
  s_red[tid] = mx;
  __syncthreads();
  if (tid < 64) {
    s_red[tid] = fmaxf(s_red[tid], s_red[tid + 64]);
  }
  __syncthreads();
  if (tid < 32) {
    float vv = fmaxf(s_red[tid], s_red[tid + 32]);
    for (int off = 16; off > 0; off >>= 1) {
      vv = fmaxf(vv, __shfl_down_sync(0xffffffffu, vv, off));
    }
    if (tid == 0) {
      s_red[0] = vv;
    }
  }
  __syncthreads();
  mx = s_red[0];

  for (int j = tid; j < ctx; j += blockDim.x) {
    sc[j] = exp_f32(__fsub_rn(sc[j], mx));
  }
  __syncthreads();
  if (tid == 0) {
    float sum = 0.0f;
    for (int j = 0; j < ctx; ++j) {
      sum = __fadd_rn(sum, sc[j]);
    }
    s_inv = __fdiv_rn(1.0f, sum);
  }
  __syncthreads();
  const float inv = s_inv;
  for (int j = tid; j < ctx; j += blockDim.x) {
    sc[j] = __fmul_rn(sc[j], inv);
  }
  __syncthreads();

  float* o_row = out + ((long long)row * n_head + h) * head_dim;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    const int g = d / KV_QGROUP;
    float acc = 0.0f;
    for (int j = 0; j < ctx; ++j) {
      const float w = sc[j];
      if (w == 0.0f) continue;
      const int slot = (j < prefix_len) ? j : arow[j - prefix_len];
      const float vv = (float)v[((long long)slot * n_head_kv + kv) * head_dim + d]
          * v_scales[((long long)slot * n_head_kv + kv) * gph + g];
      acc = __fadd_rn(acc, __fmul_rn(w, vv));
    }
    o_row[d] = acc;
  }
}

// lm_head_warp_f32 — the tied LM head, ONE WARP per vocab row (vs lm_head_f32's one
// thread). Lanes stride the row by 32 so the `token_embd` reads COALESCE (the by-thread
// kernel reads one 10 KB row per thread — adjacent threads 10 KB apart, fully
// uncoalesced; the head reads the whole 1.3 GB `token_embd` per token, so this is the
// decode memory bottleneck). The warp-shuffle reduction REORDERS the f32 sum, so this is
// NOT bit-exact with the host (~1e-5 rel) — it trades the greedy 256/256 bit-match for
// the sanctioned perplexity<=1% + lockstep gate. Partial sums use __fmul_rn/__fadd_rn (no
// FMA); only the cross-lane combine reorders.
__global__ void lm_head_warp_f32(const float* __restrict__ h, const float* __restrict__ embd,
                                 const int n_embd, const int vocab, float* __restrict__ logits) {
  lm_head_warp_body<KvLoadF32>(h, embd, n_embd, vocab, logits);
}

// lm_head_warp_f16 — like lm_head_warp_f32 but reads `token_embd` as **f16** (the GGUF's
// native precision; the host widens it to f32 losslessly, so `(float)embd_f16[k]` is the
// exact same value as the f32 buffer). This **bit-identically** halves the LM head's
// dominant cost — the read of the whole `[vocab, n_embd]` table (1.3 GB f32 → 0.65 GB
// f16) every token. `__half2float` is exact (f16 ⊂ f32). The warp-reduce reorders the sum
// identically to the f32 warp kernel (so the same greedy 256/256 holds).
__global__ void lm_head_warp_f16(const float* __restrict__ h, const __half* __restrict__ embd,
                                 const int n_embd, const int vocab, float* __restrict__ logits) {
  lm_head_warp_body<KvLoadF16>(h, embd, n_embd, vocab, logits);
}

// ===========================================================================
// v0.3.6: M>1 (batched) kernels for the **prefill** forward — process all P prompt
// tokens in ONE device-resident forward instead of P sequential M=1 decode steps
// (the O(seq) latency cliff). Same per-row math as the M=1 kernels (bit-identical),
// just with a row dimension `m`. One block per row `mi = blockIdx.x` for the per-row
// reductions; the tiled GEMM already handles M>1 via grid.y; residual/relu² are
// elementwise so they reuse the M=1 kernels over `m*n` elements.
// ===========================================================================

// rmsnorm_batch_f32 — rmsnorm over `m` rows; block `mi` handles row `mi`
// (shared-staged, canonical tree order per ADR 0018). Bit-identical to
// rmsnorm_shared_f32 per row.
__global__ void rmsnorm_batch_f32(const float* __restrict__ x, const float* __restrict__ w,
                                  const float eps, const int n, const int m,
                                  float* __restrict__ out) {
  const int mi = blockIdx.x;
  if (mi >= m) return;
  extern __shared__ float s_x[];
  __shared__ float s_inv;
  __shared__ float s_red[256];
  const float* xr = x + (long long)mi * n;
  float* outr = out + (long long)mi * n;
  for (int i = threadIdx.x; i < n; i += blockDim.x) s_x[i] = xr[i];
  __syncthreads();
  // Canonical tree sum (ADR 0018): slot t = threadIdx.x folds s_x[t],
  // s_x[t+256], ... in ascending order, then a power-of-two tree combines
  // the slots — the documented cross-backend order tritium_nn::ops::rmsnorm
  // implements identically on the host. Requires blockDim.x == 256 (all
  // launches comply). Replaces the sequential thread-0 fold, which was the
  // measured decode bottleneck (a 4-cycle-per-element latency chain).
  {
    float part = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      const float xi = s_x[i];
      part = __fadd_rn(part, __fmul_rn(xi, xi));
    }
    s_red[threadIdx.x] = part;
    __syncthreads();
    // Levels 128 and 64 in shared, then warp 0 finishes in registers: the
    // shuffle pairing (t, t+off) IS the canonical tree's pairing, so the DAG
    // (and every rounding) is identical — just 6 fewer block barriers.
    for (int off = 128; off >= 64; off >>= 1) {
      if (threadIdx.x < off) {
        s_red[threadIdx.x] = __fadd_rn(s_red[threadIdx.x], s_red[threadIdx.x + off]);
      }
      __syncthreads();
    }
    if (threadIdx.x < 32) {
      float v = __fadd_rn(s_red[threadIdx.x], s_red[threadIdx.x + 32]);  // level 32
      for (int off = 16; off > 0; off >>= 1) {
        v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, off));
      }
      if (threadIdx.x == 0) {
        const float mean_sq = __fdiv_rn(v, (float)n);
        const float denom = sqrtf(__fadd_rn(mean_sq, eps));
        s_inv = __fdiv_rn(1.0f, denom);
      }
    }
  }
  __syncthreads();
  const float inv = s_inv;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    outr[i] = __fmul_rn(__fmul_rn(s_x[i], inv), w[i]);
  }
}

// embedding_gather_batch_f32 — out[r] = table[tokens[r]] for r in 0..m. `tokens` is a
// device int array (the prompt ids). One thread per (row, element).
__global__ void embedding_gather_batch_f32(const float* __restrict__ table,
                                           const int* __restrict__ tokens, const int n_embd,
                                           const int m, float* __restrict__ out) {
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long long)m * n_embd) return;
  const int row = idx / n_embd;
  const int e = idx - (long long)row * n_embd;
  out[idx] = table[(long long)tokens[row] * n_embd + e];
}

// rope_apply_batch_f32 — RoPE on `m` rows, row `r` at absolute position `positions[r]`
// (so the full cos/sin table is indexed per row). Layout `[m, n_head, head_dim]`.
__global__ void rope_apply_batch_f32(float* __restrict__ x, const float* __restrict__ cos_table,
                                     const float* __restrict__ sin_table,
                                     const int* __restrict__ positions, const int n_head,
                                     const int head_dim, const int m) {
  const int half = head_dim >> 1;
  const int per_row = n_head * half;
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long long)m * per_row) return;
  const int row = idx / per_row;
  const int rem = idx - (long long)row * per_row;
  const int head = rem / half;
  const int j = rem - head * half;
  const int pos = positions[row];
  // Dead row (position -1, batching P2 C2): no rotation — the row's q/k are
  // never appended or attended, and a -1 cos/sin index would read OOB.
  if (pos < 0) return;
  const long long base = ((long long)row * n_head + head) * head_dim;
  const float c = cos_table[(long long)pos * half + j];
  const float s = sin_table[(long long)pos * half + j];
  const float a = x[base + j];
  const float b = x[base + j + half];
  x[base + j] = __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
  x[base + j + half] = __fadd_rn(__fmul_rn(b, c), __fmul_rn(a, s));
}

// act_quant_batch_f32 — per-row int8 absmax quant of `[m, k]`; block `mi` does row `mi`,
// writing q_out[mi] + act_scale[mi]. Bit-identical to act_quant_tiled_f32 per row.
__global__ void act_quant_batch_f32(const float* __restrict__ act, const int k, const int m,
                                    float* __restrict__ q_out, float* __restrict__ act_scale) {
  const int mi = blockIdx.x;
  if (mi >= m) return;
  __shared__ float s_red[256];
  __shared__ float s_gamma;
  const float* ar = act + (long long)mi * k;
  float* qr = q_out + (long long)mi * k;
  float local = 0.0f;
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    const float a = fabsf(ar[i]);
    if (a > local) local = a;
  }
  s_red[threadIdx.x] = local;
  __syncthreads();
  // max is exact under any order: shared levels 128/64, warp-0 shuffle tail
  // (6 fewer barriers); thread 0 lands the result back in s_red[0] for the
  // existing consumer below.
  for (int off = 128; off >= 64; off >>= 1) {
    if (threadIdx.x < off) {
      const float o = s_red[threadIdx.x + off];
      if (o > s_red[threadIdx.x]) s_red[threadIdx.x] = o;
    }
    __syncthreads();
  }
  if (threadIdx.x < 32) {
    float v = fmaxf(s_red[threadIdx.x], s_red[threadIdx.x + 32]);
    for (int off = 16; off > 0; off >>= 1) {
      v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    }
    if (threadIdx.x == 0) {
      s_red[0] = v;
    }
  }
  if (threadIdx.x == 0) {
    const float gamma = s_red[0];
    s_gamma = gamma;
    act_scale[mi] = (gamma == 0.0f) ? 0.0f : __fdiv_rn(gamma, 127.0f);
  }
  __syncthreads();
  const float gamma = s_gamma;
  if (gamma == 0.0f) {
    for (int i = threadIdx.x; i < k; i += blockDim.x) qr[i] = 0.0f;
    return;
  }
  const float s = __fdiv_rn(127.0f, gamma);
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    const float scaled = rintf(__fmul_rn(ar[i], s));
    qr[i] = fminf(fmaxf(scaled, -128.0f), 127.0f);
  }
}

// scale_mul_batch_f32 — out[mi*n + ni] *= act_scale[mi] (per-row activation-dequant fold).
__global__ void scale_mul_batch_f32(float* __restrict__ out, const float* __restrict__ act_scale,
                                    const int n, const int m) {
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long long)m * n) return;
  const int row = idx / n;
  out[idx] = __fmul_rn(out[idx], act_scale[row]);
}

// kv_append_batch_f32 — append `m` new k/v rows at `cache_len` into the arena: copy
// src[m, kv_width] → kv_base[(cache_len + r)*kv_width ...]. One thread per (row, element).
__global__ void kv_append_batch_f32(const float* __restrict__ src, float* __restrict__ kv_base,
                                    const int cache_len, const int kv_width, const int m) {
  kv_append_batch_body<KvStoreF32>(src, kv_base, cache_len, kv_width, m);
}

// gqa_attention_batch_f32 — causal attention for `m` query rows (prefill). Query row `r`
// is at absolute position `causal_offset + r` and attends keys `0..=causal_offset+r`. One
// warp per (row, head): lane-per-key scaled dots → lane-0 softmax (exp_f32) → lane-per-
// output-dim weighted sum. Bit-identical per (row, head) to gqa_attention_decode_warp_g.
// `scores` is `[m, n_head, ctx_max]` scratch (strided by ctx_max). k/v are
// `[ctx_max, n_head_kv, head_dim]`.
__global__ void gqa_attention_batch_f32(const float* __restrict__ q, const float* __restrict__ k,
                                        const float* __restrict__ v, float* __restrict__ out,
                                        float* __restrict__ scores, const int ctx_max,
                                        const int n_head, const int n_head_kv, const int head_dim,
                                        const float scale, const int causal_offset, const int m) {
  gqa_attention_batch_body<KvLoadF32>(q, k, v, out, scores, ctx_max, n_head, n_head_kv, head_dim, scale,
      causal_offset, m);
}

// gqa_attention_batch_v2_f32 — the v2 prefill attention (see the v2 body doc):
// one block per (row, head), shared-staged K + shared scores, bit-identical
// per (row, head) to gqa_attention_batch_f32. Grid (n_head, m), block
// ATTN_V2_THREADS. Host dispatches v2 only when head_dim <= ATTN_V2_HDMAX and
// causal_offset + m <= ATTN_V2_MAX_CTX.
__global__ void __launch_bounds__(ATTN_V2_THREADS) gqa_attention_batch_v2_f32(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ out, const int n_head, const int n_head_kv, const int head_dim,
    const float scale, const int causal_offset, const int m) {
  gqa_attention_batch_v2_body<KvLoadF32>(q, k, v, out, n_head, n_head_kv, head_dim, scale,
      causal_offset, m);
}

// gqa_attention_batch_v3_f32 — Q-blocked prefill attention (see the v3 body
// doc): grid (n_head, ceil(m/ATTN_V3_BQ)), block ATTN_V3_THREADS, scores is
// the [m, n_head, ctx_max] scratch. Bit-identical to gqa_attention_batch_f32
// per (row, head). Host dispatches when head_dim <= ATTN_V2_HDMAX (no ctx
// bound).
__global__ void __launch_bounds__(ATTN_V3_THREADS) gqa_attention_batch_v3_f32(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ out, float* __restrict__ scores, const int ctx_max, const int n_head,
    const int n_head_kv, const int head_dim, const float scale, const int causal_offset,
    const int m) {
  gqa_attention_batch_v3_body<KvLoadF32>(q, k, v, out, scores, ctx_max, n_head, n_head_kv,
      head_dim, scale, causal_offset, m);
}

// gqa_attention_tree_f32 — BASTION tree-verify attention (ADR 0014).
// N draft-tree nodes share the committed prefix KV; node `row` attends
// `[0, prefix_len)` (the shared prefix, in arena order) followed by its
// ancestor chain INCLUDING itself (arena slot indices `anc[row*max_anc + t]`,
// `t < n_anc[row]`, root-first). Same per-key d-order dots, host-order softmax
// fold, and j-order weighted sums as `gqa_attention_batch_f32` — for a CHAIN
// tree (anc = prefix_len+0..=prefix_len+row) the key sequence is exactly the
// batch kernel's `[0, ctx)`, so chain-tree output is bit-identical to the
// batched prefill attention (the ADR 0014 parity gate).
__global__ void gqa_attention_tree_f32(const float* __restrict__ q, const float* __restrict__ k,
                                       const float* __restrict__ v, float* __restrict__ out,
                                       float* __restrict__ scores,
                                       const int* __restrict__ anc,    // [m, max_anc] arena slots
                                       const int* __restrict__ n_anc,  // [m] = depth+1
                                       const int ctx_max, const int n_head, const int n_head_kv,
                                       const int head_dim, const float scale,
                                       const int prefix_len, const int max_anc, const int m) {
  const long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const long long total = (long long)m * n_head;
  if (warp >= total) return;
  const int row = warp / n_head;
  const int h = warp - (long long)row * n_head;
  const int na = n_anc[row];
  const int ctx = prefix_len + na;
  const int n_rep = n_head / n_head_kv;
  const int kv = h / n_rep;
  const float* q_row = q + ((long long)row * n_head + h) * head_dim;
  const int* arow = anc + (long long)row * max_anc;
  float* sc = scores + ((long long)row * n_head + h) * ctx_max;

  for (int j = lane; j < ctx; j += 32) {
    const int slot = (j < prefix_len) ? j : arow[j - prefix_len];
    const float* k_row = k + ((long long)slot * n_head_kv + kv) * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      dot = __fadd_rn(dot, __fmul_rn(q_row[d], k_row[d]));
    }
    sc[j] = __fmul_rn(dot, scale);
  }
  __syncwarp();
  if (lane == 0) {
    float mx = -INFINITY;
    for (int j = 0; j < ctx; ++j) {
      if (sc[j] > mx) mx = sc[j];
    }
    float sum = 0.0f;
    for (int j = 0; j < ctx; ++j) {
      const float e = exp_f32(__fsub_rn(sc[j], mx));
      sc[j] = e;
      sum = __fadd_rn(sum, e);
    }
    const float inv = __fdiv_rn(1.0f, sum);
    for (int j = 0; j < ctx; ++j) {
      sc[j] = __fmul_rn(sc[j], inv);
    }
  }
  __syncwarp();
  float* o_row = out + ((long long)row * n_head + h) * head_dim;
  for (int d = lane; d < head_dim; d += 32) {
    float acc = 0.0f;
    for (int j = 0; j < ctx; ++j) {
      const float w = sc[j];
      if (w == 0.0f) continue;
      const int slot = (j < prefix_len) ? j : arow[j - prefix_len];
      const float* v_row = v + ((long long)slot * n_head_kv + kv) * head_dim;
      acc = __fadd_rn(acc, __fmul_rn(w, v_row[d]));
    }
    o_row[d] = acc;
  }
}

// ===========================================================================
// v0.3.7: M=N batched DECODE kernels — N concurrent sequences, one token each per step.
// Unlike prefill (one sequence, all rows share the growing KV), each row r is a distinct
// sequence with its OWN KV slice and its OWN position. The per-sequence KV is laid out
// `[n, max_ctx, n_head_kv, head_dim]` (seq r's KV is the contiguous `[max_ctx, kv_width]`
// at offset r·max_ctx·kv_width). Per row math is bit-identical to the M=1 decode.
// ===========================================================================

}  // extern "C" — KV MAPPING codecs + M=N KV kernel bodies (ADR 0025, C3).

// ── KV mapping (ADR 0025 paged KV) ──────────────────────────────────────────
// One body per M=N KV kernel, templated on PAGED. `if constexpr` prunes to
// exactly the retired hand-written dense source (the dense shims' SASS
// byte-identity vs those kernels is the proof obligation, tools/sass_diff.sh;
// a struct-shaped mapping codec was tried first and drifted the schedule —
// the constexpr-pruned body is the shape ptxas reproduces). Paged mapping:
// physical token = table_row[j >> KV_PAGE_SHIFT] · KV_PAGE_TOKENS +
// (j & KV_PAGE_MASK); pools are `[pool_pages, KV_PAGE_TOKENS, kv_width]`.
// An unmapped (-1) table entry is a host-allocator bug, not a fallback — the
// kernels do not guard it.
#define KV_PAGE_TOKENS 256
#define KV_PAGE_SHIFT 8
#define KV_PAGE_MASK 255

// Per-lane register budget for one warp's accumulator: head_dim / 32, capped
// at 8 (head_dim ≤ 256; the Rust launch asserts it). Hoisted above the
// template body that uses it.
#define SPLIT_MAX_HD_PER_LANE 8

// kv_append_mdecode_body — append `n` new k/v rows, row r → seq r's KV at positions[r].
template <bool PAGED>
static __device__ __forceinline__ void kv_append_mdecode_body(
    const float* __restrict__ src, float* __restrict__ kv_base,
    const int* __restrict__ positions, const int* __restrict__ table, const int tstride,
    const int max_ctx, const int kv_width, const int n) {
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long long)n * kv_width) return;
  const int row = idx / kv_width;
  // Dead row (position -1, batching P2 C2): touch NOTHING — an unguarded -1
  // would write into the previous row's arena tail (or before the whole
  // arena for row 0). This is the paged-KV contract: a dead row owns no
  // write slot.
  if (positions[row] < 0) return;
  const int e = idx - (long long)row * kv_width;
  long long tok;
  if constexpr (PAGED) {
    const int p = positions[row];
    tok = (long long)table[(long long)row * tstride + (p >> KV_PAGE_SHIFT)] * KV_PAGE_TOKENS +
          (p & KV_PAGE_MASK);
  } else {
    tok = (long long)row * max_ctx + positions[row];
  }
  const long long off = tok * kv_width + e;
  kv_base[off] = src[(long long)row * kv_width + e];
}

// gqa_attention_split_partial_body — warp per (row, head, split); see the
// dense shim's doc below. Only the KV addressing is PAGED-conditional;
// everything else is the retired kernel verbatim.
template <bool PAGED>
static __device__ __forceinline__ void gqa_attention_split_partial_body(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ partials, const int* __restrict__ positions,
    const int* __restrict__ table, const int tstride, const int max_ctx,
    const int n_head, const int n_head_kv, const int head_dim, const float scale, const int n,
    const int n_split, const int chunk) {
  const long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= (long long)n * n_head * n_split) return;
  const int s = warp % n_split;
  const long long rh = warp / n_split;  // row*n_head + h
  const int row = rh / n_head;
  const int h = rh - (long long)row * n_head;
  const int ctx = positions[row] + 1;  // keys 0..=positions[row]
  const int start = s * chunk;
  const int end = min(start + chunk, ctx);
  float* part = partials + warp * (long long)(head_dim + 2);

  // Split beyond this row's ctx → identity partial (m = -inf contributes 0 in combine).
  if (start >= ctx) {
    for (int d = lane; d < head_dim; d += 32) part[d] = 0.0f;
    if (lane == 0) {
      part[head_dim] = -INFINITY;
      part[head_dim + 1] = 0.0f;
    }
    return;
  }

  const int n_rep = n_head / n_head_kv;
  const int kv = h / n_rep;
  const long long kv_seq = PAGED ? 0 : (long long)row * max_ctx * n_head_kv * head_dim;
  const float* k_seq = k + kv_seq;
  const float* v_seq = v + kv_seq;
  const int* trow = PAGED ? table + (long long)row * tstride : nullptr;
  const float* q_row = q + rh * (long long)head_dim;

  float acc[SPLIT_MAX_HD_PER_LANE];
  const int per_lane = (head_dim + 31) / 32;
  for (int i = 0; i < per_lane; ++i) acc[i] = 0.0f;
  float m = -INFINITY, l = 0.0f;

  for (int j = start; j < end; ++j) {
    long long jt;
    if constexpr (PAGED) {
      jt = (long long)trow[j >> KV_PAGE_SHIFT] * KV_PAGE_TOKENS + (j & KV_PAGE_MASK);
    } else {
      jt = j;
    }
    const float* k_row = k_seq + (jt * n_head_kv + kv) * head_dim;
    // Lane-strided partial dot, then a warp tree-reduce to the full q·k.
    float pd = 0.0f;
    for (int d = lane; d < head_dim; d += 32) pd = __fadd_rn(pd, __fmul_rn(q_row[d], k_row[d]));
    for (int off = 16; off > 0; off >>= 1) pd += __shfl_down_sync(0xffffffff, pd, off);
    float sj = __shfl_sync(0xffffffff, pd, 0);  // broadcast the reduced dot to all lanes
    sj = __fmul_rn(sj, scale);
    const float m_new = fmaxf(m, sj);
    const float corr = exp_f32(__fsub_rn(m, m_new));  // 0 on the first key (m = -inf)
    const float p = exp_f32(__fsub_rn(sj, m_new));
    l = __fadd_rn(__fmul_rn(l, corr), p);
    const float* v_row = v_seq + (jt * n_head_kv + kv) * head_dim;
    for (int i = 0, d = lane; d < head_dim; d += 32, ++i)
      acc[i] = __fadd_rn(__fmul_rn(acc[i], corr), __fmul_rn(p, v_row[d]));
    m = m_new;
  }
  for (int i = 0, d = lane; d < head_dim; d += 32, ++i) part[d] = acc[i];
  if (lane == 0) {
    part[head_dim] = m;
    part[head_dim + 1] = l;
  }
}

extern "C" {

// kv_append_mdecode_f32 — append `n` new k/v rows, row r → seq r's KV at positions[r].
__global__ void kv_append_mdecode_f32(const float* __restrict__ src, float* __restrict__ kv_base,
                                      const int* __restrict__ positions, const int max_ctx,
                                      const int kv_width, const int n) {
  kv_append_mdecode_body<false>(src, kv_base, positions, nullptr, 0, max_ctx, kv_width, n);
}

// kv_append_mdecode_paged_f32 — the paged twin: `kv_base` is the page POOL
// (`[pool_pages, KV_PAGE_TOKENS, kv_width]`), `table` the `[n, tstride]`
// page table. Same dead-row contract.
__global__ void kv_append_mdecode_paged_f32(const float* __restrict__ src,
                                            float* __restrict__ kv_base,
                                            const int* __restrict__ positions,
                                            const int* __restrict__ table, const int tstride,
                                            const int kv_width, const int n) {
  kv_append_mdecode_body<true>(src, kv_base, positions, table, tstride, 0, kv_width, n);
}


// ===========================================================================
// v0.3.8 (Track-2): on-device sampling for the batched M=N decode graph — fold the
// LM head + greedy argmax into the captured graph so a step returns N token ids
// (N·4 bytes) instead of N·vocab·4 bytes of logits (the 33 MB/step readback at N=64).
// ===========================================================================

// lm_head_tiled_f16 — batched LM head: [m, n_embd] · embd_f16[vocab, n_embd] -> [m, vocab].
// One warp per vocab row, computing LMHEAD_ROW_TILE output rows at once: each warp reads its
// `embd` row from the 0.66 GB f16 table ONCE and reuses it across the row-tile, so the table
// is streamed ceil(m / LMHEAD_ROW_TILE) times per step instead of once per row — the embd
// read is the dominant cost at large N (~930 GB/s, ~the 4090's bandwidth). `h` (m·n_embd, a
// few hundred KB) stays hot in L2. BIT-IDENTICAL per row to lm_head_warp_f16 (same f16 read,
// same __fadd_rn/__fmul_rn per k, same warp tree-reduce order).
#define LMHEAD_ROW_TILE 8
__global__ void lm_head_tiled_f16(const float* __restrict__ h, const __half* __restrict__ embd,
                                  const int n_embd, const int vocab, const int m,
                                  float* __restrict__ logits) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= vocab) return;
  const int row0 = blockIdx.y * LMHEAD_ROW_TILE;
  const __half* erow = embd + (long long)warp * n_embd;
  float acc[LMHEAD_ROW_TILE];
#pragma unroll
  for (int r = 0; r < LMHEAD_ROW_TILE; ++r) acc[r] = 0.0f;
  for (int k = lane; k < n_embd; k += 32) {
    const float e = __half2float(erow[k]);
#pragma unroll
    for (int r = 0; r < LMHEAD_ROW_TILE; ++r) {
      const int mi = row0 + r;
      if (mi < m) acc[r] = __fadd_rn(acc[r], __fmul_rn(h[(long long)mi * n_embd + k], e));
    }
  }
#pragma unroll
  for (int r = 0; r < LMHEAD_ROW_TILE; ++r) {
    float a = acc[r];
    for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffff, a, off);
    const int mi = row0 + r;
    if (lane == 0 && mi < m) logits[(long long)mi * vocab + warp] = a;
  }
}

// argmax_rows_f32 — per-row greedy argmax: [m, vocab] -> [m] i32. One block (256 threads)
// per row. Matches host `sample_greedy` (Iterator::max_by keeps the LATER element on
// equality), so ties resolve to the HIGHEST index. The forward never emits NaN logits, so
// no NaN special-case is needed.
__global__ void argmax_rows_f32(const float* __restrict__ logits, const int vocab, const int m,
                                int* __restrict__ out) {
  const int mi = blockIdx.x;
  if (mi >= m) return;
  const float* row = logits + (long long)mi * vocab;
  float best_val = -INFINITY;
  int best_idx = -1;
  for (int j = threadIdx.x; j < vocab; j += blockDim.x) {
    const float v = row[j];
    if (v > best_val || (v == best_val && j > best_idx)) {
      best_val = v;
      best_idx = j;
    }
  }
  __shared__ float s_val[256];
  __shared__ int s_idx[256];
  s_val[threadIdx.x] = best_val;
  s_idx[threadIdx.x] = best_idx;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float ov = s_val[threadIdx.x + stride];
      const int oi = s_idx[threadIdx.x + stride];
      if (ov > s_val[threadIdx.x] || (ov == s_val[threadIdx.x] && oi > s_idx[threadIdx.x])) {
        s_val[threadIdx.x] = ov;
        s_idx[threadIdx.x] = oi;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) out[mi] = s_idx[0];
}

// ===========================================================================
// Flash-decoding (split-KV) attention — the low-N occupancy fix. The decode
// attention `gqa_attention_mdecode_f32` runs warp-per-(row,head): at N=1 that is
// ~n_head warps = a few blocks, leaving a 128-SM GPU ~idle (ncu: SM 0.2% / DRAM
// 0.4% busy), latency-bound on the KV read whose cost grows with ctx. These two
// kernels split each (row,head)'s key range into `n_split` chunks across `n_split`
// blocks, so a single decode row fills the GPU. Numerically equal to the direct
// softmax within fp tolerance (the online-softmax merge reorders the sums) — NOT
// bit-exact, so the gates that consume it are tolerance-based (vs the transformers
// reference / parity), and eager + graph share these kernels to stay mutually exact.
// ===========================================================================


// gqa_attention_split_partial_f32 — warp per (row, head, split). Online-softmax over
// this warp's key chunk `[s*chunk, min((s+1)*chunk, ctx))` → a partial
// {acc[head_dim] relative to the chunk max, m = chunk max, l = chunk sumexp}, written
// to `partials[warp][head_dim+2]` (warp index = (row*n_head+h)*n_split + s).
// Chunked argmax pair — the single-kernel argmax_rows_f32 spawns only m
// blocks (measured 129us at m=13: a 128k-vocab scan on 13 of 128 SMs). The
// partial kernel fans a row's scan over ARGMAX_CHUNKS blocks; the combine
// folds the per-chunk (val, idx) pairs with the SAME tie rule (equal value ->
// higher index, matching host sample_greedy's max_by), so the result is
// exactly argmax_rows_f32's.
#define ARGMAX_CHUNKS 16
__global__ void argmax_rows_partial_f32(const float* __restrict__ logits, const int vocab,
                                        const int m, float* __restrict__ pvals,
                                        int* __restrict__ pidx) {
  const int mi = blockIdx.x;
  const int c = blockIdx.y;
  if (mi >= m) return;
  const int chunk_len = (vocab + ARGMAX_CHUNKS - 1) / ARGMAX_CHUNKS;
  const int j0 = c * chunk_len;
  const int j1 = min(j0 + chunk_len, vocab);
  const float* row = logits + (long long)mi * vocab;
  float best_val = -INFINITY;
  int best_idx = -1;
  for (int j = j0 + threadIdx.x; j < j1; j += blockDim.x) {
    const float v = row[j];
    if (v > best_val || (v == best_val && j > best_idx)) {
      best_val = v;
      best_idx = j;
    }
  }
  __shared__ float s_val[256];
  __shared__ int s_idx[256];
  s_val[threadIdx.x] = best_val;
  s_idx[threadIdx.x] = best_idx;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float ov = s_val[threadIdx.x + stride];
      const int oi = s_idx[threadIdx.x + stride];
      if (ov > s_val[threadIdx.x] || (ov == s_val[threadIdx.x] && oi > s_idx[threadIdx.x])) {
        s_val[threadIdx.x] = ov;
        s_idx[threadIdx.x] = oi;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    pvals[mi * ARGMAX_CHUNKS + c] = s_val[0];
    pidx[mi * ARGMAX_CHUNKS + c] = s_idx[0];
  }
}

__global__ void argmax_rows_combine_f32(const float* __restrict__ pvals,
                                        const int* __restrict__ pidx, const int m,
                                        int* __restrict__ out) {
  const int mi = blockIdx.x;
  if (mi >= m) return;
  const int lane = threadIdx.x & 31;
  float best_val = -INFINITY;
  int best_idx = -1;
  if (lane < ARGMAX_CHUNKS) {
    best_val = pvals[mi * ARGMAX_CHUNKS + lane];
    best_idx = pidx[mi * ARGMAX_CHUNKS + lane];
  }
  for (int off = 16; off > 0; off >>= 1) {
    const float ov = __shfl_down_sync(0xffffffffu, best_val, off);
    const int oi = __shfl_down_sync(0xffffffffu, best_idx, off);
    if (ov > best_val || (ov == best_val && oi > best_idx)) {
      best_val = ov;
      best_idx = oi;
    }
  }
  if (lane == 0) out[mi] = best_idx;
}

__global__ void gqa_attention_split_partial_f32(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ partials, const int* __restrict__ positions, const int max_ctx,
    const int n_head, const int n_head_kv, const int head_dim, const float scale, const int n,
    const int n_split, const int chunk) {
  gqa_attention_split_partial_body<false>(q, k, v, partials, positions, nullptr, 0, max_ctx,
                                               n_head, n_head_kv, head_dim, scale, n, n_split,
                                               chunk);
}

// gqa_attention_split_partial_paged_f32 — the paged twin: `k`/`v` are the page
// POOLS, `table` the `[n, tstride]` page table (bit-identical values to dense —
// paging changes addresses, never the reduction order).
__global__ void gqa_attention_split_partial_paged_f32(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ partials, const int* __restrict__ positions,
    const int* __restrict__ table, const int tstride,
    const int n_head, const int n_head_kv, const int head_dim, const float scale, const int n,
    const int n_split, const int chunk) {
  gqa_attention_split_partial_body<true>(q, k, v, partials, positions, table, tstride, 0,
                                               n_head, n_head_kv, head_dim, scale, n, n_split,
                                               chunk);
}

// gqa_attention_combine_f32 — warp per (row, head). Flash-merge the `n_split` partials
// into out[head_dim]: global max M over splits, L = Σ_s l_s·exp(m_s−M), and
// out[d] = (Σ_s exp(m_s−M)·acc_s[d]) / L.
__global__ void gqa_attention_combine_f32(const float* __restrict__ partials,
                                          float* __restrict__ out, const int n_head,
                                          const int head_dim, const int n, const int n_split) {
  const long long rh = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;  // row*n_head+h
  const int lane = threadIdx.x & 31;
  if (rh >= (long long)n * n_head) return;
  const float* base = partials + rh * (long long)n_split * (head_dim + 2);

  float M = -INFINITY;
  for (int s = 0; s < n_split; ++s) {
    const float ms = base[(long long)s * (head_dim + 2) + head_dim];
    if (ms > M) M = ms;
  }
  float L = 0.0f;
  for (int s = 0; s < n_split; ++s) {
    const float ms = base[(long long)s * (head_dim + 2) + head_dim];
    if (ms != -INFINITY) {
      const float ls = base[(long long)s * (head_dim + 2) + head_dim + 1];
      L = __fadd_rn(L, __fmul_rn(ls, exp_f32(__fsub_rn(ms, M))));
    }
  }
  float* o_row = out + rh * (long long)head_dim;
  // Dead row (position -1, batching P2 C2): every split emitted the identity
  // partial (m = -inf, l = 0), so L == 0 and 1/L would turn the 0-sum into
  // NaN. Emit zeros instead (finite, deterministic; the row's output is
  // discarded). Live rows always have L > 0 — this branch never retouches
  // their arithmetic.
  if (L == 0.0f) {
    for (int d = lane; d < head_dim; d += 32) o_row[d] = 0.0f;
    return;
  }
  const float inv = __fdiv_rn(1.0f, L);
  for (int d = lane; d < head_dim; d += 32) {
    float a = 0.0f;
    for (int s = 0; s < n_split; ++s) {
      const float ms = base[(long long)s * (head_dim + 2) + head_dim];
      if (ms == -INFINITY) continue;
      const float w = exp_f32(__fsub_rn(ms, M));
      a = __fadd_rn(a, __fmul_rn(w, base[(long long)s * (head_dim + 2) + d]));
    }
    o_row[d] = __fmul_rn(a, inv);
  }
}

// draft_chain_advance — the L1' chained-draft glue (ADR 0032): between two
// replays of the captured M=1 decode graph, feed the device argmax result
// back into the control block ON DEVICE, so a k-token greedy draft costs ONE
// host round-trip (final sync + k*4B readback) instead of one per token.
// Launched single-thread, stream-ordered between replays on the capture
// stream: [graph replay] -> [argmax partial/combine] -> [this] -> [replay]...
//
//   halt[0] != 0 -> frozen: store the -1 sentinel and touch nothing else.
//                   Later replays re-run with a STALE ctrl — they rewrite the
//                   same KV row with the same values (deterministic kernels),
//                   which is idempotent and unobserved past the watermark.
//   id == eos    -> store the id, then halt BEFORE advancing ctrl (matches
//                   the host loop's "the draft believes the turn ends here"
//                   early break: the EOS itself is drafted, never fed).
//   otherwise    -> store the id and advance ctrl = [id, pos+1, cache_len+1]
//                   for the next replay (pos == cache_len in decode).
__global__ void draft_chain_advance(int* __restrict__ ctrl,
                                    const int* __restrict__ am_out,
                                    int* __restrict__ chain_out,
                                    int* __restrict__ halt,
                                    const int step, const int eos) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  if (halt[0] != 0) {
    chain_out[step] = -1;
    return;
  }
  const int t = am_out[0];
  chain_out[step] = t;
  if (t == eos) {
    halt[0] = 1;
    return;
  }
  ctrl[0] = t;
  ctrl[1] += 1;
  ctrl[2] += 1;
}

}  // extern "C"
