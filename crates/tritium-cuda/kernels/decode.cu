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

  // absmax slots: threads < 256 own s_red[t]; threads ≥ 256 park their max in
  // s_red[256 + (t-256)] (s_red is 1024 wide) and one extra exact max-merge
  // level folds the upper half in. Max is order-free, so any shape is exact.
  s_red[threadIdx.x] = local_max;
  __syncthreads();
  if (threadIdx.x < 256 && threadIdx.x + 256 < blockDim.x) {
    s_red[threadIdx.x] = fmaxf(s_red[threadIdx.x], s_red[threadIdx.x + 256]);
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
__global__ void kv_append_f32(const float* __restrict__ src,
                              float* __restrict__ kv_base,   // [max_ctx*kv_width]
                              const int* __restrict__ ctrl,
                              const int kv_width) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < kv_width) {
    const long long off = (long long)ctrl[2] * kv_width + i;
    kv_base[off] = src[i];
  }
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
    kv_k_base[row + base + j] = __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
    kv_k_base[row + base + j + half] = __fadd_rn(__fmul_rn(b, c), __fmul_rn(a, s));
  } else if (idx < q_total + k_total + kv_width) {
    const int i = idx - q_total - k_total;
    kv_v_base[row + i] = v[i];
  }
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

__global__ void gqa_attention_scores_g(const float* __restrict__ q,
                                       const float* __restrict__ k,
                                       float* __restrict__ scores,
                                       const int* __restrict__ ctrl,
                                       const int max_ctx, const int n_head,
                                       const int n_head_kv, const int head_dim,
                                       const float scale) {
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
  const float4* k0 = (const float4*)(k + ((long long)j0 * n_head_kv + kv) * head_dim);
  const float4* k1 = (const float4*)(k + ((long long)j1 * n_head_kv + kv) * head_dim);
  const float4* k2 = (const float4*)(k + ((long long)j2 * n_head_kv + kv) * head_dim);
  const float4* k3 = (const float4*)(k + ((long long)j3 * n_head_kv + kv) * head_dim);
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
      const float4 a = k0[d];
      const float4 b = k1[d];
      const float4 c = k2[d];
      const float4 e = k3[d];
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
    const float4* ks[4] = {k0, k1, k2, k3};
    const int js[4] = {j0, j1, j2, j3};
    for (int t = 0; t < 4; ++t) {
      if (js[t] >= ctx) break;
      float dot = 0.0f;
#pragma unroll 4
      for (int d = 0; d < hd4; ++d) {
        const float4 qq = qv[d];
        const float4 a = ks[t][d];
        dot = __fadd_rn(dot, __fmul_rn(qq.x, a.x));
        dot = __fadd_rn(dot, __fmul_rn(qq.y, a.y));
        dot = __fadd_rn(dot, __fmul_rn(qq.z, a.z));
        dot = __fadd_rn(dot, __fmul_rn(qq.w, a.w));
      }
      sc[js[t]] = __fmul_rn(dot, scale);
    }
  }
}

__global__ void gqa_attention_reduce_g(const float* __restrict__ v,
                                       const float* __restrict__ scores,
                                       float* __restrict__ out,
                                       const int* __restrict__ ctrl,
                                       const int max_ctx, const int n_head,
                                       const int n_head_kv, const int head_dim) {
  extern __shared__ float sc[];
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
    sc[j] = gsc[j];
  }
  __syncthreads();

  // max: block tree — exact (f32 max never rounds; fmaxf skips NaN like the
  // sequential `>` scan).
  float m = -INFINITY;
  for (int j = tid; j < ctx; j += blockDim.x) {
    m = fmaxf(m, sc[j]);
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
    sc[j] = exp_f32(__fsub_rn(sc[j], m));
  }
  __syncthreads();

  // sum: the ONE rounded f32 fold — sequential on thread 0 in j-order.
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

  // weighted sum: one output dim per thread, j-order chain per dim (bit-exact).
  // The v load is unconditional so the compiler batches loads across the
  // unrolled j iterations; the w==0 skip predicates only the fadd.
  float* o_row = out + (long long)h * head_dim;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
#pragma unroll 8
    for (int j = 0; j < ctx; ++j) {
      const float w = sc[j];
      const float vv = v[((long long)j * n_head_kv + kv) * head_dim + d];
      if (w != 0.0f) {
        acc = __fadd_rn(acc, __fmul_rn(w, vv));
      }
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
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= vocab) return;
  const float* row = embd + (long long)warp * n_embd;
  float acc = 0.0f;
  for (int k = lane; k < n_embd; k += 32) {
    acc = __fadd_rn(acc, __fmul_rn(h[k], row[k]));  // coalesced across the warp
  }
  // Warp-shuffle tree reduction (reorders the sum vs the host's sequential fold).
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffff, acc, off);
  }
  if (lane == 0) logits[warp] = acc;
}

// lm_head_warp_f16 — like lm_head_warp_f32 but reads `token_embd` as **f16** (the GGUF's
// native precision; the host widens it to f32 losslessly, so `(float)embd_f16[k]` is the
// exact same value as the f32 buffer). This **bit-identically** halves the LM head's
// dominant cost — the read of the whole `[vocab, n_embd]` table (1.3 GB f32 → 0.65 GB
// f16) every token. `__half2float` is exact (f16 ⊂ f32). The warp-reduce reorders the sum
// identically to the f32 warp kernel (so the same greedy 256/256 holds).
__global__ void lm_head_warp_f16(const float* __restrict__ h, const __half* __restrict__ embd,
                                 const int n_embd, const int vocab, float* __restrict__ logits) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= vocab) return;
  const __half* row = embd + (long long)warp * n_embd;
  float acc = 0.0f;
  for (int k = lane; k < n_embd; k += 32) {
    acc = __fadd_rn(acc, __fmul_rn(h[k], __half2float(row[k])));  // coalesced f16 read
  }
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffff, acc, off);
  }
  if (lane == 0) logits[warp] = acc;
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
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long long)m * kv_width) return;
  const int row = idx / kv_width;
  const int e = idx - (long long)row * kv_width;
  kv_base[((long long)(cache_len + row)) * kv_width + e] = src[(long long)row * kv_width + e];
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
    const float* k_row = k + ((long long)j * n_head_kv + kv) * head_dim;
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
      const float* v_row = v + ((long long)j * n_head_kv + kv) * head_dim;
      acc = __fadd_rn(acc, __fmul_rn(w, v_row[d]));
    }
    o_row[d] = acc;
  }
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

// kv_append_mdecode_f32 — append `n` new k/v rows, row r → seq r's KV at positions[r].
__global__ void kv_append_mdecode_f32(const float* __restrict__ src, float* __restrict__ kv_base,
                                      const int* __restrict__ positions, const int max_ctx,
                                      const int kv_width, const int n) {
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long long)n * kv_width) return;
  const int row = idx / kv_width;
  const int e = idx - (long long)row * kv_width;
  const long long off = ((long long)row * max_ctx + positions[row]) * kv_width + e;
  kv_base[off] = src[(long long)row * kv_width + e];
}

// gqa_attention_mdecode_f32 — M=N decode attention: row r (seq r) attends seq r's KV
// `0..=positions[r]`. Warp per (row, head); same lane-per-key dots / lane-0 softmax /
// lane-per-output-dim weighted sum as the M=1 warp kernel, but indexing seq r's KV slice.
// `scores` is `[n, n_head, max_ctx]`.
__global__ void gqa_attention_mdecode_f32(const float* __restrict__ q, const float* __restrict__ k,
                                          const float* __restrict__ v, float* __restrict__ out,
                                          float* __restrict__ scores, const int* __restrict__ positions,
                                          const int max_ctx, const int n_head, const int n_head_kv,
                                          const int head_dim, const float scale, const int n) {
  const long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= (long long)n * n_head) return;
  const int row = warp / n_head;
  const int h = warp - (long long)row * n_head;
  const int ctx = positions[row] + 1;  // keys 0..=positions[row]
  const int n_rep = n_head / n_head_kv;
  const int kv = h / n_rep;
  const long long kv_seq = (long long)row * max_ctx * n_head_kv * head_dim;
  const float* k_seq = k + kv_seq;
  const float* v_seq = v + kv_seq;
  const float* q_row = q + ((long long)row * n_head + h) * head_dim;
  float* sc = scores + ((long long)row * n_head + h) * max_ctx;

  for (int j = lane; j < ctx; j += 32) {
    const float* k_row = k_seq + ((long long)j * n_head_kv + kv) * head_dim;
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
      const float* v_row = v_seq + ((long long)j * n_head_kv + kv) * head_dim;
      acc = __fadd_rn(acc, __fmul_rn(w, v_row[d]));
    }
    o_row[d] = acc;
  }
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

// Per-lane register budget for one warp's accumulator: head_dim / 32, capped at 8
// (head_dim ≤ 256). The Rust launch asserts head_dim ≤ 256.
#define SPLIT_MAX_HD_PER_LANE 8

// gqa_attention_split_partial_f32 — warp per (row, head, split). Online-softmax over
// this warp's key chunk `[s*chunk, min((s+1)*chunk, ctx))` → a partial
// {acc[head_dim] relative to the chunk max, m = chunk max, l = chunk sumexp}, written
// to `partials[warp][head_dim+2]` (warp index = (row*n_head+h)*n_split + s).
__global__ void gqa_attention_split_partial_f32(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ partials, const int* __restrict__ positions, const int max_ctx,
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
  const long long kv_seq = (long long)row * max_ctx * n_head_kv * head_dim;
  const float* k_seq = k + kv_seq;
  const float* v_seq = v + kv_seq;
  const float* q_row = q + rh * (long long)head_dim;

  float acc[SPLIT_MAX_HD_PER_LANE];
  const int per_lane = (head_dim + 31) / 32;
  for (int i = 0; i < per_lane; ++i) acc[i] = 0.0f;
  float m = -INFINITY, l = 0.0f;

  for (int j = start; j < end; ++j) {
    const float* k_row = k_seq + ((long long)j * n_head_kv + kv) * head_dim;
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
    const float* v_row = v_seq + ((long long)j * n_head_kv + kv) * head_dim;
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
  const float inv = __fdiv_rn(1.0f, L);
  float* o_row = out + rh * (long long)head_dim;
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

}  // extern "C"
