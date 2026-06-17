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

// rmsnorm_shared_f32 — **bit-identical** to rmsnorm_f32 but ~8× faster for the graph
// (v0.3.3+ perf). rmsnorm_f32's thread-0 sum is latency-bound: one thread reads `n`
// floats one-at-a-time from global, unable to hide ~400-cycle memory latency. Here the
// whole block first stages `x` into shared memory with a COALESCED load, then thread 0
// sums from shared **in the same i=0..n order** — so the f32 sum is byte-identical (no
// reorder, unlike a tree reduction), keeping greedy 256/256, while the sum is now
// compute-bound instead of memory-latency-bound. Dynamic shared = `n * 4` bytes (the
// launch sets it); n_ff=6912 → 27 KiB < 48 KiB. The elementwise reads the staged `s_x`.
__global__ void rmsnorm_shared_f32(const float* __restrict__ x,
                                   const float* __restrict__ w,
                                   const float eps,
                                   const int n,
                                   float* __restrict__ out) {
  extern __shared__ float s_x[];
  __shared__ float s_inv;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    s_x[i] = x[i];  // coalesced stage into shared
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float ss = 0.0f;
    for (int i = 0; i < n; ++i) {
      const float xi = s_x[i];
      ss = __fadd_rn(ss, __fmul_rn(xi, xi));  // same i=0..n order as rmsnorm_f32
    }
    const float mean_sq = __fdiv_rn(ss, (float)n);
    const float denom = sqrtf(__fadd_rn(mean_sq, eps));
    s_inv = __fdiv_rn(1.0f, denom);
  }
  __syncthreads();
  const float inv = s_inv;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    out[i] = __fmul_rn(__fmul_rn(s_x[i], inv), w[i]);
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
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if (threadIdx.x < off) {
      const float o = s_red[threadIdx.x + off];
      if (o > s_red[threadIdx.x]) s_red[threadIdx.x] = o;
    }
    __syncthreads();
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
// reduction, so no f32 sum is reordered.
//   * scores: lane-per-key — each lane runs the FULL sequential `head_dim` dot for its
//     keys (j = lane, lane+32, …), so each dot keeps the host's d-order.
//   * softmax: lane 0 only, sequential over ctx (same max→exp→sum→divide order).
//   * weighted sum: lane-per-output-dim — lane `d` sums `Σ_j w[j]·v[j][d]` in the host's
//     j-order, so each output keeps its order.
// `__syncwarp` separates the phases. ctrl[2] = cache_len (ctx = +1, limit = cache_len);
// `scores` is strided by the fixed `max_ctx`.
__global__ void gqa_attention_decode_warp_g(const float* __restrict__ q,
                                            const float* __restrict__ k,
                                            const float* __restrict__ v,
                                            float* __restrict__ out,
                                            float* __restrict__ scores,
                                            const int* __restrict__ ctrl,
                                            const int max_ctx, const int n_head,
                                            const int n_head_kv, const int head_dim,
                                            const float scale) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= n_head) return;
  const int h = warp;
  const int cache_len = ctrl[2];
  const int ctx = cache_len + 1;
  const int limit = cache_len;
  const int n_rep = n_head / n_head_kv;
  const int kv = h / n_rep;
  const float* q_row = q + (long long)h * head_dim;
  float* sc = scores + (long long)h * max_ctx;

  // scores: lane-per-key; each lane's dot is sequential over head_dim (bit-exact).
  for (int j = lane; j < ctx; j += 32) {
    if (j > limit) {  // defensive — unreachable in decode (ctx == limit + 1), kept for parity
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
  __syncwarp();

  // softmax: lane 0 only, in the host's sequential order.
  if (lane == 0) {
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
  }
  __syncwarp();

  // weighted sum: lane-per-output-dim; each output sums over j in the host's order.
  float* o_row = out + (long long)h * head_dim;
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

}  // extern "C"
