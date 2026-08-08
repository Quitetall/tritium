// gqa_attention_v3.hip — v3 Q-blocked prefill GQA attention: the HIP twin of
// tritium-cuda's `gqa_attention_batch_v3_f32` (kernels/decode.cu, the
// `gqa_attention_batch_v3_body` family; Track E2). f32 KV only for now (the
// CUDA `_h` f16-KV twin is follow-up work, as on the Metal port).
//
// STATUS: compile/lint-verified only — authored on a Linux/NVIDIA box with no
// AMD GPU; first executed (and gated against the pinned-order host reference
// in `src/attn.rs`) in the budgeted MI300X cloud session. Runbook: the
// `src/attn.rs` module docs.
//
// The second reference for this port is the Metal twin
// (crates/tritium-metal/src/attention.metal), which established the
// twin+pinned-order discipline on a non-CUDA target; the HIP source below
// stays closer to the CUDA spelling because HIP is CUDA-shaped.
//
// ## Wavefront design: 32-lane LOGICAL lanes on any physical wave size
//
// CUDA launches 256 threads = 8 warps x 32 lanes. MI300X (CDNA3) wavefronts
// are 64-wide, RDNA defaults to 32-wide — so "warp" arithmetic cannot be
// written against warpSize. Two options were analyzed:
//
//   (a) Keep the CUDA 32-lane groups as LOGICAL lanes: warp = tid >> 5,
//       lane = tid & 31, exactly decode.cu. All cross-lane traffic uses
//       width-32 shuffles (`__shfl_xor(v, mask, 32)` / `__shfl(v, 0, 32)`),
//       which HIP defines for width < warpSize as independent 32-lane
//       segments. At wave64 one physical wavefront hosts TWO logical groups
//       (rows w and w+1); at wave32 the mapping is 1:1 with CUDA.
//   (b) Restructure to 64-lane rows (warp = tid >> 6, 4 wavefronts, RPW = 2).
//
// (a) is chosen. Order-preservation argument: every PINNED chain in v3 is
// sequential on a single thread — the phase-1 dot chain (one lane owns the
// whole d-ascending fold for its key), the phase-2 softmax sum (logical
// lane 0, sequential j), and the phase-3 V fold (one thread per output dim,
// sequential j). The only cross-lane operations are the phase-2 fmax
// butterfly (max is EXACT under any order — the rev-1/v2/v3 NaN/-0.0 corner
// arguments in decode.cu carry over verbatim) and a lane-0 value broadcast
// (no arithmetic). Under (a) the thread<->work mapping is IDENTICAL to CUDA
// at the logical level, so every pinned per-row order is preserved verbatim,
// and the constants (THREADS/BQ/KCH/WARPS/RPW) stay byte-for-byte equal to
// decode.cu's — reviewable line-by-line. (b) would also keep the sequential
// chains (they are per-thread), but it forks the constants per architecture,
// changes which thread owns which (row, key) pair, breaks the consts-pin
// twin test, and silently degrades on wave32 — rejected.
//
// The kernel is therefore wave-size-AGNOSTIC by construction: it needs only
//   * width-32 shuffle segments (legal for warpSize 32 and 64 — width must
//     be a power of two <= warpSize; every AMD GPU is one of the two), and
//   * workgroup barriers for cross-thread visibility (below).
// The host still refuses dispatch on a device whose probed warpSize is not
// 32 or 64 (the `attn_v3_wave_probe` kernel + init-time check in rocm.rs —
// the analogue of Metal's thread_execution_width == 32 refusal), so an
// unforeseen architecture falls back to the host reference instead of
// running an unvalidated mapping. A compile-time wave-size static_assert is
// deliberately NOT used: `__AMDGCN_WAVEFRONT_SIZE__` is deprecated (removed
// in ROCm 7), and the kernel does not depend on the value anyway.
//
// ## The one structural deviation from the CUDA source: the phase-2 barrier
//
// CUDA v3 has a `__syncwarp()` between the elementwise exp and the lane-0
// sequential sum (lane 0 reads every lane's exp writes to the global scores
// strip). At wave64 the enclosing `if (r < nrows)` guard is uniform per
// LOGICAL group but can diverge between the two halves of one physical
// wavefront, so a wave-level sync inside the guard would be
// non-wavefront-uniform. Instead, phase 2 is split into two guarded passes
// with one UNCONDITIONAL workgroup barrier between them:
//     2a: per-row butterfly max + elementwise exp      (guarded)
//     __syncthreads()                                   (every thread)
//     2b: per-row lane-0 sequential sum + broadcast + elementwise scale
//         (guarded)
// A workgroup barrier is strictly stronger than the warp sync it replaces,
// is reached by every thread (legal), and touches no arithmetic chain — the
// per-row order max -> exp -> sum -> scale is unchanged. (`__syncthreads()`
// on AMD is s_waitcnt + s_barrier; combined with the workgroup-coherent
// per-CU vector cache it orders global-memory writes within the workgroup —
// the same property phases 1->2 and 2->3 already rely on, in this kernel
// and in the CUDA original.)
//
// ## Pinned orders (the ADR 0022 twin contract, per (row, head))
//
//   * phase 1 dot: one lane owns the whole d-ascending add chain per key;
//   * phase 2: butterfly fmax (order-exact), elementwise exp, softmax SUM
//     sequential j-ascending on logical lane 0 only, one 1/sum divide,
//     elementwise scale;
//   * phase 3: per output dim, sequential j-ascending V fold across
//     ascending chunks with the rev-1 `w == 0.0f` zero-skip; keys past a
//     row's causal limit stage a 0 weight and take the same skip.
//
// ## Arithmetic discipline
//
// hipcc (clang) defaults to `-ffp-contract=fast-honor-pragmas` for HIP
// device code, which may fuse a separate mul + add into an FMA ACROSS
// statements — that would change every pinned chain's rounding. Two
// independent guards keep +,*,/ as IEEE round-to-nearest two-step ops
// (the HIP meaning of CUDA's __fadd_rn/__fmul_rn/__fdiv_rn):
//   * the file-scope `#pragma clang fp contract(off)` below (honored by the
//     default contract mode), and
//   * build.rs compiles this translation unit with `-ffp-contract=off`.
// Every mul-add in a pinned chain is additionally split into two statements
// through a named temporary (the Metal-port spelling), so even a conforming
// compiler with contraction enabled could not legally fuse it.
//
// The exponential: CUDA's exp_f32 is `__double2float_rn(exp((double)x))` —
// f64 exp rounded once, bit-identical to glibc expf for essentially every
// input. HIP HAS double (fast native f64 on CDNA), so this port keeps the
// same spelling; AMD's ocml f64 exp is ~1 ULP like CUDA's, so the round-once
// result is EXPECTED to match the host `f32::exp` bit-for-bit — but that is
// verified, not assumed: the MI300X gate asserts a tight exp-only tolerance
// as the hard bar and bit-equality as a second, separately-reported tier
// (see rocm.rs `attn_v3_matches_pinned_host_reference_or_skip`).

#include <hip/hip_runtime.h>

// Forbid mul+add contraction in every chain below (see the banner).
#pragma clang fp contract(off)

// MUST mirror decode.cu's ATTN_V3_* defines and src/attn.rs's constants (the
// attn.rs consts-pin test parses these lines; keep them bare numbers).
// ATTN_V3_HDMAX is decode.cu's ATTN_V2_HDMAX (v3 reuses the v2 bound there;
// this crate has no v2 kernel to borrow a name from — same note as Metal).
#define ATTN_V3_THREADS 256
#define ATTN_V3_BQ 8
#define ATTN_V3_KCH 32
#define ATTN_V3_HDMAX 128
// Logical 32-lane groups (CUDA warps), NOT physical wavefronts — see banner.
#define ATTN_V3_WARPS (ATTN_V3_THREADS / 32)
// Query rows owned per logical group in phases 1 and 2 (group w owns rows
// w, w+WARPS, ...).
#define ATTN_V3_RPW (ATTN_V3_BQ / ATTN_V3_WARPS)
// RPW = 0 (more groups than BQ rows) compiles phases 1/2 to nothing — every
// output garbage; catch it in every build (decode.cu's static_assert).
static_assert(ATTN_V3_BQ % ATTN_V3_WARPS == 0 && ATTN_V3_RPW >= 1,
              "ATTN_V3_BQ must be a positive multiple of ATTN_V3_WARPS");
// Phase 1 maps `key = tid & 31`, so a chunk can stage at most 32 keys (the
// CUDA/Metal sources share this assumption implicitly; pin it here).
// == not <=: the phase-2a "scores written by THIS thread" comment reasons
// from jglob % 32 == lane, which needs KCH exactly 32 (review 5593fda F3;
// a KCH retune must revisit that reasoning even though the phase-1 trailing
// __syncthreads already orders the writes either way).
static_assert(ATTN_V3_KCH == 32, "phase-1 key ownership is tid & 31");
// LDS budget: CDNA/GCN workgroups have 64 KiB of LDS (Apple's bound on the
// Metal twin is 32 KiB; the staging fits both).
static_assert((ATTN_V3_KCH * (ATTN_V3_HDMAX + 1) + ATTN_V3_BQ * ATTN_V3_HDMAX +
               ATTN_V3_BQ * (ATTN_V3_KCH + 1)) * 4 <= 65536,
              "v3 LDS staging exceeds the 64 KiB workgroup budget");

// exp_f32 — the softmax exponential, computed in **double** then rounded to
// float: the exact CUDA decode.cu spelling (HIP provides both `exp(double)`
// and `__double2float_rn`). See the banner for the bit-parity expectation.
__device__ __forceinline__ float exp_f32(float x) {
  return __double2float_rn(exp((double)x));
}

extern "C" {

// attn_v3_wave_probe — writes the device's physical wavefront width (the HIP
// device-side `warpSize` builtin) to `out[0]`. Launched once at backend init
// (1 block x 1 thread); the host refuses the v3 device rung unless the probe
// reports 32 or 64 (the dispatch check mirroring Metal's execution-width
// gate). A probe kernel is used instead of `hipDeviceGetAttribute` /
// `hipGetDeviceProperties` because the attribute enum values and the props
// struct layout have both changed across ROCm major releases, while a module
// launch + memcpy is the ABI surface this crate already depends on.
__global__ void attn_v3_wave_probe(int* __restrict__ out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = warpSize;
  }
}

// gqa_attention_batch_v3_f32 — grid (n_head, ceil(m / ATTN_V3_BQ)) blocks of
// ATTN_V3_THREADS, exactly the CUDA launch. q/out are [m, n_head, head_dim];
// k/v are KV arenas [>= causal_offset + m, n_head_kv, head_dim]; scores is
// the global [m, n_head, ctx_max] scratch (the lever that removes the v2 ctx
// cap — this crate never had a v2, but the scratch contract is the CUDA
// one). All per-row bases widen to 64-bit (the v0.6.9 indexing discipline,
// the same (long long) casts as decode.cu). Host contract (validated +
// bounded in src/attn.rs): head_dim <= ATTN_V3_HDMAX, causal_offset + m <=
// ctx_max, every scalar fits i32.
__global__ void gqa_attention_batch_v3_f32(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, float* __restrict__ out,
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
  // the per-chunk score strip phase 3 reads broadcast from LDS.
  __shared__ __align__(16) float s_kv[ATTN_V3_KCH * (ATTN_V3_HDMAX + 1)];
  __shared__ __align__(16) float s_q[ATTN_V3_BQ * ATTN_V3_HDMAX];
  __shared__ float s_w[ATTN_V3_BQ][ATTN_V3_KCH + 1];

  for (int idx = tid; idx < nrows * head_dim; idx += ATTN_V3_THREADS) {
    const int r = idx / head_dim;
    const int d = idx - r * head_dim;
    s_q[r * ATTN_V3_HDMAX + d] =
        q[((long long)(row0 + r) * n_head + h) * head_dim + d];
  }
  __syncthreads();

  const int kstride = head_dim + 1;

  // Phase 1: scaled dots. Each staged KCH-key chunk feeds all BQ rows:
  // thread t owns key t&31 for rows (t>>5) + rr*WARPS (lanes sharing a key
  // broadcast the staged K row; the 32 lanes of a logical group share a
  // query row -> broadcast s_q reads). Keys past a row's causal limit write
  // nothing.
  for (int c0 = 0; c0 < ctx_top; c0 += ATTN_V3_KCH) {
    const int nk = min(ATTN_V3_KCH, ctx_top - c0);
    for (int idx = tid; idx < nk * head_dim; idx += ATTN_V3_THREADS) {
      const int j = idx / head_dim;
      const int d = idx - j * head_dim;
      s_kv[j * kstride + d] =
          k[((long long)(c0 + j) * n_head_kv + kv) * head_dim + d];
    }
    __syncthreads();
    const int key = tid & 31;
    if (key < nk) {
      const float* kr = &s_kv[key * kstride];
      const int jglob = c0 + key;
#pragma unroll
      for (int rr = 0; rr < ATTN_V3_RPW; ++rr) {
        const int r = (tid >> 5) + rr * ATTN_V3_WARPS;
        if (r < nrows && jglob <= causal_offset + row0 + r) {
          const float* qr = &s_q[r * ATTN_V3_HDMAX];
          // PINNED: the d-ascending dot chain (rev-1's order). The named
          // temporary + contract(off) keep mul and add as two roundings.
          float dot = 0.0f;
          for (int d = 0; d < head_dim; ++d) {
            const float prod = qr[d] * kr[d];
            dot = dot + prod;
          }
          scores[((long long)(row0 + r) * n_head + h) * ctx_max + jglob] =
              dot * scale;
        }
      }
    }
    __syncthreads();
  }

  // Phase 2: per-row softmax; logical group w owns rows w, w+WARPS, ...
  // Butterfly fmaxf (exact under any order), elementwise exp, ORDER-PINNED
  // sequential sum on logical lane 0, elementwise inv scale. Split into two
  // guarded passes around ONE unconditional workgroup barrier — the wave64-
  // legal replacement for CUDA's in-guard __syncwarp (see banner); the
  // per-row order max -> exp -> sum -> scale is untouched. Phase-1 scores
  // for a row's stripe j were written by THIS thread (jglob % 32 == lane and
  // the same logical group, KCH == 32), so the strided reads in 2a need no
  // extra cross-thread ordering beyond the phase-1 barrier.
  const int warp = tid >> 5;
  const int lane = tid & 31;
  // 2a: butterfly max + elementwise exp.
#pragma unroll
  for (int rr = 0; rr < ATTN_V3_RPW; ++rr) {
    const int r = warp + rr * ATTN_V3_WARPS;
    if (r < nrows) {
      float* sc = scores + ((long long)(row0 + r) * n_head + h) * ctx_max;
      const int ctx = causal_offset + row0 + r + 1;
      float local = -INFINITY;
      for (int j = lane; j < ctx; j += 32) local = fmaxf(local, sc[j]);
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        local = fmaxf(local, __shfl_xor(local, off, 32));
      }
      const float mx = local;
      for (int j = lane; j < ctx; j += 32) {
        sc[j] = exp_f32(sc[j] - mx);
      }
    }
  }
  // Every thread reaches this barrier (the guards above only skip work):
  // logical lane 0 is about to read every lane's exp writes (global mem).
  __syncthreads();
  // 2b: lane-0 sequential sum, broadcast, elementwise scale.
#pragma unroll
  for (int rr = 0; rr < ATTN_V3_RPW; ++rr) {
    const int r = warp + rr * ATTN_V3_WARPS;
    if (r < nrows) {
      float* sc = scores + ((long long)(row0 + r) * n_head + h) * ctx_max;
      const int ctx = causal_offset + row0 + r + 1;
      float inv = 0.0f;
      if (lane == 0) {
        // PINNED: THE softmax sum — sequential j-ascending on one lane.
        float sum = 0.0f;
        for (int j = 0; j < ctx; ++j) {
          sum = sum + sc[j];
        }
        inv = 1.0f / sum;
      }
      inv = __shfl(inv, 0, 32);
      for (int j = lane; j < ctx; j += 32) {
        sc[j] = sc[j] * inv;
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
          v[((long long)(c0 + j) * n_head_kv + kv) * head_dim + d];
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
          // Rows >= nrows read UNSTAGED s_w garbage here — safe by design
          // (as in the CUDA twin): their acc[r] can only poison a register
          // that the nrows-guarded store below never emits (no FP traps on
          // this path).
          const float w = s_w[r][jj];
          if (w != 0.0f) {
            // PINNED: the per-(row, d) j-ascending fold chain.
            const float prod = w * vv;
            acc[r] = acc[r] + prod;
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

}  // extern "C"
