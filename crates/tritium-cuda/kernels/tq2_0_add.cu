// tq2_0_add.cu — addition-only ternary mixed-precision GEMM (mpGEMM).
//
// Mirrors Microsoft BitNet's W1.58A8 packing (4 trits per byte, 2 bits each,
// stored as `code = trit + 1 ∈ {0,1,2}`) — the same TQ2_0 byte layout produced
// host-side by `tritium-format` (a port of ggml `quantize_row_tq2_0_ref`). See
// https://github.com/microsoft/BitNet for the reference W1.58A8 kernels.
//
// Correctness-first, no shared-memory tiling: one output element per thread
// (the design's perf weight for this kernel is 0.30). Each thread walks the K
// dimension, decodes a 2-bit code into {-1, 0, +1}, and accumulates the
// activation by SIGN ONLY — add for +1, subtract for -1, skip for 0. No
// per-element multiply. The per-output-channel scale is the single multiply at
// the end. This must match `tritium_core::reference_mpgemm` within 1e-4:
//
//     out[m, n] = scales[n] * Σ_k act[m, k] * trit[n, k]
//
// Layout (all row-major, matching the Rust contract):
//   act     : f32 [M, K]
//   weights : u8  [N * row_bytes]   TQ2_0-packed, output-major
//   scales  : f32 [N]               per-output-channel
//   out     : f32 [M, N]
//
// TQ2_0 block geometry (QK_K = 256 trits per block):
//   * Each block is 66 bytes: 64 `qs` bytes then a 2-byte f16 scale (ignored
//     here — the block scale is fixed to 1.0 by the packer; channel scaling is
//     applied via `scales`).
//   * Within a block, element index e ∈ [0, 256) decodes as:
//       c = e / 128            (chunk 0 or 1)
//       m = e % 32             (byte within the chunk's 32-byte half)
//       l = (e % 128) / 32     (which 2-bit slot in that byte)
//     so the code lives in qs[c*32 + m] at bit position 2*l.
//   * A row of K trits spans nb = ceil(K / 256) such blocks laid end to end;
//     row_bytes = nb * 66. Trailing trits past K in the final block are padding
//     and are never read because the K loop stops at K.

#include <cuda_fp16.h>

#define QK_K 256
#define TQ2_0_BLOCK_BYTES 66
#define TQ2_0_QS_BYTES 64

extern "C" __global__ void tq2_0_add_mpgemm(
    const float* __restrict__ act,      // [M, K]
    const unsigned char* __restrict__ weights, // [N * row_bytes]
    const float* __restrict__ scales,   // [N]
    float* __restrict__ out,            // [M, N]
    const int m,
    const int n,
    const int k,
    const int row_bytes) {              // bytes per packed weight row (nb * 66)
    // One thread per output element. Global index over the [M, N] grid.
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)m * (long long)n;
    if (idx >= total) {
        return;
    }
    const int mi = (int)(idx / n);
    const int ni = (int)(idx % n);

    const float* arow = act + (long long)mi * k;
    const unsigned char* wrow = weights + (long long)ni * row_bytes;

    float acc = 0.0f;
    for (int ki = 0; ki < k; ++ki) {
        // Locate the packed 2-bit code for trit ki of this row.
        const int block = ki / QK_K;            // which 256-trit block
        const int e = ki - block * QK_K;        // element index within the block
        const int c = e >> 7;                   // e / 128  -> chunk 0/1
        const int mm = e & 31;                  // e % 32   -> byte within half
        const int l = (e & 127) >> 5;           // (e % 128) / 32 -> 2-bit slot
        const int byte_off = block * TQ2_0_BLOCK_BYTES + c * 32 + mm;
        const unsigned int code = ((unsigned int)wrow[byte_off] >> (2 * l)) & 3u;
        // code: 0 -> -1, 1 -> 0, 2 -> +1. Accumulate by sign, no multiply.
        const float a = arow[ki];
        if (code == 2u) {
            acc += a;
        } else if (code == 0u) {
            acc -= a;
        }
        // code == 1u (trit 0) and the unused code 3 contribute nothing.
    }

    out[(long long)mi * n + ni] = acc * scales[ni];
}

// tq2_0_add_mpgemm_tiled — the decode-oriented (memory-bound, small-M) sibling of
// the kernel above (v0.30 WF-A, ADR 0005). Same add/sub/skip arithmetic and the
// identical TQ2_0 decode, but parallelized for batch=1 decode where the floor is
// reading the weight bytes:
//
//   * One CUDA block owns one activation row `mi` and stages its `K` floats into
//     shared memory once (cooperatively), so every warp in the block reads the
//     activations from shared instead of re-fetching them from global.
//   * One *warp* computes one output `ni`: the 32 lanes split the K loop
//     (lane `l` walks ki = l, l+32, …), decode + accumulate by sign, then a
//     `__shfl_down_sync` tree-reduction sums the 32 partials. Lane 0 applies the
//     per-channel scale and writes the result.
//
// The lane-split + tree-reduction sums K in a different order than the sequential
// reference, so the result matches `reference_mpgemm` within the 1e-4 relative
// tolerance (ADR 0002) rather than bit-exactly — exactly what the contract asks of
// the float path. The host only routes shapes whose `K` activations fit the shared
// budget here (`K <= 8192` floats = 32 KiB); larger K falls back to the kernel
// above.
#define WARP_SIZE 32

extern "C" __global__ void tq2_0_add_mpgemm_tiled(
    const float* __restrict__ act,      // [M, K]
    const unsigned char* __restrict__ weights, // [N * row_bytes]
    const float* __restrict__ scales,   // [N]
    float* __restrict__ out,            // [M, N]
    const int m,
    const int n,
    const int k,
    const int row_bytes) {
    extern __shared__ float s_act[];    // K activations for this block's row `mi`

    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int mi = blockIdx.y;          // one block-row per output row

    // Defensive only: the host launches grid.y == m, so mi < m always holds here.
    // Kept so the kernel stays safe if a future launch ever over-dispatches Y.
    if (mi >= m) {
        return;
    }

    // Stage this row's activations into shared memory (all threads cooperate).
    const float* arow = act + (long long)mi * k;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        s_act[i] = arow[i];
    }
    __syncthreads();

    // This warp's output column. Warps past N have no work but must still have
    // reached the barrier above, so the early-out comes after `__syncthreads`.
    const int ni = blockIdx.x * warps_per_block + warp_id;
    if (ni >= n) {
        return;
    }

    const unsigned char* wrow = weights + (long long)ni * row_bytes;
    // Accumulate in double: the lane-split + tree-reduction sums K in a different
    // order than the sequential reference, so an f32 accumulator drifts past the
    // 1e-4 ternary tolerance on cancellation-heavy rows. A double accumulator keeps
    // the tiled result within 1e-4 of the f32 reference (and near-bit-identical for
    // the model forward, protecting v0.20 greedy parity).
    double acc = 0.0;
    for (int ki = lane; ki < k; ki += WARP_SIZE) {
        const int block = ki / QK_K;
        const int e = ki - block * QK_K;
        const int c = e >> 7;
        const int mm = e & 31;
        const int l = (e & 127) >> 5;
        const int byte_off = block * TQ2_0_BLOCK_BYTES + c * 32 + mm;
        const unsigned int code = ((unsigned int)wrow[byte_off] >> (2 * l)) & 3u;
        const double a = (double)s_act[ki];
        if (code == 2u) {
            acc += a;
        } else if (code == 0u) {
            acc -= a;
        }
    }

    // Tree-reduce the 32 lane partials in double. Full mask: every lane
    // participates (a lane with no ki simply contributes 0).
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }

    if (lane == 0) {
        out[(long long)mi * n + ni] = (float)(acc * (double)scales[ni]);
    }
}

// ─── Fused-scaled variants (v0.6.0 opt #15) ─────────────────────────────────
// Identical to the two kernels above, but the epilogue multiplies by both the
// per-channel weight scale AND the per-token activation scale in one write:
//     out[mi, ni] = acc * scales[ni] * act_scale[mi]
// This eliminates the separate `scale_mul_f32` kernel launch + its full
// read-write pass over the output buffer (~182 launches/token saved).

extern "C" __global__ void tq2_0_add_mpgemm_tiled_scaled(
    const float* __restrict__ act,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ scales,
    const float* __restrict__ act_scale,  // [M] per-token activation scale
    float* __restrict__ out,
    const int m,
    const int n,
    const int k,
    const int row_bytes) {
    extern __shared__ float s_act[];

    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int mi = blockIdx.y;
    if (mi >= m) {
        return;
    }

    const float* arow = act + (long long)mi * k;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        s_act[i] = arow[i];
    }
    __syncthreads();

    const int ni = blockIdx.x * warps_per_block + warp_id;
    if (ni >= n) {
        return;
    }

    const unsigned char* wrow = weights + (long long)ni * row_bytes;
    double acc = 0.0;
    for (int ki = lane; ki < k; ki += WARP_SIZE) {
        const int block = ki / QK_K;
        const int e = ki - block * QK_K;
        const int c = e >> 7;
        const int mm = e & 31;
        const int l = (e & 127) >> 5;
        const int byte_off = block * TQ2_0_BLOCK_BYTES + c * 32 + mm;
        const unsigned int code = ((unsigned int)wrow[byte_off] >> (2 * l)) & 3u;
        const double a = (double)s_act[ki];
        if (code == 2u) {
            acc += a;
        } else if (code == 0u) {
            acc -= a;
        }
    }
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        out[(long long)mi * n + ni] = (float)(acc * (double)scales[ni]) * act_scale[mi];
    }
}

// f32-accumulate variant of `tq2_0_add_mpgemm_tiled` for the v0.3.2 CUDA-graph decode
// (perf) path. The double accumulator above is correct but SLOW on consumer GPUs: the
// RTX 4090 runs f64 at 1/64 the f32 rate, and the decode forward issues ~210 of these
// per token — measured as the decode bottleneck. Accumulating + tree-reducing in f32
// reorders the K sum past the 1e-4 bit-match tolerance on cancellation-heavy rows, so the
// graph path is gated by **perplexity ≤ 1% + lockstep argmax**, not greedy bit-match (the
// eager `step` keeps the double kernel and its 256/256 parity).
//
// Byte-once K loop (v1.x decode opt): the TQ2_0 layout stores the four trits at
// e, e+32, e+64, e+96 (within a 128-trit chunk) in ONE byte's four 2-bit slots, so
// the old `ki += 32` lane stride re-read the same byte four times. Here each lane
// reads its chunk byte once and consumes all four slots — 4× fewer weight-byte
// loads and 4× less index math per trit. Measured ~2× on the 4090 decode shapes.
extern "C" __global__ void tq2_0_add_mpgemm_tiled_f32(
    const float* __restrict__ act,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ scales,
    float* __restrict__ out,
    const int m,
    const int n,
    const int k,
    const int row_bytes) {
    extern __shared__ float s_act[];

    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int mi = blockIdx.y;
    if (mi >= m) {
        return;
    }
    const float* arow = act + (long long)mi * k;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        s_act[i] = arow[i];
    }
    __syncthreads();

    const int ni = blockIdx.x * warps_per_block + warp_id;
    if (ni >= n) {
        return;
    }
    const unsigned char* wrow = weights + (long long)ni * row_bytes;
    float acc = 0.0f;
    // Full 128-trit chunks: lane `l` owns byte `qs[c*32 + l]`, whose four 2-bit
    // slots are the trits at ki = base + slot*32 + l. Branchless decode:
    // trit = code - 1 ∈ {-1, 0, +1} for the valid codes {0,1,2}, so `acc += a * trit`
    // equals the add/sub/skip branch bit-for-bit (a*1=a, a*-1=-a, a*0=0 → no-op add).
    const int full = k & ~127;
    int base = 0;
    for (; base < full; base += 128) {
        const int block = base >> 8;               // base / QK_K
        const int c = (base >> 7) & 1;             // which 128-trit chunk
        const unsigned int byte = wrow[block * TQ2_0_BLOCK_BYTES + c * 32 + lane];
        const float* sa = s_act + base + lane;
        acc += sa[0] * (float)((int)(byte & 3u) - 1);
        acc += sa[32] * (float)((int)((byte >> 2) & 3u) - 1);
        acc += sa[64] * (float)((int)((byte >> 4) & 3u) - 1);
        acc += sa[96] * (float)((int)((byte >> 6) & 3u) - 1);
    }
    // Tail (k % 128 != 0): the original per-trit decode.
    for (int ki = base + lane; ki < k; ki += WARP_SIZE) {
        const int block = ki / QK_K;
        const int e = ki - block * QK_K;
        const int c = e >> 7;
        const int mm = e & 31;
        const int l = (e & 127) >> 5;
        const int byte_off = block * TQ2_0_BLOCK_BYTES + c * 32 + mm;
        const unsigned int code = ((unsigned int)wrow[byte_off] >> (2 * l)) & 3u;
        acc += s_act[ki] * (float)((int)code - 1);
    }
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        out[(long long)mi * n + ni] = acc * scales[ni];
    }
}

// Sparse-aware variant of `tq2_0_add_mpgemm_tiled_f32`. Accepts a pre-computed
// per-row bitmap where each bit indicates whether the corresponding 256-trit
// block is all-zero (every qs byte == 0x55). When a block's bit is SET, the
// kernel skips the decode + accumulate for that block entirely — saving memory
// bandwidth and compute for the ~42.8% of ternary weights that are zero.
//
// The bitmap is laid out as one u32 per 32 blocks, per row. Row `ni`'s bitmap
// starts at `block_bitmap[ni * words_per_row]` where
// `words_per_row = ceil(k / (QK_K * 32))`.
//
// If `block_bitmap` is NULL, behavior is identical to the non-sparse variant
// (no blocks are skipped).
extern "C" __global__ void tq2_0_add_mpgemm_tiled_f32_sparse(
    const float* __restrict__ act,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ scales,
    const unsigned int* __restrict__ block_bitmap,
    float* __restrict__ out,
    const int m,
    const int n,
    const int k,
    const int row_bytes,
    const int words_per_row) {
    extern __shared__ float s_act[];

    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int mi = blockIdx.y;
    if (mi >= m) {
        return;
    }
    const float* arow = act + (long long)mi * k;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        s_act[i] = arow[i];
    }
    __syncthreads();

    const int ni = blockIdx.x * warps_per_block + warp_id;
    if (ni >= n) {
        return;
    }
    const unsigned char* wrow = weights + (long long)ni * row_bytes;
    // Pointer to this row's bitmap words.
    const unsigned int* row_bm = block_bitmap ? block_bitmap + (long long)ni * words_per_row : 0;
    float acc = 0.0f;
    // Byte-once chunk loop (see `tq2_0_add_mpgemm_tiled_f32`), with the per-256-block
    // bitmap check hoisted to once per 128-trit chunk (both chunks of a block share
    // the block's bit).
    const int full = k & ~127;
    int base = 0;
    for (; base < full; base += 128) {
        const int block = base >> 8;
        if (row_bm && (row_bm[block / 32] & (1u << (block % 32)))) {
            continue;
        }
        const int c = (base >> 7) & 1;
        const unsigned int byte = wrow[block * TQ2_0_BLOCK_BYTES + c * 32 + lane];
        const float* sa = s_act + base + lane;
        acc += sa[0] * (float)((int)(byte & 3u) - 1);
        acc += sa[32] * (float)((int)((byte >> 2) & 3u) - 1);
        acc += sa[64] * (float)((int)((byte >> 4) & 3u) - 1);
        acc += sa[96] * (float)((int)((byte >> 6) & 3u) - 1);
    }
    for (int ki = base + lane; ki < k; ki += WARP_SIZE) {
        const int block = ki / QK_K;
        if (row_bm && (row_bm[block / 32] & (1u << (block % 32)))) {
            continue;
        }
        const int e = ki - block * QK_K;
        const int c = e >> 7;
        const int mm = e & 31;
        const int l = (e & 127) >> 5;
        const int byte_off = block * TQ2_0_BLOCK_BYTES + c * 32 + mm;
        const unsigned int code = ((unsigned int)wrow[byte_off] >> (2 * l)) & 3u;
        acc += s_act[ki] * (float)((int)code - 1);
    }
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        out[(long long)mi * n + ni] = acc * scales[ni];
    }
}

// ─── DP4A fused-scaled kernels (v1.x decode opt) ─────────────────────────────
//
// The `_scaled` family is only ever launched on **A8-quantized activations**:
// every call site feeds `d_qact`, now the PACKED INT8 output of
// `act_quant_tiled_i8` / `rmsnorm_quant_i8` / `act_quant_batch_i8` (the
// W1.58A8 protocol; `act_scale` carries the dequant factor). The contraction
// runs on the int8 dot-product units via `__dp4a` (sm_61+, under the
// compute_75 build floor):
//
//   * Activations are read DIRECTLY from global as packed int8x4 words via
//     `__ldg` — no shared staging, no block-wide barrier. The row is tiny
//     (K bytes ≤ 6.9 KiB) and every block reads the same one, so it lives in
//     L1/L2 after the first touch; the earlier f32 staging design made every
//     N-column block re-read a 4×-bigger f32 row through L2 (more traffic
//     than the weights themselves at N=2560). Rows must be 4-byte aligned:
//     the host guarantees `k % 4 == 0` (all decode dims are multiples of 256).
//   * Each lane reads FOUR consecutive qs bytes as two u16 loads (the 66-byte
//     block stride only guarantees 2-byte alignment) — one byte-once load of
//     16 trits — and extracts one 2-bit slot across all four bytes at a time:
//     `(w >> 2*slot) & 0x03030303` then a per-byte `- 1` (`__vsub4`) yields
//     four int8 trits ∈ {-1,0,+1} ready for one `__dp4a` against the packed
//     activation word. Lane l owns slot = l/8 and byte quad j = l%8, covering
//     a whole 256-trit block per iteration with two independent accumulators.
//   * The int32 accumulate is EXACT (|Σ| ≤ 127·K < 2³¹) and order-independent,
//     so the result is bit-identical to the f32-fold original on this path:
//     f32 sums of integer-valued products also stayed exact (< 2²⁴), and the
//     epilogue multiplies are unchanged. Greedy/perplexity gates are unaffected.
//
// Measured on the RTX 4090 (M=1, standalone harness): ~2.5× vs the pre-dp4a
// f32 kernel at N=K=2560 and ~70–85% of the weight-bandwidth speed-of-light
// across the decode shapes. These kernels use NO dynamic shared memory.
//
// Named `_i8_scaled` (not `_f32_scaled`) so any stale host code that still
// passes an f32 activation buffer fails at kernel-resolve time, not silently.

// Fused-scaled variant (v0.6.0 opt #15 epilogue). DP4A contraction, see above.
extern "C" __global__ void tq2_0_add_mpgemm_tiled_i8_scaled(
    const signed char* __restrict__ qact,  // [M, K] packed int8, k % 4 == 0
    const unsigned char* __restrict__ weights,
    const float* __restrict__ scales,
    const float* __restrict__ act_scale,
    float* __restrict__ out,
    const int m,
    const int n,
    const int k,
    const int row_bytes) {
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int mi = blockIdx.y;
    if (mi >= m) {
        return;
    }
    const int* arow = (const int*)(qact + (long long)mi * k);
    const int ni = blockIdx.x * warps_per_block + warp_id;
    if (ni >= n) {
        return;
    }
    const unsigned char* wrow = weights + (long long)ni * row_bytes;
    int acc0 = 0;
    int acc1 = 0;
    const int slot = lane >> 3;        // which 2-bit slot this lane extracts
    const int j4 = (lane & 7) * 4;     // first of this lane's four qs bytes
    const int nblocks = k >> 8;        // full 256-trit blocks
    for (int b = 0; b < nblocks; ++b) {
        const unsigned char* qs = wrow + b * TQ2_0_BLOCK_BYTES + j4;
        const unsigned int w0 = (unsigned int)*(const unsigned short*)qs |
                                ((unsigned int)*(const unsigned short*)(qs + 2) << 16);
        const unsigned int w1 = (unsigned int)*(const unsigned short*)(qs + 32) |
                                ((unsigned int)*(const unsigned short*)(qs + 34) << 16);
        const int kidx = (b << 8) + slot * 32 + j4;
        acc0 = __dp4a((int)__vsub4((w0 >> (2 * slot)) & 0x03030303u, 0x01010101u),
                      __ldg(arow + (kidx >> 2)), acc0);
        acc1 = __dp4a((int)__vsub4((w1 >> (2 * slot)) & 0x03030303u, 0x01010101u),
                      __ldg(arow + ((kidx + 128) >> 2)), acc1);
    }
    int acc = acc0 + acc1;
    // Tail (k % 256 != 0): scalar per-trit decode in exact int math.
    const signed char* abytes = (const signed char*)arow;
    for (int ki = (k & ~255) + lane; ki < k; ki += WARP_SIZE) {
        const int block = ki / QK_K;
        const int e = ki - block * QK_K;
        const int c = e >> 7;
        const int mm = e & 31;
        const int l = (e & 127) >> 5;
        const unsigned int code =
            ((unsigned int)wrow[block * TQ2_0_BLOCK_BYTES + c * 32 + mm] >> (2 * l)) & 3u;
        acc += ((int)code - 1) * (int)abytes[ki];
    }
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        out[(long long)mi * n + ni] = (float)acc * scales[ni] * act_scale[mi];
    }
}

// Sparse-aware variant of `tq2_0_add_mpgemm_tiled_i8_scaled`. Same DP4A
// contraction + packed-int8-activation contract; same bitmap layout as
// `tq2_0_add_mpgemm_tiled_f32_sparse` (one bit per 256-trit block, set = all-
// zero, skipped). NULL bitmap → identical to the non-sparse variant.
extern "C" __global__ void tq2_0_add_mpgemm_tiled_i8_scaled_sparse(
    const signed char* __restrict__ qact,  // [M, K] packed int8, k % 4 == 0
    const unsigned char* __restrict__ weights,
    const float* __restrict__ scales,
    const float* __restrict__ act_scale,
    const unsigned int* __restrict__ block_bitmap,
    float* __restrict__ out,
    const int m,
    const int n,
    const int k,
    const int row_bytes,
    const int words_per_row) {
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int mi = blockIdx.y;
    if (mi >= m) {
        return;
    }
    const int* arow = (const int*)(qact + (long long)mi * k);
    const int ni = blockIdx.x * warps_per_block + warp_id;
    if (ni >= n) {
        return;
    }
    const unsigned char* wrow = weights + (long long)ni * row_bytes;
    const unsigned int* row_bm = block_bitmap ? block_bitmap + (long long)ni * words_per_row : 0;
    int acc0 = 0;
    int acc1 = 0;
    const int slot = lane >> 3;
    const int j4 = (lane & 7) * 4;
    const int nblocks = k >> 8;
    for (int b = 0; b < nblocks; ++b) {
        if (row_bm && (row_bm[b / 32] & (1u << (b % 32)))) {
            continue;
        }
        const unsigned char* qs = wrow + b * TQ2_0_BLOCK_BYTES + j4;
        const unsigned int w0 = (unsigned int)*(const unsigned short*)qs |
                                ((unsigned int)*(const unsigned short*)(qs + 2) << 16);
        const unsigned int w1 = (unsigned int)*(const unsigned short*)(qs + 32) |
                                ((unsigned int)*(const unsigned short*)(qs + 34) << 16);
        const int kidx = (b << 8) + slot * 32 + j4;
        acc0 = __dp4a((int)__vsub4((w0 >> (2 * slot)) & 0x03030303u, 0x01010101u),
                      __ldg(arow + (kidx >> 2)), acc0);
        acc1 = __dp4a((int)__vsub4((w1 >> (2 * slot)) & 0x03030303u, 0x01010101u),
                      __ldg(arow + ((kidx + 128) >> 2)), acc1);
    }
    int acc = acc0 + acc1;
    const signed char* abytes = (const signed char*)arow;
    for (int ki = (k & ~255) + lane; ki < k; ki += WARP_SIZE) {
        const int block = ki / QK_K;
        if (row_bm && (row_bm[block / 32] & (1u << (block % 32)))) {
            continue;
        }
        const int e = ki - block * QK_K;
        const int c = e >> 7;
        const int mm = e & 31;
        const int l = (e & 127) >> 5;
        const unsigned int code =
            ((unsigned int)wrow[block * TQ2_0_BLOCK_BYTES + c * 32 + mm] >> (2 * l)) & 3u;
        acc += ((int)code - 1) * (int)abytes[ki];
    }
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        out[(long long)mi * n + ni] = (float)acc * scales[ni] * act_scale[mi];
    }
}

// Fused-scaled + residual-add variant (v0.7.0 opt Phase 2 epilogue, DP4A
// contraction — same packed-int8-activation contract as above).
// Epilogue: out[mi,ni] = residual[mi,ni] + acc * scales[ni] * act_scale[mi]
// Eliminates the separate residual_add_f32 kernel launch + its memory pass.
extern "C" __global__ void tq2_0_add_mpgemm_tiled_i8_scaled_residual(
    const signed char* __restrict__ qact,  // [M, K] packed int8, k % 4 == 0
    const unsigned char* __restrict__ weights,
    const float* __restrict__ scales,
    const float* __restrict__ act_scale,
    const float* __restrict__ residual,
    float* __restrict__ out,
    const int m,
    const int n,
    const int k,
    const int row_bytes) {
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int mi = blockIdx.y;
    if (mi >= m) {
        return;
    }
    const int* arow = (const int*)(qact + (long long)mi * k);
    const int ni = blockIdx.x * warps_per_block + warp_id;
    if (ni >= n) {
        return;
    }
    const unsigned char* wrow = weights + (long long)ni * row_bytes;
    int acc0 = 0;
    int acc1 = 0;
    const int slot = lane >> 3;
    const int j4 = (lane & 7) * 4;
    const int nblocks = k >> 8;
    for (int b = 0; b < nblocks; ++b) {
        const unsigned char* qs = wrow + b * TQ2_0_BLOCK_BYTES + j4;
        const unsigned int w0 = (unsigned int)*(const unsigned short*)qs |
                                ((unsigned int)*(const unsigned short*)(qs + 2) << 16);
        const unsigned int w1 = (unsigned int)*(const unsigned short*)(qs + 32) |
                                ((unsigned int)*(const unsigned short*)(qs + 34) << 16);
        const int kidx = (b << 8) + slot * 32 + j4;
        acc0 = __dp4a((int)__vsub4((w0 >> (2 * slot)) & 0x03030303u, 0x01010101u),
                      __ldg(arow + (kidx >> 2)), acc0);
        acc1 = __dp4a((int)__vsub4((w1 >> (2 * slot)) & 0x03030303u, 0x01010101u),
                      __ldg(arow + ((kidx + 128) >> 2)), acc1);
    }
    int acc = acc0 + acc1;
    const signed char* abytes = (const signed char*)arow;
    for (int ki = (k & ~255) + lane; ki < k; ki += WARP_SIZE) {
        const int block = ki / QK_K;
        const int e = ki - block * QK_K;
        const int c = e >> 7;
        const int mm = e & 31;
        const int l = (e & 127) >> 5;
        const unsigned int code =
            ((unsigned int)wrow[block * TQ2_0_BLOCK_BYTES + c * 32 + mm] >> (2 * l)) & 3u;
        acc += ((int)code - 1) * (int)abytes[ki];
    }
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        const long long idx = (long long)mi * n + ni;
        out[idx] = __fadd_rn(residual[idx], (float)acc * scales[ni] * act_scale[mi]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SALT multi-plane accumulate (v0.4.0, ADR 0001/0006).
//
// A SALT-quantized weight row is a sum of T ternary planes, each a standard TQ2_0
// row whose per-256-block f16 scale carries that plane's per-group AbsMean scale:
//
//     W[ni, k] = Σ_p scale_p[ni, block(k)] · trit_p[ni, k]
//
// so the GEMM is
//
//     out[mi, ni] = Σ_k act[mi, k] · W[ni, k]
//                 = Σ_p Σ_k act[mi, k] · scale_p[block(k)] · trit_p[ni, k].
//
// This MUST match `tritium_format::dequant_salt_row` → fp32 reference matmul within
// 1e-4 (the lane-split + warp tree-reduce reorders the K-sum, exactly like the f32
// tiled kernel above is within 1e-4 of the sequential reference). Unlike the plain
// decode GEMM, the per-block f16 scale is **read from the weight bytes** and applied
// per term — the decode path instead fixes the block scale to 1.0 and folds a single
// per-channel scale at the end.
//
// Planes are laid out plane-major: plane p of row ni starts at
//   weights + p*plane_stride + ni*row_bytes,   with plane_stride = N * row_bytes.
extern "C" __global__ void salt_mpgemm_tiled_f32(
    const float* __restrict__ act,             // [M, K]
    const unsigned char* __restrict__ weights, // [T, N, row_bytes] plane-major
    float* __restrict__ out,                   // [M, N]
    const int m,
    const int n,
    const int k,
    const int row_bytes,                       // nb * 66
    const int t_planes,
    const long long plane_stride) {            // N * row_bytes
    extern __shared__ float s_act[];

    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int mi = blockIdx.y;
    if (mi >= m) {
        return;
    }

    const float* arow = act + (long long)mi * k;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        s_act[i] = arow[i];
    }
    __syncthreads();

    const int ni = blockIdx.x * warps_per_block + warp_id;
    if (ni >= n) {
        return;
    }

    float acc = 0.0f;
    for (int p = 0; p < t_planes; ++p) {
        const unsigned char* wrow =
            weights + (long long)p * plane_stride + (long long)ni * row_bytes;
        for (int ki = lane; ki < k; ki += WARP_SIZE) {
            const int block = ki / QK_K;
            const int e = ki - block * QK_K;
            const int c = e >> 7;
            const int mm = e & 31;
            const int l = (e & 127) >> 5;
            const int blk = block * TQ2_0_BLOCK_BYTES;
            const unsigned int code = ((unsigned int)wrow[blk + c * 32 + mm] >> (2 * l)) & 3u;
            // Per-block f16 scale = the block's last 2 bytes, little-endian, exactly
            // as `tritium_format::read_scale` reads it. __half2float is exact (f16 ⊂ f32).
            const unsigned short sbits =
                (unsigned short)wrow[blk + TQ2_0_QS_BYTES] |
                ((unsigned short)wrow[blk + TQ2_0_QS_BYTES + 1] << 8);
            const float s = __half2float(__ushort_as_half(sbits));
            acc += s_act[ki] * (float)((int)code - 1) * s;
        }
    }
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        out[(long long)mi * n + ni] = acc;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TQ1_0-native decode GEMMs (A2, Track A): the entropy-dense weight format
// (1.625 b/w payload vs TQ2_0's 2.0) read NATIVELY — ~21% less weight traffic
// on the DRAM-bound GEMMs, ~21% less resident weight VRAM.
//
// TQ1_0 block (54 B): qs[48] + qh[4] + f16 scale, 5 trits/byte (3^5 = 243).
// Decode contract — must match `tritium_format::tq1::decode` EXACTLY:
//     q = (uint8_t)(byte * pow3[n]);   trit = ((q * 3) >> 8) - 1   ∈ {-1,0,1}
// Element map (per 256-trit block):
//     e = n*32 + m        ← qs[m]       n ∈ 0..5, m ∈ 0..32
//     e = 160 + n*16 + m  ← qs[32+m]    n ∈ 0..5, m ∈ 0..16
//     e = 240 + n*4 + j   ← qh[j]       n ∈ 0..4, j ∈ 0..4
//
// Every element base above is 4-aligned, so a block is exactly 64 dp4a words:
// 40 (5 levels × 8 byte-quads) + 20 (5 × 4) + 4 (qh: 4 levels × 1 quad) —
// two words per lane, the same acc0/acc1 shape as the TQ2 kernel. The word→
// (byte offset, pow3, element) map is loop-invariant per lane. Blocks are
// 54 B (≡ 2 mod 4), so quads load as two u16s exactly like the TQ2 kernel.
//
// Integer accumulation ⇒ bit-exact vs the TQ2 kernel on the same trits
// (gated by `tq1_matches_tq2_tiled_scaled`).
#define TQ1_0_BLOCK_BYTES 54

__device__ __forceinline__ int tq1_decode4(unsigned int bytes4, unsigned int p3) {
    int w8 = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned int b = (bytes4 >> (8 * j)) & 0xffu;
        const unsigned int q = (b * p3) & 0xffu;   // u8 wraparound is the scheme
        const int t = (int)((q * 3u) >> 8) - 1;    // {-1, 0, 1}
        w8 |= (t & 0xff) << (8 * j);
    }
    return w8;
}

__device__ __forceinline__ void tq1_word(int w, int* boff, unsigned int* p3, int* e) {
    const unsigned int P3[5] = {1u, 3u, 9u, 27u, 81u};
    if (w < 40) {
        const int nn = w >> 3, g = w & 7;
        *boff = g * 4;
        *p3 = P3[nn];
        *e = nn * 32 + g * 4;
    } else if (w < 60) {
        const int v = w - 40, nn = v >> 2, g = v & 3;
        *boff = 32 + g * 4;
        *p3 = P3[nn];
        *e = 160 + nn * 16 + g * 4;
    } else {
        const int nn = w - 60;
        *boff = 48;
        *p3 = P3[nn];
        *e = 240 + nn * 4;
    }
}

// Scalar single-trit decode for the k % 256 tail (exact same contract).
__device__ __forceinline__ int tq1_trit(const unsigned char* blk, int e) {
    const unsigned int P3[5] = {1u, 3u, 9u, 27u, 81u};
    unsigned int byte, p3;
    if (e < 160) {
        byte = blk[e & 31];
        p3 = P3[e >> 5];
    } else if (e < 240) {
        const int r = e - 160;
        byte = blk[32 + (r & 15)];
        p3 = P3[r >> 4];
    } else {
        const int r = e - 240;
        byte = blk[48 + (r & 3)];
        p3 = P3[r >> 2];
    }
    const unsigned int q = (byte * p3) & 0xffu;
    return (int)((q * 3u) >> 8) - 1;
}

extern "C" __global__ void tq1_0_add_mpgemm_tiled_i8_scaled(
    const signed char* __restrict__ qact,  // [M, K] packed int8, k % 4 == 0
    const unsigned char* __restrict__ weights,
    const float* __restrict__ scales,
    const float* __restrict__ act_scale,
    float* __restrict__ out,
    const int m,
    const int n,
    const int k,
    const int row_bytes) {
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int mi = blockIdx.y;
    if (mi >= m) {
        return;
    }
    const int* arow = (const int*)(qact + (long long)mi * k);
    const int ni = blockIdx.x * warps_per_block + warp_id;
    if (ni >= n) {
        return;
    }
    const unsigned char* wrow = weights + (long long)ni * row_bytes;
    int boff0, boff1, e0, e1;
    unsigned int p30, p31;
    tq1_word(lane, &boff0, &p30, &e0);
    tq1_word(lane + 32, &boff1, &p31, &e1);
    int acc0 = 0;
    int acc1 = 0;
    const int nblocks = k >> 8;
    for (int b = 0; b < nblocks; ++b) {
        const unsigned char* blk = wrow + b * TQ1_0_BLOCK_BYTES;
        const unsigned char* s0 = blk + boff0;
        const unsigned char* s1 = blk + boff1;
        const unsigned int b0 = (unsigned int)*(const unsigned short*)s0 |
                                ((unsigned int)*(const unsigned short*)(s0 + 2) << 16);
        const unsigned int b1 = (unsigned int)*(const unsigned short*)s1 |
                                ((unsigned int)*(const unsigned short*)(s1 + 2) << 16);
        const int kbase = b << 8;
        acc0 = __dp4a(tq1_decode4(b0, p30), __ldg(arow + ((kbase + e0) >> 2)), acc0);
        acc1 = __dp4a(tq1_decode4(b1, p31), __ldg(arow + ((kbase + e1) >> 2)), acc1);
    }
    int acc = acc0 + acc1;
    // Tail (k % 256 != 0): scalar per-trit decode in exact int math.
    const signed char* abytes = (const signed char*)arow;
    for (int ki = (k & ~255) + lane; ki < k; ki += WARP_SIZE) {
        const int block = ki / QK_K;
        acc += tq1_trit(wrow + block * TQ1_0_BLOCK_BYTES, ki - block * QK_K) *
               (int)abytes[ki];
    }
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        out[(long long)mi * n + ni] = (float)acc * scales[ni] * act_scale[mi];
    }
}

// Residual twin: identical contraction, `out = residual + acc·scales·act_scale`
// epilogue (mirrors `tq2_0_add_mpgemm_tiled_i8_scaled_residual`).
extern "C" __global__ void tq1_0_add_mpgemm_tiled_i8_scaled_residual(
    const signed char* __restrict__ qact,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ scales,
    const float* __restrict__ act_scale,
    const float* __restrict__ residual,
    float* __restrict__ out,
    const int m,
    const int n,
    const int k,
    const int row_bytes) {
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int mi = blockIdx.y;
    if (mi >= m) {
        return;
    }
    const int* arow = (const int*)(qact + (long long)mi * k);
    const int ni = blockIdx.x * warps_per_block + warp_id;
    if (ni >= n) {
        return;
    }
    const unsigned char* wrow = weights + (long long)ni * row_bytes;
    int boff0, boff1, e0, e1;
    unsigned int p30, p31;
    tq1_word(lane, &boff0, &p30, &e0);
    tq1_word(lane + 32, &boff1, &p31, &e1);
    int acc0 = 0;
    int acc1 = 0;
    const int nblocks = k >> 8;
    for (int b = 0; b < nblocks; ++b) {
        const unsigned char* blk = wrow + b * TQ1_0_BLOCK_BYTES;
        const unsigned char* s0 = blk + boff0;
        const unsigned char* s1 = blk + boff1;
        const unsigned int b0 = (unsigned int)*(const unsigned short*)s0 |
                                ((unsigned int)*(const unsigned short*)(s0 + 2) << 16);
        const unsigned int b1 = (unsigned int)*(const unsigned short*)s1 |
                                ((unsigned int)*(const unsigned short*)(s1 + 2) << 16);
        const int kbase = b << 8;
        acc0 = __dp4a(tq1_decode4(b0, p30), __ldg(arow + ((kbase + e0) >> 2)), acc0);
        acc1 = __dp4a(tq1_decode4(b1, p31), __ldg(arow + ((kbase + e1) >> 2)), acc1);
    }
    int acc = acc0 + acc1;
    const signed char* abytes = (const signed char*)arow;
    for (int ki = (k & ~255) + lane; ki < k; ki += WARP_SIZE) {
        const int block = ki / QK_K;
        acc += tq1_trit(wrow + block * TQ1_0_BLOCK_BYTES, ki - block * QK_K) *
               (int)abytes[ki];
    }
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        const long long idx = (long long)mi * n + ni;
        out[idx] = __fadd_rn(residual[idx], (float)acc * scales[ni] * act_scale[mi]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TB1 bitmap+signs GEMM (A4 prototype): presence plane (1 b/w) + compacted
// sign stream (1 b/nonzero) = 1 + (1-p) b/w — 1.578 at BitNet's p=0.422,
// under dense TQ1_0's 1.625, sparsity-adaptive, and ZEROS COST NO SIGN BIT:
// element-level skipping falls out of the layout.
//
// Row layout: [presence: nb·32 B][signs: bit per nonzero, 4B-aligned]. Rows
// are variable-length → per-row byte offsets. The warp walks blocks in order
// keeping a RUNNING sign-bit base; within a block each lane owns one
// presence byte (8 elements = 2 dp4a words), its sign-stream position found
// by a warp-wide popcount prefix scan (5 shfl_up) — no divergence, dense
// coalesced loads. k % 256 == 0 required (prototype; gateup shapes qualify).
//
// ARENA CONTRACT: the weights arena MUST carry >= 2 (use 4) trailing slack
// bytes past the last row. The sign window reads signs[byte0] and
// signs[byte0+1] unconditionally — a trailing cnt==0 lane on the LAST row
// with nnz % 32 in {0, 25..=31} lands byte0(+1) up to 2 bytes past the
// packed data. Host uploaders must append the slack (see run_tb1 / the
// bench harness).
extern "C" __global__ void tb1_mpgemm_tiled_i8_scaled(
    const signed char* __restrict__ qact,   // [M, K] packed int8, k % 4 == 0
    const unsigned char* __restrict__ weights,
    const unsigned int* __restrict__ row_offsets,  // [N] byte offset per row
    const float* __restrict__ scales,
    const float* __restrict__ act_scale,
    float* __restrict__ out,
    const int m,
    const int n,
    const int k) {
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int mi = blockIdx.y;
    if (mi >= m) {
        return;
    }
    const int* arow = (const int*)(qact + (long long)mi * k);
    const int ni = blockIdx.x * warps_per_block + warp_id;
    if (ni >= n) {
        return;
    }
    const int nb = k >> 8;
    const unsigned char* presence = weights + row_offsets[ni];
    const unsigned char* signs = presence + nb * 32;
    int acc = 0;
    int sign_base = 0;  // running bit offset into this row's sign stream
    for (int b = 0; b < nb; ++b) {
        const unsigned int pb = presence[b * 32 + lane];
        const int cnt = __popc(pb);
        // Inclusive prefix sum of per-lane popcounts across the warp.
        int pre = cnt;
#pragma unroll
        for (int off = 1; off < WARP_SIZE; off <<= 1) {
            const int up = __shfl_up_sync(0xffffffffu, pre, off);
            if (lane >= off) {
                pre += up;
            }
        }
        const int my_start = sign_base + pre - cnt;  // exclusive prefix
        // ≤8 sign bits span ≤2 bytes of the stream.
        const int byte0 = my_start >> 3;
        const unsigned int sw =
            ((unsigned int)signs[byte0] | ((unsigned int)signs[byte0 + 1] << 8)) >>
            (my_start & 7);
        // Two dp4a words from the byte's low/high nibbles; the high nibble's
        // signs start after the low nibble's nonzeros.
        const int e = (b << 8) + lane * 8;
        int sbit = 0;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            int w8 = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int idx = half * 4 + j;
                if (pb & (1u << idx)) {
                    const int t = (sw >> sbit) & 1 ? 1 : -1;
                    w8 |= (t & 0xff) << (8 * j);
                    ++sbit;
                }
            }
            acc = __dp4a(w8, __ldg(arow + ((e + half * 4) >> 2)), acc);
        }
        // Advance the running base by the whole block's nonzeros (lane 31's
        // inclusive prefix).
        sign_base += __shfl_sync(0xffffffffu, pre, WARP_SIZE - 1);
    }
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        out[(long long)mi * n + ni] = (float)acc * scales[ni] * act_scale[mi];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TB1 LUT decode (task #58 redesign attempt): identical format, identical
// accumulator math and epilogue — only the UNPACK changes. The round-16
// refutation blamed the warp prefix scan, but the compiled loop's histogram
// says otherwise: the scan is 10 of ~137 instructions/block; ~60% is the
// per-element bit-test/select chains (setp+selp+shift+or per trit, with a
// data-dependent serial `sbit` carry across the 8 elements). This variant
// deletes those chains with two shared LUTs:
//
//   s_pres[16]  : nibble  -> 4 bytes of 0x00/0x01 (presence expansion)
//   s_dep[256]  : (nibble p, 4-bit sign window w) -> the COMPACTED low
//                 popc(p) bits of w DEPOSITED as 0x00/0x01 bytes at p's set
//                 positions. Window bits >= popc(p) are never consumed at
//                 build time, so runtime passes a raw 4-bit window with
//                 garbage high bits and needs NO mask.
//
// Ternary bytes then come from pure byte-SIMD:
//   w8 = __vsub4(__vadd4(dep, dep), pres)   // {0,1}*2 - {0,1} = {-1, 0, +1}
//
// The scan, the sign-window funnel, and the running sign_base stay verbatim
// (they were never the cost). Same launch geometry, same arena contract
// (>= 2, use 4, trailing slack bytes), k % 256 == 0.
extern "C" __global__ void tb1_mpgemm_lut_i8_scaled(
    const signed char* __restrict__ qact,   // [M, K] packed int8, k % 4 == 0
    const unsigned char* __restrict__ weights,
    const unsigned int* __restrict__ row_offsets,  // [N] byte offset per row
    const float* __restrict__ scales,
    const float* __restrict__ act_scale,
    float* __restrict__ out,
    const int m,
    const int n,
    const int k) {
    __shared__ unsigned int s_dep[256];
    __shared__ unsigned int s_pres[16];
    // Build the LUTs cooperatively (blockDim.x == 256 in the shared launch
    // config; guard anyway so a smaller block still fills every entry).
    for (int t = threadIdx.x; t < 256; t += blockDim.x) {
        const int p = t >> 4;
        const int w = t & 15;
        unsigned int dep = 0;
        int sbit = 0;
        unsigned int pres = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (p & (1 << j)) {
                pres |= 1u << (8 * j);
                if ((w >> sbit) & 1) {
                    dep |= 1u << (8 * j);
                }
                ++sbit;
            }
        }
        s_dep[t] = dep;
        if (w == 0) {
            s_pres[p] = pres;
        }
    }
    __syncthreads();

    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int mi = blockIdx.y;
    if (mi >= m) {
        return;
    }
    const int* arow = (const int*)(qact + (long long)mi * k);
    const int ni = blockIdx.x * warps_per_block + warp_id;
    if (ni >= n) {
        return;
    }
    const int nb = k >> 8;
    const unsigned char* presence = weights + row_offsets[ni];
    const unsigned char* signs = presence + nb * 32;
    int acc = 0;
    int sign_base = 0;  // running bit offset into this row's sign stream
    for (int b = 0; b < nb; ++b) {
        const unsigned int pb = presence[b * 32 + lane];
        const int cnt = __popc(pb);
        // Inclusive prefix sum of per-lane popcounts across the warp.
        int pre = cnt;
#pragma unroll
        for (int off = 1; off < WARP_SIZE; off <<= 1) {
            const int up = __shfl_up_sync(0xffffffffu, pre, off);
            if (lane >= off) {
                pre += up;
            }
        }
        const int my_start = sign_base + pre - cnt;  // exclusive prefix
        // ≤8 sign bits span ≤2 bytes of the stream.
        const int byte0 = my_start >> 3;
        const unsigned int sw =
            ((unsigned int)signs[byte0] | ((unsigned int)signs[byte0 + 1] << 8)) >>
            (my_start & 7);
        const int e = (b << 8) + lane * 8;
        // Low nibble: window = sw's low bits; high nibble's window starts
        // after the low nibble's popc(pl) consumed sign bits.
        const unsigned int pl = pb & 15u;
        const unsigned int ph = pb >> 4;
        const int cl = __popc(pl);
        {
            const unsigned int dep = s_dep[(pl << 4) | (sw & 15u)];
            const int w8 = (int)__vsub4(__vadd4(dep, dep), s_pres[pl]);
            acc = __dp4a(w8, __ldg(arow + (e >> 2)), acc);
        }
        {
            const unsigned int dep = s_dep[(ph << 4) | ((sw >> cl) & 15u)];
            const int w8 = (int)__vsub4(__vadd4(dep, dep), s_pres[ph]);
            acc = __dp4a(w8, __ldg(arow + ((e + 4) >> 2)), acc);
        }
        // Advance the running base by the whole block's nonzeros (lane 31's
        // inclusive prefix).
        sign_base += __shfl_sync(0xffffffffu, pre, WARP_SIZE - 1);
    }
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        out[(long long)mi * n + ni] = (float)acc * scales[ni] * act_scale[mi];
    }
}
