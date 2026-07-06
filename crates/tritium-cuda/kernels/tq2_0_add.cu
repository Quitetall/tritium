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
// every call site feeds `d_qact`, the output of `act_quant_tiled_f32` /
// `rmsnorm_quant_f32` / `act_quant_batch_f32` — integer-valued f32 in
// [-128, 127] (the W1.58A8 protocol; `act_scale` carries the dequant factor).
// That precondition lets these kernels run the contraction on the int8 dot-
// product units via `__dp4a` (sm_61+, under the compute_75 build floor):
//
//   * The staging pass converts the integer-valued floats to packed int8x4
//     words in shared memory (`__float2int_rn` is exact on integers).
//   * Each lane reads FOUR consecutive qs bytes as two u16 loads (the 66-byte
//     block stride only guarantees 2-byte alignment) — one byte-once load of
//     16 trits — and extracts one 2-bit slot across all four bytes at a time:
//     `(w >> 2*slot) & 0x03030303` then a per-byte `- 1` (`__vsub4`) yields
//     four int8 trits ∈ {-1,0,+1} ready for one `__dp4a` against the packed
//     activation word. Lane l owns slot = l/8 and byte quad j = l%8, covering
//     a whole 256-trit block per iteration with two independent accumulators.
//   * The int32 accumulate is EXACT (|Σ| ≤ 127·K < 2³¹) and order-independent,
//     so the result is bit-identical to the previous f32 kernel on this path:
//     f32 sums of integer-valued products also stayed exact (< 2²⁴), and the
//     epilogue multiplies are unchanged. Greedy/perplexity gates are unaffected.
//
// Measured on the RTX 4090 vs the previous f32 tiled kernel (M=1): ~2.7×–3.9×
// per shape, reaching ~86% of the weight-bandwidth speed-of-light at
// N=K=6912. If a future call site ever passes NON-integer activations here,
// route it to `tq2_0_add_mpgemm_tiled_f32` instead — `__float2int_rn` rounds.

// Fused-scaled variant (v0.6.0 opt #15 epilogue). DP4A contraction, see above.
extern "C" __global__ void tq2_0_add_mpgemm_tiled_f32_scaled(
    const float* __restrict__ act,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ scales,
    const float* __restrict__ act_scale,
    float* __restrict__ out,
    const int m,
    const int n,
    const int k,
    const int row_bytes) {
    extern __shared__ int s_qact[];    // ceil(k/4) packed int8x4 words
    signed char* s_bytes = (signed char*)s_qact;

    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int mi = blockIdx.y;
    if (mi >= m) {
        return;
    }
    // Stage the row as int8 (exact: values are A8-quantized integers). Byte
    // writes to shared are race-free; the pad up to a word boundary is zeroed.
    const float* arow = act + (long long)mi * k;
    const int kpad = (k + 3) & ~3;
    for (int i = threadIdx.x; i < kpad; i += blockDim.x) {
        s_bytes[i] = (i < k) ? (signed char)__float2int_rn(arow[i]) : 0;
    }
    __syncthreads();

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
                      s_qact[kidx >> 2], acc0);
        acc1 = __dp4a((int)__vsub4((w1 >> (2 * slot)) & 0x03030303u, 0x01010101u),
                      s_qact[(kidx + 128) >> 2], acc1);
    }
    int acc = acc0 + acc1;
    // Tail (k % 256 != 0): scalar per-trit decode in exact int math.
    for (int ki = (k & ~255) + lane; ki < k; ki += WARP_SIZE) {
        const int block = ki / QK_K;
        const int e = ki - block * QK_K;
        const int c = e >> 7;
        const int mm = e & 31;
        const int l = (e & 127) >> 5;
        const unsigned int code =
            ((unsigned int)wrow[block * TQ2_0_BLOCK_BYTES + c * 32 + mm] >> (2 * l)) & 3u;
        acc += ((int)code - 1) * (int)s_bytes[ki];
    }
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        out[(long long)mi * n + ni] = (float)acc * scales[ni] * act_scale[mi];
    }
}

// Sparse-aware variant of `tq2_0_add_mpgemm_tiled_f32_scaled`. Same DP4A
// contraction + integer-valued-activation precondition; same bitmap layout as
// `tq2_0_add_mpgemm_tiled_f32_sparse` (one bit per 256-trit block, set = all-
// zero, skipped). NULL bitmap → identical to the non-sparse variant.
extern "C" __global__ void tq2_0_add_mpgemm_tiled_f32_scaled_sparse(
    const float* __restrict__ act,
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
    extern __shared__ int s_qact[];
    signed char* s_bytes = (signed char*)s_qact;

    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int mi = blockIdx.y;
    if (mi >= m) {
        return;
    }
    const float* arow = act + (long long)mi * k;
    const int kpad = (k + 3) & ~3;
    for (int i = threadIdx.x; i < kpad; i += blockDim.x) {
        s_bytes[i] = (i < k) ? (signed char)__float2int_rn(arow[i]) : 0;
    }
    __syncthreads();

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
                      s_qact[kidx >> 2], acc0);
        acc1 = __dp4a((int)__vsub4((w1 >> (2 * slot)) & 0x03030303u, 0x01010101u),
                      s_qact[(kidx + 128) >> 2], acc1);
    }
    int acc = acc0 + acc1;
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
        acc += ((int)code - 1) * (int)s_bytes[ki];
    }
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        out[(long long)mi * n + ni] = (float)acc * scales[ni] * act_scale[mi];
    }
}

// Fused-scaled + residual-add variant (v0.7.0 opt Phase 2 epilogue, DP4A
// contraction — same integer-valued-activation precondition as above).
// Epilogue: out[mi,ni] = residual[mi,ni] + acc * scales[ni] * act_scale[mi]
// Eliminates the separate residual_add_f32 kernel launch + its memory pass.
extern "C" __global__ void tq2_0_add_mpgemm_tiled_f32_scaled_residual(
    const float* __restrict__ act,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ scales,
    const float* __restrict__ act_scale,
    const float* __restrict__ residual,
    float* __restrict__ out,
    const int m,
    const int n,
    const int k,
    const int row_bytes) {
    extern __shared__ int s_qact[];
    signed char* s_bytes = (signed char*)s_qact;

    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int mi = blockIdx.y;
    if (mi >= m) {
        return;
    }
    const float* arow = act + (long long)mi * k;
    const int kpad = (k + 3) & ~3;
    for (int i = threadIdx.x; i < kpad; i += blockDim.x) {
        s_bytes[i] = (i < k) ? (signed char)__float2int_rn(arow[i]) : 0;
    }
    __syncthreads();

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
                      s_qact[kidx >> 2], acc0);
        acc1 = __dp4a((int)__vsub4((w1 >> (2 * slot)) & 0x03030303u, 0x01010101u),
                      s_qact[(kidx + 128) >> 2], acc1);
    }
    int acc = acc0 + acc1;
    for (int ki = (k & ~255) + lane; ki < k; ki += WARP_SIZE) {
        const int block = ki / QK_K;
        const int e = ki - block * QK_K;
        const int c = e >> 7;
        const int mm = e & 31;
        const int l = (e & 127) >> 5;
        const unsigned int code =
            ((unsigned int)wrow[block * TQ2_0_BLOCK_BYTES + c * 32 + mm] >> (2 * l)) & 3u;
        acc += ((int)code - 1) * (int)s_bytes[ki];
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
