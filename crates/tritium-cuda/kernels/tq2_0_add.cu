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

// f32-accumulate variant of `tq2_0_add_mpgemm_tiled` for the v0.3.2 CUDA-graph decode
// (perf) path. The double accumulator above is correct but SLOW on consumer GPUs: the
// RTX 4090 runs f64 at 1/64 the f32 rate, and the decode forward issues ~210 of these
// per token — measured as the decode bottleneck. Accumulating + tree-reducing in f32
// reorders the K sum past the 1e-4 bit-match tolerance on cancellation-heavy rows, so the
// graph path is gated by **perplexity ≤ 1% + lockstep argmax**, not greedy bit-match (the
// eager `step` keeps the double kernel and its 256/256 parity). Identical layout/decode;
// only the accumulator type differs.
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
    for (int ki = lane; ki < k; ki += WARP_SIZE) {
        const int block = ki / QK_K;
        const int e = ki - block * QK_K;
        const int c = e >> 7;
        const int mm = e & 31;
        const int l = (e & 127) >> 5;
        const int byte_off = block * TQ2_0_BLOCK_BYTES + c * 32 + mm;
        const unsigned int code = ((unsigned int)wrow[byte_off] >> (2 * l)) & 3u;
        const float a = s_act[ki];
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
        out[(long long)mi * n + ni] = acc * scales[ni];
    }
}
