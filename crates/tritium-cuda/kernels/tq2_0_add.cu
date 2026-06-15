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
