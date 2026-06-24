// mpgemm.metal — add/sub/skip ternary mixed-precision GEMM in Metal Shading
// Language. Two kernels share the exact add/subtract/skip f32 accumulation form of
// `tritium_core::reference_mpgemm` (branch on sign; NO multiply, NO tree/simdgroup
// reduction — any reorder blows the 1e-4 cancellation bar):
//
//   * `mpgemm`        — weights pre-widened HOST-side to one i32 per trit. Used for
//                       TQ1_0 (the small/rare format). 32 bit/trit on device.
//   * `mpgemm_tq2_0`  — weights stay PACKED on device (TQ2_0, 66-byte blocks); the
//                       2-bit code is decoded per-k INSIDE the kernel, a direct port
//                       of the verified tritium-cuda / tritium-rocm add-only kernel.
//                       Device memory stays at the packed ~2.06 bit/trit (16× less
//                       than widening), matching the cuda/rocm backends so real
//                       multi-billion-parameter models fit Apple unified memory.
//
// One thread computes one
//   out[m, n] = scales[n] * sum_k act[m, k] (+/-) w[n, k]
// `Dims` is passed by value (set_bytes) and matches the repr(C) `Dims` on the Rust
// side; `lane_stride` flattens the 2-D dispatch grid back to a linear output index
// so M*N can exceed any single dimension's threadgroup ceiling. `row_bytes` is the
// packed stride per weight row (read only by `mpgemm_tq2_0`; 0 for `mpgemm`).

#include <metal_stdlib>
using namespace metal;

struct Dims {
    uint m;
    uint n;
    uint k;
    uint lane_stride;   // = threadgroups_x * TG_SIZE: x-extent of the 2-D grid
    uint row_bytes;     // packed bytes per weight row (mpgemm_tq2_0 only; 0 otherwise)
};

// TQ2_0 packing geometry — MUST match tritium-format and the cuda/hip kernels:
//   QK_K = 256 trits per block; each block is 66 bytes (64 `qs` bytes then a 2-byte
//   f16 scale that the packer fixes to 1.0 and we ignore — channel scaling is the
//   single end multiply). code = trit + 1 in {0,1,2}: 0 -> -1, 1 -> 0, 2 -> +1.
constant uint QK_K = 256u;
constant uint TQ2_0_BLOCK_BYTES = 66u;

// Widened path: one i32 per trit in {-1,0,1} (host-unpacked). Used for TQ1_0.
kernel void mpgemm(
    device const float*  act      [[buffer(0)]],   // [M*K] row-major
    device const int*    weights  [[buffer(1)]],   // [N*K] row-major {-1,0,1}
    device const float*  scales   [[buffer(2)]],   // [N]
    device       float*  out_buf  [[buffer(3)]],   // [M*N] row-major
    constant     Dims&   dims     [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint idx = gid.y * dims.lane_stride + gid.x;   // linear output index
    uint total = dims.m * dims.n;
    if (idx >= total) {              // tail / 2-D padding threads — guard FIRST
        return;
    }
    uint m = idx / dims.n;
    uint n = idx % dims.n;
    // 64-bit row bases: m*k / n*k can exceed u32 for large models. idx/total stay
    // u32 (bounded by the host's m*n <= u32::MAX guard); only the per-row element
    // bases need widening — mirrors the cuda/hip reference's `(long long)` casts.
    ulong act_base = (ulong)m * dims.k;
    ulong w_base   = (ulong)n * dims.k;

    float acc = 0.0f;
    for (uint kk = 0u; kk < dims.k; kk = kk + 1u) {
        int   t = weights[w_base + kk];        // i32 in {-1,0,1}
        float a = act[act_base + kk];
        if (t == 1) {
            acc = acc + a;
        } else if (t == -1) {
            acc = acc - a;
        }
        // t == 0 -> skip
    }
    out_buf[idx] = scales[n] * acc;            // fold per-channel scale ONCE
}

// Packed path: decode TQ2_0 2-bit codes in-kernel (port of the verified
// tritium-cuda/tritium-rocm `tq2_0_add` kernel). Weights stay packed on device.
kernel void mpgemm_tq2_0(
    device const float*  act      [[buffer(0)]],   // [M*K] row-major
    device const uchar*  weights  [[buffer(1)]],   // [N * row_bytes] packed TQ2_0
    device const float*  scales   [[buffer(2)]],   // [N]
    device       float*  out_buf  [[buffer(3)]],   // [M*N] row-major
    constant     Dims&   dims     [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint idx = gid.y * dims.lane_stride + gid.x;
    uint total = dims.m * dims.n;
    if (idx >= total) {
        return;
    }
    uint m = idx / dims.n;
    uint n = idx % dims.n;
    // 64-bit bases: m*k and n*row_bytes can exceed u32 for large models. idx/total
    // stay u32 (bounded by the host's m*n <= u32::MAX guard); only the per-row
    // bases need widening — mirrors the cuda/hip reference's `(long long)` casts.
    ulong act_base = (ulong)m * dims.k;
    device const uchar* wrow = weights + (ulong)n * dims.row_bytes;

    float acc = 0.0f;
    for (uint kk = 0u; kk < dims.k; kk = kk + 1u) {
        // Locate the packed 2-bit code for trit kk of this row (same math as the
        // cuda/hip kernels and tritium-format's TQ2_0 layout).
        uint block    = kk / QK_K;             // which 256-trit block
        uint e        = kk - block * QK_K;     // element index within the block
        uint c        = e >> 7;                // e / 128   -> chunk 0/1
        uint mm       = e & 31u;               // e % 32    -> byte within half
        uint l        = (e & 127u) >> 5;       // (e % 128) / 32 -> 2-bit slot
        uint byte_off = block * TQ2_0_BLOCK_BYTES + c * 32u + mm;
        uint code     = ((uint)wrow[byte_off] >> (2u * l)) & 3u;
        float a = act[act_base + kk];
        // code 0 -> -1 (sub), 1 -> 0 (skip), 2 -> +1 (add). Accumulate by sign.
        if (code == 2u) {
            acc = acc + a;
        } else if (code == 0u) {
            acc = acc - a;
        }
    }
    out_buf[idx] = scales[n] * acc;
}
