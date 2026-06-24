// mpgemm.metal — add/sub/skip ternary mixed-precision GEMM in Metal Shading
// Language. A direct port of the tritium-wgpu WGSL kernel (src/mpgemm.wgsl) to
// MSL, compiled at runtime from this source string via
// `device.newLibraryWithSource:`.
//
// One thread computes one
//   out[m, n] = scales[n] * sum_k act[m, k] (+/-) w[n, k]
// for ternary weights w in {-1, 0, +1}, host-unpacked and widened to i32 (one
// i32 per trit, exactly like the wgpu path's std430 `array<i32>`). f32
// accumulation, in the SAME add/subtract/skip form the reference uses
// (tritium_core::reference_mpgemm):
//   trit ==  1 -> acc += act[k]
//   trit == -1 -> acc -= act[k]
//   trit ==  0 -> skip
// This is deliberately NOT `acc += act * float(trit)`: branching on the sign
// keeps the per-(m,n) accumulation ORDER and FORM identical to the reference,
// which is what holds the 1e-4 relative bar on the high-cancellation boundary
// vectors. Do not "optimize" this into a multiply or a tree/simdgroup reduction
// — those reorder the sum and blow the tolerance budget (same warning as the
// WGSL kernel).
//
// Dims is passed by value as a small constant buffer (set_bytes), matching the
// repr(C) `Dims` struct on the Rust side. `lane_stride` flattens the 2-D
// dispatch grid back to a linear output index so M*N can exceed any single
// dimension's threadgroup ceiling — identical scheme to the wgpu kernel.

#include <metal_stdlib>
using namespace metal;

struct Dims {
    uint m;
    uint n;
    uint k;
    uint lane_stride;   // = threadgroups_x * TG_SIZE: x-extent of the 2-D grid
};

kernel void mpgemm(
    device const float*  act      [[buffer(0)]],   // [M*K] row-major
    device const int*    weights  [[buffer(1)]],   // [N*K] row-major {-1,0,1}
    device const float*  scales   [[buffer(2)]],   // [N]
    device       float*  out_buf  [[buffer(3)]],   // [M*N] row-major
    constant     Dims&   dims     [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    // Flatten the 2-D dispatch grid: each y-row holds `lane_stride` x-threads.
    uint idx = gid.y * dims.lane_stride + gid.x;   // linear output index
    uint total = dims.m * dims.n;
    if (idx >= total) {              // tail / 2-D padding threads — guard FIRST
        return;
    }

    uint m = idx / dims.n;           // row of out / row of act
    uint n = idx % dims.n;           // col of out / row of weights

    uint act_base = m * dims.k;
    uint w_base   = n * dims.k;

    float acc = 0.0f;
    // Sequential f32 k-loop, add/sub/skip — mirrors reference_mpgemm exactly.
    for (uint kk = 0u; kk < dims.k; kk = kk + 1u) {
        int   t = weights[w_base + kk];        // i32 in {-1,0,1}
        float a = act[act_base + kk];
        if (t == 1) {
            acc = acc + a;
        } else if (t == -1) {
            acc = acc - a;
        }
        // t == 0 -> skip (matches the reference skip; no multiply, no NaN)
    }

    out_buf[idx] = scales[n] * acc;            // fold per-channel scale ONCE
}
