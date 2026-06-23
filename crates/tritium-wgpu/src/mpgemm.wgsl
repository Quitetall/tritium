// One invocation computes one out[m, n] = scales[n] * sum_k act[m, k] (+/-) w[n, k].
// w is ternary in {-1, 0, +1}, widened host-side to i32. f32 accumulation, in the
// SAME add/subtract/skip form the reference uses (tritium_core::reference_mpgemm):
//   trit ==  1 -> acc += act[k]
//   trit == -1 -> acc -= act[k]
//   trit ==  0 -> skip
// This is deliberately NOT `acc += act * f32(trit)`: branching on the sign keeps
// the per-(m,n) accumulation order AND form identical to the reference, which is
// what holds the 1e-4 relative bar on the high-cancellation boundary vectors.
// Do not "optimize" this into a multiply or reorder the loop (no subgroup/tree/
// atomic reductions — they reorder the sum and blow the tolerance budget).

struct Dims {
    m: u32,
    n: u32,
    k: u32,
    lane_stride: u32,   // = workgroups_x * 64: x-extent of the 2-D dispatch grid,
                        // used to flatten (gid.x, gid.y) back to a linear output
                        // index so M*N can exceed the 65535-workgroups-per-dim cap.
};

@group(0) @binding(0) var<uniform>             dims    : Dims;
@group(0) @binding(1) var<storage, read>       act     : array<f32>;  // [M*K] row-major
@group(0) @binding(2) var<storage, read>       weights : array<i32>;  // [N*K] row-major {-1,0,1}
@group(0) @binding(3) var<storage, read>       scales  : array<f32>;  // [N]
@group(0) @binding(4) var<storage, read_write> out_buf : array<f32>;  // [M*N] row-major

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Flatten the 2-D dispatch grid: each y-row holds `lane_stride` x-invocations.
    let idx = gid.y * dims.lane_stride + gid.x;   // linear output index
    let total = dims.m * dims.n;
    if (idx >= total) {              // tail / 2-D padding invocations — guard FIRST
        return;
    }

    let m = idx / dims.n;            // row of out / row of act
    let n = idx % dims.n;            // col of out / row of weights

    let act_base = m * dims.k;
    let w_base = n * dims.k;

    var acc: f32 = 0.0;
    // Sequential f32 k-loop, add/sub/skip — mirrors reference_mpgemm exactly.
    for (var kk: u32 = 0u; kk < dims.k; kk = kk + 1u) {
        let t = weights[w_base + kk];          // i32 in {-1,0,1}
        let a = act[act_base + kk];
        if (t == 1) {
            acc = acc + a;
        } else if (t == -1) {
            acc = acc - a;
        }
        // t == 0 -> skip (no NaN injected from 0*x; matches the reference skip)
    }

    out_buf[idx] = scales[n] * acc;            // fold per-channel scale ONCE at end
}
