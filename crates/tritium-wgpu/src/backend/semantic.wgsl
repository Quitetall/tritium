// Plain f32 GEMM for device-owned semantic tensors, not ternary clipping/STE.
struct Dims { m: u32, n: u32, k: u32, lane_stride: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> act: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.y * dims.lane_stride + gid.x;
    if (index >= dims.m * dims.n) { return; }
    let row = index / dims.n;
    let column = index % dims.n;
    var sum: f32 = 0.0;
    for (var k = 0u; k < dims.k; k = k + 1u) {
        sum = sum + act[row * dims.k + k] * weights[column * dims.k + k];
    }
    output[index] = sum;
}
