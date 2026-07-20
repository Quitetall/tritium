struct Params {
    len: u32,
    operation: u32,
    scalar: f32,
    padding: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> left: array<f32>;
@group(0) @binding(2) var<storage, read> right: array<f32>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= params.len) {
        return;
    }
    switch params.operation {
        case 0u: { result[index] = left[index]; }
        case 1u: { result[index] = 0.0; }
        case 2u: { result[index] = left[index] * params.scalar; }
        case 3u: { result[index] = left[index] + right[index]; }
        case 4u: { result[index] = left[index] * right[index]; }
        default: { result[index] = 0.0; }
    }
}
