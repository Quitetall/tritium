struct Params {
    len: u32,
    operation: u32,
    scalar: f32,
    auxiliary: u32,
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
        case 5u: {
            let positive = max(left[index], 0.0);
            result[index] = positive * positive;
        }
        case 6u: { result[index] = right[index] * 2.0 * max(left[index], 0.0); }
        case 7u: {
            let sigmoid = 1.0 / (1.0 + exp(-left[index]));
            result[index] = left[index] * sigmoid;
        }
        case 8u: {
            let sigmoid = 1.0 / (1.0 + exp(-left[index]));
            result[index] = right[index]
                * (sigmoid + left[index] * sigmoid * (1.0 - sigmoid));
        }
        case 9u: {
            let row = index / params.auxiliary;
            let column = index % params.auxiliary;
            result[index] = select(-1e30, left[index], column <= row);
        }
        case 10u: {
            let row = index / params.auxiliary;
            let column = index % params.auxiliary;
            result[index] = select(0.0, left[index], column <= row);
        }
        default: { result[index] = 0.0; }
    }
}
