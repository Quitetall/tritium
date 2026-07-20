struct Params {
    len: u32,
    padding_0: u32,
    padding_1: u32,
    padding_2: u32,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    correction1: f32,
    correction2: f32,
    shrink: f32,
    padding_3: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> parameter: array<f32>;
@group(0) @binding(2) var<storage, read> gradient: array<f32>;
@group(0) @binding(3) var<storage, read> moment1: array<f32>;
@group(0) @binding(4) var<storage, read> moment2: array<f32>;
@group(0) @binding(5) var<storage, read_write> updated_parameter: array<f32>;
@group(0) @binding(6) var<storage, read_write> updated_moment1: array<f32>;
@group(0) @binding(7) var<storage, read_write> updated_moment2: array<f32>;
@group(0) @binding(8) var<storage, read_write> scratch1: array<f32>;
@group(0) @binding(9) var<storage, read_write> scratch2: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < params.len) {
        let g = gradient[index];
        updated_moment1[index] = params.beta1 * moment1[index];
        scratch1[index] = (1.0 - params.beta1) * g;
        updated_moment2[index] = params.beta2 * moment2[index];
        scratch2[index] = (1.0 - params.beta2) * g;
        updated_parameter[index] = parameter[index] * params.shrink;
    }
}
