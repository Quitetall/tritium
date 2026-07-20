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
@group(0) @binding(5) var<storage, read_write> updated_parameter: array<f32>;
@group(0) @binding(6) var<storage, read_write> updated_moment1: array<f32>;
@group(0) @binding(7) var<storage, read_write> updated_moment2: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < params.len) {
        let m = updated_moment1[index];
        let v = updated_moment2[index];
        let adaptive = params.learning_rate
            * (m / params.correction1 / (sqrt(v / params.correction2) + params.epsilon));
        updated_parameter[index] = updated_parameter[index] - adaptive;
    }
}
