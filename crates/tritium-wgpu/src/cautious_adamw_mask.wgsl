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
@group(0) @binding(2) var<storage, read> gradient: array<f32>;
@group(0) @binding(6) var<storage, read_write> updated_moment1: array<f32>;
@group(0) @binding(7) var<storage, read_write> updated_moment2: array<f32>;
@group(0) @binding(8) var<storage, read_write> scratch1: array<f32>;
@group(0) @binding(10) var<storage, read_write> aligned: atomic<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < params.len) {
        let update = (updated_moment1[index] / params.correction1)
            / (sqrt(updated_moment2[index] / params.correction2) + params.epsilon);
        if (update * gradient[index] > 0.0) {
            scratch1[index] = update;
            atomicAdd(&aligned, 1u);
        } else {
            scratch1[index] = 0.0;
        }
    }
}
