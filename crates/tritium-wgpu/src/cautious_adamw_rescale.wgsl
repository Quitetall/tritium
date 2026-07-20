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
@group(0) @binding(8) var<storage, read_write> scratch1: array<f32>;
@group(0) @binding(10) var<storage, read_write> aligned: atomic<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < params.len) {
        let rescale = f32(params.len) / (f32(atomicLoad(&aligned)) + 1.0);
        scratch1[index] = scratch1[index] * rescale;
    }
}
