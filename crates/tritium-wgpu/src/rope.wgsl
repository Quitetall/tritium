struct Params {
    n_token: u32,
    n_head: u32,
    head_dim: u32,
    inverse: u32,
    theta: f32,
    padding_0: f32,
    padding_1: f32,
    padding_2: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> positions: array<u32>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half = params.head_dim / 2u;
    let total = params.n_token * params.n_head * half;
    let pair = gid.x;
    if (pair >= total) {
        return;
    }
    let j = pair % half;
    let head = (pair / half) % params.n_head;
    let token = pair / (params.n_head * half);
    let base = (token * params.n_head + head) * params.head_dim;
    let exponent = -2.0 * f32(j) / f32(params.head_dim);
    let angle = f32(positions[token]) * pow(params.theta, exponent);
    let cosine = cos(angle);
    let sine_sign = select(1.0, -1.0, params.inverse != 0u);
    let sine = sine_sign * sin(angle);
    let a = x[base + j];
    let b = x[base + j + half];
    result[base + j] = a * cosine - b * sine;
    result[base + j + half] = b * cosine + a * sine;
}
