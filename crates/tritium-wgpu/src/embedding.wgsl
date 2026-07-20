struct Params {
    vocab: u32,
    width: u32,
    sequence: u32,
    operation: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> tokens: array<u32>;
@group(0) @binding(3) var<storage, read> gradient: array<f32>;
@group(0) @binding(4) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (params.operation == 0u) {
        if (index >= params.sequence * params.width) { return; }
        let sequence_index = index / params.width;
        let column = index % params.width;
        result[index] = weight[tokens[sequence_index] * params.width + column];
    } else {
        if (index >= params.vocab * params.width) { return; }
        let token = index / params.width;
        let column = index % params.width;
        var accumulator = 0.0;
        for (var sequence_index = 0u; sequence_index < params.sequence;
             sequence_index = sequence_index + 1u) {
            if (tokens[sequence_index] == token) {
                accumulator = accumulator
                    + gradient[sequence_index * params.width + column];
            }
        }
        result[index] = accumulator;
    }
}
