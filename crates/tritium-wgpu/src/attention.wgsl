struct AttentionParams {
    seq: u32,
    n_head: u32,
    n_kv_head: u32,
    head_dim: u32,
    causal: u32,
    execution: u32,
    padding_0: u32,
    padding_1: u32,
}

@group(0) @binding(0) var<uniform> params: AttentionParams;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read> grad_output: array<f32>;
@group(0) @binding(5) var<storage, read_write> output_0: array<f32>;
@group(0) @binding(6) var<storage, read_write> output_1: array<f32>;
@group(0) @binding(7) var<storage, read_write> output_2: array<f32>;
@group(0) @binding(8) var<storage, read_write> probabilities: array<f32>;
@group(0) @binding(9) var<storage, read_write> grad_probabilities: array<f32>;

fn vector_index(token: u32, head: u32, head_count: u32, lane: u32) -> u32 {
    return (token * head_count + head) * params.head_dim + lane;
}

fn compute_probabilities(head: u32, kv_head: u32, scale: f32) {
    for (var query = 0u; query < params.seq; query++) {
        let row = query * params.seq;
        for (var key = 0u; key < params.seq; key++) {
            var score = 0.0;
            for (var lane = 0u; lane < params.head_dim; lane++) {
                score += q[vector_index(query, head, params.n_head, lane)]
                    * k[vector_index(key, kv_head, params.n_kv_head, lane)];
            }
            probabilities[row + key] = select(score * scale, bitcast<f32>(0xff800000u), params.causal != 0u && key > query);
        }
        var maximum = bitcast<f32>(0xff800000u);
        for (var key = 0u; key < params.seq; key++) {
            maximum = max(maximum, probabilities[row + key]);
        }
        var sum = 0.0;
        for (var key = 0u; key < params.seq; key++) {
            let exponential = select(exp(probabilities[row + key] - maximum), 0.0, params.causal != 0u && key > query);
            probabilities[row + key] = exponential;
            sum += exponential;
        }
        for (var key = 0u; key < params.seq; key++) {
            probabilities[row + key] /= sum;
        }
    }
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u || gid.y != 0u || gid.z != 0u) {
        return;
    }
    let group_size = params.n_head / params.n_kv_head;
    let scale = inverseSqrt(f32(params.head_dim));
    if (params.execution == 0u) {
        for (var head = 0u; head < params.n_head; head++) {
            let kv_head = head / group_size;
            compute_probabilities(head, kv_head, scale);
            for (var query = 0u; query < params.seq; query++) {
                for (var lane = 0u; lane < params.head_dim; lane++) {
                    var accumulator = 0.0;
                    for (var key = 0u; key < params.seq; key++) {
                        accumulator += probabilities[query * params.seq + key]
                            * v[vector_index(key, kv_head, params.n_kv_head, lane)];
                    }
                    output_0[vector_index(query, head, params.n_head, lane)] = accumulator;
                }
            }
        }
        return;
    }

    var head_cursor = params.n_head;
    loop {
        if (head_cursor == 0u) {
            break;
        }
        head_cursor -= 1u;
        let head = head_cursor;
        let kv_head = head / group_size;
        compute_probabilities(head, kv_head, scale);
        for (var index = 0u; index < params.seq * params.seq; index++) {
            grad_probabilities[index] = 0.0;
        }
        for (var query = 0u; query < params.seq; query++) {
            for (var lane = 0u; lane < params.head_dim; lane++) {
                let gradient = grad_output[vector_index(query, head, params.n_head, lane)];
                for (var key = 0u; key < params.seq; key++) {
                    let probability_index = query * params.seq + key;
                    let value_index = vector_index(key, kv_head, params.n_kv_head, lane);
                    grad_probabilities[probability_index] += gradient * v[value_index];
                    output_2[value_index] += gradient * probabilities[probability_index];
                }
            }
        }
        for (var query = 0u; query < params.seq; query++) {
            let row = query * params.seq;
            var contraction = 0.0;
            for (var key = 0u; key < params.seq; key++) {
                contraction += probabilities[row + key] * grad_probabilities[row + key];
            }
            for (var key = 0u; key < params.seq; key++) {
                let index = row + key;
                grad_probabilities[index] = select(
                    probabilities[index] * (grad_probabilities[index] - contraction) * scale,
                    0.0,
                    params.causal != 0u && key > query,
                );
            }
        }
        for (var query = 0u; query < params.seq; query++) {
            for (var key = 0u; key < params.seq; key++) {
                let gradient = grad_probabilities[query * params.seq + key];
                for (var lane = 0u; lane < params.head_dim; lane++) {
                    let query_index = vector_index(query, head, params.n_head, lane);
                    let key_index = vector_index(key, kv_head, params.n_kv_head, lane);
                    output_0[query_index] += gradient * k[key_index];
                    output_1[key_index] += gradient * q[query_index];
                }
            }
        }
    }
}
