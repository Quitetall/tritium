struct Params {
    rows: u32,
    cols: u32,
    k: u32,
    execution: u32,
    padding_0: f32,
    padding_1: f32,
    padding_2: f32,
    padding_3: f32,
};

// Chrome Tint rejects a bitcast infinity literal. This finite sentinel still
// underflows through exp() and preserves masked-softmax behavior.
const NEGATIVE_LARGE: f32 = -3.402823466e+38;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> probabilities: array<f32>;
@group(0) @binding(4) var<storage, read> grad_output: array<f32>;
@group(0) @binding(5) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(1, 1, 1)
fn main() {
    if (params.execution == 0u) {
        var total = 0.0;
        for (var row = 0u; row < params.rows; row += 1u) {
            let dense_base = row * params.cols;
            let sparse_base = row * params.k;
            var maximum = NEGATIVE_LARGE;
            for (var column = 0u; column < params.cols; column += 1u) {
                maximum = max(maximum, logits[dense_base + column]);
            }
            var sum = 0.0;
            for (var column = 0u; column < params.cols; column += 1u) {
                sum += exp(logits[dense_base + column] - maximum);
            }
            for (var sparse_column = 0u; sparse_column < params.k; sparse_column += 1u) {
                let column = indices[sparse_base + sparse_column];
                let probability = exp(logits[dense_base + column] - maximum) / sum;
                total -= probabilities[sparse_base + sparse_column]
                    * log(max(probability, 1.17549435e-38));
            }
        }
        result[0] = total / f32(params.rows);
    } else {
        for (var row = 0u; row < params.rows; row += 1u) {
            let dense_base = row * params.cols;
            let sparse_base = row * params.k;
            var maximum = NEGATIVE_LARGE;
            for (var column = 0u; column < params.cols; column += 1u) {
                maximum = max(maximum, logits[dense_base + column]);
            }
            var sum = 0.0;
            for (var column = 0u; column < params.cols; column += 1u) {
                sum += exp(logits[dense_base + column] - maximum);
            }
            var probability_sum = 0.0;
            for (var sparse_column = 0u; sparse_column < params.k; sparse_column += 1u) {
                probability_sum += probabilities[sparse_base + sparse_column];
            }
            let scale = grad_output[0] / f32(params.rows);
            for (var column = 0u; column < params.cols; column += 1u) {
                let probability = exp(logits[dense_base + column] - maximum) / sum;
                result[dense_base + column] = scale * probability * probability_sum;
            }
            for (var sparse_column = 0u; sparse_column < params.k; sparse_column += 1u) {
                let column = indices[sparse_base + sparse_column];
                result[dense_base + column] -=
                    scale * probabilities[sparse_base + sparse_column];
            }
        }
    }
}
