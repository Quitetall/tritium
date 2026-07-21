struct Params {
    rows: u32,
    cols: u32,
    execution: u32,
    padding: u32,
    padding_4: f32,
    padding_1: f32,
    padding_2: f32,
    padding_3: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read> targets: array<f32>;
@group(0) @binding(3) var<storage, read> grad_output: array<f32>;
@group(0) @binding(4) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(1, 1, 1)
fn main() {
    if (params.execution == 0u) {
        var total = 0.0;
        for (var row = 0u; row < params.rows; row += 1u) {
            let base = row * params.cols;
            var maximum = bitcast<f32>(0xff800000u);
            for (var column = 0u; column < params.cols; column += 1u) {
                maximum = max(maximum, logits[base + column]);
            }
            var sum = 0.0;
            for (var column = 0u; column < params.cols; column += 1u) {
                sum += exp(logits[base + column] - maximum);
            }
            for (var column = 0u; column < params.cols; column += 1u) {
                let probability = exp(logits[base + column] - maximum) / sum;
                total -= targets[base + column] * log(max(probability, 1.17549435e-38));
            }
        }
        result[0] = total / f32(params.rows);
    } else {
        for (var row = 0u; row < params.rows; row += 1u) {
            let base = row * params.cols;
            var maximum = bitcast<f32>(0xff800000u);
            for (var column = 0u; column < params.cols; column += 1u) {
                maximum = max(maximum, logits[base + column]);
            }
            var sum = 0.0;
            var target_sum = 0.0;
            for (var column = 0u; column < params.cols; column += 1u) {
                sum += exp(logits[base + column] - maximum);
                target_sum += targets[base + column];
            }
            for (var column = 0u; column < params.cols; column += 1u) {
                let probability = exp(logits[base + column] - maximum) / sum;
                result[base + column] = (grad_output[0] / f32(params.rows))
                    * (probability * target_sum - targets[base + column]);
            }
        }
    }
}
