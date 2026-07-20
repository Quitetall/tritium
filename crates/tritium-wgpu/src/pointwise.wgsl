struct Params {
    len: u32,
    operation: u32,
    scalar: f32,
    auxiliary: u32,
    secondary: u32,
    tertiary: u32,
    padding_0: u32,
    padding_1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> left: array<f32>;
@group(0) @binding(2) var<storage, read> right: array<f32>;
@group(0) @binding(3) var<storage, read> extra: array<f32>;
@group(0) @binding(4) var<storage, read_write> result: array<f32>;

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
        case 11u: {
            let rows = params.len / params.auxiliary;
            if (index >= rows) { return; }
            let base = index * params.auxiliary;
            var maximum = -3.402823466e38;
            for (var column = 0u; column < params.auxiliary; column = column + 1u) {
                maximum = max(maximum, left[base + column]);
            }
            var sum = 0.0;
            for (var column = 0u; column < params.auxiliary; column = column + 1u) {
                let exponential = exp(left[base + column] - maximum);
                result[base + column] = exponential;
                sum = sum + exponential;
            }
            for (var column = 0u; column < params.auxiliary; column = column + 1u) {
                result[base + column] = result[base + column] / sum;
            }
        }
        case 12u: {
            let rows = params.len / params.auxiliary;
            if (index >= rows) { return; }
            let base = index * params.auxiliary;
            var maximum = -3.402823466e38;
            for (var column = 0u; column < params.auxiliary; column = column + 1u) {
                maximum = max(maximum, left[base + column]);
            }
            var sum = 0.0;
            for (var column = 0u; column < params.auxiliary; column = column + 1u) {
                sum = sum + exp(left[base + column] - maximum);
            }
            var contraction = 0.0;
            for (var column = 0u; column < params.auxiliary; column = column + 1u) {
                let probability = exp(left[base + column] - maximum) / sum;
                contraction = contraction + probability * right[base + column];
            }
            for (var column = 0u; column < params.auxiliary; column = column + 1u) {
                let probability = exp(left[base + column] - maximum) / sum;
                result[base + column] = probability * (right[base + column] - contraction);
            }
        }
        case 13u: {
            let rows = params.len / params.auxiliary;
            if (index >= rows) { return; }
            let base = index * params.auxiliary;
            var mean_square = 0.0;
            for (var column = 0u; column < params.auxiliary; column = column + 1u) {
                let value = left[base + column];
                mean_square = mean_square + value * value;
            }
            mean_square = mean_square / f32(params.auxiliary);
            let inverse = 1.0 / sqrt(mean_square + params.scalar);
            for (var column = 0u; column < params.auxiliary; column = column + 1u) {
                result[base + column] = left[base + column] * inverse * right[column];
            }
        }
        case 14u: {
            let rows = params.len / params.auxiliary;
            if (index >= rows) { return; }
            let base = index * params.auxiliary;
            var mean_square = 0.0;
            for (var column = 0u; column < params.auxiliary; column = column + 1u) {
                let value = left[base + column];
                mean_square = mean_square + value * value;
            }
            mean_square = mean_square / f32(params.auxiliary);
            let inverse = 1.0 / sqrt(mean_square + params.scalar);
            var contraction = 0.0;
            for (var column = 0u; column < params.auxiliary; column = column + 1u) {
                contraction = contraction
                    + extra[base + column] * right[column] * left[base + column];
            }
            let correction = inverse * inverse * inverse * contraction
                / f32(params.auxiliary);
            for (var column = 0u; column < params.auxiliary; column = column + 1u) {
                result[base + column] = inverse * extra[base + column] * right[column]
                    - correction * left[base + column];
            }
        }
        case 15u: {
            if (index >= params.auxiliary) { return; }
            let rows = params.len / params.auxiliary;
            var gradient = 0.0;
            for (var row = 0u; row < rows; row = row + 1u) {
                let base = row * params.auxiliary;
                var mean_square = 0.0;
                for (var column = 0u; column < params.auxiliary; column = column + 1u) {
                    let value = left[base + column];
                    mean_square = mean_square + value * value;
                }
                mean_square = mean_square / f32(params.auxiliary);
                let inverse = 1.0 / sqrt(mean_square + params.scalar);
                gradient = gradient + extra[base + index] * left[base + index] * inverse;
            }
            result[index] = gradient;
        }
        case 16u: {
            if (index != 0u) { return; }
            var loss = 0.0;
            for (var item = 0u; item < params.len; item = item + 1u) {
                let difference = left[item] - right[item];
                loss = loss + difference * difference;
            }
            result[0] = loss / f32(params.len);
        }
        case 17u: {
            result[index] = params.scalar * 2.0 * (left[index] - right[index])
                / f32(params.len);
        }
        case 18u: { result[index] = left[index] + right[index % params.auxiliary]; }
        case 19u: { result[index] = extra[index]; }
        case 20u: {
            if (index >= params.auxiliary) { return; }
            let rows = params.len / params.auxiliary;
            var gradient = 0.0;
            for (var row = 0u; row < rows; row = row + 1u) {
                gradient = gradient + extra[row * params.auxiliary + index];
            }
            result[index] = gradient;
        }
        case 21u: { result[index] = left[index] - params.scalar * right[index]; }
        case 22u: {
            let rows = params.len / params.auxiliary;
            result[index] = left[(index % rows) * params.auxiliary + index / rows];
        }
        case 23u: {
            let rows = params.len / params.auxiliary;
            result[index] = left[(index % params.auxiliary) * rows + index / params.auxiliary];
        }
        case 24u: {
            let sliced_columns = params.tertiary;
            let row = index / sliced_columns;
            let column = index % sliced_columns;
            result[index] = left[row * params.auxiliary + params.secondary + column];
        }
        case 25u: {
            let row = index / params.auxiliary;
            let column = index % params.auxiliary;
            if (column >= params.secondary && column < params.secondary + params.tertiary) {
                result[index] = left[row * params.tertiary + column - params.secondary];
            } else {
                result[index] = 0.0;
            }
        }
        case 26u: {
            let row = index / params.secondary;
            let output_column = index % params.secondary;
            var accumulator = 0.0;
            for (var inner = 0u; inner < params.tertiary; inner = inner + 1u) {
                accumulator = accumulator
                    + left[row * params.tertiary + inner]
                    * right[output_column * params.tertiary + inner];
            }
            result[index] = accumulator;
        }
        case 27u: {
            let row = index / params.tertiary;
            let inner = index % params.tertiary;
            var accumulator = 0.0;
            for (var output_column = 0u; output_column < params.secondary;
                 output_column = output_column + 1u) {
                accumulator = accumulator
                    + extra[row * params.secondary + output_column]
                    * right[output_column * params.tertiary + inner];
            }
            result[index] = accumulator;
        }
        case 28u: {
            let output_column = index / params.tertiary;
            let inner = index % params.tertiary;
            var accumulator = 0.0;
            for (var row = 0u; row < params.auxiliary; row = row + 1u) {
                accumulator = accumulator
                    + extra[row * params.secondary + output_column]
                    * left[row * params.tertiary + inner];
            }
            result[index] = accumulator;
        }
        case 29u: {
            let row = index / params.secondary;
            let output_column = index % params.secondary;
            var accumulator = 0.0;
            for (var inner = 0u; inner < params.tertiary; inner = inner + 1u) {
                accumulator = accumulator
                    + left[row * params.tertiary + inner]
                    * right[output_column * params.tertiary + inner];
            }
            result[index] = extra[output_column] * accumulator;
        }
        case 30u: {
            let row = index / params.tertiary;
            let inner = index % params.tertiary;
            var accumulator = 0.0;
            for (var output_column = 0u; output_column < params.secondary;
                 output_column = output_column + 1u) {
                accumulator = accumulator
                    + left[row * params.secondary + output_column]
                    * extra[output_column]
                    * right[output_column * params.tertiary + inner];
            }
            result[index] = accumulator;
        }
        case 31u: {
            let output_column = index / params.tertiary;
            let inner = index % params.tertiary;
            var accumulator = 0.0;
            for (var row = 0u; row < params.auxiliary; row = row + 1u) {
                accumulator = accumulator
                    + left[row * params.secondary + output_column]
                    * extra[output_column]
                    * right[row * params.tertiary + inner];
            }
            result[index] = accumulator;
        }
        case 32u: {
            let output_column = index;
            var gradient = 0.0;
            for (var row = 0u; row < params.auxiliary; row = row + 1u) {
                var contraction = 0.0;
                for (var inner = 0u; inner < params.tertiary; inner = inner + 1u) {
                    contraction = contraction
                        + right[row * params.tertiary + inner]
                        * extra[output_column * params.tertiary + inner];
                }
                gradient = gradient
                    + left[row * params.secondary + output_column] * contraction;
            }
            result[index] = gradient;
        }
        case 33u: {
            let row = index / params.auxiliary;
            let row_scale = right[row];
            if (row_scale == 0.0) {
                result[index] = 0.0;
            } else {
                result[index] = clamp(left[index] / row_scale, -1.0, 1.0);
            }
        }
        case 34u: {
            let row = index / params.auxiliary;
            let row_scale = right[row];
            if (row_scale != 0.0 && abs(left[index] / row_scale) < 1.0) {
                result[index] = extra[index] / row_scale;
            } else {
                result[index] = 0.0;
            }
        }
        default: { result[index] = 0.0; }
    }
}
