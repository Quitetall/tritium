struct Params {
    rows: u32,
    cols: u32,
    planes: u32,
    padding: u32,
};

struct NormalParts {
    mantissa: u32,
    exponent: i32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> residual: array<f32>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;

fn normal_parts(bits: u32) -> NormalParts {
    let exponent_bits = (bits >> 23u) & 255u;
    var mantissa = bits & 0x007fffffu;
    var exponent = -126i;
    if (exponent_bits != 0u) {
        mantissa |= 0x00800000u;
        exponent = i32(exponent_bits) - 127i;
    } else {
        loop {
            if ((mantissa & 0x00800000u) != 0u) {
                break;
            }
            mantissa <<= 1u;
            exponent -= 1i;
        }
    }
    return NormalParts(mantissa, exponent);
}

fn div_cr(numerator: f32, denominator: f32) -> f32 {
    let numerator_bits = bitcast<u32>(numerator);
    let denominator_bits = bitcast<u32>(denominator);
    if ((numerator_bits & 0x7fffffffu) == 0u) {
        return numerator;
    }
    let sign = (numerator_bits ^ denominator_bits) & 0x80000000u;
    let left = normal_parts(numerator_bits & 0x7fffffffu);
    let right = normal_parts(denominator_bits & 0x7fffffffu);
    var exponent = left.exponent - right.exponent;
    var dividend = left.mantissa;
    if (dividend < right.mantissa) {
        dividend <<= 1u;
        exponent -= 1i;
    }
    var remainder = dividend - right.mantissa;
    var quotient = 1u;
    for (var bit = 0u; bit < 25u; bit += 1u) {
        remainder <<= 1u;
        quotient <<= 1u;
        if (remainder >= right.mantissa) {
            remainder -= right.mantissa;
            quotient |= 1u;
        }
    }
    var significand = quotient >> 2u;
    let guard = (quotient & 2u) != 0u;
    let round_or_sticky = (quotient & 1u) != 0u || remainder != 0u;
    if (guard && (round_or_sticky || (significand & 1u) != 0u)) {
        significand += 1u;
    }
    if (significand == 0x01000000u) {
        significand >>= 1u;
        exponent += 1i;
    }
    let biased = exponent + 127i;
    if (biased <= 0i || biased >= 255i) {
        return numerator / denominator;
    }
    return bitcast<f32>(sign | (u32(biased) << 23u) | (significand & 0x007fffffu));
}

fn round_away(value: f32) -> f32 {
    return select(ceil(value - 0.5), floor(value + 0.5), value >= 0.0);
}

@compute @workgroup_size(1, 1, 1)
fn main() {
    for (var row = 0u; row < params.rows; row += 1u) {
        let base = row * params.cols;
        for (var column = 0u; column < params.cols; column += 1u) {
            residual[column] = weight[base + column];
            result[base + column] = 0.0;
        }
        for (var plane = 0u; plane < params.planes; plane += 1u) {
            var sum = 0.0;
            for (var column = 0u; column < params.cols; column += 1u) {
                sum += abs(residual[column]);
            }
            let scale = div_cr(sum, f32(params.cols));
            if (scale != 0.0) {
                for (var column = 0u; column < params.cols; column += 1u) {
                    let index = base + column;
                    let old = residual[column];
                    let trit = clamp(round_away(div_cr(old, scale)), -1.0, 1.0);
                    residual[column] = scale * trit;
                    result[index] = result[index] + residual[column];
                    residual[column] = old - residual[column];
                }
            }
        }
    }
}
