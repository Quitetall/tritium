struct Params {
    total: u32,
    len: u32,
    bound: u32,
    estimator: u32,
    execution: u32,
    alpha: f32,
    seed_low: u32,
    seed_high: u32,
};

struct U64 {
    low: u32,
    high: u32,
};

struct NormalParts {
    mantissa: u32,
    exponent: i32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> levels: array<u32>;
@group(0) @binding(3) var<storage, read> upstream: array<f32>;
@group(0) @binding(4) var<storage, read_write> result: array<f32>;

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

fn add64(left: U64, right: U64) -> U64 {
    let low = left.low + right.low;
    let carry = select(0u, 1u, low < left.low);
    return U64(low, left.high + right.high + carry);
}

fn shift_left_1(value: U64) -> U64 {
    return U64(value.low << 1u, (value.high << 1u) | (value.low >> 31u));
}

fn shift_left_13(value: U64) -> U64 {
    return U64(value.low << 13u, (value.high << 13u) | (value.low >> 19u));
}

fn shift_left_17(value: U64) -> U64 {
    return U64(value.low << 17u, (value.high << 17u) | (value.low >> 15u));
}

fn shift_right_7(value: U64) -> U64 {
    return U64((value.low >> 7u) | (value.high << 25u), value.high >> 7u);
}

fn multiply_index(index: u32) -> U64 {
    var bits = index;
    var addend = U64(0x7f4a7c15u, 0x9e3779b9u);
    var product = U64(0u, 0u);
    for (var position = 0u; position < 32u; position += 1u) {
        if ((bits & 1u) != 0u) {
            product = add64(product, addend);
        }
        bits >>= 1u;
        addend = shift_left_1(addend);
    }
    return product;
}

fn modulo_million(value: U64) -> u32 {
    var remainder = 0u;
    for (var position = 63i; position >= 0i; position -= 1i) {
        var bit = 0u;
        if (position >= 32i) {
            bit = (value.high >> u32(position - 32i)) & 1u;
        } else {
            bit = (value.low >> u32(position)) & 1u;
        }
        remainder = remainder * 2u + bit;
        if (remainder >= 1000000u) {
            remainder -= 1000000u;
        }
    }
    return remainder;
}

fn sample_uniform(index: u32) -> f32 {
    let product = multiply_index(index);
    var state = U64(params.seed_low ^ product.low, params.seed_high ^ product.high);
    state.low |= 1u;
    let left13 = shift_left_13(state);
    state = U64(state.low ^ left13.low, state.high ^ left13.high);
    let right7 = shift_right_7(state);
    state = U64(state.low ^ right7.low, state.high ^ right7.high);
    let left17 = shift_left_17(state);
    state = U64(state.low ^ left17.low, state.high ^ left17.high);
    return div_cr(f32(modulo_million(state)), 1000000.0);
}

fn bound_value(value: f32) -> f32 {
    return select(clamp(value, -1.0, 1.0), tanh(value), params.bound == 1u);
}

fn bound_derivative(value: f32) -> f32 {
    if (params.bound == 0u) {
        return select(0.0, 1.0, abs(value) < 1.0);
    }
    let bounded = tanh(value);
    return 1.0 - bounded * bounded;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= params.total) {
        return;
    }
    if (params.execution == 0u) {
        let maximum = f32(levels[index / params.len] - 1u);
        let bounded = bound_value(x[index]);
        var position = (bounded + 1.0) * 0.5 * maximum;
        var code = 0.0;
        if (params.estimator == 2u) {
            position = clamp(position, 0.0, maximum);
            let base = floor(position);
            code = base + select(0.0, 1.0, sample_uniform(index) < position - base);
        } else {
            code = floor(position + 0.5);
        }
        code = clamp(code, 0.0, maximum);
        result[index] = div_cr(code, maximum) * 2.0 - 1.0;
    } else {
        var derivative = bound_derivative(x[index]);
        if (params.estimator == 1u) {
            let maximum = f32(levels[index / params.len] - 1u);
            let position = (bound_value(x[index]) + 1.0) * 0.5 * maximum;
            derivative *= 1.0 - params.alpha * cos(6.283185307179586 * position);
        }
        result[index] = upstream[index] * derivative;
    }
}
