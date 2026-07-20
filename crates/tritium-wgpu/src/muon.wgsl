struct Params {
    len: u32,
    rows: u32,
    cols: u32,
    steps: u32,
    momentum_decay: f32,
    scale: f32,
    shrink: f32,
    padding: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> parameter: array<f32>;
@group(0) @binding(2) var<storage, read> gradient: array<f32>;
@group(0) @binding(3) var<storage, read_write> momentum: array<f32>;
@group(0) @binding(4) var<storage, read_write> workspace: array<f32>;

fn sqrt_cr(value: f32) -> f32 {
    var bits = bitcast<u32>(value);
    if (bits == 0u) {
        return value;
    }
    var exponent = i32(bits >> 23u);
    if (exponent == 0) {
        var shifts = 0i;
        loop {
            if ((bits & 0x00800000u) != 0u) {
                break;
            }
            bits <<= 1u;
            shifts += 1i;
        }
        exponent -= shifts - 1i;
    }
    exponent -= 127i;
    bits = (bits & 0x007fffffu) | 0x00800000u;
    if ((exponent & 1i) != 0i) {
        bits += bits;
    }
    exponent >>= 1u;
    bits += bits;
    var root = 0u;
    var accumulator = 0u;
    var moving = 0x01000000u;
    loop {
        if (moving == 0u) {
            break;
        }
        let trial = accumulator + moving;
        if (trial <= bits) {
            accumulator = trial + moving;
            bits -= trial;
            root += moving;
        }
        bits += bits;
        moving >>= 1u;
    }
    if (bits != 0u) {
        root += root & 1u;
    }
    return bitcast<f32>((root >> 1u) + 0x3f000000u + (u32(exponent) << 23u));
}

struct NormalParts {
    mantissa: u32,
    exponent: i32,
};

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

@compute @workgroup_size(1, 1, 1)
fn main() {
    let n = params.len;
    let r = min(params.rows, params.cols);
    let c = max(params.rows, params.cols);
    let square = r * r;
    let x_offset = 0u;
    let xt_offset = n;
    let bx_offset = 2u * n;
    let gram_offset = 3u * n;
    let gram2_offset = gram_offset + square;
    let bmat_offset = gram2_offset + square;
    let temporary = bmat_offset + square;

    for (var index = 0u; index < n; index += 1u) {
        workspace[temporary] = params.momentum_decay * momentum[index];
        storageBarrier();
        momentum[index] = workspace[temporary] + gradient[index];
    }
    storageBarrier();
    for (var index = 0u; index < n; index += 1u) {
        workspace[bx_offset + index] = momentum[index] * momentum[index];
    }
    storageBarrier();
    var norm2 = 0.0;
    for (var index = 0u; index < n; index += 1u) {
        norm2 += workspace[bx_offset + index];
    }
    let fnorm = sqrt_cr(norm2) + 1.0e-7;
    let transposed = params.rows > params.cols;
    for (var row = 0u; row < r; row += 1u) {
        for (var col = 0u; col < c; col += 1u) {
            let source = select(row * params.cols + col, col * params.cols + row, transposed);
            workspace[x_offset + row * c + col] = div_cr(momentum[source], fnorm);
        }
    }

    for (var iteration = 0u; iteration < params.steps; iteration += 1u) {
        storageBarrier();
        for (var row = 0u; row < r; row += 1u) {
            for (var col = 0u; col < r; col += 1u) {
                var sum = 0.0;
                for (var k = 0u; k < c; k += 1u) {
                    workspace[temporary] = workspace[x_offset + row * c + k]
                        * workspace[x_offset + col * c + k];
                    storageBarrier();
                    sum += workspace[temporary];
                }
                workspace[gram_offset + row * r + col] = sum;
            }
        }
        storageBarrier();
        for (var row = 0u; row < r; row += 1u) {
            for (var col = 0u; col < r; col += 1u) {
                var sum = 0.0;
                for (var k = 0u; k < r; k += 1u) {
                    workspace[temporary] = workspace[gram_offset + row * r + k]
                        * workspace[gram_offset + col * r + k];
                    storageBarrier();
                    sum += workspace[temporary];
                }
                workspace[gram2_offset + row * r + col] = sum;
            }
        }
        storageBarrier();
        for (var index = 0u; index < square; index += 1u) {
            workspace[temporary] = -4.775 * workspace[gram_offset + index];
            workspace[temporary + 1u] = 2.0315 * workspace[gram2_offset + index];
            storageBarrier();
            workspace[bmat_offset + index] = workspace[temporary] + workspace[temporary + 1u];
        }
        for (var row = 0u; row < c; row += 1u) {
            for (var col = 0u; col < r; col += 1u) {
                workspace[xt_offset + row * r + col] = workspace[x_offset + col * c + row];
            }
        }
        storageBarrier();
        for (var row = 0u; row < r; row += 1u) {
            for (var col = 0u; col < c; col += 1u) {
                var sum = 0.0;
                for (var k = 0u; k < r; k += 1u) {
                    workspace[temporary] = workspace[bmat_offset + row * r + k]
                        * workspace[xt_offset + col * r + k];
                    storageBarrier();
                    sum += workspace[temporary];
                }
                workspace[bx_offset + row * c + col] = sum;
            }
        }
        storageBarrier();
        for (var index = 0u; index < n; index += 1u) {
            workspace[temporary] = 3.4445 * workspace[x_offset + index];
            storageBarrier();
            workspace[x_offset + index] = workspace[temporary] + workspace[bx_offset + index];
        }
    }

    storageBarrier();
    for (var row = 0u; row < params.rows; row += 1u) {
        for (var col = 0u; col < params.cols; col += 1u) {
            let index = row * params.cols + col;
            let ortho = select(index, col * c + row, transposed);
            workspace[temporary] = parameter[index] * params.shrink;
            workspace[temporary + 1u] = params.scale * workspace[x_offset + ortho];
            storageBarrier();
            parameter[index] = workspace[temporary] - workspace[temporary + 1u];
        }
    }
}
