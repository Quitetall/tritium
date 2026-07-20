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
@group(0) @binding(1) var<storage, read_write> parameter: array<f32>;
@group(0) @binding(2) var<storage, read> gradient: array<f32>;
@group(0) @binding(3) var<storage, read_write> moment1: array<u32>;
@group(0) @binding(4) var<storage, read_write> moment2: array<u32>;
@group(0) @binding(5) var<storage, read_write> moment1_scale: array<f32>;
@group(0) @binding(6) var<storage, read_write> moment2_scale: array<f32>;
@group(0) @binding(7) var<storage, read_write> scratch1: array<f32>;
@group(0) @binding(8) var<storage, read_write> scratch2: array<f32>;

var<workgroup> block_m: array<f32, 256>;
var<workgroup> block_v: array<f32, 256>;

fn signed_code(code: u32) -> f32 {
    if (code < 128u) {
        return f32(code);
    }
    return f32(i32(code) - 256);
}

// Correctly-rounded binary32 square root. Integer restoring algorithm from
// fdlibm/musl; round-to-nearest-even is explicit, independent of driver sqrt.
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
            bits = bits << 1u;
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
    let mantissa = (root >> 1u) + 0x3f000000u;
    return bitcast<f32>(mantissa + (u32(exponent) << 23u));
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

// Correctly-rounded binary32 division for finite nonzero normal results.
// Long division emits guard+round bits; remaining remainder is sticky.
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

@compute @workgroup_size(64, 1, 1)
fn dequantize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < params.len) {
        let block = index / 256u;
        moment1[index] = bitcast<u32>(signed_code(moment1[index]) * moment1_scale[block]);
        moment2[index] = bitcast<u32>(f32(moment2[index]) * moment2_scale[block]);
    }
}

@compute @workgroup_size(64, 1, 1)
fn square_variance(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < params.len) {
        let root = bitcast<f32>(moment2[index]);
        moment2[index] = bitcast<u32>(root * root);
    }
}

@compute @workgroup_size(64, 1, 1)
fn products(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < params.len) {
        let g = gradient[index];
        moment1[index] = bitcast<u32>(params.beta1 * bitcast<f32>(moment1[index]));
        scratch1[index] = (1.0 - params.beta1) * g;
        moment2[index] = bitcast<u32>(params.beta2 * bitcast<f32>(moment2[index]));
        scratch2[index] = (1.0 - params.beta2) * g;
        parameter[index] = parameter[index] * params.shrink;
    }
}

@compute @workgroup_size(64, 1, 1)
fn finish_products(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < params.len) {
        moment1[index] = bitcast<u32>(bitcast<f32>(moment1[index]) + scratch1[index]);
        scratch2[index] = scratch2[index] * gradient[index];
    }
}

@compute @workgroup_size(64, 1, 1)
fn finish_variance(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < params.len) {
        moment2[index] = bitcast<u32>(bitcast<f32>(moment2[index]) + scratch2[index]);
    }
}

@compute @workgroup_size(64, 1, 1)
fn update_parameter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < params.len) {
        let m = bitcast<f32>(moment1[index]);
        let v = bitcast<f32>(moment2[index]);
        let corrected_m = div_cr(m, params.correction1);
        let corrected_v = div_cr(v, params.correction2);
        let adaptive = params.learning_rate
            * div_cr(corrected_m, sqrt_cr(corrected_v) + params.epsilon);
        parameter[index] = parameter[index] - adaptive;
    }
}

@compute @workgroup_size(256, 1, 1)
fn reduce_scales(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let index = group.x * 256u + local.x;
    var m = 0.0;
    var root = 0.0;
    if (index < params.len) {
        m = abs(bitcast<f32>(moment1[index]));
        root = sqrt_cr(bitcast<f32>(moment2[index]));
        moment2[index] = bitcast<u32>(root);
    }
    block_m[local.x] = m;
    block_v[local.x] = root;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (local.x < stride) {
            block_m[local.x] = max(block_m[local.x], block_m[local.x + stride]);
            block_v[local.x] = max(block_v[local.x], block_v[local.x + stride]);
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }
    if (local.x == 0u) {
        moment1_scale[group.x] = select(0.0, div_cr(block_m[0], 127.0), block_m[0] > 0.0);
        moment2_scale[group.x] = select(0.0, div_cr(block_v[0], 255.0), block_v[0] > 0.0);
    }
}

@compute @workgroup_size(64, 1, 1)
fn quantize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < params.len) {
        let block = index / 256u;
        let m_scale = moment1_scale[block];
        let v_scale = moment2_scale[block];
        let m = bitcast<f32>(moment1[index]);
        let root = bitcast<f32>(moment2[index]);
        if (m_scale > 0.0) {
            let code = i32(clamp(round(div_cr(m, m_scale)), -127.0, 127.0));
            moment1[index] = u32(code) & 255u;
        } else {
            moment1[index] = 0u;
        }
        if (v_scale > 0.0 && root > 0.0) {
            moment2[index] = u32(clamp(round(div_cr(root, v_scale)), 1.0, 255.0));
        } else {
            moment2[index] = 0u;
        }
    }
}
