struct ConvParams {
    batch: u32,
    c_in: u32,
    c_out: u32,
    input_h: u32,
    input_w: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    pad_top: u32,
    pad_left: u32,
    groups: u32,
    output_h: u32,
    output_w: u32,
    execution: u32,
    pad_bottom: u32,
    pad_right: u32,
    padding: u32,
}

@group(0) @binding(0) var<uniform> params: ConvParams;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> scale: array<f32>;
@group(0) @binding(4) var<storage, read> grad_output: array<f32>;
@group(0) @binding(5) var<storage, read_write> result: array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_weight: array<f32>;
@group(0) @binding(7) var<storage, read_write> grad_scale: array<f32>;

fn input_value(b: u32, channel: u32, oh: u32, ow: u32, kh: u32, kw: u32) -> f32 {
    let ih = i32(oh * params.stride_h + kh * params.dilation_h) - i32(params.pad_top);
    let iw = i32(ow * params.stride_w + kw * params.dilation_w) - i32(params.pad_left);
    if (ih < 0 || ih >= i32(params.input_h) || iw < 0 || iw >= i32(params.input_w)) {
        return 0.0;
    }
    let index = ((b * params.c_in + channel) * params.input_h + u32(ih)) * params.input_w + u32(iw);
    return x[index];
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u || gid.y != 0u || gid.z != 0u) {
        return;
    }
    let c_in_per_group = params.c_in / params.groups;
    let c_out_per_group = params.c_out / params.groups;
    let patch_columns = c_in_per_group * params.kernel_h * params.kernel_w;
    let patch_rows = params.output_h * params.output_w;

    if (params.execution == 0u) {
        for (var b = 0u; b < params.batch; b++) {
            for (var group = 0u; group < params.groups; group++) {
                for (var row = 0u; row < patch_rows; row++) {
                    let oh = row / params.output_w;
                    let ow = row - oh * params.output_w;
                    for (var n = 0u; n < c_out_per_group; n++) {
                        let co = group * c_out_per_group + n;
                        var acc = 0.0;
                        for (var column = 0u; column < patch_columns; column++) {
                            let kw = column % params.kernel_w;
                            let rest = column / params.kernel_w;
                            let kh = rest % params.kernel_h;
                            let ci_local = rest / params.kernel_h;
                            let channel = group * c_in_per_group + ci_local;
                            acc += input_value(b, channel, oh, ow, kh, kw) * weight[co * patch_columns + column];
                        }
                        let output_index = ((b * params.c_out + co) * params.output_h + oh) * params.output_w + ow;
                        result[output_index] = scale[co] * acc;
                    }
                }
            }
        }
        return;
    }

    for (var b = 0u; b < params.batch; b++) {
        for (var group = 0u; group < params.groups; group++) {
            for (var row_start = 0u; row_start < patch_rows; row_start += 32u) {
                let row_count = min(32u, patch_rows - row_start);
                for (var local_row = 0u; local_row < row_count; local_row++) {
                    let row = row_start + local_row;
                    let oh = row / params.output_w;
                    let ow = row - oh * params.output_w;
                    for (var n = 0u; n < c_out_per_group; n++) {
                        let co = group * c_out_per_group + n;
                        let output_index = ((b * params.c_out + co) * params.output_h + oh) * params.output_w + ow;
                        let gy = grad_output[output_index];
                        let s = scale[co];
                        var product = 0.0;
                        for (var column = 0u; column < patch_columns; column++) {
                            let kw = column % params.kernel_w;
                            let rest = column / params.kernel_w;
                            let kh = rest % params.kernel_h;
                            let ci_local = rest / params.kernel_h;
                            let channel = group * c_in_per_group + ci_local;
                            let activation = input_value(b, channel, oh, ow, kh, kw);
                            let weight_index = co * patch_columns + column;
                            let w = weight[weight_index];
                            product += activation * w;
                            grad_weight[weight_index] += gy * s * activation;

                            let ih = i32(oh * params.stride_h + kh * params.dilation_h) - i32(params.pad_top);
                            let iw = i32(ow * params.stride_w + kw * params.dilation_w) - i32(params.pad_left);
                            if (ih >= 0 && ih < i32(params.input_h) && iw >= 0 && iw < i32(params.input_w)) {
                                let input_index = ((b * params.c_in + channel) * params.input_h + u32(ih)) * params.input_w + u32(iw);
                                result[input_index] += gy * s * w;
                            }
                        }
                        grad_scale[co] += gy * product;
                    }
                }
            }
        }
    }
}
