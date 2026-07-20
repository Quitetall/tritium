//! Regenerate the checked-in plan-0049 portable-training tracer corpus.
//!
//! Run deliberately when widening semantic coverage:
//!
//! ```text
//! cargo run -p tritium-train --example freeze_training_vectors
//! ```

use std::path::Path;

use half::f16;
use serde::Serialize;
use tritium_format::salt_v2::SaltV2Codec;
use tritium_format::salt_v2_package::{
    SaltV2Package, SaltV2Plane, SaltV2Tensor, SaltV2Tile, write_salt_v2_package,
};
use tritium_spec::{TrainingOpManifestV1, TrainingVectorSetV1};
use tritium_train::{
    AdamState, AdamW, CautiousAdamW, Int8AdamW, Muon, Optimizer, Sgd,
    checkpoint::{Checkpoint, LeafCheckpoint, write_checkpoint},
    ops::{
        act, attention, bias, conv1d, conv2d, dense, elementwise, embed,
        fsq::{self, FsqBound, FsqCfg, FsqSte},
        loss, matmul, norm, rope, shape, softmax, ste,
    },
};

#[derive(Serialize)]
struct Corpus {
    schema_id: &'static str,
    schema_version: u32,
    manifest_digest: String,
    cases: Vec<Case>,
}

#[derive(Serialize)]
struct Case {
    case_id: &'static str,
    operation: &'static str,
    execution: &'static str,
    tolerance: Tolerance,
    inputs: Vec<Buffer>,
    attributes: Vec<Attribute>,
    expected: Expected,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Tolerance {
    BitExact,
    AbsoluteRelative {
        absolute_bits: u32,
        relative_bits: u32,
    },
}

#[derive(Serialize)]
struct Buffer {
    name: &'static str,
    shape: Vec<u64>,
    data: Data,
}

#[derive(Serialize)]
#[serde(tag = "dtype", rename_all = "snake_case")]
enum Data {
    F32 { bits: Vec<u32> },
    U32 { values: Vec<u32> },
    Bytes { values: Vec<u8> },
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Attribute {
    F32 {
        name: &'static str,
        bits: u32,
    },
    U64 {
        name: &'static str,
        value: u64,
    },
    Bool {
        name: &'static str,
        value: bool,
    },
    U64List {
        name: &'static str,
        values: Vec<u64>,
    },
    U32List {
        name: &'static str,
        values: Vec<u32>,
    },
    Text {
        name: &'static str,
        value: &'static str,
    },
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Expected {
    Success {
        outputs: Vec<Buffer>,
        scratch_bytes_max: u64,
    },
    Error {
        category: &'static str,
        code: &'static str,
        outputs: Vec<Buffer>,
    },
}

fn f32_buffer(name: &'static str, shape: &[u64], values: &[f32]) -> Buffer {
    Buffer {
        name,
        shape: shape.to_vec(),
        data: Data::F32 {
            bits: values.iter().map(|value| value.to_bits()).collect(),
        },
    }
}

fn u32_buffer(name: &'static str, shape: &[u64], values: &[u32]) -> Buffer {
    Buffer {
        name,
        shape: shape.to_vec(),
        data: Data::U32 {
            values: values.to_vec(),
        },
    }
}

fn bytes_buffer(name: &'static str, shape: &[u64], values: &[u8]) -> Buffer {
    Buffer {
        name,
        shape: shape.to_vec(),
        data: Data::Bytes {
            values: values.to_vec(),
        },
    }
}

fn fsq_attributes(
    channels: u64,
    len: u64,
    levels: &[u32],
    bound: &'static str,
    ste: &'static str,
    alpha: f32,
    seed: u64,
) -> Vec<Attribute> {
    vec![
        Attribute::U64 {
            name: "channels",
            value: channels,
        },
        Attribute::U64 {
            name: "len",
            value: len,
        },
        Attribute::U32List {
            name: "levels",
            values: levels.to_vec(),
        },
        Attribute::Text {
            name: "bound",
            value: bound,
        },
        Attribute::Text {
            name: "ste",
            value: ste,
        },
        Attribute::F32 {
            name: "alpha",
            bits: alpha.to_bits(),
        },
        Attribute::U64 {
            name: "seed",
            value: seed,
        },
    ]
}

fn main() {
    let left = [1.0_f32, -2.0, 0.5];
    let right = [3.0_f32, 4.0, -1.5];
    let add: Vec<_> = left.iter().zip(right).map(|(&x, y)| x + y).collect();
    let grad_output = [0.25_f32, -0.5, 2.0];

    let quant_weight = [-1.2_f32, -0.2, 0.6, 0.0, 1.5, -0.75];
    let quant_scale = [0.8_f32, 1.0];
    let quant_grad_output = [1.0_f32, -2.0, 0.5, 0.25, 1.5, -1.0];
    let ste_result = ste::quantize_surrogate(&quant_weight, &quant_scale, 2, 3);
    let ste_grads = ste::quantize_vjp(&quant_weight, &quant_scale, 2, 3, &quant_grad_output);
    let salt_result = ste::salt_quantize_forward(&quant_weight, 2, 3, 2);
    let salt_grad = ste::salt_quantize_vjp(&quant_weight, 2, 3, 2, &quant_grad_output);
    let learned_scale = [0.75_f32, 1.25];
    let lsq_result = ste::lsq_forward(&quant_weight, &learned_scale, 2, 3);
    let lsq_grads = ste::lsq_vjp(&quant_weight, &learned_scale, 2, 3, &quant_grad_output);

    let fsq_input = [-1.2_f32, -0.25, 0.6, 0.9, 0.1, -0.9];
    let fsq_cfg = FsqCfg {
        channels: 2,
        len: 3,
        levels: vec![3, 5],
        bound: FsqBound::Clamp,
    };
    let fsq_ste = FsqSte::SoftRound { alpha: 0.5 };
    let fsq_result = fsq::forward(&fsq_input, &fsq_cfg, fsq_ste);
    let fsq_grads = fsq::vjp(&fsq_input, &fsq_cfg, fsq_ste, &quant_grad_output);
    let fsq_boundary_input = [-0.75_f32, -0.5, 0.5, 0.75];
    let fsq_boundary_cfg = FsqCfg {
        channels: 1,
        len: 4,
        levels: vec![3],
        bound: FsqBound::Clamp,
    };
    let fsq_hard_result = fsq::forward(&fsq_boundary_input, &fsq_boundary_cfg, FsqSte::Hard);
    let fsq_tanh_cfg = FsqCfg {
        channels: 1,
        len: 4,
        levels: vec![5],
        bound: FsqBound::Tanh,
    };
    let fsq_tanh_grad = fsq::vjp(
        &fsq_boundary_input,
        &fsq_tanh_cfg,
        FsqSte::Hard,
        &quant_grad_output[..4],
    );
    let fsq_stochastic_input = [-0.8_f32, -0.4, 0.2, 0.7];
    let fsq_stochastic_seed_7 = fsq::forward(
        &fsq_stochastic_input,
        &fsq_tanh_cfg,
        FsqSte::Stochastic { seed: 7 },
    );
    let fsq_stochastic_seed_8 = fsq::forward(
        &fsq_stochastic_input,
        &fsq_tanh_cfg,
        FsqSte::Stochastic { seed: 8 },
    );
    assert_ne!(fsq_stochastic_seed_7, fsq_stochastic_seed_8);

    let conv1d_cfg = conv1d::Conv1dCfg {
        batch: 1,
        c_in: 2,
        c_out: 2,
        l_in: 4,
        k: 2,
        stride: 1,
        dilation: 1,
        pad_left: 1,
        pad_right: 0,
        groups: 2,
    };
    let conv1d_x = [1.0_f32, -2.0, 0.5, 3.0, -1.0, 2.0, 4.0, -0.5];
    let conv1d_weight = [1.0_f32, -0.5, 0.25, 1.5];
    let conv1d_scale = [0.75_f32, 1.25];
    let conv1d_result = conv1d::forward(&conv1d_x, &conv1d_weight, &conv1d_scale, &conv1d_cfg);
    let conv1d_grad_output = [0.5_f32, -1.0, 0.25, 2.0, -0.75, 1.5, 0.5, -0.25];
    let conv1d_grads = conv1d::vjp(
        &conv1d_x,
        &conv1d_weight,
        &conv1d_scale,
        &conv1d_cfg,
        &conv1d_grad_output,
    );
    let conv1d_zero_groups = conv1d::Conv1dCfg {
        groups: 0,
        ..conv1d_cfg
    };
    let conv1d_ragged_groups = conv1d::Conv1dCfg {
        groups: 3,
        ..conv1d_cfg
    };
    let conv1d_axis_boundary = conv1d::Conv1dCfg {
        batch: 1,
        c_in: 1,
        c_out: 1,
        l_in: 1,
        k: 1,
        stride: u32::MAX as usize,
        dilation: 1,
        pad_left: 0,
        pad_right: u32::MAX as usize,
        groups: 1,
    };
    let conv1d_scratch_overflow = conv1d::Conv1dCfg {
        l_in: 20_000_000,
        stride: 1,
        pad_right: 0,
        ..conv1d_axis_boundary
    };

    let conv2d_cfg = conv2d::Conv2dCfg {
        batch: 1,
        c_in: 2,
        c_out: 2,
        input_h: 3,
        input_w: 4,
        kernel_h: 2,
        kernel_w: 2,
        stride_h: 1,
        stride_w: 2,
        dilation_h: 1,
        dilation_w: 1,
        pad_top: 1,
        pad_bottom: 0,
        pad_left: 0,
        pad_right: 1,
        groups: 2,
    };
    let conv2d_x = [
        1.0_f32, -2.0, 0.5, 3.0, -1.0, 2.0, 4.0, -0.5, 0.25, 1.5, -3.0, 2.5, -0.75, 0.5, 2.0, -1.5,
        3.0, -2.5, 1.25, 0.75, -1.0, 4.0, -0.25, 2.25,
    ];
    let conv2d_weight = [1.0_f32, -0.5, 0.25, 1.5, -1.0, 0.75, 0.5, -0.25];
    let conv2d_scale = [0.8_f32, 1.2];
    let conv2d_result = conv2d::forward(&conv2d_x, &conv2d_weight, &conv2d_scale, &conv2d_cfg);
    let conv2d_grad_output = [
        0.5_f32, -1.0, 0.25, 2.0, -0.75, 1.5, 0.5, -0.25, 1.0, -0.5, 0.75, -1.25,
    ];
    let conv2d_grads = conv2d::vjp(
        &conv2d_x,
        &conv2d_weight,
        &conv2d_scale,
        &conv2d_cfg,
        &conv2d_grad_output,
    );
    let conv2d_zero_groups = conv2d::Conv2dCfg {
        groups: 0,
        ..conv2d_cfg
    };
    let conv2d_oversized_kernel = conv2d::Conv2dCfg {
        kernel_h: 8,
        ..conv2d_cfg
    };

    let attention_cfg = attention::AttentionCfg {
        seq: 3,
        n_head: 2,
        n_kv_head: 1,
        head_dim: 2,
        causal: true,
    };
    let attention_noncausal_cfg = attention::AttentionCfg {
        causal: false,
        ..attention_cfg
    };
    let attention_ragged_gqa_cfg = attention::AttentionCfg {
        n_head: 3,
        n_kv_head: 2,
        ..attention_cfg
    };
    let attention_q = [
        0.2_f32, -0.1, 0.4, 0.3, -0.5, 0.7, 0.1, -0.2, 0.6, 0.8, -0.3, 0.9,
    ];
    let attention_k = [0.5_f32, -0.4, 0.2, 0.1, -0.6, 0.7];
    let attention_v = [1.0_f32, -1.0, 0.5, 0.25, -0.75, 1.5];
    let attention_grad_output = [
        0.25_f32, -0.5, 0.75, 0.1, -0.2, 0.4, -0.6, 0.3, 0.9, -0.8, 0.2, 0.5,
    ];
    let attention_result =
        attention::forward(&attention_q, &attention_k, &attention_v, attention_cfg);
    let attention_noncausal_result = attention::forward(
        &attention_q,
        &attention_k,
        &attention_v,
        attention_noncausal_cfg,
    );
    let attention_grads = attention::vjp(
        &attention_q,
        &attention_k,
        &attention_v,
        attention_cfg,
        &attention_grad_output,
    );
    let attention_noncausal_grads = attention::vjp(
        &attention_q,
        &attention_k,
        &attention_v,
        attention_noncausal_cfg,
        &attention_grad_output,
    );
    let attention_gqa_cfg = attention::AttentionCfg {
        seq: 2,
        n_head: 4,
        n_kv_head: 2,
        head_dim: 1,
        causal: true,
    };
    let attention_gqa_q = [0.2_f32, -0.4, 0.7, 0.1, -0.3, 0.8, -0.6, 0.5];
    let attention_gqa_k = [0.5_f32, -0.25, -0.75, 0.9];
    let attention_gqa_v = [1.0_f32, 10.0, -2.0, 20.0];
    let attention_gqa_grad_output = [0.25_f32, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0];
    let attention_gqa_result = attention::forward(
        &attention_gqa_q,
        &attention_gqa_k,
        &attention_gqa_v,
        attention_gqa_cfg,
    );
    let attention_gqa_grads = attention::vjp(
        &attention_gqa_q,
        &attention_gqa_k,
        &attention_gqa_v,
        attention_gqa_cfg,
        &attention_gqa_grad_output,
    );
    let attention_product_limit_cfg = attention::AttentionCfg {
        seq: 65_536,
        n_head: 1,
        n_kv_head: 1,
        head_dim: 1,
        causal: true,
    };

    let mul_left = [2.0_f32, -3.0, 0.0];
    let mul_right = [-4.0_f32, 0.5, 7.0];
    let mul = elementwise::mul_forward(&mul_left, &mul_right);
    let mul_grad_output = [0.25_f32, -2.0, 3.0];
    let mul_grads = elementwise::mul_vjp(&mul_left, &mul_right, &mul_grad_output);

    let matmul_x = [1.0_f32, -2.0, 0.5, 0.0, 3.0, -1.0];
    let dense_weight = [0.5_f32, 1.0, -2.0, -1.0, 0.25, 2.0];
    let matmul_grad_output = [1.0_f32, -0.5, 2.0, 1.5];
    let dense_result = dense::forward(&matmul_x, &dense_weight, 2, 2, 3);
    let dense_grads = dense::vjp(&matmul_x, &dense_weight, 2, 2, 3, &matmul_grad_output);
    let ternary_weight = [1.0_f32, -1.0, 0.0, 0.0, 1.0, -1.0];
    let ternary_scale = [0.5_f32, 1.5];
    let ternary_result = matmul::forward(&matmul_x, &ternary_weight, &ternary_scale, 2, 2, 3);
    let ternary_grads = matmul::vjp(
        &matmul_x,
        &ternary_weight,
        &ternary_scale,
        2,
        2,
        3,
        &matmul_grad_output,
    );

    let matrix = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let transpose = dense::transpose_forward(&matrix, 2, 3);
    let transpose_grad_output = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let transpose_grad = dense::transpose_vjp(2, 3, &transpose_grad_output);

    let embedding_weight = [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let tokens = [2_u32, 0, 2];
    let embedding = embed::gather_forward(&embedding_weight, &tokens, 2);
    let embedding_grad_output = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let embedding_grad = embed::gather_vjp(4, &tokens, 2, &embedding_grad_output);

    let column_matrix = [0.0_f32, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0];
    let column_slice = shape::slice_cols_forward(&column_matrix, 2, 4, 1, 2);
    let slice_grad_output = [1.0_f32, 2.0, 3.0, 4.0];
    let slice_grad = shape::slice_cols_vjp(2, 4, 1, 2, &slice_grad_output);

    let concat_left = [1.0_f32, 2.0, 3.0, 4.0];
    let concat_right = [5.0_f32, 6.0];
    let concatenated = shape::concat_cols_forward(&[&concat_left, &concat_right], 2, &[2, 1]);
    let concat_grad_output = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let concat_grads = shape::concat_cols_vjp(2, &[2, 1], &concat_grad_output);

    let unary_input = [-2.0_f32, 0.0, 3.0];
    let unary_grad_output = [1.0_f32, 2.0, 0.5];
    let scale = 0.25_f32;
    let scaled: Vec<_> = unary_input.iter().map(|value| value * scale).collect();
    let scaled_grad: Vec<_> = unary_grad_output
        .iter()
        .map(|value| value * scale)
        .collect();
    let relu2 = act::relu2_forward(&unary_input);
    let relu2_grad = act::relu2_vjp(&unary_input, &unary_grad_output);

    let silu_input = [-1.0_f32, 0.0, 2.0];
    let silu_grad_output = [0.5_f32, -1.0, 2.0];
    let silu = act::silu_forward(&silu_input);
    let silu_grad = act::silu_vjp(&silu_input, &silu_grad_output);

    let bias_input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bias_value = [0.5_f32, -1.0, 2.0];
    let bias_grad_output = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bias_result = bias::forward(&bias_input, &bias_value, 2, 3);
    let bias_grads = bias::vjp(&bias_input, &bias_value, 2, 3, &bias_grad_output);

    let prediction = [1.0_f32, -1.0, 2.0];
    let target = [0.0_f32, 1.0, 2.5];
    let loss_grad_output = [0.5_f32];
    let mse = loss::mse_forward(&prediction, &target);
    let mse_grad = loss::mse_vjp(&prediction, &target, &loss_grad_output);

    let normalized_input = [1.0_f32, -2.0, 0.5, 0.25, 1.5, -1.0];
    let norm_weight = [1.0_f32, 0.5, 2.0];
    let transformer_grad_output = [0.5_f32, -1.0, 2.0, 1.5, 0.25, -0.75];
    let norm_epsilon = 1.0e-5_f32;
    let rmsnorm = norm::forward(&normalized_input, &norm_weight, 2, 3, norm_epsilon);
    let rmsnorm_grads = norm::vjp(
        &normalized_input,
        &norm_weight,
        2,
        3,
        norm_epsilon,
        &transformer_grad_output,
    );

    let logits = [1.0_f32, 2.0, -1.0, 0.0, -2.0, 3.0];
    let probabilities = softmax::forward(&logits, 2, 3);
    let softmax_grad = softmax::vjp(&logits, 2, 3, &transformer_grad_output);
    let causal = softmax::causal_mask_forward(&logits, 2, 3);
    let causal_grad = softmax::causal_mask_vjp(2, 3, &transformer_grad_output);
    let class_target = [0.0_f32, 1.0, 0.0, 1.0, 0.0, 0.0];
    let xent_grad_output = [0.75_f32];
    let xent = loss::softmax_xent_forward(&logits, &class_target, 2, 3);
    let xent_grad = loss::softmax_xent_vjp(&logits, &class_target, 2, 3, &xent_grad_output);

    let rope_input = [1.0_f32, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -3.0];
    let rope_positions = [0_usize, 3];
    let rope_theta = 10_000.0_f32;
    let rope_result = rope::forward(&rope_input, &rope_positions, 1, 4, rope_theta);
    let rope_grad_output = [0.5_f32, -1.0, 2.0, 0.25, 1.5, -0.5, 0.75, -2.0];
    let rope_grad = rope::vjp(&rope_positions, 1, 4, rope_theta, &rope_grad_output);

    let parameter = [1.0_f32, -2.0];
    let gradient = [0.5_f32, -0.25];
    let optimizer = Sgd::new(0.1);
    let mut updated = parameter;
    let mut state = optimizer.init_state(updated.len());
    optimizer.step(1, &mut updated, &gradient, &mut state);

    let adam_config = AdamW {
        lr: 0.01,
        beta1: 0.8,
        beta2: 0.95,
        eps: 1.0e-6,
        weight_decay: 0.02,
    };
    let adam_invalid_beta1 = AdamW {
        beta1: 1.0,
        ..adam_config
    };
    let adam_seed_parameter = [1.0_f32, -2.0, 0.5, -0.25];
    let adam_seed_gradient = [0.4_f32, -0.2, 0.0, 0.75];
    let adam_gradient = [-0.1_f32, -0.2, 0.3, 0.0];
    let mut adam_parameter = adam_seed_parameter;
    let mut adam_state = adam_config.init_state(adam_parameter.len());
    adam_config.step(1, &mut adam_parameter, &adam_seed_gradient, &mut adam_state);
    let adam_input_state = adam_state.clone();
    let mut adam_updated = adam_parameter;
    adam_config.step(2, &mut adam_updated, &adam_gradient, &mut adam_state);

    let cautious_config = CautiousAdamW(adam_config);
    let mut cautious_parameter = adam_seed_parameter;
    let mut cautious_state = cautious_config.init_state(cautious_parameter.len());
    cautious_config.step(
        1,
        &mut cautious_parameter,
        &adam_seed_gradient,
        &mut cautious_state,
    );
    let cautious_input_state = cautious_state.clone();
    let cautious_gradient = [-0.8_f32, -0.1, -0.4, 0.25];
    let mut cautious_updated = cautious_parameter;
    cautious_config.step(
        2,
        &mut cautious_updated,
        &cautious_gradient,
        &mut cautious_state,
    );

    let int8_config = Int8AdamW(adam_config);
    let mut int8_parameter = vec![0.0_f32; 260];
    let mut int8_seed_gradient = vec![0.0_f32; 260];
    int8_seed_gradient[0] = 8.0;
    int8_seed_gradient[1] = -0.25;
    int8_seed_gradient[255] = 0.5;
    int8_seed_gradient[256] = 0.001;
    int8_seed_gradient[259] = -0.002;
    let mut int8_state = int8_config.init_state(int8_parameter.len());
    int8_config.step(1, &mut int8_parameter, &int8_seed_gradient, &mut int8_state);
    let int8_input_state = int8_state.clone();
    let int8_gradient = vec![0.0_f32; 260];
    let mut int8_updated = int8_parameter.clone();
    int8_config.step(2, &mut int8_updated, &int8_gradient, &mut int8_state);
    let int8_input_m_q: Vec<u8> = int8_input_state
        .m_q
        .iter()
        .map(|&value| value as u8)
        .collect();
    let int8_output_m_q: Vec<u8> = int8_state.m_q.iter().map(|&value| value as u8).collect();

    let muon_config = Muon {
        lr: 0.02,
        momentum: 0.9,
        weight_decay: 0.01,
        rows: 2,
        cols: 3,
        ns_steps: 3,
    };
    let muon_zero_steps = Muon {
        ns_steps: 0,
        ..muon_config
    };
    let muon_seed_parameter = [0.5_f32, -0.25, 1.0, -1.0, 0.75, 0.125];
    let muon_seed_gradient = [0.2_f32, -0.4, 0.1, 0.3, -0.2, 0.5];
    let muon_gradient = [-0.1_f32, 0.25, 0.0, -0.2, 0.4, -0.3];
    let mut muon_parameter = muon_seed_parameter;
    let mut muon_state = muon_config.init_state(muon_parameter.len());
    muon_config.step(1, &mut muon_parameter, &muon_seed_gradient, &mut muon_state);
    let muon_input_momentum = muon_state.momentum.clone();
    let mut muon_updated = muon_parameter;
    muon_config.step(2, &mut muon_updated, &muon_gradient, &mut muon_state);

    let checkpoint = Checkpoint {
        step: 7,
        leaves: vec![
            LeafCheckpoint {
                param: vec![1.0_f32, -2.0, 3.0],
                state: AdamState {
                    m: vec![0.1, -0.2, 0.3],
                    v: vec![0.01, 0.04, 0.09],
                },
            },
            LeafCheckpoint {
                param: vec![4.0_f32, -5.0],
                state: AdamState {
                    m: vec![0.4, -0.5],
                    v: vec![0.16, 0.25],
                },
            },
        ],
    };
    let checkpoint_bytes = write_checkpoint(&adam_config, &checkpoint);
    let mut corrupt_checkpoint = checkpoint_bytes.clone();
    corrupt_checkpoint[0] ^= u8::MAX;
    let mut invalid_state_checkpoint = checkpoint.clone();
    invalid_state_checkpoint.leaves[0].state.v[1] = -0.04;
    let invalid_state_checkpoint_bytes = write_checkpoint(&adam_config, &invalid_state_checkpoint);

    let hard_plane = SaltV2Plane::new(vec![-1, 0, 1, 1, 0, -1], vec![f16::from_f32(0.5)])
        .expect("valid hard plane");
    let hard_tile = SaltV2Tile::new(vec![hard_plane]).expect("valid hard tile");
    let hard_tensor =
        SaltV2Tensor::new("model.weight", vec![2, 3], vec![hard_tile]).expect("valid hard tensor");
    let hard_package =
        SaltV2Package::new(SaltV2Codec::D2, vec![hard_tensor]).expect("valid hard package");
    let hard_artifact = write_salt_v2_package(&hard_package)
        .expect("encode hard package")
        .bytes;
    let mut corrupt_artifact = hard_artifact.clone();
    corrupt_artifact[0] ^= u8::MAX;

    let corpus = Corpus {
        schema_id: TrainingVectorSetV1::SCHEMA_ID,
        schema_version: TrainingVectorSetV1::SCHEMA_VERSION,
        manifest_digest: hex(&TrainingOpManifestV1::digest()),
        cases: vec![
            Case {
                case_id: "graph.add.forward.basic",
                operation: "graph.add",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("left", &[3], &left),
                    f32_buffer("right", &[3], &right),
                ],
                attributes: vec![],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[3], &add)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.add.forward.zero",
                operation: "graph.add",
                execution: "forward",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-4_f32.to_bits(),
                    relative_bits: 1.0e-4_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("left", &[1], &[0.0]),
                    f32_buffer("right", &[1], &[0.0]),
                ],
                attributes: vec![],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[1], &[0.0])],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.add.vjp.basic",
                operation: "graph.add",
                execution: "vjp",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("grad_output", &[3], &grad_output)],
                attributes: vec![],
                expected: Expected::Success {
                    outputs: vec![
                        f32_buffer("grad_left", &[3], &grad_output),
                        f32_buffer("grad_right", &[3], &grad_output),
                    ],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.ste_surrogate.forward.basic",
                operation: "graph.ste_surrogate",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("weight", &[2, 3], &quant_weight),
                    f32_buffer("scale", &[2], &quant_scale),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[2, 3], &ste_result)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.ste_surrogate.vjp.basic",
                operation: "graph.ste_surrogate",
                execution: "vjp",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("weight", &[2, 3], &quant_weight),
                    f32_buffer("scale", &[2], &quant_scale),
                    f32_buffer("grad_output", &[2, 3], &quant_grad_output),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![
                        f32_buffer("grad_weight", &[2, 3], &ste_grads[0]),
                        f32_buffer("grad_scale", &[2], &ste_grads[1]),
                    ],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.salt_ste.forward.two_planes",
                operation: "graph.salt_ste",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("weight", &[2, 3], &quant_weight)],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                    Attribute::U64 {
                        name: "planes",
                        value: 2,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[2, 3], &salt_result)],
                    scratch_bytes_max: 12,
                },
            },
            Case {
                case_id: "graph.salt_ste.vjp.identity",
                operation: "graph.salt_ste",
                execution: "vjp",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("weight", &[2, 3], &quant_weight),
                    f32_buffer("grad_output", &[2, 3], &quant_grad_output),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                    Attribute::U64 {
                        name: "planes",
                        value: 2,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("grad_weight", &[2, 3], &salt_grad)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.lsq_ste.forward.basic",
                operation: "graph.lsq_ste",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("weight", &[2, 3], &quant_weight),
                    f32_buffer("alpha", &[2], &learned_scale),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[2, 3], &lsq_result)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.lsq_ste.vjp.basic",
                operation: "graph.lsq_ste",
                execution: "vjp",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-6_f32.to_bits(),
                    relative_bits: 1.0e-6_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("weight", &[2, 3], &quant_weight),
                    f32_buffer("alpha", &[2], &learned_scale),
                    f32_buffer("grad_output", &[2, 3], &quant_grad_output),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![
                        f32_buffer("grad_weight", &[2, 3], &lsq_grads[0]),
                        f32_buffer("grad_alpha", &[2], &lsq_grads[1]),
                    ],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.fsq.forward.soft_round",
                operation: "graph.fsq",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("x", &[2, 3], &fsq_input)],
                attributes: vec![
                    Attribute::U64 {
                        name: "channels",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "len",
                        value: 3,
                    },
                    Attribute::U32List {
                        name: "levels",
                        values: vec![3, 5],
                    },
                    Attribute::Text {
                        name: "bound",
                        value: "clamp",
                    },
                    Attribute::Text {
                        name: "ste",
                        value: "soft_round",
                    },
                    Attribute::F32 {
                        name: "alpha",
                        bits: 0.5_f32.to_bits(),
                    },
                    Attribute::U64 {
                        name: "seed",
                        value: 0,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[2, 3], &fsq_result)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.fsq.vjp.soft_round",
                operation: "graph.fsq",
                execution: "vjp",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-5_f32.to_bits(),
                    relative_bits: 1.0e-5_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("x", &[2, 3], &fsq_input),
                    f32_buffer("grad_output", &[2, 3], &quant_grad_output),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "channels",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "len",
                        value: 3,
                    },
                    Attribute::U32List {
                        name: "levels",
                        values: vec![3, 5],
                    },
                    Attribute::Text {
                        name: "bound",
                        value: "clamp",
                    },
                    Attribute::Text {
                        name: "ste",
                        value: "soft_round",
                    },
                    Attribute::F32 {
                        name: "alpha",
                        bits: 0.5_f32.to_bits(),
                    },
                    Attribute::U64 {
                        name: "seed",
                        value: 0,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("grad_x", &[2, 3], &fsq_grads[0])],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.fsq.forward.hard_half_ties",
                operation: "graph.fsq",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("x", &[1, 4], &fsq_boundary_input)],
                attributes: fsq_attributes(1, 4, &[3], "clamp", "hard", 0.0, 0),
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[1, 4], &fsq_hard_result)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.fsq.vjp.hard_tanh",
                operation: "graph.fsq",
                execution: "vjp",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-6_f32.to_bits(),
                    relative_bits: 1.0e-6_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("x", &[1, 4], &fsq_boundary_input),
                    f32_buffer("grad_output", &[1, 4], &quant_grad_output[..4]),
                ],
                attributes: fsq_attributes(1, 4, &[5], "tanh", "hard", 0.0, 0),
                expected: Expected::Success {
                    outputs: vec![f32_buffer("grad_x", &[1, 4], &fsq_tanh_grad[0])],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.fsq.forward.stochastic_seed_7",
                operation: "graph.fsq",
                execution: "forward",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-6_f32.to_bits(),
                    relative_bits: 1.0e-6_f32.to_bits(),
                },
                inputs: vec![f32_buffer("x", &[1, 4], &fsq_stochastic_input)],
                attributes: fsq_attributes(1, 4, &[5], "tanh", "stochastic", 0.0, 7),
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[1, 4], &fsq_stochastic_seed_7)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.fsq.forward.stochastic_seed_8",
                operation: "graph.fsq",
                execution: "forward",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-6_f32.to_bits(),
                    relative_bits: 1.0e-6_f32.to_bits(),
                },
                inputs: vec![f32_buffer("x", &[1, 4], &fsq_stochastic_input)],
                attributes: fsq_attributes(1, 4, &[5], "tanh", "stochastic", 0.0, 8),
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[1, 4], &fsq_stochastic_seed_8)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.dense_matmul.forward.basic",
                operation: "graph.dense_matmul",
                execution: "forward",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-5_f32.to_bits(),
                    relative_bits: 1.0e-5_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("x", &[2, 3], &matmul_x),
                    f32_buffer("weight", &[2, 3], &dense_weight),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "m",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "n",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "k",
                        value: 3,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[2, 2], &dense_result)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.dense_matmul.vjp.basic",
                operation: "graph.dense_matmul",
                execution: "vjp",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-5_f32.to_bits(),
                    relative_bits: 1.0e-5_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("x", &[2, 3], &matmul_x),
                    f32_buffer("weight", &[2, 3], &dense_weight),
                    f32_buffer("grad_output", &[2, 2], &matmul_grad_output),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "m",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "n",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "k",
                        value: 3,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![
                        f32_buffer("grad_x", &[2, 3], &dense_grads[0]),
                        f32_buffer("grad_weight", &[2, 3], &dense_grads[1]),
                    ],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.ternary_matmul.forward.basic",
                operation: "graph.ternary_matmul",
                execution: "forward",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-5_f32.to_bits(),
                    relative_bits: 1.0e-5_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("activation", &[2, 3], &matmul_x),
                    f32_buffer("weight", &[2, 3], &ternary_weight),
                    f32_buffer("scale", &[2], &ternary_scale),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "m",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "n",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "k",
                        value: 3,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[2, 2], &ternary_result)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.ternary_matmul.vjp.basic",
                operation: "graph.ternary_matmul",
                execution: "vjp",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-5_f32.to_bits(),
                    relative_bits: 1.0e-5_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("activation", &[2, 3], &matmul_x),
                    f32_buffer("weight", &[2, 3], &ternary_weight),
                    f32_buffer("scale", &[2], &ternary_scale),
                    f32_buffer("grad_output", &[2, 2], &matmul_grad_output),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "m",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "n",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "k",
                        value: 3,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![
                        f32_buffer("grad_activation", &[2, 3], &ternary_grads[0]),
                        f32_buffer("grad_weight", &[2, 3], &ternary_grads[1]),
                        f32_buffer("grad_scale", &[2], &ternary_grads[2]),
                    ],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.transpose.forward.basic",
                operation: "graph.transpose",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("x", &[2, 3], &matrix)],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[3, 2], &transpose)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.transpose.vjp.basic",
                operation: "graph.transpose",
                execution: "vjp",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("grad_output", &[3, 2], &transpose_grad_output)],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("grad_x", &[2, 3], &transpose_grad[0])],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.embedding_gather.forward.repeated",
                operation: "graph.embedding_gather",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("weight", &[4, 2], &embedding_weight),
                    u32_buffer("tokens", &[3], &tokens),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "vocab",
                        value: 4,
                    },
                    Attribute::U64 {
                        name: "n_embd",
                        value: 2,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[3, 2], &embedding)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.embedding_gather.vjp.repeated",
                operation: "graph.embedding_gather",
                execution: "vjp",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("weight", &[4, 2], &embedding_weight),
                    u32_buffer("tokens", &[3], &tokens),
                    f32_buffer("grad_output", &[3, 2], &embedding_grad_output),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "vocab",
                        value: 4,
                    },
                    Attribute::U64 {
                        name: "n_embd",
                        value: 2,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("grad_weight", &[4, 2], &embedding_grad)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.slice_cols.forward.basic",
                operation: "graph.slice_cols",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("x", &[2, 4], &column_matrix)],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 4,
                    },
                    Attribute::U64 {
                        name: "start",
                        value: 1,
                    },
                    Attribute::U64 {
                        name: "len",
                        value: 2,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[2, 2], &column_slice)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.slice_cols.vjp.basic",
                operation: "graph.slice_cols",
                execution: "vjp",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("grad_output", &[2, 2], &slice_grad_output)],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 4,
                    },
                    Attribute::U64 {
                        name: "start",
                        value: 1,
                    },
                    Attribute::U64 {
                        name: "len",
                        value: 2,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("grad_x", &[2, 4], &slice_grad)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.concat_cols.forward.basic",
                operation: "graph.concat_cols",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("part.0", &[2, 2], &concat_left),
                    f32_buffer("part.1", &[2, 1], &concat_right),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64List {
                        name: "lens",
                        values: vec![2, 1],
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[2, 3], &concatenated)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.concat_cols.vjp.basic",
                operation: "graph.concat_cols",
                execution: "vjp",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("grad_output", &[2, 3], &concat_grad_output)],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64List {
                        name: "lens",
                        values: vec![2, 1],
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![
                        f32_buffer("grad_part.0", &[2, 2], &concat_grads[0]),
                        f32_buffer("grad_part.1", &[2, 1], &concat_grads[1]),
                    ],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.mul.forward.basic",
                operation: "graph.mul",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("left", &[3], &mul_left),
                    f32_buffer("right", &[3], &mul_right),
                ],
                attributes: vec![],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[3], &mul)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.mul.vjp.basic",
                operation: "graph.mul",
                execution: "vjp",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("left", &[3], &mul_left),
                    f32_buffer("right", &[3], &mul_right),
                    f32_buffer("grad_output", &[3], &mul_grad_output),
                ],
                attributes: vec![],
                expected: Expected::Success {
                    outputs: vec![
                        f32_buffer("grad_left", &[3], &mul_grads[0]),
                        f32_buffer("grad_right", &[3], &mul_grads[1]),
                    ],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.conv1d.forward.depthwise_asymmetric",
                operation: "graph.conv1d",
                execution: "forward",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-6_f32.to_bits(),
                    relative_bits: 1.0e-6_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("x", &[1, 2, 4], &conv1d_x),
                    f32_buffer("weight", &[2, 1, 2], &conv1d_weight),
                    f32_buffer("scale", &[2], &conv1d_scale),
                ],
                attributes: conv1d_attributes(&conv1d_cfg),
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[1, 2, 4], &conv1d_result)],
                    scratch_bytes_max: 80,
                },
            },
            Case {
                case_id: "graph.conv1d.vjp.depthwise_asymmetric",
                operation: "graph.conv1d",
                execution: "vjp",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-6_f32.to_bits(),
                    relative_bits: 1.0e-6_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("x", &[1, 2, 4], &conv1d_x),
                    f32_buffer("weight", &[2, 1, 2], &conv1d_weight),
                    f32_buffer("scale", &[2], &conv1d_scale),
                    f32_buffer("grad_output", &[1, 2, 4], &conv1d_grad_output),
                ],
                attributes: conv1d_attributes(&conv1d_cfg),
                expected: Expected::Success {
                    outputs: vec![
                        f32_buffer("grad_x", &[1, 2, 4], &conv1d_grads[0]),
                        f32_buffer("grad_weight", &[2, 1, 2], &conv1d_grads[1]),
                        f32_buffer("grad_scale", &[2], &conv1d_grads[2]),
                    ],
                    scratch_bytes_max: 148,
                },
            },
            Case {
                case_id: "graph.conv2d.forward.depthwise_asymmetric",
                operation: "graph.conv2d",
                execution: "forward",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-6_f32.to_bits(),
                    relative_bits: 1.0e-6_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("x", &[1, 2, 3, 4], &conv2d_x),
                    f32_buffer("weight", &[2, 1, 2, 2], &conv2d_weight),
                    f32_buffer("scale", &[2], &conv2d_scale),
                ],
                attributes: conv2d_attributes(&conv2d_cfg),
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[1, 2, 3, 2], &conv2d_result)],
                    scratch_bytes_max: 168,
                },
            },
            Case {
                case_id: "graph.conv2d.vjp.depthwise_asymmetric",
                operation: "graph.conv2d",
                execution: "vjp",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-6_f32.to_bits(),
                    relative_bits: 1.0e-6_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("x", &[1, 2, 3, 4], &conv2d_x),
                    f32_buffer("weight", &[2, 1, 2, 2], &conv2d_weight),
                    f32_buffer("scale", &[2], &conv2d_scale),
                    f32_buffer("grad_output", &[1, 2, 3, 2], &conv2d_grad_output),
                ],
                attributes: conv2d_attributes(&conv2d_cfg),
                expected: Expected::Success {
                    outputs: vec![
                        f32_buffer("grad_x", &[1, 2, 3, 4], &conv2d_grads[0]),
                        f32_buffer("grad_weight", &[2, 1, 2, 2], &conv2d_grads[1]),
                        f32_buffer("grad_scale", &[2], &conv2d_grads[2]),
                    ],
                    scratch_bytes_max: 372,
                },
            },
            Case {
                case_id: "graph.detach.forward.basic",
                operation: "graph.detach",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("x", &[3], &unary_input)],
                attributes: vec![],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[3], &unary_input)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.detach.vjp.zero",
                operation: "graph.detach",
                execution: "vjp",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("grad_output", &[3], &unary_grad_output)],
                attributes: vec![],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("grad_x", &[3], &[0.0; 3])],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.scale_const.forward.basic",
                operation: "graph.scale_const",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("x", &[3], &unary_input)],
                attributes: vec![Attribute::F32 {
                    name: "scale",
                    bits: scale.to_bits(),
                }],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[3], &scaled)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.scale_const.vjp.basic",
                operation: "graph.scale_const",
                execution: "vjp",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("grad_output", &[3], &unary_grad_output)],
                attributes: vec![Attribute::F32 {
                    name: "scale",
                    bits: scale.to_bits(),
                }],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("grad_x", &[3], &scaled_grad)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.bias.forward.basic",
                operation: "graph.bias",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("x", &[2, 3], &bias_input),
                    f32_buffer("bias", &[3], &bias_value),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[2, 3], &bias_result)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.bias.vjp.basic",
                operation: "graph.bias",
                execution: "vjp",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("x", &[2, 3], &bias_input),
                    f32_buffer("bias", &[3], &bias_value),
                    f32_buffer("grad_output", &[2, 3], &bias_grad_output),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![
                        f32_buffer("grad_x", &[2, 3], &bias_grads[0]),
                        f32_buffer("grad_bias", &[3], &bias_grads[1]),
                    ],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.relu2.forward.basic",
                operation: "graph.relu2",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("x", &[3], &unary_input)],
                attributes: vec![],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[3], &relu2)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.relu2.vjp.basic",
                operation: "graph.relu2",
                execution: "vjp",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("x", &[3], &unary_input),
                    f32_buffer("grad_output", &[3], &unary_grad_output),
                ],
                attributes: vec![],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("grad_x", &[3], &relu2_grad[0])],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.silu.forward.basic",
                operation: "graph.silu",
                execution: "forward",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-5_f32.to_bits(),
                    relative_bits: 1.0e-5_f32.to_bits(),
                },
                inputs: vec![f32_buffer("x", &[3], &silu_input)],
                attributes: vec![],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[3], &silu)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.silu.vjp.basic",
                operation: "graph.silu",
                execution: "vjp",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-5_f32.to_bits(),
                    relative_bits: 1.0e-5_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("x", &[3], &silu_input),
                    f32_buffer("grad_output", &[3], &silu_grad_output),
                ],
                attributes: vec![],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("grad_x", &[3], &silu_grad[0])],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.rmsnorm.forward.basic",
                operation: "graph.rmsnorm",
                execution: "forward",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-5_f32.to_bits(),
                    relative_bits: 1.0e-5_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("x", &[2, 3], &normalized_input),
                    f32_buffer("weight", &[3], &norm_weight),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                    Attribute::F32 {
                        name: "eps",
                        bits: norm_epsilon.to_bits(),
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[2, 3], &rmsnorm)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.rmsnorm.vjp.basic",
                operation: "graph.rmsnorm",
                execution: "vjp",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-5_f32.to_bits(),
                    relative_bits: 1.0e-5_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("x", &[2, 3], &normalized_input),
                    f32_buffer("weight", &[3], &norm_weight),
                    f32_buffer("grad_output", &[2, 3], &transformer_grad_output),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                    Attribute::F32 {
                        name: "eps",
                        bits: norm_epsilon.to_bits(),
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![
                        f32_buffer("grad_x", &[2, 3], &rmsnorm_grads[0]),
                        f32_buffer("grad_weight", &[3], &rmsnorm_grads[1]),
                    ],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.softmax.forward.basic",
                operation: "graph.softmax",
                execution: "forward",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-6_f32.to_bits(),
                    relative_bits: 1.0e-6_f32.to_bits(),
                },
                inputs: vec![f32_buffer("x", &[2, 3], &logits)],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[2, 3], &probabilities)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.softmax.vjp.basic",
                operation: "graph.softmax",
                execution: "vjp",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-6_f32.to_bits(),
                    relative_bits: 1.0e-6_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("x", &[2, 3], &logits),
                    f32_buffer("grad_output", &[2, 3], &transformer_grad_output),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("grad_x", &[2, 3], &softmax_grad[0])],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.causal_mask.forward.basic",
                operation: "graph.causal_mask",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("x", &[2, 3], &logits)],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[2, 3], &causal)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.causal_mask.vjp.basic",
                operation: "graph.causal_mask",
                execution: "vjp",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("grad_output", &[2, 3], &transformer_grad_output)],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("grad_x", &[2, 3], &causal_grad[0])],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.rope.forward.basic",
                operation: "graph.rope",
                execution: "forward",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-5_f32.to_bits(),
                    relative_bits: 1.0e-5_f32.to_bits(),
                },
                inputs: vec![f32_buffer("x", &[2, 1, 4], &rope_input)],
                attributes: vec![
                    Attribute::U64List {
                        name: "positions",
                        values: vec![0, 3],
                    },
                    Attribute::U64 {
                        name: "n_head",
                        value: 1,
                    },
                    Attribute::U64 {
                        name: "head_dim",
                        value: 4,
                    },
                    Attribute::F32 {
                        name: "theta",
                        bits: rope_theta.to_bits(),
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[2, 1, 4], &rope_result)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.rope.vjp.basic",
                operation: "graph.rope",
                execution: "vjp",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-5_f32.to_bits(),
                    relative_bits: 1.0e-5_f32.to_bits(),
                },
                inputs: vec![f32_buffer("grad_output", &[2, 1, 4], &rope_grad_output)],
                attributes: vec![
                    Attribute::U64List {
                        name: "positions",
                        values: vec![0, 3],
                    },
                    Attribute::U64 {
                        name: "n_head",
                        value: 1,
                    },
                    Attribute::U64 {
                        name: "head_dim",
                        value: 4,
                    },
                    Attribute::F32 {
                        name: "theta",
                        bits: rope_theta.to_bits(),
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("grad_x", &[2, 1, 4], &rope_grad[0])],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.attention.forward.causal_gqa",
                operation: "graph.attention",
                execution: "forward",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-6_f32.to_bits(),
                    relative_bits: 1.0e-6_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("q", &[3, 2, 2], &attention_q),
                    f32_buffer("k", &[3, 1, 2], &attention_k),
                    f32_buffer("v", &[3, 1, 2], &attention_v),
                ],
                attributes: attention_attributes(&attention_cfg),
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[3, 2, 2], &attention_result)],
                    scratch_bytes_max: 84,
                },
            },
            Case {
                case_id: "graph.attention.forward.noncausal_gqa",
                operation: "graph.attention",
                execution: "forward",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-6_f32.to_bits(),
                    relative_bits: 1.0e-6_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("q", &[3, 2, 2], &attention_q),
                    f32_buffer("k", &[3, 1, 2], &attention_k),
                    f32_buffer("v", &[3, 1, 2], &attention_v),
                ],
                attributes: attention_attributes(&attention_noncausal_cfg),
                expected: Expected::Success {
                    outputs: vec![f32_buffer(
                        "result",
                        &[3, 2, 2],
                        &attention_noncausal_result,
                    )],
                    scratch_bytes_max: 84,
                },
            },
            Case {
                case_id: "graph.attention.vjp.causal_gqa",
                operation: "graph.attention",
                execution: "vjp",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-5_f32.to_bits(),
                    relative_bits: 1.0e-5_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("q", &[3, 2, 2], &attention_q),
                    f32_buffer("k", &[3, 1, 2], &attention_k),
                    f32_buffer("v", &[3, 1, 2], &attention_v),
                    f32_buffer("grad_output", &[3, 2, 2], &attention_grad_output),
                ],
                attributes: attention_attributes(&attention_cfg),
                expected: Expected::Success {
                    outputs: vec![
                        f32_buffer("grad_q", &[3, 2, 2], &attention_grads[0]),
                        f32_buffer("grad_k", &[3, 1, 2], &attention_grads[1]),
                        f32_buffer("grad_v", &[3, 1, 2], &attention_grads[2]),
                    ],
                    scratch_bytes_max: 168,
                },
            },
            Case {
                case_id: "graph.attention.vjp.noncausal_mqa",
                operation: "graph.attention",
                execution: "vjp",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-5_f32.to_bits(),
                    relative_bits: 1.0e-5_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("q", &[3, 2, 2], &attention_q),
                    f32_buffer("k", &[3, 1, 2], &attention_k),
                    f32_buffer("v", &[3, 1, 2], &attention_v),
                    f32_buffer("grad_output", &[3, 2, 2], &attention_grad_output),
                ],
                attributes: attention_attributes(&attention_noncausal_cfg),
                expected: Expected::Success {
                    outputs: vec![
                        f32_buffer("grad_q", &[3, 2, 2], &attention_noncausal_grads[0]),
                        f32_buffer("grad_k", &[3, 1, 2], &attention_noncausal_grads[1]),
                        f32_buffer("grad_v", &[3, 1, 2], &attention_noncausal_grads[2]),
                    ],
                    scratch_bytes_max: 168,
                },
            },
            Case {
                case_id: "graph.attention.forward.multigroup_gqa",
                operation: "graph.attention",
                execution: "forward",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-6_f32.to_bits(),
                    relative_bits: 1.0e-6_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("q", &[2, 4, 1], &attention_gqa_q),
                    f32_buffer("k", &[2, 2, 1], &attention_gqa_k),
                    f32_buffer("v", &[2, 2, 1], &attention_gqa_v),
                ],
                attributes: attention_attributes(&attention_gqa_cfg),
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[2, 4, 1], &attention_gqa_result)],
                    scratch_bytes_max: 48,
                },
            },
            Case {
                case_id: "graph.attention.vjp.multigroup_gqa",
                operation: "graph.attention",
                execution: "vjp",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-5_f32.to_bits(),
                    relative_bits: 1.0e-5_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("q", &[2, 4, 1], &attention_gqa_q),
                    f32_buffer("k", &[2, 2, 1], &attention_gqa_k),
                    f32_buffer("v", &[2, 2, 1], &attention_gqa_v),
                    f32_buffer("grad_output", &[2, 4, 1], &attention_gqa_grad_output),
                ],
                attributes: attention_attributes(&attention_gqa_cfg),
                expected: Expected::Success {
                    outputs: vec![
                        f32_buffer("grad_q", &[2, 4, 1], &attention_gqa_grads[0]),
                        f32_buffer("grad_k", &[2, 2, 1], &attention_gqa_grads[1]),
                        f32_buffer("grad_v", &[2, 2, 1], &attention_gqa_grads[2]),
                    ],
                    scratch_bytes_max: 96,
                },
            },
            Case {
                case_id: "loss.mse.forward.basic",
                operation: "loss.mse",
                execution: "forward",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-6_f32.to_bits(),
                    relative_bits: 1.0e-6_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("prediction", &[3], &prediction),
                    f32_buffer("target", &[3], &target),
                ],
                attributes: vec![],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[], &mse)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "loss.mse.vjp.basic",
                operation: "loss.mse",
                execution: "vjp",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-6_f32.to_bits(),
                    relative_bits: 1.0e-6_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("prediction", &[3], &prediction),
                    f32_buffer("target", &[3], &target),
                    f32_buffer("grad_output", &[], &loss_grad_output),
                ],
                attributes: vec![],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("grad_prediction", &[3], &mse_grad[0])],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "loss.softmax_cross_entropy.forward.basic",
                operation: "loss.softmax_cross_entropy",
                execution: "forward",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-6_f32.to_bits(),
                    relative_bits: 1.0e-6_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("logits", &[2, 3], &logits),
                    f32_buffer("target", &[2, 3], &class_target),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[], &xent)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "loss.softmax_cross_entropy.vjp.basic",
                operation: "loss.softmax_cross_entropy",
                execution: "vjp",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-6_f32.to_bits(),
                    relative_bits: 1.0e-6_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("logits", &[2, 3], &logits),
                    f32_buffer("target", &[2, 3], &class_target),
                    f32_buffer("grad_output", &[], &xent_grad_output),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("grad_logits", &[2, 3], &xent_grad[0])],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "optimizer.sgd.step.basic",
                operation: "optimizer.sgd",
                execution: "step",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("parameter", &[2], &parameter),
                    f32_buffer("gradient", &[2], &gradient),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "step",
                        value: 1,
                    },
                    Attribute::F32 {
                        name: "lr",
                        bits: 0.1_f32.to_bits(),
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("parameter", &[2], &updated)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "optimizer.adamw.step.resumed_state",
                operation: "optimizer.adamw",
                execution: "step",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("parameter", &[4], &adam_parameter),
                    f32_buffer("gradient", &[4], &adam_gradient),
                    f32_buffer("moment1", &[4], &adam_input_state.m),
                    f32_buffer("moment2", &[4], &adam_input_state.v),
                ],
                attributes: adam_attributes(2, &adam_config),
                expected: Expected::Success {
                    outputs: vec![
                        f32_buffer("parameter", &[4], &adam_updated),
                        f32_buffer("moment1", &[4], &adam_state.m),
                        f32_buffer("moment2", &[4], &adam_state.v),
                    ],
                    scratch_bytes_max: 32,
                },
            },
            Case {
                case_id: "optimizer.cautious_adamw.step.masked_state",
                operation: "optimizer.cautious_adamw",
                execution: "step",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("parameter", &[4], &cautious_parameter),
                    f32_buffer("gradient", &[4], &cautious_gradient),
                    f32_buffer("moment1", &[4], &cautious_input_state.m),
                    f32_buffer("moment2", &[4], &cautious_input_state.v),
                ],
                attributes: adam_attributes(2, &adam_config),
                expected: Expected::Success {
                    outputs: vec![
                        f32_buffer("parameter", &[4], &cautious_updated),
                        f32_buffer("moment1", &[4], &cautious_state.m),
                        f32_buffer("moment2", &[4], &cautious_state.v),
                    ],
                    scratch_bytes_max: 48,
                },
            },
            Case {
                case_id: "optimizer.int8_adamw.step.quiet_spike_blocks",
                operation: "optimizer.int8_adamw",
                execution: "step",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("parameter", &[260], &int8_parameter),
                    f32_buffer("gradient", &[260], &int8_gradient),
                    bytes_buffer("moment1_q8", &[260], &int8_input_m_q),
                    bytes_buffer("moment2_q8", &[260], &int8_input_state.v_q),
                    f32_buffer("moment1_scale", &[2], &int8_input_state.m_scale),
                    f32_buffer("moment2_scale", &[2], &int8_input_state.v_scale),
                ],
                attributes: adam_attributes(2, &adam_config),
                expected: Expected::Success {
                    outputs: vec![
                        f32_buffer("parameter", &[260], &int8_updated),
                        bytes_buffer("moment1_q8", &[260], &int8_output_m_q),
                        bytes_buffer("moment2_q8", &[260], &int8_state.v_q),
                        f32_buffer("moment1_scale", &[2], &int8_state.m_scale),
                        f32_buffer("moment2_scale", &[2], &int8_state.v_scale),
                    ],
                    scratch_bytes_max: 2584,
                },
            },
            Case {
                case_id: "optimizer.muon.step.resumed_rectangular",
                operation: "optimizer.muon",
                execution: "step",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("parameter", &[2, 3], &muon_parameter),
                    f32_buffer("gradient", &[2, 3], &muon_gradient),
                    f32_buffer("momentum", &[2, 3], &muon_input_momentum),
                ],
                attributes: muon_attributes(2, &muon_config),
                expected: Expected::Success {
                    outputs: vec![
                        f32_buffer("parameter", &[2, 3], &muon_updated),
                        f32_buffer("momentum", &[2, 3], &muon_state.momentum),
                    ],
                    scratch_bytes_max: 144,
                },
            },
            Case {
                case_id: "lifecycle.checkpoint.adamw_multileaf",
                operation: "lifecycle.checkpoint",
                execution: "checkpoint",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("parameter.0", &[3], &checkpoint.leaves[0].param),
                    f32_buffer("moment1.0", &[3], &checkpoint.leaves[0].state.m),
                    f32_buffer("moment2.0", &[3], &checkpoint.leaves[0].state.v),
                    f32_buffer("parameter.1", &[2], &checkpoint.leaves[1].param),
                    f32_buffer("moment1.1", &[2], &checkpoint.leaves[1].state.m),
                    f32_buffer("moment2.1", &[2], &checkpoint.leaves[1].state.v),
                ],
                attributes: checkpoint_attributes("adamw", checkpoint.step, &[3, 2]),
                expected: Expected::Success {
                    outputs: vec![bytes_buffer(
                        "checkpoint",
                        &[checkpoint_bytes.len() as u64],
                        &checkpoint_bytes,
                    )],
                    scratch_bytes_max: checkpoint_bytes.len() as u64 + 60,
                },
            },
            Case {
                case_id: "lifecycle.resume.adamw_multileaf",
                operation: "lifecycle.resume",
                execution: "resume",
                tolerance: Tolerance::BitExact,
                inputs: vec![bytes_buffer(
                    "checkpoint",
                    &[checkpoint_bytes.len() as u64],
                    &checkpoint_bytes,
                )],
                attributes: resume_attributes("adamw", &[3, 2]),
                expected: Expected::Success {
                    outputs: vec![
                        bytes_buffer("step", &[8], &checkpoint.step.to_le_bytes()),
                        f32_buffer("parameter.0", &[3], &checkpoint.leaves[0].param),
                        f32_buffer("moment1.0", &[3], &checkpoint.leaves[0].state.m),
                        f32_buffer("moment2.0", &[3], &checkpoint.leaves[0].state.v),
                        f32_buffer("parameter.1", &[2], &checkpoint.leaves[1].param),
                        f32_buffer("moment1.1", &[2], &checkpoint.leaves[1].state.m),
                        f32_buffer("moment2.1", &[2], &checkpoint.leaves[1].state.v),
                    ],
                    scratch_bytes_max: 60,
                },
            },
            Case {
                case_id: "lifecycle.export.salt_v2_package",
                operation: "lifecycle.export",
                execution: "export",
                tolerance: Tolerance::BitExact,
                inputs: vec![bytes_buffer(
                    "package",
                    &[hard_artifact.len() as u64],
                    &hard_artifact,
                )],
                attributes: artifact_attributes(),
                expected: Expected::Success {
                    outputs: vec![bytes_buffer(
                        "artifact",
                        &[hard_artifact.len() as u64],
                        &hard_artifact,
                    )],
                    scratch_bytes_max: (hard_artifact.len() * 8 + 128 * 1024) as u64,
                },
            },
            Case {
                case_id: "lifecycle.reload.salt_v2_package",
                operation: "lifecycle.reload",
                execution: "reload",
                tolerance: Tolerance::BitExact,
                inputs: vec![bytes_buffer(
                    "artifact",
                    &[hard_artifact.len() as u64],
                    &hard_artifact,
                )],
                attributes: artifact_attributes(),
                expected: Expected::Success {
                    outputs: vec![bytes_buffer(
                        "package",
                        &[hard_artifact.len() as u64],
                        &hard_artifact,
                    )],
                    scratch_bytes_max: (hard_artifact.len() * 8 + 128 * 1024) as u64,
                },
            },
            Case {
                case_id: "graph.add.forward.nonfinite",
                operation: "graph.add",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("left", &[1], &[f32::NAN]),
                    f32_buffer("right", &[1], &[1.0]),
                ],
                attributes: vec![],
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "non_finite.left",
                    outputs: vec![f32_buffer("result", &[1], &[123.0])],
                },
            },
            Case {
                case_id: "graph.add.forward.duplicate_input",
                operation: "graph.add",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("left", &[1], &[1.0]),
                    f32_buffer("left", &[1], &[2.0]),
                ],
                attributes: vec![],
                expected: Expected::Error {
                    category: "invalid_request",
                    code: "duplicate_name.input.left",
                    outputs: vec![f32_buffer("result", &[1], &[456.0])],
                },
            },
            Case {
                case_id: "graph.transpose.forward.shape_error",
                operation: "graph.transpose",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("x", &[1, 6], &matrix)],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                ],
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "shape",
                    outputs: vec![f32_buffer("result", &[3, 2], &[123.0; 6])],
                },
            },
            Case {
                case_id: "graph.embedding_gather.forward.token_oob",
                operation: "graph.embedding_gather",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("weight", &[4, 2], &embedding_weight),
                    u32_buffer("tokens", &[1], &[4]),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "vocab",
                        value: 4,
                    },
                    Attribute::U64 {
                        name: "n_embd",
                        value: 2,
                    },
                ],
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "shape",
                    outputs: vec![f32_buffer("result", &[1, 2], &[123.0; 2])],
                },
            },
            Case {
                case_id: "graph.slice_cols.forward.bounds_error",
                operation: "graph.slice_cols",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("x", &[2, 4], &column_matrix)],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 4,
                    },
                    Attribute::U64 {
                        name: "start",
                        value: 3,
                    },
                    Attribute::U64 {
                        name: "len",
                        value: 2,
                    },
                ],
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.start.slice_bounds",
                    outputs: vec![f32_buffer("result", &[2, 2], &[123.0; 4])],
                },
            },
            Case {
                case_id: "graph.concat_cols.forward.shape_error",
                operation: "graph.concat_cols",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("part.0", &[1, 4], &concat_left),
                    f32_buffer("part.1", &[2, 1], &concat_right),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64List {
                        name: "lens",
                        values: vec![2, 1],
                    },
                ],
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "shape",
                    outputs: vec![f32_buffer("result", &[2, 3], &[123.0; 6])],
                },
            },
            Case {
                case_id: "graph.dense_matmul.forward.shape_error",
                operation: "graph.dense_matmul",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("x", &[1, 6], &matmul_x),
                    f32_buffer("weight", &[2, 3], &dense_weight),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "m",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "n",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "k",
                        value: 3,
                    },
                ],
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "shape",
                    outputs: vec![f32_buffer("result", &[2, 2], &[123.0; 4])],
                },
            },
            Case {
                case_id: "graph.ternary_matmul.forward.nonfinite_scale",
                operation: "graph.ternary_matmul",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("activation", &[2, 3], &matmul_x),
                    f32_buffer("weight", &[2, 3], &ternary_weight),
                    f32_buffer("scale", &[2], &[f32::NAN, 1.5]),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "m",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "n",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "k",
                        value: 3,
                    },
                ],
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "non_finite.scale",
                    outputs: vec![f32_buffer("result", &[2, 2], &[123.0; 4])],
                },
            },
            Case {
                case_id: "graph.rmsnorm.forward.shape_error",
                operation: "graph.rmsnorm",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("x", &[1, 6], &normalized_input),
                    f32_buffer("weight", &[3], &norm_weight),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                    Attribute::F32 {
                        name: "eps",
                        bits: norm_epsilon.to_bits(),
                    },
                ],
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "shape",
                    outputs: vec![f32_buffer("result", &[2, 3], &[123.0; 6])],
                },
            },
            Case {
                case_id: "graph.softmax.forward.shape_error",
                operation: "graph.softmax",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("x", &[1, 6], &logits)],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                ],
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "shape",
                    outputs: vec![f32_buffer("result", &[2, 3], &[123.0; 6])],
                },
            },
            Case {
                case_id: "graph.causal_mask.forward.shape_error",
                operation: "graph.causal_mask",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("x", &[1, 6], &logits)],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                ],
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "shape",
                    outputs: vec![f32_buffer("result", &[2, 3], &[123.0; 6])],
                },
            },
            Case {
                case_id: "loss.softmax_cross_entropy.forward.shape_error",
                operation: "loss.softmax_cross_entropy",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("logits", &[1, 6], &logits),
                    f32_buffer("target", &[2, 3], &class_target),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                ],
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "shape",
                    outputs: vec![f32_buffer("result", &[], &[123.0])],
                },
            },
            Case {
                case_id: "graph.rope.forward.odd_head_dim",
                operation: "graph.rope",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("x", &[2, 1, 3], &rope_input[..6])],
                attributes: vec![
                    Attribute::U64List {
                        name: "positions",
                        values: vec![0, 3],
                    },
                    Attribute::U64 {
                        name: "n_head",
                        value: 1,
                    },
                    Attribute::U64 {
                        name: "head_dim",
                        value: 3,
                    },
                    Attribute::F32 {
                        name: "theta",
                        bits: rope_theta.to_bits(),
                    },
                ],
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.head_dim.even",
                    outputs: vec![f32_buffer("result", &[2, 1, 3], &[123.0; 6])],
                },
            },
            Case {
                case_id: "loss.softmax_cross_entropy.forward.zero_rows",
                operation: "loss.softmax_cross_entropy",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("logits", &[0, 3], &[]),
                    f32_buffer("target", &[0, 3], &[]),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 0,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                ],
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.rows.positive",
                    outputs: vec![f32_buffer("result", &[], &[123.0])],
                },
            },
            Case {
                case_id: "loss.softmax_cross_entropy.forward.zero_cols",
                operation: "loss.softmax_cross_entropy",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("logits", &[1, 0], &[]),
                    f32_buffer("target", &[1, 0], &[]),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 1,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 0,
                    },
                ],
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.cols.positive",
                    outputs: vec![f32_buffer("result", &[], &[123.0])],
                },
            },
            Case {
                case_id: "graph.rope.forward.position_overflow",
                operation: "graph.rope",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("x", &[1, 1, 4], &rope_input[..4])],
                attributes: vec![
                    Attribute::U64List {
                        name: "positions",
                        values: vec![u64::from(u32::MAX) + 1],
                    },
                    Attribute::U64 {
                        name: "n_head",
                        value: 1,
                    },
                    Attribute::U64 {
                        name: "head_dim",
                        value: 4,
                    },
                    Attribute::F32 {
                        name: "theta",
                        bits: rope_theta.to_bits(),
                    },
                ],
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.positions.u32",
                    outputs: vec![f32_buffer("result", &[1, 1, 4], &[123.0; 4])],
                },
            },
            Case {
                case_id: "graph.ste_surrogate.forward.shape_error",
                operation: "graph.ste_surrogate",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("weight", &[1, 6], &quant_weight),
                    f32_buffer("scale", &[2], &quant_scale),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                ],
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "shape",
                    outputs: vec![f32_buffer("result", &[2, 3], &[123.0; 6])],
                },
            },
            Case {
                case_id: "graph.salt_ste.forward.zero_planes",
                operation: "graph.salt_ste",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("weight", &[2, 3], &quant_weight)],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                    Attribute::U64 {
                        name: "planes",
                        value: 0,
                    },
                ],
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.planes.positive",
                    outputs: vec![f32_buffer("result", &[2, 3], &[123.0; 6])],
                },
            },
            Case {
                case_id: "graph.lsq_ste.forward.shape_error",
                operation: "graph.lsq_ste",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("weight", &[1, 6], &quant_weight),
                    f32_buffer("alpha", &[2], &learned_scale),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                ],
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "shape",
                    outputs: vec![f32_buffer("result", &[2, 3], &[123.0; 6])],
                },
            },
            Case {
                case_id: "graph.fsq.forward.invalid_levels",
                operation: "graph.fsq",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("x", &[2, 3], &fsq_input)],
                attributes: vec![
                    Attribute::U64 {
                        name: "channels",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "len",
                        value: 3,
                    },
                    Attribute::U32List {
                        name: "levels",
                        values: vec![1, 5],
                    },
                    Attribute::Text {
                        name: "bound",
                        value: "clamp",
                    },
                    Attribute::Text {
                        name: "ste",
                        value: "soft_round",
                    },
                    Attribute::F32 {
                        name: "alpha",
                        bits: 0.5_f32.to_bits(),
                    },
                    Attribute::U64 {
                        name: "seed",
                        value: 0,
                    },
                ],
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.levels.min_two",
                    outputs: vec![f32_buffer("result", &[2, 3], &[123.0; 6])],
                },
            },
            Case {
                case_id: "graph.salt_ste.forward.zero_rows_huge_cols",
                operation: "graph.salt_ste",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("weight", &[0, u64::from(u32::MAX)], &[])],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 0,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: u64::from(u32::MAX),
                    },
                    Attribute::U64 {
                        name: "planes",
                        value: 2,
                    },
                ],
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.rows.positive",
                    outputs: vec![f32_buffer("result", &[], &[123.0])],
                },
            },
            Case {
                case_id: "graph.salt_ste.forward.too_many_planes",
                operation: "graph.salt_ste",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("weight", &[1, 1], &[0.5])],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 1,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 1,
                    },
                    Attribute::U64 {
                        name: "planes",
                        value: 65,
                    },
                ],
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.planes.max_64",
                    outputs: vec![f32_buffer("result", &[1, 1], &[123.0])],
                },
            },
            Case {
                case_id: "graph.salt_ste.forward.reordered_attributes",
                operation: "graph.salt_ste",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("weight", &[1, 1], &[0.5])],
                attributes: vec![
                    Attribute::U64 {
                        name: "cols",
                        value: 1,
                    },
                    Attribute::U64 {
                        name: "rows",
                        value: 1,
                    },
                    Attribute::U64 {
                        name: "planes",
                        value: 2,
                    },
                ],
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "roles.attribute",
                    outputs: vec![f32_buffer("result", &[1, 1], &[123.0])],
                },
            },
            Case {
                case_id: "graph.lsq_ste.vjp.zero_cols",
                operation: "graph.lsq_ste",
                execution: "vjp",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("weight", &[1, 0], &[]),
                    f32_buffer("alpha", &[1], &[1.0]),
                    f32_buffer("grad_output", &[1, 0], &[]),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 1,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 0,
                    },
                ],
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.cols.positive",
                    outputs: vec![
                        f32_buffer("grad_weight", &[], &[123.0]),
                        f32_buffer("grad_alpha", &[], &[456.0]),
                    ],
                },
            },
            Case {
                case_id: "graph.fsq.forward.zero_len",
                operation: "graph.fsq",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("x", &[1, 0], &[])],
                attributes: fsq_attributes(1, 0, &[3], "clamp", "hard", 0.0, 0),
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.len.positive",
                    outputs: vec![f32_buffer("result", &[], &[123.0])],
                },
            },
            Case {
                case_id: "graph.fsq.forward.invalid_alpha",
                operation: "graph.fsq",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("x", &[1, 1], &[0.25])],
                attributes: fsq_attributes(1, 1, &[3], "clamp", "soft_round", 1.5, 0),
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.alpha.unit_interval",
                    outputs: vec![f32_buffer("result", &[1, 1], &[123.0])],
                },
            },
            Case {
                case_id: "graph.fsq.forward.unknown_ste",
                operation: "graph.fsq",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("x", &[1, 1], &[0.25])],
                attributes: fsq_attributes(1, 1, &[3], "clamp", "unknown", 0.0, 0),
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.ste.known",
                    outputs: vec![f32_buffer("result", &[1, 1], &[123.0])],
                },
            },
            Case {
                case_id: "graph.conv1d.forward.zero_groups",
                operation: "graph.conv1d",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("x", &[1, 2, 4], &conv1d_x),
                    f32_buffer("weight", &[2, 1, 2], &conv1d_weight),
                    f32_buffer("scale", &[2], &conv1d_scale),
                ],
                attributes: conv1d_attributes(&conv1d_zero_groups),
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.groups.positive",
                    outputs: vec![f32_buffer("result", &[1, 2, 4], &[123.0; 8])],
                },
            },
            Case {
                case_id: "graph.conv1d.forward.ragged_groups",
                operation: "graph.conv1d",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("x", &[1, 2, 4], &conv1d_x),
                    f32_buffer("weight", &[2, 1, 2], &conv1d_weight),
                    f32_buffer("scale", &[2], &conv1d_scale),
                ],
                attributes: conv1d_attributes(&conv1d_ragged_groups),
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.groups.divides_channels",
                    outputs: vec![f32_buffer("result", &[1, 2, 4], &[123.0; 8])],
                },
            },
            Case {
                case_id: "graph.conv1d.forward.axis_u32_overflow",
                operation: "graph.conv1d",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("x", &[1, 1, 1], &[0.5]),
                    f32_buffer("weight", &[1, 1, 1], &[1.0]),
                    f32_buffer("scale", &[1], &[1.0]),
                ],
                attributes: conv1d_attributes(&conv1d_axis_boundary),
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.k.axis_u32",
                    outputs: vec![f32_buffer("result", &[], &[123.0])],
                },
            },
            Case {
                case_id: "graph.conv1d.forward.scratch_limit",
                operation: "graph.conv1d",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("x", &[1, 1, 1], &[0.5]),
                    f32_buffer("weight", &[1, 1, 1], &[1.0]),
                    f32_buffer("scale", &[1], &[1.0]),
                ],
                attributes: conv1d_attributes(&conv1d_scratch_overflow),
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.scratch.limit_64_mib",
                    outputs: vec![f32_buffer("result", &[], &[123.0])],
                },
            },
            Case {
                case_id: "graph.conv2d.forward.zero_groups",
                operation: "graph.conv2d",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("x", &[1, 2, 3, 4], &conv2d_x),
                    f32_buffer("weight", &[2, 1, 2, 2], &conv2d_weight),
                    f32_buffer("scale", &[2], &conv2d_scale),
                ],
                attributes: conv2d_attributes(&conv2d_zero_groups),
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.groups.positive",
                    outputs: vec![f32_buffer("result", &[1, 2, 3, 2], &[123.0; 12])],
                },
            },
            Case {
                case_id: "graph.conv2d.forward.oversized_kernel",
                operation: "graph.conv2d",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("x", &[1, 2, 3, 4], &conv2d_x),
                    f32_buffer("weight", &[2, 1, 2, 2], &conv2d_weight),
                    f32_buffer("scale", &[2], &conv2d_scale),
                ],
                attributes: conv2d_attributes(&conv2d_oversized_kernel),
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.kernel_h.output_nonzero",
                    outputs: vec![f32_buffer("result", &[1, 2, 3, 2], &[123.0; 12])],
                },
            },
            Case {
                case_id: "optimizer.adamw.step.zero_step",
                operation: "optimizer.adamw",
                execution: "step",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("parameter", &[2], &parameter),
                    f32_buffer("gradient", &[2], &gradient),
                    f32_buffer("moment1", &[2], &[0.0; 2]),
                    f32_buffer("moment2", &[2], &[0.0; 2]),
                ],
                attributes: adam_attributes(0, &adam_config),
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.step.one_based",
                    outputs: vec![
                        f32_buffer("parameter", &[2], &[123.0; 2]),
                        f32_buffer("moment1", &[2], &[234.0; 2]),
                        f32_buffer("moment2", &[2], &[345.0; 2]),
                    ],
                },
            },
            Case {
                case_id: "optimizer.cautious_adamw.step.invalid_beta1",
                operation: "optimizer.cautious_adamw",
                execution: "step",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("parameter", &[2], &parameter),
                    f32_buffer("gradient", &[2], &gradient),
                    f32_buffer("moment1", &[2], &[0.0; 2]),
                    f32_buffer("moment2", &[2], &[0.0; 2]),
                ],
                attributes: adam_attributes(1, &adam_invalid_beta1),
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.beta1.unit_interval_open",
                    outputs: vec![
                        f32_buffer("parameter", &[2], &[123.0; 2]),
                        f32_buffer("moment1", &[2], &[234.0; 2]),
                        f32_buffer("moment2", &[2], &[345.0; 2]),
                    ],
                },
            },
            Case {
                case_id: "optimizer.int8_adamw.step.scale_shape",
                operation: "optimizer.int8_adamw",
                execution: "step",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("parameter", &[2], &parameter),
                    f32_buffer("gradient", &[2], &gradient),
                    bytes_buffer("moment1_q8", &[2], &[0; 2]),
                    bytes_buffer("moment2_q8", &[2], &[0; 2]),
                    f32_buffer("moment1_scale", &[], &[0.0]),
                    f32_buffer("moment2_scale", &[1], &[0.0]),
                ],
                attributes: adam_attributes(1, &adam_config),
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "shape",
                    outputs: vec![
                        f32_buffer("parameter", &[2], &[123.0; 2]),
                        bytes_buffer("moment1_q8", &[2], &[123; 2]),
                        bytes_buffer("moment2_q8", &[2], &[234; 2]),
                        f32_buffer("moment1_scale", &[1], &[345.0]),
                        f32_buffer("moment2_scale", &[1], &[456.0]),
                    ],
                },
            },
            Case {
                case_id: "optimizer.muon.step.zero_ns_steps",
                operation: "optimizer.muon",
                execution: "step",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("parameter", &[2, 3], &muon_parameter),
                    f32_buffer("gradient", &[2, 3], &muon_gradient),
                    f32_buffer("momentum", &[2, 3], &muon_input_momentum),
                ],
                attributes: muon_attributes(1, &muon_zero_steps),
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.ns_steps.positive",
                    outputs: vec![
                        f32_buffer("parameter", &[2, 3], &[123.0; 6]),
                        f32_buffer("momentum", &[2, 3], &[234.0; 6]),
                    ],
                },
            },
            Case {
                case_id: "lifecycle.checkpoint.negative_second_moment",
                operation: "lifecycle.checkpoint",
                execution: "checkpoint",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("parameter.0", &[3], &checkpoint.leaves[0].param),
                    f32_buffer("moment1.0", &[3], &checkpoint.leaves[0].state.m),
                    f32_buffer("moment2.0", &[3], &[0.01, -0.04, 0.09]),
                    f32_buffer("parameter.1", &[2], &checkpoint.leaves[1].param),
                    f32_buffer("moment1.1", &[2], &checkpoint.leaves[1].state.m),
                    f32_buffer("moment2.1", &[2], &checkpoint.leaves[1].state.v),
                ],
                attributes: checkpoint_attributes("adamw", checkpoint.step, &[3, 2]),
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.moment2.nonnegative",
                    outputs: vec![bytes_buffer(
                        "checkpoint",
                        &[checkpoint_bytes.len() as u64],
                        &vec![123; checkpoint_bytes.len()],
                    )],
                },
            },
            Case {
                case_id: "lifecycle.resume.bad_magic",
                operation: "lifecycle.resume",
                execution: "resume",
                tolerance: Tolerance::BitExact,
                inputs: vec![bytes_buffer(
                    "checkpoint",
                    &[corrupt_checkpoint.len() as u64],
                    &corrupt_checkpoint,
                )],
                attributes: resume_attributes("adamw", &[3, 2]),
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.checkpoint.bad_magic",
                    outputs: vec![
                        bytes_buffer("step", &[8], &[123; 8]),
                        f32_buffer("parameter.0", &[3], &[123.0; 3]),
                        f32_buffer("moment1.0", &[3], &[234.0; 3]),
                        f32_buffer("moment2.0", &[3], &[345.0; 3]),
                        f32_buffer("parameter.1", &[2], &[123.0; 2]),
                        f32_buffer("moment1.1", &[2], &[234.0; 2]),
                        f32_buffer("moment2.1", &[2], &[345.0; 2]),
                    ],
                },
            },
            Case {
                case_id: "lifecycle.resume.negative_second_moment",
                operation: "lifecycle.resume",
                execution: "resume",
                tolerance: Tolerance::BitExact,
                inputs: vec![bytes_buffer(
                    "checkpoint",
                    &[invalid_state_checkpoint_bytes.len() as u64],
                    &invalid_state_checkpoint_bytes,
                )],
                attributes: resume_attributes("adamw", &[3, 2]),
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.moment2.nonnegative",
                    outputs: vec![
                        bytes_buffer("step", &[8], &[123; 8]),
                        f32_buffer("parameter.0", &[3], &[123.0; 3]),
                        f32_buffer("moment1.0", &[3], &[234.0; 3]),
                        f32_buffer("moment2.0", &[3], &[345.0; 3]),
                        f32_buffer("parameter.1", &[2], &[123.0; 2]),
                        f32_buffer("moment1.1", &[2], &[234.0; 2]),
                        f32_buffer("moment2.1", &[2], &[345.0; 2]),
                    ],
                },
            },
            Case {
                case_id: "lifecycle.export.unknown_format",
                operation: "lifecycle.export",
                execution: "export",
                tolerance: Tolerance::BitExact,
                inputs: vec![bytes_buffer(
                    "package",
                    &[hard_artifact.len() as u64],
                    &hard_artifact,
                )],
                attributes: vec![Attribute::Text {
                    name: "format",
                    value: "legacy_salt_bundle",
                }],
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.format.salt_v2_package_v1",
                    outputs: vec![bytes_buffer(
                        "artifact",
                        &[hard_artifact.len() as u64],
                        &vec![123; hard_artifact.len()],
                    )],
                },
            },
            Case {
                case_id: "lifecycle.reload.bad_magic",
                operation: "lifecycle.reload",
                execution: "reload",
                tolerance: Tolerance::BitExact,
                inputs: vec![bytes_buffer(
                    "artifact",
                    &[corrupt_artifact.len() as u64],
                    &corrupt_artifact,
                )],
                attributes: artifact_attributes(),
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.artifact.salt_v2_package",
                    outputs: vec![bytes_buffer(
                        "package",
                        &[hard_artifact.len() as u64],
                        &vec![123; hard_artifact.len()],
                    )],
                },
            },
            Case {
                case_id: "graph.attention.forward.ragged_gqa",
                operation: "graph.attention",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("q", &[3, 2, 2], &attention_q),
                    f32_buffer("k", &[3, 1, 2], &attention_k),
                    f32_buffer("v", &[3, 1, 2], &attention_v),
                ],
                attributes: attention_attributes(&attention_ragged_gqa_cfg),
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.n_kv_head.divides_n_head",
                    outputs: vec![f32_buffer("result", &[3, 2, 2], &[123.0; 12])],
                },
            },
            Case {
                case_id: "graph.attention.forward.product_limit",
                operation: "graph.attention",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("q", &[1], &[0.5]),
                    f32_buffer("k", &[1], &[0.25]),
                    f32_buffer("v", &[1], &[1.0]),
                ],
                attributes: attention_attributes(&attention_product_limit_cfg),
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.seq.max_elements",
                    outputs: vec![f32_buffer("result", &[], &[123.0])],
                },
            },
        ],
    };

    let mut bytes = serde_json::to_vec_pretty(&corpus).expect("serialize tracer corpus");
    bytes.push(b'\n');
    TrainingVectorSetV1::parse_json(&bytes).expect("generated corpus must validate");

    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/training/v1/vectors/v1.json");
    let temporary = path.with_extension("json.tmp");
    std::fs::write(&temporary, &bytes).expect("write temporary vector corpus");
    std::fs::rename(&temporary, &path).expect("atomically replace vector corpus");
    eprintln!(
        "froze {} cases -> {} ({})",
        corpus.cases.len(),
        path.display(),
        hex(blake3::hash(&bytes).as_bytes())
    );
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn conv1d_attributes(cfg: &conv1d::Conv1dCfg) -> Vec<Attribute> {
    [
        ("batch", cfg.batch),
        ("c_in", cfg.c_in),
        ("c_out", cfg.c_out),
        ("l_in", cfg.l_in),
        ("k", cfg.k),
        ("stride", cfg.stride),
        ("dilation", cfg.dilation),
        ("pad_left", cfg.pad_left),
        ("pad_right", cfg.pad_right),
        ("groups", cfg.groups),
    ]
    .into_iter()
    .map(|(name, value)| Attribute::U64 {
        name,
        value: value as u64,
    })
    .collect()
}

fn conv2d_attributes(cfg: &conv2d::Conv2dCfg) -> Vec<Attribute> {
    [
        ("batch", cfg.batch),
        ("c_in", cfg.c_in),
        ("c_out", cfg.c_out),
        ("input_h", cfg.input_h),
        ("input_w", cfg.input_w),
        ("kernel_h", cfg.kernel_h),
        ("kernel_w", cfg.kernel_w),
        ("stride_h", cfg.stride_h),
        ("stride_w", cfg.stride_w),
        ("dilation_h", cfg.dilation_h),
        ("dilation_w", cfg.dilation_w),
        ("pad_top", cfg.pad_top),
        ("pad_bottom", cfg.pad_bottom),
        ("pad_left", cfg.pad_left),
        ("pad_right", cfg.pad_right),
        ("groups", cfg.groups),
    ]
    .into_iter()
    .map(|(name, value)| Attribute::U64 {
        name,
        value: value as u64,
    })
    .collect()
}

fn attention_attributes(cfg: &attention::AttentionCfg) -> Vec<Attribute> {
    vec![
        Attribute::U64 {
            name: "seq",
            value: cfg.seq as u64,
        },
        Attribute::U64 {
            name: "n_head",
            value: cfg.n_head as u64,
        },
        Attribute::U64 {
            name: "n_kv_head",
            value: cfg.n_kv_head as u64,
        },
        Attribute::U64 {
            name: "head_dim",
            value: cfg.head_dim as u64,
        },
        Attribute::Bool {
            name: "causal",
            value: cfg.causal,
        },
    ]
}

fn adam_attributes(step: u64, config: &AdamW) -> Vec<Attribute> {
    vec![
        Attribute::U64 {
            name: "step",
            value: step,
        },
        Attribute::F32 {
            name: "lr",
            bits: config.lr.to_bits(),
        },
        Attribute::F32 {
            name: "beta1",
            bits: config.beta1.to_bits(),
        },
        Attribute::F32 {
            name: "beta2",
            bits: config.beta2.to_bits(),
        },
        Attribute::F32 {
            name: "eps",
            bits: config.eps.to_bits(),
        },
        Attribute::F32 {
            name: "weight_decay",
            bits: config.weight_decay.to_bits(),
        },
    ]
}

fn muon_attributes(step: u64, config: &Muon) -> Vec<Attribute> {
    vec![
        Attribute::U64 {
            name: "step",
            value: step,
        },
        Attribute::F32 {
            name: "lr",
            bits: config.lr.to_bits(),
        },
        Attribute::F32 {
            name: "momentum",
            bits: config.momentum.to_bits(),
        },
        Attribute::F32 {
            name: "weight_decay",
            bits: config.weight_decay.to_bits(),
        },
        Attribute::U64 {
            name: "rows",
            value: config.rows as u64,
        },
        Attribute::U64 {
            name: "cols",
            value: config.cols as u64,
        },
        Attribute::U64 {
            name: "ns_steps",
            value: config.ns_steps as u64,
        },
    ]
}

fn checkpoint_attributes(optimizer: &'static str, step: u64, leaf_lens: &[u64]) -> Vec<Attribute> {
    vec![
        Attribute::Text {
            name: "optimizer",
            value: optimizer,
        },
        Attribute::U64 {
            name: "step",
            value: step,
        },
        Attribute::U64List {
            name: "leaf_lens",
            values: leaf_lens.to_vec(),
        },
    ]
}

fn resume_attributes(optimizer: &'static str, leaf_lens: &[u64]) -> Vec<Attribute> {
    vec![
        Attribute::Text {
            name: "optimizer",
            value: optimizer,
        },
        Attribute::U64List {
            name: "leaf_lens",
            values: leaf_lens.to_vec(),
        },
    ]
}

fn artifact_attributes() -> Vec<Attribute> {
    vec![Attribute::Text {
        name: "format",
        value: "salt_v2_package_v1",
    }]
}
