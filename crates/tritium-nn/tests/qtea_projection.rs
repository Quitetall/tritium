//! Synthetic model execution, not pretrained checkpoint/quality qualification.
#[path = "qtea/common.rs"]
mod common;

use std::{io::Cursor, sync::Arc};
use tritium_nn::{
    ArchSpec, DenseLinear, HostQteaLinear, Mlp, ModelConfig, ModelRunner, ModelWeights, Projection,
    ProjectionActivationMode, SwiGluMlp, SwiGluTrainingModel, TokenEmbedding, TransformerBlock,
};

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn cold_read_projections_preserve_affine_residual_arithmetic() {
    let backend = tritium_cpu::CpuBackend::new();
    for (rows, cols, group) in [(4, 4, 2), (8, 8, 4), (4, 12, 6)] {
        for ratio in [0., 0.5, 1.] {
            let (tensor, decoded) = common::fixture(rows, cols, group, ratio, 3);
            let matrix = Arc::new(
                HostQteaLinear::read_from(
                    &mut Cursor::new(tensor.as_bytes()),
                    common::limits(),
                    3 * rows * 4,
                )
                .unwrap(),
            );
            assert_eq!((matrix.rows(), matrix.columns()), (rows, cols));
            assert_eq!(matrix.encoded_capacity_bytes(), tensor.as_bytes().len());
            let projection = Projection::HostQtea(Arc::clone(&matrix));
            assert_eq!(projection.activation_mode(), ProjectionActivationMode::F32);
            assert_eq!((projection.n_out(), projection.k_in()), (rows, cols));
            assert!(projection.as_ternary().is_none());
            for batch in [0, 1, 3] {
                let input: Vec<_> = (0..batch * cols)
                    .map(|i| (i % 13) as f32 / 8. - 0.5)
                    .collect();
                let mut expected = vec![0.; batch * rows];
                for b in 0..batch {
                    for r in 0..rows {
                        for c in 0..cols {
                            expected[b * rows + r] += input[b * cols + c] * decoded[r * cols + c];
                        }
                    }
                }
                let mut output = vec![99.; expected.len()];
                projection
                    .forward(&backend, &input, batch, &mut output)
                    .unwrap();
                assert_eq!(bits(&output), bits(&expected));
                let mut staging = vec![77.; output.len()];
                matrix
                    .forward_into(&input, batch, &mut output, &mut staging)
                    .unwrap();
                assert_eq!(bits(&output), bits(&expected));
            }
            assert_eq!(Arc::strong_count(&matrix), 2);
            // The diagnostic doesn't print the encoded model or coefficient arrays.
            assert!(!format!("{matrix:?}").contains("bytes: ["));
        }
    }
}

#[test]
fn reader_caps_shapes_and_numeric_failures_are_atomic() {
    let (tensor, _) = common::fixture(4, 8, 2, 1., 0);
    let bytes = tensor.as_bytes();
    for cut in [0, 8, 383, bytes.len() - 1] {
        assert!(
            HostQteaLinear::read_from(&mut Cursor::new(&bytes[..cut]), common::limits(), 64)
                .is_err()
        );
    }
    let mut trailing = bytes.to_vec();
    trailing.push(0);
    assert!(HostQteaLinear::read_from(&mut Cursor::new(trailing), common::limits(), 64).is_err());
    let mut broken = bytes.to_vec();
    broken[400] ^= 1;
    assert!(HostQteaLinear::read_from(&mut Cursor::new(broken), common::limits(), 64).is_err());
    let mut limits = common::limits();
    limits.max_tensor_bytes = bytes.len() - 1;
    assert!(HostQteaLinear::read_from(&mut Cursor::new(bytes), limits, 64).is_err());

    let matrix = HostQteaLinear::from_tensor(tensor, 32);
    let mut output = [55.; 8];
    let mut staging = [77.; 8];
    assert!(matrix.forward(&[0.; 15], 2, &mut output).is_err());
    assert_eq!(output, [55.; 8]);
    assert!(matrix.forward(&[], usize::MAX, &mut []).is_err());
    let mut input = [0.; 16];
    input[15] = f32::NAN;
    assert!(matrix.forward(&input, 2, &mut output).is_err());
    assert_eq!(output, [55.; 8]);
    input[15] = f32::INFINITY;
    assert!(
        matrix
            .forward_into(&input, 2, &mut output, &mut staging)
            .is_err()
    );
    assert_eq!(output, [55.; 8]);
    assert_eq!(staging, [77.; 8]);
    assert!(
        matrix
            .forward_into(&[0.; 16], 2, &mut output, &mut staging[..7])
            .is_err()
    );
    assert_eq!(output, [55.; 8]);
    // A late reduction overflow may dirty staging, but not caller output.
    let (overflow_tensor, decoded) = common::fixture(4, 32, 2, 1., 0);
    let overflow_matrix = HostQteaLinear::from_tensor(overflow_tensor, 32);
    let mut overflow_input = [0.; 64];
    let overflow_row = decoded
        .as_chunks::<32>()
        .0
        .iter()
        .max_by(|left, right| {
            let sum = |row: &[f32]| row.iter().map(|w| w.abs()).sum::<f32>();
            sum(left.as_slice()).total_cmp(&sum(right.as_slice()))
        })
        .unwrap();
    assert!(overflow_row.iter().map(|w| w.abs()).sum::<f32>() > 1.);
    for (value, weight) in overflow_input[32..].iter_mut().zip(overflow_row) {
        *value = f32::MAX.copysign(*weight);
    }
    assert!(
        overflow_matrix
            .forward_into(&overflow_input, 2, &mut output, &mut staging)
            .is_err()
    );
    assert_eq!(output, [55.; 8]);
    assert_eq!(&staging[..4], &[0.; 4]);
    let (tensor, _) = common::fixture(4, 8, 2, 1., 0);
    let capped = HostQteaLinear::from_tensor(tensor, 15);
    let mut short = [66.; 4];
    assert!(capped.forward(&[0.; 8], 1, &mut short).is_err());
    assert_eq!(short, [66.; 4]);
    capped.forward(&[], 0, &mut []).unwrap();
}

fn config() -> ModelConfig {
    ModelConfig {
        arch: "llama".into(),
        n_layers: 2,
        n_embd: 8,
        n_head: 2,
        n_head_kv: 1,
        head_dim: 4,
        n_ff: 12,
        n_ctx: 16,
        rope_theta: 10_000.,
        rms_eps: 1e-5,
    }
}

fn projection(rows: usize, cols: usize, seed: usize, packed: bool) -> Projection {
    let (tensor, decoded) = common::fixture(rows, cols, 2, 0.5, seed);
    if packed {
        let matrix = HostQteaLinear::read_from(
            &mut Cursor::new(tensor.as_bytes()),
            common::limits(),
            16 * rows * 4,
        )
        .unwrap();
        Projection::HostQtea(Arc::new(matrix))
    } else {
        Projection::Dense(DenseLinear::new_exact(decoded, rows, cols).unwrap())
    }
}

fn weights(packed: bool) -> ModelWeights {
    let layers = (0..2)
        .map(|layer| {
            let seed = layer * 7;
            TransformerBlock {
                attn_norm: vec![1.; 8],
                q_proj: projection(8, 8, seed, packed),
                k_proj: projection(4, 8, seed + 1, packed),
                v_proj: projection(4, 8, seed + 2, packed),
                o_proj: projection(8, 8, seed + 3, packed),
                attn_sub_norm: vec![],
                q_bias: vec![],
                k_bias: vec![],
                v_bias: vec![],
                q_norm: vec![],
                k_norm: vec![],
                ffn_norm: vec![1.; 8],
                mlp: Mlp::SwiGlu(SwiGluMlp {
                    gate: projection(12, 8, seed + 4, packed),
                    up: projection(12, 8, seed + 5, packed),
                    down: projection(8, 12, seed + 6, packed),
                }),
            }
        })
        .collect();
    ModelWeights {
        token_embd: TokenEmbedding::from_dense(
            (0..64).map(|i| (i % 7) as f32 / 16. - 0.125).collect(),
            8,
            8,
        )
        .unwrap(),
        vocab: 8,
        n_embd: 8,
        layers,
        output_norm: vec![1.; 8],
        lm_head: Some(projection(8, 8, 29, packed)),
    }
}

#[test]
fn complete_packed_transformer_prefill_decode_reset_matches_decoded_oracle() {
    let mut packed = ModelRunner::try_from_weights(
        config(),
        weights(true),
        Box::new(tritium_cpu::CpuBackend::new()),
    )
    .unwrap();
    let mut reference = ModelRunner::try_from_weights(
        config(),
        weights(false),
        Box::new(tritium_cpu::CpuBackend::new()),
    )
    .unwrap();
    for (tokens, positions) in [
        (&[1, 2, 3][..], &[0, 1, 2][..]),
        (&[4][..], &[3][..]),
        (&[5][..], &[4][..]),
    ] {
        let actual = packed.forward(tokens, positions).unwrap();
        let expected = reference.forward(tokens, positions).unwrap();
        assert!(actual.iter().all(|v| v.is_finite()));
        assert_eq!(bits(&actual), bits(&expected));
    }
    packed.reset();
    reference.reset();
    assert_eq!(
        bits(&packed.forward(&[2], &[0]).unwrap()),
        bits(&reference.forward(&[2], &[0]).unwrap())
    );
    for layer in &packed.weights.layers {
        assert!(matches!(layer.q_proj, Projection::HostQtea(_)));
        assert!(matches!(layer.k_proj, Projection::HostQtea(_)));
        assert!(matches!(layer.v_proj, Projection::HostQtea(_)));
        assert!(matches!(layer.o_proj, Projection::HostQtea(_)));
        let Mlp::SwiGlu(mlp) = &layer.mlp else {
            unreachable!()
        };
        assert!(matches!(mlp.gate, Projection::HostQtea(_)));
        assert!(matches!(mlp.up, Projection::HostQtea(_)));
        assert!(matches!(mlp.down, Projection::HostQtea(_)));
    }
    assert!(matches!(
        packed.weights.lm_head,
        Some(Projection::HostQtea(_))
    ));
}

#[test]
fn inspectable_master_training_refuses_packed_qtea() {
    let spec = ArchSpec {
        mlp: tritium_nn::MlpKind::SwiGlu,
        attn_sub_norm: false,
        ffn_sub_norm: false,
        qk_norm: false,
        qkv_bias: false,
        tied_embeddings: false,
    };
    let result = SwiGluTrainingModel::extract(&config(), &spec, &weights(true));
    let Err(error) = result else {
        panic!("hard QTEA must not become a latent training master")
    };
    assert!(
        error.to_string().contains("residual-bearing host QTEA"),
        "{error}"
    );
}

#[test]
fn shared_packed_projection_has_independent_concurrent_workspace() {
    fn require_send_sync<T: Send + Sync>() {}
    require_send_sync::<HostQteaLinear>();
    let (tensor, _) = common::fixture(8, 12, 6, 1., 3);
    let matrix = Arc::new(HostQteaLinear::from_tensor(tensor, 8 * 4));
    let input = [0.125; 12];
    let mut expected = [0.; 8];
    matrix.forward(&input, 1, &mut expected).unwrap();
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let matrix = Arc::clone(&matrix);
            std::thread::spawn(move || {
                let mut output = [99.; 8];
                matrix.forward(&input, 1, &mut output).unwrap();
                bits(&output)
            })
        })
        .collect();
    for handle in handles {
        assert_eq!(handle.join().unwrap(), bits(&expected));
    }
    assert_eq!(Arc::strong_count(&matrix), 1);
}

#[test]
fn packed_embedding_gather_and_tied_head_keep_the_same_residual_weights() {
    let (tensor, decoded) = common::fixture(8, 12, 6, 1., 3);
    let matrix = Arc::new(HostQteaLinear::from_tensor(tensor, 3 * 12 * 4));
    let table = tritium_nn::TokenEmbedding::from_host_qtea(Arc::clone(&matrix));
    let clone = table.clone();
    assert_eq!(Arc::strong_count(&matrix), 3);
    assert!(table.as_dense().is_none());
    assert!(!table.is_packed_salt());
    assert!(table.tensor_caps().is_none());
    assert_eq!(table.sparse_plane_count(), 0);
    let tokens = [7, 0, 7];
    let expected = tokens
        .iter()
        .flat_map(|&r| {
            decoded[r as usize * 12..(r as usize + 1) * 12]
                .iter()
                .copied()
        })
        .collect::<Vec<_>>();
    let mut output = [99.; 36];
    table.gather(&tokens, &mut output).unwrap();
    assert_eq!(bits(&output), bits(&expected));
    let mut staging = [77.; 36];
    matrix
        .gather_into(&tokens, &mut output, &mut staging)
        .unwrap();
    assert_eq!(bits(&output), bits(&expected));
    let unchanged = output;
    assert!(table.gather(&[0, 8, 0], &mut output).is_err());
    assert_eq!(output, unchanged);
    assert!(
        matrix
            .gather_into(&tokens, &mut output, &mut staging[..35])
            .is_err()
    );
    assert_eq!(output, unchanged);
    table.gather(&[], &mut []).unwrap();
    let hidden = [0.125; 12];
    let mut logits = [0.; 8];
    clone.unembed_exact(&hidden, &mut logits).unwrap();
    let expected: Vec<_> = decoded
        .as_chunks::<12>()
        .0
        .iter()
        .map(|row| row.iter().fold(0_f32, |sum, w| sum + 0.125 * w))
        .collect();
    assert_eq!(bits(&logits), bits(&expected));
    let cpu = tritium_cpu::CpuBackend::new();
    clone
        .gather_with_backend(&cpu, &tokens, &mut output)
        .unwrap();
    assert_eq!(output, unchanged);
    clone
        .unembed_exact_with_backend(&cpu, &hidden, &mut logits)
        .unwrap();
    assert_eq!(bits(&logits), bits(&expected));
}
