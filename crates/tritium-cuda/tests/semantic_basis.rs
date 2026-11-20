//! Physical device basis parity through the actual shared tensor interface.
#![cfg(feature = "cuda")]

use tritium_core::{Trit, reference_embed, reference_ternary_matmul};
use tritium_cuda::CudaBackend;
use tritium_format::AdditiveTensor;
use tritium_schema::{ADMITTED_LAWS, Basis, PlaneCodec};
use tritium_spec::{TensorMatmul, TensorUploadPolicy, TensorView, TernaryBackend};

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn device_forward_and_inverse_basis_match_core_over_wide_blocks_and_batches() {
    let cuda = CudaBackend::new(0).expect("basis conformance requires physical CUDA");
    let mut cases = 0;
    for cols in [4_usize, 32, 128, 256, 260, 512, 4096] {
        for block in [1_u32, 2, 4, 8, 16, 32, 128, 512, 4096] {
            if !cols.is_multiple_of(block as usize) {
                continue;
            }
            for basis in [
                Basis::Hadamard { block },
                Basis::SignedRht {
                    block,
                    seed: 7,
                    domain: 9,
                },
                Basis::SignedRht {
                    block,
                    seed: u64::MAX,
                    domain: 0,
                },
            ] {
                for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
                    let mut layout = tritium_testkit::additive_upload_fixture().layout();
                    layout.rows = 3;
                    layout.cols = cols as u64;
                    layout.max_planes = 3;
                    layout.law = ADMITTED_LAWS[0].law;
                    layout.basis = basis;
                    layout.codec = codec;
                    let source = AdditiveTensor::new(
                        layout,
                        3,
                        [Trit::NEG, Trit::ZERO, Trit::POS, Trit::NEG]
                            .into_iter()
                            .cycle()
                            .take(9 * cols)
                            .collect(),
                        (0..9 * cols.div_ceil(32))
                            .map(|i| (i % 4 + 1) as f32 * 0.125)
                            .collect(),
                    )
                    .unwrap();
                    let activations: Vec<_> = [
                        0.,
                        -0.,
                        f32::from_bits(1),
                        f32::from_bits(0x80000001),
                        1.,
                        -2.,
                        7.25,
                        -0.125,
                        1e20,
                        -1e20,
                    ]
                    .into_iter()
                    .cycle()
                    .take(3 * cols)
                    .collect();
                    let mut expected_scratch = vec![0.; activations.len()];
                    let mut expected = vec![0.; 9];
                    reference_ternary_matmul(
                        &activations,
                        &source.view().bind_matmul(),
                        3,
                        &mut expected_scratch,
                        &mut expected,
                    )
                    .unwrap();
                    let ids = [2, 0, 1, 2];
                    let mut expected_gather = vec![0.; ids.len() * cols];
                    reference_embed(&source.view().bind_gather(), &ids, &mut expected_gather)
                        .unwrap();
                    let (handle, _) = cuda
                        .upload_tensor_checked(
                            TensorView::Additive(source.view()),
                            TensorUploadPolicy {
                                native_only: true,
                                max_payload_bytes: None,
                            },
                        )
                        .unwrap();
                    drop(source);
                    let mut scratch = vec![99.; activations.len()];
                    let mut out = vec![99.; 9];
                    cuda.matmul(TensorMatmul {
                        act: &activations,
                        tensor: &*handle,
                        batch: 3,
                        transformed_act: &mut scratch,
                        out: &mut out,
                    })
                    .unwrap();
                    assert_eq!(
                        bits(&scratch),
                        bits(&expected_scratch),
                        "{cols}/{basis:?}/{codec:?} scratch"
                    );
                    assert_eq!(
                        bits(&out),
                        bits(&expected),
                        "{cols}/{basis:?}/{codec:?} matmul"
                    );
                    let mut gathered = vec![99.; expected_gather.len()];
                    cuda.embed_rows(&*handle, &ids, &mut gathered).unwrap();
                    assert_eq!(
                        bits(&gathered),
                        bits(&expected_gather),
                        "{cols}/{basis:?}/{codec:?} gather"
                    );
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 387);
    eprintln!("physical CUDA wide-basis cases={cases}");
}
