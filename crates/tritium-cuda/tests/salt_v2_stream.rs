//! Gates for the row-streaming SALT V2 GEMV (`salt_v2_stream_f32`).
//!
//! The kernel reads a row's B3 plane-tiles as aligned 32-bit words and reduces by
//! shuffle, so it reassociates the K-sum: parity is a relative-error bound against
//! the CPU reference, never equality. Every comparison also pins the receipt,
//! because a fast entry point that quietly fell back to another kernel would pass
//! an error bound while measuring nothing about this one.
//!
//! The kernel serves only B3 at scale group 128 with a whole number of 256-wide
//! tiles, so these tensors are built to that shape. They cycle one, two and three
//! planes per tile, which is what makes the per-row plane-tile table non-trivial,
//! and one shape runs past 256 tiles so rows anchor on a stored rank prefix and
//! straddle it.
#![cfg(feature = "cuda")]

use half::f16;
use tritium_cpu::salt_v2::salt_v2_matvec;
use tritium_cuda::{CudaBackend, SaltV2ForwardMode};
use tritium_format::salt_v2::SaltV2Codec;
use tritium_format::salt_v2_package::{
    SaltV2Package, SaltV2Plane, SaltV2Tensor, SaltV2Tile, SaltV2Transform,
};

const SCALE_GROUP: usize = 128;

/// Deterministic trits with every value present and exactly one zero per four
/// (so the same tensor is also valid under S34).
fn trits_for(tile: usize, plane: usize) -> Vec<i8> {
    let zero_slot = (tile + plane) % 4;
    (0..256)
        .map(|index| {
            if index % 4 == zero_slot {
                0
            } else if (index * 7 + tile * 3 + plane) % 5 < 2 {
                1
            } else {
                -1
            }
        })
        .collect()
}

fn tensor(rows: usize, columns: usize, planes_of: impl Fn(usize) -> usize) -> SaltV2Tensor {
    let tiles = (0..rows * columns / 256)
        .map(|tile| {
            let planes = (0..planes_of(tile))
                .map(|plane| {
                    let scales = (0..256 / SCALE_GROUP)
                        .map(|group| {
                            let step = (tile * 5 + group * 3 + plane) % 17;
                            f16::from_f32(0.03125 + step as f32 / 64.0)
                        })
                        .collect();
                    SaltV2Plane::new_with_scale_group_size(
                        trits_for(tile, plane),
                        scales,
                        SCALE_GROUP,
                    )
                    .unwrap()
                })
                .collect();
            SaltV2Tile::new(planes).unwrap()
        })
        .collect();
    SaltV2Tensor::new_with_layout(
        "stream.weight",
        vec![rows as u64, columns as u64],
        SaltV2Transform::None,
        SCALE_GROUP,
        tiles,
    )
    .unwrap()
}

fn activation(batch: usize, columns: usize) -> Vec<f32> {
    (0..batch * columns)
        .map(|index| ((index as f32) * 0.37).sin() * 0.8 + ((index % 13) as f32 - 6.0) / 50.0)
        .collect()
}

fn check(
    cuda: &CudaBackend,
    tensor: &SaltV2Tensor,
    codec: SaltV2Codec,
    expect: SaltV2ForwardMode,
    label: &str,
) {
    let rows = tensor.dims()[0] as usize;
    let columns = tensor.dims()[1] as usize;
    let package = SaltV2Package::new(codec, vec![tensor.clone()]).unwrap();
    let resident = cuda.upload_salt_v2(tensor, codec).unwrap();
    for batch in [1usize, 3] {
        let act = activation(batch, columns);
        let mut expected = Vec::with_capacity(batch * rows);
        for row in 0..batch {
            let slice = &act[row * columns..(row + 1) * columns];
            expected.extend(salt_v2_matvec(&package, 0, slice).unwrap().output);
        }
        let fast = cuda.salt_v2_forward_fast(&resident, &act, batch).unwrap();
        assert_eq!(
            fast.receipt.mode(),
            expect,
            "{label} {codec:?} batch {batch}: wrong kernel answered"
        );
        let peak = expected
            .iter()
            .fold(0.0f32, |peak, value| peak.max(value.abs()))
            .max(f32::MIN_POSITIVE);
        for (lane, (&got, &want)) in fast.output.iter().zip(&expected).enumerate() {
            assert!(
                (got - want).abs() / peak <= 1e-5,
                "{label} {codec:?} batch {batch} lane {lane}: {got} vs {want}"
            );
        }
    }
}

#[test]
fn stream_kernel_matches_the_reference_on_ragged_planes() {
    let Ok(cuda) = CudaBackend::new(0) else {
        eprintln!("skipping SALT V2 stream parity: no CUDA device");
        return;
    };
    let ragged = tensor(6, 1024, |tile| tile % 3 + 1);
    check(
        &cuda,
        &ragged,
        SaltV2Codec::B3,
        SaltV2ForwardMode::FastRowStream,
        "ragged",
    );
    let single = tensor(5, 512, |_| 1);
    check(
        &cuda,
        &single,
        SaltV2Codec::B3,
        SaltV2ForwardMode::FastRowStream,
        "single",
    );
}

#[test]
fn stream_kernel_matches_the_reference_across_a_rank_prefix_boundary() {
    let Ok(cuda) = CudaBackend::new(0) else {
        eprintln!("skipping SALT V2 stream rank-prefix parity: no CUDA device");
        return;
    };
    // Three tiles per row does not divide the 256-tile rank stride, so rows past
    // tile 256 anchor on a stored prefix and straddle it mid-row. 120 x 768 is
    // 360 tiles.
    let crossing = tensor(120, 768, |tile| (tile * 7) % 3 + 1);
    check(
        &cuda,
        &crossing,
        SaltV2Codec::B3,
        SaltV2ForwardMode::FastRowStream,
        "crossing",
    );
}

#[test]
fn other_codecs_take_the_warp_kernel_and_say_so() {
    let Ok(cuda) = CudaBackend::new(0) else {
        eprintln!("skipping SALT V2 stream eligibility: no CUDA device");
        return;
    };
    // The stream kernel is B3-only; D2 at the same shape must name the warp kernel.
    let ragged = tensor(6, 1024, |tile| tile % 3 + 1);
    check(
        &cuda,
        &ragged,
        SaltV2Codec::D2,
        SaltV2ForwardMode::FastWarpReduce,
        "d2",
    );
}

/// The A8 row-stream GEMV against an exact reference built from the same int8
/// activations and scales the kernel saw.
///
/// Quantization error is not the kernel's to answer for, so the reference uses
/// the device's own quantized activations, dequantized. What remains is only the
/// kernel's arithmetic: exact int32 partials, then a float fold per word.
#[test]
fn a8_kernel_matches_a_reference_on_its_own_quantized_inputs() {
    let Ok(cuda) = CudaBackend::new(0) else {
        eprintln!("skipping SALT V2 A8 parity: no CUDA device");
        return;
    };
    for (label, tensor) in [
        ("ragged", tensor(6, 1024, |tile| tile % 3 + 1)),
        ("crossing", tensor(120, 768, |tile| (tile * 7) % 3 + 1)),
    ] {
        let rows = tensor.dims()[0] as usize;
        let columns = tensor.dims()[1] as usize;
        let package = SaltV2Package::new(SaltV2Codec::B3, vec![tensor.clone()]).unwrap();
        let resident = cuda.upload_salt_v2(&tensor, SaltV2Codec::B3).unwrap();
        let act = activation(1, columns);
        let (output, quantized, scale) = cuda.salt_v2_forward_a8_probe(&resident, &act, 1).unwrap();
        // The quantizer is itself checked: every value within half a step.
        for (index, (&q, &x)) in quantized.iter().zip(&act).enumerate() {
            let step = scale[index / 128];
            assert!(
                (f32::from(q) * step - x).abs() <= step * 0.5 + 1e-6,
                "{label}: activation {index} quantized to {q} at step {step}, from {x}"
            );
        }
        let dequantized: Vec<f32> = quantized
            .iter()
            .enumerate()
            .map(|(index, &q)| f32::from(q) * scale[index / 128])
            .collect();
        let expected = salt_v2_matvec(&package, 0, &dequantized).unwrap().output;
        assert_eq!(output.len(), rows);
        let peak = expected
            .iter()
            .fold(0.0f32, |peak, value| peak.max(value.abs()))
            .max(f32::MIN_POSITIVE);
        for (lane, (&got, &want)) in output.iter().zip(&expected).enumerate() {
            assert!(
                (got - want).abs() / peak <= 1e-5,
                "{label} lane {lane}: A8 {got} vs reference on its own inputs {want}"
            );
        }
    }
}
