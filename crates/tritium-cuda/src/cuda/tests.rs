//! GPU conformance + CPU↔CUDA parity tests. Run only with `--features cuda` AND
//! a working CUDA device, so they are exercised on the Wave D GPU CI lane, never
//! on cpu-only lanes. When no device is present the tests self-skip
//! (constructing the backend returns `Err`) rather than failing.
//!
//! `run_conformance` itself packs each vector's trits to TQ2_0 (block scale
//! 1.0), uploads via `upload_weights`, runs `mpgemm` with the per-channel
//! scales, and grades against `reference_mpgemm` — so the test only has to
//! supply the TQ2_0 vectors this kernel supports.

use super::*;
use half::f16;
use std::io::Cursor;
use tritium_cpu::CpuBackend;
use tritium_cpu::salt_v2::salt_v2_matvec;
use tritium_format::salt_v2::SaltV2Codec;
use tritium_format::salt_v2_package::{
    SALT_V2_ALLOCATION_TILE_SIZE, SALT_V2_SCALE_GROUP_SIZE, SaltV2IndexedRuntimeLedger,
    SaltV2Package, SaltV2PackageReader, SaltV2Plane, SaltV2Tensor, SaltV2Tile, SaltV2Transform,
    write_salt_v2_package,
};
use tritium_testkit::{ConformanceVector, Tolerance, generate_vectors, run_conformance};

/// The full conformance set this kernel is responsible for: every TQ2_0 vector
/// from the committed generator (the kernel does not handle TQ1_0).
fn tq2_vectors() -> Vec<ConformanceVector> {
    let v: Vec<_> = generate_vectors(0xC0FFEE, 16)
        .into_iter()
        .filter(|v| v.format == TernaryFormat::Tq2_0)
        .collect();
    assert!(!v.is_empty(), "expected some tq2_0 conformance vectors");
    v
}

#[test]
fn cuda_driver_major_parses_driver_version() {
    assert_eq!(cuda_driver_major(13_030), Some(13));
    assert_eq!(cuda_driver_major(14_000), Some(14));
    assert_eq!(cuda_driver_major(0), None);
}

#[test]
fn physical_device_id_matches_nvidia_uuid_spelling() {
    let bytes = [
        0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
        0x88,
    ];
    assert_eq!(
        format_cuda_physical_id(3, bytes),
        "cuda:3:GPU-12345678-9abc-def0-1122-334455667788"
    );
}

/// Deterministic xorshift f32 fill in `[lo, hi)` — no `rand` dep.
fn seeded_f32(seed: u64, len: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            lo + (s % 1000) as f32 / 1000.0 * (hi - lo)
        })
        .collect()
}

fn salt_v2_test_plane(logical_len: usize, plane_index: usize) -> SaltV2Plane {
    let trits = (0..logical_len)
        .map(|index| {
            let group = index / 4;
            let slot = index % 4;
            let zero_slot = (group + plane_index) % 4;
            if slot == zero_slot {
                0
            } else if (index + plane_index).is_multiple_of(2) {
                1
            } else {
                -1
            }
        })
        .collect();
    let scales = (0..logical_len.div_ceil(SALT_V2_SCALE_GROUP_SIZE))
        .map(|group| f16::from_f32(0.25 + plane_index as f32 * 0.125 + group as f32 * 0.0625))
        .collect();
    SaltV2Plane::new(trits, scales).expect("valid SALT V2 test plane")
}

fn salt_v2_test_tensor(rows: usize, columns: usize, plane_counts: &[usize]) -> SaltV2Tensor {
    let coefficients = rows * columns;
    let tile_count = coefficients.div_ceil(SALT_V2_ALLOCATION_TILE_SIZE);
    assert_eq!(plane_counts.len(), tile_count);
    let tiles = plane_counts
        .iter()
        .copied()
        .enumerate()
        .map(|(tile_index, plane_count)| {
            let start = tile_index * SALT_V2_ALLOCATION_TILE_SIZE;
            let logical_len = (coefficients - start).min(SALT_V2_ALLOCATION_TILE_SIZE);
            SaltV2Tile::new(
                (0..plane_count)
                    .map(|plane_index| salt_v2_test_plane(logical_len, plane_index))
                    .collect(),
            )
            .expect("valid SALT V2 test tile")
        })
        .collect();
    SaltV2Tensor::new(
        format!("salt-v2-{rows}x{columns}"),
        vec![rows as u64, columns as u64],
        tiles,
    )
    .expect("valid SALT V2 test tensor")
}

fn salt_v2_dense_matmul(tensor: &SaltV2Tensor, activation: &[f32], m: usize) -> Vec<f32> {
    let rows = tensor.dims()[0] as usize;
    let columns = tensor.dims()[1] as usize;
    let mut dense = vec![0.0f32; rows * columns];
    for (index, weight) in dense.iter_mut().enumerate() {
        let tile_index = index / SALT_V2_ALLOCATION_TILE_SIZE;
        let local_index = index % SALT_V2_ALLOCATION_TILE_SIZE;
        for plane in tensor.tiles()[tile_index].planes() {
            *weight += plane.trits()[local_index].get() as f32
                * plane.scales()[local_index / SALT_V2_SCALE_GROUP_SIZE].to_f32();
        }
    }
    let mut output = vec![0.0f32; m * rows];
    for mi in 0..m {
        for row in 0..rows {
            let mut sum = 0.0f32;
            for column in 0..columns {
                sum += activation[mi * columns + column] * dense[row * columns + column];
            }
            output[mi * rows + row] = sum;
        }
    }
    output
}

fn salt_v2_dense_weights(tensor: &SaltV2Tensor) -> Vec<f32> {
    let rows = tensor.dims()[0] as usize;
    let columns = tensor.dims()[1] as usize;
    let mut dense = vec![0.0f32; rows * columns];
    for (index, weight) in dense.iter_mut().enumerate() {
        let tile_index = index / SALT_V2_ALLOCATION_TILE_SIZE;
        let local_index = index % SALT_V2_ALLOCATION_TILE_SIZE;
        for plane in tensor.tiles()[tile_index].planes() {
            *weight += plane.trits()[local_index].get() as f32
                * plane.scales()[local_index / SALT_V2_SCALE_GROUP_SIZE].to_f32();
        }
    }
    dense
}

#[test]
fn backend_creation_restores_callers_current_context() {
    if result::init().is_err() {
        eprintln!("skipping context-restoration gate: CUDA driver unavailable");
        return;
    }
    let before = result::ctx::get_current().expect("query context before backend construction");
    let backend = match CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            eprintln!("skipping context-restoration gate: no device ({error})");
            return;
        }
    };
    let after = result::ctx::get_current().expect("query context after backend construction");
    assert_eq!(after, before);
    drop(backend);
}

/// Plan 0043 model-loader seam: a caller-owned host output can be reused across
/// SALT V2 projections. The returned receipt remains identical to the allocating
/// API and no dense weight allocation is introduced.
#[test]
fn salt_v2_cuda_exact_forward_into_matches_allocating_api() {
    let cuda = match CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            eprintln!("skipping SALT V2 CUDA forward-into parity: no device ({error})");
            return;
        }
    };
    let tensor = salt_v2_test_tensor(3, 173, &[1, 3, 2]);
    let activation = seeded_f32(0xF012_1A70, 2 * 173, -0.75, 0.75);

    for codec in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
        let resident = cuda
            .upload_salt_v2(&tensor, codec)
            .expect("upload semantic SALT V2 tensor");
        let allocating = cuda
            .salt_v2_forward_exact(&resident, &activation, 2)
            .expect("allocating exact forward");
        let mut output = vec![f32::NAN; allocating.output.len()];
        let receipt = cuda
            .salt_v2_forward_exact_into(&resident, &activation, 2, &mut output)
            .expect("caller-owned exact forward");

        assert_eq!(output, allocating.output, "{codec:?} forward-into parity");
        assert_eq!(receipt, allocating.receipt);
        assert_eq!(receipt.dense_weight_bytes(), 0);
        assert_eq!(
            resident.allocation_receipt(),
            receipt.resident_allocation(),
            "forward must retain only the original encoded allocation"
        );

        let mut wrong = vec![0.0; output.len() - 1];
        assert!(matches!(
            cuda.salt_v2_forward_exact_into(&resident, &activation, 2, &mut wrong),
            Err(BackendError::ShapeMismatch { .. })
        ));
    }
}

/// Destructive hardware gate: launch a real device trap and prove host output
/// remains unpublished. Isolated test process must exit afterward because CUDA
/// documents fatal device exceptions as sticky context failures.
#[test]
#[cfg(feature = "device-loss-qualification")]
#[ignore = "destructively poisons this test process CUDA context"]
fn destructive_context_loss_qualification_observes_driver_failure() {
    let cuda = CudaBackend::new(0).expect("qualification requires a CUDA device");
    let tensor = salt_v2_test_tensor(1, 4, &[1]);
    let resident = cuda
        .upload_salt_v2(&tensor, SaltV2Codec::D2)
        .expect("upload qualification tensor");
    let sentinel = f32::from_bits(0x3f12_3456);
    let mut output = [sentinel];

    assert!(request_destructive_context_loss_for_qualification());
    let error = cuda
        .salt_v2_forward_exact_into(&resident, &[1.0; 4], 1, &mut output)
        .expect_err("device trap must surface as a CUDA driver failure");
    assert!(
        matches!(
            error,
            BackendError::Backend(ref message)
                if message.starts_with(
                    "destructive CUDA context-loss qualification observed sticky driver failure:"
                )
        ),
        "unexpected qualification error: {error}"
    );
    assert_eq!(output[0].to_bits(), sentinel.to_bits());
}

#[test]
fn salt_v2_cuda_forward_into_is_transactional_on_nonfinite_result() {
    let cuda = match CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            eprintln!("skipping SALT V2 CUDA forward-into transaction gate: no device ({error})");
            return;
        }
    };
    let plane = SaltV2Plane::new(vec![1, 1], vec![f16::ONE]).expect("valid overflow plane");
    let tile = SaltV2Tile::new(vec![plane]).expect("valid overflow tile");
    let tensor =
        SaltV2Tensor::new("overflow", vec![1, 2], vec![tile]).expect("valid overflow tensor");
    let resident = cuda
        .upload_salt_v2(&tensor, SaltV2Codec::D2)
        .expect("upload overflow tensor");
    let sentinel = f32::from_bits(0x3f12_3456);
    let mut output = [sentinel];

    let error = cuda
        .salt_v2_forward_exact_into(&resident, &[f32::MAX, f32::MAX], 1, &mut output)
        .expect_err("non-finite result must be rejected");
    assert!(matches!(error, BackendError::InvalidInput(_)));
    assert_eq!(output[0].to_bits(), sentinel.to_bits());
    assert!(
        cuda.salt_v2_forward_exact(&resident, &[f32::MAX, f32::MAX], 1)
            .is_err(),
        "allocating and caller-owned APIs must share rejection semantics"
    );
}

/// Token embeddings need selected semantic rows without ever materializing the
/// full dense table. Repeated IDs preserve order and duplicate the exact row;
/// all three physical codecs must reconstruct the same semantic values.
#[test]
fn salt_v2_cuda_gathers_repeated_rows_without_dense_shadow() {
    let cuda = match CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            eprintln!("skipping SALT V2 CUDA row gather parity: no device ({error})");
            return;
        }
    };
    let tensor = salt_v2_test_tensor(5, 173, &[1, 3, 2, 3]);
    let dense = salt_v2_dense_weights(&tensor);
    let selected = [4_u32, 1, 4, 0];
    let columns = tensor.dims()[1] as usize;
    let mut expected = Vec::with_capacity(selected.len() * columns);
    for &row in &selected {
        let start = row as usize * columns;
        expected.extend_from_slice(&dense[start..start + columns]);
    }

    for codec in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
        let resident = cuda
            .upload_salt_v2(&tensor, codec)
            .expect("upload semantic SALT V2 embedding");
        let package = SaltV2Package::new(codec, vec![tensor.clone()])
            .expect("codec accepts the embedding tensor");
        let encoded = write_salt_v2_package(&package).expect("encode SALT V2 embedding package");
        let mut reader = SaltV2PackageReader::new_strict(Cursor::new(encoded.bytes))
            .expect("strict embedding package reader");
        let streamed = cuda
            .upload_salt_v2_from_reader(&mut reader, tensor.name())
            .expect("stream SALT V2 embedding from package");
        let allocation_before = resident.allocation_receipt();
        let mut output = vec![f32::NAN; expected.len()];
        let receipt = cuda
            .salt_v2_gather_rows(&resident, &selected, &mut output)
            .expect("gather selected SALT V2 rows");
        let mut streamed_output = vec![f32::NAN; expected.len()];
        let streamed_receipt = cuda
            .salt_v2_gather_rows(&streamed, &selected, &mut streamed_output)
            .expect("gather selected streamed SALT V2 rows");

        for (index, (&got, &want)) in output.iter().zip(&expected).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{codec:?} gathered coefficient {index}: {got} != {want}"
            );
        }
        assert_eq!(streamed_output, output, "{codec:?} streamed gather parity");
        assert_eq!(streamed_receipt, receipt);
        assert_eq!(&output[..columns], &output[2 * columns..3 * columns]);
        assert_eq!(receipt.resident_allocation(), allocation_before);
        assert_eq!(resident.allocation_receipt(), allocation_before);
        assert_eq!(receipt.row_index_bytes(), selected.len() as u64 * 4);
        assert_eq!(
            receipt.output_bytes(),
            expected.len() as u64 * core::mem::size_of::<f32>() as u64
        );
        assert_eq!(receipt.dense_weight_bytes(), 0);
        assert_eq!(
            receipt.peak_resident_bytes(),
            allocation_before.steady_resident_bytes()
                + receipt.row_index_bytes()
                + receipt.output_bytes()
        );
    }
}

#[test]
fn salt_v2_cuda_gather_rejects_oov_before_touching_output() {
    let cuda = match CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            eprintln!("skipping SALT V2 CUDA row gather OOV gate: no device ({error})");
            return;
        }
    };
    let tensor = salt_v2_test_tensor(3, 173, &[1, 3, 2]);
    let resident = cuda
        .upload_salt_v2(&tensor, SaltV2Codec::D2)
        .expect("upload semantic SALT V2 embedding");
    let sentinel = 0x7fc0_1234_u32;
    let mut output = vec![f32::from_bits(sentinel); 2 * 173];

    let error = cuda
        .salt_v2_gather_rows(&resident, &[1, 3], &mut output)
        .expect_err("row equal to vocab size must be rejected");
    assert!(matches!(error, BackendError::InvalidInput(_)));
    assert!(output.iter().all(|value| value.to_bits() == sentinel));
}

#[test]
fn salt_v2_cuda_gather_crosses_rank_prefix_boundary() {
    let cuda = match CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            eprintln!("skipping SALT V2 CUDA row gather rank-prefix gate: no device ({error})");
            return;
        }
    };
    let plane_counts = (0..258).map(|tile| 1 + tile % 3).collect::<Vec<_>>();
    let tensor = salt_v2_test_tensor(258, SALT_V2_ALLOCATION_TILE_SIZE, &plane_counts);
    let dense = salt_v2_dense_weights(&tensor);
    let selected = [256_u32, 257, 0];
    let columns = SALT_V2_ALLOCATION_TILE_SIZE;
    let mut expected = Vec::with_capacity(selected.len() * columns);
    for &row in &selected {
        let start = row as usize * columns;
        expected.extend_from_slice(&dense[start..start + columns]);
    }

    for codec in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
        let resident = cuda
            .upload_salt_v2(&tensor, codec)
            .expect("upload rank-prefix SALT V2 embedding");
        let mut output = vec![f32::NAN; expected.len()];
        cuda.salt_v2_gather_rows(&resident, &selected, &mut output)
            .expect("gather rows across rank-prefix boundary");
        for (index, (&got, &want)) in output.iter().zip(&expected).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{codec:?} rank-prefix coefficient {index}: {got} != {want}"
            );
        }
    }
}

/// Plan 0043 Stage 6: every admitted SALT V2 codec executes directly from its
/// resident encoded payload. Mixed P=1/3/2 tiles, row-crossing macrotiles, and
/// one-/three-column matrices cover the descriptor and codec-tail boundaries;
/// the 1025-tile case crosses repeated upload-staging and rank-prefix boundaries.
/// The fast entry point is intentionally an exact-kernel alias in this first
/// correctness slice and its receipt must say so.
#[test]
fn salt_v2_cuda_matches_cpu_and_dense_without_dense_weight_storage() {
    let cuda = match CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            eprintln!("skipping SALT V2 CUDA parity: no device ({error})");
            return;
        }
    };

    let long_plane_counts = (0..1025).map(|index| 1 + index % 3).collect::<Vec<_>>();
    let cases = vec![
        salt_v2_test_tensor(3, 173, &[1, 3, 2]),
        salt_v2_test_tensor(2, 3, &[3]),
        salt_v2_test_tensor(1, 1, &[1]),
        salt_v2_test_tensor(1, 1025 * SALT_V2_ALLOCATION_TILE_SIZE, &long_plane_counts),
    ];
    for codec in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
        for tensor in &cases {
            let rows = tensor.dims()[0] as usize;
            let columns = tensor.dims()[1] as usize;
            let m = 2;
            let activation = seeded_f32(
                0x5A17 + rows as u64 * 257 + columns as u64,
                m * columns,
                -0.75,
                0.75,
            );
            let package = SaltV2Package::new(codec, vec![tensor.clone()])
                .expect("codec accepts the test tensor");
            let resident = cuda
                .upload_salt_v2(tensor, codec)
                .expect("upload semantic SALT V2 tensor");
            let encoded = write_salt_v2_package(&package).expect("encode SALT V2 package");
            let mut reader = SaltV2PackageReader::new_strict(Cursor::new(encoded.bytes))
                .expect("strict seek reader");
            let streamed = cuda
                .upload_salt_v2_from_reader(&mut reader, tensor.name())
                .expect("stream SALT V2 tensor from package");

            let exact = cuda
                .salt_v2_forward_exact(&resident, &activation, m)
                .expect("exact SALT V2 forward");
            let fast = cuda
                .salt_v2_forward_fast(&resident, &activation, m)
                .expect("fast SALT V2 forward");
            let streamed_exact = cuda
                .salt_v2_forward_exact(&streamed, &activation, m)
                .expect("streamed exact SALT V2 forward");
            let dense = salt_v2_dense_matmul(tensor, &activation, m);
            let mut cpu = Vec::with_capacity(m * rows);
            for mi in 0..m {
                cpu.extend(
                    salt_v2_matvec(&package, 0, &activation[mi * columns..(mi + 1) * columns])
                        .expect("CPU SALT V2 oracle")
                        .output,
                );
            }

            assert_eq!(exact.output.len(), m * rows);
            for (index, ((got, cpu_want), dense_want)) in
                exact.output.iter().zip(&cpu).zip(&dense).enumerate()
            {
                assert_eq!(
                    got.to_bits(),
                    cpu_want.to_bits(),
                    "exact[{index}] {codec:?} {rows}x{columns}: GPU {got} vs CPU {cpu_want}"
                );
                assert!(
                    Tolerance::relative(1e-4).accepts(*got, *dense_want),
                    "exact[{index}] {codec:?} {rows}x{columns}: GPU {got} vs dense {dense_want}"
                );
            }
            assert_eq!(fast.output, exact.output);
            assert_eq!(streamed_exact.output, exact.output);
            assert_eq!(exact.receipt.mode(), SaltV2ForwardMode::Exact);
            assert_eq!(fast.receipt.mode(), SaltV2ForwardMode::FastAliasesExact);

            let allocation = resident.allocation_receipt();
            assert_eq!(streamed.allocation_receipt(), allocation);
            let planned = SaltV2IndexedRuntimeLedger::for_tensor(tensor, codec)
                .expect("shared indexed-runtime plan");
            assert_eq!(allocation.runtime_ledger(), planned);
            let expected_payload_bytes: usize = tensor
                .tiles()
                .iter()
                .flat_map(|tile| tile.planes())
                .map(|plane| {
                    let logical_len = if codec == SaltV2Codec::S34 {
                        plane.trits().len().div_ceil(4) * 4
                    } else {
                        plane.trits().len()
                    };
                    codec
                        .ledger(logical_len)
                        .expect("test payload length is representable")
                        .physical_bytes
                })
                .sum();
            let expected_scale_bytes: usize = tensor
                .tiles()
                .iter()
                .flat_map(|tile| tile.planes())
                .map(|plane| plane.scales().len() * core::mem::size_of::<u16>())
                .sum();
            assert_eq!(allocation.payload_bytes(), expected_payload_bytes as u64);
            assert_eq!(allocation.scale_bytes(), expected_scale_bytes as u64);
            assert_eq!(
                allocation.map_bytes(),
                (tensor.tiles().len() * 2 / 8) as u64
            );
            assert_eq!(
                allocation.rank_prefix_bytes(),
                (tensor.tiles().len().saturating_sub(1) / 256 * 4) as u64
            );
            assert_eq!(
                allocation.allocation_map_bits(),
                (tensor.tiles().len() * 2) as u64
            );
            assert_eq!(
                allocation.allocation_map_embedded_bits(),
                (tensor.tiles().len() * 2 % 8) as u64
            );
            assert_eq!(allocation.dense_weight_bytes(), 0);
            assert_eq!(
                allocation.steady_resident_bytes(),
                allocation.payload_bytes()
                    + allocation.scale_bytes()
                    + allocation.map_bytes()
                    + allocation.rank_prefix_bytes()
            );
            assert_eq!(exact.receipt.dense_weight_bytes(), 0);
            assert_eq!(exact.receipt.resident_allocation(), allocation);
            assert_eq!(
                exact.receipt.steady_resident_bytes(),
                allocation.steady_resident_bytes()
            );
            assert_eq!(
                exact.receipt.peak_resident_bytes(),
                allocation.steady_resident_bytes()
                    + (activation.len() * core::mem::size_of::<f32>()) as u64
                    + (m * rows * core::mem::size_of::<f32>()) as u64
            );
        }
    }
}

/// ADR 0028 native arithmetic reduces add/sub/skip activation contributions
/// before applying a plane/group scale. Reconstructing each dense coefficient
/// first would overflow both products in this case instead of cancelling to 0.
#[test]
fn salt_v2_cuda_cancels_a_plane_group_before_scaling() {
    let cuda = match CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            eprintln!("skipping SALT V2 cancellation parity: no device ({error})");
            return;
        }
    };
    let plane = SaltV2Plane::new(vec![1, -1], vec![f16::MAX]).expect("valid cancellation plane");
    let tensor = SaltV2Tensor::new(
        "cancellation",
        vec![1, 2],
        vec![SaltV2Tile::new(vec![plane]).expect("valid cancellation tile")],
    )
    .expect("valid cancellation tensor");
    let activation = [f32::MAX, f32::MAX];

    for codec in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
        let package = SaltV2Package::new(codec, vec![tensor.clone()])
            .expect("codec accepts cancellation tensor");
        assert_eq!(
            salt_v2_matvec(&package, 0, &activation)
                .expect("CPU native cancellation")
                .output,
            [0.0]
        );
        let resident = cuda
            .upload_salt_v2(&tensor, codec)
            .expect("upload cancellation tensor");
        let exact = cuda
            .salt_v2_forward_exact(&resident, &activation, 1)
            .expect("GPU native cancellation");
        assert_eq!(exact.output, [0.0], "codec {codec:?}");
    }
}

/// Signed RHT identity is serialized today but native CUDA activation transform
/// execution is not. Upload must reject it before allocating a misleading
/// untransformed resident tensor.
#[test]
fn salt_v2_cuda_rejects_unimplemented_tensor_transform() {
    let cuda = match CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            eprintln!("skipping SALT V2 transform rejection: no device ({error})");
            return;
        }
    };
    let tensor = SaltV2Tensor::new_with_transform(
        "rotated",
        vec![1, 1],
        SaltV2Transform::SignedRht {
            seed: 7,
            domain: 11,
        },
        vec![SaltV2Tile::new(vec![salt_v2_test_plane(1, 0)]).expect("valid tile")],
    )
    .expect("valid transformed tensor");
    let error = cuda
        .upload_salt_v2(&tensor, SaltV2Codec::D2)
        .expect_err("native CUDA must not ignore SignedRht");
    match error {
        BackendError::InvalidInput(message) => {
            assert!(
                message.contains("transform"),
                "unexpected message: {message}"
            );
            assert!(
                message.contains("SignedRht"),
                "unexpected message: {message}"
            );
        }
        other => panic!("expected InvalidInput, got {other}"),
    }

    let package = SaltV2Package::new(SaltV2Codec::D2, vec![tensor]).expect("valid package");
    let encoded = write_salt_v2_package(&package).expect("encode transformed package");
    let mut reader = SaltV2PackageReader::new_strict(Cursor::new(encoded.bytes))
        .expect("strict transformed package");
    let error = cuda
        .upload_salt_v2_from_reader(&mut reader, "rotated")
        .expect_err("streamed CUDA upload must not ignore SignedRht");
    assert!(
        matches!(error, BackendError::InvalidInput(ref message) if message.contains("SignedRht")),
        "unexpected streamed transform error: {error}"
    );
}

/// Gate C on CUDA (ADR 0007): the f32 ternary-matmul backward kernels match the
/// `tritium-train` CPU `vjp` oracle within the IMMA `1e-4` bar, across square and
/// tail shapes. Self-skips when no GPU is present.
#[test]
fn train_backward_matches_cpu_vjp() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping train backward parity: no device ({e})");
            return;
        }
    };
    let tol = Tolerance::relative(1e-4);
    // square + tail shapes (non-multiples of the 256-thread block). The (2,300,3)
    // case pushes N past 256 so grad_s's own grid spans >1 block (blockIdx.x>0).
    let shapes = [
        (3, 4, 5),
        (1, 1, 7),
        (2, 3, 4),
        (8, 16, 32),
        (16, 8, 33),
        (5, 7, 1),
        (2, 300, 3),
    ];
    for (m, n, k) in shapes {
        let act = seeded_f32(1, m * k, -2.0, 2.0);
        // Real-valued (fractional) weights exercise the general contraction the
        // autograd surrogate path uses; ternary is the special case it subsumes.
        let w = seeded_f32(2, n * k, -1.0, 1.0);
        let s = seeded_f32(3, n, 0.1, 2.0);
        let gy = seeded_f32(4, m * n, -1.5, 1.5);

        // CPU oracle: vjp -> [gA, gW, gs].
        let cpu = tritium_train::ops::matmul::vjp(&act, &w, &s, m, n, k, &gy);
        let shape = GemmShape::new(m, n, k);

        let mut ga = vec![0.0f32; m * k];
        cuda.grad_a(&gy, &w, &s, shape, &mut ga).expect("grad_a");
        let mut gw = vec![0.0f32; n * k];
        cuda.grad_w(&gy, &act, &s, shape, &mut gw).expect("grad_w");
        let mut gs = vec![0.0f32; n];
        cuda.grad_s(&gy, &act, &w, shape, &mut gs).expect("grad_s");

        for (i, (&g, &c)) in ga.iter().zip(&cpu[0]).enumerate() {
            assert!(
                tol.accepts(g, c),
                "grad_a[{i}] {m}x{n}x{k}: gpu {g} vs cpu {c}"
            );
        }
        for (i, (&g, &c)) in gw.iter().zip(&cpu[1]).enumerate() {
            assert!(
                tol.accepts(g, c),
                "grad_w[{i}] {m}x{n}x{k}: gpu {g} vs cpu {c}"
            );
        }
        for (i, (&g, &c)) in gs.iter().zip(&cpu[2]).enumerate() {
            assert!(
                tol.accepts(g, c),
                "grad_s[{i}] {m}x{n}x{k}: gpu {g} vs cpu {c}"
            );
        }
    }
}

/// Plan 0046: packed backend-neutral VJP must consume the existing TQ2_0
/// allocation and match CPU without materializing a dense CUDA weight.
#[test]
fn packed_projected_vjp_matches_cpu_backend() {
    let cuda = match CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            eprintln!("skipping packed projected VJP parity: no device ({error})");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let tolerance = Tolerance::relative(1e-4);
    for (case, (m, n, k)) in [(2, 3, 5), (7, 4, 257), (1, 33, 64)]
        .into_iter()
        .enumerate()
    {
        let shape = GemmShape::new(m, n, k);
        let trits = (0..n * k)
            .map(|index| match (index + case) % 3 {
                0 => tritium_core::Trit::NEG,
                1 => tritium_core::Trit::ZERO,
                _ => tritium_core::Trit::POS,
            })
            .collect::<Vec<_>>();
        let packed = pack_tq2_0(&trits, shape);
        let cpu_weights = cpu
            .upload_weights(&packed, shape, TernaryFormat::Tq2_0)
            .expect("CPU upload");
        let cuda_weights = cuda
            .upload_weights(&packed, shape, TernaryFormat::Tq2_0)
            .expect("CUDA upload");
        let act = seeded_f32(11 + case as u64, m * k, -2.0, 2.0);
        let scales = seeded_f32(21 + case as u64, n, 0.05, 1.5);
        let grad_output = seeded_f32(31 + case as u64, m * n, -1.0, 1.0);
        let mut cpu_grad_act = vec![f32::NAN; m * k];
        let mut cpu_grad_weight = vec![f32::NAN; n * k];
        let mut cpu_grad_bias = vec![f32::NAN; n];
        let mut cuda_grad_act = vec![f32::NAN; m * k];
        let mut cuda_grad_weight = vec![f32::NAN; n * k];
        let mut cuda_grad_bias = vec![f32::NAN; n];

        cpu.mpgemm_projected_vjp(MpGemmProjectedVjp {
            act: &act,
            weights: cpu_weights.as_ref(),
            scales: &scales,
            grad_output: &grad_output,
            shape,
            format: TernaryFormat::Tq2_0,
            grad_act: &mut cpu_grad_act,
            grad_projected_weight: &mut cpu_grad_weight,
            grad_bias: Some(&mut cpu_grad_bias),
        })
        .expect("CPU projected VJP");
        cuda.mpgemm_projected_vjp(MpGemmProjectedVjp {
            act: &act,
            weights: cuda_weights.as_ref(),
            scales: &scales,
            grad_output: &grad_output,
            shape,
            format: TernaryFormat::Tq2_0,
            grad_act: &mut cuda_grad_act,
            grad_projected_weight: &mut cuda_grad_weight,
            grad_bias: Some(&mut cuda_grad_bias),
        })
        .expect("CUDA projected VJP");

        for (label, expected, actual) in [
            ("grad_act", &cpu_grad_act, &cuda_grad_act),
            ("grad_projected_weight", &cpu_grad_weight, &cuda_grad_weight),
            ("grad_bias", &cpu_grad_bias, &cuda_grad_bias),
        ] {
            for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
                assert!(
                    tolerance.accepts(actual, expected),
                    "case {case} {label}[{index}]: CUDA {actual} vs CPU {expected}"
                );
            }
        }
    }
}

/// ADR 0027 Track D: the compact training-specific SALT planes must preserve
/// Track A's per-row greedy quantizer while eliminating the dense quantized
/// weight. Exercise every supported plane count, TQ2 tails, and the K>8192
/// fallback-sized regime for both forward and activation-gradient contractions.
#[test]
fn packed_training_salt_matches_dense_resident_oracle() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping packed training SALT parity: no device ({e})");
            return;
        }
    };
    let tol = Tolerance::relative(1e-4);
    // M/N are deliberately not tile multiples. K=7 exercises the fast twin's
    // scalar fallback; larger K values dispatch its tiled kernels, including
    // the 8193-column regime and its final one-element reduction tail.
    let (m, n) = (17usize, 35usize);

    for k in [7usize, 257, 576, 8193] {
        let mut master = seeded_f32(0x5100 + k as u64, n * k, -1.25, 1.25);
        // An exact-zero row gates the zero-scale/code path. The other rows and
        // all tail shapes retain mixed signs and non-integral residuals.
        master[..k].fill(0.0);
        let act = seeded_f32(0xA000 + k as u64, m * k, -0.75, 0.75);
        let gy = seeded_f32(0xB000 + k as u64, m * n, -0.5, 0.5);

        let d_master = cuda.dev_upload(&master).expect("upload master");
        let mut d_residual = cuda.dev_alloc_zeros(n * k).expect("residual scratch");
        let d_act = cuda.dev_upload(&act).expect("upload act");
        let d_gy = cuda.dev_upload(&gy).expect("upload gy");
        let d_ones = cuda.dev_upload(&vec![1.0f32; n]).expect("upload scales");
        let mut d_dense = cuda.dev_alloc_zeros(n * k).expect("alloc dense weight");
        let mut d_dense_y = cuda.dev_alloc_zeros(m * n).expect("alloc dense y");
        let mut d_dense_ga = cuda.dev_alloc_zeros(m * k).expect("alloc dense ga");
        let shape = GemmShape::new(m, n, k);

        for planes in 1..=3 {
            let packed = cuda
                .pack_training_salt(&d_master, &mut d_residual, n, k, planes)
                .expect("pack resident SALT");
            let row_bytes = k.div_ceil(tritium_format::QK_K) * (tritium_format::QK_K / 4);
            assert_eq!(packed.packed_bytes(), planes * n * row_bytes);
            assert_eq!(
                packed.scale_bytes(),
                planes * n * core::mem::size_of::<f32>()
            );
            assert_eq!(
                packed.resident_bytes(),
                packed.packed_bytes() + packed.scale_bytes()
            );

            let dense = tritium_train::ops::ste::salt_quantize_forward(&master, n, k, planes);
            let want_y = tritium_train::ops::dense::forward(&act, &dense, m, n, k);
            let want_ga = tritium_train::ops::dense::vjp(&act, &dense, m, n, k, &gy)[0].clone();

            cuda.salt_quantize_forward_dev(&d_master, &mut d_residual, &mut d_dense, n, k, planes)
                .expect("materialize dense device SALT");
            cuda.matmul_forward_dev(&d_act, &d_dense, &d_ones, shape, &mut d_dense_y)
                .expect("dense device SALT forward");
            cuda.grad_a_dev(&d_gy, &d_dense, &d_ones, shape, &mut d_dense_ga)
                .expect("dense device SALT grad_a");
            let mut dense_device_y = vec![0.0f32; m * n];
            cuda.dev_download(&d_dense_y, &mut dense_device_y)
                .expect("download dense device y");
            let mut dense_device_ga = vec![0.0f32; m * k];
            cuda.dev_download(&d_dense_ga, &mut dense_device_ga)
                .expect("download dense device ga");

            let mut d_y = cuda.dev_alloc_zeros(m * n).expect("alloc y");
            cuda.training_salt_forward(&d_act, &packed, m, &mut d_y)
                .expect("exact packed SALT forward");
            let mut got_y = vec![0.0f32; m * n];
            cuda.dev_download(&d_y, &mut got_y).expect("download y");
            let mut d_exact_scalar_y = cuda.dev_alloc_zeros(m * n).expect("alloc scalar exact y");
            cuda.training_salt_forward_exact_scalar(&d_act, &packed, m, &mut d_exact_scalar_y)
                .expect("scalar exact packed SALT forward");
            let mut exact_scalar_y = vec![0.0f32; m * n];
            cuda.dev_download(&d_exact_scalar_y, &mut exact_scalar_y)
                .expect("download scalar exact y");
            let mut d_fast_y = cuda.dev_alloc_zeros(m * n).expect("alloc fast y");
            cuda.training_salt_forward_fast(&d_act, &packed, m, &mut d_fast_y)
                .expect("fast packed SALT forward");
            let mut fast_y = vec![0.0f32; m * n];
            cuda.dev_download(&d_fast_y, &mut fast_y)
                .expect("download fast y");
            let mut d_scalar_y = cuda.dev_alloc_zeros(m * n).expect("alloc scalar y");
            cuda.training_salt_forward_scalar(&d_act, &packed, m, &mut d_scalar_y)
                .expect("scalar-fast packed SALT forward");
            let mut scalar_y = vec![0.0f32; m * n];
            cuda.dev_download(&d_scalar_y, &mut scalar_y)
                .expect("download scalar y");

            let mut d_ga = cuda.dev_alloc_zeros(m * k).expect("alloc ga");
            cuda.training_salt_grad_a(&d_gy, &packed, m, &mut d_ga)
                .expect("exact packed SALT grad_a");
            let mut got_ga = vec![0.0f32; m * k];
            cuda.dev_download(&d_ga, &mut got_ga).expect("download ga");
            let mut d_exact_scalar_ga = cuda.dev_alloc_zeros(m * k).expect("alloc scalar exact ga");
            cuda.training_salt_grad_a_exact_scalar(&d_gy, &packed, m, &mut d_exact_scalar_ga)
                .expect("scalar exact packed SALT grad_a");
            let mut exact_scalar_ga = vec![0.0f32; m * k];
            cuda.dev_download(&d_exact_scalar_ga, &mut exact_scalar_ga)
                .expect("download scalar exact ga");
            let mut d_fast_ga = cuda.dev_alloc_zeros(m * k).expect("alloc fast ga");
            cuda.training_salt_grad_a_fast(&d_gy, &packed, m, &mut d_fast_ga)
                .expect("fast packed SALT grad_a");
            let mut fast_ga = vec![0.0f32; m * k];
            cuda.dev_download(&d_fast_ga, &mut fast_ga)
                .expect("download fast ga");
            let mut d_scalar_ga = cuda.dev_alloc_zeros(m * k).expect("alloc scalar ga");
            cuda.training_salt_grad_a_scalar(&d_gy, &packed, m, &mut d_scalar_ga)
                .expect("scalar-fast packed SALT grad_a");
            let mut scalar_ga = vec![0.0f32; m * k];
            cuda.dev_download(&d_scalar_ga, &mut scalar_ga)
                .expect("download scalar ga");

            assert_eq!(
                CudaBackend::training_salt_forward_tiled_supported(m, n, k),
                k >= 256
            );
            assert_eq!(
                CudaBackend::training_salt_grad_a_tiled_supported(m, n, k),
                k >= 128
            );
            assert_eq!(
                CudaBackend::training_salt_forward_exact_tiled_supported(m, n, k),
                k >= 32
            );
            assert_eq!(
                CudaBackend::training_salt_grad_a_exact_tiled_supported(m, n, k),
                k >= 32
            );

            for (i, (&got, &want)) in got_y.iter().zip(&want_y).enumerate() {
                assert!(
                    tol.accepts(got, want),
                    "forward[{i}] T={planes} {m}x{n}x{k}: exact {got} vs CPU dense {want}"
                );
                assert_eq!(
                    got.to_bits(),
                    dense_device_y[i].to_bits(),
                    "forward[{i}] T={planes} {m}x{n}x{k}: exact {got} vs device dense {}",
                    dense_device_y[i],
                );
                assert_eq!(
                    got.to_bits(),
                    exact_scalar_y[i].to_bits(),
                    "forward[{i}] T={planes} {m}x{n}x{k}: default exact {got} vs scalar exact {}",
                    exact_scalar_y[i],
                );
                assert!(
                    tol.accepts(fast_y[i], want),
                    "forward[{i}] T={planes} {m}x{n}x{k}: fast {} vs CPU dense {want}",
                    fast_y[i],
                );
                assert_eq!(
                    fast_y[i].to_bits(),
                    scalar_y[i].to_bits(),
                    "forward[{i}] T={planes} {m}x{n}x{k}: tiled-fast {} vs scalar-fast {}",
                    fast_y[i],
                    scalar_y[i],
                );
            }
            for (i, (&got, &want)) in got_ga.iter().zip(&want_ga).enumerate() {
                assert!(
                    tol.accepts(got, want),
                    "grad_a[{i}] T={planes} {m}x{n}x{k}: exact {got} vs CPU dense {want}"
                );
                assert_eq!(
                    got.to_bits(),
                    dense_device_ga[i].to_bits(),
                    "grad_a[{i}] T={planes} {m}x{n}x{k}: exact {got} vs device dense {}",
                    dense_device_ga[i],
                );
                assert_eq!(
                    got.to_bits(),
                    exact_scalar_ga[i].to_bits(),
                    "grad_a[{i}] T={planes} {m}x{n}x{k}: default exact {got} vs scalar exact {}",
                    exact_scalar_ga[i],
                );
                assert!(
                    tol.accepts(fast_ga[i], want),
                    "grad_a[{i}] T={planes} {m}x{n}x{k}: fast {} vs CPU dense {want}",
                    fast_ga[i],
                );
                assert_eq!(
                    fast_ga[i].to_bits(),
                    scalar_ga[i].to_bits(),
                    "grad_a[{i}] T={planes} {m}x{n}x{k}: tiled-fast {} vs scalar-fast {}",
                    fast_ga[i],
                    scalar_ga[i],
                );
            }
        }
    }

    let d_master = cuda.dev_upload(&[1.0f32; 8]).unwrap();
    let mut d_residual = cuda.dev_alloc_zeros(8).unwrap();
    assert!(matches!(
        cuda.pack_training_salt(&d_master, &mut d_residual, 2, 4, 0),
        Err(BackendError::InvalidInput(_))
    ));
    let packed = cuda
        .pack_training_salt(&d_master, &mut d_residual, 2, 4, 1)
        .unwrap();
    let short_act = cuda.dev_upload(&[1.0f32; 3]).unwrap();
    let mut out = cuda.dev_alloc_zeros(2).unwrap();
    assert!(matches!(
        cuda.training_salt_forward(&short_act, &packed, 1, &mut out),
        Err(BackendError::ShapeMismatch { .. })
    ));
    // Zero output rows are a no-launch success and leave caller storage intact.
    let mut sentinel = cuda.dev_upload(&[17.0f32]).unwrap();
    cuda.training_salt_forward(&short_act, &packed, 0, &mut sentinel)
        .unwrap();
    cuda.training_salt_grad_a(&out, &packed, 0, &mut sentinel)
        .unwrap();
    let mut got_sentinel = [0.0f32];
    cuda.dev_download(&sentinel, &mut got_sentinel).unwrap();
    assert_eq!(got_sentinel, [17.0]);
}

/// The exact grad-A tile reduces N in 64-wide chunks. Cross the boundary so
/// the parity gate covers accumulation across two shared-weight loads and the
/// barrier that protects tile reuse.
#[test]
fn packed_training_salt_exact_grad_a_crosses_n_tiles_bitwise() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping packed SALT cross-tile parity: no device ({e})");
            return;
        }
    };
    let (m, n, k) = (17usize, 67usize, 257usize);
    let master = seeded_f32(0xC055, n * k, -1.25, 1.25);
    let gy = seeded_f32(0x6A71, m * n, -0.5, 0.5);
    let d_master = cuda.dev_upload(&master).expect("upload master");
    let d_gy = cuda.dev_upload(&gy).expect("upload gy");
    let d_ones = cuda.dev_upload(&vec![1.0f32; n]).expect("upload scales");
    let mut d_residual = cuda.dev_alloc_zeros(n * k).expect("residual scratch");
    let mut d_dense = cuda.dev_alloc_zeros(n * k).expect("alloc dense weight");
    let mut d_dense_ga = cuda.dev_alloc_zeros(m * k).expect("alloc dense grad-A");
    let mut d_exact_ga = cuda
        .dev_alloc_zeros(m * k)
        .expect("alloc tiled exact grad-A");
    let mut d_scalar_ga = cuda
        .dev_alloc_zeros(m * k)
        .expect("alloc scalar exact grad-A");

    assert!(CudaBackend::training_salt_grad_a_exact_tiled_supported(
        m, n, k
    ));
    for planes in 1..=3 {
        let packed = cuda
            .pack_training_salt(&d_master, &mut d_residual, n, k, planes)
            .expect("pack resident SALT");
        cuda.salt_quantize_forward_dev(&d_master, &mut d_residual, &mut d_dense, n, k, planes)
            .expect("materialize dense device SALT");
        cuda.grad_a_dev(
            &d_gy,
            &d_dense,
            &d_ones,
            GemmShape::new(m, n, k),
            &mut d_dense_ga,
        )
        .expect("dense device grad-A");
        cuda.training_salt_grad_a(&d_gy, &packed, m, &mut d_exact_ga)
            .expect("tiled exact packed grad-A");
        cuda.training_salt_grad_a_exact_scalar(&d_gy, &packed, m, &mut d_scalar_ga)
            .expect("scalar exact packed grad-A");

        let mut dense = vec![0.0f32; m * k];
        let mut tiled = vec![0.0f32; m * k];
        let mut scalar = vec![0.0f32; m * k];
        cuda.dev_download(&d_dense_ga, &mut dense)
            .expect("download dense grad-A");
        cuda.dev_download(&d_exact_ga, &mut tiled)
            .expect("download tiled grad-A");
        cuda.dev_download(&d_scalar_ga, &mut scalar)
            .expect("download scalar grad-A");
        for (i, ((&got, &want), &oracle)) in tiled.iter().zip(&dense).zip(&scalar).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "grad_a[{i}] T={planes}: tiled exact {got} vs device dense {want}"
            );
            assert_eq!(
                got.to_bits(),
                oracle.to_bits(),
                "grad_a[{i}] T={planes}: tiled exact {got} vs scalar exact {oracle}"
            );
        }
    }
}

/// Manual Track D microbenchmark. It times requantize plus exact and fast
/// forward, activation gradient, and their combined paths with fixed resident
/// allocations, then prints latency and weight bytes. Hardware-sensitive, so
/// correctness is gated above while this remains opt-in evidence
/// (`--ignored --nocapture`).
#[test]
#[ignore = "4090 Track D performance probe"]
fn bench_packed_training_salt_vs_dense_materialization() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping packed training SALT bench: no device ({e})");
            return;
        }
    };
    let (m, n, k, planes) = (32usize, 576usize, 576usize, 3usize);
    let iters = 100u32;
    let master = seeded_f32(0x5A17, n * k, -1.25, 1.25);
    let act = seeded_f32(0xAC71, m * k, -0.75, 0.75);
    let gy = seeded_f32(0x6A71, m * n, -0.5, 0.5);
    let d_master = cuda.dev_upload(&master).unwrap();
    let d_act = cuda.dev_upload(&act).unwrap();
    let d_gy = cuda.dev_upload(&gy).unwrap();
    let d_ones = cuda.dev_upload(&vec![1.0f32; n]).unwrap();
    let mut d_residual = cuda.dev_alloc_zeros(n * k).unwrap();
    let mut d_dense = cuda.dev_alloc_zeros(n * k).unwrap();
    let mut d_dense_y = cuda.dev_alloc_zeros(m * n).unwrap();
    let mut d_dense_ga = cuda.dev_alloc_zeros(m * k).unwrap();
    let mut packed = cuda
        .pack_training_salt(&d_master, &mut d_residual, n, k, planes)
        .unwrap();
    let mut d_exact_y = cuda.dev_alloc_zeros(m * n).unwrap();
    let mut d_exact_ga = cuda.dev_alloc_zeros(m * k).unwrap();
    let mut d_exact_scalar_y = cuda.dev_alloc_zeros(m * n).unwrap();
    let mut d_exact_scalar_ga = cuda.dev_alloc_zeros(m * k).unwrap();
    let mut d_packed_y = cuda.dev_alloc_zeros(m * n).unwrap();
    let mut d_packed_ga = cuda.dev_alloc_zeros(m * k).unwrap();
    let mut d_scalar_y = cuda.dev_alloc_zeros(m * n).unwrap();
    let mut d_scalar_ga = cuda.dev_alloc_zeros(m * k).unwrap();
    let shape = GemmShape::new(m, n, k);

    for _ in 0..10 {
        cuda.salt_quantize_forward_dev(&d_master, &mut d_residual, &mut d_dense, n, k, planes)
            .unwrap();
        cuda.matmul_forward_dev(&d_act, &d_dense, &d_ones, shape, &mut d_dense_y)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let dense_start = std::time::Instant::now();
    for _ in 0..iters {
        cuda.salt_quantize_forward_dev(&d_master, &mut d_residual, &mut d_dense, n, k, planes)
            .unwrap();
        cuda.matmul_forward_dev(&d_act, &d_dense, &d_ones, shape, &mut d_dense_y)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let dense_us = dense_start.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

    for _ in 0..10 {
        cuda.repack_training_salt(&d_master, &mut d_residual, &mut packed)
            .unwrap();
        cuda.training_salt_forward_scalar(&d_act, &packed, m, &mut d_scalar_y)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let scalar_start = std::time::Instant::now();
    for _ in 0..iters {
        cuda.repack_training_salt(&d_master, &mut d_residual, &mut packed)
            .unwrap();
        cuda.training_salt_forward_scalar(&d_act, &packed, m, &mut d_scalar_y)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let scalar_us = scalar_start.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

    for _ in 0..10 {
        cuda.repack_training_salt(&d_master, &mut d_residual, &mut packed)
            .unwrap();
        cuda.training_salt_forward_exact_scalar(&d_act, &packed, m, &mut d_exact_scalar_y)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let exact_scalar_start = std::time::Instant::now();
    for _ in 0..iters {
        cuda.repack_training_salt(&d_master, &mut d_residual, &mut packed)
            .unwrap();
        cuda.training_salt_forward_exact_scalar(&d_act, &packed, m, &mut d_exact_scalar_y)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let exact_scalar_us = exact_scalar_start.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

    for _ in 0..10 {
        cuda.repack_training_salt(&d_master, &mut d_residual, &mut packed)
            .unwrap();
        cuda.training_salt_forward(&d_act, &packed, m, &mut d_exact_y)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let exact_start = std::time::Instant::now();
    for _ in 0..iters {
        cuda.repack_training_salt(&d_master, &mut d_residual, &mut packed)
            .unwrap();
        cuda.training_salt_forward(&d_act, &packed, m, &mut d_exact_y)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let exact_us = exact_start.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

    for _ in 0..10 {
        cuda.repack_training_salt(&d_master, &mut d_residual, &mut packed)
            .unwrap();
        cuda.training_salt_forward_fast(&d_act, &packed, m, &mut d_packed_y)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let packed_start = std::time::Instant::now();
    for _ in 0..iters {
        cuda.repack_training_salt(&d_master, &mut d_residual, &mut packed)
            .unwrap();
        cuda.training_salt_forward_fast(&d_act, &packed, m, &mut d_packed_y)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let packed_us = packed_start.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

    for _ in 0..10 {
        cuda.training_salt_grad_a_scalar(&d_gy, &packed, m, &mut d_scalar_ga)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let scalar_grad_start = std::time::Instant::now();
    for _ in 0..iters {
        cuda.training_salt_grad_a_scalar(&d_gy, &packed, m, &mut d_scalar_ga)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let scalar_grad_us = scalar_grad_start.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

    for _ in 0..10 {
        cuda.training_salt_grad_a_exact_scalar(&d_gy, &packed, m, &mut d_exact_scalar_ga)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let exact_scalar_grad_start = std::time::Instant::now();
    for _ in 0..iters {
        cuda.training_salt_grad_a_exact_scalar(&d_gy, &packed, m, &mut d_exact_scalar_ga)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let exact_scalar_grad_us =
        exact_scalar_grad_start.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

    for _ in 0..10 {
        cuda.training_salt_grad_a(&d_gy, &packed, m, &mut d_exact_ga)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let exact_grad_start = std::time::Instant::now();
    for _ in 0..iters {
        cuda.training_salt_grad_a(&d_gy, &packed, m, &mut d_exact_ga)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let exact_grad_us = exact_grad_start.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

    for _ in 0..10 {
        cuda.training_salt_grad_a_fast(&d_gy, &packed, m, &mut d_packed_ga)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let tiled_grad_start = std::time::Instant::now();
    for _ in 0..iters {
        cuda.training_salt_grad_a_fast(&d_gy, &packed, m, &mut d_packed_ga)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let tiled_grad_us = tiled_grad_start.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

    for _ in 0..10 {
        cuda.salt_quantize_forward_dev(&d_master, &mut d_residual, &mut d_dense, n, k, planes)
            .unwrap();
        cuda.matmul_forward_dev(&d_act, &d_dense, &d_ones, shape, &mut d_dense_y)
            .unwrap();
        cuda.grad_a_dev(&d_gy, &d_dense, &d_ones, shape, &mut d_dense_ga)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let dense_full_start = std::time::Instant::now();
    for _ in 0..iters {
        cuda.salt_quantize_forward_dev(&d_master, &mut d_residual, &mut d_dense, n, k, planes)
            .unwrap();
        cuda.matmul_forward_dev(&d_act, &d_dense, &d_ones, shape, &mut d_dense_y)
            .unwrap();
        cuda.grad_a_dev(&d_gy, &d_dense, &d_ones, shape, &mut d_dense_ga)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let dense_full_us = dense_full_start.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

    for _ in 0..10 {
        cuda.repack_training_salt(&d_master, &mut d_residual, &mut packed)
            .unwrap();
        cuda.training_salt_forward_exact_scalar(&d_act, &packed, m, &mut d_exact_scalar_y)
            .unwrap();
        cuda.training_salt_grad_a_exact_scalar(&d_gy, &packed, m, &mut d_exact_scalar_ga)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let exact_scalar_full_start = std::time::Instant::now();
    for _ in 0..iters {
        cuda.repack_training_salt(&d_master, &mut d_residual, &mut packed)
            .unwrap();
        cuda.training_salt_forward_exact_scalar(&d_act, &packed, m, &mut d_exact_scalar_y)
            .unwrap();
        cuda.training_salt_grad_a_exact_scalar(&d_gy, &packed, m, &mut d_exact_scalar_ga)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let exact_scalar_full_us =
        exact_scalar_full_start.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

    for _ in 0..10 {
        cuda.repack_training_salt(&d_master, &mut d_residual, &mut packed)
            .unwrap();
        cuda.training_salt_forward(&d_act, &packed, m, &mut d_exact_y)
            .unwrap();
        cuda.training_salt_grad_a(&d_gy, &packed, m, &mut d_exact_ga)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let exact_full_start = std::time::Instant::now();
    for _ in 0..iters {
        cuda.repack_training_salt(&d_master, &mut d_residual, &mut packed)
            .unwrap();
        cuda.training_salt_forward(&d_act, &packed, m, &mut d_exact_y)
            .unwrap();
        cuda.training_salt_grad_a(&d_gy, &packed, m, &mut d_exact_ga)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let exact_full_us = exact_full_start.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

    for _ in 0..10 {
        cuda.repack_training_salt(&d_master, &mut d_residual, &mut packed)
            .unwrap();
        cuda.training_salt_forward_fast(&d_act, &packed, m, &mut d_packed_y)
            .unwrap();
        cuda.training_salt_grad_a_fast(&d_gy, &packed, m, &mut d_packed_ga)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let packed_full_start = std::time::Instant::now();
    for _ in 0..iters {
        cuda.repack_training_salt(&d_master, &mut d_residual, &mut packed)
            .unwrap();
        cuda.training_salt_forward_fast(&d_act, &packed, m, &mut d_packed_y)
            .unwrap();
        cuda.training_salt_grad_a_fast(&d_gy, &packed, m, &mut d_packed_ga)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let packed_full_us = packed_full_start.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

    println!(
        "Track D resident SALT {m}x{n}x{k} T={planes} repack+forward: \
         dense={dense_us:.1}us, exact-tiled={exact_us:.1}us, \
         exact-scalar={exact_scalar_us:.1}us ({:.2}x), \
         packed-scalar-fast={scalar_us:.1}us, packed-tiled-fast={packed_us:.1}us; \
         tiled-fast speedup={:.2}x dense / {:.2}x scalar. grad_a: \
         exact-tiled={exact_grad_us:.1}us, exact-scalar={exact_scalar_grad_us:.1}us ({:.2}x), \
         scalar-fast={scalar_grad_us:.1}us, \
         tiled-fast={tiled_grad_us:.1}us ({:.2}x). full repack+forward+grad_a: \
         dense={dense_full_us:.1}us, exact-tiled={exact_full_us:.1}us ({:.2}x dense, {:.2}x scalar), \
         exact-scalar={exact_scalar_full_us:.1}us, \
         packed-tiled-fast={packed_full_us:.1}us ({:.2}x); \
         dense weight={} B, packed={} B ({:.1}%)",
        exact_scalar_us / exact_us,
        dense_us / packed_us,
        scalar_us / packed_us,
        exact_scalar_grad_us / exact_grad_us,
        scalar_grad_us / tiled_grad_us,
        dense_full_us / exact_full_us,
        exact_scalar_full_us / exact_full_us,
        dense_full_us / packed_full_us,
        n * k * core::mem::size_of::<f32>(),
        packed.resident_bytes(),
        packed.resident_bytes() as f64 / (n * k * core::mem::size_of::<f32>()) as f64 * 100.0,
    );
}

#[test]
fn cuda_matches_reference_within_tolerance() {
    // Skip cleanly when no GPU is present (cpu-only dev box / wrong CI lane).
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping cuda conformance: no device ({e})");
            return;
        }
    };

    let tq2 = tq2_vectors();
    let report = run_conformance(&backend, &tq2, Tolerance::default());
    assert!(
        report.is_ok(),
        "{} cuda conformance cases failed: {:?}",
        report.failed.len(),
        report.failed
    );
}

// v0.4.0 P1: the SALT multi-plane GPU GEMM must match `dequant_salt_row` → fp32
// reference matmul within 1e-4, across T∈{1,2,3}, M∈{1,2}, with each plane's
// per-block f16 scales including a zero-variance (scale 0) block and an
// outlier-heavy (large scale) block.
#[test]
fn salt_mpgemm_matches_dequant_reference() {
    use half::f16;
    use tritium_core::Trit;
    use tritium_format::{
        SaltRow, TQ2_0_BLOCK_BYTES, dequant_salt_row, num_blocks, pack_tq2_0_row,
    };

    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping salt mpgemm: no device ({e})");
            return;
        }
    };

    let k = 512usize; // 2 blocks
    let n = 6usize;
    let nb = num_blocks(k);
    let row_bytes = nb * TQ2_0_BLOCK_BYTES;

    let mut s: u64 = 0x5A17_C0DE;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s
    };

    for m in [1usize, 2] {
        for t in [1usize, 2, 3] {
            let act: Vec<f32> = (0..m * k)
                .map(|_| (next() >> 40) as f32 / (1u64 << 23) as f32 - 0.5)
                .collect();

            // planes[p][ni] = packed TQ2_0 bytes for row ni, plane p.
            let mut planes: Vec<Vec<Vec<u8>>> = Vec::with_capacity(t);
            for p in 0..t {
                let mut prows = Vec::with_capacity(n);
                for _ni in 0..n {
                    let trits: Vec<Trit> = (0..k)
                        .map(|_| Trit::from_i8(((next() >> 40) % 3) as i8 - 1).unwrap())
                        .collect();
                    let scales: Vec<f16> = (0..nb)
                        .map(|_| {
                            let pick = (next() >> 40) % 8;
                            let v = match pick {
                                0 => 0.0,  // zero-variance block
                                1 => 12.5, // outlier-heavy block
                                other => 0.05 + other as f32 * 0.3,
                            };
                            f16::from_f32(v / (p as f32 + 1.0))
                        })
                        .collect();
                    let mut bytes = vec![0u8; row_bytes];
                    pack_tq2_0_row(&trits, &scales, &mut bytes).unwrap();
                    prows.push(bytes);
                }
                planes.push(prows);
            }

            // Plane-major concatenation: plane p, then row ni.
            let mut weights = Vec::with_capacity(t * n * row_bytes);
            for prows in &planes {
                for row in prows {
                    weights.extend_from_slice(row);
                }
            }

            // Reference: dequant each row to fp32 weights, then fp64 matmul.
            let mut reference = vec![0f64; m * n];
            for ni in 0..n {
                let row = SaltRow {
                    k,
                    planes: (0..t).map(|p| planes[p][ni].clone()).collect(),
                };
                let w = dequant_salt_row(&row).unwrap();
                for mi in 0..m {
                    let mut acc = 0f64;
                    for kk in 0..k {
                        acc += act[mi * k + kk] as f64 * w[kk] as f64;
                    }
                    reference[mi * n + ni] = acc;
                }
            }

            let gpu = cuda.salt_mpgemm_dense(&act, &weights, m, n, k, t).unwrap();
            for i in 0..m * n {
                let r = reference[i];
                let tol = 1e-4 * r.abs().max(1.0);
                assert!(
                    (gpu[i] as f64 - r).abs() <= tol,
                    "salt mpgemm m={m} t={t} idx={i}: gpu={} ref={r} (tol {tol})",
                    gpu[i],
                );
            }
        }
    }
}

/// v0.4.1: flash-decoding (split-KV) attention must match the direct decode
/// attention (`gqa_attention_decode`) within tolerance — for several `n_split`
/// (chunk counts), including `n_split=1` (single chunk) and a split that leaves a
/// ragged final chunk. The online-softmax merge reorders sums, so this is a
/// tolerance gate (1e-4), not bit-exact.
#[test]
fn attn_split_kv_matches_direct_attention() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping split-kv attn: no device ({e})");
            return;
        }
    };
    let (n_head, n_head_kv, head_dim, ctx) = (8usize, 2usize, 128usize, 200usize);
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut s: u64 = 0x5F11_7A11_u64; // seed
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 40) as f32 / (1u64 << 23) as f32 - 0.5
    };
    let q: Vec<f32> = (0..n_head * head_dim).map(|_| next()).collect();
    let k: Vec<f32> = (0..ctx * n_head_kv * head_dim).map(|_| next()).collect();
    let v: Vec<f32> = (0..ctx * n_head_kv * head_dim).map(|_| next()).collect();

    let mut reference = vec![0.0f32; n_head * head_dim];
    cuda.gqa_attention_decode(
        &q,
        &k,
        &v,
        &mut reference,
        ctx,
        n_head,
        n_head_kv,
        head_dim,
        scale,
        ctx,
    )
    .expect("reference attention");

    for n_split in [1usize, 4, 7, 16] {
        let chunk = ctx.div_ceil(n_split);
        let got = cuda
            .attn_split_dense(
                &q, &k, &v, n_head, n_head_kv, head_dim, scale, ctx, n_split, chunk,
            )
            .expect("split attention");
        for i in 0..n_head * head_dim {
            let r = reference[i];
            let tol = 1e-4 * r.abs().max(1.0);
            assert!(
                (got[i] as f64 - r as f64).abs() <= tol as f64,
                "split-kv n_split={n_split} idx={i}: got={} ref={r} (tol {tol})",
                got[i],
            );
        }
    }
}

/// v0.4.0: the **resident** SALT path — upload a SALT tensor's rows once via
/// [`CudaBackend::upload_salt`], then [`CudaBackend::salt_forward`] — must match
/// the host `dequant_salt_row → fp32 matmul` reference, for T=1/2/3 (incl. ragged
/// plane counts) and survive reuse (two forwards on the same resident buffer).
/// This gates the resident decode wiring, distinct from `salt_mpgemm_dense` which
/// re-uploads per call.
#[test]
fn salt_resident_forward_matches_dequant() {
    use half::f16;
    use tritium_core::Trit;
    use tritium_format::{SaltRow, dequant_salt_row, pack_tq2_0_row};

    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping salt resident: no device ({e})");
            return;
        }
    };

    let k = 512usize;
    let n = 6usize;
    let nb = num_blocks(k);
    let row_bytes = nb * TQ2_0_BLOCK_BYTES;
    let mut s: u64 = 0x5A17_F00D;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s
    };

    // Build n rows; row ni gets `t_of(ni)` planes (ragged: not all rows equal T).
    for max_t in [1usize, 2, 3] {
        let rows: Vec<SaltRow> = (0..n)
            .map(|ni| {
                let t_row = 1 + (ni % max_t); // 1..=max_t, ragged across rows
                let planes = (0..t_row)
                    .map(|p| {
                        let trits: Vec<Trit> = (0..k)
                            .map(|_| Trit::from_i8(((next() >> 40) % 3) as i8 - 1).unwrap())
                            .collect();
                        let scales: Vec<f16> = (0..nb)
                            .map(|_| {
                                f16::from_f32(
                                    (0.05 + ((next() >> 40) % 8) as f32 * 0.3) / (p as f32 + 1.0),
                                )
                            })
                            .collect();
                        let mut bytes = vec![0u8; row_bytes];
                        pack_tq2_0_row(&trits, &scales, &mut bytes).unwrap();
                        bytes
                    })
                    .collect();
                SaltRow { k, planes }
            })
            .collect();

        // Host reference: dequant each row, fp64 matmul.
        let m = 2usize;
        let act: Vec<f32> = (0..m * k)
            .map(|_| (next() >> 40) as f32 / (1u64 << 23) as f32 - 0.5)
            .collect();
        let mut reference = vec![0f64; m * n];
        for (ni, row) in rows.iter().enumerate() {
            let w = dequant_salt_row(row).unwrap();
            for mi in 0..m {
                let mut acc = 0f64;
                for kk in 0..k {
                    acc += act[mi * k + kk] as f64 * w[kk] as f64;
                }
                reference[mi * n + ni] = acc;
            }
        }

        let lin = cuda.upload_salt(&rows, n, k).expect("upload_salt");
        // Two forwards on the same resident buffer must agree (reuse).
        let gpu = cuda.salt_forward(&lin, &act, m).expect("salt_forward");
        let gpu2 = cuda
            .salt_forward(&lin, &act, m)
            .expect("salt_forward reuse");
        assert_eq!(
            gpu, gpu2,
            "resident reuse must be deterministic (max_t={max_t})"
        );

        for i in 0..m * n {
            let r = reference[i];
            let tol = 1e-4 * r.abs().max(1.0);
            assert!(
                (gpu[i] as f64 - r).abs() <= tol,
                "salt resident max_t={max_t} idx={i}: gpu={} ref={r} (tol {tol})",
                gpu[i],
            );
        }
    }
}

/// ADR 0002 U2: CPU↔CUDA parity. The *same* committed TQ2_0 vectors run through
/// both [`CpuBackend`] and [`CudaBackend`]; every output element must agree
/// within `1e-4` relative. This is the load-bearing cross-backend gate — it
/// catches a backend that is internally self-consistent (passes conformance)
/// but disagrees with the other backend on shared inputs.
#[test]
fn cuda_matches_cpu_within_tolerance() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping cpu<->cuda parity: no device ({e})");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let tol = Tolerance::default();

    // Run both backends over the identical TQ2_0 vector set.
    let cpu_report = run_conformance(&cpu, &tq2_vectors(), tol);
    assert!(
        cpu_report.is_ok(),
        "cpu backend failed its own conformance, parity is moot: {:?}",
        cpu_report.failed
    );

    // Replay each vector through both backends and compare outputs directly,
    // rather than only against the shared reference, so any CPU/CUDA divergence
    // surfaces even within the reference tolerance band.
    for v in tq2_vectors() {
        let shape = GemmShape::new(v.m, v.n, v.k);
        let trits: Vec<_> = v
            .weights
            .iter()
            .map(|&w| tritium_core::Trit::from_i8(w).expect("vector weight in {-1,0,1}"))
            .collect();
        let packed = pack_tq2_0(&trits, shape);

        let cpu_out = run_backend(&cpu, &packed, &v.activation, &v.scales, shape);
        let cuda_out = run_backend(&cuda, &packed, &v.activation, &v.scales, shape);

        assert_eq!(
            cpu_out.len(),
            cuda_out.len(),
            "{}: output len mismatch",
            v.id
        );
        for (i, (&c, &g)) in cpu_out.iter().zip(&cuda_out).enumerate() {
            assert!(
                tol.accepts(g, c),
                "{}: cpu/cuda disagree at [{i}]: cpu={c} cuda={g}",
                v.id
            );
        }
    }
}

/// Pack an `[N, K]` trit matrix to TQ2_0 rows, block scale fixed to `1.0` (the
/// testkit convention), ready for `upload_weights`.
fn pack_tq2_0(trits: &[tritium_core::Trit], shape: GemmShape) -> Vec<u8> {
    use tritium_format::pack_tq2_0_row;
    let GemmShape { n, k, .. } = shape;
    let nb = num_blocks(k);
    let unit = vec![half::f16::ONE; nb];
    let row_bytes = nb * TQ2_0_BLOCK_BYTES;
    let mut packed = vec![0u8; n * row_bytes];
    for ni in 0..n {
        let row = &trits[ni * k..ni * k + k];
        let out = &mut packed[ni * row_bytes..(ni + 1) * row_bytes];
        pack_tq2_0_row(row, &unit, out).expect("pack tq2_0 row");
    }
    packed
}

/// Upload weights + run one TQ2_0 mpGEMM through any backend, returning `[M, N]`.
fn run_backend<B: TernaryBackend>(
    backend: &B,
    packed: &[u8],
    act: &[f32],
    scales: &[f32],
    shape: GemmShape,
) -> Vec<f32> {
    let buf = backend
        .upload_weights(packed, shape, TernaryFormat::Tq2_0)
        .expect("upload weights");
    let mut out = vec![0.0f32; shape.m * shape.n];
    backend
        .mpgemm(tritium_spec::MpGemm {
            act,
            weights: buf.as_ref(),
            scales,
            shape,
            format: TernaryFormat::Tq2_0,
            out: &mut out,
        })
        .expect("mpgemm");
    out
}

/// Upload weights + run one TQ2_0 mpGEMM through a *forced* add kernel, so a
/// test can gate each path independently of the shape-based auto-selection.
fn run_kernel(
    cuda: &CudaBackend,
    packed: &[u8],
    act: &[f32],
    scales: &[f32],
    shape: GemmShape,
    kernel: AddKernel,
) -> Vec<f32> {
    let buf = cuda
        .upload_weights(packed, shape, TernaryFormat::Tq2_0)
        .expect("upload weights");
    let mut out = vec![0.0f32; shape.m * shape.n];
    cuda.mpgemm_kernel(
        act,
        buf.as_ref(),
        scales,
        shape,
        TernaryFormat::Tq2_0,
        &mut out,
        kernel,
    )
    .expect("mpgemm_kernel");
    out
}

/// Upload weights + run the sparse-aware tiled kernel with a pre-computed
/// zero-block bitmap. Returns the output `[M, N]`.
fn run_kernel_sparse(
    cuda: &CudaBackend,
    packed: &[u8],
    act: &[f32],
    scales: &[f32],
    bitmap: &[u32],
    words_per_row: usize,
    shape: GemmShape,
) -> Vec<f32> {
    let buf = cuda
        .upload_weights(packed, shape, TernaryFormat::Tq2_0)
        .expect("upload weights");
    let mut out = vec![0.0f32; shape.m * shape.n];
    cuda.mpgemm_kernel_with_bitmap(
        act,
        buf.as_ref(),
        scales,
        bitmap,
        words_per_row,
        shape,
        TernaryFormat::Tq2_0,
        &mut out,
    )
    .expect("mpgemm_kernel_with_bitmap");
    out
}

/// Both add kernels must match the CPU reference (within tolerance) on the full
/// committed TQ2_0 conformance set. This gates the new tiled kernel directly,
/// and re-gates the simple kernel, regardless of which one auto-selection picks.
#[test]
fn both_add_kernels_match_reference() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping both-kernel gate: no device ({e})");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let tol = Tolerance::default();

    for v in tq2_vectors() {
        let shape = GemmShape::new(v.m, v.n, v.k);
        let trits: Vec<_> = v
            .weights
            .iter()
            .map(|&w| tritium_core::Trit::from_i8(w).expect("weight in {-1,0,1}"))
            .collect();
        let packed = pack_tq2_0(&trits, shape);
        let cpu_out = run_backend(&cpu, &packed, &v.activation, &v.scales, shape);

        let simple = run_kernel(
            &cuda,
            &packed,
            &v.activation,
            &v.scales,
            shape,
            AddKernel::Simple,
        );
        for (i, (&g, &c)) in simple.iter().zip(&cpu_out).enumerate() {
            assert!(tol.accepts(g, c), "{}: simple vs cpu [{i}] {g} {c}", v.id);
        }

        // The tiled kernel only accepts K within its shared-memory budget.
        if v.k <= TILED_K_MAX {
            let tiled = run_kernel(
                &cuda,
                &packed,
                &v.activation,
                &v.scales,
                shape,
                AddKernel::Tiled,
            );
            for (i, (&g, &c)) in tiled.iter().zip(&cpu_out).enumerate() {
                assert!(tol.accepts(g, c), "{}: tiled vs cpu [{i}] {g} {c}", v.id);
            }
        }
    }
}

/// The tiled kernel must be correct on boundary shapes: tail `K` (not a 256
/// multiple, so a partial final TQ2_0 block), partial warps (`N` not a multiple
/// of `WARPS_PER_BLOCK`), partial grids (`M`/`N` of 1), and `K` at the cap.
#[test]
fn tiled_handles_tail_shapes() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping tiled tail-shape gate: no device ({e})");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let tol = Tolerance::default();

    // (M, N, K) — tail K, partial warps/blocks, single rows/cols, K at the cap.
    let shapes = [
        (1usize, 1usize, 1usize),
        (1, 7, 300),
        (5, 130, 257),
        (64, 3, 2560),
        (3, 33, 6912),
        (1, 1, TILED_K_MAX),
    ];

    for (m, n, k) in shapes {
        assert!(k <= TILED_K_MAX, "test shape K exceeds the tiled cap");
        let shape = GemmShape::new(m, n, k);

        // Deterministic ternary weights, activations, and per-channel scales.
        let trits: Vec<_> = (0..n * k)
            .map(|i| tritium_core::Trit::from_i8(((i % 3) as i8) - 1).unwrap())
            .collect();
        let act: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
        let scales: Vec<f32> = (0..n).map(|j| 1.0 + (j % 4) as f32 * 0.25).collect();

        let packed = pack_tq2_0(&trits, shape);
        let cpu_out = run_backend(&cpu, &packed, &act, &scales, shape);
        let tiled = run_kernel(&cuda, &packed, &act, &scales, shape, AddKernel::Tiled);

        assert_eq!(tiled.len(), cpu_out.len(), "shape {shape:?}: len");
        for (i, (&g, &c)) in tiled.iter().zip(&cpu_out).enumerate() {
            assert!(
                tol.accepts(g, c),
                "shape {shape:?}: tiled vs cpu [{i}] tiled={g} cpu={c}"
            );
        }
    }
}

/// A2 flipped this contract: TQ1_0 UPLOADS are now first-class (the tq1
/// decode kernels read them natively) — a correct-length upload succeeds and
/// a wrong-length one is a typed InvalidInput; the HOST mpgemm path still
/// rejects the format (the resident decoder is TQ1's only consumer in v1).
#[test]
fn tq1_0_upload_accepted_host_mpgemm_rejected() {
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(_) => return, // no device: nothing to assert about format handling
    };
    let shape = GemmShape { m: 1, n: 1, k: 256 };
    // Wrong length (66 = a TQ2 block) -> typed error, not a panic.
    match backend.upload_weights(&[0u8; 66], shape, TernaryFormat::Tq1_0) {
        Err(BackendError::InvalidInput(_)) => {}
        other => panic!("expected InvalidInput, got {:?}", other.map(|_| "ok")),
    }
    // Correct length (54 = one TQ1 block) uploads.
    let buf = backend
        .upload_weights(&[0u8; 54], shape, TernaryFormat::Tq1_0)
        .expect("tq1 upload");
    // Host mpgemm rejects the format loudly.
    let act = vec![0.0f32; 256];
    let scales = vec![1.0f32];
    let mut out = vec![0.0f32; 1];
    match backend.mpgemm(tritium_spec::MpGemm {
        act: &act,
        weights: &*buf,
        scales: &scales,
        shape,
        format: TernaryFormat::Tq1_0,
        out: &mut out,
    }) {
        Err(BackendError::UnsupportedFormat(TernaryFormat::Tq1_0)) => {}
        other => panic!("expected UnsupportedFormat(Tq1_0), got {other:?}"),
    }
}

// ---- IMMA int8 tensor-core path (v0.30 WF-A part 2) ------------------------
//
// Tolerance: the conformance default (`relative = 1e-4`, ADR 0002). The IMMA
// kernel contracts in **int32**, which is *exact* for int8×ternary (no overflow
// for any BitNet K — see `kernels/tq2_0_imma.cu`), so the only float rounding is
// the single per-output `act_scale·weight_scale·acc`. The 1e-4 band is therefore
// the *reference's* own f32-accumulate rounding, not a defect of this kernel —
// no widened reduction bar is needed (cf. the tiled add-only kernel, which sums
// in double to stay inside the band; the IMMA integer accumulate is exact).

/// Build an I2_S tensor payload (`N·K/4` quant bytes + one trailing `f32` scale)
/// from an `[N, K]` row-major trit matrix, inverting the 32-byte block striping
/// (`code = trit + 1`, element `pos` of a 128-block at byte `pos%32`, shift
/// `6 - 2*(pos/32)`). `n*k` must be a multiple of 128 (the conformance shapes
/// all are: K ∈ {256, 512}).
fn build_i2s_payload(trits: &[i8], scale: f32) -> Vec<u8> {
    let n_elements = trits.len();
    assert!(
        n_elements.is_multiple_of(128),
        "i2s payload needs 128-multiple elems"
    );
    let mut quants = vec![0u8; n_elements / 4];
    for (global, &t) in trits.iter().enumerate() {
        let block = global / 128;
        let pos = global % 128;
        let group = pos / 32;
        let gp = pos % 32;
        let code = (t + 1) as u8; // {-1,0,1} -> {0,1,2}
        quants[block * 32 + gp] |= code << (6 - 2 * group);
    }
    let mut payload = quants;
    payload.extend_from_slice(&scale.to_le_bytes());
    payload
}

/// Pack an `[N, K]` trit matrix into the IMMA `I2sInt8` layout by routing it
/// through the *real* converter (`build_i2s_payload` → `convert_i2s_to_int8`),
/// so the test exercises exactly the bytes the kernel will see in production.
/// Returns the packed bytes (block scale folded into the per-tensor `scale`,
/// which the test keeps separate as the per-channel scale, so pass `scale = 1`).
fn pack_i2s_int8(trits: &[i8], shape: GemmShape) -> Vec<u8> {
    let GemmShape { n, k, .. } = shape;
    let payload = build_i2s_payload(trits, 1.0);
    let w = tritium_format::convert_i2s_to_int8(&payload, GemmShape { m: 0, n, k })
        .expect("convert i2s -> int8");
    w.bytes
}

/// IMMA == reference within tolerance over the conformance set. The vectors'
/// weights are converted to `I2sInt8`, uploaded, and run through the fused
/// `mpgemm_with_act_quant` (which routes I2sInt8 → on-device quant + IMMA). The
/// reference is `mpgemm_with_act_quant`'s contract on the *same f32 activations*:
/// `out[m,n] = act_scale[m]·weight_scale[n]·Σ q[m,k]·w[n,k]`, which the testkit
/// CPU path computes via the spec default — so this gates IMMA == host-A8 == ref
/// in one shot.
#[test]
fn imma_matches_reference_within_tolerance() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping imma conformance: no device ({e})");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let tol = Tolerance::default();

    for v in tq2_vectors() {
        let shape = GemmShape::new(v.m, v.n, v.k);

        // Reference: the host-A8 default path on the CPU backend over the SAME
        // f32 activations + per-channel weight scales.
        let cpu_buf = {
            let trits: Vec<_> = v
                .weights
                .iter()
                .map(|&w| tritium_core::Trit::from_i8(w).expect("weight in {-1,0,1}"))
                .collect();
            let packed = pack_tq2_0(&trits, shape);
            cpu.upload_weights(&packed, shape, TernaryFormat::Tq2_0)
                .expect("cpu upload")
        };
        let mut ref_out = vec![0.0f32; shape.m * shape.n];
        cpu.mpgemm_with_act_quant(tritium_spec::MpGemm {
            act: &v.activation,
            weights: cpu_buf.as_ref(),
            scales: &v.scales,
            shape,
            format: TernaryFormat::Tq2_0,
            out: &mut ref_out,
        })
        .expect("cpu host-A8 reference");

        // IMMA: upload the I2sInt8 weights, run the fused override (on-device
        // quant + tensor-core contraction).
        let imma_bytes = pack_i2s_int8(&v.weights, shape);
        let imma_buf = cuda
            .upload_weights(&imma_bytes, shape, TernaryFormat::I2sInt8)
            .expect("imma upload");
        let mut imma_out = vec![0.0f32; shape.m * shape.n];
        cuda.mpgemm_with_act_quant(tritium_spec::MpGemm {
            act: &v.activation,
            weights: imma_buf.as_ref(),
            scales: &v.scales,
            shape,
            format: TernaryFormat::I2sInt8,
            out: &mut imma_out,
        })
        .expect("imma fused mpgemm");

        assert_eq!(imma_out.len(), ref_out.len(), "{}: len", v.id);
        for (i, (&g, &c)) in imma_out.iter().zip(&ref_out).enumerate() {
            assert!(
                tol.accepts(g, c),
                "{}: imma vs host-A8 ref [{i}] imma={g} ref={c}",
                v.id
            );
        }
    }
}

/// The CUDA fused override (IMMA) == the spec host-A8 default == the v0.20
/// caller-side quant, all within tolerance — the "fused == host-A8" gate of ADR
/// 0005. Three independently-derived results over the same inputs:
///   1. `cuda.mpgemm_with_act_quant` on an I2sInt8 buffer → on-device quant + IMMA.
///   2. The spec *default* `mpgemm_with_act_quant` (host quant → `mpgemm`) run on
///      the CPU backend (a TQ2_0 buffer).
///   3. The v0.20 caller-side quant: quantize on the host, then call plain
///      `mpgemm` and fold the per-token scale by hand.
#[test]
fn imma_fused_equals_host_a8_and_caller_quant() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping fused parity: no device ({e})");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let tol = Tolerance::default();

    for v in tq2_vectors() {
        let shape = GemmShape::new(v.m, v.n, v.k);
        let GemmShape { m, n, k } = shape;
        let trits: Vec<_> = v
            .weights
            .iter()
            .map(|&w| tritium_core::Trit::from_i8(w).expect("weight in {-1,0,1}"))
            .collect();
        let tq2 = pack_tq2_0(&trits, shape);

        // (1) CUDA fused override on I2sInt8.
        let imma_bytes = pack_i2s_int8(&v.weights, shape);
        let imma_buf = cuda
            .upload_weights(&imma_bytes, shape, TernaryFormat::I2sInt8)
            .expect("imma upload");
        let mut fused = vec![0.0f32; m * n];
        cuda.mpgemm_with_act_quant(tritium_spec::MpGemm {
            act: &v.activation,
            weights: imma_buf.as_ref(),
            scales: &v.scales,
            shape,
            format: TernaryFormat::I2sInt8,
            out: &mut fused,
        })
        .expect("cuda fused");

        // (2) Spec host-A8 default on the CPU backend (TQ2_0).
        let cpu_buf = cpu
            .upload_weights(&tq2, shape, TernaryFormat::Tq2_0)
            .expect("cpu upload");
        let mut host_a8 = vec![0.0f32; m * n];
        cpu.mpgemm_with_act_quant(tritium_spec::MpGemm {
            act: &v.activation,
            weights: cpu_buf.as_ref(),
            scales: &v.scales,
            shape,
            format: TernaryFormat::Tq2_0,
            out: &mut host_a8,
        })
        .expect("cpu host-A8");

        // (3) v0.20 caller-side quant: host quant → plain `mpgemm` → fold.
        let mut q = vec![0.0f32; m * k];
        let mut act_scale = vec![0.0f32; m];
        quantize_act_int8_host(&v.activation, m, k, &mut q, &mut act_scale);
        let mut caller = vec![0.0f32; m * n];
        cpu.mpgemm(tritium_spec::MpGemm {
            act: &q,
            weights: cpu_buf.as_ref(),
            scales: &v.scales,
            shape,
            format: TernaryFormat::Tq2_0,
            out: &mut caller,
        })
        .expect("cpu plain mpgemm");
        for (row, &s) in caller.chunks_exact_mut(n).zip(act_scale.iter()) {
            for x in row {
                *x *= s;
            }
        }

        for i in 0..m * n {
            assert!(
                tol.accepts(fused[i], host_a8[i]),
                "{}: fused vs host-A8 [{i}] {} {}",
                v.id,
                fused[i],
                host_a8[i]
            );
            assert!(
                tol.accepts(fused[i], caller[i]),
                "{}: fused vs caller-quant [{i}] {} {}",
                v.id,
                fused[i],
                caller[i]
            );
        }
    }
}

/// IMMA tail/boundary shapes: M not a multiple of 16, N not a multiple of 8, and
/// single rows/cols — the padding in the I2sInt8 tiles and the kernel's global
/// bounds checks must keep every covered output correct. K stays a 256-multiple
/// (the I2_S converter needs a 128-multiple element count); the M/N tails are the
/// interesting axes for the 16×8 tile.
#[test]
fn imma_handles_tail_shapes() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping imma tail shapes: no device ({e})");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let tol = Tolerance::default();

    // (M, N, K): single row/col, partial 16-row tile, partial 8-col tile.
    let shapes = [
        (1usize, 1usize, 256usize),
        (1, 8, 256),
        (3, 5, 256),
        (16, 8, 512),
        (17, 9, 256),
        (33, 13, 512),
    ];
    for (m, n, k) in shapes {
        let shape = GemmShape::new(m, n, k);
        // Deterministic ternary weights, activations, per-channel scales.
        let raw: Vec<i8> = (0..n * k).map(|i| ((i % 3) as i8) - 1).collect();
        let act: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.3).collect();
        let scales: Vec<f32> = (0..n).map(|j| 1.0 + (j % 4) as f32 * 0.5).collect();

        // Reference: host-A8 default on the CPU backend.
        let trits: Vec<_> = raw
            .iter()
            .map(|&w| tritium_core::Trit::from_i8(w).unwrap())
            .collect();
        let cpu_buf = cpu
            .upload_weights(&pack_tq2_0(&trits, shape), shape, TernaryFormat::Tq2_0)
            .expect("cpu upload");
        let mut ref_out = vec![0.0f32; m * n];
        cpu.mpgemm_with_act_quant(tritium_spec::MpGemm {
            act: &act,
            weights: cpu_buf.as_ref(),
            scales: &scales,
            shape,
            format: TernaryFormat::Tq2_0,
            out: &mut ref_out,
        })
        .expect("cpu host-A8");

        let imma_buf = cuda
            .upload_weights(&pack_i2s_int8(&raw, shape), shape, TernaryFormat::I2sInt8)
            .expect("imma upload");
        let mut imma_out = vec![0.0f32; m * n];
        cuda.mpgemm_with_act_quant(tritium_spec::MpGemm {
            act: &act,
            weights: imma_buf.as_ref(),
            scales: &scales,
            shape,
            format: TernaryFormat::I2sInt8,
            out: &mut imma_out,
        })
        .expect("imma fused");

        for (i, (&g, &c)) in imma_out.iter().zip(&ref_out).enumerate() {
            assert!(
                tol.accepts(g, c),
                "shape {shape:?}: imma vs ref [{i}] imma={g} ref={c}"
            );
        }
    }
}

/// ADR 0026 Track P step 2: the load-time IMMA shadow (TQ2 packed rows →
/// unpack → `pack_i2s_int8_tiles`) must produce byte-identical output to the
/// production I2_S converter (`convert_i2s_to_int8`) for the same trits —
/// the kernel sees exactly one weight layout regardless of which path packed
/// it. Host-only, no GPU.
#[test]
fn imma_shadow_matches_i2s_converter_bytes() {
    use tritium_format::{TQ2_0_BLOCK_BYTES, num_blocks};
    for &(n, k) in &[(8usize, 256usize), (13, 512), (40, 2560), (5, 6912)] {
        let trits = mixed_trits(n, k, 0x77 ^ (n as u64) ^ (k as u64));
        let nb = num_blocks(k);
        let unit = vec![half::f16::ONE; nb];
        let rb = nb * TQ2_0_BLOCK_BYTES;
        let mut rows = vec![0u8; n * rb];
        for ni in 0..n {
            tritium_format::pack_tq2_0_row(
                &trits[ni * k..(ni + 1) * k],
                &unit,
                &mut rows[ni * rb..(ni + 1) * rb],
            )
            .expect("pack tq2");
        }
        let shadow = imma_shadow_bytes(&rows, n, k, rb).expect("shadow");
        let trits_i8: Vec<i8> = trits.iter().map(|t| t.get()).collect();
        let converter = pack_i2s_int8(&trits_i8, GemmShape { m: 0, n, k });
        assert_eq!(
            shadow, converter,
            "n{n} k{k}: shadow bytes != convert_i2s_to_int8 bytes"
        );
    }
}

/// ADR 0026 Track P bit-identity gate: the IMMA tensor-core kernel must be
/// **bit-identical** to the dp4a `tiled_i8_scaled` kernel on the SAME int8
/// activations, act scales and per-channel weight scales. Both contract in
/// exact i32 (order-free) and both fold `(float)acc * wscale[n] * act_scale[m]`
/// in the same association (a pure multiply chain — no FMA contraction), so
/// every output bit matches. This is what lets the prefill dispatch swap
/// kernels by M with ZERO numerics re-gating (C1 chunking, G1 first-token).
/// Shapes cover 16/8/32-tile boundaries and tails at the real K values.
#[test]
fn imma_matches_dp4a_tiled_scaled_bit_exact() {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::Ptx;
    use tritium_format::{TQ2_0_BLOCK_BYTES, num_blocks};

    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping imma-vs-dp4a gate: no device ({e})");
            return;
        }
    };
    let stream = ctx.default_stream();
    let m_add = ctx
        .load_module(Ptx::from_src(TQ2_0_ADD_PTX))
        .expect("load add module");
    let m_imma = ctx
        .load_module(Ptx::from_src(TQ2_0_IMMA_PTX))
        .expect("load imma module");
    let f_dp4a = m_add
        .load_function("tq2_0_add_mpgemm_tiled_i8_scaled")
        .expect("dp4a fn");
    let f_imma = m_imma.load_function("tq2_0_imma_mpgemm").expect("imma fn");

    // (m, n, k): tile-aligned and tail shapes at prefill-realistic K
    // (k % 256 == 0 — the TQ2 packer's block size; covers K=2560/6912-class
    // contractions via 2560 and a 6912-divisor tail mix).
    for &(m, n, k) in &[
        (16usize, 8usize, 256usize),
        (33, 13, 512),
        (128, 40, 2560),
        (7, 9, 1024),
    ] {
        let trits = mixed_trits(n, k, 0x51 ^ (m as u64) ^ (k as u64));
        // dp4a weights: TQ2_0 rows with unit block scales (both kernels ignore
        // block scales; the per-channel `scales` array is the shared truth).
        let nb = num_blocks(k);
        let unit = vec![half::f16::ONE; nb];
        let rb = nb * TQ2_0_BLOCK_BYTES;
        let mut packed_tq2 = vec![0u8; n * rb];
        for ni in 0..n {
            tritium_format::pack_tq2_0_row(
                &trits[ni * k..(ni + 1) * k],
                &unit,
                &mut packed_tq2[ni * rb..(ni + 1) * rb],
            )
            .expect("pack tq2");
        }
        // IMMA weights: the I2sInt8 tile interleave from the SAME trits.
        let trits_i8: Vec<i8> = trits.iter().map(|t| t.get()).collect();
        let packed_imma = pack_i2s_int8_tiles(&trits_i8, n, k);
        let num_ktiles = k.div_ceil(tritium_format::IMMA_K);

        let qact: Vec<i8> = (0..m * k).map(|i| ((i * 37 + 11) % 255) as i8).collect();
        let scales = seeded_f32(7, n, 0.5, 2.0);
        let act_scale = seeded_f32(13, m, 0.5, 1.5);

        let d_qact = stream.clone_htod(&qact).unwrap();
        let d_w2 = stream.clone_htod(&packed_tq2).unwrap();
        let d_wi = stream.clone_htod(&packed_imma).unwrap();
        let d_sc = stream.clone_htod(&scales).unwrap();
        let d_as = stream.clone_htod(&act_scale).unwrap();
        let (m_i, n_i, k_i, rb_i, nkt_i) =
            (m as i32, n as i32, k as i32, rb as i32, num_ktiles as i32);

        // dp4a launch (the production tiled_i8_scaled geometry).
        let dp4a_out = {
            let mut d_out = stream.alloc_zeros::<f32>(m * n).unwrap();
            let cfg = LaunchConfig {
                grid_dim: ((n as u32).div_ceil(8), m as u32, 1),
                block_dim: (8 * 32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut l = stream.launch_builder(&f_dp4a);
            l.arg(&d_qact)
                .arg(&d_w2)
                .arg(&d_sc)
                .arg(&d_as)
                .arg(&mut d_out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .arg(&rb_i);
            // SAFETY: matches `tq2_0_add_mpgemm_tiled_i8_scaled(qact, w, scales,
            // act_scale, out, m, n, k, row_bytes)`; grid.y = m.
            #[allow(unsafe_code)]
            unsafe {
                l.launch(cfg).unwrap()
            };
            let mut out = vec![0.0f32; m * n];
            stream.memcpy_dtoh(&d_out, &mut out).unwrap();
            out
        };

        // IMMA launch (one warp per 16x8 tile).
        let imma_out = {
            let mut d_out = stream.alloc_zeros::<f32>(m * n).unwrap();
            let cfg = LaunchConfig {
                grid_dim: ((n as u32).div_ceil(8), (m as u32).div_ceil(16), 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut l = stream.launch_builder(&f_imma);
            l.arg(&d_qact)
                .arg(&d_wi)
                .arg(&d_as)
                .arg(&d_sc)
                .arg(&mut d_out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .arg(&nkt_i);
            // SAFETY: matches `tq2_0_imma_mpgemm(qact, weights, act_scale,
            // weight_scale, out, m, n, k, num_ktiles)`; grid covers all tiles.
            #[allow(unsafe_code)]
            unsafe {
                l.launch(cfg).unwrap()
            };
            let mut out = vec![0.0f32; m * n];
            stream.memcpy_dtoh(&d_out, &mut out).unwrap();
            out
        };

        for (i, (a, b)) in dp4a_out.iter().zip(&imma_out).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "m{m} n{n} k{k} [{i}]: dp4a={a} imma={b} — the epilogue \
                 association drifted (bit-identity contract, ADR 0026)"
            );
        }
    }
}

// ---- WF-B: autotune + nvrtc JIT determinism (ADR 0005) ---------------------
//
// These gate the WF-B contract: a JIT-compiled tile is BIT-IDENTICAL to the AOT
// cubin for the same tile (cold-cache == warm-cache), and any tuned tile matches
// the reference within the IMMA tolerance. Both are guaranteed by construction —
// every tile does the same exact int32 mma accumulate + one f32 scale fold — but
// these tests prove it on-device across tile shapes.

/// Deterministic int8 activations / ternary weights / scales for a WF-B probe.
fn jit_probe_inputs(m: usize, n: usize, k: usize) -> (Vec<i8>, Vec<f32>, Vec<f32>, Vec<i8>) {
    let qact: Vec<i8> = (0..m * k).map(|i| ((i % 7) as i8) - 3).collect();
    let act_scale: Vec<f32> = (0..m).map(|i| 0.5 + (i % 3) as f32 * 0.25).collect();
    let wscale: Vec<f32> = (0..n).map(|j| 1.0 + (j % 4) as f32 * 0.5).collect();
    let trits: Vec<i8> = (0..n * k).map(|i| ((i % 3) as i8) - 1).collect();
    (qact, act_scale, wscale, trits)
}

/// Run one IMMA contraction with an explicit `func`/`tile` (host-quantised int8
/// inputs already supplied), returning the `[M, N]` f32 output. Drives
/// `launch_imma_tile` directly so a test can force a specific tile + kernel image
/// (AOT cubin vs a freshly JIT-compiled module).
#[allow(clippy::too_many_arguments)] // a test driver mirroring the kernel's operands
fn run_imma_tile(
    cuda: &CudaBackend,
    func: &CudaFunction,
    tile: TileConfig,
    qact: &[i8],
    packed_weights: &[u8],
    act_scale: &[f32],
    wscale: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let num_ktiles = k.div_ceil(IMMA_K);
    let d_qact = cuda.stream.clone_htod(qact).expect("htod qact");
    let d_weights = cuda
        .stream
        .clone_htod(packed_weights)
        .expect("htod weights");
    let d_act_scale = cuda.stream.clone_htod(act_scale).expect("htod act_scale");
    let d_wscale = cuda.stream.clone_htod(wscale).expect("htod wscale");
    let mut d_out = cuda.stream.alloc_zeros::<f32>(m * n).expect("alloc out");
    cuda.launch_imma_tile(
        func,
        tile,
        &d_qact,
        &d_weights,
        &d_act_scale,
        &d_wscale,
        &mut d_out,
        m as i32,
        n as i32,
        k as i32,
        num_ktiles as i32,
    )
    .expect("launch imma tile");
    let mut out = vec![0.0f32; m * n];
    cuda.stream.memcpy_dtoh(&d_out, &mut out).expect("dtoh out");
    cuda.stream.synchronize().expect("sync");
    out
}

/// COLD-CACHE (JIT) == WARM-CACHE (AOT) BIT-IDENTICAL for a fixed tile.
///
/// The AOT-equivalent tile has two realisations: the embedded AOT cubin
/// (`func_imma`, the warm/default path) and a fresh nvrtc JIT compile of the
/// rendered source (the cold path). For a range of shapes their outputs must be
/// **bit-for-bit equal** (`==` on the raw `f32`, not a tolerance) — the load-bearing
/// WF-B determinism gate. If they ever diverge, JIT and AOT are not interchangeable
/// and the autotune cache could change numerics, which ADR 0005 forbids.
#[test]
fn jit_aot_equivalent_is_bit_identical() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping JIT==AOT bit-identity: no device ({e})");
            return;
        }
    };

    // Freshly JIT-compile the AOT-equivalent tile (the cold path). The AOT side
    // is the embedded cubin resolved by `imma_function_for_tile`.
    let tile = TileConfig::AOT_EQUIVALENT;
    let (_jit_mod, jit_func) = cuda
        .imma_jit_function(tile)
        .expect("JIT-compile AOT-equivalent tile");
    let aot_func = cuda
        .imma_function_for_tile(tile)
        .expect("resolve AOT cubin");

    // Tail + clean shapes; K a 32-multiple (one whole k-tile minimum).
    let shapes = [
        (1usize, 1usize, 32usize),
        (3, 5, 64),
        (16, 8, 256),
        (17, 9, 96),
        (33, 13, 512),
        (64, 40, 2560), // a realistic-ish K (a 32-multiple, below the tiled cap)
    ];
    for (m, n, k) in shapes {
        let k = k.max(IMMA_K); // never zero k-tiles
        let k = k.div_ceil(IMMA_K) * IMMA_K; // snap to a whole k-tile
        let (qact, act_scale, wscale, trits) = jit_probe_inputs(m, n, k);
        let packed = pack_i2s_int8_tiles(&trits, n, k);

        let aot = run_imma_tile(
            &cuda, &aot_func, tile, &qact, &packed, &act_scale, &wscale, m, n, k,
        );
        let jit = run_imma_tile(
            &cuda, &jit_func, tile, &qact, &packed, &act_scale, &wscale, m, n, k,
        );

        assert_eq!(aot.len(), jit.len(), "shape ({m},{n},{k}): len");
        for (i, (&a, &j)) in aot.iter().zip(&jit).enumerate() {
            // Bit-identical: compare the raw IEEE-754 bit patterns so even a
            // signed-zero or NaN-payload difference would fail (none expected).
            assert_eq!(
                a.to_bits(),
                j.to_bits(),
                "shape ({m},{n},{k}): JIT vs AOT diverge at [{i}] aot={a} jit={j}"
            );
        }
    }
}

/// A NON-TRIVIAL JIT tile (wider M/N, deeper K, multi-warp) is ALSO bit-identical
/// to the AOT cubin. This proves the determinism guarantee holds across the tile
/// shapes the autotune search actually considers, not just the AOT-equivalent
/// anchor — the int32 accumulate is order-independent, so a 32×16/4-warp tile that
/// splits the work differently still lands on the same bits.
#[test]
fn jit_wide_tile_matches_aot_bit_identical() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping wide-tile JIT==AOT: no device ({e})");
            return;
        }
    };
    let aot_func = cuda
        .imma_function_for_tile(TileConfig::AOT_EQUIVALENT)
        .expect("AOT cubin");

    // A representative spread of the search's candidate tiles.
    let tiles = [
        TileConfig {
            tile_m: 16,
            tile_n: 8,
            tile_k: 128,
            warps: 1,
            stages: 2,
        },
        TileConfig {
            tile_m: 16,
            tile_n: 16,
            tile_k: 64,
            warps: 2,
            stages: 2,
        },
        TileConfig {
            tile_m: 32,
            tile_n: 16,
            tile_k: 64,
            warps: 4,
            stages: 2,
        },
        TileConfig {
            tile_m: 64,
            tile_n: 16,
            tile_k: 32,
            warps: 8,
            stages: 3,
        },
    ];
    let (m, n, k) = (40usize, 24usize, 256usize);
    let (qact, act_scale, wscale, trits) = jit_probe_inputs(m, n, k);
    let packed = pack_i2s_int8_tiles(&trits, n, k);

    let aot = run_imma_tile(
        &cuda,
        &aot_func,
        TileConfig::AOT_EQUIVALENT,
        &qact,
        &packed,
        &act_scale,
        &wscale,
        m,
        n,
        k,
    );

    for tile in tiles {
        assert!(tile.is_valid(), "test tile {tile:?} invalid");
        let (_m, jit_func) = cuda
            .imma_jit_function(tile)
            .unwrap_or_else(|e| panic!("JIT-compile {tile:?}: {e:?}"));
        let jit = run_imma_tile(
            &cuda, &jit_func, tile, &qact, &packed, &act_scale, &wscale, m, n, k,
        );
        for (i, (&a, &j)) in aot.iter().zip(&jit).enumerate() {
            assert_eq!(
                a.to_bits(),
                j.to_bits(),
                "tile {tile:?}: JIT vs AOT diverge at [{i}] aot={a} jit={j}"
            );
        }
    }
}

/// The TUNED config (resolved through the on-disk autotune cache + tile search)
/// matches the reference within the IMMA tolerance. Drives the full public fused
/// path (`mpgemm_with_act_quant`), which now consults the cache via
/// `resolve_imma_tile`, on a prefill-shaped problem — so this exercises the tuner
/// end-to-end (cold cache → search → winner) and gates the winner vs the CPU
/// host-A8 reference. A second call (warm cache) must agree bit-for-bit with the
/// first, since a cached tile is numerically identical to the freshly-tuned one.
#[test]
fn tuned_config_matches_reference_and_is_stable() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping tuned-config gate: no device ({e})");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let tol = Tolerance::default();

    // A prefill-shaped problem so the search has something to chew on. K is a
    // 256-multiple (the I2_S converter the reference path uses needs a
    // 128-multiple); N/M exercise partial tiles.
    let (m, n, k) = (40usize, 24usize, 256usize);
    let shape = GemmShape::new(m, n, k);
    let raw: Vec<i8> = (0..n * k).map(|i| ((i % 3) as i8) - 1).collect();
    let act: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.3).collect();
    let scales: Vec<f32> = (0..n).map(|j| 1.0 + (j % 4) as f32 * 0.5).collect();

    // Reference: host-A8 default on the CPU backend (TQ2_0).
    let trits: Vec<_> = raw
        .iter()
        .map(|&w| tritium_core::Trit::from_i8(w).unwrap())
        .collect();
    let cpu_buf = cpu
        .upload_weights(&pack_tq2_0(&trits, shape), shape, TernaryFormat::Tq2_0)
        .expect("cpu upload");
    let mut ref_out = vec![0.0f32; m * n];
    cpu.mpgemm_with_act_quant(tritium_spec::MpGemm {
        act: &act,
        weights: cpu_buf.as_ref(),
        scales: &scales,
        shape,
        format: TernaryFormat::Tq2_0,
        out: &mut ref_out,
    })
    .expect("cpu host-A8 reference");

    // Tuned path: upload I2sInt8, run the fused override (which resolves + tunes
    // the tile). Run it twice; the second call hits the in-memory + on-disk cache.
    let imma_buf = cuda
        .upload_weights(&pack_i2s_int8(&raw, shape), shape, TernaryFormat::I2sInt8)
        .expect("imma upload");
    let mut tuned1 = vec![0.0f32; m * n];
    cuda.mpgemm_with_act_quant(tritium_spec::MpGemm {
        act: &act,
        weights: imma_buf.as_ref(),
        scales: &scales,
        shape,
        format: TernaryFormat::I2sInt8,
        out: &mut tuned1,
    })
    .expect("tuned fused (cold)");
    let mut tuned2 = vec![0.0f32; m * n];
    cuda.mpgemm_with_act_quant(tritium_spec::MpGemm {
        act: &act,
        weights: imma_buf.as_ref(),
        scales: &scales,
        shape,
        format: TernaryFormat::I2sInt8,
        out: &mut tuned2,
    })
    .expect("tuned fused (warm)");

    // Tuned == reference within tolerance.
    for (i, (&g, &c)) in tuned1.iter().zip(&ref_out).enumerate() {
        assert!(tol.accepts(g, c), "tuned vs ref [{i}] tuned={g} ref={c}");
    }
    // Cold vs warm cache: bit-for-bit identical (same tile → same numerics).
    for (i, (&a, &b)) in tuned1.iter().zip(&tuned2).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "cold vs warm tuned output diverges at [{i}] cold={a} warm={b}"
        );
    }
}

/// rev-4 staging coverage: JIT tiles must match the AOT anchor **bit-for-bit**
/// on shapes the sweep never visits — `k % 16 != 0` (the cp.async fast path is
/// invalid there; the per-byte staging fallback must produce the identical
/// shared bytes) and `k % 32 != 0` with `k % 16 == 0` (a full zero-filled
/// cp.async tail chunk inside the last k-tile). Tiles are chosen to exercise
/// every rev-4 geometry regime: multi-subtile warp rectangles (WM_PER and
/// WN_PER above 1), deep pipelines (stages 3/4), and oversubscribed warps
/// (stage-only warps). The AOT kernel (`kernels/tq2_0_imma.cu`) is the
/// unchanged pinned anchor, so `to_bits` equality here proves the rev-4
/// staging rewrite is a pure perf change.
#[test]
fn imma_jit_staging_matches_aot_bitwise_at_unaligned_k() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping rev-4 staging gate: no device ({e})");
            return;
        }
    };

    let tiles = [
        // 2x2 warp grid, one sub-tile each.
        TileConfig {
            tile_m: 32,
            tile_n: 16,
            tile_k: 64,
            warps: 4,
            stages: 2,
        },
        // 1x8 warp grid, WM_PER=8, triple-buffered.
        TileConfig {
            tile_m: 128,
            tile_n: 64,
            tile_k: 64,
            warps: 8,
            stages: 3,
        },
        // WN_PER=2, quad-buffered.
        TileConfig {
            tile_m: 128,
            tile_n: 128,
            tile_k: 32,
            warps: 8,
            stages: 4,
        },
        // Oversubscribed: 4 warps, single sub-tile (3 warps stage-only).
        TileConfig::BASELINE,
        // tile_k=128 (CH=8, swizzle shift 0 — all three row bits in the XOR):
        // the rev-5 review found this CH was bit-gated only on the aligned
        // cp.async fast path; this tile runs it through the byte-fallback and
        // zero-tail regimes below too.
        TileConfig {
            tile_m: 64,
            tile_n: 32,
            tile_k: 128,
            warps: 4,
            stages: 2,
        },
    ];

    // k=40/88: k%16 != 0 (byte-staging fallback); k=48: k%16 == 0 but
    // k%32 != 0 (cp.async path with a zero-filled tail chunk in k-tile 1);
    // k=96: k%32 == 0 with num_ktiles=3, odd against tile_k=64 — a WHOLE
    // out-of-range k-tile on the cp.async path (B zfill annihilated by zero
    // A; review nit N1). n*k stays a 128-multiple (the I2_S packer's grain).
    for &(m, n, k) in &[
        (33usize, 32usize, 40usize),
        (16, 16, 88),
        (20, 24, 48),
        (20, 24, 96),
    ] {
        let shape = GemmShape::new(m, n, k);
        let raw: Vec<i8> = (0..n * k).map(|i| ((i * 7 + m) % 3) as i8 - 1).collect();
        let qact: Vec<i8> = (0..m * k)
            .map(|i| (((i * 31 + k) % 255) as i32 - 127) as i8)
            .collect();
        let act_scale: Vec<f32> = (0..m).map(|i| 0.01 + (i % 5) as f32 * 0.007).collect();
        let wscale: Vec<f32> = (0..n).map(|j| 1.0 + (j % 3) as f32 * 0.25).collect();
        let num_ktiles = k.div_ceil(32);

        let d_qact = cuda.stream.clone_htod(&qact).expect("qact htod");
        let d_weights = cuda
            .stream
            .clone_htod(&pack_i2s_int8(&raw, shape))
            .expect("weights htod");
        let d_as = cuda.stream.clone_htod(&act_scale).expect("as htod");
        let d_ws = cuda.stream.clone_htod(&wscale).expect("ws htod");

        let run = |tile: TileConfig| -> Vec<f32> {
            let func = cuda.imma_function_for_tile(tile).expect("tile function");
            let mut d_out: cudarc::driver::CudaSlice<f32> =
                cuda.stream.alloc_zeros(m * n).expect("out alloc");
            launch_imma_tile_on(
                &cuda.stream,
                &func,
                tile,
                &d_qact,
                &d_weights,
                &d_as,
                &d_ws,
                &mut d_out,
                m as i32,
                n as i32,
                k as i32,
                num_ktiles as i32,
            )
            .expect("launch");
            cuda.stream.clone_dtoh(&d_out).expect("out dtoh")
        };

        let anchor = run(TileConfig::AOT_EQUIVALENT);
        for tile in tiles {
            let got = run(tile);
            for (i, (&g, &a)) in got.iter().zip(&anchor).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    a.to_bits(),
                    "tile {tile:?} diverges from AOT anchor at ({m},{n},{k})[{i}]: \
                     jit={g} aot={a}"
                );
            }
        }
    }
}

/// v2 prefill attention gate: `gqa_attention_batch_v2_{f32,h}` must reproduce
/// the rev-1 `gqa_attention_batch_{f32,h}` **bit-for-bit** per (row, head) —
/// the v2 kernel's whole license is that it changes the launch/staging
/// mechanics (shared K chunks, shared scores, parallel max/exp) while keeping
/// every pinned fold order (per-key d-order dots, sequential-j softmax sum,
/// per-dim sequential-j V fold with the zero-skip). Regimes: head_dim 64/128,
/// GQA 4:1 and MHA, causal_offset 0 and chunk-continuation offsets, m
/// spanning multiple 64-key stage chunks.
#[test]
fn gqa_attention_batch_v2_matches_rev1_bitwise() {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::Ptx;

    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping attn v2 gate: no device ({e})");
            return;
        }
    };
    let stream = ctx.default_stream();
    let dm = ctx
        .load_module(Ptx::from_src(DECODE_PTX))
        .expect("load decode module");

    // (n_head, n_head_kv, head_dim, causal_offset, m)
    let regimes = [
        (8usize, 2usize, 128usize, 0usize, 70usize), // GQA 4:1, fresh prefill, 2+ chunks
        (8, 2, 128, 130, 70),                        // chunk continuation (ctx to 200)
        (4, 4, 64, 0, 33),                           // MHA, small head_dim
        (20, 5, 128, 500, 12),                       // 2B4T-shaped, deep offset
        (8, 2, 128, 0, 1),                           // single row
        (8, 2, 128, 3500, 84),                       // ctx to exactly ATTN_V2_MAX_CTX
    ];

    let mut seed = 0x2f17u64;
    let mut nextf = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((seed >> 33) as i32 % 1000) as f32) * 1e-3 - 0.5
    };

    for (n_head, n_head_kv, head_dim, causal_offset, m) in regimes {
        let ctx_end = causal_offset + m;
        let q: Vec<f32> = (0..m * n_head * head_dim).map(|_| nextf()).collect();
        let kv_len = ctx_end * n_head_kv * head_dim;
        let kf: Vec<f32> = (0..kv_len).map(|_| nextf()).collect();
        let vf: Vec<f32> = (0..kv_len).map(|_| nextf()).collect();
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let d_q = stream.clone_htod(&q).expect("q");
        let out_len = m * n_head * head_dim;
        let scores_len = m * n_head * ctx_end;

        // f32 pair, then f16 pair (same values through the f16 lattice).
        let kh: Vec<u16> = kf
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let vh: Vec<u16> = vf
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();

        for dtype in ["f32", "h"] {
            let (f_rev1, f_v2) = (
                dm.load_function(&format!("gqa_attention_batch_{dtype}"))
                    .expect("rev1 fn"),
                dm.load_function(&format!("gqa_attention_batch_v2_{dtype}"))
                    .expect("v2 fn"),
            );
            // Upload KV in the dtype under test.
            let (d_k32, d_v32);
            let (d_k16, d_v16);
            let (cm_i, nh_i, nhkv_i, hd_i, co_i, m_i) = (
                ctx_end as i32,
                n_head as i32,
                n_head_kv as i32,
                head_dim as i32,
                causal_offset as i32,
                m as i32,
            );

            let mut d_out1: cudarc::driver::CudaSlice<f32> =
                stream.alloc_zeros(out_len).expect("out1");
            let mut d_out2: cudarc::driver::CudaSlice<f32> =
                stream.alloc_zeros(out_len).expect("out2");
            let mut d_scores: cudarc::driver::CudaSlice<f32> =
                stream.alloc_zeros(scores_len).expect("scores");

            let cfg1 = LaunchConfig {
                grid_dim: (((m * n_head) as u32).div_ceil(8), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let cfg2 = LaunchConfig {
                grid_dim: (n_head as u32, m as u32, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };

            #[allow(unsafe_code)]
            match dtype {
                "f32" => {
                    d_k32 = stream.clone_htod(&kf).expect("k32");
                    d_v32 = stream.clone_htod(&vf).expect("v32");
                    let mut l1 = stream.launch_builder(&f_rev1);
                    l1.arg(&d_q)
                        .arg(&d_k32)
                        .arg(&d_v32)
                        .arg(&mut d_out1)
                        .arg(&mut d_scores)
                        .arg(&cm_i)
                        .arg(&nh_i)
                        .arg(&nhkv_i)
                        .arg(&hd_i)
                        .arg(&scale)
                        .arg(&co_i)
                        .arg(&m_i);
                    // SAFETY: rev-1 batch signature with ctx_max == ctx_end scratch.
                    unsafe { l1.launch(cfg1).expect("rev1 f32") };
                    let mut l2 = stream.launch_builder(&f_v2);
                    l2.arg(&d_q)
                        .arg(&d_k32)
                        .arg(&d_v32)
                        .arg(&mut d_out2)
                        .arg(&nh_i)
                        .arg(&nhkv_i)
                        .arg(&hd_i)
                        .arg(&scale)
                        .arg(&co_i)
                        .arg(&m_i);
                    // SAFETY: v2 signature (no scores scratch).
                    unsafe { l2.launch(cfg2).expect("v2 f32") };
                }
                _ => {
                    d_k16 = stream.clone_htod(&kh).expect("k16");
                    d_v16 = stream.clone_htod(&vh).expect("v16");
                    let mut l1 = stream.launch_builder(&f_rev1);
                    l1.arg(&d_q)
                        .arg(&d_k16)
                        .arg(&d_v16)
                        .arg(&mut d_out1)
                        .arg(&mut d_scores)
                        .arg(&cm_i)
                        .arg(&nh_i)
                        .arg(&nhkv_i)
                        .arg(&hd_i)
                        .arg(&scale)
                        .arg(&co_i)
                        .arg(&m_i);
                    // SAFETY: rev-1 f16 batch signature.
                    unsafe { l1.launch(cfg1).expect("rev1 h") };
                    let mut l2 = stream.launch_builder(&f_v2);
                    l2.arg(&d_q)
                        .arg(&d_k16)
                        .arg(&d_v16)
                        .arg(&mut d_out2)
                        .arg(&nh_i)
                        .arg(&nhkv_i)
                        .arg(&hd_i)
                        .arg(&scale)
                        .arg(&co_i)
                        .arg(&m_i);
                    // SAFETY: v2 f16 signature.
                    unsafe { l2.launch(cfg2).expect("v2 h") };
                }
            }

            let o1 = stream.clone_dtoh(&d_out1).expect("dtoh 1");
            let o2 = stream.clone_dtoh(&d_out2).expect("dtoh 2");
            for (i, (&a, &b)) in o1.iter().zip(&o2).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "attn v2 diverges from rev1 ({dtype}, nh={n_head} nhkv={n_head_kv} \
                     hd={head_dim} co={causal_offset} m={m}) at [{i}]: rev1={a} v2={b}"
                );
            }
        }
    }
}

/// v3 Q-blocked prefill attention gate: bit-identical to rev-1 per
/// (row, head) — same license as the v2 gate, plus the regimes v3 uniquely
/// owns: BQ-tail blocks (m % 8 != 0), the causal staircase from offset 0
/// (rows attending 1..m keys, exercising the per-(row, key) predicate and
/// the zero-staged weights past each row's limit), and ctx BEYOND the v2
/// shared cap (v3 has no ctx bound — scores live in the global scratch).
#[test]
fn gqa_attention_batch_v3_matches_rev1_bitwise() {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::Ptx;

    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping attn v3 gate: no device ({e})");
            return;
        }
    };
    let stream = ctx.default_stream();
    let dm = ctx
        .load_module(Ptx::from_src(DECODE_PTX))
        .expect("load decode module");

    // (n_head, n_head_kv, head_dim, causal_offset, m)
    let regimes = [
        (8usize, 2usize, 128usize, 0usize, 70usize), // staircase from 0, BQ tail (70 = 8*8+6)
        (20, 5, 128, 500, 12),                       // 2B4T-shaped, tail block of 4
        (4, 4, 64, 130, 33),                         // MHA, hd 64, tail of 1
        (8, 2, 128, 3800, 9),                        // ctx 3809 — PAST the v2 cap
        (8, 2, 128, 0, 1),                           // single row
    ];

    let mut seed = 0x3a11u64;
    let mut nextf = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((seed >> 33) as i32 % 1000) as f32) * 1e-3 - 0.5
    };

    for (n_head, n_head_kv, head_dim, causal_offset, m) in regimes {
        let ctx_end = causal_offset + m;
        let q: Vec<f32> = (0..m * n_head * head_dim).map(|_| nextf()).collect();
        let kv_len = ctx_end * n_head_kv * head_dim;
        let kf: Vec<f32> = (0..kv_len).map(|_| nextf()).collect();
        let vf: Vec<f32> = (0..kv_len).map(|_| nextf()).collect();
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let kh: Vec<u16> = kf
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let vh: Vec<u16> = vf
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();

        let d_q = stream.clone_htod(&q).expect("q");
        let out_len = m * n_head * head_dim;
        let scores_len = m * n_head * ctx_end;
        let (cm_i, nh_i, nhkv_i, hd_i, co_i, m_i) = (
            ctx_end as i32,
            n_head as i32,
            n_head_kv as i32,
            head_dim as i32,
            causal_offset as i32,
            m as i32,
        );

        let cfg1 = LaunchConfig {
            grid_dim: (((m * n_head) as u32).div_ceil(8), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // v3 geometry comes from the pinned consts (they mirror the decode.cu
        // defines by test) — a hardcoded block_dim here silently under-runs
        // the kernel's compile-time thread-count strides if the tune changes.
        let cfg3 = LaunchConfig {
            grid_dim: (
                n_head as u32,
                (m as u32).div_ceil(consts::ATTN_V3_BQ as u32),
                1,
            ),
            block_dim: (consts::ATTN_V3_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };

        for dtype in ["f32", "h"] {
            let f_rev1 = dm
                .load_function(&format!("gqa_attention_batch_{dtype}"))
                .expect("rev1 fn");
            let f_v3 = dm
                .load_function(&format!("gqa_attention_batch_v3_{dtype}"))
                .expect("v3 fn");
            let mut d_out1: cudarc::driver::CudaSlice<f32> =
                stream.alloc_zeros(out_len).expect("out1");
            let mut d_out3: cudarc::driver::CudaSlice<f32> =
                stream.alloc_zeros(out_len).expect("out3");
            let mut d_sc1: cudarc::driver::CudaSlice<f32> =
                stream.alloc_zeros(scores_len).expect("sc1");
            let mut d_sc3: cudarc::driver::CudaSlice<f32> =
                stream.alloc_zeros(scores_len).expect("sc3");

            macro_rules! launch_pair {
                ($dk:expr, $dv:expr) => {{
                    let mut l1 = stream.launch_builder(&f_rev1);
                    l1.arg(&d_q)
                        .arg($dk)
                        .arg($dv)
                        .arg(&mut d_out1)
                        .arg(&mut d_sc1)
                        .arg(&cm_i)
                        .arg(&nh_i)
                        .arg(&nhkv_i)
                        .arg(&hd_i)
                        .arg(&scale)
                        .arg(&co_i)
                        .arg(&m_i);
                    // SAFETY: rev-1 batch signature, ctx_max == ctx_end stride.
                    #[allow(unsafe_code)]
                    unsafe {
                        l1.launch(cfg1).expect("rev1")
                    };
                    let mut l3 = stream.launch_builder(&f_v3);
                    l3.arg(&d_q)
                        .arg($dk)
                        .arg($dv)
                        .arg(&mut d_out3)
                        .arg(&mut d_sc3)
                        .arg(&cm_i)
                        .arg(&nh_i)
                        .arg(&nhkv_i)
                        .arg(&hd_i)
                        .arg(&scale)
                        .arg(&co_i)
                        .arg(&m_i);
                    // SAFETY: v3 signature (same as rev-1's).
                    #[allow(unsafe_code)]
                    unsafe {
                        l3.launch(cfg3).expect("v3")
                    };
                }};
            }
            if dtype == "f32" {
                let d_k = stream.clone_htod(&kf).expect("k32");
                let d_v = stream.clone_htod(&vf).expect("v32");
                launch_pair!(&d_k, &d_v);
            } else {
                let d_k = stream.clone_htod(&kh).expect("k16");
                let d_v = stream.clone_htod(&vh).expect("v16");
                launch_pair!(&d_k, &d_v);
            }

            let o1 = stream.clone_dtoh(&d_out1).expect("dtoh 1");
            let o3 = stream.clone_dtoh(&d_out3).expect("dtoh 3");
            for (i, (&a, &b)) in o1.iter().zip(&o3).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "attn v3 diverges from rev1 ({dtype}, nh={n_head} nhkv={n_head_kv} \
                     hd={head_dim} co={causal_offset} m={m}) at [{i}]: rev1={a} v3={b}"
                );
            }
        }
    }
}

/// Review-66d8f58 F1 gate: the f16 ctrl tree-verify pair
/// (`gqa_attention_tree_{scores,reduce}_ctrl_h`) dispatches to the `_f16w`
/// wide-load bodies, which carry launch-uniform fallbacks to the generic
/// `<KvLoadF16>` ctrl bodies for shapes outside the wide contract
/// (scores: head_dim % 8 != 0; reduce: head_dim odd or > pair budget).
/// Every acceptance gate runs hd=128, so neither fallback nor the wide
/// paths at a non-128 shape had a test execution. Anchor: the EAGER
/// non-ctrl `_h` twins (`gqa_attention_tree_{scores,reduce}_h`) — a
/// different entry point through the generic template bodies, with
/// scalar (prefix_len, m) args and score stride = ctx_max; the ctrl pair
/// gets an equivalent device ctrl `[prefix_len, m, 0]` and score_stride =
/// ctx_max, so bit-equality pins ctrl==non-ctrl AND wide==generic:
///   * hd=100 (%4==0, %8!=0): scores takes the FALLBACK, reduce the wide
///     pair path.
///   * hd=104 (%8==0): both take the WIDE path at a non-128 shape.
///   * hd=101 (odd, reduce-only on host-synthesized scores — the scores
///     bodies require head_dim % 4 == 0): reduce takes the FALLBACK.
#[test]
fn tree_ctrl_f16w_wide_and_fallback_match_nonctrl_h_bitwise() {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::Ptx;

    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping tree ctrl f16w gate: no device ({e})");
            return;
        }
    };
    let stream = ctx.default_stream();
    let dm = ctx
        .load_module(Ptx::from_src(DECODE_PTX))
        .expect("load decode module");
    let f_scores_ctrl = dm
        .load_function("gqa_attention_tree_scores_ctrl_h")
        .expect("ctrl scores fn");
    let f_reduce_ctrl = dm
        .load_function("gqa_attention_tree_reduce_ctrl_h")
        .expect("ctrl reduce fn");
    let f_scores = dm
        .load_function("gqa_attention_tree_scores_h")
        .expect("scores fn");
    let f_reduce = dm
        .load_function("gqa_attention_tree_reduce_h")
        .expect("reduce fn");

    // Small tree: 3 real rows with varying ancestor counts over a 16-row
    // KV arena; prefix 5, ctx_max 16 (also the score stride for BOTH paths).
    let (n_head, n_head_kv, m, prefix_len, max_anc, ctx_max, kv_rows) =
        (4usize, 2usize, 3usize, 5usize, 4usize, 16usize, 16usize);
    let n_anc: Vec<i32> = vec![2, 4, 3];
    #[rustfmt::skip]
    let anc: Vec<i32> = vec![
        5, 7, 0, 0,    // row 0: 2 ancestors
        6, 8, 11, 14,  // row 1: 4
        5, 9, 13, 0,   // row 2: 3
    ];
    let tree_ctrl: Vec<i32> = vec![prefix_len as i32, m as i32, 0];

    let mut seed = 0x66d8f58u64;
    let mut nextf = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((seed >> 33) as i32 % 1000) as f32) * 1e-3 - 0.5
    };

    let d_anc = stream.clone_htod(&anc).expect("anc");
    let d_n_anc = stream.clone_htod(&n_anc).expect("n_anc");
    let d_ctrl = stream.clone_htod(&tree_ctrl).expect("tree_ctrl");
    let (cm_i, ss_i, nh_i, nhkv_i, pl_i, ma_i, m_i) = (
        ctx_max as i32,
        ctx_max as i32, // ctrl score_stride == the non-ctrl path's ctx_max
        n_head as i32,
        n_head_kv as i32,
        prefix_len as i32,
        max_anc as i32,
        m as i32,
    );
    let scores_cfg = LaunchConfig {
        grid_dim: ((m * n_head) as u32, ctx_max.div_ceil(128) as u32, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let reduce_cfg = LaunchConfig {
        grid_dim: ((m * n_head) as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: (ctx_max * 4) as u32,
    };
    let scores_len = m * n_head * ctx_max;

    // Part 1: full scores→reduce chain at hd=100 (scores FALLBACK + wide
    // reduce) and hd=104 (wide scores + wide reduce, non-128).
    for head_dim in [100usize, 104usize] {
        let hd_i = head_dim as i32;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let q: Vec<f32> = (0..m * n_head * head_dim).map(|_| nextf()).collect();
        let kv_len = kv_rows * n_head_kv * head_dim;
        let kh: Vec<u16> = (0..kv_len)
            .map(|_| f16::from_f32(nextf()).to_bits())
            .collect();
        let vh: Vec<u16> = (0..kv_len)
            .map(|_| f16::from_f32(nextf()).to_bits())
            .collect();
        let d_q = stream.clone_htod(&q).expect("q");
        let d_k = stream.clone_htod(&kh).expect("k16");
        let d_v = stream.clone_htod(&vh).expect("v16");

        let out_len = m * n_head * head_dim;
        // alloc_zeros: entries past each row's live ctx stay 0 on both
        // paths, so full-buffer comparison is well-defined.
        let mut d_sc_ctrl: cudarc::driver::CudaSlice<f32> =
            stream.alloc_zeros(scores_len).expect("sc ctrl");
        let mut d_sc_ref: cudarc::driver::CudaSlice<f32> =
            stream.alloc_zeros(scores_len).expect("sc ref");
        let mut d_out_ctrl: cudarc::driver::CudaSlice<f32> =
            stream.alloc_zeros(out_len).expect("out ctrl");
        let mut d_out_ref: cudarc::driver::CudaSlice<f32> =
            stream.alloc_zeros(out_len).expect("out ref");

        #[allow(unsafe_code)]
        {
            let mut l = stream.launch_builder(&f_scores_ctrl);
            l.arg(&d_q)
                .arg(&d_k)
                .arg(&mut d_sc_ctrl)
                .arg(&d_anc)
                .arg(&d_n_anc)
                .arg(&d_ctrl)
                .arg(&ss_i)
                .arg(&nh_i)
                .arg(&nhkv_i)
                .arg(&hd_i)
                .arg(&scale)
                .arg(&ma_i)
                .arg(&m_i);
            // SAFETY: ctrl scores signature; one-warp blocks per contract.
            unsafe { l.launch(scores_cfg).expect("ctrl scores") };
            let mut l = stream.launch_builder(&f_scores);
            l.arg(&d_q)
                .arg(&d_k)
                .arg(&mut d_sc_ref)
                .arg(&d_anc)
                .arg(&d_n_anc)
                .arg(&cm_i)
                .arg(&nh_i)
                .arg(&nhkv_i)
                .arg(&hd_i)
                .arg(&scale)
                .arg(&pl_i)
                .arg(&ma_i)
                .arg(&m_i);
            // SAFETY: non-ctrl scores signature (scalar prefix_len).
            unsafe { l.launch(scores_cfg).expect("ref scores") };
            let mut l = stream.launch_builder(&f_reduce_ctrl);
            l.arg(&d_v)
                .arg(&d_sc_ctrl)
                .arg(&mut d_out_ctrl)
                .arg(&d_anc)
                .arg(&d_n_anc)
                .arg(&d_ctrl)
                .arg(&ss_i)
                .arg(&nh_i)
                .arg(&nhkv_i)
                .arg(&hd_i)
                .arg(&ma_i)
                .arg(&m_i);
            // SAFETY: ctrl reduce signature; blockDim.x == 128 per contract.
            unsafe { l.launch(reduce_cfg).expect("ctrl reduce") };
            let mut l = stream.launch_builder(&f_reduce);
            l.arg(&d_v)
                .arg(&d_sc_ref)
                .arg(&mut d_out_ref)
                .arg(&d_anc)
                .arg(&d_n_anc)
                .arg(&cm_i)
                .arg(&nh_i)
                .arg(&nhkv_i)
                .arg(&hd_i)
                .arg(&pl_i)
                .arg(&ma_i)
                .arg(&m_i);
            // SAFETY: non-ctrl reduce signature; blockDim.x == 128 per contract.
            unsafe { l.launch(reduce_cfg).expect("ref reduce") };
        }

        let sc_c = stream.clone_dtoh(&d_sc_ctrl).expect("dtoh sc ctrl");
        let sc_r = stream.clone_dtoh(&d_sc_ref).expect("dtoh sc ref");
        for (i, (&a, &b)) in sc_c.iter().zip(&sc_r).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "ctrl_h scores diverge from non-ctrl _h (hd={head_dim}) at [{i}]: \
                 ctrl={a} ref={b}"
            );
        }
        let o_c = stream.clone_dtoh(&d_out_ctrl).expect("dtoh out ctrl");
        let o_r = stream.clone_dtoh(&d_out_ref).expect("dtoh out ref");
        for (i, (&a, &b)) in o_c.iter().zip(&o_r).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "ctrl_h reduce diverges from non-ctrl _h (hd={head_dim}) at [{i}]: \
                 ctrl={a} ref={b}"
            );
        }
    }

    // Part 2: reduce FALLBACK guard (head_dim odd) on host-synthesized
    // scores — both kernels read the same buffer, so this pins the guarded
    // generic ctrl body against the non-ctrl generic body.
    {
        let head_dim = 101usize;
        let hd_i = head_dim as i32;
        let kv_len = kv_rows * n_head_kv * head_dim;
        let vh: Vec<u16> = (0..kv_len)
            .map(|_| f16::from_f32(nextf()).to_bits())
            .collect();
        let sc_host: Vec<f32> = (0..scores_len).map(|_| nextf()).collect();
        let d_v = stream.clone_htod(&vh).expect("v16 odd");
        let d_sc = stream.clone_htod(&sc_host).expect("sc odd");
        let out_len = m * n_head * head_dim;
        let mut d_out_ctrl: cudarc::driver::CudaSlice<f32> =
            stream.alloc_zeros(out_len).expect("out ctrl odd");
        let mut d_out_ref: cudarc::driver::CudaSlice<f32> =
            stream.alloc_zeros(out_len).expect("out ref odd");

        #[allow(unsafe_code)]
        {
            let mut l = stream.launch_builder(&f_reduce_ctrl);
            l.arg(&d_v)
                .arg(&d_sc)
                .arg(&mut d_out_ctrl)
                .arg(&d_anc)
                .arg(&d_n_anc)
                .arg(&d_ctrl)
                .arg(&ss_i)
                .arg(&nh_i)
                .arg(&nhkv_i)
                .arg(&hd_i)
                .arg(&ma_i)
                .arg(&m_i);
            // SAFETY: ctrl reduce signature; blockDim.x == 128 per contract.
            unsafe { l.launch(reduce_cfg).expect("ctrl reduce odd") };
            let mut l = stream.launch_builder(&f_reduce);
            l.arg(&d_v)
                .arg(&d_sc)
                .arg(&mut d_out_ref)
                .arg(&d_anc)
                .arg(&d_n_anc)
                .arg(&cm_i)
                .arg(&nh_i)
                .arg(&nhkv_i)
                .arg(&hd_i)
                .arg(&pl_i)
                .arg(&ma_i)
                .arg(&m_i);
            // SAFETY: non-ctrl reduce signature; blockDim.x == 128 per contract.
            unsafe { l.launch(reduce_cfg).expect("ref reduce odd") };
        }

        let o_c = stream.clone_dtoh(&d_out_ctrl).expect("dtoh out ctrl odd");
        let o_r = stream.clone_dtoh(&d_out_ref).expect("dtoh out ref odd");
        for (i, (&a, &b)) in o_c.iter().zip(&o_r).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "ctrl_h reduce fallback diverges from non-ctrl _h (hd=101, odd) \
                 at [{i}]: ctrl={a} ref={b}"
            );
        }
    }
}

/// v0.3.1 de-risk: the device `rmsnorm_f32` decode kernel must reproduce the host
/// `tritium_nn::ops::rmsnorm` **bit-for-bit** (`to_bits` equal), so the fully
/// device-resident forward keeps greedy 256/256. This is the proof that a
/// sequential-f32 + FMA-disabled device kernel can match host f32 exactly; the
/// rest of the decode kernels follow the same discipline.
#[test]
fn rmsnorm_bit_matches_host() {
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping rmsnorm bit-match: no device ({e})");
            return;
        }
    };
    // Host reference — identical to `tritium_nn::ops::rmsnorm` (this crate does
    // not depend on tritium-nn, so the formula is replicated verbatim).
    // Canonical tree sum-of-squares (ADR 0018) — replicates
    // `tritium_nn::ops::rmsnorm`'s documented cross-backend order (this
    // crate does not depend on tritium-nn).
    fn sum_squares_canonical(x: &[f32]) -> f32 {
        let mut part = [0.0f32; 256];
        for (i, &v) in x.iter().enumerate() {
            part[i % 256] += v * v;
        }
        let mut off = 128;
        while off > 0 {
            for t in 0..off {
                part[t] += part[t + off];
            }
            off >>= 1;
        }
        part[0]
    }
    fn host_rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        let n = x.len();
        let mean_sq = sum_squares_canonical(x) / n as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        x.iter().zip(w).map(|(&xi, &wi)| xi * inv * wi).collect()
    }
    // BitNet hidden/ffn sizes + a few edge lengths; deterministic xorshift inputs.
    for &n in &[2560usize, 6912, 1, 17, 256, 2559] {
        let mut s = 0x1234_5678_9abc_def0u64 ^ (n as u64).wrapping_mul(0x9E37_79B9);
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f32 / (1u64 << 53) as f32) * 8.0 - 4.0
        };
        let x: Vec<f32> = (0..n).map(|_| next()).collect();
        let w: Vec<f32> = (0..n).map(|_| next()).collect();
        let eps = 1e-5f32;

        let want = host_rmsnorm(&x, &w, eps);
        let mut got = vec![0.0f32; n];
        backend
            .rmsnorm(&x, &w, eps, &mut got)
            .expect("device rmsnorm");

        for (i, (&g, &h)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.to_bits(),
                h.to_bits(),
                "rmsnorm bit mismatch n={n} i={i}: got {g} ({:#010x}) want {h} ({:#010x})",
                g.to_bits(),
                h.to_bits()
            );
        }
    }
}

/// The device `rope_apply_f32` kernel must reproduce `tritium_nn::ops::rope_apply`
/// **bit-for-bit** for one token (M=1 decode). The trig is computed exactly as the
/// host op (f64 `sin_cos` → f32, data-independent) and the f32 rotation has no FMA.
#[test]
fn rope_bit_matches_host() {
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping rope bit-match: no device ({e})");
            return;
        }
    };
    // BitNet 2B4T uses head_dim=128, n_head 20(Q)/5(KV), theta=500000.
    for &(n_head, head_dim) in &[(20usize, 128usize), (5, 128), (1, 8), (3, 64)] {
        let half = head_dim / 2;
        let theta = 500_000.0f32;
        for &pos in &[0usize, 1, 7, 255, 4095] {
            // Trig tables, identical to the host op (f64 sin_cos cast to f32).
            let theta_f64 = f64::from(theta);
            let inv_hd = 1.0 / head_dim as f64;
            let mut cos_t = vec![0.0f32; half];
            let mut sin_t = vec![0.0f32; half];
            for j in 0..half {
                let inv_freq = theta_f64.powf(-2.0 * j as f64 * inv_hd);
                let (s, c) = (pos as f64 * inv_freq).sin_cos();
                cos_t[j] = c as f32;
                sin_t[j] = s as f32;
            }
            // Deterministic input.
            let mut st = 0xDEAD_BEEF_CAFE_F00Du64
                ^ ((pos as u64) * 131 + n_head as u64 * 17 + head_dim as u64);
            let mut next = || {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                ((st >> 11) as f32 / (1u64 << 53) as f32) * 8.0 - 4.0
            };
            let x0: Vec<f32> = (0..n_head * head_dim).map(|_| next()).collect();

            // Host rope (replicated; Rust does not auto-contract a*c - b*s to FMA).
            let mut want = x0.clone();
            for head in 0..n_head {
                let base = head * head_dim;
                for j in 0..half {
                    let a = x0[base + j];
                    let b = x0[base + j + half];
                    want[base + j] = a * cos_t[j] - b * sin_t[j];
                    want[base + j + half] = b * cos_t[j] + a * sin_t[j];
                }
            }

            let mut got = x0.clone();
            backend
                .rope(&mut got, &cos_t, &sin_t, n_head, head_dim)
                .expect("device rope");

            for (i, (&g, &h)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    h.to_bits(),
                    "rope bit mismatch (n_head={n_head} head_dim={head_dim} pos={pos}) i={i}: got {g} want {h}"
                );
            }
        }
    }
}

/// Measure device softmax vs host `softmax_rows`. The reductions are bit-matched;
/// the open question is `expf` (device CUDA libm vs host glibc). Reports the max
/// ULP difference + whether bit-exact, and asserts a tight relative tolerance so
/// the result is informative without spuriously failing on a ~1-ULP exp delta.
/// This is the gate-deciding measurement: bit-exact ⇒ strict greedy 256/256 is
/// reachable; otherwise the forward uses the perplexity+lockstep fallback.
#[test]
fn softmax_vs_host_exp_divergence() {
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping softmax divergence: no device ({e})");
            return;
        }
    };
    fn host_softmax(x: &mut [f32], row_len: usize) {
        for row in x.chunks_mut(row_len) {
            let mut m = f32::NEG_INFINITY;
            for &v in row.iter() {
                if v > m {
                    m = v;
                }
            }
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                let e = (*v - m).exp();
                *v = e;
                sum += e;
            }
            let inv = 1.0f32 / sum;
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
    }
    let (rows, row_len) = (20usize, 1024usize); // decode-ish: n_head × ctx
    let mut s = 0x5151_5151_2727_2727u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f32 / (1u64 << 53) as f32) * 16.0 - 8.0
    };
    let x0: Vec<f32> = (0..rows * row_len).map(|_| next()).collect();
    let mut want = x0.clone();
    host_softmax(&mut want, row_len);
    let mut got = x0.clone();
    backend
        .softmax(&mut got, row_len, rows)
        .expect("device softmax");

    let (mut max_ulp, mut n_diff, mut max_rel) = (0i64, 0usize, 0.0f64);
    for (&g, &h) in got.iter().zip(&want) {
        let du = (i64::from(g.to_bits()) - i64::from(h.to_bits())).abs();
        if du != 0 {
            n_diff += 1;
        }
        max_ulp = max_ulp.max(du);
        if h != 0.0 {
            max_rel = max_rel.max((f64::from(g - h) / f64::from(h)).abs());
        }
    }
    eprintln!(
        "softmax device-vs-host: max_ulp={max_ulp} n_diff={n_diff}/{} max_rel={max_rel:.3e} bit_exact={}",
        got.len(),
        n_diff == 0
    );
    assert!(
        max_rel < 1e-5,
        "device softmax exp diverges too far from host: max_rel={max_rel:.3e}"
    );
}

/// `residual_add` / `embedding_gather` / `lm_head` must match host bit-for-bit:
/// the first two are exact (add / copy), the LM head reproduces the host's
/// sequential dot in k-order (no FMA).
#[test]
fn residual_embed_lmhead_bit_match_host() {
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping residual/embed/lm_head bit-match: no device ({e})");
            return;
        }
    };
    let mut s = 0xABCD_1234_5678_9876u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f32 / (1u64 << 53) as f32) * 8.0 - 4.0
    };

    // residual_add: x += y (exact).
    {
        let n = 2560usize;
        let x0: Vec<f32> = (0..n).map(|_| next()).collect();
        let y: Vec<f32> = (0..n).map(|_| next()).collect();
        let want: Vec<f32> = x0.iter().zip(&y).map(|(&a, &b)| a + b).collect();
        let mut got = x0.clone();
        backend.residual_add(&mut got, &y).expect("residual");
        for (i, (&g, &h)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), h.to_bits(), "residual_add mismatch [{i}]");
        }
    }

    // embedding_gather: out = table[tok] (exact copy).
    {
        let (vocab, n_embd) = (64usize, 256usize);
        let table: Vec<f32> = (0..vocab * n_embd).map(|_| next()).collect();
        let tok = 37usize;
        let want = &table[tok * n_embd..tok * n_embd + n_embd];
        let mut got = vec![0.0f32; n_embd];
        backend
            .embedding_gather(&table, tok, n_embd, &mut got)
            .expect("embed");
        for (i, (&g, &h)) in got.iter().zip(want).enumerate() {
            assert_eq!(g.to_bits(), h.to_bits(), "embedding_gather mismatch [{i}]");
        }
    }

    // lm_head: sequential dot, bit-exact.
    {
        let (vocab, n_embd) = (128usize, 2560usize);
        let h: Vec<f32> = (0..n_embd).map(|_| next()).collect();
        let embd: Vec<f32> = (0..vocab * n_embd).map(|_| next()).collect();
        let mut want = vec![0.0f32; vocab];
        for (v, slot) in want.iter_mut().enumerate() {
            let row = &embd[v * n_embd..v * n_embd + n_embd];
            let mut acc = 0.0f32;
            for k in 0..n_embd {
                acc += h[k] * row[k];
            }
            *slot = acc;
        }
        let mut got = vec![0.0f32; vocab];
        backend
            .lm_head(&h, &embd, n_embd, vocab, &mut got)
            .expect("lm_head");
        for (v, (&g, &hh)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.to_bits(),
                hh.to_bits(),
                "lm_head mismatch [{v}]: got {g} want {hh}"
            );
        }
    }
}

/// ADR 0036 L2 i8 head rung drift gate: `lm_head_warp_i8` must reproduce a host
/// oracle that emulates the kernel's PINNED fold order exactly (per lane
/// k = lane + 32·t ascending, per element `fadd(acc, fmul(scale, fmul(h, q)))`,
/// then the 16/8/4/2/1 shuffle tree), and `lm_head_tiled_i8` must match the warp
/// twin bit-for-bit per row (same per-element order by construction — scale and
/// q deliberately NOT pre-folded). The quantizer here mirrors
/// `build_decode_model`'s: per-64-group absmax/127, round half-away, clamp ±127.
#[test]
fn lm_head_i8_warp_and_tiled_match_host_oracle_bitwise() {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::Ptx;

    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping lm_head i8 gate: no device ({e})");
            return;
        }
    };
    let stream = ctx.default_stream();
    let dm = ctx
        .load_module(Ptx::from_src(DECODE_PTX))
        .expect("load decode module");
    let f_warp = dm.load_function("lm_head_warp_i8").expect("warp i8 fn");
    let f_tiled = dm.load_function("lm_head_tiled_i8").expect("tiled i8 fn");

    // Odd vocab exercises the tail warp; n_embd = 2560 is the 2B4T shape
    // (40 groups); m = 13 exercises a partial row tile.
    let (vocab, n_embd, m) = (257usize, 2560usize, 13usize);
    const G: usize = 64;
    let n_groups = n_embd / G;

    let mut s = 0x51AB_77E1_0F3C_2D19u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f32 / (1u64 << 53) as f32) * 4.0 - 2.0
    };
    let table: Vec<f32> = (0..vocab * n_embd).map(|_| next()).collect();
    let h: Vec<f32> = (0..m * n_embd).map(|_| next()).collect();

    // Host quantizer — MUST mirror build_decode_model's exactly.
    let mut q = vec![0i8; vocab * n_embd];
    let mut sc = vec![0.0f32; vocab * n_groups];
    for v in 0..vocab {
        let row = &table[v * n_embd..(v + 1) * n_embd];
        for g in 0..n_groups {
            let grp = &row[g * G..(g + 1) * G];
            let absmax = grp.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
            let scale = absmax / 127.0;
            sc[v * n_groups + g] = scale;
            if scale > 0.0 {
                for (j, &x) in grp.iter().enumerate() {
                    q[v * n_embd + g * G + j] = (x / scale).round().clamp(-127.0, 127.0) as i8;
                }
            }
        }
    }

    // Host oracle in the kernel's exact fp order (Rust f32 * / + are the same
    // correctly-rounded ops as __fmul_rn/__fadd_rn).
    let oracle_row = |hrow: &[f32], v: usize| -> f32 {
        let qrow = &q[v * n_embd..(v + 1) * n_embd];
        let srow = &sc[v * n_groups..(v + 1) * n_groups];
        let mut lanes = [0.0f32; 32];
        let trips = n_embd / 32;
        for (lane, acc) in lanes.iter_mut().enumerate() {
            for t in 0..trips {
                let k = lane + 32 * t;
                *acc += srow[t / 2] * (hrow[k] * f32::from(qrow[k]));
            }
        }
        for off in [16usize, 8, 4, 2, 1] {
            for l in 0..off {
                lanes[l] += lanes[l + off];
            }
        }
        lanes[0]
    };

    let d_q = stream.clone_htod(&q).expect("q htod");
    let d_sc = stream.clone_htod(&sc).expect("scales htod");
    let d_h = stream.clone_htod(&h).expect("h htod");
    let (ne_i, v_i, m_i) = (n_embd as i32, vocab as i32, m as i32);

    // Warp head over row 0's hidden state.
    let d_h0 = stream.clone_htod(&h[..n_embd]).expect("h0 htod");
    let mut d_logits: cudarc::driver::CudaSlice<f32> = stream.alloc_zeros(vocab).expect("logits");
    let cfg_warp = LaunchConfig {
        grid_dim: ((vocab as u32).div_ceil(8), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    {
        let mut l = stream.launch_builder(&f_warp);
        l.arg(&d_h0)
            .arg(&d_q)
            .arg(&d_sc)
            .arg(&ne_i)
            .arg(&v_i)
            .arg(&mut d_logits);
        #[allow(unsafe_code)]
        // SAFETY: kernel launch with matching arity/types; buffers sized
        // above outlive the launch and the following sync.
        unsafe {
            l.launch(cfg_warp).expect("launch warp i8");
        }
    }
    let mut got_warp = vec![0.0f32; vocab];
    stream.memcpy_dtoh(&d_logits, &mut got_warp).expect("dtoh");
    for (v, got) in got_warp.iter().enumerate() {
        let want = oracle_row(&h[..n_embd], v);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "lm_head_warp_i8 diverges from host oracle at [{v}]: got {got} want {want}"
        );
    }

    // Tiled head over all m rows — every row must equal the oracle (and hence
    // the warp twin) bit-for-bit.
    let mut d_logits_all: cudarc::driver::CudaSlice<f32> =
        stream.alloc_zeros(m * vocab).expect("logits all");
    let cfg_tiled = LaunchConfig {
        grid_dim: (
            ((vocab * 32) as u32).div_ceil(256),
            (m as u32).div_ceil(8), // LMHEAD_ROW_TILE
            1,
        ),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    {
        let mut l = stream.launch_builder(&f_tiled);
        l.arg(&d_h)
            .arg(&d_q)
            .arg(&d_sc)
            .arg(&ne_i)
            .arg(&v_i)
            .arg(&m_i)
            .arg(&mut d_logits_all);
        #[allow(unsafe_code)]
        // SAFETY: kernel launch with matching arity/types; buffers sized
        // above outlive the launch and the following sync.
        unsafe {
            l.launch(cfg_tiled).expect("launch tiled i8");
        }
    }
    let mut got_tiled = vec![0.0f32; m * vocab];
    stream
        .memcpy_dtoh(&d_logits_all, &mut got_tiled)
        .expect("dtoh tiled");
    for mi in 0..m {
        let hrow = &h[mi * n_embd..(mi + 1) * n_embd];
        for v in 0..vocab {
            let want = oracle_row(hrow, v);
            assert_eq!(
                got_tiled[mi * vocab + v].to_bits(),
                want.to_bits(),
                "lm_head_tiled_i8 diverges from host oracle at [{mi},{v}]"
            );
        }
    }
    eprintln!(
        "lm_head i8 drift gate: warp {vocab} rows + tiled {m}x{vocab} rows all bit-exact vs host oracle"
    );
}

/// ADR 0036 L2 head-format kernel A/B at equal M=1 shapes (the ≥1.6× gate):
/// `lm_head_warp_i8` vs `lm_head_warp_f16` on the 2B4T head shape
/// (vocab 128256 × n_embd 2560 — 656 MB f16 / 328 MB i8 + 20.5 MB scales, both
/// far beyond L2, so the timing is naturally L2-defeated). Same-session ABBA
/// (f16, i8, i8, f16), 40 timed launches per leg after warmup.
#[test]
#[ignore = "perf bench: needs a quiet GPU box; run explicitly"]
fn lm_head_i8_vs_f16_m1_abba_bench() {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::Ptx;
    use std::time::Instant;

    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping lm_head i8 bench: no device ({e})");
            return;
        }
    };
    let stream = ctx.default_stream();
    let dm = ctx
        .load_module(Ptx::from_src(DECODE_PTX))
        .expect("load decode module");
    let f_f16 = dm.load_function("lm_head_warp_f16").expect("f16 fn");
    let f_i8 = dm.load_function("lm_head_warp_i8").expect("i8 fn");

    let (vocab, n_embd) = (128_256usize, 2560usize);
    const G: usize = 64;
    let n_groups = n_embd / G;

    let mut s = 0xBEE5_1234u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
    };
    // Table filled row-block-wise to keep host memory modest.
    let mut table_f16 = vec![0u16; vocab * n_embd];
    let mut q = vec![0i8; vocab * n_embd];
    let mut sc = vec![0.0f32; vocab * n_groups];
    for v in 0..vocab {
        let mut row = vec![0.0f32; n_embd];
        for x in row.iter_mut() {
            *x = next();
        }
        for g in 0..n_groups {
            let grp = &row[g * G..(g + 1) * G];
            let absmax = grp.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
            let scale = absmax / 127.0;
            sc[v * n_groups + g] = scale;
            if scale > 0.0 {
                for (j, &x) in grp.iter().enumerate() {
                    q[v * n_embd + g * G + j] = (x / scale).round().clamp(-127.0, 127.0) as i8;
                }
            }
        }
        for (k, &x) in row.iter().enumerate() {
            table_f16[v * n_embd + k] = half::f16::from_f32(x).to_bits();
        }
    }
    let h: Vec<f32> = (0..n_embd).map(|_| next()).collect();

    let d_f16 = stream.clone_htod(&table_f16).expect("f16 htod");
    let d_q = stream.clone_htod(&q).expect("i8 htod");
    let d_sc = stream.clone_htod(&sc).expect("scales htod");
    let d_h = stream.clone_htod(&h).expect("h htod");
    let mut d_logits: cudarc::driver::CudaSlice<f32> = stream.alloc_zeros(vocab).expect("logits");
    let (ne_i, v_i) = (n_embd as i32, vocab as i32);
    let cfg = LaunchConfig {
        grid_dim: ((vocab as u32).div_ceil(8), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut time_leg = |which: &str| -> f64 {
        const WARMUP: usize = 8;
        const RUNS: usize = 40;
        for _ in 0..WARMUP {
            #[allow(unsafe_code)]
            match which {
                "f16" => {
                    let mut l = stream.launch_builder(&f_f16);
                    l.arg(&d_h)
                        .arg(&d_f16)
                        .arg(&ne_i)
                        .arg(&v_i)
                        .arg(&mut d_logits);
                    // SAFETY: same launch contract as the gated call above.
                    unsafe { l.launch(cfg).expect("warmup f16") }
                }
                _ => {
                    let mut l = stream.launch_builder(&f_i8);
                    l.arg(&d_h)
                        .arg(&d_q)
                        .arg(&d_sc)
                        .arg(&ne_i)
                        .arg(&v_i)
                        .arg(&mut d_logits);
                    // SAFETY: same launch contract as the gated call above.
                    unsafe { l.launch(cfg).expect("warmup i8") }
                }
            };
        }
        stream.synchronize().expect("warmup sync");
        let t0 = Instant::now();
        for _ in 0..RUNS {
            #[allow(unsafe_code)]
            match which {
                "f16" => {
                    let mut l = stream.launch_builder(&f_f16);
                    l.arg(&d_h)
                        .arg(&d_f16)
                        .arg(&ne_i)
                        .arg(&v_i)
                        .arg(&mut d_logits);
                    // SAFETY: same launch contract as the gated call above.
                    unsafe { l.launch(cfg).expect("run f16") }
                }
                _ => {
                    let mut l = stream.launch_builder(&f_i8);
                    l.arg(&d_h)
                        .arg(&d_q)
                        .arg(&d_sc)
                        .arg(&ne_i)
                        .arg(&v_i)
                        .arg(&mut d_logits);
                    // SAFETY: same launch contract as the gated call above.
                    unsafe { l.launch(cfg).expect("run i8") }
                }
            };
        }
        stream.synchronize().expect("leg sync");
        t0.elapsed().as_secs_f64() / RUNS as f64 * 1e6
    };

    // Same-session ABBA: f16, i8, i8, f16.
    let a1 = time_leg("f16");
    let b1 = time_leg("i8");
    let b2 = time_leg("i8");
    let a2 = time_leg("f16");
    let f16_us = (a1 + a2) / 2.0;
    let i8_us = (b1 + b2) / 2.0;
    eprintln!(
        "lm_head M=1 ABBA (us/launch): f16 [{a1:.1}, {a2:.1}] avg {f16_us:.1} | i8 [{b1:.1}, {b2:.1}] avg {i8_us:.1} | speedup {:.2}x (gate >= 1.6x)",
        f16_us / i8_us
    );
}

/// `relu2_gate` must reproduce the host BitNet squared-ReLU FFN gate `r =
/// g.max(0); g = r*r*u` **bit-for-bit**. The input deliberately straddles zero so
/// the `max(.,0)` clamp (and the gate's hard zero on negatives) is exercised.
#[test]
fn relu2_gate_bit_matches_host() {
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping relu2_gate bit-match: no device ({e})");
            return;
        }
    };
    let mut s = 0x51A7_3C9E_2D6B_8F40u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        // Range [-4, 4): ~half the gate values negative, hitting the ReLU clamp.
        ((s >> 11) as f32 / (1u64 << 53) as f32) * 8.0 - 4.0
    };
    let n = 6912usize; // BitNet 2B4T n_ff
    let gate0: Vec<f32> = (0..n).map(|_| next()).collect();
    let up: Vec<f32> = (0..n).map(|_| next()).collect();
    // Host reference: identical to layers::mlp's gating loop.
    let want: Vec<f32> = gate0
        .iter()
        .zip(&up)
        .map(|(&g, &u)| {
            let r = g.max(0.0);
            r * r * u
        })
        .collect();
    let mut got = gate0.clone();
    backend.relu2_gate(&mut got, &up).expect("relu2_gate");
    for (i, (&g, &h)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            g.to_bits(),
            h.to_bits(),
            "relu2_gate mismatch [{i}]: got {g} want {h}"
        );
    }
}

/// Device GQA attention (M=1 decode) vs host `gqa_attention`. The dots + weighted
/// sums bit-match; the inline softmax `expf` gives a ≤3-ULP / ~1e-7 divergence, so
/// this measures the max rel error (reported) and asserts it stays tiny — the
/// attention output is the only forward op carrying the exp difference.
#[test]
fn gqa_attention_decode_matches_host() {
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping attention match: no device ({e})");
            return;
        }
    };
    // BitNet 2B4T attention dims; a modest cached context for the decode token.
    let (n_head, n_head_kv, head_dim, ctx) = (20usize, 5usize, 128usize, 96usize);
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let limit = ctx - 1; // steady-state decode: all cached keys visible
    let n_rep = n_head / n_head_kv;

    let mut s = 0x0BAD_F00D_1357_2468u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f32 / (1u64 << 53) as f32) * 4.0 - 2.0
    };
    let q: Vec<f32> = (0..n_head * head_dim).map(|_| next()).collect();
    let k: Vec<f32> = (0..ctx * n_head_kv * head_dim).map(|_| next()).collect();
    let v: Vec<f32> = (0..ctx * n_head_kv * head_dim).map(|_| next()).collect();

    // Host reference — replicates ops::gqa_attention for seq=1.
    let mut want = vec![0.0f32; n_head * head_dim];
    let mut scores = vec![0.0f32; ctx];
    for h in 0..n_head {
        let kv = h / n_rep;
        let q_row = &q[h * head_dim..h * head_dim + head_dim];
        for (j, sc) in scores.iter_mut().enumerate() {
            if j > limit {
                *sc = f32::NEG_INFINITY;
                continue;
            }
            let k_row = &k[(j * n_head_kv + kv) * head_dim..][..head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_row[d] * k_row[d];
            }
            *sc = dot * scale;
        }
        let mut m = f32::NEG_INFINITY;
        for &sc in &scores {
            if sc > m {
                m = sc;
            }
        }
        let mut sum = 0.0f32;
        for sc in scores.iter_mut() {
            let e = (*sc - m).exp();
            *sc = e;
            sum += e;
        }
        let inv = 1.0f32 / sum;
        for sc in scores.iter_mut() {
            *sc *= inv;
        }
        let o = &mut want[h * head_dim..h * head_dim + head_dim];
        for (j, &w) in scores.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            let v_row = &v[(j * n_head_kv + kv) * head_dim..][..head_dim];
            for d in 0..head_dim {
                o[d] += w * v_row[d];
            }
        }
    }

    let mut got = vec![0.0f32; n_head * head_dim];
    backend
        .gqa_attention_decode(
            &q, &k, &v, &mut got, ctx, n_head, n_head_kv, head_dim, scale, limit,
        )
        .expect("device attention");

    let (mut max_ulp, mut n_diff, mut max_rel, mut max_abs) = (0i64, 0usize, 0.0f64, 0.0f64);
    for (&g, &h) in got.iter().zip(&want) {
        let du = (i64::from(g.to_bits()) - i64::from(h.to_bits())).abs();
        if du != 0 {
            n_diff += 1;
        }
        max_ulp = max_ulp.max(du);
        max_abs = max_abs.max(f64::from((g - h).abs()));
        if h != 0.0 {
            max_rel = max_rel.max((f64::from(g - h) / f64::from(h)).abs());
        }
    }
    eprintln!(
        "attention device-vs-host: max_abs={max_abs:.3e} max_rel={max_rel:.3e} max_ulp={max_ulp} n_diff={n_diff}/{}",
        got.len()
    );
    // The dots + weighted sum bit-match; the sole divergence is the softmax `expf`
    // (≤3 ULP, ~1e-6 ABSOLUTE), which inflates to a larger *relative* error only on
    // near-zero (cancellation) outputs. The meaningful metric is the absolute error,
    // which must stay tiny (it propagates into the residual stream as a small add).
    assert!(
        max_abs < 1e-3,
        "device attention absolute error too large (likely a real bug): max_abs={max_abs:.3e}"
    );
}

/// `act_quant_tiled` must reproduce `ops::quantize_activation_int8` **bit-for-bit**
/// (the int8-as-f32 values and the per-token scale), including the zero-row case.
#[test]
fn act_quant_tiled_bit_matches_host() {
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping act_quant bit-match: no device ({e})");
            return;
        }
    };
    fn host_quant(act: &[f32]) -> (Vec<f32>, f32) {
        let mut gamma = 0.0f32;
        for &v in act {
            let a = v.abs();
            if a > gamma {
                gamma = a;
            }
        }
        if gamma == 0.0 {
            return (vec![0.0; act.len()], 0.0);
        }
        let s = 127.0f32 / gamma;
        (
            act.iter()
                .map(|&v| (v * s).round_ties_even().clamp(-128.0, 127.0))
                .collect(),
            gamma / 127.0,
        )
    }
    for &k in &[2560usize, 6912, 17, 1] {
        let mut s = 0x9999_AAAA_BBBB_CCCCu64 ^ k as u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f32 / (1u64 << 53) as f32) * 8.0 - 4.0
        };
        let act: Vec<f32> = (0..k).map(|_| next()).collect();
        let (q_want, scale_want) = host_quant(&act);
        let mut q_got = vec![f32::NAN; k];
        let scale_got = backend
            .act_quant_tiled(&act, &mut q_got)
            .expect("act_quant");
        assert_eq!(
            scale_got.to_bits(),
            scale_want.to_bits(),
            "scale mismatch k={k}"
        );
        for (i, (&g, &h)) in q_got.iter().zip(&q_want).enumerate() {
            assert_eq!(g.to_bits(), h.to_bits(), "act_quant q mismatch k={k} i={i}");
        }
    }
    // Zero row → zeros + zero scale.
    let act = vec![0.0f32; 64];
    let mut q = vec![1.0f32; 64];
    let sc = backend
        .act_quant_tiled(&act, &mut q)
        .expect("act_quant zero");
    assert_eq!(sc, 0.0);
    assert!(
        q.iter().all(|&x| x == 0.0),
        "zero row must quantize to zeros"
    );
}

/// The fused `rmsnorm_quant_f32` decode kernel must reproduce host RMSNorm followed
/// by the host int8 activation-quant, **bit-for-bit** — it composes the same two ops
/// `rmsnorm_bit_matches_host` and `act_quant_tiled_bit_matches_host` already pin.
/// This is the standalone regression guard for the shared-memory aliasing bug: the
/// absmax reduction once reused `s_x` as scratch, clobbering the RMSNorm output before
/// the quant step → garbage activations that only surfaced in the end-to-end forward.
#[test]
fn rmsnorm_quant_bit_matches_host() {
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping rmsnorm_quant bit-match: no device ({e})");
            return;
        }
    };
    // Host reference = ops::rmsnorm (replicated, as elsewhere) then the host int8
    // activation-quant (absmax → 127/gamma scale → round-ties-even → clamp).
    fn host_rmsnorm_quant(x: &[f32], w: &[f32], eps: f32) -> (Vec<f32>, f32) {
        let n = x.len();
        // Canonical tree sum-of-squares (ADR 0018), as in the rmsnorm test.
        let mut part = [0.0f32; 256];
        for (i, &v) in x.iter().enumerate() {
            part[i % 256] += v * v;
        }
        let mut off = 128;
        while off > 0 {
            for t in 0..off {
                part[t] += part[t + off];
            }
            off >>= 1;
        }
        let mean_sq = part[0] / n as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        let y: Vec<f32> = x.iter().zip(w).map(|(&xi, &wi)| xi * inv * wi).collect();
        let mut gamma = 0.0f32;
        for &v in &y {
            let a = v.abs();
            if a > gamma {
                gamma = a;
            }
        }
        if gamma == 0.0 {
            return (vec![0.0; n], 0.0);
        }
        let s = 127.0f32 / gamma;
        (
            y.iter()
                .map(|&v| (v * s).round_ties_even().clamp(-128.0, 127.0))
                .collect(),
            gamma / 127.0,
        )
    }
    // BitNet hidden/ffn widths + edge lengths; deterministic xorshift inputs.
    for &n in &[2560usize, 6912, 1, 17, 256, 2559] {
        let mut s = 0x0FED_CBA9_8765_4321u64 ^ (n as u64).wrapping_mul(0x9E37_79B9);
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f32 / (1u64 << 53) as f32) * 8.0 - 4.0
        };
        let x: Vec<f32> = (0..n).map(|_| next()).collect();
        let w: Vec<f32> = (0..n).map(|_| next()).collect();
        let eps = 1e-5f32;

        let (q_want, scale_want) = host_rmsnorm_quant(&x, &w, eps);
        let mut q_got = vec![f32::NAN; n];
        let scale_got = backend
            .rmsnorm_quant(&x, &w, eps, &mut q_got)
            .expect("device rmsnorm_quant");

        assert_eq!(
            scale_got.to_bits(),
            scale_want.to_bits(),
            "rmsnorm_quant scale mismatch n={n}: got {scale_got} want {scale_want}"
        );
        for (i, (&g, &h)) in q_got.iter().zip(&q_want).enumerate() {
            assert_eq!(
                g.to_bits(),
                h.to_bits(),
                "rmsnorm_quant q mismatch n={n} i={i}: got {g} want {h}"
            );
        }
    }
    // All-zero input → zeros + zero scale (the gamma==0 branch).
    let x = vec![0.0f32; 128];
    let w = vec![1.0f32; 128];
    let mut q = vec![1.0f32; 128];
    let sc = backend
        .rmsnorm_quant(&x, &w, 1e-5, &mut q)
        .expect("rmsnorm_quant zero");
    assert_eq!(sc, 0.0, "all-zero input must give zero scale");
    assert!(
        q.iter().all(|&v| v == 0.0),
        "all-zero input must quantize to zeros"
    );
}

/// ADR 0036 L5 gate: the fused `rmsnorm_quant_batch_i8` must be BIT-identical
/// to the split `rmsnorm_batch_f32` → `act_quant_batch_i8` pair it replaces at
/// every norm→quant seam of the batch/tree trunk — i8 activations byte-equal,
/// per-row scales equal by `to_bits`. Shapes cover the 2B4T trunk widths
/// (n_embd 2560, n_ff 6912-ish via 1536) at verify-scale m, plus m=1 and an
/// all-zero row (the gamma==0 branch).
#[test]
fn rmsnorm_quant_batch_matches_split_pair_bitwise() {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::Ptx;

    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping rmsnorm_quant_batch gate: no device ({e})");
            return;
        }
    };
    let stream = ctx.default_stream();
    let dm = ctx
        .load_module(Ptx::from_src(DECODE_PTX))
        .expect("load decode module");
    let f_rmsnorm = dm.load_function("rmsnorm_batch_f32").expect("rmsnorm fn");
    let f_quant = dm.load_function("act_quant_batch_i8").expect("quant fn");
    let f_fused = dm
        .load_function("rmsnorm_quant_batch_i8")
        .expect("fused fn");

    let mut s = 0x0036_51AB_C0DE_F00Du64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f32 / (1u64 << 53) as f32) * 4.0 - 2.0
    };
    let eps = 1e-5f32;
    for (n, m) in [(2560usize, 48usize), (1536, 13), (2560, 1), (256, 3)] {
        let mut x: Vec<f32> = (0..m * n).map(|_| next()).collect();
        // Row 0 all-zero when m > 1: exercises the gamma==0 branch in batch.
        if m > 1 {
            x[..n].fill(0.0);
        }
        let w: Vec<f32> = (0..n).map(|_| next()).collect();
        let d_x = stream.clone_htod(&x).expect("x htod");
        let d_w = stream.clone_htod(&w).expect("w htod");
        let mut d_normed = stream.alloc_zeros::<f32>(m * n).expect("normed");
        let mut d_q_split = stream.alloc_zeros::<i8>(m * n).expect("q split");
        let mut d_sc_split = stream.alloc_zeros::<f32>(m).expect("sc split");
        let mut d_q_fused = stream.alloc_zeros::<i8>(m * n).expect("q fused");
        let mut d_sc_fused = stream.alloc_zeros::<f32>(m).expect("sc fused");
        let (n_i, m_i) = (n as i32, m as i32);

        // Split pair: rmsnorm_batch_f32 → act_quant_batch_i8.
        let norm_cfg = LaunchConfig {
            grid_dim: (m as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (n * 4) as u32,
        };
        let mut l = stream.launch_builder(&f_rmsnorm);
        l.arg(&d_x)
            .arg(&d_w)
            .arg(&eps)
            .arg(&n_i)
            .arg(&m_i)
            .arg(&mut d_normed);
        // SAFETY: `rmsnorm_batch_f32(x, w, eps, n, m, out)`.
        #[allow(unsafe_code)]
        unsafe { l.launch(norm_cfg) }.expect("launch rmsnorm_batch");
        let quant_cfg = LaunchConfig {
            grid_dim: (m as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = stream.launch_builder(&f_quant);
        l.arg(&d_normed)
            .arg(&n_i)
            .arg(&m_i)
            .arg(&mut d_q_split)
            .arg(&mut d_sc_split);
        // SAFETY: `act_quant_batch_i8(act, k, m, q_out, act_scale)`.
        #[allow(unsafe_code)]
        unsafe { l.launch(quant_cfg) }.expect("launch act_quant_batch");

        // Fused twin.
        let mut l = stream.launch_builder(&f_fused);
        l.arg(&d_x)
            .arg(&d_w)
            .arg(&eps)
            .arg(&n_i)
            .arg(&m_i)
            .arg(&mut d_q_fused)
            .arg(&mut d_sc_fused);
        // SAFETY: `rmsnorm_quant_batch_i8(x, w, eps, n, m, q_out, act_scale)`.
        #[allow(unsafe_code)]
        unsafe { l.launch(norm_cfg) }.expect("launch rmsnorm_quant_batch");

        let mut q_split = vec![0i8; m * n];
        let mut q_fused = vec![0i8; m * n];
        let mut sc_split = vec![0.0f32; m];
        let mut sc_fused = vec![0.0f32; m];
        stream.memcpy_dtoh(&d_q_split, &mut q_split).expect("dtoh");
        stream.memcpy_dtoh(&d_q_fused, &mut q_fused).expect("dtoh");
        stream
            .memcpy_dtoh(&d_sc_split, &mut sc_split)
            .expect("dtoh");
        stream
            .memcpy_dtoh(&d_sc_fused, &mut sc_fused)
            .expect("dtoh");
        stream.synchronize().expect("sync");

        for r in 0..m {
            assert_eq!(
                sc_split[r].to_bits(),
                sc_fused[r].to_bits(),
                "act_scale drift n={n} m={m} row {r}: split {} fused {}",
                sc_split[r],
                sc_fused[r]
            );
        }
        assert_eq!(
            q_split, q_fused,
            "fused rmsnorm_quant_batch_i8 must be bit-identical to the split \
             pair (n={n} m={m})"
        );
        if m > 1 {
            assert_eq!(sc_split[0], 0.0, "zero row must give zero scale");
            assert!(
                q_fused[..n].iter().all(|&v| v == 0),
                "zero row must quantize to zeros"
            );
        }
    }
}

/// The device GEMM chain (`mpgemm_device`: on-device quant → tiled f64 GEMM →
/// scale fold) must reproduce the host path (`quantize_activation_int8` → tiled
/// `mpgemm` → `out *= act_scale`) **bit-for-bit** — same quant, same kernel, same
/// fold, just resident. This is the GEMM half of the device-resident decode.
#[test]
fn mpgemm_device_bit_matches_host_path() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping mpgemm_device match: no device ({e})");
            return;
        }
    };
    let (n, k) = (640usize, 2560usize); // BitNet attn_k projection shape
    let shape = GemmShape::new(1, n, k);

    let mut st = 0x1357_9BDF_2468_ACE0u64;
    let trits: Vec<tritium_core::Trit> = (0..n * k)
        .map(|_| {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            tritium_core::Trit::from_i8(((st >> 33) % 3) as i8 - 1).unwrap()
        })
        .collect();
    let packed = pack_tq2_0(&trits, shape);
    let weights = cuda
        .upload_weights(&packed, shape, TernaryFormat::Tq2_0)
        .expect("upload");

    let mut sf = 0x2468_ACE0_1357_9BDFu64;
    let mut nf = || {
        sf ^= sf << 13;
        sf ^= sf >> 7;
        sf ^= sf << 17;
        ((sf >> 11) as f32 / (1u64 << 53) as f32) * 4.0 - 2.0
    };
    let normed: Vec<f32> = (0..k).map(|_| nf()).collect();
    let scales: Vec<f32> = (0..n).map(|_| 0.5 + nf().abs()).collect();

    // Host path: quantize_activation_int8 + tiled mpgemm + per-token fold.
    let (q_host, act_scale) = {
        let mut gamma = 0.0f32;
        for &v in &normed {
            let a = v.abs();
            if a > gamma {
                gamma = a;
            }
        }
        if gamma == 0.0 {
            (vec![0.0f32; k], 0.0f32)
        } else {
            let s = 127.0f32 / gamma;
            (
                normed
                    .iter()
                    .map(|&v| (v * s).round_ties_even().clamp(-128.0, 127.0))
                    .collect::<Vec<_>>(),
                gamma / 127.0,
            )
        }
    };
    let mut out_host = run_kernel(&cuda, &packed, &q_host, &scales, shape, AddKernel::Tiled);
    for v in out_host.iter_mut() {
        *v *= act_scale;
    }

    // Device chain.
    let mut out_dev = vec![0.0f32; n];
    cuda.mpgemm_device(&normed, weights.as_ref(), &scales, shape, &mut out_dev)
        .expect("mpgemm_device");

    for (i, (&g, &h)) in out_dev.iter().zip(&out_host).enumerate() {
        assert_eq!(
            g.to_bits(),
            h.to_bits(),
            "mpgemm_device mismatch [{i}]: got {g} want {h}"
        );
    }
}

/// CUDA-graph capture spike (v0.3.1 W2) — documents a hard cudarc-0.19 limitation.
///
/// Capturing the decode forward into a replayable graph would collapse the ~390
/// per-token kernel launches into one `graph.launch()`, the biggest remaining decode
/// win (the launch path is the wall at M=1). But cudarc 0.19's **safe** launch
/// (`LaunchArgs::launch`) waits on each buffer's read/write `CudaEvent` before the
/// kernel — and those events were recorded by the pre-capture uploads, so the very
/// first captured launch trips `CUDA_ERROR_STREAM_CAPTURE_ISOLATION` ("dependency
/// created on uncaptured work"). RELAXED capture mode does not help (the dependency is
/// real, not a mode artifact). The raw escape — `result::launch_kernel`, which does no
/// event tracking — needs the `sys::CUfunction` handle, but cudarc keeps
/// `CudaFunction::cu_function` `pub(crate)`, so the only way through is a *parallel*
/// raw-FFI module/function/launch path (load the PTX via `result::module::load_data`,
/// `get_function`, hand-pack params), bypassing cudarc's safe layer entirely.
///
/// That raw path is the deferred W2 work (it materially expands the `unsafe` surface
/// of this `#![deny(unsafe_code)]` crate, so it is its own gated change). This test is
/// `#[ignore]`d: it asserts the limitation still holds, so if a future cudarc makes the
/// safe launch capture-compatible, this starts passing and flags that the raw path is
/// no longer needed.
#[test]
#[ignore = "cudarc 0.19 safe launch is capture-incompatible; W2 needs the raw-FFI path"]
fn cuda_graph_capture_blocked_by_cudarc_safe_launch() {
    use cudarc::driver::sys;
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping cuda graph spike: no device ({e})");
            return;
        }
    };
    let n = 256usize;
    let x0 = vec![1.0f32; n];
    let y = vec![2.0f32; n];
    let cap = backend
        .stream
        .context()
        .new_stream()
        .expect("capture stream");
    let mut d_x = cap.clone_htod(&x0).expect("htod x");
    let d_y = cap.clone_htod(&y).expect("htod y");
    cap.synchronize().expect("sync");

    let n_i = n as i32;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    cap.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        .expect("begin_capture");
    let mut l = cap.launch_builder(&backend.func_residual);
    l.arg(&mut d_x).arg(&d_y).arg(&n_i);
    // SAFETY: `residual_add_f32(float* x, const float* y, int n)`.
    #[allow(unsafe_code)]
    let launched = unsafe { l.launch(cfg) };
    // The capture launch trips STREAM_CAPTURE_ISOLATION on cudarc 0.19. If this ever
    // succeeds, the safe launch became capture-compatible — revisit the raw-FFI plan.
    assert!(
        launched.is_err(),
        "cudarc safe launch unexpectedly captured cleanly — the raw-FFI W2 path may be unnecessary now"
    );
    let _ = cap.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
}

/// CUDA-graph **raw-FFI** capture spike (v0.3.2) — the path that works where the
/// safe launch trips isolation. Pre-extract each buffer's stable `CUdeviceptr`
/// *before* `begin_capture` (dropping the `SyncOnDrop` guard outside capture), raw-
/// load the decode PTX for a raw `CUfunction`, then capture two `residual_add_f32`
/// launches via `result::launch_kernel` (no cudarc event waits → no isolation), and
/// assert the single graph replay is **bit-identical** to the host reference. This
/// pins the v0.3.2 mechanic before the full decode forward is captured.
#[test]
fn cuda_graph_raw_launch_replay_bit_identical() {
    use cudarc::driver::{DevicePtr, DevicePtrMut, result, sys};
    use std::ffi::{CString, c_void};

    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping raw-graph spike: no device ({e})");
            return;
        }
    };
    let ctx = backend.stream.context().clone();
    ctx.bind_to_thread().expect("bind ctx");

    // Raw-load the decode PTX → a raw CUfunction (the safe CudaFunction hides
    // `cu_function`, so the captured launch needs this raw handle).
    let ptx_c = CString::new(DECODE_PTX).expect("ptx cstring");
    // SAFETY: `ptx_c` is a valid NUL-terminated PTX image; `load_data` JIT-compiles it.
    #[allow(unsafe_code)]
    let cu_module =
        unsafe { result::module::load_data(ptx_c.as_ptr() as *const c_void).expect("load_data") };
    let fname = CString::new("residual_add_f32").expect("fn cstring");
    // SAFETY: `cu_module` is a loaded module; `residual_add_f32` is one of its entry points.
    #[allow(unsafe_code)]
    let cu_func = unsafe { result::module::get_function(cu_module, fname).expect("get_function") };

    let n = 2560usize;
    let x0 = vec![1.0f32; n];
    let y = vec![2.0f32; n];
    // residual_add applied twice: ((x0 + y) + y), the kernel's single-f32-add order.
    let want: Vec<f32> = x0.iter().zip(&y).map(|(&a, &b)| (a + b) + b).collect();

    let cap = ctx.new_stream().expect("capture stream");
    let mut d_x = cap.clone_htod(&x0).expect("htod x");
    let d_y = cap.clone_htod(&y).expect("htod y");
    cap.synchronize().expect("pre-extract sync");

    // Pre-extract stable device pointers; drop the SyncOnDrop guards OUTSIDE capture
    // (their drop records an event, which is forbidden inside a capture).
    let px: sys::CUdeviceptr = {
        let (p, g) = d_x.device_ptr_mut(&cap);
        drop(g);
        p
    };
    let py: sys::CUdeviceptr = {
        let (p, g) = d_y.device_ptr(&cap);
        drop(g);
        p
    };
    cap.synchronize().expect("post-extract sync");

    let n_i = n as i32;
    let grid = ((n as u32).div_ceil(256), 1u32, 1u32);
    let block = (256u32, 1u32, 1u32);

    cap.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .expect("begin_capture");
    for _ in 0..2 {
        // kernel_params: each entry points to the arg VALUE (a CUdeviceptr for a
        // `float*`, the i32 for `int n`); these locals outlive the launch call, and
        // graph capture snapshots the values into the kernel node.
        let mut params: [*mut c_void; 3] = [
            (&px) as *const sys::CUdeviceptr as *mut c_void,
            (&py) as *const sys::CUdeviceptr as *mut c_void,
            (&n_i) as *const i32 as *mut c_void,
        ];
        // SAFETY: raw `residual_add_f32(float* x, const float* y, int n)`; params in
        // declaration order; `px`/`py` are valid device addresses (extracted above,
        // `d_x`/`d_y` alive for the test), `n_i` matches the buffer length.
        #[allow(unsafe_code)]
        unsafe {
            result::launch_kernel(cu_func, grid, block, 0, cap.cu_stream(), &mut params)
                .expect("raw capture launch");
        }
    }
    let graph = cap
        .end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .expect("end_capture")
        .expect("non-empty graph");

    // d_x is still x0 (capture did not execute). One replay runs both adds.
    graph.launch().expect("graph launch");
    cap.synchronize().expect("post-replay sync");
    let mut got = vec![0.0f32; n];
    cap.memcpy_dtoh(&d_x, &mut got).expect("dtoh");
    for (i, (&g, &h)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            g.to_bits(),
            h.to_bits(),
            "raw graph replay mismatch [{i}]: got {g} want {h}"
        );
    }

    // The captured graph holds the raw CUfunction; unload only after a final sync.
    cap.synchronize().expect("final sync");
    drop(graph);
    // SAFETY: `cu_module` was loaded above and is unloaded exactly once here, after the
    // graph (which referenced its function) is dropped and the stream is synchronized.
    #[allow(unsafe_code)]
    unsafe {
        result::module::unload(cu_module).expect("unload");
    }
}

// ── Sparse kernel tests (P1: zero-block sparsity skip) ─────────────────
use tritium_format::QK_K;

/// Build a trit vector with a known sparsity pattern: `zero_blocks` out of
/// `total_blocks` are all-zero (placed at the start), the rest are all +1.
/// Build an `[n, k]` row-major trit matrix (POS everywhere) with the first
/// `zero_blocks` TQ2_0 blocks of EACH row zeroed (partial last block respected
/// via `.min(k)`). Length is `n * k`, matching what `pack_tq2_0(.., shape)`
/// slices per row — so the per-row zero pattern is what `compute_zero_bitmaps`
/// will flag and the sparse kernel must skip.
fn make_sparse_trits(n: usize, k: usize, zero_blocks: usize) -> Vec<tritium_core::Trit> {
    let mut trits = vec![tritium_core::Trit::POS; n * k];
    for row in 0..n {
        let base = row * k;
        for b in 0..zero_blocks {
            let start = b * QK_K;
            let end = ((b + 1) * QK_K).min(k);
            for i in start..end {
                trits[base + i] = tritium_core::Trit::ZERO;
            }
        }
    }
    trits
}

/// The sparse-aware tiled kernel must match the CPU reference on mixed
/// zero/nonzero weights. This is the primary correctness gate for P1.
#[test]
fn sparse_kernel_matches_cpu_reference() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping sparse kernel test: no device ({e})");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let tol = Tolerance::default();

    // K=4096 (16 blocks), ~40% zero blocks (7 out of 16)
    let nb = 16;
    let k = nb * QK_K; // 4096
    let n = 8;
    let m = 1;
    let shape = GemmShape::new(m, n, k);

    let trits = make_sparse_trits(n, k, 7);
    let packed = pack_tq2_0(&trits, shape);
    let act = seeded_f32(42, m * k, -1.0, 1.0);
    let scales = seeded_f32(99, n, 0.5, 2.0);

    let cpu_out = run_backend(&cpu, &packed, &act, &scales, shape);

    let bitmap =
        tritium_format::compute_zero_bitmaps(&packed, n, k, nb * TQ2_0_BLOCK_BYTES).unwrap();
    let words_per_row = nb.div_ceil(32);
    let sparse_out =
        run_kernel_sparse(&cuda, &packed, &act, &scales, &bitmap, words_per_row, shape);

    for (i, (&g, &c)) in sparse_out.iter().zip(&cpu_out).enumerate() {
        assert!(tol.accepts(g, c), "sparse vs cpu [{i}]: sparse={g} cpu={c}");
    }
}

/// The sparse kernel must produce identical output to the dense tiled kernel
/// on the same weights (the bitmap just skips zero contributions, which the
/// dense kernel also skips via branchless `a * (code - 1)` where code=1).
#[test]
fn sparse_matches_dense_tiled_on_mixed_weights() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping sparse-vs-dense test: no device ({e})");
            return;
        }
    };
    let tol = Tolerance::default();

    let nb = 16;
    let k = nb * QK_K;
    let n = 8;
    let m = 1;
    let shape = GemmShape::new(m, n, k);

    let trits = make_sparse_trits(n, k, 7);
    let packed = pack_tq2_0(&trits, shape);
    let act = seeded_f32(42, m * k, -1.0, 1.0);
    let scales = seeded_f32(99, n, 0.5, 2.0);

    // Dense tiled (double-accumulator, the reference-gated kernel)
    let dense = run_kernel(&cuda, &packed, &act, &scales, shape, AddKernel::Tiled);

    // Sparse-aware tiled
    let bitmap =
        tritium_format::compute_zero_bitmaps(&packed, n, k, nb * TQ2_0_BLOCK_BYTES).unwrap();
    let words_per_row = nb.div_ceil(32);
    let sparse = run_kernel_sparse(&cuda, &packed, &act, &scales, &bitmap, words_per_row, shape);

    for (i, (&d, &s)) in dense.iter().zip(&sparse).enumerate() {
        assert!(
            tol.accepts(s, d),
            "sparse vs dense [{i}]: sparse={s} dense={d}"
        );
    }
}

/// All-zero weights: the sparse kernel must produce exactly zero output.
#[test]
fn sparse_kernel_all_zero_weights() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping sparse all-zero test: no device ({e})");
            return;
        }
    };

    let nb = 4;
    let k = nb * QK_K;
    let n = 4;
    let m = 1;
    let shape = GemmShape::new(m, n, k);

    // All-zero trits → every block is zero
    let trits = vec![tritium_core::Trit::ZERO; n * k];
    let packed = pack_tq2_0(&trits, shape);
    let act = seeded_f32(42, m * k, -1.0, 1.0);
    let scales = seeded_f32(99, n, 0.5, 2.0);

    let bitmap =
        tritium_format::compute_zero_bitmaps(&packed, n, k, nb * TQ2_0_BLOCK_BYTES).unwrap();
    let words_per_row = nb.div_ceil(32);
    let sparse = run_kernel_sparse(&cuda, &packed, &act, &scales, &bitmap, words_per_row, shape);

    for (i, &v) in sparse.iter().enumerate() {
        assert_eq!(v, 0.0, "all-zero weights should produce zero output [{i}]");
    }
}

/// No zero blocks: the sparse kernel must match the dense kernel exactly.
#[test]
fn sparse_kernel_no_zero_blocks() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping sparse no-zero test: no device ({e})");
            return;
        }
    };
    let tol = Tolerance::default();

    let nb = 8;
    let k = nb * QK_K;
    let n = 4;
    let m = 1;
    let shape = GemmShape::new(m, n, k);

    // All-positive trits → no zero blocks
    let trits = vec![tritium_core::Trit::POS; n * k];
    let packed = pack_tq2_0(&trits, shape);
    let act = seeded_f32(42, m * k, -1.0, 1.0);
    let scales = seeded_f32(99, n, 0.5, 2.0);

    let dense = run_kernel(&cuda, &packed, &act, &scales, shape, AddKernel::Tiled);

    let bitmap =
        tritium_format::compute_zero_bitmaps(&packed, n, k, nb * TQ2_0_BLOCK_BYTES).unwrap();
    let words_per_row = nb.div_ceil(32);
    let sparse = run_kernel_sparse(&cuda, &packed, &act, &scales, &bitmap, words_per_row, shape);

    for (i, (&d, &s)) in dense.iter().zip(&sparse).enumerate() {
        assert!(
            tol.accepts(s, d),
            "no-zero sparse vs dense [{i}]: sparse={s} dense={d}"
        );
    }
}

/// Boundary shape: K not a multiple of QK_K (partial last block).
#[test]
fn sparse_kernel_partial_block() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping sparse partial test: no device ({e})");
            return;
        }
    };
    let tol = Tolerance::default();

    // K=300 → 2 blocks (one full 256, one partial 44)
    let k = 300usize;
    let nb = k.div_ceil(QK_K); // 2
    let n = 4;
    let m = 1;
    let shape = GemmShape::new(m, n, k);

    let trits = make_sparse_trits(n, k, 1); // block 0 zero, block 1 nonzero
    let packed = pack_tq2_0(&trits, shape);
    let act = seeded_f32(42, m * k, -1.0, 1.0);
    let scales = seeded_f32(99, n, 0.5, 2.0);

    let cpu = CpuBackend::new();
    let cpu_out = run_backend(&cpu, &packed, &act, &scales, shape);

    let bitmap =
        tritium_format::compute_zero_bitmaps(&packed, n, k, nb * TQ2_0_BLOCK_BYTES).unwrap();
    let words_per_row = nb.div_ceil(32);
    let sparse = run_kernel_sparse(&cuda, &packed, &act, &scales, &bitmap, words_per_row, shape);

    for (i, (&g, &c)) in sparse.iter().zip(&cpu_out).enumerate() {
        assert!(
            tol.accepts(g, c),
            "partial block sparse vs cpu [{i}]: sparse={g} cpu={c}"
        );
    }
}

/// Review-98ab046 nit N2 guardrail: the host-side v2 attention dispatch
/// bounds (consts.rs) must equal the kernel's shared-sizing `#define`s in
/// decode.cu — a Rust-side value drifting LARGER than the kernel's would send
/// out-of-bounds shared indices to production that the (ctx <= 3584) bitwise
/// gates cannot catch. Parsed from source, no GPU needed.
#[test]
fn attn_v2_consts_match_decode_cu_defines() {
    let src = include_str!("../../kernels/decode.cu");
    let get = |name: &str| -> usize {
        src.lines()
            .find_map(|l| l.strip_prefix(&format!("#define {name} ")))
            .unwrap_or_else(|| panic!("#define {name} missing from decode.cu"))
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("#define {name} is not a bare number"))
    };
    assert_eq!(get("ATTN_V2_HDMAX"), ATTN_V2_HDMAX);
    assert_eq!(get("ATTN_V2_MAX_CTX"), ATTN_V2_MAX_CTX);
    assert_eq!(get("ATTN_V2_THREADS"), ATTN_V2_THREADS as usize);
    assert_eq!(get("ATTN_V3_BQ"), ATTN_V3_BQ);
    assert_eq!(get("ATTN_V3_THREADS"), ATTN_V3_THREADS as usize);
}

/// ADR 0022 guardrail with teeth: the twin-kernel family table must match
/// decode.cu. Drift — a new variant (the revisit trigger: a 4th KV rung or a
/// new attention family) or a removed one — fails here mechanically instead
/// of relying on reviewer memory. No GPU needed: this parses the source.
#[test]
fn adr_0022_twin_family_table_matches_decode_cu() {
    let src = include_str!("../../kernels/decode.cu");
    let names: Vec<&str> = src
        .lines()
        .filter_map(|l| l.strip_prefix("__global__ void "))
        .map(|l| {
            // Skip an optional `__launch_bounds__(N)` qualifier before the name
            // (the v2 attention twins carry one).
            let l = l.trim_start();
            let l = l.strip_prefix("__launch_bounds__").map_or(l, |rest| {
                rest.split_once(')')
                    .map_or(rest, |(_, after)| after)
                    .trim_start()
            });
            l.split('(').next().unwrap_or(l).trim()
        })
        .collect();
    let count = |prefix: &str| {
        names
            .iter()
            .filter(|n| {
                // exact family match: prefix, then either end or a variant
                // suffix — avoids kv_append counting kv_append_batch.
                n.strip_prefix(prefix).is_some_and(|rest| {
                    rest.is_empty()
                        || matches!(rest, "_g" | "_h" | "_q8" | "_t2" | "_f32" | "_f16" | "_i8")
                })
            })
            .count()
    };
    // The ADR 0022 family table (docs/adr/0022-twin-kernel-contract.md).
    let table = [
        ("rope_kv_fused", 4),
        ("kv_append", 4),
        ("kv_append_batch", 4),
        ("gqa_attention_scores", 3),
        ("gqa_attention_reduce", 3),
        ("gqa_attention_batch", 3),
        ("gqa_attention_batch_v2", 2),
        ("gqa_attention_batch_v3", 2),
        ("gqa_attention_tree_scores", 3),
        ("gqa_attention_tree_reduce", 3),
        ("lm_head_warp", 3),
    ];
    for (family, want) in table {
        assert_eq!(
            count(family),
            want,
            "twin family `{family}` drifted from the ADR 0022 table — update \
             the ADR (and check the revisit trigger) alongside the kernel"
        );
    }
    assert_eq!(
        names.len(),
        86,
        "decode.cu kernel count drifted from ADR 0022 — update the ADR \
         (65 → 64: gqa_attention_mdecode_f32 retired; 64 → 66: paged KV \
         twins added, ADR 0025 step 2; rmsnorm_quant_i8_fast was added and \
         DELETED by measurement — ADR 0023 rejected, +1.75% < the 3% bar; \
         66 → 68: gqa_attention_batch_v2 f32/h twins added — order-preserving \
         prefill attention, bit-identical to rev 1 by to_bits gate; 68 → 70: \
         gqa_attention_batch_v3 Q-blocked twins, same bit-identity gate; \
         70 → 71: draft_chain_advance — the L1' chained-draft glue, ADR 0032; \
         71 → 74: paged tree-verify ctrl twins — kv_append_tree_paged_g + \
         gqa_attention_tree_{{scores,reduce}}_ctrl_paged_g, L3-I3 tree verify \
         against paged BatchKv slots, bit-identical to a dense slot by the \
         cuda_tree_verify_paged_slot_matches_dense gate; 74 → 80: L3-I4 \
         batched-slots twins — kv_append_tree_slots[_paged]_g + \
         gqa_attention_tree_{{scores,reduce}}_slots[_paged]_g, per-ROW ctrl \
         so ONE forward verifies many slots' trees; the single-slot ctrl \
         kernels are retained unchanged, batched == sequential by the \
         cuda_tree_verify_slots_matches_sequential gate; \
         draft_batch_chain_advance was added and REVERTED by measurement — \
         Track B 2026-08-08, bit-identical + fully gated but -2.4%..+1.6% \
         at N=4 < the 3% bar: post-bucket-snap k~2-3 leaves only 1-2 \
         per-step round-trips to amortize; 80 → 83: f16-KV ctrl tree-verify \
         twins — kv_append_tree_h + gqa_attention_tree_{{scores,reduce}}_ctrl_h, \
         ADR 0036 L6: the SINGLE-SEQ tree graph route on the accepted f16 \
         rung (batch arenas stay f32); graph == eager-f16 by the \
         cuda_tree_verify_f16_graph gate in tritium-nn acceptance; 83 → 85: \
         lm_head_warp_i8 + lm_head_tiled_i8 — ADR 0036 L2 opt-in int8 head \
         rung (TRITIUM_LM_HEAD=i8), 64-group absmax table + f32 scales, \
         ppl/τ-gated NOT bit-identical; pinned to a host oracle + \
         warp==tiled by lm_head_i8_warp_and_tiled_match_host_oracle_bitwise; \
         85 → 86: rmsnorm_quant_batch_i8 — ADR 0036 L5 fused batch \
         rmsnorm+quant, bit-identical to the rmsnorm_batch_f32 → \
         act_quant_batch_i8 pair by the \
         rmsnorm_quant_batch_matches_split_pair_bitwise gate; the split \
         kernels stay for the final output norm and quant-only sites)"
    );
}

/// A2 gate: the TQ1_0-native i8-scaled kernel is BIT-identical to the TQ2_0
/// one on the same trits (integer accumulation; identical epilogue) — for the
/// plain and residual twins, across aligned and tail (k % 256 != 0) shapes.
#[test]
fn tq1_matches_tq2_tiled_scaled_bit_exact() {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::Ptx;
    use tritium_format::{TQ1_0_BLOCK_BYTES, num_blocks, pack_tq1_0_row};

    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping tq1-vs-tq2 gate: no device ({e})");
            return;
        }
    };
    let stream = ctx.default_stream();
    let module = ctx
        .load_module(Ptx::from_src(TQ2_0_ADD_PTX))
        .expect("load add module");
    let f_tq2 = module
        .load_function("tq2_0_add_mpgemm_tiled_i8_scaled")
        .expect("tq2 fn");
    let f_tq1 = module
        .load_function("tq1_0_add_mpgemm_tiled_i8_scaled")
        .expect("tq1 fn");
    let f_tq2_res = module
        .load_function("tq2_0_add_mpgemm_tiled_i8_scaled_residual")
        .expect("tq2 res fn");
    let f_tq1_res = module
        .load_function("tq1_0_add_mpgemm_tiled_i8_scaled_residual")
        .expect("tq1 res fn");

    // Aligned + tail shapes; m > 1 exercises the act_scale fold per row.
    for &(m, n, k) in &[(1usize, 8usize, 1024usize), (2, 5, 256 + 128)] {
        let trits = mixed_trits(n, k, 0xA2 ^ (k as u64));
        let nb = num_blocks(k);
        let unit = vec![half::f16::ONE; nb];
        let (rb2, rb1) = (nb * TQ2_0_BLOCK_BYTES, nb * TQ1_0_BLOCK_BYTES);
        let mut p2 = vec![0u8; n * rb2];
        let mut p1 = vec![0u8; n * rb1];
        for ni in 0..n {
            let row = &trits[ni * k..(ni + 1) * k];
            tritium_format::pack_tq2_0_row(row, &unit, &mut p2[ni * rb2..(ni + 1) * rb2])
                .expect("pack tq2");
            pack_tq1_0_row(row, &unit, &mut p1[ni * rb1..(ni + 1) * rb1]).expect("pack tq1");
        }
        // Deterministic i8 activations (k % 4 == 0 holds for both shapes).
        let qact: Vec<i8> = (0..m * k).map(|i| ((i * 37 + 11) % 255) as i8).collect();
        let scales = seeded_f32(7, n, 0.5, 2.0);
        let act_scale = seeded_f32(13, m, 0.5, 1.5);
        let residual = seeded_f32(21, m * n, -1.0, 1.0);

        let d_qact = stream.clone_htod(&qact).unwrap();
        let d_w2 = stream.clone_htod(&p2).unwrap();
        let d_w1 = stream.clone_htod(&p1).unwrap();
        let d_sc = stream.clone_htod(&scales).unwrap();
        let d_as = stream.clone_htod(&act_scale).unwrap();
        let d_res = stream.clone_htod(&residual).unwrap();
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);
        let (rb2_i, rb1_i) = (rb2 as i32, rb1 as i32);
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(8), m as u32, 1),
            block_dim: (8 * 32, 1, 1),
            shared_mem_bytes: 0,
        };
        let launch_plain = |f: &cudarc::driver::CudaFunction,
                            w: &cudarc::driver::CudaSlice<u8>,
                            rb: &i32|
         -> Vec<f32> {
            let mut d_out = stream.alloc_zeros::<f32>(m * n).unwrap();
            let mut l = stream.launch_builder(f);
            l.arg(&d_qact)
                .arg(w)
                .arg(&d_sc)
                .arg(&d_as)
                .arg(&mut d_out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .arg(rb);
            // SAFETY: matches the kernel signatures asserted in the kernel
            // source; grid.y = m.
            #[allow(unsafe_code)]
            unsafe {
                l.launch(cfg).unwrap()
            };
            let mut out = vec![0.0f32; m * n];
            stream.memcpy_dtoh(&d_out, &mut out).unwrap();
            out
        };
        let o2 = launch_plain(&f_tq2, &d_w2, &rb2_i);
        let o1 = launch_plain(&f_tq1, &d_w1, &rb1_i);
        for (i, (a, b)) in o2.iter().zip(&o1).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "plain m{m} n{n} k{k} [{i}]: tq2={a} tq1={b}"
            );
        }

        let launch_res = |f: &cudarc::driver::CudaFunction,
                          w: &cudarc::driver::CudaSlice<u8>,
                          rb: &i32|
         -> Vec<f32> {
            let mut d_out = stream.alloc_zeros::<f32>(m * n).unwrap();
            let mut l = stream.launch_builder(f);
            l.arg(&d_qact)
                .arg(w)
                .arg(&d_sc)
                .arg(&d_as)
                .arg(&d_res)
                .arg(&mut d_out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .arg(rb);
            // SAFETY: residual-twin signature; grid.y = m.
            #[allow(unsafe_code)]
            unsafe {
                l.launch(cfg).unwrap()
            };
            let mut out = vec![0.0f32; m * n];
            stream.memcpy_dtoh(&d_out, &mut out).unwrap();
            out
        };
        let r2 = launch_res(&f_tq2_res, &d_w2, &rb2_i);
        let r1 = launch_res(&f_tq1_res, &d_w1, &rb1_i);
        for (i, (a, b)) in r2.iter().zip(&r1).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "residual m{m} n{n} k{k} [{i}]: tq2={a} tq1={b}"
            );
        }
    }
}

/// Mixed-sign random trits (~1/3 each of -1/0/+1) — the A2/A4 bit-equality
/// gates MUST use these: `make_sparse_trits` zeroes leading blocks and emits
/// no -1 at all, which made the original gates vacuous (review-found: both
/// kernels correctly output 0.0 on all-zero weights, proving nothing).
#[cfg(test)]
fn mixed_trits(n: usize, k: usize, seed: u64) -> Vec<tritium_core::Trit> {
    let mut s = seed;
    (0..n * k)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            tritium_core::Trit::from_i8(((s >> 33) % 3) as i8 - 1).unwrap()
        })
        .collect()
}

/// A4 harness: upload TB1 rows (concatenated variable-length + offsets) and
/// launch the prototype kernel.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn run_tb1(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    f: &cudarc::driver::CudaFunction,
    trits: &[tritium_core::Trit],
    qact: &[i8],
    scales: &[f32],
    act_scale: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    let mut arena: Vec<u8> = Vec::new();
    // u32 offsets cap the arena at 4 GiB — fine for the prototype scale.
    let mut offsets: Vec<u32> = Vec::with_capacity(n);
    for ni in 0..n {
        offsets.push(arena.len() as u32);
        arena.extend(tritium_format::pack_tb1_row(&trits[ni * k..(ni + 1) * k]).unwrap());
    }
    arena.extend_from_slice(&[0u8; 4]); // sign-read slack (kernel loads byte0+1)
    let d_w = stream.clone_htod(&arena).unwrap();
    let d_off = stream.clone_htod(&offsets).unwrap();
    let d_qact = stream.clone_htod(qact).unwrap();
    let d_sc = stream.clone_htod(scales).unwrap();
    let d_as = stream.clone_htod(act_scale).unwrap();
    let mut d_out = stream.alloc_zeros::<f32>(m * n).unwrap();
    let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(8), m as u32, 1),
        block_dim: (8 * 32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut l = stream.launch_builder(f);
    l.arg(&d_qact)
        .arg(&d_w)
        .arg(&d_off)
        .arg(&d_sc)
        .arg(&d_as)
        .arg(&mut d_out)
        .arg(&m_i)
        .arg(&n_i)
        .arg(&k_i);
    // SAFETY: tb1_mpgemm_tiled_i8_scaled(qact, weights, row_offsets, scales,
    // act_scale, out, m, n, k); grid.y = m.
    #[allow(unsafe_code)]
    unsafe {
        l.launch(cfg).unwrap()
    };
    let mut out = vec![0.0f32; m * n];
    stream.memcpy_dtoh(&d_out, &mut out).unwrap();
    out
}

/// A4 gate: TB1 bitmap+signs kernel is BIT-identical to the TQ2 i8-scaled
/// kernel on the same trits (integer accumulation, same epilogue).
#[test]
fn tb1_matches_tq2_tiled_scaled_bit_exact() {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::Ptx;
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping tb1 gate: no device ({e})");
            return;
        }
    };
    let stream = ctx.default_stream();
    let module = ctx.load_module(Ptx::from_src(TQ2_0_ADD_PTX)).unwrap();
    let f_tq2 = module
        .load_function("tq2_0_add_mpgemm_tiled_i8_scaled")
        .unwrap();
    let f_tb1 = module.load_function("tb1_mpgemm_tiled_i8_scaled").unwrap();

    for &(m, n, k) in &[(1usize, 8usize, 1024usize), (2, 5, 512)] {
        let trits = mixed_trits(n, k, 0xB1 ^ (k as u64));
        let nb = num_blocks(k);
        let unit = vec![half::f16::ONE; nb];
        let rb2 = nb * TQ2_0_BLOCK_BYTES;
        let mut p2 = vec![0u8; n * rb2];
        for ni in 0..n {
            tritium_format::pack_tq2_0_row(
                &trits[ni * k..(ni + 1) * k],
                &unit,
                &mut p2[ni * rb2..(ni + 1) * rb2],
            )
            .unwrap();
        }
        let qact: Vec<i8> = (0..m * k).map(|i| ((i * 41 + 5) % 251) as i8).collect();
        let scales = seeded_f32(3, n, 0.5, 2.0);
        let act_scale = seeded_f32(9, m, 0.5, 1.5);

        // TQ2 reference launch.
        let d_qact = stream.clone_htod(&qact).unwrap();
        let d_w2 = stream.clone_htod(&p2).unwrap();
        let d_sc = stream.clone_htod(&scales).unwrap();
        let d_as = stream.clone_htod(&act_scale).unwrap();
        let mut d_out = stream.alloc_zeros::<f32>(m * n).unwrap();
        let (m_i, n_i, k_i, rb_i) = (m as i32, n as i32, k as i32, rb2 as i32);
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(8), m as u32, 1),
            block_dim: (8 * 32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = stream.launch_builder(&f_tq2);
        l.arg(&d_qact)
            .arg(&d_w2)
            .arg(&d_sc)
            .arg(&d_as)
            .arg(&mut d_out)
            .arg(&m_i)
            .arg(&n_i)
            .arg(&k_i)
            .arg(&rb_i);
        // SAFETY: 9-arg dense signature; grid.y = m.
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg).unwrap()
        };
        let mut o2 = vec![0.0f32; m * n];
        stream.memcpy_dtoh(&d_out, &mut o2).unwrap();

        let o1 = run_tb1(&stream, &f_tb1, &trits, &qact, &scales, &act_scale, m, n, k);
        for (i, (a, b)) in o2.iter().zip(&o1).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "tb1 m{m} n{n} k{k} [{i}]: tq2={a} tb1={b}"
            );
        }
    }
}

/// A4 verdict bench (run explicitly): TQ2 vs TQ1 vs TB1 kernel wall-time on
/// the REAL gateup shape (the one DRAM-bound decode GEMM) at M=1.
#[test]
#[ignore = "A4 head-to-head bench: run with --ignored --nocapture"]
fn tb1_tq1_tq2_gateup_bench() {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::Ptx;
    use tritium_format::TQ1_0_BLOCK_BYTES;
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping bench: no device ({e})");
            return;
        }
    };
    let stream = ctx.default_stream();
    let module = ctx.load_module(Ptx::from_src(TQ2_0_ADD_PTX)).unwrap();
    let (m, n, k) = (1usize, 13824usize, 2560usize); // BitNet fused gateup
    let iters = 2000u32;

    // Mixed-sign ~1/3 zeros — closer to BitNet's 42% than block-structured
    // patterns; the bench also PRINTS actual byte counts (self-documenting).
    let trits = mixed_trits(n, k, 0xBE);
    let nb = num_blocks(k);
    let unit = vec![half::f16::ONE; nb];
    let (rb2, rb1) = (nb * TQ2_0_BLOCK_BYTES, nb * TQ1_0_BLOCK_BYTES);
    let mut p2 = vec![0u8; n * rb2];
    let mut p1 = vec![0u8; n * rb1];
    let mut tb1: Vec<u8> = Vec::new();
    let mut off: Vec<u32> = Vec::new();
    for ni in 0..n {
        let row = &trits[ni * k..(ni + 1) * k];
        tritium_format::pack_tq2_0_row(row, &unit, &mut p2[ni * rb2..(ni + 1) * rb2]).unwrap();
        tritium_format::pack_tq1_0_row(row, &unit, &mut p1[ni * rb1..(ni + 1) * rb1]).unwrap();
        off.push(tb1.len() as u32);
        tb1.extend(tritium_format::pack_tb1_row(row).unwrap());
    }
    tb1.extend_from_slice(&[0u8; 4]);
    println!(
        "weight bytes: TQ2 {} | TQ1 {} ({:.1}%) | TB1 {} ({:.1}%)",
        n * rb2,
        n * rb1,
        (n * rb1) as f64 / (n * rb2) as f64 * 100.0,
        tb1.len(),
        tb1.len() as f64 / (n * rb2) as f64 * 100.0,
    );

    let qact: Vec<i8> = (0..m * k).map(|i| ((i * 37) % 253) as i8).collect();
    let scales = seeded_f32(1, n, 0.5, 2.0);
    let act_scale = seeded_f32(2, m, 0.5, 1.5);
    let d_qact = stream.clone_htod(&qact).unwrap();
    let d_sc = stream.clone_htod(&scales).unwrap();
    let d_as = stream.clone_htod(&act_scale).unwrap();
    let d_w2 = stream.clone_htod(&p2).unwrap();
    let d_w1 = stream.clone_htod(&p1).unwrap();
    let d_wb = stream.clone_htod(&tb1).unwrap();
    let d_off = stream.clone_htod(&off).unwrap();
    let mut d_out = stream.alloc_zeros::<f32>(m * n).unwrap();
    let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);
    let (rb2_i, rb1_i) = (rb2 as i32, rb1 as i32);
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(8), m as u32, 1),
        block_dim: (8 * 32, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut bench = |name: &str, which: u8| {
        let f = match which {
            0 => module
                .load_function("tq2_0_add_mpgemm_tiled_i8_scaled")
                .unwrap(),
            1 => module
                .load_function("tq1_0_add_mpgemm_tiled_i8_scaled")
                .unwrap(),
            _ => module.load_function("tb1_mpgemm_tiled_i8_scaled").unwrap(),
        };
        // Warm.
        for _ in 0..50 {
            let mut l = stream.launch_builder(&f);
            match which {
                0 => l
                    .arg(&d_qact)
                    .arg(&d_w2)
                    .arg(&d_sc)
                    .arg(&d_as)
                    .arg(&mut d_out)
                    .arg(&m_i)
                    .arg(&n_i)
                    .arg(&k_i)
                    .arg(&rb2_i),
                1 => l
                    .arg(&d_qact)
                    .arg(&d_w1)
                    .arg(&d_sc)
                    .arg(&d_as)
                    .arg(&mut d_out)
                    .arg(&m_i)
                    .arg(&n_i)
                    .arg(&k_i)
                    .arg(&rb1_i),
                _ => l
                    .arg(&d_qact)
                    .arg(&d_wb)
                    .arg(&d_off)
                    .arg(&d_sc)
                    .arg(&d_as)
                    .arg(&mut d_out)
                    .arg(&m_i)
                    .arg(&n_i)
                    .arg(&k_i),
            };
            // SAFETY: signatures as gated bit-exact above.
            #[allow(unsafe_code)]
            unsafe {
                l.launch(cfg).unwrap()
            };
        }
        stream.synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let mut l = stream.launch_builder(&f);
            match which {
                0 => l
                    .arg(&d_qact)
                    .arg(&d_w2)
                    .arg(&d_sc)
                    .arg(&d_as)
                    .arg(&mut d_out)
                    .arg(&m_i)
                    .arg(&n_i)
                    .arg(&k_i)
                    .arg(&rb2_i),
                1 => l
                    .arg(&d_qact)
                    .arg(&d_w1)
                    .arg(&d_sc)
                    .arg(&d_as)
                    .arg(&mut d_out)
                    .arg(&m_i)
                    .arg(&n_i)
                    .arg(&k_i)
                    .arg(&rb1_i),
                _ => l
                    .arg(&d_qact)
                    .arg(&d_wb)
                    .arg(&d_off)
                    .arg(&d_sc)
                    .arg(&d_as)
                    .arg(&mut d_out)
                    .arg(&m_i)
                    .arg(&n_i)
                    .arg(&k_i),
            };
            #[allow(unsafe_code)]
            // SAFETY: each branch above pushes the exact argument list for its
            // selected kernel; all device buffers cover the configured shape.
            unsafe {
                l.launch(cfg).unwrap()
            };
        }
        stream.synchronize().unwrap();
        let us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(iters);
        println!("{name}: {us:.2} µs/launch");
    };
    bench("TQ2 (2.06 b/w)", 0);
    bench("TQ1 (1.69 b/w)", 1);
    bench("TB1 (1.58 b/w)", 2);
}

/// **The framework-external linear path must not fall off a cliff for M > 1.**
///
/// Reported externally on a 4090 (2048x2048 vs torch fp16): 5.4x slower at M=1 but **830x at
/// M=2048**, growing linearly in M. Root cause in `tq2_projected_linear_forward`: one thread per
/// output element looping `k`, so (a) each of the M rows re-reads the whole weight matrix -- weight
/// traffic O(M*N*K) instead of O(N*K) -- and (b) adjacent threads differ in `ni` and touch addresses
/// `row_bytes` apart, so one warp load becomes 32 transactions. Both are invisible at M=1, which is
/// why decode benchmarks never caught it.
///
/// `tq2_0_add_mpgemm_tiled_f32_bias` stages the activation row in shared memory and gives each warp
/// one output column. This gate launches BOTH kernels on byte-identical inputs and demands they
/// agree across the M range where they diverge, plus the NaN contract the torch layer relies on.
#[test]
fn external_linear_tiled_matches_the_untiled_kernel_across_m() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let kernels = backend
        .external_kernels_for_test()
        .expect("external kernels");
    let stream = backend.stream();
    let (n, k) = (96usize, 512usize);
    let row_bytes = k.div_ceil(256) * 66;

    // Each byte holds four 2-bit codes and TQ2_0 only ever emits {0,1,2} (trit = code - 1). Code 3
    // is unreachable from any packer, and the two kernels deliberately disagree on it: the untiled
    // one falls through to zero, the tiled one computes `code - 1 = 2`. Feeding random bytes would
    // therefore fail this gate on malformed input that no producer can generate.
    let mut packed = vec![0u8; n * row_bytes];
    let mut st = 0x51D3u64;
    for b in packed.iter_mut() {
        let mut byte = 0u8;
        for slot in 0..4 {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            byte |= ((st % 3) as u8) << (2 * slot);
        }
        *b = byte;
    }
    let scales: Vec<f32> = (0..n).map(|i| 0.01 + (i % 7) as f32 * 0.003).collect();
    let bias: Vec<f32> = (0..n).map(|i| (i % 5) as f32 * 0.1 - 0.2).collect();
    let d_packed = backend.dev_upload_u8(&packed).unwrap();
    let d_scales = backend.dev_upload(&scales).unwrap();
    let d_bias = backend.dev_upload(&bias).unwrap();

    for m in [1usize, 2, 4, 16, 64, 129] {
        let act: Vec<f32> = (0..m * k)
            .map(|i| ((i * 2654435761) % 2048) as f32 / 1024.0 - 1.0)
            .collect();
        let d_act = backend.dev_upload(&act).unwrap();
        let mut d_tiled = backend.dev_alloc_zeros(m * n).unwrap();
        let mut d_plain = backend.dev_alloc_zeros(m * n).unwrap();
        let (mi, ni, ki, rb) = (m as i32, n as i32, k as i32, row_bytes as i32);

        for tiled in [true, false] {
            let out = if tiled { &mut d_tiled } else { &mut d_plain };
            let (a_p, _ga) = d_act.device_ptr(stream);
            let (w_p, _gw) = d_packed.device_ptr(stream);
            let (s_p, _gs) = d_scales.device_ptr(stream);
            let (b_p, _gb) = d_bias.device_ptr(stream);
            let (o_p, _go) = out.device_ptr(stream);
            let mut params = [
                pp(&a_p),
                pp(&w_p),
                pp(&s_p),
                pp(&b_p),
                pp(&o_p),
                pp(&mi),
                pp(&ni),
                pp(&ki),
                pp(&rb),
            ];
            let (func, grid, block, shared) = if tiled {
                const WPB: u32 = 8;
                (
                    kernels.forward_tiled_for_test(),
                    ((n as u32).div_ceil(WPB), m as u32, 1),
                    (WPB * 32, 1, 1),
                    (k * 4) as u32,
                )
            } else {
                (
                    kernels.forward_for_test(),
                    (((m * n) as u32).div_ceil(256), 1, 1),
                    (256, 1, 1),
                    0,
                )
            };
            raw_launch(func, grid, block, shared, stream.cu_stream(), &mut params).expect("launch");
        }

        let mut got = vec![0.0f32; m * n];
        let mut want = vec![0.0f32; m * n];
        backend.dev_download(&d_tiled, &mut got).unwrap();
        backend.dev_download(&d_plain, &mut want).unwrap();
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 2e-4,
            "M={m}: tiled must match the untiled kernel it replaces, max|delta| {worst:.3e}"
        );
        assert!(
            got.iter().all(|v| v.is_finite()),
            "M={m}: finite in, finite out"
        );
    }
}

/// The fp16 tiled kernel must match the fp16 untiled one it replaces.
///
/// This is the path that actually matters for training: PyTorch autocast hands `ternary_linear`
/// fp16 activations, and a **bf16** autocast is cast to fp16 too by
/// `_ternary_linear_cuda_autocast`, so an f32-only fast path never runs in a mixed-precision loop.
#[test]
fn external_linear_tiled_f16_matches_the_untiled_f16_kernel() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let kernels = backend
        .external_kernels_for_test()
        .expect("external kernels");
    let stream = backend.stream();
    let (n, k) = (96usize, 512usize);
    let row_bytes = k.div_ceil(256) * 66;
    let mut packed = vec![0u8; n * row_bytes];
    let mut st = 0x2C9Fu64;
    for b in packed.iter_mut() {
        let mut byte = 0u8;
        for slot in 0..4 {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            byte |= ((st % 3) as u8) << (2 * slot);
        }
        *b = byte;
    }
    let scales: Vec<f32> = (0..n).map(|i| 0.02 + (i % 5) as f32 * 0.004).collect();
    let d_packed = backend.dev_upload_u8(&packed).unwrap();
    let d_scales = backend.dev_upload(&scales).unwrap();

    for m in [1usize, 4, 33, 128] {
        // fp16 bit patterns built on the host, uploaded as raw u8.
        let act_f32: Vec<f32> = (0..m * k)
            .map(|i| ((i * 40503) % 512) as f32 / 512.0 - 0.5)
            .collect();
        let act_h: Vec<u8> = act_f32
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect();
        let bias_h: Vec<u8> = (0..n)
            .flat_map(|i| half::f16::from_f32((i % 3) as f32 * 0.05).to_le_bytes())
            .collect();
        let d_act = backend.dev_upload_u8(&act_h).unwrap();
        let d_bias = backend.dev_upload_u8(&bias_h).unwrap();
        let d_tiled = backend.dev_upload_u8(&vec![0u8; m * n * 2]).unwrap();
        let d_plain = backend.dev_upload_u8(&vec![0u8; m * n * 2]).unwrap();
        let (mi, ni, ki, rb) = (m as i32, n as i32, k as i32, row_bytes as i32);

        for tiled in [true, false] {
            let out = if tiled { &d_tiled } else { &d_plain };
            let (a_p, _ga) = d_act.device_ptr(stream);
            let (w_p, _gw) = d_packed.device_ptr(stream);
            let (s_p, _gs) = d_scales.device_ptr(stream);
            let (b_p, _gb) = d_bias.device_ptr(stream);
            let (o_p, _go) = out.device_ptr(stream);
            let mut params = [
                pp(&a_p),
                pp(&w_p),
                pp(&s_p),
                pp(&b_p),
                pp(&o_p),
                pp(&mi),
                pp(&ni),
                pp(&ki),
                pp(&rb),
            ];
            let (func, grid, block, shared) = if tiled {
                const WPB: u32 = 8;
                (
                    kernels.forward_tiled_f16_for_test(),
                    ((n as u32).div_ceil(WPB), m as u32, 1),
                    (WPB * 32, 1, 1),
                    (k * 4) as u32,
                )
            } else {
                (
                    kernels.forward_f16_for_test(),
                    (((m * n) as u32).div_ceil(256), 1, 1),
                    (256, 1, 1),
                    0,
                )
            };
            raw_launch(func, grid, block, shared, stream.cu_stream(), &mut params).expect("launch");
        }

        let mut got_b = vec![0u8; m * n * 2];
        let mut want_b = vec![0u8; m * n * 2];
        backend.dev_download_u8(&d_tiled, &mut got_b).unwrap();
        backend.dev_download_u8(&d_plain, &mut want_b).unwrap();
        let to_f32 = |b: &[u8]| -> Vec<f32> {
            b.chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        };
        let (got, want) = (to_f32(&got_b), to_f32(&want_b));
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // Tolerance is one fp16 ULP at this magnitude: the tiled kernel converts each activation
        // ONCE while staging, the untiled one re-converts per output column, so the two can differ
        // in the last bit. That difference favours the tiled kernel.
        assert!(
            worst < 8e-3,
            "M={m}: fp16 tiled must match fp16 untiled, max|delta| {worst:.3e}"
        );
    }
}

/// The point of the swap, measured: cost must stop scaling linearly in M.
///
/// Reports both kernels at the reporter's shape (2048x2048) across their M sweep. This is a
/// SPEEDUP RATIO gate, not an absolute-latency one -- absolute numbers depend on the card and on
/// whatever else shares it, but the untiled kernel's O(M) weight re-reads have to show up as a
/// widening gap no matter the box.
#[test]
#[ignore = "benchmark; run explicitly with --ignored --nocapture"]
fn external_linear_tiled_removes_the_m_cliff() {
    let Ok(backend) = CudaBackend::new(0) else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let kernels = backend
        .external_kernels_for_test()
        .expect("external kernels");
    let stream = backend.stream();
    let (n, k) = (2048usize, 2048usize);
    let row_bytes = k.div_ceil(256) * 66;
    let mut packed = vec![0u8; n * row_bytes];
    let mut st = 0xA71Fu64;
    for b in packed.iter_mut() {
        let mut byte = 0u8;
        for slot in 0..4 {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            byte |= ((st % 3) as u8) << (2 * slot);
        }
        *b = byte;
    }
    let scales = vec![0.02f32; n];
    let d_packed = backend.dev_upload_u8(&packed).unwrap();
    let d_scales = backend.dev_upload(&scales).unwrap();

    println!(
        "{:>7} {:>12} {:>12} {:>10}",
        "M", "untiled ms", "tiled ms", "speedup"
    );
    for m in [1usize, 4, 16, 64, 256, 2048] {
        let act = vec![0.05f32; m * k];
        let d_act = backend.dev_upload(&act).unwrap();
        let d_out = backend.dev_alloc_zeros(m * n).unwrap();
        let (mi, ni, ki, rb) = (m as i32, n as i32, k as i32, row_bytes as i32);
        let mut ms = [0.0f64; 2];
        for (slot, tiled) in [false, true].into_iter().enumerate() {
            for iter in 0..12 {
                let t0 = std::time::Instant::now();
                {
                    let (a_p, _ga) = d_act.device_ptr(stream);
                    let (w_p, _gw) = d_packed.device_ptr(stream);
                    let (s_p, _gs) = d_scales.device_ptr(stream);
                    let nullb = 0u64;
                    let (o_p, _go) = d_out.device_ptr(stream);
                    let mut params = [
                        pp(&a_p),
                        pp(&w_p),
                        pp(&s_p),
                        pp(&nullb),
                        pp(&o_p),
                        pp(&mi),
                        pp(&ni),
                        pp(&ki),
                        pp(&rb),
                    ];
                    let (func, grid, block, shared) = if tiled {
                        const WPB: u32 = 8;
                        (
                            kernels.forward_tiled_for_test(),
                            ((n as u32).div_ceil(WPB), m as u32, 1),
                            (WPB * 32, 1, 1),
                            (k * 4) as u32,
                        )
                    } else {
                        (
                            kernels.forward_for_test(),
                            (((m * n) as u32).div_ceil(256), 1, 1),
                            (256, 1, 1),
                            0,
                        )
                    };
                    raw_launch(func, grid, block, shared, stream.cu_stream(), &mut params)
                        .expect("launch");
                }
                backend.dev_synchronize().expect("sync");
                if iter >= 2 {
                    ms[slot] += t0.elapsed().as_secs_f64() * 1e3;
                }
            }
            ms[slot] /= 10.0;
        }
        println!(
            "{m:>7} {:>12.3} {:>12.3} {:>9.1}x",
            ms[0],
            ms[1],
            ms[0] / ms[1]
        );
    }
}

// ── Sparsity crossover sweep (WS-4 A2) ───────────────────────────────────────────────────────────
// At what zero density, and with what STRUCTURE, does skipping actually pay?
//
// The repo's recorded answer is "it never does" — round 10b measured base-243 entropy packing
// 1.19-1.37x slower than TQ2_0 dp4a, and round 16 measured TB1 at 2.58x slower, concluding "byte
// savings of 18-27% bought 77-196% more time". Both stand as ALU-cost measurements. Neither is a
// byte-cost measurement, because in both harnesses the weights never left L2:
// `tb1_tq1_tq2_gateup_bench` uploads ONE 9.12 MB weight buffer (10 blocks x 66 B x 13824 rows) and
// launches against it 2000 times, on a device with 72 MB of L2. After the first iteration every
// byte is served at L2 bandwidth, so an 18% byte saving had nothing to save. The log's own warning
// (OPTIMIZATION-LOG.md, "in-model the 30 layers' ~530 MB defeat L2") was never applied to the
// harness that produced the verdict.
//
// So L2 defeat is this sweep's PRIMARY CONTROL, not a caveat: every point is measured twice, once
// resident and once against enough rotating weight replicas to overflow L2. The gap between the two
// is the kernel's byte-sensitivity, and it is the quantity that decides whether a byte-saving format
// can ever win. If the ordering inverts, round 16 gets an amendment.
//
// Three further things this sweep does that a density-only sweep cannot:
//
//   * **Structure, not just density.** `mixed_trits` is uniform-1/3 unstructured — simultaneously the
//     worst case for block-skip (which needs CONTIGUOUS all-zero 256-blocks) and the best case for
//     TB1 (which wants element sparsity). Element density `p` and block-clustering `q` are
//     independent axes and the block-skip kernel responds only to `q`.
//   * **Batch.** Every recorded negative is M=1, the latency-bound regime. The skip is per
//     (row, block), so one skipped block is skipped for every one of the M rows: the win scales with
//     M, and that axis has never been measured.
//   * **A real control.** The sparse kernel at `q = 0` has the bitmap machinery fully active with
//     nothing to skip. If that is not within a couple of percent of the dense kernel, the harness is
//     measuring codegen luck and every crossover it reports is noise.
//
// S34 is deliberately absent: its density is fixed at exactly 0.25 (a point, not an axis, and BELOW
// BitNet's natural 42.2%), and its only GPU path is the one-thread-per-output `--fmad=false`
// conformance kernel. It is a bits play, not a speed play.

/// How the zeros are arranged, which matters far more than how many there are.
#[derive(Clone, Copy, Debug)]
enum SparsityStructure {
    /// Bernoulli(`p`) zeros, independent per element. Block-skip's worst case.
    Unstructured { p: f64 },
    /// Fraction `q` of aligned 256-trit blocks forced entirely zero; the rest carry whatever
    /// residual density brings the total to `p`. The only structure block-skip can exploit.
    BlockClustered { p: f64, q: f64 },
    /// Fraction `f` of output rows entirely zero — the realistic pattern (round 15 found one BitNet
    /// gate/up tensor with 43.6% dead neurons). Every block of a dead row is skippable.
    RowDead { p: f64, f: f64 },
    /// Exactly `m - keep` zeros per aligned `m`-group: 6:8, 2:4, and S34's one-in-four in one
    /// generator. What ADR 0024 targets.
    NPerM { keep: usize, m: usize },
}

impl SparsityStructure {
    fn label(self) -> String {
        match self {
            Self::Unstructured { p } => format!("unstruct p={p:.2}"),
            Self::BlockClustered { p, q } => format!("blockclust p={p:.2} q={q:.2}"),
            Self::RowDead { p, f } => format!("rowdead p={p:.2} f={f:.2}"),
            Self::NPerM { keep, m } => format!("{keep}:{m}"),
        }
    }
}

/// Build `[n, k]` trits with the requested zero structure. Deterministic in `seed`.
fn trits_structured(
    n: usize,
    k: usize,
    spec: SparsityStructure,
    seed: u64,
) -> Vec<tritium_core::Trit> {
    let mut s = seed | 1;
    let mut next = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 33) as f64 / (1u64 << 31) as f64
    };
    let sign = |u: f64| {
        if u < 0.5 {
            tritium_core::Trit::from_i8(-1).unwrap()
        } else {
            tritium_core::Trit::from_i8(1).unwrap()
        }
    };
    let zero = tritium_core::Trit::from_i8(0).unwrap();
    let mut out = vec![zero; n * k];
    let nb = k.div_ceil(QK_K);

    match spec {
        SparsityStructure::Unstructured { p } => {
            for v in out.iter_mut() {
                *v = if next() < p { zero } else { sign(next()) };
            }
        }
        SparsityStructure::BlockClustered { p, q } => {
            // Dead blocks contribute q of the total zeros; the survivors carry the remainder, so the
            // element density still lands on p.
            let residual = if q >= 1.0 {
                0.0
            } else {
                ((p - q) / (1.0 - q)).clamp(0.0, 1.0)
            };
            for row in 0..n {
                for b in 0..nb {
                    let dead = next() < q;
                    let lo = row * k + b * QK_K;
                    let hi = (lo + QK_K).min(row * k + k);
                    for v in out[lo..hi].iter_mut() {
                        *v = if dead || next() < residual {
                            zero
                        } else {
                            sign(next())
                        };
                    }
                }
            }
        }
        SparsityStructure::RowDead { p, f } => {
            let residual = if f >= 1.0 {
                0.0
            } else {
                ((p - f) / (1.0 - f)).clamp(0.0, 1.0)
            };
            for row in 0..n {
                let dead = next() < f;
                for v in out[row * k..(row + 1) * k].iter_mut() {
                    *v = if dead || next() < residual {
                        zero
                    } else {
                        sign(next())
                    };
                }
            }
        }
        SparsityStructure::NPerM { keep, m } => {
            for row in 0..n {
                for g in (0..k).step_by(m) {
                    let hi = (g + m).min(k);
                    // Keep the first `keep` slots of a randomly rotated window — an arbitrary but
                    // uniform placement, which is what an N:M kernel's index metadata assumes.
                    let rot = (next() * m as f64) as usize;
                    for (j, idx) in (g..hi).enumerate() {
                        let slot = (j + rot) % m;
                        out[row * k + idx] = if slot < keep { sign(next()) } else { zero };
                    }
                }
            }
        }
    }
    out
}

/// Measured element-zero fraction and all-zero-256-block fraction of a trit buffer — reported so a
/// requested `(p, q)` can never be confused with an achieved one.
fn zero_census(trits: &[tritium_core::Trit], n: usize, k: usize) -> (f64, f64) {
    let zeros = trits
        .iter()
        .filter(|t| **t == tritium_core::Trit::ZERO)
        .count();
    let nb = k.div_ceil(QK_K);
    let mut dead_blocks = 0usize;
    for row in 0..n {
        for b in 0..nb {
            let lo = row * k + b * QK_K;
            let hi = (lo + QK_K).min(row * k + k);
            if trits[lo..hi].iter().all(|t| *t == tritium_core::Trit::ZERO) {
                dead_blocks += 1;
            }
        }
    }
    (
        zeros as f64 / (n * k) as f64,
        dead_blocks as f64 / (n * nb) as f64,
    )
}

/// **The crossover sweep.** For each (kernel, structure, M), report µs/launch both L2-resident and
/// L2-defeated, so the byte-saving and ALU-cost components are separable.
///
/// Not a correctness test — `tb1_matches_tq2_tiled_scaled_bit_exact` and the sparse kernel's
/// NULL-bitmap identity own that. This one only times kernels those gates already proved equal.
#[test]
#[ignore = "perf sweep: needs a quiet GPU box; run explicitly"]
fn sparsity_crossover_sweep() {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::Ptx;

    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping sweep: no device ({e})");
            return;
        }
    };
    let stream = ctx.default_stream();
    let module = ctx.load_module(Ptx::from_src(TQ2_0_ADD_PTX)).unwrap();

    // Query L2 rather than hardcoding 72 MB — there is a wgpu/ROCm lane now, and a wrong constant
    // here silently turns the primary control back into the artifact it exists to remove.
    let l2_bytes = ctx
        .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
        .map(|v| v.max(0) as usize)
        .unwrap_or(72 << 20);

    let (n, k) = (13824usize, 2560usize); // round 16's BitNet fused gateup, for continuity
    let nb = num_blocks(k);
    let rb = nb * TQ2_0_BLOCK_BYTES;
    let words_per_row = nb.div_ceil(32);
    let weight_bytes = n * rb;
    // 4x L2 of distinct weight bytes, so a launch cannot be served from cache.
    let replicas = (4 * l2_bytes).div_ceil(weight_bytes).max(2);

    println!(
        "L2 {} MiB | weight buffer {:.2} MiB | {replicas} replicas = {:.0} MiB to defeat L2\n\
         shape N={n} K={k}; every point reported L2-resident AND L2-defeated.\n",
        l2_bytes >> 20,
        weight_bytes as f64 / (1 << 20) as f64,
        (replicas * weight_bytes) as f64 / (1 << 20) as f64,
    );

    let structures = [
        SparsityStructure::Unstructured { p: 0.42 },
        SparsityStructure::BlockClustered { p: 0.42, q: 0.0 },
        SparsityStructure::BlockClustered { p: 0.50, q: 0.05 },
        SparsityStructure::BlockClustered { p: 0.60, q: 0.20 },
        SparsityStructure::BlockClustered { p: 0.80, q: 0.40 },
        SparsityStructure::RowDead { p: 0.50, f: 0.10 },
        SparsityStructure::NPerM { keep: 6, m: 8 },
    ];
    let batches = [1usize, 8, 32, 256];

    let unit = vec![f16::ONE; nb];
    let scales = seeded_f32(1, n, 0.5, 2.0);
    let d_sc = stream.clone_htod(&scales).unwrap();

    println!(
        "{:<26} {:>4} {:>7} {:>7} {:>10} {:>10} {:>8} {:>9}",
        "structure / kernel",
        "M",
        "zero%",
        "dead%",
        "L2res µs",
        "L2def µs",
        "vs dense",
        "byte-sens"
    );
    println!("{}", "-".repeat(94));

    for spec in structures {
        let trits = trits_structured(n, k, spec, 0xBE11);
        let (zero_frac, dead_frac) = zero_census(&trits, n, k);

        let mut packed = vec![0u8; n * rb];
        for ni in 0..n {
            tritium_format::pack_tq2_0_row(
                &trits[ni * k..(ni + 1) * k],
                &unit,
                &mut packed[ni * rb..(ni + 1) * rb],
            )
            .unwrap();
        }
        let bitmap = tritium_format::compute_zero_bitmaps(&packed, n, k, rb).unwrap();

        // Rotating replicas: identical contents, distinct addresses, so nothing is L2-resident.
        let d_w: Vec<_> = (0..replicas)
            .map(|_| stream.clone_htod(&packed).unwrap())
            .collect();
        let d_bm: Vec<_> = (0..replicas)
            .map(|_| stream.clone_htod(&bitmap).unwrap())
            .collect();

        for m in batches {
            let qact: Vec<i8> = (0..m * k).map(|i| ((i * 37) % 253) as i8).collect();
            let act_scale = seeded_f32(2, m, 0.5, 1.5);
            let d_qact = stream.clone_htod(&qact).unwrap();
            let d_as = stream.clone_htod(&act_scale).unwrap();
            let mut d_out = stream.alloc_zeros::<f32>(m * n).unwrap();
            let (m_i, n_i, k_i, rb_i, wpr_i) = (
                m as i32,
                n as i32,
                k as i32,
                rb as i32,
                words_per_row as i32,
            );
            let cfg = LaunchConfig {
                grid_dim: ((n as u32).div_ceil(8), m as u32, 1),
                block_dim: (8 * 32, 1, 1),
                shared_mem_bytes: 0,
            };
            // Fewer iterations at large M: the kernel itself is ~M times the work.
            let iters = (2000 / m.max(1)).max(50) as u32;

            // Time one kernel. `defeat` rotates the weight replica per launch; otherwise every
            // launch reuses replica 0 and runs out of L2, which is what round 16 measured.
            let mut bench = |sparse: bool, defeat: bool| -> (f64, f64) {
                let f = module
                    .load_function(if sparse {
                        "tq2_0_add_mpgemm_tiled_i8_scaled_sparse"
                    } else {
                        "tq2_0_add_mpgemm_tiled_i8_scaled"
                    })
                    .unwrap();
                let mut launch = |i: usize| {
                    let idx = if defeat { i % replicas } else { 0 };
                    let mut l = stream.launch_builder(&f);
                    if sparse {
                        l.arg(&d_qact)
                            .arg(&d_w[idx])
                            .arg(&d_sc)
                            .arg(&d_as)
                            .arg(&d_bm[idx])
                            .arg(&mut d_out)
                            .arg(&m_i)
                            .arg(&n_i)
                            .arg(&k_i)
                            .arg(&rb_i)
                            .arg(&wpr_i);
                    } else {
                        l.arg(&d_qact)
                            .arg(&d_w[idx])
                            .arg(&d_sc)
                            .arg(&d_as)
                            .arg(&mut d_out)
                            .arg(&m_i)
                            .arg(&n_i)
                            .arg(&k_i)
                            .arg(&rb_i);
                    }
                    // SAFETY: both signatures are exercised by the bit-exactness gates above.
                    #[allow(unsafe_code)]
                    unsafe {
                        l.launch(cfg).unwrap()
                    };
                };
                for i in 0..50 {
                    launch(i);
                }
                stream.synchronize().unwrap();
                // Min-of-N across reps: contention only ever inflates, so the minimum is the
                // cleanest estimator (a median already crowned a wrong winner on this box once).
                // σ across reps is reported so a within-noise "win" cannot be mistaken for one.
                let mut samples = Vec::new();
                for _ in 0..5 {
                    let t0 = std::time::Instant::now();
                    for i in 0..iters as usize {
                        launch(i);
                    }
                    stream.synchronize().unwrap();
                    samples.push(t0.elapsed().as_secs_f64() * 1e6 / f64::from(iters));
                }
                let best = samples.iter().copied().fold(f64::INFINITY, f64::min);
                let mean = samples.iter().sum::<f64>() / samples.len() as f64;
                let var =
                    samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / samples.len() as f64;
                (best, var.sqrt())
            };

            let (dense_res, _) = bench(false, false);
            let (dense_def, dense_sigma) = bench(false, true);
            let (sparse_res, _) = bench(true, false);
            let (sparse_def, sparse_sigma) = bench(true, true);

            for (name, res, def, sigma, base) in [
                ("dense tq2 i8", dense_res, dense_def, dense_sigma, dense_def),
                (
                    "sparse blockskip",
                    sparse_res,
                    sparse_def,
                    sparse_sigma,
                    dense_def,
                ),
            ] {
                println!(
                    "{:<26} {m:>4} {:>6.1}% {:>6.1}% {res:>10.2} {def:>10.2} {:>7.3}x {:>8.2}x  (σ {sigma:.2})",
                    format!("{} / {name}", spec.label()),
                    zero_frac * 100.0,
                    dead_frac * 100.0,
                    base / def,
                    def / res,
                );
            }
        }
        println!();
    }

    println!(
        "byte-sens = L2-defeated / L2-resident. A kernel whose time barely moves when its weights \
         stop fitting in cache is ALU-bound, and no byte saving can help it; one that slows a lot \
         is byte-bound, and a denser or sparser format has room to pay. Round 16 measured only the \
         L2-resident column."
    );
}
