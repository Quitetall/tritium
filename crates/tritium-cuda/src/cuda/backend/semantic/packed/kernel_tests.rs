//! Explicit physical diagnostics, not empirical release qualification.
use super::*;
use std::time::{Duration, Instant};
use tritium_core::{PerTilePlan, Trit, UniformPlan, reference_ternary_matmul};
use tritium_schema::{ADMITTED_LAWS, AdditiveLayout, PlaneAllocation, Transport};
use tritium_spec::{TensorMetadata, TensorStreamEncoding, TernaryBackend};

fn fixture(
    rows: usize,
    cols: usize,
    codec: PlaneCodec,
    planes: u8,
    compact: bool,
    basis: Basis,
) -> AdditiveTensor {
    let layout = AdditiveLayout {
        rows: rows as u64,
        cols: cols as u64,
        tile: 256,
        group: 128,
        max_planes: planes,
        allocation: if compact {
            PlaneAllocation::PerTile
        } else {
            PlaneAllocation::Uniform
        },
        codec,
        law: ADMITTED_LAWS[usize::from(!compact)].law,
        basis,
        transport: Transport::Raw,
    };
    let counts = (0..rows * cols.div_ceil(256))
        .map(|i| [planes, planes, 1, 0][i % 4])
        .collect::<Vec<_>>();
    let (trit_count, scale_count) = if compact {
        let plan = PerTilePlan::new(layout, &counts).unwrap();
        (plan.trit_count(), plan.scale_count())
    } else {
        let plan = UniformPlan::new(layout, planes).unwrap();
        (plan.trit_count(), plan.scale_count())
    };
    let trits = (0..trit_count)
        .map(|i| match (i.wrapping_mul(17) ^ (i / 7)) % 3 {
            0 => Trit::NEG,
            1 => Trit::ZERO,
            _ => Trit::POS,
        })
        .collect();
    let scales = (0..scale_count)
        .map(|i| 0.125 + (i % 7) as f32 * 0.0625)
        .collect();
    if compact {
        AdditiveTensor::new_per_tile(layout, counts, trits, scales).unwrap()
    } else {
        AdditiveTensor::new(layout, planes, trits, scales).unwrap()
    }
}

fn median(mut values: Vec<Duration>) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn launch_kernel<const TIMED: bool>(
    backend: &CudaBackend,
    packed: &PackedAdditive,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    batch: u32,
    outputs: usize,
    cooperative: bool,
) -> f64 {
    let (function, config) = if cooperative {
        (
            &backend.func_semantic_additive_matmul_ordered_warp,
            LaunchConfig {
                grid_dim: (outputs.div_ceil(4) as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            },
        )
    } else {
        (
            &backend.func_semantic_additive_matmul,
            LaunchConfig::for_num_elems(outputs as u32),
        )
    };
    let mut launch = backend.stream.launch_builder(function);
    launch.arg(&packed.bytes).arg(input).arg(output).arg(&batch);
    if TIMED {
        launch.record_kernel_launch(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
    }
    #[allow(unsafe_code)]
    // SAFETY: admitted payloads, bounded outputs; cooperative blocks use whole warps.
    let events = unsafe { launch.launch(config) }.unwrap();
    if let Some((start, end)) = events {
        f64::from(start.elapsed_ms(&end).unwrap()) * 1000.
    } else {
        0.
    }
}

#[test]
#[ignore = "explicit physical packed-kernel timing diagnostic; not a release gate"]
fn physical_ordered_packed_kernel_timing() {
    timing(
        &[
            (512, 512),
            (2048, 2048),
            (5120, 5120),
            (5120, 17408),
            (32768, 512),
        ],
        5,
    );
}

#[test]
#[ignore = "explicit small-shape physical timing diagnostic; not a release gate"]
fn physical_ordered_small_shape_timing() {
    timing(&[(512, 512)], 31);
}

fn timing(shapes: &[(usize, usize)], sample_count: usize) {
    let backend = CudaBackend::new(0).expect("requires physical CUDA");
    for &(rows, cols) in shapes {
        for codec in [PlaneCodec::D2, PlaneCodec::B3] {
            for planes in [1, 3] {
                for compact in [false, true] {
                    let source = fixture(rows, cols, codec, planes, compact, Basis::Identity);
                    let packed = PackedAdditive::upload(&backend, source.view()).unwrap();
                    for batch in [1, 4] {
                        let host = (0..cols * batch)
                            .map(|i| ((i % 257) as f32 - 128.) / 97.)
                            .collect::<Vec<_>>();
                        let input = backend.stream.clone_htod(&host).unwrap();
                        let mut output = backend.stream.alloc_zeros::<f32>(rows * batch).unwrap();
                        let mut samples = [Vec::new(), Vec::new()];
                        let mut device_samples = [Vec::new(), Vec::new()];
                        let mut oracle = None;
                        for iteration in 0..2 * (sample_count + 2) {
                            let mode = iteration % 2;
                            backend.stream.synchronize().unwrap();
                            let start = Instant::now();
                            let device_us = launch_kernel::<true>(
                                &backend,
                                &packed,
                                &input,
                                &mut output,
                                batch as u32,
                                rows * batch,
                                mode == 1,
                            );
                            backend.stream.synchronize().unwrap();
                            if iteration >= 4 {
                                samples[mode].push(start.elapsed());
                                device_samples[mode]
                                    .push(Duration::from_secs_f64(device_us * 1e-6));
                            }
                            let actual = backend.stream.clone_dtoh(&output).unwrap();
                            let bits = actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                            if let Some(expected) = &oracle {
                                assert_eq!(&bits, expected);
                            } else {
                                oracle = Some(bits);
                            }
                        }
                        let actual = backend.stream.clone_dtoh(&output).unwrap();
                        assert!(actual.iter().all(|v| v.is_finite()));
                        eprintln!(
                            "PACKED_KERNEL rows={rows} cols={cols} codec={codec:?} planes={planes} compact={compact} batch={batch} scalar_us={:.3} warp_us={:.3} scalar_gpu_us={:.3} warp_gpu_us={:.3}",
                            median(samples[0].clone()).as_secs_f64() * 1e6,
                            median(samples[1].clone()).as_secs_f64() * 1e6,
                            median(device_samples[0].clone()).as_secs_f64() * 1e6,
                            median(device_samples[1].clone()).as_secs_f64() * 1e6,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn physical_ordered_warp_matches_scalar_and_core_bits() {
    let backend = CudaBackend::new(0).expect("requires physical CUDA");
    let mut cases = 0;
    for admitted in ADMITTED_LAWS {
        for planes in 1..=admitted.max_planes {
            for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
                let widths: &[usize] = if codec == PlaneCodec::S34 {
                    &[4, 132, 260, 1024]
                } else {
                    &[1, 31, 33, 129, 257, 1024]
                };
                for &cols in widths {
                    for group in [32, 128] {
                        for mode in 0..3 {
                            let rows = 5;
                            let layout = AdditiveLayout {
                                rows: rows as u64,
                                cols: cols as u64,
                                tile: 256,
                                group,
                                max_planes: planes,
                                allocation: if mode == 0 {
                                    PlaneAllocation::Uniform
                                } else {
                                    PlaneAllocation::PerTile
                                },
                                codec,
                                law: admitted.law,
                                basis: Basis::Identity,
                                transport: Transport::Raw,
                            };
                            let values = |n| {
                                [Trit::POS, Trit::NEG, Trit::ZERO, Trit::POS]
                                    .into_iter()
                                    .cycle()
                                    .take(n)
                                    .collect()
                            };
                            let scales =
                                |n| [0.5, 1., 2., 3.].into_iter().cycle().take(n).collect();
                            let source = if mode == 0 {
                                let p = UniformPlan::new(layout, planes).unwrap();
                                AdditiveTensor::new(
                                    layout,
                                    planes,
                                    values(p.trit_count()),
                                    scales(p.scale_count()),
                                )
                                .unwrap()
                            } else {
                                let counts = (0..rows * cols.div_ceil(256))
                                    .map(|i| {
                                        if mode == 2 {
                                            0
                                        } else {
                                            [0, 1, planes, 2, 0][i % 5].min(planes)
                                        }
                                    })
                                    .collect::<Vec<_>>();
                                let p = PerTilePlan::new(layout, &counts).unwrap();
                                let trits = values(p.trit_count());
                                let scales = scales(p.scale_count());
                                AdditiveTensor::new_per_tile(layout, counts, trits, scales).unwrap()
                            };
                            let packed = PackedAdditive::upload(&backend, source.view()).unwrap();
                            for batch in [1, 3] {
                                let host = [1e20, -1e20, 0.25, -0.5, 2., -3., 0., -0.0]
                                    .into_iter()
                                    .cycle()
                                    .take(cols * batch)
                                    .collect::<Vec<_>>();
                                let mut expected = vec![0.; rows * batch];
                                reference_ternary_matmul(
                                    &host,
                                    &source.view().bind_matmul(),
                                    batch,
                                    &mut vec![0.; host.len()],
                                    &mut expected,
                                )
                                .unwrap();
                                let input = backend.stream.clone_htod(&host).unwrap();
                                // One reserved output row remains outside the active prefix.
                                let mut output = backend
                                    .stream
                                    .clone_htod(&vec![77.; rows * batch + rows])
                                    .unwrap();
                                launch_kernel::<false>(
                                    &backend,
                                    &packed,
                                    &input,
                                    &mut output,
                                    batch as u32,
                                    rows * batch,
                                    false,
                                );
                                let scalar = backend.stream.clone_dtoh(&output).unwrap();
                                launch_kernel::<false>(
                                    &backend,
                                    &packed,
                                    &input,
                                    &mut output,
                                    batch as u32,
                                    rows * batch,
                                    true,
                                );
                                let actual = backend.stream.clone_dtoh(&output).unwrap();
                                assert_eq!(
                                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                    scalar.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                    "scalar parity: {layout:?} mode={mode} batch={batch}"
                                );
                                assert_eq!(
                                    actual[..rows * batch]
                                        .iter()
                                        .map(|v| v.to_bits())
                                        .collect::<Vec<_>>(),
                                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                    "core parity: {layout:?} mode={mode} batch={batch}"
                                );
                                assert_eq!(&actual[rows * batch..], &[77.; 5]);
                                cases += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!("ORDERED_WARP_EXACT_CASES={cases}");
    assert_eq!(cases, 1920);
}

struct Source<'a> {
    metadata: AdditiveMetadata<'a>,
    raw: &'a [u8],
}
impl TensorStreamSource for Source<'_> {
    fn metadata(&self) -> TensorMetadata<'_> {
        TensorMetadata::Additive(self.metadata)
    }
    fn encoding(&self) -> TensorStreamEncoding {
        TensorStreamEncoding::CanonicalRawAdditive
    }
    fn encoded_bytes(&self) -> u64 {
        self.raw.len() as u64
    }
    fn visit(
        &self,
        visitor: &mut dyn FnMut(&[u8]) -> Result<(), BackendError>,
    ) -> Result<(), BackendError> {
        for chunk in self.raw.chunks(4093) {
            visitor(chunk)?;
        }
        Ok(())
    }
}

#[test]
fn physical_ordered_dispatch_uses_public_upload_and_stream_seams() {
    let backend = CudaBackend::new(0).expect("requires physical CUDA");
    let mut cases = 0;
    for codec in [PlaneCodec::D2, PlaneCodec::B3] {
        for planes in [1, 3] {
            for compact in [false, true] {
                for basis in [
                    Basis::Identity,
                    Basis::Hadamard { block: 128 },
                    Basis::SignedRht {
                        block: 128,
                        seed: 7,
                        domain: 9,
                    },
                ] {
                    let source = fixture(512, 512, codec, planes, compact, basis);
                    let plan = Plan::new(source.view()).unwrap();
                    assert!(plan.ordered_warp_eligible());
                    let raw = source.encode_payload().unwrap();
                    let stream = Source {
                        metadata: AdditiveMetadata::from_view(source.view()),
                        raw: &raw,
                    };
                    let (ordinary, caps) = backend
                        .upload_tensor_checked(
                            TensorView::Additive(source.view()),
                            tritium_spec::TensorUploadPolicy {
                                native_only: true,
                                max_payload_bytes: Some(plan.bytes as u64),
                            },
                        )
                        .unwrap();
                    let streamed = backend.upload_tensor_stream(&stream).unwrap();
                    assert_eq!(caps.payload_bytes, plan.bytes as u64);
                    assert_eq!(ordinary.len_bytes(), streamed.len_bytes());
                    for tensor in [&*ordinary, &*streamed] {
                        let packed = tensor
                            .as_any()
                            .downcast_ref::<CudaTensor>()
                            .unwrap()
                            .packed
                            .as_ref()
                            .unwrap();
                        assert!(packed.ordered_warp_eligible);
                        for batch in [1, 4] {
                            let act = (0..512 * batch)
                                .map(|i| ((i % 257) as f32 - 128.) / 97.)
                                .collect::<Vec<_>>();
                            let mut expected = vec![0.; 512 * batch];
                            let mut expected_scratch = vec![0.; act.len()];
                            reference_ternary_matmul(
                                &act,
                                &source.view().bind_matmul(),
                                batch,
                                &mut expected_scratch,
                                &mut expected,
                            )
                            .unwrap();
                            let mut actual = vec![99.; expected.len()];
                            let mut scratch = vec![99.; act.len()];
                            backend
                                .matmul(TensorMatmul {
                                    act: &act,
                                    tensor,
                                    batch,
                                    transformed_act: &mut scratch,
                                    out: &mut actual,
                                })
                                .unwrap();
                            assert_eq!(
                                scratch.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                expected_scratch
                                    .iter()
                                    .map(|v| v.to_bits())
                                    .collect::<Vec<_>>()
                            );
                            assert_eq!(
                                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                            );
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    eprintln!("ORDERED_WARP_PUBLIC_CASES={cases}");
    assert_eq!(cases, 96);
}

#[test]
fn ordered_warp_metadata_selection_keeps_unmeasured_families_scalar() {
    let source = fixture(512, 512, PlaneCodec::D2, 3, true, Basis::Identity);
    let mut plan = Plan::new(source.view()).unwrap();
    assert!(plan.ordered_warp_eligible());
    let header = plan.header;
    for (index, value) in [
        (1, 511),
        (1, 17409),
        (3, 32),
        (4, 2),
        (6, 0),
        (7, 1),
        (9, HEADER as u32),
    ] {
        plan.header = header;
        plan.header[index] = value;
        assert!(!plan.ordered_warp_eligible(), "header[{index}]={value}");
    }
}
