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
    assert!(!values.is_empty(), "timing median requires samples");
    values.sort_unstable();
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        // Exact discrete midpoint, rounded down to a nanosecond. Adding the
        // two middle durations directly can overflow even when the result fits.
        values[middle - 1] + (values[middle] - values[middle - 1]) / 2
    } else {
        values[middle]
    }
}

fn timing_series(values: &[Duration]) -> serde_json::Value {
    let nanos =
        |v: Duration| u64::try_from(v.as_nanos()).expect("timing sample exceeds u64 nanoseconds");
    serde_json::json!({
        "sample_count": values.len(),
        "sample_nanoseconds": values.iter().map(|&v| nanos(v)).collect::<Vec<_>>(),
        "median_nanoseconds": nanos(median(values.to_vec())),
        "median_rule": "sorted midpoint, floor to nanoseconds for even sample counts"
    })
}

#[test]
fn timing_statistics_median_uses_even_midpoint_without_overflow() {
    let ns = Duration::from_nanos;
    assert_eq!(median(vec![ns(20), ns(10)]), ns(15));
    assert_eq!(median(vec![ns(7), ns(1), ns(2)]), ns(2));
    assert_eq!(median(vec![ns(9)]), ns(9));
    // Discrete nanosecond midpoint rounds down; sorted subtraction must not
    // overflow even when adding the two middle durations directly would.
    assert_eq!(median(vec![ns(1), ns(0)]), ns(0));
    assert_eq!(median(vec![Duration::MAX, Duration::MAX]), Duration::MAX);
}

#[test]
fn timing_statistics_series_preserves_order_and_replays_summary() {
    let samples = [20, 10, 100, 0].map(Duration::from_nanos);
    let report = timing_series(&samples);
    assert_eq!(report["sample_count"], 4);
    assert_eq!(
        report["sample_nanoseconds"],
        serde_json::json!([20, 10, 100, 0])
    );
    assert_eq!(report["median_nanoseconds"], 15);
    let restored = report["sample_nanoseconds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| Duration::from_nanos(v.as_u64().unwrap()))
        .collect();
    assert_eq!(median(restored), Duration::from_nanos(15));
    assert_eq!(samples, [20, 10, 100, 0].map(Duration::from_nanos));
}

#[test]
#[should_panic(expected = "timing median requires samples")]
fn timing_statistics_empty_samples_refuse() {
    timing_series(&[]);
}

#[test]
#[should_panic(expected = "timing sample exceeds u64 nanoseconds")]
fn timing_statistics_unrepresentable_samples_refuse() {
    timing_series(&[Duration::MAX]);
}

// Direct candidate comparisons bypass production's deliberately narrow dispatch.
fn b3_rows_function(backend: &CudaBackend, tile_rows: usize) -> cudarc::driver::CudaFunction {
    let module = backend
        .stream
        .context()
        .load_module(cudarc::nvrtc::Ptx::from_src(include_str!(concat!(
            env!("OUT_DIR"),
            "/semantic_additive.ptx"
        ))))
        .unwrap();
    module
        .load_function(&format!("semantic_additive_matmul_b3_rows_{tile_rows}"))
        .expect("private cross-row candidate must exist before physical comparison")
}

#[allow(clippy::too_many_arguments)]
fn launch_b3_rows(
    backend: &CudaBackend,
    function: &cudarc::driver::CudaFunction,
    packed: &PackedAdditive,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    rows: usize,
    batch: usize,
    tile_rows: usize,
) {
    let config = LaunchConfig {
        grid_dim: (rows.div_ceil(tile_rows) as u32, batch as u32, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: (128 * 4 + tile_rows * 28) as u32,
    };
    let batch = batch as u32;
    let mut launch = backend.stream.launch_builder(function);
    launch.arg(&packed.bytes).arg(input).arg(output).arg(&batch);
    #[allow(unsafe_code)]
    // SAFETY: the caller supplies canonical Uniform/B3/Free-F16/group128,
    // active output/activation extents, and the candidate's fixed shared layout.
    unsafe { launch.launch(config) }.unwrap();
}

#[test]
#[ignore = "explicit physical cross-row candidate parity; not a release gate"]
fn physical_b3_rows_matches_scalar_and_core_bits() {
    let backend = CudaBackend::new(0).expect("requires physical CUDA");
    let functions = [
        b3_rows_function(&backend, 8),
        b3_rows_function(&backend, 32),
    ];
    let mut cases = 0;
    for rows in [1, 7, 8, 9, 31, 32, 33] {
        for cols in [1, 31, 127, 128, 129, 257, 512] {
            for planes in [1, 3] {
                let mut layout_source =
                    fixture(rows, cols, PlaneCodec::B3, planes, false, Basis::Identity);
                let mut layout = layout_source.view().layout();
                layout.law = ADMITTED_LAWS[0].law;
                layout_source = AdditiveTensor::new(
                    layout,
                    planes,
                    layout_source.view().trits().to_vec(),
                    layout_source.view().scales().to_vec(),
                )
                .unwrap();
                let packed = PackedAdditive::upload(&backend, layout_source.view()).unwrap();
                for batch in [1, 3] {
                    let host = [1e20, -1e20, 0.25, -0.5, 2., -3., 0., -0.0]
                        .into_iter()
                        .cycle()
                        .take(cols * batch)
                        .collect::<Vec<_>>();
                    let mut expected = vec![0.; rows * batch];
                    reference_ternary_matmul(
                        &host,
                        &layout_source.view().bind_matmul(),
                        batch,
                        &mut vec![0.; host.len()],
                        &mut expected,
                    )
                    .unwrap();
                    let input = backend.stream.clone_htod(&host).unwrap();
                    let mut output = backend
                        .stream
                        .clone_htod(&vec![77.; rows * batch + 5])
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
                    for (tile_rows, function) in [8, 32].into_iter().zip(&functions) {
                        launch_b3_rows(
                            &backend,
                            function,
                            &packed,
                            &input,
                            &mut output,
                            rows,
                            batch,
                            tile_rows,
                        );
                        let actual = backend.stream.clone_dtoh(&output).unwrap();
                        assert_eq!(
                            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            scalar.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            "scalar parity rows={rows} cols={cols} planes={planes} batch={batch} tile={tile_rows}"
                        );
                        assert_eq!(
                            actual[..rows * batch]
                                .iter()
                                .map(|v| v.to_bits())
                                .collect::<Vec<_>>(),
                            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            "core parity rows={rows} cols={cols} planes={planes} batch={batch} tile={tile_rows}"
                        );
                        assert_eq!(&actual[rows * batch..], &[77.; 5]);
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 392);
    eprintln!("B3_ROWS_EXACT_CASES={cases}");
}

#[test]
#[ignore = "explicit warm-cache cross-row timing diagnostic; not a performance release gate"]
fn physical_b3_rows_kernel_timing() {
    let backend = CudaBackend::new(0).expect("requires physical CUDA");
    let functions = [
        b3_rows_function(&backend, 8),
        b3_rows_function(&backend, 32),
    ];
    for (rows, cols) in [(5120, 5120), (17408, 5120), (5120, 17408)] {
        for planes in [1, 3] {
            let source = fixture(rows, cols, PlaneCodec::B3, planes, false, Basis::Identity);
            let mut layout = source.view().layout();
            layout.law = ADMITTED_LAWS[0].law;
            let source = AdditiveTensor::new(
                layout,
                planes,
                source.view().trits().to_vec(),
                source.view().scales().to_vec(),
            )
            .unwrap();
            let packed = PackedAdditive::upload(&backend, source.view()).unwrap();
            let host = (0..cols)
                .map(|i| ((i % 257) as f32 - 128.) / 97.)
                .collect::<Vec<_>>();
            let input = backend.stream.clone_htod(&host).unwrap();
            let mut output = backend.stream.alloc_zeros::<f32>(rows).unwrap();
            let mut samples: [Vec<Duration>; 4] = core::array::from_fn(|_| Vec::new());
            launch_kernel::<false>(&backend, &packed, &input, &mut output, 1, rows, false);
            let scalar = backend.stream.clone_dtoh(&output).unwrap();
            let bits = scalar.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert!(scalar.iter().all(|v| v.is_finite()));
            // Balanced rotating/reversed order, two warmups then twelve samples
            // per mode. No cache eviction, clock control or exclusivity claim.
            for round in 0..14 {
                for offset in 0..4 {
                    let mode = (round + if round % 2 == 0 { offset } else { 3 - offset }) % 4;
                    backend.stream.synchronize().unwrap();
                    let start = Instant::now();
                    match mode {
                        0 | 1 => {
                            launch_kernel::<false>(
                                &backend,
                                &packed,
                                &input,
                                &mut output,
                                1,
                                rows,
                                mode == 1,
                            );
                        }
                        2 | 3 => launch_b3_rows(
                            &backend,
                            &functions[mode - 2],
                            &packed,
                            &input,
                            &mut output,
                            rows,
                            1,
                            [8, 32][mode - 2],
                        ),
                        _ => unreachable!(),
                    }
                    backend.stream.synchronize().unwrap();
                    if round >= 2 {
                        samples[mode].push(start.elapsed());
                    }
                    let actual = backend.stream.clone_dtoh(&output).unwrap();
                    assert_eq!(
                        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        bits,
                        "timed parity rows={rows} cols={cols} planes={planes} mode={mode}"
                    );
                }
            }
            eprintln!(
                "B3_ROWS_SAMPLES {}",
                serde_json::json!({
                    "qualified": false, "rows": rows, "cols": cols, "planes": planes, "batch": 1,
                    "scope": "Private warm-cache host launch+synchronize samples; shared hardware, no qualified throughput or full-model speed claim",
                    "scalar": timing_series(&samples[0]), "ordered_warp": timing_series(&samples[1]),
                    "rows8": timing_series(&samples[2]), "rows32": timing_series(&samples[3])
                })
            );
            let us = samples.map(|v| median(v).as_secs_f64() * 1e6);
            eprintln!(
                "B3_ROWS_TIMING rows={rows} cols={cols} planes={planes} batch=1 scalar_us={:.3} warp_us={:.3} tile8_us={:.3} tile32_us={:.3}",
                us[0], us[1], us[2], us[3]
            );
        }
    }
}

#[test]
fn b3_rows_metadata_selection_keeps_unmeasured_families_unchanged() {
    let layout = AdditiveLayout {
        rows: 17408,
        cols: 5120,
        tile: 256,
        group: 128,
        max_planes: 3,
        allocation: PlaneAllocation::Uniform,
        codec: PlaneCodec::B3,
        law: ADMITTED_LAWS[0].law,
        basis: Basis::Identity,
        transport: Transport::Raw,
    };
    for planes in [1, 3] {
        let mut plan =
            Plan::from_metadata(AdditiveMetadata::uniform(layout, planes).unwrap()).unwrap();
        assert!(plan.b3_rows_eligible());
        let header = plan.header;
        for (index, value) in [
            (0, 17409),
            (1, 17408),
            (2, 2),
            (3, 32),
            (4, 0),
            (4, 2),
            (5, 4),
            (6, 0),
            (7, 1),
            (13, 1),
        ] {
            plan.header = header;
            plan.header[index] = value;
            assert!(!plan.b3_rows_eligible(), "header[{index}]={value}");
        }
    }
}

#[test]
#[ignore = "explicit physical measured-family public view/stream parity; not full-model qualification"]
fn physical_b3_rows_dispatch_uses_public_upload_and_stream_seams() {
    let backend = CudaBackend::new(0).expect("requires physical CUDA");
    let (rows, cols) = (17408, 5120);
    let mut cases = 0;
    for planes in [1, 3] {
        for basis in [
            Basis::Identity,
            Basis::SignedRht {
                block: 128,
                seed: 7,
                domain: 9,
            },
        ] {
            let source = fixture(rows, cols, PlaneCodec::B3, planes, false, basis);
            let mut layout = source.view().layout();
            layout.law = ADMITTED_LAWS[0].law;
            let source = AdditiveTensor::new(
                layout,
                planes,
                source.view().trits().to_vec(),
                source.view().scales().to_vec(),
            )
            .unwrap();
            let plan = Plan::new(source.view()).unwrap();
            assert!(plan.b3_rows_eligible());
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
                assert!(packed.b3_rows_eligible);
                for batch in [1, 2] {
                    let host = (0..cols * batch)
                        .map(|i| ((i % 257) as f32 - 128.) / 97.)
                        .collect::<Vec<_>>();
                    let mut expected_scratch = host.clone();
                    for row in expected_scratch.chunks_exact_mut(cols) {
                        tritium_core::apply_basis(row, basis).unwrap();
                    }
                    let input = backend.stream.clone_htod(&expected_scratch).unwrap();
                    let mut output = backend.stream.alloc_zeros::<f32>(rows * batch).unwrap();
                    launch_kernel::<false>(
                        &backend,
                        packed,
                        &input,
                        &mut output,
                        batch as u32,
                        rows * batch,
                        false,
                    );
                    let expected = backend.stream.clone_dtoh(&output).unwrap();
                    let mut actual = vec![99.; rows * batch];
                    let mut scratch = vec![99.; host.len()];
                    backend
                        .matmul(TensorMatmul {
                            act: &host,
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
                        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "public dispatch parity planes={planes} basis={basis:?} batch={batch}"
                    );
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 16);
    eprintln!("B3_ROWS_PUBLIC_CASES={cases}");
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
                            "PACKED_KERNEL_SAMPLES {}",
                            serde_json::json!({
                                "qualified": false, "rows": rows, "cols": cols, "codec": format!("{codec:?}"),
                                "planes": planes, "compact": compact, "batch": batch,
                                "scope": "Private packed-kernel diagnostic; host and device-event series are distinct, no release/performance qualification",
                                "scalar_host": timing_series(&samples[0]), "warp_host": timing_series(&samples[1]),
                                "scalar_device": timing_series(&device_samples[0]), "warp_device": timing_series(&device_samples[1])
                            })
                        );
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
