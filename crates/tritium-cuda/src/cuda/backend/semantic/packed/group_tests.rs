//! Ordered-group parity, narrow selection, and fixed pre-candidate comparisons.
use super::*;

fn source(rows: usize, cols: usize, planes: u8) -> AdditiveTensor {
    let original = fixture(rows, cols, PlaneCodec::B3, planes, false, Basis::Identity);
    let mut layout = original.view().layout();
    layout.law = ADMITTED_LAWS[0].law;
    let scales = (0..original.view().scales().len())
        .map(|i| f16::from_bits([0x0001, 0x0400, 0x2e66, 0x3800, 0x4200][i % 5]).to_f32())
        .collect();
    AdditiveTensor::new(layout, planes, original.view().trits().to_vec(), scales).unwrap()
}

fn function(backend: &CudaBackend) -> cudarc::driver::CudaFunction {
    backend
        .stream
        .context()
        .load_module(cudarc::nvrtc::Ptx::from_src(include_str!(concat!(
            env!("OUT_DIR"),
            "/semantic_additive.ptx"
        ))))
        .unwrap()
        .load_function("semantic_additive_matmul_b3_groups")
        .expect("private group candidate must exist before physical comparison")
}

fn launch(
    backend: &CudaBackend,
    function: &cudarc::driver::CudaFunction,
    packed: &PackedAdditive,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    rows: u32,
    batch: u32,
) {
    let mut builder = backend.stream.launch_builder(function);
    builder
        .arg(&packed.bytes)
        .arg(input)
        .arg(output)
        .arg(&batch);
    #[allow(unsafe_code)]
    // SAFETY: this private diagnostic supplies canonical Uniform/B3/Free-F16,
    // group128, 1..=3 planes, <=17408 columns and correctly bounded buffers.
    // One whole CTA owns each output; 512 shared floats cover <=408 groups.
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (rows, batch, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 512 * 4,
        })
    }
    .unwrap();
}

#[test]
fn b3_groups_metadata_selection_keeps_unmeasured_families_unchanged() {
    for (rows, cols, planes) in [
        (5120, 5120, 1),
        (5120, 5120, 3),
        (5120, 17408, 1),
        (5120, 17408, 3),
        (17408, 5120, 3),
        (48, 5120, 1),
        (48, 5120, 3),
        (1024, 5120, 3),
        (5120, 6144, 1),
        (5120, 6144, 3),
        (6144, 5120, 1),
        (6144, 5120, 3),
        (10240, 5120, 1),
        (10240, 5120, 3),
        (12288, 5120, 1),
        (12288, 5120, 3),
    ] {
        let layout = AdditiveLayout {
            rows,
            cols,
            tile: 256,
            group: 128,
            max_planes: planes,
            allocation: PlaneAllocation::Uniform,
            codec: PlaneCodec::B3,
            law: ADMITTED_LAWS[0].law,
            basis: Basis::Identity,
            transport: Transport::Raw,
        };
        let mut plan =
            Plan::from_metadata(AdditiveMetadata::uniform(layout, planes).unwrap()).unwrap();
        assert!(plan.b3_groups_eligible());
        let header = plan.header;
        for (index, value) in [
            (0, 5121),
            (1, 5119),
            (2, 2),
            (2, 4),
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
            assert!(!plan.b3_groups_eligible(), "header[{index}]={value}");
        }
    }
    let layout = AdditiveLayout {
        rows: 17408,
        cols: 5120,
        tile: 256,
        group: 128,
        max_planes: 1,
        allocation: PlaneAllocation::Uniform,
        codec: PlaneCodec::B3,
        law: ADMITTED_LAWS[0].law,
        basis: Basis::Identity,
        transport: Transport::Raw,
    };
    let plan = Plan::from_metadata(AdditiveMetadata::uniform(layout, 1).unwrap()).unwrap();
    assert!(!plan.b3_groups_eligible());
    assert!(plan.b3_rows_eligible());
    let layout = AdditiveLayout {
        rows: 1024,
        ..layout
    };
    let plan = Plan::from_metadata(AdditiveMetadata::uniform(layout, 1).unwrap()).unwrap();
    assert!(!plan.b3_groups_eligible());
    assert!(plan.ordered_warp_eligible());
}

#[test]
#[ignore = "explicit physical ordered-group public view/stream/basis/fallback parity; not model qualification"]
fn physical_b3_groups_public_upload_and_stream_seams() {
    let backend = CudaBackend::new(0).expect("requires physical CUDA");
    let mut cases = 0;
    for (rows, cols) in [
        (5120, 5120),
        (17408, 5120),
        (5120, 17408),
        (48, 5120),
        (1024, 5120),
        (5120, 6144),
        (6144, 5120),
        (10240, 5120),
        (12288, 5120),
    ] {
        for planes in [1, 3] {
            let original = source(rows, cols, planes);
            for basis in [
                Basis::Identity,
                Basis::SignedRht {
                    block: 128,
                    seed: 7,
                    domain: 9,
                },
            ] {
                let mut layout = original.view().layout();
                layout.basis = basis;
                let source = AdditiveTensor::new(
                    layout,
                    planes,
                    original.view().trits().to_vec(),
                    original.view().scales().to_vec(),
                )
                .unwrap();
                let plan = Plan::new(source.view()).unwrap();
                let selected = !(matches!(rows, 1024 | 17408) && cols == 5120 && planes == 1);
                assert_eq!(plan.b3_groups_eligible(), selected);
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
                    assert_eq!(packed.b3_groups_eligible, selected);
                    for batch in [1, 2] {
                        let host = (0..cols * batch)
                            .map(|i| ((i % 257) as f32 - 128.) / 97.)
                            .collect::<Vec<_>>();
                        let mut transformed = host.clone();
                        for row in transformed.chunks_exact_mut(cols) {
                            tritium_core::apply_basis(row, basis).unwrap();
                        }
                        let input = backend.stream.clone_htod(&transformed).unwrap();
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
                            transformed.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                        );
                        assert_eq!(
                            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            "public parity rows={rows} cols={cols} planes={planes} basis={basis:?} batch={batch}"
                        );
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 144);
    eprintln!("B3_GROUP_PUBLIC_CASES={cases}");
}

#[test]
#[ignore = "explicit physical ordered-group parity; not model/release qualification"]
fn physical_b3_groups_matches_scalar_and_core_bits() {
    let backend = CudaBackend::new(0).expect("requires physical CUDA");
    let function = function(&backend);
    let mut cases = 0;
    for rows in [1, 3, 7, 33] {
        for cols in [1, 31, 127, 128, 129, 257, 511, 512, 513, 17407, 17408] {
            for planes in [1, 2, 3] {
                let source = source(rows, cols, planes);
                let packed = PackedAdditive::upload(&backend, source.view()).unwrap();
                for batch in [1, 2] {
                    let host = [1e20, -1e20, 0.25, -0.5, 2., -3., 0., -0., 1e-40, -1e-40]
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
                    launch(
                        &backend,
                        &function,
                        &packed,
                        &input,
                        &mut output,
                        rows as u32,
                        batch as u32,
                    );
                    let actual = backend.stream.clone_dtoh(&output).unwrap();
                    assert_eq!(
                        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        scalar.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "scalar parity rows={rows} cols={cols} planes={planes} batch={batch}"
                    );
                    assert_eq!(
                        actual[..rows * batch]
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "core parity rows={rows} cols={cols} planes={planes} batch={batch}"
                    );
                    assert_eq!(&actual[rows * batch..], &[77.; 5]);
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 264);
    eprintln!("B3_GROUP_EXACT_CASES={cases}");
}

#[test]
#[ignore = "explicit paired warm-cache group candidate timing; not a performance release gate"]
fn physical_b3_groups_kernel_timing() {
    let backend = CudaBackend::new(0).expect("requires physical CUDA");
    let function = function(&backend);
    let rows32 = b3_rows_function(&backend, 32);
    for (rows, cols) in [(5120, 5120), (17408, 5120), (5120, 17408)] {
        for planes in [1, 3] {
            let source = source(rows, cols, planes);
            let packed = PackedAdditive::upload(&backend, source.view()).unwrap();
            let host = (0..cols)
                .map(|i| ((i % 257) as f32 - 128.) / 97.)
                .collect::<Vec<_>>();
            let input = backend.stream.clone_htod(&host).unwrap();
            let mut output = backend.stream.alloc_zeros::<f32>(rows).unwrap();
            launch_kernel::<false>(&backend, &packed, &input, &mut output, 1, rows, false);
            let scalar = backend.stream.clone_dtoh(&output).unwrap();
            assert!(scalar.iter().all(|v| v.is_finite()));
            let bits = scalar.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            let mut samples: [Vec<Duration>; 2] = core::array::from_fn(|_| Vec::new());
            // Two warmup rounds, then twelve balanced alternating pairs.
            // Same input/weights/output and the fixed pre-candidate dispatch
            // control from 21b1ad95. Do not call PackedAdditive::launch here:
            // enabling a measured winner later must not change the control.
            // These are host launch+synchronize samples on shared hardware, not
            // CUDA-event-only durations, cold-cache or whole-model throughput.
            for round in 0..14 {
                for offset in 0..2 {
                    let mode = (round + offset) % 2;
                    backend.stream.synchronize().unwrap();
                    let start = Instant::now();
                    if mode == 0 {
                        if packed.b3_rows_eligible {
                            launch_b3_rows(
                                &backend,
                                &rows32,
                                &packed,
                                &input,
                                &mut output,
                                rows,
                                1,
                                32,
                            );
                        } else {
                            let cooperative =
                                packed.ordered_warp_eligible && (512..=8192).contains(&rows);
                            launch_kernel::<false>(
                                &backend,
                                &packed,
                                &input,
                                &mut output,
                                1,
                                rows,
                                cooperative,
                            );
                        }
                    } else {
                        launch(
                            &backend,
                            &function,
                            &packed,
                            &input,
                            &mut output,
                            rows as u32,
                            1,
                        );
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
                "B3_GROUP_SAMPLES {}",
                serde_json::json!({
                    "qualified": false, "rows": rows, "cols": cols, "planes": planes, "batch": 1,
                    "scope": "Private paired warm-cache host launch+synchronize observations; shared hardware, no qualified throughput or full-model speed claim",
                "baseline_source_revision": "21b1ad95faac87341d24470483cf53d1634edbb3",
                "baseline_dispatch": timing_series(&samples[0]), "groups128": timing_series(&samples[1])
                })
            );
        }
    }
}
