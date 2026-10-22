//! Packed CPU backward through real direct/stream-uploaded semantic handles.
use tritium_core::{
    AdditiveView, GemmShape, PerTilePlan, PerTileRowSpan, TernaryFormat, Trit, UniformPlan,
    reference_ternary_matmul_input_vjp,
};
use tritium_cpu::CpuBackend;
use tritium_format::AdditiveTensor;
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, Transport,
};
use tritium_spec::{
    AdditiveMetadata, BackendError, DeviceBuffer, TensorMetadata, TensorStreamEncoding, TensorView,
    TernaryBackend,
};
#[path = "common/stream_source.rs"]
mod stream_source;

fn layout(
    law: tritium_schema::AdmittedLaw,
    codec: PlaneCodec,
    group: u16,
    cols: usize,
    basis: Basis,
) -> AdditiveLayout {
    AdditiveLayout {
        rows: 3,
        cols: cols as u64,
        tile: 256,
        group,
        max_planes: law.max_planes,
        allocation: PlaneAllocation::Uniform,
        codec,
        law: law.law,
        basis,
        transport: Transport::Raw,
    }
}

fn trits(count: usize, codec: PlaneCodec) -> Vec<Trit> {
    (0..count)
        .map(|i| {
            if codec == PlaneCodec::S34 {
                [Trit::ZERO, Trit::POS, Trit::NEG, Trit::POS][i % 4]
            } else {
                Trit::from_i8(((i * 7 + i / 11) % 3) as i8 - 1).unwrap()
            }
        })
        .collect()
}

fn scales(count: usize) -> Vec<f32> {
    (0..count).map(|i| [0.125, 0.5, 1.0, 0.0][i % 4]).collect()
}

fn check(view: AdditiveView<'_>) -> usize {
    let cpu = CpuBackend::new();
    let rows = view.layout().rows as usize;
    let cols = view.layout().cols as usize;
    let encoded = AdditiveTensor::from_view(view)
        .unwrap()
        .encode_payload()
        .unwrap();
    let source = stream_source::Source::new(
        TensorMetadata::Additive(AdditiveMetadata::from_view(view)),
        TensorStreamEncoding::CanonicalRawAdditive,
        &encoded,
    );
    let direct = cpu.upload_tensor(TensorView::Additive(view)).unwrap();
    let streamed = cpu.upload_tensor_stream(&source).unwrap();
    assert_eq!(source.calls.get(), 1);
    assert_eq!(direct.len_bytes(), streamed.len_bytes());
    for uploaded in [direct, streamed] {
        let bytes = uploaded.len_bytes();
        for batch in [0, 1, 3] {
            let dy: Vec<f32> = (0..batch * rows)
                .map(|i| (i % 5) as f32 * 0.25 - 0.5)
                .collect();
            let mut expected_scratch = vec![17.0; cols];
            let mut expected = vec![19.0; batch * cols];
            reference_ternary_matmul_input_vjp(
                &dy,
                &view.bind_matmul(),
                batch,
                &mut expected_scratch,
                &mut expected,
            )
            .unwrap();
            let mut scratch = vec![17.0f32; cols];
            let mut actual = vec![19.0f32; batch * cols];
            cpu.additive_matmul_input_vjp(&*uploaded, &dy, batch, &mut scratch, &mut actual)
                .unwrap();
            assert_eq!(
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(scratch, expected_scratch);
            assert_eq!(uploaded.len_bytes(), bytes);
        }
    }
    6
}

#[test]
fn packed_uniform_codecs_laws_and_bases_match_shared_adjoint_bits() {
    let mut cases = 0;
    for &law in ADMITTED_LAWS {
        for group in [32, 128] {
            for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
                for cols in [257, 260] {
                    if codec == PlaneCodec::S34 && cols == 257 {
                        continue;
                    }
                    let mut bases = vec![Basis::Identity];
                    if cols % 4 == 0 {
                        bases.extend([
                            Basis::Hadamard { block: 4 },
                            Basis::SignedRht {
                                block: 4,
                                seed: 19,
                                domain: 7,
                            },
                        ]);
                    }
                    for basis in bases {
                        for planes in 1..=law.max_planes {
                            let layout = layout(law, codec, group, cols, basis);
                            let plan = UniformPlan::new(layout, planes).unwrap();
                            let trits = trits(plan.trit_count(), codec);
                            let scales = scales(plan.scale_count());
                            cases +=
                                check(AdditiveView::new(layout, planes, &trits, &scales).unwrap());
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases, 1320);
}

#[test]
fn compact_and_all_pruned_packed_handles_match_shared_adjoint_bits() {
    let mut cases = 0;
    for &law in ADMITTED_LAWS {
        for group in [32, 128] {
            for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
                for basis in [
                    Basis::Identity,
                    Basis::Hadamard { block: 4 },
                    Basis::SignedRht {
                        block: 4,
                        seed: 19,
                        domain: 7,
                    },
                ] {
                    let layout = AdditiveLayout {
                        allocation: PlaneAllocation::PerTile,
                        ..layout(law, codec, group, 260, basis)
                    };
                    for counts in [[0, law.max_planes, 1, 0, law.max_planes, 1], [0; 6]] {
                        let plan = PerTilePlan::new(layout, &counts).unwrap();
                        let mut spans = vec![PerTileRowSpan::default(); plan.index_len()];
                        let index = plan.build(&mut spans).unwrap();
                        let trits = trits(plan.trit_count(), codec);
                        let scales = scales(plan.scale_count());
                        cases += check(AdditiveView::new_per_tile(index, &trits, &scales).unwrap());
                    }
                }
            }
        }
    }
    assert_eq!(cases, 864);
}

#[test]
fn invalid_shapes_and_product_overflow_preserve_caller_buffers() {
    let cpu = CpuBackend::new();
    let meta = layout(ADMITTED_LAWS[0], PlaneCodec::D2, 32, 4, Basis::Identity);
    let view = AdditiveView::new(meta, 1, &[Trit::POS; 12], &[1.0; 3]).unwrap();
    let tensor = cpu.upload_tensor(TensorView::Additive(view)).unwrap();
    for (batch, gradients, workspace, output) in [
        (1, 2, 4, 4),
        (1, 3, 3, 4),
        (1, 3, 4, 3),
        (0, 1, 4, 0),
        (0, 0, 0, 0),
        (usize::MAX, 0, 4, 0),
        (usize::MAX / 3, 0, 4, 0),
    ] {
        let dy = vec![0.25; gradients];
        let mut workspace = vec![17.0; workspace];
        let mut output = vec![19.0; output];
        assert!(matches!(
            cpu.additive_matmul_input_vjp(&*tensor, &dy, batch, &mut workspace, &mut output),
            Err(BackendError::InvalidInput(_))
        ));
        assert!(workspace.iter().all(|&v| v == 17.0));
        assert!(output.iter().all(|&v| v == 19.0));
    }
}

#[derive(Debug)]
struct Foreign;
impl DeviceBuffer for Foreign {
    fn len_bytes(&self) -> usize {
        0
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[test]
fn dense_legacy_and_foreign_handles_reject_without_fallback_or_mutation() {
    let cpu = CpuBackend::new();
    let dense = cpu
        .upload_tensor(TensorView::Dense {
            rows: 1,
            cols: 4,
            values: &[1.0; 4],
        })
        .unwrap();
    let mut legacy_bytes = vec![0; tritium_format::TQ2_0_BLOCK_BYTES];
    tritium_format::pack_tq2_0_row(&[Trit::POS; 4], &[half::f16::ONE], &mut legacy_bytes).unwrap();
    let legacy = cpu
        .upload_weights(
            &legacy_bytes,
            GemmShape { m: 1, n: 1, k: 4 },
            TernaryFormat::Tq2_0,
        )
        .unwrap();
    for (index, tensor) in [&*dense, &*legacy, &Foreign as &dyn DeviceBuffer]
        .into_iter()
        .enumerate()
    {
        let mut workspace = [17.0; 4];
        let mut output = [19.0; 4];
        let error = cpu
            .additive_matmul_input_vjp(tensor, &[0.25], 1, &mut workspace, &mut output)
            .unwrap_err();
        if index == 0 {
            assert!(matches!(error, BackendError::UnsupportedTensor));
        } else {
            assert!(matches!(error, BackendError::InvalidInput(_)));
        }
        assert_eq!(workspace, [17.0; 4]);
        assert_eq!(output, [19.0; 4]);
    }
}

#[test]
fn shared_immutable_handle_supports_concurrent_input_adjoints() {
    let cpu = CpuBackend::new();
    let meta = layout(
        ADMITTED_LAWS[1],
        PlaneCodec::B3,
        128,
        260,
        Basis::SignedRht {
            block: 4,
            seed: 19,
            domain: 7,
        },
    );
    let plan = UniformPlan::new(meta, 3).unwrap();
    let trits = trits(plan.trit_count(), meta.codec);
    let scales = scales(plan.scale_count());
    let view = AdditiveView::new(meta, 3, &trits, &scales).unwrap();
    let handle = cpu.upload_tensor(TensorView::Additive(view)).unwrap();
    let mut expected = vec![0.0; 260];
    reference_ternary_matmul_input_vjp(
        &[0.25, -0.5, 0.75],
        &view.bind_matmul(),
        1,
        &mut vec![0.0; 260],
        &mut expected,
    )
    .unwrap();
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let handle = &*handle;
            let expected = &expected;
            scope.spawn(move || {
                let mut scratch = vec![0.0; 260];
                let mut actual = vec![0.0; 260];
                CpuBackend::new()
                    .additive_matmul_input_vjp(
                        handle,
                        &[0.25, -0.5, 0.75],
                        1,
                        &mut scratch,
                        &mut actual,
                    )
                    .unwrap();
                assert_eq!(actual, *expected);
            });
        }
    });
}

#[test]
fn admitted_stream_handle_preserves_payload_lease_and_upload_evidence() {
    use tritium_evidence::{
        EvidenceRecorder, TENSOR_UPLOADED_REGISTRATION, verify_jsonl_with_registry,
    };
    use tritium_runtime::{
        TensorPayloadBudget, TensorUploadContext, TensorUploadResources,
        upload_tensor_stream_logged_with_resources,
    };
    let cpu = CpuBackend::new();
    let fixture = tritium_testkit::additive_upload_fixture();
    let view = fixture.view();
    let rows = view.layout().rows as usize;
    let cols = view.layout().cols as usize;
    let bytes = fixture.encode_payload().unwrap();
    let source = stream_source::Source::new(
        TensorMetadata::Additive(AdditiveMetadata::from_view(view)),
        TensorStreamEncoding::CanonicalRawAdditive,
        &bytes,
    );
    let payload = TensorPayloadBudget::new(1024);
    let workspace = TensorPayloadBudget::new(tritium_spec::TENSOR_STREAM_CHUNK_BYTES as u64);
    let mut recorder =
        EvidenceRecorder::new("packed-input-adjoint", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
    let (handle, caps) = upload_tensor_stream_logged_with_resources(
        &cpu,
        &source,
        tritium_schema::TensorUploadPolicy {
            native_only: true,
            max_payload_bytes: Some(1024),
        },
        "owner",
        TensorUploadContext {
            recorder: &mut recorder,
            span: "upload",
            parent: None,
            logical_time: 0,
        },
        TensorUploadResources {
            payload: &payload,
            host_workspace: &workspace,
            device_workspace: &workspace,
        },
    )
    .unwrap();
    assert_eq!(payload.used_bytes(), caps.payload_bytes);
    assert_eq!(workspace.used_bytes(), 0);
    let before = recorder.to_jsonl().unwrap();
    let dy = vec![0.25; rows];
    let mut expected = vec![0.0; cols];
    reference_ternary_matmul_input_vjp(
        &dy,
        &view.bind_matmul(),
        1,
        &mut vec![0.0; cols],
        &mut expected,
    )
    .unwrap();
    let mut actual = vec![0.0; cols];
    cpu.additive_matmul_input_vjp(&*handle, &dy, 1, &mut vec![0.0; cols], &mut actual)
        .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(payload.used_bytes(), caps.payload_bytes);
    assert_eq!(recorder.to_jsonl().unwrap(), before);
    verify_jsonl_with_registry(&before, [TENSOR_UPLOADED_REGISTRATION]).unwrap();
    drop(handle);
    assert_eq!(payload.used_bytes(), 0);
}
