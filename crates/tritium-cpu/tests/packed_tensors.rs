//! Packed residency and exact arithmetic through the production tensor seam.

use tritium_core::{AdditiveView, Trit, reference_embed, reference_ternary_matmul};
use tritium_cpu::CpuBackend;
use tritium_format::AdditiveTensor;
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, PlaneRelation, ScaleAnchor,
    TensorUploadPolicy, Transport,
};
use tritium_spec::{BackendError, TensorExecution, TensorMatmul, TensorView, TernaryBackend};
#[path = "common/stream_source.rs"]
mod stream_source;

fn check(codec: PlaneCodec, rows: usize, cols: usize, plane: &[Trit]) -> usize {
    let cpu = CpuBackend::new();
    let mut cases = 0;
    for admitted in ADMITTED_LAWS {
        for group in [32, 128] {
            for planes in 1..=admitted.max_planes {
                let mut bases = vec![
                    Basis::Identity,
                    Basis::Hadamard { block: 1 },
                    Basis::SignedRht {
                        block: 1,
                        seed: 37,
                        domain: 91,
                    },
                ];
                for block in [4, 32, 128, 256] {
                    if cols.is_multiple_of(block) {
                        bases.extend([
                            Basis::Hadamard {
                                block: block as u32,
                            },
                            Basis::SignedRht {
                                block: block as u32,
                                seed: 37,
                                domain: 91,
                            },
                        ]);
                    }
                }
                for basis in bases {
                    let layout = AdditiveLayout {
                        rows: rows as u64,
                        cols: cols as u64,
                        tile: 256,
                        group,
                        max_planes: planes,
                        allocation: PlaneAllocation::Uniform,
                        codec,
                        law: admitted.law,
                        basis,
                        transport: Transport::Raw,
                    };
                    let units = if admitted.law.anchor == ScaleAnchor::Tensor {
                        1
                    } else {
                        rows * cols.div_ceil(usize::from(group))
                    };
                    let count = if matches!(admitted.law.relation, PlaneRelation::Free) {
                        units * usize::from(planes)
                    } else {
                        units
                    };
                    let scales: Vec<_> = (0..count).map(|i| [0.5, 2., 3., 0.][i % 4]).collect();
                    let trits: Vec<_> = (0..planes)
                        .flat_map(|p| {
                            plane.iter().map(move |trit| {
                                if p % 2 == 0 {
                                    *trit
                                } else {
                                    Trit::from_i8(-trit.get()).unwrap()
                                }
                            })
                        })
                        .collect();
                    let view = AdditiveView::new(layout, planes, &trits, &scales).unwrap();
                    let tensor = TensorView::Additive(view);
                    let caps = cpu.tensor_caps(tensor).unwrap().unwrap();
                    let metadata = tritium_spec::TensorMetadata::Additive(
                        tritium_spec::AdditiveMetadata::uniform(layout, planes).unwrap(),
                    );
                    let planned = cpu.tensor_upload_plan(metadata).unwrap().unwrap();
                    assert_eq!(planned.caps, caps);
                    assert_eq!(planned.host_workspace_bytes, 0);
                    assert_eq!(planned.device_workspace_bytes, 0);
                    assert_eq!(caps.execution, TensorExecution::Native);
                    let packed = AdditiveTensor::from_view(view)
                        .unwrap()
                        .encode_payload()
                        .unwrap();
                    // Both codec planes and exact scale bits retain canonical storage precision.
                    let expected_bytes = packed.len();
                    assert_eq!(caps.payload_bytes, expected_bytes as u64);
                    assert!(expected_bytes < trits.len() + 4 * count);
                    let exact = TensorUploadPolicy {
                        native_only: true,
                        max_payload_bytes: Some(expected_bytes as u64),
                    };
                    assert!(matches!(
                        cpu.upload_tensor_checked(
                            tensor,
                            TensorUploadPolicy {
                                max_payload_bytes: Some(expected_bytes as u64 - 1),
                                ..exact
                            }
                        ),
                        Err(BackendError::TensorPayloadBudgetExceeded { .. })
                    ));
                    let (uploaded, actual) = cpu.upload_tensor_checked(tensor, exact).unwrap();
                    assert_eq!(actual, caps);
                    assert_eq!(uploaded.len_bytes(), expected_bytes);
                    let act: Vec<_> = (0..3 * cols)
                        .map(|i| [1., -0.125, 3.25, -7., 0.][i % 5])
                        .collect();
                    let mut scratch = vec![0.; act.len()];
                    let mut expected_scratch = scratch.clone();
                    let mut out = vec![0.; 3 * rows];
                    let mut expected = out.clone();
                    reference_ternary_matmul(
                        &act,
                        &view.bind_matmul(),
                        3,
                        &mut expected_scratch,
                        &mut expected,
                    )
                    .unwrap();
                    cpu.matmul(TensorMatmul {
                        act: &act,
                        tensor: &*uploaded,
                        batch: 3,
                        transformed_act: &mut scratch,
                        out: &mut out,
                    })
                    .unwrap();
                    assert_eq!(
                        out.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                    );
                    assert_eq!(scratch, expected_scratch);
                    let source = stream_source::Source::new(
                        metadata,
                        tritium_spec::TensorStreamEncoding::CanonicalRawAdditive,
                        &packed,
                    );
                    let stream_plan = cpu
                        .tensor_stream_plan(metadata, source.encoding)
                        .unwrap()
                        .unwrap();
                    assert_eq!(stream_plan.upload.caps, caps);
                    assert_eq!(stream_plan.encoded_bytes, packed.len() as u64);
                    let streamed = cpu.upload_tensor_stream(&source).unwrap();
                    assert_eq!(streamed.len_bytes(), expected_bytes);
                    cpu.matmul(TensorMatmul {
                        act: &act,
                        tensor: &*streamed,
                        batch: 3,
                        transformed_act: &mut scratch,
                        out: &mut out,
                    })
                    .unwrap();
                    assert_eq!(out, expected);
                    assert_eq!(scratch, expected_scratch);
                    let ids = [rows - 1, 0, rows - 1];
                    let mut gathered = vec![0.; ids.len() * cols];
                    let mut expected_rows = gathered.clone();
                    reference_embed(&view.bind_gather(), &ids, &mut expected_rows).unwrap();
                    cpu.embed_rows(&*uploaded, &ids, &mut gathered).unwrap();
                    cpu.embed_rows(&*streamed, &ids, &mut gathered).unwrap();
                    assert_eq!(
                        gathered.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        expected_rows
                            .iter()
                            .map(|x| x.to_bits())
                            .collect::<Vec<_>>()
                    );
                    cases += 1;
                }
            }
        }
    }
    cases
}

#[test]
fn every_codec_symbol_executes_without_decoded_residency() {
    let mut cases = 0;
    for code in 0..243 {
        let mut n = code;
        let digits: Vec<_> = (0..5)
            .map(|_| {
                let digit = Trit::from_i8((n % 3) as i8 - 1).unwrap();
                n /= 3;
                digit
            })
            .collect();
        // Odd row boundaries cross D2 bytes and B3 radix bytes.
        let plane: Vec<_> = digits.iter().copied().cycle().take(9).collect();
        cases += check(PlaneCodec::D2, 3, 3, &plane);
        cases += check(PlaneCodec::B3, 3, 3, &plane);
    }
    for code in 0..32 {
        let zero = code & 3;
        let mut sign = 0;
        let digits: Vec<_> = (0..4)
            .map(|slot| {
                if slot == zero {
                    Trit::ZERO
                } else {
                    let trit = if code & (1 << (2 + sign)) == 0 {
                        Trit::NEG
                    } else {
                        Trit::POS
                    };
                    sign += 1;
                    trit
                }
            })
            .collect();
        // S34 code groups cross byte boundaries and ragged row boundaries.
        let plane: Vec<_> = digits.iter().copied().cycle().take(12).collect();
        cases += check(PlaneCodec::S34, 4, 3, &plane);
    }
    assert_eq!(cases, 31_080);
}

#[test]
fn packed_rows_span_groups_and_planes_with_canonical_tails() {
    for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
        let cols = if codec == PlaneCodec::S34 { 259 } else { 263 };
        let rows = 4;
        let plane: Vec<_> = [Trit::POS, Trit::ZERO, Trit::NEG, Trit::POS]
            .into_iter()
            .cycle()
            .take(rows * cols)
            .collect();
        assert_eq!(check(codec, rows, cols, &plane), 60);
    }
}

#[test]
fn invalid_s34_content_is_rejected_before_admission() {
    let cpu = CpuBackend::new();
    let layout = AdditiveLayout {
        rows: 1,
        cols: 4,
        tile: 256,
        group: 32,
        max_planes: 1,
        allocation: PlaneAllocation::Uniform,
        codec: PlaneCodec::S34,
        law: ADMITTED_LAWS[1].law,
        basis: Basis::Identity,
        transport: Transport::Raw,
    };
    let view = TensorView::Additive(AdditiveView::new(layout, 1, &[Trit::POS; 4], &[1.]).unwrap());
    assert!(cpu.tensor_caps(view).is_err());
    assert!(cpu.upload_tensor(view).is_err());
    assert!(
        cpu.upload_tensor_checked(view, TensorUploadPolicy::default())
            .is_err()
    );
}

#[test]
fn rotations_and_parallel_rows_preserve_exact_reference_bits() {
    for threads in [1, 4] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
                    let plane: Vec<_> = [Trit::POS, Trit::NEG, Trit::ZERO, Trit::POS]
                        .into_iter()
                        .cycle()
                        .take(37 * 256)
                        .collect();
                    assert_eq!(check(codec, 37, 256, &plane), 220);
                }
            });
    }
}

#[test]
fn packed_calls_reject_shapes_and_all_bad_ids_before_mutation() {
    let cpu = CpuBackend::new();
    let fixture = tritium_testkit::additive_upload_fixture();
    let tensor = cpu
        .upload_tensor(TensorView::Additive(fixture.view()))
        .unwrap();
    let mut out = [99.; 2];
    let mut scratch = [99.; 4];
    assert!(
        cpu.matmul(TensorMatmul {
            act: &[1.; 3],
            tensor: &*tensor,
            batch: 1,
            transformed_act: &mut scratch,
            out: &mut out
        })
        .is_err()
    );
    assert_eq!(out, [99.; 2]);
    assert_eq!(scratch, [99.; 4]);
    assert!(
        cpu.matmul(TensorMatmul {
            act: &[],
            tensor: &*tensor,
            batch: usize::MAX,
            transformed_act: &mut scratch,
            out: &mut out
        })
        .is_err()
    );
    assert_eq!(out, [99.; 2]);
    assert_eq!(scratch, [99.; 4]);
    let mut gathered = [99.; 8];
    assert!(cpu.embed_rows(&*tensor, &[0, 2], &mut gathered).is_err());
    assert_eq!(gathered, [99.; 8]);
    assert!(cpu.embed_rows(&*tensor, &[0], &mut gathered).is_err());
    assert_eq!(gathered, [99.; 8]);
    cpu.embed_rows(&*tensor, &[], &mut []).unwrap();
    cpu.matmul(TensorMatmul {
        act: &[],
        tensor: &*tensor,
        batch: 0,
        transformed_act: &mut [],
        out: &mut [],
    })
    .unwrap();
}

#[test]
fn logged_aggregate_admission_counts_packed_bytes_and_tied_handles_once() {
    use std::sync::Arc;
    use tritium_evidence::{
        EvidenceRecorder, TENSOR_UPLOADED_REGISTRATION, verify_jsonl_with_registry,
    };
    use tritium_runtime::{
        TensorPayloadBudget, TensorUploadContext, TensorUploadError, upload_tensor_logged_admitted,
    };
    use tritium_spec::DeviceBuffer;

    let cpu = CpuBackend::new();
    let fixture = tritium_testkit::additive_upload_fixture();
    let view = TensorView::Additive(fixture.view());
    assert_eq!(view.decoded_payload_bytes().unwrap(), 16);
    let budget = TensorPayloadBudget::new(20);
    let mut recorder =
        EvidenceRecorder::new("cpu-packed-budget", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
    let mut upload = |name| {
        upload_tensor_logged_admitted(
            &cpu,
            view,
            TensorUploadPolicy {
                native_only: true,
                max_payload_bytes: Some(10),
            },
            name,
            TensorUploadContext {
                recorder: &mut recorder,
                span: "load",
                parent: None,
                logical_time: 0,
            },
            &budget,
        )
    };
    let (first, caps) = upload("embedding").unwrap();
    assert_eq!(caps.payload_bytes, 10);
    let (second, _) = upload("projection").unwrap();
    assert_eq!(budget.used_bytes(), 20);
    assert!(matches!(
        upload("rejected"),
        Err(TensorUploadError::Backend(
            BackendError::TensorPayloadBudgetExceeded {
                requested: 30,
                limit: 20
            }
        ))
    ));
    let first: Arc<dyn DeviceBuffer> = Arc::from(first);
    let tied = Arc::clone(&first);
    drop(first);
    assert_eq!(budget.used_bytes(), 20);
    let mut out = [99.; 2];
    let mut scratch = [99.; 4];
    cpu.matmul(TensorMatmul {
        act: &[1.; 4],
        tensor: &*tied,
        batch: 1,
        transformed_act: &mut scratch,
        out: &mut out,
    })
    .unwrap();
    assert_eq!(out, [-1., -2.]);
    drop(second);
    assert_eq!(budget.used_bytes(), 10);
    let (third, _) = upload("replacement").unwrap();
    assert_eq!(budget.used_bytes(), 20);
    drop(tied);
    drop(third);
    assert_eq!(budget.used_bytes(), 0);
    assert_eq!(recorder.events().len(), 3);
    let verified = verify_jsonl_with_registry(
        &recorder.to_jsonl().unwrap(),
        [TENSOR_UPLOADED_REGISTRATION],
    )
    .unwrap();
    assert_eq!(verified.root_digest, recorder.root_digest().unwrap());
}
