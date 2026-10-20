//! Candidate compact allocation cases; not an independently frozen profile.

use tritium_core::{PerTilePlan, Trit, reference_embed, reference_ternary_matmul};
use tritium_format::AdditiveTensor;
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, TensorExecution,
    TensorUploadPolicy, Transport,
};
use tritium_spec::{BackendError, TensorMatmul, TensorView, TernaryBackend};

/// Replay candidate compact upload/matmul/gather cases against the shared core.
///
/// All admitted laws/groups, three codecs/bases, ragged tails, noncontiguous
/// higher planes and all-pruned tensors are exercised. Existing frozen vectors
/// are unchanged. Bit parity is software evidence, not profile freeze, physical
/// memory/performance qualification or independent empirical release clearance.
pub fn assert_candidate_per_tile_conformance(
    backend: &dyn TernaryBackend,
    execution: TensorExecution,
) -> usize {
    let mut cases = 0;
    // Three batches produce 120 outputs, above the CPU parallel-row grain.
    let row_count = 40;
    for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
        let widths: &[usize] = if codec == PlaneCodec::S34 {
            &[4, 128, 256, 260, 512]
        } else {
            &[1, 33, 129, 255, 256, 257, 511, 512]
        };
        for admitted in ADMITTED_LAWS {
            for group in [32, 128] {
                for &cols in widths {
                    for basis in [
                        Basis::Identity,
                        Basis::Hadamard { block: 4 },
                        Basis::SignedRht {
                            block: 4,
                            seed: 7,
                            domain: 9,
                        },
                    ] {
                        if basis != Basis::Identity && !cols.is_multiple_of(4) {
                            continue;
                        }
                        for pruned in [false, true] {
                            let layout = AdditiveLayout {
                                rows: row_count as u64,
                                cols: cols as u64,
                                tile: 256,
                                group,
                                max_planes: admitted.max_planes,
                                allocation: PlaneAllocation::PerTile,
                                codec,
                                law: admitted.law,
                                basis,
                                transport: Transport::Raw,
                            };
                            let counts: Vec<_> = [1_u8, 3, 2, 0, 3, 0, 1, 2]
                                .into_iter()
                                .cycle()
                                .take(row_count * cols.div_ceil(256))
                                .map(|n| {
                                    if pruned {
                                        0
                                    } else {
                                        n.min(admitted.max_planes)
                                    }
                                })
                                .collect();
                            let plan = PerTilePlan::new(layout, &counts).unwrap();
                            let trits = [Trit::POS, Trit::NEG, Trit::ZERO, Trit::POS]
                                .into_iter()
                                .cycle()
                                .take(plan.trit_count())
                                .collect();
                            let scales = [0.5, 1., 2., 3.]
                                .into_iter()
                                .cycle()
                                .take(plan.scale_count())
                                .collect();
                            let owned = AdditiveTensor::new_per_tile(layout, counts, trits, scales)
                                .unwrap();
                            let view = TensorView::Additive(owned.view());
                            let caps = backend.tensor_caps(view).unwrap().unwrap();
                            assert_eq!(caps.execution, execution);
                            if caps.payload_bytes > 0 {
                                assert!(matches!(
                                    backend.upload_tensor_checked(
                                        view,
                                        TensorUploadPolicy {
                                            native_only: execution == TensorExecution::Native,
                                            max_payload_bytes: Some(caps.payload_bytes - 1),
                                        }
                                    ),
                                    Err(BackendError::TensorPayloadBudgetExceeded { .. })
                                ));
                            }
                            let policy = TensorUploadPolicy {
                                native_only: execution == TensorExecution::Native,
                                max_payload_bytes: Some(caps.payload_bytes),
                            };
                            let (uploaded, reported) =
                                backend.upload_tensor_checked(view, policy).unwrap();
                            assert_eq!(reported, caps);
                            assert_eq!(uploaded.len_bytes() as u64, caps.payload_bytes);
                            let act: Vec<_> = [1e20, -1e20, 0.25, -0.5, 2., -3., 0., 1.]
                                .into_iter()
                                .cycle()
                                .take(3 * cols)
                                .collect();
                            let mut expected_scratch = vec![0.; act.len()];
                            let mut expected = vec![0.; 3 * row_count];
                            reference_ternary_matmul(
                                &act,
                                &owned.view().bind_matmul(),
                                3,
                                &mut expected_scratch,
                                &mut expected,
                            )
                            .unwrap();
                            let ids = [row_count - 1, 0, row_count / 2, 1, row_count - 1];
                            let mut expected_rows = vec![0.; ids.len() * cols];
                            reference_embed(&owned.view().bind_gather(), &ids, &mut expected_rows)
                                .unwrap();
                            // The device handle owns all packed data and geometry.
                            drop(owned);
                            let mut scratch = vec![99.; act.len()];
                            let mut out = vec![99.; 3 * row_count];
                            backend
                                .matmul(TensorMatmul {
                                    act: &act,
                                    tensor: &*uploaded,
                                    batch: 3,
                                    transformed_act: &mut scratch,
                                    out: &mut out,
                                })
                                .unwrap();
                            assert_eq!(
                                scratch.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                                expected_scratch
                                    .iter()
                                    .map(|x| x.to_bits())
                                    .collect::<Vec<_>>()
                            );
                            assert_eq!(
                                out.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                                expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                                "{layout:?}"
                            );
                            let mut rows = vec![99.; expected_rows.len()];
                            backend.embed_rows(&*uploaded, &ids, &mut rows).unwrap();
                            assert_eq!(
                                rows.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                                expected_rows
                                    .iter()
                                    .map(|x| x.to_bits())
                                    .collect::<Vec<_>>(),
                                "{layout:?}"
                            );
                            rows.fill(99.);
                            assert!(
                                backend
                                    .embed_rows(&*uploaded, &[0, row_count, 0, 0, 0], &mut rows)
                                    .is_err()
                            );
                            assert!(rows.iter().all(|x| *x == 99.));
                            scratch.fill(99.);
                            out.fill(99.);
                            assert!(
                                backend
                                    .matmul(TensorMatmul {
                                        act: &act[..act.len() - 1],
                                        tensor: &*uploaded,
                                        batch: 3,
                                        transformed_act: &mut scratch,
                                        out: &mut out
                                    })
                                    .is_err()
                            );
                            assert!(scratch.iter().chain(&out).all(|x| *x == 99.));
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases, 624);
    cases
}
