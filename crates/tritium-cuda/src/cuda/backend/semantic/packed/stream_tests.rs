//! Pure native-envelope/number parity; never initializes a CUDA context.
use super::*;
use tritium_core::{PerTilePlan, Trit, UniformPlan};
use tritium_schema::{ADMITTED_LAWS, AdditiveLayout, PlaneAllocation, Transport};
use tritium_spec::{TensorMetadata, TensorStreamEncoding};
struct Source<'a> {
    metadata: AdditiveMetadata<'a>,
    bytes: &'a [u8],
    chunk: usize,
    fail: bool,
}
impl TensorStreamSource for Source<'_> {
    fn metadata(&self) -> TensorMetadata<'_> {
        TensorMetadata::Additive(self.metadata)
    }
    fn encoding(&self) -> TensorStreamEncoding {
        TensorStreamEncoding::CanonicalRawAdditive
    }
    fn encoded_bytes(&self) -> u64 {
        self.bytes.len() as u64
    }
    fn visit(
        &self,
        visitor: &mut dyn FnMut(&[u8]) -> Result<(), BackendError>,
    ) -> Result<(), BackendError> {
        for chunk in self.bytes.chunks(self.chunk) {
            visitor(chunk)?;
        }
        if self.fail {
            return Err(BackendError::Backend("source terminal fault".into()));
        }
        Ok(())
    }
}
#[test]
fn encoded_stream_native_envelopes_and_scalar_cursors_match_decoded_oracle_all_profiles() {
    let mut cases = 0;
    for law in ADMITTED_LAWS {
        for planes in 1..=law.max_planes {
            for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
                for group in [32, 128] {
                    for mode in 0..3 {
                        let layout = AdditiveLayout {
                            rows: 3,
                            cols: 260,
                            tile: 256,
                            group,
                            max_planes: planes,
                            allocation: if mode == 0 {
                                PlaneAllocation::Uniform
                            } else {
                                PlaneAllocation::PerTile
                            },
                            codec,
                            law: law.law,
                            basis: Basis::SignedRht {
                                block: 4,
                                seed: 7,
                                domain: 9,
                            },
                            transport: Transport::Raw,
                        };
                        let values = |len| {
                            (0..len)
                                .map(|i| {
                                    if i % 4 == 0 {
                                        Trit::ZERO
                                    } else if i % 2 == 0 {
                                        Trit::NEG
                                    } else {
                                        Trit::POS
                                    }
                                })
                                .collect()
                        };
                        let tensor = if mode == 0 {
                            let p = UniformPlan::new(layout, planes).unwrap();
                            AdditiveTensor::new(
                                layout,
                                planes,
                                values(p.trit_count()),
                                vec![0.5; p.scale_count()],
                            )
                            .unwrap()
                        } else {
                            let counts = if mode == 2 {
                                vec![0; 6]
                            } else {
                                [0, 1, planes, 2, 2, 0]
                                    .into_iter()
                                    .map(|n| n.min(planes))
                                    .collect()
                            };
                            let p = PerTilePlan::new(layout, &counts).unwrap();
                            let trits = values(p.trit_count());
                            let scales = vec![0.5; p.scale_count()];
                            AdditiveTensor::new_per_tile(layout, counts, trits, scales).unwrap()
                        };
                        let view = tensor.view();
                        let metadata = AdditiveMetadata::from_view(view);
                        let raw = tensor.encode_payload().unwrap();
                        let source = Source {
                            metadata,
                            bytes: &raw,
                            chunk: 7,
                            fail: false,
                        };
                        let bytes = PackedAdditive::stage_stream(metadata, &source).unwrap();
                        assert_eq!(bytes, PackedAdditive::stage_view(view).unwrap().0);
                        let plan = Plan::from_metadata(metadata).unwrap();
                        let (payload, workspace, encoded) =
                            PackedAdditive::stream_metadata(metadata).unwrap();
                        assert_eq!((payload, encoded), (bytes.len(), raw.len()));
                        assert_eq!(workspace, 3 * 65536);
                        let mut suffix = Vec::new();
                        plan.visit_suffix(metadata, |part| {
                            suffix.extend_from_slice(part);
                            Ok(())
                        })
                        .unwrap();
                        assert_eq!(
                            suffix,
                            &bytes[plan.header[10] as usize..],
                            "bounded suffix must match independent legacy layout"
                        );
                        let mut expected = Vec::new();
                        let mut row = vec![0.; 260];
                        for r in 0..3 {
                            view.dequant_row_into(r, &mut row).unwrap();
                            expected.extend_from_slice(&row);
                        }
                        let mut actual = Vec::new();
                        plan.visit_weights(&bytes, |_, _, v| {
                            actual.push(v);
                            Ok(())
                        })
                        .unwrap();
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
    eprintln!("CUDA_STREAM_HOST_NATIVE_PROFILE_CASES={cases}");
    assert_eq!(
        cases,
        ADMITTED_LAWS
            .iter()
            .map(|l| usize::from(l.max_planes))
            .sum::<usize>()
            * 3
            * 2
            * 3
    );
}

#[test]
fn encoded_stream_finite_cancellation_and_actual_overflow_keep_the_existing_contract() {
    let layout = AdditiveLayout {
        rows: 2,
        cols: 4,
        tile: 256,
        group: 32,
        max_planes: 3,
        allocation: PlaneAllocation::Uniform,
        codec: PlaneCodec::D2,
        law: tritium_schema::ScaleLaw {
            anchor: ScaleAnchor::Group,
            relation: PlaneRelation::Free,
            precision: ScalePrecision::F32,
        },
        basis: Basis::Identity,
        transport: Transport::Raw,
    };
    for cancel in [false, true] {
        let trits = if cancel {
            [Trit::POS; 8]
                .into_iter()
                .chain([Trit::NEG; 8])
                .chain([Trit::ZERO; 8])
                .collect()
        } else {
            vec![Trit::POS; 24]
        };
        let tensor = AdditiveTensor::new(layout, 3, trits, vec![f32::MAX; 6]).unwrap();
        let raw = tensor.encode_payload().unwrap();
        let metadata = AdditiveMetadata::from_view(tensor.view());
        let source = Source {
            metadata,
            bytes: &raw,
            chunk: 1,
            fail: false,
        };
        assert_eq!(
            PackedAdditive::stage_stream(metadata, &source).is_ok(),
            cancel
        );
        assert_eq!(PackedAdditive::stage_view(tensor.view()).is_ok(), cancel);
    }
    let compact = AdditiveLayout {
        allocation: PlaneAllocation::PerTile,
        ..layout
    };
    for cancel in [false, true] {
        let trits = if cancel {
            [Trit::POS; 8]
                .into_iter()
                .chain([Trit::NEG; 4])
                .chain([Trit::ZERO; 4])
                .collect()
        } else {
            vec![Trit::POS; 16]
        };
        let tensor =
            AdditiveTensor::new_per_tile(compact, vec![1, 3], trits, vec![f32::MAX; 4]).unwrap();
        let raw = tensor.encode_payload().unwrap();
        let metadata = AdditiveMetadata::from_view(tensor.view());
        let source = Source {
            metadata,
            bytes: &raw,
            chunk: 1,
            fail: false,
        };
        assert_eq!(
            PackedAdditive::stage_stream(metadata, &source).is_ok(),
            cancel
        );
        assert_eq!(PackedAdditive::stage_view(tensor.view()).is_ok(), cancel);
    }
}

#[test]
fn encoded_stream_canonical_padding_chunk_and_terminal_faults_refuse_private_staging() {
    let tensor = tritium_testkit::additive_upload_fixture();
    let metadata = AdditiveMetadata::from_view(tensor.view());
    let raw = tensor.encode_payload().unwrap();
    let source = Source {
        metadata,
        bytes: &raw,
        chunk: 1,
        fail: true,
    };
    assert!(PackedAdditive::stage_stream(metadata, &source).is_err());
    let mut bad = raw.clone();
    bad[0] |= 3;
    assert!(
        PackedAdditive::stage_stream(
            metadata,
            &Source {
                metadata,
                bytes: &bad,
                chunk: 1,
                fail: false
            }
        )
        .is_err()
    );
    assert!(
        PackedAdditive::stage_stream(
            metadata,
            &Source {
                metadata,
                bytes: &raw[..raw.len() - 1],
                chunk: 1,
                fail: false
            }
        )
        .is_err()
    );
}
