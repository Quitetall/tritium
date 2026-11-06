use tritium_core::{PerTilePlan, Trit};
use tritium_cpu::CpuBackend;
use tritium_format::AdditiveTensor;
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, PlaneRelation, ScaleAnchor,
    TensorUploadPolicy, Transport,
};
use tritium_spec::{AdditiveMetadata, TensorMetadata, TensorView, TernaryBackend};
#[path = "common/stream_source.rs"]
mod stream_source;

#[test]
fn metadata_plans_match_actual_compact_upload_for_all_laws_codecs_and_pruning() {
    let cpu = CpuBackend::new();
    for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
        for law in ADMITTED_LAWS {
            for counts in [[0, 1, 2, 0], [0, 0, 0, 0], [2, 2, 1, 1]] {
                let counts = counts.map(|count| count.min(law.max_planes));
                let layout = AdditiveLayout {
                    rows: 2,
                    cols: 512,
                    tile: 256,
                    group: 128,
                    max_planes: 2.min(law.max_planes),
                    allocation: PlaneAllocation::PerTile,
                    codec,
                    law: law.law,
                    basis: Basis::SignedRht {
                        block: 128,
                        seed: 3,
                        domain: 9,
                    },
                    transport: Transport::Raw,
                };
                let geometry = PerTilePlan::new(layout, &counts).unwrap();
                let trits: Vec<_> = (0..geometry.trit_count())
                    .map(|i| if i % 4 == 0 { Trit::ZERO } else { Trit::POS })
                    .collect();
                let scales = vec![0.5; geometry.scale_count()];
                let owned =
                    AdditiveTensor::new_per_tile(layout, counts.to_vec(), trits, scales).unwrap();
                let metadata =
                    TensorMetadata::Additive(AdditiveMetadata::per_tile(layout, &counts).unwrap());
                let plan = cpu.tensor_upload_plan(metadata).unwrap().unwrap();
                let view = TensorView::Additive(owned.view());
                assert_eq!(cpu.tensor_caps(view).unwrap(), Some(plan.caps));
                let (buffer, caps) = cpu
                    .upload_tensor_checked(view, TensorUploadPolicy::default())
                    .unwrap();
                assert_eq!(caps, plan.caps);
                assert_eq!(buffer.len_bytes() as u64, plan.caps.payload_bytes);
                assert_eq!(plan.host_workspace_bytes, 0);
                assert_eq!(plan.device_workspace_bytes, 0);
                let bytes = owned.encode_payload().unwrap();
                let source = stream_source::Source::new(
                    metadata,
                    tritium_spec::TensorStreamEncoding::CanonicalRawAdditive,
                    &bytes,
                );
                let streamed = cpu.upload_tensor_stream(&source).unwrap();
                assert_eq!(streamed.len_bytes(), buffer.len_bytes());
                let mut streamed_rows = vec![0.; 1024];
                let mut decoded_rows = vec![0.; 1024];
                cpu.embed_rows(&*buffer, &[1, 0], &mut decoded_rows)
                    .unwrap();
                cpu.embed_rows(&*streamed, &[1, 0], &mut streamed_rows)
                    .unwrap();
                assert_eq!(streamed_rows, decoded_rows);
            }
        }
    }
}

#[test]
fn plans_refuse_bad_geometry_and_do_not_substitute_for_numerical_validation() {
    let cpu = CpuBackend::new();
    assert_eq!(
        cpu.tensor_upload_plan(TensorMetadata::Dense { rows: 0, cols: 4 })
            .unwrap()
            .unwrap()
            .caps
            .payload_bytes,
        0
    );
    assert!(
        cpu.tensor_upload_plan(TensorMetadata::Dense {
            rows: usize::MAX,
            cols: 2
        })
        .is_err()
    );
    let mut layout = AdditiveLayout {
        rows: 1,
        cols: 32,
        tile: 256,
        group: 32,
        max_planes: 1,
        allocation: PlaneAllocation::Uniform,
        codec: PlaneCodec::S34,
        law: ADMITTED_LAWS[0].law,
        basis: Basis::Identity,
        transport: Transport::Raw,
    };
    layout.law.anchor = ScaleAnchor::Group;
    layout.law.relation = PlaneRelation::Free;
    let metadata = TensorMetadata::Additive(AdditiveMetadata::uniform(layout, 1).unwrap());
    assert!(cpu.tensor_upload_plan(metadata).unwrap().is_some());
    let bad = tritium_core::AdditiveView::new(layout, 1, &[Trit::POS; 32], &[0.5]).unwrap();
    assert!(cpu.tensor_caps(TensorView::Additive(bad)).is_err());
}
