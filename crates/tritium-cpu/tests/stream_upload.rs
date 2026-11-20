use tritium_cpu::CpuBackend;
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, ScalePrecision, Transport,
};
use tritium_spec::{
    AdditiveMetadata, TENSOR_STREAM_CHUNK_BYTES, TensorMetadata, TensorStreamEncoding,
    TernaryBackend,
};
#[path = "common/stream_source.rs"]
mod stream_source;
use stream_source::Source;

#[test]
fn dense_precision_and_every_scalar_split_widen_into_the_same_final_buffer() {
    let cpu = CpuBackend::new();
    let values = [0., -0., 0.5, -2., 64., 1.25];
    for precision in [
        ScalePrecision::F16,
        ScalePrecision::Bf16,
        ScalePrecision::F32,
    ] {
        let bytes: Vec<_> = values
            .iter()
            .flat_map(|&v| match precision {
                ScalePrecision::F16 => half::f16::from_f32(v).to_bits().to_le_bytes().to_vec(),
                ScalePrecision::Bf16 => half::bf16::from_f32(v).to_bits().to_le_bytes().to_vec(),
                _ => v.to_le_bytes().to_vec(),
            })
            .collect();
        for chunk in 1..=9 {
            let mut source = Source::new(
                TensorMetadata::Dense { rows: 2, cols: 3 },
                TensorStreamEncoding::DenseLittleEndian(precision),
                &bytes,
            );
            source.chunk = chunk;
            let buffer = cpu.upload_tensor_stream(&source).unwrap();
            assert_eq!(buffer.len_bytes(), 24);
            let mut out = [0.; 6];
            cpu.embed_rows(&*buffer, &[0, 1], &mut out).unwrap();
            assert_eq!(out.map(f32::to_bits), values.map(f32::to_bits));
        }
    }
}

#[test]
fn stream_length_chunk_finiteness_codec_padding_and_source_faults_fail_closed() {
    let cpu = CpuBackend::new();
    let metadata = TensorMetadata::Dense { rows: 1, cols: 1 };
    for bytes in [
        &[0u8; 3][..],
        &[0u8; 5][..],
        &f32::NAN.to_le_bytes(),
        &f32::INFINITY.to_le_bytes(),
    ] {
        let mut source = Source::new(
            metadata,
            TensorStreamEncoding::DenseLittleEndian(ScalePrecision::F32),
            bytes,
        );
        source.declared = 4;
        assert!(cpu.upload_tensor_stream(&source).is_err());
    }
    let bytes = [0u8; 4];
    let mut source = Source::new(
        metadata,
        TensorStreamEncoding::DenseLittleEndian(ScalePrecision::F32),
        &bytes,
    );
    source.fail = true;
    assert!(cpu.upload_tensor_stream(&source).is_err());
    let bytes = vec![0u8; TENSOR_STREAM_CHUNK_BYTES + 4];
    let mut source = Source::new(
        TensorMetadata::Dense {
            rows: 1,
            cols: bytes.len() / 4,
        },
        TensorStreamEncoding::DenseLittleEndian(ScalePrecision::F32),
        &bytes,
    );
    source.chunk = bytes.len();
    assert!(cpu.upload_tensor_stream(&source).is_err());
    let layout = AdditiveLayout {
        rows: 1,
        cols: 1,
        tile: 256,
        group: 32,
        max_planes: 1,
        allocation: PlaneAllocation::Uniform,
        codec: PlaneCodec::D2,
        law: ADMITTED_LAWS[0].law,
        basis: Basis::Identity,
        transport: Transport::Raw,
    };
    let metadata = TensorMetadata::Additive(AdditiveMetadata::uniform(layout, 1).unwrap());
    let tensor =
        tritium_format::AdditiveTensor::new(layout, 1, vec![tritium_core::Trit::POS], vec![0.5])
            .unwrap();
    let mut bytes = tensor.encode_payload().unwrap();
    bytes[0] |= 0xc0; // Non-canonical padding / reserved symbol.
    assert!(
        cpu.upload_tensor_stream(&Source::new(
            metadata,
            TensorStreamEncoding::CanonicalRawAdditive,
            &bytes
        ))
        .is_err()
    );
    assert!(
        cpu.tensor_stream_plan(
            metadata,
            TensorStreamEncoding::DenseLittleEndian(ScalePrecision::F32)
        )
        .unwrap()
        .is_none()
    );
}
