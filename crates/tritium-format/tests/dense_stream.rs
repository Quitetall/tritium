use tritium_format::trit_model::DensePayloadDecoder;
use tritium_schema::ScalePrecision;

#[test]
fn every_precision_scalar_split_and_empty_payload_preserve_exact_values() {
    let values = [0., -0., -2., 0.5, 128., 1.25];
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
        for split in 1..=9 {
            let mut actual = [999.; 6];
            let mut decoder = DensePayloadDecoder::new(precision, &mut actual).unwrap();
            for chunk in bytes.chunks(split) {
                decoder.push(chunk).unwrap();
            }
            decoder.finish().unwrap();
            assert_eq!(actual.map(f32::to_bits), values.map(f32::to_bits));
        }
        DensePayloadDecoder::new(precision, &mut [])
            .unwrap()
            .finish()
            .unwrap();
    }
}
#[test]
fn overflow_short_trailing_nonfinite_and_ignored_push_errors_stay_poisoned() {
    assert!(DensePayloadDecoder::byte_len(ScalePrecision::F32, usize::MAX).is_err());
    for bytes in [
        &[0u8; 3][..],
        &[0u8; 5][..],
        &f32::NAN.to_le_bytes(),
        &f32::INFINITY.to_le_bytes(),
    ] {
        let mut output = [99.];
        let mut decoder = DensePayloadDecoder::new(ScalePrecision::F32, &mut output).unwrap();
        let _ = decoder.push(bytes);
        assert!(decoder.finish().is_err());
    }
    let mut output = [99.];
    let mut decoder = DensePayloadDecoder::new(ScalePrecision::F32, &mut output).unwrap();
    assert!(decoder.push(&[0u8; 5]).is_err());
    assert!(decoder.push(&[0u8; 4]).is_err());
    assert!(decoder.finish().is_err());
}
