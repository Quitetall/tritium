//! CPU-only fixture proofs; no CUDA context or physical-memory qualification.
#[path = "support/encoded_head.rs"]
mod fixture;
use fixture::{EncodedHead, SCALE_BITS};
use tritium_core::{Trit, UniformPlan};
use tritium_format::{AdditiveStreamValidator, AdditiveTensor};
use tritium_spec::{BackendError, TENSOR_STREAM_CHUNK_BYTES, TensorStreamSource};

#[test]
fn generated_windows_equal_independent_small_owned_codec_oracle() {
    let mut cases = 0;
    for (rows, cols) in [1, 3, 7]
        .into_iter()
        .flat_map(|rows| [1, 4, 5, 31, 127, 128, 129, 257].map(|cols| (rows, cols)))
        .chain([(7, 17408), (6001, 129)])
    {
        for planes in 1..=3 {
            for seed in [None, Some(7), Some(20261020)] {
                let source = EncodedHead::new(rows, cols, planes);
                let source = seed.map_or(source, |seed| {
                    EncodedHead::new(rows, cols, planes).counter_mixed(seed)
                });
                let source_ref = &source;
                let plan = UniformPlan::new(source.layout, planes).unwrap();
                let trits = (0..planes as usize)
                    .flat_map(|plane| {
                        (0..rows * cols).map(move |index| {
                            // Independent per-coefficient radix lookup, rather than
                            // the stream generator's rolling encoded-byte counter.
                            let code = source_ref.code_at(plane, index / 5) as usize;
                            let digit = (code / 3usize.pow((index % 5) as u32)) % 3;
                            Trit::from_i8(digit as i8 - 1).unwrap()
                        })
                    })
                    .collect();
                let scales = (0..plan.scale_count())
                    .map(|index| half::f16::from_bits(SCALE_BITS[index % 5]).to_f32())
                    .collect();
                let expected = AdditiveTensor::new(source.layout, planes, trits, scales)
                    .unwrap()
                    .encode_payload()
                    .unwrap();
                let mut actual = Vec::new();
                let mut validator =
                    AdditiveStreamValidator::new_uniform(source.layout, planes, expected.len())
                        .unwrap();
                source
                    .visit(&mut |chunk| {
                        assert!(!chunk.is_empty() && chunk.len() <= TENSOR_STREAM_CHUNK_BYTES);
                        validator.push(chunk).unwrap();
                        actual.extend_from_slice(chunk);
                        Ok(())
                    })
                    .unwrap();
                validator.finish().unwrap();
                assert_eq!(actual, expected, "rows={rows} cols={cols} planes={planes}");
                assert_eq!(source.encoded_bytes(), actual.len() as u64);
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 234);
}

#[test]
fn multi_window_stream_propagates_visitor_failure_without_later_calls() {
    let source = EncodedHead::new(7, 17408, 3);
    let mut calls = 0;
    let error = source
        .visit(&mut |chunk| {
            calls += 1;
            assert_eq!(chunk.len(), TENSOR_STREAM_CHUNK_BYTES);
            Err(BackendError::Backend("fixture visitor stop".into()))
        })
        .unwrap_err();
    assert!(error.to_string().contains("fixture visitor stop"));
    assert_eq!(calls, 1);
}

#[test]
fn head_geometry_and_payload_are_described_without_allocating_coefficients() {
    let source = EncodedHead::new(248320, 5120, 3);
    assert_eq!(source.encoded_bytes(), 822_435_840);
    assert!(std::mem::size_of_val(&source) < 1024);
    assert_eq!(source.planes, 3);
    // Size of this metadata value is not stack/RSS/VRAM qualification.
}
