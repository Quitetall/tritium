//! Canonical data transport, not authenticated capture or numerical qualification.
use tritium_quantize::{CurvatureSourceId, InputGram, InputGramAccumulator, InputGramDecodeError};

fn fixture(bound: bool) -> InputGram {
    let mut a = if bound {
        InputGramAccumulator::new_bound(
            3,
            CurvatureSourceId::new([1; 32], [2; 32], [3; 32]).unwrap(),
            17,
        )
        .unwrap()
    } else {
        InputGramAccumulator::new(3).unwrap()
    };
    a.accumulate_batch(
        &[1., -0., -2., 4., 3., -1., -0., 0., 2.],
        3,
        Some(&[0.5, 0., 2.]),
        Some(&[true, false, true]),
    )
    .unwrap();
    a.finish().unwrap()
}
fn digest(bytes: &[u8]) -> [u8; 32] {
    let mut h = blake3::Hasher::new_derive_key("tritium salt v2 input gram v1");
    h.update(bytes);
    *h.finalize().as_bytes()
}
#[test]
fn bound_and_unbound_round_trips_preserve_all_data_and_existing_identity() {
    for bound in [false, true] {
        let gram = fixture(bound);
        let bytes = gram.canonical_bytes();
        assert_eq!(gram.digest(), digest(&bytes));
        let read = InputGram::decode_canonical(&bytes, gram.digest(), 72).unwrap();
        assert_eq!(read, gram);
        assert_eq!(read.canonical_bytes(), bytes);
        assert_eq!(read.digest(), gram.digest());
        assert_eq!(read.sample_count(), 3);
        assert_eq!(read.selected_count(), 2);
        assert_eq!(read.total_weight(), 2.5);
        assert_eq!(read.source_id().is_some(), bound);
    }
}
#[test]
fn malformed_canonical_fields_never_receive_even_with_matching_raw_digest() {
    let bytes = fixture(true).canonical_bytes();
    // Version2 bound layout: dimension5, counts13/21, weight29, sourceflag37,
    // source38..134, traces134..198, entrycount198, values206.
    let mut variants = Vec::new();
    for offset in [0, 4, 37] {
        let mut b = bytes.clone();
        b[offset] = 9;
        variants.push(b);
    }
    for offset in [5, 13, 21, 38, 70, 102] {
        let mut b = bytes.clone();
        let count = if offset >= 38 { 32 } else { 8 };
        b[offset..offset + count].fill(0);
        variants.push(b);
    }
    for (offset, bits) in [
        (29, f64::INFINITY.to_bits()),
        (29, (-0f64).to_bits()),
        (29, (-1f64).to_bits()),
        (206, f64::NAN.to_bits()),
        (206, (-0f64).to_bits()),
    ] {
        let mut b = bytes.clone();
        b[offset..offset + 8].copy_from_slice(&bits.to_le_bytes());
        variants.push(b);
    }
    let mut selected = bytes.clone();
    selected[21..29].copy_from_slice(&4u64.to_le_bytes());
    variants.push(selected);
    let mut count = bytes.clone();
    count[198..206].copy_from_slice(&8u64.to_le_bytes());
    variants.push(count);
    let mut trailing = bytes.clone();
    trailing.push(0);
    variants.push(trailing);
    for b in variants {
        assert!(InputGram::decode_canonical(&b, digest(&b), 72).is_err());
    }
    for end in 0..bytes.len() {
        assert!(InputGram::decode_canonical(&bytes[..end], digest(&bytes[..end]), 72).is_err());
    }
}
#[test]
fn independent_identity_capacity_and_overflow_refusals() {
    let gram = fixture(true);
    let bytes = gram.canonical_bytes();
    for cap in [0, 1, 71] {
        assert_eq!(
            InputGram::decode_canonical(&bytes, gram.digest(), cap),
            Err(InputGramDecodeError::Limit)
        );
    }
    for expected in [[0; 32], [9; 32]] {
        assert_eq!(
            InputGram::decode_canonical(&bytes, expected, 72),
            Err(InputGramDecodeError::Identity)
        );
    }
    let mut b = bytes.clone();
    b[198..206].copy_from_slice(&u64::MAX.to_le_bytes());
    assert_eq!(
        InputGram::decode_canonical(&b, digest(&b), usize::MAX),
        Err(InputGramDecodeError::Limit)
    );
    let mut b = bytes.clone();
    b[5..13].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(InputGram::decode_canonical(&b, digest(&b), 72).is_err());
}
