//! Differential checks against the retained original directory codec.
use super::*;

fn parity(bytes: &[u8]) {
    assert_eq!(decode_directory(bytes), decode_reference_directory(bytes));
}

fn inventory(size: usize) -> Vec<TritBlobInfo> {
    let types = [
        DType::Ternary,
        DType::I4,
        DType::I8,
        DType::U8,
        DType::F8E4M3,
        DType::F8E5M2,
        DType::F4E2M1,
        DType::F16,
        DType::BF16,
        DType::F32,
    ];
    let widths = [
        0,
        23,
        24,
        255,
        256,
        65535,
        65536,
        u64::from(u32::MAX),
        u64::MAX,
    ];
    let mut blobs: Vec<_> = (0..size)
        .map(|index| {
            let kind = match index % 12 {
                0 => TritBlobKind::AdditiveTensor,
                1 => TritBlobKind::Asset(format!("asset-{index:06}-λ-🦀")),
                n => TritBlobKind::DenseTensor(types[n - 2]),
            };
            let mut id = [0; 32];
            id[..8].copy_from_slice(&(index as u64).to_be_bytes());
            TritBlobInfo {
                id: BlobId::from_bytes(id),
                kind,
                offset: widths[index % widths.len()],
                length: widths[(index + 4) % widths.len()],
            }
        })
        .collect();
    blobs.sort_by(compare_blob_entries);
    blobs
}

#[test]
fn borrowed_directory_matches_original_codec_for_all_kinds_and_integer_widths() {
    for size in [0, 1, 2, 12, 23, 24, 25, 255, 256, 257, 4096] {
        let blobs = inventory(size);
        let bytes = encode_directory(&blobs).unwrap();
        assert_eq!(
            decode_borrowed_directory(&bytes).unwrap(),
            Some(blobs.clone())
        );
        assert_eq!(decode_directory(&bytes).unwrap(), blobs);
        parity(&bytes);
    }
    // Identical IDs with distinct kinds/names remain legal directory records.
    let mut blobs = inventory(25);
    for blob in &mut blobs {
        blob.id = BlobId::from_bytes([0; 32]);
        blob.offset = 256;
        blob.length = 0;
    }
    blobs.sort_by(compare_blob_entries);
    blobs.dedup();
    let bytes = encode_directory(&blobs).unwrap();
    assert_eq!(decode_borrowed_directory(&bytes).unwrap(), Some(blobs));
    parity(&bytes);
}

#[test]
fn borrowed_directory_preserves_original_refusals_under_truncation_and_mutation() {
    let bytes = encode_directory(&inventory(25)).unwrap();
    let mut cases = 0usize;
    for length in 0..=bytes.len() {
        parity(&bytes[..length]);
        cases += 1;
    }
    for offset in 0..bytes.len() {
        for mask in [1, 0x20, 0x80, 0xff] {
            let mut changed = bytes.clone();
            changed[offset] ^= mask;
            parity(&changed);
            cases += 1;
        }
    }
    for suffix in [&[0][..], &[0x9f, 0xff][..], &[0xc0, 0][..]] {
        let mut changed = bytes.clone();
        changed.extend_from_slice(suffix);
        parity(&changed);
        cases += 1;
    }
    eprintln!("DIRECTORY_DIFFERENTIAL_MUTATIONS cases={cases}");
}

#[test]
fn directory_shape_kind_alias_order_and_reference_forms_keep_original_errors() {
    let id = Value::Bytes(vec![0; 32]);
    let entry = |tag, aux| {
        Value::Array(vec![
            id.clone(),
            Value::Integer(tag),
            aux,
            Value::Integer(256.into()),
            Value::Integer(0.into()),
        ])
    };
    let entries = [
        entry(0.into(), Value::Text("not-null".into())),
        entry(1.into(), Value::Text("F16".into())),
        entry(1.into(), Value::Null),
        entry(2.into(), Value::Text(String::new())),
        entry(3.into(), Value::Null),
        entry((-1).into(), Value::Null),
        entry(
            2.into(),
            Value::Tag(0, Box::new(Value::Text("asset".into()))),
        ),
        entry(2.into(), Value::Float(0.5)),
        Value::Null,
        Value::Array(vec![]),
        Value::Array(vec![
            Value::Bytes(vec![0; 31]),
            Value::Integer(0.into()),
            Value::Null,
            Value::Integer(256.into()),
            Value::Integer(0.into()),
        ]),
    ];
    for entry in entries {
        let root = Value::Array(vec![Value::Integer(1.into()), Value::Array(vec![entry])]);
        parity(&encode_cbor_value(&root).unwrap());
    }
    for version in [0, 2, u64::MAX] {
        parity(
            &encode_cbor_value(&Value::Array(vec![
                Value::Integer(version.into()),
                Value::Array(vec![]),
            ]))
            .unwrap(),
        );
    }
    let mut blobs = inventory(12);
    blobs.swap(0, 1);
    parity(&encode_directory(&blobs).unwrap());
    let blobs = vec![inventory(1)[0].clone(); 2];
    parity(&encode_directory(&blobs).unwrap());
    for bytes in [
        &[0x9f, 1, 0x80, 0xff][..],
        &[0x82, 0x18, 1, 0x80][..],
        &[
            0x82, 1, 0x9b, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        ][..],
    ] {
        parity(bytes);
    }
}

#[test]
fn oversized_inventory_keeps_original_cap_error_precedence() {
    let count = u32::try_from(MAX_BLOBS + 1).unwrap();
    let mut bytes = vec![0x82, 1, 0x9a];
    bytes.extend_from_slice(&count.to_be_bytes());
    // The original checks the inventory cap before individual tuple shape.
    bytes.resize(bytes.len() + count as usize, 0xf6);
    assert_eq!(
        decode_directory(&bytes),
        Err(TritPackageError::MetadataLimitExceeded)
    );
    parity(&bytes);
    // Floats need the original codec, including its canonical width rules.
    let mut floats = bytes[..7].to_vec();
    floats.extend_from_slice(&[0xf9, 0x38, 0]);
    floats.extend_from_slice(&bytes[8..]);
    assert_eq!(
        decode_directory(&floats),
        Err(TritPackageError::MetadataLimitExceeded)
    );
    parity(&floats);
}
