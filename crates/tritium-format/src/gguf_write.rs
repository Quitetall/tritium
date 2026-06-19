//! GGUF v3 container *writer* — the mirror of [`crate::read_gguf`].
//!
//! Serializes a metadata table and a set of tensor payloads into a byte buffer
//! that [`crate::read_gguf`] parses back identically. Every multibyte integer is
//! little-endian. Tensor payloads are laid out sequentially in the data section,
//! each at an offset rounded up to the file alignment (`general.alignment`, or
//! [`crate::DEFAULT_ALIGNMENT`] when absent) — the convention ggml/llama.cpp use.
//!
//! The writer is the inverse of the reader on the round-trip path: for any
//! metadata map and tensor set the reader accepts,
//! `read_gguf(write_gguf(..)) == (metadata, tensors)`.

use std::collections::BTreeMap;

use crate::gguf::{DEFAULT_ALIGNMENT, GgufError, GgufValue};

/// Append a GGUF string: `u64` little-endian byte-length, then the UTF-8 bytes.
fn push_gguf_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// The `value_type` tag the reader expects for each [`GgufValue`] variant.
fn value_type_tag(v: &GgufValue) -> u32 {
    match v {
        GgufValue::U8(_) => 0,
        GgufValue::I8(_) => 1,
        GgufValue::U16(_) => 2,
        GgufValue::I16(_) => 3,
        GgufValue::U32(_) => 4,
        GgufValue::I32(_) => 5,
        GgufValue::F32(_) => 6,
        GgufValue::Bool(_) => 7,
        GgufValue::String(_) => 8,
        GgufValue::Array(_) => 9,
        GgufValue::U64(_) => 10,
        GgufValue::I64(_) => 11,
        GgufValue::F64(_) => 12,
    }
}

/// Append a scalar value (everything except [`GgufValue::Array`]) in the exact
/// little-endian encoding [`crate::read_gguf`]'s `read_value` consumes.
///
/// Returns [`GgufError::UnknownValueType`]`(9)` for a nested array, which GGUF
/// forbids and the reader rejects — keeping the writer round-trip-faithful.
fn push_scalar(out: &mut Vec<u8>, v: &GgufValue) -> Result<(), GgufError> {
    match v {
        GgufValue::U8(x) => out.push(*x),
        GgufValue::I8(x) => out.push(*x as u8),
        GgufValue::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::F32(x) => out.extend_from_slice(&x.to_bits().to_le_bytes()),
        GgufValue::Bool(b) => out.push(u8::from(*b)),
        GgufValue::String(s) => push_gguf_string(out, s),
        GgufValue::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::F64(x) => out.extend_from_slice(&x.to_bits().to_le_bytes()),
        GgufValue::Array(_) => return Err(GgufError::UnknownValueType(9)),
    }
    Ok(())
}

/// Append a full metadata value: its `value_type` tag, then the payload. For an
/// array, the element `value_type`, the `u64` count, then each element (which
/// must itself be a scalar — GGUF arrays do not nest).
fn push_value(out: &mut Vec<u8>, v: &GgufValue) -> Result<(), GgufError> {
    out.extend_from_slice(&value_type_tag(v).to_le_bytes());
    match v {
        GgufValue::Array(items) => {
            // An empty array carries no element to infer the child type from;
            // U8 (tag 0) round-trips structurally since there are no elements.
            let child_tag = items.first().map_or(0, value_type_tag);
            out.extend_from_slice(&child_tag.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                if value_type_tag(item) != child_tag {
                    // Heterogeneous arrays are unrepresentable in GGUF.
                    return Err(GgufError::UnknownValueType(9));
                }
                push_scalar(out, item)?;
            }
            Ok(())
        }
        scalar => push_scalar(out, scalar),
    }
}

/// Round `pos` up to the next multiple of `align` (`align` assumed non-zero),
/// overflow-checked. Mirrors the reader's `align_up`.
fn align_up(pos: u64, align: u64) -> Result<u64, GgufError> {
    if align == 0 {
        return Ok(pos);
    }
    let bumped = pos.checked_add(align - 1).ok_or(GgufError::DimsOverflow)?;
    Ok(bumped - (bumped % align))
}

/// One tensor to emit: name, shape (ggml order, fastest-varying first), ggml
/// type-id, and the already-packed payload bytes.
///
/// `data.len()` should equal the byte size the reader computes from `dims` and
/// `ggml_type` for sized types; the writer lays the next tensor after it either
/// way, but a short payload would make the reader's bounds check read into the
/// padding.
#[derive(Debug)]
pub struct TensorOut<'a> {
    /// Tensor name (written as a GGUF string).
    pub name: String,
    /// Shape, fastest-varying dimension first.
    pub dims: Vec<u64>,
    /// ggml type-id (e.g. [`crate::GGML_TYPE_TQ2_0`]).
    pub ggml_type: u32,
    /// Packed payload bytes.
    pub data: &'a [u8],
}

/// Serialize a GGUF v2/v3 container.
///
/// `version` must be 2 or 3. Tensor offsets are assigned sequentially, each
/// aligned up to the effective alignment (from `general.alignment` in
/// `metadata`, else [`crate::DEFAULT_ALIGNMENT`]).
///
/// # Errors
/// - [`GgufError::UnsupportedVersion`] if `version` is not 2 or 3.
/// - [`GgufError::DimsOverflow`] on offset/length arithmetic overflow.
pub fn write_gguf(
    version: u32,
    metadata: &BTreeMap<String, GgufValue>,
    tensors: &[TensorOut<'_>],
) -> Result<Vec<u8>, GgufError> {
    if version != 2 && version != 3 {
        return Err(GgufError::UnsupportedVersion(version));
    }

    // Effective alignment: `general.alignment` if present and non-zero, else default.
    let alignment = metadata
        .get("general.alignment")
        .and_then(GgufValue::as_u64)
        .filter(|&a| a != 0)
        .unwrap_or(DEFAULT_ALIGNMENT);

    // Assign each tensor a data-section-relative offset, sequential and aligned.
    // `rel` tracks the running end of the data section.
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut rel: u64 = 0;
    for t in tensors {
        let off = align_up(rel, alignment)?;
        offsets.push(off);
        rel = off
            .checked_add(t.data.len() as u64)
            .ok_or(GgufError::DimsOverflow)?;
    }

    // Header.
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

    // Metadata table — BTreeMap iterates in key order; the reader re-inserts into
    // a BTreeMap, so any order round-trips, and key order keeps output deterministic.
    for (key, value) in metadata {
        push_gguf_string(&mut out, key);
        push_value(&mut out, value)?;
    }

    // Tensor-info table.
    for (t, &off) in tensors.iter().zip(&offsets) {
        push_gguf_string(&mut out, &t.name);
        out.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for &d in &t.dims {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&t.ggml_type.to_le_bytes());
        out.extend_from_slice(&off.to_le_bytes());
    }

    // Pad up to the aligned data-section start.
    let header_end = out.len() as u64;
    let tensor_data_offset = align_up(header_end, alignment)?;
    out.resize(tensor_data_offset as usize, 0);

    // Data section: each payload at `tensor_data_offset + offset`, zero-padding gaps.
    for (t, &off) in tensors.iter().zip(&offsets) {
        let abs = tensor_data_offset
            .checked_add(off)
            .ok_or(GgufError::DimsOverflow)? as usize;
        if out.len() < abs {
            out.resize(abs, 0);
        }
        out.extend_from_slice(t.data);
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GGML_TYPE_TQ2_0, read_gguf};

    #[test]
    fn round_trips_metadata_and_tensors() {
        let mut meta = BTreeMap::new();
        meta.insert(
            "general.architecture".to_string(),
            GgufValue::String("llama".to_string()),
        );
        meta.insert("general.alignment".to_string(), GgufValue::U32(32));
        meta.insert(
            "test.array".to_string(),
            GgufValue::Array(vec![
                GgufValue::U32(1),
                GgufValue::U32(2),
                GgufValue::U32(3),
            ]),
        );

        // Two TQ2_0 tensors of 256 elements => 66 bytes each.
        let w0: Vec<u8> = (0..66u8).collect();
        let w1: Vec<u8> = (0..66u8).map(|b| b ^ 0xAB).collect();
        let tensors = vec![
            TensorOut {
                name: "blk.0.weight".to_string(),
                dims: vec![256],
                ggml_type: GGML_TYPE_TQ2_0,
                data: &w0,
            },
            TensorOut {
                name: "blk.1.weight".to_string(),
                dims: vec![256],
                ggml_type: GGML_TYPE_TQ2_0,
                data: &w1,
            },
        ];

        let bytes = write_gguf(3, &meta, &tensors).expect("write");
        let parsed = read_gguf(&bytes).expect("read back");

        assert_eq!(parsed.version, 3);
        assert_eq!(parsed.metadata, meta, "metadata must round-trip exactly");
        assert_eq!(parsed.tensors.len(), 2);

        // Payloads land at their reported offsets, byte-identical.
        for (t, want) in parsed.tensors.iter().zip([&w0, &w1]) {
            let start = (parsed.tensor_data_offset + t.offset) as usize;
            let got = &bytes[start..start + want.len()];
            assert_eq!(got, want.as_slice(), "{} payload mismatch", t.name);
            assert_eq!(t.offset % parsed.alignment(), 0, "offset must be aligned");
        }
    }

    #[test]
    fn every_scalar_value_type_round_trips() {
        let mut meta = BTreeMap::new();
        meta.insert("a.u8".into(), GgufValue::U8(7));
        meta.insert("a.i8".into(), GgufValue::I8(-7));
        meta.insert("a.u16".into(), GgufValue::U16(0xBEEF));
        meta.insert("a.i16".into(), GgufValue::I16(-12345));
        meta.insert("a.u32".into(), GgufValue::U32(0xDEAD_BEEF));
        meta.insert("a.i32".into(), GgufValue::I32(-2_000_000_000));
        meta.insert("a.f32".into(), GgufValue::F32(3.5));
        meta.insert("a.bool_t".into(), GgufValue::Bool(true));
        meta.insert("a.bool_f".into(), GgufValue::Bool(false));
        meta.insert("a.string".into(), GgufValue::String("héllo".into()));
        meta.insert("a.u64".into(), GgufValue::U64(0x0123_4567_89AB_CDEF));
        meta.insert("a.i64".into(), GgufValue::I64(-9_000_000_000));
        meta.insert("a.f64".into(), GgufValue::F64(-42.125));

        let bytes = write_gguf(3, &meta, &[]).expect("write");
        let parsed = read_gguf(&bytes).expect("read");
        assert_eq!(parsed.metadata, meta);
    }

    #[test]
    fn custom_alignment_64_respected() {
        let mut meta = BTreeMap::new();
        meta.insert("general.alignment".into(), GgufValue::U32(64));
        let w0: Vec<u8> = vec![1; 66];
        let w1: Vec<u8> = vec![2; 66];
        let tensors = vec![
            TensorOut {
                name: "t0".into(),
                dims: vec![256],
                ggml_type: GGML_TYPE_TQ2_0,
                data: &w0,
            },
            TensorOut {
                name: "t1".into(),
                dims: vec![256],
                ggml_type: GGML_TYPE_TQ2_0,
                data: &w1,
            },
        ];
        let bytes = write_gguf(3, &meta, &tensors).expect("write");
        let parsed = read_gguf(&bytes).expect("read");
        assert_eq!(parsed.alignment(), 64);
        assert_eq!(parsed.tensor_data_offset % 64, 0);
        // Second tensor's offset rounds 66 up to 128 (next multiple of 64).
        assert_eq!(parsed.tensors[1].offset, 128);
        let s = (parsed.tensor_data_offset + parsed.tensors[1].offset) as usize;
        assert_eq!(&bytes[s..s + 66], w1.as_slice());
    }

    #[test]
    fn empty_array_round_trips() {
        let mut meta = BTreeMap::new();
        meta.insert("a.empty".into(), GgufValue::Array(vec![]));
        let bytes = write_gguf(3, &meta, &[]).expect("write");
        let parsed = read_gguf(&bytes).expect("read");
        match parsed.get_metadata("a.empty") {
            Some(GgufValue::Array(v)) => assert!(v.is_empty()),
            other => panic!("expected empty array, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_version_rejected() {
        let meta = BTreeMap::new();
        assert_eq!(
            write_gguf(1, &meta, &[]).unwrap_err(),
            GgufError::UnsupportedVersion(1)
        );
        assert!(write_gguf(2, &meta, &[]).is_ok());
    }

    #[test]
    fn nested_array_rejected() {
        let mut meta = BTreeMap::new();
        meta.insert(
            "a.nested".into(),
            GgufValue::Array(vec![GgufValue::Array(vec![GgufValue::U8(1)])]),
        );
        assert!(write_gguf(3, &meta, &[]).is_err());
    }
}
