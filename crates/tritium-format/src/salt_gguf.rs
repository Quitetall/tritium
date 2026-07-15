//! SALT-in-GGUF container — a whole SALT model packaged in a GGUF envelope, the
//! alternative to the [`crate::write_salt_bundle`] sidecar for `tritium quantize
//! --format gguf`.
//!
//! Each SALT tensor becomes one GGUF tensor of the tritium-private type
//! [`GGML_TYPE_TRITIUM_SALT`], shaped `[k, rows]` (ggml order), whose payload is
//! the tensor's `rows` self-describing [`pack_salt_row`] blobs concatenated — the
//! same per-tensor blob the sidecar stores, just wrapped in GGUF instead of TSLB.
//! A `tritium.salt.format` metadata marker identifies the container.
//!
//! This is a *tritium* SALT container, not a drop-in llama.cpp model: the private
//! type id signals "not standard ggml". Readers also accept self-contained tritium
//! model GGUFs that mix private SALT matrices with sized standard tensors such as
//! F32 norms. The round-trip `read_salt_gguf(write_salt_gguf(..)) == input` is exact.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    FormatError, GgufFile, GgufValue, PackedSaltRow, PackedSaltTensor, SALT_HEADER_BYTES, SaltRow,
    SaltTensor, TensorInfo, TensorOut, packed_salt_row_len, read_gguf, unpack_packed_salt_row,
    write_gguf,
};

/// tritium-private ggml type-id for a SALT tensor (per-row TQ2_0 planes). Standard
/// ggml `enum ggml_type` occupies 0..=39; 169 is well clear, so a standard loader
/// sees an unknown type rather than misreading the bytes as a real ggml tensor.
pub const GGML_TYPE_TRITIUM_SALT: u32 = 169;

/// Metadata key marking a tritium SALT-in-GGUF container.
pub const SALT_GGUF_FORMAT_KEY: &str = "tritium.salt.format";

/// Value stored under [`SALT_GGUF_FORMAT_KEY`] (format + version tag).
pub const SALT_GGUF_FORMAT_VALUE: &str = "salt-rows.v1";

/// GGUF version the container is written as.
const GGUF_VERSION: u32 = 3;

/// Serialize a whole SALT model into a GGUF container.
///
/// Each entry is `(name, salt_rows)`; the row length `k` is taken from the first
/// row, so a tensor must have at least one row. Tensors are emitted in input order.
///
/// # Errors
/// [`FormatError::WrongBlockLen`] if a tensor has zero rows, any [`crate::pack_salt_row`]
/// error, [`FormatError::SaltInvalidScale`] if a scale is non-finite or negative
/// (including signed zero), or [`FormatError::Gguf`] if the GGUF layer rejects the result.
pub fn write_salt_gguf(tensors: &[(&str, &[SaltRow])]) -> Result<Vec<u8>, FormatError> {
    // Pack each tensor's rows into one contiguous blob (the per-tensor payload).
    let mut packed: Vec<(String, u64, u64, Vec<u8>)> = Vec::with_capacity(tensors.len());
    for (name, rows) in tensors {
        let k = rows
            .first()
            .map(|r| r.k)
            .ok_or(FormatError::WrongBlockLen {
                expected: 1,
                got: 0,
            })?;
        let mut blob = Vec::new();
        for row in *rows {
            blob.extend_from_slice(&pack_salt_row_checked(row, k)?);
        }
        packed.push((name.to_string(), k as u64, rows.len() as u64, blob));
    }

    let mut metadata = BTreeMap::new();
    metadata.insert("general.alignment".to_string(), GgufValue::U32(32));
    metadata.insert(
        SALT_GGUF_FORMAT_KEY.to_string(),
        GgufValue::String(SALT_GGUF_FORMAT_VALUE.to_string()),
    );

    // ggml dims are fastest-varying first: a [rows, k] matrix is dims = [k, rows].
    let outs: Vec<TensorOut<'_>> = packed
        .iter()
        .map(|(name, k, rows, blob)| TensorOut {
            name: name.clone(),
            dims: vec![*k, *rows],
            ggml_type: GGML_TYPE_TRITIUM_SALT,
            data: blob,
        })
        .collect();

    Ok(write_gguf(GGUF_VERSION, &metadata, &outs)?)
}

/// Pack one row, ensuring its `k` matches the tensor's declared `k`.
fn pack_salt_row_checked(row: &SaltRow, k: usize) -> Result<Vec<u8>, FormatError> {
    if row.k != k {
        return Err(FormatError::WrongBlockLen {
            expected: k,
            got: row.k,
        });
    }
    let encoded = crate::pack_salt_row(row)?;
    for plane in &row.planes {
        for block in plane.chunks_exact(crate::TQ2_0_BLOCK_BYTES) {
            let scale_offset = crate::TQ2_0_BLOCK_BYTES - 2;
            let bits = u16::from_le_bytes([block[scale_offset], block[scale_offset + 1]]);
            let scale = half::f16::from_bits(bits);
            if !scale.is_finite() || scale.is_sign_negative() {
                return Err(FormatError::SaltInvalidScale(bits));
            }
        }
    }
    Ok(encoded)
}

/// Parse a tritium SALT-in-GGUF container back into its tensors.
///
/// Reads the GGUF envelope, verifies the [`SALT_GGUF_FORMAT_KEY`] marker, then for
/// each [`GGML_TYPE_TRITIUM_SALT`] tensor walks its `rows` self-describing packed
/// rows. Sized non-SALT tensors are validated and ignored. Tensor names must be
/// unique; table offsets must match the canonical aligned layout; SALT walks are
/// bounded by the next tensor offset (EOF only for the final tensor); and every
/// inter-tensor padding byte must be zero. Corrupt or truncated input errors rather
/// than panicking.
///
/// # Errors
/// [`FormatError::Gguf`] on a malformed GGUF envelope; [`FormatError::SaltGgufBadFormat`]
/// if the marker is absent/wrong, a SALT tensor has other than 2 dims, tensor layout
/// is non-canonical, or an unsized non-SALT private type is present;
/// [`FormatError::WrongBlockLen`] on a truncated or length-mismatched payload, or
/// any [`crate::unpack_salt_row`] error.
pub fn read_salt_gguf(bytes: &[u8]) -> Result<Vec<SaltTensor>, FormatError> {
    read_salt_gguf_packed(bytes)?
        .into_iter()
        .map(|tensor| {
            let salt_rows = tensor
                .salt_rows
                .into_iter()
                .map(PackedSaltRow::into_dense)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(SaltTensor {
                name: tensor.name,
                rows: tensor.rows,
                k: tensor.k,
                salt_rows,
            })
        })
        .collect()
}

/// Parse a tritium SALT-in-GGUF container while preserving progressive sparse planes.
///
/// This has the same envelope, tensor-table, bounds, and padding validation as
/// [`read_salt_gguf`], but returns [`PackedSaltTensor`] rows instead of expanding sparse
/// residuals into dense TQ2_0 bytes.
///
/// # Errors
/// Same errors as [`read_salt_gguf`].
pub fn read_salt_gguf_packed(bytes: &[u8]) -> Result<Vec<PackedSaltTensor>, FormatError> {
    let f = read_gguf(bytes)?;
    if f.get_metadata(SALT_GGUF_FORMAT_KEY)
        .and_then(GgufValue::as_str)
        != Some(SALT_GGUF_FORMAT_VALUE)
    {
        return Err(FormatError::SaltGgufBadFormat);
    }

    validate_tensor_names_and_offsets(bytes, &f)?;

    let mut out = Vec::new();
    for (index, t) in f.tensors.iter().enumerate() {
        let blob = tensor_payload_interval(bytes, &f, index)?;
        if t.ggml_type != GGML_TYPE_TRITIUM_SALT {
            validate_sized_tensor_payload(&f, index, t, blob)?;
            continue;
        }
        if t.dims.len() != 2 {
            return Err(FormatError::SaltGgufBadFormat);
        }
        let k = usize::try_from(t.dims[0]).map_err(|_| FormatError::SaltGgufBadFormat)?;
        let rows = usize::try_from(t.dims[1]).map_err(|_| FormatError::SaltGgufBadFormat)?;
        let mut off = 0usize;
        // Each row is ≥ SALT_HEADER_BYTES; cap the reserve so a crafted `rows` cannot
        // preallocate unboundedly before the per-row bounds check below errors.
        let mut salt_rows = Vec::with_capacity(rows.min(blob.len() / SALT_HEADER_BYTES + 1));
        for _ in 0..rows {
            let remaining = blob.get(off..).ok_or(FormatError::WrongBlockLen {
                expected: off,
                got: blob.len(),
            })?;
            if remaining.len() < SALT_HEADER_BYTES {
                return Err(FormatError::WrongBlockLen {
                    expected: off + SALT_HEADER_BYTES,
                    got: blob.len(),
                });
            }
            let row_len = packed_salt_row_len(remaining)?;
            let end = off
                .checked_add(row_len)
                .ok_or(FormatError::SaltGgufBadFormat)?;
            let encoded = blob.get(off..end).ok_or(FormatError::WrongBlockLen {
                expected: end,
                got: blob.len(),
            })?;
            let row = unpack_packed_salt_row(encoded)?;
            if row.k() != k {
                return Err(FormatError::WrongBlockLen {
                    expected: k,
                    got: row.k(),
                });
            }
            salt_rows.push(row);
            off = end;
        }
        validate_payload_tail(&f, index, blob, off)?;
        out.push(PackedSaltTensor {
            name: t.name.clone(),
            rows,
            k,
            salt_rows,
        });
    }
    Ok(out)
}

/// Round `value` up to `alignment`, rejecting arithmetic overflow.
fn align_up(value: u64, alignment: u64) -> Result<u64, FormatError> {
    let bumped = value
        .checked_add(alignment - 1)
        .ok_or(FormatError::SaltGgufBadFormat)?;
    Ok(bumped - bumped % alignment)
}

/// Enforce the deterministic tensor table layout emitted by both GGUF writers.
///
/// Names are unique, the first tensor starts at relative offset zero, and every
/// subsequent offset is aligned and strictly after the preceding non-empty
/// payload. Exact adjacency is checked after the private SALT lengths are known.
fn validate_tensor_names_and_offsets(bytes: &[u8], f: &GgufFile) -> Result<(), FormatError> {
    let data_start =
        usize::try_from(f.tensor_data_offset).map_err(|_| FormatError::SaltGgufBadFormat)?;
    let data = bytes
        .get(data_start..)
        .ok_or(FormatError::SaltGgufBadFormat)?;
    if f.tensors.is_empty() {
        return if data.is_empty() {
            Ok(())
        } else {
            Err(FormatError::SaltGgufBadFormat)
        };
    }

    let alignment = f.alignment();
    let data_len = u64::try_from(data.len()).map_err(|_| FormatError::SaltGgufBadFormat)?;
    let mut names = BTreeSet::new();
    let mut previous = None;
    for t in &f.tensors {
        if !names.insert(t.name.as_str()) || t.offset % alignment != 0 || t.offset > data_len {
            return Err(FormatError::SaltGgufBadFormat);
        }
        if let Some(previous) = previous {
            if t.offset <= previous {
                return Err(FormatError::SaltGgufBadFormat);
            }
        } else if t.offset != 0 {
            return Err(FormatError::SaltGgufBadFormat);
        }
        previous = Some(t.offset);
    }
    Ok(())
}

/// Borrow exactly one table entry's physical interval. A non-final tensor ends
/// at the next table offset; only the final tensor is allowed to end at EOF.
fn tensor_payload_interval<'a>(
    bytes: &'a [u8],
    f: &GgufFile,
    index: usize,
) -> Result<&'a [u8], FormatError> {
    let tensor = f.tensors.get(index).ok_or(FormatError::SaltGgufBadFormat)?;
    let relative_end = f
        .tensors
        .get(index + 1)
        .map_or_else(
            || {
                u64::try_from(bytes.len())
                    .ok()
                    .and_then(|len| len.checked_sub(f.tensor_data_offset))
            },
            |next| Some(next.offset),
        )
        .ok_or(FormatError::SaltGgufBadFormat)?;
    let start = f
        .tensor_data_offset
        .checked_add(tensor.offset)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or(FormatError::SaltGgufBadFormat)?;
    let end = f
        .tensor_data_offset
        .checked_add(relative_end)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or(FormatError::SaltGgufBadFormat)?;
    bytes.get(start..end).ok_or(FormatError::SaltGgufBadFormat)
}

/// Sized standard tensors use the generic reader's exact byte count. Unknown
/// private types cannot be validated safely in a SALT model and are rejected.
fn validate_sized_tensor_payload(
    f: &GgufFile,
    index: usize,
    tensor: &TensorInfo,
    blob: &[u8],
) -> Result<(), FormatError> {
    let used = usize::try_from(tensor.n_bytes).map_err(|_| FormatError::SaltGgufBadFormat)?;
    if used == 0 {
        return Err(FormatError::SaltGgufBadFormat);
    }
    validate_payload_tail(f, index, blob, used)
}

/// Check one parsed SALT payload's exact end and its canonical alignment gap.
fn validate_payload_tail(
    f: &GgufFile,
    index: usize,
    blob: &[u8],
    used: usize,
) -> Result<(), FormatError> {
    if index + 1 == f.tensors.len() {
        if used != blob.len() {
            return Err(FormatError::WrongBlockLen {
                expected: used,
                got: blob.len(),
            });
        }
        return Ok(());
    }

    let tensor = &f.tensors[index];
    let used = u64::try_from(used).map_err(|_| FormatError::SaltGgufBadFormat)?;
    let payload_end = tensor
        .offset
        .checked_add(used)
        .ok_or(FormatError::SaltGgufBadFormat)?;
    let expected_next = align_up(payload_end, f.alignment())?;
    if f.tensors[index + 1].offset != expected_next {
        return Err(FormatError::SaltGgufBadFormat);
    }
    validate_payload_tail_for_alignment(
        blob,
        usize::try_from(used).map_err(|_| FormatError::SaltGgufBadFormat)?,
    )
}

/// Alignment bytes are canonical zeroes; they are never part of a tensor.
fn validate_payload_tail_for_alignment(blob: &[u8], used: usize) -> Result<(), FormatError> {
    let padding = blob.get(used..).ok_or(FormatError::WrongBlockLen {
        expected: used,
        got: blob.len(),
    })?;
    if padding.iter().any(|&byte| byte != 0) {
        return Err(FormatError::SaltGgufBadFormat);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TQ2_0_BLOCK_BYTES, dequant_salt_row, num_blocks, pack_salt_row, pack_tq2_0_row};
    use half::f16;
    use tritium_core::Trit;

    fn metadata(alignment: u32) -> BTreeMap<String, GgufValue> {
        BTreeMap::from([
            ("general.alignment".to_owned(), GgufValue::U32(alignment)),
            (
                SALT_GGUF_FORMAT_KEY.to_owned(),
                GgufValue::String(SALT_GGUF_FORMAT_VALUE.to_owned()),
            ),
        ])
    }

    fn f32_payload(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    /// Locate tensor offset fields in a writer-produced fixture. This deliberately
    /// understands only the two metadata value kinds emitted by `metadata` above.
    fn tensor_offset_fields(bytes: &[u8]) -> Vec<usize> {
        fn u32_at(bytes: &[u8], position: &mut usize) -> u32 {
            let end = *position + 4;
            let value = u32::from_le_bytes(bytes[*position..end].try_into().unwrap());
            *position = end;
            value
        }
        fn u64_at(bytes: &[u8], position: &mut usize) -> u64 {
            let end = *position + 8;
            let value = u64::from_le_bytes(bytes[*position..end].try_into().unwrap());
            *position = end;
            value
        }
        fn skip_string(bytes: &[u8], position: &mut usize) {
            let len = usize::try_from(u64_at(bytes, position)).unwrap();
            *position += len;
        }

        let tensor_count =
            usize::try_from(u64::from_le_bytes(bytes[8..16].try_into().unwrap())).unwrap();
        let metadata_count = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let mut position = 24;
        for _ in 0..metadata_count {
            skip_string(bytes, &mut position);
            match u32_at(bytes, &mut position) {
                4 => position += 4,
                8 => skip_string(bytes, &mut position),
                other => panic!("unexpected fixture metadata type {other}"),
            }
        }
        let mut fields = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            skip_string(bytes, &mut position);
            let dims = usize::try_from(u32_at(bytes, &mut position)).unwrap();
            position += dims * 8;
            position += 4;
            fields.push(position);
            position += 8;
        }
        fields
    }

    /// A SALT row of `t` planes over `k` trits (deterministic dummy data).
    fn row(k: usize, t: usize, seed: u8) -> SaltRow {
        let nb = num_blocks(k);
        let planes = (0..t)
            .map(|p| {
                let trits: Vec<Trit> = (0..k)
                    .map(|i| {
                        Trit::from_i8(((i as i32 + p as i32 + seed as i32) % 3 - 1) as i8).unwrap()
                    })
                    .collect();
                let scales = vec![f16::from_f32(0.5 + p as f32); nb];
                let mut bytes = vec![0u8; nb * TQ2_0_BLOCK_BYTES];
                pack_tq2_0_row(&trits, &scales, &mut bytes).unwrap();
                bytes
            })
            .collect();
        SaltRow { k, planes }
    }

    #[test]
    fn round_trips_exact() {
        // Two tensors, ragged plane counts and a non-256-multiple k.
        let a: Vec<SaltRow> = (0..3).map(|r| row(768, 2, r as u8)).collect();
        let b: Vec<SaltRow> = (0..2).map(|r| row(300, 3, (r + 7) as u8)).collect();
        let tensors: Vec<(&str, &[SaltRow])> =
            vec![("blk.0.attn.q.weight", &a), ("blk.0.mlp.up.weight", &b)];

        let bytes = write_salt_gguf(&tensors).expect("write");
        let got = read_salt_gguf(&bytes).expect("read");

        assert_eq!(got.len(), 2);
        assert_eq!(got[0].name, "blk.0.attn.q.weight");
        assert_eq!(got[0].rows, 3);
        assert_eq!(got[0].k, 768);
        assert_eq!(got[0].salt_rows, a, "tensor 0 rows must round-trip exactly");
        assert_eq!(got[1].name, "blk.0.mlp.up.weight");
        assert_eq!(got[1].salt_rows, b, "tensor 1 rows must round-trip exactly");
    }

    #[test]
    fn writer_rejects_scales_the_strict_reader_cannot_accept() {
        for bits in [f16::NAN.to_bits(), f16::INFINITY.to_bits(), 0xbc00, 0x8000] {
            let mut invalid = row(512, 2, 1);
            invalid.planes[1][2 * TQ2_0_BLOCK_BYTES - 2..].copy_from_slice(&bits.to_le_bytes());
            assert_eq!(
                write_salt_gguf(&[("invalid.weight", &[invalid])]),
                Err(FormatError::SaltInvalidScale(bits))
            );
        }

        let mut positive_zero = row(512, 2, 1);
        positive_zero.planes[1][2 * TQ2_0_BLOCK_BYTES - 2..]
            .copy_from_slice(&f16::ZERO.to_bits().to_le_bytes());
        assert!(write_salt_gguf(&[("zero.weight", &[positive_zero])]).is_ok());
    }

    #[test]
    fn parses_as_valid_gguf_with_marker() {
        let a: Vec<SaltRow> = vec![row(512, 1, 1)];
        let bytes = write_salt_gguf(&[("w", &a)]).expect("write");

        // It's a real GGUF: the generic reader parses it and sees the marker + tensor.
        let f = read_gguf(&bytes).expect("generic gguf parse");
        assert_eq!(
            f.get_metadata(SALT_GGUF_FORMAT_KEY)
                .and_then(GgufValue::as_str),
            Some(SALT_GGUF_FORMAT_VALUE)
        );
        let t = f.tensor("w").expect("tensor present");
        assert_eq!(t.ggml_type, GGML_TYPE_TRITIUM_SALT);
        assert_eq!(t.dims, vec![512, 1]); // [k, rows]
    }

    #[test]
    fn dequant_matches_after_round_trip() {
        let a: Vec<SaltRow> = (0..4).map(|r| row(256, 2, r as u8)).collect();
        let want: Vec<Vec<f32>> = a.iter().map(|r| dequant_salt_row(r).unwrap()).collect();

        let bytes = write_salt_gguf(&[("w", &a)]).expect("write");
        let got = read_salt_gguf(&bytes).expect("read");

        for (r, w) in got[0].salt_rows.iter().zip(&want) {
            assert_eq!(&dequant_salt_row(r).unwrap(), w);
        }
    }

    #[test]
    fn mixed_salt_and_f32_tensors_use_disjoint_bounded_payloads() {
        let salt_row = row(256, 1, 9);
        let salt = pack_salt_row(&salt_row).unwrap();
        let norm_a = f32_payload(&[1.0, 2.0, 3.0, 4.0]);
        let norm_b = f32_payload(&[5.0, 6.0]);
        let tensors = [
            TensorOut {
                name: "input_norm.weight".to_owned(),
                dims: vec![4],
                ggml_type: 0,
                data: &norm_a,
            },
            TensorOut {
                name: "model.layers.0.mlp.up_proj.weight".to_owned(),
                dims: vec![256, 1],
                ggml_type: GGML_TYPE_TRITIUM_SALT,
                data: &salt,
            },
            TensorOut {
                name: "output_norm.weight".to_owned(),
                dims: vec![2],
                ggml_type: 0,
                data: &norm_b,
            },
        ];
        let bytes = write_gguf(3, &metadata(32), &tensors).unwrap();

        let got = read_salt_gguf(&bytes).expect("mixed self-contained model");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].name, "model.layers.0.mlp.up_proj.weight");
        assert_eq!(got[0].salt_rows, vec![salt_row]);
    }

    #[test]
    fn salt_row_count_cannot_consume_a_following_f32_tensor() {
        // Three planes make this row 208 bytes, preserving exact adjacency under
        // the minimum official eight-byte GGUF alignment.
        let first = pack_salt_row(&row(256, 3, 1)).unwrap();
        // A valid encoded SALT row is also a multiple of four bytes. Present those
        // bytes as an F32 tensor so an EOF-bounded SALT walk would accept it as the
        // forged second row. The table boundary must stop that walk first.
        let disguised_following = pack_salt_row(&row(256, 1, 2)).unwrap();
        assert!(disguised_following.len().is_multiple_of(4));
        let tensors = [
            TensorOut {
                name: "salt".to_owned(),
                dims: vec![256, 2],
                ggml_type: GGML_TYPE_TRITIUM_SALT,
                data: &first,
            },
            TensorOut {
                name: "norm".to_owned(),
                dims: vec![(disguised_following.len() / 4) as u64],
                ggml_type: 0,
                data: &disguised_following,
            },
        ];
        let bytes = write_gguf(3, &metadata(8), &tensors).unwrap();

        assert!(read_gguf(&bytes).is_ok(), "generic envelope remains valid");
        assert!(read_salt_gguf(&bytes).is_err());
    }

    #[test]
    fn rejects_row_k_that_disagrees_with_tensor_shape() {
        let encoded = pack_salt_row(&row(256, 1, 1)).unwrap();
        let tensors = [TensorOut {
            name: "salt".to_owned(),
            dims: vec![255, 1],
            ggml_type: GGML_TYPE_TRITIUM_SALT,
            data: &encoded,
        }];
        let bytes = write_gguf(3, &metadata(32), &tensors).unwrap();

        assert_eq!(
            read_salt_gguf(&bytes).unwrap_err(),
            FormatError::WrongBlockLen {
                expected: 255,
                got: 256,
            }
        );
    }

    #[test]
    fn rejects_duplicate_tensor_names() {
        let encoded = pack_salt_row(&row(256, 1, 1)).unwrap();
        let norm = f32_payload(&[1.0]);
        let tensors = [
            TensorOut {
                name: "duplicate".to_owned(),
                dims: vec![256, 1],
                ggml_type: GGML_TYPE_TRITIUM_SALT,
                data: &encoded,
            },
            TensorOut {
                name: "duplicate".to_owned(),
                dims: vec![1],
                ggml_type: 0,
                data: &norm,
            },
        ];
        let bytes = write_gguf(3, &metadata(32), &tensors).unwrap();

        assert_eq!(
            read_salt_gguf(&bytes).unwrap_err(),
            FormatError::SaltGgufBadFormat
        );
    }

    #[test]
    fn rejects_overlapping_or_non_monotonic_tensor_offsets() {
        let encoded = pack_salt_row(&row(256, 1, 1)).unwrap();
        let norm = f32_payload(&[1.0, 2.0]);
        let tensors = [
            TensorOut {
                name: "salt".to_owned(),
                dims: vec![256, 1],
                ggml_type: GGML_TYPE_TRITIUM_SALT,
                data: &encoded,
            },
            TensorOut {
                name: "norm".to_owned(),
                dims: vec![2],
                ggml_type: 0,
                data: &norm,
            },
        ];
        let mut bytes = write_gguf(3, &metadata(32), &tensors).unwrap();
        let fields = tensor_offset_fields(&bytes);
        bytes[fields[1]..fields[1] + 8].copy_from_slice(&0u64.to_le_bytes());

        assert!(
            read_gguf(&bytes).is_ok(),
            "generic reader does not own overlap policy"
        );
        assert_eq!(
            read_salt_gguf(&bytes).unwrap_err(),
            FormatError::SaltGgufBadFormat
        );
    }

    #[test]
    fn rejects_unaligned_tensor_offsets_even_when_payload_is_in_bounds() {
        let encoded = pack_salt_row(&row(256, 1, 1)).unwrap();
        let norm = f32_payload(&[1.0, 2.0]);
        let tensors = [
            TensorOut {
                name: "salt".to_owned(),
                dims: vec![256, 1],
                ggml_type: GGML_TYPE_TRITIUM_SALT,
                data: &encoded,
            },
            TensorOut {
                name: "norm".to_owned(),
                dims: vec![2],
                ggml_type: 0,
                data: &norm,
            },
        ];
        let mut bytes = write_gguf(3, &metadata(32), &tensors).unwrap();
        let parsed = read_gguf(&bytes).unwrap();
        let original_offset = parsed.tensors[1].offset;
        let original_start = usize::try_from(parsed.tensor_data_offset + original_offset).unwrap();
        bytes.insert(original_start, 0);
        let fields = tensor_offset_fields(&bytes);
        bytes[fields[1]..fields[1] + 8].copy_from_slice(&(original_offset + 1).to_le_bytes());

        assert!(read_gguf(&bytes).is_ok());
        assert_eq!(
            read_salt_gguf(&bytes).unwrap_err(),
            FormatError::SaltGgufBadFormat
        );
    }

    #[test]
    fn rejects_aligned_but_noncanonical_extra_gap() {
        let encoded = pack_salt_row(&row(256, 1, 1)).unwrap();
        let norm = f32_payload(&[1.0, 2.0]);
        let tensors = [
            TensorOut {
                name: "salt".to_owned(),
                dims: vec![256, 1],
                ggml_type: GGML_TYPE_TRITIUM_SALT,
                data: &encoded,
            },
            TensorOut {
                name: "norm".to_owned(),
                dims: vec![2],
                ggml_type: 0,
                data: &norm,
            },
        ];
        let mut bytes = write_gguf(3, &metadata(32), &tensors).unwrap();
        let parsed = read_gguf(&bytes).unwrap();
        let original_offset = parsed.tensors[1].offset;
        let original_start = usize::try_from(parsed.tensor_data_offset + original_offset).unwrap();
        bytes.splice(original_start..original_start, [0; 32]);
        let fields = tensor_offset_fields(&bytes);
        bytes[fields[1]..fields[1] + 8].copy_from_slice(&(original_offset + 32).to_le_bytes());

        assert!(read_gguf(&bytes).is_ok());
        assert_eq!(
            read_salt_gguf(&bytes).unwrap_err(),
            FormatError::SaltGgufBadFormat
        );
    }

    #[test]
    fn rejects_nonzero_inter_tensor_padding() {
        let encoded = pack_salt_row(&row(256, 1, 1)).unwrap();
        let norm = f32_payload(&[1.0]);
        let tensors = [
            TensorOut {
                name: "salt".to_owned(),
                dims: vec![256, 1],
                ggml_type: GGML_TYPE_TRITIUM_SALT,
                data: &encoded,
            },
            TensorOut {
                name: "norm".to_owned(),
                dims: vec![1],
                ggml_type: 0,
                data: &norm,
            },
        ];
        let mut bytes = write_gguf(3, &metadata(32), &tensors).unwrap();
        let f = read_gguf(&bytes).unwrap();
        let padding_byte = usize::try_from(f.tensor_data_offset).unwrap() + encoded.len();
        assert!(
            padding_byte < usize::try_from(f.tensor_data_offset + f.tensors[1].offset).unwrap()
        );
        bytes[padding_byte] = 1;

        assert_eq!(
            read_salt_gguf(&bytes).unwrap_err(),
            FormatError::SaltGgufBadFormat
        );
    }

    #[test]
    fn rejects_bytes_after_the_final_tensor() {
        let rows = vec![row(256, 1, 1)];
        let mut bytes = write_salt_gguf(&[("salt", &rows)]).unwrap();
        bytes.push(0);

        assert_eq!(
            read_salt_gguf(&bytes).unwrap_err(),
            FormatError::WrongBlockLen {
                expected: pack_salt_row(&rows[0]).unwrap().len(),
                got: pack_salt_row(&rows[0]).unwrap().len() + 1,
            }
        );
    }

    #[test]
    fn rejects_non_salt_gguf() {
        // A valid GGUF without the marker is rejected as not-a-SALT-container.
        let meta = BTreeMap::new();
        let plain = write_gguf(3, &meta, &[]).expect("write plain gguf");
        assert_eq!(
            read_salt_gguf(&plain).unwrap_err(),
            FormatError::SaltGgufBadFormat
        );
    }

    #[test]
    fn truncated_payload_errors_not_panics() {
        let a: Vec<SaltRow> = (0..3).map(|r| row(512, 2, r as u8)).collect();
        let full = write_salt_gguf(&[("w", &a)]).expect("write");
        // Lopping bytes off the end must error cleanly at the row walk, never panic.
        for cut in 1..40 {
            let _ = read_salt_gguf(&full[..full.len() - cut]);
        }
    }
}
