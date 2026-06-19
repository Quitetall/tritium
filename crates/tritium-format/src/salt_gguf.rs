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
//! type id signals "not standard ggml", and a safetensors source carries none of
//! the architecture metadata a runnable GGUF needs. The round-trip
//! `read_salt_gguf(write_salt_gguf(..)) == input` is exact.

use std::collections::BTreeMap;

use crate::{
    FormatError, GgufValue, SALT_HEADER_BYTES, SaltRow, SaltTensor, TQ2_0_BLOCK_BYTES, TensorOut,
    num_blocks, read_gguf, unpack_salt_row, write_gguf,
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
/// [`FormatError::WrongBlockLen`] if a tensor has zero rows, any [`pack_salt_row`]
/// error, or [`FormatError::Gguf`] if the GGUF layer rejects the result.
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
    crate::pack_salt_row(row)
}

/// Parse a tritium SALT-in-GGUF container back into its tensors.
///
/// Reads the GGUF envelope, verifies the [`SALT_GGUF_FORMAT_KEY`] marker, then for
/// each [`GGML_TYPE_TRITIUM_SALT`] tensor walks its `rows` self-describing packed
/// rows. Non-SALT tensors are ignored. Every field is bounds-checked; corrupt or
/// truncated input errors rather than panicking.
///
/// # Errors
/// [`FormatError::Gguf`] on a malformed GGUF envelope; [`FormatError::SaltGgufBadFormat`]
/// if the marker is absent/wrong or a SALT tensor has other than 2 dims;
/// [`FormatError::WrongBlockLen`] on a truncated payload, or any [`unpack_salt_row`] error.
pub fn read_salt_gguf(bytes: &[u8]) -> Result<Vec<SaltTensor>, FormatError> {
    let f = read_gguf(bytes)?;
    if f.get_metadata(SALT_GGUF_FORMAT_KEY)
        .and_then(GgufValue::as_str)
        != Some(SALT_GGUF_FORMAT_VALUE)
    {
        return Err(FormatError::SaltGgufBadFormat);
    }

    let mut out = Vec::new();
    for t in &f.tensors {
        if t.ggml_type != GGML_TYPE_TRITIUM_SALT {
            continue;
        }
        if t.dims.len() != 2 {
            return Err(FormatError::SaltGgufBadFormat);
        }
        let k = usize::try_from(t.dims[0]).map_err(|_| FormatError::SaltGgufBadFormat)?;
        let rows = usize::try_from(t.dims[1]).map_err(|_| FormatError::SaltGgufBadFormat)?;

        // The payload lives at `tensor_data_offset + offset`; its length is not in
        // the reader (private type sizes to 0), so we walk the `rows` self-describing
        // rows from there, bounds-checking each against the buffer end.
        let start = f
            .tensor_data_offset
            .checked_add(t.offset)
            .and_then(|s| usize::try_from(s).ok())
            .ok_or(FormatError::SaltGgufBadFormat)?;
        let blob = bytes.get(start..).ok_or(FormatError::SaltGgufBadFormat)?;

        // `k` comes from a u64 GGUF dim (uncapped on 64-bit, unlike the bundle's
        // u32 `k`). `num_blocks(k) = ⌈k/256⌉ ≤ k/256`, so `·66` cannot wrap usize
        // for any u64 `k` — but check it anyway so the guard stays sound if the
        // block constant or `k` source ever changes (the bundle relies on its u32
        // cap for the same invariant).
        let plane_bytes = num_blocks(k)
            .checked_mul(TQ2_0_BLOCK_BYTES)
            .ok_or(FormatError::SaltGgufBadFormat)?;
        let mut off = 0usize;
        // Each row is ≥ SALT_HEADER_BYTES; cap the reserve so a crafted `rows` cannot
        // preallocate unboundedly before the per-row bounds check below errors.
        let mut salt_rows = Vec::with_capacity(rows.min(blob.len() / SALT_HEADER_BYTES + 1));
        for _ in 0..rows {
            if off + SALT_HEADER_BYTES > blob.len() {
                return Err(FormatError::WrongBlockLen {
                    expected: off + SALT_HEADER_BYTES,
                    got: blob.len(),
                });
            }
            // Plane count `T` sits at header byte 5 (magic[4] + version[1]).
            let t_planes = blob[off + 5] as usize;
            let row_len = t_planes
                .checked_mul(plane_bytes)
                .and_then(|p| p.checked_add(SALT_HEADER_BYTES))
                .ok_or(FormatError::WrongBlockLen {
                    expected: usize::MAX,
                    got: blob.len(),
                })?;
            if off + row_len > blob.len() {
                return Err(FormatError::WrongBlockLen {
                    expected: off + row_len,
                    got: blob.len(),
                });
            }
            salt_rows.push(unpack_salt_row(&blob[off..off + row_len])?);
            off += row_len;
        }
        out.push(SaltTensor {
            name: t.name.clone(),
            rows,
            k,
            salt_rows,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dequant_salt_row, pack_tq2_0_row};
    use half::f16;
    use tritium_core::Trit;

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
