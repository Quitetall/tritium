//! Whole-model **SALT bundle** — a single-file container of many [`SaltRow`]s, the
//! artifact `tritium quantize` writes (ADR 0006).
//!
//! [`pack_salt_row`](crate::pack_salt_row)/[`unpack_salt_row`](crate::unpack_salt_row)
//! serialize one row; a real model has thousands across many tensors. The bundle wraps
//! them with a per-tensor index so a loader can find a named tensor's rows. Its payloads
//! may mix legacy dense v1 rows and progressive dense-or-sparse v2 rows.
//!
//! Layout (little-endian):
//! ```text
//! magic b"TSLB" (4) | version u8 | _reserved u8 | tensor_count u32
//! index[tensor_count]: name_len u16 | name utf8 | rows u32 | k u32 | data_len u64
//! data: per tensor, `rows` self-describing SALT rows concatenated (data_len bytes total)
//! ```
//! Each row is self-describing (its own [`SALT_HEADER_BYTES`] header carries `T` and `k`),
//! so the reader walks a tensor's `rows` packed rows without per-row offsets.

use crate::{
    FormatError, SALT_HEADER_BYTES, SaltRow, pack_progressive_salt_row, pack_salt_row,
    packed_salt_row_len, unpack_salt_row_prefix,
};

/// Bundle magic: `b"TSLB"` (Tritium SALT Bundle).
pub const SALT_BUNDLE_MAGIC: [u8; 4] = *b"TSLB";

/// Current bundle format version.
pub const SALT_BUNDLE_VERSION: u8 = 1;

/// One tensor recovered from a bundle: its name, shape (`rows × k`), and SALT rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SaltTensor {
    /// Tensor name (e.g. `"model.layers.0.self_attn.q_proj.weight"`).
    pub name: String,
    /// Output channels (rows of the original matrix).
    pub rows: usize,
    /// Input features per row.
    pub k: usize,
    /// One [`SaltRow`] per output channel; `rows.len() == rows`.
    pub salt_rows: Vec<SaltRow>,
}

/// Serialize a whole model to a SALT bundle. Each entry is `(name, salt_rows)`; the row
/// length `k` is taken from the rows (so a tensor must have at least one row).
///
/// # Errors
/// [`FormatError::WrongBlockLen`] if a tensor has zero rows, inconsistent row lengths,
/// or a count/name exceeds its fixed-width index field; plus any underlying
/// [`pack_salt_row`] error.
pub fn write_salt_bundle(tensors: &[(&str, &[SaltRow])]) -> Result<Vec<u8>, FormatError> {
    write_salt_bundle_with(tensors, pack_salt_row)
}

/// Serialize a whole model using progressive v2 rows.
///
/// Base planes remain dense. Residual planes below `max_sparse_density` use the
/// sparse sidecar only when it is physically smaller. [`read_salt_bundle`] expands
/// the result into ordinary [`SaltRow`] values for existing runtimes.
///
/// # Errors
/// Same errors as [`write_salt_bundle`] and [`pack_progressive_salt_row`].
pub fn write_progressive_salt_bundle(
    tensors: &[(&str, &[SaltRow])],
    max_sparse_density: f32,
) -> Result<Vec<u8>, FormatError> {
    write_salt_bundle_with(tensors, |row| {
        pack_progressive_salt_row(row, max_sparse_density)
    })
}

fn write_salt_bundle_with<F>(
    tensors: &[(&str, &[SaltRow])],
    pack_row: F,
) -> Result<Vec<u8>, FormatError>
where
    F: Fn(&SaltRow) -> Result<Vec<u8>, FormatError>,
{
    struct PackedTensor {
        name_len: u16,
        rows: u32,
        k: u32,
        data_len: u64,
        blob: Vec<u8>,
    }

    let tensor_count = u32::try_from(tensors.len()).map_err(|_| FormatError::WrongBlockLen {
        expected: u32::MAX as usize,
        got: tensors.len(),
    })?;
    // Pack each tensor's rows into one contiguous blob first (so we know its data_len).
    let mut packed_tensors = Vec::with_capacity(tensors.len());
    for (name, rows) in tensors {
        let name_len = u16::try_from(name.len()).map_err(|_| FormatError::WrongBlockLen {
            expected: u16::MAX as usize,
            got: name.len(),
        })?;
        let row_count = u32::try_from(rows.len()).map_err(|_| FormatError::WrongBlockLen {
            expected: u32::MAX as usize,
            got: rows.len(),
        })?;
        let k = rows
            .first()
            .map(|r| r.k)
            .ok_or(FormatError::WrongBlockLen {
                expected: 1,
                got: 0,
            })?;
        let encoded_k = u32::try_from(k).map_err(|_| FormatError::SaltRowTooLong(k))?;
        let mut blob = Vec::new();
        for row in *rows {
            if row.k != k {
                return Err(FormatError::WrongBlockLen {
                    expected: k,
                    got: row.k,
                });
            }
            blob.extend_from_slice(&pack_row(row)?);
        }
        let data_len = u64::try_from(blob.len()).map_err(|_| FormatError::WrongBlockLen {
            expected: usize::MAX,
            got: blob.len(),
        })?;
        packed_tensors.push(PackedTensor {
            name_len,
            rows: row_count,
            k: encoded_k,
            data_len,
            blob,
        });
    }

    let mut out = Vec::new();
    out.extend_from_slice(&SALT_BUNDLE_MAGIC);
    out.push(SALT_BUNDLE_VERSION);
    out.push(0); // reserved
    out.extend_from_slice(&tensor_count.to_le_bytes());
    for ((name, _), packed) in tensors.iter().zip(&packed_tensors) {
        out.extend_from_slice(&packed.name_len.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&packed.rows.to_le_bytes());
        out.extend_from_slice(&packed.k.to_le_bytes());
        out.extend_from_slice(&packed.data_len.to_le_bytes());
    }
    for packed in &packed_tensors {
        out.extend_from_slice(&packed.blob);
    }
    Ok(out)
}

/// A little-endian cursor that errors (never panics) on a short read.
struct Cursor<'a> {
    b: &'a [u8],
    o: usize,
}
impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], FormatError> {
        let end = self.o.checked_add(n).ok_or(FormatError::WrongBlockLen {
            expected: n,
            got: 0,
        })?;
        if end > self.b.len() {
            return Err(FormatError::WrongBlockLen {
                expected: end,
                got: self.b.len(),
            });
        }
        let s = &self.b[self.o..end];
        self.o = end;
        Ok(s)
    }
    fn u16(&mut self) -> Result<usize, FormatError> {
        let s = self.take(2)?;
        Ok(u16::from_le_bytes([s[0], s[1]]) as usize)
    }
    fn u32(&mut self) -> Result<usize, FormatError> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize)
    }
    fn u64(&mut self) -> Result<usize, FormatError> {
        let s = self.take(8)?;
        Ok(u64::from_le_bytes(s.try_into().expect("8 bytes")) as usize)
    }
}

/// Parse a SALT bundle into its tensors, enforcing magic + version and bounds-checking
/// every field (a corrupt or truncated bundle errors, never panics or reads OOB).
///
/// # Errors
/// [`FormatError::SaltBadMagic`] on a bad magic, [`FormatError::UnsupportedSaltVersion`] on
/// a version this build can't read, [`FormatError::WrongBlockLen`] on truncation/length
/// disagreement, or any underlying [`unpack_salt_row`] error.
pub fn read_salt_bundle(bytes: &[u8]) -> Result<Vec<SaltTensor>, FormatError> {
    read_salt_bundle_prefix(bytes, usize::MAX)
}

/// Parse a SALT bundle while materializing at most `max_planes` per row.
///
/// Both legacy dense v1 and progressive v2 row payloads are accepted. Full row
/// framing and all payloads are validated even when only a prefix is retained.
///
/// # Errors
/// Same errors as [`read_salt_bundle`] and [`unpack_salt_row_prefix`](crate::unpack_salt_row_prefix).
pub fn read_salt_bundle_prefix(
    bytes: &[u8],
    max_planes: usize,
) -> Result<Vec<SaltTensor>, FormatError> {
    let mut c = Cursor { b: bytes, o: 0 };
    if c.take(4)? != SALT_BUNDLE_MAGIC {
        return Err(FormatError::SaltBadMagic);
    }
    let version = c.take(1)?[0];
    if version != SALT_BUNDLE_VERSION {
        return Err(FormatError::UnsupportedSaltVersion(version));
    }
    let _reserved = c.take(1)?;
    let tensor_count = c.u32()?;

    // Index, then a data region whose layout the index pins down.
    struct Entry {
        name: String,
        rows: usize,
        k: usize,
        data_len: usize,
    }
    // Cap the reservation by what the buffer could actually hold (each index entry is
    // ≥18 bytes: name_len 2 + rows 4 + k 4 + data_len 8). A crafted `tensor_count` of
    // u32::MAX on a tiny file must not reserve gigabytes before the per-entry `take` errors.
    let mut index = Vec::with_capacity(tensor_count.min(bytes.len() / 18));
    for _ in 0..tensor_count {
        let name_len = c.u16()?;
        let name = core::str::from_utf8(c.take(name_len)?)
            .map_err(|_| FormatError::SaltBadMagic)?
            .to_owned();
        let rows = c.u32()?;
        let k = c.u32()?;
        let data_len = c.u64()?;
        index.push(Entry {
            name,
            rows,
            k,
            data_len,
        });
    }

    let mut out = Vec::with_capacity(tensor_count.min(bytes.len() / 18));
    for e in index {
        let blob = c.take(e.data_len)?;
        // Walk the blob's `rows` self-describing packed rows.
        let mut off = 0usize;
        // Cap by the blob (each row is ≥SALT_HEADER_BYTES) so a huge `e.rows` can't reserve
        // unboundedly before the per-row bounds check below errors.
        let mut salt_rows = Vec::with_capacity(e.rows.min(blob.len() / SALT_HEADER_BYTES + 1));
        for _ in 0..e.rows {
            if off + SALT_HEADER_BYTES > blob.len() {
                return Err(FormatError::WrongBlockLen {
                    expected: off + SALT_HEADER_BYTES,
                    got: blob.len(),
                });
            }
            let row_len = packed_salt_row_len(&blob[off..])?;
            let row = unpack_salt_row_prefix(&blob[off..off + row_len], max_planes)?;
            if row.k != e.k {
                return Err(FormatError::WrongBlockLen {
                    expected: e.k,
                    got: row.k,
                });
            }
            salt_rows.push(row);
            off += row_len;
        }
        if off != blob.len() {
            return Err(FormatError::WrongBlockLen {
                expected: off,
                got: blob.len(),
            });
        }
        out.push(SaltTensor {
            name: e.name,
            rows: e.rows,
            k: e.k,
            salt_rows,
        });
    }
    if c.o != bytes.len() {
        return Err(FormatError::WrongBlockLen {
            expected: c.o,
            got: bytes.len(),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TQ2_0_BLOCK_BYTES, num_blocks, pack_tq2_0_row};
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

    fn sparse_residual(k: usize, stride: usize) -> Vec<u8> {
        let trits: Vec<Trit> = (0..k)
            .map(|i| {
                if i % stride == 0 {
                    Trit::from_i8(if (i / stride).is_multiple_of(2) {
                        1
                    } else {
                        -1
                    })
                    .unwrap()
                } else {
                    Trit::ZERO
                }
            })
            .collect();
        let scales = vec![f16::from_f32(0.125); num_blocks(k)];
        let mut packed = vec![0u8; num_blocks(k) * TQ2_0_BLOCK_BYTES];
        pack_tq2_0_row(&trits, &scales, &mut packed).unwrap();
        packed
    }

    #[test]
    fn progressive_bundle_shrinks_roundtrips_and_loads_prefix() {
        let mut a = row(4096, 1, 1);
        a.planes.push(sparse_residual(4096, 64));
        let rows = vec![a];
        let tensors: Vec<(&str, &[SaltRow])> = vec![("w", &rows)];

        let legacy = write_salt_bundle(&tensors).expect("legacy bundle");
        let progressive =
            write_progressive_salt_bundle(&tensors, 0.10).expect("progressive bundle");

        assert!(progressive.len() < legacy.len());
        assert_eq!(
            progressive,
            write_progressive_salt_bundle(&tensors, 0.10).expect("deterministic bundle")
        );
        assert_eq!(
            read_salt_bundle(&progressive).expect("full bundle")[0].salt_rows,
            rows
        );
        let prefix = read_salt_bundle_prefix(&progressive, 1).expect("prefix bundle");
        assert_eq!(
            prefix[0].salt_rows[0].planes,
            vec![rows[0].planes[0].clone()]
        );
    }

    #[test]
    fn bundle_roundtrips_multi_tensor_varied_t() {
        // Three tensors, different shapes; rows within a tensor carry DIFFERENT plane counts.
        let t_a: Vec<SaltRow> = vec![row(256, 1, 1), row(256, 3, 2), row(256, 2, 3)];
        let t_b: Vec<SaltRow> = vec![row(512, 2, 4), row(512, 1, 5)];
        let t_c: Vec<SaltRow> = vec![row(257, 1, 6)]; // partial last block
        let tensors: Vec<(&str, &[SaltRow])> = vec![
            ("q_proj.weight", &t_a),
            ("down.weight", &t_b),
            ("odd.weight", &t_c),
        ];
        let packed = write_salt_bundle(&tensors).unwrap();
        let got = read_salt_bundle(&packed).unwrap();

        assert_eq!(got.len(), 3);
        assert_eq!(got[0].name, "q_proj.weight");
        assert_eq!((got[0].rows, got[0].k), (3, 256));
        assert_eq!(got[0].salt_rows, t_a);
        assert_eq!(got[1].salt_rows, t_b);
        assert_eq!(got[2].salt_rows, t_c);
    }

    #[test]
    fn bundle_is_deterministic() {
        let t_a = vec![row(256, 2, 7)];
        let tensors: Vec<(&str, &[SaltRow])> = vec![("w", &t_a)];
        assert_eq!(
            write_salt_bundle(&tensors).unwrap(),
            write_salt_bundle(&tensors).unwrap()
        );
    }

    #[test]
    fn writer_rejects_inconsistent_rows_and_oversized_names() {
        let inconsistent = vec![row(256, 1, 1), row(512, 1, 2)];
        assert!(matches!(
            write_progressive_salt_bundle(&[("w", &inconsistent)], 0.10),
            Err(FormatError::WrongBlockLen {
                expected: 256,
                got: 512
            })
        ));

        let long_name = "w".repeat(u16::MAX as usize + 1);
        let rows = vec![row(256, 1, 3)];
        assert!(matches!(
            write_progressive_salt_bundle(&[(&long_name, &rows)], 0.10),
            Err(FormatError::WrongBlockLen { .. })
        ));
    }

    #[test]
    fn bad_magic_and_truncation_rejected() {
        let t_a = vec![row(256, 1, 8)];
        let tensors: Vec<(&str, &[SaltRow])> = vec![("w", &t_a)];
        let mut packed = write_salt_bundle(&tensors).unwrap();
        // bad magic
        let mut bad = packed.clone();
        bad[0] = b'X';
        assert!(matches!(
            read_salt_bundle(&bad),
            Err(FormatError::SaltBadMagic)
        ));
        // truncated mid-data
        packed.truncate(packed.len() - 10);
        assert!(matches!(
            read_salt_bundle(&packed),
            Err(FormatError::WrongBlockLen { .. })
        ));
    }

    #[test]
    fn huge_counts_error_without_unbounded_alloc() {
        // A tiny buffer claiming u32::MAX tensors must error from the bounds check, not
        // reserve gigabytes via with_capacity. (If the cap regressed this would OOM the test.)
        let mut b = Vec::new();
        b.extend_from_slice(&SALT_BUNDLE_MAGIC);
        b.push(SALT_BUNDLE_VERSION);
        b.push(0);
        b.extend_from_slice(&u32::MAX.to_le_bytes()); // tensor_count = 4 billion
        assert!(matches!(
            read_salt_bundle(&b),
            Err(FormatError::WrongBlockLen { .. })
        ));

        // One tensor claiming u32::MAX rows but a tiny data blob → per-row bounds check errors.
        let mut b2 = Vec::new();
        b2.extend_from_slice(&SALT_BUNDLE_MAGIC);
        b2.push(SALT_BUNDLE_VERSION);
        b2.push(0);
        b2.extend_from_slice(&1u32.to_le_bytes()); // 1 tensor
        b2.extend_from_slice(&1u16.to_le_bytes()); // name_len 1
        b2.push(b'w'); // name
        b2.extend_from_slice(&u32::MAX.to_le_bytes()); // rows = 4 billion
        b2.extend_from_slice(&256u32.to_le_bytes()); // k
        b2.extend_from_slice(&4u64.to_le_bytes()); // data_len = 4 bytes (too small for any row)
        b2.extend_from_slice(&[0u8; 4]); // the 4 data bytes
        assert!(matches!(
            read_salt_bundle(&b2),
            Err(FormatError::WrongBlockLen { .. })
        ));
    }
}
