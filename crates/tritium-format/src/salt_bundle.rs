//! Whole-model **SALT bundle** — a single-file container of many [`SaltRow`]s, the
//! artifact `tritium quantize` writes (ADR 0006).
//!
//! [`pack_salt_row`]/[`unpack_salt_row`] serialize one row; a real model has thousands
//! across many tensors. The bundle wraps them with a per-tensor index so a loader can find
//! a named tensor's rows. It stays a thin envelope: the row payload is exactly
//! [`pack_salt_row`] output, so the bundle invents no new trit packing.
//!
//! Layout (little-endian):
//! ```text
//! magic b"TSLB" (4) | version u8 | _reserved u8 | tensor_count u32
//! index[tensor_count]: name_len u16 | name utf8 | rows u32 | k u32 | data_len u64
//! data: per tensor, `rows` × pack_salt_row(row) concatenated (data_len bytes total)
//! ```
//! Each row is self-describing (its own [`SALT_HEADER_BYTES`] header carries `T` and `k`),
//! so the reader walks a tensor's `rows` packed rows without per-row offsets.

use crate::{
    FormatError, SALT_HEADER_BYTES, SaltRow, TQ2_0_BLOCK_BYTES, num_blocks, pack_salt_row,
    unpack_salt_row,
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
/// [`FormatError::WrongBlockLen`] if a tensor has zero rows (no `k` to record), or any
/// underlying [`pack_salt_row`] error.
pub fn write_salt_bundle(tensors: &[(&str, &[SaltRow])]) -> Result<Vec<u8>, FormatError> {
    // Pack each tensor's rows into one contiguous blob first (so we know its data_len).
    let mut blobs: Vec<(usize, Vec<u8>)> = Vec::with_capacity(tensors.len());
    for (_, rows) in tensors {
        let k = rows.first().map(|r| r.k).ok_or(FormatError::WrongBlockLen { expected: 1, got: 0 })?;
        let mut blob = Vec::new();
        for row in *rows {
            blob.extend_from_slice(&pack_salt_row(row)?);
        }
        blobs.push((k, blob));
    }

    let mut out = Vec::new();
    out.extend_from_slice(&SALT_BUNDLE_MAGIC);
    out.push(SALT_BUNDLE_VERSION);
    out.push(0); // reserved
    out.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for ((name, rows), (k, blob)) in tensors.iter().zip(&blobs) {
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&(rows.len() as u32).to_le_bytes());
        out.extend_from_slice(&(*k as u32).to_le_bytes());
        out.extend_from_slice(&(blob.len() as u64).to_le_bytes());
    }
    for (_, blob) in &blobs {
        out.extend_from_slice(blob);
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
        let end = self.o.checked_add(n).ok_or(FormatError::WrongBlockLen { expected: n, got: 0 })?;
        if end > self.b.len() {
            return Err(FormatError::WrongBlockLen { expected: end, got: self.b.len() });
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
        index.push(Entry { name, rows, k, data_len });
    }

    let mut out = Vec::with_capacity(tensor_count.min(bytes.len() / 18));
    for e in index {
        let blob = c.take(e.data_len)?;
        // Walk the blob's `rows` self-describing packed rows.
        let plane_bytes = num_blocks(e.k) * TQ2_0_BLOCK_BYTES;
        let mut off = 0usize;
        // Cap by the blob (each row is ≥SALT_HEADER_BYTES) so a huge `e.rows` can't reserve
        // unboundedly before the per-row bounds check below errors.
        let mut salt_rows = Vec::with_capacity(e.rows.min(blob.len() / SALT_HEADER_BYTES + 1));
        for _ in 0..e.rows {
            if off + SALT_HEADER_BYTES > blob.len() {
                return Err(FormatError::WrongBlockLen { expected: off + SALT_HEADER_BYTES, got: blob.len() });
            }
            let t = blob[off + 5] as usize;
            // Checked: on 64-bit the downstream bounds check already covers this, but a
            // 32-bit `usize` could wrap (t ≤ 255, plane_bytes up to ~1.1 GB) — `read_salt_bundle`
            // promises never to panic, so compute the length without overflow.
            let row_len = t
                .checked_mul(plane_bytes)
                .and_then(|p| p.checked_add(SALT_HEADER_BYTES))
                .ok_or(FormatError::WrongBlockLen { expected: usize::MAX, got: blob.len() })?;
            if off + row_len > blob.len() {
                return Err(FormatError::WrongBlockLen { expected: off + row_len, got: blob.len() });
            }
            salt_rows.push(unpack_salt_row(&blob[off..off + row_len])?);
            off += row_len;
        }
        out.push(SaltTensor { name: e.name, rows: e.rows, k: e.k, salt_rows });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack_tq2_0_row;
    use half::f16;
    use tritium_core::Trit;

    /// A SALT row of `t` planes over `k` trits (deterministic dummy data).
    fn row(k: usize, t: usize, seed: u8) -> SaltRow {
        let nb = num_blocks(k);
        let planes = (0..t)
            .map(|p| {
                let trits: Vec<Trit> = (0..k)
                    .map(|i| Trit::from_i8(((i as i32 + p as i32 + seed as i32) % 3 - 1) as i8).unwrap())
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
        assert_eq!(write_salt_bundle(&tensors).unwrap(), write_salt_bundle(&tensors).unwrap());
    }

    #[test]
    fn bad_magic_and_truncation_rejected() {
        let t_a = vec![row(256, 1, 8)];
        let tensors: Vec<(&str, &[SaltRow])> = vec![("w", &t_a)];
        let mut packed = write_salt_bundle(&tensors).unwrap();
        // bad magic
        let mut bad = packed.clone();
        bad[0] = b'X';
        assert!(matches!(read_salt_bundle(&bad), Err(FormatError::SaltBadMagic)));
        // truncated mid-data
        packed.truncate(packed.len() - 10);
        assert!(matches!(read_salt_bundle(&packed), Err(FormatError::WrongBlockLen { .. })));
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
        assert!(matches!(read_salt_bundle(&b), Err(FormatError::WrongBlockLen { .. })));

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
        assert!(matches!(read_salt_bundle(&b2), Err(FormatError::WrongBlockLen { .. })));
    }
}
