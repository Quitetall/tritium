//! `TSLJ` — a SALT bundle stored as joint-symbol entropy code, with **no padding**.
//!
//! # Why the file and the kernel format should differ
//!
//! The TQ2_0 bundle stores exactly what the kernel consumes, and pays for it twice on disk. Each
//! trit takes 2 bits where `log2(3) = 1.585` suffices, and every row is padded to whole 256-trit
//! blocks — a 576-wide row occupies 768 slots, 33% of which hold no weight. Measured on a converted
//! SmolLM2-135M at T=3, the shipped file is **7.965 bpw** while `convert` reported 6.1875.
//!
//! Block alignment is a property of the kernel, not of the model. So this container stores the
//! information and nothing else: one canonical Huffman code word per real weight over its `3^T`
//! joint state ([`crate::salt_joint_code`]), and the per-plane scales exactly as TQ2_0 stores them.
//! The reader rebuilds the TQ2_0 rows once at load, so padding exists only in memory, where the
//! kernel needs it, and nothing downstream changes. Same artifact: **4.577 bpw**.
//!
//! # Contract
//!
//! - **Byte-identical rows.** Decoding yields [`SaltRow`]s whose plane bytes equal the ones written.
//!   Scales are copied as raw `f16` bits, never through a float, so no rounding can intervene.
//! - **Padding must be zero.** Slots past `k` are not stored; they decode as the zero trit. A row
//!   with anything else there is refused at write time rather than silently changed.
//! - **Uniform `T`** across the bundle, one code for the whole model. The ladder path writes uniform
//!   `T`; a mixed or progressive bundle stays TQ2_0.
//! - **Per-tensor streams**, so a reader decodes one named tensor without touching the rest.
//! - **A distinct magic.** A reader that only knows `TSLB` fails closed on `TSLJ` instead of
//!   misreading a different body layout.
//!
//! # Layout (little-endian)
//!
//! ```text
//! magic "TSLJ" | version u8 | planes u8 | rotation_group u16 (0 = none) | tensor_count u32
//! code lengths: 3^planes × u8
//! directory, per tensor:
//!     name_len u16 | name | rows u32 | k u32 | payload_offset u64 | scale_bytes u64 | stream_bytes u64
//! payloads, per tensor:  scales (rows × blocks × planes × 2 bytes) | Huffman stream
//! ```

use core::fmt;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

use half::f16;
use tritium_core::Trit;

use crate::salt_joint_code::{self, JointCode, JointCodeError, JointStream};
use crate::{FormatError, QK_K, SaltRow, TQ2_0_BLOCK_BYTES, pack_tq2_0_block, unpack_tq2_0_block};

/// File magic for the joint-coded SALT bundle.
pub const SALT_JOINT_BUNDLE_MAGIC: [u8; 4] = *b"TSLJ";
/// Current container version.
pub const SALT_JOINT_BUNDLE_VERSION: u8 = 1;

const HEADER_BYTES: usize = 12;
const DIR_FIXED_BYTES: usize = 2 + 4 + 4 + 8 + 8 + 8;
/// A tensor directory can never claim more entries than this. Bounds allocation on a crafted file.
const MAX_TENSORS: usize = 1 << 20;

/// Errors from writing or reading a joint-coded bundle.
#[derive(Debug)]
#[non_exhaustive]
pub enum JointBundleError {
    /// Not a `TSLJ` file.
    BadMagic,
    /// A version this reader does not know.
    UnsupportedVersion(u8),
    /// Structural damage: a count, offset, or length that cannot be right.
    Malformed(&'static str),
    /// Rows in the bundle do not share one plane count.
    MixedPlanes {
        /// The tensor where the mismatch was found.
        tensor: String,
    },
    /// A padding slot past `k` held a non-zero trit, which this container cannot represent.
    NonZeroPadding {
        /// The offending tensor.
        tensor: String,
        /// The offending row.
        row: usize,
    },
    /// No tensor of that name.
    NotFound(String),
    /// The entropy code itself.
    Code(JointCodeError),
    /// TQ2_0 block packing.
    Format(FormatError),
    /// The underlying reader.
    Io(std::io::Error),
}

impl fmt::Display for JointBundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic => write!(f, "not a TSLJ joint-coded SALT bundle"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported TSLJ version {v}"),
            Self::Malformed(why) => write!(f, "malformed TSLJ bundle: {why}"),
            Self::MixedPlanes { tensor } => write!(
                f,
                "tensor `{tensor}` has a different plane count; a joint bundle needs uniform T"
            ),
            Self::NonZeroPadding { tensor, row } => write!(
                f,
                "tensor `{tensor}` row {row} has non-zero trits past k, which TSLJ does not store"
            ),
            Self::NotFound(name) => write!(f, "no tensor named `{name}`"),
            Self::Code(e) => write!(f, "joint code: {e}"),
            Self::Format(e) => write!(f, "TQ2_0: {e}"),
            Self::Io(e) => write!(f, "io: {e}"),
        }
    }
}

impl std::error::Error for JointBundleError {}

impl From<JointCodeError> for JointBundleError {
    fn from(e: JointCodeError) -> Self {
        Self::Code(e)
    }
}
impl From<FormatError> for JointBundleError {
    fn from(e: FormatError) -> Self {
        Self::Format(e)
    }
}
impl From<std::io::Error> for JointBundleError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Symbols and raw scale bytes for one tensor, extracted from its TQ2_0 rows.
struct Extracted {
    symbols: Vec<u16>,
    scales: Vec<u8>,
}

fn extract(name: &str, rows: &[SaltRow], planes: usize) -> Result<Extracted, JointBundleError> {
    let mut symbols = Vec::new();
    let mut scales = Vec::new();
    let mut digits: Vec<Vec<i8>> = Vec::new();
    for (r, row) in rows.iter().enumerate() {
        if row.planes.len() != planes {
            return Err(JointBundleError::MixedPlanes {
                tensor: name.to_owned(),
            });
        }
        let k = row.k;
        let blocks = k.div_ceil(QK_K);
        digits.resize_with(planes, Vec::new);
        for (p, plane) in row.planes.iter().enumerate() {
            if plane.len() != blocks * TQ2_0_BLOCK_BYTES {
                return Err(JointBundleError::Malformed("plane length does not match k"));
            }
            let d = &mut digits[p];
            d.clear();
            d.resize(k, 0);
            for b in 0..blocks {
                let blk = &plane[b * TQ2_0_BLOCK_BYTES..(b + 1) * TQ2_0_BLOCK_BYTES];
                let mut trits = [Trit::ZERO; QK_K];
                let mut scale = f16::ZERO;
                unpack_tq2_0_block(blk, &mut trits, &mut scale)?;
                let start = b * QK_K;
                for (i, &trit) in trits.iter().enumerate() {
                    let idx = start + i;
                    if idx < k {
                        d[idx] = i8::from(trit);
                    } else if trit != Trit::ZERO {
                        return Err(JointBundleError::NonZeroPadding {
                            tensor: name.to_owned(),
                            row: r,
                        });
                    }
                }
                // Raw bits, not the decoded float: nothing may round a stored scale.
                scales.extend_from_slice(&blk[TQ2_0_BLOCK_BYTES - 2..]);
            }
        }
        let refs: Vec<&[i8]> = digits.iter().map(Vec::as_slice).collect();
        symbols.extend(salt_joint_code::joint_symbols(&refs)?);
    }
    Ok(Extracted { symbols, scales })
}

/// Write a joint-coded bundle. `rotation_group` is recorded exactly as the TQ2_0 v2 bundle records
/// it; the reader hands it back so the runtime rotates the same way.
///
/// # Errors
/// [`JointBundleError::MixedPlanes`] for non-uniform `T`, [`JointBundleError::NonZeroPadding`] for a
/// row this container cannot represent, or a code/format error.
pub fn write_joint_salt_bundle(
    tensors: &[(&str, &[SaltRow])],
    rotation_group: Option<u16>,
) -> Result<Vec<u8>, JointBundleError> {
    let planes = tensors
        .iter()
        .find_map(|(_, rows)| rows.first())
        .map(|row| row.planes.len())
        .ok_or(JointBundleError::Malformed("bundle has no rows"))?;
    if tensors.len() > MAX_TENSORS {
        return Err(JointBundleError::Malformed("too many tensors"));
    }
    let alphabet =
        3usize.pow(u32::try_from(planes).map_err(|_| JointBundleError::Malformed("planes"))?);

    let extracted: Vec<Extracted> = tensors
        .iter()
        .map(|(name, rows)| extract(name, rows, planes))
        .collect::<Result<_, _>>()?;
    let mut freq = vec![0u64; alphabet];
    for e in &extracted {
        for &s in &e.symbols {
            freq[usize::from(s)] += 1;
        }
    }
    let code = JointCode::build(&freq, planes)?;
    let streams: Vec<JointStream> = extracted
        .iter()
        .map(|e| salt_joint_code::encode(&code, &e.symbols, e.symbols.len().max(1)))
        .collect::<Result<_, _>>()?;

    let dir_bytes: usize = tensors
        .iter()
        .map(|(name, _)| DIR_FIXED_BYTES + name.len())
        .sum();
    let mut offset = (HEADER_BYTES + alphabet + dir_bytes) as u64;

    let mut out = Vec::new();
    out.extend_from_slice(&SALT_JOINT_BUNDLE_MAGIC);
    out.push(SALT_JOINT_BUNDLE_VERSION);
    out.push(planes as u8);
    out.extend_from_slice(&rotation_group.unwrap_or(0).to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    out.extend_from_slice(code.lengths());

    for (((name, rows), e), s) in tensors.iter().zip(&extracted).zip(&streams) {
        let name_len =
            u16::try_from(name.len()).map_err(|_| JointBundleError::Malformed("name too long"))?;
        let k = rows.first().map_or(0, |r| r.k);
        if rows.iter().any(|r| r.k != k) {
            return Err(JointBundleError::Malformed(
                "rows of one tensor disagree on k",
            ));
        }
        out.extend_from_slice(&name_len.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&(rows.len() as u32).to_le_bytes());
        out.extend_from_slice(&(k as u32).to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&(e.scales.len() as u64).to_le_bytes());
        out.extend_from_slice(&(s.bits.len() as u64).to_le_bytes());
        offset += (e.scales.len() + s.bits.len()) as u64;
    }
    for (e, s) in extracted.iter().zip(&streams) {
        out.extend_from_slice(&e.scales);
        out.extend_from_slice(&s.bits);
    }
    Ok(out)
}

/// One tensor's directory entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JointTensorInfo {
    /// Tensor name.
    pub name: String,
    /// Output rows.
    pub rows: usize,
    /// Input width.
    pub k: usize,
    payload_offset: u64,
    scale_bytes: u64,
    stream_bytes: u64,
}

/// A seekable reader over a `TSLJ` bundle. Construction reads only the header and directory.
#[derive(Debug)]
pub struct JointSaltBundleReader<R> {
    source: R,
    planes: usize,
    rotation_group: Option<u16>,
    code: JointCode,
    entries: Vec<JointTensorInfo>,
    by_name: HashMap<String, usize>,
}

fn read_u8<R: Read>(r: &mut R) -> Result<u8, JointBundleError> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}
fn read_u16<R: Read>(r: &mut R) -> Result<u16, JointBundleError> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}
fn read_u32<R: Read>(r: &mut R) -> Result<u32, JointBundleError> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64<R: Read>(r: &mut R) -> Result<u64, JointBundleError> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

impl<R: Read + Seek> JointSaltBundleReader<R> {
    /// Parse the header and directory, validating every offset against the file length.
    ///
    /// # Errors
    /// [`JointBundleError::BadMagic`], [`JointBundleError::UnsupportedVersion`], or
    /// [`JointBundleError::Malformed`] for any count or offset that cannot be right.
    pub fn new_strict(mut source: R) -> Result<Self, JointBundleError> {
        let file_len = source.seek(SeekFrom::End(0))?;
        source.seek(SeekFrom::Start(0))?;
        let mut magic = [0u8; 4];
        source.read_exact(&mut magic)?;
        if magic != SALT_JOINT_BUNDLE_MAGIC {
            return Err(JointBundleError::BadMagic);
        }
        let version = read_u8(&mut source)?;
        if version != SALT_JOINT_BUNDLE_VERSION {
            return Err(JointBundleError::UnsupportedVersion(version));
        }
        let planes = usize::from(read_u8(&mut source)?);
        if planes == 0 || planes > salt_joint_code::MAX_JOINT_PLANES {
            return Err(JointBundleError::Malformed("plane count out of range"));
        }
        let rotation = read_u16(&mut source)?;
        let rotation_group = (rotation != 0).then_some(rotation);
        let count = read_u32(&mut source)? as usize;
        if count > MAX_TENSORS {
            return Err(JointBundleError::Malformed("tensor count exceeds limit"));
        }
        let alphabet = 3usize.pow(planes as u32);
        let mut lengths = vec![0u8; alphabet];
        source.read_exact(&mut lengths)?;
        let code = JointCode::from_lengths(lengths, planes)?;

        let mut entries = Vec::with_capacity(count.min(4096));
        let mut by_name = HashMap::with_capacity(count.min(4096));
        let mut pos = (HEADER_BYTES + alphabet) as u64;
        for _ in 0..count {
            let name_len = usize::from(read_u16(&mut source)?);
            pos += (DIR_FIXED_BYTES + name_len) as u64;
            if pos > file_len {
                return Err(JointBundleError::Malformed(
                    "directory runs past end of file",
                ));
            }
            let mut name = vec![0u8; name_len];
            source.read_exact(&mut name)?;
            let name = String::from_utf8(name)
                .map_err(|_| JointBundleError::Malformed("tensor name is not UTF-8"))?;
            let rows = read_u32(&mut source)? as usize;
            let k = read_u32(&mut source)? as usize;
            let payload_offset = read_u64(&mut source)?;
            let scale_bytes = read_u64(&mut source)?;
            let stream_bytes = read_u64(&mut source)?;
            let expected_scales = (rows as u64)
                .checked_mul(k.div_ceil(QK_K) as u64)
                .and_then(|v| v.checked_mul(planes as u64 * 2))
                .ok_or(JointBundleError::Malformed("scale size overflows"))?;
            if scale_bytes != expected_scales {
                return Err(JointBundleError::Malformed(
                    "scale byte count disagrees with rows, k and planes",
                ));
            }
            let end = payload_offset
                .checked_add(scale_bytes)
                .and_then(|v| v.checked_add(stream_bytes))
                .ok_or(JointBundleError::Malformed("payload size overflows"))?;
            if end > file_len {
                return Err(JointBundleError::Malformed("payload runs past end of file"));
            }
            if by_name.insert(name.clone(), entries.len()).is_some() {
                return Err(JointBundleError::Malformed("duplicate tensor name"));
            }
            entries.push(JointTensorInfo {
                name,
                rows,
                k,
                payload_offset,
                scale_bytes,
                stream_bytes,
            });
        }
        Ok(Self {
            source,
            planes,
            rotation_group,
            code,
            entries,
            by_name,
        })
    }

    /// The Hadamard group the weights were fitted under, if any — same meaning as the TQ2_0 bundle.
    #[must_use]
    pub const fn rotation_group(&self) -> Option<u16> {
        self.rotation_group
    }

    /// Plane count shared by every row.
    #[must_use]
    pub const fn planes(&self) -> usize {
        self.planes
    }

    /// Directory entry for `name`.
    #[must_use]
    pub fn tensor_info(&self, name: &str) -> Option<&JointTensorInfo> {
        self.by_name.get(name).map(|&i| &self.entries[i])
    }

    /// Every tensor, in file order.
    #[must_use]
    pub fn tensors(&self) -> &[JointTensorInfo] {
        &self.entries
    }

    /// Decode one tensor back into TQ2_0 rows, byte-identical to what was written.
    ///
    /// # Errors
    /// [`JointBundleError::NotFound`], or a code, format, or io error for a damaged payload.
    pub fn read_tensor(&mut self, name: &str) -> Result<Vec<SaltRow>, JointBundleError> {
        let info = self
            .tensor_info(name)
            .cloned()
            .ok_or_else(|| JointBundleError::NotFound(name.to_owned()))?;
        let scale_len = usize::try_from(info.scale_bytes)
            .map_err(|_| JointBundleError::Malformed("scale payload too large"))?;
        let stream_len = usize::try_from(info.stream_bytes)
            .map_err(|_| JointBundleError::Malformed("stream payload too large"))?;
        self.source.seek(SeekFrom::Start(info.payload_offset))?;
        let mut scales = vec![0u8; scale_len];
        self.source.read_exact(&mut scales)?;
        let mut bits = vec![0u8; stream_len];
        self.source.read_exact(&mut bits)?;

        let n = info
            .rows
            .checked_mul(info.k)
            .ok_or(JointBundleError::Malformed("symbol count overflows"))?;
        let stream = JointStream {
            bits,
            block_offsets: vec![0],
            block: n.max(1),
            symbols: n,
        };
        let symbols = salt_joint_code::decode(&self.code, &stream)?;

        let blocks = info.k.div_ceil(QK_K);
        let mut out = Vec::with_capacity(info.rows);
        let mut scale_at = 0usize;
        for r in 0..info.rows {
            let digits = salt_joint_code::planes_from_symbols(
                &symbols[r * info.k..(r + 1) * info.k],
                self.planes,
            )?;
            let mut planes_out = Vec::with_capacity(self.planes);
            for digit_plane in &digits {
                let mut plane = vec![0u8; blocks * TQ2_0_BLOCK_BYTES];
                for b in 0..blocks {
                    let mut trits = [Trit::ZERO; QK_K];
                    let start = b * QK_K;
                    for (i, t) in trits.iter_mut().enumerate() {
                        if let Some(&d) = digit_plane.get(start + i) {
                            *t = Trit::from_i8(d).map_err(FormatError::from)?;
                        }
                    }
                    let bits = u16::from_le_bytes([scales[scale_at], scales[scale_at + 1]]);
                    scale_at += 2;
                    pack_tq2_0_block(
                        &trits,
                        f16::from_bits(bits),
                        &mut plane[b * TQ2_0_BLOCK_BYTES..(b + 1) * TQ2_0_BLOCK_BYTES],
                    )?;
                }
                planes_out.push(plane);
            }
            out.push(SaltRow {
                k: info.k,
                planes: planes_out,
            });
        }
        Ok(out)
    }
}

/// Decode a whole in-memory bundle of either container into tensors, dispatching on the magic.
///
/// For tools and tests that inspect a converted artifact without caring how it was stored. The
/// runtime does not use this — it decodes one tensor at a time from a file.
///
/// # Errors
/// Any error either reader reports.
pub fn read_any_salt_bundle(bytes: &[u8]) -> Result<Vec<crate::SaltTensor>, JointBundleError> {
    if bytes.len() >= 4 && bytes[..4] == SALT_JOINT_BUNDLE_MAGIC {
        let mut reader = JointSaltBundleReader::new_strict(std::io::Cursor::new(bytes))?;
        let infos = reader.tensors().to_vec();
        infos
            .into_iter()
            .map(|info| {
                let salt_rows = reader.read_tensor(&info.name)?;
                Ok(crate::SaltTensor {
                    name: info.name,
                    rows: info.rows,
                    k: info.k,
                    salt_rows,
                })
            })
            .collect()
    } else {
        Ok(crate::read_salt_bundle(bytes)?)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    /// Build TQ2_0 rows from explicit digits and scales — zero padding, as the ladder writes.
    fn rows_from(k: usize, n_rows: usize, planes: usize, seed: u64) -> Vec<SaltRow> {
        let mut s = seed | 1;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let blocks = k.div_ceil(QK_K);
        (0..n_rows)
            .map(|_| {
                let planes_v = (0..planes)
                    .map(|p| {
                        let mut plane = vec![0u8; blocks * TQ2_0_BLOCK_BYTES];
                        for b in 0..blocks {
                            let mut trits = [Trit::ZERO; QK_K];
                            for (i, trit) in trits.iter_mut().enumerate() {
                                if b * QK_K + i < k {
                                    // Peaked at zero, like the ladder's digits.
                                    let v = match next() % 7 {
                                        0 => -1,
                                        1 => 1,
                                        _ => 0,
                                    };
                                    *trit = Trit::from_i8(v).unwrap();
                                }
                            }
                            let scale = f16::from_f32(
                                0.01 + (p as f32) * 0.003 + (next() % 97) as f32 * 1e-4,
                            );
                            pack_tq2_0_block(
                                &trits,
                                scale,
                                &mut plane[b * TQ2_0_BLOCK_BYTES..(b + 1) * TQ2_0_BLOCK_BYTES],
                            )
                            .unwrap();
                        }
                        plane
                    })
                    .collect();
                SaltRow {
                    k,
                    planes: planes_v,
                }
            })
            .collect()
    }

    /// The contract that makes this safe to load: every decoded row is byte-identical, including
    /// ragged widths whose last block is mostly padding.
    #[test]
    fn rows_roundtrip_byte_identical_including_ragged_widths() {
        let a = rows_from(576, 7, 3, 0xA1); // 3 blocks, last one 64/256 real
        let b = rows_from(256, 5, 3, 0xB2); // exact block
        let c = rows_from(1, 3, 3, 0xC3); // 1 real trit in a 256-slot block
        let tensors: Vec<(&str, &[SaltRow])> = vec![("a", &a), ("b", &b), ("c", &c)];
        for rot in [None, Some(256u16)] {
            let bytes = write_joint_salt_bundle(&tensors, rot).unwrap();
            let mut reader = JointSaltBundleReader::new_strict(Cursor::new(bytes)).unwrap();
            assert_eq!(reader.rotation_group(), rot);
            assert_eq!(reader.planes(), 3);
            for (name, rows) in &tensors {
                assert_eq!(&reader.read_tensor(name).unwrap(), rows, "tensor {name}");
            }
        }
    }

    /// Any tensor decodes alone, in any order — the loader asks for tensors by name.
    #[test]
    fn tensors_decode_independently_and_out_of_order() {
        let a = rows_from(300, 4, 2, 1);
        let b = rows_from(700, 3, 2, 2);
        let bytes = write_joint_salt_bundle(&[("a", &a), ("b", &b)], None).unwrap();
        let mut reader = JointSaltBundleReader::new_strict(Cursor::new(bytes)).unwrap();
        assert_eq!(reader.read_tensor("b").unwrap(), b);
        assert_eq!(reader.read_tensor("a").unwrap(), a);
        assert_eq!(reader.read_tensor("b").unwrap(), b);
        assert!(matches!(
            reader.read_tensor("missing"),
            Err(JointBundleError::NotFound(_))
        ));
    }

    /// The point of the container: it must be smaller than the TQ2_0 rows it replaces.
    #[test]
    fn the_container_is_smaller_than_the_padded_rows() {
        let a = rows_from(576, 64, 3, 9);
        let dense: usize = a.iter().flat_map(|r| &r.planes).map(Vec::len).sum();
        let bytes = write_joint_salt_bundle(&[("a", &a)], None).unwrap();
        assert!(
            bytes.len() < dense * 2 / 3,
            "joint bundle {} bytes vs {dense} bytes of padded TQ2_0 rows",
            bytes.len()
        );
    }

    #[test]
    fn non_zero_padding_is_refused_not_silently_dropped() {
        let mut rows = rows_from(10, 1, 2, 3);
        // Put a +1 trit in a padding slot (index 200 > k = 10): element 200 is chunk 1, n=2, m=8,
        // so byte 32+8 bits 4..6; set trit+1 = 2.
        let byte = &mut rows[0].planes[0][40];
        *byte = (*byte & !(0b11 << 4)) | (0b10 << 4);
        assert!(matches!(
            write_joint_salt_bundle(&[("x", &rows)], None),
            Err(JointBundleError::NonZeroPadding { row: 0, .. })
        ));
    }

    /// Both containers must hand a tool the same tensors.
    #[test]
    fn read_any_returns_identical_tensors_from_either_container() {
        let a = rows_from(576, 4, 3, 11);
        let b = rows_from(300, 2, 3, 12);
        let tensors: Vec<(&str, &[SaltRow])> = vec![("a", &a), ("b", &b)];
        let dense = crate::write_salt_bundle(&tensors).unwrap();
        let joint = write_joint_salt_bundle(&tensors, None).unwrap();
        assert_eq!(
            read_any_salt_bundle(&joint).unwrap(),
            read_any_salt_bundle(&dense).unwrap()
        );
    }

    #[test]
    fn mixed_plane_counts_are_refused() {
        let a = rows_from(64, 2, 3, 4);
        let b = rows_from(64, 2, 2, 5);
        assert!(matches!(
            write_joint_salt_bundle(&[("a", &a), ("b", &b)], None),
            Err(JointBundleError::MixedPlanes { .. })
        ));
    }

    #[test]
    fn a_tq2_reader_magic_is_refused_and_damage_is_reported() {
        let a = rows_from(128, 3, 2, 6);
        let bytes = write_joint_salt_bundle(&[("a", &a)], None).unwrap();

        let mut wrong = bytes.clone();
        wrong[..4].copy_from_slice(b"TSLB");
        assert!(matches!(
            JointSaltBundleReader::new_strict(Cursor::new(wrong)),
            Err(JointBundleError::BadMagic)
        ));

        let mut ver = bytes.clone();
        ver[4] = 99;
        assert!(matches!(
            JointSaltBundleReader::new_strict(Cursor::new(ver)),
            Err(JointBundleError::UnsupportedVersion(99))
        ));

        // Truncated: the directory's payload claims run past the end.
        let truncated = bytes[..bytes.len() - 5].to_vec();
        assert!(matches!(
            JointSaltBundleReader::new_strict(Cursor::new(truncated)),
            Err(JointBundleError::Malformed(_))
        ));

        // A crafted tensor count must not drive an allocation.
        let mut huge = bytes;
        huge[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(JointSaltBundleReader::new_strict(Cursor::new(huge)).is_err());
    }
}
