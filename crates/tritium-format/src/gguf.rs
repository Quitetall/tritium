//! GGUF v2/v3 container reader (no writer).
//!
//! Parses the [GGUF] header, metadata key/value table, and tensor-info table of a
//! GGUF file held entirely in memory, then exposes read-only accessors. Every
//! multibyte integer is little-endian. The parser is total: no input — however
//! malformed or truncated — can panic or read out of bounds; every read is
//! length-checked and converts a short read into [`GgufError::Truncated`].
//!
//! Only the container is parsed. Tensor *payloads* are left in the original byte
//! slice; [`GgufFile::tensor_data_offset`] plus each [`TensorInfo::offset`] /
//! [`TensorInfo::n_bytes`] locate them. The ggml type-id of each tensor is exposed
//! verbatim (see [`GGML_TYPE_TQ1_0`] / [`GGML_TYPE_TQ2_0`]); this reader does not
//! dequantize.
//!
//! [GGUF]: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

use core::fmt;
use std::collections::BTreeMap;

/// ggml type-id for `TQ1_0` (base-3, 5 trits/byte). Verified against ggml's
/// `enum ggml_type` in `ggml.h`.
pub const GGML_TYPE_TQ1_0: u32 = 34;

/// ggml type-id for `TQ2_0` (2 bits/trit, 4/byte). Verified against ggml's
/// `enum ggml_type` in `ggml.h`.
pub const GGML_TYPE_TQ2_0: u32 = 35;

/// Default tensor-data alignment when `general.alignment` is absent (ggml convention).
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// GGUF magic, little-endian bytes `G G U F`.
const GGUF_MAGIC: [u8; 4] = *b"GGUF";

/// Upper bound on a single GGUF string length, to reject absurd allocations from
/// adversarial input before attempting a read. 64 MiB is far above any real key.
const MAX_STRING_LEN: u64 = 64 * 1024 * 1024;

/// Upper bound on array / table element counts, to reject overflow-y inputs early.
const MAX_COUNT: u64 = 1u64 << 32;

/// Maximum tensor dimension count accepted. ggml's `GGML_MAX_DIMS` is 4; 8 leaves
/// headroom. Bounding this keeps `n_dims` from driving a huge upfront allocation.
const MAX_DIMS: u32 = 8;

/// Cap on how many elements to *preallocate* from a file-declared count. The real
/// container grows on demand as entries are read (and a short buffer truncates
/// long before the declared count), so this only bounds the speculative reserve.
const MAX_PREALLOC: usize = 4096;

/// Errors raised while reading a GGUF container.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GgufError {
    /// The first four bytes were not the ASCII magic `GGUF`.
    BadMagic,
    /// The version field was not a supported value (only 2 and 3 are accepted).
    UnsupportedVersion(u32),
    /// The buffer ended before a required field could be read.
    Truncated,
    /// A declared string length exceeded the sane upper bound ([`MAX_STRING_LEN`]).
    StringTooLong,
    /// A tensor's dimension product, or a declared count, overflowed `u64`/`usize`.
    DimsOverflow,
    /// A tensor's `offset + n_bytes` fell outside the tensor-data section.
    OffsetOutOfBounds,
    /// A metadata value declared a `value_type` id that GGUF does not define.
    UnknownValueType(u32),
    /// A string field held bytes that were not valid UTF-8.
    InvalidUtf8,
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgufError::BadMagic => write!(f, "bad GGUF magic (expected 'GGUF')"),
            GgufError::UnsupportedVersion(v) => write!(f, "unsupported GGUF version {v}"),
            GgufError::Truncated => write!(f, "buffer truncated: read past end of input"),
            GgufError::StringTooLong => write!(f, "GGUF string length exceeds the sane bound"),
            GgufError::DimsOverflow => write!(f, "tensor dimensions or count overflowed"),
            GgufError::OffsetOutOfBounds => {
                write!(f, "tensor offset/size outside the data section")
            }
            GgufError::UnknownValueType(t) => write!(f, "unknown metadata value type {t}"),
            GgufError::InvalidUtf8 => write!(f, "GGUF string was not valid UTF-8"),
        }
    }
}

impl std::error::Error for GgufError {}

/// A bounds-checked, little-endian, forward-only reader over a byte slice.
///
/// Every accessor returns [`GgufError::Truncated`] rather than panicking when the
/// remaining input is too short, which is what makes the whole parser total.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Cursor { buf, pos: 0 }
    }

    /// Borrow the next `n` bytes and advance, or [`GgufError::Truncated`].
    fn take(&mut self, n: usize) -> Result<&'a [u8], GgufError> {
        let end = self.pos.checked_add(n).ok_or(GgufError::Truncated)?;
        let slice = self.buf.get(self.pos..end).ok_or(GgufError::Truncated)?;
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, GgufError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// Read a GGUF string: `u64` byte-length followed by UTF-8 bytes.
    fn gguf_string(&mut self) -> Result<String, GgufError> {
        let len = self.u64()?;
        if len > MAX_STRING_LEN {
            return Err(GgufError::StringTooLong);
        }
        // `len <= MAX_STRING_LEN` (64 MiB) fits usize on every supported target.
        let bytes = self.take(len as usize)?;
        core::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| GgufError::InvalidUtf8)
    }
}

/// A scalar or aggregate value read from the GGUF metadata table.
///
/// Integers keep their declared width; [`GgufValue::Array`] preserves element
/// order and may nest only one level (GGUF arrays do not hold arrays).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum GgufValue {
    /// `value_type` 0.
    U8(u8),
    /// `value_type` 1.
    I8(i8),
    /// `value_type` 2.
    U16(u16),
    /// `value_type` 3.
    I16(i16),
    /// `value_type` 4.
    U32(u32),
    /// `value_type` 5.
    I32(i32),
    /// `value_type` 6.
    F32(f32),
    /// `value_type` 7.
    Bool(bool),
    /// `value_type` 8.
    String(String),
    /// `value_type` 9: a homogeneous list of scalars.
    Array(Vec<GgufValue>),
    /// `value_type` 10.
    U64(u64),
    /// `value_type` 11.
    I64(i64),
    /// `value_type` 12.
    F64(f64),
}

impl GgufValue {
    /// Interpret this value as an unsigned integer, widening any integer width.
    ///
    /// Returns `None` for non-integer kinds. Used to read `general.alignment`,
    /// which GGUF may store as any unsigned width.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::U8(v) => Some(u64::from(*v)),
            GgufValue::U16(v) => Some(u64::from(*v)),
            GgufValue::U32(v) => Some(u64::from(*v)),
            GgufValue::U64(v) => Some(*v),
            GgufValue::I8(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I16(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I32(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    /// Borrow this value as a string slice, or `None` if it is not a string.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s),
            _ => None,
        }
    }
}

/// Read one metadata value of the given `value_type` from the cursor.
///
/// `depth` guards against an array-of-arrays (illegal in GGUF) and any other
/// pathological nesting: it is 0 at the top level and 1 inside an array.
fn read_value(cur: &mut Cursor<'_>, value_type: u32, depth: u32) -> Result<GgufValue, GgufError> {
    Ok(match value_type {
        0 => GgufValue::U8(cur.u8()?),
        1 => GgufValue::I8(cur.u8()? as i8),
        2 => GgufValue::U16(cur.u16()?),
        3 => GgufValue::I16(cur.u16()? as i16),
        4 => GgufValue::U32(cur.u32()?),
        5 => GgufValue::I32(cur.u32()? as i32),
        6 => GgufValue::F32(f32::from_bits(cur.u32()?)),
        7 => GgufValue::Bool(cur.u8()? != 0),
        8 => GgufValue::String(cur.gguf_string()?),
        9 => {
            if depth > 0 {
                // GGUF forbids nested arrays; treat as an unknown shape.
                return Err(GgufError::UnknownValueType(9));
            }
            let child_type = cur.u32()?;
            let count = cur.u64()?;
            if count > MAX_COUNT {
                return Err(GgufError::DimsOverflow);
            }
            // `count <= MAX_COUNT` (2^32) fits usize on 64-bit; cap to be safe.
            let count = usize::try_from(count).map_err(|_| GgufError::DimsOverflow)?;
            let mut items = Vec::with_capacity(count.min(1024));
            for _ in 0..count {
                items.push(read_value(cur, child_type, depth + 1)?);
            }
            GgufValue::Array(items)
        }
        10 => GgufValue::U64(cur.u64()?),
        11 => GgufValue::I64(cur.u64()? as i64),
        12 => GgufValue::F64(f64::from_bits(cur.u64()?)),
        other => return Err(GgufError::UnknownValueType(other)),
    })
}

/// Description of one tensor's container entry (its payload is not parsed here).
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    /// Tensor name (a GGUF string).
    pub name: String,
    /// Shape, fastest-varying dimension first (ggml order). May be empty for a scalar.
    pub dims: Vec<u64>,
    /// ggml type-id (e.g. [`GGML_TYPE_TQ2_0`]). Exposed verbatim.
    pub ggml_type: u32,
    /// Byte offset of the payload, relative to [`GgufFile::tensor_data_offset`].
    pub offset: u64,
    /// Payload size in bytes, computed from `dims` and the type's block layout.
    pub n_bytes: u64,
}

impl TensorInfo {
    /// Number of elements = product of `dims` (1 for a 0-dimensional tensor).
    ///
    /// # Errors
    /// [`GgufError::DimsOverflow`] if the product overflows `u64`.
    pub fn element_count(&self) -> Result<u64, GgufError> {
        let mut n: u64 = 1;
        for &d in &self.dims {
            n = n.checked_mul(d).ok_or(GgufError::DimsOverflow)?;
        }
        Ok(n)
    }
}

/// Byte size of one packed block, and elements per block, for a ggml type-id.
///
/// Returns `None` for types this reader does not size (only the ternary formats
/// and the common float/`int8` types relevant to ternary models are listed); for
/// those, [`TensorInfo::n_bytes`] is left as 0 and callers should not trust it.
fn type_block_layout(ggml_type: u32) -> Option<(u64, u64)> {
    // (block_size_bytes, elements_per_block)
    match ggml_type {
        0 => Some((4, 1)),                  // F32
        1 => Some((2, 1)),                  // F16
        8 => Some((34, 32)),                // Q8_0: 32 q + f16 scale
        24 => Some((1, 1)),                 // I8
        25 => Some((2, 1)),                 // I16
        26 => Some((4, 1)),                 // I32
        30 => Some((2, 1)),                 // BF16
        GGML_TYPE_TQ1_0 => Some((54, 256)), // qs[48]+qh[4]+f16
        GGML_TYPE_TQ2_0 => Some((66, 256)), // qs[64]+f16
        _ => None,
    }
}

/// Compute payload byte-size from element count and type, mirroring ggml's
/// `nbytes = (n_elements / block_n) * block_size`.
fn tensor_n_bytes(ggml_type: u32, n_elements: u64) -> Result<u64, GgufError> {
    let Some((block_size, block_n)) = type_block_layout(ggml_type) else {
        // Unknown type: we cannot size it; report 0 rather than guess.
        return Ok(0);
    };
    // n_blocks = ceil(n_elements / block_n); ggml tensors are block-aligned, but
    // div_ceil keeps us safe against any non-multiple input without panicking.
    let n_blocks = n_elements.div_ceil(block_n);
    n_blocks
        .checked_mul(block_size)
        .ok_or(GgufError::DimsOverflow)
}

/// A fully-parsed GGUF container: header fields, metadata, and tensor table.
///
/// The original byte buffer is *not* retained; payloads are located via
/// [`Self::tensor_data_offset`] plus each [`TensorInfo`].
#[derive(Debug, Clone)]
pub struct GgufFile {
    /// File-format version (2 or 3).
    pub version: u32,
    /// Metadata key/value pairs, keyed by their dotted name.
    pub metadata: BTreeMap<String, GgufValue>,
    /// Tensor table, in file order.
    pub tensors: Vec<TensorInfo>,
    /// Absolute byte offset where the (aligned) tensor-data section begins.
    pub tensor_data_offset: u64,
}

impl GgufFile {
    /// Look up a metadata value by its full dotted key.
    #[must_use]
    pub fn get_metadata(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    /// Look up a tensor by name (linear scan; tensor tables are small).
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// The effective tensor-data alignment: `general.alignment` if present and
    /// non-zero, else [`DEFAULT_ALIGNMENT`].
    #[must_use]
    pub fn alignment(&self) -> u64 {
        self.get_metadata("general.alignment")
            .and_then(GgufValue::as_u64)
            .filter(|&a| a != 0)
            .unwrap_or(DEFAULT_ALIGNMENT)
    }
}

/// Round `pos` up to the next multiple of `align` (`align` assumed non-zero).
fn align_up(pos: u64, align: u64) -> Result<u64, GgufError> {
    if align == 0 {
        return Ok(pos);
    }
    // pos + (align - 1), rounded down to a multiple of align — overflow-checked.
    let bumped = pos.checked_add(align - 1).ok_or(GgufError::DimsOverflow)?;
    Ok(bumped - (bumped % align))
}

/// Parse a GGUF v2/v3 container from an in-memory byte slice.
///
/// On success returns a [`GgufFile`] with header fields, the full metadata table,
/// and a sized tensor table. Tensor payloads are *not* copied; locate them with
/// [`GgufFile::tensor_data_offset`] and each [`TensorInfo`].
///
/// # Errors
/// Returns a typed [`GgufError`] for any malformed input — bad magic, an
/// unsupported version, truncation, an over-long string, a count/dimension
/// overflow, a tensor offset outside the data section, or an unknown metadata
/// value type. It never panics and never reads out of bounds.
///
/// # Examples
/// ```
/// use tritium_format::read_gguf;
/// // An empty buffer is too short for the magic, so it errors rather than panics.
/// assert!(read_gguf(&[]).is_err());
/// ```
pub fn read_gguf(buf: &[u8]) -> Result<GgufFile, GgufError> {
    let mut cur = Cursor::new(buf);

    let magic = cur.take(4)?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::BadMagic);
    }

    let version = cur.u32()?;
    if version != 2 && version != 3 {
        return Err(GgufError::UnsupportedVersion(version));
    }

    let tensor_count = cur.u64()?;
    let metadata_kv_count = cur.u64()?;
    if tensor_count > MAX_COUNT || metadata_kv_count > MAX_COUNT {
        return Err(GgufError::DimsOverflow);
    }

    // Metadata table.
    let mut metadata = BTreeMap::new();
    for _ in 0..metadata_kv_count {
        let key = cur.gguf_string()?;
        let value_type = cur.u32()?;
        let value = read_value(&mut cur, value_type, 0)?;
        metadata.insert(key, value);
    }

    // Resolve alignment from the metadata we just parsed (default 32).
    let alignment = metadata
        .get("general.alignment")
        .and_then(GgufValue::as_u64)
        .filter(|&a| a != 0)
        .unwrap_or(DEFAULT_ALIGNMENT);

    // Tensor-info table.
    // `tensor_count` is attacker-controlled — cap the speculative reserve; the loop
    // grows as real entries are read and a short buffer truncates first.
    let tensor_count_usize = usize::try_from(tensor_count).map_err(|_| GgufError::DimsOverflow)?;
    let mut raw_tensors: Vec<(String, Vec<u64>, u32, u64)> =
        Vec::with_capacity(tensor_count_usize.min(MAX_PREALLOC));
    for _ in 0..tensor_count {
        let name = cur.gguf_string()?;
        let n_dims = cur.u32()?;
        if n_dims > MAX_DIMS {
            return Err(GgufError::DimsOverflow);
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(cur.u64()?);
        }
        let ggml_type = cur.u32()?;
        let offset = cur.u64()?;
        raw_tensors.push((name, dims, ggml_type, offset));
    }

    // The data section starts at the current position, padded up to `alignment`.
    let header_end = cur.pos as u64;
    let tensor_data_offset = align_up(header_end, alignment)?;

    // Total bytes available for payloads (0 if the data section was truncated away).
    let data_section_len = (buf.len() as u64).saturating_sub(tensor_data_offset);

    // Size each tensor and bounds-check its [offset, offset+n_bytes) span.
    let mut tensors = Vec::with_capacity(raw_tensors.len());
    for (name, dims, ggml_type, offset) in raw_tensors {
        let mut n_elements: u64 = 1;
        for &d in &dims {
            n_elements = n_elements.checked_mul(d).ok_or(GgufError::DimsOverflow)?;
        }
        let n_bytes = tensor_n_bytes(ggml_type, n_elements)?;
        // Validate the span only for types we can size; unknown types yield 0.
        if n_bytes > 0 {
            let end = offset.checked_add(n_bytes).ok_or(GgufError::DimsOverflow)?;
            if end > data_section_len {
                return Err(GgufError::OffsetOutOfBounds);
            }
        } else if offset > data_section_len {
            return Err(GgufError::OffsetOutOfBounds);
        }
        tensors.push(TensorInfo {
            name,
            dims,
            ggml_type,
            offset,
            n_bytes,
        });
    }

    Ok(GgufFile {
        version,
        metadata,
        tensors,
        tensor_data_offset,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal GGUF builder for tests: assembles a valid little-endian buffer.
    struct GgufBuilder {
        body: Vec<u8>,
        version: u32,
        n_tensors: u64,
        n_meta: u64,
        meta: Vec<u8>,
        tinfo: Vec<u8>,
    }

    fn push_str(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    impl GgufBuilder {
        fn new(version: u32) -> Self {
            GgufBuilder {
                body: Vec::new(),
                version,
                n_tensors: 0,
                n_meta: 0,
                meta: Vec::new(),
                tinfo: Vec::new(),
            }
        }

        fn meta_u32(&mut self, key: &str, v: u32) -> &mut Self {
            push_str(&mut self.meta, key);
            self.meta.extend_from_slice(&4u32.to_le_bytes()); // value_type U32
            self.meta.extend_from_slice(&v.to_le_bytes());
            self.n_meta += 1;
            self
        }

        fn meta_string(&mut self, key: &str, v: &str) -> &mut Self {
            push_str(&mut self.meta, key);
            self.meta.extend_from_slice(&8u32.to_le_bytes()); // value_type STRING
            push_str(&mut self.meta, v);
            self.n_meta += 1;
            self
        }

        fn meta_u32_array(&mut self, key: &str, vs: &[u32]) -> &mut Self {
            push_str(&mut self.meta, key);
            self.meta.extend_from_slice(&9u32.to_le_bytes()); // value_type ARRAY
            self.meta.extend_from_slice(&4u32.to_le_bytes()); // child U32
            self.meta
                .extend_from_slice(&(vs.len() as u64).to_le_bytes());
            for &v in vs {
                self.meta.extend_from_slice(&v.to_le_bytes());
            }
            self.n_meta += 1;
            self
        }

        fn tensor(&mut self, name: &str, dims: &[u64], ggml_type: u32, offset: u64) -> &mut Self {
            push_str(&mut self.tinfo, name);
            self.tinfo
                .extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for &d in dims {
                self.tinfo.extend_from_slice(&d.to_le_bytes());
            }
            self.tinfo.extend_from_slice(&ggml_type.to_le_bytes());
            self.tinfo.extend_from_slice(&offset.to_le_bytes());
            self.n_tensors += 1;
            self
        }

        /// Build the header+meta+tinfo, pad to `alignment`, then append `data`.
        fn build_with_data(&self, alignment: u64, data: &[u8]) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend_from_slice(&GGUF_MAGIC);
            out.extend_from_slice(&self.version.to_le_bytes());
            out.extend_from_slice(&self.n_tensors.to_le_bytes());
            out.extend_from_slice(&self.n_meta.to_le_bytes());
            out.extend_from_slice(&self.meta);
            out.extend_from_slice(&self.tinfo);
            let _ = &self.body;
            // Pad to alignment.
            while !(out.len() as u64).is_multiple_of(alignment) {
                out.push(0);
            }
            out.extend_from_slice(data);
            out
        }

        fn build(&self, alignment: u64) -> Vec<u8> {
            self.build_with_data(alignment, &[])
        }
    }

    #[test]
    fn rejects_adversarial_n_dims_without_oom() {
        // A tensor declaring n_dims = u32::MAX must error before any allocation,
        // not attempt a ~34 GB Vec::with_capacity (which aborts the process).
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC);
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&1u64.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&0u64.to_le_bytes()); // n_meta
        push_str(&mut buf, "w"); // tensor name
        buf.extend_from_slice(&u32::MAX.to_le_bytes()); // n_dims = ~4.29e9
        assert!(matches!(read_gguf(&buf), Err(GgufError::DimsOverflow)));
    }

    #[test]
    fn huge_tensor_count_truncates_not_ooms() {
        // A header claiming ~4.29e9 tensors with no bodies must truncate-error,
        // not speculatively preallocate a giant Vec.
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC);
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(u32::MAX as u64).to_le_bytes()); // n_tensors < MAX_COUNT
        buf.extend_from_slice(&0u64.to_le_bytes()); // n_meta
        assert!(read_gguf(&buf).is_err()); // truncates on first tensor read, no abort
    }

    #[test]
    fn parses_minimal_valid_file() {
        // Two tensors, default alignment 32, with payload bytes present.
        let mut b = GgufBuilder::new(3);
        b.meta_string("general.architecture", "llama")
            .meta_u32("general.alignment", 32)
            .meta_u32_array("test.array", &[1, 2, 3]);
        // TQ2_0 tensor of 256 elements => 1 block => 66 bytes, at offset 0.
        b.tensor("blk.0.weight", &[256], GGML_TYPE_TQ2_0, 0);
        // F32 tensor of 4 elements => 16 bytes, aligned to 32 => offset 64.
        b.tensor("blk.0.bias", &[4], 0, 64);

        let data = vec![0u8; 64 + 16];
        let buf = b.build_with_data(32, &data);

        let f = read_gguf(&buf).expect("parse");
        assert_eq!(f.version, 3);
        assert_eq!(f.alignment(), 32);
        assert_eq!(f.tensors.len(), 2);

        let w = f.tensor("blk.0.weight").expect("weight tensor");
        assert_eq!(w.dims, vec![256]);
        assert_eq!(w.ggml_type, GGML_TYPE_TQ2_0);
        assert_eq!(w.offset, 0);
        assert_eq!(w.n_bytes, 66);

        let bias = f.tensor("blk.0.bias").expect("bias tensor");
        assert_eq!(bias.ggml_type, 0);
        assert_eq!(bias.offset, 64);
        assert_eq!(bias.n_bytes, 16);

        assert_eq!(
            f.get_metadata("general.architecture")
                .and_then(|v| v.as_str()),
            Some("llama")
        );
        match f.get_metadata("test.array") {
            Some(GgufValue::Array(items)) => assert_eq!(items.len(), 3),
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn version_2_accepted() {
        let mut b = GgufBuilder::new(2);
        b.tensor("t", &[256], GGML_TYPE_TQ1_0, 0);
        let buf = b.build_with_data(32, &[0u8; 54]);
        let f = read_gguf(&buf).expect("v2 parse");
        assert_eq!(f.version, 2);
        assert_eq!(f.tensor("t").unwrap().n_bytes, 54);
    }

    #[test]
    fn bad_magic_errors() {
        let mut buf = GgufBuilder::new(3).build(32);
        buf[0] = b'X';
        assert_eq!(read_gguf(&buf).unwrap_err(), GgufError::BadMagic);
    }

    #[test]
    fn bad_version_errors() {
        let buf = GgufBuilder::new(1).build(32);
        assert_eq!(
            read_gguf(&buf).unwrap_err(),
            GgufError::UnsupportedVersion(1)
        );
        let buf = GgufBuilder::new(99).build(32);
        assert_eq!(
            read_gguf(&buf).unwrap_err(),
            GgufError::UnsupportedVersion(99)
        );
    }

    #[test]
    fn truncated_buffers_error_not_panic() {
        let mut b = GgufBuilder::new(3);
        b.meta_string("general.architecture", "llama");
        b.tensor("blk.0.weight", &[256], GGML_TYPE_TQ2_0, 0);
        let full = b.build_with_data(32, &[0u8; 66]);

        // Every prefix of a valid buffer must error, never panic.
        for len in 0..full.len() {
            let res = read_gguf(&full[..len]);
            // Short prefixes can't possibly be a complete, valid file.
            assert!(res.is_err(), "prefix len {len} unexpectedly parsed");
        }
    }

    #[test]
    fn empty_buffer_errors() {
        assert_eq!(read_gguf(&[]).unwrap_err(), GgufError::Truncated);
    }

    #[test]
    fn unknown_value_type_errors() {
        // Hand-build: magic, v3, 0 tensors, 1 meta with bogus value_type 999.
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC);
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensors
        buf.extend_from_slice(&1u64.to_le_bytes()); // meta
        push_str(&mut buf, "k");
        buf.extend_from_slice(&999u32.to_le_bytes()); // bad value_type
        assert_eq!(
            read_gguf(&buf).unwrap_err(),
            GgufError::UnknownValueType(999)
        );
    }

    #[test]
    fn string_too_long_errors() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC);
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        // A key whose declared length is absurd.
        buf.extend_from_slice(&(u64::MAX).to_le_bytes());
        assert_eq!(read_gguf(&buf).unwrap_err(), GgufError::StringTooLong);
    }

    #[test]
    fn offset_out_of_bounds_errors() {
        let mut b = GgufBuilder::new(3);
        // 256-elt TQ2_0 needs 66 bytes, but offset 1000 is past the (empty) data.
        b.tensor("t", &[256], GGML_TYPE_TQ2_0, 1000);
        let buf = b.build_with_data(32, &[]);
        assert_eq!(read_gguf(&buf).unwrap_err(), GgufError::OffsetOutOfBounds);
    }

    #[test]
    fn dims_overflow_errors() {
        let mut b = GgufBuilder::new(3);
        // Two dims whose product overflows u64.
        b.tensor("t", &[u64::MAX, 2], GGML_TYPE_TQ2_0, 0);
        let buf = b.build_with_data(32, &[]);
        assert_eq!(read_gguf(&buf).unwrap_err(), GgufError::DimsOverflow);
    }

    #[test]
    fn custom_alignment_respected() {
        let mut b = GgufBuilder::new(3);
        b.meta_u32("general.alignment", 64);
        b.tensor("t", &[256], GGML_TYPE_TQ2_0, 0);
        let buf = b.build_with_data(64, &[0u8; 66]);
        let f = read_gguf(&buf).expect("parse");
        assert_eq!(f.alignment(), 64);
        assert_eq!(f.tensor_data_offset % 64, 0);
    }

    #[test]
    fn unknown_ggml_type_is_sized_zero() {
        let mut b = GgufBuilder::new(3);
        b.tensor("t", &[10], 9999, 0); // unknown type id
        let buf = b.build_with_data(32, &[]);
        let f = read_gguf(&buf).expect("parse");
        assert_eq!(f.tensor("t").unwrap().n_bytes, 0);
    }

    #[test]
    fn no_panic_on_arbitrary_short_inputs() {
        // Fuzz-lite: many byte patterns, none may panic.
        for seed in 0u32..2000 {
            let n = (seed % 73) as usize;
            let bytes: Vec<u8> = (0..n)
                .map(|i| (seed.wrapping_mul(31) ^ i as u32) as u8)
                .collect();
            let _ = read_gguf(&bytes); // must not panic
        }
    }
}
