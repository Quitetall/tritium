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

use core::fmt;
use std::collections::BTreeMap;
use std::io::{self, Write};

use crate::gguf::{
    GgufError, GgufValue, MAX_METADATA_ARRAY_ELEMENTS, MAX_METADATA_DEPTH, metadata_alignment,
    tensor_n_bytes, tensor_type_is_sized,
};

const STREAM_WRITE_CHUNK_BYTES: usize = 8 * 1024;

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

/// Append a full metadata value and recursively encode homogeneous arrays.
fn push_value(
    out: &mut Vec<u8>,
    v: &GgufValue,
    total_array_elements: &mut u64,
) -> Result<(), GgufError> {
    out.extend_from_slice(&value_type_tag(v).to_le_bytes());
    push_value_payload(out, v, 0, total_array_elements)
}

/// Append a value payload whose type tag is carried by its parent.
fn push_value_payload(
    out: &mut Vec<u8>,
    value: &GgufValue,
    depth: u8,
    total_array_elements: &mut u64,
) -> Result<(), GgufError> {
    match value {
        GgufValue::Array(items) => {
            if depth >= MAX_METADATA_DEPTH {
                return Err(GgufError::DimsOverflow);
            }
            let count = u64::try_from(items.len()).map_err(|_| GgufError::DimsOverflow)?;
            *total_array_elements = total_array_elements
                .checked_add(count)
                .ok_or(GgufError::DimsOverflow)?;
            if *total_array_elements > MAX_METADATA_ARRAY_ELEMENTS {
                return Err(GgufError::DimsOverflow);
            }
            // An empty array carries no element to infer the child type from;
            // U8 (tag 0) round-trips structurally since there are no elements.
            let child_tag = items.first().map_or(0, value_type_tag);
            out.extend_from_slice(&child_tag.to_le_bytes());
            out.extend_from_slice(&count.to_le_bytes());
            for item in items {
                if value_type_tag(item) != child_tag {
                    // Heterogeneous arrays are unrepresentable in GGUF.
                    return Err(GgufError::UnknownValueType(9));
                }
                push_value_payload(out, item, depth + 1, total_array_elements)?;
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
/// `data.len()` must equal the byte size the reader computes from `dims` and
/// `ggml_type` for known sized types. Unknown/custom type IDs retain their caller-
/// declared payload length.
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

/// Tensor metadata for a streaming GGUF payload whose exact byte length is known.
///
/// Specs are emitted in slice order. [`GgufStreamWriter`] requires payload chunks
/// to use the corresponding zero-based tensor index and writes the same tensor-info
/// table and aligned offsets as [`write_gguf`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgufTensorSpec {
    /// Tensor name (written as a GGUF string).
    pub name: String,
    /// Shape, fastest-varying dimension first.
    pub dims: Vec<u64>,
    /// ggml type-id (e.g. [`crate::GGML_TYPE_TQ2_0`]).
    pub ggml_type: u32,
    /// Exact number of payload bytes the stream must provide.
    pub data_len: u64,
}

/// Errors raised while constructing or writing a streaming GGUF container.
#[derive(Debug)]
#[non_exhaustive]
pub enum GgufWriteError {
    /// GGUF metadata, version, or layout was invalid.
    Gguf(GgufError),
    /// The destination writer failed.
    Io(io::Error),
    /// A payload chunk named a tensor other than the next tensor in stream order.
    TensorOutOfOrder {
        /// Tensor index required by the stream state.
        expected: usize,
        /// Tensor index supplied by the caller.
        got: usize,
    },
    /// A payload chunk was supplied after every declared tensor was complete.
    StreamComplete {
        /// Tensor index supplied by the caller.
        got: usize,
    },
    /// A tensor payload chunk would exceed its declared byte length.
    TensorTooLong {
        /// Tensor index being written.
        tensor: usize,
        /// Declared payload length.
        expected: u64,
        /// Total bytes that would have been written after accepting the chunk.
        attempted: u64,
    },
    /// [`GgufStreamWriter::finish`] was called before a payload reached its length.
    TensorTooShort {
        /// Tensor index that remains incomplete.
        tensor: usize,
        /// Declared payload length.
        expected: u64,
        /// Payload bytes successfully written.
        written: u64,
    },
    /// An earlier destination failure may have left a partial stream.
    Poisoned,
}

impl fmt::Display for GgufWriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gguf(error) => write!(f, "GGUF layout: {error}"),
            Self::Io(error) => write!(f, "GGUF destination: {error}"),
            Self::TensorOutOfOrder { expected, got } => write!(
                f,
                "GGUF tensor payload out of order: expected index {expected}, got {got}"
            ),
            Self::StreamComplete { got } => write!(
                f,
                "GGUF tensor payload index {got} supplied after the stream was complete"
            ),
            Self::TensorTooLong {
                tensor,
                expected,
                attempted,
            } => write!(
                f,
                "GGUF tensor {tensor} payload exceeds {expected} bytes (attempted {attempted})"
            ),
            Self::TensorTooShort {
                tensor,
                expected,
                written,
            } => write!(
                f,
                "GGUF tensor {tensor} payload is short: expected {expected} bytes, wrote {written}"
            ),
            Self::Poisoned => f.write_str("GGUF stream is poisoned after a destination failure"),
        }
    }
}

impl std::error::Error for GgufWriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Gguf(error) => Some(error),
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<GgufError> for GgufWriteError {
    fn from(error: GgufError) -> Self {
        Self::Gguf(error)
    }
}

impl From<io::Error> for GgufWriteError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

struct GgufLayout {
    header: Vec<u8>,
    tensor_data_offset: u64,
    offsets: Vec<u64>,
}

fn build_layout(
    version: u32,
    metadata: &BTreeMap<String, GgufValue>,
    tensors: &[GgufTensorSpec],
) -> Result<GgufLayout, GgufError> {
    if version != 2 && version != 3 {
        return Err(GgufError::UnsupportedVersion(version));
    }

    let alignment = metadata_alignment(metadata)?;

    let mut offsets = Vec::with_capacity(tensors.len());
    let mut relative_end = 0u64;
    for tensor in tensors {
        let expected_data_len = tensor_n_bytes(tensor.ggml_type, &tensor.dims)?;
        if tensor_type_is_sized(tensor.ggml_type) && tensor.data_len != expected_data_len {
            return Err(GgufError::InvalidTensorShape);
        }
        let offset = align_up(relative_end, alignment)?;
        offsets.push(offset);
        relative_end = offset
            .checked_add(tensor.data_len)
            .ok_or(GgufError::DimsOverflow)?;
    }

    let mut header = Vec::new();
    header.extend_from_slice(b"GGUF");
    header.extend_from_slice(&version.to_le_bytes());
    header.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    header.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    let mut total_array_elements = 0u64;
    for (key, value) in metadata {
        push_gguf_string(&mut header, key);
        push_value(&mut header, value, &mut total_array_elements)?;
    }
    for (tensor, &offset) in tensors.iter().zip(&offsets) {
        push_gguf_string(&mut header, &tensor.name);
        header.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
        for &dimension in &tensor.dims {
            header.extend_from_slice(&dimension.to_le_bytes());
        }
        header.extend_from_slice(&tensor.ggml_type.to_le_bytes());
        header.extend_from_slice(&offset.to_le_bytes());
    }

    let header_end = u64::try_from(header.len()).map_err(|_| GgufError::DimsOverflow)?;
    let tensor_data_offset = align_up(header_end, alignment)?;
    tensor_data_offset
        .checked_add(relative_end)
        .ok_or(GgufError::DimsOverflow)?;
    Ok(GgufLayout {
        header,
        tensor_data_offset,
        offsets,
    })
}

fn write_all_bounded<W: Write>(writer: &mut W, bytes: &[u8]) -> io::Result<()> {
    for chunk in bytes.chunks(STREAM_WRITE_CHUNK_BYTES) {
        writer.write_all(chunk)?;
    }
    Ok(())
}

/// Stateful exact-length GGUF v2/v3 writer.
///
/// Construction writes the deterministic metadata and tensor-info header. Call
/// [`write_tensor_chunk`](Self::write_tensor_chunk) with monotonically increasing
/// tensor indices; each tensor may be split into any number of chunks. Alignment
/// padding is emitted as bounded zero-filled writes. [`finish`](Self::finish)
/// succeeds only after every declared payload byte has been written.
#[derive(Debug)]
pub struct GgufStreamWriter<W: Write> {
    writer: W,
    tensor_lengths: Vec<u64>,
    offsets: Vec<u64>,
    tensor_data_offset: u64,
    position: u64,
    next_tensor: usize,
    tensor_written: u64,
    poisoned: bool,
}

impl<W: Write> GgufStreamWriter<W> {
    /// Write a GGUF header and prepare to receive exact-length tensor payloads.
    ///
    /// # Errors
    /// Returns [`GgufWriteError::Gguf`] for an invalid version, metadata value, or
    /// overflowing layout, and [`GgufWriteError::Io`] if the header cannot be written.
    pub fn new(
        mut writer: W,
        version: u32,
        metadata: &BTreeMap<String, GgufValue>,
        tensors: &[GgufTensorSpec],
    ) -> Result<Self, GgufWriteError> {
        let layout = build_layout(version, metadata, tensors)?;
        write_all_bounded(&mut writer, &layout.header)?;
        let position = u64::try_from(layout.header.len())
            .map_err(|_| GgufWriteError::Gguf(GgufError::DimsOverflow))?;
        let mut stream = Self {
            writer,
            tensor_lengths: tensors.iter().map(|tensor| tensor.data_len).collect(),
            offsets: layout.offsets,
            tensor_data_offset: layout.tensor_data_offset,
            position,
            next_tensor: 0,
            tensor_written: 0,
            poisoned: false,
        };
        stream.pad_to(stream.tensor_data_offset)?;
        stream.prepare_next_tensor()?;
        Ok(stream)
    }

    /// Append one chunk of the indexed tensor's payload.
    ///
    /// Chunks for a tensor must be contiguous, and tensors must be written in spec
    /// order. A chunk that would exceed the declared payload length is rejected
    /// before any of its bytes reach the destination.
    ///
    /// # Errors
    /// Returns a typed ordering/length error, [`GgufWriteError::Io`] on destination
    /// failure, or [`GgufWriteError::Poisoned`] after an earlier I/O failure.
    pub fn write_tensor_chunk(
        &mut self,
        tensor_index: usize,
        chunk: &[u8],
    ) -> Result<(), GgufWriteError> {
        if self.poisoned {
            return Err(GgufWriteError::Poisoned);
        }
        if self.next_tensor == self.tensor_lengths.len() {
            return Err(GgufWriteError::StreamComplete { got: tensor_index });
        }
        if tensor_index != self.next_tensor {
            return Err(GgufWriteError::TensorOutOfOrder {
                expected: self.next_tensor,
                got: tensor_index,
            });
        }
        let chunk_len = u64::try_from(chunk.len())
            .map_err(|_| GgufWriteError::Gguf(GgufError::DimsOverflow))?;
        let attempted =
            self.tensor_written
                .checked_add(chunk_len)
                .ok_or(GgufWriteError::TensorTooLong {
                    tensor: tensor_index,
                    expected: self.tensor_lengths[tensor_index],
                    attempted: u64::MAX,
                })?;
        let expected = self.tensor_lengths[tensor_index];
        if attempted > expected {
            return Err(GgufWriteError::TensorTooLong {
                tensor: tensor_index,
                expected,
                attempted,
            });
        }
        if let Err(error) = write_all_bounded(&mut self.writer, chunk) {
            self.poisoned = true;
            return Err(GgufWriteError::Io(error));
        }
        self.position = self
            .position
            .checked_add(chunk_len)
            .ok_or(GgufWriteError::Gguf(GgufError::DimsOverflow))?;
        self.tensor_written = attempted;
        if attempted == expected {
            self.next_tensor += 1;
            self.tensor_written = 0;
            self.prepare_next_tensor()?;
        }
        Ok(())
    }

    /// Finish the container and return the destination writer.
    ///
    /// # Errors
    /// Returns [`GgufWriteError::TensorTooShort`] if any payload is incomplete,
    /// [`GgufWriteError::Poisoned`] after a prior destination failure, or
    /// [`GgufWriteError::Io`] if flushing the exact stream fails.
    pub fn finish(mut self) -> Result<W, GgufWriteError> {
        if self.poisoned {
            return Err(GgufWriteError::Poisoned);
        }
        if self.next_tensor < self.tensor_lengths.len() {
            return Err(GgufWriteError::TensorTooShort {
                tensor: self.next_tensor,
                expected: self.tensor_lengths[self.next_tensor],
                written: self.tensor_written,
            });
        }
        self.writer.flush()?;
        Ok(self.writer)
    }

    fn prepare_next_tensor(&mut self) -> Result<(), GgufWriteError> {
        while self.next_tensor < self.tensor_lengths.len() {
            let target = self
                .tensor_data_offset
                .checked_add(self.offsets[self.next_tensor])
                .ok_or(GgufWriteError::Gguf(GgufError::DimsOverflow))?;
            self.pad_to(target)?;
            if self.tensor_lengths[self.next_tensor] != 0 {
                break;
            }
            self.next_tensor += 1;
        }
        Ok(())
    }

    fn pad_to(&mut self, target: u64) -> Result<(), GgufWriteError> {
        const ZEROES: [u8; STREAM_WRITE_CHUNK_BYTES] = [0; STREAM_WRITE_CHUNK_BYTES];
        let mut remaining = target
            .checked_sub(self.position)
            .ok_or(GgufWriteError::Gguf(GgufError::DimsOverflow))?;
        while remaining != 0 {
            let count = usize::try_from(remaining.min(ZEROES.len() as u64))
                .expect("bounded zero-padding chunk fits usize");
            if let Err(error) = self.writer.write_all(&ZEROES[..count]) {
                self.poisoned = true;
                return Err(GgufWriteError::Io(error));
            }
            self.position += count as u64;
            remaining -= count as u64;
        }
        Ok(())
    }
}

/// Serialize a GGUF v2/v3 container.
///
/// `version` must be 2 or 3. Tensor offsets are assigned sequentially, each
/// aligned up to the effective alignment (from `general.alignment` in
/// `metadata`, else [`crate::DEFAULT_ALIGNMENT`]).
///
/// # Errors
/// - [`GgufError::UnsupportedVersion`] if `version` is not 2 or 3.
/// - [`GgufError::InvalidAlignment`] for invalid declared alignment.
/// - [`GgufError::InvalidTensorShape`] when a known type's shape or payload length
///   disagrees with its GGML block layout.
/// - [`GgufError::UnknownValueType`] for a heterogeneous metadata array.
/// - [`GgufError::DimsOverflow`] on offset/length arithmetic overflow or excessive
///   metadata-array depth/elements.
pub fn write_gguf(
    version: u32,
    metadata: &BTreeMap<String, GgufValue>,
    tensors: &[TensorOut<'_>],
) -> Result<Vec<u8>, GgufError> {
    let specs: Vec<GgufTensorSpec> = tensors
        .iter()
        .map(|tensor| GgufTensorSpec {
            name: tensor.name.clone(),
            dims: tensor.dims.clone(),
            ggml_type: tensor.ggml_type,
            data_len: tensor.data.len() as u64,
        })
        .collect();
    let layout = build_layout(version, metadata, &specs)?;
    let mut out = layout.header;
    let tensor_data_offset =
        usize::try_from(layout.tensor_data_offset).map_err(|_| GgufError::DimsOverflow)?;
    out.resize(tensor_data_offset, 0);

    // Data section: each payload at `tensor_data_offset + offset`, zero-padding gaps.
    for (tensor, &offset) in tensors.iter().zip(&layout.offsets) {
        let absolute = layout
            .tensor_data_offset
            .checked_add(offset)
            .ok_or(GgufError::DimsOverflow)?;
        let absolute = usize::try_from(absolute).map_err(|_| GgufError::DimsOverflow)?;
        if out.len() < absolute {
            out.resize(absolute, 0);
        }
        out.extend_from_slice(tensor.data);
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
    fn known_tensor_geometry_and_payload_length_must_match() {
        let metadata = BTreeMap::new();
        let one_block = vec![0; 66];
        let malformed_row = [TensorOut {
            name: "bad-row".into(),
            dims: vec![128, 2],
            ggml_type: GGML_TYPE_TQ2_0,
            data: &one_block,
        }];
        assert!(matches!(
            write_gguf(3, &metadata, &malformed_row),
            Err(GgufError::InvalidTensorShape)
        ));

        let short_f32 = [TensorOut {
            name: "short-f32".into(),
            dims: vec![2],
            ggml_type: 0,
            data: &1.0f32.to_le_bytes(),
        }];
        assert!(matches!(
            write_gguf(3, &metadata, &short_f32),
            Err(GgufError::InvalidTensorShape)
        ));

        let custom = [TensorOut {
            name: "custom".into(),
            dims: vec![128, 2],
            ggml_type: 169,
            data: &one_block,
        }];
        assert!(write_gguf(3, &metadata, &custom).is_ok());
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
    fn nested_homogeneous_arrays_round_trip() {
        let mut meta = BTreeMap::new();
        meta.insert(
            "a.nested".into(),
            GgufValue::Array(vec![
                GgufValue::Array(vec![
                    GgufValue::Array(vec![GgufValue::U8(1), GgufValue::U8(2)]),
                    GgufValue::Array(vec![GgufValue::U8(3)]),
                ]),
                GgufValue::Array(vec![GgufValue::Array(vec![])]),
            ]),
        );
        let bytes = write_gguf(3, &meta, &[]).expect("write nested arrays");
        let parsed = read_gguf(&bytes).expect("read nested arrays");
        assert_eq!(parsed.metadata, meta);
    }

    #[test]
    fn heterogeneous_arrays_remain_rejected() {
        let metadata = BTreeMap::from([(
            "a.mixed".into(),
            GgufValue::Array(vec![GgufValue::U8(1), GgufValue::U16(2)]),
        )]);
        assert!(write_gguf(3, &metadata, &[]).is_err());
    }

    #[test]
    fn metadata_array_depth_is_bounded() {
        let mut value = GgufValue::U8(1);
        for _ in 0..=MAX_METADATA_DEPTH {
            value = GgufValue::Array(vec![value]);
        }
        let metadata = BTreeMap::from([("a.too_deep".into(), value)]);
        assert_eq!(
            write_gguf(3, &metadata, &[]).unwrap_err(),
            GgufError::DimsOverflow
        );
    }

    #[test]
    fn invalid_declared_alignment_is_rejected() {
        for alignment in [
            GgufValue::U32(0),
            GgufValue::U32(12),
            GgufValue::I32(-8),
            GgufValue::U64(32),
        ] {
            let metadata = BTreeMap::from([("general.alignment".into(), alignment)]);
            assert_eq!(
                write_gguf(3, &metadata, &[]).unwrap_err(),
                GgufError::InvalidAlignment
            );
        }
    }

    #[test]
    fn streaming_bytes_match_in_memory_writer() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".into(),
            GgufValue::String("llama".into()),
        );
        metadata.insert("general.alignment".into(), GgufValue::U32(64));
        metadata.insert(
            "test.array".into(),
            GgufValue::Array(vec![GgufValue::U32(3), GgufValue::U32(5)]),
        );
        let first: Vec<u8> = (0..66u8).collect();
        let second: Vec<u8> = (0..132u16).map(|byte| byte as u8 ^ 0xA5).collect();
        let tensors = vec![
            TensorOut {
                name: "blk.0.weight".into(),
                dims: vec![256],
                ggml_type: GGML_TYPE_TQ2_0,
                data: &first,
            },
            TensorOut {
                name: "blk.1.weight".into(),
                dims: vec![512],
                ggml_type: GGML_TYPE_TQ2_0,
                data: &second,
            },
        ];
        let specs = vec![
            GgufTensorSpec {
                name: "blk.0.weight".into(),
                dims: vec![256],
                ggml_type: GGML_TYPE_TQ2_0,
                data_len: first.len() as u64,
            },
            GgufTensorSpec {
                name: "blk.1.weight".into(),
                dims: vec![512],
                ggml_type: GGML_TYPE_TQ2_0,
                data_len: second.len() as u64,
            },
        ];

        for version in [2, 3] {
            let expected = write_gguf(version, &metadata, &tensors).expect("in-memory write");
            let mut writer =
                GgufStreamWriter::new(Vec::new(), version, &metadata, &specs).expect("stream");
            writer.write_tensor_chunk(0, &first).expect("first tensor");
            writer
                .write_tensor_chunk(1, &second)
                .expect("second tensor");
            let actual = writer.finish().expect("finish");
            assert_eq!(actual, expected, "version {version}");
        }
    }

    #[test]
    fn streaming_accepts_chunked_tensor_payloads() {
        let mut metadata = BTreeMap::new();
        metadata.insert("general.alignment".into(), GgufValue::U32(16));
        let first = b"abcdefg";
        let second = b"012345678";
        let tensors = vec![
            TensorOut {
                name: "first".into(),
                dims: vec![first.len() as u64],
                ggml_type: 169,
                data: first,
            },
            TensorOut {
                name: "second".into(),
                dims: vec![second.len() as u64],
                ggml_type: 169,
                data: second,
            },
        ];
        let specs = vec![
            GgufTensorSpec {
                name: "first".into(),
                dims: vec![first.len() as u64],
                ggml_type: 169,
                data_len: first.len() as u64,
            },
            GgufTensorSpec {
                name: "second".into(),
                dims: vec![second.len() as u64],
                ggml_type: 169,
                data_len: second.len() as u64,
            },
        ];
        let expected = write_gguf(3, &metadata, &tensors).expect("in-memory write");

        let mut writer = GgufStreamWriter::new(Vec::new(), 3, &metadata, &specs).expect("stream");
        for chunk in first.chunks(2) {
            writer.write_tensor_chunk(0, chunk).expect("first chunk");
        }
        for chunk in second.chunks(4) {
            writer.write_tensor_chunk(1, chunk).expect("second chunk");
        }

        assert_eq!(writer.finish().expect("finish"), expected);
    }

    #[test]
    fn streaming_rejects_short_long_and_out_of_order_payloads() {
        let metadata = BTreeMap::new();
        let specs = vec![
            GgufTensorSpec {
                name: "first".into(),
                dims: vec![4],
                ggml_type: 169,
                data_len: 4,
            },
            GgufTensorSpec {
                name: "second".into(),
                dims: vec![2],
                ggml_type: 169,
                data_len: 2,
            },
        ];

        let mut short = GgufStreamWriter::new(Vec::new(), 3, &metadata, &specs).expect("stream");
        short
            .write_tensor_chunk(0, &[1, 2, 3])
            .expect("partial payload");
        assert!(matches!(
            short.finish(),
            Err(GgufWriteError::TensorTooShort {
                tensor: 0,
                expected: 4,
                written: 3,
            })
        ));

        let mut recoverable =
            GgufStreamWriter::new(Vec::new(), 3, &metadata, &specs).expect("stream");
        assert!(matches!(
            recoverable.write_tensor_chunk(1, &[9]),
            Err(GgufWriteError::TensorOutOfOrder {
                expected: 0,
                got: 1,
            })
        ));
        assert!(matches!(
            recoverable.write_tensor_chunk(0, &[1, 2, 3, 4, 5]),
            Err(GgufWriteError::TensorTooLong {
                tensor: 0,
                expected: 4,
                attempted: 5,
            })
        ));
        recoverable
            .write_tensor_chunk(0, &[1, 2, 3, 4])
            .expect("exact first payload after rejected calls");
        recoverable
            .write_tensor_chunk(1, &[5, 6])
            .expect("exact second payload");
        assert!(matches!(
            recoverable.write_tensor_chunk(2, &[7]),
            Err(GgufWriteError::StreamComplete { got: 2 })
        ));
        assert!(recoverable.finish().is_ok());
    }

    #[test]
    fn streaming_padding_is_deterministic_and_zero_filled() {
        let mut metadata = BTreeMap::new();
        metadata.insert("general.alignment".into(), GgufValue::U32(64));
        let specs = vec![
            GgufTensorSpec {
                name: "first".into(),
                dims: vec![3],
                ggml_type: 169,
                data_len: 3,
            },
            GgufTensorSpec {
                name: "second".into(),
                dims: vec![2],
                ggml_type: 169,
                data_len: 2,
            },
        ];
        let write = || {
            let mut writer =
                GgufStreamWriter::new(Vec::new(), 3, &metadata, &specs).expect("stream");
            writer
                .write_tensor_chunk(0, &[1, 2, 3])
                .expect("first payload");
            writer
                .write_tensor_chunk(1, &[4, 5])
                .expect("second payload");
            writer.finish().expect("finish")
        };

        let first = write();
        let second = write();
        assert_eq!(first, second);
        let parsed = read_gguf(&first).expect("parse");
        assert_eq!(parsed.tensors[0].offset, 0);
        assert_eq!(parsed.tensors[1].offset, 64);
        let data = parsed.tensor_data_offset as usize;
        assert_eq!(&first[data..data + 3], &[1, 2, 3]);
        assert!(first[data + 3..data + 64].iter().all(|&byte| byte == 0));
        assert_eq!(&first[data + 64..data + 66], &[4, 5]);
    }

    #[derive(Default)]
    struct InstrumentedWriter {
        bytes: Vec<u8>,
        max_write: usize,
        calls: usize,
    }

    impl Write for InstrumentedWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.max_write = self.max_write.max(bytes.len());
            self.calls += 1;
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn streaming_bounds_destination_writes_for_large_padding_and_payload_chunks() {
        const MAX_WRITE_BYTES: usize = 8192;
        const PAYLOAD_BYTES: usize = 256 * 1024;
        let mut metadata = BTreeMap::new();
        metadata.insert("general.alignment".into(), GgufValue::U32(1024 * 1024));
        let specs = vec![GgufTensorSpec {
            name: "large".into(),
            dims: vec![PAYLOAD_BYTES as u64],
            ggml_type: 169,
            data_len: PAYLOAD_BYTES as u64,
        }];
        let mut writer = GgufStreamWriter::new(InstrumentedWriter::default(), 3, &metadata, &specs)
            .expect("stream");
        let payload = vec![0xA5; PAYLOAD_BYTES];
        writer
            .write_tensor_chunk(0, &payload)
            .expect("bounded payload");
        let destination = writer.finish().expect("finish");

        assert!(destination.calls > PAYLOAD_BYTES / MAX_WRITE_BYTES);
        assert!(destination.max_write <= MAX_WRITE_BYTES);
        assert_eq!(
            &destination.bytes[destination.bytes.len() - PAYLOAD_BYTES..],
            &payload
        );
    }
}
