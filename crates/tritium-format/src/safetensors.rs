//! Minimal **safetensors** reader — just enough to load fp weights as `f32`.
//!
//! The container is: an 8-byte little-endian `u64` header length, then that many
//! bytes of JSON mapping each tensor name to `{dtype, shape, data_offsets:[a,b]}`
//! (plus an optional `__metadata__` entry), then the raw tensor bytes. Offsets are
//! relative to the start of the data region (right after the header).
//!
//! This is the fp **source** reader for SALT (ADR 0006): the BitNet
//! `*-bf16` master stores its weights as `BF16`, which we widen losslessly to
//! `f32` for [`tritium_quantize::quantize_tensor`]. `F16` and `F32` are also
//! supported; other dtypes error rather than silently mis-read.

use std::collections::BTreeMap;
use std::fmt;
use std::io::{self, Read, Seek, SeekFrom};

use half::{bf16, f16};
use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer};

/// Errors from parsing or reading a safetensors buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SafeTensorsError {
    /// Buffer is shorter than the 8-byte length prefix.
    TooShort,
    /// The declared header length runs past the end of the buffer.
    BadHeaderLen {
        /// Header length declared by the 8-byte little-endian prefix.
        declared: usize,
        /// Bytes actually available in the buffer after the prefix.
        available: usize,
    },
    /// The declared JSON header exceeds the safetensors format's 100 MB limit.
    HeaderTooLarge {
        /// Header length declared by the 8-byte little-endian prefix.
        declared: u64,
        /// Maximum accepted header length.
        limit: usize,
    },
    /// The JSON header failed to parse.
    Json(String),
    /// A tensor name was not present in the header.
    NotFound(String),
    /// A tensor's `data_offsets` fall outside the data region.
    OutOfBounds(String),
    /// A tensor's byte span does not match `shape × dtype_size`.
    LengthMismatch {
        /// Tensor name.
        name: String,
        /// Bytes the shape+dtype imply.
        expected: usize,
        /// Bytes the offsets span.
        got: usize,
    },
    /// A dtype this reader cannot widen to `f32`.
    UnsupportedDtype {
        /// Tensor name.
        name: String,
        /// The dtype string from the header.
        dtype: String,
    },
    /// A tensor's shape (or `shape × dtype_size`) overflows `usize` — a crafted
    /// header claiming an impossibly large tensor.
    ShapeOverflow {
        /// Tensor name.
        name: String,
    },
    /// A seek or read on a seek-backed source failed.
    Io {
        /// Operation that failed.
        operation: &'static str,
        /// Stable I/O error category.
        kind: io::ErrorKind,
        /// Platform error message.
        message: String,
    },
    /// A bounded header or decoded-output allocation could not be reserved.
    AllocationFailed {
        /// Number of bytes requested.
        requested_bytes: usize,
    },
    /// Tensor offsets do not form one contiguous, complete data region.
    InvalidLayout(String),
    /// A raw-payload visitor was given a zero-byte chunk limit.
    InvalidChunkSize {
        /// Requested maximum bytes per visitor call.
        requested: usize,
    },
}

impl fmt::Display for SafeTensorsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SafeTensorsError::TooShort => {
                write!(f, "safetensors: buffer shorter than 8-byte prefix")
            }
            SafeTensorsError::BadHeaderLen {
                declared,
                available,
            } => {
                write!(
                    f,
                    "safetensors: header len {declared} exceeds buffer ({available} bytes)"
                )
            }
            SafeTensorsError::HeaderTooLarge { declared, limit } => write!(
                f,
                "safetensors: header len {declared} exceeds the {limit}-byte limit"
            ),
            SafeTensorsError::Json(e) => write!(f, "safetensors: header JSON: {e}"),
            SafeTensorsError::NotFound(n) => write!(f, "safetensors: tensor `{n}` not found"),
            SafeTensorsError::OutOfBounds(n) => {
                write!(f, "safetensors: tensor `{n}` offsets out of bounds")
            }
            SafeTensorsError::LengthMismatch {
                name,
                expected,
                got,
            } => write!(
                f,
                "safetensors: tensor `{name}` spans {got} bytes, shape+dtype implies {expected}"
            ),
            SafeTensorsError::UnsupportedDtype { name, dtype } => {
                write!(
                    f,
                    "safetensors: tensor `{name}` has unsupported dtype `{dtype}`"
                )
            }
            SafeTensorsError::ShapeOverflow { name } => {
                write!(f, "safetensors: tensor `{name}` shape overflows usize")
            }
            SafeTensorsError::Io {
                operation,
                kind,
                message,
            } => write!(f, "safetensors: {operation} failed ({kind:?}): {message}"),
            SafeTensorsError::AllocationFailed { requested_bytes } => {
                write!(f, "safetensors: could not reserve {requested_bytes} bytes")
            }
            SafeTensorsError::InvalidLayout(message) => {
                write!(f, "safetensors: invalid data layout: {message}")
            }
            SafeTensorsError::InvalidChunkSize { requested } => write!(
                f,
                "safetensors: raw tensor visitor chunk size must be positive, got {requested}"
            ),
        }
    }
}

impl std::error::Error for SafeTensorsError {}

/// One tensor's header entry.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

struct UniqueHeader(BTreeMap<String, RawTensor>);

struct UniqueMetadata;

impl<'de> Deserialize<'de> for UniqueMetadata {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct UniqueMetadataVisitor;

        impl<'de> Visitor<'de> for UniqueMetadataVisitor {
            type Value = UniqueMetadata;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a string-to-string metadata object with unique keys")
            }

            fn visit_map<A>(self, mut access: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut keys = BTreeMap::new();
                while let Some(key) = access.next_key::<String>()? {
                    if keys.insert(key.clone(), ()).is_some() {
                        return Err(de::Error::custom(format!("duplicate metadata key `{key}`")));
                    }
                    access.next_value::<String>()?;
                }
                Ok(UniqueMetadata)
            }
        }

        deserializer.deserialize_map(UniqueMetadataVisitor)
    }
}

impl<'de> Deserialize<'de> for UniqueHeader {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct UniqueHeaderVisitor;

        impl<'de> Visitor<'de> for UniqueHeaderVisitor {
            type Value = UniqueHeader;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a safetensors header object with unique keys")
            }

            fn visit_map<A>(self, mut access: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut tensors = BTreeMap::new();
                let mut metadata_seen = false;
                while let Some(name) = access.next_key::<String>()? {
                    if name == "__metadata__" {
                        if metadata_seen {
                            return Err(de::Error::custom("duplicate header key `__metadata__`"));
                        }
                        metadata_seen = true;
                        access.next_value::<UniqueMetadata>()?;
                    } else {
                        if tensors.contains_key(&name) {
                            return Err(de::Error::custom(format!(
                                "duplicate header key `{name}`"
                            )));
                        }
                        let tensor = access.next_value::<RawTensor>()?;
                        tensors.insert(name, tensor);
                    }
                }
                Ok(UniqueHeader(tensors))
            }
        }

        deserializer.deserialize_map(UniqueHeaderVisitor)
    }
}

const MAX_HEADER_LEN: usize = 100_000_000;
const READ_CHUNK_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, Copy)]
enum FloatDtype {
    Bf16,
    F16,
    F32,
}

impl FloatDtype {
    const fn byte_size(self) -> usize {
        match self {
            Self::Bf16 | Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct RawTensorLayout {
    offset: u64,
    byte_len: usize,
    numel: usize,
}

#[derive(Debug, Clone, Copy)]
struct TensorLayout {
    offset: u64,
    byte_len: usize,
    numel: usize,
    dtype: FloatDtype,
}

fn raw_tensor_layout_entry(
    name: &str,
    tensor: &RawTensor,
    data_len: u64,
) -> Result<RawTensorLayout, SafeTensorsError> {
    let [start, end] = tensor.data_offsets;
    if start > end || end > data_len {
        return Err(SafeTensorsError::OutOfBounds(name.to_owned()));
    }
    let byte_len = usize::try_from(end - start).map_err(|_| SafeTensorsError::ShapeOverflow {
        name: name.to_owned(),
    })?;
    let bits = storage_bits(&tensor.dtype).ok_or_else(|| SafeTensorsError::UnsupportedDtype {
        name: name.to_owned(),
        dtype: tensor.dtype.clone(),
    })?;
    let numel = tensor
        .shape
        .iter()
        .try_fold(1usize, |product, &dimension| product.checked_mul(dimension))
        .ok_or_else(|| SafeTensorsError::ShapeOverflow {
            name: name.to_owned(),
        })?;
    let bit_len = numel
        .checked_mul(bits)
        .ok_or_else(|| SafeTensorsError::ShapeOverflow {
            name: name.to_owned(),
        })?;
    if bit_len % 8 != 0 {
        return Err(SafeTensorsError::InvalidLayout(format!(
            "tensor `{name}` has a non-byte-aligned {bit_len}-bit payload"
        )));
    }
    let expected = bit_len / 8;
    if byte_len != expected {
        return Err(SafeTensorsError::LengthMismatch {
            name: name.to_owned(),
            expected,
            got: byte_len,
        });
    }
    Ok(RawTensorLayout {
        offset: start,
        byte_len,
        numel,
    })
}

fn checked_header_len(declared: u64) -> Result<usize, SafeTensorsError> {
    if declared > MAX_HEADER_LEN as u64 {
        return Err(SafeTensorsError::HeaderTooLarge {
            declared,
            limit: MAX_HEADER_LEN,
        });
    }
    usize::try_from(declared).map_err(|_| SafeTensorsError::HeaderTooLarge {
        declared,
        limit: MAX_HEADER_LEN,
    })
}

fn parse_header(header: &[u8]) -> Result<BTreeMap<String, RawTensor>, SafeTensorsError> {
    if header.first() != Some(&b'{') {
        return Err(SafeTensorsError::Json(
            "header must begin with `{`".to_owned(),
        ));
    }
    // Duplicate keys fail closed at every supported object level. The optional
    // `__metadata__` entry is not a tensor and accepts only string values.
    let UniqueHeader(tensors) = serde_json::from_slice(header)
        .map_err(|error| SafeTensorsError::Json(error.to_string()))?;
    Ok(tensors)
}

fn storage_bits(dtype: &str) -> Option<usize> {
    match dtype {
        "F4" => Some(4),
        "F6_E2M3" | "F6_E3M2" => Some(6),
        "BOOL" | "U8" | "I8" | "F8_E5M2" | "F8_E4M3" | "F8_E8M0" | "F8_E4M3FNUZ"
        | "F8_E5M2FNUZ" => Some(8),
        "I16" | "U16" | "F16" | "BF16" => Some(16),
        "I32" | "U32" | "F32" => Some(32),
        "I64" | "U64" | "F64" | "C64" => Some(64),
        _ => None,
    }
}

fn validate_layout(
    tensors: &BTreeMap<String, RawTensor>,
    data_len: u64,
) -> Result<(), SafeTensorsError> {
    let requested_bytes = tensors
        .len()
        .saturating_mul(size_of::<(&String, &RawTensor)>());
    let mut ordered = Vec::new();
    ordered
        .try_reserve_exact(tensors.len())
        .map_err(|_| SafeTensorsError::AllocationFailed { requested_bytes })?;
    ordered.extend(tensors.iter());
    ordered.sort_unstable_by_key(|(_, tensor)| tensor.data_offsets);

    let mut cursor = 0u64;
    for (name, tensor) in ordered {
        let [start, end] = tensor.data_offsets;
        if start != cursor || end < start {
            return Err(SafeTensorsError::InvalidLayout(format!(
                "tensor `{name}` starts at {start}, expected {cursor}, and ends at {end}"
            )));
        }
        raw_tensor_layout_entry(name, tensor, data_len)?;
        cursor = end;
    }
    if cursor != data_len {
        return Err(SafeTensorsError::InvalidLayout(format!(
            "tensor metadata covers {cursor} of {data_len} data bytes"
        )));
    }
    Ok(())
}

fn tensor_layout(
    tensors: &BTreeMap<String, RawTensor>,
    name: &str,
    data_len: u64,
) -> Result<TensorLayout, SafeTensorsError> {
    let tensor = tensors
        .get(name)
        .ok_or_else(|| SafeTensorsError::NotFound(name.to_owned()))?;
    let raw = raw_tensor_layout_entry(name, tensor, data_len)?;
    let dtype = match tensor.dtype.as_str() {
        "BF16" => FloatDtype::Bf16,
        "F16" => FloatDtype::F16,
        "F32" => FloatDtype::F32,
        other => {
            return Err(SafeTensorsError::UnsupportedDtype {
                name: name.to_owned(),
                dtype: other.to_owned(),
            });
        }
    };
    // Check the widened allocation size separately. This can overflow even when
    // the source BF16/F16 span fits in `usize`.
    raw.numel
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| SafeTensorsError::ShapeOverflow {
            name: name.to_owned(),
        })?;
    Ok(TensorLayout {
        offset: raw.offset,
        byte_len: raw.byte_len,
        numel: raw.numel,
        dtype,
    })
}

fn raw_tensor_layout(
    tensors: &BTreeMap<String, RawTensor>,
    name: &str,
    data_len: u64,
) -> Result<RawTensorLayout, SafeTensorsError> {
    let tensor = tensors
        .get(name)
        .ok_or_else(|| SafeTensorsError::NotFound(name.to_owned()))?;
    raw_tensor_layout_entry(name, tensor, data_len)
}

fn reserve_f32(numel: usize) -> Result<Vec<f32>, SafeTensorsError> {
    let requested_bytes =
        numel
            .checked_mul(size_of::<f32>())
            .ok_or(SafeTensorsError::AllocationFailed {
                requested_bytes: usize::MAX,
            })?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(numel)
        .map_err(|_| SafeTensorsError::AllocationFailed { requested_bytes })?;
    Ok(output)
}

fn append_f32(output: &mut Vec<f32>, dtype: FloatDtype, raw: &[u8]) {
    match dtype {
        FloatDtype::Bf16 => output.extend(
            raw.chunks_exact(2)
                .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32()),
        ),
        FloatDtype::F16 => output.extend(
            raw.chunks_exact(2)
                .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32()),
        ),
        FloatDtype::F32 => output.extend(
            raw.chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
        ),
    }
}

fn visit_reader_range<R: Read + Seek>(
    reader: &mut R,
    absolute: u64,
    byte_len: usize,
    max_chunk_bytes: usize,
    mut visit: impl FnMut(&[u8]),
) -> Result<(), SafeTensorsError> {
    if max_chunk_bytes == 0 {
        return Err(SafeTensorsError::InvalidChunkSize {
            requested: max_chunk_bytes,
        });
    }
    if byte_len == 0 {
        return Ok(());
    }

    reader
        .seek(SeekFrom::Start(absolute))
        .map_err(|error| io_error("seek to tensor payload", error))?;
    let mut scratch = [0u8; READ_CHUNK_BYTES];
    let chunk_capacity = max_chunk_bytes.min(scratch.len());
    let mut remaining = byte_len;
    while remaining != 0 {
        let count = remaining.min(chunk_capacity);
        reader
            .read_exact(&mut scratch[..count])
            .map_err(|error| io_error("read tensor payload", error))?;
        visit(&scratch[..count]);
        remaining -= count;
    }
    Ok(())
}

/// A parsed safetensors buffer: the header table + a borrow of the data region.
#[derive(Debug)]
pub struct SafeTensors<'a> {
    tensors: BTreeMap<String, RawTensor>,
    data: &'a [u8],
}

impl<'a> SafeTensors<'a> {
    /// Parse the header of a safetensors buffer (no tensor data is copied).
    ///
    /// # Errors
    /// [`SafeTensorsError::TooShort`] / [`SafeTensorsError::BadHeaderLen`] /
    /// [`SafeTensorsError::HeaderTooLarge`] on a malformed prefix;
    /// [`SafeTensorsError::Json`] on an unparseable header; or a typed layout,
    /// dtype, shape, or length error for invalid tensor metadata.
    pub fn parse(buf: &'a [u8]) -> Result<Self, SafeTensorsError> {
        if buf.len() < 8 {
            return Err(SafeTensorsError::TooShort);
        }
        let n = checked_header_len(u64::from_le_bytes(buf[0..8].try_into().unwrap()))?;
        let header_end = 8usize.checked_add(n).filter(|&e| e <= buf.len()).ok_or(
            SafeTensorsError::BadHeaderLen {
                declared: n,
                available: buf.len().saturating_sub(8),
            },
        )?;

        let tensors = parse_header(&buf[8..header_end])?;
        let data = &buf[header_end..];
        validate_layout(&tensors, data.len() as u64)?;
        Ok(SafeTensors { tensors, data })
    }

    /// Tensor names, sorted.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// Number of tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether there are no tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// A tensor's shape, or `None` if absent.
    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|t| t.shape.as_slice())
    }

    /// A tensor's dtype string (e.g. `"BF16"`), or `None` if absent.
    pub fn dtype(&self, name: &str) -> Option<&str> {
        self.tensors.get(name).map(|t| t.dtype.as_str())
    }

    /// Read a tensor's data widened to `f32`, row-major. `BF16`/`F16` widen
    /// losslessly; `F32` is read directly.
    ///
    /// # Errors
    /// [`SafeTensorsError::NotFound`] / [`OutOfBounds`](SafeTensorsError::OutOfBounds)
    /// / [`LengthMismatch`](SafeTensorsError::LengthMismatch) /
    /// [`UnsupportedDtype`](SafeTensorsError::UnsupportedDtype) /
    /// [`ShapeOverflow`](SafeTensorsError::ShapeOverflow) /
    /// [`AllocationFailed`](SafeTensorsError::AllocationFailed).
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, SafeTensorsError> {
        let layout = tensor_layout(&self.tensors, name, self.data.len() as u64)?;
        let start = usize::try_from(layout.offset)
            .map_err(|_| SafeTensorsError::OutOfBounds(name.to_owned()))?;
        let end = start
            .checked_add(layout.byte_len)
            .ok_or_else(|| SafeTensorsError::OutOfBounds(name.to_owned()))?;
        let raw = self
            .data
            .get(start..end)
            .ok_or_else(|| SafeTensorsError::OutOfBounds(name.to_owned()))?;
        let mut output = reserve_f32(layout.numel)?;
        append_f32(&mut output, layout.dtype, raw);
        Ok(output)
    }
}

/// A seek-backed safetensors index that retains only the parsed JSON header.
///
/// This type itself reads only the 8-byte prefix and bounded header during
/// construction. Tensor payloads are fetched on demand with absolute seeks and
/// widened through a fixed-size scratch buffer. Whether `R` buffers or retains
/// source bytes is controlled by that reader; Tritium's HF adapter uses
/// unbuffered [`std::fs::File`] handles.
#[derive(Debug)]
pub struct SafeTensorsReader<R> {
    tensors: BTreeMap<String, RawTensor>,
    reader: R,
    data_start: u64,
    data_len: u64,
}

impl<R: Read + Seek> SafeTensorsReader<R> {
    /// Index a seekable safetensors source without reading tensor payloads.
    ///
    /// # Errors
    /// Returns a typed [`SafeTensorsError`] for malformed or oversized headers,
    /// failed seeks/reads, or a failed bounded header allocation.
    pub fn new(mut reader: R) -> Result<Self, SafeTensorsError> {
        let file_len = reader
            .seek(SeekFrom::End(0))
            .map_err(|error| io_error("seek to source end", error))?;
        reader
            .seek(SeekFrom::Start(0))
            .map_err(|error| io_error("seek to source start", error))?;
        if file_len < 8 {
            return Err(SafeTensorsError::TooShort);
        }

        let mut prefix = [0u8; 8];
        reader
            .read_exact(&mut prefix)
            .map_err(|error| io_error("read length prefix", error))?;
        let header_len = checked_header_len(u64::from_le_bytes(prefix))?;
        let available_u64 = file_len.saturating_sub(8);
        if header_len as u64 > available_u64 {
            return Err(SafeTensorsError::BadHeaderLen {
                declared: header_len,
                available: usize::try_from(available_u64).unwrap_or(usize::MAX),
            });
        }

        let mut header = Vec::new();
        header
            .try_reserve_exact(header_len)
            .map_err(|_| SafeTensorsError::AllocationFailed {
                requested_bytes: header_len,
            })?;
        header.resize(header_len, 0);
        reader
            .read_exact(&mut header)
            .map_err(|error| io_error("read JSON header", error))?;
        let tensors = parse_header(&header)?;
        let data_start = 8u64 + header_len as u64;
        let data_len = file_len - data_start;
        validate_layout(&tensors, data_len)?;
        Ok(Self {
            tensors,
            reader,
            data_start,
            data_len,
        })
    }

    /// Tensor names, sorted.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// Number of tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether there are no tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// A tensor's shape, or `None` if absent.
    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|tensor| tensor.shape.as_slice())
    }

    /// A tensor's dtype string (for example, `"BF16"`), or `None` if absent.
    pub fn dtype(&self, name: &str) -> Option<&str> {
        self.tensors.get(name).map(|tensor| tensor.dtype.as_str())
    }

    /// Visit one tensor's exact stored payload without widening or allocating
    /// storage proportional to the tensor size.
    ///
    /// `visit` receives consecutive, non-empty slices in row-major storage
    /// order. Each slice is at most `max_chunk_bytes` bytes (and at most the
    /// reader's fixed 64 KiB staging bound). A zero-sized tensor succeeds
    /// without invoking `visit`. Use [`Self::dtype`] and [`Self::shape`] to
    /// domain-separate a content digest from tensors with different metadata.
    ///
    /// Every call seeks to an absolute payload offset, so it does not depend on
    /// the cursor left by an earlier tensor read and does not affect later
    /// reader operations. The source is live and callbacks are nontransactional:
    /// an I/O error can occur after earlier chunks were delivered, and concurrent
    /// source mutation can produce a mixed stream. Callers must discard partial
    /// effects on `Err` and keep identity inputs stable for the full visit.
    ///
    /// # Errors
    /// Returns a typed [`SafeTensorsError`] for an absent or malformed tensor,
    /// a zero `max_chunk_bytes`, or a failed absolute seek/read.
    pub fn visit_tensor_bytes(
        &mut self,
        name: &str,
        max_chunk_bytes: usize,
        visit: impl FnMut(&[u8]),
    ) -> Result<(), SafeTensorsError> {
        let layout = raw_tensor_layout(&self.tensors, name, self.data_len)?;
        let absolute = self
            .data_start
            .checked_add(layout.offset)
            .ok_or_else(|| SafeTensorsError::OutOfBounds(name.to_owned()))?;
        visit_reader_range(
            &mut self.reader,
            absolute,
            layout.byte_len,
            max_chunk_bytes,
            visit,
        )
    }

    /// Seek to one tensor and return its row-major values widened to `f32`.
    /// Reads are chunked to bound raw staging memory independently of tensor size.
    ///
    /// # Errors
    /// Returns a typed [`SafeTensorsError`] for an absent/malformed tensor, a
    /// failed output reservation, or a failed absolute seek/read.
    pub fn tensor_f32(&mut self, name: &str) -> Result<Vec<f32>, SafeTensorsError> {
        let layout = tensor_layout(&self.tensors, name, self.data_len)?;
        let absolute = self
            .data_start
            .checked_add(layout.offset)
            .ok_or_else(|| SafeTensorsError::OutOfBounds(name.to_owned()))?;
        let mut output = reserve_f32(layout.numel)?;
        let alignment = layout.dtype.byte_size();
        let chunk_capacity = READ_CHUNK_BYTES - READ_CHUNK_BYTES % alignment;
        visit_reader_range(
            &mut self.reader,
            absolute,
            layout.byte_len,
            chunk_capacity,
            |raw| append_f32(&mut output, layout.dtype, raw),
        )?;
        debug_assert_eq!(output.len(), layout.numel);
        Ok(output)
    }
}

fn io_error(operation: &'static str, error: io::Error) -> SafeTensorsError {
    SafeTensorsError::Io {
        operation,
        kind: error.kind(),
        message: error.to_string(),
    }
}

/// Parse a safetensors buffer's header (no tensor data is copied), returning a
/// [`SafeTensors`] view borrowing `bytes`. Free-function entry point mirroring
/// [`crate::read_gguf`]; equivalent to [`SafeTensors::parse`].
///
/// # Errors
/// [`SafeTensorsError::TooShort`] / [`SafeTensorsError::BadHeaderLen`] /
/// [`SafeTensorsError::HeaderTooLarge`] on a malformed prefix;
/// [`SafeTensorsError::Json`] on an unparseable header; or a typed layout,
/// dtype, shape, or length error for invalid tensor metadata.
pub fn read_safetensors(bytes: &[u8]) -> Result<SafeTensors<'_>, SafeTensorsError> {
    SafeTensors::parse(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny safetensors buffer in memory for the roundtrip test.
    fn build(header: &str, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn parse_and_read_bf16_f16_f32() {
        // bf16 [2,2] = {1.0, -2.0, 0.5, -0.25}; f32 [3] = {3.0, -4.0, 0.125}.
        let mut data = Vec::new();
        for v in [1.0f32, -2.0, 0.5, -0.25] {
            data.extend_from_slice(&bf16::from_f32(v).to_bits().to_le_bytes());
        }
        for v in [9.0f32, -8.0] {
            data.extend_from_slice(&f16::from_f32(v).to_bits().to_le_bytes());
        }
        for v in [3.0f32, -4.0, 0.125] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        // offsets: bf16 0..8, f16 8..12, f32 12..24
        let header = r#"{"__metadata__":{"format":"pt"},"a_bf16":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},"b_f16":{"dtype":"F16","shape":[2],"data_offsets":[8,12]},"c_f32":{"dtype":"F32","shape":[3],"data_offsets":[12,24]}}"#;
        let buf = build(header, &data);

        let st = SafeTensors::parse(&buf).unwrap();
        assert_eq!(st.len(), 3, "__metadata__ is not a tensor");
        assert_eq!(st.shape("a_bf16"), Some(&[2usize, 2][..]));
        assert_eq!(st.dtype("a_bf16"), Some("BF16"));

        // bf16 widening is exact for these values.
        assert_eq!(
            st.tensor_f32("a_bf16").unwrap(),
            vec![1.0, -2.0, 0.5, -0.25]
        );
        assert_eq!(st.tensor_f32("b_f16").unwrap(), vec![9.0, -8.0]);
        assert_eq!(st.tensor_f32("c_f32").unwrap(), vec![3.0, -4.0, 0.125]);
    }

    #[test]
    fn errors_are_typed() {
        assert_eq!(
            SafeTensors::parse(&[0u8; 4]).unwrap_err(),
            SafeTensorsError::TooShort
        );

        // header len past the buffer
        let mut bad = 9999u64.to_le_bytes().to_vec();
        bad.extend_from_slice(b"{}");
        assert!(matches!(
            SafeTensors::parse(&bad),
            Err(SafeTensorsError::BadHeaderLen { .. })
        ));

        let header = r#"{"x":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
        let st_buf = build(header, &[0u8; 4]);
        let st = SafeTensors::parse(&st_buf).unwrap();
        assert!(matches!(
            st.tensor_f32("missing"),
            Err(SafeTensorsError::NotFound(_))
        ));

        // unsupported dtype
        let h2 = r#"{"x":{"dtype":"I64","shape":[1],"data_offsets":[0,8]}}"#;
        let b2 = build(h2, &[0u8; 8]);
        let st2 = SafeTensors::parse(&b2).unwrap();
        assert!(matches!(
            st2.tensor_f32("x"),
            Err(SafeTensorsError::UnsupportedDtype { .. })
        ));

        // shape/length mismatch: shape says 4 bf16 (8 bytes), offsets give 4
        let h3 = r#"{"x":{"dtype":"BF16","shape":[4],"data_offsets":[0,4]}}"#;
        let b3 = build(h3, &[0u8; 4]);
        assert!(matches!(
            SafeTensors::parse(&b3),
            Err(SafeTensorsError::LengthMismatch { .. })
        ));

        // crafted overflowing shape: product wraps usize → ShapeOverflow, no panic
        // (must hold in debug, where `*` would otherwise panic on overflow).
        let h4 = r#"{"x":{"dtype":"BF16","shape":[9223372036854775807,4],"data_offsets":[0,2]}}"#;
        let b4 = build(h4, &[0u8; 2]);
        assert!(matches!(
            SafeTensors::parse(&b4),
            Err(SafeTensorsError::ShapeOverflow { .. })
        ));
    }

    #[test]
    fn whole_container_layout_and_unique_names_fail_closed() {
        let cases = [
            (
                r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[1,5]}}"#,
                5,
            ),
            (
                r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
                4,
            ),
            (
                r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
                8,
            ),
            (
                r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"y":{"dtype":"F32","shape":[2],"data_offsets":[4,8]}}"#,
                8,
            ),
        ];
        for (header, data_len) in cases {
            let bytes = build(header, &vec![0; data_len]);
            assert!(SafeTensors::parse(&bytes).is_err(), "header: {header}");
            assert!(
                SafeTensorsReader::new(std::io::Cursor::new(bytes)).is_err(),
                "seek header: {header}"
            );
        }

        let duplicate = r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        assert!(matches!(
            SafeTensors::parse(&build(duplicate, &[0; 4])),
            Err(SafeTensorsError::Json(message)) if message.contains("duplicate header key")
        ));

        for header in [
            r#"{"__metadata__":{"format":7}}"#,
            r#"{"__metadata__":{"format":"pt","format":"torch"}}"#,
            r#"{"x":{"dtype":"F32","dtype":"F16","shape":[1],"data_offsets":[0,4]}}"#,
            r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"extra":0}}"#,
        ] {
            let bytes = build(header, &[0; 4]);
            assert!(
                matches!(SafeTensors::parse(&bytes), Err(SafeTensorsError::Json(_))),
                "borrowed header: {header}"
            );
            assert!(
                matches!(
                    SafeTensorsReader::new(std::io::Cursor::new(bytes)),
                    Err(SafeTensorsError::Json(_))
                ),
                "seek header: {header}"
            );
        }
    }
}
