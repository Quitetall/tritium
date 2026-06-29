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

use half::{bf16, f16};
use serde::Deserialize;

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
        }
    }
}

impl std::error::Error for SafeTensorsError {}

/// One tensor's header entry.
#[derive(Debug, Clone, Deserialize)]
struct RawTensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
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
    /// [`SafeTensorsError::TooShort`] / [`SafeTensorsError::BadHeaderLen`] on a
    /// malformed prefix; [`SafeTensorsError::Json`] on an unparseable header.
    pub fn parse(buf: &'a [u8]) -> Result<Self, SafeTensorsError> {
        if buf.len() < 8 {
            return Err(SafeTensorsError::TooShort);
        }
        let n = u64::from_le_bytes(buf[0..8].try_into().unwrap()) as usize;
        let header_end = 8usize.checked_add(n).filter(|&e| e <= buf.len()).ok_or(
            SafeTensorsError::BadHeaderLen {
                declared: n,
                available: buf.len().saturating_sub(8),
            },
        )?;

        // Parse as a generic map so the optional `__metadata__` string-map entry
        // (which is not a tensor) can be dropped before typed deserialization.
        let mut raw: BTreeMap<String, serde_json::Value> =
            serde_json::from_slice(&buf[8..header_end])
                .map_err(|e| SafeTensorsError::Json(e.to_string()))?;
        raw.remove("__metadata__");

        let mut tensors = BTreeMap::new();
        for (name, value) in raw {
            let t: RawTensor =
                serde_json::from_value(value).map_err(|e| SafeTensorsError::Json(e.to_string()))?;
            tensors.insert(name, t);
        }
        Ok(SafeTensors {
            tensors,
            data: &buf[header_end..],
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
    /// [`UnsupportedDtype`](SafeTensorsError::UnsupportedDtype).
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, SafeTensorsError> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| SafeTensorsError::NotFound(name.to_owned()))?;
        let [a, b] = t.data_offsets;
        let raw = self
            .data
            .get(a..b)
            .ok_or_else(|| SafeTensorsError::OutOfBounds(name.to_owned()))?;

        let dsize = match t.dtype.as_str() {
            "BF16" | "F16" => 2usize,
            "F32" => 4,
            other => {
                return Err(SafeTensorsError::UnsupportedDtype {
                    name: name.to_owned(),
                    dtype: other.to_owned(),
                });
            }
        };
        // Untrusted header: a crafted shape can overflow `usize`, which would make
        // the byte-length check below meaningless. Compute `numel × dsize` with
        // checked arithmetic and reject overflow rather than wrap.
        let expected = t
            .shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .and_then(|numel| numel.checked_mul(dsize))
            .ok_or_else(|| SafeTensorsError::ShapeOverflow {
                name: name.to_owned(),
            })?;
        if raw.len() != expected {
            return Err(SafeTensorsError::LengthMismatch {
                name: name.to_owned(),
                expected,
                got: raw.len(),
            });
        }

        let out = match t.dtype.as_str() {
            "BF16" => raw
                .chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            "F16" => raw
                .chunks_exact(2)
                .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            "F32" => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            _ => unreachable!("dtype validated above"),
        };
        Ok(out)
    }
}

/// Parse a safetensors buffer's header (no tensor data is copied), returning a
/// [`SafeTensors`] view borrowing `bytes`. Free-function entry point mirroring
/// [`crate::read_gguf`]; equivalent to [`SafeTensors::parse`].
///
/// # Errors
/// [`SafeTensorsError::TooShort`] / [`SafeTensorsError::BadHeaderLen`] on a
/// malformed prefix; [`SafeTensorsError::Json`] on an unparseable header.
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
        let st3 = SafeTensors::parse(&b3).unwrap();
        assert!(matches!(
            st3.tensor_f32("x"),
            Err(SafeTensorsError::LengthMismatch { .. })
        ));

        // crafted overflowing shape: product wraps usize → ShapeOverflow, no panic
        // (must hold in debug, where `*` would otherwise panic on overflow).
        let h4 = r#"{"x":{"dtype":"BF16","shape":[9223372036854775807,4],"data_offsets":[0,2]}}"#;
        let b4 = build(h4, &[0u8; 2]);
        let st4 = SafeTensors::parse(&b4).unwrap();
        assert!(matches!(
            st4.tensor_f32("x"),
            Err(SafeTensorsError::ShapeOverflow { .. })
        ));
    }
}
