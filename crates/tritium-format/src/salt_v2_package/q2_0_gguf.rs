//! Streaming CompactV1 P=1 SALT V2 to standard Q2_0 GGUF export.

use core::fmt;
use std::collections::BTreeMap;
use std::io::{Read, Seek, Write};

use crate::{GGML_TYPE_Q2_0, GgufStreamWriter, GgufTensorSpec, GgufValue, GgufWriteError};

use super::reader::{
    CompactQ2VisitError, validate_compact_q2_0_tensor,
    visit_compact_q2_0_tensor_without_package_verification,
};
use super::{CompactQ2ExportError, SaltV2PackageReadError, SaltV2PackageReader};

/// Reserved GGUF metadata key binding output to exact source SALT V2 package bytes.
pub const COMPACT_Q2_SOURCE_PACKAGE_ID_KEY: &str = "tritium.q2_0.source_salt_v2_package_id";

/// Reserved GGUF metadata key naming exact export compatibility profile.
pub const COMPACT_Q2_EXPORT_PROFILE_KEY: &str = "tritium.q2_0.export_profile";

const COMPACT_Q2_EXPORT_PROFILE: &str = "compact-v1-p1-g128";

/// Errors from streaming a compatible SALT V2 package into standard Q2_0 GGUF.
#[derive(Debug)]
#[non_exhaustive]
pub enum CompactQ2GgufExportError {
    /// One named tensor failed CompactV1 Q2_0 compatibility or conversion.
    Tensor {
        /// Tensor name from the SALT V2 package.
        name: String,
        /// Exact tensor-level compatibility or conversion error.
        source: CompactQ2ExportError,
    },
    /// Caller metadata attempted to define an exporter-owned provenance key.
    ReservedMetadataKey(String),
    /// Export-plan storage could not be reserved.
    AllocationFailed {
        /// Exact number of bytes requested by the failed reservation.
        requested_bytes: usize,
    },
    /// Terminal exact-package integrity verification failed.
    Read(SaltV2PackageReadError),
    /// GGUF layout or destination streaming failed.
    Gguf(GgufWriteError),
}

impl fmt::Display for CompactQ2GgufExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tensor { name, source } => {
                write!(f, "Compact Q2_0 GGUF tensor `{name}`: {source}")
            }
            Self::ReservedMetadataKey(key) => write!(
                f,
                "Compact Q2_0 GGUF metadata key `{key}` is reserved by exporter"
            ),
            Self::AllocationFailed { requested_bytes } => write!(
                f,
                "Compact Q2_0 GGUF plan allocation of {requested_bytes} bytes failed"
            ),
            Self::Read(error) => write!(f, "Compact Q2_0 GGUF source: {error}"),
            Self::Gguf(error) => write!(f, "Compact Q2_0 GGUF output: {error}"),
        }
    }
}

impl std::error::Error for CompactQ2GgufExportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Tensor { source, .. } => Some(source),
            Self::Read(error) => Some(error),
            Self::Gguf(error) => Some(error),
            Self::ReservedMetadataKey(_) | Self::AllocationFailed { .. } => None,
        }
    }
}

impl From<SaltV2PackageReadError> for CompactQ2GgufExportError {
    fn from(error: SaltV2PackageReadError) -> Self {
        Self::Read(error)
    }
}

impl From<GgufWriteError> for CompactQ2GgufExportError {
    fn from(error: GgufWriteError) -> Self {
        Self::Gguf(error)
    }
}

/// Stream every compatible SALT V2 tensor into a standard Q2_0 GGUF container.
///
/// SALT dimensions are row-major semantic order; GGUF dimensions are emitted in
/// reverse, fastest-varying-first order. Every tensor is compatibility-checked
/// before the GGUF header reaches `writer`. Output metadata binds the exact source
/// [`crate::PackageId`] and export profile using exporter-owned reserved keys.
/// Conversion retains only one allocation tile's 72-byte Q2_0 output plus bounded
/// SALT decode scratch. Source package identity is verified once after all visits,
/// avoiding one whole-package rehash per tensor. Caller metadata is consumed so
/// large metadata values are not cloned infallibly.
///
/// Destination writes are not transactional. On any returned error, callers must
/// discard the destination and publish it only after `Ok` returns.
///
/// # Errors
/// Returns a typed tensor error if any source tensor is incompatible, rejects
/// caller-owned reserved metadata, propagates GGUF layout/I/O failures, and fails
/// terminally if exact SALT V2 source bytes changed during conversion.
pub fn write_compact_q2_0_gguf<R: Read + Seek, W: Write>(
    reader: &mut SaltV2PackageReader<R>,
    writer: W,
    version: u32,
    mut metadata: BTreeMap<String, GgufValue>,
) -> Result<W, CompactQ2GgufExportError> {
    for key in [
        COMPACT_Q2_SOURCE_PACKAGE_ID_KEY,
        COMPACT_Q2_EXPORT_PROFILE_KEY,
    ] {
        if metadata.contains_key(key) {
            return Err(CompactQ2GgufExportError::ReservedMetadataKey(
                key.to_owned(),
            ));
        }
    }

    let mut names = Vec::new();
    names.try_reserve_exact(reader.len()).map_err(|_| {
        CompactQ2GgufExportError::AllocationFailed {
            requested_bytes: reader.len().saturating_mul(core::mem::size_of::<String>()),
        }
    })?;
    for name in reader.tensor_names_encoded_order() {
        names.push(try_clone_string(name)?);
    }
    let mut specs = Vec::new();
    specs.try_reserve_exact(names.len()).map_err(|_| {
        CompactQ2GgufExportError::AllocationFailed {
            requested_bytes: names
                .len()
                .saturating_mul(core::mem::size_of::<GgufTensorSpec>()),
        }
    })?;
    for name in &names {
        let data_len = match validate_compact_q2_0_tensor(reader, name) {
            Ok(data_len) => data_len,
            Err(source) => {
                return Err(CompactQ2GgufExportError::Tensor {
                    name: try_clone_string(name)?,
                    source,
                });
            }
        };
        let source_dims = reader
            .tensor_info(name)
            .expect("validation found the named tensor")
            .dims();
        let mut dims = Vec::new();
        dims.try_reserve_exact(source_dims.len()).map_err(|_| {
            CompactQ2GgufExportError::AllocationFailed {
                requested_bytes: source_dims
                    .len()
                    .saturating_mul(core::mem::size_of::<u64>()),
            }
        })?;
        dims.extend_from_slice(source_dims);
        dims.reverse();
        let data_len = match u64::try_from(data_len) {
            Ok(data_len) => data_len,
            Err(_) => {
                return Err(CompactQ2GgufExportError::Tensor {
                    name: try_clone_string(name)?,
                    source: CompactQ2ExportError::LengthOverflow,
                });
            }
        };
        specs.push(GgufTensorSpec {
            name: try_clone_string(name)?,
            dims,
            ggml_type: GGML_TYPE_Q2_0,
            data_len,
        });
    }

    metadata.insert(
        COMPACT_Q2_SOURCE_PACKAGE_ID_KEY.to_owned(),
        GgufValue::String(reader.package_id().to_string()),
    );
    metadata.insert(
        COMPACT_Q2_EXPORT_PROFILE_KEY.to_owned(),
        GgufValue::String(COMPACT_Q2_EXPORT_PROFILE.to_owned()),
    );
    let mut stream = GgufStreamWriter::new(writer, version, &metadata, &specs)?;
    for (tensor_index, name) in names.iter().enumerate() {
        match visit_compact_q2_0_tensor_without_package_verification(reader, name, |chunk| {
            stream.write_tensor_chunk(tensor_index, chunk)
        }) {
            Ok(()) => {}
            Err(CompactQ2VisitError::Export(source)) => {
                return Err(CompactQ2GgufExportError::Tensor {
                    name: try_clone_string(name)?,
                    source,
                });
            }
            Err(CompactQ2VisitError::Sink(error)) => return Err(error.into()),
        }
    }
    reader.verify_unchanged()?;
    Ok(stream.finish()?)
}

fn try_clone_string(value: &str) -> Result<String, CompactQ2GgufExportError> {
    let mut output = String::new();
    output.try_reserve_exact(value.len()).map_err(|_| {
        CompactQ2GgufExportError::AllocationFailed {
            requested_bytes: value.len(),
        }
    })?;
    output.push_str(value);
    Ok(output)
}
