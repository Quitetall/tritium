//! Reference codec for the versioned `.trit` package envelope.

use core::fmt;
use std::collections::BTreeMap;

use ciborium::value::{CanonicalValue, Value};
use tritium_core::DType;
use tritium_schema::BlobId;

use crate::artifact::PackageId;

/// `.trit` file magic.
pub const TRIT_MAGIC: [u8; 8] = *b"TRITIUM\0";
const TRIT_FOOTER_MAGIC: [u8; 8] = *b"TRITEND\0";
/// Fixed `.trit` header size from ADR 0044.
pub const TRIT_HEADER_BYTES: usize = 64;
const FOOTER_BYTES: usize = 72;
/// Required blob alignment from ADR 0044.
pub const TRIT_BLOB_ALIGNMENT: usize = 256;
/// Current `.trit` major version.
pub const TRIT_PACKAGE_MAJOR: u16 = 1;
const TRIT_PACKAGE_MINOR: u16 = 0;
const DIRECTORY_VERSION: u64 = 1;
const MAX_BLOBS: usize = 1_000_000;
const MAX_MANIFEST_BYTES: usize = 256 * 1024 * 1024;
const MAX_DIRECTORY_BYTES: usize = 256 * 1024 * 1024;

/// Semantic kind associated with one content-addressed blob.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TritBlobKind {
    /// Additive ternary tensor payload.
    AdditiveTensor,
    /// Dense tensor payload with its element dtype.
    DenseTensor(DType),
    /// Model asset such as config, tokenizer, or chat template.
    Asset(String),
}

/// A borrowed payload supplied to the `.trit` package writer.
#[derive(Clone, Debug)]
pub struct TritBlob<'a> {
    /// Semantic kind recorded in the package directory.
    pub kind: TritBlobKind,
    /// Exact stored blob bytes.
    pub bytes: &'a [u8],
}

/// Indexed package entry for one blob kind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TritBlobInfo {
    /// BLAKE3 content identity of the exact blob bytes.
    pub id: BlobId,
    /// Semantic kind recorded in the directory.
    pub kind: TritBlobKind,
    /// Byte offset from the start of the package.
    pub offset: u64,
    /// Exact stored byte length.
    pub length: u64,
}

/// A validated, borrowed `.trit` package view.
#[derive(Debug)]
pub struct TritPackage<'a> {
    bytes: &'a [u8],
    manifest: &'a [u8],
    blobs: Vec<TritBlobInfo>,
}

impl<'a> TritPackage<'a> {
    /// Borrow the canonical CBOR manifest bytes.
    #[must_use]
    pub const fn manifest(&self) -> &'a [u8] {
        self.manifest
    }

    /// Borrow indexed blob records in canonical directory order.
    #[must_use]
    pub fn blobs(&self) -> &[TritBlobInfo] {
        &self.blobs
    }

    /// Return the exact-byte package identity.
    #[must_use]
    pub fn package_id(&self) -> PackageId {
        PackageId::from_package_bytes(self.bytes)
    }

    /// Borrow and re-verify one blob by its directory entry.
    pub fn blob(&self, info: &TritBlobInfo) -> Result<&'a [u8], TritPackageError> {
        let start = usize::try_from(info.offset).map_err(|_| TritPackageError::LengthOverflow)?;
        let len = usize::try_from(info.length).map_err(|_| TritPackageError::LengthOverflow)?;
        let end = start
            .checked_add(len)
            .ok_or(TritPackageError::LengthOverflow)?;
        let bytes = self
            .bytes
            .get(start..end)
            .ok_or(TritPackageError::InvalidDirectory)?;
        if blake3::hash(bytes).as_bytes() != info.id.as_bytes() {
            return Err(TritPackageError::BlobDigestMismatch(info.id));
        }
        Ok(bytes)
    }
}

/// Failure writing or validating a `.trit` package.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TritPackageError {
    /// Input is shorter than the minimum package envelope.
    Truncated,
    /// Package header or footer magic is wrong.
    BadMagic,
    /// The package major/minor version is unsupported.
    UnsupportedVersion {
        /// Major version read from the header.
        major: u16,
        /// Minor version read from the header.
        minor: u16,
    },
    /// Required-feature bits include a feature unknown to this implementation.
    UnknownRequiredFeatures(u32),
    /// Reserved header bytes are not zero.
    NonZeroReserved,
    /// A metadata or blob length cannot be represented safely.
    LengthOverflow,
    /// Manifest or directory exceeds its explicit parser bound.
    MetadataLimitExceeded,
    /// Manifest is not deterministic, definite-length CBOR in the supported profile.
    NonCanonicalManifest,
    /// Directory bytes are malformed or not in canonical form.
    InvalidDirectory,
    /// A declared metadata or blob digest does not match its bytes.
    DigestMismatch,
    /// A blob digest disagrees with the bytes at its declared offset.
    BlobDigestMismatch(BlobId),
    /// Two distinct payloads have the same content identity (cryptographic collision).
    BlobIdentityCollision,
    /// A fallible output allocation failed.
    AllocationFailed {
        /// Exact byte or element count requested by the failed reservation.
        requested: usize,
    },
    /// An asset name is empty.
    EmptyAssetName,
}

impl fmt::Display for TritPackageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("trit package is truncated"),
            Self::BadMagic => f.write_str("trit package header or footer magic is invalid"),
            Self::UnsupportedVersion { major, minor } => {
                write!(f, "unsupported trit package version {major}.{minor}")
            }
            Self::UnknownRequiredFeatures(bits) => {
                write!(f, "unknown required trit package feature bits {bits:#010x}")
            }
            Self::NonZeroReserved => f.write_str("trit package reserved bytes are nonzero"),
            Self::LengthOverflow => f.write_str("trit package length arithmetic overflowed"),
            Self::MetadataLimitExceeded => f.write_str("trit package metadata exceeds its limit"),
            Self::NonCanonicalManifest => {
                f.write_str("trit package manifest is not canonical CBOR")
            }
            Self::InvalidDirectory => f.write_str("trit package directory is invalid"),
            Self::DigestMismatch => f.write_str("trit package metadata digest mismatch"),
            Self::BlobDigestMismatch(id) => write!(f, "trit blob {id:?} digest mismatch"),
            Self::BlobIdentityCollision => {
                f.write_str("distinct trit blob payloads have the same content identity")
            }
            Self::AllocationFailed { requested } => {
                write!(f, "trit package allocation of {requested} bytes failed")
            }
            Self::EmptyAssetName => f.write_str("trit asset name must not be empty"),
        }
    }
}

impl std::error::Error for TritPackageError {}

/// Build a deterministic `.trit` package in memory.
///
/// Blob records are deduplicated by BLAKE3 content identity and aligned to 256
/// bytes. The manifest and footer-located directory use deterministic CBOR;
/// the directory and manifest bytes are independently hashed in the footer.
/// Manifest maps use canonical key order; floating-point values are not admitted
/// in this manifest profile (store tensor scales in typed blobs instead).
pub fn write_trit_package(
    manifest_cbor: &[u8],
    blobs: &[TritBlob<'_>],
) -> Result<Vec<u8>, TritPackageError> {
    if manifest_cbor.len() > MAX_MANIFEST_BYTES || blobs.len() > MAX_BLOBS {
        return Err(TritPackageError::MetadataLimitExceeded);
    }
    validate_canonical_manifest(manifest_cbor)?;
    for blob in blobs {
        if let TritBlobKind::Asset(name) = &blob.kind
            && name.is_empty()
        {
            return Err(TritPackageError::EmptyAssetName);
        }
    }

    let mut unique = BTreeMap::<[u8; 32], (&[u8], u64)>::new();
    for blob in blobs {
        let digest = *blake3::hash(blob.bytes).as_bytes();
        let length =
            u64::try_from(blob.bytes.len()).map_err(|_| TritPackageError::LengthOverflow)?;
        if let Some((existing, _)) = unique.get(&digest) {
            if *existing != blob.bytes {
                return Err(TritPackageError::BlobIdentityCollision);
            }
        } else {
            unique.insert(digest, (blob.bytes, length));
        }
    }

    let mut offsets = BTreeMap::<[u8; 32], u64>::new();
    let mut cursor = TRIT_HEADER_BYTES;
    for (digest, (payload, _)) in &unique {
        cursor = align_up(cursor)?;
        offsets.insert(
            *digest,
            u64::try_from(cursor).map_err(|_| TritPackageError::LengthOverflow)?,
        );
        cursor = cursor
            .checked_add(payload.len())
            .ok_or(TritPackageError::LengthOverflow)?;
    }
    let manifest_offset = cursor;
    cursor = cursor
        .checked_add(manifest_cbor.len())
        .ok_or(TritPackageError::LengthOverflow)?;
    let directory_offset = cursor;
    let mut directory_entries = Vec::new();
    directory_entries
        .try_reserve_exact(blobs.len())
        .map_err(|_| TritPackageError::AllocationFailed {
            requested: blobs.len(),
        })?;
    for blob in blobs {
        let id = *blake3::hash(blob.bytes).as_bytes();
        directory_entries.push(TritBlobInfo {
            id: BlobId::from_bytes(id),
            kind: blob.kind.clone(),
            offset: offsets[&id],
            length: u64::try_from(blob.bytes.len())
                .map_err(|_| TritPackageError::LengthOverflow)?,
        });
    }
    directory_entries.sort_by(|left, right| {
        left.id
            .as_bytes()
            .cmp(right.id.as_bytes())
            .then_with(|| kind_sort_key(&left.kind).cmp(&kind_sort_key(&right.kind)))
    });
    directory_entries.dedup_by(|right, left| right.id == left.id && right.kind == left.kind);
    let directory = encode_directory(&directory_entries)?;
    if directory.len() > MAX_DIRECTORY_BYTES {
        return Err(TritPackageError::MetadataLimitExceeded);
    }
    cursor = cursor
        .checked_add(directory.len())
        .and_then(|value| value.checked_add(FOOTER_BYTES))
        .ok_or(TritPackageError::LengthOverflow)?;

    let mut output = Vec::new();
    output
        .try_reserve_exact(cursor)
        .map_err(|_| TritPackageError::AllocationFailed { requested: cursor })?;
    output.resize(TRIT_HEADER_BYTES, 0);
    for (digest, (payload, _)) in unique {
        let expected_offset =
            usize::try_from(offsets[&digest]).map_err(|_| TritPackageError::LengthOverflow)?;
        output.resize(expected_offset, 0);
        output.extend_from_slice(payload);
    }
    if output.len() != manifest_offset {
        return Err(TritPackageError::InvalidDirectory);
    }
    output.extend_from_slice(manifest_cbor);
    if output.len() != directory_offset {
        return Err(TritPackageError::InvalidDirectory);
    }
    output.extend_from_slice(&directory);
    output.extend_from_slice(&TRIT_FOOTER_MAGIC);
    output.extend_from_slice(blake3::hash(manifest_cbor).as_bytes());
    output.extend_from_slice(blake3::hash(&directory).as_bytes());

    output[..8].copy_from_slice(&TRIT_MAGIC);
    output[8..10].copy_from_slice(&TRIT_PACKAGE_MAJOR.to_le_bytes());
    output[10..12].copy_from_slice(&TRIT_PACKAGE_MINOR.to_le_bytes());
    output[12..16].copy_from_slice(&0u32.to_le_bytes());
    write_u64(&mut output, 16, manifest_offset)?;
    write_u64(&mut output, 24, manifest_cbor.len())?;
    write_u64(&mut output, 32, directory_offset)?;
    write_u64(&mut output, 40, directory.len())?;
    Ok(output)
}

/// Parse and fully validate a `.trit` package held in memory.
pub fn read_trit_package(bytes: &[u8]) -> Result<TritPackage<'_>, TritPackageError> {
    if bytes.len() < TRIT_HEADER_BYTES + FOOTER_BYTES {
        return Err(TritPackageError::Truncated);
    }
    if bytes[..8] != TRIT_MAGIC
        || bytes[bytes.len() - FOOTER_BYTES..bytes.len() - FOOTER_BYTES + 8] != TRIT_FOOTER_MAGIC
    {
        return Err(TritPackageError::BadMagic);
    }
    let major = read_u16(bytes, 8)?;
    let minor = read_u16(bytes, 10)?;
    if major != TRIT_PACKAGE_MAJOR || minor != TRIT_PACKAGE_MINOR {
        return Err(TritPackageError::UnsupportedVersion { major, minor });
    }
    let features = read_u32(bytes, 12)?;
    if features != 0 {
        return Err(TritPackageError::UnknownRequiredFeatures(features));
    }
    if bytes[48..TRIT_HEADER_BYTES].iter().any(|byte| *byte != 0) {
        return Err(TritPackageError::NonZeroReserved);
    }
    let manifest_offset = as_usize(read_u64(bytes, 16)?)?;
    let manifest_len = as_usize(read_u64(bytes, 24)?)?;
    let directory_offset = as_usize(read_u64(bytes, 32)?)?;
    let directory_len = as_usize(read_u64(bytes, 40)?)?;
    if manifest_len > MAX_MANIFEST_BYTES || directory_len > MAX_DIRECTORY_BYTES {
        return Err(TritPackageError::MetadataLimitExceeded);
    }
    let manifest_end = manifest_offset
        .checked_add(manifest_len)
        .ok_or(TritPackageError::LengthOverflow)?;
    let directory_end = directory_offset
        .checked_add(directory_len)
        .ok_or(TritPackageError::LengthOverflow)?;
    let footer_offset = bytes.len() - FOOTER_BYTES;
    if manifest_offset < TRIT_HEADER_BYTES
        || manifest_end != directory_offset
        || directory_end != footer_offset
    {
        return Err(TritPackageError::InvalidDirectory);
    }
    let manifest = bytes
        .get(manifest_offset..manifest_end)
        .ok_or(TritPackageError::Truncated)?;
    let directory = bytes
        .get(directory_offset..directory_end)
        .ok_or(TritPackageError::Truncated)?;
    let footer = &bytes[footer_offset + 8..];
    if blake3::hash(manifest).as_bytes() != &footer[..32]
        || blake3::hash(directory).as_bytes() != &footer[32..]
    {
        return Err(TritPackageError::DigestMismatch);
    }
    validate_canonical_manifest(manifest)?;
    let blobs = decode_directory(directory)?;
    validate_blob_ranges(bytes, manifest_offset, &blobs)?;
    Ok(TritPackage {
        bytes,
        manifest,
        blobs,
    })
}

fn validate_blob_ranges(
    package: &[u8],
    manifest_offset: usize,
    blobs: &[TritBlobInfo],
) -> Result<(), TritPackageError> {
    let mut unique = BTreeMap::<[u8; 32], (usize, usize)>::new();
    for blob in blobs {
        let offset = as_usize(blob.offset)?;
        let length = as_usize(blob.length)?;
        let end = offset
            .checked_add(length)
            .ok_or(TritPackageError::LengthOverflow)?;
        if offset < TRIT_HEADER_BYTES || offset % TRIT_BLOB_ALIGNMENT != 0 || end > manifest_offset
        {
            return Err(TritPackageError::InvalidDirectory);
        }
        if let Some(previous) = unique.insert(*blob.id.as_bytes(), (offset, end))
            && previous != (offset, end)
        {
            return Err(TritPackageError::InvalidDirectory);
        }
        let payload = package
            .get(offset..end)
            .ok_or(TritPackageError::InvalidDirectory)?;
        if blake3::hash(payload).as_bytes() != blob.id.as_bytes() {
            return Err(TritPackageError::BlobDigestMismatch(blob.id));
        }
    }
    let mut ranges: Vec<(usize, usize)> = unique.values().copied().collect();
    ranges.sort_unstable();
    let mut previous_end = TRIT_HEADER_BYTES;
    for (start, end) in ranges {
        if start < previous_end || package[previous_end..start].iter().any(|byte| *byte != 0) {
            return Err(TritPackageError::InvalidDirectory);
        }
        previous_end = end;
    }
    if package[previous_end..manifest_offset]
        .iter()
        .any(|byte| *byte != 0)
    {
        return Err(TritPackageError::InvalidDirectory);
    }
    Ok(())
}

fn encode_directory(blobs: &[TritBlobInfo]) -> Result<Vec<u8>, TritPackageError> {
    let entries = blobs
        .iter()
        .map(|blob| {
            let (kind, aux) = encode_kind(&blob.kind);
            Value::Array(vec![
                Value::Bytes(blob.id.as_bytes().to_vec()),
                Value::Integer(kind.into()),
                aux,
                Value::Integer(blob.offset.into()),
                Value::Integer(blob.length.into()),
            ])
        })
        .collect();
    encode_cbor_value(&Value::Array(vec![
        Value::Integer(DIRECTORY_VERSION.into()),
        Value::Array(entries),
    ]))
}

fn decode_directory(bytes: &[u8]) -> Result<Vec<TritBlobInfo>, TritPackageError> {
    let value: Value =
        ciborium::de::from_reader(bytes).map_err(|_| TritPackageError::InvalidDirectory)?;
    if encode_cbor_value(&value)? != bytes {
        return Err(TritPackageError::InvalidDirectory);
    }
    let Value::Array(root) = value else {
        return Err(TritPackageError::InvalidDirectory);
    };
    if root.len() != 2 || integer(&root[0])? != DIRECTORY_VERSION {
        return Err(TritPackageError::InvalidDirectory);
    }
    let Value::Array(entries) = &root[1] else {
        return Err(TritPackageError::InvalidDirectory);
    };
    if entries.len() > MAX_BLOBS {
        return Err(TritPackageError::MetadataLimitExceeded);
    }
    let mut blobs = Vec::new();
    blobs
        .try_reserve_exact(entries.len())
        .map_err(|_| TritPackageError::AllocationFailed {
            requested: entries.len(),
        })?;
    for entry in entries {
        let Value::Array(fields) = entry else {
            return Err(TritPackageError::InvalidDirectory);
        };
        if fields.len() != 5 {
            return Err(TritPackageError::InvalidDirectory);
        }
        let Value::Bytes(id) = &fields[0] else {
            return Err(TritPackageError::InvalidDirectory);
        };
        let id: [u8; 32] = id
            .as_slice()
            .try_into()
            .map_err(|_| TritPackageError::InvalidDirectory)?;
        let kind_tag = integer(&fields[1])?;
        let kind = decode_kind(kind_tag, &fields[2])?;
        blobs.push(TritBlobInfo {
            id: BlobId::from_bytes(id),
            kind,
            offset: integer(&fields[3])?,
            length: integer(&fields[4])?,
        });
    }
    if blobs
        .windows(2)
        .any(|pair| compare_blob_entries(&pair[0], &pair[1]).is_ge())
    {
        return Err(TritPackageError::InvalidDirectory);
    }
    if encode_directory(&blobs)? != bytes {
        return Err(TritPackageError::InvalidDirectory);
    }
    Ok(blobs)
}

fn encode_kind(kind: &TritBlobKind) -> (u64, Value) {
    match kind {
        TritBlobKind::AdditiveTensor => (0, Value::Null),
        TritBlobKind::DenseTensor(dtype) => (1, Value::Text(dtype.to_string())),
        TritBlobKind::Asset(name) => (2, Value::Text(name.clone())),
    }
}

fn decode_kind(tag: u64, value: &Value) -> Result<TritBlobKind, TritPackageError> {
    match (tag, value) {
        (0, Value::Null) => Ok(TritBlobKind::AdditiveTensor),
        (1, Value::Text(name)) => dtype_from_str(name).map(TritBlobKind::DenseTensor),
        (2, Value::Text(name)) if !name.is_empty() => Ok(TritBlobKind::Asset(name.clone())),
        _ => Err(TritPackageError::InvalidDirectory),
    }
}

fn dtype_from_str(name: &str) -> Result<DType, TritPackageError> {
    match name {
        "ternary" => Ok(DType::Ternary),
        "i4" => Ok(DType::I4),
        "i8" => Ok(DType::I8),
        "u8" => Ok(DType::U8),
        "f8e4m3" => Ok(DType::F8E4M3),
        "f8e5m2" => Ok(DType::F8E5M2),
        "f4e2m1" => Ok(DType::F4E2M1),
        "f16" => Ok(DType::F16),
        "bf16" => Ok(DType::BF16),
        "f32" => Ok(DType::F32),
        _ => Err(TritPackageError::InvalidDirectory),
    }
}

fn validate_canonical_manifest(bytes: &[u8]) -> Result<(), TritPackageError> {
    if bytes.is_empty() || bytes.len() > MAX_MANIFEST_BYTES {
        return Err(TritPackageError::NonCanonicalManifest);
    }
    let mut value: Value =
        ciborium::de::from_reader(bytes).map_err(|_| TritPackageError::NonCanonicalManifest)?;
    canonicalize_value(&mut value, 0)?;
    if encode_cbor_value(&value)? != bytes {
        return Err(TritPackageError::NonCanonicalManifest);
    }
    Ok(())
}

fn canonicalize_value(value: &mut Value, depth: usize) -> Result<(), TritPackageError> {
    if depth > 64 {
        return Err(TritPackageError::NonCanonicalManifest);
    }
    match value {
        Value::Array(items) => {
            for item in items {
                canonicalize_value(item, depth + 1)?;
            }
        }
        Value::Map(entries) => {
            for (key, item) in entries.iter_mut() {
                canonicalize_value(key, depth + 1)?;
                canonicalize_value(item, depth + 1)?;
            }
            entries.sort_by_cached_key(|entry| CanonicalValue::from(entry.0.clone()));
            if entries.windows(2).any(|pair| pair[0].0 == pair[1].0) {
                return Err(TritPackageError::NonCanonicalManifest);
            }
        }
        Value::Tag(_, inner) => canonicalize_value(inner, depth + 1)?,
        Value::Float(_) => return Err(TritPackageError::NonCanonicalManifest),
        _ => {}
    }
    Ok(())
}

fn encode_cbor_value(value: &Value) -> Result<Vec<u8>, TritPackageError> {
    let mut bytes = Vec::new();
    ciborium::ser::into_writer(value, &mut bytes)
        .map_err(|_| TritPackageError::InvalidDirectory)?;
    Ok(bytes)
}

fn integer(value: &Value) -> Result<u64, TritPackageError> {
    value
        .as_integer()
        .and_then(|number| u64::try_from(number).ok())
        .ok_or(TritPackageError::InvalidDirectory)
}

fn kind_sort_key(kind: &TritBlobKind) -> (u8, String) {
    match kind {
        TritBlobKind::AdditiveTensor => (0, String::new()),
        TritBlobKind::DenseTensor(dtype) => (1, dtype.to_string()),
        TritBlobKind::Asset(name) => (2, name.clone()),
    }
}

fn compare_blob_entries(left: &TritBlobInfo, right: &TritBlobInfo) -> core::cmp::Ordering {
    left.id
        .as_bytes()
        .cmp(right.id.as_bytes())
        .then_with(|| kind_sort_key(&left.kind).cmp(&kind_sort_key(&right.kind)))
}

fn align_up(value: usize) -> Result<usize, TritPackageError> {
    value
        .checked_add(TRIT_BLOB_ALIGNMENT - 1)
        .map(|value| value / TRIT_BLOB_ALIGNMENT * TRIT_BLOB_ALIGNMENT)
        .ok_or(TritPackageError::LengthOverflow)
}

fn write_u64(bytes: &mut [u8], offset: usize, value: usize) -> Result<(), TritPackageError> {
    let value = u64::try_from(value).map_err(|_| TritPackageError::LengthOverflow)?;
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, TritPackageError> {
    Ok(u16::from_le_bytes(
        bytes
            .get(offset..offset + 2)
            .ok_or(TritPackageError::Truncated)?
            .try_into()
            .map_err(|_| TritPackageError::Truncated)?,
    ))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, TritPackageError> {
    Ok(u32::from_le_bytes(
        bytes
            .get(offset..offset + 4)
            .ok_or(TritPackageError::Truncated)?
            .try_into()
            .map_err(|_| TritPackageError::Truncated)?,
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, TritPackageError> {
    Ok(u64::from_le_bytes(
        bytes
            .get(offset..offset + 8)
            .ok_or(TritPackageError::Truncated)?
            .try_into()
            .map_err(|_| TritPackageError::Truncated)?,
    ))
}

fn as_usize(value: u64) -> Result<usize, TritPackageError> {
    usize::try_from(value).map_err(|_| TritPackageError::LengthOverflow)
}

#[cfg(test)]
mod tests {
    use super::{
        TRIT_BLOB_ALIGNMENT, TritBlob, TritBlobKind, TritPackageError, read_trit_package,
        validate_canonical_manifest, write_trit_package,
    };
    use crate::PackageId;
    use tritium_core::DType;

    const MANIFEST: &[u8] = &[0x82, 0x01, 0x02]; // canonical CBOR [1, 2]

    #[test]
    fn package_roundtrips_content_addressed_blobs_and_manifest() {
        let package = write_trit_package(
            MANIFEST,
            &[
                TritBlob {
                    kind: TritBlobKind::AdditiveTensor,
                    bytes: b"ternary payload",
                },
                TritBlob {
                    kind: TritBlobKind::DenseTensor(DType::F16),
                    bytes: b"dense payload",
                },
                TritBlob {
                    kind: TritBlobKind::Asset("tokenizer.json".into()),
                    bytes: b"{}",
                },
            ],
        )
        .expect("package writes");
        let parsed = read_trit_package(&package).expect("package reads");
        assert_eq!(parsed.manifest(), MANIFEST);
        assert_eq!(parsed.blobs().len(), 3);
        assert_eq!(
            parsed.package_id().as_bytes(),
            PackageId::from_package_bytes(&package).as_bytes()
        );
        for blob in parsed.blobs() {
            assert_eq!(blob.offset as usize % TRIT_BLOB_ALIGNMENT, 0);
            assert_eq!(
                parsed.blob(blob).expect("blob digest verifies").len() as u64,
                blob.length
            );
        }
        assert_eq!(
            write_trit_package(
                MANIFEST,
                &[
                    TritBlob {
                        kind: TritBlobKind::AdditiveTensor,
                        bytes: b"ternary payload",
                    },
                    TritBlob {
                        kind: TritBlobKind::DenseTensor(DType::F16),
                        bytes: b"dense payload",
                    },
                    TritBlob {
                        kind: TritBlobKind::Asset("tokenizer.json".into()),
                        bytes: b"{}",
                    },
                ],
            )
            .expect("deterministic repeat write"),
            package
        );
    }

    #[test]
    fn identical_payloads_share_aligned_storage() {
        let package = write_trit_package(
            MANIFEST,
            &[
                TritBlob {
                    kind: TritBlobKind::AdditiveTensor,
                    bytes: b"shared",
                },
                TritBlob {
                    kind: TritBlobKind::Asset("alias.bin".into()),
                    bytes: b"shared",
                },
                TritBlob {
                    kind: TritBlobKind::AdditiveTensor,
                    bytes: b"shared",
                },
            ],
        )
        .expect("package writes");
        let parsed = read_trit_package(&package).expect("package reads");
        assert_eq!(parsed.blobs().len(), 2);
        assert_eq!(parsed.blobs()[0].offset, parsed.blobs()[1].offset);
        assert_eq!(parsed.blobs()[0].id, parsed.blobs()[1].id);
    }

    #[test]
    fn reader_rejects_unknown_features_and_tampered_payload_or_manifest() {
        let package = write_trit_package(
            MANIFEST,
            &[TritBlob {
                kind: TritBlobKind::Asset("asset".into()),
                bytes: b"content",
            }],
        )
        .expect("package writes");
        let mut unknown_feature = package.clone();
        unknown_feature[12] = 1;
        assert!(matches!(
            read_trit_package(&unknown_feature),
            Err(TritPackageError::UnknownRequiredFeatures(1))
        ));

        let mut changed_manifest = package.clone();
        let manifest_offset = u64::from_le_bytes(
            changed_manifest[16..24]
                .try_into()
                .expect("header manifest offset"),
        ) as usize;
        changed_manifest[manifest_offset] ^= 1;
        assert!(matches!(
            read_trit_package(&changed_manifest),
            Err(TritPackageError::DigestMismatch)
        ));

        let mut changed_blob = package;
        let parsed = read_trit_package(&changed_blob).expect("valid package");
        let offset = usize::try_from(parsed.blobs()[0].offset).expect("host offset");
        changed_blob[offset] ^= 1;
        assert!(matches!(
            read_trit_package(&changed_blob),
            Err(TritPackageError::BlobDigestMismatch(_))
        ));
    }

    #[test]
    fn manifest_must_use_deterministic_supported_cbor() {
        assert!(validate_canonical_manifest(&[0x82, 0x01, 0x02]).is_ok());
        assert_eq!(
            validate_canonical_manifest(&[0x82, 0x18, 0x01, 0x02]),
            Err(TritPackageError::NonCanonicalManifest)
        );
        assert_eq!(
            validate_canonical_manifest(&[0xf9, 0x3c, 0x00]), // float16 1.0
            Err(TritPackageError::NonCanonicalManifest)
        );
        let canonical_map = [0xa2, 0x61, b'a', 0x01, 0x62, b'b', b'b', 0x02];
        assert!(validate_canonical_manifest(&canonical_map).is_ok());
        let reversed_map = [0xa2, 0x62, b'b', b'b', 0x02, 0x61, b'a', 0x01];
        assert_eq!(
            validate_canonical_manifest(&reversed_map),
            Err(TritPackageError::NonCanonicalManifest)
        );
        let duplicate_key = [0xa2, 0x61, b'a', 0x01, 0x61, b'a', 0x02];
        assert_eq!(
            validate_canonical_manifest(&duplicate_key),
            Err(TritPackageError::NonCanonicalManifest)
        );
    }
}
