//! Canonical semantic-model manifests and domain-separated artifact identities.
//!
//! [`ModelId`] names inference semantics: architecture configuration plus named,
//! shaped logical tensor content. [`PackageId`] names exact transport bytes.
//! Repacking the same logical model therefore preserves its model ID and changes
//! its package ID.

use core::fmt;

const MANIFEST_MAGIC: [u8; 4] = *b"TRSM";
const MANIFEST_VERSION: u8 = 1;
const CONFIG_HASH_CONTEXT: &str = "tritium semantic model config v1";
const TENSOR_HASH_CONTEXT: &str = "tritium semantic tensor content v1";
const MODEL_ID_CONTEXT: &str = "tritium semantic model id v1";
const PACKAGE_ID_CONTEXT: &str = "tritium transport package id v1";

fn domain_hash(context: &'static str, bytes: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key(context);
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}

fn write_len(out: &mut Vec<u8>, len: usize) {
    let len = u32::try_from(len).expect("manifest construction validates u32 field lengths");
    out.extend_from_slice(&len.to_le_bytes());
}

fn fmt_id(f: &mut fmt::Formatter<'_>, prefix: &str, digest: &[u8; 32]) -> fmt::Result {
    f.write_str(prefix)?;
    for byte in digest {
        write!(f, "{byte:02x}")?;
    }
    Ok(())
}

/// Why a semantic-model manifest could not be built or decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ArtifactError {
    /// The architecture identifier was empty.
    EmptyArchitecture,
    /// A tensor name was empty.
    EmptyTensorName,
    /// A tensor had no dimensions.
    EmptyTensorShape(String),
    /// A tensor dimension was zero.
    ZeroTensorDimension {
        /// Tensor containing the zero dimension.
        tensor: String,
        /// Zero-based dimension index.
        dimension: usize,
    },
    /// Two tensors used the same canonical name.
    DuplicateTensorName(String),
    /// A string, dimension list, or tensor list exceeded the canonical u32 count field.
    CountOverflow,
    /// The manifest did not begin with the semantic-manifest magic bytes.
    BadMagic,
    /// The manifest version is not supported by this build.
    UnsupportedVersion(u8),
    /// The manifest ended before a declared field was complete.
    Truncated,
    /// A manifest string was not valid UTF-8.
    InvalidUtf8,
    /// The bytes decoded but were not in canonical form.
    NonCanonical,
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyArchitecture => f.write_str("semantic manifest architecture is empty"),
            Self::EmptyTensorName => f.write_str("semantic tensor name is empty"),
            Self::EmptyTensorShape(name) => {
                write!(f, "semantic tensor `{name}` has no dimensions")
            }
            Self::ZeroTensorDimension { tensor, dimension } => write!(
                f,
                "semantic tensor `{tensor}` has zero-sized dimension {dimension}"
            ),
            Self::DuplicateTensorName(name) => {
                write!(f, "duplicate semantic tensor name `{name}`")
            }
            Self::CountOverflow => f.write_str("semantic manifest field exceeds u32 capacity"),
            Self::BadMagic => f.write_str("semantic manifest has bad magic"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported semantic manifest version {version}")
            }
            Self::Truncated => f.write_str("semantic manifest is truncated"),
            Self::InvalidUtf8 => f.write_str("semantic manifest contains invalid UTF-8"),
            Self::NonCanonical => f.write_str("semantic manifest is not canonically encoded"),
        }
    }
}

impl std::error::Error for ArtifactError {}

/// Content identity of a semantic model, independent of its transport package.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ModelId([u8; 32]);

impl ModelId {
    /// Return the raw 256-bit digest.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_id(f, "trm1_", &self.0)
    }
}

/// Content identity of exact package bytes, including layout and compression.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PackageId([u8; 32]);

impl PackageId {
    /// Hash exact package bytes using the versioned transport-ID domain.
    pub fn from_package_bytes(bytes: &[u8]) -> Self {
        let mut hasher = PackageHasher::new();
        hasher.update(bytes);
        hasher.finalize()
    }

    /// Return the raw 256-bit digest.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Incremental exact-byte hasher for a [`PackageId`].
///
/// Splitting an artifact across any number of [`update`](Self::update) calls
/// produces the same identity as [`PackageId::from_package_bytes`]. This lets
/// streaming writers identify large artifacts without retaining them in memory.
#[derive(Clone, Debug)]
pub struct PackageHasher(blake3::Hasher);

impl PackageHasher {
    /// Start hashing an exact transport package with the current package-ID domain.
    pub fn new() -> Self {
        Self(blake3::Hasher::new_derive_key(PACKAGE_ID_CONTEXT))
    }

    /// Add the next exact, ordered package-byte chunk.
    pub fn update(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }

    /// Finish hashing and return the package identity.
    pub fn finalize(self) -> PackageId {
        PackageId(*self.0.finalize().as_bytes())
    }
}

impl Default for PackageHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for PackageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_id(f, "trp1_", &self.0)
    }
}

/// One named tensor in a semantic-model manifest.
///
/// `logical_bytes` must use the architecture adapter's canonical logical
/// encoding, including dtype and scale semantics. Transport padding, compression,
/// and container offsets must not be included.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticTensor {
    name: String,
    shape: Vec<u64>,
    content_digest: [u8; 32],
}

impl SemanticTensor {
    /// Build and validate a tensor entry, hashing its canonical logical bytes.
    ///
    /// # Errors
    /// Returns [`ArtifactError`] for an empty name or invalid shape.
    pub fn new(
        name: impl Into<String>,
        shape: Vec<u64>,
        logical_bytes: &[u8],
    ) -> Result<Self, ArtifactError> {
        Self::from_digest(
            name.into(),
            shape,
            domain_hash(TENSOR_HASH_CONTEXT, logical_bytes),
        )
    }

    fn from_digest(
        name: String,
        shape: Vec<u64>,
        content_digest: [u8; 32],
    ) -> Result<Self, ArtifactError> {
        if name.is_empty() {
            return Err(ArtifactError::EmptyTensorName);
        }
        if shape.is_empty() {
            return Err(ArtifactError::EmptyTensorShape(name));
        }
        if shape.len() > u32::MAX as usize {
            return Err(ArtifactError::CountOverflow);
        }
        if let Some(dimension) = shape.iter().position(|&size| size == 0) {
            return Err(ArtifactError::ZeroTensorDimension {
                tensor: name,
                dimension,
            });
        }
        if name.len() > u32::MAX as usize {
            return Err(ArtifactError::CountOverflow);
        }
        Ok(Self {
            name,
            shape,
            content_digest,
        })
    }

    /// Return the canonical tensor name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the logical tensor shape.
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Return the domain-separated digest of canonical logical tensor bytes.
    pub fn content_digest(&self) -> &[u8; 32] {
        &self.content_digest
    }
}

/// Canonical semantic-model manifest used to derive a [`ModelId`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticModelManifest {
    architecture: String,
    config_digest: [u8; 32],
    tensors: Vec<SemanticTensor>,
}

impl SemanticModelManifest {
    /// Build a canonical manifest from architecture configuration and tensors.
    ///
    /// Tensor entries are sorted by name, making the resulting bytes independent
    /// of discovery order. `canonical_config_bytes` must exclude source paths and
    /// other provenance that does not affect inference semantics.
    ///
    /// # Errors
    /// Returns [`ArtifactError`] for empty architecture, count overflow, invalid
    /// tensor entries, or duplicate canonical tensor names.
    pub fn new(
        architecture: impl Into<String>,
        canonical_config_bytes: &[u8],
        tensors: Vec<SemanticTensor>,
    ) -> Result<Self, ArtifactError> {
        Self::from_parts(
            architecture.into(),
            domain_hash(CONFIG_HASH_CONTEXT, canonical_config_bytes),
            tensors,
        )
    }

    fn from_parts(
        architecture: String,
        config_digest: [u8; 32],
        mut tensors: Vec<SemanticTensor>,
    ) -> Result<Self, ArtifactError> {
        if architecture.is_empty() {
            return Err(ArtifactError::EmptyArchitecture);
        }
        if architecture.len() > u32::MAX as usize || tensors.len() > u32::MAX as usize {
            return Err(ArtifactError::CountOverflow);
        }
        tensors.sort_by(|left, right| left.name.cmp(&right.name));
        if let Some(pair) = tensors.windows(2).find(|pair| pair[0].name == pair[1].name) {
            return Err(ArtifactError::DuplicateTensorName(pair[0].name.clone()));
        }
        Ok(Self {
            architecture,
            config_digest,
            tensors,
        })
    }

    /// Return the canonical architecture identifier.
    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    /// Return the digest of canonical inference configuration bytes.
    pub fn config_digest(&self) -> &[u8; 32] {
        &self.config_digest
    }

    /// Return tensors sorted by canonical name.
    pub fn tensors(&self) -> &[SemanticTensor] {
        &self.tensors
    }

    /// Serialize the versioned canonical manifest encoding.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&MANIFEST_MAGIC);
        out.push(MANIFEST_VERSION);
        write_len(&mut out, self.architecture.len());
        out.extend_from_slice(self.architecture.as_bytes());
        out.extend_from_slice(&self.config_digest);
        write_len(&mut out, self.tensors.len());
        for tensor in &self.tensors {
            write_len(&mut out, tensor.name.len());
            out.extend_from_slice(tensor.name.as_bytes());
            write_len(&mut out, tensor.shape.len());
            for &dimension in &tensor.shape {
                out.extend_from_slice(&dimension.to_le_bytes());
            }
            out.extend_from_slice(&tensor.content_digest);
        }
        out
    }

    /// Decode a manifest only if its bytes use the canonical representation.
    ///
    /// # Errors
    /// Returns [`ArtifactError`] for malformed, unsupported, duplicate, or
    /// non-canonical input. The parser bounds every allocation by available bytes.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ArtifactError> {
        let mut cursor = Cursor::new(bytes);
        if cursor.take(4)? != MANIFEST_MAGIC {
            return Err(ArtifactError::BadMagic);
        }
        let version = cursor.u8()?;
        if version != MANIFEST_VERSION {
            return Err(ArtifactError::UnsupportedVersion(version));
        }
        let architecture = cursor.string()?;
        let config_digest = cursor.digest()?;
        let tensor_count = cursor.u32()?;
        // Even a syntactically invalid entry needs name_len + rank + digest.
        // Reject impossible counts before parsing and grow only as entries are
        // successfully decoded; never preallocate directly from untrusted input.
        const MIN_ENCODED_TENSOR_BYTES: usize = 4 + 4 + 32;
        if tensor_count > cursor.remaining() / MIN_ENCODED_TENSOR_BYTES {
            return Err(ArtifactError::Truncated);
        }
        let mut tensors = Vec::new();
        for _ in 0..tensor_count {
            let name = cursor.string()?;
            let rank = cursor.u32()?;
            if rank > cursor.remaining() / 8 {
                return Err(ArtifactError::Truncated);
            }
            let mut shape = Vec::with_capacity(rank);
            for _ in 0..rank {
                shape.push(cursor.u64()?);
            }
            let content_digest = cursor.digest()?;
            tensors.push(SemanticTensor::from_digest(name, shape, content_digest)?);
        }
        if cursor.remaining() != 0 {
            return Err(ArtifactError::NonCanonical);
        }
        if architecture.is_empty() {
            return Err(ArtifactError::EmptyArchitecture);
        }
        for pair in tensors.windows(2) {
            if pair[0].name == pair[1].name {
                return Err(ArtifactError::DuplicateTensorName(pair[0].name.clone()));
            }
            if pair[0].name > pair[1].name {
                return Err(ArtifactError::NonCanonical);
            }
        }
        let manifest = Self {
            architecture,
            config_digest,
            tensors,
        };
        if manifest.canonical_bytes() != bytes {
            return Err(ArtifactError::NonCanonical);
        }
        Ok(manifest)
    }

    /// Derive the semantic identity from canonical manifest bytes.
    pub fn model_id(&self) -> ModelId {
        ModelId(domain_hash(MODEL_ID_CONTEXT, &self.canonical_bytes()))
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], ArtifactError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(ArtifactError::Truncated)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(ArtifactError::Truncated)?;
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, ArtifactError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<usize, ArtifactError> {
        let bytes: [u8; 4] = self
            .take(4)?
            .try_into()
            .map_err(|_| ArtifactError::Truncated)?;
        Ok(u32::from_le_bytes(bytes) as usize)
    }

    fn u64(&mut self) -> Result<u64, ArtifactError> {
        let bytes: [u8; 8] = self
            .take(8)?
            .try_into()
            .map_err(|_| ArtifactError::Truncated)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn digest(&mut self) -> Result<[u8; 32], ArtifactError> {
        self.take(32)?
            .try_into()
            .map_err(|_| ArtifactError::Truncated)
    }

    fn string(&mut self) -> Result<String, ArtifactError> {
        let len = self.u32()?;
        let bytes = self.take(len)?;
        core::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| ArtifactError::InvalidUtf8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(name: &str, value: &[u8]) -> SemanticTensor {
        SemanticTensor::new(name, vec![2, 2], value).expect("valid tensor")
    }

    #[test]
    fn model_identity_is_canonical_and_order_independent() {
        let a = SemanticModelManifest::new(
            "qwen3",
            br#"{"hidden_size":4}"#,
            vec![tensor("z.weight", b"z"), tensor("a.weight", b"a")],
        )
        .expect("valid manifest");
        let b = SemanticModelManifest::new(
            "qwen3",
            br#"{"hidden_size":4}"#,
            vec![tensor("a.weight", b"a"), tensor("z.weight", b"z")],
        )
        .expect("valid manifest");

        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
        assert_eq!(a.model_id(), b.model_id());
        assert_eq!(a.tensors()[0].name(), "a.weight");
    }

    #[test]
    fn semantic_and_transport_ids_are_domain_separated() {
        let manifest = SemanticModelManifest::new(
            "gemma4",
            b"canonical-config",
            vec![tensor("model.embed.weight", b"logical-values")],
        )
        .expect("valid manifest");
        let canonical = manifest.canonical_bytes();
        let model_id = manifest.model_id();
        let package_id = PackageId::from_package_bytes(&canonical);

        assert_ne!(model_id.as_bytes(), package_id.as_bytes());
        assert_eq!(
            model_id.to_string(),
            "trm1_66944bf9f401ee8fbd12119229e4823ec65183b8802289debd94b2b72b76b3ba"
        );
        assert_eq!(
            package_id.to_string(),
            "trp1_c7d2cae8984389043f021ccdf376cf4fa10ea1ab3160671262695ecaa18b5129"
        );
    }

    #[test]
    fn canonical_manifest_roundtrips_and_rejects_corruption() {
        let manifest = SemanticModelManifest::new(
            "glm5",
            b"config",
            vec![tensor("a", b"one"), tensor("b", b"two")],
        )
        .expect("valid manifest");
        let bytes = manifest.canonical_bytes();

        assert_eq!(
            SemanticModelManifest::from_canonical_bytes(&bytes).expect("roundtrip"),
            manifest
        );
        assert!(SemanticModelManifest::from_canonical_bytes(&bytes[..bytes.len() - 1]).is_err());

        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(SemanticModelManifest::from_canonical_bytes(&trailing).is_err());

        let mut bad_magic = bytes;
        bad_magic[0] ^= 0xff;
        assert!(matches!(
            SemanticModelManifest::from_canonical_bytes(&bad_magic),
            Err(ArtifactError::BadMagic)
        ));
    }

    #[test]
    fn duplicate_tensor_names_are_rejected() {
        let err = SemanticModelManifest::new(
            "qwen3",
            b"config",
            vec![tensor("same", b"one"), tensor("same", b"two")],
        )
        .expect_err("duplicate names must fail");

        assert!(matches!(err, ArtifactError::DuplicateTensorName(_)));
    }

    #[test]
    fn package_identity_changes_with_transport_bytes() {
        let a = PackageId::from_package_bytes(b"package-a");
        let b = PackageId::from_package_bytes(b"package-b");
        assert_ne!(a, b);
    }

    #[test]
    fn incremental_package_identity_matches_one_shot_for_any_chunking() {
        let bytes = b"exact package bytes spanning several deliberately uneven chunks";
        let expected = PackageId::from_package_bytes(bytes);

        for chunk_len in [1, 2, 3, 7, 16, bytes.len()] {
            let mut hasher = PackageHasher::new();
            for chunk in bytes.chunks(chunk_len) {
                hasher.update(chunk);
            }
            assert_eq!(hasher.finalize(), expected);
        }

        assert_eq!(
            PackageHasher::default().finalize(),
            PackageId::from_package_bytes(&[])
        );
    }
}
