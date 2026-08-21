//! Pinned Qwen3.6-27B source-metadata coverage and conversion policy.
//!
//! The coverage manifest is deliberately model- and revision-specific. It
//! admits exactly the official SafeTensors metadata set selected by the active
//! campaign, classifies every tensor, and excludes the packaged vision tower
//! from the current language-plus-MTP conversion scope.
//!
//! This seam binds canonical source metadata only. Before conversion, its
//! caller must pair the manifest with the campaign's separately verified
//! source [`tritium_format::ModelId`], which binds tensor content.

use core::fmt;

/// Immutable Qwen3.6-27B revision selected by the active campaign.
pub const QWEN36_27B_COVERAGE_REVISION: &str = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9";

const CANONICAL_POLICY_MAGIC: [u8; 8] = *b"TQ36COV\0";
const CANONICAL_POLICY_VERSION: u8 = 1;
const POLICY_DIGEST_CONTEXT: &str = "tritium qwen3.6 coverage policy v1";
const EXPECTED_TENSORS: usize = 1_199;
const MAX_NAME_BYTES: usize = 128;
const MAX_DTYPE_BYTES: usize = 16;
const MAX_RANK: usize = 5;
const MAX_REVISION_BYTES: usize = 128;
const EXPECTED_METADATA_RECORD_BYTES: u64 = 75_705;
const EXPECTED_METADATA_DIGEST: [u8; 32] = [
    0xad, 0xd3, 0x32, 0xd2, 0x3a, 0x10, 0x12, 0xaa, 0x1d, 0x77, 0x33, 0x9e, 0x04, 0x61, 0x30, 0x42,
    0xaa, 0x56, 0xfb, 0x7c, 0xab, 0x49, 0x08, 0xf7, 0x72, 0x6a, 0xc7, 0x8b, 0x78, 0xef, 0x2d, 0x8e,
];

const HIDDEN: &[u64] = &[5_120];
const HEAD: &[u64] = &[248_320, 5_120];
const MLP_DOWN: &[u64] = &[5_120, 17_408];
const MLP_UP: &[u64] = &[17_408, 5_120];
const FULL_K_NORM: &[u64] = &[256];
const FULL_KV: &[u64] = &[1_024, 5_120];
const FULL_O: &[u64] = &[5_120, 6_144];
const FULL_Q: &[u64] = &[12_288, 5_120];
const DELTA_CHANNELS: &[u64] = &[48];
const DELTA_CONV: &[u64] = &[10_240, 1, 4];
const DELTA_AB: &[u64] = &[48, 5_120];
const DELTA_QKV: &[u64] = &[10_240, 5_120];
const DELTA_Z: &[u64] = &[6_144, 5_120];
const DELTA_NORM: &[u64] = &[128];
const DELTA_OUT: &[u64] = &[5_120, 6_144];
const MTP_FC: &[u64] = &[5_120, 10_240];
const VISION_HIDDEN: &[u64] = &[1_152];
const VISION_QKV_BIAS: &[u64] = &[3_456];
const VISION_QKV: &[u64] = &[3_456, 1_152];
const VISION_PROJ: &[u64] = &[1_152, 1_152];
const VISION_MLP_UP_BIAS: &[u64] = &[4_304];
const VISION_MLP_UP: &[u64] = &[4_304, 1_152];
const VISION_MLP_DOWN: &[u64] = &[1_152, 4_304];
const VISION_PATCH: &[u64] = &[1_152, 3, 2, 16, 16];
const VISION_POSITION: &[u64] = &[2_304, 1_152];
const VISION_MERGER_HIDDEN: &[u64] = &[4_608];
const VISION_MERGER_UP: &[u64] = &[4_608, 4_608];
const VISION_MERGER_DOWN: &[u64] = &[5_120, 4_608];

/// Borrowed source tensor metadata accepted by the coverage seam.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen35TensorMetadata<'a> {
    name: &'a str,
    dtype: &'a str,
    shape: &'a [u64],
}

impl<'a> Qwen35TensorMetadata<'a> {
    /// Construct borrowed metadata for one SafeTensors entry.
    pub const fn new(name: &'a str, dtype: &'a str, shape: &'a [u64]) -> Self {
        Self { name, dtype, shape }
    }
}

/// Model component that owns a pinned source tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Qwen35TensorScope {
    /// Main autoregressive language model, including embedding and output head.
    Language,
    /// Bundled one-layer multi-token-prediction drafter.
    MtpDrafter,
    /// Packaged vision tower and projector, deferred from the current product slice.
    DeferredVision,
}

/// Exact semantic role assigned to a Qwen3.6-27B tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Qwen35TensorRole {
    /// Language-model token embedding table.
    TokenEmbedding,
    /// Untied language-model output head.
    OutputHead,
    /// Scale or bias used by a normalization operation.
    Normalization,
    /// Dense language or MTP feed-forward projection.
    MlpProjection,
    /// Full-attention query, key, value, or output projection.
    FullAttentionProjection,
    /// Gated DeltaNet input or output projection.
    DeltaNetProjection,
    /// Gated DeltaNet recurrent-state parameter.
    DeltaNetState,
    /// Gated DeltaNet causal convolution kernel.
    DeltaNetConvolution,
    /// MTP hidden-and-embedding fusion projection.
    MtpFusionProjection,
    /// Vision-transformer attention projection.
    VisionAttentionProjection,
    /// Vision-transformer feed-forward projection.
    VisionMlpProjection,
    /// Vision patch-embedding convolution.
    VisionPatchEmbedding,
    /// Vision positional embedding table.
    VisionPositionalEmbedding,
    /// Vision-to-language merger projection.
    VisionMergerProjection,
    /// Additive bias outside the role-specific normalization category.
    Bias,
}

/// Frozen conversion action for a covered source tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Qwen35CoverageDisposition {
    /// Apply sensitivity-aware additive ternary quantization.
    AdditiveTernary,
    /// Retain the exact BF16 source tensor in the language-plus-MTP artifact.
    PreserveSource,
    /// Metadata-cover but exclude the vision tensor from the current product scope.
    ExcludedFutureVision,
}

/// Language-layer execution kind fixed by the checkpoint schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Qwen35LanguageLayerKind {
    /// Gated DeltaNet linear-attention layer.
    DeltaNet,
    /// Conventional gated full-attention layer.
    FullAttention,
}

/// Source precision admitted by the pinned coverage contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Qwen35SourceDtype {
    /// Brain floating point with eight exponent bits and seven mantissa bits.
    Bfloat16,
}

impl Qwen35SourceDtype {
    const fn canonical_name(self) -> &'static str {
        match self {
            Self::Bfloat16 => "BF16",
        }
    }
}

/// One deterministic, validated coverage-manifest entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen35CoverageEntry {
    name: String,
    dtype: Qwen35SourceDtype,
    shape: Vec<u64>,
    coefficients: u64,
    scope: Qwen35TensorScope,
    role: Qwen35TensorRole,
    disposition: Qwen35CoverageDisposition,
}

impl Qwen35CoverageEntry {
    /// Canonical source tensor name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Validated source precision.
    pub const fn dtype(&self) -> Qwen35SourceDtype {
        self.dtype
    }

    /// Logical tensor dimensions in SafeTensors order.
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Checked product of all logical dimensions.
    pub const fn coefficients(&self) -> u64 {
        self.coefficients
    }

    /// Model component owning this tensor.
    pub const fn scope(&self) -> Qwen35TensorScope {
        self.scope
    }

    /// Exact semantic role assigned by the pinned adapter.
    pub const fn role(&self) -> Qwen35TensorRole {
        self.role
    }

    /// Frozen conversion action for this tensor.
    pub const fn disposition(&self) -> Qwen35CoverageDisposition {
        self.disposition
    }
}

/// Tensor and coefficient totals for one coverage partition.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Qwen35CoverageTotals {
    tensors: u64,
    coefficients: u64,
}

impl Qwen35CoverageTotals {
    /// Number of source tensors in the partition.
    pub const fn tensors(self) -> u64 {
        self.tensors
    }

    /// Number of logical source coefficients in the partition.
    pub const fn coefficients(self) -> u64 {
        self.coefficients
    }

    fn add(&mut self, coefficients: u64) -> Result<(), Qwen35CoverageError> {
        self.tensors = self
            .tensors
            .checked_add(1)
            .ok_or(Qwen35CoverageError::SummaryOverflow)?;
        self.coefficients = self
            .coefficients
            .checked_add(coefficients)
            .ok_or(Qwen35CoverageError::SummaryOverflow)?;
        Ok(())
    }
}

/// Frozen coverage totals by component and disposition.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Qwen35CoverageSummary {
    total: Qwen35CoverageTotals,
    language: Qwen35CoverageTotals,
    mtp: Qwen35CoverageTotals,
    vision: Qwen35CoverageTotals,
    included: Qwen35CoverageTotals,
    additive_ternary: Qwen35CoverageTotals,
    preserve_source: Qwen35CoverageTotals,
    excluded_future_vision: Qwen35CoverageTotals,
}

impl Qwen35CoverageSummary {
    /// Totals across language, MTP, and deferred vision metadata.
    pub const fn total(self) -> Qwen35CoverageTotals {
        self.total
    }

    /// Totals for the main language model.
    pub const fn language(self) -> Qwen35CoverageTotals {
        self.language
    }

    /// Totals for the bundled MTP drafter.
    pub const fn mtp(self) -> Qwen35CoverageTotals {
        self.mtp
    }

    /// Metadata-covered totals for the vision component excluded from this product scope.
    pub const fn vision(self) -> Qwen35CoverageTotals {
        self.vision
    }

    /// Totals included in the current language-plus-MTP artifact.
    pub const fn included(self) -> Qwen35CoverageTotals {
        self.included
    }

    /// Totals assigned to additive ternary quantization.
    pub const fn additive_ternary(self) -> Qwen35CoverageTotals {
        self.additive_ternary
    }

    /// Included totals retained at exact source precision.
    pub const fn preserve_source(self) -> Qwen35CoverageTotals {
        self.preserve_source
    }

    /// Totals assigned the exact preregistered future-vision exclusion.
    pub const fn excluded_future_vision(self) -> Qwen35CoverageTotals {
        self.excluded_future_vision
    }

    fn include(&mut self, entry: &Qwen35CoverageEntry) -> Result<(), Qwen35CoverageError> {
        self.total.add(entry.coefficients)?;
        match entry.scope {
            Qwen35TensorScope::Language => {
                self.language.add(entry.coefficients)?;
                self.included.add(entry.coefficients)?;
            }
            Qwen35TensorScope::MtpDrafter => {
                self.mtp.add(entry.coefficients)?;
                self.included.add(entry.coefficients)?;
            }
            Qwen35TensorScope::DeferredVision => self.vision.add(entry.coefficients)?,
        }
        match entry.disposition {
            Qwen35CoverageDisposition::AdditiveTernary => {
                self.additive_ternary.add(entry.coefficients)?
            }
            Qwen35CoverageDisposition::PreserveSource => {
                self.preserve_source.add(entry.coefficients)?
            }
            Qwen35CoverageDisposition::ExcludedFutureVision => {
                self.excluded_future_vision.add(entry.coefficients)?
            }
        }
        Ok(())
    }
}

/// Deterministic coverage and conversion policy for the pinned checkpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen35CoverageManifest {
    entries: Vec<Qwen35CoverageEntry>,
    summary: Qwen35CoverageSummary,
    metadata_digest: [u8; 32],
    metadata_record_bytes: u64,
}

impl Qwen35CoverageManifest {
    /// Validate official tensor metadata and freeze its complete conversion policy.
    ///
    /// Input order does not affect the result. The method bounds tensor count,
    /// name length, dtype length, and rank before copying, uses fallible reserve
    /// operations for caller-controlled metadata, and checks every coefficient
    /// product and aggregate.
    ///
    /// This constructor does not read tensor payloads. The campaign driver must
    /// pair a successful manifest with its separately verified source
    /// [`tritium_format::ModelId`] before conversion begins.
    ///
    /// # Errors
    /// Returns [`Qwen35CoverageError`] unless the revision, complete tensor set,
    /// dtype, names, shapes, layer schedule, canonical metadata digest, and
    /// frozen coverage totals all match the campaign-pinned checkpoint.
    pub fn from_metadata<'a>(
        pinned_revision: &str,
        metadata: impl IntoIterator<Item = Qwen35TensorMetadata<'a>>,
    ) -> Result<Self, Qwen35CoverageError> {
        if pinned_revision != QWEN36_27B_COVERAGE_REVISION {
            return Err(Qwen35CoverageError::WrongRevision);
        }

        let mut entries = Vec::new();
        entries
            .try_reserve_exact(EXPECTED_TENSORS)
            .map_err(|_| Qwen35CoverageError::AllocationFailed)?;

        for tensor in metadata {
            if entries.len() == EXPECTED_TENSORS {
                return Err(Qwen35CoverageError::TooManyTensors {
                    maximum: EXPECTED_TENSORS,
                });
            }
            entries.push(validate_entry(tensor)?);
        }

        entries.sort_unstable_by(|left, right| left.name.cmp(&right.name));
        if let Some(pair) = entries.windows(2).find(|pair| pair[0].name == pair[1].name) {
            return Err(Qwen35CoverageError::DuplicateTensor(bounded_string(
                &pair[0].name,
                "tensor name",
                MAX_NAME_BYTES,
            )?));
        }
        if entries.len() != EXPECTED_TENSORS {
            return Err(Qwen35CoverageError::MissingTensorMetadata {
                expected: EXPECTED_TENSORS,
                actual: entries.len(),
            });
        }

        let (metadata_digest, metadata_record_bytes) = hash_metadata(&entries)?;
        if metadata_digest != EXPECTED_METADATA_DIGEST
            || metadata_record_bytes != EXPECTED_METADATA_RECORD_BYTES
        {
            return Err(Qwen35CoverageError::MetadataIdentityMismatch {
                expected_digest: EXPECTED_METADATA_DIGEST,
                actual_digest: metadata_digest,
                expected_bytes: EXPECTED_METADATA_RECORD_BYTES,
                actual_bytes: metadata_record_bytes,
            });
        }

        let mut summary = Qwen35CoverageSummary::default();
        for entry in &entries {
            summary.include(entry)?;
        }
        if !summary_is_expected(summary) {
            return Err(Qwen35CoverageError::CoverageSummaryMismatch);
        }

        Ok(Self {
            entries,
            summary,
            metadata_digest,
            metadata_record_bytes,
        })
    }

    /// Canonical entries sorted by source tensor name.
    pub fn entries(&self) -> &[Qwen35CoverageEntry] {
        &self.entries
    }

    /// Frozen totals by source component and conversion disposition.
    pub const fn summary(&self) -> Qwen35CoverageSummary {
        self.summary
    }

    /// BLAKE3 digest of sorted `name\tdtype\tcomma-shape\n` records.
    pub const fn metadata_digest(&self) -> &[u8; 32] {
        &self.metadata_digest
    }

    /// Length of the canonical metadata record stream in bytes.
    pub const fn metadata_record_bytes(&self) -> u64 {
        self.metadata_record_bytes
    }

    /// Exact SafeTensors payload bytes required by this all-BF16 source policy.
    ///
    /// Construction has already validated the frozen coefficient totals and
    /// that every source tensor uses two-byte BF16 storage.
    pub const fn expected_source_payload_bytes(&self) -> u64 {
        self.summary.total.coefficients * 2
    }

    /// Canonical bytes binding every admitted tensor to its exact conversion action.
    ///
    /// Unlike [`Self::metadata_digest`], this record includes scope, semantic
    /// role, disposition, and checked coefficient count as well as source
    /// name, dtype, and shape. It is therefore suitable for durable campaign
    /// admission and recipe provenance.
    #[must_use]
    pub fn canonical_policy_bytes(&self) -> Vec<u8> {
        let mut output = Vec::new();
        output.extend_from_slice(&CANONICAL_POLICY_MAGIC);
        output.push(CANONICAL_POLICY_VERSION);
        write_policy_bytes(&mut output, QWEN36_27B_COVERAGE_REVISION.as_bytes());
        write_policy_u32(&mut output, self.entries.len());
        for entry in &self.entries {
            write_policy_bytes(&mut output, entry.name.as_bytes());
            output.push(dtype_tag(entry.dtype));
            write_policy_u32(&mut output, entry.shape.len());
            for &dimension in &entry.shape {
                output.extend_from_slice(&dimension.to_le_bytes());
            }
            output.extend_from_slice(&entry.coefficients.to_le_bytes());
            output.extend_from_slice(&[
                scope_tag(entry.scope),
                role_tag(entry.role),
                disposition_tag(entry.disposition),
            ]);
        }
        output
    }

    /// Domain-separated identity of [`Self::canonical_policy_bytes`].
    #[must_use]
    pub fn policy_digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new_derive_key(POLICY_DIGEST_CONTEXT);
        hasher.update(&self.canonical_policy_bytes());
        *hasher.finalize().as_bytes()
    }

    /// Decode and revalidate an exact canonical coverage-policy record.
    ///
    /// The decoder never trusts serialized classifications. It reconstructs
    /// the pinned manifest from name, dtype, and shape, then requires every
    /// serialized coefficient count, scope, role, and disposition to equal the
    /// current immutable policy and finally requires byte-for-byte canonical
    /// re-encoding.
    ///
    /// # Errors
    /// Returns [`Qwen35CoverageError`] for malformed, unsupported, noncanonical,
    /// or no-longer-pinned policy bytes.
    pub fn from_canonical_policy_bytes(bytes: &[u8]) -> Result<Self, Qwen35CoverageError> {
        let mut cursor = PolicyCursor::new(bytes);
        if cursor.take(CANONICAL_POLICY_MAGIC.len())? != CANONICAL_POLICY_MAGIC {
            return Err(Qwen35CoverageError::MalformedCanonicalPolicy("magic"));
        }
        let version = cursor.u8()?;
        if version != CANONICAL_POLICY_VERSION {
            return Err(Qwen35CoverageError::UnsupportedCanonicalPolicyVersion(
                version,
            ));
        }
        let revision = cursor.string(MAX_REVISION_BYTES, "revision")?;
        if revision != QWEN36_27B_COVERAGE_REVISION {
            return Err(Qwen35CoverageError::WrongRevision);
        }
        let entry_count = cursor.u32()? as usize;
        if entry_count != EXPECTED_TENSORS {
            return Err(Qwen35CoverageError::MalformedCanonicalPolicy(
                "tensor count",
            ));
        }

        let mut decoded = Vec::new();
        decoded
            .try_reserve_exact(entry_count)
            .map_err(|_| Qwen35CoverageError::AllocationFailed)?;
        for _ in 0..entry_count {
            let name = cursor.string(MAX_NAME_BYTES, "tensor name")?.to_owned();
            let dtype = cursor.u8()?;
            if dtype != dtype_tag(Qwen35SourceDtype::Bfloat16) {
                return Err(Qwen35CoverageError::MalformedCanonicalPolicy("dtype tag"));
            }
            let rank = cursor.u32()? as usize;
            if rank == 0 || rank > MAX_RANK {
                return Err(Qwen35CoverageError::MalformedCanonicalPolicy("rank"));
            }
            let mut shape = Vec::with_capacity(rank);
            for _ in 0..rank {
                shape.push(cursor.u64()?);
            }
            decoded.push(DecodedPolicyEntry {
                name,
                shape,
                coefficients: cursor.u64()?,
                scope: cursor.u8()?,
                role: cursor.u8()?,
                disposition: cursor.u8()?,
            });
        }
        if cursor.remaining() != 0 {
            return Err(Qwen35CoverageError::NonCanonicalPolicy);
        }

        let manifest = Self::from_metadata(
            revision,
            decoded
                .iter()
                .map(|entry| Qwen35TensorMetadata::new(&entry.name, "BF16", &entry.shape)),
        )?;
        for (encoded, actual) in decoded.iter().zip(&manifest.entries) {
            if encoded.name != actual.name
                || encoded.shape != actual.shape
                || encoded.coefficients != actual.coefficients
                || encoded.scope != scope_tag(actual.scope)
                || encoded.role != role_tag(actual.role)
                || encoded.disposition != disposition_tag(actual.disposition)
            {
                return Err(Qwen35CoverageError::NonCanonicalPolicy);
            }
        }
        if manifest.canonical_policy_bytes() != bytes {
            return Err(Qwen35CoverageError::NonCanonicalPolicy);
        }
        Ok(manifest)
    }
}

/// Why pinned Qwen3.6-27B metadata was rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Qwen35CoverageError {
    /// Revision did not equal the immutable campaign revision.
    WrongRevision,
    /// Input exceeded the bounded official tensor count.
    TooManyTensors {
        /// Maximum admitted tensor count.
        maximum: usize,
    },
    /// A tensor name was empty.
    EmptyTensorName,
    /// A caller-controlled metadata field exceeded its bound.
    MetadataFieldTooLong {
        /// Field rejected by the bound.
        field: &'static str,
        /// Maximum admitted bytes.
        maximum: usize,
    },
    /// Source tensor dtype was not exactly BF16.
    UnsupportedDtype(String),
    /// Tensor name is outside the pinned checkpoint contract.
    UnknownTensor(String),
    /// Tensor used an attention family forbidden by its scheduled layer kind.
    WrongLayerKind {
        /// Rejected tensor name.
        tensor: String,
        /// Zero-based language layer index.
        layer: u8,
        /// Layer kind fixed by the checkpoint schedule.
        expected: Qwen35LanguageLayerKind,
    },
    /// Tensor rank differed from the pinned shape.
    WrongRank {
        /// Rejected tensor name.
        tensor: String,
        /// Pinned rank.
        expected: usize,
        /// Supplied rank.
        actual: usize,
    },
    /// A tensor dimension was zero.
    ZeroDimension {
        /// Rejected tensor name.
        tensor: String,
        /// Zero-based dimension index.
        dimension: usize,
    },
    /// Tensor coefficient product overflowed `u64`.
    CoefficientOverflow(String),
    /// Tensor dimensions differed from the pinned shape.
    WrongShape(String),
    /// Caller supplied the same canonical tensor name more than once.
    DuplicateTensor(String),
    /// Fewer than the complete pinned tensor set were supplied.
    MissingTensorMetadata {
        /// Required tensor count.
        expected: usize,
        /// Supplied tensor count.
        actual: usize,
    },
    /// Canonical record digest or byte count differed from the pinned identity.
    MetadataIdentityMismatch {
        /// Required BLAKE3 metadata digest.
        expected_digest: [u8; 32],
        /// Computed BLAKE3 metadata digest.
        actual_digest: [u8; 32],
        /// Required canonical byte count.
        expected_bytes: u64,
        /// Supplied canonical byte count.
        actual_bytes: u64,
    },
    /// Checked coverage totals differed from the frozen policy.
    CoverageSummaryMismatch,
    /// Canonical coverage policy used an unsupported format version.
    UnsupportedCanonicalPolicyVersion(u8),
    /// Canonical coverage policy was truncated or structurally malformed.
    MalformedCanonicalPolicy(&'static str),
    /// Coverage policy decoded but did not use the unique canonical encoding.
    NonCanonicalPolicy,
    /// A checked aggregate exceeded `u64`.
    SummaryOverflow,
    /// A bounded caller-controlled allocation could not be reserved.
    AllocationFailed,
}

impl fmt::Display for Qwen35CoverageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongRevision => write!(
                f,
                "revision is not pinned Qwen3.6-27B revision {QWEN36_27B_COVERAGE_REVISION}"
            ),
            Self::TooManyTensors { maximum } => {
                write!(f, "tensor metadata exceeds maximum count {maximum}")
            }
            Self::EmptyTensorName => f.write_str("tensor name is empty"),
            Self::MetadataFieldTooLong { field, maximum } => {
                write!(f, "{field} exceeds maximum length {maximum}")
            }
            Self::UnsupportedDtype(tensor) => write!(
                f,
                "tensor `{tensor}` does not use exact BF16 source dtype; pre-quantized exports are not admissible"
            ),
            Self::UnknownTensor(tensor) => {
                write!(
                    f,
                    "tensor `{tensor}` is not in the pinned checkpoint contract"
                )
            }
            Self::WrongLayerKind {
                tensor,
                layer,
                expected,
            } => write!(
                f,
                "tensor `{tensor}` conflicts with layer {layer} kind {expected:?}"
            ),
            Self::WrongRank {
                tensor,
                expected,
                actual,
            } => write!(
                f,
                "tensor `{tensor}` has rank {actual}, expected rank {expected}"
            ),
            Self::ZeroDimension { tensor, dimension } => {
                write!(f, "tensor `{tensor}` has zero-sized dimension {dimension}")
            }
            Self::CoefficientOverflow(tensor) => {
                write!(f, "tensor `{tensor}` coefficient count overflows u64")
            }
            Self::WrongShape(tensor) => {
                write!(f, "tensor `{tensor}` shape differs from pinned metadata")
            }
            Self::DuplicateTensor(tensor) => {
                write!(f, "tensor `{tensor}` appears more than once")
            }
            Self::MissingTensorMetadata { expected, actual } => write!(
                f,
                "incomplete tensor metadata: expected {expected}, got {actual}"
            ),
            Self::MetadataIdentityMismatch {
                expected_digest,
                actual_digest,
                expected_bytes,
                actual_bytes,
            } => write!(
                f,
                "canonical metadata identity mismatch: expected {expected_bytes} bytes and digest \
                 {expected_digest:02x?}, got {actual_bytes} bytes and digest {actual_digest:02x?}"
            ),
            Self::CoverageSummaryMismatch => {
                f.write_str("coverage totals differ from frozen Qwen3.6-27B policy")
            }
            Self::UnsupportedCanonicalPolicyVersion(version) => {
                write!(
                    f,
                    "unsupported canonical Qwen3.6 coverage policy version {version}"
                )
            }
            Self::MalformedCanonicalPolicy(field) => {
                write!(f, "malformed canonical Qwen3.6 coverage policy {field}")
            }
            Self::NonCanonicalPolicy => {
                f.write_str("Qwen3.6 coverage policy is not canonically encoded")
            }
            Self::SummaryOverflow => f.write_str("coverage aggregate overflows u64"),
            Self::AllocationFailed => f.write_str("bounded metadata allocation failed"),
        }
    }
}

impl std::error::Error for Qwen35CoverageError {}

#[derive(Debug)]
struct DecodedPolicyEntry {
    name: String,
    shape: Vec<u64>,
    coefficients: u64,
    scope: u8,
    role: u8,
    disposition: u8,
}

struct PolicyCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> PolicyCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], Qwen35CoverageError> {
        let end =
            self.offset
                .checked_add(count)
                .ok_or(Qwen35CoverageError::MalformedCanonicalPolicy(
                    "length overflow",
                ))?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(Qwen35CoverageError::MalformedCanonicalPolicy("truncated"))?;
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, Qwen35CoverageError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, Qwen35CoverageError> {
        let mut bytes = [0_u8; 4];
        bytes.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, Qwen35CoverageError> {
        let mut bytes = [0_u8; 8];
        bytes.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(bytes))
    }

    fn string(
        &mut self,
        maximum: usize,
        field: &'static str,
    ) -> Result<&'a str, Qwen35CoverageError> {
        let length = self.u32()? as usize;
        if length == 0 || length > maximum {
            return Err(Qwen35CoverageError::MalformedCanonicalPolicy(field));
        }
        core::str::from_utf8(self.take(length)?)
            .map_err(|_| Qwen35CoverageError::MalformedCanonicalPolicy(field))
    }
}

fn write_policy_u32(output: &mut Vec<u8>, value: usize) {
    output.extend_from_slice(
        &u32::try_from(value)
            .expect("validated Qwen3.6 policy length fits u32")
            .to_le_bytes(),
    );
}

fn write_policy_bytes(output: &mut Vec<u8>, value: &[u8]) {
    write_policy_u32(output, value.len());
    output.extend_from_slice(value);
}

const fn dtype_tag(dtype: Qwen35SourceDtype) -> u8 {
    match dtype {
        Qwen35SourceDtype::Bfloat16 => 1,
    }
}

const fn scope_tag(scope: Qwen35TensorScope) -> u8 {
    match scope {
        Qwen35TensorScope::Language => 1,
        Qwen35TensorScope::MtpDrafter => 2,
        Qwen35TensorScope::DeferredVision => 3,
    }
}

const fn role_tag(role: Qwen35TensorRole) -> u8 {
    match role {
        Qwen35TensorRole::TokenEmbedding => 1,
        Qwen35TensorRole::OutputHead => 2,
        Qwen35TensorRole::Normalization => 3,
        Qwen35TensorRole::MlpProjection => 4,
        Qwen35TensorRole::FullAttentionProjection => 5,
        Qwen35TensorRole::DeltaNetProjection => 6,
        Qwen35TensorRole::DeltaNetState => 7,
        Qwen35TensorRole::DeltaNetConvolution => 8,
        Qwen35TensorRole::MtpFusionProjection => 9,
        Qwen35TensorRole::VisionAttentionProjection => 10,
        Qwen35TensorRole::VisionMlpProjection => 11,
        Qwen35TensorRole::VisionPatchEmbedding => 12,
        Qwen35TensorRole::VisionPositionalEmbedding => 13,
        Qwen35TensorRole::VisionMergerProjection => 14,
        Qwen35TensorRole::Bias => 15,
    }
}

const fn disposition_tag(disposition: Qwen35CoverageDisposition) -> u8 {
    match disposition {
        Qwen35CoverageDisposition::AdditiveTernary => 1,
        Qwen35CoverageDisposition::PreserveSource => 2,
        Qwen35CoverageDisposition::ExcludedFutureVision => 3,
    }
}

#[derive(Clone, Copy)]
struct ExpectedTensor {
    shape: &'static [u64],
    scope: Qwen35TensorScope,
    role: Qwen35TensorRole,
}

#[derive(Clone, Copy)]
enum ClassificationFailure {
    Unknown,
    WrongLayerKind {
        layer: u8,
        expected: Qwen35LanguageLayerKind,
    },
}

fn validate_entry(
    tensor: Qwen35TensorMetadata<'_>,
) -> Result<Qwen35CoverageEntry, Qwen35CoverageError> {
    if tensor.name.is_empty() {
        return Err(Qwen35CoverageError::EmptyTensorName);
    }
    let name = bounded_string(tensor.name, "tensor name", MAX_NAME_BYTES)?;
    if tensor.dtype.len() > MAX_DTYPE_BYTES {
        return Err(Qwen35CoverageError::MetadataFieldTooLong {
            field: "tensor dtype",
            maximum: MAX_DTYPE_BYTES,
        });
    }
    if tensor.dtype != "BF16" {
        return Err(Qwen35CoverageError::UnsupportedDtype(name));
    }

    let expected = match classify_tensor(tensor.name) {
        Ok(expected) => expected,
        Err(ClassificationFailure::Unknown) => {
            return Err(Qwen35CoverageError::UnknownTensor(name));
        }
        Err(ClassificationFailure::WrongLayerKind { layer, expected }) => {
            return Err(Qwen35CoverageError::WrongLayerKind {
                tensor: name,
                layer,
                expected,
            });
        }
    };

    if tensor.shape.len() != expected.shape.len() || tensor.shape.len() > MAX_RANK {
        return Err(Qwen35CoverageError::WrongRank {
            tensor: name,
            expected: expected.shape.len(),
            actual: tensor.shape.len(),
        });
    }
    if let Some(dimension) = tensor.shape.iter().position(|&size| size == 0) {
        return Err(Qwen35CoverageError::ZeroDimension {
            tensor: name,
            dimension,
        });
    }
    let coefficients = tensor
        .shape
        .iter()
        .try_fold(1_u64, |product, &size| product.checked_mul(size));
    let Some(coefficients) = coefficients else {
        return Err(Qwen35CoverageError::CoefficientOverflow(name));
    };
    if tensor.shape != expected.shape {
        return Err(Qwen35CoverageError::WrongShape(name));
    }

    let mut shape = Vec::new();
    shape
        .try_reserve_exact(tensor.shape.len())
        .map_err(|_| Qwen35CoverageError::AllocationFailed)?;
    shape.extend_from_slice(tensor.shape);
    let disposition = if expected.scope == Qwen35TensorScope::DeferredVision {
        Qwen35CoverageDisposition::ExcludedFutureVision
    } else if shape.len() == 2 {
        Qwen35CoverageDisposition::AdditiveTernary
    } else {
        Qwen35CoverageDisposition::PreserveSource
    };

    Ok(Qwen35CoverageEntry {
        name,
        dtype: Qwen35SourceDtype::Bfloat16,
        shape,
        coefficients,
        scope: expected.scope,
        role: expected.role,
        disposition,
    })
}

fn bounded_string(
    value: &str,
    field: &'static str,
    maximum: usize,
) -> Result<String, Qwen35CoverageError> {
    if value.len() > maximum {
        return Err(Qwen35CoverageError::MetadataFieldTooLong { field, maximum });
    }
    let mut output = String::new();
    output
        .try_reserve_exact(value.len())
        .map_err(|_| Qwen35CoverageError::AllocationFailed)?;
    output.push_str(value);
    Ok(output)
}

fn classify_tensor(name: &str) -> Result<ExpectedTensor, ClassificationFailure> {
    match name {
        "lm_head.weight" => return Ok(language(HEAD, Qwen35TensorRole::OutputHead)),
        "model.language_model.embed_tokens.weight" => {
            return Ok(language(HEAD, Qwen35TensorRole::TokenEmbedding));
        }
        "model.language_model.norm.weight" => {
            return Ok(language(HIDDEN, Qwen35TensorRole::Normalization));
        }
        _ => {}
    }
    if let Some(rest) = name.strip_prefix("model.language_model.layers.") {
        return classify_language_layer(rest);
    }
    if name.starts_with("mtp.") {
        return classify_mtp(name).ok_or(ClassificationFailure::Unknown);
    }
    if name.starts_with("model.visual.") {
        return classify_vision(name).ok_or(ClassificationFailure::Unknown);
    }
    Err(ClassificationFailure::Unknown)
}

fn classify_language_layer(rest: &str) -> Result<ExpectedTensor, ClassificationFailure> {
    let Some((layer_text, suffix)) = rest.split_once('.') else {
        return Err(ClassificationFailure::Unknown);
    };
    let Ok(layer) = layer_text.parse::<u8>() else {
        return Err(ClassificationFailure::Unknown);
    };
    if layer >= 64 {
        return Err(ClassificationFailure::Unknown);
    }
    let kind = if layer % 4 == 3 {
        Qwen35LanguageLayerKind::FullAttention
    } else {
        Qwen35LanguageLayerKind::DeltaNet
    };

    let common = match suffix {
        "input_layernorm.weight" | "post_attention_layernorm.weight" => {
            Some(language(HIDDEN, Qwen35TensorRole::Normalization))
        }
        "mlp.down_proj.weight" => Some(language(MLP_DOWN, Qwen35TensorRole::MlpProjection)),
        "mlp.gate_proj.weight" | "mlp.up_proj.weight" => {
            Some(language(MLP_UP, Qwen35TensorRole::MlpProjection))
        }
        _ => None,
    };
    if let Some(expected) = common {
        return Ok(expected);
    }

    match kind {
        Qwen35LanguageLayerKind::DeltaNet => {
            if suffix.starts_with("self_attn.") {
                return Err(ClassificationFailure::WrongLayerKind {
                    layer,
                    expected: kind,
                });
            }
            classify_delta(suffix).ok_or(ClassificationFailure::Unknown)
        }
        Qwen35LanguageLayerKind::FullAttention => {
            if suffix.starts_with("linear_attn.") {
                return Err(ClassificationFailure::WrongLayerKind {
                    layer,
                    expected: kind,
                });
            }
            classify_full_attention(suffix).ok_or(ClassificationFailure::Unknown)
        }
    }
}

fn classify_delta(suffix: &str) -> Option<ExpectedTensor> {
    let (shape, role) = match suffix {
        "linear_attn.A_log" | "linear_attn.dt_bias" => {
            (DELTA_CHANNELS, Qwen35TensorRole::DeltaNetState)
        }
        "linear_attn.conv1d.weight" => (DELTA_CONV, Qwen35TensorRole::DeltaNetConvolution),
        "linear_attn.in_proj_a.weight" | "linear_attn.in_proj_b.weight" => {
            (DELTA_AB, Qwen35TensorRole::DeltaNetProjection)
        }
        "linear_attn.in_proj_qkv.weight" => (DELTA_QKV, Qwen35TensorRole::DeltaNetProjection),
        "linear_attn.in_proj_z.weight" => (DELTA_Z, Qwen35TensorRole::DeltaNetProjection),
        "linear_attn.norm.weight" => (DELTA_NORM, Qwen35TensorRole::Normalization),
        "linear_attn.out_proj.weight" => (DELTA_OUT, Qwen35TensorRole::DeltaNetProjection),
        _ => return None,
    };
    Some(language(shape, role))
}

fn classify_full_attention(suffix: &str) -> Option<ExpectedTensor> {
    let (shape, role) = match suffix {
        "self_attn.k_norm.weight" | "self_attn.q_norm.weight" => {
            (FULL_K_NORM, Qwen35TensorRole::Normalization)
        }
        "self_attn.k_proj.weight" | "self_attn.v_proj.weight" => {
            (FULL_KV, Qwen35TensorRole::FullAttentionProjection)
        }
        "self_attn.o_proj.weight" => (FULL_O, Qwen35TensorRole::FullAttentionProjection),
        "self_attn.q_proj.weight" => (FULL_Q, Qwen35TensorRole::FullAttentionProjection),
        _ => return None,
    };
    Some(language(shape, role))
}

fn classify_mtp(name: &str) -> Option<ExpectedTensor> {
    let (shape, role) = match name {
        "mtp.fc.weight" => (MTP_FC, Qwen35TensorRole::MtpFusionProjection),
        "mtp.layers.0.input_layernorm.weight"
        | "mtp.layers.0.post_attention_layernorm.weight"
        | "mtp.norm.weight"
        | "mtp.pre_fc_norm_embedding.weight"
        | "mtp.pre_fc_norm_hidden.weight" => (HIDDEN, Qwen35TensorRole::Normalization),
        "mtp.layers.0.mlp.down_proj.weight" => (MLP_DOWN, Qwen35TensorRole::MlpProjection),
        "mtp.layers.0.mlp.gate_proj.weight" | "mtp.layers.0.mlp.up_proj.weight" => {
            (MLP_UP, Qwen35TensorRole::MlpProjection)
        }
        "mtp.layers.0.self_attn.k_norm.weight" | "mtp.layers.0.self_attn.q_norm.weight" => {
            (FULL_K_NORM, Qwen35TensorRole::Normalization)
        }
        "mtp.layers.0.self_attn.k_proj.weight" | "mtp.layers.0.self_attn.v_proj.weight" => {
            (FULL_KV, Qwen35TensorRole::FullAttentionProjection)
        }
        "mtp.layers.0.self_attn.o_proj.weight" => {
            (FULL_O, Qwen35TensorRole::FullAttentionProjection)
        }
        "mtp.layers.0.self_attn.q_proj.weight" => {
            (FULL_Q, Qwen35TensorRole::FullAttentionProjection)
        }
        _ => return None,
    };
    Some(ExpectedTensor {
        shape,
        scope: Qwen35TensorScope::MtpDrafter,
        role,
    })
}

fn classify_vision(name: &str) -> Option<ExpectedTensor> {
    match name {
        "model.visual.patch_embed.proj.weight" => {
            return Some(vision(VISION_PATCH, Qwen35TensorRole::VisionPatchEmbedding));
        }
        "model.visual.patch_embed.proj.bias" => {
            return Some(vision(VISION_HIDDEN, Qwen35TensorRole::Bias));
        }
        "model.visual.pos_embed.weight" => {
            return Some(vision(
                VISION_POSITION,
                Qwen35TensorRole::VisionPositionalEmbedding,
            ));
        }
        "model.visual.merger.linear_fc1.bias" => {
            return Some(vision(VISION_MERGER_HIDDEN, Qwen35TensorRole::Bias));
        }
        "model.visual.merger.linear_fc1.weight" => {
            return Some(vision(
                VISION_MERGER_UP,
                Qwen35TensorRole::VisionMergerProjection,
            ));
        }
        "model.visual.merger.linear_fc2.bias" => {
            return Some(vision(HIDDEN, Qwen35TensorRole::Bias));
        }
        "model.visual.merger.linear_fc2.weight" => {
            return Some(vision(
                VISION_MERGER_DOWN,
                Qwen35TensorRole::VisionMergerProjection,
            ));
        }
        "model.visual.merger.norm.bias" | "model.visual.merger.norm.weight" => {
            return Some(vision(VISION_HIDDEN, Qwen35TensorRole::Normalization));
        }
        _ => {}
    }

    let rest = name.strip_prefix("model.visual.blocks.")?;
    let (block_text, suffix) = rest.split_once('.')?;
    let block = block_text.parse::<u8>().ok()?;
    if block >= 27 {
        return None;
    }
    let (shape, role) = match suffix {
        "attn.proj.bias" => (VISION_HIDDEN, Qwen35TensorRole::Bias),
        "attn.proj.weight" => (VISION_PROJ, Qwen35TensorRole::VisionAttentionProjection),
        "attn.qkv.bias" => (VISION_QKV_BIAS, Qwen35TensorRole::Bias),
        "attn.qkv.weight" => (VISION_QKV, Qwen35TensorRole::VisionAttentionProjection),
        "mlp.linear_fc1.bias" => (VISION_MLP_UP_BIAS, Qwen35TensorRole::Bias),
        "mlp.linear_fc1.weight" => (VISION_MLP_UP, Qwen35TensorRole::VisionMlpProjection),
        "mlp.linear_fc2.bias" => (VISION_HIDDEN, Qwen35TensorRole::Bias),
        "mlp.linear_fc2.weight" => (VISION_MLP_DOWN, Qwen35TensorRole::VisionMlpProjection),
        "norm1.bias" | "norm1.weight" | "norm2.bias" | "norm2.weight" => {
            (VISION_HIDDEN, Qwen35TensorRole::Normalization)
        }
        _ => return None,
    };
    Some(vision(shape, role))
}

const fn language(shape: &'static [u64], role: Qwen35TensorRole) -> ExpectedTensor {
    ExpectedTensor {
        shape,
        scope: Qwen35TensorScope::Language,
        role,
    }
}

const fn vision(shape: &'static [u64], role: Qwen35TensorRole) -> ExpectedTensor {
    ExpectedTensor {
        shape,
        scope: Qwen35TensorScope::DeferredVision,
        role,
    }
}

fn hash_metadata(entries: &[Qwen35CoverageEntry]) -> Result<([u8; 32], u64), Qwen35CoverageError> {
    let mut hasher = blake3::Hasher::new();
    let mut bytes = 0_u64;
    for entry in entries {
        hash_part(&mut hasher, &mut bytes, entry.name.as_bytes())?;
        hash_part(&mut hasher, &mut bytes, b"\t")?;
        hash_part(
            &mut hasher,
            &mut bytes,
            entry.dtype.canonical_name().as_bytes(),
        )?;
        hash_part(&mut hasher, &mut bytes, b"\t")?;
        for (index, &dimension) in entry.shape.iter().enumerate() {
            if index != 0 {
                hash_part(&mut hasher, &mut bytes, b",")?;
            }
            let mut buffer = [0_u8; 20];
            let digits = decimal_bytes(dimension, &mut buffer);
            hash_part(&mut hasher, &mut bytes, digits)?;
        }
        hash_part(&mut hasher, &mut bytes, b"\n")?;
    }
    Ok((*hasher.finalize().as_bytes(), bytes))
}

fn hash_part(
    hasher: &mut blake3::Hasher,
    total: &mut u64,
    bytes: &[u8],
) -> Result<(), Qwen35CoverageError> {
    let length = u64::try_from(bytes.len()).map_err(|_| Qwen35CoverageError::SummaryOverflow)?;
    *total = total
        .checked_add(length)
        .ok_or(Qwen35CoverageError::SummaryOverflow)?;
    hasher.update(bytes);
    Ok(())
}

fn decimal_bytes(mut value: u64, buffer: &mut [u8; 20]) -> &[u8] {
    let mut cursor = buffer.len();
    loop {
        cursor -= 1;
        buffer[cursor] = b'0' + u8::try_from(value % 10).expect("decimal digit fits u8");
        value /= 10;
        if value == 0 {
            return &buffer[cursor..];
        }
    }
}

fn summary_is_expected(summary: Qwen35CoverageSummary) -> bool {
    summary.total == totals(1_199, 27_781_427_952)
        && summary.language == totals(851, 26_895_998_464)
        && summary.mtp == totals(15, 424_699_392)
        && summary.vision == totals(333, 460_730_096)
        && summary.included == totals(866, 27_320_697_856)
        && summary.additive_ternary == totals(506, 27_318_026_240)
        && summary.preserve_source == totals(360, 2_671_616)
        && summary.excluded_future_vision == totals(333, 460_730_096)
}

const fn totals(tensors: u64, coefficients: u64) -> Qwen35CoverageTotals {
    Qwen35CoverageTotals {
        tensors,
        coefficients,
    }
}
