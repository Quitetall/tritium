//! Parent-bound CompactV1/NearLosslessV1 allocation selection.

// The admission/execution stack builds on unix-only staging primitives
// (create_temporary_file, PackedMapCursor et al.); gate the module with them.
#[cfg(unix)]
mod package_admission;

#[cfg(unix)]
pub use package_admission::{
    Qwen36AdmittedExecutionReceipt, Qwen36AdmittedExecutionSession, Qwen36ExecutionBackend,
    Qwen36ExecutionReplayError, Qwen36ExecutionSessionOpenError, Qwen36ExecutionVisitError,
    Qwen36FinalLogitsOutputBindingError, Qwen36FinalLogitsOutputBindingReceipt,
    Qwen36PackageAdmissionError, Qwen36PackageAdmissionReceipt, Qwen36PackageAdmittedCampaignStore,
    Qwen36PackageProfileReceipt, Qwen36PackageRuntimeLedger, Qwen36PackageScaleOnlyCampaignStore,
    Qwen36PackageVisitError, Qwen36PvParentContext,
};
#[cfg(all(unix, feature = "cuda"))]
pub use package_admission::{
    Qwen36PvPackageAdmissionError, Qwen36PvPackageAdmissionReceipt,
    Qwen36PvPackageAdmittedCampaignStore, Qwen36PvPackageVisitError,
};

use std::{
    convert::Infallible,
    error::Error,
    fmt,
    fs::{self, File},
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use tritium_format::{
    ModelId,
    salt_v2::SaltV2Codec,
    salt_v2_master::{SaltV2FitConstraint, SaltV2MasterTensorDecoder, SaltV2MasterVisitError},
    salt_v2_package::{
        SaltV2StreamTensorSpec, SaltV2Transform, SaltV2UniformPhysicalBytes,
        SaltV2UniformRateError, SaltV2UniformRateModel,
    },
};
use tritium_quantize::{
    ByteDelta, NestedProfileBudgets, PackedPlaneCounts, PackedUniformProfilePlanner,
    PhysicalAllocError, PhysicalBytes, ProfileBudget, SaltV2Profile, UniformPrefixCurve,
    UniformProfileAllocError,
};

use crate::{
    ContentId, TensorPutError, TensorRecordReceipt, TensorRecordSpec, TensorVisitError,
    TensorWorkStore,
    tensor_work_store::{create_temporary_file, ensure_durable_directory},
};

use super::super::Qwen36SourceIdentityStatus;
use super::{
    CHECKSUM_BYTES, FixedCampaignMode, PinnedDirectory, Qwen36AdditiveCampaignStore,
    Qwen36AdditiveMasterReceipt, Qwen36CompleteWorkspaceReceipt, Qwen36TensorWorkError,
    SaltV2MasterTensorSpec, identity_status_from_tag, identity_status_tag, is_zero, persist_exact,
    read_regular_bounded, validate_directories, work_io,
};

const SELECTION_DIRECTORY: &str = "selected-allocation";
const SELECTION_FILE: &str = "selected-allocation.tq36a";
const SELECTION_MAGIC: [u8; 8] = *b"TSQ36AL\0";
const MAP_METADATA_MAGIC: [u8; 8] = *b"TSQ36MP\0";
const FORMAT_VERSION: u16 = 1;
const SELECTION_CHECKSUM_CONTEXT: &str = "tritium qwen3.6 selected allocation checksum v1";
const ALLOCATION_MAP_CONTEXT: &str = "tritium qwen3.6 selected allocation map v1";
const NESTED_ALLOCATION_CONTEXT: &str = "tritium qwen3.6 nested allocation pair v1";
const SELECTED_LOSS_CONTEXT: &str = "tritium qwen3.6 selected prefix loss v1";
const MAP_SCHEMA: &[u8] = b"tritium qwen3.6 packed selected allocation map v1";
const MAX_SELECTION_BYTES: u64 = 512 * 1024;
const MAP_WRITE_BUFFER_BYTES: usize = 64 * 1024;
const MAP_WRITE_BUFFER_BYTES_U64: u64 = 64 * 1024;

/// Immutable policy selecting one codec and two exact nested profile budgets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen36SelectedAllocationSpec {
    codec: SaltV2Codec,
    allocator_id: [u8; 32],
    recipe_id: [u8; 32],
    budgets: NestedProfileBudgets,
    spec_id: ContentId,
}

impl Qwen36SelectedAllocationSpec {
    /// Construct a canonical nested-allocation policy.
    ///
    /// Both profiles share one codec and one fixed metadata cost curve. Compact
    /// ceilings must be componentwise no larger than NearLossless ceilings.
    ///
    /// # Errors
    /// Rejects zero provenance, an unsupported codec, contradictory metadata,
    /// zero ceilings, or non-nested serialized/resident ceilings.
    pub fn new(
        codec: SaltV2Codec,
        allocator_id: [u8; 32],
        recipe_id: [u8; 32],
        budgets: NestedProfileBudgets,
    ) -> Result<Self, Qwen36TensorWorkError> {
        codec_tag(codec)?;
        if is_zero(&allocator_id) || is_zero(&recipe_id) {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "selected allocation provenance",
            ));
        }
        if budgets.compact.metadata != budgets.near_lossless.metadata {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "selected allocation metadata costs",
            ));
        }
        if budgets.compact.maximum.serialized == 0
            || budgets.compact.maximum.resident == 0
            || budgets.near_lossless.maximum.serialized == 0
            || budgets.near_lossless.maximum.resident == 0
            || !bytes_componentwise_le(budgets.compact.maximum, budgets.near_lossless.maximum)
        {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "selected allocation profile ceilings",
            ));
        }
        let mut spec = Self {
            codec,
            allocator_id,
            recipe_id,
            budgets,
            spec_id: ContentId::from_digest([0; 32]),
        };
        spec.spec_id = ContentId::of_bytes(&spec.canonical_bytes()?);
        Ok(spec)
    }

    /// Construct exact full-tile package budgets from two physical ceilings.
    ///
    /// Canonical file headers/maps and indexed-runtime maps/rank prefixes are
    /// derived from the complete ordered master catalog. Both minimum one-plane
    /// packages must fit, and every tensor must be full-tile and representable by
    /// the indexed runtime.
    ///
    /// # Errors
    /// Returns [`Qwen36PhysicalAllocationError`] for incompatible codec/fit
    /// constraints, invalid provenance or ceilings, ragged/unindexable geometry,
    /// or a ceiling below the mandatory prefix.
    #[allow(clippy::too_many_arguments)]
    pub fn for_uniform_full_tiles(
        codec: SaltV2Codec,
        allocator_id: [u8; 32],
        recipe_id: [u8; 32],
        masters: &[SaltV2MasterTensorSpec],
        compact_maximum: PhysicalBytes,
        near_lossless_maximum: PhysicalBytes,
    ) -> Result<Self, Qwen36PhysicalAllocationError> {
        for master in masters {
            validate_codec_compatibility(codec, master.geometry().constraint)?;
        }
        let rate = uniform_rate_model(codec, masters)?;
        let metadata = ByteDelta::declared(PhysicalBytes {
            serialized: rate.fixed_serialized_bytes(),
            resident: rate.fixed_resident_bytes(),
        });
        let spec = Self::new(
            codec,
            allocator_id,
            recipe_id,
            NestedProfileBudgets {
                compact: ProfileBudget {
                    maximum: compact_maximum,
                    metadata,
                },
                near_lossless: ProfileBudget {
                    maximum: near_lossless_maximum,
                    metadata,
                },
            },
        )?;
        let _ = maximum_present_planes(rate, compact_maximum, SaltV2Profile::CompactV1)?;
        let _ = maximum_present_planes(rate, near_lossless_maximum, SaltV2Profile::NearLosslessV1)?;
        Ok(spec)
    }

    /// Physical codec shared by both selected profiles.
    #[must_use]
    pub const fn codec(&self) -> SaltV2Codec {
        self.codec
    }

    /// Identity of the allocator implementation/tie policy that produced counts.
    #[must_use]
    pub const fn allocator_id(&self) -> &[u8; 32] {
        &self.allocator_id
    }

    /// Identity of the exact allocation recipe and candidate construction.
    #[must_use]
    pub const fn recipe_id(&self) -> &[u8; 32] {
        &self.recipe_id
    }

    /// Exact serialized and resident budgets for both nested profiles.
    #[must_use]
    pub const fn budgets(&self) -> NestedProfileBudgets {
        self.budgets
    }

    /// Content identity of this complete canonical selection policy.
    #[must_use]
    pub const fn spec_id(&self) -> ContentId {
        self.spec_id
    }

    fn canonical_bytes(&self) -> Result<Vec<u8>, Qwen36TensorWorkError> {
        let mut output = Vec::new();
        output.push(codec_tag(self.codec)?);
        output.extend_from_slice(&[0; 7]);
        output.extend_from_slice(&self.allocator_id);
        output.extend_from_slice(&self.recipe_id);
        encode_profile_budget(&mut output, self.budgets.compact);
        encode_profile_budget(&mut output, self.budgets.near_lossless);
        Ok(output)
    }
}

/// One selected profile's immutable packed-map and parent-derived identities.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen36SelectedProfileReceipt {
    allocation_map_id: ContentId,
    map_record: TensorRecordReceipt,
    selected_planes: u64,
    selected_loss_id: ContentId,
}

impl Qwen36SelectedProfileReceipt {
    /// Semantic identity of canonical tensor boundaries and every selected count.
    #[must_use]
    pub const fn allocation_map_id(&self) -> ContentId {
        self.allocation_map_id
    }

    /// CAS record containing the canonical two-bit-per-tile map payload.
    #[must_use]
    pub const fn map_record(&self) -> &TensorRecordReceipt {
        &self.map_record
    }

    /// Sum of selected additive planes across every allocation tile.
    #[must_use]
    pub const fn selected_planes(&self) -> u64 {
        self.selected_planes
    }

    /// Identity of every selected parent prefix-loss point in canonical order.
    #[must_use]
    pub const fn selected_loss_id(&self) -> ContentId {
        self.selected_loss_id
    }
}

/// Durable structural proof of one nested CompactV1/NearLosslessV1 selection.
///
/// This receipt binds exact parent masters and selected prefix counts. It is not an
/// optimality, quality, package-admission, or official-source publication proof.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen36SelectedAllocationReceipt {
    selection_id: ContentId,
    parent_completion_id: ContentId,
    base_workspace_id: ContentId,
    campaign_id: ContentId,
    master_set_id: [u8; 32],
    source_model_id: ModelId,
    identity_status: Qwen36SourceIdentityStatus,
    spec: Qwen36SelectedAllocationSpec,
    tensor_count: u64,
    tile_count: u64,
    nested_allocation_id: ContentId,
    compact: Qwen36SelectedProfileReceipt,
    near_lossless: Qwen36SelectedProfileReceipt,
}

/// Failure while binding a caller-provided lazy allocation stream.
#[derive(Debug)]
pub enum Qwen36SelectedAllocationBindError<E> {
    /// Parent, allocation, filesystem, or durable publication validation failed.
    Campaign(Qwen36TensorWorkError),
    /// The caller's allocation source failed before publication.
    Source(E),
}

impl<E: fmt::Display> fmt::Display for Qwen36SelectedAllocationBindError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Campaign(error) => write!(formatter, "selected allocation bind failed: {error}"),
            Self::Source(error) => write!(formatter, "selected allocation source failed: {error}"),
        }
    }
}

impl<E: Error + 'static> Error for Qwen36SelectedAllocationBindError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Campaign(error) => Some(error),
            Self::Source(error) => Some(error),
        }
    }
}

impl<E> From<Qwen36TensorWorkError> for Qwen36SelectedAllocationBindError<E> {
    fn from(error: Qwen36TensorWorkError) -> Self {
        Self::Campaign(error)
    }
}

/// Failure while deriving and durably binding an exact package-byte allocation.
#[derive(Debug)]
pub enum Qwen36PhysicalAllocationError {
    /// Parent campaign, master, filesystem, or durable publication failed.
    Campaign(Qwen36TensorWorkError),
    /// Canonical package geometry cannot use the uniform full-tile rate model.
    Rate(SaltV2UniformRateError),
    /// Exact compact allocation failed.
    Allocation(UniformProfileAllocError<Infallible>),
    /// The policy's declared metadata differs from canonical package/runtime metadata.
    MetadataMismatch {
        /// Metadata derived from the canonical package and indexed runtime.
        modeled: PhysicalBytes,
        /// Metadata carried by the allocation policy.
        specified: PhysicalBytes,
    },
    /// Even one mandatory plane per tile exceeds one profile's exact ceilings.
    BudgetTooSmall {
        /// Profile whose minimum package cannot fit.
        profile: SaltV2Profile,
        /// Exact mandatory physical bytes.
        required: PhysicalBytes,
        /// Policy ceilings.
        maximum: PhysicalBytes,
    },
}

impl fmt::Display for Qwen36PhysicalAllocationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Campaign(error) => {
                write!(formatter, "physical allocation campaign failed: {error}")
            }
            Self::Rate(error) => {
                write!(formatter, "physical allocation rate model failed: {error}")
            }
            Self::Allocation(error) => {
                write!(formatter, "physical allocation solver failed: {error}")
            }
            Self::MetadataMismatch { modeled, specified } => write!(
                formatter,
                "allocation metadata {specified:?} differs from canonical package/runtime metadata {modeled:?}"
            ),
            Self::BudgetTooSmall {
                profile,
                required,
                maximum,
            } => write!(
                formatter,
                "{profile:?} requires {required:?}, exceeding exact ceilings {maximum:?}"
            ),
        }
    }
}

impl Error for Qwen36PhysicalAllocationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Campaign(error) => Some(error),
            Self::Rate(error) => Some(error),
            Self::Allocation(error) => Some(error),
            Self::MetadataMismatch { .. } | Self::BudgetTooSmall { .. } => None,
        }
    }
}

impl From<Qwen36TensorWorkError> for Qwen36PhysicalAllocationError {
    fn from(error: Qwen36TensorWorkError) -> Self {
        Self::Campaign(error)
    }
}

impl From<SaltV2UniformRateError> for Qwen36PhysicalAllocationError {
    fn from(error: SaltV2UniformRateError) -> Self {
        Self::Rate(error)
    }
}

impl From<UniformProfileAllocError<Infallible>> for Qwen36PhysicalAllocationError {
    fn from(error: UniformProfileAllocError<Infallible>) -> Self {
        Self::Allocation(error)
    }
}

impl From<PhysicalAllocError> for Qwen36PhysicalAllocationError {
    fn from(error: PhysicalAllocError) -> Self {
        Self::Allocation(UniformProfileAllocError::Allocation(error))
    }
}

impl Qwen36SelectedAllocationReceipt {
    /// Content identity of the complete canonical nested selection receipt.
    #[must_use]
    pub const fn selection_id(&self) -> ContentId {
        self.selection_id
    }

    /// Exact sealed PTQ completion over which allocation was selected.
    #[must_use]
    pub const fn parent_completion_id(&self) -> ContentId {
        self.parent_completion_id
    }

    /// Exact preserved-source workspace inherited from the parent.
    #[must_use]
    pub const fn base_workspace_id(&self) -> ContentId {
        self.base_workspace_id
    }

    /// Exact PTQ campaign whose ordered masters are being allocated.
    #[must_use]
    pub const fn campaign_id(&self) -> ContentId {
        self.campaign_id
    }

    /// Ordered portable tensor-master aggregate bound by both maps.
    #[must_use]
    pub const fn master_set_id(&self) -> [u8; 32] {
        self.master_set_id
    }

    /// Exact canonical allocation policy and package-admission ceilings.
    #[must_use]
    pub const fn spec(&self) -> &Qwen36SelectedAllocationSpec {
        &self.spec
    }

    /// Number of canonically ordered additive tensors.
    #[must_use]
    pub const fn tensor_count(&self) -> u64 {
        self.tensor_count
    }

    /// Number of selected 256-coefficient allocation tiles per profile.
    #[must_use]
    pub const fn tile_count(&self) -> u64 {
        self.tile_count
    }

    /// Identity of the validated per-tile Compact/Near nesting relation.
    #[must_use]
    pub const fn nested_allocation_id(&self) -> ContentId {
        self.nested_allocation_id
    }

    /// CompactV1 selected map and parent-derived loss identity.
    #[must_use]
    pub const fn compact(&self) -> &Qwen36SelectedProfileReceipt {
        &self.compact
    }

    /// NearLosslessV1 selected map and parent-derived loss identity.
    #[must_use]
    pub const fn near_lossless(&self) -> &Qwen36SelectedProfileReceipt {
        &self.near_lossless
    }

    fn canonical_bytes(&self) -> Result<Vec<u8>, Qwen36TensorWorkError> {
        let mut output = Vec::new();
        output.extend_from_slice(&SELECTION_MAGIC);
        output.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        output.extend_from_slice(&0_u16.to_le_bytes());
        output.extend_from_slice(self.parent_completion_id.as_bytes());
        output.extend_from_slice(self.base_workspace_id.as_bytes());
        output.extend_from_slice(self.campaign_id.as_bytes());
        output.extend_from_slice(&self.master_set_id);
        output.extend_from_slice(self.source_model_id.as_bytes());
        output.push(identity_status_tag(self.identity_status));
        output.extend_from_slice(&[0; 7]);
        let spec = self.spec.canonical_bytes()?;
        let spec_len = u32::try_from(spec.len())
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("allocation spec"))?;
        output.extend_from_slice(&spec_len.to_le_bytes());
        output.extend_from_slice(&spec);
        output.extend_from_slice(&self.tensor_count.to_le_bytes());
        output.extend_from_slice(&self.tile_count.to_le_bytes());
        output.extend_from_slice(self.nested_allocation_id.as_bytes());
        encode_profile_receipt(&mut output, &self.compact)?;
        encode_profile_receipt(&mut output, &self.near_lossless)?;
        let mut hasher = blake3::Hasher::new_derive_key(SELECTION_CHECKSUM_CONTEXT);
        hasher.update(&output);
        output.extend_from_slice(hasher.finalize().as_bytes());
        Ok(output)
    }

    fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, Qwen36TensorWorkError> {
        if u64::try_from(bytes.len())
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("selected allocation manifest"))?
            > MAX_SELECTION_BYTES
            || bytes.len() < SELECTION_MAGIC.len() + CHECKSUM_BYTES
        {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "selected allocation manifest length",
            ));
        }
        let checksum_offset = bytes.len() - CHECKSUM_BYTES;
        let (payload, checksum) = bytes.split_at(checksum_offset);
        let mut checksum_hasher = blake3::Hasher::new_derive_key(SELECTION_CHECKSUM_CONTEXT);
        checksum_hasher.update(payload);
        if checksum_hasher.finalize().as_bytes() != checksum {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "selected allocation checksum",
            ));
        }
        let mut cursor = super::CanonicalCursor::new(payload);
        if cursor.take(SELECTION_MAGIC.len())? != SELECTION_MAGIC
            || cursor.u16()? != FORMAT_VERSION
            || cursor.u16()? != 0
        {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "selected allocation header",
            ));
        }
        let parent_completion_id = ContentId::from_digest(cursor.digest()?);
        let base_workspace_id = ContentId::from_digest(cursor.digest()?);
        let campaign_id = ContentId::from_digest(cursor.digest()?);
        let master_set_id = cursor.digest()?;
        let source_model_id = ModelId::from_digest(cursor.digest()?);
        let identity_status = identity_status_from_tag(cursor.u8()?)?;
        if cursor.take(7)? != [0; 7] {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "selected allocation reserved bytes",
            ));
        }
        let spec_len = usize::try_from(cursor.u32()?)
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("allocation spec"))?;
        let spec = decode_allocation_spec(cursor.take(spec_len)?)?;
        let tensor_count = cursor.u64()?;
        let tile_count = cursor.u64()?;
        let nested_allocation_id = ContentId::from_digest(cursor.digest()?);
        let compact = decode_profile_receipt(&mut cursor)?;
        let near_lossless = decode_profile_receipt(&mut cursor)?;
        if cursor.remaining() != 0 {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "selected allocation trailing bytes",
            ));
        }
        let receipt = Self {
            selection_id: ContentId::of_bytes(bytes),
            parent_completion_id,
            base_workspace_id,
            campaign_id,
            master_set_id,
            source_model_id,
            identity_status,
            spec,
            tensor_count,
            tile_count,
            nested_allocation_id,
            compact,
            near_lossless,
        };
        if receipt.canonical_bytes()? != bytes {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "noncanonical selected allocation",
            ));
        }
        Ok(receipt)
    }
}

/// Typed capability proving that one sealed PTQ parent has a durable selection.
#[derive(Debug)]
pub struct Qwen36AllocatedCampaignStore<'parent, 'store, 'source> {
    parent: &'parent Qwen36AdditiveCampaignStore<'store, 'source>,
    receipt: Qwen36SelectedAllocationReceipt,
    parent_completion: Qwen36CompleteWorkspaceReceipt,
}

fn physical_bytes(bytes: SaltV2UniformPhysicalBytes) -> PhysicalBytes {
    PhysicalBytes {
        serialized: bytes.serialized,
        resident: bytes.resident,
    }
}

fn uniform_rate_model(
    codec: SaltV2Codec,
    masters: &[SaltV2MasterTensorSpec],
) -> Result<SaltV2UniformRateModel, Qwen36PhysicalAllocationError> {
    let mut specs = Vec::new();
    specs
        .try_reserve_exact(masters.len())
        .map_err(|_| Qwen36TensorWorkError::AllocationFailed)?;
    for master in masters {
        specs.push(
            SaltV2StreamTensorSpec::new(
                master.name(),
                master.shape().to_vec(),
                SaltV2Transform::None,
            )
            .map_err(SaltV2UniformRateError::from)?,
        );
    }
    SaltV2UniformRateModel::new(codec, &specs).map_err(Into::into)
}

fn maximum_present_planes(
    rate: SaltV2UniformRateModel,
    maximum: PhysicalBytes,
    profile: SaltV2Profile,
) -> Result<u64, Qwen36PhysicalAllocationError> {
    rate.maximum_present_planes(SaltV2UniformPhysicalBytes {
        serialized: maximum.serialized,
        resident: maximum.resident,
    })
    .ok_or_else(|| {
        let required = rate
            .physical_bytes(rate.tile_count())
            .expect("uniform rate model always represents its mandatory prefix");
        Qwen36PhysicalAllocationError::BudgetTooSmall {
            profile,
            required: physical_bytes(required),
            maximum,
        }
    })
}

impl Qwen36AllocatedCampaignStore<'_, '_, '_> {
    /// Durable parent-bound nested-selection receipt carried by this capability.
    #[must_use]
    pub const fn receipt(&self) -> &Qwen36SelectedAllocationReceipt {
        &self.receipt
    }
}

impl<'store, 'source> Qwen36AdditiveCampaignStore<'store, 'source> {
    /// Reopen an existing durable selection, or derive it once when absent.
    ///
    /// An existing selection is never silently replaced: its exact policy must
    /// match `spec`, and every parent/map record is revalidated by
    /// [`Self::reopen_selected_allocation`]. This keeps resumable package
    /// reconciliation from repeating the expensive physical-allocation scans.
    #[cfg(unix)]
    pub fn reopen_or_allocate_selected_allocation<'parent>(
        &'parent self,
        spec: Qwen36SelectedAllocationSpec,
    ) -> Result<Qwen36AllocatedCampaignStore<'parent, 'store, 'source>, Qwen36PhysicalAllocationError>
    {
        let selection_path = self.root.join(SELECTION_DIRECTORY).join(SELECTION_FILE);
        match fs::symlink_metadata(&selection_path) {
            Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
                Err(Qwen36PhysicalAllocationError::Campaign(
                    Qwen36TensorWorkError::InvalidPath("selected allocation manifest"),
                ))
            }
            Ok(_) => {
                let allocated = self.reopen_selected_allocation()?;
                if allocated.receipt().spec() != &spec {
                    return Err(Qwen36PhysicalAllocationError::Campaign(
                        Qwen36TensorWorkError::WorkspaceMismatch("selected allocation policy"),
                    ));
                }
                Ok(allocated)
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                self.allocate_selected_allocation(spec)
            }
            Err(error) => Err(Qwen36PhysicalAllocationError::Campaign(work_io(
                "inspect selected allocation manifest",
                error,
            ))),
        }
    }

    /// Derive and durably bind the exact Compact/Near allocation from verified masters.
    ///
    /// The canonical package rate model converts both serialized and indexed-runtime
    /// ceilings into exact plane cardinalities. Parent masters are verified and decoded
    /// once per profile directly into the callback-driven compact solver; only ranked
    /// upgrade records and two-bit maps scale with tile count. Hessian prefix loss is the
    /// optimization objective. Counts beyond a tile's admitted prefix receive zero gain
    /// and therefore cannot be selected.
    ///
    /// # Errors
    /// Fails closed for changed parent state, ragged or unindexable package geometry,
    /// contradictory metadata, undersized budgets, allocation failure, or publication
    /// failure.
    #[cfg(unix)]
    pub fn allocate_selected_allocation<'parent>(
        &'parent self,
        spec: Qwen36SelectedAllocationSpec,
    ) -> Result<Qwen36AllocatedCampaignStore<'parent, 'store, 'source>, Qwen36PhysicalAllocationError>
    {
        let (_, parent_manifest, _) = self.require_complete_verified(FixedCampaignMode::Capture)?;
        let rate = uniform_rate_model(spec.codec(), &self.spec.expected_masters)?;
        let modeled_metadata = PhysicalBytes {
            serialized: rate.fixed_serialized_bytes(),
            resident: rate.fixed_resident_bytes(),
        };
        let budgets = spec.budgets();
        let specified_metadata = budgets.compact.metadata.effective();
        if specified_metadata != modeled_metadata {
            return Err(Qwen36PhysicalAllocationError::MetadataMismatch {
                modeled: modeled_metadata,
                specified: specified_metadata,
            });
        }

        let compact_maximum =
            maximum_present_planes(rate, budgets.compact.maximum, SaltV2Profile::CompactV1)?;
        let compact_floor =
            PackedPlaneCounts::filled(rate.tile_count(), 1, SaltV2Profile::CompactV1)?;
        let mut compact_planner = PackedUniformProfilePlanner::new(
            rate.tile_count(),
            &compact_floor,
            compact_maximum - rate.tile_count(),
            SaltV2Profile::CompactV1,
        )?;
        self.push_verified_hessian_curves(&parent_manifest.masters, &mut compact_planner)?;
        let compact = compact_planner.finish()?;

        let near_maximum = maximum_present_planes(
            rate,
            budgets.near_lossless.maximum,
            SaltV2Profile::NearLosslessV1,
        )?;
        if near_maximum < compact.present_planes {
            let required = rate
                .physical_bytes(compact.present_planes)
                .expect("compact allocation is rate-model representable");
            return Err(Qwen36PhysicalAllocationError::BudgetTooSmall {
                profile: SaltV2Profile::NearLosslessV1,
                required: physical_bytes(required),
                maximum: budgets.near_lossless.maximum,
            });
        }
        let mut near_planner = PackedUniformProfilePlanner::new(
            rate.tile_count(),
            &compact.plane_counts,
            near_maximum - compact.present_planes,
            SaltV2Profile::NearLosslessV1,
        )?;
        self.push_verified_hessian_curves(&parent_manifest.masters, &mut near_planner)?;
        let near = near_planner.finish()?;
        let tile_count = rate.tile_count();
        let counts = (0..tile_count).map(|tile| {
            Ok::<_, Infallible>((
                compact
                    .plane_counts
                    .get(tile)
                    .expect("compact map covers rate-model tiles"),
                near.plane_counts
                    .get(tile)
                    .expect("near map covers rate-model tiles"),
            ))
        });
        self.bind_selected_allocation(spec, counts)
            .map_err(|error| match error {
                Qwen36SelectedAllocationBindError::Campaign(error) => {
                    Qwen36PhysicalAllocationError::Campaign(error)
                }
                Qwen36SelectedAllocationBindError::Source(source) => match source {},
            })
    }

    /// Reject physical allocation where stable file identity is unavailable.
    #[cfg(not(unix))]
    pub fn allocate_selected_allocation<'parent>(
        &'parent self,
        _spec: Qwen36SelectedAllocationSpec,
    ) -> Result<Qwen36AllocatedCampaignStore<'parent, 'store, 'source>, Qwen36PhysicalAllocationError>
    {
        Err(Qwen36TensorWorkError::AdditiveCampaignUnsupportedPlatform.into())
    }

    #[cfg(unix)]
    fn push_verified_hessian_curves(
        &self,
        parent_masters: &[Qwen36AdditiveMasterReceipt],
        planner: &mut PackedUniformProfilePlanner<'_>,
    ) -> Result<(), Qwen36PhysicalAllocationError> {
        if parent_masters.len() != self.spec.expected_masters.len() {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "physical allocation parent tensor count",
            )
            .into());
        }
        for (expected, parent_master) in self.spec.expected_masters.iter().zip(parent_masters) {
            let mut decoder =
                SaltV2MasterTensorDecoder::new(expected).map_err(Qwen36TensorWorkError::Master)?;
            let mut local_tiles = 0usize;
            self.objects
                .try_visit_verified(
                    &parent_master.record,
                    super::MASTER_STREAM_CHUNK_BYTES,
                    |chunk| {
                        decoder
                            .try_push(chunk, &mut |tile| {
                                let admitted = usize::from(tile.admissible_planes());
                                if admitted == 0 || admitted > 3 || admitted > tile.losses().len() {
                                    return Err(Qwen36PhysicalAllocationError::Campaign(
                                        Qwen36TensorWorkError::WorkspaceMismatch(
                                            "physical allocation admitted prefix",
                                        ),
                                    ));
                                }
                                let admitted_loss = tile.losses()[admitted - 1].hessian();
                                let mut losses = [admitted_loss; 3];
                                for (destination, loss) in
                                    losses.iter_mut().zip(&tile.losses()[..admitted])
                                {
                                    *destination = loss.hessian();
                                }
                                let curve = UniformPrefixCurve::new(losses).map_err(|error| {
                                    Qwen36PhysicalAllocationError::Allocation(
                                        UniformProfileAllocError::Allocation(error),
                                    )
                                })?;
                                planner
                                    .push(curve)
                                    .map_err(Qwen36PhysicalAllocationError::Allocation)?;
                                local_tiles = local_tiles.checked_add(1).ok_or({
                                    Qwen36PhysicalAllocationError::Campaign(
                                        Qwen36TensorWorkError::LengthOverflow(
                                            "physical allocation tensor tiles",
                                        ),
                                    )
                                })?;
                                Ok(())
                            })
                            .map_err(|error| match error {
                                SaltV2MasterVisitError::Master(error) => {
                                    Qwen36PhysicalAllocationError::Campaign(
                                        Qwen36TensorWorkError::Master(error),
                                    )
                                }
                                SaltV2MasterVisitError::Visitor(error) => error,
                            })
                    },
                )
                .map_err(|error| match error {
                    TensorVisitError::Store(error) => Qwen36PhysicalAllocationError::Campaign(
                        Qwen36TensorWorkError::TensorStore(error),
                    ),
                    TensorVisitError::Sink(error) => error,
                })?;
            decoder.finish().map_err(Qwen36TensorWorkError::Master)?;
            if local_tiles != expected.tile_count() {
                return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                    "physical allocation tensor tile count",
                )
                .into());
            }
        }
        Ok(())
    }

    /// Validate and durably bind one exact nested allocation to this sealed PTQ parent.
    ///
    /// `counts` is pulled exactly once per parent tile plus one terminal read.
    /// Both profiles are packed through bounded staging into separate canonical
    /// two-bit CAS records. The small manifest is published last, so a source or
    /// validation failure never exposes a reachable selection.
    ///
    /// # Errors
    /// Returns [`Qwen36SelectedAllocationBindError::Campaign`] unless the parent
    /// strictly reopens as complete, the source covers every tile exactly once,
    /// Compact is a prefix of NearLossless, every count is admitted by its parent
    /// master, and durable publication succeeds. A caller error is returned as
    /// [`Qwen36SelectedAllocationBindError::Source`].
    #[cfg(unix)]
    pub fn bind_selected_allocation<'parent, I, E>(
        &'parent self,
        spec: Qwen36SelectedAllocationSpec,
        counts: I,
    ) -> Result<
        Qwen36AllocatedCampaignStore<'parent, 'store, 'source>,
        Qwen36SelectedAllocationBindError<E>,
    >
    where
        I: IntoIterator<Item = Result<(u8, u8), E>>,
    {
        let _mutation = self.begin_mutation()?;
        let (parent_completion, parent_manifest, _) =
            self.require_complete_verified(FixedCampaignMode::Capture)?;

        let (root, objects, directories) = self.open_selection_store()?;
        reclaim_selection_orphans(
            &root,
            &objects,
            Some((&parent_completion, self.campaign_id)),
        )?;
        let mut compact_staged = StagedPackedMap::new(objects.temporary_dir(), "compact.map")?;
        let mut near_staged = StagedPackedMap::new(objects.temporary_dir(), "near.map")?;
        let mut counts = counts.into_iter();
        let validated = self.stage_nested_allocation(
            &parent_completion,
            &parent_manifest.masters,
            &spec,
            &mut counts,
            &mut compact_staged,
            &mut near_staged,
        )?;

        let compact_record_spec = allocation_map_record_spec(
            &parent_completion,
            &spec,
            SaltV2Profile::CompactV1,
            validated.tensor_count,
            validated.tile_count,
        )?;
        let near_record_spec = allocation_map_record_spec(
            &parent_completion,
            &spec,
            SaltV2Profile::NearLosslessV1,
            validated.tensor_count,
            validated.tile_count,
        )?;
        let compact_record =
            put_staged_allocation_map(&objects, &compact_record_spec, &mut compact_staged)?;
        let near_record = put_staged_allocation_map(&objects, &near_record_spec, &mut near_staged)?;
        drop((compact_staged, near_staged));

        let current_parent = self.require_complete_verified(FixedCampaignMode::Skip)?.0;
        if current_parent != parent_completion {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "selected allocation terminal parent",
            )
            .into());
        }
        let compact = selected_profile_receipt(
            compact_record,
            validated.compact_map_id,
            validated.compact_loss_id,
            validated.compact_selected_planes,
        );
        let near_lossless = selected_profile_receipt(
            near_record,
            validated.near_map_id,
            validated.near_loss_id,
            validated.near_selected_planes,
        );
        let mut receipt = Qwen36SelectedAllocationReceipt {
            selection_id: ContentId::from_digest([0; 32]),
            parent_completion_id: parent_completion.completion_id(),
            base_workspace_id: parent_completion.base_workspace_id(),
            campaign_id: parent_completion.campaign_id(),
            master_set_id: parent_completion.master_set_id(),
            source_model_id: parent_completion.source_model_id(),
            identity_status: parent_completion.identity_status(),
            spec,
            tensor_count: validated.tensor_count,
            tile_count: validated.tile_count,
            nested_allocation_id: validated.nested_id,
            compact,
            near_lossless,
        };
        let bytes = receipt.canonical_bytes()?;
        receipt.selection_id = ContentId::of_bytes(&bytes);
        validate_receipt_binding(&receipt, &parent_completion, self.campaign_id)?;
        validate_map_record_descriptors(&receipt)?;
        verify_map_records(
            self,
            &objects,
            &self.spec.expected_masters,
            &parent_manifest.masters,
            &receipt,
            &parent_completion,
        )?;
        let current_parent = self.require_complete_verified(FixedCampaignMode::Skip)?.0;
        if current_parent != parent_completion {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "selected allocation terminal parent",
            )
            .into());
        }
        persist_exact(
            &root.join(SELECTION_FILE),
            &bytes,
            "selected allocation manifest",
        )?;
        self.verify_completion_receipt(&parent_completion)?;
        validate_directories(&directories)?;
        self.verify_selected_allocation_receipt(&receipt, &parent_completion)?;

        Ok(Qwen36AllocatedCampaignStore {
            parent: self,
            receipt,
            parent_completion,
        })
    }

    /// Reopen the exact durable nested allocation already bound to this parent.
    ///
    /// # Errors
    /// Returns [`Qwen36TensorWorkError`] for a missing, malformed, changed, or
    /// parent-incompatible manifest/map record.
    #[cfg(unix)]
    pub fn reopen_selected_allocation<'parent>(
        &'parent self,
    ) -> Result<Qwen36AllocatedCampaignStore<'parent, 'store, 'source>, Qwen36TensorWorkError> {
        let (parent_completion, parent_manifest, _) =
            self.require_complete_verified(FixedCampaignMode::Capture)?;
        let (root, objects, directories) = self.open_selection_store()?;
        let receipt = read_selection_receipt(&root.join(SELECTION_FILE))?;
        validate_receipt_binding(&receipt, &parent_completion, self.campaign_id)?;
        validate_map_record_descriptors(&receipt)?;
        verify_map_records(
            self,
            &objects,
            &self.spec.expected_masters,
            &parent_manifest.masters,
            &receipt,
            &parent_completion,
        )?;
        self.verify_completion_receipt(&parent_completion)?;
        validate_directories(&directories)?;
        Ok(Qwen36AllocatedCampaignStore {
            parent: self,
            receipt,
            parent_completion,
        })
    }

    /// Reject selected-allocation mutation where stable file identity is unavailable.
    #[cfg(not(unix))]
    pub fn bind_selected_allocation<'parent, I, E>(
        &'parent self,
        _spec: Qwen36SelectedAllocationSpec,
        _counts: I,
    ) -> Result<
        Qwen36AllocatedCampaignStore<'parent, 'store, 'source>,
        Qwen36SelectedAllocationBindError<E>,
    >
    where
        I: IntoIterator<Item = Result<(u8, u8), E>>,
    {
        Err(Qwen36SelectedAllocationBindError::Campaign(
            Qwen36TensorWorkError::AdditiveCampaignUnsupportedPlatform,
        ))
    }

    /// Reject selected-allocation reopen where stable file identity is unavailable.
    #[cfg(not(unix))]
    pub fn reopen_selected_allocation<'parent>(
        &'parent self,
    ) -> Result<Qwen36AllocatedCampaignStore<'parent, 'store, 'source>, Qwen36TensorWorkError> {
        Err(Qwen36TensorWorkError::AdditiveCampaignUnsupportedPlatform)
    }

    pub(super) fn verify_selected_allocation_receipt(
        &self,
        expected: &Qwen36SelectedAllocationReceipt,
        parent_completion: &Qwen36CompleteWorkspaceReceipt,
    ) -> Result<(), Qwen36TensorWorkError> {
        #[cfg(not(unix))]
        {
            let _ = (expected, parent_completion);
            Err(Qwen36TensorWorkError::AdditiveCampaignUnsupportedPlatform)
        }
        #[cfg(unix)]
        {
            let (current_parent, parent_manifest, _) =
                self.require_complete_verified(FixedCampaignMode::Capture)?;
            if current_parent != *parent_completion {
                return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                    "selected allocation parent completion",
                ));
            }
            let (root, objects, directories) = self.open_selection_store()?;
            let current = read_selection_receipt(&root.join(SELECTION_FILE))?;
            if current != *expected {
                return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                    "selected allocation receipt",
                ));
            }
            validate_receipt_binding(&current, parent_completion, self.campaign_id)?;
            validate_map_record_descriptors(&current)?;
            verify_map_records(
                self,
                &objects,
                &self.spec.expected_masters,
                &parent_manifest.masters,
                &current,
                parent_completion,
            )?;
            validate_directories(&directories)?;
            self.verify_completion_receipt(parent_completion)
        }
    }

    #[cfg(unix)]
    fn open_selection_store(
        &self,
    ) -> Result<(PathBuf, TensorWorkStore, Vec<PinnedDirectory>), Qwen36TensorWorkError> {
        self.ensure_current()?;
        let root = self.root.join(SELECTION_DIRECTORY);
        ensure_durable_directory(&root, "selected allocation directory")
            .map_err(Qwen36TensorWorkError::TensorStore)?;
        let objects = TensorWorkStore::open(&root).map_err(Qwen36TensorWorkError::TensorStore)?;
        objects
            .scavenge_temporary()
            .map_err(Qwen36TensorWorkError::TensorStore)?;
        let paths = [
            root.as_path(),
            objects.objects_dir(),
            objects.temporary_dir(),
        ];
        let mut directories = Vec::new();
        directories
            .try_reserve_exact(paths.len())
            .map_err(|_| Qwen36TensorWorkError::AllocationFailed)?;
        for path in paths {
            directories.push(PinnedDirectory::pin(path)?);
        }
        validate_directories(&directories)?;
        self.ensure_current()?;
        Ok((root, objects, directories))
    }

    #[cfg(unix)]
    fn stage_nested_allocation<I, E>(
        &self,
        completion: &Qwen36CompleteWorkspaceReceipt,
        parent_masters: &[Qwen36AdditiveMasterReceipt],
        spec: &Qwen36SelectedAllocationSpec,
        counts: &mut I,
        compact_staged: &mut StagedPackedMap,
        near_staged: &mut StagedPackedMap,
    ) -> Result<ValidatedNestedAllocation, Qwen36SelectedAllocationBindError<E>>
    where
        I: Iterator<Item = Result<(u8, u8), E>>,
    {
        if parent_masters.len() != self.spec.expected_masters.len() {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "selected allocation parent tensor count",
            )
            .into());
        }
        let tensor_count = u64::try_from(self.spec.expected_masters.len())
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("selected tensor count"))?;
        let tile_count = self
            .spec
            .expected_masters
            .iter()
            .try_fold(0_u64, |total, master| {
                total
                    .checked_add(u64::try_from(master.tile_count()).map_err(|_| {
                        Qwen36TensorWorkError::LengthOverflow("selected allocation tile count")
                    })?)
                    .ok_or(Qwen36TensorWorkError::LengthOverflow(
                        "selected allocation tile count",
                    ))
            })?;
        let mut compact_map = ProfileIdentityHasher::new(
            SaltV2Profile::CompactV1,
            completion,
            spec,
            tensor_count,
            tile_count,
        );
        let mut near_map = ProfileIdentityHasher::new(
            SaltV2Profile::NearLosslessV1,
            completion,
            spec,
            tensor_count,
            tile_count,
        );
        let mut compact_loss =
            ProfileLossHasher::new(SaltV2Profile::CompactV1, completion, spec, tile_count);
        let mut near_loss =
            ProfileLossHasher::new(SaltV2Profile::NearLosslessV1, completion, spec, tile_count);
        let mut nested = blake3::Hasher::new_derive_key(NESTED_ALLOCATION_CONTEXT);
        nested.update(completion.completion_id().as_bytes());
        nested.update(&completion.master_set_id());
        nested.update(spec.spec_id().as_bytes());
        nested.update(&tensor_count.to_le_bytes());
        nested.update(&tile_count.to_le_bytes());

        let mut global_tile = 0_u64;
        let mut compact_selected_planes = 0_u64;
        let mut near_selected_planes = 0_u64;
        for (ordinal, (expected, parent_master)) in self
            .spec
            .expected_masters
            .iter()
            .zip(parent_masters)
            .enumerate()
        {
            validate_codec_compatibility(spec.codec, expected.geometry().constraint)?;
            compact_map.begin_tensor(ordinal, expected)?;
            near_map.begin_tensor(ordinal, expected)?;
            compact_loss.begin_tensor(ordinal, expected)?;
            near_loss.begin_tensor(ordinal, expected)?;
            let ordinal_u64 = u64::try_from(ordinal)
                .map_err(|_| Qwen36TensorWorkError::LengthOverflow("selected tensor ordinal"))?;
            let name_len = u64::try_from(expected.name().len())
                .map_err(|_| Qwen36TensorWorkError::LengthOverflow("selected tensor name"))?;
            let expected_tiles = u64::try_from(expected.tile_count())
                .map_err(|_| Qwen36TensorWorkError::LengthOverflow("selected tensor tiles"))?;
            nested.update(&ordinal_u64.to_le_bytes());
            nested.update(&name_len.to_le_bytes());
            nested.update(expected.name().as_bytes());
            nested.update(&expected_tiles.to_le_bytes());

            let mut decoder =
                SaltV2MasterTensorDecoder::new(expected).map_err(Qwen36TensorWorkError::Master)?;
            let mut local_tile = 0_usize;
            self.objects
                .try_visit_verified(
                    &parent_master.record,
                    super::MASTER_STREAM_CHUNK_BYTES,
                    |chunk| {
                        decoder
                            .try_push(chunk, &mut |tile| {
                                let (compact, near) = match counts.next() {
                                    Some(Ok(pair)) => pair,
                                    Some(Err(error)) => {
                                        return Err(Qwen36SelectedAllocationBindError::Source(
                                            error,
                                        ));
                                    }
                                    None => {
                                        return Err(Qwen36SelectedAllocationBindError::Campaign(
                                            Qwen36TensorWorkError::WorkspaceMismatch(
                                                "selected allocation source is short",
                                            ),
                                        ));
                                    }
                                };
                                if compact == 0
                                    || near == 0
                                    || compact > near
                                    || near > tile.admissible_planes()
                                {
                                    return Err(Qwen36SelectedAllocationBindError::Campaign(
                                        Qwen36TensorWorkError::WorkspaceMismatch(
                                            "selected nested admissible prefixes",
                                        ),
                                    ));
                                }
                                compact_staged
                                    .push_count(compact)
                                    .map_err(Qwen36SelectedAllocationBindError::Campaign)?;
                                near_staged
                                    .push_count(near)
                                    .map_err(Qwen36SelectedAllocationBindError::Campaign)?;
                                compact_map.push_count(compact);
                                near_map.push_count(near);
                                compact_loss.push_loss(tile.losses()[usize::from(compact - 1)]);
                                near_loss.push_loss(tile.losses()[usize::from(near - 1)]);
                                nested.update(&[compact, near]);
                                compact_selected_planes = compact_selected_planes
                                    .checked_add(u64::from(compact))
                                    .ok_or_else(|| {
                                        Qwen36SelectedAllocationBindError::Campaign(
                                            Qwen36TensorWorkError::LengthOverflow(
                                                "selected CompactV1 plane count",
                                            ),
                                        )
                                    })?;
                                near_selected_planes = near_selected_planes
                                    .checked_add(u64::from(near))
                                    .ok_or_else(|| {
                                        Qwen36SelectedAllocationBindError::Campaign(
                                            Qwen36TensorWorkError::LengthOverflow(
                                                "selected NearLosslessV1 plane count",
                                            ),
                                        )
                                    })?;
                                global_tile += 1;
                                local_tile += 1;
                                Ok(())
                            })
                            .map_err(|error| match error {
                                SaltV2MasterVisitError::Master(error) => {
                                    Qwen36SelectedAllocationBindError::Campaign(
                                        Qwen36TensorWorkError::Master(error),
                                    )
                                }
                                SaltV2MasterVisitError::Visitor(error) => error,
                            })
                    },
                )
                .map_err(|error| match error {
                    TensorVisitError::Store(error) => Qwen36SelectedAllocationBindError::Campaign(
                        Qwen36TensorWorkError::TensorStore(error),
                    ),
                    TensorVisitError::Sink(error) => error,
                })?;
            decoder.finish().map_err(Qwen36TensorWorkError::Master)?;
            if local_tile != expected.tile_count() {
                return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                    "selected allocation tensor tile count",
                )
                .into());
            }
        }
        match counts.next() {
            Some(Ok(_)) => {
                return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                    "selected allocation source is long",
                )
                .into());
            }
            Some(Err(error)) => return Err(Qwen36SelectedAllocationBindError::Source(error)),
            None => {}
        }
        if global_tile != tile_count {
            return Err(
                Qwen36TensorWorkError::WorkspaceMismatch("selected allocation tile count").into(),
            );
        }
        compact_staged.seal(packed_map_bytes(tile_count)?)?;
        near_staged.seal(packed_map_bytes(tile_count)?)?;
        self.verify_completion_receipt(completion)?;
        Ok(ValidatedNestedAllocation {
            tensor_count,
            tile_count,
            compact_map_id: compact_map.finish(),
            near_map_id: near_map.finish(),
            compact_loss_id: compact_loss.finish(),
            near_loss_id: near_loss.finish(),
            nested_id: ContentId::from_digest(*nested.finalize().as_bytes()),
            compact_selected_planes,
            near_selected_planes,
        })
    }
}

impl<'parent, 'store, 'source> Qwen36AllocatedCampaignStore<'parent, 'store, 'source> {
    pub(crate) fn verify_cheap_current(&self) -> Result<(), Qwen36TensorWorkError> {
        let (root, _objects, _directories) = self.parent.open_selection_store()?;
        let current = read_selection_receipt(&root.join(SELECTION_FILE))?;
        if current != self.receipt {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "selected allocation receipt",
            ));
        }
        validate_receipt_binding(&current, &self.parent_completion, self.parent.campaign_id)?;
        validate_map_record_descriptors(&current)?;
        self.parent
            .verify_completion_receipt(&self.parent_completion)
    }

    /// Strictly revalidate this allocation, both maps, and every parent prefix.
    ///
    /// # Errors
    /// Returns [`Qwen36TensorWorkError`] for changed, missing, malformed, or
    /// parent-incompatible durable state.
    pub fn verify_current(&self) -> Result<(), Qwen36TensorWorkError> {
        self.parent
            .verify_selected_allocation_receipt(&self.receipt, &self.parent_completion)
    }
}

#[derive(Clone, Copy, Debug)]
struct ValidatedNestedAllocation {
    tensor_count: u64,
    tile_count: u64,
    compact_map_id: ContentId,
    near_map_id: ContentId,
    compact_loss_id: ContentId,
    near_loss_id: ContentId,
    nested_id: ContentId,
    compact_selected_planes: u64,
    near_selected_planes: u64,
}

#[derive(Debug)]
struct ProfileIdentityHasher {
    hasher: blake3::Hasher,
}

impl ProfileIdentityHasher {
    fn new(
        profile: SaltV2Profile,
        completion: &Qwen36CompleteWorkspaceReceipt,
        spec: &Qwen36SelectedAllocationSpec,
        tensor_count: u64,
        tile_count: u64,
    ) -> Self {
        let mut hasher = blake3::Hasher::new_derive_key(ALLOCATION_MAP_CONTEXT);
        hasher.update(&[profile_tag(profile)]);
        hasher.update(completion.completion_id().as_bytes());
        hasher.update(completion.campaign_id().as_bytes());
        hasher.update(&completion.master_set_id());
        hasher.update(spec.spec_id().as_bytes());
        hasher.update(&tensor_count.to_le_bytes());
        hasher.update(&tile_count.to_le_bytes());
        Self { hasher }
    }

    fn from_receipt(profile: SaltV2Profile, receipt: &Qwen36SelectedAllocationReceipt) -> Self {
        let mut hasher = blake3::Hasher::new_derive_key(ALLOCATION_MAP_CONTEXT);
        hasher.update(&[profile_tag(profile)]);
        hasher.update(receipt.parent_completion_id.as_bytes());
        hasher.update(receipt.campaign_id.as_bytes());
        hasher.update(&receipt.master_set_id);
        hasher.update(receipt.spec.spec_id().as_bytes());
        hasher.update(&receipt.tensor_count.to_le_bytes());
        hasher.update(&receipt.tile_count.to_le_bytes());
        Self { hasher }
    }

    fn begin_tensor(
        &mut self,
        ordinal: usize,
        spec: &SaltV2MasterTensorSpec,
    ) -> Result<(), Qwen36TensorWorkError> {
        let ordinal = u64::try_from(ordinal)
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("selected tensor ordinal"))?;
        let name_len = u64::try_from(spec.name().len())
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("selected tensor name"))?;
        let rank = u64::try_from(spec.shape().len())
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("selected tensor rank"))?;
        self.hasher.update(&ordinal.to_le_bytes());
        self.hasher.update(&name_len.to_le_bytes());
        self.hasher.update(spec.name().as_bytes());
        self.hasher.update(&rank.to_le_bytes());
        for dimension in spec.shape() {
            self.hasher.update(&dimension.to_le_bytes());
        }
        let tile_count = u64::try_from(spec.tile_count())
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("selected tensor tiles"))?;
        self.hasher.update(&tile_count.to_le_bytes());
        Ok(())
    }

    fn push_count(&mut self, count: u8) {
        self.hasher.update(&[count]);
    }

    fn finish(self) -> ContentId {
        ContentId::from_digest(*self.hasher.finalize().as_bytes())
    }
}

#[derive(Debug)]
struct ProfileLossHasher {
    hasher: blake3::Hasher,
}

impl ProfileLossHasher {
    fn new(
        profile: SaltV2Profile,
        completion: &Qwen36CompleteWorkspaceReceipt,
        spec: &Qwen36SelectedAllocationSpec,
        tile_count: u64,
    ) -> Self {
        let mut hasher = blake3::Hasher::new_derive_key(SELECTED_LOSS_CONTEXT);
        hasher.update(&[profile_tag(profile)]);
        hasher.update(completion.completion_id().as_bytes());
        hasher.update(&completion.master_set_id());
        hasher.update(spec.spec_id().as_bytes());
        hasher.update(&tile_count.to_le_bytes());
        Self { hasher }
    }

    fn begin_tensor(
        &mut self,
        ordinal: usize,
        spec: &SaltV2MasterTensorSpec,
    ) -> Result<(), Qwen36TensorWorkError> {
        let ordinal = u64::try_from(ordinal)
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("selected tensor ordinal"))?;
        let name_len = u64::try_from(spec.name().len())
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("selected tensor name"))?;
        self.hasher.update(&ordinal.to_le_bytes());
        self.hasher.update(&name_len.to_le_bytes());
        self.hasher.update(spec.name().as_bytes());
        Ok(())
    }

    fn push_loss(&mut self, loss: tritium_format::salt_v2_master::SaltV2PrefixLoss) {
        self.hasher.update(&loss.hessian().to_bits().to_le_bytes());
        self.hasher
            .update(&loss.frobenius().to_bits().to_le_bytes());
    }

    fn finish(self) -> ContentId {
        ContentId::from_digest(*self.hasher.finalize().as_bytes())
    }
}

fn validate_codec_compatibility(
    codec: SaltV2Codec,
    constraint: SaltV2FitConstraint,
) -> Result<(), Qwen36TensorWorkError> {
    if codec == SaltV2Codec::S34 && constraint != SaltV2FitConstraint::S34 {
        return Err(Qwen36TensorWorkError::WorkspaceMismatch(
            "selected allocation codec and fit constraint",
        ));
    }
    codec_tag(codec).map(|_| ())
}

fn selected_profile_receipt(
    map_record: TensorRecordReceipt,
    allocation_map_id: ContentId,
    selected_loss_id: ContentId,
    selected_planes: u64,
) -> Qwen36SelectedProfileReceipt {
    Qwen36SelectedProfileReceipt {
        allocation_map_id,
        map_record,
        selected_planes,
        selected_loss_id,
    }
}

fn allocation_map_record_spec(
    completion: &Qwen36CompleteWorkspaceReceipt,
    spec: &Qwen36SelectedAllocationSpec,
    profile: SaltV2Profile,
    tensor_count: u64,
    tile_count: u64,
) -> Result<TensorRecordSpec, Qwen36TensorWorkError> {
    allocation_map_record_spec_from_fields(
        completion.completion_id(),
        completion.base_workspace_id(),
        completion.campaign_id(),
        completion.master_set_id(),
        completion.source_model_id(),
        spec,
        profile,
        tensor_count,
        tile_count,
    )
}

#[allow(clippy::too_many_arguments)]
fn allocation_map_record_spec_from_fields(
    parent_completion_id: ContentId,
    base_workspace_id: ContentId,
    campaign_id: ContentId,
    master_set_id: [u8; 32],
    source_model_id: ModelId,
    spec: &Qwen36SelectedAllocationSpec,
    profile: SaltV2Profile,
    tensor_count: u64,
    tile_count: u64,
) -> Result<TensorRecordSpec, Qwen36TensorWorkError> {
    if tile_count == 0 {
        return Err(Qwen36TensorWorkError::WorkspaceMalformed(
            "selected allocation tile count",
        ));
    }
    let mut metadata = Vec::new();
    metadata.extend_from_slice(&MAP_METADATA_MAGIC);
    metadata.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    metadata.push(profile_tag(profile));
    metadata.push(codec_tag(spec.codec)?);
    metadata.extend_from_slice(parent_completion_id.as_bytes());
    metadata.extend_from_slice(base_workspace_id.as_bytes());
    metadata.extend_from_slice(campaign_id.as_bytes());
    metadata.extend_from_slice(&master_set_id);
    metadata.extend_from_slice(spec.spec_id().as_bytes());
    metadata.extend_from_slice(&tensor_count.to_le_bytes());
    metadata.extend_from_slice(&tile_count.to_le_bytes());
    let payload_bytes = packed_map_bytes(tile_count)?;
    TensorRecordSpec::new(
        ContentId::of_bytes(MAP_SCHEMA),
        source_model_id,
        master_set_id,
        match profile {
            SaltV2Profile::CompactV1 => "salt-v2.compact-v1.allocation-map",
            SaltV2Profile::NearLosslessV1 => "salt-v2.near-lossless-v1.allocation-map",
        },
        vec![tile_count],
        metadata,
        payload_bytes,
    )
    .map_err(Qwen36TensorWorkError::TensorStore)
}

fn packed_map_bytes(tile_count: u64) -> Result<u64, Qwen36TensorWorkError> {
    tile_count
        .checked_add(3)
        .map(|count| count / 4)
        .ok_or(Qwen36TensorWorkError::LengthOverflow(
            "selected allocation map bytes",
        ))
}

#[cfg(unix)]
#[derive(Debug)]
struct StagedPackedMap {
    path: PathBuf,
    file: File,
    buffer: Vec<u8>,
    partial_byte: u8,
    partial_slots: u8,
    written: u64,
    digest: blake3::Hasher,
    sealed: Option<(u64, [u8; 32])>,
}

#[cfg(unix)]
impl StagedPackedMap {
    fn new(directory: &Path, prefix: &str) -> Result<Self, Qwen36TensorWorkError> {
        let (path, file) =
            create_temporary_file(directory, prefix).map_err(Qwen36TensorWorkError::TensorStore)?;
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(MAP_WRITE_BUFFER_BYTES)
            .map_err(|_| Qwen36TensorWorkError::AllocationFailed)?;
        Ok(Self {
            path,
            file,
            buffer,
            partial_byte: 0,
            partial_slots: 0,
            written: 0,
            digest: blake3::Hasher::new(),
            sealed: None,
        })
    }

    fn push_count(&mut self, count: u8) -> Result<(), Qwen36TensorWorkError> {
        if self.sealed.is_some() || !(1..=3).contains(&count) {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "selected allocation staged count",
            ));
        }
        self.partial_byte |= (count - 1) << (self.partial_slots * 2);
        self.partial_slots += 1;
        if self.partial_slots == 4 {
            self.push_byte(self.partial_byte)?;
            self.partial_byte = 0;
            self.partial_slots = 0;
        }
        Ok(())
    }

    fn write_verified_chunk(&mut self, bytes: &[u8]) -> Result<(), Qwen36TensorWorkError> {
        if self.sealed.is_some() || self.partial_slots != 0 || !self.buffer.is_empty() {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "selected allocation staged map state",
            ));
        }
        self.file
            .write_all(bytes)
            .map_err(|error| work_io("stage selected allocation map", error))?;
        self.digest.update(bytes);
        self.written = self
            .written
            .checked_add(u64::try_from(bytes.len()).map_err(|_| {
                Qwen36TensorWorkError::LengthOverflow("selected allocation staged map")
            })?)
            .ok_or(Qwen36TensorWorkError::LengthOverflow(
                "selected allocation staged map",
            ))?;
        Ok(())
    }

    fn push_byte(&mut self, byte: u8) -> Result<(), Qwen36TensorWorkError> {
        self.buffer.push(byte);
        if self.buffer.len() == MAP_WRITE_BUFFER_BYTES {
            self.flush_buffer()?;
        }
        Ok(())
    }

    fn flush_buffer(&mut self) -> Result<(), Qwen36TensorWorkError> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        self.file
            .write_all(&self.buffer)
            .map_err(|error| work_io("stage selected allocation map", error))?;
        self.digest.update(&self.buffer);
        self.written = self
            .written
            .checked_add(u64::try_from(self.buffer.len()).map_err(|_| {
                Qwen36TensorWorkError::LengthOverflow("selected allocation staged map")
            })?)
            .ok_or(Qwen36TensorWorkError::LengthOverflow(
                "selected allocation staged map",
            ))?;
        self.buffer.clear();
        Ok(())
    }

    fn seal(&mut self, expected_bytes: u64) -> Result<(), Qwen36TensorWorkError> {
        if self.sealed.is_some() {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "selected allocation staged map reseal",
            ));
        }
        if self.partial_slots != 0 {
            self.push_byte(self.partial_byte)?;
            self.partial_byte = 0;
            self.partial_slots = 0;
        }
        self.flush_buffer()?;
        self.file
            .flush()
            .map_err(|error| work_io("flush selected allocation map", error))?;
        let actual = self
            .file
            .metadata()
            .map_err(|error| work_io("inspect selected allocation staging", error))?
            .len();
        if actual != expected_bytes || self.written != expected_bytes {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "selected allocation staged map length",
            ));
        }
        self.file
            .seek(SeekFrom::Start(0))
            .map_err(|error| work_io("rewind selected allocation map", error))?;
        self.sealed = Some((expected_bytes, *self.digest.finalize().as_bytes()));
        Ok(())
    }

    fn copy_verified_to(&mut self, writer: &mut impl Write) -> io::Result<()> {
        let (expected_bytes, expected_digest) = self.sealed.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "unsealed selected allocation map",
            )
        })?;
        let before = self.file.metadata()?.len();
        if before != expected_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "changed selected allocation staging length",
            ));
        }
        self.file.seek(SeekFrom::Start(0))?;
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(MAP_WRITE_BUFFER_BYTES)
            .map_err(|_| io::Error::other("selected allocation staging allocation failed"))?;
        buffer.resize(MAP_WRITE_BUFFER_BYTES, 0);
        let mut remaining = expected_bytes;
        let mut hasher = blake3::Hasher::new();
        while remaining != 0 {
            let count = usize::try_from(remaining.min(MAP_WRITE_BUFFER_BYTES_U64))
                .map_err(|_| io::Error::other("selected allocation map length overflow"))?;
            self.file.read_exact(&mut buffer[..count])?;
            hasher.update(&buffer[..count]);
            writer.write_all(&buffer[..count])?;
            remaining -= u64::try_from(count)
                .map_err(|_| io::Error::other("selected allocation map length overflow"))?;
        }
        let after = self.file.metadata()?.len();
        if after != before || hasher.finalize().as_bytes() != &expected_digest {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "changed selected allocation staging bytes",
            ));
        }
        Ok(())
    }

    fn cursor(&mut self, tile_count: u64) -> Result<PackedMapCursor<'_>, Qwen36TensorWorkError> {
        let (expected_bytes, expected_digest) =
            self.sealed
                .ok_or(Qwen36TensorWorkError::WorkspaceMalformed(
                    "unsealed selected allocation map",
                ))?;
        self.file
            .seek(SeekFrom::Start(0))
            .map_err(|error| work_io("rewind selected allocation map", error))?;
        let before_len = self
            .file
            .metadata()
            .map_err(|error| work_io("inspect selected allocation staging", error))?
            .len();
        if before_len != expected_bytes || expected_bytes != packed_map_bytes(tile_count)? {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "selected allocation staged map length",
            ));
        }
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(MAP_WRITE_BUFFER_BYTES)
            .map_err(|_| Qwen36TensorWorkError::AllocationFailed)?;
        buffer.resize(MAP_WRITE_BUFFER_BYTES, 0);
        Ok(PackedMapCursor {
            file: &mut self.file,
            buffer,
            buffer_start: 0,
            buffer_end: 0,
            loaded_bytes: 0,
            expected_bytes,
            before_len,
            expected_digest,
            digest: blake3::Hasher::new(),
            current_byte: 0,
            next_slot: 4,
            consumed_tiles: 0,
            tile_count,
        })
    }
}

#[cfg(unix)]
impl Drop for StagedPackedMap {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[cfg(unix)]
#[derive(Debug)]
struct PackedMapCursor<'a> {
    file: &'a mut File,
    buffer: Vec<u8>,
    buffer_start: usize,
    buffer_end: usize,
    loaded_bytes: u64,
    expected_bytes: u64,
    before_len: u64,
    expected_digest: [u8; 32],
    digest: blake3::Hasher,
    current_byte: u8,
    next_slot: u8,
    consumed_tiles: u64,
    tile_count: u64,
}

#[cfg(unix)]
impl PackedMapCursor<'_> {
    fn next_count(&mut self) -> Result<Option<u8>, Qwen36TensorWorkError> {
        if self.consumed_tiles == self.tile_count {
            return Ok(None);
        }
        if self.next_slot == 4 {
            self.current_byte = self.next_byte()?;
            self.next_slot = 0;
        }
        let code = (self.current_byte >> (self.next_slot * 2)) & 0b11;
        if code == 0b11 {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "selected allocation map code",
            ));
        }
        self.next_slot += 1;
        self.consumed_tiles += 1;
        Ok(Some(code + 1))
    }

    fn next_byte(&mut self) -> Result<u8, Qwen36TensorWorkError> {
        if self.buffer_start == self.buffer_end {
            let remaining = self.expected_bytes.checked_sub(self.loaded_bytes).ok_or(
                Qwen36TensorWorkError::LengthOverflow("selected allocation staged map"),
            )?;
            if remaining == 0 {
                return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                    "selected allocation map is short",
                ));
            }
            let count =
                usize::try_from(remaining.min(MAP_WRITE_BUFFER_BYTES_U64)).map_err(|_| {
                    Qwen36TensorWorkError::LengthOverflow("selected allocation staged map")
                })?;
            self.file
                .read_exact(&mut self.buffer[..count])
                .map_err(|error| work_io("read selected allocation staging", error))?;
            self.digest.update(&self.buffer[..count]);
            self.loaded_bytes = self
                .loaded_bytes
                .checked_add(u64::try_from(count).map_err(|_| {
                    Qwen36TensorWorkError::LengthOverflow("selected allocation staged map")
                })?)
                .ok_or(Qwen36TensorWorkError::LengthOverflow(
                    "selected allocation staged map",
                ))?;
            self.buffer_start = 0;
            self.buffer_end = count;
        }
        let byte = self.buffer[self.buffer_start];
        self.buffer_start += 1;
        Ok(byte)
    }

    fn finish(self) -> Result<(), Qwen36TensorWorkError> {
        if self.consumed_tiles != self.tile_count || self.loaded_bytes != self.expected_bytes {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "selected allocation map coverage",
            ));
        }
        if self.next_slot < 4 {
            for slot in self.next_slot..4 {
                if ((self.current_byte >> (slot * 2)) & 0b11) != 0 {
                    return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                        "selected allocation map padding",
                    ));
                }
            }
        }
        let after_len = self
            .file
            .metadata()
            .map_err(|error| work_io("reinspect selected allocation staging", error))?
            .len();
        if after_len != self.before_len
            || self.digest.finalize().as_bytes() != &self.expected_digest
        {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "selected allocation staged map changed",
            ));
        }
        Ok(())
    }
}

#[cfg(unix)]
fn put_staged_allocation_map(
    store: &TensorWorkStore,
    spec: &TensorRecordSpec,
    staged: &mut StagedPackedMap,
) -> Result<TensorRecordReceipt, Qwen36TensorWorkError> {
    store
        .put(spec, |writer| staged.copy_verified_to(writer))
        .map_err(|error| match error {
            TensorPutError::Store(error) => Qwen36TensorWorkError::TensorStore(error),
            TensorPutError::Producer(error) => work_io("write selected allocation map", error),
        })
}

fn profile_tag(profile: SaltV2Profile) -> u8 {
    match profile {
        SaltV2Profile::CompactV1 => 1,
        SaltV2Profile::NearLosslessV1 => 2,
    }
}

fn decode_allocation_spec(
    bytes: &[u8],
) -> Result<Qwen36SelectedAllocationSpec, Qwen36TensorWorkError> {
    let mut cursor = super::CanonicalCursor::new(bytes);
    let codec = codec_from_tag(cursor.u8()?)?;
    if cursor.take(7)? != [0; 7] {
        return Err(Qwen36TensorWorkError::WorkspaceMalformed(
            "selected allocation spec reserved bytes",
        ));
    }
    let allocator_id = cursor.digest()?;
    let recipe_id = cursor.digest()?;
    let compact = decode_profile_budget(&mut cursor)?;
    let near_lossless = decode_profile_budget(&mut cursor)?;
    if cursor.remaining() != 0 {
        return Err(Qwen36TensorWorkError::WorkspaceMalformed(
            "selected allocation spec trailing bytes",
        ));
    }
    let spec = Qwen36SelectedAllocationSpec::new(
        codec,
        allocator_id,
        recipe_id,
        NestedProfileBudgets {
            compact,
            near_lossless,
        },
    )?;
    if spec.canonical_bytes()? != bytes {
        return Err(Qwen36TensorWorkError::WorkspaceMalformed(
            "noncanonical selected allocation spec",
        ));
    }
    Ok(spec)
}

fn decode_profile_budget(
    cursor: &mut super::CanonicalCursor<'_>,
) -> Result<ProfileBudget, Qwen36TensorWorkError> {
    Ok(ProfileBudget {
        maximum: decode_physical_bytes(cursor)?,
        metadata: decode_byte_delta(cursor)?,
    })
}

fn decode_byte_delta(
    cursor: &mut super::CanonicalCursor<'_>,
) -> Result<ByteDelta, Qwen36TensorWorkError> {
    let declared = decode_physical_bytes(cursor)?;
    let measured_tag = cursor.u8()?;
    if cursor.take(7)? != [0; 7] {
        return Err(Qwen36TensorWorkError::WorkspaceMalformed(
            "selected allocation byte delta reserved bytes",
        ));
    }
    let measured = decode_physical_bytes(cursor)?;
    match measured_tag {
        0 if measured == PhysicalBytes::ZERO => Ok(ByteDelta::declared(declared)),
        1 => Ok(ByteDelta::measured(declared, measured)),
        _ => Err(Qwen36TensorWorkError::WorkspaceMalformed(
            "selected allocation byte delta",
        )),
    }
}

fn decode_physical_bytes(
    cursor: &mut super::CanonicalCursor<'_>,
) -> Result<PhysicalBytes, Qwen36TensorWorkError> {
    Ok(PhysicalBytes {
        serialized: cursor.u64()?,
        resident: cursor.u64()?,
    })
}

fn decode_profile_receipt(
    cursor: &mut super::CanonicalCursor<'_>,
) -> Result<Qwen36SelectedProfileReceipt, Qwen36TensorWorkError> {
    let allocation_map_id = ContentId::from_digest(cursor.digest()?);
    let selected_loss_id = ContentId::from_digest(cursor.digest()?);
    let selected_planes = cursor.u64()?;
    let record_len = usize::try_from(cursor.u32()?)
        .map_err(|_| Qwen36TensorWorkError::LengthOverflow("allocation map receipt"))?;
    let map_record = TensorRecordReceipt::from_canonical_bytes(cursor.take(record_len)?)
        .map_err(Qwen36TensorWorkError::TensorStore)?;
    Ok(Qwen36SelectedProfileReceipt {
        allocation_map_id,
        map_record,
        selected_planes,
        selected_loss_id,
    })
}

fn read_selection_receipt(
    path: &Path,
) -> Result<Qwen36SelectedAllocationReceipt, Qwen36TensorWorkError> {
    let bytes = read_regular_bounded(path, MAX_SELECTION_BYTES, "selected allocation manifest")?;
    Qwen36SelectedAllocationReceipt::from_canonical_bytes(&bytes)
}

fn validate_receipt_binding(
    receipt: &Qwen36SelectedAllocationReceipt,
    completion: &Qwen36CompleteWorkspaceReceipt,
    campaign_id: ContentId,
) -> Result<(), Qwen36TensorWorkError> {
    if receipt.parent_completion_id != completion.completion_id()
        || receipt.base_workspace_id != completion.base_workspace_id()
        || receipt.campaign_id != completion.campaign_id()
        || receipt.campaign_id != campaign_id
        || receipt.master_set_id != completion.master_set_id()
        || receipt.source_model_id != completion.source_model_id()
        || receipt.identity_status != completion.identity_status()
        || receipt.tensor_count == 0
        || receipt.tile_count == 0
        || receipt.compact.selected_planes > receipt.near_lossless.selected_planes
    {
        return Err(Qwen36TensorWorkError::WorkspaceMismatch(
            "selected allocation parent binding",
        ));
    }
    Ok(())
}

fn validate_map_record_descriptors(
    receipt: &Qwen36SelectedAllocationReceipt,
) -> Result<(), Qwen36TensorWorkError> {
    for (profile, selected) in [
        (SaltV2Profile::CompactV1, &receipt.compact),
        (SaltV2Profile::NearLosslessV1, &receipt.near_lossless),
    ] {
        let expected = allocation_map_record_spec_from_fields(
            receipt.parent_completion_id,
            receipt.base_workspace_id,
            receipt.campaign_id,
            receipt.master_set_id,
            receipt.source_model_id,
            &receipt.spec,
            profile,
            receipt.tensor_count,
            receipt.tile_count,
        )?;
        if !selected.map_record.matches_spec(&expected) {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "selected allocation map descriptor",
            ));
        }
    }
    Ok(())
}

#[cfg(unix)]
fn stage_verified_map(
    store: &TensorWorkStore,
    receipt: &TensorRecordReceipt,
    prefix: &str,
) -> Result<StagedPackedMap, Qwen36TensorWorkError> {
    let mut staged = StagedPackedMap::new(store.temporary_dir(), prefix)?;
    store
        .try_visit_verified(receipt, MAP_WRITE_BUFFER_BYTES, |chunk| {
            staged.write_verified_chunk(chunk)
        })
        .map_err(|error| match error {
            TensorVisitError::Store(error) => Qwen36TensorWorkError::TensorStore(error),
            TensorVisitError::Sink(error) => error,
        })?;
    staged.seal(receipt.info().payload_bytes())?;
    Ok(staged)
}

#[cfg(unix)]
fn verify_map_records(
    parent: &Qwen36AdditiveCampaignStore<'_, '_>,
    store: &TensorWorkStore,
    masters: &[SaltV2MasterTensorSpec],
    parent_masters: &[Qwen36AdditiveMasterReceipt],
    receipt: &Qwen36SelectedAllocationReceipt,
    completion: &Qwen36CompleteWorkspaceReceipt,
) -> Result<(), Qwen36TensorWorkError> {
    let expected_tensor_count = u64::try_from(masters.len())
        .map_err(|_| Qwen36TensorWorkError::LengthOverflow("selected tensor count"))?;
    let expected_tile_count = masters.iter().try_fold(0_u64, |total, master| {
        total
            .checked_add(u64::try_from(master.tile_count()).map_err(|_| {
                Qwen36TensorWorkError::LengthOverflow("selected allocation tile count")
            })?)
            .ok_or(Qwen36TensorWorkError::LengthOverflow(
                "selected allocation tile count",
            ))
    })?;
    if receipt.tensor_count != expected_tensor_count
        || receipt.tile_count != expected_tile_count
        || parent_masters.len() != masters.len()
    {
        return Err(Qwen36TensorWorkError::WorkspaceMismatch(
            "selected allocation governed coverage",
        ));
    }

    let mut compact_staged =
        stage_verified_map(store, &receipt.compact.map_record, "compact.verify")?;
    let mut near_staged =
        stage_verified_map(store, &receipt.near_lossless.map_record, "near.verify")?;
    let mut compact_counts = compact_staged.cursor(receipt.tile_count)?;
    let mut near_counts = near_staged.cursor(receipt.tile_count)?;
    let mut compact_map = ProfileIdentityHasher::from_receipt(SaltV2Profile::CompactV1, receipt);
    let mut near_map = ProfileIdentityHasher::from_receipt(SaltV2Profile::NearLosslessV1, receipt);
    let mut compact_loss = ProfileLossHasher::new(
        SaltV2Profile::CompactV1,
        completion,
        &receipt.spec,
        receipt.tile_count,
    );
    let mut near_loss = ProfileLossHasher::new(
        SaltV2Profile::NearLosslessV1,
        completion,
        &receipt.spec,
        receipt.tile_count,
    );
    let mut nested = blake3::Hasher::new_derive_key(NESTED_ALLOCATION_CONTEXT);
    nested.update(completion.completion_id().as_bytes());
    nested.update(&completion.master_set_id());
    nested.update(receipt.spec.spec_id().as_bytes());
    nested.update(&receipt.tensor_count.to_le_bytes());
    nested.update(&receipt.tile_count.to_le_bytes());
    let mut compact_selected_planes = 0_u64;
    let mut near_selected_planes = 0_u64;
    let mut global_tile = 0_u64;

    for (ordinal, (expected, parent_master)) in masters.iter().zip(parent_masters).enumerate() {
        validate_codec_compatibility(receipt.spec.codec, expected.geometry().constraint)?;
        compact_map.begin_tensor(ordinal, expected)?;
        near_map.begin_tensor(ordinal, expected)?;
        compact_loss.begin_tensor(ordinal, expected)?;
        near_loss.begin_tensor(ordinal, expected)?;
        let ordinal_u64 = u64::try_from(ordinal)
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("selected tensor ordinal"))?;
        let name_len = u64::try_from(expected.name().len())
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("selected tensor name"))?;
        let tensor_tiles = u64::try_from(expected.tile_count())
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("selected tensor tiles"))?;
        nested.update(&ordinal_u64.to_le_bytes());
        nested.update(&name_len.to_le_bytes());
        nested.update(expected.name().as_bytes());
        nested.update(&tensor_tiles.to_le_bytes());

        let mut decoder =
            SaltV2MasterTensorDecoder::new(expected).map_err(Qwen36TensorWorkError::Master)?;
        let mut local_tile = 0_usize;
        parent
            .objects
            .try_visit_verified(
                &parent_master.record,
                super::MASTER_STREAM_CHUNK_BYTES,
                |chunk| {
                    decoder
                        .try_push(chunk, &mut |tile| {
                            let compact = compact_counts.next_count()?.ok_or(
                                Qwen36TensorWorkError::WorkspaceMismatch(
                                    "selected CompactV1 map is short",
                                ),
                            )?;
                            let near = near_counts.next_count()?.ok_or(
                                Qwen36TensorWorkError::WorkspaceMismatch(
                                    "selected NearLosslessV1 map is short",
                                ),
                            )?;
                            if compact > near || near > tile.admissible_planes() {
                                return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                                    "selected nested admissible prefixes",
                                ));
                            }
                            compact_map.push_count(compact);
                            near_map.push_count(near);
                            compact_loss.push_loss(tile.losses()[usize::from(compact - 1)]);
                            near_loss.push_loss(tile.losses()[usize::from(near - 1)]);
                            nested.update(&[compact, near]);
                            compact_selected_planes = compact_selected_planes
                                .checked_add(u64::from(compact))
                                .ok_or(Qwen36TensorWorkError::LengthOverflow(
                                    "selected CompactV1 plane count",
                                ))?;
                            near_selected_planes = near_selected_planes
                                .checked_add(u64::from(near))
                                .ok_or(Qwen36TensorWorkError::LengthOverflow(
                                    "selected NearLosslessV1 plane count",
                                ))?;
                            global_tile = global_tile.checked_add(1).ok_or(
                                Qwen36TensorWorkError::LengthOverflow(
                                    "selected allocation tile count",
                                ),
                            )?;
                            local_tile += 1;
                            Ok(())
                        })
                        .map_err(|error| match error {
                            SaltV2MasterVisitError::Master(error) => {
                                Qwen36TensorWorkError::Master(error)
                            }
                            SaltV2MasterVisitError::Visitor(error) => error,
                        })
                },
            )
            .map_err(|error| match error {
                TensorVisitError::Store(error) => Qwen36TensorWorkError::TensorStore(error),
                TensorVisitError::Sink(error) => error,
            })?;
        let decoded = decoder.finish().map_err(Qwen36TensorWorkError::Master)?;
        if local_tile != expected.tile_count()
            || decoded.tensor_master_id() != parent_master.tensor_master_id()
        {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "selected allocation parent tensor master",
            ));
        }
    }

    if compact_counts.next_count()?.is_some()
        || near_counts.next_count()?.is_some()
        || global_tile != receipt.tile_count
    {
        return Err(Qwen36TensorWorkError::WorkspaceMismatch(
            "selected allocation map coverage",
        ));
    }
    compact_counts.finish()?;
    near_counts.finish()?;
    let validated = ValidatedNestedAllocation {
        tensor_count: receipt.tensor_count,
        tile_count: receipt.tile_count,
        compact_map_id: compact_map.finish(),
        near_map_id: near_map.finish(),
        compact_loss_id: compact_loss.finish(),
        near_loss_id: near_loss.finish(),
        nested_id: ContentId::from_digest(*nested.finalize().as_bytes()),
        compact_selected_planes,
        near_selected_planes,
    };
    if validated.compact_map_id != receipt.compact.allocation_map_id
        || validated.near_map_id != receipt.near_lossless.allocation_map_id
        || validated.compact_loss_id != receipt.compact.selected_loss_id
        || validated.near_loss_id != receipt.near_lossless.selected_loss_id
        || validated.nested_id != receipt.nested_allocation_id
        || validated.compact_selected_planes != receipt.compact.selected_planes
        || validated.near_selected_planes != receipt.near_lossless.selected_planes
    {
        return Err(Qwen36TensorWorkError::WorkspaceMismatch(
            "selected allocation semantic receipt",
        ));
    }
    parent.verify_completion_receipt(completion)
}

#[cfg(all(test, unix))]
pub(super) fn rewrite_selected_allocation_for_test(
    parent: &Qwen36AdditiveCampaignStore<'_, '_>,
    compact_counts: &[u8],
    near_counts: &[u8],
    nested_override: Option<ContentId>,
) -> Result<Qwen36SelectedAllocationReceipt, Qwen36TensorWorkError> {
    let (completion, parent_manifest, _) =
        parent.require_complete_verified(FixedCampaignMode::Capture)?;
    let (root, objects, _) = parent.open_selection_store()?;
    let mut receipt = read_selection_receipt(&root.join(SELECTION_FILE))?;
    let tile_count = usize::try_from(receipt.tile_count)
        .map_err(|_| Qwen36TensorWorkError::LengthOverflow("selected allocation tile count"))?;
    if compact_counts.len() != tile_count || near_counts.len() != tile_count {
        return Err(Qwen36TensorWorkError::WorkspaceMismatch(
            "selected allocation test map length",
        ));
    }

    let mut compact_map = ProfileIdentityHasher::from_receipt(SaltV2Profile::CompactV1, &receipt);
    let mut near_map = ProfileIdentityHasher::from_receipt(SaltV2Profile::NearLosslessV1, &receipt);
    let mut compact_loss = ProfileLossHasher::new(
        SaltV2Profile::CompactV1,
        &completion,
        &receipt.spec,
        receipt.tile_count,
    );
    let mut near_loss = ProfileLossHasher::new(
        SaltV2Profile::NearLosslessV1,
        &completion,
        &receipt.spec,
        receipt.tile_count,
    );
    let mut nested = blake3::Hasher::new_derive_key(NESTED_ALLOCATION_CONTEXT);
    nested.update(completion.completion_id().as_bytes());
    nested.update(&completion.master_set_id());
    nested.update(receipt.spec.spec_id().as_bytes());
    nested.update(&receipt.tensor_count.to_le_bytes());
    nested.update(&receipt.tile_count.to_le_bytes());
    let mut global_tile = 0_usize;
    let mut compact_selected_planes = 0_u64;
    let mut near_selected_planes = 0_u64;
    for (ordinal, (expected, parent_master)) in parent
        .spec
        .expected_masters
        .iter()
        .zip(&parent_manifest.masters)
        .enumerate()
    {
        compact_map.begin_tensor(ordinal, expected)?;
        near_map.begin_tensor(ordinal, expected)?;
        compact_loss.begin_tensor(ordinal, expected)?;
        near_loss.begin_tensor(ordinal, expected)?;
        let ordinal_u64 = u64::try_from(ordinal)
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("selected tensor ordinal"))?;
        let name_len = u64::try_from(expected.name().len())
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("selected tensor name"))?;
        let tensor_tiles = u64::try_from(expected.tile_count())
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("selected tensor tiles"))?;
        nested.update(&ordinal_u64.to_le_bytes());
        nested.update(&name_len.to_le_bytes());
        nested.update(expected.name().as_bytes());
        nested.update(&tensor_tiles.to_le_bytes());
        let mut decoder =
            SaltV2MasterTensorDecoder::new(expected).map_err(Qwen36TensorWorkError::Master)?;
        parent
            .objects
            .try_visit_verified(
                &parent_master.record,
                super::MASTER_STREAM_CHUNK_BYTES,
                |chunk| {
                    decoder
                        .try_push(chunk, &mut |tile| {
                            let compact = *compact_counts.get(global_tile).ok_or(
                                Qwen36TensorWorkError::WorkspaceMismatch(
                                    "selected allocation test CompactV1 map",
                                ),
                            )?;
                            let near = *near_counts.get(global_tile).ok_or(
                                Qwen36TensorWorkError::WorkspaceMismatch(
                                    "selected allocation test NearLosslessV1 map",
                                ),
                            )?;
                            if compact == 0
                                || near == 0
                                || usize::from(compact) > tile.losses().len()
                                || usize::from(near) > tile.losses().len()
                            {
                                return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                                    "selected allocation test plane count",
                                ));
                            }
                            compact_map.push_count(compact);
                            near_map.push_count(near);
                            compact_loss.push_loss(tile.losses()[usize::from(compact - 1)]);
                            near_loss.push_loss(tile.losses()[usize::from(near - 1)]);
                            nested.update(&[compact, near]);
                            compact_selected_planes = compact_selected_planes
                                .checked_add(u64::from(compact))
                                .ok_or(Qwen36TensorWorkError::LengthOverflow(
                                    "selected CompactV1 plane count",
                                ))?;
                            near_selected_planes = near_selected_planes
                                .checked_add(u64::from(near))
                                .ok_or(Qwen36TensorWorkError::LengthOverflow(
                                    "selected NearLosslessV1 plane count",
                                ))?;
                            global_tile += 1;
                            Ok(())
                        })
                        .map_err(|error| match error {
                            SaltV2MasterVisitError::Master(error) => {
                                Qwen36TensorWorkError::Master(error)
                            }
                            SaltV2MasterVisitError::Visitor(error) => error,
                        })
                },
            )
            .map_err(|error| match error {
                TensorVisitError::Store(error) => Qwen36TensorWorkError::TensorStore(error),
                TensorVisitError::Sink(error) => error,
            })?;
        decoder.finish().map_err(Qwen36TensorWorkError::Master)?;
    }
    if global_tile != tile_count {
        return Err(Qwen36TensorWorkError::WorkspaceMismatch(
            "selected allocation test map coverage",
        ));
    }

    let mut compact_staged = StagedPackedMap::new(objects.temporary_dir(), "compact.test")?;
    let mut near_staged = StagedPackedMap::new(objects.temporary_dir(), "near.test")?;
    for count in compact_counts {
        compact_staged.push_count(*count)?;
    }
    for count in near_counts {
        near_staged.push_count(*count)?;
    }
    let map_bytes = packed_map_bytes(receipt.tile_count)?;
    compact_staged.seal(map_bytes)?;
    near_staged.seal(map_bytes)?;
    let compact_spec = allocation_map_record_spec_from_fields(
        receipt.parent_completion_id,
        receipt.base_workspace_id,
        receipt.campaign_id,
        receipt.master_set_id,
        receipt.source_model_id,
        &receipt.spec,
        SaltV2Profile::CompactV1,
        receipt.tensor_count,
        receipt.tile_count,
    )?;
    let near_spec = allocation_map_record_spec_from_fields(
        receipt.parent_completion_id,
        receipt.base_workspace_id,
        receipt.campaign_id,
        receipt.master_set_id,
        receipt.source_model_id,
        &receipt.spec,
        SaltV2Profile::NearLosslessV1,
        receipt.tensor_count,
        receipt.tile_count,
    )?;
    receipt.compact.map_record =
        put_staged_allocation_map(&objects, &compact_spec, &mut compact_staged)?;
    receipt.near_lossless.map_record =
        put_staged_allocation_map(&objects, &near_spec, &mut near_staged)?;
    receipt.compact.allocation_map_id = compact_map.finish();
    receipt.near_lossless.allocation_map_id = near_map.finish();
    receipt.compact.selected_loss_id = compact_loss.finish();
    receipt.near_lossless.selected_loss_id = near_loss.finish();
    receipt.compact.selected_planes = compact_selected_planes;
    receipt.near_lossless.selected_planes = near_selected_planes;
    receipt.nested_allocation_id =
        nested_override.unwrap_or_else(|| ContentId::from_digest(*nested.finalize().as_bytes()));
    let bytes = receipt.canonical_bytes()?;
    receipt.selection_id = ContentId::of_bytes(&bytes);
    fs::write(root.join(SELECTION_FILE), bytes)
        .map_err(|error| work_io("rewrite selected allocation test manifest", error))?;
    Ok(receipt)
}

#[cfg(unix)]
fn reclaim_selection_orphans(
    root: &Path,
    objects: &TensorWorkStore,
    expected_parent: Option<(&Qwen36CompleteWorkspaceReceipt, ContentId)>,
) -> Result<(), Qwen36TensorWorkError> {
    let manifest_path = root.join(SELECTION_FILE);
    let retained = match fs::symlink_metadata(&manifest_path) {
        Ok(_) => {
            let receipt = read_selection_receipt(&manifest_path)?;
            if let Some((completion, campaign_id)) = expected_parent {
                validate_receipt_binding(&receipt, completion, campaign_id)?;
            }
            validate_map_record_descriptors(&receipt)?;
            let compact = objects
                .open_verified(&receipt.compact.map_record)
                .map_err(Qwen36TensorWorkError::TensorStore)?;
            let near = objects
                .open_verified(&receipt.near_lossless.map_record)
                .map_err(Qwen36TensorWorkError::TensorStore)?;
            drop((compact, near));
            let mut retained = vec![
                receipt.compact.map_record.record_id(),
                receipt.near_lossless.map_record.record_id(),
            ];
            retained.extend(package_admission::admission_record_ids_if_present(
                root, objects,
            )?);
            retained
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => Vec::new(),
        Err(error) => return Err(work_io("inspect selected allocation manifest", error)),
    };
    let sweep = objects
        .prepare_unreferenced_scavenge(&retained)
        .map_err(Qwen36TensorWorkError::TensorStore)?;
    objects
        .commit_unreferenced_scavenge(sweep)
        .map_err(Qwen36TensorWorkError::TensorStore)
}

fn encode_profile_receipt(
    output: &mut Vec<u8>,
    profile: &Qwen36SelectedProfileReceipt,
) -> Result<(), Qwen36TensorWorkError> {
    output.extend_from_slice(profile.allocation_map_id.as_bytes());
    output.extend_from_slice(profile.selected_loss_id.as_bytes());
    output.extend_from_slice(&profile.selected_planes.to_le_bytes());
    let record = profile
        .map_record
        .canonical_bytes()
        .map_err(Qwen36TensorWorkError::TensorStore)?;
    let record_len = u32::try_from(record.len())
        .map_err(|_| Qwen36TensorWorkError::LengthOverflow("allocation map receipt"))?;
    output.extend_from_slice(&record_len.to_le_bytes());
    output.extend_from_slice(&record);
    Ok(())
}

fn encode_profile_budget(output: &mut Vec<u8>, budget: ProfileBudget) {
    encode_physical_bytes(output, budget.maximum);
    encode_byte_delta(output, budget.metadata);
}

fn encode_byte_delta(output: &mut Vec<u8>, delta: ByteDelta) {
    encode_physical_bytes(output, delta.declared);
    match delta.measured {
        Some(measured) => {
            output.push(1);
            output.extend_from_slice(&[0; 7]);
            encode_physical_bytes(output, measured);
        }
        None => {
            output.extend_from_slice(&[0; 8]);
            encode_physical_bytes(output, PhysicalBytes::ZERO);
        }
    }
}

fn encode_physical_bytes(output: &mut Vec<u8>, bytes: PhysicalBytes) {
    output.extend_from_slice(&bytes.serialized.to_le_bytes());
    output.extend_from_slice(&bytes.resident.to_le_bytes());
}

const fn bytes_componentwise_le(left: PhysicalBytes, right: PhysicalBytes) -> bool {
    left.serialized <= right.serialized && left.resident <= right.resident
}

fn codec_tag(codec: SaltV2Codec) -> Result<u8, Qwen36TensorWorkError> {
    match codec {
        SaltV2Codec::D2 => Ok(1),
        SaltV2Codec::B3 => Ok(2),
        SaltV2Codec::S34 => Ok(3),
        _ => Err(Qwen36TensorWorkError::WorkspaceMalformed(
            "selected allocation codec",
        )),
    }
}

fn codec_from_tag(tag: u8) -> Result<SaltV2Codec, Qwen36TensorWorkError> {
    match tag {
        1 => Ok(SaltV2Codec::D2),
        2 => Ok(SaltV2Codec::B3),
        3 => Ok(SaltV2Codec::S34),
        _ => Err(Qwen36TensorWorkError::WorkspaceMalformed(
            "selected allocation codec",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::packed_map_bytes;

    #[test]
    fn flagship_profile_map_size_is_exact_and_stays_out_of_small_manifests() {
        const QWEN36_ADDITIVE_TILES: u64 = 106_711_040;
        assert_eq!(packed_map_bytes(QWEN36_ADDITIVE_TILES).unwrap(), 26_677_760);
    }
}
