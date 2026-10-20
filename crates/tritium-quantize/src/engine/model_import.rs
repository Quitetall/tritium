//! Whole legacy matrix model caller; codecs/source/publication remain shared.
use super::model::platform;
use super::{
    DirectoryTensorSink, ObservedWeightSource, PublicationWarning, QuantEngineError,
    TritModelPublicationError, WeightSource,
};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
};
use tritium_core::DType;
use tritium_evidence::{
    EvidenceError, EvidenceEvent, EvidenceRecorder, LegacyModelImportBoundView,
    LegacyModelImportView, SealedEvidence, VerifiedLog,
};
use tritium_format::salt_v2_package::{
    SaltV2PackageReadError, SaltV2PackageReader, SaltV2Transform,
};
use tritium_format::trit_model::{
    TritModelReadError, TritModelReader, imported_manifest_cbor_size_bounds,
    tensor_meta_cbor_size_bounds, tensor_storage_cbor_bytes,
};
use tritium_format::{
    SaltTensorSpoolError, SaltTensorSpoolLimits, SaltTensorSpoolPlan, TritBlobKind, TritBlobSize,
    TritPackageReadLimits, TritPackageSizeBounds, TritSpoolBlob, import_salt_v2_tensor_to_spool,
    inspect_salt_v2_tensor_spool, trit_package_size_bounds,
};
use tritium_schema::*;

/// Explicit complete legacy import request, not a fitting or model quality plan.
#[derive(Clone, Debug)]
pub struct SaltV2ModelImportRequest {
    /// Single-level KeepDense plan for actual preserved tensors and all assets.
    /// Existing raw-source identity requires at least one companion tensor.
    pub companions: QuantPlan,
    /// Canonical raw D2/B3 target, not a newly admitted execution profile.
    pub codec: PlaneCodec,
    /// Explicit target scale group, subject to exact original scale-bit proof.
    pub group: u16,
    /// Existing per-tensor importer logical source/working/output limits.
    pub tensor_limits: SaltTensorSpoolLimits,
    /// Complete canonical tensor/dense/asset spool bytes, not aggregate heap/RSS.
    pub max_spool_bytes: u64,
    /// Complete typed package receiving/output caps, not aggregate heap/RSS.
    pub receiving: TritPackageReadLimits,
}

/// One strict-indexed matrix's read-only logical import requirements.
#[derive(Clone, Debug)]
pub struct SaltV2MatrixImportPlan {
    /// Original tensor name, never a renamed/fitted source.
    pub name: String,
    /// Actual target layout derived by the same model importer.
    pub layout: AdditiveLayout,
    /// Count/codec/index requirements; scale representability is not proven.
    pub spool: SaltTensorSpoolPlan,
}
/// Logical model spool planning, not import success, full container/RSS or qualification.
#[derive(Clone, Debug)]
pub struct SaltV2ModelImportPlan {
    /// All actual source matrices in canonical order.
    pub matrices: Vec<SaltV2MatrixImportPlan>,
    /// Canonical additive coefficient/scale payload sum.
    pub additive_payload_bytes: u64,
    /// Preserved dense payload sum; bytes are not read/proven by inspection.
    pub dense_payload_bytes: u64,
    /// Actual observed asset byte sum.
    pub asset_payload_bytes: u64,
    /// Exact logical raw spool payload sum, excluding container/log overhead.
    pub spool_payload_bytes: u64,
    /// Sum of exact storage fragments, excluding all other manifest fields.
    /// A lower bound only: passing it does not admit the full manifest/package.
    pub manifest_lower_bound_bytes: u64,
    /// Complete encoded-container bounds; not exact output, disk or RSS admission.
    pub container_bounds: TritPackageSizeBounds,
}

fn matrix_inventory<R: Read + Seek>(
    request: &SaltV2ModelImportRequest,
    legacy: &SaltV2PackageReader<R>,
) -> Result<Vec<(String, AdditiveLayout)>, SaltV2ModelImportError> {
    let plan = &request.companions;
    plan.validate().map_err(QuantEngineError::Plan)?;
    if plan.levels.len() != 1
        || plan
            .tensors
            .iter()
            .any(|t| t.action != QuantAction::KeepDense)
    {
        return Err(refusal("import companions must be one KeepDense level"));
    }
    if legacy.is_empty() {
        return Err(refusal("legacy matrix inventory is empty"));
    }
    let mut matrices = Vec::new();
    matrices
        .try_reserve_exact(legacy.len())
        .map_err(|_| refusal("import matrix inventory allocation"))?;
    for name in legacy.tensor_names() {
        if plan
            .tensors
            .binary_search_by(|t| t.source.name.as_str().cmp(name))
            .is_ok()
        {
            return Err(refusal("legacy and companion tensor names collide"));
        }
        let info = legacy
            .tensor_info(name)
            .ok_or_else(|| refusal("legacy metadata missing"))?;
        let [rows, cols] = info.dims() else {
            return Err(refusal("legacy model import requires rank-two matrices"));
        };
        if info.transform() != SaltV2Transform::None {
            return Err(refusal(
                "legacy model transform lacks an admitted exact basis",
            ));
        }
        let maximum = legacy
            .tensor_plane_counts(name)
            .map_err(SaltV2ModelImportError::Legacy)?
            .max()
            .ok_or_else(|| refusal("legacy counts absent"))?;
        let layout = AdditiveLayout {
            rows: *rows,
            cols: *cols,
            tile: 256,
            group: request.group,
            max_planes: u8::try_from(maximum).map_err(|_| refusal("legacy plane count"))?,
            allocation: PlaneAllocation::PerTile,
            codec: request.codec,
            law: ScaleLaw {
                anchor: ScaleAnchor::Group,
                relation: PlaneRelation::Free,
                precision: ScalePrecision::F16,
            },
            basis: Basis::Identity,
            transport: Transport::Raw,
        };
        layout.validate().map_err(|_| refusal("import layout"))?;
        matrices.push((name.to_owned(), layout));
    }
    Ok(matrices)
}

/// Inspect through shared strict metadata/count/encoder sizing before any writes.
/// Asset bytes are bounded observations; companion data bytes and target scale-bit
/// representability are not proved. This is not prepared-import evidence, source
/// official authentication, full container/RSS admission or publication permission.
/// Rejects a manifest receiving cap below the guaranteed storage-fragment bound.
/// # Errors
/// Invalid/mismatched inventory, unsupported layout, logical caps or source errors.
pub fn inspect_salt_v2_model_import<R: Read + Seek>(
    request: &SaltV2ModelImportRequest,
    legacy: &SaltV2PackageReader<R>,
    companions: &mut dyn WeightSource,
) -> Result<SaltV2ModelImportPlan, SaltV2ModelImportError> {
    let inventory = matrix_inventory(request, legacy)?;
    let plan = &request.companions;
    let actual = companions.inventory()?;
    if actual.len() != plan.tensors.len()
        || actual
            .iter()
            .zip(&plan.tensors)
            .any(|(a, b)| a != &b.source)
        || companions.asset_names() != plan.assets
    {
        return Err(refusal(
            "inspection companion inventory disagrees with plan",
        ));
    }
    let sum = |a: u64, b: u64| {
        a.checked_add(b)
            .ok_or_else(|| refusal("import planning payload overflow"))
    };
    let mut matrices = Vec::new();
    matrices
        .try_reserve_exact(inventory.len())
        .map_err(|_| refusal("inspection allocation"))?;
    let mut additive_payload_bytes = 0;
    let mut manifest_lower_bound_bytes = 0;
    let mut metadata_bounds = Vec::new();
    let mut blob_sizes = Vec::new();
    for (name, layout) in inventory {
        let spool = inspect_salt_v2_tensor_spool(legacy, &name, layout, request.tensor_limits)
            .map_err(SaltV2ModelImportError::Tensor)?;
        additive_payload_bytes = sum(additive_payload_bytes, spool.payload_bytes)?;
        manifest_lower_bound_bytes = sum(manifest_lower_bound_bytes, spool.storage_metadata_bytes)?;
        metadata_bounds.push(spool.tensor_metadata_bounds);
        blob_sizes.push(TritBlobSize {
            kind: TritBlobKind::AdditiveTensor,
            length: spool.payload_bytes,
        });
        matrices.push(SaltV2MatrixImportPlan {
            name,
            layout,
            spool,
        });
    }
    let mut dense_payload_bytes = 0;
    for t in &plan.tensors {
        let bytes = t.source.byte_len().map_err(QuantEngineError::Plan)?;
        if bytes > plan.limits.max_source_bytes {
            return Err(refusal("inspection dense source exceeds cap"));
        }
        dense_payload_bytes = sum(dense_payload_bytes, bytes)?;
        let storage = TensorStorage::Dense {
            shape: t.source.shape.clone(),
            precision: t.source.precision,
        };
        manifest_lower_bound_bytes = sum(
            manifest_lower_bound_bytes,
            tensor_storage_cbor_bytes(&storage)
                .map_err(|_| refusal("inspection storage metadata serialization"))?,
        )?;
        metadata_bounds.push(
            tensor_meta_cbor_size_bounds(t.source.name.clone(), storage)
                .map_err(|_| refusal("inspection tensor metadata serialization"))?,
        );
        blob_sizes.push(TritBlobSize {
            kind: TritBlobKind::DenseTensor(dense_dtype(t.source.precision)?),
            length: bytes,
        });
    }
    if manifest_lower_bound_bytes > request.receiving.max_manifest_bytes as u64 {
        return Err(refusal(
            "inspection manifest storage lower bound exceeds receiving cap (other manifest fields not covered)",
        ));
    }
    let mut asset_payload_bytes = 0;
    for name in &plan.assets {
        let bytes = companions.asset(
            name,
            platform(plan.limits.max_asset_bytes, "inspection asset cap")?,
        )?;
        if bytes.len() as u64 > plan.limits.max_asset_bytes {
            return Err(refusal("inspection asset exceeds cap"));
        }
        asset_payload_bytes = sum(asset_payload_bytes, bytes.len() as u64)?;
        blob_sizes.push(TritBlobSize {
            kind: TritBlobKind::Asset(name.clone()),
            length: bytes.len() as u64,
        });
    }
    let spool_payload_bytes = sum(
        sum(additive_payload_bytes, dense_payload_bytes)?,
        asset_payload_bytes,
    )?;
    if spool_payload_bytes > request.max_spool_bytes
        || spool_payload_bytes > request.receiving.max_source_bytes
        || spool_payload_bytes > plan.limits.max_retained_blob_bytes
    {
        return Err(refusal(
            "inspection aggregate raw spool payload exceeds cap (container overhead not covered)",
        ));
    }
    let manifest_bounds = imported_manifest_cbor_size_bounds(
        &plan.architecture,
        &plan.levels[0],
        &legacy.package_id().to_string(),
        &plan.assets,
        &metadata_bounds,
    )
    .map_err(|_| refusal("inspection full manifest sizing failed"))?;
    let container_bounds =
        trit_package_size_bounds(manifest_bounds, &blob_sizes).map_err(|_| {
            refusal("inspection full container sizing failed or exceeds global metadata caps")
        })?;
    if container_bounds.manifest.upper_bytes > request.receiving.max_manifest_bytes as u64 {
        return Err(refusal(
            "inspection full manifest upper bound exceeds receiving cap (actual digest byte widths unknown)",
        ));
    }
    if container_bounds.directory.upper_bytes > request.receiving.max_directory_bytes as u64 {
        return Err(refusal(
            "inspection directory upper bound exceeds receiving cap (deduplication unknown)",
        ));
    }
    if container_bounds.package.upper_bytes > request.receiving.max_source_bytes
        || container_bounds.package.upper_bytes > plan.limits.max_package_bytes
    {
        return Err(refusal(
            "inspection full package upper bound exceeds receiving cap (deduplication and alignment unknown)",
        ));
    }
    Ok(SaltV2ModelImportPlan {
        matrices,
        additive_payload_bytes,
        dense_payload_bytes,
        asset_payload_bytes,
        spool_payload_bytes,
        manifest_lower_bound_bytes,
        container_bounds,
    })
}

/// Published ordinary Imported model, original prepared log and explicit warnings.
#[derive(Debug)]
pub struct ImportedSaltV2Model {
    /// Retained admitted output File; no whole package/legacy master Vec required.
    pub model: TritModelReader<File>,
    /// Actual sealed prepared-input log, not a fit/publication/quality receipt.
    pub evidence: SealedEvidence,
    /// Existing successful-commit cleanup/durability caveats.
    pub warnings: Vec<PublicationWarning>,
}

/// Original whole-source error at the publisher's once-only terminal guard.
#[derive(Debug)]
#[non_exhaustive]
pub enum SaltV2ModelGuardError {
    /// Original strict legacy source changed or failed.
    Legacy(SaltV2PackageReadError),
    /// Original observed raw companion source changed or failed.
    Companion(QuantEngineError),
}
impl core::fmt::Display for SaltV2ModelGuardError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Legacy(e) => e.fmt(f),
            Self::Companion(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for SaltV2ModelGuardError {}

/// No new fit/source/published success follows from an import refusal.
#[derive(Debug)]
#[non_exhaustive]
pub enum SaltV2ModelImportError {
    /// Request, observed raw companion, metadata or logical cap refused.
    Companion(QuantEngineError),
    /// Existing tile-wise canonical legacy importer refused.
    Tensor(SaltTensorSpoolError),
    /// Original strict legacy metadata/count read failed.
    Legacy(SaltV2PackageReadError),
    /// Caller-owned spool I/O failed; an unpublished tail may remain.
    Io(std::io::Error),
    /// Actual prepared evidence registration/sealing/fold failed.
    Evidence(EvidenceError),
    /// Existing typed model/log replay refused.
    Replay(TritModelReadError),
    /// Shared typed staging or directory publication refused before commit.
    Publication(TritModelPublicationError<SaltV2ModelGuardError>),
}
impl core::fmt::Display for SaltV2ModelImportError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Companion(e) => e.fmt(f),
            Self::Tensor(e) => e.fmt(f),
            Self::Legacy(e) => e.fmt(f),
            Self::Io(e) => e.fmt(f),
            Self::Evidence(e) => e.fmt(f),
            Self::Replay(e) => e.fmt(f),
            Self::Publication(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for SaltV2ModelImportError {}
impl From<QuantEngineError> for SaltV2ModelImportError {
    fn from(e: QuantEngineError) -> Self {
        Self::Companion(e)
    }
}
impl From<std::io::Error> for SaltV2ModelImportError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<EvidenceError> for SaltV2ModelImportError {
    fn from(e: EvidenceError) -> Self {
        Self::Evidence(e)
    }
}

/// Import all actual legacy matrices plus observed companions through shared modules.
///
/// Derives shapes/counts/legacy ID from the original strict reader, preserves
/// exact dense precision/shape/assets and declares Imported, never Direct/prefix.
/// Refuses non-matrix ternary shapes rather than losing original axes. Requires
/// an empty caller-owned spool and fresh registered recorder. Fills one legacy
/// tensor at a time, retains metadata/counts, and reuses typed staging/publication.
/// Original strict legacy and observed companions are terminal-guarded once.
///
/// Caller owns immutable source/spool/parent custody, failed spool-tail cleanup,
/// source completeness against architecture/config, original official identity,
/// aggregate memory, bit/runtime/security/power-loss/distribution qualification.
/// Companion WeightSource can own one bounded raw dense tensor; encoded caps do
/// not cover that allocation, retained metadata, capacity slack or whole RSS.
/// # Errors
/// Invalid/unsupported request, changed source, caps/I/O/evidence/staging or
/// no-clobber commit failure. No alternate fitter/transform/provenance is inferred.
pub fn import_salt_v2_model<R: Read + Seek, S: Read + Write + Seek>(
    request: &SaltV2ModelImportRequest,
    legacy: &mut SaltV2PackageReader<R>,
    companions: &mut dyn WeightSource,
    spool: &mut S,
    destination: &mut DirectoryTensorSink,
    recorder: EvidenceRecorder,
) -> Result<ImportedSaltV2Model, SaltV2ModelImportError> {
    import_model(
        request,
        legacy,
        companions,
        spool,
        destination,
        ImportRecording {
            recorder,
            bound: false,
        },
    )
}

/// Same exact importer/publication with compact canonical-manifest-bound evidence.
/// Caller must register LegacyModelImportBoundPrepared; no fitted/source-quality
/// or whole-RSS claim is inferred. Embedded v1 remains the separate default API.
/// # Errors
/// The same source, geometry, caps, scale/trit, staging/guard/no-clobber errors,
/// plus strict compact evidence/actual canonical-manifest binding refusal.
pub fn import_salt_v2_model_bound<R: Read + Seek, S: Read + Write + Seek>(
    request: &SaltV2ModelImportRequest,
    legacy: &mut SaltV2PackageReader<R>,
    companions: &mut dyn WeightSource,
    spool: &mut S,
    destination: &mut DirectoryTensorSink,
    recorder: EvidenceRecorder,
) -> Result<ImportedSaltV2Model, SaltV2ModelImportError> {
    import_model(
        request,
        legacy,
        companions,
        spool,
        destination,
        ImportRecording {
            recorder,
            bound: true,
        },
    )
}

struct ImportRecording {
    recorder: EvidenceRecorder,
    bound: bool,
}
impl ImportRecording {
    fn check(&self) -> Result<(), SaltV2ModelImportError> {
        if !self.recorder.events().is_empty() {
            return Err(refusal("import needs a fresh recorder"));
        }
        if self.bound {
            self.recorder
                .check_emit::<LegacyModelImportBoundPrepared>("migration", None, 0)?;
        } else {
            self.recorder
                .check_emit::<LegacyModelImportPrepared>("migration", None, 0)?;
        }
        Ok(())
    }
    fn seal(
        mut self,
        manifest: ModelManifest,
        legacy_id: String,
        observed: &ObservedWeightSource<'_>,
    ) -> Result<(ModelManifest, SealedEvidence), SaltV2ModelImportError> {
        if self.bound {
            let preparation = LegacyModelImportBoundView::prepare(
                &manifest,
                legacy_id,
                observed.source_id(),
                observed.tensor_bindings().to_vec(),
                observed.asset_bindings().to_vec(),
            )?;
            self.recorder.emit("migration", None, 0, &preparation)?;
            let evidence = self.recorder.seal()?;
            let views = LegacyModelImportBoundView::fold(evidence.log(), &manifest)?;
            if views.len() != 1 || views[0].observation() != &preparation {
                return Err(refusal("bound prepared import log binding"));
            }
            Ok((manifest, evidence))
        } else {
            let preparation = LegacyModelImportPrepared {
                legacy_id,
                companion_source: observed.source_id(),
                companion_tensors: observed.tensor_bindings().to_vec(),
                companion_assets: observed.asset_bindings().to_vec(),
                manifest,
            };
            self.recorder.emit("migration", None, 0, &preparation)?;
            let evidence = self.recorder.seal()?;
            let views = LegacyModelImportView::fold(evidence.log())?;
            if views.len() != 1 || views[0].observation() != &preparation {
                return Err(refusal("prepared import log binding"));
            }
            Ok((preparation.manifest, evidence))
        }
    }
}

fn import_model<R: Read + Seek, S: Read + Write + Seek>(
    request: &SaltV2ModelImportRequest,
    legacy: &mut SaltV2PackageReader<R>,
    companions: &mut dyn WeightSource,
    spool: &mut S,
    destination: &mut DirectoryTensorSink,
    recording: ImportRecording,
) -> Result<ImportedSaltV2Model, SaltV2ModelImportError> {
    use SaltV2ModelImportError as E;
    let plan = &request.companions;
    plan.validate().map_err(QuantEngineError::Plan)?;
    if plan.levels.len() != 1
        || plan
            .tensors
            .iter()
            .any(|t| t.action != QuantAction::KeepDense)
    {
        return Err(refusal("import companions must be one KeepDense level"));
    }
    recording.check()?;
    if spool.seek(SeekFrom::End(0))? != 0 || spool.seek(SeekFrom::Start(0))? != 0 {
        return Err(refusal("model import spool must be empty"));
    }
    // Public library callers get the same pre-write admission as the CLI.
    // This lower-bound check never replaces exact staging/receiving validation.
    inspect_salt_v2_model_import(request, legacy, companions)?;
    let matrices = matrix_inventory(request, legacy)?;
    let mut observed = ObservedWeightSource::new(plan, companions)?;
    let mut blobs = Vec::new();
    let mut tensors = Vec::new();
    for (name, layout) in matrices {
        let mut limits = request.tensor_limits;
        limits.max_payload_bytes = limits
            .max_payload_bytes
            .min(remaining(spool, request.max_spool_bytes)?);
        let staged = import_salt_v2_tensor_to_spool(legacy, &name, layout, limits, spool)
            .map_err(E::Tensor)?;
        tensors.push(TensorMeta {
            name,
            blob: reference(staged.blob.id),
            storage: TensorStorage::Additive {
                layout,
                planes: PlaneCounts::PerTile(staged.tile_counts),
            },
        });
        blobs.push(staged.blob);
    }
    for tensor in &plan.tensors {
        let source = &tensor.source;
        let length = source.byte_len().map_err(QuantEngineError::Plan)?;
        let bytes = observed.tensor(&source.name, platform(length, "import companion bytes")?)?;
        let dtype = dense_dtype(source.precision)?;
        let blob = append(
            spool,
            TritBlobKind::DenseTensor(dtype),
            &bytes,
            request.max_spool_bytes,
        )?;
        tensors.push(TensorMeta {
            name: source.name.clone(),
            blob: reference(blob.id),
            storage: TensorStorage::Dense {
                shape: source.shape.clone(),
                precision: source.precision,
            },
        });
        blobs.push(blob);
    }
    let mut assets = Vec::new();
    for name in &plan.assets {
        let bytes = observed.asset(
            name,
            platform(plan.limits.max_asset_bytes, "import asset bytes")?,
        )?;
        let blob = append(
            spool,
            TritBlobKind::Asset(name.clone()),
            &bytes,
            request.max_spool_bytes,
        )?;
        assets.push(ManifestAsset {
            name: name.clone(),
            blob: reference(blob.id),
        });
        blobs.push(blob);
    }
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    let level = plan.levels[0].clone();
    let legacy_id = legacy.package_id().to_string();
    let manifest = ModelManifest::imported(
        plan.architecture.clone(),
        level,
        legacy_id.clone(),
        assets,
        tensors,
    );
    let (manifest, evidence) = recording.seal(manifest, legacy_id, &observed)?;
    let (model, warnings) = destination
        .publish_spooled_model(
            &manifest,
            &blobs,
            spool,
            request.receiving,
            &evidence,
            || {
                observed
                    .revalidate()
                    .map_err(SaltV2ModelGuardError::Companion)?;
                // Original legacy check follows every companion read as well.
                legacy
                    .verify_unchanged()
                    .map_err(SaltV2ModelGuardError::Legacy)
            },
        )
        .map_err(E::Publication)?;
    Ok(ImportedSaltV2Model {
        model,
        evidence,
        warnings,
    })
}

fn dense_dtype(precision: ScalePrecision) -> Result<DType, SaltV2ModelImportError> {
    match precision {
        ScalePrecision::F16 => Ok(DType::F16),
        ScalePrecision::Bf16 => Ok(DType::BF16),
        ScalePrecision::F32 => Ok(DType::F32),
        _ => Err(refusal("import dense precision")),
    }
}

/// Reverify an actual retained package against recorded import preparation.
/// No independent original-source replay, fit or release verdict is inferred.
/// # Errors
/// Missing/ambiguous/mismatched import log, inventory/payload or terminal identity.
pub fn verify_imported_model_with_log<R: Read + Seek>(
    model: &mut TritModelReader<R>,
    log: &VerifiedLog,
) -> Result<LegacyModelImportView, SaltV2ModelImportError> {
    match verify_imported_model_evidence(model, log)? {
        ImportedModelEvidenceView::Embedded(view) => Ok(view),
        ImportedModelEvidenceView::ManifestBound(_) => {
            Err(refusal("embedded v1 import preparation required"))
        }
    }
}

/// Verified supported preparation view, not a public-release/source-truth verdict.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ImportedModelEvidenceView {
    /// Original embedded-manifest v1 observation.
    Embedded(LegacyModelImportView),
    /// Compact actual-canonical-manifest-bound observation.
    ManifestBound(LegacyModelImportBoundView),
}
impl ImportedModelEvidenceView {
    /// Actual recorded original legacy identity (its source truth remains external).
    #[must_use]
    pub fn legacy_id(&self) -> &str {
        match self {
            Self::Embedded(v) => &v.observation().legacy_id,
            Self::ManifestBound(v) => &v.observation().legacy_id,
        }
    }
}

/// Cold verify either supported preparation with complete payload and exact bytes.
/// Requires exactly one preparation across both protocols; mixed/duplicate logs
/// are rejected, not selected. This does not authenticate original source/quality.
/// # Errors
/// Corrupt/ambiguous/version-mismatched evidence, full manifest/binding mismatch,
/// missing or invalid payload inventory or terminal exact package identity failure.
pub fn verify_imported_model_evidence<R: Read + Seek>(
    model: &mut TritModelReader<R>,
    log: &VerifiedLog,
) -> Result<ImportedModelEvidenceView, SaltV2ModelImportError> {
    let preparations: Vec<_> = log
        .events
        .iter()
        .filter(|e| {
            e.schema == LegacyModelImportPrepared::SCHEMA_ID
                || e.schema == LegacyModelImportBoundPrepared::SCHEMA_ID
        })
        .collect();
    if preparations.len() != 1 {
        return Err(refusal("exactly one imported preparation required"));
    }
    let view = if preparations[0].schema == LegacyModelImportPrepared::SCHEMA_ID {
        let mut views = LegacyModelImportView::fold(log)?;
        let view = views.remove(0);
        if &view.observation().manifest != model.manifest() {
            return Err(refusal("import log/model manifest mismatch"));
        }
        ImportedModelEvidenceView::Embedded(view)
    } else {
        let mut views = LegacyModelImportBoundView::fold(log, model.manifest())?;
        ImportedModelEvidenceView::ManifestBound(views.remove(0))
    };
    model
        .validate_complete_payload_inventory()
        .map_err(QuantEngineError::Model)?;
    model.verify().map_err(SaltV2ModelImportError::Replay)?;
    Ok(view)
}
fn refusal(detail: &'static str) -> SaltV2ModelImportError {
    QuantEngineError::Binding(detail).into()
}
fn reference(id: BlobId) -> BlobReference {
    BlobReference { id, shard: None }
}
fn remaining<S: Seek>(spool: &mut S, cap: u64) -> Result<u64, SaltV2ModelImportError> {
    let position = spool.stream_position()?;
    if spool.seek(SeekFrom::End(0))? != position {
        return Err(refusal("import spool cursor must be at its end"));
    }
    cap.checked_sub(position)
        .ok_or_else(|| refusal("complete import spool cap"))
}
fn append<S: Write + Seek>(
    spool: &mut S,
    kind: TritBlobKind,
    bytes: &[u8],
    cap: u64,
) -> Result<TritSpoolBlob, SaltV2ModelImportError> {
    let left = remaining(spool, cap)?;
    if bytes.len() as u64 > left {
        return Err(refusal("complete import spool cap"));
    }
    let blob = TritSpoolBlob {
        kind,
        id: BlobId::from_bytes(*blake3::hash(bytes).as_bytes()),
        offset: spool.stream_position()?,
        length: bytes.len() as u64,
    };
    for chunk in bytes.chunks(64 * 1024) {
        spool.write_all(chunk)?;
    }
    Ok(blob)
}
