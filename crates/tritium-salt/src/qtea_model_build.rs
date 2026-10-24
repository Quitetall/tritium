//! Complete unpublished QTEA cohorts from observed bytes and exact admitted work.
mod verify;
use crate::{
    AdmittedSolverPlan, ContentId, FrontierStageRequest, QteaCalibratedTensorInput,
    QteaTensorOrigin, QteaTensorWorkPlan,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    io::{Read, Seek, SeekFrom, Write},
};
use tritium_format::{
    TritBlobKind, TritPackageReadLimits, TritSpoolBlob, inspect_trit_spool_layout,
    qtea::model::{
        QTEA_HYBRID_MODEL_PROFILE, QTEA_MODEL_PROFILE, QteaHybridModelReadLimits,
        QteaModelManifest, QteaModelReadLimits, QteaModelReader, QteaModelTensor,
        encode_qtea_hybrid_model_manifest, encode_qtea_model_manifest,
    },
    trit_model::DensePayloadDecoder,
    write_trit_package_from_spool,
};
use tritium_nn::{QteaModelInventory, QteaModelProfile};
use tritium_quantize::{
    InputGram,
    engine::{CalibratedTensor, ObservedWeightSource, WeightSource},
    qtea_reference::{TensorFit, TensorFitConfig},
};
use tritium_schema::{
    BlobId, NamedAssetDigest, OriginalSourceId, QuantSourceBinding, ScalePrecision,
};
pub use verify::verify_qtea_model_build_record;

/// Separate positive caller ceilings, not total process/model/cache/RSS admission.
#[derive(Clone, Copy, Debug)]
pub struct QteaModelBuildLimits {
    /// Complete output receiving caps.
    pub receiving: TritPackageReadLimits,
    /// Existing source config/assets/owners/vector/QTEA limits.
    pub model: QteaModelReadLimits,
    /// Positive higher-rank preserved elements; standard profile cannot use them.
    pub max_preserved_tensor_elements: usize,
    /// Maximum actual capacity bytes of one widened original matrix.
    pub max_widened_tensor_bytes: usize,
    /// Maximum native tensor fit logical bytes.
    pub max_native_work_bytes: usize,
    /// Existing borrowed-array/native/export combined logical scope per tensor.
    pub max_tensor_work_bytes: usize,
    /// Private spool payload cap, excluding metadata/output/observation residency.
    pub max_spool_bytes: u64,
    /// Maximum encoded generated fit-observation asset bytes/capacity.
    pub max_fit_record_bytes: usize,
}
/// Actual curvature supplied by the caller; these paths are never relabelled.
pub enum QteaOwnerCurvature<'a> {
    /// Existing actual Full/Matmul/input-only collector result, with its refusals.
    Calibrated(&'a CalibratedTensor),
    /// Explicit native bound Gram, not an automatic identity/capture fallback.
    Native(&'a InputGram),
}
impl std::fmt::Debug for QteaOwnerCurvature<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Calibrated(_) => f.write_str("Calibrated(<borrowed immutable capture>)"),
            Self::Native(g) => f
                .debug_struct("Native")
                .field("dimension", &g.dimension())
                .finish_non_exhaustive(),
        }
    }
}
/// Existing selected admission; expected ID must come from the caller's selection.
#[derive(Clone, Debug)]
pub struct QteaSelectedAdmission {
    /// Opaque already-admitted exact solver/machine/evidence token.
    pub plan: AdmittedSolverPlan,
    /// Exact matching worker request.
    pub request: FrontierStageRequest,
    /// Separately selected expected plan identity, not inferred by this builder.
    pub expected_plan_id: ContentId,
}
/// Actual fit observation, not measured resources or a completion/qualification receipt.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QteaOwnerFitObservation {
    name: String,
    source_tensor_index: u32,
    curvature: String,
    input_id: ContentId,
    plan_id: ContentId,
    request: FrontierStageRequest,
    estimate: crate::ResourceVector,
    machine_id: ContentId,
    evidence_id: ContentId,
    output_id: ContentId,
    tensor_id: ContentId,
    blob_id: BlobId,
    recipe_digest: [u8; 32],
    weights_digest: [u8; 32],
    gram_digest: [u8; 32],
    activation_cache: [u8; 32],
    token_stream: [u8; 32],
}
/// Bounded data asset bound by the model package; not authenticated execution.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QteaModelBuildRecord {
    schema: String,
    qualification: bool,
    campaign: ContentId,
    source_model: OriginalSourceId,
    architecture: String,
    source_tensors: Vec<QuantSourceBinding>,
    source_assets: Vec<NamedAssetDigest>,
    fits: Vec<QteaOwnerFitObservation>,
}
impl QteaModelBuildRecord {
    /// Actual number of completed native matrix work observations.
    pub fn fitted_matrices(&self) -> usize {
        self.fits.len()
    }
    /// Original raw source content identity, not original container custody.
    pub const fn source_model(&self) -> OriginalSourceId {
        self.source_model
    }
}
/// Successful private stream construction; not durable filesystem publication.
#[derive(Debug)]
pub struct QteaModelBuildResult {
    package_id: tritium_format::PackageId,
    record: QteaModelBuildRecord,
}
impl QteaModelBuildResult {
    /// Actual emitted, reopened and terminally checked complete bytes identity.
    pub const fn package_id(&self) -> tritium_format::PackageId {
        self.package_id
    }
    /// Actual bound software observations; no measured ResourceVector fabricated.
    pub fn record(&self) -> &QteaModelBuildRecord {
        &self.record
    }
}
/// Refusal may leave private partial streams, but never a successful build result.
#[derive(Debug)]
pub struct QteaModelBuildError {
    reason: String,
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
}
impl std::fmt::Display for QteaModelBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.reason)
    }
}
impl std::error::Error for QteaModelBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_ref()
            .map(|e| e.as_ref() as &(dyn std::error::Error + 'static))
    }
}
impl QteaModelBuildError {
    /// Preserve a caller selection refusal without inventing an admission or
    /// requiring the caller to discard its original typed diagnostic.
    pub fn selection(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        cause(error)
    }
}
fn fail(e: impl std::fmt::Display) -> QteaModelBuildError {
    QteaModelBuildError {
        reason: e.to_string(),
        source: None,
    }
}
fn cause(e: impl std::error::Error + Send + Sync + 'static) -> QteaModelBuildError {
    QteaModelBuildError {
        reason: format!("QTEA model build: {e}"),
        source: Some(Box::new(e)),
    }
}

/// One poisoned-on-error unpublished cohort. Never owns a filesystem destination.
/// Caller owns immutable external-source/spool custody and discards failed streams.
pub struct QteaModelBuilder<'a, S> {
    source: ObservedWeightSource<'a>,
    inventory: QteaModelInventory,
    spool: S,
    limits: QteaModelBuildLimits,
    manifest: QteaModelManifest,
    blobs: Vec<TritSpoolBlob>,
    record: QteaModelBuildRecord,
    poisoned: bool,
}
/// One freshly verified and widened owner, dropped after inspection or fitting.
/// Never retained across the inspect/execute source revalidation boundary.
struct PreparedOwner {
    binding: QuantSourceBinding,
    index: u32,
    weights: Vec<f32>,
}
impl PreparedOwner {
    fn work<'a>(
        &'a self,
        curvature: QteaOwnerCurvature<'a>,
        config: TensorFitConfig,
        source_model: OriginalSourceId,
        campaign: ContentId,
        limits: QteaModelBuildLimits,
    ) -> Result<(QteaTensorWorkPlan<'a>, &'static str), QteaModelBuildError> {
        let origin = QteaTensorOrigin {
            source_id: campaign,
            owner: &self.binding.source.name,
            tensor_index: self.index,
            source_tensor_digest: *self.binding.content.as_bytes(),
        };
        match curvature {
            QteaOwnerCurvature::Calibrated(calibration) => Ok((
                QteaTensorWorkPlan::from_calibrated(
                    QteaCalibratedTensorInput {
                        original: &self.binding.source,
                        source_model,
                        weights: &self.weights,
                        calibration,
                        origin,
                    },
                    config,
                    limits.model.tensor,
                    limits.max_native_work_bytes,
                    limits.max_tensor_work_bytes,
                )
                .map_err(cause)?,
                "captured_full_matmul",
            )),
            QteaOwnerCurvature::Native(gram) => {
                if gram
                    .source_id()
                    .is_none_or(|s| s.source_model_digest() != *source_model.as_bytes())
                {
                    return Err(fail("native Gram source-model mismatch"));
                }
                let numerical = TensorFit::plan(
                    &self.weights,
                    usize::try_from(self.binding.source.shape[0]).map_err(cause)?,
                    usize::try_from(self.binding.source.shape[1]).map_err(cause)?,
                    gram,
                    config,
                    limits.max_native_work_bytes,
                )
                .map_err(cause)?;
                Ok((
                    QteaTensorWorkPlan::new(
                        numerical,
                        origin,
                        limits.model.tensor,
                        limits.max_tensor_work_bytes,
                    )
                    .map_err(cause)?,
                    "native_supplied",
                ))
            }
        }
    }
}
impl<S> std::fmt::Debug for QteaModelBuilder<'_, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QteaModelBuilder")
            .field("source", &self.record.source_model)
            .field("fits", &self.record.fits.len())
            .field("poisoned", &self.poisoned)
            .finish_non_exhaustive()
    }
}
impl<'a, S: Read + Write + Seek> QteaModelBuilder<'a, S> {
    /// Build and commit an explicitly unqualified candidate through the shared
    /// no-clobber directory transaction. Source/numeric/record checks in finish
    /// precede the sink's independent actual-byte admission and commit marker.
    /// Does not create evidence.jsonl, qualify MTP/model/resources/security or
    /// authorize distribution. Post-commit warnings are returned, not rollback.
    /// # Errors
    /// Original build/validation/no-clobber/pre-commit I/O refusal; owned pending
    /// paths are cleaned without recursively deleting unexpected foreign files.
    pub fn publish_candidate(
        self,
        sink: &mut tritium_quantize::engine::DirectoryTensorSink,
    ) -> Result<
        (
            QteaModelBuildResult,
            Vec<tritium_quantize::engine::PublicationWarning>,
        ),
        tritium_quantize::engine::CandidatePublicationError<QteaModelBuildError>,
    > {
        let limits = self.limits.receiving;
        sink.publish_unqualified_candidate(limits, |file| {
            let built = self.finish(file)?;
            let id = built.package_id();
            Ok((built, id))
        })
    }
    /// Prepare complete actual observed source metadata before fitting/spool writes.
    /// No source lock, machine estimate, dense model or publication is created.
    /// # Errors
    /// Invalid positive caps/campaign, nonempty spool, reserved asset, source/config/
    /// architecture/shape disagreement. Observer construction has its own raw limits.
    pub fn new(
        mut source: ObservedWeightSource<'a>,
        profile: QteaModelProfile,
        campaign: ContentId,
        mut spool: S,
        limits: QteaModelBuildLimits,
    ) -> Result<Self, QteaModelBuildError> {
        if campaign.as_bytes() == &[0; 32]
            || limits.max_spool_bytes == 0
            || [
                limits.model.max_tensors,
                limits.model.max_assets,
                limits.model.max_asset_bytes,
                limits.model.max_config_bytes,
                limits.model.max_vector_elements,
                limits.model.tensor.max_tensor_bytes,
                limits.model.tensor.max_validation_bytes,
                limits.max_preserved_tensor_elements,
                limits.max_widened_tensor_bytes,
                limits.max_native_work_bytes,
                limits.max_tensor_work_bytes,
                limits.max_fit_record_bytes,
                limits.receiving.max_manifest_bytes,
                limits.receiving.max_directory_bytes,
            ]
            .contains(&0)
            || limits.receiving.max_source_bytes == 0
        {
            return Err(fail(
                "QTEA build requires positive explicit limits and campaign identity",
            ));
        }
        if spool.seek(SeekFrom::End(0)).map_err(cause)? != 0 {
            return Err(fail("QTEA spool must be empty"));
        }
        if source.tensor_bindings().len() > limits.model.max_tensors
            || source.asset_bindings().len() > limits.model.max_assets
        {
            return Err(fail("QTEA original metadata count cap exceeded"));
        }
        let assets = source.asset_names();
        if assets.iter().any(|n| n == "qtea-fit.json")
            || !assets.iter().any(|n| n == "config.json")
            || assets.len() > limits.model.max_assets
        {
            return Err(fail(
                "QTEA source asset inventory/reserved fit asset mismatch",
            ));
        }
        let config = source
            .asset("config.json", limits.model.max_config_bytes)
            .map_err(cause)?
            .into_owned();
        let shapes = source
            .tensor_bindings()
            .iter()
            .map(|t| (t.source.name.clone(), t.source.shape.clone()))
            .collect();
        let inventory =
            QteaModelInventory::prepare(&config, profile, shapes, limits.model.max_tensors)
                .map_err(cause)?;
        if inventory.architecture_name() != source.architecture() {
            return Err(fail("observed source/config architecture mismatch"));
        }
        let source_model = source.source_id();
        let record = QteaModelBuildRecord {
            schema: "tritium.qtea.model-build-observation.v1".into(),
            qualification: false,
            campaign,
            source_model,
            architecture: source.architecture().into(),
            source_tensors: source.tensor_bindings().to_vec(),
            source_assets: source.asset_bindings().to_vec(),
            fits: vec![],
        };
        let manifest = QteaModelManifest {
            profile: match profile {
                QteaModelProfile::Standard => QTEA_MODEL_PROFILE,
                QteaModelProfile::Qwen35Hybrid => QTEA_HYBRID_MODEL_PROFILE,
            }
            .into(),
            version: 1,
            source_model: *source_model.as_bytes(),
            config: BlobId::from_bytes([0; 32]),
            assets: BTreeMap::new(),
            tensors: BTreeMap::new(),
        };
        Ok(Self {
            source,
            inventory,
            spool,
            limits,
            manifest,
            blobs: vec![],
            record,
            poisoned: false,
        })
    }
    /// Inspect actual owner work without fitting, spool access or owner insertion.
    /// The callback can select admission or copy planning data, but only borrows
    /// the plan. Widened weights are dropped on return, never cached for execution.
    /// Errors do not poison this cohort. Already fitted/poisoned owners are refused.
    ///
    /// Not SPD/fit success, aggregate spool availability, machine/physical admission
    /// or source authentication. A later fit rereads the source and reconstructs
    /// work; the caller must bind its selected request to that exact work identity.
    /// # Errors
    /// Source/owner/curvature, allocation and numerical/work cap refusals shared
    /// with fitting, or the caller's inspection error. No spool operation occurs.
    pub fn inspect_owner<R>(
        &mut self,
        name: &str,
        curvature: QteaOwnerCurvature<'_>,
        config: TensorFitConfig,
        inspect: impl FnOnce(&QteaTensorWorkPlan<'_>) -> Result<R, QteaModelBuildError>,
    ) -> Result<R, QteaModelBuildError> {
        if self.poisoned {
            return Err(fail("QTEA cohort is poisoned"));
        }
        let prepared = self.prepare_owner(name)?;
        let (work, _) = prepared.work(
            curvature,
            config,
            self.record.source_model,
            self.record.campaign,
            self.limits,
        )?;
        inspect(&work)
    }
    /// Fit one canonical matrix from its actual observed raw bytes and explicit
    /// curvature. Runs the existing exact admission before the native fitter.
    /// Any refusal poisons this cohort; no retry/fallback or measured receipt.
    /// # Errors
    /// Wrong/duplicate owner/source/curvature, caps/selection/admission/native,
    /// codec/partial spool write. Failure may leave private bytes only.
    pub fn fit_owner(
        &mut self,
        name: &str,
        curvature: QteaOwnerCurvature<'_>,
        config: TensorFitConfig,
        select: impl FnOnce(
            &QteaTensorWorkPlan<'_>,
        ) -> Result<QteaSelectedAdmission, QteaModelBuildError>,
    ) -> Result<(), QteaModelBuildError> {
        if self.poisoned {
            return Err(fail("QTEA cohort is poisoned"));
        }
        self.poisoned = true;
        let prepared = self.prepare_owner(name)?;
        let (work, kind) = prepared.work(
            curvature,
            config,
            self.record.source_model,
            self.record.campaign,
            self.limits,
        )?;
        self.fit_planned(work, kind, select)?;
        self.poisoned = false;
        Ok(())
    }
    fn prepare_owner(&mut self, name: &str) -> Result<PreparedOwner, QteaModelBuildError> {
        let (index, binding) = self
            .record
            .source_tensors
            .iter()
            .enumerate()
            .find(|(_, b)| b.source.name == name)
            .ok_or_else(|| fail("unknown QTEA owner"))?;
        let binding = binding.clone();
        let [rows, cols] = binding.source.shape.as_slice() else {
            return Err(fail("only matrix owners can be fitted"));
        };
        if self.manifest.tensors.contains_key(name) {
            return Err(fail("duplicate QTEA owner fit"));
        }
        let count = usize::try_from(
            rows.checked_mul(*cols)
                .ok_or_else(|| fail("source shape overflow"))?,
        )
        .map_err(cause)?;
        if count
            .checked_mul(4)
            .is_none_or(|n| n > self.limits.max_widened_tensor_bytes)
        {
            return Err(fail("widened source byte cap exceeded"));
        }
        let mut weights = Vec::new();
        weights.try_reserve_exact(count).map_err(cause)?;
        if weights
            .capacity()
            .checked_mul(4)
            .is_none_or(|n| n > self.limits.max_widened_tensor_bytes)
        {
            return Err(fail("widened source capacity cap exceeded"));
        }
        weights.resize(count, 0.);
        {
            let bytes =
                usize::try_from(binding.source.byte_len().map_err(cause)?).map_err(cause)?;
            let raw = self.source.tensor(name, bytes).map_err(cause)?;
            let mut decoder =
                DensePayloadDecoder::new(binding.source.precision, &mut weights).map_err(cause)?;
            decoder.push(&raw).map_err(cause)?;
            decoder.finish().map_err(cause)?;
        }
        Ok(PreparedOwner {
            binding,
            index: u32::try_from(index).map_err(cause)?,
            weights,
        })
    }
    fn fit_planned(
        &mut self,
        work: QteaTensorWorkPlan<'_>,
        kind: &'static str,
        select: impl FnOnce(
            &QteaTensorWorkPlan<'_>,
        ) -> Result<QteaSelectedAdmission, QteaModelBuildError>,
    ) -> Result<(), QteaModelBuildError> {
        let origin = work.origin();
        let name = origin.owner;
        let metadata = work.metadata();
        let offset = self.spool.seek(SeekFrom::End(0)).map_err(cause)?;
        let available = self
            .limits
            .max_spool_bytes
            .checked_sub(offset)
            .ok_or_else(|| fail("QTEA spool cap exceeded"))?;
        if work.byte_ledger().total_bytes as u64 > available {
            return Err(fail("QTEA planned tensor exceeds remaining spool cap"));
        }
        let selection = select(&work)?;
        let admitted = work
            .admit(
                &selection.plan,
                &selection.request,
                selection.expected_plan_id,
            )
            .map_err(cause)?;
        let mut writer = SpoolWriter::new(&mut self.spool, available);
        let fitted = admitted.fit_to(&mut writer).map_err(cause)?;
        let bytes = writer.bytes;
        let blob = BlobId::from_bytes(*writer.hash.finalize().as_bytes());
        let result = fitted.work_result();
        if bytes != result.byte_ledger().total_bytes as u64
            || ContentId::from_digest(*writer.content.finalize().as_bytes()) != result.tensor_id()
        {
            return Err(fail("native fit/spool byte ledger mismatch"));
        }
        let meta = result.metadata();
        self.manifest.tensors.insert(
            name.into(),
            QteaModelTensor::Qtea {
                blob,
                rows: metadata.rows as u64,
                columns: metadata.columns as u64,
            },
        );
        self.blobs.push(TritSpoolBlob {
            kind: TritBlobKind::QteaTensor,
            id: blob,
            offset,
            length: bytes,
        });
        self.record.fits.push(QteaOwnerFitObservation {
            name: name.into(),
            source_tensor_index: origin.tensor_index,
            curvature: kind.into(),
            input_id: result.input_id(),
            plan_id: fitted.plan_id(),
            request: selection.request,
            estimate: fitted.resource_estimate().resources(),
            machine_id: fitted.resource_estimate().machine_id(),
            evidence_id: fitted.resource_estimate().evidence_id(),
            output_id: fitted.output_id(),
            tensor_id: result.tensor_id(),
            blob_id: blob,
            recipe_digest: meta.recipe.digest(),
            weights_digest: meta.bindings.weights,
            gram_digest: meta.bindings.gram,
            activation_cache: meta.bindings.activation_cache,
            token_stream: meta.bindings.token_stream,
        });
        Ok(())
    }
    fn append(&mut self, kind: TritBlobKind, bytes: &[u8]) -> Result<BlobId, QteaModelBuildError> {
        let offset = self.spool.seek(SeekFrom::End(0)).map_err(cause)?;
        let available = self
            .limits
            .max_spool_bytes
            .checked_sub(offset)
            .ok_or_else(|| fail("QTEA spool cap exceeded"))?;
        let mut writer = SpoolWriter::new(&mut self.spool, available);
        writer.write_all(bytes).map_err(cause)?;
        let id = BlobId::from_bytes(*writer.hash.finalize().as_bytes());
        self.blobs.push(TritSpoolBlob {
            kind,
            id,
            offset,
            length: writer.bytes,
        });
        Ok(id)
    }
    /// Complete and verify a private output stream; not no-clobber/fsync publication.
    /// Every matrix must already have exactly one successful admitted fit. Original
    /// preserved/asset bytes are copied unchanged; source revalidation and final
    /// output verification precede success. Caller discards partial streams on error.
    /// # Errors
    /// Poisoned/incomplete cohort, nonempty output, original mutation, caps,
    /// encoding/I/O or reopened payload/identity mismatch. No successful partial result.
    pub fn finish(
        mut self,
        output: &mut (impl Read + Write + Seek),
    ) -> Result<QteaModelBuildResult, QteaModelBuildError> {
        if self.poisoned {
            return Err(fail("QTEA cohort is poisoned"));
        }
        if output.seek(SeekFrom::End(0)).map_err(cause)? != 0 {
            return Err(fail("QTEA output must be empty"));
        }
        for (name, shape) in self.inventory.tensors() {
            if shape.len() == 2 && !self.manifest.tensors.contains_key(name) {
                return Err(fail("QTEA model has unfitted matrix owners"));
            }
        }
        for binding in self.record.source_tensors.clone() {
            let shape = &binding.source.shape;
            if shape.len() == 2 {
                continue;
            }
            let count = shape.iter().try_fold(1_u64, |n, d| {
                n.checked_mul(*d)
                    .ok_or_else(|| fail("preserved shape overflow"))
            })?;
            let cap = if shape.len() == 1 {
                self.limits.model.max_vector_elements
            } else {
                self.limits.max_preserved_tensor_elements
            };
            if count > cap as u64 {
                return Err(fail("preserved state cap exceeded"));
            }
            let max = usize::try_from(binding.source.byte_len().map_err(cause)?).map_err(cause)?;
            let raw = self
                .source
                .tensor(&binding.source.name, max)
                .map_err(cause)?
                .into_owned();
            let dtype = match binding.source.precision {
                ScalePrecision::F16 => tritium_core::DType::F16,
                ScalePrecision::Bf16 => tritium_core::DType::BF16,
                ScalePrecision::F32 => tritium_core::DType::F32,
                _ => return Err(fail("unsupported preserved precision")),
            };
            let blob = self.append(TritBlobKind::DenseTensor(dtype), &raw)?;
            let tensor = if shape.len() == 1 {
                QteaModelTensor::Vector {
                    blob,
                    length: count,
                    precision: binding.source.precision,
                }
            } else {
                QteaModelTensor::Preserved {
                    blob,
                    shape: shape.clone(),
                    precision: binding.source.precision,
                }
            };
            self.manifest.tensors.insert(binding.source.name, tensor);
        }
        for name in self.source.asset_names() {
            let cap = if name == "config.json" {
                self.limits.model.max_config_bytes
            } else {
                self.limits.model.max_asset_bytes
            };
            let bytes = self.source.asset(&name, cap).map_err(cause)?.into_owned();
            let blob = self.append(TritBlobKind::Asset(name.clone()), &bytes)?;
            if name == "config.json" {
                self.manifest.config = blob;
            } else {
                self.manifest.assets.insert(name, blob);
            }
        }
        let mut record = RecordWriter {
            bytes: vec![],
            cap: self
                .limits
                .max_fit_record_bytes
                .min(self.limits.model.max_asset_bytes),
        };
        serde_json::to_writer(&mut record, &self.record).map_err(cause)?;
        let blob = self.append(TritBlobKind::Asset("qtea-fit.json".into()), &record.bytes)?;
        self.manifest.assets.insert("qtea-fit.json".into(), blob);
        let metadata = match self.inventory.profile() {
            QteaModelProfile::Standard => {
                encode_qtea_model_manifest(&self.manifest, self.limits.receiving.max_manifest_bytes)
            }
            QteaModelProfile::Qwen35Hybrid => encode_qtea_hybrid_model_manifest(
                &self.manifest,
                self.limits.receiving.max_manifest_bytes,
            ),
        }
        .map_err(cause)?;
        let size = inspect_trit_spool_layout(&metadata, &self.blobs).map_err(cause)?;
        if size.directory_bytes > self.limits.receiving.max_directory_bytes as u64
            || size.package_bytes > self.limits.receiving.max_source_bytes
        {
            return Err(fail("QTEA complete package receiving cap exceeded"));
        }
        let package_id = write_trit_package_from_spool(
            &metadata,
            &self.blobs,
            &mut self.spool,
            usize::try_from(self.limits.receiving.max_source_bytes).map_err(cause)?,
            output,
        )
        .map_err(cause)?;
        output.flush().map_err(cause)?;
        output.seek(SeekFrom::Start(0)).map_err(cause)?;
        let mut reader = match self.inventory.profile() {
            QteaModelProfile::Standard => QteaModelReader::open(
                output,
                self.limits.receiving,
                self.limits.model,
                *self.record.source_model.as_bytes(),
            ),
            QteaModelProfile::Qwen35Hybrid => QteaModelReader::open_hybrid(
                output,
                self.limits.receiving,
                QteaHybridModelReadLimits {
                    base: self.limits.model,
                    max_preserved_tensor_elements: self.limits.max_preserved_tensor_elements,
                },
                *self.record.source_model.as_bytes(),
            ),
        }
        .map_err(cause)?;
        if reader.package_id() != package_id {
            return Err(fail("QTEA writer/reopened package identity mismatch"));
        }
        verify_qtea_model_build_record(&mut reader, self.limits.max_fit_record_bytes)?;
        self.source.revalidate().map_err(cause)?;
        reader.verify().map_err(cause)?;
        Ok(QteaModelBuildResult {
            package_id,
            record: self.record,
        })
    }
}
struct SpoolWriter<'a, W> {
    sink: &'a mut W,
    cap: u64,
    bytes: u64,
    hash: blake3::Hasher,
    content: blake3::Hasher,
}
impl<'a, W> SpoolWriter<'a, W> {
    fn new(sink: &'a mut W, cap: u64) -> Self {
        Self {
            sink,
            cap,
            bytes: 0,
            hash: blake3::Hasher::new(),
            content: blake3::Hasher::new_derive_key(crate::CONTENT_ID_CONTEXT),
        }
    }
}
impl<W: Write> Write for SpoolWriter<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() as u64 > self.cap.saturating_sub(self.bytes) {
            return Err(std::io::Error::other("QTEA spool cap exceeded"));
        }
        let n = self.sink.write(bytes)?;
        if n > bytes.len() {
            return Err(std::io::Error::other("invalid spool writer byte count"));
        }
        self.bytes += n as u64;
        self.hash.update(&bytes[..n]);
        self.content.update(&bytes[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.sink.flush()
    }
}
struct RecordWriter {
    bytes: Vec<u8>,
    cap: usize,
}
impl Write for RecordWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.cap.saturating_sub(self.bytes.len()) {
            return Err(std::io::Error::other("QTEA fit record cap exceeded"));
        }
        self.bytes
            .try_reserve_exact(bytes.len())
            .map_err(std::io::Error::other)?;
        if self.bytes.capacity() > self.cap {
            return Err(std::io::Error::other("QTEA fit record capacity exceeded"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
