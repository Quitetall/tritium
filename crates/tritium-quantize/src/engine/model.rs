//! Complete model fitting through one source/sink seam (ADR 0058 candidate).

use super::StagedSalienceSource;
use super::salience_model::SalienceProductionContext;
use super::{
    CalibrationCacheError, CalibrationSource, DirectFitError, OwnerCalibrationSource,
    OwnerScoreCacheError, RecordedCalibratedFitError, RecordedDirectFitError,
    RecordedOwnerScoreFitError, fit_calibrated_levels_recorded, fit_direct_levels_recorded,
    fit_owner_score_levels_recorded,
};
use std::{
    borrow::Cow,
    collections::BTreeMap,
    io::{Read, Seek},
};
use tritium_core::DType;
use tritium_evidence::{
    CalibratedQuantizationReceiptView, EvidenceError, EvidenceRecorder,
    OwnerQuantizationReceiptView, QuantizationReceiptView, SealedEvidence,
};
use tritium_format::trit_model::{
    TritModelError, TritModelReadError, TritModelReader, TritModelVisitError, decode_dense_f32,
    encode_model_manifest, read_trit_model, validate_dense_payload,
};
use tritium_format::{
    TritBlob, TritBlobKind, TritPackageError, TritPackageReadLimits, write_trit_package_bounded,
};
use tritium_schema::{
    BlobId, BlobReference, CalibratedQuantPlan, CalibratedQuantizationCompleted,
    CalibratedQuantizationStarted, DirectFitMetric, ModelId, ModelManifest, ModelQuantPlan,
    NamedAssetDigest, OriginalSourceId, OwnerQuantPlan, OwnerQuantizationCompleted,
    OwnerQuantizationStarted, PackageId, PlaneCounts, QuantActionRef, QuantOutputLevel, QuantPlan,
    QuantPlanError, QuantSourceBinding, QuantSourceTensor, QuantizationCompleted,
    QuantizationStarted, SalienceConversionLimits, ScalePrecision, SemanticTensorDigest,
    TensorMeta, TensorStorage,
};

/// Original byte provider; adapters never supply trusted input digests.
pub trait WeightSource {
    /// Complete sorted original interpretations; no payload read is required.
    fn inventory(&self) -> Result<Vec<QuantSourceTensor>, QuantEngineError>;
    /// Complete sorted original asset basenames.
    fn asset_names(&self) -> Vec<String>;
    /// Actual original raw bytes, bounded before allocation in owned adapters.
    fn tensor(&mut self, name: &str, max_bytes: usize) -> Result<Cow<'_, [u8]>, QuantEngineError>;
    /// Exact original asset bytes, not reserialized JSON.
    fn asset(&mut self, name: &str, max_bytes: usize) -> Result<Cow<'_, [u8]>, QuantEngineError>;
    /// Requested original row-major input Gram; absence is an error, not fallback.
    fn input_gram(
        &mut self,
        name: &str,
        max_entries: usize,
    ) -> Result<Cow<'_, [f64]>, QuantEngineError>;
}

/// Destination for an already checked package and the actual sealed fit-stage log.
pub trait TensorSink {
    /// Publish without overwriting an existing destination. A committed artifact
    /// with post-commit cleanup/sync trouble returns explicit warnings.
    fn publish(
        &mut self,
        package: &[u8],
        evidence: &SealedEvidence,
    ) -> Result<Vec<PublicationWarning>, QuantEngineError>;
}

/// Publication succeeded but a post-commit filesystem operation failed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PublicationWarning {
    /// The private pending hard-link name could not be removed.
    PendingCleanup,
    /// The directory could not be synced after the no-clobber commit point.
    DirectorySync,
}

/// Conversion errors never silently substitute a different fit or source.
#[derive(Debug)]
#[non_exhaustive]
pub enum QuantEngineError {
    /// Invalid complete structural plan.
    Plan(QuantPlanError),
    /// Existing production numerical admission failure.
    Admission(DirectFitError),
    /// Existing recorded fitter failed.
    Fit(RecordedDirectFitError),
    /// Actual cache loading/reduction failed.
    Cache(CalibrationCacheError),
    /// Actual recorded calibrated controller failed.
    CalibratedFit(RecordedCalibratedFitError),
    /// Actual owner cache file/content admission failed.
    OwnerCache(OwnerScoreCacheError),
    /// Actual recorded owner-window fitting failed.
    OwnerFit(RecordedOwnerScoreFitError),
    /// Actual evidence integrity or typed-fold failure.
    Evidence(EvidenceError),
    /// Existing typed model/payload reader failed.
    Model(TritModelError),
    /// Existing seekable typed model/source reader failed.
    ModelRead(TritModelReadError),
    /// Original typed model/source visitor or destination write failure.
    ModelVisit(TritModelVisitError<std::io::Error>),
    /// Existing physical package writer failed.
    Package(TritPackageError),
    /// Source metadata, missing payload or failed I/O.
    Source(String),
    /// Exact inventory or actual package/log bindings disagree.
    Binding(&'static str),
    /// Logical byte/entry bound exceeded or unrepresentable.
    Limit(&'static str),
    /// Filesystem publication failed before its commit point.
    Io(std::io::Error),
    /// Destination already holds an artifact and may not be overwritten.
    DestinationExists,
    /// A fallible retained-source/output allocation failed.
    Allocation,
}
impl core::fmt::Display for QuantEngineError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "quant engine rejected: {self:?}")
    }
}
impl std::error::Error for QuantEngineError {}
impl From<EvidenceError> for QuantEngineError {
    fn from(e: EvidenceError) -> Self {
        Self::Evidence(e)
    }
}
impl From<TritModelError> for QuantEngineError {
    fn from(e: TritModelError) -> Self {
        Self::Model(e)
    }
}
impl From<TritModelReadError> for QuantEngineError {
    fn from(error: TritModelReadError) -> Self {
        Self::ModelRead(error)
    }
}
impl From<TritPackageError> for QuantEngineError {
    fn from(e: TritPackageError) -> Self {
        Self::Package(e)
    }
}
impl From<std::io::Error> for QuantEngineError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Actual published identities and the immutable original fit-stage log.
/// Does not claim model quality, architecture execution, speed, RSS or release clearance.
#[derive(Debug)]
pub struct QuantizedModel {
    /// Candidate identity of actual original bytes, not fitted semantics or a source lock.
    pub source_id: OriginalSourceId,
    /// Exact complete package bytes identity.
    pub package_id: PackageId,
    /// Checked semantic identity of every complete level.
    pub model_ids: BTreeMap<String, ModelId>,
    /// Actual consumed recorder; provenance uses this root, not a fabricated one.
    pub evidence: SealedEvidence,
    /// Publication happened; these are explicit post-commit durability/cleanup caveats.
    pub warnings: Vec<PublicationWarning>,
}

/// Rust-owned complete conversion controller; numerical work is not duplicated.
#[derive(Debug)]
pub struct Engine;
#[derive(Default)]
struct FitSources<'a, 'b> {
    calibrated: Option<&'a mut dyn CalibrationSource>,
    owners: Option<&'b mut dyn OwnerCalibrationSource>,
}
impl<'a> FitSources<'a, '_> {
    fn calibrated(source: &'a mut dyn CalibrationSource) -> Self {
        Self {
            calibrated: Some(source),
            owners: None,
        }
    }
}
impl Engine {
    /// Observe the complete original model identity before calibration, without fitting.
    ///
    /// Reads actual assets and raw tensors through the same source seam and validates
    /// their requested complete inventory/interpretation. Original payloads are not
    /// widened or retained by the engine. Logical source/asset limits apply; this
    /// does not lock mutable source files or admit caller/adapter/full-process RSS.
    /// A subsequent conversion must compare its actual identity; this observation
    /// alone does not bind calibration to a package or establish release provenance.
    /// # Errors
    /// Invalid plan, changed inventory, source I/O/payload or logical byte limits.
    pub fn observe_source(
        plan: &QuantPlan,
        source: &mut dyn WeightSource,
    ) -> Result<OriginalSourceId, QuantEngineError> {
        Ok(super::ObservedWeightSource::new(plan, source)?.source_id())
    }

    /// Fit complete levels from original sources, seal evidence, write and check
    /// a canonical package, then publish through the sink.
    ///
    /// Initially retains unique encoded blobs and the final package in memory.
    /// Logical limits do not admit complete RSS or out-of-core execution.
    /// # Errors
    /// Invalid plan/context/inventory/bytes/metric/fit/bindings or pre-commit I/O.
    pub fn run(
        plan: &QuantPlan,
        source: &mut dyn WeightSource,
        sink: &mut dyn TensorSink,
        recorder: EvidenceRecorder,
    ) -> Result<QuantizedModel, QuantEngineError> {
        Self::run_impl(
            ModelQuantPlan::Uncalibrated(plan),
            source,
            FitSources::default(),
            sink,
            recorder,
            None,
            None,
        )
    }

    /// Convert only if the actual complete original inventory matches preparation.
    ///
    /// The expected identity should come from actual `observe_source` execution.
    /// Conversion derives its own identity from bytes actually consumed and checks
    /// equality before allocating/publishing the final package. Fitting may already
    /// have occurred when a mismatch is detected; there is no second original-byte
    /// read or fallback. Existing unguarded conversion retains its original semantics.
    ///
    /// This runtime predicate does not lock mutable sources, verify calibration
    /// cache/objective/data, or add a recorded expected-identity claim to the v1 log.
    /// The existing view still exposes actual source identity; audited metric/package
    /// provenance needs its separately typed binding/verification.
    /// # Errors
    /// Original consumed source differs, or any ordinary conversion error occurs.
    pub fn run_with_source_guard(
        plan: &QuantPlan,
        source: &mut dyn WeightSource,
        sink: &mut dyn TensorSink,
        recorder: EvidenceRecorder,
        expected_source: OriginalSourceId,
    ) -> Result<QuantizedModel, QuantEngineError> {
        Self::run_impl(
            ModelQuantPlan::Uncalibrated(plan),
            source,
            FitSources::default(),
            sink,
            recorder,
            Some(expected_source),
            None,
        )
    }

    /// Fit a complete calibrated model through the same source/package/publication path.
    /// Actual consumed original identity must equal the model/cache expectation before
    /// successful completion or publication. No source reread, metric fallback or
    /// fabricated v1 lifecycle. Logical bounds are not total RSS or recipe truth.
    /// # Errors
    /// Invalid source/cache/plan/recorder, actual fit, output binding or publication.
    pub fn run_calibrated(
        plan: &CalibratedQuantPlan,
        source: &mut dyn WeightSource,
        caches: &mut dyn CalibrationSource,
        sink: &mut dyn TensorSink,
        recorder: EvidenceRecorder,
    ) -> Result<QuantizedModel, QuantEngineError> {
        Self::run_impl(
            ModelQuantPlan::Calibrated(plan),
            source,
            FitSources::calibrated(caches),
            sink,
            recorder,
            Some(plan.original_source),
            None,
        )
    }

    /// Convert a complete owner-calibrated model using the shared engine.
    /// Actual consumed original bytes must match every cache's preparation before
    /// completion/publication. Logical limits do not prove recipe custody or RSS.
    pub fn run_owner(
        plan: &OwnerQuantPlan,
        source: &mut dyn WeightSource,
        owners: &mut dyn OwnerCalibrationSource,
        calibrated: Option<&mut dyn CalibrationSource>,
        sink: &mut dyn TensorSink,
        recorder: EvidenceRecorder,
    ) -> Result<QuantizedModel, QuantEngineError> {
        Self::run_impl(
            ModelQuantPlan::Owner(plan),
            source,
            FitSources {
                calibrated,
                owners: Some(owners),
            },
            sink,
            recorder,
            None,
            None,
        )
    }

    /// Convert an actual admitted immutable B stage into mandatory joined C.
    /// Uses the same fitter, writer and publication path as ordinary conversion.
    /// Structural lineage is not captured-graph, numerical replay or qualification.
    /// # Errors
    /// Stage/request/source/output/root disagreement, limits or ordinary conversion failure.
    pub fn run_salience(
        plan: &QuantPlan,
        stage: &mut StagedSalienceSource<'_>,
        sink: &mut dyn TensorSink,
        recorder: EvidenceRecorder,
        limits: SalienceConversionLimits,
    ) -> Result<QuantizedModel, QuantEngineError> {
        let context = SalienceProductionContext::from_stage(stage, limits)?;
        let expected = stage.source_id();
        Self::run_impl(
            ModelQuantPlan::Uncalibrated(plan),
            stage,
            FitSources::default(),
            sink,
            recorder,
            Some(expected),
            Some(context),
        )
    }

    /// Fit actual B-bound caches through the same mandatory joined C path.
    /// Does not rescale or relabel A caches, rerun graph capture or clone dense B.
    /// # Errors
    /// Foreign original/cache binding, limits or ordinary calibrated conversion failure.
    pub fn run_calibrated_salience(
        plan: &CalibratedQuantPlan,
        stage: &mut StagedSalienceSource<'_>,
        caches: &mut dyn CalibrationSource,
        sink: &mut dyn TensorSink,
        recorder: EvidenceRecorder,
        limits: SalienceConversionLimits,
    ) -> Result<QuantizedModel, QuantEngineError> {
        if plan.original_source != stage.source_id() {
            return Err(QuantEngineError::Binding(
                "calibrated request differs from actual B stage",
            ));
        }
        let context = SalienceProductionContext::from_stage(stage, limits)?;
        Self::run_impl(
            ModelQuantPlan::Calibrated(plan),
            stage,
            FitSources::calibrated(caches),
            sink,
            recorder,
            Some(plan.original_source),
            Some(context),
        )
    }

    /// Fit through the distinct mandatory capture-root carrier, using the common engine.
    /// Requires actual B and a complete registered capture log whose exact cache
    /// bindings agree with every calibrated action. Does not prove graph replay,
    /// clone dense B, recast plain fits or publish/repack an ordinary C artifact.
    /// # Errors
    /// Missing/foreign capture/stage/cache, uncalibrated actions, caps or conversion failure.
    pub fn run_captured_salience(
        plan: &CalibratedQuantPlan,
        stage: &mut StagedSalienceSource<'_>,
        capture_log: &tritium_evidence::VerifiedLog,
        caches: &mut dyn CalibrationSource,
        sink: &mut dyn TensorSink,
        recorder: EvidenceRecorder,
        limits: SalienceConversionLimits,
    ) -> Result<QuantizedModel, QuantEngineError> {
        if plan.original_source != stage.source_id() {
            return Err(QuantEngineError::Binding(
                "captured request differs from actual B",
            ));
        }
        let context = SalienceProductionContext::from_capture(stage, capture_log, limits)?;
        context.check_capture_plan(plan)?;
        Self::run_impl(
            ModelQuantPlan::Calibrated(plan),
            stage,
            FitSources::calibrated(caches),
            sink,
            recorder,
            Some(plan.original_source),
            Some(context),
        )
    }

    /// Fit through mandatory original-A and B capture ancestry with the common engine.
    /// Only an explicitly admitted original-captured B stage can select this carrier.
    /// No supplied original reference, plain-fit fallback, new fitter or repack is accepted.
    /// # Errors
    /// Wrong stage profile, actual B/capture/cache/request mismatch, caps or publication failure.
    pub fn run_original_captured_salience(
        plan: &CalibratedQuantPlan,
        stage: &mut StagedSalienceSource<'_>,
        capture_log: &tritium_evidence::VerifiedLog,
        caches: &mut dyn CalibrationSource,
        sink: &mut dyn TensorSink,
        recorder: EvidenceRecorder,
        limits: SalienceConversionLimits,
    ) -> Result<QuantizedModel, QuantEngineError> {
        if plan.original_source != stage.source_id() {
            return Err(QuantEngineError::Binding(
                "original-captured request differs from actual B",
            ));
        }
        let context = SalienceProductionContext::from_original_capture(stage, capture_log, limits)?;
        context.check_capture_plan(plan)?;
        Self::run_impl(
            ModelQuantPlan::Calibrated(plan),
            stage,
            FitSources::calibrated(caches),
            sink,
            recorder,
            Some(plan.original_source),
            Some(context),
        )
    }

    fn run_impl(
        plan: ModelQuantPlan<'_>,
        source: &mut dyn WeightSource,
        fit_sources: FitSources<'_, '_>,
        sink: &mut dyn TensorSink,
        mut recorder: EvidenceRecorder,
        expected_source: Option<OriginalSourceId>,
        salience: Option<SalienceProductionContext>,
    ) -> Result<QuantizedModel, QuantEngineError> {
        let mut caches = fit_sources.calibrated;
        let mut owners = fit_sources.owners;
        plan.validate().map_err(QuantEngineError::Plan)?;
        if !recorder.events().is_empty() {
            return Err(QuantEngineError::Binding(
                "engine requires a fresh recorder",
            ));
        }
        match plan {
            ModelQuantPlan::Uncalibrated(_) => {
                recorder.check_emit::<QuantizationStarted>("quantize", None, 0)?;
                recorder.check_emit::<QuantizationCompleted>("quantize", None, 1)?;
            }
            ModelQuantPlan::Calibrated(_) => {
                recorder.check_emit::<CalibratedQuantizationStarted>("quantize", None, 0)?;
                recorder.check_emit::<CalibratedQuantizationCompleted>("quantize", None, 1)?;
            }
            ModelQuantPlan::Owner(_) => {
                recorder.check_emit::<OwnerQuantizationStarted>("quantize", None, 0)?;
                recorder.check_emit::<OwnerQuantizationCompleted>("quantize", None, 1)?;
                let names = plan
                    .tensors()
                    .filter(|t| matches!(t.action, QuantActionRef::OwnerFit(_)))
                    .map(|t| t.source.name.clone())
                    .collect::<Vec<_>>();
                if owners
                    .as_deref()
                    .ok_or(QuantEngineError::Binding("owner calibration source absent"))?
                    .names()
                    != names
                {
                    return Err(QuantEngineError::Binding(
                        "complete owner inventory changed",
                    ));
                }
            }
            _ => return Err(QuantEngineError::Binding("unsupported model request")),
        }
        if plan.cache_limits().is_some() {
            let names = plan
                .tensors()
                .filter(|t| matches!(t.action, QuantActionRef::CalibratedFit(_)))
                .map(|t| t.source.name.clone())
                .collect::<Vec<_>>();
            if names.is_empty()
                && caches
                    .as_deref()
                    .is_none_or(|source| source.names().is_empty())
            {
                // No legacy cache is invented for an owner-only request.
            } else if caches
                .as_deref()
                .ok_or(QuantEngineError::Binding("calibration source absent"))?
                .names()
                != names
            {
                return Err(QuantEngineError::Binding(
                    "complete calibration inventory changed",
                ));
            }
        }
        let limits = plan.limits();
        let mut total = 0u64;
        for (index, tensor) in plan.tensors().enumerate() {
            total = total
                .checked_add(tensor.source.byte_len().map_err(QuantEngineError::Plan)?)
                .ok_or(QuantEngineError::Limit("source bytes"))?;
            if let QuantActionRef::Fit { plan, metric } = tensor.action {
                super::direct::admit_plan(
                    plan,
                    matches!(metric, DirectFitMetric::InputGramF64 { .. }),
                )
                .map_err(QuantEngineError::Admission)?;
                recorder.check_emit::<tritium_schema::DirectLevelsFitted>(
                    &format!("tensor-{index}"),
                    Some("quantize"),
                    0,
                )?;
            }
            if let QuantActionRef::CalibratedFit(plan) = tensor.action {
                super::direct::admit_plan(&plan.direct, true)
                    .map_err(QuantEngineError::Admission)?;
                recorder.check_emit::<tritium_schema::CalibratedLevelsFitted>(
                    &format!("tensor-{index}"),
                    Some("quantize"),
                    0,
                )?;
            }
            if let QuantActionRef::OwnerFit(plan) = tensor.action {
                super::direct::admit_plan(&plan.direct, true)
                    .map_err(QuantEngineError::Admission)?;
                recorder.check_owner_score_fit_context(
                    &format!("tensor-{index}"),
                    Some("quantize"),
                    0,
                    plan,
                )?;
            }
        }
        bound(total, limits.max_source_bytes, "source bytes")?;
        check_source_inventory(plan, source)?;
        match plan {
            ModelQuantPlan::Uncalibrated(plan) => {
                recorder.emit(
                    "quantize",
                    None,
                    0,
                    &QuantizationStarted { plan: plan.clone() },
                )?;
            }
            ModelQuantPlan::Calibrated(plan) => {
                recorder.emit(
                    "quantize",
                    None,
                    0,
                    &CalibratedQuantizationStarted { plan: plan.clone() },
                )?;
            }
            ModelQuantPlan::Owner(plan) => {
                recorder.emit(
                    "quantize",
                    None,
                    0,
                    &OwnerQuantizationStarted { plan: plan.clone() },
                )?;
            }
            _ => return Err(QuantEngineError::Binding("unsupported model request")),
        }
        let mut blobs = BlobStore::new(limits.max_retained_blob_bytes);
        let mut assets = Vec::new();
        let mut asset_bytes = 0u64;
        for name in plan.assets() {
            let remaining = limits
                .max_asset_bytes
                .checked_sub(asset_bytes)
                .ok_or(QuantEngineError::Limit("asset bytes"))?;
            let bytes = source.asset(name, platform(remaining, "asset bytes")?)?;
            bound(bytes.len() as u64, remaining, "asset bytes")?;
            asset_bytes += bytes.len() as u64;
            let content = blobs.insert(TritBlobKind::Asset(name.clone()), &bytes)?;
            assets.push(NamedAssetDigest {
                name: name.clone(),
                content,
            });
        }
        let mut sources = Vec::new();
        let mut levels = plan
            .levels()
            .iter()
            .map(|name| QuantOutputLevel {
                name: name.clone(),
                tensors: Vec::new(),
            })
            .collect::<Vec<_>>();
        for (index, request) in plan.tensors().enumerate() {
            let original = request.source;
            let bytes = source.tensor(
                &original.name,
                platform(
                    original.byte_len().map_err(QuantEngineError::Plan)?,
                    "source bytes",
                )?,
            )?;
            sources.push(source_binding(original, &bytes)?);
            match request.action {
                QuantActionRef::KeepDense => {
                    let kind = match original.precision {
                        ScalePrecision::F16 => TritBlobKind::DenseTensor(DType::F16),
                        ScalePrecision::Bf16 => TritBlobKind::DenseTensor(DType::BF16),
                        ScalePrecision::F32 => TritBlobKind::DenseTensor(DType::F32),
                        _ => return Err(QuantEngineError::Binding("unsupported dense precision")),
                    };
                    let id = blobs.insert(kind, &bytes)?;
                    for level in &mut levels {
                        level.tensors.push(TensorMeta {
                            name: original.name.clone(),
                            blob: local(id),
                            storage: TensorStorage::Dense {
                                shape: original.shape.clone(),
                                precision: original.precision,
                            },
                        });
                    }
                }
                action if action.direct().is_some() => {
                    let weights = decode_dense_f32(
                        &original.name,
                        &original.shape,
                        original.precision,
                        &bytes,
                    )?;
                    drop(bytes);
                    let fitted = match action {
                        QuantActionRef::Fit { plan, metric } => {
                            let gram = match metric {
                                DirectFitMetric::Identity => None,
                                DirectFitMetric::InputGramF64 { content } => {
                                    let count = plan
                                        .layout
                                        .cols
                                        .checked_mul(plan.layout.cols)
                                        .ok_or(QuantEngineError::Limit("input Gram"))?;
                                    let values = source.input_gram(
                                        &original.name,
                                        platform(count, "input Gram")?,
                                    )?;
                                    if values.len() as u64 != count
                                        || super::recorded::gram_digest(plan.layout.cols, &values)
                                            != *content
                                    {
                                        return Err(QuantEngineError::Binding(
                                            "requested Gram changed",
                                        ));
                                    }
                                    Some(values)
                                }
                                _ => return Err(QuantEngineError::Binding("unsupported metric")),
                            };
                            fit_direct_levels_recorded(
                                &original.name,
                                plan,
                                &weights,
                                gram.as_deref(),
                                &mut recorder,
                                &format!("tensor-{index}"),
                                Some("quantize"),
                                0,
                            )
                            .map_err(QuantEngineError::Fit)?
                            .levels
                        }
                        QuantActionRef::CalibratedFit(request) => {
                            let (_, max_package) = plan.cache_limits().ok_or(
                                QuantEngineError::Binding("calibrated model request absent"),
                            )?;
                            let cache = caches
                                .as_deref_mut()
                                .ok_or(QuantEngineError::Binding("calibration source absent"))?
                                .cache(&request.cache, max_package)?;
                            fit_calibrated_levels_recorded(
                                request,
                                &weights,
                                cache,
                                &mut recorder,
                                &format!("tensor-{index}"),
                                Some("quantize"),
                                0,
                            )
                            .map_err(QuantEngineError::CalibratedFit)?
                            .levels
                        }
                        QuantActionRef::OwnerFit(request) => {
                            let (_, max_package) = plan
                                .cache_limits()
                                .ok_or(QuantEngineError::Binding("owner model request absent"))?;
                            let stream = owners
                                .as_deref_mut()
                                .ok_or(QuantEngineError::Binding("owner source absent"))?
                                .stream(request, max_package)?;
                            fit_owner_score_levels_recorded(
                                request,
                                &weights,
                                &stream,
                                &mut recorder,
                                &format!("tensor-{index}"),
                                Some("quantize"),
                                0,
                            )
                            .map_err(QuantEngineError::OwnerFit)?
                            .levels
                        }
                        _ => return Err(QuantEngineError::Binding("unsupported fit request")),
                    };
                    for fitted in fitted {
                        let bytes = fitted
                            .tensor
                            .encode_payload()
                            .map_err(|e| QuantEngineError::Model(TritModelError::Additive(e)))?;
                        let id = blobs.insert(TritBlobKind::AdditiveTensor, &bytes)?;
                        let level = levels
                            .iter_mut()
                            .find(|l| l.name == fitted.level)
                            .ok_or(QuantEngineError::Binding("fit level changed"))?;
                        level.tensors.push(TensorMeta {
                            name: original.name.clone(),
                            blob: local(id),
                            storage: TensorStorage::Additive {
                                layout: fitted.tensor.layout(),
                                planes: PlaneCounts::Uniform(fitted.tensor.plane_count()),
                            },
                        });
                    }
                }
                _ => return Err(QuantEngineError::Binding("unsupported action")),
            }
        }
        let actual_source =
            tritium_evidence::original_source_id(plan.architecture(), &sources, &assets)
                .map_err(QuantEngineError::Plan)?;
        let completed = QuantizationCompleted {
            sources,
            assets,
            levels,
        };
        match plan {
            ModelQuantPlan::Uncalibrated(_) => {
                recorder.emit("quantize", None, 1, &completed)?;
            }
            ModelQuantPlan::Calibrated(request) => {
                if actual_source != request.original_source {
                    return Err(QuantEngineError::Binding(
                        "actual original source differs from calibration",
                    ));
                }
                recorder.emit(
                    "quantize",
                    None,
                    1,
                    &CalibratedQuantizationCompleted {
                        original_source: actual_source,
                        output: completed,
                    },
                )?;
            }
            ModelQuantPlan::Owner(request) => {
                if actual_source != request.original_source {
                    return Err(QuantEngineError::Binding(
                        "actual original source differs from owner preparation",
                    ));
                }
                recorder.emit(
                    "quantize",
                    None,
                    1,
                    &OwnerQuantizationCompleted {
                        original_source: actual_source,
                        output: completed,
                    },
                )?;
            }
            _ => return Err(QuantEngineError::Binding("unsupported model request")),
        }
        let evidence = recorder.seal()?;
        let views = conversion_views(evidence.log())?;
        if views.len() != 1 {
            return Err(QuantEngineError::Binding(
                "expected exactly one completed conversion",
            ));
        }
        let view = &views[0];
        if expected_source.is_some_and(|expected| expected != view.source_id()) {
            return Err(QuantEngineError::Binding(
                "actual original source differs from preparation",
            ));
        }
        let manifest = match &salience {
            Some(context) => context.manifest_bytes(view)?,
            None => encode_model_manifest(view.manifest())?,
        };
        let max_package_bytes = salience
            .as_ref()
            .map_or(limits.max_package_bytes, |context| {
                limits.max_package_bytes.min(context.max_package_bytes())
            });
        let records = blobs.records();
        let package = write_trit_package_bounded(
            &manifest,
            &records,
            platform(max_package_bytes, "package bytes")?,
        )?;
        let model = match &salience {
            Some(context) => context.open(&package, evidence.log())?.into_model(),
            None => bound_package(&package, evidence.log())?,
        };
        let package_id = PackageId::from_bytes(*model.package_id().as_bytes());
        let model_ids = plan
            .levels()
            .iter()
            .map(|name| Ok((name.clone(), model.model_id(name)?)))
            .collect::<Result<BTreeMap<_, _>, TritModelError>>()?;
        let warnings = sink.publish(&package, &evidence)?;
        Ok(QuantizedModel {
            source_id: view.source_id(),
            package_id,
            model_ids,
            evidence,
            warnings,
        })
    }
}

pub(super) fn check_source_inventory(
    plan: ModelQuantPlan<'_>,
    source: &dyn WeightSource,
) -> Result<(), QuantEngineError> {
    if source
        .inventory()?
        .iter()
        .ne(plan.tensors().map(|t| t.source))
        || source.asset_names() != plan.assets()
    {
        return Err(QuantEngineError::Binding(
            "complete original inventory changed",
        ));
    }
    Ok(())
}

pub(super) fn source_binding(
    original: &QuantSourceTensor,
    bytes: &[u8],
) -> Result<QuantSourceBinding, QuantEngineError> {
    validate_dense_payload(&original.name, &original.shape, original.precision, bytes)?;
    Ok(QuantSourceBinding {
        source: original.clone(),
        content: BlobId::from_bytes(*blake3::hash(bytes).as_bytes()),
    })
}

/// Verify a real typed package against a complete actual fit log.
/// Rechecks all payloads, roots, inventories, ancestry and fitted semantics.
/// Does not independently refit source weights or qualify quality/performance.
/// # Errors
/// Malformed package/log or mismatched complete binding.
pub fn verify_package_with_fit_log(
    package: &[u8],
    log: &tritium_evidence::VerifiedLog,
) -> Result<(), QuantEngineError> {
    bound_package(package, log).map(|_| ())
}

/// Open a seekable typed package and bind it to its complete actual fit log.
///
/// Uses the same manifest, inventory, ancestry and fitted-semantic checks as
/// [`verify_package_with_fit_log`], without a whole-package byte image. Typed
/// admission and terminal verification recheck complete payload/source identity.
/// Metadata and log allocations remain separately admitted; this is not a
/// whole-process memory bound, immutable writer snapshot or quality qualification.
/// The returned reader retains its source under the existing reader contract.
///
/// # Errors
/// Invalid package/log binding, source mutation, byte limits or actual I/O.
pub fn open_model_reader_with_fit_log<R: Read + Seek>(
    source: R,
    limits: TritPackageReadLimits,
    log: &tritium_evidence::VerifiedLog,
) -> Result<TritModelReader<R>, QuantEngineError> {
    let views = conversion_views(log)?;
    if views.len() != 1 {
        return Err(QuantEngineError::Binding(
            "expected exactly one completed conversion",
        ));
    }
    let mut model = TritModelReader::open(source, limits)?;
    verify_conversion_model(&model, &views[0])?;
    model.verify()?;
    Ok(model)
}

fn bound_package<'a>(
    package: &'a [u8],
    log: &tritium_evidence::VerifiedLog,
) -> Result<tritium_format::trit_model::TritModel<'a>, QuantEngineError> {
    let views = conversion_views(log)?;
    if views.len() != 1 {
        return Err(QuantEngineError::Binding(
            "expected exactly one completed conversion",
        ));
    }
    let view = &views[0];
    let model = read_trit_model(package)?;
    verify_conversion_model(&model, view)?;
    Ok(model)
}

pub(super) trait ConversionModel {
    fn manifest(&self) -> &ModelManifest;
    fn tensor_digest(
        &self,
        level: &str,
        name: &str,
    ) -> Result<SemanticTensorDigest, TritModelError>;
}

impl ConversionModel for tritium_format::trit_model::TritModel<'_> {
    fn manifest(&self) -> &ModelManifest {
        self.manifest()
    }
    fn tensor_digest(
        &self,
        level: &str,
        name: &str,
    ) -> Result<SemanticTensorDigest, TritModelError> {
        self.tensor_digest(level, name)
    }
}

impl<R: Read + Seek> ConversionModel for TritModelReader<R> {
    fn manifest(&self) -> &ModelManifest {
        self.manifest()
    }
    fn tensor_digest(
        &self,
        level: &str,
        name: &str,
    ) -> Result<SemanticTensorDigest, TritModelError> {
        self.tensor_digest(level, name)
    }
}

pub(super) fn verify_conversion_model(
    model: &impl ConversionModel,
    view: &ConversionView,
) -> Result<(), QuantEngineError> {
    if model.manifest() != view.manifest() {
        return Err(QuantEngineError::Binding(
            "manifest differs from sealed fit stage",
        ));
    }
    for fit in view.fits() {
        for output in &fit.observation().outputs {
            if model.tensor_digest(&output.level, &fit.observation().tensor)? != output.content {
                return Err(QuantEngineError::Binding("fitted semantic output changed"));
            }
        }
    }
    for fit in view.calibrated_fits() {
        let observation = fit.observation();
        for output in &observation.outputs {
            if model.tensor_digest(&output.level, &observation.plan.cache.request.tensor.name)?
                != output.content
            {
                return Err(QuantEngineError::Binding(
                    "calibrated semantic output changed",
                ));
            }
        }
    }
    for fit in view.owner_fits() {
        let observation = fit.observation();
        let name = &observation.plan.windows[0].binding.content.tensor.name;
        for output in &observation.outputs {
            if model.tensor_digest(&output.level, name)? != output.content {
                return Err(QuantEngineError::Binding("owner semantic output changed"));
            }
        }
    }
    Ok(())
}

pub(super) enum ConversionView {
    Plain(QuantizationReceiptView),
    Calibrated(CalibratedQuantizationReceiptView),
    Owner(OwnerQuantizationReceiptView),
}
impl ConversionView {
    pub(super) fn reference(&self) -> &tritium_schema::EvidenceReference {
        match self {
            Self::Plain(v) => v.reference(),
            Self::Calibrated(v) => v.reference(),
            Self::Owner(v) => v.reference(),
        }
    }
    pub(super) fn completed(&self) -> &QuantizationCompleted {
        match self {
            Self::Plain(v) => v.completed(),
            Self::Calibrated(v) => &v.completed().output,
            Self::Owner(v) => &v.completed().output,
        }
    }
    pub(super) fn manifest(&self) -> &tritium_schema::ModelManifest {
        match self {
            Self::Plain(v) => v.manifest(),
            Self::Calibrated(v) => v.manifest(),
            Self::Owner(v) => v.manifest(),
        }
    }
    pub(super) fn source_id(&self) -> OriginalSourceId {
        match self {
            Self::Plain(v) => v.source_id(),
            Self::Calibrated(v) => v.source_id(),
            Self::Owner(v) => v.source_id(),
        }
    }
    pub(super) fn fits(&self) -> &[tritium_evidence::DirectFitReceiptView] {
        match self {
            Self::Plain(v) => v.fits(),
            Self::Calibrated(v) => v.fits(),
            Self::Owner(v) => v.fits(),
        }
    }
    pub(super) fn calibrated_fits(&self) -> &[tritium_evidence::CalibratedFitReceiptView] {
        match self {
            Self::Plain(_) => &[],
            Self::Calibrated(v) => v.calibrated_fits(),
            Self::Owner(v) => v.calibrated_fits(),
        }
    }
    pub(super) fn owner_fits(&self) -> &[tritium_evidence::OwnerScoreFitReceiptView] {
        match self {
            Self::Owner(v) => v.owner_fits(),
            _ => &[],
        }
    }
}
pub(super) fn conversion_views(
    log: &tritium_evidence::VerifiedLog,
) -> Result<Vec<ConversionView>, EvidenceError> {
    let mut views = QuantizationReceiptView::fold(log)?
        .into_iter()
        .map(ConversionView::Plain)
        .collect::<Vec<_>>();
    views.extend(
        OwnerQuantizationReceiptView::fold(log)?
            .into_iter()
            .map(ConversionView::Owner),
    );
    views.extend(
        CalibratedQuantizationReceiptView::fold(log)?
            .into_iter()
            .map(ConversionView::Calibrated),
    );
    Ok(views)
}

fn local(id: BlobId) -> BlobReference {
    BlobReference { id, shard: None }
}
pub(super) fn platform(n: u64, resource: &'static str) -> Result<usize, QuantEngineError> {
    usize::try_from(n).map_err(|_| QuantEngineError::Limit(resource))
}
pub(super) fn bound(n: u64, limit: u64, resource: &'static str) -> Result<(), QuantEngineError> {
    if n > limit {
        Err(QuantEngineError::Limit(resource))
    } else {
        Ok(())
    }
}
struct BlobStore {
    blobs: BTreeMap<[u8; 32], (Vec<u8>, Vec<TritBlobKind>)>,
    retained: u64,
    limit: u64,
}
impl BlobStore {
    fn new(limit: u64) -> Self {
        Self {
            blobs: BTreeMap::new(),
            retained: 0,
            limit,
        }
    }
    fn insert(&mut self, kind: TritBlobKind, bytes: &[u8]) -> Result<BlobId, QuantEngineError> {
        let digest = *blake3::hash(bytes).as_bytes();
        if let Some((old, kinds)) = self.blobs.get_mut(&digest) {
            if old != bytes {
                return Err(QuantEngineError::Binding("blob identity collision"));
            }
            if !kinds.contains(&kind) {
                kinds.push(kind);
            }
        } else {
            let retained = self
                .retained
                .checked_add(bytes.len() as u64)
                .ok_or(QuantEngineError::Limit("retained blobs"))?;
            bound(retained, self.limit, "retained blobs")?;
            let mut owned = Vec::new();
            owned
                .try_reserve_exact(bytes.len())
                .map_err(|_| QuantEngineError::Allocation)?;
            owned.extend_from_slice(bytes);
            self.blobs.insert(digest, (owned, vec![kind]));
            self.retained = retained;
        }
        Ok(BlobId::from_bytes(digest))
    }
    fn records(&self) -> Vec<TritBlob<'_>> {
        self.blobs
            .values()
            .flat_map(|(bytes, kinds)| {
                kinds.iter().map(|kind| TritBlob {
                    kind: kind.clone(),
                    bytes: bytes.as_slice(),
                })
            })
            .collect()
    }
}
