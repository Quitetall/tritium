//! Complete model fitting through one source/sink seam (ADR 0058 candidate).

use super::{DirectFitError, RecordedDirectFitError, fit_direct_levels_recorded};
use std::{borrow::Cow, collections::BTreeMap};
use tritium_core::DType;
use tritium_evidence::{EvidenceError, EvidenceRecorder, QuantizationReceiptView, SealedEvidence};
use tritium_format::trit_model::{
    TritModelError, decode_dense_f32, encode_model_manifest, read_trit_model,
    validate_dense_payload,
};
use tritium_format::{TritBlob, TritBlobKind, TritPackageError, write_trit_package_bounded};
use tritium_schema::{
    BlobId, BlobReference, DirectFitMetric, ModelId, NamedAssetDigest, OriginalSourceId, PackageId,
    PlaneCounts, QuantAction, QuantOutputLevel, QuantPlan, QuantPlanError, QuantSourceBinding,
    QuantSourceTensor, QuantizationCompleted, QuantizationStarted, ScalePrecision, TensorMeta,
    TensorStorage,
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
    /// Actual evidence integrity or typed-fold failure.
    Evidence(EvidenceError),
    /// Existing typed model/payload reader failed.
    Model(TritModelError),
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
        plan.validate().map_err(QuantEngineError::Plan)?;
        check_source_inventory(plan, source)?;
        let total = plan.tensors.iter().try_fold(0u64, |total, tensor| {
            total
                .checked_add(tensor.source.byte_len().map_err(QuantEngineError::Plan)?)
                .ok_or(QuantEngineError::Limit("source bytes"))
        })?;
        bound(total, plan.limits.max_source_bytes, "source bytes")?;
        let mut assets = Vec::new();
        let mut asset_bytes = 0u64;
        for name in &plan.assets {
            let remaining = plan.limits.max_asset_bytes - asset_bytes;
            let bytes = source.asset(name, platform(remaining, "asset bytes")?)?;
            bound(bytes.len() as u64, remaining, "asset bytes")?;
            asset_bytes += bytes.len() as u64;
            assets.push(NamedAssetDigest {
                name: name.clone(),
                content: BlobId::from_bytes(*blake3::hash(&bytes).as_bytes()),
            });
        }
        let mut tensors = Vec::new();
        for request in &plan.tensors {
            let original = &request.source;
            let bytes = source.tensor(
                &original.name,
                platform(
                    original.byte_len().map_err(QuantEngineError::Plan)?,
                    "source bytes",
                )?,
            )?;
            tensors.push(source_binding(original, &bytes)?);
        }
        tritium_evidence::original_source_id(&plan.architecture, &tensors, &assets)
            .map_err(QuantEngineError::Plan)
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
        Self::run_impl(plan, source, sink, recorder, None)
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
        Self::run_impl(plan, source, sink, recorder, Some(expected_source))
    }

    fn run_impl(
        plan: &QuantPlan,
        source: &mut dyn WeightSource,
        sink: &mut dyn TensorSink,
        mut recorder: EvidenceRecorder,
        expected_source: Option<OriginalSourceId>,
    ) -> Result<QuantizedModel, QuantEngineError> {
        plan.validate().map_err(QuantEngineError::Plan)?;
        if !recorder.events().is_empty() {
            return Err(QuantEngineError::Binding(
                "engine requires a fresh recorder",
            ));
        }
        recorder.check_emit::<QuantizationStarted>("quantize", None, 0)?;
        recorder.check_emit::<QuantizationCompleted>("quantize", None, 1)?;
        let mut total = 0u64;
        for (index, tensor) in plan.tensors.iter().enumerate() {
            total = total
                .checked_add(tensor.source.byte_len().map_err(QuantEngineError::Plan)?)
                .ok_or(QuantEngineError::Limit("source bytes"))?;
            if let QuantAction::Fit { plan, metric } = &tensor.action {
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
        }
        bound(total, plan.limits.max_source_bytes, "source bytes")?;
        check_source_inventory(plan, source)?;
        recorder.emit(
            "quantize",
            None,
            0,
            &QuantizationStarted { plan: plan.clone() },
        )?;
        let mut blobs = BlobStore::new(plan.limits.max_retained_blob_bytes);
        let mut assets = Vec::new();
        let mut asset_bytes = 0u64;
        for name in &plan.assets {
            let remaining = plan
                .limits
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
            .levels
            .iter()
            .map(|name| QuantOutputLevel {
                name: name.clone(),
                tensors: Vec::new(),
            })
            .collect::<Vec<_>>();
        for (index, request) in plan.tensors.iter().enumerate() {
            let original = &request.source;
            let bytes = source.tensor(
                &original.name,
                platform(
                    original.byte_len().map_err(QuantEngineError::Plan)?,
                    "source bytes",
                )?,
            )?;
            sources.push(source_binding(original, &bytes)?);
            match &request.action {
                QuantAction::KeepDense => {
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
                QuantAction::Fit { plan, metric } => {
                    let weights = decode_dense_f32(
                        &original.name,
                        &original.shape,
                        original.precision,
                        &bytes,
                    )?;
                    drop(bytes);
                    let gram = match metric {
                        DirectFitMetric::Identity => None,
                        DirectFitMetric::InputGramF64 { content } => {
                            let count = plan
                                .layout
                                .cols
                                .checked_mul(plan.layout.cols)
                                .ok_or(QuantEngineError::Limit("input Gram"))?;
                            let values = source
                                .input_gram(&original.name, platform(count, "input Gram")?)?;
                            if values.len() as u64 != count
                                || super::recorded::gram_digest(plan.layout.cols, &values)
                                    != *content
                            {
                                return Err(QuantEngineError::Binding("requested Gram changed"));
                            }
                            Some(values)
                        }
                        _ => return Err(QuantEngineError::Binding("unsupported metric")),
                    };
                    let fitted = fit_direct_levels_recorded(
                        &original.name,
                        plan,
                        &weights,
                        gram.as_deref(),
                        &mut recorder,
                        &format!("tensor-{index}"),
                        Some("quantize"),
                        0,
                    )
                    .map_err(QuantEngineError::Fit)?;
                    for fitted in fitted.levels {
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
        recorder.emit(
            "quantize",
            None,
            1,
            &QuantizationCompleted {
                sources,
                assets,
                levels,
            },
        )?;
        let evidence = recorder.seal()?;
        let views = QuantizationReceiptView::fold(evidence.log())?;
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
        let manifest = encode_model_manifest(view.manifest())?;
        let records = blobs.records();
        let package = write_trit_package_bounded(
            &manifest,
            &records,
            platform(plan.limits.max_package_bytes, "package bytes")?,
        )?;
        let model = bound_package(&package, evidence.log())?;
        let package_id = PackageId::from_bytes(*model.package_id().as_bytes());
        let model_ids = plan
            .levels
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

fn check_source_inventory(
    plan: &QuantPlan,
    source: &dyn WeightSource,
) -> Result<(), QuantEngineError> {
    if source
        .inventory()?
        .iter()
        .ne(plan.tensors.iter().map(|t| &t.source))
        || source.asset_names() != plan.assets
    {
        return Err(QuantEngineError::Binding(
            "complete original inventory changed",
        ));
    }
    Ok(())
}

fn source_binding(
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

fn bound_package<'a>(
    package: &'a [u8],
    log: &tritium_evidence::VerifiedLog,
) -> Result<tritium_format::trit_model::TritModel<'a>, QuantEngineError> {
    let views = QuantizationReceiptView::fold(log)?;
    if views.len() != 1 {
        return Err(QuantEngineError::Binding(
            "expected exactly one completed conversion",
        ));
    }
    let view = &views[0];
    let model = read_trit_model(package)?;
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
    Ok(model)
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
