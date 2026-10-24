//! Explicit experimental file orchestration shared by CLI and Python.
use crate::{
    ContentId, FRONTIER_QTEA_REFERENCE_SOLVER_ID, FrontierOrdering, FrontierProfile,
    FrontierResourceEstimate, FrontierResourceEstimator, FrontierSolverError, FrontierStage,
    FrontierStageRequest, QteaModelBuildError, QteaModelBuildLimits, QteaModelBuilder,
    QteaOwnerCurvature, QteaReferenceAdapter, QteaSelectedAdmission, QteaTensorWorkPlan,
    ResourceVector, SolverRegistry, SolverRequest, SolverTrust,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    io::{Cursor, Read, Seek, Write},
    path::PathBuf,
    sync::Arc,
};
use tritium_format::{
    TritPackageReadLimits,
    qtea::{QteaLimits, model::QteaModelReadLimits},
};
use tritium_nn::QteaModelProfile;
use tritium_quantize::{
    InputGram,
    engine::{
        CalibratedTensor, CalibrationCache, DirectoryTensorSink, FileWeightSource,
        ObservedWeightSource, read_original_file_bounded,
    },
    qtea_reference::TensorFitConfig,
};
use tritium_schema::{
    CalibrationCacheId, CalibrationCacheLimits, CalibrationPartition, ModelQuantPlan,
    NamedAssetDigest, OriginalSourceId, QuantAction, QuantPlan, QuantSourceBinding,
};

fn error(e: impl std::fmt::Display) -> QteaModelBuildError {
    QteaModelBuildError::selection(std::io::Error::other(e.to_string()))
}
fn encode(value: &impl Serialize, limit: usize) -> Result<Vec<u8>, QteaModelBuildError> {
    struct Bounded {
        bytes: Vec<u8>,
        limit: usize,
    }
    impl Write for Bounded {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            let length = self
                .bytes
                .len()
                .checked_add(bytes.len())
                .filter(|n| *n <= self.limit)
                .ok_or_else(|| std::io::Error::other("campaign JSON exceeds output cap"))?;
            self.bytes
                .try_reserve_exact(length - self.bytes.len())
                .map_err(std::io::Error::other)?;
            if self.bytes.capacity() > self.limit {
                return Err(std::io::Error::other(
                    "campaign JSON capacity exceeds output cap",
                ));
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    if limit == 0 {
        return Err(error("campaign JSON cap must be positive"));
    }
    let mut out = Bounded {
        bytes: Vec::new(),
        limit,
    };
    serde_json::to_writer(&mut out, value).map_err(error)?;
    Ok(out.bytes)
}

/// Positive per-domain caps; not total process/cache/allocator RSS admission.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QteaCampaignLimits {
    /// Complete output package byte ceiling.
    pub max_package_bytes: u64,
    /// Output manifest ceiling.
    pub max_manifest_bytes: usize,
    /// Output directory ceiling.
    pub max_directory_bytes: usize,
    /// Complete original/model owner count.
    pub max_tensors: usize,
    /// Model assets including generated fit observations.
    pub max_assets: usize,
    /// One model asset.
    pub max_asset_bytes: usize,
    /// Config asset.
    pub max_config_bytes: usize,
    /// Preserved vector elements.
    pub max_vector_elements: usize,
    /// Explicit hybrid higher-rank preserved elements.
    pub max_preserved_tensor_elements: usize,
    /// One encoded tensor capacity.
    pub max_tensor_bytes: usize,
    /// Tensor validation workspace.
    pub max_validation_bytes: usize,
    /// Actual widened original matrix capacity.
    pub max_widened_tensor_bytes: usize,
    /// Native numerical fit workspace.
    pub max_native_work_bytes: usize,
    /// Combined logical fit/export scope.
    pub max_tensor_work_bytes: usize,
    /// Private spool payload bytes.
    pub max_spool_bytes: u64,
    /// Generated fit-observation asset capacity.
    pub max_fit_record_bytes: usize,
    /// One encoded native Gram or calibration-cache input file.
    pub max_curvature_file_bytes: u64,
}
impl QteaCampaignLimits {
    fn build(self) -> Result<QteaModelBuildLimits, QteaModelBuildError> {
        if [
            self.max_package_bytes,
            self.max_spool_bytes,
            self.max_curvature_file_bytes,
        ]
        .contains(&0)
            || [
                self.max_manifest_bytes,
                self.max_directory_bytes,
                self.max_tensors,
                self.max_assets,
                self.max_asset_bytes,
                self.max_config_bytes,
                self.max_vector_elements,
                self.max_preserved_tensor_elements,
                self.max_tensor_bytes,
                self.max_validation_bytes,
                self.max_widened_tensor_bytes,
                self.max_native_work_bytes,
                self.max_tensor_work_bytes,
                self.max_fit_record_bytes,
            ]
            .contains(&0)
        {
            return Err(error("campaign limits must be positive"));
        }
        Ok(QteaModelBuildLimits {
            receiving: TritPackageReadLimits {
                max_source_bytes: self.max_package_bytes,
                max_manifest_bytes: self.max_manifest_bytes,
                max_directory_bytes: self.max_directory_bytes,
            },
            model: QteaModelReadLimits {
                max_tensors: self.max_tensors,
                max_assets: self.max_assets,
                max_asset_bytes: self.max_asset_bytes,
                max_config_bytes: self.max_config_bytes,
                max_vector_elements: self.max_vector_elements,
                tensor: QteaLimits {
                    max_tensor_bytes: self.max_tensor_bytes,
                    max_validation_bytes: self.max_validation_bytes,
                },
            },
            max_preserved_tensor_elements: self.max_preserved_tensor_elements,
            max_widened_tensor_bytes: self.max_widened_tensor_bytes,
            max_native_work_bytes: self.max_native_work_bytes,
            max_tensor_work_bytes: self.max_tensor_work_bytes,
            max_spool_bytes: self.max_spool_bytes,
            max_fit_record_bytes: self.max_fit_record_bytes,
        })
    }
}
/// Explicit inventory/runtime profile; no architecture inference fallback.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QteaCampaignModelProfile {
    /// Complete supported standard language architecture.
    Standard,
    /// Exact retained hybrid language/MTP graph, MTP remains Unverified.
    Qwen35Hybrid,
}
impl From<QteaCampaignModelProfile> for QteaModelProfile {
    fn from(value: QteaCampaignModelProfile) -> Self {
        match value {
            QteaCampaignModelProfile::Standard => Self::Standard,
            QteaCampaignModelProfile::Qwen35Hybrid => Self::Qwen35Hybrid,
        }
    }
}
/// Source-only preparation before real curvature/cache identities exist.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QteaFileSourceRequest {
    /// Must equal tritium.qtea.source-preparation.v1.
    pub schema: String,
    /// Declared campaign context, not original-source authentication.
    pub campaign: ContentId,
    /// Complete original-only plan.
    pub source: QuantPlan,
    /// Explicit supported graph profile.
    pub model_profile: QteaCampaignModelProfile,
    /// Same explicit positive model/work/output limits as subsequent campaigns.
    pub limits: QteaCampaignLimits,
}
fn validate_original(
    source: &QuantPlan,
    limits: QteaCampaignLimits,
    campaign: ContentId,
) -> Result<(), QteaModelBuildError> {
    source.validate().map_err(error)?;
    limits.build()?;
    if campaign.as_bytes() == &[0; 32] || source.tensors.len() > limits.max_tensors {
        return Err(error("source campaign identity/inventory mismatch"));
    }
    if source
        .tensors
        .iter()
        .any(|t| !matches!(t.action, QuantAction::KeepDense))
    {
        return Err(error(
            "QTEA observation plan requires original-only KeepDense actions",
        ));
    }
    Ok(())
}
impl QteaFileSourceRequest {
    /// Receive strict bounded source data, without placeholder curvature/run IDs.
    /// # Errors
    /// JSON cap/syntax/schema/source/limit/campaign refusal.
    pub fn from_slice(bytes: &[u8], max_bytes: usize) -> Result<Self, QteaModelBuildError> {
        if max_bytes == 0 || bytes.len() > max_bytes {
            return Err(error("source request exceeds positive JSON cap"));
        }
        let value: Self = serde_json::from_slice(bytes).map_err(error)?;
        value.validate()?;
        Ok(value)
    }
    fn validate(&self) -> Result<(), QteaModelBuildError> {
        if self.schema != "tritium.qtea.source-preparation.v1" {
            return Err(error("unsupported QTEA source schema"));
        }
        validate_original(&self.source, self.limits, self.campaign)
    }
}
/// Actual content observation after complete graph admission; data, not a lock or approval.
#[derive(Clone, Debug)]
pub struct QteaOriginalSourceObservation {
    /// Actual declared and graph-checked architecture.
    pub architecture: String,
    /// Actual original raw tensor/asset content identity.
    pub source_model: OriginalSourceId,
    /// Complete actual original tensor byte bindings.
    pub tensors: Vec<QuantSourceBinding>,
    /// Complete actual original asset byte bindings.
    pub assets: Vec<NamedAssetDigest>,
}
/// Observe actual files and admit source graph before any calibration exists.
/// # Errors
/// Invalid source, unexpected curvature paths, actual byte/config/graph or cap refusal.
pub fn prepare_qtea_file_source(
    r: &QteaFileSourceRequest,
    files: &QteaCampaignFiles,
) -> Result<QteaOriginalSourceObservation, QteaModelBuildError> {
    r.validate()?;
    if !files.curvature.is_empty() {
        return Err(error(
            "source-only preparation cannot ignore curvature files",
        ));
    }
    let mut raw = FileWeightSource::open(
        ModelQuantPlan::Uncalibrated(&r.source),
        &files.source_shards,
        files.assets.clone(),
        BTreeMap::new(),
    )
    .map_err(error)?;
    let observed = ObservedWeightSource::new(&r.source, &mut raw).map_err(error)?;
    let result = QteaOriginalSourceObservation {
        architecture: observed.architecture().to_owned(),
        source_model: observed.source_id(),
        tensors: observed.tensor_bindings().to_vec(),
        assets: observed.asset_bindings().to_vec(),
    };
    let mut builder = QteaModelBuilder::new(
        observed,
        r.model_profile.into(),
        r.campaign,
        Cursor::new(Vec::new()),
        r.limits.build()?,
    )?;
    builder.revalidate_original_source()?;
    Ok(result)
}
/// Imported data mode; neither variant authenticates its producer or capture.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum QteaCampaignCurvature {
    /// Existing canonical native Gram; never an automatic identity fallback.
    Native {
        /// Independently selected expected canonical Gram digest.
        digest: [u8; 32],
        /// Positive decoded f64 value capacity; excludes encoded input.
        max_value_bytes: usize,
    },
    /// Actual existing cache package reduced by the common collector.
    CalibrationCache {
        /// Exact selected cache identity.
        id: CalibrationCacheId,
        /// Existing logical cache limits.
        limits: CalibrationCacheLimits,
        /// Existing numerical collector budget.
        max_numeric_bytes: u64,
        /// Positive within-context replay chunk bound, not new model contexts.
        max_batch_samples: usize,
    },
}
/// Caller assertions, not measurements, authenticated machine identity or approval.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QteaCallerEstimate {
    /// Complete estimated resource vector; not observed usage.
    pub resources: ResourceVector,
    /// Caller-declared machine inventory content identity.
    pub machine_id: ContentId,
    /// Caller-declared estimator evidence content identity.
    pub evidence_id: ContentId,
}
impl QteaCallerEstimate {
    fn checked(self) -> Result<FrontierResourceEstimate, QteaModelBuildError> {
        FrontierResourceEstimate::new(self.resources, self.machine_id, self.evidence_id)
            .map_err(error)
    }
}
impl FrontierResourceEstimator for QteaCallerEstimate {
    fn estimate(&self, _: &SolverRequest) -> Result<FrontierResourceEstimate, FrontierSolverError> {
        FrontierResourceEstimate::new(self.resources, self.machine_id, self.evidence_id)
            .map_err(|e| FrontierSolverError::new(e.to_string()))
    }
}
/// One canonical matrix, with no alias/metric/recipe guessing.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QteaCampaignOwner {
    /// Original canonical owner name.
    pub name: String,
    /// Complete explicit numerical recipe; no serde defaults.
    pub recipe: TensorFitConfig,
    /// Explicit imported curvature mode.
    pub curvature: QteaCampaignCurvature,
    /// Exact fixed single-QTEA Experimental policy; no search/fallback.
    pub profile: FrontierProfile,
    /// Caller estimate subject to existing budget and known-work floor checks.
    pub estimate: QteaCallerEstimate,
}
/// Versioned content request; paths are deliberately separate operational inputs.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QteaFileCampaignRequest {
    /// Must equal tritium.qtea.file-campaign.v1.
    pub schema: String,
    /// Explicit campaign artifact identity, not original source custody.
    pub campaign: ContentId,
    /// Explicit frontier work run identity.
    pub run: ContentId,
    /// Complete original-only plan; actions must all be KeepDense.
    pub source: QuantPlan,
    /// Explicit supported model profile.
    pub model_profile: QteaCampaignModelProfile,
    /// Positive distinct per-domain caps.
    pub limits: QteaCampaignLimits,
    /// Exactly all matrix owners, in canonical source-name order.
    pub owners: Vec<QteaCampaignOwner>,
}
impl QteaFileCampaignRequest {
    /// Parse bounded strict request data, then validate inventories and policy.
    /// # Errors
    /// Positive JSON cap, syntax/schema/context/inventory/profile/estimate refusal.
    pub fn from_slice(bytes: &[u8], max_bytes: usize) -> Result<Self, QteaModelBuildError> {
        if max_bytes == 0 || bytes.len() > max_bytes {
            return Err(error("campaign request exceeds positive JSON cap"));
        }
        let value: Self = serde_json::from_slice(bytes).map_err(error)?;
        value.validate()?;
        Ok(value)
    }
    fn validate(&self) -> Result<(), QteaModelBuildError> {
        validate_original(&self.source, self.limits, self.campaign)?;
        if self.schema != "tritium.qtea.file-campaign.v1"
            || self.campaign.as_bytes() == &[0; 32]
            || self.run.as_bytes() == &[0; 32]
            || self.source.tensors.len() > self.limits.max_tensors
            || self.owners.is_empty()
        {
            return Err(error("campaign schema/identity/inventory mismatch"));
        }
        let expected = self
            .source
            .tensors
            .iter()
            .filter(|t| t.source.shape.len() == 2)
            .map(|t| t.source.name.as_str());
        if !self.owners.iter().map(|o| o.name.as_str()).eq(expected) {
            return Err(error(
                "campaign requires exact ordered matrix-owner inventory",
            ));
        }
        for owner in &self.owners {
            let p = &owner.profile;
            if p.ordering() != FrontierOrdering::Fixed
                || p.minimum_trust() != SolverTrust::Experimental
                || p.fallback_profile().is_some()
                || p.auto_select()
                || p.solver_ids().len() != 1
                || p.solver_ids()[0].as_str() != FRONTIER_QTEA_REFERENCE_SOLVER_ID
            {
                return Err(error(
                    "campaign requires fixed explicit Experimental QTEA policy without fallback",
                ));
            }
            owner.estimate.checked()?;
            match &owner.curvature {
                QteaCampaignCurvature::Native {
                    digest,
                    max_value_bytes,
                } if *digest == [0; 32] || *max_value_bytes == 0 => {
                    return Err(error(
                        "native curvature requires expected identity and positive capacity",
                    ));
                }
                QteaCampaignCurvature::CalibrationCache {
                    limits,
                    max_numeric_bytes,
                    max_batch_samples,
                    ..
                } if limits.max_payload_bytes == 0
                    || limits.max_samples == 0
                    || limits.max_windows == 0
                    || *max_numeric_bytes == 0
                    || *max_batch_samples == 0 =>
                {
                    return Err(error("calibration cache/replay limits must be positive"));
                }
                _ => {}
            }
        }
        Ok(())
    }
    fn id(&self) -> Result<ContentId, QteaModelBuildError> {
        Ok(ContentId::of_bytes(
            &serde_json::to_vec(self).map_err(error)?,
        ))
    }
}
/// Exact explicit file inventories. Caller owns immutable source/path custody.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QteaCampaignFiles {
    /// All actual original safetensors shards.
    pub source_shards: Vec<PathBuf>,
    /// Exactly source-plan assets.
    #[serde(deserialize_with = "unique_paths")]
    pub assets: BTreeMap<String, PathBuf>,
    /// Exactly one native Gram/cache file per matrix owner.
    #[serde(deserialize_with = "unique_paths")]
    pub curvature: BTreeMap<String, PathBuf>,
}
fn unique_paths<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<BTreeMap<String, PathBuf>, D::Error> {
    struct Unique;
    impl<'de> serde::de::Visitor<'de> for Unique {
        type Value = BTreeMap<String, PathBuf>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("unique named paths")
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut map: A,
        ) -> Result<Self::Value, A::Error> {
            let mut result = BTreeMap::new();
            while let Some((name, path)) = map.next_entry::<String, PathBuf>()? {
                if result.insert(name, path).is_some() {
                    return Err(serde::de::Error::custom("duplicate file binding"));
                }
            }
            Ok(result)
        }
    }
    d.deserialize_map(Unique)
}
enum Curvature {
    Native(Box<InputGram>),
    Calibrated(Box<CalibratedTensor>),
}
impl Curvature {
    fn borrowed(&self) -> QteaOwnerCurvature<'_> {
        match self {
            Self::Native(g) => QteaOwnerCurvature::Native(g),
            Self::Calibrated(c) => QteaOwnerCurvature::Calibrated(c),
        }
    }
}
impl QteaCampaignFiles {
    fn validate(&self, r: &QteaFileCampaignRequest) -> Result<(), QteaModelBuildError> {
        r.validate()?;
        if !self
            .curvature
            .keys()
            .map(String::as_str)
            .eq(r.owners.iter().map(|o| o.name.as_str()))
        {
            return Err(error("curvature file inventory does not match campaign"));
        }
        Ok(())
    }
    fn curvature(
        &self,
        r: &QteaFileCampaignRequest,
        o: &QteaCampaignOwner,
    ) -> Result<Curvature, QteaModelBuildError> {
        let bytes =
            read_original_file_bounded(&self.curvature[&o.name], r.limits.max_curvature_file_bytes)
                .map_err(error)?;
        match &o.curvature {
            QteaCampaignCurvature::Native {
                digest,
                max_value_bytes,
            } => Ok(Curvature::Native(Box::new(
                InputGram::decode_canonical(&bytes, *digest, *max_value_bytes).map_err(error)?,
            ))),
            QteaCampaignCurvature::CalibrationCache {
                id,
                limits,
                max_numeric_bytes,
                max_batch_samples,
            } => {
                let cache = CalibrationCache::decode_package(
                    &bytes,
                    *id,
                    *limits,
                    usize::try_from(r.limits.max_curvature_file_bytes).map_err(error)?,
                )
                .map_err(error)?;
                Ok(Curvature::Calibrated(Box::new(
                    cache
                        .collect_chunked(
                            CalibrationPartition::Full,
                            0.,
                            *max_numeric_bytes,
                            *max_batch_samples,
                        )
                        .map_err(error)?,
                )))
            }
        }
    }
}
/// Selected planning data, not deserialized opaque admission or authenticated approval.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QteaCampaignOwnerInspection {
    name: String,
    input_id: ContentId,
    plan_id: ContentId,
    request: FrontierStageRequest,
    logical_work_bytes: usize,
    encoded_bytes: usize,
}
/// Unqualified inspection output; execution reconstructs admission against real inputs.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QteaCampaignInspection {
    schema: String,
    qualification: bool,
    request_id: ContentId,
    source_model: OriginalSourceId,
    owners: Vec<QteaCampaignOwnerInspection>,
}
impl QteaCampaignInspection {
    /// Encode bounded planning data; no admission authority is serialized.
    /// # Errors
    /// Positive encoded capacity, allocation or JSON serialization refusal.
    pub fn to_json(&self, max_bytes: usize) -> Result<Vec<u8>, QteaModelBuildError> {
        if self.schema != "tritium.qtea.campaign-inspection.v1" || self.qualification {
            return Err(error("unsupported or qualified inspection data"));
        }
        encode(self, max_bytes)
    }
    /// Actual observed raw source content identity, not official-source authority.
    pub const fn source_model(&self) -> OriginalSourceId {
        self.source_model
    }
    /// Number of inspected matrix owners; no fitting occurred.
    pub fn owners(&self) -> usize {
        self.owners.len()
    }
    /// Receive bounded planning data; execution still validates every identity.
    /// # Errors
    /// Oversize/invalid JSON, unsupported schema or a qualification claim.
    pub fn from_slice(bytes: &[u8], max_bytes: usize) -> Result<Self, QteaModelBuildError> {
        if max_bytes == 0 || bytes.len() > max_bytes {
            return Err(error("inspection exceeds positive JSON cap"));
        }
        let value: Self = serde_json::from_slice(bytes).map_err(error)?;
        if value.schema != "tritium.qtea.campaign-inspection.v1" || value.qualification {
            return Err(error("unsupported or qualified inspection data"));
        }
        Ok(value)
    }
}
fn selection(
    r: &QteaFileCampaignRequest,
    o: &QteaCampaignOwner,
    w: &QteaTensorWorkPlan<'_>,
) -> Result<QteaSelectedAdmission, QteaModelBuildError> {
    let mut registry = SolverRegistry::new();
    registry
        .register(Arc::new(
            QteaReferenceAdapter::new()
                .map_err(error)?
                .with_resource_estimator(Arc::new(o.estimate)),
        ))
        .map_err(error)?;
    let m = w.metadata();
    let plan = registry
        .plan(
            &o.profile.solver_ids()[0],
            o.profile.minimum_trust(),
            &o.profile
                .request(m.rows as u64, m.columns as u64)
                .map_err(error)?,
        )
        .map_err(error)?;
    let request = FrontierStageRequest::new(
        r.run,
        r.campaign,
        w.origin().tensor_index,
        FrontierStage::Fit,
        &o.profile,
        &plan,
        w.input_id(),
    )
    .map_err(error)?;
    let expected_plan_id = plan.content_id();
    w.check_admission(&plan, &request, expected_plan_id)
        .map_err(error)?;
    Ok(QteaSelectedAdmission {
        plan,
        request,
        expected_plan_id,
    })
}
fn inspect_work(
    o: &QteaCampaignOwner,
    w: &QteaTensorWorkPlan<'_>,
    s: &QteaSelectedAdmission,
) -> QteaCampaignOwnerInspection {
    QteaCampaignOwnerInspection {
        name: o.name.clone(),
        input_id: w.input_id(),
        plan_id: s.expected_plan_id,
        request: s.request.clone(),
        logical_work_bytes: w.memory().total_logical_bytes,
        encoded_bytes: w.byte_ledger().total_bytes,
    }
}
/// Read/verify actual originals and curvature, without fitting or output/spool writes.
/// # Errors
/// Invalid request/files/source/curvature, caps or existing exact admission refusal.
pub fn inspect_qtea_file_campaign(
    r: &QteaFileCampaignRequest,
    files: &QteaCampaignFiles,
) -> Result<QteaCampaignInspection, QteaModelBuildError> {
    files.validate(r)?;
    let mut raw = FileWeightSource::open(
        ModelQuantPlan::Uncalibrated(&r.source),
        &files.source_shards,
        files.assets.clone(),
        BTreeMap::new(),
    )
    .map_err(error)?;
    let observed = ObservedWeightSource::new(&r.source, &mut raw).map_err(error)?;
    let source_model = observed.source_id();
    let mut builder = QteaModelBuilder::new(
        observed,
        r.model_profile.into(),
        r.campaign,
        Cursor::new(Vec::new()),
        r.limits.build()?,
    )?;
    let mut owners = Vec::new();
    owners.try_reserve_exact(r.owners.len()).map_err(error)?;
    for o in &r.owners {
        let curvature = files.curvature(r, o)?;
        owners.push(
            builder.inspect_owner(&o.name, curvature.borrowed(), o.recipe, |w| {
                let selected = selection(r, o, w)?;
                Ok(inspect_work(o, w, &selected))
            })?,
        );
    }
    builder.revalidate_original_source()?;
    Ok(QteaCampaignInspection {
        schema: "tritium.qtea.campaign-inspection.v1".into(),
        qualification: false,
        request_id: r.id()?,
        source_model,
        owners,
    })
}
/// Actual unqualified candidate publication; not measured resources or release approval.
#[derive(Debug, Serialize)]
pub struct QteaFileCampaignResult {
    schema: String,
    /// Always false; not a qualification receipt.
    qualification: bool,
    /// Actual committed package bytes identity.
    package_id: [u8; 32],
    /// Actual original raw source content identity.
    source_model: OriginalSourceId,
    /// Explicit post-commit publication warnings, never rollback.
    publication_warnings: Vec<String>,
}
impl QteaFileCampaignResult {
    /// Actual committed package identity; not quality or release acceptance.
    pub const fn package_id(&self) -> [u8; 32] {
        self.package_id
    }
    /// Observed original raw content identity, not official source authentication.
    pub const fn source_model(&self) -> OriginalSourceId {
        self.source_model
    }
    /// Explicit cleanup/sync warnings after successful commit, not rollback.
    pub fn publication_warnings(&self) -> &[String] {
        &self.publication_warnings
    }
    /// Encode bounded publication observations, not a qualification receipt.
    /// # Errors
    /// Positive encoded capacity, allocation or JSON serialization refusal.
    pub fn to_json(&self, max_bytes: usize) -> Result<Vec<u8>, QteaModelBuildError> {
        encode(self, max_bytes)
    }
}
/// Reobserve inputs, reconstruct opaque admission, fit and publish a full candidate.
/// Caller supplies an empty private spool and a new destination sink; immutable,
/// exclusive source/spool/directory custody remains caller scope. On error the
/// spool may contain private bytes and must be discarded. No adaptive retry.
/// # Errors
/// Wrong/stale selected request/source/work/plan, source/curvature/cap/fit/I/O or
/// existing publication refusal. Successful storage is not qualification.
pub fn run_qtea_file_campaign<S: Read + Write + Seek>(
    r: &QteaFileCampaignRequest,
    files: &QteaCampaignFiles,
    expected: &QteaCampaignInspection,
    spool: S,
    sink: &mut DirectoryTensorSink,
) -> Result<QteaFileCampaignResult, QteaModelBuildError> {
    files.validate(r)?;
    if expected.schema != "tritium.qtea.campaign-inspection.v1"
        || expected.qualification
        || expected.request_id != r.id()?
        || !expected
            .owners
            .iter()
            .map(|o| o.name.as_str())
            .eq(r.owners.iter().map(|o| o.name.as_str()))
    {
        return Err(error("selected inspection does not match campaign request"));
    }
    let mut raw = FileWeightSource::open(
        ModelQuantPlan::Uncalibrated(&r.source),
        &files.source_shards,
        files.assets.clone(),
        BTreeMap::new(),
    )
    .map_err(error)?;
    let observed = ObservedWeightSource::new(&r.source, &mut raw).map_err(error)?;
    let source_model = observed.source_id();
    if source_model != expected.source_model {
        return Err(error("selected original source identity changed"));
    }
    let mut builder = QteaModelBuilder::new(
        observed,
        r.model_profile.into(),
        r.campaign,
        spool,
        r.limits.build()?,
    )?;
    for (o, e) in r.owners.iter().zip(&expected.owners) {
        let curvature = files.curvature(r, o)?;
        builder.fit_owner(&o.name, curvature.borrowed(), o.recipe, |w| {
            let selected = selection(r, o, w)?;
            let actual = inspect_work(o, w, &selected);
            if actual.input_id != e.input_id
                || actual.plan_id != e.plan_id
                || actual.request != e.request
                || actual.logical_work_bytes != e.logical_work_bytes
                || actual.encoded_bytes != e.encoded_bytes
            {
                return Err(error("actual work differs from selected inspection"));
            }
            Ok(selected)
        })?;
    }
    let (built, warnings) = builder.publish_candidate(sink).map_err(error)?;
    Ok(QteaFileCampaignResult {
        schema: "tritium.qtea.campaign-result.v1".into(),
        qualification: false,
        package_id: *built.package_id().as_bytes(),
        source_model,
        publication_warnings: warnings.iter().map(|w| format!("{w:?}")).collect(),
    })
}

/// File adapter shared by CLI/Python. Creates only a new private spool file;
/// removes it after the generic runner closes it, including failed work. Caller
/// owns immutable/exclusive source, spool path and destination-parent custody.
/// No adversarial path-writer or power-loss cleanup guarantee is claimed.
/// # Errors
/// Existing destination/spool, creation, ordinary runner or failed cleanup. After
/// successful publication, cleanup failure is a warning, never rollback.
pub fn run_qtea_file_campaign_to_directory(
    r: &QteaFileCampaignRequest,
    files: &QteaCampaignFiles,
    expected: &QteaCampaignInspection,
    spool_path: &std::path::Path,
    output: &std::path::Path,
) -> Result<QteaFileCampaignResult, QteaModelBuildError> {
    match std::fs::symlink_metadata(output) {
        Ok(_) => return Err(error("campaign destination already exists")),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(error(e)),
    }
    let spool = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(spool_path)
        .map_err(error)?;
    let result = run_qtea_file_campaign(
        r,
        files,
        expected,
        spool,
        &mut DirectoryTensorSink::new(output),
    );
    let cleanup = std::fs::remove_file(spool_path);
    match (result, cleanup) {
        (Ok(mut value), Err(e)) => {
            value
                .publication_warnings
                .push(format!("private spool cleanup failed: {e}"));
            Ok(value)
        }
        (Err(e), Err(cleanup)) => Err(error(format!(
            "{e}; private spool cleanup failed: {cleanup}"
        ))),
        (result, Ok(())) => result,
    }
}
