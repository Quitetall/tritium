//! Mandatory dense-stage/log binding and borrowed B adapter, not captured-graph admission.
use super::model::{bound, source_binding};
use super::{
    DeclaredSalienceStatistics, DirectoryTensorSink, PublicationWarning, QuantEngineError,
    RecordedSalienceSource, SalienceSourceError, SourceTransform, WeightSource,
};
use std::{borrow::Cow, collections::BTreeMap, io::Write};
use tritium_core::DType;
use tritium_evidence::{SalienceTransformView, VerifiedLog};
use tritium_format::trit_salience::{
    decode_salience_stage_manifest, encode_salience_stage_manifest,
};
use tritium_format::{
    TritBlob, TritBlobKind, TritPackage, TritPackageWriteError,
    read_trit_package_with_manifest_limit, write_trit_package_bounded, write_trit_package_to,
};
use tritium_schema::{
    BlobId, OriginalSourceId, PackageId, QuantSourceTensor, SalienceSourceStageManifest,
    SalienceStageLimits, ScalePrecision, SchemaId, salience_moment_asset,
};

/// Actual no-clobber stage publication, not arithmetic or release qualification.
#[derive(Debug)]
pub struct PublishedSalienceStage {
    /// Exact complete stage package identity, distinct from B's source identity.
    pub package_id: PackageId,
    /// Explicit post-commit cleanup/directory sync warnings.
    pub warnings: Vec<PublicationWarning>,
}

impl RecordedSalienceSource {
    /// Encode the actual immutable B, original assets and supplied bound moments.
    /// Uses the same common layout as writer export; retains the complete output.
    /// # Errors
    /// Source/log/statistic disagreement, metadata/physical/logical caps or allocation.
    pub fn encode_stage_package(
        &self,
        statistics: &BTreeMap<String, DeclaredSalienceStatistics>,
        limits: SalienceStageLimits,
    ) -> Result<Vec<u8>, SalienceSourceError> {
        self.with_stage_package(statistics, limits, |manifest, blobs| {
            write_trit_package_bounded(manifest, blobs, size(limits.max_package_bytes)?)
                .map_err(QuantEngineError::from)
                .map_err(SalienceSourceError::from)
        })
    }

    /// Stream the same package, without retaining a complete second output buffer.
    /// Retains B, encoded moments and metadata; not full RSS or out-of-core admission.
    /// A failed writer may hold a prefix and never returns a success identity.
    /// # Errors
    /// Source/log/statistic/cap disagreement or destination I/O failure.
    pub fn write_stage_package_to(
        &self,
        statistics: &BTreeMap<String, DeclaredSalienceStatistics>,
        limits: SalienceStageLimits,
        destination: &mut impl Write,
    ) -> Result<PackageId, SalienceSourceError> {
        self.with_stage_package(statistics, limits, |manifest, blobs| {
            write_trit_package_to(
                manifest,
                blobs,
                size(limits.max_package_bytes)?,
                destination,
            )
            .map(id)
            .map_err(write_error)
            .map_err(SalienceSourceError::from)
        })
    }

    /// Publish actual stage and log through the existing shared directory commit protocol.
    /// An existing destination is never overwritten; failure exposes no admitted stage.
    /// # Errors
    /// Input/statistic/log/cap mismatch or pre-commit filesystem/write failure.
    pub fn publish_stage(
        &self,
        statistics: &BTreeMap<String, DeclaredSalienceStatistics>,
        limits: SalienceStageLimits,
        destination: &mut DirectoryTensorSink,
    ) -> Result<PublishedSalienceStage, SalienceSourceError> {
        self.with_stage_package(statistics, limits, |manifest, blobs| {
            let maximum = size(limits.max_package_bytes)?;
            let (package_id, warnings) =
                destination.publish_with_writer(&self.evidence, |file| {
                    write_trit_package_to(manifest, blobs, maximum, file)
                        .map(id)
                        .map_err(write_error)
                })?;
            Ok(PublishedSalienceStage {
                package_id,
                warnings,
            })
        })
    }

    fn with_stage_package<R>(
        &self,
        statistics: &BTreeMap<String, DeclaredSalienceStatistics>,
        limits: SalienceStageLimits,
        emit: impl FnOnce(&[u8], &[TritBlob<'_>]) -> Result<R, SalienceSourceError>,
    ) -> Result<R, SalienceSourceError> {
        let view = one_view(self.evidence.log())?;
        if &view.completed().observation != self.source.observation() {
            return Err(binding("stage source and recorded transform disagree"));
        }
        let manifest = SalienceSourceStageManifest {
            schema: SchemaId { major: 1, minor: 0 },
            transform: view.reference().clone(),
            observation: self.source.observation().clone(),
        };
        manifest
            .validate_limits(limits)
            .map_err(SalienceSourceError::Plan)?;
        let encoded = encode_salience_stage_manifest(&manifest, size(limits.max_manifest_bytes)?)
            .map_err(QuantEngineError::from)?;
        let source = self.source.raw_source();
        let mut blobs = Vec::new();
        let mut asset_bytes = 0u64;
        for tensor_binding in &manifest.observation.derived_tensors {
            let tensor = source
                .tensors
                .get(&tensor_binding.source.name)
                .ok_or_else(|| binding("missing actual derived tensor"))?;
            if source_binding(&tensor_binding.source, &tensor.bytes)? != *tensor_binding {
                return Err(binding("actual derived content changed"));
            }
            blobs.push(TritBlob {
                kind: dense_kind(tensor_binding.source.precision)?,
                bytes: &tensor.bytes,
            });
        }
        for asset in &manifest.observation.assets {
            let bytes = source
                .assets
                .get(&asset.name)
                .ok_or_else(|| binding("missing actual original asset"))?;
            asset_bytes = asset_bytes
                .checked_add(bytes.len() as u64)
                .ok_or_else(|| binding("asset size overflow"))?;
            if *blake3::hash(bytes).as_bytes() != *asset.content.as_bytes() {
                return Err(binding("actual original asset changed"));
            }
            blobs.push(TritBlob {
                kind: TritBlobKind::Asset(asset.name.clone()),
                bytes,
            });
        }
        bound(asset_bytes, limits.max_asset_bytes, "stage asset bytes")?;
        if statistics.len() != manifest.observation.plan.operations.len() {
            return Err(SalienceSourceError::Statistics);
        }
        let mut moments = Vec::new();
        for (index, operation) in manifest.observation.plan.operations.iter().enumerate() {
            let values = statistics
                .get(&operation.name)
                .ok_or(SalienceSourceError::Statistics)?;
            if values.binding()? != operation.moments {
                return Err(SalienceSourceError::Statistics);
            }
            let count = size(
                operation
                    .moments
                    .channels
                    .checked_mul(8)
                    .ok_or_else(|| binding("statistic size overflow"))?,
            )?;
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(count)
                .map_err(|_| QuantEngineError::Allocation)?;
            for value in &values.moments {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            moments.push((salience_moment_asset(index), bytes));
        }
        blobs.extend(moments.iter().map(|(name, bytes)| TritBlob {
            kind: TritBlobKind::Asset(name.clone()),
            bytes,
        }));
        emit(&encoded, &blobs)
    }
}

/// Borrowed immutable stage implementing the existing source seam without dense copies.
/// Requires the actual complete transform log. Opening proves actual B/log/structure
/// agreement, not transform arithmetic, executed capture, source locks or full RSS.
#[derive(Debug)]
pub struct StagedSalienceSource<'a> {
    package: TritPackage<'a>,
    manifest: SalienceSourceStageManifest,
    log: &'a VerifiedLog,
    tensors: BTreeMap<String, usize>,
    assets: BTreeMap<String, usize>,
    moments: Vec<usize>,
}
impl<'a> StagedSalienceSource<'a> {
    /// Check exact profile, receiving caps, mandatory log, every payload and actual B identity.
    /// Unknown or extra directory entries fail; deduplicated content is allowed.
    /// # Errors
    /// Malformed/foreign/incomplete stage/log, changed payload or any receiving cap.
    pub fn open(
        bytes: &'a [u8],
        log: &'a VerifiedLog,
        limits: SalienceStageLimits,
    ) -> Result<Self, SalienceSourceError> {
        bound(
            bytes.len() as u64,
            limits.max_package_bytes,
            "stage package bytes",
        )?;
        let package =
            read_trit_package_with_manifest_limit(bytes, size(limits.max_manifest_bytes)?)
                .map_err(QuantEngineError::from)?;
        let manifest =
            decode_salience_stage_manifest(package.manifest(), size(limits.max_manifest_bytes)?)
                .map_err(QuantEngineError::from)?;
        manifest
            .validate_limits(limits)
            .map_err(SalienceSourceError::Plan)?;
        let view = one_view(log)?;
        if manifest.transform != *view.reference()
            || manifest.observation != view.completed().observation
        {
            return Err(binding(
                "stage manifest differs from mandatory transform log",
            ));
        }
        let mut index = BTreeMap::new();
        for (i, blob) in package.blobs().iter().enumerate() {
            if index.insert(key(blob.id, &blob.kind)?, i).is_some() {
                return Err(binding("duplicate stage directory binding"));
            }
        }
        let mut tensors = BTreeMap::new();
        let mut actual_tensors = Vec::new();
        // A shared content/kind entry may be referenced by multiple logical tensors.
        let mut used = std::collections::BTreeSet::new();
        for tensor in &manifest.observation.derived_tensors {
            let i = *index
                .get(&key(tensor.content, &dense_kind(tensor.source.precision)?)?)
                .ok_or_else(|| binding("missing stage tensor binding"))?;
            let payload = package
                .blob(&package.blobs()[i])
                .map_err(QuantEngineError::from)?;
            if source_binding(&tensor.source, payload)? != *tensor {
                return Err(binding("actual stage tensor content changed"));
            }
            actual_tensors.push(tensor.clone());
            tensors.insert(tensor.source.name.clone(), i);
            used.insert(i);
        }
        let mut assets = BTreeMap::new();
        let mut asset_bytes = 0u64;
        for asset in &manifest.observation.assets {
            let i = *index
                .get(&key(
                    asset.content,
                    &TritBlobKind::Asset(asset.name.clone()),
                )?)
                .ok_or_else(|| binding("missing stage asset binding"))?;
            let payload = package
                .blob(&package.blobs()[i])
                .map_err(QuantEngineError::from)?;
            asset_bytes = asset_bytes
                .checked_add(payload.len() as u64)
                .ok_or_else(|| binding("asset size overflow"))?;
            assets.insert(asset.name.clone(), i);
            used.insert(i);
        }
        bound(asset_bytes, limits.max_asset_bytes, "stage asset bytes")?;
        let mut moments = Vec::new();
        for (op, request) in manifest.observation.plan.operations.iter().enumerate() {
            let i = *index
                .get(&key(
                    request.moments.content,
                    &TritBlobKind::Asset(salience_moment_asset(op)),
                )?)
                .ok_or_else(|| binding("missing stage moment binding"))?;
            let payload = package
                .blob(&package.blobs()[i])
                .map_err(QuantEngineError::from)?;
            let count = request
                .moments
                .channels
                .checked_mul(8)
                .ok_or_else(|| binding("statistic size overflow"))?;
            if payload.len() as u64 != count
                || payload.as_chunks::<8>().0.iter().any(|b| {
                    let v = f64::from_le_bytes(*b);
                    !v.is_finite() || v < 0.0
                })
            {
                return Err(SalienceSourceError::Statistics);
            }
            moments.push(i);
            used.insert(i);
        }
        if used.len() != index.len() {
            return Err(binding("extra stage directory binding"));
        }
        let actual = tritium_evidence::original_source_id(
            &manifest.observation.plan.architecture,
            &actual_tensors,
            &manifest.observation.assets,
        )
        .map_err(QuantEngineError::Plan)?;
        if actual != manifest.observation.derived_source {
            return Err(binding("actual stage B identity differs"));
        }
        Ok(Self {
            package,
            manifest,
            log,
            tensors,
            assets,
            moments,
        })
    }

    /// Checked immutable stage metadata, not independent arithmetic truth.
    #[must_use]
    pub fn manifest(&self) -> &SalienceSourceStageManifest {
        &self.manifest
    }

    /// Exact physical stage identity, not B's source identity or a release verdict.
    #[must_use]
    pub fn package_id(&self) -> PackageId {
        id(self.package.package_id())
    }

    /// Actual complete B identity checked against bytes during open.
    #[must_use]
    pub fn source_id(&self) -> OriginalSourceId {
        self.manifest.observation.derived_source
    }

    /// Reconstruct only the bound raw moment vectors; retains no dense tensor shadow.
    /// # Errors
    /// Payload read failure; stage open already admitted scalar counts and logical caps.
    pub fn statistics(
        &self,
    ) -> Result<BTreeMap<String, DeclaredSalienceStatistics>, SalienceSourceError> {
        let mut values = BTreeMap::new();
        for (request, &i) in self
            .manifest
            .observation
            .plan
            .operations
            .iter()
            .zip(&self.moments)
        {
            let bytes = self
                .package
                .blob(&self.package.blobs()[i])
                .map_err(QuantEngineError::from)?;
            let mut moments = Vec::new();
            moments
                .try_reserve_exact(bytes.len() / 8)
                .map_err(|_| QuantEngineError::Allocation)?;
            moments.extend(
                bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| f64::from_le_bytes(*b)),
            );
            values.insert(
                request.name.clone(),
                DeclaredSalienceStatistics {
                    moments,
                    rows: request.moments.rows,
                },
            );
        }
        Ok(values)
    }

    /// Re-execute actual A->B numerics using the durable stored moment vectors.
    /// This materializes the existing in-memory derived source temporarily;
    /// it is not out-of-core capture, model-quality or release qualification.
    /// # Errors
    /// Actual A/moment/numerical replay disagrees with the recorded complete observation.
    pub fn verify_against_original(
        &self,
        original: &mut dyn WeightSource,
    ) -> Result<(), SalienceSourceError> {
        let actual = SourceTransform::replay(self.log, original, &self.statistics()?)?;
        if actual.observation() != &self.manifest.observation {
            return Err(SalienceSourceError::Replay);
        }
        Ok(())
    }
}
impl WeightSource for StagedSalienceSource<'_> {
    fn inventory(&self) -> Result<Vec<QuantSourceTensor>, QuantEngineError> {
        Ok(self
            .manifest
            .observation
            .derived_tensors
            .iter()
            .map(|t| t.source.clone())
            .collect())
    }
    fn asset_names(&self) -> Vec<String> {
        self.assets.keys().cloned().collect()
    }
    fn tensor(&mut self, name: &str, maximum: usize) -> Result<Cow<'_, [u8]>, QuantEngineError> {
        payload(&self.package, &self.tensors, name, maximum)
    }
    fn asset(&mut self, name: &str, maximum: usize) -> Result<Cow<'_, [u8]>, QuantEngineError> {
        payload(&self.package, &self.assets, name, maximum)
    }
    fn input_gram(
        &mut self,
        _name: &str,
        _maximum: usize,
    ) -> Result<Cow<'_, [f64]>, QuantEngineError> {
        Err(QuantEngineError::Source(
            "no captured B Gram in declared source stage".into(),
        ))
    }
}
fn payload<'a>(
    package: &'a TritPackage<'_>,
    index: &BTreeMap<String, usize>,
    name: &str,
    maximum: usize,
) -> Result<Cow<'a, [u8]>, QuantEngineError> {
    let &i = index
        .get(name)
        .ok_or_else(|| QuantEngineError::Source("missing stage binding".into()))?;
    let bytes = package.blob(&package.blobs()[i])?;
    bound(
        bytes.len() as u64,
        maximum as u64,
        "stage requested payload",
    )?;
    Ok(Cow::Borrowed(bytes))
}
fn one_view(log: &VerifiedLog) -> Result<SalienceTransformView, SalienceSourceError> {
    let mut views = SalienceTransformView::fold(log)?;
    if views.len() != 1 || log.events.len() != 2 {
        return Err(binding("expected standalone complete transform log"));
    }
    Ok(views.remove(0))
}
fn size(n: u64) -> Result<usize, SalienceSourceError> {
    usize::try_from(n).map_err(|_| QuantEngineError::Limit("stage platform byte size").into())
}
fn id(p: tritium_format::PackageId) -> PackageId {
    PackageId::from_bytes(*p.as_bytes())
}
fn binding(message: &'static str) -> SalienceSourceError {
    QuantEngineError::Binding(message).into()
}
fn write_error(e: TritPackageWriteError) -> QuantEngineError {
    match e {
        TritPackageWriteError::Package(e) => e.into(),
        TritPackageWriteError::Io(e) => e.into(),
        _ => QuantEngineError::Binding("unsupported package writer failure"),
    }
}
fn dense_kind(precision: ScalePrecision) -> Result<TritBlobKind, SalienceSourceError> {
    Ok(TritBlobKind::DenseTensor(match precision {
        ScalePrecision::F16 => DType::F16,
        ScalePrecision::Bf16 => DType::BF16,
        ScalePrecision::F32 => DType::F32,
        _ => return Err(binding("unsupported stage precision")),
    }))
}
fn key(
    content: BlobId,
    kind: &TritBlobKind,
) -> Result<([u8; 32], u8, String), SalienceSourceError> {
    let (tag, name) = match kind {
        TritBlobKind::Asset(name) => (0, name.clone()),
        TritBlobKind::DenseTensor(DType::F16) => (1, String::new()),
        TritBlobKind::DenseTensor(DType::BF16) => (2, String::new()),
        TritBlobKind::DenseTensor(DType::F32) => (3, String::new()),
        _ => return Err(binding("unsupported stage blob kind")),
    };
    Ok((*content.as_bytes(), tag, name))
}
