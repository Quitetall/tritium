//! Actual declared-statistics A->B source stage, without a fake fitting event.
use super::model::{bound, source_binding};
use super::{
    InMemoryWeightSource, QuantEngineError, RawTensor, SalienceFoldError, SaliencePair,
    WeightSource,
};
use std::{borrow::Cow, collections::BTreeMap};
use tritium_format::trit_model::decode_dense_f32;
use tritium_schema::{
    BlobId, NamedAssetDigest, QuantSourceTensor, SalienceInverseAxis, SalienceMomentBinding,
    SalienceNumerics, SalienceOperationOutput, SalienceStatisticsKind,
    SalienceTransformObservation, SalienceTransformPlan, SalienceTransformPlanError,
    ScalePrecision,
};

/// Actual supplied vectors and declared rows, not proof of graph capture.
#[derive(Clone, Debug)]
pub struct DeclaredSalienceStatistics {
    /// Raw per-channel f64 second-moment sums.
    pub moments: Vec<f64>,
    /// Explicit positive declared sample rows.
    pub rows: u64,
}
impl DeclaredSalienceStatistics {
    /// Bind actual raw moment bytes and declared interpretation without a copied preimage.
    /// # Errors
    /// Empty, negative/nonfinite moments or no declared rows.
    pub fn binding(&self) -> Result<SalienceMomentBinding, SalienceSourceError> {
        if self.rows == 0
            || self.moments.is_empty()
            || self.moments.iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(SalienceSourceError::Statistics);
        }
        let mut hash = blake3::Hasher::new();
        for value in &self.moments {
            hash.update(&value.to_le_bytes());
        }
        Ok(SalienceMomentBinding {
            content: BlobId::from_bytes(*hash.finalize().as_bytes()),
            channels: u64::try_from(self.moments.len())
                .map_err(|_| SalienceSourceError::Statistics)?,
            rows: self.rows,
        })
    }
}

/// Failure returns no derived source and never mutates caller-owned source bytes.
#[non_exhaustive]
#[derive(Debug)]
pub enum SalienceSourceError {
    /// Typed complete request/geometry/logical resource rejection.
    Plan(SalienceTransformPlanError),
    /// Actual source/payload/identity/allocation rejection from shared machinery.
    Source(QuantEngineError),
    /// Actual declared moment vectors/counts do not match the complete request.
    Statistics,
    /// The shared paired numerical operation rejected before changing a pair.
    Fold(SalienceFoldError),
    /// Unsupported policy; never silently substitute one.
    Unsupported,
}
impl core::fmt::Display for SalienceSourceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "salience source rejected: {self:?}")
    }
}
impl std::error::Error for SalienceSourceError {}
impl From<QuantEngineError> for SalienceSourceError {
    fn from(error: QuantEngineError) -> Self {
        Self::Source(error)
    }
}

/// Private-field immutable B source, usable by the existing fitting/cache seam.
///
/// Holds one complete derived payload in memory. Not a durable staged package,
/// sealed log, source lock, full RSS admission or graph-capture qualification.
#[derive(Debug)]
pub struct DerivedSalienceSource {
    source: InMemoryWeightSource,
    observation: SalienceTransformObservation,
}
impl DerivedSalienceSource {
    /// Actual complete A/B observation; returned metadata cannot mutate this source.
    #[must_use]
    pub fn observation(&self) -> &SalienceTransformObservation {
        &self.observation
    }
}
impl WeightSource for DerivedSalienceSource {
    fn inventory(&self) -> Result<Vec<QuantSourceTensor>, QuantEngineError> {
        self.source.inventory()
    }
    fn asset_names(&self) -> Vec<String> {
        self.source.asset_names()
    }
    fn tensor(&mut self, name: &str, max_bytes: usize) -> Result<Cow<'_, [u8]>, QuantEngineError> {
        self.source.tensor(name, max_bytes)
    }
    fn asset(&mut self, name: &str, max_bytes: usize) -> Result<Cow<'_, [u8]>, QuantEngineError> {
        self.source.asset(name, max_bytes)
    }
    // Old A Grams are deliberately not copied into B. Actual B-bound capture is separate.
    fn input_gram(
        &mut self,
        name: &str,
        max_entries: usize,
    ) -> Result<Cow<'_, [f64]>, QuantEngineError> {
        self.source.input_gram(name, max_entries)
    }
}

/// Shared complete paired source controller; graph adapters supply actual named pairs.
#[derive(Debug)]
pub struct SourceTransform;
impl SourceTransform {
    /// Consume actual A bytes, check identity/moments, apply ordered paired numerics,
    /// then return one immutable B with complete actual raw bindings.
    ///
    /// Does not infer graph eligibility, claim captured statistics or publish a
    /// package. Existing B->C fitting alone does not admit complete A->B->C lineage.
    /// # Errors
    /// Plan/inventory/actual source/moments/bounds/policy/numerical errors. No B is returned.
    pub fn run(
        plan: &SalienceTransformPlan,
        source: &mut dyn WeightSource,
        statistics: &BTreeMap<String, DeclaredSalienceStatistics>,
    ) -> Result<DerivedSalienceSource, SalienceSourceError> {
        plan.validate().map_err(SalienceSourceError::Plan)?;
        match (plan.numerics, plan.statistics_kind) {
            (SalienceNumerics::LegacyF64ScaleF32V1, SalienceStatisticsKind::Declared) => {}
            _ => return Err(SalienceSourceError::Unsupported),
        }
        check_inventory(plan, source)?;
        if statistics.len() != plan.operations.len() {
            return Err(SalienceSourceError::Statistics);
        }
        for op in &plan.operations {
            if statistics
                .get(&op.name)
                .ok_or(SalienceSourceError::Statistics)?
                .binding()?
                != op.moments
            {
                return Err(SalienceSourceError::Statistics);
            }
        }
        let mut derived = InMemoryWeightSource::default();
        let mut assets = Vec::new();
        let mut remaining = plan.limits.max_asset_bytes;
        for name in &plan.assets {
            let bytes = source.asset(name, platform(remaining)?)?;
            bound(bytes.len() as u64, remaining, "transform assets")?;
            remaining -= bytes.len() as u64;
            assets.push(NamedAssetDigest {
                name: name.clone(),
                content: BlobId::from_bytes(*blake3::hash(&bytes).as_bytes()),
            });
            derived.assets.insert(name.clone(), copy_bytes(&bytes)?);
        }
        let mut originals = Vec::new();
        for tensor in &plan.tensors {
            let bytes = source.tensor(
                &tensor.name,
                platform(tensor.byte_len().map_err(QuantEngineError::Plan)?)?,
            )?;
            originals.push(source_binding(tensor, &bytes)?);
            derived.tensors.insert(
                tensor.name.clone(),
                RawTensor {
                    shape: tensor.shape.clone(),
                    precision: tensor.precision,
                    bytes: copy_bytes(&bytes)?,
                },
            );
        }
        check_inventory(plan, source)?;
        let original_source =
            tritium_evidence::original_source_id(&plan.architecture, &originals, &assets)
                .map_err(QuantEngineError::Plan)?;
        if original_source != plan.original_source {
            return Err(
                QuantEngineError::Binding("transform consumed original source differs").into(),
            );
        }
        let mut operations = Vec::new();
        for op in &plan.operations {
            let (inverse_shape, mut inverse) = take_decoded(&mut derived, &op.inverse)?;
            let mut projections = Vec::new();
            let mut shapes = Vec::new();
            for name in &op.projections {
                let (shape, values) = take_decoded(&mut derived, name)?;
                shapes.push(shape);
                projections.push(values);
            }
            let columns = match op.inverse_axis {
                SalienceInverseAxis::NormChannels => 1,
                SalienceInverseAxis::LinearOutputRows => platform(inverse_shape[1])?,
                _ => return Err(SalienceSourceError::Unsupported),
            };
            let values = &statistics[&op.name];
            let scales = SaliencePair {
                projections: &mut projections
                    .iter_mut()
                    .map(Vec::as_mut_slice)
                    .collect::<Vec<_>>(),
                inverse: &mut inverse,
                inverse_columns: columns,
            }
            .apply(
                &values.moments,
                platform(values.rows)?,
                f64::from_le_bytes(op.alpha_bits),
            )
            .map_err(SalienceSourceError::Fold)?;
            store(&mut derived, &op.inverse, inverse_shape, &inverse)?;
            for ((name, shape), values) in op.projections.iter().zip(shapes).zip(projections) {
                store(&mut derived, name, shape, &values)?;
            }
            operations.push(SalienceOperationOutput {
                operation: op.name.clone(),
                scale_bits: scales.into_iter().map(f32::to_le_bytes).collect(),
            });
        }
        let mut outputs = Vec::new();
        for (name, tensor) in &derived.tensors {
            outputs.push(source_binding(
                &QuantSourceTensor {
                    name: name.clone(),
                    shape: tensor.shape.clone(),
                    precision: tensor.precision,
                },
                &tensor.bytes,
            )?);
        }
        let derived_source =
            tritium_evidence::original_source_id(&plan.architecture, &outputs, &assets)
                .map_err(QuantEngineError::Plan)?;
        Ok(DerivedSalienceSource {
            source: derived,
            observation: SalienceTransformObservation {
                plan: plan.clone(),
                original_source,
                derived_source,
                original_tensors: originals,
                derived_tensors: outputs,
                assets,
                operations,
            },
        })
    }
}
fn check_inventory(
    plan: &SalienceTransformPlan,
    source: &dyn WeightSource,
) -> Result<(), SalienceSourceError> {
    if source.inventory()? != plan.tensors || source.asset_names() != plan.assets {
        return Err(
            QuantEngineError::Binding("transform complete source inventory differs").into(),
        );
    }
    Ok(())
}
fn platform(value: u64) -> Result<usize, QuantEngineError> {
    usize::try_from(value).map_err(|_| QuantEngineError::Limit("transform platform size"))
}
fn copy_bytes(bytes: &[u8]) -> Result<Vec<u8>, QuantEngineError> {
    let mut copy = Vec::new();
    copy.try_reserve_exact(bytes.len())
        .map_err(|_| QuantEngineError::Allocation)?;
    copy.extend_from_slice(bytes);
    Ok(copy)
}
fn take_decoded(
    source: &mut InMemoryWeightSource,
    name: &str,
) -> Result<(Vec<u64>, Vec<f32>), QuantEngineError> {
    let tensor = source
        .tensors
        .remove(name)
        .ok_or(QuantEngineError::Binding("transform member missing"))?;
    let values = decode_dense_f32(name, &tensor.shape, tensor.precision, &tensor.bytes)?;
    Ok((tensor.shape, values))
}
fn store(
    source: &mut InMemoryWeightSource,
    name: &str,
    shape: Vec<u64>,
    values: &[f32],
) -> Result<(), QuantEngineError> {
    let mut bytes = Vec::new();
    let size = values
        .len()
        .checked_mul(4)
        .ok_or(QuantEngineError::Limit("transform output bytes"))?;
    bytes
        .try_reserve_exact(size)
        .map_err(|_| QuantEngineError::Allocation)?;
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    source.tensors.insert(
        name.into(),
        RawTensor {
            shape,
            precision: ScalePrecision::F32,
            bytes,
        },
    );
    Ok(())
}
