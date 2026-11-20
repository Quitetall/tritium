//! Observe once and check each later raw read, without retaining dense payloads.
use super::model::{bound, check_source_inventory, platform, source_binding};
use super::{QuantEngineError, WeightSource};
use std::borrow::Cow;
use tritium_schema::{
    BlobId, ModelQuantPlan, NamedAssetDigest, OriginalSourceId, QuantPlan, QuantSourceBinding,
    QuantSourceTensor,
};

/// Borrowed raw source with controller-derived, immutable expected bindings.
/// Every tensor/asset read checks exact observed bytes before returning them.
/// Metadata is retained, not another full raw model. This does not freeze files,
/// certify input Grams, execute calibration or admit full process RSS.
pub struct ObservedWeightSource<'a> {
    source: &'a mut dyn WeightSource,
    architecture: String,
    identity: OriginalSourceId,
    tensors: Vec<QuantSourceBinding>,
    assets: Vec<NamedAssetDigest>,
    asset_lengths: Vec<u64>,
}

impl std::fmt::Debug for ObservedWeightSource<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObservedWeightSource")
            .field("identity", &self.identity)
            .field("tensor_count", &self.tensors.len())
            .field("asset_count", &self.assets.len())
            .finish_non_exhaustive()
    }
}

impl<'a> ObservedWeightSource<'a> {
    /// Observe complete actual raw contents with the existing plan and limits.
    /// Actions are validated but not executed; no fit/package/evidence is emitted.
    /// # Errors
    /// Invalid plan/inventory, raw payload, source I/O or logical byte limits.
    pub fn new(
        plan: &QuantPlan,
        source: &'a mut dyn WeightSource,
    ) -> Result<Self, QuantEngineError> {
        plan.validate().map_err(QuantEngineError::Plan)?;
        check_source_inventory(ModelQuantPlan::Uncalibrated(plan), source)?;
        let total = plan.tensors.iter().try_fold(0u64, |total, tensor| {
            total
                .checked_add(tensor.source.byte_len().map_err(QuantEngineError::Plan)?)
                .ok_or(QuantEngineError::Limit("source bytes"))
        })?;
        bound(total, plan.limits.max_source_bytes, "source bytes")?;
        let mut assets = Vec::new();
        let mut asset_lengths = Vec::new();
        let mut asset_bytes = 0u64;
        for name in &plan.assets {
            let remaining = plan.limits.max_asset_bytes - asset_bytes;
            let bytes = source.asset(name, platform(remaining, "asset bytes")?)?;
            bound(bytes.len() as u64, remaining, "asset bytes")?;
            asset_bytes += bytes.len() as u64;
            asset_lengths.push(bytes.len() as u64);
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
        let identity = tritium_evidence::original_source_id(&plan.architecture, &tensors, &assets)
            .map_err(QuantEngineError::Plan)?;
        Ok(Self {
            source,
            architecture: plan.architecture.clone(),
            identity,
            tensors,
            assets,
            asset_lengths,
        })
    }

    /// Actual complete stored-byte identity, unchanged by f32 widening.
    #[must_use]
    pub const fn source_id(&self) -> OriginalSourceId {
        self.identity
    }

    /// Architecture used by the shared raw-content identity derivation.
    #[must_use]
    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    /// Complete immutable tensor interpretations and actual raw-content bindings.
    #[must_use]
    pub fn tensor_bindings(&self) -> &[QuantSourceBinding] {
        &self.tensors
    }

    /// Complete immutable asset bindings; no asset payloads are retained.
    #[must_use]
    pub fn asset_bindings(&self) -> &[NamedAssetDigest] {
        &self.assets
    }

    /// Reread the complete current source and require every original binding.
    /// Uses the same checked reads; it does not lock files or certify future reads.
    /// # Errors
    /// Changed inventory, interpretation, actual payload/asset or source I/O.
    pub fn revalidate(&mut self) -> Result<(), QuantEngineError> {
        self.inventory()?;
        for index in 0..self.tensors.len() {
            let tensor = &self.tensors[index].source;
            let name = tensor.name.clone();
            let bytes = platform(
                tensor.byte_len().map_err(QuantEngineError::Plan)?,
                "source bytes",
            )?;
            self.tensor(&name, bytes)?;
        }
        for index in 0..self.assets.len() {
            let name = self.assets[index].name.clone();
            self.asset(&name, platform(self.asset_lengths[index], "asset bytes")?)?;
        }
        self.inventory()?;
        Ok(())
    }
}

impl WeightSource for ObservedWeightSource<'_> {
    fn inventory(&self) -> Result<Vec<QuantSourceTensor>, QuantEngineError> {
        let inventory = self.source.inventory()?;
        if inventory.iter().ne(self.tensors.iter().map(|b| &b.source))
            || self
                .source
                .asset_names()
                .iter()
                .map(String::as_str)
                .ne(self.assets.iter().map(|a| a.name.as_str()))
        {
            return Err(QuantEngineError::Binding(
                "complete observed inventory changed",
            ));
        }
        Ok(inventory)
    }

    fn asset_names(&self) -> Vec<String> {
        self.assets.iter().map(|a| a.name.clone()).collect()
    }

    fn tensor(&mut self, name: &str, max_bytes: usize) -> Result<Cow<'_, [u8]>, QuantEngineError> {
        let index = self
            .tensors
            .binary_search_by(|b| b.source.name.as_str().cmp(name))
            .map_err(|_| QuantEngineError::Binding("tensor outside observed source"))?;
        let expected = &self.tensors[index];
        let maximum = expected.source.byte_len().map_err(QuantEngineError::Plan)?;
        bound(maximum, max_bytes as u64, "observed tensor bytes")?;
        let bytes = self
            .source
            .tensor(name, platform(maximum, "source bytes")?)?;
        if source_binding(&expected.source, &bytes)? != *expected {
            return Err(QuantEngineError::Binding("observed tensor content changed"));
        }
        Ok(bytes)
    }

    fn asset(&mut self, name: &str, max_bytes: usize) -> Result<Cow<'_, [u8]>, QuantEngineError> {
        let index = self
            .assets
            .binary_search_by(|a| a.name.as_str().cmp(name))
            .map_err(|_| QuantEngineError::Binding("asset outside observed source"))?;
        let maximum = self.asset_lengths[index];
        bound(maximum, max_bytes as u64, "observed asset bytes")?;
        let bytes = self.source.asset(name, platform(maximum, "asset bytes")?)?;
        if bytes.len() as u64 != maximum
            || BlobId::from_bytes(*blake3::hash(&bytes).as_bytes()) != self.assets[index].content
        {
            return Err(QuantEngineError::Binding("observed asset content changed"));
        }
        Ok(bytes)
    }

    fn input_gram(
        &mut self,
        name: &str,
        max_entries: usize,
    ) -> Result<Cow<'_, [f64]>, QuantEngineError> {
        self.tensors
            .binary_search_by(|b| b.source.name.as_str().cmp(name))
            .map_err(|_| QuantEngineError::Binding("tensor outside observed source"))?;
        let values = self.source.input_gram(name, max_entries)?;
        bound(values.len() as u64, max_entries as u64, "input Gram")?;
        Ok(values)
    }
}
