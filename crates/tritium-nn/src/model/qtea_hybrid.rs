//! Exact Qwen hybrid schema/assembler over explicitly residual-bearing QTEA.
#[cfg(test)]
#[path = "qtea_hybrid_tests.rs"]
mod tests;
use super::{
    qtea::{HostQteaModelLimits, failure, invalid, publish, read_tokenizer},
    qtea_owners::Owners,
    qwen35_hf::{Qwen35HfTensorSource, load_language_weights, load_mtp_weights},
};
use crate::{NnError, Projection, TokenEmbedding};
use std::{
    cell::RefCell,
    io::{Read, Seek},
};
use tritium_format::qtea::model::{QteaHybridModelReadLimits, QteaModelReader};

/// Explicit hybrid limits. Preserved higher-rank tensors are not norm vectors.
#[derive(Clone, Copy, Debug)]
pub struct HostQteaHybridModelLimits {
    /// Existing source, vector, matrix, logical weight and per-call staging caps.
    pub base: HostQteaModelLimits,
    /// Positive maximum element count of each exact-shaped preserved tensor.
    pub max_preserved_tensor_elements: usize,
}
struct Source(RefCell<Owners>);
impl Qwen35HfTensorSource for Source {
    fn tensor_f32_exact(&self, name: &str, expected: &[usize]) -> Result<Vec<f32>, NnError> {
        let (shape, values) = self
            .0
            .borrow_mut()
            .preserved
            .remove(name)
            .ok_or_else(|| invalid("missing/consumed QTEA preserved owner"))?;
        if !shape.iter().copied().eq(expected.iter().map(|d| *d as u64)) {
            return Err(invalid("hybrid preserved original shape mismatch"));
        }
        Ok(values)
    }
    fn projection_exact(
        &self,
        name: &str,
        rows: usize,
        columns: usize,
    ) -> Result<Projection, NnError> {
        let matrix = self
            .0
            .borrow_mut()
            .matrices
            .remove(name)
            .ok_or_else(|| invalid("missing/consumed QTEA matrix owner"))?;
        if matrix.rows() != rows || matrix.columns() != columns {
            return Err(invalid("hybrid packed matrix shape mismatch"));
        }
        Ok(Projection::HostQtea(matrix))
    }
    fn token_embedding_exact(
        &self,
        name: &str,
        rows: usize,
        columns: usize,
    ) -> Result<TokenEmbedding, NnError> {
        let Projection::HostQtea(matrix) = self.projection_exact(name, rows, columns)? else {
            return Err(invalid("hybrid embedding must remain packed"));
        };
        Ok(TokenEmbedding::from_host_qtea(matrix))
    }
}
impl super::LoadedModel {
    /// Load a complete packed Qwen-family hybrid language/MTP candidate.
    /// Real tokenizer/schema/cache admission precedes numerical weights. Preserved
    /// convolution/state retain exact shapes; matrices never use a dense fallback.
    /// Retained MTP remains Unverified; no draft, source or release authority.
    /// # Errors
    /// Context, profile, source/config/schema/shape/role, caps, numeric, graph,
    /// tokenizer, terminal source verification or final recording refusal.
    pub fn load_qtea_hybrid(
        source: impl Read + Seek,
        expected_source_model: [u8; 32],
        limits: HostQteaHybridModelLimits,
        stream: super::StreamLimits,
        context: tritium_runtime::TensorUploadContext<'_>,
    ) -> Result<Self, NnError> {
        context
            .recorder
            .check_emit::<tritium_schema::ModelLoaded>(
                context.span,
                context.parent,
                context.logical_time,
            )
            .map_err(failure)?;
        if limits.base.max_staging_bytes == 0 {
            return Err(invalid("hybrid QTEA staging cap must be positive"));
        }
        let mut model = limits.base.model;
        model.max_config_bytes = model
            .max_config_bytes
            .min(super::hf::MAX_CONFIG_JSON_BYTES as usize);
        let mut reader = QteaModelReader::open_hybrid(
            source,
            limits.base.receiving,
            QteaHybridModelReadLimits {
                base: model,
                max_preserved_tensor_elements: limits.max_preserved_tensor_elements,
            },
            expected_source_model,
        )
        .map_err(failure)?;
        let bytes = reader.config().map_err(failure)?;
        let inventory = super::QteaModelInventory::from_manifest(
            &bytes,
            super::QteaModelProfile::Qwen35Hybrid,
            reader.manifest(),
            model.max_tensors,
        )?;
        let super::qtea_inventory::PreparedGraph::Hybrid(config) = inventory.graph else {
            return Err(invalid("hybrid QTEA profile mismatch"));
        };
        let config = *config;
        super::qtea_owners::plan(&reader, limits.base.max_weight_bytes)?;
        let admission = super::CacheAdmission::hybrid(&config, stream)?;
        let tokenizer = read_tokenizer(&mut reader, config.text.vocab_size as usize)?;
        let owners = Source(RefCell::new(super::qtea_owners::read(
            &mut reader,
            limits.base,
        )?));
        let language_weights = load_language_weights(&owners, &config.text)?;
        let mtp_weights = load_mtp_weights(&owners, &config.text)?;
        let owners = owners.0.into_inner();
        if !owners.matrices.is_empty() || !owners.preserved.is_empty() {
            return Err(invalid("unconsumed hybrid QTEA owner"));
        }
        let backend = tritium_cpu::CpuBackend::new();
        let physical_device = tritium_spec::TernaryBackend::physical_device_id(&backend).to_owned();
        let language =
            super::Qwen35TextRunner::new(&config.text, language_weights, Box::new(backend))?;
        let mtp = super::UnverifiedQwen35Mtp::new(&language, mtp_weights)?;
        let runner = super::Runner::hybrid(config, language, mtp, Some(admission))?;
        reader.verify().map_err(failure)?;
        publish(
            runner,
            tokenizer,
            reader.package_id(),
            expected_source_model,
            physical_device,
            context,
        )
    }
}
