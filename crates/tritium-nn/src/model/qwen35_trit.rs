//! Qwen graph assembly from the ordinary retained typed reader/shared tensor path.
use super::{
    qwen35_hf::{
        Qwen35HfTensorSource, Qwen35TensorSchemaRole, load_language_weights, load_mtp_weights,
        qwen35_language_mtp_tensor_schema,
    },
    source::{dense_values, matrix_shape, upload_matrix},
};
use crate::{
    NnError, Projection, Qwen35CheckpointConfig, Qwen35MtpWeights, Qwen35TextWeights, TensorLoader,
    TokenEmbedding,
};
use std::{
    cell::RefCell,
    io::{Read, Seek},
};
use tritium_core::{PerTilePlan, UniformPlan};
use tritium_format::trit_model::TritModelReader;
use tritium_schema::{PlaneCounts, TensorMeta, TensorStorage};

/// Explicit source and logical decode ceilings for shared Qwen weights assembly.
/// Not capacity, aggregate heap/RSS, preserved-vector/KV or backend-memory admission.
#[derive(Clone, Copy, Debug)]
pub struct Qwen35TritReadLimits {
    /// Config asset encoded byte ceiling before allocation.
    pub max_config_bytes: u64,
    /// Per-tensor encoded byte ceiling before allocation.
    pub max_tensor_bytes: u64,
    /// Per-tensor logical decoded buffers: trits, f32 scales and count/index buffers,
    /// or f32 dense coefficients. Excludes source bytes, capacity and handles.
    pub max_decoded_tensor_bytes: u64,
}

impl Qwen35TextWeights {
    /// Assemble complete Qwen language/MTP weights through the shared tensor backend.
    ///
    /// Derives the graph from actual config and exact complete tensor inventory.
    /// Metadata/caps/recorder preflight precedes any matrix upload. Source bytes
    /// are bounded per tensor and decoded only for the current upload; no complete
    /// package or retained dense matrix shadow is introduced. Preserved vectors
    /// and convolution arrays retain exact source shape and widening semantics.
    /// Reuses the ordinary typed upload helpers, graph builders and basis binding.
    ///
    /// Terminal source failure drops private partial handles/leases; prior upload
    /// observations remain observations, not model-load/quality/completion evidence.
    /// Returned MTP weights are unbound: construct UnverifiedQwen35Mtp with the
    /// target runner, then use its existing source-bound parity gate. This is not
    /// the unified public Runner/loader, official source authentication, aggregate
    /// memory, GPU/performance or release qualification. Source custody is external.
    /// # Errors
    /// Config/inventory/shape/storage/cap/evidence, source/decode or backend errors.
    pub fn load_trit_admitted<R: Read + Seek>(
        model: &mut TritModelReader<R>,
        level: &str,
        loader: &mut TensorLoader<'_>,
        limits: Qwen35TritReadLimits,
    ) -> Result<(Qwen35CheckpointConfig, Self, Qwen35MtpWeights), NnError> {
        if [
            limits.max_config_bytes,
            limits.max_tensor_bytes,
            limits.max_decoded_tensor_bytes,
        ]
        .contains(&0)
        {
            return Err(invalid("source/decode limits must be positive"));
        }
        loader.check_evidence()?;
        model
            .validate_complete_payload_inventory()
            .map_err(invalid)?;
        model.verify().map_err(invalid)?;
        let length = model.asset_byte_len("config.json").map_err(invalid)?;
        check(length, limits.max_config_bytes, "config bytes")?;
        let config_bytes = collect(length, |visit| {
            model.visit_asset("config.json", visit).map_err(invalid)
        })?;
        let config = Qwen35CheckpointConfig::from_hf_config(
            std::str::from_utf8(&config_bytes).map_err(invalid)?,
        )?;
        if model.manifest().architecture != config.model_type {
            return Err(invalid("manifest/config architecture mismatch"));
        }
        let schema = qwen35_language_mtp_tensor_schema(&config.text)?;
        let selected = model
            .manifest()
            .levels
            .iter()
            .find(|l| l.name == level)
            .ok_or_else(|| invalid("absent Qwen level"))?;
        if selected.tensors.len() != schema.len() {
            return Err(invalid("Qwen language/MTP inventory count mismatch"));
        }
        for entry in &schema {
            let meta = model.tensor_meta(level, &entry.name).map_err(invalid)?;
            let expected = entry.shape.iter().map(|&d| d as u64).collect::<Vec<_>>();
            match (&meta.storage, entry.role) {
                (TensorStorage::Dense { shape, .. }, _) if shape == &expected => {}
                (TensorStorage::Additive { .. }, Qwen35TensorSchemaRole::Matrix)
                    if expected.len() == 2
                        && matrix_shape(meta)? == (entry.shape[0], entry.shape[1]) => {}
                _ => {
                    return Err(invalid(format!(
                        "{}: Qwen shape/storage mismatch",
                        entry.name
                    )));
                }
            }
            check(
                model.tensor_byte_len(level, &entry.name).map_err(invalid)?,
                limits.max_tensor_bytes,
                "encoded tensor bytes",
            )?;
            check(
                decoded_bytes(meta)?,
                limits.max_decoded_tensor_bytes,
                "logical decoded tensor bytes",
            )?;
        }
        let source = Source {
            model: RefCell::new(model),
            level,
            loader: RefCell::new(loader),
            limits,
        };
        let language = load_language_weights(&source, &config.text)?;
        let mtp = load_mtp_weights(&source, &config.text)?;
        source.model.borrow_mut().verify().map_err(invalid)?;
        Ok((config, language, mtp))
    }
}

fn invalid(error: impl std::fmt::Display) -> NnError {
    NnError::InvalidArtifact(format!("typed Qwen assembly: {error}"))
}
pub(super) fn check(actual: u64, maximum: u64, what: &str) -> Result<(), NnError> {
    if actual > maximum {
        return Err(invalid(format!("{what} {actual} exceeds {maximum}")));
    }
    Ok(())
}
pub(super) fn decoded_bytes(meta: &TensorMeta) -> Result<u64, NnError> {
    match &meta.storage {
        TensorStorage::Dense { shape, .. } => shape.iter().try_fold(4u64, |n, d| {
            n.checked_mul(*d)
                .ok_or_else(|| invalid("dense decoded size overflow"))
        }),
        TensorStorage::Additive { layout, planes } => {
            let (trits, scales, auxiliary) = match planes {
                PlaneCounts::Uniform(n) => {
                    let p = UniformPlan::new(*layout, *n).map_err(|e| invalid(format!("{e:?}")))?;
                    (p.trit_count(), p.scale_count(), 0)
                }
                PlaneCounts::PerTile(counts) => {
                    let p =
                        PerTilePlan::new(*layout, counts).map_err(|e| invalid(format!("{e:?}")))?;
                    (
                        p.trit_count(),
                        p.scale_count(),
                        p.buffer_len_bytes()
                            .map_err(|e| invalid(format!("{e:?}")))?,
                    )
                }
                _ => return Err(invalid("unsupported plane allocation")),
            };
            let bytes = scales
                .checked_mul(4)
                .and_then(|n| n.checked_add(trits))
                .and_then(|n| n.checked_add(auxiliary))
                .ok_or_else(|| invalid("additive decoded size overflow"))?;
            u64::try_from(bytes).map_err(invalid)
        }
        _ => Err(invalid("unsupported tensor storage")),
    }
}
type Visitor<'a> = &'a mut dyn FnMut(&[u8]) -> Result<(), NnError>;
pub(super) fn collect(
    length: u64,
    mut read: impl FnMut(Visitor<'_>) -> Result<(), NnError>,
) -> Result<Vec<u8>, NnError> {
    let length = usize::try_from(length).map_err(invalid)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(length).map_err(invalid)?;
    read(&mut |chunk| {
        if bytes
            .len()
            .checked_add(chunk.len())
            .is_none_or(|n| n > length)
        {
            return Err(invalid("source exceeds admitted encoded length"));
        }
        bytes.extend_from_slice(chunk);
        Ok(())
    })?;
    if bytes.len() != length {
        return Err(invalid("source shorter than admitted encoded length"));
    }
    Ok(bytes)
}
struct Source<'a, 'ctx, R: Read + Seek> {
    model: RefCell<&'a mut TritModelReader<R>>,
    level: &'a str,
    loader: RefCell<&'a mut TensorLoader<'ctx>>,
    limits: Qwen35TritReadLimits,
}
impl<R: Read + Seek> Source<'_, '_, R> {
    fn payload(&self, model: &mut TritModelReader<R>, name: &str) -> Result<Vec<u8>, NnError> {
        let len = model.tensor_byte_len(self.level, name).map_err(invalid)?;
        check(len, self.limits.max_tensor_bytes, "encoded tensor bytes")?;
        collect(len, |visit| {
            model.visit_tensor(self.level, name, visit).map_err(invalid)
        })
    }
}
impl<R: Read + Seek> Qwen35HfTensorSource for Source<'_, '_, R> {
    fn tensor_f32_exact(&self, name: &str, expected: &[usize]) -> Result<Vec<f32>, NnError> {
        let mut model = self.model.borrow_mut();
        let payload = self.payload(&mut model, name)?;
        let meta = model.tensor_meta(self.level, name).map_err(invalid)?;
        let TensorStorage::Dense { shape, .. } = &meta.storage else {
            return Err(invalid("preserved Qwen tensor must be dense"));
        };
        if !shape.iter().copied().eq(expected.iter().map(|&d| d as u64)) {
            return Err(invalid("preserved shape mismatch"));
        }
        dense_values(meta, &payload)
    }
    fn projection_exact(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Projection, NnError> {
        let mut model = self.model.borrow_mut();
        let payload = self.payload(&mut model, name)?;
        let meta = model.tensor_meta(self.level, name).map_err(invalid)?;
        Ok(Projection::Tensor(upload_matrix(
            &mut self.loader.borrow_mut(),
            meta,
            &payload,
            rows,
            cols,
        )?))
    }
    fn token_embedding_exact(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<TokenEmbedding, NnError> {
        let mut model = self.model.borrow_mut();
        let payload = self.payload(&mut model, name)?;
        let meta = model.tensor_meta(self.level, name).map_err(invalid)?;
        Ok(TokenEmbedding::from_linear(upload_matrix(
            &mut self.loader.borrow_mut(),
            meta,
            &payload,
            rows,
            cols,
        )?))
    }
}
