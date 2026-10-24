//! Complete standard-model admission over the distinct packed QTEA carrier.
use super::{ModelWeights, hf::NameSchema};
use crate::{ArchSpec, ModelConfig, NnError, Projection, TokenEmbedding};
use std::{
    cell::RefCell,
    collections::BTreeMap,
    io::{Read, Seek},
};
use tritium_format::{
    TritPackageReadLimits,
    qtea::model::{QteaModelReadLimits, QteaModelReader},
};

/// Explicit source, per-owner and model weight-payload limits for host QTEA.
/// Metadata, Arc/allocator overhead, transient vector decode, staging, caches and
/// total process/RSS are separate domains, not covered by `max_weight_bytes`.
#[derive(Clone, Copy, Debug)]
pub struct HostQteaModelLimits {
    /// Existing whole-source and encoded metadata admission.
    pub receiving: TritPackageReadLimits,
    /// Per-owner/config/asset admission.
    pub model: QteaModelReadLimits,
    /// Aggregate unique packed Vec capacities plus each preserved F32 vector capacity.
    pub max_weight_bytes: usize,
    /// Maximum staging bytes per individual projection/gather call, not concurrent aggregate.
    pub max_staging_bytes: usize,
}
pub(super) fn invalid(message: &str) -> NnError {
    NnError::InvalidArtifact(message.into())
}
pub(super) fn failure(error: impl core::fmt::Display) -> NnError {
    NnError::InvalidArtifact(error.to_string())
}
pub(super) fn add_weight(used: &mut usize, bytes: usize, limit: usize) -> Result<(), NnError> {
    *used = used
        .checked_add(bytes)
        .filter(|sum| *sum <= limit)
        .ok_or_else(|| invalid("QTEA weight payload ceiling exceeded"))?;
    Ok(())
}

#[cfg(feature = "tokenizer")]
pub(super) fn read_tokenizer<R: Read + Seek>(
    reader: &mut QteaModelReader<R>,
    vocab: usize,
) -> Result<super::HfJsonTokenizer, NnError> {
    for (name, cap) in [
        ("tokenizer.json", 256_u64 * 1024 * 1024),
        ("tokenizer_config.json", 1024 * 1024),
    ] {
        let bytes = reader
            .asset_bytes(name)
            .map_err(|error| NnError::Tokenizer(error.to_string()))?;
        if bytes > cap {
            return Err(NnError::Tokenizer(format!(
                "{name} exceeds QTEA text source size limit"
            )));
        }
    }
    let token_bytes = reader
        .asset("tokenizer.json")
        .map_err(|error| NnError::Tokenizer(error.to_string()))?;
    let config_bytes = reader
        .asset("tokenizer_config.json")
        .map_err(|error| NnError::Tokenizer(error.to_string()))?;
    let tokenizer = super::HfJsonTokenizer::from_bytes(&token_bytes, &config_bytes)?;
    tokenizer.validate_model_vocab(vocab)?;
    Ok(tokenizer)
}

#[cfg(feature = "tokenizer")]
pub(super) fn publish(
    mut runner: super::Runner,
    tokenizer: super::HfJsonTokenizer,
    package: tritium_format::PackageId,
    expected_source_model: [u8; 32],
    physical_device: String,
    context: tritium_runtime::TensorUploadContext<'_>,
) -> Result<super::LoadedModel, NnError> {
    runner.qtea_package = Some(package);
    let architecture = runner.architecture().clone();
    let observation = tritium_schema::ModelLoaded {
        source: tritium_schema::LoadedSourceIdentity::Qtea {
            package_id: tritium_schema::PackageId::from_bytes(*package.as_bytes()),
            declared_source_model: expected_source_model,
        },
        architecture: architecture.name().into(),
        backend: "qtea-scalar-host".into(),
        physical_device,
        vocab_size: architecture.vocab_size() as u64,
        layer_count: architecture.layer_count() as u64,
    };
    let receipt = context
        .recorder
        .emit_model_loaded(
            context.span,
            context.parent,
            context.logical_time,
            &observation,
        )
        .map_err(|e| NnError::Backend(format!("model load evidence: {e}")))?;
    Ok(super::LoadedModel {
        runner,
        tokenizer,
        architecture,
        receipt,
    })
}

impl ModelWeights {
    /// Cold-load a complete standard model, retaining every matrix as packed QTEA.
    /// Expected source agreement is content-bound assertion, not original-source
    /// authentication. Config tying supplies one shared embedding/head owner.
    /// No HF path or dense projection fallback is used.
    /// # Errors
    /// Unknown architecture/profile, incomplete or wrong-role/shape inventory,
    /// source/payload mismatch, nonfinite vectors, caps, mutation or I/O.
    pub fn load_qtea(
        source: impl Read + Seek,
        expected_source_model: [u8; 32],
        limits: HostQteaModelLimits,
    ) -> Result<(ModelConfig, ArchSpec, Self, tritium_format::PackageId), NnError> {
        let (config, spec, weights, package, ()) = Self::load_qtea_with_admission(
            source,
            expected_source_model,
            limits,
            |_, _, _| Ok(()),
        )?;
        Ok((config, spec, weights, package))
    }

    fn load_qtea_with_admission<R: Read + Seek, A>(
        source: R,
        expected_source_model: [u8; 32],
        limits: HostQteaModelLimits,
        admit: impl FnOnce(&mut QteaModelReader<R>, &ModelConfig, usize) -> Result<A, NnError>,
    ) -> Result<(ModelConfig, ArchSpec, Self, tritium_format::PackageId, A), NnError> {
        let mut model_limits = limits.model;
        model_limits.max_config_bytes = model_limits
            .max_config_bytes
            .min(super::hf::MAX_CONFIG_JSON_BYTES as usize);
        let mut reader = QteaModelReader::open(
            source,
            limits.receiving,
            model_limits,
            expected_source_model,
        )
        .map_err(failure)?;
        let config_bytes = reader.config().map_err(failure)?;
        let inventory = super::QteaModelInventory::from_manifest(
            &config_bytes,
            super::QteaModelProfile::Standard,
            reader.manifest(),
            model_limits.max_tensors,
        )?;
        let super::qtea_inventory::PreparedGraph::Standard(prepared) = inventory.graph else {
            return Err(invalid("standard QTEA profile mismatch"));
        };
        let super::qtea_inventory::StandardInventory {
            config,
            spec,
            plan,
            vocab,
        } = *prepared;
        super::qtea_owners::plan(&reader, limits.max_weight_bytes)?;
        let admission = admit(&mut reader, &config, vocab)?;
        let owners = super::qtea_owners::read(&mut reader, limits)?;
        let mut packed = owners.matrices;
        let vectors = owners
            .preserved
            .into_iter()
            .map(|(name, (_, values))| (name, values))
            .collect::<BTreeMap<_, _>>();
        let embedding = packed
            .remove(NameSchema::Hf.top("token_embd"))
            .ok_or_else(|| invalid("QTEA embedding absent"))?;
        let vectors = RefCell::new(vectors);
        let weights = plan.assemble(
            &config,
            &spec,
            TokenEmbedding::from_host_qtea(embedding),
            |name, _| {
                vectors
                    .borrow_mut()
                    .remove(name)
                    .ok_or_else(|| NnError::MissingTensor(name.into()))
            },
            |name, _, _| {
                packed
                    .remove(name)
                    .map(Projection::HostQtea)
                    .ok_or_else(|| NnError::MissingTensor(name.into()))
            },
        )?;
        if !packed.is_empty() || !vectors.borrow().is_empty() {
            return Err(invalid("unconsumed QTEA model owner"));
        }
        reader.verify().map_err(failure)?;
        Ok((config, spec, weights, reader.package_id(), admission))
    }
}

#[cfg(feature = "tokenizer")]
impl super::Runner {
    /// Load this standard carrier into the common explicitly host-only runner.
    /// Stream admission covers existing cache domains, not full physical memory.
    /// Tokenizer/chat assets are not selected or interpreted by this method.
    /// # Errors
    /// All weight admission errors plus invalid stream/caches/graph geometry.
    pub fn load_qtea(
        source: impl Read + Seek,
        expected_source_model: [u8; 32],
        limits: HostQteaModelLimits,
        stream: super::StreamLimits,
    ) -> Result<Self, NnError> {
        let (config, spec, weights, package, admission) = ModelWeights::load_qtea_with_admission(
            source,
            expected_source_model,
            limits,
            |_, config, _| super::CacheAdmission::standard(config, stream),
        )?;
        let mut runner = Self::standard(
            config,
            weights,
            Box::new(tritium_cpu::CpuBackend::new()),
            spec,
            Some(admission),
        )?;
        runner.qtea_package = Some(package);
        Ok(runner)
    }
}

#[cfg(feature = "tokenizer")]
impl super::LoadedModel {
    /// Load packed host QTEA and its own bound tokenizer into the common text interface.
    /// Both tokenizer assets come from the same retained package handle and their
    /// IDs are checked before weight materialization. No source template is inferred.
    ///
    /// Only ModelLoaded registration is required by this explicit host path;
    /// the common recording context does not imply any TensorUploaded event.
    /// The returned observation is not independent/source/quality qualification.
    /// # Errors
    /// Invalid recording context, assets/templates/vocabulary, model or stream
    /// admission, changed source/I/O, private assembly or final recording failure.
    pub fn load_qtea(
        source: impl Read + Seek,
        expected_source_model: [u8; 32],
        limits: HostQteaModelLimits,
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
            .map_err(|error| NnError::Backend(format!("model load evidence: {error}")))?;
        let (config, spec, weights, package, (admission, tokenizer)) =
            ModelWeights::load_qtea_with_admission(
                source,
                expected_source_model,
                limits,
                |reader, config, vocab| {
                    let admission = super::CacheAdmission::standard(config, stream)?;
                    let tokenizer = read_tokenizer(reader, vocab)?;
                    Ok((admission, tokenizer))
                },
            )?;
        let backend = tritium_cpu::CpuBackend::new();
        let physical_device = tritium_spec::TernaryBackend::physical_device_id(&backend).to_owned();
        let runner =
            super::Runner::standard(config, weights, Box::new(backend), spec, Some(admission))?;
        publish(
            runner,
            tokenizer,
            package,
            expected_source_model,
            physical_device,
            context,
        )
    }
}
