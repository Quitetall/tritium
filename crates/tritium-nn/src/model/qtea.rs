//! Complete standard-model admission over the distinct packed QTEA carrier.
use super::{
    ModelWeights,
    hf::{NameSchema, declared_vocab_size, resolve_optional_attention_names},
    standard::{StandardModelPlan, TensorGeometry},
};
use crate::{ArchSpec, HostQteaLinear, ModelConfig, NnError, Projection, TokenEmbedding};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    io::{Read, Seek},
    sync::Arc,
};
use tritium_format::{
    TritPackageReadLimits,
    qtea::model::{QteaModelReadLimits, QteaModelReader, QteaModelTensor},
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
fn invalid(message: &str) -> NnError {
    NnError::InvalidArtifact(message.into())
}
fn failure(error: impl core::fmt::Display) -> NnError {
    NnError::InvalidArtifact(error.to_string())
}
fn add_weight(used: &mut usize, bytes: usize, limit: usize) -> Result<(), NnError> {
    *used = used
        .checked_add(bytes)
        .filter(|sum| *sum <= limit)
        .ok_or_else(|| invalid("QTEA weight payload ceiling exceeded"))?;
    Ok(())
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
        let config_text = std::str::from_utf8(&config_bytes).map_err(failure)?;
        let (config, mut spec) = ModelConfig::from_hf_config(config_text)?;
        let json = serde_json::from_str(config_text).map_err(failure)?;
        let manifest = reader.manifest();
        // Bound layer metadata by the already capped complete owner inventory.
        let minimum = (config.n_layers as usize)
            .checked_mul(9)
            .and_then(|n| n.checked_add(2))
            .ok_or_else(|| invalid("QTEA architecture inventory overflow"))?;
        if minimum > manifest.tensors.len() {
            return Err(invalid("QTEA architecture exceeds owner inventory"));
        }
        (spec.qkv_bias, spec.qk_norm) = resolve_optional_attention_names(
            &config,
            &json,
            manifest.tensors.keys().map(String::as_str),
        )?;
        let Some(QteaModelTensor::Qtea { rows, columns, .. }) =
            manifest.tensors.get(NameSchema::Hf.top("token_embd"))
        else {
            return Err(invalid("QTEA model requires a packed embedding owner"));
        };
        let vocab = usize::try_from(*rows).map_err(failure)?;
        if *columns != u64::from(config.n_embd)
            || declared_vocab_size(&json)?.is_some_and(|v| v != vocab)
        {
            return Err(invalid("QTEA embedding/config geometry mismatch"));
        }
        let plan = StandardModelPlan::new(&config, &spec, NameSchema::Hf, vocab)?;
        plan.admit_inventory(manifest.tensors.len(), |name| {
            manifest.tensors.get(name).map(|tensor| match tensor {
                QteaModelTensor::Qtea { rows, columns, .. } => TensorGeometry::Matrix {
                    rows: *rows,
                    cols: *columns,
                },
                QteaModelTensor::Vector { length, .. } => {
                    TensorGeometry::Vector { length: *length }
                }
            })
        })?;
        let mut planned = 0;
        let mut unique = BTreeSet::new();
        for (name, tensor) in &manifest.tensors {
            let bytes = match tensor {
                QteaModelTensor::Qtea { blob, .. } if unique.insert(*blob.as_bytes()) => {
                    usize::try_from(reader.tensor_bytes(name).map_err(failure)?).map_err(failure)?
                }
                QteaModelTensor::Qtea { .. } => 0,
                QteaModelTensor::Vector { length, .. } => usize::try_from(*length)
                    .map_err(failure)?
                    .checked_mul(4)
                    .ok_or_else(|| invalid("QTEA vector size overflow"))?,
            };
            add_weight(&mut planned, bytes, limits.max_weight_bytes)?;
        }
        // Retain only a bounded metadata inventory while mutable reads proceed.
        let names: Vec<_> = manifest.tensors.keys().cloned().collect();
        let admission = admit(&mut reader, &config, vocab)?;
        let mut packed = BTreeMap::new();
        let mut vectors = BTreeMap::new();
        let mut cache = BTreeMap::<[u8; 32], Arc<HostQteaLinear>>::new();
        let mut retained = 0;
        for name in names {
            match reader.manifest().tensors[&name].clone() {
                QteaModelTensor::Qtea {
                    blob,
                    rows,
                    columns,
                } => {
                    let key = *blob.as_bytes();
                    let matrix = if let Some(matrix) = cache.get(&key) {
                        if matrix.rows() as u64 != rows || matrix.columns() as u64 != columns {
                            return Err(invalid(
                                "shared QTEA blob has incompatible owner geometry",
                            ));
                        }
                        Arc::clone(matrix)
                    } else {
                        let tensor = reader.qtea(&name).map_err(failure)?;
                        add_weight(
                            &mut retained,
                            tensor.buffer_capacity_bytes(),
                            limits.max_weight_bytes,
                        )?;
                        let matrix = Arc::new(HostQteaLinear::from_tensor(
                            tensor,
                            limits.max_staging_bytes,
                        ));
                        cache.insert(key, Arc::clone(&matrix));
                        matrix
                    };
                    packed.insert(name, matrix);
                }
                QteaModelTensor::Vector { .. } => {
                    let values = reader.vector(&name).map_err(failure)?;
                    add_weight(
                        &mut retained,
                        values
                            .capacity()
                            .checked_mul(4)
                            .ok_or_else(|| invalid("QTEA vector capacity overflow"))?,
                        limits.max_weight_bytes,
                    )?;
                    vectors.insert(name, values);
                }
            }
        }
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
                    let tokenizer =
                        super::HfJsonTokenizer::from_bytes(&token_bytes, &config_bytes)?;
                    tokenizer.validate_model_vocab(vocab)?;
                    Ok((admission, tokenizer))
                },
            )?;
        let backend = tritium_cpu::CpuBackend::new();
        let observation = tritium_schema::ModelLoaded {
            source: tritium_schema::LoadedSourceIdentity::Qtea {
                package_id: tritium_schema::PackageId::from_bytes(*package.as_bytes()),
                declared_source_model: expected_source_model,
            },
            architecture: config.arch.clone(),
            backend: "qtea-scalar-host".into(),
            physical_device: tritium_spec::TernaryBackend::physical_device_id(&backend).to_owned(),
            vocab_size: u64::try_from(weights.vocab).map_err(failure)?,
            layer_count: u64::from(config.n_layers),
        };
        let mut runner =
            super::Runner::standard(config, weights, Box::new(backend), spec, Some(admission))?;
        runner.qtea_package = Some(package);
        let architecture = runner.architecture().clone();
        let receipt = context
            .recorder
            .emit_model_loaded(
                context.span,
                context.parent,
                context.logical_time,
                &observation,
            )
            .map_err(|error| NnError::Backend(format!("model load evidence: {error}")))?;
        Ok(Self {
            runner,
            tokenizer,
            architecture,
            receipt,
        })
    }
}
