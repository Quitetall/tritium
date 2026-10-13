//! Private retained-source adapter behind the one public text loader.
use super::{
    HfJsonTokenizer, LoadedModel, Qwen35TritReadLimits, Runner,
    qwen35_trit::{check, collect, decoded_bytes},
    source::{dense_values, matrix_shape, prepare_trit_metadata, upload_matrix},
};
use crate::{
    NnError, Projection, Qwen35CheckpointConfig, Qwen35TextRunner, Qwen35TextWeights, TensorLoader,
    TokenEmbedding, UnverifiedQwen35Mtp,
};
use std::{
    cell::RefCell,
    fs::File,
    io::{Cursor, Read, Seek},
    path::Path,
};
use tritium_format::{TritPackageReadLimits, trit_model::TritModelReader};
use tritium_runtime::{TensorPayloadBudget, TensorUploadContext};
use tritium_schema::{LoadedSourceIdentity, ModelLoaded, TensorStorage, TensorUploadPolicy};
use tritium_spec::TernaryBackend;

fn invalid(e: impl std::fmt::Display) -> NnError {
    NnError::InvalidArtifact(format!("retained model load: {e}"))
}
#[allow(clippy::too_many_arguments)]
pub(super) fn load_file(
    path: &Path,
    level: &str,
    receiving: TritPackageReadLimits,
    limits: Qwen35TritReadLimits,
    backend: Box<dyn TernaryBackend>,
    policy: TensorUploadPolicy,
    budget: &TensorPayloadBudget,
    execution: Option<super::StreamLimits>,
    context: TensorUploadContext<'_>,
) -> Result<LoadedModel, NnError> {
    positive(limits)?;
    let model =
        TritModelReader::open(File::open(path).map_err(invalid)?, receiving).map_err(invalid)?;
    load_reader(
        model, level, limits, backend, policy, budget, execution, context,
    )
}
pub(super) fn load_bytes(
    bytes: &[u8],
    level: &str,
    backend: Box<dyn TernaryBackend>,
    policy: TensorUploadPolicy,
    budget: &TensorPayloadBudget,
    execution: Option<super::StreamLimits>,
    context: TensorUploadContext<'_>,
) -> Result<LoadedModel, NnError> {
    // Preserve the borrowed source's historical admission envelope. The caller
    // already owns these bytes; file callers must instead supply explicit caps.
    let model = TritModelReader::open(
        Cursor::new(bytes),
        TritPackageReadLimits {
            max_source_bytes: bytes.len() as u64,
            max_manifest_bytes: bytes.len(),
            max_directory_bytes: bytes.len(),
        },
    )
    .map_err(invalid)?;
    load_reader(
        model,
        level,
        Qwen35TritReadLimits {
            max_config_bytes: 16 * 1024 * 1024,
            max_tensor_bytes: u64::MAX,
            max_decoded_tensor_bytes: u64::MAX,
        },
        backend,
        policy,
        budget,
        execution,
        context,
    )
}
fn positive(l: Qwen35TritReadLimits) -> Result<(), NnError> {
    if [
        l.max_config_bytes,
        l.max_tensor_bytes,
        l.max_decoded_tensor_bytes,
    ]
    .contains(&0)
    {
        return Err(invalid("read/decode caps must be positive"));
    }
    Ok(())
}
fn asset<R: Read + Seek>(
    model: &mut TritModelReader<R>,
    name: &str,
    max: u64,
) -> Result<Vec<u8>, NnError> {
    let n = model.asset_byte_len(name).map_err(invalid)?;
    check(n, max, "asset bytes")?;
    collect(n, |visit| model.visit_asset(name, visit).map_err(invalid))
}
fn tensor<R: Read + Seek>(
    model: &mut TritModelReader<R>,
    level: &str,
    name: &str,
    limits: Qwen35TritReadLimits,
) -> Result<Vec<u8>, NnError> {
    let n = model.tensor_byte_len(level, name).map_err(invalid)?;
    check(n, limits.max_tensor_bytes, "tensor bytes")?;
    collect(n, |visit| {
        model.visit_tensor(level, name, visit).map_err(invalid)
    })
}
#[allow(clippy::too_many_arguments)]
fn load_reader<R: Read + Seek>(
    mut model: TritModelReader<R>,
    level: &str,
    limits: Qwen35TritReadLimits,
    backend: Box<dyn TernaryBackend>,
    policy: TensorUploadPolicy,
    budget: &TensorPayloadBudget,
    execution: Option<super::StreamLimits>,
    context: TensorUploadContext<'_>,
) -> Result<LoadedModel, NnError> {
    positive(limits)?;
    model
        .validate_complete_payload_inventory()
        .map_err(invalid)?;
    let config_bytes = asset(&mut model, "config.json", limits.max_config_bytes)?;
    let text = std::str::from_utf8(&config_bytes).map_err(invalid)?;
    let json: serde_json::Value = serde_json::from_str(text).map_err(invalid)?;
    let hybrid = json["model_type"] == "qwen3_5";
    let standard = if hybrid {
        None
    } else {
        Some(prepare_trit_metadata(
            model.manifest(),
            level,
            &config_bytes,
        )?)
    };
    let qwen = if hybrid {
        Some(Qwen35CheckpointConfig::from_hf_config(text)?)
    } else {
        None
    };
    let admission = execution
        .map(|limits| {
            if let Some((config, _, _)) = &standard {
                super::CacheAdmission::standard(config, limits)
            } else {
                super::CacheAdmission::hybrid(
                    qwen.as_ref()
                        .ok_or_else(|| invalid("missing hybrid config"))?,
                    limits,
                )
            }
        })
        .transpose()?;
    let vocab = if let Some(qwen) = &qwen {
        qwen.text.vocab_size as usize
    } else {
        let meta = model
            .tensor_meta(level, "model.embed_tokens.weight")
            .map_err(invalid)?;
        matrix_shape(meta)?.0
    };
    // Exact hybrid inventory is admitted by the shared assembly before upload;
    // standard inventory is already admitted by the common model plan above.
    let selected = model
        .manifest()
        .levels
        .iter()
        .find(|l| l.name == level)
        .ok_or_else(|| invalid("absent selected level"))?;
    for meta in &selected.tensors {
        check(
            model.tensor_byte_len(level, &meta.name).map_err(invalid)?,
            limits.max_tensor_bytes,
            "encoded tensor bytes",
        )?;
        check(
            decoded_bytes(meta)?,
            limits.max_decoded_tensor_bytes,
            "logical decoded tensor bytes",
        )?;
    }
    let tokenizer_bytes = asset(&mut model, "tokenizer.json", 256 * 1024 * 1024)?;
    let tokenizer_config = asset(&mut model, "tokenizer_config.json", 16 * 1024 * 1024)?;
    let tokenizer = HfJsonTokenizer::from_bytes(&tokenizer_bytes, &tokenizer_config)?;
    tokenizer.validate_model_vocab(vocab)?;
    let source = LoadedSourceIdentity::Trit {
        package_id: tritium_schema::PackageId::from_bytes(*model.package_id().as_bytes()),
        model_id: model.model_id(level).map_err(invalid)?,
        level: level.to_owned(),
    };
    let backend_name = backend.device_id().to_owned();
    let physical_device = backend.physical_device_id().to_owned();
    let runner = {
        let mut loader = TensorLoader::new(
            backend.as_ref(),
            policy,
            budget,
            TensorUploadContext {
                recorder: &mut *context.recorder,
                span: context.span,
                parent: context.parent,
                logical_time: context.logical_time,
            },
        );
        if standard.is_some() {
            // Source binding is rechecked before any private matrix handle exists.
            model.verify().map_err(invalid)?;
        }
        if let Some((config, spec, plan)) = standard {
            let payload = tensor(&mut model, level, &plan.embedding.name, limits)?;
            let meta = model
                .tensor_meta(level, &plan.embedding.name)
                .map_err(invalid)?;
            let embedding = TokenEmbedding::from_linear(upload_matrix(
                &mut loader,
                meta,
                &payload,
                vocab,
                config.n_embd as usize,
            )?);
            drop(payload);
            let model = RefCell::new(&mut model);
            let loader = RefCell::new(&mut loader);
            let vectors = |name: &str, request: super::hf::DenseTensorRequest| {
                let super::hf::DenseTensorRequest::Vector { len } = request else {
                    return Err(invalid("embedding must use shared handle"));
                };
                let mut model = model.borrow_mut();
                let payload = tensor(&mut model, level, name, limits)?;
                let meta = model.tensor_meta(level, name).map_err(invalid)?;
                if !matches!(&meta.storage, TensorStorage::Dense { shape, .. } if shape.as_slice() == [len as u64])
                {
                    return Err(invalid("preserved vector shape"));
                }
                dense_values(meta, &payload)
            };
            let projections = |name: &str, rows, cols| {
                let mut model = model.borrow_mut();
                let payload = tensor(&mut model, level, name, limits)?;
                let meta = model.tensor_meta(level, name).map_err(invalid)?;
                upload_matrix(&mut loader.borrow_mut(), meta, &payload, rows, cols)
                    .map(Projection::Tensor)
            };
            let weights = plan.assemble(&config, &spec, embedding, vectors, projections)?;
            Runner::standard(
                super::ModelRunner::try_from_weights(config, weights, backend)?,
                spec,
                admission,
            )
        } else {
            let (config, weights, mtp_weights) =
                Qwen35TextWeights::load_trit_admitted(&mut model, level, &mut loader, limits)?;
            let language = Qwen35TextRunner::new(&config.text, weights, backend)?;
            let mtp = UnverifiedQwen35Mtp::new(&language, mtp_weights)?;
            Runner::hybrid(config, language, mtp, admission)?
        }
    };
    model.verify().map_err(invalid)?;
    let observation = ModelLoaded {
        source,
        architecture: runner.architecture().name().to_owned(),
        backend: backend_name,
        physical_device,
        vocab_size: vocab as u64,
        layer_count: runner.architecture().layer_count() as u64,
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
    Ok(LoadedModel {
        architecture: runner.architecture().clone(),
        runner,
        tokenizer,
        receipt,
    })
}
