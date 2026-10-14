//! Private retained-source adapter behind the one public text loader.
use super::{
    LoadedModel, Qwen35TritReadLimits, Runner,
    qwen35_trit::{assemble_prepared, check, collect},
    retained_plan::{Graph, Prepared, positive, prepare},
    source::{dense_values, upload_matrix},
};
use crate::{
    NnError, Projection, Qwen35TextRunner, TensorLoader, TokenEmbedding, UnverifiedQwen35Mtp,
};
use std::{
    cell::RefCell,
    fs::File,
    io::{Cursor, Read, Seek},
    path::Path,
};
use tritium_format::{TritPackageReadLimits, trit_model::TritModelReader};
use tritium_runtime::{TensorPayloadBudget, TensorUploadContext};
use tritium_schema::{ModelLoaded, TensorStorage, TensorUploadPolicy};
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
    let file = File::open(path).map_err(invalid)?;
    let source_bytes = file.metadata().map_err(invalid)?.len();
    let model = TritModelReader::open(file, receiving).map_err(invalid)?;
    load_reader(
        model,
        source_bytes,
        level,
        limits,
        backend,
        policy,
        budget,
        execution,
        context,
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
        bytes.len() as u64,
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
    source_bytes: u64,
    level: &str,
    limits: Qwen35TritReadLimits,
    backend: Box<dyn TernaryBackend>,
    policy: TensorUploadPolicy,
    budget: &TensorPayloadBudget,
    execution: Option<super::StreamLimits>,
    context: TensorUploadContext<'_>,
) -> Result<LoadedModel, NnError> {
    let Prepared {
        graph,
        tokenizer,
        plan,
    } = prepare(&mut model, level, source_bytes, limits, execution)?;
    let vocab = plan.architecture().vocab_size();
    let admission = plan.cache_admission();
    let source = plan.source().clone();
    drop(plan);
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
        match graph {
            Graph::Standard(prepared) => {
                let (config, spec, plan) = *prepared;
                model.verify().map_err(invalid)?;
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
                Runner::standard(config, weights, backend, spec, admission)?
            }
            Graph::Hybrid(prepared) => {
                let config = *prepared;
                let (weights, mtp_weights) =
                    assemble_prepared(&mut model, level, &mut loader, limits, &config)?;
                let language = Qwen35TextRunner::new(&config.text, weights, backend)?;
                let mtp = UnverifiedQwen35Mtp::new(&language, mtp_weights)?;
                Runner::hybrid(config, language, mtp, admission)?
            }
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
