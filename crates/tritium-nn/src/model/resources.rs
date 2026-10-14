//! Aggregate logical admission for the common retained-source loader.
use super::{ModelLoadPlan, retained_plan::Graph};
use crate::{NnError, Qwen35TensorSchemaRole, qwen35_language_mtp_tensor_schema};
use std::{
    collections::BTreeMap,
    io::{Read, Seek},
};
use tritium_format::trit_model::TritModelReader;
use tritium_runtime::{TensorPayloadBudget, TensorStorageReservation};
use tritium_spec::{
    TENSOR_STREAM_CHUNK_BYTES, TensorStreamPlan, TensorUploadPlan, TensorUploadPolicy,
    TernaryBackend, validate_tensor_upload_policy,
};

/// Explicit logical storage domains for resource-planned retained loading.
/// Reuse the same ledger to charge overlapping domains to one aggregate cap.
/// Source metadata, tokenizer heap, allocator/driver costs and execution scratch
/// are excluded/unknown. This is not a physical fit or release qualification.
#[derive(Debug, Clone, Copy)]
pub struct ModelLoadResources<'a> {
    /// Whole-model matrix payload, including MTP and counting tied handles once.
    pub payload: &'a TensorPayloadBudget,
    /// Maximum source working storage during assembly: current encoded+decoded
    /// pair in compatibility mode, bounded matrix windows/preserved pairs in
    /// mandatory streaming mode. Excludes metadata and allocator overhead.
    pub source_workspace: &'a TensorPayloadBudget,
    /// Maximum additional explicit backend host upload workspace.
    pub host_workspace: &'a TensorPayloadBudget,
    /// Maximum additional explicit backend device upload workspace.
    pub device_workspace: &'a TensorPayloadBudget,
    /// Widened preserved tensors plus maximum logical host language cache.
    pub retained_host: &'a TensorPayloadBudget,
}

/// Immutable covered logical storage ceilings admitted before materializing weights.
/// Not physical RAM/VRAM, native execution, model quality or a readiness receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelResourceAdmission {
    /// Aggregate backend-declared owned matrix payload.
    pub payload_bytes: u64,
    /// Conservative maximum encoded+decoded current tensor pair.
    pub source_workspace_bytes: u64,
    /// Conservative maximum additional explicit backend host upload workspace.
    pub host_workspace_bytes: u64,
    /// Conservative maximum additional explicit backend device upload workspace.
    pub device_workspace_bytes: u64,
    /// Preserved f32 coefficients plus admitted maximum host language cache.
    pub retained_host_bytes: u64,
}

pub(super) struct LoadReservations {
    pub(super) payload: TensorPayloadBudget,
    pub(super) host_workspace: TensorPayloadBudget,
    pub(super) device_workspace: TensorPayloadBudget,
    pub(super) plans: BTreeMap<String, TensorUploadPlan>,
    pub(super) stream_plans: Option<BTreeMap<String, TensorStreamPlan>>,
    pub(super) admission: ModelResourceAdmission,
    pub(super) retained_host: TensorStorageReservation,
    _source_workspace: TensorStorageReservation,
}

fn invalid(e: impl std::fmt::Display) -> NnError {
    NnError::ResourceExhausted(format!("model resource admission: {e}"))
}
fn add(sum: &mut u64, bytes: u64) -> Result<(), NnError> {
    *sum = sum
        .checked_add(bytes)
        .ok_or_else(|| invalid("byte sum overflow"))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn reserve<R: Read + Seek>(
    model: &TritModelReader<R>,
    level: &str,
    graph: &Graph,
    source: &ModelLoadPlan,
    backend: &dyn TernaryBackend,
    policy: TensorUploadPolicy,
    resources: ModelLoadResources<'_>,
    streamed: bool,
) -> Result<LoadReservations, NnError> {
    if backend.resident_decoder_requested() {
        return Err(invalid("resident graph resource planning is unavailable"));
    }
    let mut admission = ModelResourceAdmission {
        payload_bytes: 0,
        source_workspace_bytes: if streamed {
            TENSOR_STREAM_CHUNK_BYTES as u64
        } else {
            source.maximum_tensor_working_payload_bytes()
        },
        host_workspace_bytes: 0,
        device_workspace_bytes: 0,
        retained_host_bytes: source
            .cache_admission()
            .ok_or_else(|| invalid("explicit stream limits are required"))?
            .logical_host_cache_payload_bytes(),
    };
    let mut plans = BTreeMap::new();
    let mut stream_plans = streamed.then(BTreeMap::new);
    let mut visit = |name: &str, matrix: bool| -> Result<(), NnError> {
        let meta = model.tensor_meta(level, name).map_err(invalid)?;
        if !matrix {
            add(
                &mut admission.retained_host_bytes,
                super::qwen35_trit::decoded_bytes(meta)?,
            )?;
            if streamed {
                let pair = source
                    .tensors()
                    .iter()
                    .find(|t| t.name() == name)
                    .ok_or_else(|| invalid("absent preserved source plan"))?;
                admission.source_workspace_bytes = admission.source_workspace_bytes.max(
                    pair.encoded_bytes()
                        .checked_add(pair.logical_decoded_bytes())
                        .ok_or_else(|| invalid("preserved working pair overflow"))?,
                );
            }
            return Ok(());
        }
        let metadata = super::tensor_stream::metadata(meta).map_err(invalid)?;
        let plan = if let Some(plans) = &mut stream_plans {
            let plan = backend
                .tensor_stream_plan(
                    metadata,
                    super::tensor_stream::encoding(meta).map_err(invalid)?,
                )
                .map_err(invalid)?
                .ok_or_else(|| invalid(format!("{name}: backend streaming plan is unknown")))?;
            if plan.encoded_bytes != model.tensor_byte_len(level, name).map_err(invalid)? {
                return Err(invalid(format!("{name}: stream encoded geometry mismatch")));
            }
            plans.insert(name.to_owned(), plan);
            plan.upload
        } else {
            backend
                .tensor_upload_plan(metadata)
                .map_err(invalid)?
                .ok_or_else(|| invalid(format!("{name}: backend upload plan is unknown")))?
        };
        validate_tensor_upload_policy(plan.caps, policy).map_err(invalid)?;
        add(&mut admission.payload_bytes, plan.caps.payload_bytes)?;
        admission.host_workspace_bytes = admission
            .host_workspace_bytes
            .max(plan.host_workspace_bytes);
        admission.device_workspace_bytes = admission
            .device_workspace_bytes
            .max(plan.device_workspace_bytes);
        if plans.insert(name.to_owned(), plan).is_some() {
            return Err(invalid("duplicate graph upload request"));
        }
        Ok(())
    };
    match graph {
        Graph::Standard(p) => {
            // The authoritative graph omits a separate head request when tied.
            for request in p.2.tensors() {
                visit(&request.name, request.shape.len() == 2)?;
            }
        }
        Graph::Hybrid(config) => {
            // Rank alone is insufficient: convolution tensors remain host-owned.
            for entry in qwen35_language_mtp_tensor_schema(&config.text)? {
                visit(&entry.name, entry.role == Qwen35TensorSchemaRole::Matrix)?;
            }
        }
    }
    // All domains are reserved before any complete tensor is read/decompressed.
    // A later refusal drops earlier guards/partitions, including on shared ledgers.
    let payload = resources
        .payload
        .partition(admission.payload_bytes)
        .map_err(invalid)?;
    let retained_host = resources
        .retained_host
        .reserve_storage(admission.retained_host_bytes)
        .map_err(invalid)?;
    let source_workspace = resources
        .source_workspace
        .reserve_storage(admission.source_workspace_bytes)
        .map_err(invalid)?;
    let host_workspace = resources
        .host_workspace
        .partition(admission.host_workspace_bytes)
        .map_err(invalid)?;
    let device_workspace = resources
        .device_workspace
        .partition(admission.device_workspace_bytes)
        .map_err(invalid)?;
    Ok(LoadReservations {
        payload,
        host_workspace,
        device_workspace,
        plans,
        stream_plans,
        admission,
        retained_host,
        _source_workspace: source_workspace,
    })
}
