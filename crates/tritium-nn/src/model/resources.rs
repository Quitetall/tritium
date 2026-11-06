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
    BackendError, DecoderAllocationMetadata, DecoderLimits, DecoderPayloadPlan,
    TENSOR_STREAM_CHUNK_BYTES, TensorMetadata, TensorStreamEncoding, TensorStreamPlan,
    TensorUploadPlan, TensorUploadPolicy, TernaryBackend, validate_tensor_upload_policy,
};

/// Explicit logical storage domains for resource-planned retained loading.
/// Reuse the same ledger to charge overlapping domains to one aggregate cap.
/// Source metadata, tokenizer heap, allocator/driver, host execution scratch
/// and resident host metadata are excluded/unknown. This is not a physical fit
/// or release qualification.
#[derive(Debug, Clone, Copy)]
pub struct ModelLoadResources<'a> {
    /// Matrix payload, including MTP and tied handles once; resident selection
    /// reserves its complete configured device envelope in this parent ledger.
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
    /// Reserved configured matrix/vector/cache/workspace ceiling, if resident.
    /// Not an exact graph estimate, actual payload or physical VRAM measurement.
    pub resident_device_payload_limit: Option<u64>,
    /// Frozen checked structural graph declaration, not actual physical memory.
    pub resident_device_payload_plan: Option<DecoderPayloadPlan>,
    /// Conservative maximum encoded+decoded current tensor pair.
    pub source_workspace_bytes: u64,
    /// Conservative maximum additional explicit backend host upload workspace.
    pub host_workspace_bytes: u64,
    /// Conservative maximum additional explicit backend device upload workspace.
    pub device_workspace_bytes: u64,
    /// Preserved f32 coefficients plus admitted maximum host language cache.
    pub retained_host_bytes: u64,
}
impl ModelResourceAdmission {
    /// Parent payload reservation: exact matrix plan or conservative resident
    /// envelope including those matrices. Never a physical-memory measurement.
    #[must_use]
    pub fn reserved_payload_bytes(&self) -> u64 {
        self.resident_device_payload_limit
            .unwrap_or(self.payload_bytes)
    }
}

fn resident_limit(backend: &dyn TernaryBackend) -> Result<Option<u64>, NnError> {
    match (
        backend.resident_decoder_requested(),
        backend.resident_decoder_payload_limit(),
    ) {
        (false, None) => Ok(None),
        (true, Some(limit)) if limit > 0 => u64::try_from(limit).map(Some).map_err(invalid),
        _ => Err(invalid(
            "resident payload envelope is unknown or inconsistent",
        )),
    }
}
pub(super) fn validate_selection(
    backend: &dyn TernaryBackend,
    reservations: &LoadReservations,
) -> Result<(), NnError> {
    if resident_limit(backend)? != reservations.admission.resident_device_payload_limit {
        return Err(invalid(
            "resident payload envelope changed before graph construction",
        ));
    }
    if let Some(allocation) = &reservations.resident {
        let current = backend
            .decoder_payload_plan(allocation.metadata(), allocation.limits)
            .map_err(invalid)?;
        if current != reservations.admission.resident_device_payload_plan {
            return Err(invalid(
                "resident structural plan changed before graph construction",
            ));
        }
    }
    Ok(())
}
pub(super) fn validate_runner(
    runner: &super::Runner,
    admission: ModelResourceAdmission,
) -> Result<(), NnError> {
    if let Some(limit) = admission.resident_device_payload_limit {
        let actual = runner
            .resident_payload_bytes()
            .and_then(|n| u64::try_from(n).ok());
        if runner.execution() != tritium_spec::DecoderExecution::ResidentDeviceF32
            || !actual.is_some_and(|n| n >= admission.payload_bytes && n <= limit)
            || admission
                .resident_device_payload_plan
                .is_none_or(|p| actual != u64::try_from(p.payload_bytes()).ok())
        {
            return Err(invalid(
                "resident executor payload or execution differs from envelope",
            ));
        }
    }
    Ok(())
}

pub(super) struct LoadReservations {
    pub(super) payload: TensorPayloadBudget,
    pub(super) host_workspace: TensorPayloadBudget,
    pub(super) device_workspace: TensorPayloadBudget,
    pub(super) plans: BTreeMap<String, TensorUploadPlan>,
    pub(super) stream_plans: Option<BTreeMap<String, TensorStreamPlan>>,
    pub(super) admission: ModelResourceAdmission,
    pub(super) retained_host: TensorStorageReservation,
    resident: Option<super::resident_metadata::Allocation>,
    _source_workspace: TensorStorageReservation,
}

struct AggregatePlan {
    plans: BTreeMap<String, TensorUploadPlan>,
    stream_plans: Option<BTreeMap<String, TensorStreamPlan>>,
    admission: ModelResourceAdmission,
}
struct PlannedTensor {
    upload: TensorUploadPlan,
    stream: Option<TensorStreamPlan>,
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
    let resident_limit = resident_limit(backend)?;
    let AggregatePlan {
        plans,
        stream_plans,
        mut admission,
    } = calculate(
        model,
        level,
        graph,
        source,
        policy,
        streamed,
        |meta, encoding| {
            if let Some(encoding) = encoding {
                backend.tensor_stream_plan(meta, encoding).map(|plan| {
                    plan.map(|stream| PlannedTensor {
                        upload: stream.upload,
                        stream: Some(stream),
                    })
                })
            } else {
                backend.tensor_upload_plan(meta).map(|plan| {
                    plan.map(|upload| PlannedTensor {
                        upload,
                        stream: None,
                    })
                })
            }
        },
    )?;
    admission.resident_device_payload_limit = resident_limit;
    if resident_limit.is_some_and(|limit| limit < admission.payload_bytes) {
        return Err(invalid(
            "resident payload envelope is smaller than matrix payload",
        ));
    }
    let resident = resident_limit
        .map(|limit| {
            let allocation = super::resident_metadata::compile(
                model,
                level,
                graph,
                source,
                admission.payload_bytes,
            )?;
            admission.resident_device_payload_plan = Some(declare_resident(
                &allocation,
                admission.payload_bytes,
                limit,
                |m, l| backend.decoder_payload_plan(m, l),
            )?);
            Ok::<_, NnError>(allocation)
        })
        .transpose()?;
    // All domains are reserved before any complete tensor is read/decompressed.
    // A later refusal drops earlier guards/partitions, including on shared ledgers.
    let payload = resources
        .payload
        .partition(admission.reserved_payload_bytes())
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
        resident,
        _source_workspace: source_workspace,
    })
}

#[allow(clippy::too_many_arguments)]
fn calculate<R: Read + Seek>(
    model: &TritModelReader<R>,
    level: &str,
    graph: &Graph,
    source: &ModelLoadPlan,
    policy: TensorUploadPolicy,
    streamed: bool,
    mut planner: impl FnMut(
        TensorMetadata<'_>,
        Option<TensorStreamEncoding>,
    ) -> Result<Option<PlannedTensor>, BackendError>,
) -> Result<AggregatePlan, NnError> {
    let mut admission = ModelResourceAdmission {
        payload_bytes: 0,
        resident_device_payload_limit: None,
        resident_device_payload_plan: None,
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
        let encoding = streamed
            .then(|| super::tensor_stream::encoding(meta).map_err(invalid))
            .transpose()?;
        let planned = planner(metadata, encoding)
            .map_err(invalid)?
            .ok_or_else(|| {
                invalid(format!(
                    "{name}: backend {} plan is unknown",
                    if streamed { "streaming" } else { "upload" }
                ))
            })?;
        let plan = planned.upload;
        if let Some(plans) = &mut stream_plans {
            let stream = planned
                .stream
                .ok_or_else(|| invalid("missing required streaming plan"))?;
            if stream.encoded_bytes != model.tensor_byte_len(level, name).map_err(invalid)? {
                return Err(invalid(format!("{name}: stream encoded geometry mismatch")));
            }
            plans.insert(name.to_owned(), stream);
        }
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
    Ok(AggregatePlan {
        plans,
        stream_plans,
        admission,
    })
}

pub(super) fn describe_streamed<R: Read + Seek>(
    model: &TritModelReader<R>,
    level: &str,
    graph: &Graph,
    source: &ModelLoadPlan,
    policy: TensorUploadPolicy,
    mut planner: impl FnMut(
        TensorMetadata<'_>,
        TensorStreamEncoding,
    ) -> Result<Option<TensorStreamPlan>, BackendError>,
) -> Result<ModelResourceAdmission, NnError> {
    calculate(
        model,
        level,
        graph,
        source,
        policy,
        true,
        |meta, encoding| {
            planner(
                meta,
                encoding.expect("streamed calculation supplies encoding"),
            )
            .map(|plan| {
                plan.map(|stream| PlannedTensor {
                    upload: stream.upload,
                    stream: Some(stream),
                })
            })
        },
    )
    .map(|plan| plan.admission)
}

fn declare_resident(
    allocation: &super::resident_metadata::Allocation,
    matrix_bytes: u64,
    limit: u64,
    mut planner: impl FnMut(
        DecoderAllocationMetadata<'_>,
        DecoderLimits,
    ) -> Result<Option<DecoderPayloadPlan>, BackendError>,
) -> Result<DecoderPayloadPlan, NnError> {
    if limit == 0 {
        return Err(invalid("resident payload ceiling must be positive"));
    }
    let plan = planner(allocation.metadata(), allocation.limits)
        .map_err(invalid)?
        .ok_or_else(|| invalid("resident structural plan is unknown"))?;
    if u64::try_from(plan.matrix_payload_bytes()).map_err(invalid)? != matrix_bytes {
        return Err(invalid(
            "resident plan matrix payload differs from upload plans",
        ));
    }
    if u64::try_from(plan.payload_bytes()).map_err(invalid)? > limit {
        return Err(invalid(
            "resident structural plan exceeds configured envelope",
        ));
    }
    Ok(plan)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn describe_streamed_resident<R: Read + Seek>(
    model: &TritModelReader<R>,
    level: &str,
    graph: &Graph,
    source: &ModelLoadPlan,
    policy: TensorUploadPolicy,
    planner: impl FnMut(
        TensorMetadata<'_>,
        TensorStreamEncoding,
    ) -> Result<Option<TensorStreamPlan>, BackendError>,
    limit: u64,
    decoder: impl FnMut(
        DecoderAllocationMetadata<'_>,
        DecoderLimits,
    ) -> Result<Option<DecoderPayloadPlan>, BackendError>,
) -> Result<ModelResourceAdmission, NnError> {
    let mut admission = describe_streamed(model, level, graph, source, policy, planner)?;
    let allocation =
        super::resident_metadata::compile(model, level, graph, source, admission.payload_bytes)?;
    admission.resident_device_payload_limit = Some(limit);
    admission.resident_device_payload_plan = Some(declare_resident(
        &allocation,
        admission.payload_bytes,
        limit,
        decoder,
    )?);
    Ok(admission)
}
