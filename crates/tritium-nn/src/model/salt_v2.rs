//! Strict assembly of a runnable model from a SALT V2 package.
//!
//! The package owns every two-dimensional model tensor. Configuration and the
//! preserved one-dimensional norms, biases, and QK-normalization weights come
//! from the Hugging Face model directory. Package tensors stream directly into
//! their final CUDA representation; the loader retains no semantic trit or dense
//! weight shadow.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use tritium_cuda::{CudaBackend, SaltV2ResidentTensor};
use tritium_format::salt_v2::SaltV2Codec;
use tritium_format::salt_v2_package::{
    SaltV2IndexedRuntimeLedger, SaltV2PackageReader, SaltV2TensorInfo, SaltV2Transform,
};
use tritium_format::{PackageHasher, PackageId, SemanticTensor};

use super::ModelWeights;
use super::hf::{
    DenseTensorRequest, NameSchema, build_standard_model_with_embedding, declared_vocab_size,
    read_config_json, resolve_optional_attention_weights,
};
use super::hf_shards::HfShardSet;
use crate::config::ModelConfig;
use crate::error::NnError;
use crate::layers::{Projection, TokenEmbedding};

const PACKAGE_IO_BUFFER_BYTES: usize = 64 * 1024;

/// One package tensor proven resident in its final indexed CUDA representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SaltV2LoadedTensorReceipt {
    name: String,
    runtime: SaltV2IndexedRuntimeLedger,
}

impl SaltV2LoadedTensorReceipt {
    /// Canonical package tensor name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Exact requested device-allocation ledger returned by the resident handle.
    #[must_use]
    pub const fn runtime(&self) -> SaltV2IndexedRuntimeLedger {
        self.runtime
    }
}

/// One preserved fp32 vector measured from the actual loaded safetensors bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SaltV2PreservedTensorReceipt {
    name: String,
    elements: u64,
    requested_bytes: u64,
    content_digest: [u8; 32],
}

impl SaltV2PreservedTensorReceipt {
    /// Canonical Hugging Face tensor name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Number of retained fp32 elements.
    #[must_use]
    pub const fn elements(&self) -> u64 {
        self.elements
    }

    /// Exact logical fp32 bytes requested by the retained vector.
    #[must_use]
    pub const fn requested_bytes(&self) -> u64 {
        self.requested_bytes
    }

    /// Domain-separated semantic-tensor digest of canonical little-endian fp32 bits.
    #[must_use]
    pub const fn content_digest(&self) -> &[u8; 32] {
        &self.content_digest
    }
}

/// Content-bound whole-model weight-allocation evidence for a SALT V2 load.
///
/// Runtime ledgers are stored in canonical package order so they can be passed
/// directly to physical-size verification. Preserved fp32 vectors are sorted by
/// name. Byte totals are requested logical allocation lengths: allocator pool
/// rounding, CUDA context/module memory, KV cache, and transient workspaces are
/// deliberately outside this weight receipt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SaltV2ModelAllocationReceipt {
    package_id: PackageId,
    codec: SaltV2Codec,
    tensors: Vec<SaltV2LoadedTensorReceipt>,
    runtime_ledgers: Vec<SaltV2IndexedRuntimeLedger>,
    preserved: Vec<SaltV2PreservedTensorReceipt>,
    quantized_parameters: u64,
    preserved_parameters: u64,
    payload_bytes: u64,
    scale_bytes: u64,
    map_bytes: u64,
    rank_prefix_bytes: u64,
    v2_resident_bytes: u64,
    preserved_fp32_bytes: u64,
    tracked_weight_bytes: u64,
}

impl SaltV2ModelAllocationReceipt {
    /// Exact package-byte identity required by the caller and rechecked after load.
    #[must_use]
    pub const fn package_id(&self) -> PackageId {
        self.package_id
    }

    /// Physical codec shared by every package tensor.
    #[must_use]
    pub const fn codec(&self) -> SaltV2Codec {
        self.codec
    }

    /// Actual resident tensor receipts in canonical package order.
    #[must_use]
    pub fn tensors(&self) -> &[SaltV2LoadedTensorReceipt] {
        &self.tensors
    }

    /// Runtime ledgers in canonical package order, ready for physical-size verification.
    #[must_use]
    pub fn runtime_ledgers(&self) -> &[SaltV2IndexedRuntimeLedger] {
        &self.runtime_ledgers
    }

    /// Preserved fp32 vectors sorted by canonical name.
    #[must_use]
    pub fn preserved_tensors(&self) -> &[SaltV2PreservedTensorReceipt] {
        &self.preserved
    }

    /// Coefficients stored in additive SALT V2 tensors.
    #[must_use]
    pub const fn quantized_parameters(&self) -> u64 {
        self.quantized_parameters
    }

    /// Elements retained as preserved fp32 vectors.
    #[must_use]
    pub const fn preserved_parameters(&self) -> u64 {
        self.preserved_parameters
    }

    /// Exact resident codec-payload bytes.
    #[must_use]
    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    /// Exact resident group128 fp16-scale bytes.
    #[must_use]
    pub const fn scale_bytes(&self) -> u64 {
        self.scale_bytes
    }

    /// Exact allocated two-bit map bytes, excluding scalar-carried tail bits.
    #[must_use]
    pub const fn map_bytes(&self) -> u64 {
        self.map_bytes
    }

    /// Exact allocated coarse rank-prefix bytes.
    #[must_use]
    pub const fn rank_prefix_bytes(&self) -> u64 {
        self.rank_prefix_bytes
    }

    /// Sum of every SALT V2 tensor's final device allocations.
    #[must_use]
    pub const fn v2_resident_bytes(&self) -> u64 {
        self.v2_resident_bytes
    }

    /// Exact logical bytes requested by preserved host fp32 vectors.
    #[must_use]
    pub const fn preserved_fp32_bytes(&self) -> u64 {
        self.preserved_fp32_bytes
    }

    /// Tracked steady weight bytes across final device tensors and preserved host vectors.
    #[must_use]
    pub const fn tracked_weight_bytes(&self) -> u64 {
        self.tracked_weight_bytes
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct RuntimeTotals {
    quantized_parameters: u64,
    payload_bytes: u64,
    scale_bytes: u64,
    map_bytes: u64,
    rank_prefix_bytes: u64,
    resident_bytes: u64,
}

impl RuntimeTotals {
    fn checked_add(
        &mut self,
        coefficients: u64,
        runtime: SaltV2IndexedRuntimeLedger,
    ) -> Result<(), NnError> {
        self.quantized_parameters = checked_add(
            self.quantized_parameters,
            coefficients,
            "SALT V2 parameter count",
        )?;
        self.payload_bytes = checked_add(
            self.payload_bytes,
            runtime.payload_bytes(),
            "SALT V2 payload bytes",
        )?;
        self.scale_bytes = checked_add(
            self.scale_bytes,
            runtime.scale_bytes(),
            "SALT V2 scale bytes",
        )?;
        self.map_bytes = checked_add(
            self.map_bytes,
            runtime.allocation_map_bytes(),
            "SALT V2 map bytes",
        )?;
        self.rank_prefix_bytes = checked_add(
            self.rank_prefix_bytes,
            runtime.rank_prefix_bytes(),
            "SALT V2 rank-prefix bytes",
        )?;
        self.resident_bytes = checked_add(
            self.resident_bytes,
            runtime.steady_resident_bytes(),
            "SALT V2 resident bytes",
        )?;
        if runtime.dense_shadow_bytes() != 0 {
            return Err(NnError::Backend(
                "SALT V2 resident receipt contains a dense weight shadow".to_owned(),
            ));
        }
        Ok(())
    }
}

impl ModelWeights {
    /// Load a complete SALT V2 standard-transformer model onto one CUDA backend.
    ///
    /// `expected_package_id` must come from the fit/conversion receipt. The exact
    /// package bytes are checked before indexing and again after every required
    /// tensor has streamed into its final device allocation. Every package tensor
    /// must map to exactly one model-owned two-dimensional weight; missing, extra,
    /// duplicate, transformed, or shape-disagreeing tensors fail transactionally.
    /// Qwen QKV biases and QK norms are retained from the master safetensors.
    ///
    /// # Errors
    /// Returns [`NnError`] for configuration/shard failures, package identity or
    /// coverage disagreement, invalid tensor geometry, allocation failure, or a
    /// CUDA upload error. No model is returned on a partial load.
    pub fn load_salt_v2(
        model_dir: &Path,
        package_path: &Path,
        expected_package_id: PackageId,
        cuda: &CudaBackend,
    ) -> Result<(ModelConfig, ModelWeights, SaltV2ModelAllocationReceipt), NnError> {
        let cfg_json = read_config_json(&model_dir.join("config.json"))?;
        let (config, mut spec) = ModelConfig::from_hf_config(&cfg_json)?;
        let config_value: serde_json::Value = serde_json::from_str(&cfg_json)
            .map_err(|error| NnError::MissingConfig(format!("invalid config.json: {error}")))?;
        let declared_vocab = declared_vocab_size(&config_value)?;

        let shards = HfShardSet::open(model_dir)?;
        (spec.qkv_bias, spec.qk_norm) =
            resolve_optional_attention_weights(&config, &config_value, &shards)?;

        let file = File::open(package_path).map_err(|error| {
            NnError::MissingTensor(format!("open {}: {error}", package_path.display()))
        })?;
        let metadata = file.metadata().map_err(|error| {
            NnError::MissingTensor(format!("stat {}: {error}", package_path.display()))
        })?;
        if !metadata.is_file() {
            return Err(NnError::MissingTensor(format!(
                "{} is not a regular file",
                package_path.display()
            )));
        }
        let mut source = BufReader::with_capacity(PACKAGE_IO_BUFFER_BYTES, file);
        let initial_id = hash_package(&mut source, package_path)?;
        if initial_id != expected_package_id {
            return Err(package_identity_error(
                package_path,
                expected_package_id,
                initial_id,
            ));
        }
        source.seek(SeekFrom::Start(0)).map_err(|error| {
            NnError::MissingTensor(format!("rewind {}: {error}", package_path.display()))
        })?;
        let mut reader = SaltV2PackageReader::new_strict(source).map_err(|error| {
            NnError::MissingTensor(format!("index {}: {error}", package_path.display()))
        })?;
        if reader.package_id() != expected_package_id {
            return Err(package_identity_error(
                package_path,
                expected_package_id,
                reader.package_id(),
            ));
        }
        let codec = reader.codec();
        let package_order = reader
            .tensor_names_encoded_order()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let package_names = package_order.iter().cloned().collect::<BTreeSet<_>>();
        let mut actual = BTreeMap::<String, SaltV2IndexedRuntimeLedger>::new();

        let n_embd = config.n_embd as usize;
        let embedding_name = NameSchema::Hf.top("token_embd");
        let embedding = upload_matrix(
            &mut reader,
            cuda,
            embedding_name,
            declared_vocab,
            n_embd,
            &mut actual,
        )?;
        let token_embd = TokenEmbedding::from_salt_v2_resident(embedding)?;

        let preserved = RefCell::new(BTreeMap::<String, SaltV2PreservedTensorReceipt>::new());
        let dense = |name: &str, request: DenseTensorRequest| -> Result<Vec<f32>, NnError> {
            let DenseTensorRequest::Vector { len } = request else {
                return Err(NnError::Backend(format!(
                    "unexpected dense token-table request for `{name}`"
                )));
            };
            let values = shards.tensor_f32_exact(name, &[len])?;
            let receipt = preserved_tensor_receipt(name, &values)?;
            let replaced = preserved
                .try_borrow_mut()
                .map_err(|_| NnError::Backend("reentrant preserved-tensor load".to_owned()))?
                .insert(name.to_owned(), receipt);
            if replaced.is_some() {
                return Err(NnError::Backend(format!(
                    "preserved tensor `{name}` was requested more than once"
                )));
            }
            Ok(values)
        };

        let weights = build_standard_model_with_embedding(
            &config,
            &spec,
            NameSchema::Hf,
            token_embd,
            dense,
            |name, rows, columns| {
                upload_matrix(&mut reader, cuda, name, Some(rows), columns, &mut actual)
                    .map(Projection::SaltV2)
            },
        )?;

        let loaded_names = actual.keys().cloned().collect::<BTreeSet<_>>();
        if loaded_names != package_names {
            let missing = package_names
                .difference(&loaded_names)
                .take(4)
                .cloned()
                .collect::<Vec<_>>();
            let extra = loaded_names
                .difference(&package_names)
                .take(4)
                .cloned()
                .collect::<Vec<_>>();
            return Err(NnError::Backend(format!(
                "SALT V2 model/package tensor coverage differs: unconsumed package tensors={missing:?}, model tensors absent from package={extra:?}"
            )));
        }

        let mut runtime_totals = RuntimeTotals::default();
        let mut tensors = Vec::new();
        let mut runtime_ledgers = Vec::new();
        tensors
            .try_reserve_exact(package_order.len())
            .map_err(|error| {
                NnError::Backend(format!("allocate SALT V2 tensor receipts: {error}"))
            })?;
        runtime_ledgers
            .try_reserve_exact(package_order.len())
            .map_err(|error| {
                NnError::Backend(format!("allocate SALT V2 runtime ledgers: {error}"))
            })?;
        for name in &package_order {
            let info = reader.tensor_info(name).ok_or_else(|| {
                NnError::Backend(format!("indexed SALT V2 tensor `{name}` disappeared"))
            })?;
            let runtime = *actual.get(name).ok_or_else(|| {
                NnError::Backend(format!("SALT V2 tensor `{name}` has no resident receipt"))
            })?;
            runtime_totals.checked_add(
                u64::try_from(info.logical_coefficients()).map_err(|_| {
                    NnError::Backend(format!("SALT V2 tensor `{name}` coefficient overflow"))
                })?,
                runtime,
            )?;
            tensors.push(SaltV2LoadedTensorReceipt {
                name: name.clone(),
                runtime,
            });
            runtime_ledgers.push(runtime);
        }

        let preserved = preserved.into_inner().into_values().collect::<Vec<_>>();
        let mut preserved_parameters = 0_u64;
        let mut preserved_fp32_bytes = 0_u64;
        for tensor in &preserved {
            preserved_parameters = checked_add(
                preserved_parameters,
                tensor.elements,
                "preserved parameter count",
            )?;
            preserved_fp32_bytes = checked_add(
                preserved_fp32_bytes,
                tensor.requested_bytes,
                "preserved fp32 bytes",
            )?;
        }

        let mut source = reader.into_inner();
        source.seek(SeekFrom::Start(0)).map_err(|error| {
            NnError::MissingTensor(format!(
                "rewind {} after load: {error}",
                package_path.display()
            ))
        })?;
        let terminal_id = hash_package(&mut source, package_path)?;
        if terminal_id != expected_package_id {
            return Err(package_identity_error(
                package_path,
                expected_package_id,
                terminal_id,
            ));
        }

        let receipt = SaltV2ModelAllocationReceipt {
            package_id: expected_package_id,
            codec,
            tensors,
            runtime_ledgers,
            preserved,
            quantized_parameters: runtime_totals.quantized_parameters,
            preserved_parameters,
            payload_bytes: runtime_totals.payload_bytes,
            scale_bytes: runtime_totals.scale_bytes,
            map_bytes: runtime_totals.map_bytes,
            rank_prefix_bytes: runtime_totals.rank_prefix_bytes,
            v2_resident_bytes: runtime_totals.resident_bytes,
            preserved_fp32_bytes,
            tracked_weight_bytes: checked_add(
                runtime_totals.resident_bytes,
                preserved_fp32_bytes,
                "tracked SALT V2 weight bytes",
            )?,
        };
        Ok((config, weights, receipt))
    }
}

fn upload_matrix(
    reader: &mut SaltV2PackageReader<BufReader<File>>,
    cuda: &CudaBackend,
    name: &str,
    expected_rows: Option<usize>,
    expected_columns: usize,
    actual: &mut BTreeMap<String, SaltV2IndexedRuntimeLedger>,
) -> Result<Arc<SaltV2ResidentTensor>, NnError> {
    if actual.contains_key(name) {
        return Err(NnError::Backend(format!(
            "SALT V2 tensor `{name}` was requested more than once"
        )));
    }
    let info = reader
        .tensor_info(name)
        .cloned()
        .ok_or_else(|| NnError::MissingTensor(name.to_owned()))?;
    validate_matrix_info(name, &info, expected_rows, expected_columns)?;
    let planned = info.runtime_ledger();
    let resident = cuda
        .upload_salt_v2_from_reader(reader, name)
        .map_err(|error| NnError::Backend(format!("upload SALT V2 tensor `{name}`: {error}")))?;
    let measured = resident.allocation_receipt();
    if measured.codec() != reader.codec() || measured.runtime_ledger() != planned {
        return Err(NnError::Backend(format!(
            "resident SALT V2 receipt for `{name}` disagrees with the opened package"
        )));
    }
    actual.insert(name.to_owned(), measured.runtime_ledger());
    Ok(Arc::new(resident))
}

fn validate_matrix_info(
    name: &str,
    info: &SaltV2TensorInfo,
    expected_rows: Option<usize>,
    expected_columns: usize,
) -> Result<(), NnError> {
    if !matches!(info.transform(), SaltV2Transform::None) {
        return Err(NnError::Backend(format!(
            "SALT V2 tensor `{name}` requires unsupported transform {:?}",
            info.transform()
        )));
    }
    if info.dims().len() != 2 {
        return Err(NnError::Shape {
            expected: 2,
            got: info.dims().len(),
        });
    }
    let rows = usize::try_from(info.dims()[0])
        .map_err(|_| NnError::Backend(format!("SALT V2 tensor `{name}` rows overflow usize")))?;
    let columns = usize::try_from(info.dims()[1])
        .map_err(|_| NnError::Backend(format!("SALT V2 tensor `{name}` columns overflow usize")))?;
    if let Some(expected) = expected_rows
        && rows != expected
    {
        return Err(NnError::Shape {
            expected: expected.saturating_mul(expected_columns),
            got: rows.saturating_mul(columns),
        });
    }
    if columns != expected_columns {
        return Err(NnError::Shape {
            expected: expected_rows
                .unwrap_or(rows)
                .saturating_mul(expected_columns),
            got: rows.saturating_mul(columns),
        });
    }
    Ok(())
}

fn preserved_tensor_receipt(
    name: &str,
    values: &[f32],
) -> Result<SaltV2PreservedTensorReceipt, NnError> {
    let byte_len = values
        .len()
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| NnError::Backend(format!("preserved tensor `{name}` byte overflow")))?;
    let mut logical_bytes = Vec::new();
    logical_bytes.try_reserve_exact(byte_len).map_err(|error| {
        NnError::Backend(format!(
            "allocate {byte_len} digest bytes for preserved tensor `{name}`: {error}"
        ))
    })?;
    for value in values {
        logical_bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    let elements = u64::try_from(values.len())
        .map_err(|_| NnError::Backend(format!("preserved tensor `{name}` length overflow")))?;
    let semantic = SemanticTensor::new(name, vec![elements], &logical_bytes).map_err(|error| {
        NnError::Backend(format!("identify preserved tensor `{name}`: {error}"))
    })?;
    Ok(SaltV2PreservedTensorReceipt {
        name: name.to_owned(),
        elements,
        requested_bytes: u64::try_from(byte_len).map_err(|_| {
            NnError::Backend(format!("preserved tensor `{name}` byte count overflow"))
        })?,
        content_digest: *semantic.content_digest(),
    })
}

fn hash_package(source: &mut BufReader<File>, package_path: &Path) -> Result<PackageId, NnError> {
    let mut hasher = PackageHasher::new();
    let mut buffer = [0_u8; PACKAGE_IO_BUFFER_BYTES];
    loop {
        let count = source.read(&mut buffer).map_err(|error| {
            NnError::MissingTensor(format!("hash {}: {error}", package_path.display()))
        })?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hasher.finalize())
}

fn package_identity_error(
    package_path: &Path,
    expected: PackageId,
    measured: PackageId,
) -> NnError {
    NnError::MissingTensor(format!(
        "{} identity mismatch: expected {expected}, measured {measured}",
        package_path.display()
    ))
}

fn checked_add(left: u64, right: u64, field: &str) -> Result<u64, NnError> {
    left.checked_add(right)
        .ok_or_else(|| NnError::Backend(format!("{field} overflow")))
}
