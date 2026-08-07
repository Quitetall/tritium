//! Python boundary for strictly admitted Stage-7 token evidence.

use pyo3::{
    create_exception,
    exceptions::{PyMemoryError, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyBytes,
};
use tritium_salt::{
    STAGE7_TOKENS_PER_SEQUENCE, Stage7EvidenceError, Stage7Partition,
    Stage7TokenBatch as NativeStage7TokenBatch,
    Stage7TokenEvidencePack as NativeStage7TokenEvidencePack,
    Stage7TokenEvidenceReceipt as NativeStage7TokenEvidenceReceipt,
};

create_exception!(
    tritium._tritium,
    Stage7EvidenceContractError,
    PyValueError,
    "A Stage-7 token pack violates its frozen evidence contract."
);
create_exception!(
    tritium._tritium,
    Stage7EvidenceIoError,
    PyRuntimeError,
    "A Stage-7 token pack could not be read or revalidated."
);
create_exception!(
    tritium._tritium,
    Stage7EvidenceStateError,
    PyRuntimeError,
    "A terminal Stage-7 token reader was used again."
);

/// Immutable identity established by strict Stage-7 pack admission.
#[pyclass(frozen, module = "tritium._tritium", skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct Stage7TokenEvidenceReceipt {
    pack_id: String,
    tokenizer_digest: String,
    tokenizer_vocab_size: u32,
    token_payload_sha256: String,
}

impl From<&NativeStage7TokenEvidenceReceipt> for Stage7TokenEvidenceReceipt {
    fn from(receipt: &NativeStage7TokenEvidenceReceipt) -> Self {
        Self {
            pack_id: receipt.pack_id().to_owned(),
            tokenizer_digest: receipt.tokenizer_digest().to_owned(),
            tokenizer_vocab_size: receipt.tokenizer_vocab_size(),
            token_payload_sha256: receipt.token_payload_sha256().to_owned(),
        }
    }
}

#[pymethods]
impl Stage7TokenEvidenceReceipt {
    /// Canonical campaign pack identity.
    #[getter]
    fn pack_id(&self) -> &str {
        &self.pack_id
    }

    /// Canonical tokenizer asset-tree identity.
    #[getter]
    fn tokenizer_digest(&self) -> &str {
        &self.tokenizer_digest
    }

    /// Vocabulary ceiling enforced during admission and every read.
    #[getter]
    const fn tokenizer_vocab_size(&self) -> u32 {
        self.tokenizer_vocab_size
    }

    /// SHA-256 identity of exact token payload bytes.
    #[getter]
    fn token_payload_sha256(&self) -> &str {
        &self.token_payload_sha256
    }

    fn __repr__(&self) -> String {
        format!(
            "Stage7TokenEvidenceReceipt(pack_id='{}', tokenizer_vocab_size={})",
            self.pack_id, self.tokenizer_vocab_size
        )
    }
}

/// One bounded sequence window read from retained Stage-7 payload handle.
#[pyclass(frozen, module = "tritium._tritium", skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct Stage7TokenBatch {
    partition: String,
    sampling_seed: u64,
    start_sequence: usize,
    sequence_ids: Vec<String>,
    ordered_token_sha256: String,
    tokens: Vec<u32>,
}

impl From<NativeStage7TokenBatch> for Stage7TokenBatch {
    fn from(batch: NativeStage7TokenBatch) -> Self {
        Self {
            partition: batch.partition().as_str().to_owned(),
            sampling_seed: batch.sampling_seed(),
            start_sequence: batch.start_sequence(),
            sequence_ids: batch.sequence_ids().to_vec(),
            ordered_token_sha256: batch.ordered_token_sha256().to_owned(),
            tokens: batch.tokens().to_vec(),
        }
    }
}

#[pymethods]
impl Stage7TokenBatch {
    /// Frozen partition name.
    #[getter]
    fn partition(&self) -> &str {
        &self.partition
    }

    /// Frozen partition sampling seed.
    #[getter]
    const fn sampling_seed(&self) -> u64 {
        self.sampling_seed
    }

    /// First selected sequence ordinal.
    #[getter]
    const fn start_sequence(&self) -> usize {
        self.start_sequence
    }

    /// Number of complete sequences.
    #[getter]
    fn sequence_count(&self) -> usize {
        self.sequence_ids.len()
    }

    /// Frozen token geometry per sequence.
    #[getter]
    const fn tokens_per_sequence(&self) -> usize {
        STAGE7_TOKENS_PER_SEQUENCE
    }

    /// Exact selected token count.
    #[getter]
    fn token_count(&self) -> usize {
        self.tokens.len()
    }

    /// Ordered sequence identities.
    #[getter]
    fn sequence_ids(&self) -> Vec<String> {
        self.sequence_ids.clone()
    }

    /// SHA-256 identity of exact concatenated selected token bytes.
    #[getter]
    fn ordered_token_sha256(&self) -> &str {
        &self.ordered_token_sha256
    }

    /// Canonical little-endian `u32` token payload without Python scalar expansion.
    #[getter]
    fn tokens_u32le(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let encoded = py.detach(|| {
            let byte_count = self
                .tokens
                .len()
                .checked_mul(core::mem::size_of::<u32>())
                .ok_or(())?;
            let mut encoded = Vec::new();
            encoded.try_reserve_exact(byte_count).map_err(|_| ())?;
            for token in &self.tokens {
                encoded.extend_from_slice(&token.to_le_bytes());
            }
            Ok::<_, ()>(encoded)
        });
        let encoded =
            encoded.map_err(|()| PyMemoryError::new_err("allocate Stage-7 token bytes failed"))?;
        Ok(PyBytes::new(py, &encoded).unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "Stage7TokenBatch(partition='{}', start_sequence={}, sequence_count={})",
            self.partition,
            self.start_sequence,
            self.sequence_ids.len()
        )
    }
}

/// Stateful, same-handle Stage-7 token evidence reader.
#[pyclass(module = "tritium._tritium")]
pub(crate) struct Stage7TokenEvidencePack {
    pack: Option<NativeStage7TokenEvidencePack>,
}

#[pymethods]
impl Stage7TokenEvidencePack {
    /// Strictly admit one campaign- and tokenizer-bound token pack.
    #[new]
    fn new(
        py: Python<'_>,
        manifest_path: &str,
        expected_pack_id: &str,
        expected_tokenizer_digest: &str,
        expected_tokenizer_vocab_size: u32,
    ) -> PyResult<Self> {
        if manifest_path.is_empty() {
            return Err(Stage7EvidenceContractError::new_err(
                "manifest_path must not be empty",
            ));
        }
        let manifest_path = manifest_path.to_owned();
        let expected_pack_id = expected_pack_id.to_owned();
        let expected_tokenizer_digest = expected_tokenizer_digest.to_owned();
        let pack = py
            .detach(move || {
                NativeStage7TokenEvidencePack::open(
                    manifest_path,
                    &expected_pack_id,
                    &expected_tokenizer_digest,
                    expected_tokenizer_vocab_size,
                )
            })
            .map_err(stage7_error)?;
        Ok(Self { pack: Some(pack) })
    }

    /// Construction-time content identity.
    #[getter]
    fn receipt(&self) -> PyResult<Stage7TokenEvidenceReceipt> {
        self.pack
            .as_ref()
            .map(|pack| pack.receipt().into())
            .ok_or_else(stage7_state_error)
    }

    /// Read one nonempty bounded sequence window from retained payload handle.
    fn read_sequences(
        &mut self,
        py: Python<'_>,
        partition: &str,
        start_sequence: usize,
        sequence_count: usize,
    ) -> PyResult<Stage7TokenBatch> {
        let partition = partition.parse::<Stage7Partition>().map_err(stage7_error)?;
        let pack = self.pack.as_mut().ok_or_else(stage7_state_error)?;
        py.detach(|| {
            pack.read_sequences(partition, start_sequence, sequence_count)
                .map(Into::into)
        })
        .map_err(stage7_error)
    }

    /// Terminally rehash retained payload handle and close reader.
    fn finish(&mut self, py: Python<'_>) -> PyResult<Stage7TokenEvidenceReceipt> {
        let pack = self.pack.take().ok_or_else(stage7_state_error)?;
        py.detach(|| pack.finish())
            .map(|receipt| (&receipt).into())
            .map_err(stage7_error)
    }

    fn __repr__(&self) -> String {
        format!("Stage7TokenEvidencePack(active={})", self.pack.is_some())
    }
}

fn stage7_error(error: Stage7EvidenceError) -> PyErr {
    match error {
        Stage7EvidenceError::Io { .. } => Stage7EvidenceIoError::new_err(error.to_string()),
        Stage7EvidenceError::Json(_) | Stage7EvidenceError::Invalid(_) => {
            Stage7EvidenceContractError::new_err(error.to_string())
        }
    }
}

fn stage7_state_error() -> PyErr {
    Stage7EvidenceStateError::new_err("Stage-7 token reader is already terminal")
}

pub(crate) fn register_exceptions(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add(
        "Stage7EvidenceContractError",
        module.py().get_type::<Stage7EvidenceContractError>(),
    )?;
    module.add(
        "Stage7EvidenceIoError",
        module.py().get_type::<Stage7EvidenceIoError>(),
    )?;
    module.add(
        "Stage7EvidenceStateError",
        module.py().get_type::<Stage7EvidenceStateError>(),
    )?;
    Ok(())
}
