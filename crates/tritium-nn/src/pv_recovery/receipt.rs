use tritium_train::PvStepReceipt;

use super::DevicePvRecoveryError;
use super::identity::{evidence_digest, representation_digest};
use super::wire::Reader;

const MAGIC: &[u8; 5] = b"TPVR1";
const CHECKSUM_BYTES: usize = 32;
const TENSOR_RECEIPT_BYTES: usize = 132;
const FIXED_BODY_BYTES: usize = 261;

/// Deterministic evidence for one model-wide alternating P/V step.
#[derive(Clone, Debug, PartialEq)]
pub struct DevicePvRecoveryStepReceipt {
    pub(super) plan_digest: [u8; 32],
    pub(super) optimizer_step: u64,
    pub(super) tensor_receipts: Vec<PvStepReceipt>,
    pub(super) materialized_gradient_elements: usize,
    pub(super) peak_live_gradient_elements: usize,
    pub(super) peak_host_gradient_elements: usize,
    pub(super) peak_live_activation_elements: usize,
    pub(super) naive_activation_elements: usize,
    pub(super) host_representation_bytes: usize,
    pub(super) host_optimizer_bytes: usize,
    pub(super) host_campaign_bytes: usize,
    pub(super) resident_packed_bytes: usize,
    pub(super) serialized_checkpoint_bytes: usize,
    pub(super) source_state_digest: [u8; 32],
    pub(super) batch_digest: [u8; 32],
    pub(super) representation_digest: [u8; 32],
    pub(super) evidence_digest: [u8; 32],
}

impl DevicePvRecoveryStepReceipt {
    /// Exact model, parent-representation, and optimizer-recipe identity.
    #[must_use]
    pub const fn plan_digest(&self) -> [u8; 32] {
        self.plan_digest
    }

    /// Step payload ledger excludes allocator metadata and transient fitting scratch.
    /// Release physical-memory qualification must therefore use a campaign-level
    /// measured receipt rather than this kernel-step receipt alone.
    #[must_use]
    pub const fn physical_accounting_complete(&self) -> bool {
        false
    }

    /// Source code, backend/hardware, seed, tokenizer, and full physical evidence
    /// belong to the outer campaign receipt. This local step receipt cannot alone
    /// satisfy a public release-evidence gate.
    #[must_use]
    pub const fn release_evidence_complete(&self) -> bool {
        false
    }

    #[must_use]
    pub const fn optimizer_step(&self) -> u64 {
        self.optimizer_step
    }

    #[must_use]
    pub fn tensor_receipts(&self) -> &[PvStepReceipt] {
        &self.tensor_receipts
    }

    #[must_use]
    pub const fn materialized_gradient_elements(&self) -> usize {
        self.materialized_gradient_elements
    }

    #[must_use]
    pub const fn peak_live_gradient_elements(&self) -> usize {
        self.peak_live_gradient_elements
    }

    /// Maximum f32 elements staged in reusable host gradient scratch.
    #[must_use]
    pub const fn peak_host_gradient_elements(&self) -> usize {
        self.peak_host_gradient_elements
    }

    #[must_use]
    pub const fn peak_live_activation_elements(&self) -> usize {
        self.peak_live_activation_elements
    }

    #[must_use]
    pub const fn naive_activation_elements(&self) -> usize {
        self.naive_activation_elements
    }

    /// Host trit and f16-scale payload retained after this step.
    #[must_use]
    pub const fn host_representation_bytes(&self) -> usize {
        self.host_representation_bytes
    }

    /// Host f32 Adam first/second-moment payload retained after this step.
    #[must_use]
    pub const fn host_optimizer_bytes(&self) -> usize {
        self.host_optimizer_bytes
    }

    /// Host f64 in-flight accumulation payload retained after this step.
    #[must_use]
    pub const fn host_campaign_bytes(&self) -> usize {
        self.host_campaign_bytes
    }

    /// CUDA packed-code plus f32-scale payload retained for execution.
    #[must_use]
    pub const fn resident_packed_bytes(&self) -> usize {
        self.resident_packed_bytes
    }

    /// Exact canonical TPVM2 byte count for post-step durable state.
    #[must_use]
    pub const fn serialized_checkpoint_bytes(&self) -> usize {
        self.serialized_checkpoint_bytes
    }

    /// Exact pre-step PV representation and optimizer-state identity.
    #[must_use]
    pub const fn source_state_digest(&self) -> [u8; 32] {
        self.source_state_digest
    }

    /// Exact token and target payload identity for this optimizer step.
    #[must_use]
    pub const fn batch_digest(&self) -> [u8; 32] {
        self.batch_digest
    }

    #[must_use]
    pub const fn peak_device_gradient_bytes(&self) -> usize {
        self.peak_live_gradient_elements * core::mem::size_of::<f32>()
    }

    #[must_use]
    pub const fn peak_host_gradient_bytes(&self) -> usize {
        self.peak_host_gradient_elements * core::mem::size_of::<f32>()
    }

    #[must_use]
    pub const fn peak_live_activation_bytes(&self) -> usize {
        self.peak_live_activation_elements * core::mem::size_of::<f32>()
    }

    #[must_use]
    pub const fn representation_digest(&self) -> [u8; 32] {
        self.representation_digest
    }

    /// Source-, data-, plan-, step-, and representation-bound evidence identity.
    #[must_use]
    pub const fn evidence_digest(&self) -> [u8; 32] {
        self.evidence_digest
    }

    /// Canonical checksum-bound form for durable campaign evidence.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, DevicePvRecoveryError> {
        validate(self)?;
        let receipts = self
            .tensor_receipts
            .iter()
            .map(PvStepReceipt::checkpoint_bytes)
            .collect::<Vec<_>>();
        let nested_bytes = receipts
            .len()
            .checked_mul(8 + TENSOR_RECEIPT_BYTES)
            .ok_or_else(receipt_size_error)?;
        let capacity = FIXED_BODY_BYTES
            .checked_add(nested_bytes)
            .and_then(|bytes| bytes.checked_add(CHECKSUM_BYTES))
            .ok_or_else(receipt_size_error)?;
        let mut out = Vec::new();
        out.try_reserve_exact(capacity)
            .map_err(|_| receipt_error("receipt allocation failed"))?;
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&self.plan_digest);
        out.extend_from_slice(&self.optimizer_step.to_le_bytes());
        append_usize(&mut out, receipts.len())?;
        for value in self.numeric_fields() {
            append_usize(&mut out, value)?;
        }
        for digest in self.evidence_fields() {
            out.extend_from_slice(&digest);
        }
        for receipt in receipts {
            if receipt.len() != TENSOR_RECEIPT_BYTES {
                return Err(receipt_error(
                    "tensor receipt has unexpected canonical length",
                ));
            }
            append_usize(&mut out, receipt.len())?;
            out.extend_from_slice(&receipt);
        }
        let checksum = blake3::hash(&out);
        out.extend_from_slice(checksum.as_bytes());
        debug_assert_eq!(out.len(), capacity);
        Ok(out)
    }

    /// Strictly reopen canonical model-step evidence.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, DevicePvRecoveryError> {
        let body_len = bytes
            .len()
            .checked_sub(CHECKSUM_BYTES)
            .ok_or_else(|| receipt_error("receipt is truncated"))?;
        let (body, checksum) = bytes.split_at(body_len);
        if blake3::hash(body).as_bytes() != checksum {
            return Err(receipt_error("receipt checksum mismatch"));
        }
        let mut reader = Reader::new(body);
        if reader.take(MAGIC.len())? != MAGIC {
            return Err(receipt_error("bad receipt magic or version"));
        }
        let plan_digest = reader.array()?;
        let optimizer_step = reader.u64()?;
        let tensor_count = reader.usize()?;
        let materialized_gradient_elements = reader.usize()?;
        let peak_live_gradient_elements = reader.usize()?;
        let peak_host_gradient_elements = reader.usize()?;
        let peak_live_activation_elements = reader.usize()?;
        let naive_activation_elements = reader.usize()?;
        let host_representation_bytes = reader.usize()?;
        let host_optimizer_bytes = reader.usize()?;
        let host_campaign_bytes = reader.usize()?;
        let resident_packed_bytes = reader.usize()?;
        let serialized_checkpoint_bytes = reader.usize()?;
        let source_state_digest = reader.array()?;
        let batch_digest = reader.array()?;
        let representation_digest = reader.array()?;
        let evidence_digest = reader.array()?;
        let expected_nested = tensor_count
            .checked_mul(8 + TENSOR_RECEIPT_BYTES)
            .ok_or_else(receipt_size_error)?;
        if tensor_count == 0 || reader.remaining() != expected_nested {
            return Err(receipt_error(
                "receipt tensor count or payload length is invalid",
            ));
        }
        let mut tensor_receipts = Vec::new();
        tensor_receipts
            .try_reserve_exact(tensor_count)
            .map_err(|_| receipt_error("tensor receipt allocation failed"))?;
        for _ in 0..tensor_count {
            let length = reader.usize()?;
            if length != TENSOR_RECEIPT_BYTES {
                return Err(receipt_error("tensor receipt length is not canonical"));
            }
            tensor_receipts.push(PvStepReceipt::resume(reader.take(length)?)?);
        }
        if reader.remaining() != 0 {
            return Err(receipt_error("receipt has trailing bytes"));
        }
        let receipt = Self {
            plan_digest,
            optimizer_step,
            tensor_receipts,
            materialized_gradient_elements,
            peak_live_gradient_elements,
            peak_host_gradient_elements,
            peak_live_activation_elements,
            naive_activation_elements,
            host_representation_bytes,
            host_optimizer_bytes,
            host_campaign_bytes,
            resident_packed_bytes,
            serialized_checkpoint_bytes,
            source_state_digest,
            batch_digest,
            representation_digest,
            evidence_digest,
        };
        validate(&receipt)?;
        Ok(receipt)
    }

    fn numeric_fields(&self) -> [usize; 10] {
        [
            self.materialized_gradient_elements,
            self.peak_live_gradient_elements,
            self.peak_host_gradient_elements,
            self.peak_live_activation_elements,
            self.naive_activation_elements,
            self.host_representation_bytes,
            self.host_optimizer_bytes,
            self.host_campaign_bytes,
            self.resident_packed_bytes,
            self.serialized_checkpoint_bytes,
        ]
    }

    fn evidence_fields(&self) -> [[u8; 32]; 4] {
        [
            self.source_state_digest,
            self.batch_digest,
            self.representation_digest,
            self.evidence_digest,
        ]
    }
}

pub(super) fn canonical_receipt_digest(
    receipt: &DevicePvRecoveryStepReceipt,
) -> Result<[u8; 32], DevicePvRecoveryError> {
    Ok(*blake3::hash(&receipt.canonical_bytes()?).as_bytes())
}

fn validate(receipt: &DevicePvRecoveryStepReceipt) -> Result<(), DevicePvRecoveryError> {
    if receipt.optimizer_step == 0 || receipt.tensor_receipts.is_empty() {
        return Err(receipt_error(
            "receipt step and tensor set must be non-zero",
        ));
    }
    if [
        receipt.plan_digest,
        receipt.source_state_digest,
        receipt.batch_digest,
        receipt.representation_digest,
        receipt.evidence_digest,
    ]
    .contains(&[0; 32])
    {
        return Err(receipt_error("receipt identity digest is missing"));
    }
    if receipt
        .tensor_receipts
        .iter()
        .any(|tensor| tensor.optimizer_step() != receipt.optimizer_step)
    {
        return Err(receipt_error("tensor receipt optimizer step mismatch"));
    }
    if receipt.materialized_gradient_elements == 0
        || receipt.peak_live_gradient_elements == 0
        || receipt.peak_host_gradient_elements == 0
        || receipt.peak_live_gradient_elements > receipt.materialized_gradient_elements
        || receipt.peak_host_gradient_elements > receipt.peak_live_gradient_elements
        || receipt.peak_live_activation_elements > receipt.naive_activation_elements
        || receipt.host_representation_bytes == 0
        || receipt.host_optimizer_bytes == 0
        || receipt.host_campaign_bytes != 0
        || receipt.resident_packed_bytes == 0
        || receipt.serialized_checkpoint_bytes == 0
    {
        return Err(receipt_error("receipt physical counters are inconsistent"));
    }
    for elements in [
        receipt.materialized_gradient_elements,
        receipt.peak_live_gradient_elements,
        receipt.peak_host_gradient_elements,
        receipt.peak_live_activation_elements,
        receipt.naive_activation_elements,
    ] {
        elements
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| receipt_error("receipt element count overflows byte accounting"))?;
    }
    let expected_representation = representation_digest(
        receipt.plan_digest,
        receipt.optimizer_step,
        receipt
            .tensor_receipts
            .iter()
            .map(PvStepReceipt::representation_digest),
    );
    if receipt.representation_digest != expected_representation {
        return Err(receipt_error("receipt representation identity mismatch"));
    }
    let expected_evidence = evidence_digest(
        receipt.plan_digest,
        receipt.source_state_digest,
        receipt.batch_digest,
        receipt.optimizer_step,
        receipt.representation_digest,
    );
    if receipt.evidence_digest != expected_evidence {
        return Err(receipt_error("receipt evidence identity mismatch"));
    }
    Ok(())
}

fn append_usize(out: &mut Vec<u8>, value: usize) -> Result<(), DevicePvRecoveryError> {
    let value = u64::try_from(value)
        .map_err(|_| receipt_error("receipt value exceeds canonical wire range"))?;
    out.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn receipt_size_error() -> DevicePvRecoveryError {
    receipt_error("receipt size overflows host range")
}

fn receipt_error(reason: &str) -> DevicePvRecoveryError {
    DevicePvRecoveryError::Checkpoint(reason.into())
}
