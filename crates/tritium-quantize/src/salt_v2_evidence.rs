//! Durable factorized curvature evidence for bounded-memory SALT V2 fitting.

use core::{fmt, mem::size_of};
use std::io::{self, Read, Write};

use crate::{
    CurvatureArtifact, CurvatureSourceId, DensePsdMetric, SaltV2Curvature, SaltV2TensorFitInput,
};

const MAGIC: [u8; 4] = *b"S2KF";
const VERSION: u16 = 1;
const CHECKSUM_CONTEXT: &str = "tritium salt v2 kronecker evidence checksum v1";
const CHECKSUM_BYTES: usize = 32;
const MAX_NAME_BYTES: usize = 1024 * 1024;
const GROUP_SIZE: usize = 128;
const FIXED_PAYLOAD_BYTES: usize = 184;
const GROUP_PAYLOAD_BYTES: usize = size_of::<u32>() + GROUP_SIZE * GROUP_SIZE * size_of::<f64>();

/// Canonical, source-bound factorized curvature for one additive tensor.
#[derive(Clone, Debug)]
pub struct SaltV2KroneckerEvidence {
    kind: SaltV2Curvature,
    source_id: CurvatureSourceId,
    upstream_evidence_digest: [u8; 32],
    tensor_index: u64,
    tensor_name: String,
    rows: usize,
    columns: usize,
    input_groups: Vec<DensePsdMetric>,
    output_weights: Vec<f64>,
    damping: f64,
    record_digest: [u8; 32],
}

impl SaltV2KroneckerEvidence {
    /// Validate and canonicalize one factorized evidence record.
    ///
    /// Zero values are normalized to positive zero before identity is derived.
    /// `columns` must be G128-aligned, with one input block per column group and
    /// one output scalar per row.
    ///
    /// # Errors
    /// Rejects unsupported curvature kinds, empty or oversized names, malformed
    /// geometry, missing evidence identity, non-finite/negative factors, zero
    /// effective row metrics, invalid PSD input blocks, or allocation overflow.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kind: SaltV2Curvature,
        source_id: CurvatureSourceId,
        upstream_evidence_digest: [u8; 32],
        tensor_index: u64,
        tensor_name: impl Into<String>,
        rows: usize,
        columns: usize,
        input_groups: Vec<DensePsdMetric>,
        output_weights: Vec<f64>,
        damping: f64,
    ) -> Result<Self, SaltV2KroneckerEvidenceError> {
        if !matches!(
            kind,
            SaltV2Curvature::InputHessian
                | SaltV2Curvature::GuidedFisher
                | SaltV2Curvature::ForwardKlKronecker
        ) {
            return Err(SaltV2KroneckerEvidenceError::Malformed("curvature kind"));
        }
        if upstream_evidence_digest == [0; 32] {
            return Err(SaltV2KroneckerEvidenceError::Malformed(
                "upstream evidence digest",
            ));
        }
        let tensor_name = tensor_name.into();
        if tensor_name.is_empty() || tensor_name.len() > MAX_NAME_BYTES {
            return Err(SaltV2KroneckerEvidenceError::Malformed("tensor name"));
        }
        let expected_groups = columns
            .checked_div(GROUP_SIZE)
            .filter(|_| rows > 0 && columns > 0 && columns.is_multiple_of(GROUP_SIZE));
        if expected_groups != Some(input_groups.len()) || output_weights.len() != rows {
            return Err(SaltV2KroneckerEvidenceError::Malformed("factor geometry"));
        }
        if !damping.is_finite() || damping < 0.0 {
            return Err(SaltV2KroneckerEvidenceError::Malformed("damping"));
        }
        let damping = canonical_zero(damping);

        let mut canonical_groups = Vec::new();
        canonical_groups
            .try_reserve_exact(input_groups.len())
            .map_err(|_| SaltV2KroneckerEvidenceError::AllocationFailed)?;
        for group in input_groups {
            if group.dimension() != GROUP_SIZE {
                return Err(SaltV2KroneckerEvidenceError::Malformed(
                    "input group dimension",
                ));
            }
            let values = group
                .as_slice()
                .iter()
                .map(|value| canonical_zero(*value))
                .collect::<Vec<_>>();
            canonical_groups.push(
                DensePsdMetric::new(GROUP_SIZE, &values)
                    .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("input group metric"))?,
            );
        }

        let mut canonical_outputs = Vec::new();
        canonical_outputs
            .try_reserve_exact(output_weights.len())
            .map_err(|_| SaltV2KroneckerEvidenceError::AllocationFailed)?;
        for output in output_weights {
            if !output.is_finite() || output < 0.0 || (output == 0.0 && damping == 0.0) {
                return Err(SaltV2KroneckerEvidenceError::Malformed("output curvature"));
            }
            canonical_outputs.push(canonical_zero(output));
        }

        let mut record = Self {
            kind,
            source_id,
            upstream_evidence_digest,
            tensor_index,
            tensor_name,
            rows,
            columns,
            input_groups: canonical_groups,
            output_weights: canonical_outputs,
            damping,
            record_digest: [0; 32],
        };
        let payload = record.encode_payload()?;
        record.record_digest = checksum(&payload);
        Ok(record)
    }

    /// Curvature algorithm represented by this record.
    #[must_use]
    pub const fn kind(&self) -> SaltV2Curvature {
        self.kind
    }

    /// Immutable source-model/cache/token-stream identity.
    #[must_use]
    pub const fn source_id(&self) -> CurvatureSourceId {
        self.source_id
    }

    /// Digest of the upstream accumulator or builder evidence.
    #[must_use]
    pub const fn upstream_evidence_digest(&self) -> [u8; 32] {
        self.upstream_evidence_digest
    }

    /// Global architecture-adapter tensor ordinal.
    #[must_use]
    pub const fn tensor_index(&self) -> u64 {
        self.tensor_index
    }

    /// Canonical source tensor name.
    #[must_use]
    pub fn tensor_name(&self) -> &str {
        &self.tensor_name
    }

    /// Matrix output rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Matrix input columns.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Shared input-side G128 PSD blocks.
    #[must_use]
    pub fn input_groups(&self) -> &[DensePsdMetric] {
        &self.input_groups
    }

    /// Output-side Fisher/KL scalars.
    #[must_use]
    pub fn output_weights(&self) -> &[f64] {
        &self.output_weights
    }

    /// Diagonal damping applied after Kronecker scaling.
    #[must_use]
    pub const fn damping(&self) -> f64 {
        self.damping
    }

    /// Digest of the complete canonical record payload.
    #[must_use]
    pub const fn record_digest(&self) -> [u8; 32] {
        self.record_digest
    }

    /// Exact canonical identity and encoded length without materializing bytes.
    ///
    /// # Errors
    /// Returns a checked-length failure if the validated geometry cannot be
    /// represented by the canonical record layout.
    pub fn receipt(&self) -> Result<SaltV2KroneckerEvidenceReceipt, SaltV2KroneckerEvidenceError> {
        let bytes = payload_len(
            self.tensor_name.len(),
            self.input_groups.len(),
            self.output_weights.len(),
        )
        .and_then(|bytes| bytes.checked_add(CHECKSUM_BYTES))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(SaltV2KroneckerEvidenceError::Malformed("encoded length"))?;
        Ok(SaltV2KroneckerEvidenceReceipt {
            record_digest: self.record_digest,
            bytes,
        })
    }

    /// Verify exact canonical bytes previously written for this record.
    ///
    /// Verification uses fixed memory: it hashes the expected payload length,
    /// checks both the computed and terminal checksum against this record's
    /// identity, and rejects truncation or trailing bytes without decoding or
    /// re-encoding a second record-sized buffer.
    ///
    /// # Errors
    /// Rejects I/O failure, truncation, trailing bytes, or any payload/checksum
    /// mismatch with this validated record.
    pub fn verify_written(
        &self,
        mut reader: impl Read,
    ) -> Result<SaltV2KroneckerEvidenceReceipt, SaltV2KroneckerEvidenceError> {
        const VERIFY_BUFFER_BYTES: usize = 8 * 1024;
        let receipt = self.receipt()?;
        let payload_bytes = receipt
            .bytes
            .checked_sub(CHECKSUM_BYTES as u64)
            .ok_or(SaltV2KroneckerEvidenceError::Malformed("encoded length"))?;
        let mut payload = (&mut reader).take(payload_bytes);
        let mut buffer = [0_u8; VERIFY_BUFFER_BYTES];
        let mut hasher = blake3::Hasher::new_derive_key(CHECKSUM_CONTEXT);
        let mut consumed = 0_u64;
        loop {
            let count = payload
                .read(&mut buffer)
                .map_err(|error| evidence_io("verify written evidence", error))?;
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
            consumed = consumed
                .checked_add(count as u64)
                .ok_or(SaltV2KroneckerEvidenceError::Malformed("encoded length"))?;
        }
        if consumed != payload_bytes {
            return Err(SaltV2KroneckerEvidenceError::Malformed(
                "truncated written record",
            ));
        }
        let mut terminal = [0_u8; CHECKSUM_BYTES];
        read_exact_evidence(&mut reader, &mut terminal)?;
        let computed = *hasher.finalize().as_bytes();
        if computed != self.record_digest || terminal != self.record_digest {
            return Err(SaltV2KroneckerEvidenceError::Malformed(
                "written record identity",
            ));
        }
        let mut trailing = [0_u8; 1];
        if reader
            .read(&mut trailing)
            .map_err(|error| evidence_io("verify written evidence", error))?
            != 0
        {
            return Err(SaltV2KroneckerEvidenceError::Malformed(
                "trailing written record",
            ));
        }
        Ok(receipt)
    }

    /// Reconstruct the borrowed fit-time curvature artifact.
    #[must_use]
    pub fn artifact(&self) -> CurvatureArtifact<'_> {
        let factors =
            crate::KroneckerCurvature::new(&self.input_groups, &self.output_weights, self.damping);
        match self.kind {
            SaltV2Curvature::InputHessian => CurvatureArtifact::input_hessian_kronecker(
                self.source_id,
                self.record_digest,
                factors,
            ),
            SaltV2Curvature::GuidedFisher => CurvatureArtifact::guided_fisher_kronecker(
                self.source_id,
                self.record_digest,
                factors,
            ),
            SaltV2Curvature::ForwardKlKronecker => CurvatureArtifact::forward_kl_kronecker_factors(
                self.source_id,
                self.record_digest,
                factors,
            ),
            SaltV2Curvature::DiagonalFisher => {
                unreachable!("constructor rejects diagonal Fisher")
            }
        }
    }

    /// Join this evidence to one caller-owned widened source matrix.
    ///
    /// # Errors
    /// Rejects a weight slice whose length differs from the record's exact
    /// matrix geometry.
    pub fn tensor_fit_input<'a>(
        &'a self,
        weights: &'a [f32],
    ) -> Result<SaltV2TensorFitInput<'a>, SaltV2KroneckerEvidenceError> {
        let expected = self
            .rows
            .checked_mul(self.columns)
            .ok_or(SaltV2KroneckerEvidenceError::Malformed("tensor geometry"))?;
        if weights.len() != expected {
            return Err(SaltV2KroneckerEvidenceError::WeightLengthMismatch {
                expected,
                got: weights.len(),
            });
        }
        Ok(SaltV2TensorFitInput {
            name: &self.tensor_name,
            weights,
            rows: self.rows,
            cols: self.columns,
            curvature: self.artifact(),
        })
    }

    /// Encode the exact canonical record and terminal checksum.
    ///
    /// # Errors
    /// Returns a checked length or allocation failure.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, SaltV2KroneckerEvidenceError> {
        let mut bytes = self.encode_payload()?;
        let digest = checksum(&bytes);
        if digest != self.record_digest {
            return Err(SaltV2KroneckerEvidenceError::Malformed("record identity"));
        }
        bytes
            .try_reserve_exact(CHECKSUM_BYTES)
            .map_err(|_| SaltV2KroneckerEvidenceError::AllocationFailed)?;
        bytes.extend_from_slice(&digest);
        Ok(bytes)
    }

    /// Decode and verify one complete canonical record.
    ///
    /// # Errors
    /// Rejects truncation, corruption, trailing/noncanonical bytes, invalid
    /// counts, geometry, factors, provenance, or allocation overflow.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, SaltV2KroneckerEvidenceError> {
        if bytes.len() < MAGIC.len() + 2 + 1 + 1 + CHECKSUM_BYTES {
            return Err(SaltV2KroneckerEvidenceError::Malformed("truncated record"));
        }
        let checksum_offset = bytes.len() - CHECKSUM_BYTES;
        let (payload, recorded_checksum) = bytes.split_at(checksum_offset);
        if checksum(payload).as_slice() != recorded_checksum {
            return Err(SaltV2KroneckerEvidenceError::Malformed("checksum"));
        }
        let mut cursor = Cursor::new(payload);
        if cursor.take(MAGIC.len())? != MAGIC {
            return Err(SaltV2KroneckerEvidenceError::Malformed("magic"));
        }
        if cursor.u16()? != VERSION || cursor.u8()? != 0 {
            return Err(SaltV2KroneckerEvidenceError::Malformed(
                "version or reserved byte",
            ));
        }
        let kind = kind_from_tag(cursor.u8()?)?;
        let tensor_index = cursor.u64()?;
        let rows = usize::try_from(cursor.u64()?)
            .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("rows"))?;
        let columns = usize::try_from(cursor.u64()?)
            .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("columns"))?;
        let name_len = usize::try_from(cursor.u32()?)
            .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("name length"))?;
        let group_count = usize::try_from(cursor.u32()?)
            .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("group count"))?;
        let output_count = usize::try_from(cursor.u64()?)
            .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("output count"))?;
        let damping = f64::from_bits(cursor.u64()?);
        let source_id =
            CurvatureSourceId::new(cursor.digest()?, cursor.digest()?, cursor.digest()?)
                .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("source identity"))?;
        let upstream_evidence_digest = cursor.digest()?;
        if name_len == 0 || name_len > MAX_NAME_BYTES {
            return Err(SaltV2KroneckerEvidenceError::Malformed("name length"));
        }
        let expected_groups = columns
            .checked_div(GROUP_SIZE)
            .filter(|_| rows > 0 && columns > 0 && columns.is_multiple_of(GROUP_SIZE));
        if expected_groups != Some(group_count) || output_count != rows {
            return Err(SaltV2KroneckerEvidenceError::Malformed("factor counts"));
        }
        let expected_payload_len = payload_len(name_len, group_count, output_count)
            .ok_or(SaltV2KroneckerEvidenceError::Malformed("encoded length"))?;
        if expected_payload_len != payload.len() {
            return Err(SaltV2KroneckerEvidenceError::Malformed("encoded length"));
        }
        let name = std::str::from_utf8(cursor.take(name_len)?)
            .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("tensor name utf8"))?;
        let mut tensor_name = String::new();
        tensor_name
            .try_reserve_exact(name_len)
            .map_err(|_| SaltV2KroneckerEvidenceError::AllocationFailed)?;
        tensor_name.push_str(name);

        let mut input_groups = Vec::new();
        input_groups
            .try_reserve_exact(group_count)
            .map_err(|_| SaltV2KroneckerEvidenceError::AllocationFailed)?;
        for _ in 0..group_count {
            let dimension = usize::try_from(cursor.u32()?)
                .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("group dimension"))?;
            if dimension != GROUP_SIZE {
                return Err(SaltV2KroneckerEvidenceError::Malformed("group dimension"));
            }
            let value_count = dimension
                .checked_mul(dimension)
                .ok_or(SaltV2KroneckerEvidenceError::Malformed("group size"))?;
            let mut values = Vec::new();
            values
                .try_reserve_exact(value_count)
                .map_err(|_| SaltV2KroneckerEvidenceError::AllocationFailed)?;
            for _ in 0..value_count {
                values.push(f64::from_bits(cursor.u64()?));
            }
            input_groups.push(
                DensePsdMetric::new(dimension, &values)
                    .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("input group metric"))?,
            );
        }
        let mut output_weights = Vec::new();
        output_weights
            .try_reserve_exact(output_count)
            .map_err(|_| SaltV2KroneckerEvidenceError::AllocationFailed)?;
        for _ in 0..output_count {
            output_weights.push(f64::from_bits(cursor.u64()?));
        }
        if cursor.remaining() != 0 {
            return Err(SaltV2KroneckerEvidenceError::Malformed("trailing bytes"));
        }
        let record = Self::new(
            kind,
            source_id,
            upstream_evidence_digest,
            tensor_index,
            tensor_name,
            rows,
            columns,
            input_groups,
            output_weights,
            damping,
        )?;
        if record.canonical_bytes()? != bytes {
            return Err(SaltV2KroneckerEvidenceError::Malformed(
                "noncanonical record",
            ));
        }
        Ok(record)
    }

    /// Read one record through a hard byte ceiling.
    ///
    /// # Errors
    /// Rejects a zero limit, I/O failure, input exceeding `max_bytes`, or any
    /// canonical decode failure.
    pub fn read_from(
        reader: impl Read,
        max_bytes: u64,
    ) -> Result<Self, SaltV2KroneckerEvidenceError> {
        if max_bytes == 0 {
            return Err(SaltV2KroneckerEvidenceError::SizeLimitExceeded { max_bytes });
        }
        let read_limit = max_bytes
            .checked_add(1)
            .ok_or(SaltV2KroneckerEvidenceError::SizeLimitExceeded { max_bytes })?;
        let mut bytes = Vec::new();
        let reserve = usize::try_from(max_bytes.min(16 * 1024 * 1024))
            .map_err(|_| SaltV2KroneckerEvidenceError::AllocationFailed)?;
        bytes
            .try_reserve(reserve)
            .map_err(|_| SaltV2KroneckerEvidenceError::AllocationFailed)?;
        reader
            .take(read_limit)
            .read_to_end(&mut bytes)
            .map_err(|error| evidence_io("read evidence", error))?;
        if bytes.len() as u64 > max_bytes {
            return Err(SaltV2KroneckerEvidenceError::SizeLimitExceeded { max_bytes });
        }
        Self::from_canonical_bytes(&bytes)
    }

    /// Write one canonical record and return its exact content receipt.
    ///
    /// # Errors
    /// Returns an encoding or output I/O failure.
    pub fn write_to(
        &self,
        mut writer: impl Write,
    ) -> Result<SaltV2KroneckerEvidenceReceipt, SaltV2KroneckerEvidenceError> {
        let bytes = self.canonical_bytes()?;
        writer
            .write_all(&bytes)
            .map_err(|error| evidence_io("write evidence", error))?;
        let receipt = self.receipt()?;
        debug_assert_eq!(receipt.bytes, bytes.len() as u64);
        Ok(receipt)
    }

    fn encode_payload(&self) -> Result<Vec<u8>, SaltV2KroneckerEvidenceError> {
        let name_len = u32::try_from(self.tensor_name.len())
            .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("name length"))?;
        let group_count = u32::try_from(self.input_groups.len())
            .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("group count"))?;
        let output_count = u64::try_from(self.output_weights.len())
            .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("output count"))?;
        let encoded_len = payload_len(
            self.tensor_name.len(),
            self.input_groups.len(),
            self.output_weights.len(),
        )
        .ok_or(SaltV2KroneckerEvidenceError::Malformed("encoded length"))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(encoded_len)
            .map_err(|_| SaltV2KroneckerEvidenceError::AllocationFailed)?;
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.push(0);
        bytes.push(kind_tag(self.kind));
        bytes.extend_from_slice(&self.tensor_index.to_le_bytes());
        bytes.extend_from_slice(
            &u64::try_from(self.rows)
                .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("rows"))?
                .to_le_bytes(),
        );
        bytes.extend_from_slice(
            &u64::try_from(self.columns)
                .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("columns"))?
                .to_le_bytes(),
        );
        bytes.extend_from_slice(&name_len.to_le_bytes());
        bytes.extend_from_slice(&group_count.to_le_bytes());
        bytes.extend_from_slice(&output_count.to_le_bytes());
        bytes.extend_from_slice(&self.damping.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.source_id.source_model_digest());
        bytes.extend_from_slice(&self.source_id.activation_cache_digest());
        bytes.extend_from_slice(&self.source_id.token_stream_digest());
        bytes.extend_from_slice(&self.upstream_evidence_digest);
        bytes.extend_from_slice(self.tensor_name.as_bytes());
        for group in &self.input_groups {
            bytes.extend_from_slice(
                &u32::try_from(group.dimension())
                    .map_err(|_| SaltV2KroneckerEvidenceError::Malformed("group dimension"))?
                    .to_le_bytes(),
            );
            for value in group.as_slice() {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        for output in &self.output_weights {
            bytes.extend_from_slice(&output.to_bits().to_le_bytes());
        }
        debug_assert_eq!(bytes.len(), encoded_len);
        Ok(bytes)
    }
}

/// Exact identity and length of one written evidence record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SaltV2KroneckerEvidenceReceipt {
    record_digest: [u8; 32],
    bytes: u64,
}

impl SaltV2KroneckerEvidenceReceipt {
    /// Canonical record payload digest.
    #[must_use]
    pub const fn record_digest(self) -> [u8; 32] {
        self.record_digest
    }

    /// Exact bytes written, including the terminal checksum.
    #[must_use]
    pub const fn bytes(self) -> u64 {
        self.bytes
    }
}

/// Failure while creating, reopening, or writing factorized curvature evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SaltV2KroneckerEvidenceError {
    /// A stable schema, identity, geometry, or numerical invariant failed.
    Malformed(&'static str),
    /// Caller-owned source weights did not match the record geometry.
    WeightLengthMismatch {
        /// Required number of row-major weights.
        expected: usize,
        /// Supplied number of weights.
        got: usize,
    },
    /// Bounded input exceeded the caller-authorized byte ceiling.
    SizeLimitExceeded {
        /// Maximum admitted bytes.
        max_bytes: u64,
    },
    /// A bounded allocation failed.
    AllocationFailed,
    /// Portable input/output failure.
    Io {
        /// Stable operation label.
        operation: &'static str,
        /// Portable I/O category.
        kind: io::ErrorKind,
    },
}

impl fmt::Display for SaltV2KroneckerEvidenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Malformed(field) => write!(formatter, "malformed Kronecker evidence: {field}"),
            Self::WeightLengthMismatch { expected, got } => write!(
                formatter,
                "Kronecker evidence needs {expected} source weights, received {got}"
            ),
            Self::SizeLimitExceeded { max_bytes } => write!(
                formatter,
                "Kronecker evidence exceeds the {max_bytes}-byte input limit"
            ),
            Self::AllocationFailed => formatter.write_str("Kronecker evidence allocation failed"),
            Self::Io { operation, kind } => {
                write!(formatter, "Kronecker evidence {operation} failed: {kind:?}")
            }
        }
    }
}

impl std::error::Error for SaltV2KroneckerEvidenceError {}

fn kind_tag(kind: SaltV2Curvature) -> u8 {
    match kind {
        SaltV2Curvature::InputHessian => 1,
        SaltV2Curvature::GuidedFisher => 2,
        SaltV2Curvature::ForwardKlKronecker => 3,
        SaltV2Curvature::DiagonalFisher => 0,
    }
}

fn kind_from_tag(tag: u8) -> Result<SaltV2Curvature, SaltV2KroneckerEvidenceError> {
    match tag {
        1 => Ok(SaltV2Curvature::InputHessian),
        2 => Ok(SaltV2Curvature::GuidedFisher),
        3 => Ok(SaltV2Curvature::ForwardKlKronecker),
        _ => Err(SaltV2KroneckerEvidenceError::Malformed("curvature kind")),
    }
}

fn checksum(payload: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key(CHECKSUM_CONTEXT);
    hasher.update(payload);
    *hasher.finalize().as_bytes()
}

fn payload_len(name_len: usize, group_count: usize, output_count: usize) -> Option<usize> {
    group_count
        .checked_mul(GROUP_PAYLOAD_BYTES)
        .and_then(|groups| FIXED_PAYLOAD_BYTES.checked_add(groups))
        .and_then(|length| length.checked_add(name_len))
        .and_then(|length| {
            output_count
                .checked_mul(size_of::<f64>())
                .and_then(|outputs| length.checked_add(outputs))
        })
}

fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

fn read_exact_evidence(
    reader: &mut impl Read,
    mut output: &mut [u8],
) -> Result<(), SaltV2KroneckerEvidenceError> {
    while !output.is_empty() {
        match reader.read(output) {
            Ok(0) => {
                return Err(SaltV2KroneckerEvidenceError::Malformed(
                    "truncated written record",
                ));
            }
            Ok(count) => output = &mut output[count..],
            Err(error) => return Err(evidence_io("verify written evidence", error)),
        }
    }
    Ok(())
}

fn evidence_io(operation: &'static str, error: io::Error) -> SaltV2KroneckerEvidenceError {
    SaltV2KroneckerEvidenceError::Io {
        operation,
        kind: error.kind(),
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], SaltV2KroneckerEvidenceError> {
        let end = self
            .offset
            .checked_add(count)
            .filter(|end| *end <= self.bytes.len())
            .ok_or(SaltV2KroneckerEvidenceError::Malformed("truncated field"))?;
        let value = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, SaltV2KroneckerEvidenceError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, SaltV2KroneckerEvidenceError> {
        let mut bytes = [0; 2];
        bytes.copy_from_slice(self.take(2)?);
        Ok(u16::from_le_bytes(bytes))
    }

    fn u32(&mut self) -> Result<u32, SaltV2KroneckerEvidenceError> {
        let mut bytes = [0; 4];
        bytes.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, SaltV2KroneckerEvidenceError> {
        let mut bytes = [0; 8];
        bytes.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(bytes))
    }

    fn digest(&mut self) -> Result<[u8; 32], SaltV2KroneckerEvidenceError> {
        let mut bytes = [0; 32];
        bytes.copy_from_slice(self.take(32)?);
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ActivationCache, ActivationCacheBuilder, ActivationCacheSpec, ActivationChunk,
        ActivationDType, ActivationDigest, PhysicalRateTarget, SaltV2Config, SaltV2Packing,
        SaltV2TensorMasterFitInput, fit_salt_v2_tensor_master,
    };
    use tritium_format::ModelId;

    fn source_id() -> CurvatureSourceId {
        CurvatureSourceId::new([1; 32], [2; 32], [3; 32]).unwrap()
    }

    fn identity_group() -> DensePsdMetric {
        let values = (0..GROUP_SIZE * GROUP_SIZE)
            .map(|index| {
                if index / GROUP_SIZE == index % GROUP_SIZE {
                    1.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        DensePsdMetric::new(GROUP_SIZE, &values).unwrap()
    }

    fn evidence_for(source_id: CurvatureSourceId) -> SaltV2KroneckerEvidence {
        SaltV2KroneckerEvidence::new(
            SaltV2Curvature::GuidedFisher,
            source_id,
            [4; 32],
            17,
            "model.layers.3.mlp.down_proj.weight",
            2,
            GROUP_SIZE,
            vec![identity_group()],
            vec![0.5, 1.5],
            0.125,
        )
        .unwrap()
    }

    fn evidence() -> SaltV2KroneckerEvidence {
        evidence_for(source_id())
    }

    fn activation_cache() -> ActivationCache {
        let spec = ActivationCacheSpec::new(
            0,
            "x",
            1,
            1,
            ActivationDType::Float32,
            ActivationDigest::from_bytes([3; 32]),
            1,
        )
        .unwrap();
        let mut builder = ActivationCacheBuilder::new(spec.clone());
        builder
            .ingest(ActivationChunk::new(&spec, 0, 1, vec![1.0], vec![true], vec![1]).unwrap())
            .unwrap();
        builder.finalize().unwrap()
    }

    #[test]
    fn canonical_record_round_trips_and_binds_every_factor() {
        let original = evidence();
        let bytes = original.canonical_bytes().unwrap();
        let reopened = SaltV2KroneckerEvidence::from_canonical_bytes(&bytes).unwrap();
        let receipt = original.receipt().unwrap();
        assert_eq!(receipt.record_digest(), original.record_digest());
        assert_eq!(receipt.bytes(), bytes.len() as u64);
        assert_eq!(reopened.canonical_bytes().unwrap(), bytes);
        assert_eq!(reopened.receipt().unwrap(), receipt);
        assert_eq!(reopened.record_digest(), original.record_digest());
        assert_eq!(reopened.tensor_index(), 17);
        assert_eq!(reopened.tensor_name(), original.tensor_name());
        assert_eq!(reopened.artifact().digest(), original.artifact().digest());

        let changed = SaltV2KroneckerEvidence::new(
            original.kind(),
            original.source_id(),
            original.upstream_evidence_digest(),
            original.tensor_index(),
            original.tensor_name(),
            original.rows(),
            original.columns(),
            original.input_groups().to_vec(),
            vec![0.5, 1.75],
            original.damping(),
        )
        .unwrap();
        assert_ne!(changed.record_digest(), original.record_digest());
        assert_ne!(changed.artifact().digest(), original.artifact().digest());
    }

    #[test]
    fn bounded_reader_and_corruption_fail_closed() {
        let record = evidence();
        let bytes = record.canonical_bytes().unwrap();
        assert_eq!(
            record.verify_written(bytes.as_slice()).unwrap(),
            record.receipt().unwrap()
        );
        assert!(matches!(
            SaltV2KroneckerEvidence::read_from(bytes.as_slice(), bytes.len() as u64 - 1),
            Err(SaltV2KroneckerEvidenceError::SizeLimitExceeded { .. })
        ));
        assert!(SaltV2KroneckerEvidence::read_from(bytes.as_slice(), bytes.len() as u64).is_ok());
        for index in [0, bytes.len() / 2, bytes.len() - 1] {
            let mut corrupt = bytes.clone();
            corrupt[index] ^= 1;
            assert!(SaltV2KroneckerEvidence::from_canonical_bytes(&corrupt).is_err());
            assert!(record.verify_written(corrupt.as_slice()).is_err());
        }
        for length in 0..bytes.len().min(256) {
            assert!(SaltV2KroneckerEvidence::from_canonical_bytes(&bytes[..length]).is_err());
            assert!(record.verify_written(&bytes[..length]).is_err());
        }
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(record.verify_written(trailing.as_slice()).is_err());

        let mut forged = bytes;
        forged[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
        forged[40..48].copy_from_slice(&u64::MAX.to_le_bytes());
        let checksum_offset = forged.len() - CHECKSUM_BYTES;
        let digest = checksum(&forged[..checksum_offset]);
        forged[checksum_offset..].copy_from_slice(&digest);
        assert!(matches!(
            SaltV2KroneckerEvidence::from_canonical_bytes(&forged),
            Err(SaltV2KroneckerEvidenceError::Malformed("encoded length"))
        ));
    }

    #[test]
    fn reopened_record_drives_the_same_tensor_master_bytes() {
        let cache = activation_cache();
        let source_id = CurvatureSourceId::new(
            [1; 32],
            cache.digest().into_bytes(),
            cache.spec().source_digest().into_bytes(),
        )
        .unwrap();
        let original = evidence_for(source_id);
        let bytes = original.canonical_bytes().unwrap();
        let reopened = SaltV2KroneckerEvidence::from_canonical_bytes(&bytes).unwrap();
        let weights = (0..2 * GROUP_SIZE)
            .map(|index| (index as f32 - 127.0) / 61.0)
            .collect::<Vec<_>>();
        let mut recipe = SaltV2Config {
            curvature: SaltV2Curvature::GuidedFisher,
            packing: SaltV2Packing::B3,
            rate: PhysicalRateTarget {
                max_matrix_bytes: 100_000,
                max_artifact_bytes: 100_000,
                max_resident_bytes: None,
            },
            ..SaltV2Config::default()
        };
        recipe.coordinate_sweeps = 2;
        recipe.em_restarts = 1;
        let fit = |evidence: &SaltV2KroneckerEvidence, sink: &mut Vec<u8>| {
            fit_salt_v2_tensor_master(
                SaltV2TensorMasterFitInput {
                    tensor: evidence.tensor_fit_input(&weights).unwrap(),
                    activations: &cache,
                    source_model_id: ModelId::from_digest([1; 32]),
                    tensor_index: evidence.tensor_index(),
                    source_tensor_digest: [5; 32],
                },
                &recipe,
                sink,
            )
            .unwrap();
        };
        let mut left = Vec::new();
        let mut right = Vec::new();
        fit(&original, &mut left);
        fit(&reopened, &mut right);
        assert_eq!(left, right);
    }

    #[test]
    fn writer_receipt_matches_canonical_record() {
        let evidence = evidence();
        let mut bytes = Vec::new();
        let receipt = evidence.write_to(&mut bytes).unwrap();
        assert_eq!(receipt.record_digest(), evidence.record_digest());
        assert_eq!(receipt.bytes(), bytes.len() as u64);
        assert_eq!(bytes, evidence.canonical_bytes().unwrap());
    }
}
