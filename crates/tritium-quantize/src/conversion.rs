//! Resumable control-plane state for a full-model additive-PTQ conversion.
//!
//! The state machine makes stage order and failure semantics explicit. Its
//! deterministic binary checkpoint records only control state and stage-output
//! digests; bulk tensors remain in separately content-addressed artifacts.

use core::fmt;
use tritium_format::PackageId;

/// Conversion-state checkpoint magic (`TQCS`).
pub const CONVERSION_STATE_MAGIC: [u8; 4] = *b"TQCS";

/// Current conversion-state checkpoint version.
pub const CONVERSION_STATE_VERSION: u8 = 1;

const NO_STAGE: u8 = u8::MAX;
const CHECKSUM_BYTES: usize = 32;

/// Ordered stages of the additive-PTQ conversion pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ConversionStage {
    /// Discover and fingerprint source configuration and tensors.
    Ingest,
    /// Build the deterministic calibration sample and activation cache.
    Calibrate,
    /// Measure sensitivity, outliers, and baseline quality.
    Profile,
    /// Search quantization recipes and rate allocations.
    Search,
    /// Run optional reconstruction or teacher-guided refinement.
    Refine,
    /// Encode the chosen progressive ternary package.
    Pack,
    /// Run integrity, fidelity, task, and runtime gates.
    Validate,
    /// Atomically publish immutable artifacts and provenance.
    Publish,
}

impl ConversionStage {
    /// Canonical pipeline order.
    pub const ALL: [Self; 8] = [
        Self::Ingest,
        Self::Calibrate,
        Self::Profile,
        Self::Search,
        Self::Refine,
        Self::Pack,
        Self::Validate,
        Self::Publish,
    ];

    fn code(self) -> u8 {
        match self {
            Self::Ingest => 0,
            Self::Calibrate => 1,
            Self::Profile => 2,
            Self::Search => 3,
            Self::Refine => 4,
            Self::Pack => 5,
            Self::Validate => 6,
            Self::Publish => 7,
        }
    }

    fn from_code(code: u8) -> Result<Self, ConversionError> {
        Self::ALL
            .get(code as usize)
            .copied()
            .ok_or(ConversionError::InvalidStage(code))
    }

    fn next(self) -> Option<Self> {
        Self::ALL.get(self.code() as usize + 1).copied()
    }
}

/// Lifecycle state of a conversion run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RunStatus {
    /// The current stage is ready to begin.
    Ready,
    /// A worker owns an active attempt of the current stage.
    Running,
    /// The last attempt failed and may be retried.
    RetryableFailure,
    /// The last attempt failed and requires operator or recipe changes.
    TerminalFailure,
    /// Every stage, including publish, completed.
    Succeeded,
}

impl RunStatus {
    fn code(self) -> u8 {
        match self {
            Self::Ready => 0,
            Self::Running => 1,
            Self::RetryableFailure => 2,
            Self::TerminalFailure => 3,
            Self::Succeeded => 4,
        }
    }

    fn from_code(code: u8) -> Result<Self, ConversionError> {
        match code {
            0 => Ok(Self::Ready),
            1 => Ok(Self::Running),
            2 => Ok(Self::RetryableFailure),
            3 => Ok(Self::TerminalFailure),
            4 => Ok(Self::Succeeded),
            _ => Err(ConversionError::InvalidStatus(code)),
        }
    }
}

/// A newly claimed attempt of the current conversion stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageAttempt {
    stage: ConversionStage,
    number: u32,
}

impl StageAttempt {
    /// Stage claimed by this attempt.
    pub fn stage(&self) -> ConversionStage {
        self.stage
    }

    /// One-based attempt number for this stage.
    pub fn number(&self) -> u32 {
        self.number
    }
}

/// Immutable receipt for one completed conversion stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageReceipt {
    stage: ConversionStage,
    attempts: u32,
    output_id: [u8; 32],
}

impl StageReceipt {
    /// Completed stage.
    pub fn stage(&self) -> ConversionStage {
        self.stage
    }

    /// Number of attempts needed to complete the stage.
    pub fn attempts(&self) -> u32 {
        self.attempts
    }

    /// Digest of the immutable stage-output artifact.
    pub fn output_id(&self) -> &[u8; 32] {
        &self.output_id
    }
}

/// Structured failure of a conversion-stage attempt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageFailure {
    stage: ConversionStage,
    retryable: bool,
    code: String,
    message: String,
}

impl StageFailure {
    /// Failed stage.
    pub fn stage(&self) -> ConversionStage {
        self.stage
    }

    /// Whether the same recipe may retry this stage.
    pub fn retryable(&self) -> bool {
        self.retryable
    }

    /// Stable machine-readable failure code.
    pub fn code(&self) -> &str {
        &self.code
    }

    /// Human-readable diagnostic, which must not be used for control flow.
    pub fn message(&self) -> &str {
        &self.message
    }
}

/// Persisted, resumable state of one immutable conversion recipe.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConversionRun {
    recipe_id: [u8; 32],
    status: RunStatus,
    current: Option<ConversionStage>,
    attempts: u32,
    receipts: Vec<StageReceipt>,
    failure: Option<StageFailure>,
}

impl ConversionRun {
    /// Start a run at [`ConversionStage::Ingest`].
    pub fn new(recipe_id: [u8; 32]) -> Self {
        Self {
            recipe_id,
            status: RunStatus::Ready,
            current: Some(ConversionStage::Ingest),
            attempts: 0,
            receipts: Vec::new(),
            failure: None,
        }
    }

    /// Digest of the immutable conversion recipe and source selection.
    pub fn recipe_id(&self) -> &[u8; 32] {
        &self.recipe_id
    }

    /// Current run lifecycle state.
    pub fn status(&self) -> RunStatus {
        self.status
    }

    /// Stage awaiting work, running, or failed; `None` only after success.
    pub fn current_stage(&self) -> Option<ConversionStage> {
        self.current
    }

    /// Attempts already started for the current stage.
    pub fn current_attempts(&self) -> u32 {
        self.attempts
    }

    /// Ordered immutable receipts for completed stages.
    pub fn receipts(&self) -> &[StageReceipt] {
        &self.receipts
    }

    /// Most recent failure, present only in a failure status.
    pub fn failure(&self) -> Option<&StageFailure> {
        self.failure.as_ref()
    }

    /// Claim the current ready or retryable stage and increment its attempt.
    ///
    /// # Errors
    /// Returns [`ConversionError::InvalidTransition`] when already running,
    /// terminally failed, or complete, and [`ConversionError::AttemptOverflow`]
    /// if the attempt counter is exhausted.
    pub fn begin_stage(&mut self) -> Result<StageAttempt, ConversionError> {
        if !matches!(self.status, RunStatus::Ready | RunStatus::RetryableFailure) {
            return Err(ConversionError::InvalidTransition {
                status: self.status,
                operation: "begin_stage",
            });
        }
        let stage = self.current.ok_or(ConversionError::InvalidState(
            "runnable run has no current stage",
        ))?;
        self.attempts = self
            .attempts
            .checked_add(1)
            .ok_or(ConversionError::AttemptOverflow)?;
        self.status = RunStatus::Running;
        self.failure = None;
        Ok(StageAttempt {
            stage,
            number: self.attempts,
        })
    }

    /// Complete the running stage and advance to the next stage.
    ///
    /// `output_id` identifies the stage's immutable checkpoint artifact.
    ///
    /// # Errors
    /// Returns [`ConversionError::InvalidTransition`] unless a stage is running.
    pub fn complete_stage(&mut self, output_id: [u8; 32]) -> Result<(), ConversionError> {
        if self.status != RunStatus::Running {
            return Err(ConversionError::InvalidTransition {
                status: self.status,
                operation: "complete_stage",
            });
        }
        let stage = self.current.ok_or(ConversionError::InvalidState(
            "running run has no current stage",
        ))?;
        self.receipts.push(StageReceipt {
            stage,
            attempts: self.attempts,
            output_id,
        });
        self.failure = None;
        self.attempts = 0;
        self.current = stage.next();
        self.status = if self.current.is_some() {
            RunStatus::Ready
        } else {
            RunStatus::Succeeded
        };
        Ok(())
    }

    /// Record failure of the running stage.
    ///
    /// A retryable failure may be claimed again with [`Self::begin_stage`]. A
    /// terminal failure cannot transition without constructing a new recipe/run.
    ///
    /// # Errors
    /// Returns [`ConversionError`] unless a stage is running, when `code` is
    /// empty, or when either string exceeds the checkpoint format's u32 length.
    pub fn fail_stage(
        &mut self,
        code: impl Into<String>,
        message: impl Into<String>,
        retryable: bool,
    ) -> Result<(), ConversionError> {
        if self.status != RunStatus::Running {
            return Err(ConversionError::InvalidTransition {
                status: self.status,
                operation: "fail_stage",
            });
        }
        let code = code.into();
        let message = message.into();
        if code.is_empty() {
            return Err(ConversionError::EmptyFailureCode);
        }
        if code.len() > u32::MAX as usize || message.len() > u32::MAX as usize {
            return Err(ConversionError::FieldTooLong);
        }
        let stage = self.current.ok_or(ConversionError::InvalidState(
            "running run has no current stage",
        ))?;
        self.status = if retryable {
            RunStatus::RetryableFailure
        } else {
            RunStatus::TerminalFailure
        };
        self.failure = Some(StageFailure {
            stage,
            retryable,
            code,
            message,
        });
        Ok(())
    }

    /// Convert a persisted running attempt whose worker lease was lost into a
    /// retryable failure of the same stage.
    ///
    /// # Errors
    /// Has the same error conditions as [`Self::fail_stage`].
    pub fn recover_interrupted(
        &mut self,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Result<(), ConversionError> {
        self.fail_stage(code, message, true)
    }

    /// Serialize deterministic conversion control state.
    ///
    /// # Errors
    /// Returns [`ConversionError::FieldTooLong`] if a failure string does not fit
    /// the version-1 u32 length fields.
    pub fn to_bytes(&self) -> Result<Vec<u8>, ConversionError> {
        self.validate()?;
        let mut out = Vec::new();
        out.extend_from_slice(&CONVERSION_STATE_MAGIC);
        out.push(CONVERSION_STATE_VERSION);
        out.extend_from_slice(&self.recipe_id);
        out.push(self.status.code());
        out.push(self.current.map_or(NO_STAGE, ConversionStage::code));
        out.extend_from_slice(&self.attempts.to_le_bytes());
        out.extend_from_slice(&(self.receipts.len() as u32).to_le_bytes());
        for receipt in &self.receipts {
            out.push(receipt.stage.code());
            out.extend_from_slice(&receipt.attempts.to_le_bytes());
            out.extend_from_slice(&receipt.output_id);
        }
        match &self.failure {
            None => out.push(0),
            Some(failure) => {
                out.push(1);
                out.push(failure.stage.code());
                out.push(u8::from(failure.retryable));
                write_string(&mut out, &failure.code)?;
                write_string(&mut out, &failure.message)?;
            }
        }
        let checksum = PackageId::from_package_bytes(&out);
        out.extend_from_slice(checksum.as_bytes());
        Ok(out)
    }

    /// Restore and fully validate deterministic conversion control state.
    ///
    /// # Errors
    /// Returns [`ConversionError`] for malformed, unsupported, trailing, or
    /// internally inconsistent bytes. Parsing never allocates from an unbounded
    /// declared count.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ConversionError> {
        let payload_len = bytes
            .len()
            .checked_sub(CHECKSUM_BYTES)
            .ok_or(ConversionError::Truncated)?;
        let (payload, checksum) = bytes.split_at(payload_len);
        if PackageId::from_package_bytes(payload).as_bytes() != checksum {
            return Err(ConversionError::ChecksumMismatch);
        }
        let mut cursor = Cursor::new(payload);
        if cursor.take(4)? != CONVERSION_STATE_MAGIC {
            return Err(ConversionError::BadMagic);
        }
        let version = cursor.u8()?;
        if version != CONVERSION_STATE_VERSION {
            return Err(ConversionError::UnsupportedVersion(version));
        }
        let recipe_id = cursor.digest()?;
        let status = RunStatus::from_code(cursor.u8()?)?;
        let current = match cursor.u8()? {
            NO_STAGE => None,
            code => Some(ConversionStage::from_code(code)?),
        };
        let attempts = cursor.u32()?;
        let receipt_count = cursor.u32()? as usize;
        const RECEIPT_BYTES: usize = 1 + 4 + 32;
        if receipt_count > cursor.remaining() / RECEIPT_BYTES {
            return Err(ConversionError::Truncated);
        }
        let mut receipts = Vec::new();
        for _ in 0..receipt_count {
            receipts.push(StageReceipt {
                stage: ConversionStage::from_code(cursor.u8()?)?,
                attempts: cursor.u32()?,
                output_id: cursor.digest()?,
            });
        }
        let failure = match cursor.u8()? {
            0 => None,
            1 => {
                let stage = ConversionStage::from_code(cursor.u8()?)?;
                let retryable = match cursor.u8()? {
                    0 => false,
                    1 => true,
                    _ => return Err(ConversionError::InvalidState("invalid retryable flag")),
                };
                Some(StageFailure {
                    stage,
                    retryable,
                    code: cursor.string()?,
                    message: cursor.string()?,
                })
            }
            _ => {
                return Err(ConversionError::InvalidState(
                    "invalid failure-present flag",
                ));
            }
        };
        if cursor.remaining() != 0 {
            return Err(ConversionError::TrailingBytes(cursor.remaining()));
        }
        let run = Self {
            recipe_id,
            status,
            current,
            attempts,
            receipts,
            failure,
        };
        run.validate()?;
        Ok(run)
    }

    fn validate(&self) -> Result<(), ConversionError> {
        if self.receipts.len() > ConversionStage::ALL.len() {
            return Err(ConversionError::InvalidState("too many stage receipts"));
        }
        for (index, receipt) in self.receipts.iter().enumerate() {
            if receipt.stage != ConversionStage::ALL[index] {
                return Err(ConversionError::InvalidState(
                    "stage receipts are not a canonical prefix",
                ));
            }
            if receipt.attempts == 0 {
                return Err(ConversionError::InvalidState(
                    "completed stage has zero attempts",
                ));
            }
        }
        let expected_current = ConversionStage::ALL.get(self.receipts.len()).copied();
        match self.status {
            RunStatus::Ready => {
                if self.current != expected_current || self.attempts != 0 || self.failure.is_some()
                {
                    return Err(ConversionError::InvalidState("invalid ready state"));
                }
            }
            RunStatus::Running => {
                if self.current != expected_current || self.attempts == 0 || self.failure.is_some()
                {
                    return Err(ConversionError::InvalidState("invalid running state"));
                }
            }
            RunStatus::RetryableFailure | RunStatus::TerminalFailure => {
                let failure = self.failure.as_ref().ok_or(ConversionError::InvalidState(
                    "failure status lacks failure",
                ))?;
                let retryable = self.status == RunStatus::RetryableFailure;
                if self.current != expected_current
                    || self.attempts == 0
                    || failure.stage
                        != self.current.ok_or(ConversionError::InvalidState(
                            "failure state has no current stage",
                        ))?
                    || failure.retryable != retryable
                    || failure.code.is_empty()
                {
                    return Err(ConversionError::InvalidState("invalid failure state"));
                }
            }
            RunStatus::Succeeded => {
                if self.receipts.len() != ConversionStage::ALL.len()
                    || self.current.is_some()
                    || self.attempts != 0
                    || self.failure.is_some()
                {
                    return Err(ConversionError::InvalidState("invalid succeeded state"));
                }
            }
        }
        Ok(())
    }
}

fn write_string(out: &mut Vec<u8>, value: &str) -> Result<(), ConversionError> {
    let len = u32::try_from(value.len()).map_err(|_| ConversionError::FieldTooLong)?;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

/// Why a conversion transition or state checkpoint was rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConversionError {
    /// Operation is not legal in the current lifecycle state.
    InvalidTransition {
        /// Current lifecycle state.
        status: RunStatus,
        /// Attempted operation.
        operation: &'static str,
    },
    /// The current stage's u32 attempt counter overflowed.
    AttemptOverflow,
    /// Failure code was empty.
    EmptyFailureCode,
    /// A persisted string exceeded its u32 length field.
    FieldTooLong,
    /// State checkpoint magic did not match [`CONVERSION_STATE_MAGIC`].
    BadMagic,
    /// State checkpoint version is not supported.
    UnsupportedVersion(u8),
    /// State checkpoint ended before a declared field completed.
    Truncated,
    /// The exact-byte checkpoint checksum did not match its payload.
    ChecksumMismatch,
    /// Bytes remained after the checkpoint was decoded.
    TrailingBytes(usize),
    /// A stage discriminant was not recognized.
    InvalidStage(u8),
    /// A run-status discriminant was not recognized.
    InvalidStatus(u8),
    /// A persisted string was not valid UTF-8.
    InvalidUtf8,
    /// Decoded fields violated state-machine invariants.
    InvalidState(&'static str),
}

impl fmt::Display for ConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTransition { status, operation } => {
                write!(f, "cannot {operation} while conversion is {status:?}")
            }
            Self::AttemptOverflow => f.write_str("conversion attempt counter overflowed"),
            Self::EmptyFailureCode => f.write_str("conversion failure code is empty"),
            Self::FieldTooLong => f.write_str("conversion-state field exceeds u32 capacity"),
            Self::BadMagic => f.write_str("conversion state has bad magic"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported conversion-state version {version}")
            }
            Self::Truncated => f.write_str("conversion state is truncated"),
            Self::ChecksumMismatch => f.write_str("conversion-state checksum mismatch"),
            Self::TrailingBytes(count) => {
                write!(f, "conversion state has {count} trailing bytes")
            }
            Self::InvalidStage(stage) => write!(f, "invalid conversion stage {stage}"),
            Self::InvalidStatus(status) => write!(f, "invalid conversion status {status}"),
            Self::InvalidUtf8 => f.write_str("conversion state contains invalid UTF-8"),
            Self::InvalidState(reason) => write!(f, "invalid conversion state: {reason}"),
        }
    }
}

impl std::error::Error for ConversionError {}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], ConversionError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(ConversionError::Truncated)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(ConversionError::Truncated)?;
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, ConversionError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, ConversionError> {
        let bytes: [u8; 4] = self
            .take(4)?
            .try_into()
            .map_err(|_| ConversionError::Truncated)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn digest(&mut self) -> Result<[u8; 32], ConversionError> {
        self.take(32)?
            .try_into()
            .map_err(|_| ConversionError::Truncated)
    }

    fn string(&mut self) -> Result<String, ConversionError> {
        let len = self.u32()? as usize;
        let bytes = self.take(len)?;
        core::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| ConversionError::InvalidUtf8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RECIPE_ID: [u8; 32] = [0x42; 32];

    #[test]
    fn complete_run_is_ordered_resumable_and_deterministic() {
        let mut run = ConversionRun::new(RECIPE_ID);
        for (index, expected) in ConversionStage::ALL.into_iter().enumerate() {
            assert_eq!(run.current_stage(), Some(expected));
            let attempt = run.begin_stage().expect("begin stage");
            assert_eq!(attempt.stage(), expected);
            assert_eq!(attempt.number(), 1);
            run.complete_stage([index as u8; 32])
                .expect("complete stage");
        }

        assert_eq!(run.status(), RunStatus::Succeeded);
        assert_eq!(run.current_stage(), None);
        assert_eq!(run.receipts().len(), ConversionStage::ALL.len());
        let bytes = run.to_bytes().expect("serialize");
        assert_eq!(bytes, run.to_bytes().expect("deterministic serialize"));
        assert_eq!(ConversionRun::from_bytes(&bytes).expect("resume"), run);
    }

    #[test]
    fn retryable_failure_retries_same_stage_but_terminal_failure_stops() {
        let mut run = ConversionRun::new(RECIPE_ID);
        run.begin_stage().expect("attempt one");
        run.fail_stage("source_timeout", "remote read timed out", true)
            .expect("retryable failure");
        assert_eq!(run.status(), RunStatus::RetryableFailure);
        assert_eq!(run.current_stage(), Some(ConversionStage::Ingest));

        let retry = run.begin_stage().expect("retry");
        assert_eq!(retry.number(), 2);
        run.fail_stage("bad_checkpoint", "source digest mismatch", false)
            .expect("terminal failure");
        assert_eq!(run.status(), RunStatus::TerminalFailure);
        assert!(run.begin_stage().is_err());
    }

    #[test]
    fn interrupted_attempt_is_explicitly_recovered_after_reload() {
        let mut run = ConversionRun::new(RECIPE_ID);
        run.begin_stage().expect("begin");
        let bytes = run.to_bytes().expect("serialize running state");
        let mut resumed = ConversionRun::from_bytes(&bytes).expect("reload");

        resumed
            .recover_interrupted("worker_lost", "worker lease expired")
            .expect("recover");
        assert_eq!(resumed.status(), RunStatus::RetryableFailure);
        assert_eq!(resumed.begin_stage().expect("retry").number(), 2);
    }

    #[test]
    fn invalid_transitions_and_corrupt_state_are_rejected() {
        let mut run = ConversionRun::new(RECIPE_ID);
        assert!(run.complete_stage([0; 32]).is_err());
        assert!(run.fail_stage("nope", "not running", true).is_err());

        let bytes = run.to_bytes().expect("serialize");
        let mut bit_flip = bytes.clone();
        bit_flip[10] ^= 1;
        assert!(matches!(
            ConversionRun::from_bytes(&bit_flip),
            Err(ConversionError::ChecksumMismatch)
        ));
        assert!(ConversionRun::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        let mut trailing = bytes;
        trailing.push(0);
        assert!(ConversionRun::from_bytes(&trailing).is_err());
    }
}
