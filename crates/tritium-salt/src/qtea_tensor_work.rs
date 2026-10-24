//! Owner-bound native QTEA work, below machine/frontier admission or publication.
pub(crate) fn work_input_id(origin: QteaTensorOrigin<'_>, numerical: [u8; 32]) -> ContentId {
    let mut hash = blake3::Hasher::new_derive_key(CONTENT_ID_CONTEXT);
    hash.update(b"tritium qtea tensor work input v1\0");
    hash.update(origin.source_id.as_bytes());
    hash.update(&(origin.owner.len() as u64).to_le_bytes());
    hash.update(origin.owner.as_bytes());
    hash.update(&origin.tensor_index.to_le_bytes());
    hash.update(&origin.source_tensor_digest);
    hash.update(&numerical);
    ContentId::from_digest(*hash.finalize().as_bytes())
}

use std::{error::Error, fmt, io::Write, mem::size_of};
use tritium_format::qtea::{QteaByteLedger, QteaError, QteaLimits, QteaMetadata};
use tritium_quantize::qtea_reference::{ArithmeticError, TensorFitPlan};

use crate::{CONTENT_ID_CONTEXT, ContentId};

/// Declared original storage and canonical owner, not authenticated source custody.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QteaTensorOrigin<'a> {
    /// Campaign source artifact; distinct from the numerical source-model digest.
    pub source_id: ContentId,
    /// Canonical tensor owner, not an independently fitted alias.
    pub owner: &'a str,
    /// Original source tensor ordinal.
    pub tensor_index: u32,
    /// Digest of original stored tensor bytes, not widened numerical f32 values.
    pub source_tensor_digest: [u8; 32],
}

/// Conservative logical array/work scope for one native fit plus export.
///
/// Not stack/handle/allocator overhead, caller sink buffers, process/model/cache RSS, runtime staging,
/// measured RAM, machine admission or qualification. Encoded capacity uses the
/// explicit caller ceiling, not an assumption that allocation equals file length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QteaTensorWorkMemory {
    /// Borrowed original f32 weights and full f64 Gram values.
    pub borrowed_input_bytes: usize,
    /// Native bound, including retained fitted state and live fitting workspace.
    pub native_fit_bytes: usize,
    /// Explicit encoded Vec capacity ceiling retained during export overlap.
    pub encoded_capacity_ceiling: usize,
    /// Logical column flags used for encoded validation.
    pub validation_bytes: usize,
    /// Sum of the above, deliberately conservative rather than a measured peak.
    pub total_logical_bytes: usize,
}

/// Checked owner/input/recipe and fit/export work; not registry admission.
///
/// Owns the immutable numerical plan's borrows, with no new tensor scan/copy or
/// allocation during construction. A successful plan does not assert SPD or fit
/// completion. No installable solver identity or qualification is produced.
/// The owner name remains immutably borrowed through execution:
/// ```compile_fail,E0502
/// use tritium_salt::{ContentId, QteaTensorOrigin, QteaTensorWorkPlan};
/// use tritium_quantize::qtea_reference::TensorFitPlan;
/// use tritium_format::qtea::QteaLimits;
/// fn invalid(numerical: TensorFitPlan<'_>) {
///     let mut name = String::from("model.weight");
///     let work = QteaTensorWorkPlan::new(numerical, QteaTensorOrigin {
///         source_id: ContentId::of_bytes(b"source"), owner: &name,
///         tensor_index: 0, source_tensor_digest: [9; 32],
///     }, QteaLimits { max_tensor_bytes: 4096, max_validation_bytes: 4096 }, usize::MAX).unwrap();
///     name.clear();
///     let _ = work.fit_to(&mut Vec::new());
/// }
/// ```
pub struct QteaTensorWorkPlan<'a> {
    numerical: TensorFitPlan<'a>,
    origin: QteaTensorOrigin<'a>,
    limits: QteaLimits,
    metadata: QteaMetadata,
    ledger: QteaByteLedger,
    memory: QteaTensorWorkMemory,
    input_id: ContentId,
}

impl fmt::Debug for QteaTensorWorkPlan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QteaTensorWorkPlan")
            .field("origin", &self.origin)
            .field("input_id", &self.input_id)
            .field("memory", &self.memory)
            .finish_non_exhaustive()
    }
}

impl<'a> QteaTensorWorkPlan<'a> {
    /// Bind canonical owner/storage and price the complete logical fit/export scope.
    ///
    /// Operational caps are not part of mathematical work identity. The caller
    /// must separately budget other retained model/cache/process state and supply
    /// machine evidence before any future frontier admission.
    ///
    /// # Errors
    /// Refuses missing/invalid origin, codec limits, arithmetic overflow or an
    /// insufficient combined logical-byte ceiling, without fitting or sink access.
    pub fn new(
        numerical: TensorFitPlan<'a>,
        origin: QteaTensorOrigin<'a>,
        limits: QteaLimits,
        max_logical_bytes: usize,
    ) -> Result<Self, QteaTensorWorkError> {
        if origin.source_id.as_bytes() == &[0; 32]
            || origin.source_tensor_digest == [0; 32]
            || origin.owner.is_empty()
            || origin.owner.len() > 4096
            || origin.owner.chars().any(char::is_control)
        {
            return Err(QteaTensorWorkError::InvalidOrigin);
        }
        let ledger = numerical.encoded_ledger(limits)?;
        let metadata = numerical.metadata();
        let (r, c) = (metadata.rows, metadata.columns);
        let borrowed_input_bytes = r
            .checked_mul(c)
            .and_then(|n| n.checked_mul(size_of::<f32>()))
            .and_then(|weights| {
                c.checked_mul(c)
                    .and_then(|n| n.checked_mul(size_of::<f64>()))
                    .and_then(|gram| weights.checked_add(gram))
            })
            .ok_or(QteaTensorWorkError::SizeOverflow)?;
        let native_fit_bytes = numerical.required_work_bytes();
        let total_logical_bytes = borrowed_input_bytes
            .checked_add(native_fit_bytes)
            .and_then(|n| n.checked_add(limits.max_tensor_bytes))
            .and_then(|n| n.checked_add(c))
            .ok_or(QteaTensorWorkError::SizeOverflow)?;
        if total_logical_bytes > max_logical_bytes {
            return Err(QteaTensorWorkError::LogicalLimit {
                required: total_logical_bytes,
                limit: max_logical_bytes,
            });
        }
        let memory = QteaTensorWorkMemory {
            borrowed_input_bytes,
            native_fit_bytes,
            encoded_capacity_ceiling: limits.max_tensor_bytes,
            validation_bytes: c,
            total_logical_bytes,
        };
        let input_id = work_input_id(origin, numerical.input_digest());
        Ok(Self {
            numerical,
            origin,
            limits,
            metadata,
            ledger,
            memory,
            input_id,
        })
    }

    /// Owner/source assertions bound into the input ID, not verified custody.
    pub const fn origin(&self) -> QteaTensorOrigin<'a> {
        self.origin
    }
    /// Work identity, distinct from raw-storage, wire-trailer and output-file IDs.
    pub const fn input_id(&self) -> ContentId {
        self.input_id
    }
    /// Checked numerical metadata with actual weight/Gram/recipe fingerprints.
    pub const fn metadata(&self) -> QteaMetadata {
        self.metadata
    }
    /// Exact serialized byte ledger, computed by the format module.
    pub const fn byte_ledger(&self) -> QteaByteLedger {
        self.ledger
    }
    /// Complete declared logical array/work bound, not physical memory evidence.
    pub const fn memory(&self) -> QteaTensorWorkMemory {
        self.memory
    }

    /// Fit the exact inputs, check actual export and write unpublished tensor bytes.
    ///
    /// No writer call occurs before fitting/export/metadata checks. A partial sink
    /// failure can leave unpublished bytes but returns no successful work result.
    /// A successful generic Write is not durable publication or qualification.
    ///
    /// # Errors
    /// Propagates native numerical/allocator, codec, output-invariant and writer
    /// errors. There is no fallback, repair, adaptive retry or catalog activation.
    pub fn fit_to(
        self,
        sink: &mut impl Write,
    ) -> Result<QteaTensorWorkResult, QteaTensorWorkError> {
        let fit = self.numerical.fit()?;
        let encoded = fit.to_qtea_tensor(self.limits)?;
        if encoded.view().metadata() != self.metadata
            || encoded.view().recipe_digest() != self.metadata.recipe.digest()
            || encoded.view().byte_ledger() != self.ledger
        {
            return Err(QteaTensorWorkError::OutputMetadataMismatch);
        }
        drop(fit);
        let tensor_id = ContentId::of_bytes(encoded.as_bytes());
        let wire_digest = encoded.view().content_digest();
        encoded.write_to(sink)?;
        Ok(QteaTensorWorkResult {
            input_id: self.input_id,
            tensor_id,
            wire_digest,
            metadata: self.metadata,
            ledger: self.ledger,
        })
    }
}

/// Successful native tensor work/write observation, not an admitted stage receipt.
#[derive(Clone, Debug, PartialEq)]
pub struct QteaTensorWorkResult {
    input_id: ContentId,
    tensor_id: ContentId,
    wire_digest: [u8; 32],
    metadata: QteaMetadata,
    ledger: QteaByteLedger,
}

impl QteaTensorWorkResult {
    /// Exact owner/storage/numerical input work identity.
    pub const fn input_id(&self) -> ContentId {
        self.input_id
    }
    /// Generic ContentId of every actual serialized byte, including its trailer.
    pub const fn tensor_id(&self) -> ContentId {
        self.tensor_id
    }
    /// QTEA wire integrity trailer; not the complete-file ContentId.
    pub const fn wire_digest(&self) -> [u8; 32] {
        self.wire_digest
    }
    /// Actual export metadata checked against planned metadata before writing.
    pub const fn metadata(&self) -> QteaMetadata {
        self.metadata
    }
    /// Exact serialized ledger checked before writing.
    pub const fn byte_ledger(&self) -> QteaByteLedger {
        self.ledger
    }
}

/// Owner/cap/native/codec/writer refusal; never a successful work result.
#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub enum QteaTensorWorkError {
    /// Missing source/storage binding or invalid/oversized owner name.
    InvalidOrigin,
    /// Logical byte accounting overflowed the host index width.
    SizeOverflow,
    /// Complete logical fit/export scope exceeds the caller ceiling.
    LogicalLimit {
        /// Required conservative logical bytes.
        required: usize,
        /// Caller ceiling, not a measured physical allocation budget.
        limit: usize,
    },
    /// Native fitting failed before any writer access.
    Fit(ArithmeticError),
    /// Codec/export or sink error, potentially leaving partial unpublished bytes.
    Format(QteaError),
    /// Native export drifted from the checked metadata or exact byte ledger.
    OutputMetadataMismatch,
}

impl fmt::Display for QteaTensorWorkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QTEA tensor work refused: {self:?}")
    }
}
impl Error for QteaTensorWorkError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Fit(error) => Some(error),
            Self::Format(error) => Some(error),
            _ => None,
        }
    }
}
impl From<ArithmeticError> for QteaTensorWorkError {
    fn from(error: ArithmeticError) -> Self {
        Self::Fit(error)
    }
}
impl From<QteaError> for QteaTensorWorkError {
    fn from(error: QteaError) -> Self {
        Self::Format(error)
    }
}
