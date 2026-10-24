//! Versioned residual-bearing QTEA tensors. Not SALT or a qualification receipt.
//!
//! Immutable checked bytes retain radix3 trits, exact f32 affine/column parameters,
//! packed1:4 FP8 residuals and provenance. No dense weight expansion is required.
//! Caller limits are logical/owned byte bounds, not allocator overhead or RSS.

mod layout;
mod tensor;
mod write;

pub use tensor::{QteaResidualView, QteaTensor, QteaTensorView};

/// Fixed V1 file magic.
pub const QTEA_MAGIC: [u8; 8] = *b"TQTEA1\0\0";
/// Only admitted wire version.
pub const QTEA_VERSION: u32 = 1;
/// Fixed header, including reserved zeros.
pub const QTEA_HEADER_BYTES: usize = 384;
/// Exact native numerical recipe being stored, not a frontier solver ID.
pub const QTEA_NUMERICAL_ID: &str = "qtea.cpu-tensor.f32.v1";

/// Original native numerical recipe. Hashing alone does not admit its values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QteaRecipe {
    /// Original group and feedback-block width.
    pub group_size: usize,
    /// Salient fraction, rounded to whole groups/ties-to-even.
    pub salient_ratio: f64,
    /// Initial dead-zone coefficient.
    pub delta_coefficient: f32,
    /// Original-group fit rounds.
    pub fit_iterations: u32,
    /// Column refinement rounds.
    pub rescale_iterations: u32,
    /// Declared relative damping; no retry or fallback.
    pub damping: f32,
    /// Exponential block feedback decay.
    pub decay: f64,
    /// Relative pivot admission floor, not a condition-number estimate.
    pub relative_pivot_floor: f32,
}

impl QteaRecipe {
    /// Same bit-preserving numerical-recipe digest as the native fitter.
    /// This pure fingerprint is not a validity, execution or custody claim.
    pub fn digest(self) -> [u8; 32] {
        let mut h = blake3::Hasher::new_derive_key("tritium qtea tensor recipe v1");
        h.update(QTEA_NUMERICAL_ID.as_bytes());
        h.update(&(self.group_size as u64).to_le_bytes());
        h.update(&self.salient_ratio.to_bits().to_le_bytes());
        h.update(&self.delta_coefficient.to_bits().to_le_bytes());
        h.update(&self.fit_iterations.to_le_bytes());
        h.update(&self.rescale_iterations.to_le_bytes());
        h.update(&self.damping.to_bits().to_le_bytes());
        h.update(&self.decay.to_bits().to_le_bytes());
        h.update(&self.relative_pivot_floor.to_bits().to_le_bytes());
        *h.finalize().as_bytes()
    }
}

/// Actual input fingerprints supplied by the producer, not independent attestation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QteaBindings {
    /// Original f32 weight bits/geometry identity.
    pub weights: [u8; 32],
    /// Canonical full input Gram identity.
    pub gram: [u8; 32],
    /// Source checkpoint identity.
    pub source_model: [u8; 32],
    /// Captured activation-cache identity.
    pub activation_cache: [u8; 32],
    /// Ordered calibration/token-stream identity.
    pub token_stream: [u8; 32],
}

/// Geometry, exact recipe and input bindings for one residual-bearing tensor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QteaMetadata {
    /// Original output rows.
    pub rows: usize,
    /// Original input columns.
    pub columns: usize,
    /// Exact native fitting policy.
    pub recipe: QteaRecipe,
    /// Producer-supplied actual input identities.
    pub bindings: QteaBindings,
}

/// Mandatory receiver/constructor budgets. No unbounded default is supplied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QteaLimits {
    /// Maximum serialized bytes and owned Vec capacity; excludes fixed handle/RSS.
    pub max_tensor_bytes: usize,
    /// Maximum logical validation flag-array bytes (one per column).
    pub max_validation_bytes: usize,
}

/// Borrowed original group parameters; all slices must have exactly `rows` values.
#[derive(Clone, Copy, Debug)]
pub struct QteaGroupRef<'a> {
    /// Positive finite magnitudes.
    pub alpha: &'a [f32],
    /// Finite affine centers.
    pub beta: &'a [f32],
}

/// Borrowed original-column sparse correction, not a dense side tensor.
#[derive(Clone, Copy, Debug)]
pub struct QteaResidualRef<'a> {
    /// Original input column.
    pub column: usize,
    /// Positive finite multiplier used before E4M3FN encoding.
    pub scale_multiplier: f32,
    /// One0..3 position per four rows.
    pub indices: &'a [u8],
    /// One finite E4M3FN byte per four rows.
    pub codes: &'a [u8],
}

/// Exact serialized components, not empirical physical-memory/compression evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct QteaByteLedger {
    /// Fixed wire header.
    pub header_bytes: usize,
    /// Canonical five-trits-per-byte payload, including terminal digits.
    pub trit_bytes: usize,
    /// Exact executable alpha/beta f32 values; fitting-only delta is excluded.
    pub group_parameter_bytes: usize,
    /// Original column-v f32 values.
    pub column_multiplier_bytes: usize,
    /// Original-index fitting permutation.
    pub order_bytes: usize,
    /// Column/scale pairs of salient records.
    pub residual_metadata_bytes: usize,
    /// Packed two-bit positions, including terminal bits.
    pub residual_position_bytes: usize,
    /// Actual finite FP8 code bytes.
    pub residual_code_bytes: usize,
    /// Domain-separated integrity trailer.
    pub checksum_bytes: usize,
    /// Every serialized byte, exactly.
    pub total_bytes: usize,
}

/// Fail-closed wire/constructor refusal; no success receipt is generated.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum QteaError {
    /// Unsupported magic/version/header, reserved bits or length disagreement.
    InvalidHeader,
    /// Unsupported/overflowing geometry.
    InvalidGeometry,
    /// Invalid numerical recipe, identity or recipe digest.
    InvalidMetadata,
    /// Invalid canonical codes, parameters, order, residuals or integrity digest.
    InvalidBody,
    /// Caller tensor/capacity limit exceeded.
    TensorLimit,
    /// Column flag-array limit exceeded before allocation/body reading.
    ValidationLimit,
    /// Fallible allocation failed.
    AllocationFailed,
    /// Public accessor index was outside admitted geometry.
    InvalidIndex,
    /// Reader/writer failed; not successful publication.
    Io(std::io::ErrorKind),
}

impl core::fmt::Display for QteaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "QTEA tensor refused: {self:?}")
    }
}
impl std::error::Error for QteaError {}
impl From<std::io::Error> for QteaError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error.kind())
    }
}

fn allocate(len: usize) -> Result<Vec<u8>, QteaError> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .map_err(|_| QteaError::AllocationFailed)?;
    bytes.resize(len, 0);
    Ok(bytes)
}
fn f32_at(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("admitted section"),
    )
}
fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("admitted section"),
    )
}
fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("admitted header"),
    )
}
fn fingerprint(bytes: &[u8]) -> [u8; 32] {
    let mut h = blake3::Hasher::new_derive_key("tritium qtea tensor wire v1");
    h.update(bytes);
    *h.finalize().as_bytes()
}
fn fp8(code: u8) -> f32 {
    let mag = code & 127;
    let e = i32::from(mag >> 3);
    let m = f32::from(mag & 7);
    let value = if e == 0 {
        m / 512.
    } else {
        (8. + m) * 2.0_f32.powi(e - 10)
    };
    if code & 128 == 0 { value } else { -value }
}
