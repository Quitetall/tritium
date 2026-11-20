//! Explicit encoded source protocol; no implicit full-decoding adapter.
use crate::{BackendError, TensorMetadata, TensorUploadPlan};
use tritium_schema::ScalePrecision;

/// Maximum borrowed encoded window; implementations must not retain visitor bytes.
pub const TENSOR_STREAM_CHUNK_BYTES: usize = 64 * 1024;

/// Source coding, independent of the receiving backend's physical layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TensorStreamEncoding {
    /// Canonical raw additive planes followed by exact little-endian stored scales.
    /// Geometry, codec, law and per-tile counts are bound by sealed metadata.
    CanonicalRawAdditive,
    /// Dense row-major little-endian f16/bf16/f32 scalar bytes.
    DenseLittleEndian(ScalePrecision),
}

/// Structural encoded-source plan. Values/canonical padding/source identity still
/// require validation during upload. This is not a physical-memory qualification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorStreamPlan {
    /// Backend retained payload and explicit additional upload workspaces.
    pub upload: TensorUploadPlan,
    /// Exact expected encoded source byte count; no compressed-size substitution.
    pub encoded_bytes: u64,
}

/// Repeatable synchronous encoded source, with externally bound immutable metadata.
/// Visit emits sequential windows <=TENSOR_STREAM_CHUNK_BYTES and must fail on
/// its own digest/I/O/length faults. No hidden complete encoded/decoded master is
/// permitted by this interface. Sources requiring mutation may use interior
/// mutability while keeping metadata borrowed and immutable. Backend effects must
/// remain private until successful visit and terminal enclosing source checks.
pub trait TensorStreamSource {
    /// Sealed geometry; borrowed counts must not change while the source exists.
    fn metadata(&self) -> TensorMetadata<'_>;
    /// Actual externally bound source encoding, not the receiving native format.
    fn encoding(&self) -> TensorStreamEncoding;
    /// Exact encoded length, checked against the backend plan before source I/O.
    fn encoded_bytes(&self) -> u64;
    /// Visit borrowed sequential bytes. Empty source emits no windows.
    fn visit(
        &self,
        visitor: &mut dyn FnMut(&[u8]) -> Result<(), BackendError>,
    ) -> Result<(), BackendError>;
}
