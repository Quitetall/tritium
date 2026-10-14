//! Allocation-free geometry for backend payload/workspace planning.
use crate::{BackendError, TensorCaps, TensorView};
use tritium_core::{AdditiveView, PerTilePlan, UniformPlan};
use tritium_schema::AdditiveLayout;

#[derive(Clone, Copy, Debug)]
enum Allocation<'a> {
    Uniform(UniformPlan),
    PerTile(PerTilePlan<'a>),
    View(AdditiveView<'a>),
}

/// Sealed additive geometry, without allocating decoded coefficients or indices.
/// Metadata preparation does not validate actual scales, trit symbols or S34 zeros.
#[derive(Clone, Copy, Debug)]
pub struct AdditiveMetadata<'a> {
    layout: AdditiveLayout,
    allocation: Allocation<'a>,
    count_bytes: Option<usize>,
}
fn invalid(e: impl core::fmt::Debug) -> BackendError {
    BackendError::InvalidInput(format!("tensor metadata: {e:?}"))
}
impl<'a> AdditiveMetadata<'a> {
    /// Validate a Uniform layout/count without allocating payload storage.
    pub fn uniform(layout: AdditiveLayout, planes: u8) -> Result<Self, BackendError> {
        Ok(Self {
            layout,
            allocation: Allocation::Uniform(UniformPlan::new(layout, planes).map_err(invalid)?),
            count_bytes: None,
        })
    }
    /// Validate borrowed exact PerTile counts without copying them or building an index.
    pub fn per_tile(layout: AdditiveLayout, counts: &'a [u8]) -> Result<Self, BackendError> {
        Ok(Self {
            layout,
            allocation: Allocation::PerTile(PerTilePlan::new(layout, counts).map_err(invalid)?),
            count_bytes: Some(counts.len()),
        })
    }
    /// Borrow an already sealed decoded view without rescanning/rebuilding its index.
    pub fn from_view(view: AdditiveView<'a>) -> Self {
        Self {
            layout: view.layout(),
            allocation: Allocation::View(view),
            count_bytes: view.tile_counts().map(<[u8]>::len),
        }
    }
    /// Exact declared law, basis, dimensions and coding.
    pub const fn layout(self) -> AdditiveLayout {
        self.layout
    }
    /// Actual present-plane maximum, including zero for fully pruned PerTile.
    pub const fn plane_count(self) -> usize {
        match self.allocation {
            Allocation::Uniform(p) => p.plane_count(),
            Allocation::PerTile(p) => p.plane_count(),
            Allocation::View(v) => v.plane_count(),
        }
    }
    /// Exact stored scalar count, not derived tied-plane values.
    pub fn scale_count(self) -> usize {
        match self.allocation {
            Allocation::Uniform(p) => p.scale_count(),
            Allocation::PerTile(p) => p.scale_count(),
            Allocation::View(v) => v.scales().len(),
        }
    }
    /// Exact per-plane coefficient length; metadata-only PerTile scans borrowed counts.
    pub fn plane_trit_count(self, plane: usize) -> Result<usize, BackendError> {
        match self.allocation {
            Allocation::Uniform(p) if plane < p.plane_count() => {
                Ok(p.trit_count() / p.plane_count())
            }
            Allocation::Uniform(_) => Err(invalid("absent plane")),
            Allocation::PerTile(p) => p.plane_trit_count(plane).map_err(invalid),
            Allocation::View(v) => Ok(v.plane_trits(plane).map_err(invalid)?.len()),
        }
    }
    /// Logical owned count/index buffers used by the current core index, not capacity/RSS.
    pub fn auxiliary_bytes(self) -> Result<usize, BackendError> {
        match self.allocation {
            Allocation::Uniform(_) => Ok(0),
            Allocation::PerTile(p) => p.buffer_len_bytes().map_err(invalid),
            Allocation::View(v) => v.auxiliary_buffer_len_bytes().map_err(invalid),
        }
    }
    /// Exact number of borrowed per-tile count bytes, or None for Uniform.
    pub fn count_bytes(self) -> Option<usize> {
        self.count_bytes
    }
}

/// Structural upload input without a decoded matrix. Not numerical validation or support.
#[derive(Clone, Copy, Debug)]
// Keep sealed borrowed geometry on the stack; boxing would allocate during
// metadata preflight and lose Copy. No coefficient/index buffers are copied.
#[allow(clippy::large_enum_variant)]
pub enum TensorMetadata<'a> {
    /// Additive law/basis/count geometry.
    Additive(AdditiveMetadata<'a>),
    /// Dense f32 upload dimensions (source widening remains the caller's responsibility).
    Dense {
        /// Number of output rows.
        rows: usize,
        /// Number of input columns.
        cols: usize,
    },
}
impl<'a> TensorMetadata<'a> {
    /// Borrow structural metadata from an existing decoded view.
    pub fn from_view(view: TensorView<'a>) -> Result<Self, BackendError> {
        Ok(match view {
            TensorView::Additive(v) => Self::Additive(AdditiveMetadata::from_view(v)),
            TensorView::Dense { rows, cols, .. } => {
                view.decoded_payload_bytes()?;
                Self::Dense { rows, cols }
            }
        })
    }
    /// Validate dense dimensions and return f32 payload bytes without allocation.
    pub fn dense_bytes(rows: usize, cols: usize) -> Result<usize, BackendError> {
        rows.checked_mul(cols)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| invalid("dense size overflow"))
    }
}

/// Backend-declared logical retained payload and explicit upload workspace.
/// Numerical checks remain required at actual upload. This excludes allocator,
/// driver/pinned staging, source/decode buffers and runtime/KV/device execution
/// scratch; it is not aggregate physical-memory or execution qualification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorUploadPlan {
    /// Exact expected execution tier and owned payload declaration.
    pub caps: TensorCaps,
    /// Maximum additional explicit host upload workspace, excluding retained payload.
    pub host_workspace_bytes: u64,
    /// Maximum additional explicit device upload workspace, excluding retained payload.
    pub device_workspace_bytes: u64,
}
