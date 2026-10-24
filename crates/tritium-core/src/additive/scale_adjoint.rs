use super::{
    AdditiveAdjointSpan, AdditiveError, AdditiveView, Allocation, BasisBound, Gather, Matmul,
    RowTiles, UniformPlan, apply_basis,
};
use crate::{PerTileIndex, Trit};
use tritium_schema::{AdditiveLayout, PlaneRelation, ScaleAnchor, ScalePrecision};

/// Checked scale-adjoint geometry shared by decoded and packed readers.
///
/// Owns no coefficients/scales and authenticates no payload. A reader must return
/// the admitted Trit at `(plane, absolute plane-major coefficient offset)`; the
/// offset includes all previous planes, including compact allocation. Only
/// present coefficients are visited, after full shape/index preflight. Reader
/// side effects/panics are not transactional. No allocation is performed here.
#[derive(Clone, Copy, Debug)]
pub struct AdditiveScaleVjpPlan<'a> {
    allocation: ScaleAllocation<'a>,
}

#[derive(Clone, Copy, Debug)]
enum ScaleAllocation<'a> {
    Uniform(UniformPlan),
    PerTile(PerTileIndex<'a>),
}

impl<'a> AdditiveScaleVjpPlan<'a> {
    /// Reuse checked uniform geometry without copying or revalidating payloads.
    #[must_use]
    pub const fn from_uniform(plan: UniformPlan) -> Self {
        Self {
            allocation: ScaleAllocation::Uniform(plan),
        }
    }

    /// Borrow the existing checked compact index; all-pruned geometry is valid.
    #[must_use]
    pub const fn from_per_tile(index: PerTileIndex<'a>) -> Self {
        Self {
            allocation: ScaleAllocation::PerTile(index),
        }
    }

    /// Number of actual stored scales, excluding derived tied-plane scales.
    #[must_use]
    pub const fn stored_scale_count(&self) -> usize {
        match self.allocation {
            ScaleAllocation::Uniform(plan) => plan.scale_count(),
            ScaleAllocation::PerTile(index) => index.scale_count(),
        }
    }

    /// Declared storage precision; this getter does not narrow any value.
    #[must_use]
    pub fn scale_precision(&self) -> ScalePrecision {
        self.layout().law.precision
    }

    /// Check scale length, finite nonnegative values and derived tied scales.
    ///
    /// Does not enforce exact storage precision or reject negative zero; those
    /// canonical storage rules belong to the format layer. No payload is read.
    pub fn validate_scales(&self, scales: &[f32]) -> Result<(), AdditiveError> {
        if scales.len() != self.stored_scale_count() {
            return Err(AdditiveError::ScaleCountMismatch {
                expected: self.stored_scale_count(),
                got: scales.len(),
            });
        }
        super::validate_scales(self.layout(), self.plane_count(), scales)
    }

    /// Accumulate a matmul scale VJP through an admitted coefficient reader.
    ///
    /// Uses the exact semantics, shape/preflight, accumulation order, caller
    /// buffers and finite-value policy of [`accumulate_ternary_matmul_scale_vjp`].
    pub fn matmul(
        &self,
        activations: &[f32],
        output_gradients: &[f32],
        batch: usize,
        row_workspace: &mut [f32],
        out: &mut [f64],
        read: impl FnMut(usize, usize) -> Trit,
    ) -> Result<(), AdditiveError> {
        self.matmul_span(
            activations,
            output_gradients,
            batch,
            row_workspace,
            AdditiveAdjointSpan::full(out),
            read,
        )
    }

    /// Accumulate a contiguous stored-scale span, preserving the full VJP's
    /// per-coordinate accumulation order and tied-plane factors. No allocation.
    /// Full input/span preflight precedes payload reads or buffer mutation.
    pub fn matmul_span(
        &self,
        activations: &[f32],
        output_gradients: &[f32],
        batch: usize,
        row_workspace: &mut [f32],
        output: AdditiveAdjointSpan<'_>,
        mut read: impl FnMut(usize, usize) -> Trit,
    ) -> Result<(), AdditiveError> {
        let rows = usize::try_from(self.layout().rows).map_err(|_| AdditiveError::SizeOverflow)?;
        let cols = usize::try_from(self.layout().cols).map_err(|_| AdditiveError::SizeOverflow)?;
        let inputs = batch.checked_mul(cols).ok_or(AdditiveError::SizeOverflow)?;
        let outputs = batch.checked_mul(rows).ok_or(AdditiveError::SizeOverflow)?;
        if activations.len() != inputs
            || output_gradients.len() != outputs
            || row_workspace.len() != cols
        {
            return Err(AdditiveError::MatmulShape);
        }
        let (span, out) = output.checked(self.stored_scale_count())?;
        let selected_rows = super::adjoint_span::rows_for_span(self, rows, &span, false);
        if selected_rows.is_empty() {
            return Ok(());
        }
        for (activation, dy) in activations
            .chunks_exact(cols)
            .zip(output_gradients.chunks_exact(rows))
        {
            row_workspace.copy_from_slice(activation);
            apply_basis(row_workspace, self.layout().basis)?;
            for row in selected_rows.clone() {
                accumulate_row(
                    self,
                    row,
                    row_workspace,
                    f64::from(dy[row]),
                    out,
                    &span,
                    &mut read,
                );
            }
        }
        Ok(())
    }

    /// Accumulate an embedding scale VJP through an admitted coefficient reader.
    ///
    /// Uses [`accumulate_embed_scale_vjp`]'s original-coordinate cotangents,
    /// duplicate-index order and preflight, without allocating or reading scales.
    pub fn embed(
        &self,
        ids: &[usize],
        output_gradients: &[f32],
        row_workspace: &mut [f32],
        out: &mut [f64],
        read: impl FnMut(usize, usize) -> Trit,
    ) -> Result<(), AdditiveError> {
        self.embed_span(
            ids,
            output_gradients,
            row_workspace,
            AdditiveAdjointSpan::full(out),
            read,
        )
    }

    /// Accumulate an embedding's contiguous stored-scale span, including tied
    /// planes and duplicate ids in occurrence order. No allocation. All shapes,
    /// indices and span bounds are checked before reads or mutation.
    pub fn embed_span(
        &self,
        ids: &[usize],
        output_gradients: &[f32],
        row_workspace: &mut [f32],
        output: AdditiveAdjointSpan<'_>,
        mut read: impl FnMut(usize, usize) -> Trit,
    ) -> Result<(), AdditiveError> {
        let rows = usize::try_from(self.layout().rows).map_err(|_| AdditiveError::SizeOverflow)?;
        let cols = usize::try_from(self.layout().cols).map_err(|_| AdditiveError::SizeOverflow)?;
        let outputs = ids
            .len()
            .checked_mul(cols)
            .ok_or(AdditiveError::SizeOverflow)?;
        if output_gradients.len() != outputs || row_workspace.len() != cols {
            return Err(AdditiveError::MatmulShape);
        }
        let (span, out) = output.checked(self.stored_scale_count())?;
        if ids.iter().any(|&row| row >= rows) {
            return Err(AdditiveError::RowOutOfRange);
        }
        let full = span.start == 0 && span.end == self.stored_scale_count();
        for (&row, dy) in ids.iter().zip(output_gradients.chunks_exact(cols)) {
            if !full && !super::adjoint_span::row_intersects(self, row, &span, false) {
                continue;
            }
            row_workspace.copy_from_slice(dy);
            apply_basis(row_workspace, self.layout().basis)?;
            accumulate_row(self, row, row_workspace, 1.0, out, &span, &mut read);
        }
        Ok(())
    }

    pub(super) fn layout(&self) -> AdditiveLayout {
        match self.allocation {
            ScaleAllocation::Uniform(plan) => plan.layout,
            ScaleAllocation::PerTile(index) => index.layout,
        }
    }
    pub(super) fn plane_count(&self) -> usize {
        match self.allocation {
            ScaleAllocation::Uniform(plan) => plan.plane_count(),
            ScaleAllocation::PerTile(index) => index.plane_count(),
        }
    }
    pub(super) fn row_tiles(&self, plane: usize, row: usize) -> RowTiles<'a> {
        match self.allocation {
            ScaleAllocation::Uniform(plan) => RowTiles::Uniform(plan.row_tiles(plane, row)),
            ScaleAllocation::PerTile(index) => RowTiles::PerTile(index.row_tiles(plane, row)),
        }
    }

    pub(super) fn present_weight_count(&self) -> usize {
        match self.allocation {
            ScaleAllocation::Uniform(plan) => plan.trit_count() / plan.plane_count(),
            ScaleAllocation::PerTile(index) if index.plane_count() != 0 => {
                index.plane_range(0).len()
            }
            ScaleAllocation::PerTile(_) => 0,
        }
    }
}

pub(super) fn for_view<'a>(view: &AdditiveView<'a>) -> AdditiveScaleVjpPlan<'a> {
    match view.allocation {
        Allocation::Uniform {
            layout,
            planes,
            scale_units,
        } => AdditiveScaleVjpPlan::from_uniform(UniformPlan {
            layout,
            planes,
            scale_units,
            trit_count: view.trits.len(),
            scale_count: view.scales.len(),
        }),
        Allocation::PerTile(index) => AdditiveScaleVjpPlan::from_per_tile(index),
    }
}

/// Accumulate the matmul's continuous **stored-scale** vector-Jacobian product.
///
/// For `y = A B x`, uses `B x`, not the inverse basis. Shared row/tile cursors
/// account for pruned coefficients and tied/base or tensor-wide scale ownership.
/// The tied-plane multiplier starts at one and multiplies the f32 ratio once per
/// plane. This ignores floating-point/storage rounding; it is not a code STE.
///
/// Inputs have `batch * cols` and `batch * rows` elements. Workspace has `cols`
/// f32 elements and `out` has one f64 value per actual stored scale. All lengths
/// and products are checked before mutation. Adds to existing `out` in batch,
/// row, plane, tile, column order; empty batch leaves both buffers untouched.
/// No heap or dense owner-gradient matrix is allocated. The view contains decoded
/// trits, not a packed backend handle. Finite-value/overflow policy is the caller's.
///
/// ```compile_fail
/// use tritium_core::{AdditiveView, accumulate_ternary_matmul_scale_vjp};
/// fn wrong(view: AdditiveView<'_>) {
///     accumulate_ternary_matmul_scale_vjp(&[], &view.bind_gather(), &[], 0,
///         &mut [], &mut []).unwrap();
/// }
/// ```
pub fn accumulate_ternary_matmul_scale_vjp(
    activations: &[f32],
    weights: &BasisBound<AdditiveView<'_>, Matmul>,
    output_gradients: &[f32],
    batch: usize,
    row_workspace: &mut [f32],
    out: &mut [f64],
) -> Result<(), AdditiveError> {
    let weights = &weights.tensor;
    for_view(weights).matmul(
        activations,
        output_gradients,
        batch,
        row_workspace,
        out,
        |_, index| weights.trits[index],
    )
}

/// Accumulate embedding/gather output cotangents into the actual stored scales.
///
/// Forward rows are `B^T A[row]`, so their stored-row cotangents are `B dy`.
/// Repeated indices accumulate in occurrence order. Uses the same continuous
/// scale law and caller-owned f64 accumulation as the matmul scale adjoint.
/// `output_gradients` has `ids.len() * cols` values, workspace has `cols`, and
/// `out` has the stored-scale length. All shapes/products/indices are checked
/// before mutation; empty ids leave both buffers untouched. No allocation or
/// dense gradient; finite/overflow policy belongs to the caller.
///
/// ```compile_fail
/// use tritium_core::{AdditiveView, accumulate_embed_scale_vjp};
/// fn wrong(view: AdditiveView<'_>) {
///     accumulate_embed_scale_vjp(&view.bind_matmul(), &[], &[],
///         &mut [], &mut []).unwrap();
/// }
/// ```
pub fn accumulate_embed_scale_vjp(
    weights: &BasisBound<AdditiveView<'_>, Gather>,
    ids: &[usize],
    output_gradients: &[f32],
    row_workspace: &mut [f32],
    out: &mut [f64],
) -> Result<(), AdditiveError> {
    let weights = &weights.tensor;
    for_view(weights).embed(ids, output_gradients, row_workspace, out, |_, index| {
        weights.trits[index]
    })
}

fn accumulate_row(
    weights: &AdditiveScaleVjpPlan<'_>,
    row: usize,
    values: &[f32],
    dy: f64,
    out: &mut [f64],
    span: &core::ops::Range<usize>,
    read: &mut impl FnMut(usize, usize) -> Trit,
) {
    let layout = weights.layout();
    let group = usize::from(layout.group);
    let mut plane_factor = 1.0f32;
    for plane in 0..weights.plane_count() {
        for tile in weights.row_tiles(plane, row) {
            let scale_end = tile.scale
                + if layout.law.anchor == ScaleAnchor::Group {
                    tile.len.div_ceil(group)
                } else {
                    1
                };
            if tile.scale >= span.end || scale_end <= span.start {
                continue;
            }
            for offset in 0..tile.len {
                let scale = tile.scale
                    + if layout.law.anchor == ScaleAnchor::Group {
                        offset / group
                    } else {
                        0
                    };
                if !span.contains(&scale) {
                    continue;
                }
                let trit = read(plane, tile.trit + offset);
                if !trit.is_zero() {
                    out[scale - span.start] += f64::from(values[tile.col + offset])
                        * f64::from(trit.get())
                        * dy
                        * f64::from(plane_factor);
                }
            }
        }
        if let PlaneRelation::Tied { num, den } = layout.law.relation {
            plane_factor *= num as f32 / den as f32;
        }
    }
}
