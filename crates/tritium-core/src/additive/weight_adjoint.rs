use super::{
    AdditiveAdjointSpan, AdditiveError, AdditiveScaleVjpPlan, AdditiveView, BasisBound, Gather,
    Matmul, UniformPlan, apply_basis, scale_adjoint,
};
use crate::PerTileIndex;

/// Allocation-free continuous adjoints of the present logical stored weights.
///
/// Output is in base-plane coefficient order, including zero trits but excluding
/// pruned tiles. This differentiates `A` in `y = A B x`, not individual trits,
/// scales, planes or the original-basis weights. Payload values are not read.
/// Geometry is shared with scale VJPs; no graph or payload is authenticated.
/// Like scale VJPs, this continuous derivative ignores floating-point rounding.
#[derive(Clone, Copy, Debug)]
pub struct AdditiveWeightVjpPlan<'a> {
    geometry: AdditiveScaleVjpPlan<'a>,
}

impl<'a> From<AdditiveScaleVjpPlan<'a>> for AdditiveWeightVjpPlan<'a> {
    fn from(geometry: AdditiveScaleVjpPlan<'a>) -> Self {
        Self { geometry }
    }
}

impl<'a> AdditiveWeightVjpPlan<'a> {
    /// Reuse checked uniform geometry without reading coefficients or scales.
    #[must_use]
    pub const fn from_uniform(plan: UniformPlan) -> Self {
        Self {
            geometry: AdditiveScaleVjpPlan::from_uniform(plan),
        }
    }

    /// Borrow a checked compact index, including all-pruned geometry.
    #[must_use]
    pub const fn from_per_tile(index: PerTileIndex<'a>) -> Self {
        Self {
            geometry: AdditiveScaleVjpPlan::from_per_tile(index),
        }
    }

    /// Number of present base-plane coordinates, not all plane coefficients.
    #[must_use]
    pub fn present_weight_count(&self) -> usize {
        self.geometry.present_weight_count()
    }

    /// Add `dy * (B x)` into present stored-weight cotangents.
    ///
    /// Inputs have batch*columns and batch*rows values; workspace is exactly
    /// columns f32 and output is present_weight_count f64. Preflight precedes
    /// mutation. Accumulation order is batch, row, tile, column. Empty uses leave
    /// both buffers untouched. No allocation; finite/overflow policy is caller's.
    pub fn matmul(
        &self,
        activations: &[f32],
        output_gradients: &[f32],
        batch: usize,
        row_workspace: &mut [f32],
        out: &mut [f64],
    ) -> Result<(), AdditiveError> {
        self.matmul_span(
            activations,
            output_gradients,
            batch,
            row_workspace,
            AdditiveAdjointSpan::full(out),
        )
    }

    /// Accumulate a contiguous present-weight output span with the full VJP's
    /// per-coordinate order and preflight. No allocation or dense gradient.
    /// Invalid spans refuse before mutation; empty uses leave buffers untouched.
    pub fn matmul_span(
        &self,
        activations: &[f32],
        output_gradients: &[f32],
        batch: usize,
        row_workspace: &mut [f32],
        output: AdditiveAdjointSpan<'_>,
    ) -> Result<(), AdditiveError> {
        let layout = self.geometry.layout();
        let rows = usize::try_from(layout.rows).map_err(|_| AdditiveError::SizeOverflow)?;
        let cols = usize::try_from(layout.cols).map_err(|_| AdditiveError::SizeOverflow)?;
        let inputs = batch.checked_mul(cols).ok_or(AdditiveError::SizeOverflow)?;
        let outputs = batch.checked_mul(rows).ok_or(AdditiveError::SizeOverflow)?;
        if activations.len() != inputs
            || output_gradients.len() != outputs
            || row_workspace.len() != cols
        {
            return Err(AdditiveError::MatmulShape);
        }
        let (span, out) = output.checked(self.present_weight_count())?;
        if self.geometry.plane_count() == 0 {
            return Ok(());
        }
        let selected_rows = super::adjoint_span::rows_for_span(&self.geometry, rows, &span, true);
        if selected_rows.is_empty() {
            return Ok(());
        }
        for (x, dy) in activations
            .chunks_exact(cols)
            .zip(output_gradients.chunks_exact(rows))
        {
            row_workspace.copy_from_slice(x);
            apply_basis(row_workspace, layout.basis)?;
            for row in selected_rows.clone() {
                self.accumulate_row(row, row_workspace, f64::from(dy[row]), &span, out);
            }
        }
        Ok(())
    }

    /// Add embedding cotangents `B dy` in index occurrence order.
    ///
    /// Repeated indices add repeatedly. Cotangents have ids.len()*columns values,
    /// workspace exactly columns and output present_weight_count. All indices
    /// and shapes are checked before mutation. Empty uses leave buffers untouched.
    /// No allocation; finite/overflow policy belongs to the caller.
    pub fn embed(
        &self,
        ids: &[usize],
        output_gradients: &[f32],
        row_workspace: &mut [f32],
        out: &mut [f64],
    ) -> Result<(), AdditiveError> {
        self.embed_span(
            ids,
            output_gradients,
            row_workspace,
            AdditiveAdjointSpan::full(out),
        )
    }

    /// Accumulate an embedding's contiguous present-weight output span.
    /// Duplicate-index order and forward basis match the full VJP. All indices
    /// and span bounds are checked before mutation, even for an empty span.
    pub fn embed_span(
        &self,
        ids: &[usize],
        output_gradients: &[f32],
        row_workspace: &mut [f32],
        output: AdditiveAdjointSpan<'_>,
    ) -> Result<(), AdditiveError> {
        let layout = self.geometry.layout();
        let rows = usize::try_from(layout.rows).map_err(|_| AdditiveError::SizeOverflow)?;
        let cols = usize::try_from(layout.cols).map_err(|_| AdditiveError::SizeOverflow)?;
        let outputs = ids
            .len()
            .checked_mul(cols)
            .ok_or(AdditiveError::SizeOverflow)?;
        if output_gradients.len() != outputs || row_workspace.len() != cols {
            return Err(AdditiveError::MatmulShape);
        }
        let (span, out) = output.checked(self.present_weight_count())?;
        if ids.iter().any(|&row| row >= rows) {
            return Err(AdditiveError::RowOutOfRange);
        }
        if self.geometry.plane_count() == 0 {
            return Ok(());
        }
        let full = span.start == 0 && span.end == self.present_weight_count();
        for (&row, dy) in ids.iter().zip(output_gradients.chunks_exact(cols)) {
            if !full && !super::adjoint_span::row_intersects(&self.geometry, row, &span, true) {
                continue;
            }
            row_workspace.copy_from_slice(dy);
            apply_basis(row_workspace, layout.basis)?;
            self.accumulate_row(row, row_workspace, 1.0, &span, out);
        }
        Ok(())
    }

    fn accumulate_row(
        &self,
        row: usize,
        values: &[f32],
        dy: f64,
        span: &core::ops::Range<usize>,
        out: &mut [f64],
    ) {
        for tile in self.geometry.row_tiles(0, row) {
            for index in tile.trit.max(span.start)..(tile.trit + tile.len).min(span.end) {
                out[index - span.start] += f64::from(values[tile.col + index - tile.trit]) * dy;
            }
        }
    }
}

/// Accumulate matmul's continuous present stored-weight VJP without allocation.
///
/// See [`AdditiveWeightVjpPlan::matmul`]. Requires the matmul consumer binding;
/// does not differentiate the ternary code or allocate a dense gradient master.
///
/// ```compile_fail
/// use tritium_core::{AdditiveView, accumulate_ternary_matmul_weight_vjp};
/// fn wrong(view: AdditiveView<'_>) {
///     accumulate_ternary_matmul_weight_vjp(&[], &view.bind_gather(), &[], 0,
///         &mut [], &mut []).unwrap();
/// }
/// ```
pub fn accumulate_ternary_matmul_weight_vjp(
    activations: &[f32],
    weights: &BasisBound<AdditiveView<'_>, Matmul>,
    output_gradients: &[f32],
    batch: usize,
    row_workspace: &mut [f32],
    out: &mut [f64],
) -> Result<(), AdditiveError> {
    AdditiveWeightVjpPlan::from(scale_adjoint::for_view(&weights.tensor)).matmul(
        activations,
        output_gradients,
        batch,
        row_workspace,
        out,
    )
}

/// Accumulate embedding's continuous present stored-weight VJP without allocation.
///
/// See [`AdditiveWeightVjpPlan::embed`]. Requires the gather consumer binding;
/// signed RHT applies forward `B` to cotangents, not inverse `B^T`.
///
/// ```compile_fail
/// use tritium_core::{AdditiveView, accumulate_embed_weight_vjp};
/// fn wrong(view: AdditiveView<'_>) {
///     accumulate_embed_weight_vjp(&view.bind_matmul(), &[], &[],
///         &mut [], &mut []).unwrap();
/// }
/// ```
pub fn accumulate_embed_weight_vjp(
    weights: &BasisBound<AdditiveView<'_>, Gather>,
    ids: &[usize],
    output_gradients: &[f32],
    row_workspace: &mut [f32],
    out: &mut [f64],
) -> Result<(), AdditiveError> {
    AdditiveWeightVjpPlan::from(scale_adjoint::for_view(&weights.tensor)).embed(
        ids,
        output_gradients,
        row_workspace,
        out,
    )
}
