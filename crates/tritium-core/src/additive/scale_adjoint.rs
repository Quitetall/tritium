use super::{AdditiveError, AdditiveView, BasisBound, Gather, Matmul, apply_basis};
use tritium_schema::{PlaneRelation, ScaleAnchor};

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
    let rows = usize::try_from(weights.layout().rows).map_err(|_| AdditiveError::SizeOverflow)?;
    let cols = usize::try_from(weights.layout().cols).map_err(|_| AdditiveError::SizeOverflow)?;
    let inputs = batch.checked_mul(cols).ok_or(AdditiveError::SizeOverflow)?;
    let outputs = batch.checked_mul(rows).ok_or(AdditiveError::SizeOverflow)?;
    if activations.len() != inputs
        || output_gradients.len() != outputs
        || row_workspace.len() != cols
        || out.len() != weights.scales().len()
    {
        return Err(AdditiveError::MatmulShape);
    }
    for (activation, dy) in activations
        .chunks_exact(cols)
        .zip(output_gradients.chunks_exact(rows))
    {
        row_workspace.copy_from_slice(activation);
        apply_basis(row_workspace, weights.layout().basis)?;
        for (row, &gradient) in dy.iter().enumerate() {
            accumulate_row(weights, row, row_workspace, f64::from(gradient), out);
        }
    }
    Ok(())
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
    let rows = usize::try_from(weights.layout().rows).map_err(|_| AdditiveError::SizeOverflow)?;
    let cols = usize::try_from(weights.layout().cols).map_err(|_| AdditiveError::SizeOverflow)?;
    let outputs = ids
        .len()
        .checked_mul(cols)
        .ok_or(AdditiveError::SizeOverflow)?;
    if output_gradients.len() != outputs
        || row_workspace.len() != cols
        || out.len() != weights.scales().len()
    {
        return Err(AdditiveError::MatmulShape);
    }
    if ids.iter().any(|&row| row >= rows) {
        return Err(AdditiveError::RowOutOfRange);
    }
    for (&row, dy) in ids.iter().zip(output_gradients.chunks_exact(cols)) {
        row_workspace.copy_from_slice(dy);
        apply_basis(row_workspace, weights.layout().basis)?;
        accumulate_row(weights, row, row_workspace, 1.0, out);
    }
    Ok(())
}

fn accumulate_row(
    weights: &AdditiveView<'_>,
    row: usize,
    values: &[f32],
    dy: f64,
    out: &mut [f64],
) {
    let layout = weights.layout();
    let group = usize::from(layout.group);
    let mut plane_factor = 1.0f32;
    for plane in 0..weights.plane_count() {
        for tile in weights.row_tiles(plane, row) {
            for offset in 0..tile.len {
                let trit = weights.trits[tile.trit + offset];
                if !trit.is_zero() {
                    let scale = tile.scale
                        + if layout.law.anchor == ScaleAnchor::Group {
                            offset / group
                        } else {
                            0
                        };
                    out[scale] += f64::from(values[tile.col + offset])
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
