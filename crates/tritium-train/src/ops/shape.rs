//! Column slice + concat — the reshape primitives for multi-head / GQA attention (plan 0040).
//!
//! Per-head attention splits a `[seq, n_head·head_dim]` projection into per-head `[seq, head_dim]`
//! slices, runs scaled-dot-product-attention per head, then concatenates the head outputs back.
//! `slice_cols` + `concat_cols` are exactly those two reshapes, made differentiable.

/// `Y[r, c] = X[r, start + c]` for `c in 0..len` — extract a contiguous column range from a
/// `[rows, cols]` row-major matrix. `Y` is `[rows, len]`. Panics if `start + len > cols`.
pub fn slice_cols_forward(
    x: &[f32],
    rows: usize,
    cols: usize,
    start: usize,
    len: usize,
) -> Vec<f32> {
    assert!(
        start + len <= cols,
        "slice [{start}, {start}+{len}) exceeds {cols} cols"
    );
    let mut y = vec![0.0f32; rows * len];
    for r in 0..rows {
        y[r * len..r * len + len].copy_from_slice(&x[r * cols + start..r * cols + start + len]);
    }
    y
}

/// vjp of [`slice_cols_forward`]: scatter `grad_out [rows, len]` into the `[rows, cols]` column
/// range (zero elsewhere). Returns the single input's gradient.
pub fn slice_cols_vjp(
    rows: usize,
    cols: usize,
    start: usize,
    len: usize,
    grad_out: &[f32],
) -> Vec<f32> {
    let mut gx = vec![0.0f32; rows * cols];
    for r in 0..rows {
        gx[r * cols + start..r * cols + start + len]
            .copy_from_slice(&grad_out[r * len..r * len + len]);
    }
    gx
}

/// Concatenate `parts` (each `[rows, lens[i]]`, row-major) along columns → `[rows, Σ lens]`.
/// Panics if `parts.len() != lens.len()`.
pub fn concat_cols_forward(parts: &[&[f32]], rows: usize, lens: &[usize]) -> Vec<f32> {
    assert_eq!(parts.len(), lens.len(), "one length per part");
    let total: usize = lens.iter().sum();
    let mut y = vec![0.0f32; rows * total];
    for r in 0..rows {
        let mut off = 0;
        for (p, &len) in parts.iter().zip(lens) {
            y[r * total + off..r * total + off + len].copy_from_slice(&p[r * len..r * len + len]);
            off += len;
        }
    }
    y
}

/// vjp of [`concat_cols_forward`]: split `grad_out [rows, Σ lens]` back into one gradient per part.
pub fn concat_cols_vjp(rows: usize, lens: &[usize], grad_out: &[f32]) -> Vec<Vec<f32>> {
    let total: usize = lens.iter().sum();
    let mut grads: Vec<Vec<f32>> = lens.iter().map(|&len| vec![0.0f32; rows * len]).collect();
    for r in 0..rows {
        let mut off = 0;
        for (gi, &len) in lens.iter().enumerate() {
            grads[gi][r * len..r * len + len]
                .copy_from_slice(&grad_out[r * total + off..r * total + off + len]);
            off += len;
        }
    }
    grads
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradcheck::{GradCheckCfg, check_op};

    #[test]
    fn slice_cols_extracts_the_range() {
        // [2,4] → cols [1,3)
        let x = vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0];
        assert_eq!(
            slice_cols_forward(&x, 2, 4, 1, 2),
            vec![1.0, 2.0, 11.0, 12.0]
        );
    }

    #[test]
    fn concat_cols_is_the_inverse_of_slicing() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0]; // [2,4]
        let a = slice_cols_forward(&x, 2, 4, 0, 2);
        let b = slice_cols_forward(&x, 2, 4, 2, 2);
        assert_eq!(concat_cols_forward(&[&a, &b], 2, &[2, 2]), x);
    }

    #[test]
    fn slice_cols_gradcheck() {
        let (rows, cols, start, len) = (3, 5, 1, 3);
        let x: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.37).sin()).collect();
        check_op(
            |ins| slice_cols_forward(ins[0], rows, cols, start, len),
            |_ins, g| vec![slice_cols_vjp(rows, cols, start, len, g)],
            &[x],
            &[0],
            GradCheckCfg::default(),
        )
        .expect("slice_cols vjp");
    }

    #[test]
    fn concat_cols_gradcheck() {
        let (rows, lens) = (3usize, [2usize, 3, 1]);
        let parts: Vec<Vec<f32>> = lens
            .iter()
            .enumerate()
            .map(|(pi, &len)| {
                (0..rows * len)
                    .map(|i| ((i + pi) as f32 * 0.21).cos())
                    .collect()
            })
            .collect();
        check_op(
            |ins| concat_cols_forward(ins, rows, &lens),
            |_ins, g| concat_cols_vjp(rows, &lens, g),
            &parts,
            &[0, 1, 2],
            GradCheckCfg::default(),
        )
        .expect("concat_cols vjp");
    }
}
