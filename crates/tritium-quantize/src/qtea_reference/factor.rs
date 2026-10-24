// Bunch-Kaufman pivot selection derived from LAPACK SSYTF2 (lower, 1x1 branch),
// revision6ec7f2bc4ecf4c4a93496aa2fa519575bc0e39ca. Copyright (c)1992-2023
// University of Tennessee/Tennessee Research Foundation;2000-2023 University
// of California Berkeley;2006-2023 University of Colorado Denver.
// Full upstream terms retained in LICENSE-LAPACK.

use super::{ArithmeticError, filled, finite};

pub(super) fn permuted(
    gram: &[f32],
    n: usize,
    order: &[usize],
    damping: f32,
) -> Result<Vec<f32>, ArithmeticError> {
    let m = order.len();
    let mut output = filled(m.checked_mul(m).ok_or(ArithmeticError::InvalidShape)?, 0.0)?;
    for (i, &row) in order.iter().enumerate() {
        for (j, &column) in order.iter().enumerate() {
            output[i * m + j] = gram[row * n + column];
        }
    }
    let diagonal = (0..m).try_fold(0.0, |sum, i| finite(sum + output[i * m + i]))?;
    let amount = finite(damping * (diagonal / m as f32))?;
    for i in 0..m {
        output[i * m + i] = finite(output[i * m + i] + amount)?;
    }
    Ok(output)
}

fn cholesky(matrix: &mut [f32], n: usize, relative_floor: f32) -> Result<(), ArithmeticError> {
    let max_diag = (0..n).fold(0.0_f32, |peak, i| peak.max(matrix[i * n + i]));
    let floor = finite(max_diag * relative_floor)?;
    if max_diag <= 0.0 || floor <= 0.0 {
        return Err(ArithmeticError::InvalidGram);
    }
    for row in 0..n {
        for column in 0..=row {
            let mut value = matrix[row * n + column];
            for k in 0..column {
                value = finite(value - finite(matrix[row * n + k] * matrix[column * n + k])?)?;
            }
            matrix[row * n + column] = if row == column {
                if value <= floor {
                    return Err(ArithmeticError::InvalidGram);
                }
                finite(value.sqrt())?
            } else {
                finite(value / matrix[column * n + column])?
            };
        }
        for column in row + 1..n {
            matrix[row * n + column] = 0.0;
        }
    }
    Ok(())
}

pub(super) fn order_rest(
    gram: &[f32],
    n: usize,
    order: &mut [usize],
    damping: f32,
    relative_floor: f32,
) -> Result<(), ArithmeticError> {
    let m = order.len();
    if m == 0 {
        return Ok(());
    }
    let mut matrix = permuted(gram, n, order, damping)?;
    // SPD admission is independent of pivot selection. Do not use indefinite
    // 2x2 pivots or a diagonal fallback to conceal unusable curvature.
    let mut admission = filled(matrix.len(), 0.0)?;
    admission.copy_from_slice(&matrix);
    cholesky(&mut admission, m, relative_floor)?;
    drop(admission);
    let alpha = (1.0_f32 + 17.0_f32.sqrt()) / 8.0;
    for k in 0..m {
        let absakk = matrix[k * m + k].abs();
        let (mut imax, mut colmax) = (k, 0.0);
        for i in k + 1..m {
            let value = matrix[i * m + k].abs();
            if value > colmax {
                imax = i;
                colmax = value;
            }
        }
        let mut pivot = k;
        if absakk < alpha * colmax {
            let mut rowmax = 0.0_f32;
            for j in k..m {
                if j != imax {
                    rowmax = rowmax.max(matrix[imax * m + j].abs());
                }
            }
            if absakk >= finite(alpha * colmax * (colmax / rowmax))? {
                // 1x1 current pivot.
            } else if matrix[imax * m + imax].abs() >= alpha * rowmax {
                pivot = imax;
            } else {
                return Err(ArithmeticError::InvalidGram);
            }
        }
        if pivot != k {
            for j in k..m {
                matrix.swap(k * m + j, pivot * m + j);
            }
            for i in k..m {
                matrix.swap(i * m + k, i * m + pivot);
            }
            order.swap(k, pivot);
        }
        let diagonal = matrix[k * m + k];
        if diagonal <= 0.0 {
            return Err(ArithmeticError::InvalidGram);
        }
        let reciprocal = finite(1.0 / diagonal)?;
        for j in k + 1..m {
            let scaled = finite(matrix[j * m + k] * reciprocal)?;
            for i in j..m {
                let value = finite(matrix[i * m + j] - finite(matrix[i * m + k] * scaled)?)?;
                matrix[i * m + j] = value;
                matrix[j * m + i] = value;
            }
        }
    }
    Ok(())
}

pub(super) fn inverse_upper(
    mut matrix: Vec<f32>,
    n: usize,
    floor: f32,
) -> Result<Vec<f32>, ArithmeticError> {
    cholesky(&mut matrix, n, floor)?;
    let mut inverse = filled(matrix.len(), 0.0)?;
    let mut work = filled(n, 0.0)?;
    for column in 0..n {
        for row in 0..n {
            let mut value = if row == column { 1.0 } else { 0.0 };
            for k in 0..row {
                value = finite(value - finite(matrix[row * n + k] * work[k])?)?;
            }
            work[row] = finite(value / matrix[row * n + row])?;
        }
        for row in (0..n).rev() {
            let mut value = work[row];
            for k in row + 1..n {
                value = finite(value - finite(matrix[k * n + row] * inverse[k * n + column])?)?;
            }
            inverse[row * n + column] = finite(value / matrix[row * n + row])?;
        }
    }
    drop(matrix);
    cholesky(&mut inverse, n, floor)?;
    for row in 0..n {
        for column in 0..row {
            inverse[column * n + row] = inverse[row * n + column];
            inverse[row * n + column] = 0.0;
        }
    }
    Ok(inverse)
}
