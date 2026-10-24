// Copyright (c) 2026 The QTEA Authors. Derived from pinned qtea.py.
// MIT notice in ../../qtea_reference/LICENSE-MIT.

use super::{TensorFit, TensorResidual};
use crate::qtea_reference::{ArithmeticError, RefinedColumn, SparseResidualColumn, filled, finite};

pub(super) fn run(
    fit: &mut TensorFit,
    original: &[f32],
    upper: &[f32],
    salient: usize,
) -> Result<(), ArithmeticError> {
    let (rows, columns, width) = (fit.rows, fit.columns, fit.config.group_size);
    let mut working = filled(original.len(), 0.0)?;
    for row in 0..rows {
        for (column, &source) in fit.order.iter().enumerate() {
            working[row * columns + column] = original[row * columns + source];
        }
    }
    let mut block = filled(rows * width, 0.0)?;
    let mut errors = filled(rows * width, 0.0)?;
    let mut column_values = filled(rows, 0.0)?;
    let mut residual_values = filled(rows, 0.0)?;
    let mut dense_residual = filled(rows, 0.0)?;
    for start in (0..columns).step_by(width) {
        let stop = start + width;
        for row in 0..rows {
            block[row * width..(row + 1) * width]
                .copy_from_slice(&working[row * columns + start..row * columns + stop]);
        }
        let diagonal_mean = finite(
            (0..width).try_fold(0.0, |sum, i| {
                finite(sum + upper[(start + i) * columns + start + i])
            })? / width as f32,
        )?;
        for local in 0..width {
            let position = start + local;
            let original_column = fit.order[position];
            for row in 0..rows {
                column_values[row] = block[row * width + local];
            }
            let RefinedColumn {
                trits,
                multiplier,
                mut values,
            } = fit.groups[original_column / width].refine_column(
                &column_values,
                fit.config.rescale_iterations,
                9 * rows,
            )?;
            fit.multipliers[original_column] = multiplier;
            for (row, &trit) in trits.iter().enumerate() {
                fit.trits[row * columns + original_column] = trit;
            }
            if position < salient {
                for row in 0..rows {
                    residual_values[row] = finite(column_values[row] - values[row])?;
                }
                let residual = SparseResidualColumn::fit(&residual_values, rows / 2)?;
                residual.decode_into(&mut dense_residual)?;
                for row in 0..rows {
                    values[row] = finite(values[row] + dense_residual[row])?;
                }
                fit.residuals.push(TensorResidual {
                    column: original_column,
                    residual,
                });
            }
            let d = upper[position * columns + position];
            let curvature = finite(d / diagonal_mean)?.clamp(0.3, 3.0);
            let gamma = finite(
                (-fit.config.decay * f64::from(curvature) * local as f64 / width as f64).exp()
                    as f32,
            )?;
            for row in 0..rows {
                let error = finite(finite(column_values[row] - values[row])? / d)?;
                errors[row * width + local] = error;
                for target in local..width {
                    let propagated = finite(error * upper[position * columns + start + target])?;
                    block[row * width + target] =
                        finite(block[row * width + target] - finite(gamma * propagated)?)?;
                }
            }
        }
        for row in 0..rows {
            for target in stop..columns {
                let mut correction = 0.0;
                for local in 0..width {
                    correction = finite(
                        correction
                            + finite(
                                errors[row * width + local]
                                    * upper[(start + local) * columns + target],
                            )?,
                    )?;
                }
                working[row * columns + target] =
                    finite(working[row * columns + target] - correction)?;
            }
        }
    }
    Ok(())
}
