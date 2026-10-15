//! Exact storage views: no repartitioning, estimator or numerical approximation.
//! The common controller validates geometry before materialization or block reads.
use super::DirectFitError;
use super::direct::reserved_vec;
use std::borrow::Cow;

pub(super) struct MetricBlock<'a> {
    pub width: usize,
    pub values: Cow<'a, [f64]>,
}

pub(super) enum DirectMetric<'a> {
    Dense(Cow<'a, [f64]>),
    BlockDiagonal(Vec<MetricBlock<'a>>),
}

impl<'a> DirectMetric<'a> {
    pub(super) fn validate(&self, cols: usize) -> Result<(), DirectFitError> {
        match self {
            Self::Dense(values) => check_length(values.len(), cols)?,
            Self::BlockDiagonal(blocks) => {
                let mut total = 0_usize;
                for block in blocks {
                    if block.width == 0 {
                        return Err(DirectFitError::Unsupported("zero metric block width"));
                    }
                    check_length(block.values.len(), block.width)?;
                    total = total
                        .checked_add(block.width)
                        .ok_or(DirectFitError::ShapeOverflow)?;
                }
                if total != cols {
                    return Err(DirectFitError::Unsupported("metric block coverage"));
                }
            }
        }
        let mut scale = 0.0_f64;
        self.visit_stored(cols, |index, value| {
            if !value.is_finite() || (index / cols == index % cols && value < 0.) {
                return Err(DirectFitError::InvalidGram { index });
            }
            scale = scale.max(value.abs());
            Ok(())
        })?;
        // The original dense admission uses one global scale, not one per block.
        let scale = scale.max(f64::MIN_POSITIVE);
        match self {
            Self::Dense(values) => check_symmetry(values, cols, scale),
            Self::BlockDiagonal(blocks) => {
                for block in blocks {
                    check_symmetry(&block.values, block.width, scale)?;
                }
                Ok(())
            }
        }
    }

    fn visit_stored(
        &self,
        cols: usize,
        mut visit: impl FnMut(usize, f64) -> Result<(), DirectFitError>,
    ) -> Result<(), DirectFitError> {
        match self {
            Self::Dense(values) => {
                for (index, &value) in values.iter().enumerate() {
                    visit(index, value)?;
                }
            }
            Self::BlockDiagonal(blocks) => {
                let mut start = 0;
                for block in blocks {
                    for (index, &value) in block.values.iter().enumerate() {
                        visit(
                            (start + index / block.width) * cols + start + index % block.width,
                            value,
                        )?;
                    }
                    start += block.width;
                }
            }
        }
        Ok(())
    }

    pub(super) fn as_dense(&self) -> Option<&[f64]> {
        match self {
            Self::Dense(values) => Some(values),
            Self::BlockDiagonal(_) => None,
        }
    }

    // Coupled solvers and existing two-axis basis transforms retain their exact
    // dense arithmetic. This conversion only copies entries and implicit zeros.
    pub(super) fn into_dense(self, cols: usize) -> Result<Cow<'a, [f64]>, DirectFitError> {
        match self {
            Self::Dense(values) => Ok(values),
            Self::BlockDiagonal(blocks) => {
                let mut dense = reserved_vec(cols * cols)?;
                dense.resize(cols * cols, 0.);
                let mut start = 0;
                for block in blocks {
                    for row in 0..block.width {
                        dense[(start + row) * cols + start
                            ..(start + row) * cols + start + block.width]
                            .copy_from_slice(
                                &block.values[row * block.width..(row + 1) * block.width],
                            );
                    }
                    start += block.width;
                }
                Ok(Cow::Owned(dense))
            }
        }
    }

    // The execution group can cross several calibration blocks or end ragged.
    // Preserve every within-block entry and each implicit off-block zero.
    pub(super) fn block(
        &self,
        cols: usize,
        start: usize,
        end: usize,
    ) -> Result<Cow<'_, [f64]>, DirectFitError> {
        let width = end - start;
        if let Self::BlockDiagonal(blocks) = self {
            let mut offset = 0;
            for block in blocks {
                if start == offset && end == offset + block.width {
                    return Ok(Cow::Borrowed(&block.values));
                }
                offset += block.width;
            }
        }
        let mut local = reserved_vec(width * width)?;
        match self {
            Self::Dense(values) => {
                for row in start..end {
                    local.extend_from_slice(&values[row * cols + start..row * cols + end]);
                }
            }
            Self::BlockDiagonal(blocks) => {
                local.resize(width * width, 0.);
                let mut offset = 0;
                for block in blocks {
                    let stop = offset + block.width;
                    let first = start.max(offset);
                    let last = end.min(stop);
                    for row in first..last {
                        let source = (row - offset) * block.width + first - offset;
                        let destination = (row - start) * width + first - start;
                        local[destination..destination + last - first]
                            .copy_from_slice(&block.values[source..source + last - first]);
                    }
                    offset = stop;
                }
            }
        }
        Ok(Cow::Owned(local))
    }
}

fn check_length(actual: usize, cols: usize) -> Result<(), DirectFitError> {
    let expected = cols
        .checked_mul(cols)
        .ok_or(DirectFitError::ShapeOverflow)?;
    if actual != expected {
        return Err(DirectFitError::GramLength { expected, actual });
    }
    Ok(())
}

fn check_symmetry(values: &[f64], cols: usize, scale: f64) -> Result<(), DirectFitError> {
    for row in 0..cols {
        for col in row + 1..cols {
            if (values[row * cols + col] / scale - values[col * cols + row] / scale).abs() > 1e-10 {
                return Err(DirectFitError::AsymmetricGram);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ragged_cross_block_views_and_dense_materialization_preserve_bits() {
        let blocks = DirectMetric::BlockDiagonal(vec![
            MetricBlock {
                width: 2,
                values: Cow::Owned(vec![3., -0., -0., 2.]),
            },
            MetricBlock {
                width: 3,
                values: Cow::Owned(vec![4., 0.25, 0.5, 0.25, 5., 0.75, 0.5, 0.75, 6.]),
            },
        ]);
        blocks.validate(5).unwrap();
        let dense = DirectMetric::Dense(Cow::Owned(vec![
            3., -0., 0., 0., 0., -0., 2., 0., 0., 0., 0., 0., 4., 0.25, 0.5, 0., 0., 0.25, 5.,
            0.75, 0., 0., 0.5, 0.75, 6.,
        ]));
        for start in 0..5 {
            for end in start + 1..=5 {
                assert_eq!(
                    blocks
                        .block(5, start, end)
                        .unwrap()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    dense
                        .block(5, start, end)
                        .unwrap()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
            }
        }
        assert!(matches!(blocks.block(5, 0, 2).unwrap(), Cow::Borrowed(_)));
        assert!(matches!(blocks.block(5, 2, 5).unwrap(), Cow::Borrowed(_)));
        assert_eq!(
            blocks
                .into_dense(5)
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            dense
                .as_dense()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn structured_admission_preserves_global_tolerance_and_logical_error_indices() {
        let mut blocks = DirectMetric::BlockDiagonal(vec![
            MetricBlock {
                width: 1,
                values: Cow::Owned(vec![1e300]),
            },
            MetricBlock {
                width: 2,
                values: Cow::Owned(vec![1., 0.01, 0.04, 1.]),
            },
        ]);
        blocks.validate(3).unwrap();
        let DirectMetric::BlockDiagonal(values) = &mut blocks else {
            unreachable!()
        };
        values[1].values.to_mut()[3] = -1.;
        assert_eq!(
            blocks.validate(3),
            Err(DirectFitError::InvalidGram { index: 8 })
        );
        assert!(blocks.validate(4).is_err());
    }

    #[test]
    fn dense_and_structured_admission_report_the_same_invalid_entries() {
        for values in [
            [f64::NAN, 0., 0., 1.],
            [1., f64::INFINITY, 0., 1.],
            [1., 0., 0., -1.],
            [1., 0.25, 0.75, 1.],
            [-0., 0., 0., 1.],
        ] {
            let make = || {
                DirectMetric::BlockDiagonal(vec![
                    MetricBlock {
                        width: 1,
                        values: Cow::Borrowed(&[1.]),
                    },
                    MetricBlock {
                        width: 2,
                        values: Cow::Borrowed(&values),
                    },
                ])
            };
            let dense = DirectMetric::Dense(make().into_dense(3).unwrap());
            assert_eq!(make().validate(3), dense.validate(3));
        }
        let wrong_length = DirectMetric::BlockDiagonal(vec![MetricBlock {
            width: 2,
            values: Cow::Borrowed(&[1., 0., 1.]),
        }]);
        assert_eq!(
            wrong_length.validate(2),
            Err(DirectFitError::GramLength {
                expected: 4,
                actual: 3
            })
        );
        assert!(
            DirectMetric::BlockDiagonal(vec![MetricBlock {
                width: 0,
                values: Cow::Borrowed(&[]),
            }])
            .validate(0)
            .is_err()
        );
    }
}
