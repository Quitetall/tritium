//! Bounded shared-owner chain-rule reconstruction (candidate ADR 0092).
//!
//! This is not a curvature estimator, cache, source verifier or fit receipt.
//! Sum all actual uses before an enclosing estimator forms an outer product.

/// One actual invocation of an ordinary operation on the same `[rows, columns]` owner.
#[derive(Clone, Copy, Debug)]
pub enum OwnerScoreUse<'a> {
    /// `output = input W^T`: row-major input and full downstream output adjoint.
    Matmul {
        /// Number of flattened input/output samples in this invocation.
        samples: usize,
        /// Row-major `[samples, columns]` input values.
        inputs: &'a [f32],
        /// Row-major `[samples, rows]` output adjoints.
        adjoints: &'a [f32],
    },
    /// Ordinary lookup, without padding, frequency scaling or custom backward rules.
    Gather {
        /// Signed owner-row indices, including repeated occurrences.
        indices: &'a [i64],
        /// Row-major `[indices.len(), columns]` full downstream output adjoints.
        adjoints: &'a [f32],
    },
}

/// Rejection at the numerical owner-score seam; no estimator fallback is performed.
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum OwnerScoreError {
    /// Empty/overflowing geometry or an incorrectly sized buffer.
    Shape,
    /// Negative or out-of-range owner-row index.
    Index,
    /// Nonfinite values in any observation, including an unrequested row.
    Nonfinite,
    /// Logical borrowed observations or returned row exceed their respective limit.
    Budget {
        /// Required logical bytes.
        required: u64,
        /// Declared byte limit.
        limit: u64,
    },
    /// Byte-count overflow or allocation failure.
    Allocation,
}
impl core::fmt::Display for OwnerScoreError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "owner score rejected: {self:?}")
    }
}
impl std::error::Error for OwnerScoreError {}

/// Validated borrowed observations for one objective/probe window.
///
/// All uses must be actual invocations of this owner, in canonical coordinates.
/// No source identity, invocation completeness or objective execution is inferred.
/// Values remain borrowed, not snapshotted; foreign-buffer immutability is the
/// adapter/controller's obligation. No token mask is applied at this seam.
#[derive(Debug)]
pub struct OwnerScoreWindow<'a> {
    rows: usize,
    columns: usize,
    uses: &'a [OwnerScoreUse<'a>],
}

impl<'a> OwnerScoreWindow<'a> {
    /// Validate once before row replay; count every supplied logical buffer even
    /// when a caller aliases storage. The limit excludes descriptor/tape/RSS bytes.
    pub fn new(
        rows: usize,
        columns: usize,
        uses: &'a [OwnerScoreUse<'a>],
        max_observation_bytes: u64,
    ) -> Result<Self, OwnerScoreError> {
        if rows == 0 || columns == 0 || uses.is_empty() {
            return Err(OwnerScoreError::Shape);
        }
        let mut bytes = 0u64;
        for usage in uses {
            match *usage {
                OwnerScoreUse::Matmul {
                    samples,
                    inputs,
                    adjoints,
                } => {
                    if samples == 0
                        || samples.checked_mul(columns) != Some(inputs.len())
                        || samples.checked_mul(rows) != Some(adjoints.len())
                    {
                        return Err(OwnerScoreError::Shape);
                    }
                    add_bytes(&mut bytes, inputs.len(), 4)?;
                    add_bytes(&mut bytes, adjoints.len(), 4)?;
                }
                OwnerScoreUse::Gather { indices, adjoints } => {
                    if indices.is_empty()
                        || indices.len().checked_mul(columns) != Some(adjoints.len())
                    {
                        return Err(OwnerScoreError::Shape);
                    }
                    add_bytes(&mut bytes, indices.len(), 8)?;
                    add_bytes(&mut bytes, adjoints.len(), 4)?;
                }
            }
        }
        admit(bytes, max_observation_bytes)?;
        // Validate all values, not merely the selected row. Row replay need not
        // rescan vocabulary-sized buffers for each requested row.
        for usage in uses {
            let adjoints = match *usage {
                OwnerScoreUse::Matmul {
                    inputs, adjoints, ..
                } => {
                    finite(inputs)?;
                    adjoints
                }
                OwnerScoreUse::Gather { indices, adjoints } => {
                    if indices
                        .iter()
                        .any(|&index| usize::try_from(index).map_or(true, |index| index >= rows))
                    {
                        return Err(OwnerScoreError::Index);
                    }
                    adjoints
                }
            };
            finite(adjoints)?;
        }
        Ok(Self {
            rows,
            columns,
            uses,
        })
    }

    /// Reconstruct one row using f64 multiplication/accumulation in use/sample
    /// order. Only `columns * 8` output bytes are allocated; failure returns no
    /// partial row. This does not promise framework f32 reduction bit parity.
    pub fn row_score(
        &self,
        row: usize,
        max_workspace_bytes: u64,
    ) -> Result<Vec<f64>, OwnerScoreError> {
        if row >= self.rows {
            return Err(OwnerScoreError::Index);
        }
        let mut bytes = 0;
        add_bytes(&mut bytes, self.columns, 8)?;
        admit(bytes, max_workspace_bytes)?;
        let mut score = Vec::new();
        score
            .try_reserve_exact(self.columns)
            .map_err(|_| OwnerScoreError::Allocation)?;
        score.resize(self.columns, 0.0);
        for usage in self.uses {
            match *usage {
                OwnerScoreUse::Matmul {
                    samples,
                    inputs,
                    adjoints,
                } => {
                    for sample in 0..samples {
                        let delta = f64::from(adjoints[sample * self.rows + row]);
                        for (value, &input) in score
                            .iter_mut()
                            .zip(&inputs[sample * self.columns..(sample + 1) * self.columns])
                        {
                            *value += delta * f64::from(input);
                        }
                    }
                }
                OwnerScoreUse::Gather { indices, adjoints } => {
                    for (sample, &index) in indices.iter().enumerate() {
                        if usize::try_from(index) == Ok(row) {
                            for (value, &delta) in score
                                .iter_mut()
                                .zip(&adjoints[sample * self.columns..(sample + 1) * self.columns])
                            {
                                *value += f64::from(delta);
                            }
                        }
                    }
                }
            }
        }
        if score.iter().any(|value| !value.is_finite()) {
            return Err(OwnerScoreError::Nonfinite);
        }
        Ok(score)
    }
}

fn add_bytes(bytes: &mut u64, count: usize, width: u64) -> Result<(), OwnerScoreError> {
    *bytes = u64::try_from(count)
        .ok()
        .and_then(|count| count.checked_mul(width))
        .and_then(|amount| bytes.checked_add(amount))
        .ok_or(OwnerScoreError::Allocation)?;
    Ok(())
}
fn admit(required: u64, limit: u64) -> Result<(), OwnerScoreError> {
    if required > limit {
        return Err(OwnerScoreError::Budget { required, limit });
    }
    Ok(())
}
fn finite(values: &[f32]) -> Result<(), OwnerScoreError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(OwnerScoreError::Nonfinite);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owner_cross_terms_reinforce_and_cancel() {
        for delta in [1.0, -1.0] {
            let uses = [
                OwnerScoreUse::Matmul {
                    samples: 1,
                    inputs: &[1.0],
                    adjoints: &[1.0],
                },
                OwnerScoreUse::Gather {
                    indices: &[0],
                    adjoints: &[delta],
                },
            ];
            let window = OwnerScoreWindow::new(1, 1, &uses, 20).unwrap();
            let score = window.row_score(0, 8).unwrap()[0];
            assert_eq!(score * score, if delta > 0.0 { 4.0 } else { 0.0 });
            assert_ne!(score * score, 2.0); // Separate-use squared scores.
        }
    }

    #[test]
    fn repeated_calls_and_indices_count_before_squaring() {
        let uses = [
            OwnerScoreUse::Gather {
                indices: &[1, 1, 0],
                adjoints: &[1.0, 2.0, 3.0, 4.0, 8.0, 9.0],
            },
            OwnerScoreUse::Matmul {
                samples: 2,
                inputs: &[2.0, 3.0, 5.0, 7.0],
                adjoints: &[0.0, 2.0, 3.0, 4.0],
            },
            OwnerScoreUse::Gather {
                indices: &[1],
                adjoints: &[-1.0, -2.0],
            },
        ];
        let window = OwnerScoreWindow::new(2, 2, &uses, 1000).unwrap();
        assert_eq!(window.row_score(1, 16).unwrap(), [27.0, 38.0]);
        assert_eq!(window.row_score(0, 16).unwrap(), [23.0, 30.0]);
        assert_eq!(window.row_score(1, 16).unwrap(), [27.0, 38.0]);
    }

    #[test]
    fn rejects_all_nonfinite_content_and_indices() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let adjoints = [0.0, bad];
            let uses = [OwnerScoreUse::Matmul {
                samples: 1,
                inputs: &[1.0],
                adjoints: &adjoints,
            }];
            assert_eq!(
                OwnerScoreWindow::new(2, 1, &uses, 100).unwrap_err(),
                OwnerScoreError::Nonfinite
            );
        }
        for index in [-1, 2, i64::MAX] {
            let indices = [index];
            let uses = [OwnerScoreUse::Gather {
                indices: &indices,
                adjoints: &[1.0],
            }];
            assert_eq!(
                OwnerScoreWindow::new(2, 1, &uses, 100).unwrap_err(),
                OwnerScoreError::Index
            );
        }
    }

    #[test]
    fn admission_is_checked_before_row_allocation_and_independent_of_vocabulary() {
        let uses = [OwnerScoreUse::Gather {
            indices: &[0],
            adjoints: &[1.0, 2.0],
        }];
        assert_eq!(
            OwnerScoreWindow::new(1_000_000, 2, &uses, 15).unwrap_err(),
            OwnerScoreError::Budget {
                required: 16,
                limit: 15
            }
        );
        let window = OwnerScoreWindow::new(1_000_000, 2, &uses, 16).unwrap();
        assert_eq!(
            window.row_score(0, 15).unwrap_err(),
            OwnerScoreError::Budget {
                required: 16,
                limit: 15
            }
        );
        assert_eq!(window.row_score(999_999, 16).unwrap(), [0.0, 0.0]);
        assert_eq!(
            window.row_score(1_000_000, 16).unwrap_err(),
            OwnerScoreError::Index
        );
    }

    #[test]
    fn malformed_geometry_rejects_including_late_use_and_overflow() {
        assert_eq!(
            OwnerScoreWindow::new(1, 1, &[], 100).unwrap_err(),
            OwnerScoreError::Shape
        );
        let uses = [
            OwnerScoreUse::Gather {
                indices: &[0],
                adjoints: &[1.0, 2.0],
            },
            OwnerScoreUse::Matmul {
                samples: usize::MAX,
                inputs: &[],
                adjoints: &[],
            },
        ];
        assert_eq!(
            OwnerScoreWindow::new(1, 2, &uses, 100).unwrap_err(),
            OwnerScoreError::Shape
        );
        let uses = [OwnerScoreUse::Matmul {
            samples: 2,
            inputs: &[1.0, 2.0],
            adjoints: &[1.0],
        }];
        assert_eq!(
            OwnerScoreWindow::new(1, 1, &uses, 100).unwrap_err(),
            OwnerScoreError::Shape
        );
    }

    #[test]
    fn products_widen_before_multiplication() {
        let uses = [OwnerScoreUse::Matmul {
            samples: 1,
            inputs: &[f32::MAX],
            adjoints: &[f32::MAX],
        }];
        let window = OwnerScoreWindow::new(1, 1, &uses, 8).unwrap();
        assert_eq!(
            window.row_score(0, 8).unwrap()[0],
            f64::from(f32::MAX).powi(2)
        );
    }
}
