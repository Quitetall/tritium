use super::{AdditiveError, AdditiveScaleVjpPlan};
use core::{fmt, ops::Range};
use tritium_schema::ScaleAnchor;

/// Caller-owned output for a contiguous span of one owner's adjoint coordinates.
///
/// The operation validates its start and length against the complete stored-scale
/// or present-weight coordinate count before mutation. This is not a new owner,
/// tensor shard, payload authentication or a complete gradient observation.
pub struct AdditiveAdjointSpan<'a> {
    start: usize,
    values: &'a mut [f64],
    full: bool,
}

impl fmt::Debug for AdditiveAdjointSpan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AdditiveAdjointSpan")
            .field("start", &self.start)
            .field("len", &self.values.len())
            .finish()
    }
}

impl<'a> AdditiveAdjointSpan<'a> {
    /// Borrow output starting at the given global stored-coordinate ordinal.
    /// No allocation, validation or mutation occurs until an operation uses it.
    #[must_use]
    pub const fn new(start: usize, values: &'a mut [f64]) -> Self {
        Self {
            start,
            values,
            full: false,
        }
    }

    // Existing full-output interfaces retain their exact-length preflight.
    pub(super) const fn full(values: &'a mut [f64]) -> Self {
        Self {
            start: 0,
            values,
            full: true,
        }
    }

    pub(super) fn checked(
        self,
        count: usize,
    ) -> Result<(Range<usize>, &'a mut [f64]), AdditiveError> {
        if self.full && self.values.len() != count {
            return Err(AdditiveError::MatmulShape);
        }
        let end = self
            .start
            .checked_add(self.values.len())
            .ok_or(AdditiveError::SizeOverflow)?;
        if end > count {
            return Err(AdditiveError::MatmulShape);
        }
        Ok((self.start..end, self.values))
    }
}

pub(super) fn row_intersects(
    plan: &AdditiveScaleVjpPlan<'_>,
    row: usize,
    span: &Range<usize>,
    weights: bool,
) -> bool {
    if span.is_empty() {
        return false;
    }
    let layout = plan.layout();
    let planes = if weights {
        plan.plane_count().min(1)
    } else {
        plan.plane_count()
    };
    (0..planes).any(|plane| {
        plan.row_tiles(plane, row).any(|tile| {
            let (start, len) = if weights {
                (tile.trit, tile.len)
            } else {
                (
                    tile.scale,
                    if layout.law.anchor == ScaleAnchor::Group {
                        tile.len.div_ceil(usize::from(layout.group))
                    } else {
                        1
                    },
                )
            };
            start < span.end && span.start < start + len
        })
    })
}

pub(super) fn rows_for_span(
    plan: &AdditiveScaleVjpPlan<'_>,
    rows: usize,
    span: &Range<usize>,
    weights: bool,
) -> Range<usize> {
    let count = if weights {
        plan.present_weight_count()
    } else {
        plan.stored_scale_count()
    };
    if span.start == 0 && span.end == count {
        return 0..rows;
    }
    let mut first = rows;
    let mut end = 0;
    for row in 0..rows {
        if row_intersects(plan, row, span, weights) {
            first = first.min(row);
            end = row + 1;
        }
    }
    if first == rows { 0..0 } else { first..end }
}
