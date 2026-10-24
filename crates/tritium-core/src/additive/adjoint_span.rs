use super::{AdditiveError, AdditiveScaleVjpPlan};
use core::{fmt, ops::Range};
use tritium_schema::{PlaneRelation, ScaleAnchor};

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
    if weights {
        return plan.plane_count() != 0 && intersects(&plan.row_trit_range(0, row), span);
    }
    (0..stored_planes(plan)).any(|plane| intersects(&plan.row_scale_range(plane, row), span))
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
    if span.is_empty() {
        return 0..0;
    }
    if weights {
        return indexed_rows(rows, span, |row| plan.row_trit_range(0, row));
    }
    let (mut first, mut end) = (rows, 0);
    for plane in 0..stored_planes(plan) {
        let next = if plan.layout().law.anchor == ScaleAnchor::Tensor {
            let scale = plan.row_scale_range(plane, 0).start;
            if !span.contains(&scale) {
                continue;
            }
            let entire_plane =
                plan.row_trit_range(plane, 0).start..plan.row_trit_range(plane, rows - 1).end;
            indexed_rows(rows, &entire_plane, |row| plan.row_trit_range(plane, row))
        } else {
            indexed_rows(rows, span, |row| plan.row_scale_range(plane, row))
        };
        if !next.is_empty() {
            first = first.min(next.start);
            end = end.max(next.end);
        }
    }
    if first < end { first..end } else { 0..0 }
}

fn stored_planes(plan: &AdditiveScaleVjpPlan<'_>) -> usize {
    if matches!(plan.layout().law.relation, PlaneRelation::Tied { .. }) {
        plan.plane_count().min(1)
    } else {
        plan.plane_count()
    }
}

fn intersects(a: &Range<usize>, b: &Range<usize>) -> bool {
    !a.is_empty() && !b.is_empty() && a.start < b.end && b.start < a.end
}

fn indexed_rows(
    rows: usize,
    span: &Range<usize>,
    mut range: impl FnMut(usize) -> Range<usize>,
) -> Range<usize> {
    if rows == 0 || span.is_empty() {
        return 0..0;
    }
    // Clamp to this plane first: otherwise leading/trailing empty-row plateaus
    // could be spuriously included when a span crosses plane boundaries.
    let start = span.start.max(range(0).start);
    let stop = span.end.min(range(rows - 1).end);
    if start >= stop {
        return 0..0;
    }
    let first = prefix_len(rows, |row| range(row).end <= start);
    let end = prefix_len(rows, |row| range(row).start < stop);
    if first < end { first..end } else { 0..0 }
}

fn prefix_len(rows: usize, mut before: impl FnMut(usize) -> bool) -> usize {
    let (mut start, mut end) = (0, rows);
    while start < end {
        let mid = start + (end - start) / 2;
        if before(mid) {
            start = mid + 1;
        } else {
            end = mid;
        }
    }
    start
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    #[test]
    fn lookup_work_is_bounded_by_logarithmic_prefix_reads() {
        for rows in [1, 1000, 1_000_000_000] {
            let mut reads = 0;
            let limit = 2 * usize::BITS as usize + 2;
            let result = indexed_rows(rows, &(rows * 3 - 1..rows * 3), |row| {
                reads += 1;
                assert!(
                    reads <= limit,
                    "metadata lookup scanned rows: reads={reads}"
                );
                row * 3..(row + 1) * 3
            });
            assert_eq!(result, rows - 1..rows);
        }
    }

    #[test]
    fn prefix_lookup_matches_linear_oracle_with_empty_rows_and_external_spans() {
        for rows in 0..32 {
            let mut prefix = alloc::vec![0];
            for row in 0..rows {
                let n = if row < 3 || row + 3 >= rows || row % 3 == 0 {
                    0
                } else {
                    row % 5
                };
                prefix.push(prefix.last().unwrap() + n);
            }
            let total = *prefix.last().unwrap();
            for start in 0..total + 3 {
                for end in start..(start + 7).min(total + 3) {
                    let span = start..end;
                    let mut wanted = rows..0;
                    for row in 0..rows {
                        if intersects(&(prefix[row]..prefix[row + 1]), &span) {
                            wanted.start = wanted.start.min(row);
                            wanted.end = row + 1;
                        }
                    }
                    if wanted.is_empty() {
                        wanted = 0..0;
                    }
                    assert_eq!(
                        indexed_rows(rows, &span, |r| prefix[r]..prefix[r + 1]),
                        wanted
                    );
                }
            }
        }
    }

    fn tile_oracle(
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

    #[test]
    fn indexed_geometry_matches_original_tile_walk_for_all_laws_and_pruning() {
        use crate::{OwnedPerTileIndex, UniformPlan};
        use tritium_schema::{
            ADMITTED_LAWS, AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, Transport,
        };
        for admitted in ADMITTED_LAWS {
            for planes in 1..=admitted.max_planes {
                for mode in 0..3 {
                    let layout = AdditiveLayout {
                        rows: 12,
                        cols: 260,
                        tile: 256,
                        group: 32,
                        max_planes: admitted.max_planes,
                        allocation: if mode == 0 {
                            PlaneAllocation::Uniform
                        } else {
                            PlaneAllocation::PerTile
                        },
                        codec: PlaneCodec::D2,
                        law: admitted.law,
                        basis: Basis::Identity,
                        transport: Transport::Raw,
                    };
                    let counts: alloc::vec::Vec<_> = (0..24)
                        .map(|tile| {
                            let row = tile / 2;
                            if mode == 2 || !(2..10).contains(&row) || row % 3 == 0 {
                                0
                            } else if tile % 2 == 0 {
                                planes
                            } else {
                                1
                            }
                        })
                        .collect();
                    let index =
                        (mode != 0).then(|| OwnedPerTileIndex::new(layout, counts).unwrap());
                    let plan = if let Some(index) = &index {
                        AdditiveScaleVjpPlan::from_per_tile(index.as_index())
                    } else {
                        AdditiveScaleVjpPlan::from_uniform(
                            UniformPlan::new(layout, planes).unwrap(),
                        )
                    };
                    for weights in [false, true] {
                        let count = if weights {
                            plan.present_weight_count()
                        } else {
                            plan.stored_scale_count()
                        };
                        for start in [
                            0,
                            1.min(count),
                            count / 3,
                            count / 2,
                            count.saturating_sub(1),
                            count,
                        ] {
                            for len in [0, 1, 3, 37, count - start] {
                                let span = start..(start + len).min(count);
                                let mut wanted = 0..0;
                                for row in 0..12 {
                                    let yes = tile_oracle(&plan, row, &span, weights);
                                    assert_eq!(row_intersects(&plan, row, &span, weights), yes);
                                    if yes {
                                        if wanted.is_empty() {
                                            wanted.start = row;
                                        }
                                        wanted.end = row + 1;
                                    }
                                }
                                if wanted.is_empty() {
                                    wanted = 0..0;
                                }
                                // The full-output path deliberately retains scratch behavior
                                // for all rows, even for a completely pruned owner.
                                if span.start == 0 && span.end == count {
                                    wanted = 0..12;
                                }
                                assert_eq!(rows_for_span(&plan, 12, &span, weights), wanted);
                            }
                        }
                    }
                }
            }
        }
    }
}
