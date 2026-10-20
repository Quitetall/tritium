//! Checked compact plane-major geometry, shared by all allocation-aware views.

use tritium_schema::{AdditiveLayout, LayoutError, PlaneAllocation, PlaneRelation, ScaleAnchor};

use crate::AdditiveError;

/// Opaque caller-owned storage for one compact plane's row offsets.
///
/// Allocate exactly [`PerTileIndex::required_storage`] entries. The builder
/// validates all geometry before writing them; offsets are not wire data.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PerTileRowSpan {
    trit: usize,
    scale: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Geometry {
    rows: usize,
    cols: usize,
    tiles: usize,
    planes: usize,
    trits: usize,
    scales: usize,
    slots: usize,
}

/// Allocation-free preflight for compact payload and index sizing.
///
/// Uses the same geometry validator as the runtime index. A format decoder can
/// check its physical ledger before reserving coefficient or index buffers.
#[derive(Clone, Copy, Debug)]
pub struct PerTilePlan<'a> {
    layout: AdditiveLayout,
    counts: &'a [u8],
    geometry: Geometry,
}

impl<'a> PerTilePlan<'a> {
    /// Validate the layout and exact row-major counts without allocating.
    pub fn new(layout: AdditiveLayout, counts: &'a [u8]) -> Result<Self, AdditiveError> {
        Ok(Self {
            layout,
            counts,
            geometry: geometry(layout, counts)?,
        })
    }

    /// Exact required row span count.
    #[must_use]
    pub const fn index_len(&self) -> usize {
        self.geometry.slots
    }

    /// Checked logical lengths of count and row-span buffers in bytes.
    ///
    /// Excludes capacity slack, handles and allocator overhead; not full RSS.
    pub fn buffer_len_bytes(&self) -> Result<usize, AdditiveError> {
        buffer_len_bytes(self.counts.len(), self.geometry.slots)
    }

    /// Exact total compact coefficient count.
    #[must_use]
    pub const fn trit_count(&self) -> usize {
        self.geometry.trits
    }

    /// Exact stored scale count.
    #[must_use]
    pub const fn scale_count(&self) -> usize {
        self.geometry.scales
    }

    /// Actual maximum tile count, possibly zero.
    #[must_use]
    pub const fn plane_count(&self) -> usize {
        self.geometry.planes
    }

    /// Logical coefficient count for one present compact plane.
    ///
    /// Scans borrowed counts without allocating. This preflight is for codecs,
    /// not a replacement for the O(1) row lookup of a built runtime index.
    pub fn plane_trit_count(&self, plane: usize) -> Result<usize, AdditiveError> {
        if plane >= self.geometry.planes {
            return Err(AdditiveError::InvalidPlaneCount);
        }
        let mut total = 0;
        for (tile, &count) in self.counts.iter().enumerate() {
            if usize::from(count) > plane {
                total += (self.geometry.cols - (tile % self.geometry.tiles) * 256).min(256);
            }
        }
        Ok(total) // Bounded by the checked total coefficient count.
    }

    /// Populate exact caller-owned storage after preflight, without allocation.
    pub fn build(
        self,
        storage: &'a mut [PerTileRowSpan],
    ) -> Result<PerTileIndex<'a>, AdditiveError> {
        PerTileIndex::build_validated(self.layout, self.counts, storage, self.geometry)
    }
}

/// Fallibly constructed owned compact index; never self-referential.
///
/// Its immutable buffers yield a sealed borrowed index without geometry scans.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OwnedPerTileIndex {
    layout: AdditiveLayout,
    geometry: Geometry,
    counts: alloc::vec::Vec<u8>,
    spans: alloc::vec::Vec<PerTileRowSpan>,
}

#[cfg(feature = "alloc")]
impl OwnedPerTileIndex {
    /// Validate owned counts and fallibly reserve exactly sized row-span storage.
    pub fn new(layout: AdditiveLayout, counts: alloc::vec::Vec<u8>) -> Result<Self, AdditiveError> {
        let plan = PerTilePlan::new(layout, &counts)?;
        let geometry = plan.geometry;
        let requested = geometry
            .slots
            .checked_mul(core::mem::size_of::<PerTileRowSpan>())
            .ok_or(AdditiveError::SizeOverflow)?;
        let mut spans = alloc::vec::Vec::new();
        spans
            .try_reserve_exact(geometry.slots)
            .map_err(|_| AdditiveError::AllocationFailed { requested })?;
        spans.resize(geometry.slots, PerTileRowSpan::default());
        plan.build(&mut spans)?;
        Ok(Self {
            layout,
            geometry,
            counts,
            spans,
        })
    }

    /// Borrow the sealed runtime index with no allocation or geometry rescan.
    #[must_use]
    pub fn as_index(&self) -> PerTileIndex<'_> {
        PerTileIndex {
            layout: self.layout,
            geometry: self.geometry,
            counts: &self.counts,
            spans: &self.spans,
        }
    }

    /// Borrow immutable row-major tile counts without rebuilding a view.
    #[must_use]
    pub fn tile_counts(&self) -> &[u8] {
        &self.counts
    }

    /// Retained count and row-span buffer capacities in bytes.
    ///
    /// Excludes struct/allocator overhead and is not full process RSS.
    #[must_use]
    pub fn storage_bytes(&self) -> usize {
        self.counts.capacity() + self.spans.capacity() * core::mem::size_of::<PerTileRowSpan>()
    }
}

/// Validated borrowed index for row-aligned compact additive payloads.
///
/// Payload order is plane, row, active tile, logical coefficient. Group/Free
/// scales follow that order; Group/Tied stores base scales for nonpruned tiles;
/// Tensor/Free stores one global scale per present plane. Zero-count tiles have
/// no payload. Index storage is O(rows * actual maximum count), not O(weights).
/// This structural index does not admit a native backend or qualify a profile.
#[derive(Clone, Copy, Debug)]
pub struct PerTileIndex<'a> {
    pub(crate) layout: AdditiveLayout,
    geometry: Geometry,
    counts: &'a [u8],
    spans: &'a [PerTileRowSpan],
}

impl<'a> PerTileIndex<'a> {
    /// Exact number of caller-owned row spans, after validating counts/geometry.
    pub fn required_storage(layout: AdditiveLayout, counts: &[u8]) -> Result<usize, AdditiveError> {
        Ok(PerTilePlan::new(layout, counts)?.index_len())
    }

    /// Build the index without allocating, changing storage only after preflight.
    ///
    /// Counts must contain exactly `rows * ceil(cols/256)` row-major entries.
    /// Storage must have the exact required length; on error it is unchanged.
    pub fn build(
        layout: AdditiveLayout,
        counts: &'a [u8],
        storage: &'a mut [PerTileRowSpan],
    ) -> Result<Self, AdditiveError> {
        PerTilePlan::new(layout, counts)?.build(storage)
    }

    fn build_validated(
        layout: AdditiveLayout,
        counts: &'a [u8],
        storage: &'a mut [PerTileRowSpan],
        geometry: Geometry,
    ) -> Result<Self, AdditiveError> {
        if storage.len() != geometry.slots {
            return Err(AdditiveError::IndexStorageLength {
                expected: geometry.slots,
                got: storage.len(),
            });
        }
        let mut trit = 0;
        let mut scale = 0;
        for plane in 0..geometry.planes {
            for row in 0..geometry.rows {
                let scale_start = match (layout.law.anchor, layout.law.relation) {
                    (ScaleAnchor::Tensor, _) => plane,
                    (_, PlaneRelation::Tied { .. }) if plane > 0 => storage[row].scale,
                    _ => scale,
                };
                storage[plane * geometry.rows + row] = PerTileRowSpan {
                    trit,
                    scale: scale_start,
                };
                for tile in 0..geometry.tiles {
                    let count = usize::from(counts[row * geometry.tiles + tile]);
                    let len = (geometry.cols - tile * 256).min(256);
                    if count > plane {
                        trit += len;
                        if layout.law.anchor == ScaleAnchor::Group
                            && (layout.law.relation == PlaneRelation::Free || plane == 0)
                        {
                            scale += len.div_ceil(usize::from(layout.group));
                        }
                    }
                }
            }
        }
        debug_assert_eq!(trit, geometry.trits);
        Ok(Self {
            layout,
            geometry,
            counts,
            spans: storage,
        })
    }

    /// Exact compact coefficient count implied by the validated tile counts.
    #[must_use]
    pub const fn trit_count(&self) -> usize {
        self.geometry.trits
    }

    /// Exact stored scale count implied by the validated counts and scale law.
    #[must_use]
    pub const fn scale_count(&self) -> usize {
        self.geometry.scales
    }

    /// Checked logical lengths of borrowed count and row-span buffers in bytes.
    ///
    /// This is not their allocator capacity or full process RSS.
    pub fn buffer_len_bytes(&self) -> Result<usize, AdditiveError> {
        buffer_len_bytes(self.counts.len(), self.spans.len())
    }

    /// Iterate present tiles in one logical row using the shared sealed cursor.
    ///
    /// Starts at the indexed row; never scans preceding rows. Tile offsets are
    /// absolute within compact trit/stored-scale buffers. Invalid row or plane
    /// indices return typed errors, including every plane of all-pruned data.
    pub fn tiles_in_row(&self, plane: usize, row: usize) -> Result<ActiveTiles<'a>, AdditiveError> {
        if row >= self.geometry.rows {
            return Err(AdditiveError::RowOutOfRange);
        }
        if plane >= self.geometry.planes {
            return Err(AdditiveError::InvalidPlaneCount);
        }
        Ok(self.row_tiles(plane, row))
    }

    pub(crate) const fn plane_count(&self) -> usize {
        self.geometry.planes
    }

    pub(crate) const fn counts(&self) -> &'a [u8] {
        self.counts
    }

    pub(crate) fn plane_range(&self, plane: usize) -> core::ops::Range<usize> {
        let start = self.spans[plane * self.geometry.rows].trit;
        let end = if plane + 1 == self.geometry.planes {
            self.geometry.trits
        } else {
            self.spans[(plane + 1) * self.geometry.rows].trit
        };
        start..end
    }

    pub(crate) fn row_tiles(&self, plane: usize, row: usize) -> ActiveTiles<'a> {
        let span = self.spans[plane * self.geometry.rows + row];
        ActiveTiles {
            index: *self,
            plane,
            row,
            next: 0,
            trit: span.trit,
            scale: span.scale,
        }
    }
}

/// Opaque present tile yielded by the shared compact row cursor.
#[derive(Debug)]
pub struct ActiveTile {
    pub(crate) col: usize,
    pub(crate) len: usize,
    pub(crate) trit: usize,
    pub(crate) scale: usize,
}

impl ActiveTile {
    /// Logical columns occupied by this tile, with an unpadded ragged tail.
    #[must_use]
    pub fn columns(&self) -> core::ops::Range<usize> {
        self.col..self.col + self.len
    }

    /// Absolute start in the plane-major compact trit buffer.
    #[must_use]
    pub const fn trit_offset(&self) -> usize {
        self.trit
    }

    /// Absolute stored-scale start; tied laws refer to base-plane anchors.
    #[must_use]
    pub const fn scale_offset(&self) -> usize {
        self.scale
    }
}

/// Allocation-free iterator over a single checked compact row/plane.
#[derive(Debug)]
pub struct ActiveTiles<'a> {
    index: PerTileIndex<'a>,
    plane: usize,
    row: usize,
    next: usize,
    trit: usize,
    scale: usize,
}

fn buffer_len_bytes(counts: usize, slots: usize) -> Result<usize, AdditiveError> {
    slots
        .checked_mul(core::mem::size_of::<PerTileRowSpan>())
        .and_then(|bytes| bytes.checked_add(counts))
        .ok_or(AdditiveError::SizeOverflow)
}

impl Iterator for ActiveTiles<'_> {
    type Item = ActiveTile;

    fn next(&mut self) -> Option<Self::Item> {
        while self.next < self.index.geometry.tiles {
            let tile = self.next;
            self.next += 1;
            let count = usize::from(self.index.counts[self.row * self.index.geometry.tiles + tile]);
            let col = tile * 256;
            let len = (self.index.geometry.cols - col).min(256);
            let active = count > self.plane;
            let result = ActiveTile {
                col,
                len,
                trit: self.trit,
                scale: self.scale,
            };
            if self.index.layout.law.anchor == ScaleAnchor::Group {
                let stored = match self.index.layout.law.relation {
                    PlaneRelation::Free => active,
                    PlaneRelation::Tied { .. } => count > 0,
                    _ => unreachable!("validated scale relation"),
                };
                if stored {
                    self.scale += len.div_ceil(usize::from(self.index.layout.group));
                }
            }
            if active {
                self.trit += len;
                return Some(result);
            }
        }
        None
    }
}

fn geometry(layout: AdditiveLayout, counts: &[u8]) -> Result<Geometry, AdditiveError> {
    layout.validate().map_err(AdditiveError::InvalidLayout)?;
    if layout.allocation != PlaneAllocation::PerTile {
        return Err(AdditiveError::UnsupportedAllocation);
    }
    let rows = usize::try_from(layout.rows).map_err(|_| AdditiveError::SizeOverflow)?;
    let cols = usize::try_from(layout.cols).map_err(|_| AdditiveError::SizeOverflow)?;
    rows.checked_mul(cols).ok_or(AdditiveError::SizeOverflow)?;
    let tiles = cols.div_ceil(256);
    let expected = rows.checked_mul(tiles).ok_or(AdditiveError::SizeOverflow)?;
    if counts.len() != expected {
        return Err(AdditiveError::TileCountMismatch {
            expected,
            got: counts.len(),
        });
    }
    let mut planes = 0;
    let mut trits = 0usize;
    let mut scales = 0usize;
    for (tile, &count) in counts.iter().enumerate() {
        if count > layout.max_planes {
            return Err(AdditiveError::InvalidTilePlaneCount {
                tile,
                count,
                max: layout.max_planes,
            });
        }
        planes = planes.max(usize::from(count));
        let len = (cols - (tile % tiles) * 256).min(256);
        let coefficients = len
            .checked_mul(usize::from(count))
            .ok_or(AdditiveError::SizeOverflow)?;
        trits = trits
            .checked_add(coefficients)
            .ok_or(AdditiveError::SizeOverflow)?;
        let groups = len.div_ceil(usize::from(layout.group));
        let tile_scales = match (layout.law.anchor, layout.law.relation) {
            (ScaleAnchor::Group, PlaneRelation::Free) => groups * usize::from(count),
            (ScaleAnchor::Group, PlaneRelation::Tied { .. }) => groups * usize::from(count > 0),
            (ScaleAnchor::Tensor, PlaneRelation::Free) => 0,
            _ => return Err(AdditiveError::InvalidLayout(LayoutError::UnsupportedLaw)),
        };
        scales = scales
            .checked_add(tile_scales)
            .ok_or(AdditiveError::SizeOverflow)?;
    }
    if layout.law.anchor == ScaleAnchor::Tensor {
        scales = planes;
    }
    let slots = rows
        .checked_mul(planes)
        .ok_or(AdditiveError::SizeOverflow)?;
    Ok(Geometry {
        rows,
        cols,
        tiles,
        planes,
        trits,
        scales,
        slots,
    })
}
