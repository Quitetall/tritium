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

#[derive(Clone, Copy, Debug)]
struct Geometry {
    rows: usize,
    cols: usize,
    tiles: usize,
    planes: usize,
    trits: usize,
    scales: usize,
    slots: usize,
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
        Ok(geometry(layout, counts)?.slots)
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
        let geometry = geometry(layout, counts)?;
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

pub(crate) struct ActiveTile {
    pub(crate) col: usize,
    pub(crate) len: usize,
    pub(crate) trit: usize,
    pub(crate) scale: usize,
}

pub(crate) struct ActiveTiles<'a> {
    index: PerTileIndex<'a>,
    plane: usize,
    row: usize,
    next: usize,
    trit: usize,
    scale: usize,
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
