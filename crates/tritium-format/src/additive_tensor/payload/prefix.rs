//! Prefix comparison over shared sealed geometry and bounded codec windows.
use super::*;
use tritium_core::{
    PerTileActiveTile as ActiveTile, PerTileActiveTiles as ActiveTiles, PerTileIndex,
    PerTileRowSpan, UniformTiles,
};

impl AdditivePayload<'_> {
    /// Compare logical retained coefficients and exact stored scales to a parent.
    ///
    /// Uses the core logical-prefix contract, independent of physical codec and
    /// unused capacity. Both payloads were completely validated at construction.
    /// Counts/bytes stay borrowed; temporary storage is compact row spans plus
    /// two fixed codec windows, not decoded coefficient/scale masters. This does
    /// not admit direct-fit lineage, aggregate memory or backend execution.
    pub fn is_logical_prefix_of(
        &self,
        parent: &AdditivePayload<'_>,
    ) -> Result<bool, AdditiveTensorError> {
        let c = self.layout;
        let p = parent.layout;
        if c.rows != p.rows
            || c.cols != p.cols
            || c.tile != p.tile
            || c.group != p.group
            || c.law != p.law
            || c.basis != p.basis
            || self.plane_count() > parent.plane_count()
        {
            return Ok(false);
        }
        let mut child_spans = spans(self)?;
        let mut parent_spans = spans(parent)?;
        let child_index = index(self, &mut child_spans)?;
        let parent_index = index(parent, &mut parent_spans)?;
        let width = scale_width(c.law.precision)?;
        for plane in 0..self.plane_count() {
            let mut child_window = Window::new(self, plane)?;
            let mut parent_window = Window::new(parent, plane)?;
            for row in 0..c.rows as usize {
                let mut parents = parent_index.tiles(plane, row)?;
                for child in child_index.tiles(plane, row)? {
                    let child_cols = child.columns();
                    let Some(retained) =
                        parents.find(|tile| tile.columns().start >= child_cols.start)
                    else {
                        return Ok(false);
                    };
                    if retained.columns() != child_cols {
                        return Ok(false);
                    }
                    let mut remaining = child_cols.len();
                    let mut child_at = child.trit_offset() - child_window.logical_start;
                    let mut parent_at = retained.trit_offset() - parent_window.logical_start;
                    while remaining != 0 {
                        let child_data = child_window.at(child_at)?;
                        let parent_data = parent_window.at(parent_at)?;
                        let count = remaining.min(child_data.len()).min(parent_data.len());
                        if child_data[..count] != parent_data[..count] {
                            return Ok(false);
                        }
                        remaining -= count;
                        child_at += count;
                        parent_at += count;
                    }
                    let scales = if c.law.anchor == ScaleAnchor::Group {
                        child_cols.len().div_ceil(usize::from(c.group))
                    } else {
                        1
                    };
                    let child_start = child.scale_offset() * width;
                    let parent_start = retained.scale_offset() * width;
                    let len = scales * width;
                    if self.scale_bytes()[child_start..child_start + len]
                        != parent.scale_bytes()[parent_start..parent_start + len]
                    {
                        return Ok(false);
                    }
                }
            }
        }
        Ok(true)
    }
}

fn spans(payload: &AdditivePayload<'_>) -> Result<Vec<PerTileRowSpan>, AdditiveTensorError> {
    let count = match payload.allocation {
        Allocation::Uniform(_) => 0,
        Allocation::PerTile(counts) => PerTilePlan::new(payload.layout, counts)
            .map_err(AdditiveTensorError::InvalidView)?
            .index_len(),
    };
    let mut spans = reserved_payload(count)?;
    spans.resize(count, PerTileRowSpan::default());
    Ok(spans)
}

enum TileIndex<'a> {
    Uniform(UniformPlan),
    Compact(PerTileIndex<'a>),
}
enum Tiles<'a> {
    Uniform(UniformTiles),
    Compact(ActiveTiles<'a>),
}
impl Iterator for Tiles<'_> {
    type Item = ActiveTile;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Uniform(tiles) => tiles.next(),
            Self::Compact(tiles) => tiles.next(),
        }
    }
}
impl TileIndex<'_> {
    fn tiles(&self, plane: usize, row: usize) -> Result<Tiles<'_>, AdditiveTensorError> {
        match self {
            Self::Uniform(plan) => plan.tiles_in_row(plane, row).map(Tiles::Uniform),
            Self::Compact(index) => index.tiles_in_row(plane, row).map(Tiles::Compact),
        }
        .map_err(AdditiveTensorError::InvalidView)
    }
}
fn index<'a>(
    payload: &'a AdditivePayload<'_>,
    spans: &'a mut [PerTileRowSpan],
) -> Result<TileIndex<'a>, AdditiveTensorError> {
    match payload.allocation {
        Allocation::Uniform(count) => {
            UniformPlan::new(payload.layout, count).map(TileIndex::Uniform)
        }
        Allocation::PerTile(counts) => PerTilePlan::new(payload.layout, counts)
            .and_then(|plan| plan.build(spans))
            .map(TileIndex::Compact),
    }
    .map_err(AdditiveTensorError::InvalidView)
}

struct Window<'a> {
    codec: SaltV2Codec,
    data: &'a [u8],
    logical_start: usize,
    trit_count: usize,
    scratch: Vec<Trit>,
    loaded: Option<usize>,
}
impl<'a> Window<'a> {
    fn new(payload: &AdditivePayload<'a>, plane: usize) -> Result<Self, AdditiveTensorError> {
        let range = &payload.planes[plane];
        Ok(Self {
            codec: salt_codec(payload.layout.codec)?,
            data: &payload.bytes[range.offset..range.offset + range.bytes],
            logical_start: payload.planes[..plane]
                .iter()
                .map(|plane| plane.trits)
                .sum(),
            trit_count: range.trits,
            scratch: reserved_payload(SCRATCH_TRITS)?,
            loaded: None,
        })
    }
    fn at(&mut self, offset: usize) -> Result<&[Trit], AdditiveTensorError> {
        if offset >= self.trit_count {
            return Err(AdditiveTensorError::LengthOverflow);
        }
        let start = offset / SCRATCH_TRITS * SCRATCH_TRITS;
        if self.loaded != Some(start) {
            let count = (self.trit_count - start).min(SCRATCH_TRITS);
            let byte_offset = encoded_plane_len(self.codec, start)?;
            let len = encoded_plane_len(self.codec, count)?;
            let bytes = &self.data[byte_offset..byte_offset + len];
            match self.codec {
                SaltV2Codec::D2 => unpack_d2_into(bytes, count, &mut self.scratch),
                SaltV2Codec::B3 => unpack_b3_into(bytes, count, &mut self.scratch),
                SaltV2Codec::S34 => unpack_s34_into(bytes, count, &mut self.scratch),
            }
            .map_err(|error| AdditiveTensorError::Codec(shift_error(error, start, byte_offset)))?;
            self.loaded = Some(start);
        }
        Ok(&self.scratch[offset - start..])
    }
}
