//! Prefix comparison over shared sealed geometry and bounded range windows.
use super::*;
use tritium_core::{
    PerTileActiveTile as ActiveTile, PerTileActiveTiles as ActiveTiles, PerTileIndex,
    PerTileRowSpan, UniformTiles,
};

impl AdditivePayload<'_> {
    /// Compare retained logical coefficients and exact stored scales to a parent.
    /// Both payloads are completely validated; uses shared geometry and two fixed
    /// codec windows, not decoded masters or physical flat-buffer prefixes.
    pub fn is_logical_prefix_of(
        &self,
        parent: &AdditivePayload<'_>,
    ) -> Result<bool, AdditiveTensorError> {
        self.plan.is_logical_prefix_of(
            0,
            &parent.plan,
            1,
            &mut SlicePayloads::<AdditiveTensorError, 2>::new([self.bytes, parent.bytes]),
        )
    }
}

impl PayloadPlan<'_> {
    /// Plans must have undergone complete payload validation at the model seam.
    pub(crate) fn is_logical_prefix_of<S: PayloadSource>(
        &self,
        child_blob: usize,
        parent: &PayloadPlan<'_>,
        parent_blob: usize,
        source: &mut S,
    ) -> Result<bool, S::Error> {
        let c = self.layout;
        let p = parent.layout;
        if c.rows != p.rows
            || c.cols != p.cols
            || c.tile != p.tile
            || c.group != p.group
            || c.law != p.law
            || c.basis != p.basis
            || self.planes.len() > parent.planes.len()
        {
            return Ok(false);
        }
        let mut child_spans = spans(self)?;
        let mut parent_spans = spans(parent)?;
        let child_index = index(self, &mut child_spans)?;
        let parent_index = index(parent, &mut parent_spans)?;
        let width = scale_width(c.law.precision)?;
        let mut child_scales = [0u8; 256 * 4];
        let mut parent_scales = [0u8; 256 * 4];
        for plane in 0..self.planes.len() {
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
                        let child_data = child_window.at(child_at, child_blob, source)?;
                        let parent_data = parent_window.at(parent_at, parent_blob, source)?;
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
                    let len = scales * width;
                    let child_data = child_scales
                        .get_mut(..len)
                        .ok_or(AdditiveTensorError::LengthOverflow)?;
                    let parent_data = parent_scales
                        .get_mut(..len)
                        .ok_or(AdditiveTensorError::LengthOverflow)?;
                    source.read_range(
                        child_blob,
                        self.scale_offset + child.scale_offset() * width,
                        child_data,
                    )?;
                    source.read_range(
                        parent_blob,
                        parent.scale_offset + retained.scale_offset() * width,
                        parent_data,
                    )?;
                    if child_data != parent_data {
                        return Ok(false);
                    }
                }
            }
        }
        Ok(true)
    }
}

fn spans(payload: &PayloadPlan<'_>) -> Result<Vec<PerTileRowSpan>, AdditiveTensorError> {
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
    payload: &'a PayloadPlan<'_>,
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

struct Window {
    codec: SaltV2Codec,
    offset: usize,
    logical_start: usize,
    trit_count: usize,
    scratch: Vec<Trit>,
    loaded: Option<usize>,
}
impl Window {
    fn new(payload: &PayloadPlan<'_>, plane: usize) -> Result<Self, AdditiveTensorError> {
        let range = &payload.planes[plane];
        Ok(Self {
            codec: salt_codec(payload.layout.codec)?,
            offset: range.offset,
            logical_start: payload.planes[..plane]
                .iter()
                .map(|plane| plane.trits)
                .sum(),
            trit_count: range.trits,
            scratch: reserved_payload(SCRATCH_TRITS)?,
            loaded: None,
        })
    }
    fn at<S: PayloadSource>(
        &mut self,
        offset: usize,
        blob: usize,
        source: &mut S,
    ) -> Result<&[Trit], S::Error> {
        if offset >= self.trit_count {
            return Err(AdditiveTensorError::LengthOverflow.into());
        }
        let start = offset / SCRATCH_TRITS * SCRATCH_TRITS;
        if self.loaded != Some(start) {
            let count = (self.trit_count - start).min(SCRATCH_TRITS);
            let byte_offset = encoded_plane_len(self.codec, start)?;
            let len = encoded_plane_len(self.codec, count)?;
            let mut bytes = [0u8; SCRATCH_TRITS];
            source.read_range(blob, self.offset + byte_offset, &mut bytes[..len])?;
            decode_window(
                self.codec,
                &bytes[..len],
                count,
                &mut self.scratch,
                start,
                byte_offset,
            )?;
            self.loaded = Some(start);
        }
        Ok(&self.scratch[offset - start..])
    }
}
