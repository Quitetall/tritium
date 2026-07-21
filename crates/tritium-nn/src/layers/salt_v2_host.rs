//! Descriptor-free host execution for one indexed SALT V2 matrix.

use core::mem::size_of;
use std::io::{Read, Seek};

use half::f16;
use tritium_core::Trit;
use tritium_format::{
    salt_v2::{S34_TRITS_PER_GROUP, SaltV2Codec},
    salt_v2_package::{
        PackedSaltV2PlaneRef, SALT_V2_ALLOCATION_TILE_SIZE,
        SALT_V2_INDEXED_RUNTIME_RANK_STRIDE_TILES, SALT_V2_MAX_PLANES, SALT_V2_SCALE_GROUP_SIZE,
        SaltV2PackageReader, SaltV2TensorInfo, SaltV2Transform, unpack_salt_v2_plane_into,
    },
};

use crate::NnError;

/// Compact codec-resident SALT V2 matrix for portable host execution.
///
/// The representation mirrors the CUDA indexed layout: one packed payload arena,
/// one f16 scale arena, a two-bit plane-count map, and one u32 plane-rank prefix
/// per 256 allocation tiles. It deliberately retains no dense shadow and no
/// per-tile or per-plane heap descriptors.
#[derive(Clone, Debug)]
pub struct HostSaltV2Linear {
    codec: SaltV2Codec,
    rows: usize,
    columns: usize,
    logical_coefficients: usize,
    tile_count: usize,
    plane_count: usize,
    payload: Box<[u8]>,
    scales: Box<[f16]>,
    allocation_map: Box<[u8]>,
    rank_prefixes: Box<[u32]>,
    terminal_map_value: u32,
}

impl HostSaltV2Linear {
    /// Stream one named matrix from a strict seek-backed package into final host arenas.
    ///
    /// Package bytes are revalidated through the reader before the value is
    /// published. Allocation sizes come from the reader's exact runtime ledger;
    /// at most one plane is borrowed from bounded reader staging at a time.
    ///
    /// # Errors
    /// Rejects a missing tensor, non-matrix or transformed geometry, malformed or
    /// mutated package bytes, an internal ledger disagreement, and host allocation
    /// exhaustion.
    pub fn from_reader<R: Read + Seek>(
        reader: &mut SaltV2PackageReader<R>,
        name: &str,
    ) -> Result<Self, NnError> {
        let info = reader
            .tensor_info(name)
            .cloned()
            .ok_or_else(|| NnError::MissingTensor(name.to_owned()))?;
        let (rows, columns) = validate_geometry(name, &info)?;
        let codec = reader.codec();
        let planned = info.runtime_ledger();
        let payload_bytes = planned_len(planned.payload_bytes(), "payload")?;
        let scale_bytes = planned_len(planned.scale_bytes(), "scale")?;
        if !scale_bytes.is_multiple_of(size_of::<f16>()) {
            return Err(invalid(name, "scale ledger is not f16-aligned"));
        }
        let scale_count = scale_bytes / size_of::<f16>();
        let map_bytes = planned_len(planned.allocation_map_bytes(), "allocation-map")?;
        let rank_bytes = planned_len(planned.rank_prefix_bytes(), "rank-prefix")?;
        if !rank_bytes.is_multiple_of(size_of::<u32>()) {
            return Err(invalid(name, "rank-prefix ledger is not u32-aligned"));
        }
        let rank_count = rank_bytes / size_of::<u32>();

        let mut payload = reserved_vec(payload_bytes, "payload")?;
        let mut scales = reserved_vec(scale_count, "scales")?;
        let mut allocation_map = reserved_vec(map_bytes, "allocation map")?;
        allocation_map.resize(map_bytes, 0);
        let mut rank_prefixes = reserved_vec(rank_count, "rank prefixes")?;
        let stored_map_bits = map_bytes
            .checked_mul(u8::BITS as usize)
            .ok_or_else(|| invalid(name, "allocation-map bit count overflows host usize"))?;

        let mut next_tile = 0usize;
        let mut next_plane = 0usize;
        let mut planes_before_tile = 0usize;
        let mut terminal_map_value = 0u32;
        let mut callback_error = None;
        reader
            .visit_packed_tensor(name, |plane| {
                if callback_error.is_some() {
                    return;
                }
                let result = stage_plane(
                    plane,
                    &info,
                    &mut next_tile,
                    &mut next_plane,
                    &mut planes_before_tile,
                    &mut payload,
                    &mut scales,
                    &mut allocation_map,
                    stored_map_bits,
                    &mut terminal_map_value,
                    &mut rank_prefixes,
                );
                if let Err(error) = result {
                    callback_error = Some(error);
                }
            })
            .map_err(|error| invalid(name, &format!("package visit failed: {error}")))?;
        if let Some(error) = callback_error {
            return Err(error);
        }
        let expected_planes = planned_len(planned.present_planes(), "present-plane count")?;
        if payload.len() != payload_bytes
            || scales.len() != scale_count
            || next_tile != info.tile_count()
            || next_plane != 0
            || planes_before_tile != expected_planes
            || rank_prefixes.len() != rank_count
        {
            return Err(invalid(
                name,
                &format!(
                    "streamed arenas disagree with validated ledger: payload={}/{payload_bytes}, scales={}/{scale_count}, tiles={}/{}, planes={planes_before_tile}/{expected_planes}, rank-prefixes={}/{rank_count}",
                    payload.len(),
                    scales.len(),
                    next_tile,
                    info.tile_count(),
                    rank_prefixes.len(),
                ),
            ));
        }

        Ok(Self {
            codec,
            rows,
            columns,
            logical_coefficients: info.logical_coefficients(),
            tile_count: info.tile_count(),
            plane_count: expected_planes,
            payload: payload.into_boxed_slice(),
            scales: scales.into_boxed_slice(),
            allocation_map: allocation_map.into_boxed_slice(),
            rank_prefixes: rank_prefixes.into_boxed_slice(),
            terminal_map_value,
        })
    }

    /// Output rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Input columns.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Number of physically present additive planes.
    #[must_use]
    pub const fn plane_count(&self) -> usize {
        self.plane_count
    }

    /// Physical codec used by every packed plane.
    #[must_use]
    pub const fn codec(&self) -> SaltV2Codec {
        self.codec
    }

    /// Descriptor-free packed payload arena.
    #[must_use]
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    /// Descriptor-free f16 group-scale arena.
    #[must_use]
    pub fn scales(&self) -> &[f16] {
        &self.scales
    }

    /// Complete bytes of the two-bit plane-count map.
    #[must_use]
    pub fn allocation_map(&self) -> &[u8] {
        &self.allocation_map
    }

    /// Cumulative plane ranks at 256-tile boundaries.
    #[must_use]
    pub fn rank_prefixes(&self) -> &[u32] {
        &self.rank_prefixes
    }

    /// Remaining zero-padded plane-count map bits carried outside the arena.
    #[must_use]
    pub const fn terminal_map_value(&self) -> u32 {
        self.terminal_map_value
    }

    /// Exact requested payload, scale, map, and rank-prefix bytes.
    ///
    /// Allocator bookkeeping and this fixed-size handle are excluded. No dense
    /// shadow or per-plane descriptor allocation exists.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.payload
            .len()
            .saturating_add(self.scales.len().saturating_mul(size_of::<f16>()))
            .saturating_add(self.allocation_map.len())
            .saturating_add(self.rank_prefixes.len().saturating_mul(size_of::<u32>()))
    }

    /// Portable fp32 activation contraction without a dense weight shadow.
    ///
    /// Every output follows the canonical row, group128, plane, then column
    /// reduction order. One reusable 256-trit buffer is the only heap scratch.
    ///
    /// # Errors
    /// Returns [`NnError::Shape`] for incompatible buffers and
    /// [`NnError::Backend`] for non-finite arithmetic or impossible indexed
    /// metadata.
    pub fn forward(
        &self,
        activation: &[f32],
        batch: usize,
        output: &mut [f32],
    ) -> Result<(), NnError> {
        let input_len = checked_product(batch, self.columns, activation.len())?;
        let output_len = checked_product(batch, self.rows, output.len())?;
        require_len(activation.len(), input_len)?;
        require_len(output.len(), output_len)?;
        if let Some((index, value)) = activation
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(NnError::Backend(format!(
                "host SALT V2 activation {index} is non-finite ({:#010x})",
                value.to_bits()
            )));
        }

        let mut decoded = decode_scratch()?;
        for batch_index in 0..batch {
            let input = &activation[batch_index * self.columns..(batch_index + 1) * self.columns];
            let output = &mut output[batch_index * self.rows..(batch_index + 1) * self.rows];
            for (row, slot) in output.iter_mut().enumerate() {
                *slot = self.dot_row(input, row, &mut decoded)?;
            }
        }
        Ok(())
    }

    /// Reconstruct selected matrix rows directly into caller storage.
    ///
    /// This is the token-embedding gather path. It shares the same packed arenas
    /// as projection/unembedding and never materializes the full dense matrix.
    ///
    /// # Errors
    /// Returns [`NnError::Shape`] for a wrong output length,
    /// [`NnError::MissingTensor`] for an out-of-range row, and
    /// [`NnError::Backend`] for impossible indexed metadata.
    pub fn gather_rows(&self, rows: &[u32], output: &mut [f32]) -> Result<(), NnError> {
        let expected = checked_product(rows.len(), self.columns, output.len())?;
        require_len(output.len(), expected)?;
        if let Some(row) = rows
            .iter()
            .copied()
            .map(u64::from)
            .find(|&row| row >= self.rows as u64)
        {
            return Err(NnError::MissingTensor(format!("token_embd row {row}")));
        }
        output.fill(0.0);
        let mut decoded = decode_scratch()?;
        for (destination, &row) in output.chunks_mut(self.columns).zip(rows) {
            self.reconstruct_row(row as usize, destination, &mut decoded)?;
        }
        Ok(())
    }

    fn dot_row(
        &self,
        activation: &[f32],
        row: usize,
        decoded: &mut Vec<Trit>,
    ) -> Result<f32, NnError> {
        let row_base = row
            .checked_mul(self.columns)
            .ok_or_else(|| internal("row offset overflows host usize"))?;
        let row_end = row_base
            .checked_add(self.columns)
            .ok_or_else(|| internal("row end overflows host usize"))?;
        let mut accumulator = 0.0f32;
        let mut coefficient = row_base;
        while coefficient < row_end {
            let tile_index = coefficient / SALT_V2_ALLOCATION_TILE_SIZE;
            let tile_base = tile_index * SALT_V2_ALLOCATION_TILE_SIZE;
            let local_start = coefficient - tile_base;
            let logical_len = self.tile_logical_len(tile_index)?;
            let group_index = local_start / SALT_V2_SCALE_GROUP_SIZE;
            let group_end = ((group_index + 1) * SALT_V2_SCALE_GROUP_SIZE).min(logical_len);
            let segment_len = (group_end - local_start).min(row_end - coefficient);
            let segment_end = coefficient + segment_len;
            let terminal_column = segment_end - row_base - 1;
            let plane_count = self.tile_plane_count(tile_index)?;
            for plane_index in 0..plane_count {
                let (packed, scales) = self.plane(tile_index, plane_index, logical_len)?;
                unpack_salt_v2_plane_into(self.codec, packed, logical_len, decoded)
                    .map_err(|error| internal(&format!("decode indexed plane: {error}")))?;
                let scale = scales
                    .get(group_index)
                    .ok_or_else(|| internal("indexed plane scale is absent"))?
                    .to_f32();
                let mut group_accumulator = 0.0f32;
                for current in coefficient..segment_end {
                    let column = current - row_base;
                    match decoded[current - tile_base].get() {
                        -1 => group_accumulator -= activation[column],
                        0 => {}
                        1 => group_accumulator += activation[column],
                        _ => return Err(internal("decoded trit is outside {-1,0,1}")),
                    }
                    if !group_accumulator.is_finite() {
                        return Err(non_finite(row, column));
                    }
                }
                let contribution = group_accumulator * scale;
                if !contribution.is_finite() {
                    return Err(non_finite(row, terminal_column));
                }
                accumulator += contribution;
                if !accumulator.is_finite() {
                    return Err(non_finite(row, terminal_column));
                }
            }
            coefficient = segment_end;
        }
        Ok(accumulator)
    }

    fn reconstruct_row(
        &self,
        row: usize,
        output: &mut [f32],
        decoded: &mut Vec<Trit>,
    ) -> Result<(), NnError> {
        let row_base = row
            .checked_mul(self.columns)
            .ok_or_else(|| internal("row offset overflows host usize"))?;
        let row_end = row_base + self.columns;
        let mut coefficient = row_base;
        while coefficient < row_end {
            let tile_index = coefficient / SALT_V2_ALLOCATION_TILE_SIZE;
            let tile_base = tile_index * SALT_V2_ALLOCATION_TILE_SIZE;
            let local_start = coefficient - tile_base;
            let logical_len = self.tile_logical_len(tile_index)?;
            let segment_len = (logical_len - local_start).min(row_end - coefficient);
            let plane_count = self.tile_plane_count(tile_index)?;
            for plane_index in 0..plane_count {
                let (packed, scales) = self.plane(tile_index, plane_index, logical_len)?;
                unpack_salt_v2_plane_into(self.codec, packed, logical_len, decoded)
                    .map_err(|error| internal(&format!("decode indexed plane: {error}")))?;
                for local in local_start..local_start + segment_len {
                    let scale = scales[local / SALT_V2_SCALE_GROUP_SIZE].to_f32();
                    output[coefficient - row_base + local - local_start] +=
                        decoded[local].to_f32() * scale;
                }
            }
            coefficient += segment_len;
        }
        Ok(())
    }

    fn plane(
        &self,
        tile_index: usize,
        plane_index: usize,
        logical_len: usize,
    ) -> Result<(&[u8], &[f16]), NnError> {
        let rank = self.plane_rank_before(tile_index)?;
        let full_payload_bytes = stored_payload_bytes(self.codec, SALT_V2_ALLOCATION_TILE_SIZE)?;
        let payload_bytes = stored_payload_bytes(self.codec, logical_len)?;
        let payload_start = rank
            .checked_mul(full_payload_bytes)
            .and_then(|offset| offset.checked_add(plane_index.checked_mul(payload_bytes)?))
            .ok_or_else(|| internal("payload offset overflows host usize"))?;
        let scale_count = logical_len.div_ceil(SALT_V2_SCALE_GROUP_SIZE);
        let scale_start = rank
            .checked_mul(SALT_V2_ALLOCATION_TILE_SIZE / SALT_V2_SCALE_GROUP_SIZE)
            .and_then(|offset| offset.checked_add(plane_index.checked_mul(scale_count)?))
            .ok_or_else(|| internal("scale offset overflows host usize"))?;
        let packed = self
            .payload
            .get(payload_start..payload_start + payload_bytes)
            .ok_or_else(|| internal("indexed payload range is absent"))?;
        let scales = self
            .scales
            .get(scale_start..scale_start + scale_count)
            .ok_or_else(|| internal("indexed scale range is absent"))?;
        Ok((packed, scales))
    }

    fn tile_logical_len(&self, tile_index: usize) -> Result<usize, NnError> {
        let start = tile_index
            .checked_mul(SALT_V2_ALLOCATION_TILE_SIZE)
            .ok_or_else(|| internal("tile offset overflows host usize"))?;
        let remaining = self
            .logical_coefficients
            .checked_sub(start)
            .ok_or_else(|| internal("tile starts after logical tensor"))?;
        Ok(remaining.min(SALT_V2_ALLOCATION_TILE_SIZE))
    }

    fn tile_plane_count(&self, tile_index: usize) -> Result<usize, NnError> {
        if tile_index >= self.tile_count {
            return Err(internal("tile index exceeds allocation map"));
        }
        let bit = tile_index
            .checked_mul(2)
            .ok_or_else(|| internal("allocation-map offset overflows host usize"))?;
        let stored_bits = self.allocation_map.len() * u8::BITS as usize;
        let code = if bit < stored_bits {
            (self.allocation_map[bit / 8] >> (bit % 8)) & 0b11
        } else {
            ((self.terminal_map_value >> (bit - stored_bits)) & 0b11) as u8
        };
        let count = usize::from(code) + 1;
        if count > SALT_V2_MAX_PLANES {
            return Err(internal("allocation map encodes too many planes"));
        }
        Ok(count)
    }

    fn plane_rank_before(&self, tile_index: usize) -> Result<usize, NnError> {
        let block = tile_index / SALT_V2_INDEXED_RUNTIME_RANK_STRIDE_TILES;
        let block_start = block * SALT_V2_INDEXED_RUNTIME_RANK_STRIDE_TILES;
        let mut rank = if block == 0 {
            0usize
        } else {
            usize::try_from(
                *self
                    .rank_prefixes
                    .get(block - 1)
                    .ok_or_else(|| internal("rank prefix is absent"))?,
            )
            .map_err(|_| internal("rank prefix exceeds host usize"))?
        };
        for current in block_start..tile_index {
            rank = rank
                .checked_add(self.tile_plane_count(current)?)
                .ok_or_else(|| internal("plane rank overflows host usize"))?;
        }
        Ok(rank)
    }
}

#[allow(clippy::too_many_arguments)]
fn stage_plane(
    plane: PackedSaltV2PlaneRef<'_>,
    info: &SaltV2TensorInfo,
    next_tile: &mut usize,
    next_plane: &mut usize,
    planes_before_tile: &mut usize,
    payload: &mut Vec<u8>,
    scales: &mut Vec<f16>,
    allocation_map: &mut [u8],
    stored_map_bits: usize,
    terminal_map_value: &mut u32,
    rank_prefixes: &mut Vec<u32>,
) -> Result<(), NnError> {
    if plane.tile_index() != *next_tile || plane.plane_index() != *next_plane {
        return Err(internal("package visitor changed canonical plane order"));
    }
    let tile_start = plane
        .tile_index()
        .checked_mul(SALT_V2_ALLOCATION_TILE_SIZE)
        .ok_or_else(|| internal("visited tile offset overflows host usize"))?;
    let expected_len = info
        .logical_coefficients()
        .checked_sub(tile_start)
        .ok_or_else(|| internal("visited tile starts after tensor"))?
        .min(SALT_V2_ALLOCATION_TILE_SIZE);
    if plane.logical_len() != expected_len
        || !(1..=SALT_V2_MAX_PLANES).contains(&plane.plane_count())
    {
        return Err(internal(
            "visited plane geometry disagrees with tensor catalog",
        ));
    }
    if plane.plane_index() == 0 {
        if plane.tile_index() != 0
            && plane
                .tile_index()
                .is_multiple_of(SALT_V2_INDEXED_RUNTIME_RANK_STRIDE_TILES)
        {
            rank_prefixes.push(
                u32::try_from(*planes_before_tile)
                    .map_err(|_| internal("plane rank exceeds indexed u32 contract"))?,
            );
        }
        let map_code = u8::try_from(plane.plane_count() - 1)
            .map_err(|_| internal("plane count underflows map code"))?;
        let map_bit = plane
            .tile_index()
            .checked_mul(2)
            .ok_or_else(|| internal("allocation-map bit offset overflows host usize"))?;
        if map_bit < stored_map_bits {
            allocation_map[map_bit / 8] |= map_code << (map_bit % 8);
        } else {
            *terminal_map_value |= u32::from(map_code) << (map_bit - stored_map_bits);
        }
    }
    payload.extend_from_slice(plane.packed_bytes());
    scales.extend_from_slice(plane.scales());
    *next_plane += 1;
    if *next_plane == plane.plane_count() {
        *planes_before_tile = planes_before_tile
            .checked_add(plane.plane_count())
            .ok_or_else(|| internal("present-plane count overflows host usize"))?;
        *next_tile += 1;
        *next_plane = 0;
    }
    Ok(())
}

fn validate_geometry(name: &str, info: &SaltV2TensorInfo) -> Result<(usize, usize), NnError> {
    if info.transform() != SaltV2Transform::None {
        return Err(invalid(
            name,
            &format!(
                "unsupported transform {:?}; expected None",
                info.transform()
            ),
        ));
    }
    let [row_dim, column_dim] = info.dims() else {
        return Err(invalid(
            name,
            &format!("matrix rank must be 2, got {}", info.dims().len()),
        ));
    };
    let rows =
        usize::try_from(*row_dim).map_err(|_| invalid(name, "row dimension exceeds usize"))?;
    let columns = usize::try_from(*column_dim)
        .map_err(|_| invalid(name, "column dimension exceeds usize"))?;
    let coefficients = rows
        .checked_mul(columns)
        .ok_or_else(|| invalid(name, "matrix dimension product overflows usize"))?;
    if rows == 0 || columns == 0 || coefficients != info.logical_coefficients() {
        return Err(invalid(
            name,
            "matrix dimensions disagree with coefficient count",
        ));
    }
    Ok((rows, columns))
}

fn stored_payload_bytes(codec: SaltV2Codec, logical_len: usize) -> Result<usize, NnError> {
    let stored_len = if codec == SaltV2Codec::S34 {
        logical_len.div_ceil(S34_TRITS_PER_GROUP) * S34_TRITS_PER_GROUP
    } else {
        logical_len
    };
    codec
        .ledger(stored_len)
        .map(|ledger| ledger.physical_bytes)
        .map_err(|error| internal(&format!("codec ledger failed: {error}")))
}

fn reserved_vec<T>(elements: usize, description: &str) -> Result<Vec<T>, NnError> {
    let mut values = Vec::new();
    values.try_reserve_exact(elements).map_err(|_| {
        NnError::ResourceExhausted(format!(
            "allocate {elements} elements for host SALT V2 {description}"
        ))
    })?;
    Ok(values)
}

fn decode_scratch() -> Result<Vec<Trit>, NnError> {
    reserved_vec(SALT_V2_ALLOCATION_TILE_SIZE, "decode scratch")
}

fn planned_len(value: u64, description: &str) -> Result<usize, NnError> {
    usize::try_from(value).map_err(|_| {
        NnError::ResourceExhausted(format!(
            "host SALT V2 {description} length {value} exceeds host usize"
        ))
    })
}

fn checked_product(left: usize, right: usize, got: usize) -> Result<usize, NnError> {
    left.checked_mul(right).ok_or(NnError::Shape {
        expected: usize::MAX,
        got,
    })
}

fn require_len(got: usize, expected: usize) -> Result<(), NnError> {
    if got != expected {
        return Err(NnError::Shape { expected, got });
    }
    Ok(())
}

fn invalid(name: &str, reason: &str) -> NnError {
    NnError::InvalidArtifact(format!("SALT V2 tensor `{name}`: {reason}"))
}

fn internal(reason: &str) -> NnError {
    NnError::Backend(format!("host SALT V2 indexed layout: {reason}"))
}

fn non_finite(row: usize, column: usize) -> NnError {
    NnError::Backend(format!(
        "host SALT V2 output row {row} became non-finite at column {column}"
    ))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use half::f16;
    use tritium_cpu::salt_v2::{salt_v2_coefficient, salt_v2_matvec_into};
    use tritium_format::{
        salt_v2::SaltV2Codec,
        salt_v2_package::{
            SaltV2Package, SaltV2PackageReader, SaltV2Plane, SaltV2Tensor, SaltV2Tile,
            SaltV2Transform, write_salt_v2_package,
        },
    };

    use super::HostSaltV2Linear;

    fn plane(len: usize, seed: usize, scale: f32) -> SaltV2Plane {
        SaltV2Plane::new(
            (0..len)
                .map(|index| match (index + seed) % 4 {
                    0 => 0,
                    1 => -1,
                    _ => 1,
                })
                .collect(),
            (0..len.div_ceil(128))
                .map(|group| f16::from_f32(scale + group as f32 * 0.125))
                .collect(),
        )
        .unwrap()
    }

    fn resident(codec: SaltV2Codec, tensor: &SaltV2Tensor) -> (SaltV2Package, HostSaltV2Linear) {
        let package = SaltV2Package::new(codec, vec![tensor.clone()]).unwrap();
        let encoded = write_salt_v2_package(&package).unwrap();
        let mut reader = SaltV2PackageReader::new_strict(Cursor::new(encoded.bytes)).unwrap();
        let linear = HostSaltV2Linear::from_reader(&mut reader, tensor.name()).unwrap();
        (package, linear)
    }

    #[test]
    fn compact_host_projection_is_bit_exact_across_rows_groups_and_tile_boundary() {
        let tensor = SaltV2Tensor::new(
            "weight",
            vec![2, 150],
            vec![
                SaltV2Tile::new(vec![plane(256, 0, 0.5), plane(256, 1, 0.25)]).unwrap(),
                SaltV2Tile::new(vec![plane(44, 2, 0.75), plane(44, 3, 0.375)]).unwrap(),
            ],
        )
        .unwrap();
        let activation = (0..300)
            .map(|index| index as f32 * 0.0078125 - 0.5)
            .collect::<Vec<_>>();
        for codec in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
            let (package, linear) = resident(codec, &tensor);
            let mut output = vec![0.0; 4];
            linear.forward(&activation, 2, &mut output).unwrap();
            for batch in 0..2 {
                let mut expected = [0.0; 2];
                salt_v2_matvec_into(
                    &package,
                    0,
                    &activation[batch * 150..(batch + 1) * 150],
                    &mut expected,
                )
                .unwrap();
                assert_eq!(&output[batch * 2..(batch + 1) * 2], &expected);
            }
            let mut gathered = vec![0.0; 300];
            linear.gather_rows(&[1, 0], &mut gathered).unwrap();
            for (destination, source_row) in [1_usize, 0].into_iter().enumerate() {
                for column in 0..150 {
                    assert_eq!(
                        gathered[destination * 150 + column],
                        salt_v2_coefficient(&tensor, source_row * 150 + column).unwrap()
                    );
                }
            }
            assert_eq!(linear.plane_count(), 4);
            assert!(linear.resident_bytes() < tensor.logical_coefficients() * size_of::<f32>());
        }
    }

    #[test]
    fn compact_index_and_gather_cross_the_rank_prefix_boundary() {
        let mut tiles = Vec::new();
        for tile in 0..257 {
            let count = tile % 3 + 1;
            tiles.push(
                SaltV2Tile::new(
                    (0..count)
                        .map(|plane_index| plane(256, tile + plane_index, 0.125))
                        .collect(),
                )
                .unwrap(),
            );
        }
        let tensor = SaltV2Tensor::new("wide", vec![257, 256], tiles).unwrap();
        let (_package, linear) = resident(SaltV2Codec::D2, &tensor);
        let mut gathered = vec![0.0; 512];
        linear.gather_rows(&[256, 0], &mut gathered).unwrap();
        assert_eq!(linear.rank_prefixes.len(), 1);
        assert_eq!(
            linear.plane_count(),
            (0..257_usize).map(|tile| tile % 3 + 1).sum::<usize>()
        );
        for (destination, source_row) in [256_usize, 0].into_iter().enumerate() {
            for column in 0..256 {
                assert_eq!(
                    gathered[destination * 256 + column],
                    salt_v2_coefficient(&tensor, source_row * 256 + column).unwrap()
                );
            }
        }
    }

    #[test]
    fn transformed_tensor_fails_closed() {
        let tensor = SaltV2Tensor::new_with_transform(
            "rotated",
            vec![2, 2],
            SaltV2Transform::SignedRht {
                seed: 17,
                domain: 23,
            },
            vec![SaltV2Tile::new(vec![plane(4, 0, 0.5)]).unwrap()],
        )
        .unwrap();
        let package = SaltV2Package::new(SaltV2Codec::B3, vec![tensor]).unwrap();
        let encoded = write_salt_v2_package(&package).unwrap();
        let mut reader = SaltV2PackageReader::new_strict(Cursor::new(encoded.bytes)).unwrap();
        let error = HostSaltV2Linear::from_reader(&mut reader, "rotated").unwrap_err();
        assert!(matches!(error, crate::NnError::InvalidArtifact(_)));
        assert!(error.to_string().contains("unsupported transform"));
    }
}
