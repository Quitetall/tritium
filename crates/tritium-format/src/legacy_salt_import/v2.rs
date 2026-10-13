//! Bounded adaptation of the existing strict flattened SALT V2 reader.
mod spool;
pub use spool::{
    SaltTensorSpoolError, SaltTensorSpoolImport, SaltTensorSpoolLimits, SaltTensorSpoolPlan,
    import_salt_v2_tensor_to_spool, inspect_salt_v2_tensor_spool,
};
use std::io::{Read, Seek};

use half::f16;
use tritium_core::{
    OwnedPerTileIndex, PerTileActiveTile, PerTileActiveTiles, PerTileIndex, PerTilePlan, Trit,
};
use tritium_schema::{
    AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, PlaneRelation, ScaleAnchor, ScalePrecision,
    Transport, admitted_execution_group,
};

use super::{SaltTensorImportError as E, SaltTensorImportLimits, import_bound};
use crate::{
    AdditiveTensor, AdditiveTensorError,
    salt_v2::{SaltV2Codec, unpack_b3_into, unpack_d2_into, unpack_s34_into},
    salt_v2_package::{
        PackedSaltV2PlaneRef, SALT_V2_MAX_PLANES, SaltV2PackageReadError, SaltV2PackageReader,
        SaltV2Transform,
    },
};

/// Import one strictly indexed SALT V2 tensor without a legacy/dense master.
///
/// Selected-tensor visits must verify before any result is returned. The model
/// migration caller must verify the whole source unchanged after its final
/// reads and before publication, and retain original shape/identity/provenance.
/// Logical payload limits do not cover the reader, capacities or whole RSS.
/// This adapter accepts explicit PerTile Group/Free/F16 G32/G128 raw D2/B3
/// output with Identity basis; source groups/codecs are preserved logically,
/// not newly admitted execution profiles. Flat source tiles must have one count
/// per target tile and exactly one scale bit pattern per target subgroup.
/// Source-byte admission counts encoded coefficient plus scale bytes, excluding
/// container metadata. Source transform None is required; a missing RHT block
/// cannot be inferred. Rank one has one row; higher ranks flatten leading axes.
///
/// # Errors
/// Rejects unsupported or nonrepresentable count/scale partitions, malformed
/// geometry, explicit limits, changed source, codec data or allocation failure.
pub fn import_salt_v2_tensor<R: Read + Seek>(
    reader: &mut SaltV2PackageReader<R>,
    name: &str,
    layout: AdditiveLayout,
    limits: SaltTensorImportLimits,
) -> Result<AdditiveTensor, E> {
    let prepared = prepare(reader, name, layout, limits)?;
    let Prepared {
        width,
        source_group,
        counts,
    } = prepared;
    let plan = PerTilePlan::new(layout, &counts).map_err(view_error)?;
    let trit_count = plan.trit_count();
    let scale_count = plan.scale_count();
    import_bound("output trits", trit_count, limits.max_trits)?;
    import_bound("output scales", scale_count, limits.max_scales)?;
    let bytes = scale_count
        .checked_mul(core::mem::size_of::<f32>())
        .and_then(|n| n.checked_add(trit_count))
        .and_then(|n| n.checked_add(plan.buffer_len_bytes().ok()?))
        .ok_or(E::Overflow)?;
    import_bound("owned payload bytes", bytes, limits.max_owned_bytes)?;

    let group = usize::from(layout.group);
    prove_scales(reader, name, width, group, source_group)?;

    // Only a fully verified representability proof permits retained output.
    let owned = OwnedPerTileIndex::new(layout, counts).map_err(view_error)?;
    let index = owned.as_index();
    let mut trits = reserved(trit_count, Trit::ZERO)?;
    let mut scales = reserved(scale_count, 0.0)?;
    let mut decoded = Vec::new();
    decoded.try_reserve_exact(256).map_err(|_| E::Allocation)?;
    let mut cursors: [Option<RowCursor<'_>>; SALT_V2_MAX_PLANES] = core::array::from_fn(|_| None);
    let codec = reader.codec();
    let mut fill = Ok(());
    let mut filled_trits = 0usize;
    reader
        .visit_packed_tensor(name, |plane| {
            if fill.is_ok() {
                fill = decode(codec, plane, &mut decoded).and_then(|()| {
                    walk_runs(plane, width, group, source_group, |run| {
                        let tile = locate(
                            index,
                            &mut cursors[plane.plane_index()],
                            plane.plane_index(),
                            run.row,
                            run.column,
                        )?;
                        let columns = tile.columns();
                        if run.column + run.len > columns.end {
                            return Err(E::Geometry);
                        }
                        let within = run.column - columns.start;
                        let offset = tile.trit_offset() + within;
                        trits[offset..offset + run.len]
                            .copy_from_slice(&decoded[run.source..run.source + run.len]);
                        scales[tile.scale_offset() + within / group] =
                            f16::from_bits(run.bits).to_f32();
                        filled_trits = filled_trits.checked_add(run.len).ok_or(E::Overflow)?;
                        Ok(())
                    })
                });
            }
        })
        .map_err(E::Read)?;
    fill?;
    if filled_trits != trit_count {
        return Err(E::Geometry);
    }
    AdditiveTensor::from_per_tile_index(owned, trits, scales).map_err(E::Tensor)
}

struct Prepared {
    width: usize,
    source_group: usize,
    counts: Vec<u8>,
}

fn prepare<R: Read + Seek>(
    reader: &SaltV2PackageReader<R>,
    name: &str,
    layout: AdditiveLayout,
    limits: SaltTensorImportLimits,
) -> Result<Prepared, E> {
    layout.validate().map_err(E::Layout)?;
    if layout.allocation != PlaneAllocation::PerTile
        || layout.law.anchor != ScaleAnchor::Group
        || layout.law.relation != PlaneRelation::Free
        || layout.law.precision != ScalePrecision::F16
        || !admitted_execution_group(layout.law, layout.group)
        || !matches!(layout.codec, PlaneCodec::D2 | PlaneCodec::B3)
        || layout.transport != Transport::Raw
        || layout.basis != Basis::Identity
    {
        return Err(E::Unsupported("lossless flattened V2 layout"));
    }
    let info = reader
        .tensor_info(name)
        .ok_or_else(|| E::Read(SaltV2PackageReadError::TensorNotFound(name.to_owned())))?;
    if info.transform() != SaltV2Transform::None {
        return Err(E::Unsupported("V2 transform lacks a bound block"));
    }
    let cols = *info.dims().last().ok_or(E::Geometry)?;
    let rows = info.dims()[..info.dims().len() - 1]
        .iter()
        .try_fold(1u64, |n, &dim| n.checked_mul(dim).ok_or(E::Overflow))?;
    if layout.rows != rows || layout.cols != cols {
        return Err(E::Geometry);
    }
    let width = usize::try_from(cols).map_err(|_| E::Overflow)?;
    let height = usize::try_from(rows).map_err(|_| E::Overflow)?;
    let source_group = info.scale_group_size();
    let source_bytes = info
        .encoded_payload_bytes()
        .checked_add(info.encoded_scale_bytes())
        .ok_or(E::Overflow)?;
    import_bound(
        "source bytes",
        usize::try_from(source_bytes).map_err(|_| E::Overflow)?,
        limits.max_source_bytes,
    )?;
    let counts = remap_counts(reader, name, width, height, layout.max_planes, limits)?;
    Ok(Prepared {
        width,
        source_group,
        counts,
    })
}

fn prove_scales<R: Read + Seek>(
    reader: &mut SaltV2PackageReader<R>,
    name: &str,
    width: usize,
    group: usize,
    source_group: usize,
) -> Result<(), E> {
    // Reader visits stage at most their existing bounded batch. This frontier
    // has the frozen legacy plane cap, not a new global execution plane cap.
    let mut frontier = [None; SALT_V2_MAX_PLANES];
    let mut proof = Ok(());
    reader
        .visit_packed_tensor(name, |plane| {
            if proof.is_ok() {
                proof = walk_runs(plane, width, group, source_group, |run| {
                    check_scale(&mut frontier[plane.plane_index()], run, width, group)
                });
            }
        })
        .map_err(E::Read)?;
    proof
}

fn view_error(error: tritium_core::AdditiveError) -> E {
    E::Tensor(AdditiveTensorError::InvalidView(error))
}

fn reserved<T: Clone>(len: usize, value: T) -> Result<Vec<T>, E> {
    let mut output = Vec::new();
    output.try_reserve_exact(len).map_err(|_| E::Allocation)?;
    output.resize(len, value);
    Ok(output)
}

fn remap_counts<R: Read + Seek>(
    reader: &SaltV2PackageReader<R>,
    name: &str,
    width: usize,
    height: usize,
    max_planes: u8,
    limits: SaltTensorImportLimits,
) -> Result<Vec<u8>, E> {
    let tiles = width.div_ceil(256);
    let len = height.checked_mul(tiles).ok_or(E::Overflow)?;
    import_bound("count metadata bytes", len, limits.max_owned_bytes)?;
    let mut counts = reserved(len, u8::MAX)?;
    let logical = height.checked_mul(width).ok_or(E::Overflow)?;
    for (tile, count) in reader
        .tensor_plane_counts(name)
        .map_err(E::Read)?
        .enumerate()
    {
        let count = u8::try_from(count).map_err(|_| E::Overflow)?;
        if count > max_planes {
            return Err(E::Geometry);
        }
        let mut position = tile.checked_mul(256).ok_or(E::Overflow)?;
        let end = position.saturating_add(256).min(logical);
        while position < end {
            let row = position / width;
            let column = position % width;
            let target = row * tiles + column / 256;
            if counts[target] != u8::MAX && counts[target] != count {
                return Err(E::Unsupported(
                    "V2 count partition is not row-tile representable",
                ));
            }
            counts[target] = count;
            let target_end = (column / 256 * 256).saturating_add(256).min(width);
            position += (target_end - column).min(end - position);
        }
    }
    Ok(counts)
}

struct Run {
    row: usize,
    column: usize,
    source: usize,
    len: usize,
    bits: u16,
}

fn walk_runs<F: From<E>>(
    plane: PackedSaltV2PlaneRef<'_>,
    width: usize,
    group: usize,
    source_group: usize,
    mut apply: impl FnMut(Run) -> Result<(), F>,
) -> Result<(), F> {
    let base = plane.tile_index().checked_mul(256).ok_or(E::Overflow)?;
    let mut source = 0;
    while source < plane.logical_len() {
        let position = base.checked_add(source).ok_or(E::Overflow)?;
        let column = position % width;
        let len = (plane.logical_len() - source)
            .min(source_group - source % source_group)
            .min(width - column)
            .min(group - column % group);
        apply(Run {
            row: position / width,
            column,
            source,
            len,
            bits: plane.scales()[source / source_group].to_bits(),
        })?;
        source += len;
    }
    Ok(())
}

fn check_scale(
    frontier: &mut Option<(usize, u16)>,
    run: Run,
    width: usize,
    group: usize,
) -> Result<(), E> {
    let key = run.row * width.div_ceil(group) + run.column / group;
    if matches!(*frontier, Some((previous, bits)) if previous == key && bits != run.bits) {
        return Err(E::Unsupported(
            "V2 scale partition is not subgroup representable",
        ));
    }
    *frontier = Some((key, run.bits));
    Ok(())
}

struct RowCursor<'a> {
    row: usize,
    tiles: PerTileActiveTiles<'a>,
    current: Option<PerTileActiveTile>,
}

fn locate<'a, 'c>(
    index: PerTileIndex<'a>,
    cursor: &'c mut Option<RowCursor<'a>>,
    plane: usize,
    row: usize,
    column: usize,
) -> Result<&'c PerTileActiveTile, E> {
    if cursor.as_ref().is_none_or(|cursor| cursor.row != row) {
        *cursor = Some(RowCursor {
            row,
            tiles: index.tiles_in_row(plane, row).map_err(view_error)?,
            current: None,
        });
    }
    let cursor = cursor.as_mut().ok_or(E::Geometry)?;
    while cursor
        .current
        .as_ref()
        .is_none_or(|tile| tile.columns().end <= column)
    {
        cursor.current = Some(cursor.tiles.next().ok_or(E::Geometry)?);
    }
    let tile = cursor.current.as_ref().ok_or(E::Geometry)?;
    if column < tile.columns().start {
        return Err(E::Geometry);
    }
    Ok(tile)
}

fn decode(
    codec: SaltV2Codec,
    plane: PackedSaltV2PlaneRef<'_>,
    trits: &mut Vec<Trit>,
) -> Result<(), E> {
    match codec {
        SaltV2Codec::D2 => unpack_d2_into(plane.packed_bytes(), plane.logical_len(), trits),
        SaltV2Codec::B3 => unpack_b3_into(plane.packed_bytes(), plane.logical_len(), trits),
        SaltV2Codec::S34 => unpack_s34_into(plane.packed_bytes(), plane.logical_len(), trits),
    }
    .map_err(|error| E::Tensor(AdditiveTensorError::Codec(error)))
}
