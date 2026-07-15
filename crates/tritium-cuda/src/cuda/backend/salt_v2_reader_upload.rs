//! Bounded-staging SALT V2 package upload into final CUDA arenas.

use std::io::{Read, Seek};

use tritium_format::salt_v2_package::{
    PackedSaltV2PlaneRef, SALT_V2_ALLOCATION_TILE_SIZE, SALT_V2_INDEXED_RUNTIME_RANK_STRIDE_TILES,
    SaltV2PackageReader, SaltV2TensorInfo,
};

use super::*;

const MAX_UPLOAD_STAGING_BYTES: usize = 64 * 1024;

impl CudaBackend {
    /// Stream one named tensor from a strict SALT V2 package into exact CUDA arenas.
    ///
    /// Payload and scale allocations are sized from the reader's validated
    /// [`SaltV2IndexedRuntimeLedger`] before the first visit. Canonical packed
    /// planes then move through fixed 64 KiB host buffers directly into their
    /// final device ranges. The only tensor-sized host allocation is the compact
    /// two-bit map plus its u32 rank prefixes; no semantic trit or dense-weight
    /// tensor is materialized.
    ///
    /// Publication is transactional with respect to source mutation: callbacks
    /// may fill private device buffers before the reader's terminal digest check,
    /// but an error drops those buffers and no resident handle is returned.
    ///
    /// # Errors
    /// Rejects a missing or mutated tensor, non-matrix or transformed geometry,
    /// values outside the kernel's u32 ABI, internal receipt disagreement, host
    /// staging exhaustion, and CUDA allocation or transfer failures.
    pub fn upload_salt_v2_from_reader<R: Read + Seek>(
        &self,
        reader: &mut SaltV2PackageReader<R>,
        name: &str,
    ) -> Result<SaltV2ResidentTensor, BackendError> {
        let info = reader.tensor_info(name).cloned().ok_or_else(|| {
            BackendError::InvalidInput(format!("SALT V2 tensor `{name}` is absent"))
        })?;
        let codec = reader.codec();
        let geometry = validate_geometry(&info, codec)?;
        let planned = info.runtime_ledger();
        let payload_bytes = to_usize(planned.payload_bytes(), "payload bytes")?;
        let scale_bytes = to_usize(planned.scale_bytes(), "scale bytes")?;
        if !scale_bytes.is_multiple_of(core::mem::size_of::<u16>()) {
            return Err(BackendError::InvalidInput(
                "SALT V2 scale byte count is not u16-aligned".into(),
            ));
        }
        let scale_count = scale_bytes / core::mem::size_of::<u16>();
        let map_bytes = to_usize(planned.allocation_map_bytes(), "allocation map bytes")?;
        let rank_prefix_bytes = to_usize(planned.rank_prefix_bytes(), "rank-prefix bytes")?;
        if !rank_prefix_bytes.is_multiple_of(core::mem::size_of::<u32>()) {
            return Err(BackendError::InvalidInput(
                "SALT V2 rank-prefix byte count is not u32-aligned".into(),
            ));
        }
        let index_bytes = map_bytes.checked_add(rank_prefix_bytes).ok_or_else(|| {
            BackendError::InvalidInput("SALT V2 index bytes overflow usize".into())
        })?;
        let stored_map_bits = map_bytes.checked_mul(u8::BITS as usize).ok_or_else(|| {
            BackendError::InvalidInput("SALT V2 allocation-map bits overflow usize".into())
        })?;

        let mut device_payload = self
            .stream
            .alloc_zeros::<u8>(payload_bytes)
            .map_err(|error| {
                alloc_or_backend("allocate streamed SALT V2 payload", &error, payload_bytes)
            })?;
        let mut device_scales = self
            .stream
            .alloc_zeros::<u16>(scale_count)
            .map_err(|error| {
                alloc_or_backend("allocate streamed SALT V2 scales", &error, scale_bytes)
            })?;

        let mut allocation_map = Vec::<u8>::new();
        allocation_map
            .try_reserve_exact(map_bytes)
            .map_err(|_| BackendError::OutOfMemory {
                requested: map_bytes,
            })?;
        allocation_map.resize(map_bytes, 0);
        let rank_prefix_count = rank_prefix_bytes / core::mem::size_of::<u32>();
        let mut rank_prefixes = Vec::<u32>::new();
        rank_prefixes
            .try_reserve_exact(rank_prefix_count)
            .map_err(|_| BackendError::OutOfMemory {
                requested: rank_prefix_bytes,
            })?;
        let mut payload_staging = Vec::<u8>::new();
        payload_staging
            .try_reserve_exact(MAX_UPLOAD_STAGING_BYTES)
            .map_err(|_| BackendError::OutOfMemory {
                requested: MAX_UPLOAD_STAGING_BYTES,
            })?;
        let max_scale_staging = MAX_UPLOAD_STAGING_BYTES / core::mem::size_of::<u16>();
        let mut scale_staging = Vec::<u16>::new();
        scale_staging
            .try_reserve_exact(max_scale_staging)
            .map_err(|_| BackendError::OutOfMemory {
                requested: MAX_UPLOAD_STAGING_BYTES,
            })?;

        let stream = Arc::clone(&self.stream);
        let mut payload_offset = 0usize;
        let mut scale_offset = 0usize;
        let mut next_tile = 0usize;
        let mut next_plane = 0usize;
        let mut planes_before_tile = 0usize;
        let mut planes_seen = 0usize;
        let mut terminal_map_value = 0u32;
        let mut callback_error = None;

        let visit = reader.visit_packed_tensor(name, |plane| {
            if callback_error.is_some() {
                return;
            }
            let result = (|| {
                validate_visit_order(plane, next_tile, next_plane, &info)?;
                if plane.plane_index() == 0 {
                    if plane.tile_index() != 0
                        && plane
                            .tile_index()
                            .is_multiple_of(SALT_V2_INDEXED_RUNTIME_RANK_STRIDE_TILES)
                    {
                        rank_prefixes.push(to_u32(planes_before_tile, "rank prefix")?);
                    }
                    let map_code = u8::try_from(plane.plane_count() - 1).map_err(|_| {
                        BackendError::InvalidInput("SALT V2 plane count underflow".into())
                    })?;
                    let map_bit = plane.tile_index().checked_mul(2).ok_or_else(|| {
                        BackendError::InvalidInput("SALT V2 map bit offset overflows usize".into())
                    })?;
                    if map_bit < stored_map_bits {
                        allocation_map[map_bit / 8] |= map_code << (map_bit % 8);
                    } else {
                        terminal_map_value |= u32::from(map_code) << (map_bit - stored_map_bits);
                    }
                }

                stage_payload(
                    &stream,
                    &mut device_payload,
                    &mut payload_staging,
                    &mut payload_offset,
                    plane.packed_bytes(),
                )?;
                stage_scales(
                    &stream,
                    &mut device_scales,
                    &mut scale_staging,
                    &mut scale_offset,
                    plane.scales(),
                )?;
                planes_seen = planes_seen.checked_add(1).ok_or_else(|| {
                    BackendError::InvalidInput("SALT V2 present plane count overflows usize".into())
                })?;
                next_plane += 1;
                if next_plane == plane.plane_count() {
                    planes_before_tile = planes_before_tile
                        .checked_add(plane.plane_count())
                        .ok_or_else(|| {
                            BackendError::InvalidInput("SALT V2 plane rank overflows usize".into())
                        })?;
                    next_tile += 1;
                    next_plane = 0;
                }
                Ok(())
            })();
            if let Err(error) = result {
                callback_error = Some(error);
            }
        });
        if let Err(error) = visit {
            return Err(BackendError::InvalidInput(format!(
                "read SALT V2 tensor `{name}`: {error}"
            )));
        }
        if let Some(error) = callback_error {
            return Err(error);
        }
        flush_staged_u8(
            &stream,
            &mut device_payload,
            &mut payload_staging,
            &mut payload_offset,
            "upload streamed SALT V2 payload",
        )?;
        flush_staged_u16(
            &stream,
            &mut device_scales,
            &mut scale_staging,
            &mut scale_offset,
            "upload streamed SALT V2 scales",
        )?;

        let expected_planes = to_usize(planned.present_planes(), "present plane count")?;
        let completed = payload_offset == payload_bytes
            && scale_offset == scale_count
            && next_tile == info.tile_count()
            && next_plane == 0
            && planes_seen == expected_planes
            && planes_before_tile == expected_planes
            && rank_prefixes.len() == rank_prefix_count;
        if !completed {
            return Err(BackendError::InvalidInput(format!(
                "streamed SALT V2 upload disagrees with validated plan: payload={payload_offset}/{payload_bytes}, scales={scale_offset}/{scale_count}, tiles={next_tile}/{}, planes={planes_seen}/{expected_planes}, rank-prefixes={}/{}",
                info.tile_count(),
                rank_prefixes.len(),
                rank_prefix_count,
            )));
        }

        allocation_map
            .try_reserve_exact(rank_prefix_bytes)
            .map_err(|_| BackendError::OutOfMemory {
                requested: index_bytes,
            })?;
        for prefix in &rank_prefixes {
            allocation_map.extend_from_slice(&prefix.to_le_bytes());
        }
        if allocation_map.len() != index_bytes {
            return Err(BackendError::InvalidInput(
                "streamed SALT V2 index length disagrees with validated plan".into(),
            ));
        }
        let device_index = if allocation_map.is_empty() {
            None
        } else {
            Some(stream.clone_htod(&allocation_map).map_err(|error| {
                alloc_or_backend("upload streamed SALT V2 compact index", &error, index_bytes)
            })?)
        };
        let receipt = SaltV2ResidentAllocationReceipt::new(codec, planned);
        Ok(SaltV2ResidentTensor {
            payload: device_payload,
            scales: device_scales,
            index_metadata: device_index,
            rows: geometry.rows,
            columns: geometry.columns,
            tile_count: info.tile_count(),
            plane_count: expected_planes,
            codec_tag: geometry.codec_tag,
            allocation_map_bytes: to_u32(map_bytes, "allocation map bytes")?,
            rank_prefix_count: to_u32(rank_prefix_count, "rank prefix count")?,
            terminal_map_value,
            receipt,
        })
    }
}

#[derive(Clone, Copy)]
struct UploadGeometry {
    rows: usize,
    columns: usize,
    codec_tag: u32,
}

fn validate_geometry(
    info: &SaltV2TensorInfo,
    codec: SaltV2Codec,
) -> Result<UploadGeometry, BackendError> {
    if !matches!(info.transform(), SaltV2Transform::None) {
        return Err(BackendError::InvalidInput(format!(
            "SALT V2 CUDA does not implement tensor transform {:?}; only None is accepted",
            info.transform()
        )));
    }
    if info.dims().len() != 2 {
        return Err(BackendError::InvalidInput(format!(
            "SALT V2 CUDA requires rank 2, got rank {}",
            info.dims().len()
        )));
    }
    let rows = usize::try_from(info.dims()[0]).map_err(|_| {
        BackendError::InvalidInput(format!(
            "SALT V2 row dimension {} exceeds host usize",
            info.dims()[0]
        ))
    })?;
    let columns = usize::try_from(info.dims()[1]).map_err(|_| {
        BackendError::InvalidInput(format!(
            "SALT V2 column dimension {} exceeds host usize",
            info.dims()[1]
        ))
    })?;
    let coefficients = rows.checked_mul(columns).ok_or_else(|| {
        BackendError::InvalidInput("SALT V2 matrix dimension product overflows usize".into())
    })?;
    if coefficients != info.logical_coefficients() {
        return Err(BackendError::InvalidInput(format!(
            "SALT V2 shape has {coefficients} coefficients but tensor has {}",
            info.logical_coefficients()
        )));
    }
    let codec_tag = match codec {
        SaltV2Codec::D2 => 0,
        SaltV2Codec::B3 => 1,
        SaltV2Codec::S34 => 2,
        _ => {
            return Err(BackendError::InvalidInput(format!(
                "unsupported SALT V2 CUDA codec {codec:?}"
            )));
        }
    };
    to_u32(rows, "row count")?;
    to_u32(columns, "column count")?;
    to_u32(info.tile_count(), "tile count")?;
    Ok(UploadGeometry {
        rows,
        columns,
        codec_tag,
    })
}

fn validate_visit_order(
    plane: PackedSaltV2PlaneRef<'_>,
    expected_tile: usize,
    expected_plane: usize,
    info: &SaltV2TensorInfo,
) -> Result<(), BackendError> {
    if plane.tile_index() != expected_tile || plane.plane_index() != expected_plane {
        return Err(BackendError::InvalidInput(format!(
            "SALT V2 visitor order changed: got tile {} plane {}, expected tile {expected_tile} plane {expected_plane}",
            plane.tile_index(),
            plane.plane_index()
        )));
    }
    let consumed = plane
        .tile_index()
        .checked_mul(SALT_V2_ALLOCATION_TILE_SIZE)
        .ok_or_else(|| {
            BackendError::InvalidInput("SALT V2 tile coefficient offset overflows usize".into())
        })?;
    let remaining = info
        .logical_coefficients()
        .checked_sub(consumed)
        .ok_or_else(|| {
            BackendError::InvalidInput(format!(
                "SALT V2 tile {} starts past the logical tensor length",
                plane.tile_index()
            ))
        })?;
    let expected_len = remaining.min(SALT_V2_ALLOCATION_TILE_SIZE);
    if plane.logical_len() != expected_len {
        return Err(BackendError::InvalidInput(format!(
            "SALT V2 tile {} has logical length {}, expected {expected_len}",
            plane.tile_index(),
            plane.logical_len()
        )));
    }
    Ok(())
}

fn stage_payload(
    stream: &Arc<CudaStream>,
    device: &mut CudaSlice<u8>,
    staging: &mut Vec<u8>,
    offset: &mut usize,
    bytes: &[u8],
) -> Result<(), BackendError> {
    if staging
        .len()
        .checked_add(bytes.len())
        .ok_or_else(|| BackendError::InvalidInput("SALT V2 staging length overflow".into()))?
        > MAX_UPLOAD_STAGING_BYTES
    {
        flush_staged_u8(
            stream,
            device,
            staging,
            offset,
            "upload streamed SALT V2 payload",
        )?;
    }
    if bytes.len() > MAX_UPLOAD_STAGING_BYTES {
        return Err(BackendError::InvalidInput(format!(
            "one SALT V2 plane has {} payload bytes, above the staging bound",
            bytes.len()
        )));
    }
    staging.extend_from_slice(bytes);
    Ok(())
}

fn stage_scales(
    stream: &Arc<CudaStream>,
    device: &mut CudaSlice<u16>,
    staging: &mut Vec<u16>,
    offset: &mut usize,
    scales: &[half::f16],
) -> Result<(), BackendError> {
    let max_elements = MAX_UPLOAD_STAGING_BYTES / core::mem::size_of::<u16>();
    if staging
        .len()
        .checked_add(scales.len())
        .ok_or_else(|| BackendError::InvalidInput("SALT V2 scale staging overflow".into()))?
        > max_elements
    {
        flush_staged_u16(
            stream,
            device,
            staging,
            offset,
            "upload streamed SALT V2 scales",
        )?;
    }
    if scales.len() > max_elements {
        return Err(BackendError::InvalidInput(format!(
            "one SALT V2 plane has {} scales, above the staging bound",
            scales.len()
        )));
    }
    staging.extend(scales.iter().map(|scale| scale.to_bits()));
    Ok(())
}

fn flush_staged_u8(
    stream: &Arc<CudaStream>,
    device: &mut CudaSlice<u8>,
    staging: &mut Vec<u8>,
    offset: &mut usize,
    context: &str,
) -> Result<(), BackendError> {
    if staging.is_empty() {
        return Ok(());
    }
    let end = offset
        .checked_add(staging.len())
        .ok_or_else(|| BackendError::InvalidInput("SALT V2 payload offset overflow".into()))?;
    let mut destination = device.try_slice_mut(*offset..end).ok_or_else(|| {
        BackendError::InvalidInput("SALT V2 payload upload exceeds final arena".into())
    })?;
    stream
        .memcpy_htod(staging, &mut destination)
        .map_err(|error| driver_err(context, &error))?;
    *offset = end;
    staging.clear();
    Ok(())
}

fn flush_staged_u16(
    stream: &Arc<CudaStream>,
    device: &mut CudaSlice<u16>,
    staging: &mut Vec<u16>,
    offset: &mut usize,
    context: &str,
) -> Result<(), BackendError> {
    if staging.is_empty() {
        return Ok(());
    }
    let end = offset
        .checked_add(staging.len())
        .ok_or_else(|| BackendError::InvalidInput("SALT V2 scale offset overflow".into()))?;
    let mut destination = device.try_slice_mut(*offset..end).ok_or_else(|| {
        BackendError::InvalidInput("SALT V2 scale upload exceeds final arena".into())
    })?;
    stream
        .memcpy_htod(staging, &mut destination)
        .map_err(|error| driver_err(context, &error))?;
    *offset = end;
    staging.clear();
    Ok(())
}

fn to_usize(value: u64, field: &str) -> Result<usize, BackendError> {
    usize::try_from(value)
        .map_err(|_| BackendError::InvalidInput(format!("SALT V2 {field} exceeds host usize")))
}

fn to_u32(value: usize, field: &str) -> Result<u32, BackendError> {
    u32::try_from(value).map_err(|_| {
        BackendError::InvalidInput(format!("SALT V2 {field} exceeds the u32 kernel ABI"))
    })
}
