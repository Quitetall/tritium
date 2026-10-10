//! Out-of-core materialization of immutable SALT V2 scale-refinement children.

use core::fmt;
use std::io::{Read, Seek, SeekFrom, Write};

use blake3::Hasher;

use super::{
    SaltV2PackageError, SaltV2PackageReadError, SaltV2PackageReader, SaltV2PackageStreamError,
    SaltV2PackageStreamPlan, SaltV2PackageStreamWriter, SaltV2Plane, SaltV2ScaleUpdate,
    SaltV2StreamTensorSpec,
};
use crate::{PackageHasher, PackageId};

const UPDATE_SET_CONTEXT: &str = "tritium salt-v2 scale update set v1";
const CHILD_LINEAGE_CONTEXT: &str = "tritium salt-v2 scale update child lineage v1";

/// Errors while materializing and independently reopening a scale-refined child.
#[derive(Debug)]
#[non_exhaustive]
pub enum SaltV2ScaleUpdateChildError {
    /// Strict parent-package read or child-package reopen failed.
    Read(SaltV2PackageReadError),
    /// Package metadata, scale geometry, or codec validation failed.
    Package(SaltV2PackageError),
    /// The canonical package stream could not be planned or written.
    Stream(SaltV2PackageStreamError),
    /// The update list is empty, duplicated, unordered, or does not target the parent.
    InvalidUpdateSet,
    /// One or more requested updates were not applied exactly once.
    UnusedUpdate,
    /// Exact child-package hashing failed.
    Io(std::io::Error),
}

impl fmt::Display for SaltV2ScaleUpdateChildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read(error) => write!(f, "scale-update child package read: {error}"),
            Self::Package(error) => write!(f, "scale-update child package: {error}"),
            Self::Stream(error) => write!(f, "scale-update child stream: {error}"),
            Self::InvalidUpdateSet => f.write_str("scale-update set is not canonical"),
            Self::UnusedUpdate => f.write_str("scale-update set contains an unapplied target"),
            Self::Io(error) => write!(f, "scale-update child package I/O: {error}"),
        }
    }
}

impl std::error::Error for SaltV2ScaleUpdateChildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read(error) => Some(error),
            Self::Package(error) => Some(error),
            Self::Stream(error) => Some(error),
            Self::Io(error) => Some(error),
            Self::InvalidUpdateSet | Self::UnusedUpdate => None,
        }
    }
}

impl From<SaltV2PackageReadError> for SaltV2ScaleUpdateChildError {
    fn from(error: SaltV2PackageReadError) -> Self {
        Self::Read(error)
    }
}

impl From<SaltV2PackageError> for SaltV2ScaleUpdateChildError {
    fn from(error: SaltV2PackageError) -> Self {
        Self::Package(error)
    }
}

impl From<SaltV2PackageStreamError> for SaltV2ScaleUpdateChildError {
    fn from(error: SaltV2PackageStreamError) -> Self {
        Self::Stream(error)
    }
}

/// Exact lineage for one independently reopened scale-refined package.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SaltV2ScaleUpdateChild {
    parent_package_id: PackageId,
    update_set_digest: [u8; 32],
    child_package_id: PackageId,
    lineage_id: [u8; 32],
}

impl SaltV2ScaleUpdateChild {
    /// Compute the canonical ordered update-set identity without writing a child.
    ///
    /// # Errors
    /// Rejects an empty, duplicate, or non-canonically ordered update list.
    pub fn update_set_digest_for(
        updates: &[SaltV2ScaleUpdate],
    ) -> Result<[u8; 32], SaltV2ScaleUpdateChildError> {
        if updates.is_empty() {
            return Err(SaltV2ScaleUpdateChildError::InvalidUpdateSet);
        }
        let mut previous = None;
        let mut hasher = Hasher::new_derive_key(UPDATE_SET_CONTEXT);
        hasher.update(&(updates.len() as u64).to_le_bytes());
        for update in updates {
            let target = (
                update.tensor_index(),
                update.tile_index(),
                update.plane_index(),
            );
            if previous.is_some_and(|value| target <= value) {
                return Err(SaltV2ScaleUpdateChildError::InvalidUpdateSet);
            }
            previous = Some(target);
            hasher.update(&(target.0 as u64).to_le_bytes());
            hasher.update(&(target.1 as u64).to_le_bytes());
            hasher.update(&(target.2 as u64).to_le_bytes());
            hasher.update(&(update.scales().len() as u64).to_le_bytes());
            for scale in update.scales() {
                hasher.update(&scale.to_bits().to_le_bytes());
            }
        }
        Ok(*hasher.finalize().as_bytes())
    }

    /// Exact transport identity of the package that was refined.
    #[must_use]
    pub const fn parent_package_id(self) -> PackageId {
        self.parent_package_id
    }

    /// Domain-separated identity of the canonical ordered update set.
    #[must_use]
    pub const fn update_set_digest(self) -> [u8; 32] {
        self.update_set_digest
    }

    /// Exact transport identity of the serialized and strictly reopened child.
    #[must_use]
    pub const fn child_package_id(self) -> PackageId {
        self.child_package_id
    }

    /// Domain-separated identity binding parent, update set, and child package.
    #[must_use]
    pub const fn lineage_id(self) -> [u8; 32] {
        self.lineage_id
    }
}

/// Stream scale updates into a new immutable package without materializing the model.
///
/// Only the current 256-coefficient tile is decoded at a time. The output must be
/// an empty seekable file or buffer. On any error it may contain a partial package;
/// callers must write to a temporary destination and publish it atomically only
/// after this function succeeds. The input reader must already have been opened
/// with strict package validation.
///
/// # Errors
/// Rejects noncanonical or inapplicable updates, invalid package data, I/O errors,
/// and any child that does not pass strict reopen validation.
pub fn write_salt_v2_scale_update_child<R, W>(
    parent: &mut SaltV2PackageReader<R>,
    output: W,
    updates: &[SaltV2ScaleUpdate],
) -> Result<(W, SaltV2ScaleUpdateChild), SaltV2ScaleUpdateChildError>
where
    R: Read + Seek,
    W: Read + Write + Seek,
{
    if updates.is_empty() {
        return Err(SaltV2ScaleUpdateChildError::InvalidUpdateSet);
    }
    let parent_package_id = parent.package_id();
    let update_set_digest = SaltV2ScaleUpdateChild::update_set_digest_for(updates)?;

    let names = parent
        .tensor_names_encoded_order()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let mut specs = Vec::new();
    let mut plane_counts = Vec::new();
    specs
        .try_reserve_exact(names.len())
        .map_err(|_| SaltV2ScaleUpdateChildError::Package(SaltV2PackageError::AllocationFailed))?;
    for name in &names {
        let info = parent
            .tensor_info(name)
            .ok_or_else(|| SaltV2PackageReadError::TensorNotFound(name.clone()))?;
        specs.push(SaltV2StreamTensorSpec::new_with_layout(
            name.clone(),
            info.dims().to_vec(),
            info.transform(),
            info.scale_group_size(),
        )?);
        plane_counts.extend(
            parent
                .tensor_plane_counts(name)?
                .map(|count| u8::try_from(count).expect("strict parser limits planes to three")),
        );
    }
    let plan = SaltV2PackageStreamPlan::new(parent.codec(), specs, plane_counts)?;
    let mut writer = SaltV2PackageStreamWriter::new(output, plan)?;
    let mut update_cursor = 0usize;

    for (tensor_index, name) in names.iter().enumerate() {
        let info = parent
            .tensor_info(name)
            .expect("tensor name came from the strict parent index");
        let scale_group_size = info.scale_group_size();
        let codec = parent.codec();
        let mut tile_planes = Vec::<SaltV2Plane>::with_capacity(3);
        let mut callback_error = None;
        parent.visit_packed_tensor(name, |packed_plane| {
            if callback_error.is_some() {
                return;
            }
            let decoded = match codec {
                super::SaltV2Codec::D2 => crate::salt_v2::unpack_d2(
                    packed_plane.packed_bytes(),
                    packed_plane.logical_len(),
                ),
                super::SaltV2Codec::B3 => crate::salt_v2::unpack_b3(
                    packed_plane.packed_bytes(),
                    packed_plane.logical_len(),
                ),
                super::SaltV2Codec::S34 => crate::salt_v2::unpack_s34(
                    packed_plane.packed_bytes(),
                    packed_plane.logical_len(),
                ),
            };
            let trits = match decoded {
                Ok(trits) => trits.into_iter().map(|trit| trit.get()).collect(),
                Err(error) => {
                    callback_error = Some(SaltV2ScaleUpdateChildError::Package(error.into()));
                    return;
                }
            };
            let key = (
                tensor_index,
                packed_plane.tile_index(),
                packed_plane.plane_index(),
            );
            let scales = match updates.get(update_cursor) {
                Some(update) => {
                    let target = (
                        update.tensor_index(),
                        update.tile_index(),
                        update.plane_index(),
                    );
                    match target.cmp(&key) {
                        core::cmp::Ordering::Less => {
                            callback_error = Some(SaltV2ScaleUpdateChildError::UnusedUpdate);
                            return;
                        }
                        core::cmp::Ordering::Equal => {
                            update_cursor += 1;
                            update.scales().to_vec()
                        }
                        core::cmp::Ordering::Greater => packed_plane.scales().to_vec(),
                    }
                }
                None => packed_plane.scales().to_vec(),
            };
            let plane =
                match SaltV2Plane::new_with_scale_group_size(trits, scales, scale_group_size) {
                    Ok(plane) => plane,
                    Err(error) => {
                        callback_error = Some(SaltV2ScaleUpdateChildError::Package(error));
                        return;
                    }
                };
            let tile_complete = tile_planes.len() + 1 == packed_plane.plane_count();
            tile_planes.push(plane);
            if tile_complete {
                if let Err(error) = writer.push_planes(&tile_planes) {
                    callback_error = Some(SaltV2ScaleUpdateChildError::Stream(error));
                }
                tile_planes.clear();
            }
        })?;
        if let Some(error) = callback_error {
            return Err(error);
        }
        if !tile_planes.is_empty() {
            return Err(SaltV2ScaleUpdateChildError::InvalidUpdateSet);
        }
    }
    if update_cursor != updates.len() {
        return Err(SaltV2ScaleUpdateChildError::UnusedUpdate);
    }
    parent.verify_unchanged()?;

    let (mut output, _) = writer.finish()?;
    output
        .seek(SeekFrom::Start(0))
        .map_err(SaltV2ScaleUpdateChildError::Io)?;
    let mut package_hasher = PackageHasher::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = output
            .read(&mut buffer)
            .map_err(SaltV2ScaleUpdateChildError::Io)?;
        if count == 0 {
            break;
        }
        package_hasher.update(&buffer[..count]);
    }
    let child_package_id = package_hasher.finalize();
    output
        .seek(SeekFrom::Start(0))
        .map_err(SaltV2ScaleUpdateChildError::Io)?;
    let reopened = SaltV2PackageReader::new_strict(&mut output)?;
    if reopened.package_id() != child_package_id {
        return Err(SaltV2ScaleUpdateChildError::InvalidUpdateSet);
    }
    drop(reopened);
    let mut lineage_hasher = Hasher::new_derive_key(CHILD_LINEAGE_CONTEXT);
    lineage_hasher.update(parent_package_id.as_bytes());
    lineage_hasher.update(&update_set_digest);
    lineage_hasher.update(child_package_id.as_bytes());
    let lineage_id = *lineage_hasher.finalize().as_bytes();
    output
        .seek(SeekFrom::End(0))
        .map_err(SaltV2ScaleUpdateChildError::Io)?;

    Ok((
        output,
        SaltV2ScaleUpdateChild {
            parent_package_id,
            update_set_digest,
            child_package_id,
            lineage_id,
        },
    ))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use half::f16;

    use super::*;
    use crate::salt_v2::SaltV2Codec;
    use crate::salt_v2_package::{SaltV2Package, SaltV2Tensor, SaltV2Tile, write_salt_v2_package};

    fn encoded_parent(codec: super::super::SaltV2Codec) -> Vec<u8> {
        let raw_trits = (0..256)
            .map(|index| if index % 4 == 1 { 0 } else { 1 })
            .collect();
        let plane = SaltV2Plane::new(raw_trits, vec![f16::ONE; 2]).unwrap();
        let tensor = SaltV2Tensor::new(
            "projection.weight",
            vec![256],
            vec![SaltV2Tile::new(vec![plane]).unwrap()],
        )
        .unwrap();
        write_salt_v2_package(&SaltV2Package::new(codec, vec![tensor]).unwrap())
            .unwrap()
            .bytes
    }

    fn ragged_encoded_parent(codec: SaltV2Codec) -> Vec<u8> {
        let make_plane = |len: usize, phase: usize| {
            let trits = (0..len)
                .map(|index| match (index + phase) % 4 {
                    0 => -1,
                    1 => 0,
                    _ => 1,
                })
                .collect();
            let scales = (0..len.div_ceil(super::super::SALT_V2_SCALE_GROUP_SIZE))
                .map(|group| f16::from_f32(0.5 + (group + phase) as f32 / 8.0))
                .collect();
            SaltV2Plane::new(trits, scales).expect("valid plane")
        };
        let make_tile = |len: usize, phase: usize, plane_count: usize| {
            SaltV2Tile::new(
                (0..plane_count)
                    .map(|plane| make_plane(len, phase + plane))
                    .collect(),
            )
            .expect("valid tile")
        };
        let tensor = SaltV2Tensor::new(
            "projection.weight",
            vec![596],
            vec![
                make_tile(256, 0, 1),
                make_tile(256, 1, 3),
                make_tile(84, 2, 2),
            ],
        )
        .expect("valid ragged tensor");
        write_salt_v2_package(&SaltV2Package::new(codec, vec![tensor]).unwrap())
            .unwrap()
            .bytes
    }

    fn packed_planes(
        reader: &mut SaltV2PackageReader<Cursor<Vec<u8>>>,
        codec: SaltV2Codec,
    ) -> Vec<(usize, usize, Vec<f16>, Vec<tritium_core::Trit>)> {
        let mut observed = Vec::new();
        reader
            .visit_packed_tensor("projection.weight", |plane| {
                let trits = match codec {
                    SaltV2Codec::D2 => {
                        crate::salt_v2::unpack_d2(plane.packed_bytes(), plane.logical_len())
                    }
                    SaltV2Codec::B3 => {
                        crate::salt_v2::unpack_b3(plane.packed_bytes(), plane.logical_len())
                    }
                    SaltV2Codec::S34 => {
                        crate::salt_v2::unpack_s34(plane.packed_bytes(), plane.logical_len())
                    }
                }
                .unwrap();
                observed.push((
                    plane.tile_index(),
                    plane.plane_index(),
                    plane.scales().to_vec(),
                    trits,
                ));
            })
            .unwrap();
        observed
    }

    #[test]
    fn scale_child_is_out_of_core_reopenable_and_preserves_trits_for_every_codec() {
        for codec in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
            let mut parent =
                SaltV2PackageReader::new_strict(Cursor::new(encoded_parent(codec))).unwrap();
            let parent_id = parent.package_id();
            let update = SaltV2ScaleUpdate::new(0, 0, 0, vec![f16::from_f32(2.0); 2]).unwrap();

            let (mut output, lineage) =
                write_salt_v2_scale_update_child(&mut parent, Cursor::new(Vec::new()), &[update])
                    .unwrap();

            assert_eq!(lineage.parent_package_id(), parent_id);
            assert_ne!(lineage.child_package_id(), parent_id);
            assert_ne!(lineage.update_set_digest(), [0; 32]);
            assert_ne!(lineage.lineage_id(), [0; 32]);
            output.seek(SeekFrom::Start(0)).unwrap();
            let mut child = SaltV2PackageReader::new_strict(output).unwrap();
            let mut observed = Vec::new();
            let mut observed_trits = Vec::new();
            child
                .visit_packed_tensor("projection.weight", |plane| {
                    observed.push((
                        plane.tile_index(),
                        plane.plane_index(),
                        plane.scales().to_vec(),
                    ));
                    observed_trits.extend(
                        match codec {
                            SaltV2Codec::D2 => {
                                crate::salt_v2::unpack_d2(plane.packed_bytes(), plane.logical_len())
                            }
                            SaltV2Codec::B3 => {
                                crate::salt_v2::unpack_b3(plane.packed_bytes(), plane.logical_len())
                            }
                            SaltV2Codec::S34 => crate::salt_v2::unpack_s34(
                                plane.packed_bytes(),
                                plane.logical_len(),
                            ),
                        }
                        .unwrap(),
                    );
                })
                .unwrap();
            assert_eq!(observed, vec![(0, 0, vec![f16::from_f32(2.0); 2])]);
            let expected_trits = (0..256)
                .map(|index| {
                    tritium_core::Trit::from_i8(if index % 4 == 1 { 0 } else { 1 }).unwrap()
                })
                .collect::<Vec<_>>();
            assert_eq!(observed_trits, expected_trits);
        }
    }

    #[test]
    fn scale_child_updates_later_plane_in_ragged_multiplane_package() {
        for codec in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
            let bytes = ragged_encoded_parent(codec);
            let mut parent = SaltV2PackageReader::new_strict(Cursor::new(bytes)).unwrap();
            let original = packed_planes(&mut parent, codec);
            let update = SaltV2ScaleUpdate::new(0, 1, 2, vec![f16::from_f32(2.0); 2]).unwrap();

            let (output, _) =
                write_salt_v2_scale_update_child(&mut parent, Cursor::new(Vec::new()), &[update])
                    .unwrap();
            let mut child = SaltV2PackageReader::new_strict(output).unwrap();
            let refined = packed_planes(&mut child, codec);

            assert_eq!(refined.len(), original.len(), "codec {codec:?}");
            for (before, after) in original.iter().zip(&refined) {
                assert_eq!((after.0, after.1), (before.0, before.1), "codec {codec:?}");
                assert_eq!(after.3, before.3, "trits changed for {codec:?}");
                if (after.0, after.1) == (1, 2) {
                    assert_eq!(after.2, vec![f16::from_f32(2.0); 2], "codec {codec:?}");
                } else {
                    assert_eq!(after.2, before.2, "untargeted scales changed for {codec:?}");
                }
            }
        }
    }

    #[test]
    fn scale_child_rejects_update_target_outside_parent_without_success_receipt() {
        let mut parent =
            SaltV2PackageReader::new_strict(Cursor::new(encoded_parent(SaltV2Codec::D2))).unwrap();
        let update = SaltV2ScaleUpdate::new(3, 0, 0, vec![f16::from_f32(2.0)]).unwrap();
        let error =
            write_salt_v2_scale_update_child(&mut parent, Cursor::new(Vec::new()), &[update])
                .unwrap_err();
        assert!(matches!(error, SaltV2ScaleUpdateChildError::UnusedUpdate));
    }
}
