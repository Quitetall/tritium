//! Complete encoded-byte bounds before unknown payload identities are created.
use super::{
    FOOTER_BYTES, MAX_BLOBS, MAX_DIRECTORY_BYTES, MAX_MANIFEST_BYTES, TRIT_BLOB_ALIGNMENT,
    TRIT_HEADER_BYTES, TritBlobKind, TritPackageError, directory_fields, directory_value,
};
use crate::trit_model::metadata_cbor_bytes;
use ciborium::value::Value;

/// Guaranteed encoded-size interval; neither bound is an observed artifact size.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EncodedSizeBounds {
    /// Guaranteed minimum under the stated unresolved identity/deduplication scope.
    pub lower_bytes: u64,
    /// Conservative maximum; admission requires the cap to cover this value.
    pub upper_bytes: u64,
}
/// Known blob kind and encoded length, deliberately without a content identity.
#[derive(Clone, Debug)]
pub struct TritBlobSize {
    /// Actual directory kind/asset name/dense dtype.
    pub kind: TritBlobKind,
    /// Exact planned encoded payload length.
    pub length: u64,
}
/// Encoded package sizing only, not source/disk/heap/physical-memory qualification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TritPackageSizeBounds {
    /// Complete typed manifest interval, including all metadata fields.
    pub manifest: EncodedSizeBounds,
    /// Complete directory interval; unknown deduplication makes the lower bound loose.
    pub directory: EncodedSizeBounds,
    /// Complete header/alignment/payload/manifest/directory/footer interval.
    pub package: EncodedSizeBounds,
}
fn size<T: serde::Serialize>(value: &T) -> Result<u64, TritPackageError> {
    metadata_cbor_bytes(value).map_err(|_| TritPackageError::LengthOverflow)
}
fn add(a: u64, b: u64) -> Result<u64, TritPackageError> {
    a.checked_add(b).ok_or(TritPackageError::LengthOverflow)
}

/// Bound the owning .trit encoder's complete output before identities/dedup.
/// Upper bound assumes no dedup and at most alignment-1 padding per input.
/// Directory fields/kinds reuse the actual encoder. Unknown digest bytes have
/// fixed byte-string width here (unlike model manifest integer arrays).
/// Bounds do not replace actual encoding, receiving validation or source guards.
/// # Errors
/// Invalid intervals/counts, global metadata limits or checked size overflow.
pub fn trit_package_size_bounds(
    manifest: EncodedSizeBounds,
    blobs: &[TritBlobSize],
) -> Result<TritPackageSizeBounds, TritPackageError> {
    if manifest.lower_bytes == 0
        || manifest.lower_bytes > manifest.upper_bytes
        || blobs.len() > MAX_BLOBS
    {
        return Err(TritPackageError::InvalidDirectory);
    }
    if manifest.upper_bytes > MAX_MANIFEST_BYTES as u64 {
        return Err(TritPackageError::MetadataLimitExceeded);
    }
    let directory_lower = size(&directory_value(Vec::new()))?;
    let mut directory_upper = add(directory_lower, size(&(blobs.len() as u64))? - size(&0u64)?)?;
    let mut payload_sum = 0;
    let mut payload_max = 0;
    for blob in blobs {
        payload_sum = add(payload_sum, blob.length)?;
        payload_max = payload_max.max(blob.length);
        let fields = directory_fields(&blob.kind, u64::MAX, blob.length);
        let head_delta = size(&((fields.len() + 1) as u64))? - size(&(fields.len() as u64))?;
        // The source codec emits a definite byte string for a BLAKE3 digest.
        // Count its argument width through the codec, never construct an identity.
        let digest_width = core::mem::size_of::<tritium_schema::DigestBytes>() as u64;
        let id_bytes = add(size(&digest_width)?, digest_width)?;
        directory_upper = add(
            directory_upper,
            add(add(size(&Value::Array(fields))?, head_delta)?, id_bytes)?,
        )?;
    }
    if directory_upper > MAX_DIRECTORY_BYTES as u64 {
        return Err(TritPackageError::MetadataLimitExceeded);
    }
    let padding = (blobs.len() as u64)
        .checked_mul((TRIT_BLOB_ALIGNMENT - 1) as u64)
        .ok_or(TritPackageError::LengthOverflow)?;
    let upper_bytes = add(
        add(
            add(
                add(add(TRIT_HEADER_BYTES as u64, padding)?, payload_sum)?,
                manifest.upper_bytes,
            )?,
            directory_upper,
        )?,
        FOOTER_BYTES as u64,
    )?;
    // The owning writer's layout/index uses usize even with a File destination.
    usize::try_from(upper_bytes).map_err(|_| TritPackageError::LengthOverflow)?;
    let first = if blobs.is_empty() {
        TRIT_HEADER_BYTES
    } else {
        super::align_up(TRIT_HEADER_BYTES)?
    };
    let lower_bytes = add(
        add(
            add(add(first as u64, payload_max)?, manifest.lower_bytes)?,
            directory_lower,
        )?,
        FOOTER_BYTES as u64,
    )?;
    Ok(TritPackageSizeBounds {
        manifest,
        directory: EncodedSizeBounds {
            lower_bytes: directory_lower,
            upper_bytes: directory_upper,
        },
        package: EncodedSizeBounds {
            lower_bytes,
            upper_bytes,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TritBlob, read_trit_package, write_trit_package, write_trit_package_to};
    use tritium_core::DType;
    #[test]
    fn container_bounds_cover_actual_alignment_dedup_and_shared_kinds() {
        let manifest = [0xf6]; // Actual canonical null metadata; envelope-only test.
        let exact = EncodedSizeBounds {
            lower_bytes: 1,
            upper_bytes: 1,
        };
        for length in [0, 1, 23, 24, 255, 256, 65_535, 65_536] {
            for count in [0, 1, 23, 24, 256] {
                let data = vec![7u8; length];
                let blobs: Vec<_> = (0..count)
                    .map(|i| TritBlob {
                        bytes: &data,
                        kind: match i % 3 {
                            0 => TritBlobKind::AdditiveTensor,
                            1 => TritBlobKind::DenseTensor(DType::F32),
                            _ => TritBlobKind::Asset(format!("asset{i:04}.json")),
                        },
                    })
                    .collect();
                let sizes: Vec<_> = blobs
                    .iter()
                    .map(|b| TritBlobSize {
                        kind: b.kind.clone(),
                        length: b.bytes.len() as u64,
                    })
                    .collect();
                let bounds = trit_package_size_bounds(exact, &sizes).unwrap();
                let output = write_trit_package(&manifest, &blobs).unwrap();
                let actual = output.len() as u64;
                assert!(
                    bounds.package.lower_bytes <= actual && actual <= bounds.package.upper_bytes,
                    "{length}/{count}: {bounds:?} vs {actual}"
                );
                let admitted = read_trit_package(&output).unwrap();
                let directory = super::super::encode_directory(admitted.blobs())
                    .unwrap()
                    .len() as u64;
                assert!(
                    bounds.directory.lower_bytes <= directory
                        && directory <= bounds.directory.upper_bytes
                );
                let mut streamed = Vec::new();
                write_trit_package_to(
                    &manifest,
                    &blobs,
                    bounds.package.upper_bytes as usize,
                    &mut streamed,
                )
                .unwrap();
                assert_eq!(streamed, output);
            }
        }
    }
    #[test]
    fn invalid_intervals_global_caps_and_size_overflow_fail_closed() {
        for manifest in [
            EncodedSizeBounds {
                lower_bytes: 0,
                upper_bytes: 1,
            },
            EncodedSizeBounds {
                lower_bytes: 2,
                upper_bytes: 1,
            },
            EncodedSizeBounds {
                lower_bytes: 1,
                upper_bytes: MAX_MANIFEST_BYTES as u64 + 1,
            },
        ] {
            assert!(trit_package_size_bounds(manifest, &[]).is_err());
        }
        assert!(
            trit_package_size_bounds(
                EncodedSizeBounds {
                    lower_bytes: 1,
                    upper_bytes: 1
                },
                &[TritBlobSize {
                    kind: TritBlobKind::AdditiveTensor,
                    length: u64::MAX
                }]
            )
            .is_err()
        );
    }
}
