//! Schema-derived byte bounds. No unresolved identity can enter publication.
use super::{TritModelError, metadata_cbor_bytes};
use crate::trit_package::EncodedSizeBounds;
use tritium_schema::{
    BlobReference, DigestBytes, ManifestAsset, ModelManifest, TensorMeta, TensorStorage,
};

// These are extreme byte-width cases, not content identities. There is no
// conversion to BlobId, validation, manifest encoding or publication here.
#[derive(Clone, Copy)]
enum DigestWidth {
    Minimum,
    Maximum,
}
impl serde::Serialize for DigestWidth {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let bytes: DigestBytes = match self {
            Self::Minimum => [u8::MIN; 32],
            Self::Maximum => [u8::MAX; 32],
        };
        bytes.serialize(serializer)
    }
}
fn overflow() -> TritModelError {
    crate::TritPackageError::LengthOverflow.into()
}

/// Full local tensor-record size bounds, using real storage/counts and name.
/// Consumes storage so a prepared count vector need not be cloned or retained.
/// The unknown digest's byte widths are bounded, not fabricated blob identities.
/// Does not prove structural validity, payload bits, import or model quality.
/// Sizing-only manifest views cannot enter the resolved encoder:
///
/// ```compile_fail
/// fn unresolved(manifest: &tritium_schema::ModelManifest<()>) {
///     tritium_format::trit_model::encode_model_manifest(manifest).unwrap();
/// }
/// ```
/// # Errors
/// Checked size overflow or schema serialization error.
pub fn tensor_meta_cbor_size_bounds(
    name: String,
    storage: TensorStorage,
) -> Result<EncodedSizeBounds, TritModelError> {
    let mut tensor = TensorMeta {
        name,
        storage,
        blob: BlobReference {
            id: DigestWidth::Minimum,
            shard: None,
        },
    };
    let lower_bytes = metadata_cbor_bytes(&tensor)?;
    tensor.blob.id = DigestWidth::Maximum;
    let upper_bytes = metadata_cbor_bytes(&tensor)?;
    Ok(EncodedSizeBounds {
        lower_bytes,
        upper_bytes,
    })
}

/// Full single Imported-level manifest bounds, excluding container/evidence.
/// Uses the schema's shared imported construction and exact serialized record
/// bounds. Tensor count-array bytes are not retained. Canonical ordering only
/// rearranges the admitted primitive-key profile; encoded lengths are unchanged.
/// No future content identity, deduplication, structural or empirical claim.
/// # Errors
/// Empty/excessive inventory, invalid bounds, overflow or serializer error.
pub fn imported_manifest_cbor_size_bounds(
    architecture: &str,
    level: &str,
    legacy_id: &str,
    assets: &[String],
    tensors: &[EncodedSizeBounds],
) -> Result<EncodedSizeBounds, TritModelError> {
    if tensors.is_empty() || tensors.len() > 1_000_000 || assets.len() > 4096 {
        return Err(tritium_schema::ModelManifestError::InvalidCount.into());
    }
    let mut envelope = ModelManifest::imported(
        architecture.to_owned(),
        level.to_owned(),
        legacy_id.to_owned(),
        assets
            .iter()
            .map(|name| ManifestAsset {
                name: name.clone(),
                blob: BlobReference {
                    id: DigestWidth::Minimum,
                    shard: None,
                },
            })
            .collect(),
        Vec::new(),
    );
    let mut lower_bytes = metadata_cbor_bytes(&envelope)?;
    for asset in &mut envelope.assets {
        asset.blob.id = DigestWidth::Maximum;
    }
    let mut upper_bytes = metadata_cbor_bytes(&envelope)?;
    // Definite CBOR arrays and unsigned integers share the same argument-width
    // rules. Ask the actual codec for that width, replacing the empty array head.
    let header_delta = metadata_cbor_bytes(&(tensors.len() as u64))?
        .checked_sub(metadata_cbor_bytes(&0u64)?)
        .ok_or_else(overflow)?;
    lower_bytes = lower_bytes.checked_add(header_delta).ok_or_else(overflow)?;
    upper_bytes = upper_bytes.checked_add(header_delta).ok_or_else(overflow)?;
    for size in tensors {
        if size.lower_bytes > size.upper_bytes {
            return Err(overflow());
        }
        lower_bytes = lower_bytes
            .checked_add(size.lower_bytes)
            .ok_or_else(overflow)?;
        upper_bytes = upper_bytes
            .checked_add(size.upper_bytes)
            .ok_or_else(overflow)?;
    }
    Ok(EncodedSizeBounds {
        lower_bytes,
        upper_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tritium_schema::{BlobId, ScalePrecision};
    #[test]
    fn full_manifest_bounds_cover_actual_ids_and_collection_headers() {
        for count in [1, 23, 24, 255, 256] {
            let mut tensors = Vec::new();
            let mut sizes = Vec::new();
            for i in 0..count {
                let storage = TensorStorage::Dense {
                    shape: vec![i + 1],
                    precision: ScalePrecision::F16,
                };
                let name = format!("weight{i:04}");
                sizes.push(tensor_meta_cbor_size_bounds(name.clone(), storage.clone()).unwrap());
                let payload = vec![0; (i as usize + 1) * 2];
                tensors.push(TensorMeta {
                    name,
                    storage,
                    blob: BlobReference {
                        id: BlobId::from_bytes(*blake3::hash(&payload).as_bytes()),
                        shard: None,
                    },
                });
            }
            let names = vec!["config.json".to_owned(), "tokenizer.json".to_owned()];
            let assets = names
                .iter()
                .map(|name| ManifestAsset {
                    name: name.clone(),
                    blob: BlobReference {
                        id: BlobId::from_bytes(*blake3::hash(name.as_bytes()).as_bytes()),
                        shard: None,
                    },
                })
                .collect();
            let manifest = ModelManifest::imported(
                "fixture".into(),
                "imported".into(),
                "fixture-original".into(),
                assets,
                tensors,
            );
            let actual = super::super::encode_model_manifest(&manifest)
                .unwrap()
                .len() as u64;
            let bounds = imported_manifest_cbor_size_bounds(
                "fixture",
                "imported",
                "fixture-original",
                &names,
                &sizes,
            )
            .unwrap();
            assert!(
                bounds.lower_bytes <= actual && actual <= bounds.upper_bytes,
                "{count}: {bounds:?} vs {actual}"
            );
            assert_eq!(bounds.upper_bytes - bounds.lower_bytes, 32 * (count + 2));
        }
    }
    #[test]
    fn manifest_bounds_reject_empty_invalid_and_overflowing_records() {
        for sizes in [
            vec![],
            vec![EncodedSizeBounds {
                lower_bytes: 2,
                upper_bytes: 1,
            }],
            vec![EncodedSizeBounds {
                lower_bytes: u64::MAX,
                upper_bytes: u64::MAX,
            }],
        ] {
            assert!(
                imported_manifest_cbor_size_bounds("fixture", "imported", "original", &[], &sizes)
                    .is_err()
            );
        }
    }
}
