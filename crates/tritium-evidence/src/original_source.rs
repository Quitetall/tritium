//! Candidate original-source identity shared by producers and typed receipt replay.

use tritium_schema::{
    NamedAssetDigest, OriginalSourceId, QuantPlanError, QuantSourceBinding, ScalePrecision,
    validate_original_source_inventory,
};

/// Hash a canonical inventory of original content, not fitted model semantics.
///
/// ADR0060 candidate domain/preimage: BLAKE3 derive-key context
/// `tritium.trit.original-source.v1`; length-prefixed UTF-8 architecture; tensor
/// count; each tensor's name, rank, dimensions, precision tag (F16=0, BF16=1,
/// F32=2) and exact raw-byte digest; asset count; each asset's name and exact
/// byte digest. All lengths/counts/dimensions are u64 little-endian. Inventories
/// are strictly sorted/unique; digests are fixed 32 bytes. Hashing does not
/// allocate a metadata preimage or read payloads. Producers must verify the
/// actual complete byte inventory; replay alone cannot prove source truth.
///
/// Excludes paths, container offsets/shards, fit recipe, output levels and
/// calibration data. No old model/package/receipt hash domain is changed.
/// # Errors
/// Invalid source inventory names/order/counts/geometry.
pub fn original_source_id(
    architecture: &str,
    tensors: &[QuantSourceBinding],
    assets: &[NamedAssetDigest],
) -> Result<OriginalSourceId, QuantPlanError> {
    validate_original_source_inventory(architecture, tensors, assets)?;
    let mut hash = blake3::Hasher::new_derive_key("tritium.trit.original-source.v1");
    text(&mut hash, architecture);
    number(&mut hash, tensors.len() as u64);
    for tensor in tensors {
        text(&mut hash, &tensor.source.name);
        number(&mut hash, tensor.source.shape.len() as u64);
        for &dimension in &tensor.source.shape {
            number(&mut hash, dimension);
        }
        let precision = match tensor.source.precision {
            ScalePrecision::F16 => 0,
            ScalePrecision::Bf16 => 1,
            ScalePrecision::F32 => 2,
            _ => return Err(QuantPlanError::InvalidSourceShape),
        };
        hash.update(&[precision]);
        hash.update(tensor.content.as_bytes());
    }
    number(&mut hash, assets.len() as u64);
    for asset in assets {
        text(&mut hash, &asset.name);
        hash.update(asset.content.as_bytes());
    }
    Ok(OriginalSourceId::from_bytes(*hash.finalize().as_bytes()))
}

fn number(hash: &mut blake3::Hasher, value: u64) {
    hash.update(&value.to_le_bytes());
}
fn text(hash: &mut blake3::Hasher, value: &str) {
    number(hash, value.len() as u64);
    hash.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use tritium_schema::{BlobId, QuantSourceTensor};

    fn fixture() -> (Vec<QuantSourceBinding>, Vec<NamedAssetDigest>) {
        (
            vec![QuantSourceBinding {
                source: QuantSourceTensor {
                    name: "weight".into(),
                    shape: vec![2, 3],
                    precision: ScalePrecision::F16,
                },
                content: BlobId::from_bytes(*blake3::hash(&[0; 12]).as_bytes()),
            }],
            vec![NamedAssetDigest {
                name: "config.json".into(),
                content: BlobId::from_bytes(*blake3::hash(b"{}").as_bytes()),
            }],
        )
    }

    #[test]
    fn every_original_content_interpretation_is_bound() {
        let (tensors, assets) = fixture();
        let identity = original_source_id("fixture", &tensors, &assets).unwrap();
        assert_eq!(
            identity,
            original_source_id("fixture", &tensors, &assets).unwrap()
        );
        assert_ne!(
            identity,
            original_source_id("other", &tensors, &assets).unwrap()
        );
        let mut altered = tensors.clone();
        altered[0].source.shape = vec![3, 2];
        assert_ne!(
            identity,
            original_source_id("fixture", &altered, &assets).unwrap()
        );
        altered = tensors.clone();
        altered[0].source.precision = ScalePrecision::Bf16;
        assert_ne!(
            identity,
            original_source_id("fixture", &altered, &assets).unwrap()
        );
        altered[0].source.precision = ScalePrecision::F32;
        assert_ne!(
            identity,
            original_source_id("fixture", &altered, &assets).unwrap()
        );
        altered = tensors.clone();
        altered[0].source.name = "other".into();
        assert_ne!(
            identity,
            original_source_id("fixture", &altered, &assets).unwrap()
        );
        altered = tensors.clone();
        altered[0].content = BlobId::from_bytes(*blake3::hash(&[1; 12]).as_bytes());
        assert_ne!(
            identity,
            original_source_id("fixture", &altered, &assets).unwrap()
        );
        let mut altered_assets = assets.clone();
        altered_assets[0].name = "tokenizer.json".into();
        assert_ne!(
            identity,
            original_source_id("fixture", &tensors, &altered_assets).unwrap()
        );
        altered_assets[0].name = "config.json".into();
        altered_assets[0].content = BlobId::from_bytes(*blake3::hash(b"{ }").as_bytes());
        assert_ne!(
            identity,
            original_source_id("fixture", &tensors, &altered_assets).unwrap()
        );
        assert_ne!(identity.as_bytes(), tensors[0].content.as_bytes());
    }

    #[test]
    fn rejects_ambiguous_or_invalid_inventories_before_hashing() {
        let (mut tensors, mut assets) = fixture();
        assert!(original_source_id("fixture", &[], &assets).is_err());
        assert!(original_source_id("", &tensors, &assets).is_err());
        tensors.push(tensors[0].clone());
        assert!(original_source_id("fixture", &tensors, &assets).is_err());
        tensors[1].source.name = "a".into();
        assert!(original_source_id("fixture", &tensors, &assets).is_err());
        tensors.pop();
        for shape in [vec![], vec![0], vec![u64::MAX, 2]] {
            tensors[0].source.shape = shape;
            assert!(original_source_id("fixture", &tensors, &assets).is_err());
        }
        tensors[0].source.shape = vec![2, 3];
        assets.push(assets[0].clone());
        assert!(original_source_id("fixture", &tensors, &assets).is_err());
        assets.pop();
        assets[0].name = "../config.json".into();
        assert!(original_source_id("fixture", &tensors, &assets).is_err());
    }
}
