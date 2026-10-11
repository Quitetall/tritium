//! Mandatory joined C metadata; decoding is not stage/log/numerical admission.
use crate::trit_model::{TritModelError, decode_model_manifest, encode_model_manifest};
use crate::trit_package::{canonicalize_value, encode_cbor_value, parse_canonical_manifest};
use ciborium::value::Value;
use tritium_schema::SalienceConversionManifest;

const PROFILE: &str = "tritium.salience-conversion.v1";

/// Encode the mandatory carrier using the same Rust model canonicalizer.
/// Does not establish actual stage/log identity or numerical truth.
/// # Errors
/// Invalid structure/version, codec failure or exceeded encoded metadata ceiling.
pub fn encode_salience_conversion_manifest(
    manifest: &SalienceConversionManifest,
    max_bytes: usize,
) -> Result<Vec<u8>, TritModelError> {
    let mut canonical = manifest.clone();
    canonical.model = decode_model_manifest(&encode_model_manifest(&canonical.model)?)?;
    canonical
        .validate()
        .map_err(|error| TritModelError::Schema(error.to_string()))?;
    let mut value = Value::Array(vec![
        Value::Text(PROFILE.into()),
        Value::serialized(&canonical).map_err(|error| TritModelError::Schema(error.to_string()))?,
    ]);
    canonicalize_value(&mut value, 0)?;
    let bytes = encode_cbor_value(&value)?;
    if bytes.len() > max_bytes {
        return Err(TritModelError::Schema(
            "salience conversion metadata limit".into(),
        ));
    }
    Ok(bytes)
}

/// Decode only the exact canonical carrier; no lineage can be silently unwrapped.
/// The caller must use the common joined loader before claiming admission.
/// # Errors
/// Unknown profile/version/fields, noncanonical inventory/encoding or metadata ceiling.
pub fn decode_salience_conversion_manifest(
    bytes: &[u8],
    max_bytes: usize,
) -> Result<SalienceConversionManifest, TritModelError> {
    if bytes.len() > max_bytes {
        return Err(TritModelError::Schema(
            "salience conversion metadata limit".into(),
        ));
    }
    let Value::Array(items) = parse_canonical_manifest(bytes)? else {
        return Err(TritModelError::NonCanonical);
    };
    if items.len() != 2 || items[0].as_text() != Some(PROFILE) {
        return Err(TritModelError::NonCanonical);
    }
    let manifest = items[1]
        .deserialized()
        .map_err(|error| TritModelError::Schema(error.to_string()))?;
    if encode_salience_conversion_manifest(&manifest, max_bytes)? != bytes {
        return Err(TritModelError::NonCanonical);
    }
    Ok(manifest)
}
