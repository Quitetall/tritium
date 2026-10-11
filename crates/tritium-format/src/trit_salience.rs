//! Mandatory durable B stage metadata in the unchanged `.trit` envelope.
use crate::trit_model::TritModelError;
use crate::trit_package::{canonicalize_value, encode_cbor_value, parse_canonical_manifest};
use ciborium::value::Value;
use tritium_schema::SalienceSourceStageManifest;

const PROFILE: &str = "tritium.salience-source-stage.v1";

/// Encode only the Rust-owned typed/canonical stage profile, not fit provenance.
/// # Errors
/// Invalid structure, canonical codec error or an exceeded metadata byte cap.
pub fn encode_salience_stage_manifest(
    manifest: &SalienceSourceStageManifest,
    max_bytes: usize,
) -> Result<Vec<u8>, TritModelError> {
    manifest
        .validate()
        .map_err(|e| TritModelError::Schema(e.to_string()))?;
    let mut value = Value::Array(vec![
        Value::Text(PROFILE.into()),
        Value::serialized(manifest).map_err(|e| TritModelError::Schema(e.to_string()))?,
    ]);
    canonicalize_value(&mut value, 0)?;
    let bytes = encode_cbor_value(&value)?;
    if bytes.len() > max_bytes {
        return Err(TritModelError::Schema(
            "salience stage metadata limit".into(),
        ));
    }
    Ok(bytes)
}

/// Decode only the exact mandatory profile; actual log/payload validation is separate.
/// # Errors
/// Unknown profile/version/fields, noncanonical metadata, invalid structure or cap.
pub fn decode_salience_stage_manifest(
    bytes: &[u8],
    max_bytes: usize,
) -> Result<SalienceSourceStageManifest, TritModelError> {
    if bytes.len() > max_bytes {
        return Err(TritModelError::Schema(
            "salience stage metadata limit".into(),
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
        .map_err(|e| TritModelError::Schema(e.to_string()))?;
    if encode_salience_stage_manifest(&manifest, max_bytes)? != bytes {
        return Err(TritModelError::NonCanonical);
    }
    Ok(manifest)
}
