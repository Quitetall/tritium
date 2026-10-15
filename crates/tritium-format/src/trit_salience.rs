//! Mandatory durable B stage metadata in the unchanged `.trit` envelope.
use crate::trit_model::TritModelError;
use crate::trit_package::{canonicalize_value, encode_cbor_value, parse_canonical_manifest};
use ciborium::value::Value;
use tritium_schema::{OriginalCapturedSalienceSourceStageManifest, SalienceSourceStageManifest};

const PROFILE: &str = "tritium.salience-source-stage.v1";
const ORIGINAL_CAPTURED_PROFILE: &str = "tritium.original-captured-salience-source-stage.v1";

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
    encode_profile(manifest, PROFILE, max_bytes)
}

/// Encode only the distinct mandatory original-capture stage profile.
/// # Errors
/// Invalid strict lineage/reference, canonical codec or metadata cap.
pub fn encode_original_captured_salience_stage_manifest(
    manifest: &OriginalCapturedSalienceSourceStageManifest,
    max_bytes: usize,
) -> Result<Vec<u8>, TritModelError> {
    manifest
        .validate()
        .map_err(|e| TritModelError::Schema(e.to_string()))?;
    encode_profile(manifest, ORIGINAL_CAPTURED_PROFILE, max_bytes)
}

fn encode_profile<T: serde::Serialize>(
    manifest: &T,
    profile: &str,
    max_bytes: usize,
) -> Result<Vec<u8>, TritModelError> {
    let mut value = Value::Array(vec![
        Value::Text(profile.into()),
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
    decode_profile(bytes, PROFILE, max_bytes, encode_salience_stage_manifest)
}

/// Decode the exact original-captured profile; both complete logs are still required.
/// # Errors
/// Wrong profile/version/fields, noncanonical bytes, references or metadata cap.
pub fn decode_original_captured_salience_stage_manifest(
    bytes: &[u8],
    max_bytes: usize,
) -> Result<OriginalCapturedSalienceSourceStageManifest, TritModelError> {
    decode_profile(
        bytes,
        ORIGINAL_CAPTURED_PROFILE,
        max_bytes,
        encode_original_captured_salience_stage_manifest,
    )
}

fn decode_profile<T: serde::de::DeserializeOwned>(
    bytes: &[u8],
    profile: &str,
    max_bytes: usize,
    encode: impl FnOnce(&T, usize) -> Result<Vec<u8>, TritModelError>,
) -> Result<T, TritModelError> {
    if bytes.len() > max_bytes {
        return Err(TritModelError::Schema(
            "salience stage metadata limit".into(),
        ));
    }
    let Value::Array(items) = parse_canonical_manifest(bytes)? else {
        return Err(TritModelError::NonCanonical);
    };
    if items.len() != 2 || items[0].as_text() != Some(profile) {
        return Err(TritModelError::NonCanonical);
    }
    let manifest = items[1]
        .deserialized()
        .map_err(|e| TritModelError::Schema(e.to_string()))?;
    if encode(&manifest, max_bytes)? != bytes {
        return Err(TritModelError::NonCanonical);
    }
    Ok(manifest)
}
