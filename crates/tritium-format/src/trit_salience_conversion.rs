//! Mandatory joined C metadata; decoding is not stage/log/numerical admission.
use crate::trit_model::{TritModelError, decode_model_manifest, encode_model_manifest};
use crate::trit_package::{canonicalize_value, encode_cbor_value, parse_canonical_manifest};
use ciborium::value::Value;
use tritium_schema::{
    CapturedSalienceConversionManifest, OriginalCapturedSalienceConversionManifest,
    SalienceConversionManifest,
};

const PROFILE: &str = "tritium.salience-conversion.v1";
const CAPTURED_PROFILE: &str = "tritium.captured-salience-conversion.v1";
const ORIGINAL_CAPTURED_PROFILE: &str = "tritium.original-captured-salience-conversion.v1";

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
    encode_profile(&canonical, PROFILE, max_bytes)
}

/// Decode only the exact canonical carrier; no lineage can be silently unwrapped.
/// The caller must use the common joined loader before claiming admission.
/// # Errors
/// Unknown profile/version/fields, noncanonical inventory/encoding or metadata ceiling.
pub fn decode_salience_conversion_manifest(
    bytes: &[u8],
    max_bytes: usize,
) -> Result<SalienceConversionManifest, TritModelError> {
    decode_profile(
        bytes,
        PROFILE,
        max_bytes,
        encode_salience_conversion_manifest,
    )
}

/// Encode a distinct mandatory capture-root carrier, not ordinary C or qualification.
/// # Errors
/// Invalid lineage/capture reference, codec or metadata cap.
pub fn encode_captured_salience_conversion_manifest(
    manifest: &CapturedSalienceConversionManifest,
    max_bytes: usize,
) -> Result<Vec<u8>, TritModelError> {
    let mut canonical = manifest.clone();
    canonical.lineage = decode_salience_conversion_manifest(
        &encode_salience_conversion_manifest(&canonical.lineage, max_bytes)?,
        max_bytes,
    )?;
    canonical
        .validate()
        .map_err(|e| TritModelError::Schema(e.to_string()))?;
    encode_profile(&canonical, CAPTURED_PROFILE, max_bytes)
}

/// Decode only the strict captured profile; actual joined capture admission is separate.
/// # Errors
/// Wrong profile/version/fields, canonical bytes, lineage/reference or metadata ceiling.
pub fn decode_captured_salience_conversion_manifest(
    bytes: &[u8],
    max_bytes: usize,
) -> Result<CapturedSalienceConversionManifest, TritModelError> {
    decode_profile(
        bytes,
        CAPTURED_PROFILE,
        max_bytes,
        encode_captured_salience_conversion_manifest,
    )
}

/// Encode the distinct original-A plus B capture carrier before package publication.
/// # Errors
/// Invalid inner lineage/references, codec or complete encoded metadata cap.
pub fn encode_original_captured_salience_conversion_manifest(
    manifest: &OriginalCapturedSalienceConversionManifest,
    max_bytes: usize,
) -> Result<Vec<u8>, TritModelError> {
    let mut canonical = manifest.clone();
    canonical.lineage = decode_captured_salience_conversion_manifest(
        &encode_captured_salience_conversion_manifest(&canonical.lineage, max_bytes)?,
        max_bytes,
    )?;
    canonical
        .validate()
        .map_err(|e| TritModelError::Schema(e.to_string()))?;
    encode_profile(&canonical, ORIGINAL_CAPTURED_PROFILE, max_bytes)
}

/// Decode only the strict original-captured C profile; joined admission is separate.
/// # Errors
/// Wrong profile/version/fields, noncanonical lineage/reference bytes or metadata cap.
pub fn decode_original_captured_salience_conversion_manifest(
    bytes: &[u8],
    max_bytes: usize,
) -> Result<OriginalCapturedSalienceConversionManifest, TritModelError> {
    decode_profile(
        bytes,
        ORIGINAL_CAPTURED_PROFILE,
        max_bytes,
        encode_original_captured_salience_conversion_manifest,
    )
}

fn encode_profile<T: serde::Serialize>(
    canonical: &T,
    profile: &str,
    max_bytes: usize,
) -> Result<Vec<u8>, TritModelError> {
    let mut value = Value::Array(vec![
        Value::Text(profile.into()),
        Value::serialized(canonical).map_err(|e| TritModelError::Schema(e.to_string()))?,
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

fn decode_profile<T: serde::de::DeserializeOwned>(
    bytes: &[u8],
    profile: &str,
    max_bytes: usize,
    encode: impl FnOnce(&T, usize) -> Result<Vec<u8>, TritModelError>,
) -> Result<T, TritModelError> {
    if bytes.len() > max_bytes {
        return Err(TritModelError::Schema(
            "salience conversion metadata limit".into(),
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
        .map_err(|error| TritModelError::Schema(error.to_string()))?;
    if encode(&manifest, max_bytes)? != bytes {
        return Err(TritModelError::NonCanonical);
    }
    Ok(manifest)
}
