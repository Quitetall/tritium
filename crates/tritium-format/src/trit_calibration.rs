//! Typed candidate cache manifest inside the unchanged `.trit` envelope.
use crate::trit_model::TritModelError;
use crate::trit_package::{canonicalize_value, encode_cbor_value, parse_canonical_manifest};
use ciborium::value::Value;
use tritium_schema::CalibrationCacheManifest;

/// Metadata cap, separate from numerical payload or aggregate process memory.
pub const MAX_CALIBRATION_MANIFEST_BYTES: usize = 1024 * 1024;
const PROFILE: &str = "tritium.calibration-cache.v1";

/// Encode the Rust-owned candidate manifest with the common canonical CBOR codec.
/// # Errors
/// Invalid structural counts/admission, unsupported metadata or metadata limit.
pub fn encode_calibration_cache_manifest(
    m: &CalibrationCacheManifest,
) -> Result<Vec<u8>, TritModelError> {
    m.validate(m.binding.request.limits)
        .map_err(|e| TritModelError::Schema(format!("{e:?}")))?;
    let mut value = Value::Array(vec![
        Value::Text(PROFILE.into()),
        Value::serialized(m).map_err(|e| TritModelError::Schema(e.to_string()))?,
    ]);
    canonicalize_value(&mut value, 0)?;
    let bytes = encode_cbor_value(&value)?;
    if bytes.len() > MAX_CALIBRATION_MANIFEST_BYTES {
        return Err(TritModelError::Schema("calibration metadata limit".into()));
    }
    Ok(bytes)
}

/// Decode only the exact typed/canonical profile; claimed payload identities stay untrusted.
/// # Errors
/// Unknown profile/fields, duplicate/noncanonical metadata, invalid counts or limit.
pub fn decode_calibration_cache_manifest(
    bytes: &[u8],
) -> Result<CalibrationCacheManifest, TritModelError> {
    if bytes.len() > MAX_CALIBRATION_MANIFEST_BYTES {
        return Err(TritModelError::Schema("calibration metadata limit".into()));
    }
    let value = parse_canonical_manifest(bytes)?;
    let Value::Array(items) = value else {
        return Err(TritModelError::NonCanonical);
    };
    if items.len() != 2 || items[0].as_text() != Some(PROFILE) {
        return Err(TritModelError::NonCanonical);
    }
    let m: CalibrationCacheManifest = items[1]
        .deserialized()
        .map_err(|e| TritModelError::Schema(e.to_string()))?;
    if encode_calibration_cache_manifest(&m)? != bytes {
        return Err(TritModelError::NonCanonical);
    }
    Ok(m)
}
