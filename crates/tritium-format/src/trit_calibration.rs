//! Typed candidate cache manifest inside the unchanged `.trit` envelope.
use crate::trit_model::TritModelError;
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
    crate::trit_profile::encode(PROFILE, m, MAX_CALIBRATION_MANIFEST_BYTES)
}

/// Decode only the exact typed/canonical profile; claimed payload identities stay untrusted.
/// # Errors
/// Unknown profile/fields, duplicate/noncanonical metadata, invalid counts or limit.
pub fn decode_calibration_cache_manifest(
    bytes: &[u8],
) -> Result<CalibrationCacheManifest, TritModelError> {
    let m: CalibrationCacheManifest =
        crate::trit_profile::decode(PROFILE, bytes, MAX_CALIBRATION_MANIFEST_BYTES)?;
    if encode_calibration_cache_manifest(&m)? != bytes {
        return Err(TritModelError::NonCanonical);
    }
    Ok(m)
}
