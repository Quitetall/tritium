//! Typed owner/window cache metadata in the unchanged common .trit envelope.
use crate::trit_model::TritModelError;
use tritium_schema::{OwnerScoreCacheContent, OwnerScoreCacheManifest};

/// Separate metadata cap, not a numerical or process-memory qualification.
pub const MAX_OWNER_SCORE_MANIFEST_BYTES: usize = 1024 * 1024;
const PROFILE: &str = "tritium.owner-score-cache.v1";
const CONTENT: &str = "tritium.owner-score-content.v1";

/// Canonical logical identity preimage; excludes producer/receiver ceilings.
pub fn encode_owner_score_content(
    content: &OwnerScoreCacheContent,
) -> Result<Vec<u8>, TritModelError> {
    crate::trit_profile::encode(CONTENT, content, MAX_OWNER_SCORE_MANIFEST_BYTES)
}
/// Encode structurally admitted metadata; numerical/content truth is separate.
pub fn encode_owner_score_manifest(
    manifest: &OwnerScoreCacheManifest,
) -> Result<Vec<u8>, TritModelError> {
    if !manifest.valid(manifest.limits) {
        return Err(TritModelError::Schema(
            "owner cache manifest rejected".into(),
        ));
    }
    crate::trit_profile::encode(PROFILE, manifest, MAX_OWNER_SCORE_MANIFEST_BYTES)
}
/// Decode only exact canonical profile/fields and producer structural admission.
/// The cache receiver must recompute every content identity and receiving bound.
pub fn decode_owner_score_manifest(
    bytes: &[u8],
) -> Result<OwnerScoreCacheManifest, TritModelError> {
    let manifest: OwnerScoreCacheManifest =
        crate::trit_profile::decode(PROFILE, bytes, MAX_OWNER_SCORE_MANIFEST_BYTES)?;
    if !manifest.valid(manifest.limits) {
        return Err(TritModelError::Schema(
            "owner cache manifest rejected".into(),
        ));
    }
    Ok(manifest)
}
