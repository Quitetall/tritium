//! One common canonical typed metadata profile codec.
use crate::trit_model::TritModelError;
use crate::trit_package::{canonicalize_value, encode_cbor_value, parse_canonical_manifest};
use ciborium::value::Value;

pub(crate) fn encode<T: serde::Serialize>(
    profile: &str,
    value: &T,
    cap: usize,
) -> Result<Vec<u8>, TritModelError> {
    let mut value = Value::Array(vec![
        Value::Text(profile.into()),
        Value::serialized(value).map_err(|e| TritModelError::Schema(e.to_string()))?,
    ]);
    canonicalize_value(&mut value, 0)?;
    let bytes = encode_cbor_value(&value)?;
    if bytes.len() > cap {
        return Err(TritModelError::Schema(
            "typed profile metadata limit".into(),
        ));
    }
    Ok(bytes)
}

pub(crate) fn decode<T: serde::Serialize + serde::de::DeserializeOwned>(
    profile: &str,
    bytes: &[u8],
    cap: usize,
) -> Result<T, TritModelError> {
    if bytes.len() > cap {
        return Err(TritModelError::Schema(
            "typed profile metadata limit".into(),
        ));
    }
    let value = parse_canonical_manifest(bytes)?;
    let Value::Array(items) = value else {
        return Err(TritModelError::NonCanonical);
    };
    if items.len() != 2 || items[0].as_text() != Some(profile) {
        return Err(TritModelError::NonCanonical);
    }
    let value: T = items[1]
        .deserialized()
        .map_err(|e| TritModelError::Schema(e.to_string()))?;
    if encode(profile, &value, cap)? != bytes {
        return Err(TritModelError::NonCanonical);
    }
    Ok(value)
}
