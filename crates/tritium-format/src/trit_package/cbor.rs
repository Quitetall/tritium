//! Borrowed fast path for the reference codec's untagged, primitive-key profile.
//! Tags/compound keys defer to that codec; no second manifest vocabulary exists.

use super::MAX_MANIFEST_BYTES;
use std::io::{self, Write};

#[derive(Debug, Eq, PartialEq)]
pub(super) enum ScanError {
    Invalid,
    Reference,
}
type Result<T> = std::result::Result<T, ScanError>;

pub(super) struct CappedOutput {
    pub bytes: Vec<u8>,
    pub exceeded: bool,
    limit: usize,
}
impl Default for CappedOutput {
    fn default() -> Self {
        Self::with_limit(MAX_MANIFEST_BYTES)
    }
}
impl Write for CappedOutput {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.append(bytes, self.limit)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
impl CappedOutput {
    pub(super) fn with_limit(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            exceeded: false,
            limit,
        }
    }
    fn append(&mut self, bytes: &[u8], limit: usize) -> io::Result<usize> {
        if bytes.len() > limit.saturating_sub(self.bytes.len()) {
            self.exceeded = true;
            return Err(io::Error::other("manifest byte ceiling"));
        }
        self.bytes
            .try_reserve(bytes.len())
            .map_err(io::Error::other)?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
}

struct Head {
    major: u8,
    argument: u64,
}
fn take<'a>(bytes: &'a [u8], at: &mut usize, count: usize) -> Result<&'a [u8]> {
    let end = at.checked_add(count).ok_or(ScanError::Invalid)?;
    let out = bytes.get(*at..end).ok_or(ScanError::Invalid)?;
    *at = end;
    Ok(out)
}
fn head(bytes: &[u8], at: &mut usize) -> Result<Head> {
    let initial = take(bytes, at, 1)?[0];
    let major = initial >> 5;
    let additional = initial & 31;
    if major == 6 {
        return Err(ScanError::Reference);
    }
    if major == 7 {
        return if matches!(additional, 20..=22) {
            Ok(Head {
                major,
                argument: u64::from(additional),
            })
        } else {
            Err(ScanError::Invalid)
        };
    }
    let (width, minimum) = match additional {
        0..=23 => {
            return Ok(Head {
                major,
                argument: u64::from(additional),
            });
        }
        24 => (1, 24),
        25 => (2, 256),
        26 => (4, 65536),
        27 => (8, 1u64 << 32),
        _ => return Err(ScanError::Invalid),
    };
    let argument = take(bytes, at, width)?
        .iter()
        .fold(0u64, |number, byte| (number << 8) | u64::from(*byte));
    if argument < minimum {
        return Err(ScanError::Invalid);
    }
    Ok(Head { major, argument })
}
fn count(argument: u64, remaining: usize, minimum_bytes: usize) -> Result<usize> {
    let count = usize::try_from(argument).map_err(|_| ScanError::Invalid)?;
    if count > remaining / minimum_bytes {
        return Err(ScanError::Invalid);
    }
    Ok(count)
}
fn scalar(bytes: &[u8], at: &mut usize, header: &Head) -> Result<bool> {
    match header.major {
        0 | 1 | 7 => Ok(true),
        2 | 3 => {
            let length = count(header.argument, bytes.len() - *at, 1)?;
            let payload = take(bytes, at, length)?;
            if header.major == 3 && std::str::from_utf8(payload).is_err() {
                return Err(ScanError::Invalid);
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}
fn compare_keys(left: &[u8], right: &[u8]) -> std::cmp::Ordering {
    // Identical to CanonicalValue for the admitted primitive keys, including
    // mixed kinds. The reference comparator has special tag rules, so never
    // apply this comparison to tags or compound keys.
    left.len().cmp(&right.len()).then_with(|| left.cmp(right))
}
fn key(bytes: &[u8], at: &mut usize, depth: usize) -> Result<()> {
    if depth > 64 {
        return Err(ScanError::Invalid);
    }
    let header = head(bytes, at)?;
    if scalar(bytes, at, &header)? {
        Ok(())
    } else {
        Err(ScanError::Reference)
    }
}
fn item(bytes: &[u8], at: &mut usize, depth: usize) -> Result<()> {
    if depth > 64 {
        return Err(ScanError::Invalid);
    }
    let header = head(bytes, at)?;
    if scalar(bytes, at, &header)? {
        return Ok(());
    }
    match header.major {
        4 => {
            let length = count(header.argument, bytes.len() - *at, 1)?;
            for _ in 0..length {
                item(bytes, at, depth + 1)?;
            }
        }
        5 => {
            let length = count(header.argument, bytes.len() - *at, 2)?;
            let mut previous: Option<&[u8]> = None;
            for _ in 0..length {
                let start = *at;
                key(bytes, at, depth + 1)?;
                let current = &bytes[start..*at];
                if previous.is_some_and(|prev| !compare_keys(prev, current).is_lt()) {
                    return Err(ScanError::Invalid);
                }
                previous = Some(current);
                item(bytes, at, depth + 1)?;
            }
        }
        _ => return Err(ScanError::Invalid),
    }
    Ok(())
}
pub(super) fn validate(bytes: &[u8]) -> Result<()> {
    validate_with_limit(bytes, MAX_MANIFEST_BYTES)
}
pub(super) fn validate_with_limit(bytes: &[u8], limit: usize) -> Result<()> {
    if bytes.is_empty() || bytes.len() > limit {
        return Err(ScanError::Invalid);
    }
    let mut at = 0;
    item(bytes, &mut at, 0)?;
    if at != bytes.len() {
        return Err(ScanError::Invalid);
    }
    Ok(())
}

/// Borrowed traversal of the existing canonical directory tuples. These are
/// internal views over wire ranges, not a second owned metadata vocabulary.
pub(super) struct Directory<'a> {
    bytes: &'a [u8],
    at: usize,
    remaining: usize,
}

pub(super) struct DirectoryEntry<'a> {
    pub id: &'a [u8; 32],
    pub kind: u64,
    pub aux: Option<&'a str>,
    pub offset: u64,
    pub length: u64,
}

impl<'a> Directory<'a> {
    pub(super) fn new(bytes: &'a [u8]) -> Result<Self> {
        validate_with_limit(bytes, super::MAX_DIRECTORY_BYTES)?;
        let mut at = 0;
        let root = head(bytes, &mut at)?;
        if root.major != 4 || root.argument != 2 {
            return Err(ScanError::Invalid);
        }
        if unsigned(bytes, &mut at)? != super::DIRECTORY_VERSION {
            return Err(ScanError::Invalid);
        }
        let entries = head(bytes, &mut at)?;
        if entries.major != 4 {
            return Err(ScanError::Invalid);
        }
        let remaining = count(entries.argument, bytes.len() - at, 1)?;
        Ok(Self {
            bytes,
            at,
            remaining,
        })
    }

    pub(super) fn len(&self) -> usize {
        self.remaining
    }

    pub(super) fn next(&mut self) -> Result<Option<DirectoryEntry<'a>>> {
        if self.remaining == 0 {
            return if self.at == self.bytes.len() {
                Ok(None)
            } else {
                Err(ScanError::Invalid)
            };
        }
        let fields = head(self.bytes, &mut self.at)?;
        if fields.major != 4 || fields.argument != 5 {
            return Err(ScanError::Invalid);
        }
        let id = head(self.bytes, &mut self.at)?;
        if id.major != 2 || id.argument != 32 {
            return Err(ScanError::Invalid);
        }
        let id = take(self.bytes, &mut self.at, 32)?
            .try_into()
            .map_err(|_| ScanError::Invalid)?;
        let kind = unsigned(self.bytes, &mut self.at)?;
        let aux = head(self.bytes, &mut self.at)?;
        let aux = match (aux.major, aux.argument) {
            (7, 22) => None,
            (3, length) => {
                let length = count(length, self.bytes.len() - self.at, 1)?;
                Some(
                    std::str::from_utf8(take(self.bytes, &mut self.at, length)?)
                        .map_err(|_| ScanError::Invalid)?,
                )
            }
            _ => return Err(ScanError::Invalid),
        };
        let offset = unsigned(self.bytes, &mut self.at)?;
        let length = unsigned(self.bytes, &mut self.at)?;
        self.remaining -= 1;
        Ok(Some(DirectoryEntry {
            id,
            kind,
            aux,
            offset,
            length,
        }))
    }
}

fn unsigned(bytes: &[u8], at: &mut usize) -> Result<u64> {
    let value = head(bytes, at)?;
    if value.major != 0 {
        return Err(ScanError::Invalid);
    }
    Ok(value.argument)
}

struct Entry {
    start: usize,
    key_end: usize,
    end: usize,
}
fn order(bytes: &mut [u8], at: &mut usize, depth: usize) -> Result<()> {
    if depth > 64 {
        return Err(ScanError::Invalid);
    }
    let header = head(bytes, at)?;
    if scalar(bytes, at, &header)? {
        return Ok(());
    }
    match header.major {
        4 => {
            let length = count(header.argument, bytes.len() - *at, 1)?;
            for _ in 0..length {
                order(bytes, at, depth + 1)?;
            }
        }
        5 => {
            let length = count(header.argument, bytes.len() - *at, 2)?;
            let body = *at;
            let mut entries = Vec::new();
            entries
                .try_reserve_exact(length)
                .map_err(|_| ScanError::Invalid)?;
            for _ in 0..length {
                let start = *at;
                key(bytes, at, depth + 1)?;
                let key_end = *at;
                order(bytes, at, depth + 1)?;
                entries.push(Entry {
                    start,
                    key_end,
                    end: *at,
                });
            }
            entries.sort_unstable_by(|a, b| {
                compare_keys(&bytes[a.start..a.key_end], &bytes[b.start..b.key_end])
            });
            if entries.windows(2).any(|pair| {
                bytes[pair[0].start..pair[0].key_end] == bytes[pair[1].start..pair[1].key_end]
            }) {
                return Err(ScanError::Invalid);
            }
            let mut cursor = body;
            for index in 0..entries.len() {
                let selected = &entries[index];
                let start = selected.start;
                let end = selected.end;
                let length = end - start;
                if cursor != start {
                    // Move one complete key/value entry. Value bodies are not
                    // decoded/cloned, and children have already been ordered.
                    bytes[cursor..end].rotate_right(length);
                    for entry in &mut entries[index + 1..] {
                        if entry.start >= cursor && entry.end <= start {
                            entry.start += length;
                            entry.key_end += length;
                            entry.end += length;
                        }
                    }
                }
                cursor += length;
            }
            debug_assert_eq!(cursor, *at);
        }
        _ => return Err(ScanError::Invalid),
    }
    Ok(())
}
pub(super) fn canonicalize(bytes: &mut [u8]) -> Result<()> {
    canonicalize_with_limit(bytes, MAX_MANIFEST_BYTES)
}
pub(super) fn canonicalize_with_limit(bytes: &mut [u8], limit: usize) -> Result<()> {
    if bytes.is_empty() || bytes.len() > limit {
        return Err(ScanError::Invalid);
    }
    let mut at = 0;
    order(bytes, &mut at, 0)?;
    if at != bytes.len() {
        return Err(ScanError::Invalid);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trit_package::{
        canonicalize_value, encode_cbor_value, parse_canonical_manifest,
        validate_canonical_manifest,
    };
    use ciborium::value::Value;
    use proptest::prelude::*;

    fn value() -> impl Strategy<Value = Value> {
        let leaf = prop_oneof![
            Just(Value::Null),
            any::<bool>().prop_map(Value::Bool),
            any::<u64>().prop_map(|n| Value::Integer(n.into())),
            any::<i64>().prop_map(|n| Value::Integer(n.into())),
            "[a-zéλ]{0,32}".prop_map(Value::Text),
            prop::collection::vec(any::<u8>(), 0..32).prop_map(Value::Bytes),
            Just(Value::Float(1.0)),
        ];
        leaf.prop_recursive(4, 64, 8, |inner| {
            prop_oneof![
                prop::collection::vec(inner.clone(), 0..8).prop_map(Value::Array),
                prop::collection::vec((inner.clone(), inner.clone()), 0..8).prop_map(Value::Map),
                (any::<u64>(), inner).prop_map(|(tag, item)| Value::Tag(tag, Box::new(item))),
            ]
        })
    }

    fn same_verdict(bytes: &[u8]) {
        assert_eq!(
            validate_canonical_manifest(bytes).is_ok(),
            parse_canonical_manifest(bytes).is_ok(),
            "envelope/reference disagreement for {bytes:x?}",
        );
    }

    #[test]
    fn output_ceiling_accepts_exact_and_rejects_short_without_partial_write() {
        let mut output = CappedOutput::default();
        assert_eq!(output.append(&[1, 2], 3).unwrap(), 2);
        assert!(output.append(&[3, 4], 3).is_err());
        assert!(output.exceeded);
        assert_eq!(output.bytes, [1, 2]);
        assert_eq!(output.append(&[3], 3).unwrap(), 1);
        assert_eq!(output.bytes, [1, 2, 3]);
    }

    #[test]
    fn boundary_heads_aliases_lengths_utf8_depth_and_keys_match_reference() {
        for major in 0..=5 {
            for argument in [
                0u64,
                23,
                24,
                255,
                256,
                65535,
                65536,
                u32::MAX.into(),
                u64::MAX,
            ] {
                let widths = [(24, 1), (25, 2), (26, 4), (27, 8)];
                for (additional, width) in widths {
                    let mut bytes = vec![(major << 5) | additional];
                    bytes.extend_from_slice(&argument.to_be_bytes()[8 - width..]);
                    same_verdict(&bytes);
                    for truncated in 0..bytes.len() {
                        same_verdict(&bytes[..truncated]);
                    }
                }
            }
        }
        for bytes in [
            &[0x7f, 0x61, b'a', 0xff][..], // indefinite text
            &[0x9f, 1, 2, 0xff],           // indefinite array
            &[0xbf, 1, 2, 0xff],           // indefinite map
            &[0x61, 0xff],                 // invalid UTF-8
            &[0xf7],                       // undefined is a null alias in the reference
            &[0xf8, 20],                   // nonminimal simple encoding
            &[0xc2, 0x41, 1],              // bignum alias must use the reference semantics
            &[0xa2, 0xf4, 0, 0xf5, 1],
            &[0xa2, 0x20, 0, 0, 1], // reversed mixed primitive keys
            &[0xa2, 0, 0, 0, 1],    // duplicate key
            &[0x80, 0],             // trailing item
        ] {
            same_verdict(bytes);
        }
        for depth in 63..=66 {
            let mut bytes = vec![0x81; depth];
            bytes.push(0);
            same_verdict(&bytes);
            *bytes.last_mut().unwrap() = 0x80;
            same_verdict(&bytes);
        }
        // Reference CanonicalValue uses its own tagged-key comparison. Defer
        // rather than replacing it with encoded-length ordering.
        let tagged = Value::Map(vec![
            (
                Value::Tag(1, Box::new(Value::Integer(256.into()))),
                Value::Null,
            ),
            (
                Value::Tag(2, Box::new(Value::Integer(0.into()))),
                Value::Null,
            ),
        ]);
        let bytes = encode_cbor_value(&tagged).unwrap();
        assert_eq!(validate(&bytes), Err(ScanError::Reference));
        same_verdict(&bytes);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]
        #[test]
        fn arbitrary_bytes_match_reference(bytes in prop::collection::vec(any::<u8>(), 0..1024)) {
            same_verdict(&bytes);
        }

        #[test]
        fn structured_values_and_in_place_order_match_reference(mut value in value()) {
            let original = encode_cbor_value(&value).unwrap();
            same_verdict(&original);
            let mut ordered = original.clone();
            let actual = canonicalize(&mut ordered);
            let expected = canonicalize_value(&mut value, 0)
                .and_then(|()| encode_cbor_value(&value));
            if actual.is_ok() {
                prop_assert_eq!(ordered.as_slice(), expected.as_ref().unwrap().as_slice());
                prop_assert!(validate(&ordered).is_ok());
            }
            if let Ok(bytes) = expected {
                same_verdict(&bytes);
                let mut trailing = bytes.clone();
                trailing.push(0);
                same_verdict(&trailing);
                for cut in [0, bytes.len() / 2, bytes.len().saturating_sub(1)] {
                    same_verdict(&bytes[..cut]);
                }
            }
        }
    }
}
