//! Lower-case hex encoding for digests.
//!
//! Moved here (not newly written) from `stage7_evidence` when sha2 0.11 changed `Digest::finalize`
//! to return `hybrid_array::Array`, which — unlike the old `GenericArray` — does not implement
//! `LowerHex`. Every `format!("{:x}", digest)` in the crate stopped compiling, and the fix needed a
//! home that was not one module's private helper.
//!
//! **There are still three other private copies of this function** in `campaign.rs`,
//! `campaign_artifact.rs` and `salt.rs`, differing only in signature (`[u8; 32]` by value or by
//! reference). Collapsing them means touching ~65 call sites across files under active development,
//! which does not belong in a dependency bump; this at least stops the count from growing.

/// Encode `bytes` as lower-case hex.
///
/// Takes a slice rather than `[u8; 32]` so it serves any digest width, and so a `sha2` output
/// coerces directly without naming its array type.
pub(crate) fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::hex_digest;

    /// Known-answer test across the sha2 0.10 -> 0.11 bump. The upgrade changed `finalize`'s return
    /// type from `GenericArray` to `hybrid_array::Array` and removed the `io::Write` impl, so every
    /// digest call site in this crate had to change shape. None of that may alter a byte of output:
    /// these digests are written into release evidence and compared across runs.
    ///
    /// `SHA-256("abc")` is the canonical FIPS 180-4 vector.
    #[test]
    fn sha256_of_abc_matches_the_published_vector() {
        use sha2::Digest as _;
        let digest = sha2::Sha256::digest(b"abc");
        assert_eq!(
            hex_digest(&digest),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    /// The empty input is the other standard vector, and it also pins the `with_capacity(0)` path.
    #[test]
    fn sha256_of_empty_matches_the_published_vector() {
        use sha2::Digest as _;
        assert_eq!(
            hex_digest(&sha2::Sha256::digest(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    /// Streaming through `hash_reader`-style `update` calls must equal the one-shot digest, which is
    /// what makes the `io::copy` replacement safe.
    #[test]
    fn chunked_update_equals_one_shot() {
        use sha2::Digest as _;
        let mut hasher = sha2::Sha256::new();
        hasher.update(b"a");
        hasher.update(b"b");
        hasher.update(b"c");
        assert_eq!(
            hex_digest(&hasher.finalize()),
            hex_digest(&sha2::Sha256::digest(b"abc"))
        );
    }
}
