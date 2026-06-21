//! Parser-fuzz gate (plan 0012): `read_tqidx` / `read_tqbin` must be *total* — on any byte
//! string they return `Ok` or `Err`, never panic, read out of bounds, or attempt an unbounded
//! allocation. The structured strategies feed valid-looking headers with adversarial length
//! fields (the `f32_vec`/`salt_bundle` lesson: a crafted `n_tokens`/`shard_count` must error from
//! a bounds check, not reserve gigabytes first).

use proptest::prelude::*;
use tritium_format::{read_tqbin, read_tqidx};

proptest! {
    /// Arbitrary bytes never panic either reader.
    #[test]
    fn readers_never_panic_on_arbitrary_bytes(bytes in prop::collection::vec(any::<u8>(), 0..512)) {
        let _ = read_tqbin(&bytes);
        let _ = read_tqidx(&bytes);
    }

    /// A valid `.tqbin` magic+version with an adversarial `n_tokens` and a short body must error
    /// (never OOM via `with_capacity`, never OOB).
    #[test]
    fn tqbin_crafted_header_errors_without_oom(n_tokens in any::<u64>(), body_len in 0usize..64) {
        let mut b = Vec::new();
        b.extend_from_slice(b"TQBN");
        b.push(1); // version
        b.extend_from_slice(&[0u8; 3]); // reserved
        b.extend_from_slice(&n_tokens.to_le_bytes());
        b.extend_from_slice(&vec![0u8; body_len]);
        // Either it parsed (n_tokens small enough that body covered it) or it errored — never panicked.
        if let Ok(tokens) = read_tqbin(&b) {
            prop_assert_eq!(tokens.len() as u64, n_tokens);
        }
    }

    /// A valid `.tqidx` magic+version with an adversarial `shard_count` and a short body must error
    /// without reserving unboundedly.
    #[test]
    fn tqidx_crafted_header_errors_without_oom(
        seq_len in any::<u32>(), shard_count in any::<u32>(), body_len in 0usize..64,
    ) {
        let mut b = Vec::new();
        b.extend_from_slice(b"TQIX");
        b.push(1); // version
        b.push(0); // reserved
        b.extend_from_slice(&seq_len.to_le_bytes());
        b.extend_from_slice(&shard_count.to_le_bytes());
        b.extend_from_slice(&vec![0u8; body_len]);
        let _ = read_tqidx(&b);
    }
}
