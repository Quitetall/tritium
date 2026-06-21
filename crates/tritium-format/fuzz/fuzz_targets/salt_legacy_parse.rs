//! Fuzz target: `read_legacy_as_salt` must never panic / read out of bounds on
//! arbitrary bytes. It takes `(tq2_row, k)`; the first two bytes pick `k`
//! (so the fuzzer explores both the length-mismatch early-return and the
//! valid-length deep decode) and the rest is the row. The reader is total by
//! construction (exact-length check before any work, then bounds-checked block
//! decode); a crash or UB is the only failure it can surface.
//!
//! Run: `cargo +nightly fuzz run salt_legacy_parse`.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    // k from the first two bytes; the length check rejects mismatches cheaply
    // (no allocation before it), so even a large k is safe to feed.
    let k = u16::from_le_bytes([data[0], data[1]]) as usize;
    let _ = tritium_format::read_legacy_as_salt(&data[2..], k);
});
