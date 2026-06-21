//! Fuzz target: `read_salt_bundle` must never panic / read out of bounds on
//! arbitrary bytes. The bundle reader is total by construction (LE-framed,
//! bounds-checked, alloc-capped); this harness feeds it raw input and discards
//! the result — a crash or UB is the only failure it can surface.
//!
//! Run: `cargo +nightly fuzz run salt_bundle_parse`.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = tritium_format::read_salt_bundle(data);
});
