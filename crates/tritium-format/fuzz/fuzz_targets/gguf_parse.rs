//! Fuzz target: `read_gguf` must never panic / read out of bounds on arbitrary
//! bytes. The reader is total by construction (bounds-checked cursor), so the
//! libfuzzer harness simply feeds it raw input and discards the result; a crash
//! or UB is the only failure mode this target can surface.
//!
//! Run (needs `cargo install cargo-fuzz` and a nightly toolchain):
//! `cargo +nightly fuzz run gguf_parse`.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Result intentionally ignored: we only care that this never panics or
    // triggers UB. Any Ok(GgufFile) is also exercised so its fields are computed.
    let _ = tritium_format::read_gguf(data);
});
