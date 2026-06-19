//! Fuzz target: `read_salt_gguf` must never panic / read out of bounds on
//! arbitrary bytes. It layers the SALT-in-GGUF row walk on the (total) GGUF
//! reader; this target proves the added layer is total too — a crash or UB is
//! the only failure mode it can surface.
//!
//! Run: `cargo +nightly fuzz run salt_gguf_parse`.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = tritium_format::read_salt_gguf(data);
});
