//! Fuzz target: `read_tqidx` must never panic / read out of bounds on arbitrary
//! bytes. The index reader is total by construction (LE-framed header +
//! bounds-checked shard table); this harness feeds it raw input and discards the
//! result — a crash or UB is the only failure it can surface.
//!
//! Run: `cargo +nightly fuzz run tqidx_parse`.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = tritium_format::read_tqidx(data);
});
