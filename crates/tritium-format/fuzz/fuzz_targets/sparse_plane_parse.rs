//! Fuzz target: `unpack_sparse_plane` must never panic / read out of bounds on
//! arbitrary bytes. The reader is total by construction (exact-length +
//! checked arithmetic + per-entry bounds checks); this harness feeds it raw
//! input and discards the result — a crash or UB is the only failure it surfaces.
//!
//! Run: `cargo +nightly fuzz run sparse_plane_parse`.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = tritium_format::unpack_sparse_plane(data);
});
