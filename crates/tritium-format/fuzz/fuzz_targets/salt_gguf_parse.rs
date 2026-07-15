//! Fuzz target: eager and strict seek-backed SALT-GGUF readers must never panic
//! or read out of bounds on arbitrary bytes.
//!
//! Run: `cargo +nightly fuzz run salt_gguf_parse`.
#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Cursor;

fuzz_target!(|data: &[u8]| {
    let _ = tritium_format::read_salt_gguf(data);
    if let Ok(mut reader) = tritium_format::SaltGgufReader::new_strict(Cursor::new(data)) {
        let names = reader.tensor_names().map(str::to_owned).collect::<Vec<_>>();
        for name in names {
            let _ = reader.visit_packed_tensor(&name, |_| {});
        }
    }
});
