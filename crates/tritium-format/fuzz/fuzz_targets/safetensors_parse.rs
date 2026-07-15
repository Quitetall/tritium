//! Fuzz target: borrowed and seek-backed safetensors parsing plus one selected
//! tensor read must never panic or read out of bounds on arbitrary bytes.
//!
//! Run: `cargo +nightly fuzz run safetensors_parse`.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let borrowed = tritium_format::SafeTensors::parse(data);
    let seek = tritium_format::SafeTensorsReader::new(std::io::Cursor::new(data));
    if let (Ok(borrowed), Ok(mut seek)) = (borrowed, seek) {
        let borrowed_names = borrowed.names().map(str::to_owned).collect::<Vec<_>>();
        let seek_names = seek.names().map(str::to_owned).collect::<Vec<_>>();
        assert_eq!(seek_names, borrowed_names);
        if let Some(name) = borrowed_names.first() {
            assert_eq!(seek.shape(name), borrowed.shape(name));
            assert_eq!(seek.dtype(name), borrowed.dtype(name));
            match (seek.tensor_f32(name), borrowed.tensor_f32(name)) {
                (Ok(seek_values), Ok(borrowed_values)) => {
                    assert_eq!(seek_values.len(), borrowed_values.len());
                    for (seek_value, borrowed_value) in seek_values.iter().zip(&borrowed_values) {
                        assert_eq!(seek_value.to_bits(), borrowed_value.to_bits());
                    }
                }
                (Err(seek_error), Err(borrowed_error)) => {
                    assert_eq!(seek_error, borrowed_error);
                }
                (seek_result, borrowed_result) => {
                    panic!(
                        "reader disagreement: seek={seek_result:?}, borrowed={borrowed_result:?}"
                    );
                }
            }
        }
    }
});
