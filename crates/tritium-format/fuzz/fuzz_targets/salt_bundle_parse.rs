//! Fuzz target: borrowed and seek-backed bundle readers must never panic / read
//! out of bounds on arbitrary bytes. The readers are total by construction
//! (LE-framed, bounds-checked, alloc-capped); this harness feeds both raw input.
//!
//! Run: `cargo +nightly fuzz run salt_bundle_parse`.
#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Cursor;
use tritium_format::{
    SaltBundleIndex, SaltBundleReadError, SaltBundleReader, unpack_packed_salt_row,
};

fuzz_target!(|data: &[u8]| {
    let _ = tritium_format::read_salt_bundle(data);
    let borrowed = SaltBundleIndex::new(data);
    let seek = SaltBundleReader::new_strict(Cursor::new(data));
    match (borrowed, seek) {
        (Ok(borrowed), Ok(mut seek)) => {
            let mut names = borrowed.tensor_names().collect::<Vec<_>>();
            names.sort_unstable();
            assert_eq!(names, seek.tensor_names().collect::<Vec<_>>());
            for name in names {
                let expected = borrowed.tensor(name).unwrap().decode_packed().unwrap();
                assert_eq!(
                    seek.tensor_info(name).unwrap().shape(),
                    (expected.rows, expected.k)
                );
                let mut actual = Vec::new();
                seek.visit_packed_tensor(name, |row| {
                    actual.push(unpack_packed_salt_row(row.encoded_bytes()).unwrap());
                })
                .unwrap();
                assert_eq!(actual, expected.salt_rows);
            }
        }
        (Err(_), Ok(_)) => panic!("seek reader accepted a bundle rejected by borrowed parsing"),
        (Ok(_), Err(SaltBundleReadError::LimitExceeded { .. })) => {}
        (Ok(_), Err(SaltBundleReadError::AllocationFailed { .. })) => {}
        (Ok(_), Err(error)) => panic!("seek reader rejected borrowed-valid bundle: {error}"),
        (Err(_), Err(_)) => {}
    }
});
