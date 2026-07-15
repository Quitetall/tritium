#![no_main]

use libfuzzer_sys::fuzz_target;
use tritium_format::salt_v2_package::read_salt_v2_package;

fuzz_target!(|data: &[u8]| {
    let _ = read_salt_v2_package(data);
});
