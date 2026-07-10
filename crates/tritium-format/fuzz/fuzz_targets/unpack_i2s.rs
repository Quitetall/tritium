//! I2_S tensor decode must never panic: first 4 bytes pick n_elements
//! (bounded), the rest is the payload.
#![no_main]
use libfuzzer_sys::fuzz_target;
use tritium_core::Trit;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let n = (u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize) % 65_536;
    let payload = &data[4..];
    let mut trits = vec![Trit::ZERO; n];
    let _ = tritium_format::unpack_i2s_tensor(payload, n, &mut trits);
});
