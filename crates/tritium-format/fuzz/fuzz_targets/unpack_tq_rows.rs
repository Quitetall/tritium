//! TQ1_0 / TQ2_0 row decode must never panic: byte 0 selects the format,
//! bytes 1..3 pick k (bounded), the rest is the packed row.
#![no_main]
use half::f16;
use libfuzzer_sys::fuzz_target;
use tritium_core::Trit;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }
    let k = (u16::from_le_bytes([data[1], data[2]]) as usize) % 16_384;
    let nb = k.div_ceil(256);
    let payload = &data[3..];
    let mut trits = vec![Trit::ZERO; k];
    let mut scales = vec![f16::ZERO; nb];
    if data[0] & 1 == 0 {
        let _ = tritium_format::unpack_tq2_0_row(payload, &mut trits, &mut scales);
    } else {
        let _ = tritium_format::unpack_tq1_0_row(payload, &mut trits, &mut scales);
    }
});
