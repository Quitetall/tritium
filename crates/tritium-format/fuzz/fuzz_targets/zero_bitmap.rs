//! Zero-bitmap computation must never panic (threat-model tracked target):
//! bytes 0..8 pick (n, k, row_bytes) — including absurd values that must be
//! rejected by checked arithmetic, not wrapped into OOB slices.
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let n = (u16::from_le_bytes([data[0], data[1]]) as usize) % 512;
    let k = u16::from_le_bytes([data[2], data[3]]) as usize;
    let row_bytes = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let payload = &data[8..];
    let _ = tritium_format::compute_zero_bitmap(payload, k);
    let _ = tritium_format::compute_zero_bitmaps(payload, n, k, row_bytes);
});
