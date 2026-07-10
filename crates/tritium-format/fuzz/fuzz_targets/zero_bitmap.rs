//! Zero-bitmap computation must never panic (threat-model tracked target):
//! bytes 0..18 pick (n, row_bytes, k) at FULL width — n and row_bytes as raw
//! u64s, so wrapped products, `row_bytes = 0`, and capacity-overflow-sized
//! allocations are all reachable and must surface as typed errors, never
//! panics or aborts (both functions are total: every allocation is bounded
//! by the input length once the size preconditions pass).
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 18 {
        return;
    }
    let n = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
    let row_bytes = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
    // Low bytes of row_bytes double as a full-width k draw: both functions
    // are total, so absurd k is free extra coverage, not an OOM risk.
    let k = u16::from_le_bytes([data[16], data[17]]) as usize;
    let payload = &data[18..];
    let _ = tritium_format::compute_zero_bitmap(payload, k);
    let _ = tritium_format::compute_zero_bitmap(payload, n); // absurd k too
    let _ = tritium_format::compute_zero_bitmaps(payload, n, k, row_bytes);
    let _ = tritium_format::compute_zero_bitmaps(payload, n, row_bytes, row_bytes);
});
