//! Benchmark support for Tritium (v0.30, ADR 0005 / WF-E).
//!
//! Shared fixtures for the divan microbenchmarks in `benches/`. The benches
//! themselves live in `benches/*.rs`; this lib holds the helpers they share so a
//! bench file stays a thin harness. Skeleton today — WF-E grows this into the
//! full perf harness (GPU benches, end-to-end tokens/sec on BitNet, roofline /
//! %-of-SOL measurement, and the `>5%` regression gate).

use half::f16;
use tritium_core::Trit;
use tritium_format::{TQ2_0_BLOCK_BYTES, num_blocks, pack_tq2_0_row};

/// Build TQ2_0-packed `[N, K]` ternary weights with unit block scales and a
/// deterministic ternary pattern — the weight fixture for the mpGEMM microbenches.
///
/// # Panics
/// Panics if `pack_tq2_0_row` rejects a row (it cannot for in-range trits); this
/// is a test/bench fixture, so a panic is the right failure mode.
#[must_use]
pub fn packed_tq2_0_weights(n: usize, k: usize) -> Vec<u8> {
    let nb = num_blocks(k);
    let row_bytes = nb * TQ2_0_BLOCK_BYTES;
    let unit = vec![f16::ONE; nb];
    let mut packed = vec![0u8; n * row_bytes];
    for ni in 0..n {
        let row: Vec<Trit> = (0..k)
            .map(|ki| Trit::from_i8(((ni + ki) % 3) as i8 - 1).expect("pattern is in {-1,0,1}"))
            .collect();
        let out = &mut packed[ni * row_bytes..(ni + 1) * row_bytes];
        pack_tq2_0_row(&row, &unit, out).expect("pack tq2_0 row");
    }
    packed
}
