//! **What a SALT artifact actually costs.** Every bpw figure this campaign reports assumed the D2
//! codec — 2 bits per trit. A trit carries `log2(3) = 1.585` bits of entropy, so D2 wastes ~20% of
//! its trit bits, and the repo already ships a denser codec that recovers it: `B3` packs five
//! radix-3 trits per byte (1.6 bits/trit), matching the rate Vec-LUT reports as SOTA.
//!
//! These gates pin two things:
//! 1. `ternary_bits_per_weight_codec` returns the **packer's own** byte count, not a re-derived
//!    constant — so a reported bpw cannot drift from what the artifact costs.
//! 2. The denser codec is genuinely **lossless**: pack→unpack must return the identical trits.
//!    A bits win that changed the weights would not be a win.

use tritium_format::salt_v2::{self, SaltV2Codec};
use tritium_train::ops::ste;

fn seeded_trits(seed: u64, n: usize) -> Vec<tritium_core::Trit> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            match s % 3 {
                0 => tritium_core::Trit::NEG,
                1 => tritium_core::Trit::ZERO,
                _ => tritium_core::Trit::POS,
            }
        })
        .collect()
}

/// The accounting must come from the packer. If `ledger` and the reported bpw ever disagree, the
/// number in a paper table is not the number on disk.
#[test]
fn reported_bpw_matches_the_packers_ledger() {
    for &run in &[128usize, 256, 576, 1536] {
        for codec in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
            let ledger = codec.ledger(run).expect("ledger");
            let trit_bits = (ledger.physical_bytes * 8) as f64 / run as f64;
            for t in 1..=3usize {
                for group in [128usize, 256] {
                    let got = ste::ternary_bits_per_weight_codec(t, group, codec, run);
                    let want = t as f64 * (trit_bits + 16.0 / group as f64);
                    assert!(
                        (got - want).abs() < 1e-12,
                        "{codec:?} run{run} T{t} g{group}: {got} vs ledger-derived {want}"
                    );
                }
            }
        }
    }
    // D2 is the historical default, so the existing figures must be untouched.
    for t in 1..=3usize {
        for group in [128usize, 256] {
            let legacy = ste::ternary_bits_per_weight(t, group);
            let via_codec = ste::ternary_bits_per_weight_codec(t, group, SaltV2Codec::D2, group);
            assert!(
                (legacy - via_codec).abs() < 1e-12,
                "D2 must reproduce the committed accounting exactly (T{t} g{group})"
            );
        }
    }
    // TQ2_0's published figure must still fall out.
    assert!((ste::ternary_bits_per_weight(1, 256) - 2.0625).abs() < 1e-9);
}

/// The headline: what B3 saves on the shapes this campaign actually reports.
#[test]
fn b3_cuts_the_reported_bpw_by_about_a_fifth() {
    println!(
        "{:<28} {:>9} {:>9} {:>8}",
        "configuration", "D2 bpw", "B3 bpw", "saving"
    );
    for (t, group) in [(1usize, 128usize), (2, 128), (3, 128), (3, 256)] {
        let d2 = ste::ternary_bits_per_weight_codec(t, group, SaltV2Codec::D2, group);
        let b3 = ste::ternary_bits_per_weight_codec(t, group, SaltV2Codec::B3, group);
        println!(
            "{:<28} {:>9.3} {:>9.3} {:>7.1}%",
            format!("T={t} g{group}"),
            d2,
            b3,
            (1.0 - b3 / d2) * 100.0
        );
        assert!(b3 < d2, "B3 must be denser than D2 (T{t} g{group})");
    }
    // The bits-matched row: T=1 g128 must land UNDER every shipping ternary format
    // (Ternary Bonsai 2.125 bpw, Fairy2i ~2.0) once packed with B3.
    let t1 = ste::ternary_bits_per_weight_codec(1, 128, SaltV2Codec::B3, 128);
    println!("\nT=1 g128 under B3: {t1:.3} bpw (Bonsai ships 2.125, Fairy2i ~2.0)");
    assert!(
        t1 < 2.0,
        "T=1 B3 must undercut shipping ternary formats, got {t1}"
    );
}

/// A denser packing is only a win if it changes nothing. Round-trip every codec on trit streams
/// including lengths that do NOT divide the codec unit, which is where padding bugs live.
#[test]
fn every_codec_round_trips_losslessly_including_ragged_runs() {
    for &n in &[1usize, 4, 5, 7, 128, 129, 256, 576, 1000] {
        let trits = seeded_trits(0xB3C0 ^ n as u64, n);

        let d2 = salt_v2::pack_d2(&trits).expect("pack d2");
        assert_eq!(d2.len(), SaltV2Codec::D2.ledger(n).unwrap().physical_bytes);
        assert_eq!(
            salt_v2::unpack_d2(&d2, n).expect("unpack d2"),
            trits,
            "D2 n={n}"
        );

        let b3 = salt_v2::pack_b3(&trits).expect("pack b3");
        assert_eq!(b3.len(), SaltV2Codec::B3.ledger(n).unwrap().physical_bytes);
        assert_eq!(
            salt_v2::unpack_b3(&b3, n).expect("unpack b3"),
            trits,
            "B3 n={n}"
        );

        assert!(
            b3.len() <= d2.len(),
            "B3 must never cost more than D2 (n={n}: {} vs {})",
            b3.len(),
            d2.len()
        );
    }
}
