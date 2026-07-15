//! SALT V2 host-codec microbenchmarks.
//!
//! The sweep covers terminal codec-unit boundaries, the default 128-trit SALT
//! group boundary, and a 1 Mi-trit throughput case. Each timed pack/unpack path
//! is preceded by a round-trip and exact physical-byte assertion, so a codec
//! implementation cannot silently trade canonical size for speed.

use divan::{
    Bencher,
    counter::{BytesCount, ItemsCount},
};
use tritium_core::Trit;
use tritium_format::salt_v2::{
    B3_TRITS_PER_BYTE, D2_TRITS_PER_BYTE, S34_BITS_PER_GROUP, S34_TRITS_PER_GROUP, SaltV2Codec,
    SaltV2CodecError, pack_b3, pack_d2, pack_s34, unpack_b3, unpack_d2, unpack_s34,
};

const DENSE_LENGTHS: [usize; 10] = [1, 3, 4, 5, 6, 127, 128, 129, 4_096, 1 << 20];
const S34_LENGTHS: [usize; 10] = [4, 8, 28, 32, 36, 124, 128, 132, 4_096, 1 << 20];

#[derive(Clone, Copy, Debug)]
enum DenseDistribution {
    Balanced,
    ZeroHeavy,
    NoZeros,
}

const DENSE_DISTRIBUTIONS: [DenseDistribution; 3] = [
    DenseDistribution::Balanced,
    DenseDistribution::ZeroHeavy,
    DenseDistribution::NoZeros,
];

#[derive(Clone, Copy, Debug)]
enum S34Distribution {
    ScatteredMixedSigns,
    FixedZeroMixedSigns,
    ScatteredPositive,
}

const S34_DISTRIBUTIONS: [S34Distribution; 3] = [
    S34Distribution::ScatteredMixedSigns,
    S34Distribution::FixedZeroMixedSigns,
    S34Distribution::ScatteredPositive,
];

#[derive(Clone, Copy, Debug)]
struct DenseCase {
    len: usize,
    distribution: DenseDistribution,
}

#[derive(Clone, Copy, Debug)]
struct S34Case {
    len: usize,
    distribution: S34Distribution,
}

fn dense_cases() -> impl Iterator<Item = DenseCase> {
    DENSE_LENGTHS.into_iter().flat_map(|len| {
        DENSE_DISTRIBUTIONS
            .into_iter()
            .map(move |distribution| DenseCase { len, distribution })
    })
}

fn s34_cases() -> impl Iterator<Item = S34Case> {
    S34_LENGTHS.into_iter().flat_map(|len| {
        S34_DISTRIBUTIONS
            .into_iter()
            .map(move |distribution| S34Case { len, distribution })
    })
}

fn dense_fixture(case: DenseCase) -> Vec<Trit> {
    (0..case.len)
        .map(|index| {
            let symbol = pseudo_random(index, 0x243f_6a88_85a3_08d3);
            match case.distribution {
                DenseDistribution::Balanced => match symbol % 3 {
                    0 => Trit::NEG,
                    1 => Trit::ZERO,
                    _ => Trit::POS,
                },
                DenseDistribution::ZeroHeavy => match symbol % 8 {
                    0 => Trit::NEG,
                    1 => Trit::POS,
                    _ => Trit::ZERO,
                },
                DenseDistribution::NoZeros => {
                    if symbol & 1 == 0 {
                        Trit::NEG
                    } else {
                        Trit::POS
                    }
                }
            }
        })
        .collect()
}

fn s34_fixture(case: S34Case) -> Vec<Trit> {
    debug_assert!(case.len.is_multiple_of(4));
    (0..case.len)
        .map(|index| {
            let group = index / 4;
            let slot = index % 4;
            let zero_slot = match case.distribution {
                S34Distribution::FixedZeroMixedSigns => 1,
                S34Distribution::ScatteredMixedSigns | S34Distribution::ScatteredPositive => {
                    (pseudo_random(group, 0x1319_8a2e_0370_7344) & 3) as usize
                }
            };
            if slot == zero_slot {
                return Trit::ZERO;
            }
            match case.distribution {
                S34Distribution::ScatteredPositive => Trit::POS,
                S34Distribution::ScatteredMixedSigns | S34Distribution::FixedZeroMixedSigns => {
                    if pseudo_random(index, 0xa409_3822_299f_31d0) & 1 == 0 {
                        Trit::NEG
                    } else {
                        Trit::POS
                    }
                }
            }
        })
        .collect()
}

fn pseudo_random(index: usize, seed: u64) -> u64 {
    let mut value = (index as u64).wrapping_add(seed);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn assert_fixture(codec: SaltV2Codec, input: &[Trit], packed: &[u8], decoded: &[Trit]) {
    let ledger = codec.ledger(input.len()).expect("codec ledger");
    let expected_bytes = match codec {
        SaltV2Codec::D2 => input.len().div_ceil(D2_TRITS_PER_BYTE),
        SaltV2Codec::B3 => input.len().div_ceil(B3_TRITS_PER_BYTE),
        SaltV2Codec::S34 => {
            let groups = input.len() / S34_TRITS_PER_GROUP;
            (groups * S34_BITS_PER_GROUP).div_ceil(8)
        }
        other => panic!("no benchmark size formula for {other:?}"),
    };
    assert_eq!(ledger.physical_bytes, expected_bytes);
    assert_eq!(packed.len(), expected_bytes);
    assert_eq!(decoded, input);
}

fn bench_pack<Pack, Unpack>(
    bencher: Bencher,
    input: Vec<Trit>,
    codec: SaltV2Codec,
    pack: Pack,
    unpack: Unpack,
) where
    Pack: Fn(&[Trit]) -> Result<Vec<u8>, SaltV2CodecError>,
    Unpack: Fn(&[u8], usize) -> Result<Vec<Trit>, SaltV2CodecError>,
{
    let packed = pack(&input).expect("pack codec fixture");
    let decoded = unpack(&packed, input.len()).expect("unpack codec fixture");
    assert_fixture(codec, &input, &packed, &decoded);

    bencher
        .counter(ItemsCount::new(input.len()))
        .counter(BytesCount::new(packed.len()))
        .bench_local(|| pack(divan::black_box(&input)).expect("pack codec"));
}

fn bench_unpack<Pack, Unpack>(
    bencher: Bencher,
    input: Vec<Trit>,
    codec: SaltV2Codec,
    pack: Pack,
    unpack: Unpack,
) where
    Pack: Fn(&[Trit]) -> Result<Vec<u8>, SaltV2CodecError>,
    Unpack: Fn(&[u8], usize) -> Result<Vec<Trit>, SaltV2CodecError>,
{
    let packed = pack(&input).expect("pack codec fixture");
    let decoded = unpack(&packed, input.len()).expect("unpack codec fixture");
    assert_fixture(codec, &input, &packed, &decoded);

    bencher
        .counter(ItemsCount::new(input.len()))
        .counter(BytesCount::new(packed.len()))
        .bench_local(|| unpack(divan::black_box(&packed), input.len()).expect("unpack codec"));
}

fn main() {
    divan::main();
}

#[divan::bench(args = dense_cases())]
fn d2_pack(bencher: Bencher, case: DenseCase) {
    bench_pack(
        bencher,
        dense_fixture(case),
        SaltV2Codec::D2,
        pack_d2,
        unpack_d2,
    );
}

#[divan::bench(args = dense_cases())]
fn d2_unpack(bencher: Bencher, case: DenseCase) {
    bench_unpack(
        bencher,
        dense_fixture(case),
        SaltV2Codec::D2,
        pack_d2,
        unpack_d2,
    );
}

#[divan::bench(args = dense_cases())]
fn b3_pack(bencher: Bencher, case: DenseCase) {
    bench_pack(
        bencher,
        dense_fixture(case),
        SaltV2Codec::B3,
        pack_b3,
        unpack_b3,
    );
}

#[divan::bench(args = dense_cases())]
fn b3_unpack(bencher: Bencher, case: DenseCase) {
    bench_unpack(
        bencher,
        dense_fixture(case),
        SaltV2Codec::B3,
        pack_b3,
        unpack_b3,
    );
}

// These matched cases feed byte-identical, S34-valid trits through every codec.
// Keep them separate from the broader dense-only sweep so codec throughput is
// not confounded with a different input distribution.
#[divan::bench(args = s34_cases())]
fn matched_d2_pack(bencher: Bencher, case: S34Case) {
    bench_pack(
        bencher,
        s34_fixture(case),
        SaltV2Codec::D2,
        pack_d2,
        unpack_d2,
    );
}

#[divan::bench(args = s34_cases())]
fn matched_d2_unpack(bencher: Bencher, case: S34Case) {
    bench_unpack(
        bencher,
        s34_fixture(case),
        SaltV2Codec::D2,
        pack_d2,
        unpack_d2,
    );
}

#[divan::bench(args = s34_cases())]
fn matched_b3_pack(bencher: Bencher, case: S34Case) {
    bench_pack(
        bencher,
        s34_fixture(case),
        SaltV2Codec::B3,
        pack_b3,
        unpack_b3,
    );
}

#[divan::bench(args = s34_cases())]
fn matched_b3_unpack(bencher: Bencher, case: S34Case) {
    bench_unpack(
        bencher,
        s34_fixture(case),
        SaltV2Codec::B3,
        pack_b3,
        unpack_b3,
    );
}

#[divan::bench(args = s34_cases())]
fn s34_pack(bencher: Bencher, case: S34Case) {
    bench_pack(
        bencher,
        s34_fixture(case),
        SaltV2Codec::S34,
        pack_s34,
        unpack_s34,
    );
}

#[divan::bench(args = s34_cases())]
fn s34_unpack(bencher: Bencher, case: S34Case) {
    bench_unpack(
        bencher,
        s34_fixture(case),
        SaltV2Codec::S34,
        pack_s34,
        unpack_s34,
    );
}
