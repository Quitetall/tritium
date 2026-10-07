//! SALT V2 joint fitter microbenchmarks for the production G128 PTQ path.
//!
//! The cases use the same restart count, iteration cap, scale precision, ridge,
//! and relay basins as the public additive PTQ bridge. Fixture construction and
//! a correctness preflight are outside the timed loop.

use divan::{Bencher, counter::ItemsCount};
use tritium_quantize::{
    JointFitConfig, JointFitMetric, RelayBasins, ScalePrecision, fit_joint_ternary,
};

const GROUP_SIZE: usize = 128;
const PLANES: [usize; 3] = [1, 2, 3];

fn fixture() -> (Vec<f32>, Vec<f64>) {
    let weights = (0..GROUP_SIZE)
        .map(|index| {
            let bits = mix64(index as u64 ^ 0x243f_6a88_85a3_08d3);
            let centered = ((bits >> 40) as i32) - (1 << 23);
            centered as f32 / (1 << 22) as f32
        })
        .collect();
    let diagonal = (0..GROUP_SIZE)
        .map(|index| {
            let bits = mix64(index as u64 ^ 0x1319_8a2e_0370_7344);
            0.25 + ((bits >> 40) as f64) / ((1_u64 << 24) as f64)
        })
        .collect();
    (weights, diagonal)
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn config(planes: usize) -> JointFitConfig {
    JointFitConfig {
        planes,
        max_iterations: 16,
        ridge: 1e-8,
        em_restarts: 4,
        ridge_condition_limit: 1e6,
        scale_precision: ScalePrecision::F16,
        relay_basins: RelayBasins {
            softened: true,
            modulated: true,
        },
    }
}

fn main() {
    divan::main();
}

#[divan::bench(args = PLANES)]
fn joint_diagonal_g128(bencher: Bencher, planes: usize) {
    let (weights, diagonal) = fixture();
    let fit_config = config(planes);
    let preflight = fit_joint_ternary(&weights, JointFitMetric::DiagonalF64(&diagonal), fit_config)
        .expect("SALT G128 preflight fit");
    assert_eq!(preflight.scales.len(), planes);
    assert_eq!(preflight.trits.len(), planes);
    assert!(preflight.objective.is_finite());

    bencher
        .counter(ItemsCount::new(GROUP_SIZE))
        .bench_local(|| {
            let fit = fit_joint_ternary(
                divan::black_box(&weights),
                JointFitMetric::DiagonalF64(divan::black_box(&diagonal)),
                fit_config,
            )
            .expect("SALT G128 fit");
            divan::black_box(fit.objective);
        });
}
