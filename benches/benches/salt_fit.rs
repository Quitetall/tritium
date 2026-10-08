//! SALT V2 joint fitter microbenchmarks for the production G128 PTQ path.
//!
//! The cases use the same restart count, iteration cap, scale precision, ridge,
//! and relay basins as the public additive PTQ bridge. Fixture construction and
//! a correctness preflight are outside the timed loop.

use divan::{Bencher, counter::ItemsCount};
use rayon::prelude::*;
use tritium_quantize::{
    JointFitConfig, JointFitMetric, RelayBasins, ScalePrecision, fit_joint_ternary,
};

const GROUP_SIZE: usize = 128;
const ROWS: usize = 64;
const PLANES: [usize; 3] = [1, 2, 3];
const THREAD_COUNTS: [usize; 2] = [1, 4];

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

/// Measure independent row-fit throughput with the four-worker limit of the hosted CPU lane.
///
/// The row values are repeated from the deterministic G128 fixture to isolate solver throughput
/// from fixture-generation cost. The worker pool is constructed before timing, matching the
/// production path where Rayon initializes once and processes many rows.
#[divan::bench(args = THREAD_COUNTS)]
fn joint_diagonal_g128_rows(bencher: Bencher, threads: usize) {
    let (row, diagonal) = fixture();
    let weights = row.repeat(ROWS);
    let fit_config = config(2);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("row-fit benchmark pool");
    let fit_rows = || {
        pool.install(|| {
            weights
                .par_chunks_exact(GROUP_SIZE)
                .map(|row_weights| {
                    fit_joint_ternary(
                        row_weights,
                        JointFitMetric::DiagonalF64(&diagonal),
                        fit_config,
                    )
                    .expect("SALT G128 row fit")
                    .objective
                })
                .sum::<f64>()
        })
    };
    assert!(fit_rows().is_finite(), "row-fit preflight objective");

    bencher
        .counter(ItemsCount::new(ROWS))
        .bench_local(|| divan::black_box(fit_rows()));
}

const GROUPING_GROUPS: usize = 64;
const GROUPING_ROWS_PER_GROUP: usize = 32;
const GROUPING_PLANES: usize = 2;

type CompactRowFit = (Vec<f32>, Vec<Vec<i8>>, f64);

fn grouped_fixture() -> (Vec<f32>, Vec<Vec<f64>>) {
    let weights = (0..GROUPING_GROUPS * GROUPING_ROWS_PER_GROUP * GROUP_SIZE)
        .map(|index| {
            let bits = mix64(index as u64 ^ 0xa409_3822_299f_31d0);
            let centered = ((bits >> 40) as i32) - (1 << 23);
            centered as f32 / (1 << 22) as f32
        })
        .collect();
    let diagonals = (0..GROUPING_GROUPS)
        .map(|group| {
            (0..GROUP_SIZE)
                .map(|index| {
                    let bits = mix64((group * GROUP_SIZE + index) as u64 ^ 0x082e_fa98_ec4e_6c89);
                    0.25 + ((bits >> 40) as f64) / ((1_u64 << 24) as f64)
                })
                .collect()
        })
        .collect();
    (weights, diagonals)
}

fn append_compact_fit(
    fit: CompactRowFit,
    scales: &mut Vec<Vec<f32>>,
    trits: &mut [Vec<i8>],
    objective: &mut f64,
) {
    let (row_scales, row_trits, row_objective) = fit;
    *objective += row_objective;
    scales.push(row_scales);
    for (plane, row) in row_trits.into_iter().enumerate() {
        trits[plane].extend(row);
    }
}

fn finish_grouped_bench(
    scales: Vec<Vec<f32>>,
    trits: Vec<Vec<i8>>,
    objective: f64,
) -> (Vec<Vec<f32>>, Vec<Vec<i8>>, f64) {
    (scales, trits, objective)
}

/// Compare the bridge's flat fit collection against compact-result and
/// group-at-a-time collection with fixed, production-like solver settings.
/// This isolates Rayon batching/retention costs from Python, model loading,
/// calibration, ONNX, and runner placement.
#[divan::bench(args = [
    "flat-full",
    "flat-compact",
    "grouped-compact",
    "batch-8-compact"
])]
fn joint_diagonal_bridge_collection(bencher: Bencher, strategy: &str) {
    let (weights, diagonals) = grouped_fixture();
    let fit_config = config(GROUPING_PLANES);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("four-worker benchmark pool");
    let total_rows = GROUPING_GROUPS * GROUPING_ROWS_PER_GROUP;
    let total_weights = total_rows * GROUP_SIZE;
    let output_trits = || {
        (0..GROUPING_PLANES)
            .map(|_| Vec::with_capacity(total_weights))
            .collect::<Vec<_>>()
    };

    let run = |strategy: &str| {
        let mut scales = Vec::with_capacity(total_rows);
        let mut trits = output_trits();
        let mut objective = 0.0;
        match strategy {
            "flat-full" => {
                let fits = pool.install(|| {
                    weights
                        .par_chunks_exact(GROUP_SIZE)
                        .enumerate()
                        .map(|(row_index, row_weights)| {
                            let group = row_index / GROUPING_ROWS_PER_GROUP;
                            fit_joint_ternary(
                                row_weights,
                                JointFitMetric::DiagonalF64(&diagonals[group]),
                                fit_config,
                            )
                            .expect("full fit")
                        })
                        .collect::<Vec<_>>()
                });
                for fit in fits {
                    append_compact_fit(
                        (fit.scales, fit.trits, fit.objective),
                        &mut scales,
                        &mut trits,
                        &mut objective,
                    );
                }
            }
            "flat-compact" => {
                let fits = pool.install(|| {
                    weights
                        .par_chunks_exact(GROUP_SIZE)
                        .enumerate()
                        .map(|(row_index, row_weights)| {
                            let group = row_index / GROUPING_ROWS_PER_GROUP;
                            fit_joint_ternary(
                                row_weights,
                                JointFitMetric::DiagonalF64(&diagonals[group]),
                                fit_config,
                            )
                            .map(|fit| (fit.scales, fit.trits, fit.objective))
                            .expect("compact fit")
                        })
                        .collect::<Vec<_>>()
                });
                for fit in fits {
                    append_compact_fit(fit, &mut scales, &mut trits, &mut objective);
                }
            }
            "grouped-compact" => pool.install(|| {
                for (group, diagonal) in diagonals.iter().enumerate() {
                    let start = group * GROUPING_ROWS_PER_GROUP * GROUP_SIZE;
                    let end = start + GROUPING_ROWS_PER_GROUP * GROUP_SIZE;
                    let fits = weights[start..end]
                        .par_chunks_exact(GROUP_SIZE)
                        .map(|row_weights| {
                            fit_joint_ternary(
                                row_weights,
                                JointFitMetric::DiagonalF64(diagonal),
                                fit_config,
                            )
                            .map(|fit| (fit.scales, fit.trits, fit.objective))
                            .expect("grouped compact fit")
                        })
                        .collect::<Vec<_>>();
                    for fit in fits {
                        append_compact_fit(fit, &mut scales, &mut trits, &mut objective);
                    }
                }
            }),
            "batch-8-compact" => pool.install(|| {
                const GROUPS_PER_BATCH: usize = 8;
                let rows_per_group = GROUPING_ROWS_PER_GROUP;
                let rows_per_batch = GROUPS_PER_BATCH * rows_per_group;
                let weights_per_batch = rows_per_batch * GROUP_SIZE;
                for first_group in (0..GROUPING_GROUPS).step_by(GROUPS_PER_BATCH) {
                    let group_count = (GROUPING_GROUPS - first_group).min(GROUPS_PER_BATCH);
                    let first_row = first_group * rows_per_group;
                    let row_count = group_count * rows_per_group;
                    let first_weight = first_row * GROUP_SIZE;
                    let last_weight = first_weight + row_count * GROUP_SIZE;
                    let batch_weights = &weights[first_weight..last_weight];
                    debug_assert!(row_count * GROUP_SIZE <= weights_per_batch);
                    let fits = batch_weights
                        .par_chunks_exact(GROUP_SIZE)
                        .enumerate()
                        .map(|(batch_row, row_weights)| {
                            let group = first_group + batch_row / rows_per_group;
                            fit_joint_ternary(
                                row_weights,
                                JointFitMetric::DiagonalF64(&diagonals[group]),
                                fit_config,
                            )
                            .map(|fit| (fit.scales, fit.trits, fit.objective))
                            .expect("batched compact fit")
                        })
                        .collect::<Vec<_>>();
                    for fit in fits {
                        append_compact_fit(fit, &mut scales, &mut trits, &mut objective);
                    }
                }
            }),
            _ => unreachable!("declared benchmark strategy"),
        }
        finish_grouped_bench(scales, trits, objective)
    };

    let preflight = run("flat-full");
    assert!(preflight.2.is_finite(), "finite bridge benchmark objective");
    assert_eq!(preflight, run("flat-compact"), "compact flat output parity");
    assert_eq!(preflight, run("grouped-compact"), "grouped output parity");
    assert_eq!(preflight, run("batch-8-compact"), "batched output parity");
    bencher
        .counter(ItemsCount::new(total_rows))
        .bench_local(|| divan::black_box(run(strategy)));
}
