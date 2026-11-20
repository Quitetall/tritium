//! Produce the measured native-kernel portion of a plan-0043 Stage-7 receipt.
//!
//! Run this executable under Compute Sanitizer. The receipt binds to the clean
//! source revision embedded by `tritium-build-info`; the sanitizer log is
//! checked independently by `qualify-stage7-recipe-freeze.py`.

use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process;

use half::f16;
use serde::Serialize;
use tritium_cpu::salt_v2::{salt_v2_coefficient, salt_v2_matvec};
use tritium_cuda::{CudaBackend, SaltV2ForwardMode, train::CudaTrainBackendV1};
use tritium_format::salt_v2::SaltV2Codec;
use tritium_format::salt_v2_package::{
    SaltV2Package, SaltV2Plane, SaltV2Tensor, SaltV2Tile, SaltV2Transform,
};
use tritium_spec::TernaryBackend;

const TOLERANCE: f32 = 1e-4;
const OUTPUT_ROWS: usize = 4;

#[derive(Serialize)]
struct NativeReceipt {
    schema: &'static str,
    result: &'static str,
    release: String,
    source_revision: String,
    model_revision: String,
    physical_device: String,
    driver: String,
    sanitizer_version: String,
    sanitizer_log: String,
    cases: Vec<NativeCase>,
}

#[derive(Serialize)]
struct NativeCase {
    codec: &'static str,
    group_size: usize,
    planes: usize,
    mode: &'static str,
    rows: usize,
    columns: usize,
    short_final_group: bool,
    plane_schedule: &'static str,
    packing: &'static str,
    cpu_max_abs_error: f32,
    cuda_max_abs_error: f32,
    tolerance: f32,
    dense_materialized_bytes: u64,
}

struct Args {
    release: String,
    source_revision: String,
    model_revision: String,
    sanitizer_version: String,
    sanitizer_log: String,
    output: PathBuf,
    device: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Stage-7 native matrix failed: {error:#}");
        process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    let args = parse_args()?;
    let expected = format!("source-git:{}", args.source_revision);
    anyhow::ensure!(
        CudaTrainBackendV1::source_identity() == expected,
        "CUDA backend was not compiled from exact {expected}"
    );
    anyhow::ensure!(
        clean_revision(&args.source_revision),
        "source revision must be a full lowercase Git SHA"
    );
    anyhow::ensure!(
        !args.release.is_empty() && !args.model_revision.is_empty(),
        "release and model revision must be nonempty"
    );
    anyhow::ensure!(
        !args.sanitizer_version.trim().is_empty(),
        "sanitizer version must be supplied from `compute-sanitizer --version`"
    );

    let cuda = CudaBackend::new(args.device)?;
    let name = cuda.capabilities().device_name;
    let device = format!("cuda:{}:{name}", args.device);
    let trainer = CudaTrainBackendV1::new(args.device)?;
    let driver = trainer.cuda_driver_version();
    anyhow::ensure!(driver > 0, "CUDA driver version is unavailable");

    let mut cases = Vec::with_capacity(144);
    for (codec, codec_name, packing) in [
        (SaltV2Codec::D2, "D2", "direct-2bit"),
        (SaltV2Codec::B3, "B3", "radix-3"),
        (SaltV2Codec::S34, "S34", "s34"),
    ] {
        for group_size in [64_usize, 128, 256] {
            for plane_count in [2_usize, 3] {
                for schedule in ["uniform", "mixed"] {
                    for (short_final_group, rows, columns) in [
                        (false, 64_usize, group_size * 2),
                        (true, 3_usize, group_size + 17),
                    ] {
                        let tensor = build_tensor(
                            codec_name,
                            group_size,
                            plane_count,
                            schedule,
                            rows,
                            columns,
                        )?;
                        let package = SaltV2Package::new(codec, vec![tensor.clone()])?;
                        // The receipt geometry is activation batch rows by K.
                        // A fixed four-row resident matrix gives even the
                        // G64 short-K case multiple 256-coefficient tiles, so
                        // its mixed plane schedule is genuinely exercised.
                        let activation = (0..rows * columns)
                            .map(|index| ((index % columns) as f32 - columns as f32 / 2.0) / 97.0)
                            .collect::<Vec<_>>();
                        let (cpu, independent_cpu) =
                            cpu_batch_reference(&package, &tensor, &activation, rows, columns)?;
                        let resident = cuda.upload_salt_v2(&tensor, codec)?;
                        let exact = cuda.salt_v2_forward_exact(&resident, &activation, rows)?;
                        let fast = cuda.salt_v2_forward_fast(&resident, &activation, rows)?;

                        anyhow::ensure!(
                            exact.receipt.dense_weight_bytes() == 0
                                && fast.receipt.dense_weight_bytes() == 0,
                            "dense weight materialization detected for {codec_name} G{group_size}"
                        );
                        anyhow::ensure!(
                            exact.output.len() == cpu.len() && fast.output.len() == cpu.len(),
                            "output length differs for {codec_name} G{group_size}"
                        );
                        let cpu_error = max_abs_error(&cpu, &independent_cpu);
                        let exact_error = max_abs_error(&independent_cpu, &exact.output);
                        let fast_error = max_abs_error(&independent_cpu, &fast.output);
                        anyhow::ensure!(
                            cpu_error <= TOLERANCE && fast_error <= TOLERANCE,
                            "parity exceeds tolerance for {codec_name} G{group_size}: exact={exact_error}, fast={fast_error}"
                        );
                        anyhow::ensure!(
                            exact.receipt.mode() == SaltV2ForwardMode::Exact,
                            "exact dispatch label differs for {codec_name} G{group_size}"
                        );
                        let expected_fast_mode = if columns.is_multiple_of(256) {
                            SaltV2ForwardMode::FastWarpReduce
                        } else {
                            SaltV2ForwardMode::FastAliasesExact
                        };
                        anyhow::ensure!(
                            fast.receipt.mode() == expected_fast_mode,
                            "fast dispatch mode differs for {codec_name} G{group_size}"
                        );

                        for mode in ["exact", "fast"] {
                            cases.push(NativeCase {
                                codec: codec_name,
                                group_size,
                                planes: plane_count,
                                mode,
                                rows,
                                columns,
                                short_final_group,
                                plane_schedule: schedule,
                                packing,
                                cpu_max_abs_error: cpu_error,
                                cuda_max_abs_error: if mode == "exact" {
                                    exact_error
                                } else {
                                    fast_error
                                },
                                tolerance: TOLERANCE,
                                dense_materialized_bytes: 0,
                            });
                        }
                    }
                }
            }
        }
    }
    anyhow::ensure!(
        cases.len() == 144,
        "native matrix did not run all 144 cases"
    );

    let receipt = NativeReceipt {
        schema: "tritium.stage7-native-kernels.v1",
        result: "pass",
        release: args.release,
        source_revision: args.source_revision,
        model_revision: args.model_revision,
        physical_device: device,
        driver: format!("cuda-driver-api:{driver}"),
        sanitizer_version: args.sanitizer_version,
        sanitizer_log: args.sanitizer_log,
        cases,
    };
    publish_immutable(&args.output, &serde_json::to_vec_pretty(&receipt)?)?;
    println!(
        "Stage-7 native matrix completed: {} cases",
        receipt.cases.len()
    );
    Ok(())
}

fn build_tensor(
    codec: &str,
    group_size: usize,
    plane_count: usize,
    schedule: &str,
    rows: usize,
    columns: usize,
) -> anyhow::Result<SaltV2Tensor> {
    let coefficient_count = OUTPUT_ROWS
        .checked_mul(columns)
        .ok_or_else(|| anyhow::anyhow!("shape overflow"))?;
    let tile_count = coefficient_count.div_ceil(256);
    let mut tiles = Vec::with_capacity(tile_count);
    for tile_index in 0..tile_count {
        let tile_len = (coefficient_count - tile_index * 256).min(256);
        let present_planes = if schedule == "mixed" && tile_index % 2 == 1 {
            plane_count - 1
        } else {
            plane_count
        };
        let mut planes = Vec::with_capacity(present_planes);
        for plane_index in 0..present_planes {
            let trits = (0..tile_len)
                .map(|index| match (index + tile_index + plane_index) % 4 {
                    0 => 0,
                    1 | 2 => 1,
                    _ => -1,
                })
                .collect::<Vec<i8>>();
            let scales = (0..tile_len.div_ceil(group_size))
                .map(|group| {
                    f16::from_f32(
                        0.125 + ((tile_index * 4 + group + plane_index) % 19) as f32 / 64.0,
                    )
                })
                .collect();
            planes.push(SaltV2Plane::new_with_scale_group_size(
                trits, scales, group_size,
            )?);
        }
        tiles.push(SaltV2Tile::new(planes)?);
    }
    let tensor = SaltV2Tensor::new_with_layout(
        format!("stage7-{codec}-g{group_size}-{rows}x{columns}"),
        vec![OUTPUT_ROWS as u64, columns as u64],
        SaltV2Transform::None,
        group_size,
        tiles,
    )?;
    let realized_plane_counts = tensor
        .tiles()
        .iter()
        .map(|tile| tile.planes().len())
        .collect::<std::collections::BTreeSet<_>>();
    anyhow::ensure!(
        (schedule == "mixed" && realized_plane_counts.len() > 1)
            || (schedule == "uniform"
                && realized_plane_counts.len() == 1
                && realized_plane_counts.contains(&plane_count)),
        "requested {schedule} plane schedule was not realized"
    );
    Ok(tensor)
}

fn max_abs_error(expected: &[f32], actual: &[f32]) -> f32 {
    expected
        .iter()
        .zip(actual)
        .fold(0.0_f32, |maximum, (left, right)| {
            maximum.max((left - right).abs())
        })
}

/// Independent coefficient-wise CPU check. Coefficients are consumed one at a
/// time, so the harness never constructs or stores a dense weight matrix.
fn independent_matvec(tensor: &SaltV2Tensor, activation: &[f32]) -> anyhow::Result<Vec<f32>> {
    let rows = usize::try_from(tensor.dims()[0])?;
    let columns = usize::try_from(tensor.dims()[1])?;
    anyhow::ensure!(activation.len() == columns, "activation geometry differs");
    let mut output = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut sum = 0.0_f32;
        for (column, value) in activation.iter().enumerate() {
            let coefficient = salt_v2_coefficient(tensor, row * columns + column)?;
            sum += coefficient * value;
        }
        output.push(sum);
    }
    Ok(output)
}

fn cpu_batch_reference(
    package: &SaltV2Package,
    tensor: &SaltV2Tensor,
    activations: &[f32],
    batch_rows: usize,
    columns: usize,
) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    anyhow::ensure!(
        activations.len() == batch_rows * columns,
        "activation batch differs"
    );
    let mut cpu = Vec::with_capacity(batch_rows * OUTPUT_ROWS);
    let mut independent = Vec::with_capacity(batch_rows * OUTPUT_ROWS);
    for activation in activations.chunks_exact(columns) {
        cpu.extend(salt_v2_matvec(package, 0, activation)?.output);
        independent.extend(independent_matvec(tensor, activation)?);
    }
    Ok((cpu, independent))
}

fn clean_revision(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn parse_args() -> anyhow::Result<Args> {
    let mut values = env::args().skip(1);
    let mut output = None;
    let mut release = None;
    let mut source_revision = None;
    let mut model_revision = None;
    let mut sanitizer_version = None;
    let mut sanitizer_log = None;
    let mut device = 0_usize;
    while let Some(flag) = values.next() {
        let value = values
            .next()
            .ok_or_else(|| anyhow::anyhow!("missing value for {flag}"))?;
        match flag.as_str() {
            "--output" => output = Some(PathBuf::from(value)),
            "--release" => release = Some(value),
            "--source-revision" => source_revision = Some(value),
            "--model-revision" => model_revision = Some(value),
            "--sanitizer-version" => sanitizer_version = Some(value),
            "--sanitizer-log" => sanitizer_log = Some(value),
            "--device" => device = value.parse()?,
            _ => anyhow::bail!("unknown option: {flag}"),
        }
    }
    Ok(Args {
        release: release.ok_or_else(|| anyhow::anyhow!("--release is required"))?,
        source_revision: source_revision
            .ok_or_else(|| anyhow::anyhow!("--source-revision is required"))?,
        model_revision: model_revision
            .ok_or_else(|| anyhow::anyhow!("--model-revision is required"))?,
        sanitizer_version: sanitizer_version
            .ok_or_else(|| anyhow::anyhow!("--sanitizer-version is required"))?,
        sanitizer_log: sanitizer_log
            .ok_or_else(|| anyhow::anyhow!("--sanitizer-log is required"))?,
        output: output.ok_or_else(|| anyhow::anyhow!("--output is required"))?,
        device,
    })
}

fn publish_immutable(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    anyhow::ensure!(
        !path.as_os_str().is_empty() && !path.exists() && !path.is_symlink(),
        "receipt output must be a new non-symlink path"
    );
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let file_name = path
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("receipt output must have a filename"))?
        .to_string_lossy();
    let temporary = parent.join(format!(".{file_name}.{}.tmp", process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    file.write_all(bytes)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    drop(file);
    match fs::hard_link(&temporary, path) {
        Ok(()) => {
            fs::remove_file(&temporary)?;
            Ok(())
        }
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            Err(error.into())
        }
    }
}
