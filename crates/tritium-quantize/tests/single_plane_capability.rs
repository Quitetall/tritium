//! **Is the shipped 27B artifact's single plane mis-fit, or is it paying for its curvature metric?**
//!
//! Decoding `qwen36-ptq-b3-r2-r3-565abdee` against its fp master put single-plane tiles at 0.66–0.72
//! relative Frobenius error, where a brute-force scale sweep on the same weights reaches 0.44–0.49.
//! Refitting only the scale on the artifact's own trits recovered barely a third of that, so the
//! codes differ too.
//!
//! None of that convicts the fitter. [`fit_joint_ternary`] minimizes `errorᵀ H error` under a
//! calibration-derived curvature metric, and a fit that is optimal under `H` has no obligation to be
//! optimal under plain squared error. Calling the gap a defect without scoring in the fit's own
//! metric would repeat this project's own proxy-gap mistake.
//!
//! So this asks the one question that separates the two explanations without needing the artifact's
//! lost curvature evidence: run the real fitter at one plane under [`JointFitMetric::Identity`],
//! where its objective *is* squared error. If it reaches the brute-force floor, the fitter is
//! capable and the artifact's error is the price of its metric. If it does not, the fitter is short
//! of its own objective and the metric explains nothing.
//!
//! ```text
//! TRITIUM_FP_DIR=/mnt/4tb/models/qwen36-27b-6a9e13bd \
//!   cargo test -p tritium-quantize --release --test single_plane_capability -- --ignored --nocapture
//! ```

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use tritium_quantize::{JointFitConfig, JointFitMetric, ScalePrecision, fit_joint_ternary};

/// Scale group width in the shipped artifact.
const GROUP: usize = 128;
/// Coefficients scored per tensor. 8,192 groups is a stable error at a bounded read.
const ELEMENTS: usize = GROUP * 8_192;

/// Read the leading `max_elements` values of `name` from a sharded safetensors master.
fn read_fp_prefix(dir: &Path, name: &str, max_elements: usize) -> Option<Vec<f32>> {
    let index: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json")).ok()?)
            .ok()?;
    let shard = index["weight_map"][name].as_str()?;
    let mut file = File::open(dir.join(shard)).ok()?;
    let mut length = [0u8; 8];
    file.read_exact(&mut length).ok()?;
    let header_len = u64::from_le_bytes(length) as usize;
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header).ok()?;
    let header: serde_json::Value = serde_json::from_slice(&header).ok()?;
    let entry = &header[name];
    let dtype = entry["dtype"].as_str()?.to_owned();
    let elements: u64 = entry["shape"]
        .as_array()?
        .iter()
        .map(|value| value.as_u64().unwrap_or(0))
        .product();
    let start = entry["data_offsets"][0].as_u64()?;
    let width = match dtype.as_str() {
        "BF16" | "F16" => 2usize,
        "F32" => 4,
        _ => return None,
    };
    let take = max_elements.min(elements as usize);
    let mut raw = vec![0u8; take * width];
    file.seek(SeekFrom::Start(8 + header_len as u64 + start))
        .ok()?;
    file.read_exact(&mut raw).ok()?;
    Some(match dtype.as_str() {
        "BF16" => raw
            .chunks(2)
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
            .collect(),
        "F16" => raw
            .chunks(2)
            .map(|b| f32::from(half::f16::from_le_bytes([b[0], b[1]])))
            .collect(),
        _ => raw
            .chunks(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
    })
}

/// Best achievable single-plane error for one group, by exhaustive scale sweep.
///
/// Ternary codes are determined by the scale (round-to-nearest, clamped), so sweeping the scale
/// sweeps the whole solution space up to rounding, and 1,024 steps resolves it far below the gap
/// under test. This is the reference the fitter is measured against.
fn brute_force_group(group: &[f32]) -> f64 {
    let peak = group.iter().fold(0.0f32, |acc, w| acc.max(w.abs()));
    let mut best: f64 = group.iter().map(|w| f64::from(*w) * f64::from(*w)).sum();
    for step in 1..=1_024 {
        let scale = f64::from(peak) * f64::from(step) / 1_024.0;
        if scale <= 0.0 {
            continue;
        }
        let residual: f64 = group
            .iter()
            .map(|w| {
                let level = (f64::from(*w) / scale).round().clamp(-1.0, 1.0);
                (f64::from(*w) - level * scale).powi(2)
            })
            .sum();
        best = best.min(residual);
    }
    best
}

#[test]
#[ignore = "needs the Qwen3.6 fp master; set TRITIUM_FP_DIR"]
fn the_fitter_reaches_its_own_objective_at_one_plane() {
    let Ok(fp_dir) = std::env::var("TRITIUM_FP_DIR") else {
        eprintln!("skipping: set TRITIUM_FP_DIR");
        return;
    };
    let fp_dir = PathBuf::from(fp_dir);
    let config = JointFitConfig {
        planes: 1,
        scale_precision: ScalePrecision::F16,
        ..JointFitConfig::default()
    };

    println!(
        "{:<52} {:>10} {:>10} {:>9} {:>9}",
        "tensor", "fitter", "brute", "excess", "artifact"
    );
    println!("{}", "-".repeat(96));
    // The artifact's measured single-plane error, for the same tensors, from
    // tritium-format's qwen36_artifact_fidelity diagnostic.
    for (name, artifact) in [
        ("model.language_model.embed_tokens.weight", 0.7181),
        ("model.language_model.layers.0.mlp.down_proj.weight", 0.6724),
        (
            "model.language_model.layers.32.mlp.down_proj.weight",
            0.6728,
        ),
    ] {
        let Some(weights) = read_fp_prefix(&fp_dir, name, ELEMENTS) else {
            println!("{name:<52} (fp master read failed)");
            continue;
        };
        let (mut fitted, mut brute, mut energy) = (0.0f64, 0.0f64, 0.0f64);
        for group in weights.chunks(GROUP) {
            let fit = fit_joint_ternary(group, JointFitMetric::Identity, config)
                .expect("identity-metric single-plane fit");
            fitted += group
                .iter()
                .zip(&fit.reconstruction)
                .map(|(want, got)| f64::from(want - got).powi(2))
                .sum::<f64>();
            brute += brute_force_group(group);
            energy += group
                .iter()
                .map(|w| f64::from(*w) * f64::from(*w))
                .sum::<f64>();
        }
        let fitted = (fitted / energy).sqrt();
        let brute = (brute / energy).sqrt();
        println!(
            "{name:<52} {fitted:>10.4} {brute:>10.4} {:>8.1}% {artifact:>9.4}",
            (fitted / brute - 1.0) * 100.0
        );
    }
    println!(
        "\n`fitter` is fit_joint_ternary under Identity curvature, where its objective is exactly\n\
         squared error. `brute` is the exhaustive per-group scale sweep. `excess` is how far the\n\
         fitter lands above its own optimum. `artifact` is what the shipped 27B package measured at\n\
         the same rate, under its own calibration metric."
    );
}

/// **How much squared-error does a curvature metric plausibly cost at one plane?**
///
/// The test above clears the fitter: under Identity it reaches its own optimum exactly, so the
/// artifact's 0.67 against a 0.44 floor is the metric, not a bug. That answers "is it broken" but
/// not "is it reasonable" — a 52% excess in relative error is a lot to attribute to reweighting.
///
/// Curvature evidence for the shipped package is gone, so the spread cannot be recovered directly.
/// It can be bounded from the other side: fit the same weights under diagonal metrics of known
/// spread and read off the Frobenius excess each one buys. That maps spread onto excess, and says
/// whether 52% sits inside the range a sane calibration metric produces or far outside it.
///
/// The metric is deterministic — a fixed log-spaced ramp across each group — so this measures the
/// effect of spread itself rather than of any particular sampled draw.
#[test]
#[ignore = "needs the Qwen3.6 fp master; set TRITIUM_FP_DIR"]
fn what_frobenius_excess_does_metric_spread_buy() {
    let Ok(fp_dir) = std::env::var("TRITIUM_FP_DIR") else {
        eprintln!("skipping: set TRITIUM_FP_DIR");
        return;
    };
    let fp_dir = PathBuf::from(fp_dir);
    let name = "model.language_model.layers.0.mlp.down_proj.weight";
    let Some(weights) = read_fp_prefix(&fp_dir, name, ELEMENTS) else {
        eprintln!("skipping: fp master read failed");
        return;
    };
    let config = JointFitConfig {
        planes: 1,
        scale_precision: ScalePrecision::F16,
        ..JointFitConfig::default()
    };

    println!("{name}\n");
    println!("{:>12} {:>10} {:>10}", "metric ratio", "rel err", "excess");
    println!("{}", "-".repeat(36));
    let energy: f64 = weights.iter().map(|w| f64::from(*w) * f64::from(*w)).sum();
    let floor = {
        let total: f64 = weights.chunks(GROUP).map(brute_force_group).sum();
        (total / energy).sqrt()
    };
    for ratio in [1.0f64, 4.0, 16.0, 64.0, 256.0, 1024.0] {
        // Log-spaced ramp from 1 to `ratio` across the group: max/min curvature is exactly `ratio`.
        let metric: Vec<f64> = (0..GROUP)
            .map(|index| ratio.powf(index as f64 / (GROUP - 1) as f64))
            .collect();
        let mut error = 0.0f64;
        for group in weights.chunks(GROUP) {
            let fit = fit_joint_ternary(group, JointFitMetric::DiagonalF64(&metric), config)
                .expect("diagonal-metric single-plane fit");
            error += group
                .iter()
                .zip(&fit.reconstruction)
                .map(|(want, got)| f64::from(want - got).powi(2))
                .sum::<f64>();
        }
        let error = (error / energy).sqrt();
        println!(
            "{ratio:>12.0} {error:>10.4} {:>9.1}%",
            (error / floor - 1.0) * 100.0
        );
    }
    println!(
        "\nExcess is relative Frobenius error over the unweighted floor of {floor:.4}. The shipped\n\
         artifact measured 0.6724 on this tensor, an excess of {:.1}%.",
        (0.6724 / floor - 1.0) * 100.0
    );
}

/// **Is a pruned prefix of a joint fit still a good fit?**
///
/// The spread sweep refutes the metric: 1024:1 curvature buys 2.1% Frobenius excess, and the
/// artifact shows 54%. Twenty-five times the most extreme metric tested is not reweighting.
///
/// The remaining candidate is the pipeline, not the fitter. `residual_expand` is documented as
/// prefix-stable — "the first `T` planes never change when you ask for more" — because it fits each
/// plane greedily against the previous residual. [`fit_joint_ternary`] is not greedy. It optimizes
/// all planes *together*, so plane 0 of a three-plane joint fit is whatever best serves the trio,
/// which is not the best single plane. Prefix stability is exactly what a joint fit gives up.
///
/// The allocator then assigns plane counts per tile and keeps that many planes. If the fit happened
/// at the maximum and the allocator pruned afterwards, every tile it cut to one plane kept a plane 0
/// chosen for a decomposition that no longer exists.
///
/// This measures that directly, under Identity so the metric cannot be blamed: fit three planes,
/// keep the first, and compare against fitting one plane outright.
#[test]
#[ignore = "needs the Qwen3.6 fp master; set TRITIUM_FP_DIR"]
fn a_pruned_joint_fit_is_not_a_single_plane_fit() {
    let Ok(fp_dir) = std::env::var("TRITIUM_FP_DIR") else {
        eprintln!("skipping: set TRITIUM_FP_DIR");
        return;
    };
    let fp_dir = PathBuf::from(fp_dir);
    let single = JointFitConfig {
        planes: 1,
        scale_precision: ScalePrecision::F16,
        ..JointFitConfig::default()
    };
    let triple = JointFitConfig {
        planes: 3,
        ..single
    };

    println!(
        "{:<52} {:>9} {:>9} {:>9} {:>9}",
        "tensor", "fit T=1", "T=3[..1]", "excess", "artifact"
    );
    println!("{}", "-".repeat(94));
    for (name, artifact) in [
        ("model.language_model.embed_tokens.weight", 0.7181),
        ("model.language_model.layers.0.mlp.down_proj.weight", 0.6724),
        (
            "model.language_model.layers.32.mlp.down_proj.weight",
            0.6728,
        ),
    ] {
        let Some(weights) = read_fp_prefix(&fp_dir, name, ELEMENTS) else {
            println!("{name:<52} (fp master read failed)");
            continue;
        };
        let (mut direct, mut pruned, mut energy) = (0.0f64, 0.0f64, 0.0f64);
        for group in weights.chunks(GROUP) {
            let one = fit_joint_ternary(group, JointFitMetric::Identity, single)
                .expect("single-plane fit");
            direct += group
                .iter()
                .zip(&one.reconstruction)
                .map(|(want, got)| f64::from(want - got).powi(2))
                .sum::<f64>();
            let three = fit_joint_ternary(group, JointFitMetric::Identity, triple)
                .expect("three-plane fit");
            // Keep plane 0 alone, exactly as a tile allocated one plane would ship.
            let scale = three.scales[0];
            pruned += group
                .iter()
                .zip(&three.trits[0])
                .map(|(want, trit)| f64::from(want - scale * f32::from(*trit)).powi(2))
                .sum::<f64>();
            energy += group
                .iter()
                .map(|w| f64::from(*w) * f64::from(*w))
                .sum::<f64>();
        }
        let direct = (direct / energy).sqrt();
        let pruned = (pruned / energy).sqrt();
        println!(
            "{name:<52} {direct:>9.4} {pruned:>9.4} {:>8.1}% {artifact:>9.4}",
            (pruned / direct - 1.0) * 100.0
        );
    }
    println!(
        "\n`fit T=1` fits one plane outright. `T=3[..1]` fits three jointly and keeps the first, which\n\
         is what a tile the allocator cut to one plane carries. Both under Identity curvature, so any\n\
         gap is the pruning, not the metric. Compare each against what the artifact measured."
    );
}

/// **What does each candidate fix for prefix pruning actually cost?**
///
/// Three ways out of the pruning defect were named, and two of them are the same measurement. A fit
/// stored per plane count and a refit after allocation both end up fitting exactly the count that
/// ships, so their quality is identical — they differ in master storage and in when the fit runs,
/// not in what comes out. That leaves two distinct quality options against the status quo:
///
/// - **prune** — what ships today: fit three planes jointly, keep the first `T`.
/// - **direct** — fit exactly `T` planes. Optimal at every `T`, costs a fit (and a stored copy) per
///   count rather than one shared Pmax fit.
/// - **greedy** — prefix-stable by construction: fit one plane, fit one plane to the residual, and
///   again. One stored chain serves every `T`, like today, but each prefix is a real fit. What it
///   gives up is joint optimality at the top count.
///
/// The question the table answers is whether prefix stability is free. If `greedy` matches `direct`
/// at low `T` and loses to it at `T=3`, the cost of keeping one stored chain is exactly that gap,
/// and it can be weighed against tripling the master.
///
/// All under Identity curvature so the metric plays no part. `direct` at `T=3` and `prune` at `T=3`
/// are the same fit and must print the same number — a free control on the harness.
#[test]
#[ignore = "needs the Qwen3.6 fp master; set TRITIUM_FP_DIR"]
fn what_each_fix_for_prefix_pruning_costs() {
    let Ok(fp_dir) = std::env::var("TRITIUM_FP_DIR") else {
        eprintln!("skipping: set TRITIUM_FP_DIR");
        return;
    };
    let fp_dir = PathBuf::from(fp_dir);
    let plane_config = |planes: usize| JointFitConfig {
        planes,
        scale_precision: ScalePrecision::F16,
        ..JointFitConfig::default()
    };

    for name in [
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.32.mlp.down_proj.weight",
        "model.language_model.embed_tokens.weight",
    ] {
        let Some(weights) = read_fp_prefix(&fp_dir, name, ELEMENTS) else {
            println!("{name}: fp master read failed\n");
            continue;
        };
        // [strategy][plane count - 1]
        let mut squared = [[0.0f64; 3]; 3];
        let mut energy = 0.0f64;
        for group in weights.chunks(GROUP) {
            energy += group
                .iter()
                .map(|w| f64::from(*w) * f64::from(*w))
                .sum::<f64>();

            // prune: one joint Pmax fit, sliced.
            let pmax = fit_joint_ternary(group, JointFitMetric::Identity, plane_config(3))
                .expect("three-plane fit");
            let mut running = vec![0.0f32; group.len()];
            for (slot, plane) in squared[0].iter_mut().enumerate() {
                let scale = pmax.scales[slot];
                for (value, trit) in running.iter_mut().zip(&pmax.trits[slot]) {
                    *value += scale * f32::from(*trit);
                }
                *plane += group
                    .iter()
                    .zip(&running)
                    .map(|(want, got)| f64::from(want - got).powi(2))
                    .sum::<f64>();
            }

            // direct: a separate joint fit at each count.
            for planes in 1..=3 {
                let fit = fit_joint_ternary(group, JointFitMetric::Identity, plane_config(planes))
                    .expect("direct fit");
                squared[1][planes - 1] += group
                    .iter()
                    .zip(&fit.reconstruction)
                    .map(|(want, got)| f64::from(want - got).powi(2))
                    .sum::<f64>();
            }

            // greedy: one plane at a time against the running residual.
            let mut residual: Vec<f32> = group.to_vec();
            for plane in &mut squared[2] {
                let fit = fit_joint_ternary(&residual, JointFitMetric::Identity, plane_config(1))
                    .expect("greedy plane fit");
                for (value, got) in residual.iter_mut().zip(&fit.reconstruction) {
                    *value -= got;
                }
                *plane += residual.iter().map(|r| f64::from(*r).powi(2)).sum::<f64>();
            }
        }

        println!("{name}");
        println!(
            "{:>10} {:>8} {:>9} {:>9} {:>9} {:>17}",
            "planes", "bpw", "prune", "direct", "greedy", "greedy vs direct"
        );
        println!("{}", "-".repeat(68));
        for planes in 1..=3usize {
            let error = |strategy: usize| (squared[strategy][planes - 1] / energy).sqrt();
            let (prune, direct, greedy) = (error(0), error(1), error(2));
            println!(
                "{planes:>10} {:>8.2} {prune:>9.4} {direct:>9.4} {greedy:>9.4} {:>16.1}%",
                1.625 * planes as f64 + 16.0 * planes as f64 / GROUP as f64,
                (greedy / direct - 1.0) * 100.0
            );
        }
        println!();
    }
    println!(
        "`prune` is today's pipeline. `direct` is a fit per plane count, which is also exactly what a\n\
         refit after allocation produces. `greedy` keeps one stored chain whose every prefix is a real\n\
         fit. At three planes `prune` and `direct` are the same fit and must agree."
    );
}
