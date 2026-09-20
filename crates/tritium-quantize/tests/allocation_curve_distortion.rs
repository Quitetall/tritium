//! **What does the allocator lose by ranking upgrades on pruned losses?**
//!
//! `salt_v2_master` writes one loss per prefix of a single Pmax fit, and the qwen36 spool replays
//! `tile.losses()[..admitted]` into the planner. Those prefix losses are the `prune` column measured
//! in `single_plane_capability` — 1.6× the true cost at one plane and 2.4× at two — so the allocator
//! is ranking plane upgrades against numbers that misstate what every count below Pmax costs.
//!
//! Wrong inputs do not automatically mean wrong decisions: the allocator ranks by *differences*, and
//! a curve that is uniformly inflated could rank identically. This measures whether the distortion
//! survives that. Both curves are built from the same weights, handed to the real
//! [`allocate_uniform_profile_packed`] at an identical budget, and both resulting plane maps are
//! then scored under the true per-count losses — the distortion that actually ships once tiles are
//! fit at the count they carry.
//!
//! ```text
//! TRITIUM_FP_DIR=/mnt/4tb/models/qwen36-27b-6a9e13bd \
//!   cargo test -p tritium-quantize --release --test allocation_curve_distortion -- --ignored --nocapture
//! ```

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use tritium_quantize::{
    JointFitConfig, JointFitMetric, PackedPlaneCounts, SaltV2Profile, ScalePrecision,
    UniformPrefixCurve, allocate_uniform_profile_packed, fit_joint_ternary,
};

/// Scale group width, and half an allocation tile.
const GROUP: usize = 128;
/// Allocation tile width in coefficients.
const TILE: usize = 256;
/// Tiles drawn per tensor. Three tensors gives 6,144 tiles, enough for a stable allocation.
const TILES: usize = 2_048;

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

/// Cumulative squared error after 1, 2 and 3 planes, under both fitting disciplines.
///
/// `pruned` slices one joint Pmax fit, which is what the master stores today. `direct` fits each
/// count outright, which is what a tile refit at its allocated count would carry.
fn tile_curves(tile: &[f32]) -> (UniformPrefixCurve, UniformPrefixCurve) {
    let config = |planes: usize| JointFitConfig {
        planes,
        scale_precision: ScalePrecision::F16,
        ..JointFitConfig::default()
    };
    let (mut pruned, mut direct) = ([0.0f64; 3], [0.0f64; 3]);
    for group in tile.chunks(GROUP) {
        let pmax = fit_joint_ternary(group, JointFitMetric::Identity, config(3)).expect("pmax fit");
        let mut running = vec![0.0f32; group.len()];
        for (index, slot) in pruned.iter_mut().enumerate() {
            let scale = pmax.scales[index];
            for (value, trit) in running.iter_mut().zip(&pmax.trits[index]) {
                *value += scale * f32::from(*trit);
            }
            *slot += group
                .iter()
                .zip(&running)
                .map(|(want, got)| f64::from(want - got).powi(2))
                .sum::<f64>();
        }
        for (index, slot) in direct.iter_mut().enumerate() {
            let fit = fit_joint_ternary(group, JointFitMetric::Identity, config(index + 1))
                .expect("direct fit");
            *slot += group
                .iter()
                .zip(&fit.reconstruction)
                .map(|(want, got)| f64::from(want - got).powi(2))
                .sum::<f64>();
        }
    }
    // A joint fit at a higher count can score marginally above a lower one under f16 scales; the
    // allocator requires a non-increasing curve, so clamp rather than reject the tile.
    for curve in [&mut pruned, &mut direct] {
        for index in 1..3 {
            curve[index] = curve[index].min(curve[index - 1]);
        }
    }
    (
        UniformPrefixCurve::new(pruned).expect("pruned curve"),
        UniformPrefixCurve::new(direct).expect("direct curve"),
    )
}

/// Total true distortion of one plane map, scored on the direct curves.
fn score(counts: &PackedPlaneCounts, truth: &[UniformPrefixCurve]) -> f64 {
    (0..truth.len())
        .map(|tile| {
            let planes = counts.get(tile as u64).expect("plane count") as usize;
            truth[tile].distortions()[planes - 1]
        })
        .sum()
}

#[test]
#[ignore = "needs the Qwen3.6 fp master; set TRITIUM_FP_DIR"]
fn ranking_upgrades_on_pruned_losses_misallocates_planes() {
    let Ok(fp_dir) = std::env::var("TRITIUM_FP_DIR") else {
        eprintln!("skipping: set TRITIUM_FP_DIR");
        return;
    };
    let fp_dir = PathBuf::from(fp_dir);

    let mut pruned_curves = Vec::new();
    let mut direct_curves = Vec::new();
    for name in [
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.32.mlp.down_proj.weight",
    ] {
        let Some(weights) = read_fp_prefix(&fp_dir, name, TILES * TILE) else {
            eprintln!("skipping: {name} read failed");
            return;
        };
        for tile in weights.chunks(TILE) {
            let (pruned, direct) = tile_curves(tile);
            pruned_curves.push(pruned);
            direct_curves.push(direct);
        }
    }
    let tiles = pruned_curves.len() as u64;
    let floors = PackedPlaneCounts::filled(tiles, 1, SaltV2Profile::CompactV1).expect("floors");

    println!("{tiles} tiles, {} coefficients\n", tiles as usize * TILE);
    println!(
        "{:>12} {:>14} {:>14} {:>9} {:>11}",
        "extra/tile", "decided pruned", "decided direct", "penalty", "tiles moved"
    );
    println!("{}", "-".repeat(66));
    for fraction in [0.14f64, 0.28, 0.50, 1.00, 1.50] {
        let capacity = (fraction * tiles as f64).round() as u64;
        let allocate = |curves: &[UniformPrefixCurve]| {
            allocate_uniform_profile_packed(
                tiles,
                &floors,
                capacity,
                SaltV2Profile::CompactV1,
                curves
                    .iter()
                    .copied()
                    .map(Ok::<_, std::convert::Infallible>),
            )
            .expect("allocation")
        };
        let on_pruned = allocate(&pruned_curves);
        let on_direct = allocate(&direct_curves);
        let (bad, good) = (
            score(&on_pruned.plane_counts, &direct_curves),
            score(&on_direct.plane_counts, &direct_curves),
        );
        let moved = (0..tiles)
            .filter(|tile| on_pruned.plane_counts.get(*tile) != on_direct.plane_counts.get(*tile))
            .count();
        println!(
            "{fraction:>12.2} {bad:>14.4} {good:>14.4} {:>8.2}% {moved:>11}",
            (bad / good - 1.0) * 100.0
        );
    }
    println!(
        "\nBoth columns are true distortion, scored on the direct curves — the error that ships once\n\
         a tile is fit at the count it carries. `decided pruned` allocated on the prefix losses the\n\
         master writes today; `decided direct` allocated on the truth. `penalty` is what the wrong\n\
         input costs at an identical plane budget, and `tiles moved` is how many assignments differ."
    );
}
