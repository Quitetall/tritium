//! **What did the shipped Qwen3.6-27B SALT V2 artifact actually cost, tensor by tensor?**
//!
//! Byte accounting says the compact profile averages 2.24 bpw and pins
//! `model.language_model.embed_tokens.weight` at exactly 1.7500 bpw — one B3 plane — in BOTH the
//! compact and the near-lossless profile, while near-lossless raises other tensors as far as 5.25.
//! Accounting cannot say what that costs. This decodes the artifact and compares it against the fp
//! master it was made from.
//!
//! Reports relative Frobenius error and the plane histogram per tensor, so a tensor starved by the
//! allocator is visible as a number instead of inferred from its rate.
//!
//! ```text
//! TRITIUM_TSALT2=/mnt/4tb/tmp/qwen36-ptq-b3-r2-r3-565abdee/compact.tsalt2 \
//! TRITIUM_FP_DIR=/mnt/4tb/models/qwen36-27b-6a9e13bd \
//!   cargo test -p tritium-format --release --test qwen36_artifact_fidelity -- --ignored --nocapture
//! ```

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use tritium_format::salt_v2::{SaltV2Codec, unpack_b3, unpack_d2};
use tritium_format::salt_v2_package::{SALT_V2_ALLOCATION_TILE_SIZE, SaltV2PackageReader};

/// Tiles compared per tensor. 200k tiles is 51.2M coefficients — enough for a stable error, and it
/// bounds the fp bytes read out of a 55 GB master.
const MAX_TILES: usize = 200_000;

/// Locate `name` in a sharded safetensors master and read its leading `max_elements` values.
fn read_fp_prefix(dir: &Path, name: &str, max_elements: usize) -> Option<Vec<f32>> {
    let index: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json")).ok()?)
            .ok()?;
    let shard = index["weight_map"][name].as_str()?;
    let mut file = File::open(dir.join(shard)).ok()?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len).ok()?;
    let header_len = u64::from_le_bytes(len) as usize;
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header).ok()?;
    let header: serde_json::Value = serde_json::from_slice(&header).ok()?;
    let entry = &header[name];
    let dtype = entry["dtype"].as_str()?.to_owned();
    let elements: u64 = entry["shape"]
        .as_array()?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0))
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
            .chunks_exact(2)
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
            .collect(),
        "F16" => raw
            .chunks_exact(2)
            .map(|b| f32::from(half::f16::from_le_bytes([b[0], b[1]])))
            .collect(),
        _ => raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
    })
}

#[test]
#[ignore = "needs the Qwen3.6 artifact and its fp master; set TRITIUM_TSALT2 and TRITIUM_FP_DIR"]
fn qwen36_artifact_error_against_the_fp_master() {
    let (Ok(package), Ok(fp_dir)) = (
        std::env::var("TRITIUM_TSALT2"),
        std::env::var("TRITIUM_FP_DIR"),
    ) else {
        eprintln!("skipping: set TRITIUM_TSALT2 and TRITIUM_FP_DIR");
        return;
    };
    let fp_dir = PathBuf::from(fp_dir);
    let source = BufReader::with_capacity(1 << 20, File::open(&package).expect("open package"));
    let mut reader = SaltV2PackageReader::new_strict(source).expect("strict package read");
    let codec = reader.codec();

    let names = [
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.self_attn.o_proj.weight",
        "model.language_model.layers.16.mlp.down_proj.weight",
        "model.language_model.layers.32.mlp.down_proj.weight",
        "model.language_model.layers.47.mlp.down_proj.weight",
    ];

    println!("package: {package}\ncodec: {codec:?}\n");
    println!(
        "{:<56} {:>7} {:>9} {:>7}  {}",
        "tensor", "bpw", "rel err", "xform", "planes/tile"
    );
    println!("{}", "-".repeat(110));
    for name in names {
        let Some(info) = reader.tensor_info(name).cloned() else {
            println!("{name:<56} (not in package)");
            continue;
        };
        let group = info.scale_group_size();
        let tiles = info.tile_count().min(MAX_TILES);
        let Some(reference) = read_fp_prefix(&fp_dir, name, tiles * SALT_V2_ALLOCATION_TILE_SIZE)
        else {
            println!("{name:<56} (fp master read failed)");
            continue;
        };
        let mut recon = vec![0.0f32; reference.len()];
        let mut histogram: BTreeMap<usize, usize> = BTreeMap::new();
        reader
            .visit_packed_tensor(name, |plane| {
                let base = plane.tile_index() * SALT_V2_ALLOCATION_TILE_SIZE;
                if base >= recon.len() {
                    return;
                }
                if plane.plane_index() == 0 {
                    *histogram.entry(plane.plane_count()).or_default() += 1;
                }
                let trits = match codec {
                    SaltV2Codec::B3 => unpack_b3(plane.packed_bytes(), plane.logical_len()),
                    SaltV2Codec::D2 => unpack_d2(plane.packed_bytes(), plane.logical_len()),
                    other => panic!("this diagnostic does not decode {other:?}"),
                }
                .expect("canonical plane payload");
                let scales = plane.scales();
                for (offset, trit) in trits.iter().enumerate() {
                    let Some(slot) = recon.get_mut(base + offset) else {
                        break;
                    };
                    *slot += trit.to_f32() * scales[offset / group].to_f32();
                }
            })
            .expect("visit tensor planes");

        let (mut error, mut energy) = (0.0f64, 0.0f64);
        for (want, got) in reference.iter().zip(&recon) {
            error += f64::from(want - got).powi(2);
            energy += f64::from(*want).powi(2);
        }
        let bits = (info.encoded_payload_bytes() + info.encoded_scale_bytes()) as f64 * 8.0;
        let planes = histogram
            .iter()
            .map(|(count, tiles)| format!("T{count}:{tiles}"))
            .collect::<Vec<_>>()
            .join(" ");
        println!(
            "{name:<56} {:>7.4} {:>9.4} {:>7?}  {planes}",
            bits / info.logical_coefficients() as f64,
            (error / energy).sqrt(),
            info.transform(),
        );
    }
    println!(
        "\nrel err is ‖W−Ŵ‖/‖W‖ over the first {MAX_TILES} tiles. One ternary plane on a roughly \
         Gaussian tensor reconstructs near 0.4; three planes near 0.07."
    );
}

/// **How many tensors did the allocator actually fund?**
///
/// The per-tensor run above shows `lm_head` at three planes and everything it sampled at one. This
/// walks every tensor in the package — metadata only, no payload decode — so the claim "the
/// allocator spent the whole budget on one tensor" is either confirmed or refuted on all 506.
#[test]
#[ignore = "needs the Qwen3.6 artifact; set TRITIUM_TSALT2"]
fn qwen36_allocation_census() {
    let Ok(package) = std::env::var("TRITIUM_TSALT2") else {
        eprintln!("skipping: set TRITIUM_TSALT2");
        return;
    };
    let source = BufReader::with_capacity(1 << 20, File::open(&package).expect("open package"));
    let reader = SaltV2PackageReader::new_strict(source).expect("strict package read");

    let names: Vec<String> = reader.tensor_names().map(str::to_owned).collect();
    let mut by_rate: BTreeMap<u64, (usize, u64)> = BTreeMap::new();
    let (mut total_bits, mut total_coefficients) = (0u128, 0u128);
    let mut widest: Vec<(f64, String)> = Vec::new();
    for name in &names {
        let info = reader.tensor_info(name).expect("listed tensor");
        let bits = (info.encoded_payload_bytes() + info.encoded_scale_bytes()) as u128 * 8;
        let coefficients = info.logical_coefficients() as u128;
        total_bits += bits;
        total_coefficients += coefficients;
        let bpw = bits as f64 / coefficients as f64;
        // Bucket by hundredths of a bit: allocation differences below that are rounding.
        let entry = by_rate.entry((bpw * 100.0).round() as u64).or_default();
        entry.0 += 1;
        entry.1 += info.logical_coefficients() as u64;
        widest.push((bpw, name.clone()));
    }
    widest.sort_by(|a, b| b.0.total_cmp(&a.0));

    println!("package: {package}\ntensors: {}", names.len());
    println!("\n{:>8}  {:>7}  {:>16}", "bpw", "tensors", "coefficients");
    for (rate, (count, coefficients)) in &by_rate {
        println!(
            "{:>8.2}  {count:>7}  {coefficients:>16}",
            *rate as f64 / 100.0
        );
    }
    println!("\ntop rates:");
    for (bpw, name) in widest.iter().take(8) {
        println!("  {bpw:>7.4}  {name}");
    }
    println!(
        "\noverall {:.4} bpw over {total_coefficients} coefficients",
        total_bits as f64 / total_coefficients as f64
    );
}

/// **Is one plane inherently this bad, or is the artifact's single plane badly scaled?**
///
/// The fidelity run measures 0.66–0.72 relative error at one plane. A ternary quantizer with a
/// per-group scale chosen to minimise squared error sits near 0.45 on Gaussian weights, so the two
/// numbers answer different questions: what one plane costs, versus what this artifact's one plane
/// costs. This computes the floor directly from the fp master by sweeping the group scale.
#[test]
#[ignore = "needs the fp master; set TRITIUM_FP_DIR"]
fn single_plane_floor_from_the_fp_master() {
    let Ok(fp_dir) = std::env::var("TRITIUM_FP_DIR") else {
        eprintln!("skipping: set TRITIUM_FP_DIR");
        return;
    };
    let fp_dir = PathBuf::from(fp_dir);
    const GROUP: usize = 128;
    const ELEMENTS: usize = 128 * 8192;

    println!("{:<56} {:>10} {:>10}", "tensor", "T=1 floor", "T=2 floor");
    println!("{}", "-".repeat(80));
    for name in [
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.32.mlp.down_proj.weight",
    ] {
        let Some(weights) = read_fp_prefix(&fp_dir, name, ELEMENTS) else {
            println!("{name:<56} (fp master read failed)");
            continue;
        };
        let mut floors = [0.0f64; 2];
        for (planes, floor) in floors.iter_mut().enumerate() {
            let (mut error, mut energy) = (0.0f64, 0.0f64);
            // Levels reachable by `planes + 1` tied ternary planes on the 1/3 ladder.
            let limit = (3i32.pow(planes as u32 + 1) - 1) / 2;
            for group in weights.chunks(GROUP) {
                let peak = group.iter().fold(0.0f32, |acc, w| acc.max(w.abs()));
                let mut best: f64 = group.iter().map(|w| f64::from(*w) * f64::from(*w)).sum();
                for step in 1..=256 {
                    let scale = f64::from(peak) * f64::from(step) / 256.0 / f64::from(limit);
                    if scale <= 0.0 {
                        continue;
                    }
                    let residual: f64 = group
                        .iter()
                        .map(|w| {
                            let level = (f64::from(*w) / scale)
                                .round()
                                .clamp(f64::from(-limit), f64::from(limit));
                            (f64::from(*w) - level * scale).powi(2)
                        })
                        .sum();
                    best = best.min(residual);
                }
                error += best;
                energy += group
                    .iter()
                    .map(|w| f64::from(*w) * f64::from(*w))
                    .sum::<f64>();
            }
            *floor = (error / energy).sqrt();
        }
        println!("{name:<56} {:>10.4} {:>10.4}", floors[0], floors[1]);
    }
    println!(
        "\nFloors are per-group optimal scales over {ELEMENTS} coefficients, g{GROUP}, exhaustive \
         256-step scale sweep. Compare against the artifact's measured error at the same rate."
    );
}

/// **Is the artifact's single plane mis-scaled, or are its trits wrong?**
///
/// The artifact reconstructs single-plane tiles at 0.66–0.72 relative error; an optimally scaled
/// single ternary plane on the same weights reaches 0.44–0.49. That 3 dB is either in the stored
/// scale or in the stored trit assignment, and the two have different fixes.
///
/// The separation is exact: hold the artifact's trits fixed and replace each group's stored scale
/// with the least-squares optimum `⟨w,t⟩/⟨t,t⟩`. If the error falls to the floor, the codes are fine
/// and only the scale is wrong. If it stays high, the assignment itself is wrong.
#[test]
#[ignore = "needs the Qwen3.6 artifact and its fp master; set TRITIUM_TSALT2 and TRITIUM_FP_DIR"]
fn stored_scale_versus_refit_scale_on_the_same_trits() {
    let (Ok(package), Ok(fp_dir)) = (
        std::env::var("TRITIUM_TSALT2"),
        std::env::var("TRITIUM_FP_DIR"),
    ) else {
        eprintln!("skipping: set TRITIUM_TSALT2 and TRITIUM_FP_DIR");
        return;
    };
    let fp_dir = PathBuf::from(fp_dir);
    let source = BufReader::with_capacity(1 << 20, File::open(&package).expect("open package"));
    let mut reader = SaltV2PackageReader::new_strict(source).expect("strict package read");
    let codec = reader.codec();
    const TILES: usize = 8192;

    println!(
        "{:<52} {:>6} {:>8} {:>8} {:>8} {:>7}",
        "tensor", "group", "stored", "refit", "nonzero", "|s|/rms"
    );
    println!("{}", "-".repeat(96));
    for name in [
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.32.mlp.down_proj.weight",
    ] {
        let Some(info) = reader.tensor_info(name).cloned() else {
            println!("{name:<52} (not in package)");
            continue;
        };
        let group = info.scale_group_size();
        let tiles = info.tile_count().min(TILES);
        let Some(reference) = read_fp_prefix(&fp_dir, name, tiles * SALT_V2_ALLOCATION_TILE_SIZE)
        else {
            println!("{name:<52} (fp master read failed)");
            continue;
        };
        // One accumulator per group: stored-scale residual, refit residual, and the sums the
        // least-squares scale needs. Single-plane tiles only — a multi-plane tile has no single
        // scale to refit and is skipped.
        let groups = reference.len() / group;
        let mut stored_error = vec![0.0f64; groups];
        let mut dot = vec![0.0f64; groups];
        let mut code_energy = vec![0.0f64; groups];
        let mut stored_scale = vec![0.0f64; groups];
        let mut counted = vec![false; groups];
        let mut nonzero = 0u64;
        let mut coefficients = 0u64;
        reader
            .visit_packed_tensor(name, |plane| {
                let base = plane.tile_index() * SALT_V2_ALLOCATION_TILE_SIZE;
                if base >= reference.len() || plane.plane_count() != 1 {
                    return;
                }
                let trits = match codec {
                    SaltV2Codec::B3 => unpack_b3(plane.packed_bytes(), plane.logical_len()),
                    SaltV2Codec::D2 => unpack_d2(plane.packed_bytes(), plane.logical_len()),
                    other => panic!("this diagnostic does not decode {other:?}"),
                }
                .expect("canonical plane payload");
                for (offset, trit) in trits.iter().enumerate() {
                    let Some(want) = reference.get(base + offset).copied().map(f64::from) else {
                        break;
                    };
                    let index = (base + offset) / group;
                    let scale = f64::from(plane.scales()[offset / group].to_f32());
                    let code = f64::from(trit.get());
                    stored_error[index] += (want - code * scale).powi(2);
                    dot[index] += want * code;
                    code_energy[index] += code * code;
                    stored_scale[index] = scale;
                    counted[index] = true;
                    nonzero += u64::from(!trit.is_zero());
                    coefficients += 1;
                }
            })
            .expect("visit tensor planes");

        let (mut stored, mut refit, mut energy, mut ratio, mut live) = (0.0, 0.0, 0.0, 0.0, 0u64);
        for index in 0..groups {
            if !counted[index] {
                continue;
            }
            let window = &reference[index * group..(index + 1) * group];
            let group_energy: f64 = window.iter().map(|w| f64::from(*w).powi(2)).sum();
            let best = if code_energy[index] > 0.0 {
                dot[index] / code_energy[index]
            } else {
                0.0
            };
            stored += stored_error[index];
            refit += group_energy - best * dot[index];
            energy += group_energy;
            let rms = (group_energy / group as f64).sqrt();
            ratio += stored_scale[index] / rms;
            live += 1;
        }
        if live == 0 {
            println!("{name:<52} {group:>6} (no single-plane tiles in range)");
            continue;
        }
        println!(
            "{name:<52} {group:>6} {:>8.4} {:>8.4} {:>8.3} {:>7.3}",
            (stored / energy).sqrt(),
            (refit / energy).sqrt(),
            nonzero as f64 / coefficients as f64,
            ratio / live as f64,
        );
    }
    println!(
        "\n`stored` uses the artifact's own scales, `refit` the least-squares scale on the artifact's\n\
         own trits. `nonzero` is the fraction of trits that are not zero — an optimal ternary plane\n\
         on Gaussian weights leaves about 0.46 of them live. `|s|/rms` is the stored scale over the\n\
         group RMS; the optimum is near 1.0."
    );
}
