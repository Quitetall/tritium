//! **Is the 1/3 ladder actually prefix-decodable, and at what price?**
//!
//! ADR 0043 L-A claims prefix-decodability for the ladder: a lower plane count is a slice of the
//! same artifact. The same claim is *false* for SALT V2's free-scale joint fit — measured on this fp
//! master, a sliced prefix is 1.6× worse than a direct fit at one plane and 2.4× at two
//! (`tritium-quantize/tests/single_plane_capability.rs`) — so the ladder version has to be measured
//! rather than inherited from the argument.
//!
//! The argument is good but not complete. Dropping the lowest balanced-ternary digit is
//! round-to-nearest onto a grid three times coarser, so a ladder prefix is a genuine
//! `3^(T−1)`-level uniform quantizer, never a leftover. What it cannot be is *optimally scaled*:
//! its step is `3·Δ_T`, inherited from a Δ tuned for `T` planes, where a direct `T−1` fit picks its
//! own. This measures that residual gap, unrotated and Hadamard-rotated (rotation is orthonormal,
//! so error in the rotated basis is error in the weight basis).
//!
//! ```text
//! TRITIUM_FP_DIR=/mnt/4tb/models/qwen36-27b-6a9e13bd \
//!   cargo test -p tritium-nn --release --test ladder_prefix_decodability -- --ignored --nocapture
//! ```

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use tritium_train::ops::ste::{self, RotationPolicy};

const GROUP: usize = 128;
const GROUPS: usize = 8_192;
/// The CLI default.
const GRID: usize = 256;
const PMAX: usize = 4;

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
    if entry["dtype"].as_str()? != "BF16" {
        return None;
    }
    let start = entry["data_offsets"][0].as_u64()?;
    let mut raw = vec![0u8; max_elements * 2];
    file.seek(SeekFrom::Start(8 + header_len as u64 + start))
        .ok()?;
    file.read_exact(&mut raw).ok()?;
    Some(
        raw.chunks(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
    )
}

/// Squared error of the first `planes` planes of each group's ladder fit.
fn prefix_error(weights: &[f32], fits: &[(f32, Vec<Vec<i8>>)], planes: usize) -> f64 {
    weights
        .chunks(GROUP)
        .zip(fits)
        .map(|(group, (top, trits))| {
            group
                .iter()
                .enumerate()
                .map(|(index, want)| {
                    let mut got = 0.0f32;
                    let mut scale = *top;
                    for plane in trits.iter().take(planes) {
                        got += scale * f32::from(plane[index]);
                        scale /= 3.0;
                    }
                    f64::from(want - got).powi(2)
                })
                .sum::<f64>()
        })
        .sum()
}

/// Squared error of a `planes`-deep ladder after refining each group's step by least squares.
///
/// The shipped fitter searches `Δ` on a log grid of ratio `2^(-1/4)` — 16% between candidates — and
/// stops there. With the integer codes `k` held fixed the optimal step is closed-form,
/// `Δ* = Σ w·k / Σ k²`, and re-rounding at `Δ*` then repeating is a Lloyd iteration that cannot
/// raise the error. Three rounds from the grid's own answer measures what that coarse grid leaves.
fn refined_error(weights: &[f32], fits: &[(f32, Vec<Vec<i8>>)], planes: usize) -> f64 {
    let kmax = ((3i32.pow(planes as u32) - 1) / 2) as f32;
    weights
        .chunks(GROUP)
        .zip(fits)
        .map(|(group, (top, _))| {
            let mut delta = top / 3f32.powi(planes as i32 - 1);
            for _ in 0..3 {
                let (mut wk, mut kk) = (0.0f64, 0.0f64);
                for w in group {
                    let k = f64::from((w / delta).round().clamp(-kmax, kmax));
                    wk += f64::from(*w) * k;
                    kk += k * k;
                }
                if kk > 0.0 && wk > 0.0 {
                    delta = (wk / kk) as f32;
                }
            }
            group
                .iter()
                .map(|w| {
                    let k = (w / delta).round().clamp(-kmax, kmax);
                    f64::from(w - k * delta).powi(2)
                })
                .sum::<f64>()
        })
        .sum()
}

#[test]
#[ignore = "needs the Qwen3.6 fp master; set TRITIUM_FP_DIR"]
fn a_ladder_prefix_is_a_real_quantizer_with_an_inherited_step() {
    let Ok(fp_dir) = std::env::var("TRITIUM_FP_DIR") else {
        eprintln!("skipping: set TRITIUM_FP_DIR");
        return;
    };
    let fp_dir = PathBuf::from(fp_dir);
    for name in [
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.embed_tokens.weight",
    ] {
        let Some(raw) = read_fp_prefix(&fp_dir, name, GROUP * GROUPS) else {
            println!("{name}: fp master read failed");
            continue;
        };
        for rotated in [false, true] {
            let mut weights = raw.clone();
            if rotated {
                for group in weights.chunks_mut(GROUP) {
                    ste::fast_hadamard(group);
                }
            }
            let energy: f64 = weights.iter().map(|w| f64::from(*w).powi(2)).sum();
            let fit = |planes: usize| {
                ste::geometric_ladder_fit(
                    &weights,
                    GROUPS,
                    GROUP,
                    planes,
                    GROUP,
                    GRID,
                    RotationPolicy::Never,
                )
            };
            let master = fit(PMAX);
            println!(
                "{name}  [{}]",
                if rotated {
                    "Hadamard-rotated"
                } else {
                    "unrotated"
                }
            );
            println!(
                "{:>8} {:>12} {:>10} {:>9} {:>10} {:>9}",
                "planes", "T=4 prefix", "direct", "excess", "refined", "gain"
            );
            println!("{}", "-".repeat(64));
            for planes in 1..=PMAX {
                let fitted = fit(planes);
                let prefix = (prefix_error(&weights, &master, planes) / energy).sqrt();
                let direct = (prefix_error(&weights, &fitted, planes) / energy).sqrt();
                let refined = (refined_error(&weights, &fitted, planes) / energy).sqrt();
                println!(
                    "{planes:>8} {prefix:>12.4} {direct:>10.4} {:>8.1}% {refined:>10.4} {:>8.1}%",
                    (prefix / direct - 1.0) * 100.0,
                    (refined / direct - 1.0) * 100.0
                );
            }
            println!();
        }
    }
    println!(
        "`T=4 prefix` slices one four-plane ladder fit; `direct` fits the ladder at that count. For\n\
         comparison, the same slice of SALT V2's free-scale joint fit costs +60% at one plane and\n\
         +141% at two on this down_proj."
    );
}
