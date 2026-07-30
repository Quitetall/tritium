//! Plan 0054 WS-B measurement: relay-basin A/B on real checkpoint tensors.
//!
//! For each sampled G128 group of each sampled tensor, runs `fit_joint_ternary`
//! with basins OFF and ON (same config otherwise) and reports the objective
//! delta distribution plus wall time. This is the pre-grid evidence run — the
//! Stage-7 bracket remains the deciding gate; this reports, it does not decide.
//!
//! Usage:
//!   cargo run --release -p tritium-quantize --example relay_basin_ab -- \
//!     <safetensors-path> <planes> <groups-per-tensor> <max-tensors>

use std::time::Instant;

use tritium_format::SafeTensors;
use tritium_quantize::{JointFitConfig, JointFitMetric, RelayBasins, fit_joint_ternary};

const GROUP: usize = 128;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("safetensors path");
    let planes: usize = args.get(2).map_or(2, |s| s.parse().expect("planes"));
    let groups_per_tensor: usize = args.get(3).map_or(64, |s| s.parse().expect("groups"));
    let max_tensors: usize = args.get(4).map_or(8, |s| s.parse().expect("tensors"));

    let bytes = std::fs::read(path).expect("read safetensors");
    let st = SafeTensors::parse(&bytes).expect("parse safetensors");

    let mut names: Vec<String> = st
        .names()
        .filter(|n| n.contains("proj") || n.contains("mlp"))
        .map(str::to_string)
        .collect();
    names.sort();
    let stride = (names.len() / max_tensors).max(1);
    let picked: Vec<&String> = names.iter().step_by(stride).take(max_tensors).collect();

    let mut wins = 0usize;
    let mut ties = 0usize;
    let mut total = 0usize;
    let mut rel_improvements: Vec<f64> = Vec::new();
    let mut wall_off = 0.0f64;
    let mut wall_on = 0.0f64;

    for name in &picked {
        let data = st.tensor_f32(name).expect("tensor f32");
        let group_stride = (data.len() / GROUP / groups_per_tensor).max(1) * GROUP;
        let mut offset = 0usize;
        let mut taken = 0usize;
        while offset + GROUP <= data.len() && taken < groups_per_tensor {
            let group = &data[offset..offset + GROUP];
            let base = JointFitConfig {
                planes,
                ..JointFitConfig::default()
            };
            let with = JointFitConfig {
                relay_basins: RelayBasins {
                    softened: true,
                    modulated: true,
                },
                ..base
            };
            let t0 = Instant::now();
            let off = fit_joint_ternary(group, JointFitMetric::Identity, base).expect("fit off");
            wall_off += t0.elapsed().as_secs_f64();
            let t0 = Instant::now();
            let on = fit_joint_ternary(group, JointFitMetric::Identity, with).expect("fit on");
            wall_on += t0.elapsed().as_secs_f64();

            assert!(
                on.objective <= off.objective + 1e-12,
                "never-worse violated at {name}:{offset}"
            );
            total += 1;
            if on.objective + 1e-15 < off.objective {
                wins += 1;
                if off.objective > 0.0 {
                    rel_improvements.push(1.0 - on.objective / off.objective);
                }
            } else {
                ties += 1;
            }
            offset += group_stride;
            taken += 1;
        }
    }

    rel_improvements.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = rel_improvements
        .get(rel_improvements.len() / 2)
        .copied()
        .unwrap_or(0.0);
    let max = rel_improvements.last().copied().unwrap_or(0.0);
    println!(
        "tensors={} groups={} planes={}",
        picked.len(),
        total,
        planes
    );
    println!(
        "basin wins={} ({:.1}%) ties={} median-rel-improvement={:.3e} max={:.3e}",
        wins,
        100.0 * wins as f64 / total.max(1) as f64,
        ties,
        median,
        max
    );
    println!(
        "wall: off={:.2}s on={:.2}s overhead={:.2}x",
        wall_off,
        wall_on,
        wall_on / wall_off.max(1e-9)
    );
}
