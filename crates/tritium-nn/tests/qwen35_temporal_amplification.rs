//! **Does ternary error in the recurrent path compound with position? (H4, first-principles plan)**
//!
//! PrismML kept 26.2M parameters of Bonsai 2 in fp — the DeltaNet gates (`in_proj_a/b`), the
//! causal conv and the decay parameters — and reports that "long context reasoning and coding
//! deteriorate first". Both point at the same mechanism: a per-token error in a *recurrent* state
//! is carried into every later token, where an error in an MLP is not. If that is real, WikiText
//! perplexity over short windows hides exactly the failure that matters, and a ternary conversion
//! (Objective A) needs a treatment for the recurrent path that the rest of the model does not.
//!
//! The measurement: ternarize ONE tensor family at a time (weights only, geometric ladder, g128,
//! group-Hadamard basis, fp activations — the published SmolLM2 basis) and record the excess NLL
//! over fp **per position bucket**, paired on the same tokens. A recurrent amplification shows up as
//! excess that GROWS with position; a family whose excess is flat is position-neutral.
//!
//! Controls:
//! - `gates` (DeltaNet `in_proj_a` + `in_proj_b`, 0.59M params) is matched by `mlp rows`, the same
//!   number of `gate_proj` rows in the same 18 layers — equal parameter count, non-recurrent.
//! - `deltanet proj` (qkv, z, out) is matched by `mlp (linear layers)`, the MLPs of the same layers.
//! - `full attn` is the softmax-attention reference: it also sees all positions, but carries no
//!   recurrent state.
//!
//! # Measured 2026-09-23 — Qwen3.5-0.8B-Base, T=2, 4 × 2048 WikiText-2 test tokens
//!
//! ```text
//! arm                        params      pos≥0   pos≥64  pos≥256  pos≥512 pos≥1024
//! fp NLL                          —     3.8667   2.9892   2.8373   2.5336   2.5808
//! Δ gates                    589824    +0.0062  -0.0003  +0.0009  +0.0006  +0.0011
//! Δ mlp rows (control)       589824    -0.0019  +0.0002  +0.0003  +0.0016  +0.0010
//! Δ deltanet proj         188743680    +0.1661  +0.1175  +0.1169  +0.1369  +0.1380
//! Δ mlp (linear layers)   198180864    +0.1380  +0.1286  +0.0743  +0.1121  +0.1063
//! Δ full attn              44040192    +0.1018  +0.0507  +0.0407  +0.0451  +0.0387
//! Δ all projections       497614848    +0.5127  +0.3937  +0.3698  +0.3930  +0.3980
//! ```
//!
//! **No recurrent amplification within 2k tokens.** The DeltaNet gates, which PrismML keeps in fp,
//! cost ~0.001 nats at T=2, identical to their matched MLP rows, and flat with position. The
//! DeltaNet projections rise from +0.117 to +0.138 past position 512 (+18%), but the matched MLP
//! control moves by as much in both directions (0.074..0.129). The quadratic-validity result (H1)
//! puts per-bucket sample noise at this order, so the rise is not resolved. Softmax attention's
//! excess FALLS with position (0.10 → 0.04). Every family pays most at positions < 64.
//! Bounds: 2k context, PTQ, one model. PrismML's claim is about long-context reasoning, which this
//! does not reach, so T3 (temporal amplification) is unsupported at ≤ 2k and untested beyond.
//!
//! ```text
//! TRITIUM_QWEN35_DIR=/mnt/4tb/models/qwen3.5-0.8b-base-tritium \
//! TRITIUM_CORPUS=$HOME/.cache/tritium-corpora/wikitext2_400k_32k_qwen35.json \
//!   cargo test -p tritium-nn --release --test qwen35_temporal_amplification -- --ignored --nocapture
//! ```
//! Knobs: `TRITIUM_H4_T` (planes, default 2), `TRITIUM_H4_WINDOW` (default 2048),
//! `TRITIUM_H4_WINDOWS` (default 4), `TRITIUM_H4_ARMS` (comma-separated labels, default all).

use std::path::PathBuf;
use std::time::Instant;

use tritium_nn::Qwen35HfSource;
use tritium_train::ops::ste::{self, RotationPolicy};

const GROUP: usize = 128;
const GRID: usize = 16;
/// Positions `[BUCKETS[i], BUCKETS[i+1])` are pooled; the last bucket is open-ended.
const BUCKETS: [usize; 6] = [0, 64, 256, 512, 1024, 2048];

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn eval_ids() -> Vec<u32> {
    let path =
        std::env::var("TRITIUM_CORPUS").expect("set TRITIUM_CORPUS to a Qwen-tokenized corpus");
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("corpus")).expect("json");
    v["eval_ids"]
        .as_array()
        .expect("eval_ids")
        .iter()
        .map(|x| x.as_u64().expect("id") as u32)
        .collect()
}

/// Which tensors an arm ternarizes, and how many leading rows of each (`None` = all rows).
struct Arm {
    label: &'static str,
    select: fn(&str) -> Option<Option<usize>>,
}

const GATE_ROWS: usize = 32; // in_proj_a (16) + in_proj_b (16) rows, per layer

const ARMS: [Arm; 6] = [
    Arm {
        label: "gates",
        select: |n| {
            (n.ends_with("in_proj_a.weight") || n.ends_with("in_proj_b.weight")).then_some(None)
        },
    },
    Arm {
        label: "mlp rows",
        // Same layers (the linear-attention ones), same parameter count, non-recurrent.
        select: |n| {
            (n.ends_with("mlp.gate_proj.weight") && is_linear_layer(n)).then_some(Some(GATE_ROWS))
        },
    },
    Arm {
        label: "deltanet proj",
        select: |n| {
            (n.ends_with("in_proj_qkv.weight")
                || n.ends_with("in_proj_z.weight")
                || n.ends_with("linear_attn.out_proj.weight"))
            .then_some(None)
        },
    },
    Arm {
        label: "mlp (linear layers)",
        select: |n| (n.contains(".mlp.") && is_linear_layer(n)).then_some(None),
    },
    Arm {
        label: "full attn",
        select: |n| (n.contains(".self_attn.") && n.ends_with("_proj.weight")).then_some(None),
    },
    Arm {
        label: "all projections",
        select: |n| {
            (n.ends_with("_proj.weight")
                || n.ends_with("in_proj_qkv.weight")
                || n.ends_with("in_proj_a.weight")
                || n.ends_with("in_proj_b.weight")
                || n.ends_with("in_proj_z.weight"))
            .then_some(None)
        },
    },
];

/// Qwen3.5-0.8B: every fourth layer (3, 7, …) is full attention, the rest are DeltaNet.
fn is_linear_layer(name: &str) -> bool {
    layer_index(name).is_some_and(|i| i % 4 != 3)
}

fn layer_index(name: &str) -> Option<usize> {
    let rest = name.split(".layers.").nth(1)?;
    rest.split('.').next()?.parse().ok()
}

/// Per-bucket (nll sum, count) for one model over the eval windows.
fn bucket_nll(
    dir: &std::path::Path,
    arm: Option<&Arm>,
    t: usize,
    windows: &[&[u32]],
) -> (Vec<(f64, usize)>, usize) {
    let quantized = std::cell::Cell::new(0usize);
    let transform = |name: &str, shape: &[usize], values: &mut [f32]| {
        let Some(arm) = arm else { return };
        let Some(rows_limit) = (arm.select)(name) else {
            return;
        };
        let [rows, cols] = shape else { return };
        let rows = rows_limit.map_or(*rows, |r| r.min(*rows));
        let slice = &mut values[..rows * cols];
        let q = ste::salt_quantize_forward_grouped_geometric(
            slice,
            rows,
            *cols,
            t,
            GROUP,
            GRID,
            RotationPolicy::Always,
        );
        slice.copy_from_slice(&q);
        quantized.set(quantized.get() + rows * cols);
    };
    let model = Qwen35HfSource::open(dir)
        .expect("open")
        .verify_semantic_identity()
        .expect("verify")
        .load_language_transformed(Box::new(tritium_cpu::CpuBackend::new()), &transform)
        .expect("load");
    let runner = model.runner();
    let mut buckets = vec![(0.0f64, 0usize); BUCKETS.len()];
    for window in windows {
        let mut cache = runner.new_cache(window.len()).expect("cache");
        let out = runner.forward(window, &mut cache).expect("forward");
        for pos in 0..window.len() - 1 {
            let logits = runner.logits_for_row(&out, pos).expect("row logits");
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let lse = f64::from(max)
                + logits
                    .iter()
                    .map(|&l| f64::from(l - max).exp())
                    .sum::<f64>()
                    .ln();
            let nll = lse - f64::from(logits[window[pos + 1] as usize]);
            let b = BUCKETS
                .iter()
                .rposition(|&start| pos >= start)
                .expect("bucket");
            buckets[b].0 += nll;
            buckets[b].1 += 1;
        }
    }
    (buckets, quantized.get())
}

#[test]
#[ignore = "H4: needs Qwen3.5-0.8B-Base and a Qwen-tokenized corpus; run explicitly"]
fn recurrent_error_amplification_by_position() {
    let dir = PathBuf::from(
        std::env::var("TRITIUM_QWEN35_DIR")
            .unwrap_or_else(|_| "/mnt/4tb/models/qwen3.5-0.8b-base-tritium".into()),
    );
    if !dir.join("config.json").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let t = env_usize("TRITIUM_H4_T", 2);
    let window = env_usize("TRITIUM_H4_WINDOW", 2048);
    let n_windows = env_usize("TRITIUM_H4_WINDOWS", 4);
    let only: Option<Vec<String>> = std::env::var("TRITIUM_H4_ARMS")
        .ok()
        .map(|v| v.split(',').map(|s| s.trim().to_owned()).collect());
    let ids = eval_ids();
    let windows: Vec<&[u32]> = ids.chunks_exact(window).take(n_windows).collect();
    assert_eq!(
        windows.len(),
        n_windows,
        "corpus too short for the windows asked"
    );
    let edges: Vec<usize> = BUCKETS.iter().copied().filter(|&b| b < window).collect();

    let started = Instant::now();
    let (fp, _) = bucket_nll(&dir, None, t, &windows);
    println!(
        "Qwen3.5-0.8B-Base | fp activations | weights ladder T={t} g{GROUP} group-Hadamard | \
         {n_windows} x {window} tokens (WikiText-2 test) | fp pass {:.0}s",
        started.elapsed().as_secs_f64()
    );
    let header: Vec<String> = edges.iter().map(|b| format!("pos≥{b}")).collect();
    println!(
        "{:<22} {:>10}  {}",
        "arm",
        "params",
        header.iter().map(|h| format!("{h:>9}")).collect::<String>()
    );
    println!(
        "{:<22} {:>10}  {}",
        "fp NLL",
        "—",
        edges
            .iter()
            .enumerate()
            .map(|(i, _)| format!("{:>9.4}", fp[i].0 / fp[i].1 as f64))
            .collect::<String>()
    );
    for arm in &ARMS {
        if only
            .as_ref()
            .is_some_and(|o| !o.iter().any(|l| l == arm.label))
        {
            continue;
        }
        let started = Instant::now();
        let (q, params) = bucket_nll(&dir, Some(arm), t, &windows);
        let excess: Vec<f64> = edges
            .iter()
            .enumerate()
            .map(|(i, _)| (q[i].0 - fp[i].0) / fp[i].1 as f64)
            .collect();
        println!(
            "{:<22} {:>10}  {}   ({:.0}s)",
            format!("Δ {}", arm.label),
            params,
            excess
                .iter()
                .map(|e| format!("{e:>+9.4}"))
                .collect::<String>(),
            started.elapsed().as_secs_f64()
        );
    }
    println!(
        "\nΔ = paired excess NLL (nats/token) over fp in each position bucket. Recurrent amplification\n\
         = Δ rising with position for a DeltaNet family while its matched control stays flat."
    );
}
