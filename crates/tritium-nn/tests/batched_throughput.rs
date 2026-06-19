//! Batched M=N decode throughput (Track-2 perf, GPU) — gated, run explicitly.
//!
//! Measures aggregate decode tokens/sec as a function of the concurrent-sequence
//! count `N`, driving [`CudaDecodeModel::decode_batch`] directly. M=1 decode is
//! occupancy-bound (~19% util on a 4090), so a single stream leaves the card
//! starved; this quantifies how much of the memory roofline batching reclaims —
//! the headroom asserted but never measured through v0.3.7.
//!
//! The KV arena scales `N × max_ctx`, so the context is shrunk to a small window
//! (we only decode a short burst) to let large `N` fit in 24 GB.
//!
//! Run:
//! ```text
//! cargo test -p tritium-nn --release --features cuda --test batched_throughput \
//!     -- --ignored --nocapture
//! ```
//!
//! ## Measured (RTX 4090, 64 steps, full-logits readback per step)
//!
//! ```text
//!    N   eager tok/s   graph tok/s     g/eager       g/142
//!    1          48.7         107.3       2.20x       0.76x
//!    2          61.5         171.7       2.79x       1.21x
//!    4          70.2         240.8       3.43x       1.70x
//!    8          74.5         298.5       4.01x       2.10x
//!   16          76.8         337.5       4.39x       2.38x
//!   32          74.9         333.3       4.45x       2.35x
//!   64          73.2         355.8       4.86x       2.51x
//! ```
//!
//! **Finding:** `decode_batch` is the *eager* (un-graph-captured) path — its aggregate
//! saturates at ~75 tok/s by N≈8 and flatlines *below* the 142 tok/s M=1 `step_graph`,
//! because per-kernel launch overhead dominates the M=N forward. Graph-capturing that
//! forward ([`CudaDecodeModel::decode_batch_graph`], bit-identical to eager per the
//! `cuda_batch_decode_graph_matches_eager` gate) is a **2.2×–4.9× win** that lifts the
//! aggregate past the 142 single-stream line from N≥2 and to **2.5× (356 tok/s) at
//! N=64**. The graph N=1 (107 tok/s) still trails the M=1 `step_graph` (142) because the
//! batch graph keeps q/k/v and gate/up *unfused*; fusing them (the M=1 graph already
//! does) is the next Track-2 step, alongside on-device sampling to drop the per-step
//! full-logits dtoh (33 MB/step at N=64).

#![cfg(feature = "cuda")]

use std::path::Path;
use std::time::Instant;

use tritium_cuda::{BatchKv, CudaDecodeModel};
use tritium_nn::ModelRunner;

const GGUF_PATH: &str = "/home/brianklam/.cache/tritium-models/bitnet-2b4t-gguf/ggml-model-i2_s.gguf";

/// Small KV window: we decode `WARM + STEPS` (~72) positions, so 512 is ample and
/// keeps `N × max_ctx` arenas inside 24 GB for large `N`.
const CTX: u32 = 512;
const WARM: usize = 8;
const STEPS: usize = 64;
const NS: &[usize] = &[1, 2, 4, 8, 16, 32, 64];

fn load_cuda() -> Option<ModelRunner> {
    if !Path::new(GGUF_PATH).exists() {
        eprintln!("skipping: {GGUF_PATH} absent");
        return None;
    }
    let bytes = std::fs::read(GGUF_PATH).ok()?;
    let init = tritium_runtime::BACKENDS
        .iter()
        .find(|e| e.name == "cuda")
        .map(|e| e.init)?;
    let backend = match init() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping: cuda backend init failed ({e}); no device?");
            return None;
        }
    };
    let file = tritium_format::read_gguf(&bytes).expect("parse gguf");
    Some(ModelRunner::load(&file, &bytes, backend).expect("load model"))
}

#[test]
#[ignore = "GPU throughput measurement; run explicitly with --features cuda --ignored"]
fn batched_decode_throughput() {
    let mut runner = match load_cuda() {
        Some(r) => r,
        None => return,
    };
    // Shrink the context BEFORE the resident decoder builds, so `new_batch(N)`
    // allocates a small `N × CTX` KV arena (the full 4096 ctx would OOM at large N).
    runner.config.n_ctx = CTX;

    let model = match runner.resident_cuda().expect("resident build") {
        Some(m) => m,
        None => {
            eprintln!("skipping: backend is not CUDA (no resident decoder)");
            return;
        }
    };

    let tok = 791u32; // an arbitrary valid token id; values don't affect timing
    println!("\nBatched M=N decode throughput (RTX 4090, {STEPS} steps):");
    println!("  M=1 single-stream baseline ≈ 142 tok/s (occupancy-bound, ~19% util)\n");
    println!(
        "  {:>4}  {:>12}  {:>12}  {:>10}  {:>10}",
        "N", "eager tok/s", "graph tok/s", "g/eager", "g/142"
    );

    // Time `STEPS` decodes of a fresh `n`-sequence batch through `step_fn`, returning
    // aggregate tokens/sec (n·STEPS / elapsed). `None` if the batch can't be allocated.
    let bench = |model: &mut CudaDecodeModel,
                 n: usize,
                 step: &dyn Fn(&mut CudaDecodeModel, &mut BatchKv, &[u32])|
     -> Option<f64> {
        let mut batch = match model.new_batch(n) {
            Ok(b) => b,
            Err(e) => {
                println!("  N={n}: new_batch failed ({e}) — likely VRAM cap, stopping");
                return None;
            }
        };
        let tokens = vec![tok; n];
        for _ in 0..WARM {
            step(model, &mut batch, &tokens);
        }
        let start = Instant::now();
        for _ in 0..STEPS {
            step(model, &mut batch, &tokens);
        }
        let secs = start.elapsed().as_secs_f64();
        Some((n * STEPS) as f64 / secs)
    };

    for &n in NS {
        let eager = bench(model, n, &|m, b, t| {
            m.decode_batch(b, t).expect("eager decode");
        });
        let Some(eager) = eager else { break };
        let graph = bench(model, n, &|m, b, t| {
            m.decode_batch_graph(b, t).expect("graph decode");
        });
        let Some(graph) = graph else { break };
        let vs_eager = graph / eager;
        let vs_142 = graph / 142.0;
        println!("  {n:>4}  {eager:>12.1}  {graph:>12.1}  {vs_eager:>9.2}x  {vs_142:>9.2}x");
    }
}
