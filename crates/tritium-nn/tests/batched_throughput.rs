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
//! ## Measured (RTX 4090, 64 steps)
//!
//! ```text
//!    N   eager tok/s   graph tok/s  argmax tok/s    amax/142
//!    1          48.2         105.4         105.1       0.74x
//!    2          60.8         169.1         168.7       1.19x
//!    4          69.3         237.9         238.1       1.68x
//!    8          73.8         292.9         294.2       2.07x
//!   16          76.4         335.0         336.3       2.37x
//!   32          77.4         357.0         359.1       2.53x
//!   64          73.0         349.9         342.9       2.41x
//! ```
//!
//! **Graph capture is the win:** `decode_batch` (eager) saturates ~75 tok/s by N≈8 —
//! below the 142 M=1 `step_graph` — because per-kernel launch overhead dominates the M=N
//! forward. Graph-capturing it ([`CudaDecodeModel::decode_batch_graph`], bit-identical per
//! the `cuda_batch_decode_graph_matches_eager` gate) is **2.2×–4.9×**, past 142 from N≥2 to
//! ~350 (2.5×) at N=64.
//!
//! **On-device sampling is correctness, NOT throughput (measured negative result).**
//! `decode_batch_graph_argmax` folds the LM head + greedy argmax into the graph and returns
//! N token ids (N·4 B) instead of N·vocab·4 B of logits — yet it matches the logits path
//! tok/s at every N. The per-step logits readback was **never the bottleneck**: at N=64,
//! ~5.5 steps/s · 33 MB ≈ 180 MB/s, trivial vs PCIe's ~25 GB/s. It is the right *serving*
//! primitive (gated by `cuda_batch_decode_graph_argmax_matches_greedy`), just not faster.
//!
//! **The real large-N cost is the LM head re-reading the embd table per row.** Both paths
//! run a warp-per-vocab-row head that reads the whole 0.66 GB f16 `token_embd` table *once
//! per row* → ~41 GB/step at N=64, dwarfing the ~0.6 GB ternary forward weights. A
//! table-read-once / tiled LM head (reuse `embd` across the N-row tile) is the next large-N
//! lever; fusing q/k/v / gate/up (the M=1 graph already does) lifts N=1 toward 142.

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
        "  {:>4}  {:>12}  {:>12}  {:>12}  {:>10}",
        "N", "eager tok/s", "graph tok/s", "argmax tok/s", "amax/142"
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
        // On-device greedy: the whole step is one graph replay + an N·4-byte token readback
        // (no per-step logits dtoh) — the serving fast path.
        let argmax = bench(model, n, &|m, b, t| {
            m.decode_batch_graph_argmax(b, t).expect("argmax decode");
        });
        let Some(argmax) = argmax else { break };
        let amax_vs_142 = argmax / 142.0;
        println!("  {n:>4}  {eager:>12.1}  {graph:>12.1}  {argmax:>12.1}  {amax_vs_142:>9.2}x");
    }
}
