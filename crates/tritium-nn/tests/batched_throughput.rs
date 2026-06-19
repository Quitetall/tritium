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
//! ## Measured baseline (RTX 4090, 64 steps, full-logits readback per step)
//!
//! ```text
//!   N    agg tok/s   per-stream   vs N=1
//!   1       43.9        43.9       1.00x
//!   2       54.4        27.2       1.24x
//!   4       64.2        16.1       1.46x
//!   8       73.3         9.2       1.67x
//!  16       75.5         4.7       1.72x
//!  32       74.3         2.3       1.69x
//!  64       75.6         1.2       1.72x
//! ```
//!
//! **Honest finding (refutes the "batching fills the GPU" assertion):**
//! `decode_batch` is the *eager* (un-graph-captured) path — N=1 here is 43.9 tok/s
//! vs **142 tok/s** for the M=1 `step_graph` (the CUDA graph is a ~3.2× win that the
//! batched path discards). Aggregate **saturates at ~75 tok/s by N≈8 and goes flat**,
//! i.e. *below* the 142 single-stream graph path. The M=N decode is bit-exact (the
//! v0.3.7 gate) but **not a throughput win as built** — it needs the same CUDA-graph
//! capture the M=1 path got (+ on-device sampling to avoid the per-step logits dtoh,
//! 33 MB/step at N=64). Tracked as Track-2 perf work.

#![cfg(feature = "cuda")]

use std::path::Path;
use std::time::Instant;

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
    println!("  {:>4}  {:>14}  {:>14}  {:>8}", "N", "agg tok/s", "per-stream", "vs N=1");

    let mut base_per_stream = f64::NAN;
    for &n in NS {
        let mut batch = match model.new_batch(n) {
            Ok(b) => b,
            Err(e) => {
                println!("  N={n}: new_batch failed ({e}) — likely VRAM cap, stopping");
                break;
            }
        };
        let tokens = vec![tok; n];
        for _ in 0..WARM {
            model.decode_batch(&mut batch, &tokens).expect("warmup");
        }
        let start = Instant::now();
        for _ in 0..STEPS {
            model.decode_batch(&mut batch, &tokens).expect("decode");
        }
        let secs = start.elapsed().as_secs_f64();
        let agg = (n * STEPS) as f64 / secs;
        let per_stream = agg / n as f64;
        if n == 1 {
            base_per_stream = per_stream;
        }
        let speedup = agg / (base_per_stream * 1.0); // agg vs the N=1 aggregate
        println!("  {n:>4}  {agg:>14.1}  {per_stream:>14.1}  {speedup:>7.2}x");
    }
}
