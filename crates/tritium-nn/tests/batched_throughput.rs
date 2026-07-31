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
//! ## Measured (RTX 4090, 64 steps; `argmax` = tiled LM head)
//!
//! ```text
//!    N   eager tok/s   graph tok/s  argmax tok/s    amax/142
//!    1          52.4         114.9         113.9       0.80x
//!    2          66.2         183.2         195.7       1.38x
//!    4          75.8         260.4         303.9       2.14x
//!    8          81.0         323.0         410.7       2.89x
//!   16          83.5         365.2         483.2       3.40x
//!   32          80.8         337.5         457.5       3.22x
//!   64          75.1         366.0         474.3       3.34x
//! ```
//!
//! **Graph capture is the foundation win:** `decode_batch` (eager) saturates ~80 tok/s by
//! N≈8 — below the 142 M=1 `step_graph` — because per-kernel launch overhead dominates the
//! M=N forward. Graph-capturing it ([`CudaDecodeModel::decode_batch_graph`], bit-identical
//! per `cuda_batch_decode_graph_matches_eager`) clears that, reaching ~366 at N=64.
//!
//! **On-device sampling + a tiled LM head is the serving path.**
//! [`CudaDecodeModel::decode_batch_graph_argmax`] folds the LM head + greedy argmax into the
//! graph (returns N token ids, gated by `cuda_batch_decode_graph_argmax_matches_greedy`).
//! The readback itself was never the bottleneck (~180 MB/s at N=64 vs PCIe ~25 GB/s); the
//! win is the **tiled LM head**. An `nsys` trace at N=64 showed the per-row head
//! (`lm_head_warp_f16`) was **29% of GPU time**, reading the 0.66 GB f16 `token_embd` table
//! *once per row* at ~930 GB/s (≈ the 4090's bandwidth). The tiled head reads each embd row
//! once per 8-row tile (`LMHEAD_ROW_TILE`), cutting that ~8×: **474 tok/s at N=64 = 3.34×
//! the M=1 142**, +38% over the per-row head.
//!
//! **The floor is now the ternary GEMM** (`tq2_0_add_mpgemm_tiled_f32`, 68% of GPU time at
//! N=64, compute-bound). Fusing q/k/v / gate/up trims launches/staging; the big GEMM lever
//! is the deferred IMMA (int8 tensor-core) path.

#![cfg(feature = "cuda")]

use std::path::Path;
use std::time::Instant;

use tritium_cuda::{BatchKv, CudaDecodeModel};
use tritium_nn::ModelRunner;

/// Model cache root: override via `TRITIUM_MODEL_DIR`; default `~/.cache/tritium-models`; tests skip cleanly when absent.
static GGUF_PATH: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    let dir = std::env::var("TRITIUM_MODEL_DIR").unwrap_or_else(|_| {
        format!(
            "{}/.cache/tritium-models",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    format!("{dir}/bitnet-2b4t-gguf/ggml-model-i2_s.gguf")
});

/// Small KV window: we decode `WARM + STEPS` (~72) positions, so 512 is ample and
/// keeps `N × max_ctx` arenas inside 24 GB for large `N`.
const CTX: u32 = 512;
const WARM: usize = 8;
const STEPS: usize = 64;
const NS: &[usize] = &[1, 2, 4, 8, 16, 32, 64];

fn load_cuda() -> Option<ModelRunner> {
    if !Path::new(&*GGUF_PATH).exists() {
        eprintln!("skipping: {} absent", &*GGUF_PATH);
        return None;
    }
    let bytes = std::fs::read(&*GGUF_PATH).ok()?;
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

/// Minimal profiling driver for `nsys`/`ncu`: load the model, build one `N`-sequence batch
/// (`TRITIUM_PROFILE_N`, default 32), warm the graph, then run `TRITIUM_PROFILE_STEPS`
/// (default 8) `decode_batch_graph` steps — nothing else — so the profiler sees only the
/// decode kernels, not the 7×3 sweep. Run under the profiler, e.g.:
/// ```text
/// nsys profile -o /tmp/decode \
///   ./target/release/deps/batched_throughput-<hash> profile_decode_burst --ignored --exact
/// ```
#[test]
#[ignore = "profiling driver for ncu/nsys; run the test binary under the profiler"]
fn profile_decode_burst() {
    let mut runner = match load_cuda() {
        Some(r) => r,
        None => return,
    };
    runner.config.n_ctx = CTX;
    let model = match runner.resident_cuda().expect("resident build") {
        Some(m) => m,
        None => {
            eprintln!("skipping: not a cuda resident");
            return;
        }
    };
    let n: usize = std::env::var("TRITIUM_PROFILE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let steps: usize = std::env::var("TRITIUM_PROFILE_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let mut batch = model.new_batch(n).expect("new_batch");
    let tokens = vec![791u32; n];
    for _ in 0..4 {
        model
            .decode_batch_graph(&mut batch, &tokens)
            .expect("warmup");
    }
    for _ in 0..steps {
        model.decode_batch_graph(&mut batch, &tokens).expect("step");
    }
    println!("profiled {steps} decode_batch_graph steps at N={n}");
}
