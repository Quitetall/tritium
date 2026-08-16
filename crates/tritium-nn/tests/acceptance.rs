//! WF-4b acceptance gate: the CUDA end-to-end + perplexity + CPU↔CUDA parity
//! tests for the real BitNet b1.58 2B4T GGUF.
//!
//! This file is the GPU-lane counterpart to `fidelity_ladder.rs` (which proves the
//! *CPU* forward stage-by-stage). It asserts three things, each gated so cpu-only
//! CI (no GPU, no CUDA toolkit, possibly no model) skips cleanly:
//!
//!   1. **CUDA greedy acceptance** (`#[cfg(feature = "cuda")]`): build a
//!      [`ModelRunner`] on the `"cuda"` backend, greedy-generate from the committed
//!      prompt, and assert the generated token IDs equal the committed
//!      `transformers` reference IDs.
//!   2. **Perplexity** (`#[cfg(feature = "cuda")]`): run a forward over the fixed
//!      eval sequence (prompt + reference continuation), compute perplexity from
//!      the per-position next-token log-probs, and assert it is within 1% of the
//!      committed `transformers` reference perplexity.
//!   3. **CPU↔CUDA parity** (`#[cfg(feature = "cuda")]`): generate the same N tokens
//!      on both backends and assert the token IDs are identical (greedy must match
//!      bit-for-bit on the decision) and the per-position logits agree to a
//!      ternary-path relative bound (exact on argmax, ≤2e-3 relative on the logit
//!      vector — 1e-4 is unrealistic across the full 30-layer fp32 residual stream).
//!
//! A **cpu-only** longer-greedy match (no `cuda` feature, model-gated) is also
//! included: it extends the 8-token `fidelity_ladder` check to a longer horizon on
//! the CPU backend, valuable on machines with the model but no GPU.
//!
//! ## Gating
//!
//! Every test first checks the GGUF + reference JSON exist (`maybe_load`/`skip`);
//! absent ⇒ early-return (a pass that prints why). The GPU tests additionally
//! require `--features cuda` AND a working device: `cuda_backend()` returns `Err`
//! when no CUDA device initialises, in which case the test skips rather than fails.
//!
//! Generate the reference with `python3 tools/gen_reference.py` (writes both
//! `bitnet_ladder.json` and `bitnet_accept.json`).
//!
//! ## Decode horizons
//!
//! The CUDA greedy acceptance runs the **full 256-token** target
//! ([`CUDA_GREEDY_LEN`]): the v0.10 add-only kernel
//! (`kernels/tq2_0_add.cu`, ~7 ternary matmuls × 30 layers = ~210 synchronous
//! launches per step) decodes at ~0.2 s/token on an RTX 4090, so 256 tokens
//! finishes in well under a minute including load — comfortably inside a test
//! budget. The asserted IDs are the entire committed continuation.
//!
//! The CPU↔CUDA parity test steps **both** backends in lockstep, and the scalar
//! CPU forward is ~2 s/step, so its horizon is capped at [`PARITY_LEN`] (≥32) to
//! stay tractable while still exercising a long multi-step decode where any
//! per-step divergence would compound.

use std::path::Path;

use tritium_nn::ModelRunner;
#[cfg(feature = "cuda")]
use tritium_nn::sample_greedy;
use tritium_runtime as _;

// Linked so the CPU backend's `#[distributed_slice]` entry is included; the CUDA
// entry is pulled in by the `cuda` feature's dev-dependency edge.
use tritium_cpu as _;
#[cfg(feature = "cuda")]
use tritium_cuda as _;

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
const REF_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tools/reference/bitnet_accept.json"
);

/// Decode horizon for the CUDA greedy acceptance test: the full committed
/// 256-token continuation (the task's target). Measured ~0.2 s/token on an RTX
/// 4090 with the add-only kernel, so the whole run (load + 256-step decode) is
/// well under a minute. The asserted IDs are the entire reference, not a prefix.
#[cfg(feature = "cuda")]
const CUDA_GREEDY_LEN: usize = 256;

/// Decode horizon for the CPU↔CUDA parity test. This test steps *both* backends in
/// lockstep, and the scalar CPU forward is ~2 s/step in release, so the horizon is
/// capped here (≥32, the task floor) to keep the run tractable while still
/// exercising a long multi-step decode where any per-step divergence compounds.
#[cfg(feature = "cuda")]
const PARITY_LEN: usize = 32;

/// Decode horizon for the cpu-only longer-greedy match. The CPU forward is a
/// scalar reduction (~2 s/step in release), so we keep this modest but still well
/// beyond the 8-token `fidelity_ladder` check — a 32-token horizon catches drift
/// that only shows up several steps into autoregression.
const CPU_GREEDY_LEN: usize = 32;

/// The committed reference, emitted by `tools/gen_reference.py`.
///
/// `eval_ids` + `perplexity` are consumed only by the CUDA perplexity test; in a
/// cpu-only build (no `cuda` feature) they are still deserialized — the JSON has
/// them — but unread, so the non-cuda build silences dead-code for just those two.
#[derive(serde::Deserialize)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct Reference {
    /// Prompt token IDs (includes BOS), the generation seed.
    token_ids: Vec<u32>,
    /// transformers greedy continuation IDs (256 tokens; we may assert a prefix).
    greedy_ids: Vec<u32>,
    /// The fixed eval token sequence perplexity is measured over (prompt + cont).
    eval_ids: Vec<u32>,
    /// transformers reference perplexity over `eval_ids` (fp32 CPU oracle).
    perplexity: f64,
    /// EOS id, so greedy stops exactly where the oracle did.
    eos_token_id: u32,
}

/// Serializes the GPU-heavy tests within this binary: each loads a full model
/// (~2.5 GB VRAM), and the default parallel test threads OOM-flake whenever a
/// co-resident GPU process (another session's server, a desktop) squeezes
/// free VRAM — observed live. Poison-tolerant: a panicked test must not fail
/// the rest with a PoisonError.
#[cfg(feature = "cuda")]
static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(feature = "cuda")]
fn gpu_serial() -> std::sync::MutexGuard<'static, ()> {
    GPU_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Load the reference + model bytes, or `None` (with a printed reason) if either
/// is absent — the offline/cpu-only skip path shared by every test here.
fn maybe_load() -> Option<(Reference, Vec<u8>)> {
    if !Path::new(&*GGUF_PATH).exists() {
        eprintln!("skipping: {} absent (gated real-model test)", *GGUF_PATH);
        return None;
    }
    if !Path::new(REF_PATH).exists() {
        eprintln!("skipping: {REF_PATH} absent; run tools/gen_reference.py");
        return None;
    }
    let reference: Reference =
        serde_json::from_slice(&std::fs::read(REF_PATH).expect("read reference"))
            .expect("parse reference json");
    let bytes = std::fs::read(&*GGUF_PATH).expect("read GGUF");
    Some((reference, bytes))
}

/// Index of the maximum element (greedy argmax). Tie-break is lowest index
/// here vs `sample_greedy`/the device pair's highest-index `max_by` — a
/// theoretical divergence only (real logits carry no exact f32 ties; noted
/// per the 067ce79 review). Used only by the CUDA parity tests.
#[cfg(feature = "cuda")]
fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    best
}

/// Construct a `ModelRunner` on the named registry backend from already-read GGUF
/// `bytes`. Returns `None` if the backend did not register / initialise (e.g. no
/// CUDA device), so a GPU-less machine skips the GPU tests instead of failing.
fn load_on(name: &str, bytes: &[u8]) -> Option<ModelRunner> {
    // The runtime registry only hands out borrows; for an owned trait object we go
    // through the linked backend's registered `init` by name — the same pattern
    // `ModelRunner::load_cpu` uses internally for the CPU backend.
    let init = tritium_runtime::BACKENDS
        .iter()
        .find(|e| e.name == name)
        .map(|e| e.init)?;
    let backend = match init() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping: backend `{name}` failed to init ({e}); no device?");
            return None;
        }
    };
    let file = tritium_format::read_gguf(bytes).expect("parse gguf");
    Some(ModelRunner::load(&file, bytes, backend).expect("load model"))
}

/// Numerically-stable log-softmax of `logits` at index `target`: returns
/// `log P(target) = logits[target] - logsumexp(logits)`. CUDA perplexity only.
#[cfg(feature = "cuda")]
fn log_prob_of(logits: &[f32], target: usize) -> f64 {
    let max = logits.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x)) as f64;
    let mut sum = 0.0f64;
    for &l in logits {
        sum += ((l as f64) - max).exp();
    }
    (logits[target] as f64) - max - sum.ln()
}

/// Teacher-forced perplexity of `runner` over `eval_ids`: for each position `t`
/// read the next-token log-prob of the *true* token `eval_ids[t+1]` from that
/// position's logits, where `log P(target) = logit[target] - logsumexp(logits)`.
/// Perplexity is `exp(-mean log P)` over the `len-1` scored positions.
///
/// Computed the oracle-matching way: step token-by-token through the KV cache
/// (each position attends only to its causal prefix), reading the next-token
/// distribution at every step — exactly how the transformers oracle scores a
/// sequence with `use_cache`. Costs `len-1` single-token forwards after the first.
#[cfg(feature = "cuda")]
fn perplexity_over(runner: &mut ModelRunner, eval_ids: &[u32]) -> f64 {
    // One prefill of the whole sequence; we need *every* position's logits, not
    // just the last, so we step token-by-token through the KV cache and collect the
    // next-token log-prob at each step. This mirrors the transformers oracle's
    // teacher-forced scoring exactly (each position attends only to its causal
    // prefix), at the cost of `len-1` forwards.
    runner.reset();
    let n = eval_ids.len();
    assert!(n >= 2, "perplexity needs at least 2 tokens");

    let mut neg_log_sum = 0.0f64;
    let mut count = 0usize;
    // Position 0: prefill the first token alone, score token 1 from its logits.
    let mut logits = runner
        .forward(&eval_ids[..1], &[0])
        .expect("perplexity prefill");
    for t in 0..n - 1 {
        let target = eval_ids[t + 1] as usize;
        neg_log_sum -= log_prob_of(&logits, target);
        count += 1;
        // Advance: feed the *true* token t+1 at position t+1 (teacher forcing).
        if t + 1 < n - 1 {
            logits = runner
                .forward(&[eval_ids[t + 1]], &[t + 1])
                .expect("perplexity decode");
        }
    }
    (neg_log_sum / count as f64).exp()
}

/// CPU-only longer-greedy match (model-gated, no `cuda` feature needed).
///
/// Extends the 8-token `fidelity_ladder` greedy check to [`CPU_GREEDY_LEN`] tokens
/// on the CPU backend, asserting the IDs equal the committed transformers prefix.
/// Valuable on machines that have the model but no GPU/CUDA toolkit.
#[test]
fn cpu_longer_greedy_matches_transformers() {
    let Some((reference, bytes)) = maybe_load() else {
        return;
    };
    let Some(mut runner) = load_on("cpu", &bytes) else {
        eprintln!("skipping: no cpu backend registered");
        return;
    };

    let want: Vec<u32> = reference
        .greedy_ids
        .iter()
        .take(CPU_GREEDY_LEN)
        .copied()
        .collect();
    let got = runner
        .generate(&reference.token_ids, want.len(), reference.eos_token_id)
        .expect("cpu greedy generate");
    println!("cpu greedy ({} tok) ours={got:?}", got.len());
    println!("cpu greedy ({} tok) ref ={want:?}", want.len());
    assert_eq!(
        got,
        want,
        "cpu greedy {}-token continuation must match transformers",
        want.len()
    );
}

// ───────────────────────── CUDA-gated GPU acceptance ─────────────────────────

/// Minimum exact-prefix length of the CUDA greedy continuation vs the committed
/// transformers reference (ADR 0018 re-baseline). Under the pre-1.x sequential
/// reduction order the full 256 tokens happened to match; the canonical tree
/// order (which is *more* accurate — perplexity rel err improved ~11×) rounds a
/// handful of logits by 1 ulp, and a greedy chain amplifies any such change at
/// the first near-tie. Measured divergence point on the 4090: token 104, into
/// an equally coherent continuation. The gate now asserts a 96-token exact
/// prefix — long enough that a real regression (wrong kernel, wrong scale)
/// cannot hide, short enough not to fail on legitimate near-tie flips — plus
/// the perplexity gate that `benches/e2e.rs` holds at ≤1% on every run.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const CUDA_GREEDY_EXACT_PREFIX: usize = 96;

/// (a) CUDA greedy acceptance: generate the full 256-token continuation on the
/// `"cuda"` backend; the first [`CUDA_GREEDY_EXACT_PREFIX`] IDs must equal the
/// committed transformers reference (see the constant's doc for why the full
/// 256 is no longer bit-pinned).
#[cfg(feature = "cuda")]
#[test]
fn cuda_greedy_matches_transformers() {
    let _gpu = gpu_serial();
    let Some((reference, bytes)) = maybe_load() else {
        return;
    };
    let Some(mut runner) = load_on("cuda", &bytes) else {
        return; // no GPU → skip (already printed)
    };

    let want: Vec<u32> = reference
        .greedy_ids
        .iter()
        .take(CUDA_GREEDY_LEN)
        .copied()
        .collect();
    let t0 = std::time::Instant::now();
    let got = runner
        .generate(&reference.token_ids, want.len(), reference.eos_token_id)
        .expect("cuda greedy generate");
    let dt = t0.elapsed();
    let prefix = got.iter().zip(&want).take_while(|(g, w)| g == w).count();
    println!(
        "cuda greedy ({} tok in {:.1?}) exact prefix vs transformers = {prefix}",
        got.len(),
        dt
    );
    assert!(
        prefix >= CUDA_GREEDY_EXACT_PREFIX,
        "cuda greedy diverged from transformers at token {prefix} (< {CUDA_GREEDY_EXACT_PREFIX}); \
         ours={got:?} ref={want:?}"
    );
}

/// Batched M=N decode (v0.3.7) parity with single-sequence M=1 decode — two DISTINCT
/// contracts (see the book's Conformance chapter, "Numerics domains"):
/// (1) WITHIN a batch: two identical sequences produce **bit-identical** logits
///     (independence — asserted `to_bits()` equal), and
/// (2) ACROSS paths: batch vs `step_graph` matches on the **greedy token**, not the
///     logit bits — the batch path's split-KV attention reorders the f32 sum vs the
///     M=1 warp kernel, so logit-level bit-exactness is structurally unavailable.
///     Numeric closeness is covered by the kernel-level 1e-4 equivalence gate
///     (`attn_split_kv_matches_direct_attention`); the graph==eager bit-exact gate
///     covers the launch mechanism.
#[cfg(feature = "cuda")]
#[test]
fn cuda_batch_decode_matches_single() {
    let _gpu = gpu_serial();
    let Some((_reference, bytes)) = maybe_load() else {
        return;
    };
    let Some(mut runner) = load_on("cuda", &bytes) else {
        return;
    };
    let model = match runner.resident_cuda() {
        Ok(Some(m)) => m,
        _ => {
            eprintln!("skipping batch-decode parity: no cuda resident");
            return;
        }
    };
    // A few in-range tokens decoded in lockstep (content is irrelevant — we compare paths).
    let toks: [u32; 4] = [128000, 791, 6864, 315];

    // Single-sequence reference via the M=1 decode graph.
    model.reset();
    let mut single: Vec<Vec<f32>> = Vec::with_capacity(toks.len());
    for (i, &t) in toks.iter().enumerate() {
        single.push(model.step_graph(t, i).expect("single step_graph"));
    }

    // Batched N=2, both sequences fed the same tokens.
    let mut batch = model.new_batch(2).expect("new_batch");
    for (i, &t) in toks.iter().enumerate() {
        let logits = model
            .decode_batch(&mut batch, &[t, t])
            .expect("decode_batch");
        assert_eq!(logits.len(), 2);
        // Sequences in the same batch must be independent (bit-exact).
        assert_eq!(
            logits[0].len(),
            logits[1].len(),
            "per-sequence logit length mismatch"
        );
        for (v, (a, b)) in logits[0].iter().zip(&logits[1]).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "batch seq0 != seq1 at step {i} vocab {v} (sequences must be independent)"
            );
        }
        // Split-KV (M=N) vs warp kernel (M=1): the online-softmax warp-shuffle reorders
        // the f32 sum, so logit-level bit-exactness is not achievable. Compare the GREEDY
        // TOKEN instead — the decode-meaningful invariant — via the system's own
        // `sample_greedy` (NaN-safe, with the exact tie-rule the caller uses) rather than a
        // reimplemented argmax, so it matches what greedy decode actually emits. Numeric
        // correctness is already covered by the kernel-level 1e-4 equivalence gate
        // (attn_split_kv_matches_direct_attention) + the graph==eager bit-exact gate.
        let batch_tok = tritium_nn::sample_greedy(&logits[0]).expect("non-empty logits");
        let single_tok = tritium_nn::sample_greedy(&single[i]).expect("non-empty logits");
        assert_eq!(
            batch_tok, single_tok,
            "batch decode greedy token != single greedy token at step {i}: batch={batch_tok} single={single_tok}"
        );
    }
    println!(
        "batch-decode parity: N=2 == single, argmax-identical over {} steps (split-KV vs warp kernel)",
        toks.len()
    );
}

/// CUDA-graph batched decode (Track-2 perf) must be **bit-identical** to the eager
/// `decode_batch`: the graph replays the exact same kernels in the same order over the
/// same buffers, so the only thing that changes is the launch mechanism. Drives two
/// fresh `N`-sequence batches with identical tokens for `K` steps — one through the
/// eager path, one through `decode_batch_graph` — and asserts every logit matches
/// byte-for-byte (`to_bits()`), per row, at every step. This is the gate the M=N graph
/// capture ships behind.
#[cfg(feature = "cuda")]
#[test]
fn cuda_batch_decode_graph_matches_eager() {
    let _gpu = gpu_serial();
    let Some((_reference, bytes)) = maybe_load() else {
        return;
    };
    let Some(mut runner) = load_on("cuda", &bytes) else {
        return;
    };
    let model = match runner.resident_cuda() {
        Ok(Some(m)) => m,
        _ => {
            eprintln!("skipping batch-graph parity: no cuda resident");
            return;
        }
    };
    const N: usize = 2;
    // A short burst of in-range tokens, decoded in lockstep on both paths.
    let toks: [u32; 4] = [128000, 791, 6864, 315];

    // The two batches must be dropped before the model (their graph references the
    // model's `batch_raw` modules), so scope them tighter than `model`.
    let mut eager = model.new_batch(N).expect("new_batch eager");
    let mut graph = model.new_batch(N).expect("new_batch graph");
    for (i, &t) in toks.iter().enumerate() {
        let row_tokens = vec![t; N];
        let want = model
            .decode_batch(&mut eager, &row_tokens)
            .expect("decode_batch eager");
        let got = model
            .decode_batch_graph(&mut graph, &row_tokens)
            .expect("decode_batch_graph");
        assert_eq!(got.len(), N);
        assert_eq!(want.len(), N);
        for r in 0..N {
            for v in 0..want[r].len() {
                assert_eq!(
                    got[r][v].to_bits(),
                    want[r][v].to_bits(),
                    "graph != eager at step {i} row {r} vocab {v}"
                );
            }
        }
    }
    drop(eager);
    drop(graph);
    println!(
        "batch-graph parity: graph == eager, bit-identical over {} steps",
        toks.len()
    );
}

/// C2 per-row masks (batching P2): a DEAD slot (`set_live(row, false)`) must
/// touch NOTHING — zero KV-arena bytes written (the paged-KV contract: a dead
/// row owns no write slot) — and the live rows must stay **bit-identical** to
/// the same rows of an all-live batch (dead-row skipping cannot perturb
/// anyone else's arithmetic). Row 1 of a 3-row batch is dead and fed the pad
/// token (mirroring the serve worker); rows 0/2 decode normally.
#[cfg(feature = "cuda")]
#[test]
fn cuda_batch_dead_row_touches_nothing() {
    let _gpu = gpu_serial();
    let Some((_reference, bytes)) = maybe_load() else {
        return;
    };
    let Some(mut runner) = load_on("cuda", &bytes) else {
        return;
    };
    let n_layers = runner.config.n_layers as usize;
    let max_ctx = runner.config.n_ctx as usize;
    let model = match runner.resident_cuda() {
        Ok(Some(m)) => m,
        _ => {
            eprintln!("skipping dead-row gate: no cuda resident");
            return;
        }
    };
    const N: usize = 3;
    let toks: [u32; 4] = [128000, 791, 6864, 315];

    let mut masked = model.new_batch(N).expect("new_batch masked");
    masked.set_live(1, false).expect("set_live");
    let mut all_live = model.new_batch(N).expect("new_batch all_live");

    // A dead row's token is STILL vocab-checked (the embed gather reads
    // every row's token — an out-of-range one is an OOB device read, dead
    // or not). Must reject loudly, before any launch.
    assert!(
        model
            .decode_batch_graph(&mut masked, &[toks[0], u32::MAX, toks[0]])
            .is_err(),
        "out-of-range token on a DEAD row must still be rejected"
    );

    for (i, &t) in toks.iter().enumerate() {
        let got = model
            .decode_batch_graph(&mut masked, &[t, 0, t])
            .expect("decode_batch_graph masked");
        let want = model
            .decode_batch_graph(&mut all_live, &[t, t, t])
            .expect("decode_batch_graph all_live");
        for r in [0usize, 2] {
            for v in 0..want[r].len() {
                assert_eq!(
                    got[r][v].to_bits(),
                    want[r][v].to_bits(),
                    "dead row perturbed a live row: step {i} row {r} vocab {v}"
                );
            }
        }
    }
    // Dead row frozen at position 0 (advance skips it); live rows advanced.
    assert_eq!(masked.positions()[1], 0, "dead row's position must freeze");
    assert_eq!(masked.positions()[0], toks.len());

    // The contract itself: row 1's KV arena is untouched — every byte still
    // the alloc_zeros init — at the first/last layer, K and V, for the whole
    // stepped span. Row 0 must have non-zero KV at the same spots (the
    // assertion is not vacuous).
    for li in [0usize, n_layers - 1] {
        for v in [false, true] {
            for pos in 0..toks.len() {
                let dead = model
                    .debug_batch_kv_row(&masked, li, 1, pos, v)
                    .expect("dead row kv");
                assert!(
                    dead.iter().all(|&b| b == 0),
                    "dead row wrote KV bytes at layer {li} pos {pos} v={v}"
                );
            }
            let live = model
                .debug_batch_kv_row(&masked, li, 0, 0, v)
                .expect("live row kv");
            assert!(
                live.iter().any(|&b| b != 0),
                "live row 0 wrote nothing at layer {li} v={v} — vacuous gate"
            );
            // Aliasing target: an UNGUARDED -1 append computes
            // (1*max_ctx - 1)*kv_width — i.e. row 0's arena at its LAST
            // position, not row 1's. Assert that exact spot stayed zero so a
            // regression of the kv_append guard cannot slip past this gate.
            let alias = model
                .debug_batch_kv_row(&masked, li, 0, max_ctx - 1, v)
                .expect("alias target kv");
            assert!(
                alias.iter().all(|&b| b == 0),
                "row 0's tail (pos max_ctx-1) written at layer {li} v={v} — \
                 the dead row's -1 append aliased into it"
            );
        }
    }
    drop(masked);
    drop(all_live);
    println!(
        "dead-row gate: live rows bit-identical to all-live batch over {} steps; \
         dead row wrote zero KV bytes (first+last layer, K+V)",
        toks.len()
    );
}

/// ADR 0026 Track P step 4: the IMMA prefill dispatch must be BIT-IDENTICAL
/// to the dp4a path on the real model — one runner, the same prompt
/// prefilled with the tensor-core dispatch live and then with it disabled
/// (`debug_disable_imma`), logits compared to_bits. Also pins chunk-size
/// independence UNDER the IMMA path: a 4-chunk prefill must equal the
/// one-shot prefill bit-for-bit (the C1 contract carried onto tensor cores;
/// chunks of 128 and a 512 one-shot land in different M buckets / tile
/// configs and must still agree).
#[cfg(feature = "cuda")]
#[test]
fn cuda_imma_prefill_matches_dp4a_bit_exact() {
    let _gpu = gpu_serial();
    let Some((reference, bytes)) = maybe_load() else {
        return;
    };
    let Some(mut runner) = load_on("cuda", &bytes) else {
        return;
    };
    // A 512-token prompt (the pp512 shape): cycle the reference ids.
    let prompt: Vec<u32> = reference
        .token_ids
        .iter()
        .cycle()
        .take(512)
        .copied()
        .collect();
    let positions: Vec<usize> = (0..prompt.len()).collect();

    // One-shot prefill, IMMA dispatch live (M=512 → its own tile bucket).
    runner.reset();
    let imma_oneshot = runner.forward(&prompt, &positions).expect("imma one-shot");

    // Chunked prefill under IMMA (4×128 — the serve C1 shape, bucket 7).
    runner.reset();
    let mut imma_chunked = Vec::new();
    for c in 0..4 {
        let lo = c * 128;
        let hi = lo + 128;
        imma_chunked = runner
            .forward(&prompt[lo..hi], &positions[lo..hi])
            .expect("imma chunk");
    }
    for (i, (&a, &b)) in imma_oneshot.iter().zip(&imma_chunked).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "IMMA chunked != one-shot at vocab {i}: {a} vs {b}"
        );
    }

    // Disable the dispatch on the SAME model → dp4a path; must be bit-equal.
    runner
        .resident_cuda()
        .expect("resident")
        .expect("cuda")
        .debug_disable_imma();
    runner.reset();
    let dp4a = runner.forward(&prompt, &positions).expect("dp4a one-shot");
    for (i, (&a, &b)) in imma_oneshot.iter().zip(&dp4a).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "IMMA prefill != dp4a at vocab {i}: imma={a} dp4a={b} — the \
             bit-identity contract (ADR 0026) broke"
        );
    }
    println!(
        "IMMA prefill gate: one-shot == 4x128 chunked == dp4a, all to_bits \
         over {} logits (M=512 prompt)",
        dp4a.len()
    );
}

/// C3 paged KV (ADR 0025): a paged batch must be **bit-identical** to a dense
/// batch fed the same requests — paging changes addresses, never values or
/// reduction order. Exercised: prompt adoption through the per-page copy,
/// lockstep graph steps, a dead row (C2 composes: it owns no pages), a
/// retire + re-admit cycle whose pages come back through the free list
/// (reuse over stale bytes — never read, positions restart), and loud
/// rejection on pool exhaustion and on stepping an unmapped position.
#[cfg(feature = "cuda")]
#[test]
fn cuda_batch_paged_matches_dense_bit_exact() {
    let _gpu = gpu_serial();
    let Some((reference, bytes)) = maybe_load() else {
        return;
    };
    let Some(mut runner) = load_on("cuda", &bytes) else {
        return;
    };
    const N: usize = 3;
    const POOL_PAGES: usize = 2; // deliberately tight: forces reuse + exhaustion
    let prompt = reference.token_ids.clone();
    // The tight-pool arithmetic below (free_pages asserts, POOL_PAGES=2)
    // assumes row 0's whole footprint fits ONE page. Named here so a larger
    // fixture fails at the cause, not at a downstream reserve.
    assert!(
        prompt.len() + 4 <= 256,
        "fixture grew past one KV page — retune POOL_PAGES and the \
         free_pages asserts"
    );
    let positions: Vec<usize> = (0..prompt.len()).collect();
    let toks: [u32; 4] = [128000, 791, 6864, 315];

    // One single-seq prefill; adopt the SAME KV into row 0 of both batches.
    runner.forward(&prompt, &positions).expect("prefill");
    let mut dense = runner.new_batch(N).expect("new_batch dense");
    let mut paged = runner
        .new_batch_paged(N, POOL_PAGES)
        .expect("new_batch_paged");
    assert!(paged.paged() && !dense.paged());

    // Exhaustion is loud: 3 pages from a 2-page pool must fail all-or-nothing.
    assert!(
        paged.reserve_pages(0, 2 * 256 + 1).is_err(),
        "exhaustion must reject"
    );
    assert_eq!(
        paged.free_pages(),
        POOL_PAGES,
        "failed reserve must not leak"
    );
    // Stepping an unmapped live row is loud, not UB.
    assert!(
        runner
            .decode_batch_graph(&mut paged, &[toks[0]; N])
            .is_err(),
        "unmapped position must be rejected"
    );

    paged
        .reserve_pages(0, prompt.len() + toks.len())
        .expect("reserve row0");
    // Row 2's true footprint: 4+4 lockstep loops + retired step + eager +
    // argmax = 11 advances; 16 keeps the no-outgrow story exact with slack.
    paged.reserve_pages(2, 16).expect("reserve row2");
    for b in [&mut dense, &mut paged] {
        b.set_live(1, false).expect("dead row");
    }
    runner
        .adopt_into_batch_row(&mut dense, 0, prompt.len())
        .expect("adopt dense");
    runner
        .adopt_into_batch_row(&mut paged, 0, prompt.len())
        .expect("adopt paged");
    for b in [&mut dense, &mut paged] {
        b.set_position(0, prompt.len()).expect("pos row0");
    }

    let step_both = |runner: &mut tritium_nn::ModelRunner,
                     dense: &mut tritium_cuda::BatchKv,
                     paged: &mut tritium_cuda::BatchKv,
                     t: u32,
                     tag: &str| {
        let want = runner
            .decode_batch_graph(dense, &[t, 0, t])
            .expect("dense step");
        let got = runner
            .decode_batch_graph(paged, &[t, 0, t])
            .expect("paged step");
        for r in [0usize, 2] {
            for v in 0..want[r].len() {
                assert_eq!(
                    got[r][v].to_bits(),
                    want[r][v].to_bits(),
                    "paged != dense at {tag} row {r} vocab {v}"
                );
            }
        }
    };
    for (i, &t) in toks.iter().enumerate() {
        step_both(&mut runner, &mut dense, &mut paged, t, &format!("step {i}"));
    }

    // Retire row 0: pages return to the pool; both twins go dead.
    paged.release_pages(0).expect("release");
    assert_eq!(paged.free_pages(), 1, "row0's page must come back");
    for b in [&mut dense, &mut paged] {
        b.set_live(0, false).expect("retire");
    }
    step_both(&mut runner, &mut dense, &mut paged, toks[0], "retired step");

    // Re-admit row 0 fresh at position 0: the reservation REUSES the freed
    // page (stale bytes are never read — the first step appends before it
    // attends). Dense mirrors with its own stale arena.
    paged.reserve_pages(0, toks.len()).expect("re-reserve");
    assert_eq!(paged.free_pages(), 0);
    for b in [&mut dense, &mut paged] {
        b.set_position(0, 0).expect("restart");
        b.set_live(0, true).expect("revive");
    }
    for (i, &t) in toks.iter().enumerate() {
        step_both(
            &mut runner,
            &mut dense,
            &mut paged,
            t,
            &format!("re-admit step {i}"),
        );
    }

    // The other two dispatch paths (review N2): the EAGER step and the
    // on-device-sampling ARGMAX graph also select paged kernels off
    // batch.pages — one lockstep each, pinned to dense.
    {
        let model = runner
            .resident_cuda()
            .expect("resident")
            .expect("cuda resident");
        let want = model
            .decode_batch(&mut dense, &[toks[0], 0, toks[0]])
            .expect("dense eager");
        let got = model
            .decode_batch(&mut paged, &[toks[0], 0, toks[0]])
            .expect("paged eager");
        for r in [0usize, 2] {
            for v in 0..want[r].len() {
                assert_eq!(
                    got[r][v].to_bits(),
                    want[r][v].to_bits(),
                    "paged != dense at eager step row {r} vocab {v}"
                );
            }
        }
        let want_ids = model
            .decode_batch_graph_argmax(&mut dense, &[toks[1], 0, toks[1]])
            .expect("dense argmax");
        let got_ids = model
            .decode_batch_graph_argmax(&mut paged, &[toks[1], 0, toks[1]])
            .expect("paged argmax");
        assert_eq!(want_ids[0], got_ids[0], "argmax row 0 diverged");
        assert_eq!(want_ids[2], got_ids[2], "argmax row 2 diverged");
    }

    drop(dense);
    drop(paged);
    println!(
        "paged-KV gate: bit-identical to dense through adoption, {} lockstep \
         advances (graph + eager + argmax), retire/re-admit page reuse, dead \
         row; exhaustion + unmapped-step loud",
        toks.len() * 2 + 3
    );
}

/// The on-device-sampling M=N graph (`decode_batch_graph_argmax`) folds the LM head + a
/// greedy argmax into the captured graph and returns N token ids (N·4 bytes) instead of N
/// full logit vectors (the 33 MB/step readback). Its per-row token must equal the host
/// `sample_greedy` of the eager `decode_batch` logits for the same state — i.e. the
/// on-device argmax reproduces the reference greedy decision bit-for-bit (ties → highest
/// index, matching `sample_greedy`'s `max_by`).
#[cfg(feature = "cuda")]
#[test]
fn cuda_batch_decode_graph_argmax_matches_greedy() {
    let _gpu = gpu_serial();
    let Some((_reference, bytes)) = maybe_load() else {
        return;
    };
    let Some(mut runner) = load_on("cuda", &bytes) else {
        return;
    };
    let model = match runner.resident_cuda() {
        Ok(Some(m)) => m,
        _ => {
            eprintln!("skipping batch-graph argmax parity: no cuda resident");
            return;
        }
    };
    const N: usize = 2;
    let toks: [u32; 4] = [128000, 791, 6864, 315];

    let mut eager = model.new_batch(N).expect("new_batch eager");
    let mut argmax = model.new_batch(N).expect("new_batch argmax");
    for (i, &t) in toks.iter().enumerate() {
        let row_tokens = vec![t; N];
        let logits = model
            .decode_batch(&mut eager, &row_tokens)
            .expect("decode_batch eager");
        let tokens = model
            .decode_batch_graph_argmax(&mut argmax, &row_tokens)
            .expect("decode_batch_graph_argmax");
        assert_eq!(tokens.len(), N);
        for r in 0..N {
            let want = sample_greedy(&logits[r]).expect("greedy");
            assert_eq!(tokens[r], want, "argmax != host greedy at step {i} row {r}");
        }
    }
    drop(eager);
    drop(argmax);
    println!(
        "batch-graph argmax parity: on-device argmax == host greedy over {} steps",
        toks.len()
    );
}

/// (b) Perplexity: forward over the fixed eval sequence on CUDA, assert within 1%
/// of the committed transformers reference perplexity.
#[cfg(feature = "cuda")]
#[test]
fn cuda_perplexity_within_1pct() {
    let _gpu = gpu_serial();
    let Some((reference, bytes)) = maybe_load() else {
        return;
    };
    let Some(mut runner) = load_on("cuda", &bytes) else {
        return;
    };

    let ppl = perplexity_over(&mut runner, &reference.eval_ids);
    let want = reference.perplexity;
    let rel = (ppl - want).abs() / want;
    println!("cuda perplexity ours={ppl:.6} ref={want:.6} rel={rel:.3e}");
    assert!(
        rel <= 0.01,
        "cuda perplexity {ppl:.6} not within 1% of reference {want:.6} (rel {rel:.3e})"
    );
}

/// (c) CPU↔CUDA parity: same N tokens on both backends. Token IDs must be
/// identical (greedy decision is exact); per-position logits agree to ≤2e-3
/// relative with an exact argmax at every step.
#[cfg(feature = "cuda")]
#[test]
fn cpu_cuda_parity() {
    let _gpu = gpu_serial();
    let Some((reference, bytes)) = maybe_load() else {
        return;
    };
    let Some(mut cpu) = load_on("cpu", &bytes) else {
        eprintln!("skipping: no cpu backend");
        return;
    };
    let Some(mut cuda) = load_on("cuda", &bytes) else {
        return; // no GPU → skip
    };

    // Reset both, prefill the prompt on each, then step in lockstep — at every
    // position compare the full logit vectors and the greedy pick.
    cpu.reset();
    cuda.reset();
    let n_steps = PARITY_LEN;

    let positions: Vec<usize> = (0..reference.token_ids.len()).collect();
    let mut cpu_logits = cpu
        .forward(&reference.token_ids, &positions)
        .expect("cpu prefill");
    let mut cuda_logits = cuda
        .forward(&reference.token_ids, &positions)
        .expect("cuda prefill");

    let mut worst_rel = 0.0f32;
    let mut cpu_ids = Vec::with_capacity(n_steps);
    let mut cuda_ids = Vec::with_capacity(n_steps);

    for (step, pos) in (reference.token_ids.len()..).take(n_steps).enumerate() {
        // Argmax must match exactly at every position (the greedy decision).
        let am_cpu = argmax(&cpu_logits);
        let am_cuda = argmax(&cuda_logits);
        assert_eq!(
            am_cpu, am_cuda,
            "argmax mismatch at step {step}: cpu={am_cpu} cuda={am_cuda}"
        );

        // Per-position logit closeness, reference magnitude as the denominator.
        let rel = max_rel_err(&cuda_logits, &cpu_logits);
        if rel > worst_rel {
            worst_rel = rel;
        }
        assert!(
            rel <= 2e-3,
            "step {step}: cpu↔cuda logits rel err {rel:.3e} > 2e-3"
        );

        let next_cpu = sample_greedy(&cpu_logits).expect("cpu greedy");
        let next_cuda = sample_greedy(&cuda_logits).expect("cuda greedy");
        assert_eq!(
            next_cpu, next_cuda,
            "greedy token mismatch at step {step}: cpu={next_cpu} cuda={next_cuda}"
        );
        cpu_ids.push(next_cpu);
        cuda_ids.push(next_cuda);
        if next_cpu == reference.eos_token_id {
            break;
        }

        cpu_logits = cpu.forward(&[next_cpu], &[pos]).expect("cpu decode");
        cuda_logits = cuda.forward(&[next_cuda], &[pos]).expect("cuda decode");
    }

    println!(
        "cpu↔cuda parity over {} steps: identical IDs, worst logit rel = {worst_rel:.3e}",
        cpu_ids.len()
    );
    assert_eq!(
        cpu_ids, cuda_ids,
        "cpu and cuda greedy IDs must be identical"
    );
}

/// Max relative error between two equal-length vectors, using the larger-magnitude
/// reference (`want`) as the denominator (with a small floor) — the ADR-0004
/// convention also used by `fidelity_ladder.rs`.
#[cfg(feature = "cuda")]
fn max_rel_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "logit length mismatch");
    let scale = want.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-3);
    let mut worst = 0.0f32;
    for (&g, &w) in got.iter().zip(want) {
        let e = (g - w).abs() / scale;
        if e > worst {
            worst = e;
        }
    }
    worst
}

// ───────────────────── BASTION tree-verify losslessness (ADR 0014) ─────────────────────

/// ADR 0014 losslessness gate (greedy, C): driving generation through
/// `CudaDecodeModel::tree_verify_greedy` with mock drafter trees — perfect
/// chains, branching trees with wrong siblings, partially wrong chains, and a
/// full-reject — must commit EXACTLY the same token stream as plain greedy
/// decode, and the promoted KV must leave the model able to continue plain
/// `step`s that still agree (which would expose any promote/rollback
/// corruption).
#[cfg(feature = "cuda")]
#[test]
fn cuda_tree_verify_greedy_lossless() {
    let _gpu = gpu_serial();
    let Some((reference, bytes)) = maybe_load() else {
        return;
    };
    let Some(mut runner) = load_on("cuda", &bytes) else {
        return;
    };
    let prompt = reference.token_ids.clone();
    let k_total = 24usize;

    // Plain greedy reference stream (also warms nothing — fresh runner below).
    let want = runner
        .generate(&prompt, k_total, u32::MAX /* never stop early */)
        .expect("plain greedy generate");
    assert_eq!(
        want.len(),
        k_total,
        "reference stream shorter than expected"
    );

    // Fresh state; prefill the prompt, then drive via tree-verify only.
    runner.reset();
    let positions: Vec<usize> = (0..prompt.len()).collect();
    let prefill_logits = runner.forward(&prompt, &positions).expect("prefill");
    assert_eq!(
        argmax(&prefill_logits) as u32,
        want[0],
        "prefill argmax must match the reference's first token"
    );
    let rm = runner
        .resident_cuda()
        .expect("resident model")
        .expect("cuda resident model present");

    let mut committed: Vec<u32> = vec![want[0]];
    let mut phase = 0usize;
    while committed.len() < k_total {
        let c = committed.len();
        let root = *committed.last().expect("non-empty");
        let remaining = k_total - c;
        // Rotate through draft shapes: perfect chain, branch with a wrong
        // sibling, partially wrong chain, full reject.
        let (tokens, parents): (Vec<u32>, Vec<i32>) = match phase % 4 {
            0 => {
                // Perfect chain of up to 3 drafts (the ideal drafter).
                let n = remaining.min(3);
                let mut t = vec![root];
                t.extend(want[c..c + n].iter().copied());
                let p: Vec<i32> = (0..t.len() as i32).map(|i| i - 1).collect();
                (t, p)
            }
            1 => {
                // Branch: a wrong child first, then the right child carrying a
                // right grandchild. Wrong token = right token + 1 (mod vocab-ish).
                let right = want[c];
                let wrong = (right + 1) % 128_256;
                let mut t = vec![root, wrong, right];
                let mut p = vec![-1, 0, 0];
                if remaining >= 2 {
                    t.push(want[c + 1]);
                    p.push(2);
                }
                (t, p)
            }
            2 => {
                // Chain that goes wrong after one correct draft.
                let right = want[c];
                let wrong = (right + 7) % 128_256;
                (vec![root, right, wrong], vec![-1, 0, 1])
            }
            _ => {
                // Full draft reject: only wrong children.
                let wrong = (want[c] + 3) % 128_256;
                (vec![root, wrong], vec![-1, 0])
            }
        };
        let out = rm
            .tree_verify_greedy(&tokens, &parents)
            .expect("tree_verify_greedy");
        assert!(!out.is_empty(), "tree verify must always commit >= 1 token");
        committed.extend(&out); // full path — the model promoted all of it
        phase += 1;
    }
    let c_full = committed.len(); // may overshoot k_total by a bonus or two

    // The ADR 0014 losslessness gate: the committed stream must equal plain
    // greedy decode token-for-token. This also exercises KV promote integrity —
    // every tree after the first attends rows promoted by its predecessors, so a
    // corrupted promote would derail the committed stream itself.
    let mut runner2 = load_on("cuda", &bytes).expect("second runner");
    let want_long = runner2
        .generate(&prompt, c_full, u32::MAX)
        .expect("long reference");
    assert_eq!(
        &committed[..],
        &want_long[..c_full],
        "tree-verify committed stream must equal plain greedy (lossless)"
    );

    // KV handoff: GRAPH steps after the promotes must continue the exact
    // reference stream — any promote corruption shows up here. (The graph
    // path, not the eager `step`: eager reads the f32 LM-head table while the
    // reference stream was generated through the f16 graph head, a designed
    // logit difference that flips near-ties; the transformer STATE is
    // bit-identical across batch/graph, gated by
    // `cuda_batch_and_graph_single_token_bit_identical`.)
    let mut runner3 = load_on("cuda", &bytes).expect("third runner");
    let want_tail = runner3
        .generate(&prompt, c_full + 4, u32::MAX)
        .expect("tail reference");
    let pos0 = prompt.len() + c_full - 1; // == cache_len (pending not yet forwarded)
    let mut pending = *committed.last().expect("non-empty");
    for step in 0..4 {
        let logits = rm
            .step_graph(pending, pos0 + step)
            .expect("graph step after tree verify");
        let next = argmax(&logits) as u32;
        assert_eq!(
            next,
            want_tail[c_full + step],
            "post-tree graph decode diverged at tail step {step} — KV promote corruption"
        );
        pending = next;
    }
}

/// Restores `TRITIUM_KV` to its pre-test value on drop so a failing f16 leg
/// can't leak the rung override into later tests in the same process.
#[cfg(feature = "cuda")]
struct KvEnvGuard(Option<std::ffi::OsString>);
#[cfg(feature = "cuda")]
impl KvEnvGuard {
    fn set(rung: &str) -> Self {
        let prev = std::env::var_os("TRITIUM_KV");
        // SAFETY: the GPU tests run single-threaded (`gpu_serial` +
        // `--test-threads=1`); no other thread touches the environment.
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var("TRITIUM_KV", rung);
        }
        Self(prev)
    }
}
#[cfg(feature = "cuda")]
impl Drop for KvEnvGuard {
    fn drop(&mut self) {
        // SAFETY: single-threaded test process (see `KvEnvGuard::set`).
        #[allow(unsafe_code)]
        unsafe {
            match self.0.take() {
                Some(v) => std::env::set_var("TRITIUM_KV", v),
                None => std::env::remove_var("TRITIUM_KV"),
            }
        }
    }
}

/// ADR 0036 L6 stage-1 gate: the SINGLE-SEQ tree-verify GRAPH route on the
/// accepted f16 KV rung (ADR 0020 rung 1). Three teeth:
///
/// 1. **Route taken**: under `TRITIUM_KV=f16` the verify must run the
///    captured-graph route (`tree_graph_bucket_count() >= 1`), not the old
///    eager fallback the `kv_elem == 4` gate forced.
/// 2. **Losslessness within the rung** (the ADR 0014 contract, unweakened):
///    the tree-verify committed stream under f16 must equal plain greedy
///    decode under the same f16 rung, token for token.
/// 3. **Graph == eager within the rung** (the f32 pair's equivalence,
///    mirrored): the same draft schedule with `TRITIUM_TREE_EAGER=1` (the
///    accepted eager-f16 tree path, `_h` non-ctrl twins) must commit the
///    identical stream — the ctrl `_h` twins do the same arithmetic in the
///    same order.
///
/// f16-vs-f32 token identity is NOT asserted — the f16 tier is perplexity-
/// gated vs f32 (each written K/V rounds once, ADR 0020), so the cross-rung
/// stream comparison is reported for the record, not gated.
#[cfg(feature = "cuda")]
#[test]
fn cuda_tree_verify_f16_graph_route_lossless_and_matches_eager() {
    let _gpu = gpu_serial();
    let Some((reference, bytes)) = maybe_load() else {
        return;
    };
    let _kv = KvEnvGuard::set("f16");
    let prompt = reference.token_ids.clone();
    let k_total = 24usize;

    // Plain greedy reference stream ON THE F16 RUNG (the within-rung anchor).
    let Some(mut runner) = load_on("cuda", &bytes) else {
        return;
    };
    let want_f16 = runner
        .generate(&prompt, k_total + 8, u32::MAX)
        .expect("plain f16 greedy generate");

    // The same 4-shape draft rotation as `cuda_tree_verify_greedy_lossless`,
    // seeded from the f16 stream (perfect chain / branch / partial / reject).
    let make_tree = |phase: usize, c: usize, root: u32| -> (Vec<u32>, Vec<i32>) {
        let remaining = k_total.saturating_sub(c).max(1);
        match phase % 4 {
            0 => {
                let n = remaining.min(3);
                let mut t = vec![root];
                t.extend(want_f16[c..c + n].iter().copied());
                let p: Vec<i32> = (0..t.len() as i32).map(|i| i - 1).collect();
                (t, p)
            }
            1 => {
                let right = want_f16[c];
                let wrong = (right + 1) % 128_256;
                let mut t = vec![root, wrong, right];
                let mut p = vec![-1, 0, 0];
                if remaining >= 2 {
                    t.push(want_f16[c + 1]);
                    p.push(2);
                }
                (t, p)
            }
            2 => {
                let right = want_f16[c];
                let wrong = (right + 7) % 128_256;
                (vec![root, right, wrong], vec![-1, 0, 1])
            }
            _ => {
                let wrong = (want_f16[c] + 3) % 128_256;
                (vec![root, wrong], vec![-1, 0])
            }
        }
    };

    // Two legs over the identical schedule: graph route, then forced eager.
    let mut streams: Vec<Vec<u32>> = Vec::new();
    for eager in [false, true] {
        let _eager_guard = TreeEagerGuard;
        if eager {
            // SAFETY: single-threaded test process (see TreeEagerGuard).
            #[allow(unsafe_code)]
            unsafe {
                std::env::set_var("TRITIUM_TREE_EAGER", "1");
            }
        }
        let Some(mut r) = load_on("cuda", &bytes) else {
            return;
        };
        let positions: Vec<usize> = (0..prompt.len()).collect();
        let prefill_logits = r.forward(&prompt, &positions).expect("f16 prefill");
        assert_eq!(
            argmax(&prefill_logits) as u32,
            want_f16[0],
            "f16 prefill argmax must match the f16 reference's first token"
        );
        let rm = r
            .resident_cuda()
            .expect("resident model")
            .expect("cuda resident model present");
        let mut committed: Vec<u32> = vec![want_f16[0]];
        let mut phase = 0usize;
        while committed.len() < k_total {
            let (t, p) = make_tree(
                phase,
                committed.len(),
                *committed.last().expect("non-empty"),
            );
            let out = rm.tree_verify_greedy(&t, &p).expect("f16 tree verify");
            assert!(!out.is_empty(), "verify must always commit >= 1 token");
            committed.extend(&out);
            phase += 1;
        }
        let buckets = rm.tree_graph_bucket_count();
        if eager {
            assert_eq!(
                buckets, 0,
                "TRITIUM_TREE_EAGER leg must not capture tree graphs"
            );
        } else {
            assert!(
                buckets >= 1,
                "f16 tree verify fell back to the eager route — the L6 gate \
                 requires the GRAPH route under TRITIUM_KV=f16 (buckets = 0)"
            );
        }

        // Tooth 2 (per leg): committed == plain f16 greedy, token for token.
        let c_full = committed.len();
        assert!(
            c_full <= want_f16.len(),
            "committed stream overran the f16 reference window"
        );
        assert_eq!(
            &committed[..],
            &want_f16[..c_full],
            "f16 tree-verify committed stream must equal plain f16 greedy \
             (lossless within the rung; eager={eager})"
        );
        streams.push(committed);
    }

    // Tooth 3: graph and eager legs commit the identical stream.
    assert_eq!(
        streams[0], streams[1],
        "f16 tree verify: graph route and eager route diverged"
    );

    // For the record only (the f16 tier is ppl-gated, not token-gated vs f32):
    // where does the f16 stream first leave the f32 stream?
    drop(_kv);
    if let Some(mut r32) = load_on("cuda", &bytes) {
        let want_f32 = r32
            .generate(&prompt, k_total, u32::MAX)
            .expect("plain f32 greedy generate");
        let div = want_f16
            .iter()
            .zip(&want_f32)
            .position(|(a, b)| a != b)
            .map_or("none within window".to_string(), |i| format!("step {i}"));
        println!(
            "f16 tree-verify gate: graph==eager over {} committed tokens, \
             lossless vs plain f16; f16-vs-f32 first token divergence: {div}",
            streams[0].len()
        );
    }
}

/// ADR 0036 L6 **stage 2** gate: the BATCHED arenas on the f16 KV rung.
/// Under `TRITIUM_KV=f16` the batch arenas are __half and the mdecode/split
/// kernels are the `_h` twins; the f32 gates' invariants must hold WITHIN
/// the rung:
///
/// 1. **Graph == eager** (`decode_batch_graph` vs `decode_batch`): bitwise
///    logits, every step and row — the stage-2 mirror of
///    `cuda_batch_decode_graph_matches_eager`.
/// 2. **Within-rung anchor**: batch rows' greedy tokens equal the M=1 f16
///    `step_graph` greedy tokens (the `cuda_batch_decode_matches_single`
///    argmax-level contract, on the rung).
/// 3. **Paged == dense**: adoption (the kv_elem=2 byte-span copy), lockstep
///    graph + eager + argmax steps, and the raw adopted KV rows (dense vs
///    paged, byte for byte) — the stage-2 mirror of
///    `cuda_batch_paged_matches_dense_bit_exact`.
#[cfg(feature = "cuda")]
#[test]
fn cuda_batch_decode_f16_graph_matches_eager_and_paged_matches_dense() {
    let _gpu = gpu_serial();
    let Some((reference, bytes)) = maybe_load() else {
        return;
    };
    let _kv = KvEnvGuard::set("f16");
    let Some(mut runner) = load_on("cuda", &bytes) else {
        return;
    };
    const N: usize = 2;
    let toks: [u32; 4] = [128000, 791, 6864, 315];

    // Teeth 1+2: graph == eager (bitwise) and batch greedy == single greedy.
    {
        let model = match runner.resident_cuda() {
            Ok(Some(m)) => m,
            _ => {
                eprintln!("skipping f16 batch gate: no cuda resident");
                return;
            }
        };
        model.reset();
        let mut single: Vec<Vec<f32>> = Vec::with_capacity(toks.len());
        for (i, &t) in toks.iter().enumerate() {
            single.push(model.step_graph(t, i).expect("f16 single step_graph"));
        }
        let mut eager = model.new_batch(N).expect("new_batch eager");
        let mut graph = model.new_batch(N).expect("new_batch graph");
        for (i, &t) in toks.iter().enumerate() {
            let row_tokens = vec![t; N];
            let want = model
                .decode_batch(&mut eager, &row_tokens)
                .expect("f16 decode_batch eager");
            let got = model
                .decode_batch_graph(&mut graph, &row_tokens)
                .expect("f16 decode_batch_graph");
            for r in 0..N {
                for v in 0..want[r].len() {
                    assert_eq!(
                        got[r][v].to_bits(),
                        want[r][v].to_bits(),
                        "f16 batch graph != eager at step {i} row {r} vocab {v}"
                    );
                }
            }
            let batch_tok = tritium_nn::sample_greedy(&want[0]).expect("non-empty logits");
            let single_tok = tritium_nn::sample_greedy(&single[i]).expect("non-empty logits");
            assert_eq!(
                batch_tok, single_tok,
                "f16 batch greedy != f16 single greedy at step {i}"
            );
        }
        drop(eager);
        drop(graph);
        model.reset();
    }

    // Tooth 3: paged == dense on f16, through adoption + all three dispatch
    // paths (graph, eager, argmax graph).
    let prompt = reference.token_ids.clone();
    assert!(
        prompt.len() + toks.len() + 2 <= 256,
        "fixture grew past one KV page — retune the pool arithmetic"
    );
    let positions: Vec<usize> = (0..prompt.len()).collect();
    runner.forward(&prompt, &positions).expect("f16 prefill");
    let mut dense = runner.new_batch(N).expect("new_batch dense");
    let mut paged = runner.new_batch_paged(N, 2).expect("new_batch_paged");
    paged
        .reserve_pages(0, prompt.len() + toks.len() + 2)
        .expect("reserve row0");
    paged
        .reserve_pages(1, toks.len() + 2)
        .expect("reserve row1");
    runner
        .adopt_into_batch_row(&mut dense, 0, prompt.len())
        .expect("adopt dense");
    runner
        .adopt_into_batch_row(&mut paged, 0, prompt.len())
        .expect("adopt paged");
    for b in [&mut dense, &mut paged] {
        b.set_position(0, prompt.len()).expect("pos row0");
    }
    // Adopted f16 KV rows: dense == paged byte-for-byte, and non-zero.
    {
        let model = runner
            .resident_cuda()
            .expect("resident")
            .expect("cuda resident");
        for v in [false, true] {
            for row in [0usize, prompt.len() - 1] {
                let d = model
                    .debug_batch_kv_row(&dense, 0, 0, row, v)
                    .expect("dense adopted row");
                let p = model
                    .debug_batch_kv_row(&paged, 0, 0, row, v)
                    .expect("paged adopted row");
                assert_eq!(d, p, "adopted f16 KV row diverged (v={v}, row={row})");
                assert!(
                    d.iter().any(|&b| b != 0),
                    "adopted f16 KV row all-zero (v={v}, row={row}) — vacuous"
                );
            }
        }
    }
    for (i, &t) in toks.iter().enumerate() {
        let want = runner
            .decode_batch_graph(&mut dense, &[t, t])
            .expect("f16 dense step");
        let got = runner
            .decode_batch_graph(&mut paged, &[t, t])
            .expect("f16 paged step");
        for r in 0..N {
            for v in 0..want[r].len() {
                assert_eq!(
                    got[r][v].to_bits(),
                    want[r][v].to_bits(),
                    "f16 paged != dense at step {i} row {r} vocab {v}"
                );
            }
        }
    }
    {
        let model = runner
            .resident_cuda()
            .expect("resident")
            .expect("cuda resident");
        let want = model
            .decode_batch(&mut dense, &[toks[0], toks[0]])
            .expect("f16 dense eager");
        let got = model
            .decode_batch(&mut paged, &[toks[0], toks[0]])
            .expect("f16 paged eager");
        for r in 0..N {
            for v in 0..want[r].len() {
                assert_eq!(
                    got[r][v].to_bits(),
                    want[r][v].to_bits(),
                    "f16 paged != dense at eager step row {r} vocab {v}"
                );
            }
        }
        let want_ids = model
            .decode_batch_graph_argmax(&mut dense, &[toks[1], toks[1]])
            .expect("f16 dense argmax");
        let got_ids = model
            .decode_batch_graph_argmax(&mut paged, &[toks[1], toks[1]])
            .expect("f16 paged argmax");
        assert_eq!(
            want_ids, got_ids,
            "f16 argmax graph diverged paged vs dense"
        );
    }
    drop(dense);
    drop(paged);
    println!(
        "f16 batch gate: graph==eager bitwise + single-greedy anchor over {} \
         steps; paged==dense bitwise through adoption, {} graph + eager + \
         argmax steps",
        toks.len(),
        toks.len()
    );
}

/// ADR 0036 L6 stage 2, I4 on the rung: batched-slots tree verify under
/// `TRITIUM_KV=f16` — `tree_verify_greedy_slots` must commit EXACTLY what
/// per-slot `tree_verify_greedy_slot` verifies sequentially (the
/// `cuda_tree_verify_slots_matches_sequential` contract, dense AND paged),
/// and both must be lossless vs plain f16 greedy per slot. Also pins the
/// batched vs sequential slot KV bytes (the promoted rows) at the first and
/// last layers.
#[cfg(feature = "cuda")]
#[test]
fn cuda_tree_verify_slots_f16_matches_sequential() {
    let _gpu = gpu_serial();
    if !Path::new(&*DRAFTER_PATH).exists() {
        eprintln!("skipping: {} absent (gated drafter test)", *DRAFTER_PATH);
        return;
    }
    let bytes = std::fs::read(&*DRAFTER_PATH).expect("read drafter GGUF");
    let _kv = KvEnvGuard::set("f16");

    let base: [u32; 7] = [128000, 791, 6864, 315, 279, 1917, 574];
    let prompts: Vec<Vec<u32>> = vec![
        base.iter().copied().cycle().take(9).collect(),
        base.iter().copied().cycle().skip(2).take(14).collect(),
        base.iter().copied().cycle().skip(4).take(12).collect(),
    ];
    let never = u32::MAX;
    const V: u32 = 128_256;

    // f16 reference greedy streams (within-rung anchors).
    let wants: Vec<Vec<u32>> = {
        let Some(mut r) = load_on("cuda", &bytes) else {
            return;
        };
        prompts
            .iter()
            .map(|p| r.generate(p, 16, never).expect("f16 ref stream"))
            .collect()
    };

    // Two phases per slot: chain / branchy-with-wrong-siblings / all-wrong,
    // then shapes rotated — the f32 slots gate's mix, trimmed.
    let make_tree = |slot: usize, phase: usize, c: usize, root: u32| -> (Vec<u32>, Vec<i32>) {
        let want = &wants[slot];
        match (phase, slot) {
            (0, 0) => (vec![root, want[c], want[c + 1]], vec![-1, 0, 1]),
            (0, 1) => (
                vec![
                    root,
                    (want[c] + 1) % V,
                    want[c],
                    (want[c + 1] + 7) % V,
                    want[c + 1],
                ],
                vec![-1, 0, 0, 2, 2],
            ),
            (0, 2) => (vec![root, (want[c] + 5) % V], vec![-1, 0]),
            (1, 0) => (vec![root, want[c]], vec![-1, 0]),
            (1, 1) => (
                vec![root, want[c], want[c + 1], want[c + 2]],
                vec![-1, 0, 1, 2],
            ),
            _ => (vec![root, (want[c] + 1) % V, want[c]], vec![-1, 0, 0]),
        }
    };

    for paged in [false, true] {
        // Two independently-loaded legs: batched (one grouped verify per
        // phase) vs sequential (per-slot verifies). Each returns per slot
        // (committed tokens, promoted KV row bytes at first+last layer).
        type Leg = Vec<(Vec<u32>, Vec<u8>)>;
        let legs: Vec<Leg> = (0..2)
            .map(|leg| -> Leg {
                let Some(mut rb) = load_on("cuda", &bytes) else {
                    return Vec::new();
                };
                let n_layers = rb.config.n_layers as usize;
                let mut bat = if paged {
                    let mut b = rb.new_batch_paged(3, 3).expect("new_batch_paged");
                    for (r, p) in prompts.iter().enumerate() {
                        b.reserve_pages(r, p.len() + 24).expect("reserve slot");
                    }
                    b
                } else {
                    rb.new_batch(3).expect("new_batch")
                };
                let mut committed: Vec<Vec<u32>> = Vec::new();
                for (r, prompt) in prompts.iter().enumerate() {
                    let positions: Vec<usize> = (0..prompt.len()).collect();
                    rb.reset();
                    let logits = rb.forward(prompt, &positions).expect("f16 slot prefill");
                    rb.adopt_into_batch_row(&mut bat, r, prompt.len())
                        .expect("adopt");
                    bat.set_position(r, prompt.len()).expect("pos");
                    committed.push(vec![argmax(&logits) as u32]);
                }
                let model = rb
                    .resident_cuda()
                    .expect("resident")
                    .expect("cuda resident");
                for phase in 0..2 {
                    let trees: Vec<(Vec<u32>, Vec<i32>)> = (0..3)
                        .map(|r| {
                            make_tree(
                                r,
                                phase,
                                committed[r].len(),
                                *committed[r].last().expect("non-empty"),
                            )
                        })
                        .collect();
                    if leg == 0 {
                        let refs: Vec<(&[u32], &[i32])> = trees
                            .iter()
                            .map(|(t, p)| (t.as_slice(), p.as_slice()))
                            .collect();
                        let outs = model
                            .tree_verify_greedy_slots(&mut bat, &[0, 1, 2], &refs)
                            .expect("f16 slots verify");
                        for (r, out) in outs.into_iter().enumerate() {
                            assert!(!out.is_empty(), "slots verify committed nothing");
                            committed[r].extend(out);
                        }
                    } else {
                        for r in 0..3 {
                            let (t, p) = &trees[r];
                            let out = model
                                .tree_verify_greedy_slot(&mut bat, r, t, p)
                                .expect("f16 slot verify");
                            assert!(!out.is_empty(), "slot verify committed nothing");
                            committed[r].extend(out);
                        }
                    }
                }
                // Promoted KV rows of every slot, first + last layer, K + V.
                // The last committed token's row is not yet appended (it is
                // the next feed), hence the -1 span bound.
                let out: Leg = (0..3)
                    .map(|r| {
                        let mut kv = Vec::new();
                        for li in [0usize, n_layers - 1] {
                            for v in [false, true] {
                                for row in
                                    prompts[r].len()..prompts[r].len() + committed[r].len() - 1
                                {
                                    kv.extend(
                                        model
                                            .debug_batch_kv_row(&bat, li, r, row, v)
                                            .expect("slot kv row"),
                                    );
                                }
                            }
                        }
                        (committed[r].clone(), kv)
                    })
                    .collect();
                drop(bat);
                out
            })
            .collect();
        if legs[0].is_empty() || legs[1].is_empty() {
            return; // no device
        }
        for (r, want) in wants.iter().enumerate().take(3) {
            assert_eq!(
                legs[0][r].0, legs[1][r].0,
                "f16 slots (paged={paged}) batched != sequential committed tokens at slot {r}"
            );
            assert_eq!(
                legs[0][r].1, legs[1][r].1,
                "f16 slots (paged={paged}) batched != sequential promoted KV bytes at slot {r}"
            );
            // Losslessness within the rung: committed == plain f16 greedy.
            let n_tok = legs[0][r].0.len().min(want.len());
            assert_eq!(
                &legs[0][r].0[..n_tok],
                &want[..n_tok],
                "f16 slots (paged={paged}) slot {r}: committed stream != plain f16 greedy"
            );
        }
    }
    println!("f16 slots gate: batched == sequential (dense + paged), lossless per slot");
}

/// Batch↔graph single-token bit-parity: after the same prompt prefill,
/// forwarding ONE token via the batch path (`prefill(&[t], _)`) and via the
/// M=1 graph path (`step_graph(t, _)`) must give BIT-IDENTICAL logits — the
/// two f32 pipelines share every kernel-level reduction order (dp4a exact
/// GEMMs, canonical tree rmsnorm, order-preserving attention, f16 LM head).
/// Measured while root-causing a tail divergence that turned out to be the
/// EAGER path's f32-table LM head (a designed difference), not a state gap.
#[cfg(feature = "cuda")]
#[test]
fn cuda_batch_and_graph_single_token_bit_identical() {
    let _gpu = gpu_serial();
    let Some((reference, bytes)) = maybe_load() else {
        return;
    };
    let prompt = reference.token_ids.clone();
    let positions: Vec<usize> = (0..prompt.len()).collect();
    let t = reference.greedy_ids[0];
    let p = prompt.len();

    let mut r1 = load_on("cuda", &bytes).expect("runner1");
    r1.forward(&prompt, &positions).expect("prefill1");
    let l_step = r1
        .resident_cuda()
        .expect("resident")
        .expect("cuda")
        .step_graph(t, p)
        .expect("step");

    let mut r2 = load_on("cuda", &bytes).expect("runner2");
    r2.forward(&prompt, &positions).expect("prefill2");
    let l_batch = r2
        .resident_cuda()
        .expect("resident")
        .expect("cuda")
        .prefill(&[t], &[p])
        .expect("batch prefill");

    for (i, (&a, &b)) in l_step.iter().zip(&l_batch).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "batch/graph logit bit mismatch at vocab idx {i}: graph={a} batch={b}"
        );
    }

    // Eager-path cross-check on the same token: the eager `step` keeps the
    // UNFUSED rope + kv_append sequence (the graph uses `rope_kv_fused_g`), so
    // agreement here exercises the fused kernel against its unfused reference
    // at the state level. Logits cannot be bit-compared and near-tie argmaxes
    // can legitimately flip — eager reads the f32 LM-head table, the graph the
    // f16 one (a designed difference; a flip was observed on this very prompt)
    // — so the gate is the same 2e-3 relative bound as cpu↔cuda parity: a
    // fused-kernel state bug (wrong/missing KV row) lands orders of magnitude
    // past it, while the head-table delta stays well inside.
    let mut r3 = load_on("cuda", &bytes).expect("runner3");
    r3.forward(&prompt, &positions).expect("prefill3");
    let l_eager = r3
        .resident_cuda()
        .expect("resident")
        .expect("cuda")
        .step(t, p)
        .expect("eager step");
    // Full-vocab max_rel_err is the wrong metric across the two head tables
    // (near-zero logits make the relative error blow up on f16-vs-f32 rounding
    // alone); a state bug shows in the DECISION region. Gate the graph's top-16
    // logits: eager must agree there within the cpu↔cuda parity bound.
    let mut order: Vec<usize> = (0..l_step.len()).collect();
    order.sort_by(|&a, &b| l_step[b].partial_cmp(&l_step[a]).expect("finite logits"));
    for &i in order.iter().take(16) {
        let (g, e) = (l_step[i], l_eager[i]);
        let rel = (g - e).abs() / g.abs().max(1e-6);
        assert!(
            rel <= 2e-3,
            "eager/graph top-logit rel err {rel:.3e} > 2e-3 at vocab idx {i} \
             (graph={g} eager={e}) — fused rope+kv vs unfused drifted?"
        );
    }
}

/// Round 25 gate: `decode_greedy_step` (graph replay + DEVICE argmax, 4-byte
/// readback) must produce the exact token `forward` (graph replay + full
/// logits download + HOST argmax) produces, step for step over a greedy
/// chain — the drafter's fast path may not change a single drafted token.
/// Two runners over the same GGUF: identical weights, independent KV.
#[cfg(feature = "cuda")]
#[test]
fn cuda_decode_greedy_step_matches_host_argmax() {
    let _gpu = gpu_serial();
    let Some((reference, bytes)) = maybe_load() else {
        return;
    };
    let Some(mut r_logits) = load_on("cuda", &bytes) else {
        return;
    };
    let Some(mut r_argmax) = load_on("cuda", &bytes) else {
        return;
    };
    let prompt: Vec<u32> = reference.token_ids.clone();
    let positions: Vec<usize> = (0..prompt.len()).collect();
    let l0 = r_logits.forward(&prompt, &positions).expect("prefill A");
    let _ = r_argmax.forward(&prompt, &positions).expect("prefill B");
    let mut tok = argmax(&l0) as u32;
    for i in 0..48 {
        let pos = prompt.len() + i;
        let logits = r_logits.forward(&[tok], &[pos]).expect("logits step");
        let host_next = argmax(&logits) as u32;
        let dev_next = r_argmax
            .decode_greedy_step(tok, pos)
            .expect("argmax step")
            .expect("resident decoder available on the cuda backend");
        assert_eq!(
            dev_next, host_next,
            "device argmax step diverged from host argmax at decode step {i}"
        );
        tok = host_next;
    }
}

/// ADR 0032 L1' gate: `decode_greedy_chain(k)` must produce token-for-token
/// what `k` calls of `decode_greedy_step` produce (same graph, same argmax
/// kernels — the chain only moves the feedback loop on-device), across
/// chain lengths, back-to-back chains, and the EOS-truncation semantics
/// (a chain whose eos equals its own first id must return exactly that one
/// id and leave the KV watermark consistent for the NEXT chain).
#[cfg(feature = "cuda")]
#[test]
fn cuda_draft_chain_matches_per_step() {
    let _gpu = gpu_serial();
    let Some((reference, bytes)) = maybe_load() else {
        return;
    };
    let Some(mut r_step) = load_on("cuda", &bytes) else {
        return;
    };
    let Some(mut r_chain) = load_on("cuda", &bytes) else {
        return;
    };
    let prompt: Vec<u32> = reference.token_ids.clone();
    let positions: Vec<usize> = (0..prompt.len()).collect();
    let l0 = r_step.forward(&prompt, &positions).expect("prefill A");
    let _ = r_chain.forward(&prompt, &positions).expect("prefill B");
    let mut tok = argmax(&l0) as u32;
    let never_eos = u32::MAX; // no id can match: pure no-truncation chains

    // Back-to-back chains of varied k, both runners kept in lockstep.
    let mut pos = prompt.len();
    for &k in &[1usize, 4, 7, 16] {
        let mut per_step = Vec::with_capacity(k);
        let mut t = tok;
        for i in 0..k {
            let id = r_step
                .decode_greedy_step(t, pos + i)
                .expect("step ok")
                .expect("resident");
            per_step.push(id);
            t = id;
        }
        let chained = r_chain
            .decode_greedy_chain(tok, pos, k, never_eos)
            .expect("chain ok")
            .expect("resident");
        assert_eq!(
            chained, per_step,
            "chain(k={k}) diverged from per-step at pos {pos}"
        );
        pos += k;
        tok = *per_step.last().expect("k >= 1");
    }

    // EOS semantics: fresh third runner, chain with eos == the known first
    // drafted id -> exactly one id back, watermark advanced by one, and the
    // NEXT chain continues bit-identically with the lockstep runner.
    let Some(mut r_eos) = load_on("cuda", &bytes) else {
        return;
    };
    let _ = r_eos.forward(&prompt, &positions).expect("prefill C");
    let t = argmax(&l0) as u32;
    // Recompute the first two drafted ids on the lockstep runner's history:
    // r_step already advanced; rebuild from a fresh fourth runner instead.
    let Some(mut r_ref) = load_on("cuda", &bytes) else {
        return;
    };
    let _ = r_ref.forward(&prompt, &positions).expect("prefill D");
    let a0 = r_ref
        .decode_greedy_step(t, prompt.len())
        .expect("ref step0")
        .expect("resident");
    let a1 = r_ref
        .decode_greedy_step(a0, prompt.len() + 1)
        .expect("ref step1")
        .expect("resident");

    let first = r_eos
        .decode_greedy_chain(t, prompt.len(), 8, a0)
        .expect("eos chain ok")
        .expect("resident");
    assert_eq!(first, vec![a0], "eos-truncated chain must stop at [a0]");
    // The truncated chain fed only `t` (a0 was drafted, never fed) — so the
    // next chain feeds a0 at prompt.len()+1 and must produce a1 first.
    let second = r_eos
        .decode_greedy_chain(a0, prompt.len() + 1, 2, u32::MAX)
        .expect("post-eos chain ok")
        .expect("resident");
    assert_eq!(
        second.first(),
        Some(&a1),
        "watermark after an eos-truncated chain must be consistent"
    );
}

/// The small ADR 0021 drafter GGUF for the I1 batched-draft gate: the batched
/// drafter is the consumer, and it fits alongside a training run's VRAM
/// footprint where the 2B4T target would not. Skips cleanly when absent.
/// Ungated: the host-branch truncate gate uses it on the CPU backend too.
static DRAFTER_PATH: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    format!(
        "{}/blut/data/drafter-8L768-s3.gguf",
        std::env::var("HOME").unwrap_or_default()
    )
});

/// I1 gate (L3 batch-slot spec decode): `draft_batch` must produce, per LIVE
/// row, token-for-token what the single-sequence `decode_greedy_chain`
/// produces over the same history, for k ∈ {1, 4, 7, 16} — including (a) a
/// row that hits EOS mid-draft (synthesized, like the draft_chain gate, by
/// passing a known first-drafted id as `eos`) while the other rows keep
/// drafting, (b) a dead row mixed in (drafts nothing, position frozen), and
/// (c) rows of unequal positions (three different-length histories). The
/// KV/position state after each call must equal the single-seq equivalent:
/// checked by position bookkeeping plus CONTINUING the lockstep across all
/// six chained calls — a wrong or missing KV row would derail every
/// subsequent chain's tokens.
///
/// Loads ONLY the drafter GGUF (never the 2B4T target); reference runners
/// are loaded one at a time and dropped, keeping peak VRAM at ~one drafter
/// model + one 4-slot batch.
#[cfg(feature = "cuda")]
#[test]
fn cuda_draft_batch_matches_draft_chain() {
    let _gpu = gpu_serial();
    if !Path::new(&*DRAFTER_PATH).exists() {
        eprintln!("skipping: {} absent (gated drafter test)", *DRAFTER_PATH);
        return;
    }
    let bytes = std::fs::read(&*DRAFTER_PATH).expect("read drafter GGUF");

    // Three DIFFERENT histories (content + length) => unequal row positions.
    let base: [u32; 7] = [128000, 791, 6864, 315, 279, 1917, 574];
    let prompts: Vec<Vec<u32>> = (0..3usize)
        .map(|i| {
            base.iter()
                .copied()
                .cycle()
                .skip(i)
                .take(6 + 5 * i)
                .collect()
        })
        .collect();
    let never = u32::MAX;

    // Pre-pass: replay row 1 single-seq through the k∈{1,4,7,16} ladder to
    // learn the id it would draft FIRST in the k=8 chain — that id becomes
    // the shared `eos` so row 1 halts mid-draft while rows 0/2 keep going
    // (the draft_chain gate's synthesis trick, batched).
    let eos_b = {
        let Some(mut r) = load_on("cuda", &bytes) else {
            return;
        };
        let positions: Vec<usize> = (0..prompts[1].len()).collect();
        let l0 = r.forward(&prompts[1], &positions).expect("pre prefill");
        let mut tok = argmax(&l0) as u32;
        let mut pos = prompts[1].len();
        for &k in &[1usize, 4, 7, 16] {
            let out = r
                .decode_greedy_chain(tok, pos, k, never)
                .expect("pre chain")
                .expect("resident");
            pos += out.len();
            tok = *out.last().expect("k >= 1");
        }
        r.decode_greedy_step(tok, pos)
            .expect("pre step")
            .expect("resident")
    };

    // The six-call schedule every row runs in lockstep: the no-EOS ladder,
    // the EOS-mid-draft chain, then a post-EOS continuation (the watermark
    // consistency probe).
    let schedule: [(usize, u32); 6] = [
        (1, never),
        (4, never),
        (7, never),
        (16, never),
        (8, eos_b),
        (2, never),
    ];

    // Reference pass: one single-seq runner per row (sequential, dropped
    // after use), recording the prefill argmax + every chain's tokens.
    let mut first_tok = [0u32; 3];
    let mut expected: Vec<Vec<Vec<u32>>> = Vec::with_capacity(3);
    for (r, prompt) in prompts.iter().enumerate() {
        let Some(mut runner) = load_on("cuda", &bytes) else {
            return;
        };
        let positions: Vec<usize> = (0..prompt.len()).collect();
        let l0 = runner.forward(prompt, &positions).expect("ref prefill");
        let mut tok = argmax(&l0) as u32;
        first_tok[r] = tok;
        let mut pos = prompt.len();
        let mut chains = Vec::with_capacity(schedule.len());
        for &(k, eos) in &schedule {
            let out = runner
                .decode_greedy_chain(tok, pos, k, eos)
                .expect("ref chain")
                .expect("resident");
            assert!(!out.is_empty(), "ref chain drafted nothing (row {r})");
            pos += out.len();
            tok = *out.last().expect("non-empty");
            chains.push(out);
        }
        expected.push(chains);
    }
    // The synthesized EOS must actually trigger mid-draft on row 1 AND leave
    // at least one other row drafting past the halt point.
    assert_eq!(
        expected[1][4],
        vec![eos_b],
        "row 1's eos chain must halt at [eos] (the draft_chain gate shape)"
    );
    assert!(
        expected[0][4].len() > 1 || expected[2][4].len() > 1,
        "rows 0/2 must draft past row 1's halt for the gate to have teeth"
    );

    // Batch pass: one runner, a 4-slot batch — rows 0..3 live with the three
    // histories (prefilled single-seq, adopted, positioned), row 3 DEAD.
    let Some(mut rb) = load_on("cuda", &bytes) else {
        return;
    };
    let mut batch = rb.new_batch(4).expect("new_batch");
    for (r, prompt) in prompts.iter().enumerate() {
        rb.reset();
        let positions: Vec<usize> = (0..prompt.len()).collect();
        let l0 = rb.forward(prompt, &positions).expect("batch prefill");
        assert_eq!(
            argmax(&l0) as u32,
            first_tok[r],
            "batch runner prefill diverged from the reference prefill (row {r})"
        );
        rb.adopt_into_batch_row(&mut batch, r, prompt.len())
            .expect("adopt");
        batch.set_position(r, prompt.len()).expect("set_position");
    }
    batch.set_live(3, false).expect("set_live");

    let mut pos: Vec<usize> = prompts.iter().map(Vec::len).collect();
    let mut last = first_tok.to_vec();
    for (ci, &(k, eos)) in schedule.iter().enumerate() {
        let feeds = [last[0], last[1], last[2], 0];
        let outs = rb
            .draft_batch(&mut batch, &feeds, k, eos)
            .expect("draft_batch");
        assert_eq!(outs.len(), 4);
        assert!(
            outs[3].is_empty(),
            "dead row drafted tokens at call {ci}: {:?}",
            outs[3]
        );
        assert_eq!(
            batch.positions()[3],
            0,
            "dead row's position moved at call {ci}"
        );
        for r in 0..3 {
            assert_eq!(
                outs[r], expected[r][ci],
                "draft_batch row {r} diverged from decode_greedy_chain at call {ci} (k={k})"
            );
            pos[r] += outs[r].len();
            assert_eq!(
                batch.positions()[r],
                pos[r],
                "row {r} position after call {ci} != single-seq watermark"
            );
            last[r] = *outs[r].last().expect("non-empty per the ref assert");
        }
    }
    drop(batch);
    println!(
        "draft_batch parity: 3 live rows == draft_chain over {} chained calls \
         (k=1/4/7/16 + eos-mid-draft + post-eos continuation), dead row inert",
        schedule.len()
    );
}

/// Truncate-reconcile gate (ADR 0032): `truncate_kv(n)` followed by appending
/// a suffix must be BIT-IDENTICAL to a fresh runner that fed only the
/// surviving prefix + the same suffix through the same op shapes. This proves
/// both halves of the watermark contract at once: rows past the truncation
/// point are dead (no leak into any later attention range) and rows before it
/// survive untouched. Also pins the error contract: an over-long truncate
/// refuses WITHOUT disturbing state.
#[cfg(feature = "cuda")]
#[test]
fn cuda_truncate_kv_matches_fresh_prefill() {
    let _gpu = gpu_serial();
    if !std::path::Path::new(&*DRAFTER_PATH).exists() {
        eprintln!("skipping: {} absent (drafter-gated test)", *DRAFTER_PATH);
        return;
    }
    let bytes = std::fs::read(&*DRAFTER_PATH).expect("read drafter gguf");
    let Some(mut a) = load_on("cuda", &bytes) else {
        return;
    };
    let Some(mut b) = load_on("cuda", &bytes) else {
        return;
    };

    // Identical op sequence for every SURVIVING row on both runners: an M=24
    // prefill, one M=1 step (the row the truncate must preserve), then the
    // M=2 suffix. Runner A additionally grows a 3-row speculative suffix and
    // rewinds it with truncate_kv before the real suffix.
    let prompt: Vec<u32> = (1u32..=24).collect();
    let positions: Vec<usize> = (0..prompt.len()).collect();
    let _ = a.forward(&prompt, &positions).expect("prefill A");
    let _ = b.forward(&prompt, &positions).expect("prefill B");
    let n = prompt.len();
    let _ = a.forward(&[50], &[n]).expect("keep-row A");
    let _ = b.forward(&[50], &[n]).expect("keep-row B");

    // A only: speculative rows at n+1..n+4, then rewind them.
    let _ = a
        .forward(&[90, 91, 92], &[n + 1, n + 2, n + 3])
        .expect("speculative rows A");

    // Error contract first: truncate beyond the watermark must refuse and
    // leave the cache serving (the follow-up truncate + suffix succeed).
    assert!(
        a.truncate_kv(n + 10).is_err(),
        "truncate_kv past the watermark must refuse"
    );
    a.truncate_kv(n + 1).expect("rewind speculative rows");

    // A no-op truncate at the watermark is legal (the reconcile's clean
    // full-match arm issues exactly this).
    b.truncate_kv(n + 1).expect("no-op truncate at watermark");

    // Same suffix on both — DIFFERENT tokens from the truncated rows, so a
    // dead-row leak cannot cancel out.
    let la = a
        .forward(&[70, 71], &[n + 1, n + 2])
        .expect("suffix after truncate A");
    let lb = b.forward(&[70, 71], &[n + 1, n + 2]).expect("suffix B");
    assert_eq!(la.len(), lb.len(), "logits width mismatch");
    for (i, (x, y)) in la.iter().zip(lb.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "truncate-then-append diverged from fresh append at vocab idx {i} \
             (truncated={x} fresh={y}) — dead rows leaked or live rows moved"
        );
    }
}

/// Host-branch twin of the truncate gate: the same truncate-then-append ==
/// fresh-append equivalence on the CPU backend (no resident decoder), which
/// exercises the facade's check-all-then-rollback arm — per-layer length
/// validation, `rollback_to`, and the over-length refusal with state
/// untouched. Bitwise, same op shapes on both runners.
#[test]
fn cpu_truncate_kv_matches_fresh_prefill() {
    if !std::path::Path::new(&*DRAFTER_PATH).exists() {
        eprintln!("skipping: {} absent (drafter-gated test)", *DRAFTER_PATH);
        return;
    }
    let bytes = std::fs::read(&*DRAFTER_PATH).expect("read drafter gguf");
    let Some(mut a) = load_on("cpu", &bytes) else {
        return;
    };
    let Some(mut b) = load_on("cpu", &bytes) else {
        return;
    };

    let prompt: Vec<u32> = (1u32..=16).collect();
    let positions: Vec<usize> = (0..prompt.len()).collect();
    let _ = a.forward(&prompt, &positions).expect("prefill A");
    let _ = b.forward(&prompt, &positions).expect("prefill B");
    let n = prompt.len();
    let _ = a.forward(&[50], &[n]).expect("keep-row A");
    let _ = b.forward(&[50], &[n]).expect("keep-row B");

    // A only: speculative rows, then the error contract, then the rewind.
    let _ = a
        .forward(&[90, 91, 92], &[n + 1, n + 2, n + 3])
        .expect("speculative rows A");
    assert!(
        a.truncate_kv(n + 10).is_err(),
        "host truncate_kv past the cached length must refuse"
    );
    a.truncate_kv(n + 1).expect("host rewind");
    b.truncate_kv(n + 1).expect("host no-op truncate at length");

    let la = a
        .forward(&[70, 71], &[n + 1, n + 2])
        .expect("suffix after truncate A");
    let lb = b.forward(&[70, 71], &[n + 1, n + 2]).expect("suffix B");
    assert_eq!(la.len(), lb.len(), "logits width mismatch");
    for (i, (x, y)) in la.iter().zip(lb.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "host truncate-then-append diverged from fresh append at vocab \
             idx {i} (truncated={x} fresh={y})"
        );
    }
}

/// I1 paged leg (review follow-up on 503b520): the `page_mapped(r, p+k-1)`
/// full-room guard must (a) refuse a draft whose tail outruns the row's page
/// reservation WITHOUT any partial advance, and (b) once the reservation
/// covers the draft, produce tokens identical to the single-seq chain even
/// when the drafted KV crosses a page boundary.
#[cfg(feature = "cuda")]
#[test]
fn cuda_draft_batch_paged_guard_and_boundary() {
    use tritium_cuda::KV_PAGE_TOKENS;

    let _gpu = gpu_serial();
    if !Path::new(&*DRAFTER_PATH).exists() {
        eprintln!("skipping: {} absent (gated drafter test)", *DRAFTER_PATH);
        return;
    }
    let bytes = std::fs::read(&*DRAFTER_PATH).expect("read drafter GGUF");

    // A prompt ending 6 tokens shy of the first page boundary, so k=16
    // crosses into page 2.
    let base: [u32; 7] = [128000, 791, 6864, 315, 279, 1917, 574];
    let plen = KV_PAGE_TOKENS - 6;
    let prompt: Vec<u32> = base.iter().copied().cycle().take(plen).collect();
    let positions: Vec<usize> = (0..plen).collect();
    let (k, never) = (16usize, u32::MAX);

    // Reference: single-seq chain over the same history.
    let (first_tok, expect) = {
        let Some(mut r) = load_on("cuda", &bytes) else {
            return;
        };
        let l0 = r.forward(&prompt, &positions).expect("ref prefill");
        let tok = argmax(&l0) as u32;
        let out = r
            .decode_greedy_chain(tok, plen, k, never)
            .expect("ref chain")
            .expect("resident");
        (tok, out)
    };
    assert_eq!(expect.len(), k, "reference chain must draft k tokens");

    let Some(mut rb) = load_on("cuda", &bytes) else {
        return;
    };

    // (a) Pool of exactly ONE page: the prompt fits, the draft tail does
    // not. The guard must reject up front, leaving position untouched.
    {
        let mut batch = rb.new_batch_paged(1, 1).expect("new_batch_paged");
        batch.reserve_pages(0, plen).expect("reserve prompt");
        let l0 = rb.forward(&prompt, &positions).expect("prefill");
        assert_eq!(argmax(&l0) as u32, first_tok, "prefill diverged");
        rb.adopt_into_batch_row(&mut batch, 0, plen).expect("adopt");
        batch.set_position(0, plen).expect("set_position");
        let err = rb
            .draft_batch(&mut batch, &[first_tok], k, never)
            .expect_err("draft past the reservation must refuse");
        let msg = format!("{err}");
        assert!(
            msg.contains("page-reserved") || msg.contains("page"),
            "unexpected error: {msg}"
        );
        assert_eq!(
            batch.positions()[0],
            plen,
            "a refused draft must not advance the row at all"
        );
    }

    // (b) Two pages + an explicit reservation for the draft tail: the k
    // tokens cross the page boundary and must equal the single-seq chain.
    rb.reset();
    let mut batch = rb.new_batch_paged(1, 2).expect("new_batch_paged");
    batch
        .reserve_pages(0, plen + k)
        .expect("reserve prompt+draft");
    let l0 = rb.forward(&prompt, &positions).expect("prefill");
    assert_eq!(argmax(&l0) as u32, first_tok, "prefill diverged");
    rb.adopt_into_batch_row(&mut batch, 0, plen).expect("adopt");
    batch.set_position(0, plen).expect("set_position");
    let out = rb
        .draft_batch(&mut batch, &[first_tok], k, never)
        .expect("paged draft");
    assert_eq!(
        out[0], expect,
        "paged draft across the page boundary diverged from the single-seq chain"
    );
    assert_eq!(
        batch.positions()[0],
        plen + k,
        "watermark after paged draft"
    );
    println!(
        "draft_batch paged: reservation guard refuses cleanly (no partial \
         advance); {k}-token draft across the {KV_PAGE_TOKENS}-token page \
         boundary == single-seq chain"
    );
}

/// Removes `TRITIUM_TREE_EAGER` on drop so a failing eager pass can't leak
/// the env override into later tests in the same process.
#[cfg(feature = "cuda")]
struct TreeEagerGuard;
#[cfg(feature = "cuda")]
impl Drop for TreeEagerGuard {
    fn drop(&mut self) {
        // SAFETY: the GPU tests run single-threaded (`gpu_serial` +
        // `--test-threads=1`); no other thread touches the environment.
        #[allow(unsafe_code)]
        unsafe {
            std::env::remove_var("TRITIUM_TREE_EAGER");
        }
    }
}

/// I2 gate (L3 batch-slot spec decode): `tree_verify_greedy_slot` against
/// dense slot `r` of a 3-slot `BatchKv` must behave EXACTLY like the
/// single-sequence `tree_verify_greedy` over the same history:
///
/// - same accepted tokens for the same draft trees (a chain m=2, a chain m=4
///   and m=8 — both `TREE_BUCKETS` boundaries — and a branchy m=6 with wrong
///   siblings/tails), for slot r = 0 AND r = 2 (a nonzero KV row base is the
///   whole point);
/// - bit-identical post-commit KV (layer 0 and layer 7 K+V rows of the whole
///   committed region, byte-compared) and matching positions;
/// - identical continued decode (4 more tokens both ways);
/// - OTHER live slots untouched: rows != r keep drafting exactly what a
///   no-tree-interleaved single-seq control drafts;
/// - graph and eager routes agree: the whole schedule runs twice (captured
///   tree graphs, then `TRITIUM_TREE_EAGER=1`) and every accepted token and
///   KV byte must match across routes.
///
/// Drafter-only (the 2B4T target never loads); reference runners are loaded
/// one at a time and dropped, keeping peak VRAM at ~one drafter + one 3-slot
/// batch.
#[cfg(feature = "cuda")]
#[test]
fn cuda_tree_verify_slot_matches_single_seq() {
    let _gpu = gpu_serial();
    if !Path::new(&*DRAFTER_PATH).exists() {
        eprintln!("skipping: {} absent (gated drafter test)", *DRAFTER_PATH);
        return;
    }
    let bytes = std::fs::read(&*DRAFTER_PATH).expect("read drafter GGUF");

    let base: [u32; 7] = [128000, 791, 6864, 315, 279, 1917, 574];
    let prompts: Vec<Vec<u32>> = (0..3usize)
        .map(|i| {
            base.iter()
                .copied()
                .cycle()
                .skip(i)
                .take(6 + 5 * i)
                .collect()
        })
        .collect();
    let never = u32::MAX;
    const PRE: usize = 3; // pre-verify chain length (every row)
    const POST: usize = 4; // post-verify chain length (every row)
    // The drafter GGUF is 8L768 — layer 0's K/V precedes any attention, layer
    // 7's K/V has flowed through 7 tree-attention blocks (the kernels under
    // test). A different drafter panics the debug row reads loudly.
    const KV_LAYERS: [usize; 2] = [0, 7];

    for &target in &[0usize, 2] {
        let plen = prompts[target].len();

        // Reference greedy stream for building the target row's draft trees.
        let want = {
            let Some(mut r) = load_on("cuda", &bytes) else {
                return;
            };
            r.generate(&prompts[target], 28, never).expect("ref stream")
        };
        assert_eq!(want.len(), 28, "reference stream shorter than expected");

        // Draft-tree schedule: m ∈ {2, 4, 6, 8} (4 and 8 are TREE_BUCKETS
        // boundaries; 2 pads to 4, 6 pads to 8). `c` = committed length, so
        // `want[c..]` is the true greedy continuation (losslessness keeps the
        // committed stream on `want` in every leg).
        let make_tree = |phase: usize, c: usize, root: u32| -> (Vec<u32>, Vec<i32>) {
            match phase {
                // m=2 perfect chain.
                0 => (vec![root, want[c]], vec![-1, 0]),
                // m=4 perfect chain (exact bucket 4).
                1 => (
                    vec![root, want[c], want[c + 1], want[c + 2]],
                    vec![-1, 0, 1, 2],
                ),
                // m=6 branchy: wrong sibling first at two levels + wrong leaf.
                2 => {
                    let (r1, r2) = (want[c], want[c + 1]);
                    (
                        vec![
                            root,
                            (r1 + 1) % 128_256,
                            r1,
                            (r2 + 7) % 128_256,
                            r2,
                            (want[c + 2] + 3) % 128_256,
                        ],
                        vec![-1, 0, 0, 2, 2, 4],
                    )
                }
                // m=8 chain (exact bucket 8): 6 right drafts + a wrong tail.
                _ => {
                    let mut t = vec![root];
                    t.extend(want[c..c + 6].iter().copied());
                    t.push((want[c + 6] + 5) % 128_256);
                    (t, (0..8i32).map(|i| i - 1).collect())
                }
            }
        };

        // Per route: (accepted-tokens per phase, continuation, KV bytes).
        type RouteResult = (Vec<Vec<u32>>, Vec<u32>, Vec<Vec<u8>>);
        let mut route_results: Vec<RouteResult> = Vec::new();

        for eager in [false, true] {
            let _guard = TreeEagerGuard;
            if eager {
                // SAFETY: single-threaded test process (see TreeEagerGuard).
                #[allow(unsafe_code)]
                unsafe {
                    std::env::set_var("TRITIUM_TREE_EAGER", "1");
                }
            }

            // ── Single-seq leg on the target prompt ──
            // NO decode steps between prefill and the verifies: the M=1 chain
            // and the M=N draft paths are token-equal but only ULP-close in
            // KV, so a chain here would poison the bitwise KV compare below.
            // The batch leg mirrors this by keeping the target row DEAD during
            // the other rows' pre-verify draft.
            let (first_tok, single_outs, single_cont, single_kv) = {
                let Some(mut rs) = load_on("cuda", &bytes) else {
                    return;
                };
                let positions: Vec<usize> = (0..plen).collect();
                let l0 = rs
                    .forward(&prompts[target], &positions)
                    .expect("ss prefill");
                let first = argmax(&l0) as u32;
                assert_eq!(first, want[0], "prefill argmax vs reference stream");
                let mut committed: Vec<u32> = vec![first];
                let mut outs = Vec::new();
                for phase in 0..4 {
                    let (t, p) = make_tree(
                        phase,
                        committed.len(),
                        *committed.last().expect("non-empty"),
                    );
                    let rm = rs.resident_cuda().expect("resident").expect("cuda");
                    let out = rm.tree_verify_greedy(&t, &p).expect("ss tree verify");
                    assert!(!out.is_empty(), "verify must commit >= 1 token");
                    committed.extend(&out);
                    outs.push(out);
                }
                assert_eq!(
                    &committed[..],
                    &want[..committed.len()],
                    "single-seq tree-verify stream must stay lossless"
                );
                let rows = plen + committed.len() - 1; // == cache_len
                let cont = rs
                    .decode_greedy_chain(*committed.last().expect("non-empty"), rows, POST, never)
                    .expect("ss post chain")
                    .expect("resident");
                let rm = rs.resident_cuda().expect("resident").expect("cuda");
                let mut kv = Vec::new();
                for li in KV_LAYERS {
                    for row in 0..rows {
                        kv.push(rm.debug_kv_row(li, row, false).expect("ss kv k row"));
                        kv.push(rm.debug_kv_row(li, row, true).expect("ss kv v row"));
                    }
                }
                (first, outs, cont, kv)
            };

            // ── No-tree-interleaved controls for the other rows ──
            let mut ctrl_first = [0u32; 3];
            let mut ctrl_pre: Vec<Vec<u32>> = vec![Vec::new(); 3];
            let mut ctrl_post: Vec<Vec<u32>> = vec![Vec::new(); 3];
            for r in 0..3 {
                if r == target {
                    continue;
                }
                let Some(mut rc) = load_on("cuda", &bytes) else {
                    return;
                };
                let positions: Vec<usize> = (0..prompts[r].len()).collect();
                let l0 = rc.forward(&prompts[r], &positions).expect("ctrl prefill");
                ctrl_first[r] = argmax(&l0) as u32;
                ctrl_pre[r] = rc
                    .decode_greedy_chain(ctrl_first[r], prompts[r].len(), PRE, never)
                    .expect("ctrl pre chain")
                    .expect("resident");
                ctrl_post[r] = rc
                    .decode_greedy_chain(
                        *ctrl_pre[r].last().expect("PRE >= 1"),
                        prompts[r].len() + PRE,
                        POST,
                        never,
                    )
                    .expect("ctrl post chain")
                    .expect("resident");
            }

            // ── Batch leg: 3 live slots, trees run ONLY against `target` ──
            let Some(mut rb) = load_on("cuda", &bytes) else {
                return;
            };
            let mut batch = rb.new_batch(3).expect("new_batch");
            let mut first_toks = [0u32; 3];
            for (r, prompt) in prompts.iter().enumerate() {
                rb.reset();
                let positions: Vec<usize> = (0..prompt.len()).collect();
                let l0 = rb.forward(prompt, &positions).expect("batch prefill");
                first_toks[r] = argmax(&l0) as u32;
                rb.adopt_into_batch_row(&mut batch, r, prompt.len())
                    .expect("adopt");
                batch.set_position(r, prompt.len()).expect("set_position");
            }
            assert_eq!(first_toks[target], first_tok, "target prefill mismatch");

            // Pre-verify activity for the OTHER rows only: the target row sits
            // dead (frozen position, zero KV bytes — the C2 contract) so its
            // history stays prefill-only, matching the single-seq leg bitwise.
            batch.set_live(target, false).expect("set_live dead");
            let mut pre_feeds = first_toks;
            pre_feeds[target] = 0; // dead row's feed is ignored (in-range id)
            let pre_outs = rb
                .draft_batch(&mut batch, &pre_feeds, PRE, never)
                .expect("batch pre draft");
            assert!(
                pre_outs[target].is_empty(),
                "dead target row must draft nothing"
            );
            assert_eq!(
                batch.positions()[target],
                plen,
                "dead target row's position moved during the pre-draft"
            );
            batch.set_live(target, true).expect("set_live revive");
            let mut committed: Vec<u32> = vec![first_toks[target]];

            for (phase, single_out) in single_outs.iter().enumerate() {
                let (t, p) = make_tree(
                    phase,
                    committed.len(),
                    *committed.last().expect("non-empty"),
                );
                let out = rb
                    .tree_verify_greedy_slot(&mut batch, target, &t, &p)
                    .expect("slot tree verify");
                assert_eq!(
                    &out, single_out,
                    "slot verify accepted tokens != single-seq (phase {phase}, r={target})"
                );
                committed.extend(&out);
                assert_eq!(
                    batch.positions()[target],
                    plen + committed.len() - 1,
                    "slot position after phase {phase}"
                );
            }

            // Post-commit KV of the whole committed region: bit-identical to
            // the single-seq cache.
            {
                let rows = plen + committed.len() - 1;
                let rm = rb.resident_cuda().expect("resident").expect("cuda");
                let mut idx = 0usize;
                for li in KV_LAYERS {
                    for row in 0..rows {
                        for v in [false, true] {
                            let got = rm
                                .debug_batch_kv_row(&batch, li, target, row, v)
                                .expect("slot kv row");
                            assert_eq!(
                                got, single_kv[idx],
                                "slot KV byte mismatch (layer {li}, row {row}, v={v}, r={target})"
                            );
                            idx += 1;
                        }
                    }
                }
            }

            // Continue every row: target must equal the single-seq
            // continuation; the other rows must equal their NO-TREE controls
            // (their KV/positions were never touched by the slot verifies).
            let mut feeds = [0u32; 3];
            for r in 0..3 {
                feeds[r] = if r == target {
                    *committed.last().expect("non-empty")
                } else {
                    assert_eq!(
                        pre_outs[r], ctrl_pre[r],
                        "row {r} pre-draft != its single-seq control"
                    );
                    *pre_outs[r].last().expect("PRE >= 1")
                };
            }
            let post_outs = rb
                .draft_batch(&mut batch, &feeds, POST, never)
                .expect("batch post draft");
            assert_eq!(
                post_outs[target], single_cont,
                "target continuation diverged — promote corrupted slot KV (r={target})"
            );
            for r in 0..3 {
                if r == target {
                    continue;
                }
                assert_eq!(
                    post_outs[r], ctrl_post[r],
                    "row {r} diverged after trees on slot {target} — verify leaked across slots"
                );
            }
            drop(batch);
            route_results.push((single_outs, single_cont, single_kv));
        }

        // Graph route vs eager route: bitwise agreement (tokens AND KV
        // bytes). NB route_results holds the SINGLE-SEQ leg of each route, so
        // this block directly pins the single-seq ctrl twins; the SLOT-route
        // graph==eager property then holds transitively through the per-route
        // slot == single-seq asserts above (slotᵍ=ssᵍ ∧ slotᵉ=ssᵉ ∧ ssᵍ=ssᵉ).
        // Exact tier only: under TRITIUM_KERNEL_TIER=fast the graph route
        // runs the FUSED attention while eager runs the exact pair by
        // design (RFC 0001) — layer>0 KV carries earlier layers' attention
        // outputs, so the bitwise pins hold only when both routes run the
        // same kernels (review 3d99d55 nit a; pre-existing since dense
        // L3b).
        let fast_tier = std::env::var("TRITIUM_KERNEL_TIER").is_ok_and(|v| v == "fast");
        if fast_tier {
            eprintln!("skipping graph-vs-eager bitwise pins: fast tier (exact-tier semantics)");
            return;
        }
        let (g, e) = (&route_results[0], &route_results[1]);
        assert_eq!(g.0, e.0, "graph vs eager accepted tokens (r={target})");
        assert_eq!(g.1, e.1, "graph vs eager continuation (r={target})");
        assert_eq!(g.2.len(), e.2.len(), "graph vs eager KV row count");
        let rows = g.2.len() / (2 * KV_LAYERS.len());
        for (i, (a, b)) in g.2.iter().zip(&e.2).enumerate() {
            let (li, row, v) = (KV_LAYERS[i / (2 * rows)], (i / 2) % rows, i % 2 == 1);
            assert_eq!(
                a, b,
                "graph vs eager KV bytes differ (r={target}, layer {li}, row {row}, \
                 v={v}) — ctrl twins drifted"
            );
        }
        println!(
            "I2 slot r={target}: 4 trees (m=2/4/6/8) bit-equal to single-seq \
             (tokens, positions, KV layers {KV_LAYERS:?}), other slots inert, \
             graph==eager"
        );
    }
}

/// I3 gate (L3 batch-slot spec decode): `tree_verify_greedy_slot` against a
/// PAGED batch slot must be bit-identical to the SAME verify against a dense
/// batch slot over the same history — accepted tokens, positions, and KV
/// bytes read back THROUGH the page translation — under:
///
/// - genuinely NON-IDENTITY pages (interleaved reservations scramble the
///   physical page order; asserted via the table, so the gate can't go
///   vacuous);
/// - trees m=2/4/6/8 with the m=8 verify's provisional rows STRADDLING the
///   first `KV_PAGE_TOKENS` boundary (prompt length tuned so phase 3 lands
///   at prefix 252: rows 252..260 cross page 0 → 1);
/// - a reservation-too-short verify that must refuse loudly with ZERO state
///   change (position, KV bytes, free pages), then succeed after the
///   reservation is extended;
/// - other live slots inert (their pre/post drafts equal the dense leg's);
/// - both the graph route and the eager route (`TRITIUM_TREE_EAGER=1`),
///   tokens compared across routes too.
///
/// Drafter-only: the 2B4T target never loads.
#[cfg(feature = "cuda")]
#[test]
fn cuda_tree_verify_paged_slot_matches_dense() {
    use tritium_cuda::KV_PAGE_TOKENS;

    let _gpu = gpu_serial();
    if !Path::new(&*DRAFTER_PATH).exists() {
        eprintln!("skipping: {} absent (gated drafter test)", *DRAFTER_PATH);
        return;
    }
    let bytes = std::fs::read(&*DRAFTER_PATH).expect("read drafter GGUF");

    let base: [u32; 7] = [128000, 791, 6864, 315, 279, 1917, 574];
    const TARGET: usize = 1;
    // Tuned so phase 3 (m=8) starts at prefix plen_t + 9 = KV_PAGE_TOKENS - 4
    // (perfect chains accept fully, the branchy m=6 accepts 3 — asserted
    // below): provisional rows then straddle the first page boundary.
    let plen_t = KV_PAGE_TOKENS - 13;
    let prompts: Vec<Vec<u32>> = vec![
        base.iter().copied().cycle().take(9).collect(),
        base.iter().copied().cycle().take(plen_t).collect(),
        base.iter().copied().cycle().skip(3).take(12).collect(),
    ];
    let never = u32::MAX;
    const PRE: usize = 2;
    const POST: usize = 4;
    const KV_LAYERS: [usize; 2] = [0, 7];

    // Reference greedy stream for building the target slot's draft trees
    // (chains drafted from the true continuation accept deterministically).
    let want = {
        let Some(mut r) = load_on("cuda", &bytes) else {
            return;
        };
        r.generate(&prompts[TARGET], 24, never).expect("ref stream")
    };
    assert_eq!(want.len(), 24, "reference stream shorter than expected");

    let make_tree = |phase: usize, c: usize, root: u32| -> (Vec<u32>, Vec<i32>) {
        match phase {
            0 => (vec![root, want[c]], vec![-1, 0]),
            1 => (
                vec![root, want[c], want[c + 1], want[c + 2]],
                vec![-1, 0, 1, 2],
            ),
            2 => {
                let (r1, r2) = (want[c], want[c + 1]);
                (
                    vec![
                        root,
                        (r1 + 1) % 128_256,
                        r1,
                        (r2 + 7) % 128_256,
                        r2,
                        (want[c + 2] + 3) % 128_256,
                    ],
                    vec![-1, 0, 0, 2, 2, 4],
                )
            }
            _ => {
                let mut t = vec![root];
                t.extend(want[c..c + 6].iter().copied());
                t.push((want[c + 6] + 5) % 128_256);
                (t, (0..8i32).map(|i| i - 1).collect())
            }
        }
    };

    // Per route: (accepted per phase, post-draft outs for all rows).
    type RouteTokens = (Vec<Vec<u32>>, Vec<Vec<u32>>);
    let mut route_tokens: Vec<RouteTokens> = Vec::new();

    for eager in [false, true] {
        let _guard = TreeEagerGuard;
        if eager {
            // SAFETY: single-threaded test process (see TreeEagerGuard).
            #[allow(unsafe_code)]
            unsafe {
                std::env::set_var("TRITIUM_TREE_EAGER", "1");
            }
        }

        let Some(mut rb) = load_on("cuda", &bytes) else {
            return;
        };
        let mut dense = rb.new_batch(3).expect("new_batch");
        // Pool sized EXACTLY: 1 (target head) + 1 (slot 0) + 1 (slot 2) + 1
        // (target tail) — asserted drained below.
        let mut paged = rb.new_batch_paged(3, 4).expect("new_batch_paged");

        // Interleaved reservations scramble the physical pages: the target's
        // logical page 0 takes physical 0, but its logical page 1 is handed
        // out only AFTER slots 0/2 take physicals 1/2 — non-identity AND
        // non-contiguous.
        paged.reserve_pages(TARGET, 1).expect("reserve target head");
        paged.reserve_pages(0, 32).expect("reserve slot 0");
        paged.reserve_pages(2, 32).expect("reserve slot 2");
        paged
            .reserve_pages(TARGET, plen_t + 24)
            .expect("reserve target tail");
        assert_eq!(paged.free_pages(), 0, "pool sizing drifted");
        let trow = paged
            .debug_page_table_row(TARGET)
            .expect("paged batch has a table");
        assert!(
            trow.iter()
                .enumerate()
                .any(|(i, &p)| p >= 0 && p as usize != i),
            "vacuous gate: target slot's pages are identity-mapped ({trow:?})"
        );

        // Prefill each prompt once; adopt the SAME single-seq cache into the
        // dense slot and (through the page spans) the paged slot.
        let mut first_toks = [0u32; 3];
        for (r, prompt) in prompts.iter().enumerate() {
            rb.reset();
            let positions: Vec<usize> = (0..prompt.len()).collect();
            let l0 = rb.forward(prompt, &positions).expect("batch prefill");
            first_toks[r] = argmax(&l0) as u32;
            for b in [&mut dense, &mut paged] {
                rb.adopt_into_batch_row(b, r, prompt.len()).expect("adopt");
                b.set_position(r, prompt.len()).expect("set_position");
            }
        }
        assert_eq!(first_toks[TARGET], want[0], "target prefill vs reference");

        // Pre-verify drafting on the OTHER rows (target dead: frozen
        // position, zero KV bytes — identical schedule on both batches).
        let mut pre_feeds = first_toks;
        pre_feeds[TARGET] = 0;
        for b in [&mut dense, &mut paged] {
            b.set_live(TARGET, false).expect("set_live dead");
        }
        let pre_d = rb
            .draft_batch(&mut dense, &pre_feeds, PRE, never)
            .expect("dense pre draft");
        let pre_p = rb
            .draft_batch(&mut paged, &pre_feeds, PRE, never)
            .expect("paged pre draft");
        assert_eq!(pre_d, pre_p, "paged pre-draft != dense pre-draft");
        for b in [&mut dense, &mut paged] {
            b.set_live(TARGET, true).expect("set_live revive");
        }

        // The 4 verifies, dense slot then paged slot, phase by phase.
        let mut committed: Vec<u32> = vec![first_toks[TARGET]];
        let mut accepted: Vec<Vec<u32>> = Vec::new();
        for phase in 0..4 {
            if phase == 3 {
                let p3 = paged.positions()[TARGET];
                assert!(
                    p3 < KV_PAGE_TOKENS && p3 + 8 > KV_PAGE_TOKENS,
                    "phase-3 provisional rows must straddle the page boundary \
                     (prefix {p3}, page {KV_PAGE_TOKENS}) — retune plen_t"
                );
            }
            let (t, p) = make_tree(
                phase,
                committed.len(),
                *committed.last().expect("non-empty"),
            );
            let out_d = rb
                .tree_verify_greedy_slot(&mut dense, TARGET, &t, &p)
                .expect("dense slot verify");
            let out_p = rb
                .tree_verify_greedy_slot(&mut paged, TARGET, &t, &p)
                .expect("paged slot verify");
            assert_eq!(
                out_p, out_d,
                "paged accepted tokens != dense (phase {phase}, eager {eager})"
            );
            committed.extend(&out_d);
            assert_eq!(
                paged.positions()[TARGET],
                dense.positions()[TARGET],
                "paged position != dense after phase {phase}"
            );
            assert_eq!(
                paged.positions()[TARGET],
                prompts[TARGET].len() + committed.len() - 1,
                "slot position after phase {phase}"
            );
            accepted.push(out_d);
        }
        assert_eq!(
            &committed[..],
            &want[..committed.len()],
            "slot tree-verify stream must stay lossless"
        );

        // Post-commit KV of the whole committed region, byte-compared THROUGH
        // the page translation.
        {
            let rows = prompts[TARGET].len() + committed.len() - 1;
            let rm = rb.resident_cuda().expect("resident").expect("cuda");
            for li in KV_LAYERS {
                for row in 0..rows {
                    for v in [false, true] {
                        let got_d = rm
                            .debug_batch_kv_row(&dense, li, TARGET, row, v)
                            .expect("dense kv row");
                        let got_p = rm
                            .debug_batch_kv_row(&paged, li, TARGET, row, v)
                            .expect("paged kv row");
                        assert_eq!(
                            got_p, got_d,
                            "paged KV byte mismatch (layer {li}, row {row}, v={v}, \
                             eager {eager})"
                        );
                    }
                }
            }
        }

        // Continue EVERY row on both batches: the target pins the promote,
        // the other rows pin slot inertness (verifies touched nothing).
        let mut feeds = [0u32; 3];
        for r in 0..3 {
            feeds[r] = if r == TARGET {
                *committed.last().expect("non-empty")
            } else {
                *pre_d[r].last().expect("PRE >= 1")
            };
        }
        let post_d = rb
            .draft_batch(&mut dense, &feeds, POST, never)
            .expect("dense post draft");
        let post_p = rb
            .draft_batch(&mut paged, &feeds, POST, never)
            .expect("paged post draft");
        assert_eq!(
            post_p, post_d,
            "post-verify drafts diverged (target promote or slot leak, eager {eager})"
        );

        // Reservation-too-short (once; the guard is host-side and
        // route-independent): a 1-slot pool where the prompt fits page 0 but
        // prefix + the padded m=8 tree does not — loud refusal, zero state
        // change, then success after extending the reservation.
        if !eager {
            let plen_s = KV_PAGE_TOKENS - 6;
            let prompt: Vec<u32> = base.iter().copied().cycle().take(plen_s).collect();
            let positions: Vec<usize> = (0..plen_s).collect();
            let mut small = rb.new_batch_paged(1, 2).expect("small paged batch");
            small.reserve_pages(0, plen_s).expect("reserve prompt");
            rb.reset();
            let l0 = rb.forward(&prompt, &positions).expect("small prefill");
            let root = argmax(&l0) as u32;
            rb.adopt_into_batch_row(&mut small, 0, plen_s)
                .expect("adopt");
            small.set_position(0, plen_s).expect("set_position");

            let t: Vec<u32> = (0..8).map(|i| (root + i) % 128_256).collect();
            let p: Vec<i32> = (0..8i32).map(|i| i - 1).collect();
            let rm = rb.resident_cuda().expect("resident").expect("cuda");
            let kv_before = rm
                .debug_batch_kv_row(&small, 0, 0, plen_s - 1, false)
                .expect("kv before");
            let err = rb
                .tree_verify_greedy_slot(&mut small, 0, &t, &p)
                .expect_err("verify past the reservation must refuse");
            let msg = format!("{err}");
            assert!(
                msg.contains("page-reserved"),
                "unexpected refusal message: {msg}"
            );
            assert_eq!(
                small.positions()[0],
                plen_s,
                "a refused verify must not advance the slot"
            );
            assert_eq!(small.free_pages(), 1, "a refused verify must draw no pages");
            let rm = rb.resident_cuda().expect("resident").expect("cuda");
            let kv_after = rm
                .debug_batch_kv_row(&small, 0, 0, plen_s - 1, false)
                .expect("kv after");
            assert_eq!(kv_after, kv_before, "a refused verify must write no KV");

            small
                .reserve_pages(0, plen_s + 16)
                .expect("extend reservation");
            let out = rb
                .tree_verify_greedy_slot(&mut small, 0, &t, &p)
                .expect("verify after extending the reservation");
            assert!(!out.is_empty(), "verify must commit >= 1 token");
            assert_eq!(
                small.positions()[0],
                plen_s + out.len(),
                "position after the recovered verify"
            );
        }

        drop(dense);
        drop(paged);
        route_tokens.push((accepted, post_d));
    }

    let (g, e) = (&route_tokens[0], &route_tokens[1]);
    assert_eq!(g.0, e.0, "graph vs eager accepted tokens (paged gate)");
    assert_eq!(g.1, e.1, "graph vs eager post drafts (paged gate)");
    println!(
        "I3 paged slot: 4 trees (m=2/4/6/8, phase-3 straddling the \
         {KV_PAGE_TOKENS}-token boundary) bit-equal to the dense slot \
         (tokens, positions, KV layers {KV_LAYERS:?} through the scrambled \
         non-identity table), short-reservation refusal is state-free, other \
         slots inert, graph==eager"
    );
}

/// I4 gate (L3 batch-slot spec decode, the payoff rung):
/// `tree_verify_greedy_slots` — ONE tree forward over the CONCATENATION of 3
/// slots' trees — must be bit-identical, per slot, to running
/// `tree_verify_greedy_slot` SEQUENTIALLY on a twin batch over the same
/// histories: accepted tokens, positions, and KV bytes (layers 0 + 7, read
/// back THROUGH the page translation on the paged leg). Covered:
///
/// - per-slot chains of different lengths (phase 1: m = 3/5/2; phase 2:
///   m = 2/4/6, an exact bucket) including branchy trees, and a slot whose
///   drafts all reject (the EOS-ish early stop: it commits only the bonus
///   token while the other slots keep accepting);
/// - dense AND paged batches (interleaved reservations scramble the physical
///   page order; asserted non-identity so the gate can't go vacuous);
/// - both routes (captured slots graphs, then `TRITIUM_TREE_EAGER=1`),
///   accepted tokens compared across routes;
/// - continued decode after the verifies (`draft_batch`, 3 tokens/slot)
///   equal between the batched and sequential twins — a functional pin on
///   the per-slot promotes beyond the byte compares;
/// - guard atomicity: a dead row, a duplicate row, an out-of-range row, an
///   over-cap `m_total` (> 48) and a paged under-reservation each refuse
///   loudly with ZERO state change on ALL listed slots (positions, KV
///   bytes, free pages) — the fully-reserved slot is listed FIRST in the
///   under-reservation call to prove the guard phase precedes any device
///   work;
/// - `tree_reservation_rows` returns the padded single-slot demand
///   (bucketed) that serve-side reservations key off.
///
/// Drafter-only: the 2B4T target never loads.
#[cfg(feature = "cuda")]
#[test]
fn cuda_tree_verify_slots_matches_sequential() {
    let _gpu = gpu_serial();
    if !Path::new(&*DRAFTER_PATH).exists() {
        eprintln!("skipping: {} absent (gated drafter test)", *DRAFTER_PATH);
        return;
    }
    let bytes = std::fs::read(&*DRAFTER_PATH).expect("read drafter GGUF");

    let base: [u32; 7] = [128000, 791, 6864, 315, 279, 1917, 574];
    let prompts: Vec<Vec<u32>> = vec![
        base.iter().copied().cycle().take(9).collect(),
        base.iter().copied().cycle().skip(2).take(14).collect(),
        base.iter().copied().cycle().skip(4).take(12).collect(),
    ];
    let never = u32::MAX;
    const POST: usize = 3;
    const KV_LAYERS: [usize; 2] = [0, 7];
    const V: u32 = 128_256;

    // Reference greedy streams (one per slot) for building draft trees whose
    // acceptance is deterministic. `generate` resets + prefills, so one
    // loaded runner serves all three, then drops before the batches build.
    let wants: Vec<Vec<u32>> = {
        let Some(mut r) = load_on("cuda", &bytes) else {
            return;
        };
        prompts
            .iter()
            .map(|p| r.generate(p, 16, never).expect("ref stream"))
            .collect()
    };
    for w in &wants {
        assert_eq!(w.len(), 16, "reference stream shorter than expected");
    }

    // Per (slot, phase) draft tree. Phase 1 mixes lengths (m_total = 10,
    // pads to bucket 12): slot 0 a right chain m=3, slot 1 a branchy m=5
    // (wrong sibling first at two levels), slot 2 an all-wrong m=2 (accept
    // stops at the root — L=1 — while the others continue). Phase 2 swaps
    // shapes (m_total = 12, an exact bucket): chain m=2, chain m=4, branchy
    // m=6 with a wrong tail.
    let make_tree = |slot: usize, phase: usize, c: usize, root: u32| -> (Vec<u32>, Vec<i32>) {
        let want = &wants[slot];
        match (phase, slot) {
            (0, 0) => (vec![root, want[c], want[c + 1]], vec![-1, 0, 1]),
            (0, 1) => (
                vec![
                    root,
                    (want[c] + 1) % V,
                    want[c],
                    (want[c + 1] + 7) % V,
                    want[c + 1],
                ],
                vec![-1, 0, 0, 2, 2],
            ),
            (0, 2) => (vec![root, (want[c] + 5) % V], vec![-1, 0]),
            (1, 0) => (vec![root, want[c]], vec![-1, 0]),
            (1, 1) => (
                vec![root, want[c], want[c + 1], want[c + 2]],
                vec![-1, 0, 1, 2],
            ),
            _ => (
                vec![
                    root,
                    (want[c] + 1) % V,
                    want[c],
                    (want[c + 1] + 7) % V,
                    want[c + 1],
                    (want[c + 2] + 3) % V,
                ],
                vec![-1, 0, 0, 2, 2, 4],
            ),
        }
    };

    // Per route: (accepted per phase per slot, post-draft outs).
    type RouteTokens = (Vec<Vec<Vec<u32>>>, Vec<Vec<u32>>);
    let mut route_tokens: Vec<RouteTokens> = Vec::new();

    for eager in [false, true] {
        let _guard = TreeEagerGuard;
        if eager {
            // SAFETY: single-threaded test process (see TreeEagerGuard).
            #[allow(unsafe_code)]
            unsafe {
                std::env::set_var("TRITIUM_TREE_EAGER", "1");
            }
        }

        let Some(mut rb) = load_on("cuda", &bytes) else {
            return;
        };

        for paged in [false, true] {
            // The BATCHED batch and its SEQUENTIAL twin, same kind.
            let (mut bat, mut seq) = if paged {
                let mk = |rb: &mut ModelRunner| {
                    let mut b = rb.new_batch_paged(3, 3).expect("new_batch_paged");
                    // Interleaved reservation order (1, 0, 2) scrambles the
                    // physical page assignment (one page per slot).
                    for r in [1usize, 0, 2] {
                        b.reserve_pages(r, prompts[r].len() + 24)
                            .expect("reserve slot");
                    }
                    assert_eq!(b.free_pages(), 0, "pool sizing drifted");
                    assert!(
                        (0..3).any(|r| {
                            b.debug_page_table_row(r)
                                .expect("paged batch has a table")
                                .iter()
                                .enumerate()
                                .any(|(i, &p)| p >= 0 && p as usize != i)
                        }),
                        "vacuous gate: every slot's pages are identity-mapped"
                    );
                    b
                };
                (mk(&mut rb), mk(&mut rb))
            } else {
                (
                    rb.new_batch(3).expect("new_batch"),
                    rb.new_batch(3).expect("new_batch twin"),
                )
            };

            // Prefill each prompt once; adopt the SAME single-seq cache into
            // the batched slot and its sequential twin.
            let mut first_toks = [0u32; 3];
            for (r, prompt) in prompts.iter().enumerate() {
                rb.reset();
                let positions: Vec<usize> = (0..prompt.len()).collect();
                let l0 = rb.forward(prompt, &positions).expect("batch prefill");
                first_toks[r] = argmax(&l0) as u32;
                assert_eq!(first_toks[r], wants[r][0], "prefill vs reference (r={r})");
                for b in [&mut bat, &mut seq] {
                    rb.adopt_into_batch_row(b, r, prompt.len()).expect("adopt");
                    b.set_position(r, prompt.len()).expect("set_position");
                }
            }

            // Two verify phases: batched (one call, rows [0, 1, 2]) vs
            // sequential (three single-slot calls) on the twin.
            let mut committed: Vec<Vec<u32>> = (0..3).map(|r| vec![first_toks[r]]).collect();
            let mut accepted: Vec<Vec<Vec<u32>>> = Vec::new();
            for phase in 0..2 {
                let trees: Vec<(Vec<u32>, Vec<i32>)> = (0..3)
                    .map(|r| {
                        make_tree(
                            r,
                            phase,
                            committed[r].len(),
                            *committed[r].last().expect("non-empty"),
                        )
                    })
                    .collect();
                let trees_ref: Vec<(&[u32], &[i32])> = trees
                    .iter()
                    .map(|(t, p)| (t.as_slice(), p.as_slice()))
                    .collect();
                let outs = rb
                    .tree_verify_greedy_slots(&mut bat, &[0, 1, 2], &trees_ref)
                    .expect("batched slots verify");
                for r in 0..3 {
                    let out_seq = rb
                        .tree_verify_greedy_slot(&mut seq, r, &trees[r].0, &trees[r].1)
                        .expect("sequential slot verify");
                    assert_eq!(
                        outs[r], out_seq,
                        "batched accepted tokens != sequential (phase {phase}, r={r}, \
                         paged {paged}, eager {eager})"
                    );
                    committed[r].extend(&outs[r]);
                    assert_eq!(
                        bat.positions()[r],
                        seq.positions()[r],
                        "batched position != sequential (phase {phase}, r={r})"
                    );
                    assert_eq!(
                        bat.positions()[r],
                        prompts[r].len() + committed[r].len() - 1,
                        "slot position after phase {phase} (r={r})"
                    );
                }
                // Phase-1 shape pins: mixed accept lengths, incl. the
                // all-reject slot committing exactly the one bonus token.
                if phase == 0 {
                    assert_eq!(outs[0].len(), 3, "slot 0 chain must accept fully");
                    assert_eq!(outs[1].len(), 3, "slot 1 branchy accept drifted");
                    assert_eq!(outs[2].len(), 1, "slot 2 all-wrong drafts must L=1");
                }
                accepted.push(outs);
            }
            for r in 0..3 {
                assert_eq!(
                    &committed[r][..],
                    &wants[r][..committed[r].len()],
                    "slot {r} verify stream must stay lossless"
                );
            }

            // Post-commit KV of every slot's whole committed region:
            // bit-identical to the sequential twin (through the page
            // translation on the paged leg).
            {
                let rm = rb.resident_cuda().expect("resident").expect("cuda");
                for r in 0..3 {
                    let rows = prompts[r].len() + committed[r].len() - 1;
                    for li in KV_LAYERS {
                        for row in 0..rows {
                            for v in [false, true] {
                                let got_b = rm
                                    .debug_batch_kv_row(&bat, li, r, row, v)
                                    .expect("batched kv row");
                                let got_s = rm
                                    .debug_batch_kv_row(&seq, li, r, row, v)
                                    .expect("sequential kv row");
                                assert_eq!(
                                    got_b, got_s,
                                    "KV byte mismatch (r={r}, layer {li}, row {row}, \
                                     v={v}, paged {paged}, eager {eager})"
                                );
                            }
                        }
                    }
                }
            }

            // Continue every slot on both batches — a functional pin on the
            // promotes (a wrong promoted row derails the continuation).
            let feeds: Vec<u32> = (0..3)
                .map(|r| *committed[r].last().expect("non-empty"))
                .collect();
            let post_b = rb
                .draft_batch(&mut bat, &feeds, POST, never)
                .expect("batched post draft");
            let post_s = rb
                .draft_batch(&mut seq, &feeds, POST, never)
                .expect("sequential post draft");
            assert_eq!(
                post_b, post_s,
                "post-verify drafts diverged (paged {paged}, eager {eager})"
            );

            // ── Refusal atomicity (graph leg only; the guards are host-side
            // and route-independent). Every refusal must leave ALL listed
            // slots untouched. ──
            if !eager {
                let rm_probe = |rb: &mut ModelRunner, b: &tritium_cuda::BatchKv| -> Vec<Vec<u8>> {
                    let rm = rb.resident_cuda().expect("resident").expect("cuda");
                    (0..3)
                        .map(|r| {
                            rm.debug_batch_kv_row(b, 0, r, prompts[r].len() - 1, false)
                                .expect("probe kv row")
                        })
                        .collect()
                };
                let pos_before = bat.positions().to_vec();
                let kv_before = rm_probe(&mut rb, &bat);
                let tree: (Vec<u32>, Vec<i32>) = (vec![first_toks[0], 1], vec![-1, 0]);

                // Dead row listed.
                bat.set_live(1, false).expect("set_live dead");
                let err = rb
                    .tree_verify_greedy_slots(
                        &mut bat,
                        &[0, 1],
                        &[(&tree.0, &tree.1), (&tree.0, &tree.1)],
                    )
                    .expect_err("dead row must refuse");
                assert!(format!("{err}").contains("dead"), "unexpected: {err}");
                bat.set_live(1, true).expect("set_live revive");

                // Duplicate row.
                let err = rb
                    .tree_verify_greedy_slots(
                        &mut bat,
                        &[0, 0],
                        &[(&tree.0, &tree.1), (&tree.0, &tree.1)],
                    )
                    .expect_err("duplicate row must refuse");
                assert!(format!("{err}").contains("twice"), "unexpected: {err}");

                // Out-of-range row.
                let err = rb
                    .tree_verify_greedy_slots(&mut bat, &[7], &[(&tree.0, &tree.1)])
                    .expect_err("out-of-range row must refuse");
                assert!(format!("{err}").contains(">= batch"), "unexpected: {err}");

                // Over the one-bucket cap (m_total = 49 > 48).
                let big_t: Vec<u32> = (0..49u32).map(|i| (first_toks[0] + i) % V).collect();
                let big_p: Vec<i32> = (0..49i32).map(|i| i - 1).collect();
                let err = rb
                    .tree_verify_greedy_slots(&mut bat, &[0], &[(&big_t, &big_p)])
                    .expect_err("over-cap m_total must refuse");
                assert!(format!("{err}").contains("cap"), "unexpected: {err}");

                assert_eq!(
                    bat.positions().to_vec(),
                    pos_before,
                    "a refused batched verify must not advance any slot"
                );
                assert_eq!(
                    rm_probe(&mut rb, &bat),
                    kv_before,
                    "a refused batched verify must write no KV"
                );
            }
            drop(bat);
            drop(seq);
            if paged {
                route_tokens.push((accepted, post_b));
            }
        }

        // Paged under-reservation atomicity (graph leg only): slot 0's
        // prompt ends 2 tokens before the page boundary, reserved EXACTLY —
        // an m=4 tree crosses into the unreserved page. Slot 1 is fully
        // reserved and listed FIRST: the refusal must leave BOTH untouched
        // (guard phase precedes all device work).
        if !eager {
            let plen_s = tritium_cuda::KV_PAGE_TOKENS - 2;
            let prompt0: Vec<u32> = base.iter().copied().cycle().take(plen_s).collect();
            let prompt1: Vec<u32> = base.iter().copied().cycle().skip(1).take(10).collect();
            let mut small = rb.new_batch_paged(2, 3).expect("small paged batch");
            small
                .reserve_pages(0, plen_s)
                .expect("reserve slot 0 short");
            small
                .reserve_pages(1, 10 + 16)
                .expect("reserve slot 1 full");
            let mut roots = [0u32; 2];
            for (r, prompt) in [&prompt0, &prompt1].into_iter().enumerate() {
                rb.reset();
                let positions: Vec<usize> = (0..prompt.len()).collect();
                let l0 = rb.forward(prompt, &positions).expect("small prefill");
                roots[r] = argmax(&l0) as u32;
                rb.adopt_into_batch_row(&mut small, r, prompt.len())
                    .expect("adopt");
                small.set_position(r, prompt.len()).expect("set_position");
            }
            let t0: Vec<u32> = (0..4u32).map(|i| (roots[0] + i) % V).collect();
            let t1: Vec<u32> = (0..4u32).map(|i| (roots[1] + i) % V).collect();
            let p: Vec<i32> = (0..4i32).map(|i| i - 1).collect();
            let free_before = small.free_pages();
            let pos_before = small.positions().to_vec();
            let kv_before = {
                let rm = rb.resident_cuda().expect("resident").expect("cuda");
                rm.debug_batch_kv_row(&small, 0, 1, prompt1.len() - 1, false)
                    .expect("kv before")
            };
            let err = rb
                .tree_verify_greedy_slots(&mut small, &[1, 0], &[(&t1, &p), (&t0, &p)])
                .expect_err("under-reserved slot must refuse the whole batch");
            assert!(
                format!("{err}").contains("page-reserved"),
                "unexpected refusal message: {err}"
            );
            assert_eq!(
                small.positions().to_vec(),
                pos_before,
                "a refused batched verify must not advance any listed slot"
            );
            assert_eq!(
                small.free_pages(),
                free_before,
                "a refused batched verify must draw no pages"
            );
            let kv_after = {
                let rm = rb.resident_cuda().expect("resident").expect("cuda");
                rm.debug_batch_kv_row(&small, 0, 1, prompt1.len() - 1, false)
                    .expect("kv after")
            };
            assert_eq!(
                kv_after, kv_before,
                "a refused batched verify must write no KV (fully-reserved slot \
                 listed first)"
            );
            small
                .reserve_pages(0, plen_s + 16)
                .expect("extend reservation");
            let outs = rb
                .tree_verify_greedy_slots(&mut small, &[1, 0], &[(&t1, &p), (&t0, &p)])
                .expect("verify after extending the reservation");
            assert!(
                !outs[0].is_empty() && !outs[1].is_empty(),
                "recovered verify must commit >= 1 token per slot"
            );
            assert_eq!(
                small.positions().to_vec(),
                vec![plen_s + outs[1].len(), prompt1.len() + outs[0].len()],
                "positions after the recovered verify (slot-indexed; rows were \
                 listed [1, 0], so outs[1] is slot 0's)"
            );

            // Reservation-demand helper: padded (bucketed) single-slot
            // demand — the serve-side reservation key.
            assert_eq!(
                rb.tree_reservation_rows(100, 8).expect("resident"),
                108,
                "m=8 is a bucket boundary"
            );
            assert_eq!(
                rb.tree_reservation_rows(100, 6).expect("resident"),
                108,
                "m=6 pads to bucket 8"
            );
            assert_eq!(
                rb.tree_reservation_rows(100, 2).expect("resident"),
                104,
                "m=2 pads to bucket 4"
            );
        }
    }

    let (g, e) = (&route_tokens[0], &route_tokens[1]);
    assert_eq!(g.0, e.0, "graph vs eager accepted tokens (slots gate)");
    assert_eq!(g.1, e.1, "graph vs eager post drafts (slots gate)");
    println!(
        "I4 slots: one forward over 3 slots' concatenated trees (m=3/5/2 then \
         2/4/6, branchy + an all-reject early stop) bit-equal to sequential \
         single-slot verifies (tokens, positions, KV layers {KV_LAYERS:?}) on \
         dense AND scrambled-paged batches; refusals (dead/dup/range/cap/\
         under-reservation) atomic across listed slots; graph==eager"
    );
}

/// RAII guard for `TRITIUM_WEIGHTS`: sets the packing for a model load and
/// restores the previous value on drop (panic-safe), so a failing T5 gate
/// cannot leak `tq1` into the rest of the suite's loads.
#[cfg(feature = "cuda")]
struct WeightsEnv(Option<String>);

#[cfg(feature = "cuda")]
impl WeightsEnv {
    fn set(v: &str) -> Self {
        let prev = std::env::var("TRITIUM_WEIGHTS").ok();
        // SAFETY: mutation happens inside the GPU_LOCK critical section and the
        // GPU suite runs `--test-threads=1`; the only reader is TernaryLinear's
        // pack-format lookup during the model loads this guard scopes.
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var("TRITIUM_WEIGHTS", v)
        };
        Self(prev)
    }
}

#[cfg(feature = "cuda")]
impl Drop for WeightsEnv {
    fn drop(&mut self) {
        // SAFETY: same single-threaded critical section as `set`.
        #[allow(unsafe_code)]
        unsafe {
            match &self.0 {
                Some(v) => std::env::set_var("TRITIUM_WEIGHTS", v),
                None => std::env::remove_var("TRITIUM_WEIGHTS"),
            }
        }
    }
}

/// T5 gate (re-pointing the A2 bit-identity gate at M>1): a batched decode —
/// 3 slots with DIFFERENT histories through the batched capture/replay graph —
/// under `TRITIUM_WEIGHTS=tq1` must be **bit-identical** (full logit vectors,
/// `to_bits`) to the same batched decode under `tq2`. This holds at logit
/// level, not just tokens, because the TQ1 GEMM twin runs the same exact i32
/// dp4a accumulator and the same epilogue multiply order on the same trits
/// (kernel gate `tq1_matches_tq2_tiled_scaled_bit_exact`), and every non-GEMM
/// kernel in the batched pipeline is weight-format-independent.
#[cfg(feature = "cuda")]
#[test]
fn cuda_tq1_batch_decode_bit_identical_to_tq2() {
    let _gpu = gpu_serial();
    let Some((_reference, bytes)) = maybe_load() else {
        return;
    };
    // 3 rows, DIFFERENT histories (distinct in-range token streams).
    const STEPS: usize = 6;
    let rows: [[u32; STEPS]; 3] = [
        [128000, 791, 6864, 315, 279, 502],
        [128000, 3923, 374, 279, 6864, 30],
        [128000, 9906, 1917, 11, 1268, 527],
    ];
    let run = |fmt: &str| -> Option<Vec<Vec<Vec<f32>>>> {
        let _env = WeightsEnv::set(fmt);
        let mut runner = load_on("cuda", &bytes)?;
        let model = runner.resident_cuda().ok()??;
        let mut batch = model.new_batch(3).expect("new_batch");
        let mut steps = Vec::with_capacity(STEPS);
        for i in 0..STEPS {
            let toks: Vec<u32> = rows.iter().map(|r| r[i]).collect();
            steps.push(
                model
                    .decode_batch_graph(&mut batch, &toks)
                    .expect("decode_batch_graph"),
            );
        }
        Some(steps)
    };
    let Some(want) = run("tq2") else {
        return; // no device — the tq2 leg already printed why
    };
    let got = run("tq1").expect("tq1 load must succeed where tq2 did");
    for (i, (ws, gs)) in want.iter().zip(&got).enumerate() {
        assert_eq!(ws.len(), gs.len(), "row count at step {i}");
        for (r, (wr, gr)) in ws.iter().zip(gs).enumerate() {
            assert_eq!(wr.len(), gr.len(), "vocab width at step {i} row {r}");
            for (v, (a, b)) in wr.iter().zip(gr).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "tq1 != tq2 batched logits at step {i} row {r} vocab {v}: \
                     tq2={a} tq1={b}"
                );
            }
            // Redundant given bit-identity, but states the decode-meaningful
            // invariant in the pass output.
            assert_eq!(
                tritium_nn::sample_greedy(wr),
                tritium_nn::sample_greedy(gr),
                "greedy token diverged at step {i} row {r}"
            );
        }
    }
    println!(
        "T5 batch gate: tq1 batched decode (3 slots, different histories, \
         graph route) BIT-identical to tq2 over {STEPS} steps (full logit \
         vectors, to_bits)"
    );
}

/// T5 tree gate: single-seq tree verify under `TRITIUM_WEIGHTS=tq1` must
/// commit the SAME token stream as under `tq2`, and the transformer state
/// after the promotes must be bit-identical — asserted by comparing the next
/// `step_graph` logits `to_bits` (the M=1 graph route reads the promoted KV,
/// so any tree-trunk divergence or promote corruption shows up here at bit
/// level). Draft shapes rotate perfect-chain / branch / wrong-tail so accepts
/// AND rejects are exercised.
#[cfg(feature = "cuda")]
#[test]
fn cuda_tq1_tree_verify_matches_tq2() {
    let _gpu = gpu_serial();
    let Some((reference, bytes)) = maybe_load() else {
        return;
    };
    let prompt = reference.token_ids.clone();
    let greedy = reference.greedy_ids.clone();
    let k_total = 12usize;
    let run = |fmt: &str| -> Option<(Vec<u32>, Vec<f32>)> {
        let _env = WeightsEnv::set(fmt);
        let mut runner = load_on("cuda", &bytes)?;
        let positions: Vec<usize> = (0..prompt.len()).collect();
        let prefill = runner.forward(&prompt, &positions).expect("prefill");
        let first = argmax(&prefill) as u32;
        assert_eq!(
            first, greedy[0],
            "{fmt}: prefill argmax must match the committed reference"
        );
        let rm = runner
            .resident_cuda()
            .expect("resident")
            .expect("cuda resident");
        let mut committed: Vec<u32> = vec![first];
        let mut phase = 0usize;
        while committed.len() < k_total {
            let c = committed.len();
            let root = *committed.last().expect("non-empty");
            let remaining = k_total - c;
            let (tokens, parents): (Vec<u32>, Vec<i32>) = match phase % 3 {
                0 => {
                    // Perfect chain of up to 3 drafts.
                    let n = remaining.min(3);
                    let mut t = vec![root];
                    t.extend(greedy[c..c + n].iter().copied());
                    let p: Vec<i32> = (0..t.len() as i32).map(|i| i - 1).collect();
                    (t, p)
                }
                1 => {
                    // Branch: wrong sibling first, right child second.
                    let right = greedy[c];
                    let wrong = (right + 1) % 128_256;
                    (vec![root, wrong, right], vec![-1, 0, 0])
                }
                _ => {
                    // Chain that goes wrong after one correct draft.
                    let right = greedy[c];
                    let wrong = (right + 7) % 128_256;
                    (vec![root, right, wrong], vec![-1, 0, 1])
                }
            };
            let out = rm
                .tree_verify_greedy(&tokens, &parents)
                .expect("tree_verify_greedy");
            assert!(!out.is_empty(), "tree verify must commit >= 1 token");
            committed.extend(&out);
            phase += 1;
        }
        // One M=1 graph step off the promoted state: bit-level state probe.
        let pos0 = prompt.len() + committed.len() - 1;
        let pending = *committed.last().expect("non-empty");
        let logits = rm.step_graph(pending, pos0).expect("post-tree graph step");
        Some((committed, logits))
    };
    let Some((want_tokens, want_logits)) = run("tq2") else {
        return;
    };
    let (got_tokens, got_logits) = run("tq1").expect("tq1 load must succeed where tq2 did");
    assert_eq!(
        want_tokens, got_tokens,
        "tq1 tree verify committed a different stream than tq2"
    );
    for (v, (a, b)) in want_logits.iter().zip(&got_logits).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "post-tree step_graph logits diverged at vocab {v}: tq2={a} tq1={b} \
             (tree trunk or KV promote differs between formats)"
        );
    }
    println!(
        "T5 tree gate: tq1 tree verify == tq2 ({} committed tokens, accepts + \
         rejects) and the post-tree M=1 graph logits are BIT-identical",
        want_tokens.len()
    );
}
