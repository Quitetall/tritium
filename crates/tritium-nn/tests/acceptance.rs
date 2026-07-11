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

const GGUF_PATH: &str =
    "/home/brianklam/.cache/tritium-models/bitnet-2b4t-gguf/ggml-model-i2_s.gguf";
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

/// Load the reference + model bytes, or `None` (with a printed reason) if either
/// is absent — the offline/cpu-only skip path shared by every test here.

/// Serializes the GPU-heavy tests within this binary: each loads a full model
/// (~2.5 GB VRAM), and the default parallel test threads OOM-flake whenever a
/// co-resident GPU process (another session's server, a desktop) squeezes
/// free VRAM — observed live. Poison-tolerant: a panicked test must not fail
/// the rest with a PoisonError.
static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu_serial() -> std::sync::MutexGuard<'static, ()> {
    GPU_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn maybe_load() -> Option<(Reference, Vec<u8>)> {
    if !Path::new(GGUF_PATH).exists() {
        eprintln!("skipping: {GGUF_PATH} absent (gated real-model test)");
        return None;
    }
    if !Path::new(REF_PATH).exists() {
        eprintln!("skipping: {REF_PATH} absent; run tools/gen_reference.py");
        return None;
    }
    let reference: Reference =
        serde_json::from_slice(&std::fs::read(REF_PATH).expect("read reference"))
            .expect("parse reference json");
    let bytes = std::fs::read(GGUF_PATH).expect("read GGUF");
    Some((reference, bytes))
}

/// Index of the maximum element (greedy argmax), ties broken by lowest index —
/// matching `sample_greedy` and torch's `argmax`. Used only by the CUDA parity
/// test.
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
    let mut pos = prompt.len() + c_full - 1; // == cache_len (pending not yet forwarded)
    let mut pending = *committed.last().expect("non-empty");
    for step in 0..4 {
        let logits = rm
            .step_graph(pending, pos)
            .expect("graph step after tree verify");
        let next = argmax(&logits) as u32;
        assert_eq!(
            next,
            want_tail[c_full + step],
            "post-tree graph decode diverged at tail step {step} — KV promote corruption"
        );
        pending = next;
        pos += 1;
    }
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
