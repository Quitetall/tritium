//! End-to-end tokens/sec on BitNet 2B4T (v0.30 WF-E, ADR 0005) — `cuda` + model
//! gated, divan.
//!
//! Reuses the v0.20 [`ModelRunner`] on the **`"cuda"`** backend to measure the two
//! throughput numbers the milestone cares about, each **coupled to an
//! unchanged-perplexity assertion** so no accuracy is silently traded for speed:
//!
//! - **decode tokens/sec** — single-token autoregressive steps through the KV cache
//!   (the memory-bound batch-1 path the device-resident decode targets). Reported
//!   against the [`bitnet_2b4t_decode_ceiling`] roofline (%-of-SOL denominator) and
//!   checked for a `>5%` regression vs our own committed [`TRITIUM_2B4T_DECODE_4090`]
//!   `BuiltOnBox` figure (the competitor numbers are CPU / have no GPU ternary path,
//!   so they are printed as context, not used as the GPU gate denominator).
//! - **prefill tokens/sec** — one forward over the whole prompt (the compute-bound
//!   path the IMMA kernel targets at larger `M`).
//!
//! **Perplexity coupling.** Before timing, the bench computes teacher-forced
//! perplexity over the committed eval sequence and asserts it is within 1% of the
//! `transformers` reference — the *same* gate `tests/acceptance.rs` enforces. A perf
//! number is only meaningful if the model still produces the right distribution, so
//! a perplexity drift `panic!`s the bench rather than reporting a fast-but-wrong
//! tok/s.
//!
//! ## Gating
//!
//! The body is `#[cfg(feature = "cuda")]`; without the feature this file is an empty
//! `divan::main()`. With the feature it additionally requires the GGUF + reference
//! JSON to exist (else it prints why and skips) and a working CUDA device (else
//! backend init returns `Err` and it skips). So cpu-only lanes, GPU-less boxes, and
//! model-less boxes all skip cleanly instead of failing.

#![cfg_attr(not(feature = "cuda"), allow(unused_crate_dependencies))]

fn main() {
    divan::main();
}

#[cfg(feature = "cuda")]
mod cuda_e2e {
    use std::path::Path;

    use divan::{Bencher, counter::ItemsCount};
    use tritium_benches::{
        BITNET_CPP_2B4T_DECODE, REGRESSION_DROP_THRESHOLD, TRITIUM_2B4T_DECODE_4090,
        bitnet_2b4t_decode_ceiling, check_regression,
    };
    use tritium_nn::ModelRunner;

    // Pull the CUDA backend's registration into the bench binary.
    use tritium_cuda as _;

    /// Model cache root: override via `TRITIUM_MODEL_DIR`; default `~/.cache/tritium-models`; benches skip cleanly when absent.
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
        "/../tools/reference/bitnet_accept.json"
    );

    /// Decode steps timed per bench iteration. Each is one single-token forward
    /// through the KV cache (~0.2 s/token with the add-only kernel on a 4090), so a
    /// handful is enough for a stable per-token rate without a multi-minute bench.
    const DECODE_STEPS: usize = 8;

    /// The committed reference (subset of `tools/reference/bitnet_accept.json`).
    #[derive(serde::Deserialize)]
    struct Reference {
        /// Prompt token IDs (includes BOS) — the prefill input + decode seed.
        token_ids: Vec<u32>,
        /// The fixed eval sequence perplexity is measured over (prompt + cont).
        eval_ids: Vec<u32>,
        /// `transformers` reference perplexity over `eval_ids` (fp32 CPU oracle).
        perplexity: f64,
    }

    /// Load the reference + model bytes, or `None` (with a printed reason) if either
    /// is absent — the offline/model-less skip path.
    fn maybe_load() -> Option<(Reference, Vec<u8>)> {
        if !Path::new(&*GGUF_PATH).exists() {
            eprintln!(
                "skipping e2e bench: {} absent (gated real-model bench)",
                &*GGUF_PATH
            );
            return None;
        }
        if !Path::new(REF_PATH).exists() {
            eprintln!("skipping e2e bench: {REF_PATH} absent; run tools/gen_reference.py");
            return None;
        }
        let reference: Reference =
            serde_json::from_slice(&std::fs::read(REF_PATH).expect("read reference"))
                .expect("parse reference json");
        let bytes = std::fs::read(&*GGUF_PATH).expect("read GGUF");
        Some((reference, bytes))
    }

    /// Build a `ModelRunner` on the `"cuda"` backend from already-read GGUF `bytes`,
    /// or `None` if the backend did not init (no device → skip, not fail).
    fn load_cuda(bytes: &[u8]) -> Option<ModelRunner> {
        let init = tritium_runtime::BACKENDS
            .iter()
            .find(|e| e.name == "cuda")
            .map(|e| e.init)?;
        let backend = match init() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping e2e bench: cuda backend init failed ({e}); no device?");
                return None;
            }
        };
        let file = tritium_format::read_gguf(bytes).expect("parse gguf");
        Some(ModelRunner::load(&file, bytes, backend).expect("load model"))
    }

    /// Numerically-stable `log P(target) = logit[target] - logsumexp(logits)`.
    fn log_prob_of(logits: &[f32], target: usize) -> f64 {
        let max = logits.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x)) as f64;
        let mut sum = 0.0f64;
        for &l in logits {
            sum += ((l as f64) - max).exp();
        }
        (logits[target] as f64) - max - sum.ln()
    }

    /// Teacher-forced perplexity of `runner` over `eval_ids` — `exp(-mean log P)` of
    /// the true next token at each position, stepped token-by-token through the KV
    /// cache exactly as the `transformers` oracle scores with `use_cache`. Identical
    /// to `tests/acceptance.rs::perplexity_over`, the coupling gate.
    fn perplexity_over(runner: &mut ModelRunner, eval_ids: &[u32]) -> f64 {
        runner.reset();
        let n = eval_ids.len();
        assert!(n >= 2, "perplexity needs at least 2 tokens");
        let mut neg_log_sum = 0.0f64;
        let mut count = 0usize;
        let mut logits = runner
            .forward(&eval_ids[..1], &[0])
            .expect("perplexity prefill");
        for t in 0..n - 1 {
            let target = eval_ids[t + 1] as usize;
            neg_log_sum -= log_prob_of(&logits, target);
            count += 1;
            if t + 1 < n - 1 {
                logits = runner
                    .forward(&[eval_ids[t + 1]], &[t + 1])
                    .expect("perplexity decode");
            }
        }
        (neg_log_sum / count as f64).exp()
    }

    /// Assert perplexity is within 1% of the reference — the unchanged-accuracy gate
    /// every perf number here is coupled to. `panic!`s (fails the bench) on drift, so
    /// a fast-but-wrong configuration can never report a green tokens/sec.
    fn assert_perplexity_unchanged(runner: &mut ModelRunner, reference: &Reference) {
        let ppl = perplexity_over(runner, &reference.eval_ids);
        let want = reference.perplexity;
        let rel = (ppl - want).abs() / want;
        println!("e2e perplexity ours={ppl:.6} ref={want:.6} rel={rel:.3e}");
        assert!(
            rel <= 0.01,
            "perplexity {ppl:.6} not within 1% of reference {want:.6} (rel {rel:.3e}) — \
             accuracy regressed, tokens/sec is meaningless"
        );
    }

    /// **Decode tokens/sec.** Times [`DECODE_STEPS`] single-token forwards after a
    /// prompt prefill, the batch-1 memory-bound path. The `ItemsCount` of
    /// `DECODE_STEPS` makes divan print tokens/sec directly. Couples to perplexity
    /// once up-front, then computes a **decode-only** rate (the per-iteration prompt
    /// prefill is excluded from the accumulated time, so the figure is apples-to-apples
    /// with what bitnet.cpp reports), prints the %-of-roofline, and **`panic!`s on a
    /// `>5%` regression** vs the committed competitor baseline — so the scheduled CI
    /// lane that runs this bench actually fails on a baseline drop (ADR 0005's
    /// "perf-regression job fails on a >5% tokens/sec drop").
    ///
    /// Note the decode-only rate (and thus the regression `assert!`) is computed from
    /// the time/tokens **accumulated across all divan iterations**, so the gate fires
    /// once after the harness completes, not per-iteration. divan's own throughput line
    /// (its `ItemsCount`) includes each iteration's prefill in the wall clock and so
    /// reads lower than this decode-only figure; the printed `decode ≈ … tok/s` line is
    /// the authoritative one for the gate.
    #[divan::bench]
    fn decode_tokens_per_sec(bencher: Bencher) {
        let Some((reference, bytes)) = maybe_load() else {
            return;
        };
        let Some(mut runner) = load_cuda(&bytes) else {
            return;
        };

        // Accuracy gate first: a regressed model makes any tok/s meaningless.
        assert_perplexity_unchanged(&mut runner, &reference);

        // Roofline + baseline context (printed once; uses the actual loaded bytes for
        // the ceiling so it tracks the real model on disk). The regression gate keys on
        // our own `BuiltOnBox` figure (`TRITIUM_2B4T_DECODE_4090`): there is no
        // obtainable GPU *ternary* competitor — llama.cpp's CUDA backend has no
        // TQ/I2_S mul-mat kernel and cannot load this artifact (see the baseline's
        // doc-comment), and bitnet.cpp's published numbers are CPU. Those are printed as
        // context only; the gate is "don't regress vs our measured decode".
        let ceiling = bitnet_2b4t_decode_ceiling();
        println!(
            "decode roofline ceiling ≈ {ceiling:.1} tok/s (peak HBM BW / model bytes); \
             regression gate `{}` = {:.1} tok/s ({:?}); competitor context: `{}` {:.1} tok/s (CPU, {:?})",
            TRITIUM_2B4T_DECODE_4090.name,
            TRITIUM_2B4T_DECODE_4090.tokens_per_sec,
            TRITIUM_2B4T_DECODE_4090.source,
            BITNET_CPP_2B4T_DECODE.name,
            BITNET_CPP_2B4T_DECODE.tokens_per_sec,
            BITNET_CPP_2B4T_DECODE.source,
        );

        // Reset + re-prefill each iteration so every timed run decodes from the same
        // cache state — but accumulate **only the decode interval** (prefill is timed
        // separately and excluded) so the external rate matches bitnet.cpp's
        // decode-only number. `decode_secs` / `decode_tokens` is the authoritative
        // decode tok/s the regression gate keys on.
        let prompt = reference.token_ids.clone();
        let positions: Vec<usize> = (0..prompt.len()).collect();
        let mut decode_secs = 0.0f64;
        let mut decode_tokens = 0u64;

        bencher
            .counter(ItemsCount::new(DECODE_STEPS))
            .bench_local(|| {
                runner.reset();
                let _ = runner.forward(&prompt, &positions).expect("prefill");
                // Time only the decode steps, not the prefill above.
                let d0 = std::time::Instant::now();
                for step in 0..DECODE_STEPS {
                    let pos = prompt.len() + step;
                    // Feed a fixed token id (0) at the next position: we time the
                    // forward, not the sampled content, so determinism of the *input*
                    // is all that matters for a stable rate.
                    let _ = runner.forward(&[0u32], &[pos]).expect("decode step");
                }
                decode_secs += d0.elapsed().as_secs_f64();
                decode_tokens += DECODE_STEPS as u64;
            });

        // Decode-only rate (prefill excluded) — the figure comparable to bitnet.cpp.
        if decode_tokens > 0 && decode_secs > 0.0 {
            let decode_tps = decode_tokens as f64 / decode_secs;
            let pct = 100.0 * decode_tps / ceiling;
            let report = check_regression(decode_tps, &TRITIUM_2B4T_DECODE_4090);
            println!(
                "decode ≈ {decode_tps:.1} tok/s (decode-only) — {pct:.1}% of roofline; \
                 vs baseline drop = {:.1}% ({})",
                100.0 * report.drop_fraction,
                if report.regressed { "REGRESSION" } else { "ok" },
            );
            // The gate: a >5% slowdown vs the recorded baseline fails the bench (and
            // thus the scheduled CI lane). A speedup never trips it. Our decode rate is
            // far above the conservative published competitor floor, so this fires only
            // on a genuine regression once a tighter `BuiltOnBox` baseline is recorded.
            assert!(
                !report.regressed,
                "PERF REGRESSION: decode {decode_tps:.1} tok/s is >{:.0}% below baseline `{}` \
                 ({:.1} tok/s)",
                REGRESSION_DROP_THRESHOLD * 100.0,
                report.baseline_name,
                report.baseline_tps,
            );
        }
    }

    /// **Prefill tokens/sec.** Times one forward over the whole prompt (the
    /// compute-bound large-`M` path). `ItemsCount` = prompt length so divan prints
    /// tokens/sec. Coupled to the same perplexity gate.
    #[divan::bench]
    fn prefill_tokens_per_sec(bencher: Bencher) {
        let Some((reference, bytes)) = maybe_load() else {
            return;
        };
        let Some(mut runner) = load_cuda(&bytes) else {
            return;
        };
        assert_perplexity_unchanged(&mut runner, &reference);

        let prompt = reference.token_ids.clone();
        let positions: Vec<usize> = (0..prompt.len()).collect();

        bencher
            .counter(ItemsCount::new(prompt.len()))
            .bench_local(|| {
                runner.reset();
                let _ = runner.forward(&prompt, &positions).expect("prefill");
            });
    }
}
