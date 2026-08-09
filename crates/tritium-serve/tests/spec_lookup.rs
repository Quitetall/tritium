//! Prompt-lookup speculative decoding gates (model + GPU gated, `cuda` feature).
//!
//! Losslessness: the spec-lookup stream must equal the plain greedy stream
//! token-for-token (every emitted token is the target's own argmax — the
//! BASTION verifier only ever commits those). Also prints both wall times; the
//! committed reference continuation is highly repetitive, so the lookup
//! drafter should land multi-token commits and beat plain decode.

#![cfg(feature = "cuda")]

use std::path::Path;

use tritium_serve::{GenRequest, Generator, RunnerGenerator, Sampling};

use tritium_cpu as _;
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

fn load_runner(bytes: &[u8]) -> Option<tritium_nn::ModelRunner> {
    let init = tritium_runtime::BACKENDS
        .iter()
        .find(|e| e.name == "cuda")
        .map(|e| e.init)?;
    let backend = match init() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping: cuda backend failed to init ({e})");
            return None;
        }
    };
    let file = tritium_format::read_gguf(bytes).expect("parse gguf");
    Some(tritium_nn::ModelRunner::load(&file, bytes, backend).expect("load model"))
}

fn collect(generator: &mut dyn Generator, req: &GenRequest) -> (Vec<u32>, std::time::Duration) {
    let mut out = Vec::new();
    let t0 = std::time::Instant::now();
    generator
        .generate(req, &mut |step| {
            out.push(step.token);
            true
        })
        .expect("generate");
    (out, t0.elapsed())
}

/// temp→0 gate for the SAMPLING accept rule: TopK{k:1} makes p̃ collapse to
/// the argmax candidate at probability 1, so the whole sampled machinery
/// (tree_verify_logits → host accept walk → tree_commit) becomes
/// deterministic and must reproduce the plain greedy stream token-for-token.
#[test]
fn cuda_spec_sampled_topk1_matches_plain_greedy() {
    if !Path::new(&*GGUF_PATH).exists() {
        eprintln!("skipping: {} absent (gated real-model test)", *GGUF_PATH);
        return;
    }
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(REF_PATH).expect("read reference"))
            .expect("parse reference");
    let prompt: Vec<u32> = reference["token_ids"]
        .as_array()
        .expect("token_ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect();
    let bytes = std::fs::read(&*GGUF_PATH).expect("read gguf");

    let greedy_req = GenRequest {
        prompt_tokens: prompt.clone(),
        max_new: 128,
        sampling: Sampling::Greedy,
        stop_eos: false,
        logprobs: None,
    };
    let sampled_req = GenRequest {
        prompt_tokens: prompt,
        max_new: 128,
        sampling: Sampling::TopK {
            k: 1,
            temp: 1.0,
            seed: 42,
        },
        stop_eos: false,
        logprobs: None,
    };

    let Some(runner) = load_runner(&bytes) else {
        return;
    };
    let mut plain = RunnerGenerator::new(runner, u32::MAX);
    let (want, _) = collect(&mut plain, &greedy_req);

    let Some(runner) = load_runner(&bytes) else {
        return;
    };
    let mut spec = RunnerGenerator::new(runner, u32::MAX).with_spec_lookup(true);
    let (got, t) = collect(&mut spec, &sampled_req);
    println!("spec-sampled k=1: {} tok in {t:.2?}", got.len());
    assert_eq!(
        got, want,
        "spec-sampled TopK{{k:1}} must equal plain greedy"
    );

    // Same seed twice → identical stream (the spec path is deterministic).
    let Some(runner) = load_runner(&bytes) else {
        return;
    };
    let mut spec2 = RunnerGenerator::new(runner, u32::MAX).with_spec_lookup(true);
    let (got2, _) = collect(&mut spec2, &sampled_req);
    assert_eq!(got2, got, "same-seed spec-sampled runs must be identical");
}

#[test]
fn cuda_spec_lookup_matches_plain_greedy() {
    if !Path::new(&*GGUF_PATH).exists() {
        eprintln!("skipping: {} absent (gated real-model test)", *GGUF_PATH);
        return;
    }
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(REF_PATH).expect("read reference"))
            .expect("parse reference");
    let prompt: Vec<u32> = reference["token_ids"]
        .as_array()
        .expect("token_ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect();
    let bytes = std::fs::read(&*GGUF_PATH).expect("read gguf");

    let req = GenRequest {
        prompt_tokens: prompt.clone(),
        max_new: 224,
        sampling: Sampling::Greedy,
        stop_eos: false,
        logprobs: None,
    };
    // Warmup request: builds the CUDA graph + JIT outside the timed runs.
    let warm = GenRequest {
        prompt_tokens: prompt,
        max_new: 4,
        sampling: Sampling::Greedy,
        stop_eos: false,
        logprobs: None,
    };

    let Some(runner) = load_runner(&bytes) else {
        return;
    };
    let mut plain = RunnerGenerator::new(runner, u32::MAX);
    let _ = collect(&mut plain, &warm);
    let (want, t_plain) = collect(&mut plain, &req);

    let Some(runner) = load_runner(&bytes) else {
        return;
    };
    let mut spec = RunnerGenerator::new(runner, u32::MAX).with_spec_lookup(true);
    let _ = collect(&mut spec, &warm);
    let (got, t_spec) = collect(&mut spec, &req);

    println!(
        "spec-lookup: plain {} tok in {t_plain:.2?} ({:.1} tok/s) | spec {} tok in {t_spec:.2?} ({:.1} tok/s) | speedup {:.2}x",
        want.len(),
        want.len() as f64 / t_plain.as_secs_f64(),
        got.len(),
        got.len() as f64 / t_spec.as_secs_f64(),
        t_plain.as_secs_f64() / t_spec.as_secs_f64(),
    );
    assert_eq!(
        got, want,
        "spec-lookup stream must equal plain greedy (lossless)"
    );
}

// ───────────── adaptive spec on/off governor (TRITIUM_SPEC_ADAPTIVE) ─────────────

/// Env guard for `TRITIUM_SPEC_ADAPTIVE`; restores the prior value on drop.
struct AdaptiveEnv(Option<std::ffi::OsString>);
impl AdaptiveEnv {
    fn set(v: &str) -> Self {
        let prev = std::env::var_os("TRITIUM_SPEC_ADAPTIVE");
        // SAFETY: these gated tests run single-threaded (--test-threads=1 —
        // the release-suite house rule); no other thread touches the
        // environment.
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var("TRITIUM_SPEC_ADAPTIVE", v);
        }
        Self(prev)
    }
}
impl Drop for AdaptiveEnv {
    fn drop(&mut self) {
        // SAFETY: single-threaded test (see `AdaptiveEnv::set`).
        #[allow(unsafe_code)]
        unsafe {
            match self.0.take() {
                Some(v) => std::env::set_var("TRITIUM_SPEC_ADAPTIVE", v),
                None => std::env::remove_var("TRITIUM_SPEC_ADAPTIVE"),
            }
        }
    }
}

fn ref_prompt() -> Vec<u32> {
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(REF_PATH).expect("read reference"))
            .expect("parse reference");
    reference["token_ids"]
        .as_array()
        .expect("token_ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect()
}

/// Adaptive-governor gate (the long-ctx τ-collapse lever):
///
/// Leg 1 (dormancy): with the governor at its default (ON) on the healthy
/// short-ctx fixture, the stream equals plain greedy AND the suppression
/// counter never moves — the lever must not fire where spec wins.
///
/// Leg 2 (forced collapse): `TRITIUM_SPEC_ADAPTIVE=force` classifies every
/// verify as collapsed — suppression must engage (observable: the
/// suppressed-plain counter moves), verifies must drop to the entry streak
/// plus the periodic probes, and the stream must STILL equal plain greedy:
/// suppression changes WHEN spec runs, never what is committed.
#[test]
fn cuda_spec_adaptive_forced_collapse_matches_plain_greedy() {
    if !Path::new(&*GGUF_PATH).exists() {
        eprintln!("skipping: {} absent (gated real-model test)", *GGUF_PATH);
        return;
    }
    let prompt = ref_prompt();
    let bytes = std::fs::read(&*GGUF_PATH).expect("read gguf");
    let max_new = 224usize;
    let req = GenRequest {
        prompt_tokens: prompt.clone(),
        max_new,
        sampling: Sampling::Greedy,
        stop_eos: false,
        logprobs: None,
    };
    let warm = GenRequest {
        prompt_tokens: prompt,
        max_new: 4,
        sampling: Sampling::Greedy,
        stop_eos: false,
        logprobs: None,
    };

    let Some(runner) = load_runner(&bytes) else {
        return;
    };
    let mut plain = RunnerGenerator::new(runner, u32::MAX);
    let _ = collect(&mut plain, &warm);
    let (want, _) = collect(&mut plain, &req);
    drop(plain);

    let suppressed = || {
        tritium_serve::generator::SPEC_SUPPRESSED_PLAIN.load(std::sync::atomic::Ordering::Relaxed)
    };
    let verifies =
        || tritium_serve::generator::SPEC_VERIFIES.load(std::sync::atomic::Ordering::Relaxed);

    // Leg 1 — default ON, healthy acceptance: dormant.
    {
        let _env = AdaptiveEnv::set("1");
        let Some(runner) = load_runner(&bytes) else {
            return;
        };
        let mut spec = RunnerGenerator::new(runner, u32::MAX).with_spec_lookup(true);
        let _ = collect(&mut spec, &warm);
        let s0 = suppressed();
        let (got, _) = collect(&mut spec, &req);
        assert_eq!(got, want, "adaptive-on short-ctx stream != plain greedy");
        assert_eq!(
            suppressed() - s0,
            0,
            "the governor fired on a healthy short-ctx run (must stay dormant)"
        );
    }

    // Leg 2 — forced collapse: suppression engages, stream stays exact.
    {
        let _env = AdaptiveEnv::set("force");
        let Some(runner) = load_runner(&bytes) else {
            return;
        };
        let mut spec = RunnerGenerator::new(runner, u32::MAX).with_spec_lookup(true);
        // No warm run: warm verifies would consume the entry streak already.
        let s0 = suppressed();
        let v0 = verifies();
        let (got, _) = collect(&mut spec, &req);
        let s_delta = suppressed() - s0;
        let v_delta = verifies() - v0;
        println!(
            "adaptive forced-collapse: {s_delta} suppressed-plain tokens, \
             {v_delta} verifies over {max_new} tokens"
        );
        assert_eq!(got, want, "forced-collapse stream != plain greedy");
        assert!(
            s_delta > 0,
            "forced collapse never engaged suppression (counter flat)"
        );
        // Entry costs exactly ENTRY_STREAK (4) verifies; after that only the
        // 64-token probes verify: <= 4 + ceil(224/64) + slack.
        assert!(
            v_delta <= 12,
            "{v_delta} verifies under forced collapse — suppression is not \
             actually stopping the verify traffic"
        );
    }
}

/// ABBA bench for the adaptive lever (ignored; run explicitly, serially):
///
/// ```text
/// cargo test -p tritium-serve --features cuda,serve --release --test spec_lookup \
///   cuda_spec_adaptive_bench -- --ignored --nocapture --test-threads=1
/// ```
///
/// Kernel tier / KV rung come from the ambient `TRITIUM_KERNEL_TIER` /
/// `TRITIUM_KV` (run once per tier). Two shapes, each ABBA
/// (on, off, off, on): short ctx (the lever must be dormant — within noise)
/// and long ctx ~3776 with a shuffled prompt (breaks the fixture's
/// periodicity so the drafter's acceptance collapses, the sweep's regime —
/// adaptive should close the spec-slowdown hole toward plain).
#[test]
#[ignore = "bench: run explicitly with --ignored --nocapture"]
fn cuda_spec_adaptive_bench() {
    if !Path::new(&*GGUF_PATH).exists() {
        eprintln!("skipping: {} absent (gated real-model test)", *GGUF_PATH);
        return;
    }
    let draft_path = format!(
        "{}/blut/data/drafter-8L768-s3.gguf",
        std::env::var("HOME").unwrap_or_default()
    );
    if !Path::new(&draft_path).exists() {
        eprintln!("skipping: {draft_path} absent (gated drafter bench)");
        return;
    }
    let base = ref_prompt();
    let bytes = std::fs::read(&*GGUF_PATH).expect("read gguf");
    let dbytes = std::fs::read(&draft_path).expect("read draft gguf");
    let Some(target) = load_runner(&bytes) else {
        return;
    };
    let draft = load_runner(&dbytes).expect("draft runner");
    if draft.weights.vocab != target.weights.vocab {
        eprintln!("skipping: drafter/target vocab mismatch");
        return;
    }
    let mut g = RunnerGenerator::new(target, u32::MAX).with_draft_model(draft);
    let max_new = 256usize;

    // Deterministic LCG shuffle over the reference ids: keeps every token in
    // the model's real distribution but destroys the periodicity that would
    // hand the drafter free acceptance at long ctx.
    let shuffled = |len: usize| -> Vec<u32> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                base[(state >> 33) as usize % base.len()]
            })
            .collect()
    };

    for (name, prompt) in [
        (
            "short-ctx-64",
            base.iter().cycle().take(64).copied().collect::<Vec<u32>>(),
        ),
        ("long-ctx-3776", shuffled(3776)),
    ] {
        let req = GenRequest {
            prompt_tokens: prompt.clone(),
            max_new,
            sampling: Sampling::Greedy,
            stop_eos: false,
            logprobs: None,
        };
        let warm = GenRequest {
            prompt_tokens: prompt,
            max_new: 4,
            sampling: Sampling::Greedy,
            stop_eos: false,
            logprobs: None,
        };
        fn run_leg(
            g: &mut RunnerGenerator,
            req: &GenRequest,
            name: &str,
            env: &str,
        ) -> Option<(f64, Vec<u32>)> {
            let _e = AdaptiveEnv::set(env);
            let mut out = Vec::new();
            let t0 = std::time::Instant::now();
            if let Err(e) = g.generate(req, &mut |step| {
                out.push(step.token);
                true
            }) {
                eprintln!("skipping {name}: {e}");
                return None;
            }
            Some((out.len() as f64 / t0.elapsed().as_secs_f64(), out))
        }
        {
            let _e = AdaptiveEnv::set("1");
            if g.generate(&warm, &mut |_| true).is_err() {
                eprintln!("skipping {name}: prompt does not fit the model/drafter context");
                continue;
            }
        }
        // Full-length warm legs, one per arm, OFF the clock: the first
        // full-length run after a shape change pays JIT/graph work that a
        // 4-token warm request does not reach (measured ~4-20% on the first
        // timed leg regardless of arm).
        if run_leg(&mut g, &req, name, "1").is_none() || run_leg(&mut g, &req, name, "0").is_none()
        {
            continue;
        }
        // ABBA: on, off, off, on.
        let Some((a1, sa)) = run_leg(&mut g, &req, name, "1") else {
            continue;
        };
        let Some((b1, sb)) = run_leg(&mut g, &req, name, "0") else {
            continue;
        };
        let Some((b2, _)) = run_leg(&mut g, &req, name, "0") else {
            continue;
        };
        let Some((a2, _)) = run_leg(&mut g, &req, name, "1") else {
            continue;
        };
        assert_eq!(sa, sb, "{name}: adaptive stream != non-adaptive (lossless)");
        let (on, off) = ((a1 + a2) / 2.0, (b1 + b2) / 2.0);
        println!(
            "adaptive-bench {name}: on {a1:.1}/{a2:.1} tok/s (mean {on:.1}) | \
             off {b1:.1}/{b2:.1} tok/s (mean {off:.1}) | on/off {:.3}x",
            on / off
        );
    }
}
