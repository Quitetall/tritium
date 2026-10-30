//! Device-chained greedy decode gates (model + GPU gated, `cuda` feature).
//!
//! The chained path (`TRITIUM_GREEDY_CHAIN`, default on) replays the captured
//! decode graph several times per host round trip with a device argmax between
//! replays. It must be indistinguishable from the per-token plain loop: the
//! same tokens and the same finish reasons for every request shape, including
//! lengths that are not a multiple of the chain size, EOS inside a chain,
//! requests that do not stop on EOS, and a client that cancels mid-chain
//! (after which the next request must be unaffected). The test also times both
//! paths interleaved and prints the decode rates.

#![cfg(feature = "cuda")]

use std::path::Path;

use tritium_serve::{FinishReason, GenRequest, Generator, RunnerGenerator, Sampling};

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

/// Tokens and finish reasons of one request, stopping early when `cancel_at`
/// tokens have been seen (the client hanging up).
fn run(
    generator: &mut dyn Generator,
    req: &GenRequest,
    chained: bool,
    cancel_at: Option<usize>,
) -> (Vec<u32>, Vec<Option<FinishReason>>) {
    // SAFETY: this file runs one test function, so nothing else reads the
    // variable concurrently; it is read once per request on this thread.
    unsafe { std::env::set_var("TRITIUM_GREEDY_CHAIN", if chained { "1" } else { "0" }) };
    let mut tokens = Vec::new();
    let mut reasons = Vec::new();
    generator
        .generate(req, &mut |step| {
            tokens.push(step.token);
            reasons.push(step.finish_reason);
            cancel_at.is_none_or(|limit| tokens.len() < limit)
        })
        .expect("generate");
    (tokens, reasons)
}

fn request(prompt: &[u32], max_new: usize, stop_eos: bool) -> GenRequest {
    GenRequest {
        prompt_tokens: prompt.to_vec(),
        max_new,
        sampling: Sampling::Greedy,
        stop_eos,
        logprobs: None,
    }
}

#[test]
fn cuda_greedy_chain_matches_plain_greedy() {
    if !Path::new(&*GGUF_PATH).exists() {
        eprintln!("skipping: {} absent (gated real-model test)", *GGUF_PATH);
        return;
    }
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(REF_PATH).expect("read reference"))
            .expect("parse reference");
    // `eval_ids` is the prompt followed by its greedy continuation: slices of
    // it make prompts of several lengths.
    let ids: Vec<u32> = reference["eval_ids"]
        .as_array()
        .expect("eval_ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect();
    let bytes = std::fs::read(&*GGUF_PATH).expect("read gguf");
    let Some(runner) = load_runner(&bytes) else {
        return;
    };
    // EOS is token 13 ("."), which recurs through the continuation, so requests
    // that stop on EOS end inside chains at several depths.
    let eos = 13;
    let mut generator = RunnerGenerator::new(runner, eos);

    let prompts = [ids[..6].to_vec(), ids[..21].to_vec(), ids[..64].to_vec()];
    for prompt in &prompts {
        for max_new in [1usize, 2, 3, 7, 16, 37, 100] {
            for stop_eos in [true, false] {
                let req = request(prompt, max_new, stop_eos);
                let plain = run(&mut generator, &req, false, None);
                let chained = run(&mut generator, &req, true, None);
                assert_eq!(
                    plain,
                    chained,
                    "prompt {} tokens, max_new {max_new}, stop_eos {stop_eos}",
                    prompt.len()
                );
            }
        }
    }

    // A client that cancels mid-chain sees the same prefix, and the next
    // request is unaffected by the chain's abandoned tail.
    let req = request(&ids[..6], 64, false);
    let full = run(&mut generator, &req, false, None);
    for cancel_at in [1usize, 2, 5, 9, 20] {
        let cancelled = run(&mut generator, &req, true, Some(cancel_at));
        assert_eq!(
            cancelled.0[..],
            full.0[..cancel_at.min(full.0.len())],
            "cancel at {cancel_at}"
        );
        let after = run(&mut generator, &req, true, None);
        assert_eq!(after, full, "request after a cancel at {cancel_at}");
    }

    // Interleaved timing, ABBA, 256 tokens without EOS stops.
    let req = request(&ids[..6], 256, false);
    let mut plain_s = Vec::new();
    let mut chained_s = Vec::new();
    for round in 0..8 {
        let chained = [false, true, true, false][round % 4];
        let started = std::time::Instant::now();
        let (tokens, _) = run(&mut generator, &req, chained, None);
        let rate = tokens.len() as f64 / started.elapsed().as_secs_f64();
        if chained {
            chained_s.push(rate);
        } else {
            plain_s.push(rate);
        }
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
    };
    eprintln!(
        "greedy decode, 256 tokens incl. prefill: plain {:.1} tok/s, chained {:.1} tok/s (medians of 4)",
        median(&mut plain_s),
        median(&mut chained_s)
    );
}
