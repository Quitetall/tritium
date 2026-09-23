//! **Does Tritium compute the same function as PrismML's own runtime for Ternary Bonsai 2 27B?**
//!
//! Bonsai 2 is a Hadamard-folded, group-128 ternary Qwen3.8-27B shipped as a llama.cpp `qwen35`
//! GGUF. Loading it means undoing every transform llama.cpp's converter applied (norm offsets,
//! `A_log`, the DeltaNet value-head tiling) and applying PrismML's signed block-Hadamard basis on
//! the activation side. Each of those can be wrong in a way that still loads and runs, so the only
//! evidence that counts is agreement with the reference implementation on the same tokens.
//!
//! The oracle is PrismML's llama.cpp fork (`llama-server /completion` with `n_probs`), recorded
//! to JSON. This loads the same GGUF on Tritium's host path and compares greedy continuations and
//! the top-token log-probabilities at the first steps.
//!
//! The two runtimes quantize activations differently (Tritium: int8 per token; llama.cpp: q8_1
//! per 32-block) and sum in different orders, so logits agree closely rather than bit for bit.
//!
//! ```text
//! TRITIUM_BONSAI_GGUF=/mnt/4tb/models/bonsai2-27b/Ternary-Bonsai-2-27B-PQ2_0.gguf \
//! TRITIUM_BONSAI_CONFIG=/mnt/4tb/models/bonsai2-27b/base-config/config.json \
//! TRITIUM_BONSAI_ORACLE=scratchpad/bonsai-oracle.json TRITIUM_BONSAI_PROMPT=760,6511,314,9338,369 \
//!   cargo test -p tritium-nn --release --test bonsai2_parity -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use tritium_nn::Qwen35GgufLanguageModel;

fn log_softmax_at(logits: &[f32], index: usize) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().map(|l| f64::from(l - max).exp()).sum();
    f64::from(logits[index] - max) - sum.ln()
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = i;
        }
    }
    best as u32
}

#[test]
#[ignore = "needs the Ternary Bonsai 2 27B GGUF (7.2 GB) and a recorded llama.cpp oracle"]
fn bonsai2_matches_the_llama_cpp_reference() {
    let var = |name: &str| std::env::var(name).ok();
    let (Some(gguf), Some(config), Some(oracle), Some(prompt)) = (
        var("TRITIUM_BONSAI_GGUF"),
        var("TRITIUM_BONSAI_CONFIG"),
        var("TRITIUM_BONSAI_ORACLE"),
        var("TRITIUM_BONSAI_PROMPT"),
    ) else {
        eprintln!("skipping: set TRITIUM_BONSAI_GGUF, _CONFIG, _ORACLE and _PROMPT");
        return;
    };
    let prompt: Vec<u32> = prompt
        .split(',')
        .map(|t| t.trim().parse().expect("prompt token id"))
        .collect();
    let oracle: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&oracle).expect("read oracle")).expect("oracle json");
    let want_tokens: Vec<u32> = oracle["tokens"]
        .as_array()
        .expect("oracle tokens")
        .iter()
        .map(|t| t.as_u64().expect("token id") as u32)
        .collect();
    let want_probs = oracle["completion_probabilities"]
        .as_array()
        .expect("oracle probabilities");

    let started = Instant::now();
    let model = Qwen35GgufLanguageModel::load(
        &PathBuf::from(gguf),
        &std::fs::read_to_string(config).expect("read config.json"),
        Box::new(tritium_cpu::CpuBackend::new()),
    )
    .expect("load Bonsai 2 GGUF");
    println!(
        "loaded in {:.1}s: {:?}",
        started.elapsed().as_secs_f64(),
        model.receipt()
    );
    let receipt = model.receipt();
    assert_eq!(
        receipt.hadamard_block,
        Some(1024),
        "Bonsai 2 declares a 1024-wide Hadamard"
    );
    assert!(
        receipt.folded_embedding,
        "the token table is folded and must be un-rotated"
    );
    assert_eq!(
        receipt.ternary_projections, 402,
        "every PQ2_0 tensor is consumed"
    );

    let runner = model.runner();
    let mut cache = runner
        .new_cache(prompt.len() + want_tokens.len() + 1)
        .expect("cache");
    let mut output = runner.forward(&prompt, &mut cache).expect("prefill");
    let mut got_tokens = Vec::new();
    let mut worst_logprob_gap = 0.0f64;
    for step in 0..want_tokens.len() {
        let logits = output.last_logits();
        let token = argmax(logits);
        if let Some(entries) = want_probs
            .get(step)
            .and_then(|p| p["top_logprobs"].as_array())
        {
            print!("step {step:2}: ours {token:6}  top-5 oracle/ours logprob:");
            for entry in entries {
                let id = entry["id"].as_u64().expect("id") as usize;
                let want = entry["logprob"].as_f64().expect("logprob");
                let got = log_softmax_at(logits, id);
                worst_logprob_gap = worst_logprob_gap.max((got - want).abs());
                print!("  {id}:{want:.3}/{got:.3}");
            }
            println!();
        }
        got_tokens.push(token);
        if step + 1 < want_tokens.len() {
            output = runner.forward(&[token], &mut cache).expect("decode step");
        }
    }
    let agree = got_tokens
        .iter()
        .zip(&want_tokens)
        .take_while(|(a, b)| a == b)
        .count();
    println!("oracle: {want_tokens:?}\nours:   {got_tokens:?}");
    println!(
        "greedy prefix agreement {agree}/{}; worst top-5 logprob gap {worst_logprob_gap:.3} nats; {:.1}s total",
        want_tokens.len(),
        started.elapsed().as_secs_f64()
    );
    assert_eq!(
        got_tokens[0], want_tokens[0],
        "the first greedy token differs: the loaded function is not the reference's"
    );
    assert!(
        agree >= want_tokens.len() / 2,
        "greedy continuations diverge after {agree} tokens"
    );
    assert!(
        worst_logprob_gap < 0.5,
        "top-5 log-probabilities differ by {worst_logprob_gap:.3} nats from the reference"
    );
}
