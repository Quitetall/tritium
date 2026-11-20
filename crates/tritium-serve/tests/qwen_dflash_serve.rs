//! DFlash speculative decoding through the serving generator.
//!
//! On one `QwenGenerator` (fast tier, drafter attached): a greedy request without
//! logprobs takes the speculative path, the same request with logprobs takes the
//! plain logits path; their tokens must match on every prompt, and prompt reuse
//! (follow-up turns resuming from a snapshot) must not disturb the drafter.
//! Reports both speeds.
//!
//! Gated on `TRITIUM_QWEN36_BUNDLE`, `TRITIUM_DFLASH_DIR` and a CUDA device;
//! `TRITIUM_DFLASH_PROMPTS` (JSON list of token-id lists) and
//! `TRITIUM_DFLASH_TOKENS` (default 192) are optional.
#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::time::Instant;

use tritium_serve::{GenRequest, Generator, QwenGenerator, QwenNumerics, Sampling, Step};

const CHAT: [u32; 26] = [
    248045, 846, 198, 7734, 264, 2716, 13901, 24228, 1204, 264, 41163, 3992, 1558, 26023, 1414,
    799, 3817, 506, 264, 854, 13, 248046, 198, 248045, 74455, 198,
];

fn run(
    generator: &mut QwenGenerator,
    prompt: &[u32],
    max_new: usize,
    logprobs: bool,
) -> (Vec<u32>, f64) {
    let request = GenRequest {
        prompt_tokens: prompt.to_vec(),
        max_new,
        sampling: Sampling::Greedy,
        stop_eos: true,
        logprobs: logprobs.then_some(0),
    };
    let mut tokens = Vec::new();
    let started = Instant::now();
    generator
        .generate(&request, &mut |step: Step| {
            tokens.push(step.token);
            true
        })
        .expect("generate");
    (tokens, started.elapsed().as_secs_f64() * 1000.0)
}

#[test]
#[ignore = "needs TRITIUM_QWEN36_BUNDLE, TRITIUM_DFLASH_DIR and a CUDA device"]
fn speculative_serving_matches_plain_greedy() {
    let (Ok(bundle), Ok(drafter)) = (
        std::env::var("TRITIUM_QWEN36_BUNDLE"),
        std::env::var("TRITIUM_DFLASH_DIR"),
    ) else {
        eprintln!("skipping: set TRITIUM_QWEN36_BUNDLE and TRITIUM_DFLASH_DIR");
        return;
    };
    let prompts: Vec<Vec<u32>> = std::env::var("TRITIUM_DFLASH_PROMPTS").map_or_else(
        |_| vec![CHAT.to_vec()],
        |path| serde_json::from_slice(&std::fs::read(path).expect("read prompts")).expect("json"),
    );
    let max_new: usize = std::env::var("TRITIUM_DFLASH_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(192);
    let backend = tritium_cuda::CudaBackend::new(0).expect("cuda device");
    let model = tritium_nn::Qwen35SaltV2LanguageMtpModel::load_bundle_profile(
        &PathBuf::from(bundle),
        "compact-v1",
        Box::new(backend),
    )
    .expect("load bundle");
    let mut generator = QwenGenerator::with_numerics(model, 248046, QwenNumerics::Fast, Some(2048))
        .expect("fast tier")
        .with_dflash(&PathBuf::from(drafter), None)
        .expect("attach drafter");

    let (mut spec_ms, mut plain_ms, mut tokens_total) = (0.0, 0.0, 0usize);
    for (index, prompt) in prompts.iter().enumerate() {
        let (plain, plain_time) = run(&mut generator, prompt, max_new, true);
        let (spec, spec_time) = run(&mut generator, prompt, max_new, false);
        // A follow-up turn: the previous exchange plus a short user message,
        // which resumes from the prompt-end snapshot.
        let mut follow_up = prompt.clone();
        follow_up.extend_from_slice(&spec);
        follow_up.extend_from_slice(&[
            248046, 198, 248045, 846, 198, 3838, 13, 248046, 198, 248045, 74455, 198,
        ]);
        let (follow_plain, _) = run(&mut generator, &follow_up, 64, true);
        let (follow_spec, _) = run(&mut generator, &follow_up, 64, false);
        eprintln!(
            "prompt {index}: {} tokens, plain {:.2} ms/tok, speculative {:.2} ms/tok ({:.2}x); \
             same tokens: {}; follow-up same: {}",
            spec.len(),
            plain_time / plain.len() as f64,
            spec_time / spec.len() as f64,
            plain_time / spec_time * spec.len() as f64 / plain.len() as f64,
            plain == spec,
            follow_plain == follow_spec
        );
        assert_eq!(
            plain, spec,
            "prompt {index}: speculative tokens differ from plain greedy"
        );
        assert_eq!(
            follow_plain, follow_spec,
            "prompt {index}: follow-up differs"
        );
        spec_ms += spec_time;
        plain_ms += plain_time;
        tokens_total += spec.len();
    }
    eprintln!(
        "overall: plain {:.2} ms/tok, speculative {:.2} ms/tok ({:.2}x)",
        plain_ms / tokens_total as f64,
        spec_ms / tokens_total as f64,
        plain_ms / spec_ms
    );
}
