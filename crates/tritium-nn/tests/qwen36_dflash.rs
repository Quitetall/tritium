//! DFlash speculative decoding on the resident Qwen3.6 executor, end to end.
//!
//! For each prompt: plain greedy decode, then the draft/verify/commit loop with
//! the real drafter, over the same number of tokens. The speculative output must
//! equal greedy decode token for token (RFC 0001's greedy-identity leg for the
//! tensor-core verification path); the test reports tokens per verify cycle and
//! the decode speed of both.
//!
//! - `TRITIUM_QWEN36_BUNDLE`   bundle directory (required)
//! - `TRITIUM_QWEN36_PROFILE`  profile (default `compact-v1`)
//! - `TRITIUM_DFLASH_DIR`      drafter directory (required)
//! - `TRITIUM_DFLASH_PROMPTS`  JSON list of prompt token-id lists (default: one chat prompt)
//! - `TRITIUM_DFLASH_TOKENS`   tokens per prompt (default 256)
//! - `TRITIUM_DFLASH_BLOCK`    verify block size (default: the drafter's)
#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::time::Instant;

use tritium_nn::{Qwen35SaltV2LanguageMtpModel, attach_dflash_drafter};

const CHAT: [u32; 26] = [
    248045, 846, 198, 7734, 264, 2716, 13901, 24228, 1204, 264, 41163, 3992, 1558, 26023, 1414,
    799, 3817, 506, 264, 854, 13, 248046, 198, 248045, 74455, 198,
];

#[test]
#[ignore = "needs TRITIUM_QWEN36_BUNDLE, TRITIUM_DFLASH_DIR and a CUDA device"]
fn dflash_speculative_decode_matches_greedy() {
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
    let tokens: usize = std::env::var("TRITIUM_DFLASH_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(256);
    let block: Option<usize> = std::env::var("TRITIUM_DFLASH_BLOCK")
        .ok()
        .and_then(|value| value.parse().ok());
    let profile =
        std::env::var("TRITIUM_QWEN36_PROFILE").unwrap_or_else(|_| "compact-v1".to_owned());

    let backend = tritium_cuda::CudaBackend::new(0).expect("cuda device");
    let model = Qwen35SaltV2LanguageMtpModel::load_bundle_profile(
        &PathBuf::from(bundle),
        &profile,
        Box::new(backend),
    )
    .expect("load bundle");
    let longest = prompts.iter().map(Vec::len).max().unwrap_or(0);
    let mut executor = model
        .runner()
        .cuda_resident(longest + tokens + 2 * tritium_cuda::QWEN35_VERIFY_MAX)
        .expect("build executor")
        .expect("this bundle should be eligible for the resident executor");
    let loaded = Instant::now();
    attach_dflash_drafter(&mut executor, &PathBuf::from(drafter), block).expect("attach drafter");
    eprintln!("drafter attached in {:.1}s", loaded.elapsed().as_secs_f64());

    let (mut plain_ms, mut spec_ms, mut all_cycles, mut all_tokens) = (0.0, 0.0, 0usize, 0usize);
    let mut failures = Vec::new();
    for (index, prompt) in prompts.iter().enumerate() {
        executor.reset().unwrap();
        let mut reference = vec![executor.prefill(prompt).unwrap()];
        let started = Instant::now();
        while reference.len() < tokens {
            let next = executor.step(*reference.last().unwrap()).unwrap();
            reference.push(next);
        }
        let plain = started.elapsed().as_secs_f64() * 1000.0;

        executor.reset().unwrap();
        let mut produced = vec![executor.prefill(prompt).unwrap()];
        let (mut cycles, mut draft_ms, mut verify_ms) = (0usize, 0.0f64, 0.0f64);
        let started = Instant::now();
        while produced.len() < tokens {
            let anchor = *produced.last().unwrap();
            let clock = Instant::now();
            let drafts = executor.draft(anchor).unwrap();
            draft_ms += clock.elapsed().as_secs_f64() * 1000.0;
            let mut rows = vec![anchor];
            rows.extend_from_slice(&drafts);
            let clock = Instant::now();
            let greedy = executor.verify(&rows).unwrap();
            let mut accepted = 0;
            while accepted + 1 < rows.len() && rows[accepted + 1] == greedy[accepted] {
                accepted += 1;
            }
            executor.commit(accepted + 1).unwrap();
            verify_ms += clock.elapsed().as_secs_f64() * 1000.0;
            produced.extend_from_slice(&greedy[..=accepted]);
            cycles += 1;
        }
        let spec = started.elapsed().as_secs_f64() * 1000.0;
        produced.truncate(tokens);
        let generated = tokens - 1;
        let divergence = produced.iter().zip(&reference).position(|(a, b)| a != b);
        eprintln!(
            "prompt {index}: tau {:.2}  plain {:.2} ms/tok  spec {:.2} ms/tok ({:.2}x)  \
             draft {:.2} ms  verify+commit {:.2} ms per cycle  {}",
            generated as f64 / cycles as f64,
            plain / generated as f64,
            spec / generated as f64,
            plain / spec,
            draft_ms / cycles as f64,
            verify_ms / cycles as f64,
            divergence.map_or_else(|| "identical".to_owned(), |at| format!("DIVERGES at {at}"))
        );
        if let Some(at) = divergence {
            failures.push((index, at));
        }
        plain_ms += plain;
        spec_ms += spec;
        all_cycles += cycles;
        all_tokens += generated;
    }
    eprintln!(
        "overall: tau {:.3}, plain {:.2} ms/tok, spec {:.2} ms/tok, speedup {:.2}x",
        all_tokens as f64 / all_cycles as f64,
        plain_ms / all_tokens as f64,
        spec_ms / all_tokens as f64,
        plain_ms / spec_ms
    );
    assert!(
        failures.is_empty(),
        "speculative output diverged from greedy: {failures:?}"
    );
}
