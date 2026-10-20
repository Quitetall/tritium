//! The device-resident Qwen3.6 executor against the host-orchestrated forward.
//!
//! The executor is fast-tier: its projections reassociate the K-sum and its decode
//! kernels use CUDA transcendentals, so it is close to the host path, not equal to
//! it. This gate says how close, on the real bundle: at every step it compares the
//! full logits row (relative to the row's peak magnitude) and requires the greedy
//! token to agree. The host path runs with `TRITIUM_SALT_V2_FAST=1` so both sides
//! use the same GEMV numerics and what remains is the executor's own difference.
//!
//! Gated on `TRITIUM_QWEN36_BUNDLE` and a CUDA device; ignored by default.
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use tritium_nn::{Qwen35SaltV2LanguageMtpModel, sample_greedy};

const PROMPT: [u32; 26] = [
    248045, 846, 198, 7734, 264, 2716, 13901, 24228, 1204, 264, 41163, 3992, 1558, 26023, 1414,
    799, 3817, 506, 264, 854, 13, 248046, 198, 248045, 74455, 198,
];

#[test]
#[ignore = "needs TRITIUM_QWEN36_BUNDLE and a CUDA device"]
fn resident_executor_tracks_the_host_forward_on_the_real_bundle() {
    let Ok(bundle) = std::env::var("TRITIUM_QWEN36_BUNDLE") else {
        eprintln!("skipping: set TRITIUM_QWEN36_BUNDLE");
        return;
    };
    let profile =
        std::env::var("TRITIUM_QWEN36_PROFILE").unwrap_or_else(|_| "compact-v1".to_owned());
    let steps: usize = std::env::var("TRITIUM_QWEN36_STEPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(32);
    // SAFETY: set before any forward runs; this test is the variable's only writer.
    unsafe { std::env::set_var("TRITIUM_SALT_V2_FAST", "1") };

    let backend = tritium_cuda::CudaBackend::new(0).expect("cuda device");
    let model = Qwen35SaltV2LanguageMtpModel::load_bundle_profile(
        &PathBuf::from(bundle),
        &profile,
        Box::new(backend),
    )
    .expect("load bundle");
    let runner = model.runner();
    let capacity = PROMPT.len() + steps + 1;
    let mut executor = runner
        .cuda_resident(capacity)
        .expect("build executor")
        .expect("this bundle should be eligible for the resident executor");

    let mut cache = runner.new_cache(capacity).expect("host cache");
    let mut worst = 0.0f32;
    let mut disagreements = 0usize;
    let mut compare = |step: usize, host: &[f32], device: &[f32]| {
        let peak = host
            .iter()
            .fold(0.0f32, |peak, value| peak.max(value.abs()));
        let error = host
            .iter()
            .zip(device)
            .fold(0.0f32, |max, (a, b)| max.max((a - b).abs()))
            / peak.max(f32::MIN_POSITIVE);
        worst = worst.max(error);
        let (host_token, device_token) = (sample_greedy(host), sample_greedy(device));
        if host_token != device_token {
            disagreements += 1;
        }
        eprintln!(
            "step {step:3}  rel err {error:.3e}  host {host_token:?}  device {device_token:?}"
        );
        host_token.expect("greedy token")
    };

    // Prompt, one token at a time on both sides so every position is compared.
    let mut next = 0u32;
    for (step, &token) in PROMPT.iter().enumerate() {
        let host = runner.forward(&[token], &mut cache).expect("host forward");
        let device = executor.step_logits(token).expect("executor forward");
        next = compare(step, host.last_logits(), &device);
    }
    // Greedy continuation, both fed the host's choice so a single divergence does
    // not cascade into every later comparison.
    for step in 0..steps {
        let host = runner.forward(&[next], &mut cache).expect("host forward");
        let device = executor.step_logits(next).expect("executor forward");
        next = compare(PROMPT.len() + step, host.last_logits(), &device);
    }
    eprintln!(
        "worst relative logit error {worst:.3e}; greedy disagreements {disagreements} of {}",
        PROMPT.len() + steps
    );
    assert!(
        worst <= 1e-3,
        "resident executor drifted from the host forward: worst relative error {worst:.3e}"
    );
    assert_eq!(
        disagreements, 0,
        "resident executor chose a different greedy token than the host forward"
    );
}

/// Prompt reuse: resuming from a snapshot must be indistinguishable from
/// prefilling the whole prompt. Snapshot after prompt A, decode unrelated tokens
/// (dirtying the recurrent state and the KV cache past A), restore, then feed B's
/// suffix; the logits must equal a fresh prefill of B bit for bit.
#[test]
#[ignore = "needs TRITIUM_QWEN36_BUNDLE and a CUDA device"]
fn resident_snapshot_resume_is_bit_identical_to_a_fresh_prefill() {
    let Ok(bundle) = std::env::var("TRITIUM_QWEN36_BUNDLE") else {
        eprintln!("skipping: set TRITIUM_QWEN36_BUNDLE");
        return;
    };
    let profile =
        std::env::var("TRITIUM_QWEN36_PROFILE").unwrap_or_else(|_| "compact-v1".to_owned());
    let backend = tritium_cuda::CudaBackend::new(0).expect("cuda device");
    let model = Qwen35SaltV2LanguageMtpModel::load_bundle_profile(
        &PathBuf::from(bundle),
        &profile,
        Box::new(backend),
    )
    .expect("load bundle");
    let mut executor = model
        .runner()
        .cuda_resident(128)
        .expect("build executor")
        .expect("this bundle should be eligible for the resident executor");
    let suffix = [7734u32, 264, 2716, 13901, 24228, 1204, 13, 198];
    let whole: Vec<u32> = PROMPT.iter().copied().chain(suffix).collect();

    // Fresh: the whole of B in one sequence.
    executor.reset().unwrap();
    let (&last, rest) = whole.split_last().unwrap();
    executor.prefill(rest).unwrap();
    let fresh = executor.step_logits(last).unwrap();

    // Resumed: A, snapshot, unrelated decoding, restore, then B's suffix.
    executor.reset().unwrap();
    executor.prefill(&PROMPT).unwrap();
    let mut snapshot = None;
    executor.save_snapshot(&mut snapshot).unwrap();
    assert_eq!(snapshot.as_ref().unwrap().position(), PROMPT.len());
    for token in [11u32, 22, 33, 44, 55, 66, 77, 88, 99, 111] {
        executor.step(token).unwrap();
    }
    executor
        .restore_snapshot(snapshot.as_ref().unwrap())
        .unwrap();
    assert_eq!(executor.position(), PROMPT.len());
    let (&suffix_last, suffix_rest) = suffix.split_last().unwrap();
    executor.prefill(suffix_rest).unwrap();
    let resumed = executor.step_logits(suffix_last).unwrap();

    let differing = fresh
        .iter()
        .zip(&resumed)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(differing, 0, "{differing} logits differ after resume");
}

/// Batched prefill (`TRITIUM_QWEN35_BATCHED_PREFILL`, default on) must compute
/// exactly what feeding the prompt one decode step at a time does: for prompts
/// that end inside, on and past chunk boundaries, from a reset and from a
/// restored snapshot, the logits after the prompt and over the next decode
/// steps (which read every DeltaNet state and KV row the prompt wrote) are
/// equal bit for bit.
#[test]
#[ignore = "needs TRITIUM_QWEN36_BUNDLE and a CUDA device"]
fn batched_prefill_is_bit_identical_to_token_by_token() {
    let Ok(bundle) = std::env::var("TRITIUM_QWEN36_BUNDLE") else {
        eprintln!("skipping: set TRITIUM_QWEN36_BUNDLE");
        return;
    };
    let profile =
        std::env::var("TRITIUM_QWEN36_PROFILE").unwrap_or_else(|_| "compact-v1".to_owned());
    let backend = tritium_cuda::CudaBackend::new(0).expect("cuda device");
    let model = Qwen35SaltV2LanguageMtpModel::load_bundle_profile(
        &PathBuf::from(bundle),
        &profile,
        Box::new(backend),
    )
    .expect("load bundle");
    let mut executor = model
        .runner()
        .cuda_resident(PROMPT.len() + 256)
        .expect("build executor")
        .expect("this bundle should be eligible for the resident executor");
    // Deterministic, varied ids well inside the vocabulary.
    let mut seed = 0x2545_f491u32;
    let tokens: Vec<u32> = (0..210)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) % 150_000
        })
        .collect();
    let decode = [11u32, 2716, 13];

    // Feed `prompt` (all but its last token through `prefill`), then the decode
    // tokens, returning every logits row as bits.
    let mut run = |batched: bool, prefix: bool, prompt: &[u32]| {
        // SAFETY: run filtered (`--test-threads=1` or this test alone); the
        // executor reads the variable on this thread only.
        unsafe {
            std::env::set_var(
                "TRITIUM_QWEN35_BATCHED_PREFILL",
                if batched { "1" } else { "0" },
            );
            // The bit-identity claim is the multi-row GEMV's; the tensor-core
            // projections are gated separately below.
            std::env::set_var("TRITIUM_QWEN35_TC_PREFILL", "0");
        }
        executor.reset().unwrap();
        if prefix {
            // A snapshot after a shared prefix, dirtied, then restored.
            executor.prefill(&PROMPT).unwrap();
            let mut snapshot = None;
            executor.save_snapshot(&mut snapshot).unwrap();
            for token in [5u32, 6, 7, 8, 9] {
                executor.step(token).unwrap();
            }
            executor
                .restore_snapshot(snapshot.as_ref().unwrap())
                .unwrap();
        }
        let (&last, rest) = prompt.split_last().unwrap();
        if !rest.is_empty() {
            executor.prefill(rest).unwrap();
        }
        let mut rows = vec![executor.step_logits(last).unwrap()];
        for &token in &decode {
            rows.push(executor.step_logits(token).unwrap());
        }
        rows.into_iter()
            .map(|row| row.into_iter().map(f32::to_bits).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    };

    for prefix in [false, true] {
        for len in [2usize, 3, 4, 8, 17, 65, 66, 67, 130, 201] {
            let prompt = &tokens[..len];
            let reference = run(false, prefix, prompt);
            let batched = run(true, prefix, prompt);
            for (step, (a, b)) in reference.iter().zip(&batched).enumerate() {
                let differing = a.iter().zip(b).filter(|(x, y)| x != y).count();
                assert_eq!(
                    differing, 0,
                    "prompt {len} (prefix {prefix}): {differing} logits differ at step {step}"
                );
            }
            eprintln!(
                "prompt {len:3} prefix {prefix}: bit-identical over {} rows",
                reference.len()
            );
        }
    }
}

/// Tensor-core prefill (`TRITIUM_QWEN35_TC_PREFILL`, default on) against the
/// multi-row GEMV it replaces, on RFC 0001's fast-tier legs:
///
/// - in situ: every tensor-core projection inside real prefill forwards, against
///   the GEMV on the same input, max peak-relative error <= 1e-4;
/// - final logits after the prompt, peak-relative error <= 2e-3 (reported);
/// - greedy decode after the prompt identical token for token over 256 tokens;
/// - teacher-forced perplexity of text continuations after a prefilled context,
///   ratio <= 1.001.
///
/// Decode itself is the same GEMV path in both arms, so every difference here is
/// the prefill's. Needs `TRITIUM_QWEN36_CORPUS` (a `{"eval_ids": [...]}` JSON of
/// Qwen-tokenized text, e.g. `~/.cache/tritium-corpora/wikitext2_400k_32k_qwen35.json`)
/// for the perplexity leg.
#[test]
#[ignore = "needs TRITIUM_QWEN36_BUNDLE, TRITIUM_QWEN36_CORPUS and a CUDA device"]
fn tensor_core_prefill_is_within_the_fast_tier_gate() {
    let Ok(bundle) = std::env::var("TRITIUM_QWEN36_BUNDLE") else {
        eprintln!("skipping: set TRITIUM_QWEN36_BUNDLE");
        return;
    };
    let Ok(corpus) = std::env::var("TRITIUM_QWEN36_CORPUS") else {
        eprintln!("skipping: set TRITIUM_QWEN36_CORPUS");
        return;
    };
    let corpus: serde_json::Value =
        serde_json::from_slice(&std::fs::read(corpus).expect("read corpus")).expect("corpus json");
    let text: Vec<u32> = corpus["eval_ids"]
        .as_array()
        .expect("eval_ids")
        .iter()
        .map(|id| u32::try_from(id.as_u64().expect("token id")).expect("u32 id"))
        .collect();
    let profile =
        std::env::var("TRITIUM_QWEN36_PROFILE").unwrap_or_else(|_| "compact-v1".to_owned());
    let backend = tritium_cuda::CudaBackend::new(0).expect("cuda device");
    let model = Qwen35SaltV2LanguageMtpModel::load_bundle_profile(
        &PathBuf::from(bundle),
        &profile,
        Box::new(backend),
    )
    .expect("load bundle");
    const GREEDY: usize = 256;
    const CONTEXT: usize = 192;
    const CONTINUATION: usize = 64;
    let mut executor = model
        .runner()
        .cuda_resident(CONTEXT + GREEDY + CONTINUATION + 8)
        .expect("build executor")
        .expect("this bundle should be eligible for the resident executor");
    let set_tc = |on: bool| {
        // SAFETY: run filtered; the executor reads the variable on this thread only.
        unsafe { std::env::set_var("TRITIUM_QWEN35_TC_PREFILL", if on { "1" } else { "0" }) };
    };
    let peak_relative = |a: &[f32], b: &[f32]| {
        let peak = a.iter().fold(0.0f32, |peak, value| peak.max(value.abs()));
        a.iter()
            .zip(b)
            .fold(0.0f32, |max, (x, y)| max.max((x - y).abs()))
            / peak.max(f32::MIN_POSITIVE)
    };

    let chat: Vec<u32> = PROMPT.iter().copied().cycle().take(200).collect();
    let prompts: [&[u32]; 3] = [&chat, &text[..201], &text[4096..4096 + 130]];

    // In situ.
    set_tc(true);
    executor.set_tensor_core_probe(Some(0.0));
    for prompt in prompts {
        executor.reset().unwrap();
        executor.prefill(prompt).unwrap();
    }
    let in_situ = executor
        .set_tensor_core_probe(None)
        .expect("probe was running");
    eprintln!("in-situ projection drift: worst peak-relative error {in_situ:.3e}");

    // Final logits and greedy identity.
    let mut worst_logits = 0.0f32;
    let mut greedy_mismatches = Vec::new();
    for (index, prompt) in prompts.iter().enumerate() {
        let mut arm = |tc: bool| {
            set_tc(tc);
            executor.reset().unwrap();
            let (&last, rest) = prompt.split_last().unwrap();
            executor.prefill(rest).unwrap();
            let logits = executor.step_logits(last).unwrap();
            let mut next = sample_greedy(&logits).expect("greedy token");
            let mut tokens = vec![next];
            for _ in 1..GREEDY {
                next = executor.step(next).unwrap();
                tokens.push(next);
            }
            (logits, tokens)
        };
        let (exact_logits, exact_tokens) = arm(false);
        let (tc_logits, tc_tokens) = arm(true);
        let error = peak_relative(&exact_logits, &tc_logits);
        worst_logits = worst_logits.max(error);
        let first_divergence = exact_tokens
            .iter()
            .zip(&tc_tokens)
            .position(|(a, b)| a != b);
        eprintln!(
            "prompt {index} ({} tokens): logits {error:.3e}, greedy {}",
            prompt.len(),
            first_divergence.map_or_else(
                || format!("identical over {GREEDY}"),
                |at| format!("diverges at token {at}")
            )
        );
        if let Some(at) = first_divergence {
            greedy_mismatches.push((index, at));
        }
    }

    // Teacher-forced perplexity of continuations after a prefilled context.
    let mut nll = [0.0f64; 2];
    let mut scored = 0usize;
    for window in 0..8 {
        let start = 8192 + window * (CONTEXT + CONTINUATION);
        let context = &text[start..start + CONTEXT];
        let continuation = &text[start + CONTEXT..start + CONTEXT + CONTINUATION];
        for (arm, tc) in [false, true].into_iter().enumerate() {
            set_tc(tc);
            executor.reset().unwrap();
            let (&last, rest) = context.split_last().unwrap();
            executor.prefill(rest).unwrap();
            let mut logits = executor.step_logits(last).unwrap();
            for &target in continuation {
                let peak = logits.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
                let sum: f64 = logits.iter().map(|&v| f64::from(v - peak).exp()).sum();
                nll[arm] += sum.ln() - f64::from(logits[target as usize] - peak);
                logits = executor.step_logits(target).unwrap();
            }
        }
        scored += CONTINUATION;
    }
    let ppl = nll.map(|total| (total / scored as f64).exp());
    let ratio = ppl[1] / ppl[0];
    eprintln!(
        "perplexity over {scored} positions: exact {:.6}, tensor-core {:.6}, ratio {ratio:.6}",
        ppl[0], ppl[1]
    );
    eprintln!("final-logit drift: worst peak-relative error {worst_logits:.3e}");

    assert!(
        in_situ <= 1e-4,
        "in-situ projection drift {in_situ:.3e} exceeds 1e-4"
    );
    assert!(
        worst_logits <= 2e-3,
        "final-logit drift {worst_logits:.3e} exceeds 2e-3"
    );
    assert!(
        greedy_mismatches.is_empty(),
        "greedy decode diverged after tensor-core prefill: {greedy_mismatches:?}"
    );
    assert!(ratio <= 1.001, "perplexity ratio {ratio:.6} exceeds 1.001");
}
