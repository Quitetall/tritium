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
