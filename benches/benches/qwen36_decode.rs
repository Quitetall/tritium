//! Greedy decode latency on the Qwen3.6 SALT V2 bundle — `cuda` + bundle gated.
//!
//! The measurement harness for the SALT decode campaign. It loads the bundle once,
//! then times single-token greedy decode steps through the KV cache: the path a
//! served chat completion spends nearly all of its time in.
//!
//! **This reports; it does not gate.** Unlike `e2e`, nothing here asserts against a
//! committed number. A decode gate for this model is a gate change and rides on an
//! ADR (ADR 0044 D14); until then the harness produces evidence, not verdicts.
//!
//! ## Interleaved A/B in one process
//!
//! Single runs on the shared-desktop 4090 spread by ~20%, which hides any effect
//! under ~10%. `TRITIUM_QWEN36_AB=NAME=a,b` alternates an environment toggle
//! between rounds in ABBA order inside one process, so both arms see the same
//! loaded model, the same allocator state and the same box, and the report gives
//! each arm's median. It also records each round's generated tokens, and says
//! whether the arms agree: a fast tier that changes output is a quality event, not
//! a speed result.
//!
//! ## Environment
//!
//! - `TRITIUM_QWEN36_BUNDLE`   bundle directory (required; absent → skip)
//! - `TRITIUM_QWEN36_PROFILE`  `compact-v1` (default) or `near-lossless-v1`
//! - `TRITIUM_QWEN36_TOKENS`   decode steps per round (default 64)
//! - `TRITIUM_QWEN36_ROUNDS`   rounds, a multiple of 4 under A/B (default 4)
//! - `TRITIUM_QWEN36_WARMUP`   untimed leading decode steps per round (default 4)
//! - `TRITIUM_QWEN36_AB`       optional `NAME=a,b` toggle for interleaved A/B
//! - `TRITIUM_QWEN36_RESIDENT` `1` runs a round through the device-resident executor
//!   (fast tier) instead of the host-orchestrated forward; usable as an A/B toggle

fn main() {
    #[cfg(not(feature = "cuda"))]
    eprintln!("qwen36_decode: no body compiled (needs `--features cuda`).");
    #[cfg(feature = "cuda")]
    cuda_qwen36::run();
}

#[cfg(feature = "cuda")]
mod cuda_qwen36 {
    use std::path::PathBuf;
    use std::time::Instant;

    use tritium_nn::{Qwen35SaltV2LanguageMtpModel, sample_greedy};

    // Pull the CUDA backend's registration into the bench binary.
    use tritium_cuda as _;

    /// `<|im_start|>user\nWrite a short paragraph explaining how a transformer
    /// language model generates text one token at a time.<|im_end|>\n
    /// <|im_start|>assistant\n`, encoded with the bundle's own `tokenizer.json`.
    /// Fixed ids keep the harness free of a tokenizer dependency and make every
    /// round's input identical.
    const PROMPT: [u32; 26] = [
        248045, 846, 198, 7734, 264, 2716, 13901, 24228, 1204, 264, 41163, 3992, 1558, 26023, 1414,
        799, 3817, 506, 264, 854, 13, 248046, 198, 248045, 74455, 198,
    ];

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    }

    /// One arm of an A/B: the variable to set, and the value that selects it.
    #[derive(Clone)]
    struct Arm {
        label: String,
        assignment: Option<(String, String)>,
    }

    fn arms() -> Vec<Arm> {
        let Ok(spec) = std::env::var("TRITIUM_QWEN36_AB") else {
            return vec![Arm {
                label: "baseline".to_owned(),
                assignment: None,
            }];
        };
        let (name, values) = spec
            .split_once('=')
            .expect("TRITIUM_QWEN36_AB must look like NAME=a,b");
        let (a, b) = values
            .split_once(',')
            .expect("TRITIUM_QWEN36_AB must name exactly two values");
        [a, b]
            .into_iter()
            .map(|value| Arm {
                label: format!("{name}={value}"),
                assignment: Some((name.to_owned(), value.to_owned())),
            })
            .collect()
    }

    fn median(values: &mut [f64]) -> f64 {
        values.sort_by(f64::total_cmp);
        let mid = values.len() / 2;
        if values.len().is_multiple_of(2) {
            (values[mid - 1] + values[mid]) / 2.0
        } else {
            values[mid]
        }
    }

    struct Round {
        arm: usize,
        per_token_ms: f64,
        tokens: Vec<u32>,
    }

    pub(crate) fn run() {
        let Ok(bundle) = std::env::var("TRITIUM_QWEN36_BUNDLE") else {
            eprintln!("skipping qwen36_decode: set TRITIUM_QWEN36_BUNDLE to a bundle directory");
            return;
        };
        let bundle = PathBuf::from(bundle);
        let profile =
            std::env::var("TRITIUM_QWEN36_PROFILE").unwrap_or_else(|_| "compact-v1".to_owned());
        let steps = env_usize("TRITIUM_QWEN36_TOKENS", 64);
        let warmup = env_usize("TRITIUM_QWEN36_WARMUP", 4);
        let arms = arms();
        let rounds = env_usize("TRITIUM_QWEN36_ROUNDS", 4);
        assert!(
            arms.len() == 1 || rounds.is_multiple_of(4),
            "an interleaved A/B needs a multiple of 4 rounds for complete ABBA blocks"
        );

        let Some(init) = tritium_runtime::BACKENDS
            .iter()
            .find(|entry| entry.name == "cuda")
            .map(|entry| entry.init)
        else {
            eprintln!("skipping qwen36_decode: no cuda backend registered");
            return;
        };
        let backend = match init() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping qwen36_decode: cuda init failed ({error})");
                return;
            }
        };
        let device = backend.physical_device_id().to_owned();

        eprintln!(
            "qwen36_decode: loading {} ({profile}) on {device}",
            bundle.display()
        );
        let loaded = Instant::now();
        let model = Qwen35SaltV2LanguageMtpModel::load_bundle_profile(&bundle, &profile, backend)
            .expect("load bundle");
        eprintln!(
            "qwen36_decode: loaded in {:.1}s",
            loaded.elapsed().as_secs_f64()
        );
        let runner = model.runner();
        let capacity = PROMPT.len() + warmup + steps + 1;
        // Built once when eligible; each round decides from TRITIUM_QWEN36_RESIDENT
        // whether to use it, so it can be one arm of an in-process A/B.
        let mut resident = runner
            .cuda_resident(capacity)
            .expect("build resident executor");

        let mut results = Vec::new();
        for round in 0..rounds {
            // ABBA within each block of four, so drift across the run cancels.
            let arm = if arms.len() == 1 {
                0
            } else {
                [0, 1, 1, 0][round % 4]
            };
            if let Some((name, value)) = &arms[arm].assignment {
                // SAFETY: set between rounds, while no forward is in flight; the
                // flags are read on the calling thread at the start of each call.
                unsafe { std::env::set_var(name, value) };
            }

            let use_resident = std::env::var("TRITIUM_QWEN36_RESIDENT").as_deref() == Ok("1");
            let mut tokens = Vec::with_capacity(warmup + steps);
            let mut timed = Vec::with_capacity(steps);
            if use_resident {
                let executor = resident
                    .as_mut()
                    .expect("TRITIUM_QWEN36_RESIDENT=1 but this bundle has no resident executor");
                executor.reset().expect("reset executor");
                let mut next = executor.prefill(&PROMPT).expect("prefill");
                for step in 0..warmup + steps {
                    tokens.push(next);
                    let started = Instant::now();
                    next = executor.step(next).expect("decode step");
                    if step >= warmup {
                        timed.push(started.elapsed().as_secs_f64() * 1000.0);
                    }
                }
            } else {
                let mut cache = runner.new_cache(capacity).expect("allocate cache");
                let mut output = runner.forward(&PROMPT, &mut cache).expect("prefill");
                let mut next = sample_greedy(output.last_logits()).expect("greedy token");
                for step in 0..warmup + steps {
                    tokens.push(next);
                    let started = Instant::now();
                    output = runner.forward(&[next], &mut cache).expect("decode step");
                    next = sample_greedy(output.last_logits()).expect("greedy token");
                    if step >= warmup {
                        timed.push(started.elapsed().as_secs_f64() * 1000.0);
                    }
                }
            }
            let per_token_ms = median(&mut timed);
            eprintln!(
                "round {round:2} {:<32} {per_token_ms:8.2} ms/token  {:7.2} tok/s",
                arms[arm].label,
                1000.0 / per_token_ms
            );
            results.push(Round {
                arm,
                per_token_ms,
                tokens,
            });
        }

        println!(
            "qwen36_decode: bundle={} profile={profile} device={device} steps={steps} \
             warmup={warmup} rounds={rounds}",
            bundle.display()
        );
        for (index, arm) in arms.iter().enumerate() {
            let mut per_token: Vec<f64> = results
                .iter()
                .filter(|round| round.arm == index)
                .map(|round| round.per_token_ms)
                .collect();
            let ms = median(&mut per_token);
            println!(
                "  {:<32} median {ms:8.2} ms/token  {:7.2} tok/s  (n={})",
                arm.label,
                1000.0 / ms,
                per_token.len()
            );
        }
        let reference = &results[0].tokens;
        let identical = results.iter().all(|round| &round.tokens == reference);
        println!(
            "  generated tokens identical across all rounds: {identical}{}",
            if identical {
                String::new()
            } else {
                "  -- the arms disagree on output; treat this as a quality result".to_owned()
            }
        );
    }
}
