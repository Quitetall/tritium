//! Measured MTP draft-acceptance rate on a real Qwen3.6 SALT V2 bundle.
//!
//! Speculative decoding's whole economics hang on one number: how often the
//! drafter's token is the one the target would have emitted anyway. Nothing
//! about the loop is worth building until that number is known for this
//! architecture and this artifact, so this test measures it directly rather
//! than assuming a published figure transfers.
//!
//! Gated on `TRITIUM_QWEN36_BUNDLE` pointing at a bundle directory, because it
//! needs a ~19 GB artifact and a CUDA device. Absent either, it skips loudly.
#![cfg(all(feature = "cuda", feature = "tokenizer"))]

use std::path::PathBuf;

use tritium_nn::{HfJsonTokenizer, Qwen35SaltV2LanguageMtpModel, Tokenizer, sample_greedy};

fn bundle_path() -> Option<PathBuf> {
    match std::env::var("TRITIUM_QWEN36_BUNDLE") {
        Ok(value) if !value.is_empty() => Some(PathBuf::from(value)),
        _ => None,
    }
}

#[test]
#[ignore = "needs TRITIUM_QWEN36_BUNDLE and a CUDA device"]
fn mtp_draft_acceptance_rate_on_the_real_bundle() {
    let Some(bundle) = bundle_path() else {
        eprintln!("skipping: set TRITIUM_QWEN36_BUNDLE to a bundle directory");
        return;
    };
    let backend = match tritium_cuda::CudaBackend::new(0) {
        Ok(backend) => Box::new(backend),
        Err(error) => {
            eprintln!("skipping MTP acceptance measurement: no CUDA device ({error})");
            return;
        }
    };
    let profile =
        std::env::var("TRITIUM_QWEN36_PROFILE").unwrap_or_else(|_| "compact-v1".to_owned());
    let model =
        Qwen35SaltV2LanguageMtpModel::load_bundle_profile(&bundle, &profile, backend).unwrap();
    let (model, tokenizer_json, tokenizer_config_json) = model.into_serving_assets();
    let tokenizer = HfJsonTokenizer::from_bytes(&tokenizer_json, &tokenizer_config_json).unwrap();

    let prompt = std::env::var("TRITIUM_MTP_PROMPT").unwrap_or_else(|_| {
        "The capital of France is Paris. The capital of Japan is Tokyo. \
         The capital of Italy is"
            .to_owned()
    });
    let prompt_tokens = tokenizer.encode(&prompt).unwrap();
    let steps: usize = std::env::var("TRITIUM_MTP_STEPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(32);

    let runner = model.runner();
    // The drafter carries no parity receipt and backs no claim; it only decides
    // how often the target's verify pass gets a free token.
    let drafter = model.mtp().draft_only_runner();

    let capacity = prompt_tokens.len() + steps + 2;
    let mut cache = runner.new_cache(capacity).unwrap();
    let mut mtp_cache = drafter.new_cache(capacity).unwrap();

    let mut output = runner.forward(&prompt_tokens, &mut cache).unwrap();
    let mut next = sample_greedy(output.last_logits()).unwrap();

    let mut accepted = 0usize;
    let mut measured = 0usize;
    for _ in 0..steps {
        // MTP sees the target's hidden states plus the token just sampled, and
        // predicts the one after it.
        let drafted = match drafter.forward(runner, &output, next, &mut mtp_cache) {
            Ok(mtp_output) => sample_greedy(mtp_output.last_logits()).unwrap(),
            Err(error) => {
                panic!("MTP draft failed at step {measured}: {error}");
            }
        };
        // The target's own answer for that same position.
        output = runner.forward(&[next], &mut cache).unwrap();
        let actual = sample_greedy(output.last_logits()).unwrap();
        if drafted == actual {
            accepted += 1;
        }
        measured += 1;
        next = actual;
    }

    let rate = accepted as f64 / measured as f64;
    eprintln!(
        "MTP draft acceptance: {accepted}/{measured} = {:.1}%\n\
         speculative decode speedup at this rate needs a verify pass cheaper \
         than sequential decode by more than 1/(1 + rate)",
        rate * 100.0
    );
    assert!(measured > 0, "no steps were measured");
}
