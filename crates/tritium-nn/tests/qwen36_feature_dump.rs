//! Record greedy decode runs of the resident Qwen3.6 executor with the hidden
//! states a DFlash drafter conditions on, for offline drafter-acceptance studies.
//!
//! Not a gate: it writes evidence files and asserts nothing about them.
//!
//! - `TRITIUM_QWEN36_BUNDLE`        bundle directory (required)
//! - `TRITIUM_QWEN36_PROFILE`       profile (default `compact-v1`)
//! - `TRITIUM_FEATURE_PROMPTS`      JSON list of prompt token-id lists (required)
//! - `TRITIUM_FEATURE_DUMP_DIR`     output directory (required)
//! - `TRITIUM_FEATURE_TOKENS`       greedy tokens per prompt (default 256)
//! - `TRITIUM_FEATURE_LAYERS`       comma-separated decoder layers (default `1,16,31,46,61`)
//!
//! Per prompt `i` it writes `run{i}.json` (`{"tokens": [...], "positions": n}`:
//! the prompt then the greedy continuation) and `run{i}.f32`, little-endian
//! `[positions, layers, hidden]` floats: the residual stream after each layer at
//! every position the executor consumed.
#![cfg(feature = "cuda")]

use std::io::Write;
use std::path::PathBuf;

use tritium_nn::Qwen35SaltV2LanguageMtpModel;

#[test]
#[ignore = "needs TRITIUM_QWEN36_BUNDLE, prompts, an output directory and a CUDA device"]
fn dump_greedy_runs_with_drafter_features() {
    let (Ok(bundle), Ok(prompts), Ok(dir)) = (
        std::env::var("TRITIUM_QWEN36_BUNDLE"),
        std::env::var("TRITIUM_FEATURE_PROMPTS"),
        std::env::var("TRITIUM_FEATURE_DUMP_DIR"),
    ) else {
        eprintln!(
            "skipping: set TRITIUM_QWEN36_BUNDLE, TRITIUM_FEATURE_PROMPTS, TRITIUM_FEATURE_DUMP_DIR"
        );
        return;
    };
    let prompts: Vec<Vec<u32>> =
        serde_json::from_slice(&std::fs::read(prompts).expect("read prompts"))
            .expect("prompts json");
    let tokens: usize = std::env::var("TRITIUM_FEATURE_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(256);
    let layers: Vec<usize> = std::env::var("TRITIUM_FEATURE_LAYERS")
        .unwrap_or_else(|_| "1,16,31,46,61".to_owned())
        .split(',')
        .map(|layer| layer.trim().parse().expect("layer index"))
        .collect();
    let profile =
        std::env::var("TRITIUM_QWEN36_PROFILE").unwrap_or_else(|_| "compact-v1".to_owned());
    let dir = PathBuf::from(dir);
    std::fs::create_dir_all(&dir).expect("create dump dir");

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
        .cuda_resident(longest + tokens + 1)
        .expect("build executor")
        .expect("this bundle should be eligible for the resident executor");
    executor.set_feature_capture(&layers).expect("capture");

    for (index, prompt) in prompts.iter().enumerate() {
        executor.reset().expect("reset");
        let mut sequence = prompt.clone();
        let mut features = std::io::BufWriter::new(
            std::fs::File::create(dir.join(format!("run{index}.f32"))).unwrap(),
        );
        let mut positions = 0usize;
        let mut next = 0u32;
        // Prompt and continuation alike go through decode steps, one position each.
        for step in 0..prompt.len() + tokens - 1 {
            let token = if step < prompt.len() {
                prompt[step]
            } else {
                next
            };
            next = executor.step(token).expect("decode step");
            let (rows, data) = executor.captured_features().expect("features");
            assert_eq!(rows, 1);
            for value in data {
                features.write_all(&value.to_le_bytes()).unwrap();
            }
            positions += 1;
            if step + 1 >= prompt.len() {
                sequence.push(next);
            }
        }
        features.flush().unwrap();
        let record = serde_json::json!({ "tokens": sequence, "positions": positions, "prompt": prompt.len() });
        std::fs::write(dir.join(format!("run{index}.json")), record.to_string()).unwrap();
        eprintln!(
            "run {index}: prompt {} + {} tokens",
            prompt.len(),
            sequence.len() - prompt.len()
        );
    }
}
