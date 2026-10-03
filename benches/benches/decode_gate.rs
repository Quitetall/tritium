//! Reproducible RTX 4090 BitNet decode baseline for ADR 0044 P0.
//!
//! This is a measurement harness, not a quality or release-qualification claim.
//! It records the model/source/device identity and raw per-sample timings so later
//! comparisons can re-derive medians instead of trusting a single summary number.

#![cfg_attr(not(feature = "cuda"), allow(unused_crate_dependencies))]

#[cfg(feature = "cuda")]
mod gate {
    use std::collections::BTreeMap;
    use std::error::Error;
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    use serde::{Deserialize, Serialize};
    use sha2::{Digest, Sha256};
    use tritium_nn::ModelRunner;

    use tritium_cuda as _;

    const CONTEXTS: [usize; 2] = [128, 2048];
    const DECODE_STEPS: usize = 32;
    const TREE_NODES: usize = 6;
    const SAMPLES: usize = 5;
    const WARMUPS: usize = 2;

    #[derive(Deserialize)]
    struct Reference {
        token_ids: Vec<u32>,
    }

    #[derive(Serialize)]
    struct Report {
        schema: &'static str,
        run_id: String,
        source_revision: String,
        source_dirty: bool,
        started_unix_ms: u128,
        benchmark_pid: u32,
        device: String,
        gpu_activity_before: String,
        gpu_activity_at_end: String,
        gpu_processes_before: String,
        gpu_processes_at_end: String,
        model_path: String,
        model_bytes: u64,
        model_sha256: String,
        model_architecture: String,
        model_layers: u32,
        model_context_limit: u32,
        artifact_quantization: &'static str,
        execution_quantization: &'static str,
        runtime_options: BTreeMap<String, String>,
        contexts: [usize; 2],
        decode_steps: usize,
        tree_nodes_per_sequence: usize,
        warmups: usize,
        samples: usize,
        results: Vec<CaseResult>,
        claim_boundary: &'static str,
    }

    #[derive(Serialize)]
    struct CaseResult {
        workload: &'static str,
        context_tokens: usize,
        batch_size: usize,
        samples_ms: Vec<f64>,
        median_ms: f64,
        work_unit: &'static str,
        aggregate_work_units_per_second: f64,
        per_sequence_work_units_per_second: f64,
        tree_nodes_per_verify: Option<usize>,
    }

    fn model_path() -> PathBuf {
        let path = if let Some(path) = std::env::var_os("TRITIUM_BITNET_GGUF") {
            PathBuf::from(path)
        } else {
            let root = std::env::var_os("TRITIUM_MODEL_DIR")
                .map(PathBuf::from)
                .or_else(|| {
                    std::env::var_os("HOME")
                        .map(|home| PathBuf::from(home).join(".cache/tritium-models"))
                })
                .unwrap_or_else(|| PathBuf::from(".cache/tritium-models"));
            root.join("bitnet-2b4t-gguf/ggml-model-i2_s.gguf")
        };
        resolve_from_repo(path)
    }

    fn resolve_from_repo(path: PathBuf) -> PathBuf {
        if path.is_absolute() {
            path
        } else {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("bench package is a workspace member")
                .join(path)
        }
    }

    fn run_text(program: &str, args: &[&str]) -> String {
        Command::new(program)
            .args(args)
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
            .unwrap_or_else(|| "unavailable".to_owned())
    }

    fn revision() -> String {
        run_text("git", &["rev-parse", "HEAD"])
    }

    fn source_dirty() -> bool {
        !run_text("git", &["status", "--porcelain"]).is_empty()
    }

    fn runtime_options() -> BTreeMap<String, String> {
        [
            "CUDA_VISIBLE_DEVICES",
            "TRITIUM_KV",
            "TRITIUM_KV_F16",
            "TRITIUM_KERNEL_TIER",
            "TRITIUM_TREE_NB",
            "TRITIUM_LM_HEAD",
        ]
        .into_iter()
        .map(|name| {
            (
                name.to_owned(),
                std::env::var(name).unwrap_or_else(|_| "<unset>".to_owned()),
            )
        })
        .collect()
    }

    fn prompt(reference: &[u32], context: usize) -> (Vec<u32>, Vec<usize>) {
        assert!(
            !reference.is_empty(),
            "reference prompt must contain tokens"
        );
        let tokens = (0..context)
            .map(|index| reference[index % reference.len()])
            .collect();
        let positions = (0..context).collect();
        (tokens, positions)
    }

    fn argmax(logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map_or(0, |(index, _)| index as u32)
    }

    fn prefill(
        runner: &mut ModelRunner,
        reference: &[u32],
        context: usize,
    ) -> Result<u32, Box<dyn Error>> {
        runner.reset();
        let (tokens, positions) = prompt(reference, context);
        Ok(argmax(&runner.forward(&tokens, &positions)?))
    }

    fn median(values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        sorted[sorted.len() / 2]
    }

    fn result(
        workload: &'static str,
        context: usize,
        batch: usize,
        samples_ms: Vec<f64>,
        work_units: usize,
        work_unit: &'static str,
        tree_nodes: Option<usize>,
    ) -> CaseResult {
        let median_ms = median(&samples_ms);
        let aggregate_work_units_per_second = work_units as f64 * 1000.0 / median_ms;
        CaseResult {
            workload,
            context_tokens: context,
            batch_size: batch,
            samples_ms,
            median_ms,
            work_unit,
            aggregate_work_units_per_second,
            per_sequence_work_units_per_second: aggregate_work_units_per_second / batch as f64,
            tree_nodes_per_verify: tree_nodes,
        }
    }

    fn bench_decode_one(
        runner: &mut ModelRunner,
        reference: &[u32],
        context: usize,
    ) -> Result<CaseResult, Box<dyn Error>> {
        for _ in 0..WARMUPS {
            let mut token = prefill(runner, reference, context)?;
            for step in 0..4 {
                token = argmax(&runner.forward(&[token], &[context + step])?);
            }
        }
        let mut samples = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            let mut token = prefill(runner, reference, context)?;
            let start = Instant::now();
            for step in 0..DECODE_STEPS {
                token = argmax(&runner.forward(&[token], &[context + step])?);
            }
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        Ok(result(
            "decode",
            context,
            1,
            samples,
            DECODE_STEPS,
            "token",
            None,
        ))
    }

    fn seed_batch(
        runner: &mut ModelRunner,
        batch: &mut tritium_cuda::BatchKv,
        reference: &[u32],
        context: usize,
    ) -> Result<Vec<u32>, Box<dyn Error>> {
        let token = prefill(runner, reference, context)?;
        for row in 0..batch.len() {
            runner.adopt_into_batch_row(batch, row, context)?;
            batch.set_position(row, context)?;
        }
        Ok(vec![token; batch.len()])
    }

    fn bench_decode_batch(
        runner: &mut ModelRunner,
        reference: &[u32],
        context: usize,
    ) -> Result<CaseResult, Box<dyn Error>> {
        let batch_size = 8;
        let mut batch = runner.new_batch(batch_size)?;
        for _ in 0..WARMUPS {
            let mut tokens = seed_batch(runner, &mut batch, reference, context)?;
            for _ in 0..4 {
                tokens = runner
                    .decode_batch_graph(&mut batch, &tokens)?
                    .iter()
                    .map(|logits| argmax(logits))
                    .collect();
            }
        }
        let mut samples = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            let mut tokens = seed_batch(runner, &mut batch, reference, context)?;
            let start = Instant::now();
            for _ in 0..DECODE_STEPS {
                tokens = runner
                    .decode_batch_graph(&mut batch, &tokens)?
                    .iter()
                    .map(|logits| argmax(logits))
                    .collect();
            }
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        Ok(result(
            "decode",
            context,
            batch_size,
            samples,
            DECODE_STEPS * batch_size,
            "token",
            None,
        ))
    }

    fn tree(reference: &[u32], root: u32) -> (Vec<u32>, Vec<i32>) {
        let mut tokens = vec![root; TREE_NODES];
        for index in 1..TREE_NODES {
            tokens[index] = reference[index % reference.len()];
        }
        (tokens, vec![-1, 0, 0, 1, 1, 2])
    }

    fn bench_tree_one(
        runner: &mut ModelRunner,
        reference: &[u32],
        context: usize,
    ) -> Result<CaseResult, Box<dyn Error>> {
        for _ in 0..WARMUPS {
            let root = prefill(runner, reference, context)?;
            let (tokens, parents) = tree(reference, root);
            let accepted = runner.tree_verify_greedy(&tokens, &parents)?;
            assert!(
                !accepted.is_empty(),
                "tree verify must commit at least one token"
            );
        }
        let mut samples = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            let root = prefill(runner, reference, context)?;
            let (tokens, parents) = tree(reference, root);
            let start = Instant::now();
            let accepted = runner.tree_verify_greedy(&tokens, &parents)?;
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
            assert!(
                !accepted.is_empty(),
                "tree verify must commit at least one token"
            );
        }
        Ok(result(
            "tree_verify",
            context,
            1,
            samples,
            TREE_NODES,
            "draft_node",
            Some(TREE_NODES),
        ))
    }

    fn bench_tree_batch(
        runner: &mut ModelRunner,
        reference: &[u32],
        context: usize,
    ) -> Result<CaseResult, Box<dyn Error>> {
        let batch_size = 8;
        let mut batch = runner.new_batch(batch_size)?;
        let rows: Vec<usize> = (0..batch_size).collect();
        let measure = |runner: &mut ModelRunner,
                       batch: &mut tritium_cuda::BatchKv|
         -> Result<f64, Box<dyn Error>> {
            let root = seed_batch(runner, batch, reference, context)?[0];
            let trees: Vec<_> = (0..batch_size).map(|_| tree(reference, root)).collect();
            let borrowed: Vec<_> = trees
                .iter()
                .map(|(tokens, parents)| (tokens.as_slice(), parents.as_slice()))
                .collect();
            let start = Instant::now();
            let accepted = runner.tree_verify_greedy_slots(batch, &rows, &borrowed)?;
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(accepted.len(), batch_size);
            assert!(accepted.iter().all(|path| !path.is_empty()));
            Ok(elapsed)
        };
        for _ in 0..WARMUPS {
            let _ = measure(runner, &mut batch)?;
        }
        let mut samples = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            samples.push(measure(runner, &mut batch)?);
        }
        Ok(result(
            "tree_verify",
            context,
            batch_size,
            samples,
            batch_size * TREE_NODES,
            "draft_node",
            Some(TREE_NODES),
        ))
    }

    fn write_report(report: &Report) -> Result<(), Box<dyn Error>> {
        let serialized = serde_json::to_vec_pretty(report)?;
        if let Some(path) = std::env::var_os("TRITIUM_DECODE_GATE_OUT") {
            let path = resolve_from_repo(PathBuf::from(path));
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, &serialized)?;
            println!("wrote {}", path.display());
        } else {
            println!("{}", String::from_utf8(serialized)?);
        }
        Ok(())
    }

    pub(super) fn run() -> Result<(), Box<dyn Error>> {
        let started_unix_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
        let device = run_text(
            "nvidia-smi",
            &[
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ],
        );
        if device == "unavailable" || !device.contains("4090") {
            return Err(format!("ADR 0044 P0 requires an RTX 4090; detected {device}").into());
        }
        let gpu_activity_before = run_text(
            "nvidia-smi",
            &[
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader",
            ],
        );
        let gpu_processes_before = run_text(
            "nvidia-smi",
            &[
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
        );

        let path = model_path();
        let bytes = std::fs::read(&path)?;
        let model_sha256 = Sha256::digest(&bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let file = tritium_format::read_gguf(&bytes)?;
        let backend = tritium_runtime::BACKENDS
            .iter()
            .find(|entry| entry.name == "cuda")
            .ok_or("CUDA backend is not registered")?;
        let backend = (backend.init)()?;
        let mut runner = ModelRunner::load(&file, &bytes, backend)?;
        if !runner.try_resident_decoder()? {
            return Err("model did not admit the CUDA TQ2_0 resident decode path".into());
        }
        let model_architecture = runner.config.arch.clone();
        let model_layers = runner.config.n_layers;
        let model_context_limit = runner.config.n_ctx;
        let reference_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../tools/reference/bitnet_accept.json");
        let reference: Reference = serde_json::from_slice(&std::fs::read(reference_path)?)?;
        if runner.config.n_ctx
            < *CONTEXTS.last().expect("fixed contexts") as u32 + DECODE_STEPS as u32
        {
            return Err(format!(
                "model context {} is too short for P0 matrix",
                runner.config.n_ctx
            )
            .into());
        }

        let mut results = Vec::with_capacity(CONTEXTS.len() * 4);
        for context in CONTEXTS {
            results.push(bench_decode_one(
                &mut runner,
                &reference.token_ids,
                context,
            )?);
            results.push(bench_decode_batch(
                &mut runner,
                &reference.token_ids,
                context,
            )?);
            results.push(bench_tree_one(&mut runner, &reference.token_ids, context)?);
            results.push(bench_tree_batch(
                &mut runner,
                &reference.token_ids,
                context,
            )?);
        }

        let gpu_activity_at_end = run_text(
            "nvidia-smi",
            &[
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader",
            ],
        );
        let gpu_processes_at_end = run_text(
            "nvidia-smi",
            &[
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
        );

        let report = Report {
            schema: "tritium.decode-gate.v1",
            run_id: format!("decode-gate-{}-{}", started_unix_ms, std::process::id()),
            source_revision: revision(),
            source_dirty: source_dirty(),
            started_unix_ms,
            benchmark_pid: std::process::id(),
            device,
            gpu_activity_before,
            gpu_activity_at_end,
            gpu_processes_before,
            gpu_processes_at_end,
            model_path: path.display().to_string(),
            model_bytes: bytes.len() as u64,
            model_sha256,
            model_architecture,
            model_layers,
            model_context_limit,
            artifact_quantization: "GGUF I2_S",
            execution_quantization: "TQ2_0 CUDA resident decode",
            runtime_options: runtime_options(),
            contexts: CONTEXTS,
            decode_steps: DECODE_STEPS,
            tree_nodes_per_sequence: TREE_NODES,
            warmups: WARMUPS,
            samples: SAMPLES,
            results,
            claim_boundary: "performance baseline only; not model-quality or release qualification",
        };
        write_report(&report)
    }
}

fn main() {
    #[cfg(feature = "cuda")]
    if let Err(error) = gate::run() {
        eprintln!("decode gate failed: {error}");
        std::process::exit(1);
    }

    #[cfg(not(feature = "cuda"))]
    eprintln!("decode_gate requires `--features cuda` and a real RTX 4090");
}
