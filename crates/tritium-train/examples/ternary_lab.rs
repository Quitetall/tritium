//! EAT-O: Evidence-Accumulating Ternary Optimizer.
//! Isolated research harness: cargo run -p tritium-train --example ternary_lab -- --help
//! No production API or format changes. All experimental learning arithmetic is integer.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    clippy::struct_excessive_bools
)]
#[path = "ternary_lab/engine.rs"]
mod engine;
#[path = "ternary_lab/model.rs"]
mod model;
#[path = "ternary_lab/numeric.rs"]
mod numeric;
use engine::{Config, EatOptimizer, History, Precision, Route};
use model::{Example, Model};
use numeric::Rng;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};

#[derive(Serialize, Deserialize)]
struct Checkpoint {
    schema: u32,
    dataset: String,
    sample_seed: u64,
    engine: EatOptimizer,
}
fn hash(data: &[u8]) -> String {
    blake3::hash(data).to_hex().to_string()
}
fn digest_data(data: &[Example]) -> String {
    let mut h = blake3::Hasher::new();
    for x in data {
        h.update(&(x.label as u64).to_le_bytes());
        h.update(&(x.pixels.len() as u64).to_le_bytes());
        h.update(&x.pixels);
    }
    h.finalize().to_hex().to_string()
}
fn evaluate(m: &Model, data: &[Example]) -> (i128, usize) {
    (
        data.iter().map(|x| m.loss(x)).sum(),
        data.iter().filter(|x| m.correct(x)).count(),
    )
}
fn toy() -> Vec<Example> {
    (0..32)
        .map(|i| Example {
            pixels: vec![
                if i % 2 == 0 { 255 } else { 0 },
                if i % 2 == 1 { 255 } else { 0 },
            ],
            label: i % 2,
        })
        .collect()
}
fn write_new(path: &str, bytes: &[u8]) -> Result<(), String> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("{path}: {e}"))?;
    f.write_all(bytes).map_err(|e| e.to_string())?;
    f.sync_all().map_err(|e| e.to_string())
}
fn run() -> Result<(), String> {
    let mut args = BTreeMap::new();
    let mut iter = std::env::args().skip(1);
    while let Some(key) = iter.next() {
        if key == "--help" {
            println!(
                "EAT-O — Evidence-Accumulating Ternary Optimizer\n\nternary_lab [--data UNCOMPRESSED_MNIST_IDX_DIR] [--steps 1000] [--seed 1]\n\
                [--route backprop|probe] [--history statistics|replay|both]\n\
                [--precision adaptive|fixed8|fixed24|integer32] [--hysteresis yes|no] [--simple yes|no]\n\
                [--threshold 4] [--coordinates 128] [--budget BYTES] [--replay 256]\n\
                [--eval-limit N] [--split validation|test] [--resume FILE] [--checkpoint NEW_FILE]\n\
                [--report NEW_FILE] [--verify CHECKPOINT]\n\
                Default is the tiny synthetic diagnostic; --data selects 784x128x10 MNIST.\n\
                --steps means additional steps; checkpoint binds data, config, RNG, and sample order.\n\
                MNIST default optimizer+replay budget: 12 bytes/parameter. Toy oracle: 64 KiB."
            );
            return Ok(());
        }
        let value = iter
            .next()
            .ok_or_else(|| format!("missing value for {key}"))?;
        if args.insert(key.clone(), value).is_some() {
            return Err(format!("duplicate {key}"));
        }
    }
    let allowed = [
        "--data",
        "--steps",
        "--seed",
        "--route",
        "--history",
        "--precision",
        "--hysteresis",
        "--simple",
        "--threshold",
        "--coordinates",
        "--budget",
        "--replay",
        "--eval-limit",
        "--eval-offset",
        "--split",
        "--resume",
        "--checkpoint",
        "--report",
        "--verify",
        "--work-limit",
    ];
    for k in args.keys() {
        if !allowed.contains(&k.as_str()) {
            return Err(format!("unknown option {k}"));
        }
    }
    if let Some(path) = args.get("--verify") {
        let mut c: Checkpoint =
            serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        if c.schema != 1 || c.sample_seed == 0 {
            return Err("unsupported checkpoint".into());
        }
        c.engine.compact();
        c.engine.validate()?;
        println!("checkpoint structural verification PASS; no learning-quality claim");
        return Ok(());
    }
    let get = |key: &str, default: &str| args.get(key).cloned().unwrap_or_else(|| default.into());
    let number = |key: &str, default: usize| -> Result<usize, String> {
        get(key, &default.to_string())
            .parse()
            .map_err(|_| format!("invalid {key}"))
    };
    let seed = number("--seed", 1)? as u64;
    if seed == 0 {
        return Err("seed must be nonzero".into());
    }
    let (train, eval, dataset_name) = if let Some(dir) = args.get("--data") {
        let p = Path::new(dir);
        let all = model::mnist(
            &p.join("train-images-idx3-ubyte"),
            &p.join("train-labels-idx1-ubyte"),
        )?;
        if all.len() != 60000 {
            return Err("expected canonical 60000-example MNIST training set".into());
        }
        let eval = match get("--split", "validation").as_str() {
            "validation" => all[55000..].to_vec(),
            "test" => model::mnist(
                &p.join("t10k-images-idx3-ubyte"),
                &p.join("t10k-labels-idx1-ubyte"),
            )?,
            _ => return Err("invalid split".into()),
        };
        (all[..55000].to_vec(), eval, "mnist")
    } else {
        (toy(), toy(), "synthetic-diagnostic-not-held-out")
    };
    let data_digest = digest_data(&train);
    let mut checkpoint = if let Some(path) = args.get("--resume") {
        for key in [
            "--seed",
            "--route",
            "--history",
            "--precision",
            "--hysteresis",
            "--simple",
            "--threshold",
            "--coordinates",
            "--budget",
            "--replay",
        ] {
            if args.contains_key(key) {
                return Err(format!("{key} cannot override a resumed experiment"));
            }
        }
        let mut c: Checkpoint =
            serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        if c.schema != 1 || c.dataset != data_digest || c.sample_seed == 0 {
            return Err("checkpoint/data identity mismatch".into());
        }
        c.engine.compact();
        c.engine.validate()?;
        c
    } else {
        let mut rng = Rng(seed);
        let m = if dataset_name == "mnist" {
            Model::new(784, 128, 10, &mut rng)
        } else {
            let mut m = Model::new(2, 4, 2, &mut rng);
            m.input_shift = 0;
            m.output_shift = 0;
            m
        };
        let config = Config {
            route: match get("--route", "backprop").as_str() {
                "backprop" => Route::Backprop,
                "probe" => Route::Probe,
                _ => return Err("invalid route".into()),
            },
            history: match get("--history", "statistics").as_str() {
                "statistics" => History::Statistics,
                "replay" => History::Replay,
                "both" => History::Both,
                _ => return Err("invalid history".into()),
            },
            precision: match get("--precision", "adaptive").as_str() {
                "adaptive" => Precision::Adaptive,
                "fixed8" => Precision::Fixed8,
                "fixed24" => Precision::Fixed24,
                "integer32" => Precision::Integer32,
                _ => return Err("invalid precision".into()),
            },
            hysteresis: match get("--hysteresis", "yes").as_str() {
                "yes" => true,
                "no" => false,
                _ => return Err("invalid hysteresis".into()),
            },
            simple: match get("--simple", "no").as_str() {
                "yes" => true,
                "no" => false,
                _ => return Err("invalid simple".into()),
            },
            threshold: number("--threshold", 4)? as i128,
            budget: number(
                "--budget",
                if dataset_name == "mnist" {
                    12 * m.trits.len()
                } else {
                    65536
                },
            )?,
            replay_limit: number("--replay", 256)?,
            coordinates: number("--coordinates", 128)?,
        };
        Checkpoint {
            schema: 1,
            dataset: data_digest.clone(),
            sample_seed: seed,
            engine: EatOptimizer::new(m, config, seed)?,
        }
    };
    let eval_offset = number("--eval-offset", 0)?;
    if eval_offset >= eval.len() {
        return Err("evaluation offset outside split".into());
    }
    let eval = &eval[eval_offset..];
    let eval_limit = number("--eval-limit", eval.len())?.min(eval.len());
    if eval_limit == 0 {
        return Err("empty evaluation".into());
    }
    let eval = &eval[..eval_limit];
    let initial = evaluate(&checkpoint.engine.model, eval);
    let start = std::time::Instant::now();
    let work_limit = number("--work-limit", usize::MAX)? as u64;
    let starting_work = checkpoint.engine.metrics.contraction_terms;
    for _ in 0..number("--steps", 1000)? {
        if checkpoint.engine.metrics.contraction_terms - starting_work >= work_limit {
            break;
        }
        // Data order is independent of optimizer RNG consumption and algorithm choice.
        let mut sampler = Rng(checkpoint
            .sample_seed
            .wrapping_add(
                checkpoint
                    .engine
                    .metrics
                    .steps
                    .wrapping_mul(0x9e3779b97f4a7c15),
            )
            .max(1));
        let i = sampler.below(train.len() as u64) as usize;
        checkpoint.engine.step(&train[i]);
    }
    let elapsed = start.elapsed().as_millis();
    checkpoint.engine.validate()?;
    let final_eval = evaluate(&checkpoint.engine.model, eval);
    let bytes = serde_json::to_vec(&checkpoint).map_err(|e| e.to_string())?;
    if let Some(path) = args.get("--checkpoint") {
        write_new(path, &bytes)?;
    }
    let source = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .map_err(|e| e.to_string())?;
    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .map_err(|e| e.to_string())?;
    let proc_status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let rss = proc_status
        .lines()
        .find(|l| l.starts_with("VmHWM:"))
        .unwrap_or("unavailable");
    let report = serde_json::json!({"schema":1,"qualification":"UNKNOWN","dataset":dataset_name,
        "data_digest":data_digest,"evaluation_digest":digest_data(eval),"split":get("--split","validation"),
        "evaluation_count":eval.len(),"evaluation_offset":eval_offset,"initial_loss_sum":initial.0,"loss_sum":final_eval.0,
        "initial_correct":initial.1,"correct":final_eval.1,"seed":checkpoint.sample_seed,
        "config":checkpoint.engine.config,"metrics":checkpoint.engine.metrics,
        "optimizer_replay_allocated_bytes":checkpoint.engine.state_bytes(),
        "weight_allocated_bytes":checkpoint.engine.model.trits.capacity(),"checkpoint_bytes":bytes.len(),
        "checkpoint_digest":hash(&bytes),"elapsed_training_ms":elapsed,"process_peak_rss":rss,
        "source_revision":String::from_utf8_lossy(&source.stdout).trim(),"source_dirty":!dirty.stdout.is_empty(),
        "implementation_digest":hash(concat!(include_str!("ternary_lab.rs"),include_str!("ternary_lab/numeric.rs"),include_str!("ternary_lab/model.rs"),include_str!("ternary_lab/engine.rs")).as_bytes()),
        "machine":std::env::consts::ARCH,"device":"CPU integer reference",
        "zero_weights":checkpoint.engine.model.trits.iter().filter(|&&q|q==0).count(),
        "accumulator_clips":checkpoint.engine.blocks.iter().map(|b|b.bank.clipped).sum::<u64>(),
        "accumulator_rescalings":checkpoint.engine.blocks.iter().map(|b|b.bank.rescalings).sum::<u64>(),
        "replay_examples":checkpoint.engine.replay.len(),"parameters":checkpoint.engine.model.trits.len()});
    let report = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    if let Some(path) = args.get("--report") {
        write_new(path, report.as_bytes())?;
    }
    println!("{report}");
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("EAT-O (ternary_lab): {e}");
        std::process::exit(1);
    }
}
