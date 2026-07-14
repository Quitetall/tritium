//! ADR 0027 Track A sequence-scaling evidence.
//!
//! Each invocation measures one fresh-process cell. The caller selects the
//! baseline (`track0`) or device-resident (`resident`) path and one of the
//! committed sequence lengths. No speed assertion belongs in a single cell:
//! the emitted machine-readable records are compared after all six cells run.

#![cfg(feature = "cuda")]

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use serde_json::json;
use tritium_cuda::CudaBackend;
use tritium_cuda::train::{DeviceTape, DeviceTensor, DeviceTrainParam, DeviceTrainer};
use tritium_format::TeacherCacheHeader;
use tritium_nn::{
    ModelWeights, TeacherCacheReader, TiedSwiGluTrainingModel, hash_teacher_corpus,
    resident_device_forward, semantic_training_model_digest,
};
use tritium_train::ops::ste;
use tritium_train::{AdamW, Optimizer};

const SALT_PLANES: usize = 2;
const LEARNING_RATE: f32 = 2e-3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Track0,
    Resident,
}

impl Mode {
    fn parse(value: &str) -> Self {
        match value {
            "track0" => Self::Track0,
            "resident" => Self::Resident,
            other => panic!("TRITIUM_TRACK_A_MODE must be track0 or resident, got {other:?}"),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Track0 => "track0",
            Self::Resident => "resident",
        }
    }
}

struct Harness<'a> {
    backend: &'a CudaBackend,
    model: &'a TiedSwiGluTrainingModel,
    train_ids: &'a [u32],
    seq_len: usize,
    windows: usize,
    warmup: usize,
    samples: usize,
    timing_fence: &'a DeviceTensor,
    optimizer: AdamW,
}

impl Harness<'_> {
    fn total_steps(&self) -> usize {
        self.warmup
            .checked_add(self.samples)
            .expect("warmup + sample count overflow")
    }

    fn window(&self, index: usize) -> &[u32] {
        let offset = index
            .checked_mul(self.seq_len)
            .expect("training-window offset overflow");
        &self.train_ids[offset..offset + self.seq_len]
    }

    fn read_step_inputs(
        &self,
        cache: &mut TeacherCacheReader,
        step_zero_based: usize,
        target_host: &mut [f32],
    ) -> (u64, Vec<i32>) {
        let window_index = step_zero_based % self.windows;
        cache
            .read_window(window_index as u64, target_host)
            .unwrap_or_else(|error| panic!("read teacher window {window_index}: {error}"));
        let tokens = self
            .window(window_index)
            .iter()
            .map(|&token| i32::try_from(token).expect("token id exceeds i32"))
            .collect();
        (window_index as u64, tokens)
    }

    fn measured_window_indices(&self) -> Vec<u64> {
        (self.warmup..self.total_steps())
            .map(|step| (step % self.windows) as u64)
            .collect()
    }
}

#[test]
#[ignore = "ADR 0027 Track A: one explicit CUDA sequence-scaling cell"]
fn track_a_sequence_scaling_cell() {
    let mode = Mode::parse(&required_env("TRITIUM_TRACK_A_MODE"));
    let seq_len = parse_required_usize("TRITIUM_TRACK_A_SEQ");
    assert!(
        matches!(seq_len, 32 | 64 | 128),
        "TRITIUM_TRACK_A_SEQ must be 32, 64, or 128, got {seq_len}"
    );
    let warmup = parse_required_usize("TRITIUM_TIMING_WARMUP");
    assert!(warmup > 0, "TRITIUM_TIMING_WARMUP must be positive");
    let samples = parse_required_usize("TRITIUM_TIMING_SAMPLES");
    assert!(samples > 0, "TRITIUM_TIMING_SAMPLES must be positive");
    let corpus_path = PathBuf::from(required_env("TRITIUM_CORPUS"));
    let cache_path = PathBuf::from(required_env("TRITIUM_TEACHER_CACHE"));
    let model_dir = model_dir();
    assert!(
        model_dir.join("model.safetensors").is_file(),
        "production SmolLM2-135M weights are absent at {}",
        model_dir.display()
    );

    let train_ids = load_train_ids(&corpus_path);
    assert!(!train_ids.is_empty(), "training corpus is empty");
    assert_eq!(
        train_ids.len() % seq_len,
        0,
        "training corpus must contain an exact number of {seq_len}-token windows"
    );
    let windows = train_ids.len() / seq_len;
    assert!(windows > 0, "training corpus contains no complete window");

    let (config, spec, weights) = ModelWeights::load_hf(&model_dir).expect("load SmolLM2-135M");
    let mut model =
        TiedSwiGluTrainingModel::extract(&config, &spec, &weights).expect("adapt training model");
    drop(weights);
    assert_eq!(
        model.architecture().n_ff,
        config.n_ff as usize,
        "training model must retain the production intermediate width"
    );
    assert!(
        seq_len <= model.architecture().n_ctx,
        "sequence length exceeds model context"
    );
    assert!(
        train_ids
            .iter()
            .all(|&token| token as usize <= model.architecture().vocab.saturating_sub(1)),
        "training corpus contains a token outside the model vocabulary"
    );

    let model_digest = semantic_training_model_digest(&config, &spec, &model);
    let corpus_digest = hash_teacher_corpus(&train_ids, seq_len as u32);
    let cache_digest = hash_file(&cache_path);
    let expected_header = TeacherCacheHeader {
        seq_len: seq_len as u32,
        vocab: model.architecture().vocab as u32,
        windows: windows as u64,
        model_hash: model_digest,
        corpus_hash: corpus_digest,
    };
    let mut teacher_cache = TeacherCacheReader::open(&cache_path, &expected_header)
        .expect("open exact model/corpus/geometry teacher cache");

    // Drain the canonical masters only after computing the semantic identity.
    // The model retains the authoritative parameter order and geometry used by
    // both paths, while each path starts from this same single host allocation.
    let initial_masters = model.take_parameter_masters();
    assert_eq!(initial_masters.len(), model.parameters().len());

    let backend = CudaBackend::new(0).expect("open CUDA device 0");
    let (device, _memory_telemetry) = backend
        .start_memory_telemetry()
        .expect("capture synchronized CUDA identity");
    let timing_fence =
        DeviceTensor::upload(&backend, &[0.0]).expect("allocate one-float timing fence");
    let harness = Harness {
        backend: &backend,
        model: &model,
        train_ids: &train_ids,
        seq_len,
        windows,
        warmup,
        samples,
        timing_fence: &timing_fence,
        optimizer: AdamW::new(LEARNING_RATE),
    };

    let (raw_step_ms, diagnostic_nll) = match mode {
        Mode::Track0 => run_track0(&harness, &mut teacher_cache, initial_masters),
        Mode::Resident => run_resident(&harness, &mut teacher_cache, initial_masters),
    };
    assert_eq!(raw_step_ms.len(), samples);
    assert!(
        raw_step_ms
            .iter()
            .all(|sample| sample.is_finite() && *sample > 0.0),
        "every measured step must have a finite positive duration: {raw_step_ms:?}"
    );
    assert!(
        diagnostic_nll.is_finite() && diagnostic_nll > 0.0,
        "the exact-window student NLL must be finite and positive, got {diagnostic_nll}"
    );
    let mean_step_ms = raw_step_ms.iter().sum::<f64>() / raw_step_ms.len() as f64;
    assert!(mean_step_ms.is_finite() && mean_step_ms > 0.0);

    println!(
        "TRITIUM_TRACK_A_RESULT={}",
        serde_json::to_string(&json!({
            "schema": "tritium.adr0027.track-a-sequence-scaling.v1",
            "mode": mode.as_str(),
            "seq_len": seq_len,
            "warmup_steps": warmup,
            "measured_steps": samples,
            "measured_window_indices": harness.measured_window_indices(),
            "raw_step_ms": raw_step_ms,
            "mean_step_ms": mean_step_ms,
            "diagnostic_window_index": harness.total_steps() % windows,
            "diagnostic_nll": diagnostic_nll,
            "salt_planes": SALT_PLANES,
            "adamw": {
                "learning_rate": harness.optimizer.lr,
                "beta1": harness.optimizer.beta1,
                "beta2": harness.optimizer.beta2,
                "epsilon": harness.optimizer.eps,
                "weight_decay": harness.optimizer.weight_decay,
            },
            "model_digest": hex_digest(model_digest),
            "corpus_digest": hex_digest(corpus_digest),
            "teacher_cache_digest": hex_digest(cache_digest),
            "teacher_cache_windows": windows,
            "git_commit": git_head(),
            "model_dir": model_dir,
            "corpus_path": corpus_path,
            "teacher_cache_path": cache_path,
            "cuda": {
                "ordinal": device.ordinal,
                "device_name": device.device_name,
                "pci_bus_id": device.pci_bus_id,
                "driver_api_version": device.cuda_driver_version,
            },
        }))
        .expect("serialize Track A evidence")
    );
}

fn run_track0(
    harness: &Harness<'_>,
    cache: &mut TeacherCacheReader,
    mut masters: Vec<Vec<f32>>,
) -> (Vec<f64>, f64) {
    let mut states: Vec<_> = masters
        .iter()
        .map(|master| harness.optimizer.init_state(master.len()))
        .collect();
    let mut target_host = vec![0.0; harness.seq_len * harness.model.architecture().vocab];
    let mut raw_step_ms = Vec::with_capacity(harness.samples);

    for step_zero_based in 0..harness.total_steps() {
        let started = Instant::now();
        let (_window_index, tokens) =
            harness.read_step_inputs(cache, step_zero_based, &mut target_host);
        let target =
            DeviceTensor::upload(harness.backend, &target_host).expect("upload teacher target");
        let dense = quantize_masters(harness.model, &masters);
        let uploaded: Vec<_> = dense
            .iter()
            .map(|weight| {
                DeviceTensor::upload(harness.backend, weight).expect("upload Track-0 weight")
            })
            .collect();
        let weights: Vec<_> = uploaded.iter().collect();
        let gradients = {
            let mut tape = DeviceTape::new(harness.backend, harness.model.architecture().vocab)
                .expect("construct Track-0 device tape");
            let forward = resident_device_forward(&mut tape, harness.model, &weights, &tokens)
                .expect("build Track-0 dense forward");
            tape.xent_backward_device(
                forward.logits,
                &target,
                harness.seq_len,
                harness.model.architecture().vocab,
                &forward.master_leaves,
            )
            .expect("run Track-0 device backward")
        };
        drop(weights);
        for (index, ((master, state), _parameter)) in masters
            .iter_mut()
            .zip(&mut states)
            .zip(harness.model.parameters())
            .enumerate()
        {
            let gradient = gradients
                .download(harness.backend, index)
                .unwrap_or_else(|error| panic!("download Track-0 gradient {index}: {error}"));
            harness
                .optimizer
                .step((step_zero_based + 1) as u64, master, &gradient, state);
        }
        drop(gradients);
        drop(uploaded);
        drop(dense);
        drop(target);
        drop(tokens);
        harness
            .timing_fence
            .download(harness.backend)
            .expect("synchronize Track-0 step with one-float fence");
        let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
        if step_zero_based >= harness.warmup {
            raw_step_ms.push(elapsed_ms);
        }
    }

    let nll = diagnostic_track0_nll(harness, cache, &masters, &mut target_host);
    (raw_step_ms, nll)
}

fn run_resident(
    harness: &Harness<'_>,
    cache: &mut TeacherCacheReader,
    initial_masters: Vec<Vec<f32>>,
) -> (Vec<f64>, f64) {
    let specs: Vec<_> = initial_masters
        .iter()
        .zip(harness.model.parameters())
        .map(|(master, parameter)| DeviceTrainParam {
            master,
            rows: parameter.rows,
            cols: parameter.cols,
            salt_planes: SALT_PLANES,
            optimizer: harness.optimizer,
        })
        .collect();
    let mut trainer =
        DeviceTrainer::new(harness.backend, &specs).expect("upload resident trainer state");
    drop(specs);
    drop(initial_masters);

    let mut target_host = vec![0.0; harness.seq_len * harness.model.architecture().vocab];
    let mut raw_step_ms = Vec::with_capacity(harness.samples);
    for step_zero_based in 0..harness.total_steps() {
        let started = Instant::now();
        let (_window_index, tokens) =
            harness.read_step_inputs(cache, step_zero_based, &mut target_host);
        let target =
            DeviceTensor::upload(harness.backend, &target_host).expect("upload teacher target");
        trainer
            .prepare_quantized()
            .expect("reconstruct resident SALT weights");
        let weights: Vec<_> = (0..trainer.len())
            .map(|index| trainer.quantized(index).expect("borrow resident weight"))
            .collect();
        let gradients = {
            let mut tape = DeviceTape::new(harness.backend, harness.model.architecture().vocab)
                .expect("construct resident device tape");
            let forward = resident_device_forward(&mut tape, harness.model, &weights, &tokens)
                .expect("build resident dense forward");
            tape.xent_backward_device(
                forward.logits,
                &target,
                harness.seq_len,
                harness.model.architecture().vocab,
                &forward.master_leaves,
            )
            .expect("run resident device backward")
        };
        drop(weights);
        trainer
            .step(gradients, (step_zero_based + 1) as u64)
            .expect("apply resident AdamW step");
        drop(target);
        drop(tokens);
        harness
            .timing_fence
            .download(harness.backend)
            .expect("synchronize resident step with one-float fence");
        let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
        if step_zero_based >= harness.warmup {
            raw_step_ms.push(elapsed_ms);
        }
    }

    let nll = diagnostic_resident_nll(harness, cache, &mut trainer, &mut target_host);
    (raw_step_ms, nll)
}

fn diagnostic_track0_nll(
    harness: &Harness<'_>,
    cache: &mut TeacherCacheReader,
    masters: &[Vec<f32>],
    target_host: &mut [f32],
) -> f64 {
    let step = harness.total_steps();
    let (_window_index, tokens) = harness.read_step_inputs(cache, step, target_host);
    let dense = quantize_masters(harness.model, masters);
    let uploaded: Vec<_> = dense
        .iter()
        .map(|weight| {
            DeviceTensor::upload(harness.backend, weight).expect("upload diagnostic weight")
        })
        .collect();
    let weights: Vec<_> = uploaded.iter().collect();
    let mut tape = DeviceTape::new(harness.backend, harness.model.architecture().vocab)
        .expect("construct Track-0 diagnostic tape");
    let forward = resident_device_forward(&mut tape, harness.model, &weights, &tokens)
        .expect("build Track-0 diagnostic forward");
    let logits = tape
        .value(forward.logits)
        .expect("download Track-0 diagnostic logits");
    soft_target_nll(
        &logits,
        target_host,
        harness.seq_len,
        harness.model.architecture().vocab,
    )
}

fn diagnostic_resident_nll(
    harness: &Harness<'_>,
    cache: &mut TeacherCacheReader,
    trainer: &mut DeviceTrainer<'_>,
    target_host: &mut [f32],
) -> f64 {
    let step = harness.total_steps();
    let (_window_index, tokens) = harness.read_step_inputs(cache, step, target_host);
    trainer
        .prepare_quantized()
        .expect("reconstruct diagnostic resident weights");
    let weights: Vec<_> = (0..trainer.len())
        .map(|index| trainer.quantized(index).expect("borrow diagnostic weight"))
        .collect();
    let mut tape = DeviceTape::new(harness.backend, harness.model.architecture().vocab)
        .expect("construct resident diagnostic tape");
    let forward = resident_device_forward(&mut tape, harness.model, &weights, &tokens)
        .expect("build resident diagnostic forward");
    let logits = tape
        .value(forward.logits)
        .expect("download resident diagnostic logits");
    soft_target_nll(
        &logits,
        target_host,
        harness.seq_len,
        harness.model.architecture().vocab,
    )
}

fn quantize_masters(model: &TiedSwiGluTrainingModel, masters: &[Vec<f32>]) -> Vec<Vec<f32>> {
    assert_eq!(masters.len(), model.parameters().len());
    masters
        .iter()
        .zip(model.parameters())
        .map(|(master, parameter)| {
            ste::salt_quantize_forward(master, parameter.rows, parameter.cols, SALT_PLANES)
        })
        .collect()
}

fn soft_target_nll(logits: &[f32], target: &[f32], rows: usize, cols: usize) -> f64 {
    let expected = rows.checked_mul(cols).expect("NLL shape overflow");
    assert_eq!(logits.len(), expected, "diagnostic logits shape");
    assert_eq!(target.len(), expected, "diagnostic target shape");
    let mut total = 0.0f64;
    for (row_index, (logit_row, target_row)) in logits
        .chunks_exact(cols)
        .zip(target.chunks_exact(cols))
        .enumerate()
    {
        assert!(
            logit_row.iter().all(|value| value.is_finite()),
            "diagnostic logits row {row_index} contains a non-finite value"
        );
        assert!(
            target_row
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0),
            "teacher target row {row_index} is not a finite probability vector"
        );
        let target_sum = target_row.iter().map(|&value| value as f64).sum::<f64>();
        assert!(
            (target_sum - 1.0).abs() <= 1e-3,
            "teacher target row {row_index} sums to {target_sum}, not one"
        );
        let max = logit_row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let log_sum_exp = max
            + logit_row
                .iter()
                .map(|&value| ((value as f64) - max).exp())
                .sum::<f64>()
                .ln();
        total += target_row
            .iter()
            .zip(logit_row)
            .map(|(&probability, &logit)| probability as f64 * (log_sum_exp - logit as f64))
            .sum::<f64>();
    }
    total / rows as f64
}

fn required_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set explicitly"))
}

fn parse_required_usize(name: &str) -> usize {
    let value = required_env(name);
    value
        .parse::<usize>()
        .unwrap_or_else(|_| panic!("{name} must be an unsigned integer, got {value:?}"))
}

fn model_dir() -> PathBuf {
    let home = std::env::var_os("HOME").expect("HOME must locate the production model cache");
    PathBuf::from(home).join(".cache/tritium-models/smollm2-135m")
}

fn load_train_ids(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("read corpus {}: {error}", path.display()));
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|error| panic!("parse corpus {}: {error}", path.display()));
    value
        .as_array()
        .or_else(|| value.get("train_ids").and_then(serde_json::Value::as_array))
        .unwrap_or_else(|| {
            panic!(
                "corpus {} must be a token array or contain a train_ids array",
                path.display()
            )
        })
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let token = value
                .as_u64()
                .unwrap_or_else(|| panic!("corpus train_ids[{index}] is not an unsigned integer"));
            u32::try_from(token).unwrap_or_else(|_| panic!("corpus train_ids[{index}] exceeds u32"))
        })
        .collect()
}

fn hash_file(path: &Path) -> [u8; 32] {
    let mut file =
        File::open(path).unwrap_or_else(|error| panic!("open {}: {error}", path.display()));
    let mut hash = blake3::Hasher::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .unwrap_or_else(|error| panic!("hash {}: {error}", path.display()));
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    *hash.finalize().as_bytes()
}

fn hex_digest(digest: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn git_head() -> String {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crate must live under the repository root");
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(repo)
        .output()
        .expect("run git rev-parse for evidence identity");
    assert!(
        output.status.success(),
        "git rev-parse failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let commit = String::from_utf8(output.stdout).expect("git commit identity is not UTF-8");
    let commit = commit.trim();
    assert_eq!(commit.len(), 40, "git commit identity is not 40-hex");
    assert!(
        commit.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "git commit identity is not hexadecimal"
    );
    commit.to_owned()
}
