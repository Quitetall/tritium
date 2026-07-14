//! Physical CUDA-pool A/B for activation checkpointing on the real packed
//! SmolLM2-135M training graph.
//!
//! Each observation runs in a fresh child process so a previous policy cannot
//! seed allocator state for the next one. The parent runs KeepAll/SqrtDepth and
//! then SqrtDepth/KeepAll, while every child warms its own policy before
//! resetting the async-pool used-memory high-water mark. Long-lived packed
//! weights and the target stay resident across the reset; the asserted value is
//! therefore the high-water delta above that common baseline.
//!
//! ```text
//! cargo test -p tritium-nn --features cuda --release \
//!   --test checkpoint_pool_memory_ab -- --ignored --nocapture
//! ```
#![cfg(feature = "cuda")]

mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::{Command, Output};

use common::{device_forward_packed, extract};
use tritium_cuda::CudaBackend;
use tritium_cuda::train::{
    CheckpointPolicy, DeviceBackwardStats, DevicePackedSaltWeight, DeviceTape, DeviceTensor,
    PackedSaltComputePolicy,
};
use tritium_nn::ModelRunner;

const CHILD_ENV: &str = "TRITIUM_CHECKPOINT_POOL_AB_CHILD";
const RESULT_PREFIX: &str = "TRITIUM_CHECKPOINT_POOL_AB_RESULT";
const PLANES: usize = 2;
const SEQ: usize = 32;
const LAYER0_DOWN_MASTER: usize = 7;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Policy {
    KeepAll,
    SqrtDepth,
}

impl Policy {
    fn label(self) -> &'static str {
        match self {
            Self::KeepAll => "keep-all",
            Self::SqrtDepth => "sqrt-depth",
        }
    }

    fn parse(value: &str) -> Self {
        match value {
            "keep-all" => Self::KeepAll,
            "sqrt-depth" => Self::SqrtDepth,
            other => panic!("unknown {CHILD_ENV} policy {other:?}"),
        }
    }

    fn checkpoint(self, layers: usize) -> CheckpointPolicy {
        match self {
            Self::KeepAll => CheckpointPolicy::KeepAll,
            Self::SqrtDepth => CheckpointPolicy::SqrtDepth(layers),
        }
    }
}

#[derive(Debug)]
struct PathEvidence {
    stats: DeviceBackwardStats,
    logits_hash: String,
    gradient_hash: String,
}

#[derive(Clone, Debug)]
struct Measurement {
    policy: Policy,
    baseline_bytes: u64,
    peak_bytes: u64,
    delta_bytes: u64,
    end_bytes: u64,
    packed_bytes: u64,
    naive_activation_elements: u64,
    peak_activation_elements: u64,
    recomputed_ops: u64,
    logits_hash: String,
    gradient_hash: String,
    pci_bus_id: String,
    cuda_driver_version: u64,
}

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".cache/tritium-models/smollm2-135m")
}

fn tokens() -> [i32; SEQ] {
    const SEED: [i32; 8] = [1, 338, 263, 1243, 310, 278, 4086, 29889];
    std::array::from_fn(|index| SEED[index % SEED.len()])
}

fn target(vocab: usize) -> Vec<f32> {
    let mut state = 7_u64;
    let mut logits: Vec<f32> = (0..SEQ * vocab)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 1000) as f32 / 500.0 - 1.0
        })
        .collect();
    for row in logits.chunks_mut(vocab) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for value in row.iter_mut() {
            *value = (*value - max).exp();
            sum += *value;
        }
        for value in row {
            *value /= sum;
        }
    }
    logits
}

fn hash_f32(values: &[f32]) -> String {
    let mut hasher = blake3::Hasher::new();
    for value in values {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

fn run_path(
    backend: &CudaBackend,
    arch: &common::Arch,
    packed_weights: &[DevicePackedSaltWeight],
    target: &DeviceTensor,
    policy: Policy,
    capture_fingerprints: bool,
) -> PathEvidence {
    let tokens = tokens();
    let mut tape = DeviceTape::new_with_policies(
        backend,
        arch.vocab,
        policy.checkpoint(arch.n_layers),
        PackedSaltComputePolicy::Exact,
    )
    .expect("create packed device tape");
    let (logits_id, master_ids) =
        device_forward_packed(&mut tape, arch, packed_weights, &tokens, SEQ);
    let logits_hash = if capture_fingerprints {
        hash_f32(&tape.value(logits_id).expect("download logits fingerprint"))
    } else {
        String::new()
    };
    let gradients = tape
        .xent_backward_device(
            logits_id,
            target,
            SEQ,
            arch.vocab,
            &[master_ids[LAYER0_DOWN_MASTER]],
        )
        .expect("packed backward");
    let stats = gradients.backward_stats();
    let gradient_hash = if capture_fingerprints {
        hash_f32(
            &gradients
                .download(backend, 0)
                .expect("download layer-0 down gradient fingerprint"),
        )
    } else {
        String::new()
    };
    PathEvidence {
        stats,
        logits_hash,
        gradient_hash,
    }
}

fn child_measure(policy: Policy) -> Measurement {
    let dir = model_dir();
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, masters, shapes) = extract(&runner);
    drop(runner);

    let backend = CudaBackend::new(0).expect("open CUDA device");
    let target = DeviceTensor::upload(&backend, &target(arch.vocab)).expect("upload target");
    let packed_weights: Vec<DevicePackedSaltWeight> = masters
        .iter()
        .zip(&shapes)
        .map(|(master, &(rows, cols))| {
            DevicePackedSaltWeight::from_host(&backend, master, rows, cols, PLANES)
                .expect("pack SALT weight")
        })
        .collect();
    let packed_bytes = packed_weights
        .iter()
        .map(DevicePackedSaltWeight::resident_bytes)
        .sum::<usize>();
    drop(masters);
    drop(shapes);

    // Warm the complete candidate path before recording allocator state. This
    // covers checkpoint replay for SqrtDepth and every forward/backward kernel
    // used by the measured pass.
    let _warm = run_path(&backend, &arch, &packed_weights, &target, policy, false);

    let (identity, telemetry) = backend
        .start_memory_telemetry()
        .expect("start exact async-pool telemetry");
    let baseline = telemetry
        .reset_synchronized()
        .expect("reset pool high-water after warmup");
    let evidence = run_path(&backend, &arch, &packed_weights, &target, policy, true);
    let measured = telemetry
        .sample_synchronized()
        .expect("sample pool high-water");

    assert_eq!(
        measured.pool_used_current_bytes, baseline.pool_used_current_bytes,
        "measured path left pool allocations live, so its high-water delta is confounded"
    );
    let delta_bytes = measured
        .pool_used_high_water_bytes
        .checked_sub(baseline.pool_used_current_bytes)
        .expect("pool high-water below baseline");

    Measurement {
        policy,
        baseline_bytes: baseline.pool_used_current_bytes,
        peak_bytes: measured.pool_used_high_water_bytes,
        delta_bytes,
        end_bytes: measured.pool_used_current_bytes,
        packed_bytes: u64::try_from(packed_bytes).expect("packed bytes exceed u64"),
        naive_activation_elements: u64::try_from(evidence.stats.naive_activation_elements)
            .expect("naive activation count exceeds u64"),
        peak_activation_elements: u64::try_from(evidence.stats.peak_live_activation_elements)
            .expect("peak activation count exceeds u64"),
        recomputed_ops: u64::try_from(evidence.stats.recomputed_ops)
            .expect("recomputed op count exceeds u64"),
        logits_hash: evidence.logits_hash,
        gradient_hash: evidence.gradient_hash,
        pci_bus_id: identity.pci_bus_id,
        cuda_driver_version: u64::from(identity.cuda_driver_version),
    }
}

fn print_measurement(measurement: &Measurement) {
    println!(
        "{RESULT_PREFIX} policy={} baseline={} peak={} delta={} end={} packed={} \
         naive_activation={} peak_activation={} recomputed={} logits_hash={} \
         gradient_hash={} pci={} driver={}",
        measurement.policy.label(),
        measurement.baseline_bytes,
        measurement.peak_bytes,
        measurement.delta_bytes,
        measurement.end_bytes,
        measurement.packed_bytes,
        measurement.naive_activation_elements,
        measurement.peak_activation_elements,
        measurement.recomputed_ops,
        measurement.logits_hash,
        measurement.gradient_hash,
        measurement.pci_bus_id,
        measurement.cuda_driver_version,
    );
}

fn field<'a>(fields: &'a BTreeMap<&str, &str>, name: &str) -> &'a str {
    fields
        .get(name)
        .copied()
        .unwrap_or_else(|| panic!("child result omitted {name}"))
}

fn number(fields: &BTreeMap<&str, &str>, name: &str) -> u64 {
    field(fields, name)
        .parse()
        .unwrap_or_else(|error| panic!("child result has invalid {name}: {error}"))
}

fn parse_measurement(output: &Output, expected: Policy) -> Measurement {
    assert!(
        output.status.success(),
        "{} child failed\nstdout:\n{}\nstderr:\n{}",
        expected.label(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8(output.stdout.clone()).expect("child stdout is not UTF-8");
    let line = stdout
        .lines()
        .find_map(|line| {
            line.find(RESULT_PREFIX)
                .map(|prefix_start| &line[prefix_start..])
        })
        .unwrap_or_else(|| panic!("{} child emitted no result:\n{stdout}", expected.label()));
    let fields: BTreeMap<&str, &str> = line
        .split_whitespace()
        .skip(1)
        .map(|item| {
            item.split_once('=')
                .unwrap_or_else(|| panic!("malformed child result field {item:?}"))
        })
        .collect();
    let policy = Policy::parse(field(&fields, "policy"));
    assert_eq!(policy, expected, "child measured the wrong policy");
    Measurement {
        policy,
        baseline_bytes: number(&fields, "baseline"),
        peak_bytes: number(&fields, "peak"),
        delta_bytes: number(&fields, "delta"),
        end_bytes: number(&fields, "end"),
        packed_bytes: number(&fields, "packed"),
        naive_activation_elements: number(&fields, "naive_activation"),
        peak_activation_elements: number(&fields, "peak_activation"),
        recomputed_ops: number(&fields, "recomputed"),
        logits_hash: field(&fields, "logits_hash").to_owned(),
        gradient_hash: field(&fields, "gradient_hash").to_owned(),
        pci_bus_id: field(&fields, "pci").to_owned(),
        cuda_driver_version: number(&fields, "driver"),
    }
}

fn spawn_child(policy: Policy) -> Measurement {
    let output = Command::new(std::env::current_exe().expect("locate integration-test binary"))
        .args([
            "--exact",
            "physical_checkpoint_keep_all_vs_sqrt_depth_uses_less_pool_memory",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_ENV, policy.label())
        .output()
        .unwrap_or_else(|error| panic!("spawn {} child: {error}", policy.label()));
    parse_measurement(&output, policy)
}

#[test]
#[ignore = "ADR 0027 physical CUDA-pool A/B; requires local SmolLM2-135M weights"]
fn physical_checkpoint_keep_all_vs_sqrt_depth_uses_less_pool_memory() {
    if let Ok(value) = std::env::var(CHILD_ENV) {
        print_measurement(&child_measure(Policy::parse(&value)));
        return;
    }

    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }

    // Fresh processes make allocator/JIT state policy-local; reversing the
    // launch order supplies a second observation against temporal drift.
    let observations = [
        spawn_child(Policy::KeepAll),
        spawn_child(Policy::SqrtDepth),
        spawn_child(Policy::SqrtDepth),
        spawn_child(Policy::KeepAll),
    ];
    let keep = [&observations[0], &observations[3]];
    let checkpoint = [&observations[1], &observations[2]];

    let first = &observations[0];
    for observation in &observations[1..] {
        assert_eq!(
            observation.baseline_bytes, first.baseline_bytes,
            "long-lived pool baseline changed across fresh child processes"
        );
        assert_eq!(
            observation.end_bytes, observation.baseline_bytes,
            "child did not return to its measured baseline"
        );
        assert_eq!(observation.packed_bytes, first.packed_bytes);
        assert_eq!(
            observation.naive_activation_elements, first.naive_activation_elements,
            "policies did not execute the same logical graph"
        );
        assert_eq!(observation.logits_hash, first.logits_hash);
        assert_eq!(observation.gradient_hash, first.gradient_hash);
        assert_eq!(observation.pci_bus_id, first.pci_bus_id);
        assert_eq!(observation.cuda_driver_version, first.cuda_driver_version);
    }

    for observation in keep {
        assert_eq!(observation.recomputed_ops, 0);
        assert_eq!(
            observation.peak_activation_elements, observation.naive_activation_elements,
            "KeepAll unexpectedly evicted logical activations"
        );
    }
    for observation in checkpoint {
        assert!(
            observation.recomputed_ops > 0,
            "SqrtDepth did not replay checkpointed operations"
        );
        assert!(
            observation.peak_activation_elements < observation.naive_activation_elements,
            "SqrtDepth did not lower logical activation residency"
        );
    }

    let min_keep_delta = keep
        .iter()
        .map(|observation| observation.delta_bytes)
        .min()
        .expect("two KeepAll observations");
    let max_checkpoint_delta = checkpoint
        .iter()
        .map(|observation| observation.delta_bytes)
        .max()
        .expect("two SqrtDepth observations");
    println!(
        "ADR 0027 physical checkpoint A/B: SmolLM2-135M packed Exact seq={SEQ}; \
         KeepAll deltas=[{}, {}] bytes, SqrtDepth deltas=[{}, {}] bytes; \
         conservative reduction={} bytes; baseline={} packed={} driver={} pci={}; \
         logits_hash={} gradient_hash={}",
        keep[0].delta_bytes,
        keep[1].delta_bytes,
        checkpoint[0].delta_bytes,
        checkpoint[1].delta_bytes,
        min_keep_delta.saturating_sub(max_checkpoint_delta),
        first.baseline_bytes,
        first.packed_bytes,
        first.cuda_driver_version,
        first.pci_bus_id,
        first.logits_hash,
        first.gradient_hash,
    );
    assert!(
        max_checkpoint_delta < min_keep_delta,
        "SqrtDepth physical peak did not beat either KeepAll observation: {observations:#?}"
    );
}
