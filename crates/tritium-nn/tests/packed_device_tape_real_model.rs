//! Real-model gate for compact SALT training execution. It compares the
//! ordinary dense T=2 SALT device tape against packed projections plus
//! sqrt-depth activation checkpointing on the local SmolLM2-135M weights.
//! OPEN (RTX 4090, seq 8): the strict whole-model max-absolute parity gate is
//! still negative at logits 1.698e-4, tied-embedding gradient 2.508e-4, and
//! layer-0 down gradient 6.714e-5. Compact residency is 0.1615x dense and the
//! measured checkpoint activation peak is 0.1846x naive residency.
//!
//! ```text
//! cargo test -p tritium-nn --features cuda --release --test packed_device_tape_real_model -- --ignored --nocapture
//! ```
#![cfg(feature = "cuda")]

mod common;

use std::path::PathBuf;
use std::time::Instant;

use common::{device_forward, device_forward_packed, extract};
use tritium_cuda::CudaBackend;
use tritium_cuda::train::{
    CheckpointPolicy, DeviceBackwardStats, DevicePackedSaltWeight, DeviceTape, DeviceTensor,
};
use tritium_nn::ModelRunner;
use tritium_train::ops::ste;

const PLANES: usize = 2;

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".cache/tritium-models/smollm2-135m")
}

fn seeded(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 1000) as f32 / 500.0 - 1.0
        })
        .collect()
}

fn row_softmax(logits: &[f32], vocab: usize) -> Vec<f32> {
    let mut probabilities = logits.to_vec();
    for row in probabilities.chunks_mut(vocab) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for value in row.iter_mut() {
            *value = (*value - max).exp();
            sum += *value;
        }
        for value in row.iter_mut() {
            *value /= sum;
        }
    }
    probabilities
}

fn max_abs_diff(reference: &[f32], actual: &[f32]) -> f32 {
    assert_eq!(reference.len(), actual.len());
    reference
        .iter()
        .zip(actual)
        .map(|(&expected, &got)| (expected - got).abs())
        .fold(0.0, f32::max)
}

struct PathResult {
    logits: Vec<f32>,
    tied_embedding_grad: Vec<f32>,
    layer0_down_grad: Vec<f32>,
    stats: DeviceBackwardStats,
    elapsed_ms: f64,
}

#[test]
#[ignore = "OPEN: packed whole-model parity misses strict 1e-4; needs local SmolLM2-135M + CUDA"]
fn packed_salt_checkpointed_smollm2_matches_dense_salt() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }

    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, masters, shapes) = extract(&runner);
    let tokens = [1_i32, 338, 263, 1243, 310, 278, 4086, 29889];
    let seq = tokens.len();
    let target_host = row_softmax(&seeded(7, seq * arch.vocab), arch.vocab);
    let backend = CudaBackend::new(0).expect("open CUDA device");
    let target = DeviceTensor::upload(&backend, &target_host).expect("upload target");

    let dense_salt: Vec<Vec<f32>> = masters
        .iter()
        .zip(&shapes)
        .map(|(master, &(rows, cols))| ste::salt_quantize_forward(master, rows, cols, PLANES))
        .collect();
    let dense_bytes: usize = dense_salt
        .iter()
        .map(|weight| weight.len() * size_of::<f32>())
        .sum();
    let dense = {
        let started = Instant::now();
        let mut tape = DeviceTape::new(&backend, arch.vocab).expect("dense device tape");
        let (logits_id, weight_ids) = device_forward(&mut tape, &arch, &dense_salt, &tokens, seq);
        let logits = tape.value(logits_id).expect("download dense logits");
        let gradients = tape
            .xent_backward_device(
                logits_id,
                &target,
                seq,
                arch.vocab,
                &[weight_ids[0], weight_ids[7]],
            )
            .expect("dense SALT backward");
        PathResult {
            logits,
            tied_embedding_grad: gradients
                .download(&backend, 0)
                .expect("download dense tied-embedding gradient"),
            layer0_down_grad: gradients
                .download(&backend, 1)
                .expect("download dense layer-0 down gradient"),
            stats: gradients.backward_stats(),
            elapsed_ms: started.elapsed().as_secs_f64() * 1e3,
        }
    };
    drop(dense_salt);

    let packed_weights: Vec<DevicePackedSaltWeight> = masters
        .iter()
        .zip(&shapes)
        .map(|(master, &(rows, cols))| {
            DevicePackedSaltWeight::from_host(&backend, master, rows, cols, PLANES)
                .expect("pack SALT weight")
        })
        .collect();
    assert!(
        packed_weights
            .iter()
            .all(DevicePackedSaltWeight::is_prepared)
    );
    let packed_code_bytes: usize = packed_weights
        .iter()
        .map(DevicePackedSaltWeight::packed_bytes)
        .sum();
    let packed_scale_bytes: usize = packed_weights
        .iter()
        .map(DevicePackedSaltWeight::scale_bytes)
        .sum();
    let packed_bytes: usize = packed_weights
        .iter()
        .map(DevicePackedSaltWeight::resident_bytes)
        .sum();
    assert_eq!(packed_bytes, packed_code_bytes + packed_scale_bytes);

    let packed = {
        let started = Instant::now();
        let mut tape = DeviceTape::new_with_checkpoint_policy(
            &backend,
            arch.vocab,
            CheckpointPolicy::SqrtDepth(arch.n_layers),
        )
        .expect("packed checkpointed device tape");
        let (logits_id, master_ids) =
            device_forward_packed(&mut tape, &arch, &packed_weights, &tokens, seq);
        let logits = tape.value(logits_id).expect("download packed logits");
        let gradients = tape
            .xent_backward_device(
                logits_id,
                &target,
                seq,
                arch.vocab,
                &[master_ids[0], master_ids[7]],
            )
            .expect("packed SALT backward");
        PathResult {
            logits,
            tied_embedding_grad: gradients
                .download(&backend, 0)
                .expect("download packed tied-embedding gradient"),
            layer0_down_grad: gradients
                .download(&backend, 1)
                .expect("download packed layer-0 down gradient"),
            stats: gradients.backward_stats(),
            elapsed_ms: started.elapsed().as_secs_f64() * 1e3,
        }
    };

    let logits_error = max_abs_diff(&dense.logits, &packed.logits);
    let tied_embedding_grad_error =
        max_abs_diff(&dense.tied_embedding_grad, &packed.tied_embedding_grad);
    let layer0_down_grad_error = max_abs_diff(&dense.layer0_down_grad, &packed.layer0_down_grad);
    let byte_ratio = packed_bytes as f64 / dense_bytes as f64;
    let activation_ratio = packed.stats.peak_live_activation_elements as f64
        / packed.stats.naive_activation_elements as f64;

    println!(
        "packed SALT real-model gate: SmolLM2-135M, T={PLANES}, {} layers, seq {seq}; \
         max-abs deltas logits={logits_error:.3e}, \
         tied_embedding_grad={tied_embedding_grad_error:.3e}, \
         layer0_down_grad={layer0_down_grad_error:.3e}; packed bytes={packed_bytes} \
         (codes={packed_code_bytes}, scales={packed_scale_bytes}) vs dense bytes={dense_bytes} \
         ({byte_ratio:.4}x); checkpoint activations peak={} vs naive={} ({activation_ratio:.4}x), \
         recomputed_ops={}; elapsed dense={:.0}ms packed={:.0}ms",
        arch.n_layers,
        packed.stats.peak_live_activation_elements,
        packed.stats.naive_activation_elements,
        packed.stats.recomputed_ops,
        dense.elapsed_ms,
        packed.elapsed_ms,
    );

    assert!(
        logits_error < 1e-4,
        "packed logits max-abs error {logits_error:.3e}"
    );
    assert!(
        tied_embedding_grad_error < 1e-4,
        "packed tied-embedding gradient max-abs error {tied_embedding_grad_error:.3e}"
    );
    assert!(
        layer0_down_grad_error < 1e-4,
        "packed layer-0 down gradient max-abs error {layer0_down_grad_error:.3e}"
    );
    assert!(
        packed.stats.recomputed_ops > 0,
        "checkpoint replay did not run"
    );
    assert!(
        packed.stats.peak_live_activation_elements < packed.stats.naive_activation_elements,
        "checkpointing did not reduce peak activation residency: {:?}",
        packed.stats
    );
    assert!(
        byte_ratio < 0.17,
        "packed SALT uses {byte_ratio:.4}x dense storage, above the measured gate"
    );
}
