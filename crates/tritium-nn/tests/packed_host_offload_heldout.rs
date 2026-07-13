//! ADR 0027 packed-SALT plus host-offloaded AdamW real-model gate.
//!
//! This keeps the held-out corpus, T=2 teacher-distillation methodology, 40-step
//! quality threshold, and 1123ms Track-0 timing threshold from
//! `salt_distill_heldout`. The student reads compact packed planes through a
//! sqrt-depth checkpointed `DeviceTape`; all latent masters and Adam moments
//! live in `HostOffloadTrainer` between steps.
//!
//! OPEN (RTX 4090, online teacher, 40 steps): quality passes at 963x recovery
//! (distilled perplexity 2207.241 versus 2.125e6 PTQ), but mean step time is
//! 1855ms and misses the unchanged 1123ms Track-0 performance gate.
//!
//! ```text
//! cargo test -p tritium-nn --features cuda --release --test packed_host_offload_heldout -- --ignored --nocapture
//! ```
#![cfg(feature = "cuda")]

mod common;

use std::path::PathBuf;
use std::time::Instant;

use common::{device_forward, device_forward_packed, extract, logits_of, perplexity};
use tritium_cuda::CudaBackend;
use tritium_cuda::train::{
    CheckpointPolicy, DevicePackedSaltWeight, DeviceTape, DeviceTensor, DeviceTrainParam,
    HostOffloadTrainer,
};
use tritium_format::TeacherCacheHeader;
use tritium_nn::{ModelRunner, TeacherCacheReader, hash_teacher_corpus, hash_teacher_weights};
use tritium_train::AdamW;
use tritium_train::ops::ste;

const PLANES: usize = 2;
const TRAIN_SEQ: usize = 32;
const STEPS: u64 = 40;
const LR: f32 = 2e-3;
const MIN_RECOVERY: f64 = 900.0;
const TRACK0_STEP_MS: f64 = 1123.0;

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".cache/tritium-models/smollm2-135m")
}

fn corpus() -> (Vec<u32>, Vec<u32>) {
    let path = std::env::var("TRITIUM_CORPUS").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tools/reference/heldout_corpus.json"
        )
        .to_string()
    });
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).expect("corpus json")).expect("parse corpus");
    let ids = |key: &str| {
        json[key]
            .as_array()
            .expect(key)
            .iter()
            .map(|value| value.as_u64().expect("token id") as u32)
            .collect::<Vec<_>>()
    };
    (ids("train_ids"), ids("eval_ids"))
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

/// Repack as an all-or-stale operation. A failed individual repack invalidates
/// every handle so no later forward can mix old and new master generations.
fn repack_all_or_stale(
    backend: &CudaBackend,
    packed: &mut [DevicePackedSaltWeight],
    trainer: &HostOffloadTrainer<'_>,
) {
    assert_eq!(packed.len(), trainer.len());
    let mut failure = None;
    for (index, handle) in packed.iter_mut().enumerate() {
        match trainer
            .master(index)
            .and_then(|master| handle.repack_from_host(backend, master))
        {
            Ok(()) => {}
            Err(error) => {
                failure = Some((index, error));
                break;
            }
        }
    }
    if let Some((index, error)) = failure {
        for handle in packed.iter_mut() {
            handle.mark_stale();
        }
        panic!("packed repack failed at parameter {index}; all handles invalidated: {error}");
    }
    assert!(
        packed.iter().all(DevicePackedSaltWeight::is_prepared),
        "successful repack must prepare every packed handle"
    );
}

#[test]
#[ignore = "OPEN: 40-step packed+HostOffload passes recovery but misses strict 1123ms gate"]
fn packed_host_offload_recovers_heldout_within_track0_time() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let steps = std::env::var("TRITIUM_DISTILL_STEPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(STEPS);
    assert!(steps > 0, "distillation needs at least one step");

    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, fp, shapes) = extract(&runner);
    let (mut train_ids, eval_ids) = corpus();
    if let Some(limit) = std::env::var("TRITIUM_TRAIN_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
    {
        train_ids.truncate(limit);
    }
    let windows: Vec<&[u32]> = train_ids.chunks_exact(TRAIN_SEQ).collect();
    assert!(!windows.is_empty(), "train corpus shorter than one window");

    let backend = CudaBackend::new(0).expect("open CUDA device");
    let ppl_fp = perplexity(&logits_of(&fp, &arch, &eval_ids), &eval_ids, arch.vocab);
    let ptq: Vec<Vec<f32>> = fp
        .iter()
        .zip(&shapes)
        .map(|(master, &(rows, cols))| ste::salt_quantize_forward(master, rows, cols, PLANES))
        .collect();
    let dense_parameter_bytes: usize = ptq
        .iter()
        .map(|weight| weight.len() * size_of::<f32>())
        .sum();
    let ppl_ptq = perplexity(&logits_of(&ptq, &arch, &eval_ids), &eval_ids, arch.vocab);
    drop(ptq);

    let optimizer = AdamW::new(LR);
    let specs: Vec<_> = fp
        .iter()
        .zip(&shapes)
        .map(|(master, &(rows, cols))| DeviceTrainParam {
            master,
            rows,
            cols,
            salt_planes: PLANES,
            optimizer,
        })
        .collect();
    let mut trainer = HostOffloadTrainer::new(&backend, &specs).expect("host-offload trainer");
    let mut packed: Vec<DevicePackedSaltWeight> = (0..trainer.len())
        .map(|index| {
            let metadata = trainer
                .parameter_metadata(index)
                .expect("parameter metadata");
            DevicePackedSaltWeight::from_host(
                &backend,
                trainer.master(index).expect("initial master"),
                metadata.rows,
                metadata.cols,
                metadata.salt_planes,
            )
            .expect("initial packed SALT weight")
        })
        .collect();
    assert!(packed.iter().all(DevicePackedSaltWeight::is_prepared));
    let packed_code_bytes: usize = packed
        .iter()
        .map(DevicePackedSaltWeight::packed_bytes)
        .sum();
    let packed_scale_bytes: usize = packed.iter().map(DevicePackedSaltWeight::scale_bytes).sum();
    let packed_parameter_bytes: usize = packed
        .iter()
        .map(DevicePackedSaltWeight::resident_bytes)
        .sum();
    assert_eq!(
        packed_parameter_bytes,
        packed_code_bytes + packed_scale_bytes
    );

    let mut teacher_cache = std::env::var_os("TRITIUM_TEACHER_CACHE").map(|path| {
        let expected = TeacherCacheHeader {
            seq_len: TRAIN_SEQ as u32,
            vocab: arch.vocab as u32,
            windows: windows.len() as u64,
            model_hash: hash_teacher_weights(fp.iter().map(Vec::as_slice)),
            corpus_hash: hash_teacher_corpus(&train_ids, TRAIN_SEQ as u32),
        };
        TeacherCacheReader::open(path, &expected).expect("open matching teacher cache")
    });
    let mut cached_target = vec![0.0; TRAIN_SEQ * arch.vocab];
    let mut elapsed_ms = 0.0;
    let mut max_activation_peak = 0usize;
    let mut max_naive_activations = 0usize;
    let mut total_recomputed_ops = 0usize;

    for step in 1..=steps {
        let started = Instant::now();
        let window_index = ((step - 1) as usize) % windows.len();
        let tokens = windows[window_index];
        let tokens_i32: Vec<i32> = tokens.iter().map(|&token| token as i32).collect();
        let target = if let Some(cache) = teacher_cache.as_mut() {
            cache
                .read_window(window_index as u64, &mut cached_target)
                .expect("read cached teacher window");
            DeviceTensor::upload(&backend, &cached_target).expect("upload cached target")
        } else {
            let probabilities = {
                let mut teacher = DeviceTape::new(&backend, arch.vocab).expect("teacher tape");
                let (logits, _) = device_forward(&mut teacher, &arch, &fp, &tokens_i32, TRAIN_SEQ);
                row_softmax(
                    &teacher.value(logits).expect("download teacher logits"),
                    arch.vocab,
                )
            };
            DeviceTensor::upload(&backend, &probabilities).expect("upload online teacher target")
        };

        let (gradients, backward_stats) = {
            let mut tape = DeviceTape::new_with_checkpoint_policy(
                &backend,
                arch.vocab,
                CheckpointPolicy::SqrtDepth(arch.n_layers),
            )
            .expect("packed checkpointed tape");
            let (logits, master_ids) =
                device_forward_packed(&mut tape, &arch, &packed, &tokens_i32, TRAIN_SEQ);
            let gradients = tape
                .xent_backward_device(logits, &target, TRAIN_SEQ, arch.vocab, &master_ids)
                .expect("packed student backward");
            let stats = gradients.backward_stats();
            (gradients, stats)
        };
        assert!(
            backward_stats.recomputed_ops > 0,
            "sqrt-depth checkpoint replay did not run at step {step}"
        );
        assert!(
            backward_stats.peak_live_activation_elements < backward_stats.naive_activation_elements,
            "checkpointing did not reduce activation residency at step {step}: {backward_stats:?}"
        );
        max_activation_peak = max_activation_peak.max(backward_stats.peak_live_activation_elements);
        max_naive_activations = max_naive_activations.max(backward_stats.naive_activation_elements);
        total_recomputed_ops = total_recomputed_ops
            .checked_add(backward_stats.recomputed_ops)
            .expect("recomputed-op count overflow");

        for handle in &mut packed {
            handle.mark_stale();
        }
        assert!(
            packed.iter().all(|handle| !handle.is_prepared()),
            "every packed handle must be stale before the master update"
        );
        if let Err(error) = trainer.step(gradients, step) {
            assert!(packed.iter().all(|handle| !handle.is_prepared()));
            panic!("host-offload step {step} failed with all packed handles stale: {error}");
        }
        repack_all_or_stale(&backend, &mut packed, &trainer);

        elapsed_ms += started.elapsed().as_secs_f64() * 1e3;
        if step == 1 || step % 10 == 0 || step == steps {
            eprintln!("  packed+offload step {step}/{steps} (window {window_index})");
        }
    }

    let distilled: Vec<Vec<f32>> = shapes
        .iter()
        .enumerate()
        .map(|(index, &(rows, cols))| {
            ste::salt_quantize_forward(
                trainer.master(index).expect("trained host master"),
                rows,
                cols,
                PLANES,
            )
        })
        .collect();
    let ppl_distilled = perplexity(
        &logits_of(&distilled, &arch, &eval_ids),
        &eval_ids,
        arch.vocab,
    );
    let recovery = ppl_ptq / ppl_distilled;
    let mean_step_ms = elapsed_ms / steps as f64;
    let packed_ratio = packed_parameter_bytes as f64 / dense_parameter_bytes as f64;
    let activation_ratio = max_activation_peak as f64 / max_naive_activations as f64;
    let offload = trainer.stats();
    let host_optimizer_bytes = offload.host_optimizer_elements * size_of::<f32>();
    let peak_optimizer_device_bytes = offload.peak_optimizer_device_elements * size_of::<f32>();
    let resident_gradient_bytes = offload.resident_input_gradient_elements * size_of::<f32>();
    let teacher_source = if teacher_cache.is_some() {
        "offline cache"
    } else {
        "online device teacher"
    };

    println!(
        "packed+HostOffload held-out SmolLM2-135M (T={PLANES}, {steps} steps, {} train tokens / \
         {} windows, {} eval tokens): fp ppl={ppl_fp:.3}, PTQ ppl={ppl_ptq:.3e}, distilled \
         ppl={ppl_distilled:.3}, recovery={recovery:.0}x; mean step={mean_step_ms:.0}ms vs \
         Track-0 {TRACK0_STEP_MS:.0}ms ({teacher_source}); packed bytes={packed_parameter_bytes} \
         (codes={packed_code_bytes}, scales={packed_scale_bytes}) vs dense \
         {dense_parameter_bytes} ({packed_ratio:.4}x); checkpoint peak={max_activation_peak} vs \
         naive={max_naive_activations} ({activation_ratio:.4}x), recomputed_ops total=\
         {total_recomputed_ops}; offloaded optimizer host bytes={host_optimizer_bytes}, peak device \
         staging bytes={peak_optimizer_device_bytes}, resident gradient bytes={resident_gradient_bytes}",
        train_ids.len(),
        windows.len(),
        eval_ids.len(),
    );

    assert_eq!(
        offload.peak_optimizer_device_elements,
        offload.largest_parameter_elements * 3,
        "optimizer staging must remain bounded by one master plus two moments"
    );
    assert!(
        offload.peak_optimizer_device_elements < offload.host_optimizer_elements,
        "optimizer staging must be leaf-bounded, not model-bounded: {offload:?}"
    );
    assert_eq!(
        resident_gradient_bytes, dense_parameter_bytes,
        "requesting every master id must return one full-model gradient collection"
    );
    assert!(ppl_ptq > ppl_fp, "PTQ must degrade held-out perplexity");
    assert!(
        recovery >= MIN_RECOVERY,
        "packed+offload recovered only {recovery:.0}x vs PTQ; gate requires {MIN_RECOVERY:.0}x"
    );
    assert!(
        mean_step_ms < TRACK0_STEP_MS,
        "packed+offload mean step {mean_step_ms:.0}ms did not beat Track-0 {TRACK0_STEP_MS:.0}ms"
    );
}
