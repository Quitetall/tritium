//! Actual public-loader resident hybrid graph. Toy semantics, not model quality.
use super::*;
use tritium_nn::StreamLimits;
use tritium_spec::DecoderExecution;

fn cuda(limit: Option<usize>) -> Option<tritium_cuda::CudaBackend> {
    match tritium_cuda::CudaBackend::new(0) {
        Ok(b) => Some(match limit {
            Some(n) => b.with_resident_decoder(n),
            None => b,
        }),
        Err(e) if std::env::var("TRITIUM_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("required CUDA: {e}")
        }
        Err(e) => {
            eprintln!("SKIP resident hybrid: {e}");
            None
        }
    }
}
fn bounded(
    bytes: &[u8],
    backend: Box<dyn TernaryBackend>,
    budget: &TensorPayloadBudget,
) -> Result<tritium_nn::LoadedModel, tritium_nn::NnError> {
    tritium_nn::load_with_limits(
        ModelSource::Trit {
            bytes,
            level: "typed",
        },
        backend,
        TensorUploadPolicy::default(),
        budget,
        StreamLimits {
            max_context: 7,
            max_segment_tokens: 3,
            max_host_cache_payload_bytes: u64::MAX,
        },
        TensorUploadContext {
            recorder: &mut recorder(),
            span: "load",
            parent: None,
            logical_time: 0,
        },
    )
}
fn compare(got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len());
    assert!(expected.iter().any(|v| v.abs() > 1e-5), "zero oracle");
    for (&a, &b) in got.iter().zip(expected) {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= 2e-4 + 2e-4 * b.abs(),
            "resident={a}, host={b}"
        );
    }
}
fn package(profile: usize, mode: usize) -> Package {
    let (schedule, interval) = match mode {
        0 => (vec!["linear_attention", "full_attention"], 2),
        1 => (
            vec!["linear_attention", "linear_attention", "full_attention"],
            3,
        ),
        2 => (["linear_attention", "full_attention"].repeat(2), 2),
        3 => (
            vec![
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            4,
        ),
        _ => panic!("fixture schedule"),
    };
    // Reuse actual mixer-specific bindings, never fabricate QKV placeholders.
    let mut p = text_package(profile != 0, false);
    let original = std::mem::take(&mut p.manifest.levels[0].tensors);
    p.manifest.levels[0].tensors.extend(
        original
            .iter()
            .filter(|m| !m.name.starts_with("model.language_model.layers."))
            .cloned(),
    );
    for (at, kind) in schedule.iter().enumerate() {
        let source = if *kind == "linear_attention" { 0 } else { 1 };
        let prefix = format!("model.language_model.layers.{source}.");
        for old in original.iter().filter(|m| m.name.starts_with(&prefix)) {
            let mut meta = old.clone();
            meta.name =
                meta.name
                    .replacen(&prefix, &format!("model.language_model.layers.{at}."), 1);
            p.manifest.levels[0].tensors.push(meta);
        }
    }
    let mut config: serde_json::Value = serde_json::from_str(&config_json()).unwrap();
    config["text_config"]["full_attention_interval"] = interval.into();
    config["text_config"]["num_hidden_layers"] = (schedule.len() as u64).into();
    config["text_config"]["layer_types"] = serde_json::json!(schedule);
    asset(&mut p, "config.json", serde_json::to_vec(&config).unwrap());
    if profile >= 2 {
        let bytes = p.bytes();
        let model = tritium_format::trit_model::read_trit_model(&bytes).unwrap();
        for meta in &mut p.manifest.levels[0].tensors {
            let TensorStorage::Additive { layout, planes } = &mut meta.storage else {
                continue;
            };
            let (old, payload) = model.tensor("typed", &meta.name).unwrap();
            let decoded =
                tritium_format::trit_model::decode_additive_payload(old, payload).unwrap();
            layout.codec = if profile == 2 || profile == 4 {
                PlaneCodec::B3
            } else {
                PlaneCodec::D2
            };
            layout.basis = if profile == 2 {
                Basis::Hadamard { block: 2 }
            } else {
                Basis::SignedRht {
                    block: 2,
                    seed: 7,
                    domain: 9,
                }
            };
            let tensor = if profile == 4 {
                layout.allocation = PlaneAllocation::PerTile;
                let rows = layout.rows as usize;
                let cols = layout.cols as usize;
                assert!(cols < layout.group as usize && cols < layout.tile as usize);
                let mut trits = Vec::new();
                let mut scales = Vec::new();
                let counts: Vec<u8> = (0..rows).map(|r| (r % 4) as u8).collect();
                for (r, &count) in counts.iter().enumerate() {
                    for p in 0..count as usize {
                        trits.extend_from_slice(
                            &decoded.trits()[(p * rows + r) * cols..(p * rows + r + 1) * cols],
                        );
                        scales.push(decoded.scales()[p * rows + r]);
                    }
                }
                *planes = PlaneCounts::PerTile(counts.clone());
                AdditiveTensor::new_per_tile(*layout, counts, trits, scales).unwrap()
            } else {
                AdditiveTensor::new(
                    *layout,
                    3,
                    decoded.trits().to_vec(),
                    decoded.scales().to_vec(),
                )
                .unwrap()
            };
            let payload = tensor.encode_payload().unwrap();
            meta.blob = BlobReference {
                id: BlobId::from_bytes(*blake3::hash(&payload).as_bytes()),
                shard: None,
            };
            // Repeated layer handles share a blob; the writer accepts one copy.
            if !p.blobs.iter().any(|(_, b)| b == &payload) {
                p.blobs.push((TritBlobKind::AdditiveTensor, payload));
            }
        }
    }
    p
}

#[test]
fn whole_hybrid_mixed_schedules_dense_packed_rotated_and_per_tile_match_host() {
    let Some(probe) = cuda(None) else {
        return;
    };
    assert!(!probe.resident_decoder_requested());
    drop(probe);
    let mut comparisons = 0;
    for profile in 0..5 {
        for mode in 0..4 {
            let bytes = package(profile, mode).bytes();
            let cpu_budget = TensorPayloadBudget::new(1_000_000);
            let gpu_budget = TensorPayloadBudget::new(1_000_000);
            let mut cpu = bounded(
                &bytes,
                Box::new(tritium_cpu::CpuBackend::new()),
                &cpu_budget,
            )
            .unwrap();
            let mut gpu = bounded(
                &bytes,
                Box::new(cuda(Some(1_000_000)).unwrap()),
                &gpu_budget,
            )
            .unwrap();
            assert_eq!(gpu.runner.execution(), DecoderExecution::ResidentDeviceF32);
            assert_eq!(cpu.runner.execution(), DecoderExecution::Host);
            assert_eq!(gpu.runner.mtp_status(), Some(Qwen35MtpStatus::Unverified));
            assert_eq!(gpu.runner.mtp_status(), cpu.runner.mtp_status());
            for (tokens, positions) in [
                (&[1, 2, 4][..], &[0, 1, 2][..]),
                (&[5][..], &[3][..]),
                (&[6][..], &[4][..]),
                (&[1][..], &[5][..]),
                (&[2][..], &[6][..]),
            ] {
                compare(
                    &gpu.runner.forward(tokens, positions).unwrap(),
                    &cpu.runner.forward(tokens, positions).unwrap(),
                );
                comparisons += 1;
            }
            assert_eq!(gpu.runner.cache_len(), 7);
            assert!(gpu.runner.step(1).is_err());
            gpu.runner.reset();
            cpu.runner.reset();
            compare(
                &gpu.runner.prefill(&[2, 1]).unwrap(),
                &cpu.runner.prefill(&[2, 1]).unwrap(),
            );
            comparisons += 1;
            drop(cpu);
            drop(gpu);
            assert_eq!(gpu_budget.used_bytes(), 0);
            assert_eq!(cpu_budget.used_bytes(), 0);
        }
    }
    assert_eq!(comparisons, 120);
    eprintln!("whole resident hybrid comparisons={comparisons}, graphs=20");
}

#[test]
fn hybrid_all_mixer_cancellation_exact_retry_limits_and_payload_admission() {
    let Some(backend) = cuda(Some(1_000_000)) else {
        return;
    };
    let bytes = package(4, 2).bytes();
    let budget = TensorPayloadBudget::new(1_000_000);
    let mut gpu = bounded(&bytes, Box::new(backend), &budget).unwrap();
    let payload = gpu.runner.resident_payload_bytes().unwrap();
    let prefix = gpu.runner.prefill(&[1, 2]).unwrap();
    let baseline = gpu.runner.step(4).unwrap();
    let mut cancellations = 0;
    for cut in 1..=9 {
        gpu.runner.reset();
        assert_eq!(gpu.runner.prefill(&[1, 2]).unwrap(), prefix);
        let calls = std::cell::Cell::new(0);
        let result = gpu
            .runner
            .forward_cancellable(&[4], &[2], &|| {
                calls.set(calls.get() + 1);
                calls.get() >= cut
            })
            .unwrap();
        if let Some(logits) = result {
            assert_eq!(logits, baseline);
            assert_eq!(gpu.runner.cache_len(), 3);
        } else {
            cancellations += 1;
            assert_eq!(gpu.runner.cache_len(), 2);
            assert_eq!(gpu.runner.step(4).unwrap(), baseline);
        }
    }
    assert!(cancellations >= 7);
    gpu.runner.reset();
    gpu.runner.prefill(&[1, 2]).unwrap();
    for (tokens, pos) in [
        (&[4, 5][..], &[2, 3][..]),
        (&[4][..], &[3][..]),
        (&[7][..], &[2][..]),
        (&[][..], &[][..]),
    ] {
        assert!(gpu.runner.forward(tokens, pos).is_err());
        assert_eq!(gpu.runner.cache_len(), 2);
    }
    assert_eq!(gpu.runner.step(4).unwrap(), baseline);
    drop(gpu);
    assert_eq!(budget.used_bytes(), 0);
    assert!(bounded(&bytes, Box::new(cuda(Some(payload - 1)).unwrap()), &budget).is_err());
    assert_eq!(budget.used_bytes(), 0);
    let exact = bounded(&bytes, Box::new(cuda(Some(payload)).unwrap()), &budget).unwrap();
    assert_eq!(exact.runner.resident_payload_bytes(), Some(payload));
    drop(exact);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn hybrid_nonfinite_final_head_aborts_recurrence_and_attention_before_retry() {
    let Some(backend) = cuda(Some(1_000_000)) else {
        return;
    };
    let mut ts = tensors();
    for t in &mut ts {
        t.values.fill(0.);
        if t.name == "model.language_model.embed_tokens.weight" {
            t.values[4 * H..5 * H].fill(1.);
        }
        if t.name == "lm_head.weight" {
            t.values.fill(f32::MAX);
        }
    }
    let mut p = Package::new(ts, false);
    for (kind, bytes) in text_package(false, false).blobs {
        if let TritBlobKind::Asset(name) = kind
            && name != "config.json"
            && bytes != b"{}"
        {
            asset(&mut p, &name, bytes);
        }
    }
    let bytes = p.bytes();
    let budget = TensorPayloadBudget::new(1_000_000);
    let mut gpu = bounded(&bytes, Box::new(backend), &budget).unwrap();
    assert!(gpu.runner.prefill(&[1]).unwrap().iter().all(|&v| v == 0.));
    let baseline = gpu.runner.step(2).unwrap();
    gpu.runner.reset();
    gpu.runner.prefill(&[1]).unwrap();
    assert!(gpu.runner.step(4).is_err());
    assert_eq!(gpu.runner.cache_len(), 1);
    assert_eq!(gpu.runner.step(2).unwrap(), baseline);
    drop(gpu);
    assert_eq!(budget.used_bytes(), 0);
}
