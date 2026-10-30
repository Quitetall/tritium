use super::*;
use tritium_nn::{ModelLoadResources, StreamLimits, inspect_trit_file, inspect_trit_file_streamed};

#[test]
fn streamed_inspection_matches_actual_hybrid_language_mtp_resource_admission_without_upload() {
    for uniform in [false, true] {
        let root = Scratch::new(&format!("streamed-inspection-{uniform}"));
        let path = root.0.join("model.trit");
        let bytes = text_package(uniform, false).bytes();
        std::fs::write(&path, &bytes).unwrap();
        let stream = StreamLimits {
            max_context: 16,
            max_segment_tokens: 4,
            max_host_cache_payload_bytes: 1_000_000,
        };
        let ledgers: [TensorPayloadBudget; 5] =
            std::array::from_fn(|_| TensorPayloadBudget::new(1_000_000));
        let mut evidence = recorder();
        let mut planned_names = 0;
        let plan = inspect_trit_file_streamed(
            &path,
            "typed",
            receiving(),
            limits(),
            stream,
            TensorUploadPolicy::default(),
            |metadata, encoding| {
                planned_names += 1;
                tritium_cpu::CpuBackend.tensor_stream_plan(metadata, encoding)
            },
        )
        .unwrap();
        assert!(evidence.events().is_empty());
        assert!(ledgers.iter().all(|b| b.used_bytes() == 0));
        let config = Qwen35CheckpointConfig::from_hf_config(&config_json()).unwrap();
        let schema = qwen35_language_mtp_tensor_schema(&config.text).unwrap();
        assert_eq!(
            planned_names,
            schema
                .iter()
                .filter(|s| s.role == Qwen35TensorSchemaRole::Matrix)
                .count()
        );
        let reader = tritium_format::trit_model::read_trit_model(&bytes).unwrap();
        let preserved = schema
            .iter()
            .filter(|s| s.role != Qwen35TensorSchemaRole::Matrix)
            .map(|s| {
                let (meta, _) = reader.tensor("typed", &s.name).unwrap();
                let TensorStorage::Dense { shape, .. } = &meta.storage else {
                    panic!("preserved storage")
                };
                shape.iter().product::<u64>() * 4
            })
            .sum::<u64>();
        assert_eq!(
            plan.retained_host_bytes(),
            preserved
                + plan
                    .source_plan()
                    .cache_admission()
                    .unwrap()
                    .logical_host_cache_payload_bytes()
        );
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        let loaded = tritium_nn::load_streamed_with_resources(
            file_source(&path),
            Box::new(tritium_cpu::CpuBackend),
            TensorUploadPolicy::default(),
            ModelLoadResources {
                payload: &ledgers[0],
                source_workspace: &ledgers[1],
                host_workspace: &ledgers[2],
                device_workspace: &ledgers[3],
                retained_host: &ledgers[4],
            },
            stream,
            TensorUploadContext {
                recorder: &mut evidence,
                span: "load",
                parent: None,
                logical_time: 0,
            },
        )
        .unwrap();
        let admitted = loaded.runner.resource_admission().unwrap();
        assert_eq!(
            (
                plan.payload_bytes(),
                plan.source_workspace_bytes(),
                plan.host_workspace_bytes(),
                plan.device_workspace_bytes(),
                plan.retained_host_bytes()
            ),
            (
                admitted.payload_bytes,
                admitted.source_workspace_bytes,
                admitted.host_workspace_bytes,
                admitted.device_workspace_bytes,
                admitted.retained_host_bytes
            )
        );
        assert_eq!(
            plan.source_plan().source(),
            &loaded.receipt.observation().source
        );
        drop(loaded);
        assert!(ledgers.iter().all(|b| b.used_bytes() == 0));
    }
}

#[test]
fn streamed_inspection_refuses_unknown_geometry_caps_and_terminal_source_mutation() {
    let root = Scratch::new("streamed-inspection-refusals");
    let path = root.0.join("model.trit");
    let bytes = text_package(true, false).bytes();
    let stream = StreamLimits {
        max_context: 16,
        max_segment_tokens: 4,
        max_host_cache_payload_bytes: 1_000_000,
    };
    for case in 0..6 {
        std::fs::write(&path, &bytes).unwrap();
        let mut caps = limits();
        if case == 2 {
            caps.max_tensor_bytes = 1;
        }
        let mut mutated = false;
        let policy = TensorUploadPolicy {
            native_only: false,
            max_payload_bytes: (case == 5).then_some(1),
        };
        let error = inspect_trit_file_streamed(
            &path,
            "typed",
            receiving(),
            caps,
            stream,
            policy,
            |metadata, encoding| {
                if case == 0 {
                    return Ok(None);
                }
                let mut plan = tritium_cpu::CpuBackend
                    .tensor_stream_plan(metadata, encoding)?
                    .unwrap();
                if case == 1 {
                    plan.encoded_bytes += 1;
                }
                if case == 3 && !mutated {
                    let mut file = OpenOptions::new()
                        .read(true)
                        .write(true)
                        .open(&path)
                        .unwrap();
                    file.seek(SeekFrom::End(-1)).unwrap();
                    file.write_all(&[bytes[bytes.len() - 1] ^ 1]).unwrap();
                    file.flush().unwrap();
                    mutated = true;
                }
                if case == 4 {
                    plan.upload.caps.payload_bytes = u64::MAX;
                }
                Ok(Some(plan))
            },
        )
        .unwrap_err()
        .to_string();
        assert!(!error.is_empty());
        if case == 0 {
            assert!(error.contains("plan is unknown"), "{error}");
        }
        if case == 1 {
            assert!(error.contains("encoded geometry mismatch"), "{error}");
        }
        if case == 4 {
            assert!(error.contains("overflow"), "{error}");
        }
    }
}

#[test]
fn shared_inspection_matches_actual_metadata_and_public_load_without_upload() {
    for uniform in [false, true] {
        let root = Scratch::new(&format!("inspection-{uniform}"));
        let path = root.0.join("model.trit");
        let bytes = text_package(uniform, false).bytes();
        std::fs::write(&path, &bytes).unwrap();
        let stream = StreamLimits {
            max_context: 16,
            max_segment_tokens: 4,
            max_host_cache_payload_bytes: 1_000_000,
        };
        let budget = TensorPayloadBudget::new(1_000_000);
        let count = Arc::new(AtomicUsize::new(0));
        let mut evidence = recorder();
        let plan = inspect_trit_file(&path, "typed", receiving(), limits(), Some(stream)).unwrap();
        assert_eq!(budget.used_bytes(), 0);
        assert_eq!(count.load(Ordering::SeqCst), 0);
        assert!(evidence.events().is_empty());
        assert_eq!(plan.source_bytes(), bytes.len() as u64);
        assert_eq!(plan.architecture().name(), "qwen3_5");
        assert_eq!(plan.architecture().vocab_size(), 7);
        assert!(plan.backend_payload_bytes().is_none());
        assert_eq!(plan.cache_admission().unwrap().limits(), stream);
        let reader = tritium_format::trit_model::read_trit_model(&bytes).unwrap();
        let mut decoded_sum = 0;
        let mut encoded_sum = 0;
        let mut maximum = 0;
        for tensor in plan.tensors() {
            let (meta, encoded) = reader.tensor("typed", tensor.name()).unwrap();
            let decoded = match &meta.storage {
                TensorStorage::Dense { shape, .. } => shape.iter().product::<u64>() * 4,
                TensorStorage::Additive { .. } => {
                    let owned =
                        tritium_format::trit_model::decode_additive_payload(meta, encoded).unwrap();
                    let auxiliary = if owned.tile_counts().is_some() {
                        owned.view().auxiliary_buffer_len_bytes().unwrap()
                    } else {
                        0
                    };
                    (owned.trits().len() + owned.scales().len() * 4 + auxiliary) as u64
                }
                _ => panic!("fixture storage"),
            };
            assert_eq!(tensor.encoded_bytes(), encoded.len() as u64);
            assert_eq!(tensor.logical_decoded_bytes(), decoded);
            encoded_sum += encoded.len() as u64;
            decoded_sum += decoded;
            maximum = maximum.max(encoded.len() as u64 + decoded);
        }
        assert_eq!(plan.encoded_tensor_bytes_sum(), encoded_sum);
        assert_eq!(plan.logical_decoded_tensor_bytes_sum(), decoded_sum);
        assert_eq!(plan.maximum_tensor_working_payload_bytes(), maximum);
        let loaded = tritium_nn::load_with_limits(
            ModelSource::TritFile {
                path: &path,
                level: "typed",
                receiving: receiving(),
                limits: limits(),
            },
            Box::new(Probe {
                count: count.clone(),
                fail_at: None,
                mutate: None,
            }),
            TensorUploadPolicy::default(),
            &budget,
            stream,
            TensorUploadContext {
                recorder: &mut evidence,
                span: "load",
                parent: None,
                logical_time: 0,
            },
        )
        .unwrap();
        assert_eq!(plan.source(), &loaded.receipt.observation().source);
        assert_eq!(
            plan.architecture().layer_count(),
            loaded.architecture.layer_count()
        );
        assert_eq!(plan.cache_admission(), loaded.runner.cache_admission());
        assert!(count.load(Ordering::SeqCst) > 0);
        drop(loaded);
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn shared_inspection_refuses_the_same_preupload_source_faults() {
    let root = Scratch::new("inspection-refusals");
    let path = root.0.join("model.trit");
    for case in 0..8 {
        let mut package = text_package(false, case == 0);
        match case {
            1 => package.manifest.levels[0]
                .tensors
                .retain(|t| !t.name.starts_with("mtp.")),
            2 => package.manifest.architecture = "wrong".into(),
            _ => {}
        }
        let mut bytes = package.bytes();
        if case == 7 {
            let last = bytes.len() - 1;
            bytes[last] ^= 1;
        }
        std::fs::write(&path, bytes).unwrap();
        let mut caps = limits();
        match case {
            4 => caps.max_config_bytes = 1,
            5 => caps.max_tensor_bytes = 1,
            6 => caps.max_decoded_tensor_bytes = 1,
            _ => {}
        }
        let level = if case == 3 { "absent" } else { "typed" };
        assert!(
            inspect_trit_file(&path, level, receiving(), caps, None).is_err(),
            "case {case}"
        );
        let count = Arc::new(AtomicUsize::new(0));
        let budget = TensorPayloadBudget::new(1_000_000);
        let mut evidence = recorder();
        assert!(
            load(
                ModelSource::TritFile {
                    path: &path,
                    level,
                    receiving: receiving(),
                    limits: caps
                },
                Box::new(Probe {
                    count: count.clone(),
                    fail_at: None,
                    mutate: None
                }),
                &budget,
                &mut evidence,
            )
            .is_err(),
            "load case {case}"
        );
        assert_eq!(count.load(Ordering::SeqCst), 0);
        assert_eq!(budget.used_bytes(), 0);
        assert!(
            !evidence
                .events()
                .iter()
                .any(|event| event.schema == ModelLoaded::SCHEMA_ID)
        );
    }
}
