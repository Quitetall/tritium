use super::*;
use tritium_nn::{StreamLimits, inspect_trit_file};

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
