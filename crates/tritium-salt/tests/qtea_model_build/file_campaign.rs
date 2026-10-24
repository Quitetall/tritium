use super::*;
use std::path::{Path, PathBuf};
use tritium_quantize::engine::{CalibrationCacheBuilder, DirectoryTensorSink};
struct Fixture {
    root: PathBuf,
    request: QteaFileCampaignRequest,
    files: QteaCampaignFiles,
    #[cfg(feature = "host-qtea")]
    original: InMemoryWeightSource,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.root).unwrap();
    }
}
fn safetensors(source: &InMemoryWeightSource) -> Vec<u8> {
    let mut header = BTreeMap::new();
    let mut raw = Vec::new();
    for (name, t) in &source.tensors {
        let start = raw.len();
        raw.extend_from_slice(&t.bytes);
        let dtype = match t.precision {
            ScalePrecision::F32 => "F32",
            ScalePrecision::F16 => "F16",
            ScalePrecision::Bf16 => "BF16",
            _ => unreachable!(),
        };
        header.insert(
            name,
            serde_json::json!({"dtype":dtype,"shape":t.shape,"data_offsets":[start,raw.len()]}),
        );
    }
    let mut h = serde_json::to_vec(&header).unwrap();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(raw);
    out
}
fn fixture(hybrid: bool, tied: bool, precision: ScalePrecision) -> Fixture {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let root = std::env::temp_dir().join(format!(
        "qtea-file-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    ));
    std::fs::create_dir(&root).unwrap();
    let mut original = source(hybrid, tied, precision);
    let plan = observation_plan(&original, hybrid);
    let model = ObservedWeightSource::new(&plan, &mut original)
        .unwrap()
        .source_id();
    let shard = root.join("original.safetensors");
    std::fs::write(&shard, safetensors(&original)).unwrap();
    let assets = original
        .assets
        .iter()
        .map(|(n, b)| {
            let p = root.join(n);
            std::fs::write(&p, b).unwrap();
            (n.clone(), p)
        })
        .collect();
    let budget = ResourceVector::new(1 << 24, 0, 1 << 24, 1 << 22, 0, 1 << 24, 10000, None);
    let profile = FrontierProfile::new(
        FrontierProfileId::new("test.qtea-file").unwrap(),
        FrontierOrdering::Fixed,
        vec![SolverId::new(FRONTIER_QTEA_REFERENCE_SOLVER_ID).unwrap()],
        SolverTrust::Experimental,
        budget,
        None,
        false,
    )
    .unwrap();
    let mut curvature = BTreeMap::new();
    let mut owners = Vec::new();
    for (index, t) in plan
        .tensors
        .iter()
        .filter(|t| t.source.shape.len() == 2)
        .enumerate()
    {
        let tensor = &t.source;
        let n = tensor.shape[1] as usize;
        let path = root.join(format!("curvature-{index}"));
        let mode = if tensor.name.ends_with("embed_tokens.weight") {
            let g = gram(n, model);
            std::fs::write(&path, g.canonical_bytes()).unwrap();
            QteaCampaignCurvature::Native {
                digest: g.digest(),
                max_value_bytes: n * n * 8,
            }
        } else {
            let limits = CalibrationCacheLimits {
                max_payload_bytes: 1 << 20,
                max_samples: 128,
                max_windows: 4,
            };
            let mut b = CalibrationCacheBuilder::new(
                CalibrationCacheRequest {
                    tensor: tensor.clone(),
                    model,
                    consumer: CalibrationConsumer::Matmul,
                    output: CalibrationOutput::None,
                    sample_start: 0,
                    limits,
                },
                b"synthetic tokenizer context",
                b"synthetic objective, not a quality dataset",
            )
            .unwrap();
            let mut x = vec![0.; n * n];
            for c in 0..n {
                x[c * n + c] = 1.;
            }
            b.capture(
                &(0..n as u32).collect::<Vec<_>>(),
                CalibrationBatch {
                    samples: n,
                    input: CalibrationInput::Activations(&x),
                    output: CalibrationOutputBatch::None,
                    weights: None,
                    mask: None,
                },
            )
            .unwrap();
            let cache = b.finish().unwrap();
            std::fs::write(&path, cache.encode_package(1 << 20).unwrap()).unwrap();
            QteaCampaignCurvature::CalibrationCache {
                id: cache.id(),
                limits,
                max_numeric_bytes: 1 << 20,
                max_batch_samples: 3,
            }
        };
        curvature.insert(tensor.name.clone(), path);
        owners.push(QteaCampaignOwner {
            name: tensor.name.clone(),
            recipe: fit_config(),
            curvature: mode,
            profile: profile.clone(),
            estimate: QteaCallerEstimate {
                resources: budget,
                machine_id: ContentId::of_bytes(b"synthetic machine assertion"),
                evidence_id: ContentId::of_bytes(b"synthetic estimate assertion"),
            },
        });
    }
    let l = limits();
    Fixture {
        root,
        request: QteaFileCampaignRequest {
            schema: "tritium.qtea.file-campaign.v1".into(),
            campaign: campaign(),
            run: ContentId::of_bytes(b"file campaign synthetic run"),
            source: plan,
            model_profile: if hybrid {
                QteaCampaignModelProfile::Qwen35Hybrid
            } else {
                QteaCampaignModelProfile::Standard
            },
            limits: QteaCampaignLimits {
                max_package_bytes: l.receiving.max_source_bytes,
                max_manifest_bytes: l.receiving.max_manifest_bytes,
                max_directory_bytes: l.receiving.max_directory_bytes,
                max_tensors: l.model.max_tensors,
                max_assets: l.model.max_assets,
                max_asset_bytes: l.model.max_asset_bytes,
                max_config_bytes: l.model.max_config_bytes,
                max_vector_elements: l.model.max_vector_elements,
                max_preserved_tensor_elements: l.max_preserved_tensor_elements,
                max_tensor_bytes: l.model.tensor.max_tensor_bytes,
                max_validation_bytes: l.model.tensor.max_validation_bytes,
                max_widened_tensor_bytes: l.max_widened_tensor_bytes,
                max_native_work_bytes: l.max_native_work_bytes,
                max_tensor_work_bytes: l.max_tensor_work_bytes,
                max_spool_bytes: l.max_spool_bytes,
                max_fit_record_bytes: l.max_fit_record_bytes,
                max_curvature_file_bytes: 1 << 20,
            },
            owners,
        },
        files: QteaCampaignFiles {
            source_shards: vec![shard],
            assets,
            curvature,
        },
        #[cfg(feature = "host-qtea")]
        original,
    }
}
fn run(
    f: &Fixture,
    selected: &QteaCampaignInspection,
    out: &Path,
) -> Result<QteaFileCampaignResult, QteaModelBuildError> {
    run_qtea_file_campaign(
        &f.request,
        &f.files,
        selected,
        Cursor::new(vec![]),
        &mut DirectoryTensorSink::new(out),
    )
}
#[test]
fn actual_file_inspection_replay_publication_and_cold_models() {
    for (hybrid, tied, precision) in [
        (false, true, ScalePrecision::F32),
        (false, false, ScalePrecision::F16),
        (true, false, ScalePrecision::Bf16),
    ] {
        let f = fixture(hybrid, tied, precision);
        let request =
            QteaFileCampaignRequest::from_slice(&serde_json::to_vec(&f.request).unwrap(), 1 << 20)
                .unwrap();
        let inspected = inspect_qtea_file_campaign(&request, &f.files).unwrap();
        assert_eq!(inspected.owners(), request.owners.len());
        let encoded = inspected.to_json(1 << 20).unwrap();
        assert!(inspected.to_json(1).is_err());
        let selected = QteaCampaignInspection::from_slice(&encoded, encoded.len()).unwrap();
        assert_eq!(
            inspect_qtea_file_campaign(&request, &f.files)
                .unwrap()
                .to_json(1 << 20)
                .unwrap(),
            encoded
        );
        let out = f.root.join("candidate");
        assert!(!out.exists());
        let result = run(&f, &selected, &out).unwrap();
        assert!(
            !serde_json::to_value(&result).unwrap()["qualification"]
                .as_bool()
                .unwrap()
        );
        assert!(result.publication_warnings().is_empty());
        assert!(result.to_json(1).is_err());
        let bytes = std::fs::read(out.join("model.trit")).unwrap();
        assert_eq!(
            *tritium_format::PackageId::from_package_bytes(&bytes).as_bytes(),
            result.package_id()
        );
        assert!(!out.join("evidence.jsonl").exists());
        let mut model = reader(&bytes, hybrid, selected.source_model());
        assert_eq!(
            verify_qtea_model_build_record(&mut model, 1 << 20)
                .unwrap()
                .fitted_matrices(),
            selected.owners()
        );
        assert!(run(&f, &selected, &out).is_err());
        #[cfg(feature = "host-qtea")]
        cold_and_export(
            &bytes,
            hybrid,
            tied,
            selected.source_model(),
            &f.original,
            None,
        );
        if let Some(root) = std::env::var_os("TRITIUM_QTEA_FILE_FIXTURE_OUTPUT") {
            let dest = PathBuf::from(root).join(if hybrid {
                "hybrid"
            } else if tied {
                "tied"
            } else {
                "untied"
            });
            std::fs::create_dir(&dest).unwrap();
            std::fs::write(
                dest.join("request.json"),
                serde_json::to_vec(&request).unwrap(),
            )
            .unwrap();
            std::fs::write(dest.join("inspection.json"), &encoded).unwrap();
            std::fs::write(dest.join("model.trit"), &bytes).unwrap();
            // Export explicit durable synthetic inputs, not paths to deleted scratch.
            let inputs = dest.join("inputs");
            std::fs::create_dir(&inputs).unwrap();
            for p in f
                .files
                .source_shards
                .iter()
                .chain(f.files.assets.values())
                .chain(f.files.curvature.values())
            {
                std::fs::copy(p, inputs.join(p.file_name().unwrap())).unwrap();
            }
            let relocated = QteaCampaignFiles {
                source_shards: f
                    .files
                    .source_shards
                    .iter()
                    .map(|p| inputs.join(p.file_name().unwrap()))
                    .collect(),
                assets: f
                    .files
                    .assets
                    .iter()
                    .map(|(n, p)| (n.clone(), inputs.join(p.file_name().unwrap())))
                    .collect(),
                curvature: f
                    .files
                    .curvature
                    .iter()
                    .map(|(n, p)| (n.clone(), inputs.join(p.file_name().unwrap())))
                    .collect(),
            };
            std::fs::write(
                dest.join("paths.json"),
                serde_json::to_vec(&relocated).unwrap(),
            )
            .unwrap();
        }
    }
}
#[test]
fn stale_context_and_selected_report_refuse_without_destination() {
    for fault in 0..9 {
        let mut f = fixture(false, true, ScalePrecision::F32);
        let inspected = inspect_qtea_file_campaign(&f.request, &f.files).unwrap();
        let mut selected: serde_json::Value =
            serde_json::from_slice(&inspected.to_json(1 << 20).unwrap()).unwrap();
        match fault {
            0 => f.request.owners[0].recipe.salient_ratio = 0.,
            1 => {
                let p = &f.files.source_shards[0];
                let mut bytes = std::fs::read(p).unwrap();
                *bytes.last_mut().unwrap() ^= 1;
                std::fs::write(p, bytes).unwrap();
            }
            2 => {
                let p = f.files.curvature.values().next().unwrap();
                let mut bytes = std::fs::read(p).unwrap();
                *bytes.last_mut().unwrap() ^= 1;
                std::fs::write(p, bytes).unwrap();
            }
            3 => selected["qualification"] = true.into(),
            4 => {
                selected["owners"][0]["input_id"] =
                    serde_json::to_value(ContentId::of_bytes(b"substitution")).unwrap()
            }
            5 => {
                selected["owners"][0]["plan_id"] =
                    serde_json::to_value(ContentId::of_bytes(b"other machine estimate")).unwrap()
            }
            6 => {
                selected["owners"].as_array_mut().unwrap().pop();
            }
            7 => selected["owners"][0]["logical_work_bytes"] = 1.into(),
            _ => {
                f.files
                    .curvature
                    .insert("extra".into(), f.root.join("absent"));
            }
        }
        let out = f.root.join("must-not-publish");
        match QteaCampaignInspection::from_slice(&serde_json::to_vec(&selected).unwrap(), 1 << 20) {
            Ok(s) => assert!(run(&f, &s, &out).is_err()),
            Err(_) => assert_eq!(fault, 3),
        }
        assert!(!out.exists());
    }
}
#[test]
fn strict_request_caps_inventory_policy_and_estimate_refusals() {
    for fault in 0..8 {
        let mut f = fixture(false, true, ScalePrecision::F32);
        match fault {
            0 => f.request.limits.max_curvature_file_bytes = 1,
            1 => {
                f.request.owners.pop();
            }
            2 => f.request.owners.swap(0, 1),
            3 => {
                f.request.owners[0].estimate.resources =
                    ResourceVector::new(0, 0, 0, 0, 0, 0, 0, None)
            }
            4 => f.request.owners[0].estimate.machine_id = ContentId::from_digest([0; 32]),
            5 => f.request.limits.max_native_work_bytes = 1,
            6 => f.request.schema = "unknown".into(),
            _ => {
                let p = &f.request.owners[0].profile;
                f.request.owners[0].profile = FrontierProfile::new(
                    p.id().clone(),
                    FrontierOrdering::Search,
                    p.solver_ids().to_vec(),
                    p.minimum_trust(),
                    p.budget(),
                    None,
                    false,
                )
                .unwrap();
            }
        }
        assert!(inspect_qtea_file_campaign(&f.request, &f.files).is_err());
    }
    let f = fixture(false, true, ScalePrecision::F32);
    let bytes = serde_json::to_vec(&f.request).unwrap();
    assert!(QteaFileCampaignRequest::from_slice(&bytes, bytes.len() - 1).is_err());
    let mut v = serde_json::to_value(&f.request).unwrap();
    v["unexpected"] = true.into();
    assert!(
        QteaFileCampaignRequest::from_slice(&serde_json::to_vec(&v).unwrap(), 1 << 20).is_err()
    );
}

#[test]
fn file_adapter_cleans_owned_spools_preserves_existing_paths_and_refuses_nonregular_inputs() {
    let mut f = fixture(false, true, ScalePrecision::F32);
    let selected = inspect_qtea_file_campaign(&f.request, &f.files).unwrap();
    let spool = f.root.join("private-spool");
    let out = f.root.join("file-candidate");
    let result =
        run_qtea_file_campaign_to_directory(&f.request, &f.files, &selected, &spool, &out).unwrap();
    assert!(result.publication_warnings().is_empty());
    assert!(!spool.exists());
    assert!(
        run_qtea_file_campaign_to_directory(&f.request, &f.files, &selected, &spool, &out).is_err()
    );
    assert!(!spool.exists());
    std::fs::write(&spool, b"foreign spool").unwrap();
    assert!(
        run_qtea_file_campaign_to_directory(
            &f.request,
            &f.files,
            &selected,
            &spool,
            &f.root.join("other")
        )
        .is_err()
    );
    assert_eq!(std::fs::read(&spool).unwrap(), b"foreign spool");
    let failed_spool = f.root.join("failed-spool");
    f.request.owners[0].recipe.salient_ratio = 0.;
    assert!(
        run_qtea_file_campaign_to_directory(
            &f.request,
            &f.files,
            &selected,
            &failed_spool,
            &f.root.join("failed")
        )
        .is_err()
    );
    assert!(!failed_spool.exists());
    f.request.owners[0].recipe = fit_config();
    let late = f.files.curvature.values().last().unwrap();
    let mut bytes = std::fs::read(late).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    std::fs::write(late, bytes).unwrap();
    assert!(
        run_qtea_file_campaign_to_directory(
            &f.request,
            &f.files,
            &selected,
            &failed_spool,
            &f.root.join("late-failure")
        )
        .is_err()
    );
    assert!(!failed_spool.exists() && !f.root.join("late-failure").exists());
    let name = f.request.owners[0].name.clone();
    f.files.curvature.insert(name.clone(), f.root.clone());
    assert!(inspect_qtea_file_campaign(&f.request, &f.files).is_err());
    #[cfg(unix)]
    {
        let fifo = f.root.join("fifo");
        assert!(
            std::process::Command::new("mkfifo")
                .arg(&fifo)
                .status()
                .unwrap()
                .success()
        );
        f.files.curvature.insert(name, fifo);
        assert!(inspect_qtea_file_campaign(&f.request, &f.files).is_err());
    }
}

#[test]
fn ambiguous_path_maps_and_nested_recipe_fields_are_not_silently_accepted() {
    let duplicate = br#"{"source_shards":["a"],"assets":{"x":"a","x":"b"},"curvature":{}}"#;
    assert!(serde_json::from_slice::<QteaCampaignFiles>(duplicate).is_err());
    let f = fixture(false, true, ScalePrecision::F32);
    let mut v = serde_json::to_value(&f.request).unwrap();
    v["owners"][0]["recipe"]["guess"] = true.into();
    assert!(
        QteaFileCampaignRequest::from_slice(&serde_json::to_vec(&v).unwrap(), 1 << 20).is_err()
    );
}

#[test]
fn source_preparation_needs_no_placeholder_curvature_or_run_and_checks_complete_graph() {
    for hybrid in [false, true] {
        let f = fixture(hybrid, false, ScalePrecision::F32);
        let request = QteaFileSourceRequest {
            schema: "tritium.qtea.source-preparation.v1".into(),
            campaign: f.request.campaign,
            source: f.request.source.clone(),
            model_profile: f.request.model_profile,
            limits: f.request.limits,
        };
        assert!(prepare_qtea_file_source(&request, &f.files).is_err());
        let mut files = f.files.clone();
        files.curvature.clear();
        let bytes = serde_json::to_vec(&request).unwrap();
        assert!(QteaFileSourceRequest::from_slice(&bytes, 1).is_err());
        let parsed = QteaFileSourceRequest::from_slice(&bytes, bytes.len()).unwrap();
        let observed = prepare_qtea_file_source(&parsed, &files).unwrap();
        assert_eq!(
            observed.source_model,
            inspect_qtea_file_campaign(&f.request, &f.files)
                .unwrap()
                .source_model()
        );
        assert_eq!(observed.tensors.len(), parsed.source.tensors.len());
        assert_eq!(observed.assets.len(), parsed.source.assets.len());
        let mut wrong = parsed.clone();
        wrong.model_profile = if hybrid {
            QteaCampaignModelProfile::Standard
        } else {
            QteaCampaignModelProfile::Qwen35Hybrid
        };
        assert!(prepare_qtea_file_source(&wrong, &files).is_err());
    }
}
