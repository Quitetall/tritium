//! Actual source/collector/admitted-fit/model stream; synthetic, not qualification.
use std::{
    collections::BTreeMap,
    io::{Cursor, Read, Seek, SeekFrom, Write},
    sync::Arc,
};
use tritium_format::{
    TritPackageReadLimits,
    qtea::{
        QteaLimits,
        model::{QteaHybridModelReadLimits, QteaModelReadLimits, QteaModelReader},
    },
};
use tritium_nn::{QteaModelInventory, QteaModelProfile};
use tritium_quantize::{
    CurvatureSourceId, InputGram, InputGramAccumulator,
    engine::{
        CalibratedTensor, CalibrationBatch, CalibrationCollector, CalibrationInput,
        CalibrationOutputBatch, InMemoryWeightSource, ObservedWeightSource, RawTensor,
        WeightSource,
    },
    qtea_reference::TensorFitConfig,
};
use tritium_salt::*;
use tritium_schema::*;

fn config(hybrid: bool, tied: bool) -> serde_json::Value {
    if !hybrid {
        return serde_json::json!({"model_type":"llama","hidden_size":8,"intermediate_size":12,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"max_position_embeddings":16,"rope_theta":10000.,"rms_norm_eps":1e-5,"hidden_act":"silu","tie_word_embeddings":tied,"vocab_size":8});
    }
    serde_json::json!({"architectures":["Qwen3_5ForConditionalGeneration"],"language_model_only":false,"model_type":"qwen3_5","tie_word_embeddings":false,"vision_config":{"model_type":"qwen3_5"},"text_config":{"attention_bias":false,"attention_dropout":0.0,"attn_output_gate":true,"dtype":"bfloat16","full_attention_interval":2,"head_dim":4,"hidden_act":"silu","hidden_size":8,"intermediate_size":12,"layer_types":["linear_attention","full_attention"],"linear_conv_kernel_dim":4,"linear_key_head_dim":2,"linear_num_key_heads":2,"linear_num_value_heads":4,"linear_value_head_dim":2,"mamba_ssm_dtype":"float32","max_position_embeddings":16,"model_type":"qwen3_5_text","mtp_num_hidden_layers":1,"mtp_use_dedicated_embeddings":false,"num_attention_heads":2,"num_hidden_layers":2,"num_key_value_heads":1,"output_gate_type":"swish","partial_rotary_factor":0.5,"rms_norm_eps":1e-6,"rope_parameters":{"mrope_interleaved":true,"mrope_section":[1,0,0],"partial_rotary_factor":0.5,"rope_theta":10000.0,"rope_type":"default"},"tie_word_embeddings":false,"use_cache":true,"vocab_size":8}})
}
fn source(hybrid: bool, tied: bool, precision: ScalePrecision) -> InMemoryWeightSource {
    let cfg = config(hybrid, tied);
    let shapes = if hybrid {
        tritium_nn::qwen35_language_mtp_tensor_schema(
            &tritium_nn::Qwen35CheckpointConfig::from_hf_config(&cfg.to_string())
                .unwrap()
                .text,
        )
        .unwrap()
        .into_iter()
        .map(|e| {
            (
                e.name,
                e.shape.into_iter().map(|d| d as u64).collect::<Vec<_>>(),
            )
        })
        .collect::<BTreeMap<_, _>>()
    } else {
        let mut shapes = BTreeMap::from([
            ("model.embed_tokens.weight".into(), vec![8, 8]),
            ("model.norm.weight".into(), vec![8]),
        ]);
        if !tied {
            shapes.insert("lm_head.weight".into(), vec![8, 8]);
        }
        for layer in 0..2 {
            let p = format!("model.layers.{layer}");
            for suffix in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
                shapes.insert(format!("{p}.{suffix}"), vec![8]);
            }
            for (suffix, r, c) in [
                ("self_attn.q_proj.weight", 8, 8),
                ("self_attn.k_proj.weight", 4, 8),
                ("self_attn.v_proj.weight", 4, 8),
                ("self_attn.o_proj.weight", 8, 8),
                ("mlp.gate_proj.weight", 12, 8),
                ("mlp.up_proj.weight", 12, 8),
                ("mlp.down_proj.weight", 8, 12),
            ] {
                shapes.insert(format!("{p}.{suffix}"), vec![r, c]);
            }
        }
        shapes
    };
    let mut source = InMemoryWeightSource::default();
    for (seed, (name, shape)) in shapes.into_iter().enumerate() {
        let count = shape.iter().product::<u64>() as usize;
        let values = (0..count)
            .map(|i| {
                if shape.len() == 2 {
                    let c = i % shape[1] as usize;
                    let r = i / shape[1] as usize;
                    ((((r + 1) * (c + 3) + seed) % 17) as f32 - 8.) / 64.
                        + c as f32 / 1024.
                        + r as f32 / 512.
                } else if !hybrid {
                    1.
                } else if shape.len() > 1 {
                    0.125
                } else if name.ends_with("A_log") {
                    -1.
                } else {
                    0.
                }
            })
            .collect::<Vec<_>>();
        let bytes = match precision {
            ScalePrecision::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ScalePrecision::F16 => values
                .iter()
                .flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes())
                .collect(),
            ScalePrecision::Bf16 => values
                .iter()
                .flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes())
                .collect(),
            _ => panic!("unsupported synthetic source precision"),
        };
        source.tensors.insert(
            name,
            RawTensor {
                shape,
                precision,
                bytes,
            },
        );
    }
    source
        .assets
        .insert("config.json".into(), serde_json::to_vec(&cfg).unwrap());
    source.assets.insert("tokenizer.json".into(),serde_json::to_vec(&serde_json::json!({"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"Testing":1,"partner":2,"[EOS]":3,"[BOS]":4,"user":5,"assistant":6,"reply":7},"unk_token":"[UNK]"}})).unwrap());
    source.assets.insert("tokenizer_config.json".into(),serde_json::to_vec(&serde_json::json!({"bos_token":"[BOS]","eos_token":"[EOS]","tokenizer_class":"PreTrainedTokenizerFast","chat_template":"{% for message in messages %}{% if message['role'] not in ['system','user','assistant','tool'] %}{{ raise_exception('invalid role') }}{% endif %}{% endfor %}{% if enable_thinking is defined and enable_thinking %}reply {% endif %}{{ messages[0]['content'] }} partner"})).unwrap());
    source
}
fn observation_plan(source: &dyn WeightSource, hybrid: bool) -> QuantPlan {
    QuantPlan {
        architecture: if hybrid { "qwen3_5" } else { "llama" }.into(),
        tensors: source
            .inventory()
            .unwrap()
            .into_iter()
            .map(|source| QuantTensorPlan {
                source,
                action: QuantAction::KeepDense,
            })
            .collect(),
        assets: source.asset_names(),
        levels: vec!["original-only".into()],
        limits: QuantLimits {
            max_source_bytes: 1 << 20,
            max_asset_bytes: 1 << 20,
            max_retained_blob_bytes: 1 << 20,
            max_package_bytes: 1 << 20,
        },
    }
}
fn limits() -> QteaModelBuildLimits {
    QteaModelBuildLimits {
        receiving: TritPackageReadLimits {
            max_source_bytes: 1 << 22,
            max_manifest_bytes: 1 << 16,
            max_directory_bytes: 1 << 16,
        },
        model: QteaModelReadLimits {
            max_tensors: 128,
            max_assets: 8,
            max_asset_bytes: 1 << 20,
            max_config_bytes: 4096,
            max_vector_elements: 32,
            tensor: QteaLimits {
                max_tensor_bytes: 1 << 16,
                max_validation_bytes: 1 << 16,
            },
        },
        max_preserved_tensor_elements: 64,
        max_widened_tensor_bytes: 1 << 16,
        max_native_work_bytes: 1 << 20,
        max_tensor_work_bytes: 1 << 22,
        max_spool_bytes: 1 << 22,
        max_fit_record_bytes: 1 << 20,
    }
}
fn campaign() -> ContentId {
    ContentId::of_bytes(b"synthetic cohort; not hardware/source authority")
}
fn fit_config() -> TensorFitConfig {
    TensorFitConfig {
        group_size: 4,
        salient_ratio: 0.5,
        ..Default::default()
    }
}
fn gram(cols: usize, model: OriginalSourceId) -> InputGram {
    let mut g = InputGramAccumulator::new_bound(
        cols,
        CurvatureSourceId::new(
            *model.as_bytes(),
            *ContentId::of_bytes(b"explicit synthetic native metric").as_bytes(),
            *ContentId::of_bytes(b"synthetic native sample IDs").as_bytes(),
        )
        .unwrap(),
        0,
    )
    .unwrap();
    let mut x = vec![0.; cols * cols];
    for c in 0..cols {
        x[c * cols + c] = 1.;
    }
    g.accumulate_batch(&x, cols, None, None).unwrap();
    g.finish().unwrap()
}
fn calibrated(original: &QuantSourceTensor, model: OriginalSourceId) -> CalibratedTensor {
    let cols = original.shape[1] as usize;
    let mut p = CalibrationPlan {
        tensor: original.clone(),
        source: CalibrationSource {
            model: FitInputDigest(*model.as_bytes()),
            cache: FitInputDigest(
                *ContentId::of_bytes(b"synthetic captured activation labels, not custody")
                    .as_bytes(),
            ),
            tokens: FitInputDigest(
                *ContentId::of_bytes(b"synthetic capture token labels").as_bytes(),
            ),
        },
        partition: CalibrationPartition::Full,
        consumer: CalibrationConsumer::Matmul,
        output: CalibrationOutput::None,
        sample_start: 0,
        damping: 0.,
        max_numeric_bytes: 1 << 20,
    };
    p.max_numeric_bytes = CalibrationCollector::required_numeric_bytes(&p).unwrap();
    let mut collector = CalibrationCollector::new(p).unwrap();
    let mut x = vec![0.; cols * cols];
    for c in 0..cols {
        x[c * cols + c] = 1.;
    }
    collector
        .accumulate(CalibrationBatch {
            samples: cols,
            input: CalibrationInput::Activations(&x),
            output: CalibrationOutputBatch::None,
            weights: None,
            mask: None,
        })
        .unwrap();
    collector.finish().unwrap()
}
#[derive(Debug)]
struct PlanningOnly {
    descriptor: SolverDescriptor,
    estimate: FrontierResourceEstimate,
}
impl FrontierSolver for PlanningOnly {
    fn descriptor(&self) -> &SolverDescriptor {
        &self.descriptor
    }
    fn estimate(&self, _: &SolverRequest) -> Result<FrontierResourceEstimate, FrontierSolverError> {
        Ok(self.estimate)
    }
}
fn select(work: &QteaTensorWorkPlan<'_>) -> QteaSelectedAdmission {
    let m = work.memory();
    let v = ResourceVector::new(
        m.total_logical_bytes as u64,
        0,
        0,
        work.byte_ledger().total_bytes as u64,
        0,
        (m.total_logical_bytes - m.borrowed_input_bytes) as u64,
        0,
        None,
    );
    let descriptor = QteaReferenceAdapter::new().unwrap().descriptor().clone();
    let id = descriptor.id().clone();
    let profile = FrontierProfile::new(
        FrontierProfileId::new("test.qtea-model-build").unwrap(),
        FrontierOrdering::Fixed,
        vec![id.clone()],
        SolverTrust::Experimental,
        v,
        None,
        false,
    )
    .unwrap();
    let mut registry = SolverRegistry::new();
    registry
        .register(Arc::new(PlanningOnly {
            descriptor,
            estimate: FrontierResourceEstimate::new(
                v,
                ContentId::of_bytes(b"synthetic machine, not inventory"),
                ContentId::of_bytes(b"synthetic estimate, not measurements"),
            )
            .unwrap(),
        }))
        .unwrap();
    let meta = work.metadata();
    let plan = registry
        .plan(
            &id,
            SolverTrust::Experimental,
            &profile
                .request(meta.rows as u64, meta.columns as u64)
                .unwrap(),
        )
        .unwrap();
    let request = FrontierStageRequest::new(
        ContentId::of_bytes(b"synthetic model build run"),
        work.origin().source_id,
        work.origin().tensor_index,
        FrontierStage::Fit,
        &profile,
        &plan,
        work.input_id(),
    )
    .unwrap();
    QteaSelectedAdmission {
        expected_plan_id: plan.content_id(),
        plan,
        request,
    }
}
fn fitted_builder<'a>(
    source: &'a mut dyn WeightSource,
    hybrid: bool,
) -> (QteaModelBuilder<'a, Cursor<Vec<u8>>>, OriginalSourceId) {
    let request = observation_plan(source, hybrid);
    let observed = ObservedWeightSource::new(&request, source).unwrap();
    let model = observed.source_id();
    let originals = observed
        .tensor_bindings()
        .iter()
        .map(|b| b.source.clone())
        .collect::<Vec<_>>();
    let profile = if hybrid {
        QteaModelProfile::Qwen35Hybrid
    } else {
        QteaModelProfile::Standard
    };
    let mut builder =
        QteaModelBuilder::new(observed, profile, campaign(), Cursor::new(vec![]), limits())
            .unwrap();
    for original in &originals {
        if original.shape.len() != 2 {
            continue;
        }
        if original.name.ends_with("embed_tokens.weight") {
            let g = gram(original.shape[1] as usize, model);
            let inspected = builder
                .inspect_owner(
                    &original.name,
                    QteaOwnerCurvature::Native(&g),
                    fit_config(),
                    |w| {
                        Ok((
                            w.input_id(),
                            w.metadata(),
                            w.memory(),
                            w.byte_ledger(),
                            select(w),
                        ))
                    },
                )
                .unwrap();
            builder
                .fit_owner(
                    &original.name,
                    QteaOwnerCurvature::Native(&g),
                    fit_config(),
                    |w| {
                        assert_eq!(
                            (w.input_id(), w.metadata(), w.memory(), w.byte_ledger()),
                            (inspected.0, inspected.1, inspected.2, inspected.3)
                        );
                        Ok(inspected.4)
                    },
                )
                .unwrap();
        } else {
            let c = calibrated(original, model);
            let inspected = builder
                .inspect_owner(
                    &original.name,
                    QteaOwnerCurvature::Calibrated(&c),
                    fit_config(),
                    |w| {
                        Ok((
                            w.input_id(),
                            w.metadata(),
                            w.memory(),
                            w.byte_ledger(),
                            select(w),
                        ))
                    },
                )
                .unwrap();
            builder
                .fit_owner(
                    &original.name,
                    QteaOwnerCurvature::Calibrated(&c),
                    fit_config(),
                    |w| {
                        assert_eq!(
                            (w.input_id(), w.metadata(), w.memory(), w.byte_ledger()),
                            (inspected.0, inspected.1, inspected.2, inspected.3)
                        );
                        Ok(inspected.4)
                    },
                )
                .unwrap();
        }
    }

    (builder, model)
}
fn build(
    source: &mut dyn WeightSource,
    hybrid: bool,
    tied: bool,
) -> (Vec<u8>, QteaModelBuildResult) {
    if !hybrid && tied {
        assert!(
            !source
                .inventory()
                .unwrap()
                .iter()
                .any(|t| t.name == "lm_head.weight")
        );
    }
    let (builder, model) = fitted_builder(source, hybrid);
    let mut output = Cursor::new(vec![]);
    let result = builder.finish(&mut output).unwrap();
    let bytes = output.into_inner();
    assert_eq!(
        result.package_id(),
        tritium_format::PackageId::from_package_bytes(&bytes)
    );
    assert_eq!(result.record().source_model(), model);
    (bytes, result)
}
fn reader(bytes: &[u8], hybrid: bool, model: OriginalSourceId) -> QteaModelReader<Cursor<&[u8]>> {
    if hybrid {
        QteaModelReader::open_hybrid(
            Cursor::new(bytes),
            limits().receiving,
            QteaHybridModelReadLimits {
                base: limits().model,
                max_preserved_tensor_elements: 64,
            },
            *model.as_bytes(),
        )
        .unwrap()
    } else {
        QteaModelReader::open(
            Cursor::new(bytes),
            limits().receiving,
            limits().model,
            *model.as_bytes(),
        )
        .unwrap()
    }
}
#[test]
fn observed_collector_admitted_fits_build_complete_standard_and_hybrid_models() {
    for (hybrid, tied, precision) in [
        (false, true, ScalePrecision::F32),
        (false, false, ScalePrecision::F16),
        (true, false, ScalePrecision::Bf16),
    ] {
        let mut original = source(hybrid, tied, precision);
        let expected = original.clone();
        let (bytes, result) = build(&mut original, hybrid, tied);
        let mut model = reader(&bytes, hybrid, result.record().source_model());
        let checked = verify_qtea_model_build_record(&mut model, 1 << 20).unwrap();
        assert_eq!(
            checked.fitted_matrices(),
            expected
                .tensors
                .values()
                .filter(|t| t.shape.len() == 2)
                .count()
        );
        assert_eq!(
            serde_json::to_value(result.record()).unwrap(),
            serde_json::to_value(checked).unwrap()
        );
        for (name, value) in &expected.assets {
            assert_eq!(model.asset_bytes(name).is_ok(), name != "config.json");
            if name == "config.json" {
                assert_eq!(model.config().unwrap(), value.as_slice());
            } else {
                assert_eq!(model.asset(name).unwrap(), value.as_slice());
            }
        }
        model.verify().unwrap();
        #[cfg(feature = "host-qtea")]
        cold_and_export(
            &bytes,
            hybrid,
            tied,
            result.record().source_model(),
            &expected,
            None,
        );
    }
}
#[cfg(feature = "host-qtea")]
fn cold_and_export(
    bytes: &[u8],
    hybrid: bool,
    tied: bool,
    source: OriginalSourceId,
    original: &InMemoryWeightSource,
    published: Option<&std::path::Path>,
) {
    use tritium_nn::{
        ChatRenderOptions, HostQteaHybridModelLimits, HostQteaModelLimits, LoadedModel,
        StreamLimits, Tokenizer,
    };
    let mut recorder = tritium_evidence::EvidenceRecorder::new(
        "built-model",
        [tritium_evidence::MODEL_LOADED_REGISTRATION],
    )
    .unwrap();
    let base = HostQteaModelLimits {
        receiving: limits().receiving,
        model: limits().model,
        max_weight_bytes: 1 << 22,
        max_staging_bytes: 1 << 16,
    };
    let stream = StreamLimits {
        max_context: 8,
        max_segment_tokens: 3,
        max_host_cache_payload_bytes: 4096,
    };
    let context = tritium_runtime::TensorUploadContext {
        recorder: &mut recorder,
        span: "load",
        parent: None,
        logical_time: 0,
    };
    let mut loaded = if hybrid {
        LoadedModel::load_qtea_hybrid(
            Cursor::new(bytes),
            *source.as_bytes(),
            HostQteaHybridModelLimits {
                base,
                max_preserved_tensor_elements: 64,
            },
            stream,
            context,
        )
        .unwrap()
    } else {
        LoadedModel::load_qtea(
            Cursor::new(bytes),
            *source.as_bytes(),
            base,
            stream,
            context,
        )
        .unwrap()
    };
    assert_eq!(
        loaded.runner.mtp_status(),
        hybrid.then_some(tritium_nn::Qwen35MtpStatus::Unverified)
    );
    let prompt = loaded
        .tokenizer
        .render_chat_prompt(&[("user", "Testing")], 1024)
        .unwrap()
        .unwrap();
    let ids = loaded.tokenizer.encode(&prompt).unwrap();
    assert_eq!(ids, [1, 2]);
    let thinking = loaded
        .tokenizer
        .render_chat_prompt_with_options(
            &[("user", "Testing")],
            ChatRenderOptions::default().with_thinking(true),
            1024,
        )
        .unwrap()
        .unwrap();
    let thinking_ids = loaded.tokenizer.encode(&thinking).unwrap();
    assert_eq!(thinking_ids, [7, 1, 2]);
    let generated = loaded
        .runner
        .generate_bounded(&ids, 3, loaded.tokenizer.eos())
        .unwrap();
    if let Some(root) = std::env::var_os("TRITIUM_QTEA_BUILD_FIXTURE_OUTPUT") {
        let dir = std::path::PathBuf::from(root).join(if hybrid {
            "hybrid"
        } else if tied {
            "tied"
        } else {
            "untied"
        });
        std::fs::create_dir(&dir).unwrap();
        let write = |name: &str, bytes: &[u8]| {
            std::fs::File::create_new(dir.join(name))
                .unwrap()
                .write_all(bytes)
                .unwrap()
        };
        let mut model = reader(bytes, hybrid, source);
        if let Some(path) = published {
            std::fs::hard_link(path, dir.join("model.trit")).unwrap();
            write(
                "candidate.json",
                &std::fs::read(path.parent().unwrap().join("candidate.json")).unwrap(),
            );
        } else {
            write("model.trit", bytes);
        }
        let mut without = original.clone();
        let mut tokenizer_config: serde_json::Value =
            serde_json::from_slice(&without.assets["tokenizer_config.json"]).unwrap();
        tokenizer_config
            .as_object_mut()
            .unwrap()
            .remove("chat_template");
        without.assets.insert(
            "tokenizer_config.json".into(),
            serde_json::to_vec(&tokenizer_config).unwrap(),
        );
        let (no_template, no_template_result) = build(&mut without, hybrid, tied);
        write("model-no-template.trit", &no_template);
        write(
            "no-template-source-model.json",
            &serde_json::to_vec(no_template_result.record().source_model().as_bytes()).unwrap(),
        );
        write("config.json", &model.config().unwrap());
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            write(name, &model.asset(name).unwrap());
        }
        write("actual-text.json",&serde_json::to_vec(&serde_json::json!({"schema":"tritium.qtea.text-software-observation.v1","qualified":false,"package_id":loaded.runner.qtea_package_id().unwrap().as_bytes(),"source_model":source.as_bytes(),"messages":[{"role":"user","content":"Testing"}],"prompt":prompt,"input_ids":ids,"thinking_prompt":thinking,"thinking_input_ids":thinking_ids,"generated_ids":generated})).unwrap());
    }
}

#[test]
fn incomplete_nonempty_and_failed_owner_cohorts_never_complete() {
    for failure in 0..5 {
        let mut original = source(false, true, ScalePrecision::F32);
        let p = observation_plan(&original, false);
        let observed = ObservedWeightSource::new(&p, &mut original).unwrap();
        let model = observed.source_id();
        let mut caps = limits();
        if failure == 1 {
            caps.max_spool_bytes = 1;
        }
        if failure == 2 {
            caps.max_widened_tensor_bytes = 1;
        }
        let mut builder = QteaModelBuilder::new(
            observed,
            QteaModelProfile::Standard,
            campaign(),
            Cursor::new(vec![]),
            caps,
        )
        .unwrap();
        let g = gram(8, model);
        let mut selections = 0;
        if failure > 0 {
            let result = builder.fit_owner(
                "model.embed_tokens.weight",
                QteaOwnerCurvature::Native(&g),
                fit_config(),
                |w| {
                    selections += 1;
                    if failure == 4 {
                        return Err(QteaModelBuildError::selection(std::io::Error::other(
                            "caller selection unavailable",
                        )));
                    }
                    let mut selected = select(w);
                    if failure == 3 {
                        selected.expected_plan_id =
                            ContentId::of_bytes(b"wrong selected admission");
                    }
                    Ok(selected)
                },
            );
            assert!(result.is_err());
            assert_eq!(selections, usize::from(failure >= 3));
        }
        let mut out = Cursor::new(vec![]);
        assert!(builder.finish(&mut out).is_err());
        assert!(out.into_inner().is_empty());
    }
    let mut original = source(false, true, ScalePrecision::F32);
    let p = observation_plan(&original, false);
    let observed = ObservedWeightSource::new(&p, &mut original).unwrap();
    let b = QteaModelBuilder::new(
        observed,
        QteaModelProfile::Standard,
        campaign(),
        Cursor::new(vec![]),
        limits(),
    )
    .unwrap();
    let mut out = Cursor::new(b"existing".to_vec());
    assert!(b.finish(&mut out).is_err());
    assert_eq!(out.into_inner(), b"existing");
}

#[test]
fn shared_inventory_refuses_missing_extra_wrong_shapes_and_architecture() {
    for hybrid in [false, true] {
        let original = source(hybrid, false, ScalePrecision::F32);
        let cfg = &original.assets["config.json"];
        let shapes = original
            .tensors
            .iter()
            .map(|(n, t)| (n.clone(), t.shape.clone()))
            .collect::<BTreeMap<_, _>>();
        let profile = if hybrid {
            QteaModelProfile::Qwen35Hybrid
        } else {
            QteaModelProfile::Standard
        };
        assert!(QteaModelInventory::prepare(cfg, profile, shapes.clone(), 128).is_ok());
        for mode in 0..3 {
            let mut altered = shapes.clone();
            let first = altered.keys().next().unwrap().clone();
            match mode {
                0 => {
                    altered.remove(&first);
                }
                1 => {
                    altered.insert("unexpected.weight".into(), vec![8, 8]);
                }
                _ => altered.get_mut(&first).unwrap()[0] += 1,
            };
            assert!(QteaModelInventory::prepare(cfg, profile, altered, 128).is_err());
        }
        assert!(QteaModelInventory::prepare(cfg, profile, shapes.clone(), 1).is_err());
        assert!(
            QteaModelInventory::prepare(
                cfg,
                if hybrid {
                    QteaModelProfile::Standard
                } else {
                    QteaModelProfile::Qwen35Hybrid
                },
                shapes,
                128
            )
            .is_err()
        );
    }
}
struct BrokenStream {
    inner: Cursor<Vec<u8>>,
    left: usize,
    writes: Arc<std::sync::atomic::AtomicUsize>,
}
impl Read for BrokenStream {
    fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
        self.inner.read(b)
    }
}
impl Seek for BrokenStream {
    fn seek(&mut self, p: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(p)
    }
}
impl Write for BrokenStream {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.writes
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if self.left == 0 {
            return Err(std::io::Error::other("injected private writer failure"));
        }
        let n = self.inner.write(&b[..b.len().min(self.left)])?;
        self.left -= n;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
#[test]
fn partial_spool_errors_poison_and_duplicate_owner_has_no_retry() {
    for broken in [false, true] {
        let mut original = source(false, true, ScalePrecision::F32);
        let p = observation_plan(&original, false);
        let observed = ObservedWeightSource::new(&p, &mut original).unwrap();
        let model = observed.source_id();
        let writes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let spool = BrokenStream {
            inner: Cursor::new(vec![]),
            left: if broken { 7 } else { usize::MAX },
            writes: writes.clone(),
        };
        let mut builder = QteaModelBuilder::new(
            observed,
            QteaModelProfile::Standard,
            campaign(),
            spool,
            limits(),
        )
        .unwrap();
        let g = gram(8, model);
        let fitted = builder.fit_owner(
            "model.embed_tokens.weight",
            QteaOwnerCurvature::Native(&g),
            fit_config(),
            |w| Ok(select(w)),
        );
        if broken {
            let error = fitted.unwrap_err();
            assert!(std::error::Error::source(&error).is_some());
            assert!(matches!(
                std::error::Error::source(&error)
                    .unwrap()
                    .downcast_ref::<QteaFrontierWorkError>(),
                Some(QteaFrontierWorkError::Work(QteaTensorWorkError::Format(
                    tritium_format::qtea::QteaError::Io(std::io::ErrorKind::Other)
                )))
            ));
        } else {
            fitted.unwrap();
            assert!(
                builder
                    .inspect_owner::<()>(
                        "model.embed_tokens.weight",
                        QteaOwnerCurvature::Native(&g),
                        fit_config(),
                        |_| panic!("fitted owner reached inspector")
                    )
                    .unwrap_err()
                    .to_string()
                    .contains("duplicate")
            );
            let mut selections = 0;
            assert!(
                builder
                    .fit_owner(
                        "model.embed_tokens.weight",
                        QteaOwnerCurvature::Native(&g),
                        fit_config(),
                        |w| {
                            selections += 1;
                            Ok(select(w))
                        }
                    )
                    .is_err()
            );
            assert_eq!(selections, 0);
        }
        assert!(writes.load(std::sync::atomic::Ordering::SeqCst) > 0);
        let mut output = Cursor::new(vec![]);
        assert!(builder.finish(&mut output).is_err());
        assert!(output.into_inner().is_empty());
    }
}
#[test]
fn tampered_fit_observations_never_verify_as_binding_or_qualification() {
    let mut original = source(false, true, ScalePrecision::F32);
    let (bytes, result) = build(&mut original, false, true);
    for mode in 0..5 {
        let outer = tritium_format::read_trit_package(&bytes).unwrap();
        let mut read = reader(&bytes, false, result.record().source_model());
        let mut manifest = read.manifest().clone();
        let mut record: serde_json::Value =
            serde_json::from_slice(&read.asset("qtea-fit.json").unwrap()).unwrap();
        match mode {
            0 => record["qualification"] = true.into(),
            1 => record["fits"][0]["weights_digest"][0] = 99.into(),
            2 => {
                let value = record["fits"][0].clone();
                record["fits"].as_array_mut().unwrap().push(value);
            }
            3 => {
                record["fits"][0]["plan_id"] =
                    serde_json::to_value(ContentId::of_bytes(b"wrong plan identity")).unwrap()
            }
            _ => {
                record["fits"][0]["output_id"] =
                    serde_json::to_value(ContentId::of_bytes(b"wrong output identity")).unwrap()
            }
        }
        let replaced = serde_json::to_vec(&record).unwrap();
        manifest.assets.insert(
            "qtea-fit.json".into(),
            BlobId::from_bytes(*blake3::hash(&replaced).as_bytes()),
        );
        let metadata =
            tritium_format::qtea::model::encode_qtea_model_manifest(&manifest, 1 << 16).unwrap();
        let blobs = outer
            .blobs()
            .iter()
            .map(|b| tritium_format::TritBlob {
                kind: b.kind.clone(),
                bytes: if b.kind == tritium_format::TritBlobKind::Asset("qtea-fit.json".into()) {
                    &replaced
                } else {
                    outer.blob(b).unwrap()
                },
            })
            .collect::<Vec<_>>();
        let altered =
            tritium_format::write_trit_package_bounded(&metadata, &blobs, 1 << 22).unwrap();
        let mut read = reader(&altered, false, result.record().source_model());
        assert!(verify_qtea_model_build_record(&mut read, 1 << 20).is_err());
    }
}

struct GuardedSpool {
    inner: Cursor<Vec<u8>>,
    forbidden: Arc<std::sync::atomic::AtomicBool>,
}
impl GuardedSpool {
    fn check(&self) {
        assert!(
            !self.forbidden.load(std::sync::atomic::Ordering::SeqCst),
            "inspection must not access the spool"
        );
    }
}
impl Read for GuardedSpool {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        self.check();
        self.inner.read(bytes)
    }
}
impl Write for GuardedSpool {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.check();
        self.inner.write(bytes)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.check();
        self.inner.flush()
    }
}
impl Seek for GuardedSpool {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.check();
        self.inner.seek(position)
    }
}

#[test]
fn inspection_never_accesses_spool_or_marks_an_owner_fitted_and_errors_are_retryable() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let mut raw = source(false, true, ScalePrecision::F32);
    let p = observation_plan(&raw, false);
    let observed = ObservedWeightSource::new(&p, &mut raw).unwrap();
    let model = observed.source_id();
    let forbidden = Arc::new(AtomicBool::new(false));
    let mut builder = QteaModelBuilder::new(
        observed,
        QteaModelProfile::Standard,
        campaign(),
        GuardedSpool {
            inner: Cursor::new(vec![]),
            forbidden: forbidden.clone(),
        },
        limits(),
    )
    .unwrap();
    forbidden.store(true, Ordering::SeqCst);
    let g = gram(8, model);
    let wrong_model = gram(8, OriginalSourceId::from_bytes([9; 32]));
    let wrong_shape = gram(4, model);
    for (name, curvature, config) in [
        ("absent", &g, fit_config()),
        ("model.norm.weight", &g, fit_config()),
        ("model.embed_tokens.weight", &wrong_model, fit_config()),
        ("model.embed_tokens.weight", &wrong_shape, fit_config()),
        (
            "model.embed_tokens.weight",
            &g,
            TensorFitConfig {
                group_size: 0,
                ..fit_config()
            },
        ),
    ] {
        let error = builder
            .inspect_owner::<()>(name, QteaOwnerCurvature::Native(curvature), config, |_| {
                panic!("invalid work reached inspector")
            })
            .unwrap_err();
        assert!(!error.to_string().contains("poisoned"));
    }
    let name = "model.embed_tokens.weight";
    let callback_error = builder
        .inspect_owner(name, QteaOwnerCurvature::Native(&g), fit_config(), |_| {
            Err::<(), _>(QteaModelBuildError::selection(std::io::Error::other(
                "inspection declined",
            )))
        })
        .unwrap_err();
    assert!(callback_error.to_string().contains("inspection declined"));
    let inspected = builder
        .inspect_owner(name, QteaOwnerCurvature::Native(&g), fit_config(), |w| {
            Ok(select(w))
        })
        .unwrap();
    let repeated = builder
        .inspect_owner(name, QteaOwnerCurvature::Native(&g), fit_config(), |w| {
            Ok(w.input_id())
        })
        .unwrap();
    assert_eq!(repeated, inspected.request.input_id());
    // A successful inspection leaves the cohort incomplete, and finish refuses
    // before touching its guarded spool or output.
    let mut output = Cursor::new(vec![]);
    assert!(
        builder
            .finish(&mut output)
            .unwrap_err()
            .to_string()
            .contains("unfitted")
    );
    assert!(output.into_inner().is_empty());
}

#[test]
fn inspection_is_not_spd_validation_or_native_fit_success() {
    let mut raw = source(false, true, ScalePrecision::F32);
    let p = observation_plan(&raw, false);
    let observed = ObservedWeightSource::new(&p, &mut raw).unwrap();
    let model = observed.source_id();
    let mut singular =
        InputGramAccumulator::new_bound(8, gram(8, model).source_id().unwrap(), 0).unwrap();
    singular.accumulate_batch(&[0.; 64], 8, None, None).unwrap();
    let singular = singular.finish().unwrap();
    let mut builder = QteaModelBuilder::new(
        observed,
        QteaModelProfile::Standard,
        campaign(),
        Cursor::new(vec![]),
        limits(),
    )
    .unwrap();
    let selection = builder
        .inspect_owner(
            "model.embed_tokens.weight",
            QteaOwnerCurvature::Native(&singular),
            fit_config(),
            |w| Ok(select(w)),
        )
        .unwrap();
    let mut calls = 0;
    assert!(
        builder
            .fit_owner(
                "model.embed_tokens.weight",
                QteaOwnerCurvature::Native(&singular),
                fit_config(),
                |_| {
                    calls += 1;
                    Ok(selection)
                }
            )
            .is_err()
    );
    assert_eq!(calls, 1);
    assert!(
        builder
            .inspect_owner::<()>(
                "model.embed_tokens.weight",
                QteaOwnerCurvature::Native(&singular),
                fit_config(),
                |_| panic!("poisoned work inspected")
            )
            .unwrap_err()
            .to_string()
            .contains("poisoned")
    );
    let mut output = Cursor::new(vec![]);
    assert!(builder.finish(&mut output).is_err());
    assert!(output.into_inner().is_empty());
}

#[test]
fn inspection_enforces_the_same_widened_native_work_and_wire_caps() {
    for fault in 0..5 {
        let mut raw = source(false, true, ScalePrecision::F32);
        let p = observation_plan(&raw, false);
        let observed = ObservedWeightSource::new(&p, &mut raw).unwrap();
        let g = gram(8, observed.source_id());
        let mut caps = limits();
        match fault {
            0 => caps.max_widened_tensor_bytes = 1,
            1 => caps.max_native_work_bytes = 1,
            2 => caps.max_tensor_work_bytes = 1,
            3 => caps.model.tensor.max_tensor_bytes = 1,
            _ => caps.model.tensor.max_validation_bytes = 1,
        }
        let mut builder = QteaModelBuilder::new(
            observed,
            QteaModelProfile::Standard,
            campaign(),
            Cursor::new(vec![]),
            caps,
        )
        .unwrap();
        let inspect = builder
            .inspect_owner::<()>(
                "model.embed_tokens.weight",
                QteaOwnerCurvature::Native(&g),
                fit_config(),
                |_| panic!("over-cap work reached inspector"),
            )
            .unwrap_err();
        let fit = builder
            .fit_owner(
                "model.embed_tokens.weight",
                QteaOwnerCurvature::Native(&g),
                fit_config(),
                |_| panic!("over-cap work reached selector"),
            )
            .unwrap_err();
        assert_eq!(inspect.to_string(), fit.to_string());
        assert!(!fit.to_string().contains("poisoned"));
    }
}

struct SharedSource(std::rc::Rc<std::cell::RefCell<InMemoryWeightSource>>);
impl WeightSource for SharedSource {
    fn inventory(
        &self,
    ) -> Result<Vec<QuantSourceTensor>, tritium_quantize::engine::QuantEngineError> {
        self.0.borrow().inventory()
    }
    fn asset_names(&self) -> Vec<String> {
        self.0.borrow().asset_names()
    }
    fn tensor(
        &mut self,
        name: &str,
        max: usize,
    ) -> Result<std::borrow::Cow<'_, [u8]>, tritium_quantize::engine::QuantEngineError> {
        Ok(std::borrow::Cow::Owned(
            self.0.borrow_mut().tensor(name, max)?.into_owned(),
        ))
    }
    fn asset(
        &mut self,
        name: &str,
        max: usize,
    ) -> Result<std::borrow::Cow<'_, [u8]>, tritium_quantize::engine::QuantEngineError> {
        Ok(std::borrow::Cow::Owned(
            self.0.borrow_mut().asset(name, max)?.into_owned(),
        ))
    }
    fn input_gram(
        &mut self,
        name: &str,
        max: usize,
    ) -> Result<std::borrow::Cow<'_, [f64]>, tritium_quantize::engine::QuantEngineError> {
        Ok(std::borrow::Cow::Owned(
            self.0.borrow_mut().input_gram(name, max)?.into_owned(),
        ))
    }
}

#[test]
fn execution_rereads_original_and_refuses_stale_curvature_or_recipe_selection() {
    for fault in 0..3 {
        let data = std::rc::Rc::new(std::cell::RefCell::new(source(
            false,
            true,
            ScalePrecision::F32,
        )));
        let mut raw = SharedSource(data.clone());
        let p = observation_plan(&raw, false);
        let observed = ObservedWeightSource::new(&p, &mut raw).unwrap();
        let model = observed.source_id();
        let writes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut builder = QteaModelBuilder::new(
            observed,
            QteaModelProfile::Standard,
            campaign(),
            BrokenStream {
                inner: Cursor::new(vec![]),
                left: usize::MAX,
                writes: writes.clone(),
            },
            limits(),
        )
        .unwrap();
        let name = "model.embed_tokens.weight";
        let g = gram(8, model);
        let selection = builder
            .inspect_owner(name, QteaOwnerCurvature::Native(&g), fit_config(), |w| {
                Ok(select(w))
            })
            .unwrap();
        let mut changed = InputGramAccumulator::new_bound(8, g.source_id().unwrap(), 0).unwrap();
        let mut values = [0.; 64];
        for c in 0..8 {
            values[c * 8 + c] = 2.;
        }
        changed.accumulate_batch(&values, 8, None, None).unwrap();
        let changed = changed.finish().unwrap();
        let curvature = if fault == 1 { &changed } else { &g };
        let config = if fault == 2 {
            TensorFitConfig {
                salient_ratio: 0.,
                ..fit_config()
            }
        } else {
            fit_config()
        };
        if fault == 0 {
            data.borrow_mut().tensors.get_mut(name).unwrap().bytes[0] ^= 1;
            let error = builder
                .inspect_owner::<()>(name, QteaOwnerCurvature::Native(curvature), config, |_| {
                    panic!("changed original reached inspector")
                })
                .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("observed tensor content changed"),
                "{error}"
            );
        } else {
            let changed_id = builder
                .inspect_owner(name, QteaOwnerCurvature::Native(curvature), config, |w| {
                    Ok(w.input_id())
                })
                .unwrap();
            assert_ne!(changed_id, selection.request.input_id());
        }
        let mut calls = 0;
        let error = builder
            .fit_owner(name, QteaOwnerCurvature::Native(curvature), config, |_| {
                calls += 1;
                Ok(selection)
            })
            .unwrap_err();
        assert_eq!(calls, usize::from(fault != 0));
        if fault == 0 {
            assert!(
                error
                    .to_string()
                    .contains("observed tensor content changed"),
                "{error}"
            );
        }
        assert_eq!(writes.load(std::sync::atomic::Ordering::SeqCst), 0);
        let mut output = Cursor::new(vec![]);
        assert!(builder.finish(&mut output).is_err());
        assert!(output.into_inner().is_empty());
    }
}
#[test]
fn source_mutation_output_failure_and_late_caps_produce_no_successful_result() {
    for fault in 0..4 {
        let data = std::rc::Rc::new(std::cell::RefCell::new(source(
            false,
            true,
            ScalePrecision::F32,
        )));
        let mut raw = SharedSource(data.clone());
        let p = observation_plan(&raw, false);
        let observed = ObservedWeightSource::new(&p, &mut raw).unwrap();
        let model = observed.source_id();
        let names = observed
            .tensor_bindings()
            .iter()
            .filter(|b| b.source.shape.len() == 2)
            .map(|b| (b.source.name.clone(), b.source.shape[1] as usize))
            .collect::<Vec<_>>();
        let mut caps = limits();
        if fault == 2 {
            caps.max_fit_record_bytes = 1;
        }
        if fault == 3 {
            caps.receiving.max_source_bytes = 1;
        }
        let mut builder = QteaModelBuilder::new(
            observed,
            QteaModelProfile::Standard,
            campaign(),
            Cursor::new(vec![]),
            caps,
        )
        .unwrap();
        for (name, cols) in names {
            let g = gram(cols, model);
            builder
                .fit_owner(&name, QteaOwnerCurvature::Native(&g), fit_config(), |w| {
                    Ok(select(w))
                })
                .unwrap();
        }
        if fault == 0 {
            data.borrow_mut()
                .tensors
                .get_mut("model.embed_tokens.weight")
                .unwrap()
                .bytes[0] ^= 1;
        }
        if fault == 1 {
            let writes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let mut out = BrokenStream {
                inner: Cursor::new(vec![]),
                left: 7,
                writes: writes.clone(),
            };
            assert!(builder.finish(&mut out).is_err());
            assert!(writes.load(std::sync::atomic::Ordering::SeqCst) > 0);
            assert!(!out.inner.into_inner().is_empty());
        } else {
            let mut out = Cursor::new(vec![]);
            let error = builder.finish(&mut out).unwrap_err();
            if fault == 0 {
                assert!(
                    error
                        .to_string()
                        .contains("observed tensor content changed"),
                    "{error}"
                );
                assert!(!out.into_inner().is_empty());
            } else {
                assert!(out.into_inner().is_empty());
            }
        }
    }
}

#[test]
fn grouped_and_gather_statistics_are_not_relabelled_or_retried_as_native() {
    for gather in [false, true] {
        let mut raw = source(false, true, ScalePrecision::F32);
        let p = observation_plan(&raw, false);
        let observed = ObservedWeightSource::new(&p, &mut raw).unwrap();
        let model = observed.source_id();
        let original = observed
            .tensor_bindings()
            .iter()
            .find(|b| b.source.name == "model.embed_tokens.weight")
            .unwrap()
            .source
            .clone();
        let plan = CalibrationPlan {
            tensor: original,
            source: CalibrationSource {
                model: FitInputDigest(*model.as_bytes()),
                cache: FitInputDigest([2; 32]),
                tokens: FitInputDigest([3; 32]),
            },
            partition: if gather {
                CalibrationPartition::Full
            } else {
                CalibrationPartition::Groups { columns: 4 }
            },
            consumer: if gather {
                CalibrationConsumer::GatherIdentity
            } else {
                CalibrationConsumer::Matmul
            },
            output: if gather {
                CalibrationOutput::IndexedFisher
            } else {
                CalibrationOutput::None
            },
            sample_start: 0,
            damping: 0.,
            max_numeric_bytes: 1 << 20,
        };
        let mut collector = CalibrationCollector::new(plan).unwrap();
        let mut x = vec![0.; 64];
        for c in 0..8 {
            x[c * 8 + c] = 1.;
        }
        let indices = (0..8).collect::<Vec<_>>();
        let factors = [1.; 8];
        collector
            .accumulate(CalibrationBatch {
                samples: 8,
                input: if gather {
                    CalibrationInput::GatherIdentity
                } else {
                    CalibrationInput::Activations(&x)
                },
                output: if gather {
                    CalibrationOutputBatch::Indexed {
                        indices: &indices,
                        factors: &factors,
                    }
                } else {
                    CalibrationOutputBatch::None
                },
                weights: None,
                mask: None,
            })
            .unwrap();
        let c = collector.finish().unwrap();
        let mut builder = QteaModelBuilder::new(
            observed,
            QteaModelProfile::Standard,
            campaign(),
            Cursor::new(vec![]),
            limits(),
        )
        .unwrap();
        let mut selections = 0;
        let inspected_error = builder
            .inspect_owner::<()>(
                "model.embed_tokens.weight",
                QteaOwnerCurvature::Calibrated(&c),
                fit_config(),
                |_| panic!("unsupported capture reached inspector"),
            )
            .unwrap_err();
        assert!(matches!(
            std::error::Error::source(&inspected_error)
                .unwrap()
                .downcast_ref::<QteaCalibrationError>(),
            Some(QteaCalibrationError::RequiresInputOnlyFullGram)
        ));
        let error = builder
            .fit_owner(
                "model.embed_tokens.weight",
                QteaOwnerCurvature::Calibrated(&c),
                fit_config(),
                |w| {
                    selections += 1;
                    Ok(select(w))
                },
            )
            .unwrap_err();
        assert_eq!(selections, 0);
        assert!(matches!(
            std::error::Error::source(&error)
                .unwrap()
                .downcast_ref::<QteaCalibrationError>(),
            Some(QteaCalibrationError::RequiresInputOnlyFullGram)
        ));
        let mut output = Cursor::new(vec![]);
        assert!(builder.finish(&mut output).is_err());
        assert!(output.into_inner().is_empty());
    }
}

#[test]
fn complete_builders_publish_unqualified_standard_and_hybrid_candidates_without_overwrite() {
    use tritium_quantize::engine::{DirectoryTensorSink, UnqualifiedCandidateRecord};
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let root = std::env::temp_dir().join(format!(
        "qtea-candidate-builder-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    ));
    std::fs::create_dir(&root).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }
    let _cleanup = Cleanup(root.clone());
    for (hybrid, tied, precision) in [
        (false, true, ScalePrecision::F32),
        (false, false, ScalePrecision::F16),
        (true, false, ScalePrecision::Bf16),
    ] {
        let mut original = source(hybrid, tied, precision);
        let mut private_source = original.clone();
        #[cfg(feature = "host-qtea")]
        let expected_source = original.clone();
        let (private_bytes, private_result) = build(&mut private_source, hybrid, tied);
        let (builder, model) = fitted_builder(&mut original, hybrid);
        let dir = root.join(if hybrid {
            "hybrid"
        } else if tied {
            "tied"
        } else {
            "untied"
        });
        let mut sink = DirectoryTensorSink::new(&dir);
        let (published, warnings) = builder.publish_candidate(&mut sink).unwrap();
        assert!(warnings.is_empty());
        assert_eq!(published.package_id(), private_result.package_id());
        let bytes = std::fs::read(dir.join("model.trit")).unwrap();
        assert_eq!(bytes, private_bytes);
        let record = UnqualifiedCandidateRecord::from_slice(
            &std::fs::read(dir.join("candidate.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            record.package_id().as_bytes(),
            published.package_id().as_bytes()
        );
        assert!(!dir.join("evidence.jsonl").exists());
        assert!(!dir.join(".pending.trit").exists());
        let mut read = reader(&bytes, hybrid, model);
        verify_qtea_model_build_record(&mut read, 1 << 20).unwrap();
        let (retry, _) = fitted_builder(&mut private_source, hybrid);
        assert!(retry.publish_candidate(&mut sink).is_err());
        assert_eq!(std::fs::read(dir.join("model.trit")).unwrap(), bytes);
        #[cfg(feature = "host-qtea")]
        {
            use tritium_nn::{
                HostQteaHybridModelLimits, HostQteaModelLimits, LoadedModel, StreamLimits,
            };
            let mut evidence = tritium_evidence::EvidenceRecorder::new(
                "published-cold",
                [tritium_evidence::MODEL_LOADED_REGISTRATION],
            )
            .unwrap();
            let context = tritium_runtime::TensorUploadContext {
                recorder: &mut evidence,
                span: "load",
                parent: None,
                logical_time: 0,
            };
            let base = HostQteaModelLimits {
                receiving: limits().receiving,
                model: limits().model,
                max_weight_bytes: 1 << 22,
                max_staging_bytes: 1 << 16,
            };
            let stream = StreamLimits {
                max_context: 8,
                max_segment_tokens: 3,
                max_host_cache_payload_bytes: 4096,
            };
            let file = std::fs::File::open(dir.join("model.trit")).unwrap();
            let mut loaded = if hybrid {
                LoadedModel::load_qtea_hybrid(
                    file,
                    *model.as_bytes(),
                    HostQteaHybridModelLimits {
                        base,
                        max_preserved_tensor_elements: 64,
                    },
                    stream,
                    context,
                )
                .unwrap()
            } else {
                LoadedModel::load_qtea(file, *model.as_bytes(), base, stream, context).unwrap()
            };
            loaded.runner.generate_bounded(&[1, 2], 3, 3).unwrap();
            assert_eq!(
                loaded.runner.mtp_status(),
                hybrid.then_some(tritium_nn::Qwen35MtpStatus::Unverified)
            );
            cold_and_export(
                &bytes,
                hybrid,
                tied,
                model,
                &expected_source,
                Some(&dir.join("model.trit")),
            );
        }
    }
    let mut source = source(false, true, ScalePrecision::F32);
    let plan = observation_plan(&source, false);
    let observed = ObservedWeightSource::new(&plan, &mut source).unwrap();
    let builder = QteaModelBuilder::new(
        observed,
        QteaModelProfile::Standard,
        campaign(),
        Cursor::new(vec![]),
        limits(),
    )
    .unwrap();
    let failed = root.join("incomplete");
    assert!(
        builder
            .publish_candidate(&mut DirectoryTensorSink::new(&failed))
            .is_err()
    );
    assert!(!failed.exists());
}
