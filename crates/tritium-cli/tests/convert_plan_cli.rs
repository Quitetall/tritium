//! Actual command/source/cache/sink calls compared with the actual production engine.
use std::{
    collections::BTreeMap,
    ffi::OsString,
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};
use tritium_evidence::*;
use tritium_quantize::engine::*;
use tritium_schema::*;

#[path = "convert_plan_cli/original_stage.rs"]
mod original_stage;

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "tritium-convert-plan-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
fn run(args: &[OsString]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_tritium"))
        // Thread count is part of the recorded fit context, hence the package
        // identity. Match the actual reference pool (including an installed
        // local pool), not a fixed child-only count or the host's CPU count.
        .env(
            "RAYON_NUM_THREADS",
            rayon::current_num_threads().to_string(),
        )
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdout = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();
    let out = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        stdout.read_to_end(&mut bytes).unwrap();
        bytes
    });
    let err = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        stderr.read_to_end(&mut bytes).unwrap();
        bytes
    });
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if start.elapsed() > Duration::from_secs(60) {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("CLI command exceeded its 60-second test deadline");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    Output {
        status,
        stdout: out.join().unwrap(),
        stderr: err.join().unwrap(),
    }
}
fn recorder() -> EvidenceRecorder {
    EvidenceRecorder::new(
        "filesystem-conversion",
        [
            DIRECT_LEVELS_FITTED_REGISTRATION,
            QUANTIZATION_STARTED_REGISTRATION,
            QUANTIZATION_COMPLETED_REGISTRATION,
            CALIBRATED_LEVELS_FITTED_REGISTRATION,
            CALIBRATED_QUANTIZATION_STARTED_REGISTRATION,
            CALIBRATED_QUANTIZATION_COMPLETED_REGISTRATION,
        ],
    )
    .unwrap()
}
fn direct() -> DirectFitPlan {
    DirectFitPlan {
        layout: AdditiveLayout {
            rows: 1,
            cols: 32,
            tile: 256,
            group: 32,
            max_planes: 2,
            allocation: PlaneAllocation::Uniform,
            codec: PlaneCodec::B3,
            law: ScaleLaw {
                anchor: ScaleAnchor::Group,
                relation: PlaneRelation::Free,
                precision: ScalePrecision::F32,
            },
            basis: Basis::Identity,
            transport: Transport::Raw,
        },
        profile: DirectFitProfile::JointIndependentGroups(JointGroupFitOptions {
            max_iterations: 4,
            ridge: 1e-8,
            em_restarts: 1,
            ridge_condition_limit: 1e6,
            softened_relay: false,
            modulated_relay: false,
        }),
        levels: vec![
            DirectFitLevel {
                name: "compact".into(),
                planes: 1,
            },
            DirectFitLevel {
                name: "full".into(),
                planes: 2,
            },
        ],
        limits: DirectFitLimits {
            max_weights: 32,
            max_metric_entries: 1024,
            max_output_trits: 96,
        },
    }
}
fn source(precision: ScalePrecision) -> InMemoryWeightSource {
    let bytes = |values: Vec<f32>| {
        values
            .into_iter()
            .flat_map(|v| match precision {
                ScalePrecision::F32 => v.to_le_bytes().to_vec(),
                ScalePrecision::F16 => half::f16::from_f32(v).to_bits().to_le_bytes().to_vec(),
                ScalePrecision::Bf16 => half::bf16::from_f32(v).to_bits().to_le_bytes().to_vec(),
                _ => unreachable!(),
            })
            .collect()
    };
    InMemoryWeightSource {
        tensors: BTreeMap::from([
            (
                "norm".into(),
                RawTensor {
                    shape: vec![2],
                    precision,
                    bytes: bytes(vec![-0.0, 1.0]),
                },
            ),
            (
                "weight".into(),
                RawTensor {
                    shape: vec![1, 32],
                    precision,
                    bytes: bytes((0..32).map(|i| (i as f32 - 16.) / 16.).collect()),
                },
            ),
        ]),
        assets: BTreeMap::from([
            ("config.json".into(), b"{}".to_vec()),
            ("tokenizer.json".into(), b"{}".to_vec()),
        ]),
        grams: BTreeMap::new(),
    }
}
fn plan(source: &dyn WeightSource) -> QuantPlan {
    QuantPlan {
        architecture: "filesystem-fixture".into(),
        tensors: source
            .inventory()
            .unwrap()
            .into_iter()
            .map(|source| {
                let action = if source.name == "weight" {
                    QuantAction::Fit {
                        plan: Box::new(direct()),
                        metric: DirectFitMetric::Identity,
                    }
                } else {
                    QuantAction::KeepDense
                };
                QuantTensorPlan { source, action }
            })
            .collect(),
        assets: source.asset_names(),
        levels: vec!["compact".into(), "full".into()],
        limits: QuantLimits {
            max_source_bytes: 1024,
            max_asset_bytes: 4,
            max_retained_blob_bytes: 4096,
            max_package_bytes: 100_000,
        },
    }
}
fn shard(name: &str, raw: &RawTensor) -> Vec<u8> {
    let dtype = match raw.precision {
        ScalePrecision::F16 => "F16",
        ScalePrecision::Bf16 => "BF16",
        ScalePrecision::F32 => "F32",
        _ => unreachable!(),
    };
    let mut header = serde_json::to_vec(&serde_json::json!({name: {"dtype":dtype, "shape":raw.shape, "data_offsets":[0,raw.bytes.len()]}})).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(&raw.bytes);
    bytes
}
fn arguments(
    dir: &Path,
    source: &InMemoryWeightSource,
    calibrated: bool,
    destination: &str,
) -> Vec<OsString> {
    let mut args: Vec<OsString> = [
        "convert",
        if calibrated {
            "--calibrated-plan"
        } else {
            "--plan"
        },
    ]
    .map(Into::into)
    .into();
    args.push(dir.join("plan.json").into());
    args.extend([
        OsString::from("--conversion-run-id"),
        OsString::from("filesystem-conversion"),
        OsString::from("--out"),
        dir.join(destination).into(),
    ]);
    for (name, tensor) in &source.tensors {
        let path = dir.join(format!("{name}.safetensors"));
        std::fs::write(&path, shard(name, tensor)).unwrap();
        args.extend([OsString::from("--source-shard"), path.into()]);
    }
    for (name, bytes) in &source.assets {
        let path = dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        args.extend([
            OsString::from("--asset"),
            OsString::from(format!("{name}={}", path.display())),
        ]);
    }
    args
}
fn check(dir: &Path, args: &[OsString], expected: &MemoryTensorSink) {
    let result = run(args);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let summary: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    let output = Path::new(summary["output"].as_str().unwrap());
    let package = std::fs::read(output.join("model.trit")).unwrap();
    let jsonl = std::fs::read_to_string(output.join("evidence.jsonl")).unwrap();
    let (actual, evidence) = expected.artifact().unwrap();
    assert_eq!(package, actual);
    assert_eq!(jsonl, evidence.jsonl());
    verify_package_with_fit_log(&package, &verify_jsonl(&jsonl).unwrap()).unwrap();
    assert!(
        run(&[
            "evidence".into(),
            "verify".into(),
            output.join("evidence.jsonl").into(),
            "--evidence".into(),
            "off".into()
        ])
        .status
        .success()
    );
    assert!(dir.exists());
}

#[test]
fn actual_plain_filesystem_conversion_matches_engine_for_all_original_precisions() {
    for precision in [
        ScalePrecision::F16,
        ScalePrecision::Bf16,
        ScalePrecision::F32,
    ] {
        let dir = Scratch::new();
        let mut raw = source(precision);
        let request = plan(&raw);
        std::fs::write(
            dir.0.join("plan.json"),
            serde_json::to_vec(&request).unwrap(),
        )
        .unwrap();
        let args = arguments(&dir.0, &raw, false, "output");
        let mut sink = MemoryTensorSink::default();
        Engine::run(&request, &mut raw, &mut sink, recorder()).unwrap();
        check(&dir.0, &args, &sink);
        let mut again = args.clone();
        let i = again.iter().position(|v| v == "--out").unwrap();
        again[i + 1] = dir.0.join("again").into();
        check(&dir.0, &again, &sink);
        let before = std::fs::read(dir.0.join("output/model.trit")).unwrap();
        assert!(!run(&args).status.success());
        assert_eq!(
            before,
            std::fs::read(dir.0.join("output/model.trit")).unwrap()
        );
    }
}

fn calibrated(raw: &mut InMemoryWeightSource) -> (CalibratedQuantPlan, CalibrationCache) {
    let plain = plan(raw);
    let model = Engine::observe_source(&plain, raw).unwrap();
    let original = plain
        .tensors
        .iter()
        .find(|t| t.source.name == "weight")
        .unwrap()
        .source
        .clone();
    let mut input = vec![0.; 1024];
    for i in 0..32 {
        input[i * 32 + i] = 1.;
    }
    let mut builder = CalibrationCacheBuilder::new(
        CalibrationCacheRequest {
            tensor: original,
            model,
            consumer: CalibrationConsumer::Matmul,
            output: CalibrationOutput::None,
            sample_start: 0,
            limits: CalibrationCacheLimits {
                max_payload_bytes: 100_000,
                max_samples: 32,
                max_windows: 1,
            },
        },
        &raw.assets["tokenizer.json"],
        b"toy fixture declared objective",
    )
    .unwrap();
    builder
        .capture(
            &(0..32).collect::<Vec<u32>>(),
            CalibrationBatch {
                samples: 32,
                input: CalibrationInput::Activations(&input),
                output: CalibrationOutputBatch::None,
                weights: None,
                mask: None,
            },
        )
        .unwrap();
    let cache = builder.finish().unwrap();
    let mut fit = CalibratedFitPlan {
        direct: direct(),
        cache: cache.binding().clone(),
        partition: CalibrationPartition::Full,
        damping: 0.,
        max_numeric_bytes: 1,
    };
    fit.max_numeric_bytes =
        CalibrationCollector::required_numeric_bytes(&fit.calibration_plan().unwrap()).unwrap()
            + cache.binding().payload_bytes;
    let request = CalibratedQuantPlan {
        architecture: plain.architecture,
        original_source: model,
        tensors: plain
            .tensors
            .into_iter()
            .map(|t| CalibratedQuantTensorPlan {
                source: t.source.clone(),
                action: if t.source.name == "weight" {
                    CalibratedQuantAction::CalibratedFit {
                        plan: Box::new(fit.clone()),
                    }
                } else {
                    CalibratedQuantAction::KeepDense
                },
            })
            .collect(),
        assets: plain.assets,
        levels: plain.levels,
        limits: plain.limits,
        max_cache_payload_bytes: 100_000,
        max_cache_package_bytes: 100_000,
    };
    (request, cache)
}
#[test]
fn actual_calibrated_filesystem_conversion_matches_in_memory_engine() {
    for precision in [
        ScalePrecision::F16,
        ScalePrecision::Bf16,
        ScalePrecision::F32,
    ] {
        let dir = Scratch::new();
        let mut raw = source(precision);
        let (request, cache) = calibrated(&mut raw);
        std::fs::write(
            dir.0.join("plan.json"),
            serde_json::to_vec(&request).unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.0.join("cache.trit"),
            cache.encode_package(100_000).unwrap(),
        )
        .unwrap();
        let mut args = arguments(&dir.0, &raw, true, "output");
        args.extend([
            "--calibration-cache".into(),
            format!("weight={}", dir.0.join("cache.trit").display()).into(),
        ]);
        let mut caches = InMemoryCalibrationSource::new([cache]).unwrap();
        let mut sink = MemoryTensorSink::default();
        Engine::run_calibrated(&request, &mut raw, &mut caches, &mut sink, recorder()).unwrap();
        check(&dir.0, &args, &sink);
        let i = args.iter().position(|v| v == "--out").unwrap();
        args[i + 1] = dir.0.join("foreign-source").into();
        let mut foreign = raw.tensors["weight"].clone();
        let changed = match precision {
            ScalePrecision::F32 => 0.25f32.to_le_bytes().to_vec(),
            ScalePrecision::F16 => half::f16::from_f32(0.25).to_bits().to_le_bytes().to_vec(),
            ScalePrecision::Bf16 => half::bf16::from_f32(0.25).to_bits().to_le_bytes().to_vec(),
            _ => unreachable!(),
        };
        foreign.bytes[..changed.len()].copy_from_slice(&changed);
        std::fs::write(dir.0.join("weight.safetensors"), shard("weight", &foreign)).unwrap();
        assert!(!run(&args).status.success());
        assert!(!dir.0.join("foreign-source").exists());
        std::fs::write(
            dir.0.join("weight.safetensors"),
            shard("weight", &raw.tensors["weight"]),
        )
        .unwrap();
        args[i + 1] = dir.0.join("corrupt-cache").into();
        std::fs::write(dir.0.join("cache.trit"), b"corrupt .trit").unwrap();
        assert!(!run(&args).status.success());
        assert!(!dir.0.join("corrupt-cache").exists());
        args.truncate(args.len() - 2);
        let i = args.iter().position(|v| v == "--out").unwrap();
        args[i + 1] = dir.0.join("missing").into();
        assert!(!run(&args).status.success());
        assert!(!dir.0.join("missing").exists());
    }
}

#[test]
fn actual_cli_and_reference_match_in_distinct_rayon_pools() {
    for threads in [1, 3] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                actual_plain_filesystem_conversion_matches_engine_for_all_original_precisions();
                actual_calibrated_filesystem_conversion_matches_in_memory_engine();
                actual_raw_gram_bits_are_bound_and_swaps_reject_without_publication();
            });
    }
}

#[test]
fn actual_raw_gram_bits_are_bound_and_swaps_reject_without_publication() {
    let dir = Scratch::new();
    let mut raw = source(ScalePrecision::F32);
    let mut request = plan(&raw);
    let mut gram = vec![0.; 1024];
    for i in 0..32 {
        gram[i * 32 + i] = 1. + i as f64 / 100.;
    }
    let weights: Vec<f32> = raw.tensors["weight"]
        .bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let fit = fit_direct_levels_recorded(
        "weight",
        &direct(),
        &weights,
        Some(&gram),
        &mut recorder(),
        "probe",
        None,
        0,
    )
    .unwrap();
    if let QuantAction::Fit { metric, .. } = &mut request.tensors[1].action {
        *metric = fit.observation.observation().metric.clone();
    }
    std::fs::write(
        dir.0.join("plan.json"),
        serde_json::to_vec(&request).unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.0.join("gram.bin"),
        gram.iter()
            .flat_map(|g| g.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let mut args = arguments(&dir.0, &raw, false, "output");
    args.extend([
        "--input-gram".into(),
        format!("weight={}", dir.0.join("gram.bin").display()).into(),
    ]);
    raw.grams.insert("weight".into(), gram.clone());
    let mut sink = MemoryTensorSink::default();
    Engine::run(&request, &mut raw, &mut sink, recorder()).unwrap();
    check(&dir.0, &args, &sink);
    gram[0] += 0.1;
    std::fs::write(
        dir.0.join("gram.bin"),
        gram.iter()
            .flat_map(|g| g.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let i = args.iter().position(|v| v == "--out").unwrap();
    args[i + 1] = dir.0.join("swapped").into();
    assert!(!run(&args).status.success());
    assert!(!dir.0.join("swapped").exists());
}

#[test]
fn invalid_files_and_bindings_reject_before_publication() {
    let dir = Scratch::new();
    let raw = source(ScalePrecision::F32);
    let request = plan(&raw);
    let plan_bytes = serde_json::to_vec(&request).unwrap();
    std::fs::write(dir.0.join("plan.json"), &plan_bytes).unwrap();
    let base = arguments(&dir.0, &raw, false, "output");
    for extra in [
        vec!["--max-plan-bytes".into(), "1".into()],
        vec!["--max-plan-bytes".into(), "0".into()],
        vec![
            "--asset".into(),
            format!("config.json={}", dir.0.join("config.json").display()).into(),
        ],
        vec![
            "--asset".into(),
            format!("extra.json={}", dir.0.join("config.json").display()).into(),
        ],
        vec![
            "--input-gram".into(),
            format!("weight={}", dir.0.join("config.json").display()).into(),
        ],
        vec![
            "--source-shard".into(),
            dir.0.join("weight.safetensors").into(),
        ],
    ] {
        let mut args = base.clone();
        args.extend(extra);
        assert!(!run(&args).status.success());
        assert!(!dir.0.join("output").exists());
    }
    let mut unknown: serde_json::Value = serde_json::from_slice(&plan_bytes).unwrap();
    unknown["unknown"] = true.into();
    std::fs::write(
        dir.0.join("plan.json"),
        serde_json::to_vec(&unknown).unwrap(),
    )
    .unwrap();
    assert!(!run(&base).status.success());
    assert!(!dir.0.join("output").exists());
    std::fs::write(dir.0.join("plan.json"), plan_bytes).unwrap();
    std::fs::write(dir.0.join("config.json"), b"oversize").unwrap();
    assert!(!run(&base).status.success());
    assert!(!dir.0.join("output").exists());
    std::fs::write(dir.0.join("config.json"), b"{}").unwrap();
    std::fs::write(dir.0.join("weight.safetensors"), b"truncated").unwrap();
    assert!(!run(&base).status.success());
    assert!(!dir.0.join("output").exists());
}
