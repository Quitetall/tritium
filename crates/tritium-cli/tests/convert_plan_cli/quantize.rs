//! Two actual command receivers and the same engine, with no alternate fitter.
use super::*;

fn output(args: &mut [OsString], path: &Path) {
    let i = args.iter().position(|value| value == "--out").unwrap();
    args[i + 1] = path.into();
}
fn reject(args: &[OsString], path: &Path) {
    let result = run(args);
    assert!(
        !result.status.success(),
        "unexpected receiver success for {args:?}"
    );
    assert!(result.stdout.is_empty());
    assert!(!String::from_utf8_lossy(&result.stderr).contains("panicked"));
    assert!(!path.exists());
}

fn raw_receivers() {
    for precision in [
        ScalePrecision::F32,
        ScalePrecision::F16,
        ScalePrecision::Bf16,
    ] {
        for cached in [false, true] {
            let dir = Scratch::new();
            let mut raw = source(precision);
            let mut sink = MemoryTensorSink::default();
            let cache = if cached {
                let (request, cache) = calibrated(&mut raw);
                std::fs::write(
                    dir.0.join("plan.json"),
                    serde_json::to_vec(&request).unwrap(),
                )
                .unwrap();
                let cache_bytes = cache.encode_package(100_000).unwrap();
                std::fs::write(dir.0.join("cache.trit"), &cache_bytes).unwrap();
                let mut caches = InMemoryCalibrationSource::new([cache]).unwrap();
                Engine::run_calibrated(&request, &mut raw, &mut caches, &mut sink, recorder())
                    .unwrap();
                Some(cache_bytes)
            } else {
                let request = plan(&raw);
                std::fs::write(
                    dir.0.join("plan.json"),
                    serde_json::to_vec(&request).unwrap(),
                )
                .unwrap();
                Engine::run(&request, &mut raw, &mut sink, recorder()).unwrap();
                None
            };
            let mut args = arguments(&dir.0, &raw, cached, "converted");
            if cached {
                args.extend([
                    "--calibration-cache".into(),
                    format!("weight={}", dir.0.join("cache.trit").display()).into(),
                ]);
            }
            check(&dir.0, &args, &sink);
            args[0] = "quantize".into();
            output(&mut args, &dir.0.join("quantized"));
            // Disabling the outer lifecycle summary must not disable fit evidence.
            args.extend(["--evidence".into(), "off".into()]);
            check(&dir.0, &args, &sink);
            for name in ["model.trit", "evidence.jsonl"] {
                assert_eq!(
                    std::fs::read(dir.0.join("converted").join(name)).unwrap(),
                    std::fs::read(dir.0.join("quantized").join(name)).unwrap()
                );
            }
            let before = std::fs::read(dir.0.join("quantized/model.trit")).unwrap();
            assert!(!run(&args).status.success());
            assert_eq!(
                before,
                std::fs::read(dir.0.join("quantized/model.trit")).unwrap()
            );
            output(&mut args, &dir.0.join("again"));
            check(&dir.0, &args, &sink);
            if cached
                && precision == ScalePrecision::Bf16
                && let Some(destination) = std::env::var_os("TRITIUM_TYPED_QUANTIZE_OUTPUT")
            {
                let destination = PathBuf::from(destination);
                for entry in std::fs::read_dir(&dir.0).unwrap() {
                    let entry = entry.unwrap();
                    if entry.file_type().unwrap().is_file() {
                        std::fs::copy(entry.path(), destination.join(entry.file_name())).unwrap();
                    }
                }
                for (folder, prefix) in [("converted", "convert"), ("quantized", "quantize")] {
                    for name in ["model.trit", "evidence.jsonl"] {
                        std::fs::copy(
                            dir.0.join(folder).join(name),
                            destination.join(format!("{prefix}-{name}")),
                        )
                        .unwrap();
                    }
                }
                let (package, evidence) = sink.artifact().unwrap();
                std::fs::write(destination.join("engine-model.trit"), package).unwrap();
                std::fs::write(destination.join("engine-evidence.jsonl"), evidence.jsonl())
                    .unwrap();
            }
            for (flag, value) in [
                ("--max-plan-bytes", "0"),
                ("--max-plan-bytes", "1"),
                ("--asset", "extra=missing"),
                ("--source-shard", "missing"),
                ("--input-gram", "extra=missing"),
            ] {
                let mut bad = args.clone();
                output(&mut bad, &dir.0.join("rejected"));
                bad.extend([flag.into(), value.into()]);
                reject(&bad, &dir.0.join("rejected"));
            }
            output(&mut args, &dir.0.join("rejected"));
            if let Some(cache) = cache {
                for bytes in [b"corrupt".as_slice(), &[]] {
                    std::fs::write(dir.0.join("cache.trit"), bytes).unwrap();
                    reject(&args, &dir.0.join("rejected"));
                }
                std::fs::write(dir.0.join("cache.trit"), cache).unwrap();
            }
            let mut changed = raw.tensors["weight"].clone();
            changed.bytes[0] ^= 1;
            std::fs::write(dir.0.join("weight.safetensors"), shard("weight", &changed)).unwrap();
            if cached {
                // Calibration freezes the original source; its old cache cannot
                // be applied to the changed actual weight bytes.
                reject(&args, &dir.0.join("rejected"));
            } else {
                // A plain plan binds interpretation/recipe, not a frozen weight
                // identity. This is a new producer input, never substitution
                // into the already published immutable package.
                let request = plan(&raw);
                let original = Engine::observe_source(&request, &mut raw).unwrap();
                raw.tensors.insert("weight".into(), changed);
                let observed = Engine::observe_source(&request, &mut raw).unwrap();
                assert_ne!(original, observed);
                let mut changed_sink = MemoryTensorSink::default();
                Engine::run(&request, &mut raw, &mut changed_sink, recorder()).unwrap();
                assert_ne!(
                    sink.artifact().unwrap().0,
                    changed_sink.artifact().unwrap().0
                );
                assert_ne!(
                    sink.artifact().unwrap().1.log().root_digest,
                    changed_sink.artifact().unwrap().1.log().root_digest
                );
                output(&mut args, &dir.0.join("changed-source"));
                check(&dir.0, &args, &changed_sink);
                args[0] = "convert".into();
                output(&mut args, &dir.0.join("changed-source-convert"));
                check(&dir.0, &args, &changed_sink);
                assert_eq!(
                    before,
                    std::fs::read(dir.0.join("quantized/model.trit")).unwrap()
                );
            }
        }
    }
}

#[test]
fn typed_quantize_plain_and_calibrated_match_convert_and_engine_in_actual_pools() {
    for threads in [1, 3] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(raw_receivers);
    }
}

#[test]
fn typed_quantize_binds_actual_raw_gram_bits_and_rejects_substitution() {
    let dir = Scratch::new();
    let mut raw = source(ScalePrecision::F32);
    let mut request = plan(&raw);
    let gram: Vec<f64> = (0..1024)
        .map(|i| {
            if i / 32 == i % 32 {
                1. + (i / 32) as f64 / 100.
            } else {
                0.
            }
        })
        .collect();
    let weights: Vec<f32> = raw.tensors["weight"]
        .bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bits| f32::from_le_bytes(*bits))
        .collect();
    let fitted = fit_direct_levels_recorded(
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
        *metric = fitted.observation.observation().metric.clone();
    }
    std::fs::write(
        dir.0.join("plan.json"),
        serde_json::to_vec(&request).unwrap(),
    )
    .unwrap();
    let bits: Vec<u8> = gram.iter().flat_map(|value| value.to_le_bytes()).collect();
    std::fs::write(dir.0.join("gram.bin"), &bits).unwrap();
    let mut args = arguments(&dir.0, &raw, false, "quantized");
    args[0] = "quantize".into();
    args.extend([
        "--input-gram".into(),
        format!("weight={}", dir.0.join("gram.bin").display()).into(),
    ]);
    raw.grams.insert("weight".into(), gram);
    let mut sink = MemoryTensorSink::default();
    Engine::run(&request, &mut raw, &mut sink, recorder()).unwrap();
    check(&dir.0, &args, &sink);
    let mut wrong = bits;
    wrong[0] ^= 1;
    std::fs::write(dir.0.join("gram.bin"), &wrong).unwrap();
    output(&mut args, &dir.0.join("rejected"));
    reject(&args, &dir.0.join("rejected"));
}
