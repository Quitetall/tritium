//! Real Rust-generated plans/caches exercised through the source abi3 receiver.
use super::*;

#[test]
#[ignore = "requires this checkout's built abi3 extension and Python pytest"]
fn actual_python_file_conversion_matches_core_and_cli() {
    let root = Scratch::new();
    for precision in [
        ScalePrecision::F16,
        ScalePrecision::Bf16,
        ScalePrecision::F32,
    ] {
        for mode in ["direct", "gram", "calibrated"] {
            let dir = root.0.join(format!("{precision:?}-{mode}"));
            std::fs::create_dir(&dir).unwrap();
            let mut raw = source(precision);
            let mut sink = MemoryTensorSink::default();
            let (json, cache) = if mode == "calibrated" {
                let (request, cache) = calibrated(&mut raw);
                let bytes = cache.encode_package(100_000).unwrap();
                let mut caches = InMemoryCalibrationSource::new([cache]).unwrap();
                Engine::run_calibrated(&request, &mut raw, &mut caches, &mut sink, recorder())
                    .unwrap();
                (serde_json::to_vec(&request).unwrap(), Some(bytes))
            } else {
                let mut request = plan(&raw);
                if mode == "gram" {
                    let mut gram = vec![0.0f64; 1024];
                    for i in 0..32 {
                        gram[i * 32 + i] = 1.0 + i as f64 / 100.0;
                    }
                    let weights = tritium_format::trit_model::decode_dense_f32(
                        "weight",
                        &[1, 32],
                        precision,
                        &raw.tensors["weight"].bytes,
                    )
                    .unwrap();
                    let QuantAction::Fit { plan, metric } = &mut request.tensors[1].action else {
                        panic!("fit");
                    };
                    plan.layout.basis = Basis::SignedRht {
                        block: 32,
                        seed: 123,
                        domain: 456,
                    };
                    let probe = fit_direct_levels_recorded(
                        "weight",
                        plan,
                        &weights,
                        Some(&gram),
                        &mut recorder(),
                        "probe",
                        None,
                        0,
                    )
                    .unwrap();
                    *metric = probe.observation.observation().metric.clone();
                    std::fs::write(
                        dir.join("gram.bin"),
                        gram.iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                    raw.grams.insert("weight".into(), gram);
                }
                Engine::run(&request, &mut raw, &mut sink, recorder()).unwrap();
                (serde_json::to_vec(&request).unwrap(), None)
            };
            std::fs::write(dir.join("plan.json"), json).unwrap();
            let mut args = arguments(&dir, &raw, mode == "calibrated", "cli");
            if let Some(bytes) = cache {
                std::fs::write(dir.join("cache.trit"), bytes).unwrap();
                args.extend([
                    "--calibration-cache".into(),
                    format!("weight={}", dir.join("cache.trit").display()).into(),
                ]);
            }
            if mode == "gram" {
                args.extend([
                    "--input-gram".into(),
                    format!("weight={}", dir.join("gram.bin").display()).into(),
                ]);
            }
            check(&dir, &args, &sink);
            std::fs::write(dir.join("norm.raw"), &raw.tensors["norm"].bytes).unwrap();
        }
    }
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let python = std::env::var_os("TRITIUM_NATIVE_TEST_PYTHON").unwrap_or_else(|| "python".into());
    let mut child = Command::new(python)
        .args([
            "-I",
            "-m",
            "pytest",
            "-q",
            "crates/tritium-py/tests/test_model_conversion.py",
        ])
        .current_dir(repo)
        .env("TRITIUM_MODEL_FIXTURES", &root.0)
        .env(
            "RAYON_NUM_THREADS",
            rayon::current_num_threads().to_string(),
        )
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .stdin(Stdio::null())
        .spawn()
        .unwrap();
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "native Python conversion probe failed");
            break;
        }
        if start.elapsed() > Duration::from_secs(120) {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("native Python probe exceeded 120 seconds");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}
