//! Continuous-batching admission gate (model + GPU gated, `cuda` feature):
//! [`copy_kv_into_batch_row`] must reproduce the single-sequence arena
//! BIT-FOR-BIT in the slot — every layer, every row, K and V.
#![cfg(feature = "cuda")]
use std::path::Path;

/// Model cache root: override via `TRITIUM_MODEL_DIR`; default `~/.cache/tritium-models`; tests skip cleanly when absent.
static GGUF_PATH: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    let dir = std::env::var("TRITIUM_MODEL_DIR").unwrap_or_else(|_| {
        format!(
            "{}/.cache/tritium-models",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    format!("{dir}/bitnet-2b4t-gguf/ggml-model-i2_s.gguf")
});

/// Both tests here load a full model (~2.5 GB VRAM) — serialize them within
/// this binary (same OOM-flake pattern the acceptance suite guards against).
static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu_serial() -> std::sync::MutexGuard<'static, ()> {
    GPU_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[test]
fn adopt_copy_is_bit_exact() {
    let _gpu = gpu_serial();
    if !Path::new(&*GGUF_PATH).exists() {
        eprintln!("skipping: {} absent (gated real-model test)", *GGUF_PATH);
        return;
    }
    let bytes = std::fs::read(&*GGUF_PATH).expect("read");
    let file = tritium_format::read_gguf(&bytes).expect("parse");
    let init = tritium_runtime::BACKENDS
        .iter()
        .find(|e| e.name == "cuda")
        .expect("cuda")
        .init;
    let Ok(backend) = init() else { return };
    let mut runner = tritium_nn::ModelRunner::load(&file, &bytes, backend).expect("load");
    let prompt: Vec<u32> = [128000u32, 791, 6864, 315, 9822, 374]
        .iter()
        .cycle()
        .take(26)
        .copied()
        .collect();
    let positions: Vec<usize> = (0..prompt.len()).collect();
    runner.forward(&prompt, &positions).expect("prefill");
    let rm = runner.resident_cuda().expect("resident").expect("cuda");
    let mut batch = rm.new_batch(2).expect("batch");
    rm.copy_kv_into_batch_row(&mut batch, 1, 26).expect("adopt");
    let layers = 30usize;
    for li in 0..layers {
        for row in 0..26usize {
            for v in [false, true] {
                let a = rm.debug_kv_row(li, row, v).expect("src");
                let b = rm.debug_batch_kv_row(&batch, li, 1, row, v).expect("dst");
                assert_eq!(a, b, "layer {li} row {row} v={v} differs after adoption");
            }
        }
    }
    println!("adoption copy bit-exact: 30 layers x 26 rows x K/V");
}

/// A capture that fails mid-body must not poison the capture stream: the
/// next graph capture + replay (and everything after) has to work.
#[test]
fn failed_capture_does_not_poison_the_stream() {
    let _gpu = gpu_serial();
    if !Path::new(&*GGUF_PATH).exists() {
        eprintln!("skipping: {} absent (gated real-model test)", *GGUF_PATH);
        return;
    }
    let bytes = std::fs::read(&*GGUF_PATH).expect("read");
    let file = tritium_format::read_gguf(&bytes).expect("parse");
    let init = tritium_runtime::BACKENDS
        .iter()
        .find(|e| e.name == "cuda")
        .expect("cuda")
        .init;
    let Ok(backend) = init() else { return };
    let mut runner = tritium_nn::ModelRunner::load(&file, &bytes, backend).expect("load");
    let prompt = [128000u32, 791, 6864, 315, 9822, 374];
    let positions: Vec<usize> = (0..prompt.len()).collect();

    // Inject a failed capture BEFORE the decode graph exists, then decode:
    // graph capture + replay must succeed on the recovered stream.
    {
        let rm = runner.resident_cuda().expect("r").expect("c");
        rm.debug_fail_capture().expect("injected failure handled");
    }
    let logits = runner.forward(&prompt, &positions).expect("prefill");
    let t0 = tritium_nn::sample_greedy(&logits).expect("token");
    let logits = runner
        .forward(&[t0], &[prompt.len()])
        .expect("step_graph after failed capture");
    let _ = tritium_nn::sample_greedy(&logits).expect("token");

    // And again with the graph already captured (replay path).
    {
        let rm = runner.resident_cuda().expect("r").expect("c");
        rm.debug_fail_capture()
            .expect("second injected failure handled");
    }
    let logits = runner
        .forward(&[t0], &[prompt.len() + 1])
        .expect("step_graph after second failed capture");
    let _ = tritium_nn::sample_greedy(&logits).expect("token");
    println!("stream survived two injected capture failures");
}
