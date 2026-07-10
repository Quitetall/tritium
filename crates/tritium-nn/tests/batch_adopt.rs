//! Continuous-batching admission gate (model + GPU gated, `cuda` feature):
//! [`copy_kv_into_batch_row`] must reproduce the single-sequence arena
//! BIT-FOR-BIT in the slot — every layer, every row, K and V.
#![cfg(feature = "cuda")]
use std::path::Path;

const GGUF_PATH: &str =
    "/home/brianklam/.cache/tritium-models/bitnet-2b4t-gguf/ggml-model-i2_s.gguf";

#[test]
fn adopt_copy_is_bit_exact() {
    if !Path::new(GGUF_PATH).exists() {
        eprintln!("skipping: {GGUF_PATH} absent (gated real-model test)");
        return;
    }
    let bytes = std::fs::read(GGUF_PATH).expect("read");
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
