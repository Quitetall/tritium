//! End-to-end validation of the I2_S decoder against the REAL official BitNet 2B4T
//! GGUF (`microsoft/bitnet-b1.58-2B-4T-gguf`, `ggml-model-i2_s.gguf`).
//!
//! This is gated on the model being present on disk: if the file is absent the test
//! is skipped (returns early) rather than failing, so the offline CI lane stays
//! green. WF-1 confirmed the file is downloaded, so it runs here.
//!
//! What it pins (all confirmed in WF-1 against the file + the reference HF checkpoint
//! `microsoft/bitnet-b1.58-2B-4T`):
//! - the ternary weight tensors' ggml type-id is exactly `GGML_TYPE_I2_S` (36);
//! - each I2_S tensor payload is `n_elements/4` quant bytes + a single trailing `f32`
//!   per-tensor scale (the next tensor begins 32-aligned after that);
//! - decoding via `unpack_i2s_tensor` yields only valid trits `{-1,0,+1}`, and the
//!   recovered scale is a sane positive magnitude.

use std::path::Path;

use tritium_core::Trit;
use tritium_format::{GGML_TYPE_I2_S, read_gguf, unpack_i2s_tensor};

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

#[test]
fn decodes_real_bitnet_i2s_tensor() {
    if !Path::new(&*GGUF_PATH).exists() {
        eprintln!(
            "skipping: {} not present (gated real-file test)",
            *GGUF_PATH
        );
        return;
    }

    let bytes = std::fs::read(&*GGUF_PATH).expect("read real GGUF");
    let f = read_gguf(&bytes).expect("parse real GGUF");

    assert_eq!(
        f.get_metadata("general.architecture")
            .and_then(|v| v.as_str()),
        Some("bitnet-b1.58"),
        "architecture metadata"
    );

    // Every ternary weight tensor must carry the confirmed I2_S type-id, and there
    // are many of them (210 in this checkpoint).
    let i2s_tensors: Vec<_> = f
        .tensors
        .iter()
        .filter(|t| t.ggml_type == GGML_TYPE_I2_S)
        .collect();
    assert!(
        i2s_tensors.len() >= 200,
        "expected ~210 I2_S tensors, found {}",
        i2s_tensors.len()
    );

    // Decode `blk.0.attn_q.weight` in full and validate it end-to-end.
    let t = f
        .tensor("blk.0.attn_q.weight")
        .expect("blk.0.attn_q.weight present");
    assert_eq!(t.ggml_type, GGML_TYPE_I2_S);

    let n_elements = t.element_count().expect("element count") as usize;
    assert_eq!(n_elements, 2560 * 2560);

    // Locate the payload: data section + this tensor's relative offset. The payload
    // length is the I2_S size we expect (quants + 4-byte scale), well within the
    // remaining bytes; pass that exact slice to the decoder.
    let start = (f.tensor_data_offset + t.offset) as usize;
    let payload_len = n_elements / 4 + 4;
    let payload = &bytes[start..start + payload_len];

    let mut trits = vec![Trit::ZERO; n_elements];
    let scale = unpack_i2s_tensor(payload, n_elements, &mut trits).expect("decode I2_S tensor");

    // Scale is a positive magnitude (this tensor's is ~1.2188548).
    assert!(scale.is_finite() && scale > 0.0, "scale = {scale}");
    assert!(
        (scale - 1.218_854_8).abs() < 1e-4,
        "scale {scale} != expected ~1.2188548"
    );

    // Every decoded value is a valid trit, and all three values occur (it is a real
    // dense weight matrix, not a degenerate all-zero block).
    let mut counts = [0usize; 3]; // [-1, 0, +1]
    for tr in &trits {
        match tr.get() {
            -1 => counts[0] += 1,
            0 => counts[1] += 1,
            1 => counts[2] += 1,
            other => panic!("non-ternary decoded value {other}"),
        }
    }
    assert!(
        counts[0] > 0 && counts[1] > 0 && counts[2] > 0,
        "{counts:?}"
    );
    assert_eq!(counts.iter().sum::<usize>(), n_elements);

    // The leading trits match `microsoft/bitnet-b1.58-2B-4T`'s unpacked
    // `q_proj.weight` row 0 (validated bit-exactly against the HF checkpoint in
    // WF-4), under the `trit = code - 1` mapping. Spot-check the first 16.
    let want_trits = [0i8, 0, 0, -1, 0, -1, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0];
    for (i, &w) in want_trits.iter().enumerate() {
        assert_eq!(trits[i].get(), w, "trit[{i}] mismatch vs HF golden");
    }
    let deq: Vec<f32> = trits[..16].iter().map(|t| t.to_f32() * scale).collect();
    for (i, (&g, &w)) in deq.iter().zip(want_trits.iter()).enumerate() {
        assert!(
            (g - w as f32 * scale).abs() < 1e-6,
            "dequant[{i}] {g} != {}",
            w as f32 * scale
        );
    }
}
