//! Validates the GGUF reader against a REAL ggml-format file produced by the
//! official `gguf` Python writer (the same one llama.cpp ships). The fixture holds
//! TQ2_0 / TQ1_0 / F16 / F32 tensors and BitNet-style metadata; the expected values
//! below were emitted by `gguf.GGUFReader` reading the same bytes back, so this
//! test pins our reader to the official format. (Generator: tools/gen_gguf_fixture.py.)

use tritium_format::{GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, read_gguf};

const FIXTURE: &[u8] = include_bytes!("fixtures/bitnet_tiny.gguf");

#[test]
fn parses_official_gguf_writer_output() {
    let f = read_gguf(FIXTURE).expect("real GGUF fixture must parse");

    assert_eq!(f.version, 3);
    assert_eq!(f.alignment(), 32);
    assert_eq!(f.tensor_data_offset, 640);

    // (name, dims [ne order], ggml_type, ABSOLUTE data offset) — from gguf.GGUFReader.
    let expected: &[(&str, &[u64], u32, u64)] = &[
        ("token_embd.weight", &[256, 32], 1, 640), // F16
        ("blk.0.attn_q.weight", &[256, 8], GGML_TYPE_TQ2_0, 17024),
        ("blk.0.ffn_down.weight", &[256, 4], GGML_TYPE_TQ1_0, 17568),
        ("output_norm.weight", &[256], 0, 17792), // F32
    ];
    assert_eq!(f.tensors.len(), expected.len());
    for &(name, dims, ty, abs_off) in expected {
        let t = f
            .tensor(name)
            .unwrap_or_else(|| panic!("missing tensor {name}"));
        assert_eq!(t.dims, dims, "{name} dims");
        assert_eq!(t.ggml_type, ty, "{name} ggml_type");
        assert_eq!(
            f.tensor_data_offset + t.offset,
            abs_off,
            "{name} absolute offset"
        );
    }

    // Metadata our ModelConfig consumes.
    assert_eq!(
        f.get_metadata("general.architecture")
            .and_then(|v| v.as_str()),
        Some("bitnet")
    );
    assert_eq!(
        f.get_metadata("bitnet.embedding_length")
            .and_then(|v| v.as_u64()),
        Some(256)
    );
}
