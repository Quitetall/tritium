//! `ModelConfig::from_gguf` against the real BitNet-shaped GGUF fixture produced by
//! the official `gguf` writer (lives in the sibling tritium-format crate).

use tritium_format::read_gguf;
use tritium_nn::ModelConfig;

#[test]
fn model_config_from_real_gguf() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tritium-format/tests/fixtures/bitnet_tiny.gguf"
    );
    let bytes = std::fs::read(path).expect("read fixture");
    let file = read_gguf(&bytes).expect("parse fixture");
    let c = ModelConfig::from_gguf(&file).expect("config from fixture");

    assert_eq!(c.arch, "bitnet");
    assert_eq!(c.n_layers, 1);
    assert_eq!(c.n_embd, 256);
    assert_eq!(c.n_head, 4);
    assert_eq!(c.n_head_kv, 2);
    assert_eq!(c.n_ff, 64);
    assert_eq!(c.n_ctx, 4096);
    assert_eq!(c.head_dim(), 64); // 256 / 4
    assert_eq!(c.gqa_group(), 2); // 4 / 2
    assert!((c.rope_theta - 500_000.0).abs() < 1.0);
    assert!((c.rms_eps - 1e-5).abs() < 1e-9);
}
