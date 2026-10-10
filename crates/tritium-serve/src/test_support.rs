//! Shared tiny physical CUDA fixture; never a model/release qualification.

use tritium_nn::{
    Mlp, ModelConfig, ModelRunner, ModelWeights, Projection, Relu2Mlp, TernaryLinear,
    TokenEmbedding, TransformerBlock,
};

// Each fixture models one worker's sequential ownership. CudaBackend instances
// currently share the device's primary context and default stream: independent
// test threads can invalidate another fixture's graph capture. Keep graphs
// enabled while isolating fixture lifetimes, including destruction. This does
// not qualify concurrent models sharing that context (see the evidence note).
pub(crate) fn cuda_fixture_guard() -> std::sync::MutexGuard<'static, ()> {
    static DEVICE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    DEVICE
        .lock()
        .expect("CUDA fixture failed while holding device ownership")
}

pub(crate) fn tiny_cuda_runner(n_ctx: u32) -> Option<ModelRunner> {
    let backend = match tritium_cuda::CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            assert!(
                std::env::var("TRITIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                "required CUDA fixture cannot initialize: {error}"
            );
            eprintln!("UNKNOWN: CUDA fixture unavailable ({error})");
            return None;
        }
    };
    let projection = |rows, cols, seed| {
        let trits: Vec<_> = (0..rows * cols)
            .map(|i| tritium_core::Trit::from_i8(((i * 7 + seed) % 3) as i8 - 1).unwrap())
            .collect();
        Projection::Ternary(TernaryLinear::new(&backend, &trits, rows, cols, 0.03125).unwrap())
    };
    let config = ModelConfig {
        arch: "bitnet".into(),
        n_layers: 2,
        n_embd: 64,
        n_head: 2,
        n_head_kv: 1,
        head_dim: 32,
        n_ff: 64,
        n_ctx,
        rope_theta: 10_000.0,
        rms_eps: 1e-5,
    };
    let weights = ModelWeights {
        token_embd: TokenEmbedding::from_dense(
            (0..512)
                .map(|i| ((i * 5 % 23) as f32 - 11.0) / 64.0)
                .collect(),
            8,
            64,
        )
        .unwrap(),
        vocab: 8,
        n_embd: 64,
        layers: (0..2)
            .map(|li| TransformerBlock {
                attn_norm: vec![1.0; 64],
                q_proj: projection(64, 64, li + 1),
                k_proj: projection(32, 64, li + 2),
                v_proj: projection(32, 64, li + 3),
                o_proj: projection(64, 64, li + 4),
                attn_sub_norm: Vec::new(),
                q_bias: Vec::new(),
                k_bias: Vec::new(),
                v_bias: Vec::new(),
                q_norm: Vec::new(),
                k_norm: Vec::new(),
                ffn_norm: vec![1.0; 64],
                mlp: Mlp::Relu2(Relu2Mlp {
                    gate: projection(64, 64, li + 5),
                    up: projection(64, 64, li + 6),
                    down: projection(64, 64, li + 7),
                    ffn_sub_norm: Vec::new(),
                    rms_eps: 1e-5,
                }),
            })
            .collect(),
        output_norm: vec![1.0; 64],
        lm_head: None,
    };
    Some(ModelRunner::from_weights(
        config,
        weights,
        Box::new(backend),
    ))
}

pub(crate) fn prefix_bytes(runner: &mut ModelRunner) -> Vec<Vec<u8>> {
    let model = &*runner
        .resident_cuda()
        .unwrap()
        .expect("fixture is resident");
    (0..2)
        .flat_map(|layer| {
            (0..model.cache_len())
                .flat_map(move |row| [false, true].map(move |value| (layer, row, value)))
        })
        .map(|(layer, row, value)| model.debug_kv_row(layer, row, value).unwrap())
        .collect()
}
