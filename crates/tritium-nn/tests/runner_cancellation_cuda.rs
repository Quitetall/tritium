//! Tiny physical CUDA cancellation checks, not model or release qualification.
#![cfg(feature = "cuda")]

use std::cell::Cell;

use tritium_core::Trit;
use tritium_nn::{
    Mlp, ModelConfig, ModelRunner, ModelWeights, Projection, Relu2Mlp, TernaryLinear,
    TokenEmbedding, TransformerBlock,
};

fn runner() -> Option<ModelRunner> {
    let backend = match tritium_cuda::CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            assert!(
                std::env::var("TRITIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                "required CUDA fixture cannot initialize: {error}"
            );
            eprintln!("UNKNOWN: CUDA unavailable ({error})");
            return None;
        }
    };
    let projection = |rows, cols, seed| {
        let trits: Vec<_> = (0..rows * cols)
            .map(|i| Trit::from_i8(((i * 7 + seed) % 3) as i8 - 1).unwrap())
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
        n_ctx: 16,
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

fn prefix_bytes(model: &tritium_cuda::CudaDecodeModel) -> Vec<Vec<u8>> {
    (0..2)
        .flat_map(|layer| {
            (0..model.cache_len()).flat_map(move |row| {
                [false, true]
                    .into_iter()
                    .map(move |value| model.debug_kv_row(layer, row, value).unwrap())
            })
        })
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn resident_prefill_cancellation_preserves_prefix_and_recovers() {
    let Some(mut runner) = runner() else { return };
    let model = runner
        .resident_cuda()
        .unwrap()
        .expect("fixture is resident");
    for seed in [&[][..], &[0, 1, 2][..]] {
        let seed_positions: Vec<_> = (0..seed.len()).collect();
        let tokens = [3, 4];
        let positions = [seed.len(), seed.len() + 1];
        let prepare = |model: &mut tritium_cuda::CudaDecodeModel| {
            model.reset();
            if !seed.is_empty() {
                model.prefill(seed, &seed_positions).unwrap();
            }
        };
        prepare(model);
        let expected = model.prefill(&tokens, &positions).unwrap();
        prepare(model);
        let checks = Cell::new(0);
        let actual = model
            .prefill_cancellable(&tokens, &positions, &|| {
                checks.set(checks.get() + 1);
                false
            })
            .unwrap()
            .unwrap();
        assert_eq!(bits(&actual), bits(&expected));
        assert!(checks.get() >= 5, "native launch groups need checkpoints");
        for cancel_at in 1..=checks.get() {
            prepare(model);
            let before = prefix_bytes(model);
            let count = Cell::new(0);
            assert!(
                model
                    .prefill_cancellable(&tokens, &positions, &|| {
                        count.set(count.get() + 1);
                        count.get() == cancel_at
                    })
                    .unwrap()
                    .is_none()
            );
            assert_eq!(model.cache_len(), seed.len());
            assert_eq!(prefix_bytes(model), before);
            let recovered = model.prefill(&tokens, &positions).unwrap();
            assert_eq!(bits(&recovered), bits(&expected));
        }
    }
}

#[test]
fn resident_graph_cancellation_rewinds_without_host_fallback() {
    let Some(mut runner) = runner() else { return };
    let seed = [0, 1, 2];
    let positions = [0, 1, 2];
    runner.forward(&seed, &positions).unwrap();
    let expected = runner.forward(&[3], &[3]).unwrap();
    runner.reset();
    runner.forward(&seed, &positions).unwrap();
    let before = prefix_bytes(runner.resident_cuda().unwrap().unwrap());
    let count = Cell::new(0);
    assert!(
        runner
            .forward_cancellable(&[3], &[3], &|| {
                count.set(count.get() + 1);
                count.get() == 3
            })
            .unwrap()
            .is_none()
    );
    let resident = runner.resident_cuda().unwrap().unwrap();
    assert_eq!(resident.cache_len(), 3);
    assert_eq!(prefix_bytes(resident), before);
    assert!(runner.kv.iter().all(|cache| cache.len == 0));
    assert_eq!(bits(&runner.forward(&[3], &[3]).unwrap()), bits(&expected));
}
