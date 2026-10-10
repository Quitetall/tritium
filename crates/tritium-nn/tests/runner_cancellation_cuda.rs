//! Tiny physical CUDA cancellation checks, not model or release qualification.
#![cfg(feature = "cuda")]

use std::cell::Cell;

use tritium_core::Trit;
use tritium_nn::{
    Mlp, ModelConfig, ModelRunner, ModelWeights, Projection, Relu2Mlp, TernaryLinear,
    TokenEmbedding, TransformerBlock,
};

fn runner() -> Option<ModelRunner> {
    runner_with_context(16)
}

fn runner_with_context(n_ctx: u32) -> Option<ModelRunner> {
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

#[test]
fn resident_tree_cancellation_stops_before_promotion() {
    let Some(mut runner) = runner() else { return };
    let model = runner
        .resident_cuda()
        .unwrap()
        .expect("fixture is resident");
    model.prefill(&[0, 1, 2], &[0, 1, 2]).unwrap();
    let before = prefix_bytes(model);
    let polls = Cell::new(0);
    let output = model
        .tree_verify_greedy_cancellable(&[3, 4], &[-1, 0], &|| {
            polls.set(polls.get() + 1);
            polls.get() == 2
        })
        .unwrap();
    assert!(
        output.is_none(),
        "entered tree verification ignored cancellation"
    );
    assert_eq!(model.cache_len(), 3, "cancelled tree promoted KV");
    assert_eq!(prefix_bytes(model), before);
}

#[test]
fn resident_tree_all_checkpoints_preserve_prefix_and_recover() {
    // 16 takes the existing captured-graph route. 12289 exceeds its declared
    // 48 KiB context-smem eligibility limit and naturally takes eager; no
    // dispatch kill switch or environment mutation manufactures coverage.
    for context in [16, 12289] {
        let Some(mut runner) = runner_with_context(context) else {
            return;
        };
        let model = runner
            .resident_cuda()
            .unwrap()
            .expect("fixture is resident");
        for seed in [&[][..], &[0, 1, 2][..]] {
            let prepare = |model: &mut tritium_cuda::CudaDecodeModel| {
                model.reset();
                if !seed.is_empty() {
                    model
                        .prefill(seed, &(0..seed.len()).collect::<Vec<_>>())
                        .unwrap();
                }
            };
            let tokens = [3, 4, 5];
            let parents = [-1, 0, 0];
            prepare(model);
            let expected = model.tree_verify_greedy(&tokens, &parents).unwrap();
            let expected_len = model.cache_len();
            let expected_kv = prefix_bytes(model);
            prepare(model);
            let polls = Cell::new(0);
            assert_eq!(
                model
                    .tree_verify_greedy_cancellable(&tokens, &parents, &|| {
                        polls.set(polls.get() + 1);
                        false
                    })
                    .unwrap()
                    .unwrap(),
                expected
            );
            assert_eq!(model.cache_len(), expected_len);
            assert_eq!(prefix_bytes(model), expected_kv);
            assert!(polls.get() >= 6, "native tree needs internal checkpoints");
            assert_eq!(model.tree_graph_bucket_count() > 0, context == 16);
            for cancel_at in 1..=polls.get() {
                prepare(model);
                let before = prefix_bytes(model);
                let count = Cell::new(0);
                assert!(
                    model
                        .tree_verify_greedy_cancellable(&tokens, &parents, &|| {
                            count.set(count.get() + 1);
                            count.get() == cancel_at
                        })
                        .unwrap()
                        .is_none(),
                    "context {context}, checkpoint {cancel_at}"
                );
                assert_eq!(model.cache_len(), seed.len());
                assert_eq!(prefix_bytes(model), before);
                assert_eq!(
                    model.tree_verify_greedy(&tokens, &parents).unwrap(),
                    expected
                );
                assert_eq!(prefix_bytes(model), expected_kv);
            }
        }
    }
}

#[test]
fn resident_tree_logits_cancellation_retires_only_entered_authorization() {
    for context in [16, 12289] {
        let Some(mut runner) = runner_with_context(context) else {
            return;
        };
        let model = runner
            .resident_cuda()
            .unwrap()
            .expect("fixture is resident");
        let prepare = |model: &mut tritium_cuda::CudaDecodeModel| {
            model.reset();
            model.prefill(&[0, 1, 2], &[0, 1, 2]).unwrap();
        };
        let tokens = [3, 4, 5];
        let parents = [-1, 0, 1];
        prepare(model);
        let expected = model.tree_verify_logits(&tokens, &parents).unwrap();
        prepare(model);
        let polls = Cell::new(0);
        let actual = model
            .tree_verify_logits_cancellable(&tokens, &parents, &|| {
                polls.set(polls.get() + 1);
                false
            })
            .unwrap()
            .unwrap();
        assert_eq!(bits(&actual), bits(&expected));
        model.tree_commit(&[0]).unwrap();
        assert_eq!(model.cache_len(), 4);
        for cancel_at in 1..=polls.get() {
            prepare(model);
            let before = prefix_bytes(model);
            // An older pending tree is preserved only if no work begins.
            model.tree_verify_logits(&tokens, &parents).unwrap();
            let count = Cell::new(0);
            assert!(
                model
                    .tree_verify_logits_cancellable(&tokens, &parents, &|| {
                        count.set(count.get() + 1);
                        count.get() == cancel_at
                    })
                    .unwrap()
                    .is_none()
            );
            assert_eq!(model.cache_len(), 3);
            assert_eq!(prefix_bytes(model), before);
            if cancel_at == 1 {
                model.tree_commit(&[0]).unwrap();
                assert_eq!(model.cache_len(), 4);
                prepare(model);
            } else {
                assert!(matches!(
                    model.tree_commit(&[0]),
                    Err(tritium_spec::BackendError::InvalidInput(_))
                ));
            }
            assert_eq!(
                bits(&model.tree_verify_logits(&tokens, &parents).unwrap()),
                bits(&expected)
            );
            model.tree_commit(&[0]).unwrap();
            assert_eq!(model.cache_len(), 4);
        }
        // Validation/backend errors remain errors, not successful cancellation.
        prepare(model);
        assert!(
            model
                .tree_verify_logits_cancellable(&tokens, &[-1], &|| false)
                .is_err()
        );
        assert!(
            model
                .tree_verify_greedy_cancellable(&[8], &[-1], &|| false)
                .is_err()
        );
    }
}

fn batch_prefix_bytes(
    model: &tritium_cuda::CudaDecodeModel,
    batch: &tritium_cuda::BatchKv,
) -> Vec<Vec<u8>> {
    (0..2)
        .flat_map(|layer| {
            (0..2).flat_map(move |slot| {
                (0..batch.positions()[slot])
                    .flat_map(move |row| [false, true].map(move |value| (layer, slot, row, value)))
            })
        })
        .map(|(layer, slot, row, value)| {
            model
                .debug_batch_kv_row(batch, layer, slot, row, value)
                .unwrap()
        })
        .collect()
}

#[test]
fn resident_tree_slot_cancellation_preserves_peers_and_solo_authorization() {
    for context in [16, 12289] {
        let Some(mut runner) = runner_with_context(context) else {
            return;
        };
        let model = runner
            .resident_cuda()
            .unwrap()
            .expect("fixture is resident");
        for paged in [false, true] {
            let mut batch = if paged {
                model.new_batch_paged(2, 2).unwrap()
            } else {
                model.new_batch(2).unwrap()
            };
            model.reset();
            model.prefill(&[0, 1, 2], &[0, 1, 2]).unwrap();
            for slot in 0..2 {
                if paged {
                    batch.reserve_pages(slot, 16).unwrap();
                }
                model.copy_kv_into_batch_row(&mut batch, slot, 3).unwrap();
                batch.set_position(slot, 3).unwrap();
                batch.set_live(slot, true).unwrap();
            }
            let tokens = [3, 4, 5];
            let parents = [-1, 0, 1];
            let before = batch_prefix_bytes(model, &batch);
            let free = batch.free_pages();
            let page_tables = [batch.debug_page_table_row(0), batch.debug_page_table_row(1)];
            let polls = Cell::new(0);
            let expected = model
                .tree_verify_greedy_slot_cancellable(&mut batch, 0, &tokens, &parents, &|| {
                    polls.set(polls.get() + 1);
                    false
                })
                .unwrap()
                .unwrap();
            assert!(polls.get() >= 6);
            let expected_position = batch.positions()[0];
            let expected_kv = batch_prefix_bytes(model, &batch);
            for cancel_at in 1..=polls.get() {
                batch.set_position(0, 3).unwrap();
                model.tree_verify_logits(&tokens, &parents).unwrap();
                let count = Cell::new(0);
                assert!(
                    model
                        .tree_verify_greedy_slot_cancellable(
                            &mut batch,
                            0,
                            &tokens,
                            &parents,
                            &|| {
                                count.set(count.get() + 1);
                                count.get() == cancel_at
                            }
                        )
                        .unwrap()
                        .is_none()
                );
                assert_eq!(batch.positions(), &[3, 3]);
                assert_eq!(batch_prefix_bytes(model, &batch), before);
                assert_eq!(batch.free_pages(), free);
                assert_eq!(
                    [batch.debug_page_table_row(0), batch.debug_page_table_row(1)],
                    page_tables
                );
                // Slot work must not revoke the unrelated solo tree's authority.
                model.tree_commit(&[0]).unwrap();
                assert_eq!(model.cache_len(), 4);
                model.truncate_kv(3).unwrap();
                let recovered = model
                    .tree_verify_greedy_slot(&mut batch, 0, &tokens, &parents)
                    .unwrap();
                assert_eq!(recovered, expected);
                assert_eq!(batch.positions()[0], expected_position);
                assert_eq!(batch.positions()[1], 3);
                assert_eq!(batch_prefix_bytes(model, &batch), expected_kv);
            }
        }
    }
}

#[test]
fn resident_tree_cold_capture_cancellation_reuses_scratch_safely() {
    for cancel_at in 3..=7 {
        let Some(mut runner) = runner() else { return };
        let model = runner
            .resident_cuda()
            .unwrap()
            .expect("fixture is resident");
        model.prefill(&[0, 1, 2], &[0, 1, 2]).unwrap();
        assert_eq!(model.tree_graph_bucket_count(), 0);
        let before = prefix_bytes(model);
        let count = Cell::new(0);
        assert!(
            model
                .tree_verify_greedy_cancellable(&[3, 4], &[-1, 0], &|| {
                    count.set(count.get() + 1);
                    count.get() == cancel_at
                })
                .unwrap()
                .is_none()
        );
        assert_eq!(model.cache_len(), 3);
        assert_eq!(prefix_bytes(model), before);
        assert!(
            !model
                .tree_verify_greedy(&[3, 4], &[-1, 0])
                .unwrap()
                .is_empty()
        );
        assert_eq!(model.tree_graph_bucket_count(), 1);
    }
}

#[test]
fn resident_tree_facade_cancels_without_host_fallback() {
    let Some(mut runner) = runner() else { return };
    // Pre-cancellation must avoid native initialization and validation.
    assert!(
        runner
            .tree_verify_greedy_cancellable(&[8], &[], &|| true)
            .unwrap()
            .is_none()
    );
    runner.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();
    let before = prefix_bytes(runner.resident_cuda().unwrap().unwrap());
    let polls = Cell::new(0);
    assert!(
        runner
            .tree_verify_greedy_cancellable(&[3, 4], &[-1, 0], &|| {
                polls.set(polls.get() + 1);
                polls.get() == 5
            })
            .unwrap()
            .is_none()
    );
    assert_eq!(
        prefix_bytes(runner.resident_cuda().unwrap().unwrap()),
        before
    );
    assert_eq!(runner.resident_cuda().unwrap().unwrap().cache_len(), 3);
    assert!(runner.kv.iter().all(|cache| cache.len == 0));
    let logits = runner
        .tree_verify_logits_cancellable(&[3, 4], &[-1, 0], &|| false)
        .unwrap()
        .unwrap();
    assert_eq!(logits.len(), 16);
    runner.tree_commit(&[0]).unwrap();
    assert_eq!(runner.resident_cuda().unwrap().unwrap().cache_len(), 4);
}
