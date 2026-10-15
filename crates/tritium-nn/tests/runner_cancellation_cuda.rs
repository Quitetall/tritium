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
fn lockstep_decode_cancels_after_entry_without_advancing_any_row() {
    let Some(mut runner) = runner() else { return };
    let model = runner.resident_cuda().unwrap().unwrap();
    for path in 0..3 {
        let mut batch = model.new_batch(2).unwrap();
        let polls = Cell::new(0);
        let query = || {
            polls.set(polls.get() + 1);
            polls.get() == 2
        };
        let cancelled = match path {
            0 => model
                .decode_batch_cancellable(&mut batch, &[3, 6], &query)
                .unwrap()
                .is_none(),
            1 => model
                .decode_batch_graph_cancellable(&mut batch, &[3, 6], &query)
                .unwrap()
                .is_none(),
            _ => model
                .decode_batch_graph_argmax_cancellable(&mut batch, &[3, 6], &query)
                .unwrap()
                .is_none(),
        };
        assert!(cancelled, "path {path} must poll after entry");
        assert_eq!(batch.positions(), &[0, 0]);
    }
}

#[test]
fn independently_owned_resident_graphs_match_serial_references() {
    use std::sync::mpsc;
    use std::time::Duration;

    // Distinct prefixes exercise independent KV ownership, not two copies of
    // one request. Serial references use separate runners, then leave scope.
    let inputs = [([0, 1, 2], 3, [4, 5, 6]), ([4, 5, 6], 7, [0, 1, 2])];
    let mut references = Vec::new();
    for (seed, token, tree) in inputs {
        let Some(mut reference) = runner() else {
            return;
        };
        reference.forward(&seed, &[0, 1, 2]).unwrap();
        let decode = bits(&reference.forward(&[token], &[3]).unwrap());
        let model = reference.resident_cuda().unwrap().unwrap();
        let accepted = model.tree_verify_greedy(&tree, &[-1, 0, 0]).unwrap();
        assert!(model.tree_graph_bucket_count() > 0);
        references.push((decode, accepted, prefix_bytes(model), model.cache_len()));
    }
    let (left_tx, right_rx) = mpsc::sync_channel(1);
    let (right_tx, left_rx) = mpsc::sync_channel(1);
    let endpoints = [(left_tx, left_rx), (right_tx, right_rx)];
    std::thread::scope(|scope| {
        let mut workers = Vec::new();
        for ((input, expected), (tx, rx)) in inputs.into_iter().zip(references).zip(endpoints) {
            workers.push(scope.spawn(move || {
                let (seed, token, tree) = input;
                let mut owned = runner().expect("required peer CUDA fixture");
                let rendezvous = || {
                    tx.send(()).unwrap();
                    // Unlike a barrier, a failed peer disconnects this wait.
                    rx.recv_timeout(Duration::from_secs(5)).unwrap();
                };
                for _ in 0..8 {
                    owned.reset();
                    rendezvous();
                    owned.forward(&seed, &[0, 1, 2]).unwrap();
                    rendezvous();
                    assert_eq!(bits(&owned.forward(&[token], &[3]).unwrap()), expected.0);
                    rendezvous();
                    let model = owned.resident_cuda().unwrap().unwrap();
                    assert_eq!(
                        model.tree_verify_greedy(&tree, &[-1, 0, 0]).unwrap(),
                        expected.1
                    );
                    assert!(model.tree_graph_bucket_count() > 0);
                    assert_eq!(prefix_bytes(model), expected.2);
                    assert_eq!(model.cache_len(), expected.3);
                    assert!(owned.kv.iter().all(|cache| cache.len == 0));
                }
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }
    });
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
            (0..batch.positions().len()).flat_map(move |slot| {
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

fn lockstep_fixture(
    model: &mut tritium_cuda::CudaDecodeModel,
    paged: bool,
    prefix: usize,
) -> tritium_cuda::BatchKv {
    let mut batch = if paged {
        model.new_batch_paged(3, 3).unwrap()
    } else {
        model.new_batch(3).unwrap()
    };
    for (row, seed) in [[0, 1, 2], [4, 5, 6], [6, 7, 0]].iter().enumerate() {
        model.reset();
        model.prefill(seed, &[0, 1, 2]).unwrap();
        if paged {
            batch.reserve_pages(row, 16).unwrap();
        }
        let p = if row == 0 {
            prefix
        } else if row == 1 {
            2
        } else {
            3
        };
        model.copy_kv_into_batch_row(&mut batch, row, p).unwrap();
        batch.set_position(row, p).unwrap();
        batch.set_live(row, row != 2).unwrap();
    }
    batch
}

fn lockstep_output(
    model: &mut tritium_cuda::CudaDecodeModel,
    batch: &mut tritium_cuda::BatchKv,
    path: usize,
    tokens: &[u32],
    query: Option<&dyn Fn() -> bool>,
) -> Result<Option<Vec<Vec<u32>>>, tritium_spec::BackendError> {
    match (path, query) {
        (0, None) => model
            .decode_batch(batch, tokens)
            .map(|out| Some(out.iter().map(|row| bits(row)).collect())),
        (1, None) => model
            .decode_batch_graph(batch, tokens)
            .map(|out| Some(out.iter().map(|row| bits(row)).collect())),
        (_, None) => model
            .decode_batch_graph_argmax(batch, tokens)
            .map(|out| Some(out.into_iter().map(|id| vec![id]).collect())),
        (0, Some(query)) => model
            .decode_batch_cancellable(batch, tokens, query)
            .map(|out| out.map(|out| out.iter().map(|row| bits(row)).collect())),
        (1, Some(query)) => model
            .decode_batch_graph_cancellable(batch, tokens, query)
            .map(|out| out.map(|out| out.iter().map(|row| bits(row)).collect())),
        (_, Some(query)) => model
            .decode_batch_graph_argmax_cancellable(batch, tokens, query)
            .map(|out| out.map(|out| out.into_iter().map(|id| vec![id]).collect())),
    }
}

#[test]
fn lockstep_decode_all_checkpoints_preserve_prefix_pages_and_live_peers() {
    let Some(mut runner) = runner() else { return };
    let model = runner.resident_cuda().unwrap().unwrap();
    for path in 0..3 {
        for paged in [false, true] {
            for prefix in [0, 3] {
                let tokens = [3, 6, 0];
                let mut reference = lockstep_fixture(model, paged, prefix);
                let positions = reference.positions().to_vec();
                let before = batch_prefix_bytes(model, &reference);
                let expected = lockstep_output(model, &mut reference, path, &tokens, None)
                    .unwrap()
                    .unwrap();
                let expected_positions = reference.positions().to_vec();
                let expected_kv = batch_prefix_bytes(model, &reference);
                assert_eq!(expected_positions, [prefix + 1, 3, 3]); // dead row stays frozen
                for warm in [false, true] {
                    let prepare = |model: &mut tritium_cuda::CudaDecodeModel| {
                        let mut batch = lockstep_fixture(model, paged, prefix);
                        if warm {
                            lockstep_output(model, &mut batch, path, &tokens, None).unwrap();
                            for (row, &p) in positions.iter().enumerate() {
                                batch.set_position(row, p).unwrap();
                            }
                        }
                        batch
                    };
                    let mut batch = prepare(model);
                    let polls = Cell::new(0);
                    assert_eq!(
                        lockstep_output(
                            model,
                            &mut batch,
                            path,
                            &tokens,
                            Some(&|| {
                                polls.set(polls.get() + 1);
                                false
                            })
                        )
                        .unwrap()
                        .unwrap(),
                        expected
                    );
                    assert!(polls.get() >= 5, "in-operation checkpoints on path {path}");
                    for cancel_at in 1..=polls.get() {
                        let mut batch = prepare(model);
                        let free = batch.free_pages();
                        let pages: Vec<_> =
                            (0..3).map(|row| batch.debug_page_table_row(row)).collect();
                        // Batch work must not invalidate independent single-sequence authority.
                        model.tree_verify_logits(&[3], &[-1]).unwrap();
                        let count = Cell::new(0);
                        assert!(
                            lockstep_output(
                                model,
                                &mut batch,
                                path,
                                &tokens,
                                Some(&|| {
                                    count.set(count.get() + 1);
                                    count.get() == cancel_at
                                })
                            )
                            .unwrap()
                            .is_none()
                        );
                        assert_eq!(count.get(), cancel_at);
                        assert_eq!(batch.positions(), positions);
                        assert_eq!(batch_prefix_bytes(model, &batch), before);
                        assert_eq!(batch.free_pages(), free);
                        assert_eq!(
                            (0..3)
                                .map(|row| batch.debug_page_table_row(row))
                                .collect::<Vec<_>>(),
                            pages
                        );
                        let captured = batch.debug_decode_graphs();
                        if path == 0 || (!warm && cancel_at == 1) {
                            assert_eq!(captured, (false, false));
                        } else {
                            assert_eq!(captured, (path == 1, path == 2));
                        }
                        model.tree_commit(&[0]).unwrap();
                        model.truncate_kv(3).unwrap();
                        assert_eq!(
                            lockstep_output(model, &mut batch, path, &tokens, None)
                                .unwrap()
                                .unwrap(),
                            expected
                        );
                        assert_eq!(batch.positions(), expected_positions);
                        assert_eq!(batch_prefix_bytes(model, &batch), expected_kv);
                    }
                }
                for invalid in [&[3, 6][..], &[3, 6, 8][..]] {
                    let mut batch = lockstep_fixture(model, paged, prefix);
                    assert!(matches!(
                        lockstep_output(model, &mut batch, path, invalid, Some(&|| false)),
                        Err(tritium_spec::BackendError::InvalidInput(_))
                    ));
                    assert_eq!(batch.positions(), positions);
                    assert!(
                        lockstep_output(model, &mut batch, path, invalid, Some(&|| true))
                            .unwrap()
                            .is_none()
                    );
                }
            }
        }
    }
}

#[test]
fn lockstep_graph_facade_cancels_cold_capture_without_host_adoption() {
    let Some(mut runner) = runner() else { return };
    let mut batch = runner.new_batch_paged(2, 2).unwrap();
    for row in 0..2 {
        batch.reserve_pages(row, 16).unwrap();
    }
    assert_eq!(batch.debug_decode_graphs(), (false, false));
    let polls = Cell::new(0);
    assert!(
        runner
            .decode_batch_graph_cancellable(&mut batch, &[3, 6], &|| {
                polls.set(polls.get() + 1);
                polls.get() == 3 // facade entry, native entry, then post-capture
            })
            .unwrap()
            .is_none()
    );
    assert_eq!(batch.debug_decode_graphs(), (true, false));
    assert_eq!(batch.positions(), &[0, 0]);
    assert!(runner.kv.iter().all(|cache| cache.len == 0));
    let controlled = runner
        .decode_batch_graph_cancellable(&mut batch, &[3, 6], &|| false)
        .unwrap()
        .unwrap();
    for row in 0..2 {
        batch.set_position(row, 0).unwrap();
    }
    let ordinary = runner.decode_batch_graph(&mut batch, &[3, 6]).unwrap();
    assert_eq!(
        controlled.iter().map(|row| bits(row)).collect::<Vec<_>>(),
        ordinary.iter().map(|row| bits(row)).collect::<Vec<_>>()
    );
}

#[test]
fn lockstep_validation_errors_do_not_advance_or_capture() {
    let Some(mut runner) = runner() else { return };
    let model = runner.resident_cuda().unwrap().unwrap();
    for path in 0..3 {
        let mut unmapped = model.new_batch_paged(2, 2).unwrap();
        assert!(matches!(
            lockstep_output(model, &mut unmapped, path, &[3, 6], Some(&|| false)),
            Err(tritium_spec::BackendError::InvalidInput(_))
        ));
        assert_eq!(unmapped.positions(), &[0, 0]);
        assert_eq!(unmapped.debug_decode_graphs(), (false, false));
        let mut overflow = model.new_batch(2).unwrap();
        overflow.set_position(0, 16).unwrap();
        assert!(matches!(
            lockstep_output(model, &mut overflow, path, &[3, 6], Some(&|| false)),
            Err(tritium_spec::BackendError::InvalidInput(_))
        ));
        assert_eq!(overflow.positions(), &[16, 0]);
        assert_eq!(overflow.debug_decode_graphs(), (false, false));
        overflow.set_live(0, false).unwrap();
        lockstep_output(model, &mut overflow, path, &[3, 6], Some(&|| false))
            .unwrap()
            .unwrap();
        assert_eq!(overflow.positions(), &[16, 1]);
    }
}

#[test]
fn resident_draft_chain_and_step_cancel_without_prefix_publication() {
    let Some(mut runner) = runner() else { return };
    let model = runner.resident_cuda().unwrap().unwrap();
    for prefix in [0, 3] {
        let prepare = |model: &mut tritium_cuda::CudaDecodeModel| {
            model.reset();
            if prefix != 0 {
                model.prefill(&[0, 1, 2], &[0, 1, 2]).unwrap();
            }
        };
        prepare(model);
        let first = model.step_graph_argmax(3, prefix).unwrap();
        for eos in [u32::MAX, first] {
            prepare(model);
            let before = prefix_bytes(model);
            let mut expected = Vec::new();
            let mut token = 3;
            for i in 0..3 {
                let id = model.step_graph_argmax(token, prefix + i).unwrap();
                expected.push(id);
                if id == eos {
                    break;
                }
                token = id;
            }
            let expected_kv = prefix_bytes(model);
            prepare(model);
            let polls = Cell::new(0);
            assert_eq!(
                model
                    .draft_chain_cancellable(3, prefix, 3, eos, &|| {
                        polls.set(polls.get() + 1);
                        false
                    })
                    .unwrap()
                    .unwrap(),
                expected
            );
            assert_eq!(prefix_bytes(model), expected_kv);
            assert!(polls.get() >= 10);
            for cancel_at in 1..=polls.get() {
                prepare(model);
                // Entry-only cancellation retains authority; any entered
                // solo decode invalidates it just like ordinary decode.
                model.tree_verify_logits(&[3], &[-1]).unwrap();
                let count = Cell::new(0);
                assert!(
                    model
                        .draft_chain_cancellable(3, prefix, 3, eos, &|| {
                            count.set(count.get() + 1);
                            count.get() == cancel_at
                        })
                        .unwrap()
                        .is_none()
                );
                assert_eq!(model.cache_len(), prefix);
                assert_eq!(prefix_bytes(model), before);
                if cancel_at == 1 {
                    model.tree_commit(&[0]).unwrap();
                    model.truncate_kv(prefix).unwrap();
                } else {
                    assert!(model.tree_commit(&[0]).is_err());
                }
                assert_eq!(model.draft_chain(3, prefix, 3, eos).unwrap(), expected);
                assert_eq!(prefix_bytes(model), expected_kv);
            }
        }
        prepare(model);
        let before = prefix_bytes(model);
        for cancel_at in 1..=3 {
            let count = Cell::new(0);
            assert!(
                model
                    .step_graph_argmax_cancellable(3, prefix, &|| {
                        count.set(count.get() + 1);
                        count.get() == cancel_at
                    })
                    .unwrap()
                    .is_none()
            );
            assert_eq!(model.cache_len(), prefix);
            assert_eq!(prefix_bytes(model), before);
            assert_eq!(model.step_graph_argmax(3, prefix).unwrap(), first);
            model.truncate_kv(prefix).unwrap();
        }
        assert!(
            model
                .draft_chain_cancellable(8, prefix, 3, u32::MAX, &|| false)
                .is_err()
        );
    }
}

#[test]
fn resident_draft_cold_facade_cancellation_recovers_without_host_adoption() {
    for prefix in [0, 3] {
        for chain in [false, true] {
            let Some(mut reference) = runner() else {
                return;
            };
            let prepare = |runner: &mut ModelRunner| {
                if prefix != 0 {
                    runner.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();
                }
            };
            prepare(&mut reference);
            let polls = Cell::new(0);
            let query = || {
                polls.set(polls.get() + 1);
                false
            };
            let expected = if chain {
                reference
                    .decode_greedy_chain_cancellable(3, prefix, 3, u32::MAX, &query)
                    .unwrap()
                    .unwrap()
            } else {
                vec![
                    reference
                        .decode_greedy_step_cancellable(3, prefix, &query)
                        .unwrap()
                        .unwrap(),
                ]
            };
            let expected_kv = prefix_bytes(reference.resident_cuda().unwrap().unwrap());
            assert!(polls.get() >= 4);
            for cancel_at in 1..=polls.get() {
                let mut owned = runner().expect("required cold CUDA fixture");
                prepare(&mut owned);
                let before = prefix_bytes(owned.resident_cuda().unwrap().unwrap());
                let count = Cell::new(0);
                let cancel = || {
                    count.set(count.get() + 1);
                    count.get() == cancel_at
                };
                if chain {
                    assert!(
                        owned
                            .decode_greedy_chain_cancellable(3, prefix, 3, u32::MAX, &cancel)
                            .unwrap()
                            .is_none()
                    );
                } else {
                    assert!(
                        owned
                            .decode_greedy_step_cancellable(3, prefix, &cancel)
                            .unwrap()
                            .is_none()
                    );
                }
                let model = owned.resident_cuda().unwrap().unwrap();
                assert_eq!(model.cache_len(), prefix);
                assert_eq!(prefix_bytes(model), before);
                assert!(owned.kv.iter().all(|cache| cache.len == 0));
                let recovered = if chain {
                    owned
                        .decode_greedy_chain(3, prefix, 3, u32::MAX)
                        .unwrap()
                        .unwrap()
                } else {
                    vec![owned.decode_greedy_step(3, prefix).unwrap().unwrap()]
                };
                assert_eq!(recovered, expected);
                assert_eq!(
                    prefix_bytes(owned.resident_cuda().unwrap().unwrap()),
                    expected_kv
                );
                assert!(owned.kv.iter().all(|cache| cache.len == 0));
            }
        }
    }
}

#[test]
fn resident_draft_batch_cancellation_restores_halts_dead_rows_and_pages() {
    let Some(mut runner) = runner() else { return };
    let model = runner.resident_cuda().unwrap().unwrap();
    for paged in [false, true] {
        for prefix in [0, 3] {
            let mut batch = if paged {
                model.new_batch_paged(3, 3).unwrap()
            } else {
                model.new_batch(3).unwrap()
            };
            model.reset();
            model.prefill(&[0, 1, 2], &[0, 1, 2]).unwrap();
            for row in 0..3 {
                if paged {
                    batch.reserve_pages(row, 16).unwrap();
                }
                let p = if row == 2 { 3 } else { prefix };
                model.copy_kv_into_batch_row(&mut batch, row, p).unwrap();
                batch.set_position(row, p).unwrap();
                batch.set_live(row, row != 2).unwrap();
            }
            let positions = batch.positions().to_vec();
            let before = batch_prefix_bytes(model, &batch);
            let free = batch.free_pages();
            let pages: Vec<_> = (0..3).map(|r| batch.debug_page_table_row(r)).collect();
            // Dead-row input may be invalid: the drafter masks it before the
            // unconditional embed guard, and must never feed it on retry.
            let feeds = [3, 6, u32::MAX];
            let first = model.draft_batch(&mut batch, &feeds, 1, u32::MAX).unwrap()[0][0];
            for eos in [u32::MAX, first] {
                for (row, &p) in positions.iter().enumerate() {
                    batch.set_position(row, p).unwrap();
                }
                let expected = model.draft_batch(&mut batch, &feeds, 3, eos).unwrap();
                assert!(expected[2].is_empty());
                let expected_positions = batch.positions().to_vec();
                let expected_kv = batch_prefix_bytes(model, &batch);
                for (row, &p) in positions.iter().enumerate() {
                    batch.set_position(row, p).unwrap();
                }
                let polls = Cell::new(0);
                assert_eq!(
                    model
                        .draft_batch_cancellable(&mut batch, &feeds, 3, eos, &|| {
                            polls.set(polls.get() + 1);
                            false
                        })
                        .unwrap()
                        .unwrap(),
                    expected
                );
                assert!(polls.get() >= 4);
                for cancel_at in 1..=polls.get() {
                    for (row, &p) in positions.iter().enumerate() {
                        batch.set_position(row, p).unwrap();
                    }
                    model.tree_verify_logits(&[3], &[-1]).unwrap();
                    let count = Cell::new(0);
                    assert!(
                        model
                            .draft_batch_cancellable(&mut batch, &feeds, 3, eos, &|| {
                                count.set(count.get() + 1);
                                count.get() == cancel_at
                            })
                            .unwrap()
                            .is_none()
                    );
                    assert_eq!(batch.positions(), positions);
                    assert_eq!(batch_prefix_bytes(model, &batch), before);
                    assert_eq!(batch.free_pages(), free);
                    assert_eq!(
                        (0..3)
                            .map(|r| batch.debug_page_table_row(r))
                            .collect::<Vec<_>>(),
                        pages
                    );
                    model.tree_commit(&[0]).unwrap(); // batch leaves solo authority intact
                    model.truncate_kv(3).unwrap();
                    assert_eq!(
                        model.draft_batch(&mut batch, &feeds, 3, eos).unwrap(),
                        expected
                    );
                    assert_eq!(batch.positions(), expected_positions);
                    assert_eq!(batch_prefix_bytes(model, &batch), expected_kv);
                }
            }
        }
    }
}

#[test]
fn resident_tree_group_cancellation_has_no_partial_commit() {
    for context in [16, 12289] {
        let Some(mut runner) = runner_with_context(context) else {
            return;
        };
        let model = runner.resident_cuda().unwrap().unwrap();
        for paged in [false, true] {
            for prefix in [0, 3] {
                let mut batch = if paged {
                    model.new_batch_paged(3, 3).unwrap()
                } else {
                    model.new_batch(3).unwrap()
                };
                model.reset();
                model.prefill(&[0, 1, 2], &[0, 1, 2]).unwrap();
                for row in 0..3 {
                    if paged {
                        batch.reserve_pages(row, 16).unwrap();
                    }
                    let n = if row == 2 { 3 } else { prefix };
                    model.copy_kv_into_batch_row(&mut batch, row, n).unwrap();
                    batch.set_position(row, n).unwrap();
                    batch.set_live(row, true).unwrap();
                }
                // Distinct shapes and reverse slot order exercise concatenated
                // offsets; the third live row must never enter this group.
                let tokens = [[3, 4, 5], [6, 7, 0]];
                let trees = [
                    (&tokens[0][..], &[-1, 0, 0][..]),
                    (&tokens[1][..2], &[-1, 0][..]),
                ];
                let rows = [1, 0];
                let before = batch_prefix_bytes(model, &batch);
                let positions = batch.positions().to_vec();
                let free = batch.free_pages();
                let pages: Vec<_> = (0..3).map(|r| batch.debug_page_table_row(r)).collect();
                let mut expected = Vec::new();
                for (&row, &(t, p)) in rows.iter().zip(&trees) {
                    expected.push(
                        model
                            .tree_verify_greedy_slot(&mut batch, row, t, p)
                            .unwrap(),
                    );
                }
                let expected_positions = batch.positions().to_vec();
                let expected_kv = batch_prefix_bytes(model, &batch);
                for &row in &rows {
                    batch.set_position(row, prefix).unwrap();
                }
                let checks = Cell::new(0);
                assert_eq!(
                    model
                        .tree_verify_greedy_slots_cancellable(&mut batch, &rows, &trees, &|| {
                            checks.set(checks.get() + 1);
                            false
                        })
                        .unwrap()
                        .unwrap(),
                    expected
                );
                assert!(checks.get() >= 8, "must exercise in-operation checkpoints");
                assert_eq!(batch.positions(), expected_positions);
                assert_eq!(batch_prefix_bytes(model, &batch), expected_kv);
                assert_eq!(
                    batch.debug_tree_slots_graph_bucket_count() > 0,
                    context == 16
                );
                for cancel_at in 1..=checks.get() {
                    for &row in &rows {
                        batch.set_position(row, prefix).unwrap();
                    }
                    // Group work must leave unrelated solo authorization intact.
                    model.tree_verify_logits(&[3, 4], &[-1, 0]).unwrap();
                    let count = Cell::new(0);
                    assert!(
                        model
                            .tree_verify_greedy_slots_cancellable(
                                &mut batch,
                                &rows,
                                &trees,
                                &|| {
                                    count.set(count.get() + 1);
                                    count.get() == cancel_at
                                }
                            )
                            .unwrap()
                            .is_none()
                    );
                    assert_eq!(count.get(), cancel_at);
                    assert_eq!(batch.positions(), positions);
                    assert_eq!(batch_prefix_bytes(model, &batch), before);
                    assert_eq!(batch.free_pages(), free);
                    assert_eq!(
                        (0..3)
                            .map(|r| batch.debug_page_table_row(r))
                            .collect::<Vec<_>>(),
                        pages
                    );
                    model.tree_commit(&[0]).unwrap();
                    assert_eq!(model.cache_len(), 4);
                    model.truncate_kv(3).unwrap();
                    assert_eq!(
                        model
                            .tree_verify_greedy_slots(&mut batch, &rows, &trees)
                            .unwrap(),
                        expected
                    );
                    assert_eq!(batch.positions(), expected_positions);
                    assert_eq!(batch_prefix_bytes(model, &batch), expected_kv);
                }
                for &row in &rows {
                    batch.set_position(row, prefix).unwrap();
                }
                assert!(
                    model
                        .tree_verify_greedy_slots_cancellable(&mut batch, &[1, 1], &trees, &|| {
                            false
                        })
                        .is_err()
                );
                assert_eq!(batch.positions(), positions);
                assert_eq!(batch_prefix_bytes(model, &batch), before);
            }
        }
    }
}

#[test]
fn resident_tree_group_cold_capture_and_facade_recover() {
    let Some(mut runner) = runner() else { return };
    runner.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();
    for paged in [false, true] {
        for cancel_at in 1..=9 {
            let mut batch = if paged {
                runner.new_batch_paged(2, 2).unwrap()
            } else {
                runner.new_batch(2).unwrap()
            };
            for row in 0..2 {
                if paged {
                    batch.reserve_pages(row, 16).unwrap();
                }
                runner.adopt_into_batch_row(&mut batch, row, 3).unwrap();
                batch.set_position(row, 3).unwrap();
            }
            assert_eq!(batch.debug_tree_slots_graph_bucket_count(), 0);
            let before = batch_prefix_bytes(runner.resident_cuda().unwrap().unwrap(), &batch);
            let count = Cell::new(0);
            let trees = [(&[3, 4][..], &[-1, 0][..]), (&[5][..], &[-1][..])];
            assert!(
                runner
                    .tree_verify_greedy_slots_cancellable(&mut batch, &[0, 1], &trees, &|| {
                        count.set(count.get() + 1);
                        count.get() == cancel_at
                    })
                    .unwrap()
                    .is_none()
            );
            assert_eq!(count.get(), cancel_at);
            assert_eq!(batch.positions(), &[3, 3]);
            assert_eq!(
                batch_prefix_bytes(runner.resident_cuda().unwrap().unwrap(), &batch),
                before
            );
            assert!(runner.kv.iter().all(|cache| cache.len == 0));
            let recovered = runner
                .tree_verify_greedy_slots(&mut batch, &[0, 1], &trees)
                .unwrap();
            assert_eq!(recovered.len(), 2);
            assert!(recovered.iter().all(|tokens| !tokens.is_empty()));
            assert_eq!(batch.debug_tree_slots_graph_bucket_count(), 1);
        }
    }
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
