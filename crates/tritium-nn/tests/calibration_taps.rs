//! Execute the real graph and inspect actual pre-transform outputs after backward.
use tritium_nn::calibrate::{self, Arch, Calib, ProjectionRole, ProjectionTap, Tap};
use tritium_quantize::engine::{
    CalibrationBatch, CalibrationCacheBuilder, CalibrationInput, CalibrationOutputBatch,
};
use tritium_schema::{
    BlobId, CalibrationCacheLimits, CalibrationCacheRequest, CalibrationConsumer,
    CalibrationOutput, CalibrationPartition, NamedAssetDigest, QuantSourceBinding,
    QuantSourceTensor, ScalePrecision,
};
use tritium_train::Tape;

fn fixture() -> (Arch, Vec<Vec<f32>>) {
    let a = Arch {
        attn_norms: vec![vec![1.; 6]; 2],
        ffn_norms: vec![vec![1.; 6]; 2],
        out_norm: vec![1.; 6],
        n_embd: 6,
        n_head: 2,
        n_head_kv: 1,
        head_dim: 2,
        ff: 9,
        vocab: 17,
        eps: 1e-5,
        theta: 10_000.,
        n_layers: 2,
    };
    let mut shapes = vec![(a.vocab, a.n_embd)];
    for _ in 0..a.n_layers {
        shapes.extend([(4, 6), (2, 6), (2, 6), (6, 4), (9, 6), (9, 6), (6, 9)]);
    }
    let w = shapes
        .into_iter()
        .enumerate()
        .map(|(index, (rows, cols))| {
            (0..rows * cols)
                .map(|i| (((i * 13 + index * 7) % 37) as f32 - 18.) / 41.)
                .collect()
        })
        .collect();
    (a, w)
}

fn traced(t: &mut Tape, w: &[Vec<f32>], a: &Arch, tokens: &[u32]) -> (usize, Vec<ProjectionTap>) {
    let ids: Vec<_> = w.iter().map(|w| t.leaf(w.clone())).collect();
    let mut taps = Vec::new();
    let logits = calibrate::forward_observed(t, &ids, a, tokens, &mut |_, tap| taps.push(tap));
    (logits, taps)
}

#[test]
fn complete_original_projection_inventory_and_post_backward_vjp_are_real() {
    let (a, w) = fixture();
    let mut t = Tape::new();
    let (logits, taps) = traced(&mut t, &w, &a, &[1, 4, 2]);
    assert_eq!(taps.len(), 2 + 7 * a.n_layers);
    assert_eq!(taps[0].role, ProjectionRole::Embedding);
    assert_eq!(taps.last().unwrap().role, ProjectionRole::Head);
    assert_eq!(taps[0].weight, taps.last().unwrap().weight);
    assert_eq!(taps[0].weight_index, 0);
    assert!(taps[0].input.is_none());
    assert_ne!(taps[0].output, taps.last().unwrap().output);
    let target = t.leaf(vec![0.25; t.value(logits).len()]);
    let loss = t.mse(logits, target);
    let gradients = t.backward(loss);
    for li in 0..a.n_layers {
        let layer = &taps[1 + 7 * li..1 + 7 * (li + 1)];
        assert_eq!(
            layer.iter().map(|tap| tap.role).collect::<Vec<_>>(),
            [
                ProjectionRole::Query,
                ProjectionRole::Key,
                ProjectionRole::Value,
                ProjectionRole::Output,
                ProjectionRole::Gate,
                ProjectionRole::Up,
                ProjectionRole::Down
            ]
        );
        assert_eq!(layer[0].input, layer[1].input);
        assert_eq!(layer[1].input, layer[2].input);
        assert_eq!(layer[4].input, layer[5].input);
        for tap in layer {
            assert_eq!(tap.layer, Some(li));
            assert_eq!(tap.samples, 3);
            assert_eq!(t.value(tap.weight).len(), tap.rows * tap.columns);
            assert_eq!(t.value(tap.output).len(), tap.samples * tap.rows);
            let x = t.value(tap.input.unwrap());
            let gy = &gradients[tap.output];
            assert!(gy.iter().all(|value| value.is_finite()));
            assert!(gy.iter().any(|value| *value != 0.));
            // The existing VJP is the actual dense projection operation. Wrong
            // Q/K-after-RoPE or gate-after-SiLU IDs would fail this comparison.
            let expected = tritium_train::ops::dense::vjp(
                x,
                t.value(tap.weight),
                tap.samples,
                tap.rows,
                tap.columns,
                gy,
            );
            assert_eq!(expected[1], gradients[tap.weight]);
        }
    }
}

#[test]
fn forward_adapters_preserve_logits_and_legacy_three_role_diagonals() {
    let (a, w) = fixture();
    let tokens = [1, 4, 2];
    let mut t = Tape::new();
    let (logits, taps) = traced(&mut t, &w, &a, &tokens);
    let mut plain = Tape::new();
    let ids: Vec<_> = w.iter().map(|w| plain.leaf(w.clone())).collect();
    let plain_logits = calibrate::forward(&mut plain, &ids, &a, &tokens);
    assert_eq!(t.value(logits), plain.value(plain_logits));
    let mut aq = Tape::new();
    let ids: Vec<_> = w.iter().map(|w| aq.leaf(w.clone())).collect();
    let mut inputs = Vec::new();
    let aq_logits = calibrate::forward_aq(&mut aq, &ids, &a, &tokens, &mut |kind, li, _, _, _| {
        inputs.push((kind, li))
    });
    assert_eq!(t.value(logits), aq.value(aq_logits));
    assert_eq!(inputs.len(), 4 * a.n_layers + 1);
    assert!(inputs.contains(&(Tap::OProjIn, 0)));
    assert!(inputs.contains(&(Tap::Head, a.n_layers)));
    let mut legacy = Calib::new(&a);
    calibrate::calibrate(&w, &a, &tokens, &mut legacy);
    let mut expected = Calib::new(&a);
    for tap in taps {
        let Some(li) = tap.layer else { continue };
        let acc = match tap.role {
            ProjectionRole::Query => &mut expected.attn_in[li],
            ProjectionRole::Gate => &mut expected.ffn_in[li],
            ProjectionRole::Down => &mut expected.down_in[li],
            _ => continue,
        };
        calibrate::accumulate(&t, tap.input.unwrap(), tap.samples, tap.columns, acc);
    }
    assert_eq!(legacy.rows, tokens.len());
    assert_eq!(legacy.attn_in, expected.attn_in);
    assert_eq!(legacy.ffn_in, expected.ffn_in);
    assert_eq!(legacy.down_in, expected.down_in);
}

#[test]
fn actual_post_objective_buffers_replay_into_the_shared_collector() {
    let (a, w) = fixture();
    let tokens = [1, 4, 2];
    let mut t = Tape::new();
    let (logits, taps) = traced(&mut t, &w, &a, &tokens);
    let target = t.leaf(vec![0.25; t.value(logits).len()]);
    let loss = t.mse(logits, target);
    let gradients = t.backward(loss);
    // Canonical inventory of all actual toy forward parameters, not an HF/quality receipt.
    let names = calibrate::weight_names(&a);
    let mut sources = Vec::new();
    let mut add = |name: String, shape: Vec<u64>, values: &[f32]| {
        let bytes = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        sources.push(QuantSourceBinding {
            source: QuantSourceTensor {
                name,
                shape,
                precision: ScalePrecision::F32,
            },
            content: BlobId::from_bytes(*blake3::hash(&bytes).as_bytes()),
        });
    };
    for (index, values) in w.iter().enumerate() {
        let tap = taps.iter().find(|tap| tap.weight_index == index).unwrap();
        add(
            names[index].clone(),
            vec![tap.rows as u64, tap.columns as u64],
            values,
        );
    }
    for li in 0..a.n_layers {
        add(
            format!("model.layers.{li}.input_layernorm.weight"),
            vec![a.n_embd as u64],
            &a.attn_norms[li],
        );
        add(
            format!("model.layers.{li}.post_attention_layernorm.weight"),
            vec![a.n_embd as u64],
            &a.ffn_norms[li],
        );
    }
    add(
        "model.norm.weight".into(),
        vec![a.n_embd as u64],
        &a.out_norm,
    );
    sources.sort_by(|a, b| a.source.name.cmp(&b.source.name));
    let config = serde_json::to_vec(&serde_json::json!({"n_embd":a.n_embd,"n_head":a.n_head,"n_head_kv":a.n_head_kv,"head_dim":a.head_dim,"ff":a.ff,"vocab":a.vocab,"eps":a.eps,"theta":a.theta,"n_layers":a.n_layers})).unwrap();
    let tokenizer = b"actual-test-token-id-stream-no-production-tokenizer-claim";
    let assets = [
        NamedAssetDigest {
            name: "config.json".into(),
            content: BlobId::from_bytes(*blake3::hash(&config).as_bytes()),
        },
        NamedAssetDigest {
            name: "tokenizer.json".into(),
            content: BlobId::from_bytes(*blake3::hash(tokenizer).as_bytes()),
        },
    ];
    let model = tritium_evidence::original_source_id("toy-gqa", &sources, &assets).unwrap();
    for tap in taps {
        let Some(input) = tap.input else { continue }; // no invented dense embedding input
        let request = CalibrationCacheRequest {
            tensor: QuantSourceTensor {
                name: names[tap.weight_index].clone(),
                shape: vec![tap.rows as u64, tap.columns as u64],
                precision: ScalePrecision::F32,
            },
            model,
            output: CalibrationOutput::DenseFisher,
            consumer: CalibrationConsumer::Matmul,
            sample_start: 0,
            limits: CalibrationCacheLimits {
                max_payload_bytes: 1024 * 1024,
                max_samples: tokens.len() as u64,
                max_windows: 1,
            },
        };
        let mut cache = CalibrationCacheBuilder::new(
            request,
            tokenizer,
            b"test-Tape-mse(logits,target=0.25)-mean",
        )
        .unwrap();
        cache
            .capture(
                &tokens,
                CalibrationBatch {
                    samples: tap.samples,
                    input: CalibrationInput::Activations(t.value(input)),
                    output: CalibrationOutputBatch::Dense(&gradients[tap.output]),
                    weights: None,
                    mask: None,
                },
            )
            .unwrap();
        let cache = cache.finish().unwrap();
        let result = cache
            .collect(CalibrationPartition::Full, 0.125, 4 * 1024 * 1024)
            .unwrap();
        let chunked = cache
            .collect_chunked(CalibrationPartition::Full, 0.125, 4 * 1024 * 1024, 1)
            .unwrap();
        assert_eq!(result.factors(), chunked.factors());
        assert_eq!(result.factors().inputs[0].samples, tap.samples as u64);
        assert_eq!(
            result.factors().output.as_ref().unwrap().dimension,
            tap.rows as u64
        );
        result.kfac_metric(0, 0).unwrap();
    }
}

#[test]
#[ignore = "explicit local fp model fixture required; absence is failure, not a passing skip"]
fn real_smollm_forward_matches_runner_and_exposes_all_post_backward_roles() {
    let dir = std::env::var("TRITIUM_MODEL_DIR")
        .expect("set TRITIUM_MODEL_DIR to original dense SmolLM2 fixture");
    let mut runner = tritium_nn::ModelRunner::from_hf(
        std::path::Path::new(&dir),
        Box::new(tritium_cpu::CpuBackend::new()),
    )
    .expect("actual original fixture");
    let tokens = [1, 10];
    let reference = runner.forward(&tokens, &[0, 1]).unwrap();
    let (a, w, _) = calibrate::extract(&runner);
    drop(runner);
    let mut t = Tape::new();
    let (logits, taps) = traced(&mut t, &w, &a, &tokens);
    let last = &t.value(logits)[a.vocab..];
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    let delta = last
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let range = reference.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - reference.iter().copied().fold(f32::INFINITY, f32::min);
    let prediction = argmax(last);
    assert_eq!(prediction, argmax(&reference));
    assert!(delta / range < 5e-3); // unchanged tape_model_conformance contract
    let target = t.leaf(vec![0.; t.value(logits).len()]);
    let loss = t.mse(logits, target);
    let gradients = t.backward(loss);
    assert_eq!(taps.len(), 2 + 7 * a.n_layers);
    for tap in taps {
        let width = if tap.role == ProjectionRole::Embedding {
            tap.columns
        } else {
            tap.rows
        };
        assert_eq!(gradients[tap.output].len(), tokens.len() * width);
        assert!(gradients[tap.output].iter().all(|v| v.is_finite()));
        assert!(gradients[tap.output].iter().any(|v| *v != 0.));
    }
    println!(
        "actual SmolLM2: {} layers, {} roles, argmax={}, relative-logit-range-error={:.9e}; actual MSE backward, not a quality gate",
        a.n_layers,
        2 + 7 * a.n_layers,
        prediction,
        delta / range
    );
}
