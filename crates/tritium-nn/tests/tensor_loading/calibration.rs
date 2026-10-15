//! Actual shared calibration graph vs the deployed host runner, not a copied oracle.
use super::HfFixture;
use tritium_nn::{ModelRunner, calibrate};
use tritium_train::Tape;

fn forward(weights: &[Vec<f32>], arch: &calibrate::Arch, tokens: &[u32]) -> Vec<f32> {
    let mut tape = Tape::new();
    let ids: Vec<_> = weights.iter().map(|w| tape.leaf(w.clone())).collect();
    let logits = calibrate::forward(&mut tape, &ids, arch, tokens);
    tape.value(logits).to_vec()
}

#[test]
fn qk_normalized_capture_matches_actual_runner_and_original_projection_vjps() {
    for head_dim in [2, 6] {
        let fixture = HfFixture::new("qwen3", true, head_dim, None);
        let mut runner =
            ModelRunner::from_hf(&fixture.dir, Box::new(tritium_cpu::CpuBackend::new())).unwrap();
        // Deliberately nonuniform learned values, not an identity-norm fixture.
        runner.weights.layers[0].q_norm = (0..head_dim).map(|i| 0.7 + 0.13 * i as f32).collect();
        runner.weights.layers[0].k_norm = (0..head_dim).map(|i| 1.2 - 0.07 * i as f32).collect();
        let (arch, weights, _) = calibrate::extract(&runner);
        let norm = arch.qk_norms[0].as_ref().unwrap();
        assert_eq!(norm.query, runner.weights.layers[0].q_norm);
        assert_eq!(norm.key, runner.weights.layers[0].k_norm);
        let exported = calibrate::norm_tensors(&arch);
        assert!(exported.iter().any(|(name, value)| name
            == "model.layers.0.self_attn.q_norm.weight"
            && *value == norm.query));
        assert!(exported.iter().any(|(name, value)| name
            == "model.layers.0.self_attn.k_norm.weight"
            && *value == norm.key));
        for tokens in [&[1][..], &[1, 4, 2][..], &[1, 4, 2, 7][..]] {
            runner.reset();
            let positions: Vec<_> = (0..tokens.len()).collect();
            let expected = runner.forward(tokens, &positions).unwrap();
            let mut tape = Tape::new();
            let ids: Vec<_> = weights.iter().map(|w| tape.leaf(w.clone())).collect();
            let mut taps = Vec::new();
            let logits =
                calibrate::forward_observed(&mut tape, &ids, &arch, tokens, &mut |_, tap| {
                    taps.push(tap)
                });
            let all = tape.value(logits);
            let last = &all[(tokens.len() - 1) * arch.vocab..];
            let max_error = last
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                max_error < 2e-5,
                "head_dim={head_dim}, max_error={max_error}"
            );
            assert_eq!(all, forward(&weights, &arch, tokens));
            let target = tape.leaf(vec![0.25; tokens.len() * arch.vocab]);
            let loss = tape.mse(logits, target);
            let gradients = tape.backward(loss);
            for tap in &taps[1..4] {
                let expected = tritium_train::ops::dense::vjp(
                    tape.value(tap.input.unwrap()),
                    tape.value(tap.weight),
                    tap.samples,
                    tap.rows,
                    tap.columns,
                    &gradients[tap.output],
                );
                assert_eq!(gradients[tap.weight], expected[1]);
                assert!(gradients[tap.weight].iter().all(|g| g.is_finite()));
            }
        }
        let normalized = forward(&weights, &arch, &[1, 4, 2]);
        let mut absent = arch.clone();
        absent.qk_norms[0] = None;
        assert_ne!(normalized, forward(&weights, &absent, &[1, 4, 2]));
        let mut calibration = calibrate::Calib::new(&arch);
        calibrate::calibrate(&weights, &arch, &[1, 4, 2], &mut calibration);
        let (_, folded) = calibrate::fold(
            &weights,
            &calibrate::extract(&runner).2,
            &arch,
            &calibration,
            0.75,
        );
        assert_eq!(folded.qk_norms, arch.qk_norms);
    }
}
