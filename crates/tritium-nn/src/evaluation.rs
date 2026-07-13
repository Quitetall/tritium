//! Deterministic reference evaluation helpers.

use crate::{ModelRunner, NnError};

/// Exact result of teacher-forced next-token scoring over fixed-size windows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TeacherForcedPerplexity {
    /// `exp(negative_log_likelihood / token_count)`.
    pub perplexity: f64,
    /// Sum of negative log likelihoods over every scored next token.
    pub negative_log_likelihood: f64,
    /// Exact number of scored next tokens: `window_count * (seq_len - 1)`.
    pub token_count: u64,
    /// Exact number of complete windows evaluated.
    pub window_count: u64,
}

impl TeacherForcedPerplexity {
    /// Mean negative log likelihood per scored token.
    #[must_use]
    pub fn mean_negative_log_likelihood(self) -> f64 {
        self.negative_log_likelihood / self.token_count as f64
    }
}

/// Score exact, non-overlapping token windows with teacher forcing.
///
/// `token_ids` must contain one or more whole windows of exactly `seq_len`
/// tokens, and `seq_len` must be at least two. The runner's KV state is reset
/// before each window. Within a window, token `i` is forwarded at absolute
/// position `i`, and its logits score token `i + 1`. Log probabilities use a
/// stable f64 log-sum-exp over the complete vocabulary.
///
/// # Errors
/// Returns [`NnError::Shape`] for an invalid window geometry or out-of-vocabulary
/// target, propagates model-forward errors, and returns [`NnError::Backend`] if
/// any logit, accumulated loss, or final perplexity is non-finite.
pub fn teacher_forced_perplexity_windows(
    runner: &mut ModelRunner,
    token_ids: &[u32],
    seq_len: usize,
) -> Result<TeacherForcedPerplexity, NnError> {
    score_windows(runner, token_ids, seq_len)
}

trait WindowRunner {
    fn reset_window(&mut self);
    fn vocab_size(&self) -> usize;
    fn forward_token(&mut self, token: u32, position: usize) -> Result<Vec<f32>, NnError>;
}

impl WindowRunner for ModelRunner {
    fn reset_window(&mut self) {
        self.reset();
    }

    fn vocab_size(&self) -> usize {
        self.weights.vocab
    }

    fn forward_token(&mut self, token: u32, position: usize) -> Result<Vec<f32>, NnError> {
        self.forward(&[token], &[position])
    }
}

fn score_windows(
    runner: &mut impl WindowRunner,
    token_ids: &[u32],
    seq_len: usize,
) -> Result<TeacherForcedPerplexity, NnError> {
    if seq_len < 2 {
        return Err(NnError::Shape {
            expected: 2,
            got: seq_len,
        });
    }
    if token_ids.is_empty() || !token_ids.len().is_multiple_of(seq_len) {
        let expected = token_ids.len().div_ceil(seq_len).saturating_mul(seq_len);
        return Err(NnError::Shape {
            expected: expected.max(seq_len),
            got: token_ids.len(),
        });
    }

    let window_count = token_ids.len() / seq_len;
    let token_count_usize = window_count
        .checked_mul(seq_len - 1)
        .ok_or(NnError::Shape {
            expected: usize::MAX,
            got: token_ids.len(),
        })?;
    let token_count = u64::try_from(token_count_usize).map_err(|_| NnError::Shape {
        expected: usize::MAX,
        got: token_ids.len(),
    })?;
    let window_count_u64 = u64::try_from(window_count).map_err(|_| NnError::Shape {
        expected: usize::MAX,
        got: token_ids.len(),
    })?;
    let mut negative_log_likelihood = 0.0_f64;

    for window in token_ids.chunks_exact(seq_len) {
        runner.reset_window();
        for position in 0..seq_len - 1 {
            let logits = runner.forward_token(window[position], position)?;
            let vocab = runner.vocab_size();
            if logits.len() != vocab {
                return Err(NnError::Shape {
                    expected: vocab,
                    got: logits.len(),
                });
            }
            let target = usize::try_from(window[position + 1]).map_err(|_| NnError::Shape {
                expected: vocab,
                got: usize::MAX,
            })?;
            if target >= vocab {
                return Err(NnError::Shape {
                    expected: vocab,
                    got: target.saturating_add(1),
                });
            }
            let loss = negative_log_probability(&logits, target)?;
            negative_log_likelihood += loss;
            if !negative_log_likelihood.is_finite() {
                return Err(NnError::Backend(
                    "teacher-forced negative log likelihood became non-finite".to_owned(),
                ));
            }
        }
    }

    let mean = negative_log_likelihood / token_count as f64;
    let perplexity = mean.exp();
    if !mean.is_finite() || !perplexity.is_finite() {
        return Err(NnError::Backend(
            "teacher-forced perplexity became non-finite".to_owned(),
        ));
    }
    Ok(TeacherForcedPerplexity {
        perplexity,
        negative_log_likelihood,
        token_count,
        window_count: window_count_u64,
    })
}

fn negative_log_probability(logits: &[f32], target: usize) -> Result<f64, NnError> {
    if logits.is_empty() || target >= logits.len() {
        return Err(NnError::Shape {
            expected: logits.len(),
            got: target.saturating_add(1),
        });
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(NnError::Backend(
            "teacher-forced logits contain a non-finite value".to_owned(),
        ));
    }
    let max = logits
        .iter()
        .copied()
        .map(f64::from)
        .fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = logits
        .iter()
        .map(|&value| (f64::from(value) - max).exp())
        .sum();
    let log_sum_exp = max + sum_exp.ln();
    let loss = log_sum_exp - f64::from(logits[target]);
    if loss.is_finite() {
        Ok(loss)
    } else {
        Err(NnError::Backend(
            "teacher-forced log probability is non-finite".to_owned(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;

    struct MockRunner {
        vocab: usize,
        logits: VecDeque<Vec<f32>>,
        resets: usize,
        calls: Vec<(u32, usize)>,
    }

    impl WindowRunner for MockRunner {
        fn reset_window(&mut self) {
            self.resets += 1;
        }

        fn vocab_size(&self) -> usize {
            self.vocab
        }

        fn forward_token(&mut self, token: u32, position: usize) -> Result<Vec<f32>, NnError> {
            self.calls.push((token, position));
            self.logits
                .pop_front()
                .ok_or_else(|| NnError::Backend("mock logits exhausted".to_owned()))
        }
    }

    fn mock(logits: Vec<Vec<f32>>) -> MockRunner {
        MockRunner {
            vocab: logits.first().map_or(0, Vec::len),
            logits: logits.into(),
            resets: 0,
            calls: Vec::new(),
        }
    }

    #[test]
    fn scores_whole_windows_resets_and_counts_exactly() {
        let logits = vec![vec![0.0, 1.0, 2.0]; 4];
        let mut runner = mock(logits);
        let score = score_windows(&mut runner, &[0, 2, 1, 2, 0, 1], 3).expect("score");

        assert_eq!(runner.resets, 2);
        assert_eq!(runner.calls, vec![(0, 0), (2, 1), (2, 0), (0, 1)]);
        assert_eq!(score.window_count, 2);
        assert_eq!(score.token_count, 4);

        let lse = (0.0_f64.exp() + 1.0_f64.exp() + 2.0_f64.exp()).ln();
        let expected_nll = (lse - 2.0) + (lse - 1.0) + (lse - 0.0) + (lse - 1.0);
        assert!((score.negative_log_likelihood - expected_nll).abs() < 1e-12);
        assert!((score.perplexity - (expected_nll / 4.0).exp()).abs() < 1e-12);
    }

    #[test]
    fn stable_f64_logsumexp_handles_extreme_finite_logits() {
        let mut runner = mock(vec![vec![10_000.0, 9_999.0]]);
        let score = score_windows(&mut runner, &[0, 1], 2).expect("score");
        let expected = (1.0 + (-1.0_f64).exp()).ln() + 1.0;
        assert!((score.negative_log_likelihood - expected).abs() < 1e-12);
        assert!(score.perplexity.is_finite());
    }

    #[test]
    fn rejects_partial_or_degenerate_windows() {
        let mut runner = mock(vec![vec![0.0, 1.0]]);
        assert!(matches!(
            score_windows(&mut runner, &[0, 1], 1),
            Err(NnError::Shape { .. })
        ));
        assert!(matches!(
            score_windows(&mut runner, &[0, 1, 0], 2),
            Err(NnError::Shape { .. })
        ));
        assert!(matches!(
            score_windows(&mut runner, &[], 2),
            Err(NnError::Shape { .. })
        ));
    }

    #[test]
    fn rejects_nonfinite_logits_and_bad_targets() {
        let mut nonfinite = mock(vec![vec![0.0, f32::NAN]]);
        assert!(matches!(
            score_windows(&mut nonfinite, &[0, 1], 2),
            Err(NnError::Backend(_))
        ));

        let mut target = mock(vec![vec![0.0, 1.0]]);
        assert!(matches!(
            score_windows(&mut target, &[0, 2], 2),
            Err(NnError::Shape { .. })
        ));
    }
}
