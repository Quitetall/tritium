//! Canonical grouped-query scaled-dot-product attention for portable training.
//!
//! Projections and RoPE remain separate graph operations. This primitive owns
//! causal/noncausal GQA over row-major `Q:[seq,n_head,head_dim]` and
//! `K/V:[seq,n_kv_head,head_dim]`, matching the boundary a fused backend kernel
//! can implement without duplicating model projections.

/// Geometry for one self-attention operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionCfg {
    /// Query/key/value sequence length.
    pub seq: usize,
    /// Number of query heads.
    pub n_head: usize,
    /// Number of shared key/value heads.
    pub n_kv_head: usize,
    /// Scalar lanes per head.
    pub head_dim: usize,
    /// Whether key `j` is hidden from query `i` when `j > i`.
    pub causal: bool,
}

impl AttentionCfg {
    /// Query/output element count.
    #[must_use]
    pub fn query_elements(self) -> Option<usize> {
        self.seq
            .checked_mul(self.n_head)?
            .checked_mul(self.head_dim)
    }

    /// Key/value element count.
    #[must_use]
    pub fn kv_elements(self) -> Option<usize> {
        self.seq
            .checked_mul(self.n_kv_head)?
            .checked_mul(self.head_dim)
    }

    /// Whether geometry and flat buffers obey the GQA contract.
    #[must_use]
    pub fn buffers_fit(self, q_len: usize, k_len: usize, v_len: usize, out_len: usize) -> bool {
        self.seq != 0
            && self.n_head != 0
            && self.n_kv_head != 0
            && self.head_dim != 0
            && self.n_head.is_multiple_of(self.n_kv_head)
            && self.query_elements() == Some(q_len)
            && self.kv_elements() == Some(k_len)
            && self.kv_elements() == Some(v_len)
            && self.query_elements() == Some(out_len)
    }
}

/// Canonical GQA scaled-dot-product attention forward.
#[must_use]
pub fn forward(q: &[f32], k: &[f32], v: &[f32], cfg: AttentionCfg) -> Vec<f32> {
    debug_assert!(cfg.buffers_fit(q.len(), k.len(), v.len(), q.len()));
    let mut output = vec![0.0_f32; q.len()];
    let mut probabilities = vec![0.0_f32; cfg.seq * cfg.seq];
    let group_size = cfg.n_head / cfg.n_kv_head;
    let scale = 1.0 / (cfg.head_dim as f32).sqrt();
    for head in 0..cfg.n_head {
        let kv_head = head / group_size;
        attention_probabilities(q, k, cfg, head, kv_head, scale, &mut probabilities);
        for query in 0..cfg.seq {
            for lane in 0..cfg.head_dim {
                let mut accumulator = 0.0_f32;
                for key in 0..cfg.seq {
                    accumulator += probabilities[query * cfg.seq + key]
                        * v[vector_index(cfg, key, kv_head, cfg.n_kv_head, lane)];
                }
                output[vector_index(cfg, query, head, cfg.n_head, lane)] = accumulator;
            }
        }
    }
    output
}

/// Canonical first-order VJP returning `[grad_q, grad_k, grad_v]`.
#[must_use]
pub fn vjp(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    cfg: AttentionCfg,
    grad_output: &[f32],
) -> Vec<Vec<f32>> {
    debug_assert!(cfg.buffers_fit(q.len(), k.len(), v.len(), grad_output.len()));
    let mut grad_q = vec![0.0_f32; q.len()];
    let mut grad_k = vec![0.0_f32; k.len()];
    let mut grad_v = vec![0.0_f32; v.len()];
    let mut probabilities = vec![0.0_f32; cfg.seq * cfg.seq];
    let mut grad_probabilities = vec![0.0_f32; cfg.seq * cfg.seq];
    let group_size = cfg.n_head / cfg.n_kv_head;
    let scale = 1.0 / (cfg.head_dim as f32).sqrt();

    // Reverse head order matches reverse-mode accumulation of the composed
    // per-head reference graph when KV heads are shared by GQA.
    for head in (0..cfg.n_head).rev() {
        let kv_head = head / group_size;
        attention_probabilities(q, k, cfg, head, kv_head, scale, &mut probabilities);
        grad_probabilities.fill(0.0);
        for query in 0..cfg.seq {
            for lane in 0..cfg.head_dim {
                let gradient = grad_output[vector_index(cfg, query, head, cfg.n_head, lane)];
                for key in 0..cfg.seq {
                    let probability_index = query * cfg.seq + key;
                    let value_index = vector_index(cfg, key, kv_head, cfg.n_kv_head, lane);
                    grad_probabilities[probability_index] += gradient * v[value_index];
                    grad_v[value_index] += gradient * probabilities[probability_index];
                }
            }
        }
        for query in 0..cfg.seq {
            let row = query * cfg.seq;
            let mut contraction = 0.0_f32;
            for key in 0..cfg.seq {
                contraction += probabilities[row + key] * grad_probabilities[row + key];
            }
            for key in 0..cfg.seq {
                let index = row + key;
                grad_probabilities[index] = if cfg.causal && key > query {
                    0.0
                } else {
                    probabilities[index] * (grad_probabilities[index] - contraction) * scale
                };
            }
        }
        for query in 0..cfg.seq {
            for key in 0..cfg.seq {
                let gradient = grad_probabilities[query * cfg.seq + key];
                for lane in 0..cfg.head_dim {
                    let query_index = vector_index(cfg, query, head, cfg.n_head, lane);
                    let key_index = vector_index(cfg, key, kv_head, cfg.n_kv_head, lane);
                    grad_q[query_index] += gradient * k[key_index];
                    grad_k[key_index] += gradient * q[query_index];
                }
            }
        }
    }
    vec![grad_q, grad_k, grad_v]
}

fn attention_probabilities(
    q: &[f32],
    k: &[f32],
    cfg: AttentionCfg,
    head: usize,
    kv_head: usize,
    scale: f32,
    probabilities: &mut [f32],
) {
    for query in 0..cfg.seq {
        let row = query * cfg.seq;
        for key in 0..cfg.seq {
            let mut score = 0.0_f32;
            for lane in 0..cfg.head_dim {
                score += q[vector_index(cfg, query, head, cfg.n_head, lane)]
                    * k[vector_index(cfg, key, kv_head, cfg.n_kv_head, lane)];
            }
            probabilities[row + key] = if cfg.causal && key > query {
                f32::NEG_INFINITY
            } else {
                score * scale
            };
        }
        let maximum = probabilities[row..row + cfg.seq]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for key in 0..cfg.seq {
            let exponential = if cfg.causal && key > query {
                0.0
            } else {
                (probabilities[row + key] - maximum).exp()
            };
            probabilities[row + key] = exponential;
            sum += exponential;
        }
        for key in 0..cfg.seq {
            probabilities[row + key] /= sum;
        }
    }
}

fn vector_index(
    cfg: AttentionCfg,
    token: usize,
    head: usize,
    head_count: usize,
    lane: usize,
) -> usize {
    (token * head_count + head) * cfg.head_dim + lane
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_gqa_forward_and_vjp_are_finite_and_shaped() {
        let cfg = AttentionCfg {
            seq: 3,
            n_head: 2,
            n_kv_head: 1,
            head_dim: 2,
            causal: true,
        };
        let q = [
            0.2, -0.1, 0.4, 0.3, -0.5, 0.7, 0.1, -0.2, 0.6, 0.8, -0.3, 0.9,
        ];
        let k = [0.5, -0.4, 0.2, 0.1, -0.6, 0.7];
        let v = [1.0, -1.0, 0.5, 0.25, -0.75, 1.5];
        let output = forward(&q, &k, &v, cfg);
        assert_eq!(output.len(), q.len());
        assert!(output.iter().all(|value| value.is_finite()));
        let gradients = vjp(&q, &k, &v, cfg, &[0.25; 12]);
        assert_eq!(
            gradients.iter().map(Vec::len).collect::<Vec<_>>(),
            [12, 6, 6]
        );
        assert!(gradients.iter().flatten().all(|value| value.is_finite()));
    }

    #[test]
    fn causal_mask_is_exact_when_allowed_score_is_below_mask_sentinel_range() {
        let cfg = AttentionCfg {
            seq: 2,
            n_head: 1,
            n_kv_head: 1,
            head_dim: 1,
            causal: true,
        };
        let output = forward(&[1.0e16, 0.0], &[-1.0e16, 0.0], &[2.0, 100.0], cfg);
        assert_eq!(output[0], 2.0);
    }

    #[test]
    fn causal_gqa_vjp_matches_finite_difference() {
        let cfg = AttentionCfg {
            seq: 3,
            n_head: 2,
            n_kv_head: 1,
            head_dim: 2,
            causal: true,
        };
        let inputs = vec![
            vec![
                0.2, -0.1, 0.4, 0.3, -0.5, 0.7, 0.1, -0.2, 0.6, 0.8, -0.3, 0.9,
            ],
            vec![0.5, -0.4, 0.2, 0.1, -0.6, 0.7],
            vec![1.0, -1.0, 0.5, 0.25, -0.75, 1.5],
        ];
        let cotangent = [
            0.25_f32, -0.5, 0.75, 0.1, -0.2, 0.4, -0.6, 0.3, 0.9, -0.8, 0.2, 0.5,
        ];
        let gradients = vjp(&inputs[0], &inputs[1], &inputs[2], cfg, &cotangent);
        let loss = |values: &[Vec<f32>]| {
            forward(&values[0], &values[1], &values[2], cfg)
                .iter()
                .zip(cotangent)
                .map(|(&value, gradient)| f64::from(value) * f64::from(gradient))
                .sum::<f64>()
        };
        let step = 1.0e-3_f32;
        for input in 0..inputs.len() {
            for element in 0..inputs[input].len() {
                let mut plus = inputs.clone();
                let mut minus = inputs.clone();
                plus[input][element] += step;
                minus[input][element] -= step;
                let numeric = (loss(&plus) - loss(&minus)) / (2.0 * f64::from(step));
                let analytic = f64::from(gradients[input][element]);
                let scale = numeric.abs().max(1.0);
                assert!(
                    ((analytic - numeric) / scale).abs() < 4.0e-3,
                    "input {input} element {element}: analytic {analytic} numeric {numeric}"
                );
            }
        }
    }
}
