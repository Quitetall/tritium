//! Differentiable token-embedding gather (plan 0040 step 3).
//!
//! The input side of a transformer: look up `[seq, n_embd]` rows from a `[vocab, n_embd]` table by
//! token id. When embeddings are tied to the lm-head (as in SmolLM2/Qwen), this table is a trained,
//! SALT-quantized weight — so the gather must pass gradient back to the rows it read.

/// `Y[s, :] = weight[tokens[s], :]` — gather `[seq, n_embd]` rows from `weight [vocab, n_embd]`.
pub fn gather_forward(weight: &[f32], tokens: &[u32], n_embd: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; tokens.len() * n_embd];
    for (s, &tok) in tokens.iter().enumerate() {
        let row = tok as usize * n_embd;
        y[s * n_embd..s * n_embd + n_embd].copy_from_slice(&weight[row..row + n_embd]);
    }
    y
}

/// vjp of [`gather_forward`]: scatter-**add** `grad_out [seq, n_embd]` rows into the
/// `[vocab, n_embd]` table (a token repeated in `tokens` accumulates). Returns the full table
/// gradient. `tokens` are data (no gradient).
pub fn gather_vjp(vocab: usize, tokens: &[u32], n_embd: usize, grad_out: &[f32]) -> Vec<f32> {
    let mut gw = vec![0.0f32; vocab * n_embd];
    for (s, &tok) in tokens.iter().enumerate() {
        let row = tok as usize * n_embd;
        for c in 0..n_embd {
            gw[row + c] += grad_out[s * n_embd + c];
        }
    }
    gw
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradcheck::{GradCheckCfg, check_op};

    #[test]
    fn gather_reads_the_right_rows() {
        // vocab 3, n_embd 2
        let w = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(gather_forward(&w, &[2, 0], 2), vec![4.0, 5.0, 0.0, 1.0]);
    }

    #[test]
    fn gather_gradcheck_with_a_repeated_token() {
        // A repeated token (0 appears twice) must accumulate its gradient — the key vjp property.
        let (vocab, n_embd) = (4usize, 3usize);
        let tokens = [0u32, 3, 0, 1];
        let w: Vec<f32> = (0..vocab * n_embd)
            .map(|i| (i as f32 * 0.31).sin())
            .collect();
        check_op(
            |ins| gather_forward(ins[0], &tokens, n_embd),
            |_ins, g| vec![gather_vjp(vocab, &tokens, n_embd, g)],
            &[w],
            &[0],
            GradCheckCfg::default(),
        )
        .expect("embed gather vjp");
    }
}
