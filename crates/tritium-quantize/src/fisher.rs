//! Diagonal-Fisher → allocator-tile sensitivity (plan 0039).
//!
//! The rate-distortion [`allocate`](crate::allocate)r ranks groups by `H_g · Δerr_g / cost`.
//! Feeding it a *loss-derived* `H_g` — the diagonal Fisher `F_i = E[(∂L/∂w_i)²]`, reduced to one
//! scalar per allocator tile — is what makes plane allocation follow loss curvature instead of raw
//! weight magnitude (the "Energy ≈ Uniform" finding: magnitude barely re-ranks the greedy order).

use tritium_format::{QK_K, num_blocks};

/// Reduce a per-weight diagonal Fisher `F_i = E[(∂L/∂w_i)²]` (row-major `[rows*k]`) to one
/// sensitivity `H_g` per allocator group `(row, 256-block)`, row-major `r·nb + b`, as the **mean**
/// Fisher over the block's weights.
///
/// The mean is the right scalar proxy: the second-order loss increase from quantizing a group is
/// `Σ_{i∈g} F_i·Δw_i²`, and the allocator models it as `H_g·Σ_i Δw_i²`; `H_g = mean_i F_i` makes
/// the two equal when `F` is flat on the tile (a 256-block of one output channel — reasonably
/// smooth) and a first-order approximation otherwise.
///
/// The result length is `rows·num_blocks(k)` — exactly the contract of
/// [`Sensitivity::Custom`](crate::Sensitivity::Custom). Panics if `per_weight.len() != rows·k`.
pub fn tile_sensitivity(per_weight: &[f64], rows: usize, k: usize) -> Vec<f64> {
    assert_eq!(
        per_weight.len(),
        rows * k,
        "Fisher length {} must equal rows*k = {}",
        per_weight.len(),
        rows * k
    );
    let nb = num_blocks(k);
    let mut out = Vec::with_capacity(rows * nb);
    for r in 0..rows {
        let row = &per_weight[r * k..r * k + k];
        for b in 0..nb {
            let start = b * QK_K;
            let end = (start + QK_K).min(k);
            let block = &row[start..end];
            out.push(block.iter().sum::<f64>() / block.len() as f64);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{QuantConfig, Sensitivity, quantize_tensor};

    #[test]
    fn tile_sensitivity_means_per_block_short_row() {
        // rows=2, k=4 → one block per row (4 < 256); H_g = row mean.
        let f = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let h = tile_sensitivity(&f, 2, 4);
        assert_eq!(h.len(), 2 * num_blocks(4));
        assert_eq!(h, vec![2.5, 25.0]);
    }

    #[test]
    fn tile_sensitivity_splits_multi_block_row() {
        // k=300 → 2 blocks (256 + 44). Fill block0 with 1.0, block1 with 2.0.
        let mut f = vec![1.0f64; 300];
        for v in f.iter_mut().skip(256) {
            *v = 2.0;
        }
        let h = tile_sensitivity(&f, 1, 300);
        assert_eq!(h.len(), num_blocks(300));
        assert_eq!(h.len(), 2);
        assert!((h[0] - 1.0).abs() < 1e-12 && (h[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn tile_sensitivity_feeds_custom_allocation() {
        // The reduced vector is accepted as Sensitivity::Custom (length rows*nb) and allocates.
        let rows = 3;
        let k = 512; // 2 blocks/row → 6 groups
        let weights: Vec<f32> = (0..rows * k)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
            .collect();
        let fisher: Vec<f64> = (0..rows * k).map(|i| (i % 5 + 1) as f64).collect();
        let h = tile_sensitivity(&fisher, rows, k);
        assert_eq!(h.len(), rows * num_blocks(k));
        let cfg = QuantConfig {
            budget_bpw: 2.0,
            sensitivity: Sensitivity::Custom(h),
            ..Default::default()
        };
        let q = quantize_tensor(&weights, rows, k, &cfg).expect("custom-Fisher allocation");
        assert_eq!(q.plane_counts.len(), rows * num_blocks(k));
    }

    #[test]
    #[should_panic(expected = "must equal rows*k")]
    fn tile_sensitivity_rejects_wrong_length() {
        tile_sensitivity(&[1.0, 2.0, 3.0], 2, 4);
    }
}
