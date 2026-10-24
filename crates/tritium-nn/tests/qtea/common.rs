use tritium_core::Trit;
use tritium_format::qtea::{
    QteaBindings, QteaGroupRef, QteaLimits, QteaMetadata, QteaRecipe, QteaResidualRef, QteaTensor,
};

pub(crate) fn limits() -> QteaLimits {
    QteaLimits {
        max_tensor_bytes: 1 << 20,
        max_validation_bytes: 1 << 16,
    }
}

// Independent, deterministic decoded oracle. No fitter or QTEA decoder is used
// to derive expected weights; 0x38/0xb8 are E4M3 +1/-1, respectively.
pub(crate) fn fixture(
    rows: usize,
    cols: usize,
    group: usize,
    ratio: f64,
    seed: usize,
) -> (QteaTensor, Vec<f32>) {
    let meta = QteaMetadata {
        rows,
        columns: cols,
        recipe: QteaRecipe {
            group_size: group,
            salient_ratio: ratio,
            delta_coefficient: 0.96,
            fit_iterations: 1,
            rescale_iterations: 2,
            damping: 0.007,
            decay: 1.,
            relative_pivot_floor: 1e-7,
        },
        bindings: QteaBindings {
            weights: [1; 32],
            gram: [2; 32],
            source_model: [3; 32],
            activation_cache: [4; 32],
            token_stream: [5; 32],
        },
    };
    let trits: Vec<_> = (0..rows * cols)
        .map(|i| [Trit::NEG, Trit::ZERO, Trit::POS][(i + seed) % 3])
        .collect();
    let alpha: Vec<Vec<_>> = (0..cols / group)
        .map(|g| {
            (0..rows)
                .map(|r| ((r + g + seed) % 4 + 1) as f32 / 16.)
                .collect()
        })
        .collect();
    let beta: Vec<Vec<_>> = (0..cols / group)
        .map(|g| {
            (0..rows)
                .map(|r| ((r + g + seed) % 5) as f32 / 64. - 0.03125)
                .collect()
        })
        .collect();
    let multipliers: Vec<_> = (0..cols)
        .map(|c| ((c + seed) % 3 + 1) as f32 / 2.)
        .collect();
    let order: Vec<_> = (0..cols).rev().collect();
    let count = if ratio == 0. {
        0
    } else {
        ((cols / group) as f64 * ratio).round_ties_even().max(1.) as usize * group
    };
    let selected: Vec<_> = (cols - count..cols).collect();
    let indices: Vec<Vec<u8>> = selected
        .iter()
        .map(|c| (0..rows / 4).map(|b| ((c + b + seed) % 4) as u8).collect())
        .collect();
    let codes: Vec<Vec<u8>> = selected
        .iter()
        .map(|c| {
            (0..rows / 4)
                .map(|b| {
                    if (c + b + seed).is_multiple_of(2) {
                        0x38
                    } else {
                        0xb8
                    }
                })
                .collect()
        })
        .collect();
    let mut decoded = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let mut weight = (alpha[c / group][r] * multipliers[c]) * trits[r * cols + c].to_f32()
                + beta[c / group][r];
            if let Some(i) = selected.iter().position(|column| *column == c)
                && usize::from(indices[i][r / 4]) == r % 4
            {
                weight += if codes[i][r / 4] == 0x38 {
                    0.0625
                } else {
                    -0.0625
                };
            }
            decoded.push(weight);
        }
    }
    let tensor = QteaTensor::from_parts(
        meta,
        &trits,
        &multipliers,
        &order,
        alpha
            .iter()
            .zip(&beta)
            .map(|(alpha, beta)| QteaGroupRef { alpha, beta }),
        selected
            .iter()
            .enumerate()
            .map(|(i, &column)| QteaResidualRef {
                column,
                // Stored inverse scale: decoded residual is FP8 / multiplier.
                scale_multiplier: 16.,
                indices: &indices[i],
                codes: &codes[i],
            }),
        limits(),
    )
    .unwrap();
    (tensor, decoded)
}
