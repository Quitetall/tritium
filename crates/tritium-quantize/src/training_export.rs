//! Conversion from the training SALT representation to deployable [`SaltRow`] bytes.
//!
//! Training fits every plane over a complete weight row with an `f32` AbsMean scale. The
//! inference format stores an `f16` scale in every 256-trit TQ2 block. This adapter preserves
//! the training trit decisions, narrows each fitted row scale once, and repeats that same
//! narrowed scale across all blocks in the plane.

use core::fmt;

use half::f16;
use tritium_format::{FormatError, SaltRow, TQ2_0_BLOCK_BYTES, num_blocks, pack_tq2_0_row};

use crate::absmean_ternary;

/// Statistics for the f32-training-scale to f16-deployment-scale conversion of one row.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TrainingSaltExportStats {
    /// Largest absolute difference between a fitted f32 scale and its f16 deployment value.
    pub scale_max_abs_error: f64,
    /// Largest absolute reconstruction difference caused only by narrowing the fitted scales.
    pub reconstruction_max_abs_delta: f64,
    /// Sum of squared reconstruction differences caused only by narrowing the fitted scales.
    pub reconstruction_squared_error_sum: f64,
    /// Number of reconstructed weights represented by the squared-error sum.
    pub reconstruction_element_count: u64,
}

/// Why a training SALT row could not be converted to the inference representation.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum TrainingSaltExportError {
    /// The requested plane count was outside the training runtime's `1..=3` contract.
    InvalidPlaneCount {
        /// Plane count supplied by the caller.
        got: usize,
    },
    /// A trained weight row had no contraction elements.
    EmptyRow,
    /// A master weight was NaN or infinite.
    NonFiniteWeight {
        /// Position of the invalid value within the row.
        index: usize,
    },
    /// Whole-row AbsMean overflowed or otherwise produced a nonfinite scale.
    NonFiniteScale {
        /// Zero-based residual-plane index.
        plane: usize,
    },
    /// A finite fitted scale was outside the finite f16 deployment range.
    ScaleOverflow {
        /// Zero-based residual-plane index.
        plane: usize,
        /// Original fitted f32 scale encoded with [`f32::to_bits`].
        scale_bits: u32,
    },
    /// A nonzero fitted scale narrowed to zero in the f16 deployment format.
    ScaleUnderflow {
        /// Zero-based residual-plane index.
        plane: usize,
        /// Original fitted f32 scale encoded with [`f32::to_bits`].
        scale_bits: u32,
    },
    /// Packing the inference row failed.
    Format(FormatError),
}

impl fmt::Display for TrainingSaltExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPlaneCount { got } => {
                write!(f, "training SALT plane count must be in 1..=3, got {got}")
            }
            Self::EmptyRow => write!(f, "training SALT master row must not be empty"),
            Self::NonFiniteWeight { index } => {
                write!(
                    f,
                    "training SALT master row contains a nonfinite value at {index}"
                )
            }
            Self::NonFiniteScale { plane } => {
                write!(
                    f,
                    "training SALT plane {plane} produced a nonfinite f32 scale"
                )
            }
            Self::ScaleOverflow { plane, scale_bits } => write!(
                f,
                "training SALT plane {plane} scale {} overflows f16",
                f32::from_bits(*scale_bits)
            ),
            Self::ScaleUnderflow { plane, scale_bits } => write!(
                f,
                "training SALT plane {plane} nonzero scale {} underflows to f16 zero",
                f32::from_bits(*scale_bits)
            ),
            Self::Format(error) => write!(f, "pack training SALT row: {error}"),
        }
    }
}

impl std::error::Error for TrainingSaltExportError {}

impl From<FormatError> for TrainingSaltExportError {
    fn from(error: FormatError) -> Self {
        Self::Format(error)
    }
}

/// Export one complete trained weight row to a fixed-plane inference [`SaltRow`].
///
/// Trits are fitted in f32 over the whole row. Each plane's fitted f32 row scale is then
/// narrowed once to f16 and duplicated into every TQ2 block without refitting the trits.
/// Additional working storage is linear in the row width; the returned fixed-plane payload is
/// linear in `master_row.len() * planes`.
///
/// # Errors
/// Returns [`TrainingSaltExportError`] when `planes` is outside `1..=3`, the row is empty, a master
/// value or fitted scale is nonfinite, a nonzero fitted scale is not representable as a finite
/// nonzero f16, or the packed inference row cannot be produced.
pub fn export_training_salt_row(
    master_row: &[f32],
    planes: usize,
) -> Result<(SaltRow, TrainingSaltExportStats), TrainingSaltExportError> {
    if !(1..=3).contains(&planes) {
        return Err(TrainingSaltExportError::InvalidPlaneCount { got: planes });
    }
    if master_row.is_empty() {
        return Err(TrainingSaltExportError::EmptyRow);
    }
    if let Some(index) = master_row.iter().position(|value| !value.is_finite()) {
        return Err(TrainingSaltExportError::NonFiniteWeight { index });
    }
    let blocks = num_blocks(master_row.len());
    let mut residual = master_row.to_vec();
    let mut packed_planes = Vec::with_capacity(planes);
    let mut training_reconstruction = vec![0.0f32; master_row.len()];
    let mut deployment_reconstruction = vec![0.0f32; master_row.len()];
    let mut stats = TrainingSaltExportStats {
        reconstruction_element_count: master_row.len() as u64,
        ..TrainingSaltExportStats::default()
    };

    for plane_index in 0..planes {
        let plane = absmean_ternary(&residual);
        if !plane.scale.is_finite() {
            return Err(TrainingSaltExportError::NonFiniteScale { plane: plane_index });
        }
        let narrowed = f16::from_f32(plane.scale);
        if narrowed.is_infinite() {
            return Err(TrainingSaltExportError::ScaleOverflow {
                plane: plane_index,
                scale_bits: plane.scale.to_bits(),
            });
        }
        if plane.scale != 0.0 && narrowed == f16::ZERO {
            return Err(TrainingSaltExportError::ScaleUnderflow {
                plane: plane_index,
                scale_bits: plane.scale.to_bits(),
            });
        }
        let narrowed_f32 = narrowed.to_f32();
        let scale_delta = f64::from(narrowed_f32) - f64::from(plane.scale);
        stats.scale_max_abs_error = stats.scale_max_abs_error.max(scale_delta.abs());

        for (((value, training), deployment), trit) in residual
            .iter_mut()
            .zip(&mut training_reconstruction)
            .zip(&mut deployment_reconstruction)
            .zip(&plane.trits)
        {
            let trit = trit.to_f32();
            let training_contribution = plane.scale * trit;
            *value -= training_contribution;
            *training += training_contribution;
            *deployment += narrowed_f32 * trit;
        }

        let scales = vec![narrowed; blocks];
        let mut packed = vec![0u8; blocks * TQ2_0_BLOCK_BYTES];
        pack_tq2_0_row(&plane.trits, &scales, &mut packed)?;
        packed_planes.push(packed);
    }

    for (training, deployment) in training_reconstruction
        .into_iter()
        .zip(deployment_reconstruction)
    {
        let delta = f64::from(deployment) - f64::from(training);
        stats.reconstruction_max_abs_delta = stats.reconstruction_max_abs_delta.max(delta.abs());
        stats.reconstruction_squared_error_sum += delta * delta;
    }

    Ok((
        SaltRow {
            k: master_row.len(),
            planes: packed_planes,
        },
        stats,
    ))
}

#[cfg(test)]
mod tests {
    use half::f16;
    use tritium_core::Trit;
    use tritium_format::{
        QK_K, TQ2_0_BLOCK_BYTES, dequant_salt_row, num_blocks, unpack_tq2_0_block, unpack_tq2_0_row,
    };

    use crate::{BaseScaleScope, QuantConfig, Sensitivity, TRIT_BITS, quantize_tensor};

    use super::{TrainingSaltExportError, export_training_salt_row};

    fn training_oracle(master: &[f32], planes: usize) -> Vec<(f32, Vec<Trit>)> {
        let mut residual = master.to_vec();
        let mut out = Vec::with_capacity(planes);
        for _ in 0..planes {
            let scale =
                residual.iter().map(|value| value.abs()).sum::<f32>() / residual.len() as f32;
            let trits: Vec<Trit> = residual
                .iter()
                .map(|value| {
                    let code = if scale == 0.0 {
                        0
                    } else {
                        (value / scale).round().clamp(-1.0, 1.0) as i8
                    };
                    Trit::from_i8(code).expect("oracle clamps to a trit")
                })
                .collect();
            for (value, trit) in residual.iter_mut().zip(&trits) {
                *value -= scale * trit.to_f32();
            }
            out.push((scale, trits));
        }
        out
    }

    fn oracle_reconstruction(planes: &[(f32, Vec<Trit>)]) -> Vec<f32> {
        let mut out = vec![0.0f32; planes[0].1.len()];
        for (scale, trits) in planes {
            for (value, trit) in out.iter_mut().zip(trits) {
                *value += *scale * trit.to_f32();
            }
        }
        out
    }

    fn projected_oracle_reconstruction(planes: &[(f32, Vec<Trit>)]) -> Vec<f32> {
        let mut out = vec![0.0f32; planes[0].1.len()];
        for (scale, trits) in planes {
            let scale = f16::from_f32(*scale).to_f32();
            for (value, trit) in out.iter_mut().zip(trits) {
                *value += scale * trit.to_f32();
            }
        }
        out
    }

    fn varied_master(k: usize) -> Vec<f32> {
        (0..k)
            .map(|index| {
                if index.is_multiple_of(97) {
                    0.0
                } else {
                    let centered = ((index * 73) % 211) as f32 - 105.0;
                    centered / 19.0 + (index % 7) as f32 / 128.0
                }
            })
            .collect()
    }

    #[test]
    fn rejects_plane_counts_outside_the_training_contract() {
        for planes in [0, 4] {
            assert_eq!(
                export_training_salt_row(&[1.0], planes),
                Err(TrainingSaltExportError::InvalidPlaneCount { got: planes })
            );
        }
    }

    #[test]
    fn rejects_an_empty_training_row_for_every_supported_plane_count() {
        for planes in 1..=3 {
            assert_eq!(
                export_training_salt_row(&[], planes),
                Err(TrainingSaltExportError::EmptyRow)
            );
        }
    }

    #[test]
    fn rejects_nonfinite_master_values_with_their_position() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                export_training_salt_row(&[1.0, value, 2.0], 2),
                Err(TrainingSaltExportError::NonFiniteWeight { index: 1 })
            );
        }
    }

    #[test]
    fn rejects_a_nonfinite_fitted_scale() {
        assert_eq!(
            export_training_salt_row(&[f32::MAX, f32::MAX], 1),
            Err(TrainingSaltExportError::NonFiniteScale { plane: 0 })
        );
    }

    #[test]
    fn rejects_a_finite_scale_that_overflows_f16() {
        let scale = 70_000.0f32;
        assert_eq!(
            export_training_salt_row(&[scale], 1),
            Err(TrainingSaltExportError::ScaleOverflow {
                plane: 0,
                scale_bits: scale.to_bits(),
            })
        );
    }

    #[test]
    fn rejects_a_nonzero_scale_that_underflows_to_f16_zero() {
        let scale = f32::from_bits(1);
        assert_eq!(
            export_training_salt_row(&[scale], 1),
            Err(TrainingSaltExportError::ScaleUnderflow {
                plane: 0,
                scale_bits: scale.to_bits(),
            })
        );
    }

    #[test]
    fn stats_measure_the_reloaded_f16_projection_against_training_reconstruction() {
        let master = [0.1, -0.35, 0.9, -1.7, 2.3, -0.04, 0.57];
        let oracle = training_oracle(&master, 3);
        let training_reconstruction = oracle_reconstruction(&oracle);

        let (row, stats) = export_training_salt_row(&master, 3).expect("export row");
        let deployment_reconstruction = dequant_salt_row(&row).expect("dequant exported row");
        let mut max_abs_delta = 0.0f64;
        let mut squared_error_sum = 0.0f64;
        for (training, deployment) in training_reconstruction
            .iter()
            .zip(&deployment_reconstruction)
        {
            let delta = f64::from(*deployment) - f64::from(*training);
            max_abs_delta = max_abs_delta.max(delta.abs());
            squared_error_sum += delta * delta;
        }

        assert_eq!(
            stats.reconstruction_max_abs_delta.to_bits(),
            max_abs_delta.to_bits()
        );
        assert_eq!(
            stats.reconstruction_squared_error_sum.to_bits(),
            squared_error_sum.to_bits()
        );
        assert_eq!(stats.reconstruction_element_count, master.len() as u64);
    }

    #[test]
    fn exact_training_codes_and_row_scales_hold_for_all_campaign_shapes() {
        for k in [7, 257, 576, 8193] {
            let master = varied_master(k);
            for planes in 1..=3 {
                let oracle = training_oracle(&master, planes);
                let (row, stats) =
                    export_training_salt_row(&master, planes).expect("export campaign row");
                assert_eq!(row.k, k);
                assert_eq!(row.planes.len(), planes);

                let mut expected_scale_max_abs_error = 0.0f64;
                for (plane_index, ((scale, expected_trits), packed)) in
                    oracle.iter().zip(&row.planes).enumerate()
                {
                    let mut trits = vec![Trit::ZERO; k];
                    let mut scales = vec![f16::ZERO; num_blocks(k)];
                    unpack_tq2_0_row(packed, &mut trits, &mut scales)
                        .expect("unpack exported plane");
                    assert_eq!(trits, *expected_trits, "K={k} T={planes} p={plane_index}");
                    let narrowed = f16::from_f32(*scale);
                    assert!(
                        scales
                            .iter()
                            .all(|value| value.to_bits() == narrowed.to_bits())
                    );
                    expected_scale_max_abs_error = expected_scale_max_abs_error
                        .max((f64::from(narrowed.to_f32()) - f64::from(*scale)).abs());
                }
                assert_eq!(
                    stats.scale_max_abs_error.to_bits(),
                    expected_scale_max_abs_error.to_bits(),
                    "K={k} T={planes}"
                );

                let dequantized = dequant_salt_row(&row).expect("dequant exported row");
                let expected = projected_oracle_reconstruction(&oracle);
                assert_eq!(
                    dequantized
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    "K={k} T={planes}"
                );
            }
        }
    }

    #[test]
    fn zero_rows_emit_fixed_canonical_zero_planes() {
        let master = vec![0.0f32; 257];
        let (row, stats) = export_training_salt_row(&master, 3).expect("export zero row");

        assert_eq!(row.planes.len(), 3);
        for plane in &row.planes {
            assert_eq!(plane.len(), 2 * TQ2_0_BLOCK_BYTES);
            for block in plane.as_chunks::<TQ2_0_BLOCK_BYTES>().0 {
                assert!(block[..64].iter().all(|byte| *byte == 0x55));
                assert_eq!(&block[64..], &[0, 0]);
            }
        }
        assert_eq!(
            stats,
            super::TrainingSaltExportStats {
                reconstruction_element_count: master.len() as u64,
                ..super::TrainingSaltExportStats::default()
            }
        );
    }

    #[test]
    fn a_collapsed_residual_keeps_canonical_zero_trailing_planes() {
        let (row, _) = export_training_salt_row(&[1.0, -1.0], 3).expect("export exact row");
        assert_eq!(row.planes.len(), 3);
        for plane in &row.planes[1..] {
            assert!(plane[..64].iter().all(|byte| *byte == 0x55));
            assert_eq!(&plane[64..], &[0, 0]);
        }
    }

    #[test]
    fn training_rounding_thresholds_are_preserved() {
        let master = [0.5, -0.5, 0.25, -0.25, 1.5, -1.5, 1.75, -1.75];
        let (row, _) = export_training_salt_row(&master, 1).expect("export threshold row");
        let mut trits = vec![Trit::ZERO; master.len()];
        let mut scales = vec![f16::ZERO; 1];
        unpack_tq2_0_row(&row.planes[0], &mut trits, &mut scales).expect("unpack plane");
        assert_eq!(
            trits,
            vec![
                Trit::POS,
                Trit::NEG,
                Trit::ZERO,
                Trit::ZERO,
                Trit::POS,
                Trit::NEG,
                Trit::POS,
                Trit::NEG,
            ]
        );
        assert_eq!(scales[0].to_bits(), f16::ONE.to_bits());
    }

    #[test]
    fn partial_final_blocks_are_zero_padded_canonically() {
        let master = vec![1.0f32; 257];
        let (row, _) = export_training_salt_row(&master, 1).expect("export tailed row");
        let last_block = &row.planes[0][TQ2_0_BLOCK_BYTES..2 * TQ2_0_BLOCK_BYTES];
        let mut trits = [Trit::ZERO; QK_K];
        let mut scale = f16::ZERO;
        unpack_tq2_0_block(last_block, &mut trits, &mut scale).expect("unpack tail block");
        assert_eq!(trits[0], Trit::POS);
        assert!(trits[1..].iter().all(|trit| *trit == Trit::ZERO));
        assert_eq!(scale.to_bits(), f16::ONE.to_bits());
    }

    #[test]
    fn ordinary_block_quantization_is_not_training_salt_export() {
        let mut master = vec![1.0f32; QK_K];
        master.extend(vec![100.0f32; QK_K]);
        let (training_row, _) = export_training_salt_row(&master, 1).expect("export training row");
        let ordinary = quantize_tensor(
            &master,
            1,
            master.len(),
            &QuantConfig {
                budget_bpw: TRIT_BITS,
                t_min: 1,
                t_max: 1,
                sensitivity: Sensitivity::Uniform,
                scale_group: BaseScaleScope::Block,
            },
        )
        .expect("ordinary block quantization");

        assert_ne!(training_row, ordinary.salt_rows[0]);
        let mut training_trits = vec![Trit::ZERO; master.len()];
        let mut ordinary_trits = vec![Trit::ZERO; master.len()];
        let mut scales = vec![f16::ZERO; 2];
        unpack_tq2_0_row(&training_row.planes[0], &mut training_trits, &mut scales)
            .expect("unpack training export");
        unpack_tq2_0_row(
            &ordinary.salt_rows[0].planes[0],
            &mut ordinary_trits,
            &mut scales,
        )
        .expect("unpack ordinary quantization");
        assert_eq!(training_trits[0], Trit::ZERO);
        assert_eq!(ordinary_trits[0], Trit::POS);
    }

    #[test]
    fn t1_preserves_whole_row_trits_and_repeats_the_narrowed_scale() {
        let master = [1.0, -1.0, 0.0, 0.4, -0.4, 2.0, -2.0];

        let (row, _stats) = export_training_salt_row(&master, 1).expect("export row");

        assert_eq!(row.k, master.len());
        assert_eq!(row.planes.len(), 1);
        let blocks = num_blocks(master.len());
        let mut trits = vec![Trit::ZERO; master.len()];
        let mut scales = vec![f16::ZERO; blocks];
        unpack_tq2_0_row(&row.planes[0], &mut trits, &mut scales).expect("unpack plane");
        assert_eq!(
            trits,
            vec![
                Trit::POS,
                Trit::NEG,
                Trit::ZERO,
                Trit::ZERO,
                Trit::ZERO,
                Trit::POS,
                Trit::NEG,
            ]
        );
        let expected_scale = f16::from_f32(6.8 / 7.0);
        assert!(
            scales
                .iter()
                .all(|scale| scale.to_bits() == expected_scale.to_bits())
        );
        assert_eq!(row.planes[0].len(), blocks * TQ2_0_BLOCK_BYTES);
    }
}
