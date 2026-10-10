//! Exact outputs captured on e7bc4f9f before kernel relocation.
//! Regression fingerprints are not artifact identities or model-quality qualification.
use tritium_nn::salt_fit::{
    ActivationAwareConfig, ScaleRefitMode, fit_tensor_with_scale_refit_mode,
};

fn inputs(rows: usize, cols: usize) -> (Vec<f32>, Vec<f64>) {
    let weights = (0..rows * cols)
        .map(|i| (((i * 37 + 19) % 113) as f32 - 56.) / 41.)
        .collect();
    let gram = (0..cols * cols)
        .map(|i| {
            let (row, col) = (i / cols, i % cols);
            if row == col {
                1.0 + row as f64 / 17.
            } else {
                0.31f64.powi(row.abs_diff(col) as i32)
            }
        })
        .collect();
    (weights, gram)
}

fn digest(fits: &[(f32, Vec<Vec<i8>>)]) -> String {
    let mut h = blake3::Hasher::new();
    h.update(&(fits.len() as u64).to_le_bytes());
    for (scale, planes) in fits {
        h.update(&scale.to_bits().to_le_bytes());
        h.update(&(planes.len() as u64).to_le_bytes());
        for plane in planes {
            h.update(&(plane.len() as u64).to_le_bytes());
            for &trit in plane {
                h.update(&[trit as u8]);
            }
        }
    }
    h.finalize().to_hex().to_string()
}

#[test]
fn shipping_f32_accum64_fit_has_fixed_scale_bits_and_trits() {
    for (case, rows, cols, group, planes, rotate, decay, ramp, sweeps, refit, mode) in [
        (
            "plain",
            2,
            12,
            4,
            3,
            false,
            1.,
            false,
            0,
            false,
            ScaleRefitMode::PostPass,
        ),
        (
            "rotated-ragged",
            3,
            13,
            8,
            2,
            true,
            1.,
            true,
            0,
            false,
            ScaleRefitMode::PostPass,
        ),
        (
            "decay-ramp-search",
            2,
            12,
            4,
            3,
            true,
            0.6,
            true,
            3,
            true,
            ScaleRefitMode::PostPass,
        ),
        (
            "in-loop-refit",
            3,
            13,
            4,
            2,
            false,
            0.75,
            false,
            0,
            true,
            ScaleRefitMode::InLoop,
        ),
    ] {
        let (weights, gram) = inputs(rows, cols);
        let config = ActivationAwareConfig {
            planes,
            group,
            grid: 16,
            damp: 0.01,
            search_sweeps: sweeps,
            refit_scale: refit,
            rotate,
            decay,
            decay_ramp: ramp,
        };
        let fits =
            fit_tensor_with_scale_refit_mode(&weights, rows, cols, &gram, &config, mode).unwrap();
        let direct = tritium_quantize::engine::fit_tensor_with_scale_refit_mode(
            &weights, rows, cols, &gram, &config, mode,
        )
        .unwrap();
        let expected = match case {
            "plain" => "5c56efc2a7a033f12ecdf529a8cb2b691f050699d868fb99e631811121a0a581",
            "rotated-ragged" => "8b709b73d3e81039b34e2643ded63367d11d4238a8c6fb42b4816308e502084d",
            "decay-ramp-search" => {
                "596e77a0f1b3621bb0904d3c7f40739e4042758c0870b26f70069c5e01c48bd9"
            }
            "in-loop-refit" => "50ceea338237781a34248613a3b0d0514002083bd882fd9903b2e8d4ac7e0aa9",
            _ => unreachable!(),
        };
        assert_eq!(
            digest(&fits),
            expected,
            "pre-move production fixture {case}"
        );
        assert_eq!(digest(&direct), expected, "shared engine fixture {case}");
    }
}

#[test]
fn shipping_hadamard_has_fixed_output_bits() {
    let mut hash = blake3::Hasher::new();
    for exponent in 0..=12 {
        let len = 1usize << exponent;
        let mut values: Vec<f32> = (0..len)
            .map(|i| (((i * 37 + 19) % 113) as f32 - 56.) / 41.)
            .collect();
        tritium_train::ops::ste::fast_hadamard(&mut values);
        hash.update(&(len as u64).to_le_bytes());
        for value in values {
            hash.update(&value.to_bits().to_le_bytes());
        }
    }
    assert_eq!(
        hash.finalize().to_hex().as_str(),
        "945e5fc340f9ab8ccea9b145894b3322c674c83c063db930125420659f3d77ec"
    );
}
