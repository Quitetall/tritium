//! Frozen shipping-fitter bytes captured before shared decay consolidation.

use tritium_nn::salt_fit::{
    ActivationAwareConfig, ScaleRefitMode, fit_tensor_with_scale_refit_mode,
};

#[test]
fn shipping_ladder_fit_retains_original_scale_bits_and_trits() {
    let rows: usize = 2;
    let columns: usize = 16;
    let weights = (0..rows * columns)
        .map(|index| ((index * 7919 % 257) as f32 - 128.0) / 91.0)
        .collect::<Vec<_>>();
    let gram = (0..columns * columns)
        .map(|index| {
            let row = index / columns;
            let column = index % columns;
            0.3_f64.powi(row.abs_diff(column) as i32)
        })
        .collect::<Vec<_>>();
    let mut captured = Vec::new();
    for decay in [1.0, 0.5] {
        for ramp in [false, true] {
            for rotate in [false, true] {
                for mode in [ScaleRefitMode::PostPass, ScaleRefitMode::InLoop] {
                    let config = ActivationAwareConfig {
                        planes: 3,
                        group: 8,
                        grid: 8,
                        damp: 0.01,
                        search_sweeps: 2,
                        refit_scale: true,
                        rotate,
                        decay,
                        decay_ramp: ramp,
                    };
                    let result = fit_tensor_with_scale_refit_mode(
                        &weights, rows, columns, &gram, &config, mode,
                    )
                    .expect("finite fixture fit");
                    assert_eq!(result.len(), 4);
                    let mut digest = blake3::Hasher::new();
                    for (scale, planes) in result {
                        digest.update(&scale.to_bits().to_le_bytes());
                        for plane in planes {
                            for trit in plane {
                                digest.update(&trit.to_le_bytes());
                            }
                        }
                    }
                    captured.push(digest.finalize().to_hex().to_string());
                }
            }
        }
    }
    // Captured from the unchanged fitter at source 33337528 before consolidation.
    // Order is decay, ramp, rotation, then scale-refit mode as above.
    let original = [
        "9ef2d932448b6c8700e19f8de0405956b728c0c8dd17be55929f6833381f4953",
        "5cd95b07efbe8edf20ed89cc3ada97884f266d4799b0f30031f2fce377cd36c7",
        "4c2afaed2cdcd6ef15a0a9404898e168368c40f7991f960254d6f49f68b98adb",
        "bbd1718c0707e20a398f92ee179a40e8e61cfae5810b79825f06b40b17ffea8e",
        "9ef2d932448b6c8700e19f8de0405956b728c0c8dd17be55929f6833381f4953",
        "5cd95b07efbe8edf20ed89cc3ada97884f266d4799b0f30031f2fce377cd36c7",
        "4c2afaed2cdcd6ef15a0a9404898e168368c40f7991f960254d6f49f68b98adb",
        "bbd1718c0707e20a398f92ee179a40e8e61cfae5810b79825f06b40b17ffea8e",
        "8f87dd08ec50ea6be9eb454478478713ec727756f094ec11e446e70e1b91da47",
        "d8edc0fb320438af667a29bcaea7e9bf829bffbcfd480e8434e4618c6bc54103",
        "4c2afaed2cdcd6ef15a0a9404898e168368c40f7991f960254d6f49f68b98adb",
        "52b2dc2d19c36b1889c9546241610736630f17d71632823c7ecfee03fab81ba3",
        "9ef2d932448b6c8700e19f8de0405956b728c0c8dd17be55929f6833381f4953",
        "5b5b41229cfa0602a8cbe031116a58f47a6006469d53a8d7f95ea53b9af16bd3",
        "4c2afaed2cdcd6ef15a0a9404898e168368c40f7991f960254d6f49f68b98adb",
        "58a8e9eb05a766a31d08a81f290efedd082f9b32d1d0c3e77d78531ac67b69dd",
    ];
    assert_eq!(
        captured.iter().map(String::as_str).collect::<Vec<_>>(),
        original
    );
}

#[test]
fn invalid_decay_rejects_before_fitting_instead_of_clamping_or_propagating_nan() {
    for decay in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.01, 1.01] {
        let config = ActivationAwareConfig {
            decay,
            ..ActivationAwareConfig::default()
        };
        assert!(
            fit_tensor_with_scale_refit_mode(
                &[0.25],
                1,
                1,
                &[1.0],
                &config,
                ScaleRefitMode::PostPass,
            )
            .is_none()
        );
    }
}
