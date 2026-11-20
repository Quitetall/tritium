//! Frozen legacy whole-session results captured before sharing code projection.
use half::f16;
use tritium_train::{
    AdamW, PvTernaryPlane, PvTernaryStructure, PvTernaryWeight, PvTuningConfig, PvTuningSession,
};

#[test]
fn legacy_projection_session_vectors() {
    for sparse in [false, true] {
        for planes in 1..=3 {
            let parent = PvTernaryWeight::new(
                2,
                32,
                8,
                if sparse {
                    PvTernaryStructure::S34
                } else {
                    PvTernaryStructure::Dense
                },
                (0..planes)
                    .map(|p| {
                        PvTernaryPlane::new(
                            (0..64)
                                .map(|i| {
                                    if sparse {
                                        if i % 4 == p % 4 {
                                            0
                                        } else if (i + p) % 2 == 0 {
                                            -1
                                        } else {
                                            1
                                        }
                                    } else {
                                        [-1, 0, 1][(i + p) % 3]
                                    }
                                })
                                .collect(),
                            (0..8)
                                .map(|i| f16::from_f32((1.0 + (i % 3) as f32) / (2 << p) as f32))
                                .collect(),
                        )
                    })
                    .collect(),
            )
            .unwrap();
            let adam = |lr| AdamW {
                lr,
                beta1: 0.8,
                beta2: 0.9,
                eps: 1e-5,
                weight_decay: 0.01,
            };
            let config = PvTuningConfig::builder(adam(0.01), adam(1.3))
                .max_code_change_fraction(0.75)
                .max_relative_code_change(0.9)
                .build()
                .unwrap();
            let mut session = PvTuningSession::new(parent, config).unwrap();
            for step in 1..=4 {
                let gradient: Vec<f32> = (0..64)
                    .map(|i| (((i * 7 + step * 3) % 13) as f32 - 6.0) * 0.125)
                    .collect();
                session.step(&gradient, step as u64).unwrap();
            }
            let expected = if sparse {
                [
                    "c12bcc903126c5f55a76da80403271684a68816ea7501b5db4fa542cd3f43fd7",
                    "caa6e91f0856dbdff7687348c2531f5e9b1dd28d3fd890915edcc655a469be82",
                    "323569873cf9052016aa3482059499fb81ddc8f0b41eb4f06391ef89e7845264",
                ]
            } else {
                [
                    "4eea84289093fb749cab9a1956d6554524b88cd8e4ffa88eae474e45ace26466",
                    "7b39a8a381968142737e116247cd22e1ab65fdc8b9a3b03b445481482d0d180c",
                    "c8011eb563ba1cfa193cc7a04b45530b623b9d559e73f12b60bdf30cac9e528b",
                ]
            };
            assert_eq!(
                blake3::hash(&session.checkpoint_bytes().unwrap())
                    .to_hex()
                    .as_str(),
                expected[planes - 1],
                "sparse={sparse} planes={planes}"
            );
        }
    }
}

#[test]
fn blockwise_nonfinite_legacy_proposal_preserves_no_candidate_semantics() {
    for sparse in [false, true] {
        let parent = PvTernaryWeight::new(
            1,
            4,
            4,
            if sparse {
                PvTernaryStructure::S34
            } else {
                PvTernaryStructure::Dense
            },
            vec![PvTernaryPlane::new(
                if sparse { vec![0, 1, 1, 1] } else { vec![1; 4] },
                vec![f16::ONE],
            )],
        )
        .unwrap();
        let adam = |lr| AdamW {
            lr,
            beta1: 0.0,
            beta2: 0.0,
            eps: 1e-8,
            weight_decay: 0.0,
        };
        let config = PvTuningConfig::builder(
            adam(0.001),
            AdamW {
                eps: f32::MIN_POSITIVE,
                ..adam(f32::MAX)
            },
        )
        .max_code_change_fraction(1.0)
        .build()
        .unwrap();
        let mut session = PvTuningSession::new(parent.clone(), config).unwrap();
        session.begin_blockwise_step(1, 4).unwrap();
        session.apply_gradient_block(0, &[1e-30; 4]).unwrap();
        // Legacy blockwise proposal arithmetic can overflow although its stored
        // moments remain finite. It historically found no improving candidate,
        // rather than panicking or changing checkpoint/optimizer semantics.
        let receipt = session.finish_blockwise_step().unwrap();
        assert_eq!(receipt.changed_code_units(), 0);
        assert!(receipt.v_surrogate_before().is_infinite());
        assert!(receipt.v_surrogate_after().is_infinite());
        assert_eq!(
            session.weight().planes()[0].trits(),
            parent.planes()[0].trits()
        );
        assert_eq!(session.completed_step(), 1);
    }
}
