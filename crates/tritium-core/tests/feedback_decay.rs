//! Shared decay policy is exercised through the same interface as fitters.

use tritium_core::{FeedbackDecay, FeedbackDecayError};

#[test]
fn constant_and_column_ramp_preserve_shipping_arithmetic() {
    for target in [0.0, 0.001, 0.5, 0.75, 1.0] {
        for columns in [1, 2, 3, 64, 128, 257] {
            for ramp in [false, true] {
                let policy = FeedbackDecay::new(target, ramp).unwrap();
                for column in 0..columns {
                    let original = if ramp && columns > 1 {
                        1.0 - (1.0 - target) * column as f64 / (columns - 1) as f64
                    } else {
                        target
                    };
                    assert_eq!(
                        policy.coefficient(column, columns).unwrap().to_bits(),
                        original.to_bits()
                    );
                }
            }
        }
    }
}

#[test]
fn malformed_policy_or_column_is_rejected_without_clamping() {
    for target in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.01, 1.01] {
        assert_eq!(
            FeedbackDecay::new(target, false),
            Err(FeedbackDecayError::InvalidTarget)
        );
    }
    let policy = FeedbackDecay::default();
    assert_eq!(
        policy.coefficient(0, 0),
        Err(FeedbackDecayError::InvalidColumn {
            column: 0,
            columns: 0
        })
    );
    assert_eq!(
        policy.coefficient(2, 2),
        Err(FeedbackDecayError::InvalidColumn {
            column: 2,
            columns: 2
        })
    );
    assert_eq!(policy.coefficient(0, 1).unwrap(), 1.0);
}

#[cfg(feature = "std")]
#[test]
fn automatic_calibration_rule_is_bit_exact_and_bounded() {
    use tritium_core::auto_feedback_decay;
    assert!((auto_feedback_decay(4096, 1536) - 0.5).abs() < 0.01);
    assert!((auto_feedback_decay(16384, 1536) - 0.75).abs() < 0.01);
    let mut previous = 0.0;
    for tokens in [512, 1024, 2048, 4096, 8192, 16384, 32768] {
        let value = auto_feedback_decay(tokens, 576);
        assert!(value >= previous);
        previous = value;
    }
    for tokens in [0, 512, 1024, 2048, 4096, 8192, 16384, 65536, usize::MAX] {
        for columns in [0, 1, 576, 1536, usize::MAX] {
            let original = if tokens == 0 || columns == 0 {
                0.5
            } else {
                let ratio = tokens as f64 / columns as f64;
                let t =
                    ((ratio.ln() - 2.7_f64.ln()) / (10.7_f64.ln() - 2.7_f64.ln())).clamp(0.0, 1.0);
                0.5 + t * (0.75 - 0.5)
            };
            let observed = auto_feedback_decay(tokens, columns);
            assert_eq!(observed.to_bits(), original.to_bits());
            assert!((0.5..=0.75).contains(&observed));
        }
    }
}
