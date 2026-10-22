use half::f16;
use tritium_train::{
    AdamW, PvTernaryPlane, PvTernaryStructure, PvTernaryWeight, PvTuningConfig, PvTuningError,
    PvTuningSession,
};

fn adam(lr: f32) -> AdamW {
    AdamW {
        lr,
        beta1: 0.0,
        beta2: 0.0,
        eps: 1e-8,
        weight_decay: 0.0,
    }
}

fn recipe(fraction: f32, trust_ratio: Option<f32>) -> PvTuningConfig {
    let builder = PvTuningConfig::builder(adam(0.05), adam(0.4)).max_code_change_fraction(fraction);
    match trust_ratio {
        Some(ratio) => builder.max_relative_code_change(ratio).build().unwrap(),
        None => builder.build().unwrap(),
    }
}

fn parent() -> PvTernaryWeight {
    PvTernaryWeight::new(
        1,
        4,
        4,
        PvTernaryStructure::Dense,
        vec![PvTernaryPlane::new(vec![1, -1, 1, -1], vec![f16::ONE])],
    )
    .unwrap()
}

#[test]
fn accepted_extreme_finite_gradient_steps_must_reopen_their_own_checkpoint() {
    let parent = PvTernaryWeight::new(
        1,
        2,
        2,
        PvTernaryStructure::Dense,
        vec![PvTernaryPlane::new(vec![1, -1], vec![f16::ONE])],
    )
    .unwrap();
    let config = recipe(0.5, None);
    let mut ordinary = PvTuningSession::new(parent.clone(), config).unwrap();
    ordinary.step(&[1e18, 1e18], 1).unwrap();
    assert_eq!(ordinary.weight().digest(), parent.digest());
    let valid_checkpoint = ordinary.checkpoint_bytes().unwrap();
    assert!(PvTuningSession::resume(parent.clone(), config, &valid_checkpoint).is_ok());
    let mut blocked = PvTuningSession::new(parent.clone(), config).unwrap();
    blocked.begin_blockwise_step(1, 2).unwrap();
    let before_block = blocked.blockwise_checkpoint_bytes().unwrap();
    assert!(matches!(
        blocked.apply_gradient_block(0, &[f32::MAX, f32::MAX]),
        Err(PvTuningError::Step(_))
    ));
    assert_eq!(blocked.blockwise_checkpoint_bytes().unwrap(), before_block);
    let mut session = PvTuningSession::new(parent.clone(), config).unwrap();
    let before = session.checkpoint_bytes().unwrap();
    // Equal gradients cancel in the scale derivative for these opposite trits.
    // Both input elements themselves remain finite.
    let result = session.step(&[f32::MAX, f32::MAX], 1);
    match result {
        Ok(receipt) => {
            assert_eq!(session.weight().digest(), parent.digest());
            let checkpoint = session.checkpoint_bytes().unwrap();
            let reopened = PvTuningSession::resume(parent, config, &checkpoint);
            assert!(
                reopened.is_ok(),
                "accepted step {receipt:?} produced an unreopenable checkpoint: {reopened:?}"
            );
        }
        Err(PvTuningError::Step(message)) => {
            assert_eq!(message, "code optimizer moment update became non-finite");
            assert_eq!(session.completed_step(), 0);
            assert_eq!(session.checkpoint_bytes().unwrap(), before);
            let restored = PvTuningSession::resume(parent, config, &before).unwrap();
            assert_eq!(restored.state_digest(), session.state_digest());
            session.step(&[1e18, 1e18], 1).unwrap();
            assert_eq!(session.checkpoint_bytes().unwrap(), valid_checkpoint);
        }
        Err(other) => panic!("unexpected step failure: {other:?}"),
    }
}

fn scale_moment_checkpoint_safety(blockwise: bool) {
    let parent = PvTernaryWeight::new(
        1,
        2,
        2,
        PvTernaryStructure::Dense,
        vec![PvTernaryPlane::new(vec![1, -1], vec![f16::ONE])],
    )
    .unwrap();
    let config = recipe(0.5, None);
    let mut ordinary = PvTuningSession::new(parent.clone(), config).unwrap();
    ordinary.step(&[1e18, -1e18], 1).unwrap();
    let valid_checkpoint = ordinary.checkpoint_bytes().unwrap();
    assert!(PvTuningSession::resume(parent.clone(), config, &valid_checkpoint).is_ok());

    let mut session = PvTuningSession::new(parent.clone(), config).unwrap();
    let before_step = session.checkpoint_bytes().unwrap();
    let before_attempt = if blockwise {
        session.begin_blockwise_step(1, 2).unwrap();
        // Individual code moments remain finite; their trit-weighted scale
        // derivative is larger and can overflow its squared moment.
        session.apply_gradient_block(0, &[1e19, -1e19]).unwrap();
        session.blockwise_checkpoint_bytes().unwrap()
    } else {
        before_step.clone()
    };
    let result = if blockwise {
        session.finish_blockwise_step()
    } else {
        session.step(&[1e19, -1e19], 1)
    };
    match result {
        Ok(receipt) => {
            let checkpoint = session.checkpoint_bytes().unwrap();
            let reopened = PvTuningSession::resume(parent, config, &checkpoint);
            assert!(
                reopened.is_ok(),
                "accepted scale update blockwise={blockwise} {receipt:?} produced an unreopenable checkpoint: {reopened:?}"
            );
        }
        Err(PvTuningError::Step(message)) => {
            assert_eq!(
                message,
                "continuous optimizer moment update became non-finite"
            );
            assert_eq!(session.completed_step(), 0);
            if blockwise {
                assert_eq!(
                    session.blockwise_checkpoint_bytes().unwrap(),
                    before_attempt
                );
                let restored =
                    PvTuningSession::resume_blockwise(parent.clone(), config, &before_attempt)
                        .unwrap();
                assert_eq!(restored.state_digest(), session.state_digest());
            } else {
                assert_eq!(session.checkpoint_bytes().unwrap(), before_step);
            }
            let mut retry = PvTuningSession::resume(parent, config, &before_step).unwrap();
            retry.step(&[1e18, -1e18], 1).unwrap();
            assert_eq!(retry.checkpoint_bytes().unwrap(), valid_checkpoint);
        }
        Err(other) => panic!("unexpected scale update failure: {other:?}"),
    }
}

#[test]
fn whole_scale_moment_overflow_does_not_commit_unreopenable_state() {
    scale_moment_checkpoint_safety(false);
}

#[test]
fn blockwise_scale_moment_overflow_does_not_commit_unreopenable_state() {
    scale_moment_checkpoint_safety(true);
}

#[test]
fn moment_overflow_after_resume_preserves_last_good_step_and_retry() {
    let parent = PvTernaryWeight::new(
        1,
        2,
        2,
        PvTernaryStructure::Dense,
        vec![PvTernaryPlane::new(vec![1, -1], vec![f16::ONE])],
    )
    .unwrap();
    let optimizer = AdamW {
        lr: 0.001,
        beta1: 0.9,
        beta2: 0.99,
        eps: 1e-8,
        weight_decay: 0.0,
    };
    let config = PvTuningConfig::builder(optimizer, optimizer)
        .max_code_change_fraction(0.5)
        .build()
        .unwrap();
    let mut session = PvTuningSession::new(parent.clone(), config).unwrap();
    for step in 1..=3 {
        session.step(&[1e20, 1e20], step).unwrap();
    }
    let last_good = session.checkpoint_bytes().unwrap();
    let mut resumed = PvTuningSession::resume(parent.clone(), config, &last_good).unwrap();
    assert_eq!(resumed.completed_step(), 3);
    assert_eq!(
        resumed.step(&[1e20, 1e20], 4),
        Err(PvTuningError::Step(
            "code optimizer moment update became non-finite".into()
        ))
    );
    assert_eq!(resumed.completed_step(), 3);
    assert_eq!(resumed.checkpoint_bytes().unwrap(), last_good);
    let mut control = PvTuningSession::resume(parent.clone(), config, &last_good).unwrap();
    let expected = control.step(&[1e18, 1e18], 4).unwrap();
    assert_eq!(resumed.step(&[1e18, 1e18], 4).unwrap(), expected);
    let final_bytes = resumed.checkpoint_bytes().unwrap();
    assert_eq!(final_bytes, control.checkpoint_bytes().unwrap());
    let reopened = PvTuningSession::resume(parent, config, &final_bytes).unwrap();
    assert_eq!(reopened.state_digest(), resumed.state_digest());
}

#[test]
fn checkpoint_resume_is_bit_identical_and_identity_bound() {
    let parent = parent();
    let config = recipe(0.5, Some(0.75));
    let gradients = [[0.75, -0.5, 0.25, -0.125], [-0.25, 0.75, -0.5, 0.25]];

    let mut uninterrupted = PvTuningSession::new(parent.clone(), config).unwrap();
    let first = uninterrupted.step(&gradients[0], 1).unwrap();
    let receipt_bytes = first.checkpoint_bytes();
    assert_eq!(
        tritium_train::PvStepReceipt::resume(&receipt_bytes).unwrap(),
        first
    );
    let mut corrupt_receipt = receipt_bytes.clone();
    let corrupt_index = corrupt_receipt.len() / 2;
    corrupt_receipt[corrupt_index] ^= 1;
    assert!(tritium_train::PvStepReceipt::resume(&corrupt_receipt).is_err());
    let mut trailing_receipt = receipt_bytes;
    trailing_receipt.push(0);
    assert!(tritium_train::PvStepReceipt::resume(&trailing_receipt).is_err());
    let checkpoint = uninterrupted.checkpoint_bytes().unwrap();
    let second = uninterrupted.step(&gradients[1], 2).unwrap();

    let mut resumed = PvTuningSession::resume(parent.clone(), config, &checkpoint).unwrap();
    assert_eq!(resumed.completed_step(), 1);
    assert_eq!(resumed.weight().digest(), first.representation_digest());
    let resumed_second = resumed.step(&gradients[1], 2).unwrap();
    assert_eq!(resumed_second, second);
    assert_eq!(resumed.weight(), uninterrupted.weight());

    assert!(matches!(
        PvTuningSession::resume(parent.clone(), recipe(0.25, Some(0.75)), &checkpoint),
        Err(PvTuningError::Checkpoint(_))
    ));
    let mut trailing = checkpoint.clone();
    trailing.push(0);
    assert!(matches!(
        PvTuningSession::resume(parent.clone(), config, &trailing),
        Err(PvTuningError::Checkpoint(_))
    ));
    let mut tampered = uninterrupted.checkpoint_bytes().unwrap();
    tampered[80] ^= 1;
    assert!(matches!(
        PvTuningSession::resume(parent.clone(), config, &tampered),
        Err(PvTuningError::Checkpoint(_))
    ));
    let other_parent = PvTernaryWeight::new(
        1,
        4,
        4,
        PvTernaryStructure::Dense,
        vec![PvTernaryPlane::new(vec![1, -1, 0, -1], vec![f16::ONE])],
    )
    .unwrap();
    assert!(matches!(
        PvTuningSession::resume(other_parent, config, &checkpoint),
        Err(PvTuningError::Checkpoint(_))
    ));
}

#[test]
fn blockwise_campaign_checkpoint_resumes_inside_a_tensor_bit_identically() {
    let parent = parent();
    let config = recipe(0.5, Some(0.75));
    let gradient = [0.75, -0.5, 0.25, -0.125];
    let mut uninterrupted = PvTuningSession::new(parent.clone(), config).unwrap();
    let expected = uninterrupted.step(&gradient, 1).unwrap();

    let mut partial = PvTuningSession::new(parent.clone(), config).unwrap();
    let source_state_digest = partial.state_digest();
    partial.begin_blockwise_step(1, 3).unwrap();
    partial.apply_gradient_block(0, &gradient[..3]).unwrap();
    assert_ne!(partial.state_digest(), source_state_digest);
    let checkpoint = partial.blockwise_checkpoint_bytes().unwrap();
    let ledger = partial.size_ledger().unwrap();
    assert_eq!(ledger.host_representation_bytes(), 6);
    assert_eq!(ledger.host_optimizer_bytes(), 40);
    assert_eq!(ledger.host_campaign_bytes(), 8);
    assert_eq!(ledger.serialized_checkpoint_bytes(), checkpoint.len());

    let mut resumed =
        PvTuningSession::resume_blockwise(parent.clone(), config, &checkpoint).unwrap();
    assert_eq!(resumed.completed_step(), 0);
    assert_eq!(resumed.blockwise_cursor().unwrap().next_offset(), 3);
    assert_eq!(resumed.state_digest(), partial.state_digest());
    resumed.apply_gradient_block(3, &gradient[3..]).unwrap();
    assert_eq!(resumed.finish_blockwise_step().unwrap(), expected);
    assert_eq!(resumed.weight(), uninterrupted.weight());
    assert_eq!(resumed.state_digest(), uninterrupted.state_digest());
    assert_eq!(
        resumed.checkpoint_bytes().unwrap(),
        uninterrupted.checkpoint_bytes().unwrap()
    );
    let ledger = resumed.size_ledger().unwrap();
    assert_eq!(ledger.host_campaign_bytes(), 0);
    assert_eq!(
        ledger.serialized_checkpoint_bytes(),
        resumed.checkpoint_bytes().unwrap().len()
    );

    let mut corrupt = checkpoint.clone();
    let corrupt_index = corrupt.len() / 2;
    corrupt[corrupt_index] ^= 1;
    assert!(PvTuningSession::resume_blockwise(parent.clone(), config, &corrupt).is_err());
    let mut trailing = checkpoint;
    trailing.push(0);
    assert!(PvTuningSession::resume_blockwise(parent, config, &trailing).is_err());
}

fn resign(mut checkpoint: Vec<u8>) -> Vec<u8> {
    let body_len = checkpoint.len() - 32;
    let checksum = blake3::hash(&checkpoint[..body_len]);
    checkpoint[body_len..].copy_from_slice(checksum.as_bytes());
    checkpoint
}

#[test]
fn parser_rejects_signed_corrupt_payloads_and_every_truncation() {
    let config = recipe(0.5, None);
    let parent = parent();
    let mut session = PvTuningSession::new(parent.clone(), config).unwrap();
    session.step(&[0.75, -0.5, 0.25, -0.125], 1).unwrap();
    let checkpoint = session.checkpoint_bytes().unwrap();

    for length in 0..checkpoint.len() {
        assert!(matches!(
            PvTuningSession::resume(parent.clone(), config, &checkpoint[..length]),
            Err(PvTuningError::Checkpoint(_))
        ));
    }

    // TPV1 fixed header is 103 bytes; first plane then stores four trits and one f16.
    let mut bad_trit = checkpoint.clone();
    bad_trit[103] = 2;
    assert!(PvTuningSession::resume(parent.clone(), config, &resign(bad_trit)).is_err());
    let mut infinite_scale = checkpoint.clone();
    infinite_scale[107..109].copy_from_slice(&0x7c00u16.to_le_bytes());
    assert!(PvTuningSession::resume(parent.clone(), config, &resign(infinite_scale)).is_err());
    let mut nonfinite_moment = checkpoint.clone();
    nonfinite_moment[109..113].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(PvTuningSession::resume(parent.clone(), config, &resign(nonfinite_moment)).is_err());
    let mut negative_second_moment = checkpoint.clone();
    negative_second_moment[125..129].copy_from_slice(&(-1.0f32).to_le_bytes());
    assert!(
        PvTuningSession::resume(parent.clone(), config, &resign(negative_second_moment)).is_err()
    );
    let mut wrong_geometry = checkpoint.clone();
    wrong_geometry[77..85].copy_from_slice(&2u64.to_le_bytes());
    assert!(PvTuningSession::resume(parent.clone(), config, &resign(wrong_geometry)).is_err());

    let body_len = checkpoint.len() - 32;
    let mut extra_body_byte = checkpoint[..body_len].to_vec();
    extra_body_byte.push(0);
    let checksum = blake3::hash(&extra_body_byte);
    extra_body_byte.extend_from_slice(checksum.as_bytes());
    assert!(PvTuningSession::resume(parent, config, &extra_body_byte).is_err());
}
