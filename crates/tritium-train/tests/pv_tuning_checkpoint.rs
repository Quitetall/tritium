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
