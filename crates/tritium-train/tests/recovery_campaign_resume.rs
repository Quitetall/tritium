use tritium_train::{
    RecoveryActivationRung, RecoveryCampaignDecision, RecoveryCampaignPlan, RecoveryCampaignRun,
    RecoveryCampaignTermination, RecoveryEvidenceDigest, RecoveryModelRung,
    RecoveryPredecessorEvidence, RecoveryPromotionGate, RecoverySourceModel, RecoverySourceModelId,
    RecoveryTrack,
};

fn digest(seed: u8) -> RecoveryEvidenceDigest {
    RecoveryEvidenceDigest::new([seed; 32]).unwrap()
}

fn source_id(seed: u8) -> RecoverySourceModelId {
    RecoverySourceModelId::new([seed; 32]).unwrap()
}

fn plan(track: RecoveryTrack, campaign: RecoveryEvidenceDigest) -> RecoveryCampaignPlan {
    RecoveryCampaignPlan::new(
        RecoverySourceModel::Qwen36TwentySevenBillion,
        source_id(1),
        RecoveryActivationRung::A16,
        track,
        campaign,
        RecoveryPromotionGate::new(0.0, digest(2)).unwrap(),
    )
}

fn finish(run: RecoveryCampaignRun) -> (Vec<u8>, Vec<u8>) {
    let promotion = run.prepare_promotion_evidence().unwrap();
    let promotion_bytes = promotion.to_bytes().unwrap();
    let receipt = run.finish(&promotion_bytes).unwrap();
    (receipt.to_bytes().unwrap(), promotion_bytes)
}

#[test]
fn qwen36_27b_has_its_exact_frozen_stage5_budget_identity() {
    let source = RecoverySourceModel::Qwen36TwentySevenBillion;
    assert_eq!(source.model_rung(), RecoveryModelRung::Qwen27B);
    assert_eq!(
        source.model_rung().token_cap(RecoveryTrack::ScaleOnly),
        Some(64_000_000)
    );
    assert_eq!(
        source.model_rung().token_cap(RecoveryTrack::Pv),
        Some(512_000_000)
    );
}

#[test]
fn active_campaign_checkpoint_replays_exact_progress_and_rejects_drift() {
    let campaign = digest(3);
    let ptq_plan = plan(RecoveryTrack::Ptq, campaign);
    let mut ptq = RecoveryCampaignRun::start(ptq_plan, None).unwrap();
    assert_eq!(
        ptq.record_ptq(0.5, digest(4)).unwrap(),
        RecoveryCampaignDecision::Complete(RecoveryCampaignTermination::PtqComplete)
    );
    let (ptq_receipt, ptq_promotion) = finish(ptq);
    let predecessor = RecoveryPredecessorEvidence::new(&ptq_receipt, &ptq_promotion).unwrap();

    let scale_plan = plan(RecoveryTrack::ScaleOnly, campaign);
    let mut uninterrupted = RecoveryCampaignRun::start(scale_plan, Some(predecessor)).unwrap();
    let mut cloned_fresh = uninterrupted.clone();
    for run in [&mut uninterrupted, &mut cloned_fresh] {
        assert_eq!(
            run.record_evaluation(8_000_000, 0.6, digest(5)).unwrap(),
            RecoveryCampaignDecision::Continue
        );
    }
    assert_eq!(
        cloned_fresh.checkpoint_bytes().unwrap(),
        uninterrupted.checkpoint_bytes().unwrap()
    );
    let mut cloned_partial = uninterrupted.clone();
    for run in [&mut uninterrupted, &mut cloned_fresh, &mut cloned_partial] {
        assert_eq!(
            run.record_evaluation(16_000_000, 0.7, digest(6)).unwrap(),
            RecoveryCampaignDecision::Continue
        );
    }
    assert_eq!(
        cloned_fresh.checkpoint_bytes().unwrap(),
        uninterrupted.checkpoint_bytes().unwrap()
    );
    assert_eq!(
        cloned_partial.checkpoint_bytes().unwrap(),
        uninterrupted.checkpoint_bytes().unwrap()
    );

    let checkpoint = uninterrupted.checkpoint_bytes().unwrap();
    let mut resumed =
        RecoveryCampaignRun::reopen_checkpoint(scale_plan, Some(predecessor), &checkpoint).unwrap();
    assert_eq!(resumed.checkpoint_bytes().unwrap(), checkpoint);

    for run in [&mut uninterrupted, &mut resumed] {
        assert_eq!(
            run.record_evaluation(32_000_000, 0.7, digest(7)).unwrap(),
            RecoveryCampaignDecision::Continue
        );
        assert_eq!(
            run.record_evaluation(64_000_000, 0.8, digest(8)).unwrap(),
            RecoveryCampaignDecision::Complete(RecoveryCampaignTermination::TokenCapReached)
        );
    }
    let terminal_checkpoint = resumed.checkpoint_bytes().unwrap();
    let terminal_reopened =
        RecoveryCampaignRun::reopen_checkpoint(scale_plan, Some(predecessor), &terminal_checkpoint)
            .unwrap();
    let uninterrupted_receipt = finish(uninterrupted).0;
    let resumed_receipt = finish(resumed).0;
    let terminal_reopened_receipt = finish(terminal_reopened).0;
    assert_eq!(resumed_receipt, uninterrupted_receipt);
    assert_eq!(terminal_reopened_receipt, uninterrupted_receipt);

    let mut corrupt = checkpoint.clone();
    corrupt[50] ^= 1;
    assert!(
        RecoveryCampaignRun::reopen_checkpoint(scale_plan, Some(predecessor), &corrupt,).is_err()
    );
    for end in 0..checkpoint.len() {
        assert!(
            RecoveryCampaignRun::reopen_checkpoint(
                scale_plan,
                Some(predecessor),
                &checkpoint[..end],
            )
            .is_err(),
            "truncation at {end} reopened"
        );
    }
    let mut trailing = checkpoint.clone();
    trailing.push(0);
    assert!(
        RecoveryCampaignRun::reopen_checkpoint(scale_plan, Some(predecessor), &trailing,).is_err()
    );
    let mut rehashed_invalid_history = checkpoint.clone();
    let first_observation_tag = 8 + 2 + 32 + 1;
    rehashed_invalid_history[first_observation_tag] = 0;
    let body_len = rehashed_invalid_history.len() - 32;
    let checksum = blake3::hash(&rehashed_invalid_history[..body_len]);
    rehashed_invalid_history[body_len..].copy_from_slice(checksum.as_bytes());
    assert!(
        RecoveryCampaignRun::reopen_checkpoint(
            scale_plan,
            Some(predecessor),
            &rehashed_invalid_history,
        )
        .is_err()
    );
    let wrong_plan = RecoveryCampaignPlan::new(
        RecoverySourceModel::Qwen3ThirtyTwoBillion,
        source_id(1),
        RecoveryActivationRung::A16,
        RecoveryTrack::ScaleOnly,
        campaign,
        RecoveryPromotionGate::new(0.0, digest(2)).unwrap(),
    );
    assert!(
        RecoveryCampaignRun::reopen_checkpoint(wrong_plan, Some(predecessor), &checkpoint,)
            .is_err()
    );
}
