#![cfg(feature = "cuda")]

use core::convert::Infallible;

use half::f16;
use tritium_cuda::{CudaBackend, train::DeviceTensor};
use tritium_nn::{
    ArchSpec, DenseLinear, DevicePvRecoveryCheckpointArtifact, DevicePvRecoveryError,
    DevicePvRecoveryParentContext, DevicePvRecoverySession, DevicePvRecoveryStepReceipt,
    DevicePvRecoveryWeightVisitError, Mlp, MlpKind, ModelConfig, ModelWeights, Projection,
    SwiGluMlp, TiedSwiGluTrainingModel, TokenEmbedding, TransformerBlock,
};
use tritium_train::{
    AdamW, PvTernaryPlane, PvTernaryStructure, PvTernaryWeight, PvTuningConfig,
    RecoveryActivationRung, RecoveryCampaignPlan, RecoveryCampaignReceipt, RecoveryCampaignRun,
    RecoveryEvidenceDigest, RecoveryPredecessorEvidence, RecoveryPromotionGate,
    RecoverySourceModel, RecoverySourceModelId, RecoveryTrack,
};

fn backend() -> CudaBackend {
    CudaBackend::new(0).expect("CUDA feature lane requires device 0")
}

fn dense(rows: usize, cols: usize, seed: usize) -> Projection {
    let values = (0..rows * cols)
        .map(|index| (((index + seed) % 7) as f32 - 3.0) / 8.0)
        .collect();
    Projection::Dense(DenseLinear::new_exact(values, rows, cols).unwrap())
}

fn model() -> TiedSwiGluTrainingModel {
    let config = ModelConfig {
        arch: "llama".into(),
        n_layers: 1,
        n_embd: 4,
        n_head: 2,
        n_head_kv: 1,
        head_dim: 2,
        n_ff: 4,
        n_ctx: 8,
        rope_theta: 10_000.0,
        rms_eps: 1e-5,
    };
    let spec = ArchSpec {
        mlp: MlpKind::SwiGlu,
        attn_sub_norm: false,
        ffn_sub_norm: false,
        qk_norm: false,
        qkv_bias: false,
        tied_embeddings: true,
    };
    let weights = ModelWeights {
        token_embd: TokenEmbedding::from_dense(
            (0..16).map(|index| (index as f32 - 8.0) / 16.0).collect(),
            4,
            4,
        )
        .unwrap(),
        vocab: 4,
        n_embd: 4,
        layers: vec![TransformerBlock {
            attn_norm: vec![1.0; 4],
            q_proj: dense(4, 4, 1),
            k_proj: dense(2, 4, 2),
            v_proj: dense(2, 4, 3),
            o_proj: dense(4, 4, 4),
            attn_sub_norm: Vec::new(),
            q_bias: Vec::new(),
            k_bias: Vec::new(),
            v_bias: Vec::new(),
            q_norm: Vec::new(),
            k_norm: Vec::new(),
            ffn_norm: vec![1.0; 4],
            mlp: Mlp::SwiGlu(SwiGluMlp {
                gate: dense(4, 4, 5),
                up: dense(4, 4, 6),
                down: dense(4, 4, 7),
            }),
        }],
        output_norm: vec![1.0; 4],
        lm_head: None,
    };
    TiedSwiGluTrainingModel::extract(&config, &spec, &weights).unwrap()
}

fn parent(rows: usize, cols: usize) -> PvTernaryWeight {
    assert_eq!(cols, 4);
    PvTernaryWeight::new(
        rows,
        cols,
        4,
        PvTernaryStructure::S34,
        vec![PvTernaryPlane::new(
            [1, -1, 1, 0].repeat(rows),
            vec![f16::from_f32(0.125); rows],
        )],
    )
    .unwrap()
}

fn config() -> PvTuningConfig {
    let adam = |lr| AdamW {
        lr,
        beta1: 0.0,
        beta2: 0.0,
        eps: 1e-8,
        weight_decay: 0.0,
    };
    PvTuningConfig::builder(adam(0.01), adam(0.05))
        .max_code_change_fraction(0.25)
        .build()
        .unwrap()
}

fn resign(mut checkpoint: Vec<u8>) -> Vec<u8> {
    let body_len = checkpoint.len() - 32;
    let checksum = blake3::hash(&checkpoint[..body_len]);
    checkpoint[body_len..].copy_from_slice(checksum.as_bytes());
    checkpoint
}

fn evidence(byte: u8) -> RecoveryEvidenceDigest {
    RecoveryEvidenceDigest::new([byte; 32]).unwrap()
}

fn recovery_plan(track: RecoveryTrack, campaign: RecoveryEvidenceDigest) -> RecoveryCampaignPlan {
    RecoveryCampaignPlan::new(
        RecoverySourceModel::SmolLm2OneThirtyFiveMillion,
        RecoverySourceModelId::new([200; 32]).unwrap(),
        RecoveryActivationRung::A16,
        track,
        campaign,
        RecoveryPromotionGate::new(-100.0, evidence(201)).unwrap(),
    )
}

fn finish_recovery_campaign(run: RecoveryCampaignRun) -> (RecoveryCampaignReceipt, Vec<u8>) {
    let promotion = run.prepare_promotion_evidence().unwrap();
    let bytes = promotion.to_bytes().unwrap();
    (run.finish(&bytes).unwrap(), bytes)
}

fn predecessor(receipt: &RecoveryCampaignReceipt, promotion: &[u8]) -> RecoveryPredecessorEvidence {
    RecoveryPredecessorEvidence::new(&receipt.to_bytes().unwrap(), promotion).unwrap()
}

fn completed_scale_predecessor(
    campaign: RecoveryEvidenceDigest,
) -> (RecoveryCampaignReceipt, Vec<u8>) {
    let mut ptq =
        RecoveryCampaignRun::start(recovery_plan(RecoveryTrack::Ptq, campaign), None).unwrap();
    ptq.record_ptq(-2.0, evidence(202)).unwrap();
    let (ptq, ptq_promotion) = finish_recovery_campaign(ptq);
    let mut scale = RecoveryCampaignRun::start(
        recovery_plan(RecoveryTrack::ScaleOnly, campaign),
        Some(predecessor(&ptq, &ptq_promotion)),
    )
    .unwrap();
    for (tokens, quality, digest) in [
        (1_000_000, -1.9, evidence(203)),
        (2_000_000, -1.8, evidence(204)),
        (4_000_000, -1.7, evidence(205)),
        (8_000_000, -1.6, evidence(206)),
    ] {
        scale.record_evaluation(tokens, quality, digest).unwrap();
    }
    finish_recovery_campaign(scale)
}

fn authorized_pv_campaign(campaign: RecoveryEvidenceDigest) -> RecoveryCampaignRun {
    let (scale, promotion) = completed_scale_predecessor(campaign);
    RecoveryCampaignRun::start(
        recovery_plan(RecoveryTrack::Pv, campaign),
        Some(predecessor(&scale, &promotion)),
    )
    .unwrap()
}

#[test]
fn model_pv_step_streams_real_gradients_and_accounts_for_all_retained_payloads() {
    let backend = backend();
    let mut model = model();
    let parents = model
        .parameters()
        .iter()
        .map(|parameter| parent(parameter.rows, parameter.cols))
        .collect::<Vec<_>>();
    let parent_digests = parents
        .iter()
        .map(PvTernaryWeight::digest)
        .collect::<Vec<_>>();
    drop(model.take_parameter_masters());
    assert!(
        model
            .parameters()
            .iter()
            .all(|parameter| parameter.master.is_empty())
    );
    let mut session = DevicePvRecoverySession::new_with_gradient_block_elements(
        &backend,
        &model,
        parents.clone(),
        config(),
        3,
    )
    .unwrap();
    let target = DeviceTensor::upload(&backend, &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();

    let receipt = session.step(&[0, 1], &target).unwrap();

    assert_eq!(session.completed_step(), 1);
    assert_eq!(receipt.optimizer_step(), 1);
    assert_eq!(receipt.tensor_receipts().len(), model.parameters().len());
    assert!(receipt.peak_live_gradient_elements() < receipt.materialized_gradient_elements());
    assert_eq!(receipt.peak_host_gradient_elements(), 3);
    assert_eq!(receipt.peak_host_gradient_bytes(), 12);
    assert_eq!(
        receipt.peak_device_gradient_bytes(),
        receipt.peak_live_gradient_elements() * 4
    );
    assert!(receipt.host_representation_bytes() > 0);
    assert!(receipt.host_optimizer_bytes() > receipt.host_representation_bytes());
    assert_eq!(receipt.host_campaign_bytes(), 0);
    assert!(receipt.resident_packed_bytes() > 0);
    assert_ne!(receipt.source_state_digest(), [0; 32]);
    assert_ne!(receipt.batch_digest(), [0; 32]);
    assert_ne!(receipt.evidence_digest(), [0; 32]);
    assert!(!receipt.physical_accounting_complete());
    assert!(!receipt.release_evidence_complete());
    assert_eq!(
        receipt.serialized_checkpoint_bytes(),
        session.checkpoint_bytes().unwrap().len()
    );
    let canonical = receipt.canonical_bytes().unwrap();
    assert_eq!(
        DevicePvRecoveryStepReceipt::from_canonical_bytes(&canonical).unwrap(),
        receipt
    );
    let mut corrupt = canonical.clone();
    let corrupt_index = corrupt.len() / 2;
    corrupt[corrupt_index] ^= 1;
    assert!(DevicePvRecoveryStepReceipt::from_canonical_bytes(&corrupt).is_err());
    let mut trailing = canonical;
    trailing.push(0);
    assert!(DevicePvRecoveryStepReceipt::from_canonical_bytes(&trailing).is_err());

    // TPVR1 fixed prefix: magic, plan, step, count, ten physical counters,
    // then source/batch/representation/evidence digests. Recomputing both
    // identity and checksum must not turn absent source provenance into evidence.
    let mut missing_source = receipt.canonical_bytes().unwrap();
    missing_source[133..165].fill(0);
    let mut evidence = blake3::Hasher::new();
    evidence.update(b"tritium.device-pv-step-evidence.v1\0");
    evidence.update(&missing_source[5..37]);
    evidence.update(&missing_source[133..165]);
    evidence.update(&missing_source[165..197]);
    evidence.update(&missing_source[37..45]);
    evidence.update(&missing_source[197..229]);
    missing_source[229..261].copy_from_slice(evidence.finalize().as_bytes());
    let missing_source = resign(missing_source);
    assert!(DevicePvRecoveryStepReceipt::from_canonical_bytes(&missing_source).is_err());
    let state_before_evaluation = session.checkpoint_bytes().unwrap();
    let held_out = session
        .held_out_teacher_cross_entropy(&[1, 2], &target)
        .unwrap();
    assert!(held_out.is_finite() && held_out >= 0.0);
    assert_eq!(session.checkpoint_bytes().unwrap(), state_before_evaluation);
    let wrong_target = DeviceTensor::upload(&backend, &[1.0]).unwrap();
    assert!(
        session
            .held_out_teacher_cross_entropy(&[1, 2], &wrong_target)
            .is_err()
    );
    assert!(
        receipt
            .tensor_receipts()
            .iter()
            .zip(parent_digests)
            .any(|(receipt, parent)| receipt.representation_digest() != parent)
    );
    for (index, tensor_receipt) in receipt.tensor_receipts().iter().enumerate() {
        assert_eq!(
            session.weight(index).unwrap().digest(),
            tensor_receipt.representation_digest()
        );
        assert_eq!(
            session.weight(index).unwrap().structure(),
            PvTernaryStructure::S34
        );
    }
}

#[test]
fn model_pv_checkpoint_resumes_bit_identically_and_rejects_corruption() {
    let backend = backend();
    let mut model = model();
    let parents = model
        .parameters()
        .iter()
        .map(|parameter| parent(parameter.rows, parameter.cols))
        .collect::<Vec<_>>();
    drop(model.take_parameter_masters());
    let target = DeviceTensor::upload(&backend, &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
    let mut uninterrupted =
        DevicePvRecoverySession::new(&backend, &model, parents.clone(), config()).unwrap();
    uninterrupted.step(&[0, 1], &target).unwrap();
    let checkpoint = uninterrupted.checkpoint_bytes().unwrap();
    let expected = uninterrupted.step(&[1, 2], &target).unwrap();

    let mut resumed =
        DevicePvRecoverySession::resume(&backend, &model, parents.clone(), config(), &checkpoint)
            .unwrap();
    assert_eq!(resumed.completed_step(), 1);
    assert_eq!(resumed.step(&[1, 2], &target).unwrap(), expected);
    for index in 0..parents.len() {
        assert_eq!(
            resumed.weight(index).unwrap(),
            uninterrupted.weight(index).unwrap()
        );
    }

    let mut corrupt = checkpoint.clone();
    corrupt[32] ^= 1;
    assert!(
        DevicePvRecoverySession::resume(&backend, &model, parents.clone(), config(), &corrupt,)
            .is_err()
    );
    let mut trailing = checkpoint;
    trailing.push(0);
    assert!(
        DevicePvRecoverySession::resume(&backend, &model, parents, config(), &trailing).is_err()
    );
}

#[test]
fn model_pv_campaign_resumes_from_a_persisted_mid_tensor_cursor() {
    let backend = backend();
    let mut model = model();
    let parents = model
        .parameters()
        .iter()
        .map(|parameter| parent(parameter.rows, parameter.cols))
        .collect::<Vec<_>>();
    drop(model.take_parameter_masters());
    let target = DeviceTensor::upload(&backend, &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
    let recovery_campaign = authorized_pv_campaign(evidence(206));
    let parent_context = DevicePvRecoveryParentContext::new([207; 32]).unwrap();
    let wrong_parent_context = DevicePvRecoveryParentContext::new([208; 32]).unwrap();

    let mut uninterrupted = DevicePvRecoverySession::new_for_recovery_campaign_with_parent_context_and_gradient_block_elements(
            &backend,
            &model,
            parents.clone(),
            config(),
            &recovery_campaign,
            parent_context,
            3,
        )
        .unwrap();
    let expected = uninterrupted.step(&[0, 1], &target).unwrap();

    let mut interrupted = DevicePvRecoverySession::new_for_recovery_campaign_with_parent_context_and_gradient_block_elements(
            &backend,
            &model,
            parents.clone(),
            config(),
            &recovery_campaign,
            parent_context,
            3,
        )
        .unwrap();
    let base = interrupted.checkpoint_bytes().unwrap();
    let mut campaign = None;
    assert!(
        interrupted
            .step_resumable(&[0, 1], &target, &base, |checkpoint| {
                campaign = Some(checkpoint.to_vec());
                Err(DevicePvRecoveryError::InvalidInput(
                    "intentional persistence stop".into(),
                ))
            })
            .is_err()
    );
    let campaign = campaign.expect("first gradient block must publish a campaign checkpoint");
    assert_eq!(interrupted.campaign_checkpoint_bytes().unwrap(), campaign);
    assert!(interrupted.checkpoint_bytes().is_err());
    assert!(interrupted.step(&[0, 1], &target).is_err());
    assert!(
        DevicePvRecoverySession::resume_campaign(
            &backend,
            &model,
            parents.clone(),
            config(),
            &base,
            &campaign,
        )
        .is_err()
    );
    let wrong_recovery_campaign = authorized_pv_campaign(evidence(205));
    assert!(
        DevicePvRecoverySession::resume_campaign_for_recovery_campaign(
            &backend,
            &model,
            parents.clone(),
            config(),
            &wrong_recovery_campaign,
            &base,
            &campaign,
        )
        .is_err()
    );
    assert!(
        DevicePvRecoverySession::resume_campaign_for_recovery_campaign_with_parent_context(
            &backend,
            &model,
            parents.clone(),
            config(),
            &recovery_campaign,
            wrong_parent_context,
            &base,
            &campaign,
        )
        .is_err()
    );

    let mut resumed =
        DevicePvRecoverySession::resume_campaign_for_recovery_campaign_with_parent_context(
            &backend,
            &model,
            parents.clone(),
            config(),
            &recovery_campaign,
            parent_context,
            &base,
            &campaign,
        )
        .unwrap();
    let got = resumed
        .step_resumable(&[0, 1], &target, &base, |_| Ok(()))
        .unwrap();

    assert_eq!(got, expected);
    assert_eq!(
        resumed.checkpoint_bytes().unwrap(),
        uninterrupted.checkpoint_bytes().unwrap()
    );
    for index in 0..parents.len() {
        assert_eq!(
            resumed.weight(index).unwrap(),
            uninterrupted.weight(index).unwrap()
        );
    }
}

#[test]
fn model_pv_campaign_rejects_corruption_and_wrong_base_or_batch() {
    let backend = backend();
    let mut model = model();
    let parents = model
        .parameters()
        .iter()
        .map(|parameter| parent(parameter.rows, parameter.cols))
        .collect::<Vec<_>>();
    drop(model.take_parameter_masters());
    let target = DeviceTensor::upload(&backend, &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
    let mut interrupted = DevicePvRecoverySession::new_with_gradient_block_elements(
        &backend,
        &model,
        parents.clone(),
        config(),
        3,
    )
    .unwrap();
    let base = interrupted.checkpoint_bytes().unwrap();
    let mut campaign = None;
    interrupted
        .step_resumable(&[0, 1], &target, &base, |checkpoint| {
            campaign = Some(checkpoint.to_vec());
            Err(DevicePvRecoveryError::InvalidInput("stop".into()))
        })
        .unwrap_err();
    let campaign = campaign.unwrap();

    let mut other = DevicePvRecoverySession::new_with_gradient_block_elements(
        &backend,
        &model,
        parents.clone(),
        config(),
        3,
    )
    .unwrap();
    assert_eq!(other.checkpoint_bytes().unwrap(), base);
    let mut other_campaign = None;
    other
        .step_resumable(&[0, 2], &target, &base, |checkpoint| {
            other_campaign = Some(checkpoint.to_vec());
            Err(DevicePvRecoveryError::InvalidInput("stop".into()))
        })
        .unwrap_err();
    let other_campaign = other_campaign.unwrap();
    assert_eq!(other_campaign.len(), campaign.len());
    let mut spliced = campaign.clone();
    let payload_start = 5 + 32 * 4 + 8 * 3;
    let payload_end = campaign.len() - 32;
    spliced[payload_start..payload_end]
        .copy_from_slice(&other_campaign[payload_start..payload_end]);
    assert!(
        DevicePvRecoverySession::resume_campaign(
            &backend,
            &model,
            parents.clone(),
            config(),
            &base,
            &resign(spliced),
        )
        .is_err()
    );

    let mut corrupt = campaign.clone();
    let corrupt_index = corrupt.len() / 2;
    corrupt[corrupt_index] ^= 1;
    assert!(
        DevicePvRecoverySession::resume_campaign(
            &backend,
            &model,
            parents.clone(),
            config(),
            &base,
            &corrupt,
        )
        .is_err()
    );
    let mut wrong_source = campaign.clone();
    let source_digest_offset = 5 + 32 * 2;
    wrong_source[source_digest_offset] ^= 1;
    assert!(
        DevicePvRecoverySession::resume_campaign(
            &backend,
            &model,
            parents.clone(),
            config(),
            &base,
            &resign(wrong_source),
        )
        .is_err()
    );
    let mut hostile_count = campaign.clone();
    let count_offset = 5 + 32 * 4 + 8 * 2;
    hostile_count[count_offset..count_offset + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(
        DevicePvRecoverySession::resume_campaign(
            &backend,
            &model,
            parents.clone(),
            config(),
            &base,
            &resign(hostile_count),
        )
        .is_err()
    );
    let mut trailing = campaign.clone();
    trailing.push(0);
    assert!(
        DevicePvRecoverySession::resume_campaign(
            &backend,
            &model,
            parents.clone(),
            config(),
            &base,
            &trailing,
        )
        .is_err()
    );
    let mut wrong_base = base.clone();
    let corrupt_index = wrong_base.len() / 2;
    wrong_base[corrupt_index] ^= 1;
    assert!(
        DevicePvRecoverySession::resume_campaign(
            &backend,
            &model,
            parents.clone(),
            config(),
            &wrong_base,
            &campaign,
        )
        .is_err()
    );

    let mut resumed = DevicePvRecoverySession::resume_campaign(
        &backend,
        &model,
        parents,
        config(),
        &base,
        &campaign,
    )
    .unwrap();
    assert!(
        resumed
            .step_resumable(&[0, 2], &target, &base, |_| Ok(()))
            .is_err()
    );
    assert_eq!(resumed.campaign_checkpoint_bytes().unwrap(), campaign);
}

#[test]
fn model_pv_checkpoint_artifact_binds_exact_step_and_reopens() {
    let backend = backend();
    let mut model = model();
    let parents = model
        .parameters()
        .iter()
        .map(|parameter| parent(parameter.rows, parameter.cols))
        .collect::<Vec<_>>();
    drop(model.take_parameter_masters());
    let target = DeviceTensor::upload(&backend, &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
    let campaign = authorized_pv_campaign(evidence(207));
    let mut session = DevicePvRecoverySession::new_for_recovery_campaign(
        &backend,
        &model,
        parents.clone(),
        config(),
        &campaign,
    )
    .unwrap();
    assert_eq!(
        session.recovery_campaign_context_digest(),
        Some(campaign.evidence_context_digest())
    );
    let receipt = session.step(&[0, 1], &target).unwrap();

    let artifact = session.checkpoint_artifact(&receipt, &campaign).unwrap();

    assert_eq!(
        artifact.checkpoint_bytes(),
        session.checkpoint_bytes().unwrap()
    );
    assert_eq!(
        artifact.artifact_digest().as_bytes(),
        *blake3::hash(artifact.checkpoint_bytes()).as_bytes()
    );
    assert_ne!(artifact.evidence_digest().as_bytes(), [0; 32]);
    assert_eq!(
        artifact.campaign_context_digest(),
        campaign.evidence_context_digest()
    );
    assert_eq!(artifact.step_receipt(), &receipt);
    assert!(
        DevicePvRecoveryCheckpointArtifact::reopen(
            &session,
            &campaign,
            artifact.checkpoint_bytes().to_vec(),
            artifact.manifest_bytes(),
        )
        .is_err()
    );
    let resumed = DevicePvRecoverySession::resume_for_recovery_campaign(
        &backend,
        &model,
        parents,
        config(),
        &campaign,
        artifact.checkpoint_bytes(),
    )
    .unwrap();
    let reopened = DevicePvRecoveryCheckpointArtifact::reopen(
        &resumed,
        &campaign,
        artifact.checkpoint_bytes().to_vec(),
        artifact.manifest_bytes(),
    )
    .unwrap();
    assert_eq!(reopened.checkpoint_bytes(), artifact.checkpoint_bytes());
    assert_eq!(reopened.manifest_bytes(), artifact.manifest_bytes());
    assert_eq!(reopened.artifact_digest(), artifact.artifact_digest());
    assert_eq!(reopened.evidence_digest(), artifact.evidence_digest());
    assert_eq!(reopened.step_receipt(), &receipt);
    let mut visited = Vec::new();
    let visited_count = reopened
        .try_visit_current_weights(&resumed, |name, weight| {
            visited.push((name.to_owned(), weight.digest()));
            Ok::<_, Infallible>(())
        })
        .unwrap();
    assert_eq!(visited_count, receipt.tensor_receipts().len());
    assert_eq!(visited.len(), receipt.tensor_receipts().len());
    assert_eq!(
        visited
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        model
            .parameters()
            .iter()
            .map(|parameter| parameter.name.as_str())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        visited
            .iter()
            .map(|(_, digest)| *digest)
            .collect::<Vec<_>>(),
        receipt
            .tensor_receipts()
            .iter()
            .map(|tensor| tensor.representation_digest())
            .collect::<Vec<_>>()
    );
    assert!(matches!(
        reopened.try_visit_current_weights(&resumed, |_, _| Err("stop")),
        Err(DevicePvRecoveryWeightVisitError::Visitor("stop"))
    ));
    let reversed_names = model
        .parameters()
        .iter()
        .rev()
        .map(|parameter| parameter.name.as_str())
        .collect::<Vec<_>>();
    let mut reversed = Vec::new();
    assert_eq!(
        reopened
            .try_visit_current_weights_in_order(&resumed, &reversed_names, |name, weight| {
                reversed.push((name.to_owned(), weight.digest()));
                Ok::<_, Infallible>(())
            })
            .unwrap(),
        reversed_names.len()
    );
    assert_eq!(
        reversed.iter().map(|(name, _)| name).collect::<Vec<_>>(),
        reversed_names.iter().rev().rev().collect::<Vec<_>>()
    );
    let mut duplicate_names = reversed_names.clone();
    duplicate_names[0] = duplicate_names[1];
    assert!(matches!(
        reopened.try_visit_current_weights_in_order(&resumed, &duplicate_names, |_, _| Ok::<
            _,
            Infallible,
        >(
            ()
        )),
        Err(DevicePvRecoveryWeightVisitError::Recovery(_))
    ));
    assert!(matches!(
        reopened.try_visit_current_weights_in_order(
            &resumed,
            &reversed_names[..reversed_names.len() - 1],
            |_, _| Ok::<_, Infallible>(())
        ),
        Err(DevicePvRecoveryWeightVisitError::Recovery(_))
    ));
}

#[test]
fn model_pv_checkpoint_artifact_rejects_stale_spliced_or_corrupt_evidence() {
    let backend = backend();
    let mut model = model();
    let parents = model
        .parameters()
        .iter()
        .map(|parameter| parent(parameter.rows, parameter.cols))
        .collect::<Vec<_>>();
    drop(model.take_parameter_masters());
    let target = DeviceTensor::upload(&backend, &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
    let campaign = authorized_pv_campaign(evidence(208));
    let mut session = DevicePvRecoverySession::new_for_recovery_campaign(
        &backend,
        &model,
        parents.clone(),
        config(),
        &campaign,
    )
    .unwrap();
    let first_receipt = session.step(&[0, 1], &target).unwrap();
    let first_artifact = session
        .checkpoint_artifact(&first_receipt, &campaign)
        .unwrap();
    let reopened_first_session = DevicePvRecoverySession::resume_for_recovery_campaign(
        &backend,
        &model,
        parents.clone(),
        config(),
        &campaign,
        first_artifact.checkpoint_bytes(),
    )
    .unwrap();

    let mut foreign = DevicePvRecoverySession::new_for_recovery_campaign(
        &backend,
        &model,
        parents,
        config(),
        &campaign,
    )
    .unwrap();
    let foreign_receipt = foreign.step(&[0, 2], &target).unwrap();
    assert_ne!(
        foreign_receipt.evidence_digest(),
        first_receipt.evidence_digest()
    );
    assert_eq!(
        foreign_receipt.representation_digest(),
        first_receipt.representation_digest()
    );
    let foreign_receipt_bytes = foreign_receipt.canonical_bytes().unwrap();
    let mut spliced_manifest = first_artifact.manifest_bytes().to_vec();
    let receipt_start = 5 + 32 + 32 + 8;
    let receipt_end = spliced_manifest.len() - 32;
    assert_eq!(receipt_end - receipt_start, foreign_receipt_bytes.len());
    spliced_manifest[receipt_start..receipt_end].copy_from_slice(&foreign_receipt_bytes);
    assert!(
        DevicePvRecoveryCheckpointArtifact::reopen(
            &reopened_first_session,
            &campaign,
            first_artifact.checkpoint_bytes().to_vec(),
            &resign(spliced_manifest),
        )
        .is_err()
    );
    let mut forged_receipt_bytes = first_receipt.canonical_bytes().unwrap();
    let host_representation_bytes_offset = 5 + 32 + 8 + 8 + 5 * 8;
    let forged_host_representation_bytes = u64::from_le_bytes(
        forged_receipt_bytes
            [host_representation_bytes_offset..host_representation_bytes_offset + 8]
            .try_into()
            .unwrap(),
    ) + 1;
    forged_receipt_bytes[host_representation_bytes_offset..host_representation_bytes_offset + 8]
        .copy_from_slice(&forged_host_representation_bytes.to_le_bytes());
    let forged_receipt_bytes = resign(forged_receipt_bytes);
    assert!(DevicePvRecoveryStepReceipt::from_canonical_bytes(&forged_receipt_bytes).is_ok());
    let mut forged_physical_manifest = first_artifact.manifest_bytes().to_vec();
    forged_physical_manifest[receipt_start..receipt_end].copy_from_slice(&forged_receipt_bytes);
    assert!(
        DevicePvRecoveryCheckpointArtifact::reopen(
            &reopened_first_session,
            &campaign,
            first_artifact.checkpoint_bytes().to_vec(),
            &resign(forged_physical_manifest),
        )
        .is_err()
    );

    let second_receipt = session.step(&[1, 2], &target).unwrap();
    let second_artifact = session
        .checkpoint_artifact(&second_receipt, &campaign)
        .unwrap();
    assert!(
        first_artifact
            .try_visit_current_weights(&session, |_, _| Ok::<_, Infallible>(()))
            .is_err()
    );

    assert!(
        session
            .checkpoint_artifact(&first_receipt, &campaign)
            .is_err()
    );
    assert!(
        DevicePvRecoveryCheckpointArtifact::reopen(
            &session,
            &campaign,
            second_artifact.checkpoint_bytes().to_vec(),
            first_artifact.manifest_bytes(),
        )
        .is_err()
    );
    let mut corrupt_checkpoint = first_artifact.checkpoint_bytes().to_vec();
    let corrupt_checkpoint_index = corrupt_checkpoint.len() / 2;
    corrupt_checkpoint[corrupt_checkpoint_index] ^= 1;
    assert!(
        DevicePvRecoveryCheckpointArtifact::reopen(
            &reopened_first_session,
            &campaign,
            corrupt_checkpoint,
            first_artifact.manifest_bytes(),
        )
        .is_err()
    );
    let mut corrupt_manifest = first_artifact.manifest_bytes().to_vec();
    let corrupt_manifest_index = corrupt_manifest.len() / 2;
    corrupt_manifest[corrupt_manifest_index] ^= 1;
    assert!(
        DevicePvRecoveryCheckpointArtifact::reopen(
            &reopened_first_session,
            &campaign,
            first_artifact.checkpoint_bytes().to_vec(),
            &corrupt_manifest,
        )
        .is_err()
    );
    let mut trailing_manifest = first_artifact.manifest_bytes().to_vec();
    trailing_manifest.push(0);
    assert!(
        DevicePvRecoveryCheckpointArtifact::reopen(
            &reopened_first_session,
            &campaign,
            first_artifact.checkpoint_bytes().to_vec(),
            &trailing_manifest,
        )
        .is_err()
    );
}

#[test]
fn model_pv_checkpoint_artifact_is_bound_but_not_promotion_evidence() {
    let backend = backend();
    let mut model = model();
    let parents = model
        .parameters()
        .iter()
        .map(|parameter| parent(parameter.rows, parameter.cols))
        .collect::<Vec<_>>();
    drop(model.take_parameter_masters());
    let target = DeviceTensor::upload(&backend, &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
    let campaign_digest = evidence(210);
    let (scale, scale_promotion) = completed_scale_predecessor(campaign_digest);
    let campaign = RecoveryCampaignRun::start(
        recovery_plan(RecoveryTrack::Pv, campaign_digest),
        Some(predecessor(&scale, &scale_promotion)),
    )
    .unwrap();
    let ptq = RecoveryCampaignRun::start(recovery_plan(RecoveryTrack::Ptq, campaign_digest), None)
        .unwrap();
    let mut unbound =
        DevicePvRecoverySession::new(&backend, &model, parents.clone(), config()).unwrap();
    let unbound_receipt = unbound.step(&[0, 1], &target).unwrap();
    assert!(
        unbound
            .checkpoint_artifact(&unbound_receipt, &campaign)
            .is_err()
    );
    let mut session = DevicePvRecoverySession::new_for_recovery_campaign(
        &backend,
        &model,
        parents,
        config(),
        &campaign,
    )
    .unwrap();
    let receipt = session.step(&[0, 1], &target).unwrap();
    assert!(session.checkpoint_artifact(&receipt, &ptq).is_err());
    let artifact = session.checkpoint_artifact(&receipt, &campaign).unwrap();
    let wrong_campaign_digest = evidence(214);
    let wrong_campaign = authorized_pv_campaign(wrong_campaign_digest);
    assert!(
        DevicePvRecoveryCheckpointArtifact::reopen(
            &session,
            &wrong_campaign,
            artifact.checkpoint_bytes().to_vec(),
            artifact.manifest_bytes(),
        )
        .is_err()
    );
    assert_eq!(
        artifact.campaign_context_digest(),
        campaign.evidence_context_digest()
    );
    assert_eq!(
        artifact.artifact_digest().as_bytes(),
        *blake3::hash(artifact.checkpoint_bytes()).as_bytes()
    );
    assert_ne!(artifact.evidence_digest().as_bytes(), [0; 32]);
}

#[test]
fn model_pv_package_parent_context_is_part_of_checkpoint_authority() {
    let backend = backend();
    let mut model = model();
    let parents = model
        .parameters()
        .iter()
        .map(|parameter| parent(parameter.rows, parameter.cols))
        .collect::<Vec<_>>();
    drop(model.take_parameter_masters());
    let campaign = authorized_pv_campaign(evidence(215));
    assert!(DevicePvRecoveryParentContext::new([0; 32]).is_err());
    let parent_context = DevicePvRecoveryParentContext::new([216; 32]).unwrap();
    let wrong_context = DevicePvRecoveryParentContext::new([217; 32]).unwrap();
    let mut session = DevicePvRecoverySession::new_for_recovery_campaign_with_parent_context(
        &backend,
        &model,
        parents.clone(),
        config(),
        &campaign,
        parent_context,
    )
    .unwrap();
    assert_eq!(session.parent_context(), Some(parent_context));
    let target = DeviceTensor::upload(&backend, &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
    let receipt = session.step(&[0, 1], &target).unwrap();
    let artifact = session.checkpoint_artifact(&receipt, &campaign).unwrap();
    let resumed = DevicePvRecoverySession::resume_for_recovery_campaign_with_parent_context(
        &backend,
        &model,
        parents.clone(),
        config(),
        &campaign,
        parent_context,
        artifact.checkpoint_bytes(),
    )
    .unwrap();
    assert_eq!(resumed.parent_context(), Some(parent_context));
    assert!(
        DevicePvRecoverySession::resume_for_recovery_campaign_with_parent_context(
            &backend,
            &model,
            parents.clone(),
            config(),
            &campaign,
            wrong_context,
            artifact.checkpoint_bytes(),
        )
        .is_err()
    );
    assert!(
        DevicePvRecoverySession::resume_for_recovery_campaign(
            &backend,
            &model,
            parents,
            config(),
            &campaign,
            artifact.checkpoint_bytes(),
        )
        .is_err()
    );
}
