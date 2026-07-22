use tritium_quantize::{
    CurvatureSourceId, MAX_KRONECKER_REDUCTION_SEGMENTS, SaltV2Curvature, SaltV2KroneckerEvidence,
    SaltV2KroneckerEvidenceBuildError, SaltV2KroneckerEvidenceBuilder, SaltV2KroneckerEvidenceSpec,
};

const GROUP: usize = 128;

fn source() -> CurvatureSourceId {
    CurvatureSourceId::new([1; 32], [2; 32], [3; 32]).unwrap()
}

fn spec(kind: SaltV2Curvature) -> SaltV2KroneckerEvidenceSpec {
    SaltV2KroneckerEvidenceSpec::new(kind, source(), 7, "layer.proj.weight", 2, 2 * GROUP, 0.25)
        .unwrap()
}

fn samples() -> (Vec<f32>, Vec<f32>) {
    let mut activations = Vec::with_capacity(4 * GROUP);
    activations.extend(std::iter::repeat_n(1.0, GROUP));
    activations.extend(std::iter::repeat_n(2.0, GROUP));
    activations.extend(std::iter::repeat_n(3.0, GROUP));
    activations.extend(std::iter::repeat_n(4.0, GROUP));
    (activations, vec![1.0, 2.0, 3.0, 4.0])
}

#[test]
fn guided_fisher_batches_produce_worked_kronecker_factors() {
    let (activations, gradients) = samples();
    let mut builder =
        SaltV2KroneckerEvidenceBuilder::new(spec(SaltV2Curvature::GuidedFisher)).unwrap();
    builder
        .accumulate_batch(&activations, Some(&gradients), 2, None, None)
        .unwrap();
    let evidence = builder.finish().unwrap();

    assert_eq!(evidence.tensor_index(), 7);
    assert_eq!(evidence.tensor_name(), "layer.proj.weight");
    assert_eq!(evidence.source_id(), source());
    assert_eq!(evidence.kind(), SaltV2Curvature::GuidedFisher);
    assert_eq!(evidence.rows(), 2);
    assert_eq!(evidence.columns(), 2 * GROUP);
    assert_eq!(evidence.input_groups().len(), 2);
    assert_eq!(evidence.input_groups()[0].as_slice()[0], 5.0);
    assert_eq!(evidence.input_groups()[0].as_slice()[1], 5.0);
    assert_eq!(evidence.input_groups()[1].as_slice()[0], 10.0);
    assert_eq!(evidence.input_groups()[1].as_slice()[1], 10.0);
    assert_eq!(evidence.output_weights(), &[5.0, 10.0]);
    assert_eq!(evidence.damping(), 0.25);

    let bytes = evidence.canonical_bytes().unwrap();
    assert_eq!(
        SaltV2KroneckerEvidence::from_canonical_bytes(&bytes)
            .unwrap()
            .canonical_bytes()
            .unwrap(),
        bytes
    );
}

#[test]
fn invalid_batch_is_atomic_at_public_builder_seam() {
    let (activations, gradients) = samples();
    let mut builder =
        SaltV2KroneckerEvidenceBuilder::new(spec(SaltV2Curvature::GuidedFisher)).unwrap();
    builder
        .accumulate_batch(
            &activations[..2 * GROUP],
            Some(&gradients[..2]),
            1,
            None,
            None,
        )
        .unwrap();
    let before = builder.finish().unwrap().canonical_bytes().unwrap();
    let mut invalid = gradients[2..].to_vec();
    invalid[1] = f32::NAN;
    assert!(
        builder
            .accumulate_batch(&activations[2 * GROUP..], Some(&invalid), 1, None, None,)
            .is_err()
    );
    assert_eq!(builder.finish().unwrap().canonical_bytes().unwrap(), before);
}

#[test]
fn late_non_finite_activation_is_rejected_before_atomic_mutation() {
    let (activations, gradients) = samples();
    let mut builder =
        SaltV2KroneckerEvidenceBuilder::new(spec(SaltV2Curvature::GuidedFisher)).unwrap();
    builder
        .accumulate_batch(
            &activations[..2 * GROUP],
            Some(&gradients[..2]),
            1,
            None,
            None,
        )
        .unwrap();
    let before = builder.finish().unwrap().canonical_bytes().unwrap();
    let mut invalid = activations[2 * GROUP..].to_vec();
    invalid[GROUP + 17] = f32::NAN;

    assert!(
        builder
            .accumulate_batch(&invalid, Some(&gradients[2..]), 1, None, None)
            .is_err()
    );
    assert_eq!(builder.finish().unwrap().canonical_bytes().unwrap(), before);
}

#[test]
fn forward_kl_honors_weights_and_masks() {
    let (activations, gradients) = samples();
    let mut builder =
        SaltV2KroneckerEvidenceBuilder::new(spec(SaltV2Curvature::ForwardKlKronecker)).unwrap();
    builder
        .accumulate_batch(
            &activations,
            Some(&gradients),
            2,
            Some(&[2.0, 3.0]),
            Some(&[true, false]),
        )
        .unwrap();
    let evidence = builder.finish().unwrap();

    assert_eq!(evidence.kind(), SaltV2Curvature::ForwardKlKronecker);
    assert_eq!(evidence.input_groups()[0].as_slice()[0], 1.0);
    assert_eq!(evidence.input_groups()[1].as_slice()[0], 4.0);
    assert_eq!(evidence.output_weights(), &[1.0, 4.0]);
}

#[test]
fn input_hessian_requires_no_output_gradients_and_uses_unit_rows() {
    let (activations, _) = samples();
    let mut builder =
        SaltV2KroneckerEvidenceBuilder::new(spec(SaltV2Curvature::InputHessian)).unwrap();
    builder
        .accumulate_batch(&activations, None, 2, None, None)
        .unwrap();
    let evidence = builder.finish().unwrap();
    assert_eq!(evidence.output_weights(), &[1.0, 1.0]);

    assert!(
        builder
            .accumulate_batch(&activations, Some(&[1.0; 4]), 2, None, None)
            .is_err()
    );
}

#[test]
fn out_of_order_shards_merge_to_one_shot_canonical_evidence() {
    let contract = spec(SaltV2Curvature::GuidedFisher);
    let (activations, gradients) = samples();
    let mut one_shot = SaltV2KroneckerEvidenceBuilder::new(contract.clone()).unwrap();
    one_shot
        .accumulate_batch(&activations, Some(&gradients), 2, None, None)
        .unwrap();
    one_shot
        .accumulate_batch(&activations, Some(&gradients), 2, None, None)
        .unwrap();

    let mut first = SaltV2KroneckerEvidenceBuilder::new_at(contract.clone(), 0).unwrap();
    first
        .accumulate_batch(&activations, Some(&gradients), 2, None, None)
        .unwrap();
    let mut second = SaltV2KroneckerEvidenceBuilder::new_at(contract.clone(), 2).unwrap();
    second
        .accumulate_batch(&activations, Some(&gradients), 2, None, None)
        .unwrap();
    let mut merged = SaltV2KroneckerEvidenceBuilder::new(contract).unwrap();
    merged.merge_shard(2, &second).unwrap();
    merged.merge_shard(0, &first).unwrap();

    assert_eq!(
        merged.finish().unwrap().canonical_bytes().unwrap(),
        one_shot.finish().unwrap().canonical_bytes().unwrap()
    );
}

#[test]
fn failed_shard_merges_are_atomic_and_gaps_are_not_finalizable() {
    let contract = spec(SaltV2Curvature::GuidedFisher);
    let activation = vec![1.0; 2 * GROUP];
    let gradients = [1.0, 2.0];
    let mut destination = SaltV2KroneckerEvidenceBuilder::new(contract.clone()).unwrap();
    destination
        .accumulate_batch(&activation, Some(&gradients), 1, None, None)
        .unwrap();
    let before = destination.finish().unwrap().canonical_bytes().unwrap();

    let drifted = SaltV2KroneckerEvidenceSpec::new(
        SaltV2Curvature::GuidedFisher,
        source(),
        7,
        "other.weight",
        2,
        2 * GROUP,
        0.25,
    )
    .unwrap();
    let mismatched = SaltV2KroneckerEvidenceBuilder::new_at(drifted, 1).unwrap();
    assert_eq!(
        destination.merge_shard(1, &mismatched),
        Err(SaltV2KroneckerEvidenceBuildError::SpecMismatch)
    );

    let mut overlapping = SaltV2KroneckerEvidenceBuilder::new_at(contract.clone(), 0).unwrap();
    overlapping
        .accumulate_batch(&activation, Some(&gradients), 1, None, None)
        .unwrap();
    assert!(destination.merge_shard(0, &overlapping).is_err());
    assert_eq!(
        destination.finish().unwrap().canonical_bytes().unwrap(),
        before
    );

    let mut first = SaltV2KroneckerEvidenceBuilder::new_at(contract.clone(), 0).unwrap();
    first
        .accumulate_batch(&activation, Some(&gradients), 1, None, None)
        .unwrap();
    let mut second = SaltV2KroneckerEvidenceBuilder::new_at(contract.clone(), 1).unwrap();
    second
        .accumulate_batch(&activation, Some(&gradients), 1, None, None)
        .unwrap();
    let mut gapped = SaltV2KroneckerEvidenceBuilder::new(contract).unwrap();
    gapped.merge_shard(1, &second).unwrap();
    assert!(gapped.finish().is_err());
    gapped.merge_shard(0, &first).unwrap();
    assert!(gapped.finish().is_ok());
}

#[test]
fn reduction_state_is_logarithmic_and_hard_capped() {
    let contract = spec(SaltV2Curvature::GuidedFisher);
    let activation = vec![1.0; 2 * GROUP];
    let gradients = [1.0, 1.0];
    let mut local = SaltV2KroneckerEvidenceBuilder::new(contract.clone()).unwrap();
    for _ in 0..32 {
        local
            .accumulate_batch(&activation, Some(&gradients), 1, None, None)
            .unwrap();
    }
    assert_eq!(local.residency().input_segments(), 2);
    assert_eq!(local.residency().output_segments(), 1);

    let mut merged = SaltV2KroneckerEvidenceBuilder::new(contract.clone()).unwrap();
    for ordinal in 0..MAX_KRONECKER_REDUCTION_SEGMENTS {
        let mut shard =
            SaltV2KroneckerEvidenceBuilder::new_at(contract.clone(), ordinal as u64).unwrap();
        shard
            .accumulate_batch(&activation, Some(&gradients), 1, None, None)
            .unwrap();
        merged.merge_shard(ordinal as u64, &shard).unwrap();
    }
    let before = merged.residency();
    let mut overflow =
        SaltV2KroneckerEvidenceBuilder::new_at(contract, MAX_KRONECKER_REDUCTION_SEGMENTS as u64)
            .unwrap();
    overflow
        .accumulate_batch(&activation, Some(&gradients), 1, None, None)
        .unwrap();
    assert!(matches!(
        merged.merge_shard(MAX_KRONECKER_REDUCTION_SEGMENTS as u64, &overflow),
        Err(SaltV2KroneckerEvidenceBuildError::ReductionSegmentLimitExceeded { .. })
    ));
    assert_eq!(merged.residency(), before);
    assert!(merged.finish().is_ok());
}
