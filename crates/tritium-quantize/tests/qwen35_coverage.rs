use tritium_quantize::{
    QWEN36_27B_COVERAGE_REVISION, Qwen35CoverageDisposition, Qwen35CoverageError,
    Qwen35CoverageManifest, Qwen35LanguageLayerKind, Qwen35TensorMetadata, Qwen35TensorRole,
    Qwen35TensorScope,
};

const PINNED_METADATA: &str = include_str!("fixtures/qwen36-27b-metadata.tsv");

#[derive(Clone, Debug)]
struct OwnedMetadata {
    name: String,
    dtype: String,
    shape: Vec<u64>,
}

fn fixture_metadata() -> Vec<OwnedMetadata> {
    PINNED_METADATA
        .lines()
        .map(|line| {
            let mut fields = line.split('\t');
            let name = fields.next().expect("fixture name").to_owned();
            let dtype = fields.next().expect("fixture dtype").to_owned();
            let shape = fields
                .next()
                .expect("fixture shape")
                .split(',')
                .map(|dimension| dimension.parse().expect("u64 fixture dimension"))
                .collect();
            assert!(fields.next().is_none(), "three canonical fixture fields");
            OwnedMetadata { name, dtype, shape }
        })
        .collect()
}

fn build_manifest(
    metadata: &[OwnedMetadata],
) -> Result<Qwen35CoverageManifest, tritium_quantize::Qwen35CoverageError> {
    Qwen35CoverageManifest::from_metadata(
        QWEN36_27B_COVERAGE_REVISION,
        metadata
            .iter()
            .map(|tensor| Qwen35TensorMetadata::new(&tensor.name, &tensor.dtype, &tensor.shape)),
    )
}

fn tensor_mut<'a>(metadata: &'a mut [OwnedMetadata], name: &str) -> &'a mut OwnedMetadata {
    metadata
        .iter_mut()
        .find(|tensor| tensor.name == name)
        .expect("fixture tensor")
}

#[test]
fn pinned_official_metadata_has_frozen_coverage_and_policy() {
    let metadata = fixture_metadata();
    let manifest = build_manifest(&metadata).expect("pinned metadata must be admitted");
    let summary = manifest.summary();

    assert_eq!(PINNED_METADATA.len(), 75_705);
    assert_eq!(
        manifest.metadata_digest(),
        &[
            0xad, 0xd3, 0x32, 0xd2, 0x3a, 0x10, 0x12, 0xaa, 0x1d, 0x77, 0x33, 0x9e, 0x04, 0x61,
            0x30, 0x42, 0xaa, 0x56, 0xfb, 0x7c, 0xab, 0x49, 0x08, 0xf7, 0x72, 0x6a, 0xc7, 0x8b,
            0x78, 0xef, 0x2d, 0x8e,
        ]
    );
    assert_eq!(manifest.metadata_record_bytes(), 75_705);
    assert_eq!(manifest.expected_source_payload_bytes(), 55_562_855_904);
    assert_eq!(summary.total().tensors(), 1_199);
    assert_eq!(summary.total().coefficients(), 27_781_427_952);
    assert_eq!(summary.language().tensors(), 851);
    assert_eq!(summary.language().coefficients(), 26_895_998_464);
    assert_eq!(summary.mtp().tensors(), 15);
    assert_eq!(summary.mtp().coefficients(), 424_699_392);
    assert_eq!(summary.vision().tensors(), 333);
    assert_eq!(summary.vision().coefficients(), 460_730_096);
    assert_eq!(summary.included().tensors(), 866);
    assert_eq!(summary.included().coefficients(), 27_320_697_856);
    assert_eq!(summary.additive_ternary().tensors(), 506);
    assert_eq!(summary.additive_ternary().coefficients(), 27_318_026_240);
    assert_eq!(summary.preserve_source().tensors(), 360);
    assert_eq!(summary.preserve_source().coefficients(), 2_671_616);
    assert_eq!(summary.excluded_future_vision().tensors(), 333);
    assert_eq!(summary.excluded_future_vision().coefficients(), 460_730_096);

    assert!(
        manifest
            .entries()
            .windows(2)
            .all(|pair| pair[0].name() < pair[1].name())
    );
    assert_eq!(
        manifest
            .entries()
            .iter()
            .filter(|entry| entry.scope() == Qwen35TensorScope::MtpDrafter)
            .count(),
        15
    );
    assert!(
        manifest
            .entries()
            .iter()
            .filter(|entry| entry.scope() == Qwen35TensorScope::DeferredVision)
            .all(|entry| {
                entry.disposition() == Qwen35CoverageDisposition::ExcludedFutureVision
            })
    );
}

#[test]
fn input_order_does_not_change_the_canonical_manifest() {
    let metadata = fixture_metadata();
    let forward = build_manifest(&metadata).expect("forward manifest");
    let reversed: Vec<_> = metadata.iter().cloned().rev().collect();
    let reverse = build_manifest(&reversed).expect("reverse manifest");

    assert_eq!(forward, reverse);
}

#[test]
fn canonical_policy_round_trip_binds_exact_tensor_actions() {
    let metadata = fixture_metadata();
    let manifest = build_manifest(&metadata).expect("pinned metadata");
    let bytes = manifest.canonical_policy_bytes();
    let decoded = Qwen35CoverageManifest::from_canonical_policy_bytes(&bytes)
        .expect("canonical policy round trip");

    assert_eq!(decoded, manifest);
    assert_eq!(decoded.canonical_policy_bytes(), bytes);
    assert_eq!(decoded.policy_digest(), manifest.policy_digest());

    let mut bad_version = bytes.clone();
    bad_version[8] = bad_version[8].wrapping_add(1);
    assert!(matches!(
        Qwen35CoverageManifest::from_canonical_policy_bytes(&bad_version),
        Err(Qwen35CoverageError::UnsupportedCanonicalPolicyVersion(_))
    ));

    assert!(matches!(
        Qwen35CoverageManifest::from_canonical_policy_bytes(&bytes[..bytes.len() - 1]),
        Err(Qwen35CoverageError::MalformedCanonicalPolicy(_))
    ));

    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(matches!(
        Qwen35CoverageManifest::from_canonical_policy_bytes(&trailing),
        Err(Qwen35CoverageError::NonCanonicalPolicy)
    ));

    let mut changed_action = bytes;
    *changed_action.last_mut().expect("last disposition tag") ^= 1;
    assert!(Qwen35CoverageManifest::from_canonical_policy_bytes(&changed_action).is_err());
}

#[test]
fn scopes_roles_and_rank_policy_are_explicit() {
    let metadata = fixture_metadata();
    let manifest = build_manifest(&metadata).expect("pinned metadata");

    for entry in manifest.entries() {
        let expected_scope = if entry.name().starts_with("model.visual.") {
            Qwen35TensorScope::DeferredVision
        } else if entry.name().starts_with("mtp.") {
            Qwen35TensorScope::MtpDrafter
        } else {
            Qwen35TensorScope::Language
        };
        let expected_disposition = if expected_scope == Qwen35TensorScope::DeferredVision {
            Qwen35CoverageDisposition::ExcludedFutureVision
        } else if entry.shape().len() == 2 {
            Qwen35CoverageDisposition::AdditiveTernary
        } else {
            Qwen35CoverageDisposition::PreserveSource
        };
        assert_eq!(entry.scope(), expected_scope, "scope for {}", entry.name());
        assert_eq!(
            entry.disposition(),
            expected_disposition,
            "disposition for {}",
            entry.name()
        );
    }

    let expected_roles = [
        (
            "model.language_model.embed_tokens.weight",
            Qwen35TensorRole::TokenEmbedding,
        ),
        ("lm_head.weight", Qwen35TensorRole::OutputHead),
        (
            "model.language_model.norm.weight",
            Qwen35TensorRole::Normalization,
        ),
        (
            "model.language_model.layers.0.mlp.down_proj.weight",
            Qwen35TensorRole::MlpProjection,
        ),
        (
            "model.language_model.layers.3.self_attn.q_proj.weight",
            Qwen35TensorRole::FullAttentionProjection,
        ),
        (
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            Qwen35TensorRole::DeltaNetProjection,
        ),
        (
            "model.language_model.layers.0.linear_attn.A_log",
            Qwen35TensorRole::DeltaNetState,
        ),
        (
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            Qwen35TensorRole::DeltaNetConvolution,
        ),
        ("mtp.fc.weight", Qwen35TensorRole::MtpFusionProjection),
        (
            "model.visual.blocks.0.attn.qkv.weight",
            Qwen35TensorRole::VisionAttentionProjection,
        ),
        (
            "model.visual.blocks.0.mlp.linear_fc1.weight",
            Qwen35TensorRole::VisionMlpProjection,
        ),
        (
            "model.visual.patch_embed.proj.weight",
            Qwen35TensorRole::VisionPatchEmbedding,
        ),
        (
            "model.visual.pos_embed.weight",
            Qwen35TensorRole::VisionPositionalEmbedding,
        ),
        (
            "model.visual.merger.linear_fc2.weight",
            Qwen35TensorRole::VisionMergerProjection,
        ),
        (
            "model.visual.blocks.0.attn.qkv.bias",
            Qwen35TensorRole::Bias,
        ),
    ];
    for (name, expected_role) in expected_roles {
        let entry = manifest
            .entries()
            .iter()
            .find(|entry| entry.name() == name)
            .expect("representative role tensor");
        assert_eq!(entry.role(), expected_role, "role for {name}");
    }
}

#[test]
fn revision_must_be_the_campaign_pin() {
    let metadata = fixture_metadata();
    let result = Qwen35CoverageManifest::from_metadata(
        "main",
        metadata
            .iter()
            .map(|tensor| Qwen35TensorMetadata::new(&tensor.name, &tensor.dtype, &tensor.shape)),
    );

    assert!(matches!(result, Err(Qwen35CoverageError::WrongRevision)));
}

#[test]
fn source_dtype_must_be_exact_bf16() {
    let mut metadata = fixture_metadata();
    tensor_mut(&mut metadata, "lm_head.weight").dtype = "F16".to_owned();

    assert!(matches!(
        build_manifest(&metadata),
        Err(Qwen35CoverageError::UnsupportedDtype(name)) if name == "lm_head.weight"
    ));
}

#[test]
fn unknown_tensor_name_is_rejected() {
    let mut metadata = fixture_metadata();
    tensor_mut(&mut metadata, "lm_head.weight").name = "lm_head.mystery".to_owned();

    assert!(matches!(
        build_manifest(&metadata),
        Err(Qwen35CoverageError::UnknownTensor(name)) if name == "lm_head.mystery"
    ));
}

#[test]
fn missing_tensor_metadata_is_rejected() {
    let mut metadata = fixture_metadata();
    metadata.pop();

    assert!(matches!(
        build_manifest(&metadata),
        Err(Qwen35CoverageError::MissingTensorMetadata {
            expected: 1_199,
            actual: 1_198,
        })
    ));
}

#[test]
fn duplicate_tensor_name_is_rejected() {
    let mut metadata = fixture_metadata();
    let duplicate = metadata[0].clone();
    *tensor_mut(&mut metadata, "model.language_model.embed_tokens.weight") = duplicate;

    assert!(matches!(
        build_manifest(&metadata),
        Err(Qwen35CoverageError::DuplicateTensor(name)) if name == "lm_head.weight"
    ));
}

#[test]
fn attention_tensor_must_match_the_layer_schedule() {
    let mut metadata = fixture_metadata();
    tensor_mut(
        &mut metadata,
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
    )
    .name = "model.language_model.layers.0.self_attn.q_proj.weight".to_owned();

    assert!(matches!(
        build_manifest(&metadata),
        Err(Qwen35CoverageError::WrongLayerKind {
            layer: 0,
            expected: Qwen35LanguageLayerKind::DeltaNet,
            ..
        })
    ));
}

#[test]
fn tensor_rank_must_match_the_pin() {
    let mut metadata = fixture_metadata();
    tensor_mut(&mut metadata, "lm_head.weight").shape.pop();

    assert!(matches!(
        build_manifest(&metadata),
        Err(Qwen35CoverageError::WrongRank {
            expected: 2,
            actual: 1,
            ..
        })
    ));
}

#[test]
fn zero_dimension_is_rejected_before_shape_identity() {
    let mut metadata = fixture_metadata();
    tensor_mut(&mut metadata, "lm_head.weight").shape[0] = 0;

    assert!(matches!(
        build_manifest(&metadata),
        Err(Qwen35CoverageError::ZeroDimension { dimension: 0, .. })
    ));
}

#[test]
fn coefficient_overflow_is_rejected_before_shape_identity() {
    let mut metadata = fixture_metadata();
    tensor_mut(&mut metadata, "lm_head.weight").shape = vec![u64::MAX, u64::MAX];

    assert!(matches!(
        build_manifest(&metadata),
        Err(Qwen35CoverageError::CoefficientOverflow(name)) if name == "lm_head.weight"
    ));
}

#[test]
fn same_rank_wrong_shape_is_rejected() {
    let mut metadata = fixture_metadata();
    tensor_mut(&mut metadata, "lm_head.weight").shape[0] -= 1;

    assert!(matches!(
        build_manifest(&metadata),
        Err(Qwen35CoverageError::WrongShape(name)) if name == "lm_head.weight"
    ));
}

#[test]
fn canonical_metadata_identity_rejects_noncanonical_layer_spelling() {
    let mut metadata = fixture_metadata();
    tensor_mut(
        &mut metadata,
        "model.language_model.layers.10.input_layernorm.weight",
    )
    .name = "model.language_model.layers.09.input_layernorm.weight".to_owned();

    let error = build_manifest(&metadata).expect_err("same-length identity change must fail");
    assert!(matches!(
        &error,
        Qwen35CoverageError::MetadataIdentityMismatch {
            expected_bytes: 75_705,
            actual_bytes: 75_705,
            ..
        }
    ));
    let diagnostic = error.to_string();
    assert!(diagnostic.contains("digest"));
    assert!(!diagnostic.contains("got 75706"));
}

#[test]
fn metadata_count_is_bounded_before_an_extra_entry_is_copied() {
    let mut metadata = fixture_metadata();
    metadata.push(metadata[0].clone());

    assert!(matches!(
        build_manifest(&metadata),
        Err(Qwen35CoverageError::TooManyTensors { maximum: 1_199 })
    ));
}

#[test]
fn caller_controlled_name_and_dtype_lengths_are_bounded() {
    let mut metadata = fixture_metadata();
    tensor_mut(&mut metadata, "lm_head.weight").name = "x".repeat(129);
    assert!(matches!(
        build_manifest(&metadata),
        Err(Qwen35CoverageError::MetadataFieldTooLong {
            field: "tensor name",
            maximum: 128,
        })
    ));

    let mut metadata = fixture_metadata();
    tensor_mut(&mut metadata, "lm_head.weight").dtype = "x".repeat(17);
    assert!(matches!(
        build_manifest(&metadata),
        Err(Qwen35CoverageError::MetadataFieldTooLong {
            field: "tensor dtype",
            maximum: 16,
        })
    ));
}
