use tritium_salt::{
    ArtifactByteLedger, ArtifactClaim, ByteBreakdown, ContentId, FRONTIER_ARTIFACT_SCHEMA_V1,
    FRONTIER_SOLVER_ABI_V1, FrontierArtifactError, FrontierArtifactManifest, FrontierProfileId,
    FrontierTensorArtifact, SolverDescriptor, SolverFamily, SolverId, SolverTrust,
    TensorRepresentation,
};

fn digest(label: &str) -> ContentId {
    ContentId::of_bytes(label.as_bytes())
}

fn descriptor(id: &str, family: SolverFamily) -> SolverDescriptor {
    SolverDescriptor::new(
        SolverId::new(id).unwrap(),
        family,
        SolverTrust::Registered,
        FRONTIER_SOLVER_ABI_V1,
    )
    .unwrap()
}

#[test]
fn heterogeneous_manifest_round_trips_with_exact_physical_ledger() {
    let tensors = vec![
        FrontierTensorArtifact::new(
            "model.layers.0.mlp.down_proj.weight",
            vec![4, 8],
            descriptor("externd.v1", SolverFamily::ExTernD),
            TensorRepresentation::ExpandedRankTernary,
            ArtifactClaim::PureTernary,
            digest("down recipe"),
            digest("down payload"),
            17,
            32,
        )
        .unwrap(),
        FrontierTensorArtifact::new(
            "model.layers.0.self_attn.q_proj.weight",
            vec![8, 8],
            descriptor("salt.v3", SolverFamily::Salt),
            TensorRepresentation::AdditiveTernaryPlanes,
            ArtifactClaim::ResidualBearing,
            digest("q recipe"),
            digest("q payload"),
            23,
            64,
        )
        .unwrap(),
    ];
    let ledger = ArtifactByteLedger::new(
        ByteBreakdown::new(40, 9, 7).unwrap(),
        ByteBreakdown::new(96, 11, 13).unwrap(),
        5,
    )
    .unwrap();
    let manifest = FrontierArtifactManifest::new(
        digest("source"),
        FrontierProfileId::new("research.default").unwrap(),
        tensors,
        ledger,
    )
    .unwrap();

    assert_eq!(manifest.claim(), ArtifactClaim::ResidualBearing);
    assert_eq!(manifest.ledger().total_serialized_bytes(), 56);
    assert_eq!(manifest.ledger().total_resident_bytes(), 120);
    assert_eq!(manifest.ledger().peak_working_set_bytes(), 125);
    assert_eq!(manifest.tensors()[0].element_count(), 32);
    assert_eq!(manifest.tensors()[1].element_count(), 64);

    let encoded = serde_json::to_string_pretty(&manifest).unwrap() + "\n";
    assert_eq!(encoded, include_str!("fixtures/frontier-artifact-v1.json"));
    let value = serde_json::to_value(&manifest).unwrap();
    assert_eq!(value["schema"], FRONTIER_ARTIFACT_SCHEMA_V1);
    assert_eq!(value["source_id"].as_str().unwrap().len(), 69);
    assert!(value["source_id"].as_str().unwrap().starts_with("tsc1_"));
    assert_eq!(value["claim"], "residual-bearing");
    assert_eq!(value["ledger"]["total_serialized_bytes"], 56);
    assert_eq!(value["ledger"]["total_resident_bytes"], 120);
    assert_eq!(value["ledger"]["peak_working_set_bytes"], 125);
    assert_eq!(
        serde_json::from_str::<FrontierArtifactManifest>(include_str!(
            "fixtures/frontier-artifact-v1.json"
        ))
        .unwrap(),
        manifest
    );
}

#[test]
fn byte_breakdown_deserialization_rejects_overflow() {
    let overflowing = format!(
        r#"{{"tensor_payload_bytes":{},"metadata_bytes":1,"preserved_bytes":0}}"#,
        u64::MAX
    );
    assert!(serde_json::from_str::<ByteBreakdown>(&overflowing).is_err());
}

#[test]
fn artifact_construction_refuses_incoherent_types_order_and_bytes() {
    assert!(matches!(
        FrontierTensorArtifact::new(
            "tensor",
            vec![2, 2],
            descriptor("salt.v3", SolverFamily::Salt),
            TensorRepresentation::ExpandedRankTernary,
            ArtifactClaim::PureTernary,
            digest("recipe"),
            digest("payload"),
            4,
            4,
        ),
        Err(FrontierArtifactError::RepresentationFamilyMismatch { .. })
    ));

    let tensor = |name: &str, bytes: u64| {
        FrontierTensorArtifact::new(
            name,
            vec![2, 2],
            descriptor("salt.v3", SolverFamily::Salt),
            TensorRepresentation::AdditiveTernaryPlanes,
            ArtifactClaim::PureTernary,
            digest(&format!("{name} recipe")),
            digest(&format!("{name} payload")),
            bytes,
            bytes,
        )
        .unwrap()
    };
    let ledger = ArtifactByteLedger::new(
        ByteBreakdown::new(8, 0, 0).unwrap(),
        ByteBreakdown::new(8, 0, 0).unwrap(),
        0,
    )
    .unwrap();
    assert!(matches!(
        FrontierArtifactManifest::new(
            digest("source"),
            FrontierProfileId::new("test.profile").unwrap(),
            vec![tensor("b", 4), tensor("a", 4)],
            ledger,
        ),
        Err(FrontierArtifactError::NonCanonicalTensorOrder { .. })
    ));
    assert!(matches!(
        FrontierArtifactManifest::new(
            digest("source"),
            FrontierProfileId::new("test.profile").unwrap(),
            vec![tensor("a", 4), tensor("b", 3)],
            ledger,
        ),
        Err(FrontierArtifactError::TensorByteMismatch {
            view: "serialized",
            ..
        })
    ));
    assert!(matches!(
        FrontierArtifactManifest::new(
            ContentId::from_digest([0; 32]),
            FrontierProfileId::new("test.profile").unwrap(),
            vec![tensor("a", 8)],
            ledger,
        ),
        Err(FrontierArtifactError::ZeroDigest { field: "source" })
    ));
}

#[test]
fn artifact_reader_refuses_corrupted_derived_fields_and_unknown_data() {
    let golden = include_str!("fixtures/frontier-artifact-v1.json");
    let value: serde_json::Value = serde_json::from_str(golden).unwrap();

    let mut corrupted = value.clone();
    corrupted["claim"] = "pure-ternary".into();
    assert!(serde_json::from_value::<FrontierArtifactManifest>(corrupted).is_err());

    let mut corrupted = value.clone();
    corrupted["tensors"][0]["element_count"] = 31.into();
    assert!(serde_json::from_value::<FrontierArtifactManifest>(corrupted).is_err());

    let mut corrupted = value.clone();
    corrupted["ledger"]["total_serialized_bytes"] = 55.into();
    assert!(serde_json::from_value::<FrontierArtifactManifest>(corrupted).is_err());

    let mut corrupted = value.clone();
    corrupted["source_id"] =
        "tsc1_Afd7e898ae082430ac69e1ce0bb3e1bf71689ce0659a2053a96e54bc902895b3".into();
    assert!(serde_json::from_value::<FrontierArtifactManifest>(corrupted).is_err());

    let mut corrupted = value;
    corrupted["future_field"] = true.into();
    assert!(serde_json::from_value::<FrontierArtifactManifest>(corrupted).is_err());
}
