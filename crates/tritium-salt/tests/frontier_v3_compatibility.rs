use half::f16;
use tritium_format::salt_v2::SaltV2Codec;
use tritium_format::salt_v2_package::{
    SALT_V2_PACKAGE_VERSION_SCALE_GEOMETRY, SaltV2IndexedRuntimeLedger, SaltV2Package, SaltV2Plane,
    SaltV2Tensor, SaltV2Tile, SaltV2Transform, write_salt_v2_package,
};
use tritium_salt::{
    ArtifactClaim, ContentId, FRONTIER_SOLVER_ABI_V1, FrontierCompatibilityError,
    FrontierProfileId, SolverDescriptor, SolverFamily, SolverId, SolverTrust, TensorRepresentation,
    read_salt_v2_frontier_artifact,
};

fn digest(label: &str) -> ContentId {
    ContentId::of_bytes(label.as_bytes())
}

fn tensor(name: &str, trits: [i8; 4]) -> SaltV2Tensor {
    let plane = SaltV2Plane::new(trits.to_vec(), vec![f16::from_f32(0.5)]).unwrap();
    SaltV2Tensor::new(
        name,
        vec![2, 2],
        vec![SaltV2Tile::new(vec![plane]).unwrap()],
    )
    .unwrap()
}

fn salt_descriptor() -> SolverDescriptor {
    SolverDescriptor::new(
        SolverId::new("salt.v2.compat").unwrap(),
        SolverFamily::Salt,
        SolverTrust::Registered,
        FRONTIER_SOLVER_ABI_V1,
    )
    .unwrap()
}

#[test]
fn canonical_salt_v2_bytes_adapt_to_exact_v3_manifest() {
    let package = SaltV2Package::new(
        SaltV2Codec::D2,
        vec![
            tensor("z.weight", [1, 0, -1, 1]),
            tensor("a.weight", [0, 1, 0, -1]),
        ],
    )
    .unwrap();
    let encoded = write_salt_v2_package(&package).unwrap();
    let runtime = SaltV2IndexedRuntimeLedger::for_package(&package).unwrap();
    let manifest = read_salt_v2_frontier_artifact(
        &encoded.bytes,
        digest("source"),
        FrontierProfileId::new("compat.v2").unwrap(),
        salt_descriptor(),
        digest("v2 adapter recipe"),
        19,
    )
    .unwrap();

    assert_eq!(manifest.claim(), ArtifactClaim::PureTernary);
    assert_eq!(
        manifest
            .tensors()
            .iter()
            .map(|tensor| tensor.name())
            .collect::<Vec<_>>(),
        ["a.weight", "z.weight"]
    );
    assert!(manifest.tensors().iter().all(|tensor| {
        tensor.representation() == TensorRepresentation::AdditiveTernaryPlanes
            && tensor.solver().family() == SolverFamily::Salt
    }));
    assert_ne!(
        manifest.tensors()[0].payload_id(),
        manifest.tensors()[1].payload_id()
    );
    assert_eq!(
        manifest.ledger().total_serialized_bytes(),
        encoded.ledger.total_bytes
    );
    assert_eq!(
        manifest.ledger().serialized().tensor_payload_bytes(),
        encoded.ledger.payload_bytes
    );
    assert_eq!(
        manifest.ledger().total_resident_bytes(),
        runtime.steady_resident_bytes()
    );
    assert_eq!(manifest.ledger().transient_bytes(), 19);

    let json = serde_json::to_string_pretty(&manifest).unwrap() + "\n";
    assert_eq!(
        json,
        include_str!("fixtures/frontier-salt-v2-adapter-v1.json")
    );
}

#[test]
fn salt_v2_adapter_refuses_corruption_wrong_family_and_zero_recipe() {
    let package =
        SaltV2Package::new(SaltV2Codec::B3, vec![tensor("weight", [1, 0, -1, 1])]).unwrap();
    let mut bytes = write_salt_v2_package(&package).unwrap().bytes;
    bytes[0] ^= 0xff;
    assert!(matches!(
        read_salt_v2_frontier_artifact(
            &bytes,
            digest("source"),
            FrontierProfileId::new("compat.v2").unwrap(),
            salt_descriptor(),
            digest("recipe"),
            0,
        ),
        Err(FrontierCompatibilityError::SaltV2(_))
    ));

    let bytes = write_salt_v2_package(&package).unwrap().bytes;
    let wrong_family = SolverDescriptor::new(
        SolverId::new("externd.v1").unwrap(),
        SolverFamily::ExTernD,
        SolverTrust::Registered,
        FRONTIER_SOLVER_ABI_V1,
    )
    .unwrap();
    assert!(matches!(
        read_salt_v2_frontier_artifact(
            &bytes,
            digest("source"),
            FrontierProfileId::new("compat.v2").unwrap(),
            wrong_family,
            digest("recipe"),
            0,
        ),
        Err(FrontierCompatibilityError::Artifact(_))
    ));
    assert!(matches!(
        read_salt_v2_frontier_artifact(
            &bytes,
            digest("source"),
            FrontierProfileId::new("compat.v2").unwrap(),
            salt_descriptor(),
            ContentId::from_digest([0; 32]),
            0,
        ),
        Err(FrontierCompatibilityError::Artifact(_))
    ));
}

#[test]
fn scale_geometry_v2_package_also_adapts_without_rewriting_source_bytes() {
    let plane =
        SaltV2Plane::new_with_scale_group_size(vec![1, 0, -1, 1], vec![f16::from_f32(0.5)], 64)
            .unwrap();
    let tensor = SaltV2Tensor::new_with_layout(
        "weight",
        vec![2, 2],
        SaltV2Transform::None,
        64,
        vec![SaltV2Tile::new(vec![plane]).unwrap()],
    )
    .unwrap();
    let package = SaltV2Package::new(SaltV2Codec::D2, vec![tensor]).unwrap();
    let encoded = write_salt_v2_package(&package).unwrap();
    assert_eq!(
        u16::from_le_bytes(encoded.bytes[8..10].try_into().unwrap()),
        SALT_V2_PACKAGE_VERSION_SCALE_GEOMETRY
    );

    let manifest = read_salt_v2_frontier_artifact(
        &encoded.bytes,
        digest("source"),
        FrontierProfileId::new("compat.v2").unwrap(),
        salt_descriptor(),
        digest("recipe"),
        0,
    )
    .unwrap();
    assert_eq!(
        manifest.ledger().total_serialized_bytes(),
        encoded.bytes.len() as u64
    );
    assert_eq!(manifest.tensors()[0].shape(), [2, 2]);
}
