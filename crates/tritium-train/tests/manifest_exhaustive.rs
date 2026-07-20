//! Exhaustiveness gate between public training surfaces and plan-0049 manifest IDs.

use std::collections::BTreeSet;

use tritium_spec::TrainingOpManifestV1;

const TAPE_METHODS: &[(&str, &str)] = &[
    ("ste_surrogate", "graph.ste_surrogate"),
    ("salt_ste", "graph.salt_ste"),
    ("lsq_ste", "graph.lsq_ste"),
    ("dense_matmul", "graph.dense_matmul"),
    ("transpose", "graph.transpose"),
    ("embed_gather", "graph.embedding_gather"),
    ("slice_cols", "graph.slice_cols"),
    ("concat_cols", "graph.concat_cols"),
    ("detach", "graph.detach"),
    ("scale_const", "graph.scale_const"),
    ("matmul", "graph.ternary_matmul"),
    ("conv1d", "graph.conv1d"),
    ("conv2d", "graph.conv2d"),
    ("fsq", "graph.fsq"),
    ("bias", "graph.bias"),
    ("relu2", "graph.relu2"),
    ("silu", "graph.silu"),
    ("add", "graph.add"),
    ("mul", "graph.mul"),
    ("rmsnorm", "graph.rmsnorm"),
    ("softmax", "graph.softmax"),
    ("causal_mask", "graph.causal_mask"),
    ("rope", "graph.rope"),
    ("mse", "loss.mse"),
    ("softmax_xent", "loss.softmax_cross_entropy"),
];

const OPTIMIZERS: &[(&str, &str)] = &[
    ("Sgd", "optimizer.sgd"),
    ("AdamW", "optimizer.adamw"),
    ("CautiousAdamW", "optimizer.cautious_adamw"),
    ("Int8AdamW", "optimizer.int8_adamw"),
    ("Muon", "optimizer.muon"),
];

fn public_functions(source: &str) -> BTreeSet<&str> {
    source
        .lines()
        .filter_map(|line| {
            line.trim_start()
                .strip_prefix("pub fn ")?
                .split_once('(')
                .map(|(name, _)| name)
        })
        .collect()
}

fn optimizer_impls(source: &str) -> BTreeSet<&str> {
    source
        .lines()
        .filter_map(|line| {
            line.trim_start()
                .strip_prefix("impl Optimizer for ")?
                .strip_suffix(" {")
        })
        .collect()
}

#[test]
fn every_public_tape_operation_has_one_frozen_manifest_id() {
    let actual = public_functions(include_str!("../src/tape.rs"));
    let infrastructure = [
        "new",
        "with_gemm",
        "leaf",
        "value",
        "backward",
        "try_conv2d",
    ];
    let semantic: BTreeSet<_> = actual
        .into_iter()
        .filter(|name| !infrastructure.contains(name))
        .collect();
    let mapped: BTreeSet<_> = TAPE_METHODS.iter().map(|(method, _)| *method).collect();
    assert_eq!(
        semantic, mapped,
        "public Tape surface changed; amend manifest mapping"
    );

    let manifest: BTreeSet<_> = TrainingOpManifestV1::operations()
        .iter()
        .map(|operation| operation.id)
        .collect();
    for (_, operation) in TAPE_METHODS {
        assert!(
            manifest.contains(operation),
            "missing manifest operation {operation}"
        );
    }
    assert!(manifest.contains("graph.attention"));
}

#[test]
fn every_optimizer_implementation_has_one_frozen_manifest_id() {
    let actual = optimizer_impls(include_str!("../src/optim.rs"));
    let mapped: BTreeSet<_> = OPTIMIZERS.iter().map(|(name, _)| *name).collect();
    assert_eq!(
        actual, mapped,
        "Optimizer implementation set changed; amend manifest mapping"
    );

    let manifest: BTreeSet<_> = TrainingOpManifestV1::operations()
        .iter()
        .map(|operation| operation.id)
        .collect();
    for (_, operation) in OPTIMIZERS {
        assert!(
            manifest.contains(operation),
            "missing manifest operation {operation}"
        );
    }
}

#[test]
fn lifecycle_registry_is_complete() {
    let manifest: BTreeSet<_> = TrainingOpManifestV1::operations()
        .iter()
        .map(|operation| operation.id)
        .collect();
    for operation in [
        "lifecycle.checkpoint",
        "lifecycle.resume",
        "lifecycle.export",
        "lifecycle.reload",
    ] {
        assert!(
            manifest.contains(operation),
            "missing manifest operation {operation}"
        );
    }
}
