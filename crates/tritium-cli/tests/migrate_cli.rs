//! Real binary/original File bundle/importer/publisher/replay, not a CLI mock.
#![cfg(unix)]
use half::f16;
use std::{
    fs,
    io::Cursor,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicU64, Ordering},
};
use tritium_evidence::{LegacyModelImportBoundView, LegacyModelImportView, verify_jsonl};
use tritium_format::{
    AdditiveTensor, PackageHasher, TritPackageReadLimits,
    salt_v2::SaltV2Codec,
    salt_v2_package::{
        SaltV2Package, SaltV2PackageReader, SaltV2Plane, SaltV2Tensor, SaltV2Tile, SaltV2Transform,
        write_salt_v2_package,
    },
    trit_model::{TritModelReader, encode_model_manifest, tensor_storage_cbor_bytes},
};
use tritium_schema::{LEGACY_SALT_V2_HF_ASSETS, LevelFit, PlaneCounts, TensorStorage};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "tritium-migrate-cli-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn hash(bytes: &[u8]) -> String {
    let mut h = PackageHasher::new();
    h.update(bytes);
    h.finalize().to_string()
}
fn run(args: &[&std::ffi::OsStr]) -> Output {
    // A hung child is terminated by the external deadline even if the parent test fails.
    Command::new("timeout")
        .args(["45s", env!("CARGO_BIN_EXE_tritium")])
        .arg("--evidence")
        .arg("off")
        .args(args)
        .output()
        .unwrap()
}
fn invoke(root: &Path, profile: &str, codec: &str, extra: &[&str]) -> Output {
    let bundle = root.join("bundle");
    let output = root.join("output");
    let spool = root.join("spool");
    let mut args = vec![
        "migrate".as_ref(),
        "salt-v2-bundle".as_ref(),
        "--bundle".as_ref(),
        bundle.as_os_str(),
        "--output".as_ref(),
        output.as_os_str(),
        "--spool".as_ref(),
        spool.as_os_str(),
        "--profile".as_ref(),
        profile.as_ref(),
        "--codec".as_ref(),
        codec.as_ref(),
        "--group".as_ref(),
        "128".as_ref(),
        "--import-run-id".as_ref(),
        "actual-bundle-fixture".as_ref(),
    ];
    args.extend(extra.iter().map(std::ffi::OsStr::new));
    run(&args)
}
fn inspect(root: &Path, profile: &str, codec: &str, extra: &[&str]) -> Output {
    let bundle = root.join("bundle");
    let mut args = vec![
        "migrate".as_ref(),
        "salt-v2-bundle".as_ref(),
        "--inspect-only".as_ref(),
        "--bundle".as_ref(),
        bundle.as_os_str(),
        "--profile".as_ref(),
        profile.as_ref(),
        "--codec".as_ref(),
        codec.as_ref(),
        "--group".as_ref(),
        "128".as_ref(),
    ];
    args.extend(extra.iter().map(std::ffi::OsStr::new));
    run(&args)
}

#[test]
fn compact_cli_preserves_exact_package_and_cold_replay_without_embedded_manifest() {
    for source in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
        for profile in ["compact-v1", "near-lossless-v1"] {
            for target in ["d2", "b3"] {
                let root = Scratch::new();
                fixture(&root.0, source);
                let embedded = invoke(&root.0, profile, target, &[]);
                assert!(
                    embedded.status.success(),
                    "{}",
                    String::from_utf8_lossy(&embedded.stderr)
                );
                let bytes = fs::read(root.0.join("output/model.trit")).unwrap();
                fs::rename(root.0.join("output"), root.0.join("embedded")).unwrap();
                let compact = invoke(&root.0, profile, target, &["--compact-evidence"]);
                assert!(
                    compact.status.success(),
                    "{}",
                    String::from_utf8_lossy(&compact.stderr)
                );
                assert_eq!(fs::read(root.0.join("output/model.trit")).unwrap(), bytes);
                let report: serde_json::Value = serde_json::from_slice(&compact.stdout).unwrap();
                assert_eq!(report["evidence_profile"], "canonical-manifest-bound-v1");
                assert_eq!(report["complete_model"], false);
                assert_eq!(report["official_payload_authenticated"], false);
                let log = verify_jsonl(
                    &fs::read_to_string(root.0.join("output/evidence.jsonl")).unwrap(),
                )
                .unwrap();
                assert_eq!(log.events.len(), 1);
                assert!(log.events[0].payload.get("manifest").is_none());
                let model = TritModelReader::open(
                    fs::File::open(root.0.join("output/model.trit")).unwrap(),
                    TritPackageReadLimits {
                        max_source_bytes: 1_000_000,
                        max_manifest_bytes: 65536,
                        max_directory_bytes: 65536,
                    },
                )
                .unwrap();
                assert_eq!(
                    LegacyModelImportBoundView::fold(&log, model.manifest())
                        .unwrap()
                        .len(),
                    1
                );
                assert!(verify(&root.0).status.success());
                assert!(!root.0.join("spool").exists());
                let refused = invoke(&root.0, profile, target, &["--compact-evidence"]);
                assert!(!refused.status.success());
                assert_eq!(fs::read(root.0.join("output/model.trit")).unwrap(), bytes);
                let path = root.0.join("output/model.trit");
                let mut corrupt = bytes;
                let at = corrupt.len() - 1;
                corrupt[at] ^= 1;
                fs::write(&path, corrupt).unwrap();
                assert!(!verify(&root.0).status.success());
            }
        }
    }
}

#[test]
fn readonly_inspection_matches_actual_import_payloads_without_output_or_spool() {
    for source in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
        for profile in ["compact-v1", "near-lossless-v1"] {
            for target in ["d2", "b3"] {
                let root = Scratch::new();
                fixture(&root.0, source);
                let observed = inspect(&root.0, profile, target, &[]);
                assert!(
                    observed.status.success(),
                    "{}",
                    String::from_utf8_lossy(&observed.stderr)
                );
                let report: serde_json::Value = serde_json::from_slice(&observed.stdout).unwrap();
                assert_eq!(
                    report["status"],
                    "strict-indexed-plan-not-imported-not-qualified"
                );
                assert_eq!(report["matrix_tensors"], 1);
                assert_eq!(report["preserved_tensors"], 1);
                assert_eq!(
                    report["qwen_language_mtp_inventory"],
                    "unknown-generic-architecture"
                );
                assert_eq!(report["scale_representability"], "not-checked");
                assert_eq!(report["physical_memory"], "not-covered");
                assert_eq!(report["manifest_size"], "storage-fragment-lower-bound-only");
                assert_eq!(
                    report["container_overhead"],
                    "complete-encoded-byte-bounds-not-exact-output"
                );
                assert_eq!(report["complete_model"], false);
                assert_eq!(report["official_payload_authenticated"], false);
                assert!(!root.0.join("output").exists() && !root.0.join("spool").exists());
                let imported = invoke(&root.0, profile, target, &[]);
                assert!(
                    imported.status.success(),
                    "{}",
                    String::from_utf8_lossy(&imported.stderr)
                );
                let mut model = TritModelReader::open(
                    fs::File::open(root.0.join("output/model.trit")).unwrap(),
                    TritPackageReadLimits {
                        max_source_bytes: 1_000_000,
                        max_manifest_bytes: 65_536,
                        max_directory_bytes: 65_536,
                    },
                )
                .unwrap();
                let bound = model.manifest().levels[0]
                    .tensors
                    .iter()
                    .map(|t| tensor_storage_cbor_bytes(&t.storage).unwrap())
                    .sum::<u64>();
                assert_eq!(report["manifest_lower_bound_bytes"], bound);
                assert!(bound < encode_model_manifest(model.manifest()).unwrap().len() as u64);
                let manifest_bytes = encode_model_manifest(model.manifest()).unwrap().len() as u64;
                let package_bytes = fs::metadata(root.0.join("output/model.trit"))
                    .unwrap()
                    .len();
                for (what, actual) in [("manifest", manifest_bytes), ("package", package_bytes)] {
                    assert!(
                        report[format!("{what}_bytes_lower_bound")]
                            .as_u64()
                            .unwrap()
                            <= actual
                    );
                    assert!(
                        actual
                            <= report[format!("{what}_bytes_upper_bound")]
                                .as_u64()
                                .unwrap()
                    );
                }
                let mut matrix_bytes = 0u64;
                model
                    .visit_tensor("imported", "proj", |c| -> Result<(), ()> {
                        matrix_bytes += c.len() as u64;
                        Ok(())
                    })
                    .unwrap();
                let mut dense_bytes = 0u64;
                model
                    .visit_tensor("imported", "norm", |c| -> Result<(), ()> {
                        dense_bytes += c.len() as u64;
                        Ok(())
                    })
                    .unwrap();
                let mut asset_bytes = 0u64;
                for name in LEGACY_SALT_V2_HF_ASSETS
                    .iter()
                    .map(|(n, _)| *n)
                    .chain(["tritium.json"])
                {
                    model
                        .visit_asset(name, |c| -> Result<(), ()> {
                            asset_bytes += c.len() as u64;
                            Ok(())
                        })
                        .unwrap();
                }
                assert_eq!(report["additive_payload_bytes"], matrix_bytes);
                assert_eq!(report["dense_payload_bytes"], dense_bytes);
                assert_eq!(report["asset_payload_bytes"], asset_bytes);
                assert_eq!(
                    report["spool_payload_bytes"],
                    matrix_bytes + dense_bytes + asset_bytes
                );
            }
        }
    }
}

#[test]
fn inspection_caps_and_write_options_reject_and_import_preflight_preserves_existing_spool() {
    for caps in [
        ["--max-trits", "1"],
        ["--max-working-bytes", "1"],
        ["--max-tensor-bytes", "1"],
        ["--max-spool-bytes", "1"],
    ] {
        let root = Scratch::new();
        fixture(&root.0, SaltV2Codec::B3);
        assert!(!inspect(&root.0, "compact-v1", "b3", &caps).status.success());
        assert!(!root.0.join("output").exists() && !root.0.join("spool").exists());
    }
    let root = Scratch::new();
    fixture(&root.0, SaltV2Codec::B3);
    let conflict = invoke(&root.0, "compact-v1", "b3", &["--inspect-only"]);
    assert!(!conflict.status.success());
    assert!(!root.0.join("output").exists() && !root.0.join("spool").exists());
    fs::write(root.0.join("spool"), b"owner sentinel").unwrap();
    let rejected = invoke(&root.0, "compact-v1", "b3", &["--max-trits", "1"]);
    assert!(!rejected.status.success());
    let error = String::from_utf8_lossy(&rejected.stderr);
    assert!(!error.contains("create new import spool"), "{error}");
    assert_eq!(fs::read(root.0.join("spool")).unwrap(), b"owner sentinel");
    assert!(!root.0.join("output").exists());
}

#[test]
fn manifest_lower_bound_refuses_before_create_and_is_not_full_manifest_admission() {
    let root = Scratch::new();
    fixture(&root.0, SaltV2Codec::B3);
    let observed = inspect(&root.0, "compact-v1", "b3", &[]);
    assert!(observed.status.success());
    let report: serde_json::Value = serde_json::from_slice(&observed.stdout).unwrap();
    let bound = report["manifest_lower_bound_bytes"].as_u64().unwrap();
    assert!(bound > 1);
    let too_small = (bound - 1).to_string();
    fs::write(root.0.join("spool"), b"owner sentinel").unwrap();
    for result in [
        inspect(
            &root.0,
            "compact-v1",
            "b3",
            &["--max-manifest-bytes", &too_small],
        ),
        invoke(
            &root.0,
            "compact-v1",
            "b3",
            &["--max-manifest-bytes", &too_small],
        ),
    ] {
        assert!(!result.status.success());
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(
            error.contains("manifest storage lower bound exceeds receiving cap"),
            "{error}"
        );
        assert!(!error.contains("create new import spool"), "{error}");
        assert_eq!(fs::read(root.0.join("spool")).unwrap(), b"owner sentinel");
        assert!(!root.0.join("output").exists());
    }
    fs::remove_file(root.0.join("spool")).unwrap();
    // Equality fits the known storage fragments, but the complete manifest
    // upper-bound check now refuses before staging the larger container.
    let equal = bound.to_string();
    assert!(
        !inspect(
            &root.0,
            "compact-v1",
            "b3",
            &["--max-manifest-bytes", &equal]
        )
        .status
        .success()
    );
    assert!(
        !invoke(
            &root.0,
            "compact-v1",
            "b3",
            &["--max-manifest-bytes", &equal]
        )
        .status
        .success()
    );
    assert!(!root.0.join("output").exists());
}

#[test]
fn complete_container_caps_refuse_before_spool_creation() {
    let root = Scratch::new();
    fixture(&root.0, SaltV2Codec::B3);
    let observed = inspect(&root.0, "compact-v1", "b3", &[]);
    assert!(observed.status.success());
    let report: serde_json::Value = serde_json::from_slice(&observed.stdout).unwrap();
    fs::write(root.0.join("spool"), b"owner sentinel").unwrap();
    for (flag, key, error_scope) in [
        (
            "--max-manifest-bytes",
            "manifest_bytes_upper_bound",
            "full manifest upper bound",
        ),
        (
            "--max-directory-bytes",
            "directory_bytes_upper_bound",
            "directory upper bound",
        ),
        (
            "--max-package-bytes",
            "package_bytes_upper_bound",
            "full package upper bound",
        ),
    ] {
        let cap = (report[key].as_u64().unwrap() - 1).to_string();
        for result in [
            inspect(&root.0, "compact-v1", "b3", &[flag, &cap]),
            invoke(&root.0, "compact-v1", "b3", &[flag, &cap]),
        ] {
            assert!(!result.status.success());
            let error = String::from_utf8_lossy(&result.stderr);
            assert!(error.contains(error_scope), "{flag}: {error}");
            assert!(!error.contains("create new import spool"));
            assert_eq!(fs::read(root.0.join("spool")).unwrap(), b"owner sentinel");
            assert!(!root.0.join("output").exists());
        }
    }
}

#[test]
fn metadata_inspection_does_not_replace_exact_scale_representability_proof() {
    let root = Scratch::new();
    let (mut descriptor, _) = fixture(&root.0, SaltV2Codec::B3);
    let plane = SaltV2Plane::new_with_scale_group_size(
        (0..256).map(|i| [0, 1, -1, 1][i % 4]).collect(),
        (0..4)
            .map(|i| f16::from_bits(if i % 2 == 0 { 0x3555 } else { 0x3c00 }))
            .collect(),
        64,
    )
    .unwrap();
    let tensor = SaltV2Tensor::new_with_layout(
        "proj",
        vec![2, 128],
        SaltV2Transform::None,
        64,
        vec![SaltV2Tile::new(vec![plane]).unwrap()],
    )
    .unwrap();
    let bytes = write_salt_v2_package(&SaltV2Package::new(SaltV2Codec::B3, vec![tensor]).unwrap())
        .unwrap()
        .bytes;
    let reader = SaltV2PackageReader::new_strict(Cursor::new(&bytes)).unwrap();
    descriptor["profiles"]["compact-v1"] = serde_json::json!({"file":"compact.tsalt2","package_id":hash(&bytes),
        "serialized_bytes":bytes.len(),"resident_bytes":reader.indexed_runtime_ledger().unwrap().steady_resident_bytes()});
    fs::write(root.0.join("bundle/compact.tsalt2"), &bytes).unwrap();
    fs::write(root.0.join("bundle/tritium.json"), descriptor.to_string()).unwrap();
    let inspected = inspect(&root.0, "compact-v1", "b3", &[]);
    assert!(
        inspected.status.success(),
        "{}",
        String::from_utf8_lossy(&inspected.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&inspected.stdout).unwrap();
    assert_eq!(report["scale_representability"], "not-checked");
    let imported = invoke(&root.0, "compact-v1", "b3", &[]);
    assert!(!imported.status.success());
    assert!(!root.0.join("spool").exists() && !root.0.join("output").exists());
    assert_eq!(
        fs::read(root.0.join("bundle/compact.tsalt2")).unwrap(),
        bytes
    );
}
fn profile(codec: SaltV2Codec, planes: usize) -> (Vec<u8>, SaltV2Tensor) {
    let tensor = SaltV2Tensor::new_with_layout(
        "proj",
        vec![2, 128],
        SaltV2Transform::None,
        128,
        vec![
            SaltV2Tile::new(
                (0..planes)
                    .map(|p| {
                        SaltV2Plane::new_with_scale_group_size(
                            (0..256).map(|i| [0, 1, -1, 1][(i + p) % 4]).collect(),
                            vec![f16::from_bits([1, 0x3555][p]); 2],
                            128,
                        )
                        .unwrap()
                    })
                    .collect(),
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let bytes = write_salt_v2_package(&SaltV2Package::new(codec, vec![tensor.clone()]).unwrap())
        .unwrap()
        .bytes;
    (bytes, tensor)
}
fn fixture(root: &Path, codec: SaltV2Codec) -> (serde_json::Value, [SaltV2Tensor; 2]) {
    let dir = root.join("bundle");
    fs::create_dir(&dir).unwrap();
    let mut profiles = serde_json::Map::new();
    let mut tensors = Vec::new();
    for (name, file, count) in [
        ("compact-v1", "compact.tsalt2", 1),
        ("near-lossless-v1", "near-lossless.tsalt2", 2),
    ] {
        let (bytes, tensor) = profile(codec, count);
        tensors.push(tensor);
        let reader = SaltV2PackageReader::new_strict(Cursor::new(&bytes)).unwrap();
        profiles.insert(name.into(), serde_json::json!({"file":file, "package_id":hash(&bytes),
            "serialized_bytes":bytes.len(), "resident_bytes":reader.indexed_runtime_ledger().unwrap().steady_resident_bytes()}));
        fs::write(dir.join(file), bytes).unwrap();
    }
    let mut header = serde_json::to_vec(
        &serde_json::json!({"norm":{"dtype":"BF16", "shape":[2], "data_offsets":[0,4]}}),
    )
    .unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut preserved = (header.len() as u64).to_le_bytes().to_vec();
    preserved.extend(header);
    preserved.extend([0x80, 0x3f, 0x80, 0xbf]);
    fs::write(dir.join("preserved.safetensors"), &preserved).unwrap();
    let mut assets = Vec::new();
    for (name, _) in LEGACY_SALT_V2_HF_ASSETS {
        let bytes: &[u8] = if name == "config.json" {
            b"{\"model_type\":\"fixture\"}\n"
        } else {
            b"original asset\n"
        };
        fs::write(dir.join(name), bytes).unwrap();
        assets
            .push(serde_json::json!({"file":name, "bytes":bytes.len(), "package_id":hash(bytes)}));
    }
    let value = serde_json::json!({
        "schema_version":3, "artifact_kind":"qwen3.6-language-mtp-salt-v2-hf-bundle",
        "complete_model":false, "official_payload_authenticated":false,
        "packing":match codec { SaltV2Codec::D2=>"d2", SaltV2Codec::B3=>"b3", SaltV2Codec::S34=>"s34", _=>panic!() },
        "source_revision":tritium_nn::QWEN36_27B_REVISION,
        "source_model_id":"original-model", "source_identity_status":"measured-awaiting-official-registration",
        "completion_id":"completion", "campaign_id":"campaign", "admission_id":"admission", "selection_id":"selection",
        "preserved":{"file":"preserved.safetensors", "package_id":hash(&preserved),
            "serialized_bytes":preserved.len(), "payload_bytes":4, "tensors":1},
        "profiles":profiles, "hf_assets":assets });
    fs::write(
        dir.join("tritium.json"),
        serde_json::to_vec_pretty(&value).unwrap(),
    )
    .unwrap();
    (value, tensors.try_into().unwrap())
}
fn verify(root: &Path) -> Output {
    let model = root.join("output/model.trit");
    let log = root.join("output/evidence.jsonl");
    run(&[
        "migrate".as_ref(),
        "verify".as_ref(),
        "--model".as_ref(),
        model.as_os_str(),
        "--import-evidence".as_ref(),
        log.as_os_str(),
    ])
}

#[test]
fn actual_bundle_both_profiles_all_source_codecs_exact_trits_scales_assets_and_replay() {
    for source in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
        for (selection, index) in [("compact-v1", 0), ("near-lossless-v1", 1)] {
            for target in ["d2", "b3"] {
                let scratch = Scratch::new();
                let (_, tensors) = fixture(&scratch.0, source);
                let output = invoke(&scratch.0, selection, target, &[]);
                assert!(
                    output.status.success(),
                    "{}",
                    String::from_utf8_lossy(&output.stderr)
                );
                assert!(!scratch.0.join("spool").exists());
                let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
                assert_eq!(report["status"], "imported-not-qualified");
                assert_eq!(report["complete_model"], false);
                let mut model = TritModelReader::open(
                    fs::File::open(scratch.0.join("output/model.trit")).unwrap(),
                    TritPackageReadLimits {
                        max_source_bytes: 1_000_000,
                        max_manifest_bytes: 65_536,
                        max_directory_bytes: 65_536,
                    },
                )
                .unwrap();
                assert!(matches!(model.manifest().levels[0].fit, LevelFit::Imported));
                assert_eq!(model.manifest().assets.len(), 9);
                for name in LEGACY_SALT_V2_HF_ASSETS
                    .iter()
                    .map(|(n, _)| *n)
                    .chain(["tritium.json"])
                {
                    let mut bytes = Vec::new();
                    model
                        .visit_asset(name, |chunk| -> Result<(), ()> {
                            bytes.extend_from_slice(chunk);
                            Ok(())
                        })
                        .unwrap();
                    assert_eq!(
                        bytes,
                        fs::read(scratch.0.join("bundle").join(name)).unwrap()
                    );
                }
                let mut dense = Vec::new();
                model
                    .visit_tensor("imported", "norm", |c| -> Result<(), ()> {
                        dense.extend_from_slice(c);
                        Ok(())
                    })
                    .unwrap();
                assert_eq!(dense, [0x80, 0x3f, 0x80, 0xbf]);
                let meta = model.tensor_meta("imported", "proj").unwrap().clone();
                let TensorStorage::Additive {
                    layout,
                    planes: PlaneCounts::PerTile(counts),
                } = meta.storage
                else {
                    panic!()
                };
                let mut payload = Vec::new();
                model
                    .visit_tensor("imported", "proj", |c| -> Result<(), ()> {
                        payload.extend_from_slice(c);
                        Ok(())
                    })
                    .unwrap();
                let recovered =
                    AdditiveTensor::decode_per_tile_payload(layout, &counts, &payload).unwrap();
                let tiles = tritium_core::OwnedPerTileIndex::new(layout, counts).unwrap();
                let original = &tensors[index].tiles()[0];
                for (p, plane) in original.planes().iter().enumerate() {
                    for row in 0..2 {
                        for tile in tiles.as_index().tiles_in_row(p, row).unwrap() {
                            for col in tile.columns() {
                                assert_eq!(
                                    recovered.trits()[tile.trit_offset() + col],
                                    plane.trits()[row * 128 + col]
                                );
                            }
                            assert_eq!(
                                f16::from_f32(recovered.scales()[tile.scale_offset()]).to_bits(),
                                plane.scales()[row].to_bits()
                            );
                        }
                    }
                }
                let log = verify_jsonl(
                    &fs::read_to_string(scratch.0.join("output/evidence.jsonl")).unwrap(),
                )
                .unwrap();
                assert_eq!(LegacyModelImportView::fold(&log).unwrap().len(), 1);
                assert!(verify(&scratch.0).status.success());
                let first = fs::read(scratch.0.join("output/model.trit")).unwrap();
                assert!(!invoke(&scratch.0, selection, target, &[]).status.success());
                assert_eq!(
                    first,
                    fs::read(scratch.0.join("output/model.trit")).unwrap()
                );
                assert!(!scratch.0.join("spool").exists());
            }
        }
    }
}

#[test]
fn original_descriptor_and_file_mismatches_fail_without_spool_or_output() {
    for case in 0..15 {
        let scratch = Scratch::new();
        let (mut value, _) = fixture(&scratch.0, SaltV2Codec::B3);
        match case {
            0 => value["schema_version"] = 4.into(),
            1 => value["unknown_future_field"] = true.into(),
            2 => value["complete_model"] = true.into(),
            3 => value["profiles"]["compact-v1"]["package_id"] = "wrong-domain".into(),
            4 => value["profiles"]["compact-v1"]["resident_bytes"] = 1.into(),
            5 => value["preserved"]["package_id"] = "wrong".into(),
            6 => value["preserved"]["payload_bytes"] = 3.into(),
            7 => value["preserved"]["tensors"] = 2.into(),
            8 => value["hf_assets"][0]["file"] = "../escape".into(),
            9 => {
                value["hf_assets"].as_array_mut().unwrap().pop();
            }
            10 => value["hf_assets"][0]["package_id"] = "wrong".into(),
            11 => value["packing"] = "d2".into(),
            12 => {
                fs::write(scratch.0.join("bundle/tokenizer.json"), b"tampered").unwrap();
            }
            13 => value["source_revision"] = "unpinned".into(),
            14 => value["preserved"]["unknown_field"] = true.into(),
            _ => unreachable!(),
        }
        fs::write(
            scratch.0.join("bundle/tritium.json"),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap();
        let result = invoke(&scratch.0, "compact-v1", "b3", &[]);
        assert!(!result.status.success(), "case {case}");
        assert!(!scratch.0.join("output").exists());
        assert!(!scratch.0.join("spool").exists());
    }
}

#[test]
fn caps_preexisting_spool_and_corrupt_output_log_refused() {
    for flag in [
        "--max-source-bytes",
        "--max-dense-bytes",
        "--max-asset-bytes",
        "--max-trits",
        "--max-scales",
        "--max-working-bytes",
        "--max-tensor-bytes",
        "--max-spool-bytes",
        "--max-package-bytes",
        "--max-manifest-bytes",
        "--max-directory-bytes",
    ] {
        let scratch = Scratch::new();
        fixture(&scratch.0, SaltV2Codec::B3);
        assert!(
            !invoke(&scratch.0, "compact-v1", "b3", &[flag, "1"])
                .status
                .success(),
            "{flag}"
        );
        assert!(!scratch.0.join("spool").exists());
        assert!(!scratch.0.join("output").exists());
    }
    let scratch = Scratch::new();
    fixture(&scratch.0, SaltV2Codec::B3);
    fs::write(scratch.0.join("spool"), b"foreign").unwrap();
    assert!(!invoke(&scratch.0, "compact-v1", "b3", &[]).status.success());
    assert_eq!(fs::read(scratch.0.join("spool")).unwrap(), b"foreign");
    fs::remove_file(scratch.0.join("spool")).unwrap();
    assert!(invoke(&scratch.0, "compact-v1", "b3", &[]).status.success());
    let log = scratch.0.join("output/evidence.jsonl");
    let original = fs::read(&log).unwrap();
    fs::write(&log, b"{}\n").unwrap();
    assert!(!verify(&scratch.0).status.success());
    fs::write(&log, original).unwrap();
    let model = scratch.0.join("output/model.trit");
    let mut bytes = fs::read(&model).unwrap();
    bytes[0] ^= 1;
    fs::write(&model, bytes).unwrap();
    assert!(!verify(&scratch.0).status.success());
}

#[cfg(unix)]
#[test]
fn symlink_source_files_refused() {
    for name in [
        "tritium.json",
        "preserved.safetensors",
        "compact.tsalt2",
        "tokenizer.json",
    ] {
        let scratch = Scratch::new();
        fixture(&scratch.0, SaltV2Codec::B3);
        let original = scratch.0.join("bundle").join(name);
        let relocated = scratch.0.join("original");
        fs::rename(&original, &relocated).unwrap();
        std::os::unix::fs::symlink(&relocated, &original).unwrap();
        assert!(!invoke(&scratch.0, "compact-v1", "b3", &[]).status.success());
        assert!(!scratch.0.join("spool").exists());
        assert!(!scratch.0.join("output").exists());
    }
}
