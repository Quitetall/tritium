use std::{
    fs::{self, OpenOptions},
    io::{Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    process::Command,
};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokenizers::{Tokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};

struct DatasetFixture {
    repo_id: &'static str,
    revision: &'static str,
    config: &'static str,
    data_dir: Option<&'static str>,
    text_field: &'static str,
    sequence_count: usize,
}

const DATASETS: [DatasetFixture; 3] = [
    DatasetFixture {
        repo_id: "allenai/c4",
        revision: "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
        config: "en",
        data_dir: None,
        text_field: "text",
        sequence_count: 256,
    },
    DatasetFixture {
        repo_id: "open-web-math/open-web-math",
        revision: "fde8ef8de2300f5e778f56261843dab89f230815",
        config: "default",
        data_dir: None,
        text_field: "text",
        sequence_count: 128,
    },
    DatasetFixture {
        repo_id: "bigcode/starcoderdata",
        revision: "9fc30b578cedaec69e47302df72cf00feed7c8c4",
        config: "default",
        data_dir: Some("python"),
        text_field: "content",
        sequence_count: 128,
    },
];

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn record(root: &Path, path: &str) -> Value {
    let bytes = fs::read(root.join(path)).unwrap();
    json!({"path": path, "bytes": bytes.len(), "sha256": hex(&Sha256::digest(&bytes))})
}

fn root() -> PathBuf {
    std::env::temp_dir().join(format!(
        "tritium-stage7-evidence-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
    ))
}

fn write_model(root: &Path) -> PathBuf {
    let model = root.join("model");
    fs::create_dir(&model).unwrap();
    let vocab = [
        ("[UNK]".to_owned(), 0),
        ("x".to_owned(), 1),
        ("[EOS]".to_owned(), 2),
    ]
    .into_iter()
    .collect();
    let wordlevel = WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".to_owned())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(wordlevel);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer.save(model.join("tokenizer.json"), false).unwrap();
    fs::write(
        model.join("tokenizer_config.json"),
        r#"{"eos_token":"[EOS]"}"#,
    )
    .unwrap();
    fs::write(model.join("config.json"), r#"{"vocab_size":3}"#).unwrap();
    fs::write(model.join("special_tokens_map.json"), b"{}\n").unwrap();
    fs::write(model.join("vocab.json"), b"{}\n").unwrap();
    fs::write(model.join("merges.txt"), b"#version: 0.2\n").unwrap();
    #[cfg(unix)]
    {
        let hub = root.join("models--fixture--smollm");
        let blobs = hub.join("blobs");
        let snapshot = hub.join("snapshots/0123456789abcdef0123456789abcdef01234567");
        fs::create_dir_all(&blobs).unwrap();
        fs::create_dir_all(&snapshot).unwrap();
        for name in [
            "config.json",
            "merges.txt",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ] {
            let blob = blobs.join(name);
            fs::rename(model.join(name), &blob).unwrap();
            std::os::unix::fs::symlink(&blob, snapshot.join(name)).unwrap();
        }
        fs::remove_dir(model).unwrap();
        snapshot
    }
    #[cfg(not(unix))]
    {
        model
    }
}

fn write_sampled_rows(root: &Path) -> PathBuf {
    let mut partitions = serde_json::Map::new();
    let mut sequence_ordinal = 0_usize;
    for (partition_ordinal, (partition, seed)) in [
        ("calibration", 11_u64),
        ("refinement", 12),
        ("validation", 13),
        ("evaluation", 14),
    ]
    .into_iter()
    .enumerate()
    {
        let mut datasets = Vec::new();
        for (dataset_ordinal, dataset) in DATASETS.iter().enumerate() {
            let path = format!("rows/{partition}-{dataset_ordinal}.jsonl");
            let mut rows = String::new();
            for ordinal in 0..dataset.sequence_count {
                let mut words = vec!["x"; 2_048];
                for (bit, word) in words.iter_mut().take(11).enumerate() {
                    if sequence_ordinal & (1 << bit) == 0 {
                        *word = "y";
                    }
                }
                let text = words.join(" ");
                let content_sha = hex(&Sha256::digest(text.as_bytes()));
                let row_index = partition_ordinal * 10_000 + dataset_ordinal * 1_000 + ordinal;
                rows.push_str(
                    &serde_json::to_string(&json!({
                        "row_index": row_index,
                        "content_sha256": content_sha,
                        "text": text,
                    }))
                    .unwrap(),
                );
                rows.push('\n');
                sequence_ordinal += 1;
            }
            fs::create_dir_all(root.join("rows")).unwrap();
            fs::write(root.join(&path), rows).unwrap();
            datasets.push(json!({
                "repo_id": dataset.repo_id,
                "revision": dataset.revision,
                "config": dataset.config,
                "data_dir": dataset.data_dir,
                "split": "train",
                "text_field": dataset.text_field,
                "rows": record(root, &path),
            }));
        }
        partitions.insert(
            partition.to_owned(),
            json!({
                "sampling_seed": seed,
                "datasets": datasets,
            }),
        );
    }
    let path = root.join("sampled-rows.json");
    fs::write(
        &path,
        serde_json::to_vec(&json!({
            "schema": "tritium.stage7-sampled-rows.v1",
            "partitions": partitions,
        }))
        .unwrap(),
    )
    .unwrap();
    path
}

#[test]
fn builds_content_bound_stage7_token_pack_from_sampled_rows() {
    let root = root();
    fs::create_dir(&root).unwrap();
    let model = write_model(&root);
    let sampled = write_sampled_rows(&root);
    let output = root.join("pack");
    let command = Command::new(env!("CARGO_BIN_EXE_tritium"))
        .args(["salt", "build-stage7-evidence-pack", "--model-dir"])
        .arg(&model)
        .arg("--sampled-rows")
        .arg(&sampled)
        .arg("--output-dir")
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        command.status.success(),
        "{}",
        String::from_utf8_lossy(&command.stderr)
    );
    let manifest: Value =
        serde_json::from_slice(&fs::read(output.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["schema"], "tritium.stage7-token-evidence-pack.v1");
    assert_eq!(manifest["token_encoding"], "u32le");
    assert_eq!(manifest["tokenizer_vocab_size"], 3);
    assert_eq!(
        fs::metadata(output.join("stage7.u32le")).unwrap().len(),
        16_777_216
    );
    for partition in ["calibration", "refinement", "validation", "evaluation"] {
        assert_eq!(
            manifest["partitions"][partition]["sequences"]
                .as_array()
                .unwrap()
                .len(),
            512
        );
    }
    let starcoder = manifest["partitions"]["calibration"]["sequences"]
        .as_array()
        .unwrap()
        .iter()
        .find(|sequence| sequence["dataset_repo_id"] == "bigcode/starcoderdata")
        .unwrap();
    assert_eq!(starcoder["dataset_config"], "default");
    assert_eq!(starcoder["dataset_data_dir"], "python");
    assert_eq!(starcoder["source_rows"][0]["text_field"], "content");
    let mut provenance = serde_json::Map::new();
    for (partition, seed) in [
        ("calibration", 11_u64),
        ("refinement", 12),
        ("validation", 13),
        ("evaluation", 14),
    ] {
        let members = manifest["partitions"][partition]["sequences"]
            .as_array()
            .unwrap()
            .iter()
            .map(|sequence| sequence["id"].clone())
            .collect::<Vec<_>>();
        provenance.insert(
            partition.to_owned(),
            json!({
                "sampling_seed": seed,
                "tokenizer_digest": manifest["tokenizer_digest"],
                "members": members,
                "datasets": DATASETS.iter().map(|dataset| json!({
                    "repo_id": dataset.repo_id,
                    "revision": dataset.revision,
                    "fraction_ppm": if dataset.sequence_count == 256 { 500_000 } else { 250_000 },
                })).collect::<Vec<_>>(),
            }),
        );
    }
    let campaign = root.join("campaign-fragment.json");
    fs::write(
        &campaign,
        serde_json::to_vec(&json!({"provenance": provenance})).unwrap(),
    )
    .unwrap();
    let qualifier =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/qualify-stage7-recipe-freeze.py");
    let cross_check = Command::new("python3")
        .arg("-c")
        .arg(
            "import json,runpy,sys; from pathlib import Path; m=runpy.run_path(sys.argv[1]); c=json.loads(Path(sys.argv[2]).read_bytes()); m['_validate_token_evidence_pack'](Path(sys.argv[3]),c,tokenizer_digest=sys.argv[4],tokenizer_vocab_size=3)",
        )
        .arg(qualifier)
        .arg(campaign)
        .arg(output.join("manifest.json"))
        .arg(manifest["tokenizer_digest"].as_str().unwrap())
        .output()
        .unwrap();
    assert!(
        cross_check.status.success(),
        "{}",
        String::from_utf8_lossy(&cross_check.stderr)
    );

    let inspected = Command::new(env!("CARGO_BIN_EXE_tritium"))
        .args(["salt", "inspect-stage7-evidence-pack", "--model-dir"])
        .arg(&model)
        .arg("--manifest")
        .arg(output.join("manifest.json"))
        .arg("--expected-pack-id")
        .arg(manifest["pack_id"].as_str().unwrap())
        .args([
            "--partition",
            "calibration",
            "--start-sequence",
            "0",
            "--sequence-count",
            "128",
        ])
        .output()
        .unwrap();
    assert!(
        inspected.status.success(),
        "{}",
        String::from_utf8_lossy(&inspected.stderr)
    );
    let receipt: Value = serde_json::from_slice(&inspected.stdout).unwrap();
    assert_eq!(receipt["schema"], "tritium.stage7-token-evidence-read.v1");
    assert_eq!(receipt["pack_id"], manifest["pack_id"]);
    assert_eq!(receipt["partition"], "calibration");
    assert_eq!(receipt["sampling_seed"], 11);
    assert_eq!(receipt["start_sequence"], 0);
    assert_eq!(receipt["sequence_count"], 128);
    assert_eq!(receipt["tokens_per_sequence"], 2_048);
    assert_eq!(receipt["token_count"], 128 * 2_048);
    let expected_prefix_ids = manifest["partitions"]["calibration"]["sequences"]
        .as_array()
        .unwrap()[..128]
        .iter()
        .map(|sequence| sequence["id"].clone())
        .collect::<Vec<_>>();
    assert_eq!(
        receipt["sequence_ids"].as_array().unwrap(),
        &expected_prefix_ids
    );
    let payload_prefix = &fs::read(output.join("stage7.u32le")).unwrap()[..128 * 2_048 * 4];
    assert_eq!(
        receipt["ordered_token_sha256"],
        format!("sha256:{}", hex(&Sha256::digest(payload_prefix)))
    );
    assert_eq!(receipt["terminal_validated"], true);

    let wrong_campaign = Command::new(env!("CARGO_BIN_EXE_tritium"))
        .args(["salt", "inspect-stage7-evidence-pack", "--model-dir"])
        .arg(&model)
        .arg("--manifest")
        .arg(output.join("manifest.json"))
        .arg("--expected-pack-id")
        .arg(format!("sha256:{}", "0".repeat(64)))
        .args([
            "--partition",
            "calibration",
            "--start-sequence",
            "0",
            "--sequence-count",
            "128",
        ])
        .output()
        .unwrap();
    assert!(!wrong_campaign.status.success());
    assert!(
        String::from_utf8_lossy(&wrong_campaign.stderr)
            .contains("token evidence pack differs from expected campaign"),
        "{}",
        String::from_utf8_lossy(&wrong_campaign.stderr)
    );

    let mut payload = OpenOptions::new()
        .write(true)
        .open(output.join("stage7.u32le"))
        .unwrap();
    payload.seek(SeekFrom::Start(9_000_000)).unwrap();
    payload.write_all(&[0xff]).unwrap();
    payload.sync_all().unwrap();
    drop(payload);
    let rejected = Command::new(env!("CARGO_BIN_EXE_tritium"))
        .args(["salt", "inspect-stage7-evidence-pack", "--model-dir"])
        .arg(&model)
        .arg("--manifest")
        .arg(output.join("manifest.json"))
        .arg("--expected-pack-id")
        .arg(manifest["pack_id"].as_str().unwrap())
        .args([
            "--partition",
            "calibration",
            "--start-sequence",
            "0",
            "--sequence-count",
            "128",
        ])
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    assert!(
        String::from_utf8_lossy(&rejected.stderr).contains("token payload identity differs"),
        "{}",
        String::from_utf8_lossy(&rejected.stderr)
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn rejects_empty_sampled_row_path_without_panicking() {
    let root = root();
    fs::create_dir(&root).unwrap();
    let model = write_model(&root);
    let sampled = write_sampled_rows(&root);
    let mut manifest: Value = serde_json::from_slice(&fs::read(&sampled).unwrap()).unwrap();
    manifest["partitions"]["calibration"]["datasets"][0]["rows"]["path"] = json!("");
    fs::write(&sampled, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let command = Command::new(env!("CARGO_BIN_EXE_tritium"))
        .args(["salt", "build-stage7-evidence-pack", "--model-dir"])
        .arg(&model)
        .arg("--sampled-rows")
        .arg(&sampled)
        .arg("--output-dir")
        .arg(root.join("pack"))
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&command.stderr);
    assert!(!command.status.success());
    assert!(
        stderr.contains("sampled rows path is not contained"),
        "{stderr}"
    );
    assert!(!stderr.contains("panicked"), "{stderr}");
    fs::remove_dir_all(root).unwrap();
}
