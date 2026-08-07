use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn tritium_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_tritium"))
}

#[test]
fn release_digest_is_exact_and_rejects_symlinks() {
    let root = std::env::temp_dir().join(format!("tritium-release-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let artifact = root.join("artifact.bin");
    fs::write(&artifact, b"tritium release\n").unwrap();

    let output = Command::new(tritium_bin())
        .args(["release", "digest"])
        .arg(&artifact)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["schema"], "tritium.file-identity.v1");
    assert_eq!(value["bytes"], 16);
    assert_eq!(
        value["sha256"],
        "aa5c8b507e381b1da1c6f63e1153ac0bd768d9d77ad4b0be7044aba84b89e741"
    );
    assert_eq!(
        value["blake3"],
        "46b6f12a77c83c6f92a80c903fa37743cbcda9cd08f2498f3e4e132da77c1db9"
    );

    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&artifact, root.join("link.bin")).unwrap();
        let rejected = Command::new(tritium_bin())
            .args(["release", "digest"])
            .arg(root.join("link.bin"))
            .output()
            .unwrap();
        assert!(!rejected.status.success());
        assert!(String::from_utf8_lossy(&rejected.stderr).contains("ordinary file"));
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn release_digest_stream_binds_raw_and_transport_identities() {
    let mut child = Command::new(tritium_bin())
        .args(["release", "digest-stream"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"tritium release\n")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["schema"], "tritium.stream-identity.v1");
    assert_eq!(value["bytes"], 16);
    assert_eq!(
        value["sha256"],
        "aa5c8b507e381b1da1c6f63e1153ac0bd768d9d77ad4b0be7044aba84b89e741"
    );
    assert_eq!(
        value["blake3"],
        "46b6f12a77c83c6f92a80c903fa37743cbcda9cd08f2498f3e4e132da77c1db9"
    );
    assert_eq!(
        value["package_id"],
        "trp1_dd3bacaf002574cd6227cb048d0478e6943ee48f4ed38fa7e565e1efe1829233"
    );
}
