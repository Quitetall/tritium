use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn tritium_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_tritium"))
}

fn temp_dir() -> PathBuf {
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "tritium-transport-cli-{}-{sequence}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("create test directory");
    path
}

fn run(args: &[&str]) -> Output {
    Command::new(tritium_bin())
        .args(args)
        .output()
        .expect("run tritium CLI")
}

#[test]
fn transport_cli_reports_rans_without_changing_resident_denominator_and_round_trips() {
    let directory = temp_dir();
    let source = directory.join("source.bin");
    let packed = directory.join("source.trns");
    let restored = directory.join("restored.bin");
    let logical = vec![0x5a; 4096];
    std::fs::write(&source, &logical).expect("write source");

    let pack = run(&[
        "transport",
        "pack",
        source.to_str().unwrap(),
        packed.to_str().unwrap(),
    ]);
    assert!(
        pack.status.success(),
        "{}",
        String::from_utf8_lossy(&pack.stderr)
    );

    let inspect = run(&["transport", "inspect", packed.to_str().unwrap()]);
    assert!(
        inspect.status.success(),
        "{}",
        String::from_utf8_lossy(&inspect.stderr)
    );
    let report = String::from_utf8(inspect.stdout).expect("UTF-8 report");
    assert!(report.contains("transport: TRNS v2"), "{report}");
    assert!(report.contains("logical_bytes: 4096"), "{report}");
    assert!(report.contains("rans_chunks: 1"), "{report}");
    assert!(
        report.contains("resident_denominator: logical_bytes"),
        "{report}"
    );

    let unpack = run(&[
        "transport",
        "unpack",
        packed.to_str().unwrap(),
        restored.to_str().unwrap(),
    ]);
    assert!(
        unpack.status.success(),
        "{}",
        String::from_utf8_lossy(&unpack.stderr)
    );
    assert_eq!(
        std::fs::read(restored).expect("read restored bytes"),
        logical
    );
    std::fs::remove_dir_all(directory).expect("remove test directory");
}
