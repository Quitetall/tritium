use std::process::Command;
fn command() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_tritium"));
    c.args(["--evidence", "off", "qtea"]);
    c
}
#[test]
fn qtea_requires_explicit_complete_inputs_and_selection_for_execution() {
    for args in [
        vec!["inspect"],
        vec!["run", "--request", "/absent"],
        vec![
            "inspect",
            "--request",
            "/absent",
            "--source-shard",
            "/source",
        ],
    ] {
        let output = command().args(args).output().unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("required"));
    }
}
#[test]
fn qtea_rejects_zero_json_caps_before_opening_source() {
    for cap in ["--max-json-bytes", "--max-report-bytes"] {
        let output = command()
            .args([
                "inspect",
                "--request",
                "/absent",
                "--source-shard",
                "/source",
                "--curvature",
                "owner=/metric",
                cap,
                "0",
            ])
            .output()
            .unwrap();
        assert!(!output.status.success());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains("caps must be positive"), "{error}");
        assert!(!error.contains("No such file"));
    }
}
#[test]
fn qtea_duplicate_and_empty_named_inputs_are_not_accepted_as_valid_bindings() {
    let output = command()
        .args([
            "inspect",
            "--request",
            "/absent",
            "--source-shard",
            "/source",
            "--curvature",
            "=metric",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("nonempty"));
}
