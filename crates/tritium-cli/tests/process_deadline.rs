//! Exercise the actual portable child runner without GNU coreutils.
#[path = "support/process.rs"]
mod process;
use std::{
    io::Write,
    process::Command,
    time::{Duration, Instant},
};

fn child(mode: &str) -> Command {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", "child_probe", "--nocapture"])
        .env("TRITIUM_TEST_CHILD_MODE", mode);
    command
}

#[test]
fn child_probe() {
    match std::env::var("TRITIUM_TEST_CHILD_MODE").ok().as_deref() {
        Some("sleep") => std::thread::sleep(Duration::from_secs(60)),
        Some("flood") => {
            std::io::stdout()
                .write_all(&vec![b'o'; 1024 * 1024])
                .unwrap();
            std::io::stderr()
                .write_all(&vec![b'e'; 1024 * 1024])
                .unwrap();
        }
        Some("overflow") => {
            let _ = std::io::stdout().write_all(&vec![b'x'; 5 * 1024 * 1024]);
        }
        Some("failure") => std::process::exit(7),
        _ => {}
    }
}

#[test]
fn kills_and_reaps_direct_child_at_deadline() {
    let start = Instant::now();
    let error = process::output(&mut child("sleep"), Duration::from_millis(100)).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
    assert!(start.elapsed() < Duration::from_secs(5));
}

#[test]
fn concurrently_drains_both_pipes_without_deadlock() {
    let output = process::output(&mut child("flood"), Duration::from_secs(10)).unwrap();
    assert!(output.status.success());
    assert!(output.stdout.iter().filter(|&&byte| byte == b'o').count() >= 1024 * 1024);
    assert!(output.stderr.iter().filter(|&&byte| byte == b'e').count() >= 1024 * 1024);
}

#[test]
fn preserves_exit_status_and_refuses_unbounded_output() {
    assert_eq!(
        process::output(&mut child("failure"), Duration::from_secs(10))
            .unwrap()
            .status
            .code(),
        Some(7)
    );
    assert_eq!(
        process::output(&mut child("overflow"), Duration::from_secs(10))
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::InvalidData
    );
}

#[test]
#[cfg(unix)]
fn actual_cli_runs_when_gnu_timeout_is_unavailable() {
    let legacy = Command::new("timeout")
        .env("PATH", "")
        .output()
        .unwrap_err();
    assert_eq!(legacy.kind(), std::io::ErrorKind::NotFound);
    let mut cli = Command::new(env!("CARGO_BIN_EXE_tritium"));
    cli.env("PATH", "").arg("--help");
    assert!(
        process::output(&mut cli, Duration::from_secs(10))
            .unwrap()
            .status
            .success()
    );
}
