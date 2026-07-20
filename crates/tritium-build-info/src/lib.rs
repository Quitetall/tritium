//! Build-script support for source-bound Tritium release evidence.

#![forbid(unsafe_code)]

use std::path::Path;
use std::process::Command;

/// Emit a compile-time `TRITIUM_SOURCE_ID` covering the repository commit and
/// complete tracked, staged, unstaged, and currently-untracked source state.
///
/// Release automation may inject an already-verified identity through the
/// environment. Outside a Git checkout the emitted identity is deliberately
/// unverified and receipt admission rejects it.
pub fn emit_source_identity() {
    println!("cargo:rerun-if-env-changed=TRITIUM_SOURCE_ID");
    if let Ok(identity) = std::env::var("TRITIUM_SOURCE_ID") {
        println!("cargo:rustc-env=TRITIUM_SOURCE_ID={identity}");
        return;
    }

    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("Cargo sets CARGO_MANIFEST_DIR");
    let root = Path::new(&manifest)
        .ancestors()
        .nth(2)
        .expect("workspace crate is two levels below its root");
    watch_git_path(root, "HEAD");
    watch_git_path(root, "index");

    let Ok(head) = git(root, &["rev-parse", "HEAD"]) else {
        println!("cargo:rustc-env=TRITIUM_SOURCE_ID=unverified:no-git-metadata");
        return;
    };
    let head = String::from_utf8_lossy(&head).trim().to_owned();
    if let Ok(symbolic) = git(root, &["symbolic-ref", "HEAD"]) {
        let reference = String::from_utf8_lossy(&symbolic).trim().to_owned();
        if !reference.is_empty() {
            watch_git_path(root, &reference);
        }
    }

    let tracked = git(root, &["ls-files", "-z"]).expect("git tracked-file list must be readable");
    let untracked = git(root, &["ls-files", "--others", "--exclude-standard", "-z"])
        .expect("git untracked-file list must be readable");
    for relative in tracked
        .split(|byte| *byte == 0)
        .chain(untracked.split(|byte| *byte == 0))
        .filter(|path| !path.is_empty())
    {
        let relative = std::str::from_utf8(relative).expect("git paths must be UTF-8");
        println!("cargo:rerun-if-changed={}", root.join(relative).display());
    }

    let diff = git(root, &["diff", "--binary", "HEAD", "--", "."])
        .expect("git worktree diff must be readable");
    let mut hasher = blake3::Hasher::new();
    hasher.update(&diff);
    for relative in untracked
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
    {
        hasher.update(relative);
        let relative = std::str::from_utf8(relative).expect("git paths must be UTF-8");
        hasher.update(
            &std::fs::read(root.join(relative)).expect("untracked source must be readable"),
        );
    }
    if diff.is_empty() && untracked.is_empty() {
        println!("cargo:rustc-env=TRITIUM_SOURCE_ID=source-git:{head}");
    } else {
        println!(
            "cargo:rustc-env=TRITIUM_SOURCE_ID=source-git:{head}+dirty-blake3:{}",
            hasher.finalize().to_hex()
        );
    }
}

fn watch_git_path(root: &Path, name: &str) {
    let Ok(path) = git(root, &["rev-parse", "--git-path", name]) else {
        return;
    };
    let path = String::from_utf8_lossy(&path).trim().to_owned();
    let path = Path::new(&path);
    let path = if path.is_absolute() {
        path.to_owned()
    } else {
        root.join(path)
    };
    println!("cargo:rerun-if-changed={}", path.display());
}

fn git(root: &Path, args: &[&str]) -> Result<Vec<u8>, ()> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .map_err(|_| ())?;
    output.status.success().then_some(output.stdout).ok_or(())
}
