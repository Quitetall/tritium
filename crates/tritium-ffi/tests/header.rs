//! Header gates:
//! 1. `header_is_current` — the committed `include/tritium.h` matches what
//!    `cbindgen` produces from the current source. Runs on every platform; the
//!    drift gate keeps the C header honest without a build-time source write.
//! 2. `header_compiles_as_c_and_cpp` (Linux) — the header is valid C11 and
//!    C++17. The header is platform-independent, so Linux-only coverage proves
//!    it everywhere; macOS/Windows runners exercise only the drift gate.

const REGEN: &str = "cbindgen --config crates/tritium-ffi/cbindgen.toml \
                     --crate tritium-ffi --output crates/tritium-ffi/include/tritium.h";

fn normalize(s: &str) -> String {
    s.replace("\r\n", "\n")
}

#[test]
fn header_is_current() {
    let crate_dir = env!("CARGO_MANIFEST_DIR");
    let bindings = cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(cbindgen::Config::from_root_or_default(crate_dir))
        .generate()
        .expect("cbindgen failed to generate bindings");

    let mut buf: Vec<u8> = Vec::new();
    bindings.write(&mut buf);
    let generated = normalize(&String::from_utf8(buf).expect("header is UTF-8"));

    let committed_path = format!("{crate_dir}/include/tritium.h");
    let committed = normalize(
        &std::fs::read_to_string(&committed_path).expect("committed include/tritium.h is missing"),
    );

    assert_eq!(
        generated, committed,
        "include/tritium.h is out of sync with the Rust source. Regenerate:\n  {REGEN}"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn header_compiles_as_c_and_cpp() {
    use std::process::Command;

    let crate_dir = env!("CARGO_MANIFEST_DIR");
    let include = format!("{crate_dir}/include");
    let dir = std::env::temp_dir();
    let c_src = dir.join("tritium_ffi_probe.c");
    let cpp_src = dir.join("tritium_ffi_probe.cpp");
    // Reference one symbol so the header's declarations are actually parsed/used.
    std::fs::write(
        &c_src,
        "#include <tritium.h>\nint main(void) { return (int)tritium_abi_version(); }\n",
    )
    .unwrap();
    std::fs::write(
        &cpp_src,
        "#include <tritium.h>\nint main() { return (int)tritium_abi_version(); }\n",
    )
    .unwrap();

    // -fsyntax-only: parse + type-check the header, no codegen, no link (so the
    // undefined symbol at the call site is fine — we only validate the header).
    for (compiler, src, std_flag) in [("cc", &c_src, "-std=c11"), ("c++", &cpp_src, "-std=c++17")] {
        let out = Command::new(compiler)
            .arg(std_flag)
            .arg("-fsyntax-only")
            .arg("-Wall")
            .arg("-Wextra")
            .arg(format!("-I{include}"))
            .arg(src)
            .output();
        match out {
            Ok(o) => assert!(
                o.status.success(),
                "{compiler} failed to compile tritium.h:\n{}",
                String::from_utf8_lossy(&o.stderr)
            ),
            Err(e) => panic!("{compiler} not available to compile the header: {e}"),
        }
    }
}
