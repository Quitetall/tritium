// Compiles the HIP kernel to a relocatable code object via hipcc, but only when
// the `rocm` feature is enabled (Cargo sets CARGO_FEATURE_ROCM). With the feature
// off this is a NO-OP, so cpu-only Linux builds never need a ROCm toolkit — which
// is exactly how CI builds this crate on lanes without an AMD GPU. This mirrors
// tritium-cuda's build.rs, which guards nvcc behind CARGO_FEATURE_CUDA.
//
// When the feature is on, we locate hipcc (via ROCM_PATH / HIP_PATH, else the
// conventional /opt/rocm, else PATH) and emit `tq2_0_add.co` into OUT_DIR. The
// crate then `include_bytes!`s that code object and loads it at runtime with
// `hipModuleLoadData`. A missing hipcc is a hard error with an actionable message
// rather than a confusing link failure later.
//
// We compile to a code object (`--genco`) targeting the lowest AMD GPU arch
// Tritium supports (`gfx900`, GCN5/Vega) plus newer CDNA/RDNA targets, so the
// emitted object runs on the runner's actual device. Unlike CUDA PTX (a virtual
// ISA JIT'd by the driver), AMD code objects are arch-specific, so we list the
// supported `--offload-arch` targets and let hipcc bundle them into one fat object.

use std::path::{Path, PathBuf};
use std::process::Command;

/// AMD GPU architectures this kernel is built for. The add-only kernel is plain
/// f32 + bit twiddling (no matrix cores), so it runs on every one of these; hipcc
/// bundles all listed `--offload-arch` targets into a single fat code object and
/// the loader selects the matching slice for the present device at load.
///
/// Covers GCN5 (Vega, `gfx900`), CDNA (MI100 `gfx908`, MI200 `gfx90a`, MI300
/// `gfx942`), and RDNA2/3 consumer (`gfx1030`, `gfx1100`). Extend as new AMD
/// hardware is validated on the ROCm CI lane.
const SUPPORTED_GFX_ARCHS: &[&str] =
    &["gfx900", "gfx908", "gfx90a", "gfx942", "gfx1030", "gfx1100", "gfx1201"];

fn main() {
    // Rebuild triggers regardless of feature state: cheap, and correct when the
    // feature is later toggled on. Mirrors tritium-cuda.
    println!("cargo:rerun-if-changed=kernels/tq2_0_add.hip");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");

    if std::env::var_os("CARGO_FEATURE_ROCM").is_none() {
        // Default (cpu-only) build: nothing to compile, no toolkit required.
        return;
    }

    let hipcc = find_hipcc().unwrap_or_else(|| {
        panic!(
            "tritium-rocm: the `rocm` feature is enabled but `hipcc` was not found. \
             Install the ROCm toolkit and set ROCM_PATH or HIP_PATH (or put hipcc on PATH). \
             Looked at $ROCM_PATH/bin, $HIP_PATH/bin, /opt/rocm/bin, and PATH."
        )
    });

    let out_dir = PathBuf::from(
        std::env::var_os("OUT_DIR").expect("OUT_DIR is always set by cargo for build scripts"),
    );

    compile_code_object(
        &hipcc,
        Path::new("kernels/tq2_0_add.hip"),
        &out_dir.join("tq2_0_add.co"),
    );

    // Point the linker at the HIP runtime so the `#[link(name = "amdhip64")]` in
    // src/ffi.rs resolves: `libamdhip64.so` ships in `<rocm>/lib`, which cargo does
    // not add to the linker search path by default (this is the first-compile
    // `unable to find library -lamdhip64` otherwise). The rpath embed lets the test
    // /bin load it at run time even on a box whose ld.so cache lacks the ROCm path.
    let rocm_lib = rocm_root().join("lib");
    println!("cargo:rustc-link-search=native={}", rocm_lib.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", rocm_lib.display());
}

/// Compile a single `.hip` source to a relocatable AMD GPU code object covering
/// every arch in [`SUPPORTED_GFX_ARCHS`], emitting it at `co_path`. Panics with an
/// actionable message if hipcc is missing the source, fails, or silently produces
/// nothing. Mirrors tritium-cuda's `compile_ptx`.
fn compile_code_object(hipcc: &Path, src: &Path, co_path: &Path) {
    let mut cmd = Command::new(hipcc);
    // `--genco` emits a relocatable code object (the HIP analogue of `nvcc -ptx`'s
    // standalone artifact) that `hipModuleLoadData` consumes at runtime.
    cmd.arg("--genco").arg(src).arg("-o").arg(co_path);
    // One `--offload-arch` per supported target; hipcc bundles them into one fat
    // object and the loader picks the slice matching the device.
    for arch in SUPPORTED_GFX_ARCHS {
        cmd.arg(format!("--offload-arch={arch}"));
    }
    // -O3 keeps the kernel tight; correctness is unaffected.
    cmd.arg("-O3");

    let status = cmd.status().unwrap_or_else(|e| {
        panic!(
            "tritium-rocm: failed to invoke hipcc ({}): {e}",
            hipcc.display()
        )
    });
    assert!(
        status.success(),
        "tritium-rocm: hipcc failed to compile {} (exit {status})",
        src.display()
    );
    assert!(
        co_path.exists(),
        "tritium-rocm: hipcc reported success but {} was not produced",
        co_path.display()
    );
}

/// Resolve the ROCm install prefix for locating `libamdhip64.so`: `$ROCM_PATH`,
/// then `$HIP_PATH`, then the conventional `/opt/rocm`. `libamdhip64` ships in
/// `<prefix>/lib` on ROCm 5.x/6.x.
fn rocm_root() -> PathBuf {
    for var in ["ROCM_PATH", "HIP_PATH"] {
        if let Some(root) = std::env::var_os(var) {
            return PathBuf::from(root);
        }
    }
    PathBuf::from("/opt/rocm")
}

/// Locate the `hipcc` binary: prefer the toolkit pointed at by `ROCM_PATH` /
/// `HIP_PATH`, then the conventional install prefix, then bare `hipcc` on PATH.
/// Mirrors tritium-cuda's `find_nvcc`.
fn find_hipcc() -> Option<PathBuf> {
    let exe = if cfg!(windows) { "hipcc.exe" } else { "hipcc" };

    for var in ["ROCM_PATH", "HIP_PATH"] {
        if let Some(root) = std::env::var_os(var) {
            let candidate = Path::new(&root).join("bin").join(exe);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    let conventional = Path::new("/opt/rocm/bin").join(exe);
    if conventional.is_file() {
        return Some(conventional);
    }

    // Fall back to PATH: trust hipcc to resolve if it is reachable.
    if Command::new(exe).arg("--version").output().is_ok() {
        return Some(PathBuf::from(exe));
    }

    None
}
