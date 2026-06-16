// Compiles the CUDA kernel to PTX via nvcc, but only when the `cuda` feature is
// enabled (Cargo sets CARGO_FEATURE_CUDA). With the feature off this is a no-op,
// so cpu-only builds never need a CUDA toolkit — which is exactly how CI builds
// this crate on lanes without a GPU.
//
// When the feature is on, we locate nvcc (via CUDA_PATH / CUDA_HOME, else the
// conventional /usr/local/cuda, else PATH) and emit `tq2_0_add.ptx` into OUT_DIR.
// The crate then `include_str!`s that PTX and JIT-loads it at runtime. A missing
// nvcc is a hard error with an actionable message rather than a confusing link
// failure later.
//
// PTX is a *virtual* ISA, so a single `-ptx` build targets one `compute_XX` and
// the driver JIT-recompiles it for the concrete device at load. We target the
// lowest architecture Tritium supports (compute_75) because its PTX runs on every
// newer device (Turing → Ampere → Ada → Hopper → Blackwell), which is what
// `SUPPORTED_SM_ARCHS` documents and what the GPU CI lane (Wave D) exercises.
// (CUDA 13 dropped Maxwell/Pascal/Volta, so compute_70 and below are gone.)

use std::path::{Path, PathBuf};
use std::process::Command;

/// SM architectures this kernel is validated to run on: Turing, Ampere, Ada,
/// Hopper. The emitted PTX is built for the *first* (lowest) of these; PTX is
/// forward-compatible, so the driver JITs it up to whatever device is present.
/// CUDA 13 removed Volta and earlier, so `compute_75` is the floor.
const SUPPORTED_SM_ARCHS: &[&str] = &["75", "80", "89", "90"];

fn main() {
    // Rebuild triggers regardless of feature state: cheap, and correct when the
    // feature is later toggled on.
    println!("cargo:rerun-if-changed=kernels/tq2_0_add.cu");
    // WF-A (v0.30): the IMMA prefill kernel. Its `mma.m16n8k32` int8 shape needs
    // sm_80+, so WF-A emits a SECOND PTX target (compute_80) for it here — it is
    // NOT compiled yet (the placeholder .cu only declares the entry point). The
    // rerun trigger is wired now so toggling it on later rebuilds correctly.
    println!("cargo:rerun-if-changed=kernels/tq2_0_imma.cu");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        // Default (cpu-only) build: nothing to compile, no toolkit required.
        return;
    }

    let nvcc = find_nvcc().unwrap_or_else(|| {
        panic!(
            "tritium-cuda: the `cuda` feature is enabled but `nvcc` was not found. \
             Install the CUDA toolkit and set CUDA_PATH or CUDA_HOME (or put nvcc on PATH). \
             Looked at $CUDA_PATH/bin, $CUDA_HOME/bin, /usr/local/cuda/bin, and PATH."
        )
    });

    let out_dir = PathBuf::from(
        std::env::var_os("OUT_DIR").expect("OUT_DIR is always set by cargo for build scripts"),
    );
    let ptx_path = out_dir.join("tq2_0_add.ptx");
    let src = Path::new("kernels/tq2_0_add.cu");

    // PTX is generated for the lowest supported virtual arch; the driver JITs it
    // up to the device's actual SM at load time (so one PTX covers sm_70..sm_90).
    let min_arch = SUPPORTED_SM_ARCHS
        .first()
        .expect("SUPPORTED_SM_ARCHS is never empty");

    let mut cmd = Command::new(&nvcc);
    cmd.arg("-ptx")
        .arg(src)
        .arg("-o")
        .arg(&ptx_path)
        // Virtual-only target: `arch=compute_XX,code=compute_XX` emits PTX (no
        // SASS), which the runtime driver recompiles for the present GPU.
        .arg("-gencode")
        .arg(format!("arch=compute_{min_arch},code=compute_{min_arch}"))
        // -O3 keeps the add-only loop tight; correctness is unaffected.
        .arg("-O3");

    let status = cmd.status().unwrap_or_else(|e| {
        panic!(
            "tritium-cuda: failed to invoke nvcc ({}): {e}",
            nvcc.display()
        )
    });
    assert!(
        status.success(),
        "tritium-cuda: nvcc failed to compile kernels/tq2_0_add.cu (exit {status})"
    );
    assert!(
        ptx_path.exists(),
        "tritium-cuda: nvcc reported success but {} was not produced",
        ptx_path.display()
    );
}

/// Locate the `nvcc` binary: prefer the toolkit pointed at by `CUDA_PATH` /
/// `CUDA_HOME`, then the conventional install prefix, then bare `nvcc` on PATH.
fn find_nvcc() -> Option<PathBuf> {
    let exe = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };

    for var in ["CUDA_PATH", "CUDA_HOME"] {
        if let Some(root) = std::env::var_os(var) {
            let candidate = Path::new(&root).join("bin").join(exe);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    let conventional = Path::new("/usr/local/cuda/bin").join(exe);
    if conventional.is_file() {
        return Some(conventional);
    }

    // Fall back to PATH: trust nvcc to resolve if it is reachable.
    if Command::new(exe).arg("--version").output().is_ok() {
        return Some(PathBuf::from(exe));
    }

    None
}
