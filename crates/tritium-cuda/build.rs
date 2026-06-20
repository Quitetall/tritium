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

/// Virtual arch floor for the IMMA kernel. The `mma.m16n8k32` int8 shape it uses
/// is an Ampere-and-later (sm_80+) instruction — it does not exist on the sm_75
/// (Turing) floor the add-only kernel targets — so this kernel gets its OWN PTX
/// target at `compute_80`. PTX is forward-compatible, so this one image JITs up to
/// Ampere/Ada/Hopper/Blackwell (the rest of [`SUPPORTED_SM_ARCHS`]).
const IMMA_MIN_ARCH: &str = "80";

fn main() {
    // Rebuild triggers regardless of feature state: cheap, and correct when the
    // feature is later toggled on.
    println!("cargo:rerun-if-changed=kernels/tq2_0_add.cu");
    // WF-A (v0.30): the IMMA prefill kernel. Its `mma.m16n8k32` int8 shape needs
    // sm_80+, so WF-A emits a SECOND PTX target (compute_80) for it here — it is
    // NOT compiled yet (the placeholder .cu only declares the entry point). The
    // rerun trigger is wired now so toggling it on later rebuilds correctly.
    println!("cargo:rerun-if-changed=kernels/tq2_0_imma.cu");
    // v0.3.1 (ADR 0013): the device-resident M=1 decode kernels (rmsnorm, rope,
    // attention, …) — compiled with `--fmad=false` so they bit-match the host f32
    // ops (multiply-then-add, not a fused `fma`).
    println!("cargo:rerun-if-changed=kernels/decode.cu");
    // v0.50 (ADR 0007): f32 training backward kernels (gA/gW/gs for the ternary
    // matmul). Same `--fmad=false` host-bit-match discipline as the decode kernels.
    println!("cargo:rerun-if-changed=kernels/train_grad.cu");
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

    // The add-only kernel: PTX for the lowest supported virtual arch; the driver
    // JITs it up to the device's actual SM at load time (one PTX covers sm_75..90).
    let add_min_arch = SUPPORTED_SM_ARCHS
        .first()
        .expect("SUPPORTED_SM_ARCHS is never empty");
    compile_ptx(
        &nvcc,
        Path::new("kernels/tq2_0_add.cu"),
        &out_dir.join("tq2_0_add.ptx"),
        add_min_arch,
        &[],
    );

    // The IMMA prefill kernel: its `mma.m16n8k32` int8 shape needs sm_80+, so it
    // gets a SECOND PTX target at compute_80 (the sm_75 floor cannot assemble it).
    // PTX is forward-compatible, so this image still JITs up to Ada/Hopper/etc.
    compile_ptx(
        &nvcc,
        Path::new("kernels/tq2_0_imma.cu"),
        &out_dir.join("tq2_0_imma.ptx"),
        IMMA_MIN_ARCH,
        &[],
    );

    // The device-resident M=1 decode kernels (v0.3.1). `--fmad=false` forbids the
    // compiler from fusing `a*b+c` into one rounded `fma`, so these reproduce the
    // host f32 ops bit-for-bit (the kernels also use `__fmul_rn`/`__fadd_rn`
    // explicitly; the flag is defence-in-depth for future kernels). compute_75 floor.
    compile_ptx(
        &nvcc,
        Path::new("kernels/decode.cu"),
        &out_dir.join("decode.ptx"),
        add_min_arch,
        &["--fmad=false"],
    );

    // The v0.50 training backward kernels (f32, ADR 0007). `--fmad=false` so the
    // multiply/add rounding matches the host CPU vjp oracle bit-for-bit; compute_75
    // floor (plain f32, no tensor cores).
    compile_ptx(
        &nvcc,
        Path::new("kernels/train_grad.cu"),
        &out_dir.join("train_grad.ptx"),
        add_min_arch,
        &["--fmad=false"],
    );
}

/// Compile a single `.cu` source to virtual PTX (no SASS) for `arch`, emitting it
/// at `ptx_path`. Panics with an actionable message if nvcc is missing the source,
/// fails, or silently produces nothing.
fn compile_ptx(nvcc: &Path, src: &Path, ptx_path: &Path, arch: &str, extra: &[&str]) {
    let mut cmd = Command::new(nvcc);
    cmd.arg("-ptx")
        .arg(src)
        .arg("-o")
        .arg(ptx_path)
        // Virtual-only target: `arch=compute_XX,code=compute_XX` emits PTX (no
        // SASS), which the runtime driver recompiles for the present GPU.
        .arg("-gencode")
        .arg(format!("arch=compute_{arch},code=compute_{arch}"))
        // -O3 keeps the kernel tight; correctness is unaffected.
        .arg("-O3");
    // Per-kernel extra flags (e.g. `--fmad=false` for the bit-matching decode kernels).
    for flag in extra {
        cmd.arg(flag);
    }

    let status = cmd.status().unwrap_or_else(|e| {
        panic!(
            "tritium-cuda: failed to invoke nvcc ({}): {e}",
            nvcc.display()
        )
    });
    assert!(
        status.success(),
        "tritium-cuda: nvcc failed to compile {} (exit {status})",
        src.display()
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
