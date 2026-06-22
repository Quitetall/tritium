#!/usr/bin/env bash
# gpu_session.sh — turnkey rented-GPU session for the v0.60 ≥2-GPU wall (plans 0017–0018).
#
#   clone the repo  →  ./scripts/gpu_session.sh  →  collect ./gpu-results/  →  tag v0.6.0
#
# VALIDATED on 2×A100-SXM4-80GB (production mode) 2026-06-22: 0017 wire-correctness + 0018 FSDP
# loss-parity (world=2, max|Δloss|=4.5e-8) green; full CUDA suite 51/51 on Ampere; single-GPU
# memcheck clean. The hardening below is exactly what that run needed on a fresh base image.
#
# Designed to survive a FRESH / minimal / virtualized GPU box. It handles the env gotchas that
# silently burn GPU hours:
#   • cudarc DLOPENS libnccl/libnvrtc by their UNVERSIONED soname (libnccl.so), but most images ship
#     only the versioned file (libnccl.so.2 from pip-nvidia / torch). We auto-SHIM an unversioned
#     symlink onto LD_LIBRARY_PATH so the runtime dlopen succeeds. (The IMMA JIT path needs libnvrtc.)
#   • cudarc's NCCL bindings are ABI-stable across NCCL 2.x — NCCL 2.28 runs fine against the 2.30
#     pin (validated). So we DO NOT hard-fail on NCCL < 2.30; we only require a loadable libnccl.
#   • the wall needs a PRODUCTION-mode multi-GPU box (custom CUDA kernels + multi-GPU NCCL). A
#     prototyping/virtualized single-GPU mode cannot clear it (the *_multi_gpu gates self-skip).
#
# Env knobs:
#   INSTALL_TOOLCHAIN=1        apt-install rust + cuda-nvcc/cudart/nvrtc if missing (bare base images)
#   SKIP_SANITIZER=1           skip the compute-sanitizer warm-up (slow)
#   ALLOW_VERSION_MISMATCH=1   proceed past a CUDA-MAJOR mismatch (you flipped the cudarc cuda-XXXX pin)
#   GATE_TIMEOUT=900           per-stage timeout seconds (default 900)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
RESULTS="$REPO_ROOT/gpu-results"
mkdir -p "$RESULTS"
SUMMARY="$RESULTS/summary.txt"
: > "$SUMMARY"
GATE_TIMEOUT="${GATE_TIMEOUT:-900}"
SHIM_DIR="$RESULTS/.libshim"; mkdir -p "$SHIM_DIR"

log()  { echo "[gpu_session] $*" | tee -a "$SUMMARY"; }
sec()  { echo "" | tee -a "$SUMMARY"; echo "==== $* ====" | tee -a "$SUMMARY"; }
fail() { echo "[gpu_session] FATAL: $*" | tee -a "$SUMMARY"; exit 1; }

# Shim the UNVERSIONED soname cudarc dlopens (libnccl.so / libnvrtc.so) → the versioned file that
# actually ships, onto LD_LIBRARY_PATH. Searches pip-nvidia dirs, the CUDA tree, and system libdirs.
shim_lib() {  # $1 = soname stem (libnccl / libnvrtc)
  local stem="$1" found
  found="$(find /usr/local/lib/python3*/dist-packages/nvidia /usr/local/cuda* \
                /usr/lib /usr/lib/x86_64-linux-gnu -name "${stem}.so.*" 2>/dev/null \
           | grep -viE 'stub|builtins' | head -1)"
  if [ -n "$found" ]; then
    ln -sf "$found" "$SHIM_DIR/${stem}.so"
    log "shimmed ${stem}.so -> $found"
  else
    log "WARNING: ${stem}.so* not found anywhere — cudarc's dlopen of '$stem' will fail at runtime."
  fi
}

# ───────────────────────────── 1. preflight ─────────────────────────────
sec "1. PREFLIGHT"

# Optional: bootstrap a bare base image (no rust / no nvcc — common on minimal GPU rentals).
if [ "${INSTALL_TOOLCHAIN:-0}" = "1" ]; then
  command -v rustc >/dev/null 2>&1 || {
    log "installing rust (stable, minimal)..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
  }
  # shellcheck disable=SC1091
  [ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
  if ! command -v nvcc >/dev/null 2>&1 && ! ls /usr/local/cuda*/bin/nvcc >/dev/null 2>&1; then
    CM="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]*\)\..*/\1/p' | head -1)"; CM="${CM:-13}"
    log "installing cuda-nvcc/cudart/nvrtc for CUDA ${CM} (driver-reported)..."
    export DEBIAN_FRONTEND=noninteractive
    curl -fsSL -o /tmp/ck.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
      && sudo dpkg -i /tmp/ck.deb && sudo apt-get update -q \
      && sudo apt-get install -y -q build-essential "cuda-nvcc-${CM}-0" "cuda-cudart-dev-${CM}-0" "cuda-nvrtc-${CM}-0" \
      || log "WARNING: toolchain auto-install failed — install a CUDA devel toolkit manually."
  fi
fi
# shellcheck disable=SC1091
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

# Locate nvcc (PATH or the CUDA tree) — build.rs compiles PTX with it.
NVCC="$(command -v nvcc 2>/dev/null || ls /usr/local/cuda*/bin/nvcc 2>/dev/null | sort -V | tail -1)"
[ -n "$NVCC" ] || fail "nvcc not found. Re-run with INSTALL_TOOLCHAIN=1 (bare base image) or install a CUDA *devel* toolkit."
export PATH="$(dirname "$NVCC"):$PATH"; export CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
CUDA_VER="$("$NVCC" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
CUDA_MAJOR="${CUDA_VER%%.*}"
log "CUDA toolkit: ${CUDA_VER:-UNKNOWN} (nvcc: $NVCC)"
if [ "$CUDA_MAJOR" != "13" ] && [ "${ALLOW_VERSION_MISMATCH:-0}" != "1" ]; then
  log "MISMATCH: cudarc is pinned to CUDA 13.x (cuda-13020); this box is CUDA ${CUDA_VER}."
  log "FALLBACK: in crates/tritium-cuda/Cargo.toml set the cudarc \"cuda-13020\" feature to the highest"
  log "          cuda-${CUDA_MAJOR}0XX pin <= your toolkit (pins are sparse — for CUDA 12: 12000/.../12060/12080/12090,"
  log "          NO cuda-12070). Then re-run with ALLOW_VERSION_MISMATCH=1."
  fail "CUDA major != 13 (apply the fallback, then set ALLOW_VERSION_MISMATCH=1)."
fi

command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found — no NVIDIA driver?"
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
log "GPUs: $GPU_COUNT"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader | tee -a "$SUMMARY"
nvidia-smi topo -m 2>/dev/null | tee -a "$SUMMARY" || log "  (topo -m unavailable — normal under containerized/virtualized GPUs; NCCL self-detects)"
if [ "$GPU_COUNT" -lt 2 ]; then
  log "WARNING: <2 GPUs — the 0017/0018 multi-GPU gates will SELF-SKIP (no wall cleared)."
  log "         NOTE: the box must be PRODUCTION mode (multi-GPU + custom CUDA kernels), not prototyping."
fi

# Shim the dlopen libs (the load-bearing fix) and export LD_LIBRARY_PATH for BUILD + RUNTIME.
shim_lib libnccl
shim_lib libnvrtc
export LD_LIBRARY_PATH="$SHIM_DIR:${LD_LIBRARY_PATH:-}"
log "LD_LIBRARY_PATH primed with the unversioned-.so shim dir: $SHIM_DIR"

# NCCL version — informational only. cudarc's 2.30 bindings are ABI-stable across NCCL 2.x (2.28
# validated on the A100 run); we only HARD-fail if NCCL is missing or below cudarc's floor (2.18).
NCCL_INT="$(python3 - <<'PY' 2>/dev/null
import ctypes
for name in ("libnccl.so","libnccl.so.2"):
    try:
        n = ctypes.CDLL(name); v = ctypes.c_int(); n.ncclGetVersion(ctypes.byref(v)); print(v.value); break
    except Exception: pass
PY
)"
if [ -n "$NCCL_INT" ]; then
  NCCL_MINOR=$(( (NCCL_INT / 100) % 100 )); NCCL_PATCH=$(( NCCL_INT % 100 ))
  log "NCCL: 2.${NCCL_MINOR}.${NCCL_PATCH} (cudarc 2.30 bindings are ABI-compatible across 2.x; runs as-is)"
  if [ "$NCCL_MINOR" -lt 18 ] && [ "${ALLOW_VERSION_MISMATCH:-0}" != "1" ]; then
    fail "NCCL 2.${NCCL_MINOR} is below cudarc's lowest binding (nccl-02018). Upgrade NCCL to >= 2.18."
  fi
else
  log "WARNING: could not read NCCL version (libnccl not loadable yet) — relying on the shim above."
fi

command -v rustc >/dev/null 2>&1 || fail "rustc not found — install rustup (or re-run with INSTALL_TOOLCHAIN=1)."
log "rustc: $(rustc --version)"

# ───────────────────────────── 2. build ─────────────────────────────
sec "2. BUILD (cuda + nccl, release) + compile the gate tests"
cargo build -p tritium-cuda --features cuda,nccl --release 2>&1 | tee "$RESULTS/build.log" | tail -3
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "build failed — see $RESULTS/build.log"
# Compile (don't run) the nccl gate tests now, so a test-only LINK error surfaces HERE.
cargo test -p tritium-cuda --features nccl --lib nccl:: --no-run 2>&1 | tee "$RESULTS/build_tests.log" | tail -3
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "gate-test compile/link failed — see $RESULTS/build_tests.log"
log "build: OK"

# ───────────────────────── 3. 1-GPU warm-up ─────────────────────────
sec "3. 1-GPU RE-VERIFY (datacenter arch) — full CUDA suite + compute-sanitizer"
timeout "$GATE_TIMEOUT" cargo test -p tritium-cuda --features cuda --release -- --test-threads=1 2>&1 \
  | tee "$RESULTS/cuda_suite.log" | grep -E "test result|FAILED" | tee -a "$SUMMARY"
CUDA_SUITE_EXIT="${PIPESTATUS[0]}"
[ "$CUDA_SUITE_EXIT" -eq 0 ] || log "WARNING: 1-GPU suite exit $CUDA_SUITE_EXIT (124=timeout) — see $RESULTS/cuda_suite.log (IMMA/JIT tests need libnvrtc — shimmed above)."
if [ "${SKIP_SANITIZER:-0}" != "1" ] && command -v compute-sanitizer >/dev/null 2>&1; then
  log "compute-sanitizer memcheck over the core kernels (best-effort)..."
  timeout "$GATE_TIMEOUT" compute-sanitizer --tool memcheck --target-processes all \
    cargo test -p tritium-cuda --features cuda --release -- --test-threads=1 \
    imma mpgemm_device salt backward rmsnorm rope gqa relu2 residual pretrain 2>&1 \
    | tee "$RESULTS/sanitizer.log" | grep -iE "ERROR SUMMARY|test result|leaked [1-9]" | tee -a "$SUMMARY" || true
else
  log "compute-sanitizer: skipped (SKIP_SANITIZER or tool absent)"
fi

# ──────────────────── 4–5. the ≥2-GPU wall gates ────────────────────
sec "4-5. NCCL WIRE-CORRECTNESS (0017) + FSDP LOSS-PARITY (0018)"
# timeout-wrapped (a single failed rank deadlocks NCCL forever); serialized so per-test 2-rank
# spawns don't collide on the GPUs.
timeout "$GATE_TIMEOUT" cargo test -p tritium-cuda --features nccl --lib nccl:: -- --nocapture --test-threads=1 2>&1 \
  | tee "$RESULTS/nccl_gates.log" \
  | grep -E "test nccl|test result|world=|skip|max .*loss|FAILED|panicked" | tee -a "$SUMMARY"
GATE_EXIT="${PIPESTATUS[0]}"
log "nccl gate exit: $GATE_EXIT (0=ok, 124=TIMEOUT/hang)"

# ───────────────────────────── 6. verdict ─────────────────────────────
sec "6. VERDICT"
GATES_LOG="$RESULTS/nccl_gates.log"
# NOTE: --nocapture interleaves the fsdp test's printed line between "... " and "ok", so we key the
# PASS off `test result: ok` (ALL nccl tests passed) + the all_reduce ok line + a real world>=2 run,
# rather than a fragile per-test "name ... ok" single-line match.
if [ "$GATE_EXIT" -eq 124 ]; then
  log "FAIL: the gate run TIMED OUT ($GATE_TIMEOUT s) — likely a deadlocked rank. Inspect $GATES_LOG."
elif [ "$GATE_EXIT" -ne 0 ]; then
  log "FAIL: the gate run exited $GATE_EXIT — inspect $GATES_LOG (FAILED/panicked/error)."
elif grep -q "skip nccl_all_reduce" "$GATES_LOG"; then
  log "NOT CLEARED: the multi-GPU gates SELF-SKIPPED — <2 visible GPUs (or prototyping mode). Rent a ≥2-GPU PRODUCTION box."
elif grep -q "test result: ok" "$GATES_LOG" \
     && grep -q "nccl_all_reduce_matches_sum_reference_multi_gpu ... ok" "$GATES_LOG" \
     && grep -qE "nccl_fsdp_loss_parity world=[2-9]" "$GATES_LOG"; then
  log "PASS: 0017 wire-correctness + 0018 loss-parity GREEN on $GPU_COUNT GPUs (real multi-GPU run confirmed)."
  log "BEFORE TAGGING: read the printed 'max |Δloss|', tighten nccl_fsdp_loss_parity's ABS/REL_TOL to ~10x"
  log "      it (AND them), re-run to confirm still-green, THEN: git tag -a v0.6.0 && push."
else
  log "FAIL: no positive multi-GPU evidence in $GATES_LOG (gates may not have run). Do NOT tag."
fi
log "All logs in $RESULTS/. Commit this dir (or rsync it home) BEFORE terminating the box."
