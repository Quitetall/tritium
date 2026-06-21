#!/usr/bin/env bash
# gpu_session.sh — turnkey rented-GPU session for the v0.60 ≥2-GPU wall (plans 0017–0018).
#
#   clone the repo  →  ./scripts/gpu_session.sh  →  collect ./gpu-results/  →  tag v0.60.0
#
# Preflight catches the env mismatches that silently waste GPU hours (CUDA major, NCCL version, GPU
# count) BEFORE building, and prints the EXACT one-line cudarc feature pin to flip if needed. Every
# gate run is `timeout`-wrapped (a single failed rank deadlocks NCCL forever otherwise) and the final
# verdict requires POSITIVE evidence — a real multi-GPU "... ok" line — so zero-tests-ran / a hang /
# a swallowed error can never read as PASS. All output → ./gpu-results/. Safe to re-run.
#
# Env knobs:
#   SKIP_SANITIZER=1            skip the compute-sanitizer warm-up (slow)
#   ALLOW_VERSION_MISMATCH=1    proceed past a CUDA/NCCL preflight mismatch (you applied the fallback)
#   GATE_TIMEOUT=900            per-stage timeout seconds (default 900)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
RESULTS="$REPO_ROOT/gpu-results"
mkdir -p "$RESULTS"
SUMMARY="$RESULTS/summary.txt"
: > "$SUMMARY"
GATE_TIMEOUT="${GATE_TIMEOUT:-900}"

log()  { echo "[gpu_session] $*" | tee -a "$SUMMARY"; }
sec()  { echo "" | tee -a "$SUMMARY"; echo "==== $* ====" | tee -a "$SUMMARY"; }
fail() { echo "[gpu_session] FATAL: $*" | tee -a "$SUMMARY"; exit 1; }

# ───────────────────────────── 1. preflight ─────────────────────────────
sec "1. PREFLIGHT"

command -v nvcc >/dev/null 2>&1 || fail "nvcc not found — need a CUDA *devel* image (build.rs compiles PTX). Set CUDA_PATH/CUDA_HOME."
CUDA_VER="$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
CUDA_MAJOR="${CUDA_VER%%.*}"
log "CUDA toolkit: ${CUDA_VER:-UNKNOWN}"
if [ "$CUDA_MAJOR" != "13" ] && [ "${ALLOW_VERSION_MISMATCH:-0}" != "1" ]; then
  log "MISMATCH: cudarc is pinned to CUDA 13.x (cuda-13020); this box is CUDA ${CUDA_VER}."
  log "FALLBACK: in crates/tritium-cuda/Cargo.toml change the cudarc \"cuda-13020\" feature to the"
  log "          highest cuda-${CUDA_MAJOR}0XX pin <= your toolkit. The pins are SPARSE — for CUDA 12"
  log "          they are 12000/12010/.../12060/12080/12090 (NO cuda-12070: use cuda-12060 for 12.6/12.7)."
  log "          Then re-run with ALLOW_VERSION_MISMATCH=1."
  fail "CUDA major != 13 (apply the fallback, then set ALLOW_VERSION_MISMATCH=1)."
fi

command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found — no NVIDIA driver?"
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
log "GPUs: $GPU_COUNT"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader | tee -a "$SUMMARY"
log "interconnect (nvidia-smi topo -m):"
nvidia-smi topo -m 2>/dev/null | tee -a "$SUMMARY" || log "  (topo unavailable)"
[ "$GPU_COUNT" -ge 2 ] || log "WARNING: <2 GPUs — the 0017/0018 multi-GPU gates will SELF-SKIP (no wall cleared)."

# NCCL version — read it RELIABLY via ncclGetVersion (the .so has no clean version string to grep).
# Returns an int like 23007 == 2.30.7; minor = (v/100)%100. cudarc binds NCCL 2.30 (nccl-02030).
NCCL_INT="$(python3 - <<'PY' 2>/dev/null
import ctypes
try:
    n = ctypes.CDLL("libnccl.so.2"); v = ctypes.c_int()
    n.ncclGetVersion(ctypes.byref(v)); print(v.value)
except Exception:
    pass
PY
)"
if [ -n "$NCCL_INT" ]; then
  NCCL_MINOR=$(( (NCCL_INT / 100) % 100 ))
  NCCL_PATCH=$(( NCCL_INT % 100 ))
  log "NCCL: 2.${NCCL_MINOR}.${NCCL_PATCH} (ncclGetVersion=${NCCL_INT})"
  if [ "$NCCL_MINOR" -lt 30 ] && [ "${ALLOW_VERSION_MISMATCH:-0}" != "1" ]; then
    if [ "$NCCL_MINOR" -lt 18 ]; then
      log "MISMATCH: NCCL 2.${NCCL_MINOR} is below cudarc's lowest binding (nccl-02018). Upgrade NCCL to >= 2.18."
    elif [ "$NCCL_MINOR" -eq 23 ]; then
      log "MISMATCH: NCCL 2.23 — cudarc has NO nccl-02023 pin (gap). Use cudarc/nccl-02022."
    else
      PIN="$(printf 'cudarc/nccl-02%03d' "$NCCL_MINOR")"
      log "MISMATCH: cudarc binds NCCL 2.30; this box has NCCL 2.${NCCL_MINOR}."
      log "FALLBACK: in crates/tritium-cuda/Cargo.toml set the nccl feature's cudarc pin to \"$PIN\"."
    fi
    log "          Then re-run with ALLOW_VERSION_MISMATCH=1."
    fail "NCCL < 2.30 (apply the fallback, then set ALLOW_VERSION_MISMATCH=1)."
  fi
elif [ "${ALLOW_VERSION_MISMATCH:-0}" != "1" ]; then
  log "Could not read the NCCL version (no python3 / ctypes / libnccl). cudarc needs NCCL >= 2.30."
  fail "NCCL version unverified — confirm libnccl >= 2.30 then re-run with ALLOW_VERSION_MISMATCH=1."
fi

command -v rustc >/dev/null 2>&1 || fail "rustc not found — install rustup + 1.89 toolchain."
log "rustc: $(rustc --version)"

# ───────────────────────────── 2. build ─────────────────────────────
sec "2. BUILD (cuda + nccl, release) + compile the gate tests"
cargo build -p tritium-cuda --features cuda,nccl --release 2>&1 | tee "$RESULTS/build.log" | tail -3
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "build failed — see $RESULTS/build.log"
# Compile (don't run) the nccl gate tests now, so a test-only LINK error surfaces HERE, not as a
# phantom mid-session failure.
cargo test -p tritium-cuda --features nccl --lib nccl:: --no-run 2>&1 | tee "$RESULTS/build_tests.log" | tail -3
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "gate-test compile/link failed — see $RESULTS/build_tests.log"
log "build: OK"

# ───────────────────────── 3. 1-GPU warm-up ─────────────────────────
sec "3. 1-GPU RE-VERIFY (datacenter arch)"
timeout "$GATE_TIMEOUT" cargo test -p tritium-cuda --features cuda 2>&1 | tee "$RESULTS/cuda_suite.log" | grep -E "test result|FAILED" | tee -a "$SUMMARY"
CUDA_SUITE_EXIT="${PIPESTATUS[0]}"
[ "$CUDA_SUITE_EXIT" -eq 0 ] || log "WARNING: 1-GPU suite exit $CUDA_SUITE_EXIT (124=timeout) — see $RESULTS/cuda_suite.log"
if [ "${SKIP_SANITIZER:-0}" != "1" ] && command -v compute-sanitizer >/dev/null 2>&1; then
  log "compute-sanitizer memcheck on the pretrain smoke (best-effort)..."
  timeout "$GATE_TIMEOUT" compute-sanitizer --tool memcheck --error-exitcode 1 \
    cargo test -p tritium-cuda --features cuda --release pretrain_smoke 2>&1 \
    | tee "$RESULTS/sanitizer.log" | grep -iE "ERROR SUMMARY|0 errors|leaked" | tee -a "$SUMMARY" || true
else
  log "compute-sanitizer: skipped (SKIP_SANITIZER or tool absent)"
fi

# ──────────────────── 4–5. the ≥2-GPU wall gates ────────────────────
sec "4-5. NCCL WIRE-CORRECTNESS (0017) + FSDP LOSS-PARITY (0018)"
# timeout-wrapped: a single rank failing init/collective deadlocks NCCL forever otherwise.
timeout "$GATE_TIMEOUT" cargo test -p tritium-cuda --features nccl --lib nccl:: -- --nocapture 2>&1 \
  | tee "$RESULTS/nccl_gates.log" \
  | grep -E "test nccl|test result|world=|skip|max .*loss|FAILED|panicked" | tee -a "$SUMMARY"
GATE_EXIT="${PIPESTATUS[0]}"
log "nccl gate exit: $GATE_EXIT (0=ok, 124=TIMEOUT/hang)"

# ───────────────────────────── 6. verdict ─────────────────────────────
sec "6. VERDICT"
GATES_LOG="$RESULTS/nccl_gates.log"
if [ "$GATE_EXIT" -eq 124 ]; then
  log "FAIL: the gate run TIMED OUT ($GATE_TIMEOUT s) — likely a deadlocked rank. Inspect $GATES_LOG."
elif [ "$GATE_EXIT" -ne 0 ]; then
  log "FAIL: the gate run exited $GATE_EXIT — inspect $GATES_LOG (FAILED/panicked/error)."
elif grep -q "skip nccl_all_reduce" "$GATES_LOG"; then
  log "NOT CLEARED: the multi-GPU gates SELF-SKIPPED — this box has <2 visible GPUs. Rent a ≥2-GPU box."
elif grep -q "test result: ok" "$GATES_LOG" \
     && grep -q "nccl_all_reduce_matches_sum_reference_multi_gpu ... ok" "$GATES_LOG" \
     && grep -q "nccl_fsdp_loss_parity ... ok" "$GATES_LOG" \
     && grep -qE "nccl_fsdp_loss_parity world=[2-9]" "$GATES_LOG"; then
  log "PASS: 0017 wire-correctness + 0018 loss-parity GREEN on $GPU_COUNT GPUs (real multi-GPU run confirmed)."
  log "BEFORE TAGGING: read the printed 'max |Δloss|', tighten nccl_fsdp_loss_parity's ABS/REL_TOL to ~10x"
  log "      it (AND them), re-run this script to confirm still-green, THEN: git tag -a v0.60.0 && push."
else
  log "FAIL: no positive multi-GPU evidence in $GATES_LOG (gates may not have run). Do NOT tag."
fi
log "All logs in $RESULTS/. Commit this dir (or rsync it home) BEFORE terminating the box."
