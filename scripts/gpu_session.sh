#!/usr/bin/env bash
# gpu_session.sh — turnkey rented-GPU session for the v0.60 ≥2-GPU wall (plans 0017–0018).
#
#   clone the repo  →  ./scripts/gpu_session.sh  →  collect ./gpu-results/  →  tag v0.60.0
#
# Preflight catches the env mismatches that silently waste GPU hours (CUDA major, NCCL version, GPU
# count) BEFORE building, and prints the exact one-line fallback if a pin needs flipping. Everything is
# logged to ./gpu-results/ so nothing is lost when the box is terminated. Designed to be safe to re-run.
#
# Env knobs:
#   SKIP_SANITIZER=1   skip the compute-sanitizer warm-up (it is slow)
#   ALLOW_VERSION_MISMATCH=1   proceed past a CUDA/NCCL preflight mismatch (you applied the fallback)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
RESULTS="$REPO_ROOT/gpu-results"
mkdir -p "$RESULTS"
SUMMARY="$RESULTS/summary.txt"
: > "$SUMMARY"

log()  { echo "[gpu_session] $*" | tee -a "$SUMMARY"; }
sec()  { echo "" | tee -a "$SUMMARY"; echo "==== $* ====" | tee -a "$SUMMARY"; }
fail() { echo "[gpu_session] FATAL: $*" | tee -a "$SUMMARY"; exit 1; }

# ───────────────────────────── 1. preflight ─────────────────────────────
sec "1. PREFLIGHT"

command -v nvcc >/dev/null 2>&1 || fail "nvcc not found — need a CUDA *devel* image (build.rs compiles PTX). Set CUDA_PATH/CUDA_HOME."
CUDA_VER="$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
CUDA_MAJOR="${CUDA_VER%%.*}"
log "CUDA toolkit: $CUDA_VER"
if [ "$CUDA_MAJOR" != "13" ] && [ "${ALLOW_VERSION_MISMATCH:-0}" != "1" ]; then
  log "MISMATCH: cudarc is pinned to CUDA 13.x (cuda-13020). This box is CUDA $CUDA_VER."
  log "FALLBACK: edit crates/tritium-cuda/Cargo.toml — change the cudarc feature \"cuda-13020\""
  log "          to the matching \"cuda-${CUDA_MAJOR}0XX\" (e.g. cuda-12060 for 12.6), then re-run"
  log "          with ALLOW_VERSION_MISMATCH=1."
  fail "CUDA major != 13 (set ALLOW_VERSION_MISMATCH=1 after applying the fallback)."
fi

command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found — no NVIDIA driver?"
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
log "GPUs: $GPU_COUNT"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader | tee -a "$SUMMARY"
log "interconnect (nvidia-smi topo -m):"
nvidia-smi topo -m 2>/dev/null | tee -a "$SUMMARY" || log "  (topo unavailable)"
[ "$GPU_COUNT" -ge 2 ] || log "WARNING: <2 GPUs — the 0017/0018 multi-GPU gates will SELF-SKIP (world=1 only)."

# NCCL version (cudarc binds nccl-02030 = NCCL 2.30).
NCCL_SO="$(ldconfig -p 2>/dev/null | sed -n 's/.*=> \(.*libnccl.so.2\)$/\1/p' | head -1)"
if [ -n "$NCCL_SO" ]; then
  NCCL_VER="$(strings "$NCCL_SO" 2>/dev/null | grep -oE '^2\.[0-9]+\.[0-9]+' | head -1)"
  log "NCCL: ${NCCL_VER:-present (version unread)} ($NCCL_SO)"
  if [ -n "$NCCL_VER" ]; then
    NCCL_MINOR="$(echo "$NCCL_VER" | cut -d. -f2)"
    if [ "$NCCL_MINOR" -lt 30 ] && [ "${ALLOW_VERSION_MISMATCH:-0}" != "1" ]; then
      log "MISMATCH: cudarc binds NCCL 2.30 (nccl-02030); this box has NCCL $NCCL_VER."
      log "FALLBACK: in crates/tritium-cuda/Cargo.toml set the nccl feature to"
      log "          \"cudarc/nccl-02${NCCL_MINOR}\" (e.g. nccl-02018), then re-run with ALLOW_VERSION_MISMATCH=1."
      fail "NCCL < 2.30 (set ALLOW_VERSION_MISMATCH=1 after applying the fallback)."
    fi
  fi
else
  log "WARNING: libnccl.so.2 not found via ldconfig — install NCCL or the build will fail to link."
fi

command -v rustc >/dev/null 2>&1 || fail "rustc not found — install rustup + 1.89 toolchain."
log "rustc: $(rustc --version)"

# ───────────────────────────── 2. build ─────────────────────────────
sec "2. BUILD (cuda + nccl, release)"
if cargo build -p tritium-cuda --features cuda,nccl --release 2>&1 | tee "$RESULTS/build.log" | tail -3; then
  log "build: OK"
else
  fail "build failed — see $RESULTS/build.log"
fi

# ───────────────────────── 3. 1-GPU warm-up ─────────────────────────
sec "3. 1-GPU RE-VERIFY (datacenter arch)"
cargo test -p tritium-cuda --features cuda 2>&1 | tee "$RESULTS/cuda_suite.log" | grep -E "test result|FAILED" | tee -a "$SUMMARY" || true
if [ "${SKIP_SANITIZER:-0}" != "1" ] && command -v compute-sanitizer >/dev/null 2>&1; then
  log "compute-sanitizer memcheck on the pretrain smoke (best-effort)..."
  compute-sanitizer --tool memcheck --error-exitcode 1 \
    cargo test -p tritium-cuda --features cuda --release pretrain_smoke 2>&1 \
    | tee "$RESULTS/sanitizer.log" | grep -iE "ERROR SUMMARY|0 errors|leaked" | tee -a "$SUMMARY" || true
else
  log "compute-sanitizer: skipped (SKIP_SANITIZER or tool absent)"
fi

# ──────────────────── 4–5. the ≥2-GPU wall gates ────────────────────
sec "4-5. NCCL WIRE-CORRECTNESS (0017) + FSDP LOSS-PARITY (0018)"
cargo test -p tritium-cuda --features nccl --lib nccl:: -- --nocapture 2>&1 \
  | tee "$RESULTS/nccl_gates.log" \
  | grep -E "test nccl|test result|world=|skip|max .*loss|FAILED|panicked" | tee -a "$SUMMARY" || true

# ───────────────────────────── 6. verdict ─────────────────────────────
sec "6. VERDICT"
if grep -qE "skip nccl_all_reduce" "$RESULTS/nccl_gates.log"; then
  log "The multi-GPU gates SELF-SKIPPED — this box has <2 visible GPUs. v0.60.0 NOT cleared."
elif grep -qE "FAILED|panicked|error\[" "$RESULTS/nccl_gates.log"; then
  log "FAIL: a wall gate failed — inspect $RESULTS/nccl_gates.log before tagging."
else
  log "PASS: 0017 + 0018 green on $GPU_COUNT GPUs. Tighten nccl_fsdp_loss_parity's tolerance to the"
  log "      printed max |Δloss|, then:  git add -A && git commit && git tag -a v0.60.0 && push."
fi
log "All logs in $RESULTS/. Commit this dir (or rsync it home) BEFORE terminating the box."
