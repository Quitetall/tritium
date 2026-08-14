#!/usr/bin/env bash
# Build Tritium's CUDA abi3 wheel inside an immutable manylinux_2_28 image.
set -euo pipefail

IMAGE="quay.io/pypa/manylinux_2_28_x86_64@sha256:a61875a2f84cab7df8de222ff12cabc08ff86eb4ad402ac90ba7bdaed9600cca"
RUST_TOOLCHAIN="1.89.0"
MATURIN_VERSION="1.10.2"
PLATFORM_TAG="manylinux_2_28_x86_64"
STATIC_CXX_LINKER="/io/scripts/manylinux-static-cxx-linker.sh"
NVCC_CCBIN="/opt/rh/gcc-toolset-14/root/usr/bin/g++"

if [[ "${1:-}" == "--print-contract" ]]; then
  printf '{"image":"%s","rust":"%s","maturin":"%s","linker":"%s","static_cxx":"%s","nvcc_ccbin":"%s","platform":"%s"}\n' \
    "$IMAGE" "$RUST_TOOLCHAIN" "$MATURIN_VERSION" "$STATIC_CXX_LINKER" \
    "/opt/rh/gcc-toolset-14/root/usr/lib/gcc/x86_64-redhat-linux/14/libstdc++.a" \
    "$NVCC_CCBIN" \
    "$PLATFORM_TAG"
  exit 0
fi

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [OUTPUT_DIR] | --print-contract" >&2
  exit 2
fi

ROOT="$(git rev-parse --show-toplevel)"
if [[ -n "$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "CUDA manylinux release builds require a clean Git worktree" >&2
  exit 1
fi
SOURCE_REVISION="$(git -C "$ROOT" rev-parse HEAD)"
if [[ ! "$SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Git HEAD is not a full lowercase object ID" >&2
  exit 1
fi

OUTPUT="${1:-$ROOT/dist-cuda}"
mkdir -p "$OUTPUT"
OUTPUT="$(cd "$OUTPUT" && pwd -P)"
if [[ -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "output directory must be empty: $OUTPUT" >&2
  exit 1
fi

NVCC="$(command -v nvcc || true)"
if [[ -z "$NVCC" || ! -x "$NVCC" ]]; then
  echo "nvcc is required" >&2
  exit 1
fi
CUDA_HOME="$(cd "$(dirname "$NVCC")/.." && pwd -P)"
if ! "$NVCC" --version | grep -Eq 'release 13\.'; then
  echo "CUDA 13.x is required" >&2
  exit 1
fi
if ! command -v docker >/dev/null; then
  echo "Docker is required" >&2
  exit 1
fi

RUST_SYSROOT="$(rustup run "$RUST_TOOLCHAIN" rustc --print sysroot)"
if [[ ! -x "$RUST_SYSROOT/bin/rustc" || ! -x "$RUST_SYSROOT/bin/cargo" ]]; then
  echo "Rust $RUST_TOOLCHAIN toolchain is required" >&2
  exit 1
fi

CACHE="${TRITIUM_CUDA_MANYLINUX_CACHE:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}/tritium-cuda-manylinux-cache}"
mkdir -p "$CACHE/cargo-home" "$CACHE/target"

docker run --rm \
  --volume "$ROOT:/io:ro" \
  --volume "$OUTPUT:/out" \
  --volume "$CACHE:/cargo" \
  --volume "$RUST_SYSROOT:/opt/rust:ro" \
  --volume "$CUDA_HOME:$CUDA_HOME:ro" \
  --env "CUDA_HOME=$CUDA_HOME" \
  --env "CARGO_HOME=/cargo/cargo-home" \
  --env "CARGO_TARGET_DIR=/cargo/target" \
  --env "HOST_UID=$(id -u)" \
  --env "HOST_GID=$(id -g)" \
  --env "TRITIUM_SOURCE_ID=source-git:$SOURCE_REVISION" \
  --env "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=$STATIC_CXX_LINKER" \
  --env "NVCC_CCBIN=$NVCC_CCBIN" \
  --env "PATH=/opt/rust/bin:$CUDA_HOME/bin:/opt/python/cp313-cp313/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  "$IMAGE" bash -c "
    set -euo pipefail
    /opt/python/cp313-cp313/bin/python -m pip install \
      --disable-pip-version-check --root-user-action=ignore \
      'maturin==$MATURIN_VERSION' >/dev/null
    maturin build --locked --release --features cuda --manylinux 2_28 \
      --manifest-path /io/crates/tritium-py/Cargo.toml --out /out
    shopt -s nullglob
    wheels=(/out/*.whl)
    [[ \${#wheels[@]} -eq 1 ]]
    [[ \${wheels[0]} == *-'$PLATFORM_TAG'.whl ]]
    auditwheel show \"\${wheels[0]}\"
    chown \"\$HOST_UID:\$HOST_GID\" \"\${wheels[0]}\"
  "

python "$ROOT/scripts/verify-wheel.py" "$OUTPUT" \
  --require-platform-tag "$PLATFORM_TAG"
