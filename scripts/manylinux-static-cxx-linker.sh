#!/usr/bin/env bash
# Link Rust abi3 wheels with the image's static C++ runtime.
#
# ort-sys emits `-lstdc++` after the downloaded ONNX Runtime archive. The
# manylinux image's system libstdc++ is too old for that archive, while the
# GCC-toolset static archives contain the required C++11/filesystem symbols
# without adding a newer shared-library dependency to the wheel.
set -euo pipefail

readonly CXX=/opt/rh/gcc-toolset-14/root/usr/bin/g++
readonly STDCXX=/opt/rh/gcc-toolset-14/root/usr/lib/gcc/x86_64-redhat-linux/14/libstdc++.a
readonly STDCXXFS=/opt/rh/gcc-toolset-14/root/usr/lib/gcc/x86_64-redhat-linux/14/libstdc++fs.a

[[ -x "$CXX" && -f "$STDCXX" && -f "$STDCXXFS" ]] || {
  echo "manylinux static C++ runtime is unavailable" >&2
  exit 1
}

args=()
skip_next=0
needs_stdcxx=0
for arg in "$@"; do
  if ((skip_next)); then
    if [[ "$arg" == "stdc++" || "$arg" == "stdc++fs" ]]; then
      needs_stdcxx=1
    else
      args+=("$arg")
    fi
    skip_next=0
    continue
  fi
  case "$arg" in
    -lstdc++|-lstdc++fs) needs_stdcxx=1 ;;
    -l) skip_next=1 ;;
    *) args+=("$arg") ;;
  esac
done

# Keep archives after all object/library inputs. ORT's static archive refers to
# filesystem and stream symbols, so placing these before ORT would leave them
# unresolved under the linker's one-pass archive scan.
if ((needs_stdcxx)); then
  args+=(-Wl,--start-group "$STDCXX" "$STDCXXFS" -Wl,--end-group)
fi

exec "$CXX" -static-libgcc "${args[@]}"
