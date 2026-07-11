#!/usr/bin/env bash
# ADR 0022 consolidation proof: extract per-kernel SASS from a PTX file.
#   tools/sass_diff.sh <decode.ptx> <outdir> [sm_arch]
# Emits <outdir>/<kernel>.sass per __global__ symbol. Diff two outdirs to
# prove a codec-template refactor left the instantiations byte-identical.
set -euo pipefail
PTX="$1"; OUT="$2"; ARCH="${3:-sm_89}"
mkdir -p "$OUT"
CUBIN="$OUT/all.cubin"
ptxas -arch="$ARCH" -O3 "$PTX" -o "$CUBIN"
# nvdisasm with per-function output; split on function headers.
nvdisasm -c "$CUBIN" > "$OUT/all.sass"
python3 - "$OUT" <<'PY'
import re, sys, os
out = sys.argv[1]
text = open(os.path.join(out, "all.sass")).read()
# Sections start with ".text.<name>:" or "Function : <name>"
parts = re.split(r"\n(?=//-+ \.text\.)", text)
n = 0
for p in parts:
    m = re.search(r"//-+ \.text\.(\w+) ", p)
    if not m:
        continue
    open(os.path.join(out, m.group(1) + ".sass"), "w").write(p)
    n += 1
print(f"{n} kernels extracted to {out}")
PY
