#!/usr/bin/env bash
#
# scripts/capstone.sh — CPU fresh-env end-to-end capstone SMOKE for Tritium.
#
# Exercises the SHAPE of the v1.0 capstone pipeline
#   (install/build → infer → SALT-quantize → fine-tune)
# entirely on CPU, deterministically, from a clean checkout. Every stage runs a
# REAL Tritium code path; nothing is mocked. The full-scale capstone (a real
# BitNet b1.58 GGUF + a GPU fine-tune that recovers perplexity/accuracy) needs
# hardware this lane does NOT have — see the "DEFERRED TO HARDWARE" notes printed
# at the end and docs/ROADMAP.md (v1.0 Release / ADR 0012).
#
# What is REAL here vs what is SCAFFOLDED/DEFERRED:
#   (1) BUILD     — REAL: builds the whole workspace + the `tritium` binary.
#   (2) INFER     — REAL parse + REAL runner-load attempt:
#                     • `tritium list-backends` enumerates the registered CPU backend.
#                     • `tritium inspect` parses the committed tiny ternary GGUF
#                       fixture end to end (GGUF v3 reader: version, metadata,
#                       TQ2_0/TQ1_0 tensor table).
#                     • `tritium generate` drives the FULL ModelRunner::load_cpu
#                       path (arg parse → JSON token reader → GGUF parse →
#                       ModelConfig::from_gguf → ModelWeights::load). The committed
#                       fixture is INTENTIONALLY a partial (format-valid, not a
#                       complete model), so load fails CLEANLY at the first missing
#                       weight tensor (non-zero exit, descriptive error, no panic).
#                       This proves the runner/CLI plumbing without faking a model.
#                     • DEFERRED: a forward pass that emits tokens needs a complete
#                       model (every layer's weights). The real BitNet 2B4T GGUF is
#                       multi-GB and CPU decode is ~seconds/token — out of scope for
#                       a fast hosted lane. The real-model path is covered by the
#                       model-gated `generate_on_real_model_runs` test (TRITIUM_RUN_SLOW).
#   (3) SALT      — REAL: `tritium report salt` runs the production SALT quantizer
#                   (tritium_quantize::quantize_tensor) + dequant on a synthetic
#                   fp32 matrix and reports bpw/MSE/RMSE; `tritium quantize` runs
#                   the full safetensors → SALT-bundle (.tslb) pipeline on a
#                   synthetic fp32 safetensors. Both are real algorithms on
#                   synthetic (not random-each-run) inputs.
#   (4) FINE-TUNE — REAL: runs the tritium-train CPU gates — the STE + ternary-matmul
#                   backward gradient check vs finite difference (Gate C, ADR 0007),
#                   the AdamW optimizer steps, and the tiny-transformer end-to-end
#                   gradient tape. This is the actual training engine the GPU
#                   capstone fine-tune is built on.
#                     • DEFERRED: fine-tuning a real model to RECOVER accuracy/PPL
#                       after quantization is the GPU capstone (full-model backprop;
#                       see ADR 0007 / plan 0010 — ~94.6% layerwise convergence on
#                       the 2B4T model, full-model PPL-recovery tracked to v0.60+).
#
# Usage:  scripts/capstone.sh
# Exit:   0 only if every stage's real CPU path succeeds as expected.
#
set -euo pipefail

# --- locate the repo root so the script is runnable from anywhere -------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --- deterministic, self-cleaning scratch dir --------------------------------
WORK="$(mktemp -d "${TMPDIR:-/tmp}/tritium-capstone.XXXXXX")"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

# --- pretty markers ----------------------------------------------------------
step()  { printf '\n=== STEP %s: %s ===\n' "$1" "$2"; }
pass()  { printf 'PASS [%s] %s\n' "$1" "$2"; }
info()  { printf '  - %s\n' "$1"; }
fail()  { printf 'FAIL [%s] %s\n' "$1" "$2" >&2; exit 1; }

# A tiny deterministic fp32 matrix as a flat JSON array (rows*k values).
# Pure-shell generator — no Python needed for the SALT-report input.
emit_matrix_json() {
  local rows="$1" k="$2" out="$3"
  awk -v rows="$rows" -v k="$k" 'BEGIN {
    n = rows * k;
    printf "[";
    for (i = 0; i < n; i++) {
      # deterministic, bounded, mixed-sign values (no RNG): sin-like via a cheap recurrence
      v = sin(i * 0.37) * 0.6 + cos(i * 0.11) * 0.25;
      printf "%s%.6f", (i ? "," : ""), v;
    }
    printf "]";
  }' > "$out"
}

# A tiny deterministic fp32 .safetensors with two 2D weight matrices, written
# with the exact container layout tritium-format's reader expects:
#   8-byte LE u64 header length | JSON header | raw LE f32 payload.
# Uses Python3 (guaranteed on the GitHub ubuntu runners) for the binary payload;
# if absent, the safetensors sub-step is skipped (the SALT *report* above already
# exercises the quantizer), and the script still proves the rest of the pipeline.
emit_safetensors() {
  local out="$1"
  python3 - "$out" <<'PY'
import json, struct, sys, math
# Two 2D fp32 weight matrices so `tritium quantize` finds SALT-able tensors.
tensors = {
    "model.layers.0.mlp.up_proj.weight": (8, 16),
    "model.layers.0.self_attn.q_proj.weight": (16, 16),
}
header = {"__metadata__": {"format": "pt"}}
blob, off = b"", 0
for name, (r, c) in tensors.items():
    vals = [math.sin(i * 0.17) * 0.6 + math.cos(i * 0.05) * 0.2 for i in range(r * c)]
    raw = struct.pack("<%df" % len(vals), *vals)
    header[name] = {"dtype": "F32", "shape": [r, c], "data_offsets": [off, off + len(raw)]}
    blob += raw
    off += len(raw)
hjson = json.dumps(header).encode("utf-8")
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<Q", len(hjson)))
    f.write(hjson)
    f.write(blob)
PY
}

printf '########################################################################\n'
printf '# Tritium CPU capstone SMOKE — install -> infer -> SALT -> fine-tune    #\n'
printf '# All stages run REAL CPU code paths. GPU/real-model work is DEFERRED   #\n'
printf '# (documented inline + summarized at the end).                          #\n'
printf '########################################################################\n'

# ----------------------------------------------------------------------------
# (1) BUILD  — the "install" stage from a clean checkout.
# ----------------------------------------------------------------------------
step 1 "BUILD (cargo build --workspace + the tritium binary)"
cargo build --workspace
cargo build -p tritium-cli
TRITIUM="$REPO_ROOT/target/debug/tritium"
[ -x "$TRITIUM" ] || fail BUILD "tritium binary not found at $TRITIUM after build"
pass BUILD "workspace + tritium binary built ($TRITIUM)"

# ----------------------------------------------------------------------------
# (2) INFER  — backend discovery + real GGUF parse + real runner-load attempt.
# ----------------------------------------------------------------------------
step 2 "INFER (list-backends + inspect + generate load-path)"

info "list-backends:"
"$TRITIUM" list-backends
"$TRITIUM" list-backends | grep -q '^\s*cpu' \
  || fail INFER "cpu backend not enumerated by list-backends"
pass INFER "list-backends enumerated the registered cpu backend"

FIXTURE="$REPO_ROOT/crates/tritium-format/tests/fixtures/bitnet_tiny.gguf"
[ -f "$FIXTURE" ] || fail INFER "tiny GGUF fixture missing at $FIXTURE"

info "inspect (full GGUF v3 reader over the committed tiny ternary fixture):"
"$TRITIUM" inspect "$FIXTURE"
pass INFER "inspect parsed the GGUF container end to end (metadata + ternary tensor table)"

# `generate` drives the complete ModelRunner::load_cpu path. The fixture is a
# deliberately PARTIAL model, so the expected outcome is a CLEAN non-zero exit
# with a descriptive missing-tensor error and NO panic — proving the
# arg/token/parse/config/weights plumbing without fabricating a runnable model.
TOKENS="$WORK/tokens.json"
printf '[1, 2, 3]\n' > "$TOKENS"
info "generate (drives ModelRunner::load_cpu; partial fixture => expect a CLEAN load failure):"
set +e
GEN_OUT="$("$TRITIUM" generate --model "$FIXTURE" --tokens "$TOKENS" --max-new 4 2>&1)"
GEN_RC=$?
set -e
printf '%s\n' "$GEN_OUT"
[ "$GEN_RC" -ne 0 ] || fail INFER "generate unexpectedly succeeded on the partial fixture (it is not a complete model)"
printf '%s' "$GEN_OUT" | grep -qi 'panic' \
  && fail INFER "generate PANICKED — the load path must fail cleanly, not panic"
printf '%s' "$GEN_OUT" | grep -Eqi 'failed to load model|missing tensor' \
  || fail INFER "generate did not produce the expected clean load error; got: $GEN_OUT"
pass INFER "generate exercised the runner-load path and failed CLEANLY (exit $GEN_RC, no panic)"
info "DEFERRED: a token-emitting forward needs a COMPLETE model (real BitNet 2B4T GGUF, multi-GB,"
info "          ~seconds/token on CPU). Covered by the model-gated generate_on_real_model_runs test"
info "          (TRITIUM_RUN_SLOW=1) and the GPU acceptance lane (crates/tritium-nn/tests/acceptance.rs)."

# ----------------------------------------------------------------------------
# (3) SALT  — the real SALT quantizer on synthetic fp32 inputs.
# ----------------------------------------------------------------------------
step 3 "SALT (report salt + quantize → .tslb)"

MAT="$WORK/matrix.json"
emit_matrix_json 8 16 "$MAT"
info "report salt (real tritium_quantize::quantize_tensor + dequant; bpw/MSE/RMSE):"
"$TRITIUM" report salt --input "$MAT" --rows 8 --k 16 --budgets "1.585,2.0,2.5" --format table
pass SALT "report salt ran the production SALT quantize+dequant path"

# Full safetensors -> SALT-bundle pipeline (the literal "SALT-quantize" stage)
# when Python3 is available to write the binary safetensors payload.
ST="$WORK/tiny.safetensors"
TSLB="$WORK/tiny.tslb"
if command -v python3 >/dev/null 2>&1; then
  emit_safetensors "$ST"
  info "quantize (real safetensors -> SALT bundle .tslb):"
  "$TRITIUM" quantize --input "$ST" --output "$TSLB" --bpw 2.0
  [ -s "$TSLB" ] || fail SALT "quantize did not produce a non-empty .tslb bundle"
  pass SALT "quantize produced a SALT bundle ($(wc -c <"$TSLB") bytes) from a synthetic safetensors"
else
  info "python3 not found — skipping the safetensors->.tslb sub-step (the report-salt path above"
  info "already exercised the SALT quantizer). This skip is non-fatal."
fi
info "DEFERRED: quantizing a REAL multi-GB BitNet fp master + measuring downstream PPL is the"
info "          GPU/real-model capstone (the fp safetensors master is large; eval needs the model)."

# ----------------------------------------------------------------------------
# (4) FINE-TUNE  — the tritium-train CPU training engine (gradient + optimizer).
# ----------------------------------------------------------------------------
step 4 "FINE-TUNE (tritium-train CPU gates: STE gradcheck + AdamW + tape)"
info "STE + ternary-matmul backward vs finite difference (Gate C, ADR 0007):"
cargo test -p tritium-train --test gradcheck_ste_matmul
info "AdamW optimizer (closed-form step, multi-step f64 reference, loss decreases):"
cargo test -p tritium-train --test optim_adamw
info "tiny-transformer end-to-end gradient tape:"
cargo test -p tritium-train --test tape_tiny_transformer
pass FINE-TUNE "tritium-train CPU training engine verified (STE/ternary backward + AdamW + e2e tape)"
info "DEFERRED: fine-tuning a real model to RECOVER accuracy/PPL after quantization is the GPU"
info "          capstone (full-model backprop; ADR 0007 / plan 0010, tracked to v0.60+)."

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
printf '\n########################################################################\n'
printf '# CAPSTONE SMOKE: ALL STAGES PASSED                                     #\n'
printf '#   (1) BUILD     real                                                   #\n'
printf '#   (2) INFER     real parse + real runner-load (clean partial-fail)     #\n'
printf '#   (3) SALT      real quantizer (report + .tslb bundle)                 #\n'
printf '#   (4) FINE-TUNE real CPU training engine (STE/AdamW/tape)              #\n'
printf '#                                                                        #\n'
printf '# DEFERRED TO HARDWARE (the true v1.0 capstone — ADR 0012):              #\n'
printf '#   * a real BitNet b1.58 GGUF (multi-GB) decoding tokens end to end     #\n'
printf '#   * a GPU fine-tune that RECOVERS accuracy/PPL after quantization      #\n'
printf '#   * the GPU CI matrix + Metal/ROCm parity (fenced hardware)            #\n'
printf '########################################################################\n'
printf 'PASS [CAPSTONE] CPU fresh-env e2e smoke complete\n'
