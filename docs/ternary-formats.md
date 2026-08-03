# Ternary formats: what Tritium reads, writes, and refuses

Ternary weights are trits × scales in any container. Tritium's policy is
**compute formats decoupled from interchange formats**: the loader unpacks
every supported container to trits at load, and each backend packs its own
native layout (TQ2_0 by default; TQ1_0 under the opt-in capacity rung below)
— so file bytes never touch kernel code, and any supported file generates
bit-identically to any other encoding of the same trits.

## Supported containers

| format | bpw | scale | role |
|---|---|---|---|
| **I2_S** (ggml type 36) | 2.00 | per-tensor f32 | bitnet.cpp's baseline; what BitNet checkpoints ship as |
| **TQ2_0** (ggml type 35) | 2.06 | per-block f16 | the GPU compute layout (base-4 codes → 2-bit shifts → dp4a) |
| **TQ1_0** (ggml type 34) | 1.69 | per-block f16 | storage/interchange — base-243 (5 trits/byte), ~18% smaller ternary payload |

`tritium repack --input a.gguf --output b.gguf --to tq1|tq2` converts
losslessly between them (BitNet 2B4T: 1.188 GB → 1.106 GB as TQ1_0, ~10 s).

**Ship TQ1_0, run TQ2_0 — unless VRAM is the constraint.** By default the
loader unpacks TQ1_0 to trits and the CUDA backend packs TQ2_0, so a TQ1_0
file runs at full TQ2_0 speed. `TRITIUM_WEIGHTS=tq1` (A2) opts into serving
a native TQ1-packed layout instead: a **capacity** rung — −18% weight VRAM
(2B4T: ~−160 MB) at measured e2e decode parity (uncontended interleaved ×3:
median +2.3%, "parity, possibly a hair better" — OPTIMIZATION-LOG "Round
15/16 caveat CLOSED"). The kernel-level 1.51× gateup penalty (base-243
decode costs ~24 ALU ops per dp4a word vs TQ2_0's ~3, round 16) does not
materialize end-to-end: decode is latency-bound outside the GEMMs and the
−20% gateup DRAM traffic offsets. v1 scope: the single-sequence path only —
`--batch-slots > 1`, tree/spec sessions, and `--draft-model` refuse TQ1
loudly per-request (the `gb_*` batched/tree graph builders are TQ2-only;
`build_batch`/`tree_forward` guards).

## TB1 bitmap+signs: measured, refuted, kept

A 1.578-bpw bitmap+signs layout (one zero/nonzero bit per element + a packed
sign stream) exists in-tree as **TB1**, with bit-exact kernels and gates. It
is **refuted as a decode format at BitNet's density** (round 16): on the real
gateup shape (M=1, N=13824, K=2560) it measured 2.58× slower than TQ2_0
(33.77 vs 13.11 µs/launch on a contended box — absolutes inflated, the
*ordering* is the ALU-vs-bytes signal) for a 19% byte saving, because the
per-block warp prefix scan for sign addressing serializes exactly where M=1
GEMMs are not DRAM-bound *enough* (44–68% DRAM) to hide it. Bytes don't
convert to time here. The kernel, format, and bench harness stay in-tree;
the niche survives on paper only for high-sparsity students (p ≥ ~0.6,
where the sign stream shrinks and block-skip composes) — any redesign must
close most of the 2.58× gap to TQ2_0 on the gateup microbench (same-session
interleaved, not just beat TB1's stale absolute) *before* integration is
considered.

## The f16 scale gap

TQ block scales are f16, but I2_S per-tensor scales are f32 and real BitNet
checkpoints' scales are **not f16-representable** (e.g. `blk.0.attn_q` =
1.2188548). `tritium repack` preserves the authoritative f32 scale as
`tritium.i2s_scale.<tensor>` GGUF metadata; Tritium's loader prefers it
(validating agreement with the f16 block scales), so the repacked file loads
**bit-identically** (gated by `repacked_tq1_model_loads_bit_identical`).
Foreign loaders (llama.cpp et al.) fall back to the f16 block scales —
~1e-4 relative on the scale, the TQ formats' native precision.

Per-block scales must be uniform within each **row** of a tensor. Two cases
load: all rows sharing one scale takes the exact per-tensor path (with the
`i2s_scale` metadata check), and row-uniform scales that *differ across
rows* load as a per-row-scale ternary linear — the GEMM contract is per-row
`weight_scale[n]`, so this is exact, and it is what a per-row-α trained LM
head (ADR 0032 T2a) exports as TQ2_0. All-zero blocks (scale 0) have their
trits forced to the zero trit — exact at any scale. Scales that vary *within*
a row are rejected loudly rather than silently mis-scaled.

## TL1 / TL2: a non-goal, deliberately

bitnet.cpp's TL1 (ARM) and TL2 (x86) are **kernel-tuning artifacts, not
interchange formats**: their GGUF payloads are weight permutations generated
by `codegen_tl1/tl2.py` with per-model, per-platform blocking parameters
(BM/BK), and the parameters live in the generated kernel config — not in the
gguf. A TL1 file is only meaningful to the kernel build that produced it.

Practical consequences:

- Every BitNet model is available as (or convertible to) I2_S — bitnet.cpp's
  own `convert-helper` produces I2_S — and Tritium reads that directly.
- The *technique* behind TL1/TL2 (per-activation lookup tables via
  `pshufb`/`tbl`) was assessed for Tritium's CPU backend and lost to the
  AVX2 `maddubs` int8 path on the target CPU (i5-13600K); it wins on weaker
  ARM cores, which is exactly why bitnet.cpp targets it at ARM.
- If TL1/TL2 file reading is ever genuinely needed, the trits-level loader
  hook (`load_ternary` in tritium-nn) is the extension point — it requires
  the producing build's blocking parameters as side information.
