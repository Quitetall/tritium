# Ternary formats: what Tritium reads, writes, and refuses

Ternary weights are trits × scales in any container. Tritium's policy is
**one compute format, many interchange formats**: the loader unpacks every
supported container to trits at load, and each backend packs its own native
layout — so the container never touches kernel code, and any supported file
generates bit-identically to any other encoding of the same trits.

## Supported containers

| format | bpw | scale | role |
|---|---|---|---|
| **I2_S** (ggml type 36) | 2.00 | per-tensor f32 | bitnet.cpp's baseline; what BitNet checkpoints ship as |
| **TQ2_0** (ggml type 35) | 2.06 | per-block f16 | the GPU compute layout (base-4 codes → 2-bit shifts → dp4a) |
| **TQ1_0** (ggml type 34) | 1.69 | per-block f16 | storage/interchange — base-243 (5 trits/byte), ~18% smaller ternary payload |

`tritium repack --input a.gguf --output b.gguf --to tq1|tq2` converts
losslessly between them (BitNet 2B4T: 1.188 GB → 1.106 GB as TQ1_0, ~10 s).

**Ship TQ1_0, run TQ2_0.** TQ1_0 is a *storage* format only: its base-243
decode costs ~4× the per-trit ALU of TQ2_0, and the decode GEMMs have no
DRAM headroom to buy it back (measured 1.19–1.37× slower even with a
shared-LUT + `__byte_perm` unpack — see OPTIMIZATION-LOG round 10b). The
loader erases the difference: a TQ1_0 file runs at full TQ2_0 speed because
backends never see TQ1_0 bytes.

## The f16 scale gap

TQ block scales are f16, but I2_S per-tensor scales are f32 and real BitNet
checkpoints' scales are **not f16-representable** (e.g. `blk.0.attn_q` =
1.2188548). `tritium repack` preserves the authoritative f32 scale as
`tritium.i2s_scale.<tensor>` GGUF metadata; Tritium's loader prefers it
(validating agreement with the f16 block scales), so the repacked file loads
**bit-identically** (gated by `repacked_tq1_model_loads_bit_identical`).
Foreign loaders (llama.cpp et al.) fall back to the f16 block scales —
~1e-4 relative on the scale, the TQ formats' native precision.

Per-block scales must be uniform across a tensor's nonzero blocks (pure
ternary tensors always are: every block's absmax is either the tensor scale
or zero). All-zero blocks (scale 0) have their trits forced to the zero trit
— exact at any tensor scale. Genuinely non-uniform block scales are rejected
loudly rather than silently mis-scaled through the per-tensor path.

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
