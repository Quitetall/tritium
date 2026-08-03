# Tritium benchmarks — the reproducible ledger

**Every number in this file reproduces from one command.** No number is recorded without the
exact command beside it and the environment it ran in (GPU, driver, co-resident processes).
Competitor numbers carry their exact invocations; we document their repro lines, we do not wrap
their binaries. The ledger is updated ONLY by re-running the harness — dated entries, newest
first. (ADR 0026 Track R.)

## The command

```sh
tritium report compare \
  --model <path/to/model.gguf> \
  --tokens <path/to/token-ids.json> \
  --backend cuda \
  --prompt-len 512 --decode-steps 256 --warmup 16 --reps 3 --runs 5 \
  --format json
```

Emits: decode tok/s ×3 (median), prefill tok/s + ttft p50/p95 over a 512-token prompt, and an
environment capture (GPU, driver, VRAM, co-resident compute processes, date, git commit). The
JSON is the artifact; the tables below are transcriptions of it.

**`--backend` resolves against backends linked into the binary**, so build the one you intend to
measure — the flag cannot reach a backend the binary does not contain:

| hardware | build | `--backend` |
|---|---|---|
| NVIDIA | `cargo build --release -p tritium-cli --features cuda` | `cuda` |
| AMD / Intel / any Vulkan GPU | `cargo build --release -p tritium-cli --features wgpu` | `wgpu` |
| AMD with a ROCm toolkit | `cargo build --release -p tritium-cli --features rocm` | `rocm` |

`wgpu` needs only a Vulkan driver (no vendor toolkit) and is the portable lane; `rocm` needs
`hipcc` present at build time. On any backend other than `cuda` the `gpu` field is taken from the
backend's own adapter name and suffixed with the backend that produced it —
`"<adapter name> [wgpu]"`. It is deliberately **not** taken from `nvidia-smi` there: a box can
expose several adapters, so `nvidia-smi` may describe a card the run never touched. `driver` and
`vram_*` come only from `nvidia-smi` and stay `"unavailable"` off NVIDIA — disclose that rather
than filling it in. Note `roofline_4090_pct` and `baseline_4090_drop_pct`
are computed against fixed RTX-4090 reference constants by definition (hence the names); off that
box they are a distance from a named reference point, **not** a claim about the local device.

### Methodology and honest caveats

- **Decode** = single-stream greedy steps after prefill+warmup, M=1, the memory-bound regime.
  Effective weight-stream bandwidth = tok/s × weight bytes is the honest cross-model metric.
- **Prefill (pp512)** = one 512-token `forward` p50 over N runs. First-run numbers include
  one-time JIT/tune costs unless the autotune disk cache is warm — run twice, record the second.
- **Contention**: numbers move ±10% under desktop/co-resident-GPU load; the `co_resident`
  capture field discloses the box state. The order-alternated ABBA protocol (OPTIMIZATION-LOG
  round 21) is the tie-breaker for close comparisons.
- **Competitor lines** (documented, not wrapped):
  - llama.cpp: `llama-bench -m <model.gguf> -p 512 -n 128 -ngl 99`
  - bitnet.cpp (CPU): `llama-bench -m <i2s.gguf> -p 512 -n 128 -t 14`

## Ledger

### 2026-07-30 — llama.cpp Q2_0 CUDA merged upstream (PR #25707, merged TODAY) — first same-box head-to-head

The mainstream-ternary-CUDA gap closed this morning. Same box (RTX 4090, ~5.3 GB
co-resident desktop load — CONTENDED, both engines equally), same day, ABBA
interleaved. llama.cpp = upstream master 5f55650a7 (build 1212), fresh CUDA build.
Tritium @ HEAD via the ledger decode command (256 steps, 16 warmup).

| engine | model | weights | decode tok/s (spread) | eff. weight stream |
|---|---|---|---:|---:|
| Tritium CUDA | BitNet 2B4T ternary I2_S | 1.71 GiB | 264-281 (median ~277) | ~474 GiB/s |
| llama.cpp CUDA (NEW Q2_0) | Qwen3.5-4B Q2_0 g64 | 1.42 GiB | 223-264 (mean-of-runs 248) | ~352 GiB/s |
| llama.cpp CUDA (prior ledger line) | Qwen3.5-4B TQ2_0 | 1.35 GiB | 24.0 | ~32 GiB/s |

Reading, honestly: llama.cpp's new Q2_0 CUDA path is REAL — a ~10x jump over its
TQ2_0 line, landing within ~25-35% of Tritium's bandwidth-normalized decode
efficiency on its first day. Tritium still leads (~474 vs ~352 GiB/s effective),
but "only ternary CUDA engine" is retired as a claim as of 2026-07-30. Different
models/architectures (2.4B BitNet vs 4B Qwen) — bandwidth normalization is the
comparison basis; a same-model run (Bonsai Q2_0 vs its Tritium import via the new
q2_0 reader) is the clean follow-up. pp512 for Q2_0: 12,199 +- 1,109 (llama.cpp
MMQ-class; Tritium's ledger pp512 12,275 sustained — parity band).

Repro: `llama-bench -m qwen35-4b-q2_0.gguf -p 512 -n 128 -ngl 99` (master
5f55650a7); Tritium ledger command above. Contention disclosed; quiet-box rerun
owed before any published table.



<!-- Newest first. Each entry: date, git commit, environment line, table, JSON path/attachment. -->

### 2026-07-12 — IMMA prefill live (ADR 0026 Track P steps 1-4)

Command: `tritium report compare --model ggml-model-i2_s.gguf --tokens <ids> --backend cuda
--prompt-len 512 --decode-steps 256 --warmup 16 --reps 3 --runs 5` @ git 5075683-era tree
(Track P complete through the dispatch), RTX 4090, driver 610.43.02, co-resident: kwin only.

| metric | value | vs 2026-07-11 baseline |
|---|---:|---|
| pp512 prefill | **1,969.7 tok/s** (p50 260.0 ms) | 1,068 → **+84%** |
| decode (256 steps after the 512-token prefill, ctx 512→768) | 221.5 tok/s median | shape differs from the 6-token-prompt baseline (273-276); not comparable directly |
| decode (32 steps, 64-token prompt) | 317.9 tok/s | — |

Numerics: IMMA prefill is BIT-IDENTICAL to dp4a (gate: 128,256 logits to_bits, one-shot ==
4×128-chunked == dp4a). The ≥6,000 pp512 campaign gate remains open — gap analysis in
OPTIMIZATION-LOG round 22.

### 2026-07-18 — rev-4 IMMA + v2 prefill attention (the pp512 gate falls)

Command: `tritium report compare --model ggml-model-i2_s.gguf --tokens <ids> --backend cuda
--prompt-len 512 --decode-steps 256 --warmup 16 --reps 3 --runs 5` @ b20f78f-era tree
(rev-4 IMMA JIT a3e8a49 + v2 attention 98ab046 + nit fold b20f78f), RTX 4090, driver
610.43.02, co-resident: kwin + one idle tritium-serve holding 9.1 GB VRAM (0% compute).

| metric | value | vs 2026-07-12 | vs 2026-07-11 baseline |
|---|---:|---|---|
| pp512 prefill | **9,068.8 tok/s** (p50 56.4 ms) | 1,969.7 → **4.6×** | 1,068 → **8.5×** |
| decode (256 steps after the 512-token prefill) | 222.4 tok/s median | 221.5 → unchanged | — |

Run-to-run clock variance puts pp512 at 8,400–9,100 across the session (the order-alternated
5-run ttft matrix measured p50 60.6/60.7 ms on the A/A2 pair); the compare-bundle line above
is the reproducible-command artifact. **The ADR 0026 ≥6,000 gate is PASSED** (stretch 10,000
within reach — attention is now 74% of prefill; see OPTIMIZATION-LOG round 23).

Config isolation (same command, env kill switches, order-alternated ABBA):

| config | p50 ms | pp512 tok/s |
|---|---:|---:|
| dp4a + rev-1 attention (`TRITIUM_IMMA_PREFILL=0 TRITIUM_ATTN_V2=0`) | 318.3–320.4 | ~1,600 |
| rev-4 IMMA only (`TRITIUM_ATTN_V2=0`) | 126.9–127.1 | ~4,030 |
| v2 attention only (`TRITIUM_IMMA_PREFILL=0`) | 225.6–259.0 | ~2,100 |
| both (defaults) | 60.6–60.7 | **~8,440** |

Numerics: BOTH legs are bit-identical to their predecessors by gate (rev-4 IMMA: to_bits vs
dp4a over 128,256 logits incl. one-shot == 4×128 chunked; v2 attention: to_bits vs rev-1 per
(row, head), 6 regimes incl. the ctx=3584 shared-budget boundary) — zero numerics re-gating.

Methodology caveat learned this session: an earlier run measured a STALE release binary
(pre-commit) and reported flat results — `nsys` kernel names are the cheap staleness check
(`gqa_attention_batch_v2_f32` + r4-keyed tune files must appear).

### 2026-07-18 (later) — v3 Q-blocked attention: past the 10,000 stretch

Command: same compare bundle @ v3 tree (Q-blocked attention on top of rev-4 IMMA), RTX 4090,
co-resident: kwin + the idle 9.1 GB serve.

| metric | value | note |
|---|---:|---|
| pp512 prefill | **12,274.7 tok/s** (p50 41.9 ms, 20-run bundle) | sustained-clock ABBA pairs spanned 12,103–16,893 (30.3–42.3 ms) across the box's clock states |
| v3 vs v2 attention (ABBA, same session) | **2.2–2.8×** | the trustworthy relative number; v2 measured 5,482–6,032 in the same session |
| decode | v3 on/off ABBA: 145.6/148.3 and 108.7/156.5 | statistically indistinguishable — v3 is prefill-only; today's absolute decode (~130–156) is depressed vs yesterday's 222.4 by box state, not code |

**The 10,000 stretch goal is PASSED at the slow end of the spread; the fast end (16,893)
exceeds the llama.cpp Qwen3.5-4B Q4_K_M reference (12,281).** v3 blocks 8 query rows per
(block, head): staged K feeds all 8 dot chains, each V load folds into 8 accumulators, scores
return to the global scratch (lifting v2's ctx cap — v3 has none). Bit-identical to rev-1/v2
per (row, head) by to_bits gate (staircase, BQ tails, ctx 3809 > the v2 cap, both dtypes).
nsys @ v3: attention 45% (~22.6 ms) / IMMA GEMM 43% — the profile is now balanced; further
pp512 gains need both (cp.async smem swizzle on IMMA; deeper Q-blocking or the numerics-RFC
flash rewrite on attention).

### 2026-07-11 head-to-head (reference)

From `docs/research-ternary-sota-mid2026.md` §7, Tritium @ a80e185 — the pp512 gap to
mainstream was 11.5×; at 2026-07-18 it is **1.35×**:

| engine | model | decode tok/s | pp512 tok/s |
|---|---|---:|---:|
| Tritium CUDA | BitNet 2B4T (ternary I2_S) | 301.4–302.8 | 1,068 |
| llama.cpp CUDA | Qwen3.5-4B Q4_K_M | 200.1 ± 6.5 | 12,281 ± 828 |
| llama.cpp CUDA | Qwen3.5-4B TQ2_0 | 24.0 ± 1.6 | 107 ± 5 |
| bitnet.cpp CPU (14t) | BitNet 2B4T I2_S | 23.1 ± 0.1 | ~203 |

(Quiet-box re-baseline at HEAD b701c53, 2026-07-12, 512-step decode ×3: 273.2–275.9 tok/s —
the 301–302 line above was measured at 256 steps under lighter desktop load; both are inside
the documented contention spread. The pp512 line predates the Track P work.)
