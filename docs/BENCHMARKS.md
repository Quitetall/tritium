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

### 2026-08-08 — T8 publishing sweep @ dde0c61: decode/pp512, lossless spec decode, batched multi-slot, TQ1 capacity

**Box state (disclosed up front): NOT quiet.** A parallel session's
`salt_distill_heldout` test (Tritium repo, PID 2609694) ran on-GPU for the whole sweep,
growing 388 MiB → 4.3 GiB and pulling 27–100% GPU util between my runs; desktop stack
~1.05 GiB (kwin/Xwayland/plasmashell/firefox/ghostty). RTX 4090, driver 610.57.04,
CUDA 13.3. Every number below carries that contention; relative pairs were measured
same-session order-alternated (ABBA) so both sides share it. **Quiet-box rerun of the
absolutes is owed.** Binary provenance: the shared working tree held uncommitted
parallel-session changes (including `tritium-cli/src/report.rs`), so all Tritium
binaries were built from a clean `git archive dde0c61` export
(`cargo build --release -p tritium-cli -p tritium-serve --features
tritium-cli/cuda,tritium-serve/cuda`), not the dirty tree.

#### 1. Plain decode + pp512 (the ledger command)

Command: `tritium report compare --model ggml-model-i2_s.gguf --tokens prompt512.json
--backend cuda --prompt-len 512 --decode-steps 256 --warmup 16 --reps 3 --runs 5
--format json` @ dde0c61 (clean-HEAD build). Tokens: first 512 ids of a wt103 doc
(BOS 128000). Run twice per protocol; both bundles recorded because the co-resident
distill was RAMPING between them (run 1 caught the lighter window).

| bundle | decode tok/s (3 reps) | median | pp512 tok/s | ttft p50 / p95 ms |
|---|---|---:|---:|---|
| run 1 (07:28 UTC, distill just starting) | 216.1 / 211.8 / 215.0 | **215.0** | **22,298.6** | 22.96 / 23.88 |
| run 2 (07:29 UTC, distill ramping) | 191.7 / 191.2 / 191.1 | 191.2 | 19,603.8 | 26.05 / 26.74 |

pp512 at HEAD is **19.6–22.3k tok/s even contended** — up from the 2026-07-18 ledger
line of 12,275 (v3-attention era) and above the same-day llama.cpp Q4_K_M reference
(12,608, below). Decode 191–215 sits inside the documented contention band around the
221–222 quiet-class entries; treat the decode absolutes as contended, not a regression.
JSON artifacts: session scratchpad `compare-head-run{1,2}.json` (receipts dir pattern
`docs/receipts-ws-*` is gitignored; artifacts live outside the repo).

#### 2. Solo speculative decode — lossless, tau, e2e

Harness: `~/blut/scripts/measure_tau.py` (12 wt103 prose prefixes, 256 greedy tokens
each) against two clean-HEAD `tritium-serve` instances: spec on 8124
(`--draft-model ~/blut/data/drafter-8L768-s3.gguf`), plain on 8125; both
`--model ggml-model-i2_s.gguf --backend cuda --raw-tokens --eos 4294967295
--max-completion-tokens 4096`. Cold pass discarded (prompt 0 paid ~22 s of JIT on both
servers); the warm pass is the record:

| metric | value |
|---|---:|
| losslessness | **12/12 prompts token-identical** (spec == plain greedy), both passes |
| tok/verify (tau) | **3.575** (3,057 committed / 855 verifies) |
| e2e wall | spec 10.02 s vs plain 11.08 s → **1.105×** |

Per-prompt spread was contention-noisy (spec 0.47–1.39 s per 256 tok); the lossless
count and tau are contention-immune, the 1.105× e2e is same-session relative.

#### 3. Batched multi-slot spec decode (with the 9930062 bucket snap)

Harness: measure_multi.py class (session scratchpad copy pointed at the clean-HEAD
`tritium-serve`), Round-25 protocol: server at `--batch-slots 4`, shared draft k
(`TRITIUM_MULTI_K=shared`, the shipped default; per-slot k was refuted at 9930062),
N concurrent 192-token wt103 streams, drafted-vs-undrafted ABBA×2 at the
server-launch level, 5 samples/visit → n=20 per config, p50.

| N streams | drafted p50 agg tok/s | undrafted p50 agg tok/s | speedup | tok/verify p50 |
|---|---:|---:|---:|---:|
| 2 | 189.8 (min 149.5, max 408.7) | 102.9 | **1.84×** | 2.09 |
| 4 | 276.2 (min 244.0, max 506.3) | 217.8 | **1.27×** | 2.01 |

Honest reading: the Round-25 headline (2.44×/1.52×, tok/verify 2.63) was measured on a
lighter box (5 GB idle squatter, no active compute); this session's absolutes are
~half Round-25's and the sample spread (149–506 tok/s at fixed config) shows the
distill co-resident dominating variance. These rows re-confirm drafted > undrafted at
HEAD under load; they do NOT cleanly re-measure the bucket-snap +6.6% (same-session
relative claim in 9930062) and are not comparable to Round-25's absolutes.
Quiet-box recapture owed for the headline table.

#### 4. TQ1 capacity rung (batched-spec serve)

Harness: same batched-spec server (`--batch-slots 4` + drafter), `TRITIUM_WEIGHTS=tq2`
vs `tq1`, ABBA at launch level (tq2/tq1/tq1/tq2); peak per-process VRAM sampled from
`nvidia-smi --query-compute-apps` at 0.5 s during a warm+measured 4-stream sample,
then a single-stream 256-token greedy parity completion per visit.

| weights | peak serve VRAM (2 visits) | parity (256-tok greedy) |
|---|---:|---|
| tq2 (default) | 7,648 / 7,636 MiB | reference |
| tq1 | **6,976 / 6,976 MiB (−672 MiB)** | **token-identical to tq2, all visits** |

Two new facts at HEAD: the batched-spec path now ACCEPTS `TRITIUM_WEIGHTS=tq1`
(round 15's "batch/tree reject loudly in v1" is lifted), and the capacity win holds
end-to-end in the serve process (−672 MiB ≈ the −18%-of-ternary-bytes rung from
round 15). No speed claim: the contended 4-stream aggregates spanned 259–431 tok/s
*within the tq2 pair alone*, so tq1-vs-tq2 throughput stays an open quiet-box item
(round 16's "re-bench uncontended" note stands).

#### 5. Competitor line (documented invocation, re-run same-day)

`llama-bench -m /home/brianklam/models/qwen3.5-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf -p 512
-n 128 -ngl 99`, fork build 41a666dac (8943) — the same binary as the 2026-07-11
reference line. Ran in the lightest window of the session (desktop ~1.4 GiB, distill
at 388 MiB just starting):

| engine | model | pp512 tok/s | tg128 tok/s |
|---|---|---:|---:|
| llama.cpp CUDA | Qwen3.5-4B Q4_K_M (2.54 GiB) | 12,607.84 ± 1,173.85 | 199.79 ± 3.65 |

Stable vs the 07-11 reference (12,281 ± 828 / 200.1 ± 6.5). **Still owed:** the
upstream-master Q2_0 quiet-box rerun from the 2026-07-30 entry — the upstream build
(1212) and the Q2_0 gguf are not present on this box; not improvised.

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
