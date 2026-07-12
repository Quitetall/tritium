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

<!-- Newest first. Each entry: date, git commit, environment line, table, JSON path/attachment. -->

### (pending) 2026-07-XX — post-Track-P run

To be recorded by re-running the command above on a quiet box once ADR 0026 Track P (IMMA
prefill) gates pass. Baseline to beat, from the 2026-07-11 head-to-head
(`docs/research-ternary-sota-mid2026.md` §7, Tritium @ a80e185):

| engine | model | decode tok/s | pp512 tok/s |
|---|---|---:|---:|
| Tritium CUDA | BitNet 2B4T (ternary I2_S) | 301.4–302.8 | 1,068 |
| llama.cpp CUDA | Qwen3.5-4B Q4_K_M | 200.1 ± 6.5 | 12,281 ± 828 |
| llama.cpp CUDA | Qwen3.5-4B TQ2_0 | 24.0 ± 1.6 | 107 ± 5 |
| bitnet.cpp CPU (14t) | BitNet 2B4T I2_S | 23.1 ± 0.1 | ~203 |

(Quiet-box re-baseline at HEAD b701c53, 2026-07-12, 512-step decode ×3: 273.2–275.9 tok/s —
the 301–302 line above was measured at 256 steps under lighter desktop load; both are inside
the documented contention spread. The pp512 line predates the Track P work.)
