# Benchmarks

This chapter documents Tritium's performance **methodology** — what is measured,
how, and (critically) **what each number does and does not prove**. It quotes no
throughput figure as a Tritium result here: the real tokens/sec-vs-competitor
comparison requires the pinned RTX 4090 plus the BitNet 2B4T model, which this
book's CI does not have. The committed numbers live in code
(`benches/src/lib.rs`) with explicit provenance, and the harness is built so a
CPU-hosted CI can prove the *logic* of the gates without ever owning a GPU. The
methodology follows the [v0.30 performance ADR](../../adr/0005-v030-performance.md).

## The two layers of measurement

| Layer | Crate / file | Gate | Where it runs |
|-------|--------------|------|----------------|
| Divan **microbenchmarks** | `benches/benches/mpgemm.rs`, `gpu_mpgemm.rs` | none (timing) | CPU mpGEMM always; GPU mpGEMM `--features cuda` |
| **Roofline** ceiling math | `benches/benches/roofline.rs` + `benches/src/lib.rs` | unit-tested | always (pure host arithmetic) |
| **End-to-end** tok/s | `benches/benches/e2e.rs` | perplexity-within-1% + `>5%` regression | `--features cuda` **and** model present |
| CLI **reports** | `crates/tritium-cli/src/report.rs` | reproducible JSON/table | CPU always; `--backend cuda` on a GPU |

## Microbenchmarks (divan)

The microbenches isolate the ternary mpGEMM kernel from the full forward pass.

- **`mpgemm.rs`** — the always-on CPU baseline. `cpu_mpgemm_tq2_0_decode` runs a
  decode-shaped (`M=1`) `TQ2_0` mpGEMM on the CPU backend across `K ∈
  {256, 1024}`. No GPU, no model — it compiles and runs on every lane, and it is
  the bench the hosted **`cpu-bench-smoke`** CI job executes (`cargo bench -p
  tritium-benches --bench mpgemm`) to keep the CPU kernel + the divan harness
  from bit-rotting.
- **`gpu_mpgemm.rs`** — `gpu_add_only` and `gpu_imma` sweep the BitNet 2B4T
  linear-layer shapes (`M ∈ {1, 8, 32, 256, 512}` × `(N, K) ∈ {2560, 6912}²`,
  the committed `BITNET_SHAPES`). Bodies are `#[cfg(feature = "cuda")]`, so they
  compile out on a CPU-only build and only do work with `--features cuda` on an
  NVIDIA GPU.

The shapes and the weight fixtures (`packed_tq2_0_weights`,
`packed_i2s_int8_weights`) are derived in `benches/src/lib.rs` and unit-tested —
so the *fixtures* are validated on CPU CI even though the GPU sweep is not run.

## Roofline / % of SOL

"Speed-Of-Light" (SOL) is the hardware ceiling for a kernel's *limiting*
resource. ADR 0005 splits ternary inference into two regimes, and the roofline
harness encodes the ceiling math for each. None of this requires a GPU — it is
arithmetic over committed device constants, unit-tested in `benches/src/lib.rs`,
and printed by the always-on `roofline` bench.

### Decode — memory-bound

A batch-1 autoregressive step streams every weight from HBM exactly once, so it
is bandwidth-bound:

```text
decode_tok/s  ≤  peak_HBM_bandwidth / model_weight_bytes
```

The committed device constant is the **RTX 4090** peak HBM bandwidth,
`RTX_4090_PEAK_HBM_BW_BYTES_PER_SEC = 1008e9` (GDDR6X, 384-bit × 21 Gbps), and
the model footprint is `BITNET_2B4T_I2S_BYTES = 1_187_801_280` (the `I2_S` GGUF
file size). `bitnet_2b4t_decode_ceiling()` computes the quotient — about **848.6
tok/s** — and a unit test (`decode_ceiling_is_bandwidth_over_bytes`) pins it into
the `840–860` band so a future mis-edit of either constant fails CI. **848.6
tok/s is a *ceiling*, not a measured Tritium rate** — it is the denominator the
e2e bench divides a measured rate into to report "% of roofline".

### Prefill — compute-bound

Large-`M` prefill is bound by int8 tensor-core throughput, not bandwidth:

```text
prefill_tok/s  ≤  peak_int8_TOPS / macs_per_token
```

The committed constant is `RTX_4090_PEAK_INT8_TOPS = 660.6` (dense INT8, no
sparsity). The achieved fraction is measured with NVIDIA `ncu` — the exact
invocations live in `benches/README.md`; the harness provides the *ceiling and
the command*, not a profiling pass.

## End-to-end tokens/sec, coupled to accuracy

`benches/benches/e2e.rs` is the milestone-level bench. It drives the real
`ModelRunner` over the BitNet 2B4T GGUF on the CUDA backend and measures:

- **`decode_tokens_per_sec`** — single-token steps through the KV cache (the
  memory-bound path), reported as a **decode-only** rate (per-iteration prefill
  is excluded from the accumulated time, so it is apples-to-apples with the
  competitor's decode number) and as a % of the `848.6` roofline.
- **`prefill_tokens_per_sec`** — one forward over the whole prompt (the
  compute-bound path).

Two gates make a fast-but-wrong configuration impossible to report green:

1. **Perplexity-within-1%.** Before timing, the bench computes teacher-forced
   perplexity over the committed eval sequence and asserts it is within 1% of the
   `transformers` reference (the *same* gate `crates/tritium-nn/tests/acceptance.rs`
   enforces). A perplexity drift `panic!`s the bench — a perf number is only
   meaningful if the model still produces the right distribution.
2. **`>5%` regression gate.** `check_regression(measured, &baseline)` flags a
   slowdown larger than `REGRESSION_DROP_THRESHOLD = 0.05` versus a recorded
   baseline; a speedup never trips it. The bench `panic!`s on a tripped gate, so
   the scheduled CUDA lane actually fails on a regression.

### Baselines and their provenance — read this before quoting any number

The comparison points are committed in `benches/src/lib.rs` as `Baseline`s with
an explicit `BaselineSource` (`BuiltOnBox` vs `Published(citation)`), unit-tested
to carry a citation. This is the honesty boundary the whole methodology turns on:

- **`BITNET_CPP_2B4T_DECODE`** — `28.0 tok/s`, **`Published`** (Microsoft's
  bitnet.cpp CPU figure). A conservative *CPU* number.
- **`LLAMA_CPP_2B4T_DECODE`** — `18.0 tok/s`, **`Published`**. Mainline
  `llama.cpp` *built* on the box but **cannot load** this exact `I2_S` GGUF: the
  GGUF quant type-id `36` is the removed `IQ4_NL_4_4` in current ggml, not BitNet
  `I2_S` (a fork-specific assignment) — so a `BuiltOnBox` figure for this artifact
  is not obtainable, and the published number stands as the recorded fallback.
- **`TRITIUM_2B4T_DECODE_4090`** — `130.0 tok/s`, **`BuiltOnBox`**. This is
  Tritium's own recorded decode rate on the pinned 4090, committed as a
  **conservative floor** (below the observed minimum so run-to-run clock variance
  never trips the gate). It is the **regression denominator** the e2e gate keys
  on — a real measured figure, not a competitor's.

Why the e2e gate compares Tritium against *itself*: there is **no obtainable GPU
ternary competitor** for this artifact. bitnet.cpp's published numbers are CPU,
and `llama.cpp`'s CUDA backend has no `TQ1_0`/`TQ2_0` ternary mul-mat kernel — it
runs ternary GEMM on the CPU regardless of `-ngl`. The competitor numbers are
printed as *context*; the gate is "don't regress versus our own measured decode"
plus the roofline %. The full recorded build-attempt log is in
`benches/README.md` and the constants' doc-comments.

> **Bottom line on numbers.** The throughput figures committed in code are the
> developer's recorded measurements on the pinned hardware; they are not
> re-measured by this book or by CPU CI. Reproducing them needs the pinned 4090 +
> the model. This chapter documents the *method and the gates*, not a claim that
> any particular tok/s reproduces in your environment.

## CLI reports (`tritium report`)

`tritium report` emits one-scenario, machine-readable reports (JSON / table /
both via `--format`). The implementation is `crates/tritium-cli/src/report.rs`.

- **`report decode`** — prefill the prompt, optionally warm up, then time
  `--decode-steps` single-token steps and report `tokens_per_sec`,
  `ms_per_token`, `roofline_4090_pct` (against the same `848.6` ceiling), and
  `baseline_4090_drop_pct` (against the `142.0` recorded decode figure). Runs on
  **any backend** including `cpu`.
- **`report ttft`** — time `--runs` full prefills and report p50 / p95
  time-to-first-token.
- **`report parity`** — greedy-generate the same `--max-new` tokens on **both**
  the `cpu` and `cuda` backends and report `matched_tokens` + `exact_match`. This
  one **requires** a CUDA build + device (it loads both backends).
- **`report salt`** — SALT bpw/error report for a flat JSON fp32 matrix:
  requested vs logical bpw, MSE/RMSE/max-abs error, and the residual-plane
  histogram across `--budgets`. Pure quantisation math — no model, no GPU. See
  [Quantization](./quantization.md).

> The roofline/baseline constants in `report.rs` (`1008e9`, `1_187_801_280`,
> `142.0`) are the RTX 4090 / BitNet 2B4T figures, so `roofline_4090_pct` from a
> **CPU** decode report is "% of the *4090's* memory ceiling", not "% of your
> CPU's ceiling". The report is honest about which backend ran (`backend` field);
> read the percentage with the hardware in mind.

## Regimes of ternary advantage (research, attributed)

The repository carries research notes — `docs/research-ternary-optimization.md`
and `docs/research-ternary-mathematical-advantage.md` — that analyse *where*
ternary's advantages actually pay off. These are **research/analysis documents,
not measured Tritium results**; summarised here with attribution, they frame why
the methodology above splits decode from prefill:

- **Decode is memory-bound.** Per the mathematical-advantage note, a batch-1
  ternary GEMM sits far below the roofline knee, so it is bandwidth-bound: the
  *multiply-free* property (add/sub/skip) saves essentially zero wall-clock time
  because the arithmetic units idle waiting on memory. The lever that *would*
  matter in this regime is **zero-state sparsity** — skipping the ~1/3 of weights
  that are exactly `0` to cut memory traffic — which the note flags as **not
  currently exploited** by Tritium's kernels (they decode every 2-bit code).
- **Prefill is compute-bound.** At large `M` the int8 tensor-core (IMMA) path is
  bound by TOPS, and the multiply-free / int8 contraction is where the arithmetic
  advantage shows up — which is why prefill is measured against the int8 TOPS
  roofline, not the bandwidth one.
- **Training.** The optimization note frames practical ternary training as
  fine-tuning from an FP checkpoint (the ParetoQ "90% FP pretrain + 10% QAT"
  split) rather than from scratch, and discusses STE-tape techniques (Tequila,
  annealed residuals) as the dominant levers on training quality.

> **Attribution and honesty.** The conditional "ternary beats 2-bit *only if*
> zero-sparsity is exploited" is the central claim of those research docs, and
> the corollary is that **Tritium does not yet exploit P1 (sparsity)** — its CUDA
> kernels do not skip zeros. That is a documented gap, not a shipped capability.
> The research docs live at `docs/research-*.md` in the repository tree (they are
> working analysis, not part of this guide), so they are referenced by path
> rather than linked here.

## What CPU-hosted CI proves vs what needs the pinned hardware

| Signal | Where proven | Hardware |
|--------|--------------|----------|
| CPU mpGEMM kernel + divan harness still run | `cpu-bench-smoke` (`cargo bench --bench mpgemm`) | hosted CPU CI |
| Roofline ceiling math (the `848.6` / `660.6` constants and the quotient) | unit tests in `benches/src/lib.rs` | hosted CPU CI |
| Regression-gate **logic** (`>5%` trips, speedup never trips, baselines carry citations) | unit tests (`regression_gate_trips_only_past_threshold`, `published_baselines_carry_a_citation`) | hosted CPU CI |
| SALT bpw/error report | `report salt` + `report.rs` tests | hosted CPU CI |
| CPU decode/ttft reports on the real model | `report decode/ttft --backend cpu` | any box **with the model** |
| **Measured decode/prefill tok/s, % of roofline, CPU↔CUDA parity** | `e2e` bench + `report parity` | **pinned RTX 4090 + the model** |
| Competitor-vs-Tritium tok/s comparison | recorded baselines (build log in `benches/README.md`) | pinned 4090 (and, for a real GPU competitor, not obtainable today) |

The split is deliberate: everything that can be a *unit-tested invariant* (the
ceiling formula, the gate arithmetic, the baseline provenance, the SALT error
math) runs on the hosted CPU lane, so a regression in the *methodology* fails
fast and free. The *physical* numbers — and any claim of beating a competitor —
are fenced behind the pinned hardware, exactly as the
[v1.0 release gate](../../adr/0012-v100-release.md) requires for a
third-party-reproducible benchmark report.
