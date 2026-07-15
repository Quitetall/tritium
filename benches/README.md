# tritium-benches — perf harness (v0.30, ADR 0005 / WF-E)

Divan microbenchmarks + an end-to-end tokens/sec harness + a roofline / %-of-SOL
toolkit + the competitor-baseline regression gate for Tritium.

All GPU bench bodies are behind the `cuda` cargo feature; the end-to-end bench is
additionally gated on the BitNet 2B4T GGUF being present. The CPU mpGEMM, roofline,
and SALT codec benches always run (no GPU, no model).

## Layout

| bench | gate | what it measures |
|-------|------|------------------|
| `mpgemm` | always | CPU ternary mpGEMM (v0.10 backend), the always-on baseline number |
| `gpu_mpgemm` | `cuda` | GPU **add-only** (TQ2_0, tiled+simple) and **IMMA int8** mpGEMM over the BitNet shapes |
| `e2e` | `cuda` + model | **decode** + **prefill** tokens/sec on BitNet 2B4T, each coupled to an unchanged-perplexity assertion |
| `roofline` | always | the decode `bandwidth / model_bytes` ceiling + committed competitor baselines (pure arithmetic) |
| `salt_codec` | always | SALT V2 D2/B3/S34 pack + unpack across codec/group boundaries, plus matched S34-valid tensors through all three codecs |

The shared fixtures, shapes, roofline math, and the baseline/regression types live in
`src/lib.rs` (`tritium_benches`), unit-tested on every lane.

## Running

```sh
# CPU-only (no GPU, no toolkit) — runs CPU, roofline, and SALT codec benches:
cargo bench -p tritium-benches

# GPU lane (needs nvcc + an NVIDIA GPU). Compiles the cuda-gated benches:
cargo bench -p tritium-benches --features cuda --no-run     # compile check
cargo bench -p tritium-benches --features cuda              # run all
cargo bench -p tritium-benches --features cuda --bench gpu_mpgemm   # just the GPU mpGEMM sweep

# End-to-end tokens/sec (needs the model at
# ~/.cache/tritium-models/bitnet-2b4t-gguf/ggml-model-i2_s.gguf):
cargo bench -p tritium-benches --features cuda --bench e2e

# Host-only SALT V2 codec sweep (logical trits/sec + exact packed bytes/sec):
cargo bench -p tritium-benches --bench salt_codec
```

The GPU mpGEMM benches sweep `M ∈ {1,8,32,256,512}` × `(N,K) ∈ {2560,6912}²` — the
BitNet 2B4T linear-layer shapes. The add-only `mpgemm` auto-selects the **tiled**
decode kernel for `M ≤ 64` and the **simple** kernel for larger `M`, so the one
sweep covers both add-only kernels across the crossover.

## Roofline / % of SOL

"% of SOL" (Speed-Of-Light) is the achieved fraction of the hardware ceiling for a
kernel's *limiting* resource. ADR 0005 splits it by regime:

### Decode (batch-1) — memory-bound

A decode step streams every weight from HBM exactly once, so it is bandwidth-bound:

```
decode_tok/s  ≤  peak_HBM_bandwidth / model_weight_bytes
```

On the pinned **RTX 4090** (peak HBM BW = **1008 GB/s**, GDDR6X 384-bit × 21 Gbps)
with **BitNet 2B4T** in the I2_S GGUF packing (**1 187 801 280 B ≈ 1.106 GiB**):

```
ceiling = 1008e9 / 1_187_801_280  ≈  848.6 tok/s
```

`bitnet_2b4t_decode_ceiling()` computes exactly this; the `roofline` bench prints it.
The e2e bench divides its **measured** decode tok/s by this ceiling to report the
%-of-roofline. ADR 0005 targets decode within ~10% of the peak HBM bandwidth
(≥ ~90% of DRAM-throughput SOL) and end-to-end tok/s within ~80–90% of this ceiling.

Measure the achieved HBM fraction with `ncu`:

```sh
# DRAM throughput as a % of SOL over the decode kernels of one forward step.
# (Run the acceptance/e2e binary so the real model decode launches.)
ncu --set roofline \
    --section MemoryWorkloadAnalysis \
    --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed \
    cargo test -p tritium-nn --features cuda --release cuda_greedy_matches_transformers
```

The `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` metric is the decode
% of SOL directly. The gate wants it ≥ ~90% on the tiled add-only kernel.

### Prefill (large-M) — compute-bound

Large-M prefill is bound by the int8 tensor-core throughput, not bandwidth:

```
prefill_tok/s  ≤  peak_int8_TOPS / macs_per_token
```

RTX 4090 peak **dense INT8 = 660.6 TOPS** (no sparsity;
`RTX_4090_PEAK_INT8_TOPS`). Measure the achieved fraction + tensor-pipe activity:

```sh
# IMMA (mma.m16n8k32) tensor-pipe active % + achieved int8 throughput on the
# prefill kernels.
ncu --set full \
    --metrics \
sm__pipe_tensor_op_imma.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_tensor_op_imma.sum \
    cargo bench -p tritium-benches --features cuda --bench gpu_mpgemm -- gpu_imma
```

`sm__pipe_tensor_op_imma.avg.pct_of_peak_sustained_active` is the tensor-op-active%
the gate wants high during prefill.

> This wave provides the **harness + the commands + the ceiling math**; it does not
> run `ncu` (no profiling pass is part of the bench). Run the commands above on the
> GPU lane to fill in the achieved %-of-SOL.

## Competitor baselines + the >5% regression gate

ADR 0005 requires tokens/sec ≥ parity with bitnet.cpp (floor `1.0×`, target `≥1.2×`)
and a scheduled lane that **fails on a >5% tokens/sec drop** vs a recorded baseline.

`src/lib.rs` commits the comparison points as `Baseline`s with explicit provenance
(`BaselineSource::BuiltOnBox` vs `Published(citation)`):

- `BITNET_CPP_2B4T_DECODE` — **28.0 tok/s** (published, conservative CPU figure;
  source: <https://github.com/microsoft/BitNet>).
- `LLAMA_CPP_2B4T_DECODE` — **18.0 tok/s** (published; source:
  <https://github.com/ggml-org/llama.cpp>).

These are committed as **published fallbacks**: building bitnet.cpp / llama.cpp on
the pinned box is best-effort (CMake + a C++ toolchain + a long compile). When a
local build succeeds *and can load the model*, replace the relevant constant with a
`BuiltOnBox` figure measured on the same 4090/model and note the swap in the commit
message.

> **WF-E build attempt (recorded).** Mainline `ggml-org/llama.cpp` **built cleanly**
> on this box (CPU-only Release) but **cannot load** this repo's
> `ggml-model-i2_s.gguf`: mainline GGUF reserves quant type-id `36` for the
> now-removed `IQ4_NL_4_4`, while this artifact uses id `36` for Microsoft's BitNet
> `I2_S` quant (a fork-specific assignment), so it errors with *"tensor
> 'blk.0.ffn_down.weight' of type 36 … not a multiple of block size (0)"*. The I2_S
> kernels live in the **bitnet.cpp fork**, not mainline — so the published figure is
> the committed comparison point for this exact artifact, per the plan's "if the
> build fails, record published numbers" fallback. To get a real number, build
> bitnet.cpp (below) or re-quantize the GGUF to a mainline ternary type (TQ1_0/TQ2_0).

To build the competitor locally:

```sh
# bitnet.cpp (Microsoft). Needs cmake, clang, and ~10 min.
git clone --recursive https://github.com/microsoft/BitNet.git
cd BitNet && pip install -r requirements.txt
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s
python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    -p "The capital of France is" -n 128   # reports tok/s
```

`check_regression(measured_tps, &baseline)` returns a `RegressionReport` whose
`regressed` flag is set iff `(baseline - measured) / baseline > 0.05`
(`REGRESSION_DROP_THRESHOLD`). A speedup never trips it. The scheduled CI lane (see
`.github/workflows/ci.yml`, `perf-regression`) runs the e2e bench and fails on a
tripped gate; the e2e bench also prints the verdict inline for local runs.
