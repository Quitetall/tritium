# Model Zoo

This chapter is about **what Tritium actually loads** — which on-disk model
formats the loader accepts, the one reference model the test/perf gates exercise,
and how to drive a compatible GGUF through the [`tritium` CLI](./quickstart.md).
It is deliberately conservative: a model is listed as **verified-compatible**
only where the repo's tests run it end-to-end, and as **expected-compatible**
where the loader's type handling implies it should work but no committed test
exercises that exact artifact. The source of truth is the loader
(`crates/tritium-nn/src/model/weights.rs`) and the GGUF reader
(`crates/tritium-format/src/gguf.rs`) — when this page and the code disagree, the
code wins.

<!-- BEGIN TRITIUM GENERATED RELEASE CLAIMS -->
## Audited v1.1 admission ladder

| Tier | Role | Frozen model | Admission rule |
|---|---|---|---|
| `accessible` | `tutorial` | `HuggingFaceTB/SmolLM2-135M-Instruct` | candidate artifact + admitted receipt ancestry |
| `accessible` | `recipe` | `HuggingFaceTB/SmolLM2-1.7B` | candidate artifact + admitted receipt ancestry |
| `native-reference` | `native` | `microsoft/bitnet-b1.58-2B-4T` | candidate artifact + admitted receipt ancestry |
| `flagship` | `language+mtp` | `Qwen/Qwen3.6-27B` | candidate artifact + admitted receipt ancestry |

The generated ladder names release targets only. The model-zoo receipt
binds exact revisions, tokenizer digests, cards, candidate artifacts and
evidence receipts; absent evidence remains `MISSING`.
<!-- END TRITIUM GENERATED RELEASE CLAIMS -->

## What the loader accepts

`ModelRunner::load` (and the convenience `ModelRunner::load_cpu`) read a parsed
GGUF container and map ggml tensor type-ids to internal storage. Six
type-ids are consumed by the BitNet loader (`crates/tritium-nn/src/model/weights.rs`):

| ggml type-id | name | role | how it is stored |
|--------------|------|------|------------------|
| `36` | `I2_S` | the ternary linear weights (`attn_q/k/v/output`, `ffn_gate/up/down`) | unpacked to `{-1, 0, +1}` trits + a per-tensor f32 scale, then **re-packed as `TQ2_0`** and uploaded to the backend |
| `34` | `TQ1_0` | group-256 ternary linear weights | decoded, checked for row-compatible scales, then packed into the selected backend layout |
| `35` | `TQ2_0` | group-256 ternary linear weights | decoded, checked for row-compatible scales, then packed into the selected backend layout |
| `42` | standard `Q2_0` | group-64 ternary linear weights | retained packed by `Q2Linear`; varying finite G64 scales execute exactly through the portable A8 path without a dense weight shadow |
| `0` | `F32` | norms (`attn_norm`, `ffn_norm`, the sub-norms, `output_norm`) | kept host-side as fp32 |
| `1` | `F16` | the token embedding (`token_embd.weight`) | widened to fp32 |

The loader dispatches on explicit GGUF type IDs; it does not sniff bytes.
`I2_S`/`TQ1_0`/`TQ2_0` normalize into the backend's selected compute layout.
Standard Q2_0 retains its group-64 scales and packed payload through a portable
projection because collapsing four G64 scales into one G256 scale would be lossy.
This path is correctness/interchange coverage, not yet a backend-native Q2_0
performance claim.

> The `tritium inspect` tool labels `F32`, `F16`, `TQ1_0` (id `34`), `TQ2_0`
> (id `35`), and standard `Q2_0` (id `42`) by name; other ids print as `type N`, so an `I2_S` tensor
> shows up as `type 36`. Inspect is a container dump; it does **not** imply the
> runner can load every type it can name.

### The tied LM head

The reference model sets `tie_word_embeddings = true` and ships **no**
`output.weight` tensor. The runner unembeds against the same `token_embd` matrix
it read for input embedding — a dense fp32 matmul. A checkpoint that *does* carry
a separate output projection is not what the loader expects here.

## The reference model — BitNet b1.58 2B4T

Tritium's accuracy and performance gates are written against **Microsoft's
[BitNet b1.58 2B4T](https://arxiv.org/abs/2402.17764)** in the `I2_S` GGUF
packing — the file `ggml-model-i2_s.gguf` (1 187 801 280 bytes ≈ 1.106 GiB; the
exact byte count is committed as `tritium_benches::BITNET_2B4T_I2S_BYTES` and is
the denominator of the decode roofline — see [Benchmarks](./benchmarks.md)).

This is the **only** model the repo runs end-to-end. The geometry the code is
built around (`crates/tritium-nn/src/config.rs`,
`crates/tritium-nn/tests/bench_cpu_hotpaths.rs`): `n_embd = 2560`,
`feed_forward_length = 6912`, `n_head = 20`, `n_head_kv = 5` (GQA),
`head_dim = 128`, 30 transformer blocks, a ReLU² MLP, and `attn_sub_norm` /
`ffn_sub_norm` sub-normalisations — the BitNet b1.58 layer shape.

### End-of-sequence token

Greedy generation stops at the EOS token. The CLI default is **`128001`**
(`DEFAULT_EOS` in `crates/tritium-cli/src/main.rs`) — the **LLaMA-3
end-of-text** id BitNet 2B4T inherits from the LLaMA-3 tokenizer. Override it
with `--eos <ID>` on `generate` and `report parity`.

### Verified-compatible vs expected-compatible

| Model / format | Status | Where it is exercised |
|----------------|--------|------------------------|
| **BitNet b1.58 2B4T**, `I2_S` GGUF, **CPU** backend | **Qualification pending (physical receipt)** | `crates/tritium-nn/tests/fidelity_ladder.rs` (stage-by-stage CPU forward vs a `transformers` oracle) and `acceptance.rs::cpu_longer_greedy_matches_transformers` (greedy IDs vs the committed reference) — both model-gated and self-skipping when the GGUF is absent; no committed physical receipt, so status remains `MISSING`. |
| **BitNet b1.58 2B4T**, `I2_S` GGUF, **CUDA** backend | **Qualification pending (physical receipt)** | `acceptance.rs` (`cuda_greedy_matches_transformers` — full 256-token greedy match; perplexity within 1% of the `transformers` reference; CPU↔CUDA logit parity). Requires real CUDA device + model and a committed receipt; self-skipping is not qualification evidence. |
| Any other `I2_S` GGUF with the BitNet 2B4T layer layout | **Expected-compatible** | The loader path is type-driven, not file-specific, so an `I2_S` checkpoint with the same tensor names and the supported `F32`/`F16` companions should load — but no committed test pins a *different* artifact, so it is not claimed as verified. |
| A SALT-quantized GGUF written by `tritium quantize ... --format gguf` | **Expected-compatible (SALT path)** | `tritium-format` can write a SALT bundle into a GGUF (`crates/tritium-format/src/salt_gguf.rs`); see [Quantization](./quantization.md). This is the producer side; loading such an artifact back through `ModelRunner` is not part of the committed acceptance gates. |
| BitNet b1.58 2B4T repacked as native `TQ1_0`/`TQ2_0` | **Qualification pending (real-artifact gate)** | The explicit real-model gate proves exact logits when run with the pinned external GGUF; absent artifact/receipt remains `MISSING`. |
| BitNet b1.58 2B4T repacked as standard group-64 `Q2_0` | **Qualification pending (gated token interop)** | `q2_0_interop.rs` binds Q2_0/TQ2_0 artifact and prompt digests, decodes paired ternary tensors, requires byte-identical non-ternary payloads, and compares greedy tokens. Its v2 receipt labels truncated runs `smoke`; only the frozen three-distinct-prompt × 16-token profile can qualify. |

> **No bundled weights.** The repo ships **no** model file — the 2B4T GGUF is an
> external download (it lives at
> `~/.cache/tritium-models/bitnet-2b4t-gguf/ggml-model-i2_s.gguf` on the
> developer box the gates were recorded on). Every test that needs it is
> model-gated and skips cleanly when it is missing, which is why the default CPU
> CI is green without any weights present.

## Obtaining a compatible GGUF

The reference checkpoint and its `I2_S` GGUF are published by Microsoft. The
canonical way to obtain the GGUF is the BitNet repository's setup, which converts
the released weights to the `I2_S` quantisation Tritium loads:

```sh
# Microsoft's BitNet tooling produces ggml-model-i2_s.gguf from the 2B4T release.
git clone --recursive https://github.com/microsoft/BitNet.git
cd BitNet && pip install -r requirements.txt
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s
# -> models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
```

This is the same artifact the benchmark harness references; see the recorded
loadability notes in [Benchmarks](./benchmarks.md) for why mainline `llama.cpp`
cannot load this exact `I2_S` file (a GGUF quant type-id collision).

## Running it through the CLI

Build the CLI once (`cargo build -p tritium-cli`) — the rest assumes `tritium` on
`PATH`. See [Quickstart](./quickstart.md) for the full subcommand surface; the
source of truth for flags is `crates/tritium-cli/src/main.rs`.

### Inspect the container

`inspect` parses the GGUF and prints version, metadata, architecture, alignment,
and the tensor table (name, dims, type, offset, size). A missing or corrupt file
errors cleanly and exits non-zero — it never panics.

```sh
tritium inspect ggml-model-i2_s.gguf
```

Use this first to confirm a downloaded file really is the expected container
(architecture, tensor count, and that the linear weights show as `type 36`).

### Generate from token IDs

`generate` loads the GGUF and **greedily** decodes from a reproducible JSON file
of input token IDs, stopping at EOS (default `128001`). Input is token IDs, not
text — Tritium's CLI does not bundle a tokenizer, so you provide pre-tokenised
ids (e.g. `[1, 128000, 9906]`):

```sh
# tokens.json holds a JSON array of u32 token IDs.
tritium generate --model ggml-model-i2_s.gguf --tokens tokens.json --max-new 16
```

Flags: `--max-new <N>` (default 16), `--eos <ID>` (default 128001),
`--greedy <bool>` (only `true` is accepted — sampling lives in
`tritium-serve`'s OpenAI API).

### Report decode / ttft / parity

The `report` subcommands emit reproducible JSON/table output — decode-only
throughput, time-to-first-token, and CPU-vs-CUDA greedy parity. They are covered
in detail (with their CPU-vs-GPU honesty boundaries) in
[Benchmarks](./benchmarks.md):

```sh
tritium report decode --model ggml-model-i2_s.gguf --tokens tokens.json \
  --backend cpu --decode-steps 8 --warmup 1
```

> Backends are discovered through the `linkme` `BACKENDS` registry (see
> [Architecture](./architecture.md#the-registry-linkme-self-registration)), so a
> plain build exposes `cpu`; building the CLI `--features cuda` makes `cuda`
> selectable for `report --backend cuda`.

## Caveats and pre-1.0 status

- **Token-ID interface.** The CLI consumes/produces token IDs, not text. A
  tokenizer is the caller's responsibility.
- **One verified model.** The acceptance gates are written for BitNet 2B4T in
  `I2_S`. Other architectures are out of scope until a test pins them.
- **Pre-1.0.** A real-model, fresh-environment capstone (download → infer →
  SALT-quantize → fine-tune) is a **v1.0 exit gate** that requires hardware this
  book's CI does not have; it is tracked in
  ADR 0012 (see the [research repository](https://github.com/Quitetall/tritium-research)) and is **not** claimed complete here.
