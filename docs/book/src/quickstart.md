# Quickstart

For differentiable PyTorch training from a binary wheel, use the
[installed-wheel QAT tutorial](./tutorial-pytorch-qat.md). It runs without a
source checkout or compiler and strict-reloads its hard artifact.

## Chat with a model in three commands

```sh
cargo build --release -p tritium-cli -p tritium-serve --features tritium-serve/cuda
target/release/tritium pull microsoft/bitnet-b1.58-2B-4T-gguf
target/release/tritium-serve \
  --model ~/.cache/tritium-models/microsoft--bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --backend cuda
```

Then talk to it — the tokenizer travels inside the GGUF, so this is real text
over the OpenAI wire (point LM Studio, Open WebUI or `curl` at it):

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"tritium","messages":[{"role":"user","content":"What is the capital of France?"}]}'
```

No GPU? Use `--backend cpu` (and build without the `cuda` feature). No
network exposure by default: the server binds loopback; `--host` beyond
loopback refuses to start unless `TRITIUM_AUTH_TOKEN` or the comma-separated
rotation set `TRITIUM_AUTH_TOKENS` is set. Expensive generation routes use a
fixed-cardinality per-key token bucket by default (`--rate-limit-rpm 120`,
`--rate-limit-burst 8`; set RPM to `0` only for a trusted local deployment).

### Docker

The repo ships separate hardened CPU/CUDA definitions under `deploy/oci`
(pinned build image → shell-free distroless runtime,
model as a bind mount):

```sh
(
set -e
test -z "$(git status --porcelain=v1 --untracked-files=all)" # exact developer source
revision=$(git rev-parse HEAD)
created=$(git show -s --format=%cI HEAD)
docker build -f deploy/oci/Dockerfile.cuda \
  --build-arg SOURCE_REVISION="$revision" --build-arg SOURCE_CREATED="$created" \
  -t tritium-serve .
docker run --rm --gpus all --user 10001:10001 --read-only --cap-drop ALL \
  --security-opt no-new-privileges --tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m \
  -p 127.0.0.1:8080:8080 \
  -e TRITIUM_AUTH_TOKEN=change-me \
  -v ~/.cache/tritium-models:/models:ro \
  tritium-serve --model /models/microsoft--bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --backend cuda --host 0.0.0.0
)
```

The explicit non-loopback bind makes the `-p` mapping reachable and therefore
requires the token; the server refuses to start without it.

Release candidates use `scripts/build-oci-candidate`, which additionally
requires a clean tree, vendors the locked dependency graph, disables build
network access, and emits an attested OCI archive. See `deploy/oci/README.md`.

## Building from source

Tritium builds with Cargo only — no CMake. The default build is CPU-only;
backends and frontends are gated behind feature flags so the lean build stays
free of CUDA, `wgpu`, the framework adapters, and the server stack.

## Build and test

```sh
cargo build                       # CPU-only foundation + cpu backend
cargo test -p tritium-core        # reference math + roundtrip
cargo test --workspace --exclude tritium-py   # the full default-feature suite
```

`tritium-py` is excluded from `cargo test` because it is a PyO3 extension module
(a `cdylib` with no cargo tests) — it is exercised via maturin + pytest, not
`cargo test`.

## The `tritium` CLI

The `tritium` binary (crate `tritium-cli`) is the quickest way to poke at a
model and the backend registry. Build and run it with `cargo run -p tritium-cli
-- <subcommand>`, or build it once with `cargo build -p tritium-cli` and run
`target/debug/tritium`.

```sh
cargo run -p tritium-cli -- --help
```

The subcommands:

### `pull`

Download a GGUF from the HuggingFace hub into `~/.cache/tritium-models`
(override with `TRITIUM_MODEL_CACHE`). One `.gguf` in the repo → pulls it;
several → lists them and asks for `--file`; interrupted downloads resume on
re-run (HTTP Range on the kept `.part` file).

```sh
tritium pull microsoft/bitnet-b1.58-2B-4T-gguf
tritium pull bartowski/SmolLM2-135M-Instruct-GGUF --file SmolLM2-135M-Instruct-Q4_K_M.gguf
```

### `inspect`

Parse a GGUF container and print a summary of its version, metadata,
architecture, alignment, and tensor table. A missing, short, or corrupt file
prints a clean error and exits non-zero rather than panicking.

```sh
tritium inspect model.gguf
```

### `list-backends`

Enumerate every backend the runtime discovered, with its capabilities. Because
backends self-register through the `linkme` `BACKENDS` slice (see
[Architecture](./architecture.md)), the `cpu` backend always appears; building
the CLI with `--features cuda` links `tritium-cuda` and makes a `cuda` device
appear too.

```sh
tritium list-backends
```

### `generate`

Load a GGUF model and greedily decode tokens from a **reproducible JSON file of
input token IDs**. Generation is greedy (the deterministic decode strategy) and
stops at the EOS token (default `128001`, the LLaMA-3 end-of-text token used by
BitNet 2B4T).

```sh
# tokens.json holds e.g. [1, 128000, 9906]
tritium generate --model model.gguf --tokens tokens.json --max-new 16
```

Flags: `--max-new <N>` (default 16), `--eos <ID>` (default 128001),
`--greedy <bool>`.

### `report`

Emit reproducible benchmark/validation reports. Subcommands:

- `report decode` — decode-only throughput after prefill.
- `report ttft` — time-to-first-token / prefill latency.
- `report parity` — CPU-vs-CUDA greedy parity.
- `report salt` — SALT bpw/error report for a flat JSON fp32 matrix.

Each takes `--format {both,json,table}` so reports are machine-readable.

```sh
tritium report decode --model model.gguf --tokens tokens.json \
  --backend cpu --decode-steps 8 --warmup 1
```

### `quantize`

SALT-quantize an fp `.safetensors` model to a SALT bundle (`.tslb`) or a GGUF
container. `--bpw` is the single accuracy↔size knob (`1.585` = all base ternary …
`~4.75` at T=3); see [Quantization](./quantization.md).

```sh
tritium quantize --input model.safetensors --output model.tslb --bpw 2.0
```

> The exact subcommand surface is defined in `crates/tritium-cli/src/main.rs`; if
> a flag here ever drifts, that file is the source of truth.

## Enabling a GPU backend

The CUDA backend is feature-gated and built from source: `build.rs` shells
`nvcc` to compile the `.cu` kernel, and the host side loads the PTX/cubin at
runtime via `cudarc`. Build it where a CUDA toolkit and an NVIDIA GPU are
present:

```sh
cargo test -p tritium-cuda --features cuda
```

The cross-platform `tritium-wgpu` backend (WGSL over Vulkan) and the
`tritium-wasm` backend follow the same feature-gated pattern — see
[Backends](./backends.md).
