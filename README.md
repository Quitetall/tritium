<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/tritium-header-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/brand/tritium-header-light.svg">
    <img alt="Tritium" src="assets/brand/tritium-header-dark.svg" width="420">
  </picture>
</p>

[![CI](https://github.com/Quitetall/tritium/actions/workflows/ci.yml/badge.svg)](https://github.com/Quitetall/tritium/actions/workflows/ci.yml)
[![Docs](https://github.com/Quitetall/tritium/actions/workflows/docs.yml/badge.svg)](https://github.com/Quitetall/tritium/actions/workflows/docs.yml)
[![Sanitizers](https://github.com/Quitetall/tritium/actions/workflows/sanitizers.yml/badge.svg)](https://github.com/Quitetall/tritium/actions/workflows/sanitizers.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2024-orange.svg)](Cargo.toml)
[![Benchmarks](https://img.shields.io/badge/benchmarks-reproducible_ledger-informational.svg)](docs/BENCHMARKS.md)
[![docs.rs](https://img.shields.io/docsrs/tritium-core?logo=rust&label=docs.rs)](https://docs.rs/tritium-core)
[![crates.io](https://img.shields.io/crates/v/tritium-core.svg?logo=rust)](https://crates.io/crates/tritium-core)

<p align="center">
  <em>Ternary weights, honestly measured.</em><br>
  <code>-1</code> &nbsp;<code>0</code>&nbsp; <code>+1</code>
</p>

Tritium is Apache-2.0 infrastructure for converting dense neural networks into
compact additive-ternary models, refining or training them with PyTorch, and
executing their packed weights across native and portable runtimes.

Weights use `{-1, 0, +1}` planes. One plane is the compact base model; additional
residual planes trade physical bytes for quality. Tritium treats the recipe,
calibration evidence, ancestry, packed bytes, runtime memory, and measured model
quality as one auditable artifact lifecycle—not as an informal `bits=2` label.

> **Release status:** the repository uses the `1.1.0-rc.2` candidate version. It
> is neither `LOCAL_RC_READY` nor a public v1.1 release. Package, browser,
> flagship-model, deployment, and second-machine gates remain open. The generated
> [compatibility matrix](docs/compatibility.md) is authoritative: `pending` is
> not support, and no public registry artifact is claimed before activation.

<!-- BEGIN TRITIUM GENERATED RELEASE CLAIMS -->
### Audited v1.1 model ladder

| Tier | Role | Frozen model | Admission rule |
|---|---|---|---|
| `accessible` | `tutorial` | `HuggingFaceTB/SmolLM2-135M-Instruct` | candidate artifact + admitted receipt ancestry |
| `accessible` | `recipe` | `HuggingFaceTB/SmolLM2-1.7B` | candidate artifact + admitted receipt ancestry |
| `native-reference` | `native` | `microsoft/bitnet-b1.58-2B-4T` | candidate artifact + admitted receipt ancestry |
| `flagship` | `language+mtp` | `Qwen/Qwen3.6-27B` | candidate artifact + admitted receipt ancestry |

This table is an admission inventory, not a support or SOTA claim. A row
becomes releasable only when its exact candidate artifact, immutable source
and tokenizer identity, model card, and required evidence ancestry validate.
<!-- END TRITIUM GENERATED RELEASE CLAIMS -->

## What is implemented

- Additive PTQ with one to three ternary planes, calibration receipts, strict
  physical-byte accounting, packed artifacts, scale-only refinement, and hard-PV
  refinement.
- Differentiable PyTorch QAT with STE estimators, tied-weight preservation,
  hard conversion, optimizer-compatible latent masters, strict checkpointing,
  Hugging Face integration, and ternary diagnostics.
- Packed inference kernels and runtime adapters for CPU, CUDA, Metal, ROCm,
  wgpu, WASI/WASM, browser WebGPU, C, Candle, Burn, and ONNX at different
  qualification levels.
- Native SALT formats plus compact B3 packing, sparse/additive layouts,
  seek-backed payloads, canonical manifests, digests, and bounded readers.
- OpenAI-compatible serving with bounded requests, rotating bearer
  authentication for exposed binds, per-principal admission, readiness,
  metrics, hardened OCI/Kubernetes assets, and evidence-gated deployment.
- Receipt-backed release tooling for wheels, crates, npm archives, OCI images,
  SBOMs, provenance, compatibility, model evidence, and local-RC sign-off.

Implementation is not the same as release qualification. See
the release ADR and execution plan in the
[research repository](https://github.com/Quitetall/tritium-research) for the gates (see [docs/RESEARCH.md](docs/RESEARCH.md)).
The [backend guide](docs/book/src/backends.md) describes source capabilities;
an implementation without a generated compatibility cell and admitted receipt
remains unqualified.

## PyTorch PTQ in explicit phases

The public API exposes preparation, calibration, conversion, refinement,
export, load, and inspection as distinct typed phases:

```python
from pathlib import Path

import torch
from tritium.torch import (
    TernaryConfig,
    calibrate,
    convert,
    inspect,
    load_quantized_module,
    prepare,
)

dense = torch.nn.Linear(128, 64).eval()
recipe = TernaryConfig.ptq(
    profile="compact-v1",
    target_modules=("Linear",),
)
prepared = prepare(dense, recipe, inplace=False)
calibration = calibrate(
    prepared,
    [torch.randn(4, 128) for _ in range(4)],
    evidence_dir=Path("work/calibration"),
)
result = convert(prepared, calibration, work_dir=Path("work/conversion"))
ternary = load_quantized_module(dense, result, inplace=False).eval()

print(result.artifact_id)
print(inspect(prepared))
print(ternary(torch.randn(2, 128)).shape)
```

For QAT, Hugging Face, refinement, ONNX, and browser workflows, use the
[book](docs/book/src/SUMMARY.md). The pinned SmolLM2 tutorial is generated and
headless-testable, but its clean candidate-wheel/Colab receipt is still a release
gate rather than a published quick-start claim.

## Native inference from source

The default workspace build is CPU-only and uses Cargo—no CMake:

```sh
cargo build --locked
cargo test --locked -p tritium-core -p tritium-format -p tritium-cpu
cargo run --locked -p tritium-cli -- list-backends
```

CUDA and other accelerator adapters are feature-gated because their toolchains
and physical device receipts are target-specific. Portable targets use their
own target-build and package gates. The source-level CLI and server tutorial is
in the [quickstart](docs/book/src/quickstart.md); the engine's environment
knobs are catalogued in
[environment variables](docs/book/src/environment.md). Release users should
install only exact artifacts admitted by the candidate manifest once those
artifacts are authorized and published.

### Install from crates.io (release candidate)

Only prerelease versions are published, so `cargo install` needs the explicit
version:

```sh
cargo install tritium-cli --version 1.1.0-rc.0            # the `tritium` tool
cargo install tritium-serve --version 1.1.0-rc.0 \
    --features serve                                       # the HTTP server
```

Add `--features cuda` to either for the CUDA backend (needs nvcc).

## Architecture

Dependencies point inward:

```text
foundation  (core · format · spec · testkit)
   ↑
backends    (cpu · cuda · metal · rocm · wgpu · wasm · mcu)
   ↑
runtime     (runtime · quantize · salt · nn · train)
   ↑
frontends   (ffi · python · onnx · candle · burn · cli · serve · web)
```

A frontend does not own a concrete backend, and one backend does not depend on
another. Shared schemas and conformance vectors define semantic portability;
physical device claims additionally require target receipts.

## Evidence and limitations

The v1.0 codebase has real BitNet b1.58 2B4T CPU/CUDA evidence recorded in the
[changelog](CHANGELOG.md). The v1.1 flagship target is a separately labeled
Qwen3.6-27B language-plus-MTP additive-PTQ/refinement campaign. It is not a
completed SOTA claim until its frozen quality, retention, runtime, memory, and
reproduction gates pass.

The `scripts/release-status` validator consumes an explicit candidate manifest
and evidence registry and refuses incomplete gates. This working tree does not
yet contain a sign-off candidate/registry. Current work-order blockers include:

- whole-model Qwen/MTP quality and performance evidence is not yet admitted;
- public wheel/crate/npm/container artifacts are not activated;
- browser support awaits physical Chrome, Firefox, and Safari receipts;
- whole-model Qwen/MTP inference and clean-wheel ONNX Runtime generation
  receipts remain open;
- trainable whole-model ONNX is intentionally deferred to v1.3;
- vision/multimodal tensors are identity-bound but deferred from v1.1;
- community channels are not yet activated; see [COMMUNITY.md](COMMUNITY.md) for the
  activation and moderation policy.

## Project policies

- [Contributing](CONTRIBUTING.md)
- [Governance](GOVERNANCE.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Support](SUPPORT.md)
- [Security](SECURITY.md)
- [Community channels](COMMUNITY.md)
- [Citation](CITATION.cff)

Tritium is licensed under [Apache-2.0](LICENSE). Vendored upstream work is
attributed in [NOTICE](NOTICE).
