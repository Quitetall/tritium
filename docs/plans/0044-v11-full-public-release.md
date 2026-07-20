# 0044 — Tritium v1.1 full public release

Status: **IN PROGRESS** (2026-07-20; decision/work-order slice)

- **Decision:** [ADR 0033](../adr/0033-v11-full-public-release.md)
- **Parallel empirical dependency:** [plan 0043](./0043-salt-v2-sota-campaign.md)
- **Release version:** `1.1.0`; existing `v1.0.0` remains immutable
- **Current claim status:** not achieved
- **Spend policy:** local-first; paid compute requires separate explicit approval

## Goal

Turn Tritium's existing kernels, training substrate and SALT V2 campaign engine
into a differentiable, installable, portable and production-ready ternary
research platform. Success means every ADR 0033 gate passes from published
artifacts in fresh environments and the pinned Qwen3.6-27B language-plus-MTP
proof is independently reproduced. Before activation, identical gates run on
signed local RC archives; registry publication is not a prerequisite for local
release readiness.

## Fixed decisions

- PyTorch/Hugging Face is the first public dynamic-training frontend.
- v1.2 ships a safe typed native Rust frontend plus Python bindings that run
  without PyTorch, against shared v1.1 schemas.
- v1.3 ships trainable whole-model ONNX; ONNX inference is v1.1.
- PTQ and refinement remain different runs, artifacts, costs and claims.
- Stable Python primitives are `prepare`, `calibrate`, `convert`, `refine`,
  `export`, `load`, and `inspect`; one-call QAT/PTQ functions are facades.
- Browser WebGPU training and the complete production/community suite block
  v1.1.
- Model zoo uses three audited tiers, not an arbitrary model count.
- Backend parity means the whole current Tape operation/optimizer manifest on
  CPU, CUDA, ROCm, Metal, wgpu, WASI/WASM, MCU and browser WebGPU. Semantics are
  common; performance and bounded shapes are capability-tiered.
- No paid 27B run is authorized by this plan.

## Dependency order

```text
shared schema fixtures + reference modules
        |
        +--> torch dispatcher/autograd --> phased HF/QAT/PTQ/refine --> Colab/zoo
        |
        +--> estimator registry --------> SALT production refinement --> 0043
        |
        +--> portable training spec ----> native backends --> WebGPU/TypeScript
        |
        +--> artifact/receipt contract -> ONNX/package/serve/deploy/observability
                                                       |
                                                       +--> release reproduction
```

Plan 0043 may continue in parallel through structural/local gates. No 27B spend
occurs until production driver, recipe, resume and stop gates are frozen.

## Slice map

Each numbered child plan is one reviewed commit or a small explicitly bounded
series. Child plans must contain exact edits, tests, failure branches and
hardware labels before execution.

| Plan | Deliverable | Entry gate | Exit gate |
|---|---|---|---|
| 0045 | Pure-PyTorch `TernaryLinear` and reference conversion; initial config/error/coverage types | ADR 0033 committed | forward/gradient/state/coverage tests; **done**, but cross-language schema freeze remains 0049 |
| 0046 | Zero-copy Torch dispatcher ops for CPU/CUDA; fake/meta/autocast/compile | 0045 | no-host-transfer profile; opcheck/grad parity |
| 0047 | HF quantizer plus phased `prepare`/`calibrate`/`convert`/`refine`/`export`/`load`/`inspect`; QAT/PTQ facades; Trainer/Accelerate | 0046 | independent QAT, PTQ and refined e2e; tied weights, DDP/FSDP, resume |
| 0048 | Estimator catalog/plugins and production SALT block reconstruction; separately typed scale-only/PV/S34 refinement; baseline harness | 0045 + plan-0043 driver seams | recipe ablations, lineage separation and hard-artifact parity |
| [0049](./0049-portable-training-manifest.md) | Canonical Rust/Python/TypeScript schema fixtures; exhaustive `TrainingOpManifestV1`; CPU/CUDA/ROCm/Metal/wgpu/WASI/MCU implementation | 0045 types | unknown-schema gates plus per-backend forward/VJP/optimizer/checkpoint/export receipts |
| [0050](./0050-web-training-session.md) | `@tritium-ai/web` compiled TypeScript session, WASM orchestration and whole-manifest WebGPU training | 0049 schema/manifest freeze | strict-TS package plus Chrome/Firefox/Safari WebGPU and cross-backend artifact parity |
| [0051](./0051-onnx-packaging-colab.md) | Whole-model ONNX inference, wheels/crates/PyPI/npm, compatibility matrix and Colab | 0046 + 0047 + 0049 schemas | local-RC fresh-env gates, then authorized registry smoke |
| 0052 | Hardened serving, OCI/Helm/KEDA/Knative, auth, observability, failure injection | stable artifact/load API | deployment e2e and rollback gates |
| 0053 | Guides, governance/community, three-tier zoo, independent reproduction and release | 0043 + 0045–0052 | all ADR 0033 boxes green; local signed RC, authorized activation, post-publish smoke |

Plans 0050 and 0051 are active. Plans 0052–0053 remain reserved work-order
numbers, not completed or executable documents. Each file must be written and accepted
before its implementation starts; this index must not link a nonexistent plan.

## Cross-cutting contracts

Every child plan preserves these rules:

1. **Fail closed.** Unsupported tensor/module/backend/fallback is an error unless
   policy explicitly permits and receipts it.
2. **Physical truth.** Serialized, resident, metadata, preserved and scratch
   bytes are separate measured fields. Logical trits never authorize a claim.
3. **No hidden dense path.** Claimed ternary inference/training cannot retain or
   reconstruct a dense weight shadow during steady state.
4. **No partial publication.** Conversion/export/package writes are staged,
   validated, hashed and atomically published.
5. **Reproducible identity.** Source, tokenizer, data, recipe, code, backend,
   seed and hardware identities bind every result.
6. **PTQ/refined separation.** `TernaryConfig.ptq` contains no refinement
   choice. PTQ, scale-only and hard-PV results have distinct types, parents and
   claims.
7. **Reference first.** Pure Rust/PyTorch reference behavior lands before an
   optimized adapter and remains its conformance oracle.
8. **Review every commit.** After each commit, run lamu DeepSeek V4 Pro commit
   review, verify every cited finding in source, fix real defects and re-review.

## Acceptance matrix

### PyTorch/Hugging Face

- `TernaryLinear` matches reference forward and finite-difference/analytic
  first-order gradients for activations, latent masters and scales.
- Recursive conversion preserves qualified names, tied weights, bias, dtype,
  device, train/eval state and state-dict round-trip.
- `torch.library.opcheck`, fake/meta, autocast and `torch.compile` pass supported
  CPU/CUDA matrices without graph breaks.
- Profiler sees zero steady-state H2D/D2H copies on CUDA and wrapper overhead is
  within 5% of direct Tritium execution after warmup.
- `prepare` → calibrate or ordinary QAT step → `convert` → optional `refine` →
  `export`/reload/generate works with Trainer/Accelerate DDP/FSDP. QAT, PTQ and
  refined lineages have independent tests and receipts.

### Algorithms and artifacts

- Built-in catalog covers AbsMean/annealed STE, LSQ, TWN, TTQ, sparse ternary,
  SALT additive PTQ and separate refinement.
- External estimator passes contract validation, gradient test, training and
  export without Tritium source edits.
- Block/sliding reconstruction, teacher-logit final-block loss, output-aware
  initialization, true alternating hard refinement and S34 refinement have
  matched-byte ablations.
- Hard exported package reload matches refinement model within frozen tolerance;
  no training residual or unreceipted tensor remains.

### Portable and browser training

- An exhaustive registry proves `TrainingOpManifestV1` covers every current
  public Tape op/VJP, Conv2d, SGD, AdamW, CautiousAdamW, Int8AdamW, Muon,
  checkpoint/resume and export/reload.
- CPU, CUDA, ROCm, Metal, wgpu, WASI/WASM and MCU run frozen vectors on actual
  targets; constrained profiles use bounded shapes, never skipped tests.
- `npm pack` installs into an empty strict-TypeScript project and passes type
  check, production bundle and the full session lifecycle. Chrome, Firefox and
  Safari each prove actual WebGPU; fallback never greens that gate.
- Capability and performance tables are generated from receipts, not prose.

### Interop, package and operations

- Supported whole-model ONNX export/import executes through a real ORT session;
  unsupported gradient import says trainable ONNX is v1.3.
- Local RC wheels/crates/npm/container archives install without repository or
  compiler and run their smoke suites. Registry publication and community
  creation require explicit activation approval; exact published artifacts then
  run a separate smoke.
- Colab tutorial finishes within five minutes on the tiny model, excluding first
  model download.
- Serve tests cover auth, limits, backpressure, cancellation, shutdown,
  readiness, telemetry and malformed artifacts.
- Helm/KEDA/Knative deploy, scale, restart and roll back a pinned OCI image;
  Grafana/Prometheus/OpenTelemetry expose documented ternary/runtime metrics.

### Model zoo and SOTA evidence

- SmolLM2-135M completes local/Colab/browser pipeline; SmolLM2-1.7B freezes recipe.
- BitNet b1.58 2B4T retains v1.0 parity/perplexity/performance.
- Qwen3.6-27B converts all in-scope language/MTP matrices under exact coverage;
  PTQ and refined artifacts, quality, runtime, memory, physical-size and MTP
  receipts pass plan 0043.
- Fresh second-machine reproduction regenerates model-card tables from receipts.

## Verification cadence

Run during each slice:

```bash
cargo fmt --check
cargo test -p <changed-crate>
cargo clippy -p <changed-crate> --all-targets -- -D warnings
```

Run Python/JS/backend-specific focused suites named by child plan. Before release:

```bash
cargo test --workspace --no-fail-fast
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo deny check
./scripts/check-semver.sh
```

GPU, browser, multi-GPU and platform tests must produce actual device receipts.
Skipped or zero-test hardware lanes are failures, not green gates.

## Stop conditions

- Stop and amend ADR before weakening a public quality, coverage, physical-byte,
  backend-parity or portability gate.
- Stop before any paid compute, hosted service purchase, PyPI/crates/npm publish,
  container registry push, Discord/community creation or final tag until Brian
  Lam explicitly authorizes that external action.
- Record a failed SOTA hypothesis as evidence; do not tune acceptance thresholds
  after observing flagship results.

## Done criterion

Every ADR 0033 v1.1 checkbox is green; plan 0043 flagship evidence is admitted;
all child plans are reviewed and committed; local RC artifacts reproduce in
fresh environments; version is `1.1.0`; changelog/migration/compatibility/model
cards match shipped behavior; clean signed release commit and tag candidate
exist locally. Final publication, public tag/push and community activation occur
only with explicit authorization, followed by registry smoke receipts.

## Commit

Decision reconciliation:

```text
docs(adr-0033): reconcile Tritium v1.1 release contracts
```
