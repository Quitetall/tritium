# Changelog

All notable changes to Tritium. Format loosely follows Keep a Changelog. From
**1.0**, the frozen **core** crates + the C ABI follow semver; the **evolving** tier
(`tritium-nn`/`-train`/`-cuda` + interop + `-serve`) may break in minor releases —
see `docs/v1.0-api-freeze-audit.md` for the tier policy.

> **Versioning:** SemVer (`MAJOR.MINOR.PATCH`) from **0.3.0** onward. The earlier
> tags `v0.10.0` / `v0.20.0` (the old `0.x0` milestone staircase) are immutable and
> correspond conceptually to 0.1.0 / 0.2.0.

## [Unreleased] — 1.x dev

### Changed

- **MSRV raised to 1.96, and it is now a policy rather than a constant.** `rust-version` was
  `1.89` — the version where AVX-512 intrinsics stabilised, i.e. a hard *lower bound*
  inherited as if it were a target. The floor now follows **latest stable minus two
  releases, reviewed each release, never below 1.89** (see `SUPPORT.md`). This is
  independent of `rust-toolchain.toml`, which pins the toolchain CI builds with.

> **Release-candidate mapping.** `1.1.0-rc.0` is published on crates.io for all 23
> crates, and `v1.1.0-rc.1` is tagged. Both were cut from this section rather than
> from dated `1.1.0-rc.*` headings, so there is no separate entry to read for them —
> everything below is what shipped in those candidates. The next non-RC release
> promotes this section to a dated `## [1.1.0]` heading.

### Changed

- **Transactional WebGPU cancellation:** resident training now carries each
  operation's `AbortSignal` through GPU submission. Optimizer kernels retain
  their existing candidate owners, but signalled execution defers explicit
  root-owner commit copies until submitted compute finishes uncancelled.
  Cancellation after compute submission therefore returns a recoverable
  failure without changing parameters or optimizer state; retry remains valid.
  Calls without a signal keep the one-command-buffer fast path.
- **Compiled autocast parity:** `tritium.torch.ternary_linear` now makes its
  selective CPU/CUDA autocast casts visible to `torch.compile`. Compiled CUDA
  graphs preserve the fp32 master weight for resident native kernels while
  casting only activation-facing tensors to fp16, matching eager execution
  instead of silently falling back to fp32 composite projection. Compiled
  first-order backward uses an opaque dispatcher VJP boundary so native packed
  caches remain available without tracing storage pointers through fake tensors.
- **Python distribution rename:** install release candidates with
  `pip install tritium-torch`; the import namespace remains `import tritium`.
  This aligns packaging with ADR 0033 and avoids claiming the generic `tritium`
  distribution name. The Rust workspace and wheel now share the
  `1.1.0-rc.1` candidate version (rendered as PEP 440 `1.1.0rc1` in wheel
  metadata).
- **Stable API baseline:** `scripts/check-semver.sh` now compares the seven
  frozen Rust crates with the latest reachable stable SemVer tag (`v1.0.0` for
  this candidate), replacing the obsolete pre-freeze `v0.5.*` default. See the
  [v1.0 → v1.1 migration guide](docs/book/src/migration-v1.1.md).

### Added

- **Source-bound Stage-7 PyTorch data:** `Stage7TokenEvidencePack` now exposes
  strict, same-handle token-pack admission to the `tritium-torch` wheel without
  expanding token payloads into Python scalar lists. `Stage7CausalData.open`
  selects a bounded partition window, terminally rehashes the retained payload
  handle, records both ordered-member and raw-token identities, and yields
  replayable causal-LM batches. This is the production input boundary for the
  SmolLM2-135M smoke and 1.7B recipe driver; it does not claim capture, fitting,
  package, quality, or Stage-7 completion by itself.

- **Executable Stage-7 135M smoke:** `run_stage7_smoke_model` now performs and
  strictly resumes capture, additive PTQ fitting, explicit allocation, native
  SALT V2 packaging, and causal evaluation from terminally validated token
  evidence. `run_stage7_smollm2_smoke` additionally binds the exact frozen
  SmolLM2-135M source, tokenizer, 128-sequence C4 prefix, complete rank-2 tensor
  inventory, and qualifier-compatible execution/artifact receipts. SALT V2
  packages remain byte-identical version 1 for G128-only tensors; tensors whose
  row geometry requires G64 use canonical version 2 with an explicit scale
  geometry byte. Eager and seek-backed readers, streamed writing, CPU, compact
  host, physical CUDA exact/gather, and packed ONNX kernels have G64 parity
  coverage. This engineering path does not claim recipe quality, Stage-7
  qualification, or G64 campaign promotion; those still require frozen physical
  evidence and matched-quality gates.

- **Governed Stage-7 recipe freeze:**
  `scripts/qualify-stage7-recipe-freeze.py` validates the immutable
  SmolLM2-1.7B campaign inventory, disjoint calibration/evaluation provenance,
  successive-halving promotions, matched-byte output-aware curvature win,
  R3 gap closure, task retention, physical/native prerequisite receipts, and
  scale-only/short-PV token caps. It derives exact rank-2 matrix inventory from
  pinned safetensors, strictly reopens every SALT V2 package through the native
  parser, binds four disjoint revision/seed/tokenizer/token-stream partitions,
  complete relay/window policy, source-complete S2KF sensitivity lineage, and
  HESTIA gradcheck plus portable-v3 CPU/CUDA conformance. It records soft-method
  wins, ties, losses, or tradeoffs, binds each reported refinement metric to its
  exact evaluated hard-checkpoint artifact, and emits either a content-bound
  recipe/checkpoint freeze or terminal negative result for valid missing-rate,
  physical, native, or quality-gate failures. Malformed, contradictory,
  cherry-picked, or hash-consistent non-package traces fail closed.
  Stage-7 data admission now reopens a canonical 16 MiB `u32le` token payload,
  verifies all 2,048 per-sequence spans against exact source-row provenance and
  the shared SmolLM tokenizer identity, rejects duplicate samples, and freezes
  the official C4/OpenWebMath/StarCoderData revisions. The new
  `tritium salt build-stage7-evidence-pack` command converts content-bound,
  preselected source rows into that cross-language manifest transactionally.
  `tritium salt inspect-stage7-evidence-pack` then binds that pack to the
  campaign-frozen pack ID and exact model tokenizer, validates all manifest and
  payload semantics through a retained seek-backed handle, reads one bounded
  partition window, and emits its ordered token digest for model execution.
  StarCoder provenance distinguishes config `default`, `data_dir=python`, and
  source field `content`; fixed token geometry is rejected before bounded read.

- **QAT-hard convolution artifacts:** `TernaryConv1d` and `TernaryConv2d`
  now hard-convert into inference-only composite reference modules backed by
  additive packed storage, preserve shared/grouped convolution and padding
  semantics, and export/reload through QAT-hard artifact schema v2. Typed
  per-consumer contracts bind complete module geometry and reject same-size
  shell substitutions. Complete tensor-alias ledgers include persistent
  buffers; transactional reload cannot mutate a supplied shell on failure;
  shared state remains tied across whole-model dtype/device moves. Native
  no-dense-shadow packed convolution dispatch remains a separate
  optimized-runtime gate.

- **SALT reconstruction-fidelity report** (`tritium report salt-model`): loads an fp
  (bf16/f16/f32) safetensors **master**, SALT-quantizes every 2D weight at a sweep of
  bits-per-weight budgets, and reports whole-model (and optional `--per-tensor`)
  reconstruction error — MSE/RMSE/MAE/max-abs plus relative-Frobenius error and cosine
  similarity, the weight-space proxies for output divergence (true output KL is a
  forward-pass measurement layered on top, and needs general-architecture model support
  not yet present). The arch-agnostic way to see where ternary quantization hurts and
  how added bits/sensitivity-allocation recover it. Needs the fp master — an
  already-quantized checkpoint (EXL3/GGUF-Qx) carries no fp reference and won't parse as
  fp here.
- **`tritium_quantize::{ReconStats, ReconAccum, reconstruction_stats, ReconError}`** —
  the reconstruction metric, with a `ReconAccum` of raw moments that folds tensors and
  reduces to a *true* whole-model statistic (frob_rel/cosine are ratios of summed
  moments, not averages of per-tensor ratios). Purely additive (semver-minor).
- **`--sensitivity {uniform,energy}`** on `tritium quantize` (previously hardcoded
  `Uniform`) — exposes SALT's plane-allocation sensitivity on the model-quantize path,
  matching `tritium report salt`. (Note: `energy` ≈ `uniform` whenever reconstruction
  error tracks weight magnitude; a meaningfully different allocation needs a true
  loss-sensitivity signal via `Sensitivity::Custom`.)
- **Sharded + streaming model loading** for `report salt-model`: accepts a single
  `.safetensors`, a `*.safetensors.index.json` (reads `weight_map`), or a directory, and
  **mmaps each shard** so a 50GB+ master (e.g. a 27B bf16) is paged in per-tensor rather
  than read fully into RAM. Each shard is mapped once; shard + tensor order is
  deterministic.
- **Parallel `report salt-model`**: the (independent) 2D tensors are quantized on a
  bounded worker pool (`min(cores, 12)`), turning a 27B sweep from ~an hour into minutes.
  Output is bit-identical to the sequential path — an indexed `par_iter().collect()`
  preserves tensor order, so the global moments fold in the same sequence.

## [1.0.0] — 2026-06-28 — v1.0 Release 🎉

First stable release. The v1.0 Definition of Done (ADR 0012) is met: the public API and C ABI are frozen
(tiered — see below), the **real-model GPU capstone is proven on real hardware**, and every prior milestone
gate (v0.10→v0.90) re-runs green on this commit.

### Frozen-API ergonomics pass (pre-publish polish)

Before locking the frozen tier, a final ergonomics audit + refactor (the last cheap moment for breaking
changes — applied while unpublished). Highlights:

- **Forward-compat:** `#[non_exhaustive]` added to every extensible public enum/struct that lacked it —
  quantize (`Sensitivity`, `BaseScaleScope`, `QuantError`, `AllocError`), format (`SafeTensorsError`,
  `GgufFile`, `TensorInfo`, `FormatError::SafeTensors`), testkit (`Tolerance`, `ConformanceVector`),
  ffi (`TritiumStatus`, now also `#[repr(i32)]`). Free now, impossible post-publish.
- **One-crate authoring:** `tritium-testkit` and `tritium-spec`/`-runtime` re-export the backend trait +
  shared types, so a downstream backend author depends on one crate, not three.
- **`TernaryBackend` shape:** `mpgemm`/`mpgemm_with_act_quant` now take an `MpGemm<'_>` params struct
  (kills the transposable `act`/`scales` `&[f32]` footgun); the downcast escape hatch `as_any` → `as_concrete`
  (no longer clashes with `DeviceBuffer::as_any`); `BackendError` gained `source()` chaining.
- **quantize:** `QuantConfig` lost its lifetime (`Sensitivity::Custom(Vec<f64>)`) so configs are storable;
  `ScaleGroup` → `BaseScaleScope` (disambiguated from `core::ScaleGranularity`).
- **format:** `GgufValue::as_i64/as_f64/as_f32/as_bool/as_array`; `read_safetensors` free fn mirroring
  `read_gguf`; `source()` chaining on `FormatError`.
- **C ABI (ABI v1):** `tritium_generate` gained a versioned `const TritiumGenerateOptions*` (NULL = greedy
  default) so sampling/stop-tokens/backend-select can be added later without a new symbol or ABI break;
  new `tritium_last_error()` and `tritium_model_load_bytes()`; header lengths now `size_t`.
- **core:** `Trit: Ord`; `Display` on `DType`/`TernaryFormat`; `TritError: core::error::Error` unconditionally.

The on-disk conformance JSONL is byte-identical (serde renames), so the frozen-set drift gate stays green.
The **evolving tier** (`tritium-nn`/`-train`/`-cuda` + interop + `-serve`) keeps its 1.x breaking-change runway.

### Real-model GPU capstone — PROVEN (RTX 4090, sm_89)

Real `microsoft/bitnet-b1.58-2B-4T` runs correctly end-to-end on the GPU. All five acceptance gates pass:

- `cuda_perplexity_within_1pct` — ours **1.3987** vs transformers ref 1.4028 (rel 2.96e-3)
- `cuda_greedy_matches_transformers` — **256/256** tokens token-exact
- `cpu_cuda_parity` — identical token IDs over 32 steps, worst logit rel **2.26e-6**
- `cuda_batch_decode_matches_single` — N=2 batch == single, argmax-identical
- `qat_heal_gate` — layerwise distillation **94.6%** convergence (PPL 1.40)

Getting here required fixing two latent CUDA resident-decode bugs (`302d059`): a shared-memory aliasing bug
in the fused `rmsnorm_quant_f32` kernel (the block-wide absmax reduction reused the RMSNorm-output shared
buffer `s_x` as scratch → clobbered activations → garbage logits, PPL ~10⁵–10⁶) and an **unscaled** tiled
mpgemm (`f_tiled` instead of `f_tiled_scaled`) in the per-layer decode forward. A teeth-proven conformance
gate `rmsnorm_quant_bit_matches_host` was added (`51b041d`) so the kernel can't silently regress again.

### API / C-ABI freeze (tiered)

v1.0 freezes a **stable core** under semver; other crates are an **evolving tier** that may take breaking
changes in 1.x minor releases (rationale + per-crate detail in `docs/v1.0-api-freeze-audit.md`, ADR 0012):

- **Frozen (semver-gated):** `tritium-core`, `-spec`, `-format`, `-runtime`, `-cpu`, `-quantize`, `-testkit`,
  and the **C ABI** (`TRITIUM_ABI_VERSION = 1`, cbindgen-drift + C11/C++17 gated).
- **Evolving (not semver-gated; may break in 1.x minors):** `tritium-nn`, `-train`, `-cuda`, the interop
  crates (`-candle`, `-burn`, `-onnx`), and `-serve` — these track fast-moving upstreams and ongoing
  perf/training work.

### Gates green on the release commit

`cargo fmt --check`, `cargo clippy --workspace --all-targets -D warnings`, the full CPU workspace test
suite, `cargo-semver-checks` (vs `v0.5.10` baseline), and the CPU fresh-env `capstone` — all green. GPU
validation is **fenced** (ADR-0011 amendment): CUDA conformance + the real-model capstone above on the
local 4090; Metal (M1) / ROCm (MI300X) / wgpu (4090) parity from prior fenced sessions; the GPU CI lanes
remain dispatchable `workflow_dispatch` recipes.

### Docs

Reproducible quickstart, model zoo (the BitNet 2B4T I2_S→TQ2_0 load path + loader type-id contract), and
benchmark methodology (roofline ceilings + divan microbench + e2e tok/s, with honest fenced-measurement
boundaries) — `docs/book/`.

## [0.9.0] — 2026-06-24 — v0.90 Hardening milestone COMPLETE

The **v0.90 Hardening** milestone (ADR 0011) is complete — a depth pass over the v1.0 surface, no new
capabilities. The two gates genuinely open this session are now closed and verified:

- **Sanitizers green** — `sanitizers.yml` ran clean for the first time: ASan/MSan/TSan via `-Zbuild-std`
  (tritium-cpu/ffi/format) + miri (tritium-core/format parsers). The first run surfaced a miri *isolation*
  issue — proptest's `getcwd` under the sandbox — fixed with `-Zmiri-disable-isolation` (gates only
  syscalls, not UB detection). **No real UB found.**
- **Threat model documented** — `docs/security/threat-model.md`: a 30-threat, code-grounded review across
  the model-file parsers, C ABI/FFI, HTTP server, kernels + dispatch, and supply chain — each with STRIDE +
  severity + mitigations cited to source + residual risks. Linked from `SECURITY.md`.

Already shipped on the 0.5.x/0.6.x build-ahead line: cargo-deny, fuzz breadth + corpora (8 targets, 769
files), doc-coverage + semver baseline, mdbook + dead-link lane, abi3 wheels, cpu-bench-smoke, SBOM.

The GPU-dependent gates (full CI matrix, perf-regression-on-main, compute-sanitizer) are met via the **ADR
0011 amendment** (2026-06-24): GPU parity validated by documented fenced sessions (cuda/A100, metal/M1,
rocm/MI300X, wgpu/4090) + the GPU lanes kept as `workflow_dispatch` recipes; **Metal additionally runs free
on GitHub-hosted `macos-14` every push**. "Green on every push" is waived for the paid-GPU backends absent
standing runners.

### Added
- `docs/security/threat-model.md` — the full code-grounded threat model (closes the ADR-0011 security gate).

### Changed
- CI: `gpu`/`rocm`/`wgpu`/`perf-regression`/`serve-e2e` lanes are now `workflow_dispatch` (dispatchable
  on-demand) instead of `if: false`; the `metal` lane runs free on GitHub-hosted `macos-14`; the
  `sanitizers` miri job uses `-Zmiri-disable-isolation`.

## [0.8.0] — 2026-06-24 — v0.80 Interop milestone COMPLETE (all four framework frontends)

The **v0.80 Interop** milestone (ADR 0010) is complete: Tritium's ternary mpGEMM is reachable from every
target frontend, each validated green in CI on CPU:

- **`tritium-serve`** — OpenAI-compatible HTTP/SSE server: contract + streaming + concurrency (CI lane
  *serve contract, cpu mock*). (`v0.6.2`)
- **`tritium-ffi`** — C ABI cdylib + staticlib, panic-safe, cbindgen header with a drift gate + C11/C++17
  compile check; round-trips through the C ABI. Unblocks the v1.0 C-ABI freeze. (`v0.6.3`)
- **`tritium-candle`** — candle `CustomOp1`, bit-exact vs `reference_mpgemm` (CI lane *candle interop*). (`v0.6.4`)
- **`tritium-burn`** — backend-generic burn op, bit-exact (CI lane *burn interop*). (`v0.6.5`)
- **`tritium-onnx`** — always-on bit-exact kernel + `ort` 2.x custom op == native (CI lane *onnx custom op*). (`v0.6.5`)

No new code in this release — the frontends shipped across the 0.6.x line; this tag marks the milestone now
that v0.70 (its predecessor) is complete (per ADR 0002, milestones tag in order). The abi3 Python wheel
additionally builds + imports + passes pytest on macOS arm64 (validated on the M1, `v0.6.9`).

### Validated (CI, CPU)
- *serve contract (mock)* · *candle interop* · *burn interop* · *onnx custom op* · *wasm conformance* ·
  *cpu-only test/clippy/fmt* on ubuntu + macos + windows — all green.

## [0.7.0] — 2026-06-24 — v0.70 Backend Breadth milestone COMPLETE (full backend matrix hardware-validated)

The **v0.70 Backend Breadth** milestone (ADR 0009) is complete: every backend in the matrix —
cpu · cuda · wgpu · wasm · **metal · rocm** — now passes the frozen conformance suite on its own
hardware. The two fenced-HW backends are validated on real silicon:

- **Metal** ✅ — bit-exact 89-vector conformance on a real Apple M1 (Scaleway M1-M, macOS 26.3.2),
  shipped `v0.6.9`, plus an in-kernel TQ2_0 decode for device-memory parity with cuda/rocm.
- **ROCm** ✅ — the frozen-vector conformance **ran on a real AMD Instinct MI300X** (gfx942, ROCm 7.2.4,
  Hot Aisle) — confirmed executed on the GPU, not self-skipped — matching `reference_mpgemm` within 1e-4.
  The HIP kernel fat-compiles for gfx900..gfx1100; the `build.rs` `libamdhip64` link-search fix (`5397781`)
  makes it link on a stock ROCm install.

No new code in this release — the backend code shipped across the 0.6.x line (wgpu/wasm `v0.6.1`,
metal/rocm `v0.6.8`, metal memory-parity `v0.6.9`, rocm link fix `5397781`). This tag marks the milestone
now that both fenced-HW parity runs have landed (per ADR 0002: no milestone tag before its HW gate clears).

### Validated
- Full backend matrix hardware-conformant: cpu (SIMD), cuda (Ampere, `v0.6.0`), wgpu (4090 Vulkan +
  Apple Metal HAL), wasm (wasmtime), **metal (Apple M1)**, **rocm (AMD MI300X)**.

## [0.6.9] — 2026-06-24 — Metal parity validated on Apple Silicon + Metal memory-parity (in-kernel TQ2_0 decode) + macOS-portable capstone

First hardware validation of a fenced-HW backend. **Metal parity is green on a real Apple M1** (Scaleway
M1-M, macOS 26.3.2, Rust 1.96): the blind-written MSL backend compiled and passed the frozen-vector
conformance + fused-fallback + 2-D-grid + zero-dim tests on the first compile, **zero glue fixes** — exactly
as the pre-compile audit (zero real findings) predicted. As a bonus on the same lease, the **wgpu** backend
validated on the Apple GPU via wgpu's Metal HAL (4 tests), the whole Rust workspace passed on **macOS arm64**
(407 tests / 77 binaries), and the **abi3 Python wheel** built via maturin + imported + passed its pytest
suite (FFI safety, GIL release) — coverage the wheels CI lane (artifacts-only) does not exercise.

### Changed
- **`tritium-metal` — device-memory parity with cuda/rocm.** `mpgemm.metal` gains an `mpgemm_tq2_0` kernel
  that decodes the 2-bit TQ2_0 codes **in-kernel** (a direct port of the verified cuda/hip add-only kernel),
  so TQ2_0 weights now stay **packed on device** (~2.06 bit/trit) instead of being host-widened to one `i32`
  per trit (32 bit/trit — a 16× blow-up). This removes the limitation that a multi-billion-parameter model
  could not fit Apple unified memory; the Metal backend is now memory-equivalent to cuda/rocm. TQ1_0 (the
  small/rare format) keeps the host-unpack + widen path. Validated bit-exact (≤1e-4) on a real M1 GPU
  across all 47 TQ2_0 frozen vectors plus the random large-shape and zero-dim cases.
  Kernel row/element indices use 64-bit (`ulong`) arithmetic so large shapes cannot
  overflow the index math (matching the cuda/hip reference's `long long` casts).

### Fixed
- **`scripts/capstone.sh` is now macOS-portable.** STEP 1's workspace build now `--exclude tritium-py`
  (a PyO3 `extension-module` cdylib that deliberately does not link libpython — building its cdylib under
  plain `cargo build` fails to link on macOS with undefined `_Py*` symbols), mirroring `ci.yml`. The
  capstone now runs green end-to-end (build → infer → SALT → fine-tune) on macOS arm64, not just Linux.

## [0.6.8] — 2026-06-23 — full backend matrix: tritium-metal + tritium-rocm + third-party license bundle

Lands the last two backends (ADR 0009 full scope) and the dependency-license tracking for v1.0
packaging. With cpu/cuda/wgpu/wasm, the backend matrix is now **code-complete**; the Metal/ROCm GPU
code is written but gets its first compile + parity validation on the target hardware (the new
self-hosted CI lanes) — the fenced-HW step.

### Added
- **`tritium-metal`** (Apple Metal; `metal` + `register` features) — an MSL compute kernel porting the
  wgpu WGSL add/sub/skip ternary mpGEMM (runtime-compiled; shared-storage `MTLBuffer` / unified
  memory) + a `TernaryBackend` impl + linkme `"metal"` registration + a conformance test that
  self-skips when no Metal device is present. `metal-rs` (`= 0.33.0`) is declared under
  `[target.'cfg(target_os = "macos")']`, so a Linux/CI build never resolves it; all device code is
  `cfg(target_os = "macos")`. The crate is an inert empty lib off macOS.
- **`tritium-rocm`** (AMD ROCm/HIP; `rocm` feature, off by default) — raw `extern "C"` FFI to the HIP
  runtime (zero external deps); a `build.rs` that shells `hipcc` (a **no-op** unless
  `CARGO_FEATURE_ROCM` is set) to compile `kernels/tq2_0_add.hip` (a port of the cuda add-only kernel)
  + a `TernaryBackend` impl + linkme `"rocm"` registration + a self-skipping conformance test. Inert
  empty lib without the feature.
- Two manual self-hosted CI lanes (`metal`: Apple-Silicon; `rocm`: AMD + ROCm), `if: ${{ false }}`,
  mirroring the `gpu`/`wgpu` lanes — where the GPU code first compiles + validates parity.
- **`THIRD-PARTY-LICENSES.md`** — the bundled license texts of all 369 third-party crates (via
  `cargo-about`; config `about.toml` + `about.hbs` committed for reproducibility), the v1.0-packaging
  dependency-tracking step. Covers 9 licenses incl. the newly allow-listed MPL-2.0 (`colored` ← burn)
  and CDLA-Permissive-2.0 (`webpki-roots` ← ort).

### Honesty
- The Metal/ROCm GPU code was authored **blind** (no Metal framework / no ROCm toolkit on the dev box).
  Verified here: the Linux workspace stays green (`cargo build`/`clippy`/`test --workspace`), each
  crate gated to an inert lib (`cargo tree` confirms neither the metal binding nor any hip dep is
  pulled by default; negative-registration tests pass). Adversarial port-fidelity review vs the
  verified wgpu/cuda kernels: zero findings. The kernels' first real compile + numeric parity is the
  hardware lane.

### Build
- New workspace members `tritium-metal`, `tritium-rocm`; `metal = "=0.33.0"` added to
  `[workspace.dependencies]` (macOS-target-only). Internal dep pins bumped to `0.6.8`.

## [0.6.7] — 2026-06-23 — v1.0 capstone prep: freeze audit + model-zoo/benchmark docs + CPU fresh-env e2e

The reachable **v1.0 Release** preparation (ADR 0012), on the 0.6.x line. The v1.0.0 **tag** itself
stays gated on fenced hardware (Metal/ROCm parity + the GPU CI matrix + the real-model/GPU capstone),
per ADR 0002/0012 — none of that is claimed here.

### Added
- **`docs/v1.0-api-freeze-audit.md`** — a report-only freeze-readiness audit of the whole public
  surface: a per-crate readiness table, the **C ABI confirmed frozen at v1** (via the cbindgen drift +
  C11/C++17 compile gates), the `cargo-semver-checks` enforcement process, and a prioritized
  `[breaking]`/`[additive]` list of pre-freeze API changes to consider. **Mutates no public API.**
  Measured (not assumed): `tritium-cpu`/`-nn`/`-train`/`-py` already have full doc coverage despite
  lacking `#![deny(missing_docs)]`; the semver gate is green (253 checks/crate vs the `v0.5.10`
  baseline). P0 flag: `tritium-train` exposes 12 `pub mod`s to curate before the freeze.
- **mdbook Model Zoo + Benchmarks chapters** — the real model-load path (BitNet b1.58 2B4T loads via
  the GGUF **I2_S** type-id 36, re-packed to TQ2_0; EOS `128001`) and the perf **methodology** (the
  divan microbenches, the e2e tok/s bench coupled to a perplexity gate, the `tritium report`
  subcommands) — with **zero fabricated numbers** and CPU-measurable vs GPU-required signals clearly
  separated. Wired into `SUMMARY.md`.
- **`scripts/capstone.sh` + `capstone` CI lane** — a CPU fresh-env end-to-end smoke exercising the
  **install → infer → SALT → fine-tune** pipeline on real code paths (workspace build; `list-backends`;
  a real GGUF parse + a clean runner-load-fail on the partial fixture; a real SALT quantize → `.tslb`
  bundle; the `tritium-train` STE/AdamW CPU gates). Honest DEFERRED markers mark the GPU/real-model
  capstone (the v1.0.0 tag gate).

### Build
- No workspace code or dependencies changed (docs + a shell script + a CI lane). Internal dep pins
  bumped to `0.6.7`.

## [0.6.6] — 2026-06-23 — v0.90 hardening polish: mdbook + sanitizers + wheels + CPU bench smoke

The reachable **v0.90 Hardening** tooling (ADR 0011), on the 0.6.x line. (The fenced-hardware
v0.90 items — Metal/ROCm platform-GPU lanes — remain deferred.)

### Added
- **mdbook user guide** (`docs/book/`): 9 chapters — introduction, architecture, quickstart,
  backends, quantization (SALT), training, interop, conformance, contributing — sourced from the
  real code + ADRs. A new **`docs` CI lane** builds the book and **fails on any dead internal link**
  (the `mdbook-linkcheck` backend with `warning-policy = "error"`).
- **`sanitizers` CI lane** (nightly; weekly + manual): AddressSanitizer + MemorySanitizer +
  ThreadSanitizer (via `-Zsanitizer` + `-Zbuild-std`) over the crates with hand-written `unsafe` or
  untrusted-byte parsing (`tritium-cpu`, `tritium-ffi`, `tritium-format`), plus **miri** over the
  pure-Rust crates (`tritium-core`, `tritium-format` parsers). All four run clean. (The Rust
  toolchain has no `-Zsanitizer=undefined`; MSan/TSan/miri cover the UB classes a C UBSan would —
  documented in the lane.)
- **`wheels` CI lane** (tags + PR smoke + dispatch): abi3 Python wheels via maturin for
  linux (manylinux) / macOS (universal2) / windows + an sdist, uploaded as **build artifacts only**
  (no PyPI publish). A real `cp39-abi3` manylinux wheel was built + inspected locally.
- **`cpu-bench-smoke` CI lane** (every push): compiles all benches + runs the CPU divan `mpgemm`
  microbench — the hosted half of the perf gate (the tok/s regression assertion stays in the
  self-hosted `perf-regression` lane).

### Docs accuracy (adversarial review)
- Corrected three confirmed doc-accuracy issues the review caught: distributed training is **shipped**
  (v0.60; re-exported from `tritium-train`), not "the next milestone"; the SALT pipeline now flags
  steps 2/5/6 as planned-not-yet-wired (only 1/3/4 drive the quantizer); and `tritium-cpu` dispatches
  **AVX-512 → AVX2 → NEON → scalar**, not just AVX2.

### Build
- No workspace code or dependencies changed (mdbook / maturin are CI tools). Internal dep pins bumped
  to `0.6.6`.

## [0.6.5] — 2026-06-23 — v0.80 interop COMPLETE: tritium-burn + tritium-onnx (framework backends)

The final **v0.80 Interop** slice (ADR 0010): burn and ONNX Runtime. With serve + ffi + candle, all
four framework backends are now landed. On the 0.6.x line.

### Added
- **`tritium-burn`** (feature `burn`) — [`ternary_mpgemm`]`<B: Backend>`, a backend-generic op that
  runs Tritium's ternary (BitNet b1.58) mpGEMM on a burn `Tensor`: `[M, K]` f32 activations × `[N, K]`
  packed ternary weights (TQ2_0 / TQ1_0) × `[N]` scales → `[M, N]` f32, **bit-exact** with the
  reference. A host round-trip (read → `reference_mpgemm` → rebuild) that works on any burn backend
  (NdArray, wgpu, cuda) in f32; a deferred-execution read failure on a lazy backend is returned as a
  `BurnTernaryError`, not a panic; the result is pinned to `DType::F32`. Conformance test reproduces
  the frozen vector set bit-exactly on the NdArray CPU backend + negative tests. `burn-tensor` /
  `burn-ndarray` 0.21, optional behind the feature (lean default). New `burn-conformance` CI lane.
- **`tritium-onnx`** — two layers so the always-on CI needs no native library:
  - **Layer 1** (default, zero external deps): `ternary_mpgemm_kernel`, a plain bit-exact kernel
    whose conformance test is the default-feature gate (no `ort`, no `onnxruntime`).
  - **Layer 2** (feature `onnx`): an `ort` 2.x custom operator exposing the kernel as the ONNX node
    `TritiumTernaryMpGemm` (the `Operator` / `Kernel` traits). `ort = 2.0.0-rc.12` (default-features
    off + `download-binaries` + `tls-rustls`) fetches a prebuilt onnxruntime at build, so a networked
    CI lane builds + tests `--features onnx` with no system library. The `run` kernel logic is tested
    bit-exact + the operator registers; the full native session dispatch is the `#[ignore]`d e2e. New
    `onnx-op` CI lane.

### Supply-chain
- `deny.toml` allow-lists **MPL-2.0** (`colored`, via `burn-tensor`'s `std`) and
  **CDLA-Permissive-2.0** (`webpki-roots`, via `ort`'s `tls-rustls`), both scoped + justified;
  cargo-deny `--all-features` licenses + bans clean. `ort` pin bumped from the unused `2.0.0-rc.10`
  to `=2.0.0-rc.12`.

### Build
- New workspace members `tritium-burn`, `tritium-onnx`; all framework deps optional behind their
  features so the default workspace build + `cargo test --workspace` stay free of burn/ort/onnxruntime.
  Internal dep pins bumped to `0.6.5`.

## [0.6.4] — 2026-06-23 — v0.80 interop: tritium-candle (ternary mpGEMM as a candle op)

The third **v0.80 Interop** slice (ADR 0010): expose Tritium's ternary mpGEMM as a candle-native
op, so a candle model graph can use BitNet ternary weights. On the 0.6.x line.

### Added
- **`tritium-candle`** — [`ternary_mpgemm`], a [`candle_core::CustomOp1`] (applied via
  `apply_op1_no_bwd`) that runs Tritium's ternary (BitNet b1.58) mpGEMM on a candle `Tensor`: an
  `[M, K]` f32 activation tensor times `[N, K]` packed ternary weights (TQ2_0 / TQ1_0) with `[N]`
  per-output-channel scales, producing `[M, N]` f32. `N` is taken from `scales.len()`. The kernel is
  `tritium_core::reference_mpgemm` itself, so a candle BitNet layer is **bit-exact** with the
  reference every Tritium backend is graded against. Validates dtype/contiguity/K/packed-length and
  errors (never panics) on mismatch; the op borrows its weight bytes (call once per forward).
- **Gate:** a candle-`Tensor` conformance test reproduces the full frozen vector set
  (64 random + boundary) **bit-exactly**, proving the Tensor↔slice plumbing (layout, shape,
  readback); plus negative tests for K / packed-length / non-contiguous-activation. New
  `candle-conformance` CI lane (clippy + `cargo test --features candle`, every push).

### Build
- New workspace member `tritium-candle`; the heavy `candle-core` dep (CPU-only — no
  cuda/mkl/accelerate) + the op live behind the **`candle`** feature, off by default, so the default
  workspace build and `cargo test --workspace` stay candle-free (mirrors `tritium-serve`'s `serve`).
  candle-core 0.9.2's tree is cargo-deny clean. Internal dep pins bumped to `0.6.4`.

## [0.6.3] — 2026-06-23 — v0.80 interop: tritium-ffi (C ABI cdylib + staticlib)

The second **v0.80 Interop** slice (ADR 0010): a stable C ABI so any language can drive Tritium
inference. On the 0.6.x line; unblocks the **v1.0 C-ABI freeze**.

### Added
- **`tritium-ffi`** — a `cdylib` + `staticlib` exposing a small, stable, panic-safe C API: load a GGUF
  model on the CPU backend and greedily generate token IDs from C/C++/any language. Surface (the
  cbindgen-generated `include/tritium.h`): `tritium_abi_version()` (ABI v1), `tritium_version()`,
  `tritium_model_load_file()`, `tritium_generate()` (single `max_new`-sized pass, or `out_cap=0`
  size-then-fill), `tritium_model_free()` (null-safe). Boundary discipline: every entry point
  null-checks its pointer args (→ `NullArg`, never a deref) and wraps its body in `catch_unwind`; the
  three pointer-taking functions are `unsafe extern "C"`; `*out_len` is always written when non-null.
  The cpu backend's `linkme` registration is **verified to survive linker GC into the linked
  cdylib/staticlib** (so `load_cpu` resolves). A documented `examples/roundtrip.c` shows the consumer flow.
- **Header discipline:** `include/tritium.h` is committed; a dev-only drift test regenerates it with
  cbindgen and fails on mismatch, then compiles it as **C11 + C++17** under `-Wall -Wextra` on Linux.
  No `build.rs` source write (keeps the publish clean-tree check honest).
- **Tests:** 10 ABI null/error/version tests + 2 header gates; a gated real-model round-trip
  (`TRITIUM_FFI_MODEL=<gguf>`).

### Notes
- **`panic = "abort"`:** under the default `release`/`dist` profile a panic aborts the process (the
  safe, defined FFI outcome); the `catch_unwind` → `TritiumStatus::Panic` path is reachable only when
  built with `panic = "unwind"` (the `dev`/`test` default). `panic` is a whole-artifact profile setting
  — Cargo forbids overriding it per crate.

### Build
- New workspace member `tritium-ffi`; `cbindgen` workspace dev-dependency. Internal dep pins bumped to `0.6.3`.

## [0.6.2] — 2026-06-23 — v0.80 interop: tritium-serve (OpenAI HTTP/SSE) + pyo3 security bump

The first **v0.80 Interop** slice (ADR 0010) plus a supply-chain hardening pass, on the 0.6.x line.

### Added
- **`tritium-serve`** — an OpenAI-compatible HTTP inference server (axum, behind a `serve` feature; the
  default workspace build stays free of tokio/axum). Endpoints: `/v1/chat/completions` (non-streaming
  **and SSE** streaming), `/v1/models`, `/healthz`. A `Generator` seam isolates HTTP from inference:
  `RunnerGenerator` wraps the real `ModelRunner` (re-implementing the prefill + per-step `forward`
  decode loop with per-step sampling + seed-advance + a context guard), and `MockGenerator` drives the
  **model-free contract lane**. One dedicated decode thread owns the (`&mut`-exclusive) runner behind a
  bounded queue: concurrent connections, **backpressure** (429 when full), and **graceful shutdown**
  (drain flag → in-flight SSE streams close with a well-formed terminal chunk + `[DONE]`). Per-token
  incremental detokenization (stream-concat == buffered output) + OpenAI `stop`-string matching.
  **LAMU-compatible** by OpenAI-wire fidelity (point a `local-llm` OpenAI backend at `/v1`). The ADR
  0010 gate is proven by **11 model-free contract tests**; a gated `e2e` feature runs a real-model
  round-trip. Ships the **id-passthrough tokenizer** (integer token IDs) — real LLaMA-3 BPE is the
  separate tokenizer-seam task.
- **CI:** a `serve-contract` lane (cpu, mock, every push) + a manual `serve-e2e` lane.

### Security / supply-chain
- **pyo3 `0.23 → 0.25.1`** — clears **RUSTSEC-2025-0020** (buffer overflow in `PyString::from_object`)
  and the macOS/Windows abi3 link failure on the hosted CI runners. No `tritium-py` API migration was
  needed (already on the modern Bound API; verified end-to-end via a maturin wheel + 13/13 pytest).
- **cargo-deny:** ignore **RUSTSEC-2024-0436** (`paste` unmaintained — an Apple-only, build-time
  `wgpu-hal → metal` transitive; not a vulnerability). With the pyo3 bump, the supply-chain lane is green.

### Build
- New workspace member `tritium-serve`; tokio/axum/async-stream/futures-core/http-body-util/tower
  workspace deps (all behind serve's feature or dev-only). Internal dep pins bumped to `0.6.2`.

## [0.6.1] — 2026-06-22 — v0.70 backend breadth (reachable): wgpu + wasm backends + capability-fallback contract

Three software-reachable slices of **v0.70 Backend Breadth** (ADR 0009), shipped on the 0.6.x line
ahead of the platform-GPU milestone gate. Both new backends pass the frozen v0.70 conformance set on
real targets (the 4090 Vulkan adapter; wasmtime). The v0.70 **milestone** tag still waits on Metal +
ROCm hardware parity — these are the reachable-now backends.

### Added
- **`tritium-wgpu`** — cross-platform GPU backend: a WGSL ternary mpGEMM compute shader over wgpu
  (Vulkan), validated on the **RTX 4090 Vulkan adapter** against all 89 frozen conformance vectors
  (≤1e-4) + the fused-fallback contract. Host-unpacks weights (TQ2_0/TQ1_0) to an `i32` storage
  buffer; the shader uses the reference's **add/sub/skip** accumulation form (not `act·f32(trit)`) so
  f32 round-off stays inside the 1e-4 bar on high-cancellation vectors. A **2-D workgroup dispatch**
  handles `M·N` beyond the 65535-per-dimension Vulkan limit (validated at `M·N = 4.19M`). Device-side
  validation errors are **error-scoped → `BackendError`, never a panic**; adapter selection prefers
  the discrete NVIDIA GPU and requests the adapter's real limits. All GPU code behind `--features
  wgpu`; `--features register` adds linkme self-registration.
- **`tritium-wasm`** — scalar `TernaryBackend` for `wasm32-wasip1`: the portable
  `tritium_core::reference_mpgemm` over `tritium-format`-unpacked weights, depending only on the
  wasm-clean spec/core/format crates (no rayon, no linkme — neither compiles on wasm32). Conformance
  runs **inside wasmtime** (Cranelift) on every push; bit-exact with the reference.
- **`run_fused_fallback_contract`** (tritium-testkit) — pins the no-panic-degrade contract for the
  fused W1.58A8 path: a backend advertising no fp8/IMMA must serve `mpgemm_with_act_quant` via the
  host-default fallback, graded against the host-A8 reference with a per-token scale-aware tolerance
  floor (`Tolerance::accepts_with_floor`). Exercised by CPU + wgpu + wasm.
- **CI:** a `wasm` lane (wasm32-wasip1 under wasmtime, every push) + a `wgpu` lane (self-hosted Vulkan,
  manual, like the `gpu` lane). The new crates also build/test in `cpu-only-green` via their default
  (no-GPU) builds.

### Build
- New workspace members `tritium-wgpu` / `tritium-wasm`; `wgpu = "=23.0.1"` (exact pin, satisfies
  cargo-deny `wildcards=deny`) + `pollster` workspace deps; new `.cargo/config.toml` wasmtime runner.
  Internal dep pins bumped to `0.6.1`.

## [0.6.0] — 2026-06-22 — v0.60 Pretraining + Distributed (ADR 0008): real-NCCL wall cleared on 2×A100

The **v0.60 milestone**. The from-scratch distributed-training stack — built single-GPU-reachable
across 0011–0016 and proven in simulation — is now **validated on real multi-GPU hardware**: the real
`cudarc::nccl` backend agrees with the deterministic simulated reference, closing ADR-0008's
distributed-correctness story.

### Hardware validation (2× A100-SXM4-80GB, production mode)
- **0017 NCCL wire-correctness:** all_reduce == summed reference, all_gather == ordered concat,
  broadcast == root — green at world=2 on real NCCL (2.28.9).
- **0018 FSDP loss-parity:** the tiny-MLP FSDP loop over the real `NcclProcessGroup` tracks the
  single-process reference to **`max |Δloss| = 4.5e-8`** (float-epsilon) at world=2.
- **First datacenter-arch run:** full CUDA suite **51/51 on Ampere/sm_80** (every hand-written PTX
  kernel + the JIT/autotuned IMMA codegen bit-clean); `compute-sanitizer memcheck` clean over the
  single-GPU kernels.
- The **≥80% throughput-scaling** gate stays explicitly deferred — the gate models are tiny seeded
  MLPs where comms are negligible, so a scaling figure proves nothing; it needs the separately-fenced
  real-scale resident engine. `v0.6.0` is tagged on correctness.

### Distributed stack (shipped 0011–0016 on the 0.5.x line, now milestone-tagged)
full-model CPU backward (0011) · resumable sharded data pipeline (0012) · GPU pretrain smoke (0013) ·
`ProcessGroup` trait + simulated collective backend (0014) · ZeRO-3/FSDP with gradient + loss parity
(0015) · distributed checkpoint with resharding + crash-atomic writes (0016) · real `cudarc::nccl`
backend (0017) + the 2×A100 wall (0018).

### Fixed / hardened
- **`--features cuda` now compiles** — reverted an incomplete in-flight `bl_matmul` change (the M>1
  batched-scale fusion) that left the call sites inconsistent with the definition. It slipped in
  because the GPU CI lane is `if: false`; enabling a cuda-feature build is the ROADMAP follow-up so
  this can't recur.
- **`scripts/gpu_session.sh` hardened for *any* multi-GPU box**: auto-shims the unversioned
  `libnccl.so`/`libnvrtc.so` that cudarc dlopens (images ship only `*.so.N`); tolerates any NCCL 2.x
  (the 2.30 bindings are ABI-stable — 2.28 validated); optional toolchain auto-install for bare base
  images; requires/notes production mode for multi-GPU; robust positive-evidence verdict.

### Build
- Internal workspace dep pins bumped to `0.6.0`.

## [0.5.10] — 2026-06-21 — v0.90 hardening gates: doc-coverage + API-stability

Two reachable-now release-readiness gates (ADR 0011 → ADR 0012), both CPU-only and clear of the
active GPU/training WIP. Scope: the stable, GPU-free, non-binary library crates.

### Added
- **`#![deny(missing_docs)]`** on `tritium-core`, `tritium-spec`, `tritium-format`,
  `tritium-runtime`, `tritium-testkit`, `tritium-quantize` — every public item must now be documented
  (enforced by the existing build/clippy lane). spec/runtime/testkit were already complete; documented
  the 17 remaining gaps in core/format/quantize (enum-variant struct fields + one constructor).
- **`scripts/check-semver.sh`** + a **`semver` CI lane** (`cargo-semver-checks` via
  `taiki-e/install-action`, baseline = latest `v0.5.*` tag): asserts no *unintentional* breaking API
  change to the 7 stable public-API crates. Verified locally — current API is non-breaking vs `v0.5.6`
  ("no semver update required" on all 7).

### Scope note
`tritium-cuda` (GPU build), binaries (`cli`/`benches`), `tritium-py` (cdylib ABI), and the in-flux
`tritium-nn`/`tritium-train` are excluded from both gates for now; widen as they stabilize toward v1.0.

## [0.5.9] — 2026-06-21 — Fuzz breadth: a target for every untrusted-byte parser

v0.90-hardening build-ahead (ADR 0011, U5). Completes the "every parser has a cargo-fuzz target"
invariant across the model-file trust boundary — additive (no library change).

### Added
- **5 new `tritium-format` fuzz targets**: `tqbin_parse`, `tqidx_parse`, `salt_bundle_parse`,
  `safetensors_parse`, `salt_legacy_parse` — covering `read_tqbin` / `read_tqidx` /
  `read_salt_bundle` / `SafeTensors::parse` / `read_legacy_as_salt` (the previously-unfuzzed
  untrusted-byte parsers). With the existing 3, every `tritium-format` parser now has a target.
  Verified locally (nightly + cargo-fuzz): each builds, links libFuzzer, and runs clean with no
  crashes/leaks (millions of execs, RSS ~0.5 GB).

### Changed
- **CI fuzz lane** now loops over all 8 targets (~450 s each, ~1 h total) instead of three hard-coded
  runs; adding a parser is a one-line edit.

## [0.5.8] — 2026-06-21 — Supply-chain gate (cargo-deny) + publishable internal dep pins

v0.90-hardening build-ahead (ADR 0011, supply-chain). `cargo deny check` now passes and runs as a CI
lane on every push/PR.

### Added
- **cargo-deny CI lane** (`EmbarkStudios/cargo-deny-action@v2`): checks licenses, RustSec advisories,
  bans, and sources. The action pins a recent cargo-deny that parses CVSS-4.0 advisories (the locally
  installed 0.16.x cannot), so CI is the authoritative advisory gate.

### Changed
- **`deny.toml`**: allow `Unicode-3.0` (the `unicode-ident` transitive dep's license; a permissive
  Unicode license — a normal Cargo dependency, not a vendored code port, so not in NOTICE; bundling
  dependency license texts is deferred to the v1.0 packaging work).
- **Internal workspace deps are version-pinned** (`{ path = "...", version = "0.5.7" }`) instead of
  bare path deps. The caret requirement resolves across the entire `0.5.x` build-ahead line with zero
  churn (only a `0.60.0`-style minor bump updates it, as part of that release commit); it keeps
  `wildcards = "deny"` meaningful for *external* deps (cargo-deny refuses `allow-wildcard-paths` for
  publishable crates), and is required for the v1.0 `cargo publish` gate (crates.io forbids bare path
  deps).

## [0.5.7] — 2026-06-21 — Freeze + version the conformance vector set (first v0.70 build-ahead)

The conformance suite is now a **committed, versioned, immutable artifact** instead of a set
regenerated from a seed at test time. This is the prerequisite ADR 0009 names for v0.70 backend
breadth: every future backend (`tritium-wgpu`/`tritium-wasm`/`tritium-metal`/`tritium-rocm`), every
v0.80 "matches the native reference" interop gate, and every v1.0 release re-run grades against one
reference that must not drift underneath them.

> **Build-ahead note.** v0.60.0 is gate-blocked behind the rented ≥2-GPU session (0017/0018). Per
> ADR 0002 the milestone *tags* stay sequential, but the *build* order does not: this and subsequent
> `0.5.x` point releases land software-reachable slices of v0.70+ now, so the rented session becomes a
> short verify-and-tag cascade rather than a per-milestone build-then-tag crawl.

### Added
- **`tritium-testkit::frozen`** — `frozen_vectors()` loads the committed `vectors/v070.jsonl`
  (`= generate_vectors(0xC0FFEE, 64)`, 89 vectors: 64 random + the 25-case boundary set, both packing
  formats). The path is resolved from the testkit crate's `CARGO_MANIFEST_DIR`, so any consuming crate
  finds the one canonical set regardless of its own test cwd. `VECTOR_SET_VERSION` + `FROZEN_SEED` +
  `FROZEN_COUNT` are public pins.
- **`freeze_vectors` example** — the single sanctioned, reproducible way to (re)generate the artifact:
  `cargo run -p tritium-testkit --example freeze_vectors`.

### Gates
- **`frozen_set_matches_pinned_generator`** (the teeth): the committed file must equal
  `generate_vectors(FROZEN_SEED, FROZEN_COUNT)`. Any drift — a changed generator, a changed reference
  kernel, a hand-edited file — is a hard failure, so a re-freeze must be deliberate (regenerate +
  bump `VECTOR_SET_VERSION`).
- **CPU conformance** repointed to `frozen_vectors()`. The frozen set *is* the historical
  `(0xC0FFEE, 64)` set, so this changed nothing about what CPU validates — only locked it. `cuda.rs`
  left untouched (active perf-optimization WIP); a count-monotone subset gate
  (`smaller_count_is_a_value_subset_of_the_frozen_set`) proves its seed-generated set is contained in
  the frozen set, so the freeze holds for the GPU path too.

## [0.5.6] — 2026-06-21 — Distributed checkpoint: resharding + crash-atomic writes

The sixth v0.60 increment (single-GPU-reachable; `0.5.x` line). The ADR-0008 "checkpoint resharding
`J≠K` ⇒ identical forward; resume continues" and "kill rank mid-run ⇒ clean error / no corrupt
checkpoint" gates, reachable on one machine: a sharded distributed checkpoint (DCP) over the 0015
`FlatShardPlan` layout, with crash-atomic writes.

### Added
- **`dcp` module** (`DistCheckpoint` + `save`/`load`, re-exported with `DcpError`): a checkpoint is a
  directory of one shard file per rank plus a `manifest.tdcp` committed **last** (the single commit
  point), each written `temp → fsync → rename → fsync-parent`. The global state is **world-agnostic** (a
  shard is a contiguous slice of the flattened/padded buffers), so `load` reassembles the same global
  `(param, planes, step)` regardless of save-time world `K`, and resharding to `J` is just
  `FlatShardPlan::new(leaf_lens, J)`. Optimizer state rides as parallel f32 planes (AdamW → `[m, v]`).
- **`FlatShardPlan::try_new`** (+ `FlatShardError`): the non-panicking constructor for untrusted inputs
  (a manifest parsed from disk). `new` delegates to it (panics only for trusted model code).

### Gates (28 tests green; in-scope `clippy -D warnings` + `fmt` clean)
- **byte framing** (7 inline): manifest/shard round-trip; bad-magic, trailing-bytes, truncation,
  stale-step, wrong-rank all detected.
- **save-K / reshard-J**: the loaded global is bit-identical for every `K∈{1,2,4}`; re-saved with every
  `J∈{1,2,4}` into fresh shard files and reloaded, the round-tripped global (param **and** planes **and**
  step) is identical and the forward matches — a real disk reshard.
- **distributed resume (bit-exact)**: train `HALF` steps on world `W∈{1,2,4}`, DCP-save mid-run, restore,
  continue — the resumed loss curve equals the uninterrupted curve bit-for-bit. Proves `m`/`v` + step
  survive the round-trip, not just params.
- **fault injection / crash atomicity**: a fully-written-but-uncommitted newer save does not shadow the
  committed checkpoint; non-monotonic re-save → `NonMonotonicSave`; swapped shards → `ShardMismatch`;
  truncated → `Truncated`; missing → `MissingShard`; corrupt manifest (`world==0` → `InvalidManifest`,
  huge `n_planes` → `TooManyPlanes`, bad version/magic) — never a panic, never a silent corrupt load.

### Hardened by adversarial review (load-bearing)
- The review caught that `load` (the untrusted-bytes entry point) **violated its never-panic contract**
  four ways — a corrupt manifest with `world==0`, overflowing `Σleaf_lens`, overflowing `chunk*world`, or
  a huge `n_planes` would panic/abort the loader. Fixed: `FlatShardPlan::try_new` for the structural
  fields, an `n_planes` bound (`MAX_STATE_PLANES`) before any allocation, and `ValueTooLarge` for
  oversized lengths — now all clean `DcpError`s (re-verified inline: removing the `n_planes` bound makes
  the gate panic with "capacity overflow").
- Also: a same-step re-save could tear the checkpoint (shard filenames key on step) → `save` now enforces
  a **monotonic-step** contract (`NonMonotonicSave`); and two gates lacked teeth (the reshard gate was a
  vacuous in-memory identity; the fault test used garbage orphans) → both rebuilt to round-trip real
  shard files through disk (re-verified inline: reversing shard reassembly now fails the reshard gate).

### Notes
- CPU sim, single-writer-per-directory. **Deferred (documented):** keep-last-N GC of superseded shards;
  a distributed commit barrier (0017's NCCL job). Next: the rented-2×GPU wall (0017–0018 → `v0.60.0`).
  See `docs/plans/0016`.

## [0.5.5] — 2026-06-21 — ZeRO-3 / FSDP over the simulated `ProcessGroup`: gradient + loss parity

The fifth v0.60 increment (single-GPU-reachable; `0.5.x` line). The ADR-0008 "N-GPU vs 1-GPU loss
parity" gate, reachable on one machine: shard a tiny model's params/grads/optimizer state across `world`
simulated ranks (ZeRO-3 / FSDP) over the 0014 `SimProcessGroup` — `all_gather` params before the
forward, `reduce_scatter(Avg)` grads after the backward, sharded AdamW step — and prove the result
matches an independent single-process full-batch reference.

### Added
- **`fsdp::FlatShardPlan`** (re-exported at the crate root): the FSDP "FlatParameter" descriptor.
  Concatenates the trainable leaves into one flat buffer, pads to a multiple of `world`, splits into
  `world` equal contiguous shards; `flatten`/`unflatten` between the leaf and padded-flat views,
  `shard_range` per rank. The `chunk * world` length is overflow-checked once in `new()`. This is the
  seed of 0016's distributed-checkpoint manifest `(global_shape, local_offset, shard_spec)`.
- The FSDP training orchestration (gather → fwd/bwd → reduce_scatter → sharded step) lives in the gate
  test, parameterized by the model's fwd/bwd closure — not a premature `fsdp_step` helper (single
  consumer today; cheap retrofit). Whole-model single-unit gather; per-layer gather/free is the deferred
  memory optimization (does not change the curve).

### Gates (12 tests, green; in-scope `clippy -D warnings` + `fmt` clean)
- **`FlatShardPlan`** (6 unit): flatten/unflatten roundtrip identity; `shard_range` partitions the
  padded buffer; padding zeroed; no-padding + `world=4` padding cases; `world=0` panics.
- **world=1 FSDP == baseline**, bit-exact (the orchestration is a faithful refactor of the plain loop —
  *different code*, so a real teeth check).
- **replicated-data FSDP == baseline**, bit-exact for **world∈{2,4}** (`ΣG/world == G` exactly — the
  running-sum rounding cancels under /2 and /4; verified over 20 M random gradients, and that it does
  *not* cancel under /8). Isolates the sharding mechanics from data-parallel reordering.
- **reduced gradient == full-batch gradient (the teeth)**, world∈{2,4}: the reassembled FSDP-reduced
  gradient matches the single-process full-batch gradient directly (partitioned ~1e-7; replicated
  bit-exact). Not scale-invariant — catches a wrong `ReduceOp` / dropped reduction / wrong slice.
- **partition loss-curve tracking**, world∈{2,4}: end-to-end curve within ≈1 ULP (~4.5e-8, measured) of
  the baseline — a convergence-tracking check, not the reduce-op gate.
- **determinism**: the FSDP loss curve is bit-identical across thread reschedulings (the fixed-order
  collective fold through a real training loop).

### Hardened by adversarial review (load-bearing)
- The review caught that the loss/param-curve parity gate was **blind to a wrong reduce op**: AdamW's
  update `m̂/(√v̂+ε)` is **scale-invariant in the gradient**, so an `Avg→Sum` (gradients `world`× too
  large) cancels in the adaptive normalizer — only the `ε` floor leaks (~5e-7, under the 1e-4 gate).
  Reproduced at HEAD, then fixed by adding the **gradient-level** assertion above (mutation-verified:
  `Avg→Sum` now fails it). The loss-curve gate was reframed as convergence-tracking, not the reduce-op
  gate.
- The review also corrected the plan's IEEE-754 rationale (it claimed "only world=2 replicated is
  bit-exact" and implied powers of two are safe — both wrong); the verified statement is world∈{2,4}
  bit-exact, world=8 not.

### Notes
- Model is a dense 2-layer MLP on the gradient-checked tape ops (0011) — FSDP shards flat f32 master
  weights identically whether or not the forward quantizes; ternary/QAT is gated separately (0013), and
  parity (not convergence) is what 0015 proves. CPU sim. Next on the `0.5.x` line: distributed
  checkpoint + resharding + fault injection (0016), then the rented-2×GPU wall (0017–0018 → `v0.60.0`).
  See `docs/plans/0015`.

## [0.5.4] — 2026-06-21 — Distributed collectives: `ProcessGroup` trait + thread-simulated backend

The fourth v0.60 increment (single-GPU-reachable; `0.5.x` line). The ADR-0008 collective-correctness
substrate, fully reachable on one machine: an object-safe `ProcessGroup` trait (the abstraction the
real `cudarc::nccl` backend implements at the 0017 wall) plus a **deterministic** thread-simulated
backend — N logical ranks in N threads over a shared host buffer — so the load-bearing gate,
*all-reduced grads == a single-process summed reference*, goes CI-green with no second GPU.

### Added
- **`ProcessGroup`** (object-safe) in `tritium-train::dist`: `all_reduce` / `reduce_scatter` /
  `all_gather` / `broadcast`, each `-> Result<(), DistError>`; `ReduceOp::{Sum, Avg}`. Re-exported at
  the crate root.
- **`SimProcessGroup`** — the "gloo-for-CI" backend. `SimProcessGroup::world(n)` hands back `n` handles
  sharing one `Arc<SimShared>` (per-rank staging slots behind a `Mutex` + a `std::sync::Barrier(n)`).
  Every reduction folds the slots in **fixed rank order `0..world`**, so the result is independent of
  thread scheduling and bit-identical to a single-process reference (f32 add is non-associative — the
  bridge to 0015's loss-parity and 0017's wire-correctness gates).
- **`DistError`**: `LengthMismatch` / `LengthOverflow` / `InvalidRoot` / `CollectiveMismatch` /
  `Backend` — every collective misuse returns rather than panics.

### Gates (14 tests, green)
- **all_reduce == single-process summed reference**, bit-exact, `world∈{1,2,4,8}` + a proptest over
  `world∈[1,8]`, `n∈[1,40)`; reduce_scatter / all_gather / broadcast match their references; `Avg` =
  `Sum/world` for all_reduce **and** reduce_scatter.
- **determinism** under thread scheduling for all four collectives (40–50× rerun bit-identity).
- mis-sized buffers → `Err`, never a panic; in-scope `clippy -D warnings` + `fmt` clean.

### Barrier protocol — hardened by a 10-lens adversarial review (load-bearing)
- The review caught a real **deadlock**: `reduce_scatter`/`all_gather`/`broadcast` did a local
  size/root pre-check and `return Err` *before* `publish()`/barrier #1, so a size-disagreeing rank did
  **zero** `barrier.wait()` calls while peers blocked at barrier #1 forever (a CI hang). Fixed by
  making every collective uniform — **"publish first → validate after barrier #1 → ALWAYS reach
  barrier #2"**: exactly two `barrier.wait()` on every Ok/Err path. A timeout-guarded multi-rank
  regression test provably **hangs→fails against the old code** and passes in 0.04 s against the fix.
- Review-driven additions: a per-collective **op-tag** turns a cross-collective desync into a clean
  symmetric `CollectiveMismatch` (no hang); `LengthOverflow` replaces the dishonest `usize::MAX`
  overflow sentinel.

### Notes
- CPU sim. **Documented limitations:** `std::sync::Barrier` has no break/poison (a rank that *panics*
  inside a collective strands its peers); a *different-count-of-collectives* desync is uncatchable
  (the op-tag guard only catches same-step type/order divergence) — the real NCCL backend (0017)
  brings its own timeout/abort. Next on the `0.5.x` line: ZeRO-3/FSDP over this `ProcessGroup` (0015),
  distributed checkpoint/resharding/fault-inject (0016), then the rented-2×GPU wall (0017–0018 →
  `v0.60.0`). See `docs/plans/0014`.

## [0.5.3] — 2026-06-20 — GPU pretrain smoke: training step wires the grad kernels + LR schedule

The third v0.60 increment (single-GPU-reachable; `0.5.x` line). The formerly dead-code CUDA gradient
kernels are now wired into a converging GPU QAT training step, with a learning-rate schedule and a
from-scratch tiny-model pretrain smoke — the ADR-0008 "tiny model reaches target loss" gate on one GPU.

### Added
- **`LrSchedule`** (`tritium-train`): pure-CPU linear-warmup → cosine-decay, `lr(step) -> f32`.
- **`ternary_matmul_forward`** CUDA kernel (the f32 forward companion to `grad_a/grad_w/grad_s`,
  same `--fmad=false` reduction order) + a `train_forward` host binding.
- **GPU QAT training step** (`tritium-cuda::train`, behind `cuda`): a 2-layer ternary MLP
  (`x →[Wq₁,s₁] → relu² → [Wq₂,s₂] → MSE`) with f32 master weights (off-grid init), a **learned**
  per-row output scale (quantizer-scale stop-gradiented), STE master updates
  (`ste::quantize_vjp`), and per-leaf `AdamW` driven by the schedule. `pretrain_smoke` runs it from a
  seeded init. `tritium-train` becomes a `cuda`-gated dependency (the grad kernels' first real consumer).

### Gates
- **device step == CPU tape** within 1e-4 (the composed 2-layer step's gradients vs `tritium-train`'s
  matmul vjp) **plus** a CPU **finite-difference gradcheck** of the composition's weight-master
  gradients against the smooth STE-surrogate loss — which (unlike the parity gate) catches a coherent
  miswiring shared by both engines (mutation-verified).
- **pretrain smoke**: loss `0.261 → 0.025` (90.5% drop, gated `<0.30`), no NaN, ~0.98 ms/step.
- LR-schedule property tests; `clippy -D warnings` + `fmt` clean; **`compute-sanitizer` memcheck:
  0 errors**; full 45-test cuda suite green; default no-CUDA build inert.

### Notes
- CPU + 1 GPU. The full-2B **resident** training engine (training-mode resident decoder, activation
  retention/checkpointing) is deferred — measured immaterial at smoke scale. Next on the `0.5.x` line:
  distributed correctness via a thread-simulated `ProcessGroup` (0014–0016), then the rented-2×GPU wall
  (0017–0018 → `v0.60.0`). See `docs/plans/0013`.

## [0.5.2] — 2026-06-20 — Data pipeline: deterministic resumable sharded sampler + `.tqbin`/`.tqidx`

The second v0.60 increment (single-GPU-reachable; still on the `0.5.x` line). The data substrate a
pretraining loop consumes: a deterministic, resumable, dup/loss-free sharded sampler plus the two
little-endian corpus formats.

### Added
- **`DataSampler`** (`tritium-train`): per-epoch Fisher–Yates shuffle of `0..N` driven by a
  `splitmix64`-seeded `xorshift64` stream (integer-only — reproducible across machines), with a
  **strided** rank partition (rank `r` takes `perm[r], perm[r+n_ranks], …`) so the union over ranks
  is exactly `0..N` with no duplication or loss, and a resumable `(seed, epoch, consumed)` cursor
  that restores the exact remaining order across epoch boundaries. `drop_last` gives every rank
  `floor(N/n_ranks)` samples; the per-epoch permutation is memoized so streaming an epoch is O(N).
- **`.tqbin`** (`tritium-format`): a tokenized-corpus shard — LE `magic | version | n_tokens | u32
  tokens`. **`.tqidx`**: the manifest — `seq_len`, ordered shard list, per-shard token counts; the
  global sample count is `Σ_shard floor(n_tokens / seq_len)` (`TqIndex::n_samples`). Both parsers
  are *total*: magic/version enforced, every length bounds-checked against the buffer before
  allocating (the `checkpoint.rs::f32_vec` discipline) — arbitrary bytes error, never panic/OOB/OOM.
  A shared never-panic `LeCursor` backs both.

### Gates
- Sampler property tests (proptest): deterministic permutation, exact coverage (no dup/loss), equal
  drop-last counts, exact mid-epoch resume across an epoch boundary (both `drop_last` modes).
- Parser fuzz (proptest): `read_tqbin`/`read_tqidx` total on arbitrary + crafted-header bytes.
- `clippy -D warnings` + `fmt --check` clean. An adversarial multi-lens review (23 agents) confirmed
  + fixed two majors — the `drop_last` zero-data guard and O(N) permutation memoization — plus a
  `TqBadName` UTF-8 error and determinism/resume doc accuracy.

### Notes
- CPU only. Next on the `0.5.x` line: the GPU-resident training loop + pretrain smoke (0013), then
  distributed correctness via a thread-simulated `ProcessGroup` (0014–0016), then the rented-2×GPU
  wall (0017–0018 → `v0.60.0`). See `docs/plans/0012`.

## [0.5.1] — 2026-06-20 — Full-model autograd backward (v0.60 foundation)

The first v0.60 increment (single-GPU-reachable; the v0.60 milestone proper is gated on ≥2 GPUs).
The eager reverse-mode tape now backprops a **whole transformer end-to-end** — the foundation a
pretraining loop needs, and the wall the v0.50 capstone explicitly deferred.

### Added
- New gradient-checked `tritium-train` tape ops (forward + vjp, each central-FD-checked at
  Gate-C 2e-3): **RMSNorm**, **row-wise softmax** + **causal mask**, **NeoX RoPE** (matching the
  inference `rope_apply` convention; vjp = the inverse rotation), and a **dense transpose** (for
  attention's `P·V`). The gated squared-ReLU MLP + residuals compose from existing ops.
- A composed **single-head tiny-transformer end-to-end gradient check** — rmsnorm → q/k/v → RoPE →
  scaled causal-masked softmax attention → o_proj → residual → gated-relu²-MLP + sub-norm →
  residual → output-norm → LM head → MSE — with every trainable leaf's analytic gradient matched
  to a per-element central finite difference. The v0.50-deferred full-model-backprop wall, now green.

### Notes
- CPU only. The rest of v0.60 follows as the `0.5.x` line: the GPU-resident training loop +
  pretrain smoke (0013), distributed correctness via a thread-simulated `ProcessGroup` (0014–0016),
  then the rented-2×GPU wall (0017–0018 → `v0.60.0`). See `docs/plans/0011`.

## [0.5.0] — 2026-06-20 — Training Core (STE autograd + QAT + optimizer + LoRA + CUDA backward)

The v0.50 milestone (ADR 0007): a single-node training core for ternary BitNet models. New crate
`tritium-train` — reverse-mode autograd over a flat tape of hand-written `forward`+`vjp` ops,
validated by finite-difference gradient checks (Gate C) — plus the optimizer, checkpoint, LoRA, the
CUDA backward kernels, and a heal bridge that drives the training core end-to-end on the real model.

### Added
- **`tritium-train` crate** (`#![forbid(unsafe_code)]`): a reverse-mode autograd `Tape` over an f32
  value arena, with hand-written `forward`+`vjp` for STE-quantize (straight-through estimator),
  ternary matmul, plain dense matmul, bias, squared-ReLU, MSE / softmax-cross-entropy, element-wise
  add/mul, `detach` (stop-gradient), and `scale_const`. A `gradcheck` harness finite-differences
  every trainable op — **Gate C green on CPU** (the STE vjp is the exact gradient of the
  differentiable surrogate `clamp(Wf/s_q)`, not the rounded forward).
- **AdamW optimizer** (decoupled weight decay, eps-outside-sqrt, bias-corrected) behind a minimal
  `Optimizer` trait, with a versioned, never-panic **`TOPT` training checkpoint** — optimizer state
  save/restore is **bit-exact** and a resumed run equals the uninterrupted run; no NaN/Inf over ≥1k
  steps. (Muon/Lion/Adafactor/… surveyed; AdamW is the baseline, the rest deferred with rationale in
  `docs/plans/0008`.)
- **LoRA adapters on a frozen ternary base**, composed from reusable primitives (`dense`/`detach`/
  `scale_const`): the frozen base receives **exactly zero** gradient, adapter A/B gradients match
  finite difference, the merge folds into a dense weight correctly, and rank edges `r=1` / `r=full`
  pass — all proptested.
- **CUDA f32 backward kernels** (`tritium-cuda`): `ternary_matmul_grad_a/_w/_s`, gradient-checked
  against the CPU `vjp` oracle (parity ≤1e-4 across shapes incl. tails + multi-block) and
  `compute-sanitizer memcheck`-clean — **Gate C green on CUDA** (deterministic: one thread per
  output, no atomics, `--fmad=false` for host bit-parity).
- **QAT heal bridge**: `TernaryLinear::replace_weights` (re-pack TQ2_0 + upload a re-trained ternary
  weight in place), `Projection::as_ternary_mut`, and `ModelRunner::invalidate_resident` (drop the
  cached device-resident decoder so a post-swap forward rebuilds from current weights).
- **Capstone convergence gate** (`tritium-nn`, GPU+model-gated): the QAT machinery (AdamW + STE +
  tape) drives a real BitNet-2b4t attention slice's ternary **distillation loss down ≥90%** on real
  model activations, end-to-end through the bridge (measured ~94.6%), loss-decreases, no-NaN.

### Notes
- **Scope honesty (ADR 0007 recovery gate):** a *full-model ≥90% perplexity-recovery* gate is not
  meaningful on BitNet-2b4t — its bf16 "master" is the QAT *latent* weight (garbage when run densely;
  `salt_accuracy.rs` documents this), it is already per-tensor-QAT-optimal (naive ternary ≈ deployed,
  so no gap to heal), and layerwise distillation from a short eval slice is underdetermined for
  2560-wide layers. The capstone therefore certifies **distillation-loss convergence** of the
  training core on the real model; a true model-wide PPL-recovery gate needs full-model backprop and
  a non-QAT-latent checkpoint — deferred to v0.60. Full reasoning in `docs/plans/0010`.
- **Multi-GPU is out of scope** (v0.60): this milestone is single-node.

## [0.4.1] — 2026-06-20 — Split-KV decode attention + IMMA OOB fix (perf/correctness point-release)

A capability-neutral point-release: it adds no new public API (the SALT staircase is unchanged),
only performance and correctness depth on the v0.4.0 surface. Headline: flash-decoding (split-KV)
attention wired into the resident M=N decode, and a fixed out-of-bounds read in the JIT IMMA
kernel on tail shapes.

### Added
- **Flash-decoding (split-KV) attention** (`tritium-cuda`): two new kernels —
  `gqa_attention_split_partial_f32` (warp per `(row, head, key-chunk)`; online-softmax over its
  key chunk → partial `{acc, m, l}`) and `gqa_attention_combine_f32` (flash-merge the `S` partials)
  — with a fixed chunk size so the captured CUDA-graph grid is valid for every decode step. Wired
  into **both** M=N launch sites (eager `decode_batch` + graph `decode_batch_graph`), so graph and
  eager stay bit-identical. An equivalence gate (`attn_split_kv_matches_direct_attention`) pins the
  split-KV output to the direct attention reference within tolerance.
- **U5 fuzz targets** (`cargo-fuzz`) for the two v0.4.0 parsers (SALT bundle + SALT-in-GGUF).
- **Runnable SALT example** (`crates/tritium-quantize/examples/salt_roundtrip.rs`): fp32 matrix →
  `quantize_tensor` → both containers → read back → identical-dequant assert (ADR 0002 U9).

### Fixed
- **JIT IMMA tail-shape OOB read** (U7): the JIT `tq2_0_imma_mpgemm` kernel read 1–2 bytes past
  the packed weight buffer when an autotuned `tile_n > IMMA_N` and `N` was not a tile multiple (the
  high sub-tile index exceeded `ceil(N/IMMA_N)` n-tiles). Added an `nt` bound to the weight-stage
  read guard; the padding tile was already zeroed and masked on output, so results are unchanged.
  `compute-sanitizer memcheck` is now clean on the CUDA suite.
- `cuda_batch_decode_matches_single` uses `sample_greedy` (deterministic tie-break, NaN-safe)
  instead of a panicking argmax.

### Performance
- Split-KV cuts the decode attention kernel from **57.6% → 26.6%** of N=1 GPU time (~2.2× on the
  attention kernel; nsys per-kernel breakdown). End-to-end N=1 throughput is ~flat (N=1 is
  occupancy-bound across the whole pipeline — GEMM/rmsnorm/lm_head now dominate); no regression at
  N≥2 (clean re-bench: N=64 graph ≈ baseline). See `docs/plans/0001` outcome + `docs/ROADMAP.md`.

### Internal
- `rustfmt` 1.9.0 across the workspace (greens the CI fmt gate); `.mimocode/` executor scratch
  gitignored.

## [0.4.0] — 2026-06-19 — SALT quantization (ADR 0001/0006)

SALT (Sensitivity-Allocated Layered Ternary) — the **`tritium-quantize`** crate, a TQ2_0
residual sidecar + whole-model bundle in `tritium-format`, the GPU multi-plane accumulate
kernel, and the `tritium quantize` CLI. All CPU exit gates green; the GPU kernel matches the
dequant reference; SALT is validated both ways — `salt@1.585 == deployed I2_S` on BitNet
b1.58, and a smooth monotone recon-error-vs-bpw curve on a normal fp model (gpt2). The three
items previously scoped as deferrable all landed this cycle: a **GGUF writer** + SALT-in-GGUF
container (`--format gguf`), a **resident-GPU SALT decode** primitive, and the **sparse
residual plane** + density switch. (This cycle also shipped Track-2 decode perf — see below
— orthogonal to SALT.)

### Added
- **tritium-quantize** (new crate) — the offline SALT quantizer:
  - **Residual ternary expansion** (`residual_expand`, `Plane`, `PlaneStack`): a weight
    group becomes `W ≈ Σ_p s_p·t_p`, fit greedily (each plane AbsMean-quantizes the prior
    residual). Prefix-stable, so `T=1` is *exactly* flat BitNet b1.58 AbsMean.
  - **Rate-distortion plane allocator** (`allocate`): greedy water-filling distributes a
    bits-per-weight budget across groups by sensitivity — the next plane goes to whichever
    group buys the most loss-drop-per-bit, `H_g·Δerr / (|g|·log2 3)`, capped at `T_max`.
  - **End-to-end tensor quantizer** (`quantize_tensor`): 256-block groups → global
    allocation → packed SALT rows; pluggable sensitivity (Uniform / Energy / Custom Hessian).
  - **Per-tensor base plane** (`ScaleGroup::Tensor`): the T=1 base uses one per-tensor
    AbsMean (residual planes stay per-block), reproducing the deployed BitNet b1.58 I2_S
    ternary — a QAT-trained master only works through its exact per-tensor clip, so the
    per-256-block default reconstructs the *latent* weights too faithfully and yields a
    non-working model. `Block` stays the default for normally-trained masters.
- **tritium-format** — the **TQ2_0 residual sidecar** (`SaltRow`, `pack_salt_row`,
  `unpack_salt_row`, `read_legacy_as_salt`, `dequant_salt_row`): `T` ternary planes, each a
  standard TQ2_0 row, so a `T=1` row is byte-identical to legacy plain-TQ2 and pre-SALT
  models load unchanged as flat AbsMean. Plus the whole-model **SALT bundle**
  (`write_salt_bundle`/`read_salt_bundle`, magic `b"TSLB"`) — a single-file container with a
  per-tensor index; the reader is hardened against malicious input (no OOB/overflow/OOM).
- **tritium-cuda** — the **SALT multi-plane GPU GEMM** `salt_mpgemm_tiled_f32`
  (`Σ_p scale_p·trit_p`, per-block f16 scales read from the weight bytes), matching the
  `dequant_salt_row` → fp32 reference within 1e-4 (`salt_mpgemm_matches_dequant_reference`).
- **tritium-cli** — `tritium quantize --input <safetensors> --output <out> --bpw <f>
  --scale-group {block|tensor} --format {sidecar|gguf}`: SALT-quantize every 2D weight of an
  fp model to a SALT bundle **or** a GGUF container (validated end-to-end on gpt2).
- **tritium-format GGUF writer** — `write_gguf` (+ `TensorOut`), the exact inverse of
  `read_gguf`: serializes a metadata table + tensor payloads so `read_gguf(write_gguf(..)) ==`
  input. Plus the **SALT-in-GGUF container** (`write_salt_gguf`/`read_salt_gguf`): a whole SALT
  model in a GGUF envelope, one tensor per SALT tensor under the tritium-private type id 169,
  payload = the per-tensor `pack_salt_row` blobs; reader walks rows self-describingly and is
  hardened against malicious input. Backs `quantize --format gguf`.
- **tritium-cuda resident-GPU SALT decode** — `CudaBackend::upload_salt` + `salt_forward`
  (`SaltResidentLinear`): the `salt_mpgemm_tiled_f32` kernel now runs against a VRAM-resident,
  plane-major weight uploaded once (ragged plane counts zero-padded), feeding the raw f32
  activation + per-block scales — the building block a full SALT decode forward composes per
  projection. (Previously the kernel was reachable only via a test that re-uploaded per call.)
- **tritium-format sparse residual plane** (`sparse.rs`, ADR 0001 §5) — `SparsePlane` stores a
  pruned plane as nonzeros only (ascending `(column, sign)` + per-block scales),
  `sparse_from_tq2_0`/`sparse_to_tq2_0` round-trip **byte-identically** to dense TQ2_0, and
  `choose_plane_repr` is the density switch (sparse below ~10% nonzero, else dense). The
  GPU sparse-matmul kernel (the per-arch compute win) is the v0.5+ follow-on.

### Validation (CPU exit gates, ADR 0006)
- `T=1` reduces **exactly** to flat AbsMean (golden + proptest, bit-exact reconstruction).
- Reconstruction error is **monotonic** non-increasing in plane count `T` (proptest; sound —
  with round-clamp quantization per-element `residual² ≤ w²` whenever `scale ≥ 0`).
- Allocator **never exceeds the budget**; **ordering invariant** (equal curve+size ⇒ higher
  sensitivity gets ≥ planes); **determinism** (byte-identical output).
- Sidecar **roundtrips** multi-plane weights, **reads legacy plain-TQ2**, **enforces
  version + magic**, handles pruned / zero-variance / partial-block edges.
- End-to-end `dequant_salt_row(quantize_tensor(..)) ==` independent re-expansion reference,
  bit-exact; `budget == base` reproduces flat AbsMean through the whole pipeline.
- **GPU multi-plane accumulate** matches the dequant→fp32 reference within 1e-4.
- **Accuracy (reframed for a QAT-ternary master, ADR 0006):** the original "within the fp16
  gap" gate is ill-defined for BitNet b1.58 — its bf16 "master" is *latent* QAT weights, not a
  usable forward (raw-master perplexity is garbage; the SALT curve *inverts*, higher bpw →
  worse). Reframed to (a) `salt@1.585 == deployed I2_S` on b1.58 (the per-tensor base
  reproduces the GGUF weights to f16; `tritium-nn` `salt_accuracy` + `gguf_eval_perplexity`),
  and (b) a **smooth monotone recon-error-vs-bpw curve on a normal fp model** (gpt2:
  0.540→0.387 over 1.585→3.0 bpw; `tritium-quantize` `recon_curve`). A full Qwen-arch
  perplexity curve is the deferred "real accuracy" follow-on.

### Validation — the three landed items
- **GGUF**: `write_gguf` round-trips through `read_gguf` for every value type, custom
  alignment, empty/nested arrays, version rejection; `write_salt_gguf`/`read_salt_gguf`
  round-trips exactly (incl. ragged plane counts + non-256-multiple `k`), parses as a valid
  GGUF, dequant-equal after round-trip, rejects non-SALT GGUF, fuzz-safe on truncation.
- **Resident SALT**: `salt_resident_forward_matches_dequant` — T=1/2/3 incl. ragged, each row
  vs host `dequant_salt_row → fp32 matmul` within 1e-4, two forwards on the same resident
  buffer agree.
- **Sparse plane**: byte-identical dense round-trip; `sparse_dot` **bit-exact** vs the dense
  dot; density switch picks sparse@2.5% / dense@50% and both expand identically; pruned plane
  packs smaller; pack/unpack round-trip; corrupt/truncated/oversized-`k` input errors, never
  panics. (Test load-bearingness confirmed by mutation.)

### v0.5+ follow-ons (not v0.4.0 gates)
- A full SALT decode **forward** composing the resident primitive across every projection.
- The per-arch GPU **sparse-matmul** kernel (the compute win atop the sparse storage form).
- A full **Qwen-arch perplexity** curve (the "real accuracy" validation b1.58's latent master
  cannot give).

### Track-2 decode performance (this cycle; orthogonal to SALT)
- **CUDA-graph batched M=N decode** (`decode_batch_graph`) — graph-captures the M=N forward;
  2.2–4.9× over the eager path, bit-identical (`cuda_batch_decode_graph_matches_eager`).
- **On-device greedy sampling + tiled LM head** (`decode_batch_graph_argmax`) — folds the LM
  head + argmax into the graph (returns N token ids, not N·vocab logits); an nsys-guided tiled
  LM head reads the f16 embd table once per 8-row tile. **474 tok/s at N=64 = 3.34× the M=1
  142** (`cuda_batch_decode_graph_argmax_matches_greedy`).

## [0.3.7] — 2026-06-17 — Batched M=N decode (N concurrent sequences)

Phase 2 of the M>1 work: decode N independent sequences in one device-resident M=N
forward — the serving-throughput primitive (M=N fills the GPU that a single M=1 token
can't, decode being occupancy-bound at ~19% util). Bit-match-preserving.

### Added
- **tritium-cuda** — `CudaDecodeModel::decode_batch` + `BatchKv`: each of N sequences has
  its own KV slice (`[n, max_ctx, kv_width]` per layer) and position; row r attends seq r's
  KV up to seq r's position. Two new per-sequence kernels — `kv_append_mdecode_f32` and
  `gqa_attention_mdecode_f32` (per-seq KV base + per-row limit) — plus the reused v0.3.6
  M>1 batch kernels. `new_batch(n)` allocates the per-seq KV + M=N scratch.
- **tritium-nn** — `ModelRunner::resident_cuda()` accessor (advanced/test) for the
  device-resident decoder (M=1 graph + batched M=N path).

### Validation
- `cuda_batch_decode_matches_single` — **bit-identical**: two identical sequences in one
  batch produce byte-for-byte identical logits (they are independent), and each matches the
  single-sequence `step_graph` reference (the batch kernels share the M=1 reduction order).

### Deferred (→ v0.3.8 / v0.4.0)
- A batched generate API / continuous batching (dynamic batch, paged KV) over `decode_batch`;
  a batched LM head (read token_embd once for all N rows) + batched prefill into per-seq KV.
  Then v0.4.0 (SALT, ADR 0006).

## [0.3.6] — 2026-06-17 — Batched M=P prefill (kills the sequential TTFT cliff)

The first M>1 step: prefill the **whole prompt in one device-resident forward** instead of
looping the M=1 decode graph over prompt tokens. Bit-match-preserving — greedy 256/256
exact, perplexity 2.96e-3, cpu↔cuda parity identical all still green; the batched-prefill
KV/logits match the sequential loop per row.

### Added
- **tritium-cuda** — M>1 batched kernels (`rmsnorm_batch_f32`, `embedding_gather_batch_f32`,
  `rope_apply_batch_f32`, `act_quant_batch_f32`, `scale_mul_batch_f32`, `kv_append_batch_f32`,
  `gqa_attention_batch_f32` — causal [m, ctx]: query row r attends keys 0..=causal_offset+r),
  each bit-identical per row to its M=1 sibling. `CudaDecodeModel::prefill` runs the M=P
  forward (eager safe launches — one-shot, no graph), q/k/v share one activation quant, the
  tiled GEMM handles M>1 via grid.y, final norm + f16 LM head on the last row only.
- **tritium-nn** — the runner prefills a multi-token prompt via `prefill` (one forward);
  single-token decode keeps the M=1 CUDA graph.

### Performance
- The sequential prefill re-read the 533 MB ternary weights **once per token** (memory-bound,
  ~3.6 s for a 512-token prompt). Batched reads them **once** + does the compute, so a long
  prompt prefills **~20-30× faster** (O(1) weight reads vs O(P)). For short prompts the gap is
  small and TTFT-negligible (~84 vs ~42 ms, dwarfed by decode), so prefill stays always-on.
- Decode is unchanged (~142 tok/s).

### Deferred (→ v0.3.7)
- Batched M=N **decode** (N concurrent sequences for aggregate throughput) — reuses these
  M>1 kernels but needs per-sequence KV + a per-row-KV attention + a batched generate API.
  Plus a precise long-prompt prefill benchmark; then v0.4.0 (SALT, ADR 0006).

## [0.3.5] — 2026-06-17 — Structural decode: shared quant + fused GEMMs (~142 tok/s)

Decode is occupancy/latency-bound at M=1 (~990 small graph nodes in a serial chain), so
v0.3.5 cuts the chain — bit-match-preserving as always: **~142 tok/s typical** (range
~140–148; **5.1× over the v0.3.1 eager path**), ~17% of the roofline. Greedy 256/256 exact,
perplexity 2.96e-3, cpu↔cuda parity identical all hold.

### Performance
- **Shared activation quant** — `g_gemm` split into `g_quant` + `g_matmul`, so q/k/v (and
  gate/up), which all project the same `d_normed`, quantize it once instead of per-GEMM.
- **Fused q‖k‖v and gate‖up GEMMs** (`ResidentLinear::build_fused`) — concatenate the
  parts' TQ2_0 weight rows (dtod) + scales into one arena, so a single tiled GEMM emits all
  parts' outputs (q/k/v are offset slices of `d_qkv`; gate/up halves of `d_gateup`). Three
  serial GEMMs → one bigger, better-occupancy kernel; **bit-identical** (each output's
  warp-reduce is unchanged, only the grouping). The bigger win (~+13%); costs ~340 MB of
  fused-weight arenas (the eager path keeps the Arc-shared separate weights).

### Notes — single-sequence near its M=1 ceiling
- During decode the 4090 is at ~19% utilization / ~70 W of 450 W (boost, not throttling).
  A single-token forward can't fill the GPU; per-kernel/fusion tuning is largely exhausted
  at ~142 tok/s. The next real lever is **batched (M>1) decode** for aggregate throughput
  (a separate, larger change), not more single-sequence tuning.

### Deferred (→ v0.3.6 / v0.4.0)
- Batched M>1 decode; batched device prefill; IMMA prefill (#28); live `ncu`. Then v0.4.0
  (SALT, ADR 0006).

## [0.3.4] — 2026-06-17 — Decode toward the roofline (~120 tok/s, still 256/256)

Continues the decode optimization, all **bit-match-preserving**: **~120 tok/s typical**
(range ~114–131; 85.5 → ~120, **~4.4× over the v0.3.1 eager path**), ~14% of the memory
roofline. Greedy 256/256 exact, perplexity 2.96e-3, cpu↔cuda parity identical all hold.

### Performance
- **Shared-staged rmsnorm** (`rmsnorm_shared_f32`) — rmsnorm was the #1 remaining cost
  (thread-0 sum, latency-bound on serial global reads). The block now stages the row into
  shared with a coalesced load, then thread 0 sums **from shared in the same order** — so
  the f32 sum is byte-identical (greedy holds) but compute- not latency-bound. ~8× faster
  rmsnorm; 83.8 → 113.7 tok/s. (The biggest single v0.3.4 win.)
- **Branchless ternary decode** in the f32 graph GEMM — replaced the divergent
  `if code==2/else if code==0` with `acc += a*(code-1)` (bit-identical for codes {0,1,2});
  removes warp divergence. 116.1 → 131.5 tok/s.
- **f16 `token_embd`** for the graph LM head (`lm_head_warp_f16`) — f16 is the GGUF's
  native precision (widened to f32 losslessly), so the f16 read is bit-identical at half
  the bytes. +2% (the LM head isn't the bottleneck).

### Notes — occupancy-bound at M=1
- During decode the 4090 sits at **~19% utilization / ~70 W of 450 W** (boost clock, not
  throttling): a single-token forward is too small to fill the GPU. The wall is launch/
  occupancy, not compute or bandwidth (14% of roofline). Further decode speedup is
  **structural** — batched decode, kernel fusion — not more per-kernel tuning.
- A *parallel* (tree-reduction) rmsnorm would reach ~132 tok/s but reorders the sum and
  breaks the gate (greedy diverges at token 109, fails lockstep parity); kept the bit-exact
  shared-staged version instead.

### Deferred (→ v0.3.5 / v0.4.0)
- Structural decode throughput (batched M>1 decode, GEMM fusion); batched device prefill;
  IMMA prefill (#28); live `ncu`. Then v0.4.0 (SALT, ADR 0006).

## [0.3.3] — 2026-06-17 — Parallelized decode kernels (85.5 tok/s, still 256/256)

A performance point-release continuing v0.3.2. **~1.86× more decode** (45.9 → 85.5 tok/s
on a 4090; **3.1× over the v0.3.1 eager path**), 10.1% of the memory roofline — and
**without giving up the greedy 256/256 bit-match** (perplexity 2.96e-3, cpu↔cuda parity
identical / worst logit rel 2.26e-6 all still green).

### Performance
- **Parallel `act_quant` absmax** — the per-token int8 absmax is now a block tree
  reduction (was a thread-0 sequential fold). `max` is associative, so the result is
  **bit-identical** to the sequential version; both the eager and graph paths use it.
- **Warp-per-head GQA attention** (`gqa_attention_decode_warp_g`) — the graph path's
  attention ran one thread per head (20/32 lanes idle); the warp version parallelizes
  across keys (lane-per-key dots) and output dims (lane-per-d weighted sums) with a lane-0
  softmax, so **no reduction is reordered** — bit-identical to the one-thread kernel.

### Notes
- A block-parallel rmsnorm (would have reached ~132 tok/s) was tried and **dropped**: its
  sum-of-squares reorder — though all-positive, ~1e-6 — flips a greedy near-tie by token
  109 and fails the lockstep parity, i.e. below the sanctioned perplexity+lockstep
  fallback. The graph keeps the bit-exact sequential rmsnorm.

### Deferred (→ v0.3.4)
- More headroom toward the roofline: the GEMM efficiency at M=1, an f16 `token_embd` for
  the LM-head read, a gate-holding parallel rmsnorm or a perplexity-fallback "fast mode".
  Plus the still-open items: batched device prefill, IMMA prefill (#28), live `ncu`.

## [0.3.2] — 2026-06-17 — CUDA-Graph Decode + the f32-accumulate win

A performance point-release on the v0.3.1 device-resident forward. **~1.66× decode**
(27.6 → 45.9 tok/s on a 4090) with **zero numerics regression** — greedy still 256/256
exact vs transformers, perplexity 2.96e-3, cpu↔cuda parity identical.

### Added
- **tritium-cuda** — a **raw-FFI CUDA-graph decode path** (`CudaDecodeModel::step_graph`):
  one captured graph replays the whole 30-layer forward per token. cudarc 0.19's safe
  launch is capture-incompatible (its per-buffer event waits trip
  `STREAM_CAPTURE_ISOLATION`) and hides the `CUfunction`, so the path raw-loads the PTX
  (`result::module::load_data`) for raw `CUfunction`s and launches via
  `result::launch_kernel` with pre-extracted stable `CUdeviceptr`s. New `_g` control-block
  kernels (`embedding_gather_f32_g`, `rope_apply_f32_g`, `kv_append_f32`,
  `gqa_attention_decode_f32_g`) read the per-token token/pos/cache_len from a device
  `int[4]`, so one graph replays across tokens.
- **tritium-cuda** — `tq2_0_add_mpgemm_tiled_f32` (f32-accumulate GEMM) and
  `lm_head_warp_f32` (coalesced warp-per-row LM head), used by the graph path.

### Performance
- The CUDA graph alone gave **no speedup** (collapsing ~930 launches/token → 1 replay was
  26.6 vs 27.6 tok/s) — which *proved* host launches were never the bottleneck. The real
  cost was the **double-precision GEMM accumulate** (the 4090 runs f64 at 1/64 the f32
  rate, × ~210 GEMMs/token) and the **uncoalesced 1.3 GB LM-head read**. The f32 GEMM
  (+15.8 tok/s) + warp LM head (+3.5) deliver the 1.66×.
- The eager `mpgemm`/`step` keep the double-accumulate kernel for the `1e-4` conformance
  bar over adversarial inputs; only the model-decode graph path uses f32 (the real
  activations stay ~2e-6 from the reference, far under the greedy tie margin).

### Fixed
- `step_graph` drains the default stream before replay, closing a latent cross-stream race
  if the eager `step` and `step_graph` are interleaved on one model (found by the
  adversarial review of the unsafe FFI, which otherwise verified the raw path sound).

### Deferred (→ v0.3.3)
- We are at ~5.4% of the memory roofline: parallelize the remaining sequential bit-match
  kernels (rmsnorm thread-0 sum, one-thread-per-head attention), f16 `token_embd`. Plus a
  live `ncu` artifact, batched device prefill, IMMA prefill (#28).

## [0.3.1] — 2026-06-16 — Device-Resident Decode Forward

The end-to-end performance point-release (ADR 0013): make the v0.3.0 forward fast
*end-to-end* with **zero numerics change**. BitNet 2B4T greedy still matches
transformers **256/256 exact**, perplexity **2.96e-3**, CPU↔CUDA parity identical —
now produced by a fully on-device decode that crosses the host boundary once per token
instead of ~210 times.

### Added
- **tritium-cuda** — `CudaDecodeModel`, a **device-resident M=1 decode forward**. The
  residual stream + per-layer KV cache live in VRAM across all 30 layers; every op runs
  on-device via new bit-matching decode kernels (`rmsnorm_f32`, `rope_apply_f32`,
  `gqa_attention_decode_f32`, `softmax_f32`, `residual_add_f32`, `embedding_gather_f32`,
  `lm_head_f32`, `act_quant_tiled_f32`, `scale_mul_f32`, `relu2_gate_f32`), all compiled
  `--fmad=false` and written sequential/no-FMA to reproduce the host f32 ops bit-for-bit.
  `build_decode_model` uploads dense weights once, precomputes the RoPE table, and shares
  the prefill path's ternary weights via `Arc` (no re-upload). **~6× decode speedup**
  (~27.6 tok/s vs the v0.20 host path's ~4.5 tok/s on a 4090).
- **tritium-spec** — defaulted `TernaryBackend::as_any()` downcast hook (returns `None`;
  CUDA overrides) so the runner can reach the concrete backend without touching the
  object-safe, host-slice-oriented trait.
- **tritium-nn** — the runner lazily builds + drives `CudaDecodeModel` for non-dump
  forwards on a CUDA backend (downcast dispatch); the host path stays the golden oracle.
  `tritium-cuda` is now an optional `cuda`-gated dependency (was dev-only).
- **tritium-benches** — `TRITIUM_2B4T_DECODE_4090`, the `BuiltOnBox` decode regression
  baseline the e2e gate keys on (our own measured figure, not a CPU competitor number).

### Numerics
- Softmax/attention `exp` is computed in **f64 then rounded to f32** (`exp_f32`) so it
  matches glibc `expf` (the host op) — the lever that holds greedy **bit-match** rather
  than dropping to the perplexity fallback. The only non-bit-exact op is this exp
  (≤2 ULP on ~0.05% of values); everything else is bit-exact vs the host.

### Deferred (→ v0.3.2)
- **CUDA-graph decode** — blocked by cudarc 0.19's safe launch (it waits on each
  buffer's pre-capture event → `STREAM_CAPTURE_ISOLATION`; the raw escape needs the
  `pub(crate)` `CUfunction`). Needs a parallel raw-FFI capture path + a device
  control-block kernel refactor; documented in the `#[ignore]`'d tripwire test. This is
  the launch-overhead win toward the memory roofline (decode is ~3.3% of SOL today).
- **`≥1.2×` competitor gate** — no same-HW GPU *ternary* baseline is obtainable:
  llama.cpp's CUDA backend has no TQ/I2_S mul-mat kernel and cannot load the I2_S
  artifact; bitnet.cpp's numbers are CPU. Awaits a measurable GPU competitor or the
  v0.3.2 graph (where a lead is unambiguous against the roofline).
- IMMA **prefill** path (#28); batched device prefill (today's prefill is sequential
  per-token decode); a live `ncu` artifact + the self-hosted GPU CI lanes.

## [0.3.0] — 2026-06-16 — Performance

The performance tier on the v0.2.0 spine — **fast kernels with zero numerics
change** (ADR 0005). BitNet 2B4T greedy still matches transformers 256/256 exact,
perplexity 2.81e-3, CPU↔CUDA bit-identical, with the new decode kernel as default.

### Added
- **tritium-cuda** — a **tiled add-only decode kernel** (`tq2_0_add_mpgemm_tiled`:
  warp-per-output, shared-mem-staged activations, warp reduction, f64 accumulate;
  auto-selected for decode) and an **IMMA int8 prefill kernel**
  (`tq2_0_imma_mpgemm`: `mma.m16n8k32` `s32.s8.s8.s32` tensor cores, exact int32
  accumulate, double-buffered shared unpack, `compute_80` second PTX). Fused
  `CudaBackend::mpgemm_with_act_quant` — on-device per-token int8 absmax quant →
  IMMA → scale fold. **WF-B autotune + nvrtc JIT**: `codegen::render_imma_source`
  over a `TileConfig`, a budget-pruned tile sweep, an on-disk cache keyed by
  arch+dtype+shape-bucket+CUDA-version; JIT == AOT bit-identical by construction.
- **tritium-spec** — optional `TernaryBackend::mpgemm_with_act_quant` (default impl
  = host W1.58A8); a GPU backend overrides it for the on-device fused path.
- **tritium-format** — `TernaryFormat::I2sInt8` + `convert_i2s_to_int8` (the IMMA
  tile interleave, byte-for-byte the kernel's B operand) and `convert_i2s_to_tq2_0`.
- **tritium-cpu** — AVX-512 + ARM NEON ternary kernels (bit-exact with scalar via a
  shared k-order fold) behind feature dispatch; the ISA-agnostic T-MAC LUT
  (implemented + unit-tested, off the hot path until its SIMD gather lands).
- **benches/** — divan CPU + GPU mpGEMM microbenches over 20 BitNet shapes, an
  end-to-end tokens/sec bench coupled to a perplexity check, a roofline ceiling
  (`peak_HBM / model_bytes` = 848.6 tok/s decode; 660.6 int8 TOPS prefill) + an
  `ncu` %-of-SOL recipe, and a `>5%` regression CI lane.

### Validated (RTX 4090, sm_89, nvcc 13.3; independently re-run)
- IMMA == reference (exact int32 accumulate; fragment layout audited vs the PTX ISA
  + CUTLASS); fused == host-A8 == caller quant; tail shapes on every kernel; JIT ==
  AOT bit-identical; tiled decode within 1e-4 of the sequential reference.
- **End-to-end greedy 256/256 exact, perplexity 2.81e-3** with the new kernels.
- compute-sanitizer memcheck/racecheck/synccheck **0 errors**; build + clippy **0
  warnings**; full cpu + `--features cuda` suites green.

### Notes / not yet closed
- **AVX-512 / NEON execution is lane-deferred** (the dev box is AVX2-only x86_64):
  AVX-512 compile-checked, NEON aarch64 cross-compile-checked, LUT + AVX2 + scalar
  parity gated here.
- The **`≥1.2×` bitnet.cpp end-to-end tok/s target is not yet hit**: the IMMA kernel
  is conformance-verified + microbenched but **not yet wired into the model forward**,
  which still has the v0.20 per-matmul host round-trips. The competitor baseline is
  **published** bitnet.cpp numbers (a same-HW build + a live `ncu` run are follow-on).
  v0.3.0 ships the verified fast *kernels* + harness; the *end-to-end* speedup is the
  next perf milestone.

## [0.20.0] — 2026-06-15 — Inference Spine

End-to-end token generation: **BitNet b1.58 2B4T** loads from its I2_S GGUF and
decodes tokens that match HF transformers, on CPU **and** CUDA (ADR 0004).

### Added
- **tritium-format** — I2_S decoder (`unpack_i2s_block`/`unpack_i2s_tensor`): ggml
  type-36, per-tensor f32 scale, `trit = code-1`, plain `[N,K]`; bit-exact vs the HF
  checkpoint on every layer-0 projection shape.
- **tritium-nn** — ops (RoPE NeoX, GQA attention, softmax, top-k/p sampling) vs torch
  goldens; W1.58**A8** int8 activation quant (Qb=127, round-half-to-even); paged KV
  cache (incremental==full); `TernaryLinear`/`Relu2Mlp`/`TransformerBlock` with the
  `attn_sub_norm`/`ffn_sub_norm` sub-LN; `ModelRunner::{load,forward,generate}` + a
  fidelity-ladder debug hook; tied LM head.
- **tritium-py** — PyO3 0.23 + maturin abi3 wheel: `Model.load/generate` (GIL released),
  `ternary_matmul`; every error → a Python exception.
- **tritium-cli** — `generate` subcommand.

### Validated
- **Forward fidelity** — vs transformers fp32: embedding bit-exact, per-op rungs ~1e-6,
  final-logit **argmax exact**.
- **Acceptance (RTX 4090)** — CUDA greedy **256/256 tokens exact**; **perplexity 2.81e-3**
  (≤1%); **CPU↔CUDA parity** bit-identical over 32 steps.
- **Python binding** — shape/dtype errors raise, GIL release proven, 6-thread no deadlock.

### Notes
- Tokenizer is Python-side (HF) for the acceptance harness; a native Rust tokenizer is
  deferred to v0.80. Big-model tests are gated (model download + GPU), not on cpu-CI.

## [0.10.0] — 2026-06-15 — Foundation

First milestone (ADR 0002 roadmap). A ternary mpGEMM runs bit-exact against the
reference on **CPU and CUDA**, end to end through the backend contract, registry,
and CLI. All v0.10 exit gates (U1–U9) closed.

### Added
- **tritium-core** — `Trit` (`{-1,0,+1}`, `repr(transparent)` i8), `DType`,
  `TernaryFormat`, `ScaleGranularity`/`absmean`, `GemmShape`, `reference_mpgemm`
  (the add/sub/skip ground truth), `TritError`. `no_std`-able, zero deps.
- **tritium-spec** — object-safe `TernaryBackend` trait (boxed `dyn DeviceBuffer`
  + `Any` downcast for runtime dispatch), `DeviceCaps`, `BackendError`.
- **tritium-format** — TQ1_0/TQ2_0 pack/unpack (faithful ggml port, golden +
  roundtrip tested), row-level wrappers (tail zero-pad), and a total, bounds-checked
  GGUF v2/v3 reader (`read_gguf`). cargo-fuzz target for the parser.
- **tritium-runtime** — `linkme` distributed-slice backend registry; a failing
  backend `init` is skipped, never fatal.
- **tritium-testkit** — `ConformanceVector` + `run_conformance<B: TernaryBackend>`
  graded against `reference_mpgemm`; JSONL persistence. Self-validated.
- **tritium-cpu** — AVX2 + scalar ternary mpGEMM, runtime-dispatched, rayon over
  rows. AVX2 reproduces the reference accumulation bit-for-bit. Conformance: zero
  failures.
- **tritium-cuda** — feature-gated CUDA backend (`--features cuda`): add-only
  `tq2_0_add.cu` kernel + `build.rs` nvcc→PTX + cudarc host side. Default build inert.
- **tritium-cli** — `tritium inspect <gguf>` and `tritium list-backends`.

### Security
- Bounded GGUF tensor/dimension preallocation against adversarial counts (a
  declared `n_dims` could otherwise drive a ~34 GB allocation and abort). Found by
  the commit-review policy; fixed with regression tests.

### Gates closed for `0.10.0`
- **GPU (RTX 4090, CUDA 13.3)** — CUDA kernel vs reference and **CPU↔CUDA parity
  (U2)** ✓ (cudarc 0.19, both backends ≤1e-4); `compute-sanitizer` memcheck **0
  errors** (U7) ✓.
- **Fuzz (U5)** — GGUF parser, **550,816,129 runs / 1h, 0 crashes**, RSS flat.
- **Real GGUF (0.10.5)** — reader pinned to the official `gguf` writer's output
  (TQ2_0/TQ1_0/F16/F32 tensors + metadata), fixture committed.
- `miri` is N/A (cannot execute AVX2 intrinsics); the unsafe AVX2 kernel is covered
  by audit + reviewer sign-off + bit-exact scalar parity + `compute-sanitizer`.

[0.10.0]: https://github.com/Quitetall/tritium/releases/tag/v0.10.0
