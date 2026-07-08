# Tritium ROADMAP — strategic milestones + tactical plan index

This is the **living index** that maps the permanent strategy to the executable work in flight.
It exists because Tritium is planned by a strong model and **executed by a weaker model**: the
plans must be detailed and verification-gated enough that a cheap executor can run them faithfully
and a reviewer can course-correct from the outputs alone.

## Three layers

| Layer | Where | What | Lifetime |
|-------|-------|------|----------|
| **Strategic** | `docs/adr/` | ADRs — the 0.1.0→1.0.0 arc, each milestone's scope + exit gates | permanent (append/amend) |
| **Index** | `docs/ROADMAP.md` (this file) | the ordered set of tactical plans from *now* → done, with status | living |
| **Tactical** | `docs/plans/NNNN-*.md` | one detailed, verbatim, gated executable plan per point-release / coherent feature (1–few commits) | per chunk; kept for the audit trail |

Everything the executor needs is under **`docs/`** — point the model there. **`docs/EXECUTOR.md`** is the
single entry point (protocol + which plans to run next + how to report back). `~/.claude/plans/` is
ephemeral session scratch; the durable plans live in `docs/plans/` (version-controlled, diffable).

## The per-turn loop

1. **Plan** — the strong model writes the next `docs/plans/NNNN-*.md` (see template below) and flips its
   row here to `in-progress`.
2. **Execute** — the weaker model runs the plan step-by-step, pasting the full output of each
   command and the review verdict.
3. **Review** — the strong model reads the diff + pasted outputs against each step's *expected
   output* block, then either:
   - **course-corrects** (amends the plan / writes a fix-up micro-plan), or
   - **accepts** (flips the row to `done`, writes the next plan).

The expected-output blocks (PASS signature + failure branches) are the spine: they let review
proceed from outputs alone, "no matter what happens."

---

## Strategic milestones (ADR 0002)

| Milestone | ADR | Status |
|-----------|-----|--------|
| v0.10 Foundation | 0003 | **Done** (tagged) |
| v0.20 Inference Spine | 0004 | **Done** (tagged) |
| v0.30 Performance (+ v0.3.1 device-resident, 0013) | 0005 / 0013 | **Done** |
| v0.40 SALT Quantization | 0006 | **Done** (tagged `v0.4.0`, pushed) |
| v0.4.1 perf/correctness point-release (split-KV + IMMA fix) | 0013 / — | **Done** (tagged `v0.4.1` @ `202bec1`, pushed; CUDA decode + IMMA gates re-verified on GPU) |
| v0.50 Training Core | 0007 | **Done** (tagged `v0.5.0`) — STE autograd + Gate C green CPU+CUDA (0005–0007), AdamW + bit-exact checkpoint (0008), LoRA on frozen base (0009), QAT heal bridge + distillation-convergence capstone on the real model (0010). Full-model ≥90% PPL-recovery deferred to v0.60 (needs full-model backprop + a non-QAT-latent checkpoint — see 0010). |
| v0.60 Pretraining + Distributed | 0008 | **Done** (tagged `v0.6.0`) — distributed stack 0011–0016 (shipped on the `0.5.x` line: full-model CPU backward, data pipeline, GPU pretrain smoke, ProcessGroup+sim backend, ZeRO-3/FSDP, distributed checkpoint) + real `cudarc::nccl` backend (0017) + the ≥2-GPU wall (0018) **validated on 2×A100 (production)**: 0017 wire-correctness + 0018 FSDP loss-parity (world=2, `max\|Δloss\|=4.5e-8`); full CUDA suite **51/51 on Ampere** + single-GPU memcheck-clean. **≥80% scaling deferred** (tiny gate models; needs the real-scale resident engine). |
| v0.70 Backend Breadth | 0009 | **✅ DONE — tagged `v0.7.0` (2026-06-24).** Full backend matrix (cpu/cuda/wgpu/wasm/metal/rocm) hardware-conformant. **Metal** parity bit-exact on a real Apple M1 (Scaleway M1-M, macOS 26.3.2, `v0.6.9`) + in-kernel TQ2_0 decode (memory-parity with cuda/rocm); **ROCm** parity validated on a real **AMD Instinct MI300X** (gfx942, ROCm 7.2.4, Hot Aisle) — frozen-vector conformance ran on the GPU (not skipped), 1e-4. wgpu/wasm validated `v0.6.1` (4090 Vulkan + Apple Metal HAL + wasmtime); the `build.rs` `libamdhip64` link-search fix (`5397781`) makes rocm link on a stock ROCm install. |
| v0.80 Interop | 0010 | **✅ DONE — tagged `v0.8.0` (2026-06-24).** All four framework frontends, each CI-green on CPU: **`tritium-serve`** (OpenAI HTTP/SSE contract+concurrency, `v0.6.2`; LAMU `backend_kind`); **`tritium-ffi`** (C ABI cdylib + staticlib, panic-safe, cbindgen drift + C11/C++17 gate, `v0.6.3` — unblocks the v1.0 C-ABI freeze); **`tritium-candle`** (candle `CustomOp1`, bit-exact, `v0.6.4`); **`tritium-burn`** (backend-generic op, bit-exact, `v0.6.5`); **`tritium-onnx`** (`ort` 2.x custom op == native, `v0.6.5`). abi3 py wheel builds+imports+pytest-green on macOS arm64. |
| v0.90 Hardening | 0011 | **✅ DONE — tagged `v0.9.0` (2026-06-24).** mdbook + dead-link `docs` lane; CPU `sanitizers` (ASan/MSan/TSan + miri — **ran green 2026-06-24**, miri `-Zmiri-disable-isolation` for proptest); `wheels`; `cpu-bench-smoke`; cargo-deny (`v0.5.8`), fuzz breadth + corpora (`v0.5.9`), doc-coverage + semver (`v0.5.10`), publish-check + SBOM (`v0.6.x`). **Threat model** documented (`docs/security/threat-model.md`, 30 threats). GPU-dependent gates (full CI matrix / perf-on-main / compute-sanitizer) met via the **ADR-0011 amendment**: fenced validation + dispatchable lanes + free Metal CI on GitHub `macos-14`. |
| v1.0 Release | 0012 | **✅ DONE — tagged `v1.0.0` (2026-06-28).** Tiered API/C-ABI freeze (frozen core 7 crates + C ABI under semver; nn/train/cuda/interop/serve documented as the evolving tier — `docs/v1.0-api-freeze-audit.md`). **Real-model GPU capstone PROVEN on a local RTX 4090:** real BitNet 2B4T runs correctly end-to-end — perplexity within 0.3% of transformers, greedy 256/256 token-exact, cpu↔cuda parity rel 2.3e-6, batch==single, qat_heal 94.6% layerwise convergence (decode-correctness fix `302d059`; teeth-proven conformance gate `51b041d`). Every prior gate (v0.10→v0.90) re-runs green on the release commit; GPU validation fenced (ADR-0011 amendment), CI lanes dispatchable. Metal (M1) / ROCm (MI300X) / wgpu (4090) parity from prior fenced sessions. |
| **v1.x Capstone — SALT-distillation of a SOTA model (BINDING public-launch gate)** | **0020** | **Proposed (ADR written)** — the tagged `v1.0.0` proved the *infra* on from-scratch BitNet; the **public launch (crates.io + announcement) is blocked** until Tritium ternarizes a **27–35B SOTA model to ≤1% ppl vs its fp16 teacher** via SALT-distillation (fp oracle → ternary student, SALT-as-STE, adaptive plane growth). Gate arch = a standard-transformer 32B (tractable); **Qwen3.6-27B** (Mamba/SSM hybrid) is the headline extension. Keystone = a **general inference engine** (config-driven arch registry + SALT multi-plane loader). Near-fp16 quality at ~1/5–1/8 the VRAM. |
| **Spec-decode (BASTION-style tree verify)** | **0014** | **Proposed (ADR written)** — *post-v0.4.1 point-release; Tritium = verifier, LAMU orchestrates, drafter external* |

## Tactical plan index (now → done)

Ordered. `todo` = not started, `in-progress` = executor running it, `done` = accepted + committed.

| # | Plan | Scope | Serves | Status |
|---|------|-------|--------|--------|
| # | Plan | Scope | Serves | Status | Parallel? |
|---|------|-------|--------|--------|-----------|
| 0001 | `docs/plans/0001-v0.4.1-split-kv-wiring.md` | Split-KV attention into the resident decode | v0.4.1 / ADR 0013 | **done** (`79f4939`+`899b162`) | — |
| 0002 | `docs/plans/0002-v0.4.0-doctests-example.md` | Runnable SALT example (U9) | v0.4.0 / U9 | **done** (`a71c48e`) | — |
| 0003 | `docs/plans/0003-v0.4.1-imma-oob-fix.md` | Fix the JIT IMMA tail-shape OOB read (U7); compute-sanitizer clean | v0.4.1 / U7 | **done** (`13438a4`; memcheck 0 errors) | — |
| 0004 | `docs/plans/0004-v0.4.1-release.md` | v0.4.1 CHANGELOG + version bump | v0.4.1 | **done** (`202bec1`, tagged `v0.4.1` + pushed) | sequential |
| ADR 0014 | `docs/adr/0014-spec-decode-bastion.md` | BASTION spec-decode design (Tritium = verifier) | new capability | **done** (proposed) | — |
| 0005 | `docs/plans/0005-v0.50-train-skeleton-ste.md` | tritium-train skeleton: STE + ternary-matmul backward, gradient-checked | v0.50 / ADR 0007 Gate C | **done** (`5030fa3`; CPU) | — |
| 0006 | `docs/plans/0006-v0.50-cpu-ops-tape.md` | CPU op set (bias/relu²/mse/xent/elementwise) + reverse-mode tape + composed QAT gradient | v0.50 / ADR 0007 Gate C | **done** (`2be9332`; Gate C green on CPU) | — |
| 0007 | `docs/plans/0007-v0.50-cuda-backward.md` | CUDA backward kernels (gA/gW/gs) gradient-checked vs CPU vjp + compute-sanitizer | v0.50 / ADR 0007 | **done** (single-GPU; parity 1e-4 + memcheck 0 errors) | — |
| 0008 | `docs/plans/0008-v0.50-optimizer-checkpoint.md` | AdamW + minimal Optimizer trait + bit-exact TOPT checkpoint (resume==uninterrupted, no-NaN ≥1k) | v0.50 / ADR 0007 | **done** (`05e45d2`; CPU) | — |
| 0009 | `docs/plans/0009-v0.50-lora-frozen-base.md` | LoRA on a frozen ternary base (dense/detach/scale_const primitives); frozen-base zero-grad + merge + rank edges proptested | v0.50 / ADR 0007 | **done** (CPU) | — |
| 0010 | `docs/plans/0010-v0.50-qat-heal-gate.md` | QAT heal bridge (replace_weights/invalidate_resident) + distillation-convergence capstone on the real BitNet-2b4t model | v0.50 / ADR 0007 | **done** (GPU; ~94.6% layerwise convergence; full-model PPL-recovery deferred to v0.60) | — |
| 0011 | `docs/plans/0011-v0.60-full-model-backward.md` | Full-model CPU backward: rmsnorm/softmax/rope/transpose tape vjps + tiny-transformer end-to-end gradient gate | v0.60 / ADR 0008 | **done** (`b6620cf`, tagged `v0.5.1`; CPU) | — |
| 0012 | `docs/plans/0012-v0.60-data-pipeline.md` | Data pipeline: deterministic resumable dup/loss-free sharded `DataSampler` + the `.tqbin`/`.tqidx` corpus formats (total never-panic parsers) | v0.60 / ADR 0008 | **done** (tagged `v0.5.2`; CPU) | — |
| 0013 | `docs/plans/0013-v0.60-gpu-pretrain-smoke.md` | GPU QAT training step (wires the `train_grad` forward+grad kernels) + LR schedule + from-scratch tiny-model pretrain smoke; device==CPU + finite-difference composition gradcheck | v0.60 / ADR 0008 | **done** (tagged `v0.5.3`; 1 GPU) — full-2B resident training engine deferred | — |
| 0014 | `docs/plans/0014-v0.60-process-group.md` | `ProcessGroup` trait + deterministic thread-simulated collective backend (all_reduce / reduce_scatter / all_gather / broadcast); all-reduced grads == single-process summed reference | v0.60 / ADR 0008 | **done** (tagged `v0.5.4`; CPU sim) — uniform publish-first/2-barrier protocol + op-tag desync guard, adversarial-review-hardened | — |
| 0015 | `docs/plans/0015-v0.60-fsdp-zero3.md` | ZeRO-3/FSDP over the sim PG: `FlatShardPlan` + all_gather/reduce_scatter sharded training; reduced-gradient == full-batch gradient (teeth) + replicated bit-exact (world∈{2,4}) + partition loss-curve tracking | v0.60 / ADR 0008 | **done** (tagged `v0.5.5`; CPU sim) — review-hardened: added gradient-level teeth after the loss-only gate was found blind to a wrong reduce op (AdamW scale-invariance) | — |
| 0016 | `docs/plans/0016-v0.60-distributed-checkpoint.md` | Distributed checkpoint (`dcp`): per-rank shard files + manifest, crash-atomic temp→fsync→rename, save-K/reshard-J identical-forward + bit-exact resume + fault injection | v0.60 / ADR 0008 | **done** (tagged `v0.5.6`; CPU sim) — review-hardened: never-panic load path (`try_new` + `n_planes` bound), monotonic-step contract, real disk-reshard + uncommitted-shard gates | — |
| 0017–0018 WALL | `docs/plans/0017-v0.60-nccl-wall.md` | real NCCL backend (`cudarc::nccl` behind `ProcessGroup`) + HW loss-parity / ≥80% scaling bench → tag `v0.60.0` | v0.60 / ADR 0008 | todo (rented ≥2×GPU; backend + gates + `scripts/gpu_session.sh` built/reviewed/world=1-verified) | — |
| 0019 | `docs/plans/0019-v070-freeze-conformance-set.md` | Freeze + version the conformance vector set: committed `vectors/v070.jsonl` + `frozen_vectors()` + drift gate; CPU gate repointed | v0.70 / ADR 0009 | **done** (tagged `v0.5.7`; CPU) — first **build-ahead** item (see note) | — |
| 0021 | `docs/plans/0021-v090-cargo-deny-gate.md` | Supply-chain gate: `cargo deny check` green (Unicode-3.0 allow; internal deps version-pinned) + cargo-deny CI lane | v0.90 / ADR 0011 | **done** (tagged `v0.5.8`; CPU) — build-ahead; also unblocks v1.0 `cargo publish` | — |
| 0022 | `docs/plans/0022-v090-fuzz-breadth.md` | Fuzz breadth: cargo-fuzz targets for every untrusted-byte parser (tqbin/tqidx/salt_bundle/safetensors/legacy) + 8-target CI sweep | v0.90 / ADR 0011 | **done** (tagged `v0.5.9`; CPU) — build-ahead (U5) | — |
| 0023 | `docs/plans/0023-v090-hardening-gates.md` | v0.90 hardening gates: `#![deny(missing_docs)]` on 6 foundation crates + `cargo-semver-checks` API-stability gate (`scripts/check-semver.sh` + CI lane) | v0.90/v1.0 / ADR 0011 | **done** (tagged `v0.5.10`; CPU) — build-ahead | — |
| 0024 | _(v0.6.1; workflow-designed)_ | Capability-fallback contract: `run_fused_fallback_contract` pins the no-panic fused-path (`mpgemm_with_act_quant`) degrade for no-fp8 backends, scale-aware tolerance floor; CPU + wgpu + wasm subjects | v0.70 / ADR 0009 | **done** (tagged `v0.6.1`; CPU) — build-ahead | — |
| 0025 | _(v0.6.1; workflow-designed)_ | `tritium-wgpu`: WGSL ternary mpGEMM over wgpu (Vulkan); 89-vector conformance + fused-fallback **on the 4090**; add/sub/skip shader form; 2-D dispatch beyond the 65535/dim cap; error-scoped (no-panic); adapter-select + real limits | v0.70 / ADR 0009 | **done** (tagged `v0.6.1`; 4090 Vulkan) — build-ahead | — |
| 0026 | _(v0.6.1; workflow-designed)_ | `tritium-wasm`: scalar `TernaryBackend` on `wasm32-wasip1` (`reference_mpgemm`, spec/core/format only — no rayon/linkme); conformance **inside wasmtime** (Cranelift) | v0.70 / ADR 0009 | **done** (tagged `v0.6.1`; wasm/wasmtime) — build-ahead | — |
| 0027 | _(v0.6.2; workflow-designed)_ | `tritium-serve`: OpenAI HTTP/SSE server (axum, feature-gated); `Generator` seam (RunnerGenerator + MockGenerator); one decode thread + bounded queue (concurrency, backpressure, graceful shutdown); 11 model-free contract tests = the ADR-0010 serve gate. Also the LAMU `backend_kind`. + pyo3 0.23→0.25.1 (RUSTSEC-2025-0020) + deny paste-ignore | v0.80 / ADR 0010 | **done** (tagged `v0.6.2`; CPU) — build-ahead | — |
| 0028 | _(v0.6.3; workflow-reviewed)_ | `tritium-ffi`: C ABI `cdylib`+`staticlib`; panic-safe (`catch_unwind`) + null-checked `unsafe extern "C"`; cbindgen `include/tritium.h` with a drift gate + C11/C++17 compile check; single-pass / size-then-fill `tritium_generate`; `*out_len` always defined; linkme cpu-registration survival verified in the linked artifact; 10 ABI + 2 header tests. Unblocks the **v1.0 C-ABI freeze** | v0.80 / ADR 0010 | **done** (tagged `v0.6.3`; CPU) — build-ahead | — |
| 0029 | _(v0.6.4; workflow-reviewed)_ | `tritium-candle`: Tritium ternary mpGEMM as a candle `CustomOp1` (`apply_op1_no_bwd`) over `[M,K]` f32 acts × `[N,K]` packed ternary weights × `[N]` scales → `[M,N]`; `reference_mpgemm` kernel (bit-exact); dtype/contiguity/K/packed-len validated, never panics. Feature-gated (`candle`, lean default); conformance test reproduces the frozen set bit-exactly at the candle-Tensor level; `candle-conformance` CI lane | v0.80 / ADR 0010 | **done** (tagged `v0.6.4`; CPU) — build-ahead | — |
| 0030 | _(v0.6.5; workflow-impl + workflow-review)_ | `tritium-burn`: backend-generic `ternary_mpgemm<B: Backend>` host round-trip (read → `reference_mpgemm` → rebuild, pinned `DType::F32`); bit-exact on burn's NdArray; lazy-backend read failure returns `BurnTernaryError` not a panic (`try_into_data`); f32-only (documented). Feature-gated (`burn`, lean default; burn-tensor/ndarray 0.21); `burn-conformance` CI lane. deny: MPL-2.0 allow (`colored`) | v0.80 / ADR 0010 | **done** (tagged `v0.6.5`; CPU) — build-ahead | — |
| 0031 | _(v0.6.5; workflow-impl + workflow-review)_ | `tritium-onnx`: Layer 1 always-on `ternary_mpgemm_kernel` (zero-dep, bit-exact default gate) + Layer 2 `ort` 2.x custom operator (node `TritiumTernaryMpGemm`, feature `onnx`); `ort = =2.0.0-rc.12` default-features-off + `download-binaries`+`tls-rustls` (no system lib); `run` kernel + registration tested, native session e2e `#[ignore]`d; `onnx-op` CI lane. deny: CDLA-Permissive-2.0 allow (`webpki-roots`) | v0.80 / ADR 0010 | **done** (tagged `v0.6.5`; CPU) — build-ahead | — |
| 0032 | _(v0.6.6; workflow-impl + workflow-review)_ | v0.90 hardening polish: mdbook guide (9 chapters) + `docs` dead-link lane; CPU `sanitizers` lane (ASan/MSan/TSan via `-Zbuild-std` + miri; no `-Zsanitizer=undefined` so MSan/TSan/miri stand in for UBSan); `wheels` lane (abi3 maturin, artifacts only); `cpu-bench-smoke` lane (divan mpgemm). Review fixed 3 doc-accuracy issues (distributed shipped; SALT steps 2/5/6 planned; CPU AVX-512→AVX2→NEON→scalar) | v0.90 / ADR 0011 | **done** (tagged `v0.6.6`; CPU) — reachable polish | Metal/ROCm GPU lanes deferred |
| 0033 | _(v0.6.7; workflow-impl + workflow-review, zero findings)_ | v1.0 capstone prep: `docs/v1.0-api-freeze-audit.md` (report-only freeze-readiness audit; C ABI frozen at v1; semver gate green vs `v0.5.10`; pre-freeze [breaking]/[additive] list); mdbook model-zoo + benchmarks chapters (real I2_S load path, methodology, zero fabricated numbers); `scripts/capstone.sh` + `capstone` CI lane (CPU install→infer→SALT→fine-tune e2e on real code paths, exit 0) | v1.0 / ADR 0012 | **done** (tagged `v0.6.7`; CPU) — reachable prep | v1.0.0 tag gated on Metal/ROCm + GPU matrix + real-model GPU capstone |
| 0034 | _(v0.6.8; workflow-impl + workflow-review, zero findings)_ | `tritium-metal` (MSL kernel, metal-rs, macOS-target-gated) + `tritium-rocm` (HIP kernel, raw HIP-runtime FFI, `rocm`-feature-gated, hipcc build.rs) — full backend matrix code-complete; written BLIND (no Metal/ROCm toolchain on the dev box) but Linux-green-verified (inert off-platform) + port-fidelity-reviewed vs wgpu/cuda. + `THIRD-PARTY-LICENSES.md` (cargo-about, 369 crates) for v1.0 dep-license tracking. Self-hosted `metal`/`rocm` CI lanes | v0.70 / ADR 0009 | **done** (tagged `v0.6.8`; code) — full scope | Metal/ROCm first-compile + parity on real Apple-Silicon / AMD hardware |
| 0035 | `docs/plans/0035-general-inference-engine.md` | **KEYSTONE.** Config-driven general fp inference: `ArchSpec` + `ModelConfig::from_hf_config`, SwiGLU MLP (`Mlp` dispatch), untied LM head, HF-safetensors loader (`load_hf`/`from_hf`) on the standard llama/qwen name schema. Gate: **SmolLM2-135M** loads from config.json+safetensors, greedy-token-exact vs `transformers` (last-row logit rel < 1e-3), no BitNet regression. Unblocks teacher-caching + ternary student forward | **v1.x capstone / ADR 0020 step 1** | **✅ DONE** — gate GREEN: SmolLM2-135M **16/16 token-exact, last-row logit rel-err 2.0e-6** (`ArchSpec`+`from_hf_config`, SwiGLU/`Mlp`, exact-fp `DenseLinear::new_exact`, untied head, `load_hf`/`from_hf`). Reviewed; safety guards reject Llama-3.x `rope_scaling` + Qwen QKV-bias/QK-norm (loud, not silent-wrong). Follow-ons 0036 (SALT loader), 0037 (QK-norm/QKV-bias) | — |
| — | `docs/plans/capstone-cascade.md` | **Milestone map** for the whole ADR 0020 cascade (0035→0041): deliverables, numeric gates, deps, effort, de-risking model ladder (SmolLM2→Qwen-small→32B→Qwen3.6), teacher-caching, compute plan, quality-vs-bpw methodology. Read this to see the full path from keystone → binding gate | v1.x capstone / ADR 0020 | **map** (living) | — |
| 0036 | _(cascade)_ | SALT multi-plane inference loader — run what `quantize` emits (T planes; existing kernel looped + per-plane scale accumulate). Gate: SALT-quantized SmolLM2 runs; multi-plane==Σdequant bit-parity; T=1 byte-identical | ADR 0020 step 2 | **planned** (dep 0035) | with 0037 |
| 0037 | _(cascade)_ | Arch extensions: optional QKV-bias (Qwen2/2.5) + QK-norm (Qwen3), gated by `ArchSpec` flags. Gate: small Qwen2.5 + Qwen3 token-exact vs transformers | ADR 0020 (arch) | **planned** (dep 0035) | with 0036 |
| 0038 | _(cascade)_ | **SALT-aware distillation trainer** — latent-master QAT, SALT-as-STE, teacher-logit cache, KL+hidden loss (layerwise→e2e), AdamW CPU-offload. Gate: distill SmolLM2 to ≤1% ppl small-scale (proof-of-loop) | ADR 0020 step 3 | **planned** (dep 0035,0036) | — |
| 0039 | _(cascade)_ | Real grad-Fisher sensitivity → `Sensitivity::Custom` + adaptive plane growth. Gate: Custom beats Uniform/Energy at fixed bpw; adaptive growth hits ≤1% at lower avg bpw | ADR 0020 step 4 | **planned** (dep 0038) | — |
| 0040 | _(cascade)_ | Scale plumbing to 32B — streaming quantize writer, streaming GPU load, grad checkpointing, optimizer offload at scale. Gate: 32B fp loads+runs; one distill step fits memory | ADR 0020 step 5 | **planned** (dep 0037,0038) | — |
| 0041 | _(cascade)_ | **CAPSTONE (binding gate)** — distill a standard 32B to ≤1% ppl vs fp16; report quality-vs-bpw curve + measured VRAM reduction; student runs e2e in Tritium; then Qwen3.6-27B headline. Passing unblocks the public launch (#45) | ADR 0020 step 6 / DoD | **planned** (dep all) — compute-bound | — |

> **0002 (A) and 0003 (B) are independent** — disjoint files (format/quantize+examples vs the CUDA
> JIT codegen) → safe to run concurrently. For true parallelism use a **git worktree per plan**
> (`git worktree add ../tritium-0003 main`) so two executors don't fight one index; otherwise run
> them sequentially (either order). 0004 runs only after both land + are green.

> **0001 outcome (planner review):** split-KV is correct (graph==eager bit-exact; greedy/parity
> gates hold) and a real kernel win — nsys shows attention **57.6% → 26.6%** of N=1 GPU time
> (~2.2× faster), neutral-to-better throughput across N (no high-N regression on clean re-bench).
> But end-to-end **N=1 throughput stayed ~flat (108→111)** because N=1 is occupancy-bound across
> the *whole* pipeline (GEMM 35% / rmsnorm 21% / eager lm_head 8% now dominate). The plan's
> `>120 N=1` criterion was an over-optimistic Amdahl estimate — **corrected**: the success metric
> for an attention fix is the *attention* share, not N=1 wall-clock. Further N=1 gains are
> diminishing-returns (whole-pipeline occupancy); the regimes that matter (N≥2, long-ctx, the
> argmax path) already perform well. **Lesson for future plans:** gate perf work on the *profiled
> bottleneck's* metric, not a derived end-to-end target.

> Already shipped on `main` above `v0.4.0` (this session, pre-system): rustfmt-1.9.0 reformat
> (CI fmt gate), U5 fuzz targets + `fuzz/target` untrack, split-KV attention **kernels +
> equivalence gate** (`b5173e9`) + head_dim guard (`9e70354`). Plan 0001 is the *wiring* of those
> kernels into production.

---

## Forward decomposition (v0.50 → v1.0)

The strategic ADRs **0007–0012** already define each milestone's exit gates + Definition of Done.
Tactical plans are written **just-in-time** (verbatim against the code as it exists when the
milestone starts) — writing them all now would be fiction against crates that don't exist yet. So
this section is the **map**: per milestone, the ordered *entry* tactical plans an executor should
write/run first, and the hard blocker that gates the milestone. A planner turns each `→` arrow into
one `docs/plans/NNNN-*.md` when its turn comes.

- **v0.50 Training Core — ADR 0007** *(blocker: GPU + a real fp16 source model + a known-recoverable
  fine-tune task)*
  `tritium-train` skeleton + STE autograd for the ternary matmul (finite-difference gradient-check,
  TDD) → backward kernels for the remaining ternary ops (CPU first, then CUDA; each gradient-checked)
  → optimizer + bit-exact save/restore (resume==uninterrupted golden) → LoRA on a frozen ternary
  base (zero-grad-on-base proptest; `r=1` and `r=full` edges) → QAT heal loop + the GPU
  ≥90%-gap-recovery convergence gate. *(Optional external: a BLUT training cookbook over
  `tritium-train` — AGPL boundary, cookbook→Tritium only.)*
- **v0.60 Pretraining + Distributed — ADR 0008** *(blocker: ≥2-GPU cluster; multi-node interconnect
  for the multi-node path)*
  Data pipeline (deterministic sharded shuffle, resumable mid-epoch — CPU-testable) → FSDP/DDP
  gradient/param sharding (N-GPU loss-parity vs 1-GPU) → distributed checkpoint + resharding J≠K →
  multi-node orchestration + rank-kill fault injection → scaling bench (≥80% efficiency) +
  from-scratch pretrain smoke.
- **v0.70 Backend Breadth — ADR 0009** *(blocker: per-platform hardware — Metal box, ROCm GPU,
  WebGPU/WASM target)*
  **Freeze + version the conformance vector set first** (the one CPU/CUDA pass) → then one backend
  crate per plan: `tritium-metal` → `tritium-rocm` → `tritium-wgpu`/`tritium-wasm`, each passing the
  full conformance set + cross-backend parity + acceptance-model greedy match.
- **v0.80 Interop — ADR 0010** *(blocker: acceptance model for native-parity gates; ONNX Runtime in
  CI)*
  `tritium-serve` (OpenAI-compatible HTTP — **this is also the LAMU `backend_kind` interface**;
  contract + streaming + concurrency gates) → `tritium-ffi` (C ABI + generated header; C/C++ compile
  + round-trip + null-arg fuzz + sanitizer) → `tritium-candle` / `tritium-burn` ops → `tritium-onnx`
  custom op then ORT EP.
- **v0.90 Hardening — ADR 0011** *(blocker: full per-platform CI matrix + GPU/multi-GPU lanes +
  model download)*
  **Full doctest sweep** (the ~27 v0.4.0 public items still lacking `/// # Examples`, the rest of U9
  — *startable now, good standalone agent task*) → 24h-cumulative fuzz breadth across every parser →
  full CI build/test matrix → packaging (wheels for manylinux/macOS/Windows + `cargo publish
  --dry-run`) → mdbook + dead-link check → perf-regression gate enforced on `main` → security review
  + threat model + `cargo-deny`/SBOM.
- **v1.0 Release — ADR 0012** *(blocker: real model end-to-end in a fresh env; GPU)*
  `cargo-semver-checks` baseline + public API/C-ABI freeze → re-run every prior gate on the release
  commit → third-party-reproducible quickstart + model zoo + benchmark report → capstone fresh-env
  e2e (install → infer → SALT-quantize → fine-tune).
- **Spec-decode (BASTION) — ADR 0014** *(post-v0.4.1 point-release; blocker for the speedup gate: an
  external block-diffusion drafter; correctness gates need only a mock drafter)*
  Tree-masked verify attention (sibling of split-KV) → shared-prefix KV + provisional
  commit/rollback → accept logic + losslessness gates (greedy + sampling, mock drafter) → memcheck →
  *(with a real external drafter)* end-to-end speedup bench at the roofline knee.

**Convention reminder:** milestone work is gate-blocked, not date-blocked — no work on milestone N+1
merges until N's gate is green and tagged (ADR 0002). A milestone whose hard blocker (GPU count,
per-platform HW, model download) is unavailable runs its load-bearing gate as a **documented manual
gate** on borrowed hardware before tagging.

**Build-ahead strategy (from `v0.5.7`):** ADR 0002 sequences the milestone *tags*, not the *build*.
With `v0.60.0` dammed behind the rented ≥2-GPU session, the software-reachable slices of v0.70/v0.80/
v0.90/v1.0 are landed *now* on the continuing `0.5.x` point-release line — each fully gated + reviewed,
none claiming a milestone tag. A code-grounded survey put this at **~21 reachable-now items vs ~9 true
HW-wall items** (the wall: Apple-Metal + AMD-ROCm parity, the multi-GPU NCCL/FSDP re-runs, and the
macOS/Windows CI matrix). The payoff: once the rented session tags `v0.60.0`, the downstream milestone
tags fall as a short **verify-and-tag cascade** instead of a per-milestone build-then-tag crawl.
Reachable-now order (by value/effort + dependency): **0019 freeze (done `v0.5.7`)** →
**capability-fallback contract (done `v0.6.1`)** → **`tritium-wgpu` (Vulkan on the 4090, done `v0.6.1`)
/ `tritium-wasm` (wasmtime, done `v0.6.1`)** → **`tritium-serve` (OpenAI HTTP/SSE, done `v0.6.2`)**
→ **`tritium-ffi` (C ABI, done `v0.6.3`)** → **`tritium-candle` (candle CustomOp, done `v0.6.4`)** → **`tritium-burn` + `tritium-onnx` (done `v0.6.5`; v0.80 interop COMPLETE)** →
doctest sweep / fuzz breadth (done) / `cargo-deny` (done) / semver baseline (done) (v0.90/v1.0
tooling). Remaining v0.70 = the **Metal + ROCm** platform backends (fenced HW) +
the macOS/Windows CI matrix → then the v0.70 milestone tag.

---

## Tactical plan template (`plans/NNNN-<slug>.md`)

Every tactical plan MUST follow this shape so the executor + reviewer have one contract:

```markdown
# NNNN — <title>  (serves: <ADR / milestone>)

## Goal
One paragraph: what this plan delivers + the one-line success criterion.

## Preconditions
- Branch `main` at commit <hash> (`git log --oneline -1` must show it).
- What is already done (so the executor doesn't redo it).
- `git status` must be clean before starting.

## Steps
Each step is atomic and ends with an expected-output block.

### Step N — <what>
- **Files:** exact paths.
- **Edit:** exact `old_string` → `new_string` (verbatim for CUDA/unsafe/bit-exact/parser code),
  or the full code block to add. Routine glue may be precise prose + signatures.
- **Command:** the exact shell command to run.
- **Expected output (PASS):** the exact line(s) / numeric range to see.
- **If you see instead … :** 2–3 likely divergences, each with a diagnosis + the corrective action.
- **Paste:** "paste the full output of `<command>`" (so review sees ground truth).

## Gate
The test(s)/bench that must pass + expected numbers (with tolerance). Exact command + expected output.

## Commit
Verbatim commit message (the executor commits exactly this; ends with the Co-Authored-By line).

## Review
"Review the commit with the `feature-dev:code-reviewer` subagent (project policy). Paste the
verdict + every finding verbatim." (The strong model triages findings next turn.)

## Done criterion
The exact state that means this plan is complete (tests green, bench number, clean tree).
```

### Rules for the executor (weaker model)
- **Transcribe, don't derive.** Apply edits exactly as written. Do not "improve", rename, or
  refactor beyond the plan.
- **Never skip an expected-output check.** If output diverges and no failure branch matches, **stop
  and paste the output** — do not guess a fix.
- **One commit per plan** unless the plan says otherwise; commit message verbatim.
- **Always run the review step** and paste the verdict.
