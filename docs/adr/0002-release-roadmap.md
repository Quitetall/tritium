# ADR 0002 — Release roadmap to v1.0 (depth-first, breadth-second)

- **Status:** Proposed
- **Date:** 2026-06-14
- **Deciders:** Brian Lam
- **Supersedes / relates:** builds on [ADR 0001 — SALT](./0001-salt-quantization.md)

## Context

Tritium v1.0 is the **full vision, real and usable**: ternary inference and
training, all backends (CPU, CUDA, Metal, ROCm, WebGPU/WASM), SALT quantization,
distributed training, and the frontend/interop surface (Python, ONNX, C-ABI,
candle/burn, serve). That is far too large for a single implementation push.

We reach it via a staircase of **shippable `0.x0` milestones**. Each milestone is
a tagged release that is itself real and usable on the backends it targets. We
take **Approach A — capability-vertical, depth-first**: add one full capability at
a time across the spine on **CPU + CUDA first**, then widen to all backends
(0.70), then interop (0.80), then harden (0.90), then freeze (1.0). Backend
breadth is deliberately late so the `tritium-spec` contract stabilizes on two
backends before others copy it.

The point of this ADR is not only the sequence — it is the **validation regime**.
Each milestone defines exit gates engineered so that *if the gates pass, the
milestone genuinely works* — across happy path, boundaries, failures, untrusted
input, determinism, performance, memory, and concurrency. Gates are blocking: no
work on milestone N+1 merges until milestone N's gate is green and tagged.

**Assumption (correct me):** resourcing is solo / small team, milestone-gated not
date-gated — no calendar deadlines are encoded here, only ordering and exit
criteria. Revise if a hard date or team size changes milestone sizing.

## Decision

Adopt Approach A with the partition and gates below.

**Convention (required):** every milestone has its own **status + testability ADR**
before it is tagged. Each records the milestone's `Status` (planned / in-progress /
done, with dependencies + blockers), its `Testability` (the exit gates from this
roadmap, tagged with the taxonomy below and mapped to a test technique + CI lane),
and a `Definition of done` checklist. Index:

| Milestone | ADR | Status |
|---|---|---|
| v0.10 Foundation | [0003](./0003-v010-implementation.md) | **Done** (tagged `v0.10.0`) |
| v0.20 Inference Spine | [0004](./0004-v020-inference-spine.md) | **Done** (tagged `v0.20.0`) |
| v0.30 Performance | [0005](./0005-v030-performance.md) | Planned |
| v0.40 SALT Quantization | [0006](./0006-v040-quantization.md) | Planned |
| v0.50 Training Core | [0007](./0007-v050-training-core.md) | Planned |
| v0.60 Pretraining + Distributed | [0008](./0008-v060-pretraining-distributed.md) | Planned |
| v0.70 Backend Breadth | [0009](./0009-v070-backend-breadth.md) | Planned |
| v0.80 Interop | [0010](./0010-v080-interop.md) | Planned |
| v0.90 Hardening | [0011](./0011-v090-hardening.md) | Planned |
| v1.0 Release | [0012](./0012-v100-release.md) | Planned |

---

## Validation taxonomy (applies to every milestone)

Every milestone's validation conditions are tagged with these categories. A
milestone is **not done** until it satisfies every category that applies to it.

| Code | Category | What it proves |
|------|----------|----------------|
| **C** | Correctness vs reference | Output matches `reference_mpgemm` / a defined fp64 or framework reference within the stated tolerance. |
| **P** | Cross-backend / cross-ISA parity | Every op present on >1 backend or ISA agrees on identical input. |
| **E** | Edge & boundary | Degenerate and extreme inputs behave as specified. |
| **F** | Failure & invalid input | Every bad input returns a typed error — never panics in a library path, never UB. |
| **S** | Untrusted-input safety | Parsers survive malformed/adversarial input (fuzz + sanitizers). |
| **D** | Determinism & reproducibility | Same input+seed ⇒ identical output; a documented deterministic mode exists where kernels otherwise reorder. |
| **Pe** | Performance | Meets a throughput/latency threshold; no regression past a set bound. |
| **M** | Memory & resource | No leaks; bounded peak; OOM returns an error, never aborts; streams/handles freed. |
| **Co** | Concurrency | Multi-stream / multi-thread / multi-request use is correct and race-free. |
| **Do** | Docs & API | Public items documented; a runnable example; documented errors are actually returned. |

### Tolerance conventions (fixed once, referenced everywhere)

- **Integer, packing, bit-twiddling paths:** bit-exact (`==`). No tolerance.
- **fp32-accumulate matmul:** relative error ≤ `1e-4` vs an fp64 reference.
- **fp16 / bf16 / fp8 paths:** ≤ `2e-3` relative, defined per op.
- **Cross-backend float:** within the same per-op tolerance (accumulation order differs by design).
- **Greedy decode:** exact token-ID match vs reference for ≥256 tokens.
- **Sampling decode:** output distribution matches reference within a χ²/KL bound at fixed seed.
- **Perplexity parity:** within `1%` (or `0.1` ppl, whichever larger) of the reference implementation on a fixed eval set.

### Universal exit gates (U1–U9 — required of every milestone)

- **U1 (C):** Randomized correctness suite (proptest, ≥10k cases) passes for all new kernels/ops vs reference.
- **U2 (P):** All multi-backend/ISA ops agree within tolerance on the conformance vector set.
- **U3 (E):** Boundary suite passes: empty tensors; `K` ∈ {1, block−1, block, block+1, 4096}; `M`=1 (decode) and `M`≥4096 (prefill); `N`=1; all-zero weights; all-±1 weights; saturated/zero scales; NaN/Inf activations have defined propagation.
- **U4 (F):** Every public fn returns a typed error (no panic, no UB) for shape mismatch, out-of-range trit, empty/misaligned buffers, NaN scales. Asserted by tests.
- **U5 (S):** Every parser introduced has a `cargo-fuzz` target; ≥1h fuzz with zero crashes/UB/un-erroring OOM. `miri` clean on all `unsafe`.
- **U6 (D):** Same input+seed reproduces bit-exact across runs; deterministic mode documented where applicable.
- **U7 (M):** ASan/UBSan + CUDA `compute-sanitizer` clean; no leaks; OOM path returns error.
- **U8:** `clippy` zero warnings, `rustfmt` clean, every `unsafe` block has a `// SAFETY:` note, CI green on all in-scope targets.
- **U9 (Do):** Every public item documented; milestone ships a runnable example; doctests pass.

---

## Milestones

Each milestone lists: **Scope**, **Testing blockers** (infrastructure that must
exist for the gate to be meaningful), and **Validation conditions** (the gate
itself, tagged). All milestones also inherit U1–U9.

### 0.10 — Foundation

**Scope:** `tritium-format` (TQ1_0/TQ2_0 pack/unpack + GGUF read), `tritium-spec`,
`tritium-testkit`, `tritium-cpu` (AVX2 LUT mpGEMM), `tritium-cuda` (one add-only
kernel), `tritium-runtime` (registry + dispatch), `tritium-cli inspect`.

**Testing blockers:**
- Conformance vector set generated from `reference_mpgemm`, committed, versioned.
- `cargo-fuzz` target for the GGUF reader + a seed corpus of real `.gguf` files.
- Both CPU and CUDA backends registered in the runtime registry and selectable.

**Validation conditions:**
- **C/E** Pack roundtrip: `unpack(pack(W)) == W` (bit-exact) for random ternary `W` across all formats and `K` ∈ {1, block−1, block, block+1, 4096}; padding behavior for non-block-multiple `K` is defined and tested.
- **C** Golden layout: TQ2_0 packed bytes equal a hand-computed golden vector (catches endianness/bit-order bugs); TQ1_0 base-3 (5-trit/byte) golden vector matches.
- **C/F/S** GGUF read: a real llama.cpp-produced TQ2_0 file loads with tensor shapes + scales matching llama.cpp's own dump; truncated/corrupt/wrong-magic files return a typed error (no panic); fuzz corpus finds zero crashes.
- **C** mpGEMM vs reference: CPU and CUDA each ≤ `1e-4` rel err over the randomized + boundary suites.
- **P** CPU vs CUDA parity within tolerance on the conformance set.
- **E** Degenerate: all-zero weights ⇒ zero output; `M`=1 and `M`=4096; `N`=1; NaN-in ⇒ NaN-out with no UB.
- **Do** `cli inspect` output matches a golden dump for a known model.

**Exit gate:** all above + U1–U9 on CPU+CUDA. Tag `v0.10`.

### 0.20 — Inference spine

**Scope:** `tritium-nn` (RMSNorm, RoPE, attention, KV-cache, sampling), full GGUF
model load, `tritium-py` binding, end-to-end token generation.

**Testing blockers:**
- A reference oracle wired in CI: the chosen acceptance model (**BitNet b1.58 2B4T**) running under a reference impl (bitnet.cpp/HF) to compare against.
- A fixed prompt set + eval set committed for token-match and perplexity tests.

**Validation conditions:**
- **C/E** Each nn op vs a numpy/PyTorch reference within tolerance, with edge cases: seq-len 1, max context, fully-masked positions, GQA head grouping, RoPE at position 0 and near max.
- **C** KV-cache: incremental decode output equals full-recompute output (the canonical cache bug) across cache-page boundaries and eviction.
- **C/D** End-to-end greedy decode of BitNet b1.58 2B4T produces token IDs **exactly matching** the reference impl for the fixed prompts, ≥256 tokens, CPU+CUDA.
- **C** Perplexity on the fixed eval set within `1%` of the reference impl (proves the full forward pass, not just argmax).
- **C** Sampling decode distribution matches reference within the χ²/KL bound at fixed seed.
- **F/Co** Python binding: wrong dtype/shape/non-contiguous input raises a Python exception (no segfault); DLPack tensor roundtrips; GIL released during compute (no deadlock with ≥4 host threads).
- **E** batch>1 and large-`M` prefill correct alongside `M`=1 decode.

**Exit gate:** exact greedy token match + perplexity parity on CPU+CUDA + binding safety. Tag `v0.20`.

### 0.30 — Performance

**Scope:** add-only **and** IMMA int8 CUDA paths; nvrtc JIT + on-disk autotune
cache; AVX-512 + NEON CPU paths; `benches/` harness.

**Testing blockers:**
- Baseline perf numbers recorded (llama.cpp + bitnet.cpp on the same hardware/model) and committed as the comparison point.
- Perf-regression CI job with a failing threshold.

**Validation conditions:**
- **C/P** IMMA path result == add-only path result within tolerance (two kernels, one truth); both == reference.
- **C/D** Autotuning never changes numerics beyond tolerance; a tuned config from cache reproduces the same output; cold-cache vs warm-cache identical.
- **P** AVX2 vs AVX-512 vs NEON vs scalar all agree (cross-ISA parity).
- **C/E** Tail shapes (non-tile-multiple `M`/`N`/`K`) correct under every kernel variant — the classic perf-kernel edge bug.
- **Pe** tokens/sec ≥ parity with bitnet.cpp on the same hardware/model (floor `1.0×`, target `≥1.2×`) at **unchanged perplexity** (no accuracy traded for speed); regression job fails on a `>5%` tokens/sec drop vs the recorded baseline.

**Exit gate:** speed thresholds met + zero accuracy regression + all-ISA/all-kernel parity. Tag `v0.30`.

### 0.40 — SALT quantization

**Scope:** `tritium-quantize` ([ADR 0001](./0001-salt-quantization.md)): residual
planes, mode codebook, sensitivity allocation, sparse residual; TQ2_0 residual
sidecar format; `cli quantize`.

**Testing blockers:**
- A fp16 source model + an accuracy harness (perplexity / downstream task) wired in CI.

**Validation conditions:**
- **C** Multi-plane accumulate kernel `Σ_p s_p·tmatmul` matches a SALT dequant→fp32 reference matmul within tolerance.
- **C/E** Residual reconstruction error decreases monotonically with plane count `T`; `T=1` reduces **exactly** to flat AbsMean (BitNet regression check).
- **C** Allocator respects the bpw budget exactly (`Σ |g|·1.585·T_g ≤ budget`); higher-sensitivity groups receive ≥ planes than lower (ordering invariant).
- **C/P** Sparse residual plane and dense residual plane produce identical matmul output; the density-threshold switch is correct on both sides.
- **C/E** Format sidecar roundtrips multi-plane weights; reads legacy plain-TQ2 (no residual) for backward-compat; version field enforced; edge budgets (1.58 = all base; very high = many planes), zero-variance group, outlier-heavy group all handled.
- **D** Same model+seed+budget ⇒ byte-identical packed output.
- **Pe/C** Accuracy-vs-bpw curve reported on the real model; at target bpw, within the stated gap of fp16.

**Exit gate:** kernel matches dequant reference + sparse==dense + accuracy curve meets target. Tag `v0.40`.

### 0.50 — Training core

**Scope:** `tritium-train`: STE autograd, QAT, backward kernels, optimizer,
LoRA on a ternary base. Single-node.

**Testing blockers:**
- A small fine-tune task with a known recoverable accuracy gap, in CI.

**Validation conditions:**
- **C** Gradient check: STE backward vs finite-difference numerical gradient within tolerance for **every** trainable op.
- **C** Autograd graph reproduces analytic gradients on toy problems.
- **C** LoRA: base weights receive zero gradient (frozen); adapter merge is correct; rank edges `r=1` and `r=full`.
- **C/D** Optimizer state save/restore bit-exact; resume == uninterrupted run.
- **E/D** No NaN/Inf over ≥1k steps; bf16-master mixed-precision path matches; same seed ⇒ same loss curve.
- **Pe** A real ternary fine-tune recovers `≥90%` of the lost accuracy gap vs the fp16 baseline; loss decreases (convergence smoke).

**Exit gate:** all gradient checks pass + real fine-tune recovers accuracy + reproducible. Tag `v0.50`.

### 0.60 — Pretraining + distributed

**Scope:** data pipeline, FSDP/DDP, checkpointing, multi-node, from-scratch
pretraining.

**Testing blockers:**
- A ≥2-GPU CI lane (or documented manual gate) and a tiny from-scratch model target.

**Validation conditions:**
- **C/P** N-GPU (FSDP/DDP) loss curve matches 1-GPU within tolerance for the same global batch+seed — the load-bearing distributed-correctness test.
- **C** All-reduced gradients equal a single-process summed reference.
- **C/E** Checkpoint resharding: save on K GPUs, restore on J≠K GPUs ⇒ identical forward; resume continues the loss curve.
- **F/M** Killing a rank mid-run yields a clean error or recovery, never a corrupt checkpoint.
- **C/E** Data pipeline: deterministic per-seed shuffle; no sample duplication or loss across shards (coverage test); resumable mid-epoch.
- **Pe** Near-linear throughput scaling to the target GPU count: `≥80%` scaling efficiency (per-GPU throughput vs single-GPU) at the target count.
- **C** From-scratch tiny model reaches a target loss in fixed steps (pretrain smoke).

**Exit gate:** multi-GPU == single-GPU loss + checkpoint resharding correct + ≥80% scaling efficiency. Tag `v0.60`.

### 0.70 — Backend breadth (depth → breadth pivot)

**Scope:** `tritium-metal`, `tritium-rocm`, `tritium-wgpu`/`tritium-wasm`. Each
implements `tritium-spec`.

**Testing blockers:**
- CI lanes (or documented emulation) for each new platform.

**Validation conditions:**
- **C/P** Each new backend passes the **full** conformance vector set — the same one CPU/CUDA pass. Bit-exact integer paths, ≤ε float. This is the entire reason `tritium-testkit` exists.
- **C** Every backend reproduces the acceptance model's greedy token output (or sampling distribution).
- **E/F** Platform edges: WASM memory ceiling; Metal unified-memory path; ROCm arch variants; graceful, defined fallback when a backend lacks a capability (e.g., no fp8).
- **U7/M** Per-platform sanitizer/leak checks where tooling exists.

**Exit gate:** every backend green on the same conformance suite + reference-model parity. Tag `v0.70`.

### 0.80 — Interop / frontends

**Scope:** `tritium-onnx` (custom op → EP), `tritium-candle`/`tritium-burn`,
`tritium-ffi` (C ABI), `tritium-serve` (OpenAI-compatible).

**Validation conditions:**
- **C** ONNX: a graph with `TritiumMatMul` produces output equal to the native path; ORT loads/runs it; unsupported-op fallback defined.
- **C** candle/burn: a model built in each framework runs and matches the reference.
- **F/Co** FFI: header compiles under C and C++; a C test round-trips and equals the Rust result; ABI version checked; null/invalid args ⇒ error code, never crash; calls are thread-safe.
- **C/Co/F** serve: OpenAI-schema contract test passes; streaming correctness; concurrent requests; backpressure; graceful shutdown mid-stream.

**Exit gate:** every interop surface has an end-to-end test matching the native reference. Tag `v0.80`.

### 0.90 — Hardening

**Scope:** docs (mdbook), fuzzing breadth, full CI matrix, packaging (wheels +
crates.io), perf-regression enforcement, security review.

**Validation conditions:**
- **S** All parsers ≥ 24h cumulative fuzzing with zero open findings; corpora committed.
- **M/S** ASan/UBSan/TSan/`miri`/`compute-sanitizer` clean across the whole suite.
- **U8** Full CI matrix builds+tests green on every target; wheels build for manylinux/macOS/Windows; `cargo publish --dry-run` clean for every crate.
- **Do** mdbook builds with no dead links; every public API documented; examples run in CI.
- **Pe** Perf-regression gates enforced on the main branch.
- **S** Security review completed; threat model for untrusted model files documented; `cargo-deny` clean (licenses + CVEs); SBOM generated.

**Exit gate:** zero sanitizer/fuzz findings + full CI matrix green + docs complete. Tag `v0.90`.

### 1.0 — Release

**Scope:** API/ABI freeze, semver enforcement, final docs + benchmark report.

**Validation conditions:**
- **Do** `cargo-semver-checks` baseline set; public API + C ABI frozen.
- **C** Every prior milestone gate re-run green on the release commit (no regression) — full suite.
- **Do** Quickstart, model zoo, and a benchmark report reproducible by a third party.
- **Real & usable (capstone):** in a fresh environment, following only public docs, a user can `pip`/`cargo` install, load a model, run inference, quantize with SALT, and fine-tune — validated by an end-to-end fresh-env test in CI.

**Exit gate:** all above + external reproduction of the quickstart. Tag `v1.0.0`.

---

## Consequences

- **Positive:** every milestone is independently shippable and provably working;
  the contract hardens on two backends before breadth; the heaviest research
  (SALT, distributed pretraining) sits in the middle, de-risked by a working
  foundation; the validation taxonomy gives a single, uniform definition of done.
- **Negative:** Metal/ROCm/WASM users wait until 0.70; the gate discipline is
  heavy and will feel slow on early milestones; building reference oracles
  (bitnet.cpp parity, distributed single-vs-multi) is real upfront work.

## Open questions

- **0.40 vs 0.50 order:** SALT (quantize) precedes training. Since QAT heals SALT
  loss, they could merge or swap. Kept separate so each ships independently;
  revisit if the heal step proves mandatory for SALT to hit its accuracy gate.
- **Time horizon / resourcing:** unset. Milestones are ordering+gates only; add
  dates once team size is known.
- **Acceptance model:** assumed BitNet b1.58 2B4T as the parity oracle — confirm,
  or pin an additional model.
- **Per-platform CI availability** for Metal/ROCm at 0.70 (hosted runners vs self-hosted).
