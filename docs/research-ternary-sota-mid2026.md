# Ternary Inference & Optimization: SOTA Landscape, Mid-2026

> Deep research survey — 2026-07-11. Adversarially verified claims across 5 streams (kernels/compilers, models, PTQ, hardware, serving stacks); 105 agents, 3-vote verification per claim. Complements `research-ternary-ecosystem-and-tools.md` (June 2026) with what's NEW since.
>
> **Verification key**: claims marked ✅ survived 3-vote adversarial verification; ⚠️ VENDOR = self-reported, failed or skipped verification; ❌ REFUTED = killed by ≥2/3 refute votes.

---

## Executive Summary

Five findings change Tritium's landscape since the June survey:

1. ✅ **PTQ has reached ternary.** PT²-LLM (ICLR 2026, peer-reviewed) ternarizes LLaMA-7B in 32 minutes on one A800 and matches/beats SOTA 2-bit PTQ at lower memory. QAT is no longer the only route to quality ternary checkpoints. (TWLA's W1.58A4 and PT-BitNet's 70B-scale numbers are vendor-reported — the PT-BitNet 61%-vs-51.2% comparison was **refuted** on verification.)
2. ✅ **A new ternary model family shipped.** PrismML's Ternary Bonsai (8B/4B/1.7B, Apache 2.0, Qwen3-based, *fully* ternary incl. embeddings and LM head) — the largest ternary release since BitNet 2B4T, with real uptake (57K+ downloads). ⚠️ Its benchmark-superiority table (75.5 avg, "2nd place") was **refuted** as vendor marketing; treat quality claims as unverified until independently evaluated. Its "1.58-bit" is ~2.1 effective bpw storage.
3. ⚠️ **Sparsity+ternary on GPU: promising but unproven.** spbitnet's 58.3 tok/s on RTX 3060 Laptop is internally consistent, but its headline claims — "ternary naturally fits 2:4 with ~8% pruning", "matches cuBLAS INT8", "cuSPARSELt 3-5× at batch≥16" — **all failed adversarial verification** (0-star single-author repo). The 2:4 direction remains interesting but needs first-party measurement before Tritium invests.
4. ✅ **Mainstream serving still has zero ternary support.** vLLM's BitNet PRs died unmerged (Nov 2025); the Jan 2026 feature request sits unanswered. The verified ecosystem progress is concentrated in CPU/edge kernels, compiler stacks, models, and PTQ — *not* serving stacks or silicon.
5. ⚠️ **Ternary silicon is a wave of papers, not yet product.** TeLLMe v2 (edge FPGA), VitaLLM (16nm), T-SAR (SIMD extension), KU Leuven RTL generator — all primary-source but single-team, unreplicated. No native ternary instruction shipped in any mainstream NPU/GPU; Blackwell went NVFP4 instead.

---

## 1. Kernels & Compiler Stacks

### 1.1 spbitnet — Ternary + 2:4 Sparse Tensor Cores (⚠️ mostly refuted, idea still interesting)

**Repo**: [github.com/Artemarius/spbitnet](https://github.com/Artemarius/spbitnet) — claims to implement "Sparse-BitNet" (Zhang et al., March 2026). 0-star single-author repo.

The claimed insight: ternary weights (~43% zeros) almost satisfy NVIDIA's 2:4 structured-sparsity constraint with ~8% additional pruning, unlocking Ampere+ sparse tensor cores.

**Verification verdicts (3-vote adversarial):**
- ✅ Survived (low confidence): **58.3 tok/s decode** for a 2:4-sparsified BitNet-2B4T on RTX 3060 Laptop at 2.6 GB VRAM, BitLinear fusion +15.6% — internally consistent and bandwidth-plausible, but self-reported
- ❌ REFUTED (0-3): "ternary naturally satisfies 2:4 with ~8% extra pruning"
- ❌ REFUTED (1-2): "sparse ternary GEMV matches/beats cuBLAS INT8 with 5.3× less traffic"
- ❌ REFUTED (0-3): "cuSPARSELt gives 3-5× over dense cuBLAS at batch ≥ 16"

**Tritium takeaway**: The 2:4-sparse-ternary *direction* is still conceptually attractive (hardware sparse tensor cores vs software block-skip), but every load-bearing number failed verification. If pursued, treat as a from-scratch research question: measure the true pruning fraction needed on a real ternary checkpoint, and benchmark cuSPARSELt on Tritium's own shapes before believing any speedup. Do not cite spbitnet's numbers.

### 1.2 Vec-LUT — Vector Table Lookup (MobiSys '26)

**Paper**: arXiv:2512.06443 · **Code**: [github.com/OpenBitSys/vlut.cpp](https://github.com/OpenBitSys/vlut.cpp) (llama.cpp-integrated)

Fixes the scalar-LUT paradigm's memory-bandwidth underutilization (≤40%) during *parallel* inference: constructs one unified LUT across parallel tokens and does a single 1→N lookup per index.

- Up to **4.2× over T-MAC/bitnet.cpp/llama.cpp** on edge CPUs; 2.1× energy savings
- **1.60 bits/weight lossless ternary packing** — tighter than TQ2_0's 2.0 bpw
- 2 CPU cores on Snapdragon 8 Elite beats llama.cpp's Hexagon NPU backend; 273.5 tok/s continuous batching on a $0.50/h CPU server

**Tritium takeaway**: the 1.60 bpw packing is a direct challenge to TQ2_0's 2.125 effective bpw (a ~20% bandwidth saving in the memory-bound regime); the vector-LUT idea maps to Tritium's batched decode (M>1) path.

### 1.3 Tilus — NVIDIA-backed Low-Precision GPU DSL

**Paper**: arXiv:2504.12984 · **Code**: [github.com/NVIDIA/tilus](https://github.com/NVIDIA/tilus)

Tile-level GPGPU language with a type system for **arbitrary 1-8 bit types**. Geomean speedups: 1.75× over Triton, 2.61× over Ladder, 1.03× over Marlin. Abstracts sub-byte packing/unpacking at register level — no manual bit twiddling.

**Tritium takeaway**: candidate codegen backend for kernel variants Tritium currently hand-writes; also a signal that NVIDIA sees sub-4-bit as a first-class compiler target.

### 1.4 FairyFuse + the "GPUs can't do ternary" claim to refute

**Paper**: arXiv:2604.20913 (Apr 2026)

CPU kernel: BMI2 `_pext_u32` mask decode + AVX-512 masked `vaddps/vsubps` — zero FP multiplies, zero table lookups. 32.4 tok/s for a 7B on one Xeon socket (1.24× over Q4_K_M at half the bits). Their roofline: ternary compression lifts AI from 0.25 to 8.0 OP/byte, near the CPU ridge of 13.5.

⚠️ The paper claims GPU ternary regresses **130× vs FP16** on H200 — but their GPU baseline reuses the CPU algorithm (pext-style bit extraction) instead of a GPU-native DP4A/LUT design. Tritium's decode path is a standing refutation; worth citing in the paper direction.

### 1.5 Others

- **APT-LLM** (arXiv:2508.19087): arbitrary-precision GEMM on tensor cores via bit-plane decomposition + INT1 IMMA primitives — directly adjacent to SALT multi-plane + IMMA prefill.
- **QVAC Fabric BitNet** (Tether AI, Jan 2026): llama.cpp-based **Vulkan/Metal** GPU backend for BitNet — "first GPU backend" claim, runs TQ1_0/TQ2_0. **1B model: 258 tok/s on RTX 4090.** Also does on-device LoRA fine-tuning of ternary models (1B in ~80 min on phone GPUs). Numerics gate: 99.04% same-top-token, KL < 0.0003 vs CPU reference — a fidelity methodology worth copying.
- **0xBitNet**: pure-WebGPU (WGSL) BitNet inference in browser, consumes I2_S GGUF. Small (20 stars) but proves the WGSL route Tritium's wasm story could take.

---

## 2. Models — New Ternary Checkpoints

### 2.1 Ternary Bonsai (PrismML, April 2026) — the headline release

**HF**: [prism-ml collection](https://huggingface.co/collections/prism-ml/ternary-bonsai) — 8B/4B/1.7B, Apache 2.0

| Item | Detail |
|---|---|
| Base | Qwen3 (8B = 36 layers, GQA 32/8, 65K ctx) |
| Format | "GGUF Q2_0 g128": {-1,0,+1} + FP16 scale per 128 weights (2.125 bpw eff., 2.03 GiB for 8B) — note "1.58-bit" marketing = ~2.1 bpw storage |
| Coverage | ✅ **Fully ternary incl. embeddings and LM head** (verified) |
| Quality | ⚠️ Vendor table: 75.5 avg (MMLU-R 72.6, GSM8K 91, HE+ 77.4, IFEval 81.8, BFCL 73.9) vs Qwen3-8B 79.3. The "outperforms all compared models / 2nd place" claim was **REFUTED (0-3)** as unverifiable vendor marketing — independent evaluation needed before citing |
| Speed | ⚠️ Vendor: 76 tok/s Metal M4 Pro; 82 tok/s MLX; 27 tok/s iPhone 17 Pro Max |
| Kernels | **Not upstream** — requires PrismML's llama.cpp fork (`prism` branch); upstream PR "coming soon" |

Ecosystem forming fast: Hexagon NPU builds (runanywhere), TQ2_0 GGUF conversions (ewchampion), MoE LoRA variants, a 4B ternary *image* model (bonsai-image, Flux-based) — plus community TQ2_0 repacks that Tritium can load **today**.

**Tritium takeaways**: (a) three sizes of high-quality Qwen3-arch ternary checkpoints — the Qwen-family loader (plan 0037) is exactly the right surface; (b) their Q2_0 g128 differs from ggml TQ2_0 (g256) only in group size — a format-crosswalk/import is cheap; (c) their quality-per-byte framing ("intelligence density") matches the byte-optimal-ternary paper direction, and Bonsai is the model to beat/compare.

### 2.2 Falcon-Edge (TII) + onebitllms

1B/3B BitNet-arch ternary, ~1.5T tokens, base+instruct. Novel: **one training run yields both ternary and bfloat16 variants** ("universal checkpoint"). Ternary pretrain overhead vs standard: **~20%**. `onebitllms` (Triton BitnetLinear) is now integrated into **Axolotl** — mainstream fine-tuning stack support for ternary.

### 2.3 Fairy2i — the complex-valued wildcard

arXiv:2512.02901: converts existing FP16 checkpoints into complex-weight networks with all parameters in **{±1, ±i}** (2-bit phase encoding, multiply-free) via light QAT ("PhaseQuant", a few calibration epochs). LLaMA-2-7B: WikiText-2 ppl 5.52 vs 5.47 FP16 — beats Q4_K_M. A competing sub-2-bit lattice to ternary; watch it.

---

## 3. Post-Training Quantization Reaching Ternary Quality

The big shift: **training-free ternary is now viable**, at scales QAT can't reach.

| Method | Venue | Verdict | Regime | Headline result | Cost |
|---|---|---|---|---|---|
| **PT²-LLM** | ICLR 2026 | ✅ verified | W1.58 | Matches/beats SOTA 2-bit PTQ (Slim-LLM/GPTQ/AWQ/QuIP) at lower memory (7.17× vs 5.86× compression); vs PB-LLM on LLaMA-7B: 33.4→45.1 avg acc; +2.1× e2e throughput over 2-bit on llama.cpp; validated on Qwen3-14B | **32 min, 1×A800, 128 samples** |
| **TWLA** | Jun 2026 | ⚠️ vendor | **W1.58A4** | LLaMA2-70B: 71.10 avg (92% of FP16) vs ResQ 56.43; 3.64× over FP16 on A6000 | 82 min for 7B |
| **PT-BitNet** | Neural Networks 2025 | ⚠️ method ✅ / numbers ❌ | W1.58 | Two-stage PTQ-to-ternary scaling to 70B is real (peer-reviewed); the "61% vs 51.2%" comparison was **REFUTED (0-3)** | PTQ-only |
| Spectral rotations (auto-round) | May 2026 | W2A16 | −15-58% ppl on sub-2B models; per-head PCA fix for **Qwen3 QK-norm** (ppl 136.8→89.0) | cheap |
| BCJR-QAT | May 2026 | 2 bpw trellis | Differentiable QTIP; but documents the "proxy gap" (per-layer MSE anti-correlates with end-task ppl) | consumer GPU |

**Tritium takeaways**:
- PT²-LLM's Iterative Ternary Fitting + Activation-aware Grid Alignment is directly comparable to SALT's plane fitting — benchmark SALT reconstruction against it; ITF-style alternation could improve SALT's per-plane fit.
- TWLA's W1.58A4 result matters for the IMMA/DP4A path: 4-bit activations double effective integer-pipeline throughput vs INT8.
- The rotation line (Hadamard/PCA pre-processing) is *composable* with SALT as an error-flattening front end — and the Qwen3 QK-norm fix is directly relevant to the Qwen loader.
- BCJR-QAT's proxy-gap negative result is a warning for SALT allocation: minimizing per-layer MSE can hurt end-task quality.

---

## 4. Hardware with Native Ternary Support

The direction mainstream GPU silicon took is **NVFP4** (Blackwell 5th-gen tensor cores: FP4/FP6 with 16-element FP8-scaled micro-blocks) — *not* native ternary. Ternary silicon is happening in FPGA/ASIC/CPU-extension land:

| Design | Type | Headline |
|---|---|---|
| **TeLLMe v2** (arXiv:2510.15926) | Edge FPGA | First end-to-end (prefill+decode) ternary accelerator; 25 tok/s, TTFT 0.45-0.96s, **< 5-7 W**; table-lookup matmul |
| **Slim-Llama** | 28nm ASIC (silicon-proven) | Binary/ternary LUT accelerator, **189.8 TOPS/W** |
| **VitaLLM** (arXiv:2604.27396) | 16nm ASIC | Dual-core (TINT ternary + BoothFlex attention); 70.7 tok/s in 0.223 mm² @ 66 mW — 17.4 TOPS/mm²/W |
| **T-SAR** (arXiv:2511.13676) | CPU SIMD extension | In-register LUT generation; 5.6-24.5× GEMM latency; **1.4% area, 3.2% power overhead** — a near-free ternary ISA extension template |
| **KU Leuven ternary-lut-dse** (arXiv:2604.25183) | Open-source RTL generator | Chisel generator + TSMC-16nm-validated cost model for the whole LUT-accelerator design space |

Two findings from the KU Leuven DSE that matter for Tritium's *software* kernels:
1. **LUT gains are governed by activation type**: big for FP16 activations, **nearly negligible for INT8** (optimal group size collapses to μ=1 = plain add-accumulate). → Tritium's DP4A path with INT8 activations likely gains little from a LUT kernel; the Tier-2 "LUT GEMV" action item should be re-scoped to the FP16-activation path only.
2. **Dense ternary packing at ~1.6 bpw** (groups of μ trits in ⌈log2((3^μ−1)/2)⌉+1 bits) cuts bandwidth 20% vs 2-bit packing with no runtime decode penalty in hardware — same arithmetic Vec-LUT exploits in software.

---

## 5. Serving Stacks

| Stack | Ternary status (mid-2026) |
|---|---|
| **vLLM** | **None.** PR #17588 (BitNet) closed unmerged Nov 2025; issue #33142 (Tequila/Sherry/BitNet, Jan 2026) open, no maintainer response, auto-staled. BUT: Q2 2026 roadmap commits to **W{1-8}A{16/8/4} kernel coverage via "humming-kernel"**, QuaRot/SpinQuant-style rotations, and names **DFlash as a first-class speculation backend** (EAGLE/DFlash/MTP). Sub-4-bit energy is going into **KV-cache** (TurboQuant RFC: 2-bit KV, 7.5× reduction, exact-match preserved). |
| **llama.cpp** | De-facto reference deployment for PTQ-ternarized models (TWLA and PT²-LLM both benchmark on it). TQ1_0/TQ2_0 remain CPU-oriented; PrismML forked it for Q2_0 g128 + Metal. |
| **bitnet.cpp** | Still the de-facto ternary stack; GPU kernels remain thin. |
| **QVAC Fabric BitNet** | New (Jan 2026) Vulkan/Metal GPU serving + LoRA fine-tuning; the closest thing to a "ternary serving stack" besides Tritium. |
| **MLX** | prism-ml MLX 2-bit conversions; exo-explore/mlx-bitnet. Apple-silicon ternary is real. |

**Competitive read**: the ternary serving gap Tritium occupies has *persisted for 14 months* of open vLLM requests. The window is real but closing: humming-kernel (vLLM Q2 2026) and QVAC (Vulkan) are the two vectors that could fill it. Tritium's differentiators remain CUDA-native decode perf, SALT, spec-decode verify, and training.

---

## 6. Consolidated Action Items for Tritium

Ordered by expected impact, re-ranked after verification:

1. **Load Ternary Bonsai.** ✅ The release itself, format, and full-network ternary coverage are verified. Community TQ2_0 conversions exist now; a Q2_0-g128→TQ2_0 crosswalk makes the whole family (1.7B/4B/8B, Qwen3 arch = plan 0037 surface) native. Bonus: independently evaluating Bonsai's quality (the vendor table was refuted) is itself a paper-grade contribution and the anchor for the byte-optimal paper.
2. **Benchmark SALT vs PT²-LLM.** ✅ PT²-LLM is the verified PTQ-to-ternary quality baseline. Same-checkpoint reconstruction fidelity + downstream quality; adopt ITF-style alternating grid/rounding refinement into SALT plane fitting if it wins. Open question from the research: does an ITF/AGA-initialized ternary base + SALT residual planes close the gap to fp without QAT?
3. **Consider ~1.6 bpw dense packing** (✅ Vec-LUT's I1 packing is verified at 1.60 bpw; KU Leuven's hardware encoding agrees) as a TQ2_0 successor for the memory-bound decode path: −20% weight bandwidth ≈ +20-25% decode tok/s ceiling. Open question: does the GPU decode cost of the denser packing eat the bandwidth win?
4. **Port the vector-LUT idea to GPU shared memory** for prefill/batched paths (open question flagged by the research) — but note the KU Leuven caveat that LUT gains collapse for INT8 activations; scope to f16-activation kernels.
5. **Re-scope the LUT kernel Tier-2 item.** KU Leuven DSE says LUT ≈ no-op for INT8 activations. Keep LUT only for the f32/f16-activation kernels; the DP4A path stays add-accumulate.
6. **2:4 sparse ternary: demote to research question.** ⚠️ spbitnet's enabling claims were refuted. If pursued: measure the real pruning fraction on an actual ternary checkpoint and benchmark cuSPARSELt on Tritium's shapes first-party. Related open question: do Blackwell NVFP4/INT4 tensor-core paths offer a ternary win over DP4A?
7. **W1.58A4 (TWLA-style) activation path** for prefill — vendor-reported, verify quality first.
8. **Watch/engage vLLM humming-kernel** (W1-8 coverage) — earliest upstream integration point for ternary serving; DFlash's first-class status there also validates the ADR 0014 drafter bet.
9. **Cite-and-refute FairyFuse's 130× GPU-regression claim** in the paper. ✅ The refutation itself is verified (3-0): their GPU baseline is a one-row-per-thread CPU-algorithm port; Microsoft's W2A8 GPU kernels (2-3× over BF16) and llama.cpp's merged TQ2_0 CUDA kernels (PR #11183) are counterexamples, as is Tritium's DP4A decode.

---

## 7. Where Tritium Stands — Measured Head-to-Head (2026-07-11)

Same machine (RTX 4090 24GB + i5-13600K, driver 610.43), same day, light desktop contention (~1.8 GB VRAM idle-resident). Tritium @ a80e185 (release build), llama.cpp fork @ 41a666dac (build 8943, CUDA). Tritium numbers: `tritium report decode` (256 steps, 16 warmup, ×3 runs) and `report ttft` (512-token prompt, 5 runs, p50). llama.cpp numbers: `llama-bench -p 512 -n 128 -ngl 99`.

### Decode (single-stream, the ternary headline regime)

| engine | model | weights | decode tok/s | eff. weight-stream BW |
|---|---|---|---:|---:|
| **Tritium CUDA** | BitNet 2B4T (ternary I2_S) | 1.71 GiB | **301.4–302.8** | **~517 GiB/s** |
| llama.cpp CUDA | Qwen3.5-4B Q4_K_M | 2.54 GiB | 200.1 ± 6.5 | ~508 GiB/s |
| llama.cpp CUDA | Qwen3.5-4B **TQ2_0** | 1.35 GiB | **24.0 ± 1.6** | ~32 GiB/s |
| llama.cpp CUDA | Qwen2.5-0.5B TQ2_0 | 300 MiB | 104.5 ± 18.7 | (latency-bound) |
| llama.cpp CUDA | Qwen2.5-0.5B Q4_K_M | 374 MiB | 633.5 ± 45.0 | (latency-bound) |
| bitnet.cpp CPU (14t, prior same-box round) | BitNet 2B4T I2_S | 1.71 GiB | 23.1 ± 0.1 | — |

Three readings:

1. **Bandwidth-efficiency parity with the mainstream flagship.** Effective weight-streaming bandwidth (tok/s × weight bytes) is the honest cross-model metric in the memory-bound regime: Tritium's ternary decode sustains ~517 GiB/s vs ~508 GiB/s for llama.cpp's heavily-tuned Q4_K_M CUDA path. Tritium's ternary kernels are as bandwidth-efficient as the most-optimized 4-bit path in the ecosystem — while moving 1.58-bit weights, which llama.cpp cannot do on GPU at all.
2. **llama.cpp's TQ2_0 on CUDA is a trap, not a path.** No TQ2_0 CUDA kernels exist in the tree (verified by source grep); the 4B decodes at 24 tok/s — **12.6× slower than Tritium** runs a comparable ternary model, and 8× slower than the same file would need to break even with Q4_K_M. Prefill collapses too (107 vs 12,281 tok/s for Q4_K_M). The "ternary GPU gap" the SOTA survey found in serving stacks is directly reproducible on this box.
3. **Tritium remains the only engine on this machine that runs a ternary model fast on consumer CUDA.** 302 tok/s vs the best non-Tritium ternary option measured here (bitnet.cpp CPU, 23.1 tok/s) = **13×**.

### Prefill

| engine | model | pp512 tok/s |
|---|---|---:|
| llama.cpp CUDA | Qwen3.5-4B Q4_K_M | 12,281 ± 828 |
| **Tritium CUDA** | BitNet 2B4T | **1,068 (p50 320ms/512tok; first-run compile ~1.1-1.2s)** |
| llama.cpp CUDA | Qwen3.5-4B TQ2_0 | 107 ± 5 |
| bitnet.cpp CPU | BitNet 2B4T | ~203 |

Honest reading: prefill is Tritium's biggest deficit vs mainstream — ~11× behind llama.cpp's Q4_K_M compute-bound path (which rides cuBLAS/MMQ tensor-core batched GEMMs). The IMMA prefill path narrows this but the current number is ~1.1K tok/s. Against *ternary* alternatives, Tritium still leads (10× over llama.cpp-TQ2_0, 5× over bitnet.cpp CPU).

### Against the SOTA survey's external claims (vendor-reported, different hardware)

| claim | their number | Tritium today | verdict |
|---|---|---|---|
| QVAC Fabric BitNet (Vulkan), **1B** BitNet on RTX 4090 | 258 tok/s | **302 tok/s on 2.4B** (≈2.4× the model) | Tritium ~2.8× better bandwidth-normalized |
| spbitnet sparse ternary, 2B4T on RTX 3060 Laptop (~336 GB/s) | 58.3 tok/s | 302 tok/s on ~1008 GB/s | ≈ parity bandwidth-normalized (their claims part-refuted anyway) |
| Ternary Bonsai 8B, M4 Pro Metal (~273 GB/s) | 76 tok/s | different model class | n/a — becomes comparable once Bonsai loads in Tritium |
| PT²-LLM e2e on A800 via llama.cpp | +2.1× over 2-bit | n/a (quality-side claim) | SALT-vs-PT²-LLM bench is the open action |

### Scorecard vs SOTA, by category

| category | SOTA reference | Tritium position |
|---|---|---|
| Ternary decode, consumer CUDA | (Tritium) — no faster measured alternative exists on this hw | **Leads.** 302 tok/s @2.4B; 36% of its own 848 tok/s roofline; next lever = rmsnorm_fast (Track E) + persistent-kernel decode |
| Ternary prefill, CUDA | llama.cpp Q4_K_M-class MMQ (12K tok/s @4B, non-ternary) | **~11× behind mainstream compute-bound paths**; leads all ternary paths |
| CPU ternary | bitnet.cpp TL2 / Vec-LUT (4.2× over T-MAC) | Non-goal (reference backend, 0.95 tok/s); honest 24× deficit stands |
| Spec-decode | JetFlow 9.64× (H100, trained head); BASTION 6.61× | **1.19× lossless** with model-free lookup drafter; verify ~3× a decode step (launch storm) — drafter (BLUT) + verify-graph are the gap |
| Batching/serving | vLLM continuous batching (no ternary at all) | Own continuous batching live (1.51× aggregate @8 slots, paged KV −94%, tree sessions coexist); no PagedAttention-scale scheduler |
| KV compression | vLLM TurboQuant RFC (2-bit KV) | f16 rung live (+38% @4K ctx), i8-g64 memory rung live; ternary KV measured and honestly REJECTED |
| Quantization quality | PT²-LLM (PTQ-ternary), ParetoQ (QAT) | SALT multi-plane + distill pipeline done e2e; **no downstream-benchmark eval yet** — the paper blocker and the Bonsai-eval opportunity |
| Model coverage | llama.cpp (everything) | BitNet 2B4T + SmolLM2/fp-HF loader; Qwen-arch loader in flight (plan 0037) → unlocks Bonsai family |

**Bottom line**: in the one regime where ternary's advantage is largest and unique — memory-bound single-stream decode on consumer CUDA — Tritium is measurably the fastest thing that exists today, at mainstream-grade bandwidth efficiency. It is ~11× behind mainstream on compute-bound prefill, intentionally absent on CPU, and mid-pack on spec-decode until a real drafter lands. The quality story (SALT vs PT²-LLM, Bonsai independent eval) is unmeasured — and is both the paper blocker and the field's most cite-able open question.

## Sources

Kernels/compilers: [spbitnet](https://github.com/Artemarius/spbitnet) · [Vec-LUT](https://arxiv.org/abs/2512.06443) / [vlut.cpp](https://github.com/OpenBitSys/vlut.cpp) · [Tilus](https://arxiv.org/abs/2504.12984) / [NVIDIA/tilus](https://github.com/NVIDIA/tilus) · [FairyFuse](https://arxiv.org/abs/2604.20913) · [APT-LLM](https://arxiv.org/abs/2508.19087) · [QVAC Fabric BitNet](https://github.com/tetherto/qvac-rnd-fabric-llm-bitnet) · [0xBitNet](https://github.com/m96-chan/0xBitNet)

Models: [Ternary Bonsai](https://huggingface.co/collections/prism-ml/ternary-bonsai) · [Falcon-Edge](https://huggingface.co/blog/tiiuae/falcon-edge) · [Fairy2i](https://arxiv.org/abs/2512.02901) · [Axolotl ternary](https://huggingface.co/blog/axolotl-ai-co/finetuning-ternary-llms-tii-axolotl)

PTQ: [PT²-LLM](https://arxiv.org/abs/2510.03267) · [TWLA](https://arxiv.org/abs/2606.13054) · [PT-BitNet](https://doi.org/10.1016/j.neunet.2025.107855) · [spectral rotations](https://arxiv.org/abs/2605.25203) · [PTQTP](https://hf.co/papers/2509.16989) · [BPDQ](https://hf.co/papers/2602.04163)

Hardware: [TeLLMe v2](https://arxiv.org/abs/2510.15926) · [VitaLLM](https://hf.co/papers/2604.27396) · [T-SAR](https://arxiv.org/abs/2511.13676) · [KU Leuven ternary-lut-dse](https://arxiv.org/abs/2604.25183) · [NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

Serving: [vLLM #33142](https://github.com/vllm-project/vllm/issues/33142) · [vLLM Q2 2026 roadmap #39749](https://github.com/vllm-project/vllm/issues/39749) · [TurboQuant RFC](https://github.com/vllm-project/vllm-omni/issues/2215) · [Tencent AngelSlim](https://github.com/Tencent/AngelSlim)

---

## 8. SALT V2 update through 2026-07-14

This update narrows the question from “what is the best low-bit method?” to
“what can improve a zero-point-free, lookup-free, additive ternary model without
silently changing its execution contract?” It also corrects nominal-rate comparisons
that would otherwise make the campaign unfalsifiable.

### 8.1 Claim boundary

No published conversion of a pretrained 8B- or 32B-class dense model to strict
symmetric ternary weights has demonstrated approximately zero quality loss as of the
cutoff. Native ternary training can reach parity, but it consumes 100B–300B+ training
tokens and is not evidence for an inexpensive conversion. The defensible initial SALT
V2 claim is therefore:

> Test the first zero-point-free, lookup-free, successively refinable additive
> ternary pipeline that combines exact joint plane fitting, end-loss-aware curvature,
> second-order error feedback, physically budgeted plane growth, smooth-to-hard
> recovery, and native fused additive kernels.

“SOTA additive ternary Pareto frontier” is a measured campaign outcome, not an
architectural assertion. “Global low-bit SOTA” remains gated on matched-checkpoint,
matched-rate, matched-data, and matched-hardware results.

### 8.2 Normalized quality, rate, and cost

Rates below are the paper's reported rate unless a physical correction is shown.
Whole-model file and resident rates can be higher because embeddings, heads, scales,
maps, padding, repacks, and container metadata are not consistently counted by the
literature.

| Method | Representation and optimizer | Representative published result | Published or derived cost |
|---|---|---|---|
| [AQLM](https://arxiv.org/abs/2401.06118) | Sum of arbitrary FP codewords; beam assignment; block reconstruction and optional KL tuning | Llama-2 7B/13B/70B WikiText-2 PPL 6.14/5.33/3.83 vs FP 5.12/4.57/3.12 at about 2 bpw | About 24 A100-hours for 7B; some 70B runs about 720 A100 GPU-hours |
| [VPTQ](https://arxiv.org/abs/2409.17066) | Hessian-weighted vector quantization plus GPTQ propagation and optional residual/outlier codebooks | Llama-2 7B/13B/70B PPL 6.13/5.32/3.93 | 8/12.8/76 A100 GPU-hours for 7B/13B/70B |
| [PV-Tuning](https://arxiv.org/abs/2405.14852) | Alternating continuous parameters and exact discrete code updates under teacher KL | Llama-2 7B/13B/70B PPL 5.84/5.12/3.78 | Useful 70B gains by about 384 GPU-hours; longest reported runs 1,536 GPU-hours |
| [QuIP#](https://arxiv.org/abs/2402.04396) | Randomized Hadamard transform, BlockLDLQ, E8 lattice quantization | Llama-2 7B/13B/70B PPL 6.19/5.35/3.91 with fine-tuning | 70B under 10 GPU-hours without fine-tuning and about 100 with it, excluding Hessian generation |
| [QTIP](https://proceedings.neurips.cc/paper_files/paper/2024/file/6de2e84b8da47bb2eb5e2ac96c63d2b0-Paper-Conference.pdf) | Randomized transform, BlockLDLQ, trellis-coded Gaussian quantization | Fine-tuned Llama-2 7B/13B/70B PPL 5.86/5.11/3.70 | Exactly 2-bit trellis payload; deployed rate rises with scales and whole-model exclusions |
| [GuidedQuant](https://proceedings.mlr.press/v267/kim25d.html) / [YAQA](https://arxiv.org/abs/2505.22988) | End-loss or real-Fisher curvature and feedback rounding; representation-independent | GuidedQuant improves QTIP Llama-2-7B PPL 6.82→6.11; YAQA improves QTIP Llama-3.1-8B 9.39→8.39 | Gradient/Fisher cache is the added cost; reduced YAQA configurations report about one GPU-hour |
| [BPDQ](https://arxiv.org/abs/2602.04163) | Additive binary planes, Hessian coordinate search, exact scale refit, GPTQ propagation | Qwen3-32B W3/G128 PPL 9.97 vs FP 9.34 | Exact W3/G128 rate is 3.5 bpw; Qwen2.5-7B about 40 minutes on one H20 |
| [PTQTP](https://arxiv.org/abs/2509.16989) | Two ternary planes, exact nine-way assignment, adaptive ridge scale solve | Qwen3-32B PPL 10.06 vs 8.64 and Avg7 82.09 vs 86.28 | Its two 2-bit trit planes plus two FP16 scales/G128 are **4.25 physical bpw**, not 1.58; 7B takes 11.46 minutes on one A100 |
| [UniSVQ](https://arxiv.org/abs/2606.10520) | Rotated 4D quaternary codes, learned affine transform, LDLQ | Qwen3-32B PPL 9.26 vs 7.61; Avg6 76.15 vs 78.01 | Qwen3-8B about six hours on one A100 |
| [LC-QAT](https://arxiv.org/abs/2606.10531) | Strong LDLQ initialization plus smooth differentiable code estimator | Qwen3-8B PPL 10.23 vs 9.72 after matched 4B-token training | 4B tokens on 16×A100; wall time not disclosed |
| [LLVQ](https://arxiv.org/abs/2603.11021) | Leech-lattice direction with gain coding and spherical GPTQ | Llama-2-7B 2-bit PPL 6.83 PTQ, 5.48 after 52M-token scale tuning vs FP 5.11 | 2-bit payload; end-to-end kernel throughput not reported |
| [LiftQuant](https://arxiv.org/abs/2606.04050) | High-dimensional binary lattice followed by a floating projection | Llama-2-7B PPL 6.53 at 2.00 bpw, 6.10 at 2.40 | Transform metadata about 0.008–0.011 bpw on 70B; 70B reports 31.3 tok/s on RTX 4090D at 2-bit |
| [CAT-Q](https://arxiv.org/abs/2606.26650) | Zero-point-free soft-to-hard ternary recovery | Qwen3-8B Avg 71.57→61.76; Qwen3-32B 76.43→71.19 | 512×2048 calibration tokens, 60 epochs, 8×A100; 1–60 wall-hours across 1.7B–235B |
| [HESTIA](https://arxiv.org/abs/2601.20745) | Sensitivity-aware probabilistic ternary relaxation and hardening | Llama-3.2-1B Avg 55.8→54.7; 3B 63.6→60.1 | 10B tokens; hardware and wall time not disclosed |

The PTQTP correction is load-bearing. Two independently stored ternary planes do
not cost `log2(3)` bits together. With aligned 2-bit trits and two FP16 scales per
128-weight group, the exact core-weight rate is

```text
2 planes × 2 bits/trit + (2 scales × 16 bits) / 128 weights = 4.25 bpw.
```

With ideal radix-3 packing the same two planes are already
`2×log2(3) + 32/128 = 3.420 bpw` before maps, alignment, headers, preserved tensors,
or runtime shadows. SALT V2 must report logical, serialized, resident, all-metadata,
and whole-model rates separately.

### 8.3 What SALT can import without violating additive ternary

Directly compatible mechanisms:

- exact `3^P` joint assignment for `P≤3`, adaptive ridge scale refitting, and
  PV-style sparse discrete polishing;
- GuidedQuant/YAQA end-loss curvature, BlockLDLQ/GPTQ error propagation, and
  BPDQ-style delta correction after scale changes;
- foldable signed Hadamard/outlier-shaping transforms, provided every transform is
  classified as folded or online and online cost is measured;
- overlapping 2–4-block output reconstruction, smooth-to-hard annealing, a real
  final hard tail, and conditional cached-logit KL distillation;
- allocation by measured artifact and resident bytes plus kernel latency, never by
  nominal entropy alone.

Quality-oracle-only mechanisms that must not enter the deployment representation:

- AQLM/VPTQ arbitrary floating codebooks;
- QuIP#/QTIP/LLVQ lattice or trellis decoders;
- UniSVQ affine quaternary decoding and LiftQuant's floating projection;
- dense zero points, arbitrary FP residuals, or a permanent training bypass.

Those mechanisms can be matched baselines. They cannot be called SALT-compatible
because they require lookup, multilevel decode, dense projection, or reconstructed
floating MACs.

### 8.4 Runtime and physical-format update

SALT V2 separates one semantic tensor from compiled physical codecs:

- **D2:** aligned 2-bit trits, mandatory reference and fast-path baseline;
- **B3:** five radix-3 trits per byte, compact disk/capacity candidate;
- **S34:** exactly one zero per four trits, 32 states in five bits, admitted only
  after structure-aware recovery and measured runtime wins.

Adaptive plane count must not pad every row or tile to the tensor-wide maximum.
Plane-count buckets or tile-presence bitmaps are required; one presence bit per
`8×32` tile costs 0.0039 bpw per optional plane, while a `u32` offset per tile costs
0.125 bpw and is rejected. Campaign metadata targets at most 0.01 bpw and dispatch
overhead at most 3%.

Current Tritium reference points remain important: TQ2 is 66 bytes per 256 weights,
or 2.0625 physical bpw per plane; two and three planes are 4.125 and 6.1875 bpw
before row metadata. Existing SALT GPU upload pads rows to maximum `T`, and the
existing additive kernel performs floating scale work within the weight loop. SALT
V2 is not runtime-SOTA until adaptive resident storage, one fused multi-plane kernel,
and a measured A4-capable prefill path land.

### 8.5 Cost-efficient experiment ladder

The campaign starts with Qwen3-8B and promotes only on preregistered evidence:

1. PTQ initializer on 512×2048 pinned tokens.
2. 100M-token hard-export recovery checkpoint.
3. 500M tokens.
4. 1B tokens.
5. Continue toward 10B only while held-out NLL improvement per GPU-hour remains
   above the campaign threshold.

A first-principles estimate, not a vendor quote, is approximately `6PT` FLOPs for
CE-only recovery and `8PT` with an online teacher. For 9B parameters and 10B tokens,
that is roughly 375–600 H100 GPU-hours at 400–250 effective TFLOP/s for CE, or
500–800 H100 GPU-hours with online KD. A100 CE at 100–150 effective TFLOP/s is
roughly 1,000–1,500 GPU-hours. The 1B-token rung is one tenth of those estimates.
Every rental estimate must be replaced by a 100-step measured tokens/s pilot before
the next rung is authorized.

### 8.6 Preregistered SOTA bar

An additive-ternary claim requires matched Qwen3-8B and Qwen3-32B comparisons against
PTQTP/BPDQ and ordinary W4, plus AQLM, VPTQ, QTIP, LLVQ, UniSVQ, and LiftQuant where
reproducible. Report exact package bytes, live GPU allocations, prefill/decode wall
time, calibration and evaluation digests, coverage policy, confidence intervals,
and raw receipts. The Compact profile targets no more than 2.25 physical core-weight
bpw; the NearLossless profile targets no more than 4.0. Neither target is a result.
NearLossless publishes only if strict BF16 non-inferiority gates pass after reload
from the exact artifact.
