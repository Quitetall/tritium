# Ternary ML Ecosystem: Tools, Kernels, Training Recipes, and Speculative Decoding

> Deep research survey — 2026-06-22
> Scope: emergent software, open-source tools, kernel strategies, STE implementations, speculative decoding, training recipes for ternary (1.58-bit) LLMs.

---

## Table of Contents

1. [Kernel Fusion & Decode Optimization](#1-kernel-fusion--decode-optimization)
2. [LUT-Based Ternary GEMM](#2-lut-based-ternary-gemm)
3. [Tequila & Sherry: STE Modifications](#3-tequila--sherry-ste-modifications)
4. [Speculative Decoding with Ternary Drafts](#4-speculative-decoding-with-ternary-drafts)
5. [Training Recipes for Ternary LLMs](#5-training-recipes-for-ternary-llms)
6. [Open-Source Ecosystem Survey](#6-open-source-ecosystem-survey)
7. [Tritium Action Plan](#7-tritium-action-plan)

---

## 1. Kernel Fusion & Decode Optimization

### 1.1 FlashDecoding — Split-K Attention for Decode

**Paper**: Tri Dao et al., 2023 — [pytorch.org/blog/flash-decoding](https://pytorch.org/blog/flash-decoding/)

Core insight: during decode (M=1), FlashAttention v2 parallelizes only over batch + query length, leaving most SMs idle. FlashDecoding adds KV sequence length as a third parallelization dimension via a split-K approach:

1. Split KV into chunks (zero-cost views)
2. Compute partial attention per chunk (FlashAttention kernel)
3. Reduce across chunks using log-sum-exp scalars

**Benchmarks** (A100, 16 query heads, dim 128, 2 KV heads): at B=1, seqlen=65536, FlashDecoding takes 64.4μs vs FlashAttention v2's 2300.6μs (**36× kernel speedup**). End-to-end on CodeLlama-34B: up to **8× faster decode** at long contexts.

**Tritium status**: Already implemented — `gqa_attention_split_partial_f32` + `gqa_attention_combine_f32` in `decode.cu`.

### 1.2 FlashInfer — JIT-Compiled Attention Templates

**Paper**: UC Berkeley, MLSys 2025 — [arxiv.org/abs/2501.01005](https://arxiv.org/abs/2501.01005)
**Repo**: [github.com/flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) (Apache-2.0)

Key features:
- Block-sparse KV cache format
- Ragged tensors (variable-length without padding)
- JIT-compiled attention templates
- POD-Attention (fused prefill+decode in one kernel launch for mixed batches)
- Fused MoE kernels combining expert computation with quantization
- Fused RMSNorm/LayerNorm/SiLU/GELU

**Reported**: 29-69% inter-token-latency reduction vs compiler backends; 28-30% for long-context; 13-17% for parallel generation.

**Adoptability**: The fused RMSNorm/activation pattern is directly applicable. FlashInfer's JIT template approach is the most adaptable but requires substantial CUDA engineering.

### 1.3 CUDA Graphs — Eliminating Launch Overhead

**How vLLM uses them**: `vllm/v1/worker/gpu_model_runner.py` — during initialization, capture the full model forward pass as a CUDA graph at each of several pre-defined batch sizes (powers of 2: 1, 2, 4, 8, ... up to max). At decode time: select the graph matching the current batch size, replay it. No host-side kernel launch overhead.

A typical Transformer layer has 15-30 kernels (attention QKV proj, attention compute, attention out proj, RMSNorm, gate proj, up proj, SiLU, down proj). At ~10μs each, that's 150-300μs of launch overhead per layer. CUDA Graphs compress this to near-zero.

**Tritium status**: Already implemented — `_g` variants in `decode.cu` with device control block for per-token values. Extension needed for batched decode path.

### 1.4 BitNet.cpp GPU Kernels — Register-Only DP4A

**Repo**: [github.com/microsoft/BitNet](https://github.com/microsoft/BitNet) (MIT) — `gpu/bitnet_kernels/bitnet_kernels.cu`

Format: W2A8 (2-bit weights, 8-bit activations). Every 16 two-bit values packed into one int32 with interleaving pattern `[0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15]` for efficient 4-at-a-time extraction.

Key design:
- Weight matrix divided into 16×32 blocks
- Block dims: `dim3(8, 16, 1)` = 128 threads
- **No shared memory** — entirely register-resident
- Weight unpacking via PTX `lop3.b32` ternary logic + `__vsubss4` to map unsigned 2-bit codes to signed ternary values
- Compute: `__dp4a` (int8 × int8 dot product, 4-wide). Inner loop: 4 DP4A instructions per 16 elements
- Reduction: `__shfl_down_sync` butterfly across K_block_size threads

**Benchmarks** (A100 40GB): For shape 13824×2560, W2A8 takes 18.75μs vs BF16 at 59.51μs (**3.17×**). End-to-end: BitNet-b1.58-2B-4T vs Gemma-2-2B BF16 via vLLM, 64-token output: 57.40ms vs 187.64ms (**3.27×**).

**Adoptability**: High. The key upgrade for Tritium would be switching to the register-only (no shared memory) DP4A + shuffle-down pattern.

### 1.5 Bitwise Ternary Dot Product

A pure bitwise approach: pack 32 ternary values into 64 bits (2 bits each), use AND/XOR masks to identify +1 and -1 positions, POPCOUNT to count, then compute `sum_positive - sum_negative`. This eliminates the int8 unpack step entirely.

**Expected**: 1.5-2× over the DP4A approach. No existing open-source implementation — research gap.

---

## 2. LUT-Based Ternary GEMM

### 2.1 T-MAC: LUT Methodology

**Paper**: arXiv:2407.00088 — [github.com/microsoft/T-MAC](https://github.com/microsoft/T-MAC) (MIT)

Core idea: instead of dispatching add/sub/skip per weight, precompute a lookup table of 3^μ partial sums and index into it.

For a group of μ trits:
- 3^μ LUT entries
- Each entry = `w0*a[k] + w1*a[k+1] + ... + w_{μ-1}*a[k+μ-1]`
- Code extraction: `code = (w0+1)*3^(μ-1) + (w1+1)*3^(μ-2) + ... + (w_{μ-1}+1)`
- One LUT lookup replaces μ extract+branch+accumulate sequences

**Sign-bit optimization**: Halve the table by separating sign. For ternary, 3^μ → 3^(μ-1) × 2 entries.

### 2.2 Concrete LUT Kernel Sketch for CUDA

```cuda
// Group of 3 trits: 3^3 = 27 LUT entries
// code = (w0+1)*9 + (w1+1)*3 + (w2+1) in [0, 26]
// LUT[code] = w0*a[k] + w1*a[k+1] + w2*a[k+2]

__global__ void tq2_0_lut_gemm(const float* act, const uint8_t* weights,
                                const float* scales, float* out,
                                int M, int N, int K, int row_bytes) {
    extern __shared__ float s_lut[27];

    for (int ki = lane; ki < K; ki += 3*WARP_SIZE) {
        // Build 3^3 LUT from 3 activation values
        float a0 = act[mi * K + ki];
        float a1 = act[mi * K + ki + 1];
        float a2 = act[mi * K + ki + 2];
        // LUT[0] = -a0-a1-a2, LUT[1] = -a0-a1, ..., LUT[26] = a0+a1+a2

        // Extract 3-trit code from packed weights
        uint8_t code = extract_3_trits(weights, ni, ki);

        // Single table lookup replaces 3 branches
        acc += lut[code];
    }
}
```

### 2.3 BitNet LUT Kernels (x86/ARM)

**Repo**: microsoft/BitNet `preset_kernels/` directory (MIT)

- `bitnet-lut-kernels-tl2.h` — x86 AVX2 PSHUFB LUT (TL2, groups of 3 trits)
- `bitnet-lut-kernels-tl1.h` — ARM NEON TBL LUT (TL1, groups of 2 trits)
- LUT construction logic: `three_lut_ctor` — directly portable to CUDA shared memory

**Tritium applicability**: The LUT construction logic is directly portable. For CUDA, shared memory (48-164 KB) easily holds 3^4 = 81 f32 entries (324 bytes) or 3^3 = 27 entries (108 bytes). TQ2_0 packs 4 trits per byte, so a group of 4 trits = 1 byte index extraction.

**Expected**: 10-30% over the current branchless `acc += a * (float)((int)code - 1)` pattern.

---

## 3. Tequila & Sherry: STE Modifications

### 3.1 Tequila — Reactivating Dead Weights

**Paper**: arXiv:2509.23809

**Problem**: The standard STE has a deadzone — weights where `|Wf/s_q| > 1` get zero gradient. These "saturated" weights are trapped: they can never move back toward zero, permanently reducing model capacity.

**Solution**: Add a direct gradient path for deadzone weights:

```
dL/dWf += lambda * dL/dY * X^T  (for all deadzone weights)
```

Where `lambda` is a fixed hyperparameter (default 1e-3, robust across {1e-5, 1e-1}).

**Implementation in Tritium's `ste.rs`**: In `quantize_vjp`, after computing the standard STE gradient mask, add:

```rust
for i in 0..len {
    if wf_abs[i] >= s_q {  // deadzone
        g_wf[i] += lambda * grad_out[i] / s_q;
    }
}
```

**Impact**: +2-4% accuracy on downstream tasks (Tequila shows +2.6% average on 1B model). Prevents the deadzone trapping failure mode.

### 3.2 Sherry — Structured Ternary Sparsity (3:4)

**Paper**: arXiv:2601.07892

**Key idea**: 3:4 structured sparsity (1.25 bits/weight) — only 3 out of every 4 weights are non-zero.

**Arenas module**: Prevents gradient homogenization that causes weight trapping under structured sparsity. During QAT, augment the ternary forward with a decaying full-precision residual:

```
Y = X * Q(W) * alpha + lambda_t * X * W
```

Where `lambda_t` anneals from 1 to 0 via cosine decay.

**Impact**: Without Arenas, effective rank (ER) of gradient matrices drops below 750/4096, causing weight polarization. Arenas maintains gradient diversity throughout training.

### 3.3 Cross-Paper Comparison

| Aspect | Tequila | Sherry | BitNet b1.58 | ParetoQ |
|--------|---------|--------|-------------|---------|
| STE modification | λ·dL/dY bias for deadzone | Arenas residual λ_t·X·W | Standard STE | Standard STE + learnable α |
| Lambda scheduling | Fixed (1e-3) | Cosine decay 1→0 | N/A | N/A |
| Quantization | Absmean + deadzone bias | Sparse-AbsMean 3:4 | Absmean | SEQ (stretched elastic) |
| LR | 1e-4 fixed | 1e-4 fixed | ~2e-2 cosine | 2e-5 cosine (QAT) |
| Training tokens | 10B | 10B | 100B-4T | 100B total (90/10 split) |

---

## 4. Speculative Decoding with Ternary Drafts

### 4.1 Classic Speculative Decoding

**Papers**: Leviathan et al. (2023), Chen et al. (2023)

Draft model generates K tokens, target model verifies in one forward pass. Speedup = `K / (1 + K * draft_latency / target_latency)`.

For a ternary draft (10× faster than FP16): at K=5, speedup ≈ 5 / (1 + 5/10) = **3.33×**. Acceptance rate depends on draft quality — a well-distilled ternary draft at the same parameter count as the target can achieve α ≈ 0.7-0.8.

### 4.2 EAGLE / EAGLE-2

**Paper**: arXiv:2401.15077

Autoregressive draft using feature-level speculation. The draft head operates on the target model's hidden states (not token IDs), achieving higher acceptance rates.

**EAGLE**: 2.7-3.5× speedup, acceptance rate ~0.8, avg accept length 3.2-4.5.
**EAGLE-2**: 3.05-4.26× speedup with dynamic draft trees.

**Ternary application**: A ternary EAGLE-style draft head would be 10× smaller than FP16, enabling a much larger draft model in the same memory budget. No published work on this — **research gap**.

### 4.3 Medusa

**Paper**: arXiv:2401.10774

Multi-head prediction without a separate draft model. Medusa adds extra prediction heads to the target model.

**Medusa-1**: 2.18-2.33× speedup (frozen backbone).
**Medusa-2**: 2.83-3.62× speedup (joint training).

**Ternary application**: Ternary Medusa heads on top of a FP16 backbone — minimal parameter overhead, 2.5-3× expected speedup.

### 4.4 Draft-OPD — On-Policy Distillation

**Paper**: arXiv:2605.29343

Distills a draft model specifically for speculation (optimize acceptance rate, not standalone perplexity). Uses LK-lambda loss for direct acceptance rate optimization.

**Results**: 5×+ speedup, avg accept length 5.85-6.33.

**Ternary application**: The ideal training recipe for a ternary draft model — distill from FP16 teacher, optimize for acceptance rate.

### 4.5 Self-Speculative Decoding (DEL)

**Paper**: DEL (Dynamic Exit Layers)

Uses early exits or shallow layers of the same model as draft. No auxiliary model needed.

**Results**: 2.16-2.84× speedup.

**Ternary application**: Make early layers ternary, late layers FP16. Use DEL-style dynamic exit for self-speculative drafting. Expected: 2-3× speedup with zero additional model overhead.

### 4.6 Research Gap: Ternary Draft Models

There is **no published paper** specifically using a 1.58-bit/ternary model as a draft model for speculative decoding. The theoretical case is strong:
- 10× compression enables much larger draft models in the same memory
- BitNet matches FP16 perplexity at the same scale
- Ternary compute is 3-10× faster
- The combination should yield both higher acceptance rates AND lower draft latency

**Tritium opportunity**: First-mover on ternary speculative decoding.

### 4.7 Speedup Summary

| Method | Speedup | Acceptance Rate | Key Advantage |
|--------|---------|----------------|---------------|
| Classic SD (small FP16 draft) | 1.5-2.5× | 0.52-0.79 | Simple |
| EAGLE | 2.7-3.5× | ~0.8 | Feature-level, lossless |
| EAGLE-2 | 3.05-4.26× | ~0.85 | Dynamic draft trees |
| Medusa-1 | 2.18-2.33× | N/A | No draft model needed |
| Medusa-2 | 2.83-3.62× | N/A | Joint training |
| Draft-OPD (distilled draft) | 5×+ | High | Optimizes acceptance directly |
| DEL (self-speculative) | 2.16-2.84× | Varies | Single model |
| **Ternary EAGLE (projected)** | **3-4×** | **~0.8** | **10× smaller draft** |

---

## 5. Training Recipes for Ternary LLMs

### 5.1 Microsoft's Two-Stage LR Schedule

**Source**: BitNet training tips document, Table 2

The two-stage schedule is "critical" for ternary convergence:

| Parameter | Stage 1 | Stage 2 |
|-----------|---------|---------|
| Peak LR | 1.2e-3 | 8e-4 |
| Warmup | 375 steps | 375 steps |
| Schedule | Linear decay | Linear decay |
| Weight decay | 0.1 | **0.0** |

Weight decay must drop to 0 in stage 2 because "a large weight decay leads to lower confidence scores for the 1-bit weights, causing them to change more frequently."

### 5.2 ParetoQ: 90% FP + 10% QAT

**Paper**: arXiv:2502.02631

Optimal recipe: pretrain 90% of tokens in FP16, then switch to ternary QAT for the final 10%.

| Phase | GPUs | Batch | LR | Schedule |
|-------|------|-------|-----|----------|
| FP Pretrain | 8×8 | 16 | 2.5e-3 | Linear decay |
| QAT Fine-tune | 8×8 | 8 | 1e-4 | Linear decay |

**Key result**: ParetoQ's ternary 600M model achieves 58.7 accuracy, surpassing the "1-bit era" ternary 3B model with only 1/5 parameters. On LLaMA-3 8B, ternary ParetoQ uses "only 30% of the training tokens" versus the baseline while reducing the FP gap by 37.8%.

### 5.3 BitNet b1.58 Training Recipe

**Paper**: arXiv:2402.17764, arXiv:2504.12285 (2B4T model)

| Parameter | Value |
|-----------|-------|
| Activation | Squared ReLU (not SiLU/GELU) |
| Bias | None |
| Norm | RMSNorm before attention |
| Optimizer | AdamW, betas=(0.9, 0.95) |
| Optimizer state | FP32 (m and v) |
| Latent weights | FP32 |
| Activation precision | INT8 (per-token symmetric) |
| STE | `w + (quant(w) - w).detach()` |
| Training tokens | 100B-4T (scale dependent) |
| Scaling equivalence | 13B ternary > 3B FP16; 30B ternary > 7B FP16 |

### 5.4 LoRA for Ternary

**No published work** on LoRA fine-tuning of frozen ternary bases. This is a research gap Tritium could fill.

Estimated parameters:
- LoRA rank 16-64 for frozen ternary base
- QLoRA-style: FP16 LoRA adapters on frozen ternary backbone
- Gradient checkpointing required for 24 GB VRAM

### 5.5 Recommended Tritium Training Recipe

| Hyperparameter | Value | Source |
|---|---|---|
| Optimizer | AdamW, betas=(0.9, 0.95), eps=1e-8 | Microsoft |
| Peak LR (Stage 1) | 1.2e-3 (for 1-3B models) | Microsoft |
| Peak LR (Stage 2) | 8e-4 (midway decay) | Microsoft |
| Weight decay (Stage 1) | 0.1 | Microsoft |
| Weight decay (Stage 2) | 0.0 (disabled) | Microsoft |
| Warmup | 375 steps | Microsoft |
| Schedule | Two-stage linear decay | Microsoft |
| Batch size | 256K-1M tokens | Microsoft |
| Sequence length | 2048 | Microsoft |
| Activation precision | INT8 (per-token symmetric) | Microsoft |
| Weight precision | Ternary {-1,0,1} via absmean | Microsoft |
| STE + Tequila | Add λ·dL/dY deadzone bias, λ=1e-3 | Tequila |
| STE + Sherry | Arenas residual with cosine annealing | Sherry |
| LoRA rank | 16-64 for frozen ternary base | Estimate |
| Gradient checkpointing | Required for 24 GB VRAM | Estimate |

---

## 6. Open-Source Ecosystem Survey

### 6.1 microsoft/BitNet

**Repo**: [github.com/microsoft/BitNet](https://github.com/microsoft/BitNet) — 15k+ stars, MIT license

The reference implementation. Key components:
- `gpu/bitnet_kernels/` — DP4A + lop3.b32 GPU kernels (May 2025, very new)
- `preset_kernels/` — LUT kernels for x86 (AVX2 PSHUFB) and ARM (NEON TBL)
- `include/ggml-bitnet.h` — API: `ggml_qgemm_lut()`, `ggml_preprocessor()`
- Training code: PyTorch-based, Absmean quantization, standard STE

**What to learn**: Register-only GEMV pattern, LUT construction logic, weight packing format.

### 6.2 llama.cpp / GGML

**Repo**: [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — 82k+ stars, MIT license

Ternary data types:
- `tq1_0` — 5 ternary vals/byte (3^5=243, fits in uint8 with 12 unused codes)
- `tq2_0` — 4 trits/byte (3^4=81, fits in uint8 with 185 unused codes)
- `Q2_K` — 2-bit block format (not ternary, but similar bit-packing)

**What to learn**: Bit-packing format, SIMD dispatch for dequantization, block quantization structure.

### 6.3 vLLM

**Repo**: [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) — 52k+ stars, Apache-2.0

Zero ternary support. But the infrastructure is valuable:
- PagedAttention for KV cache management
- Continuous batching
- CUDA Graphs integration
- Custom backend plugin interface

**Tritium opportunity**: Build a vLLM plugin backend using Tritium's ternary CUDA kernels.

### 6.4 TensorRT-LLM

**Repo**: [github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — 12k+ stars, Apache-2.0

Zero ternary support. Optimized for INT4/INT8. The kernel patterns (tile sizes, shared memory usage) are informative but not directly adoptable for ternary.

### 6.5 DeepSpeed

**Repo**: [github.com/microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) — 37k+ stars, Apache-2.0

1-bit gradient compression (not weight quantization). The distributed training infrastructure (ZeRO-3, FSDP) is relevant for ternary QAT at scale.

**Tritium status**: Already has NCCL groundwork for distributed training.

### 6.6 ONNX

**Repo**: [github.com/onnx/onnx](https://github.com/onnx/onnx) — 21k+ stars, MIT license

ONNX 1.23.0 includes native 2-bit integer types and QDQ (QuantizeLinear/DequantizeLinear) operators. The `ternary-bonsai-webgpu` project proves the pipeline works: ternary weights → ONNX export → ONNX Runtime Web → WebGPU execution.

**Tritium opportunity**: ONNX export for edge deployment (browser, mobile, ARM).

### 6.7 Ecosystem Position Matrix

| Capability | microsoft/BitNet | llama.cpp | vLLM | TensorRT-LLM | DeepSpeed | **Tritium** |
|------------|-----------------|-----------|------|--------------|-----------|-------------|
| Ternary CPU kernels | Yes (LUT) | Yes (tq1_0/tq2_0) | No | No | No | Reference |
| Ternary GPU kernels | New (May 2025) | No native | No | No | No | **Primary** |
| Ternary QAT training | No | No | No | No | No | **Gap** |
| Distributed ternary | No | No | No | No | 1-bit grad | NCCL ground |
| ONNX export | No | GGUF only | No | No | No | **Gap** |
| Serving framework | Basic server | Full inference | No | No | No | **Gap** |

**Tritium's unique position**: The only project targeting GPU-native ternary kernels as a first-class concern.

---

## 7. Tritium Action Plan

### Tier 1 — This Week (high impact, low complexity)

| # | Task | Where | Impact | Effort | Source |
|---|------|-------|--------|--------|--------|
| 1.1 | Two-stage LR schedule | `tritium-train/src/lr.rs` | Training quality | 2-4h | Microsoft Table 2 |
| 1.2 | Tequila deadzone bias in STE | `tritium-train/src/ops/ste.rs` | +2-4% accuracy | 3-5h | Tequila Eq 7-8 |
| 1.3 | Disable weight decay in stage 2 | `tritium-train/src/optim.rs` | Training stability | 1-2h | Microsoft tips |

### Tier 2 — This Month (high impact, medium complexity)

| # | Task | Where | Impact | Effort | Source |
|---|------|-------|--------|--------|--------|
| 2.1 | LUT-based ternary GEMV kernel | `tq2_0_lut.cu` (new) | 10-30% decode | 3-5d | T-MAC, BitNet LUT |
| 2.2 | Register-only DP4A GEMV | `tq2_0_add.cu` (new variant) | 3-6× vs FP16 | 5-7d | BitNet.cpp |
| 2.3 | Sherry Arenas in STE | `tritium-train/src/ops/ste.rs` | Training quality | 2-3d | Sherry Eq 7-8 |
| 2.4 | CUDA Graph batched decode | `tritium-cuda/src/cuda.rs` | 10-30% latency | 2-3d | vLLM pattern |

### Tier 3 — Next Quarter (exploratory, high complexity)

| # | Task | Where | Impact | Effort | Source |
|---|------|-------|--------|--------|--------|
| 3.1 | Fused attention-to-FFN kernel | `decode.cu` (new) | 15-25% overall | 1-2w | FlashInfer |
| 3.2 | Speculative decoding (ternary draft) | `tritium-spec` (new) | 3-4× throughput | 2-4w | EAGLE + Draft-OPD |
| 3.3 | Bitwise ternary dot product | `tq2_0_add.cu` (new variant) | 1.5-2× over DP4A | 1-2w | Research gap |
| 3.4 | vLLM plugin backend | `tritium-vllm` (new) | Ecosystem | 2-3w | vLLM plugin API |
| 3.5 | ONNX export | `tritium-onnx` (new) | Edge deployment | 1-2w | ONNX QDQ pattern |
| 3.6 | Int8 activation in decode path | `decode.cu` | 2× activation BW | 3-5d | Existing kernels |

### Adoptable Open-Source Code

| Source | License | What to adopt |
|--------|---------|---------------|
| microsoft/BitNet `gpu/` | MIT | DP4A + lop3.b32 weight unpacking, register-only GEMV |
| microsoft/BitNet `preset_kernels/` | MIT | LUT construction logic (`three_lut_ctor`), sign-bit optimization |
| microsoft/T-MAC | MIT | LUT methodology, group-of-N ternary indexing |
| FlashInfer | Apache-2.0 | JIT-compiled attention templates, fused RMSNorm/activation |
| llama.cpp `ggml-quants.c` | MIT | tq1_0/tq2_0 packing format reference |

### Key Insight

**Ternary inference is memory-bound, not compute-bound.** The current add/sub/skip dispatch is already near-optimal for compute. The real wins come from:

1. **Reducing memory traffic** — LUT eliminates unpack overhead, INT8 activations halve activation bandwidth, sparse-aware kernels skip zero blocks
2. **Eliminating launch overhead** — CUDA Graphs
3. **Fusing adjacent kernels** — residual-in-registers avoids HBM round-trips
4. **Using integer arithmetic** — DP4A instead of FP multiply for the ternary dot product

For training, the dominant failure modes are deadzone trapping (Tequila fix) and gradient homogenization (Sherry Arenas fix), both addressable with small STE modifications. The two-stage LR schedule is the single highest-impact training change.

---

## References

- BitNet.cpp: [github.com/microsoft/BitNet](https://github.com/microsoft/BitNet) (MIT)
- FlashDecoding: [pytorch.org/blog/flash-decoding](https://pytorch.org/blog/flash-decoding/)
- FlashInfer: [arxiv.org/abs/2501.01005](https://arxiv.org/abs/2501.01005), [github.com/flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) (Apache-2.0)
- T-MAC: [arxiv.org/abs/2407.00088](https://arxiv.org/abs/2407.00088) (MIT)
- Tequila: [arxiv.org/abs/2509.23809](https://arxiv.org/abs/2509.23809)
- Sherry: [arxiv.org/abs/2601.07892](https://arxiv.org/abs/2601.07892)
- BitNet b1.58: [arxiv.org/abs/2402.17764](https://arxiv.org/abs/2402.17764)
- BitNet 2B4T: [arxiv.org/abs/2504.12285](https://arxiv.org/abs/2504.12285)
- ParetoQ: [arxiv.org/abs/2502.02631](https://arxiv.org/abs/2502.02631)
- EAGLE: [arxiv.org/abs/2401.15077](https://arxiv.org/abs/2401.15077)
- Draft-OPD: [arxiv.org/abs/2605.29343](https://arxiv.org/abs/2605.29343)
- LK Losses: [arxiv.org/abs/2602.23881](https://arxiv.org/abs/2602.23881)
- Medusa: [arxiv.org/abs/2401.10774](https://arxiv.org/abs/2401.10774)
- LLM.int8(): [github.com/TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes) (MIT)
- vLLM: [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) (Apache-2.0)
- llama.cpp: [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT)
- ONNX: [github.com/onnx/onnx](https://github.com/onnx/onnx) (MIT)
