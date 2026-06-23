# ADR 0016 — Ternary Training Methods: CPU–GPU Hybrid and Beyond

- **Status:** Proposed
- **Date:** 2026-06-22
- **Relates:** builds on [ADR 0007](./0007-v050-training-core.md) (v0.50 Training Core), [ADR 0008](./0008-v060-pretraining-distributed.md) (v0.60 Distributed); informs the post-v0.60 training architecture
- **Context:** Tritium's training stack (v0.50) implements standard STE autograd + AdamW on a single GPU. The v0.60 distributed stack adds FSDP/DDP. This ADR examines whether ternary models have fundamentally different training economics that unlock novel architectures.

---

## 1. The insight: ternary breaks the fp16 training assumption

Every mainstream training framework (PyTorch, DeepSpeed, Megatron) assumes the same model shape:

| Component | fp16 model (7B) | Ternary model (7B) |
|-----------|----------------|-------------------|
| **Weights** | 14 GB (FP16) | **1.37 GB** (TQ2_0, 1.58 bits/param) |
| **Gradients** | 14 GB (FP16/FP32) | 14 GB (FP32 — STE requires full precision) |
| **Optimizer states** (Adam) | 28 GB (FP32 m + v) | 28 GB (FP32 m + v) |
| **Activations** (batch-dependent) | ~10-50 GB | ~10-50 GB |
| **Total VRAM** | ~70-110 GB | ~55-95 GB |

The optimizer states and gradients are **20× larger than the ternary weights.** In fp16 training, everything is the same order of magnitude, so keeping it all on GPU makes sense. In ternary training, the weight update is a tiny fraction of the memory footprint — but the optimizer step still consumes the majority of the compute time because it touches every parameter twice (m and v updates).

This asymmetry creates an opportunity that doesn't exist in fp16 training.

## 2. Why CPU isn't used in fp16 training (and why ternary is different)

### The fp16 case: CPU offloading is always worse

For fp16 training, the optimizer step is:

```
m = β1 * m + (1 - β1) * g       ← elementwise, 14 GB
v = β2 * v + (1 - β2) * g²      ← elementwise, 14 GB
θ = θ - lr * m / (√v + ε)       ← elementwise, 14 GB
```

If this runs on CPU:
- **PCIe transfer:** 14 GB gradients CPU←GPU (~0.4s at 32 GB/s) + 14 GB params CPU→GPU (~0.4s) = **0.8 seconds**
- **GPU compute:** same step on GPU = **~2-5 ms** (memory-bandwidth-bound, ~1 TB/s)
- **Verdict:** CPU offloading is **160-400× slower** due to PCIe

This is why ZeRO-Offload (DeepSpeed) is a desperation move for VRAM overflow, not an optimization. The PCIe wall makes CPU participation in the math a net loss.

### The ternary case: the math changes

For ternary training, the weight update has a unique structure:

```
# Standard Adam step (same as fp16 — runs on GPU or CPU)
m = β1 * m + (1 - β1) * g
v = β2 * v + (1 - β2) * g²
θ_fp32 = θ_fp32 - lr * m / (√v + ε)

# Ternary-specific: re-quantize weights (trivial on CPU, wasteful on GPU)
θ_ternary = sign(θ_fp32)   # {-1, 0, +1} via STE rounding
```

The critical difference: **ternary weights are 10× smaller than fp16 weights.** The data transfer calculation:

| Transfer | Size | PCIe 5.0 time | NVLink time |
|----------|------|---------------|-------------|
| Gradients GPU→CPU | 14 GB | 0.44s | 0.07s |
| Ternary weights CPU→GPU | 1.37 GB | 0.04s | 0.007s |
| **Total round-trip** | 15.37 GB | **0.48s** | 0.08s |

Compare to the GPU-only optimizer step time (~2-5 ms for elementwise ops at 1 TB/s). The PCIe overhead is still 100× the compute — **but only for the current batch size.**

The key realization: **you don't need to transfer every step.** If the CPU owns the optimizer states permanently, the per-step transfer is only:
- **GPU→CPU:** gradients (14 GB) — unavoidable, needed for the update
- **CPU→GPU:** ternary weights (1.37 GB) — the re-quantized result

But even this is too much at 32 GB/s. The real win requires a different architecture.

## 3. The architecture: persistent CPU optimizer with gradient compression

### Design: CPU owns the full-precision shadow weights

```
┌─────────────────────────────────────────────────────────────┐
│  CPU (persistent)                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ θ_fp32       │  │ Adam state   │  │ θ_ternary        │  │
│  │ (shadow)     │  │ (m, v)       │  │ (quantized)      │  │
│  │ 14 GB        │  │ 28 GB        │  │ 1.37 GB          │  │
│  └──────┬───────┘  └──────┬───────┘  └────────▲─────────┘  │
│         │                 │                    │            │
│         └───── Adam step ─┘                    │            │
│                   │                            │            │
│                   └── re-quantize ─────────────┘            │
└─────────────────────────────────────────────────────────────┘
                          │                    ▲
                    gradients             ternary weights
                    (compressed)          (1.37 GB)
                          │                    │
┌─────────────────────────┼────────────────────┼──────────────┐
│  GPU                    ▼                    │              │
│  ┌──────────────────────────────────────────┐│              │
│  │ Forward + Backward                       ││              │
│  │ (ternary GEMM + STE backward)            ││              │
│  │ Weights: 1.37 GB resident on GPU         ││              │
│  └──────────────────────────────────────────┘│              │
└──────────────────────────────────────────────┘              │
```

**What stays on GPU permanently:** ternary weights (1.37 GB), activations, KV cache
**What stays on CPU permanently:** full-precision shadow weights (14 GB), Adam states (28 GB)
**What crosses PCIe each step:** gradients (14 GB GPU→CPU), ternary weights (1.37 GB CPU→GPU)

**The bottleneck is still the gradient transfer.** 14 GB at 32 GB/s = 0.44 seconds per step. At 1000 steps/second (typical for small models), this is 440× too slow.

### The missing piece: gradient compression

The gradients don't need to be FP32 to cross PCIe. Ternary-specific gradient compression:

| Method | Compression | Gradient quality | PCIe time (7B) |
|--------|-------------|------------------|-----------------|
| FP32 (baseline) | 1× | Perfect | 0.44s |
| FP16 | 2× | Near-perfect | 0.22s |
| Top-k sparsification (1%) | 100× | Good (dense updates over time) | 4.4ms |
| Ternary gradient (STE) | 16× | Acceptable (already quantized) | 27.5ms |
| Low-rank gradient (rank-64) | ~100× | Good for fine-tuning | 4.4ms |

**Top-k sparsification** is the most promising: send only the top 1% of gradient values by magnitude. Over many steps, the full gradient is covered. This is the basis of **Deep Gradient Compression** (Lin et al., 2018) and works well in practice.

With 1% sparsification + index encoding: ~140 MB/step → **4.4 ms at 32 GB/s.** Now PCIe is not the bottleneck.

## 4. What this unlocks

### 4.1 Training on consumer hardware

A ternary 7B model with CPU-side optimizer:
- **GPU VRAM:** 1.37 GB (weights) + ~5 GB (activations, batch=1) = **~6.5 GB**
- **CPU RAM:** 14 GB (shadow) + 28 GB (Adam) = **42 GB**

This fits on a **4090 (24 GB VRAM) + 64 GB DDR5 system.** No A100, no multi-GPU, no cloud. Consumer hardware training at 7B scale.

For comparison, fp16 training of a 7B model needs ~80 GB VRAM (A100/H100) or ZeRO-Offload (3-5× slower).

### 4.2 Asynchronous pipeline parallelism

The CPU optimizer step and GPU forward/backward can overlap:

```
Step N:   [GPU: forward+backward] ──────┐
                                         │
Step N:   [CPU: Adam + re-quant] ────────┤ (async, overlapped)
                                         │
Step N+1: [GPU: forward+backward] ───────┘
```

If the CPU step finishes before the GPU step (likely — elementwise ops are fast on CPU), the PCIe transfer is hidden entirely. The training throughput is limited only by the GPU forward/backward speed.

### 4.3 Ternary-native optimizer innovations

The ternary weight space {-1, 0, +1} has structure that fp16 doesn't:

**Discrete optimization.** The weight update is a transition between three states. This is a **combinatorial optimization** problem, not a continuous one. Possible approaches:
- **Bandit-style updates:** treat each weight as a 3-armed bandit, use UCB/Thompson sampling
- **Evolutionary strategies:** perturb a subset of weights, keep changes that reduce loss
- **SignSGD on steroids:** the gradient sign is already the natural update direction for ternary

**Structured sparsity.** Ternary weights have natural zero-states. Training can exploit this:
- **Zero-bias regularization:** encourage weights toward zero (already happens with STE + Adam)
- **Dynamic sparsity:** skip zero weights in both forward and backward (kernel-level optimization)
- **Block-level sparsity:** entire 256-trit blocks can be zero — skip at the block level

**Mixed-precision gradients.** Gradients don't need to be FP32 everywhere:
- Attention gradients: FP32 (small, precision-sensitive)
- MLP gradients: FP16 or even INT8 (large, less precision-sensitive)
- Embedding gradients: FP32 (vocabulary-sensitive)
- Norm gradients: FP32 (1D, tiny)

### 4.4 Distributed training with compressed gradients

In multi-GPU training, the all-reduce of gradients is the communication bottleneck. Ternary training can compress gradients before all-reduce:

| Method | Communication volume (7B) | Quality |
|--------|--------------------------|---------|
| FP32 all-reduce (baseline) | 14 GB × (N-1)/N | Perfect |
| FP16 all-reduce | 7 GB × (N-1)/N | Near-perfect |
| Top-k sparsification | 140 MB × (N-1)/N | Good |
| Ternary gradient | 875 MB × (N-1)/N | Acceptable |

With top-k sparsification, a 2-GPU all-reduce takes ~4.4 ms instead of ~0.44s. This makes multi-node training practical even over commodity Ethernet (10 Gbps) rather than requiring InfiniBand.

## 5. Research directions

### 5.1 SALT-aware training: SALT as the STE estimator

The central insight: **SALT is not just a quantization format — it is a structured gradient estimator for ternary training.**

Standard STE says: "pretend `quantize(x) = x` in the backward pass." This is a biased hack — the gradient ignores the quantization boundary entirely. A weight of 0.49 and 0.51 get nearly identical gradients, even though they land on opposite sides of a ternary decision boundary.

SALT says something different: "represent the weight as a sum of ternary planes with fp16 residuals, let the gradient flow through the reconstruction." The residual structure *is* the straight-through path — but built from the actual ternary geometry, not an identity handwave.

```
STE:        ∂L/∂x ≈ ∂L/∂quantize(x)     (identity — ignores quantization)
SALT:       ∂L/∂x ≈ ∂L/∂(plane_0 + plane_1 + plane_2 + ...)  (structured residual)
```

**Why SALT is a better gradient estimator than STE:**

| Property | STE | SALT-aware |
|----------|-----|-----------|
| Gradient bias | High (identity through discontinuity) | Low (residual reconstruction) |
| Tunable fidelity | No (binary: identity or nothing) | Yes (more planes = better approximation) |
| Structure | None (ignores ternary geometry) | Planes mirror the actual decomposition |
| Convergence | Oscillates near decision boundaries | Planes absorb residual, smooths landscape |

**The training loop:**

```
for each step:
    # Forward: use SALT reconstruction as the weight
    W = plane_0 + plane_1 + plane_2 + ...     (fp16 residuals, differentiable)
    y = x @ W.T                                 (standard matmul)

    # Backward: gradient flows through the planes (no STE needed)
    ∂L/∂plane_i = ∂L/∂W × ∂W/∂plane_i         (standard chain rule)

    # Update: Adam on each plane independently
    plane_i -= lr × adam(∂L/∂plane_i)

    # (Optional) Periodic re-quantize planes to ternary to prevent drift
```

**Why this works:** Each plane is fp16 — the gradient is clean, unbiased, and standard. But the planes are trained *as a decomposition into ternary components.* The optimizer learns to put information where the planes can capture it. At inference time, each plane is independently quantized to ternary — and because the planes were trained as a decomposition, the quantization error from one plane is compensated by the others.

**The key knob:** number of planes. More planes = better gradient approximation = closer to fp16 accuracy. Fewer planes = more ternary = faster inference. This is a smooth tradeoff, not a binary choice.

**Comparison to alternatives:**

| Approach | Gradient quality | Training complexity | Inference efficiency |
|----------|-----------------|-------------------|---------------------|
| fp16 + PTQ | Perfect (fp16) | None | Ternary (with accuracy loss) |
| fp16 + STE QAT | Biased (identity hack) | STE autograd | Ternary (recovers accuracy) |
| **SALT-aware** | **Structured (residual)** | **Standard backprop** | **Ternary (SALT quantize)** |
| Gumbel-Softmax | Smooth but complex | Temperature scheduling | Needs re-quantization |
| Finite-difference | Unbiased but expensive | 3× forward passes | Ternary |

**Tritium's advantage:** SALT is already implemented (`tritium-format`). The plane decomposition, packing, and inference path all exist. The training experiment is: wire the SALT planes as the weight representation in the training loop, train with standard Adam, SALT-quantize at the end.

### 5.2 Ternary-specific pretraining

Current ternary models are trained fp16 then quantized (post-training) or fine-tuned with QAT. Can we pretrain directly in ternary?

**Arguments for:**
- The weight space is tiny — search is easier
- No fp16→ternary quantization loss
- Natural sparsity emerges from training (not imposed post-hoc)

**Arguments against:**
- STE bias accumulates over millions of steps
- The discrete weight space may have poor gradient signal
- fp16 pretraining has decades of optimization behind it

**With SALT-aware training, the question transforms:** instead of "pretrain in ternary vs fp16," it's "how many SALT planes do you need from scratch?" A 1-plane SALT model is b1.58 (ternary). A 4-plane SALT model is near-fp16. The training starts with many planes (easy optimization) and can prune to fewer planes (faster inference) — a smooth curriculum from fp16-like to ternary.

### 5.3 Hardware-aware training kernels

The ternary backward pass has unique properties:
- **Gradient computation:** FP32 (standard)
- **Weight update:** FP32 shadow weights → ternary quantization
- **The quantization step is trivial on CPU** (sign function, ~1 ns/param)

If the CPU owns the optimizer, the GPU kernel can be simplified:
- No Adam state on GPU (saves 28 GB VRAM)
- No FP32 shadow weights on GPU (saves 14 GB VRAM)
- GPU only needs: ternary weights + activations + gradients
- **Result: 3-4× less VRAM per model**

This enables training models 3-4× larger on the same hardware, or training with 3-4× larger batch sizes.

## 6. Proposed experiments

### Experiment 1: SALT-aware training (primary)

**Hypothesis:** A model trained with SALT-plane decomposition (fp16 residuals) achieves the same or better accuracy as fp16 + STE QAT, with no STE in the training loop.

**Experiment design:**

1. **Baseline (upper bound):** Train 125M in fp16, SALT-quantize to 3 planes (PTQ). Expected: 5-15% accuracy loss vs fp16.
2. **Treatment A (SALT-aware):** Train 125M with 3 SALT planes as the weight representation (fp16 residuals, standard Adam). SALT-quantize planes at end. Expected: ≤2% accuracy loss vs fp16.
3. **Treatment B (STE QAT):** Train 125M with fp16 + STE fine-tune (current approach). Expected: ≤3% accuracy loss vs fp16.
4. **Treatment C (plane count sweep):** Train with 1, 2, 3, 4 planes. Map the accuracy-vs-planes curve.

**Metrics:** loss curve, final accuracy vs fp16 baseline, wall-clock time, gradient variance per plane.

**Gate:** Treatment A matches or beats Treatment B (STE QAT) at the same step count. If so, SALT-aware training replaces STE as the default training method.

**Why this comes first:** This is the lowest-hanging fruit — SALT is already implemented, the training loop change is minimal (replace fp16 weights with SALT plane sum), and it validates the core hypothesis before committing to CPU-optimizer infrastructure.

### Experiment 2: CPU-optimizer ternary training

**Hypothesis:** A ternary model trained with CPU-side Adam optimizer + compressed gradient transfer achieves the same loss as GPU-side Adam, at 3-4× lower GPU VRAM usage.

**Experiment design:**

1. **Baseline:** Train a 125M ternary model with standard GPU-side Adam (current v0.50 stack)
2. **Treatment:** Same model, same hyperparameters, CPU-side Adam with:
   - Top-k gradient sparsification (1%)
   - Asynchronous pipeline (GPU forward/backward overlapped with CPU optimizer)
   - FP16 gradient transfer (fallback if top-k quality is insufficient)
3. **Metrics:** loss curve, wall-clock time, GPU VRAM usage, CPU RAM usage
4. **Gate:** Treatment matches baseline loss within 5% at the same step count; treatment uses ≤30% of baseline GPU VRAM

### Combined architecture (if both experiments validate)

SALT-aware training + CPU optimizer together:

```
GPU:  forward(SALT planes) → loss → backward → gradients
                                                     │
                                          (compressed transfer)
                                                     ▼
CPU:  Adam update each plane → re-quantize planes to ternary
                                                     │
                                          (ternary planes back to GPU)
                                                     ▼
GPU:  next forward with updated ternary weights
```

- GPU VRAM: ternary weights (1.37 GB) + activations (~5 GB) = **~6.5 GB** for 7B
- CPU RAM: SALT planes × fp16 (42 GB for 3-plane 7B) + Adam states (84 GB) = **~126 GB**
- Consumer hardware: 4090 + 128 GB DDR5 trains a 7B model with SALT-aware gradients

### Implementation plan

1. **Phase 1 (Experiment 1):** SALT-plane weight representation in `tritium-train` — `SaltWeights` struct holding N planes as fp16 tensors, forward reconstructs `W = sum(planes)`, backward flows through each plane. Wire into the existing training loop. Benchmark vs fp16 + STE QAT.
2. **Phase 2 (Experiment 2):** CPU optimizer adapter — `CpuAdam` that owns shadow weights + Adam state in CPU RAM, accepts gradients from GPU, returns quantized SALT planes.
3. **Phase 3:** Gradient compression — top-k sparsification + index encoding for the GPU→CPU transfer.
4. **Phase 4:** Async pipeline — overlap GPU backward with CPU optimizer step using CUDA streams + CPU threads.
5. **Phase 5:** Integration — combined SALT-aware + CPU-optimizer loop, benchmark on 125M and 1B models.

## 7. Implications for Tritium's roadmap

If this research direction validates, it reshapes the training milestones:

| Current plan | With CPU-optimizer |
|-------------|-------------------|
| v0.50: Single-GPU training | Same — but 3-4× larger models on same GPU |
| v0.60: Multi-GPU FSDP | Less urgent — CPU optimizer + gradient compression may be enough for 7B-13B |
| v0.70+: Scaling | Multi-node with compressed gradients over Ethernet (no InfiniBand required) |
| Consumer hardware | **7B training on a 4090 + 64 GB RAM** — no cloud needed |

The CPU-optimizer approach doesn't replace FSDP/DDP for 70B+ models, but it dramatically lowers the hardware floor for the 1B-13B range that matters most for ternary research and fine-tuning.

## 8. Open questions

1. **How many SALT planes are needed to match STE QAT?** If 2 planes suffice, SALT-aware training is cheaper than STE. If 4+ planes are needed, the advantage shrinks.
2. **Does the plane count need to vary across layers?** Attention layers may need more planes than MLP layers. A per-layer plane budget could optimize the accuracy/compute tradeoff.
3. **Does top-k gradient sparsification converge for ternary models?** The weight space is discrete — sparse gradients may cause oscillation. Needs empirical validation.
4. **What is the optimal sparsification ratio?** 1% is standard for fp16; ternary may need more or less.
5. **Can the CPU optimizer step be pipelined with the GPU backward?** The gradient for layer N is available before layer N-1's gradient — can the CPU start updating layer N while the GPU computes layer N-1's gradient?
6. **Can planes be pruned during training?** Start with 4 planes, gradually drop to 1 as training converges. This would be a smooth curriculum from fp16-like to ternary.
7. **What is the minimum gradient precision for ternary fine-tuning?** INT8? INT4? The discrete weight space may tolerate extreme quantization.

---

*This ADR is a research proposal, not an implementation plan. The experiment in Section 6 should be the first concrete step — it validates the core hypothesis before committing to the full architecture.*
