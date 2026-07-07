# The Mathematically Optimal Hardware Utilization for Ternary Inference and Training

**A first-principles derivation from the roofline model**
*June 2026*

---

## 1. The Hardware Contract

The RTX 4090 (AD102) has exactly two throughput limits that matter:

| Resource | Capacity | Symbol |
|----------|----------|--------|
| HBM bandwidth | 1000 GB/s | `BW` |
| FP32 throughput | 73.7 TFLOPS | `P` |
| L2 cache | 72 MB | `C_L2` |
| Shared memory | 128 KB/SM × 128 SMs | `S` |
| VRAM | 24 GB | `M` |

The **roofline knee** is the arithmetic intensity where the operation transitions from memory-bound to compute-bound:

```
κ = P / BW = 73.7 TFLOPS / 1000 GB/s = 73.7 ops/byte
```

Every operation with arithmetic intensity `AI < κ` is memory-bound. Every operation with `AI > κ` is compute-bound. This is not an approximation — it is the defining equation of the roofline model.

---

## 2. Ternary GEMM: Where It Sits on the Roofline

### 2.1 The Operation

The fundamental ternary operation is:

```
Y[m, n] = scale[n] * Σ_{k=0}^{K-1} act[m, k] * trit[n, k]
```

where `trit ∈ {-1, 0, +1}` and `scale` is a per-channel float. The inner loop is one conditional add/subtract per element — no multiply.

### 2.2 Arithmetic Intensity

For a single output element (M=1, one row of activations):

```
Ops per output:  K (one add/sub per element)
Bytes per output: K * 1.585/8 + 4 (weight bytes + scale)
                ≈ K * 0.198 bytes

AI_single = K / (K * 0.198) = 5.05 ops/byte
```

**5.05 ops/byte is 14.6× below the roofline knee of 73.7 ops/byte.** The operation is deep in the memory-bound regime.

For batch size M (M rows of activations sharing the same weight matrix):

```
Ops per output:  K (same — each row does K adds)
Bytes per output: (K * 0.198 + M * K * 4 / (M * K)) / M
                = K * 0.198 / M + 4 bytes (amortized weight load)

AI(M) = K / (K * 0.198 / M) = M / 0.198 = 5.05 * M ops/byte
```

The arithmetic intensity scales **linearly with batch size**. The crossover point:

```
M* = κ / 5.05 = 73.7 / 5.05 ≈ 14.6
```

| Batch size M | AI (ops/byte) | Regime | GPU utilization |
|-------------|---------------|--------|----------------|
| 1 (decode) | 5.05 | Memory-bound | 6.8% of peak |
| 4 | 20.2 | Memory-bound | 27.4% of peak |
| 8 | 40.4 | Memory-bound | 54.8% of peak |
| **15** | **75.8** | **At knee** | **~100%** |
| 64 (batch decode) | 323 | Compute-bound | ~100% |

**The fundamental tension:** Decode (M=1) wastes 93.2% of the GPU's arithmetic capacity. The GPU is waiting for memory, not computing.

### 2.3 With Zero-Block Sparsity (P1)

If 42.8% of weight blocks are zero and we skip them:

```
Effective bytes per output = K * 0.572 * 0.198 = K * 0.113
AI_sparse(M=1) = K / (K * 0.113) = 8.85 ops/byte
M*_sparse = 73.7 / 8.85 ≈ 8.3
```

Zero-skip moves the crossover from M=15 to M=8. At M=1, GPU utilization goes from 6.8% to 12.0%. **This is the best we can do for single-token decode — the hardware is fundamentally underutilized at M=1.**

---

## 3. The Optimal Inference Strategy

### 3.1 Decode (M=1): Memory-Bound, Minimize Bytes

**Objective:** Maximize tokens per second = `BW / bytes_per_token`.

**Bytes per token (one layer):**
```
W_bytes = N * K * 0.198          (weight matrix)
A_bytes = K * 4 + N * 4          (activation read + output write)
Total   = N * K * 0.198 + (K + N) * 4
```

For K=N=4096:
```
W_bytes = 3.35 MB
A_bytes = 32 KB
Total   = 3.38 MB per layer
```

For 30 layers:
```
Total per token = 30 * 3.38 MB = 101.3 MB
Peak tok/s = 1000 GB/s / 101.3 MB = 9,872 tok/s
```

**But this is the theoretical peak assuming zero overhead.** In practice:
- Kernel launch overhead: ~5-10 μs per kernel × 60+ kernels per token = 300-600 μs
- Activation quantization: one full pass over K elements
- RMSNorm, RoPE, softmax, residual: each is a full read+write of the activation
- Attention: KV cache read scales with sequence length

The **actual bottleneck** is not the matmul — it's the **per-layer non-matmul overhead**. Each transformer layer has ~10 kernel launches (norm, quant, matmul, scale, residual, rope, attention×3, gate, up, down). At 5-10 μs each, that's 50-100 μs per layer, or 1.5-3 ms per token. At 3 ms overhead, the maximum is ~333 tok/s regardless of matmul speed.

**The optimal decode strategy is kernel fusion:**

```
Fused per-layer cost = max(weight_load, fused_kernel_overhead)
                     = max(3.38 MB / 1000 GB/s, ~10 μs)
                     = max(3.38 μs, 10 μs)
                     = 10 μs  (kernel overhead dominates!)
```

With full fusion (one kernel per layer):
```
Fused tok/s = 1 / (30 * 10 μs) = 3,333 tok/s
```

With CUDA graphs (eliminate per-kernel launch overhead):
```
Graph tok/s = 1 / (30 * 3.38 μs) = 9,852 tok/s  (approaches theoretical)
```

**Tritium's current 474 tok/s (M=64 batch) suggests ~21× overhead vs theoretical.** The gap is:
1. Per-kernel launch overhead (~5×)
2. Non-matmul memory traffic (~2×)
3. Suboptimal memory access patterns (~2×)

### 3.2 Prefill (Large M): Compute-Bound, Maximize FLOPS

For prefill with M ≥ 15, the operation becomes compute-bound. The ternary multiply-free property (P2) now matters:

```
FP16 prefill: 2K FLOPs per output (K multiplies + K adds)
Ternary prefill: K ops per output (K adds/subs only)

Speedup from P2: 2K / K = 2.0× (in the compute-bound regime)
```

But int8 tensor cores (IMMA) can process 32 elements per `mma.sync` instruction, while ternary add/sub processes one element per instruction. The tensor-core advantage:

```
IMMA throughput: 32 ops / instruction / ~4 cycles = 8 ops/cycle
Ternary add:     1 op / instruction / ~4 cycles = 0.25 ops/cycle
Ratio: 32×
```

**In the compute-bound regime, tensor cores are 32× faster than scalar ternary add.** The ternary multiply-free property (P2) saves 2× over FP16, but tensor cores save 16× over scalar. **Tensor cores win by 8× even with ternary's 2× advantage.**

**The optimal prefill strategy:**
- Use IMMA tensor cores (`mma.m16n8k32 s32.s8.s8.s32`)
- Pack ternary weights into int8 format (I2sInt8 interleave)
- The int8 "multiplies" by {-1,0,+1} are sign-flips, which the tensor core handles at the same cost as a multiply
- Accumulate in int32 (exact for the value range)

**Prefill throughput (compute-bound):**
```
Ternary with IMMA: 73.7 TFLOPS / K ops per output = 73.7T / 4096 = 17.99G outputs/s
At M=4096: 17.99G / 4096 = 4,394 tokens/s (per layer)
```

### 3.3 The Optimal Kernel Fusion Hierarchy

From the roofline analysis, the optimal fusion strategy is:

```
Level 0 (current):  10+ kernels per layer → 333 tok/s theoretical
Level 1 (fused):    1 kernel per layer    → 3,333 tok/s theoretical
Level 2 (graph):    CUDA graph per layer   → 9,852 tok/s theoretical
Level 3 (resident): All layers in VRAM     → 9,852 tok/s (same, but no H2D)
```

**Level 2 is the practical optimum.** Level 3 requires all weights in VRAM (30 × 3.35 MB = 100 MB — easily fits in 24 GB). Tritium already has a device-resident decode model (`CudaDecodeModel`). The gap is in kernel fusion — the current path still launches separate kernels for norm, quant, matmul, residual, rope, attention, etc.

**The mathematically optimal single-kernel-per-layer fusion:**

```cuda
// Pseudocode for the fused ternary transformer layer
__global__ void fused_ternary_layer(
    float* residual,        // [N] — stays in registers across layers
    const uchar* weights,   // all layers' ternary weights in VRAM
    const uint* bitmap,     // zero-block bitmap per layer
    float* kv_cache,        // [layers, 2, ctx, kv_width]
    // ... norms, scales, etc.
) {
    // 1. RMSNorm: read residual, compute norm, write normed (in registers)
    float normed[N];
    rmsnorm(residual, normed);
    
    // 2. Act quant: normed → int8 (in registers)
    int8_t q[N];
    float act_scale;
    act_quant(normed, q, &act_scale);
    
    // 3. Ternary GEMM: q × weights → output (accumulate in registers)
    //    This is the ONLY HBM read (weights). Everything else is in shared/registers.
    float out[N];
    ternary_gemm_tiled(q, weights_layer, bitmap_layer, out);
    
    // 4. Scale + residual: out = residual + out * scale (in registers)
    for (int i = 0; i < N; i++)
        residual[i] += out[i] * weight_scale[i] * act_scale;
    
    // 5. RoPE + attention: read KV cache from HBM, compute, write back
    attention(residual, kv_cache, ...);
    
    // 6. FFN: same pattern as 1-4 but with gate+up+down
    ffn(residual, weights_ffn, bitmap_ffn, ...);
}
```

**Key insight:** In this fusion, HBM is read exactly once per layer (the weight matrix). All intermediate activations stay in registers or shared memory. The residual stream never touches HBM between layers — it persists in registers across the entire forward pass.

---

## 4. The Optimal Training Strategy

### 4.1 Training Has Different Arithmetic Intensity

Training has three passes per layer:

**Forward:** `Y = X @ W^T` (same as inference)
```
AI_forward = M * K / (K * N * 0.198) = M / (N * 0.198)
For M=32, N=4096: AI = 32 / 811 = 0.039 ops/byte ← memory-bound
```

**Backward (grad_A):** `gA = gY @ W`
```
Same shape as forward. AI_grad_A = M / (N * 0.198) — same memory-bound
```

**Backward (grad_W):** `gW = gY^T @ X`
```
Ops: N * M * K
Bytes: N * K * 0.198 (weight read) + M * K * 4 (activation read) + N * K * 4 (grad write)
For M=32, K=N=4096:
  Ops = 32 * 4096 * 4096 = 536.9M
  Bytes = 4096*4096*0.198 + 32*4096*4 + 4096*4096*4 = 3.35M + 0.5M + 67.1M = 70.9M
  AI_grad_W = 536.9M / 70.9M = 7.57 ops/byte ← still memory-bound!
```

Wait — let me recalculate. The weight gradient accumulates over the batch dimension:

```
gW[n, k] = Σ_m gY[m, n] * X[m, k]
```

The output `gW` is `[N, K]` = 4096 × 4096 × 4 bytes = 67.1 MB (f32 latent weights). This must be **written** to HBM. The activations `X` are `[M, K]` = 32 × 4096 × 4 = 0.5 MB. The output gradients `gY` are `[M, N]` = 32 × 4096 × 4 = 0.5 MB.

```
Total bytes = 67.1M (gW write) + 0.5M (X read) + 0.5M (gY read) = 68.1M
AI_grad_W = 536.9M / 68.1M = 7.88 ops/byte
```

**The gradient computation is also memory-bound at M=32!** The bottleneck is writing the full `[N, K]` gradient matrix to HBM.

### 4.2 Gradient Accumulation: The Training Throughput Lever

If we accumulate gradients over `G` micro-batches before the optimizer step:

```
Bytes per step = 68.1M (same — we write gW once after accumulation)
Ops per step = G * 536.9M
AI_grad_W(G) = G * 536.9M / 68.1M = G * 7.88 ops/byte
```

The crossover to compute-bound:
```
G* = 73.7 / 7.88 = 9.35
```

**With gradient accumulation over ~10 micro-batches, the backward pass becomes compute-bound.** This is the key insight for training: **gradient accumulation is not just for larger effective batch sizes — it fundamentally shifts the hardware utilization from memory-bound to compute-bound.**

| Micro-batches G | AI (ops/byte) | Regime | GPU utilization |
|----------------|---------------|--------|----------------|
| 1 | 7.88 | Memory-bound | 10.7% |
| 4 | 31.5 | Memory-bound | 42.8% |
| 8 | 63.0 | Memory-bound | 85.5% |
| **10** | **78.8** | **At knee** | **~100%** |
| 16 | 126 | Compute-bound | ~100% |

### 4.3 The Memory Wall for Training

Even with optimal compute utilization, training is bounded by VRAM:

```
Per parameter (with AdamW):
  Latent weight (f32):     4 bytes
  Gradient (f32):          4 bytes
  Adam m (f32):            4 bytes
  Adam v (f32):            4 bytes
  Activations (per layer): ~0.5 bytes amortized (with checkpointing)
  Total:                   ~16.5 bytes/param

24 GB VRAM → 24G / 16.5 = 1.45B parameters
```

**A 2B model does not fit in 24 GB with full training.** The options:

| Strategy | Params in 24 GB | Trade-off |
|----------|----------------|-----------|
| Full training | 1.45B | Too small |
| Gradient checkpointing | 1.45B (same — checkpointing saves activations, not params) | — |
| LoRA (rank 16) | ~10M trainable, 2B frozen ternary at 0.4 GB | Quality trade-off |
| FSDP across 2 GPUs | 2.9B | Needs 2 GPUs |
| CPU offload (optimizer) | ~2.5B | 10-50× slower optimizer step |

**The mathematically optimal single-GPU training strategy:**

1. **Load weights in ternary format** (0.4 GB for 2B model)
2. **Keep latent weights in f32** (8 GB for 2B model)
3. **LoRA adapters** (rank 16): ~40 MB trainable + 80 MB optimizer state
4. **Gradient checkpointing**: store only layer inputs, recompute forward
5. **Gradient accumulation** over 10 micro-batches: shifts backward to compute-bound
6. **Mixed precision**: f32 for STE latent weights, bf16 for activations where possible

```
Memory budget:
  Frozen ternary weights:  0.4 GB
  Latent weights (f32):    8.0 GB
  LoRA params + optimizer: 0.12 GB
  Activations (checkpointed): 0.5 GB
  KV cache (if applicable): 1.0 GB
  Workspace: 0.5 GB
  Total: 10.5 GB ← fits in 24 GB with headroom
```

### 4.4 The STE Gradient: Where Ternary Training Wastes Cycles

The STE backward pass is:

```rust
if (wf[i] / s).abs() < 1.0 {
    g_wf[i] = grad_out[i] / s;  // active region: gradient flows
} else {
    g_wf[i] = 0.0;  // saturated: gradient blocked
}
```

For a typical trained model, ~42.8% of weights are in the zero-bin (`|w/s| < 0.5`), ~28.6% are in the active region (`0.5 ≤ |w/s| < 1.0`), and ~28.6% are saturated (`|w/s| ≥ 1.0`).

**The STE zeros 28.6% of gradients.** These are the weights that confidently contribute +1 or -1. They receive no gradient signal, so they cannot adapt. This is the "deadzone" problem.

**Tequila's insight:** Give saturated weights a direct gradient path:

```
g_wf[i] = grad_out[i] * λ  (for saturated weights, λ ≈ 1e-3)
```

This adds ~0 FLOPs (one multiply per saturated weight) and ~0 memory (the gradient is already in registers). **Tequila is free in terms of hardware utilization — it only changes the gradient values, not the compute or memory pattern.**

---

## 5. The Mathematically Optimal Configuration

### 5.1 Decode (Single Token)

```
┌─────────────────────────────────────────────────────────┐
│ OPTIMAL DECODE CONFIGURATION                            │
├─────────────────────────────────────────────────────────┤
│ Architecture: Device-resident (all weights in VRAM)     │
│ Kernel strategy: CUDA graph per transformer layer       │
│ Matmul kernel: Tiled f32 + zero-block bitmap skip       │
│ Fusion: norm+quant+matmul+scale+residual in one kernel  │
│ Attention: Split-KV flash-decoding                      │
│ KV cache: f32 in VRAM                                   │
│                                                         │
│ Bottleneck: HBM bandwidth (1 TB/s)                      │
│ Theoretical peak: ~9,852 tok/s (single layer: 3.38 μs)  │
│ Realistic target: ~2,000-3,000 tok/s (with fusion)      │
│ Current: ~474 tok/s (M=64 batch, unfused)               │
│ Gap to close: 4-6× via fusion + zero-skip               │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Prefill (Long Sequence)

```
┌─────────────────────────────────────────────────────────┐
│ OPTIMAL PREFILL CONFIGURATION                           │
├─────────────────────────────────────────────────────────┤
│ Architecture: Batched M=P tokens, weights in VRAM       │
│ Kernel strategy: IMMA tensor cores (mma.m16n8k32)       │
│ Weight format: I2sInt8 (ternary → int8 sign-flip)       │
│ Accumulator: int32 (exact for int8 × ternary range)     │
│ Activation quant: Per-token int8 absmax                  │
│                                                         │
│ Bottleneck: Compute (73.7 TFLOPS) at M ≥ 15             │
│ Ternary advantage: 2× fewer ops than FP16 (P2)          │
│ Tensor core advantage: 32× over scalar add              │
│ Net: Tensor cores win; use IMMA                         │
└─────────────────────────────────────────────────────────┘
```

### 5.3 Training (QAT)

```
┌─────────────────────────────────────────────────────────┐
│ OPTIMAL TRAINING CONFIGURATION                          │
├─────────────────────────────────────────────────────────┤
│ Strategy: LoRA (rank 16) on frozen ternary base         │
│ Latent weights: f32 (STE needs full precision)          │
│ Optimizer: AdamW with gradient accumulation (G=10)      │
│ Checkpointing: Layer-boundary (recompute forward)       │
│ Gradient kernel: Tiled, deterministic (--fmad=false)    │
│                                                         │
│ Bottleneck: VRAM (16.5 bytes/param)                     │
│ Fits in 24 GB: ~1.45B full, ~2B with LoRA               │
│ Compute regime: Compute-bound at G ≥ 10                 │
│ Ternary advantage (P2): 2× in backward (compute-bound)  │
│ STE improvement: Tequila deadweight reactivation        │
│                                                         │
│ Key insight: Gradient accumulation shifts training      │
│ from memory-bound to compute-bound, where P2 matters.   │
└─────────────────────────────────────────────────────────┘
```

---

## 6. The Fundamental Limits

### 6.1 What Ternary Cannot Beat

**Memory bandwidth is the hard wall.** For decode (M=1), no amount of cleverness can exceed:

```
tok/s_max = BW / bytes_per_token = 1000 GB/s / 101.3 MB = 9,872 tok/s
```

This is a physical limit of the hardware. Ternary's 1.585 bits/weight is what makes `bytes_per_token` small — it's already 20× better than FP16. The remaining gap (474 vs 9,872) is engineering overhead, not a fundamental limit.

### 6.2 What Ternary Can Beat

**Ternary beats FP16 in memory-bandwidth-limited scenarios.** The ratio:

```
tok/s_ternary / tok/s_fp16 = bytes_fp16 / bytes_ternary
                            = (K * 2) / (K * 0.198)
                            = 10.1×
```

**A ternary model is 10.1× faster than the same-sized FP16 model in the memory-bound regime.** This is the fundamental advantage — not the multiply-free property, but the 20× weight compression.

But this only holds if:
1. The model fits in VRAM at ternary precision (it does — 2B ternary = 0.4 GB)
2. The kernel can skip zero weights (P1 — implemented)
3. The kernel is fused to minimize non-matmul memory traffic (not yet)

### 6.3 The Ternary vs. 2-Bit Decision Boundary

```
Ternary: 1.585 bits/weight, 42.8% zero-skip possible
2-bit:   2.000 bits/weight, no zero-skip

Effective bytes with zero-skip:
  Ternary: 0.198 * 0.572 = 0.113 bytes/weight
  2-bit:   0.250 bytes/weight (no sparsity)

Ratio: 0.250 / 0.113 = 2.21×
```

**Ternary with zero-skip is 2.21× faster than 2-bit in the memory-bound regime.** Without zero-skip, 2-bit wins (0.250 vs 0.198 = 1.26× in favor of 2-bit, plus clean SIMD packing).

**The decision boundary:**
- Zero-skip implemented → ternary wins
- Zero-skip not implemented → 2-bit wins

This is the mathematical imperative: **the zero-skip optimization is not optional — it is the defining feature that makes ternary superior to all other low-bit quantization schemes.**

---

## 7. Summary: The Three Regimes

| Regime | Condition | Bottleneck | Ternary advantage | Optimal strategy |
|--------|-----------|------------|-------------------|-----------------|
| **Decode** (M=1) | AI = 5.05 < κ | Memory BW | 10.1× over FP16 (weight compression) | Fusion + zero-skip + CUDA graphs |
| **Prefill** (M≥15) | AI ≥ 75.8 > κ | Compute | 2× over FP16 (multiply-free) | IMMA tensor cores |
| **Training** (G≥10) | AI_grad ≥ 78.8 > κ | Compute | 2× in backward (multiply-free) | LoRA + grad accum + Tequila |

**The single most important insight:** Ternary's advantage is different in each regime. In decode, it's weight compression (P1 + P3). In prefill and training, it's arithmetic efficiency (P2). Tritium has optimized P2 (multiply-free) but the decode regime — where ternary's advantage is largest (10.1× vs 2×) — needs P1 (zero-skip) and kernel fusion to realize the compression advantage.
