# ADR 0017 — PEPPER: Ternary Execution on Commodity Hardware

- **Status:** Proposed
- **Date:** 2026-06-22
- **Relates:** complements [ADR 0016](./0016-ternary-training-methods.md) (SALT-aware training); the two form the full ternary pipeline: SALT trains, PEPPER runs
- **Vision:** the ternary revolution — make {-1, 0, +1} models the default, not the exotic

---

## 1. SALT and PEPPER

SALT and PEPPER are two halves of one idea:

| | SALT | PEPPER |
|---|------|--------|
| **Full name** | Sensitivity-Allocated Layered Ternary ([ADR 0001](./0001-salt-quantization.md)) | Packed Execution with Parallel Precision for Efficient Runtimes |
| **Role** | Training: fp16 → ternary | Execution: ternary → fast |
| **What it does** | Multi-plane residual decomposition; replaces STE with structured gradient estimator | Maps ternary arithmetic onto commodity GPU hardware; eliminates multiply ops entirely |
| **ADR** | [0016](./0016-ternary-training-methods.md) | This ADR |
| **Analogy** | The recipe | The kitchen |

SALT trains the model. PEPPER runs it. Together they form a complete pipeline from fp16 training to ternary inference with no intermediate representations, no overhead, no compromises.

## 2. The goal: remove overhead between continuous and discrete

The user's stated goal: *remove overhead between the raw continuous representation of the feature space and the coarsest fine-enough representation of it.*

In conventional quantization, there are multiple intermediate steps:

```
fp32 training → fp16 checkpoint → INT8 quantization → INT4 quantization → ternary
                 (lossy)           (lossy)             (lossy)            (lossy)
```

Each step introduces error. Each step requires calibration data. Each step needs its own kernels.

**SALT + PEPPER collapses this to one step:**

```
SALT training (fp16 planes) → SALT quantize (ternary planes) → PEPPER execution (ternary kernels)
       ↑ clean gradients              ↑ structured decomposition           ↑ zero multiplies
```

No intermediate quantizations. No calibration. No INT8/INT4 stepping stones. The fp16 planes ARE the training representation, and the ternary planes ARE the inference representation. The SALT decomposition is the bridge — and it's lossless (more planes = exact reconstruction).

## 3. Exploiting NVIDIA floating-point hardware for ternary

This is the unconventional part. NVIDIA GPUs are built for floating-point matrix multiplication — tensor cores do `D = A × B + C` in FP16/BF16/FP32. Ternary models don't multiply (weights are {-1, 0, +1}). So why use floating-point hardware?

**Because the ternary multiply-add is a subset of the floating-point multiply-add, and it's faster.**

### 3.1 What ternary GEMM actually computes

For a ternary weight `w ∈ {-1, 0, +1}` and activation `a`:

```
w = +1  →  a × 1 = a         (passthrough)
w =  0  →  a × 0 = 0         (skip)
w = -1  →  a × (-1) = -a     (sign flip)
```

The "multiply" is free — it's a conditional negate or zero. The actual work is the **accumulate** (sum across the K dimension). This is an **addition-only** operation.

### 3.2 How NVIDIA hardware can execute this

**Option A: Tensor cores with identity weights.** Tensor cores do `D = A × B + C` in one instruction. If B is ternary, the multiply is trivial — but the tensor core still executes it at full throughput. The advantage: **no weight loading bottleneck.** Ternary weights are 10× smaller than FP16, so the weight fetch is 10× faster. The tensor core's multiply unit is wasted (multiplying by ±1 or 0), but the memory bandwidth savings dominate.

**Option B: CUDA cores with FMA chains.** Instead of `a × w + acc`, do:
```
if (w == +1) acc += a;
if (w == -1) acc -= a;
// if (w == 0) skip
```
This is a branch per element — bad for GPU. But if the weights are **packed as bitmasks** (2 bits per weight), the branch becomes a bitwise operation:
```
mask_hi = (packed >> 1) & 1;   // sign bit
mask_lo = packed & 1;           // nonzero bit
acc = mask_lo ? (mask_hi ? acc - a : acc + a) : acc;
```
With predication (no branch), this maps to 3-4 ALU ops per element vs 1 FMA. But the memory bandwidth is 10× lower, so the ALU cost is hidden.

**Option C: Integer tensor cores (DP4A).** NVIDIA's INT8 tensor cores do `D += A[i] * B[i]` for 4 INT8 values packed in a register. If ternary weights are encoded as INT8 values in {-1, 0, +1}, the INT8 tensor core executes the ternary GEMM at **INT8 throughput** — which is 2× FP16 throughput on Ampere/Hopper.

The encoding:
```c
// Pack 4 ternary values {-1, 0, +1} into 4 INT8 values
int8_t packed[4] = { trit_to_int8(w0), trit_to_int8(w1), trit_to_int8(w2), trit_to_int8(w3) };
// DP4A: acc += sum(a[i] * packed[i]) — runs at INT8 tensor core speed
```

This is the fastest path: **2× FP16 throughput, 10× less weight memory, zero wasted multiplies.**

### 3.3 The sparsity angle: skip zero weights

Zero weights are free in ternary — they contribute nothing. NVIDIA's **structured sparsity** hardware (Ampere+) can skip 2:4 sparse patterns. Ternary weights with ~33% zeros can be mapped to this:

```
Ternary:    [+1, 0, -1, +1, 0, 0, -1, +1, ...]
2:4 sparse: [w, 0, w, w, 0, 0, w, w, ...]      (at most 2 zeros per 4)
```

Not all ternary patterns fit 2:4 sparsity, but with weight reordering (similar to how NVIDIA's sparse tensor cores work), a significant fraction can. The hardware skips the zero elements entirely — **free speedup from ternary's natural sparsity.**

### 3.4 Unified approach: PEPPER kernel design

The PEPPER kernel for ternary GEMM:

```
1. Load ternary weights (packed, 4 trits/byte) → shared memory
2. Unpack to INT8 {-1, 0, +1} in registers
3. Load FP16 activations → registers
4. Cast activations to INT8 (or keep FP16 for accuracy)
5. DP4A tensor core: accumulate INT8 products
6. Cast result back to FP16/FP32 for output
```

Alternative path (FP16 tensor cores, higher accuracy):
```
1. Load ternary weights (packed) → shared memory
2. Unpack to FP16 {-1.0, 0.0, +1.0} in registers
3. FP16 tensor core: accumulate (multiply is trivial)
4. Output FP16/FP32
```

The choice depends on the accuracy requirement. INT8 path is 2× faster; FP16 path is more accurate. Both exploit the fact that ternary weights eliminate the multiply.

## 4. Training with PEPPER: ternary forward, fp16 backward

The SALT-aware training loop (ADR 0016) uses fp16 planes during training. But the **forward pass** can use PEPPER kernels:

```
# Training step:
# Forward: use PEPPER ternary GEMM (fast, exploits sparsity)
W_ternary = quantize(sum(planes))           # ternary weight
y = ternary_gemm(x, W_ternary)              # PEPPER kernel

# Backward: use standard fp16 GEMM (accurate gradients)
∂L/∂W = x.T @ ∂L/∂y                        # standard matmul
∂L/∂plane_i = ∂L/∂W × ∂W/∂plane_i          # chain rule through SALT

# Update: Adam on planes (fp16)
plane_i -= lr × adam(∂L/∂plane_i)
```

The forward pass is **ternary-fast** (PEPPER kernel, sparse, no multiplies). The backward pass is **fp16-accurate** (standard gradients). This gives training speed close to ternary inference speed, with gradient quality close to fp16 training.

## 5. The full pipeline: SALT + PEPPER

```
┌─────────────────────────────────────────────────────────────────┐
│  Training (SALT-aware + PEPPER forward)                         │
│                                                                 │
│  Planes (fp16) ──→ Quantize ──→ Ternary weights                │
│       ↑                           │                             │
│       │                           ▼                             │
│  Adam update ◄── Gradients ◄── PEPPER forward (ternary GEMM)   │
│                                                                 │
│  GPU: PEPPER forward (fast) + standard backward (accurate)      │
│  CPU: Adam on planes (optional, for memory savings)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                    SALT quantize (final)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Inference (PEPPER only)                                        │
│                                                                 │
│  Ternary weights (packed, 4 trits/byte)                         │
│       │                                                         │
│       ▼                                                         │
│  PEPPER kernel: INT8 DP4A or sparse FP16 tensor core            │
│       │                                                         │
│       ▼                                                         │
│  Output (FP16/FP32)                                             │
│                                                                 │
│  10× less weight memory than fp16                               │
│  2× throughput via INT8 tensor cores                             │
│  Zero wasted multiplies                                         │
│  Natural sparsity exploitation                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 6. What this means for Tritium

Tritium already has:
- SALT format and quantization (`tritium-format`)
- TQ2_0 packed ternary GEMM kernels (`tritium-cuda`)
- SALT-aware training concept (ADR 0016)

PEPPER adds:
- **INT8 tensor core path** — DP4A kernels for ternary GEMM (2× FP16 throughput)
- **Structured sparsity mapping** — ternary zeros → 2:4 sparse hardware skip
- **PEPPER forward in training** — ternary-speed forward with fp16-quality backward
- **Weight packing optimizations** — 4 trits/byte, shared memory staging, L2 residency

The first concrete deliverable: a PEPPER GEMM kernel that uses DP4A (INT8 tensor cores) for ternary inference. Benchmark against the existing TQ2_0 tiled kernel (which uses FP16 tensor cores). If DP4A is 2× faster, that's the PEPPER kernel.

## 7. Open questions

1. **Does DP4A accuracy suffice for ternary GEMM?** INT8 accumulation may lose precision for large K dimensions. Need to benchmark vs FP16 tensor core path.
2. **How much ternary sparsity can map to 2:4 structured sparsity?** Weight reordering may be needed. What's the reorder cost vs speedup?
3. **Can the PEPPER forward + standard backward training loop converge?** The forward uses quantized weights, the backward uses fp16 gradients. Does this asymmetry cause instability?
4. **What's the end-to-end speedup?** PEPPER forward (ternary, fast) + standard backward (fp16, normal) vs all-fp16 training. If the forward is 2× and the backward is unchanged, training is ~1.5× faster.
5. **Can PEPPER kernels be auto-tuned?** The optimal kernel (DP4A vs FP16 tensor core vs sparse) depends on the model shape, sparsity pattern, and GPU architecture. Can we select at runtime?

---

*SALT and PEPPER together: train in the structured residual space, execute on hardware that was built for floating-point but runs ternary faster. The ternary revolution isn't about making hardware for ternary — it's about making ternary for hardware.*
