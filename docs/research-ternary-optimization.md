# Optimizing Ternary Neural Network Training and Inference on Consumer GPUs and CPU+GPU Hybrid Systems

**A Research Synthesis for the Tritium Project**
*June 2026*

---

## 1. Executive Summary

Ternary neural networks (TNNs) quantize weights to {-1, 0, +1}, achieving 1.58 bits per weight -- a 10-20x compression over FP16. The Tritium project exploits this for multiply-free inference (add/sub/skip kernels) on consumer hardware. This report synthesizes the state of the art across five axes relevant to Tritium: GPU kernel optimization, CPU+GPU hybrid inference, training under 24GB VRAM constraints, SALT quantization quality, and novel algorithmic directions.

**Key findings for Tritium:**

1. **Ternary is memory-bound, not compute-bound.** ParetoQ [1] confirms ternary sits on the Pareto frontier with 2-bit and 3-bit for accuracy-per-byte, but hardware efficiency is worse than 2-bit due to packing overhead. Tritium's current add/sub/skip approach is correct but leaves throughput on the table -- the bottleneck is memory bandwidth, not arithmetic.

2. **Deadzone trapping is the dominant training failure mode.** Tequila [2] shows >4% accuracy gains by repurposing trapped weights as dynamic biases. Sherry's Arenas [3] prevents weight trapping via annealing residual synapses. Both are implementable as modifications to Tritium's STE tape.

3. **90% FP pretrain + 10% QAT is the optimal training split** per ParetoQ [1]. Ternary needs ~30B QAT tokens to saturate. For a solo developer, this means fine-tuning from existing FP checkpoints (e.g., BitNet b1.58 2B4T [4]) is far more practical than training from scratch.

4. **LUT-based kernels can replace add/sub/skip** for 1.6-2.2x area efficiency on ASIC [5], and the same principle applies to GPU shared-memory LUTs. Sherry's 3:4 sparsity pattern packs 4 weights into 5 bits, enabling SIMD-friendly LUT lookup [3].

5. **CPU+GPU hybrid offloading is viable** because ternary matmul on CPU (AVX2 add/sub/skip) runs at useful throughput, while GPU handles the bandwidth-critical decode path. Tritium already has both backends; the missing piece is layer-pipelined overlap.

---

## 2. GPU Kernel Optimization Strategies for Ternary MatMul

### 2.1 Current Approaches in Tritium

Tritium's CUDA backend (`tritium-cuda/src/cuda.rs`) implements three kernel variants:

- **`tq2_0_add_mpgemm`** -- the baseline add-only kernel operating on unpacked TQ2_0 blocks
- **`tq2_0_add_mpgemm_tiled`** -- decode-oriented tiled kernel (one warp per output, one block per row, activation staged in shared memory)
- **`tq2_0_imma_mpgemm`** -- IMMA tensor-core prefill kernel using `mma.m16n8k32` int32 accumulate with I2sInt8 tile interleave
- **`salt_mpgemm_tiled_f32`** -- SALT multi-plane accumulate, summing `scale_p * tmatmul(t_p)` over T stacked planes

The CPU backend (`tritium-cpu/src/kernel.rs`) uses AVX2 add/sub/skip with sequential f32 accumulation to bit-match the scalar reference.

### 2.2 Bottleneck Analysis: Memory-Bound

ParetoQ [1] provides the definitive analysis: ternary inference is **memory-bound** for all practical model sizes. The arithmetic intensity of a ternary GEMM is extremely low -- each output element requires K additions but only loads K * 1.58 bits from memory. For a 4096x4096 layer, the weight tensor is ~3.2 MB at 1.58 bits vs 32 MB at FP16, but the compute is only K additions per output -- far below the roofline knee of any modern GPU.

**Implication for Tritium:** The current tiled kernel's focus on compute efficiency (add/sub/skip) is necessary but not sufficient. The real optimization is maximizing memory bandwidth utilization. On RTX 4090 (1 TB/s HBM bandwidth), the theoretical peak for a 4096x4096 ternary layer is:

```
Weight bytes = 4096 * 4096 * 1.58 / 8 = 3.35 MB
Peak throughput = 1000 GB/s / 3.35 MB = ~298K layers/sec
At batch=1, seq=1: 298K * 4096 = ~1.22B tokens/sec (theoretical)
```

The actual 474 tok/s decode throughput suggests significant overhead from kernel launch, memory management, and the multi-layer pipeline. The gap is in **latency**, not bandwidth.

### 2.3 Consumer GPU Specific Considerations

**RTX 4090 (AD102):**
- 73.7 TFLOPS FP32, 1 TB/s HBM bandwidth
- 72 MB L2 cache (can hold the full weight matrix for models up to ~1B parameters at 1.58 bits)
- 128 KB shared memory per SM (sufficient for staging activation tiles)
- SM count: 128 (enough parallelism for decode if kernel occupancy is high)

**Key optimization targets:**
1. **L2 residency:** For models that fit in 72 MB L2, pin weight tiles to L2 via `cudaFuncSetAttribute` with `cudaFuncAttributePreferredSharedMemoryCarveout`. Tritium's current `CudaDecodeModel` already keeps residuals + KV in VRAM; extending this to weight L2 pinning could eliminate HBM reads for hot layers.

2. **Shared-memory LUT:** Replace the unpack-and-add path with a shared-memory lookup table. For 4-element groups (Sherry's 3:4 pattern), a 32-entry LUT (5 bits index) fits in 128 bytes of shared memory. The kernel loads activation groups, indexes into the LUT, and accumulates -- no unpacking, no branching on zero weights.

3. **CUDA Graphs:** Tritium already has a graph path (`CudaGraph` in the cuda.rs imports). For decode (batch=1), the full multi-layer forward pass should be captured as a single graph to eliminate per-layer launch overhead (~5-10 us per launch on AD102).

### 2.4 IMMA Tensor Cores vs. LUT

Tritium's IMMA kernel uses `mma.m16n8k32` int8 tensor cores with I2sInt8 tile interleave. This is effective for prefill (large batch) but has overhead for decode: the ternary weights must be repacked into the I2sInt8 interleave format, and the int32 accumulator must be converted back to f32.

**LUT-based alternative (from [5]):** The LUT accelerator paper shows that for ternary weights, precomputing all 3^mu partial sums in a lookup table eliminates multiplications entirely. On GPU, this maps to:
- Store the LUT in shared memory (32 entries for mu=4)
- Each warp loads activation groups, uses them as LUT indices
- Accumulate LUT outputs into registers

The GPU implementation would look like:

```cuda
// Shared memory LUT for 4-element ternary groups
__shared__ float lut[81]; // 3^4 = 81 entries
// Build phase: compute all partial sums for this activation group
for (int i = threadIdx.x; i < 81; i += blockDim.x) {
    int idx = i;
    float sum = 0;
    for (int j = 0; j < 4; j++) {
        int trit = (idx % 3) - 1; // {-1, 0, +1}
        sum += trit * act_base[j];
        idx /= 3;
    }
    lut[i] = sum * scale;
}
__syncthreads();
// Fetch phase: index into LUT with packed weight bits
int w_idx = pack_ternary_4weights(weight_group);
out += lut[w_idx];
```

This approach eliminates the branch-heavy add/sub/skip dispatch and replaces it with a single memory lookup. The build phase is O(3^mu) but amortized over many weight groups sharing the same activation tile.

---

## 3. CPU+GPU Hybrid Inference

### 3.1 Layer-Pipelined Offloading

The idea: while the GPU processes layer N's matmul, the CPU processes layer N-1's RMSNorm + RoPE + softmax (or vice versa). Tritium already has separate CPU and CUDA backends with device-resident kernels for RMSNorm, RoPE, and softmax on GPU. The missing piece is **overlapped execution**.

**Architecture:**
```
CPU thread:  [Embed] -> [Layer0 Norm+Rope] -> [Layer1 Norm+Rope] -> ...
GPU stream:            [Layer0 MatMul]       -> [Layer1 MatMul]       -> ...
```

For this to work, the CPU must finish its norm/rope work before the GPU needs the activation, and the GPU must finish its matmul before the CPU needs the output. The synchronization point is the activation transfer.

**Bandwidth matching:** On RTX 4090, PCIe 4.0 x16 provides ~32 GB/s bidirectional. A 4096-dim activation row is 16 KB (f32). At 32 GB/s, transferring one row takes ~0.5 us. The GPU matmul for that row takes ~10-50 us (depending on K). So the transfer is negligible -- the bottleneck is the GPU kernel, not the bus.

### 3.2 CPU Handles Embedding/Norm/Sampling

Tritium's `ModelRunner::forward` already calls embedding lookup (CPU), then feeds through transformer blocks. For CPU+GPU hybrid:

1. **Embedding:** Pure CPU table lookup (token -> f32 vector). Already fast.
2. **RMSNorm:** Fused into the GPU decode kernel (`rmsnorm_f32`). Keep on GPU.
3. **Sampling:** `sample_greedy` is a single argmax over vocab logits. On CPU for batch=1, this is ~1 us for vocab=32000. Negligible.
4. **LM Head:** Tied to embedding (dense f32 matmul). This is the one expensive CPU operation -- for hidden_dim=2048, vocab=32000, it's 64M f32 ops. On AVX2, ~2 ms. On GPU, ~0.1 ms.

**Recommendation:** Move the LM head matmul to GPU. It's the only CPU operation that matters at scale. The tied embedding lookup stays on CPU (it's just indexing).

### 3.3 Bandwidth Matching Between CPU and GPU

The key insight from VitaLLM [6] and related work: ternary inference is bandwidth-limited, so the optimal split places bandwidth-critical operations (matmul) on the highest-bandwidth device (GPU) and latency-tolerant operations (norm, sampling) on CPU.

For Tritium's decode path (batch=1):
- GPU bandwidth: 1 TB/s (HBM)
- CPU bandwidth: ~50 GB/s (DDR5)
- PCIe bandwidth: 32 GB/s

The ternary weight matrix is ~3.35 MB per layer (4096x4096). At 1 TB/s, the GPU loads it in 3.35 us. At 50 GB/s, the CPU would take 67 us -- 20x slower. This confirms: **matmul must stay on GPU**.

---

## 4. Training on Consumer GPUs

### 4.1 QAT Memory Budget (24GB VRAM)

A 2B-parameter ternary model requires:
- **Weights (f32 for STE):** 2B * 4 bytes = 8 GB
- **Activations (per-layer, for backward):** ~2-4 GB depending on sequence length
- **Gradients:** 8 GB (same size as weights)
- **Optimizer state (Adam):** 16 GB (2x weights for m and v)
- **Total:** ~34-38 GB -- exceeds 24 GB

This is why gradient checkpointing and LoRA are essential.

### 4.2 Gradient Checkpointing

Tritium's `tritium-train` crate already has `checkpoint.rs`. The strategy: store only layer inputs during forward, recompute activations during backward. This trades compute for memory:

- Without checkpointing: store all intermediate activations (~2-4 GB)
- With checkpointing: store only layer inputs (~0.5 GB), recompute ~2x forward compute

With checkpointing, the memory budget becomes:
- Weights: 8 GB
- Checkpointed activations: 0.5 GB
- Gradients: 8 GB
- Optimizer states: 16 GB
- **Total: 32.5 GB** -- still over 24 GB

**LoRA is required** to fit in 24 GB. With rank-16 LoRA on attention projections:
- LoRA params: ~10M * 4 bytes = 40 MB
- LoRA gradients: 40 MB
- LoRA optimizer: 80 MB
- Full weights (frozen, ternary): 8 GB (but can be loaded in 1.58-bit format = ~0.4 GB)
- **Total with LoRA + checkpointing: ~10-12 GB** -- fits comfortably

### 4.3 STE Improvements

**Tequila (Deadzone Reactivation) [2]:**
The core problem: weights in the deadzone (-Delta, Delta) receive noisy, uninformative gradients, causing "ineffective oscillation." Tequila's solution:
- Dead weights function as dynamic biases: `Y = X * W_hat * alpha + sum(lambda * w_i)` for dead weights
- Gradients for dead weights become: `dL/dw_i = x_i * dL/dY + lambda * dL/dY` -- both input-dependent and direct signal paths
- Overhead: <0.1% (bias addition is free relative to matmul)
- Result: >4% accuracy gain over SOTA, within <1% of BF16

**Implementation in Tritium's tape.rs:**
The current tape uses `ste::quantize_surrogate` for the STE backward pass. To add Tequila:
1. During forward, identify deadzone weights (|w| < Delta)
2. Accumulate their contribution as a bias term
3. During backward, add the direct gradient path for dead weights
4. The reactivation parameter lambda defaults to 1e-3 (robust across 1e-5 to 1e-1 per [2])

**Sherry Arenas [3]:**
A complementary approach: during training, add a decaying full-precision residual:
```
Y = X * T_alpha + lambda_t * X * W
```
where lambda_t anneals from 1 to 0 (cosine decay). This injects heterogeneous gradients, preventing the gradient homogenization that causes representational collapse (Effective Rank drops to <750 out of 4096). Post-training, the residual vanishes completely -- zero inference overhead.

### 4.4 Scaling Laws for Ternary Training

ParetoQ [1] establishes:
- **90% FP pretrain + 10% QAT** is the optimal split for a fixed total budget
- Lower-bit settings (1/1.58/2-bit) require ~30B QAT tokens to saturate
- 3-bit and 4-bit saturate at ~10B tokens
- Fine-tuning from pre-trained FP weights consistently outperforms training from scratch

BitNet b1.58 2B4T [4] demonstrates that ternary models match FP16 at the 2B scale with:
- Squared ReLU activation (not GELU)
- No bias terms
- 3-phase training: pretrain + SFT + DPO
- Training on publicly available web data

**When does ternary training converge?** The "Extra RMSNorm" paper [7] shows that inserting RMS normalization before quantization and applying a gradual quantization schedule (ramping from FP to ternary over training) enables stable fine-tuning. This is the simplest approach for a solo developer: start from an FP checkpoint, gradually introduce ternary quantization during fine-tuning.

---

## 5. SALT Quantization Quality Preservation

### 5.1 Multi-Plane Residual Expansion

Tritium's SALT format (`tritium-format/src/salt.rs`) represents weights as a sum of T ternary planes:
```
W ~ sum_p scale_p * trit_p
```
Each plane is a standard TQ2_0 row (per-256-block f16 scales). The first plane is the dense base; subsequent planes are residuals.

**Quality vs. storage tradeoff:**
- T=1: standard ternary, 1.58 bits/weight
- T=2: ~3.16 bits/weight, significant quality improvement
- T=3: ~4.74 bits/weight, approaching 4-bit quality

The rate-distortion plane allocation in `tritium-quantize/src/allocate.rs` determines which layers get more planes. Sensitivity-aware allocation (spending more planes on attention projections than FFN) is the current approach.

### 5.2 Sensitivity-Aware Plane Allocation Improvements

FTerViT [8] provides a key insight: LayerNorm parameters hold <0.2% of parameters but account for 34-39% of Taylor-FO importance. For Tritium, this means:

1. **Per-layer plane budgets should be proportional to Taylor-FO sensitivity**, not uniform
2. **The first and last layers** (embedding, LM head) are most sensitive -- they should get the most residual planes
3. **Middle FFN layers** are least sensitive -- T=1 is often sufficient

**Practical approach for Tritium:**
```rust
// In allocate.rs, compute per-layer sensitivity:
fn layer_sensitivity(layer_idx: usize, n_layers: usize) -> f32 {
    // U-shaped: higher at first/last layers
    let pos = layer_idx as f32 / n_layers as f32;
    1.0 + 2.0 * (1.0 - 2.0 * (pos - 0.5).abs())
}
```

### 5.3 Post-Training vs. QAT Quality Gap

ParetoQ [1] shows QAT consistently outperforms post-training quantization (PTQ) at all bit widths. For ternary, the gap is substantial:
- PTQ ternary: significant perplexity degradation (>1.0 PPL increase)
- QAT ternary (10B tokens): within 0.5 PPL of FP16
- QAT ternary (30B tokens): saturates, within 0.2-0.3 PPL

**Recommendation:** SALT's offline quantizer should be used only for rapid prototyping. Production models should use QAT via Tritium's training tape. The offline quantizer's role is to provide initial plane allocation guidance (which layers need more planes), not final quality.

### 5.4 Mode Codebook Refinement

The current SALT quantizer uses AbsMean scaling per 256-block. Sherry [3] shows that a "greedy Sparse-AbsMean strategy" -- pruning the smallest absolute magnitude element per block, then computing scaling factors as the mean absolute value of non-pruned weights -- achieves better rate-distortion. The key difference: the scale is computed over the **non-pruned** weights only, which better captures the true signal magnitude.

---

## 6. Novel Algorithmic Directions

### 6.1 LUT-Based Ternary MatMul on GPU

The LUT accelerator paper [5] formalizes the design space: for mu-element ternary groups, a 3^mu-entry LUT stores all possible partial sums. On GPU:

- **Shared memory LUT:** 3^4 = 81 entries * 4 bytes = 324 bytes per LUT. Fits easily in 128 KB shared memory per SM.
- **Build phase:** Each warp computes 81 partial sums from the activation tile. Cost: 81 * 4 = 324 FMA ops per warp.
- **Fetch phase:** Pack 4 ternary weights into a 5-bit index (Sherry's 3:4 packing), index into LUT, accumulate.

**Expected speedup over add/sub/skip:** 1.5-2x for decode (batch=1), because:
- No branching on zero weights (the LUT handles sparsity implicitly)
- Better instruction-level parallelism (one LUT lookup vs. 4 conditional adds)
- Shared memory latency (~20 cycles) vs. register file latency (~1 cycle) is amortized over many groups

### 6.2 Sparse Residual Planes + Structured Sparsity (Sherry 3:4)

Sherry's 3:4 pattern [3] enforces exactly 3 non-zero weights per 4-element block. This yields:
- 32 unique permutations per block (C(4,3) * 2^3 = 32)
- Packable into 5 bits (1 sign + 4 index), saturating a 5-bit index
- 25% sparsity stays below the 50% threshold where ternary performance degrades

**Integration with SALT:** Apply 3:4 sparsity to each residual plane independently. The base plane (T=0) uses standard ternary; residual planes (T>=1) use 3:4 sparsity. This gives:
- Base plane: 1.58 bits/weight (standard)
- Residual planes: 1.25 bits/weight (Sherry)
- Total for T=2 SALT: 1.58 + 1.25 = 2.83 bits/weight (vs. 3.16 without sparsity)

### 6.3 Learned Quantization Scales (vs. Fixed AbsMean)

ParetoQ [1] finds that **learnable scales consistently outperform statistics-based range clipping** across all bit widths. For Tritium, this means:

1. Replace the fixed AbsMean scale in TQ2_0 with a learnable per-group scale
2. During QAT, backpropagate through the scale (the STE surrogate already handles this)
3. The scale becomes a per-group parameter, adding ~0.01% parameters (negligible)

**Implementation:** In `tritium-quantize/src/quantize.rs`, change the scale computation from:
```rust
let scale = mean_abs(group); // fixed
```
to:
```rust
let scale = learnable_scale[group_idx]; // learned during QAT
```

The gradient flows through the STE surrogate: `dL/dscale = sum(dL/dY * X * sign(W))`.

### 6.4 Speculative Decoding with Ternary Draft Models

A ternary 1B model (e.g., BitNet b1.58 1B) can serve as a draft model for speculative decoding with a larger FP16 target. The draft generates K candidate tokens at ternary speed (~474 tok/s on RTX 4090), then the target verifies them in parallel.

**Expected speedup:** If the draft acceptance rate is alpha, the effective throughput is:
```
speedup = K * alpha / (1 + K * cost_draft / cost_target)
```
For alpha=0.7 (typical for same-family models), K=4, and cost_draft/cost_target=0.1:
```
speedup = 4 * 0.7 / (1 + 4 * 0.1) = 2.8 / 1.4 = 2.0x
```

This is a free speedup that doesn't require any kernel changes -- just a draft model and a verification loop.

### 6.5 Block-Diffusion Drafter for Ternary Target

An alternative to autoregressive speculative decoding: use a small ternary block-diffusion model to generate K tokens in parallel (one diffusion step), then verify against the target. This eliminates the sequential draft bottleneck.

**Feasibility:** Block-diffusion models (e.g., MDLM, SEDD) can generate multiple tokens per forward pass. A ternary block-diffusion drafter would run at ~474 * K tok/s (amortized). For K=4, that's ~1900 tok/s draft rate. The verification step is one forward pass of the target model over K tokens, which is ~Kx slower than single-token decode but accepts/rejects in bulk.

This is a research direction, not a near-term optimization. The speculative decoding approach (6.4) is simpler and well-proven.

---

## 7. Consumer GPU Deployment Recommendations

### 7.1 RTX 4090 (24GB) -- Optimal Configuration

| Component | Configuration | VRAM |
|-----------|--------------|------|
| Model (2B ternary, T=1) | ~400 MB weights | 0.4 GB |
| KV cache (4096 ctx, 22 layers) | ~1.1 GB | 1.1 GB |
| CUDA decode model (residuals in VRAM) | ~0.5 GB | 0.5 GB |
| Workspace (activations, temp) | ~0.5 GB | 0.5 GB |
| **Total** | | **~2.5 GB** |

**Headroom:** 21.5 GB free. Use this for:
- Larger KV cache (up to ~80K context at 2B)
- Batch decode (multiple sequences in parallel)
- SALT multi-plane weights (T=2 doubles weight memory to 0.8 GB, still fine)
- Larger models (7B ternary = ~1.4 GB weights, fits easily)

**Kernel config:**
- Prefill: IMMA tensor-core kernel (large batch, compute-bound)
- Decode: Tiled add-only kernel (batch=1, memory-bound)
- CUDA Graphs for full decode pipeline (eliminate per-layer launch overhead)

### 7.2 RTX 3090 (24GB) / RTX 3060 (12GB) Budget Configs

**RTX 3090 (24GB, Ampere):**
- Same memory budget as 4090
- Lower HBM bandwidth (936 GB/s vs 1 TB/s) -- ~7% slower decode
- IMMA tensor cores available (same kernel path)
- SM count: 82 vs 128 -- lower occupancy, but decode (batch=1) doesn't need full occupancy

**RTX 3060 (12GB, Ampere):**
- Tight memory: 2B ternary model fits, but KV cache is limited to ~4K context
- For longer contexts: use CPU KV cache overflow (Tritium's CPU fallback)
- Prefill on GPU, decode with CPU KV management
- Recommended: 1B ternary model (e.g., BitNet b1.58 1B) for comfortable headroom

### 7.3 CPU Fallback for Overflow

Tritium's CPU backend (`tritium-cpu`) with AVX2 add/sub/skip provides:
- ~50-100 tok/s decode on modern CPUs (DDR5, AVX2)
- No VRAM limit (uses system RAM)
- Useful for: KV cache overflow, embedding/norm offload, development/debugging

**Hybrid strategy for 12GB GPUs:**
1. Load weights on GPU (fits in 1-2 GB)
2. Run matmul on GPU
3. Overflow KV cache to CPU when GPU memory is exhausted
4. CPU handles embedding lookup + sampling

---

## 8. Research Gaps and Open Questions

1. **Ternary kernel auto-tuning on consumer GPUs.** ParetoQ [1] shows optimal tile geometry depends on data type and hardware. Tritium's autotune module (`tritium-cuda/src/autotune.rs`) should be extended to explore LUT-based kernels alongside the current add/sub/skip and IMMA paths.

2. **Activation quantization impact.** BitNet b1.58 uses 8-bit activations (W1.58A8). Tritium's `act_quant_int8_per_token` kernel handles this, but the quality impact of different activation quantization schemes (absmax vs. percentile vs. learned) on consumer GPU training is underexplored.

3. **SALT plane allocation theory.** The current allocation uses heuristics. A principled approach would compute the rate-distortion slope for each layer (how much PPL improvement per additional plane) and allocate planes greedily. This requires running the quantizer multiple times with different plane counts -- expensive but one-time.

4. **Ternary + LoRA interaction.** Does LoRA rank need to be higher for ternary models than FP16? The reduced weight precision may require more low-rank capacity to compensate. ParetoQ's scaling laws [1] suggest ternary models need ~2x hidden size to match FP (for BERT-scale), which implies LoRA rank should also scale.

5. **End-to-end training pipeline.** Tritium has the tape (`tape.rs`), optimizer (`optim.rs`), LoRA (`lora.rs`), and distributed training (`dist.rs`, `fsdp.rs`). The missing piece is a complete training script that ties these together with data loading, learning rate scheduling, and checkpointing. This is engineering, not research, but it's the bottleneck for iterating on training improvements.

6. **Ternary speculative decoding.** No published work specifically evaluates ternary draft models for speculative decoding. The acceptance rate between a ternary draft and FP16 target of the same architecture family is unknown. This is a high-value experiment for Tritium.

---

## 9. References

[1] ParetoQ: Scaling Laws in Extremely Low-bit LLM Quantization. HuggingFace Paper 2502.02631. https://huggingface.co/papers/2502.02631

[2] Tequila: Trapping-free Ternary Quantization for Large Language Models. HuggingFace Paper 2509.23809. https://huggingface.co/papers/2509.23809

[3] Sherry: Hardware-Efficient 1.25-Bit Ternary Quantization via Fine-grained Sparsification. HuggingFace Paper 2601.07892. https://huggingface.co/papers/2601.07892

[4] BitNet b1.58 2B4T. Microsoft Research. https://huggingface.co/papers/2504.12628

[5] Hardware Generation and Exploration of Lookup Table-Based Accelerators for 1.58-bit LLM Inference. HuggingFace Paper 2604.25183. https://huggingface.co/papers/2604.25183

[6] VitaLLM: A Versatile, Ultra-Compact Ternary LLM Accelerator with Dependency-Aware Scheduling. HuggingFace Paper 2604.27396. https://huggingface.co/papers/2604.27396

[7] An Extra RMSNorm is All You Need for Fine Tuning to 1.58 Bits. HuggingFace Paper 2505.08823. https://huggingface.co/papers/2505.08823

[8] FTerViT: Fully Ternary Vision Transformer. HuggingFace Paper 2605.21171. https://huggingface.co/papers/2605.21171

[9] Hybrid Gated Flow (HGF): Stabilizing 1.58-bit LLMs via Selective Low-Rank Correction. HuggingFace Paper 2602.05269. https://huggingface.co/papers/2602.05269

[10] Token-Scaled Logit Distillation for Ternary Weight Generative Language Models. HuggingFace Paper 2308.06744. https://huggingface.co/papers/2308.06744

[11] EdgeRazor: A Lightweight Framework for Large Language Models via Mixed-Precision Quantization-Aware Distillation. HuggingFace Paper 2605.04062. https://huggingface.co/papers/2605.04062
