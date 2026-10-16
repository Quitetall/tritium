# Ternary training landscape and ablation ladder

Research snapshot: 2026-10-16. Three parallel literature and code surveys, each source marked by
what was read: **V** (primary source read), **PV** (abstract only), **U** (secondary or memory).
Nothing here is a reproduction or a qualification. Companion notes: the residual-carry, rich-state
and capacity results under `docs/experiments/eat-o/{carry,rich,capacity}-20261003/`.

Tritium's purpose sets the frame: **integer-only training, quantization and serving of ternary
networks, usable from existing PyTorch scripts and tooling, and extensible.** The question is what
already exists for each part, what is still open, and how to run a ladder of experiments toward
forward-only integer training without confounding the levers.

## 1. What exists, by part of the stack

### Training: every ternary LLM trainer keeps fp master weights

| Work | Scale | Master weights | Notes |
|---|---|---|---|
| BitNet b1.58 2B4T (2504.12285) V | 2B, 4T tokens | fp, released as a separate "master weight" checkpoint | avg 54.19 vs Qwen2.5-1.5B 55.23 |
| TriLM / Spectra (2407.12327) V | 99M–3.9B, 300B tokens | "latent (or master) weights in floating point" | 3.9B matches FloatLM 3.9B |
| Tequila (2509.23809) V | Llama-3.2 1B/3B, 10B QAT tokens | fp latents | deadzone weights reused as bias; already ported |
| BitDistill (2510.13998) V | Qwen3 ≤4B | not stated | MNLI 91.40 vs fp16 91.48 |
| Falcon-Edge / onebitllms V | 1B/3B, 1.5T tokens | bf16 (fake-quant QAT) | Axolotl integration |
| torchao `bitnet_training` V | — | fp master, int8 tensor-core matmuls | no extra RMSNorm |
| Ternary Bonsai (PrismML) | 1.7B–27B | not disclosed | — |

**Master-free or integer-latent precedents** exist, but none for ternary transformers:

| Work | What it removes | Largest scale, result |
|---|---|---|
| WAGE (1802.04680) V | fp master: **8-bit integer accumulator behind a ternary forward weight**, integer SR steps | AlexNet ImageNet 51.6% top-1 error |
| WAGEUBN (1909.02384) V | everything to 8-bit; 24-bit integer update accumulator | ResNet50 67.95 vs fp 74.66 |
| NITI (2009.13108) | int8 weights, int32 accumulation | ImageNet needs 16-bit weight accumulation |
| Ghaffari 2022 (2207.08822) V | int8 block-fixed-point fwd/bwd, int16 SGD | ResNet18 69.25 vs 69.75 (float glue between layers) |
| Shalby, Pittorino & Roveri (2610.16298, 2026-10-12) PV | INT8 + power-of-two exponent for every tensor, integer optimizer | ResNet/ViT-B ImageNet claimed, **numbers unread — read first** |
| Bop (1906.02107) V | latent weight → fp EMA + threshold flip (binary) | Bi-Real ImageNet 56.6 vs 56.4 |
| BOLD (2405.16339) V | per-weight accumulator, xnor threshold flip, **reset** (binary) | ImageNet 66.9 / 70.0 distilled; first/last layers fp |
| GXNOR / DST (1705.09283) V | ternary, no hidden weights, stochastic adjacent transitions | CIFAR-10 92.50 vs 92.88 |
| LNS-Madam (2106.13914) V | master-free, log-domain multiplicative update | ResNet-50 76.14 vs 76.38 |
| ECO (2601.22101) V | master-free; **quantization residual injected into momentum** | FP8 800M loss 2.540 vs master 2.531 (naive SR 2.947); INT4 MoE-16B fine-tune matches |
| BF16+SR (2502.20566) V | fp32 master | 6.7B ppl 10.05 vs 10.11 |
| DQT (2412.04787) V | low-bit weights updated by SR | 1B ternary: curves only, simulated in fp32 |
| QAR (2609.21834) PV | shadow weights; codes updated by a "crossing law" | LLM post-training; **read in full** |

**Theory that explains our carry result.** Li et al. 2017 (1706.02379, V): pure stochastic
rounding of weights stalls on non-convex problems, and a float latent's annealing behaviour is what
rescues BinaryConnect. A deterministic carry that keeps its remainder *is* a fixed-point latent in
other coordinates. That predicts our carry-sgd = STE+SGD (92.05 vs 92.03) and predicts that
reset-on-flip and stochastic-transition methods (BOLD, GXNOR, DQT) should trail.

### Forward-only (zeroth-order, perturbation) training

| Work | Mechanism | Scale / result | Discrete weights? |
|---|---|---|---|
| Dust (qlabs, MIT) V | per-token Gaussian noise on activations; centered Δloss credit with decay; outer product with input | GPT 2M–243M, ≤20M tokens; below backprop at 0.1–1M tokens; 20M "win" is a fitted asymptote | no |
| EGGROLL (2511.16652) V | low-rank weight-space ES | int8 minGRU LM trained **integer-only**, 3.40 bits/byte; no backprop comparison for pretraining; ~180× GPU-hours | int8 |
| nano-egg (GPL-3.0) V, code read | antithetic sign fitness; int32 `Z = Σ sign·sign`; int8 weight ±1 when `|Z| > θ√pop`; **Z reset every step** | toy | int8 |
| ElasticZO-INT8 (2501.04287) V | integer ZO, sign-only Δloss, 1-bit SR updates in place | LeNet MNIST **89.8 vs BP 98.8** | int8 |
| Poor Man's Training (2411.05873) V | INT8 node perturbation × input outer product, layer-wise | MCU fine-tuning matches BP; needs norm-aware scaling | int8 |
| QuZO (2502.12346) V | doubly-stochastic quantized perturbations, unbiased | LLaMA-2 13B low-bit fine-tuning | int4/int8 |
| TerMeZO (2609.33548) V | SPSA on the 5% of fp latents nearest a ternary boundary | BitNet-2B4T fine-tune SST-2 93.1 vs BP 94.8 | ternary, fp latents |
| Ren et al. 2023 (2210.03310) V | activity-perturbation forward gradient | global loss **fails on ImageNet (98.8% error)**; local losses rescue to 58.4 | no |
| Werfel/Xie/Seung 2005 PV | node perturbation slower than SGD by #outputs; weight perturbation by #outputs × fan-in | theory | — |
| Hiratani et al. 2022 PV | node perturbation unstable under model mismatch (weight diffusion) | theory + nets | — |
| Dalm et al. (2310.00965) V | node perturbation + input decorrelation | Tiny ImageNet 13.4 vs BP 14.5 at 100 samples | no; σ² = 1e-6 |
| MeZO, HiZOO, SubZero, DiZO, ZO-Bench, ES-at-Scale | weight-space SPSA/ES | fine-tuning only (low intrinsic dimension) | no |

Hardware communities (memristor perturbation, stochastic MTJ synapses, reward-gated STDP with
leaky counters) cover "noise + reward + local leaky counter + threshold write" in pieces, on toy
tasks (PV).

### Quantization and serving

| Project | Covers | Integer end-to-end? | PyTorch-native? |
|---|---|---|---|
| bitnet.cpp (MIT, 41k★) V | serve: CPU I2_S/TL1/TL2, one CUDA W2A8 GEMV | int2×int8 GEMM; fp norms/attention | no |
| T-MAC (MIT, dormant) V | serve: CPU/NPU LUT GEMM | LUT, int8 or fp activations | no |
| BitBLAS (MIT, **archived 2026-02**) V | CUDA W2A8 | kernel only | `bitblas.Linear` |
| llama.cpp TQ1_0/TQ2_0/Q2_0 V | quantize + serve; TQ on CPU/Metal/Vulkan, Q2_0 on CUDA | no | no |
| PrismML fork (PQ2_0, PTQ1_0) V | serve, all GPUs | no | no |
| vLLM v0.31 V | **no ternary**; GGUF moved to an out-of-tree plugin; CPU BitNet PR #52195 open | no | plugin API |
| mlx-lm BitLinear V | Metal serve | fp accumulation | MLX |
| ExecuTorch / ONNX Runtime V | int1–8 weights × int8 activations / 2-bit `MatMulNBits` | partial | export |
| torchao (v0.18) V | ternary QAT, int8 training, `quantize_`, FSDP2, compile; **no ternary PTQ, no packed ternary inference subclass** | no (fp master) | **yes** |
| Brevitas, HQQ (1.58 unpacked), AutoRound (BitNet→GGUF) V | QAT / PTQ pieces | no | yes |
| PT²-LLM (Apache-2.0) U | ternary PTQ research code | — | research |

Integration lessons:
- **torchao is adoptable** because one call (`quantize_(model, Config)`) composes with compile,
  FSDP2, HF serialization, vLLM's `torchao` method and ExecuTorch export, via tensor subclasses.
- **bitsandbytes** declares ops with `torch.library.define` + `register_fake` + per-device
  `register_kernel`, which gives compile/export compatibility without forking PyTorch.
- **New GGUF type IDs split the ecosystem until upstreamed.** Bonsai `Q2_0` produced gibberish on
  older builds.

## 2. Gaps nobody fills (Tritium's positioning)

1. **Integer-only training of a real model.** Every ternary trainer keeps fp or bf16 masters with
   STE. No master-free ternary transformer has been trained from scratch with published numbers.
2. **One format across train → PTQ/QAT → serve.** Today a ternary model is re-packed at every hop
   (I2_S, TL1/2, TQ1_0, TQ2_0, Q2_0, PQ2_0, PTQ1_0, BitBLAS, MLX).
3. **A production ternary PTQ tool.** torchao, llm-compressor, AutoRound and GPTQModel have none;
   PT²-LLM is research code.
4. **GPU serving at scale.** vLLM and sglang have no ternary method; BitBLAS is archived.
5. **A PyTorch-native packed ternary tensor** that survives compile, export, FSDP2 and HF
   serialization.
6. **Integer norms, attention and softmax** in a served ternary LLM (only FPGA papers approach it).
7. **Extensible codes.** Every runtime hard-codes one ternary layout; none has a registry for
   ladders, multi-plane or coded planes (our Golay result needs exactly this).

**Interoperate, don't compete:**
- Read and write GGUF Q2_0/TQ2_0/TQ1_0, and upstream kernels.
- Ship a vLLM `register_quantization_config` plugin modelled on vllm-gguf-plugin.
- Expose the ternary type as a torchao-compatible tensor subclass + config, with
  bitsandbytes-style `torch.library` ops.
- Accept onebitllms/Axolotl/torchao BitNet checkpoints.
- Treat bitnet.cpp and T-MAC as CPU baselines.
- Do not reuse code from GPL projects (QuIP#, QTIP, nano-egg, NITRO-D, EGGROLL); cite their ideas.

## 3. What our training ideas add, and what they don't

Do **not** claim as novel:
- "No master weights" — WAGE, Bop, GXNOR, LNS-Madam, ECO and torchao int8 all precede it.
- "The residual lives in the optimizer" — ECO did it at 16B for FP8/INT4.
- "Node perturbation × input outer product on integer weights" — Poor Man's Training, two years
  before Dust.
- "Integer threshold on accumulated noise evidence" — nano-egg, with a per-step reset.

Defensibly open, as composition and regime:
1. **Ternary + integer residual-preserving carry as the only weight state + integer evidence path +
   zero-entry/exit thresholds**, with the measured finding that carry width and model capacity,
   not optimizer richness, set quality.
2. **The fused forward-only rule:** activation-space noise credit accumulated straight into a
   *persistent*, leaky integer carry, where a flip is a sequential test across steps (nano-egg
   discards evidence each step; Bop and BOLD use gradients).
3. **Any backprop-free or master-free ternary transformer trained from scratch**, at any scale with
   published numbers.

Must read before claiming anything:
- 2610.16298 (integer-only training, four days old);
- QAR 2609.21834 (shadow-free code updates);
- Gradient-Free Training of QNNs 2410.09734 (PV; claims parity, mechanism unverified);
- full-text BOLD (accumulator type), KronZO (the only ZO pretraining baseline found);
- EGGROLL Appendix H update rule.

Known failure modes for the forward-only direction:
- **Dimension tax.** Node perturbation is slower than SGD by about the number of outputs per layer.
- **Global-loss activity perturbation failed on ImageNet** (Ren) without local losses; Dust's
  per-token population is its substitute.
- **Coarse integer or ternary noise leaves the linear regime.** Decorrelated NP used σ² = 1e-6;
  ElasticZO's sign was wrong about 5% of the time; QuZO needed doubly-stochastic rounding to stay
  unbiased.
- **Carry random walk.** Noise alone drives the carry like √n; the threshold must be set from the
  null variance (nano-egg's θ√pop), and the leak caps the window (Hiratani's instability).
- **Norm-aware scaling is mandatory** (Poor Man's Training did not converge without it).
- **Attention, LM head and residual scalars need special credit** (Dust).
- **Integer ZO lags even on LeNet** (ElasticZO 89.8 vs 98.8); every strong ZO result is
  fine-tuning.

## 4. Ablation ladder

### Why EAT-O has been hard to ablate

The EAT-O controller bundles at least six interacting mechanisms: rotating partial coverage,
truncated credit, sequential updates, EMAs cleared on transition, block-shared thresholds, and
cooldown. Changing one shifts the controller's timescales, so the 20-run coverage × credit
factorial resolved nothing (73.3 / 73.3 / 72.3 / 70.8%, every contrast interval spanning zero).

The carry line did not have this problem, because every new arm had a **degenerate setting that
reproduces its predecessor bit for bit**: `rich` with no options *is* `carry-sgd`, the capacity
base arm *is* rich-averaging, and the carry gradient *is* `trunc(g/2¹¹) == sensitivity`. That
property, not more seeds, is what removes confounding.

### Rules for every rung

1. **One mechanism per rung,** behind a knob whose identity value reproduces the previous rung
   bit-exactly. A unit test asserts the identity before any run.
2. **Three orthogonal axes, varied one at a time:**
   - gradient source G;
   - update rule U;
   - model capacity M.

   The anchor is G0 (exact STE gradient), U1 (carry-sgd) and M0 (784→128→10, T=1, fixed scales).
   Capacity is re-applied only at the end, so the 97.2% capacity effect is never mixed with an
   optimizer comparison.
3. **Estimator quality is measured offline, without training.** On frozen initial, mid and final
   checkpoints, record:
   - cosine to the exact gradient, by layer and population size;
   - sign agreement;
   - which weights the estimate would flip versus which the exact gradient would flip.

   This separates "is the gradient estimate good" from "does the optimizer use it well". Training
   runs then only answer the second question.
4. **Protocol (as the EAT-O campaign):**
   - tune on seed 101, then five paired seeds of 20,000 examples;
   - validation 1000–4999, with the official test set untouched;
   - paired t intervals, and comparisons below the noise floor reported UNRESOLVED;
   - report equal-data **and** equal-forward-pass accounting;
   - record flip rate, reversal rate, zero occupancy and carry saturation.

### Axis U: update rule (G0, M0 fixed)

| Rung | Change from previous | Identity knob | Status |
|---|---|---|---|
| U0 | fp32 STE + SGD | — | 92.03% ✓ |
| U1 | integer carry, remainder kept | gradient is the exact rational STE gradient | 92.05% ✓ |
| U1a | carry width i16 | — | 91.61% ✓ |
| U2 | + latent averaging | average off → U1 | 92.21% ✓ |
| U3 | remainder kept fraction ρ (ρ=0 is BOLD/Bop reset-on-flip) | ρ=1 → U1 | open |
| U4 | + leak λ on the carry | λ=0 → U1 | open |
| U5 | stochastic adjacent transitions, no carry (GXNOR/DST) | — (control) | open |
| U6 | ECO-style residual into momentum | momentum off → U1 | open |

**Explaining the EAT-O gap (92% → 73%).** Walk from U1 toward the EAT-O controller, adding one
controller feature at a time, each with an identity knob:
1. evidence cleared on transition (U3 with ρ=0);
2. EMA forgetting (U4);
3. rotating 4,096-weight coverage;
4. credit truncation before storage;
5. sequential vs snapshot updates;
6. block-shared thresholds;
7. cooldown.

Pre-register the order, then run the reverse order once to measure order-dependence. Each step's
loss change attributes part of the gap, which the factorial around the full controller could not
do.

### Axis G: gradient source (U1, M0 fixed)

| Rung | Change | Identity / check | Prior art |
|---|---|---|---|
| G0 | exact STE gradient | anchor | — |
| G1 | offline estimator diagnostic (no training): cosine and flip agreement vs population K | — | Dust's cos(K) law |
| G2 | node perturbation, Gaussian noise, clean-forward baseline, K draws per step | G1 cosine → c_max as K grows | Dust, Poor Man's |
| G3 | antithetic ± pairs (no clean forward) | same noise draws as G2 | OpenAI-ES, nano-egg |
| G4 | ternary noise {−1,0,+1} | matched σ to G3 | ElasticZO, QuZO |
| G5 | reward quantized to a few bits, then sign-only | full-precision reward → G4 | ElasticZO, nano-egg |
| G6 | control: weight-space SPSA at matched forward passes | — | MeZO, EGGROLL |

### Axis F: fusion (best G, U1, M0 fixed)

| Rung | Change | Identity |
|---|---|---|
| F0 | K draws averaged, then one carry step | the best G rung |
| F1 | the same draws applied one per step, flips deferred to the window end | **bit-exact to F0**: ternary weights are piecewise constant, so with flips deferred every forward sees the same model |
| F2 | flips allowed mid-window (true fusion) | the *only* change from F1 |
| F3 | + leak | λ=0 → F2 |
| F4 | carry reset every window (nano-egg's per-step z-test) vs persistent | tests the sequential-test claim |
| F5 | all-integer step: ternary noise, quantized reward, integer carry | composes G4/G5 into F3 |

F1 is the key rung. Because it is bit-exact to F0, the F1→F2 difference isolates what fusion
itself costs or buys: mid-window flips changing the model while evidence is still accumulating.

### Axis M: capacity, last

Re-apply the measured capacity levers (global scale ×1.5–1.75, width, planes) to the best U and F
rungs only. Capacity raised U2 from 92.2% to 97.2%, and that effect must not be mixed into any
optimizer or estimator comparison.

### Decision points

- **Fusion is worth pursuing** if F2 or F3 matches F0 at equal forward passes, and some G rung
  reaches ≥ 91.5% at a K whose compute is within 10× of G0.
- **Kill the forward-only direction** at this scale if no G rung exceeds 90% at K = 4096. This is
  ElasticZO's regime, and the literature predicts it.
- **Promote U-axis findings** that explain the EAT-O gap into the main EAT-O controller, replacing
  features that cost quality.
