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
| Shalby, Pittorino & Roveri (2610.16298, 2026-10-12) V | INT8 mantissa + power-of-two exponent for every tensor, integer SGD+momentum, integer LUT softmax/GELU/LN, **SR everywhere** | ResNet-50 ImageNet 76.4 vs 76.7; ViT-Ti 88.0 vs 90.9; INT4 weights 70.5 (CIFAR-10); no ternary |
| Bop (1906.02107) V | latent weight → fp EMA + threshold flip (binary) | Bi-Real ImageNet 56.6 vs 56.4 |
| BOLD (2405.16339) V | per-weight accumulator, xnor threshold flip, **reset** (binary) | ImageNet 66.9 / 70.0 distilled; first/last layers fp |
| GXNOR / DST (1705.09283) V | ternary, no hidden weights, stochastic adjacent transitions | CIFAR-10 92.50 vs 92.88 |
| LNS-Madam (2106.13914) V | master-free, log-domain multiplicative update | ResNet-50 76.14 vs 76.38 |
| ECO (2601.22101) V | master-free; **quantization residual injected into momentum** | FP8 800M loss 2.540 vs master 2.531 (naive SR 2.947); INT4 MoE-16B fine-tune matches |
| BF16+SR (2502.20566) V | fp32 master | 6.7B ppl 10.05 vs 10.11 |
| DQT (2412.04787) V | low-bit weights updated by SR | 1B ternary: curves only, simulated in fp32 |
| QAR (2609.21834) V | no shadow weight; integer codes moved by a Bernoulli "crossing law", AdamW moments kept | post-training only; 2-bit Llama-3.2-3B C4 ppl 359 → 202; loses to QAT at 4 bits |

**Theory that explains our carry result.** Li et al. 2017 (1706.02379, V): pure stochastic
rounding of weights stalls on non-convex problems, and a float latent's annealing behaviour is what
rescues BinaryConnect. A deterministic carry that keeps its remainder *is* a fixed-point latent in
other coordinates. That predicts our carry-sgd = STE+SGD (92.05 vs 92.03) and predicts that
reset-on-flip and stochastic-transition methods (BOLD, GXNOR, DQT) should trail.

### Forward-only (zeroth-order, perturbation) training

| Work | Mechanism | Scale / result | Discrete weights? |
|---|---|---|---|
| Dust (qlabs, MIT) V | per-token Gaussian noise on activations; centered Δloss credit with decay; outer product with input | GPT 2M–243M, ≤20M tokens; below backprop at 0.1–1M tokens; 20M "win" is a fitted asymptote | no |
| EGGROLL (2511.16652) V | low-rank weight-space ES | int8 minimal-GRU LM pretrained **integer-only, backprop-free**: 3.40 vs 3.58 bits/byte against an SGD Transformer; ~180× GPU-hours at N = 2²⁰ | int8 |
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
| KronZO (GERAD G-2025-44) V | Kronecker-factored perturbations, directional best-of-q update | GPT-2 125M pretraining: 6.95 vs SGD 3.50 training loss at 25× wall time | no |

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

## 3. What our training ideas add, after full-text reading

Nine papers were read in full on 2026-10-16:
- integer and master-free training: 2610.16298 and ECO 2601.22101;
- flip and shadow-free update rules: QAR 2609.21834, Gradient-Free Training 2410.09734, and BOLD
  2405.16339;
- forward-only training: KronZO (GERAD G-2025-44), EGGROLL 2511.16652 (App. G–H plus the nano-egg
  code), and TerMeZO 2609.33548.

### Claim verdicts

| Claim | Verdict | Closest prior art |
|---|---|---|
| C1. Ternary weights with an integer, **remainder-keeping** carry as the only weight state; threshold flip | **Open, narrowly — and keep-vs-reset is NOT a measured advantage** (rules-20261016: reset +118 [−47, 284], unresolved) | BOLD: real-valued accumulator, **reset** on flip, binary. ECO: residual inside fp32 momentum. QAR: memoryless Bernoulli, no carry. 2610.16298: integer, discards the residual, relies on SR. None keeps the remainder or ablates keep vs reset. |
| C2. Carry width and capacity, not optimizer richness, set quality | **Open** (no paper ablates accumulator width) | BOLD needs 4× width for fp parity on ImageNet, which supports the capacity half. |
| C3. Explicit zero-entry/exit thresholds | **Open** | QAR: symmetric cell moves. GFT: ±1 steps. BOLD: binary. |
| C4. Fused forward-only rule: activation-noise credit into a **persistent, leaky** integer carry; flip as a cross-step sequential test | **Partially pre-empted** | EGGROLL App. H.3: antithetic ternary fitness, int32 accumulator, z-threshold, ±1 int8 step, no master or optimizer state, **but the accumulator is re-zeroed every step** (single-step test) and noise is in weight space (rank-1, acting on the output). TerMeZO: flip when an **fp** latent crosses the boundary. Open: persistence + leak, activation-space per-token noise, ternary weights. Cite EGGROLL H.3 as the closest. |
| C5. Master-free or backprop-free ternary **transformer** from scratch | **Open for ternary transformers only** | EGGROLL pretrains a backprop-free, master-free **int8 GRU** from scratch (3.40 vs 3.58 bits/byte vs an SGD Transformer, ~180× GPU-hours, N = 2²⁰). ECO pretrains master-free **FP8** transformers to 2.1B. GFT trains a ternary nanoGPT from scratch, but hybrid with fp embeddings and head, and badly (loss 2.55 vs fp 1.57). "First backprop-free low-bit LM" **cannot** be claimed. |
| C6. The carry predicts which weights cross a cell boundary | **Overlaps QAR** | QAR's residual phase θ and crossing law p = η\|u\|/s are the shadow-weight analysis of the same quantity. Our carry is its persistent integer, master-free counterpart. Cite and differentiate. |
| C7. Offline estimator cosine vs population size on **discrete** weights | **Open** (one targeted search found none) | cosine curves exist only for continuous nets (Ren 2023; Fournier 2023, unverified). |

### Landmines

1. **The carry is a master weight in different units.** ECO App. A proves a stored residual
   equals master-weight SGD exactly. An i32 carry costs 4 B/weight, the same as an fp32 master.
   Never pitch the carry on memory except at i16 or narrower. Frame it honestly as a fixed-point
   master / error-feedback buffer. Its case is integer-only arithmetic, multiply-free compute and
   narrow width. Our own result (carry-sgd = STE+SGD) says exactly this.
2. **Stochastic rounding is the field's default, and we don't use it.**
   - 2610.16298: deterministic rounding costs 5.4 points (89.3 vs 94.7 on ResNet-18/CIFAR-10).
   - ECO: naive no-master with round-to-nearest diverges at every scale.

   Our answer is that keeping the remainder makes deterministic rounding exact. It needs an SR
   ablation (rung U7 below).
3. **Narrow carries can quantize credit away.**
   - QAR B.4: BF16 shadows stop crossing late in training (sub-half-ulp updates are lost).
   - ECO §3.4: losing the residual gives a stationary error ∝ 1/η, which diverges as the learning
     rate decays.

   An i16 carry that clips or underflows falls into that regime. Log underflow and saturation
   counts with every narrow-carry result.
4. **Keep-remainder may be read as "just error feedback".** BOLD's convergence proof models
   accumulate-and-reset as EF-signSGD. Pre-empt this with the measured keep-vs-reset ablation
   (rung U3) and say plainly that keep-remainder is EF with an identity compressor on the
   residual.
5. **QAR finds the ternary phase is not uniform** (it concentrates mid-cell, B(ρ) < ρ). Any
   uniform-phase assumption in our flip-rate reasoning must be checked against measured carry
   histograms.
6. **Forward-only pretraining is very costly or fails:**
   - KronZO GPT-2 125M: final training loss 6.95 vs SGD 3.50, at 25× wall time per step. At 1.5B:
     8.21 vs 2.85. The gap widens with scale.
   - EGGROLL: population ~10⁶ and ~180× GPU-hours to beat a weak SGD baseline on a 6L-256D model.
     Population 2 fails, and a larger population needs a larger data batch (≈ N/4⁶).
   - Direct ZO on quantized ternary weights "barely improves" (QuZO on Falcon-E: 14.71 vs 14.0
     GSM8K).
   - ES needs σ = o(d^-½) to linearise, and coarse integer or ternary noise cannot get there.

   Treat the forward-only direction as a long shot. Its early kill criterion is real.
7. **Weak baselines discredit results.** KronZO and EGGROLL compare against plain SGD. Ours must
   include a tuned AdamW + STE ternary QAT baseline.
8. **Quote GFT numbers with care.** Its tables are internally inconsistent (Imagenette Hybrid-1
   4-bit is 84.9 in the main text, 95.6 in the appendix). Its NP-hardness theorem is worst-case
   binary exact separability and undercuts nothing we claim.

### Required comparators

- **Update rules:** Bop, DQT (= QAR with γ = 0), QAR-γ, and ECO + SR.
- **Naive master-free:** naive no-master + SR.
- **Integer baselines:** 2610.16298-style INT8 + SR, and WAGE.
- **Forward-only:** EGGROLL / nano-egg integer ES (GPL-3.0; run it, don't copy it), MeZO, KronZO's
  directional update, TerMeZO's boundary mask.
- **Strong first-order baseline:** AdamW + STE ternary QAT.

### Citations reviewers will expect

ECO, 2610.16298, NITI, WAGE/WAGEUBN, SWALP, Ghaffari 2022, APT (Huang 2022), Lin 2022 (256 KB
on-device training), error feedback (Seide 2014; EF21), Ozkara 2025, Li et al. 2017, Bop, BOLD,
GXNOR, QAR, EGGROLL, KronZO, TerMeZO, MeZO, Ren 2023, Werfel/Xie/Seung, Hiratani 2022, Dust.

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
| U3 | remainder kept fraction ρ (ρ=0 is BOLD/Bop reset-on-flip) | ρ=1 → U1 | **done**: ρ=0 91.73% (unresolved vs U1), ρ=0.5 92.08%; reset cuts transitions 10× |
| U4 | + leak λ on the carry | λ=0 → U1 | **done**: monotone cost (2⁻¹⁶ +206, 2⁻⁸ +4,404, 2⁻⁴ collapses) |
| U5 | stochastic adjacent transitions, no carry (GXNOR/DST) | — (control) | open |
| U6 | ECO-style residual into momentum | momentum off → U1 | open |
| U7 | stochastic rounding of the carry step instead of an exact remainder | SR off → U1 | **done**: rounding mode irrelevant at 2¹⁶ and 2¹² units/trit |
| U8 | carry width i16 → i8, logging underflow and saturation counts | width i32 → U1 | **done**: i8 + SR 91.45% at 1 B/weight; i8 floor collapses (65–72%) |
| — | external comparators at matched tuning budget: Bop, DQT, QAR-γ, ECO+SR, naive no-master+SR, WAGE | — | open |

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
| G1 | offline estimator diagnostic (no training): cosine and flip agreement vs population K | — | **done** (estimator-20261016): ternary antithetic noise ≥ Gaussian; σ ≥ 1 quantization step; sign-only biased; trained model needs ~4–8× more draws |
| G2 | node perturbation, Gaussian noise, clean-forward baseline, K draws per step | G1 cosine → c_max as K grows | Dust, Poor Man's |
| G3 | antithetic ± pairs (no clean forward) | same noise draws as G2 | OpenAI-ES, nano-egg |
| G4 | ternary noise {−1,0,+1} | matched σ to G3 | ElasticZO, QuZO |
| G5 | reward quantized to a few bits, then sign-only | full-precision reward → G4 | ElasticZO, nano-egg |
| G6 | control: weight-space SPSA at matched forward passes | — | MeZO, EGGROLL |
| G7 | control: EGGROLL-style rank-1 weight noise with a per-step z-test (nano-egg, run as an external binary, GPL) | — | EGGROLL H.3 |

### Axis F: fusion (best G, U1, M0 fixed)

| Rung | Change | Identity | Status |
|---|---|---|---|
| F0 | K draws averaged, then one carry step | the best G rung | **done**: K=4 88.03%, K=16 89.96% at 20k examples |
| F1 | the same draws applied one per step, flips deferred to the window end | as built: one draw on each of W examples with the forward's trits frozen (a minibatch of W), not K draws on one example, so it is **not** bit-identical to F0; `--defer 1` is the immediate step bit for bit | **done**: deferring gives no benefit (+1,478, unresolved) |
| F2 | flips allowed mid-window (true fusion) | the *only* change from F1 | **done**: K=1 at equal noisy evals ties K=4 (−279) and beats K=16 (−537 [−988, −87]); 90.11% at 16× data |
| F3 | + leak | λ=0 → F2 | **done**: hurts (2⁻¹⁶ +1,632; 2⁻¹² +5,555) |
| F4 | carry reset every window (EGGROLL H.3 / nano-egg per-step z-test) vs persistent | tests the sequential-test claim, the one element EGGROLL does not have | partly done: reset-on-flip unresolved (+1,133), keep 50% hurts; a true per-step z-test arm is still owed |
| F5 | all-integer step: ternary noise, quantized reward, integer carry | composes G4/G5 into F3 | **done by construction**: the fused default is all-integer (ternary noise, integer credit, integer carry) |

As planned, F1 was to be bit-exact to F0 and isolate fusion. As built (2026-10-16) the cleaner
test of fusion turned out to be **equal noisy evaluations**: K=1 on 4× or 16× the examples vs K=4
or K=16 on 1×. Both variants spend the same estimator compute; the only difference is whether the
draws are averaged inside a step or across steps by the carry. F1 (deferred flips) is reported as
its own control.

Results so far:
- `docs/experiments/eat-o/rules-20261016/results.md` (U3, U4, U7, U8);
- `docs/experiments/eat-o/estimator-20261016/results.md` (G1);
- `docs/experiments/eat-o/fused-20261016/results.md` (F0–F5 and the training half of G).
- `docs/experiments/eat-o/deep-20261016/results.md` (width, depth, timing: from-scratch data
  multiplier ~60–250×, grows with width; fused ≈ backprop per example);
- `docs/experiments/eat-o/recovery-20261016/results.md` (recovering a ternarized fp model:
  fused reaches 93.8% from 82.2% in 2k examples and plateaus ~95.3%; a 20% near-boundary
  sparse carry ties dense backprop recovery).

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
