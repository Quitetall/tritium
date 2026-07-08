# Capstone cascade — ADR 0020 SALT-distillation, the full plan set

The milestone map from the **keystone** (general inference) to the **binding gate** (a 27–35B
SOTA model ternarized to ≤1% ppl vs its fp16 teacher). Strategy lives in
[ADR 0020](../adr/0020-v1x-salt-distillation-capstone.md); this file sequences the **tactical
plans** that execute it, with each plan's deliverable, numeric exit gate, dependencies, effort,
and risks. Individual plans are fleshed out to executable detail **just-in-time** (0035 is
written; the rest carry enough here to start when their deps land — do not pre-write them all,
per the ROADMAP loop).

## The spine (one line)

```
0035 run fp  →  0036 run ternary  →  0037 reach Qwen-family arch
                     ↓                        ↓
                0038 distill (the loop)  ← 0039 real sensitivity + adaptive growth
                     ↓
                0040 scale to 32B  →  0041 CAPSTONE: 32B → ≤1% ppl  →  Qwen3.6 headline → launch
```

## Dependency graph

```
0035 (fp inference, KEYSTONE) ─┬─> 0036 (SALT multi-plane loader) ─┐
                               ├─> 0037 (QKV-bias + QK-norm) ──────┤
                               │                                   v
                               └────────────────────────────> 0038 (distillation trainer)
                                                                   │
                                                    0039 (Fisher sensitivity + growth)
                                                                   │
                                        0037 ─────────> 0040 (scale to 32B) ─> 0041 (CAPSTONE)
```

`0036` and `0037` are **parallelizable** (both depend only on 0035). `0038` needs `0035`+`0036`
(the student forward is a SALT multi-plane forward). `0039` refines `0038`. `0040` needs
`0037`+`0038`. `0041` needs everything.

---

## The plans

### 0035 — General fp inference (KEYSTONE) · **written**
- **Deliverable:** config-driven runner (`ArchSpec`, `from_hf_config`, SwiGLU MLP, untied head,
  `load_hf`/`from_hf`) — runs a standard Llama-arch fp model from `config.json` + safetensors.
- **Gate:** SmolLM2-135M greedy-token-exact vs `transformers`; last-row logit rel-err < 1e-3;
  no BitNet regression. *(Full spec: `docs/plans/0035-general-inference-engine.md`.)*
- **Deps:** none. **Effort:** the long pole (~1–3 mo incl. debugging real-model numerics).
  **Risk:** RoPE/RMSNorm/GQA numeric parity; the first non-BitNet forward.

### 0036 — SALT multi-plane inference loader
- **Deliverable:** run what `quantize` emits. A `Projection` path holding **T residual ternary
  planes** (SALT bundle / SALT-GGUF), forward = the existing single-plane mpgemm **looped T
  times** with per-plane scales accumulated (ADR 0001 §"the existing kernel, looped" — no new
  kernel). `ModelWeights::load_salt(dir)`.
- **Gate:** (a) a SALT-quantized SmolLM2 runs; its logits match the dequant-to-dense reference
  within tol; (b) a **multi-plane accumulate == Σ dequant** bit-parity unit test (T=1,2,3).
  T=1 must be byte-identical to the current single-plane path.
- **Deps:** 0035. **Effort:** ~1–2 wk. **Risk:** per-plane scale accumulation order/numerics;
  device-resident multi-plane on CUDA (CPU first, GPU parity follows).
- **Why it matters:** the *student forward* in distillation IS this path; and it's what delivers
  the VRAM win at inference (native planes, not dequant-to-dense).

### 0037 — Arch extensions: QKV-bias + QK-norm (Qwen-family)
- **Deliverable:** optional q/k/v additive **bias** (Qwen2/2.5) and **QK-norm** (per-head RMSNorm
  on Q,K — Qwen3), gated by the `ArchSpec` flags 0035 already carries.
- **Gate:** a small **Qwen2.5** model (has QKV-bias) token-exact vs `transformers`; a small
  **Qwen3** model (QK-norm) token-exact. Standard-no-bias path unchanged.
- **Deps:** 0035. **Effort:** ~1 wk. **Parallel with 0036.** **Risk:** low (small, well-specified
  ops); QK-norm placement (pre-RoPE) is the one gotcha.
- **Why it matters:** the likely **32B gate model** and the **Qwen3.6** headline are Qwen-family.

### 0038 — SALT-aware distillation trainer (the core loop)
- **Deliverable:** the "heal" loop as ADR 0020 specifies. Each 2D weight = fp32 **latent master**
  θ; forward **SALT-quantizes θ** and runs the multi-plane student (0036); **SALT is the STE**
  (identity backward → θ); **AdamW with CPU-offloaded optimizer** (ADR 0016 economics). **Teacher
  cache:** offline top-k logits (+ optional hidden states) from the fp oracle (external HF or
  in-Tritium 0035). **Loss:** `KL(teacher‖student) + λ·hidden-MSE`; **layerwise first** (the
  `qat_heal_gate` pattern, embarrassingly parallel), then **end-to-end**.
- **Gate (proof-of-loop at small scale):** distill **SmolLM2-135M/360M** against its own fp
  teacher; student recovers to **≤1% ppl** at some avg bpw on a held-out set. Gradient check the
  STE path; loss monotonically decreases; resume==uninterrupted (checkpoint).
- **Deps:** 0035, 0036; reuses `tritium-train` (STE autograd, AdamW, checkpoints) + the
  distillation-loss primitives. **Effort:** ~3–5 wk (the heaviest). **Risks:** STE stability at
  1.58-bit; hidden-state matching scale/weighting; activation memory (→ gradient checkpointing);
  throughput of the custom autograd.

### 0039 — Real sensitivity + adaptive plane growth
- **Deliverable:** grad-based **diagonal Fisher** per tile (accumulated `E[(∂L/∂w)²]`) →
  `Sensitivity::Custom`, feeding the SALT allocator; a **periodic plane-growth policy** — grow a
  tile's plane count where `error × sensitivity` stays high, under a bpw/quality target (the
  "add ternary params where accuracy degrades" lever).
- **Gate:** on the small distill, **Custom-Fisher beats Uniform/Energy at fixed bpw** (lower KL),
  AND adaptive growth reaches ≤1% ppl at **lower avg bpw** than uniform growth. (Directly answers
  the shipped finding that Energy ≈ Uniform.)
- **Deps:** 0038. **Effort:** ~2–3 wk. **Risk:** Fisher estimation noise; growth
  oscillation/non-convergence; re-allocation churn mid-training.

### 0040 — Scale plumbing (32B-ready)
- **Deliverable:** **streaming `quantize` writer** (today it buffers all rows in RAM — the wall
  past ~100B); **streaming GPU weight load**; **gradient checkpointing** config for 32B;
  optimizer CPU-offload wired at scale; teacher-cache sharding. Multi-GPU **only if** the target
  exceeds one card (a 32B ternary student ≈ 6–12 GB → single 24 GB GPU for inference; the
  *training* footprint — fp latents + Adam state — is what needs the offload/checkpointing).
- **Gate:** a standard **32B** fp model loads + runs (0035/0037) on the available GPU; **one
  distill step executes on 32B within the memory budget** (slow is fine).
- **Deps:** 0037, 0038. **Effort:** ~2–4 wk. **Risk:** memory/throughput at 32B on available
  hardware (likely rented boxes); custom-autograd scaling vs PyTorch.

### 0041 — CAPSTONE: 32B → ternary ≤1% ppl (the binding gate)
- **Deliverable:** full distillation of the standard **32B** gate model to **≤1% ppl** vs fp16;
  the **quality-vs-bpw curve** (1.58→~3); **measured VRAM reduction** (~1/5–1/8); the ternary
  student running **end-to-end in Tritium**. Then the **Qwen3.6-27B headline** extension (needs
  the SSM/`linear_attn` arch — a follow-on sub-plan, *stretch not blocker*).
- **Gate = ADR 0020 Definition of Done.** Passing it **unblocks the public launch** (task #45).
- **Deps:** all. **Effort:** **compute-bound** — real GPU-hours (rented); calendar depends on
  distill convergence. **Risk:** ≤1% may land at ~2.5–3 bpw (accepted, per ADR); Qwen3.6 SSM/MoE
  is a genuine further lift (kept off the binding gate).

---

## Cross-cutting

### De-risking model ladder (cheap rung before each expensive one)
```
SmolLM2-135M (fp, 0035)                     — proves the config-driven forward
  → SmolLM2-360M/1B untied (0036, 0038)     — proves SALT-run + the distill LOOP, small & cheap
  → Qwen2.5-small (0037: QKV-bias)          — proves the Qwen-family attention
  → Qwen3-small (0037: QK-norm)             — proves QK-norm
  → standard 32B (0040, 0041)               — the BINDING gate
  → Qwen3.6-27B (SSM hybrid)                — the HEADLINE (stretch)
```
Each rung is a token-exact / ≤1%-ppl checkpoint on a cheap model before committing GPU-hours to
the next. No expensive run starts before its method is proven small.

### Teacher-caching strategy
The teacher is **frozen** → cache once. Run the fp oracle over the distill corpus (external
HF/PyTorch, or in-Tritium via 0035) and store **top-k logits** (+ optional hidden states) to
disk. Student training reads the cache — teacher cost is paid once, not per epoch, and the
teacher arch need not run at teacher-quality inside Tritium to *train* the student.

### Compute / hardware plan
- **0035–0039 (small models):** local 4090 / CPU. Cheap.
- **0040–0041 (32B distill):** real GPU-hours → **rented boxes** (Thunder / Hot Aisle MI300X /
  RunPod, per prior fenced sessions). The quality-vs-bpw curve is **cheap to sample at low bpw,
  expensive to push to ≤1%** — budget the tail.
- Inference of the ternary 32B student fits a **single 24 GB card**; only *training* state needs
  the offload/checkpointing.

### Quality-vs-bpw methodology (the honest artifact)
Report **ppl + KL vs the fp teacher across avg bpw** (1.58 → ~3). The **≤1% point is the
headline**, but the **curve is the deliverable** — a single-point "≤1% at 1.58 bpw" claim would
be dishonest; 1.58→near-fp16 realistically lands at ~2.5–3 bpw (still ~1/5 VRAM).

### Effort / sequencing summary
| Plan | Effort | Deps | Parallel | Hardware |
|------|--------|------|----------|----------|
| 0035 fp inference (KEYSTONE) | ~1–3 mo | — | — | CPU/4090 |
| 0036 SALT multi-plane loader | ~1–2 wk | 0035 | with 0037 | CPU/4090 |
| 0037 QKV-bias + QK-norm | ~1 wk | 0035 | with 0036 | CPU |
| 0038 distillation trainer | ~3–5 wk | 0035, 0036 | — | 4090 |
| 0039 Fisher + adaptive growth | ~2–3 wk | 0038 | — | 4090 |
| 0040 scale to 32B | ~2–4 wk | 0037, 0038 | — | rented GPU |
| 0041 CAPSTONE run | compute-bound | all | — | rented GPU |

Critical path ≈ **0035 → 0036 → 0038 → 0039 → 0040 → 0041**; 0037 rides alongside. The **keystone
dominates**; once it lands, the small-model loop (0036+0038) proves the method before any 32B
GPU-hours are spent.

## Milestone-level risks
- **Keystone (0035) is the long pole** — everything waits on it; front-load it.
- **1.58 → ≤1% is aggressive** — adaptive growth (0039) to ~2.5–3 bpw is the plausibility lever;
  the curve is the honest framing.
- **32B distill = rented GPU-hours** — the capstone is compute-bound, not code-bound, at the end.
- **Custom autograd at 32B** — a real systems risk (memory/throughput vs PyTorch); 0040 de-risks
  with a single-step-fits check before the full run.
- **Qwen3.6 SSM/MoE is the headline, not the blocker** — the binding gate is a standard 32B; the
  hybrid arch is a demonstration after the method is proven (schedule insurance).
