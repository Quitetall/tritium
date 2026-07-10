# 0038 — SALT-aware distillation trainer  (serves: ADR 0020 step 3, the core loop)

## Goal
The "heal" loop: an fp teacher distills into a ternary student whose 2D weights are fp32 **latent
masters**, **SALT-quantized in the forward with STE**, updated by AdamW. Recover the
quantization gap that PTQ leaves (0036 measured ~0.5 rel-Frobenius at the floor).

## Reuse map (from exploration — nearly everything exists)
- `tritium-train` `Tape` (reverse-mode autograd), `AdamW` + `LrSchedule` + TOPT checkpoint,
  `ops::ste` (single-plane STE latent-master), `dense_matmul`/`matmul`/`mse`/`softmax_xent`
  (the last gives the **correct KL gradient** vs a soft teacher target — teacher entropy is a
  constant), `rmsnorm`/`rope`/`softmax`/`causal_mask`. Template: `qat_heal_gate.rs` (layerwise
  heal loop: rebuild tape → leaf(latent) → ste → matmul → mse → backward → AdamW on the latent).
- Inference (0035–0037): `load_hf` (fp teacher), `load_salt` (ternary student), `forward_dump`
  (per-layer activations for layerwise targets), the ppl harness (`salt_accuracy.rs` pattern).

## Gaps (what 0038 adds)
1. **Multi-plane SALT-STE** — single-plane STE exists; the SALT student quantizes the latent to
   T planes each forward. No multi-plane vjp today. **← THIS PLAN'S first primitive.**
2. **SiLU op** (forward+vjp) for a SwiGLU student's MLP (tape only has BitNet squared-ReLU).
3. **Teacher-logit cache** (`[seq, vocab]` or top-k) — today only last-row logits exist.
4. **End-to-end model tape** — both existing trainers are single-op-graph; no tape spans
   embeddings→N blocks→lm_head. (Layerwise-first per ADR 0020 avoids this initially.)
5. CPU-offload AdamW state (ADR 0016) — matters at 32B (plan 0040), not for the small proof.

## Steps
### Step 1 — Multi-plane SALT-STE op + tape node (this commit)
`ops::ste::salt_quantize_forward(wf, rows, cols, t)` = the T-plane residual quantize (round),
per-row AbsMean, returning the dense reconstruction `Ŵ = Σ_p s_p·trit_p`; `salt_quantize_vjp` =
straight-through (`gWf = grad_out` — the T-plane reconstruction tracks `Wf`, so pass the gradient
to the latent). `Tape::salt_ste(wf, rows, cols, t)`. **Gate (TDD):** forward equals an independent
residual-expansion reference (and `tritium_format` dequant at matching config); a toy layerwise
distillation (`dense_matmul(x, salt_ste(latent)) → mse(target)`, AdamW) **reduces the output MSE
below the naive PTQ** (recovers the quantization gap) — the atomic proof the SALT-STE loop learns.

### Step 2 — SiLU op (forward + vjp) + gradcheck
`ops::act::silu` + `Tape::silu`; Gate-C finite-difference gradcheck. Enables a SwiGLU student.

### Step 3 — Layerwise SALT distiller (reusable)
A `distill_projection(fp_weight, calib_input, teacher_output, t, steps) -> healed_latent` using
Step 1. Gate: on a real SmolLM2 projection, healed-latent output-MSE to the fp target drops ≥90%
vs naive SALT PTQ.

### Step 4 — Model-level recovery (the ADR gate, small scale)
Heal every projection of a small model (SmolLM2), install via `replace_weights`, and measure
**perplexity recovery** through the inference runner: healed ternary ppl ≪ un-healed PTQ ppl,
approaching fp. Uses `forward_dump` for per-projection calib inputs.

### Step 5 (continuation) — end-to-end logit-KL
Teacher-logit cache + the whole-model tape (or a `softmax_xent`-against-teacher fine-tune) for
true end-to-end distillation. Heavier; may split to 0038b. `T²` temperature + a real `kl` op here.

## Gate (this plan's first shippable increment = Step 1)
`salt_ste` forward matches the residual-expansion reference; the toy distillation recovers the
quantization gap (final MSE < PTQ MSE). Gradcheck N/A (STE is intentionally biased — the gate is
"the loop learns"). fmt + clippy `-D` clean; `tritium-train` tests green.

## Verification
`cargo test -p tritium-train` (Step 1 gate) green; Gate-C gradcheck for silu (Step 2). Each step
reviewed with the code-reviewer subagent; push via the deploy key.
