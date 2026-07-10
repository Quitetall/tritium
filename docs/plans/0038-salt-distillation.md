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

### Step 4 — Model-level recovery (the ADR gate, small scale) — EXECUTION-READY DESIGN

**Steps 1–3 shipped** (`93e7888`/`f899502`/`02e10a0`): SALT-STE (92.6% atomic recovery), SiLU,
and end-to-end tiny-SwiGLU distillation (KL 1.87→1.77). Step 4 = the real-model perplexity gate.

**Key enabler (no runner change needed):** `forward_dump` captures `embedding` + `hidden_states`
(the residual **after** each block). So layer `li`'s q/k/v input is reconstructable:
`input_li = (li==0 ? embedding : hidden_states[li-1])`, `attn_norm_out_li = rmsnorm_rows(input_li,
block[li].attn_norm, eps)`. That covers q/k/v for **every** layer with zero capture-plumbing.
(gate/up need `ffn_norm_out = rmsnorm(input_li + attn_out_li, ffn_norm)`; `attn_out_li` is only
dumped for layer 0 — so v1 heals **q/k/v across all layers** and leaves o/gate/up/down at PTQ.)

**Test:** `crates/tritium-nn/tests/salt_heal_ppl.rs` (`#[ignore]`d; SmolLM2-135M; deps
tritium-train + tritium-quantize, both dev-deps of tritium-nn). Reuses `perplexity(runner,
eval_ids)` (teacher-forced, salt_accuracy.rs pattern) + `ste::{salt_quantize_forward, salt_ste}` +
`AdamW`.

1. `load_hf(smollm2)` → fp runner; `eval_ids` = the committed `smollm2_ref.json` prompt (or a
   fixed token list). `ppl_fp = perplexity(fp, eval_ids)`.
2. `forward_dump(eval_ids)` → capture `embedding`, `hidden_states`. Compute `attn_norm_out_li`
   per layer (above).
3. **Build-with-transform helper** `build_model(fp_weights, f: Fn(name, &[f32], n, k) -> Vec<f32>)`
   → `ModelWeights` whose 2D projections are `Projection::Dense(DenseLinear::new(f(...), n, k))`
   (**A8** path = deployed int8-activation semantics), norms/embed copied. Mirrors
   `salt_accuracy.rs::build_weights` but general-arch + a per-weight transform.
   - **PTQ model:** `f = |_, w, n, k| dequant(salt_quantize_forward(w, n, k, T=2))`. Actually
     `salt_quantize_forward` already returns the dense reconstruction → `f = salt_quantize`.
     `ppl_ptq = perplexity(ptq, eval_ids)`.
4. **Heal q/k/v:** for each layer `li`, each of q/k/v: `fp_W` = the fp weight; `target =
   matmul(attn_norm_out_li, fp_W)` (the fp projection output); distill a latent (init `fp_W`) via
   the salt_ste loop (AdamW ~80 steps, lr 3e-3) to match `target` given input `attn_norm_out_li`;
   `healed_W = salt_quantize_forward(healed_latent, T=2)`.
5. **Healed model:** q/k/v = `healed_W`; o/gate/up/down/embed/lm_head = PTQ (`salt_quantize`).
   `ppl_healed = perplexity(healed, eval_ids)`.
6. **Gate:** `ppl_ptq > ppl_fp` (quant degrades), `ppl_healed < ppl_ptq` (q/k/v heal recovers a
   real fraction: assert `ppl_healed < ppl_ptq - k·(ppl_ptq - ppl_fp)` for a modest `k`, e.g. 0.1),
   `ppl_healed ≥ ppl_fp`. Print all three.

**Follow-on (0038b):** extend `forward_dump`/`BlockDump` to capture all layers' `attn_out` +
`ffn_norm_out` → heal o/gate/up/down too; then end-to-end logit-KL fine-tune.

> **NOTE (2026-07-10):** design finalized read-only during a build/test-classifier outage;
> implement + validate (compile, clippy -D, run the ignored gate on SmolLM2) the moment testing
> is back. Do NOT commit before it compiles + the gate passes.

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
