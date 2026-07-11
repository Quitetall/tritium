# Ternary drafter — BLUT training handoff (ADR 0021 execution spec)

Status: **Tritium side SHIPPED** (2026-07-10, `--draft-model` two-runner spec
decoding, self-speculation-validated lossless at 11.75 tok/verify). This doc
is the handoff: exactly what BLUT must produce, how it will be validated, and
the named gaps in BLUT today. Training execution is the BLUT project.

## What Tritium consumes

One GGUF file:

- **Arch**: BitNet b1.58 autoregressive transformer, **6–8 layers,
  n_embd 512–768, GQA**, ternary linears + A8 activations, RMSNorm,
  Relu²-MLP (the BitNet family Tritium's runner already executes).
- **Tokenizer**: the TARGET's (LLaMA-3, vocab 128256), f16 tied embeddings.
  `tritium-serve` **rejects a vocab mismatch at startup** — the drafter must
  share the target tokenizer exactly (embedding table ≈130M params dominates
  the student; the ternary body is nearly free).
- **Format**: I2_S or TQ2_0 ternary tensors + f16/f32 norms — i.e. the same
  tensor schema as `microsoft/bitnet-b1.58-2B-4T-gguf` (Tritium's loader is
  config-driven; matching that layout requires zero Tritium changes).

Serving (already live):

```sh
tritium-serve --model bitnet-2b4t.gguf --draft-model student.gguf --backend cuda
```

Falls back to prompt-lookup drafting when `--draft-model` is absent. Accept
telemetry: `/metrics` → `tritium_spec_verifies_total`,
`tritium_spec_committed_total` (tok/verify = ratio); `TRITIUM_SPEC_STATS=1`
prints per-request stats.

## Training recipe (BLUT plan shape)

```text
MaterializeTeacherGenerations   # sample BitNet 2B4T on prose/chat prompts
                                # (sequence-level distillation data: the
                                # drafter must imitate the TARGET's argmax
                                # stream, not ground truth)
  → TernaryDistillTrain         # NEW STAGE (gap 2): BitNet b1.58 student,
                                # STE/QAT ternary linears, CE on teacher
                                # tokens (+ optional logit KL)
  → ConvertGgufTernary          # NEW/EXTENDED (gap 3): emit I2_S or TQ2_0
  → RegisterModel
```

Existing `distill_from_teacher` recipe is the skeleton; its
`MaterializeConversations`/`distill_train`/`convert_gguf`/`register_model`
stage flow matches 1:1.

## Named gaps in BLUT today (verified 2026-07-10)

1. **No BitNet/ternary training path** — `trainer_distill.py` / `trainer.py`
   have no ternary/1.58/QAT support; `distill_train` takes a `student_base`
   HF model string and cannot define a fresh 6–8-layer ternary arch. Needs a
   student-config path + STE ternary linears (reference implementations:
   Tritium's `tritium-train` STE ops, or upstream BitNet recipes).
2. **No ternary quant emission** — `is_supported_quant`
   (`src/spec.rs:315-332`) whitelists K-quants/Q8_0/f16/bf16 only; the
   convert path shells `llama-quantize`, which needs the bitnet.cpp fork for
   I2_S. Either whitelist + route to the fork's quantizer, or convert via
   Tritium (`tritium quantize` emits SALT today; a `repack`-adjacent I2_S
   writer is a small Tritium add if BLUT prefers that route — ask).
3. **Model sizing**: no 100–200M recipe configured. Start at 8L/768; shrink
   toward 6L/512 only if tok/s headroom is insufficient (the drafter runs
   ~10× faster than the target either way; acceptance rate matters more than
   draft speed at these sizes).

## Acceptance gate (from ADR 0021)

- **≥ 6 tok/verify on prose** (self-speculation ceiling measured at 11.75;
  the lookup drafter does 3.65 on repetitive text only). At 6 tok/verify and
  ~14–19 ms/verify, projected decode ≈ **2.5–3×** plain.
- Validation procedure (no Tritium changes needed):
  1. `tritium-serve --model target.gguf --draft-model student.gguf`
  2. Run `tools/openai_compat_check.py` — output must be text-sane.
  3. Greedy determinism: same prompt with and without `--draft-model` must
     produce IDENTICAL output (spec decoding is lossless by construction;
     any diff is a bug, file it against Tritium).
  4. Read tok/verify from `/metrics` on a prose benchmark set; gate ≥ 6.
- VRAM budget: student ≈0.4–0.5 GB + KV beside the 1.2 GB target (fits any
  8 GB card; measured co-residency fine on the 4090).

## Constraints & notes

- `--draft-model` is mutually exclusive with `--batch-slots > 1` (the spec
  loop owns the single-sequence KV; batching-P2 track C4 may lift this).
- Greedy drafting only on the drafter side; the sampled accept rule treats
  drafts as deterministic proposals and stays distribution-lossless — no
  drafter sampling work needed.
- A weak drafter degrades ACCEPTANCE (speed), never correctness — safe to
  iterate on checkpoints against a live server.
