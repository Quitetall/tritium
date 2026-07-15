# ADR 0026 — SOTA campaign: close every measured gap, make the claim reproducible

Status: **PROPOSED** (2026-07-11)

- **Deciders:** Brian Lam
- **Relates:** consumes the measured head-to-head in
  `docs/research-ternary-sota-mid2026.md` §7; sequences existing proposals
  ([ADR 0021](./0021-drafter-architecture.md) drafter,
  [ADR 0024](./0024-structured-24-ternary.md) 2:4 ternary,
  [ADR 0023](./0023-relaxed-reduction-tier.md) relaxed-reduction tier) and
  the plan staircase (0037 Qwen arch → 0042 paper) into one campaign with
  a single, falsifiable exit condition. Respects
  [ADR 0019](./0019-persistent-megakernel-decode.md)'s deferral (megakernel
  premise measured weak — none of the tracks below revisit it).

## Context: what "SOTA" means here, and what the measurements say

The 2026-07-11 same-box head-to-head (RTX 4090, `research-ternary-sota-mid2026.md` §7)
establishes where Tritium actually stands:

| regime | Tritium | best alternative (same box) | position |
|---|---|---|---|
| ternary decode, CUDA | **302 tok/s** @2.4B (~517 GiB/s weight stream) | llama.cpp TQ2_0-CUDA: 24 tok/s @4B; bitnet.cpp CPU: 23 tok/s | **leads 13×** |
| decode bandwidth-efficiency | ~517 GiB/s | llama.cpp Q4_K_M (non-ternary): ~508 GiB/s | **parity with mainstream flagship** |
| ternary prefill, CUDA | 1,068 tok/s pp512 | llama.cpp Q4_K_M (non-ternary): 12,281 | **11× behind mainstream** |
| spec-decode | 1.19× (lookup drafter) | JetFlow 9.64× / BASTION 6.61× (published, H100/A100) | mid-pack |
| quality-per-byte | SALT pipeline done, **unevaluated** | PT²-LLM (ICLR 2026, verified) | **unmeasured** |

Being SOTA in ternary decode on consumer CUDA is *already true* — but it is
true by default (nobody else runs there) and unpublished. "Undoubtedly SOTA"
requires three harder things:

1. **No regime asterisks.** The prefill deficit lets any reviewer say "slower
   than llama.cpp where it's compute-bound." That asterisk must go.
2. **A quality claim, not just a speed claim.** Speed on a ternary model
   nobody wants to run is not SOTA. SALT must be measured against the
   verified PTQ-ternary baseline (PT²-LLM) and the shipped model to beat
   (Ternary Bonsai — whose vendor numbers were refuted, so an independent
   eval is itself a contribution).
3. **Reproducibility as a feature.** A SOTA claim that lives in a private
   optimization log is a blog post. The claim must ship as a harness anyone
   can re-run — same discipline as bitnet.cpp's published `llama-bench`
   lines, plus quality numbers vendors don't publish.

The competitive window is real but closing: vLLM's Q2 2026 roadmap adds
W{1-8} kernel coverage ("humming-kernel"), and QVAC (Vulkan/Metal ternary +
LoRA) is the nearest engine competitor. Every quarter the decode lead stands
unpublished, it is a quarter someone else can claim first.

## Decision (proposed)

Run four tracks, each with measurable exit gates, ordered so that each
track's artifact feeds the next. The campaign's single exit condition:

> **On one consumer GPU (RTX 4090), for ternary models, Tritium holds the
> best measured number in every regime it competes in — decode, prefill,
> spec-decode effective throughput, and quality-per-byte — and the whole
> claim reproduces from one public command.**

### Track P — Prefill: kill the 11× asterisk (IMMA mpGEMM path)

The prefill gap is not a ternary limitation — it is the absence of a
tensor-core batched-GEMM path. llama.cpp's 12K tok/s rides MMQ (int8 tensor
cores); Tritium's prefill runs the dp4a decode GEMMs at M>1.

1. **IMMA prefill GEMMs.** `mma.m16n8k32.s32.s8.s8.s32` over the existing
   I2sInt8 interleave (`tritium-format::i2s_int8`, IMMA_K/IMMA_N already
   defined): ternary weights unpack to i8 sign values once at load (VRAM cost
   at 2B: ~2.4 GB i8 shadow — acceptable; make it opt-in per-model-size), act
   quant already exists (`rmsnorm_quant_f32`). Accumulate i32, dequant in
   epilogue. This is the ADR 0024 substrate *without* waiting for 2:4: dense
   IMMA first, `mma.sp` upgrade when a 2:4-trained checkpoint exists.
2. **cp.async + ldmatrix staging** for the weight tiles (the known idiom;
   no novelty here — this is deliberately boring engineering).
3. **Scope honesty:** decode (M=1) does not change — dp4a stays. This track
   touches prefill and the M=N batched-decode GEMMs only, behind the same
   twin-kernel contract (ADR 0022): the existing f32-tiled path remains the
   parity reference; IMMA output gated at tolerance (i32 accumulation is
   exact for the value range — the gate should be *bit*-exact vs an integer
   reference, tolerance only at the f32 epilogue).

**Gates:**
- `Pe` pp512 ≥ **6,000 tok/s** on BitNet 2B4T (≥5.6× today; ~half of
  llama.cpp's Q4_K_M number on a model 1.75× smaller — bandwidth-fair parity).
  Stretch: 10,000.
- `C` IMMA prefill logits == integer-reference bit-exact pre-epilogue;
  greedy-token parity with the f32 path on the acceptance prompt set.
- `M` compute-sanitizer clean on the new path.

### Track D — Drafter: execute ADR 0021, own the spec-decode number

The verifier is done (greedy + sampling accept, tree endpoints, batch
coexistence). The 2–3× sits in the drafter. ADR 0021 already decided the
architecture (~100–200M ternary AR model, LLaMA-3 tokenizer, same engine,
BLUT-trained). This track adds only sequencing and the missing verify-cost
work:

1. **Fixed-m verify CUDA graph** (the identified eager-launch-storm fix:
   ~420 launches, 8.5 ms → target ≤4.5 ms). Engine-side, no model needed —
   do it first.
2. **BLUT drafter training run** per the ADR 0021 recipe (distill from 2B4T,
   acceptance-oriented objective — Draft-OPD-style on-policy data as
   stretch).
3. **Ternary-drafter novelty:** no published system runs a *ternary* draft
   model (survey-verified gap). The drafter fits in ~40 MB and decodes at
   small-model latency; publishing τ (accepted length) and end-to-end
   speedup for ternary-draft→ternary-target is a first — this is the
   spec-decode leg of the paper.

**Gates:**
- `Pe` verify step ≤ **4.5 ms** under the graph (measured today: 8.5).
- `Pe` end-to-end greedy ≥ **2.0×** over plain decode on the natural-text
  eval set (τ ≥ 4 tok/verify) → ≥600 effective tok/s. Stretch: 2.5×.
- `C` losslessness gates (greedy identity, 200k-MC sampling distribution)
  stay green under the trained drafter — they already run on all lanes.

### Track Q — Quality: the SALT claim, measured against the field

The paper's blocker (plan 0042) and the field's open question are the same
measurement. Three sub-items, strictly ordered:

1. **Bonsai ingestion.** Q2_0-g128 → TQ2_0 crosswalk (group-size 128→256
   re-group + scale re-fit; the community TQ2_0 requants are the fallback
   path and the cross-check). Requires plan 0037 (Qwen arch) — this track
   is its consumer, not its owner. Deliverable: Ternary-Bonsai-1.7B/4B/8B
   run in Tritium with token-parity vs the PrismML llama.cpp fork.
2. **Independent Bonsai eval.** Held-out perplexity + the standard zero-shot
   suite, published with the harness. The vendor's quality table was refuted
   on verification (0-3) — the first independent numbers are cite-able
   regardless of which way they fall.
3. **SALT vs PT²-LLM head-to-head.** Same fp master (Qwen3-family, since
   PT²-LLM validated there), matched bpw budgets, reconstruction MSE +
   held-out ppl + zero-shot. If ITF-style alternating refinement beats
   SALT's per-plane fit, adopt it into SALT (it composes — that's a result,
   not a loss). The BCJR-QAT "proxy gap" warning applies: report end-task
   metrics as primary, MSE as secondary.

**Gates:**
- `C` Bonsai-8B greedy token-parity (≥ 63/64 tokens over the prompt set) vs
  the PrismML fork on identical input.
- `Pe` Bonsai-8B decode in Tritium ≥ **250 tok/s** on the 4090 (it is 3.4×
  the 2B4T weight stream; ~250 is the bandwidth-scaled expectation — beating
  the M4-Pro Metal number 3× is table stakes, but state the honest basis).
- `Q` SALT-vs-PT²-LLM table complete at ≥2 model scales, end-task metrics
  primary. **"No win" is a valid result** (benchmarking discipline) — the
  deliverable is the measurement, and the composition experiment
  (ITF-initialized base + SALT residual planes) if SALT alone loses.

### Track R — Reproducibility: the claim ships as a command

1. **`tritium report compare`** — one subcommand that runs decode + ttft on
   a given GGUF, emits the same JSON schema as today's reports plus
   environment capture (GPU, driver, clocks, contention snapshot), and
   prints the comparison table. Competitor lines stay external (llama-bench
   invocations documented beside it) — we do not wrap competitors' binaries,
   we document their exact repro lines as §7 already does.
2. **`docs/BENCHMARKS.md`** — the public face: the §7 table, the exact
   commands, the honest caveats (contention spread, first-run compile), and
   a dated results ledger. Updated only by re-running the harness.
3. **Paper artifact** (plan 0042 intersection): the benchmark ledger and the
   Track Q tables are the paper's evaluation section, generated not written.

**Gates:**
- `E` `tritium report compare --model <gguf>` reproduces every Tritium
  number in BENCHMARKS.md within run-to-run spread, from a clean checkout.
- `E` BENCHMARKS.md carries zero numbers without a command line beside them.

## What this ADR explicitly does NOT do

- **No megakernel** (ADR 0019 stands: ~3% ceiling, measured).
- **No CPU performance work** — bitnet.cpp/Vec-LUT own CPU; Tritium's CPU
  backend remains the bit-exact reference. The SOTA claim is scoped to
  consumer CUDA and says so.
- **No 2:4 kernel work before a 2:4 checkpoint exists** (ADR 0024's own
  gate). Track P's dense IMMA is designed so `mma.sp` is a drop-in upgrade.
- **No vLLM/serving-ecosystem integration** this campaign — watch
  humming-kernel; integrate after the claim is published, not before.
- **No new quantization research** beyond the Track Q composition
  experiment; SALT's method is frozen for the measurement.

## Sequencing and dependencies

```
Track D1 (verify graph)         — engine-only, unblocked, ~days      ┐
Track P  (IMMA prefill)         — unblocked, the big kernel item     ├─ parallel
Track Q1 (Bonsai crosswalk)     — blocked on plan 0037 (in flight)   ┘
Track D2 (drafter training)     — BLUT project, external to this repo
Track Q2/Q3 (evals)             — after Q1; Q3 also needs plan 0038 artifacts
Track R  (harness + ledger)     — starts now (schema), finalizes last
```

Decode stays where it is this campaign — 302 tok/s already leads, and the
next decode lever (rmsnorm_fast, ADR 0023 Track E) proceeds independently;
its gains land in the ledger whenever they ship.

## Consequences

- **Positive:** the four measured deficits become four gated tracks; the
  SOTA claim becomes falsifiable and re-runnable instead of narrative; the
  paper's evaluation section is generated by the same harness; the ternary
  spec-decode and independent-Bonsai-eval results are publishable firsts
  regardless of outcome.
- **Negative / risk:** Track P is the largest kernel effort since split-KV
  (mitigated: dense IMMA is a well-trodden idiom; the twin-kernel contract
  bounds regression risk). Track D2 depends on a training run outside this
  repo (mitigated: D1 and the lookup drafter keep the spec path exercised;
  the 2.0× gate is conservative vs published τ for trained drafters).
  Track Q2 may produce numbers below Bonsai's marketing (that is the point —
  the deliverable is the truth). The prefill gate is stated bandwidth-fair
  ("half of Q4_K_M's number on a model 1.75× smaller"), which a hostile
  reader could still call losing; the defense is the effective-bandwidth
  framing already used in §7, stated up front.
- **The claim, precisely:** *fastest ternary inference on consumer CUDA in
  every regime (decode, prefill, spec-decode), with quality-per-byte
  measured against the best PTQ-ternary baseline and the best shipped
  ternary model, reproducible from one command.* Nothing broader; nothing
  vaguer.

## Definition of done

- [ ] `Pe` Track P: pp512 ≥ 6,000 tok/s (2B4T, 4090); `C` integer-reference
      bit-exactness pre-epilogue + greedy parity; `M` sanitizer clean.
- [ ] `Pe` Track D: verify ≤ 4.5 ms; e2e spec ≥ 2.0× greedy (τ ≥ 4);
      `C` losslessness gates green under the trained drafter.
- [ ] `C/Pe` Track Q: Bonsai-8B parity + ≥ 250 tok/s; `Q` independent Bonsai
      eval published; `Q` SALT-vs-PT²-LLM table at ≥2 scales, end-task
      primary.
- [ ] `E` Track R: BENCHMARKS.md ledger, every number re-runnable via
      `tritium report compare` + documented competitor lines.
- [ ] The one-line claim above is true on the ledger's latest dated run, and
      U1–U9 stay green throughout.
