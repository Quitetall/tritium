# Tritium ROADMAP — strategic milestones + tactical plan index

This is the **living index** that maps the permanent strategy to the executable work in flight.
It exists because Tritium is planned by a strong model and **executed by a weaker model**: the
plans must be detailed and verification-gated enough that a cheap executor can run them faithfully
and a reviewer can course-correct from the outputs alone.

## Three layers

| Layer | Where | What | Lifetime |
|-------|-------|------|----------|
| **Strategic** | `docs/adr/` | ADRs — the 0.1.0→1.0.0 arc, each milestone's scope + exit gates | permanent (append/amend) |
| **Index** | `docs/ROADMAP.md` (this file) | the ordered set of tactical plans from *now* → done, with status | living |
| **Tactical** | `docs/plans/NNNN-*.md` | one detailed, verbatim, gated executable plan per point-release / coherent feature (1–few commits) | per chunk; kept for the audit trail |

Everything the executor needs is under **`docs/`** — point the model there. **`docs/EXECUTOR.md`** is the
single entry point (protocol + which plans to run next + how to report back). `~/.claude/plans/` is
ephemeral session scratch; the durable plans live in `docs/plans/` (version-controlled, diffable).

## The per-turn loop

1. **Plan** — the strong model writes the next `docs/plans/NNNN-*.md` (see template below) and flips its
   row here to `in-progress`.
2. **Execute** — the weaker model runs the plan step-by-step, pasting the full output of each
   command and the review verdict.
3. **Review** — the strong model reads the diff + pasted outputs against each step's *expected
   output* block, then either:
   - **course-corrects** (amends the plan / writes a fix-up micro-plan), or
   - **accepts** (flips the row to `done`, writes the next plan).

The expected-output blocks (PASS signature + failure branches) are the spine: they let review
proceed from outputs alone, "no matter what happens."

---

## Strategic milestones (ADR 0002)

| Milestone | ADR | Status |
|-----------|-----|--------|
| v0.10 Foundation | 0003 | **Done** (tagged) |
| v0.20 Inference Spine | 0004 | **Done** (tagged) |
| v0.30 Performance (+ v0.3.1 device-resident, 0013) | 0005 / 0013 | **Done** |
| v0.40 SALT Quantization | 0006 | **Done** (tagged `v0.4.0`, pushed) |
| v0.4.1 perf/correctness point-release (split-KV + IMMA fix) | 0013 / — | **Done** (tagged `v0.4.1` @ `202bec1`, pushed; CUDA decode + IMMA gates re-verified on GPU) |
| v0.50 Training Core | 0007 | **Done** (tagged `v0.5.0`) — STE autograd + Gate C green CPU+CUDA (0005–0007), AdamW + bit-exact checkpoint (0008), LoRA on frozen base (0009), QAT heal bridge + distillation-convergence capstone on the real model (0010). Full-model ≥90% PPL-recovery deferred to v0.60 (needs full-model backprop + a non-QAT-latent checkpoint — see 0010). |
| v0.60 Pretraining + Distributed | 0008 | **In-progress** — full-model CPU backward (0011, `v0.5.1`) + data pipeline (0012, `v0.5.2`) + GPU pretrain smoke (0013, `v0.5.3`) + ProcessGroup trait & simulated collective backend (0014, `v0.5.4`) + ZeRO-3/FSDP over the sim PG with gradient + loss parity (0015, `v0.5.5`) **done**; shipping the single-GPU-reachable foundation as the `0.5.x` line (next: distributed checkpoint + resharding + fault-inject, 0016), with the real ≥2-GPU wall (NCCL + ≥80% scaling) deferred to a rented session → `v0.60.0` |
| v0.70 Backend Breadth | 0009 | Planned |
| v0.80 Interop (`tritium-serve`) | 0010 | Planned — *OpenAI-HTTP server doubles as the LAMU `backend_kind` interface* |
| v0.90 Hardening | 0011 | Planned |
| v1.0 Release | 0012 | Planned |
| **Spec-decode (BASTION-style tree verify)** | **0014** | **Proposed (ADR written)** — *post-v0.4.1 point-release; Tritium = verifier, LAMU orchestrates, drafter external* |

## Tactical plan index (now → done)

Ordered. `todo` = not started, `in-progress` = executor running it, `done` = accepted + committed.

| # | Plan | Scope | Serves | Status |
|---|------|-------|--------|--------|
| # | Plan | Scope | Serves | Status | Parallel? |
|---|------|-------|--------|--------|-----------|
| 0001 | `docs/plans/0001-v0.4.1-split-kv-wiring.md` | Split-KV attention into the resident decode | v0.4.1 / ADR 0013 | **done** (`79f4939`+`899b162`) | — |
| 0002 | `docs/plans/0002-v0.4.0-doctests-example.md` | Runnable SALT example (U9) | v0.4.0 / U9 | **done** (`a71c48e`) | — |
| 0003 | `docs/plans/0003-v0.4.1-imma-oob-fix.md` | Fix the JIT IMMA tail-shape OOB read (U7); compute-sanitizer clean | v0.4.1 / U7 | **done** (`13438a4`; memcheck 0 errors) | — |
| 0004 | `docs/plans/0004-v0.4.1-release.md` | v0.4.1 CHANGELOG + version bump | v0.4.1 | **done** (`202bec1`, tagged `v0.4.1` + pushed) | sequential |
| ADR 0014 | `docs/adr/0014-spec-decode-bastion.md` | BASTION spec-decode design (Tritium = verifier) | new capability | **done** (proposed) | — |
| 0005 | `docs/plans/0005-v0.50-train-skeleton-ste.md` | tritium-train skeleton: STE + ternary-matmul backward, gradient-checked | v0.50 / ADR 0007 Gate C | **done** (`5030fa3`; CPU) | — |
| 0006 | `docs/plans/0006-v0.50-cpu-ops-tape.md` | CPU op set (bias/relu²/mse/xent/elementwise) + reverse-mode tape + composed QAT gradient | v0.50 / ADR 0007 Gate C | **done** (`2be9332`; Gate C green on CPU) | — |
| 0007 | `docs/plans/0007-v0.50-cuda-backward.md` | CUDA backward kernels (gA/gW/gs) gradient-checked vs CPU vjp + compute-sanitizer | v0.50 / ADR 0007 | **done** (single-GPU; parity 1e-4 + memcheck 0 errors) | — |
| 0008 | `docs/plans/0008-v0.50-optimizer-checkpoint.md` | AdamW + minimal Optimizer trait + bit-exact TOPT checkpoint (resume==uninterrupted, no-NaN ≥1k) | v0.50 / ADR 0007 | **done** (`05e45d2`; CPU) | — |
| 0009 | `docs/plans/0009-v0.50-lora-frozen-base.md` | LoRA on a frozen ternary base (dense/detach/scale_const primitives); frozen-base zero-grad + merge + rank edges proptested | v0.50 / ADR 0007 | **done** (CPU) | — |
| 0010 | `docs/plans/0010-v0.50-qat-heal-gate.md` | QAT heal bridge (replace_weights/invalidate_resident) + distillation-convergence capstone on the real BitNet-2b4t model | v0.50 / ADR 0007 | **done** (GPU; ~94.6% layerwise convergence; full-model PPL-recovery deferred to v0.60) | — |
| 0011 | `docs/plans/0011-v0.60-full-model-backward.md` | Full-model CPU backward: rmsnorm/softmax/rope/transpose tape vjps + tiny-transformer end-to-end gradient gate | v0.60 / ADR 0008 | **done** (`b6620cf`, tagged `v0.5.1`; CPU) | — |
| 0012 | `docs/plans/0012-v0.60-data-pipeline.md` | Data pipeline: deterministic resumable dup/loss-free sharded `DataSampler` + the `.tqbin`/`.tqidx` corpus formats (total never-panic parsers) | v0.60 / ADR 0008 | **done** (tagged `v0.5.2`; CPU) | — |
| 0013 | `docs/plans/0013-v0.60-gpu-pretrain-smoke.md` | GPU QAT training step (wires the `train_grad` forward+grad kernels) + LR schedule + from-scratch tiny-model pretrain smoke; device==CPU + finite-difference composition gradcheck | v0.60 / ADR 0008 | **done** (tagged `v0.5.3`; 1 GPU) — full-2B resident training engine deferred | — |
| 0014 | `docs/plans/0014-v0.60-process-group.md` | `ProcessGroup` trait + deterministic thread-simulated collective backend (all_reduce / reduce_scatter / all_gather / broadcast); all-reduced grads == single-process summed reference | v0.60 / ADR 0008 | **done** (tagged `v0.5.4`; CPU sim) — uniform publish-first/2-barrier protocol + op-tag desync guard, adversarial-review-hardened | — |
| 0015 | `docs/plans/0015-v0.60-fsdp-zero3.md` | ZeRO-3/FSDP over the sim PG: `FlatShardPlan` + all_gather/reduce_scatter sharded training; reduced-gradient == full-batch gradient (teeth) + replicated bit-exact (world∈{2,4}) + partition loss-curve tracking | v0.60 / ADR 0008 | **done** (tagged `v0.5.5`; CPU sim) — review-hardened: added gradient-level teeth after the loss-only gate was found blind to a wrong reduce op (AdamW scale-invariance) | — |
| 0016+ | (planner writes just-in-time) | distributed checkpoint + resharding + fault-inject (0016), then the ≥2-GPU wall (0017–0018) | v0.60 / ADR 0008 | todo | per chunk |

> **0002 (A) and 0003 (B) are independent** — disjoint files (format/quantize+examples vs the CUDA
> JIT codegen) → safe to run concurrently. For true parallelism use a **git worktree per plan**
> (`git worktree add ../tritium-0003 main`) so two executors don't fight one index; otherwise run
> them sequentially (either order). 0004 runs only after both land + are green.

> **0001 outcome (planner review):** split-KV is correct (graph==eager bit-exact; greedy/parity
> gates hold) and a real kernel win — nsys shows attention **57.6% → 26.6%** of N=1 GPU time
> (~2.2× faster), neutral-to-better throughput across N (no high-N regression on clean re-bench).
> But end-to-end **N=1 throughput stayed ~flat (108→111)** because N=1 is occupancy-bound across
> the *whole* pipeline (GEMM 35% / rmsnorm 21% / eager lm_head 8% now dominate). The plan's
> `>120 N=1` criterion was an over-optimistic Amdahl estimate — **corrected**: the success metric
> for an attention fix is the *attention* share, not N=1 wall-clock. Further N=1 gains are
> diminishing-returns (whole-pipeline occupancy); the regimes that matter (N≥2, long-ctx, the
> argmax path) already perform well. **Lesson for future plans:** gate perf work on the *profiled
> bottleneck's* metric, not a derived end-to-end target.

> Already shipped on `main` above `v0.4.0` (this session, pre-system): rustfmt-1.9.0 reformat
> (CI fmt gate), U5 fuzz targets + `fuzz/target` untrack, split-KV attention **kernels +
> equivalence gate** (`b5173e9`) + head_dim guard (`9e70354`). Plan 0001 is the *wiring* of those
> kernels into production.

---

## Forward decomposition (v0.50 → v1.0)

The strategic ADRs **0007–0012** already define each milestone's exit gates + Definition of Done.
Tactical plans are written **just-in-time** (verbatim against the code as it exists when the
milestone starts) — writing them all now would be fiction against crates that don't exist yet. So
this section is the **map**: per milestone, the ordered *entry* tactical plans an executor should
write/run first, and the hard blocker that gates the milestone. A planner turns each `→` arrow into
one `docs/plans/NNNN-*.md` when its turn comes.

- **v0.50 Training Core — ADR 0007** *(blocker: GPU + a real fp16 source model + a known-recoverable
  fine-tune task)*
  `tritium-train` skeleton + STE autograd for the ternary matmul (finite-difference gradient-check,
  TDD) → backward kernels for the remaining ternary ops (CPU first, then CUDA; each gradient-checked)
  → optimizer + bit-exact save/restore (resume==uninterrupted golden) → LoRA on a frozen ternary
  base (zero-grad-on-base proptest; `r=1` and `r=full` edges) → QAT heal loop + the GPU
  ≥90%-gap-recovery convergence gate. *(Optional external: a BLUT training cookbook over
  `tritium-train` — AGPL boundary, cookbook→Tritium only.)*
- **v0.60 Pretraining + Distributed — ADR 0008** *(blocker: ≥2-GPU cluster; multi-node interconnect
  for the multi-node path)*
  Data pipeline (deterministic sharded shuffle, resumable mid-epoch — CPU-testable) → FSDP/DDP
  gradient/param sharding (N-GPU loss-parity vs 1-GPU) → distributed checkpoint + resharding J≠K →
  multi-node orchestration + rank-kill fault injection → scaling bench (≥80% efficiency) +
  from-scratch pretrain smoke.
- **v0.70 Backend Breadth — ADR 0009** *(blocker: per-platform hardware — Metal box, ROCm GPU,
  WebGPU/WASM target)*
  **Freeze + version the conformance vector set first** (the one CPU/CUDA pass) → then one backend
  crate per plan: `tritium-metal` → `tritium-rocm` → `tritium-wgpu`/`tritium-wasm`, each passing the
  full conformance set + cross-backend parity + acceptance-model greedy match.
- **v0.80 Interop — ADR 0010** *(blocker: acceptance model for native-parity gates; ONNX Runtime in
  CI)*
  `tritium-serve` (OpenAI-compatible HTTP — **this is also the LAMU `backend_kind` interface**;
  contract + streaming + concurrency gates) → `tritium-ffi` (C ABI + generated header; C/C++ compile
  + round-trip + null-arg fuzz + sanitizer) → `tritium-candle` / `tritium-burn` ops → `tritium-onnx`
  custom op then ORT EP.
- **v0.90 Hardening — ADR 0011** *(blocker: full per-platform CI matrix + GPU/multi-GPU lanes +
  model download)*
  **Full doctest sweep** (the ~27 v0.4.0 public items still lacking `/// # Examples`, the rest of U9
  — *startable now, good standalone agent task*) → 24h-cumulative fuzz breadth across every parser →
  full CI build/test matrix → packaging (wheels for manylinux/macOS/Windows + `cargo publish
  --dry-run`) → mdbook + dead-link check → perf-regression gate enforced on `main` → security review
  + threat model + `cargo-deny`/SBOM.
- **v1.0 Release — ADR 0012** *(blocker: real model end-to-end in a fresh env; GPU)*
  `cargo-semver-checks` baseline + public API/C-ABI freeze → re-run every prior gate on the release
  commit → third-party-reproducible quickstart + model zoo + benchmark report → capstone fresh-env
  e2e (install → infer → SALT-quantize → fine-tune).
- **Spec-decode (BASTION) — ADR 0014** *(post-v0.4.1 point-release; blocker for the speedup gate: an
  external block-diffusion drafter; correctness gates need only a mock drafter)*
  Tree-masked verify attention (sibling of split-KV) → shared-prefix KV + provisional
  commit/rollback → accept logic + losslessness gates (greedy + sampling, mock drafter) → memcheck →
  *(with a real external drafter)* end-to-end speedup bench at the roofline knee.

**Convention reminder:** milestone work is gate-blocked, not date-blocked — no work on milestone N+1
merges until N's gate is green and tagged (ADR 0002). A milestone whose hard blocker (GPU count,
per-platform HW, model download) is unavailable runs its load-bearing gate as a **documented manual
gate** on borrowed hardware before tagging.

---

## Tactical plan template (`plans/NNNN-<slug>.md`)

Every tactical plan MUST follow this shape so the executor + reviewer have one contract:

```markdown
# NNNN — <title>  (serves: <ADR / milestone>)

## Goal
One paragraph: what this plan delivers + the one-line success criterion.

## Preconditions
- Branch `main` at commit <hash> (`git log --oneline -1` must show it).
- What is already done (so the executor doesn't redo it).
- `git status` must be clean before starting.

## Steps
Each step is atomic and ends with an expected-output block.

### Step N — <what>
- **Files:** exact paths.
- **Edit:** exact `old_string` → `new_string` (verbatim for CUDA/unsafe/bit-exact/parser code),
  or the full code block to add. Routine glue may be precise prose + signatures.
- **Command:** the exact shell command to run.
- **Expected output (PASS):** the exact line(s) / numeric range to see.
- **If you see instead … :** 2–3 likely divergences, each with a diagnosis + the corrective action.
- **Paste:** "paste the full output of `<command>`" (so review sees ground truth).

## Gate
The test(s)/bench that must pass + expected numbers (with tolerance). Exact command + expected output.

## Commit
Verbatim commit message (the executor commits exactly this; ends with the Co-Authored-By line).

## Review
"Review the commit with the `feature-dev:code-reviewer` subagent (project policy). Paste the
verdict + every finding verbatim." (The strong model triages findings next turn.)

## Done criterion
The exact state that means this plan is complete (tests green, bench number, clean tree).
```

### Rules for the executor (weaker model)
- **Transcribe, don't derive.** Apply edits exactly as written. Do not "improve", rename, or
  refactor beyond the plan.
- **Never skip an expected-output check.** If output diverges and no failure branch matches, **stop
  and paste the output** — do not guess a fix.
- **One commit per plan** unless the plan says otherwise; commit message verbatim.
- **Always run the review step** and paste the verdict.
