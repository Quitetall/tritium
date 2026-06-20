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
| **Tactical** | `plans/NNNN-*.md` | one detailed, verbatim, gated executable plan per point-release / coherent feature (1–few commits) | per chunk; kept for the audit trail |

`~/.claude/plans/` is the ephemeral session scratch (plan mode); the **durable** plans live in
`plans/` (version-controlled, diffable, reviewable).

## The per-turn loop

1. **Plan** — the strong model writes the next `plans/NNNN-*.md` (see template below) and flips its
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
| v0.40 SALT Quantization | 0006 | **Done** (tagged `v0.4.0`, local; push pending) |
| v0.50 Training Core | 0007 | Planned — *optional BLUT cookbook integration (external, AGPL boundary)* |
| v0.60 Pretraining + Distributed | 0008 | Planned |
| v0.70 Backend Breadth | 0009 | Planned |
| v0.80 Interop (`tritium-serve`) | 0010 | Planned — *OpenAI-HTTP server doubles as the LAMU `backend_kind` interface* |
| v0.90 Hardening | 0011 | Planned |
| v1.0 Release | 0012 | Planned |
| **Spec-decode (BASTION-style tree verify)** | **0014 (to write)** | Planned — *post-v0.4.1; Tritium = verifier, LAMU orchestrates, drafter external* |

## Tactical plan index (now → done)

Ordered. `todo` = not started, `in-progress` = executor running it, `done` = accepted + committed.

| # | Plan | Scope | Serves | Status |
|---|------|-------|--------|--------|
| 0001 | `plans/0001-v0.4.1-split-kv-wiring.md` | Wire the split-KV attention kernels into the resident decode (`md_attn`/`gb_attn`, eager+graph) + re-bench N=1 | v0.4.1 / ADR 0013 | **in-progress** |
| 0002 | `plans/0002-v0.4.1-imma-oob-fix.md` (to write) | Fix the IMMA tail-shape OOB read (U7); compute-sanitizer clean | v0.4.1 / U7 | todo |
| 0003 | `plans/0003-v0.4.1-tag.md` (to write) | v0.4.1 CHANGELOG + version bump + tag; push v0.4.0+v0.4.1 | v0.4.1 | todo |
| 0004 | ADR 0014 — BASTION spec-decode design | spec-decode tree-verify scope (no code) | new milestone | todo |
| … | (mapped as milestones approach) | v0.50→v1.0 tactical breakdown | per ADR | todo |

> Already shipped on `main` above `v0.4.0` (this session, pre-system): rustfmt-1.9.0 reformat
> (CI fmt gate), U5 fuzz targets + `fuzz/target` untrack, split-KV attention **kernels +
> equivalence gate** (`b5173e9`) + head_dim guard (`9e70354`). Plan 0001 is the *wiring* of those
> kernels into production.

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
