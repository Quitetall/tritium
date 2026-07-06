# ADR 0014 — Speculative decoding: BASTION-style tree verify (Tritium = verifier)

- **Status:** Partially implemented (2026-07-06) — the **greedy tree-verify
  primitive** is live: `CudaDecodeModel::tree_verify_greedy` (batched M=N tree
  forward via `gqa_attention_tree_f32` ancestor-masked attention, provisional
  K/V rows past the watermark, accepted-path promote + O(1) rollback), gated by
  `cuda_tree_verify_greedy_lossless` (chains, branches, partial + full rejects
  all commit the exact plain-greedy stream). Still pending: the
  speculative-SAMPLING accept rule, and the end-to-end speedup gate (needs the
  external block-diffusion drafter).
- **Date:** 2026-06-20
- **Deciders:** Brian Lam
- **Relates:** builds on the M=N batched decode (v0.3.7) + split-KV flash-decoding attention
  ([ADR 0013](./0013-v031-device-resident-forward.md), v0.4.1); a **new capability** sequenced
  **after v0.4.1**, orthogonal to the v0.50→v1.0 milestone staircase ([ADR 0002](./0002-release-roadmap.md));
  the natural orchestration host is LAMU (external, MIT).

## Status

Greedy verifier landed (see Status above); the remainder of this section
describes the original full scope. Unblocked only once v0.4.1 is tagged: the tree-verify
attention is a direct sibling of the split-KV attention (it reuses the partial+combine structure and
the M=N KV layout), so the split-KV kernels must be in production and gated first. This ADR fixes the
**scope and the validation regime**; it does not schedule implementation against the capability
staircase (v0.50 training core remains the next *milestone*). Spec-decode is a **performance
capability** that can be slotted as a point-release (e.g. v0.4.2 / v0.5.x) once a drafter is wired —
it adds no new model surface, only a faster path to the *same* tokens.

**Hard blockers:**
- A **block-diffusion drafter** is required and is **out of Tritium's scope** (see Decision). The
  end-to-end speedup gate cannot be validated without one. The acceptance/losslessness gates
  (below) *can* be validated with a trivial/mock drafter, so correctness is testable before a real
  drafter exists.
- The speedup gate needs a **GPU** and the **acceptance model** (BitNet b1.58 2B4T) — model-download
  + GPU CI lanes (same as the decode perf gates).

## Context

BASTION (arXiv 2605.29727, NeurIPS 2026) is **lossless** speculative decoding that beats both
autoregressive (AR) decode and single-path block-diffusion:

1. A **block-diffusion drafter** produces, in *one* forward pass, position-wise marginals
   `q_k(·)` for a block of `B` future positions (not a single chain — a per-position distribution).
2. A query-dependent **prefix tree** is built **best-first** over those marginals: each candidate
   path `i` has score `ρ(i) = Π_k q_k(token_k)`; the highest-probability prefixes are expanded first
   under a **roofline budget controller** that caps the tree size `N` at the point where one target
   forward over `N` nodes still costs ~one AR step (verification is "free" up to the roofline knee).
3. The **target** (Tritium) verifies the **whole tree in one forward** — N nodes sharing a common
   prefix KV — and commits the **longest accepted path** under the standard speculative-sampling
   accept rule, which **preserves the target's distribution exactly** (lossless).

Reported: up to **6.61× over AR**, **+39%** over single-path block-diffusion.

**Why Tritium fits.** Tritium already has the expensive half: a batched **M=N decode** where N rows
advance concurrently, and (from v0.4.1) **split-KV attention** that parallelizes one decode row's
attention over KV-key chunks with an online-softmax combine. Tree-verify is M=N decode where the N
rows are *tree nodes sharing a prefix* and each node attends a **masked** subset of KV (its
ancestors), instead of each row attending its own full range. That is one new attention variant plus
a KV-layout change — not a new engine.

**Division of labor.** The drafter is a *different model* (block-diffusion, e.g. DFlash-style) and
the orchestration (drafter forward → tree build → budget control → target verify → commit/rollback →
loop) is a control loop, not a kernel. Tritium stays the **self-contained ternary engine + target
verifier**; the drafter and orchestration live **outside** Tritium. LAMU (MIT) already runs DFlash
block-diffusion spec-decode and manages multi-model placement, so it is the natural host — but the
boundary is what matters, not the specific host: Tritium exposes a **tree-verify primitive**, anyone
can drive it.

## Decision

Adopt BASTION-style tree verification with Tritium as the **target verifier only**. Tritium delivers
three primitives; the drafter and the orchestration/budget loop are external.

### Tritium-side scope (this ADR)

1. **Tree-masked verify attention** — a sibling of split-KV. N tree nodes share a prefix KV; each
   node attends `{shared prefix} ∪ {its tree-ancestors}` via a **per-node ancestor mask** (a compact
   `[N]`-indexed parent table + depth, expanded to the attended key set), vs today's M=N where each
   row attends its own contiguous KV range. Reuses the split-KV partial+combine online-softmax; the
   only new ingredient is the mask applied during the partial pass.
2. **Shared-prefix KV + provisional commit/rollback** — today each batch row owns a dedicated
   `[max_ctx · kv_width]` arena (`new_batch`, `kv_append_mdecode_f32`). Tree-verify wants **one
   shared prefix** KV + **N provisional** candidate K/V slots written during the tree forward, of
   which only the **accepted path** is promoted into the prefix (the rest discarded). Needs: a
   shared-prefix KV layout, a provisional-write region, and an O(accepted-length) **promote** +
   O(1) **rollback**.
3. **Accept logic** — exact longest-accepted-path selection using the deterministic
   `tritium_nn::sample_greedy` for the greedy case and the standard speculative-sampling accept rule
   for the sampling case (so the committed sequence is **distribution-identical** to plain target
   decode). The roofline ceiling (`benches` `bitnet_2b4t_decode_ceiling`) supplies the budget
   controller's `T_verify(N)` curve so the *external* controller can pick `N` at the knee.

### Out of scope (external, by design)

- The **block-diffusion drafter** (a separate model + its forward).
- The **tree builder + budget controller** (best-first expansion over drafter marginals, roofline
  `N` selection) — a control loop that *calls* Tritium's tree-verify primitive. Lives in the
  orchestrator (LAMU or any driver).
- Any change to the model surface — spec-decode is a faster path to the **same** tokens; no new
  public model capability, no SALT/training interaction.

## Testability (exit gates)

The losslessness + correctness gates are validatable with a **mock drafter** (deterministic
hand-built trees), so they do not block on an external drafter. Only the speedup gate needs a real
drafter + GPU + model.

| Gate | Tag | How tested | CI lane |
|------|-----|-----------|---------|
| **Losslessness (greedy):** tree-verify commit == plain greedy decode, token-for-token, on a fixed prompt set | C | vs-reference (plain `decode_batch` greedy) | model-download / GPU |
| **Losslessness (sampling):** committed-token distribution == target distribution within statistical tolerance over K seeds (speculative-sampling accept rule) | C/P | distribution test vs plain sampled decode | GPU |
| **Tree-masked attention parity:** a single-path tree (chain) reproduces the existing M=N decode attention bit-/tolerance-exact; a branching tree's per-node output == that node decoded standalone with its ancestor KV | C/P | vs-reference (split-KV attention) | GPU |
| **KV promote/rollback:** after accept of length `L`, the prefix KV equals plain decode's KV after `L` appends; after a full reject, prefix KV is byte-identical to pre-verify (no corruption) | C/E | golden + contract test | GPU |
| **Edge cases:** N=1 tree (degenerates to AR), full-accept, full-reject, max-depth tree, empty drafter block | E | conformance harness (boundary suite) | GPU |
| **memcheck clean** on the tree-verify attention + provisional-KV paths (the new unsafe/index surface) | M | `compute-sanitizer` | GPU |
| **Speedup (needs real drafter):** end-to-end tok/s with drafter+tree-verify ≥ AR baseline by a target factor on the acceptance model; verification stays at/under the roofline knee | Pe | bench (drafter + target) | model-download / GPU |

## Definition of done — (point-release tag, e.g. `v0.4.2` / `v0.5.x`)

- [ ] **C** Losslessness (greedy): tree-verify commits the identical token stream as plain greedy decode on the prompt set.
- [ ] **C/P** Losslessness (sampling): committed distribution matches the target within tolerance over K seeds.
- [ ] **C/P** Tree-masked attention parity: chain-tree == M=N decode; branching-node output == standalone-with-ancestors.
- [ ] **C/E** KV promote/rollback correct: accepted-length prefix == plain-decode KV; full-reject leaves prefix byte-identical.
- [ ] **E** Edge cases pass (N=1, full-accept, full-reject, max-depth, empty block).
- [ ] **M** `compute-sanitizer` clean on the new attention + provisional-KV paths.
- [ ] **Pe** (with a real external drafter) end-to-end speedup ≥ target factor on the acceptance model at/under the roofline knee.
- [ ] U1–U9 green; the drafter + orchestration boundary documented (external host, e.g. LAMU). Tag the point-release.

## Consequences

- **Positive:** large decode speedup (≤6.6× reported) with **zero quality loss**, reusing the M=N +
  split-KV substrate; Tritium stays self-contained (verifier only) while integrating cleanly with an
  external block-diffusion drafter / orchestrator.
- **Negative / risk:** the tree-masked attention + provisional-KV layout is the most intricate
  kernel/memory work since split-KV; the **speedup** gate depends on an external drafter Tritium
  doesn't own (correctness is independently testable with a mock, so this risk is isolated to the
  perf claim). The budget controller's roofline knee is hardware-specific (autotuned per device).
- **Sequencing:** after v0.4.1 (needs split-KV in production). Does **not** displace the v0.50
  training-core milestone on the capability staircase; it is an opt-in performance capability that
  can ship as a point-release whenever a drafter is available to drive it.
