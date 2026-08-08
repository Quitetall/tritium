# RFC 0001 — The relaxed decode numerics tier (`TRITIUM_KERNEL_TIER=fast`, second attempt)

Status: **ACCEPTED** (2026-08-08, Brian) — L3b and the L8 flash-prefill rewrite are unblocked under this contract.
Authorized by [ADR 0036](../adr/0036-decode-endgame.md) § "The numerics RFC";
per that ADR, **accepted or rejected both close the item**, and L3b (online-softmax
fused decode attention) plus the L8 flash-prefill rewrite stay blocked until this
RFC is decided.

## Motivation

The bit-exact sum-order contract ([ADR 0018](../adr/0018-canonical-tree-reduction-order.md))
is the product identity of the f32 single-sequence path and the reason cross-backend
parity is cheap to hold. It is also a hard latency floor under the two remaining
attention costs at decode:

- The last full decode profile (OPTIMIZATION-LOG.md:339, M=1 BitNet-2B4T) puts
  **attention at ~33%** of decode GPU time. Round 26 closed the policy-compatible
  half: L3a coalescing is REFUTED at HEAD (every coalesced variant regresses
  36–592% at kernel level because the bit-exact per-key d-order is what makes
  per-lane streaming efficient), and its verdict names the remainder explicitly:
  "Phase 3 is already coalesced; its cost is the pinned sums = RFC/L3b territory."
- **L3b** — an online-softmax fused decode attention — reorders the softmax
  accumulation and is impossible under the exact contract. Its measured prior is
  T3b (round 25 addendum 2): the shipped 1e-4 split_partial/combine pair wired as
  a tier gave **+13.3% at ctx≈3800** (81.6→92.5 tok/s) but −48.9% at short ctx
  (capture-time fixed n_split) and, being f32-only, could not stack with the
  f16-KV rung (+35% at ctx≈4K) that already owns the long-ctx regime. Deleted
  under the <3% rule. The lever is real; the containment design was wrong.
- **L8** — a flash-prefill rewrite (BENCHMARKS.md:154-156 leftovers) has the same
  blocker: online softmax reorders sums.

This is the second run at a relaxed tier. The first,
[ADR 0023](../adr/0023-relaxed-reduction-tier.md), was rejected by measurement for
its chosen first target (rmsnorm_fast: +1.75% < 3%, deleted) — rmsnorm's cost is
structural, not sum-order. That rejection binds the *target list*, not the tier
concept: ADR 0023's machinery (naming, loud-reject selector, healthz disclosure,
capture-time resolution, the 3% deletion rule) all shipped, worked, and is reused
here verbatim. What this RFC adds beyond ADR 0023 is the lesson of this week's
three precedents:

1. **T3b** (round 25 addendum 2): `cpu_cuda_parity` failed at **2.29e-2 vs the
   2e-3 bar** — the tier's e2e drift went unbounded because the fixed 64-way
   split accumulated 63 identity partials. A tier bar that only checks ppl and
   greedy tokens can hide an order-of-magnitude logit drift.
2. **L2 i8 head** (round 26 addendum 3): plain decode **+7.75%**, ADOPTED — but
   spec τ 3.575→3.427 (−4.1%, deterministic) made spec e2e **−3.1%** despite the
   44.5%-share verifier head running 1.8× faster. Adoption had to be scoped
   plain-only (`TRITIUM_LM_HEAD=i8` for undrafted decode).
3. **L6 f16-KV** (c572f75): plain long-ctx **+43.4%**, spec **−6.7%** on both
   graph and eager routes. Second lever in a week whose plain/spec optima
   diverge.

Tier bars that ignore spec-path interactions produced two scoping surprises this
week. This RFC therefore makes the spec-path measurement mandatory, not optional.

## Proposed contract

### Naming and selection

- `TRITIUM_KERNEL_TIER=exact|fast`, default `exact`; any other value rejected
  loudly at model build (the `TRITIUM_KV` / ADR 0023 pattern, already exercised
  by the T3b wiring).
- Resolved **once per model build**; CUDA graphs bake the picked symbols
  (capture-time constant — the ADR 0023/T3b precedent). No per-request or
  per-token tier switching; a process serves exactly one tier.
- Host-side single dispatch point, the `KvDtype::pick` shape (ADR 0022
  guardrail 3): a `KernelTier` enum in the CUDA backend; selection logic never
  duplicates, only kernel bodies may.
- serve discloses the active tier in `/healthz` (shipped pattern), and **every
  ledger environment line states tier + head dtype + KV rung + spec on/off** —
  the round 26 addendum 3 obligation, promoted to contract.

### Scope: which decode kernels may deviate

Eligible kernel classes (each still needs its own measured adoption, below):

- **Decode attention accumulation** (L3b): the softmax max/sum/weighted-V folds
  in the `gqa_attention_{scores,reduce}` families — online-softmax fusion,
  reordered f32 accumulation, `--fmad=true` permitted inside the fast twin.
- **Prefill attention** (L8 flash rewrite), same arithmetic freedoms.
- Nothing else. GEMMs are out of scope (integer accumulation is order-free —
  ADR 0023); rmsnorm is out of scope (refuted twice: rmsnorm_fast +1.75%, cost
  is structural); the LM head is not a tier member (it is its own sanctioned
  exception and its own format axis, below).

### Error bounds (numeric, per level)

| level | bar | source of the number |
|---|---|---|
| kernel vs exact twin | max rel err ≤ **1e-4** on the kernel's outputs, asserted in the kernel's gate | the shipped split_partial/combine pair's bar (T3b wired "the shipped 1e-4 pair"); the f16 head precedent sits at ~1e-5 rel (decode.cu:3072-3079) |
| e2e logit drift vs host exact oracle | rel err ≤ **2e-3** on final logits over the parity corpus | the `cpu_cuda_parity` bar T3b failed at 2.29e-2 — see the decision below |
| e2e perplexity | ppl(fast)/ppl(exact) ≤ **1.001** on the 5100-position teacher-forced WT-103 harness | the ADR 0036 L2 gate bar, already exercised (i8 head passed at 1.000609) |
| greedy tokens | fast tier == exact tier token-for-token over the **256-token** acceptance horizon; vs transformers, the ≥96-token exact-prefix gate unchanged | ADR 0023's gate (rmsnorm_fast passed 256/256); ADR 0018 re-baseline note |
| spec quality | τ measured; adoption requires τ unchanged within noise **for any spec-default claim** | round 26 addendum 3 (−4.1% τ deterministic sank a +7.75% kernel win) |

**Decision on the 2e-3 bar (the T3b question).** The bar applies to the **tier**,
not just to T3b. `cpu_cuda_parity`'s bit-equality leg cannot apply (the CPU host
has no fast twin, and building one per fast kernel is exactly the cross-backend
sprawl ADR 0018 exists to avoid), but its 2e-3 numeric envelope is the drift
metric every downstream gate implicitly trusts, and T3b's 2.29e-2 was a symptom
of real accumulated error (63 identity partials), not a gate technicality. So:
the fast tier does **not** run the bit-parity leg; it **must** hold rel err
≤ 2e-3 on final logits vs the host exact oracle, asserted by a tier-specific
gate over the same parity corpus. A fast kernel that is 1e-4-clean in isolation
but composes past 2e-3 e2e is rejected — that is precisely the failure T3b
demonstrated.

### The spec-path interaction clause (mandatory)

Every fast-tier kernel adoption measures, separately, before any default claim:

1. **plain decode e2e** (ledger command), short ctx AND long ctx (≥4K) — T3b's
   +13.3%/−48.9% split shows one number lies;
2. **spec decode e2e** (the ADR 0032 harness / spec_kv_bench rig), same ctx
   split;
3. **τ** on the standard drafter harness;
4. **composition with the accepted rungs that own the target regime** — a
   long-ctx claim must be measured ON f16-KV (T3b's f32-only pair claimed a
   regime f16-KV already owned; that must fail review, not measurement).

Adoption verdicts are **per serving mode**: `fast` may become the recommended
setting for plain decode while spec stays `exact` (the L2/L6 outcome shape).
There is no such thing as a mode-blind adoption. Tiers compose with KV rungs,
head dtype, and batching under ADR 0023's composition rule: each combination
inherits the weakest member's contract; `exact` + f32 KV + f16-head +
single-sequence remains the only bit-exact configuration.

### Twin containment and lifecycle

- Every fast kernel is a **twin** under [ADR 0022](../adr/0022-twin-kernel-contract.md):
  a new member of its family, listed in the family table, pinned by the drift
  test, subject to the twin-sync rule. The exact twin is **never deleted** — it
  is the conformance reference (ADR 0018 order) and the CI default.
- CI runs the exact tier by default; no existing gate changes. The fast tier
  gets its own gate set (the bounds above) run in the GPU lane.
- **The 3% deletion rule is inherited in full** (ADR 0023; enforced four times
  since — rmsnorm_fast, T3b, T3c, and L4's byte-exact revert): a fast variant
  that wins <3% e2e *in its scoped mode* is deleted, and the verdict plus
  rebuild detail goes in OPTIMIZATION-LOG. "No win" is a valid result.

### What the tier may NEVER touch

- **Tree-verify acceptance arithmetic** (ADR 0014): the accept/reject rule
  preserves the target distribution exactly — losslessness is a product
  contract (`cuda_tree_verify_greedy_lossless` and the I2/I3/I4 slot gates).
  The verify forward's *attention* may eventually use fast twins under this
  RFC's bounds, but the acceptance comparison, commit ordering, and the ctrl
  plane's bitwise slot gates are out of bounds unconditionally.
- **The canonical order itself** (ADR 0018): the exact tier's reduction order
  definition, its host implementation, conformance vectors, and testkit
  oracles. The fast tier deviates *from* the reference; it never redefines it.
- **Training kernels** (`train_grad.cu`): out of scope; decode-only RFC.
- **The drafter head** (ADR 0032: τ −24%, explicit do-not-build) — quality
  axis, not a numerics tier question.

## Alternatives

- **No tier; keep only sanctioned one-off exceptions** (the f16 head model,
  ADR 0013 + decode.cu:3072-3090). Rejected: L3b and L8 are whole-kernel-class
  rewrites, not one comment's worth of exception; ad-hoc exceptions without the
  spec-interaction clause is exactly what produced this week's two scoping
  surprises.
- **Relax the default and drop bit-exactness** — rejected outright; the exact
  tier is the repo's testing lever and customer-facing identity (ADR 0018
  alternatives table; ADR 0022 "what the numerics contract demands").
- **Graph-path-only divergence** without an env tier: rejected in ADR 0018
  already ("permanently forfeits cross-backend bit-parity — worse than either
  consistent choice").
- **Per-request tier selection**: rejected — graphs bake symbols at capture
  time; a mixed-tier process cannot state which numerics domain answered.

## Compatibility and migration

- Default behavior unchanged: `exact` remains the default, CI identity, and
  conformance reference. No existing gate, golden file, or conformance vector
  changes.
- The book's "Numerics domains" table gains the tier row (ADR 0023's
  composition rule wording) and the healthz field is already shipped.
- ROCm/Metal/wgpu: the env var is CUDA-only until a fast twin is ported;
  other backends reject `fast` loudly (the `TRITIUM_KV_F16` ops-note pattern,
  ADR 0020).
- ADR 0022's revisit math: fast twins are a new axis on the attention families.
  If consolidation triggers, tier becomes a template axis (ADR 0023 already
  said this) with the SASS re-proof procedure applying to the exact
  instantiations only.

## Evidence and verification

Acceptance checklist for the **first consumer (L3b)** — the gates it must ship
with, all green before any adoption decision:

1. Kernel gate: fast attention outputs ≤1e-4 max rel vs the exact twin at
   fixed shapes (short and long ctx, all KV rungs it claims).
2. Tier drift gate: final-logit rel err ≤2e-3 vs the host exact oracle over
   the parity corpus, run with the full fast tier active.
3. Quality gates: ppl ratio ≤1.001 (5100-position teacher-forced harness);
   greedy 256-token identity vs the exact tier; ≥96-token transformers prefix.
4. Ctx-split perf: ledger-command ABBA at short ctx AND ctx≥4K, plain decode,
   measured ON the rung that owns each regime (long ctx = f16-KV per L6; a
   f32-only fast kernel repeats T3b's mistake and is rejected at review).
5. Spec leg: spec e2e + τ under the spec_kv_bench rig, same ctx split;
   per-mode verdict recorded (plain-only adoption is a legitimate outcome).
6. Containment: ADR 0022 family-table + drift-test update; exact twin
   untouched; graphs bake the tier at capture; healthz shows it.
7. The 3% rule: ≥3% e2e in the claimed mode or the variant is deleted with
   the verdict logged.
8. Gate L3 (ADR 0036) still applies on top: attention decode GPU-time share
   reduced ≥25% relative at fixed shapes.

Structural evidence (SASS/bit gates) and physical evidence (quiet-box ABBA
ledger runs, box-state disclosed) are distinguished per the ADR 0036
measurement discipline; L2-resident byte results are inadmissible.

## Security and privacy

No trust-boundary change. The tier is disclosed in `/healthz` so operators can
see which numerics domain answered; the book documents that `fast` is not for
reproducibility-critical use (ADR 0023 risk note carried forward).

## Unresolved questions

- Whether the L3b fast kernel needs a ctrl-driven n_split (T3b's revisit note)
  or a true online-softmax single pass — an implementation choice inside the
  bounds, not a contract question.
- Whether the 2e-3 tier drift gate runs per-PR in the GPU lane or per-release;
  proposal: per-PR alongside the existing GPU gate suite (it is one forward
  pass over the parity corpus).

## Decision record

Requested from: Brian Lam. Accepting authorizes L3b and the L8 flash-prefill
rewrite under the bounds above, each still gated by its own checklist and the
3% rule. Rejecting closes ADR 0036's RFC item and permanently bounds decode
attention at the exact tier's floor (round 26: the pinned sums are the last
attention cost standing). Decision, date, rationale, and dissent to be recorded
here; on acceptance, ROADMAP's ADR 0036 row and the book's numerics-domains
table are updated in the same commit.
