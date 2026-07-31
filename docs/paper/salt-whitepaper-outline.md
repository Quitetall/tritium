# SALT: Sensitivity-Allocated Layered Ternarization with Exact Byte Accounting
## arXiv whitepaper — outline + claim ledger (draft 0, 2026-07-30)

**Category:** cs.LG (cross-list cs.AR). **Type:** methods + systems paper.
**Companion artifact:** the Tritium repo (Apache-2.0) — every number in the
paper regenerates from a committed command.

**Claim boundary (governs the whole draft):** this paper claims a METHOD and a
MEASUREMENT DISCIPLINE, demonstrated at small-to-mid scale. It does NOT claim
SOTA quality, does not claim the 27B result (plan 0043's gates own that), and
does not claim byte-optimality over fp frontiers (plan 0042's future paper owns
that once the multi-scale held-out curves exist). Every table cell cites a
receipt path or a BENCHMARKS.md ledger entry.

---

### 1. Introduction
- Ternary's moment: BitNet 2B4T → Falcon-Edge → Bonsai 27B → llama.cpp Q2_0
  CUDA (merged 2026-07-30). The gap: quality claims in this space routinely
  fail independent verification (cite our 3 documented refutations —
  vendor-neutral phrasing); nobody ships quantize→train→serve in one auditable
  lifecycle.
- Contributions (each maps to a section + receipt):
  C1. SALT: additive ternary planes `W ≈ Σ_p s_p·T_p`, zero-point-free,
      with per-group plane allocation by measured curvature.
  C2. Exact joint fitting: 3^P assignment + conditioned scale solve +
      accept-only-if-improves E/M — with a proven-optimal-on-tractable-cases
      solver and deterministic multi-start (incl. the softened-relay basins,
      never-worse by construction, measured: P=3 improves 7.4% of groups).
  C3. Kronecker curvature evidence: input-Gram × output-Fisher records with
      content-addressed identity; shared-forward capture proven byte-identical
      (3× replay reduction measured at 1.7B).
  C4. Distillation recovery: SALT-STE end-to-end training defeats catastrophic
      ternary PTQ — WikiText-2 held-out: 3.28e6 → 139.6 ppl (23,493×), with
      the honest finding that the curve is token-limited, not floored.
  C5. Physical-byte accounting as a first-class invariant: every artifact
      reports exact serialized + resident bytes; "logical bpw" is banned from
      claims. (This is the paper's quiet thesis: the field's comparison
      hygiene problem is fixable mechanically.)
  C6. Systems: the native engine executing packed planes without dense
      materialization — 4090: ~280-300 tok/s decode on 2B4T (~474 GiB/s
      effective weight stream), 12.3K pp512 via bit-identical IMMA.

### 2. Method — SALT representation (from ADR 0001/0028)
- Format: planes + non-negative f16 group scales; G128; D2/B3/S34 codecs;
  what zero-point-free buys (kernel simplicity, add/sub/skip execution).
- Rate points and exact byte ceilings (R2/R3 framing, minus 27B specifics).

### 3. Fitting — exact joint solve + basins (from salt_v2.rs, receipts)
- E/M with monotone acceptance; brute-force-optimality goldens; determinism.
- Relay basins (CAT-Q-inspired, cited): basin-internal modulation, projection
  through the exact solver; TABLE: P∈{1,2,3} win-rate/median-improvement
  (relay_basin_ab receipts).

### 4. Curvature — Kronecker evidence + shared-forward capture
- S2KF identity design (global-sample-ordinal dyadic reduction) → capture
  topology provably outside evidence identity; TABLE: A1 receipt (24 tensors,
  3.0× replay reduction, byte-identity TRUE, Amdahl split 22/78 with the
  scaling argument stated as assumption, per plan-0043 forecast rules).

### 5. Recovery — distillation defeats PTQ (Step-1 arc, ADR 0029/0031)
- FIGURE (money plot for THIS paper): held-out ppl vs training tokens —
  constant-LR vs scheduled; the LR-artifact correction; token-limited finding.
- TABLE: 23.83 fp | 3.28e6 PTQ | 139.6 recovered; 92.6% atomic recovery cite.
- Explicit negative-result subsection: uniform ternary KV rejected by
  measurement (ADR 0020 rung 3); rmsnorm_fast rejected (+1.75% < bar) —
  demonstrates the discipline is real.

### 6. Systems — execution without dense weights
- Kernel family (dp4a decode, IMMA prefill w/ bit-identity gate, split-KV,
  tree-verify spec decode w/ losslessness gates); BENCHMARKS.md ledger method;
  same-box llama.cpp Q2_0-day-one comparison (bandwidth-normalized, contention
  disclosed).

### 7. Related work
- QAT: BitNet family, ParetoQ, Tequila, HESTIA. PTQ: PT²-LLM, CAT-Q, PTQTP,
  QTIP/lattice line. Theory: Ordentlich-Polyanskiy product-distortion frame
  (the information-theoretic justification for output-aware objectives).
  Position SALT as: additive-ternary execution invariant + evidence apparatus;
  the combination, not any single mechanism, is the contribution.

### 8. Limitations (load-bearing section, not boilerplate)
- Scale ceiling of current evidence (135M-1.7B; 27B preregistered, unrun).
- Token-limited recovery curves; no floor claims.
- Single-GPU-class benchmarks; contention documented.
- Basin wins are group-level; model-level effect awaits the Stage-7 bracket.

### 9. Reproducibility statement
- One-command regeneration per table; receipt schema; the ledger discipline.

---
## TODO before submission (ranked)
1. [ ] Rerun Step-1 figure data with error bars (3 seeds) — the only new
       compute this paper needs (~GPU-hours, local).
2. [ ] Quiet-box rerun of the llama.cpp comparison (owed in ledger anyway).
3. [ ] T>2 planes on Step-1 corpus (cheap, strengthens C1+C4 link).
4. [ ] Freeze paper-repo revision; regenerate all tables from it.
5. [ ] Draft §2-5 from ADR 0001/0028 + code comments (no new results needed).
