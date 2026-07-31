# SALT whitepaper — status after whole-paper coherence pass (2026-07-30)

## Per-section word counts

| file | words | state |
|---|---:|---|
| sec0-abstract.md | 426 | drafted |
| sec1-introduction.md | 1,776 | drafted |
| sec2-representation.md | 1,931 | drafted |
| sec3-fitting.md | 1,695 | drafted |
| sec4-curvature.md | 1,658 | drafted |
| sec5-recovery.md | 1,418 | drafted (Fig 5.1 placeholder) |
| sec6-systems.md | 2,126 | drafted (quiet-box rerun owed) |
| sec7-related.md | 1,783 | drafted (LC-QAT bit-width unsourced) |
| sec8-9-limitations-repro.md | 1,323 | drafted ([REPO-URL]/[PAPER-REV] placeholders) |
| **total prose** | **14,506** | ~9–10 pages two-column before figures |

Support files: `salt-whitepaper-outline.md` (claim ledger), `references.md`
([REF:*] key ledger), this file.

## Coherence pass — what was checked and fixed

**Notation (one symbol set).** Canonical set: planes $T_p$ (trits), scales
$s_{g,p} \ge 0$, plane count $P_g \le 3$, group size G128 / macrotile 256,
fitter metric $H$ (constructed in §4 as $M_{r,g} = f_r G_g + \lambda I$, now
cross-tied in both sections). Fixed drift: §5 used $T$ for plane count
($T{=}2$, $Q_T$, $T>2$) → now $P$; §5 gap/recovery symbols $G$/$R$ collided
with Gram blocks, the §3.2 plane matrix, and the R2/R3 rate points → now
$\gamma$/$\rho$; §3.2 plane matrix $G$ → $A$; §6.6 cost model reused $V$ and
$P$ (vocabulary / plane count elsewhere) → $t_V$, $t_d$, $t_P$; §3's K-FAC
ref unified to `kfac2015`.

**Cross-references.** C1–C6 ↔ §2–§6 mapping verified; §-pointers resolve in
both directions; abstract numbers now match section numbers (decode band
harmonized to the session-median span ≈273–303 tok/s in sec0/sec1/outline;
sec6.4 keeps the raw 264–303 spread with medians). The §5.1 "500k-token
pool" vs §5.2/abstract "480k" reconciled per ADR 0029 (500k pool, 480k
consumed as training windows). `[REF:bonsai]` vs `[REF:bonsai27b]` are two
distinct PrismML releases — renamed the former `bonsai-family` instead of
merging.

**Claim boundary.** Swept for SOTA/state-of-the-art/fastest/superior/lead
phrasing. Fixed: sec1 C6 "most-optimized mainstream 4-bit CUDA path" → "a
mature mainstream 4-bit CUDA path (llama.cpp Q4_K_M)"; sec6.4 "Tritium
retains the lead" → measurement-scoped ("on this box, on this day, …
measured higher"). Remaining superlatives are all reported-by-others in
related-work context (§7) or internal measured dominance with oracle tests
(§3.4), which is inside the boundary.

## Remaining placeholders

1. **Figure 5.1** — placeholder. Needs the 3-seed error-bar rerun
   (`TRITIUM_DISTILL_SEED` axis); interim single-seed curve
   1286→365→194→140 at steps 500/5000/11000/14500 is committed.
2. **[REPO-URL] and [PAPER-REV]** (§9) — public Tritium repo URL and frozen
   paper revision, inserted at submission.
3. **NEEDS CITATION** (22 keys, see `references.md`): bastion2026, bcjr-qat,
   blockgtq, bpdq, catq, dfssm, fairyfuse, fisher-kron, guidedquant, hestia,
   kronq, littlebit, llvq, mote, oaem, pt2llm, ptqtp, slidesparse, tequila,
   unisvq, vbq, veclut.
4. **VERIFY** (5 keys): bitnetcpp, lcqat, op2026, paretoq, yaqa — targets
   proposed, ids must be confirmed before the bibliography freezes.
5. **INTERNAL** (3 keys): refuted-quality-table, refuted-ptq-comparison,
   refuted-kernel-claims — resolve to companion-repo verification records;
   decide the citation form (repo doc vs underlying artifacts) at
   bibliography time.
6. **Quiet-box rerun** of the llama.cpp Q2_0 same-box comparison (Table 6.2
   is publication-final only after it; debt recorded in §6.4 and §8.5).
7. **LC-QAT bit width** (§7.1 UNSOURCED note) — restore "at 2 bits" only
   after checking arXiv 2606.10531 directly.

## Ordered TODO to submission

1. **Three-seed Figure 5.1 data** (rotating `TRITIUM_DISTILL_SEED`) — the
   only new training compute the paper needs; replaces the single-seed
   curve and de-caveats §8.2.
2. **Quiet-box rerun** of the llama.cpp Q2_0 comparison (also owed in the
   BENCHMARKS.md ledger); update Table 6.2 and, if moved, the abstract/C6
   decode band.
3. *(Optional, cheap)* **P>2 planes on the Step-1 corpus** — strengthens the
   C1↔C4 link; currently recorded as an untested lever in §5.4.
4. **Resolve the citation ledger**: 22 NEEDS CITATION + 5 VERIFY keys in
   `references.md`; decide the form of the 3 INTERNAL refutation keys.
5. **Freeze the paper-repo revision**; fill [REPO-URL]/[PAPER-REV]; then
   regenerate every table from that revision (relay_basin_ab,
   ws_a1_cost_baseline.py, salt_distill_heldout curve, `tritium report
   compare`).
6. **LaTeX conversion**: sections → one arXiv source; keep the
   `<!-- receipt: -->` provenance as LaTeX comments (or an appendix table);
   math is already LaTeX-compatible; tables 3.6/4.5/5.1/6.1/6.2 → booktabs.
7. **Bibliography**: generate the .bib from `references.md` once resolved;
   swap [REF:key] → \cite{key}.
8. **Figure generation**: Fig 5.1 (recovery vs tokens, both LR schedules,
   error bars); consider a small Fig 6.1 (prefill arc) from Table 6.1 data.
9. **Final pass on the LaTeX**: re-run the notation/claim-boundary greps,
   check the abstract against final section numbers, page budget.
