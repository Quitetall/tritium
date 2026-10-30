# Ternary PTQ frontier: October 9 delta

**Research date:** 2026-10-09
**Comparison baseline:** [2026-10-08 frontier supplement](ternary-ptq-frontier-2026-10-08.md)
**Scope:** Targeted search for primary-source papers or official code/repository changes newly surfaced since that supplement which could materially affect Tritium v1.1 ternary PTQ or refinement priorities. This is a delta, not a fresh survey. No Tritium code, model, or benchmark was run.

## Bottom line

One relevant primary-source paper surfaced: **Distributionally Robust Quantization (DRQ)**, submitted to arXiv on October 8. Since the preceding note is also dated October 8 and arXiv reports only the submission timestamp (October 8, 04:22:29 UTC), the exact publication-after-note ordering cannot be established from the date alone. It was absent from the preceding note and was verified in the current scan.

DRQ is a useful refinement hypothesis, not evidence for ternary quality or a reason to change Tritium's frozen recipe. Its strongest implication is already reflected in Tritium's existing gate: a lower calibration reconstruction objective must not replace a candidate with worse held-out quality. Add DRQ only as a separately identified, controlled future ablation if its code-space and objective can be faithfully adapted to the selected Tritium artifact.

## Newly surfaced work

| Work and primary evidence | What authors report | Tritium relevance and disposition |
|---|---|---|
| **When Lower Reconstruction Loss Hurts: Distributionally Robust Refinement for Low-Bit LLM Quantization (DRQ)**. [arXiv record](https://arxiv.org/abs/2610.11226), [version 1 full text](https://arxiv.org/html/2610.11226v1). The arXiv record says submitted **2026-10-08 04:22:29 UTC**; v1 is dated 2026-10-08. | DRQ post-hoc searches discrete weight codes while holding the quantization grid and parameters fixed. Its objective accounts for bounded changes to the input-activation second moment, instead of minimizing only calibration reconstruction error. The authors report experiments on Llama-3.2 1B/3B, Llama-3.1 8B/70B, and Qwen3-30B-A3B, across W2/W3/W4 and six initial PTQ methods. For example, their Llama-3.1-70B GPTQ W2 row reports C4 perplexity 31.77→27.11 and mean task accuracy 48.23→54.56; their Qwen3-30B-A3B GPTQ W3 row reports mean accuracy 72.12→72.90. Their measured Llama-3.1-8B W3 refinement averages about six minutes on the reported setup. These are author-reported results. | **Not a ternary result:** the paper's experiments use conventional integer W2/W3/W4 quantization, not {-1,0,+1} planes, additive SALT, or Tritium artifacts. Its fixed-grid code-edit idea may transfer in principle, but changing trits in an additive multi-plane representation changes the feasible code geometry; a direct port or expected gain cannot be inferred. The paper also does not link an official implementation in its arXiv record. **Plan status:** informs a future, separately preregistered refinement ablation only; it does not amend ADR 0035, plan 0043, plan 0054, selected methods, or frozen gates. No Tritium reproduction was performed. |

## Effect on existing priorities and gates

- **No current-plan change.** Plan 0043 Stage 5 already says that a candidate with lower reconstruction loss may not replace a checkpoint with worse held-out perplexity or teacher KL. This is consistent with DRQ's motivation and should remain binding; the paper does not justify replacing Tritium's objective or changing any acceptance threshold.
- **Future experiment, if admitted:** compare the frozen parent with a separately identified DRQ-inspired child on the exact same admitted ternary artifact, calibration data, held-out data, and compute budget. Keep the representation fixed; charge any extra fitting cost; record calibration reconstruction, held-out perplexity/teacher KL and task metrics, and reloaded hard-code/artifact parity. Reject candidates that improve the local objective but regress held-out quality. For additive planes, first define the feasible discrete update and prove that its serialized artifact preserves the declared format.
- **Evidence boundary:** DRQ's reported results are not independent replication, not Tritium measurements, not proof of compatibility with additive ternary, and not an SOTA ranking. The scan found no other new primary-source work in the bounded October 8–9 interval that materially changes Tritium's ternary/PTQ priorities; this absence is not a claim of an exhaustive literature search.

## Validation sources

- Primary paper metadata and submission history: [arXiv:2610.11226](https://arxiv.org/abs/2610.11226).
- Method, evaluated model/bit-width scope, reported results, and limitations summarized above: [arXiv v1 full text](https://arxiv.org/html/2610.11226v1), especially Sections 3–4 and Appendix D/E.
- Existing safeguard checked against [plan 0043, Stage 5](../plans/0043-salt-v2-sota-campaign.md#stage-5--output-reconstruction-and-refinement).

This note records author claims as author claims. No source code was executed and no result here constitutes a Tritium reproduction or qualification receipt.
