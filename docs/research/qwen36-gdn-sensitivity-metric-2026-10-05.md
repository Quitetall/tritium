# Qwen3.6 GDN sensitivity metric: definition needed before measurement

**Status:** research recommendation; not an accepted protocol amendment.

**Scope:** clarify the metric required by the frozen eight-probe preflight. No
model weights were loaded and no probe or campaign was run.

## Finding

The current plan freezes the probe classes, 512-sequence calibration partition,
terminal-depth comparison and a `2x` routing threshold, but it does not define
the numerical meaning of `output_divergence` or `state_divergence`. The receipt
verifier accepts any finite nonnegative values, so structurally valid numbers
cannot currently be reproduced or interpreted as a measurement. The measurement
producer must not fill those fields until the metric is adopted in the campaign
contract.

This omission matters for a recurrent model: local projection error is not the
same as accumulated state error, and neither is the same as final hidden-output
error. Recent recurrent quantization work explicitly models accumulated state
error as the difference between quantized and unquantized states and studies
how subsequent recurrence transforms it ([DAMP, §3.3](https://arxiv.org/abs/2608.27513)).
Work on ternary recurrent models likewise reports that error accumulates through
the recurrence and that Transformer post-hoc corrections can fail
([Ternary Mamba](https://arxiv.org/abs/2606.18114)). Neither source prescribes
the exact metric or threshold for this Tritium probe; selecting one is a local
protocol decision, not a fact to import from the papers.

## Recommendation for an explicit, reproducible metric

At each frozen one-based token depth `p`, and for each of the 512 sequences,
retain the dense-reference vector `r[s,p]` and the single-matrix ternary vector
`q[s,p]`. Compute two distinct curves:

```text
output_divergence[p] = sqrt(sum_s ||q_hidden[s,p] - r_hidden[s,p]||_2^2
                            / sum_s ||r_hidden[s,p]||_2^2)

state_divergence[p]  = sqrt(sum_s ||q_state[s,p] - r_state[s,p]||_2^2
                            / sum_s ||r_state[s,p]||_2^2)
```

The numerator and denominator include every feature/state element and every
sequence exactly once. Accumulate sums of squares in `f64`, then take one square
root after division. Do not average per-sequence ratios: that would give a
near-zero-norm sequence the same weight as a normal sequence. If a denominator
is exactly zero, emit a distinct metric error and do not mint a receipt; silently
adding an epsilon would change the metric and could change routing. Store the
final nonnegative scalar at each depth as finite `f64` JSON numbers.

Use `output_divergence` for the existing `2x` routing rule. Keep state divergence
as a required diagnostic curve, not an alternative routing criterion. This
preserves the plan's rule that routing depends on terminal output divergence,
while state values expose the recurrence mechanism and catch missing/misaligned
state sampling. Retain the existing per-tensor weight MSE as a non-binding
diagnostic.

The paired samples must be taken from the same exact calibration token sequence,
same token positions, same dense source snapshot and same runtime arithmetic
except for the one selected matrix. Compute differences after promoting sample
values to `f64`; reject non-finite input samples before aggregation. The receipt
should name the metric ID/version and its aggregation semantics so a future
implementation cannot reinterpret old numbers.

## Alternatives and tradeoffs

- **Absolute RMS error** is directly interpretable in activation/state units,
  and is close to the accumulated-state difference used in DAMP. But its scale
  varies across tensor families and layers, making the existing cross-family
  `2x` threshold less comparable.
- **Cosine distance** ignores vector magnitude. That can hide a material scale
  error in the recurrent state or final hidden vector, so it is not suitable as
  the binding metric here.
- **Mean of per-sequence relative errors** is also scale-normalized, but gives a
  nearly-zero-energy sequence excessive influence. The ratio of pooled squared
  norms avoids that behavior and is straightforward to stream.
- **Task loss or teacher-logit KL** is closer to end-user quality, but this
  preflight is intentionally an inexpensive routing screen. It should not be
  substituted for the separate Stage-8 quality gate.

## Required next change before any probes

The metric ID, pooled reduction, exact-zero-denominator behavior, precision of
accumulation, and output-vs-state routing roles need to be accepted as a
preregistered amendment before measurement. Then update the receipt schema and
verifier, add hand-calculated fixtures (including zero reference norm and
non-finite samples), implement the source-bound producer, and only afterward
run the eight probes on the approved calibration pack. This note alone does not
authorize measurement and does not satisfy the GDN-sensitivity gate.

## Repository contract inspected

- [Plan 0043, GDN-sensitivity preflight](../plans/0043-salt-v2-sota-campaign.md#amendment-1--frontier-methods-candidates-and-preflight-gate-2026-07-30)
- [Release candidate, verifier status and missing producer](../release-candidate.md#gdnsensitivity-receipt-verifier-alignment-2026-10-05)
- [Receipt verifier](../../scripts/verify-qwen36-gdn-sensitivity.py)
- [Probe sample seam](../../crates/tritium-nn/src/model/qwen35.rs)
