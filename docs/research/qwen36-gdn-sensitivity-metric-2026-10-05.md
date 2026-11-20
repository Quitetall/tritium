# Qwen3.6 GDN sensitivity metric review

**Status:** research review of proposed [ADR 0049](../adr/0049-qwen36-gdn-sensitivity-metric.md); not an accepted protocol amendment.

## Current finding

The exact metric is already specified in ADR 0049, which I missed in the first
version of this note. That ADR proposes absolute RMS error for both the
final-normalized hidden vector and the selected recurrent state, pooled across
the 512 calibration sequences and all vector elements at each frozen position.
It requires f64 accumulation, rejects non-finite values, routes only on
terminal output RMS, and keeps state RMS diagnostic. My earlier relative-L2
recommendation duplicated and conflicted with that proposal; it is withdrawn.

The proposed absolute metric is defensible for this screen: its denominator
cannot become unstable for near-zero reference vectors, and output and state
curves remain separate measurements. Its limitation is that absolute RMS is
scale-dependent, so comparisons rely on the ADR's carefully matched probe
families, calibration sequences, positions, and source-precision controls. The
2x threshold is meaningful only under those frozen matching conditions.

Primary recurrent-quantization sources support measuring accumulated state
error rather than relying only on local weight error, but do not dictate this
particular formula or Tritium's routing threshold. [DAMP](https://arxiv.org/abs/2608.27513)
defines accumulated state error as the difference between quantized and
unquantized recurrent states and analyzes its propagation. [Ternary Mamba](https://arxiv.org/abs/2606.18114)
reports that errors accumulate through recurrence and that Transformer-style
post-hoc correction can fail. These results support the motivation, not the
chosen numerical contract.

## Remaining work before measurement

ADR 0049 remains **PROPOSED**. Its own acceptance gates require owner acceptance,
a versioned v2 receipt and independent verifier that names the metric, exact
sample positions and recurrent observation layer, synthetic hand-calculated
tests, and a tiny deterministic pairing/restoration test. Only after those
software and decision gates—and separate explicit campaign authorization—may
the pinned-Qwen probes run. Existing v1 receipt numbers are diagnostic only and
must not be reinterpreted as v2 measurements.

This review is not a gate pass, does not authorize probes, and makes no claim
about Qwen quality or runtime.

## Sources and repository contracts

- [ADR 0049 — Reproducible Qwen36 recurrence-sensitivity measurements](../adr/0049-qwen36-gdn-sensitivity-metric.md)
- [Plan 0043, amendment A1.2](../plans/0043-salt-v2-sota-campaign.md#amendment-1--frontier-methods-candidates-and-preflight-gate-2026-07-30)
- [Plan 0054, workstream D](../plans/0054-frontier-methods-integration.md#workstream-d--gdn-sensitivity-preflight-gate-pre-stage-8)
- [Release-candidate status](../release-candidate.md#gdnsensitivity-receipt-verifier-alignment-2026-10-05)
- [Current v1 receipt verifier](../../scripts/verify-qwen36-gdn-sensitivity.py)
