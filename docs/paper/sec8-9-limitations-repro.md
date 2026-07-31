# 8. Limitations

This section is not boilerplate; it is the other half of the claim boundary
stated in Section 1. Each limitation below marks a place where the evidence
stops and a preregistered or recorded follow-up begins, and several of them
withhold claims a less constrained reading of our own tables might suggest.

## 8.1 Scale ceiling of the evidence

Every method-evidence result in this paper — fitting, capture, and recovery —
sits between 135M and 1.7B parameters: the recovery arc of Section 5 is
SmolLM2-135M, and the fitting and capture measurements of Sections 3–4 are
SmolLM2-1.7B. (The runtime measurements of Section 6 execute a 2.4B ternary
checkpoint, but they measure the engine, not the method.)
<!-- receipt: docs/adr/0029-training-throughput-tensor-cores.md:168-189 (135M recovery); docs/receipts-ws-b-relay-basin-ab.txt and docs/receipts-ws-a1-cost-baseline-17b.json (1.7B); docs/BENCHMARKS.md 2026-07-30 entry (BitNet 2B4T, the 2.4B runtime checkpoint) -->
The flagship campaign — Qwen3.6-27B at a pinned immutable revision, with all
506 in-scope rank-2 matrices (27,318,026,240 coefficients) in the allocation
domain — is preregistered with frozen quality, rate, parity, provenance, and
publication gates, and it is unrun: the plan's own header records
"Claim status: not achieved," and a 2026-07-30 amendment adding this paper's
solver variants to the ablation grid explicitly changed no frozen threshold,
token cap, or track-separation rule.
<!-- receipt: docs/plans/0043-salt-v2-sota-campaign.md (header "Claim status: not achieved"; 506 matrices / 27,318,026,240 coefficients; Amendment 1 "changes no frozen threshold, no successive-halving protocol, no token cap") -->
Nothing here is evidence that the method's behaviour at 135M–1.7B transfers
to 27B. The preregistration exists precisely so that when that run happens,
its thresholds cannot have been chosen after seeing the data; until then, the
27B campaign is future work, not a result.

## 8.2 Token-limited recovery; no floor, no convergence claim

The full-pool recovery run of Section 5 never plateaued: it was still
descending when its 480k-token pool ran out. We therefore claim neither that
SALT-STE distillation converges to the fp teacher nor that a small-model
ternary floor exists — the data supports no asymptote of either sign, and
Section 5.3 records how a previous "ceiling" of ours dissolved once the
learning-rate schedule was corrected.
<!-- receipt: docs/adr/0029-training-throughput-tensor-cores.md:186-192 -->
The committed curves are additionally single-seed. The stochasticity axis is
a data-order seed (`TRITIUM_DISTILL_SEED`, default 0 reproducing the
committed runs), and three-seed error-bar reruns along that axis are the
top-ranked pre-submission item; until they land, Figure 5.1 carries
single-seed data and should be read with that caveat.
<!-- receipt: crates/tritium-nn/tests/salt_distill_heldout.rs:474-479; docs/paper/salt-whitepaper-outline.md:93-94 -->
The held-out corpus is WikiText-2 only; larger corpora are recorded as
untested levers, not tested ones.

## 8.3 Basin wins are group-level, under an identity metric

The softened-relay result of Section 3.6 — improvements on 7.4% of groups at
$P{=}3$ — is a *group-level objective* improvement, measured under the
identity metric on one checkpoint.
<!-- receipt: docs/receipts-ws-b-relay-basin-ab.txt -->
Two gaps remain between that and a claim anyone should act on. First, the
production fitter runs under the curvature metrics of Section 4, and the A/B
has not been repeated under a curvature metric; the deployed win rate may
differ in either direction. Second, group-objective improvements need not
move model quality at all. Both questions are assigned to the preregistered
Stage-7 successive-halving bracket, where the basin variants enter as
ablation grid rows under unchanged thresholds; the measurement harness
reports, it does not decide.
<!-- receipt: docs/plans/0043-salt-v2-sota-campaign.md (Amendment A1.1 grid additions; Stage 7); docs/plans/0054-frontier-methods-integration.md (Gates B: "must survive successive halving (thresholds unchanged)") -->

## 8.4 The shared-forward scaling argument is an assumption

Section 4.5's honest decomposition bears repeating as a limitation: at 1.7B
the $3.0\times$ replay reduction bought only a $1.17\times$ wall-clock
speedup, because the forward pass is only $\sim$22% of per-tensor capture
wall and the remaining $\sim$78% (Gram accumulation and writing) does not
shrink with grouping.
<!-- receipt: docs/receipts-ws-a1-cost-baseline-17b.json (wall times, replay_reduction=3.0, speedup_wall=1.173); decomposition in docs/plans/0054-frontier-methods-integration.md §A1/A4 -->
The argument that the forward's share — and hence the value of grouping —
grows with model size is stated as an assumption under the campaign's
forecast rules, and a same-harness measurement on sliced 27B layers must
replace it before it informs any spend decision. The measured cost table
also covers attention projections only.
<!-- receipt: docs/plans/0054-frontier-methods-integration.md §A4 (forecast rules per plan 0043) -->

## 8.5 Single hardware class

Every runtime and cost number in this paper was measured on one GPU class —
a single RTX 4090 — under disclosed co-resident load; the ledger's
methodology notes that numbers move ±10% under desktop contention, and the
same-box llama.cpp Q2_0 comparison explicitly owes a quiet-box rerun before
any published table.
<!-- receipt: docs/BENCHMARKS.md (methodology: ±10% contention, co_resident capture; 2026-07-30 entry "quiet-box rerun owed"); docs/plans/0054-frontier-methods-integration.md §A1/A4 (RTX 4090); docs/adr/0029-training-throughput-tensor-cores.md:146 -->
Bandwidth normalization (effective weight-stream GiB/s) is the cross-model
metric precisely because absolute tok/s on one contended box generalizes
poorly, but normalization does not substitute for measurements on other
hardware classes, and we make no claims about them.

## 8.6 Q2_0 interoperability is ternary-subset only

The interoperability of Section 2.7 is deliberately asymmetric. Export to
llama.cpp's Q2_0 is always well-defined, because Tritium tensors are ternary
by construction; import, however, rejects the format's fourth level ($+2$)
as out of range. Tritium therefore cannot round-trip an arbitrary Q2_0
artifact, only its ternary subset, and comparisons against the mainstream
runtime are correspondingly limited to models that are ternary in both
containers.
<!-- receipt: crates/tritium-format/src/q2_0.rs (code-3 rejection) -->

# 9. Reproducibility Statement

**Companion artifact.** The Tritium repository (Apache-2.0) is the paper's
companion artifact; the public URL and the frozen paper revision will be
inserted here at submission. [REPO-URL], revision [PAPER-REV].

**Receipt schema.** Every empirical sentence in this paper's source carries a
machine-checkable provenance comment, `<!-- receipt: path -->`, resolving to
a committed artifact: an ADR line, a receipt file with a versioned schema
string (e.g. `tritium.ws-a1-cost-baseline.v1`), a code constant, a test
name, or a benchmark-ledger entry. A number the authors could not source did
not enter the text.
<!-- receipt: docs/receipts-ws-a1-cost-baseline-17b.json ("schema" field); the convention is visible in the source of every section of this paper -->

**One command per table.** Each table regenerates from a single committed
command: the basin A/B table of Section 3.6 from the `relay_basin_ab` harness
against the pinned SmolLM2-1.7B checkpoint (arguments recorded in the receipt
header); the capture-cost table of Section 4.5 from
`tools/ws_a1_cost_baseline.py` on the frozen calibration inputs;
Table 5.1 and Figure 5.1 from the held-out distillation gate with its
documented environment knobs (`TRITIUM_DISTILL_CURVE` for the
recovery-vs-tokens curve, `TRITIUM_DISTILL_WARMUP` for the schedule,
`TRITIUM_DISTILL_SEED` for data order); and the Section 6 tables from the
ledger command `tritium report compare`, which emits the JSON artifact the
tables transcribe.
<!-- receipt: docs/receipts-ws-b-relay-basin-ab.txt (harness + args header); docs/receipts-ws-a1-cost-baseline-17b.json; crates/tritium-nn/tests/salt_distill_heldout.rs:442-479; docs/BENCHMARKS.md ("The command") -->

**Ledger rules.** The benchmark ledger's standing rule is that no number is
recorded without the exact command beside it and the environment it ran in —
GPU, driver, and co-resident processes — and the ledger is updated only by
re-running the harness. Competitor numbers carry their exact invocations; we
document their reproduction lines rather than wrapping their binaries.
<!-- receipt: docs/BENCHMARKS.md:1-7,33-35 -->

**Determinism.** Three guarantees make "same inputs, same bytes" a tested
property rather than a hope. The fitter contains no randomness and is
bitwise repeatable, asserted structurally in tests.
<!-- receipt: crates/tritium-quantize/src/salt_v2.rs test fitting_is_bitwise_deterministic -->
Curvature capture accumulates through an ordinal-keyed canonical dyadic
reduction, so batch boundaries, resume points, and capture topology provably
cannot alter the emitted records — held on a real model by the byte-identity
result of Section 4.5.
<!-- receipt: crates/tritium-quantize/src/salt_v2_curvature.rs (canonical_siblings); docs/receipts-ws-a1-cost-baseline-17b.json (byte_identity=true) -->
And as end-to-end evidence that the discipline survives contact with a real
experiment: the committed relay-basin A/B receipt is a checkpoint-pinned
rerun whose output is bit-identical to the harness's first recorded run.
<!-- receipt: docs/receipts-ws-b-relay-basin-ab.txt; first run at commit 04ef528, bit-identical rerun committed at 71a6671 -->
The one stochastic axis in the training results — data order — is an
explicit seed defaulting to the committed value, so reproducing the paper's
numbers requires the documented schedule and curve knobs above but no seed
flag.
