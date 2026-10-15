# Original salience execution and cold replay

Implementation of proposed ADR0064's original-A execution amendment, over the
distinct structural contract in67e59743. Verification is recorded separately
from frozen actual command/source evidence; this note does not assert a PASS.

SourceCalibrationModel records actual token or private text windows against
ObservedWeightSource's complete raw identity and immutable masters. Constructor
config/master budgets must match the request; window admission may be narrower.
Every window/aggregate row/moment/source/asset bound is checked before moment
allocation or block execution. Complete checked raw source reads and closing
inventory checks do not lock mutable files or qualify process RSS.

The controller uses existing calibrate_tapped, Calib and accumulate. All tokens,
including final tokens and one-token windows, pass through the shared block graph.
It accumulates the exact ordered legacy f64 sums once per attention/FFN/down input
group per layer, checks actual hook coverage/order/shape/rows and finite inputs/
sums, and moves output vectors without cloning another dense model. Vocabulary
head, loss and backward are not part of forward-only salience.
The existing per-window tape still clones master leaves and retains graph values;
this is not a no-grad, bounded-RSS or out-of-core executor.

NameSchema and immutable graph dimensions supply q/k/v+attention norm,
gate/up+FFN norm and down+up-output-row pairs. Caller Arch, pair inventory,
statistics and provenance hashes cannot replace this actual controller context.
Returned statistics and sealed evidence have read-only accessors. The existing
transform still explicitly uses Declared statistics until the mandatory A/B/C
capture ancestry/carrier amendment is implemented; no B or C promotion occurs.

One private runtime tokenizer helper serves A and B. B's minimum-two-token,
final-target omission and row/objective/cache/profile semantics stay unchanged.
A executes all tokens, including one-token windows. Neither adds special tokens,
chat templates, padding, truncation or fallback. Raw text is private evidence.

Cold replay admits recorded caps against a separate receiver, observes actual
raw A through the existing observation-only source seam, rebuilds checked
immutable masters, reruns actual encoding/block graph/reducer, and compares the
complete derived request, windows and ordered moment bindings. No supplied model,
stored activations/gradients, copied hash preimage or numerical formula replaces
the rerun. Observation-only KeepDense metadata executes no fit/package/fake B.
Without the tokenizer feature, raw-text replay rejects before any source call
or dense decoding; receiving-limit admission still precedes that capability error.

Different raw strings may encode identically and produce identical moments.
Their sealed input contexts differ. Numerical replay establishes agreement for
an input claim, not historical authorship/authentication; mandatory future
carrier ancestry must reject substitution of a new reference for the old one.
Changed encoded windows or fabricated moments/counts/consumers/source must reject.

Tests use actual toy llama/qwen3 files, F32/F16/Bf16, legacy calibration and
paired-fold byte differentials, actual external safetensors mutation, raw source
and receiving-cap rejection, actual tokenizer and cold replay. These are software
fixtures, not official-Qwen quality/performance/physical-hardware qualification.
Biased/untied/hybrid/MoE/MTP/gather extensions, full source/parser/tape/RSS/GPU/
out-of-core admission, mandatory A/B/C ancestry, surfaces and both campaign
migrations plus every full ADR0033 product/empirical/release gate remain required.
