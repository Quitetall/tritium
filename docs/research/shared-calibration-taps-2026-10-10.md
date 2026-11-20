# One actual forward and caller-owned calibration taps

Candidate continuation of proposed ADR0060, based on
`3fa7b54b381704e6c0cfc68d59ef4099815a3763`. This is not a frozen profile,
full P3 completion, Qwen/MTP coverage or empirical release evidence.

The supported tied-head, full-RoPE GQA/SwiGLU graph now has one actual forward
implementation. Plain forward, activation-transformed forward and legacy
three-role fold calibration are adapters. The legacy fold still stops before
the vocabulary head and emits only its original three input roles; it does not
gain a new covariance/Fisher claim or incur an unnecessary head allocation.
The unused private duplicate `calibrate_old` implementation is removed, not
any legacy artifact/receipt producer or accepted fixture contract.

The underlying actual attention implementation returns original Q/K/V output
ValueIds together with concatenated head outputs. Q/K IDs are before RoPE, gate
IDs before SiLU, and Output/Down/Head IDs before their downstream operations.
`forward_observed` exposes one typed runtime observation per actual projection
consumer on a caller-owned `Tape`. Q/K/V share an actual input ID; gate/up share
another. Tied embedding/head share the original weight ID/index but expose
distinct output IDs and consumers. Weight indices stay parallel to the existing
loader-owned naming convention rather than inventing a second naming scheme.

Callbacks borrow tape storage only for the call. They may retain value-ID metadata
for the caller's actual declared-objective backward phase, but cannot retain an
internal borrowed tape. Actual output gradients come from `Tape::backward`, not
from squared activations or a hand-written gradient proxy. Gather exposes no fake
dense one-hot/input ID; its output width is hidden width, not vocabulary width.
Mapping full gather-output gradients to an admitted indexed Fisher approximation
still needs a separately explicit estimator/binding; this change does not invent
one or claim that the gather tap alone clears that obligation.

Runtime observations are not registered evidence or cache/source verification.
The existing shape/ValueId/architecture contracts still apply; unsupported hybrid,
MTP, untied-head, partial-RoPE or other graphs are not covered by this tied standard
graph. A source/cache-spooling and objective identity protocol, inventory admission,
source-byte binding, row-aware fit/log/package integration, full RSS admission and
the conversion/CLI/Qwen/Python/HF adapters remain ADR0060 follow-up work.

Tests exercise the actual graph. Unique layer weights' observed pre-transform
output gradients reproduce the existing dense VJP, identifying misplaced IDs.
Adapters preserve logits and the same legacy diagonal reductions. Actual toy
post-objective buffers are hashed as a finalized diagnostic test cache before
replay into the shared collector; this is not canonical source inventory or a
registered release receipt. An explicit local SmolLM2-135M test checks runner
logits under the unchanged `tape_model_conformance` 0.5%-of-logit-range contract
and finite nonzero post-backward gradients at every emitted role. Absence of that
explicit fixture is a failure, not a passing conditional skip.

The independent compatibility probe extracts actual relevant definitions from
the original committed calibration source, retained intact beside it. It compares
27 old/new cases across layer/sequence counts: plain logits and all backward
buffers, nonidentity activation-transform logits, and legacy fold diagonals.
That differential source snapshot is an archived oracle, not a second maintained
forward implementation or a rewritten expected-result algorithm. The current
train attention mathematics are unchanged; complete training regression remains
a separate required check.

The unchanged-source check completed with exit 0 on 2026-10-10: three ordinary
tap tests, the explicitly executed original SmolLM2 fixture (30 layers, 212
roles), strict scoped Clippy/docs and all 27 differential cases passed. The
fixture's relative logit-range error was 6.840371498e-6 under the existing 5e-3
contract. These are software/fixture compatibility results, not a speed or
quality measurement. The first check's fixture-construction compile failure is
retained alongside the corrected terminal check, not overwritten. Full committed
training regression and canonical prepush checks remain required before push.

Exact commands, terminal success/failure and source hashes must be retained in
the project archive. Local software/fixture compatibility does not establish
language quality, performance, physical GPU/browser behavior, full memory,
independent release reproduction, or public activation.
