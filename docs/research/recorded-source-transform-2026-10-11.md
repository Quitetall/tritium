# Recorded declared-statistics source transformation

ADR0064's next software slice adds a real standalone transform log and actual
byte/numerical replay to the existing in-memory A->B controller. It does not
publish a dense stage, claim executed activation capture or admit an A->B->C
model package. Parent: 201a0ab9fa53e0b434ff117f95f9c1e7b67ac93c.

## Shared interfaces

`SourceTransform::run_recorded` consumes the existing `WeightSource` and actual
declared moment vectors, performs the existing checked transform, and returns
immutable B plus sealed evidence. Its recorder must register both distinct v1
events: `tritium.quantize.salience_transform_started` and
`tritium.quantize.salience_transform_completed`. The root span `transform`
contains the request at sequence 0 and actual complete observation at sequence
1. No v1 quantization event is fabricated. The existing evidence envelope,
canonical serialization, hash chain and run-root implementation are reused.

`SalienceTransformView::fold` rechecks integrity, strict payloads/versions,
lifecycle/ancestry, unchanged request, complete ordered inventories, pair output
interpretations and finite positive scale bits. It independently derives the
two recorded content identities through the existing identity module. An
unfinished start produces no completed view; orphan/duplicate completions
reject. The shared schema observation validator checks structure, not arithmetic.

`SourceTransform::replay` requires exactly one standalone completed transform,
executes the existing controller against actual A and supplied moment bytes,
and compares the entire recomputed observation with the recorded one. A log
with a valid new hash chain and internally consistent false output identities
or finite scale bits may pass structural folding but must fail actual replay.
These are intentionally different verification levels.

The actual CLI `evidence verify` and `evidence view` commands invoke the shared
typed view for these known schemas. They reject malformed known payloads even
with valid hash chains, but remain integrity/structure checks: they do not read
source weights or perform numerical replay.

## Scoped checks and retained failures

Invocation 2be024f85a594370a3f886d3193730ee is terminal exit 0:

- All 17 source-transform interface tests, including actual recording/replay,
  valid-chain false-scale/output rejection during replay, malformed inventories,
  lifecycle/version/unknown-field rejection, and source/statistic changes.
- Full evidence and quantize all-target tests, plus all 12 CLI evidence-module
  tests. The CLI fixtures call the production recorder/controller and both
  actual command handlers, rather than reproducing their algorithms.
- Rust-owned projection generation/drift check; strict schema/evidence/quantize/
  CLI all-target Clippy; strict schema/evidence/quantize docs; fmt and source guards.

The scoped log SHA256 is
251891ef63cef249c7321f494c1818a7cb24137e6f7387f75d785177d4c369a.
There were 457 test executions, including the focused suite repeated in the
all-target sweep, and 10 declared real-model ignores. They are not 457 unique
cases or real-model qualification. Earlier invocation 9fd480bf21a84ec7b6604f4c29e09209
failed because two new CLI fixtures omitted required config/tokenizer assets;
the fixtures were corrected without changing production admission. Its source,
control and complete failure log are retained in the durable verification archive.

The parent source-stage exact checks also retained two failures: transient CLI
child deadlines (unchanged default-parallel controls subsequently passed), then
three generated-directory tests' incorrect executable-path assumption under
split Cargo build directories. The latter is repaired separately in parent
201a0ab9 using Cargo's binary path; no test deadline or equality gate was relaxed.
Exact committed combined and canonical gates are still required before publication.

## Remaining obligations

The statistics class stays `Declared`; raw moment binding and replay do not
prove graph eligibility, tokenizer/teacher/token execution or activation capture.
Logical payload/vector limits are not full RSS, source locking or out-of-core.
The complete B remains in memory. Neither log is a release qualification receipt.

Durable immutable model staging, the mandatory versioned A->B->C manifest/log
carrier, shared loader/admission verifier, executed B-bound capture/replay and
CLI/Qwen/Python/HF plus both historical-campaign migrations remain required.
Legacy event bytes and C-only artifact semantics remain unchanged.

The full ADR0033 v1.1 goal still includes frozen profiles/projections, shared
shipping solvers/frontier methods/refinement, GPU/out-of-core/full memory, native
SIMD and physical backends, resident hybrid/MTP runtime, official Qwen language
and MTP quality/runtime/physical-byte/reproduction gates, browser/WASM/MCU and
distributed training, production serving/security/deployment, source-identical
release packaging, audited zoo/tutorials/community/signing, independent checks
and explicit human public activation. This software slice does not clear them.
