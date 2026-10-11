# Declared-statistics paired source stage

This implements the initial in-memory source-stage candidate in ADR0064, not
complete transformed-source package admission or a frozen research profile.
The parent is d1fb264e60ebbc7798be37f76c06ccdca3730e18.

`SourceTransform::run` consumes actual bytes through the existing `WeightSource`
interface. Its Rust-owned request declares a complete canonical tensor/asset
inventory, expected original identity, ordered named projection/partner pairs,
explicit inverse axes, raw f64 moment bindings/declared rows, alpha byte bits,
the legacy numerical policy, and logical payload/vector limits. It derives A
from the complete bytes actually consumed, rather than trusting the request.
It applies the existing checked `SaliencePair` in order and derives B from
actual output bytes using the existing original-source identity machinery.

Touched tensors are F32, including half-precision inputs when alpha is zero.
Untouched dtype/raw bits and all assets stay exact. F32 identity transforms may
have A == B; they still return an observation. The private-field returned source
is immutable through its interface and can be reused by the existing fitting
and calibration cache machinery. Old A Grams are not silently copied into B.
KeepDense during B->C conversion means exact preservation of B, not A.
Failure returns no derived candidate; the controller never writes source bytes.

Statistics remain explicitly `Declared`. Binding actual supplied moment bytes
and a declared row count is not proof of graph eligibility or executed model,
tokenizer, teacher, token selection, objective, or activation capture. The typed
observation is not a sealed log, publication manifest or qualification receipt.
No fake v1 quantization events or independent numerical/identity pipeline was
added. Existing events and ordinary C-only package semantics remain unchanged.

Logical limits cover original/derived tensor payloads, assets, actual moment
vectors and pair decoded/replacement/RMS/scale vectors. They exclude metadata,
caller allocations and complete process RSS. This retains a complete B in
memory; it is not source locking, durable spool, mmap or out-of-core admission.

## Scoped evidence

Terminal invocation2dd769ecdeaf4b4381cb54d394b2a42d passed:

- `cargo test --offline --locked -p tritium-quantize --test source_transform`:
  all11 focused cases passed, with a three-thread Rayon parent.
- Actual schema generator and `--check`: nine new Rust-derived JSON projections,
  without hand editing generated files.
- `cargo test --offline --locked -p tritium-quantize --all-targets`.
- Strict all-target Clippy for schema/quantize, strict rustdoc, formatting and
  diff/parent guards. Compilation and docs use `-D warnings`.

The focused cases exercise actual F16/BF16/F32 inputs and A/B identities,
untouched bits/assets, ordered up-column/up-row composition, alpha-zero
identity, invalid original/statistic/inventory/geometry/alias/budget inputs,
late overflow without caller mutation, strict JSON parsing, and both memory
and real seekable safetensors adapters. Actual plain/calibrated Engine calls
fit B, verify the produced package/log, preserve B norms, and reject A source
declarations or A-bound caches. Toy buffers are not executed real-model capture.

The scoped log SHA256 is
3e5b2033b888c4884cd39d5f3ffb04b4b31d819e6f972447efb5520b22b62cf3.
Two earlier scoped runs failed only in new cache fixtures: first nonzero damping
for the existing output-free profile, then an insufficient collector budget.
The fixtures now obey zero damping and derive the existing required numerical
budget plus retained cache payload. Failed controls/logs/source snapshots are
preserved; no production contract or rejection was weakened.

Exact committed combined regressions, actual old-fold byte comparison and
canonical gates are still required before ordinary publication. No full release,
real-Qwen quality, byte reduction, GPU, performance, SOTA, independent
reproduction or human activation clearance follows from these checks.

## Remaining migration and full release

Typed start/completion events, executed B-bound capture/replay, durable immutable
staging and a mandatory versioned A->B->C carrier/verifier/loader remain required.
Do not put lineage into optional ignored metadata or call B->C verification
complete transformed-source admission. Model-aware CLI/Qwen/Python/HF surfaces
and both historical campaigns still need differential migration verification.

The full v1.1 goal also retains frozen profiles/all projections, shared shipping
solvers/frontier methods/refinement, GPU/out-of-core/full RSS, native SIMD and
physical backends/resident hybrid/MTP inference, Stage7 and official Qwen quality/
runtime/physical bytes/reproduction, portable browser/MCU/distributed training,
production serving/deployment/security, identical-source release packaging,
audited model zoo/tutorials/community/signing, independent gates and human
activation. This source module does not narrow or complete that objective.
