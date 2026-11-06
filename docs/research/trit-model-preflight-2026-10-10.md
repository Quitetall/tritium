# Shared model metadata preflight

Implementation: `a51ffc0426f0a2d5c4a7d736e007fc691607e0d1`, based on
`17b3569939bf14b945740b696e156aa55b6b7dd5`. Implements the experimental
ADR 0054 candidate under accepted ADR 0044 D10/D12. ADR 0054 remains PROPOSED.

## Implemented scope

One private `StandardModelPlan` defines standard-transformer tensor roles,
names and shapes. Both metadata preflight and the existing numerical assembly
consume it; there is no second model schema or fake-model/backend dry run.
The legacy HF/GGUF/SALT builder delegates to that same description. Its
numerical order, operators and existing test tolerances are unchanged.

The shared admitted source path now prepares config, optional architecture
features, exact complete inventory, embedding vocabulary and tensor shapes/
storage before any backend upload. Selected typed levels reject missing/extra
names, incompatible vectors/matrices and manifest/config disagreement. Admitted
HF sources now also reject extra tensors and incompatible floating-point storage.
The legacy dense-master `load_hf` remains a separate compatibility interface;
it is not silently converted into an exact-inventory consumer.

Prepared loading retains parsed config/spec, its actual description and selected
typed source or already-open HF shard readers. Assembly does not reopen config
or select a new inventory. This is not a frozen HF snapshot: in-place file
mutation and unavailable HF content identity remain unresolved.

The text loader parses real tokenizer assets and checks every tokenizer ID
against admitted embedding rows before matrix upload, rather than uploading the
model and rejecting it afterwards. Existing asset limits remain unchanged.

`EvidenceRecorder::check_emit<E>(span, parent, time)` factors the same
nonmutating registration/version/span/parent/time/current-sequence checks that
actual `emit` repeats. It serializes no payload, reserves no sequence, establishes
no ancestry and cannot guarantee a later emission. The actual emission still
validates its payload and records the unchanged envelope/hash chain.
Text loading checks both ModelLoaded and TensorUploaded contexts before parsing
the source; weights-only admitted loading checks TensorUploaded. Missing
registration is now a cheap rejection, not a deliberately late upload failure.

## Checkpoint evidence

Exact-tree verifier invocation: `66e78afdecc347b7867b616f9f8b3272`.
Terminal software PASS: MainPID 0, exited, Result success, ExecMainStatus 0.
The pinned checkout was clean before and after execution. Exact results:

- Foundation crates: 553 passed, zero failed, six declared ignores.
- Default NN: 166 unit and nine integration tests passed.
- Tokenizer NN: 167 unit and thirteen integration tests passed.
- Strict CUDA/tokenizer integration: fifteen passed with
  `TRITIUM_CHECK_ONLY=0 TRITIUM_REQUIRE_CUDA=1`; not all fifteen are GPU-only.
- CLI: sixty passed, one external-BitNet-fixture test ignored.
- Scoped real CUDA/tokenizer Clippy and warning-denied default/tokenizer rustdoc
  passed. Actual bare-metal schema alloc+serde check passed.
- The separate Python/cbor2 semantic preimage calculator still agrees on the
  four unchanged candidate vectors.
- `sh scripts/verify-gates.sh prepush` passed (format/projection drift,
  bare-metal foundation, default/check-only all-feature workspace Clippy,
  Windows GNU compile and actionlint). ShellCheck was missing/warned, not passed.

Device: RTX 4090, GPU-1790118a-a6d7-4eaf-fcac-dcacac5f4351, driver 615.71.09.
These synthetic functional CUDA checks are not quiet performance runs. Additive
CUDA remains explicitly emulated, not native packed resident execution.
The exact commands and early failures are retained in the custody archive at
`~/Projects/Tritium/archive/verification/model-preflight-a51ffc04-20261010`.
All five hosted workflows on previous source `17b35699` passed. Hosted results
on this new source and independent empirical clearance are separate obligations.

Focused software checks pass: thirteen tokenizer-enabled loader integration
tests, seven load-evidence tests and warning-denied scoped tokenizer Clippy.
An instrumented CPU adapter counts actual upload calls on invalid inventories,
tokenizer IDs and missing registration; an empty evidence log alone is not the
assertion. Thirty HF rejection scenarios span Llama/Qwen2/Qwen3, tied/untied
heads and missing/extra/wrong-matrix/wrong-vector/wrong-dtype inputs. Typed
package and tokenizer rejection controls also count zero uploads. Existing
CPU/reference/dense/additive codec/basis/cached decode/text fixtures remain.
Legacy assembly also consumes the new description; path-to-path comparisons are
regression checks, not an independent architecture or actual-model oracle.
Evidence tests prove repeated prechecks do not record events or advance sequence,
and changed parent, unsafe/backwards time and unregistered events reject.

Initial tests demonstrated the old missing-tensor and out-of-range-tokenizer
uploads. Intermediate changed expectation and EmptyRun test failures are retained
in the custody archive; no numerical tolerance or frozen gate was weakened.

## Explicit limits

Metadata preflight is not a full payload scan, HF content identity, aggregate
capability/budget admission, memory measurement, residency or a faster decoder.
HF finiteness/read failures still occur while seek-reading payloads. Capability,
per-tensor policy and budget checks remain on actual upload; later failure drops
all unpublished payload leases and preserves earlier genuine upload observations.
The weight ledger still excludes parsed tokenizer, source/temporary data, dense
norm/bias vectors, scratch, KV, allocator and driver overhead. The private
description itself is metadata, not total-memory accounting.

No event schema, candidate identity preimage, artifact, scale/trit bits, plane
admission, CUDA resident kernel, numerical default, recipe or release gate changed.
These are synthetic-model functional tests, not actual Qwen quality, resident
native packed GPU performance, model memory/physical byte or release qualification.
The BitNet fixture is still unavailable: quiet five-run decode/trace and BitNet
repack qualification remain UNKNOWN. ShellCheck is unavailable, not passed.

## Full remaining goal

1. Owner-accepted/frozen typed profile and all normative/generated projections;
   HF content identity/full admission; shard/per-tile/transport/stream/mmap support,
   exact committed-fixture/Qwen migration and legacy deletion.
2. Unified Engine/QuantPlan/calibration/direct-fit allocator and SALT/QTEA/ExTernD/
   TWLA/frontier GPU/out-of-core refinement; thin PTQ/QAT/Python/HF/CLI/ONNX/FFI/
   Burn/Candle/browser/MCU surfaces and bounded/tracing evidence.
3. Native CPU SIMD/GPU/all-backend execution, resident Executor/Runner/
   ArchitectureSpec/hybrid DeltaNet/MTP, total-memory and quiet performance gates.
4. Stage-7 recipe/solver/refinement/auxiliary evidence and official Qwen language+
   MTP ternary/refined quality/task-retention/runtime/physical-byte/reproduction.
5. Physical backend/browser lifecycle/fault matrix and distributed training.
6. Serving cancellation/residency/KV/concurrency/plugins/deployment/security/
   observability and explicit human acceptance.
7. Fresh same-source RC/packages/ONNX/Colab/docs/tutorials/audited model zoo/
   community; second-machine/operator independent reproduction/security/signing
   clearance and explicit human activation/publication.

No full phase or release goal is complete. Native standalone Rust frontend v1.2
and arbitrary trainable ONNX v1.3 remain distinct from v1.1 scope.
