# Typed direct-level fitting — implementation evidence, 2026-10-10

Code: `0fa5d0bc788a7846cc4537adc9da653e59a102e8`.
Baseline: `c051a972a5cbbb6584aa313fb1d361b91f087bb4`.
Private candidate ADR0056 remains PROPOSED; no owner acceptance/profile freeze.

## Implementation

`tritium-schema::DirectFitPlan` defines a per-tensor request with a semantic
layout, explicitly named numerical profile, named plane counts and logical
element limits. Seven JSON projections are generated from the Rust types, not
handwritten. These are candidate requests, not frozen recipe identity preimages
or the complete model QuantPlan.

`tritium_quantize::engine::fit_direct_levels` produces shared owned validated
`AdditiveTensor` values. Each level fits the same original weights at its own
plane count. It neither slices a master nor refits another level's reconstruction.
The real shipping activation-aware ladder and real joint solver are reused;
there is no copied fitter or change to their numerical defaults/operation order.
The existing f64 Gram transform now has module-only visibility for reuse.
Legacy numerical goldens and resident CUDA kernels remain unchanged.

The ladder retains f32 working weights with f64 metric/feedback/search arithmetic,
the one-third tied F16 law, damping, decay/ramp and all three scale-refit placements.
The free F16/F32 joint profile is explicitly `JointIndependentGroups`, not the
complete cross-group-feedback/K-FAC SALT campaign solver. Weight and Gram basis
transforms agree; Hadamard block width is independent of scale group width.
Stored scales are rounded before constructing the owned tensor; overflow is an
error. Invalid input never silently falls back to identity/plain fitting.

Admission checks law/group/profile, geometry/options/names/plane counts, exact
input lengths, finiteness, Gram symmetry/nonnegative diagonals, shape overflow,
and source/metric/aggregate-output logical counts. The counts do not establish
complete solver temporary storage, process RSS, Rayon/device/KV/model admission.
SignedRht fitting, S34, per-tile allocation, entropy transport and new law/group
admission are explicitly rejected by this candidate, not silently emulated.

Returned levels receive no fabricated manifest `Direct` provenance or evidence
root. Source/calibration/plan/solver/output binding belongs to the full engine.

## Verification

Terminal verifier `3b651b4467314902bed8b70b9253ab0a` completed at 19:28:45 EDT,
with `MainPID=0`, `SubState=exited`, `Result=success`, `ExecMainStatus=0`, and the
clean code tree still pinned to the supplied full code identity. Exact commands
and logs are retained. Committed-tree lanes: schema/format/quantize all-targets
704 passed, zero failed, 15 declared ignores; unchanged original NN fitting and
feedback goldens four passed; strict schema/quantize docs; true bare-metal schema
alloc+serde; wasm32 unknown and WASI compilation. The eleven direct-level tests
are included in the 704, not added again. Tests include G32/G128 and ragged groups,
all ladder refit modes, exact anchors/trits against fresh production fits, a joint
source whose direct result differs from prefixes, distinct basis/group widths,
analytic two-axis curvature transformation, exact D2/B3 roundtrip, core matmul/
gather under the existing testkit tolerance, deterministic level ordering, source
immutability, invalid/indefinite metrics and basis/stored-scale overflow.

Canonical `sh scripts/verify-gates.sh prepush` passes formatting, Rust-generated
projection drift, true no-std foundations, default/all-feature/toolkit-free
workspace Clippy, Windows cross-target compilation and actionlint. ShellCheck
is unavailable and explicitly warned: workflow shell blocks were not locally
linted. Compilation is not physical browser/MCU/Windows/GPU qualification.

Initial test-interface compiler errors, an arbitrary new absolute consumer-test
threshold, and a duplicate Clippy flag were repaired and recorded. No original
goldens, numerical defaults, frozen gates or law registry entries were changed.
Fifteen model-dependent/manual profile tests remain ignored; no Qwen/GPT2 fixture
or quality/campaign test was silently promoted to a pass. See full logs for names.

## Remaining full scope

No complete P0–P7 phase or empirical release gate clears from this slice.

1. Full typed model QuantPlan/source/sink/evidence engine, precision-aware common
   feedback/pluggable fitter selection, unified calibration, direct-fit allocator;
   SALT/QTEA/ExTernD/TWLA/frontier GPU/out-of-core/refinement and thin entrypoints.
2. Accepted/frozen profile and all projections; HF full-content identity/caps/
   memory admission; `.trit` shards/per-tile/entropy/stream/mmap, exact migrations
   and gated legacy deletion.
3. Native CPU SIMD/GPU/all-backend conformance, resident Executor/Runner/
   ArchitectureSpec, hybrid DeltaNet/MTP, complete memory and quiet performance.
4. Stage7 authorized recipe/solver/refinement/auxiliary evidence; official Qwen
   language+MTP ternary/refined quality/task retention/runtime/physical bytes and
   independent reproduction.
5. Physical backend/browser/WASM/MCU/distributed-training qualification, including
   the Chrome/Firefox/Safari operation/lifecycle/fault/no-hidden-readback matrix.
6. Serving/plugin residency/cancellation/KV/concurrency/fault/latency, production
   deployment/security/observability and explicit human acceptance.
7. Same-source RC archives/wheels/ONNX/Colab/compatibility/docs/tutorials/community/
   audited zoo, second-machine/operator reproduction, security/signing clearance
   and explicit human authorization/public activation.

Native Rust frontend v1.2 and arbitrary trainable ONNX v1.3 remain distinct.
BitNet repack/quiet decode/per-kernel trace remain UNKNOWN. No campaigns, model
payloads, cloud jobs, older caches or foreign WIP were modified.

Custody destination:
`/home/brianklam/Projects/Tritium/archive/verification/direct-level-fitting-0fa5d0bc-20261010`.
Custody verifies retained bytes/original Git source, not model or release qualification.
