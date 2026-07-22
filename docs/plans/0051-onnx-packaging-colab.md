# 0051 — Whole-model ONNX, release packages and Colab

Status: **IN PROGRESS** (2026-07-20)

- **Decision:** [ADR 0033](../adr/0033-v11-full-public-release.md)
- **Parent:** [plan 0044](./0044-v11-full-public-release.md)
- **Dependencies:** plan 0046 stable Torch operator schema; plan 0047 phased
  artifact/load API; plan 0049 frozen portable schemas
- **Publication:** local release candidates only until explicitly authorized

## Goal

Make Tritium v1.1 consumable without a source checkout or compiler. A supported
whole ternary language model must export to ONNX, reload, and generate through a
real ONNX Runtime session. The same candidate revision must produce auditable
crates, CPU/CUDA Python wheels and the plan-0050 npm archive, then prove a
five-minute no-compiler Colab workflow on the tutorial model.

Trainable ONNX is not part of this work order. It remains the binding v1.3
milestone and every ONNX-facing error/document must say so explicitly.

## Frozen interop contract

The v1.1 ONNX dialect is versioned and fail-closed:

- domain `com.tritium`, opset `1`;
- `TritiumTernaryMpGemm` for packed ternary linear projection;
- `TritiumTernaryEmbedding` for selected-row packed embedding lookup;
- canonical SALT V2 package/tensor identity attributes, never raw unchecked
  filesystem paths;
- standard ONNX operators for shape, residual, normalization, activation,
  attention/KV-cache and sampling glue where ORT has exact supported semantics;
- explicit graph metadata binding source model, tokenizer, recipe, Tritium
  build, artifact package IDs and in-scope/deferred coverage.

The exporter never materializes a dense quantized weight initializer. Packed
payloads use ONNX external data for large models and bind byte length plus
BLAKE3 digest. Relative paths are normalized, contained below the model
directory and symlink-free. Duplicate initializers, unknown attributes/opsets,
missing coverage, absolute/traversal paths and unauthenticated external data are
rejected before session creation.

## Slice 1 — whole-model graph and real ORT execution

Status: **IN PROGRESS** — packed mpGEMM and selected-row embedding custom ops,
deterministic tied embedding/head graphs, strict external-data verification and
real ORT execution exist. The additive schema-v2 API now fail-closed binds
source model, resolved tokenizer, conversion recipe, Tritium build, package,
converted coverage and deferred/preserved coverage identities through
verification receipts and canonical re-encoding. The schema-v1 API and wire
format remain source-compatible and readable; version-specific verifiers reject
cross-version interpretation. Both v1 inline and v2 external-data graphs execute
through real ORT sessions. Public pre-session inspection now returns deterministic
typed diagnostics for every unsupported node, attribute, dtype and unresolved
coverage item in the admitted tied-graph and decoder-core subset. Cache-aware
causal GQA now has a dependency-free semantic oracle plus an
experimental `com.tritium` opset-2 `TritiumKvAttention` proof; real ORT sessions
execute both prompt attention and one-token continuation over supplied K/V
cache. This does not alter frozen opset 1 or replace required standard-ONNX v1.1
attention glue. A second real-ORT proof now executes the same prompt and cached
decode path using only standard opset-21 `Transpose`, `MatMul`, `Mul`, `Add` and
`Softmax` nodes, an explicit additive causal mask and complete supplied K/V
cache. Its inspector contract rejects noncanonical opsets, attribute kinds,
rank-three permutations and softmax axes. Packed Q/K/V/O projection, cache
production/update, GQA expansion and complete decoder-block serialization now
join in a production `encode_causal_lm` graph: packed tied embedding/head and
Q/K/V/O/SwiGLU projections, preserved RMSNorm vectors, residuals, causal mask,
per-layer cache concatenation and present-cache outputs. A nondegenerate tiny
2:1 GQA model executes prompt plus cached continuation through real ORT and
matches an independent dense reference for logits, greedy tokens and every K/V
element. Deterministic encoding and the pre-session inspector are part of the
same gate. Cache/GQA and FFN emission now live behind focused builder seams.
Optional per-head Q/K RMSNorm plus full-head Qwen-style RoPE use absolute prompt
and decode positions; rotated K is what enters and leaves each cache. The same
4Q/2KV, head-dim-4 ORT gate uses nonuniform Q/K norm weights, nonzero frequency
lanes and positions 0, 1 and 2, so norm/rotation/cache-order errors change the
independent logits and cache oracle. One f64-generated cos/sin table and one
slice-constant set are shared across every layer/Q/K application; extreme valid
theta/position regression requires every serialized table value to remain finite.
The causal mask is shared across layers, and aggregate initializer plus exact
protobuf bounds reject oversized inline graphs before publication.
Complete causal graphs now have an additive authenticated external-data API:
64-byte-aligned packed/value ranges bind exact length and BLAKE3, strict
verification rejects corrupt data/ranges before session creation, and real ORT
executes file-backed prompt inference against the independent logits oracle.
Verification requires model and weights BLAKE3 trust roots from the admitted
package manifest; candidate files cannot nominate their own expected digests,
so graph rewiring and internally rehashed payload mutations fail closed.
Shape-driving `int64` constants remain inline because ORT shape inference needs
their values while loading. External export now emits packed bytes and f32
initializers directly into final aligned storage without first forming an inline
protobuf or cloning packed matrices. A public regression crosses the 64 MiB
inline limit while inline export rejects the same model; direct f32 emission
also avoids a second serialized-value buffer. Generated causal masks and RoPE
tables stream into final storage with checked allocation; impossible geometry
returns a typed error before allocation.
The first architecture tensor-map adapter now maps canonical SmolLM2
Hugging Face names into the tied-head, bias-free, full-RoPE SwiGLU graph and
rejects missing names or config/tensor geometry drift. The causal graph now
also represents BitNet's ReLU2 gate activation plus attention-output and
FFN-intermediate subnorms, with real ORT/reference parity. An exact BitNet GGUF
namespace adapter maps ternarized packed embeddings/projections and preserved
norms, rejecting missing/duplicate/extra tensors including an untied
`output.weight`. Raw dense-embedding/I2_S conversion remains an upstream
quantization/import concern. Qwen3.6 still requires its heterogeneous
DeltaNet/full-attention schedule, exact architecture tensor-map adapter and MTP
composition. Conventional causal graphs now support an untied packed LM head,
prefix-only RoPE, model-wide zero-centered RMSNorm weights and sigmoid
attention-output gating, each with independent-reference ORT parity. The
zero-centered graph retains source offsets and emits the effective `1 + weight`
scale explicitly, with artifact metadata distinguishing it from ordinary
RMSNorm. Qwen's native fused query projection is consumed without repacking:
head-interleaved query/gate rows are split in graph before Q normalization and
sigmoid output gating, with independent-reference ORT parity and a typed
projection contract that makes contradictory separate gate weights
unrepresentable. Residual and query widths are independent, covering Qwen's
5120-wide stream and 24x256 query geometry. Dynamic axes and end-user generation
APIs remain open.

The projected Qwen Gated DeltaNet recurrent core is now a registered opset-2
custom operator. Packed QKV/Z/beta/decay/output projections remain composable
Tritium mpGEMMs; the core consumes explicit convolution and recurrent state and
publishes both next states. Frozen independent numeric vectors cover prompt and
cached-token transitions through real ORT, including depthwise history order,
decay-before-delta, normalized state feedback and gated RMSNorm. Whole-layer
emission now composes packed QKV/Z/beta/decay/output and SwiGLU projections,
zero-centered pre-mixer/pre-FFN norms, both residuals and explicit prior/next
state through real ORT. Heterogeneous language-model schedule composition,
including sparse layer-indexed DeltaNet and KV-cache state, now runs prompt and
cached decode through one packed causal graph. Its public full-attention type
structurally requires Q/K norm, fused head-interleaved query/gate and SwiGLU.
Inline admission dry-runs the shared emission path and aggregates every
initializer payload before cloning. An exact Qwen3.5/Qwen3.6 adapter now admits
only the canonical mixed-schedule language namespace plus all 15 bundled MTP
tensors, validates packed geometry and produces the encodable language graph.
Its flagship entry point additionally requires the pinned 64-layer
Qwen3.6-27B geometry and exact three-DeltaNet/one-attention cadence. Bundled MTP
weights now emit a separate caller-aligned graph: shifted shared-token embedding,
target-hidden fusion, forced full-attention/SwiGLU decoder, private KV cache,
final hidden rows and untied drafter logits execute for prompt and cached decode
through real ORT. A nondegenerate dense oracle uses both fusion halves, nonzero
Q/K/V/O and SwiGLU projections, and cache-sensitive continuation; logits, final
hidden rows and every prompt/decode K/V value must match. External-data emission
now produces `language.onnx`, `mtp.onnx` and one authenticated `weights.bin`.
Shared embedding/head ranges are stored once and referenced by both graphs;
strict union-range admission rejects gaps, partial/unauthorized overlaps,
identity/geometry/RMS-epsilon drift, noncanonical attention cadence and any of
three manifest-digest mismatches. Real ORT loads both file-backed graphs from
the shared arena and matches inline execution. The ONNX domain now also has a
real opset-2 `TritiumSaltV2MpGemm` and `TritiumSaltV2Embedding` substrate over
the production indexed SALT V2 arenas: D2, B3 and S34 payloads, f16 group-128
scales, adaptive one-to-three plane allocation maps and rank prefixes execute
without a dense shadow. Their dependency-free kernels match independent
additive oracles; malformed arena lengths, noncanonical map padding/ranks,
codec data, scales, activations and token indices fail closed; registered
custom-op graphs pass pre-session inspection and execute in actual ORT
sessions. Whole-Qwen emission is generic over the physical matrix storage and
now serializes SALT V2 embedding, every language projection and every MTP
projection directly. The authenticated three-file bundle aliases all eight
shared embedding/head arenas, rejects partial or mixed storage layouts, and
matches inline language and MTP execution in real ORT. `HostSaltV2Linear`
exposes borrowed arena views so a package mapper can serialize real
PTQ/refined operands without reconstruction. The Python/Hugging Face facade
and package-to-Qwen view construction are now the binding Slice 2 work. The
Qwen mapper now accepts a physical-layout-parameterized
`Qwen35PackedTensorProvider`, preserving either legacy TQ or additive SALT V2
operands through mapping. `Qwen35SaltV2PackageSource` now consumes
independently authorized regular file handles, rechecks both admitted transport
identities, materializes descriptor-free packed arenas plus exact BF16
preserved vectors from one bounded authenticated snapshot, and rejects exact
name/rank/shape or aggregate physical-ledger drift before weight
materialization. Schema-v3 manifest/path admission and the Python facade remain
the active Slice 2 boundary.

Upgrade `tritium-onnx` from a single reference custom op to a versioned operator
domain plus whole-model loader:

- keep the dependency-free bit-exact kernel as the semantic oracle;
- execute packed SALT projections/embeddings directly from verified package
  state without a persistent dense shadow;
- support the decoder-only blocks, tied embedding/head, KV cache and preserved
  tensors required by the tutorial model, BitNet 2B4T and the in-scope Qwen3.6
  language-plus-MTP artifact;
- expose deterministic token-ID greedy generation for parity tests;
- return typed unsupported-graph diagnostics naming every rejected node,
  attribute, dtype and coverage item.

Gate: export a tiny tied-weight causal LM, open it with an actual ORT `Session`,
run prompt plus cached decode, and match native Tritium logits/tokens within a
frozen tolerance. The test must prove that ORT executed the registered custom
domain. An in-memory kernel call or ignored session test is not sufficient.

## Slice 2 — PyTorch/Hugging Face ONNX facade

Status: **IN PROGRESS** — generic hard-PTQ `AdditiveTernaryLinear` graphs now
export through the public Torch ONNX exporter with B3 payloads and f16 scales
retained as output-reachable initializers. Export disables optimizer folding,
audits the graph for a persistent dense target-weight shadow, executes numerical
parity in a real CPU ONNX Runtime session and only then atomically publishes a
canonical SHA-256-ledgered bundle. Reload verifies the complete file allowlist,
external-data paths, graph interface and packed initializer geometry before
session creation; dynamic batch replay and corrupt-graph rollback are gated.
The same fallback now accepts flat Hugging Face `ModelOutput` values, preserves
shared packed embedding/head storage and gates an alternate sequence length in
a real tiny-Llama PTQ export; caller-declared symbolic input axes remain part of
the graph identity. Batch dynamism is independently gated on the generic linear
path because the tested upstream tiny-Llama export retains a batch-one
`GatherND` specialization.
The stable `export_onnx`/`load_onnx` facade now routes complete Qwen PTQ bundles,
generic module PTQ results, QAT-hard results or reopened QAT-hard artifacts, and
refined children without changing their claim type. Generic exports require an
explicit model shell where architecture is not owned by the result plus explicit
example inputs. Their canonical schema-v2 manifest binds conversion mode,
source-model digest, recipe, exact input artifact, and complete refinement
ancestry; strict reload cross-checks lineage against graph checkpoint identity
before ORT session creation. Schema-v1 module bundles remain readable, while new
typed exports write schema v2. Scale-only and hard-PV children retain those exact
discriminants rather than collapsing to a generic refinement label. Dynamic
batch/sequence names and parity tolerances flow through the same public facade
instead of a separate untracked exporter.
Latent QAT modules, prepared graphs, optimizer/checkpoint mappings, and
checkpoint directories are rejected with `trainable_onnx_requires_v1_3` before
an exporter or runtime is opened.
An isolated environment pinned to ONNX 1.22.0, ONNX Runtime 1.27.0 and
ONNXScript 0.7.1 now executes QAT-hard results, strictly reopened QAT-hard
artifacts, module-PTQ, scale-only refinement and hard-PV refinement through
public export, strict public reload and real ORT parity; the five focused
packed-module ORT gates pass together. This remains tiny-module evidence, not a
whole-model causal-generation qualification.
This is a semantic generic-module fallback, not evidence of native fused custom
operator execution, causal generation closure or Hugging Face whole-model ONNX
support. Those remain binding below.

Release admission now fail-closes through
`tritium.onnx-inference-qualification.v1`. It binds the exact installed wheel,
candidate ONNX archive and conversion-parent model bundle; requires a source- and
compiler-free environment, physical CPU ORT provider, authenticated schema-v2
language/MTP external data, executed `com.tritium` opsets 1/2, zero dense weight
initializers/shadows, prompt/cache/generation/MTP parity and a content-bound
execution trace. Graph/weight corruption, path traversal, unknown operators and
trainable export/import must all fail. Validator presence is structural; no
candidate Qwen3.6 whole-model receipt exists.

Add `tritium.torch.export_onnx` and `tritium.torch.load_onnx` over the stable
plan-0047 results:

- accept QAT hard export, PTQ and refined artifacts as distinct input types;
- preserve their exact recipe and ancestry discriminants in graph metadata;
- stage model plus external data, validate with the strict native reader, open
  an ORT session, run parity, then atomically publish the directory;
- support ordinary Hugging Face generation inputs/outputs for declared causal
  LMs without claiming `AutoModel` training support;
- reject latent masters, optimizer/checkpoint state and training graph import
  with `trainable_onnx_requires_v1_3`.

Round-trip tests cover tied weights, optional bias/QK norm, dynamic batch and
sequence axes, KV-cache continuation, corrupt external data and rollback on a
failed final parity run.

## Slice 3 — version and package closure

Status: **IN PROGRESS** — CPU abi3 wheels retain the manylinux/macOS/Windows
matrix. A separate Linux CUDA 13 wheel compiles the `tritium-py/cuda` feature,
declares its compiled backend inventory, and executes the native Rust CUDA
mpGEMM plus the installed-wheel Torch/Hugging Face lifecycle on the qualified
sm_89 runner. CUDA builds now run inside an immutable official
`manylinux_2_28_x86_64` image with exact Rust, maturin and C++ toolchain
contracts; the host CUDA 13 toolkit is mounted read-only, maturin/auditwheel
enforce policy, and the verifier rejects any other platform tag. A local
container build produced and admitted the expected tag. Candidate-revision
CI/runtime receipts remain required before compatibility is claimed. Local-RC
admission now has a shipped streaming SHA-256/BLAKE3 identity primitive and a
strict `scripts/release-status` gate. It rejects noncanonical versions, dirty or
wrong source revisions, uncontained/symlinked/duplicate artifacts, byte or
digest drift, unbound CycloneDX/SPDX documents, and provenance that does not
bind the exact artifact SHA-256, source revision, and builder identity. Actual
candidate artifacts and cross-platform receipts remain open. A deterministic
assembler now sorts artifact inputs, generates canonical in-toto/SLSA v1
statements, fsyncs metadata, rolls back failed publication, and strict-reloads
the completed allowlisted candidate. `CANDIDATE_EVIDENCE_VALID` deliberately
does not claim `LOCAL_RC_READY`; signing and aggregate release gates remain
open. Operator workflow is documented in
[`docs/release-candidate.md`](../release-candidate.md).
CPU and CUDA wheel lanes now generate deterministic CycloneDX 1.6 inventories
from each already-verified wheel. SBOM root binds exact release artifact ID,
wheel SHA-256/bytes/platform, every RECORD-covered member digest and every
declared `Requires-Dist`; CI uploads SBOM beside same wheel.

Installed-wheel CPU/CUDA differentiable lifecycle runs now publish strict
content-addressed qualification receipts binding source revision, candidate
version, independent run/machine identity, exact wheel bytes, environment,
native device/backend and frozen QAT/checkpoint/reload coverage. Release-registry
admission rehashes candidate wheel and receipt; this closes one clean-install
evidence kind but does not replace full interpreter/platform matrix admission.

ABI3 aggregation now emits dedicated `tritium.abi3-matrix-qualification.v1`
evidence bound to release, source revision, workflow run, every cell-evidence
digest and exact wheel name/SHA-256/bytes. Strict reload revalidates complete
CPython 3.9+ Linux/Windows and available macOS arm64 matrix, target/platform
contracts and single-wheel reuse. Release registry matches all three wheel
identities to candidate manifest; local archive admission remains separate.

Publish-readiness now follows crate assembly/SBOM generation with exact archive
qualification. Harness rejects missing/stale archives, revalidates safe package
and clean VCS metadata, extracts all current-version crates into an isolated
consumer, patches internal registry dependencies to those exact sources, and
stages external dependencies from exact workspace lock with `cargo vendor
--locked`. Locked all-target check then runs with network disabled and empty
`CARGO_HOME`. Content-addressed receipt binds source/release/run/machine and
toolchain, Cargo.lock digest, every archive identity and compiled package set.
Release registry requires exact equality with candidate `rust-crate` inventory.
The exact npm archive now emits a release/source/run/machine-bound qualification
receipt after a source-free offline install, 114-vector WASM conformance and
strict TypeScript consumer check. Independent admission rehashes SHA-256 and npm
SHA-512 integrity, requires the frozen 13-file inventory and clean WASM Git
identity, and matches exact bytes to the candidate `npm-archive` artifact.
This landed in `cb34670` with bounded streaming artifact validation in
`57a23ba`. A clean detached run at `cb34670` produced and re-admitted receipt
`sha256:87d018deda50ba199a15b1ce523ddeff87546e32ccbeea142a3945f6a3785bcf`;
this closes only the npm evidence kind, not the aggregate package or public
activation gates.
The exact npm archive verifier now emits its own deterministic CycloneDX 1.6
document and content-addressed qualification receipt into retained evidence. It
binds archive SHA-256/bytes and SHA-512 integrity, package/version, clean source
revision, run/machine/toolchain identity, WASM build and guest digests, every
lockfile component with strong
integrity, license and platform metadata, plus exact runtime dependency edges;
development/build dependencies remain visible with excluded scope. The release
registry admits this schema and matches it to the candidate npm archive.
Publish-readiness CI now retains every exact `.crate` beside a bound
cargo-cyclonedx inventory. Admission checks archive name/size, complete safe tar
topology, `Cargo.toml.orig`, clean exact `.cargo_vcs_info.json`, package/version,
archive SHA-256 and source revision. Local `file://`/absolute-path component IDs,
PURLs and dependency edges are canonically rewritten before upload. OCI, chart,
ONNX and model-artifact SBOM lanes remain open.

Set one candidate version source for Rust crates, Python metadata, npm metadata,
CLI output, schemas, docs and user agent. `1.1.0-rc.N` archives advance during
local qualification; only the accepted immutable revision becomes `1.1.0`.

- Run `cargo package --locked` for every publishable crate in dependency order.
- Install each `.crate` from a clean offline cargo home after dependencies are
  staged; verify features and license/notice contents.
- Build abi3 CPU wheels for supported Linux/macOS/Windows targets and separate
  CUDA wheels for the declared Linux/CUDA/PyTorch matrix.
- Install wheels into clean Python environments with no compiler/repository and
  run import, PTQ/QAT backward, export/reload, native generation and ONNX
  generation smoke.
- Build and install the exact plan-0050 npm archive in a clean strict-TypeScript
  project; source-tree imports are forbidden.
- Generate SHA-256/BLAKE3 manifests, CycloneDX/SPDX SBOMs, license reports and
  SLSA-style provenance bound to source revision and workflow identity.
- Sign local RC manifests/artifacts with an ephemeral test identity. Release
  signatures use the documented maintainer identity only after authorization.

`scripts/check-semver.sh`, API diff and dependency policy gates must pass.
Unpublished internal crates are either packaged in the ordered set or removed
from public dependency graphs; `--allow-dirty`, path leakage and network fetch
during smoke are failures.

## Slice 4 — compatibility matrix and failure policy

Generate, do not hand-edit, a compatibility table covering:

- OS/architecture and CPU ISA fallback;
- Python and supported PyTorch versions;
- CUDA runtime/driver/GPU architecture and separate ROCm/Metal capability
  labels where artifacts exist;
- ONNX Runtime version/opset;
- Node/browser versions for `@tritium-ai/web`;
- model/artifact schema versions and backward-read policy.

Each green cell points to an exact receipt. Unsupported combinations fail with
an actionable diagnostic rather than attempting a source build or CPU fallback.
Old readable artifacts remain tested; new writes always use the current schema.

## Slice 5 — five-minute Colab proof

Status: **IN PROGRESS** — generated notebook, deterministic source checker and
CUDA-wheel headless execution lane exist. A local exact-wheel run on RTX 4090
completed pinned SmolLM2-135M PTQ, compact HF save/reload, dynamic-sequence real
ORT export/replay, token generation, full-model CUDA QAT step and optimizer
checkpoint/resume in **104.66 seconds excluding first download**. It measured
537,919,488 selected dense bytes versus 82,056,969 compact checkpoint bytes
(6.56x) and 31.16% zero trits. This is development evidence; candidate-revision
CI/Colab receipts remain required before public support is claimed.

Publish a notebook source plus a non-interactive execution harness using the
pinned SmolLM2-135M tutorial artifact. From a fresh supported Colab runtime it
must, after one wheel install and excluding first model download:

1. load and inspect exact coverage/physical bytes;
2. run a bounded PTQ calibration/conversion example;
3. run one QAT forward/backward/optimizer step;
4. checkpoint/resume;
5. export/reload native and ONNX artifacts;
6. generate tokens and display ternary distribution/zero-rate diagnostics.

The measured wall time is below five minutes on declared hardware. The notebook
contains no editable absolute paths, hidden Drive dependency, registry token or
preinstalled source checkout. CI executes the notebook headlessly against local
RC artifacts before any public link is advertised.

## Slice 6 — authorized activation and immutable smoke

Only after Brian Lam explicitly authorizes publication:

1. publish crates in frozen dependency order;
2. publish exact CPU/CUDA wheel files through trusted PyPI publishing;
3. publish the exact npm archive;
4. fetch every artifact from its registry into fresh environments;
5. rerun package, native-generation, ONNX and Colab smoke;
6. archive registry URLs, immutable digests and receipts.

A post-publication failure never replaces `1.1.0`; it opens a corrective
`1.1.1`. No registry ownership, token or namespace action is inferred from this
plan.

## Verification cadence

```bash
cargo test -p tritium-onnx --features onnx
cargo clippy -p tritium-onnx --features onnx --all-targets -- -D warnings
PYTHONPATH=crates/tritium-py/python pytest -q crates/tritium-py/tests
npm --prefix packages/tritium-web ci
npm --prefix packages/tritium-web run check
./scripts/check-semver.sh
git diff --check
```

Local-RC qualification additionally runs the full offline fresh-environment
matrix and verifies every archive/SBOM/provenance digest.

## Stop conditions

- Stop before adding an ONNX training claim, dense packed-weight shadow or
  unverified external-data path.
- Stop before weakening native/ORT parity, clean-install, offline-smoke,
  compatibility or five-minute gates.
- Stop before any registry publication, namespace reservation, paid Colab/GPU
  run or signing-key use without explicit authorization.

## Done criterion

A supported whole model exports and generates through real ORT with native
parity; every local crate/wheel/npm RC installs without source or compiler;
version, compatibility, SBOM, signatures and provenance agree; the headless
Colab workflow passes in under five minutes. Authorized registry artifacts, if
activated, pass a separate immutable post-publication smoke.

## Commit sequence

```text
docs(plan-0051): freeze ONNX and packaging work order
feat(onnx): execute packed whole-model graphs
feat(torch): export and reload ternary ONNX models
build(release): produce auditable local RC archives
test(release): admit compatibility and clean-install matrix
docs(colab): prove five-minute ternary workflow
```
