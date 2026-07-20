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
coverage item in the current tied-graph subset. Decoder blocks, integrated
cache lifecycle, decoder-wide diagnostics and whole-model generation remain
open. Cache-aware causal GQA now has a dependency-free semantic oracle plus an
experimental `com.tritium` opset-2 `TritiumKvAttention` proof; real ORT sessions
execute both prompt attention and one-token continuation over supplied K/V
cache. This does not alter frozen opset 1 or replace required standard-ONNX v1.1
attention glue. Packed Q/K/V/O projection, cache production/update and complete
decoder-block serialization remain open.

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
