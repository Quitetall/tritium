# 0053 — Audited model zoo, community and v1.1 release

Status: **READY** (2026-07-20; work order frozen, implementation open)

- **Decision:** [ADR 0033](../adr/0033-v11-full-public-release.md)
- **Parent:** [plan 0044](./0044-v11-full-public-release.md)
- **Entry dependencies:** plan 0043 and plans 0045–0052 at their local-RC gates
- **Release candidate:** `1.1.0-rc.N`; final version `1.1.0`
- **External actions:** no registry publication, public tag/push, hosted model
  upload, Discord/community creation or public deployment without explicit
  authorization

## Goal

Close Tritium v1.1 with evidence rather than launch prose. The exact local
release candidate must ship a three-tier audited model zoo, current guides and
API documentation, contribution/governance/security/support policy, generated
benchmark and compatibility claims, and a second-machine reproduction. The
release revision is accepted only when every ADR 0033 checkbox is backed by a
machine-readable receipt and all source/package/image/artifact identities agree.

This plan does not lower an upstream gate. Missing flagship quality, physical
browser/backend evidence, whole-model ONNX, package, serving or reproduction
evidence remains a release blocker even when all editorial work is complete.

## Frozen release vocabulary

- **Structural:** schema, source, geometry, compile or emulation evidence. It
  never authorizes a hardware, quality, performance or complete-model claim.
- **Local RC:** exact unpublished crates, wheels, npm archive, OCI images,
  charts, model artifacts, docs and receipts produced from one clean revision.
- **Admitted evidence:** a strict receipt whose schema, source revision,
  artifact identities, command, environment and required measurements validate
  against the candidate release manifest.
- **Reproduced:** regenerated on a declared second physical machine from the
  candidate instructions and immutable inputs, not copied from the primary
  evidence directory.
- **Public:** reachable from an authorized immutable registry/repository URL and
  verified by post-publication smoke. Local RC success is not public success.
- **Tritium v1.1:** the complete product gate. Individual backend/model/package
  success must not be shortened to “v1.1 complete.”

## Slice 1 — evidence registry and generated claims

Add a versioned release-evidence registry and validator:

- one candidate manifest binds source revision, version, schema digests, toolchain,
  every package/image/chart/notebook/model artifact digest and every admitted
  receipt;
- receipt kinds cover conversion/refinement, quality, task retention, runtime,
  memory/physical bytes, backend/browser conformance, ONNX, clean install,
  serving/deployment, model-card generation and second-machine reproduction;
- validators reject duplicate identities, unknown schema fields, stale source,
  missing parents, inconsistent hardware, copied run IDs, impossible byte
  ledgers and a claim whose required evidence kind is absent;
- `scripts/release-status` emits both JSON and a human table with `PASS`, `FAIL`,
  `MISSING`, `STRUCTURAL_ONLY` and `EXTERNAL_AUTH_REQUIRED`; it exits non-zero
  until every local-RC gate is green;
- README/book/model-card/compatibility/benchmark tables are generated from this
  registry. Hand-authored measured numbers or green capability cells fail a
  drift check.

Gate: adversarial fixtures prove that stale, mismatched, duplicate, structurally
substituted and partially copied evidence cannot produce a green status. A
fresh empty registry reports every missing gate without panicking.

## Slice 2 — five-minute and recipe tiers

Qualify the accessible model-zoo tier without conflating its two roles:

- pin SmolLM2-135M source revision, tokenizer, calibration/training/evaluation
  fixtures and license; produce pure-PTQ, one-step-QAT and native/ONNX/browser
  tutorial artifacts through the public APIs;
- run load → inspect → bounded calibration/PTQ → one QAT backward/optimizer step
  → checkpoint/resume → export/reload → generation from the exact clean-install
  wheel/npm archives in under five minutes on each declared reference target,
  excluding only the first immutable model download;
- pin SmolLM2-1.7B and freeze the scalable PTQ/refinement recipe, resume and
  physical-accounting behavior. It is the recipe tier, not an automatic claim
  of five-minute browser training;
- publish cards with exact coverage, preserved tensors, serialized/resident/
  peak bytes, calibration and evaluation identities, runtime, hardware and
  known limitations.

Gate: tutorial commands are non-interactive and headless-testable. All emitted
artifacts strict-reload; tied weights and tokenizer identity survive; claimed
targets have actual receipts rather than compile-only evidence.

## Slice 3 — native ternary reference tier

Re-admit BitNet b1.58 2B4T against the v1.1 candidate:

- bind the exact upstream model/GGUF/tokenizer identities and clarify the
  `I2_S` compatibility input versus Tritium's packed runtime representation;
- rerun native load, fidelity ladder, greedy parity, perplexity, long generation,
  CPU/CUDA memory and prefill/decode benchmarks from the candidate packages;
- preserve the published/built-on-box distinction for every baseline and use
  confidence intervals plus warmup/clock/power methodology for new numbers;
- record exact artifact and resident bytes. Logical ternary rate is descriptive
  only and cannot serve as the compression denominator;
- remove self-skipping semantics from the release harness: missing model,
  backend or receipt is `MISSING`, not `PASS`.

Gate: the complete card and benchmark tables regenerate from candidate receipts,
quality gates couple to performance, and a clean CPU/CUDA install reproduces
native generation with the declared artifact.

## Slice 4 — Qwen3.6-27B flagship tier

Consume, without relabeling, the admitted plan-0043 results:

- exact pinned Qwen3.6-27B language core plus bundled one-layer MTP drafter;
  vision/multimodal tensors remain identity-bound and explicitly deferred;
- separate Compact PTQ, NearLossless PTQ and NearLossless refined artifacts,
  work IDs, ancestry, costs and claims;
- exact matrix coverage, preserved tensors, nested plane maps, serialized and
  resident physical bytes, peak conversion/load/runtime memory and no-dense-shadow
  receipts;
- held-out perplexity, task retention, long generation, code and instruction
  following, MTP acceptance/speedup, prefill/decode across declared contexts and
  batches, export/reload and failure results with confidence intervals;
- matched-physical-byte RTN/AbsMean, GPTQ/AWQ-style and admitted SALT ablations;
  additive-ternary SOTA and global low-bit Pareto are separate verdicts;
- disclose device/GPU-hours/energy where measured and every failed hypothesis.

Gate: plan 0043's binding quality and runtime gates pass, refined NearLossless
is within 1% relative held-out perplexity, every in-scope language/MTP matrix is
covered, and the exact candidate runtime serves the strict artifact. “More than
10x” appears only if a measured whole-artifact or resident denominator proves it.

## Slice 5 — definitive documentation and examples

Status: **IN PROGRESS** — the root README and book introduction now identify the
exact `1.1.0-rc.0` candidate version without claiming local-RC readiness,
separate implemented capability from receipt-qualified support, route users to
ADR 0033/current plans and expose
flagship, package, browser, ONNX, multimodal, community and reproduction
blockers. The installed-artifact quick starts, final guide qualification,
migration, installed-artifact observability qualification and full
generated-claim drift gate remain open.
The book now also contains a comprehensive first draft of the "Definitive Guide
to Ternary Deep Learning" covering representation, physical accounting,
PTQ/QAT/refinement, estimator scale and threshold mechanics, additive
allocation, sparse and S34 representations, whole-model semantics, evaluation,
and failure diagnosis. Tutorial execution and final documentation-gate review
remain open before this documentation slice can close.
The public PyTorch facade now also has bounded, network-free ternary diagnostics
and injected TensorBoard, Weights & Biases and OpenTelemetry adapters. They
report deduplicated latent or hard trit distributions, scales, gradients,
per-plane applicable saturation, reconstruction error and code-plus-scale
bytes/rate; externally measured KL/runtime/resident-memory values remain
explicitly caller-supplied. Preflight element/path budgets prevent accidental
whole-model latent projection, external estimators require an explicit purity
opt-in, W&B steps cannot decrease, and OpenTelemetry is aggregate-by-default
with explicit series ceilings. Focused adapter tests use fakes rather than
accounts, and the book documents cadence and physical-accounting limits. A
source-tree adapter smoke passes
offline with TensorBoard 2.21.0, W&B
0.28.1 and OpenTelemetry 1.44.0. Candidate clean-wheel execution and an admitted
release receipt remain open; this developer smoke is not substituted for them.
The v1.0 → v1.1 migration guide now records the frozen/evolving Rust tiers, C
ABI continuity, Python distribution rename with stable imports, typed phase
transitions, copy-on-write artifact migration, deprecation/support windows and
an operator checklist. The SemVer gate's obsolete `v0.5.*` default is fixed to
select the latest reachable stable release (`v1.0.0`), guarded by a regression;
all seven frozen crates pass that actual comparison. Candidate-wheel migration
execution and the generated release API-diff receipt remain open.

Replace the current pre-alpha/v1.0-era narrative with tested v1.1 documentation:

- README: honest product position, supported paths, five-minute quick start,
  capability matrix, evidence-linked results and limitations;
- mdBook/API reference: architecture, additive PTQ, QAT, estimator catalog,
  refinement lineages, portable/browser training, formats, ONNX, serving,
  deployment, model zoo, benchmarks and troubleshooting;
- “Definitive Guide to Ternary Deep Learning”: ternary math, scales/thresholds,
  STE choices, additive residual planes, calibration/reconstruction/refinement,
  sparsity, physical accounting, evaluation and common failure modes;
- runnable tutorials for ordinary PyTorch QAT, Hugging Face PTQ, refinement,
  external estimator plugins, browser session, ONNX generation and production
  serving;
- v1.0 → v1.1 migration, compatibility, deprecation and support windows;
- TensorBoard, Weights & Biases and OpenTelemetry examples for ternary
  distribution/zero-rate/scales/gradients without making hosted accounts a test
  requirement.

Every command in the quick starts runs against an installed local-RC artifact.
Doctests/notebooks/examples are executed; dead links, stale API names, fabricated
numbers and source-checkout-only imports fail the documentation gate.

## Slice 6 — governance, security and support

Status: **IN PROGRESS** — root contribution, conduct, maintainer-led governance,
citation and version-support policies now exist; the security policy covers the
v1.1 support window, private response targets, supply-chain and deployment
boundaries. Evidence-oriented bug, estimator, backend, model/performance and
pull-request templates require reproducible identities and distinguish skipped,
structural and physical evidence. Channel activation, moderation, escalation and
durable-archive rules are explicit. Repository-link validation and independent
policy review remain open; no external community surface has been activated.

Add the community contracts required for a durable research platform:

- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `GOVERNANCE.md`, `CITATION.cff`,
  support/version policy and maintainer/reviewer expectations;
- update `SECURITY.md` and the threat model with supported versions, private
  disclosure, response targets, artifact/model supply-chain boundaries and the
  actual v1.1 server posture;
- issue/PR/discussion templates for bugs, backend ports, estimator proposals,
  model evidence, security-safe reproduction and performance claims;
- define RFC/ADR compatibility policy, deprecation periods, conformance-vector
  changes, benchmark admission, model-zoo promotion/demotion and conflict of
  interest handling;
- GitHub remains the authoritative durable archive. A moderated Discord and
  Discussions categories are activation actions, created only after explicit
  approval, with escalation and archival rules documented first.

Gate: repository links and contact routes resolve, templates request enough
identity/evidence to reproduce claims, governance cannot silently weaken frozen
release gates, and no unstaffed or nonexistent public channel is advertised.

## Slice 7 — independent second-machine reproduction

Give a fresh operator only the candidate manifest, immutable inputs and public
candidate instructions. On a second physical machine they must:

1. verify source and local-RC artifact digests;
2. install without the repository or a compiler;
3. reproduce the tutorial tier and BitNet native tier;
4. strict-load and exercise the Qwen flagship artifacts, rerunning the declared
   bounded validation subset rather than copying primary outputs;
5. run the applicable native backend/browser/ONNX/serving smoke;
6. regenerate model-card, compatibility and release-status tables.

The reproduction receipt records operator, machine/OS/hardware, commands,
wall time, immutable inputs, output digests, divergences and independently
generated run IDs. A second container on the primary host is a clean-environment
test, not second-machine reproduction.

Gate: all required reproduced values meet frozen tolerances and generated tables
match the candidate claims. Any divergence is investigated and recorded; the
threshold is not changed after seeing it.

## Slice 8 — local release-candidate sign-off

From a clean revision with no untracked release input:

- freeze the single version, changelog, migration, compatibility and API diff;
- run workspace tests/clippy/format/deny/semver, all target-specific admitted
  lanes, clean-install archives, ONNX, browser, model-zoo, serving/deployment and
  second-machine reproduction;
- generate checksums, SBOMs, provenance, local test signatures and the final
  evidence registry; rebuild and prove reproducibility where supported;
- require independent code/security/evidence review and resolve every verified
  finding; reviewer infrastructure failure is a blocker, not a pass;
- create the signed release commit and a **local candidate tag only** after every
  local-RC box is green. Do not push the tag.

The final local status distinguishes `LOCAL_RC_READY` from
`EXTERNAL_ACTIVATION_REQUIRED`. It never reports public release success before
registry/community activation and immutable post-publication smoke.

## Slice 9 — authorized activation and post-publication smoke

Only after Brian Lam explicitly authorizes the exact manifest and actions:

1. push the signed source tag/release revision;
2. publish the exact qualified crates, wheels, npm archive, OCI images/charts,
   model artifacts and documentation without rebuilding mutable payloads;
3. create the approved GitHub Discussions/Discord/community surfaces;
4. fetch every public artifact into fresh environments and rerun immutable
   package/model/ONNX/browser/serving smokes;
5. archive URLs, registry digests, transparency/signature evidence and smoke
   receipts in the release registry.

A failed public smoke never replaces or mutates `1.1.0`; it documents impact
and opens `1.1.1`. No credential, namespace or hosted resource action is inferred
from this work order.

## Verification cadence

```bash
./scripts/release-status --candidate release/v1.1/manifest.json
cargo fmt --check
cargo test --workspace --no-fail-fast
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo deny check
./scripts/check-semver.sh
mdbook build docs/book
PYTHONPATH=crates/tritium-py/python pytest -q crates/tritium-py/tests
npm --prefix packages/tritium-web run check
git diff --check
```

Current `release-status` covers artifact/SBOM/provenance admission only and
returns `CANDIDATE_EVIDENCE_VALID` without a registry. Its optional strict
registry mode now binds the exact candidate, hard-codes every ADR 0033 gate,
admits artifact-bound CUDA training, installed-wheel clean-install, complete
candidate-bound ABI3 matrix, offline crate-archive and clean offline npm-archive
schemas, emits human and JSON blocker reports, and refuses `LOCAL_RC_READY` for
empty or partial evidence. Npm admission landed in `cb34670` and `57a23ba`, but
the aggregate package gate still requires its clean-install, compatibility-matrix
and crate-archive receipts to coexist in the exact candidate registry. Remaining
receipt-kind validators and generated-claim drift checks must land before every
model-zoo, package, browser, serving, deployment, signing and second-machine
receipt can produce `LOCAL_RC_READY`.

Exact-image serving admission now distinguishes CPU runtime, CUDA runtime and
CPU/CUDA security scans, plus cluster deployment. Each runtime receipt must bind the candidate OCI archive,
the archive-verified image manifest digest, build lineage, strict model manifest,
startup receipt, physical machine/device identity and hardened live-container
checks. Security receipts bind same candidate archive plus scanner binary,
fresh database snapshots, offline commands and zero HIGH/CRITICAL vulnerability
or secret findings. One flavor cannot satisfy another gate, and none can satisfy
deployment. No empirical OCI runtime or security receipt exists locally yet.

Progress (2026-07-21): aggregate sign-off now uses a non-circular two-layer
contract. The registry report binds its own SHA-256 and can reach only
`LOCAL_RC_EVIDENCE_READY_UNSIGNED` (exit 2). A detached SSH signature then binds
the exact candidate manifest, evidence registry and report to an allowed signer
principal; only successful detached verification emits `LOCAL_RC_READY`. No key,
signature or tag is generated implicitly. The remaining receipt validators and
empirical gates still block creation of a real passing report.

Target-specific model, GPU, browser, package, container and cluster commands
are invoked by the candidate manifest and must emit admitted receipts. A skipped
test, absent target or zero-case report is not green.

## Stop conditions

- Stop before weakening any quality, coverage, physical-byte, portability,
  security, reproduction or publication gate.
- Stop before reporting structural/synthetic/tutorial evidence as flagship,
  hardware, complete-model or SOTA evidence.
- Stop before a paid Qwen run, model hosting, registry publication, tag push,
  public deployment or community creation without explicit approval.
- Stop if generated claims differ from receipts, the candidate revision is
  dirty, a required review has no verdict, or second-machine evidence is copied
  rather than regenerated.

## Done criterion

Every ADR 0033 local-RC box is green from one clean candidate revision; all
three zoo tiers and generated claims are admitted; docs/governance/security are
current; a second physical machine reproduces the required matrix; exact local
artifacts, manifests, SBOMs, provenance and test signatures agree; and a signed
local release commit/tag candidate exists. Public v1.1 is complete only after
explicitly authorized activation and successful immutable post-publication
smoke.

## Commit sequence

```text
feat(release): validate evidence-backed claims
test(zoo): admit tutorial and recipe tiers
test(zoo): requalify native ternary reference
test(zoo): admit Qwen3.6 flagship evidence
docs(v1.1): publish definitive ternary platform guide
docs(governance): establish contribution and support policy
test(release): reproduce candidate on second machine
build(release): seal local v1.1 candidate
chore(release): activate authorized immutable release
```
