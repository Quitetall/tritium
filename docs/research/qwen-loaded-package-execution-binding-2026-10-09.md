# Qwen loaded-package execution binding

Date: 2026-10-09
Evidence class: host developer regression; not release qualification
Source baseline: `fef8a7c75c6853657a786dfe0f9fde5ca08cce4f`
Contract: ADR 0033 loaded-package execution-binding amendment, private research
commit `7a9ea94`.

## Reproduction

With the existing SSD build cache, `RUSTC_WRAPPER=` and two build jobs:

```sh
cargo test --locked -p tritium-nn --lib \
  model::qwen35_salt_v2::tests::complete_bundle_loads_and_executes_language_and_mtp_without_dense_matrix_shadows \
  -- --exact --nocapture
```

The managed reproduction and a second cached invocation both failed with exit
101. The second fixture execution took 0.02 seconds. The exact assertion symptom
was:

```text
changed weights must not emit a transcript naming the loaded package: Ok(Qwen35UntrustedRuntimeTranscript { ... })
```

The real bundle fixture successfully updated one language projection's scales,
then received a transcript naming the original loaded package. Removing that
successful update leaves the preceding parent execution checks passing. The
fixture needs a real package, language graph, successful update and transcript
boundary; a projection-only test cannot catch package mislabeling.

## Cause and change

Ranked hypotheses were a missing loaded-state guard, a non-persisting runner
identity update, and bypass of an existing guard. Source inspection shows the
update installs a new identity, while both transcript builders copied the
historical load receipt without a state comparison. The loaded model now keeps
the assembly identity and checks it before package-bound block/final execution,
including re-execution aliases. Historical receipts are not rewritten.

Regression coverage includes rejected-update preservation, immutable-child
package labeling, observer/input atomic rejection, block/state/final/re-execution
paths, fresh-cache ordinary inference and still-available candidate-shaped
output scopes. The scope seam names no loaded package and remains untrusted.

## Verification status

Rust formatting and whitespace checks passed. The exact post-fix regression
passed (0.02-second fixture execution). `cargo test --locked -p tritium-nn --lib`
passed all 167 tests with no failures or skips. `cargo test --locked -p tritium-nn
--test qwen35_text_runner` passed all 11 tests with no failures or skips. Scoped
Clippy passed with warnings denied in the managed continuation (3m 11s). The
SALT admitted-execution regression and normal commit-tree gate remain pending.
No physical
CUDA, production checkpoint, MTP oracle, numerical-quality, performance or
release-admission claim follows from this host regression.

The first scoped Clippy invocation (`cargo clippy --locked -p tritium-nn --lib
--tests -- -D warnings`) reached its 300-second limit with exit 124 and no
compiler error. This is incomplete, not a pass. Managed continuation unit
`tritium-qwen-package-binding-validate-20261009.service` reruns that command and
the SALT `admitted_qwen_execution_binds_campaign_packages_backend_tokens_and_outputs`
regression with longer explicit bounds. Its terminal results must be checked.
