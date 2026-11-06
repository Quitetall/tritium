# Release and Stage-7 source-context repair

Code source: `595bc170d28fa9ba7e893ea8aa2da6b14825e6b3`.
Scope: source admission, not empirical release qualification.

## Counterexamples

The original release/workflow regression command failed twice with six failed
assertions and two errors. It demonstrated foreign HEAD/index/root selection,
incorrect repository-absence admission, foreign release-ref resolution and a
replacement-ref dispatcher/API view. The minimized actual workflow CLI rejected
the correct requested revision with inherited foreign selectors (exit 1), but
accepted it after removing only Git selectors (exit 0).

Four additional real-repository Stage-7/capture/native tests then reproduced
three failures and one error twice: valid linked-root rejection, dirty capture
source passing preflight, dirty native source reaching the hardware boundary,
and a replacement-ref dependency view passing original-source admission.

The first extended run also contained fixture setup errors. It inadvertently
invoked Compute Sanitizer's version probe and Cargo, which failed immediately
because the disposable repository has no Cargo.toml. The corrected regression
blocks all non-Git subprocesses while retaining real Git execution. Both logs
are retained and explicitly distinguished; no model/native qualification ran.

## Repair

Twelve additional consumers now use the bounded, child-only Git-context helper:
workflow source, release status, release tag resolution, API diff, CUDA receipt
source lookup, browser native reference, second-machine reproduction preflight,
Stage-7 execution, rebind, qualifier, native matrix and Qwen capture preflight.
This complements the twelve qualification frontends repaired in `d9b59414`.

The helper exposes raw text/binary results where required. API source reads
retain trailing bytes/newlines; Stage-7 compares exact binary committed bytes.
Original Git objects remain authoritative, not replacement refs. Parent process
environment and unrelated working trees are not modified. Every helper Git
subprocess has a 30-second bound and malformed discovery/timeouts fail closed.

Stage-7 preserves executing-qualifier ownership checks and additionally binds
its executing Git helper's path and bytes to the original source HEAD. Clean
tree, canonical source revision/root, candidate bindings, numerical thresholds,
receipt schemas, release-ref ancestry and publication rules are unchanged.

Second-machine preflight no longer treats every nonzero Git result as proof of
repository absence. It requires Git's specific C-locale not-a-repository error;
ownership, invalid-command and miscellaneous errors reject. Compiler checks
remain unchanged. Tests that exclude compiler availability are Git-only tests,
not second-machine qualification evidence.

## Verification and custody

Normal commit hooks passed formatting and staged Python compilation. On the
clean committed code tree, the actual Stage-7 source-only probe returned the
exact code revision above. The original minimized workflow probe now accepts
the requested revision with and without foreign Git selectors (both exit 0).

Focused pre-commit runs passed 76 Stage-7/source/capture tests and 59 existing
release-contract tests; final source-only runs passed 29 tests. The combined
committed-tree suite passed **136 tests in 116.852s**:

```sh
python3 -B -m unittest \
  scripts.tests.test_release_source_git_context \
  scripts.tests.test_qualification_git_context \
  scripts.tests.test_capture_qwen36_from_pack \
  scripts.tests.test_qualify_stage7_recipe_freeze \
  scripts.tests.test_verify_workflow_source \
  scripts.tests.test_qualify_release_reproduction \
  scripts.tests.test_verify_release_reproduction \
  scripts.tests.test_verify_torch_dispatch_cuda_receipt \
  scripts.tests.test_generate_api_diff \
  scripts.tests.test_resolve_release_ref \
  scripts.tests.test_release_status \
  scripts.tests.test_produce_browser_native_reference \
  scripts.tests.test_run_stage7_recipe_freeze \
  scripts.tests.test_rebind_stage7_campaign \
  scripts.tests.test_verify_stage7_qualification_receipt -q
```

Normal push gates passed `cargo fmt --all --check` and
`cargo clippy --locked --workspace --all-targets -- -D warnings`; remote readback
confirmed the code revision. Logs are retained in the archive below.

A follow-up path-filter regression failed three subchecks twice: changing the
shared Git helper or either real-context regression module alone did not trigger
the wheel PR smoke. The wheel workflow now includes all three dependency paths;
the general CI script lane already discovers both regression modules. This
follow-up is workflow coverage, not additional numerical qualification.
The workflow/source-context follow-up suite passes 22 tests; `actionlint
.github/workflows/wheels.yml` and `git diff --check` pass.
Synthetic receipt PASS messages printed by unit fixtures are not real receipts.

Durable source/log/probe archive:
`/home/brianklam/Projects/Tritium/archive/verification/release-source-context-595bc170-20261010`.
Its separate stdlib verifier compares 16 source snapshots against externally
supplied original Git objects, without importing the repaired helper. Archive
hashes establish custody only; snapshots are not a standalone installation.

CI, wheels, CUDA capstone and CodeQL for previous source `264a7a67` completed
successfully. Those hosted results do not qualify this later repair source.

## Remaining obligations

Experimental EAT-O Git metadata, shell gates/build stamps and other source
consumers remain separate audit scope. This is not an all-repository provenance
or hostile-Python sandbox claim. No paid compute, large capture/fitting,
signing, publication, deployment or human release activation was performed.

ADR 0053 remains proposed/human-adoption pending; fixed-counter observability
admission is still open. Final estimator clearance, representative PTQ/refinement
and baselines, physical multi-device/backend/browser/performance, Qwen language
and MTP quality/bytes/runtime/reproduction, serving fault/residency/concurrency,
OCI/security/production deployment, final packaging/zoo/community, independent
second-machine/operator clearance and human release authorization remain open.
