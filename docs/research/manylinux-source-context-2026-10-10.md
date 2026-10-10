# Manylinux builders: original requested-source admission

Code source: `c47013b4cd158e9d6451693019566136dec1fcf5`.
Scope: packaging/source admission, not empirical wheel or model qualification.

## Reproduction

The actual CPU and CUDA shell builders still read inherited Git context after
the Python qualification consumers were repaired. Real disposable repositories
and linked worktrees reproduced twelve failing builder subcases twice:

- foreign local selectors chose the foreign source mount and source identity;
- modified, staged and untracked requested source was hidden by the foreign
  checkout;
- replacement refs made different bytes look clean under the original HEAD;
- a failed `git status` with empty stdout was treated as a clean tree.

Each initial run also failed one test for the then-missing shared-helper CLI
(13 failures total, five test methods). Config-injection cases already rejected
or selected the correct source in that initial fixture; they are regression
coverage, not additional reproduced failures. Actual Git and shell builders run
in these tests. Docker, compiler/toolchain discovery and post-build wheel
verification are stubbed; no real build or candidate evidence is generated.
Temporary fixture repositories and outputs were removed automatically.

## Repair

Both builders now call the existing `_qualification_git.py` through its
bounded binary-preserving shell bridge for root, status and HEAD. The helper
isolates its own Git children, removes inherited local selectors/config
injection, disables replacement-object views and preserves the caller's
environment. The bridge preserves stdout/stderr bytes and Git failure status;
missing commands/checkouts reject. The original Python import API is unchanged.

The status result is assigned separately before the shell condition, so a
failed probe cannot masquerade as empty clean status. Dirty source still rejects
before creating the requested output directory or reaching Docker. Both
builders retain their immutable `--print-contract`, toolchain/image/linker,
platform tag, CUDA requirements and bounded compiler-job policy. The wheel
workflow includes the new regression file in its path filter.

## Verification

The focused builder/source/resource suite initially passed nine test methods;
an additional helper CLI negative test then joined the final combined suite.
The combined suite passed 147 tests in 27.100 seconds before commit and again
on the clean committed code tree in **25.416 seconds**:

```sh
python3 -B -m unittest \
  scripts.tests.test_manylinux_source_context \
  scripts.tests.test_manylinux_build_resources \
  scripts.tests.test_build_cpu_manylinux_wheel \
  scripts.tests.test_build_cuda_manylinux_wheel \
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

`actionlint .github/workflows/wheels.yml` and `git diff --check` pass. Normal
commit hooks passed formatting, changed Python compilation and shell syntax;
normal push gates passed formatting and workspace/all-target Clippy with
warnings denied. No hook bypass was used. Synthetic preflight and Stage-7 PASS
messages printed by unit fixtures are not actual campaign qualification.

## Custody and remaining scope

Durable source/log/result archive:
`/home/brianklam/Projects/Tritium/archive/verification/manylinux-source-context-c47013b4-20261010`.
Its separate stdlib verifier checks ten snapshots against externally supplied
original Git objects without importing the repaired helper. Archive checksums
prove retained bytes only, not release readiness.

The previously retained portable CUDA wheel and CPU/CUDA dispatcher receipts
remain bound to `98489c95560629f4bd43016420d1738f63aa91e1`; this later shell
repair does not rewrite or re-admit them against a different revision. Final
candidate packages and exact-source empirical gates still require reruns.
This is not an all-repository provenance, malicious-toolchain or immutable
source-mount claim. No large fitting/capture, paid compute, model weights,
signing, publication, deployment or human activation was performed.

Remaining v1.1 fronts include measured observability admission, full Stage-7
measurement/auxiliary producers, representative PTQ/refinement/baselines,
Qwen language/MTP quality/bytes/runtime/reproduction, physical backend/browser/
multi-device performance, serving fault/residency/concurrency and production
security/deployment, final packaging/zoo/community, independent second-machine
and operator clearance, and explicit human release authorization. Accepted
ADR 0044's unified schema/format/engine migration also remains a substantial
architecture workstream; the shared helper repair is not its completion.
