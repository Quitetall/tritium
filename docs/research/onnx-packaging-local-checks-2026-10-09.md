# ONNX and packaging local checks

Date: 2026-10-09
Source baseline: `091b496fd5fced71b46059a4d3cc334c8e425871`
Contracts: ADR 0033 and plan 0051.
Evidence class: dirty-worktree developer checks, not release qualification.

## Passed checks

```sh
timeout 180 env PYTHONDONTWRITEBYTECODE=1 python -m unittest \
  scripts.tests.test_qualify_onnx_inference \
  scripts.tests.test_verify_onnx_inference_receipt \
  scripts.tests.test_generate_colab_notebook \
  scripts.tests.test_verify_wheel \
  scripts.tests.test_generate_wheel_sbom \
  scripts.tests.test_aggregate_wheel_smoke -q

timeout 120 env PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  -q -p no:cacheprovider scripts/tests/test_check_semver_baseline.py
```

The first suite passed 37 tests in 0.933 seconds. The second passed 11 tests
in 3.93 seconds. They validate producers, admission rules, generated notebook
contracts and API-checker failure/baseline policy. They do not execute the full
21-crate API comparison or produce real-model ONNX/Colab qualification.

An initial `unittest` invocation of the pytest-style API-gate file ran zero
tests and exited 5; it is not counted as a pass. The pytest invocation above
is the executed regression suite.

The generated Colab notebook check (`timeout 30 env PYTHONDONTWRITEBYTECODE=1
python scripts/generate-colab-notebook.py --check`) passed. The community
inventory checker also passed after the contributor prerequisite edit; neither
check executes Colab or constitutes independent governance review.

## API checker prerequisite

`timeout 30 scripts/check-semver.sh` exited 101 with
`error: no such command: semver-checks`. Its fail-closed behavior was preserved.
Contributor documentation now names this prerequisite explicitly.

Upstream's prebuilt GNU/Linux x86_64 `cargo-semver-checks` 0.51.0 was installed
under the existing reusable release-tool directory:
`/mnt/4tb/tritium-release-tools/cargo-semver-checks-0.51.0/`.
The 8,397,378-byte archive matched the GitHub release API SHA-256:
`cacadafaa5cd27ae6d05a74a103266917ebab4c5e43c898360e177d4fbf06a65`.
Both direct `--version` and `cargo semver-checks --version` with that directory
prepended to `PATH` returned `cargo-semver-checks 0.51.0`.
The first bounded download was incomplete and was not extracted or executed;
resumption completed and authentication preceded installation. Download and
extraction scratch was removed. No global PATH configuration was changed.

The full comparison remains unexecuted after installation. A future run must
use this tool directory in `PATH`, the shared SSD target and an explicit time
limit, and retain the actual per-crate findings. Version availability is not
API-compatibility evidence.

## Incomplete SBOM/assembly check

```sh
timeout 180 env PYTHONDONTWRITEBYTECODE=1 python -m unittest \
  scripts.tests.test_generate_deployment_sbom \
  scripts.tests.test_assemble_release_candidate \
  scripts.tests.test_generate_bundle_sbom \
  scripts.tests.test_generate_crate_sboms \
  scripts.tests.test_generate_npm_sbom -q
```

This exited 124 without a completed test result. The actual Python process was
observed in `D` state at `jbd2_log_wait_commit`; after timeout, the command and
worker PIDs were absent. It is not a test pass or a demonstrated software defect.
No gate, fsync requirement, tolerance or candidate-admission rule was weakened.
Its 44-KiB abandoned temporary fixture was identified by the exact assembler
test helper, fake digest executable, five-byte `wheel` and three-byte `npm`
payloads, and synthetic all-`a` source revision. The test PIDs were gone and no
visible descriptor/mapping/working-directory user remained. Only that newly
created fixture was removed; old scratch and campaign outputs were untouched.

The roadmap was corrected separately: OCI/chart SBOM generation and
assembly/admission already exist, but current physical candidate archives,
image security/runtime checks and deployment qualification remain open.
Planning commit `6ca420922580d91b22ba1611cc63c0b53acd52d6` was pushed and its
remote branch identity was verified.

## Binding remaining work

Actual source-free candidate ONNX language/MTP replay, the admitted production
MTP oracle, native/ORT cache and generation parity, current cross-platform
packages and Colab receipts remain required. Local unit fixtures, tool installs
and the presence of an SBOM generator do not satisfy those gates. No fitting,
paid compute, registry publication or public activation was performed.
