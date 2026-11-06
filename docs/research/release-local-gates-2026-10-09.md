# Local release-support checks

Date: 2026-10-09
Source baseline: `091b496fd5fced71b46059a4d3cc334c8e425871`
Evidence class: dirty-worktree developer checks, not release qualification
Contracts: ADR 0033; plans 0044 and 0052.

These checks were performed while separately owned Qwen and training/docs work
was present. They do not establish a clean release source, signed candidate,
physical deployment, security scan, independent review or public activation.

## Passed checks

```sh
timeout 60 python scripts/check-community-contract.py --json
timeout 120 env PYTHONDONTWRITEBYTECODE=1 python -m unittest \
  scripts.tests.test_deployment_contract \
  scripts.tests.test_verify_oci_archive -q
```

The community checker returned `result: pass`, covering 16 governance files,
14 local links, 20 public documents, three repository routes and zero unstaffed
advertised channels. This is structural inventory validation, not independent
policy review or evidence of an activated public community.

All 30 deployment/OCI tests passed (35.247 seconds). Their coverage includes
closed Helm schemas, digest/artifact/secret/security controls, bounded optional
surfaces and OCI archive/SBOM/provenance admission. Archive attestations in the
tests are synthetic fixtures; this is not an empirical image/security receipt.
Temporary fixture archives/directories were cleaned by the tests.

The real-Git hook suite was also expanded and executed:

```sh
timeout 180 env PYTHONDONTWRITEBYTECODE=1 python -m unittest \
  scripts.tests.test_precommit_web_git_isolation -v
```

Both cases passed (48.365 seconds): the outer `commit --only` temporary index,
and explicit `GIT_DIR`/`GIT_WORK_TREE` bindings. The latter also passed before any
further hook change (3.219 seconds), so no additional production fix was made.
Both verify exact selected-tree contents, preservation of unrelated staged
content, nested repository independence and complete scratch cleanup. Their
Cargo/npm stand-ins qualify only the Git isolation boundary.

## Incomplete check

```sh
timeout 60 scripts/check-deployment-manifests
```

This command exited 124 at the time limit without a Helm result. The pinned,
offline Docker invocation was inspected afterward: no remaining client PID or
container for the Helm image was present. No daemon or unrelated process was
stopped. The lint/render gate remains unverified; neither the passing Python
contracts nor an empty container list replaces its result.

Real CPU/CUDA OCI runtime/security evidence, Kubernetes failure-matrix receipts,
Prometheus/KEDA/Knative behavior, physical-device identity, and independent
release review remain required. No registry publication, hosted cluster, paid
compute or public endpoint was created.
