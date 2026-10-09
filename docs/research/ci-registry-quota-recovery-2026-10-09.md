# CI registry quota recovery — 2026-10-09

## Observed failures

At source `66f106f415c5cf3c8bda0d4930d4494594fe8979`, two hosted
qualification prerequisites failed before any Tritium check executed:

- Wheels run `37990917364`, job `114025448679`: three `docker pull
  python:3.13-slim` attempts returned `toomanyrequests: You have reached your
  unauthenticated pull rate limit` during container initialization. Every
  source-free tutorial/lifecycle/observability step was skipped.
- CI run `37990917412`, job `114024533201`: the pinned cargo-deny Docker action
  failed three builds while resolving its pinned Docker Hub Rust base image,
  with HTTP 429 and the same anonymous pull-quota message. Neither checkout nor
  dependency policy checks ran.

The logs were retrieved directly from the completed jobs' GitHub API log
endpoints. These are infrastructure failures, not successful or failed model
qualification results. Neither gate is waived.

## Correction and identity checks

The source-free lane now pulls the official Python image from ECR Public by
its exact Linux/amd64 manifest digest:

```text
public.ecr.aws/docker/library/python@sha256:8fb4cfa1a2616d7b8e0c2175cc6ad68f5729c34ea8488c0b360d2934b7be9024
```

Authenticated anonymous Registry API reads of both registries' `3.13-slim`
tags returned identical index bytes (SHA-256
`70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f`)
and the same Linux/amd64 manifest digest above. No registry token was printed
or persisted. Pinning the platform manifest prevents tag drift; no tag-only
or Docker Hub fallback was added. The lane still rejects source checkouts and
all eight named compilers, and still executes each installed-wheel receipt
producer and verifier.

The supply-chain lane now uses the repository's existing SHA-pinned
`taiki-e/install-action` composite action to install `cargo-deny@0.20.2`, the
same version as the old Docker action. Checksums are explicitly enabled and
fallback installation is disabled. At the pinned action revision, its embedded
Linux/musl archive checksum is
`9f12ed4c49936e09b48bf862b595cde2fe64fcbd9d74dfacac6131ca824c8d5f`.
The lane installs the pinned project Rust toolchain, then executes:

```sh
cargo deny --locked --all-features check licenses bans sources advisories
```

All four policy classes and all-feature coverage are retained; `--locked` now
also makes unexpected lockfile changes fatal. No deny configuration, advisory
exception, release contract, or qualification tolerance was changed.

Primary implementation references:

- [AWS: Docker Official Images on ECR Public](https://aws.amazon.com/blogs/containers/docker-official-images-now-available-on-amazon-elastic-container-registry-public/)
- [Pinned install-action documentation](https://github.com/taiki-e/install-action/blob/742a3317eac7bd62f91cd888b4eead5e784ba833/README.md)
- [Pinned cargo-deny archive manifest](https://github.com/taiki-e/install-action/blob/742a3317eac7bd62f91cd888b4eead5e784ba833/manifests/cargo-deny.json)
- [Previous action defaults](https://github.com/EmbarkStudios/cargo-deny-action/blob/3c6349835b2b7b196a839186cb8b78e02f7b5f25/action.yml)

## Local validation and limits

Both workflow regression tests were run red before their owning workflow was
changed. After the correction:

```sh
python -B -m unittest scripts.tests.test_installed_qat_tutorial \
  scripts.tests.test_verify_workflow_source \
  scripts.tests.test_observability_qualification -q
actionlint .github/workflows/ci.yml .github/workflows/wheels.yml
cargo deny --locked --all-features check licenses bans sources advisories
git diff --check
```

Results: nine tests passed; actionlint and diff checks passed; cargo-deny 0.20.2
exited zero with `advisories ok, bans ok, licenses ok, sources ok`. Existing
allowed duplicate-dependency warnings remain warnings; they were not suppressed
or reclassified.

A bounded local mirror pull completed with the pinned digest after an initial
90-second observation timed out. A second bounded pull completed; the first
timeout was not treated as corruption or proof of a registry failure. A real
container using no network, a read-only root, dropped capabilities,
`no-new-privileges`, 128 MiB memory and one CPU reported Python 3.13.16 and
passed the absence-of-source/compilers checks. Its local image size was
118,299,860 bytes. This is environment validation, not an installed tutorial
or model-quality receipt. The task-created image is removed after validation.

The installed-wheel job and pinned SmolLM2 CPU tutorial at the previous source
completed successfully in hosted wheels run `37990917364`; the source-free
lane did not. Hosted execution of these workflow corrections remains pending
until the new commit runs. Physical CUDA and the full release remain unqualified.
