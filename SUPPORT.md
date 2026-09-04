# Support and version policy

Tritium support is provided through the GitHub issue tracker. There is currently
no guaranteed commercial support or private general-help channel. Security
reports use the private route in [SECURITY.md](SECURITY.md).

## Supported versions

| Line | Status | Policy |
|---|---|---|
| `1.1.0-rc.N` | Release candidate | Development fixes only; interfaces may change before `1.1.0`. |
| Latest stable `1.x` minor | Supported | Correctness and security fixes; stable-core compatibility follows the release ADR. |
| Immediately previous stable `1.x` minor | Security maintenance | Critical/high security fixes for 90 days after the next stable minor. |
| Older releases and development snapshots | Unsupported | Upgrade before requesting a fix. |

An advertised backend, wheel, browser, model, or deployment target is supported
only when the release compatibility matrix marks its exact cell `qualified`.
`pending`, `unsupported`, compile-only, and locally modified configurations are
not support claims.

## Minimum supported Rust version

**Policy: latest stable minus two releases, reviewed every release, never below 1.89.**

`rust-version` in `Cargo.toml` is the published floor and the `msrv` CI lane compiles the
whole workspace — every feature on — at exactly that version, so it is a verified claim
rather than an aspiration.

Two things this deliberately is not:

- **It is not the toolchain we build with.** `rust-toolchain.toml` pins that separately and
  moves ahead of the floor. Conflating the two once held the build back a full year and
  blocked dependencies for no benefit.
- **It is not a promise of indefinite support for old compilers.** Tritium tracks a moving
  research frontier and its dependencies do too; a floor that never moves quietly becomes a
  veto over the ecosystem. Raising it is a normal, announced change, not an emergency.

The 1.89 lower bound is a hard technical limit, not a preference: AVX-512 intrinsics
stabilised there and the CPU kernels use them.

Raising the floor is called out in `CHANGELOG.md` for the release that does it.

## Publishing a release

Tags drive publication. Pushing `vX.Y.Z` runs `.github/workflows/release.yml`, which
calls `wheels.yml` for the build, then publishes the artifacts CI qualified — the
same ones, not a second build that resembles them.

| registry | package | credential |
|---|---|---|
| PyPI | `pytritium` | trusted publishing (OIDC) — no token stored |
| GitHub Releases | wheels, SBOMs, `SHA256SUMS`, provenance attestation | `GITHUB_TOKEN` |
| crates.io | 23 crates | trusted publishing (OIDC), or `CARGO_REGISTRY_TOKEN` — **dormant until one exists** |

A tag whose wheels do not carry its version fails the release before anything is
published, so a wheel built from the wrong tree cannot ship under a tag.

### One-time: register the PyPI trusted publisher

Publication fails closed until this exists — deliberately, so a missing publisher
never silently falls back to a long-lived token. On
<https://pypi.org/manage/project/pytritium/settings/publishing/>, add a GitHub
publisher:

| field | value |
|---|---|
| Owner | `Quitetall` |
| Repository | `tritium` |
| Workflow | `release.yml` |
| Environment | `pypi` |

Then create the `pypi` environment under repository *Settings → Environments*. Scope
the publisher to it; that environment is also where a manual approval gate goes if
releases should require a second pair of eyes.

### crates.io: trusted publishing or a token

The release job prefers **trusted publishing** — `rust-lang/crates-io-auth-action`
mints a token that lives only for that job, so nothing long-lived is stored. It is
configured **per crate**, and this workspace publishes 23 of them, so a stored
token remains supported rather than making OIDC an all-or-nothing migration.

*Preferred* — on each crate's Settings → Trusted Publishing on crates.io, add a
GitHub publisher with repository `Quitetall/tritium`, workflow `release.yml`,
environment `crates-io`.

*Or* — set a single `CARGO_REGISTRY_TOKEN` repository secret. Simpler to set up,
but it is a long-lived credential with publish rights to every crate; scope it to
the crates it needs and rotate it on a schedule.

If both exist, OIDC wins. If neither does, the job reports that it skipped and the
release still succeeds.

### Re-running a failed publish

A registry outage should not need a new tag. Run the workflow manually
(*Actions → release → Run workflow*) with the existing tag: uploads are
`skip-existing`, so re-running is safe and only fills what is missing.

## Asking for help

Include the Tritium version and source revision, OS/architecture, installation
method, backend/device, model and artifact identities, minimal reproduction,
full error, and the smallest safe logs or receipts needed to diagnose it. Remove
credentials and private model or dataset content.

Response is best effort. Reproducible security, data-corruption, and stable-core
regressions take priority over performance tuning and unsupported platforms.
