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

## Asking for help

Include the Tritium version and source revision, OS/architecture, installation
method, backend/device, model and artifact identities, minimal reproduction,
full error, and the smallest safe logs or receipts needed to diagnose it. Remove
credentials and private model or dataset content.

Response is best effort. Reproducible security, data-corruption, and stable-core
regressions take priority over performance tuning and unsupported platforms.
