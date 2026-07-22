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

## Asking for help

Include the Tritium version and source revision, OS/architecture, installation
method, backend/device, model and artifact identities, minimal reproduction,
full error, and the smallest safe logs or receipts needed to diagnose it. Remove
credentials and private model or dataset content.

Response is best effort. Reproducible security, data-corruption, and stable-core
regressions take priority over performance tuning and unsupported platforms.
